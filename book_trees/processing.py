import ebooklib
import re
import spacy
import json
import requests
from difflib import SequenceMatcher
from ebooklib import epub
from bs4 import BeautifulSoup
from django.db import transaction
from .models import EpubFile, Chapter, Character, Relationship
from typing import List, Dict, Any

# load NLP
nlp = spacy.load("en_core_web_sm")

def extract_chapters_from_epub(epub_path):
    """Extract chapters and text from an EPUB file."""
    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text("\n", strip=True)

            if "chapter" in item.get_name().lower():
                match = re.search(r'chapter[_\s-]*(\d+)', item.get_name().lower())
                chapter_number = int(match.group(1)) if match else None

                chapters.append({
                    'chapter_number': chapter_number,
                    'title': item.get_name(),
                    'content': text
                })

    return chapters

def process_epub_file(epub_id):
    epub = None
    try:
        epub = EpubFile.objects.get(id=epub_id)
        epub.status = 'pr'
        epub.save()

        file_path = epub.file.path

        chapters_data = extract_chapters_from_epub(file_path)

        for chapter_data in chapters_data:
            Chapter.objects.create(
                epub=epub,
                title=chapter_data['title'],
                content=chapter_data['content'],
                chapter_number=chapter_data['chapter_number']

            )

        epub.status = 'c'
        epub.processed = True
        epub.save()

        return True



    except Exception as e:
        if epub:
            epub.status = 'f'
            epub.error_message = str(e)
            epub.save()
        raise


def extract_characters_with_chunks(epub_id, context_sentences=2):
    """
    Extracts character names from EPUB using spaCy NER and creates chunks with context
    Avoids overlapping chunks by advancing past the context window

    Args:
        epub_id: ID of the EpubFile to process
        context_sentences: Number of sentences before/after to include in chunk

    Returns:
        Number of unique characters found
    """
    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    character_counts = {}
    first_appearance = {}

    for chapter in chapters:
        doc = nlp(chapter.content)

        sentences = list(doc.sents)
        annotated_chunks = []

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            entities_in_sentence = []

            for ent in sent.ents:
                if ent.label_ == 'PERSON':
                    name = ent.text.strip()
                    name = name.rstrip("'s")

                    # skip single letters, common false positive
                    if len(name) <= 1:
                        continue

                    # Skip if the entity contains verbs (using spaCy's POS tagging)
                    # This filters out phrases like "Tara hesitates" or "John said"
                    entity_doc = nlp(ent.text)
                    has_verb = any(token.pos_ == 'VERB' for token in entity_doc)
                    if has_verb:
                        continue

                    # Skip if mostly lowercase (likely not a proper name)
                    # e.g., "the king" vs "King Arthur"
                    words = name.split()
                    capitalized_words = sum(1 for w in words if w and w[0].isupper())
                    if len(words) > 1 and capitalized_words / len(words) < 0.5:
                        continue

                    entities_in_sentence.append(name)

                    character_counts[name] = character_counts.get(name, 0) + 1

                    if name not in first_appearance:
                        first_appearance[name] = chapter.chapter_number

            if entities_in_sentence:
                start_idx = max(0, i - context_sentences)
                end_idx = min(len(sentences), i + context_sentences + 1)

                context_sents = sentences[start_idx:end_idx]
                context_text = ' '.join([s.text.strip() for s in context_sents])

                all_characters_in_context = set(entities_in_sentence)
                for context_sent in context_sents:
                    for ent in context_sent.ents:
                        if ent.label_ == 'PERSON':
                            char_name = ent.text.strip().rstrip("'s")
                            if len(char_name) > 1:
                                all_characters_in_context.add(char_name)

                annotated_chunks.append({
                    'center_sentence': sent.text.strip(),
                    'context': context_text,
                    'characters_in_sentence': entities_in_sentence,
                    'characters_in_context': list(all_characters_in_context),
                    'sentence_index': i
                })

                i = end_idx
            else:
                i += 1

        chapter.annotated_sentences = annotated_chunks
        chapter.save()

    for name, count in character_counts.items():
        Character.objects.update_or_create(
            epub=epub,
            name=name,
            defaults={
                'mention_count': count,
                'first_appearance_chapter': first_appearance[name]
            }
        )

    return len(character_counts)


def analyze_chunk_with_llm(chunk_data: Dict[str, Any], api_key: str = None) -> List[Dict]:
    """
    Sends a curated chunk with multiple characters to Gemini 2.5 Flash-Lite for relationship analysis.

    Args:
        chunk_data: Dictionary containing:
            - context: str - The text chunk with context
            - characters_in_context: List[str] - Character names mentioned
            - chapter_number: int - Chapter number for reference
        api_key: Google API Key

    Returns:
        List of relationship dictionaries extracted from the LLM response
    """

    # Only analyze chunks with 2+ characters
    if len(chunk_data.get('characters_in_context', [])) < 2:
        return []

    # Prepare the prompt
    prompt = f"""Analyze the following text excerpt and identify relationships between characters.

Text excerpt:
{chunk_data['context']}

Characters mentioned: {', '.join(chunk_data['characters_in_context'])}

Instructions:
1. Identify EXPLICIT relationships only (stated or strongly implied in the text)
2. Return relationships as JSON in this exact format:
{{
    "relationships": [
        {{
            "character_1": "Name1",
            "character_2": "Name2",
            "relationship_type": "one of: family, romantic, friend, ally, enemy, mentor, master_servant, other",
            "specific_type": "brother/sister/father/mother/friend/rival/etc",
            "confidence": 0.0-1.0,
            "evidence": "exact quote or paraphrase from text"
        }}
    ]
}}

3. Use exact character names as they appear in the "Characters mentioned" list
4. Be specific: prefer "brother" over just "family"
5. Return ONLY valid JSON, no other text
"""

    try:
        # Call Gemini 2.5 Flash-Lite API
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}",
            headers={
                "Content-Type": "application/json",
            },
            json={
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json"  # Forces JSON output
                }
            },
            timeout=30
        )

        response.raise_for_status()

        # Extract the response
        result = response.json()

        # Gemini's response structure
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            print(f"Unexpected Gemini response format: {result}")
            return []

        # Parse the JSON response (should already be clean JSON due to responseMimeType)
        parsed_data = json.loads(content)
        relationships = parsed_data.get('relationships', [])

        # Add chapter reference to each relationship
        for rel in relationships:
            rel['chapter_number'] = chunk_data.get('chapter_number')

        return relationships

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response content: {content}")
        return []
    except Exception as e:
        print(f"Unexpected error in LLM analysis: {e}")
        return []

def merge_characters(primary_character, characters_to_merge):

def validate_and_deduplicate_characters_with_llm(epub_id: int, api_key: str = None) -> Dict[str, Any]:
    """
    Use LLM to validate character names and identify duplicates/variations.

    This replaces the old find_character_name_variations, merge_characters, and deduplicate_characters functions.

    Args:
        epub_id: ID of the EpubFile to process
        api_key: Google API Key for Gemini

    Returns:
        Dictionary with validation stats including:
            - original_count: Number of character names before validation
            - invalid_names: List of names that were removed (not real characters)
            - merged_groups: List of groups that were merged
            - final_count: Number of unique characters after processing
            - reduction: Number of duplicates removed
    """
    epub = EpubFile.objects.get(id=epub_id)
    original_characters = list(Character.objects.filter(epub=epub_id).order_by('-mention_count'))
    original_count = len(original_characters)

    if original_count == 0:
        return {
            'original_count': 0,
            'invalid_names': [],
            'merged_groups': [],
            'final_count': 0,
            'reduction': 0
        }

    # Extract just the names and their mention counts for the LLM
    character_data = [
        {
            'name': char.name,
            'mention_count': char.mention_count,
            'first_appearance': char.first_appearance_chapter
        }
        for char in original_characters
    ]

    # Prepare the prompt for the LLM
    prompt = f"""You are analyzing character names extracted from a book using NER (Named Entity Recognition). 
Some extracted names may not be real characters (e.g., places, titles, errors), and some may be variations of the same character.

Here is the list of extracted names with their mention counts:
{json.dumps(character_data, indent=2)}

Your task:
1. Identify which names are NOT real characters (e.g., place names, titles, common nouns, errors)
2. Group together names that refer to the same character (e.g., "Harry", "Harry Potter", "Potter")
3. For each group, select the CANONICAL name (usually the most complete/formal version)

Return JSON in this EXACT format:
{{
    "invalid_names": ["Name1", "Name2"],
    "character_groups": [
        {{
            "canonical_name": "Harry Potter",
            "variations": ["Harry", "Potter", "Harry James Potter"],
            "reasoning": "All refer to the protagonist"
        }},
        {{
            "canonical_name": "Hermione Granger",
            "variations": ["Hermione", "Granger"],
            "reasoning": "All refer to the same character"
        }}
    ]
}}

Guidelines:
- Only mark names as invalid if you're confident they're not characters
- Only group names if they CLEARLY refer to the same person
- The canonical_name should be one of the variations in the list
- Don't include the canonical_name in the variations list
- When in doubt, keep names separate
- Return ONLY valid JSON, no other text
"""

    try:
        # Call Gemini API
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}",
            headers={
                "Content-Type": "application/json",
            },
            json={
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.2,  # Lower temperature for more consistent analysis
                    "maxOutputTokens": 4096,
                    "responseMimeType": "application/json"
                }
            },
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        # Extract the response
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            print(f"Unexpected Gemini response format: {result}")
            return {
                'original_count': original_count,
                'invalid_names': [],
                'merged_groups': [],
                'final_count': original_count,
                'reduction': 0,
                'error': 'Unexpected API response format'
            }

        # Parse the JSON response
        validation_data = json.loads(content)
        invalid_names = validation_data.get('invalid_names', [])
        character_groups = validation_data.get('character_groups', [])

        print(f"\n{'=' * 60}")
        print("LLM Character Validation Results")
        print(f"{'=' * 60}\n")

        # Step 1: Remove invalid names
        removed_count = 0
        if invalid_names:
            print(f"Removing {len(invalid_names)} invalid names:")
            for name in invalid_names:
                print(f"  ✗ {name}")
                Character.objects.filter(epub=epub_id, name=name).delete()
                removed_count += 1
            print()

        # Step 2: Merge character groups
        merged_count = 0
        merged_groups_info = []

        if character_groups:
            print(f"Merging {len(character_groups)} character groups:\n")

            with transaction.atomic():
                for group in character_groups:
                    canonical = group['canonical_name']
                    variations = group['variations']
                    reasoning = group.get('reasoning', '')

                    # Find the canonical character
                    try:
                        primary_char = Character.objects.get(epub=epub_id, name=canonical)
                    except Character.DoesNotExist:
                        print(f"  ⚠ Warning: Canonical name '{canonical}' not found, skipping group")
                        continue

                    # Find all variation characters
                    variation_chars = []
                    for var_name in variations:
                        try:
                            var_char = Character.objects.get(epub=epub_id, name=var_name)
                            if var_char.id != primary_char.id:
                                variation_chars.append(var_char)
                        except Character.DoesNotExist:
                            continue

                    if not variation_chars:
                        continue

                    # Print merge info
                    all_names = [canonical] + variations
                    print(f"  Merging: {', '.join(all_names)}")
                    print(f"    → {canonical}")
                    if reasoning:
                        print(f"    Reason: {reasoning}")
                    print()

                    # Merge the characters
                    merge_characters_internal(primary_char, variation_chars)
                    merged_count += 1

                    merged_groups_info.append({
                        'canonical': canonical,
                        'variations': variations,
                        'reasoning': reasoning
                    })

        # Final stats
        final_count = Character.objects.filter(epub=epub_id).count()
        total_reduction = original_count - final_count

        stats = {
            'original_count': original_count,
            'invalid_names': invalid_names,
            'invalid_count': removed_count,
            'merged_groups': merged_groups_info,
            'merged_count': merged_count,
            'final_count': final_count,
            'reduction': total_reduction
        }

        print(f"{'=' * 60}")
        print("Validation Complete!")
        print(f"{'=' * 60}")
        print(f"Original characters: {original_count}")
        print(f"Invalid names removed: {removed_count}")
        print(f"Character groups merged: {merged_count}")
        print(f"Final character count: {final_count}")
        print(f"Total reduction: {total_reduction} ({(total_reduction / original_count * 100):.1f}%)")
        print(f"{'=' * 60}\n")

        return stats

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return {
            'original_count': original_count,
            'invalid_names': [],
            'merged_groups': [],
            'final_count': original_count,
            'reduction': 0,
            'error': str(e)
        }
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response content: {content}")
        return {
            'original_count': original_count,
            'invalid_names': [],
            'merged_groups': [],
            'final_count': original_count,
            'reduction': 0,
            'error': f'JSON parse error: {str(e)}'
        }
    except Exception as e:
        print(f"Unexpected error in character validation: {e}")
        return {
            'original_count': original_count,
            'invalid_names': [],
            'merged_groups': [],
            'final_count': original_count,
            'reduction': 0,
            'error': str(e)
        }


def merge_characters_internal(primary_character, characters_to_merge):
    """
    Internal helper to merge multiple character records into one primary character.
    Updates all relationships and annotations.

    Args:
        primary_character: The Character object to keep
        characters_to_merge: List of Character objects to merge into primary
    """
    with transaction.atomic():
        # Combine mention counts
        total_mentions = primary_character.mention_count
        for char in characters_to_merge:
            total_mentions += char.mention_count

        primary_character.mention_count = total_mentions

        # Store aliases (if your model supports it)
        if hasattr(primary_character, 'aliases'):
            if not primary_character.aliases:
                primary_character.aliases = []

            for char in characters_to_merge:
                if char.name not in primary_character.aliases:
                    primary_character.aliases.append(char.name)

        primary_character.save()

        # Update all relationships
        for char in characters_to_merge:
            # Update relationships where this character is character_1
            Relationship.objects.filter(character_1=char).update(character_1=primary_character)

            # Update relationships where this character is character_2
            Relationship.objects.filter(character_2=char).update(character_2=primary_character)

            # Delete the old character
            char.delete()

        # Remove duplicate relationships (same char_1, char_2, type)
        relationships = Relationship.objects.filter(
            character_1=primary_character
        ) | Relationship.objects.filter(
            character_2=primary_character
        )

        seen = set()
        for rel in relationships:
            key = (
                min(rel.character_1.id, rel.character_2.id),
                max(rel.character_1.id, rel.character_2.id),
                rel.relationship_type
            )

            if key in seen:
                # Find and merge duplicate
                duplicate = Relationship.objects.filter(
                    character_1__in=[rel.character_1, rel.character_2],
                    character_2__in=[rel.character_1, rel.character_2],
                    relationship_type=rel.relationship_type
                ).exclude(id=rel.id).first()

                if duplicate:
                    # Merge evidence before deleting
                    duplicate.evidence.extend(rel.evidence)
                    duplicate.save()
                    rel.delete()
            else:
                seen.add(key)


def extract_relationships_with_llm(epub_id: int, api_key: str = None, batch_size: int = 10):
    """
    Extract relationships from an EPUB using LLM analysis on chunked text.

    Args:
        epub_id: ID of the EpubFile to process
        api_key: API key for the LLM service
        batch_size: Number of chunks to process at once (for rate limiting)

    Returns:
        Number of relationships found
    """

    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    relationships_found = 0
    all_relationships = []

    for chapter in chapters:
        chunks = chapter.annotated_sentences or []

        for chunk in chunks:
            # process chunks with 2+ characters
            if len(chunk.get('characters_in_context', [])) < 2:
                continue

            chunk_with_chapter = {
                **chunk,
                'chapter_number': chapter.chapter_number
            }

            relationships = analyze_chunk_with_llm(chunk_with_chapter, api_key)
            all_relationships.extend(relationships)

    with transaction.atomic():
        for rel_data in all_relationships:
            try:
                char1 = Character.objects.get(epub=epub, name=rel_data['character_1'])
                char2 = Character.objects.get(epub=epub, name=rel_data['character_2'])

                rel, created = Relationship.objects.get_or_create(
                    epub=epub,
                    character_1=char1,
                    character_2=char2,
                    relationship_type=rel_data['relationship_type'],
                    defaults={
                        'confidence': rel_data.get('confidence', 0.7),
                        'evidence': []
                    }
                )

                evidence_entry = {
                    'chapter': rel_data.get('chapter_number'),
                    'specific_type': rel_data.get('specific_type'),
                    'evidence': rel_data.get('evidence'),
                    'confidence': rel_data.get('confidence')
                }

                # Check if evidence already exists before adding
                if evidence_entry not in rel.evidence:
                    rel.evidence.append(evidence_entry)

                    avg_confidence = sum(e.get('confidence', 0.7) for e in rel.evidence) / len(rel.evidence)
                    rel.confidence = min(0.95, avg_confidence)

                    rel.save()
                    relationships_found += 1

            except Character.DoesNotExist:
                print(f"Character not found: {rel_data.get('character_1')} or {rel_data.get('character_2')}")
                continue
            except Exception as e:
                print(f"Error processing relationship: {e}")
                continue

    return relationships_found


def process_book_complete(epub_id, api_key):
    """
    Complete book processing pipeline with LLM-based character validation.

    Args:
        epub_id: ID of the EpubFile to process
        api_key: Google API key for LLM analysis

    Returns:
        Dictionary with processing stats
    """
    print(f"{'=' * 50}")
    print(f"Processing EPUB ID: {epub_id}")
    print(f"{'=' * 50}\n")

    # Step 1: Extract chapters
    print("Step 1: Extracting chapters...")
    process_epub_file(epub_id)
    print("✓ Chapters extracted\n")

    # Step 2: Extract characters with context
    print("Step 2: Extracting characters with NER...")
    char_count = extract_characters_with_chunks(epub_id)
    print(f"✓ Found {char_count} potential character names\n")

    # Step 3: Validate and deduplicate characters with LLM
    print("Step 3: Validating and deduplicating characters with LLM...")
    validation_stats = validate_and_deduplicate_characters_with_llm(epub_id, api_key)

    # Step 4: Extract relationships
    print("Step 4: Extracting relationships with LLM...")
    rel_count = extract_relationships_with_llm(epub_id, api_key)
    print(f"✓ Found {rel_count} relationships\n")

    print(f"{'=' * 50}")
    print("✓ Book processing complete!")
    print(f"{'=' * 50}")

    return {
        'chapters': Chapter.objects.filter(epub_id=epub_id).count(),
        'original_characters': validation_stats['original_count'],
        'invalid_removed': validation_stats.get('invalid_count', 0),
        'groups_merged': validation_stats.get('merged_count', 0),
        'final_characters': validation_stats['final_count'],
        'character_reduction': validation_stats['reduction'],
        'relationships': rel_count
    }


# ============================================================================
# LEGACY FUNCTIONS (kept for backwards compatibility)
# ============================================================================

def extract_characters_simple(epub_id):
    """
    LEGACY FUNCTION

    Extracts character names from EPUB using spaCy NER

    Args:
        epub_id: ID of the EpubFile to process

    Returns:
        Number of unique characters found
    """
    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    character_counts = {}
    first_appearance = {}

    for chapter in chapters:
        doc = nlp(chapter.content)
        annotated_sentences = []

        for sent in doc.sents:
            entities_in_sentence = []

            for ent in sent.ents:
                if ent.label_ == 'PERSON':
                    name = ent.text.strip()
                    name = name.rstrip("'s")

                    # skip single letters, common false positive
                    if len(name) <= 1:
                        continue

                    entities_in_sentence.append(name)

                    character_counts[name] = character_counts.get(name, 0) + 1

                    if name not in first_appearance:
                        first_appearance[name] = chapter.chapter_number

            if entities_in_sentence:
                annotated_sentences.append({
                    'text': sent.text.strip(),
                    'characters': entities_in_sentence
                })

        chapter.annotated_sentences = annotated_sentences
        chapter.save()

    for name, count in character_counts.items():
        Character.objects.update_or_create(
            epub=epub,
            name=name,
            defaults={
                'mention_count': count,
                'first_appearance_chapter': first_appearance[name]
            }
        )

    return len(character_counts)


def extract_relationships_simple(epub_id):
    """
    LEGACY FUNCTION
    """
    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    relationship_keywords = {
        'brother': ['brother', 'bro'],
        'sister': ['sister', 'sis'],
        'cousin': ['cousin'],
        'father': ['father', 'dad', 'papa', 'daddy'],
        'mother': ['mother', 'mom', 'mama', 'mommy'],
        'uncle': ['uncle', 'unc'],
        'aunt': ['aunt', 'auntie']
    }

    relationships_found = 0

    for chapter in chapters:
        for sentence_data in chapter.annotated_sentences:
            characters = sentence_data['characters']
            text = sentence_data['text']

            if len(characters) >= 2:
                for relationship_type, keywords in relationship_keywords.items():
                    for keyword in keywords:

                        pattern = rf"\b{keyword}(?:'s)?\b"
                        if re.search(pattern, text, re.IGNORECASE):
                            other_chars = [c for c in characters if c not in [characters[0], characters[1]]]

                            try:
                                char1_object = Character.objects.get(epub=epub, name=characters[0])
                                char2_object = Character.objects.get(epub=epub, name=characters[1])

                            except Character.DoesNotExist:
                                continue

                            rel, created = Relationship.objects.get_or_create(
                                epub=epub,
                                character_1=char1_object,
                                character_2=char2_object,
                                relationship_type=relationship_type,
                                defaults={
                                    'confidence': 0.7,
                                    'evidence': []
                                }
                            )

                            new_evidence = {
                                'chapter': chapter.chapter_number,
                                'text': text,
                                'other_characters': other_chars
                            }

                            if new_evidence not in rel.evidence:
                                rel.evidence.append(new_evidence)

                                rel.confidence = min(0.95, 0.7 + len(rel.evidence) * 0.05)
                                rel.save()

                            relationships_found += 1
                            break

    return relationships_found
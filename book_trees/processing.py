import ebooklib
import re
import spacy
import json
import requests
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
                epub = epub,
                title=chapter_data['title'],
                content = chapter_data['content'],
                chapter_number = chapter_data['chapter_number']

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


def find_character_name_variations(epub_id):
    """
    Find character name variations that should be merged.
    Uses multiple strategies:
    1. Substring matching (e.g., "Harry" in "Harry Potter")
    2. String similarity (e.g., "Jon" vs "John")
    3. Common patterns (e.g., "Mr. Smith" vs "Smith")

    Returns:
        List of character groups that should be merged
    """
    characters = Character.objects.filter(epub=epub_id).order_by('-mention_count')

    # Build similarity groups
    groups = []
    used = set()

    for i, char1 in enumerate(characters):
        if char1.id in used:
            continue

        group = [char1]
        name1 = char1.name.lower().strip()

        for char2 in characters[i + 1:]:
            if char2.id in used:
                continue

            name2 = char2.name.lower().strip()

            # Strategy 1: Substring matching
            if name1 in name2 or name2 in name1:
                # Only merge if length difference isn't too large
                if max(len(name1), len(name2)) / min(len(name1), len(name2)) <= 2:
                    group.append(char2)
                    used.add(char2.id)
                    continue

            # Strategy 2: String similarity (for typos, abbreviations)
            similarity = SequenceMatcher(None, name1, name2).ratio()
            if similarity >= 0.85:  # Very similar
                group.append(char2)
                used.add(char2.id)
                continue

            # Strategy 3: First/last name matching
            parts1 = name1.split()
            parts2 = name2.split()

            # Check if they share a significant name part
            for part1 in parts1:
                if len(part1) >= 3:  # Skip titles like "Mr"
                    for part2 in parts2:
                        if part1 == part2 and len(part1) >= 4:
                            group.append(char2)
                            used.add(char2.id)
                            break

        if len(group) > 1:
            groups.append(group)

        used.add(char1.id)

    return groups


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
    """
    Merge multiple character records into one primary character.
    Updates all relationships and annotations.
    """
    with transaction.atomic():
        # Combine mention counts
        total_mentions = primary_character.mention_count
        for char in characters_to_merge:
            if char.id != primary_character.id:
                total_mentions += char.mention_count

        primary_character.mention_count = total_mentions

        # Store aliases (if your model supports it)
        if hasattr(primary_character, 'aliases'):
            if not primary_character.aliases:
                primary_character.aliases = []

            for char in characters_to_merge:
                if char.id != primary_character.id and char.name not in primary_character.aliases:
                    primary_character.aliases.append(char.name)

        primary_character.save()

        # Update all relationships
        for char in characters_to_merge:
            if char.id == primary_character.id:
                continue

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


def deduplicate_characters(epub_id):
    """
    Simple character deduplication using string matching.
    No LLM required - fast and free!

    Returns:
        Dictionary with deduplication stats
    """
    original_count = Character.objects.filter(epub=epub_id).count()

    # Find groups of similar names
    groups = find_character_name_variations(epub_id)

    print(f"Found {len(groups)} groups of similar character names:")

    merged_count = 0
    for group in groups:
        # Use the longest name as primary (usually the full name)
        primary = max(group, key=lambda c: len(c.name))

        names = [c.name for c in group]
        print(f"  Merging: {', '.join(names)} → {primary.name}")

        # Merge the group
        merge_characters(primary, group)
        merged_count += 1

    final_count = Character.objects.filter(epub=epub_id).count()

    stats = {
        'original_count': original_count,
        'merged_groups': merged_count,
        'final_count': final_count,
        'reduction': original_count - final_count
    }

    print(f"\n{'=' * 50}")
    print(f"Deduplication complete!")
    print(f"Original characters: {stats['original_count']}")
    print(f"Merged groups: {stats['merged_groups']}")
    print(f"Final character count: {stats['final_count']}")
    print(f"Reduction: {stats['reduction']} ({(stats['reduction'] / original_count * 100):.1f}%)")

    return stats


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

                rel.evidence.append(evidence_entry)


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
    Complete book processing pipeline with deduplication.

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
    print("Step 2: Extracting characters...")
    char_count = extract_characters_with_chunks(epub_id)
    print(f"✓ Found {char_count} character variations\n")

    # Step 3: Deduplicate characters
    print("Step 3: Deduplicating characters...")
    dedup_stats = deduplicate_characters(epub_id)
    print()

    # Step 4: Extract relationships
    print("Step 4: Extracting relationships with LLM...")
    rel_count = extract_relationships_with_llm(epub_id, api_key)
    print(f"✓ Found {rel_count} relationships\n")

    print(f"{'=' * 50}")
    print("✓ Book processing complete!")
    print(f"{'=' * 50}")

    return {
        'chapters': Chapter.objects.filter(epub_id=epub_id).count(),
        'original_characters': dedup_stats['original_count'],
        'final_characters': dedup_stats['final_count'],
        'character_reduction': dedup_stats['reduction'],
        'relationships': rel_count
    }


def extract_characters_simple(epub_id):
    """
    LEGACY FUNCTION

    Extracts character names from EPUBn using spaCy NER

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

                    #skip single letters, common false positive
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
            epub = epub,
            name = name,
            defaults = {
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

            if len(characters)  >= 2:
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

                                rel.confidence=min(0.95, 0.7 + len(rel.evidence) * 0.05)
                                rel.save()

                            relationships_found += 1
                            break

    return relationships_found









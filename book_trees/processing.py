import re
import spacy
import json
import requests
import os
import sys
import time
from django.db import transaction
from .models import EpubFile, Chapter, Character, Relationship
from .epub_processing import extract_chapters_from_epub, process_epub_file  # noqa: F401
from collections import defaultdict
from typing import List, Dict, Any
from .post_processing import consolidate_relationships
from .pre_processing import run_pre_processing


# load NLP
nlp = spacy.load("book_trees/output/model-best")
if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("senter") and not nlp.has_pipe("parser"):
    nlp.add_pipe("sentencizer", first=True)


# ============================================================================
# PRE-FILTERING: Remove obvious non-characters before LLM processing
# ============================================================================

# Hard-coded list of first-person narrative markers and pronouns
NARRATOR_MARKERS = {
    'I', 'Me', 'My', 'Mine', 'Myself',
    'We', 'Us', 'Our', 'Ours', 'Ourselves',
    'You', 'Your', 'Yours', 'Yourself', 'Yourselves',
    'He', 'Him', 'His', 'She', 'Her', 'Hers',
    'They', 'Them', 'Their', 'Theirs'
}

# Common generic descriptors that spaCy tags as PERSON
GENERIC_DESCRIPTORS = {
    'man', 'woman', 'boy', 'girl', 'child', 'person', 'people',
    'warrior', 'warriors', 'soldier', 'soldiers', 'guard', 'guards',
    'servant', 'servants', 'lord', 'lady', 'king', 'queen',
    'prince', 'princess', 'knight', 'knights', 'merchant', 'merchants',
    'priest', 'priestess', 'mage', 'wizard', 'sorcerer',
    'stranger', 'strangers', 'traveler', 'travelers',
    'father', 'mother', 'brother', 'sister', 'son', 'daughter',
    'husband', 'wife', 'friend', 'friends', 'enemy', 'enemies'
}

# Emotion/state words that get tagged as PERSON
EMOTION_ACTION_WORDS = {
    'Annoyed', 'Calm', 'Frown', 'Smile', 'Laugh', 'Cry', 'Sigh',
    'Nod', 'Shake', 'Shrug', 'Grin', 'Scowl', 'Wince', 'Gasp',
    'Surprised', 'Confused', 'Worried', 'Afraid', 'Angry', 'Happy',
    'Sad', 'Excited', 'Nervous', 'Relieved', 'Shocked', 'Stunned'
}


def is_roman_numeral(text: str) -> bool:
    """Check if text is a Roman numeral (I, II, III, IV, V, etc.)"""
    if not text:
        return False
    roman_pattern = r'^[IVXLCDM]+$'
    return bool(re.match(roman_pattern, text.upper()))


def is_likely_character(name: str, entity_tokens=None, context_text: str = "") -> bool:
    """
    Conservative pre-filter to remove obvious non-characters.

    Returns True if the name should be KEPT (likely a real character).
    Returns False if the name should be FILTERED OUT (obvious junk).
    """

    if len(name) == 1:
        return False

    if name in NARRATOR_MARKERS or name.title() in NARRATOR_MARKERS:
        return False

    if name in EMOTION_ACTION_WORDS or name.title() in EMOTION_ACTION_WORDS:
        return False

    if is_roman_numeral(name):
        return False

    if '\n' in name or '\t' in name or '\r' in name:
        return False

    special_char_count = sum(1 for c in name if not c.isalnum() and c not in "'-. ")
    if special_char_count > 2:
        return False

    words = name.split()
    if len(words) == 1 and name.islower():
        return False

    if len(words) > 1:
        capitalized_count = sum(1 for w in words if w and w[0].isupper())
        if capitalized_count / len(words) < 0.5:
            return False

    name_lower = name.lower()
    words_lower = [w.lower() for w in words]

    if len(words) == 1 and name_lower in GENERIC_DESCRIPTORS:
        return False

    if len(words) > 1 and all(w in GENERIC_DESCRIPTORS for w in words_lower):
        return False

    articles = {'the', 'a', 'an'}
    if len(words) > 1 and words_lower[0] in articles:
        return False

    if entity_tokens:
        pos_tags = [token.pos_ for token in entity_tokens]
        verb_adj_count = sum(1 for pos in pos_tags if pos in ['VERB', 'ADJ'])
        if verb_adj_count > len(pos_tags) / 2:
            return False

        verb_lemmas = [token.lemma_.lower() for token in entity_tokens if token.pos_ == 'VERB']
        common_verb_lemmas = {'say', 'tell', 'ask', 'reply', 'answer', 'shout', 'whisper', 'think'}
        if any(lemma in common_verb_lemmas for lemma in verb_lemmas):
            return False

    if any(char.isdigit() for char in name):
        if name.replace(' ', '').replace('-', '').isdigit():
            return False

    return True


# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

def make_api_request_with_retry(url, headers, json_data, max_retries=5, initial_delay=1):
    """
    Make an API request with exponential backoff retry logic.

    Args:
        url: API endpoint URL
        headers: Request headers
        json_data: JSON payload
        max_retries: Maximum number of retry attempts (default: 5)
        initial_delay: Initial delay in seconds before first retry (default: 1)

    Returns:
        Response object if successful

    Raises:
        requests.exceptions.RequestException: If all retries fail
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json_data, timeout=30)

            if response.status_code == 200:
                return response

            if response.status_code in [429, 503, 500, 502, 504]:
                error_message = f"API request failed with status {response.status_code}"

                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', error_message)
                except Exception:
                    pass

                if attempt < max_retries - 1:
                    print(f"⚠ {error_message}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    print(f"✗ All {max_retries} retry attempts failed")
                    response.raise_for_status()
            else:
                response.raise_for_status()

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"⚠ Request timed out. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
                continue
            else:
                print(f"✗ Request timed out after {max_retries} attempts")
                raise

        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"⚠ Connection error. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
                continue
            else:
                print(f"✗ Connection failed after {max_retries} attempts")
                raise

    raise requests.exceptions.RequestException(f"Failed after {max_retries} attempts")


def extract_characters_with_chunks(epub_id):
    """
    Extracts candidate character names from EPUB using spaCy NER + is_likely_character
    pre-filtering, then delegates to pre_processing.run_pre_processing for:
      - Character scoring & validation (score ≥ 2 threshold)
      - Sentence-level chunk creation (chunk only when 2+ validated characters
        appear in the same sentence)

    Saves validated characters and annotated sentence chunks to the DB.

    Args:
        epub_id: ID of the EpubFile to process

    Returns:
        Dictionary with stats:
            - raw_entities: Number of PERSON entities before any filtering
            - filtered_out: Number removed by is_likely_character
            - pre_filter_unique: Unique candidate names after is_likely_character
            - validated_characters: Number of names that passed scoring (score ≥ 2)
    """
    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    # Stats
    raw_entity_count = 0
    filtered_entity_count = 0

    # Per-chapter candidate names (post is_likely_character, pre scoring)
    candidate_names_per_chapter: Dict[int, List[str]] = {}

    # Also track first appearance for DB writes later
    first_appearance: Dict[str, int] = {}

    # Collect chapter content for pre_processing
    chapters_data = []
    chapter_docs= {}

    for chapter in chapters:
        doc = nlp(chapter.content)
        chapter_docs[chapter.chapter_number] = doc
        chapter_candidates: List[str] = []
        seen_in_chapter = set()

        for sent in doc.sents:
            for ent in sent.ents:
                if ent.label_ != 'PERSON':
                    continue

                raw_entity_count += 1

                name = ent.text.strip()
                if name.endswith("'s"):
                    name = name[:-2]

                context = sent.text.strip()

                if not is_likely_character(name, ent, context):
                    filtered_entity_count += 1
                    continue

                words = name.split()
                capitalized_words = sum(1 for w in words if w and w[0].isupper())
                if len(words) > 1 and capitalized_words / len(words) < 0.5:
                    filtered_entity_count += 1
                    continue

                if name not in seen_in_chapter:
                    chapter_candidates.append(name)
                    seen_in_chapter.add(name)

                if name not in first_appearance:
                    first_appearance[name] = chapter.chapter_number

        candidate_names_per_chapter[chapter.chapter_number] = chapter_candidates
        chapters_data.append({
            'chapter_number': chapter.chapter_number,
            'content': chapter.content,
        })

    # ── Pre-processing: score characters + build sentence chunks ────────────
    validated_characters, chunks_by_chapter = run_pre_processing(
        chapters_data,
        candidate_names_per_chapter,
    )

    validated_names = set(validated_characters.keys())

    # ── Persist validated characters ────────────────────────────────────────
    # Re-count mentions using only validated names so mention_count is accurate
    mention_counts: Dict[str, int] = {name: 0 for name in validated_names}

    for chapter in chapters:
        doc = chapter_docs[chapter.chapter_number]
        for sent in doc.sents:
            for ent in sent.ents:
                if ent.label_ != 'PERSON':
                    continue
                name = ent.text.strip()
                if name.endswith("'s"):
                    name = name[:-2]
                if name in validated_names:
                    mention_counts[name] += 1

    for name in validated_names:
        Character.objects.update_or_create(
            epub=epub,
            name=name,
            defaults={
                'mention_count': mention_counts.get(name, 0),
                'first_appearance_chapter': first_appearance.get(name, 1),
                'syntactic_score': validated_characters.get(name, 0),
            }
        )

    # ── Persist sentence chunks per chapter ─────────────────────────────────
    for chapter in chapters:
        chunks = chunks_by_chapter.get(chapter.chapter_number, [])
        chapter.annotated_sentences = chunks
        chapter.save()

    # ── Print stats ──────────────────────────────────────────────────────────
    pre_filter_unique = len(set(
        name
        for names in candidate_names_per_chapter.values()
        for name in names
    ))

    print(f"\n{'=' * 60}")
    print("Extraction & Pre-Processing Statistics")
    print(f"{'=' * 60}")
    print(f"Raw PERSON entities found:          {raw_entity_count}")
    print(f"Filtered out (is_likely_character): {filtered_entity_count}")
    print(f"Unique candidates after NER filter: {pre_filter_unique}")
    print(f"Validated characters (score ≥ 2):   {len(validated_names)}")
    if raw_entity_count > 0:
        print(f"Total reduction: {((raw_entity_count - len(validated_names)) / raw_entity_count * 100):.1f}%")
    print(f"{'=' * 60}\n")

    return {
        'raw_entities': raw_entity_count,
        'filtered_out': filtered_entity_count,
        'pre_filter_unique': pre_filter_unique,
        'unique_characters': len(validated_names),
    }


def analyze_chunks_batch_with_llm(chunks: List[Dict[str, Any]], api_key: str = None) -> List[Dict]:
    """
    Sends a batch of chunks that all share the same character set to Gemini
    2.5 Flash-Lite for relationship analysis in a single API request.

    Grouping chunks by character pair reduces API calls significantly: instead
    of one request per sentence chunk, we make one request per unique character
    pair combination found across all chunks.

    Args:
        chunks: List of chunk dicts, each containing:
            - context: str - The text excerpt
            - characters_in_context: List[str] - Character names (same across batch)
            - chapter_number: int - Chapter number for reference
        api_key: Google API Key

    Returns:
        List of relationship dictionaries extracted from the LLM response,
        each annotated with chapter_number and evidence from its source chunk.
    """
    if not chunks:
        return []

    # All chunks in the batch share the same characters (enforced by the
    # grouping logic in extract_relationships_with_llm), so we read them
    # from the first chunk.
    characters = chunks[0].get('characters_in_context', [])
    if len(characters) < 2:
        return []

    # Build a numbered list of excerpts so the LLM can reference them
    excerpts_text = "\n\n".join(
        f"Excerpt {i + 1} (Chapter {chunk.get('chapter_number', '?')}):\n{chunk['context']}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""
    You are extracting relationships between characters from a novel.

    Below are excerpts where the SAME characters appear together. Treat all excerpts as evidence for the relationship between them.

    {excerpts_text}

    Characters mentioned: {', '.join(characters)}

    Your task:
    Determine the most accurate relationship between each pair of characters.

    IMPORTANT RULES:
    1. You MUST choose the closest relationship_type from the allowed list below.
    2. DO NOT use "other" unless absolutely none of the types apply.
    3. If the relationship changes across the story, choose the dominant or most recent relationship.
    4. Use the excerpts as evidence. Do NOT invent information.
    5. Return ONE entry per character pair.

    Allowed relationship types:

    family
    romantic
    friend
    ally
    enemy
    mentor_student
    master_servant
    rival
    captor_captive
    professional
    political
    other

    Definitions:

    family → blood or marriage relationships (parent, sibling, cousin, spouse)
    romantic → lovers, romantic attraction, marriage
    friend → personal friendship or companionship
    ally → cooperative relationship toward shared goals
    enemy → hostility, betrayal, conflict, or opposition
    mentor_student → teaching or guiding relationship
    master_servant → hierarchical service relationship
    rival → competition without direct hostility
    captor_captive → imprisonment, hostage, or forced control
    professional → coworkers, military comrades, colleagues
    political → ruler/subject or political allegiance

    Return ONLY valid JSON in this format:

    {{
      "relationships": [
        {{
          "character_1": "Name",
          "character_2": "Name",
          "relationship_type": "one_of_the_types_above",
          "specific_type": "short label like brother, commander, captor, rival, etc",
          "confidence": 0.0-1.0,
          "evidence_excerpt_ids": [1,2]
        }}
      ]
    }}

    Rules for evidence:
    - evidence_excerpt_ids must reference the numbered excerpts above.
    - choose the excerpts that best support the relationship.

    Return JSON only. No explanation.
    """

    content = None

    try:
        response = make_api_request_with_retry(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json_data={
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.15,
                    "maxOutputTokens": 8192,
                    "responseMimeType": "application/json"
                }
            }
        )

        result = response.json()

        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            print(f"Unexpected Gemini response format: {result}")
            return []

        parsed_data = json.loads(content)
        relationships = parsed_data.get('relationships', [])

        # Annotate each relationship with the chapter of the first chunk in the
        # batch (consistent with the previous single-chunk behaviour).
        for rel in relationships:
            rel['chapter_number'] = chunks[0].get('chapter_number')

            excerpt_ids = rel.get("evidence_excerpt_ids", [])

            evidence_chunks = []
            for idx in excerpt_ids:
                if 1 <= idx <= len(chunks):
                    evidence_chunks.append(chunks[idx - 1])

            rel["evidence_chunks"] = evidence_chunks

        return relationships

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        if content is None:
            print("No content was received from the API")
            return []

        # Attempt recovery: strip markdown code fences
        try:
            cleaned = re.sub(r'^```(?:json)?\s*', '', content.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned).strip()
            parsed_data = json.loads(cleaned)
            relationships = parsed_data.get('relationships', [])
            for rel in relationships:
                rel['chapter_number'] = chunks[0].get('chapter_number')
                rel['evidence'] = " | ".join(c['context'] for c in chunks)
            print(f"  ↳ Recovered after stripping code fences ({len(relationships)} relationships)")
            return relationships
        except json.JSONDecodeError:
            pass

        # Partial recovery: salvage complete relationship objects
        try:
            partial_matches = re.findall(r'\{[^{}]+\}', content)
            recovered = []
            for match in partial_matches:
                try:
                    obj = json.loads(match)
                    if 'character_1' in obj and 'character_2' in obj and 'relationship_type' in obj:
                        obj['chapter_number'] = chunks[0].get('chapter_number')
                        obj['evidence'] = " | ".join(c['context'] for c in chunks)
                        recovered.append(obj)
                except json.JSONDecodeError:
                    continue
            if recovered:
                print(f"  ↳ Partially recovered {len(recovered)} relationships from truncated response")
                return recovered
        except Exception:
            pass

        print(f"  ↳ Could not recover — skipping batch. Raw content preview: {content[:200]}")
        return []
    except Exception as e:
        print(f"Unexpected error in LLM analysis: {e}")
        return []


def merge_characters_internal(primary_character, characters_to_merge):
    """
    Merge multiple character records into one primary character.
    Updates all relationships and annotations.

    Args:
        primary_character: The Character object to keep
        characters_to_merge: List of Character objects to merge into primary
    """
    with transaction.atomic():
        total_mentions = primary_character.mention_count
        for char in characters_to_merge:
            total_mentions += char.mention_count

        primary_character.mention_count = total_mentions

        if hasattr(primary_character, 'aliases'):
            if not primary_character.aliases:
                primary_character.aliases = []
            for char in characters_to_merge:
                if char.name not in primary_character.aliases:
                    primary_character.aliases.append(char.name)

        primary_character.save()

        for char in characters_to_merge:
            Relationship.objects.filter(character_1=char).update(character_1=primary_character)
            Relationship.objects.filter(character_2=char).update(character_2=primary_character)
            char.delete()

        relationships = (
            Relationship.objects.filter(character_1=primary_character)
            | Relationship.objects.filter(character_2=primary_character)
        )

        seen = set()
        for rel in relationships:
            key = (
                min(rel.character_1.id, rel.character_2.id),
                max(rel.character_1.id, rel.character_2.id),
                rel.relationship_type
            )

            if key in seen:
                duplicate = Relationship.objects.filter(
                    character_1__in=[rel.character_1, rel.character_2],
                    character_2__in=[rel.character_1, rel.character_2],
                    relationship_type=rel.relationship_type
                ).exclude(id=rel.id).first()

                if duplicate:
                    duplicate.evidence.extend(rel.evidence)
                    duplicate.save()
                    rel.delete()
            else:
                seen.add(key)


def extract_relationships_with_llm(epub_id: int, api_key: str = None, batch_size: int = 10):
    """
    Extract relationships from an EPUB using LLM analysis on sentence chunks.

    Chunks are grouped by their sorted character-pair key so that all sentences
    featuring the same pair of characters are sent to the LLM in a single
    batched request. This significantly reduces the total number of API calls
    compared to the previous one-chunk-per-request approach.

    Args:
        epub_id: ID of the EpubFile to process
        api_key: API key for the LLM service
        batch_size: Maximum number of chunks to include in a single LLM request
                    (guards against excessively long prompts)

    Returns:
        Number of relationships found
    """
    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    # ── Collect all valid chunks across every chapter ────────────────────────
    # Key: frozenset of character names -> list of chunk dicts
    chunks_by_character_pair: Dict[frozenset, List[Dict[str, Any]]] = defaultdict(list)

    for chapter in chapters:
        for chunk in (chapter.annotated_sentences or []):
            characters = chunk.get('characters_in_context', [])
            if len(characters) < 2:
                continue
            chunk_with_chapter = {**chunk, 'chapter_number': chapter.chapter_number}
            pair_key = frozenset(characters)
            chunks_by_character_pair[pair_key].append(chunk_with_chapter)

    total_groups = len(chunks_by_character_pair)
    total_chunks = sum(len(v) for v in chunks_by_character_pair.values())
    print(f"Grouped {total_chunks} chunks into {total_groups} unique character-pair groups")

    # ── Send one (or a few) LLM request(s) per character-pair group ─────────
    all_relationships = []

    for pair_key, group_chunks in chunks_by_character_pair.items():
        # Split very large groups into sub-batches to avoid oversized prompts
        for batch_start in range(0, len(group_chunks), batch_size):
            batch = group_chunks[batch_start: batch_start + batch_size]
            relationships = analyze_chunks_batch_with_llm(batch, api_key)
            all_relationships.extend(relationships)

    # Drop low-confidence relationships before saving
    all_relationships = [r for r in all_relationships if r.get('confidence', 0) >= 0.7]
    print(f"Relationships after confidence filter (≥ 0.7): {len(all_relationships)}")

    relationships_found = 0

    with transaction.atomic():
        for rel_data in all_relationships:
            try:
                char1 = Character.objects.get(epub=epub, name=rel_data['character_1'])
                char2 = Character.objects.get(epub=epub, name=rel_data['character_2'])

                if char1.name > char2.name:
                    char1, char2 = char2, char1

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
                    "chapter": rel_data.get("chapter_number"),
                    "specific_type": rel_data.get("specific_type"),
                    "confidence": rel_data.get("confidence"),
                    "chunks": rel_data.get("evidence_chunks", [])
                }

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
    Complete book processing pipeline.

    Steps:
      1. Extract chapters from EPUB
      2. NER + pre-filtering + character scoring + sentence-level chunking
      3. LLM character validation & deduplication
      4. LLM relationship extraction
      5. Post-processing / consolidation

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

    # Step 2: NER + pre-filtering + scoring + chunking
    print("Step 2: Extracting characters (NER → scoring → sentence chunks)...")
    extraction_stats = extract_characters_with_chunks(epub_id)
    print(f"✓ {extraction_stats['unique_characters']} validated characters, chunks saved\n")

    # Step 3: LLM relationship extraction
    print("Step 3: Extracting relationships with LLM...")
    rel_count = extract_relationships_with_llm(epub_id, api_key)
    print(f"✓ Found {rel_count} relationships\n")

    print(f"{'=' * 50}")
    print("✓ Book processing complete!")
    print(f"{'=' * 50}")

    consolidate_stats = consolidate_relationships(epub_id)  # noqa: F841

    return {
        'chapters': Chapter.objects.filter(epub_id=epub_id).count(),
        'original_characters': extraction_stats['raw_entities'],
        'final_characters': extraction_stats['unique_characters'],
        'character_reduction': (
            f"{extraction_stats['raw_entities'] - extraction_stats['unique_characters']} removed"
        ),
        'invalid_removed': extraction_stats.get('filtered_out', 0),
        'groups_merged': extraction_stats.get('groups_merged', 0),
        'relationships': rel_count,
        # Legacy keys
        'raw_entities': extraction_stats['raw_entities'],
        'pre_filtered': extraction_stats['filtered_out'],
        'after_pre_filter': extraction_stats['pre_filter_unique'],
    }
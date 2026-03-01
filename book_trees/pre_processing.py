import spacy
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# load NLP (reuse the same model as processing.py)
nlp = spacy.load("en_core_web_lg")


# ============================================================================
# CHARACTER SCORING & VALIDATION
# ============================================================================

# Dialogue attribution verbs — used to detect if a character is a speaker
ATTRIBUTION_VERBS = {
    'said', 'say', 'asked', 'ask', 'replied', 'reply', 'answered', 'answer',
    'shouted', 'shout', 'whispered', 'whisper', 'muttered', 'mutter',
    'called', 'call', 'cried', 'cry', 'exclaimed', 'exclaim',
    'declared', 'declare', 'announced', 'announce', 'told', 'tell',
    'laughed', 'sighed', 'sigh', 'added', 'add', 'continued', 'continue',
    'interrupted', 'interrupt', 'agreed', 'agree', 'insisted', 'insist',
}


def score_characters(
    chapters_data: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Score each candidate character across all chapters using a lightweight
    heuristic rubric and return only those that meet the threshold.

    Scoring rules
    -------------
    +2  appears ≥ 3 times (total mentions across all chapters)
    +2  appears as syntactic subject (nsubj) at least once
    +1  appears in dialogue attribution (near a speech verb)
    +1  appears across ≥ 2 distinct chapters
    −2  appears only once
    −2  never acts (never nsubj and never nobj/dobj)

    Threshold: score ≥ 2  →  validated character

    Args:
        chapters_data: list of dicts, each with keys:
            - 'chapter_number': int
            - 'content': str  (raw chapter text)
            - 'candidate_names': List[str]  (pre-filtered PERSON entities)

    Returns:
        Dict mapping validated character name → final score
    """

    # Accumulators
    mention_counts: Dict[str, int] = defaultdict(int)
    chapter_sets: Dict[str, set] = defaultdict(set)   # chapters where name appears
    is_nsubj: Dict[str, bool] = defaultdict(bool)
    is_actor: Dict[str, bool] = defaultdict(bool)      # nsubj OR nobj/dobj
    in_attribution: Dict[str, bool] = defaultdict(bool)

    for chapter in chapters_data:
        chapter_num = chapter['chapter_number']
        content = chapter['content']
        candidate_names: List[str] = chapter.get('candidate_names', [])

        if not candidate_names:
            continue

        # Build a lowercase lookup set for fast membership checks
        candidate_lower = {n.lower(): n for n in candidate_names}

        doc = nlp(content)

        for sent in doc.sents:
            sent_text_lower = sent.text.lower()

            # Check for attribution verbs in this sentence
            has_attribution = any(
                token.lemma_.lower() in ATTRIBUTION_VERBS
                for token in sent
            )

            for token in sent:
                token_lower = token.text.lower()

                # Try to match token text against known candidate names
                # (handles single-token names; multi-token handled via span below)
                if token_lower in candidate_lower:
                    canonical = candidate_lower[token_lower]
                    mention_counts[canonical] += 1
                    chapter_sets[canonical].add(chapter_num)

                    dep = token.dep_
                    if dep == 'nsubj':
                        is_nsubj[canonical] = True
                        is_actor[canonical] = True
                    elif dep in ('dobj', 'nsubjpass', 'obj', 'iobj'):
                        is_actor[canonical] = True

                    if has_attribution:
                        in_attribution[canonical] = True

            # Handle multi-token candidate names via entity spans
            for ent in sent.ents:
                if ent.label_ != 'PERSON':
                    continue

                ent_text = ent.text.strip()
                if ent_text.endswith("'s"):
                    ent_text = ent_text[:-2]

                ent_lower = ent_text.lower()
                if ent_lower not in candidate_lower:
                    continue

                canonical = candidate_lower[ent_lower]
                # Avoid double-counting single-token entities already handled above
                if len(ent) > 1:
                    mention_counts[canonical] += 1
                    chapter_sets[canonical].add(chapter_num)

                # Syntactic role: use the root token of the span
                root = ent.root
                dep = root.dep_
                if dep == 'nsubj':
                    is_nsubj[canonical] = True
                    is_actor[canonical] = True
                elif dep in ('dobj', 'nsubjpass', 'obj', 'iobj'):
                    is_actor[canonical] = True

                if has_attribution:
                    in_attribution[canonical] = True

    # ── Compute scores ──────────────────────────────────────────────────────
    scores: Dict[str, int] = {}

    all_names = set(mention_counts.keys())

    for name in all_names:
        score = 0
        count = mention_counts[name]

        # Frequency bonuses / penalties
        if count >= 3:
            score += 2
        elif count == 1:
            score -= 2

        # Syntactic subject
        if is_nsubj[name]:
            score += 2

        # Dialogue attribution
        if in_attribution[name]:
            score += 1

        # Cross-chapter spread
        if len(chapter_sets[name]) >= 2:
            score += 1

        # Never acts (never subject or object)
        if not is_actor[name]:
            score -= 2

        scores[name] = score

    # ── Apply threshold ─────────────────────────────────────────────────────
    THRESHOLD = 3
    validated = {name: score for name, score in scores.items() if score >= THRESHOLD}

    return validated


# ============================================================================
# SENTENCE-LEVEL CHUNK CREATION
# ============================================================================

def create_sentence_chunks(
    chapter_number: int,
    content: str,
    validated_character_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Create chunks only when 2 or more validated characters appear in the
    **same sentence** (replacing the old sliding-window approach).

    Args:
        chapter_number: The chapter this content belongs to.
        content: Raw text of the chapter.
        validated_character_names: List of character names that passed scoring.

    Returns:
        List of chunk dicts, each containing:
            - 'center_sentence': str
            - 'context': str  (same as center_sentence — single sentence)
            - 'characters_in_sentence': List[str]
            - 'characters_in_context': List[str]  (same list)
            - 'sentence_index': int
            - 'chapter_number': int
    """
    if not validated_character_names:
        return []

    # Build a fast lookup: lowercase name → canonical name
    name_lookup: Dict[str, str] = {n.lower(): n for n in validated_character_names}

    doc = nlp(content)
    sentences = list(doc.sents)
    chunks: List[Dict[str, Any]] = []

    for sent_idx, sent in enumerate(sentences):
        found_names: List[str] = []

        # Check entity spans first (catches multi-token names)
        seen_spans = set()
        for ent in sent.ents:
            if ent.label_ != 'PERSON':
                continue

            ent_text = ent.text.strip()
            if ent_text.endswith("'s"):
                ent_text = ent_text[:-2]

            canonical = name_lookup.get(ent_text.lower())
            if canonical and canonical not in found_names:
                found_names.append(canonical)
                # Mark which token indices are covered by this span
                for tok in ent:
                    seen_spans.add(tok.i)

        # Also check individual tokens not already covered by a span
        for token in sent:
            if token.i in seen_spans:
                continue
            canonical = name_lookup.get(token.text.lower())
            if canonical and canonical not in found_names:
                found_names.append(canonical)

        # Only create a chunk if 2+ validated characters appear in this sentence
        if len(found_names) >= 2:
            sentence_text = sent.text.strip()
            chunks.append({
                'center_sentence': sentence_text,
                'context': sentence_text,
                'characters_in_sentence': found_names,
                'characters_in_context': found_names,
                'sentence_index': sent_idx,
                'chapter_number': chapter_number,
            })

    return chunks


# ============================================================================
# COMBINED PIPELINE ENTRY POINT
# ============================================================================

def run_pre_processing(
    chapters_data: List[Dict[str, Any]],
    candidate_names_per_chapter: Dict[int, List[str]],
) -> Tuple[Dict[str, int], Dict[int, List[Dict[str, Any]]]]:
    """
    Full pre-processing pipeline:
      1. Score and validate candidate character names.
      2. Create sentence-level chunks for each chapter using only validated names.

    Args:
        chapters_data: List of dicts with keys:
            - 'chapter_number': int
            - 'content': str
        candidate_names_per_chapter: Dict mapping chapter_number → list of
            pre-filtered PERSON entity strings (output of spaCy NER +
            is_likely_character filter from processing.py).

    Returns:
        Tuple of:
            - validated_characters: Dict[name → score]
            - chunks_by_chapter: Dict[chapter_number → list of chunk dicts]
    """

    # Attach candidate names to each chapter entry for the scorer
    enriched = []
    for ch in chapters_data:
        enriched.append({
            **ch,
            'candidate_names': candidate_names_per_chapter.get(ch['chapter_number'], [])
        })

    # Step 1 — Score & filter characters
    print("\n" + "=" * 60)
    print("Pre-Processing: Character Scoring")
    print("=" * 60)

    validated_characters = score_characters(enriched)
    validated_names = list(validated_characters.keys())

    total_candidates = sum(len(v) for v in candidate_names_per_chapter.values())
    print(f"Candidate names (post-NER filter): {total_candidates}")
    print(f"Validated characters (score ≥ 2):  {len(validated_names)}")

    if validated_characters:
        print("\nTop characters by score:")
        for name, score in sorted(validated_characters.items(), key=lambda x: -x[1])[:20]:
            print(f"  {name:<30} score={score}")

    print("=" * 60 + "\n")

    # Step 2 — Create sentence-level chunks
    print("=" * 60)
    print("Pre-Processing: Sentence Chunk Creation")
    print("=" * 60)

    chunks_by_chapter: Dict[int, List[Dict[str, Any]]] = {}
    total_chunks = 0

    for ch in chapters_data:
        ch_num = ch['chapter_number']
        chunks = create_sentence_chunks(ch_num, ch['content'], validated_names)
        chunks_by_chapter[ch_num] = chunks
        total_chunks += len(chunks)

    print(f"Total sentence chunks created: {total_chunks}")
    print("=" * 60 + "\n")

    return validated_characters, chunks_by_chapter
import spacy
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from fastcoref import FCoref

# ============================================================================
# NLP MODEL INITIALISATION
# ============================================================================

# Shared spaCy model (same as processing.py)
nlp = spacy.load("en_core_web_lg")

# fastcoref model — loaded once at module level to avoid repeated disk I/O.
# FCoref is the lightweight, fast variant; swap for LingMessCoref if you want
# higher accuracy at the cost of speed.
_coref_model: Optional[FCoref] = None


def _get_coref_model() -> FCoref:
    """Lazily initialise the fastcoref model (once per process)."""
    global _coref_model
    if _coref_model is None:
        print("Loading fastcoref model (first call only)…")
        # RoBERTa doesn't support scaled_dot_product_attention yet.
        # fastcoref loads its model via FCorefModel.from_pretrained (a
        # PreTrainedModel subclass), so we patch that class method directly.
        from fastcoref.coref_models.modeling_fcoref import FCorefModel
        _orig = FCorefModel.from_pretrained.__func__
        def _patched(cls, *args, **kwargs):
            kwargs.setdefault("attn_implementation", "eager")
            return _orig(cls, *args, **kwargs)
        FCorefModel.from_pretrained = classmethod(_patched)
        try:
            _coref_model = FCoref(device='cpu')  # change to 'cuda' if GPU available
        finally:
            FCorefModel.from_pretrained = classmethod(_orig)
    return _coref_model


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

# Third-person pronouns we are willing to resolve.
# We intentionally exclude first/second-person to avoid narrator bleed.
RESOLVABLE_PRONOUNS = {
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'they', 'them', 'their', 'theirs', 'themselves',
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

    Threshold: score ≥ 3  →  validated character

    Args:
        chapters_data: list of dicts, each with keys:
            - 'chapter_number': int
            - 'content': str  (raw or coref-resolved chapter text)
            - 'candidate_names': List[str]  (pre-filtered PERSON entities)

    Returns:
        Dict mapping validated character name → final score
    """

    mention_counts: Dict[str, int] = defaultdict(int)
    chapter_sets: Dict[str, set] = defaultdict(set)
    is_nsubj: Dict[str, bool] = defaultdict(bool)
    is_actor: Dict[str, bool] = defaultdict(bool)
    in_attribution: Dict[str, bool] = defaultdict(bool)

    for chapter in chapters_data:
        chapter_num = chapter['chapter_number']
        content = chapter['content']
        candidate_names: List[str] = chapter.get('candidate_names', [])

        if not candidate_names:
            continue

        candidate_lower = {n.lower(): n for n in candidate_names}
        doc = nlp(content)

        for sent in doc.sents:
            has_attribution = any(
                token.lemma_.lower() in ATTRIBUTION_VERBS
                for token in sent
            )

            for token in sent:
                token_lower = token.text.lower()
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
                if len(ent) > 1:
                    mention_counts[canonical] += 1
                    chapter_sets[canonical].add(chapter_num)

                root = ent.root
                dep = root.dep_
                if dep == 'nsubj':
                    is_nsubj[canonical] = True
                    is_actor[canonical] = True
                elif dep in ('dobj', 'nsubjpass', 'obj', 'iobj'):
                    is_actor[canonical] = True

                if has_attribution:
                    in_attribution[canonical] = True

    scores: Dict[str, int] = {}
    for name in set(mention_counts.keys()):
        score = 0
        count = mention_counts[name]

        if count >= 3:
            score += 2
        elif count == 1:
            score -= 2

        if is_nsubj[name]:
            score += 2

        if in_attribution[name]:
            score += 1

        if len(chapter_sets[name]) >= 2:
            score += 1

        if not is_actor[name]:
            score -= 2

        scores[name] = score

    THRESHOLD = 3
    return {name: score for name, score in scores.items() if score >= THRESHOLD}


# ============================================================================
# COREFERENCE RESOLUTION  (fastcoref)
# ============================================================================

def resolve_coreferences(
    text: str,
    candidate_names: List[str],
) -> str:
    """
    Replace resolvable pronouns (he/she/they etc.) with their named-character
    antecedent using fastcoref, where the antecedent is found in
    `candidate_names`.

    Design decisions
    ----------------
    - Only third-person pronouns are substituted (RESOLVABLE_PRONOUNS).
      First/second-person pronouns are left alone to avoid narrator bleed.
    - Only antecedents that match a known candidate name are used.
      This keeps noise low: unrecognised spans are never injected into text.
    - Within each coreference cluster, the antecedent is chosen as the
      *first* mention in the cluster that matches a candidate name.
      fastcoref returns clusters as lists of (start, end) char-span tuples
      in document order, so the first match is typically the introductory
      mention.
    - Substitutions are applied right-to-left (reverse char order) so that
      earlier offsets remain valid as we rewrite the string.

    Args:
        text:             Raw chapter text.
        candidate_names:  Names to treat as valid antecedents (may be the
                          full loose NER list on the first pass, or the
                          validated list on the second pass).

    Returns:
        Rewritten text with pronouns replaced by character names.
    """
    if not candidate_names or not text.strip():
        return text

    model = _get_coref_model()
    name_lower = {n.lower(): n for n in candidate_names}

    # fastcoref returns a list of clusters; each cluster is a list of
    # (start_char, end_char) span tuples.
    preds = model.predict(texts=[text])
    # predict() returns a list when given a list input; grab the first result
    clusters = preds[0].get_clusters(as_strings=False)

    if not clusters:
        return text

    # Build a list of (start, end, replacement) for every pronoun mention
    # whose cluster contains a named antecedent.
    replacements: List[Tuple[int, int, str]] = []

    for cluster in clusters:
        # Find the first mention in this cluster that matches a candidate name
        antecedent_name: Optional[str] = None
        for start, end in cluster:
            span_text = text[start:end].strip()
            canonical = name_lower.get(span_text.lower())
            if canonical:
                antecedent_name = canonical
                break  # use first named mention as antecedent

        if antecedent_name is None:
            continue  # no named antecedent found in cluster — skip

        # Replace all pronoun mentions in this cluster with the antecedent name
        for start, end in cluster:
            span_text = text[start:end].strip()
            if span_text.lower() in RESOLVABLE_PRONOUNS:
                replacements.append((start, end, antecedent_name))

    # Apply replacements right-to-left to preserve offsets
    replacements.sort(key=lambda x: x[0], reverse=True)
    text_list = list(text)
    for start, end, name in replacements:
        text_list[start:end] = list(name)

    return "".join(text_list)


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
    same sentence.

    Args:
        chapter_number:            Chapter this content belongs to.
        content:                   Coref-resolved text of the chapter.
        validated_character_names: Names that passed scoring.

    Returns:
        List of chunk dicts with keys:
            - 'center_sentence', 'context'
            - 'characters_in_sentence', 'characters_in_context'
            - 'sentence_index', 'chapter_number'
    """
    if not validated_character_names:
        return []

    name_lookup: Dict[str, str] = {n.lower(): n for n in validated_character_names}
    doc = nlp(content)
    sentences = list(doc.sents)
    chunks: List[Dict[str, Any]] = []

    for sent_idx, sent in enumerate(sentences):
        found_names: List[str] = []
        seen_spans: set = set()

        for ent in sent.ents:
            if ent.label_ != 'PERSON':
                continue
            ent_text = ent.text.strip()
            if ent_text.endswith("'s"):
                ent_text = ent_text[:-2]
            canonical = name_lookup.get(ent_text.lower())
            if canonical and canonical not in found_names:
                found_names.append(canonical)
                for tok in ent:
                    seen_spans.add(tok.i)

        for token in sent:
            if token.i in seen_spans:
                continue
            canonical = name_lookup.get(token.text.lower())
            if canonical and canonical not in found_names:
                found_names.append(canonical)

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
    Full pre-processing pipeline (two-pass coref-aware):

      Pass 1 — Loose scoring on raw text
        1a. Run score_characters on raw text + all NER candidates.
            This gives us a broad (lower-threshold) name list to anchor
            the coreference resolver.

      Pass 2 — Coreference resolution + final scoring
        2a. Resolve coreferences in each chapter using the Pass-1 name list.
        2b. Re-run score_characters on the resolved text.
            Mention counts now include pronoun mentions → secondary characters
            that were pronoun-heavy survive the threshold.

      Pass 3 — Sentence chunking
        3.  Create sentence-level chunks on resolved text using the final
            validated name list.

    Why two scoring passes?
    -----------------------
    Coreference resolution needs a name list to know which antecedents are
    valid. But scoring needs resolved text to count mentions accurately.
    The first (loose) pass breaks the chicken-and-egg deadlock by giving the
    resolver a broad set of anchors without discarding anyone prematurely.

    Args:
        chapters_data: List of dicts with 'chapter_number' and 'content'.
        candidate_names_per_chapter: Dict mapping chapter_number → list of
            pre-filtered PERSON entity strings from processing.py NER.

    Returns:
        Tuple of:
            - validated_characters: Dict[name → score]
            - chunks_by_chapter:   Dict[chapter_number → list of chunk dicts]
    """

    # ── Attach candidate names to chapter entries ────────────────────────────
    enriched_raw = [
        {**ch, 'candidate_names': candidate_names_per_chapter.get(ch['chapter_number'], [])}
        for ch in chapters_data
    ]

    # ── Pass 1: Loose scoring on raw text ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pre-Processing Pass 1: Loose Character Scoring (raw text)")
    print("=" * 60)

    # Temporarily lower threshold to cast a wider net for coref anchors.
    # We do this by scoring normally and keeping score ≥ 1 (instead of ≥ 3).
    # We achieve this by calling score_characters and then filtering manually.
    raw_scores = score_characters(enriched_raw)  # already returns score ≥ 3

    # Also collect names that scored ≥ 1 (for coref anchoring) without
    # re-implementing the scorer — we replicate the scoring loop with a
    # lower threshold just for the anchor list.
    loose_anchor_names = _score_with_threshold(enriched_raw, threshold=1)

    total_candidates = sum(len(v) for v in candidate_names_per_chapter.values())
    print(f"Candidate names (post-NER filter):  {total_candidates}")
    print(f"Loose coref anchors (score ≥ 1):    {len(loose_anchor_names)}")
    print(f"Strict validated (score ≥ 3):       {len(raw_scores)}")

    # ── Pass 2a: Coreference resolution ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pre-Processing Pass 2a: Coreference Resolution (fastcoref)")
    print("=" * 60)

    resolved_chapters: List[Dict[str, Any]] = []
    for ch in chapters_data:
        ch_num = ch['chapter_number']
        anchors = loose_anchor_names  # use full cross-chapter anchor list
        resolved_text = resolve_coreferences(ch['content'], anchors)
        resolved_chapters.append({**ch, 'content': resolved_text})
        print(f"  Chapter {ch_num}: resolved")

    # Rebuild enriched list using resolved text
    enriched_resolved = [
        {**ch, 'candidate_names': candidate_names_per_chapter.get(ch['chapter_number'], [])}
        for ch in resolved_chapters
    ]

    # ── Pass 2b: Final scoring on resolved text ──────────────────────────────
    print("\n" + "=" * 60)
    print("Pre-Processing Pass 2b: Final Character Scoring (resolved text)")
    print("=" * 60)

    validated_characters = score_characters(enriched_resolved)
    validated_names = list(validated_characters.keys())

    print(f"Validated characters after coref (score ≥ 3): {len(validated_names)}")
    newly_retained = set(validated_names) - set(raw_scores.keys())
    if newly_retained:
        print(f"  ↳ {len(newly_retained)} character(s) retained thanks to coref resolution:")
        for name in sorted(newly_retained):
            print(f"      + {name}")

    if validated_characters:
        print("\nTop characters by score:")
        for name, score in sorted(validated_characters.items(), key=lambda x: -x[1])[:20]:
            print(f"  {name:<30} score={score}")

    print("=" * 60 + "\n")

    # ── Pass 3: Sentence-level chunk creation ────────────────────────────────
    print("=" * 60)
    print("Pre-Processing Pass 3: Sentence Chunk Creation")
    print("=" * 60)

    chunks_by_chapter: Dict[int, List[Dict[str, Any]]] = {}
    total_chunks = 0

    for ch in resolved_chapters:
        ch_num = ch['chapter_number']
        chunks = create_sentence_chunks(ch_num, ch['content'], validated_names)
        chunks_by_chapter[ch_num] = chunks
        total_chunks += len(chunks)

    print(f"Total sentence chunks created: {total_chunks}")
    print("=" * 60 + "\n")

    return validated_characters, chunks_by_chapter


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _score_with_threshold(
    chapters_data: List[Dict[str, Any]],
    threshold: int,
) -> List[str]:
    """
    Re-run the scoring logic and return names at or above `threshold`.
    Used internally to generate a loose anchor list for the coref resolver
    without exposing a threshold parameter on the public score_characters API.

    This duplicates the scoring logic from score_characters to avoid
    coupling the public function signature to internal pipeline needs.
    """
    mention_counts: Dict[str, int] = defaultdict(int)
    chapter_sets: Dict[str, set] = defaultdict(set)
    is_nsubj: Dict[str, bool] = defaultdict(bool)
    is_actor: Dict[str, bool] = defaultdict(bool)
    in_attribution: Dict[str, bool] = defaultdict(bool)

    for chapter in chapters_data:
        chapter_num = chapter['chapter_number']
        content = chapter['content']
        candidate_names: List[str] = chapter.get('candidate_names', [])
        if not candidate_names:
            continue

        candidate_lower = {n.lower(): n for n in candidate_names}
        doc = nlp(content)

        for sent in doc.sents:
            has_attribution = any(
                token.lemma_.lower() in ATTRIBUTION_VERBS for token in sent
            )
            for token in sent:
                tl = token.text.lower()
                if tl in candidate_lower:
                    canonical = candidate_lower[tl]
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
                if len(ent) > 1:
                    mention_counts[canonical] += 1
                    chapter_sets[canonical].add(chapter_num)
                root = ent.root
                dep = root.dep_
                if dep == 'nsubj':
                    is_nsubj[canonical] = True
                    is_actor[canonical] = True
                elif dep in ('dobj', 'nsubjpass', 'obj', 'iobj'):
                    is_actor[canonical] = True
                if has_attribution:
                    in_attribution[canonical] = True

    result = []
    for name in set(mention_counts.keys()):
        count = mention_counts[name]
        score = 0
        if count >= 3:
            score += 2
        elif count == 1:
            score -= 2
        if is_nsubj[name]:
            score += 2
        if in_attribution[name]:
            score += 1
        if len(chapter_sets[name]) >= 2:
            score += 1
        if not is_actor[name]:
            score -= 2
        if score >= threshold:
            result.append(name)

    return result
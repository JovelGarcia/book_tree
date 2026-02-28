"""
post_processing.py

Post-processing step to consolidate duplicate/conflicting Relationship records
for the same character pair into a single, best-supported record.

The LLM analyses many text chunks independently, so it's expected to emit
the same pair multiple times under different relationship_type labels.
This module resolves that by:

  1. Grouping all Relationship rows by (epub, char_A, char_B) — normalised so
     the lower-id character is always char_1 (matching the convention in
     extract_relationships_with_llm).
  2. Collecting every piece of evidence across all rows for that pair.
  3. Choosing the winning relationship_type via a two-pass scoring strategy:
       a. Weighted confidence sum per type.
       b. A priority hierarchy that breaks ties for semantically dominant
          types (e.g. "romantic" beats "other", "family" beats "other").
  4. Rewriting the DB: keeping one row with all merged evidence and deleting
     the rest.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from django.db import transaction

from .models import Character, EpubFile, Relationship  # adjust import path as needed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Higher number = higher priority when confidence scores are tied.
# Adjust to taste; the key insight is that specific relationship types should
# beat "other" and "unspecified" catch-all buckets.
RELATIONSHIP_TYPE_PRIORITY: dict[str, int] = {
    "romantic":       100,
    "master_servant":  90,
    "family":          80,
    "mentor":          70,
    "enemy":           60,
    "ally":            50,
    "friend":          40,
    "other":           10,
}

# If no relationship_type in the data matches the table above, assign this
# default priority so it still participates in scoring.
_DEFAULT_PRIORITY = 5

# Minimum confidence threshold: evidence entries below this are still kept
# for the record but do not contribute to the type-scoring vote.
MIN_CONFIDENCE_FOR_VOTE = 0.0   # set to e.g. 0.5 to be stricter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def consolidate_relationships(epub_id: int) -> dict[str, Any]:
    """
    Consolidate all Relationship rows for epub_id into one canonical row per
    character pair.

    Returns a stats dict:
        - pairs_processed:  number of unique (char_A, char_B) pairs examined
        - rows_before:      total Relationship rows before consolidation
        - rows_after:       total Relationship rows after consolidation
        - rows_deleted:     rows removed
        - conflicts_resolved: pairs where >1 relationship_type was present
    """
    epub = EpubFile.objects.get(id=epub_id)
    all_rels = list(Relationship.objects.filter(epub=epub).select_related("character_1", "character_2"))

    rows_before = len(all_rels)

    # ------------------------------------------------------------------
    # Step 1 — group rows by normalised character pair
    # ------------------------------------------------------------------
    # Key: (min_id, max_id) so direction doesn't matter
    groups: dict[tuple[int, int], list[Relationship]] = defaultdict(list)
    for rel in all_rels:
        key = (
            min(rel.character_1_id, rel.character_2_id),
            max(rel.character_1_id, rel.character_2_id),
        )
        groups[key].append(rel)

    pairs_processed = len(groups)
    conflicts_resolved = 0
    rows_deleted = 0

    with transaction.atomic():
        for (id_a, id_b), rels in groups.items():
            if len(rels) == 1:
                # Nothing to do — already a single row for this pair
                continue

            # --------------------------------------------------------------
            # Step 2 — collect all evidence across every row for this pair
            # --------------------------------------------------------------
            all_evidence: list[dict] = []
            seen_evidence_keys: set[tuple] = set()

            for rel in rels:
                for ev in (rel.evidence or []):
                    # Deduplicate evidence by (chapter, specific_type, first 80
                    # chars of the evidence text) — same chunk may appear in
                    # multiple rows.
                    dedup_key = (
                        ev.get("chapter"),
                        ev.get("specific_type"),
                        (ev.get("evidence") or "")[:80],
                    )
                    if dedup_key not in seen_evidence_keys:
                        seen_evidence_keys.add(dedup_key)
                        all_evidence.append(ev)

            # --------------------------------------------------------------
            # Step 3 — vote for the best relationship_type
            # --------------------------------------------------------------
            type_scores: dict[str, float] = defaultdict(float)
            type_counts: dict[str, int] = defaultdict(int)

            for rel in rels:
                rtype = rel.relationship_type
                # rel.confidence is the average confidence of its evidence
                type_scores[rtype] += rel.confidence
                type_counts[rtype] += 1

            # Also score directly from individual evidence entries so that
            # a type backed by many high-confidence snippets wins over one
            # backed by a single high-confidence row-level average.
            for ev in all_evidence:
                rtype = ev.get("relationship_type")  # may or may not be present
                if rtype and ev.get("confidence", 0) >= MIN_CONFIDENCE_FOR_VOTE:
                    type_scores[rtype] += ev.get("confidence", 0) * 0.25  # fractional bonus

            unique_types = set(type_scores.keys())
            if len(unique_types) > 1:
                conflicts_resolved += 1

            winning_type = _pick_winning_type(type_scores)

            # --------------------------------------------------------------
            # Step 4 — compute merged confidence
            # --------------------------------------------------------------
            # Recalculate as mean of all individual evidence confidences so
            # it doesn't artificially inflate from row averaging.
            confidences = [ev.get("confidence", 0.7) for ev in all_evidence if "confidence" in ev]
            merged_confidence = (sum(confidences) / len(confidences)) if confidences else 0.7
            merged_confidence = min(0.95, merged_confidence)

            # --------------------------------------------------------------
            # Step 5 — keep the best row, delete the rest
            # --------------------------------------------------------------
            # Prefer to keep a row that already has the winning type (avoids
            # unnecessary write if it already exists).
            winner_rows = [r for r in rels if r.relationship_type == winning_type]
            keeper = winner_rows[0] if winner_rows else rels[0]

            keeper.relationship_type = winning_type
            keeper.evidence = all_evidence
            keeper.confidence = merged_confidence

            # Ensure character ordering is canonical (lower id = char_1)
            char_a = Character.objects.get(id=id_a)
            char_b = Character.objects.get(id=id_b)
            keeper.character_1 = char_a
            keeper.character_2 = char_b

            keeper.save()

            for rel in rels:
                if rel.pk != keeper.pk:
                    rel.delete()
                    rows_deleted += 1

    rows_after = Relationship.objects.filter(epub=epub).count()

    stats = {
        "pairs_processed": pairs_processed,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_deleted": rows_deleted,
        "conflicts_resolved": conflicts_resolved,
    }

    _print_stats(stats)
    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_winning_type(type_scores: dict[str, float]) -> str:
    """
    Given a mapping of relationship_type → accumulated score, return the
    type that should be considered canonical for the pair.

    Strategy:
      1. Find the maximum score.
      2. Among all types tied at that score, pick the one with the highest
         priority in RELATIONSHIP_TYPE_PRIORITY.
    """
    if not type_scores:
        return "other"

    max_score = max(type_scores.values())
    # Collect all types within a small tolerance of the max score
    # (floating-point sums may not be exactly equal)
    tolerance = max_score * 0.05
    top_types = [t for t, s in type_scores.items() if s >= max_score - tolerance]

    # Break ties by priority
    top_types.sort(
        key=lambda t: RELATIONSHIP_TYPE_PRIORITY.get(t, _DEFAULT_PRIORITY),
        reverse=True,
    )
    return top_types[0]


def _print_stats(stats: dict[str, Any]) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print("Relationship Consolidation Results")
    print(f"{'=' * width}")
    print(f"  Unique character pairs examined : {stats['pairs_processed']}")
    print(f"  Relationship rows before        : {stats['rows_before']}")
    print(f"  Relationship rows after         : {stats['rows_after']}")
    print(f"  Duplicate rows deleted          : {stats['rows_deleted']}")
    print(f"  Conflicting types resolved      : {stats['conflicts_resolved']}")
    print(f"{'=' * width}\n")
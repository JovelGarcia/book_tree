# compare_models.py
# Run from your Django project root: python compare_models.py

import os
import sys
import django
import spacy
from collections import Counter, defaultdict

# Add the parent directory so Python can find the book_tree package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'book_tree.settings')
django.setup()

from book_trees.models import EpubFile, Chapter
from book_trees.processing import is_likely_character


# ── Load both models ─────────────────────────────────────────────────────────
print("Loading models...")
nlp_trf = spacy.load("en_core_web_trf")
nlp_custom = spacy.load("book_trees/output/model-best")
print("✓ Both models loaded\n")

EPUB_IDS = [15, 17, 19, 23]

def get_entities_from_chapters(nlp, epub_id, label="PERSON"):
    """Run NER over all chapters and return raw + filtered entity counters."""
    chapters = Chapter.objects.filter(epub_id=epub_id).order_by('chapter_number')
    raw = Counter()
    passed = Counter()

    for chapter in chapters:
        doc = nlp(chapter.content)
        for ent in doc.ents:
            if ent.label_ != label:
                continue
            name = ent.text.strip()
            if name.endswith("'s"):
                name = name[:-2]
            raw[name] += 1
            if is_likely_character(name, ent):
                passed[name] += 1

    return raw, passed


# ── Run comparison per EPUB ───────────────────────────────────────────────────
for epub_id in EPUB_IDS:
    try:
        epub = EpubFile.objects.get(id=epub_id)
    except EpubFile.DoesNotExist:
        print(f"EPUB {epub_id} not found, skipping.\n")
        continue

    print(f"{'='*60}")
    print(f"EPUB {epub_id}: {epub.original_filename}")
    print(f"{'='*60}")

    trf_raw, trf_passed   = get_entities_from_chapters(nlp_trf,    epub_id)
    cust_raw, cust_passed = get_entities_from_chapters(nlp_custom,  epub_id)

    # ── Summary stats ────────────────────────────────────────────────────────
    trf_filter_rate  = (len(trf_raw)  - len(trf_passed))  / max(len(trf_raw), 1)  * 100
    cust_filter_rate = (len(cust_raw) - len(cust_passed)) / max(len(cust_raw), 1) * 100

    print(f"\n{'Model':<20} {'Raw entities':>12} {'After filter':>14} {'Filtered out':>14} {'Filter rate':>12}")
    print("-" * 74)
    print(f"{'en_core_web_trf':<20} {len(trf_raw):>12} {len(trf_passed):>14} {len(trf_raw)-len(trf_passed):>14} {trf_filter_rate:>11.1f}%")
    print(f"{'model-best':<20} {len(cust_raw):>12} {len(cust_passed):>14} {len(cust_raw)-len(cust_passed):>14} {cust_filter_rate:>11.1f}%")

    # ── What each model uniquely finds (post-filter) ─────────────────────────
    only_trf    = set(trf_passed)  - set(cust_passed)
    only_custom = set(cust_passed) - set(trf_passed)
    in_both     = set(trf_passed)  & set(cust_passed)

    print(f"\n  Shared (both models):     {len(in_both)}")
    print(f"  Only TRF finds:           {len(only_trf)}")
    print(f"  Only model-best finds:    {len(only_custom)}")

    if only_trf:
        print(f"\n  ⚠ TRF-only (possible false positives):")
        for name in sorted(only_trf)[:20]:  # cap at 20 to avoid wall of text
            print(f"    - {name!r}  (seen {trf_raw[name]}x)")

    if only_custom:
        print(f"\n  ✓ model-best only (genre-specific catches):")
        for name in sorted(only_custom)[:20]:
            print(f"    + {name!r}  (seen {cust_raw[name]}x)")

    print()
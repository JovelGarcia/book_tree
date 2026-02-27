import re
from collections import defaultdict
from typing import Optional
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from .models import EpubFile, Chapter, Section


# ==============================================================================
# LANDMARK KEYWORD DEFINITIONS
# ==============================================================================

# Section dividers — short title pages that split the book into parts/books/acts.
# Detected inside the book-content zone; they become Section objects, not chapters.
SECTION_TITLE_PATTERNS = [
    r'^book\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)$',
    r'^part\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)$',
    r'^act\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)$',
    r'^volume\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)$',
    r'^section\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)$',
    r'^book\s+[ivxlcdm]+$',
    r'^part\s+[ivxlcdm]+$',
    r'^act\s+[ivxlcdm]+$',
]

FRONT_MATTER_LANDMARKS = {
    'cover':        ['cover'],
    'title_page':   ['title page', 'half title'],
    'books_by':     ['books by', 'also by', 'works by'],
    'toc':          ['table of contents', 'contents'],
    'copyright':    ['copyright'],
    'dedication':   ['dedication'],
    'foreword':     ['foreword'],
    'preface':      ['preface'],
    'introduction': ['introduction'],
    'epigraph':     ['epigraph'],
    'prologue':     ['prologue'],
}

BACK_MATTER_LANDMARKS = {
    'epilogue':        ['epilogue'],
    'afterword':       ['afterword'],
    'appendix':        ['appendix', 'appendixes', 'appendices'],
    'acknowledgment':  ['acknowledgment', 'acknowledgements', 'acknowledgement'],
    'glossary':        ['glossary'],
    'notes':           ['notes', 'endnotes', 'footnotes'],
    'bibliography':    ['bibliography', 'references', 'works cited'],
    'index':           ['index'],
    'about_author':    ['about the author', 'about the authors', 'author bio'],
    'also_by':         ['also by', 'books by', 'works by'],
    'maps':            ['maps', 'map of'],
    'about_publisher': ['about the publisher', 'about publisher'],
    'copyright':       ['copyright page', 'copyright ©', 'all rights reserved'],
}

# Landmarks that can legitimately appear in either front OR back matter.
# Position in the book (front half vs back half) determines classification.
AMBIGUOUS_LANDMARKS = {'about_author', 'also_by'}

# How far into a document (as a fraction of total leaf elements) a landmark
# keyword must appear to be considered a true structural heading.
# Keywords appearing deeper in the document are treated as narrative prose.
LANDMARK_HEADING_DEPTH_LIMIT = 0.25


# ==============================================================================
# LANDMARK HELPERS
# ==============================================================================

def _text_matches_keywords(text: str, keyword_list: list) -> bool:
    """Return True if any keyword appears in the lowercased text."""
    t = text.lower().strip()
    return any(kw in t for kw in keyword_list)


def _is_section_title_doc(soup: BeautifulSoup) -> Optional[str]:
    """
    Return a normalised section title (e.g. 'Book One', 'Part II') if this
    document is a section-divider title page, else None.

    Checks headings first; falls back to short leaf-node text for docs with
    no heading tags (common in calibre-converted EPUBs).
    """
    candidates = []

    headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
    if headings:
        candidates = [h.get_text().strip() for h in headings]
    else:
        # Short doc with no headings — check all leaf nodes
        text = soup.get_text(separator=' ').strip()
        if len(text) < 200:
            for tag in soup.find_all(True):
                if tag.find(True):
                    continue
                t = tag.get_text().strip()
                if t:
                    candidates.append(t)

    for candidate in candidates:
        for pattern in SECTION_TITLE_PATTERNS:
            if re.match(pattern, candidate.lower()):
                return candidate

    return None


def _doc_landmark_type(soup: BeautifulSoup, landmark_map: dict,
                       depth_limit: float = LANDMARK_HEADING_DEPTH_LIMIT) -> Optional[str]:
    """
    Check soup against a landmark map, return the matching key or None.

    Heading tags (h1–h4) are always checked regardless of position — they are
    explicitly structural.  Other leaf nodes (div/p/span) are only checked
    within the first `depth_limit` fraction of all leaf elements, so a keyword
    buried deep in chapter prose never triggers a false landmark.
    """
    # Priority 1: explicit heading tags — position doesn't matter
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        text = tag.get_text().strip()
        for key, keywords in landmark_map.items():
            if _text_matches_keywords(text, keywords):
                return key

    # Priority 2: short prominent leaf nodes, top fraction of document only
    all_leaves = [
        tag for tag in soup.find_all(['div', 'p', 'span'])
        if not tag.find(True)
    ]
    limit = max(1, int(len(all_leaves) * depth_limit))

    for tag in all_leaves[:limit]:
        text = tag.get_text().strip()
        if not text or len(text) > 80:
            continue
        for key, keywords in landmark_map.items():
            if _text_matches_keywords(text, keywords):
                return key

    return None


def identify_landmarks(all_docs: list) -> list:
    """
    Annotate each doc in all_docs with:
      'landmark'      — None | landmark-type string (e.g. 'toc', 'appendix')
      'section_title' — None | section title string (e.g. 'Book One')

    Front matter: scan forward, advancing the front-matter boundary only for
    landmark docs and short/empty gap docs.  The FIRST substantial
    (≥300 chars) non-landmark doc ends the front-matter zone.

    Back matter: scan backward from the end with the same logic.

    Section titles: detected on every non-landmark doc inside the book-content
    zone that looks like a short section-divider title page.
    """
    n = len(all_docs)

    for doc in all_docs:
        doc['landmark']      = None
        doc['section_title'] = None

    # ---- Forward scan: front matter ----------------------------------------
    last_front_idx = -1
    for i in range(n):
        doc  = all_docs[i]
        text = doc['soup'].get_text().strip()

        lm = _doc_landmark_type(doc['soup'], FRONT_MATTER_LANDMARKS)
        if lm is None:
            lm = _doc_landmark_type(doc['soup'], {
                k: v for k, v in BACK_MATTER_LANDMARKS.items()
                if k in AMBIGUOUS_LANDMARKS
            })

        if lm:
            doc['landmark'] = lm
            last_front_idx  = i
        elif len(text) >= 300:
            break   # Substantial non-landmark content — front matter ends here
        # Short/empty doc — don't advance boundary but keep scanning

    # ---- Backward scan: back matter ----------------------------------------
    first_back_idx = n
    for i in range(n - 1, -1, -1):
        doc  = all_docs[i]
        text = doc['soup'].get_text().strip()

        if doc['landmark'] is not None:
            continue  # Already tagged by forward pass

        lm = _doc_landmark_type(doc['soup'], BACK_MATTER_LANDMARKS)
        if lm:
            doc['landmark'] = lm
            first_back_idx  = i
        elif len(text) >= 300:
            # Peek forward: if the doc immediately after this one is already
            # a confirmed back-matter landmark, keep scanning past this
            # chapter — it's the last chapter before the back matter.
            next_is_back = (
                i + 1 < n and all_docs[i + 1]['landmark'] is not None
                and all_docs[i + 1]['landmark'] in set(BACK_MATTER_LANDMARKS.keys()) | {'separator', 'back_matter_content'}
            )
            if not next_is_back:
                break
        # Short/empty doc — keep scanning

    # ---- Gap-fill: short docs sandwiched between back-matter docs -----------
    for i in range(1, n - 1):
        if (all_docs[i]['landmark'] is None
                and all_docs[i - 1]['landmark'] is not None
                and all_docs[i + 1]['landmark'] is not None
                and i >= first_back_idx - 2):
            text = all_docs[i]['soup'].get_text().strip()
            if len(text) < 100:
                all_docs[i]['landmark'] = 'separator'

    # ---- Forward scan: back-matter landmarks embedded in book content --------
    # The backward scan only works outward from the end of the spine, so it
    # misses back-matter section headers (e.g. "APPENDIXES") that sit in the
    # middle of the spine surrounded by substantive chapter content.  Here we
    # do one forward pass through the tentative book-content zone and check
    # every short doc (< 200 chars) for back-matter keywords.  When found, we
    # tag it AND every document after it as back matter, then update
    # first_back_idx so the section-title detection below uses the tighter zone.
    _tentative_book_start = last_front_idx + 1
    _tentative_book_end   = (first_back_idx - 1) if first_back_idx < n else (n - 1)

    for i in range(_tentative_book_start, _tentative_book_end + 1):
        doc  = all_docs[i]
        if doc['landmark'] is not None:
            continue
        text = doc['soup'].get_text().strip()
        if len(text) >= 200:
            continue  # Too long to be a pure section-header; skip
        # Use depth_limit=1.0 so even the very last leaf element is checked —
        # short docs often have their only text node deep in a nested <span>.
        lm = _doc_landmark_type(doc['soup'], BACK_MATTER_LANDMARKS, depth_limit=1.0)
        if lm:
            doc['landmark'] = lm
            first_back_idx  = i
            print(f"[LANDMARK] Back-matter heading found inside book zone at "
                  f"doc[{i:03d}] {doc['item'].get_name()} — '{lm}' ({len(text)} chars); "
                  f"tagging all subsequent docs as back matter")
            # Propagate: everything after this heading is back matter too.
            for j in range(i + 1, n):
                if all_docs[j]['landmark'] is None:
                    lm_j = _doc_landmark_type(all_docs[j]['soup'],
                                              BACK_MATTER_LANDMARKS, depth_limit=1.0)
                    all_docs[j]['landmark'] = lm_j if lm_j else 'back_matter_content'
            break  # Only the first such heading matters

    # ---- Section title detection (inside book content only) ----------------
    book_start = last_front_idx + 1
    book_end   = (first_back_idx - 1) if first_back_idx < n else (n - 1)

    for i in range(book_start, book_end + 1):
        doc = all_docs[i]
        if doc['landmark'] is not None:
            continue
        st = _is_section_title_doc(doc['soup'])
        if st:
            doc['section_title'] = st
            print(f"[LANDMARK] Section title at doc[{i:03d}] "
                  f"{doc['item'].get_name()} — '{st}'")

    return all_docs


def _book_content_bounds(all_docs: list) -> tuple:
    """
    Return (start_idx, end_idx) inclusive — the real book-content zone.
    Mirrors the same skip-short-gap-docs logic used in identify_landmarks.
    """
    n = len(all_docs)

    FRONT_MATTER_TYPES = set(FRONT_MATTER_LANDMARKS.keys()) | AMBIGUOUS_LANDMARKS
    BACK_MATTER_TYPES  = set(BACK_MATTER_LANDMARKS.keys())  | {'separator', 'back_matter_content'}

    # Walk forward
    start_idx = 0
    i = 0
    while i < n:
        lm   = all_docs[i]['landmark']
        text = all_docs[i]['soup'].get_text().strip()
        if lm in FRONT_MATTER_TYPES:
            start_idx = i + 1
            i += 1
        elif len(text) < 300 and lm is None:
            i += 1   # Short gap — keep scanning without advancing boundary
        else:
            break

    # Walk backward
    end_idx = n - 1
    i = n - 1
    while i >= 0:
        lm   = all_docs[i]['landmark']
        text = all_docs[i]['soup'].get_text().strip()
        if lm in BACK_MATTER_TYPES:
            end_idx = i - 1
            i -= 1
        elif len(text) < 300 and lm is None:
            i -= 1
        else:
            break

    return start_idx, end_idx


# ==============================================================================
# SEQUENCE DETECTION  (multi-section aware)
# ==============================================================================

def _positional_sequences(candidate_docs: list, all_docs: list) -> list:
    """
    Positional fallback: when no number_candidates sequence can be found,
    use section-title boundaries (already detected in all_docs) to slice
    candidate_docs into groups and number each group 1, 2, 3, ...

    Each group spans from one section-title doc (exclusive) to the next
    section-title doc (exclusive), or to the end of the candidate list.

    Returns the same sequence-dict format as _find_chapter_sequences.
    """
    # Collect global indices of section-title docs, in order
    section_global_indices = [
        g_idx for g_idx, doc in enumerate(all_docs)
        if doc.get('section_title') is not None
    ]

    # Build a map: candidate_doc_idx → global_idx
    cand_to_global = {}
    cand_ptr = 0
    for g_idx, doc in enumerate(all_docs):
        if cand_ptr < len(candidate_docs) and doc is candidate_docs[cand_ptr]:
            cand_to_global[cand_ptr] = g_idx
            cand_ptr += 1

    global_to_cand = {v: k for k, v in cand_to_global.items()}

    if not section_global_indices:
        # No sections detected — treat the entire candidate list as one group
        substantive = [
            i for i, d in enumerate(candidate_docs)
            if len(d.get('text', '').strip()) >= 50
        ]
        if not substantive:
            return []
        print(f"[PASS2-POS] No sections — assigning {len(substantive)} chapters positionally")
        return [{
            'doc_indices':         substantive,
            'chapter_numbers':     list(range(1, len(substantive) + 1)),
            'start_candidate_idx': substantive[0],
            'end_candidate_idx':   substantive[-1],
        }]

    # Build boundary pairs: (start_global_excl, end_global_excl)
    # Each section starts just after a section-title doc and ends just before
    # the next one (or at the end of all_docs).
    boundaries = []
    for s_idx, g_start in enumerate(section_global_indices):
        g_end = section_global_indices[s_idx + 1] if s_idx + 1 < len(section_global_indices) else len(all_docs)
        boundaries.append((g_start + 1, g_end))   # exclusive on both ends

    sequences = []
    for sec_idx, (g_start, g_end) in enumerate(boundaries):
        # Find candidate docs whose global index falls within [g_start, g_end)
        group = [
            cand_idx for cand_idx, g_idx in cand_to_global.items()
            if g_start <= g_idx < g_end
        ]
        # Filter to substantive docs only (skip gap placeholders)
        group = [
            i for i in group
            if len(candidate_docs[i].get('text', '').strip()) >= 50
        ]
        if not group:
            print(f"[PASS2-POS] Section {sec_idx + 1}: no substantive docs found, skipping")
            continue

        group.sort()
        chapter_numbers = list(range(1, len(group) + 1))
        print(f"[PASS2-POS] Section {sec_idx + 1}: {len(group)} chapters assigned positionally")
        sequences.append({
            'doc_indices':         group,
            'chapter_numbers':     chapter_numbers,
            'start_candidate_idx': group[0],
            'end_candidate_idx':   group[-1],
        })

    return sequences


def _find_chapter_sequences(candidate_docs: list, all_docs: list) -> list:
    """
    Find all valid sequential chapter-number runs across candidate_docs.

    Primary method: cross-document number_candidates sequence validation.
    Fallback: positional assignment using section-title boundaries when no
    numeric sequences are found (e.g. books with no chapter numbers in HTML).

    For each depth position in number_candidates:
      - Build sequential runs (consecutive values with correct gaps).
      - Accept runs of length ≥ MIN_RUN_LENGTH (3).
      - Collect ALL non-overlapping runs at that depth.

    The depth with the greatest total doc coverage wins.

    Returns a list of sequence dicts:
      {
        'doc_indices':        [candidate_doc_idx, ...],
        'chapter_numbers':    [int, ...],
        'start_candidate_idx': int,
        'end_candidate_idx':   int,
      }
    """
    MIN_RUN_LENGTH = 3

    best_depth      = None
    best_depth_runs = []

    max_depth = max(
        (len(d.get('number_candidates', [])) for d in candidate_docs),
        default=0
    )
    print(f"[PASS2] max_depth across all candidate docs: {max_depth}")

    for depth in range(max_depth):
        values_at_depth = [
            (doc_idx, d['number_candidates'][depth])
            for doc_idx, d in enumerate(candidate_docs)
            if len(d.get('number_candidates', [])) > depth
        ]
        print(f"[PASS2] depth {depth}: values = {values_at_depth}")

        if len(values_at_depth) < MIN_RUN_LENGTH:
            print(f"[PASS2] depth {depth}: fewer than {MIN_RUN_LENGTH} docs — skipping")
            continue

        # Build all sequential runs at this depth
        runs        = []
        current_run = [values_at_depth[0]]

        for j in range(1, len(values_at_depth)):
            prev_idx, prev_val = values_at_depth[j - 1]
            curr_idx, curr_val = values_at_depth[j]
            expected = prev_val + (curr_idx - prev_idx)
            if curr_val == expected:
                current_run.append(values_at_depth[j])
            else:
                if len(current_run) >= MIN_RUN_LENGTH:
                    runs.append(current_run)
                    print(f"[PASS2] depth {depth}: accepted run len={len(current_run)} "
                          f"chapters {current_run[0][1]}–{current_run[-1][1]}")
                else:
                    print(f"[PASS2] depth {depth}: dropped short run len={len(current_run)}")
                current_run = [values_at_depth[j]]

        if len(current_run) >= MIN_RUN_LENGTH:
            runs.append(current_run)
            print(f"[PASS2] depth {depth}: accepted run len={len(current_run)} "
                  f"chapters {current_run[0][1]}–{current_run[-1][1]}")

        if not runs:
            print(f"[PASS2] depth {depth}: no runs met threshold")
            continue

        total_coverage = sum(len(r) for r in runs)
        print(f"[PASS2] depth {depth}: {len(runs)} run(s), total coverage = {total_coverage}")

        if total_coverage > sum(len(r) for r in best_depth_runs):
            best_depth      = depth
            best_depth_runs = runs

    if best_depth_runs:
        print(f"\n[PASS2] ✅ Best depth: {best_depth}, "
              f"runs: {[len(r) for r in best_depth_runs]} "
              f"(total {sum(len(r) for r in best_depth_runs)} chapters)\n")
        return [
            {
                'doc_indices':         [idx for idx, _ in run],
                'chapter_numbers':     [val for _, val in run],
                'start_candidate_idx': run[0][0],
                'end_candidate_idx':   run[-1][0],
            }
            for run in best_depth_runs
        ]

    # ---- Positional fallback ------------------------------------------------
    print("[PASS2] ❌ No numeric sequences found — falling back to positional assignment")
    return _positional_sequences(candidate_docs, all_docs)


# ==============================================================================
# SECTION ASSIGNMENT
# ==============================================================================

def _assign_sections(candidate_docs: list, all_docs: list,
                     sequences: list) -> list:
    """
    Map each sequence to a Section by looking for section-title docs that
    precede it in the global all_docs order.

    Returns a list of section dicts:
      {
        'title': str,
        'order': int,
        'sequence': <sequence dict>
      }
    """
    if not sequences:
        return []

    if len(sequences) == 1:
        return [{'title': '', 'order': 1, 'sequence': sequences[0]}]

    # Map candidate_doc_idx → global all_docs index via object identity
    candidate_to_global = {}
    cand_ptr = 0
    for g_idx, doc in enumerate(all_docs):
        if cand_ptr < len(candidate_docs) and doc is candidate_docs[cand_ptr]:
            candidate_to_global[cand_ptr] = g_idx
            cand_ptr += 1

    seq_global_starts = [
        candidate_to_global.get(seq['start_candidate_idx'],
                                seq['start_candidate_idx'])
        for seq in sequences
    ]

    section_results = []
    for s_idx, (seq, g_start) in enumerate(zip(sequences, seq_global_starts)):
        title = None
        prev_g_start = seq_global_starts[s_idx - 1] if s_idx > 0 else 0

        for g_idx in range(g_start - 1, prev_g_start - 1, -1):
            st = all_docs[g_idx].get('section_title')
            if st:
                title = st
                break

        if title is None:
            title = f'Part {s_idx + 1}'
            print(f"[SECTIONS] No section title found before sequence {s_idx + 1} "
                  f"— auto title '{title}'")

        section_results.append({
            'title':    title,
            'order':    s_idx + 1,
            'sequence': seq,
        })
        print(f"[SECTIONS] Section {s_idx + 1}: '{title}' — "
              f"{len(seq['doc_indices'])} chapters "
              f"({seq['chapter_numbers'][0]}–{seq['chapter_numbers'][-1]})")

    return section_results


# ==============================================================================
# CHAPTER EXTRACTION
# ==============================================================================

def extract_chapters_from_epub(epub_path: str) -> dict:
    """
    Extract chapters and section structure from an EPUB file.

    Returns:
      {
        'sections': [
          {
            'title':    str,   # '' means no sections (flat book)
            'order':    int,
            'chapters': [
              {
                'chapter_number': int,
                'title':   str,
                'content': str,
                'filename': str,
              }, ...
            ]
          }, ...
        ]
      }

    Phases
    ------
    Phase 0 : Parse all documents.
    Phase 1 : Landmark + section-title identification.
    Pass  1 : Filter to candidate chapter documents (inside book-content zone,
              not landmarks, not section titles, no non-chapter filename).
    Pass  2 : Multi-run sequence detection — finds all sequential chapter runs
              including restarts for multi-section books.
    Pass  3 : Assign chapter numbers (explicit methods first, sequence fallback).
    Pass  4 : Group by section; deduplicate within each section.
    """
    book = epub.read_epub(epub_path)

    non_chapter_filename_keywords = [
        'acknowledgment', 'acknowledgement', 'copyright', 'dedication',
        'foreword', 'preface', 'prologue', 'epilogue', 'afterword',
        'appendix', 'glossary', 'toc', 'cover', 'titlepage', 'title_page',
        'halftitle', 'half_title', 'also_by', 'about_author',
    ]

    # =========================================================================
    # PHASE 0 — Parse every document.
    # =========================================================================
    all_docs = []
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        content = item.get_content()
        soup    = BeautifulSoup(content, 'html.parser')
        text    = soup.get_text(separator="\n").strip()
        all_docs.append({
            'item':          item,
            'soup':          soup,
            'text':          text,
            'landmark':      None,
            'section_title': None,
        })

    print(f"\n[PHASE 0] {len(all_docs)} total documents parsed")

    # =========================================================================
    # PHASE 1 — Landmark + section-title identification.
    # =========================================================================
    all_docs = identify_landmarks(all_docs)

    print("\n[PHASE 1] Landmark summary:")
    any_found = False
    for i, doc in enumerate(all_docs):
        lm = doc['landmark']
        st = doc['section_title']
        if lm:
            print(f"  [{i:03d}] {doc['item'].get_name():<45}  landmark={lm}")
            any_found = True
        if st:
            print(f"  [{i:03d}] {doc['item'].get_name():<45}  section_title='{st}'")
            any_found = True
    if not any_found:
        print("  (none detected)")

    book_start, book_end = _book_content_bounds(all_docs)
    print(f"\n[PHASE 1] Book content zone : doc[{book_start}] → doc[{book_end}]  "
          f"({max(0, book_end - book_start + 1)} documents)")
    print(f"          Front matter      : doc[0] → doc[{book_start - 1}]")
    print(f"          Back matter       : doc[{book_end + 1}] → doc[{len(all_docs) - 1}]\n")

    # =========================================================================
    # PASS 1 — Build candidate_docs list.
    # =========================================================================
    candidate_docs = []

    for global_idx, doc in enumerate(all_docs):
        item           = doc['item']
        soup           = doc['soup']
        text           = doc['text']
        filename_lower = item.get_name().lower()

        if global_idx < book_start or global_idx > book_end:
            print(f"[PASS1] {item.get_name()} — outside book-content zone, skipping")
            continue

        if doc['landmark'] is not None:
            print(f"[PASS1] {item.get_name()} — landmark '{doc['landmark']}', skipping")
            continue

        if doc['section_title'] is not None:
            print(f"[PASS1] {item.get_name()} — section title '{doc['section_title']}', skipping")
            continue

        if not text or len(text.strip()) < 50:
            print(f"[PASS1] {item.get_name()} — only {len(text.strip())} chars, "
                  f"keeping as gap placeholder")

        # Filename-only keyword filter (heading keywords now handled by the
        # depth-limited landmark detector, not here)
        if any(kw in filename_lower for kw in non_chapter_filename_keywords):
            print(f"[PASS1] {item.get_name()} — filename matched non-chapter keyword, skipping")
            continue

        # Collect standalone numbers from the first 50 leaf elements
        number_candidates = []
        for element in soup.find_all(True)[:50]:
            if element.find(True) is not None:
                continue
            element_text = element.get_text().strip()
            if re.match(r'^\d{1,3}$', element_text):
                number_candidates.append(int(element_text))

        print(f"[PASS1] {item.get_name()} — number_candidates: {number_candidates}")

        doc['number_candidates'] = number_candidates
        candidate_docs.append(doc)

    print(f"\n[PASS1 SUMMARY] {len(candidate_docs)} candidate documents collected\n")

    # =========================================================================
    # PASS 2 — Multi-run sequence detection.
    # =========================================================================
    sequences = _find_chapter_sequences(candidate_docs, all_docs)

    # chapter_map: candidate_doc_idx → (chapter_number, seq_idx)
    chapter_map = {}
    for seq_idx, seq in enumerate(sequences):
        for cand_idx, ch_num in zip(seq['doc_indices'], seq['chapter_numbers']):
            chapter_map[cand_idx] = (ch_num, seq_idx)

    # =========================================================================
    # PASS 3 — Assign chapter numbers.
    # =========================================================================
    raw_chapters = []  # (seq_idx, chapter_dict)

    for cand_idx, doc in enumerate(candidate_docs):
        item           = doc['item']
        soup           = doc['soup']
        text           = doc['text']
        filename_lower = item.get_name().lower()
        chapter_number = None
        seq_idx        = None
        title          = item.get_name()

        # Method 1: Filename patterns
        if 'chapter' in filename_lower or 'ch' in filename_lower:
            for pattern in [r'chapter[_\s-]*(\d+)', r'ch[_\s-]*(\d+)', r'part0*(\d+)']:
                match = re.search(pattern, filename_lower)
                if match:
                    chapter_number = int(match.group(1))
                    break

        # Method 2: HTML id attributes
        if chapter_number is None:
            for element in soup.find_all(
                ['h1', 'h2', 'h3', 'div', 'section'],
                id=re.compile(r'^c0*\d+$|^chapter0*\d+$', re.I)
            ):
                id_str = element.get('id', '')
                for pattern in [r'^c0*(\d+)$', r'^chapter0*(\d+)$']:
                    id_match = re.search(pattern, id_str, re.I)
                    if id_match:
                        chapter_number = int(id_match.group(1))
                        break
                if chapter_number is not None:
                    break

        # Method 3: HTML class attributes (strict match only)
        if chapter_number is None:
            for element in soup.find_all(
                ['h1', 'h2', 'h3', 'div', 'section'],
                class_=re.compile(r'^chapter[_\s-]+\d+$', re.I)
            ):
                class_str   = ' '.join(element.get('class', []))
                class_match = re.search(r'^chapter[_\s-]+(\d+)$', class_str, re.I)
                if class_match:
                    chapter_number = int(class_match.group(1))
                    break

        # Method 4: Cross-document sequence validation
        if cand_idx in chapter_map:
            seq_ch_num, seq_idx = chapter_map[cand_idx]
            if chapter_number is None:
                chapter_number = seq_ch_num
            # Always take seq_idx from the map for correct section grouping

        if chapter_number is None:
            continue

        # Extract title from first heading
        heading = soup.find(['h1', 'h2', 'h3'])
        if heading:
            heading_text = heading.get_text().strip()
            if heading_text and len(heading_text) < 100:
                title = heading_text

        raw_chapters.append((seq_idx, {
            'chapter_number': chapter_number,
            'title':          title,
            'content':        text,
            'filename':       item.get_name(),
        }))

    # =========================================================================
    # PASS 4 — Assign sections and deduplicate.
    # =========================================================================
    section_defs   = _assign_sections(candidate_docs, all_docs, sequences)
    seq_to_section = {s['order'] - 1: s for s in section_defs}

    chapters_by_seq = defaultdict(list)
    for seq_idx, ch_dict in raw_chapters:
        key = seq_idx if seq_idx is not None else 0
        chapters_by_seq[key].append(ch_dict)

    output_sections = []
    for seq_idx in sorted(chapters_by_seq.keys()):
        s_def = seq_to_section.get(seq_idx, {
            'title': '' if len(sequences) <= 1 else f'Part {seq_idx + 1}',
            'order': seq_idx + 1,
        })

        seen   = set()
        deduped = []
        for ch in sorted(chapters_by_seq[seq_idx], key=lambda x: x['chapter_number']):
            if ch['chapter_number'] in seen:
                print(f"⚠️  Duplicate ch {ch['chapter_number']} in '{s_def['title']}' "
                      f"— skipping {ch['filename']}")
                continue
            seen.add(ch['chapter_number'])
            deduped.append(ch)

        output_sections.append({
            'title':    s_def['title'],
            'order':    s_def['order'],
            'chapters': deduped,
        })
        print(f"[OUTPUT] Section '{s_def['title']}': {len(deduped)} chapters")

    return {'sections': output_sections}


# ==============================================================================
# EPUB FILE PROCESSING (Django integration)
# ==============================================================================

def process_epub_file(epub_id):
    epub_obj = None
    try:
        epub_obj        = EpubFile.objects.get(id=epub_id)
        epub_obj.status = 'pr'
        epub_obj.save()

        result        = extract_chapters_from_epub(epub_obj.file.path)
        sections_data = result['sections']

        for section_data in sections_data:
            # Only create a Section row when there is a real title.
            # A flat book (single section, empty title) stores chapters with
            # section=None, matching the null=True, blank=True on Chapter.section.
            section_obj = None
            if section_data['title']:
                section_obj = Section.objects.create(
                    epub=epub_obj,
                    title=section_data['title'],
                    order=section_data['order'],
                )

            for chapter_data in section_data['chapters']:
                Chapter.objects.create(
                    epub=epub_obj,
                    section=section_obj,
                    title=chapter_data['title'],
                    content=chapter_data['content'],
                    chapter_number=chapter_data['chapter_number'],
                )

        epub_obj.status    = 'c'
        epub_obj.processed = True
        epub_obj.save()

        return True

    except Exception as e:
        if epub_obj:
            epub_obj.status        = 'f'
            epub_obj.error_message = str(e)
            epub_obj.save()
        raise
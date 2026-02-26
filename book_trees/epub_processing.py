import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from .models import EpubFile, Chapter


# ==============================================================================
# LANDMARK KEYWORD DEFINITIONS
# ==============================================================================

FRONT_MATTER_LANDMARKS = {
    'cover':        ['cover'],
    'title_page':   ['title page', 'half title'],
    'books_by':     ['books by', 'also by'],
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
    'epilogue':       ['epilogue'],
    'afterword':      ['afterword'],
    'appendix':       ['appendix', 'appendixes', 'appendices'],
    'acknowledgment': ['acknowledgment', 'acknowledgements', 'acknowledgement'],
    'glossary':       ['glossary'],
    'notes':          ['notes', 'endnotes', 'footnotes'],
    'bibliography':   ['bibliography', 'references', 'works cited'],
    'index':          ['index'],
    'about_author':   ['about the author', 'about the authors', 'author bio'],
    'also_by':        ['also by', 'books by'],
}

# Landmarks that can legitimately appear in either front OR back matter
AMBIGUOUS_LANDMARKS = {'about_author', 'also_by'}


# ==============================================================================
# LANDMARK HELPERS
# ==============================================================================

def _text_matches_keywords(text: str, keyword_list: list) -> bool:
    """Return True if any keyword appears in the lowercased text."""
    t = text.lower().strip()
    return any(kw in t for kw in keyword_list)


def _doc_landmark_type(soup: BeautifulSoup, landmark_map: dict) -> str | None:
    """
    Check soup against a landmark map and return the first matching landmark
    key, or None.

    Checks (in priority order):
      1. h1/h2/h3/h4 heading text
      2. Short, prominent leaf-node text (bold spans, standalone divs/p)
    """
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        text = tag.get_text(strip=True)
        for key, keywords in landmark_map.items():
            if _text_matches_keywords(text, keywords):
                return key

    # Short leaf nodes that look like section titles
    for tag in soup.find_all(['div', 'p', 'span']):
        if tag.find(True):          # skip non-leaves
            continue
        text = tag.get_text(strip=True)
        if not text or len(text) > 80:
            continue
        for key, keywords in landmark_map.items():
            if _text_matches_keywords(text, keywords):
                return key

    return None


def identify_landmarks(all_docs: list) -> list:
    """
    Walk all_docs (each entry must have 'item' and 'soup') and annotate each
    with a 'landmark' key.

    Returns the same list with 'landmark' added to every entry:
      - None  → ordinary content (presumed chapter territory)
      - str   → landmark type, e.g. 'toc', 'dedication', 'appendix'

    Strategy
    --------
    Scans from the front for front-matter landmarks and from the back for
    back-matter landmarks, stopping in each direction once we cross the
    midpoint. Ambiguous landmarks (about_author, also_by) detected in the
    front half are treated as front matter; in the back half as back matter.
    """
    n = len(all_docs)
    midpoint = n // 2

    for doc in all_docs:
        doc['landmark'] = None

    # --- Front matter scan (index 0 → midpoint) ---
    for i in range(min(midpoint + 1, n)):
        doc = all_docs[i]
        lm = _doc_landmark_type(doc['soup'], FRONT_MATTER_LANDMARKS)
        if lm:
            doc['landmark'] = lm
            continue
        # Check ambiguous landmarks in the front half
        lm = _doc_landmark_type(doc['soup'], {
            k: v for k, v in BACK_MATTER_LANDMARKS.items()
            if k in AMBIGUOUS_LANDMARKS
        })
        if lm:
            doc['landmark'] = lm

    # --- Back matter scan (index n-1 → midpoint, stop at first non-match) ---
    # Scans backwards and continues as long as back-matter (or short/empty)
    # docs are found. Stops on substantial content that isn't back matter, so
    # chapter content that mentions "appendix" in passing isn't mis-classified.
    for i in range(n - 1, midpoint - 1, -1):
        doc = all_docs[i]
        if doc['landmark'] is not None:
            continue  # Already tagged by front-matter pass; leave it.

        lm = _doc_landmark_type(doc['soup'], BACK_MATTER_LANDMARKS)
        if lm:
            doc['landmark'] = lm
        else:
            text = doc['soup'].get_text(strip=True)
            if len(text) < 100:
                pass  # Short separator — handled by gap-fill below
            else:
                break  # Substantial non-back-matter content — stop scanning

    # --- Gap-fill: short docs sandwiched between back-matter docs ---
    for i in range(1, n - 1):
        if (all_docs[i]['landmark'] is None
                and all_docs[i - 1]['landmark'] is not None
                and all_docs[i + 1]['landmark'] is not None):
            text = all_docs[i]['soup'].get_text(strip=True)
            if len(text) < 100:
                all_docs[i]['landmark'] = 'separator'

    return all_docs


def _book_content_bounds(all_docs: list) -> tuple:
    """
    Given annotated all_docs, return (start_idx, end_idx) — the inclusive
    range of indices that represent 'real book content' (i.e. not landmarks).

    start_idx : first doc after the last consecutive front-matter landmark
    end_idx   : last doc before the first consecutive back-matter landmark
                (end_idx is INCLUSIVE)
    """
    n = len(all_docs)

    FRONT_MATTER_TYPES = set(FRONT_MATTER_LANDMARKS.keys()) | AMBIGUOUS_LANDMARKS
    BACK_MATTER_TYPES  = set(BACK_MATTER_LANDMARKS.keys())  | {'separator'}

    # Walk forward to find where front matter ends
    start_idx = 0
    for i in range(n):
        lm = all_docs[i]['landmark']
        if lm in FRONT_MATTER_TYPES:
            start_idx = i + 1
        elif lm is None:
            break   # First non-landmark doc ends the front-matter zone

    # Walk backward to find where back matter begins
    end_idx = n - 1
    for i in range(n - 1, -1, -1):
        lm = all_docs[i]['landmark']
        if lm in BACK_MATTER_TYPES:
            end_idx = i - 1
        elif lm is None:
            break   # First non-landmark doc (from the back) ends back-matter zone

    return start_idx, end_idx


# ==============================================================================
# CHAPTER EXTRACTION
# ==============================================================================

def extract_chapters_from_epub(epub_path: str) -> list:
    """
    Extract chapters and text from an EPUB file.

    Chapter identification methods (in priority order):
      1. Filename patterns   (chapter_01, ch01, …)
      2. HTML id attributes  (c01, chapter01, …)
      3. HTML class attributes
      4. Cross-document sequence validation of standalone numbers near the
         top of each document

    Landmark detection is performed across ALL documents first to establish
    book boundaries (front matter / chapter content / back matter). Only
    documents that fall inside the book-content zone are eligible to be
    chapters, regardless of which chapter-identification method succeeds.
    """
    book = epub.read_epub(epub_path)

    non_chapter_keywords = [
        'acknowledgment', 'acknowledgement', 'about', 'author', 'copyright',
        'dedication', 'foreword', 'preface', 'introduction', 'prologue',
        'epilogue', 'afterword', 'appendix', 'glossary', 'contents',
        'toc', 'cover', 'title', 'half', 'bio', 'also by', 'books by',
    ]

    # =========================================================================
    # PHASE 0 — Parse every document.
    #           We need the full list for landmark detection before filtering.
    # =========================================================================
    all_docs = []

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        content = item.get_content()
        soup    = BeautifulSoup(content, 'html.parser')
        text    = soup.get_text("\n", strip=True)
        all_docs.append({
            'item':     item,
            'soup':     soup,
            'text':     text,
            'landmark': None,
        })

    print(f"\n[PHASE 0] {len(all_docs)} total documents parsed")

    # =========================================================================
    # PHASE 1 — Landmark identification across the full document list.
    # =========================================================================
    all_docs = identify_landmarks(all_docs)

    print("\n[PHASE 1] Landmark summary:")
    for i, doc in enumerate(all_docs):
        lm = doc['landmark']
        if lm:
            print(f"  [{i:03d}] {doc['item'].get_name():<40}  landmark={lm}")

    book_start, book_end = _book_content_bounds(all_docs)
    print(f"\n[PHASE 1] Book content zone : doc[{book_start}] → doc[{book_end}]  "
          f"({book_end - book_start + 1} documents)")
    print(f"          Front matter      : doc[0] → doc[{book_start - 1}]")
    print(f"          Back matter       : doc[{book_end + 1}] → doc[{len(all_docs) - 1}]")

    # =========================================================================
    # PASS 1 — Filter to candidate chapter documents.
    # =========================================================================
    candidate_docs = []

    for global_idx, doc in enumerate(all_docs):
        item           = doc['item']
        soup           = doc['soup']
        text           = doc['text']
        filename_lower = item.get_name().lower()

        # Must be inside the book-content zone
        if global_idx < book_start or global_idx > book_end:
            print(f"[PASS1] {item.get_name()} — outside book-content zone "
                  f"(global_idx={global_idx}), skipping")
            continue

        # Landmark docs are not chapters
        if doc['landmark'] is not None:
            print(f"[PASS1] {item.get_name()} — is landmark '{doc['landmark']}', skipping")
            continue

        if not text or len(text.strip()) < 50:
            print(f"[PASS1] {item.get_name()} — only {len(text.strip())} chars, "
                  f"keeping as gap placeholder")

        # Non-chapter keyword filter (filename)
        is_non_chapter = any(kw in filename_lower for kw in non_chapter_keywords)

        # Non-chapter keyword filter (headings)
        if not is_non_chapter:
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                if any(kw in heading.get_text(strip=True).lower()
                       for kw in non_chapter_keywords):
                    is_non_chapter = True
                    break

        if is_non_chapter:
            print(f"[PASS1] {item.get_name()} — matched non-chapter keyword, skipping")
            continue

        # Collect standalone numbers from the first 50 leaf elements
        number_candidates = []
        for element in soup.find_all(True)[:50]:
            if element.find(True) is not None:   # not a leaf
                continue
            element_text = element.get_text(strip=True)
            if re.match(r'^\d{1,3}$', element_text):
                number_candidates.append(int(element_text))

        print(f"[PASS1] {item.get_name()} — number_candidates: {number_candidates}")

        doc['number_candidates'] = number_candidates
        candidate_docs.append(doc)

    print(f"\n[PASS1 SUMMARY] {len(candidate_docs)} candidate documents collected\n")

    # =========================================================================
    # PASS 2 — Cross-document sequence validation.
    # =========================================================================
    max_depth = max(
        (len(d['number_candidates']) for d in candidate_docs),
        default=0
    )
    print(f"[PASS2] max_depth across all docs: {max_depth}")

    chapter_position         = None
    chapter_position_mapping = {}

    for i in range(max_depth):
        values_at_i = [
            (doc_idx, d['number_candidates'][i])
            for doc_idx, d in enumerate(candidate_docs)
            if len(d['number_candidates']) > i
        ]
        print(f"[PASS2] depth {i}: values_at_i = {values_at_i}")

        if len(values_at_i) < 2:
            print(f"[PASS2] depth {i}: skipping — fewer than 2 docs have a candidate here")
            continue

        best_run    = []
        current_run = [values_at_i[0]]

        for j in range(1, len(values_at_i)):
            prev_doc_idx, prev_val = values_at_i[j - 1]
            curr_doc_idx, curr_val = values_at_i[j]
            expected = prev_val + (curr_doc_idx - prev_doc_idx)
            if curr_val == expected:
                current_run.append(values_at_i[j])
            else:
                print(f"[PASS2] depth {i}: break in run at doc_idx={curr_doc_idx} "
                      f"(got {curr_val}, expected {expected})")
                if len(current_run) > len(best_run):
                    best_run = current_run
                current_run = [values_at_i[j]]

        if len(current_run) > len(best_run):
            best_run = current_run

        threshold = max(3, len(values_at_i) // 2)
        print(f"[PASS2] depth {i}: best_run={best_run} "
              f"(len={len(best_run)}, threshold={threshold})")

        if len(best_run) >= threshold:
            chapter_position         = i
            chapter_position_mapping = {doc_idx: val for doc_idx, val in best_run}
            print(f"[PASS2] ✅ Winning depth: {i}, mapping: {chapter_position_mapping}")
            break
        else:
            print(f"[PASS2] depth {i}: best run too short, continuing to next depth")

    if chapter_position is None:
        print("[PASS2] ❌ No valid chapter sequence found across any depth")

    # =========================================================================
    # PASS 3 — Assign chapter numbers using all methods.
    # =========================================================================
    chapters      = []
    seen_chapters = set()

    for doc in candidate_docs:
        item           = doc['item']
        soup           = doc['soup']
        text           = doc['text']
        filename_lower = item.get_name().lower()
        chapter_number = None
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
        if chapter_number is None and chapter_position is not None:
            doc_idx = candidate_docs.index(doc)
            if doc_idx in chapter_position_mapping:
                chapter_number = chapter_position_mapping[doc_idx]

        if chapter_number is None:
            continue

        # Extract title from first heading if available
        heading = soup.find(['h1', 'h2', 'h3'])
        if heading:
            heading_text = heading.get_text(strip=True)
            if heading_text and len(heading_text) < 100:
                title = heading_text

        if chapter_number in seen_chapters:
            print(f"⚠️  Duplicate chapter {chapter_number} found:")
            print(f"   Filename : {item.get_name()}")
            print(f"   Title    : {title}")
            continue

        seen_chapters.add(chapter_number)
        chapters.append({
            'chapter_number': chapter_number,
            'title':          title,
            'content':        text,
            'filename':       item.get_name(),
        })

    chapters.sort(key=lambda x: x['chapter_number'])
    return chapters


# ==============================================================================
# EPUB FILE PROCESSING (Django integration)
# ==============================================================================

def process_epub_file(epub_id):
    epub_obj = None
    try:
        epub_obj = EpubFile.objects.get(id=epub_id)
        epub_obj.status = 'pr'
        epub_obj.save()

        file_path = epub_obj.file.path

        chapters_data = extract_chapters_from_epub(file_path)

        for chapter_data in chapters_data:
            Chapter.objects.create(
                epub=epub_obj,
                title=chapter_data['title'],
                content=chapter_data['content'],
                chapter_number=chapter_data['chapter_number'],
            )

        epub_obj.status = 'c'
        epub_obj.processed = True
        epub_obj.save()

        return True

    except Exception as e:
        if epub_obj:
            epub_obj.status = 'f'
            epub_obj.error_message = str(e)
            epub_obj.save()
        raise
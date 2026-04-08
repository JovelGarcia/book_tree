# relationship_graph.py
"""
Orchestrator-Worker LangGraph for extracting scoped character relationships.

Pipeline:
  orchestrator  →  worker ×N (Send fan-out)  →  synthesizer  →  END

Depends on the character list produced by the wiki-discovery graph.
"""
from __future__ import annotations

import json
import operator
import re
import time
import threading
import random
from typing import Annotated, TypedDict
from urllib.parse import unquote

import requests
from anthropic import Anthropic, RateLimitError
from django.utils import timezone
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from .models import Character, MediaRequest, Relationship


# ── Constants ────────────────────────────────────────────────────────────────

HEADERS        = {'User-Agent': 'FandomGraphBot/1.0 (educational project; contact via GitHub)'}
CALL_DELAY     = 0.5          # seconds between Fandom API calls (sequential pre-fetch)
MAX_PAGE_CHARS = 30_000       # truncation limit per page before sending to the LLM
WORKER_MODEL   = "claude-haiku-4-5-20251001"
SYNTH_MODEL    = "claude-sonnet-4-6"
MIN_FUZZY_LEN  = 3            # minimum name length for substring matching
MAX_CONCURRENT_WORKERS = 3
_worker_semaphore = threading.Semaphore(MAX_CONCURRENT_WORKERS)
anthropic_client = Anthropic()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _api(slug: str) -> str:
    return f"https://{slug}.fandom.com/api.php"


def _get(url: str, params: dict | None = None) -> requests.Response:
    return requests.get(url, params=params, headers=HEADERS, timeout=15)


def _page_title_from_url(url: str) -> str:
    """Extract the human-readable page title from a Fandom wiki URL."""
    m = re.search(r'/wiki/(.+?)(?:\?|#|$)', url)
    raw = m.group(1) if m else url.rsplit('/', 1)[-1]
    return unquote(raw).replace('_', ' ')


def _fetch_page_wikitext(slug: str, page_title: str) -> str:
    """Return raw wikitext for *page_title*, following redirects."""
    params = {
        'action':    'parse',
        'page':      page_title,
        'prop':      'wikitext',
        'redirects': 1,
        'format':    'json',
    }
    try:
        r = _get(_api(slug), params=params)
        if not r.ok:
            return ''
        data = r.json()
        if 'error' in data:
            return ''
        return data.get('parse', {}).get('wikitext', {}).get('*', '')
    except Exception:
        return ''


def _clean_wikitext(raw: str) -> str:
    """Strip heavy wiki markup, keeping readable prose and section headers."""
    text = raw
    # Remove file / image embeds
    text = re.sub(r'\[\[(?:File|Image):.*?\]\]', '', text,
                  flags=re.IGNORECASE | re.DOTALL)
    # Simplify internal links: [[Page|Display]] → Display
    text = re.sub(r'\[\[[^]]*?\|([^]]+?)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^]]+?)\]\]', r'\1', text)
    # Collapse nested templates (iterative; covers ~6 levels of nesting)
    for _ in range(6):
        prev = text
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if text == prev:
            break
    # Strip references and HTML tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove inline category links
    text = re.sub(r'\[\[Category:.*?\]\]', '', text, flags=re.IGNORECASE)
    # Tidy whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _default_scope(media: MediaRequest) -> str:
    """Fallback scope when the caller does not provide one."""
    return (
        f"Only events, relationships, and character states established within "
        f"'{media.title}'. Ignore any sequels, future installments, or content "
        f"beyond this specific work."
    )

def _call_llm_with_backoff(
    *,
    model: str,
    max_tokens: int,
    tools: list,
    tool_choice: dict,
    messages: list,
    media_id: int,
    char_name: str,
    max_retries: int = 6,
    base_delay: float = 5.0,
) -> dict:
    """
    Call the Anthropic API with exponential backoff + jitter on 429s.
    Raises the last exception if all retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                messages=messages,
            )
        except RateLimitError as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(
                f"[WARNING] [{media_id}] Worker: rate-limited on {char_name!r} "
                f"(attempt {attempt}/{max_retries}). "
                f"Retrying in {delay:.1f}s…"
            )
            time.sleep(delay)


def _infer_and_save_aliases(characters: list, media_id: int) -> None:
    """
    Infer short-name aliases from the full character roster and persist them
    to each Character's ``aliases`` JSONField (merge, never overwrite).

    Two inference rules
    ───────────────────
    Rule A — Shared-suffix family name
        When ≥2 characters share the *same* surname-style suffix token
        (e.g. "au Grimmus"), each character's *first* whitespace-separated
        token is registered as an alias IF it is not already someone else's
        full name and is ≥ MIN_FUZZY_LEN chars.

        Example:
          "Seneca au Grimmus", "Vitalia au Grimmus", "Magnus au Grimmus"
          → aliases added: "Seneca", "Vitalia", "Magnus"

    Rule B — First-token alias for any multi-word name
        For every character whose name has ≥2 tokens, register the first
        token as an alias IF that token is not already a full name in the
        roster and is ≥ MIN_FUZZY_LEN chars.  This catches cases like
        "Eo of Lykos" → alias "Eo".

        Rule B is strictly weaker than Rule A (Rule A fires first on shared
        suffixes); Rule B fills the gap for unique compound names.
    """
    all_names_lower = {c.name.lower() for c in characters}

    # Collect suffix → list of characters that share it
    from collections import defaultdict
    suffix_groups: dict[str, list] = defaultdict(list)

    for c in characters:
        tokens = c.name.split()
        if len(tokens) >= 2:
            # "au Grimmus" = everything after the first token
            suffix = " ".join(tokens[1:]).lower()
            suffix_groups[suffix].append(c)

    to_update: list[tuple] = []   # (Character, new_alias)

    # Rule A — shared suffix
    for suffix, group in suffix_groups.items():
        if len(group) < 2:
            continue
        for c in group:
            first_token = c.name.split()[0]
            if (
                len(first_token) >= MIN_FUZZY_LEN
                and first_token.lower() not in all_names_lower
            ):
                to_update.append((c, first_token))

    # Rule B — any multi-word name whose first token isn't already a full name
    chars_covered_by_rule_a = {id(c) for c, _ in to_update}
    for c in characters:
        if id(c) in chars_covered_by_rule_a:
            continue
        tokens = c.name.split()
        if len(tokens) < 2:
            continue
        first_token = tokens[0]
        if (
            len(first_token) >= MIN_FUZZY_LEN
            and first_token.lower() not in all_names_lower
        ):
            to_update.append((c, first_token))

    # Persist — merge into existing aliases list, no duplicates
    for char, alias in to_update:
        existing = list(char.aliases or [])
        if alias not in existing:
            existing.append(alias)
            Character.objects.filter(id=char.id).update(aliases=existing)
            print(
                f"[INFO] [{media_id}] Alias inferred: "
                f"{char.name!r} → alias {alias!r}"
            )


# ── State Types ──────────────────────────────────────────────────────────────

class WorkItem(TypedDict):
    """One character prepared for the worker, including pre-fetched content."""
    name:         str
    wiki_page:    str
    page_content: str          # cleaned wikitext (may be empty if fetch failed)


class ExtractedRelationship(TypedDict):
    source:            str
    target:            str
    relationship_type: str
    description:       str


class ExtractionResult(TypedDict):
    character_name: str
    relationships:  list[ExtractedRelationship]
    page_found:     bool
    error:          str | None


class WorkerInput(TypedDict):
    """Payload delivered to each worker via Send."""
    media_id:          int
    title:             str
    media_type:        str
    scope_description: str
    wiki_slug:         str
    character_name:    str
    wiki_page:         str
    page_content:      str


class OrchestratorState(TypedDict):
    """Top-level graph state shared across all nodes."""
    media_id:          int
    title:             str
    media_type:        str
    scope_description: str
    wiki_slug:         str

    characters:        list[dict]          # raw DB rows
    work_items:        list[WorkItem]      # populated by orchestrator

    # ↓ accumulated across parallel workers via operator.add reducer
    extraction_results:     Annotated[list, operator.add]

    resolved_relationships: list[dict]
    conflicts:              list[dict]
    error:                  str | None


# ── Tool Schemas ─────────────────────────────────────────────────────────────

EXTRACT_RELATIONSHIPS_TOOL = {
    "name": "extract_relationships",
    "description": (
        "Extract character relationships from a wiki page, "
        "strictly within the specified scope boundary."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Full name of the other character.",
                        },
                        "relationship_type": {
                            "type": "string",
                            "enum": [
                                "family", "romantic", "ally", "enemy",
                                "mentor", "subordinate", "rival", "friend",
                                "acquaintance", "other",
                            ],
                            "description": "Primary category of the relationship.",
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Concise (1–2 sentence) description. "
                                "Be specific: 'father', 'commanding officer' — "
                                "not just 'family' or 'ally'."
                            ),
                        },
                    },
                    "required": ["target", "relationship_type", "description"],
                },
            }
        },
        "required": ["relationships"],
    },
}

RESOLVE_CONFLICTS_TOOL = {
    "name": "resolve_conflicts",
    "description": (
        "Resolve conflicting relationship data extracted from "
        "different character pages for the same character pair."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source":               {"type": "string"},
                        "target":               {"type": "string"},
                        "resolved_type": {
                            "type": "string",
                            "enum": [
                                "family", "romantic", "ally", "enemy",
                                "mentor", "subordinate", "rival", "friend",
                                "acquaintance", "other",
                            ],
                        },
                        "resolved_description": {"type": "string"},
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Why this resolution was chosen over the "
                                "conflicting alternatives."
                            ),
                        },
                    },
                    "required": [
                        "source", "target", "resolved_type",
                        "resolved_description", "reasoning",
                    ],
                },
            },
        },
        "required": ["resolutions"],
    },
}


# ── Node 1 · orchestrator ────────────────────────────────────────────────────

def orchestrator(state: OrchestratorState) -> OrchestratorState:
    """
    Load the character list from the DB, then sequentially pre-fetch every
    wiki page (with a polite delay) so workers only need to call the LLM.
    """
    media_id = state['media_id']
    slug     = state['wiki_slug']

    # ── Load characters ──────────────────────────────────────────────────
    characters = [
        {'name': c.name, 'wiki_page': c.wiki_page}
        for c in Character.objects.filter(media_id=media_id)
        if c.wiki_page
    ]

    if not characters:
        return {
            **state,
            'error': (
                f"No characters with wiki pages found for media_id={media_id}. "
                f"Run the wiki-discovery graph first."
            ),
        }

    print(
        f"[INFO] [{media_id}] Orchestrator: "
        f"{len(characters)} character(s) to process"
    )

    # ── Pre-fetch page content (rate-limited) ────────────────────────────
    work_items: list[WorkItem] = []
    for i, char in enumerate(characters, 1):
        page_title = _page_title_from_url(char['wiki_page'])
        time.sleep(CALL_DELAY)

        raw = _fetch_page_wikitext(slug, page_title)
        if not raw:
            print(
                f"[WARNING] [{media_id}] ({i}/{len(characters)}) "
                f"Empty page for {char['name']!r}"
            )
            work_items.append(WorkItem(
                name=char['name'],
                wiki_page=char['wiki_page'],
                page_content='',
            ))
            continue

        clean = _clean_wikitext(raw)
        if len(clean) > MAX_PAGE_CHARS:
            clean = clean[:MAX_PAGE_CHARS] + "\n\n[… content truncated …]"

        work_items.append(WorkItem(
            name=char['name'],
            wiki_page=char['wiki_page'],
            page_content=clean,
        ))

        if i % 10 == 0 or i == len(characters):
            print(
                f"[INFO] [{media_id}] Orchestrator: "
                f"pre-fetched {i}/{len(characters)} pages"
            )

    items_with_content = [w for w in work_items if w['page_content']]
    if not items_with_content:
        return {
            **state,
            'work_items': work_items,
            'error': (
                "All character wiki pages returned empty content. "
                "The wiki may be behind a login wall or the page titles "
                "may not match."
            ),
        }

    print(
        f"[INFO] [{media_id}] Orchestrator: "
        f"{len(items_with_content)}/{len(work_items)} pages have content"
    )

    MediaRequest.objects.filter(id=media_id).update(status='re')

    return {**state, 'characters': characters, 'work_items': work_items, 'error': None}


# ── Routing: fan-out to workers ──────────────────────────────────────────────

def fan_out_to_workers(state: OrchestratorState) -> list[Send]:
    """
    Return one ``Send("worker", ...)`` per character whose page had content,
    or route to the error handler if the orchestrator flagged a problem.
    """
    if state.get('error'):
        return [Send("handle_error", state)]

    sends: list[Send] = []
    for item in state['work_items']:
        if not item['page_content']:
            continue                           # skip characters with empty pages

        sends.append(Send("worker", WorkerInput(
            media_id=state['media_id'],
            title=state['title'],
            media_type=state['media_type'],
            scope_description=state['scope_description'],
            wiki_slug=state['wiki_slug'],
            character_name=item['name'],
            wiki_page=item['wiki_page'],
            page_content=item['page_content'],
        )))

    if not sends:
        return [Send("handle_error", {
            **state,
            'error': "No wiki pages had extractable content after pre-fetch.",
        })]

    return sends


# ── Node 2 · worker (one per character) ──────────────────────────────────────

def worker(state: WorkerInput) -> dict:
    """
    Receive one character's pre-fetched page content and call the LLM
    to extract scoped relationships.

    Concurrency is throttled via _worker_semaphore to avoid blowing the
    30k TPM org rate limit when many workers fire in parallel.

    Returns ``{'extraction_results': [ExtractionResult]}`` which merges
    into the parent state via the ``operator.add`` reducer.
    """
    char_name    = state['character_name']
    page_content = state['page_content']
    media_id     = state['media_id']

    print(f"[INFO] [{media_id}] Worker: queued {char_name!r}, waiting for slot…")

    with _worker_semaphore:
        print(f"[INFO] [{media_id}] Worker: extracting relationships for {char_name!r}")

        try:
            message = _call_llm_with_backoff(
                model=WORKER_MODEL,
                max_tokens=2048,
                tools=[EXTRACT_RELATIONSHIPS_TOOL],
                tool_choice={"type": "tool", "name": "extract_relationships"},
                messages=[{
                    "role": "user",
                    "content": f"""You are extracting character relationship data from a wiki page.

━━━ SCOPE CONSTRAINT (CRITICAL) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{state['scope_description']}

Character wiki pages frequently contain MAJOR SPOILERS and plot points from
FUTURE installments (sequels, later seasons, upcoming books). You MUST
strictly limit your extraction to relationships established within the
specified scope.

SCOPING RULES — follow all of these:
  1. SECTION HEADERS  If a section is named after a later installment
     (e.g. "In Golden Son", "Season 2", "Part II"), skip it entirely.
  2. TEMPORAL MARKERS  Ignore sentences containing "later", "eventually",
     "in the sequel", "by the end of the series", "after the events of …".
  3. RELATIONSHIP EVOLUTION  If a relationship changes across installments,
     report ONLY its state as of the END of the specified scope.
  4. CHARACTER STATUS  Treat characters as alive/active unless their death
     or departure occurs within scope.
  5. WHEN IN DOUBT → EXCLUDE  False negatives are strictly preferred over
     spoilers. If you cannot confidently place information within scope,
     leave it out.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHARACTER  : {char_name}
TITLE      : {state['title']}
MEDIA TYPE : {state['media_type']}

━━━ WIKI PAGE CONTENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{page_content}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTRUCTIONS:
  1. Identify which sections/paragraphs fall within scope.
  2. Extract ONLY relationships visible within those sections.
  3. Use full character names as they appear on the wiki.
  4. Be specific in descriptions — "father", "commanding officer",
     "childhood friend" — not bare category labels.
  5. Return an empty list if no in-scope relationships are found.
""",
                }],
                media_id=media_id,
                char_name=char_name,
            )

            tool_input = next(b.input for b in message.content if b.type == "tool_use")
            raw_rels   = tool_input["relationships"]

            print(
                f"[INFO] [{media_id}] Worker: {char_name} → "
                f"{len(raw_rels)} relationship(s)."
            )

            typed_rels: list[ExtractedRelationship] = [
                ExtractedRelationship(
                    source=char_name,
                    target=r['target'],
                    relationship_type=r['relationship_type'],
                    description=r['description'],
                )
                for r in raw_rels
            ]

            return {'extraction_results': [ExtractionResult(
                character_name=char_name,
                relationships=typed_rels,
                page_found=True,
                error=None,
            )]}

        except Exception as exc:
            print(f"[ERROR] [{media_id}] Worker failed for {char_name}: {exc}")
            return {'extraction_results': [ExtractionResult(
                character_name=char_name,
                relationships=[],
                page_found=True,
                error=str(exc),
            )]}


# ── Node 3 · synthesizer ────────────────────────────────────────────────────

def synthesizer(state: OrchestratorState) -> OrchestratorState:
    """
    Collect every worker's output, normalise character names, group
    relationships by pair, detect conflicts, resolve them via a single
    LLM call, and persist the final graph.
    """
    media_id = state['media_id']
    results  = state['extraction_results']

    # ── 1. Gather raw relationships ──────────────────────────────────────
    all_rels: list[ExtractedRelationship] = []
    worker_errors: list[str] = []

    for r in results:
        if r.get('error'):
            worker_errors.append(f"{r['character_name']}: {r['error']}")
        all_rels.extend(r.get('relationships', []))

    if worker_errors:
        print(
            f"[WARNING] [{media_id}] Synthesizer: "
            f"{len(worker_errors)} worker error(s): "
            + "; ".join(worker_errors[:5])
        )

    if not all_rels:
        msg = (
            "No relationships extracted from any character page. "
            "Pages may lack relationship data within the specified scope."
        )
        print(f"[WARNING] [{media_id}] {msg}")
        MediaRequest.objects.filter(id=media_id).update(
            status='rd',
            completed_at=timezone.now(),
            error_message=msg,
        )
        return {
            **state,
            'resolved_relationships': [],
            'conflicts': [],
            'error': None,
        }

    print(
        f"[INFO] [{media_id}] Synthesizer: "
        f"{len(all_rels)} raw relationship(s) from {len(results)} page(s)"
    )

    # ── 2. Build name-normalisation index ────────────────────────────────
    char_qs = list(Character.objects.filter(media_id=media_id))

    # -- 2a. Auto-infer aliases and persist them --------------------------
    _infer_and_save_aliases(char_qs, media_id)

    # Reload after alias updates so the index below picks them up
    char_qs = list(Character.objects.filter(media_id=media_id))

    # -- 2b. Build lookup: every known string → canonical Character -------
    #  Priority order (later entries can overwrite earlier ones):
    #    full name (lowest) → alias entries (higher) → exact name (highest)
    name_to_char: dict[str, Character] = {}

    for c in char_qs:
        name_to_char[c.name.lower()] = c
        for alias in (c.aliases or []):
            if alias and len(alias.strip()) >= MIN_FUZZY_LEN:
                name_to_char[alias.strip().lower()] = c

    # known_chars keeps the str→str mapping the rest of synthesizer uses
    known_chars: dict[str, str] = {k: v.name for k, v in name_to_char.items()}

    def _normalize(name: str) -> str:
        stripped = name.strip()
        low = stripped.lower()

        # 1. Exact match (full name or alias)
        if low in known_chars:
            return known_chars[low]

        # 2. Substring match — prefer the closest-length candidate
        if len(low) >= MIN_FUZZY_LEN:
            candidates = [
                (abs(len(k) - len(low)), v)
                for k, v in known_chars.items()
                if low in k or k in low
            ]
            if candidates:
                candidates.sort()
                return candidates[0][1]

        return stripped

    # Normalise all source/target names
    for rel in all_rels:
        rel['source'] = _normalize(rel['source'])
        rel['target'] = _normalize(rel['target'])

    # ── 3. Group by unordered pair ───────────────────────────────────────
    pair_map: dict[tuple[str, str], list[ExtractedRelationship]] = {}
    for rel in all_rels:
        key = tuple(sorted([rel['source'], rel['target']]))
        pair_map.setdefault(key, []).append(rel)

    # ── 4. Separate agreed pairs from conflicts ──────────────────────────
    conflicts:  list[dict] = []
    clean_rels: list[ExtractedRelationship] = []

    for pair_key, rels in pair_map.items():
        types = {r['relationship_type'] for r in rels}
        if len(types) <= 1:
            # Agreement — keep the most descriptive entry
            best = max(rels, key=lambda r: len(r.get('description', '')))
            clean_rels.append(best)
        else:
            conflicts.append({
                'pair':        list(pair_key),
                'extractions': [dict(r) for r in rels],
                'types':       list(types),
            })

    print(
        f"[INFO] [{media_id}] Synthesizer: "
        f"{len(clean_rels)} agreed pair(s), "
        f"{len(conflicts)} conflicting pair(s)"
    )

    # ── 5. Resolve conflicts via LLM ─────────────────────────────────────
    resolved_from_conflicts: list[ExtractedRelationship] = []

    if conflicts:
        try:
            block_parts: list[str] = []
            for c in conflicts:
                lines = [
                    f"• {c['pair'][0]} ↔ {c['pair'][1]}  "
                    f"(conflicting types: {', '.join(c['types'])})"
                ]
                for e in c['extractions']:
                    lines.append(
                        f"    From {e['source']}'s page: "
                        f"type={e['relationship_type']!r}  — "
                        f"{e['description']}"
                    )
                block_parts.append("\n".join(lines))

            message = anthropic_client.messages.create(
                model=SYNTH_MODEL,
                max_tokens=2048,
                tools=[RESOLVE_CONFLICTS_TOOL],
                tool_choice={"type": "tool", "name": "resolve_conflicts"},
                messages=[{
                    "role": "user",
                    "content": f"""Resolve conflicting character relationship data.

TITLE : {state['title']}
SCOPE : {state['scope_description']}

RESOLUTION GUIDELINES:
  • Specific types beat vague ones  ("mentor" > "ally",  "family" > "friend").
  • "family" and "romantic" are factual — prefer them over subjective labels.
  • If types are genuinely opposed ("ally" vs "enemy"), choose the PREDOMINANT
    state within scope.
  • More detailed descriptions carry more weight.
  • When complementary types appear from opposite perspectives
    (e.g. "mentor" from A's page, "subordinate" from B's page),
    pick the type whose perspective you assign to the *source* field
    and describe the relationship accordingly.

CONFLICTS TO RESOLVE:
{chr(10).join(block_parts)}
""",
                }],
            )

            tool_input = next((b.input for b in message.content if b.type == "tool_use"), {})
            resolutions = tool_input.get("resolutions") or []

            for res in resolutions:
                resolved_from_conflicts.append(ExtractedRelationship(
                    source=res['source'],
                    target=res['target'],
                    relationship_type=res['resolved_type'],
                    description=res['resolved_description'],
                ))
                print(
                    f"[INFO] [{media_id}] Resolved: "
                    f"{res['source']} ↔ {res['target']} → "
                    f"{res['resolved_type']}  ({res['reasoning']})"
                )

        except Exception as exc:
            print(f"[ERROR] [{media_id}] Conflict resolution failed: {exc}")
            # Fallback: keep the first extraction for each conflicting pair
            for c in conflicts:
                first = c['extractions'][0]
                resolved_from_conflicts.append(ExtractedRelationship(
                    source=first['source'],
                    target=first['target'],
                    relationship_type=first['relationship_type'],
                    description=first['description'],
                ))

    # ── 6. Merge & persist ───────────────────────────────────────────────
    final_rels = clean_rels + resolved_from_conflicts

    media     = MediaRequest.objects.get(id=media_id)
    char_objs = {c.name: c for c in Character.objects.filter(media=media)}
    char_low  = {k.lower(): v for k, v in char_objs.items()}

    saved   = 0
    skipped = 0

    for rel in final_rels:
        src = char_objs.get(rel['source']) or char_low.get(rel['source'].lower())
        tgt = char_objs.get(rel['target']) or char_low.get(rel['target'].lower())

        if src and tgt:
            Relationship.objects.update_or_create(
                media=media,
                source=src,
                target=tgt,
                defaults={
                    'relationship_type': rel['relationship_type'],
                    'description':       rel['description'],
                },
            )
            saved += 1
        else:
            skipped += 1
            missing = []
            if not src:
                missing.append(f"source={rel['source']!r}")
            if not tgt:
                missing.append(f"target={rel['target']!r}")
            print(
                f"[WARNING] [{media_id}] Skipped: "
                f"{', '.join(missing)} not in character list"
            )

    # ── 7. Mark complete ─────────────────────────────────────────────────
    media.status       = 'c'
    media.completed_at = timezone.now()
    media.save(update_fields=['status', 'completed_at'])

    print(
        f"[INFO] [{media_id}] Synthesizer complete — "
        f"{saved} saved, {skipped} skipped, "
        f"{len(conflicts)} conflict(s) resolved"
    )

    return {
        **state,
        'resolved_relationships': [dict(r) for r in final_rels],
        'conflicts':              conflicts,
        'error':                  None,
    }


# ── Error handler ────────────────────────────────────────────────────────────

def handle_error(state: OrchestratorState) -> OrchestratorState:
    media_id = state.get('media_id', '?')
    error    = state.get('error', 'unknown error')
    print(f"[ERROR] [{media_id}] Relationship extraction failed: {error}")
    MediaRequest.objects.filter(id=media_id).update(
        status='rf',
        error_message=error,
    )
    return state


# ── Graph construction ───────────────────────────────────────────────────────

def build_relationship_graph() -> StateGraph:
    g = StateGraph(OrchestratorState)

    g.add_node('orchestrator', orchestrator)
    g.add_node('worker',       worker)
    g.add_node('synthesizer',  synthesizer)
    g.add_node('handle_error', handle_error)

    g.set_entry_point('orchestrator')

    # Orchestrator fans out: one Send per character → worker,
    # or a single Send → handle_error if something went wrong.
    g.add_conditional_edges(
        'orchestrator',
        fan_out_to_workers,
        ['worker', 'handle_error'],
    )

    # All workers converge into the synthesizer (reducer merges results).
    g.add_edge('worker',       'synthesizer')
    g.add_edge('synthesizer',  END)
    g.add_edge('handle_error', END)

    return g.compile()


relationship_graph = build_relationship_graph()


# ── Public entry point ───────────────────────────────────────────────────────

def run_relationship_extraction(
    media_id: int,
    scope_description: str | None = None,
) -> None:
    """
    Extract scoped character relationships for a media title whose
    characters have already been scraped by the wiki-discovery graph.

    Parameters
    ----------
    media_id
        Primary key of the ``MediaRequest``.
    scope_description
        Natural-language boundary for what counts as "in scope".
        Example for a book series::

            "Only events and relationships from 'Red Rising' (Book 1).
             Ignore all content from 'Golden Son' (Book 2), 'Morning Star'
             (Book 3), 'Iron Gold' (Book 4), 'Dark Age' (Book 5), and
             'Light Bringer' (Book 6)."

        If ``None``, a generic scope is generated from the title.
    """
    media = MediaRequest.objects.get(id=media_id)

    # ── Pre-flight checks ────────────────────────────────────────────────
    if not media.wiki_slug:
        raise ValueError(
            f"No wiki_slug set for '{media.title}' (media_id={media_id}). "
            f"Run the wiki-discovery graph first."
        )

    char_count = Character.objects.filter(media=media).count()
    if char_count == 0:
        raise ValueError(
            f"No characters found for '{media.title}' (media_id={media_id}). "
            f"Run the wiki-discovery graph first."
        )

    scope = scope_description or _default_scope(media)

    print(
        f"[INFO] [{media_id}] Starting relationship extraction for "
        f"'{media.title}' ({char_count} characters). Scope: {scope[:120]}…"
    )

    relationship_graph.invoke({
        'media_id':               media_id,
        'title':                  media.title,
        'media_type':             media.media_type,
        'scope_description':      scope,
        'wiki_slug':              media.wiki_slug,
        'characters':             [],
        'work_items':             [],
        'extraction_results':     [],
        'resolved_relationships': [],
        'conflicts':              [],
        'error':                  None,
    })
from __future__ import annotations

import re
import time
import json
from typing import Literal, TypedDict

import requests
from anthropic import Anthropic
from django.utils import timezone
from langgraph.graph import StateGraph, END

from .models import Character, MediaRequest


# ── Constants ────────────────────────────────────────────────────────────────

HEADERS    = {'User-Agent': 'FandomGraphBot/1.0 (educational project; contact via GitHub)'}
CALL_DELAY = 0.25

FANDOM_SLUG_RE = re.compile(r'https?://([a-z0-9-]+)\.fandom\.com')

_FANDOM_SKIP_SLUGS = frozenset({
    'www', 'community', 'support', 'help', 'blog', 'mobile', 'static',
})

_QUERY_TEMPLATES = [
    '{title} fandom wiki'
]

_MEDIA_TYPE_KW: dict[str, re.Pattern] = {
    'anime':  re.compile(r'\b(anime|manga|animated)\b',         re.IGNORECASE),
    'tv':     re.compile(r'\b(tv|television|series|episode)\b', re.IGNORECASE),
    'movie':  re.compile(r'\b(film|movie|cinema|animated)\b',   re.IGNORECASE),
    'game':   re.compile(r'\b(game|gaming|video\s*game|dlc)\b', re.IGNORECASE),
    'book':   re.compile(r'\b(book|novel|literature|fiction)\b',re.IGNORECASE),
}

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_PAGE_SIZE = 5  # Brave returns up to 20 results per request on free tier

anthropic_client = Anthropic()


# ── State ────────────────────────────────────────────────────────────────────

ResolutionStrategy = Literal[
    'dedicated_scoped',
    'umbrella_scoped',
    'dedicated_unscoped',
    'no_category',
]


class WikiCandidate(TypedDict):
    slug:                   str
    url:                    str
    result_count:           int
    page_titles:            list[str]
    snippets:               list[str]
    has_character_category: bool
    top_hit_url:            str | None


class AgentState(TypedDict):
    media_id:            int
    title:               str
    media_type:          str
    wiki_candidates:     list[WikiCandidate]
    wiki_slug:           str
    wiki_url:            str
    is_umbrella_wiki:    bool
    all_categories:      list[str]
    chosen_category:     str
    metadata_categories: list[str]
    resolution_strategy: ResolutionStrategy | None
    character_names:     list[dict]
    error:               str | None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> requests.Response:
    return requests.get(url, params=params, headers=HEADERS, timeout=10)

def _api(slug: str) -> str:
    return f"https://{slug}.fandom.com/api.php"


# ── Brave Search API ──────────────────────────────────────────────────────────

def _brave_search(query: str, api_key: str, count: int = 20) -> list[tuple[str, str, str]]:
    """
    Execute a Brave web search and return [(url, title, snippet), …].

    Brave returns up to 20 results per request on the free tier (no pagination).
    Raises RuntimeError on HTTP errors or unexpected response shapes.
    Returns [] on a clean call that produced no results.
    """
    headers = {
        'Accept':               'application/json',
        'Accept-Encoding':      'gzip',
        'X-Subscription-Token': api_key,
    }
    params = {
        'q':     query,
        'count': min(count, _BRAVE_PAGE_SIZE),
    }

    try:
        r = requests.get(_BRAVE_ENDPOINT, headers=headers, params=params, timeout=10)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Brave Search network error for query {query!r}: {exc}"
        ) from exc

    if r.status_code == 401:
        raise RuntimeError(
            "Brave Search API key is invalid or missing (HTTP 401). "
            "Check BRAVE_API_KEY in your Django settings."
        )
    if r.status_code == 429:
        raise RuntimeError(
            "Brave Search rate limit or monthly quota exceeded (HTTP 429). "
            "The free tier allows 2,000 queries/month. "
            "Check usage at api.search.brave.com/app/subscriptions."
        )
    if not r.ok:
        raise RuntimeError(
            f"Brave Search HTTP {r.status_code} for query {query!r}: {r.text[:200]}"
        )

    try:
        data = r.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Brave Search returned non-JSON for query {query!r}: {exc}"
        ) from exc

    web_results = data.get('web', {}).get('results') or []
    if not web_results:
        return []

    return [
        (
            item.get('url',         '') or '',
            item.get('title',       '') or '',
            item.get('description', '') or '',
        )
        for item in web_results
        if item.get('url')
    ]


def _brave_search_fandom(title: str) -> list[WikiCandidate]:
    """
    Walk _QUERY_TEMPLATES via the Brave Search API and return ranked
    WikiCandidate objects. Stops at the first template that yields
    fandom.com URLs.

    Raises RuntimeError when:
      • BRAVE_API_KEY is missing from settings, or
      • every template failed with an API / network error, or
      • every template returned 0 results.

    Returns [] only when the API returned real URLs but none matched the
    fandom slug pattern — likely a title-spelling issue.
    """
    from django.conf import settings

    api_key = getattr(settings, 'BRAVE_API_KEY', '')

    if not api_key:
        raise RuntimeError(
            "BRAVE_API_KEY is not set in Django settings. "
            "Sign up at api.search.brave.com and add the key to your .env."
        )

    slug_results:     dict[str, list[tuple[str, str, str]]] = {}
    slug_char_cat:    dict[str, bool]                       = {}
    successful_query: str | None                            = None
    search_failures:  list[str]                             = []
    clean_runs:        int = 0
    total_raw_results: int = 0

    for template in _QUERY_TEMPLATES:
        query = template.format(title=title)
        print(f"[DEBUG] Brave Search query attempt: {query!r}")

        try:
            results = _brave_search(query, api_key)
        except RuntimeError as exc:
            reason = str(exc)
            # Auth errors are unrecoverable — no point trying further templates.
            if 'BRAVE_API_KEY' in reason or 'HTTP 401' in reason:
                raise
            search_failures.append(f"{query!r} → {reason}")
            print(f"[WARNING] Brave Search query {query!r} failed — {reason}")
            continue

        clean_runs        += 1
        total_raw_results += len(results)
        print(f"[DEBUG] Query {query!r}: Brave returned {len(results)} result(s).")

        for url, page_title, snippet in results:
            m = FANDOM_SLUG_RE.match(url)
            if not m:
                continue
            slug = m.group(1)
            if slug in _FANDOM_SKIP_SLUGS:
                continue

            slug_results.setdefault(slug, []).append((url, page_title, snippet))

            if '/wiki/Category:' in url and re.search(r'character', url, re.IGNORECASE):
                slug_char_cat[slug] = True

        if slug_results:
            successful_query = query
            print(f"[DEBUG] Query {query!r} produced {len(slug_results)} slug(s); stopping chain.")
            break

        if results:
            print(f"[DEBUG] Query {query!r}: {len(results)} result(s) returned but none parsed as fandom slugs.")
        else:
            print(f"[DEBUG] Query {query!r}: 0 results returned.")

    # ── Happy path ────────────────────────────────────────────────────────────
    if slug_results:
        candidates: list[WikiCandidate] = []
        for slug, hits in slug_results.items():
            candidates.append(WikiCandidate(
                slug=slug,
                url=f"https://{slug}.fandom.com",
                result_count=len(hits),
                page_titles=[t for _, t, _ in hits if t][:8],
                snippets=   [s for _, _, s in hits if s][:4],
                has_character_category=slug_char_cat.get(slug, False),
                top_hit_url=hits[0][0],
            ))

        candidates.sort(
            key=lambda c: (c['result_count'], c['has_character_category']),
            reverse=True,
        )

        print(
            f"[INFO] Brave Search found {len(candidates)} fandom wiki(s) for {title!r} "
            f"via query {successful_query!r}: {[c['slug'] for c in candidates]}"
        )
        return candidates

    # ── Failure diagnosis ─────────────────────────────────────────────────────
    all_errored  = clean_runs == 0
    silent_empty = clean_runs > 0 and total_raw_results == 0

    if all_errored:
        detail = '; '.join(search_failures)
        raise RuntimeError(
            f"All {len(_QUERY_TEMPLATES)} Brave Search query templates for '{title}' "
            f"failed with exceptions. Errors: {detail}"
        )

    if silent_empty:
        extra = (
            f"  Additional exception(s): {'; '.join(search_failures)}"
            if search_failures else ''
        )
        raise RuntimeError(
            f"Brave Search returned 0 results for all {len(_QUERY_TEMPLATES)} templates "
            f"for '{title}' despite {clean_runs} clean run(s). "
            f"Check your monthly quota (2,000 queries/month free) at "
            f"api.search.brave.com/app/subscriptions.{extra}"
        )

    if search_failures:
        print(
            f"[WARNING] {len(search_failures)} of {len(_QUERY_TEMPLATES)} query template(s) "
            f"also failed for {title!r}: {'; '.join(search_failures)}"
        )

    print(
        f"[WARNING] Brave Search returned {total_raw_results} total result(s) across "
        f"{clean_runs} clean run(s) for {title!r} but none parsed as fandom.com slugs. "
        f"Verify the title spelling. Templates tried: "
        f"{[t.format(title=title) for t in _QUERY_TEMPLATES]}"
    )
    return []


def _filter_relevant_categories(
    all_cats: list[str],
    title: str,
    media_type: str = '',       # ← NEW parameter
) -> list[str]:
    title_lower = title.lower()
    no_article  = re.sub(r'^(the|a|an)\s+', '', title_lower).strip()

    CHARACTER_KW = re.compile(
        r'\b(characters?|cast|individuals?|people|persons?|protagonists?|'
        r'antagonists?|hero(?:es|ines?)?|villains?|males?|females?|species|'
        r'factions?|races?|teams?|organi[sz]ations?|families|clans?|'
        r'allies|enemies|inhabitants?)\b',
        re.IGNORECASE,
    )

    # Regex for the submitted media type, if we have one
    media_type_re: re.Pattern | None = _MEDIA_TYPE_KW.get(media_type.lower())

    title_cats      = []
    char_cats       = []
    media_type_cats = []   # ← NEW bucket

    for cat in all_cats:
        cat_lower = cat.lower()
        is_title_match = (
            title_lower in cat_lower
            or (len(no_article) > 3 and no_article in cat_lower)
        )
        is_char_match       = bool(CHARACTER_KW.search(cat))
        is_media_type_match = bool(media_type_re and media_type_re.search(cat))

        if is_title_match:
            title_cats.append(cat)
        elif is_char_match and is_media_type_match:
            # e.g. "Anime characters" when media_type='anime' — high value
            media_type_cats.append(cat)
        elif is_char_match:
            char_cats.append(cat)

    # Priority: title-scoped > media-type+character > generic character
    return title_cats + media_type_cats + char_cats[:1000]


# ── Node 1: search_wiki_candidates ───────────────────────────────────────────

def search_wiki_candidates(state: AgentState) -> AgentState:
    title    = state['title']
    media_id = state['media_id']

    try:
        candidates = _brave_search_fandom(title)
    except RuntimeError as exc:
        return {**state, 'error': str(exc)}

    if not candidates:
        tried = [t.format(title=title) for t in _QUERY_TEMPLATES]
        return {**state, 'error': (
            f"Brave Search returned no Fandom results for '{title}' after trying "
            f"{len(tried)} template(s): {tried}. "
            "Verify the title spelling — the wiki may also not yet be indexed."
        )}

    print(f"[INFO] [{media_id}] Brave found {len(candidates)} wiki candidate(s): {[c['slug'] for c in candidates]}")

    return {**state, 'wiki_candidates': candidates, 'error': None}


# ── Tool schema: pick_wiki ────────────────────────────────────────────────────

PICK_WIKI_TOOL = {
    "name": "pick_wiki",
    "description": (
        "Select the single Fandom wiki that will yield the richest data "
        "for this title, based on the aggregated search signals."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chosen_slug": {
                "type": "string",
                "description": (
                    "The slug of the chosen wiki (e.g. 'red-rising'). "
                    "Must exactly match one of the slugs in the candidates list."
                ),
            },
            "is_umbrella_wiki": {
                "type": "boolean",
                "description": (
                    "True if this wiki hosts many unrelated franchises under one roof "
                    "(e.g. movies.fandom.com, pixar.fandom.com, marvel.fandom.com). "
                    "False if it is dedicated exclusively to this title's franchise. "
                    "Infer this from the slug name, the page titles, and the snippets."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Two or three sentences explaining the choice. "
                    "Reference specific signals — result_count, has_character_category, "
                    "page titles, or snippet content — to justify why this wiki is "
                    "richer than the alternatives."
                ),
            },
        },
        "required": ["chosen_slug", "is_umbrella_wiki", "reasoning"],
    },
}


# ── Node 2: claude_pick_wiki ──────────────────────────────────────────────────

def claude_pick_wiki(state: AgentState) -> AgentState:
    candidates = state['wiki_candidates']

    parts: list[str] = []
    for i, c in enumerate(candidates, 1):
        lines = [
            f"Candidate {i}  slug={c['slug']!r}",
            f"  Base URL             : {c['url']}",
            f"  Search hits          : {c['result_count']}",
            f"  Character category   : "
            f"{'YES — Category:*Character* URL found in results' if c['has_character_category'] else 'not seen in results'}",
            f"  Top result URL       : {c['top_hit_url']}",
        ]
        if c['page_titles']:
            lines.append(f"  Page titles          : {c['page_titles']}")
        if c['snippets']:
            lines.append("  Content snippets     :")
            for s in c['snippets']:
                lines.append(f"    • {s[:250]}")
        parts.append("\n".join(lines))

    candidate_block = "\n\n".join(parts)

    message = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        tools=[PICK_WIKI_TOOL],
        tool_choice={"type": "tool", "name": "pick_wiki"},
        messages=[{
            "role": "user",
            "content": f"""You are selecting the best Fandom wiki to scrape character data from.

Title     : {state['title']}
Media type: {state['media_type']}

All candidates were discovered by searching for variations of:
  "{state['title']}" site:fandom.com

{candidate_block}

Selection criteria (priority order):
1. has_character_category=YES is the strongest signal — a dedicated character category
   page was found on that wiki, which is exactly what the scraper needs.
2. A higher result_count means more pages about this title were indexed from
   that wiki, which strongly correlates with coverage depth.
3. Prefer a wiki whose page titles and snippets are clearly about THIS specific
   title rather than a passing mention on a list page.
4. Prefer a wiki that is about the title rather than a page dedicated to a chararcter.
5. A dedicated single-franchise wiki beats an umbrella wiki when all other signals
   are equal, because its categories will be title-scoped by default.

You MUST choose one of the slugs listed above.
""",
        }],
    )

    tool_input  = next(b.input for b in message.content if b.type == "tool_use")
    chosen_slug = tool_input["chosen_slug"].strip()
    is_umbrella = tool_input["is_umbrella_wiki"]
    reasoning   = tool_input["reasoning"]

    candidate_map = {c['slug']: c for c in candidates}
    if chosen_slug not in candidate_map:
        match = next(
            (s for s in candidate_map if chosen_slug in s or s in chosen_slug), None
        )
        if match:
            print(f"[INFO] [{state['media_id']}] Fuzzy matched slug {chosen_slug!r} → {match!r}")
            chosen_slug = match
        else:
            return {**state, 'error': (
                f"Claude returned slug {chosen_slug!r} which is not in the candidate set: "
                f"{list(candidate_map)}"
            )}

    chosen = candidate_map[chosen_slug]
    print(f"[INFO] [{state['media_id']}] Claude chose {chosen_slug!r} (umbrella={is_umbrella}) — {reasoning}")

    MediaRequest.objects.filter(id=state['media_id']).update(
        wiki_slug=chosen_slug,
        wiki_url=chosen['url'],
    )

    return {
        **state,
        'wiki_slug':        chosen_slug,
        'wiki_url':         chosen['url'],
        'is_umbrella_wiki': is_umbrella,
        'error':            None,
    }


# ── Node 3: fetch_categories ─────────────────────────────────────────────────

def fetch_categories(state: AgentState) -> AgentState:
    slug = state['wiki_slug']
    cats: list[str] = []
    params = {
        'action':  'query',
        'list':    'allcategories',
        'aclimit': 500,
        'format':  'json',
    }

    while True:
        try:
            r = _get(_api(slug), params=params)
            if not r.ok:
                print(
                    f"[ERROR] fetch_categories API error | status={r.status_code} "
                    f"| url={r.url} | body={r.text[:500]}"
                )
            data = r.json()
            cats += [c['*'] for c in data.get('query', {}).get('allcategories', [])]
            cont = data.get('continue', {}).get('accontinue')
            if cont:
                params['accontinue'] = cont
                time.sleep(CALL_DELAY)
            else:
                break
        except Exception as exc:
            print(f"[WARNING] [{state['media_id']}] fetch_categories error: {exc}")
            break

    print(f"[INFO] [{state['media_id']}] Fetched {len(cats)} categories from {slug!r}")
    return {**state, 'all_categories': cats, 'error': None}


# ── Tool schema: pick_category ────────────────────────────────────────────────

PICK_CATEGORY_TOOL = {
    "name": "pick_character_category",
    "description": (
        "Analyse the wiki's categories and classify the resolution strategy. "
        "This determines both what to scrape and how to flag limitations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "resolution_strategy": {
                "type": "string",
                "enum": [
                    "dedicated_scoped",
                    "umbrella_scoped",
                    "dedicated_unscoped",
                    "no_category",
                ],
                "description": (
                    "Classification of how character data is organised on this wiki:\n"
                    "  dedicated_scoped   — dedicated wiki + a category scoped to THIS title's characters\n"
                    "  umbrella_scoped    — umbrella wiki  + a category scoped to THIS title's characters\n"
                    "  dedicated_unscoped — dedicated wiki but only wiki-wide categories exist\n"
                    "  no_category        — no usable character category exists at all\n"
                    "\n"
                    "A category is 'scoped' if it is named after the specific title, film, book, or game. "
                    "Generic wiki-level categories like 'Individuals' or 'Characters' are NOT scoped."
                ),
            },
            "primary_category": {
                "type": "string",
                "description": (
                    "The single category that most directly and completely lists the main cast. "
                    "Use an empty string if resolution_strategy is 'no_category'."
                ),
            },
            "metadata_categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Other categories grouping characters by faction, race, organisation, "
                    "allegiance, species, or affiliation. Empty list if none are relevant."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the strategy classification and primary category choice.",
            },
        },
        "required": ["resolution_strategy", "primary_category", "metadata_categories", "reasoning"],
    },
}


# ── Node 4: claude_pick_category ──────────────────────────────────────────────

def claude_pick_category(state: AgentState) -> AgentState:
    all_cats   = state['all_categories']
    media_type = state['media_type']          # ← already in state, just use it

    if not all_cats:
        print(f"[INFO] [{state['media_id']}] No categories found on wiki, routing to no_category")
        _save_strategy(state['media_id'], 'no_category', '', [],
                       "No categories found on this wiki.")
        return {
            **state,
            'chosen_category':     '',
            'metadata_categories': [],
            'resolution_strategy': 'no_category',
            'error':               None,
        }

    # ↓ Pass media_type so the filter keeps "Anime characters" etc.
    filtered_cats = _filter_relevant_categories(all_cats, state['title'], media_type)

    if not filtered_cats:
        print(f"[INFO] [{state['media_id']}] No relevant categories after filtering {len(all_cats)} total")
        _save_strategy(
            state['media_id'], 'no_category', '', [],
            f"Wiki has {len(all_cats)} categories but none related to "
            f"characters or to '{state['title']}'.",
        )
        return {
            **state,
            'chosen_category':     '',
            'metadata_categories': [],
            'resolution_strategy': 'no_category',
            'error':               None,
        }

    print(f"[INFO] [{state['media_id']}] Filtered {len(all_cats)} → {len(filtered_cats)} relevant categories")

    message = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=768,
        tools=[PICK_CATEGORY_TOOL],
        tool_choice={"type": "tool", "name": "pick_character_category"},
        messages=[{
            "role": "user",
            "content": f"""You are helping build a character relationship graph for a media title.

Title     : {state['title']}
Media type: {media_type}
Wiki slug : {state['wiki_slug']}
Is umbrella wiki: {state['is_umbrella_wiki']}

Available categories (filtered; {len(filtered_cats)} of {len(all_cats)} total):
{json.dumps(filtered_cats)}

─── SCOPING RULES ────────────────────────────────────────────────────────────

A category is considered SCOPED to this title if ANY of the following is true:
  (a) Its name contains the title itself     e.g. "Death Note Characters"
  (b) Its name contains the media type       e.g. "Anime characters"
        when media_type={media_type!r} — this counts as scoped because
        a dedicated wiki for one anime series uses "Anime characters" to
        mean exactly the same cast as "Death Note Characters" would.
  (c) It combines both                       e.g. "Death Note Anime Characters"

Generic wiki-wide labels with NO title or media-type signal — such as bare
"Characters", "Individuals", "People" — are NOT scoped.

─── YOUR TWO TASKS ───────────────────────────────────────────────────────────

1. CLASSIFY resolution_strategy:
   dedicated_scoped   → dedicated wiki  + at least one SCOPED category (per rules above)
   umbrella_scoped    → umbrella wiki   + at least one SCOPED category
   dedicated_unscoped → dedicated wiki  + only generic/unscoped categories exist
   no_category        → no character category of any kind exists

2. SELECT primary_category and metadata_categories:
   Primary  : the broadest SCOPED category if one exists; otherwise the broadest
              generic character category; empty string only for no_category.
   Metadata : factions, races, organisations, allegiances, species. Empty list if none.
""",
        }],
    )

    tool_input = next(b.input for b in message.content if b.type == "tool_use")
    strategy   = tool_input["resolution_strategy"]
    primary    = tool_input["primary_category"]
    metadata   = tool_input.get("metadata_categories", [])
    reasoning  = tool_input["reasoning"]

    print(f"[INFO] [{state['media_id']}] Claude: strategy={strategy}, primary={primary!r} — {reasoning}")

    if primary:
        if primary not in all_cats:
            match = next(
                (c for c in all_cats if primary.lower() in c.lower()
                 or c.lower() in primary.lower()),
                None,
            )
            if match:
                print(f"[INFO] [{state['media_id']}] Fuzzy matched primary {primary!r} → {match!r}")
                primary = match
            else:
                print(f"[WARNING] [{state['media_id']}] Primary {primary!r} not found; downgrading to no_category")
                strategy = 'no_category'
                primary  = ''

    valid_metadata = [c for c in metadata if c in all_cats]
    dropped = set(metadata) - set(valid_metadata)
    if dropped:
        print(f"[INFO] [{state['media_id']}] Dropped invalid metadata categories: {dropped}")

    _save_strategy(state['media_id'], strategy, primary, valid_metadata, reasoning)

    return {
        **state,
        'chosen_category':     primary,
        'metadata_categories': valid_metadata,
        'resolution_strategy': strategy,
        'error':               None,
    }


def _save_strategy(
    media_id: int,
    strategy: str,
    primary: str,
    metadata: list[str],
    reasoning: str,
) -> None:
    MediaRequest.objects.filter(id=media_id).update(
        resolution_strategy=strategy,
        chosen_category=primary,
        metadata_categories=metadata,
        strategy_reasoning=reasoning,
    )


# ── Node 5a: scrape_characters ────────────────────────────────────────────────

def scrape_characters(state: AgentState) -> AgentState:
    slug     = state['wiki_slug']
    category = state['chosen_category']
    media_id = state['media_id']

    try:
        all_members: list[dict] = []
        params = {
            'action':      'query',
            'list':        'categorymembers',
            'cmtitle':     f'Category:{category}',
            'cmlimit':     500,
            'cmtype':      'page',
            'cmnamespace': 0,
            'format':      'json',
        }

        while True:
            r = _get(_api(slug), params=params)
            if not r.ok:
                return {**state, 'error': f"categorymembers call failed (HTTP {r.status_code})"}

            data    = r.json()
            members = data.get('query', {}).get('categorymembers', [])
            all_members += [m for m in members if ':' not in m['title']]

            cont = data.get('continue', {}).get('cmcontinue')
            if cont:
                params['cmcontinue'] = cont
                time.sleep(CALL_DELAY)
            else:
                break

        if not all_members:
            return {**state, 'error': f"Category '{category}' exists but has no page members."}

        media           = MediaRequest.objects.get(id=media_id)
        character_names = []

        for m in all_members:
            name      = m['title']
            wiki_page = f"https://{slug}.fandom.com/wiki/{name.replace(' ', '_')}"
            Character.objects.update_or_create(
                media=media,
                name=name,
                defaults={'wiki_page': wiki_page},
            )
            character_names.append({'name': name, 'wiki_page': wiki_page})

        media.status       = 'c'
        media.completed_at = timezone.now()
        media.save(update_fields=['status', 'completed_at'])
        print(f"[INFO] [{media_id}] Scraped {len(character_names)} characters from Category:{category}")

        return {**state, 'character_names': character_names, 'error': None}

    except Exception as exc:
        return {**state, 'error': str(exc)}


# ── Node 5b: flag_no_scope ────────────────────────────────────────────────────

def flag_no_scope(state: AgentState) -> AgentState:
    media_id = state['media_id']
    msg = (
        f"No title-scoped character category found on {state['wiki_slug']}.fandom.com "
        f"for '{state['title']}'. "
        f"Best available category is '{state['chosen_category']}' (wiki-wide). "
        f"Downstream scraping will need to filter by title."
    )
    print(f"[INFO] [{media_id}] {msg}")
    MediaRequest.objects.filter(id=media_id).update(
        status='ns',
        completed_at=timezone.now(),
        error_message=msg,
    )
    return {**state, 'error': None}


# ── Node 5c: flag_no_category ─────────────────────────────────────────────────

def flag_no_category(state: AgentState) -> AgentState:
    media_id = state['media_id']
    msg = (
        f"No character category found on {state['wiki_slug']}.fandom.com "
        f"for '{state['title']}'. "
        f"Characters may be listed in a page section rather than a category. "
        f"Manual or page-parse extraction required."
    )
    print(f"[INFO] [{media_id}] {msg}")
    MediaRequest.objects.filter(id=media_id).update(
        status='nc',
        completed_at=timezone.now(),
        error_message=msg,
    )
    return {**state, 'error': None}


# ── Node: handle_error ────────────────────────────────────────────────────────

def handle_error(state: AgentState) -> AgentState:
    print(f"[ERROR] [{state['media_id']}] Agent failed: {state['error']}")
    MediaRequest.objects.filter(id=state['media_id']).update(
        status='f',
        error_message=state['error'],
    )
    return state


# ── Routing ───────────────────────────────────────────────────────────────────

def route_on_error(state: AgentState) -> str:
    return 'handle_error' if state.get('error') else 'continue'


def route_on_strategy(state: AgentState) -> str:
    if state.get('error'):
        return 'handle_error'
    strategy = state.get('resolution_strategy')
    if strategy in ('dedicated_scoped', 'umbrella_scoped'):
        return 'scrape_characters'
    if strategy == 'dedicated_unscoped':
        return 'flag_no_scope'
    return 'flag_no_category'


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node('search_wiki_candidates', search_wiki_candidates)
    g.add_node('claude_pick_wiki',       claude_pick_wiki)
    g.add_node('fetch_categories',       fetch_categories)
    g.add_node('claude_pick_category',   claude_pick_category)
    g.add_node('scrape_characters',      scrape_characters)
    g.add_node('flag_no_scope',          flag_no_scope)
    g.add_node('flag_no_category',       flag_no_category)
    g.add_node('handle_error',           handle_error)

    g.set_entry_point('search_wiki_candidates')

    for src, dst in [
        ('search_wiki_candidates', 'claude_pick_wiki'),
        ('claude_pick_wiki',       'fetch_categories'),
        ('fetch_categories',       'claude_pick_category'),
    ]:
        g.add_conditional_edges(src, route_on_error, {
            'continue':     dst,
            'handle_error': 'handle_error',
        })

    g.add_conditional_edges('claude_pick_category', route_on_strategy, {
        'scrape_characters': 'scrape_characters',
        'flag_no_scope':     'flag_no_scope',
        'flag_no_category':  'flag_no_category',
        'handle_error':      'handle_error',
    })

    g.add_conditional_edges('scrape_characters', route_on_error, {
        'continue':     END,
        'handle_error': 'handle_error',
    })
    g.add_edge('flag_no_scope',    END)
    g.add_edge('flag_no_category', END)
    g.add_edge('handle_error',     END)

    return g.compile()


graph = build_graph()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_media_agent(media_id: int) -> None:
    media        = MediaRequest.objects.get(id=media_id)
    media.status = 'pr'
    media.save(update_fields=['status'])

    graph.invoke({
        'media_id':            media_id,
        'title':               media.title,
        'media_type':          media.media_type,
        'wiki_candidates':     [],
        'wiki_slug':           '',
        'wiki_url':            '',
        'is_umbrella_wiki':    False,
        'all_categories':      [],
        'chosen_category':     '',
        'metadata_categories': [],
        'resolution_strategy': None,
        'character_names':     [],
        'error':               None,
    })
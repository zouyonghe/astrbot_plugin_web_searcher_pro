# SearXNG Search Quality Design

## Background

The plugin already reaches a live SearXNG instance and returns usable results, but the current integration leaves search quality on the table.

Observed issues in the current code and live instance behavior:

- `services/searxng_service.py` forces `lang=zh` for every query, which hurts English and mixed-language technical searches.
- Non-image results are only deduplicated and truncated. There is no category-specific search strategy, fallback, or source-aware ranking.
- The exported `technical` tool uses the `technical` category directly, but the tested SearXNG instance returns stronger developer-oriented results from the `it` category.
- The live instance supports additional useful categories such as `map`, `files`, `social media`, and `books`, but the plugin does not expose them yet.

## Goal

Improve the plugin's real-world search quality by:

1. Choosing search language from the user's query instead of always forcing Chinese.
2. Adding category-specific search strategies and fallbacks.
3. Exposing more useful SearXNG categories as LLM tools.
4. Preserving the current plugin structure and AstrBot integration style.

## Live Instance Findings

The SearXNG instance at `http://192.168.50.20:8080` was queried directly during design work.

Key findings:

- `general` works well for ordinary web search.
- `news` works well and returns distinct news results.
- `science` produces meaningfully different results, including PubMed content.
- `technical` and `academic` often look too similar to `general` on this instance.
- `it` produces more relevant developer and troubleshooting results than `technical`.
- `map`, `files`, `social media`, and `books` are all available and return usable results.

## Chosen Approach

Use a lightweight search-strategy layer inside `SearxngService`.

This keeps the current architecture intact while making search behavior smarter. The strategy layer will decide:

- which SearXNG category to query first,
- which fallback categories to try,
- which search language to use,
- and how to lightly rank results by source and category fitness.

This is intentionally lighter than a full multi-query orchestrator. It improves quality without adding too much latency or complexity.

## Language Strategy

Introduce query-language detection based on the characters found in the user's query.

Initial heuristic:

- Chinese characters → `zh`
- Japanese kana or Japanese punctuation patterns → `ja`
- Korean Hangul → `ko`
- Mostly ASCII text with English keywords → `en`
- Mixed or unclear queries → `auto`

Rules:

- Technical and academic queries should prefer `en` when the query is clearly English.
- Chinese natural-language queries should prefer `zh`.
- If no language is clearly dominant, use `auto` instead of forcing a locale.

The design intentionally keeps this heuristic simple and deterministic. It should be easy to test and safe to refine later.

## Search Strategy by Tool

### Existing tools

- `searxng_web_search_general`
  - Primary category: `general`
  - Fallback: none
  - Language: detected from query

- `searxng_web_search_technical`
  - Primary category: `it`
  - Fallbacks: `technical`, `general`
  - Language: detected from query, with English preference for clearly English technical queries

- `searxng_web_search_academic`
  - Primary category: `academic`
  - Fallbacks: `science`, `general`
  - Language: detected from query

- `searxng_web_search_science`
  - Primary category: `science`
  - Fallbacks: `academic`, `general`
  - Language: detected from query

- `searxng_web_search_news`
  - Primary category: `news`
  - Fallback: `general`
  - Language: detected from query

- `searxng_web_search_music`
  - Primary category: `music`
  - Fallback: `general`
  - Language: detected from query

- `searxng_web_search_videos`
  - Primary category: `videos`
  - Fallback: `general`
  - Language: detected from query

- `searxng_web_search_images`
  - Keep current category behavior for this phase.
  - Image downloading and `HEAD` validation improvements are explicitly deferred.

### New tools

- `searxng_web_search_map`
  - Primary category: `map`
  - Fallback: `general`
  - Intended for places, landmarks, addresses, and geographic lookups

- `searxng_web_search_files`
  - Primary category: `files`
  - Fallback: `general`
  - Intended for PDFs, manuals, reports, and downloadable file discovery

- `searxng_web_search_social`
  - Primary category: `social media`
  - Fallback: `general`
  - Intended for communities, hashtags, social profiles, and social discussion discovery

- `searxng_web_search_books`
  - Primary category: `books`
  - Fallback: `general`
  - Intended for book discovery and bibliographic lookups

## Result Ranking

Add lightweight category-aware ranking after deduplication.

Examples:

- `technical`: prefer results from common developer sources such as Stack Overflow, GitHub, official docs, and similar technical sites.
- `academic` and `science`: prefer scholarly sources such as PubMed, arXiv, Semantic Scholar, DOI-based sources, and academic hosts.
- `news`: prefer recognized news engines and news-hosted URLs.
- `books`: prefer Open Library and book-oriented sources.

This ranking should be additive, not exclusive. If source recognition fails, preserve current result order rather than dropping data.

## Error Handling and Fallback Rules

- If a primary category returns no useful results, try fallback categories in order.
- Stop on the first non-empty result set.
- If all categories fail, return the existing empty message for the tool.
- If SearXNG rejects or silently degrades a category, the plugin still remains robust because fallback is handled in plugin code rather than assumed from the server.

## API and Structure Changes

Expected internal changes:

- Extend `SearxngService` to accept a richer search strategy object instead of only a plain category string.
- Add pure helper functions for:
  - query language detection,
  - strategy lookup,
  - category-aware result scoring.
- Keep `main.py` thin by continuing to call a shared helper for text categories.
- Add new `@llm_tool` methods in `main.py` for map, files, social, and books.
- Update user-facing docs to list the new tools.

## Validation Plan

Validation should include both logic checks and live-instance verification.

### Unit tests

- language detection for Chinese, English, Japanese, Korean, and mixed queries
- category strategy lookup for each tool
- fallback behavior when the first category is empty
- category-aware ranking rules

### Live verification against `http://192.168.50.20:8080`

- Chinese general query
- English technical query
- Chinese technical query
- academic/science query
- map query
- files query
- social query
- books query

## Non-Goals

- No image-search transport redesign in this phase
- No full multi-query parallel orchestration in this phase
- No changes to slash command names
- No change to GitHub, AUR, or webpage fetch behavior outside shared utility updates if needed

# SearXNG Tool Renaming Design

## Background

The plugin currently exports several generic `llm_tool` names such as `web_search` and `fetch_url`.
Those names are collision-prone in AstrBot when multiple plugins register similarly named tools.

## Goal

Refactor every exported `llm_tool` in this plugin to use a unique `searxng_*` namespace while preserving existing plugin behavior.

## Chosen Naming Scheme

Use descriptive names prefixed with `searxng_` and grouped by capability:

- `web_search` → `searxng_web_search_general`
- `web_search_images` → `searxng_web_search_images`
- `web_search_videos` → `searxng_web_search_videos`
- `web_search_news` → `searxng_web_search_news`
- `web_search_science` → `searxng_web_search_science`
- `web_search_music` → `searxng_web_search_music`
- `web_search_technical` → `searxng_web_search_technical`
- `web_search_academic` → `searxng_web_search_academic`
- `fetch_url` → `searxng_web_fetch_url`
- `github_search` → `searxng_github_search`

## Scope

Update:

1. `@llm_tool(...)` decorator names.
2. Internal activation/deactivation calls that still point at the old names.
3. User-facing documentation where these tool names are described as callable capabilities.

Do not change:

- Slash commands such as `/websearch` and `/github`.
- Plugin package name and repository metadata.
- Search behavior, HTTP logic, or response formatting.

## Compatibility Decision

This is a deliberate breaking change for tool identifiers only. The purpose is to eliminate runtime name collisions with other plugins. Slash commands remain stable to minimize user impact.

## Validation

Validation will be done by searching the repository to ensure no old exported tool names remain in decorator or activation/deactivation call sites.

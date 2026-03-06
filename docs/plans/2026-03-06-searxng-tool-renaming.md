# SearXNG Tool Renaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename all exported `llm_tool` identifiers to a unique `searxng_*` namespace so they no longer collide with tools from other AstrBot plugins.

**Architecture:** Keep runtime behavior unchanged and perform a narrow refactor in `main.py` only where tool identifiers are declared or activated. Update documentation to reflect the new exported names and validate with repository-wide searches.

**Tech Stack:** Python, AstrBot plugin decorators, ripgrep

---

### Task 1: Rename core web search tools

**Files:**
- Modify: `main.py`

**Step 1: Inspect current declarations**

Run: `rg -n "@llm_tool|activate_llm_tool|deactivate_llm_tool" main.py`
Expected: existing generic tool names are listed.

**Step 2: Rename the general and fetch tools**

Change:

- `web_search` → `searxng_web_search_general`
- `fetch_url` → `searxng_web_fetch_url`

Also update matching `activate_llm_tool(...)` and `deactivate_llm_tool(...)` call sites.

**Step 3: Verify replacements**

Run: `rg -n "web_search\"|fetch_url\"" main.py`
Expected: no remaining exported tool references in activation/deactivation or decorator positions.

### Task 2: Rename all category search tools

**Files:**
- Modify: `main.py`

**Step 1: Rename category decorators**

Change:

- `web_search_images` → `searxng_web_search_images`
- `web_search_videos` → `searxng_web_search_videos`
- `web_search_news` → `searxng_web_search_news`
- `web_search_science` → `searxng_web_search_science`
- `web_search_music` → `searxng_web_search_music`
- `web_search_technical` → `searxng_web_search_technical`
- `web_search_academic` → `searxng_web_search_academic`

**Step 2: Rename the GitHub helper tool**

Change:

- `github_search` → `searxng_github_search`

**Step 3: Verify replacements**

Run: `rg -n "web_search_images|web_search_videos|web_search_news|web_search_science|web_search_music|web_search_technical|web_search_academic|github_search" main.py`
Expected: only method names and slash-command names remain where intended.

### Task 3: Update docs and run final checks

**Files:**
- Modify: `README.md`

**Step 1: Document the new tool namespace**

Add a short note that exported LLM tools now use the `searxng_*` prefix to avoid collisions.

**Step 2: Run repository-wide verification**

Run: `rg -n "@llm_tool\(|activate_llm_tool|deactivate_llm_tool|searxng_"`
Expected: every exported tool name is namespaced.

**Step 3: Run targeted old-name search**

Run: `rg -n "@llm_tool\(\"(web_search|fetch_url|github_search|web_search_images|web_search_videos|web_search_news|web_search_science|web_search_music|web_search_technical|web_search_academic)\""`
Expected: no matches.

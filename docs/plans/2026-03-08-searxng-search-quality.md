# SearXNG Search Quality Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve search quality by detecting query language, adding category-specific fallback and ranking strategies, and exposing additional SearXNG search categories as LLM tools.

**Architecture:** Keep `main.py` as the thin AstrBot integration layer and move search intelligence into `services/searxng_service.py` plus small pure helper modules. Represent each tool's behavior as a strategy containing language selection, primary category, fallback categories, and lightweight ranking hints, then reuse a shared text-search entry path for both old and new tools.

**Tech Stack:** Python 3, AstrBot plugin API, SearXNG HTTP JSON API, pytest-style tests, pure helper functions for strategy and language detection.

---

### Task 1: Add query-language detection helpers

**Files:**
- Create: `utils/query_language.py`
- Test: `tests/test_query_language.py`

**Step 1: Write the failing test**

```python
from data.plugins.astrbot_plugin_web_searcher_pro.utils.query_language import detect_query_language


def test_detect_query_language_for_chinese_text():
    assert detect_query_language("怎么用 asyncio") == "zh"


def test_detect_query_language_for_english_text():
    assert detect_query_language("python asyncio tutorial") == "en"


def test_detect_query_language_for_japanese_text():
    assert detect_query_language("東京スカイツリー") == "ja"


def test_detect_query_language_for_korean_text():
    assert detect_query_language("파이썬 비동기") == "ko"


def test_detect_query_language_for_mixed_text_falls_back_to_auto():
    assert detect_query_language("OpenAI 最新 news") == "auto"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_query_language.py -q`
Expected: FAIL because the module or function does not exist yet.

**Step 3: Write minimal implementation**

Create `utils/query_language.py` with a pure function that:

- returns `zh` for CJK Han-heavy Chinese queries,
- returns `ja` when kana is present,
- returns `ko` when Hangul is present,
- returns `en` for clearly ASCII English queries,
- returns `auto` for mixed or unclear cases.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_query_language.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/query_language.py tests/test_query_language.py
git commit -m "feat: detect query language for searxng"
```

### Task 2: Add reusable search strategy definitions

**Files:**
- Create: `services/search_strategy.py`
- Test: `tests/test_search_strategy.py`

**Step 1: Write the failing test**

```python
from data.plugins.astrbot_plugin_web_searcher_pro.services.search_strategy import get_search_strategy


def test_technical_strategy_prefers_it_category():
    strategy = get_search_strategy("technical")
    assert strategy.primary_category == "it"
    assert strategy.fallback_categories == ["technical", "general"]


def test_map_strategy_uses_map_category():
    strategy = get_search_strategy("map")
    assert strategy.primary_category == "map"


def test_books_strategy_uses_books_category():
    strategy = get_search_strategy("books")
    assert strategy.primary_category == "books"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_search_strategy.py -q`
Expected: FAIL because the strategy module does not exist yet.

**Step 3: Write minimal implementation**

Create a small strategy dataclass and a lookup function with entries for:

- `general`
- `technical`
- `academic`
- `science`
- `news`
- `music`
- `videos`
- `map`
- `files`
- `social`
- `books`

Include fields for primary category, fallback categories, and a ranking profile key.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_search_strategy.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add services/search_strategy.py tests/test_search_strategy.py
git commit -m "feat: add searxng search strategies"
```

### Task 3: Add lightweight category-aware ranking helpers

**Files:**
- Create: `services/result_ranking.py`
- Test: `tests/test_result_ranking.py`

**Step 1: Write the failing test**

```python
from data.plugins.astrbot_plugin_web_searcher_pro.search_models import SearchResultItem
from data.plugins.astrbot_plugin_web_searcher_pro.services.result_ranking import rank_text_results


def test_technical_ranking_prefers_docs_and_developer_sources():
    items = [
        SearchResultItem("Forum post", "https://example.com/post", "", "", "", "", "duckduckgo", 0.9),
        SearchResultItem("Python docs", "https://docs.python.org/3/library/asyncio.html", "", "", "", "", "duckduckgo", 0.7),
    ]
    ranked = rank_text_results(items, profile="technical")
    assert ranked[0].url == "https://docs.python.org/3/library/asyncio.html"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_result_ranking.py -q`
Expected: FAIL because the ranking helper does not exist yet.

**Step 3: Write minimal implementation**

Create a ranking helper that:

- accepts a list of `SearchResultItem` values,
- applies additive bonuses by ranking profile,
- keeps the original result rather than dropping unknown sources,
- returns a re-ordered list.

Profiles should at least support `technical`, `academic`, `science`, `news`, `books`, and a neutral default.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_result_ranking.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add services/result_ranking.py tests/test_result_ranking.py
git commit -m "feat: rank searxng text results by category"
```

### Task 4: Refactor `SearxngService` to use strategy, language detection, and fallback

**Files:**
- Modify: `services/searxng_service.py`
- Modify: `tests/test_package_import.py`
- Create: `tests/test_searxng_service.py`

**Step 1: Write the failing test**

```python
import pytest

from data.plugins.astrbot_plugin_web_searcher_pro.search_models import SearchResult
from data.plugins.astrbot_plugin_web_searcher_pro.services.searxng_service import SearxngService


class DummyHttpClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def get_json(self, url, params=None, headers=None):
        self.calls.append((url, params))
        category = params["categories"]
        return 200, {"results": self.responses.get(category, [])}


@pytest.mark.asyncio
async def test_technical_search_falls_back_from_it_to_general_when_empty():
    client = DummyHttpClient({"it": [], "technical": [], "general": [{"title": "Doc", "url": "https://docs.python.org", "content": "", "engine": "duckduckgo"}]})
    service = SearxngService(client, "http://example.com")

    result = await service.search("python asyncio", category="technical", limit=5)

    assert isinstance(result, SearchResult)
    assert result.results[0].url == "https://docs.python.org"
    assert [params["categories"] for _, params in client.calls] == ["it", "technical", "general"]
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_searxng_service.py -q`
Expected: FAIL because `SearxngService` does not support strategy fallback yet.

**Step 3: Write minimal implementation**

Refactor `services/searxng_service.py` so that:

- `search()` resolves a strategy from the requested tool category key,
- language is determined by the new query-language helper,
- primary and fallback categories are tried in order,
- non-image text results are deduplicated, ranked, and truncated,
- image behavior stays unchanged for this phase.

Keep the public call shape stable for `main.py` where possible.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_searxng_service.py -q`
Expected: PASS.

**Step 5: Expand import stubs if needed**

If refactoring introduces new imports that break the package import test, extend `tests/test_package_import.py` stubs minimally so import-only verification still works.

**Step 6: Run relevant tests**

Run: `python3 -m pytest tests/test_package_import.py tests/test_query_language.py tests/test_search_strategy.py tests/test_result_ranking.py tests/test_searxng_service.py -q`
Expected: PASS.

**Step 7: Commit**

```bash
git add services/searxng_service.py tests/test_package_import.py tests/test_searxng_service.py
git add tests/test_query_language.py tests/test_search_strategy.py tests/test_result_ranking.py
git commit -m "feat: add searxng search fallback and language strategy"
```

### Task 5: Expose new LLM tools for map, files, social, and books

**Files:**
- Modify: `main.py`
- Test: `tests/test_package_import.py`

**Step 1: Write the failing test**

Add assertions to `tests/test_package_import.py` so the plugin exposes methods and docstrings for:

- `search_map`
- `search_files`
- `search_social`
- `search_books`

Also assert their parameter docs include `query (string):`.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_package_import.py -q`
Expected: FAIL because these tools do not exist yet.

**Step 3: Write minimal implementation**

Modify `main.py` to add new `@llm_tool(...)` methods:

- `@llm_tool("searxng_web_search_map")`
- `@llm_tool("searxng_web_search_files")`
- `@llm_tool("searxng_web_search_social")`
- `@llm_tool("searxng_web_search_books")`

Route each through the shared text-category helper and add proper docstrings for tool metadata.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_package_import.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add main.py tests/test_package_import.py
git commit -m "feat: add more searxng search tools"
```

### Task 6: Update README documentation for new behavior

**Files:**
- Modify: `README.md`

**Step 1: Write the failing doc check**

Manually identify missing documentation for:

- language-aware search behavior
- new search tools
- technical search using stronger developer-oriented results

**Step 2: Update the README**

Document:

- that search language is inferred from the query,
- the new tools `searxng_web_search_map`, `searxng_web_search_files`, `searxng_web_search_social`, and `searxng_web_search_books`,
- that technical search quality is improved by category strategy and fallback.

**Step 3: Verify the README mentions the new tools**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
required = [
    'searxng_web_search_map',
    'searxng_web_search_files',
    'searxng_web_search_social',
    'searxng_web_search_books',
    '根据查询语言',
]
missing = [item for item in required if item not in text]
assert not missing, missing
print('README verification passed')
PY`
Expected: PASS.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document smarter searxng search tools"
```

### Task 7: Run live verification against the target SearXNG instance

**Files:**
- No file changes required unless a live-verification helper script is needed

**Step 1: Run live checks**

Run a verification script against `http://192.168.50.20:8080` that checks:

- Chinese general query
- English technical query
- Chinese technical query
- academic query
- map query
- files query
- social query
- books query

Expected:

- requests succeed,
- strategy categories and fallback are exercised correctly,
- returned top results look category-appropriate,
- language is no longer pinned to `zh` for all queries.

**Step 2: Run the full targeted test set**

Run: `python3 -m pytest tests/test_package_import.py tests/test_query_language.py tests/test_search_strategy.py tests/test_result_ranking.py tests/test_searxng_service.py -q`
Expected: PASS.

**Step 3: Commit final verification-safe state**

```bash
git status --short
```

Expected: clean working tree.

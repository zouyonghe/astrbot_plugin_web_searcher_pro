# Aggressive Engineering Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild this plugin into a more maintainable, modular, and testable codebase while preserving a coherent external experience and filling in missing engineering capabilities.

**Architecture:** Keep `main.py` as the thin AstrBot integration layer and move search, GitHub, web fetch, image handling, config loading, and formatting into focused modules. Add a lightweight shared HTTP client abstraction, explicit error types, and pure helper functions so most logic can be validated without real network calls.

**Tech Stack:** Python, AstrBot plugin API, `aiohttp`, `pytest`, `readability-lxml`, `beautifulsoup4`, dataclasses

---

### Task 1: Create test scaffolding and lock current parsing behavior

**Files:**
- Create: `tests/test_search_models.py`
- Create: `tests/test_url_helpers.py`
- Create: `tests/conftest.py`
- Modify: `requirements.txt`

**Step 1: Write the failing model-string test**

```python
from search_models import SearchResult, SearchResultItem


def test_search_result_string_contains_title_url_and_content():
    result = SearchResult(
        results=[
            SearchResultItem(
                title="Example",
                url="https://example.com",
                img_src="",
                resolution="",
                iframe_src="",
                content="Snippet",
                engine="demo",
                score=1.0,
            )
        ]
    )

    text = str(result)

    assert "Example" in text
    assert "https://example.com" in text
    assert "Snippet" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_models.py::test_search_result_string_contains_title_url_and_content -v`
Expected: FAIL because `pytest` test scaffolding does not exist yet.

**Step 3: Add minimal test dependency and scaffolding**

```text
pytest
pytest-asyncio
```

```python
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_search_models.py::test_search_result_string_contains_title_url_and_content -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add requirements.txt tests/test_search_models.py tests/conftest.py
git commit -m "test: add initial plugin scaffolding"
```

### Task 2: Extract configuration and shared HTTP utilities

**Files:**
- Create: `config.py`
- Create: `services/http_client.py`
- Create: `errors.py`
- Create: `tests/test_config.py`
- Create: `tests/test_http_client.py`
- Modify: `main.py`

**Step 1: Write the failing config test**

```python
from config import PluginConfig


class DummyConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_plugin_config_reads_defaults():
    config = PluginConfig.from_astrbot(DummyConfig())

    assert config.searxng_api_url == "http://127.0.0.1:8080"
    assert config.enable_random_image is False
    assert config.enable_image_title is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_plugin_config_reads_defaults -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PluginConfig:
    searxng_api_url: str
    enable_random_image: bool
    enable_image_title: bool
    github_token: str

    @classmethod
    def from_astrbot(cls, config):
        return cls(
            searxng_api_url=config.get("searxng_api_url", "http://127.0.0.1:8080"),
            enable_random_image=config.get("enable_random_image", False),
            enable_image_title=config.get("enable_image_title", True),
            github_token=config.get("github_token", "").strip(),
        )
```

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RequestOptions:
    proxy: Optional[str] = None
    timeout: int = 10
```

**Step 4: Run focused tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add config.py services/http_client.py errors.py tests/test_config.py tests/test_http_client.py main.py
git commit -m "refactor: extract config and shared http utilities"
```

### Task 3: Extract pure URL and GitHub parsing helpers

**Files:**
- Create: `services/github_service.py`
- Create: `utils/url_helpers.py`
- Create: `tests/test_github_service.py`
- Create: `tests/test_url_helpers.py`
- Modify: `main.py`

**Step 1: Write the failing repository parsing test**

```python
from utils.url_helpers import extract_github_repo


def test_extract_github_repo_from_tree_url():
    repo = extract_github_repo("https://github.com/owner/repo/tree/main/src")
    assert repo == "owner/repo"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_url_helpers.py::test_extract_github_repo_from_tree_url -v`
Expected: FAIL with `ModuleNotFoundError`.

**Step 3: Write minimal implementation**

```python
import re
from urllib.parse import urlparse


GITHUB_REPO_PATTERN = re.compile(r"^/([\w\-]+/[\w\-.]+)")


def extract_github_repo(value: str):
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        if "github.com" not in parsed.netloc:
            return None
        match = GITHUB_REPO_PATTERN.match(parsed.path)
        return match.group(1) if match else None

    clone_match = re.match(r"^(?:git@github\.com:|https://github\.com/)([\w\-]+/[\w\-.]+?)(?:\.git)?$", value)
    if clone_match:
        return clone_match.group(1)

    if "/" in value and not value.startswith("http"):
        return value

    return None
```

**Step 4: Run test suite for URL/GitHub helpers**

Run: `pytest tests/test_url_helpers.py tests/test_github_service.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add services/github_service.py utils/url_helpers.py tests/test_github_service.py tests/test_url_helpers.py main.py
git commit -m "refactor: extract github and url parsing helpers"
```

### Task 4: Extract SearXNG and image services

**Files:**
- Create: `services/searxng_service.py`
- Create: `services/image_service.py`
- Create: `tests/test_searxng_service.py`
- Create: `tests/test_image_service.py`
- Modify: `search_models.py`
- Modify: `main.py`

**Step 1: Write the failing result-filter test**

```python
from search_models import SearchResult, SearchResultItem
from services.searxng_service import filter_results


def test_filter_results_limits_non_image_categories():
    result = SearchResult(
        results=[
            SearchResultItem("A", "u1", "", "", "", "", "demo", 1.0),
            SearchResultItem("B", "u2", "", "", "", "", "demo", 0.9),
        ]
    )

    filtered = filter_results(result, category="general", limit=1, engines=None)

    assert len(filtered.results) == 1
    assert filtered.results[0].title == "A"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_searxng_service.py::test_filter_results_limits_non_image_categories -v`
Expected: FAIL with missing module or function.

**Step 3: Write minimal implementation**

```python
from search_models import SearchResult


def filter_results(result: SearchResult, category: str, limit: int, engines=None) -> SearchResult:
    items = result.results
    if engines:
        items = [item for item in items if item.engine in engines]
    if category != "images":
        result.results = items[:limit]
        return result
    result.results = items[:30]
    return result
```

**Step 4: Run focused tests to verify they pass**

Run: `pytest tests/test_searxng_service.py tests/test_image_service.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add services/searxng_service.py services/image_service.py tests/test_searxng_service.py tests/test_image_service.py search_models.py main.py
git commit -m "refactor: extract searxng and image services"
```

### Task 5: Extract web fetch service and unify plugin entrypoints

**Files:**
- Create: `services/web_fetch_service.py`
- Create: `formatters/search_formatter.py`
- Create: `tests/test_web_fetch_service.py`
- Create: `tests/test_search_formatter.py`
- Modify: `main.py`

**Step 1: Write the failing formatter test**

```python
from search_models import SearchResult, SearchResultItem
from formatters.search_formatter import format_search_result


def test_format_search_result_returns_no_results_message():
    result = SearchResult(results=[])
    assert format_search_result(result, empty_message="No info") == "No info"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_formatter.py::test_format_search_result_returns_no_results_message -v`
Expected: FAIL because formatter module does not exist.

**Step 3: Write minimal implementation**

```python
from search_models import SearchResult


def format_search_result(result: SearchResult, empty_message: str) -> str:
    if not result or not result.results:
        return empty_message
    return str(result)
```

**Step 4: Run focused tests to verify they pass**

Run: `pytest tests/test_web_fetch_service.py tests/test_search_formatter.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add services/web_fetch_service.py formatters/search_formatter.py tests/test_web_fetch_service.py tests/test_search_formatter.py main.py
git commit -m "refactor: extract web fetch and formatting services"
```

### Task 6: Fill missing feature gaps and stabilize command behavior

**Files:**
- Modify: `main.py`
- Modify: `README.md`
- Modify: `_conf_schema.json`
- Create: `tests/test_command_behavior.py`

**Step 1: Write the failing behavior test**

```python
from utils.url_helpers import extract_github_repo


def test_invalid_non_http_url_is_rejected():
    assert extract_github_repo("not a url") is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_command_behavior.py::test_invalid_non_http_url_is_rejected -v`
Expected: FAIL because the test file and behavior assertions are not wired yet.

**Step 3: Write minimal implementation**

```python
def normalize_user_error(message: str) -> str:
    return message.strip() if message else "Request failed."
```

Implement in this task:
- unify invalid URL handling
- unify empty-result responses
- document supported tools and commands accurately
- either implement README-promised missing capability or remove stale documentation if capability is intentionally out of scope

**Step 4: Run targeted tests to verify they pass**

Run: `pytest tests/test_command_behavior.py tests/test_url_helpers.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add main.py README.md _conf_schema.json tests/test_command_behavior.py
git commit -m "feat: fill missing behaviors and unify command responses"
```

### Task 7: Final integration verification

**Files:**
- Modify: `main.py`
- Modify: `README.md`

**Step 1: Run the full test suite**

Run: `pytest -v`
Expected: PASS with all tests green.

**Step 2: Run structural verification searches**

Run: `rg -n "ClientSession\(" main.py services`
Expected: only centralized HTTP client code creates sessions.

Run: `rg -n "except:" main.py services utils`
Expected: no bare `except:` remains.

**Step 3: Run quick repository review**

Run: `git diff --stat`
Expected: modularized changes limited to the planned files.

**Step 4: Commit**

```bash
git add main.py README.md config.py errors.py services utils formatters tests
 git commit -m "refactor: modularize web searcher plugin"
```

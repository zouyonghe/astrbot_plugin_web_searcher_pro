import pytest

from astrbot_plugin_web_searcher_pro.services.searxng_service import SearxngService


def _text_result(title: str, url: str, *, content: str = "", engine: str = "duckduckgo", score: float = 0.0) -> dict:
    return {
        "title": title,
        "url": url,
        "content": content,
        "engine": engine,
        "score": score,
    }


def _image_result(
    title: str,
    url: str,
    img_src: str,
    *,
    resolution: str,
    engine: str = "google images",
    score: float = 0.0,
) -> dict:
    return {
        "title": title,
        "url": url,
        "img_src": img_src,
        "resolution": resolution,
        "engine": engine,
        "score": score,
    }


class DummyHttpClient:
    def __init__(self, responses, *, valid_images: set[str] | None = None):
        self.responses = responses
        self.valid_images = valid_images or set()
        self.calls: list[tuple[str, dict]] = []
        self.head_calls: list[str] = []

    async def get_json(self, url, *, params=None, headers=None):
        self.calls.append((url, dict(params or {})))
        category = str((params or {}).get("categories", ""))
        payload = self.responses.get(category, [])
        if isinstance(payload, tuple) and len(payload) == 2:
            status, data = payload
            return status, data
        return 200, {"results": list(payload)}

    async def head_ok(self, url: str) -> bool:
        self.head_calls.append(url)
        return url in self.valid_images


class ErroringHttpClient(DummyHttpClient):
    def __init__(self, error: Exception):
        super().__init__({})
        self.error = error

    async def get_json(self, url, *, params=None, headers=None):
        raise self.error


@pytest.mark.asyncio
async def test_technical_search_falls_back_from_it_to_technical_to_general_when_needed():
    client = DummyHttpClient(
        {
            "it": [],
            "technical": [],
            "general": [
                _text_result(
                    "Python docs",
                    "https://docs.python.org/3/library/asyncio.html",
                    score=0.6,
                )
            ],
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("python asyncio", category="technical", limit=5)

    assert [item.url for item in result.results] == ["https://docs.python.org/3/library/asyncio.html"]
    assert [params["categories"] for _, params in client.calls] == ["it", "technical", "general"]


@pytest.mark.asyncio
async def test_text_search_uses_detected_query_language_in_request_params():
    client = DummyHttpClient(
        {
            "general": [
                _text_result(
                    "Asyncio tutorial",
                    "https://docs.python.org/3/library/asyncio.html",
                    score=0.5,
                )
            ]
        }
    )
    service = SearxngService(client, "http://example.com")

    await service.search("python asyncio tutorial", category="general", limit=3)

    assert client.calls[0][1]["lang"] == "en"


@pytest.mark.asyncio
async def test_text_search_does_not_fallback_or_aggregate_after_primary_results():
    client = DummyHttpClient(
        {
            "it": [
                _text_result("Forum post", "https://example.com/post", score=0.9),
            ],
            "technical": [
                _text_result(
                    "Python docs",
                    "https://docs.python.org/3/library/asyncio.html",
                    score=0.7,
                ),
                _text_result("Forum post", "https://example.com/post", score=0.85),
            ],
            "general": [],
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("python asyncio", category="technical", limit=2)

    assert [item.url for item in result.results] == ["https://example.com/post"]
    assert [params["categories"] for _, params in client.calls] == ["it"]


@pytest.mark.asyncio
async def test_text_search_deduplicates_ranks_and_truncates_results_within_selected_category():
    client = DummyHttpClient(
        {
            "it": [],
            "technical": [
                _text_result(
                    "Python docs",
                    "https://docs.python.org/3/library/asyncio.html",
                    score=0.7,
                ),
                _text_result(
                    "Python docs",
                    "https://docs.python.org/3/library/asyncio.html",
                    score=0.6,
                ),
                _text_result("Forum post", "https://example.com/post", score=0.95),
                _text_result("GitHub repo", "https://github.com/python/cpython", score=0.76),
            ],
            "general": [],
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("python asyncio", category="technical", limit=2)

    assert [item.url for item in result.results] == [
        "https://docs.python.org/3/library/asyncio.html",
        "https://github.com/python/cpython",
    ]
    assert [params["categories"] for _, params in client.calls] == ["it", "technical"]
    assert len(result.results) == 2


@pytest.mark.asyncio
async def test_text_search_deduplicates_same_url_when_titles_differ():
    client = DummyHttpClient(
        {
            "general": [
                _text_result("Original title", "https://example.com/article", score=0.4),
                _text_result("Retitled article", "https://example.com/article", score=0.6),
            ]
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("example article", category="general", limit=5)

    assert len(result.results) == 1
    assert result.results[0].url == "https://example.com/article"


@pytest.mark.asyncio
async def test_text_search_keeps_higher_scored_later_duplicate():
    client = DummyHttpClient(
        {
            "general": [
                _text_result("Weaker first copy", "https://example.com/article", score=0.2),
                _text_result("Better later copy", "https://example.com/article", score=0.8),
            ]
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("example article", category="general", limit=5)

    assert len(result.results) == 1
    assert result.results[0].title == "Better later copy"
    assert result.results[0].score == 0.8


@pytest.mark.asyncio
async def test_text_search_keeps_first_duplicate_when_scores_tie():
    client = DummyHttpClient(
        {
            "general": [
                _text_result("First copy", "https://example.com/article", score=0.8),
                _text_result("Second copy", "https://example.com/article", score=0.8),
            ]
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("example article", category="general", limit=5)

    assert len(result.results) == 1
    assert result.results[0].title == "First copy"
    assert result.results[0].score == 0.8


@pytest.mark.asyncio
async def test_image_search_keeps_image_pipeline_separate_from_text_ranking():
    client = DummyHttpClient(
        {
            "images": [
                _image_result(
                    "Docs image",
                    "https://docs.python.org/3/library/asyncio.html",
                    "https://images.example/docs.png",
                    resolution="100x100",
                    score=1.0,
                ),
                _image_result(
                    "Large generic image",
                    "https://example.com/images/asyncio",
                    "https://images.example/large.png",
                    resolution="200x200",
                    score=0.5,
                ),
            ]
        },
        valid_images={"https://images.example/docs.png", "https://images.example/large.png"},
    )
    service = SearxngService(client, "http://example.com", image_result_limit=2)

    result = await service.search(
        "python asyncio",
        category="images",
        limit=2,
        engines=["google images"],
    )

    assert [params["categories"] for _, params in client.calls] == ["images"]
    assert client.head_calls == [
        "https://images.example/docs.png",
        "https://images.example/large.png",
    ]
    assert [item.img_src for item in result.results] == [
        "https://images.example/large.png",
        "https://images.example/docs.png",
    ]


@pytest.mark.asyncio
async def test_image_search_respects_per_call_limit():
    client = DummyHttpClient(
        {
            "images": [
                _image_result(
                    "Large image",
                    "https://example.com/images/1",
                    "https://images.example/1.png",
                    resolution="300x300",
                    score=0.4,
                ),
                _image_result(
                    "Medium image",
                    "https://example.com/images/2",
                    "https://images.example/2.png",
                    resolution="200x200",
                    score=0.5,
                ),
            ]
        },
        valid_images={"https://images.example/1.png", "https://images.example/2.png"},
    )
    service = SearxngService(client, "http://example.com", image_result_limit=5)

    result = await service.search("python asyncio", category="images", limit=1)

    assert len(result.results) == 1
    assert result.results[0].img_src == "https://images.example/1.png"


@pytest.mark.asyncio
async def test_search_returns_empty_result_for_malformed_results_payload():
    client = DummyHttpClient(
        {
            "general": (200, {"results": {"title": "not-a-list"}}),
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("example article", category="general", limit=5)

    assert result.is_empty


@pytest.mark.asyncio
async def test_search_returns_configuration_error_for_403_response():
    client = DummyHttpClient(
        {
            "news": (403, {"error": "Forbidden"}),
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("today news", category="news", limit=5)

    assert result.is_empty
    assert result.error_message == (
        "SearXNG returned 403 Forbidden. Please check whether the /search JSON interface is enabled and reachable."
    )


@pytest.mark.asyncio
async def test_search_falls_back_when_result_item_has_invalid_score_value():
    client = DummyHttpClient(
        {
            "it": (200, {"results": [{"title": "Broken item", "url": "https://broken.example", "score": "not-a-number"}]}),
            "technical": [
                _text_result("Python docs", "https://docs.python.org/3/library/asyncio.html", score=0.6),
            ],
            "general": [],
        }
    )
    service = SearxngService(client, "http://example.com")

    result = await service.search("python asyncio", category="technical", limit=5)

    assert [item.url for item in result.results] == ["https://docs.python.org/3/library/asyncio.html"]
    assert [params["categories"] for _, params in client.calls] == ["it", "technical"]


@pytest.mark.asyncio
async def test_search_returns_empty_result_when_get_json_raises_value_error():
    service = SearxngService(ErroringHttpClient(ValueError("bad json")), "http://example.com")

    result = await service.search("example article", category="general", limit=5)

    assert result.is_empty


@pytest.mark.asyncio
async def test_search_returns_empty_result_when_get_json_raises_type_error():
    service = SearxngService(ErroringHttpClient(TypeError("bad payload")), "http://example.com")

    result = await service.search("example article", category="general", limit=5)

    assert result.is_empty

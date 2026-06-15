from astrbot_plugin_web_searcher_pro.formatters.search_formatter import format_image_results, format_search_result
from astrbot_plugin_web_searcher_pro.search_models import SearchResult, SearchResultItem


def test_format_search_result_returns_empty_message_for_empty_result():
    result = SearchResult(results=[])
    assert format_search_result(result, empty_message="No info") == "No info"


def test_format_search_result_prefers_upstream_error_message():
    result = SearchResult(results=[], error_message="SearXNG returned 403.")

    assert format_search_result(result, empty_message="No info") == "SearXNG returned 403."


def test_format_image_results_includes_image_url_for_model():
    item = SearchResultItem(
        title="Apple logo",
        url="https://example.com/page",
        img_src="https://example.com/apple.jpg",
        resolution="1024x768",
        iframe_src="",
        content="A red apple image",
        engine="demo",
        score=1.0,
    )

    formatted = format_image_results([item])

    assert "Apple logo" in formatted
    assert "Image: https://example.com/apple.jpg" in formatted
    assert "Resolution: 1024x768" in formatted

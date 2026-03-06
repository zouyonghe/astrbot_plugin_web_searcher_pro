from astrbot_plugin_web_searcher_pro.search_models import SearchResult, SearchResultItem
from astrbot_plugin_web_searcher_pro.services.searxng_service import filter_results


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


def test_filter_results_applies_engine_filter():
    result = SearchResult(
        results=[
            SearchResultItem("A", "u1", "", "", "", "", "keep", 1.0),
            SearchResultItem("B", "u2", "", "", "", "", "drop", 0.9),
        ]
    )

    filtered = filter_results(result, category="general", limit=5, engines=["keep"])

    assert [item.engine for item in filtered.results] == ["keep"]

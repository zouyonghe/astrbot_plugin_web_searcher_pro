from astrbot_plugin_web_searcher_pro.services.image_service import parse_resolution, deduplicate_image_results
from astrbot_plugin_web_searcher_pro.search_models import SearchResultItem


def test_parse_resolution_supports_lowercase_x():
    assert parse_resolution("1920x1080") == (1920, 1080)


def test_deduplicate_image_results_removes_duplicate_urls():
    items = [
        SearchResultItem("A", "u1", "img", "", "", "", "demo", 1.0),
        SearchResultItem("B", "u2", "img", "", "", "", "demo", 0.8),
    ]

    deduped = deduplicate_image_results(items)

    assert len(deduped) == 1

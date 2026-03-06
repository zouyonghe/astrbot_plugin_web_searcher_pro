from astrbot_plugin_web_searcher_pro.formatters.search_formatter import format_search_result
from astrbot_plugin_web_searcher_pro.search_models import SearchResult


def test_format_search_result_returns_empty_message_for_empty_result():
    result = SearchResult(results=[])
    assert format_search_result(result, empty_message="No info") == "No info"

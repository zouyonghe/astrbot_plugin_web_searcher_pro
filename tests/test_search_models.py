from astrbot_plugin_web_searcher_pro.search_models import SearchResult, SearchResultItem


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

from astrbot_plugin_web_searcher_pro.search_models import SearchResultItem
from astrbot_plugin_web_searcher_pro.services.result_ranking import rank_text_results


def test_technical_ranking_prefers_docs_and_developer_sources():
    items = [
        SearchResultItem(
            title="Forum post",
            url="https://example.com/post",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.9,
        ),
        SearchResultItem(
            title="Python docs",
            url="https://docs.python.org/3/library/asyncio.html",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.7,
        ),
    ]

    ranked = rank_text_results(items, profile="technical")

    assert ranked[0].url == "https://docs.python.org/3/library/asyncio.html"


def test_unknown_sources_are_retained_in_default_ranking():
    items = [
        SearchResultItem(
            title="Unknown source",
            url="https://unknown.example/article",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.4,
        ),
        SearchResultItem(
            title="Known source",
            url="https://news.ycombinator.com/item?id=1",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.3,
        ),
    ]

    ranked = rank_text_results(items)

    assert len(ranked) == 2
    assert {item.url for item in ranked} == {
        "https://unknown.example/article",
        "https://news.ycombinator.com/item?id=1",
    }


def test_default_ranking_keeps_score_order_without_source_bonus():
    items = [
        SearchResultItem(
            title="Reuters",
            url="https://reuters.com/world/story",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.6,
        ),
        SearchResultItem(
            title="Generic",
            url="https://example.com/story",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.7,
        ),
    ]

    ranked = rank_text_results(items, profile="default")

    assert [item.url for item in ranked] == [
        "https://example.com/story",
        "https://reuters.com/world/story",
    ]


def test_academic_profile_prefers_academic_sources():
    items = [
        SearchResultItem(
            title="Generic article",
            url="https://example.com/llm-paper-summary",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.8,
        ),
        SearchResultItem(
            title="arXiv paper",
            url="https://arxiv.org/abs/1706.03762",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.5,
        ),
    ]

    ranked = rank_text_results(items, profile="academic")

    assert ranked[0].url == "https://arxiv.org/abs/1706.03762"


def test_science_profile_prefers_science_sources():
    items = [
        SearchResultItem(
            title="Generic article",
            url="https://example.com/space-news",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.75,
        ),
        SearchResultItem(
            title="NASA article",
            url="https://www.nasa.gov/missions/webb/",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.45,
        ),
    ]

    ranked = rank_text_results(items, profile="science")

    assert ranked[0].url == "https://www.nasa.gov/missions/webb/"


def test_news_profile_prefers_news_sources():
    items = [
        SearchResultItem(
            title="Generic blog",
            url="https://example.com/breaking-story",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.8,
        ),
        SearchResultItem(
            title="Reuters story",
            url="https://www.reuters.com/world/europe/story/",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.4,
        ),
    ]

    ranked = rank_text_results(items, profile="news")

    assert ranked[0].url == "https://www.reuters.com/world/europe/story/"


def test_news_profile_does_not_reward_spoofed_reuters_domain():
    items = [
        SearchResultItem(
            title="Spoofed Reuters",
            url="https://reuters.com.evil.example/story",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.5,
        ),
        SearchResultItem(
            title="Generic higher score",
            url="https://example.com/real-story",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.6,
        ),
    ]

    ranked = rank_text_results(items, profile="news")

    assert ranked[0].url == "https://example.com/real-story"


def test_technical_profile_does_not_reward_non_subdomain_github_spoof():
    items = [
        SearchResultItem(
            title="Spoofed GitHub",
            url="https://notgithub.com/project",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.5,
        ),
        SearchResultItem(
            title="Generic higher score",
            url="https://example.com/project",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.6,
        ),
    ]

    ranked = rank_text_results(items, profile="technical")

    assert ranked[0].url == "https://example.com/project"


def test_books_profile_prefers_book_domains():
    items = [
        SearchResultItem(
            title="General article",
            url="https://example.com/book-review",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.8,
        ),
        SearchResultItem(
            title="Open Library",
            url="https://openlibrary.org/works/OL45883W",
            img_src="",
            resolution="",
            iframe_src="",
            content="",
            engine="duckduckgo",
            score=0.6,
        ),
    ]

    ranked = rank_text_results(items, profile="books")

    assert ranked[0].url == "https://openlibrary.org/works/OL45883W"

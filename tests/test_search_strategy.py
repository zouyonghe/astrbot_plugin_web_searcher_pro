from astrbot_plugin_web_searcher_pro.services.search_strategy import get_search_strategy


EXPECTED_STRATEGIES = {
    "general": ("general", (), "default", "detect_query"),
    "technical": ("it", ("technical", "general"), "technical", "detect_query"),
    "academic": ("academic", ("science", "general"), "academic", "detect_query"),
    "science": ("science", ("academic", "general"), "science", "detect_query"),
    "news": ("news", ("general",), "news", "detect_query"),
    "music": ("music", ("general",), "default", "detect_query"),
    "videos": ("videos", ("general",), "default", "detect_query"),
    "map": ("map", ("general",), "default", "detect_query"),
    "files": ("files", ("general",), "default", "detect_query"),
    "social": ("social media", ("general",), "default", "detect_query"),
    "books": ("books", ("general",), "books", "detect_query"),
    "images": ("images", (), "default", "detect_query"),
}


def test_all_supported_keys_match_expected_strategy_definitions():
    for key, expected in EXPECTED_STRATEGIES.items():
        strategy = get_search_strategy(key)

        assert (
            strategy.primary_category,
            strategy.fallback_categories,
            strategy.ranking_profile,
            strategy.language_mode,
        ) == expected


def test_strategy_fallback_categories_are_immutable_tuples():
    strategy = get_search_strategy("technical")

    assert isinstance(strategy.fallback_categories, tuple)

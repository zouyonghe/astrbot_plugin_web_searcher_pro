from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SearchStrategy:
    primary_category: str
    fallback_categories: tuple[str, ...] = field(default_factory=tuple)
    ranking_profile: str = "default"
    language_mode: str = "detect_query"


_SEARCH_STRATEGIES = {
    "general": SearchStrategy(
        primary_category="general",
        fallback_categories=(),
        ranking_profile="default",
    ),
    "technical": SearchStrategy(
        primary_category="it",
        fallback_categories=("technical", "general"),
        ranking_profile="technical",
    ),
    "academic": SearchStrategy(
        primary_category="academic",
        fallback_categories=("science", "general"),
        ranking_profile="academic",
    ),
    "science": SearchStrategy(
        primary_category="science",
        fallback_categories=("academic", "general"),
        ranking_profile="science",
    ),
    "news": SearchStrategy(
        primary_category="news",
        fallback_categories=("general",),
        ranking_profile="news",
    ),
    "music": SearchStrategy(
        primary_category="music",
        fallback_categories=("general",),
        ranking_profile="default",
    ),
    "videos": SearchStrategy(
        primary_category="videos",
        fallback_categories=("general",),
        ranking_profile="default",
    ),
    "map": SearchStrategy(
        primary_category="map",
        fallback_categories=("general",),
        ranking_profile="default",
    ),
    "files": SearchStrategy(
        primary_category="files",
        fallback_categories=("general",),
        ranking_profile="default",
    ),
    "social": SearchStrategy(
        primary_category="social media",
        fallback_categories=("general",),
        ranking_profile="default",
    ),
    "books": SearchStrategy(
        primary_category="books",
        fallback_categories=("general",),
        ranking_profile="books",
    ),
    "images": SearchStrategy(
        primary_category="images",
        fallback_categories=(),
        ranking_profile="default",
    ),
}


def get_search_strategy(category: str) -> SearchStrategy:
    key = str(category or "general").strip().lower()
    return _SEARCH_STRATEGIES.get(key, _SEARCH_STRATEGIES["general"])

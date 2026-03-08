from urllib.parse import urlparse

from ..search_models import SearchResultItem


_PROFILE_PATTERNS = {
    "technical": (
        ("docs.python.org", 0.5),
        ("developer.mozilla.org", 0.45),
        ("readthedocs.io", 0.35),
        ("stackoverflow.com", 0.25),
        ("github.com", 0.2),
    ),
    "academic": (
        ("scholar.google.com", 0.4),
        ("arxiv.org", 0.45),
        ("doi.org", 0.35),
        ("ncbi.nlm.nih.gov", 0.35),
        ("researchgate.net", 0.2),
    ),
    "science": (
        ("nature.com", 0.45),
        ("science.org", 0.45),
        ("scientificamerican.com", 0.3),
        ("nasa.gov", 0.35),
        ("ncbi.nlm.nih.gov", 0.3),
    ),
    "news": (
        ("reuters.com", 0.45),
        ("apnews.com", 0.45),
        ("bbc.com", 0.3),
        ("nytimes.com", 0.25),
    ),
    "books": (
        ("openlibrary.org", 0.5),
        ("books.google.com", 0.4),
        ("goodreads.com", 0.25),
        ("worldcat.org", 0.35),
    ),
    "default": (),
}


def _matches_domain(host: str, domain: str) -> bool:
    normalized_host = host.split(":", 1)[0].rstrip(".")
    normalized_domain = domain.rstrip(".")
    return normalized_host == normalized_domain or normalized_host.endswith("." + normalized_domain)


def _bonus_for_url(url: str, profile: str) -> float:
    host = urlparse(url).netloc.lower()
    bonus = 0.0
    for pattern, value in _PROFILE_PATTERNS.get(profile, _PROFILE_PATTERNS["default"]):
        if _matches_domain(host, pattern):
            bonus += value
    return bonus


def rank_text_results(items: list[SearchResultItem], profile: str = "default") -> list[SearchResultItem]:
    normalized_profile = str(profile or "default").strip().lower() or "default"
    scored_items = []
    for index, item in enumerate(items):
        score = float(item.score) + _bonus_for_url(item.url, normalized_profile)
        scored_items.append((score, -index, item))
    scored_items.sort(reverse=True)
    return [item for _, _, item in scored_items]

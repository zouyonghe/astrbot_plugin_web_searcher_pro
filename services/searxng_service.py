import asyncio
from collections.abc import Mapping
from typing import Iterable

from ..errors import ExternalServiceError
from ..search_models import SearchResult, SearchResultItem
from ..utils.query_language import detect_query_language
from .http_client import HttpClient
from .image_service import rank_image_results
from .result_ranking import rank_text_results
from .search_strategy import get_search_strategy


def deduplicate_results(items: Iterable[SearchResultItem]) -> list[SearchResultItem]:
    best_by_url: dict[str, SearchResultItem] = {}
    order_by_url: list[str] = []
    for item in items:
        key = item.url.strip()
        if not key:
            key = f"__untitled__:{len(order_by_url)}"
        existing = best_by_url.get(key)
        if existing is None:
            best_by_url[key] = item
            order_by_url.append(key)
            continue
        if item.score > existing.score:
            best_by_url[key] = item
    return [best_by_url[key] for key in order_by_url]


def filter_results(
    result: SearchResult,
    category: str,
    limit: int,
    engines: list[str] | None = None,
    *,
    ranking_profile: str = "default",
) -> SearchResult:
    items = list(result.results)
    if engines:
        items = [item for item in items if item.engine in engines]
    items = deduplicate_results(items)
    if category != "images":
        items = rank_text_results(items, profile=ranking_profile)
        return SearchResult(results=items[:limit])
    return SearchResult(results=items[:limit])


class SearxngService:
    def __init__(self, http_client: HttpClient, base_url: str, *, image_candidate_limit: int = 120, image_result_limit: int = 12):
        self.http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.image_candidate_limit = image_candidate_limit
        self.image_result_limit = image_result_limit

    async def search(self, query: str, *, category: str = "general", limit: int = 5, engines: list[str] | None = None) -> SearchResult:
        if category != "images":
            return await self._search_text(query, category=category, limit=limit, engines=engines)

        result = await self._request_search(
            query,
            category=category,
            limit=max(limit, self.image_candidate_limit),
            lang="zh",
        )
        if result.is_empty:
            return SearchResult()
        return await self._filter_image_results(result, engines=engines, limit=limit)

    async def _request_search(self, query: str, *, category: str, limit: int, lang: str) -> SearchResult:
        params = {
            "q": query,
            "categories": category,
            "format": "json",
            "lang": lang,
            "limit": limit,
        }
        try:
            status, data = await self.http_client.get_json(f"{self.base_url}/search", params=params)
        except (ExternalServiceError, TypeError, ValueError):
            return SearchResult()
        if status != 200 or not isinstance(data, dict):
            return SearchResult()
        raw_results = data.get("results", [])
        if not isinstance(raw_results, list) or not all(isinstance(item, Mapping) for item in raw_results):
            return SearchResult()
        try:
            return SearchResult.from_iterable(raw_results)
        except (TypeError, ValueError):
            return SearchResult()

    async def _search_text(self, query: str, *, category: str, limit: int, engines: list[str] | None = None) -> SearchResult:
        strategy = get_search_strategy(category)
        requested_language = detect_query_language(query)

        ordered_categories = []
        seen_categories: set[str] = set()
        for current_category in (strategy.primary_category, *strategy.fallback_categories):
            if not current_category or current_category in seen_categories:
                continue
            seen_categories.add(current_category)
            ordered_categories.append(current_category)

        for current_category in ordered_categories:
            result = await self._request_search(
                query,
                category=current_category,
                limit=limit,
                lang=requested_language,
            )
            filtered = filter_results(
                result,
                category=category,
                limit=limit,
                engines=engines,
                ranking_profile=strategy.ranking_profile,
            )
            if filtered.results:
                return filtered

        return SearchResult()

    async def _filter_image_results(self, result: SearchResult, engines: list[str] | None = None, limit: int | None = None) -> SearchResult:
        items = list(result.results)
        if engines:
            items = [item for item in items if item.engine in engines]
        items = [item for item in items if item.img_src][: self.image_candidate_limit]
        checks = await asyncio.gather(*(self.http_client.head_ok(item.img_src) for item in items))
        valid_items = [item for item, is_valid in zip(items, checks) if is_valid]
        ranked = rank_image_results(valid_items)
        result_limit = self.image_result_limit if limit is None else min(limit, self.image_result_limit)
        return SearchResult(results=ranked[:result_limit])

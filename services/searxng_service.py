import asyncio
from typing import Iterable

from ..search_models import SearchResult, SearchResultItem
from ..errors import ExternalServiceError
from .http_client import HttpClient
from .image_service import rank_image_results


def deduplicate_results(items: Iterable[SearchResultItem]) -> list[SearchResultItem]:
    seen: set[tuple[str, str]] = set()
    deduped: list[SearchResultItem] = []
    for item in items:
        key = (item.url, item.title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def filter_results(result: SearchResult, category: str, limit: int, engines: list[str] | None = None) -> SearchResult:
    items = list(result.results)
    if engines:
        items = [item for item in items if item.engine in engines]
    items = deduplicate_results(items)
    if category != "images":
        return SearchResult(results=items[:limit])
    return SearchResult(results=items[:limit])


class SearxngService:
    def __init__(self, http_client: HttpClient, base_url: str, *, image_candidate_limit: int = 120, image_result_limit: int = 12):
        self.http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.image_candidate_limit = image_candidate_limit
        self.image_result_limit = image_result_limit

    async def search(self, query: str, *, category: str = "general", limit: int = 5, engines: list[str] | None = None) -> SearchResult:
        params = {
            "q": query,
            "categories": category,
            "format": "json",
            "lang": "zh",
            "limit": max(limit, self.image_candidate_limit if category == "images" else limit),
        }
        try:
            status, data = await self.http_client.get_json(f"{self.base_url}/search", params=params)
        except ExternalServiceError:
            return SearchResult()
        if status != 200 or not isinstance(data, dict):
            return SearchResult()

        result = SearchResult.from_iterable(data.get("results", []))
        if category != "images":
            return filter_results(result, category=category, limit=limit, engines=engines)
        return await self._filter_image_results(result, engines=engines)

    async def _filter_image_results(self, result: SearchResult, engines: list[str] | None = None) -> SearchResult:
        items = list(result.results)
        if engines:
            items = [item for item in items if item.engine in engines]
        items = [item for item in items if item.img_src][: self.image_candidate_limit]
        checks = await asyncio.gather(*(self.http_client.head_ok(item.img_src) for item in items))
        valid_items = [item for item, is_valid in zip(items, checks) if is_valid]
        ranked = rank_image_results(valid_items)
        return SearchResult(results=ranked[: self.image_result_limit])

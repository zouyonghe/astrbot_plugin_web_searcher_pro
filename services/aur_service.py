from ..errors import ExternalServiceError
from ..formatters.search_formatter import format_aur_exact_result, format_aur_results
from .http_client import HttpClient


class AurService:
    SEARCH_URL = "https://aur.archlinux.org/rpc/v5/search"

    def __init__(self, http_client: HttpClient):
        self.http_client = http_client

    async def search(self, query: str) -> str:
        if len(query.strip()) < 2:
            return "Search query must be at least 2 characters long."

        try:
            status, data = await self.http_client.get_json(self.SEARCH_URL, params={"arg": query, "by": "name"})
        except ExternalServiceError:
            return "AUR search failed due to a network error."
        if status != 200:
            return f"AUR search failed with status {status}."
        if not isinstance(data, dict):
            return "AUR search failed due to an invalid response."
        if data.get("type") == "error":
            return f"AUR search error: {data.get('error')}"
        results = data.get("results", [])
        if not results:
            return f"No AUR packages found for query: {query}"
        exact = next((item for item in results if item.get("Name") == query), None)
        if exact:
            return format_aur_exact_result(exact)
        return format_aur_results(results)

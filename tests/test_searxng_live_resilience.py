import pytest

from astrbot_plugin_web_searcher_pro.errors import ExternalServiceError
from astrbot_plugin_web_searcher_pro.services.http_client import HttpClient
from astrbot_plugin_web_searcher_pro.services.searxng_service import SearxngService


class FailingHttpClient(HttpClient):
    async def get_json(self, url, *, params=None, headers=None):
        raise ExternalServiceError("timeout")


@pytest.mark.asyncio
async def test_searxng_service_returns_empty_result_on_upstream_timeout():
    service = SearxngService(FailingHttpClient(timeout=1), "http://example.com")

    result = await service.search("OpenAI", category="general", limit=3)

    assert result.is_empty

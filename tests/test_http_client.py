import asyncio

import pytest

from astrbot_plugin_web_searcher_pro.errors import ExternalServiceError
from astrbot_plugin_web_searcher_pro.services.http_client import HttpClient


class TimeoutSession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, *args, **kwargs):
        raise asyncio.TimeoutError


@pytest.mark.asyncio
async def test_http_client_wraps_timeout_as_external_service_error(monkeypatch):
    monkeypatch.setattr("astrbot_plugin_web_searcher_pro.services.http_client.aiohttp.ClientSession", TimeoutSession)
    client = HttpClient(timeout=1)

    with pytest.raises(ExternalServiceError):
        await client.get_json("http://example.com")

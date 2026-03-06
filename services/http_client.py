import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping

import aiohttp

from ..errors import ExternalServiceError


@dataclass(slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    url: str

    def text(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        return self.body.decode(encoding, errors=errors)

    def json(self) -> Any:
        if not self.body:
            return {}
        return json.loads(self.text())


class HttpClient:
    def __init__(self, *, proxy: str | None = None, timeout: int = 15, user_agent: str | None = None):
        self.proxy = proxy
        self.timeout = timeout
        self.user_agent = user_agent or "astrbot-plugin-web-searcher-pro/2.0"

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ) -> HttpResponse:
        merged_headers = {"User-Agent": self.user_agent}
        if headers:
            merged_headers.update(headers)

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=merged_headers) as session:
                async with session.request(
                    method,
                    url,
                    params=params,
                    proxy=self.proxy,
                    allow_redirects=allow_redirects,
                ) as response:
                    body = await response.read()
                    return HttpResponse(
                        status=response.status,
                        headers=dict(response.headers),
                        body=body,
                        url=str(response.url),
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError) as exc:
            raise ExternalServiceError(str(exc) or "request timeout") from exc

    async def get_json(self, url: str, *, params: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None) -> tuple[int, Any]:
        response = await self.request("GET", url, params=params, headers=headers)
        return response.status, response.json()

    async def get_text(self, url: str, *, params: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None) -> tuple[int, str, Mapping[str, str]]:
        response = await self.request("GET", url, params=params, headers=headers)
        return response.status, response.text(), response.headers

    async def get_bytes(self, url: str, *, params: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None) -> tuple[int, bytes, Mapping[str, str]]:
        response = await self.request("GET", url, params=params, headers=headers)
        return response.status, response.body, response.headers

    async def head_ok(self, url: str) -> bool:
        try:
            response = await self.request("HEAD", url, allow_redirects=True)
            return 200 <= response.status < 400
        except ExternalServiceError:
            return False

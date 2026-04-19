import pytest

from astrbot_plugin_web_searcher_pro.services.web_fetch_service import WebFetchService


class DummyHttpClient:
    def __init__(self, html: str, status: int = 200):
        self.html = html
        self.status = status

    async def get_text(self, url: str, *, params=None, headers=None):
        return self.status, self.html, {}


class DummyGitHubService:
    async def search(self, query: str) -> str:
        return f"github:{query}"


@pytest.mark.asyncio
async def test_fetch_returns_plain_text_for_readable_pages():
    html = """
    <html>
      <head><title>Example Article</title></head>
      <body>
        <article>
          <h1>Example Article</h1>
          <p>This is a readable article body with enough text to make readability extraction useful.</p>
          <p>It should be returned as plain text instead of raw HTML so downstream tools get meaningful content.</p>
        </article>
      </body>
    </html>
    """
    service = WebFetchService(DummyHttpClient(html), DummyGitHubService())

    result = await service.fetch("https://example.com/article")

    assert "Example Article" in result
    assert "meaningful content" in result
    assert "<article>" not in result


@pytest.mark.asyncio
async def test_fetch_falls_back_to_metadata_for_client_rendered_pages():
    html = """
    <!doctype html>
    <html lang="zh-CN">
      <head>
        <meta charset="UTF-8" />
        <title>AstrBot - Agentic AI 助手</title>
      </head>
      <body>
        <div id="root"></div>
      </body>
    </html>
    """
    service = WebFetchService(DummyHttpClient(html), DummyGitHubService())

    result = await service.fetch("https://astrbot.app/")

    assert "AstrBot - Agentic AI 助手" in result
    assert "client-rendered" in result

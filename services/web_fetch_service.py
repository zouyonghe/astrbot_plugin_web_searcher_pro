import re

from bs4 import BeautifulSoup
from readability import Document

from ..errors import ExternalServiceError
from .github_service import GitHubService
from .http_client import HttpClient
from ..utils.url_helpers import extract_github_repo, is_http_url


class WebFetchService:
    def __init__(self, http_client: HttpClient, github_service: GitHubService):
        self.http_client = http_client
        self.github_service = github_service

    async def fetch(self, url: str) -> str:
        if not is_http_url(url):
            return "The provided URL is invalid."

        repo = extract_github_repo(url)
        if repo:
            return await self.github_service.search(repo)

        try:
            status, html, _ = await self.http_client.get_text(url)
        except ExternalServiceError:
            return "Fetch URL failed, please try again later."
        if status != 200:
            return "Unable to fetch website content. Please check the URL."
        summary = Document(html).summary()
        readable_text = self._extract_plain_text(summary)
        if len(readable_text) >= 80:
            return readable_text

        fallback = self._build_metadata_fallback(html)
        if fallback:
            return fallback

        return readable_text or "No readable content found on the page."

    def _extract_plain_text(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return re.sub(r"\s+", " ", " ".join(soup.stripped_strings)).strip()

    def _build_metadata_fallback(self, html: str) -> str:
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        visible_text = self._extract_plain_text(html)

        parts: list[str] = []
        if title:
            parts.append(f"Title: {title}")
        if description and description != title:
            parts.append(f"Description: {description}")

        if visible_text and visible_text not in {title, description}:
            parts.append(f"Visible text: {visible_text[:400]}")

        if self._looks_like_client_rendered_page(soup, visible_text):
            parts.append(
                "This page appears to be client-rendered. Only metadata is available without a browser runtime."
            )

        return "\n".join(parts)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            return str(og_title["content"]).strip()
        return ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        selectors = (
            {"name": "description"},
            {"property": "og:description"},
            {"name": "twitter:description"},
        )
        for attrs in selectors:
            meta = soup.find("meta", attrs=attrs)
            if meta and meta.get("content"):
                return str(meta["content"]).strip()
        return ""

    def _looks_like_client_rendered_page(self, soup: BeautifulSoup, visible_text: str) -> bool:
        if len(visible_text) >= 80:
            return False
        placeholder_ids = {"root", "app", "__next", "__nuxt", "app-root"}
        root_container = soup.find(
            attrs={"id": lambda value: isinstance(value, str) and value.strip().lower() in placeholder_ids}
        )
        return bool(root_container or soup.find(attrs={"data-reactroot": True}) or soup.find("script", attrs={"type": "module"}))

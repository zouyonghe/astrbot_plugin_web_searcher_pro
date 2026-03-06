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
        return summary or "No readable content found on the page."

import base64
from typing import Mapping

from ..errors import ExternalServiceError
from .http_client import HttpClient
from ..utils.url_helpers import extract_github_repo


class GitHubService:
    def __init__(self, http_client: HttpClient, token: str = ""):
        self.http_client = http_client
        self.token = token

    @property
    def headers(self) -> Mapping[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"token {self.token}"}

    async def search(self, query: str) -> str:
        repo_path = extract_github_repo(query)
        if repo_path:
            return await self.fetch_repo(repo_path)

        if "/" in query and not query.startswith("http"):
            exact = await self.fetch_repo(query)
            if "not found" not in exact.lower() and "error" not in exact.lower():
                return exact

        try:
            status, data = await self.http_client.get_json(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": 5},
                headers=self.headers,
            )
        except ExternalServiceError:
            return "An error occurred while fetching repository information. Please try again later."
        if status != 200:
            return f"GitHub search failed. HTTP Status: {status}"

        items = data.get("items", []) if isinstance(data, dict) else []
        if not items:
            return f"No repositories found for query: {query}"
        if len(items) == 1:
            repo = items[0].get("full_name")
            if repo:
                return await self.fetch_repo(repo)

        return "\n".join(
            f"{index + 1}. **{item.get('full_name')}** - {item.get('description') or 'No description'}"
            f"  \nClone URL: {item.get('clone_url')}"
            for index, item in enumerate(items)
        )

    async def fetch_repo(self, repo_path: str) -> str:
        try:
            status, data = await self.http_client.get_json(
                f"https://api.github.com/repos/{repo_path}",
                headers=self.headers,
            )
        except ExternalServiceError:
            return "An error occurred while fetching repository information. Please try again later."
        if status == 404:
            return f"Repository '{repo_path}' not found."
        if status != 200 or not isinstance(data, dict):
            return f"Error while fetching repository: HTTP Status {status}."

        details = (
            "**Repository Details**\n"
            f"Name: {data.get('name')}\n"
            f"Full Name: {data.get('full_name')}\n"
            f"Description: {data.get('description')}\n"
            f"Stars: {data.get('stargazers_count')}\n"
            f"Forks: {data.get('forks_count')}\n"
            f"Language: {data.get('language')}\n"
            f"URL: {data.get('html_url')}\n\n"
        )
        return details + await self._fetch_readme(data)

    async def _fetch_readme(self, repo_data: Mapping[str, object]) -> str:
        api_url = repo_data.get("url")
        if not api_url:
            return "Failed to fetch the README content."
        try:
            status, data = await self.http_client.get_json(f"{api_url}/readme", headers=self.headers)
        except ExternalServiceError:
            return "Failed to fetch the README content."
        if status == 404:
            return "This repository does not have a README file."
        if status != 200 or not isinstance(data, dict):
            return "Failed to fetch the README content."
        try:
            content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        except Exception:
            return "Failed to decode the README content."
        return "**README Content:**\n\n" + content[:4000]

import os
from dataclasses import dataclass
from typing import Mapping, Any


DEFAULT_SEARXNG_API_URL = "http://127.0.0.1:8080"
DEFAULT_REQUEST_TIMEOUT = 15
DEFAULT_IMAGE_RESULT_LIMIT = 12
DEFAULT_IMAGE_CANDIDATE_LIMIT = 120


@dataclass(frozen=True)
class PluginConfig:
    searxng_api_url: str = DEFAULT_SEARXNG_API_URL
    enable_random_image: bool = False
    enable_image_title: bool = True
    github_token: str = ""
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    image_result_limit: int = DEFAULT_IMAGE_RESULT_LIMIT
    image_candidate_limit: int = DEFAULT_IMAGE_CANDIDATE_LIMIT
    proxy: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PluginConfig":
        return cls(
            searxng_api_url=str(mapping.get("searxng_api_url", DEFAULT_SEARXNG_API_URL)).rstrip("/"),
            enable_random_image=bool(mapping.get("enable_random_image", False)),
            enable_image_title=bool(mapping.get("enable_image_title", True)),
            github_token=str(mapping.get("github_token", "")).strip(),
            request_timeout=max(3, int(mapping.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))),
            image_result_limit=max(1, int(mapping.get("image_result_limit", DEFAULT_IMAGE_RESULT_LIMIT))),
            image_candidate_limit=max(
                10,
                int(mapping.get("image_candidate_limit", DEFAULT_IMAGE_CANDIDATE_LIMIT)),
            ),
            proxy=os.environ.get("https_proxy") or os.environ.get("http_proxy"),
        )

    @classmethod
    def from_astrbot(cls, config: Mapping[str, Any]) -> "PluginConfig":
        return cls.from_mapping(config)

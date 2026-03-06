import base64
import io
from typing import Iterable

from bs4 import BeautifulSoup
from PIL import Image as PilImage

from ..search_models import SearchResultItem
from .http_client import HttpClient


def parse_resolution(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    normalized = value.lower().replace("×", "x")
    if "x" not in normalized:
        return None
    width_str, height_str = normalized.split("x", 1)
    if not width_str.isdigit() or not height_str.isdigit():
        return None
    return int(width_str), int(height_str)


def resolution_area(value: str) -> int:
    parsed = parse_resolution(value)
    if not parsed:
        return 0
    width, height = parsed
    return width * height


def deduplicate_image_results(items: Iterable[SearchResultItem]) -> list[SearchResultItem]:
    seen: set[str] = set()
    results: list[SearchResultItem] = []
    for item in items:
        key = item.img_src or item.url
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


def rank_image_results(items: Iterable[SearchResultItem]) -> list[SearchResultItem]:
    return sorted(
        deduplicate_image_results(items),
        key=lambda item: (resolution_area(item.resolution), item.score),
        reverse=True,
    )


class ImageService:
    def __init__(self, http_client: HttpClient):
        self.http_client = http_client

    def is_base64_image(self, base64_data: str) -> bool:
        try:
            image_data = base64.b64decode(base64_data)
            image = PilImage.open(io.BytesIO(image_data))
            image.verify()
            return True
        except Exception:
            return False

    async def download_base64(self, image_url: str, depth: int = 0) -> str | None:
        if not image_url or depth > 2:
            return None

        status, content, headers = await self.http_client.get_bytes(image_url)
        if status != 200:
            return None

        content_type = str(headers.get("Content-Type", "")).lower()
        if "html" in content_type:
            html = content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            og_image = soup.find("meta", attrs={"property": "og:image"})
            fallback_url = og_image.get("content") if og_image else None
            if not fallback_url:
                return None
            return await self.download_base64(fallback_url, depth + 1)

        encoded = base64.b64encode(content).decode("utf-8")
        return encoded if self.is_base64_image(encoded) else None

from datetime import datetime
from typing import Iterable, Mapping

from ..search_models import SearchResult, SearchResultItem


def format_search_result(result: SearchResult | None, empty_message: str) -> str:
    if result is not None and result.error_message:
        return result.error_message
    if result is None or result.is_empty:
        return empty_message
    return str(result)


def format_image_results(items: Iterable[SearchResultItem]) -> str:
    formatted = []
    for idx, item in enumerate(items, 1):
        details = [f"{idx}. {item.title or 'Untitled'}"]
        if item.url:
            details.append(f"Page: {item.url}")
        if item.img_src:
            details.append(f"Image: {item.img_src}")
        if item.resolution:
            details.append(f"Resolution: {item.resolution}")
        if item.content:
            details.append(f"Content: {item.content}")
        formatted.append("\n".join(details))
    return "\n\n".join(formatted)


def format_aur_exact_result(package: Mapping[str, object]) -> str:
    timestamp = package.get("LastModified")
    updated_at = "N/A"
    if isinstance(timestamp, (int, float)):
        updated_at = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return (
        "**Package Details**\n"
        f"Name: {package.get('Name')}\n"
        f"Description: {package.get('Description')}\n"
        f"Maintainer: {package.get('Maintainer') or 'N/A'}\n"
        f"Votes: {package.get('NumVotes')}\n"
        f"Popularity: {package.get('Popularity')}\n"
        f"Last Updated: {updated_at}"
    )


def format_aur_results(results: Iterable[Mapping[str, object]]) -> str:
    return "\n".join(
        f"* {item.get('Name')} - {item.get('Description')} (Votes: {item.get('NumVotes')})"
        for item in results
    )

# engines/search_models.py
from dataclasses import dataclass, field
from typing import List, Optional, TypedDict


@dataclass
class SearchResultItem:
    title: str
    url: str
    img_src: str
    resolution: str
    iframe_src: str
    content: str
    engine: str
    score: float


@dataclass
class SearchResult:
    results: List[SearchResultItem] = field(default_factory=list)

    def __iter__(self):
        """使 SearchResult 类支持迭代"""
        return iter(self.results)

    def __str__(self) -> str:
        """提供结果的字符串形式"""
        if not self.results:  # 如果 results 为空
            return "No results."

        formatted_results = [
            f"{idx + 1}. {item.title}\nLink: {item.url}\nContent: {item.content}"
            for idx, item in enumerate(self.results)
        ]
        return "\n".join(formatted_results) if formatted_results else ""
    
@dataclass
class SearchBookResultItem:
    title: str
    author: str
    year: str
    cover_url: str
    filesize: int
    ipfs_cid: str
    extension: str

@dataclass
class SearchBookResult:
    results: List[SearchBookResultItem] = field(default_factory=list)

    def __iter__(self):
        """Make SearchBookResult iterable."""
        return iter(self.results)

    def __str__(self) -> str:
        """Provide string representation of book results."""
        if not self.results:  # If results are empty
            return "No book results."

        formatted_results = [
            f"{idx + 1}. {item.title} by {item.author} ({item.year})\n" \
            f"File: {item.filesize} ({item.extension})\nCID: {item.ipfs_cid}"
            for idx, item in enumerate(self.results)
        ]
        return "\n".join(formatted_results) if formatted_results else ""

# engines/search_models.py
from dataclasses import dataclass, field
from typing import List, Optional, TypedDict


@dataclass
class SearchResultItem:
    title: str
    url: str
    img_src: str
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
        formatted_results = [
            f"{idx + 1}. {item.title}\nLink: {item.url}\nContent: {item.content}"
            for idx, item in enumerate(self.results)
        ]
        return "\n".join(formatted_results) if formatted_results else ""

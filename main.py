import asyncio
import random

from astrbot.api import *
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import *
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Image, Node, Nodes, Plain

from .config import PluginConfig
from .formatters.search_formatter import format_search_result
from .services.aur_service import AurService
from .services.github_service import GitHubService
from .services.http_client import HttpClient
from .services.image_service import ImageService
from .services.searxng_service import SearxngService
from .services.web_fetch_service import WebFetchService


GENERAL_EMPTY_MESSAGE = "No information found for your query."
GENERAL_TOOL_NAME = "searxng_web_search_general"
VIDEOS_EMPTY_MESSAGE = "No videos found for your query."
IMAGES_TOOL_NAME = "searxng_web_search_images"
VIDEOS_TOOL_NAME = "searxng_web_search_videos"
NEWS_EMPTY_MESSAGE = "No news found for your query."
NEWS_TOOL_NAME = "searxng_web_search_news"
SCIENCE_EMPTY_MESSAGE = "No science information found for your query."
SCIENCE_TOOL_NAME = "searxng_web_search_science"
MUSIC_EMPTY_MESSAGE = "No music found for your query."
MUSIC_TOOL_NAME = "searxng_web_search_music"
TECHNICAL_EMPTY_MESSAGE = "No technical information found for your query."
TECHNICAL_TOOL_NAME = "searxng_web_search_technical"
ACADEMIC_EMPTY_MESSAGE = "No academic information found for your query."
ACADEMIC_TOOL_NAME = "searxng_web_search_academic"
MAP_EMPTY_MESSAGE = "No map information found for your query."
MAP_TOOL_NAME = "searxng_web_search_map"
FILES_EMPTY_MESSAGE = "No files found for your query."
FILES_TOOL_NAME = "searxng_web_search_files"
SOCIAL_EMPTY_MESSAGE = "No social information found for your query."
SOCIAL_TOOL_NAME = "searxng_web_search_social"
BOOKS_EMPTY_MESSAGE = "No books found for your query."
BOOKS_TOOL_NAME = "searxng_web_search_books"
IMAGE_EMPTY_MESSAGE = "没有找到图片，请稍后再试。"
IMAGE_DOWNLOAD_FAILED = "图片下载失败，请稍后再试。"
FETCH_URL_TOOL_NAME = "searxng_web_fetch_url"

WEBSEARCH_TOOL_NAMES = (
    GENERAL_TOOL_NAME,
    IMAGES_TOOL_NAME,
    VIDEOS_TOOL_NAME,
    NEWS_TOOL_NAME,
    SCIENCE_TOOL_NAME,
    MUSIC_TOOL_NAME,
    TECHNICAL_TOOL_NAME,
    ACADEMIC_TOOL_NAME,
    MAP_TOOL_NAME,
    FILES_TOOL_NAME,
    SOCIAL_TOOL_NAME,
    BOOKS_TOOL_NAME,
    FETCH_URL_TOOL_NAME,
)


@register("web_searcher_pro", "buding", "更高性能的Web检索插件", "1.1.3",
          "https://github.com/zouyonghe/astrbot_plugin_web_searcher_pro")
class WebSearcherPro(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.plugin_config = PluginConfig.from_astrbot(config)
        self.http_client = HttpClient(
            proxy=self.plugin_config.proxy,
            timeout=self.plugin_config.request_timeout,
        )
        self.image_service = ImageService(self.http_client)
        self.github_service = GitHubService(self.http_client, token=self.plugin_config.github_token)
        self.searxng_service = SearxngService(
            self.http_client,
            self.plugin_config.searxng_api_url,
            image_candidate_limit=self.plugin_config.image_candidate_limit,
            image_result_limit=self.plugin_config.image_result_limit,
        )
        self.web_fetch_service = WebFetchService(self.http_client, self.github_service)
        self.aur_service = AurService(self.http_client)

    def _set_websearch_status(self, status: bool) -> None:
        provider_settings = self.context.get_config()["provider_settings"]
        provider_settings["web_search"] = status
        self.context.get_config().save_config()
        for tool_name in WEBSEARCH_TOOL_NAMES:
            if status:
                self.context.activate_llm_tool(tool_name)
            else:
                self.context.deactivate_llm_tool(tool_name)

    async def _search_text_category(self, query: str, *, category: str, empty_message: str, limit: int = 5) -> str:
        logger.info(f"Starting {category} search for: {query}")
        result = await self.searxng_service.search(query, category=category, limit=limit)
        return format_search_result(result, empty_message=empty_message)

    async def _build_image_node(self, event: AstrMessageEvent, item) -> Node | None:
        base64_image = await self.image_service.download_base64(item.img_src)
        if not base64_image:
            return None
        content = [Image.fromBase64(base64_image)]
        if self.plugin_config.enable_image_title:
            content.append(Plain(f"{item.title}\n"))
        return Node(
            uin=event.get_self_id(),
            name="IMAGE",
            content=content,
        )

    @command("websearch")
    async def websearch(self, event: AstrMessageEvent, operation: str = None):
        websearch = self.context.get_config()["provider_settings"]["web_search"]
        if operation is None:
            status_now = "开启" if websearch else "关闭"
            yield event.plain_result(
                f"当前网页搜索功能状态：{status_now}。使用 /websearch on 或者 off 启用或者关闭。"
            )
            return

        operation = operation.lower()
        if operation == "on":
            if websearch:
                yield event.plain_result("网页搜索功能已经是开启状态")
                return
            self._set_websearch_status(True)
            yield event.plain_result("已开启网页搜索功能")
            return

        if operation == "off":
            if not websearch:
                yield event.plain_result("网页搜索功能已经是关闭状态")
                return
            self._set_websearch_status(False)
            yield event.plain_result("已关闭网页搜索功能")
            return

        yield event.plain_result("操作参数错误，应为 on 或 off")

    @llm_tool(GENERAL_TOOL_NAME)
    async def search_general(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for general information.

        Args:
            query (string): A search query used to fetch general web-based information.
        """
        return await self._search_text_category(query, category="general", empty_message=GENERAL_EMPTY_MESSAGE)

    @llm_tool(IMAGES_TOOL_NAME)
    async def search_images(self, event: AstrMessageEvent, query: str):
        """Search the web for images.

        Use this tool when the user wants existing images about a topic.

        Args:
            query (string): A search query focused on the subject of the images, such as landmarks, products, people, or scenes. Do not add generic words like "images" because this tool already searches image results.
        """
        logger.info(f"Starting image search for: {query}")
        result = await self.searxng_service.search(
            query,
            category="images",
            limit=self.plugin_config.image_result_limit,
            engines=["google images", "bing images"],
        )
        if result.is_empty:
            yield event.plain_result(IMAGE_EMPTY_MESSAGE)
            return

        if self.plugin_config.enable_random_image:
            selected = random.choice(result.results)
            if not self.plugin_config.enable_image_title:
                yield event.image_result(selected.img_src)
                return
            base64_image = await self.image_service.download_base64(selected.img_src)
            if not base64_image:
                yield event.plain_result(IMAGE_DOWNLOAD_FAILED)
                return
            yield event.chain_result([Image.fromBase64(base64_image), Plain(selected.title)])
            return

        nodes = await asyncio.gather(*(self._build_image_node(event, item) for item in result.results))
        valid_nodes = [node for node in nodes if node]
        if not valid_nodes:
            yield event.plain_result(IMAGE_DOWNLOAD_FAILED)
            return
        yield event.chain_result([Nodes(valid_nodes)])

    @llm_tool(VIDEOS_TOOL_NAME)
    async def search_videos(self, event: AstrMessageEvent, query: str):
        """Search the web for videos.

        Args:
            query (string): A search query focused on the topic or keywords of the videos. Avoid adding broad helper words like "videos" because the search is already scoped to video results.
        """
        return await self._search_text_category(query, category="videos", empty_message=VIDEOS_EMPTY_MESSAGE)

    @llm_tool(NEWS_TOOL_NAME)
    async def search_news(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for news.

        Args:
            query (string): A search query focused on the topic, event, company, or person in the news. Avoid adding the word "news" unless it is part of the actual subject.
        """
        return await self._search_text_category(query, category="news", empty_message=NEWS_EMPTY_MESSAGE)

    @llm_tool(SCIENCE_TOOL_NAME)
    async def search_science(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for scientific information.

        Args:
            query (string): A search query about scientific topics, research areas, discoveries, or concepts, such as "quantum mechanics" or "CRISPR gene editing".
        """
        return await self._search_text_category(query, category="science", empty_message=SCIENCE_EMPTY_MESSAGE)

    @llm_tool(MUSIC_TOOL_NAME)
    async def search_music(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for music-related information.

        Args:
            query (string): A search query about songs, albums, artists, genres, lyrics, performances, or music topics. Avoid redundant helper words when the subject is already clear.
        """
        return await self._search_text_category(query, category="music", empty_message=MUSIC_EMPTY_MESSAGE)

    @llm_tool(TECHNICAL_TOOL_NAME)
    async def search_technical(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for technical information.

        Args:
            query (string): A search query about specific technical topics, engineering concepts, APIs, programming languages, frameworks, or troubleshooting details.
        """
        return await self._search_text_category(query, category="technical", empty_message=TECHNICAL_EMPTY_MESSAGE)

    @llm_tool(ACADEMIC_TOOL_NAME)
    async def search_academic(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for academic information.

        Args:
            query (string): A search query about academic topics, papers, authors, fields of study, or research keywords. Keep the query focused on the subject rather than generic academic wording.
        """
        return await self._search_text_category(query, category="academic", empty_message=ACADEMIC_EMPTY_MESSAGE)

    @llm_tool(MAP_TOOL_NAME)
    async def search_map(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for map-related information.

        Args:
            query (string): A search query focused on places, addresses, landmarks, routes, or geographic points of interest.
        """
        return await self._search_text_category(query, category="map", empty_message=MAP_EMPTY_MESSAGE)

    @llm_tool(FILES_TOOL_NAME)
    async def search_files(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for files.

        Args:
            query (string): A search query focused on downloadable documents, datasets, manuals, PDFs, presentations, or other file-like resources.
        """
        return await self._search_text_category(query, category="files", empty_message=FILES_EMPTY_MESSAGE)

    @llm_tool(SOCIAL_TOOL_NAME)
    async def search_social(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for social content.

        Args:
            query (string): A search query about people, accounts, posts, communities, or discussions on social platforms.
        """
        return await self._search_text_category(query, category="social", empty_message=SOCIAL_EMPTY_MESSAGE)

    @llm_tool(BOOKS_TOOL_NAME)
    async def search_books(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for books.

        Args:
            query (string): A search query about book titles, authors, editions, ISBNs, subjects, or reading recommendations.
        """
        return await self._search_text_category(query, category="books", empty_message=BOOKS_EMPTY_MESSAGE)

    @llm_tool(FETCH_URL_TOOL_NAME)
    async def fetch_website_content(self, event: AstrMessageEvent, url: str):
        """Fetch readable content from a website URL.

        Use this tool when the user provides a webpage and wants its main content, or when a GitHub repository URL should be analyzed through the GitHub tool path. Do not use it for downloading e-books or arbitrary binary files.

        Args:
            url (string): The full HTTP or HTTPS URL of the page to fetch.
        """
        logger.info(f"Fetching web content for: {url}")
        return await self.web_fetch_service.fetch(url)

    @command("aur")
    async def search_aur(self, event: AstrMessageEvent, query: str):
        logger.info(f"Searching AUR packages for: {query}")
        result = await self.aur_service.search(query)
        yield event.plain_result(result)

    @llm_tool("searxng_github_search")
    async def search_github_repo(self, event: AstrMessageEvent, query: str) -> str:
        """Search GitHub repositories or inspect a specific repository.

        Use this tool when the user asks about a GitHub repository, provides an `owner/repo`, a GitHub URL, a clone URL, or wants repository search results.

        Args:
            query (string): Repository keywords, an `owner/repo` identifier, a GitHub repository URL, or a clone URL.
        """
        logger.info(f"Searching GitHub repositories for: {query}")
        return await self.github_service.search(query)

    @command("github")
    async def github_search(self, event: AstrMessageEvent, query: str = None):
        if not query:
            yield event.plain_result("Please provide a repository name, URL, or search keywords.")
            return
        result = await self.search_github_repo(event, query)
        yield event.plain_result(result)

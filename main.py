import logging
import os
import uuid
from typing import Optional

import aiohttp
from aiohttp import TCPConnector
from readability import Document

from astrbot.api import *
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from data.plugins.astrbot_plugin_web_searcher_pro.search_models import SearchResult, SearchResultItem

logger = logging.getLogger("astrbot")

image_llm_prefix = "The images have been sent to the user. Below is the description of the images:\n"

# def is_valid_url(url: str):
#     try:
#         result = urlparse(url)
#         # 检查网络协议是否为 http 或 https 且包含域名
#         return all([result.scheme in ("http", "https"), result.netloc])
#     except Exception:
#         return False

async def is_valid_url(url):
    """验证 URL 是否有效"""
    proxy = os.environ.get("https_proxy")  # 从环境变量获取代理
    try:
        async with aiohttp.ClientSession(connector=TCPConnector(ssl=False)) as session:
            async with session.head(url, proxy=proxy) as response:
                return response.status == 200
    except Exception as e:
        logger.warning(f"Failed to access URL: {url}, error: {e}")
        return False


async def download_image_from_url(url):
    """下载图片并保存到本地，通过 https_proxy 配置"""
    temp_dir = "/app/.config/QQ/NapCat/temp"
    temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")  # 保存为 jpg 格式

    # 从环境变量获取代理
    proxy = os.environ.get("https_proxy")

    try:
        async with aiohttp.ClientSession(connector=TCPConnector(ssl=False)) as session:
            async with session.get(url, proxy=proxy) as response:
                if response.status == 200:
                    with open(temp_file, "wb") as f:
                        f.write(await response.read())
                        return temp_file
                else:
                    logger.warning(f"Failed to download image: {url}, HTTP status: {response.status}")
                    return None
    except Exception as e:
        logger.warning(f"Error while downloading image: {url}, error: {e}")
        return None

@register("web_searcher_pro", "buding", "更高性能的Web检索插件", "1.0.0",
          "https://github.com/zouyonghe/astrbot_plugin_web_searcher_pro")
class WebSearcherPro(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.proxy = os.environ.get("https_proxy")

    async def search(self, query: str, categories: str = "general", limit: int = 10) -> Optional[SearchResult]:
        """Perform a search query for a specific category.
    
        Args:
            query (str): The search query string.
            categories (str): The category to search within. Defaults to "general".
            limit (int): The maximum number of results to return. Defaults to 10.
    
        Returns:
            str: A formatted string of search results.
        """
        searxng_api_url = self.config.get("searxng_api_url", "http://127.0.0.1:8080")
        search_endpoint = f"{searxng_api_url}/search"
        params = {
            "q": query,
            "categories": categories,
            "format": "json",
            "lang": "zh",
            "limit": limit
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if not data.get("results"):
                            return None

                        results = SearchResult(
                            results=[
                                SearchResultItem(
                                    title=item.get('title', ''),
                                    url=item.get('url', '') if is_valid_url(item.get('url', '')) else '',
                                    img_src=item.get('img_src', '') if is_valid_url(item.get('img_src', '')) else '',
                                    content=item.get('content', ''),
                                    engine=item.get('engine', ''),
                                    score=item.get('score', 0.0)
                                )
                                for item in data.get("results", [])[:limit]
                            ]
                        )
                        return results
                    else:
                        logger.error(f"Failed to search SearxNG. HTTP Status: {response.status}, Params: {params}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during fetch_search_results: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during fetch_search_results: {e}")

    @filter.command("websearch")
    async def websearch(self, event: AstrMessageEvent, operation: str = None):
        def update_websearch_status(status: bool):
            """更新网页搜索状态并保存配置."""
            self.context.get_config()['provider_settings']['web_search'] = status
            self.context.get_config().save_config()
            if status:
                self.context.activate_llm_tool("web_search")
                self.context.activate_llm_tool("fetch_url")
            else:
                self.context.deactivate_llm_tool("web_search")
                self.context.deactivate_llm_tool("fetch_url")

        # 1. 检查当前状态
        websearch = self.context.get_config()['provider_settings']['web_search']
        if operation is None:
            status_now = "开启" if websearch else "关闭"
            yield event.plain_result(
                f"当前网页搜索功能状态：{status_now}。使用 /websearch on 或者 off 启用或者关闭。"
            )
            return

        # 2. 处理参数
        operation = operation.lower()  # 兼容大小写
        if operation == "on":
            if websearch:  # 避免重复操作
                yield event.plain_result("网页搜索功能已经是开启状态")
            else:
                update_websearch_status(True)
                yield event.plain_result("已开启网页搜索功能")
        elif operation == "off":
            if not websearch:  # 避免重复操作
                yield event.plain_result("网页搜索功能已经是关闭状态")
            else:
                update_websearch_status(False)
                yield event.plain_result("已关闭网页搜索功能")
        else:
            yield event.plain_result("操作参数错误，应为 on 或 off")

    @llm_tool("web_search")
    async def search_general(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for general information

        Args:
            query (string): A search query used to fetch general web-based information.
        """
        logger.info(f"Starting general search for: {query}")
        results = await self.search(query, categories="general")
        if not results:
            return "No information found for your query."
        return str(results)

    @llm_tool("web_search_images")
    async def search_images(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for images

        Args:
            query (string): A search query used to fetch image-based results.
        """
        logger.info(f"Starting image search for: {query}")
        results = await self.search(query, categories="images", limit=5)
        if not results:
            event.plain_result("No images found for your query.")
            return "No images found for your query."
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
        assert isinstance(event, AiocqhttpMessageEvent)
        client = event.bot
        for result in results.results:
            if not await is_valid_url(result.img_src):
                logger.warning(f"Invalid or unreachable URL: {result.img_src}")
                continue

            image_file = await download_image_from_url(result.img_src)
            if not image_file:
                logger.warning(f"Skipping image due to download failure: {result.url}")
                continue
            if event.is_private_chat():
                await client.send_private_msg(
                    user_id=int(event.get_sender_id()),
                    message=f"[CQ:image,file={image_file}]",
                    auto_escape=False,
                    self_id=int(event.get_self_id()),
                )
            else:
                await client.send_group_msg(
                    group_id=int(event.get_group_id()),
                    message=f"[CQ:image,file={image_file}]",
                    auto_escape=False,
                    self_id=int(event.get_self_id()),
                )
                os.remove(image_file)
        return f"{image_llm_prefix} {results}"

    @llm_tool("web_search_videos")
    async def search_videos(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for videos

        Args:
            query (string): A search query used to retrieve video-based results.
        """
        logger.info(f"Starting video search for: {query}")
        results = await self.search(query, categories="videos")
        if not results:
            return "No videos found for your query."
        return str(results)

    @llm_tool("web_search_news")
    async def search_news(self, query: str) -> str:
        """Search the web for news

        Args:
            query (string): A search query used to gather news-related articles or information.
        """
        logger.info(f"Starting news search for: {query}")
        results = await self.search(query, categories="news")
        if not results:
            return "No news found for your query."
        return str(results)

    @llm_tool("web_search_science")
    async def search_science(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for scientific information

        Args:
            query (string): A search query used to retrieve scientific research or relevant knowledge.
        """
        logger.info(f"Starting science search for: {query}")
        results = await self.search(query, categories="science")
        if not results:
            return "No science information found for your query."
        return str(results)

    @llm_tool("web_search_music")
    async def search_music(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for music-related information

        Args:
            query (string): A search query used to gather music-related content or resources.
        """
        logger.info(f"Starting music search for: {query}")
        results = await self.search(query, categories="music")
        if not results:
            return "No music found for your query."
        return str(results)

    @llm_tool("web_search_technical")
    async def search_technical(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for technical information

        Args:
            query (string): A search query used to find technical details or resources.
        """
        logger.info(f"Starting technical search for: {query}")
        results = await self.search(query, categories="technical")
        if not results:
            return "No technical information found for your query."
        return str(results)

    @llm_tool("web_search_academic")
    async def search_academic(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for academic information

        Args:
            query (string): A search query used to find academic papers, studies, or content.
        """
        logger.info(f"Starting academic search for: {query}")
        results = await self.search(query, categories="academic")
        if not results:
            return "No academic information found for your query."
        return str(results)



    @llm_tool("fetch_url")
    async def fetch_website_content(self, event: AstrMessageEvent, url: str) -> str:
        """Fetch the content of a website using the provided URL.

        Args:
            url(string): The URL of the website to fetch content from.
        """
        logger.info(f"正在通过 fetch_website_content 拉取数据: {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch URL: {url} with status {response.status}")
                        return "Unable to fetch website content. Please check the URL."
                    html_content = await response.text()
                    doc = Document(html_content)
                    return doc.summary()
        except Exception as e:
            logger.error(f"fetch_website_content 出现问题: {e}")
            return "Fetch URL failed, please try again later."

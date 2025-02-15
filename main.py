import asyncio
import json
import logging
import os
import random
from typing import Optional, Dict

import aiohttp
import yt_dlp
from readability import Document

from astrbot.api import *
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import *
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Video
from data.plugins.astrbot_plugin_web_searcher_pro.search_models import SearchResult, SearchResultItem

logger = logging.getLogger("astrbot")
temp_path = "./temp"

image_llm_prefix = "The images have been sent to the user. Below is the description of the images:\n"

# 用于跟踪每个用户的状态，记录用户请求的时间和状态
USER_STATES: Dict[str, Dict[str, float]] = {}

@register("web_searcher_pro", "buding", "更高性能的Web检索插件", "1.0.0",
          "https://github.com/zouyonghe/astrbot_plugin_web_searcher_pro")
class WebSearcherPro(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.proxy = os.environ.get("https_proxy")

    async def search(self, query: str, categories: str = "general", limit: int = 5) -> Optional[SearchResult]:
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

                        result = SearchResult(
                            results=[
                                SearchResultItem(
                                    title=item.get('title', ''),
                                    url=item.get('url', ''),
                                    img_src=item.get('img_src', ''),
                                    iframe_src=item.get('iframe_src', ''),
                                    content=item.get('content', ''),
                                    engine=item.get('engine', ''),
                                    score=item.get('score', 0.0),
                                )
                                for item in data.get("results", [])
                            ]
                        )
                        return await result_filter(result, categories, limit)
                    else:
                        logger.error(f"Failed to search SearxNG. HTTP Status: {response.status}, Params: {params}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during search: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
        
    async def _generate_response(self, event: AstrMessageEvent, result: SearchResult):
        provider = self.context.get_using_provider()
        if provider:
            description_generate_prompt = (
                f"你已经依据用户的请求`{event.get_message_str()}`发起了Web搜索的函数调用，"
                f"以下是函数调用返回的结果，"
                f"如果是图片，那么随机挑选的其中一张图片已被发送给用户，"
                f"如果是视频，那么搜索结果中第一个视频已被发送给用户，"
                f"请根据下述信息，基于你的角色以合适的语气、称呼等，生成符合人设的解说。\n\n"
                f"信息：{str(result)}"
            )

            conversation_id = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin,
                                                                                    conversation_id)
            yield event.request_llm(
                prompt=description_generate_prompt,
                func_tool_manager=None,
                session_id=event.session_id,
                contexts=json.loads(conversation.history),
                system_prompt=self.context.provider_manager.selected_default_persona.get("prompt", ""),
                image_urls=[],
                conversation=conversation,
            )

    @command("websearch")
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
        result = await self.search(query, categories="general")
        if not result or not result.results:
            return "No information found for your query."
        return str(result)

    @llm_tool("web_search_images")
    async def search_images(self, event: AstrMessageEvent, query: str):
        """Search the web for images

        Args:
            query (string): A search query used to fetch image-based results.
        """
        logger.info(f"Starting image search for: {query}")
        result = await self.search(query, categories="images")
        if not result or not result.results:
            return
        image = random.choice(result.results)
        if isinstance(image, SearchResultItem):
            yield event.image_result(image.img_src)
            result.results = [image]
        else:
            logger.error("Random image selection failed.")
            return
        try:
            async for r in self._generate_response(event, result):
                yield r
        except Exception as e:
            logger.error(f"调用 generate_response 时出错: {e}")
            yield event.plain_result("❌ 生成回复时失败，请查看控制台日志")

    @llm_tool("web_search_videos")
    async def search_videos(self, event: AstrMessageEvent, query: str):
        """Search the web for videos

        Args:
            query (string): A search query used to retrieve video-based results.
        """
        logger.info(f"Starting video search for: {query}")
        result = await self.search(query, categories="videos", limit=5)
        if not result or not result.results:
            # return "No videos found for your query."
            logger.error("No videos!")
            return
        # video = result.results[0]
        # if isinstance(video, SearchResultItem):
        #     try:
        #         downloaded_file = await _download_video(video.iframe_src)
        #         if downloaded_file == "":
        #             yield event.plain_result("下载视频失败，请稍候再试")
        #             return
        #         yield event.chain_result(Video.fromFileSystem(path=downloaded_file))
        #         os.remove(downloaded_file)
        #         async for r in self._generate_response(event, result):
        #             yield r
        #     except Exception as e:
        #         logger.error(f"下载文件失败，报错: {e}")
        return str(result)

    @llm_tool("web_search_news")
    async def search_news(self, query: str) -> str:
        """Search the web for news

        Args:
            query (string): A search query used to gather news-related articles or information.
        """
        logger.info(f"Starting news search for: {query}")
        results = await self.search(query, categories="news")
        if not results or not results.results:
            return "No news found for your query."
        if not results or not results.results:
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
        if not results or not results.results:
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
        if not results or not results.results:
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
        if not results or not results.results:
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
        if not results or not results.results:
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

async def _is_validate_image_url(img_url) -> bool:
    if not img_url:
        return False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(img_url, timeout=2) as response:
                if response.status == 200 and "image" in response.headers.get("Content-Type", "").lower():
                    return True
    except Exception:
        pass
    return False

async def _validate_and_download_video(url: str, download_path: str | None = None) -> (bool, str):
    """
    检查视频 URL 是否有效并可选地下载到本地。

    参数：
    url: str               视频的 URL。
    download_path: str     （可选）要保存视频的路径。如果为 None，则仅验证 URL。

    返回：
    Tuple[bool, str]       返回一个元组，其中第一个值表示 URL 是否有效，第二个值为下载的视频文件路径（如果下载过）。
    """
    if not url:
        return False, ""

    cookies_file = "ytb-cookies.txt"

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True, # 默认仅验证，不下载视频
        'proxy': 'http://192.168.50.3:7890',
        'cookiefile': cookies_file if os.path.exists(cookies_file) else None,
        'verbose': True,  # 启用详细调试日志
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'

    }

    if download_path:
        ydl_opts.update({
            'outtmpl': f'{download_path}/temp.%(ext)s',
            'skip_download': False  # 允许下载视频
        })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            logger.info("Starting to extract info...")
            info_dict = ydl.extract_info(url, download=bool(download_path))  # 验证或下载
            logger.info("Extraction completed.")

            if info_dict:
                downloaded_file = f"{download_path}/temp.{info_dict.get('ext', 'mp4')}" if download_path else ""
                return True, downloaded_file
    except Exception as e:
        logger.error(f"{e}")
        return False, ""

    return False, ""


async def _is_valid_video_url(video_url) -> bool:
    """
    Validates if a video URL is functional and can be processed.

    Args:
        video_url (str): The video URL to validate.

    Returns:
        bool: True if the video URL is valid, False otherwise.
    """
    is_valid, _ = await _validate_and_download_video(video_url)
    return is_valid


async def _download_video(video_url):
    """
    Downloads a video from the provided video URL if valid.

    Args:
        video_url (str): The video URL to download.

    Returns:
        str: The file path of the downloaded video, or an empty string if invalid.
    """
    if not video_url:
        return ""

    download_path = temp_path  # You can configure this path as needed.
    is_valid, downloaded_file = await _validate_and_download_video(video_url, download_path)
    return downloaded_file if is_valid else ""


async def result_filter(result: SearchResult, categories: str, limit: int) -> Optional[SearchResult]:
    if categories == "images":
        result.results = result.results[:20]
        urls = [item.img_src for item in result.results if item.img_src]
        validation_results = await asyncio.gather(*[_is_validate_image_url(url) for url in urls])
        result.results = [
            item for item, is_valid in zip(result.results, validation_results) if is_valid
        ]
        result.results = [random.choice(result.results)]
    # elif categories == "videos":
    #     result.results = result.results[:3]
    #     # 提取所有 iframe_src
    #     # urls = [item.iframe_src for item in result.results if item.iframe_src]
    #     # 对每个 URL 进行异步验证
    #     # validation_results = await asyncio.gather(*[_is_valid_video_url(url) for url in urls])
    #     for item in result.results:
    #         logger.error(f"LEN: {len(result.results)}")
    #         if await _is_valid_video_url(item.url):
    #             result.results = [item]
    #             return result
    #         logger.error("here called")
    #     return None
    else:
        result.results = result.results[:limit]
        return result

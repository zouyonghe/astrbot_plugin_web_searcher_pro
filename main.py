import asyncio
import base64
import io
import os
import random
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
from PIL import Image as Img

import aiohttp
from aiohttp import ClientPayloadError
from bs4 import BeautifulSoup
from readability import Document

from astrbot.api import *
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import *
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Image, Plain, Nodes, Node
from data.plugins.astrbot_plugin_web_searcher_pro.search_models import SearchResult, SearchResultItem


@register("web_searcher_pro", "buding", "更高性能的Web检索插件", "1.0.3",
          "https://github.com/zouyonghe/astrbot_plugin_web_searcher_pro")
class WebSearcherPro(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.proxy = os.environ.get("https_proxy")

    async def _is_url_accessible(self, url: str, proxy: bool=True) -> bool:
        """
        异步检查给定的 URL 是否可访问。

        :param url: 要检查的 URL
        :param proxy: 是否使用代理
        :return: 如果 URL 可访问返回 True，否则返回 False
        """
        try:
            async with aiohttp.ClientSession() as session:
                if proxy:
                    async with session.head(url, timeout=5, proxy=self.proxy, allow_redirects=True) as response:
                        return response.status == 200
                else:
                    async with session.head(url, timeout=5, allow_redirects=True) as response:
                        return response.status == 200
        except:
            return False  # 如果请求失败（超时、连接中断等）则返回 False

    async def download_and_convert_to_base64(self, cover_url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(cover_url, proxy=self.proxy) as response:
                    if response.status != 200:
                        return None

                    content_type = response.headers.get('Content-Type', '').lower()
                    # 如果 Content-Type 包含 html，则说明可能不是直接的图片
                    if 'html' in content_type:
                        html_content = await response.text()
                        # 使用 BeautifulSoup 提取图片地址
                        soup = BeautifulSoup(html_content, 'html.parser')
                        img_tag = soup.find('meta', attrs={'property': 'og:image'})
                        if img_tag:
                            cover_url = img_tag.get('content')
                            # 再次尝试下载真正的图片地址
                            return await self.download_and_convert_to_base64(cover_url)
                        else:
                            return None

                    # 如果是图片内容，继续下载并转为 Base64
                    content = await response.read()
                    base64_data = base64.b64encode(content).decode("utf-8")
                    return base64_data
        except (ClientPayloadError, aiohttp.ContentLengthError) as payload_error:
            logger.warning(f"Ignored ContentLengthError: {payload_error}")
            # 尝试已接收的数据部分
            if 'content' in locals():  # 如果部分内容已下载
                base64_data = base64.b64encode(content).decode("utf-8")
                if self.is_base64_image(base64_data):  # 检查 Base64 数据是否有效
                    return base64_data
        except Exception as e:
            return None

    def is_base64_image(self, base64_data: str) -> bool:
        """
        检测 Base64 数据是否为有效图片
        :param base64_data: Base64 编码的字符串
        :return: 如果是图片返回 True，否则返回 False
        """
        try:
            # 解码 Base64 数据
            image_data = base64.b64decode(base64_data)
            # 尝试用 Pillow 打开图片
            image = Img.open(io.BytesIO(image_data))
            # 如果图片能正确被打开，再检查格式是否为支持的图片格式
            image.verify()  # 验证图片
            return True  # Base64 是有效图片
        except Exception:
            return False  # 如果解析失败，说明不是图片

    async def search(self, query: str, categories: str = "general", limit: int = 5, engines: list=None) -> Optional[SearchResult]:
        """Perform a search query for a specific category.
    
        Args:
            query (str): The search query string.
            categories (str): The category to search within. Defaults to "general".
            limit (int): The maximum number of results to return. Defaults to 10.
            engines (str): The search engine to use. Defaults to None. If None, all engines will be used.
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
            "limit": limit,
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
                                    resolution=item.get('resolution', ''),
                                    iframe_src=item.get('iframe_src', ''),
                                    content=item.get('content', ''),
                                    engine=item.get('engine', ''),
                                    score=item.get('score', 0.0),
                                )
                                for item in data.get("results", [])
                            ]
                        )
                        return await self._result_filter(result, categories, limit, engines)
                    else:
                        logger.error(f"Failed to search SearxNG. HTTP Status: {response.status}, Params: {params}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during search: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")

    async def _result_filter(self, result: SearchResult, categories: str, limit: int, engines: list=None) -> Optional[SearchResult]:
        if engines:
            result.results = [item for item in result.results if item.engine in engines]

        if categories == "images":
            result.results = result.results[:400]
            urls = [item.img_src for item in result.results if item.img_src]
            validation_results = await asyncio.gather(*[self._is_url_accessible(url) for url in urls])
            result.results = [
                item for item, is_valid in zip(result.results, validation_results) if is_valid
            ]
            # result = self._find_highest_resolution_image(result, 20)
            if len(result.results) > 30:
                result.results = random.sample(result.results, 30)
        else:
            result.results = result.results[:limit]
        return result

    def _find_highest_resolution_image(self, result: SearchResult, limit: int) -> SearchResult:
        """
        从 SearchResult 中找到分辨率最高的图片。
        :param result: SearchResult 实例，包含多个 SearchResultItem。
        :param limit: 返回分辨率最高的前limit张图片
        :return: 分辨率最高的 SearchResultItem，如果没有有效的分辨率则返回 None。
        """
        resolution_items = []

        for item in result:
            # 提取 resolution 字段，并解析成宽度和高度
            if item.resolution:
                try:
                    width, height = map(int, item.resolution.lower().replace('x', '×').split('×'))
                    area = width * height

                    # 将 (面积, 图片对象) 添加到列表
                    resolution_items.append((area, item))

                except ValueError:
                    # 如果分辨率解析失败，则跳过
                    continue

        # 按面积从大到小排序
        resolution_items.sort(key=lambda x: x[0], reverse=True)

        # 取出分辨率最高的十张图片
        top_items = [item for _, item in resolution_items[:limit]]

        # 如果没有足够的图片，直接返回空结果
        if not top_items:
            result.results = []
            return result

        result.results = top_items
        return result

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
        This function should be used when searching for existing images on the web based on a given query.

        Args:
            query (string): A search query focusing on the topic or keywords of the images (e.g., "nature" for nature-related images). Avoid including general terms like "images," as the search is already scoped to image results.
        """
        logger.info(f"Starting image search for: {query}")
        engines = ["google images", "bing images"]
        result = await self.search(query, categories="images", engines=engines)
        if result and result.results:
            if self.config.get("enable_random_image", False):
                selected_image = random.choice(result.results)
                if self.config.get("enable_image_title", False):
                    chain = []
                    base64_image = await self.download_and_convert_to_base64(selected_image.img_src)
                    if base64_image:
                        chain.append(Image.fromBase64(base64_image))
                    else:
                        yield event.plain_result("下载图片失败。")
                        return
                    chain.append(Plain(f"{selected_image.title}"))
                    yield event.chain_result(chain)
                else:
                    yield event.image_result(selected_image.img_src)
                return
            else:
                # 并发下载多个图片逻辑（优化）
                async def process_image(item):
                    """处理单个图片，包括下载与Base64转换"""
                    base64_image = await self.download_and_convert_to_base64(item.img_src)

                    if base64_image:
                        chain = [Image.fromBase64(base64_image)]
                        if self.config.get("enable_image_title", False):
                            chain.append(Plain(f"{item.title}\n"))
                        return Node(
                            uin=event.get_self_id(),
                            name="IMAGE",
                            content=chain,
                        )
                    return None

                # 利用 asyncio.gather 实现并发图片下载与处理
                tasks = [process_image(item) for idx, item in enumerate(result.results)]
                nodes = await asyncio.gather(*tasks)

                # 去掉空节点
                ns = Nodes([node for node in nodes if node])
                yield event.chain_result([ns])
        else:
            yield event.plain_result("没有找到图片，请稍后再试。")

    @llm_tool("web_search_videos")
    async def search_videos(self, event: AstrMessageEvent, query: str):
        """Search the web for videos

        Args:
            query (string): A search query focusing on the topic or keywords of the videos (e.g., "sports" for sports-related videos). Avoid including general terms like "videos," as the search is already scoped to video results.
        """
        logger.info(f"Starting video search for: {query}")
        result = await self.search(query, categories="videos", limit=5)
        if not result or not result.results:
            logger.error("No videos found.")
            return "No videos found for your query."
        return str(result)

    @llm_tool("web_search_news")
    async def search_news(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for news

        Args:
            query (string): A search query focusing on the topic or keywords of the news (e.g., "sports" for sports-related news). Avoid including general terms like "news," as the search is already scoped to news articles.
        """
        logger.info(f"Starting news search for: {query}")
        results = await self.search(query, categories="news")
        if not results or not results.results:
            return "No news found for your query."
        return str(results)

    @llm_tool("web_search_science")
    async def search_science(self, event: AstrMessageEvent, query: str) -> str:
        """Search the web for scientific information

        Args:
            query (string): A search query focusing on the topic or keywords related to the scientific content (e.g., "quantum mechanics" for research on quantum mechanics). Avoid including general terms like "scientific information," as the search is already scoped to scientific results.
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
            query (string): A search query focusing on the topic or keywords related to music (e.g., "classical composers" for content about classical music). Avoid including general terms like "music," as the search is already scoped to music-related results.
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
            query (string): A search query focusing on specific technical topics or keywords (e.g., "REST API design" for information about API design). Avoid including general terms like "technical details," as the search is already scoped to technical content.
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
            query (string): A search query focusing on specific academic topics, keywords, or fields of study (e.g., "machine learning papers" for research on machine learning). Avoid including general terms like "academic content," as the search is already tailored for academic results.
        """
        logger.info(f"Starting academic search for: {query}")
        results = await self.search(query, categories="academic")
        if not results or not results.results:
            return "No academic information found for your query."
        return str(results)

    @llm_tool("fetch_url")
    async def fetch_website_content(self, event: AstrMessageEvent, url: str):
        """Fetch the content of a website using the provided URL.用于抓取网页，应在查看网页、获取网页时使用。

        Args:
            url(string): The URL of the website to fetch content from.
        """
        logger.info(f"正在通过 fetch_website_content 拉取数据: {url}")
        try:
            parsed_url = urlparse(url)
            if "github.com" in parsed_url.netloc:
                logger.info(f"检测到 GitHub 链接：{url}")

                # 提取 GitHub 链接的路径
                github_path_pattern = re.compile(r"^/([\w\-]+/[\w\-]+)")  # 匹配 owner/repo
                match = github_path_pattern.match(parsed_url.path)
                if match:
                    repo_path = match.group(1)
                    logger.debug(f"提取到 GitHub 仓库路径：{repo_path}")

                    # 调用 GitHub 搜索方法，处理仓库分析逻辑
                    return await self.search_github_repo(event, repo_path)

                # 如果不是仓库路径，返回错误提示
                return "The provided GitHub link is not a valid repository link."

            if not await self._is_url_accessible(url):
                return "Unable to access the URL."

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

    @command("aur")
    async def search_aur(self, event: AstrMessageEvent, query: str):
        """Search packages from the Arch User Repository (AUR).

        Args:
            query (string): The package name or keywords to search for in the AUR.
        """
        logger.info(f"Searching AUR packages for: {query}")
        aur_api_url = "https://aur.archlinux.org/rpc/v5/search"
        params = {"arg": query, "by": "name"}

        if len(query) < 2:
            yield event.plain_result("Search query must be at least 2 characters long.")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(aur_api_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"AUR API request failed with status {response.status}.")
                        yield event.plain_result(f"AUR search failed with status {response.status}.")
                        return

                    data = await response.json()
                    if data.get("type") == "error":
                        logger.error(f"AUR API responded with an error: {data.get('error')}.")
                        yield event.plain_result(f"AUR search error: {data.get('error')}")
                        return

                    results = data.get("results", [])
                    if not results:
                        yield event.plain_result(f"No AUR packages found for query: {query}")
                        return

                    # 精准匹配逻辑
                    exact_match = next((pkg for pkg in results if pkg.get("Name") == query), None)
                    if exact_match:
                        # 如果有精确匹配的包，则返回其详情
                        yield event.plain_result(
                            f"**Package Details**\n"
                            f"Name: {exact_match.get('Name')}\n"
                            f"Description: {exact_match.get('Description')}\n"
                            f"Maintainer: {exact_match.get('Maintainer') or 'N/A'}\n"
                            f"Votes: {exact_match.get('NumVotes')}\n"
                            f"Popularity: {exact_match.get('Popularity')}\n"
                            f"Last Updated: {datetime.fromtimestamp(exact_match.get('LastModified')).strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        return

                    formatted_results = "\n".join(
                        [f"* {item.get('Name')} - {item.get('Description')} (Votes: {item.get('NumVotes')})"
                         for item in results]
                    )
                    yield event.plain_result(formatted_results)
                    return
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during AUR search: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during AUR search: {e}")

    async def _fetch_repo_details(self, session, exact_response, headers):
        """
            Fetch repository details along with README content.

            Args:
                session: The active aiohttp session.
                exact_response: The exact repository API response.
                headers: Headers for subsequent requests.

            Returns:
                str: Detailed repository information (including README if available).
            """
        repo_data = await exact_response.json()
        details = (
            f"**Repository Details**\n"
            f"Name: {repo_data.get('name')}\n"
            f"Full Name: {repo_data.get('full_name')}\n"
            f"Description: {repo_data.get('description')}\n"
            f"Stars: {repo_data.get('stargazers_count')}\n"
            f"Forks: {repo_data.get('forks_count')}\n"
            f"Language: {repo_data.get('language')}\n"
            f"URL: {repo_data.get('html_url')}\n\n"
        )

        # Fetch README content
        readme_url = f"{repo_data['url']}/readme"
        try:
            async with session.get(readme_url, headers=headers) as readme_response:
                if readme_response.status == 200:
                    readme_data = await readme_response.json()
                    readme_content = base64.b64decode(readme_data["content"]).decode("utf-8")
                    details += "**README Content:**\n\n"
                    details += readme_content[:4000]  # Limit README to 4000 characters
                elif readme_response.status == 404:
                    details += "This repository does not have a README file."
                else:
                    details += "Failed to fetch the README content."
        except Exception as e:
            logger.error(f"Error fetching README: {e}")
            details += "An unexpected error occurred while fetching README content."

        return details

    @llm_tool("github_search")
    async def search_github_repo(self, event: AstrMessageEvent, query: str) -> str:
        """用于搜索GitHub仓库，支持直接搜索 GitHub 仓库 URL、克隆链接，以及仓库名称的模糊搜索。

        Args:
            query (string): 可以是 "owner/repo" 格式的仓库名，关键词，GitHub 仓库(URL)，或克隆地址格式。
        """
        import re
        from urllib.parse import urlparse

        search_url = "https://api.github.com/search/repositories"
        headers = {}

        # Optional: Use GitHub Token to increase API limits
        token = self.config.get("github_token", "").strip()
        if token:
            headers["Authorization"] = f"token {token}"

        try:
            # Step 1: 解析输入，检查是否是 GitHub URL，克隆链接，或特定目录路径
            url_pattern = re.compile(r"^https://github\.com/([\w\-]+/[\w\-]+)")  # 匹配 `https://github.com/owner/repo`
            clone_url_pattern = re.compile(r"^(?:git@github\.com:|https://github\.com/)([\w\-]+/[\w\-]+)(?:\.git)?$")

            # 检查是否是完整 GitHub URL (支持 /tree/ 分支/目录等)
            if url_pattern.match(query):
                parsed_url = urlparse(query)
                match = url_pattern.match(parsed_url.path)
                if match:
                    repo_path = match.group(1)  # 提取 "owner/repo"
                    exact_repo_url = f"https://api.github.com/repos/{repo_path}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(exact_repo_url, headers=headers) as exact_response:
                            if exact_response.status == 200:
                                return await self._fetch_repo_details(session, exact_response, headers)
                            elif exact_response.status == 404:
                                return f"Repository '{repo_path}' not found."
                            else:
                                return f"Error while fetching repository: HTTP Status {exact_response.status}."

            # 如果是 `git clone` 格式
            match = clone_url_pattern.match(query)
            if match:
                repo_path = match.group(1)
                exact_repo_url = f"https://api.github.com/repos/{repo_path}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(exact_repo_url, headers=headers) as exact_response:
                        if exact_response.status == 200:
                            return await self._fetch_repo_details(session, exact_response, headers)
                        elif exact_response.status == 404:
                            return f"Repository '{repo_path}' not found."
                        else:
                            return f"Error while fetching repository: HTTP Status {exact_response.status}."

            # 检查是否是 "owner/repo" 格式
            if "/" in query and not query.startswith("http"):
                exact_repo_url = f"https://api.github.com/repos/{query}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(exact_repo_url, headers=headers) as exact_response:
                        if exact_response.status == 200:
                            return await self._fetch_repo_details(session, exact_response, headers)
                        elif exact_response.status == 404:
                            pass  # 如果未找到精确匹配，则降级到模糊搜索
                        else:
                            return f"Error while fetching repository: HTTP Status {exact_response.status}."

            # Step 3: 如果不是特定格式，则执行模糊搜索
            params = {"q": query, "per_page": 5}  # 限制模糊搜索结果为 5 个
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, headers=headers) as search_response:
                    if search_response.status == 200:
                        search_data = await search_response.json()
                        items = search_data.get("items", [])
                        if not items:
                            return f"No repositories found for query: {query}"

                        # 如果只有一个结果，则继续获取详细信息
                        if len(items) == 1:
                            repo_url = items[0]["url"]
                            async with session.get(repo_url, headers=headers) as exact_response:
                                if exact_response.status == 200:
                                    return await self._fetch_repo_details(session, exact_response, headers)

                        # 有多个结果，则以列表形式返回
                        return "\n".join(
                            [f"{i + 1}. **{item['full_name']}** - {item['description'] or 'No description'}"
                             f"  \nClone URL: {item['clone_url']}"
                             for i, item in enumerate(items)]
                        )
                    else:
                        return f"GitHub search failed. HTTP Status: {search_response.status}"

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during GitHub search: {e}")
            return "An error occurred while fetching repository information. Please try again later."
        except Exception as e:
            logger.error(f"Unexpected error during GitHub search: {e}")
            return "An unexpected error occurred. Please try again later."

    @command("github")
    async def github_search(self, event: AstrMessageEvent, query: str = None):
        """GitHub 仓库搜索命令，支持模糊搜索及详细信息查询。

        Args:
            query (str): 可以是完整 GitHub URL，"git clone" 格式，目录 URL，"owner/repo" 格式的仓库名，或关键词。
        """
        if not query:
            yield event.plain_result("Please provide a repository name, URL, or search keywords.")
            return

        logger.info(f"Received GitHub search query: {query}")
        result = await self.search_github_repo(event, query)
        yield event.plain_result(result)



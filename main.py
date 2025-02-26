import asyncio
import base64
import os
import random
import re
from typing import Optional
from urllib.parse import urlparse

import aiohttp
from readability import Document

from astrbot.api import *
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import *
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Image, Plain, Nodes, Node
from data.plugins.astrbot_plugin_web_searcher_pro.search_models import SearchResult, SearchResultItem


@register("web_searcher_pro", "buding", "更高性能的Web检索插件", "1.0.1",
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
                                    resolution=item.get('resolution', ''),
                                    iframe_src=item.get('iframe_src', ''),
                                    content=item.get('content', ''),
                                    engine=item.get('engine', ''),
                                    score=item.get('score', 0.0),
                                )
                                for item in data.get("results", [])
                            ]
                        )
                        return await self._result_filter(result, categories, limit)
                    else:
                        logger.error(f"Failed to search SearxNG. HTTP Status: {response.status}, Params: {params}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during search: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")

    async def _result_filter(self, result: SearchResult, categories: str, limit: int) -> Optional[SearchResult]:
        if categories == "images":
            result.results = result.results[:50]
            urls = [item.img_src for item in result.results if item.img_src]
            validation_results = await asyncio.gather(*[self._is_validate_image_url(url) for url in urls])
            result.results = [
                item for item, is_valid in zip(result.results, validation_results) if is_valid
            ]
            result = self._find_highest_resolution_image(result, 20)
        else:
            result.results = result.results[:limit]
        return result

    async def _is_validate_image_url(self, img_url: str) -> bool:
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
        result = await self.search(query, categories="images")
        if result and result.results:
            if self.config.get("enable_random_image", False):
                selected_image = random.choice(result.results)
                if self.config.get("enable_image_title", False):
                    chain = [
                        Image.fromURL(selected_image.img_src),
                        Plain(f"{selected_image.title}")
                    ]
                    yield event.chain_result(chain)
                else:
                    yield event.image_result(selected_image.img_src)
                return
            else:
                ns = Nodes([])
                for idx, item in enumerate(result.results):
                    if self.config.get("enable_image_title", False):
                        chain = [Plain(f"{item.title}"), Image.fromURL(item.img_src)]
                    else:
                        chain = [Image.fromURL(item.img_src)]

                    node = Node(
                        uin=event.get_self_id(),
                        name="IMAGE",
                        content=chain
                    )
                    ns.nodes.append(node)
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
                        yield event.plain_result (
                            f"**Package Details**\n"
                            f"Name: {exact_match.get('Name')}\n"
                            f"Description: {exact_match.get('Description')}\n"
                            f"Maintainer: {exact_match.get('Maintainer') or 'N/A'}\n"
                            f"Votes: {exact_match.get('NumVotes')}\n"
                            f"Popularity: {exact_match.get('Popularity')}\n"
                            f"Last Updated: {exact_match.get('LastModified')}"
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

    @llm_tool("fetch_url")
    async def fetch_website_content(self, event: AstrMessageEvent, url: str) -> str:
        """Fetch the content of a website using the provided URL.This function must not be used to download e-books.
        When to use:
            1. A user requests the content or summary of a particular website via a query or message.
            2. You need to programmatically retrieve and parse the HTML content of a webpage.
            3. A URL is provided, and content extraction is required.
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

    async def get_book_details(self, book_ids: list) -> Optional[dict]:
        """通过书籍 ID 获取详细信息"""
        DETAIL_API_URL = "https://lgate.glitternode.ru/v1/book"
        headers = {"Content-Type": "application/json"}
        payload = {"book_ids": book_ids}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(DETAIL_API_URL, headers=headers, json=payload, proxy=self.proxy) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", {}).get("book", {})
                    else:
                        logger.error(f"请求书籍详细信息失败，状态码: {response.status}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP 客户端错误: {e}")
        except Exception as e:
            logger.error(f"发生意外错误: {e}")

        return None

    async def search_books_with_details(self, word: str) -> Optional[dict]:
        """搜索书籍并获取前 10 本书籍的详细信息"""
        API_URL = "https://lgate.glitternode.ru/v1/searchV2"
        headers = {"Content-Type": "application/json"}
        payload = {
            "address": "",
            "word": word
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, headers=headers, json=payload, proxy=self.proxy) as response:
                    if response.status == 200:
                        data = await response.json()

                        # 获取书籍 ID 列表
                        book_data = data["data"].get("book", [])
                        if not book_data:
                            logger.info("未找到相关书籍。")
                            return None

                        book_ids = [item.get("id") for item in book_data[:50]]  # 获取前 50 本书籍的 ID
                        if not book_ids:
                            logger.info("未能提取书籍 ID。")
                            return None

                        # 调用详细信息 API
                        detailed_books = await self.get_book_details(book_ids)
                        if not detailed_books:
                            logger.info("未获取书籍详细信息。")
                            return None

                        # 返回包含搜索结果及详细信息的数据
                        return {
                            "search_results": book_data[:50],  # 原始的前 50 本搜索结果
                            "detailed_books": detailed_books  # 完整详细信息
                        }

                    else:
                        logger.error(f"请求书籍搜索失败，状态码: {response.status}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP 客户端错误: {e}")
        except Exception as e:
            logger.error(f"发生意外错误: {e}")

        return None

    @command("search_books")
    async def search_books_command(self, event: AstrMessageEvent, query: str = None):
        """搜索书籍并输出详细信息"""
        if not query:
            yield event.plain_result("请提供书籍关键词以进行搜索。")
            return

        logger.info(f"Received book search query: {query}")
        results = await self.search_books_with_details(query)

        if not results:
            yield event.plain_result("未找到相关书籍。")
            return

        # 输出搜索结果和详细信息
        search_results = results.get("search_results", [])
        detailed_books = results.get("detailed_books", {})

        ns = Nodes([])

        for index, book in enumerate(search_results, start=1):
            book_id = book.get("id")
            detail = detailed_books.get(book_id, {}).get("book", {})

            chain = [
                Plain(f"标题: {book.get('title', '未知')}\n"),
                Plain(f"作者: {book.get('author', '未知')}\n"),
                Plain(f"语言: {detail.get('language', '未知')}\n"),
                Plain(f"文件大小: {detail.get('filesize', '未知')}\n"),
                Plain(f"文件类型: {detail.get('extension', '未知')}\n"),
                Plain(f"年份: {detail.get('year', '未知')}\n"),
                Plain(f"IPFS CID: {detail.get('ipfs_cid', '未知')}\n"),
            ]

            node = Node(
                uin=event.get_self_id(),
                name="BOOK_INFO",
                content=chain
            )
            ns.nodes.append(node)

        yield event.chain_result([ns])

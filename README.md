# WebSearcherPro 插件

更高性能的 Web 检索 AstrBot 插件，支持多种类别的内容搜索以及 URL 内容提取。

# 配置

可参考 `https://docs.openwebui.com/tutorials/web-search/searxng` 部署SearXNG，细节较多，建议熟悉Linux和命令行的同学尝试。

# 技术实现

插件使用 Python 编写，基于 AstrBot 提供了多种高性能检索功能，支持灵活扩展和定制。以下是一些主要实现特点：

- 异步操作：通过 `aiohttp` 和 `asyncio` 实现高效的 HTTP 请求和任务并发处理。
- 自定义过滤：支持对检索结果进行过滤，比如基于分辨率筛选图片。
- 丰富的指令：包括网页搜索、图片搜索、GitHub 仓库查询、书籍详细信息检索等。
- API 综合利用：对于不同类别的数据检索，使用了诸如 SearXNG、Liber3 和 GitHub 等不同的 API。
- 可配置：支持通过配置文件添加代理、自定义 API 地址，并且某些功能具有随机性（如随机图片选择）。

# 使用方法

1. 初始化 AstrBot 并安装 WebSearcherPro 插件。
2. 配置 `searxng_api_url` 等选项。
3. 使用命令控制功能（如 `/websearch on` 开启网页搜索功能）。
4. 通过自然语言调用对应指令实现功能。

# 指令列表

| 指令/自然语言           | 参数           | 示例               | 说明                |
|-------------------|--------------|------------------|-------------------|
| `/websearch`      | `on/off` 或 空 | `/websearch on`  | 开启/关闭网页搜索功能       |
| `websearch_images` | `关键词`        | `搜索图片 晴空塔`       | 搜索相关图片            |
| `websearch_videos` | `关键词`        | `搜索视频 哆啦A梦`      | 搜索相关视频            |
| `websearch_news`  | `关键词`        | `搜索新闻 冬奥会`       | 搜索相关新闻            |
| `websearch_music` | `关键词`        | `搜索音乐 许嵩`        | 搜索相关音乐资源          |
| `websearch_science` | `关键词`        | `搜索科学 黑洞`        | 搜索科学内容            |
| `websearch_academic` | `关键词`        | `搜索学术 MLP`       | 搜索学术论文或者学术资料      |
| `/github`         | `URL/关键词`    | `/github flask`  | 搜索 GitHub 仓库      |
| `/liber3`         | `书籍关键词`      | `/liber3 python` | 从 Liber3 API 搜索书籍 |

# 注意事项

1. **API 配额**：某些功能依赖外部 API（如 GitHub），其接口访问频率可能会受限。建议配置相应 Token 提升配额容量。
2. **图片搜索标题**：若启用图片标题功能（`enable_image_title: true`），图片结果会包含标题，默认关闭。

# 问题反馈

如出现问题，请通过以下途径联系：
QQ: 1259085392，或者提交 GitHub
Issue：[WebSearcherPro 仓库](https://github.com/zouyonghe/astrbot_plugin_web_searcher_pro/issues)
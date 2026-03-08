# WebSearcherPro 插件

基于 SearXNG 的 AstrBot Web 检索插件，支持网页搜索、图片搜索、网页正文抓取、GitHub 仓库查询和 AUR 包检索。

## LLM 工具命名

为避免与其他插件发生 `llm_tool` 名称碰撞，插件导出的工具名已统一改为 `searxng_*` 前缀：

- `searxng_web_search_general`
- `searxng_web_search_images`
- `searxng_web_search_videos`
- `searxng_web_search_news`
- `searxng_web_search_science`
- `searxng_web_search_music`
- `searxng_web_search_technical`
- `searxng_web_search_academic`
- `searxng_web_search_map`
- `searxng_web_search_files`
- `searxng_web_search_social`
- `searxng_web_search_books`
- `searxng_web_fetch_url`
- `searxng_github_search`

## 架构

重构后代码按职责拆分：

- `main.py`：AstrBot 插件入口、命令注册、工具绑定
- `config.py`：配置加载与默认值
- `services/`：SearXNG、GitHub、网页抓取、图片处理、AUR、HTTP 客户端
- `formatters/`：统一文本格式化
- `utils/`：URL 解析等纯函数
- `tests/`：可在 `astrbot` 环境中运行的单测

## 配置

可参考 `https://docs.openwebui.com/tutorials/web-search/searxng` 部署 SearXNG。

当前支持的关键配置：

- `searxng_api_url`：SearXNG API 地址
- `request_timeout`：统一网络请求超时秒数
- `enable_random_image`：图片搜索是否只返回一张随机图
- `enable_image_title`：图片结果是否附带标题
- `image_result_limit`：图片搜索最终返回数量上限
- `image_candidate_limit`：图片搜索预筛选候选数量
- `github_token`：提高 GitHub API 配额

## 使用方法

1. 安装插件并配置 `searxng_api_url`。
2. 使用 `/websearch on` 开启网页搜索工具。
3. 通过命令或自然语言触发对应能力。

## 指令列表

| 指令/工具 | 参数 | 示例 | 说明 |
|---|---|---|---|
| `/websearch` | `on/off` 或 空 | `/websearch on` | 开启/关闭网页搜索能力 |
| `searxng_web_search_general` | `关键词` | `搜索 Python asyncio` | 通用网页搜索 |
| `searxng_web_search_images` | `关键词` | `搜索图片 晴空塔` | 图片搜索 |
| `searxng_web_search_videos` | `关键词` | `搜索视频 哆啦A梦` | 视频搜索 |
| `searxng_web_search_news` | `关键词` | `搜索新闻 冬奥会` | 新闻搜索 |
| `searxng_web_search_science` | `关键词` | `搜索科学 黑洞` | 科学搜索 |
| `searxng_web_search_technical` | `关键词` | `搜索技术 Python asyncio` | 技术搜索，优先开发者结果并自动回退 |
| `searxng_web_search_academic` | `关键词` | `搜索学术 MLP` | 学术搜索 |
| `searxng_web_search_map` | `地点/关键词` | `搜索地图 上海迪士尼` | 地图搜索 |
| `searxng_web_search_files` | `关键词` | `搜索文件 Python PDF 教程` | 文件搜索 |
| `searxng_web_search_social` | `关键词` | `搜索社交 Mastodon OpenAI` | 社交内容搜索 |
| `searxng_web_search_books` | `关键词` | `搜索图书 深度学习` | 图书搜索 |
| `searxng_web_fetch_url` | `URL` | `获取 https://example.com` | 提取网页正文或 GitHub 仓库内容 |
| `/github` | `URL/关键词` | `/github flask` | 搜索 GitHub 仓库 |
| `searxng_github_search` | `URL/关键词` | `搜索 GitHub owner/repo` | GitHub 仓库工具 |
| `/aur` | `包名/关键词` | `/aur firefox` | 查询 AUR 包 |

## 搜索行为

- 文本搜索会根据查询语言自动推断搜索语言；混合或不明确的查询会回退为 `auto`。
- `searxng_web_search_technical` 会优先使用更适合开发者内容的分类，并在结果不足时按策略自动回退，提升技术文档与资料命中率。
- `searxng_web_search_map`、`searxng_web_search_files`、`searxng_web_search_social`、`searxng_web_search_books` 分别用于地图、文件、社交内容和图书搜索。

## 开发与测试

建议在 `mamba` 的 `astrbot` 环境内执行：

- 安装运行依赖：`pip install -r requirements.txt`
- 安装测试依赖：`pip install -r requirements-dev.txt`
- 运行测试：`pytest -q`

## 注意事项

1. GitHub、网页抓取、AUR 等能力依赖外部服务，建议配置合理超时与代理。
2. 图片搜索会先做可访问性校验，再按分辨率和搜索得分排序。
3. 若启用 `enable_random_image`，图片模式只返回一张结果。

## 问题反馈

如出现问题，请通过以下途径联系：
QQ: 1259085392，或者提交 GitHub Issue：
[WebSearcherPro 仓库](https://github.com/zouyonghe/astrbot_plugin_web_searcher_pro/issues)

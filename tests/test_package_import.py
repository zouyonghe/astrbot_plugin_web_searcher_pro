import asyncio
import importlib
import re
import sys
import types
from pathlib import Path


MAIN_MODULE_NAME = "data.plugins.astrbot_plugin_web_searcher_pro.main"


def _stub_module(monkeypatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _import_plugin_main(monkeypatch):
    project_root = Path(__file__).resolve().parents[4]
    plugin_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if Path(entry or ".").resolve() != plugin_root],
    )
    monkeypatch.syspath_prepend(str(project_root))

    for module_name in [
        MAIN_MODULE_NAME,
        "astrbot",
        "astrbot.api",
        "astrbot.api.event",
        "astrbot.api.event.filter",
        "astrbot.api.star",
        "astrbot.core",
        "astrbot.core.message",
        "astrbot.core.message.components",
        "aiohttp",
        "bs4",
        "PIL",
        "readability",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    astrbot = _stub_module(monkeypatch, "astrbot")
    api = _stub_module(monkeypatch, "astrbot.api")
    event = _stub_module(monkeypatch, "astrbot.api.event")
    event_filter = _stub_module(monkeypatch, "astrbot.api.event.filter")
    star = _stub_module(monkeypatch, "astrbot.api.star")
    core = _stub_module(monkeypatch, "astrbot.core")
    core_message = _stub_module(monkeypatch, "astrbot.core.message")
    components = _stub_module(monkeypatch, "astrbot.core.message.components")
    aiohttp = _stub_module(monkeypatch, "aiohttp")
    bs4 = _stub_module(monkeypatch, "bs4")
    pil = _stub_module(monkeypatch, "PIL")
    readability = _stub_module(monkeypatch, "readability")

    class DummyStar:
        def __init__(self, context=None):
            self.context = context

    class DummyContext:
        pass

    class DummyConfig(dict):
        pass

    def register(*args, **kwargs):
        def decorator(obj):
            return obj
        return decorator

    def llm_tool(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def command(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
        def error(self, *args, **kwargs):
            pass

    class DummyClientError(Exception):
        pass

    class DummyClientTimeout:
        def __init__(self, *args, **kwargs):
            pass

    class DummyClientSession:
        def __init__(self, *args, **kwargs):
            pass

    class DummyDocument:
        def __init__(self, html):
            self.html = html

        def summary(self):
            return self.html

    api.__dict__.update(
        logger=DummyLogger(),
        AstrBotConfig=DummyConfig,
        llm_tool=llm_tool,
        command=command,
        __all__=["logger", "AstrBotConfig", "llm_tool", "command"],
    )
    event.__dict__.update(AstrMessageEvent=type("AstrMessageEvent", (), {}))
    star.__dict__.update(Context=DummyContext, Star=DummyStar, register=register)
    components.__dict__.update(
        Image=type("Image", (), {"fromBase64": staticmethod(lambda value: value)}),
        Node=type("Node", (), {}),
        Nodes=type("Nodes", (), {}),
        Plain=type("Plain", (), {}),
    )
    aiohttp.__dict__.update(
        ClientError=DummyClientError,
        ClientTimeout=DummyClientTimeout,
        ClientSession=DummyClientSession,
    )
    bs4.__dict__.update(BeautifulSoup=type("BeautifulSoup", (), {}))
    pil.__dict__.update(Image=type("PilImage", (), {"open": staticmethod(lambda value: value)}))
    readability.__dict__.update(Document=DummyDocument)

    return importlib.import_module(MAIN_MODULE_NAME)


def test_plugin_main_imports_as_package(monkeypatch):
    module = _import_plugin_main(monkeypatch)

    assert hasattr(module, "WebSearcherPro")


def test_llm_tools_keep_parameter_docstrings(monkeypatch):
    module = _import_plugin_main(monkeypatch)
    expected_parameters = {
        "search_general": "query",
        "search_images": "query",
        "search_videos": "query",
        "search_news": "query",
        "search_science": "query",
        "search_music": "query",
        "search_technical": "query",
        "search_academic": "query",
        "search_map": "query",
        "search_files": "query",
        "search_social": "query",
        "search_books": "query",
        "fetch_website_content": "url",
        "search_github_repo": "query",
    }

    for method_name, parameter_name in expected_parameters.items():
        method = getattr(module.WebSearcherPro, method_name)
        docstring = method.__doc__

        assert docstring, f"{method_name} should keep a docstring for tool metadata"
        assert "Args:" in docstring, f"{method_name} should document parameters for the LLM tool"
        assert re.search(rf"{parameter_name}\s*\((?:str|string)\)\s*:", docstring), (
            f"{method_name} should describe the {parameter_name} parameter"
        )


class DummyProviderConfig(dict):
    def __init__(self):
        super().__init__(provider_settings={"web_search": False})
        self.saved = False

    def save_config(self):
        self.saved = True


class DummyContext:
    def __init__(self):
        self.config = DummyProviderConfig()
        self.activated_tools = []
        self.deactivated_tools = []

    def get_config(self):
        return self.config

    def activate_llm_tool(self, tool_name: str):
        self.activated_tools.append(tool_name)

    def deactivate_llm_tool(self, tool_name: str):
        self.deactivated_tools.append(tool_name)


def test_set_websearch_status_toggles_all_websearch_tools(monkeypatch):
    module = _import_plugin_main(monkeypatch)
    plugin = object.__new__(module.WebSearcherPro)
    plugin.context = DummyContext()

    plugin._set_websearch_status(True)

    expected_tools = {
        "searxng_web_search_general",
        "searxng_web_search_images",
        "searxng_web_search_videos",
        "searxng_web_search_news",
        "searxng_web_search_science",
        "searxng_web_search_music",
        "searxng_web_search_technical",
        "searxng_web_search_academic",
        "searxng_web_search_map",
        "searxng_web_search_files",
        "searxng_web_search_social",
        "searxng_web_search_books",
        "searxng_web_fetch_url",
    }

    assert plugin.context.config["provider_settings"]["web_search"] is True
    assert plugin.context.config.saved is True
    assert set(plugin.context.activated_tools) == expected_tools
    assert len(plugin.context.activated_tools) == len(expected_tools)

    plugin._set_websearch_status(False)

    assert plugin.context.config["provider_settings"]["web_search"] is False
    assert set(plugin.context.deactivated_tools) == expected_tools
    assert len(plugin.context.deactivated_tools) == len(expected_tools)


def test_new_text_search_tools_forward_category_and_empty_message(monkeypatch):
    module = _import_plugin_main(monkeypatch)
    expectations = {
        "search_map": ("map", module.MAP_EMPTY_MESSAGE),
        "search_files": ("files", module.FILES_EMPTY_MESSAGE),
        "search_social": ("social", module.SOCIAL_EMPTY_MESSAGE),
        "search_books": ("books", module.BOOKS_EMPTY_MESSAGE),
    }

    for method_name, (expected_category, expected_empty_message) in expectations.items():
        plugin = object.__new__(module.WebSearcherPro)
        calls = []

        async def fake_search_text_category(query: str, *, category: str, empty_message: str, limit: int = 5) -> str:
            calls.append((query, category, empty_message, limit))
            return f"forwarded:{category}"

        monkeypatch.setattr(plugin, "_search_text_category", fake_search_text_category, raising=False)

        result = asyncio.run(getattr(module.WebSearcherPro, method_name)(plugin, None, "test query"))

        assert result == f"forwarded:{expected_category}"
        assert calls == [("test query", expected_category, expected_empty_message, 5)]

import importlib
import re
import sys
import types
from pathlib import Path


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _import_plugin_main(monkeypatch):
    project_root = Path(__file__).resolve().parents[4]
    plugin_root = Path(__file__).resolve().parents[1]
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != plugin_root]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    astrbot = _stub_module("astrbot")
    api = _stub_module("astrbot.api")
    event = _stub_module("astrbot.api.event")
    event_filter = _stub_module("astrbot.api.event.filter")
    star = _stub_module("astrbot.api.star")
    core = _stub_module("astrbot.core")
    core_message = _stub_module("astrbot.core.message")
    components = _stub_module("astrbot.core.message.components")
    aiohttp = _stub_module("aiohttp")
    bs4 = _stub_module("bs4")
    pil = _stub_module("PIL")
    readability = _stub_module("readability")

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

    monkeypatch.syspath_prepend(str(project_root))
    sys.modules.pop("data.plugins.astrbot_plugin_web_searcher_pro.main", None)

    return importlib.import_module("data.plugins.astrbot_plugin_web_searcher_pro.main")


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

import importlib
import sys
import types
from pathlib import Path


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def test_plugin_main_imports_as_package(monkeypatch):
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

    api.logger = DummyLogger()
    api.AstrBotConfig = DummyConfig
    api.llm_tool = llm_tool
    api.command = command
    api.__all__ = ["logger", "AstrBotConfig", "llm_tool", "command"]
    event.AstrMessageEvent = type("AstrMessageEvent", (), {})
    star.Context = DummyContext
    star.Star = DummyStar
    star.register = register
    components.Image = type("Image", (), {"fromBase64": staticmethod(lambda value: value)})
    components.Node = type("Node", (), {})
    components.Nodes = type("Nodes", (), {})
    components.Plain = type("Plain", (), {})

    monkeypatch.syspath_prepend(str(project_root))
    sys.modules.pop("data.plugins.astrbot_plugin_web_searcher_pro.main", None)

    module = importlib.import_module("data.plugins.astrbot_plugin_web_searcher_pro.main")

    assert hasattr(module, "WebSearcherPro")

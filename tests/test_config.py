from astrbot_plugin_web_searcher_pro.config import PluginConfig


class DummyConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_plugin_config_reads_defaults():
    config = PluginConfig.from_mapping(DummyConfig())

    assert config.searxng_api_url == "http://127.0.0.1:8080"
    assert config.enable_random_image is False
    assert config.enable_image_title is True
    assert config.github_token == ""

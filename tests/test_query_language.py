from astrbot_plugin_web_searcher_pro.utils.query_language import detect_query_language


def test_detect_query_language_for_chinese_text():
    assert detect_query_language("\u600e\u4e48\u7528 asyncio") == "zh"


def test_detect_query_language_for_english_text():
    assert detect_query_language("python asyncio tutorial") == "en"


def test_detect_query_language_for_japanese_text():
    assert detect_query_language("\u6771\u4eac\u30b9\u30ab\u30a4\u30c4\u30ea\u30fc") == "ja"


def test_detect_query_language_for_korean_text():
    assert detect_query_language("\ud30c\uc774\uc36c \ube44\ub3d9\uae30") == "ko"


def test_detect_query_language_for_mixed_text_falls_back_to_auto():
    assert detect_query_language("OpenAI \u6700\u65b0 news") == "auto"

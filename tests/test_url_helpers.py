from astrbot_plugin_web_searcher_pro.utils.url_helpers import extract_github_repo, is_http_url


def test_extract_github_repo_from_tree_url():
    repo = extract_github_repo("https://github.com/owner/repo/tree/main/src")
    assert repo == "owner/repo"


def test_extract_github_repo_from_clone_url():
    repo = extract_github_repo("git@github.com:owner/repo.git")
    assert repo == "owner/repo"


def test_is_http_url_rejects_plain_text():
    assert is_http_url("not a url") is False

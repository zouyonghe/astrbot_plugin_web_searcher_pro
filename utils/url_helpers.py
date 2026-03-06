import re
from urllib.parse import urlparse


GITHUB_REPO_PATH = re.compile(r"^/([\w.-]+/[\w.-]+)")
GITHUB_CLONE_PATTERN = re.compile(r"^(?:git@github\.com:|https://github\.com/)([\w.-]+/[\w.-]+?)(?:\.git)?$")


def is_http_url(value: str) -> bool:
    if not value or not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def extract_github_repo(value: str) -> str | None:
    if not value:
        return None

    clone_match = GITHUB_CLONE_PATTERN.match(value)
    if clone_match:
        return clone_match.group(1)

    if is_http_url(value):
        parsed = urlparse(value)
        if "github.com" not in parsed.netloc:
            return None
        match = GITHUB_REPO_PATH.match(parsed.path)
        return match.group(1) if match else None

    if "/" in value and not value.startswith("http"):
        owner, repo, *_ = value.split("/") + [""]
        if owner and repo:
            return f"{owner}/{repo.removesuffix('.git')}"

    return None

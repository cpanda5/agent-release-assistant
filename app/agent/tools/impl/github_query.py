"""
Three read-only GitHub tools backed by the public GitHub REST API via httpx.

Queries only — no writes, no branch/PR creation. Anonymous access is capped at
60 requests/hour; set ``GITHUB_TOKEN`` for 5000/hour. ``require_confirmation``
is False because these operations are safe to execute without a user gate.
"""
import httpx

from app.agent.tools.base import BaseTool, ToolContext, ToolResult
from app.config import settings

_GITHUB_API = "https://api.github.com"
_TIMEOUT = 15.0


def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "agent-release-assistant",
    }
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


async def _gh_get(path: str, params: dict | None = None) -> dict | list:
    """GET /path against the GitHub API. Raises httpx.HTTPStatusError on 4xx/5xx."""
    url = path if path.startswith("http") else f"{_GITHUB_API}{path}"
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers=_headers(), params=params)
        resp.raise_for_status()
        return resp.json()


def _format_error(status: int, body: str) -> str:
    return f"GitHub API 请求失败 (HTTP {status})：{body[:200]}"


# ── 1. github_search ────────────────────────────────────────────────────── #

class GitHubSearchTool(BaseTool):
    """
    Search GitHub issues / pull requests / code / repositories via the
    Search API (``GET /search/{entity}``). Returns up to ``per_page`` matches.
    """

    name = "github_search"
    description = (
        "在 GitHub 上搜索 issues、pull requests、代码或仓库。"
        "查询语法遵循 GitHub Search（如 'repo:owner/name is:issue state:open bug'）。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "GitHub 搜索查询（支持 qualifiers，如 repo:owner/name、is:issue、state:open 等）",
            },
            "type": {
                "type": "string",
                "enum": ["issues", "repositories", "code", "pullrequests"],
                "default": "issues",
                "description": "要搜索的对象类型。pullrequests 使用 issues 端点并加 is:pr 限定。",
            },
            "per_page": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "返回的结果数量",
            },
        },
        "required": ["query"],
    }
    timeout_seconds = _TIMEOUT
    require_confirmation = False

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        query = params.get("query", "").strip()
        if not query:
            return ToolResult(success=False, content="query 不能为空")

        search_type = params.get("type", "issues")
        per_page = min(max(int(params.get("per_page", 5)), 1), 20)

        # pullrequests maps to the issues endpoint with an `is:pr` qualifier.
        if search_type == "pullrequests":
            endpoint = "/search/issues"
            if "is:pr" not in query and "is:pull_request" not in query:
                query = f"{query} is:pull_request"
        elif search_type == "issues":
            endpoint = "/search/issues"
        elif search_type == "repositories":
            endpoint = "/search/repositories"
        elif search_type == "code":
            endpoint = "/search/code"
        else:
            return ToolResult(success=False, content=f"不支持的搜索类型：{search_type}")

        try:
            data = await _gh_get(
                endpoint,
                params={"q": query, "per_page": per_page, "sort": "updated", "order": "desc"},
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, content=_format_error(e.response.status_code, e.response.text))
        except httpx.HTTPError as e:
            return ToolResult(success=False, content=f"GitHub API 请求异常：{e}")

        items = data.get("items", []) if isinstance(data, dict) else []
        if not items:
            return ToolResult(
                success=True,
                content=f"未找到与 \"{query}\" 匹配的结果（total_count={data.get('total_count', 0) if isinstance(data, dict) else 0}）。",
            )

        # Condense each hit to the fields the LLM actually needs.
        rows: list[str] = []
        for it in items:
            if search_type == "repositories":
                rows.append(
                    f"- {it.get('full_name')} ★{it.get('stargazers_count', 0)} "
                    f"| {it.get('description') or '(no description)'} "
                    f"| {it.get('html_url')}"
                )
            elif search_type == "code":
                repo = it.get("repository", {}).get("full_name", "?")
                rows.append(f"- {repo}:{it.get('path')} | {it.get('html_url')}")
            else:  # issues / pull requests
                rows.append(
                    f"- #{it.get('number')} [{it.get('state')}] {it.get('title')} "
                    f"| {it.get('html_url')}"
                )
        return ToolResult(
            success=True,
            content=(
                f"搜索类型={search_type}，共 {data.get('total_count', len(items))} 条结果，"
                f"展示前 {len(rows)} 条：\n" + "\n".join(rows)
            ),
        )


# ── 2. github_get_issue ─────────────────────────────────────────────────── #

class GitHubGetIssueTool(BaseTool):
    """Fetch a single issue by repo + number."""

    name = "github_get_issue"
    description = "获取一个 GitHub issue 的详细信息（标题、状态、作者、正文、评论数）。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "仓库，格式 owner/repo",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue 编号",
            },
        },
        "required": ["repo", "issue_number"],
    }
    timeout_seconds = _TIMEOUT
    require_confirmation = False

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        repo = (params.get("repo") or "").strip()
        number = params.get("issue_number")
        if not repo or "/" not in repo:
            return ToolResult(success=False, content="repo 必须为 owner/repo 格式")
        if not isinstance(number, int) or number <= 0:
            return ToolResult(success=False, content="issue_number 必须为正整数")

        try:
            data = await _gh_get(f"/repos/{repo}/issues/{number}")
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, content=_format_error(e.response.status_code, e.response.text))
        except httpx.HTTPError as e:
            return ToolResult(success=False, content=f"GitHub API 请求异常：{e}")

        # Issues endpoint also serves PRs; surface PR-ness so the LLM can
        # redirect the user to github_get_pr if needed.
        is_pr = "pull_request" in data
        body = (data.get("body") or "")[:2000]
        labels = ", ".join(l.get("name", "") for l in data.get("labels", []))
        summary = (
            f"#{data.get('number')} {(data.get('title'))} "
            f"({'PR' if is_pr else 'Issue'})\n"
            f"状态: {data.get('state')}\n"
            f"作者: {data.get('user', {}).get('login')}\n"
            f"标签: {labels or '(无)'}\n"
            f"评论数: {data.get('comments')}\n"
            f"创建: {data.get('created_at')}  更新: {data.get('updated_at')}\n"
            f"链接: {data.get('html_url')}\n\n"
            f"正文:\n{body}"
        )
        return ToolResult(success=True, content=summary.strip())


# ── 3. github_get_pr ───────────────────────────────────────────────────── #

class GitHubGetPRTool(BaseTool):
    """Fetch a single pull request with files changed summary."""

    name = "github_get_pr"
    description = (
        "获取一个 GitHub Pull Request 的详细信息（标题、状态、作者、正文、"
        "变更文件数、增删行数）。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "仓库，格式 owner/repo",
            },
            "pr_number": {
                "type": "integer",
                "description": "Pull Request 编号",
            },
        },
        "required": ["repo", "pr_number"],
    }
    timeout_seconds = _TIMEOUT
    require_confirmation = False

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        repo = (params.get("repo") or "").strip()
        number = params.get("pr_number")
        if not repo or "/" not in repo:
            return ToolResult(success=False, content="repo 必须为 owner/repo 格式")
        if not isinstance(number, int) or number <= 0:
            return ToolResult(success=False, content="pr_number 必须为正整数")

        try:
            pr = await _gh_get(f"/repos/{repo}/pulls/{number}")
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, content=_format_error(e.response.status_code, e.response.text))
        except httpx.HTTPError as e:
            return ToolResult(success=False, content=f"GitHub API 请求异常：{e}")

        # Also fetch the list of files changed — useful for code review.
        files_info = ""
        try:
            files = await _gh_get(
                f"/repos/{repo}/pulls/{number}/files",
                params={"per_page": 30},
            )
            file_rows = []
            for f in files[:30]:
                file_rows.append(
                    f"  - {f.get('status')} {f.get('filename')} "
                    f"(+{f.get('additions', 0)}/-{f.get('deletions', 0)})"
                )
            files_info = "\n".join(file_rows)
            if len(files) > 30:
                files_info += f"\n  …还有 {len(files) - 30} 个文件未列出"
        except httpx.HTTPError:
            files_info = "(获取变更文件列表失败)"

        body = (pr.get("body") or "")[:2000]
        summary = (
            f"PR #{pr.get('number')} {pr.get('title')}\n"
            f"状态: {pr.get('state')}  可合并: {pr.get('mergeable')}\n"
            f"作者: {pr.get('user', {}).get('login')}\n"
            f"分支: {pr.get('head', {}).get('ref')} → {pr.get('base', {}).get('ref')}\n"
            f"变更: +{pr.get('additions', 0)}/-{pr.get('deletions', 0)}，"
            f"{pr.get('changed_files', 0)} 个文件\n"
            f"评论: {pr.get('comments')}  review 评论: {pr.get('review_comments')}\n"
            f"创建: {pr.get('created_at')}  更新: {pr.get('updated_at')}\n"
            f"链接: {pr.get('html_url')}\n\n"
            f"正文:\n{body}\n\n"
            f"变更文件:\n{files_info}"
        )
        return ToolResult(success=True, content=summary.strip())

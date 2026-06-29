"""Lightweight web search context for chat sessions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
import json
import re
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

if TYPE_CHECKING:
    from src.chat.config import ChatConfig


ANCHOR_RE = re.compile(
    r"<a\b(?=[^>]*class=['\"]result-link['\"])(?=[^>]*href=['\"]([^'\"]+)['\"])[^>]*>"
    r"(.*?)</a>",
    re.DOTALL | re.IGNORECASE,
)
SNIPPET_RE = re.compile(
    r"<td[^>]+class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
    re.DOTALL | re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")

MANUAL_SEARCH_PATTERNS = (
    "検索",
    "調べ",
    "ググ",
    "web",
    "ウェブ",
    "ネット",
    "ソース",
    "出典",
)

CURRENT_INFO_PATTERNS = (
    "最新",
    "現在",
    "いま",
    "今 ",
    "今年",
    "ニュース",
    "価格",
    "値段",
    "相場",
    "株価",
    "為替",
    "天気",
    "予定",
    "発売",
    "リリース",
    "アップデート",
    "障害",
    "不具合",
    "バージョン",
    "仕様",
    "CEO",
    "首相",
    "大統領",
)

LOCAL_CONTEXT_PATTERNS = (
    "このpc",
    "このPC",
    "この環境",
    "このリポジトリ",
    "このrepo",
    "このコード",
    "ここに",
    "ローカル",
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchContext:
    """Auto-searches when a prompt likely needs current external information."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        auto: bool = True,
        max_results: int = 4,
        timeout_sec: float = 8.0,
        cache_path: str | Path | None = None,
    ):
        self.enabled = enabled
        self.auto = auto
        self.max_results = max(1, min(max_results, 8))
        self.timeout_sec = timeout_sec
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[dict]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def build_context_prompt(self, query: str) -> str:
        if not self.enabled or not query.strip():
            return ""
        if self.auto and not should_search(query):
            return ""

        results = self.search(query)
        if not results:
            return (
                "\n\n[Web検索]\n"
                f"検索クエリ: {query}\n"
                "検索を試したが、使える結果は取れなかった。必要なら断定せずに返すこと。\n"
            )

        lines = [
            "\n\n[Web検索結果]",
            f"検索日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"検索クエリ: {query}",
            "以下は自動検索で得た外部情報。使う場合は、結果に基づくことを短く示し、結果にない内容は断定しないこと。",
        ]
        for i, result in enumerate(results[: self.max_results], 1):
            lines.extend(
                [
                    f"{i}. {result.title}",
                    f"   URL: {result.url}",
                    f"   概要: {result.snippet}",
                ]
            )
        return "\n".join(lines) + "\n"

    def search(self, query: str) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        cache_key = query.lower()
        if cache_key in self._cache:
            return [SearchResult(**item) for item in self._cache[cache_key]][: self.max_results]

        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        try:
            with httpx.Client(timeout=self.timeout_sec, follow_redirects=True, headers=headers) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except Exception:
            return []

        results = parse_duckduckgo_lite(resp.text, max_results=self.max_results)
        self._cache[cache_key] = [r.__dict__ for r in results]
        self._save_cache()
        return results

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def should_search(text: str) -> bool:
    lowered = text.lower()
    if any(pattern.lower() in lowered for pattern in LOCAL_CONTEXT_PATTERNS):
        return any(pattern.lower() in lowered for pattern in MANUAL_SEARCH_PATTERNS)
    if URL_RE.search(text):
        return True
    if any(pattern.lower() in lowered for pattern in MANUAL_SEARCH_PATTERNS):
        return True
    if any(pattern.lower() in lowered for pattern in CURRENT_INFO_PATTERNS):
        return True
    if re.search(r"\b20[2-9][0-9]\b", text):
        return True
    return False


def parse_duckduckgo_lite(html: str, *, max_results: int = 4) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen: set[str] = set()
    matches = list(ANCHOR_RE.finditer(html))
    for index, match in enumerate(matches):
        raw_url, raw_title = match.groups()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(html)
        snippet_match = SNIPPET_RE.search(html, match.end(), end)
        raw_snippet = snippet_match.group(1) if snippet_match else ""
        url = decode_duckduckgo_url(raw_url)
        title = clean_html_text(raw_title)
        snippet = clean_html_text(raw_snippet)
        if not url or not title or url in seen:
            continue
        seen.add(url)
        results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


def clean_html_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = unescape(text)
    return SPACE_RE.sub(" ", text).strip()


def decode_duckduckgo_url(url: str) -> str:
    url = unescape(url).strip()
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return uddg[0]
    return url


def create_web_search_context(config: "ChatConfig") -> WebSearchContext | None:
    if not getattr(config, "web_search_enabled", False):
        return None
    cache_path = getattr(config, "web_search_cache_path", "data/web_search_cache.json")
    return WebSearchContext(
        enabled=True,
        auto=getattr(config, "web_search_auto", True),
        max_results=getattr(config, "web_search_max_results", 4),
        timeout_sec=getattr(config, "web_search_timeout_sec", 8.0),
        cache_path=Path(cache_path) if cache_path else None,
    )

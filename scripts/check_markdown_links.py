#!/usr/bin/env python3
"""Check that relative local links in Markdown files resolve to existing files.

Offline-only checker used by the CI quality gate:

- only relative local links are validated (resolved from the containing file)
- absolute URIs are ignored: any RFC-style scheme is detected case-insensitively
- pure anchor links (``#section``) are ignored
- site-root absolute links (``/path``) and Windows absolute paths are ignored
  (drive-letter, UNC ``\\\\`` and rooted ``\\`` paths)
- pseudo-links inside fenced code blocks are ignored
- resolved targets must stay inside the repository root and be regular files;
  path/symlink escapes and directory targets are rejected
- unreadable files are reported as issues (non-zero exit)

Exits non-zero and prints ``path:line: broken link -> target`` for every
failure so the offending location is actionable.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

_FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})")

# CommonMark backslash escapes apply only to ASCII punctuation.
_ESCAPABLE_PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

# Destination alternatives follow CommonMark closely enough:
#   - bare: no whitespace, angle brackets or quotes; parentheses allowed when
#     balanced to at least one nested level; backslash escapes supported
#   - angle-bracketed: any characters except ">" and newline
_CHAR_DEST = r"(?:\\.|[^()<>\s\n\"'])"
_CHAR_INNER = r"(?:\\.|[^()\n])"
_DEST_BARE = _CHAR_DEST + r"*(?:\(" + _CHAR_INNER + r"*\)" + _CHAR_DEST + r"*)*"
_TITLE = r"""(?:"(?:\\.|[^"\n])*"|'(?:\\.|[^'\n])*'|\((?:\\.|[^)\n])*\))"""
_LINK_PATTERN = re.compile(
    r"(?<!\\)!?\[[^\]\n]*\]\(\s*"
    r"(?:<(?P<angle>(?:\\.|[^>\n])*)>|(?P<dest>" + _DEST_BARE + r"))"
    r"(?:\s+(?P<title>" + _TITLE + r"))?\s*\)"
)


def _is_windows_absolute(target: str) -> bool:
    """True for drive-letter, UNC and single-backslash rooted Windows paths."""
    if target.startswith("\\") or target.startswith("//"):
        return True
    return (
        len(target) >= 3
        and target[0].isalpha()
        and target[1] == ":"
        and target[2] in "\\/"
    )


def _is_url(target: str) -> bool:
    """True when urllib recognises an RFC-style URI scheme (case-insensitive)."""
    return bool(urlparse(target).scheme)


def _unescape(text: str) -> str:
    """Remove CommonMark backslash escapes for ASCII punctuation only.

    A backslash before an ordinary (non-punctuation) character is kept as a
    literal backslash.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char == "\\" and i + 1 < n and text[i + 1] in _ESCAPABLE_PUNCTUATION:
            out.append(text[i + 1])
            i += 2
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _split_anchor(target: str) -> str:
    """Return the path portion, splitting only on an unescaped ``#``.

    A ``#`` is escaped when preceded by an odd number of backslashes.
    """
    i = 0
    n = len(target)
    while i < n:
        if target[i] == "#":
            backslashes = 0
            j = i - 1
            while j >= 0 and target[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                return target[:i]
        i += 1
    return target


def iter_markdown_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(path.rglob("*.md"))
        elif path.is_file() and path.suffix == ".md":
            files.add(path)
    return sorted(files)


def find_broken_links(
    md_file: Path,
    repository_root: Path | None = None,
) -> list[tuple[int, str]]:
    """Return ``(line_number, target)`` for broken relative links in a file.

    ``repository_root`` bounds resolution: decoded targets that escape it,
    resolve to a directory, or do not exist are reported as broken. When
    omitted it defaults to the containing directory of ``md_file``.
    """
    if repository_root is None:
        repository_root = md_file.parent
    root = repository_root.resolve()
    broken: list[tuple[int, str]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    try:
        lines = md_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [(0, f"unreadable: {exc}")]

    for lineno, raw in enumerate(lines, start=1):
        fence_match = _FENCE_PATTERN.match(raw)
        if fence_match:
            marker = fence_match.group(1)
            char = marker[0]
            length = len(marker)
            if in_fence and char == fence_char and length >= fence_len:
                in_fence = False
                fence_char = ""
                fence_len = 0
            elif not in_fence:
                in_fence = True
                fence_char = char
                fence_len = length
            continue
        if in_fence:
            continue

        for match in _LINK_PATTERN.finditer(raw):
            if match.group("dest") is not None:
                target = match.group("dest")
            else:
                target = match.group("angle")
            target = target.strip()
            if _is_windows_absolute(target) or _is_url(target):
                continue
            path_part = _split_anchor(target).strip()
            if not path_part or path_part.startswith("/"):
                continue
            decoded = unquote(_unescape(path_part))
            if _is_windows_absolute(decoded):
                continue
            resolved = (md_file.parent / decoded).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                broken.append((lineno, target))
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify relative local Markdown links resolve to files."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Markdown files or directories to scan")
    args = parser.parse_args(argv)

    for path in args.paths:
        if not (path.is_dir() or (path.is_file() and path.suffix == ".md")):
            parser.error(f"not a markdown file or directory: {path}")

    root = Path.cwd().resolve()
    errors: list[str] = []
    for md_file in iter_markdown_files(args.paths):
        for lineno, target in find_broken_links(md_file, root):
            errors.append(f"{md_file}:{lineno}: broken link -> {target}")

    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
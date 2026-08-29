"""Unit tests for scripts/check_markdown_links.py."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_markdown_links.py"
_spec = importlib.util.spec_from_file_location("check_markdown_links", _SCRIPT)
assert _spec and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


class MarkdownLinkCheckerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, content: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_valid_relative_link_passes(self) -> None:
        self._write("docs/target.md", "# Target")
        md = self._write("docs/index.md", "see [target](./target.md)")
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_broken_relative_link_reports_path_and_line(self) -> None:
        md = self._write("docs/index.md", "line one\nsee [missing](./missing.md)")
        self.assertEqual(checker.find_broken_links(md, self.root), [(2, "./missing.md")])

    def test_parent_relative_link_resolves(self) -> None:
        self._write("docs/nested/deep.md", "# Deep")
        md = self._write("docs/nested/index.md", "up [doc](../other.md)")
        self.assertEqual(checker.find_broken_links(md, self.root), [(1, "../other.md")])

    def test_anchor_only_link_ignored(self) -> None:
        md = self._write("docs/index.md", "see [section](#intro)")
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_url_and_mailto_links_ignored(self) -> None:
        md = self._write(
            "docs/index.md",
            "[web](https://example.com/x) [http](http://example.com) [mail](mailto:a@b.c)",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_rfc_scheme_urls_ignored_case_insensitively(self) -> None:
        md = self._write(
            "docs/index.md",
            "[a](FTP://x/y) [b](mailto:z@w.c) [c](tel:+1-2) [d](git+ssh://h/x) [e](SCHEME:y)",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_windows_absolute_paths_ignored(self) -> None:
        md = self._write(
            "docs/index.md",
            r"[a](C:\docs\x.md) [b](C:/docs/y.md)",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_fenced_code_pseudo_links_ignored(self) -> None:
        md = self._write(
            "docs/index.md",
            "```markdown\n[broken](./nope.md)\n```\n"
            "~~~\n[broken2](./nope2.md)\n~~~\n"
            "text before\n```\n[broken3](./nope3.md)\n```\n",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_fence_requires_matching_marker_and_length(self) -> None:
        self._write("docs/target.md", "# T")
        md = self._write(
            "docs/index.md",
            "````\n[broken](./nope.md)\n```\n[real](./target.md)\n",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_fence_different_marker_does_not_close(self) -> None:
        self._write("docs/target.md", "# T")
        md = self._write(
            "docs/index.md",
            "```\n[broken](./nope.md)\n~~~\n[real](./target.md)\n",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_fence_closes_with_longer_marker(self) -> None:
        md = self._write(
            "docs/index.md",
            "```\n[broken](./nope.md)\n````\n[real](./missing2.md)\n",
        )
        self.assertEqual(
            checker.find_broken_links(md, self.root), [(4, "./missing2.md")]
        )

    def test_link_with_anchor_suffix_checks_file_part(self) -> None:
        self._write("docs/target.md", "# T")
        md = self._write("docs/index.md", "see [target](./target.md#section)")
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_angle_bracketed_link_with_spaces(self) -> None:
        self._write("docs/my target.md", "# T")
        md = self._write("docs/index.md", "see [target](<./my target.md>)")
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_links_with_titles_parsed(self) -> None:
        self._write("docs/target.md", "# T")
        md = self._write(
            "docs/index.md",
            'see [a](./target.md "Double") [b](./target.md \'Single\') '
            "[c](./target.md (Paren)) [d](<./target.md> 'T')",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_nested_parentheses_in_path(self) -> None:
        self._write("docs/target (copy).md", "# T")
        self._write("docs/sub(1)/x.md", "# X")
        md = self._write(
            "docs/index.md",
            "see [a](<./target (copy).md>) [b](./sub(1)/x.md)",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_escaped_characters_in_dest(self) -> None:
        self._write("docs/a(b).md", "# T")
        md = self._write("docs/index.md", "see [x](<./a(b).md>) [y](./a\\(b\\).md)")
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_single_backslash_rooted_paths_ignored(self) -> None:
        md = self._write(
            "docs/index.md",
            r"[a](\docs\x.md) [b](\x.md)",
        )
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_unescape_keeps_non_punctuation_backslash(self) -> None:
        self.assertEqual(checker._unescape(r"a\b.md"), r"a\b.md")
        self.assertEqual(checker._unescape(r"a\#b.md"), "a#b.md")

    def test_escaped_hash_filename_resolves(self) -> None:
        self._write("docs/a#b.md", "# T")
        md = self._write("docs/index.md", "see [x](./a\\#b.md)")
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_unescaped_anchor_still_splits(self) -> None:
        self._write("docs/target.md", "# T")
        md = self._write("docs/index.md", "see [x](./target.md#section)")
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_site_root_absolute_link_ignored(self) -> None:
        md = self._write("docs/index.md", "see [asset](/static/x)")
        self.assertEqual(checker.find_broken_links(md, self.root), [])

    def test_directory_target_rejected(self) -> None:
        self._write("docs/sub/index.md", "# Sub")
        md = self._write("docs/index.md", "see [dir](./sub)")
        self.assertEqual(checker.find_broken_links(md, self.root), [(1, "./sub")])

    def test_path_escape_outside_root_rejected(self) -> None:
        md = self._write("docs/index.md", "see [out](../../outside.md)")
        self.assertEqual(
            checker.find_broken_links(md, self.root), [(1, "../../outside.md")]
        )

    def test_default_root_contains_to_file_directory(self) -> None:
        self._write("docs/other.md", "# O")
        md = self._write("docs/nested/index.md", "up [doc](../other.md)")
        self.assertEqual(
            checker.find_broken_links(md), [(1, "../other.md")]
        )

    def test_symlink_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / "secret.md"
            outside.write_text("# secret", encoding="utf-8")
            link = self.root / "docs" / "leak.md"
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable on this platform: {exc}")
            md = self._write("docs/index.md", "see [leak](./leak.md)")
            self.assertEqual(checker.find_broken_links(md, self.root), [(1, "./leak.md")])

    def test_unreadable_file_reported_as_issue(self) -> None:
        md = self._write("docs/index.md", "see [x](./y.md)")
        with unittest.mock.patch.object(
            checker.Path, "read_text", side_effect=PermissionError("denied")
        ):
            issues = checker.find_broken_links(md, self.root)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0][0], 0)
        self.assertIn("unreadable", issues[0][1])

    def test_main_unreadable_returns_nonzero(self) -> None:
        self._write("docs/index.md", "see [x](./y.md)")
        with unittest.mock.patch.object(
            checker.Path, "read_text", side_effect=PermissionError("denied")
        ):
            self.assertEqual(checker.main([str(self.root / "docs")]), 1)

    def test_iter_markdown_files_finds_recursively(self) -> None:
        self._write("docs/a.md", "# A")
        self._write("docs/nested/b.md", "# B")
        self._write("docs/notmd.txt", "plain")
        files = checker.iter_markdown_files([self.root / "docs"])
        self.assertEqual(
            [p.relative_to(self.root).as_posix() for p in files],
            ["docs/a.md", "docs/nested/b.md"],
        )

    def test_main_exit_code(self) -> None:
        self._write("docs/index.md", "see [missing](./missing.md)")
        with unittest.mock.patch.object(checker.Path, "cwd", lambda: checker.Path(self.root)):
            self.assertEqual(checker.main([str(self.root / "docs")]), 1)

    def test_main_clean_exit_code(self) -> None:
        self._write("docs/target.md", "# T")
        self._write("docs/index.md", "see [target](./target.md)")
        with unittest.mock.patch.object(checker.Path, "cwd", lambda: checker.Path(self.root)):
            self.assertEqual(checker.main([str(self.root / "docs")]), 0)


if __name__ == "__main__":
    unittest.main()
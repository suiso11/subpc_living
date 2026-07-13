import json
import tempfile
import time
import unittest
from pathlib import Path

from src.chat import history_admin


def _write_session(directory: Path, name: str, *, saved_at: str, preview: str) -> Path:
    path = directory / name
    path.write_text(json.dumps({
        "session_id": name.replace("session_", "").replace(".json", ""),
        "created_at": saved_at,
        "saved_at": saved_at,
        "turn_count": 2,
        "messages": [
            {"role": "user", "content": preview},
            {"role": "assistant", "content": "了解"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return path


class HistoryAdminTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_sessions_sorted_and_previewed(self) -> None:
        _write_session(self.dir, "session_a.json", saved_at="2026-07-01T10:00:00", preview="古い方")
        _write_session(self.dir, "session_b.json", saved_at="2026-07-04T10:00:00", preview="新しい方")
        sessions = history_admin.list_sessions(self.dir)
        self.assertEqual([s["file"] for s in sessions], ["session_b.json", "session_a.json"])
        self.assertEqual(sessions[0]["preview"], "新しい方")
        self.assertEqual(sessions[0]["turn_count"], 2)

    def test_list_sessions_handles_broken_json(self) -> None:
        (self.dir / "session_broken.json").write_text("{not json", encoding="utf-8")
        sessions = history_admin.list_sessions(self.dir)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["preview"], "(読み込み不可)")

    def test_list_sessions_missing_dir(self) -> None:
        self.assertEqual(history_admin.list_sessions(self.dir / "nope"), [])

    def test_read_session(self) -> None:
        _write_session(self.dir, "session_x.json", saved_at="2026-07-04T10:00:00", preview="こんにちは")
        data = history_admin.read_session(self.dir, "session_x.json")
        self.assertIsNotNone(data)
        self.assertEqual(data["messages"][0]["content"], "こんにちは")

    def test_read_session_rejects_traversal(self) -> None:
        self.assertIsNone(history_admin.read_session(self.dir, "../secrets.json"))
        self.assertIsNone(history_admin.read_session(self.dir, "session_../x.json"))
        self.assertIsNone(history_admin.read_session(self.dir, "other.json"))

    def test_delete_session(self) -> None:
        path = _write_session(self.dir, "session_del.json", saved_at="2026-07-04T10:00:00", preview="x")
        self.assertTrue(history_admin.delete_session(self.dir, "session_del.json"))
        self.assertFalse(path.exists())
        self.assertFalse(history_admin.delete_session(self.dir, "session_del.json"))
        self.assertFalse(history_admin.delete_session(self.dir, "../evil.json"))

    def test_prune_sessions(self) -> None:
        for i in range(5):
            path = _write_session(
                self.dir, f"session_{i}.json",
                saved_at=f"2026-07-0{i + 1}T10:00:00", preview=str(i),
            )
            mtime = time.time() - (5 - i) * 60
            import os
            os.utime(path, (mtime, mtime))
        removed = history_admin.prune_sessions(self.dir, 2)
        self.assertEqual(removed, 3)
        remaining = sorted(p.name for p in self.dir.glob("session_*.json"))
        self.assertEqual(remaining, ["session_3.json", "session_4.json"])

    def test_prune_sessions_zero_is_noop(self) -> None:
        _write_session(self.dir, "session_keep.json", saved_at="2026-07-04T10:00:00", preview="x")
        self.assertEqual(history_admin.prune_sessions(self.dir, 0), 0)
        self.assertTrue((self.dir / "session_keep.json").exists())

    def test_safe_session_id_rejects_traversal_and_long_values(self) -> None:
        self.assertTrue(history_admin.is_safe_session_id("web_123.abc-test"))
        for value in ("", "../secret", "a/b", "a\\b", "x y", "a;b", "a" * 129, None):
            self.assertFalse(history_admin.is_safe_session_id(value))

    def test_read_session_by_id(self) -> None:
        _write_session(
            self.dir, "session_web_123.json",
            saved_at="2026-07-04T10:00:00", preview="続き",
        )
        data = history_admin.read_session_by_id(self.dir, "web_123")
        self.assertEqual(data["session_id"], "web_123")
        self.assertEqual(data["messages"][0]["content"], "続き")
        self.assertIsNone(history_admin.read_session_by_id(self.dir, "../secret"))
        self.assertIsNone(history_admin.read_session_by_id(self.dir, "nope_404"))

    def test_read_session_by_id_rejects_mismatched_payload(self) -> None:
        _write_session(
            self.dir, "session_web_123.json",
            saved_at="2026-07-04T10:00:00", preview="x",
        )
        data = json.loads((self.dir / "session_web_123.json").read_text(encoding="utf-8"))
        data["session_id"] = "other"
        (self.dir / "session_web_123.json").write_text(json.dumps(data), encoding="utf-8")
        self.assertIsNone(history_admin.read_session_by_id(self.dir, "web_123"))

    def test_latest_valid_session_uses_latest_and_skips_broken_file(self) -> None:
        _write_session(
            self.dir, "session_old.json",
            saved_at="2026-07-01T10:00:00", preview="old",
        )
        _write_session(
            self.dir, "session_new.json",
            saved_at="2026-07-04T10:00:00", preview="new",
        )
        (self.dir / "session_broken.json").write_text("{broken", encoding="utf-8")
        latest = history_admin.read_latest_valid_session(self.dir)
        self.assertEqual(latest["session_id"], "new")

    def test_read_latest_valid_session_empty_dir(self) -> None:
        self.assertIsNone(history_admin.read_latest_valid_session(self.dir))
        self.assertIsNone(history_admin.read_latest_valid_session(self.dir / "nope"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class DiscoverVrmModelsTest(unittest.TestCase):
    def test_missing_dir_returns_empty(self) -> None:
        from src.desktop.avatar import discover_vrm_models

        self.assertEqual(discover_vrm_models(Path(tempfile.gettempdir()) / "no-such-vrm-dir"), [])

    def test_sorted_vrm_files_only(self) -> None:
        from src.desktop.avatar import discover_vrm_models

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "b.vrm").write_bytes(b"glTF")
            (d / "a.vrm").write_bytes(b"glTF")
            (d / "not-a-model.txt").write_text("x")
            (d / "sub") .mkdir()
            (d / "sub" / "nested.vrm").write_bytes(b"glTF")
            found = discover_vrm_models(d)
            self.assertEqual([p.name for p in found], ["a.vrm", "b.vrm"])

    def test_sorted_includes_glb(self) -> None:
        from src.desktop.avatar import discover_vrm_models

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "c.vrm").write_bytes(b"glTF")
            (d / "a.glb").write_bytes(b"glTF")
            (d / "b.vrm").write_bytes(b"glTF")
            (d / "d.txt").write_text("x")
            found = discover_vrm_models(d)
            self.assertEqual([p.name for p in found], ["a.glb", "b.vrm", "c.vrm"])

    def test_glb_only(self) -> None:
        from src.desktop.avatar import discover_vrm_models

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "z.glb").write_bytes(b"glTF")
            (d / "a.glb").write_bytes(b"glTF")
            found = discover_vrm_models(d)
            self.assertEqual([p.name for p in found], ["a.glb", "z.glb"])


class IsProbableVrmTest(unittest.TestCase):
    def test_gltf_magic_accepted(self) -> None:
        from src.desktop.avatar import is_probable_vrm

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.vrm"
            p.write_bytes(b"glTF" + b"\x00" * 8)
            self.assertTrue(is_probable_vrm(p))

    def test_wrong_magic_rejected(self) -> None:
        from src.desktop.avatar import is_probable_vrm

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.vrm"
            p.write_bytes(b"PK\x03\x04")
            self.assertFalse(is_probable_vrm(p))

    def test_missing_file_rejected(self) -> None:
        from src.desktop.avatar import is_probable_vrm

        self.assertFalse(is_probable_vrm(Path(tempfile.gettempdir()) / "no-such.vrm"))


class ResolveAvatarModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vrm_dir = self.root / "models" / "vrm"
        self.vrm_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, magic: bytes = b"glTF") -> Path:
        p = self.vrm_dir / name
        p.write_bytes(magic)
        return p

    def test_explicit_takes_precedence(self) -> None:
        from src.desktop.avatar import resolve_avatar_model

        dropped = self._write("a.vrm")
        explicit = self._write("z.vrm")
        resolved = resolve_avatar_model(
            self.root, explicit=explicit, env={"DESKTOP_VRM_MODEL": ""}
        )
        self.assertEqual(resolved, explicit)
        self.assertEqual(dropped.name, "a.vrm")

    def test_explicit_missing_returns_none(self) -> None:
        from src.desktop.avatar import resolve_avatar_model

        self.assertIsNone(
            resolve_avatar_model(self.root, explicit=self.root / "nope.vrm", env={})
        )

    def test_env_var_second(self) -> None:
        from src.desktop.avatar import resolve_avatar_model

        self._write("a.vrm")
        via_env = self._write("env-model.vrm")
        resolved = resolve_avatar_model(
            self.root, env={"DESKTOP_VRM_MODEL": str(via_env)}
        )
        self.assertEqual(resolved, via_env)

    def test_env_missing_falls_through_to_discovery(self) -> None:
        from src.desktop.avatar import resolve_avatar_model

        first = self._write("a.vrm")
        resolved = resolve_avatar_model(
            self.root, env={"DESKTOP_VRM_MODEL": str(self.root / "missing.vrm")}
        )
        self.assertEqual(resolved, first)

    def test_discovery_first_in_sorted_order(self) -> None:
        from src.desktop.avatar import resolve_avatar_model

        self._write("b.vrm")
        expected = self._write("a.vrm")
        resolved = resolve_avatar_model(self.root, env={})
        self.assertEqual(resolved, expected)

    def test_glb_discovered_alongside_vrm(self) -> None:
        from src.desktop.avatar import resolve_avatar_model

        self._write("b.vrm")
        expected = self._write("a.glb")
        resolved = resolve_avatar_model(self.root, env={})
        self.assertEqual(resolved, expected)

    def test_glb_only_still_resolves(self) -> None:
        from src.desktop.avatar import resolve_avatar_model

        expected = self._write("z.glb")
        resolved = resolve_avatar_model(self.root, env={})
        self.assertEqual(resolved, expected)

    def test_empty_dir_returns_none(self) -> None:
        from src.desktop.avatar import resolve_avatar_model

        self.assertIsNone(resolve_avatar_model(self.root, env={}))

    def test_default_dir_uses_repo_layout(self) -> None:
        from src.desktop.avatar import default_vrm_dir

        d = default_vrm_dir()
        self.assertEqual(d.name, "vrm")
        self.assertEqual(d.parent.name, "models")


if __name__ == "__main__":
    unittest.main()

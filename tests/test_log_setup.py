import logging
import tempfile
import unittest
from pathlib import Path

from src.service.log_setup import resolve_level, setup_logging


class LogSetupTest(unittest.TestCase):
    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    def test_resolve_level(self) -> None:
        self.assertEqual(resolve_level("debug"), logging.DEBUG)
        self.assertEqual(resolve_level("WARNING"), logging.WARNING)
        self.assertEqual(resolve_level("unknown"), logging.INFO)

    def test_setup_writes_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = setup_logging("test-svc", log_dir=tmp, level="INFO")
            logger.info("ファイル出力テスト %s", 123)
            for handler in logging.getLogger().handlers:
                handler.flush()
            log_file = Path(tmp) / "test-svc.log"
            self.assertTrue(log_file.exists())
            content = log_file.read_text(encoding="utf-8")
            self.assertIn("ファイル出力テスト 123", content)
            self.assertIn("[test-svc]", content)

    def test_repeated_setup_does_not_duplicate_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            setup_logging("svc", log_dir=tmp)
            setup_logging("svc", log_dir=tmp)
            self.assertEqual(len(logging.getLogger().handlers), 2)


if __name__ == "__main__":
    unittest.main()

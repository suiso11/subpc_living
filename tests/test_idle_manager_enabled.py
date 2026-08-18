import unittest
from unittest.mock import patch

from src.service.idle import create_idle_manager, is_idle_manager_enabled


class IdleManagerEnabledTest(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(is_idle_manager_enabled({}))

    def test_requires_explicit_truthy_value(self):
        for value in ("1", "true", "TRUE", " yes ", "on"):
            with self.subTest(value=value):
                self.assertTrue(is_idle_manager_enabled({"IDLE_MANAGER_ENABLED": value}))

    def test_unknown_and_false_values_stay_disabled(self):
        for value in ("0", "false", "off", "no", "unexpected", ""):
            with self.subTest(value=value):
                self.assertFalse(is_idle_manager_enabled({"IDLE_MANAGER_ENABLED": value}))

    @patch("src.service.idle.IdleManager")
    def test_factory_does_not_construct_manager_by_default(self, manager_class):
        self.assertIsNone(create_idle_manager({}))
        manager_class.assert_not_called()

    @patch("src.service.idle.IdleManager")
    def test_factory_constructs_manager_after_explicit_opt_in(self, manager_class):
        manager = create_idle_manager({"IDLE_MANAGER_ENABLED": "true"})
        self.assertIs(manager, manager_class.return_value)
        manager_class.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

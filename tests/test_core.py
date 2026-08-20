from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.app.core.privacy import Pseudonymizer, coarse_ip_network
from backend.app.core.settings import Settings


class PrivacyTests(unittest.TestCase):
    def test_pseudonyms_are_stable_and_namespace_separated(self) -> None:
        service = Pseudonymizer("a sufficiently long local secret")
        first = service.pseudonymize("user", "person-1")
        self.assertEqual(first, service.pseudonymize("user", "person-1"))
        self.assertNotEqual(first, service.pseudonymize("device", "person-1"))
        self.assertNotIn("person-1", first)

    def test_coarse_ip_network(self) -> None:
        self.assertEqual(coarse_ip_network("192.168.10.42"), "192.168.10.0/24")
        self.assertEqual(coarse_ip_network("2001:db8:1:2::5"), "2001:db8:1::/48")


class SettingsTests(unittest.TestCase):
    def test_non_local_mode_rejects_default_secret(self) -> None:
        with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
            with self.assertRaisesRegex(ValueError, "PSEUDONYM_KEY"):
                Settings.from_env()

    def test_explicit_test_settings_validate(self) -> None:
        environment = {
            "APP_ENV": "test",
            "PSEUDONYM_KEY": "x" * 32,
            "CORS_ORIGINS": "http://one.test,http://two.test",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.cors_origins, ("http://one.test", "http://two.test"))


if __name__ == "__main__":
    unittest.main()

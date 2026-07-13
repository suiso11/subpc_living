from __future__ import annotations

import asyncio
import unittest

from src.web import server


class WebAchievementsPageTest(unittest.TestCase):
    def test_achievements_route_returns_the_page(self) -> None:
        response = asyncio.run(server.achievements_page())
        self.assertEqual(response.status_code, 200)
        body = response.body.decode("utf-8")
        self.assertIn("<title>実績 | SUBPC BUDDY</title>", body)
        self.assertIn('src="/static/achievements.js"', body)


if __name__ == "__main__":
    unittest.main()

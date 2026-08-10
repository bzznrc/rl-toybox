"""Small shared-contract smoke test."""

from __future__ import annotations

import unittest

from core.arcade_style import ACCENT_PAIRS, NEUTRAL_COLORS


class SmokeTests(unittest.TestCase):
    def test_palette_has_twelve_unique_colors(self) -> None:
        colors = (*NEUTRAL_COLORS, *(color for pair in ACCENT_PAIRS for color in pair))
        self.assertEqual(len(NEUTRAL_COLORS), 4)
        self.assertEqual(len(ACCENT_PAIRS), 4)
        self.assertEqual(len(colors), 12)
        self.assertEqual(len(set(colors)), 12)


if __name__ == "__main__":
    unittest.main()

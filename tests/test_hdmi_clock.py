"""Regress the board's HWH / live-PLL clock mismatch before DMA starts."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ps'))
from phase3_hdmi_overlay import ensure_hdmi_reference_clock


class ClockRegisters:
    def __init__(self, initial, reject=False):
        self.actual = initial
        self.writes = []
        self.reject = reject

    @property
    def fclk0_mhz(self):
        return self.actual

    @fclk0_mhz.setter
    def fclk0_mhz(self, target):
        self.writes.append(target)
        if not self.reject:
            self.actual = target


class HdmiClockTests(unittest.TestCase):
    @patch('phase3_hdmi_overlay.time.sleep')
    def test_live_pll_mismatch_is_corrected(self, sleep):
        clocks = ClockRegisters(62.5)
        self.assertEqual(ensure_hdmi_reference_clock(clocks), 100.0)
        self.assertEqual(clocks.writes, [100.0])
        sleep.assert_called_once()
        # Same 0.252 ratio is used by the stock-like MMCM and pixel divider.
        self.assertAlmostEqual(clocks.actual * .252, 25.2)

    @patch('phase3_hdmi_overlay.time.sleep')
    def test_correct_clock_is_not_reprogrammed(self, sleep):
        clocks = ClockRegisters(100.0)
        ensure_hdmi_reference_clock(clocks)
        self.assertEqual(clocks.writes, [])
        sleep.assert_not_called()

    @patch('phase3_hdmi_overlay.time.sleep')
    def test_failed_reconfiguration_aborts(self, sleep):
        for actual in (62.5, float('nan')):
            with self.subTest(actual=actual), self.assertRaisesRegex(RuntimeError, 'FCLK0=100'):
                ensure_hdmi_reference_clock(ClockRegisters(actual, reject=True))


if __name__ == '__main__':
    unittest.main()

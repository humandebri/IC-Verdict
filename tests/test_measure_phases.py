"""Regression checks for the instruction-cost model in tools/measure_phases.py.

The efficiency reported in docs/PERFORMANCE_MEASUREMENTS.md was wrong three times, and
all three errors pushed the same way: the MAC denominator was too large, so the kernel
looked more efficient than it is. These tests pin the model itself, because a future
edit that restores any of the three should fail here rather than in a document.
"""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from measure_phases import UPDATE_INSTRUCTION_LIMIT, encoder_macs  # noqa: E402

# The two tiers quoted in the document, at the 128-token profile.
MEASURE_M = {"hidden_size": 512, "intermediate_size": 2048, "layers": 4, "local_attention": 128}
MEASURE_6L768 = {"hidden_size": 768, "intermediate_size": 1968, "layers": 6, "local_attention": 128}


class EncoderMacs(unittest.TestCase):
    def test_mlp_counts_three_projections(self):
        # wi is [2*inter, h] (gate and up) and wo is [h, inter], so one token costs
        # 3*inter*h, not 4*inter*h. Charging wi's width for wo too inflated the
        # denominator by 4/3 and understated instructions/MAC by the same factor.
        h, inter = 512, 2048
        config = {"hidden_size": h, "intermediate_size": inter, "layers": 1, "local_attention": 128}
        self.assertEqual(encoder_macs(config, 1), 4 * h * h + 3 * inter * h + 2 * h)

    def test_documented_totals(self):
        self.assertEqual(encoder_macs(MEASURE_M, 128), 2_214_592_512)
        self.assertEqual(encoder_macs(MEASURE_6L768, 128), 5_445_255_168)

    def test_attention_window_is_bounded_by_tokens(self):
        # Below the sliding window attention grows quadratically, so 4x the tokens must
        # cost more than 4x but less than 16x; a linear model would hide a token-length
        # assumption, which is exactly the error the tool used to make.
        short, long = encoder_macs(MEASURE_M, 32), encoder_macs(MEASURE_M, 128)
        self.assertGreater(long, 4 * short)
        self.assertLess(long, 16 * short)

    def test_documented_efficiency(self):
        # 10,946,018,414 instructions for one measure-m question at 128 tokens.
        self.assertAlmostEqual(10_946_018_414 / encoder_macs(MEASURE_M, 128), 4.94, places=2)

    def test_update_budget_is_the_ic_ceiling(self):
        self.assertEqual(UPDATE_INSTRUCTION_LIMIT, 40_000_000_000)

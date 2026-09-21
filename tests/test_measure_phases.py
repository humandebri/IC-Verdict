"""Regression checks for the instruction-cost model in tools/measure_phases.py.

The efficiency reported in docs/PERFORMANCE_MEASUREMENTS.md was wrong three times, and
all three errors pushed the same way: the MAC denominator was too large, so the kernel
looked more efficient than it is. These tests pin the model itself, because a future
edit that restores any of the three should fail here rather than in a document.

The tier configs carry `global_every` because only `i % global_every != 0` layers are
windowed: a model that charges the sliding window to every layer is wrong above the
window size and right below it, which is exactly the kind of error that survives a
short-input check.
"""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from measure_phases import UPDATE_INSTRUCTION_LIMIT, encoder_macs  # noqa: E402

# The two tiers quoted in the document, at the 128-token profile. `global_every` is read
# off fixtures/measure-{m,6l768}/manifest.json (4 and 3 respectively).
MEASURE_M = {"hidden_size": 512, "intermediate_size": 2048, "layers": 4,
             "local_attention": 128, "global_every": 4}
MEASURE_6L768 = {"hidden_size": 768, "intermediate_size": 1968, "layers": 6,
                 "local_attention": 128, "global_every": 3}


def flat_window_macs(config: dict, tokens: int) -> int:
    """The previous model: the window charged to every layer, every query."""
    h, inter = config["hidden_size"], config["intermediate_size"]
    window = min(tokens, config["local_attention"])
    return (4 * h * h + 3 * inter * h + 2 * window * h) * tokens * config["layers"]


class EncoderMacs(unittest.TestCase):
    def test_mlp_counts_three_projections(self):
        # wi is [2*inter, h] (gate and up) and wo is [h, inter], so one token costs
        # 3*inter*h, not 4*inter*h. Charging wi's width for wo too inflated the
        # denominator by 4/3 and understated instructions/MAC by the same factor.
        h, inter = 512, 2048
        config = {"hidden_size": h, "intermediate_size": inter, "layers": 1,
                  "local_attention": 128, "global_every": 1}
        self.assertEqual(encoder_macs(config, 1), 4 * h * h + 3 * inter * h + 2 * h)

    def test_only_non_global_layers_are_windowed(self):
        # Layer 0 is global (`0 % global_every == 0`), layer 1 is windowed. At 256 tokens
        # the global layer sees all 256 keys while the flat model charged it the window,
        # so the flat denominator is too *small* here -- the opposite of the 128-token
        # case, where it is too large because windowed queries lose keys at the edges.
        # At 64 tokens -- fewer than the 129 keys a windowed query could see -- the two
        # models must agree exactly.
        h, inter = 512, 2048
        config = {"hidden_size": h, "intermediate_size": inter, "layers": 2,
                  "local_attention": 128, "global_every": 2}
        self.assertGreater(encoder_macs(config, 256), flat_window_macs(config, 256))
        self.assertEqual(encoder_macs(config, 64), flat_window_macs(config, 64))

    def test_windowed_layer_counts_edge_queries(self):
        # The mask is `abs(q - k) > distance -> masked`, so a query sees 2*distance+1 keys
        # in the middle and fewer at the edges. Using `local_attention` as the key count
        # (or assuming every query sees the full window) is what the third error was.
        h, inter, tokens, distance = 8, 16, 20, 4
        config = {"hidden_size": h, "intermediate_size": inter, "layers": 2,
                  "local_attention": 2 * distance, "global_every": 2}
        dense = (4 * h * h + 3 * inter * h) * tokens * 2
        attention = 2 * h * tokens * tokens          # layer 0 is global
        attention += sum(2 * h * (min(tokens, q + distance + 1) - max(0, q - distance))
                         for q in range(tokens))
        self.assertEqual(encoder_macs(config, tokens), dense + attention)

    def test_documented_totals(self):
        self.assertEqual(encoder_macs(MEASURE_M, 128), 2_202_206_208)
        self.assertEqual(encoder_macs(MEASURE_6L768, 128), 5_420_482_560)

    def test_attention_window_is_bounded_by_tokens(self):
        # Below the sliding window attention grows quadratically, so 4x the tokens must
        # cost more than 4x but less than 16x; a linear model would hide a token-length
        # assumption, which is exactly the error the tool used to make.
        short, long = encoder_macs(MEASURE_M, 32), encoder_macs(MEASURE_M, 128)
        self.assertGreater(long, 4 * short)
        self.assertLess(long, 16 * short)

    def test_documented_efficiency(self):
        # 10,946,018,414 and 26,585,076,637 instructions for one question at 128 tokens.
        self.assertAlmostEqual(10_946_018_414 / encoder_macs(MEASURE_M, 128), 4.97, places=2)
        self.assertAlmostEqual(26_585_076_637 / encoder_macs(MEASURE_6L768, 128), 4.90, places=2)

    def test_update_budget_is_the_ic_ceiling(self):
        self.assertEqual(UPDATE_INSTRUCTION_LIMIT, 40_000_000_000)

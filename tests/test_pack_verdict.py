"""Contract tests for `tools/pack_verdict.py` (review findings 18, 19, 20).

Three ways the exporter used to produce a pack that is wrong rather than failing:

  * the tensor byte length came only from the safetensors header, so a header whose
    `data_offsets` disagree with the declared shape exported "successfully" and was
    rejected by the Rust loader after a 600 MiB upload;
  * `--random` without `--test` wrote `test_only: false`, which the loader maps to
    `BackendKind::Checkpoint` -- random weights attributed to a real commit;
  * `projector_hidden_act` (and `normalize_features`, and a `layer_types` list that
    disagrees with `global_attn_every_n_layers`) fell back silently to a different
    computation.

The exporter is driven as a subprocess, the way the tools call it.
"""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "tools" / "pack_verdict.py"

# Small but structurally faithful: two layers, layer 0 without attn_norm.
CONFIG = {
    "encoder_config": {
        "vocab_size": 32, "hidden_size": 8, "num_hidden_layers": 2,
        "num_attention_heads": 2, "intermediate_size": 12, "norm_eps": 1e-5,
        "cls_token_id": 1, "sep_token_id": 2,
        "global_attn_every_n_layers": 2, "local_attention": 4,
        "rope_parameters": {"full_attention": {"rope_theta": 160000.0},
                            "sliding_attention": {"rope_theta": 10000.0}},
    },
    "class_token_index": 3,
    "max_num_classes": 4,
    "projector_hidden_act": "gelu",
    "normalize_features": False,
}


def checkpoint_name(canonical: str) -> str:
    """Canonical pack name -> the upstream name `canonical_name` maps back from."""
    if canonical == "embeddings.weight":
        return "model.encoder_model.embeddings.tok_embeddings.weight"
    if canonical in ("embeddings.norm.weight", "final_norm.weight"):
        return "model.encoder_model." + canonical
    if canonical.startswith("encoder."):
        _, index, tail = canonical.split(".", 2)
        upstream = {"qkv.weight": "attn.Wqkv.weight", "out.weight": "attn.Wo.weight",
                    "attn_norm.weight": "attn_norm.weight", "wi.weight": "mlp.Wi.weight",
                    "wo.weight": "mlp.Wo.weight", "mlp_norm.weight": "mlp_norm.weight"}[tail]
        return f"model.encoder_model.layers.{index}.{upstream}"
    if canonical.startswith(("text_projector", "classes_projector")):
        return "model." + canonical
    raise AssertionError(f"unmapped canonical name {canonical}")


def inventory() -> dict[str, list[int]]:
    """Canonical name -> shape, mirroring `verdict_candle::expected_tensors`."""
    h, inter, vocab = 8, 12, 32
    shapes = {"embeddings.weight": [vocab, h], "embeddings.norm.weight": [h],
              "final_norm.weight": [h]}
    for i in range(2):
        p = f"encoder.{i}"
        if i > 0:
            shapes[f"{p}.attn_norm.weight"] = [h]
        shapes[f"{p}.qkv.weight"] = [3 * h, h]
        shapes[f"{p}.out.weight"] = [h, h]
        shapes[f"{p}.mlp_norm.weight"] = [h]
        shapes[f"{p}.wi.weight"] = [2 * inter, h]
        shapes[f"{p}.wo.weight"] = [h, inter]
    for p in ("text_projector", "classes_projector"):
        shapes[f"{p}.linear_1.weight"] = [h, h]
        shapes[f"{p}.linear_1.bias"] = [h]
        shapes[f"{p}.linear_2.weight"] = [h, h]
        shapes[f"{p}.linear_2.bias"] = [h]
    return shapes


def safetensors(path: Path, shorten: str | None = None) -> None:
    """Write a minimal F32 safetensors file; `shorten` truncates one tensor's span."""
    header: dict[str, dict] = {}
    offset = 0
    for canonical, shape in sorted(inventory().items()):
        length = 4
        for dim in shape:
            length *= dim
        if canonical == shorten:
            length //= 2
        header[checkpoint_name(canonical)] = {"dtype": "F32", "shape": shape,
                                              "data_offsets": [offset, offset + length]}
        offset += length
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


class PackVerdictTests(unittest.TestCase):
    def _run(self, config: dict, *extra: str, verify=None) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "config.json").write_text(json.dumps(config))
            (tmp / "tokenizer.json").write_text("{}")
            safetensors(tmp / "model.safetensors")
            result = subprocess.run(
                [sys.executable, str(PACK), "--safetensors", str(tmp / "model.safetensors"),
                 "--config", str(tmp / "config.json"), "--tokenizer", str(tmp / "tokenizer.json"),
                 "--out", str(tmp / "pack"), "--source-repo", "local-test",
                 "--source-revision", "a" * 40, *extra],
                capture_output=True, text=True, cwd=ROOT)
            if verify is not None and result.returncode == 0:
                verify(tmp / "pack")
            return result

    def test_valid_checkpoint_exports_a_self_consistent_pack(self) -> None:
        # The pack is checked by reading it back: the manifest's offsets, lengths and
        # checksums must describe the bytes in model.bin. Asserting only the exit code
        # would not notice a manifest that disagrees with its own blob, which is the
        # failure mode that used to reach the 600 MiB upload.
        def verify(pack: Path) -> None:
            manifest = json.loads((pack / "manifest.json").read_text())
            blob = (pack / "model.bin").read_bytes()
            self.assertEqual(manifest["format"], "ic-verdict-int8-pack-v1")
            self.assertEqual({t["name"] for t in manifest["tensors"]}, set(inventory()))
            self.assertEqual(manifest["total_bytes"], len(blob))
            offset = 0
            for tensor in manifest["tensors"]:
                self.assertEqual(tensor["offset"], offset, tensor["name"])
                shape = tensor["shape"]
                elements = 1
                for dim in shape:
                    elements *= dim
                if len(shape) == 2:
                    rows, cols = shape
                    length = elements + 4 * rows * ((cols + 31) // 32)
                    self.assertEqual(tensor["encoding"], "i8_block32_symmetric")
                else:
                    length = 4 * elements
                    self.assertEqual(tensor["encoding"], "f32_le")
                self.assertEqual(tensor["length"], length, tensor["name"])
                chunk = blob[offset:offset + tensor["length"]]
                self.assertEqual(list(hashlib.sha256(chunk).digest()), tensor["sha256"], tensor["name"])
                offset += tensor["length"]
            self.assertEqual(offset, len(blob))

        result = self._run(CONFIG, verify=verify)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("wrote 22 tensors", result.stdout)

    def test_header_length_disagreeing_with_shape_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "config.json").write_text(json.dumps(CONFIG))
            (tmp / "tokenizer.json").write_text("{}")
            safetensors(tmp / "model.safetensors", shorten="encoder.1.wo.weight")
            result = subprocess.run(
                [sys.executable, str(PACK), "--safetensors", str(tmp / "model.safetensors"),
                 "--config", str(tmp / "config.json"), "--tokenizer", str(tmp / "tokenizer.json"),
                 "--out", str(tmp / "pack"), "--source-repo", "local-test",
                 "--source-revision", "a" * 40],
                capture_output=True, text=True, cwd=ROOT)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("invalid shape, data type, or offset", result.stderr)

    def test_random_without_test_is_refused(self) -> None:
        result = self._run(CONFIG, "--random")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("--test", result.stderr)

    def test_unknown_projector_activation_is_refused(self) -> None:
        config = json.loads(json.dumps(CONFIG))
        config["projector_hidden_act"] = "gelu_new"
        result = self._run(config)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("gelu_new", result.stderr)

    def test_normalize_features_is_refused(self) -> None:
        config = json.loads(json.dumps(CONFIG))
        config["normalize_features"] = True
        result = self._run(config)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("normalize_features", result.stderr)

    def test_layer_types_disagreeing_with_the_rule_is_refused(self) -> None:
        config = json.loads(json.dumps(CONFIG))
        config["encoder_config"]["layer_types"] = ["sliding_attention", "sliding_attention"]
        result = self._run(config)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("layer_types", result.stderr)

    def test_layer_types_matching_the_rule_is_accepted(self) -> None:
        config = json.loads(json.dumps(CONFIG))
        config["encoder_config"]["layer_types"] = ["full_attention", "sliding_attention"]
        result = self._run(config)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

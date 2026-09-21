#!/usr/bin/env python3
"""Convert the openJev (GLiClass) safetensors checkpoint into a canonical F32 pack.

The pack is what both the native parity tool and the `verdict-engine` canister
load: `manifest.json` plus one contiguous `model.bin`. Nothing is quantised and
nothing is reordered silently — a tensor that is missing, unexpected or the wrong
shape aborts the conversion.

    python3 tools/pack_verdict.py \
        --safetensors models/verdict-151m/model.safetensors \
        --config models/verdict-151m/config.json \
        --tokenizer models/verdict-151m/tokenizer.json \
        --out models/verdict-pack \
        --source-repo heman10x/rlcd-modernbert-151m \
        --source-revision 70fa19828074e1199e4a793c4af4dba3bfd1d222

`--test` marks the pack as a synthetic fixture: `test_only=true`, a placeholder
source revision, and random weights read from `--random` instead of a checkpoint.
The canister accepts a test pack for smoke runs and reports its kind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

# Tensor names in the checkpoint and the canonical names the Rust loader expects.
# The encoder half is shared with the Laya backend by design.
LAYER_TENSORS = {
    "attn.Wqkv.weight": "qkv.weight",
    "attn.Wo.weight": "out.weight",
    "attn_norm.weight": "attn_norm.weight",
    "mlp.Wi.weight": "wi.weight",
    "mlp.Wo.weight": "wo.weight",
    "mlp_norm.weight": "mlp_norm.weight",
}
PROJECTORS = ("text_projector", "classes_projector")
PROJECTOR_TENSORS = ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")
# Present in the checkpoint, absent from the forward pass: `normalize_features`
# is false, so `logit_scale` is never applied (see docs/GLICLASS_FORWARD_SPEC.md).
IGNORED = {"model.logit_scale"}

FMT = "ic-verdict-f32-pack-v1"


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def verdict_config(raw: dict) -> dict:
    """Translate the checkpoint config into the fields the Rust loader reads."""
    enc = raw["encoder_config"]
    rope = enc.get("rope_parameters") or {}
    full = rope.get("full_attention") or {}
    sliding = rope.get("sliding_attention") or {}
    global_theta = full.get("rope_theta", enc.get("global_rope_theta", 160000.0))
    local_theta = sliding.get("rope_theta", enc.get("local_rope_theta", 10000.0))
    return {
        "vocab_size": enc["vocab_size"],
        "hidden_size": enc["hidden_size"],
        "layers": enc["num_hidden_layers"],
        "attention_heads": enc["num_attention_heads"],
        "intermediate_size": enc["intermediate_size"],
        "norm_eps": enc.get("norm_eps", enc.get("layer_norm_eps", 1e-5)),
        "global_every": enc.get("global_attn_every_n_layers", 3),
        "local_attention": enc.get("local_attention", 128),
        "global_rope_theta": global_theta,
        "local_rope_theta": local_theta,
        # HF makes layer 0's attention norm an Identity, so its weight is absent.
        # The real value is read off the tensor inventory below.
        "first_layer_attention_norm": False,
        "cls_token_id": enc.get("cls_token_id", 50281),
        "sep_token_id": enc.get("sep_token_id", 50282),
        "class_token_id": raw["class_token_index"],
        "max_classes": raw.get("max_num_classes", 25),
        # Explicit, not a fallback: `gelu_new`/`silu`/`quick_gelu` used to silently become
        # Relu, which changes the logits without changing the file. `main` validates the
        # key first, so a KeyError here means the two checks drifted apart.
        "projector_activation": {"gelu": "Gelu", "relu": "Relu"}[raw.get("projector_hidden_act", "gelu")],
    }


def expected_tensors(cfg: dict) -> dict[str, list[int]]:
    """Mirror of `verdict_candle::expected_tensors`."""
    h = cfg["hidden_size"]
    out: dict[str, list[int]] = {
        "embeddings.weight": [cfg["vocab_size"], h],
        "embeddings.norm.weight": [h],
        "final_norm.weight": [h],
    }
    for i in range(cfg["layers"]):
        p = f"encoder.{i}"
        if i > 0 or cfg["first_layer_attention_norm"]:
            out[f"{p}.attn_norm.weight"] = [h]
        out[f"{p}.qkv.weight"] = [3 * h, h]
        out[f"{p}.out.weight"] = [h, h]
        out[f"{p}.mlp_norm.weight"] = [h]
        out[f"{p}.wi.weight"] = [2 * cfg["intermediate_size"], h]
        out[f"{p}.wo.weight"] = [h, cfg["intermediate_size"]]
    for p in PROJECTORS:
        for t in PROJECTOR_TENSORS:
            shape = [h, h] if t.endswith("weight") else [h]
            out[f"{p}.{t}"] = shape
    return out


def canonical_name(name: str) -> str | None:
    """Checkpoint tensor name -> canonical pack name (None when not used)."""
    if name in IGNORED:
        return None
    prefix = "model.encoder_model."
    if name.startswith(prefix):
        rest = name[len(prefix) :]
        if rest == "embeddings.tok_embeddings.weight":
            return "embeddings.weight"
        if rest == "embeddings.norm.weight":
            return "embeddings.norm.weight"
        if rest == "final_norm.weight":
            return "final_norm.weight"
        if rest.startswith("layers."):
            _, index, tail = rest.split(".", 2)
            if tail not in LAYER_TENSORS:
                return None
            return f"encoder.{index}.{LAYER_TENSORS[tail]}"
        return None
    if name.startswith("model.") and name.split(".")[1] in PROJECTORS:
        return name[len("model.") :]
    return None


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length <= 0 or length > 64 * 1024 * 1024:
            raise SystemExit(f"implausible safetensors header length {length}")
        header = json.loads(handle.read(length))
    if "__metadata__" in header:
        print("note: checkpoint carries __metadata__; ignored", file=sys.stderr)
    return header, 8 + length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--safetensors", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-repo", default="heman10x/rlcd-modernbert-151m")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--test", action="store_true", help="mark the pack as a synthetic fixture")
    parser.add_argument("--random", action="store_true", help="write deterministic random weights instead")
    args = parser.parse_args()

    if args.random and not args.test:
        # `--random` writes unverified weights and never opens --safetensors. Combined
        # with a 40-hex --source-revision it produced test_only=false, which the Rust
        # loader maps to BackendKind::Checkpoint, attributing random weights to a real
        # commit in the decision provenance.
        raise SystemExit("--random writes unverified weights; combine it with --test so the "
                         "pack cannot be labelled a real checkpoint")

    raw_config = json.loads(args.config.read_text())
    # These flags change the arithmetic, not just the metadata, and the forward implements
    # exactly one convention. Refusing is the only honest option: the alternative is a
    # silent fallback that loads and scores wrongly.
    activation = raw_config.get("projector_hidden_act", "gelu")
    if activation not in ("gelu", "relu"):
        raise SystemExit(f"projector_hidden_act {activation!r} is not supported: the port "
                         f"implements gelu (erf) and relu only")
    if raw_config.get("normalize_features", False):
        raise SystemExit("normalize_features=true is not supported: this port does not apply "
                         "model.logit_scale, so the pack would be scored unscaled")
    encoder = raw_config.get("encoder_config", {})
    layer_types = encoder.get("layer_types")
    if layer_types is not None:
        every = encoder.get("global_attn_every_n_layers", 3)
        derived = ["full_attention" if i % every == 0 else "sliding_attention"
                   for i in range(len(layer_types))]
        if list(layer_types) != derived:
            raise SystemExit("encoder_config.layer_types disagrees with "
                             "i%global_attn_every_n_layers; the loader derives global/local from "
                             "the rule, so this pack would use the wrong window and RoPE theta")
    cfg = verdict_config(raw_config)
    found: dict[str, tuple[list[int], str, int, int]] = {}
    if args.random:
        # Fixture mode: no checkpoint, no inventory. Layer 0's norm is decided by
        # the config, as it must be for a pack that has no checkpoint to read.
        cfg["first_layer_attention_norm"] = bool(cfg["first_layer_attention_norm"])
        expected = expected_tensors(cfg)
        total_params = 0
        for shape in expected.values():
            count = 1
            for dim in shape:
                count *= dim
            total_params += count
        if total_params > 5_000_000:
            raise SystemExit(f"--random refuses {total_params} params; use --safetensors for a real pack")
        found = {name: (shape, name, 0, 4 * math.prod(shape)) for name, shape in expected.items()}
    else:
        header, data_start = read_safetensors_header(args.safetensors)
        cfg["first_layer_attention_norm"] = "model.encoder_model.layers.0.attn_norm.weight" in header
        expected = expected_tensors(cfg)
        # Two passes so nothing is written until the whole mapping has been checked.
        unmapped: list[str] = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            canonical = canonical_name(name)
            if canonical is None:
                if name in IGNORED:
                    continue
                unmapped.append(name)
                continue
            if info["dtype"] != "F32":
                raise SystemExit(f"{name}: dtype {info['dtype']} is not F32")
            shape = list(info["shape"])
            begin, end = info["data_offsets"]
            found[canonical] = (shape, name, data_start + begin, end - begin)

        if unmapped:
            raise SystemExit(f"unmapped checkpoint tensors: {sorted(unmapped)[:5]} (+{max(0, len(unmapped) - 5)} more)")
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        if missing or extra:
            raise SystemExit(f"tensor set mismatch: missing={missing[:5]} extra={extra[:5]}")
        for canonical, (shape, name, _, length) in found.items():
            if shape != expected[canonical]:
                raise SystemExit(f"{name}: shape {shape} != expected {expected[canonical]}")
            # `length` comes only from the safetensors header's data_offsets. The Rust
            # loader recomputes 4*numel(shape) and rejects a mismatch, so without this
            # check the export "succeeded", the 600 MiB pack was uploaded, and the failure
            # surfaced at canister load instead of here.
            want = 4 * math.prod(shape)
            if length != want:
                raise SystemExit(f"{name}: header declares {length} bytes for shape {shape} "
                                 f"(expected {want})")

    args.out.mkdir(parents=True, exist_ok=True)
    order = sorted(expected)
    tensors = []
    total = 0
    source = None
    with (args.out / "model.bin").open("wb") as out:
        source = None if args.random else args.safetensors.open("rb")
        for canonical in order:
            shape, _, offset, length = found[canonical]
            if args.random:
                # Deterministic low-magnitude weights, same generator as the tests.
                state = 0x5EED
                count = 1
                for dim in shape:
                    count *= dim
                chunk = bytearray()
                for _ in range(count):
                    state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
                    value = (((state >> 33) & 0xFFFFFFFF) / 0x7FFFFFFF) - 1.0
                    chunk += struct.pack("<f", value * 0.4)
                payload = bytes(chunk)
            else:
                source.seek(offset)
                payload = source.read(length)
            if len(payload) != length:
                raise SystemExit(f"{canonical}: short read ({len(payload)} != {length})")
            out.write(payload)
            tensors.append(
                {
                    "name": canonical,
                    "shape": shape,
                    "offset": total,
                    "length": length,
                    "sha256": list(sha256(payload)),
                }
            )
            total += length
    if source is not None:
        source.close()

    revision = args.source_revision or ("0" * 40 if args.test else "")
    if not args.test and (len(revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in revision)):
        raise SystemExit("a real pack requires --source-revision as a 40-hex commit id")
    manifest = {
        "format": FMT,
        "source_repo": args.source_repo,
        "source_revision": revision,
        "test_only": bool(args.test),
        "tokenizer_sha256": list(sha256(args.tokenizer.read_bytes())),
        "config": cfg,
        "total_bytes": total,
        "tensors": tensors,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"wrote {args.out}/manifest.json and model.bin")
    print(f"  tensors={len(tensors)} total_bytes={total} ({total / 1048576:.1f} MiB) test_only={bool(args.test)}")
    print(f"  hidden={cfg['hidden_size']} layers={cfg['layers']} classes={cfg['max_classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

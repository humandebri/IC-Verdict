#!/usr/bin/env python3
"""Bridge IC-Laya's canonical F32 pack to the real upstream Laya checkpoint.

Why this exists
---------------
`tools/pack_checkpoint.py` deliberately refuses to guess a mapping: it requires a
caller-supplied, reviewed mapping and config. This tool produces the *draft* of
those two files from facts read out of the actual checkpoint, and reports every
place where the real checkpoint differs from what the Rust loader assumes.

What it does NOT do
-------------------
It does not verify numerical parity. Structural reconciliation (names, shapes,
dtype, token ids) is necessary but not sufficient: QKV ordering, RoPE layout,
the pre-norm/post-norm convention, and attention-window semantics must still be
confirmed against the source implementation by comparing logits. `discover`
prints those open questions instead of asserting they are solved.

Usage
-----
    # Facts only. Downloads the safetensors *header* (~21 KiB), never the weights.
    python3 tools/laya_port_bridge.py discover --out checkpoints/laya-port

    # Additionally export the pack, if the weights are already on disk.
    python3 tools/laya_port_bridge.py export --out checkpoints/laya-port \
        --weights /path/to/model.safetensors

Facts encoded below were read from
`convaiinnovations/laya-typed-decisions` @ `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`
(encoder/config.json, rl_agent_config.json, model.safetensors header,
tokenizer/tokenizer.json). Re-verify them if that revision changes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "convaiinnovations/laya-typed-decisions"
# Immutable revision the facts below were read from.
REVISION = "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"

# Upstream special token ids, read from tokenizer/tokenizer.json `added_tokens`.
# NOTE: upstream PAD is 50283 and MASK is 50284. The bundled synthetic fixture
# uses 50283 for MASK; that difference is a real porting hazard, not a typo.
UPSTREAM_SPECIAL = {"cls": 50281, "sep": 50282, "pad": 50283, "mask": 50284}
UPSTREAM_SPECIAL_LITERALS = ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]

# Which upstream attention layers are global vs sliding is decided by
# encoder/config.json `layer_types`, not by the canonical `i % global_every`
# rule. They agree for this revision, but the rule is not the source of truth.
ASSUMED_DELTAS = [
    (
        "tensor namespace",
        "embeddings.weight / encoder.{i}.qkv.weight / decision.{i}.qkv.weight",
        "encoder.embeddings.tok_embeddings.weight / encoder.layers.{i}.attn.Wqkv.weight / "
        "head.layers.{i}.self_attn.in_proj_weight",
        "Every canonical tensor name but one differs. pack_checkpoint.py would have "
        "rejected an assumed mapping.",
    ),
    (
        "qtype embedding",
        "qtype.weight",
        "type_emb.weight",
        "Same role ([3, hidden] primitive embedding), different name.",
    ),
    (
        "dtype",
        "F32 required by the loader",
        "F16 on disk",
        "The exporter must cast to F32; the loader rejects non-F32 tensors.",
    ),
    (
        "scorer",
        "norm/dense/out",
        "scorer.0 (norm) / scorer.1 (dense) / scorer.3 (out)",
        "Dropout modules occupy indices 2 and 4, so `scorer.2` is absent. "
        "An index-contiguous guess breaks here.",
    ),
    (
        "act head",
        "not modelled",
        "act_head.0/2 present",
        "Escalation head exists upstream. The design says to keep it and skip it "
        "only as an output branch; the loader does not read it yet.",
    ),
    (
        "mask token id",
        "fixture supplies 50283",
        "50284 ([PAD] is 50283)",
        "A fixture-derived mask id would mark the wrong token and produce "
        "plausible-looking but wrong logits.",
    ),
    (
        "calibration temperature",
        "single scalar `temperature`",
        "temperature[3] by primitive + `temperature_by_options` by option count",
        "Candid `Calibration.temperature` is a scalar. Per-primitive and "
        "per-option-count temperatures cannot be represented without a schema change.",
    ),
    (
        "embedding/index width",
        "hidden_size x vocab_size",
        "50368 x 1024 F16",
        "Fine as a shape, but see the pack-size finding below.",
    ),
    (
        "attention normalization",
        "encoder.{i}.attn_norm for i>0",
        "attn_norm absent on layer 0 only",
        "Matches `first_layer_attention_norm=false`. Consistent, not a defect.",
    ),
]


def fetch(url: str, limit: int | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ic-laya-port-bridge"})
    if limit is not None:
        request.add_header("Range", f"bytes=0-{limit - 1}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def safetensors_header(source: str | Path) -> dict:
    """Read only the JSON header of a safetensors file (no tensor data)."""
    if isinstance(source, Path):
        with source.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            raw = handle.read(length)
    else:
        head = fetch(source, 8)
        length = int.from_bytes(head, "little")
        raw = fetch(source, 8 + length)[8:]
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header


def translate_config(upstream: dict, agent: dict) -> dict:
    """Map upstream ModernBERT+Laya config onto IC-Laya's ModelConfig fields."""
    rope = upstream["rope_parameters"]
    # `hidden_activation` is the encoder MLP activation; the decision head's
    # activation is not separately named upstream, and upstream defaults to GELU.
    activation = str(upstream.get("hidden_activation", "gelu")).capitalize()
    canonical = {
        "vocab_size": upstream["vocab_size"],
        "hidden_size": upstream["hidden_size"],
        "layers": upstream["num_hidden_layers"],
        "attention_heads": upstream["num_attention_heads"],
        "intermediate_size": upstream["intermediate_size"],
        "norm_eps": upstream["norm_eps"],
        "global_every": upstream["global_attn_every_n_layers"],
        "local_attention": upstream["local_attention"],
        "global_rope_theta": rope["full_attention"]["rope_theta"],
        "local_rope_theta": rope["sliding_attention"]["rope_theta"],
        # Layer 0 has no attn_norm tensor upstream, which is what this flag means.
        "first_layer_attention_norm": False,
        "decision_layers": agent["head_layers"],
        "decision_heads": upstream["num_attention_heads"],
        # head.layers.N.linear1.weight is [4096, 1024].
        "decision_ff": 4096,
        "decision_norm_eps": upstream["norm_eps"],
        "decision_norm_first": True,
        "decision_activation": activation,
        "scorer_norm_eps": upstream["norm_eps"],
        "qtypes": 3,
        "mask_token_id": UPSTREAM_SPECIAL["mask"],
    }
    return canonical


def build_mapping(hidden: int, layers: int, head_layers: int, encoder_config: dict) -> dict:
    """Canonical tensor name -> upstream source expression.

    QKV concatenation order is recorded as PyTorch's MultiheadAttention layout
    (Q, K, V). That is the documented default, but it is NOT verified against
    upstream logits, so `discover` lists it as an open question.
    """
    mapping: dict[str, object] = {
        "embeddings.weight": "encoder.embeddings.tok_embeddings.weight",
        "embeddings.norm.weight": "encoder.embeddings.norm.weight",
        "final_norm.weight": "encoder.final_norm.weight",
        "qtype.weight": "type_emb.weight",
        "scorer.norm.weight": "scorer.0.weight",
        "scorer.norm.bias": "scorer.0.bias",
        "scorer.dense.weight": "scorer.1.weight",
        "scorer.dense.bias": "scorer.1.bias",
        "scorer.out.weight": "scorer.3.weight",
        "scorer.out.bias": "scorer.3.bias",
    }
    for index in range(layers):
        prefix = f"encoder.layers.{index}"
        canonical = f"encoder.{index}"
        if index > 0:
            mapping[f"{canonical}.attn_norm.weight"] = f"{prefix}.attn_norm.weight"
        # Upstream already stores the projection fused as [3*hidden, hidden], which is
        # exactly the canonical layout, so this is a direct copy, not a concat.
        mapping[f"{canonical}.qkv.weight"] = f"{prefix}.attn.Wqkv.weight"
        mapping[f"{canonical}.out.weight"] = f"{prefix}.attn.Wo.weight"
        mapping[f"{canonical}.mlp_norm.weight"] = f"{prefix}.mlp_norm.weight"
        mapping[f"{canonical}.wi.weight"] = f"{prefix}.mlp.Wi.weight"
        mapping[f"{canonical}.wo.weight"] = f"{prefix}.mlp.Wo.weight"
    for index in range(head_layers):
        prefix = f"head.layers.{index}"
        canonical = f"decision.{index}"
        mapping[f"{canonical}.qkv.weight"] = f"{prefix}.self_attn.in_proj_weight"
        mapping[f"{canonical}.qkv.bias"] = f"{prefix}.self_attn.in_proj_bias"
        mapping[f"{canonical}.out.weight"] = f"{prefix}.self_attn.out_proj.weight"
        mapping[f"{canonical}.out.bias"] = f"{prefix}.self_attn.out_proj.bias"
        mapping[f"{canonical}.norm1.weight"] = f"{prefix}.norm1.weight"
        mapping[f"{canonical}.norm1.bias"] = f"{prefix}.norm1.bias"
        mapping[f"{canonical}.norm2.weight"] = f"{prefix}.norm2.weight"
        mapping[f"{canonical}.norm2.bias"] = f"{prefix}.norm2.bias"
        mapping[f"{canonical}.linear1.weight"] = f"{prefix}.linear1.weight"
        mapping[f"{canonical}.linear1.bias"] = f"{prefix}.linear1.bias"
        mapping[f"{canonical}.linear2.weight"] = f"{prefix}.linear2.weight"
        mapping[f"{canonical}.linear2.bias"] = f"{prefix}.linear2.bias"
    return mapping


def compare_shapes(mapping: dict, header: dict, expected: dict) -> list[str]:
    problems: list[str] = []
    for canonical, spec in sorted(mapping.items()):
        if isinstance(spec, dict):
            spec = spec.get("concat", [None])[0]
        if spec not in header:
            problems.append(f"{canonical}: source tensor {spec!r} not in checkpoint")
            continue
        actual = list(header[spec]["shape"])
        want = list(expected.get(canonical, []))
        # Canonical stores Linear weights as [out, in]; upstream does too.
        if want and actual != want:
            problems.append(f"{canonical}: {spec} shape {actual} != canonical {want}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("discover", "export"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument(
            "--header",
            type=Path,
            help="local safetensors file to read the header from instead of the network",
        )
        if name == "export":
            command.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    header = safetensors_header(args.header if args.header else f"{BASE}/model.safetensors")
    upstream_config = json.loads(fetch(f"{BASE}/encoder/config.json"))
    agent = json.loads(fetch(f"{BASE}/rl_agent_config.json"))
    canonical_config = translate_config(upstream_config, agent)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_reference import shapes  # noqa: E402

    expected = shapes(canonical_config)
    mapping = build_mapping(
        canonical_config["hidden_size"],
        canonical_config["layers"],
        canonical_config["decision_layers"],
        upstream_config,
    )

    missing = sorted(set(expected) - set(mapping))
    problems = compare_shapes(mapping, header, expected)

    # Source tensors the canonical pack has no slot for. These are not necessarily
    # dead: act_head is the escalation branch the design says to keep, and
    # `temperature` carries the per-primitive calibration the scalar Candid field
    # cannot express. Dropping them silently would lose required behaviour.
    claimed = {spec for spec in mapping.values() if isinstance(spec, str)}
    unclaimed = sorted(name for name in header if name not in claimed)

    elements = sum(math.prod(tensor["shape"]) for tensor in header.values())
    report = {
        "source_repo": REPO,
        "source_revision": REVISION,
        "upstream_architectures": upstream_config.get("architectures"),
        "params_millions": round(elements / 1e6, 1),
        "stored_dtype": sorted({tensor["dtype"] for tensor in header.values()}),
        "stored_bytes": sum(math.prod(t["shape"]) * 2 for t in header.values()),
        "canonical_f32_pack_bytes": sum(math.prod(t["shape"]) * 4 for t in header.values()),
        "canonical_f32_pack_limit_bytes": 2 * 1024**3,
        "special_tokens": UPSTREAM_SPECIAL,
        "special_token_literals": UPSTREAM_SPECIAL_LITERALS,
        "temperatures": {
            "by_primitive": agent.get("temperature"),
            "by_option_count": agent.get("temperature_by_options"),
        },
        "training": agent.get("training"),
        "deltas_from_assumption": [
            {"aspect": aspect, "canonical_assumed": assumed, "upstream_actual": actual, "consequence": why}
            for aspect, assumed, actual, why in ASSUMED_DELTAS
        ],
        "canonical_tensors_missing_from_mapping": missing,
        "upstream_tensors_with_no_canonical_slot": unclaimed,
        "shape_or_presence_problems": problems,
        "open_questions_not_resolved_here": [
            "QKV row order inside Wqkv / in_proj_weight (assumed PyTorch Q,K,V; unverified)",
            "RoPE layout and whether rotary is applied to Q/K after the head split",
            "Sliding-window semantics: upstream `local_attention=128` vs canonical window of local_attention/2",
            "Decision head is post-norm upstream pre-LN; confirm norm placement against source",
            "Whether the decision head sees the CLS row, pooled row, or mask-marker rows only",
            "What `head_max_len=256` and `max_prefixes=6` imply for the 128-token Compact128 profile",
            "act_head/escalate must be kept as an output branch; it is not read by the loader yet",
        ],
        "parity_verified": False,
    }

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "canonical.config.json").write_text(json.dumps(canonical_config, indent=2) + "\n")
    (out / "tensor_map.draft.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (out / "upstream.config.json").write_text(json.dumps(upstream_config, indent=2) + "\n")
    (out / "upstream.rl_agent_config.json").write_text(json.dumps(agent, indent=2) + "\n")
    (out / "port_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: report[k] for k in (
        "params_millions", "stored_dtype", "canonical_f32_pack_bytes",
        "canonical_f32_pack_limit_bytes", "special_tokens",
        "canonical_tensors_missing_from_mapping", "shape_or_presence_problems",
    )}, indent=2))
    print(f"\nwrote {out}/canonical.config.json, tensor_map.draft.json, port_report.json")
    print("parity_verified: false -- see open_questions_not_resolved_here in port_report.json")

    if args.command == "export":
        from pack_checkpoint import export  # noqa: E402

        manifest = export(
            source=args.weights,
            config=out / "canonical.config.json",
            mapping=out / "tensor_map.draft.json",
            tokenizer=_download_tokenizer(out),
            out=out / "pack",
            repo=REPO,
            revision=REVISION,
            qtypes=[0, 1, 2],
            test_only=False,
        )
        print(json.dumps({"pack_tensors": len(manifest["tensors"]), "pack_bytes": manifest["total_bytes"]}, indent=2))
    return 0


def _download_tokenizer(out: Path) -> Path:
    target = out / "tokenizer.json"
    if not target.exists():
        target.write_bytes(fetch(f"{BASE}/tokenizer/tokenizer.json"))
    return target


if __name__ == "__main__":
    raise SystemExit(main())

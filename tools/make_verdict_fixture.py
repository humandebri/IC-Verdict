#!/usr/bin/env python3
"""Build a tiny openJev (GLiClass) fixture pack for canister smoke runs.

The fixture exercises the same kernels as the real checkpoint — same crate, same
pack format, same upload and warm-up path — at a size a local replica can load in
seconds. It is NOT the model: `pack_verdict.py --test` marks it `test_only`, the
weights are deterministic random values, and the tokenizer is a whitespace
WordLevel stand-in. Nothing measured on it is evidence about checkpoint quality.

    python3 tools/make_verdict_fixture.py

Writes `fixtures/verdict-tiny/{config.json,tokenizer.json,manifest.json,model.bin}`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "verdict-tiny"

# Small but structurally faithful: two layers, one global and one sliding, a
# four-head attention, and a class token id that is a real added token.
CONFIG = {
    "encoder_config": {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 48,
        "norm_eps": 1e-5,
        "global_attn_every_n_layers": 2,
        "local_attention": 8,
        "cls_token_id": 1,
        "sep_token_id": 2,
        "pad_token_id": 0,
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    },
    "class_token_index": 3,
    "max_num_classes": 8,
    "projector_hidden_act": "gelu",
    "normalize_features": False,
}


def added(identifier: int, content: str) -> dict:
    return {"id": identifier, "content": content, "single_word": False, "lstrip": False,
            "rstrip": False, "normalized": False, "special": True}


TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        added(0, "[PAD]"), added(1, "[CLS]"), added(2, "[SEP]"),
        added(3, "<<LABEL>>"), added(4, "[MASK]"), added(5, "<<SEP>>"),
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "<<LABEL>>": 3, "[MASK]": 4,
            "<<SEP>>": 5, "[UNK]": 6, "insufficient": 7, "evidence": 8,
            "Question:": 9, "Context:": 10, "hello": 11, "true": 12, "false": 13,
        },
        "unk_token": "[UNK]",
    },
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=1))
    (OUT / "tokenizer.json").write_text(json.dumps(TOKENIZER, indent=1))
    command = [
        sys.executable, str(ROOT / "tools" / "pack_verdict.py"),
        "--safetensors", str(OUT / "tokenizer.json"),  # unused in --random mode
        "--config", str(OUT / "config.json"),
        "--tokenizer", str(OUT / "tokenizer.json"),
        "--out", str(OUT),
        "--source-repo", "local-fixture",
        "--test", "--random",
    ]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        return completed.returncode
    manifest = json.loads((OUT / "manifest.json").read_text())
    print(f"fixture ready: {OUT}")
    print(f"  specials cls={CONFIG['encoder_config']['cls_token_id']} "
          f"sep={CONFIG['encoder_config']['sep_token_id']} class={CONFIG['class_token_index']}")
    print(f"  example ids: [1,3,11,3,12,2] -> two class slots at positions 1 and 3")
    print(f"  tensors={len(manifest['tensors'])} bytes={manifest['total_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

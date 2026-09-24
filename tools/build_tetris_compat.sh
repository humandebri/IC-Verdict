#!/usr/bin/env bash
set -euo pipefail

# Build the temporary handover Wasm without adding game code to the final canister.
repo="$(cd "$(dirname "$0")/.." && pwd)"
compat_dir="$(mktemp -d "${TMPDIR:-/tmp}/ic-laya-compat.XXXXXX")"
trap 'rm -rf "$compat_dir"' EXIT

cp -R "$repo/Cargo.toml" "$repo/Cargo.lock" "$repo/rust-toolchain.toml" \
  "$repo/.cargo" "$repo/crates" "$repo/canisters" "$repo/tools" "$compat_dir/"
cp "$repo/tools/tetris_compat_query.rs" "$compat_dir/canisters/verdict-engine/src/tetris_compat.rs"
python3 - "$compat_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "canisters/verdict-engine"
lib = root / "src/lib.rs"
source = lib.read_text()
assert "mod tetris_compat;" not in source
assert "mod billing;" in source
lib.write_text(source.replace("mod billing;", "mod billing;\nmod tetris_compat;\nuse tetris_compat::TetrisQueryInput;", 1))

manifest = root / "Cargo.toml"
source = manifest.read_text()
assert "sha2.workspace = true" in source
manifest.write_text(source.replace("sha2.workspace = true", "sha2.workspace = true\nserde_json.workspace = true", 1))
PY

(cd "$compat_dir" && CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$repo/target}" bash tools/build_one.sh verdict-engine)
mkdir -p "$repo/build/compat-model-query"
cp "$compat_dir/build/verdict-engine.wasm" "$compat_dir/build/verdict-engine.did" "$repo/build/compat-model-query/"
cp "$repo/tools/tetris_compat_query.rs" "$repo/build/compat-model-query/tetris_compat.rs"
shasum -a 256 "$repo/build/compat-model-query/verdict-engine.wasm"

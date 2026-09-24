#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
package="${1:?usage: tools/build_one.sh decision-engine|verdict-engine|executor|mock-ledger}"
case "$package" in decision-engine|verdict-engine|executor|mock-ledger) ;; *) echo "unsupported package" >&2; exit 2;; esac
command -v cargo >/dev/null || { echo 'cargo is required; Rust builds were not verified in the delivery environment.' >&2; exit 1; }
mkdir -p build
run() { cargo "$@"; }
# Native Candid generation and Wasm use exactly the same source feature set.
run run --quiet -p "$package" --example "$package-export" > "build/$package.did.tmp"
test -s "build/$package.did.tmp"
mv "build/$package.did.tmp" "build/$package.did"
run build --release --target wasm32-unknown-unknown -p "$package" --lib
artifact="${package//-/_}"
# Honour CARGO_TARGET_DIR: cargo writes the artifact there, not under ./target.
# Resolve through cargo itself so a relative override, an absolute override, and
# a .cargo/config.toml build.target-dir all land on the same path.
target_dir="$(run metadata --no-deps --format-version 1 | "${PYTHON:-python3}" -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
wasm="$target_dir/wasm32-unknown-unknown/release/$artifact.wasm"
test -f "$wasm" || { echo "expected Wasm artifact not found: $wasm" >&2; exit 1; }
cp "$wasm" "build/$package.wasm"
if [[ "$package" == verdict-engine ]]; then
  "${PYTHON:-python3}" tools/attach_candid.py embed "build/$package.wasm" "build/$package.did"
fi
echo "Built build/$package.wasm and .did; no deployment performed."

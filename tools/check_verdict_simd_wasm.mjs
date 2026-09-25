// Run the SIMD correctness example in a real Wasm engine, without a replica.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';

const path = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/examples/wasm_check.wasm';
const bytes = readFileSync(path);
const { instance } = await WebAssembly.instantiate(bytes, {});
const start = performance.now();
const cases = instance.exports.check(); // A mismatch traps and fails this process.
assert.ok(cases > 0, 'the Wasm example must execute its checks');
const report = {
  cases, passed: true, milliseconds: performance.now() - start,
  wasm_sha256: createHash('sha256').update(bytes).digest('hex'),
};
if (process.argv[3]) writeFileSync(process.argv[3], JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));

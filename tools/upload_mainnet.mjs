// Upload to a staging canister with icp's existing signer, without exporting keys.
// A live model is cleared by begin_upload, so the public canister is never a target.
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';

const mode = process.argv[2];
assert.ok(mode === '--check' || mode === '--upload', 'Use --check or --upload --canister STAGING_ID');
const liveConfig = readFileSync('web/tetris/.env.production', 'utf8');
const liveId = liveConfig.match(/^VITE_CANISTER_ID=(\S+)$/m)?.[1];
assert.ok(liveId, 'Missing public canister ID in web/tetris/.env.production');
let target;
if (mode === '--upload') {
  assert.equal(process.argv[3], '--canister', 'Use --upload --canister STAGING_ID');
  target = process.argv[4];
  assert.ok(target && process.argv.length === 5, 'Specify one staging canister ID');
  assert.notEqual(target, liveId, 'Refusing to clear the live model; upload to a staging canister');
} else {
  assert.equal(process.argv.length, 3, 'Use --check without a canister ID');
}
const manifestBytes = readFileSync('models/verdict-pack/manifest.json');
const manifest = JSON.parse(manifestBytes);
const model = readFileSync('models/verdict-pack/model.bin');
const tokenizer = readFileSync('models/verdict-151m/tokenizer.json');
const hash = bytes => createHash('sha256').update(bytes).digest();
assert.equal(manifest.format, 'ic-verdict-int8-pack-v1');
assert.equal(manifest.test_only, false);
assert.equal(model.length, manifest.total_bytes);
assert.deepEqual(hash(tokenizer), Buffer.from(manifest.tokenizer_sha256));
let end = 0;
for (const t of manifest.tensors) {
  assert.equal(t.offset, end);
  assert.equal(t.encoding, t.shape.length === 2 ? 'i8_row_symmetric' : 'f32_le');
  const elements = t.shape.reduce((a, b) => a * b, 1);
  assert.equal(t.length, t.shape.length === 2 ? elements + t.shape[0] * 4 : elements * 4);
  end += t.length;
  assert.deepEqual(hash(model.subarray(t.offset, end)), Buffer.from(t.sha256), t.name);
}
assert.equal(end, model.length);
console.log(`Verified ${manifest.tensors.length} tensors; INT8 matrix pack ${model.length} bytes; model hash ${hash(manifestBytes).toString('hex')}`);
if (mode === '--check') process.exit(0);
const temp = mkdtempSync(join(tmpdir(), 'openjev-upload-'));
const blob = bytes => 'blob "' + bytes.toString('hex').replace(/../g, '\\$&') + '"';
function call(method, args) {
  const path = join(temp, 'args.did');
  writeFileSync(path, args);
  const reply = execFileSync('icp', ['canister', 'call', target, method,
    '--args-file', path, '--args-format', 'candid', '--candid', 'build/verdict-engine.did',
    '-n', 'ic', '--identity', 'production'], { encoding: 'utf8', timeout: 180000, maxBuffer: 1024 * 1024 });
  assert.match(reply, /variant\s*\{\s*Ok\b/, `${method}: ${reply}`);
  return reply;
}
try {
  call('begin_upload', `(${blob(manifestBytes)}, ${tokenizer.length}:nat64, record {cls=50281:nat32;sep=50282:nat32;mask=50284:nat32;pad=50283:nat32;literals=vec {"[CLS]";"[SEP]";"[MASK]";"[PAD]"}})`);
  const bytes = Buffer.concat([model, tokenizer]);
  for (let offset = 0; offset < bytes.length;) {
    const next = Math.min(offset + 1048576, bytes.length);
    const reply = call('upload_chunk', `(${offset}:nat64, ${blob(bytes.subarray(offset, next))})`);
    const confirmed = Number(reply.match(/Ok\s*=\s*([\d_]+)/)?.[1].replaceAll('_', ''));
    assert.equal(confirmed, next);
    offset = next;
    console.log(`Uploaded ${offset}/${bytes.length}`);
  }
  call('start_warmup', '()');
  let done = false;
  for (let i = 0; i < manifest.tensors.length; i++) {
    const reply = call('warmup_next', '()');
    console.log(`Warm ${i + 1}/${manifest.tensors.length}`);
    if (/Ok\s*=\s*true/.test(reply)) { done = true; break; }
  }
  assert.ok(done, 'Warm-up did not complete');
} finally {
  rmSync(temp, { recursive: true });
}

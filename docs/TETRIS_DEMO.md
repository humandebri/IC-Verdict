# On-chain Blocks — query-only demo

Large raw measurement JSON files mentioned below are local outputs. Git includes derived summaries with source hashes under [`artifacts/summaries/`](../artifacts/summaries/); see the [artifact policy](../artifacts/README.md). Full traces must be regenerated or obtained separately.

**Current browser rules:** [TETRIS_REALTIME.md](TETRIS_REALTIME.md). The browser
now uses independent gravity, NES NTSC-based rules and a four-option query adapter.
The playback mechanics, 100-piece limit and UI acceptance descriptions below are
historical. The legacy two-option backend API is retained for compatibility; see
TETRIS_REALTIME.md for the current frontend contract and comparison benchmark.

The browser owns the 10×20 board, seed, score and history. The resident per-row
INT8 model chooses between two rule-generated placements through a public query.
No game-state update, cross-canister call, external inference service or FP32
weight upload occurs while playing. Scores are unofficial and browser-editable;
query responses are not consensus-backed decision receipts.

Public UI: https://openjev.kinic.xyz. Mainnet target, deployment state, hashes,
and operator controls are recorded in [MAINNET.md](MAINNET.md).

## Run locally

Use the existing project local network (`icp network status -e local`) and a
disposable verdict-engine instance with the real per-row pack uploaded and warm.
Build the backend with `bash tools/build_one.sh verdict-engine`, then upgrade
that specific instance (do not reinstall). The old pack and owner are retained.
The new public demo is disabled by default. Re-warm after upgrading, then have
the existing owner call `set_tetris_enabled (true)` as an update.

```sh
cd web/tetris
npm ci
VITE_IC_HOST=http://localhost:8011 VITE_CANISTER_ID=<local-canister-id> VITE_IC_LOCAL=true npm run dev
```

Alternatively copy the values from `.env.example` into ignored `.env.local`.
`npm run build` produces `dist/` without deploying or modifying mainnet.
The checked-in `.env.production` selects the mainnet host/canister ID and sets
`VITE_IC_LOCAL=false`. Fetching a root key is explicitly restricted to loopback hosts.
No owner keys, identities or secrets belong in frontend environment variables.

## Rules and decision contract

- Seven-bag pieces, deterministic xorshift32 shuffle, seed 0 maps to `0x9e3779b9`.
- Normalize and deduplicate rotations; enumerate vertical-drop destinations,
  filtering to paths reachable from a centered spawn using left/right, rotation
  and downward moves. No wall kicks, hold or lookahead. Render legal path frames
  at ~105ms/step, a lock delay, then a line-clear animation. This is playback of
  the chosen placement, not model-controlled real-time input.
- Clear full rows and measure holes (empty cells below an occupied cell),
  maximum height, and lines cleared.
- Rank by `10 * lines - 8 * holes - height`; ties: rotation then x ascending.
  Present the best placement and the next placement with distinct features.
  If only one feature set exists, no model call is made and the UI says so.
- One Start button begins continuous play with a fresh random seed. A single
  board shows NEXT, score, lines and level; the benchmark still has a fixed-rule
  baseline, but it is no longer a second board in the UI.
- Stop on blocked spawn/no legal placement or at 100 pieces. Score is
  100/300/500/800 for 1/2/3/4 lines times the pre-clear level; level rises each
  ten lines. Reduced motion skips intermediate movement frames.
- The last actual query response is visible beneath the board: selected index,
  scores, tokens, instructions and round-trip time. Expand it for model hash
  (hex) and prompt; instruction counts use strings to preserve integer precision.
  Single-candidate turns are explicitly labeled as query-free.

`tetris_decide_query(vec TetrisFeatures)` accepts exactly two records:
`holes: nat16 (0..200)`, `height: nat8 (0..20)`, `lines: nat8 (0..4)`.
It returns `Result<TetrisReply, Error>` with selection 0/1, two scores, model
hash, input token count, inference instruction count and the actual prompt.
There are no caller-supplied strings, token IDs, temperature or hidden actions.
The prompt is `<<LABEL>>holes H height Y lines L` for each option followed by
`<<SEP>>Tetris: choose placement.`, with the checkpoint's CLS/SEP wrapping.
Temperature is 1.0; scores are not calibrated correctness probabilities.
The adapter checks 36 tokens and then the existing installed query cost guard.

`tetris_status()` is public and returns enabled/warmed, model hash and the
lesser of 36 and the installed query token ceiling. Only the existing owner
can call `set_tetris_enabled(bool)`. Generic inference remains allowlisted.
Public demo access has no durable per-user quota: query cannot update quota
state. Operators can disable it; UI throttling is not abuse prevention.

## Persistence and request lifecycle

The backend stores an explicit versioned wrapper around the unchanged old
Persistent payload. Restore tries the new wrapper, then the previous payload,
then the older legacy layout. Old installs default to disabled; existing
owner, model, upload metadata and cost configuration are retained. Enabled
state survives upgrades; the heap model must be warmed again as before.
Downgrading to a Wasm that does not understand the new wrapper is not supported.

Only one UI turn is in flight. Start is disabled until a run ends or fails.
Hidden tabs pause playback and subsequent requests; visibility resumes the run.
A 15-second timeout ignores a late response but does not claim to cancel remote
execution; the lock stays until transport completion to avoid concurrent
retries. No update or fixed-rule fallback is used on error. A changed model
hash stops the game until restart. Inference requests are at least one second apart.

## Verification commands

```sh
cargo test -p verdict-engine --lib
cargo test -p verdict-engine real_tokenizer_all_feature_values_fit_36_tokens -- --ignored --nocapture
python3 tools/tetris_smoke.py --canister <local-id> --owner <local-test-identity> --pem <local-test-pem>
cd web/tetris
npm test
npm run build
IC_HOST=http://localhost:8011 IC_LOCAL=true CANISTER_ID=<local-id> npm run bench
```

The smoke test toggles the demo, upgrades the specified disposable canister,
warms the saved pack and leaves it enabled. It checks anonymous game access,
unauthorized administration, generic query denial, bounds and restoration.
It records model outputs separately from allocation-sensitive instruction counts.
Benchmark output is `artifacts/tetris_benchmark.json`: ten seeds, up to 100
pieces each, per-turn measurements and explicit errors; model wins are not a gate.
Local-only `BENCH_INTERVAL_MS=0` removes the benchmark's default one-second
pause; it does not change UI throttling. Query timings include local transport
and may benefit from replica query caching, so they are not mainnet latency
claims. The returned instruction count measures inference, not the entire request.

For interactive UI regressions, start Vite and run the function in
`web/tetris/scripts/ui-check.js` using Playwright CLI `run-code`. It mocks only
the API module for failure/timing cases, labels its mock prompt, and removes
the route afterward. Real-canister smoke and browser tests are separate evidence.

## Recorded local acceptance (2026-09-22)

The results below describe the original two-board UI and original candidate
enumeration, before the arcade redesign. They are not a benchmark of the new
reachable-path filtering. Arcade acceptance: 15 unit tests and 13 mocked browser
checks passed (single-button autoplay, movement, query fields, single-flight,
visibility pause/resume, 390px layout, errors, disabled service, timeout/late
reply, reduced motion). The canister and model are unchanged.

Test target: `4fbx2-kt777-77775-aaabq-cai`, `http://localhost:8011/`.
The pre-demo per-row instance was upgraded without reuploading the weights;
the demo initially reported disabled. A second new-version upgrade preserved
the enabled flag and identical selection/scores after warm-up.

- Rust workspace tests and Wasm build passed. The dedicated tokenizer test
  checked all 21,105 feature tuples repeated in both slots: maximum 24 tokens.
- Frontend build and 13 unit tests passed. Chromium checked 15 UI scenarios,
  including stop/reset, timeout/late reply, disabled service, model change,
  hidden tab, duplicate click, and a 390px-wide mobile layout.
- Real browser query selected a placement and advanced the board; recorded
  traffic used query/read_state, not update calls, during play.
- Ten fixed seeds, at most 100 pieces each: 780 model placements, 767 inference
  queries, zero query errors, maximum 2,625,118,018 inference instructions.
- Total lines: model **171**, fixed rule **239**. By lines: model wins 3,
  loses 6, draws 1. This is a small, heuristic-assisted demo, not a general
  Tetris intelligence or model-quality benchmark.
- Mean local query round trip was approximately 185ms, with query cache effects
  possible. This is not a mainnet latency estimate.

Evidence: `artifacts/tetris_smoke.json`, `artifacts/tetris_benchmark.json`.
The local test target remains enabled and warm; no mainnet deployment occurred.

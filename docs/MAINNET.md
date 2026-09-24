# OpenJev mainnet deployment

## Browser-owned Tetris and generic model query release (2026-09-24)

- Public site: https://openjev.kinic.xyz, Worker `openjev` version
  `51a2dde9-f9d2-425f-b921-1d58b6617a2a`.
- The existing canister was first upgraded to compatibility Wasm
  `584657715086775ff0e785053b175c777d27261abd13029b71bc1aeeb662204e`
  and warmed. After the Web switch and public smoke, it was upgraded to the
  final model-only Wasm
  `2c403877b2fac5c4563fe33fa71712174ed4818c1a499fbdf0621eaa2afff680`
  and warmed again. The old Tetris-specific API is absent from the final Candid.
- The original model digest was retained. Anonymous `decide_query` succeeded on
  both versions. The public Chrome smoke after the final upgrade completed one
  model turn with one decision query and no game update; comparison mode
  completed two turns without canister traffic.
- Final status snapshot: Running, 732,530,282,054 cycles. No model upload,
  funding, DNS or controller change. Deployment used `bridge-seal-log-reader`,
  which has the same principal as `production`; the `production` Keychain alias
  itself was unavailable. Wrangler OAuth was authenticated.
- See [TETRIS_BROWSER_MODEL_QUERY.md](TETRIS_BROWSER_MODEL_QUERY.md) for the
  browser/canister boundary, local checks and release sequence.

Created on 2026-09-22. Balances below are creation-time snapshots.

- Network: `ic`
- OpenJev canister: `qojfj-6qaaa-aaaam-qjkaq-cai`
- Controller identity: `production`
- Controller principal: `lqfvd-m7ihy-e5dvc-gngvr-blzbt-pupeq-6t7ua-r7v4p-bvqjw-ea7gl-4qe`
- Creation funding: 2,000,000,000,000 cycles from the identity's cycles-ledger account.
- Ledger debit observed: 2,000,100,000,000 cycles including ledger fee.
- Execution balance after creation: 1,499,998,056,000 cycles; creation costs
  account for the difference from the requested funding (plus initial usage).
- Remaining source ledger balance: 2,584,220,847,489 cycles.
- Initial status: Running, no installed Wasm (`module_hash: null`).

The canister was created with `--detached`; no project ID mapping was changed.
Neither existing Wiki staging nor the previously identified Index canister
was modified.

## Relative-cost Tetris production release (2026-09-23)

- Public site: https://openjev.kinic.xyz; Worker `a697c497-7645-4ebb-88a9-639c9b35cda9`.
- The same canister was upgraded to Wasm `78f7b42df1aa878f76f2c0e9f978388cbd613469ec29433f680e9edd11e95d34`, identical to the locally tested build. Existing model/stable data retained; all 142 tensors warmed. No model upload or funding.
- Candidates now cover up to four distinct feature outcomes. Independent relative costs for holes, height and missed clears feed the model; the combined heuristic score is not passed. Maximum input is 41 tokens; each game turn still uses one query.
- Three production model queries matched native selections, candidates and input boards. Maximum measured model instructions: 3,791,167,093. Zero game update calls; stored game/budget counters unchanged.
- The English UI discloses candidate grouping and per-feature cost/rank calculation. Frontend build and Wrangler dry-run passed. Public Chromium advanced eight turns / two cleared lines from one Start click, showing 41 tokens and one model pass/query, with zero console errors or game update calls.
- Evidence: `artifacts/tetris-prompt-search/production-release.json`, `production-query-check.json`, and `production-status.json`. Earlier sections below are historical.
- Rollback artifacts: `build/tetris-placement-query/verdict-engine.wasm` and Worker `7853f415-03e5-4810-b44b-bc66bf733961`; a Wasm rollback also requires model warmup.

## English single-click UI release (2026-09-23)

- Worker `7853f415-03e5-4810-b44b-bc66bf733961` translates all visible copy, settings, descriptions, and accessible labels to English (`lang=en`).
- Start game and New game immediately enable autoplay. Pause remains available during inference/animation, stopping subsequent turns after the current move finishes. Resume continues without creating a new game.
- Canister and four-candidate, one-query protocol unchanged; no backend update or warmup for this UI release.
- Build/dry-run passed. Local Chromium verified a single Start click advancing 23 turns, Pause, no Japanese text, no 390px overflow or console errors. Public Chromium also advanced three turns from one Start click with English-only text.

## Four-candidate query production release (2026-09-23)

- Current public UI: https://openjev.kinic.xyz, Worker `f7f52122-e960-4f3a-8cc9-4bf02af57ac9`.
- Same canister upgraded to `e28b37bd2a31ce0352359b254029852088c30d4506939853d5352d9b966adf76`. Model unchanged, all 142 tensors warmed; no model reupload or funding.
- Game start is browser-only. Each move invokes `tetris_placement_query` exactly once with board/seed/turn/lines/mode. Canister uniformly samples up to four legal placements independently of evaluation quality, performs model selection and computes the next board.
- The browser holds the board. Query replies are not persistent certified decision records; the UI now exports input/output JSON without a certification claim.
- Old update-model allowance set to zero. The query game has no shared 50-turn budget or 32-session capacity; individual games still stop at 128 turns.
- Production Chromium verified no traffic on start, exactly one additional query for one move, zero update `/call` requests, and zero console errors. Four candidates, 43 tokens, 4,019,862,822 model instructions, 1.00 s measured response including network. No horizontal overflow at 390px.
- Local real-model regression: three model turns and three heuristic turns, each below 5B model instructions, no update calls or stored game/budget changes, and independently replayed board/features. Unit tests/build/dry-run passed.
- Evidence: `artifacts/tetris-placement-query/`. Prior update release below is historical. Current behavior is documented in [TETRIS_PLACEMENT_QUERY.md](TETRIS_PLACEMENT_QUERY.md).

## Placement demo production release (2026-09-23)

- Published https://openjev.kinic.xyz with the restored green arcade UI and on-canister placement selection.
- Upgraded the same canister with locally tested Wasm `ed2dc08bdbb105c4409722a637197a01728fea1a00f53af98b64753db2940e32`, frozen under `build/tetris-placement-release/`. No reinstall, model reupload, funding, or controller changes.
- Preserved model `0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0`; all 142 tensors warmed.
- Worker version `a897e9d4-2af0-4ab5-ac16-0d5a8df8a519`; previous frontend version `5c07f5a5-f6e9-43aa-860b-4583e58e4ace` remains the frontend rollback candidate. The old Wasm does not understand the new game snapshot: do not downgrade it blindly.
- Public Chromium: one real model turn, 12 candidates, 99 input tokens, 9,625,341,081 model instructions, 3.09 s response including network/consensus. Mainnet certificate, canister ID, certified index and record #1 hash verified; zero console errors.
- Measured balance decrease across game creation, one move and idle time: 9,705,538,155 cycles. Remaining after smoke: 1,008,474,588,622 cycles. These are snapshots, not fixed per-turn pricing.
- Initial model allowance was 8 turns for the smoke; set to 50 remaining model turns after measurement. This public allowance is shared across users and stops at exhaustion; no automatic refill. Heuristic mode does not consume it. Existing 32-game capacity also applies.
- Evidence: `artifacts/tetris-placement/production-*.json`. Build and Wrangler dry-run passed before publishing. Earlier local-only statements below describe historical releases.

## Tetris v3 production release (2026-09-23 JST)

- Upgraded the existing canister with the locally accepted history-capable Wasm:
  `0962574cae166a5a438e62a588d3351408613f52b867e6e18ed7311ba3d8074b`.
  The frozen artifact is `build/tetris-query-v3-history/verdict-engine.wasm`;
  unrelated in-progress source changes were not rebuilt into this release.
- Preserved the saved model and enable flag; warmed all 142 tensors using the
  existing `production` identity. Model hash remains `0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0`.
- Applied the tested cost guard: fixed 170M, per-token 92M, update budget 40B;
  the v3 ceiling is 52 tokens. A mainnet 52-token history request succeeded with
  4,866,657,354 measured instructions, below the 5B query limit. Legacy v1 works.
- Published Worker `openjev`, version `5c07f5a5-f6e9-43aa-860b-4583e58e4ace`.
  Frontend rollback version: `2c90baef-582a-4976-9516-b7f29ecf1dcf`;
  the new backend retains the old frontend's API. No reinstall is needed.
- Frontend tests, production build and Wrangler dry-run passed. Public Chromium
  smoke reached 2 placements, 31 requests, 30 replies and 29 executed controls,
  with 2 expired decisions and zero query/page errors. Mean response RTT was
  0.884 s, maximum 1.101 s; one request was pending at the checkpoint.
  The actual prompt included previous pose/action/outcome and older actions.
  No mobile overflow at 390px; gameplay used query traffic, no update calls.
- Post-release status: Running, expected Wasm/controller, execution balance
  1,069,805,948,346 cycles (snapshot). No funding or model reupload performed.
- With one actual query deliberately held in Chromium, the piece fell two rows
  before any reply; releasing the request produced a real model response.
  The first harness attempt failed to detect the binary request; switching to
  byte-buffer matching corrected the harness, after which the check passed.
- Evidence: `artifacts/tetris_query_v3_production_{release,smoke,live,gravity}.json`.
  This is a short functional check, not evidence of stronger play.

The remaining sections document earlier releases and their historical values.

## Release configuration (2026-09-22)

- Wasm SHA-256: `b4da1963c401f72a4d34913018edd644a29d3250b856917a7befda44b3e7e123`
- Model manifest SHA-256: `0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0`
- Matrix representation: per-row INT8; pack 152,245,512 bytes, 142 tensors.
  One-dimensional normalization/bias auxiliaries and row scales remain FP32.
- UI: https://openjev.kinic.xyz
- Cloudflare account: `9029b5f9de5b2e820eaf4ed562bcb0e7`, Worker: `openjev`
- Worker version: `2c90baef-582a-4976-9516-b7f29ecf1dcf` (English NES-based real-time UI)
- Previous UI / rollback version: `4448392c-58b0-460d-ab26-02171a8766c6`
- Earlier arcade version: `32c0daea-9096-40fd-8600-46293f014834`
- Initial two-board version: `1b6df52f-d013-4c4b-a63d-ee586ec3c52b`
- Deployment CLI: existing global Wrangler 4.107.0 (no tool upgrade).
- UI configuration: `web/tetris/wrangler.jsonc`, `web/tetris/.env.production`.
- Source: working tree based on `b5d20f4`, including the Tetris implementation
  and production configuration. Not a committed release tag.

Backend Wasm is installed, all 142 tensors are warm, and public gameplay is
enabled. The deployed model hash matches the manifest hash above.
The uploader uses the existing `production` signer via icp, without exporting
private keys. `node tools/upload_mainnet.mjs --check` verifies all tensor hashes
and matrix encodings without remote writes. `--upload --canister STAGING_ID`
starts a fresh upload only on a separate staging canister; the script refuses
the canister configured in `web/tetris/.env.production`. Verify the staged model
before changing the public frontend's canister ID. Do not use this command to
resume an interrupted upload.

Rebuild frontend with `npm --prefix web/tetris run build`, preview with
`wrangler deploy --dry-run --config web/tetris/wrangler.jsonc`. Remote deploy
requires approval. Frontend rollback uses a recorded known-good Worker version;
the preceding Japanese real-time release is retained as the rollback version. To suspend gameplay (owner-only
update; requires operator authorization):

```sh
icp canister call qojfj-6qaaa-aaaam-qjkaq-cai set_tetris_enabled '(false)' -n ic --identity production --candid build/verdict-engine.did
```

Do not reinstall to roll back: reinstall destroys canister state. Normal
upgrades preserve the packed model but require warm-up again before inference.

## Production verification

### English UI release

- UI copy, accessible labels and errors translated; document language is `en`.
  Game mechanics, model input and canister are unchanged.
- 29 unit tests, build, dry-run and 12 controlled browser checks passed.
- Public Chromium smoke confirmed English on the unversioned URL, Start-to-query
  operation, full response display, and no overflow at 390px. One actual response
  contained 24 tokens and 2,734,384,586 instructions with the unchanged model hash.
- The controlled test route accepts Vite cache-busting query parameters.

### Current real-time release

See [TETRIS_REALTIME.md](TETRIS_REALTIME.md) for exact rules, deviations from NES,
clock semantics and benchmark definitions. This release still changes only static
frontend assets, with no canister/model/funding/DNS mutations.

- 29 unit tests, production build and Wrangler dry-run passed.
- 10 controlled browser checks passed: pending-query gravity, lock before reply,
  stale-result rejection, single flight, error continuation, benchmark validity
  on visibility changes, returned fields and 390px layout.
- Live mainnet smoke with seed 42 reached three placements and three actual
  model responses with no deadline/placement misses, skips or query errors at
  that checkpoint. Mean RTT ~0.6482s, max ~0.7548s. One response: 24 tokens,
  2,734,607,339 instructions, matching the unchanged manifest hash.
- This is a short functional smoke, not a completed-game benchmark or latency SLA.
- Final smoke checkpoint: 36 placements, 11 lines, level 1 (715 ms/row), 37
  responses, score 3262, no deadline/placement misses, skips or query errors;
  benchmark validity remained true. The 390px viewport had no overflow and the
  unversioned URL served the NES UI. A navigation-interrupted test predicate
  logged one harness TypeError; no application exception was observed.

### Earlier releases

The measurements below are from the initial deployment. The subsequent arcade
UI changes only static frontend assets; Wasm, model, owner and DNS remain unchanged.
Arcade release: one-button autoplay with legal movement playback, NEXT/score/lines,
and expandable actual query responses. Fifteen unit tests and thirteen mocked
browser regression checks passed before publication.
Live Chromium then confirmed 12 automatic placements after one Start click,
actual query values (24 tokens; one observed reply 2,734,527,597 instructions),
zero console errors, query-only inference traffic, and no overflow at 390px.
The unversioned public URL serves the new arcade title. No canister upgrade,
model upload, additional funding or DNS change was performed for this UI release.

- Backend build and Rust library tests passed; UI 13 tests and production build
  passed; Wrangler dry-run passed before publication.
- Anonymous `tetris_status`: enabled and warmed; adapter cap 36 tokens.
  Generic `query_limits`: 40 tokens, 5,000,000,000 instruction budget.
- Anonymous edge-feature query: 24 tokens, 2,734,826,153 measured instructions,
  selected candidate 0. Empty candidate list rejected; generic anonymous
  inference remains Unauthorized after enabling the game.
- Live Chromium browser at the public HTTPS domain: seed 42, two placements;
  the first required no inference, the second used real model inference and
  selected A with 62.2% score (not calibrated confidence). Round trip ~0.69 s,
  24 tokens, 2,734,384,586 measured instructions. This is one measurement, not
  a latency guarantee or load test.
- Browser calls used query/read_state, not game-state updates. Console had
  no errors. Both boards advanced. The production viewport was resized to
  390px; detailed mobile-layout regression coverage remains the local test.
- Post-deployment execution balance: 1,121,959,076,685 cycles; memory reported
  402,915,838 bytes; idle burn 11,157,710,295 cycles/day. These are snapshots,
  not fixed running costs; automatic top-up/monitoring is not configured.
- No additional funding, existing-canister changes, key exports, or Git commit.

Read current state before any deployment; the recorded balances are snapshots:

```sh
icp canister status qojfj-6qaaa-aaaam-qjkaq-cai -n ic --identity production --json
icp cycles balance -n ic --identity production --json
```

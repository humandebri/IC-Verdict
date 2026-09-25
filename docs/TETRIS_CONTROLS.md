# Query v3: repeated direct controls

Large raw measurement JSON files mentioned below are local outputs. Git includes derived summaries with source hashes under [`artifacts/summaries/`](../artifacts/summaries/); see the [artifact policy](../artifacts/README.md). Full traces must be regenerated or obtained separately.

The browser now asks the model for one input at a time throughout a falling piece.
It does not select a landing or automatically execute a route. The model is still
a classifier: its candidates are legal left/right/clockwise/down/wait actions, not
arbitrary generated text. There is no limit on the number of decisions per piece.
The UI keeps only the latest 100 request records to bound memory. Each request
also carries context from the same falling piece; nothing is persisted inside
the model between query calls.

## Contract

`tetris_decide_v3_query` accepts the numeric `TetrisControlRequest`:

| Field | Meaning / bound |
| --- | --- |
| piece, next | 0..6 = I, O, T, S, Z, J, L |
| x, y | Pivot column 0..9 and row 0..19 |
| rotation | Index in the piece's existing fixed-pivot rotation table |
| heights | Exactly ten column heights, each 0..20, measured on locked cells |
| holes | Total empty cells below a covered cell in a column, 0..200 |
| legal_mask | Bits 0..4 = left, right, clockwise, down, wait; down/wait required |
| previous | Optional previous query pose (x/y/rotation), selected action 0..4 and outcome 0 executed / 1 blocked / 2 discarded |
| history | Up to two earlier executed action IDs, in chronological order; requires previous |

The client excludes lateral moves/rotations that collide, and excludes rotation
for O. Down is always a candidate, including when it would lock the piece.
The server validates bounds, rotation count, and action mask. The browser remains
the authority for game physics; this endpoint does not certify board correctness.

The prompt uses canonical-order labels `left`, `right`, `rotate`, `down`, `wait`
and the fixed state text:

```
Clear lines. Tetris T x5 y0 r0 next I heights 0 0 0 0 0 0 0 0 0 0 holes0
```

Only enabled labels are included. The five-label layout uses at most 40 tokens without history and 52 with history
with the checkpoint tokenizer. `tetris_v3_status` reports the smaller of 52 and
the installed query cost guard's limit. The UI requires 52 before starting;
requests are never silently truncated. When available, the prompt appends
`prev x5 y0 r0 right executed history left right`. The previous pose is the
snapshot on which that previous answer was based. The final two words are up to
two successful actions before that answer, not proposed or discarded inputs.
Previous board/NEXT data are not duplicated: locked cells and NEXT stay constant
while the same piece falls. Context resets on a new piece. Errors with no chosen
action do not invent one; blocked/discarded previous replies are marked explicitly. Both the adapter cap and 5B instruction
budget guard apply. Full occupancy and hole positions are not sent to the model.

The reply includes `action` (the stable 0..4 ID, **not an index into the candidate
list**), `actions` in canonical order, corresponding `scores`, model hash,
`input_tokens`, `measured_instructions`, and the actual `prompt`. Scores are not
calibrated probabilities of correct play. The original v1/v2 methods remain
available. All three share the existing persisted enable switch; generic inference
and operator mutations retain their existing authorization. No new stable state
or public update method is introduced.

## Timing, stale state and failures

NES-derived physics, scoring, randomizer, top-out, and entry delays are retained.
The only rotation control is clockwise, without wall kicks. A successful down
input moves one row, earns one soft-drop point on lock, and resets the gravity
counter; a blocked down input locks immediately. It cannot also cause a gravity
drop in the same frame. Wait does not reset gravity or hold a key.

There is at most one request in flight and at least one second between starts.
Gravity continues during inference, errors and queued input. On arrival, the
engine checks request ID, piece ID and visibility epoch. It queues one input for
the next eligible frame (six-frame lateral/rotation and two-frame down cadence),
then checks collision against the actual pose again. A blocked input is recorded
without a replacement action. After execution, rejection or failure the next
eligible request uses a fresh snapshot. Pending and queued work prevent duplicate
requests; duplicate replies have no effect.

Locking invalidates that piece's pending/queued work. Hiding or resuming the tab
changes an epoch, preventing pre-pause responses from affecting resumed play.
A hidden tab pauses and invalidates benchmark comparability, as before. Long
foreground stalls account for elapsed physics frames and invalidate comparability.
Model changes terminate the browser run. Network errors release the request slot
and gravity continues; retries obey the normal dispatch interval.

Metrics distinguish queries, executed inputs, blocked inputs, late decisions,
visibility-stale decisions, errors, and successful response RTT. The operation
history lists returned actions and their execution outcomes explicitly. An error
can also count as a deadline miss if the piece locked before the error arrived.

## Verification and comparison

Unit tests cover repeated inputs on one piece, single execution, cadence, down
scoring/lock, collisions after query delay, lock-before-response, pause epochs,
errors/retries, bounded history, continued gravity and retained NES rules.
`controls_real_tokenizer_budget` checks every piece/NEXT/pose combination, every
height value in each column, all hole counts, valid masks, previous poses, actions, outcomes and earlier-action combinations against 52 tokens.

`tools/tetris_v3_smoke.py` upgrades only an explicitly selected existing managed
loopback target, warms its saved weights, checks v1/v2/v3 and the operator switch,
and verifies that generic inference remains protected. `web/tetris/scripts/ui-check.js`
is the controlled Chromium check; `ui-live.js` exercises the actual local query.

Run the comparison with the existing `npm run bench` command and explicit local
`IC_HOST`, `IC_LOCAL=true`, and `CANISTER_ID`. Defaults are ten seeds, a 100-piece
safety limit and one simulated hour. Truncation is reported; actual UI play has no
piece limit. Each policy uses the same seeded piece stream and dispatch cadence.

- Model: actual local query RTT advances the virtual frame clock, including failures.
- Random: uniform legal action, with an independent seeded random stream and zero RTT.
- Fixed rule: full-board search over clockwise-only reachable landings, ranked by
  `10*lines - 8*holes - height`, then shortest route, rotation, x and y. Only the
  next action is applied before re-evaluation; no target route is automatically played.
  No reachable landing means wait; an already-reached landing means down. RTT is zero.

The fixed rule sees more information than the model. These are end-to-end policy
comparisons, not a matched-observation model-quality test. Local RTT excludes
browser rendering and does not predict mainnet latency. Results go to
`artifacts/tetris_query_v3_history_benchmark.json`; the initial 40-token v3 run
is in `tetris_query_v3_benchmark.json` and v2 evidence is historical.

The local acceptance below was followed by the authorized production release on
2026-09-23 JST; see [MAINNET.md](MAINNET.md). The frontend requires the
history-capable v3 and calibrated 52-token guard and must not be published
independently against an older backend.

## Initial 40-token acceptance (2026-09-23 JST)

- Existing disposable target: `4fbx2-kt777-77775-aaabq-cai`, `http://localhost:8011/`.
- Upgrade controller: anonymous (`2vxsx-fae`); warm-up/switch owner:
  `ic-verdict-acceptance-h1vkKw`. Saved weights, owner and enable state retained.
- Frozen Wasm SHA-256: `35a16200b030bbb5323fa0a63eed438754d06d3bce988547c3409bb9aca1eec9`.
- Model hash: `0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0`.
- Rust unit tests, tokenizer-bound checks, frontend tests/build and local smoke passed.
  The smoke covered 44 checks; the five-action boundary request used
  40 tokens and 3,613,179,897 inference instructions, below the 5B limit.
- Controlled Chromium checks passed 17 assertions including repeated decisions,
  falling while pending, late/error responses and 390px layout.
- Live Chromium seed 42 reached 2 placed pieces, 17 queries and 17 executed
  inputs with zero errors, blocked or late inputs and no page exceptions/overflow.
  Successful-response mean RTT was 755.0 ms, max 10202.4 ms.
  The slow initial request is included; these are local observations, not a latency guarantee.
- Evidence: `artifacts/tetris_query_v3_smoke.json`, `tetris_query_v3_ui.json`,
  and `tetris_query_v3_live.json`. No production deployment was performed.

The 40-token comparison was retained while history support was being built; local
builds and other host work overlapped parts of that run. It is not an isolated
latency comparison against the history version.

Initial 40-token run (10 seeds):

| Policy | Lines | Pieces | Queries | Late | Truncated runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| model | 0 | 120 | 819 | 71 | 0 |
| heuristic | 158 | 625 | 0 | 0 | 2 |
| random | 0 | 151 | 0 | 0 | 0 |

All query errors were zero. The two truncated fixed-rule runs reached the
100-piece benchmark limit; their results are not completed games.

## History-capable 52-token acceptance (2026-09-23 JST)

- Same local target and model as above. Frozen Wasm SHA-256:
  `0962574cae166a5a438e62a588d3351408613f52b867e6e18ed7311ba3d8074b`.
- After upgrading and warming the verified optimized kernel, the owner set
  `cost_fixed=170000000`, `cost_per_token=92000000`, update budget 40B.
  `query_limits` and `tetris_v3_status` now report 52; v1/v2 caps remain 36/40.
  This changes only the selected local target's calibration; no production settings changed.
- 52 local smoke checks passed, including no-history startup, history bounds,
  previous action/outcome bounds and all three APIs. The 52-token boundary request
  used 4,755,634,061 forward instructions.
- Rust tests, tokenizer checks, 40 frontend tests and the frontend build passed.
  Eighteen controlled Chromium assertions passed, including observing previous
  input/action/outcome in the next query.
- Actual Chromium queries reached 2 placed pieces with 12 requests, 11 replies,
  10 executed controls and 2 deadline misses. One request remained pending at the
  snapshot. Query/page errors and mobile overflow were zero. Mean completed-response
  RTT was 2445.8 ms, maximum 7213.9 ms. The returned prompt
  included `prev x5 y0 r0 right executed`, confirming server-side history rendering.
- Evidence: `artifacts/tetris_query_v3_history_smoke.json`,
  `tetris_query_v3_history_ui.json`, and `tetris_query_v3_history_live.json`.

Identical requests may benefit from the replica's query cache. Reported instruction
sums add the values carried by replies; they are not a cache-adjusted measurement
of total host computation. Host load is uncontrolled, so RTT changes between
these runs do not establish a causal history or kernel speed effect.

History-capable 52-token run (same 10 seeds):

| Policy | Lines | Pieces | Queries | Late | Blocked | Truncated runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model | 0 | 126 | 701 | 115 | 5 | 0 |
| heuristic | 158 | 625 | 0 | 0 | 0 | 2 |
| random | 0 | 151 | 0 | 0 | 0 | 0 |

All model/random games reached top-out; two fixed-rule games reached the 100-piece
limit. Model query errors were zero. Mean local RTT was 1474.6 ms,
maximum 8495.7 ms. Maximum input length was 52 tokens.
The model repeated its previous action in 421/460 cases where that action
was still available. Both model versions cleared zero lines; history support is
not evidence of stronger play. Full-board access and zero RTT favor the fixed rule.

An independent audit checked all 7481 recorded decisions against earlier inputs
and actual execution outcomes, including resetting context on a new piece. All
20 fixed-rule/random baseline results matched the initial run exactly. Every real
model reply fit the 52-token/5B bounds and returned only actions in the sent mask.
The installed Wasm hash still matched the frozen history-capable build after
measurement. Evidence: `artifacts/tetris_query_v3_history_benchmark.json` and
`artifacts/tetris_query_v3_history_validation.json`.

## Input-only follow-up

See [TETRIS_INPUT_STUDY.md](TETRIS_INPUT_STUDY.md) for the frozen-weight input
comparison, token bounds and quality gate. This study does not change public v3.

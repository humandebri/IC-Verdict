> Historical query v2 design and results. Current source uses [repeated direct controls (v3)](TETRIS_CONTROLS.md). Production release history is recorded separately in MAINNET.md.

Large raw measurement JSON files mentioned below are local outputs. Git includes derived summaries with source hashes under [`artifacts/summaries/`](../artifacts/summaries/); see the [artifact policy](../artifacts/README.md). Full traces must be regenerated or obtained separately.

# Real-time benchmark: NES NTSC-based rules

This supersedes the browser playback mechanics in TETRIS_DEMO.md. The v2 source described here
uses the four-option query v2 adapter; see MAINNET.md for the separately recorded
production release. The model weights and NES timing rules are unchanged. Gravity advances while
the network request is outstanding, and late decisions cannot change locked
pieces. The primary control remains a single Start button.

## Rules and sources

The reference is Nintendo's NES NTSC A-Type, starting at level 0. Gravity uses
60.0988 simulation frames per second and the frames-per-row table
48,43,38,33,28,23,18,13,8,6,5,5,5,4,4,4,3,3,3, then 2 through level 28,
then 1 from level 29. Level increases each ten lines from the level-0 start.
Line scores use 40/100/300/1200 times (level+1); a level-changing clear uses
the updated level. NEXT shows one piece. Fixed pivot rotations have no kicks.
An unsuccessful scheduled downward move locks immediately. There is no hold,
hard drop, additional lock delay, or 100-piece stop. Play ends on top-out.

Sources consulted (no ROM, graphics, music or assembly routines incorporated):

- [Nintendo instruction manual transcription](https://www.world-of-nintendo.com/manuals/nes/tetris.shtml): A-Type, controls, NEXT and scoring.
- [NES disassembly](https://github.com/CelestialAmber/TetrisNESDisasm/blob/master/main.asm): `framesPerDropTable`, `orientationTable`, `pointsTable`, line/level update ordering and spawn/lock behavior.

This is a rules-based browser implementation, **not a cycle-exact emulator**.
Differences are intentional and disclosed in the UI:

- Seeded xorshift plus one reroll replaces the frame-dependent NES RNG. It is
  not seven-bag; repeats and droughts can occur. `?seed=42` replays the same stream.
- The deterministic controller issues at most one lateral/rotation input per
  six frames. It does not emulate DAS charging/roll inputs. Soft drop moves one
  row every two frames and awards one point per successful soft-drop row.
- Score is not capped; original BCD scoring bugs and post-255-level bugs are
  not emulated. Entry delay is height-based (10–18 frames), plus 17–20 frames
  on line clear. No original artwork/sounds are used.
- Browser visibility pauses the session and marks its benchmark invalid for
  comparisons. Reduced-motion styling does not change physics or deadlines.

## Decision loop

At spawn, enumerate reachable landing options without ranking their quality.
Keep up to four: distribute slots among available rotations, then sample across
the horizontal range of each rotation. With four rotations there is one slot
per rotation; with two there are two per rotation; O has four horizontal choices.
Equal `(holes, height, lines)` tuples are retained when the actual placements differ.
The fixed formula `10*lines - 8*holes - height` is used only by the benchmark baseline.
This samples a small part of the legal action space; it does not evaluate every placement.

One-option pieces need no query. While waiting, the piece has gravity but no chosen movement. On response,
first advance simulation to the real arrival time, then plan a legal path from
the current pose to the chosen landing. Movement is bounded by the controller
rate, with gravity still running. If unreachable, do not substitute the other
candidate. Query errors also leave gravity running without a fallback decision.

A monotonic piece ID prevents a result from affecting a later piece. Every
physics frame is accounted for on foreground stalls rather than slowing time;
stalls exceeding 1 second additionally invalidate benchmark comparisons.
Maximum one request is outstanding, request starts are at least one second
apart, and restart is disabled until the outstanding request settles.
There is no game-time extension or artificial timeout pause.

## Query v2 contract and budget

`tetris_decide_v2_query` receives a record with `piece` and `next` (0..6 in
I/O/T/S/Z/J/L order), `roughness` (0..180), and 2..4 `options`. Each option has
`holes: nat16` (0..200), `height: nat8` (0..20), and `lines: nat8` (0..4).
Features are measured after that candidate's line clear. Roughness describes
the current board: the sum of absolute height differences between adjacent columns.

The fixed prompt contains one `<<LABEL>>holesH heightY linesL` per option and
ends with `<<SEP>>Tetris P next N roughR`. CLS/SEP wrap the actual token input.
Two/three/four candidates use at most 25/32/39 tokens with the installed tokenizer.
The real-tokenizer test enumerates every feature tuple and every context combination.
The adapter caps input at 40 tokens and also applies the installed query cost guard.
The game checks `tetris_v2_status().max_tokens >= 39` before starting.

The current piece, NEXT and board roughness are extra context, not a lookahead
search. Full cells, individual column heights, hole locations, placement coordinates
and movement paths are **not** sent to the model. Four options with column-height
context measured 48 tokens; five with piece/NEXT measured 44. Those do not fit the
current 40-token query budget. This version keeps inference query-only.

The response contains a selected index (0..options.length-1), one softmax score
per option, model hash, token count, measured inference instructions and the prompt.
The UI displays A–D, the sent input, and separately the local target coordinates.
Scores are not calibrated correctness probabilities. More options do not establish
stronger play; the comparison below measures that separately.

The legacy `tetris_decide_query(vec Features)` and `tetris_status()` remain compatible
(exactly two options, 36-token adapter cap). Both adapters use the existing persisted
operator switch. No new public update method, quota state, caller string or weight upload
is introduced. The new frontend requires the v2 backend; there is no silent v1 fallback.

## Metrics

- DEADLINE MISS: a queried piece locked before an answer arrived.
- PLACEMENT MISS: an answered/single-option target became unreachable or actual
  lock pose differed from the target. This includes controller speed, not just AI.
- QUERY / SKIP: requests issued / multi-option pieces that locked without a
  request (e.g. an earlier request was still outstanding or throttling applied).
- AVG / MAX RTT: full network round trip for successful responses, including
  late responses. Errors have a separate counter.
- JSON records rules version, seed, model hash, target, score/level/lines,
  piece count, completion state, counters, and validity.

These are **end-to-end, candidate-assisted, client-reported** results, not pure
model throughput or consensus-certified scores. Compare the same seed, rules,
model, device, foreground/visibility conditions and network. Candidate generation,
the input-rate limit and query throttling all affect performance.

## Verification

`npm --prefix web/tetris test` includes deterministic clock tests for gravity,
level boundaries, scores, response deadlines, stale response isolation, collision
paths, browser stalls, visibility, errors, line clearing, and top-out.
`web/tetris/scripts/ui-check.js` uses controlled pending responses to prove
falling/locking continue during queries and that late replies are only displayed.
Run its function using the installed Playwright CLI's `run-code` argument.

`tools/tetris_v2_smoke.py` upgrades an existing disposable managed-local canister,
warms its saved pack, checks both API versions and rejects out-of-bound inputs.
It verifies the retained model/switch and owner-only shutdown without mainnet access.

```sh
cd web/tetris
IC_HOST=http://localhost:8011 IC_LOCAL=true CANISTER_ID=<local-id> npm run bench
```

This now compares `model`, `heuristic`, and `random` across the same ten seeds and
the same four-option sampling policy. The heuristic chooses the highest fixed score
inside that pool; it is not an unrestricted all-placement baseline. Random choices
use a separate seeded stream so policy choices do not change the piece sequence.
The benchmark uses the real-time controller with a virtual frame clock. Actual local
query RTT advances gravity (rounded up to one frame), including stale-response and
unreachable-target behavior. There is one RPC at a time and a one-second **simulation**
dispatch interval; fast wall-clock execution is loopback-only. Browser rendering and
device delays are excluded. This is not a mainnet or live-browser latency benchmark.
Baselines compute immediately, so their RTT is zero. Report both lines/pieces and misses.

`BENCH_SEEDS` and `BENCH_PIECES` default to 10 and 100. The measurement ends at top-out,
the piece limit, or one simulated hour; the latter two are marked as truncated and
do not change the actual game's unlimited-play rule. Results are written to
`artifacts/tetris_query_v2_benchmark.json`; the prior two-option benchmark is historical.


## Recorded local query v2 acceptance (2026-09-22)

Target: `4fbx2-kt777-77775-aaabq-cai`, `http://localhost:8011/`.
Only this existing disposable local target was upgraded, using its anonymous controller.
The existing owner `ic-verdict-acceptance-h1vkKw` authorized warm-up and switch checks;
owner configuration, packed weights and project canister mappings were retained.
Wasm SHA-256: `d928b2190bf53369a65657006536f5bcb4a8df70302e9be31dd5316aa9cd150b`;
verified unchanged by canister status after the comparison. The existing model
manifest hash remains `0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0`.
No mainnet deployment is represented by these local results.

Rust tests, the exhaustive tokenizer-bound test, frontend unit tests and production
build passed. Local upgrade/warm-up verified retained weights/switch, legacy API
compatibility, 2/3/4-score results, bounded inputs, owner-only administration and
continued protection of generic inference. A four-option boundary request used
39 tokens and 4,470,822,155 inference instructions (below the 5,000,000,000 limit).
Fifteen controlled Chromium checks covered four-score/input display, stale results,
continued gravity, single-flight operation, errors and a 390px viewport.
A separate real-browser smoke reached three placed pieces with real four-score
responses (39 tokens), query-only traffic, no page exceptions and no overflow at 390px.

Ten fixed seeds, all played to top-out (no truncated runs):

| Policy | Total lines | Total pieces | Deadline misses | Placement misses |
| --- | ---: | ---: | ---: | ---: |
| Model | 1 | 253 | 2 | 10 |
| Fixed rule within the same four options | 16 | 344 | 0 | 0 |
| Seeded random within the same four options | 0 | 244 | 0 | 0 |

The model made 253 queries with zero query errors; mean local RTT was 752.6 ms,
maximum 1,704.7 ms. These measurements may include cache/host load effects and
are not mainnet latency estimates. Greater freedom did **not** outperform the
fixed rule in this sample; this change is not evidence of stronger Tetris play.
The limited candidate pool also constrains both baselines.

An additional 11-case probe compared the accepted prompt against a 40-token
variant explicitly requesting line clears and hole avoidance. Agreement with the
fixed rule was 3/11 versus 2/11; that is not a correctness oracle, and the small probe
did not support adopting the longer wording. The accepted prompt remains 39 tokens.

Evidence: `artifacts/tetris_query_v2_smoke.json`, `artifacts/tetris_query_v2_benchmark.json`,
`artifacts/tetris_query_v2_prompt_probe.json`, `artifacts/tetris_query_v2_ui.json`,
and `artifacts/tetris_query_v2_live.json` (actual browser queries, separate from mocks).

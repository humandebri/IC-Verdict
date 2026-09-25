# Tetris input-only study

Large raw measurement JSON files mentioned below are local outputs. Git includes derived summaries with source hashes under [`artifacts/summaries/`](../artifacts/summaries/); see the [artifact policy](../artifacts/README.md). Full traces must be regenerated or obtained separately.

This study keeps the deployed 151M INT8 weights, five direct controls, 52-token
limit and 5B query budget. There is no training, action recommendation, forced
rotation, search fallback or production mutation. Public v3 remains unchanged.
A v4 adapter is conditional on passing the quality gates below.

## Reproduce

Use the existing model pack/tokenizer and dependencies; do not rebuild or upgrade
the canister to make the study pass.

```sh
qrun -- cargo build --release -p verdict-candle --bin tetris_probe
npm --prefix web/tetris run study:inputs
IC_HOST=http://localhost:8011 IC_LOCAL=true \
CANISTER_ID=4fbx2-kt777-77775-aaabq-cai \
STUDY_OWNER_PEM=/path/to/existing/local-owner.pem \
npm --prefix web/tetris run study:query
```

The second command provides token bounds and exploratory native inference. The
third is the authoritative evaluation: it checks the managed loopback endpoint,
frozen Wasm hash, model digest and 52-token readiness before using the existing
owner's protected `decide_query`. It makes queries only, with no installs, state
updates, registration, funding or model upload. The private key stays in memory.
The public API does not acquire an arbitrary-prompt endpoint.

Each command writes JSON evidence to `artifacts/tetris-input-study/`. Fixture,
model-manifest and tokenizer hashes bind the stages together. Query selection is
saved before held-out queries run. Native inference is diagnostic only: the
native binary and frozen Wasm can disagree on close scores, so its results must
not authorize a release. `--native-games` optionally runs native-only games;
`--resume-games` resumes that diagnostic stage after verifying provenance.

## Inputs and token bounds

All variants see the current piece, pivot, rotation and NEXT. Candidates retain
stable IDs 0 left, 1 right, 2 clockwise, 3 down, 4 wait, with the current legal
mask. No variant receives the test oracle, a future board or an action score.

| State | Short labels | Explicit labels | Eligible |
| --- | ---: | ---: | --- |
| Heights, total holes, previous pose/action/outcome, two older actions | 52 | 57 | Short only |
| Heights and total holes, no history | 40 | 45 | Both |
| Heights and per-column hole counts, no history | 47 | 52 | Both |
| Twenty row occupancy masks, no history | 66 | 71 | Neither |

Short labels are `left/right/rotate/down/wait`; explicit labels are
`move left/move right/rotate clockwise/move down/wait`. Rows run top to bottom,
with the leftmost cell as bit 0. Numeric bounds include height 0..20, column holes
0..19, total holes 0..200, row masks 0..1023 and all piece/pose/history values.

The real tokenizer is BPE with the ByteLevel regex and NFC normalization (identity on these ASCII prompts). The bound
enumerates every value of each variable segment at fixed pretoken boundaries,
sums the maximum lengths, and checks concatenated token IDs against whole-prompt
encoding for each substitution into the maximum witness. All five labels and
full history bound their subsets. This is not a maximum inferred from sampled
games. Over-budget formats are excluded rather than truncated or cropped.
The full-board encoding fitting one 47-token example did **not** establish a
52-token bound: its domain maximum is 66.

## Fixtures and interpretation

There are 50 development and 50 held-out states, each with ten rotation, left,
right, aligned-drop and collision-constrained examples. Every split covers all
six rotatable pieces for rotation and all seven pieces for the other categories.
Development cavities have pivot x <= 4; held-out cavities have pivot x >= 5.
Whole board occupancy is disjoint across the splits, and states are unique.
Fixtures and expected actions are fixed before inference.

These are synthetic, near-stack, one-control tactics: apply one legal control,
then allow gravity to lock. The test-only oracle prioritizes cleared lines, then
fewer holes, then lower height; ties are accepted. An aligned state may accept
both down and wait. A blocked down locks the current pose, as in the actual game.
"Rotation required" means required to solve this one-control tactic, not a proof
that every longer legal sequence requires that exact first control. This suite
is a diagnostic rather than a comprehensive distribution of real games.

For history ablation, all fixtures have a valid earlier pose one row above and
previous waits. This intentionally tests the observed repetition failure; it is
not a representative distribution of previous actions. Real-game comparisons
supply actual per-piece history. An initial I-only generator run is retained in
`preliminary-i-only/`, explicitly excluded after a coverage audit. The formal
suite was selected for piece coverage, not for model scores.

## Evaluation and release gate

Select the alternative with the highest development accuracy; break ties by the
proven domain token bound, then variant ID. Only the selected alternative and
current `history-short` are evaluated on the held-out set. Record rotation
availability/selections, required-rotation accuracy and repeated actions.

Run seeds 1..10 for both policies using the real frozen canister's choices but
charge exactly 1000 ms before each control to isolate decision quality. Natural
gravity, one pending query per game, the one-second dispatch gap and late-answer
rejection are unchanged. Up to four independent games run concurrently. Identical
prompts may share cached answers in this fixed-latency phase; this does not claim
measured RTT or total instruction consumption. Each game is capped at 100 pieces
or one simulated hour, with truncation reported. An additional seed 42 run per
policy charges actual local query RTT, sequentially, with the study client cache
disabled. Replica-side query caching may still occur; this is not a cold-cache
latency measurement or a mainnet latency estimate.

All gates must pass: held-out accuracy >= 80%; required-rotation accuracy >= 80%;
at least five of ten fixed-latency games clear lines; mean lines >= 1; mean pieces
survived >= current policy; query errors and budget violations zero. Random
rotation or a change in rotation frequency alone is not success. The next step
on failure is a report of the limitation, not training, a different model or an
unapproved production rollout.

Results are recorded in `query-result.json`; development, held-out decisions,
complete game traces, actual-latency games, domain-boundary query measurements
and native/Wasm differences are kept in adjacent JSON files.

## Measured results (2026-09-23 JST)

Authoritative local-query development results:

| Variant | Correct / 50 | Required rotations correct / 10 |
| --- | ---: | ---: |
| history-short | 18 | 0 |
| no-history-short | 12 | 0 |
| no-history-explicit | 12 | 1 |
| columns-short | 13 | 0 |
| columns-explicit | 20 | 0 |

The selected alternative is `columns-explicit`. On the untouched held-out set,
current v3 scored 14/50 and the alternative 16/50; both scored
0/10 on required rotation. These fail the predeclared 80% gates.

Native and frozen-Wasm decisions differed in 1/250 development queries.
All final accuracy numbers above use actual canister responses. The earlier
partial native game runs and aborted parity spot-check are exploratory evidence,
not release-gate results. The query study independently rechecks all domain
boundaries and all 250 development decisions.

Ten-seed fixed-1000-ms local-query comparison (all games reached top-out):

| Variant | Lines | Pieces | Decisions | Rotation offered / chosen |
| --- | ---: | ---: | ---: | ---: |
| history-short | 0 | 133 | 1206 | 1066 / 0 |
| columns-explicit | 0 | 132 | 1073 | 913 / 0 |

The alternative chose down 229 times versus 20 for current v3, but neither
policy ever selected rotation or cleared a line. Mean survival was 13.2 versus
13.3 pieces. This fails the gameplay gates as well as the held-out accuracy
gates. No v4 adapter or public UI change is justified by these results.

Sequential seed-42 runs with actual local RTT also cleared zero lines: current
v3 survived 16 pieces and the alternative 15; neither run was truncated.
Across the authoritative study, 2900 actual queries completed with
zero errors; the maximum input was 52 tokens and the maximum
reported forward instruction count was 4,757,068,579, below 5B.

An independent audit checked all 2,279 fixed-latency decisions: legal action IDs,
score argmax, previous-input/action/outcome linkage, older executed history,
per-piece reset, execution/late counters and token/instruction bounds all passed.
See `verification.json`. Four study regression tests plus the existing forty
frontend tests and the production build pass. The native probe builds in release
mode. Public frontend asset hashes remain identical to the deployed v3 release.

**Outcome: quality gate failed.** No v4 endpoint, canister upgrade, public UI
change, model training or production rollout was performed. Input wording and
column statistics changed action preferences but did not produce competent
rotation or line clearing under the agreed constraints. These experiments do
not prove that every possible prompt would fail; they establish that the tested
input-only alternatives do not justify publication.

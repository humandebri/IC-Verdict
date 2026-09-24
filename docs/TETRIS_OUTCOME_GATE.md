# Tetris outcome comparison prerequisite

The approved plan requires this gate to pass before implementing time-aware
landing search, a v4 API, or UI integration. The existing model, weights and query
implementation remain fixed. This gate failed; those later phases were not run.
Production was not changed.

## Protocol

`tools/tetris_outcome_gate.py` freezes 50 paired comparisons (100 questions) before
inference. Each question has five actions: left, right, rotate, down and wait.
Exactly one outcome is better: more cleared lines, fewer holes, or lower height;
all other features are equal. The paired question swaps the better outcome to a
different action. Each action is correct in exactly 20 questions. There are 34
line, 34 hole and 32 height questions.

The state is always `Tetris. Choose next move.` Candidate labels look like
`left lines1 holes0 height12`. Only the state and candidate labels are sent;
expected answers and fixture metadata are not. There was no prompt tuning after
observing the results. This tests this particular outcome representation, not all
possible ways of prompting the model.

Acceptance requires at least 90/100 correct, at least 16/20 for every action,
and both answers correct in at least 40/50 swapped pairs.

## Local canister result — 2026-09-23

| Measure | Observed | Required |
| --- | ---: | ---: |
| Overall correct | 3/100 | 90/100 |
| Left correct | 0/20 | 16/20 |
| Right correct | 0/20 | 16/20 |
| Rotate correct | 2/20 | 16/20 |
| Down correct | 0/20 | 16/20 |
| Wait correct | 1/20 | 16/20 |
| Both swapped answers correct | 0/50 | 40/50 |

Selections were left 0, right 3, rotate 28, down 0 and wait 69. The query path
worked, but this representation did not cause choices to follow the preferred
outcome reliably. The experiment therefore stops before search, v4, UI and game
comparisons; it does not establish that another representation must also fail.

All 101 actual local canister queries (one boundary query plus 100 questions)
completed with zero query errors. Maximum observed input was 51 tokens and
maximum instructions were 4,747,040,636, within the 52-token/5-billion limits.
The tokenizer boundary proof performed 1,398 checks, including all nonempty
combinations of omitted, safe and game-over actions. The tokenizer uses NFC
normalization, which leaves these ASCII prompts unchanged, and ByteLevel regex
pretokenization. Native code was used only for tokenization, not inference.

## Reproduction and evidence

With the existing frozen local canister running, its registered owner identity,
and `target/release/tetris_probe` built:

```sh
python3 tools/tetris_outcome_gate.py --canister 4fbx2-kt777-77775-aaabq-cai --owner ic-verdict-acceptance-h1vkKw
python3 -m unittest discover -s tools -p test_tetris_outcome_gate.py
```

The runner verifies the loopback network, frozen WASM and model hashes, warmed
model, token count, instruction budget, probabilities and returned choice. It
uses the existing `icp` signer and makes no update, install or deployment calls.
A completed failed quality gate is recorded as `passed: false`; command failure
is reserved for protocol or execution failures.

Evidence is in `artifacts/tetris-outcome-gate/`: `fixtures.json` contains the
frozen cases, `responses.json` the actual query responses, `boundary-query.json`
the boundary query, `token-bound.json` the token proof, and `result.json` the
aggregate result and provenance hashes. Four unit tests passed; independent
recomputation from the 100 responses matched the recorded aggregate. The Tetris
web build also passed after correcting the prior input-study tokenizer guard
from no normalization to NFC.

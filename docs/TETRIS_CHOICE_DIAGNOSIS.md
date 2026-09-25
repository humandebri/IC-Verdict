# Four-choice Tetris diagnosis

No production changes or canister calls. The local native INT8 probe uses the deployed model manifest `0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0` and exact current prompt. All four saved native/Wasm selection cross-checks matched. This is a small parity check, not a proof of equivalence for every input.

## Findings

| Check | Result |
| --- | --- |
| Uniform candidate sampling misses every maximum-score legal placement | 147 / 247 boards (59.5%) |
| At least two offered placements have identical model-visible features | 62 / 247 boards (25.1%) |
| Same placement selected across all 24 permutations | 0 / 8 real-board cases |
| Heuristic-best placement selected on real-board permutations | 81 / 192 (42.2%) |
| Strictly dominated placement selected on real-board permutations | 91 / 192 (47.4%) |
| Unique dominating candidate selected on synthetic permutations | 40 / 144 (27.8%) |

Real-case majority consistency averaged 79.2%: the model usually preferred one candidate, but all eight cases changed their selection under some ordering. This is not evidence that the model always chooses a fixed slot.

The corpus uses the production Rust legal-placement generator and four-candidate sampler, four seeds, and two deterministic policies (full-set heuristic and four-candidate heuristic), up to 32 turns each. Nine four-candidate-policy turns were unavailable after top-out, leaving 247 boards. The proxy score is `10*lines - 8*holes - height`, not a proven optimal Tetris move. Full-set and sampled-set policies lose the best score on 80/128 and 67/119 boards respectively.

Eight real cases were selected at evenly spaced indices among corpus boards whose four offered candidates have distinct feature triples, before inference. Each was tested under all 24 permutations. Six synthetic feature-only cases were also tested under all 24 permutations; these tuples need not correspond to simultaneous reachable placements. The 192 and 144 counts are repeated permutations, not independent questions or a general accuracy benchmark. A random four-way selector would have 25% expected success on the unique-winner synthetic tests; no statistical significance claim is made.

Concrete failures: with holes and height equal, the model chose zero cleared lines over four in 24/24 orderings. With lines and holes equal, it chose height 7 in 20/24 orderings instead of height 2 (correct in 2/24). It selected zero holes over positive holes in 24/24, but selected two holes over 3/7/15 holes only in 4/24. The current compact feature format therefore does not reliably induce relative numerical preferences.

Both stages contribute: uniform sampling removes useful options, and the model still makes dominated selections when a better option is present. Query latency is not the explanation; the offline test reproduces the input and uses the same weights. The study does not isolate whether different wording, task-specific training, or a different checkpoint can fix model selection. Merely increasing candidate count or keeping the best heuristic move does not establish a model contribution.

Recommendation: retain the demo's honest attribution; do not claim improved play. Use these cases as a regression gate before changing prompts or candidate generation. Any heuristic-based shortlist must be disclosed as code assistance and evaluated separately from model selection. Keep the deployed UI/model unchanged until a proposed change measurably improves both dominant-choice selection and order stability.

## Reproduction

`cargo test -p verdict-engine export_choice_diagnostic_corpus --lib -- --ignored --nocapture` writes the exact Rust corpus. Then `python3 tools/tetris_choice_diagnosis.py` runs the existing local probe. Checkpoints and probe must exist locally; nothing is downloaded.

Evidence: `artifacts/tetris-choice-diagnosis/{protocol,parity,corpus,permutations,inferences,candidate-losses,result}.json`. The protocol pins checkpoint, tokenizer, probe and corpus hashes. All 338 unique inference requests used at most 43 tokens. No runtime configuration, model weights, frontend, or production deployment was changed.

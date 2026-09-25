# Tetris prompt and candidate search

The selected implementation uses diverse candidates and independent relative costs.
It was validated on the existing local canister and subsequently deployed to production on 2026-09-23. See `production-release.json` and `production-query-check.json` in the artifact directory for release verification.

## Results

The final eight seeds (277, 281, 283, 293, 307, 311, 313, 317) were frozen before the selected cost format's game trials. Each game uses the unchanged Rust physics, seeded seven-bag, and 128-turn cap.

| Policy | Mean pieces placed | Mean lines | Games clearing a line |
|---|---:|---:|---:|
| Previous uniform candidates + numeric model prompt | 26.875 | 0 | 0/8 |
| Diverse candidates + numeric model prompt | 20.125 | 0.125 | 1/8 |
| Diverse candidates + selected model prompt | 45.75 | 6.5 | 7/8 |
| Same diverse candidates + heuristic | 39.625 | 5.0 | 8/8 |
| Same diverse candidates + random choice | 19.875 | 0 | 0/8 |

These eight games demonstrate improvement over the previous implementation, not general superiority over a heuristic. One selected-model game still cleared no lines. Development games are reported separately: eight seeds averaged 9.5 lines and 51.625 pieces.

## What changed

Code enumerates reachable placements, retaining the first placement for each distinct `(lines, holes, height)` tuple. Starting from the first tuple, it adds the candidate farthest from its nearest selected tuple by unweighted L1 feature distance, up to four. Distance ties use the last candidate in the existing rotation/x/y order. This selects different outcomes, not a combined quality score; identical-feature geometry alternatives are lost, which remains a limitation.

The prompt is `Min holes`. A candidate label is, for example, `min holes,mid height,max missed`. Code computes `missed = max(offered cleared lines) - candidate cleared lines`, then independently maps holes, height and missed clears to `min`, `mid`, `max` or `equal`. Lower is better for every field. Intermediate magnitudes are intentionally discarded. The model chooses the tradeoff; no heuristic score, best-placement ID, or aggregate ranking enters its request.

The heuristic remains `10*lines - 8*holes - height` on the same offered candidates. It is used for the visible comparison and the explicitly selected model-free mode. Model errors do not fall back to the heuristic. Legacy update APIs and stored receipts retain their old prompt. The new query receipt rules end in `query-feature-cost-v1`.

## Search and rejected alternatives

114 configurations were considered; 75 passed the relevant token screening and were inferred, across numeric labels, number words, units, relative words, relative ranks, feature order, common-field omission, punctuation removal, and finally uniform-direction costs. There were also small exploratory vocabulary probes. This is a broad search, not an exhaustive claim about all possible prompts.

Early rank prompts improved feature comparisons but did not improve actual play with uniform candidate sampling. Candidate sampling was therefore measured separately over 32 heuristic games. Uniform sampling averaged 0.594 lines; feature-diverse sampling averaged 6.438, versus 31.125 when the heuristic could access every placement.

A holes-first rank prompt then improved real games, but failed every pure line-clear comparison: its fields mixed minimizing holes/height with maximizing clears. Explicit mixed-direction wording did not fix that probe. The selected representation turns line clears into a per-feature shortfall, making every direction consistent. This is code assistance and is disclosed in the English UI.

The selected format passes all 72 single-feature/order checks (three features × all 24 candidate permutations). These collapse to 36 unique model requests, so they are not 72 independent accuracy samples. They test the normalized features, not understanding of a raw Tetris board or general arithmetic. Earlier holdouts that informed later changes are not presented as final held-out evidence.

Common-field omission and punctuation removal were tested before selection. Their game performance failed the predeclared compression gate. The final cost format is nevertheless smaller: **at most 41 tokens**, versus 43 previously; mean input across the 366 fresh model turns was 40.51 tokens. The tokenizer check covers all 64 relative-cost labels at candidate counts 1–4.

## Local canister and UI verification

The unchanged INT8 151M checkpoint was used throughout. Final local canister games for seeds 11 and 277 placed 39/31 pieces and cleared 4/1 lines. All **70 choices**, candidate lists, and input boards match native inference; after-boards and features were independently recomputed. Each turn made one query, zero update calls, and left server game/budget counters unchanged. Maximum model instructions were **3,686,936,071**, below 5 billion; maximum input was 41 tokens. These measurements are local, not mainnet latency guarantees.

Rust Tetris tests, the Wasm build, and the frontend build passed. Playwright Chromium verified single-click autoplay and Pause, the English assistance disclosure, `1 model pass / 1 query`, no console errors, and no horizontal overflow at 390px. The model reached 22 turns / 4 lines in that UI check.

## Evidence and reproduction

`artifacts/tetris-prompt-search/summary.json` is the final aggregate with model/tokenizer/Wasm hashes. The directory also contains frozen protocols, raw inferences, feature comparisons, complete game traces, local responses, and UI evidence. Earlier `result.json` and `final-selection.json` are intermediate rank-format results; `cost-selection.json` and `summary.json` identify the final implementation.

```sh
python3 tools/tetris_prompt_search.py
python3 tools/tetris_rank_search.py
python3 tools/tetris_semantic_search.py
python3 tools/tetris_short_rank_search.py
python3 tools/tetris_cost_prompt.py
TETRIS_SAMPLER=diverse TETRIS_RUN_TAG=cost-fresh TETRIS_SEEDS=277,281,283,293,307,311,313,317 TETRIS_POLICIES=cost/0,baseline,four-heuristic,random qrun -- cargo test -p verdict-engine compare_prompt_rollouts --lib -- --ignored --nocapture
TETRIS_SAMPLER=uniform TETRIS_RUN_TAG=cost-original TETRIS_SEEDS=277,281,283,293,307,311,313,317 TETRIS_POLICIES=baseline qrun -- cargo test -p verdict-engine compare_prompt_rollouts --lib -- --ignored --nocapture
# Against the already upgraded, warmed local canister only:
node --import ./web/tetris/node_modules/tsx/dist/loader.mjs web/tetris/scripts/feature-ranks-smoke.ts
python3 tools/summarize_tetris_prompt_search.py
```

The search did not retrain the checkpoint, fund canisters, or change certification. The user subsequently authorized production deployment; its evidence is recorded separately. Query logs remain un-certified response logs, not consensus-persisted decisions.

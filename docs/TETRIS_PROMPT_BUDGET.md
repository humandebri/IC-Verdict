# Token budget and explicit numerical preferences

Offline study of the unchanged 151M INT8 checkpoint, using the real tokenizer and existing native probe. No canister calls, training, or deployment. The target is to reduce input tokens while expressing the direction of each feature preference; reduced length alone is not sufficient to adopt a prompt.

| Variant | Context | Candidate representation | Four-candidate tokens |
| --- | --- | --- | --- |
| Baseline | `Tetris. Clear lines, avoid holes and height.` | `lines4 holes199 height20` | 43 |
| Explicit | `More lines, fewer holes, lower height` | `lines4 holes199 height20` | 39 |
| Compact | `Max lines,min holes,height` | `lines4 holes199 height20` | 38 |
| Tuple | `lines/holes/height: max/min/min` | `4/199/20` | 38 |

Counts include all four candidate labels, model delimiters and CLS/SEP tokens. The token scan varies lines 0–4, holes 0–200 and height 0–20 one field at a time; counts were constant for each representation. This is a measured scan, not a claim that character count equals token count.

Development uses the previous eight real-board cases and six feature-dominance cases, each under four cyclic orders. A shorter candidate must improve dominance selection over baseline to proceed. Among those, choose greatest dominance success, then fewer dominated real-case choices, then fewer tokens. This selected Explicit (39 tokens). Compact and Tuple did not improve dominance selection in development.

Confirmation uses six additional real-board feature sets excluded from development and six new deterministic synthetic dominance cases (seed 9301), under all 24 orders, comparing only baseline and the selected prompt. Adoption requires better dominance success and no increase in dominated real-case choices. Repeated permutations are not independent samples; heuristic-best score is not proven optimal Tetris play. Final results are in `artifacts/tetris-prompt-budget/result.json`.

Reproduce with `python3 tools/tetris_prompt_budget.py` after generating the corpus documented in `TETRIS_CHOICE_DIAGNOSIS.md`. Protocol, token bounds, individual selections and raw inference outputs are retained under `artifacts/tetris-prompt-budget/`.

## Result

No prompt was adopted. The selected Explicit variant reduced the full input from 43 to 39 tokens (9.3%), but held-out dominance success was 19/144 versus baseline 23/144. Real-case heuristic-best selection was 53/144 versus 56/144, and dominated selections were 74/144 versus 73/144. Both prompts were order-unstable on all six real cases. These small, correlated tests do not establish a statistically significant regression; they fail to provide evidence for adopting the new prompt.

The 38-token variants also failed the development gate. In total, 800 unique local inferences were recorded. The deployed prompt, model, UI and canister remain unchanged. Explicit preference wording alone did not solve numerical selection on these tests.

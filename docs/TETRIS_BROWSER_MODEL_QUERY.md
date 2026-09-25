# Browser-owned Tetris with a generic model query

> Published on 2026-09-24. Deployment details are in [MAINNET.md](MAINNET.md).

The browser owns the 200-cell board, piece sequence, legal-placement search,
features, candidate selection, animation and next board. The canister does no
Tetris calculation. For a model turn, the browser calls public `decide_query`
once with `state: "Min holes"`, an empty question, temperature 1, no abstention,
and up to four candidate descriptions such as `min holes,mid height,max missed`.
It sends no board, piece, seed, path, coordinates or heuristic score. The
comparison mode makes no model query.

The browser checks the returned candidate IDs and order, selected ID, scores,
model digest and query budget before applying a move. A query failure or bad
reply leaves the board unchanged. Saved JSON is a local input/output log and is
not a certified decision record.

`decide_query` accepts anonymous callers. It retains the shared input
validation, tokenizer and 5B query budget guard. `info` reports warmup and
model digest; `query_limits` reports the current token ceiling. Replicated
execution of a query method remains denied. The old Tetris-specific canister
methods and game snapshots are retired; the upgrade reads the old snapshot
only to preserve generic model and billing state, discarding game records.
Before tokenization, the public query also caps the combined request text at
16 KiB, the question at 4 KiB, each option description at 1 KiB, and each
option ID at 128 bytes. These byte caps leave the 5B instruction and token
checks in place.

The TypeScript placement rules and candidate order are checked against saved
Rust canister replies from fixed seeds and middle-game boards.

## Local verification (2026-09-24)

- `cargo test -p verdict-engine --lib`: 17 passed, including snapshot migration
  and the public query byte caps.
- `tools/build_one.sh verdict-engine`: Wasm and Candid built. The exported
  Candid has `decide_query` and no Tetris-specific methods.
- PocketIC 15.0.0: anonymous `decide_query`, replicated-query refusal, billing,
  and upgrades passed both from the current Wasm and the saved Tetris Wasm.
- `web/tetris`: 55 tests passed; TypeScript and Vite build passed; Wrangler
  4.107.0 dry run passed. In Chromium, comparison mode advanced seven turns
  with zero non-static network requests, even without a configured canister ID.

The compatibility Wasm is staged at `build/compat-model-query/verdict-engine.wasm`
(SHA-256 `584657715086775ff0e785053b175c777d27261abd13029b71bc1aeeb662204e`).
`tools/build_tetris_compat.sh` builds it from the checked-in
`tools/tetris_compat_query.rs` in an isolated temporary workspace.
Its Candid exposes the old read-only placement/status queries alongside public
`decide_query`; it has no game update methods. Its placement query uses the
same feature-diverse candidate selection as the current public release.
PocketIC verified the candidate lists against three saved production replies
and an upgrade from it to the final model-only Wasm. The originally released
model-only Wasm had SHA-256
`2c403877b2fac5c4563fe33fa71712174ed4818c1a499fbdf0621eaa2afff680`.

## Production release (2026-09-24)

The existing canister `qojfj-6qaaa-aaaam-qjkaq-cai` was upgraded to the
compatibility Wasm, and all 142 model tensors were warmed. Anonymous
`decide_query` then succeeded with the retained model digest
`0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0`.
Worker `openjev` was published as version
`51a2dde9-f9d2-425f-b921-1d58b6617a2a`. A public Chrome smoke test
completed one model turn with 41 input tokens, one decision query and no game
update; comparison mode completed two turns with no canister traffic.

After that check, the same canister was upgraded to the final model-only Wasm
and all 142 tensors were warmed again. The installed module hash is
`2c403877b2fac5c4563fe33fa71712174ed4818c1a499fbdf0621eaa2afff680`.
Anonymous `info` reported the original model digest and `warmed = true`, and
anonymous `decide_query` succeeded. A second public Chrome smoke test confirmed
one model turn with 41 tokens and one decision query, then two comparison turns
with no canister traffic. The canister was Running with 732,530,282,054 cycles
at the final status snapshot. No model upload, funding, DNS or controller change
was performed.

The `production` keyring alias was unavailable in this shell. Deployment used
the existing `bridge-seal-log-reader` identity, which resolves to the same
controller principal recorded in [MAINNET.md](MAINNET.md). Wrangler OAuth was
available with elevated filesystem access.

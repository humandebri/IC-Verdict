# Browser-owned Tetris with a generic model query

> Implemented in the workspace; production release is pending. The currently
> deployed behavior is recorded in [MAINNET.md](MAINNET.md).

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
and an upgrade from it to the final model-only Wasm. The final Wasm is
`build/verdict-engine.wasm` (SHA-256
`2c403877b2fac5c4563fe33fa71712174ed4818c1a499fbdf0621eaa2afff680`).

Production is still on the old canister and Web release. Publication needs the
`production` identity in macOS Keychain and a Cloudflare Wrangler login on this
host. After access is restored, publish the compatibility Wasm, warm it, switch
the Web, then publish the final model-only Wasm and warm it again.

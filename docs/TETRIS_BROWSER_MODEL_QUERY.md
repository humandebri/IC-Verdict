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

The TypeScript placement rules and candidate order are checked against saved
Rust canister replies from fixed seeds and middle-game boards.

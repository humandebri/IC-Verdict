# Execution limitations

`cargo`, `rustc`, `rustup`, and `dfx` were absent from the authoring runtime.
Attempts to obtain a Rust toolchain failed because external downloads/DNS were unavailable.
No Cargo dependency resolution or native/Wasm build was completed, and no Cargo.lock was fabricated.

The delivered CI workflow has not been run remotely. Python reference tests and
synthetic PyTorch/NumPy calculations are the only executable behavioral tests run here.
See verification.json and the raw logs for their exact scope and environment versions.

//! IC entropy backend for `getrandom` 0.3 on `wasm32-unknown-unknown`.
//!
//! Why this exists: `getrandom` 0.3 has no OS source on this target and fails to
//! compile unless the `wasm_js` feature (wasm-bindgen/js-sys, unusable inside a
//! canister) or a custom backend is selected. It reaches our wasm graph
//! transitively and non-optionally:
//!
//! ```text
//! candle-core -> rand  -> rand_core -> getrandom
//! tokenizers  -> ahash                -> getrandom
//! ```
//!
//! `.cargo/config.toml` sets `--cfg getrandom_backend="custom"` for the wasm32
//! target, so `getrandom` calls the `__getrandom_v03_custom` symbol defined below.
//!
//! ## Security scope
//!
//! The stream is ChaCha20 keyed by SHA-256 over the message's IC environment:
//! the running canister, the caller, `ic0.time`, and a counter. Every byte of
//! every field contributes (see `seed`).
//!
//! It is **not** a substitute for `raw_rand`. Two properties limit it:
//!
//! 1. `ic0.time` is validator-influenced, so an adversary who can steer the
//!    round's timestamp knows part of the input.
//! 2. `STATE` and `COUNTER` are thread-locals, and the IC reconstitutes the Wasm
//!    instance for every message. Both therefore reset per message, so the
//!    counter does not distinguish two `getrandom` calls that land in the same
//!    nanosecond for the same caller and canister. Two calls in that situation
//!    derive the same key and would produce the same stream.
//!
//! Those are acceptable only because of what this module actually serves:
//! candle's `Tensor::rand` initialization and `ahash` hash-map seeding. Neither
//! is a security boundary here — no `HashMap`/`HashSet` exists anywhere in the
//! workspace (everything is `BTreeMap`), so an ahash seed is not observable, and
//! this canister keeps no secrets and signs nothing.
//!
//! **Do not use this for keys, nonces, salting, or any security token.** If such
//! a need appears, replace it with the management canister's `raw_rand`, which is
//! asynchronous and therefore cannot be reached through this synchronous
//! `getrandom` interface.
//!
//! The PRNG runs on plain arithmetic and does not call `getrandom`, so there is no
//! recursion into this backend.

use core::cell::RefCell;
use core::mem::MaybeUninit;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

thread_local! {
    static STATE: RefCell<Option<ChaCha20Rng>> = const { RefCell::new(None) };
    static COUNTER: RefCell<u64> = const { RefCell::new(0) };
}

/// Derives the ChaCha20 key from the per-message IC environment.
///
/// Every field is length-framed before hashing. Writing the principals at fixed
/// offsets instead would silently discard most of their bytes: a principal is up
/// to 29 bytes, so a 32-byte buffer split four ways keeps only the first 8 bytes
/// of each field and the slots still look full.
fn seed() -> [u8; 32] {
    let counter = COUNTER.with(|c| {
        let mut c = c.borrow_mut();
        *c = c.wrapping_add(1);
        *c
    });
    let mut hasher = Sha256::new();
    hasher.update(b"ic-laya/getrandom-seed/v1");
    for field in [
        ic_cdk::api::canister_self().as_slice(),
        ic_cdk::api::msg_caller().as_slice(),
        &ic_cdk::api::time().to_be_bytes(),
        &counter.to_be_bytes(),
    ] {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}

/// Called by `getrandom`'s `custom` backend. The signature must match
/// `getrandom-0.3.4/src/backends/custom.rs` exactly; a mismatch fails to link.
/// `getrandom::Error` wraps a `NonZeroI32`, so success is `Ok(())` — there is no
/// "no error" constant to return.
///
/// # Safety
/// `dest` must be valid for writes of `len` bytes, per the getrandom backend
/// contract.
#[allow(unsafe_op_in_unsafe_fn)]
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom::Error> {
    if dest.is_null() && len != 0 {
        return Err(getrandom::Error::UNEXPECTED);
    }
    if len == 0 {
        return Ok(());
    }
    // SAFETY: the caller guarantees `dest` is valid for `len` bytes, and the
    // resulting slice is fully overwritten below.
    let out = unsafe { core::slice::from_raw_parts_mut(dest as *mut MaybeUninit<u8>, len) };
    // `try_borrow_mut`, not `borrow_mut`: a re-entrant call would otherwise panic
    // on the borrow and trap the whole canister update. Returning an error keeps
    // the failure local to the caller that asked for random bytes.
    STATE.with(|cell| {
        let mut slot = match cell.try_borrow_mut() {
            Ok(slot) => slot,
            Err(_) => return Err(getrandom::Error::UNEXPECTED),
        };
        let rng = slot.get_or_insert_with(|| ChaCha20Rng::from_seed(seed()));
        let mut written = 0usize;
        while written < len {
            let word = rng.next_u64().to_le_bytes();
            let take = word.len().min(len - written);
            for (i, byte) in word[..take].iter().enumerate() {
                out[written + i].write(*byte);
            }
            written += take;
        }
        Ok(())
    })
}

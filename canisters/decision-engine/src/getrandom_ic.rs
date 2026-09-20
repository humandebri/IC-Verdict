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
//! The stream is ChaCha20 keyed from `ic0.time`, the caller, the running canister,
//! and a monotonic counter. That is unpredictable to an off-chain observer but is
//! **not** a substitute for `raw_rand`: `time` is validator-influenced and the
//! seed is reproducible in principle by anyone who learns those inputs. It is
//! therefore suitable only for what this module actually serves — candle's
//! `Tensor::rand` initialization and `ahash` hash-map seeding. It must not be used
//! to generate keys, nonces, or any security token; no such call exists in this
//! canister, which keeps no secrets and signs nothing.
//!
//! The PRNG runs on plain arithmetic and does not call `getrandom`, so there is no
//! recursion into this backend.

use core::cell::RefCell;
use core::mem::MaybeUninit;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

thread_local! {
    static STATE: RefCell<Option<ChaCha20Rng>> = const { RefCell::new(None) };
    static COUNTER: RefCell<u64> = const { RefCell::new(0) };
}

fn seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    let canister = ic_cdk::api::canister_self();
    let caller = ic_cdk::api::msg_caller();
    let time = ic_cdk::api::time();
    let spin = COUNTER.with(|c| {
        let mut c = c.borrow_mut();
        *c = c.wrapping_add(1);
        *c
    });
    let mut put = |offset: usize, bytes: &[u8]| {
        let end = (offset + bytes.len()).min(seed.len());
        if offset < end {
            seed[offset..end].copy_from_slice(&bytes[..end - offset]);
        }
    };
    put(0, canister.as_slice());
    put(8, caller.as_slice());
    put(16, &time.to_le_bytes());
    put(24, &spin.to_le_bytes());
    seed
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
    STATE.with(|cell| {
        let mut slot = cell.borrow_mut();
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
    });
    Ok(())
}

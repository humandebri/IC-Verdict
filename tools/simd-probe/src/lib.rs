//! Which `core::arch::wasm32` intrinsics survive into v128 instructions here?
//!
//! Independent of the workspace and of candle: four exported kernels that differ
//! only in how the inner broadcast is expressed. Build and count the SIMD
//! instructions per exported function:
//!
//!   cargo build --release --target wasm32-unknown-unknown --manifest-path tools/simd-probe/Cargo.toml
//!   wasm-tools print tools/simd-probe/target/wasm32-unknown-unknown/release/simd_probe.wasm
//!
//! Each function is exported (`#[no_mangle]`), so nothing is dead-code eliminated.
#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn probe_vector_loads(a: *const f32, b: *const f32, out: *mut f32) {
    use core::arch::wasm32::*;
    unsafe {
        let mut acc = v128_load(a.cast::<v128>());
        for i in 1..4 {
            let av = v128_load(a.add(i * 4).cast::<v128>());
            let bv = v128_load(b.add(i * 4).cast::<v128>());
            acc = f32x4_add(acc, f32x4_mul(av, bv));
        }
        v128_store(out.cast::<v128>(), acc);
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn probe_splat_f32x4(a: *const f32, b: *const f32, out: *mut f32) {
    use core::arch::wasm32::*;
    unsafe {
        let mut acc = f32x4_splat(0.0);
        for i in 0..4 {
            let av = f32x4_splat(*a.add(i));
            let bv = v128_load(b.add(i * 4).cast::<v128>());
            acc = f32x4_add(acc, f32x4_mul(av, bv));
        }
        v128_store(out.cast::<v128>(), acc);
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn probe_load32_splat(a: *const f32, b: *const f32, out: *mut f32) {
    use core::arch::wasm32::*;
    unsafe {
        let mut acc = f32x4_splat(0.0);
        for i in 0..4 {
            let av = v128_load32_splat(a.add(i).cast::<u32>());
            let bv = v128_load(b.add(i * 4).cast::<v128>());
            acc = f32x4_add(acc, f32x4_mul(av, bv));
        }
        v128_store(out.cast::<v128>(), acc);
    }
}

/// Same as `probe_splat_f32x4` but with no `#[target_feature]` attribute, to test
/// whether the attribute is what makes the difference.
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn probe_splat_no_attribute(a: *const f32, b: *const f32, out: *mut f32) {
    use core::arch::wasm32::*;
    unsafe {
        let mut acc = f32x4_splat(0.0);
        for i in 0..4 {
            let av = f32x4_splat(*a.add(i));
            let bv = v128_load(b.add(i * 4).cast::<v128>());
            acc = f32x4_add(acc, f32x4_mul(av, bv));
        }
        v128_store(out.cast::<v128>(), acc);
    }
}

// --- plain-loop route -----------------------------------------------------------
//
// `core::arch::global_asm!` is not stable on wasm32 (E0658), so a hand-written wasm
// kernel needs nightly or a precompiled object. This variant needs neither: it is
// ordinary safe Rust, and it relies on LLVM's loop/SLP vectoriser under a
// module-wide `-C target-feature=+simd128`.
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub fn probe_autovec(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        let arow = &a[i * k..i * k + k];
        let orow = &mut out[i * n..i * n + n];
        for j in (0..n).step_by(4) {
            let mut acc = [0.0f32; 4];
            for p in 0..k {
                let av = arow[p];
                let w = &b[p * n + j..p * n + j + 4];
                acc[0] += av * w[0];
                acc[1] += av * w[1];
                acc[2] += av * w[2];
                acc[3] += av * w[3];
            }
            orow[j..j + 4].copy_from_slice(&acc);
        }
    }
}

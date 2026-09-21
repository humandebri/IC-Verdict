//! Hand-written wasm f32x4 dense kernel for the four matmuls that dominate the
//! forward pass.
//!
//! ## Why this exists
//!
//! Measured inside the canister at T=120 (real checkpoint, `bench_matmul`):
//! `candle`'s gemm runs every shape at **2.65 instructions/MAC**, and the phase
//! profile puts 91% of the budget in four dense matmuls. The artifact *does*
//! contain `gemm_f32::microkernel::simd128` functions — but `simd128` is not a
//! default feature of `wasm32-unknown-unknown` (rustc 1.97 `--print cfg` lists
//! bulk-memory, multivalue, mutable-globals, nontrapping-fptoint, reference-types,
//! sign-ext only), so gemm never selects them and falls back to its scalar kernels.
//!
//! Enabling `simd128` for the whole build would fix that, but it does not compile:
//! `candle-core 0.11.0` gates `vec_dot_f16` in `src/cpu/mod.rs` on
//! `any(target_feature = "neon", "avx2", "simd128")` while its `cpu/simd128.rs`
//! defines only `CurrentCpu`, so the `CurrentCpuF16` the gate needs does not exist
//! for this target. A per-function `#[target_feature(enable = "simd128")]` needs no
//! global flag, and that is exactly how gemm's own SIMD kernels got into the
//! artifact in the first place. So the kernel is enabled that way here.
//!
//! ## Safety argument for the one `unsafe` in this crate
//!
//! `simd128` instructions are accepted by the IC's Wasm validator and execute on
//! ICP: gemm's SIMD functions are already in the deployed module, and this
//! repository's wasm-feature probe called `i32x4.mul` / `i32x4.dot_i16x8` /
//! `i16x8.extend_low_i8x16` on a local replica (ADR-017 and the performance
//! measurements, both since removed; they remain in the git history at the commit
//! before `refactor!: drop the Laya backend`). The intrinsic calls are `unsafe` only
//! because Rust requires that for `core::arch` intrinsics; the call sites are
//! wrapped once, in `matmul_nt`, and the pointer arithmetic there is bounded by the
//! slice lengths it is given.
#![deny(unsafe_op_in_unsafe_fn)]

/// `#[inline(always)]` wrappers around the intrinsics.
///
/// This is the whole trick, and it cost four experiments to find. Called directly,
/// `core::arch::wasm32` intrinsics are emitted as separate out-of-line functions in
/// this toolchain (visible as `core::arch::wasm32::simd128::v128_store` in the
/// artifact, reproducible with `tools/simd-probe`). Every per-MAC call boundary then
/// costs more than the arithmetic: the kernel measured 12.5 instructions/MAC against
/// gemm's 2.65. gemm wraps its intrinsics the same way
/// (`gemm-f32-0.19.0/src/microkernel.rs:500`), which is why its SIMD is real.
#[cfg(target_arch = "wasm32")]
mod wasm_ops {
    use core::arch::wasm32::*;
    #[inline(always)]
    pub(super) unsafe fn splat(pointer: *const f32) -> v128 {
        unsafe { v128_load32_splat(pointer.cast::<u32>()) }
    }
    #[inline(always)]
    pub(super) unsafe fn load(pointer: *const f32) -> v128 {
        unsafe { v128_load(pointer.cast::<v128>()) }
    }
    #[inline(always)]
    pub(super) unsafe fn store(pointer: *mut f32, value: v128) {
        unsafe { v128_store(pointer.cast::<v128>(), value) }
    }
    #[inline(always)]
    pub(super) unsafe fn zero() -> v128 {
        f32x4_splat(0.0)
    }
    #[inline(always)]
    pub(super) unsafe fn mul_add(a: v128, b: v128, accumulator: v128) -> v128 {
        f32x4_add(accumulator, f32x4_mul(a, b))
    }
}

/// `n`-vectorized dense matmul: `out[m,n] = a[m,k] · b[k,n]`.
///
/// Row-major throughout, and `b` is expected in the `[k, n]` layout the model
/// already stores (the transposed weight). A scalar loop with the same signature is
/// always available and is what non-wasm targets and unaligned inputs use, so the
/// two can be compared on the same data.
///
/// Returns `false` when the SIMD path could not be used (non-wasm target, `n` not a
/// multiple of four, or a pointer that is not 16-byte aligned), in which case
/// `out` holds the scalar result.
#[must_use]
pub fn matmul_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) -> bool {
    assert!(a.len() >= m * k, "a is {m}x{k}");
    assert!(b.len() >= k * n, "b is {k}x{n}");
    assert!(out.len() >= m * n, "out is {m}x{n}");
    #[cfg(target_arch = "wasm32")]
    {
        if n % 4 == 0 && aligned16(b.as_ptr()) && aligned16(out.as_ptr()) {
            // SAFETY: `simd128` executes on ICP (see the module docs), `n % 4 == 0`
            // keeps every 4-wide column step inside one row, both pointers are
            // 16-byte aligned, and the lengths were checked above.
            if m >= 4 && aligned16(a.as_ptr()) {
                unsafe { gemm_block4x4_u4(a, b, m, k, n, out) };
                // Measured: 4x4 = 3.628 instr/MAC, 2x4 variant = 4.504 (fewer weight
                // reuses), so 4x4 stays the default. Both still spill accumulators.
                let m4 = m / 4 * 4;
                for r in m4..m {
                    for c in 0..n {
                        let mut acc = 0.0f32;
                        for p in 0..k {
                            acc += a[r * k + p] * b[p * n + c];
                        }
                        out[r * n + c] = acc;
                    }
                }
            } else {
                unsafe { gemm_n4(a, b, m, k, n, out) };
            }
            return true;
        }
    }
    matmul_scalar(a, b, m, k, n, out);
    false
}

pub fn aligned16<T>(pointer: *const T) -> bool {
    pointer as usize % 16 == 0
}

/// Reference implementation. Also the fallback for every non-wasm target, which is
/// what makes the SIMD path testable against something independent.
pub fn matmul_scalar(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        let arow = &a[i * k..i * k + k];
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += arow[p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}


/// 4 rows × 4 columns per inner block.
///
/// The 1×4 kernel reloads the whole weight block once per output row, so the B load
/// dominates: measured 6.0 instructions/MAC against gemm's 2.5. Holding four row
/// accumulators reuses each 4-wide weight load four times, which is the same trick
/// gemm uses with 4×3. Per k-step this is one 4-wide load, four broadcasts, four
/// multiplies, four adds and three pointer updates: about 1.2 instructions/MAC.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline(never)]
unsafe fn gemm_block4x4(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    use wasm_ops::{load, mul_add, splat, store, zero};
    let m4 = m / 4 * 4;
    let n4 = n / 4 * 4;
    let mut i = 0;
    while i < m4 {
        let row = unsafe { a.as_ptr().add(i * k) };
        let orow = unsafe { out.as_mut_ptr().add(i * n) };
        let mut j = 0;
        while j < n4 {
            let mut acc0 = unsafe { zero() };
            let mut acc1 = unsafe { zero() };
            let mut acc2 = unsafe { zero() };
            let mut acc3 = unsafe { zero() };
            let mut ap = row;
            let mut bp = unsafe { b.as_ptr().add(j) };
            let mut p = 0;
            while p < k {
                // SAFETY: `n % 4 == 0` keeps this 4-wide load inside one row of b, the
                // buffers are 16-byte aligned (checked by the caller) and every row
                // index is below `m`.
                let weight = unsafe { load(bp) };
                acc0 = unsafe { mul_add(splat(ap), weight, acc0) };
                acc1 = unsafe { mul_add(splat(ap.add(k)), weight, acc1) };
                acc2 = unsafe { mul_add(splat(ap.add(2 * k)), weight, acc2) };
                acc3 = unsafe { mul_add(splat(ap.add(3 * k)), weight, acc3) };
                ap = unsafe { ap.add(1) };
                bp = unsafe { bp.add(n) };
                p += 1;
            }
            unsafe { store(orow.add(j), acc0) };
            unsafe { store(orow.add(j + n), acc1) };
            unsafe { store(orow.add(j + 2 * n), acc2) };
            unsafe { store(orow.add(j + 3 * n), acc3) };
            j += 4;
        }
        i += 4;
    }
}

/// 2 rows × 4 columns per inner block.
///
/// The 4×4 kernel spills its four `v128` accumulators to the stack inside the k-loop
/// (the artifact shows `v128.store` of a zero constant per iteration), which is why it
/// measures 3.6 instructions/MAC instead of the ~1.2 the instruction mix predicts.
/// Two accumulators lower the register pressure; the trade is fewer weight reuses.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline(never)]
unsafe fn gemm_block2x4(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    use wasm_ops::{load, mul_add, splat, store, zero};
    let m2 = m / 2 * 2;
    let n4 = n / 4 * 4;
    let mut i = 0;
    while i < m2 {
        let row = unsafe { a.as_ptr().add(i * k) };
        let orow = unsafe { out.as_mut_ptr().add(i * n) };
        let mut j = 0;
        while j < n4 {
            let mut acc0 = unsafe { zero() };
            let mut acc1 = unsafe { zero() };
            let mut ap = row;
            let mut bp = unsafe { b.as_ptr().add(j) };
            let mut p = 0;
            while p < k {
                // SAFETY: `n % 4 == 0`, both buffers are 16-byte aligned and every row
                // index is below `m`; checked by the caller.
                let weight = unsafe { load(bp) };
                acc0 = unsafe { mul_add(splat(ap), weight, acc0) };
                acc1 = unsafe { mul_add(splat(ap.add(k)), weight, acc1) };
                ap = unsafe { ap.add(1) };
                bp = unsafe { bp.add(n) };
                p += 1;
            }
            unsafe { store(orow.add(j), acc0) };
            unsafe { store(orow.add(j + n), acc1) };
            j += 4;
        }
        i += 2;
    }
}


/// 4 rows × 4 columns, with the k-loop unrolled four deep.
///
/// Why: the generated code for the plain version spends about 31 non-vector
/// instructions per k-step (21 `local.get`/`set` plus 10 `i32` address updates) against
/// only 13 vector instructions, because wasm has no register file — every operand is a
/// local. Unrolling four k-steps amortises that bookkeeping over 64 MACs instead of 16,
/// which is the difference between 3.6 and the ~1.3 instructions/MAC this instruction
/// mix predicts. `k` is normally a multiple of four (768, 1152, ...); the remainder runs
/// as single steps.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline(never)]
unsafe fn gemm_block4x4_u4(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    use wasm_ops::{load, mul_add, splat, store, zero};
    let m4 = m / 4 * 4;
    let n4 = n / 4 * 4;
    let stride = n;
    let stride4 = n * 4;
    let mut i = 0;
    while i < m4 {
        let row = unsafe { a.as_ptr().add(i * k) };
        let orow = unsafe { out.as_mut_ptr().add(i * n) };
        let mut j = 0;
        while j < n4 {
            let mut acc0 = unsafe { zero() };
            let mut acc1 = unsafe { zero() };
            let mut acc2 = unsafe { zero() };
            let mut acc3 = unsafe { zero() };
            let mut ap = row;
            let mut bp = unsafe { b.as_ptr().add(j) };
            let mut p = 0;
            while p + 4 <= k {
                // SAFETY: `n % 4 == 0` keeps each 4-wide load inside one row of b, all
                // buffers are 16-byte aligned and every row index is below `m`.
                let w0 = unsafe { load(bp) };
                let w1 = unsafe { load(bp.add(stride)) };
                let w2 = unsafe { load(bp.add(stride * 2)) };
                let w3 = unsafe { load(bp.add(stride * 3)) };
                let a0 = unsafe { splat(ap) };
                let a1 = unsafe { splat(ap.add(1)) };
                let a2 = unsafe { splat(ap.add(2)) };
                let a3 = unsafe { splat(ap.add(3)) };
                acc0 = unsafe { mul_add(a3, w3, mul_add(a2, w2, mul_add(a1, w1, mul_add(a0, w0, acc0)))) };
                let b0 = unsafe { splat(ap.add(k)) };
                let b1 = unsafe { splat(ap.add(k + 1)) };
                let b2 = unsafe { splat(ap.add(k + 2)) };
                let b3 = unsafe { splat(ap.add(k + 3)) };
                acc1 = unsafe { mul_add(b3, w3, mul_add(b2, w2, mul_add(b1, w1, mul_add(b0, w0, acc1)))) };
                let c0 = unsafe { splat(ap.add(2 * k)) };
                let c1 = unsafe { splat(ap.add(2 * k + 1)) };
                let c2 = unsafe { splat(ap.add(2 * k + 2)) };
                let c3 = unsafe { splat(ap.add(2 * k + 3)) };
                acc2 = unsafe { mul_add(c3, w3, mul_add(c2, w2, mul_add(c1, w1, mul_add(c0, w0, acc2)))) };
                let d0 = unsafe { splat(ap.add(3 * k)) };
                let d1 = unsafe { splat(ap.add(3 * k + 1)) };
                let d2 = unsafe { splat(ap.add(3 * k + 2)) };
                let d3 = unsafe { splat(ap.add(3 * k + 3)) };
                acc3 = unsafe { mul_add(d3, w3, mul_add(d2, w2, mul_add(d1, w1, mul_add(d0, w0, acc3)))) };
                ap = unsafe { ap.add(4) };
                bp = unsafe { bp.add(stride4) };
                p += 4;
            }
            while p < k {
                let weight = unsafe { load(bp) };
                acc0 = unsafe { mul_add(splat(ap), weight, acc0) };
                acc1 = unsafe { mul_add(splat(ap.add(k)), weight, acc1) };
                acc2 = unsafe { mul_add(splat(ap.add(2 * k)), weight, acc2) };
                acc3 = unsafe { mul_add(splat(ap.add(3 * k)), weight, acc3) };
                ap = unsafe { ap.add(1) };
                bp = unsafe { bp.add(stride) };
                p += 1;
            }
            unsafe { store(orow.add(j), acc0) };
            unsafe { store(orow.add(j + n), acc1) };
            unsafe { store(orow.add(j + 2 * n), acc2) };
            unsafe { store(orow.add(j + 3 * n), acc3) };
            j += 4;
        }
        i += 4;
    }
}

/// A 16-byte aligned `f32` buffer.
///
/// `Vec<f32>` only guarantees 4-byte alignment and `v128_load` traps on a
/// misaligned pointer, so weight and activation buffers are built here and the
/// first aligned index is remembered.
pub struct AlignedF32 {
    data: Vec<f32>,
    start: usize,
    len: usize,
}

impl AlignedF32 {
    pub fn from_slice(values: &[f32]) -> Self {
        // Worst case the pointer lands 12 bytes past an aligned address, hence four
        // extra elements of slack.
        let mut data: Vec<f32> = Vec::with_capacity(values.len() + 4);
        data.extend_from_slice(values);
        data.resize(values.len() + 4, 0.0);
        let start = match (16 - (data.as_ptr() as usize % 16)) % 16 {
            0 => 0,
            4 => 1,
            8 => 2,
            _ => 3,
        };
        Self { data, start, len: values.len() }
    }
    pub fn zeros(len: usize) -> Self {
        Self::from_slice(&vec![0.0f32; len])
    }
    pub fn as_slice(&self) -> &[f32] {
        &self.data[self.start..self.start + self.len]
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        let (start, len) = (self.start, self.len);
        &mut self.data[start..start + len]
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn gemm_n4(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    use wasm_ops::{load, mul_add, splat, store, zero};
    let n4 = n / 4 * 4;
    let mut i = 0;
    while i < m {
        let arow = unsafe { a.as_ptr().add(i * k) };
        let orow = unsafe { out.as_mut_ptr().add(i * n) };
        let mut j = 0;
        while j < n4 {
            // SAFETY: pointers derive from slices whose lengths were checked in
            // `matmul_nt`; `n % 4 == 0` keeps 4-wide steps inside one row and both
            // buffers are 16-byte aligned.
            let mut accumulator = unsafe { zero() };
            let mut p = 0;
            while p < k {
                let activation = unsafe { splat(arow.add(p)) };
                let weight = unsafe { load(b.as_ptr().add(p * n + j)) };
                accumulator = unsafe { mul_add(activation, weight, accumulator) };
                p += 1;
            }
            unsafe { store(orow.add(j), accumulator) };
            j += 4;
        }
        while j < n {
            let mut accumulator = 0.0f32;
            for p in 0..k {
                accumulator += unsafe { *arow.add(p) } * unsafe { *b.as_ptr().add(p * n + j) };
            }
            unsafe { *orow.add(j) = accumulator };
            j += 1;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let a = (0..m * k).map(|i| ((i % 13) as f32) * 0.25 - 1.5).collect();
        let b = (0..k * n).map(|i| ((i % 7) as f32) * 0.5 - 1.5).collect();
        (a, b)
    }

    #[test]
    fn scalar_matmul_matches_a_hand_computed_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let mut out = vec![0.0; 4];
        matmul_scalar(&a, &b, 2, 2, 2, &mut out);
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn aligned_buffer_reports_an_aligned_base() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let buffer = AlignedF32::from_slice(&values);
        assert!(aligned16(buffer.as_slice().as_ptr()));
        assert_eq!(buffer.as_slice(), values.as_slice());
    }

    #[test]
    fn int8_matmul_tracks_the_f32_result() {
        let (m, k, n) = (4usize, 32usize, 5usize);
        let (a, b) = sample(m, k, n);
        // b here is [k, n]; the int8 kernel wants the weight as [n, k].
        let mut w = vec![0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                w[j * k + p] = b[p * n + j];
            }
        }
        let (aq, asx) = quantize_acts_i8_i16(&a, m, k);
        let (wq, wsx) = quantize_rows_i8(&w, n, k);
        let mut got = vec![0f32; m * n];
        let simd = matmul_i8(&aq, &wq, &asx, &wsx, m, k, n, &mut got);
        let mut want = vec![0f32; m * n];
        matmul_scalar(&a, &b, m, k, n, &mut want);
        let scale = want.iter().fold(1e-6f32, |acc, v| acc.max(v.abs()));
        for (x, y) in got.iter().zip(want.iter()) {
            assert!((x - y).abs() / scale < 2e-2, "{x} vs {y}");
        }
        // Native always takes the scalar reference path.
        #[cfg(not(target_arch = "wasm32"))]
        assert!(!simd);
    }

    #[test]
    fn matmul_nt_returns_the_same_result_as_the_reference() {
        let (a, b) = sample(3, 8, 12);
        let mut got = vec![0.0; 3 * 12];
        let mut want = vec![0.0; 3 * 12];
        matmul_scalar(&a, &b, 3, 8, 12, &mut want);
        let simd = matmul_nt(&a, &b, 3, 8, 12, &mut got);
        for (x, y) in got.iter().zip(want.iter()) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
        // On native this is always the scalar path, which is what the assertion above
        // checks; the canister compares the two paths on the wasm target.
        #[cfg(not(target_arch = "wasm32"))]
        assert!(!simd);
    }
}

/// Codegen probes. These answer a single question that the model's performance
/// depends on: which `core::arch::wasm32` intrinsics survive into v128 instructions
/// in this toolchain, and which are scalarised. `gemm`'s wasm microkernel only uses
/// vector loads (it packs the left operand first), and its artifact contains real
/// SIMD; the kernels here use a splat and the artifact does not. Counting the SIMD
/// instructions in each probe function in the built `.wasm` settles it:
///
///     wasm-tools print build/verdict-engine.wasm | grep -A2 simd_probe
///
/// Not part of the model. Kept because the answer decides whether a hand-written
/// int8/ternary kernel is even expressible here.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn simd_probe_vector_loads(a: &[f32], b: &[f32], out: &mut [f32]) {
    use core::arch::wasm32::*;
    unsafe {
        let mut acc = v128_load(a.as_ptr().cast::<v128>());
        for i in 1..4 {
            let av = v128_load(a.as_ptr().add(i * 4).cast::<v128>());
            let bv = v128_load(b.as_ptr().add(i * 4).cast::<v128>());
            acc = f32x4_add(acc, f32x4_mul(av, bv));
        }
        v128_store(out.as_mut_ptr().cast::<v128>(), acc);
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn simd_probe_splat(a: &[f32], b: &[f32], out: &mut [f32]) {
    use core::arch::wasm32::*;
    unsafe {
        let mut acc = f32x4_splat(0.0);
        for i in 0..4 {
            let av = f32x4_splat(*a.as_ptr().add(i));
            let bv = v128_load(b.as_ptr().add(i * 4).cast::<v128>());
            acc = f32x4_add(acc, f32x4_mul(av, bv));
        }
        v128_store(out.as_mut_ptr().cast::<v128>(), acc);
    }
}

// --- int8 kernel -----------------------------------------------------------------
//
// Why the layout matters: `i32x4.dot_i16x8_s` multiplies eight pairs and adds them
// two at a time, so a *dot product along k* is the shape it wants. That means the
// weight row for one output must be contiguous in k — which is exactly the original
// checkpoint layout `[n, k]`, so no transpose is needed (the f32 path had to
// pre-transpose to `[k, n]` to get vector loads along n). Quantisation is per row,
// i.e. one scale per output element and one per token, and the accumulators stay in
// i32: 127*127*1152 = 1.9e7, far inside i32 range.

/// Per-row symmetric int8 quantisation. Returns the quantised values and the scale per
/// row (one scale for every `cols` values).
///
/// The scale uses the 99.9th percentile of the row magnitude rather than the maximum, so
/// a single outlier cannot stretch the scale and coarsen every other weight in the row.
/// Outliers then clip to +-127. Measured effect on the golden gate is in
/// docs/VERDICT_ENGINE.md 5.1.6.
pub fn quantize_rows_i8(src: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![1.0f32; rows];
    let mut scratch = vec![0.0f32; cols];
    for r in 0..rows {
        let row = &src[r * cols..r * cols + cols];
        scratch.copy_from_slice(row);
        scratch.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal));
        let rank = ((cols as f32 * 0.999) as usize).min(cols - 1);
        let max = scratch[rank].abs();
        let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
        scales[r] = scale;
        for (i, v) in row.iter().enumerate() {
            q[r * cols + i] = (v / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

/// Activation quantisation into the i16 range.
///
/// The activations were already stored as i16, so using the full i16 range instead of
/// the i8 range costs nothing at inference and gives 64x finer resolution. The range is
/// bounded by the i32 accumulator: `range * 127 * k` must stay below `2^31`
/// (127 * 1152 = 146304, so anything up to ~14600 is safe).
pub const ACTIVATION_RANGE: f32 = 8192.0;

pub fn quantize_acts_i16(src: &[f32], rows: usize, cols: usize) -> (Vec<i16>, Vec<f32>) {
    let mut q = vec![0i16; rows * cols];
    let mut scales = vec![1.0f32; rows];
    for r in 0..rows {
        let row = &src[r * cols..r * cols + cols];
        let max = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        // Dividing by the scale costs a full division per element and measured 167
        // instructions per element; multiplying by the reciprocal and rounding with
        // `f32x4.nearest` in blocks of eight is a small fraction of that.
        let inv_scale = if max > 0.0 { ACTIVATION_RANGE / max } else { 1.0 };
        scales[r] = if max > 0.0 { max / ACTIVATION_RANGE } else { 1.0 };
        let out = &mut q[r * cols..r * cols + cols];
        #[cfg(target_arch = "wasm32")]
        if cols % 8 == 0 {
            // SAFETY: the loop runs while `p + 8 <= cols`, so both the 8-wide load and
            // the 8-wide store stay inside the row slices.
            unsafe { quantize_row_i16_simd(row, out, inv_scale) };
            continue;
        }
        for (i, v) in row.iter().enumerate() {
            out[i] = (v * inv_scale).round().clamp(-ACTIVATION_RANGE, ACTIVATION_RANGE) as i16;
        }
    }
    (q, scales)
}

/// Eight activations per iteration: multiply by the reciprocal, round to nearest and
/// saturate into i16.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn quantize_row_i16_simd(row: &[f32], out: &mut [i16], inv_scale: f32) {
    use core::arch::wasm32::*;
    let scale = f32x4_splat(inv_scale);
    let mut p = 0;
    while p + 8 <= row.len() {
        let a = unsafe { v128_load(row.as_ptr().add(p).cast()) };
        let b = unsafe { v128_load(row.as_ptr().add(p + 4).cast()) };
        let ai = i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_mul(a, scale)));
        let bi = i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_mul(b, scale)));
        unsafe { v128_store(out.as_mut_ptr().add(p).cast(), i16x8_narrow_i32x4(ai, bi)) };
        p += 8;
    }
}

/// Kept for the earlier measurements; prefer [`quantize_acts_i16`].
pub fn quantize_acts_i8_i16(src: &[f32], rows: usize, cols: usize) -> (Vec<i16>, Vec<f32>) {
    let (q, scales) = quantize_rows_i8(src, rows, cols);
    (q.iter().map(|v| *v as i16).collect(), scales)
}

/// `out[m, n] = a[m, k] · w[n, k]ᵀ` with both sides pre-widened to i16.
///
/// Returns `true` when the wasm SIMD path ran; the scalar path is the reference used
/// by the native tests.
#[must_use]
pub fn matmul_i8(a: &[i16], w: &[i8], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) -> bool {
    assert!(a.len() >= m * k && w.len() >= n * k && sx.len() >= m && sw.len() >= n && out.len() >= m * n);
    #[cfg(target_arch = "wasm32")]
    {
        if k % 8 == 0 {
            // SAFETY: see the module-level note; `k % 8 == 0` keeps every 8-wide load
            // inside one row and all pointers come from slices checked above.
            match k {
                768 => unsafe { matmul_i8_simd_k8::<768>(a, w, sx, sw, m, n, out) },
                1152 => unsafe { matmul_i8_simd_k8::<1152>(a, w, sx, sw, m, n, out) },
                _ => unsafe { matmul_i8_simd(a, w, sx, sw, m, k, n, out) },
            }
            return true;
        }
    }
    matmul_i8_scalar(a, w, sx, sw, m, k, n, out);
    false
}

/// Reference implementation of the int8 kernel (also the native path).
pub fn matmul_i8_scalar(a: &[i16], w: &[i8], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += a[i * k + p] as i32 * w[j * k + p] as i32;
            }
            out[i * n + j] = acc as f32 * sx[i] * sw[j];
        }
    }
}

/// Two output rows per pass, so each weight load feeds both rows.
///
/// The 1-row kernel spends about as many instructions on weight addresses as on the
/// arithmetic; sharing every 8-wide weight vector between two rows roughly halves that
/// per MAC, at the cost of eight i32x4 accumulators. Measured in docs/VERDICT_ENGINE.md
/// 5.1.10. `K` is a constant so the four weight rows of a column block sit at fixed
/// offsets.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn matmul_i8_simd_k2<const K: usize>(a: &[i16], w: &[i8], sx: &[f32], sw: &[f32], m: usize, n: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    let n4 = n / 4 * 4;
    let m2 = m / 2 * 2;
    let mut i = 0;
    while i < m2 {
        let r0 = unsafe { a.as_ptr().add(i * K) };
        let r1 = unsafe { r0.add(K) };
        let o0 = unsafe { out.as_mut_ptr().add(i * n) };
        let o1 = unsafe { o0.add(n) };
        let mut j = 0;
        while j < n4 {
            let mut c00 = i32x4_splat(0); let mut c01 = i32x4_splat(0);
            let mut c02 = i32x4_splat(0); let mut c03 = i32x4_splat(0);
            let mut c10 = i32x4_splat(0); let mut c11 = i32x4_splat(0);
            let mut c12 = i32x4_splat(0); let mut c13 = i32x4_splat(0);
            let wbase = unsafe { w.as_ptr().add(j * K) };
            let mut wp = wbase;
            let mut xa = r0;
            let mut xb = r1;
            let mut p = 0;
            while p + 32 <= K {
                let xa0 = unsafe { v128_load(xa.cast()) };
                let xa1 = unsafe { v128_load(xa.add(8).cast()) };
                let xa2 = unsafe { v128_load(xa.add(16).cast()) };
                let xa3 = unsafe { v128_load(xa.add(24).cast()) };
                let xb0 = unsafe { v128_load(xb.cast()) };
                let xb1 = unsafe { v128_load(xb.add(8).cast()) };
                let xb2 = unsafe { v128_load(xb.add(16).cast()) };
                let xb3 = unsafe { v128_load(xb.add(24).cast()) };
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xa0, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xb0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(8).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xa1, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xb1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(16).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xa2, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xb2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(24).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xa3, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xb3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xa0, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xb0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 8).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xa1, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xb1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 16).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xa2, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xb2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 24).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xa3, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xb3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xa0, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xb0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 8).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xa1, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xb1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 16).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xa2, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xb2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 24).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xa3, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xb3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(xa0, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(xb0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 8).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(xa1, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(xb1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 16).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(xa2, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(xb2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 24).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(xa3, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(xb3, wv));
                xa = unsafe { xa.add(32) };
                xb = unsafe { xb.add(32) };
                wp = unsafe { wp.add(32) };
                p += 32;
            }
            while p < K {
                let xav = unsafe { v128_load(xa.cast()) };
                let xbv = unsafe { v128_load(xb.cast()) };
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xav, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xbv, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xav, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xbv, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xav, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xbv, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(xav, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(xbv, wv));
                xa = unsafe { xa.add(8) };
                xb = unsafe { xb.add(8) };
                wp = unsafe { wp.add(8) };
                p += 8;
            }
            let reduce = |acc: v128| -> f32 {
                (i32x4_extract_lane::<0>(acc) + i32x4_extract_lane::<1>(acc)
                    + i32x4_extract_lane::<2>(acc) + i32x4_extract_lane::<3>(acc)) as f32
            };
            unsafe {
                *o0.add(j) = reduce(c00) * sx[i] * sw[j];
                *o0.add(j + 1) = reduce(c01) * sx[i] * sw[j + 1];
                *o0.add(j + 2) = reduce(c02) * sx[i] * sw[j + 2];
                *o0.add(j + 3) = reduce(c03) * sx[i] * sw[j + 3];
                *o1.add(j) = reduce(c10) * sx[i + 1] * sw[j];
                *o1.add(j + 1) = reduce(c11) * sx[i + 1] * sw[j + 1];
                *o1.add(j + 2) = reduce(c12) * sx[i + 1] * sw[j + 2];
                *o1.add(j + 3) = reduce(c13) * sx[i + 1] * sw[j + 3];
            }
            j += 4;
        }
        while j < n {
            for r in 0..2 {
                let row = if r == 0 { r0 } else { r1 };
                let mut acc = 0i32;
                for p in 0..K { acc += unsafe { *row.add(p) as i32 * *w.as_ptr().add(j * K + p) as i32 }; }
                unsafe { *if r == 0 { o0 } else { o1 }.add(j) = acc as f32 * sx[i + r] * sw[j] };
            }
            j += 1;
        }
        i += 2;
    }
    // Odd row left over: plain scalar row.
    while i < m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..K { acc += unsafe { *a.as_ptr().add(i * K + p) as i32 * *w.as_ptr().add(j * K + p) as i32 }; }
            unsafe { *out.as_mut_ptr().add(i * n + j) = acc as f32 * sx[i] * sw[j] };
        }
        i += 1;
    }
}

/// Four output rows per pass: each 8-wide weight vector now feeds four rows.
///
/// Same idea as the two-row kernel, one step further. The tail (`m % 4` rows) is handed
/// to the two-row kernel, which in turn handles its own odd row. Measured in
/// docs/VERDICT_ENGINE.md 5.1.11.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn matmul_i8_simd_k4<const K: usize>(a: &[i16], w: &[i8], sx: &[f32], sw: &[f32], m: usize, n: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    let n4 = n / 4 * 4;
    let m4 = m / 4 * 4;
    let mut i = 0;
    while i < m4 {
        let r: [*const i16; 4] = [
            unsafe { a.as_ptr().add(i * K) },
            unsafe { a.as_ptr().add((i + 1) * K) },
            unsafe { a.as_ptr().add((i + 2) * K) },
            unsafe { a.as_ptr().add((i + 3) * K) },
        ];
        let o: [*mut f32; 4] = [
            unsafe { out.as_mut_ptr().add(i * n) },
            unsafe { out.as_mut_ptr().add((i + 1) * n) },
            unsafe { out.as_mut_ptr().add((i + 2) * n) },
            unsafe { out.as_mut_ptr().add((i + 3) * n) },
        ];
        let mut j = 0;
        while j < n4 {
            let mut c00 = i32x4_splat(0); let mut c01 = i32x4_splat(0); let mut c02 = i32x4_splat(0); let mut c03 = i32x4_splat(0);
            let mut c10 = i32x4_splat(0); let mut c11 = i32x4_splat(0); let mut c12 = i32x4_splat(0); let mut c13 = i32x4_splat(0);
            let mut c20 = i32x4_splat(0); let mut c21 = i32x4_splat(0); let mut c22 = i32x4_splat(0); let mut c23 = i32x4_splat(0);
            let mut c30 = i32x4_splat(0); let mut c31 = i32x4_splat(0); let mut c32 = i32x4_splat(0); let mut c33 = i32x4_splat(0);
            let wbase = unsafe { w.as_ptr().add(j * K) };
            let mut wp = wbase;
            let mut xp: [*const i16; 4] = [r[0], r[1], r[2], r[3]];
            let mut p = 0;
            while p + 32 <= K {
                let x0_0 = unsafe { v128_load(xp[0].add(0).cast()) };
                let x0_1 = unsafe { v128_load(xp[0].add(8).cast()) };
                let x0_2 = unsafe { v128_load(xp[0].add(16).cast()) };
                let x0_3 = unsafe { v128_load(xp[0].add(24).cast()) };
                let x1_0 = unsafe { v128_load(xp[1].add(0).cast()) };
                let x1_1 = unsafe { v128_load(xp[1].add(8).cast()) };
                let x1_2 = unsafe { v128_load(xp[1].add(16).cast()) };
                let x1_3 = unsafe { v128_load(xp[1].add(24).cast()) };
                let x2_0 = unsafe { v128_load(xp[2].add(0).cast()) };
                let x2_1 = unsafe { v128_load(xp[2].add(8).cast()) };
                let x2_2 = unsafe { v128_load(xp[2].add(16).cast()) };
                let x2_3 = unsafe { v128_load(xp[2].add(24).cast()) };
                let x3_0 = unsafe { v128_load(xp[3].add(0).cast()) };
                let x3_1 = unsafe { v128_load(xp[3].add(8).cast()) };
                let x3_2 = unsafe { v128_load(xp[3].add(16).cast()) };
                let x3_3 = unsafe { v128_load(xp[3].add(24).cast()) };
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0 * K).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0_0, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1_0, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2_0, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3_0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0 * K + 8).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0_1, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1_1, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2_1, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3_1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0 * K + 16).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0_2, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1_2, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2_2, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3_2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0 * K + 24).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0_3, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1_3, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2_3, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3_3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0_0, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1_0, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2_0, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3_0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 8).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0_1, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1_1, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2_1, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3_1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 16).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0_2, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1_2, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2_2, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3_2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 24).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0_3, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1_3, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2_3, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3_3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0_0, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1_0, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2_0, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3_0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 8).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0_1, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1_1, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2_1, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3_1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 16).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0_2, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1_2, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2_2, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3_2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 24).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0_3, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1_3, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2_3, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3_3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0_0, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1_0, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2_0, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3_0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 8).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0_1, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1_1, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2_1, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3_1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 16).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0_2, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1_2, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2_2, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3_2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 24).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0_3, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1_3, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2_3, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3_3, wv));
                for rr in 0..4 { xp[rr] = unsafe { xp[rr].add(32) }; }
                wp = unsafe { wp.add(32) };
                p += 32;
            }
            while p < K {
                let x0v = unsafe { v128_load(xp[0].cast()) };
                let x1v = unsafe { v128_load(xp[1].cast()) };
                let x2v = unsafe { v128_load(xp[2].cast()) };
                let x3v = unsafe { v128_load(xp[3].cast()) };
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0v, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1v, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2v, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3v, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0v, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1v, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2v, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3v, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0v, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1v, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2v, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3v, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0v, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1v, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2v, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3v, wv));
                for rr in 0..4 { xp[rr] = unsafe { xp[rr].add(8) }; }
                wp = unsafe { wp.add(8) };
                p += 8;
            }
            let reduce = |acc: v128| -> f32 {
                (i32x4_extract_lane::<0>(acc) + i32x4_extract_lane::<1>(acc)
                    + i32x4_extract_lane::<2>(acc) + i32x4_extract_lane::<3>(acc)) as f32
            };
            unsafe {
                *o[0].add(j + 0) = reduce(c00) * sx[i + 0] * sw[j + 0];
                *o[0].add(j + 1) = reduce(c01) * sx[i + 0] * sw[j + 1];
                *o[0].add(j + 2) = reduce(c02) * sx[i + 0] * sw[j + 2];
                *o[0].add(j + 3) = reduce(c03) * sx[i + 0] * sw[j + 3];
                *o[1].add(j + 0) = reduce(c10) * sx[i + 1] * sw[j + 0];
                *o[1].add(j + 1) = reduce(c11) * sx[i + 1] * sw[j + 1];
                *o[1].add(j + 2) = reduce(c12) * sx[i + 1] * sw[j + 2];
                *o[1].add(j + 3) = reduce(c13) * sx[i + 1] * sw[j + 3];
                *o[2].add(j + 0) = reduce(c20) * sx[i + 2] * sw[j + 0];
                *o[2].add(j + 1) = reduce(c21) * sx[i + 2] * sw[j + 1];
                *o[2].add(j + 2) = reduce(c22) * sx[i + 2] * sw[j + 2];
                *o[2].add(j + 3) = reduce(c23) * sx[i + 2] * sw[j + 3];
                *o[3].add(j + 0) = reduce(c30) * sx[i + 3] * sw[j + 0];
                *o[3].add(j + 1) = reduce(c31) * sx[i + 3] * sw[j + 1];
                *o[3].add(j + 2) = reduce(c32) * sx[i + 3] * sw[j + 2];
                *o[3].add(j + 3) = reduce(c33) * sx[i + 3] * sw[j + 3];
            }
            j += 4;
        }
        while j < n {
            for rr in 0..4 {
                let mut acc = 0i32;
                for p in 0..K { acc += unsafe { *r[rr].add(p) as i32 * *w.as_ptr().add(j * K + p) as i32 }; }
                unsafe { *o[rr].add(j) = acc as f32 * sx[i + rr] * sw[j] };
            }
            j += 1;
        }
        i += 4;
    }
    if m4 < m {
        unsafe { matmul_i8_simd_k2::<K>(&a[m4 * K..], w, &sx[m4..], sw, m - m4, n, &mut out[m4 * n..]) };
    }
}

/// Eight output rows per pass. The tail (`m % 8`) is handed to the four-row kernel.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn matmul_i8_simd_k8<const K: usize>(a: &[i16], w: &[i8], sx: &[f32], sw: &[f32], m: usize, n: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    let n4 = n / 4 * 4;
    let m8 = m / 8 * 8;
    let mut i = 0;
    while i < m8 {
        let mut r: [*const i16; 8] = [core::ptr::null(); 8];
        let mut o: [*mut f32; 8] = [core::ptr::null_mut(); 8];
        for k in 0..8 {
            r[k] = unsafe { a.as_ptr().add((i + k) * K) };
            o[k] = unsafe { out.as_mut_ptr().add((i + k) * n) };
        }
        let mut j = 0;
        while j < n4 {
            let mut c00 = i32x4_splat(0); let mut c01 = i32x4_splat(0); let mut c02 = i32x4_splat(0); let mut c03 = i32x4_splat(0);
            let mut c10 = i32x4_splat(0); let mut c11 = i32x4_splat(0); let mut c12 = i32x4_splat(0); let mut c13 = i32x4_splat(0);
            let mut c20 = i32x4_splat(0); let mut c21 = i32x4_splat(0); let mut c22 = i32x4_splat(0); let mut c23 = i32x4_splat(0);
            let mut c30 = i32x4_splat(0); let mut c31 = i32x4_splat(0); let mut c32 = i32x4_splat(0); let mut c33 = i32x4_splat(0);
            let mut c40 = i32x4_splat(0); let mut c41 = i32x4_splat(0); let mut c42 = i32x4_splat(0); let mut c43 = i32x4_splat(0);
            let mut c50 = i32x4_splat(0); let mut c51 = i32x4_splat(0); let mut c52 = i32x4_splat(0); let mut c53 = i32x4_splat(0);
            let mut c60 = i32x4_splat(0); let mut c61 = i32x4_splat(0); let mut c62 = i32x4_splat(0); let mut c63 = i32x4_splat(0);
            let mut c70 = i32x4_splat(0); let mut c71 = i32x4_splat(0); let mut c72 = i32x4_splat(0); let mut c73 = i32x4_splat(0);
            let wbase = unsafe { w.as_ptr().add(j * K) };
            let mut wp = wbase;
            let mut xp: [*const i16; 8] = r;
            let mut p = 0;
            while p + 32 <= K {
                let x0_0 = unsafe { v128_load(xp[0].add(0).cast()) };
                let x0_1 = unsafe { v128_load(xp[0].add(8).cast()) };
                let x0_2 = unsafe { v128_load(xp[0].add(16).cast()) };
                let x0_3 = unsafe { v128_load(xp[0].add(24).cast()) };
                let x1_0 = unsafe { v128_load(xp[1].add(0).cast()) };
                let x1_1 = unsafe { v128_load(xp[1].add(8).cast()) };
                let x1_2 = unsafe { v128_load(xp[1].add(16).cast()) };
                let x1_3 = unsafe { v128_load(xp[1].add(24).cast()) };
                let x2_0 = unsafe { v128_load(xp[2].add(0).cast()) };
                let x2_1 = unsafe { v128_load(xp[2].add(8).cast()) };
                let x2_2 = unsafe { v128_load(xp[2].add(16).cast()) };
                let x2_3 = unsafe { v128_load(xp[2].add(24).cast()) };
                let x3_0 = unsafe { v128_load(xp[3].add(0).cast()) };
                let x3_1 = unsafe { v128_load(xp[3].add(8).cast()) };
                let x3_2 = unsafe { v128_load(xp[3].add(16).cast()) };
                let x3_3 = unsafe { v128_load(xp[3].add(24).cast()) };
                let x4_0 = unsafe { v128_load(xp[4].add(0).cast()) };
                let x4_1 = unsafe { v128_load(xp[4].add(8).cast()) };
                let x4_2 = unsafe { v128_load(xp[4].add(16).cast()) };
                let x4_3 = unsafe { v128_load(xp[4].add(24).cast()) };
                let x5_0 = unsafe { v128_load(xp[5].add(0).cast()) };
                let x5_1 = unsafe { v128_load(xp[5].add(8).cast()) };
                let x5_2 = unsafe { v128_load(xp[5].add(16).cast()) };
                let x5_3 = unsafe { v128_load(xp[5].add(24).cast()) };
                let x6_0 = unsafe { v128_load(xp[6].add(0).cast()) };
                let x6_1 = unsafe { v128_load(xp[6].add(8).cast()) };
                let x6_2 = unsafe { v128_load(xp[6].add(16).cast()) };
                let x6_3 = unsafe { v128_load(xp[6].add(24).cast()) };
                let x7_0 = unsafe { v128_load(xp[7].add(0).cast()) };
                let x7_1 = unsafe { v128_load(xp[7].add(8).cast()) };
                let x7_2 = unsafe { v128_load(xp[7].add(16).cast()) };
                let x7_3 = unsafe { v128_load(xp[7].add(24).cast()) };
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0 * K).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0_0, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1_0, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2_0, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3_0, wv));
                c40 = i32x4_add(c40, i32x4_dot_i16x8(x4_0, wv));
                c50 = i32x4_add(c50, i32x4_dot_i16x8(x5_0, wv));
                c60 = i32x4_add(c60, i32x4_dot_i16x8(x6_0, wv));
                c70 = i32x4_add(c70, i32x4_dot_i16x8(x7_0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0 * K + 8).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0_1, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1_1, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2_1, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3_1, wv));
                c40 = i32x4_add(c40, i32x4_dot_i16x8(x4_1, wv));
                c50 = i32x4_add(c50, i32x4_dot_i16x8(x5_1, wv));
                c60 = i32x4_add(c60, i32x4_dot_i16x8(x6_1, wv));
                c70 = i32x4_add(c70, i32x4_dot_i16x8(x7_1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0 * K + 16).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0_2, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1_2, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2_2, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3_2, wv));
                c40 = i32x4_add(c40, i32x4_dot_i16x8(x4_2, wv));
                c50 = i32x4_add(c50, i32x4_dot_i16x8(x5_2, wv));
                c60 = i32x4_add(c60, i32x4_dot_i16x8(x6_2, wv));
                c70 = i32x4_add(c70, i32x4_dot_i16x8(x7_2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0 * K + 24).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0_3, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1_3, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2_3, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3_3, wv));
                c40 = i32x4_add(c40, i32x4_dot_i16x8(x4_3, wv));
                c50 = i32x4_add(c50, i32x4_dot_i16x8(x5_3, wv));
                c60 = i32x4_add(c60, i32x4_dot_i16x8(x6_3, wv));
                c70 = i32x4_add(c70, i32x4_dot_i16x8(x7_3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0_0, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1_0, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2_0, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3_0, wv));
                c41 = i32x4_add(c41, i32x4_dot_i16x8(x4_0, wv));
                c51 = i32x4_add(c51, i32x4_dot_i16x8(x5_0, wv));
                c61 = i32x4_add(c61, i32x4_dot_i16x8(x6_0, wv));
                c71 = i32x4_add(c71, i32x4_dot_i16x8(x7_0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 8).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0_1, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1_1, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2_1, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3_1, wv));
                c41 = i32x4_add(c41, i32x4_dot_i16x8(x4_1, wv));
                c51 = i32x4_add(c51, i32x4_dot_i16x8(x5_1, wv));
                c61 = i32x4_add(c61, i32x4_dot_i16x8(x6_1, wv));
                c71 = i32x4_add(c71, i32x4_dot_i16x8(x7_1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 16).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0_2, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1_2, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2_2, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3_2, wv));
                c41 = i32x4_add(c41, i32x4_dot_i16x8(x4_2, wv));
                c51 = i32x4_add(c51, i32x4_dot_i16x8(x5_2, wv));
                c61 = i32x4_add(c61, i32x4_dot_i16x8(x6_2, wv));
                c71 = i32x4_add(c71, i32x4_dot_i16x8(x7_2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K + 24).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0_3, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1_3, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2_3, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3_3, wv));
                c41 = i32x4_add(c41, i32x4_dot_i16x8(x4_3, wv));
                c51 = i32x4_add(c51, i32x4_dot_i16x8(x5_3, wv));
                c61 = i32x4_add(c61, i32x4_dot_i16x8(x6_3, wv));
                c71 = i32x4_add(c71, i32x4_dot_i16x8(x7_3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0_0, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1_0, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2_0, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3_0, wv));
                c42 = i32x4_add(c42, i32x4_dot_i16x8(x4_0, wv));
                c52 = i32x4_add(c52, i32x4_dot_i16x8(x5_0, wv));
                c62 = i32x4_add(c62, i32x4_dot_i16x8(x6_0, wv));
                c72 = i32x4_add(c72, i32x4_dot_i16x8(x7_0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 8).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0_1, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1_1, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2_1, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3_1, wv));
                c42 = i32x4_add(c42, i32x4_dot_i16x8(x4_1, wv));
                c52 = i32x4_add(c52, i32x4_dot_i16x8(x5_1, wv));
                c62 = i32x4_add(c62, i32x4_dot_i16x8(x6_1, wv));
                c72 = i32x4_add(c72, i32x4_dot_i16x8(x7_1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 16).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0_2, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1_2, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2_2, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3_2, wv));
                c42 = i32x4_add(c42, i32x4_dot_i16x8(x4_2, wv));
                c52 = i32x4_add(c52, i32x4_dot_i16x8(x5_2, wv));
                c62 = i32x4_add(c62, i32x4_dot_i16x8(x6_2, wv));
                c72 = i32x4_add(c72, i32x4_dot_i16x8(x7_2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K + 24).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0_3, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1_3, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2_3, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3_3, wv));
                c42 = i32x4_add(c42, i32x4_dot_i16x8(x4_3, wv));
                c52 = i32x4_add(c52, i32x4_dot_i16x8(x5_3, wv));
                c62 = i32x4_add(c62, i32x4_dot_i16x8(x6_3, wv));
                c72 = i32x4_add(c72, i32x4_dot_i16x8(x7_3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0_0, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1_0, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2_0, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3_0, wv));
                c43 = i32x4_add(c43, i32x4_dot_i16x8(x4_0, wv));
                c53 = i32x4_add(c53, i32x4_dot_i16x8(x5_0, wv));
                c63 = i32x4_add(c63, i32x4_dot_i16x8(x6_0, wv));
                c73 = i32x4_add(c73, i32x4_dot_i16x8(x7_0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 8).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0_1, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1_1, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2_1, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3_1, wv));
                c43 = i32x4_add(c43, i32x4_dot_i16x8(x4_1, wv));
                c53 = i32x4_add(c53, i32x4_dot_i16x8(x5_1, wv));
                c63 = i32x4_add(c63, i32x4_dot_i16x8(x6_1, wv));
                c73 = i32x4_add(c73, i32x4_dot_i16x8(x7_1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 16).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0_2, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1_2, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2_2, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3_2, wv));
                c43 = i32x4_add(c43, i32x4_dot_i16x8(x4_2, wv));
                c53 = i32x4_add(c53, i32x4_dot_i16x8(x5_2, wv));
                c63 = i32x4_add(c63, i32x4_dot_i16x8(x6_2, wv));
                c73 = i32x4_add(c73, i32x4_dot_i16x8(x7_2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K + 24).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0_3, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1_3, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2_3, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3_3, wv));
                c43 = i32x4_add(c43, i32x4_dot_i16x8(x4_3, wv));
                c53 = i32x4_add(c53, i32x4_dot_i16x8(x5_3, wv));
                c63 = i32x4_add(c63, i32x4_dot_i16x8(x6_3, wv));
                c73 = i32x4_add(c73, i32x4_dot_i16x8(x7_3, wv));
                for rr in 0..8 { xp[rr] = unsafe { xp[rr].add(32) }; }
                wp = unsafe { wp.add(32) };
                p += 32;
            }
            while p < K {
                let x0v = unsafe { v128_load(xp[0].cast()) };
                let x1v = unsafe { v128_load(xp[1].cast()) };
                let x2v = unsafe { v128_load(xp[2].cast()) };
                let x3v = unsafe { v128_load(xp[3].cast()) };
                let x4v = unsafe { v128_load(xp[4].cast()) };
                let x5v = unsafe { v128_load(xp[5].cast()) };
                let x6v = unsafe { v128_load(xp[6].cast()) };
                let x7v = unsafe { v128_load(xp[7].cast()) };
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(0).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(x0v, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(x1v, wv));
                c20 = i32x4_add(c20, i32x4_dot_i16x8(x2v, wv));
                c30 = i32x4_add(c30, i32x4_dot_i16x8(x3v, wv));
                c40 = i32x4_add(c40, i32x4_dot_i16x8(x4v, wv));
                c50 = i32x4_add(c50, i32x4_dot_i16x8(x5v, wv));
                c60 = i32x4_add(c60, i32x4_dot_i16x8(x6v, wv));
                c70 = i32x4_add(c70, i32x4_dot_i16x8(x7v, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(1 * K).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(x0v, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(x1v, wv));
                c21 = i32x4_add(c21, i32x4_dot_i16x8(x2v, wv));
                c31 = i32x4_add(c31, i32x4_dot_i16x8(x3v, wv));
                c41 = i32x4_add(c41, i32x4_dot_i16x8(x4v, wv));
                c51 = i32x4_add(c51, i32x4_dot_i16x8(x5v, wv));
                c61 = i32x4_add(c61, i32x4_dot_i16x8(x6v, wv));
                c71 = i32x4_add(c71, i32x4_dot_i16x8(x7v, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(2 * K).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(x0v, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(x1v, wv));
                c22 = i32x4_add(c22, i32x4_dot_i16x8(x2v, wv));
                c32 = i32x4_add(c32, i32x4_dot_i16x8(x3v, wv));
                c42 = i32x4_add(c42, i32x4_dot_i16x8(x4v, wv));
                c52 = i32x4_add(c52, i32x4_dot_i16x8(x5v, wv));
                c62 = i32x4_add(c62, i32x4_dot_i16x8(x6v, wv));
                c72 = i32x4_add(c72, i32x4_dot_i16x8(x7v, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load(wp.add(3 * K).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(x0v, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(x1v, wv));
                c23 = i32x4_add(c23, i32x4_dot_i16x8(x2v, wv));
                c33 = i32x4_add(c33, i32x4_dot_i16x8(x3v, wv));
                c43 = i32x4_add(c43, i32x4_dot_i16x8(x4v, wv));
                c53 = i32x4_add(c53, i32x4_dot_i16x8(x5v, wv));
                c63 = i32x4_add(c63, i32x4_dot_i16x8(x6v, wv));
                c73 = i32x4_add(c73, i32x4_dot_i16x8(x7v, wv));
                for rr in 0..8 { xp[rr] = unsafe { xp[rr].add(8) }; }
                wp = unsafe { wp.add(8) };
                p += 8;
            }
            let reduce = |acc: v128| -> f32 {
                (i32x4_extract_lane::<0>(acc) + i32x4_extract_lane::<1>(acc)
                    + i32x4_extract_lane::<2>(acc) + i32x4_extract_lane::<3>(acc)) as f32
            };
            unsafe {
                *o[0].add(j + 0) = reduce(c00) * sx[i + 0] * sw[j + 0];
                *o[0].add(j + 1) = reduce(c01) * sx[i + 0] * sw[j + 1];
                *o[0].add(j + 2) = reduce(c02) * sx[i + 0] * sw[j + 2];
                *o[0].add(j + 3) = reduce(c03) * sx[i + 0] * sw[j + 3];
                *o[1].add(j + 0) = reduce(c10) * sx[i + 1] * sw[j + 0];
                *o[1].add(j + 1) = reduce(c11) * sx[i + 1] * sw[j + 1];
                *o[1].add(j + 2) = reduce(c12) * sx[i + 1] * sw[j + 2];
                *o[1].add(j + 3) = reduce(c13) * sx[i + 1] * sw[j + 3];
                *o[2].add(j + 0) = reduce(c20) * sx[i + 2] * sw[j + 0];
                *o[2].add(j + 1) = reduce(c21) * sx[i + 2] * sw[j + 1];
                *o[2].add(j + 2) = reduce(c22) * sx[i + 2] * sw[j + 2];
                *o[2].add(j + 3) = reduce(c23) * sx[i + 2] * sw[j + 3];
                *o[3].add(j + 0) = reduce(c30) * sx[i + 3] * sw[j + 0];
                *o[3].add(j + 1) = reduce(c31) * sx[i + 3] * sw[j + 1];
                *o[3].add(j + 2) = reduce(c32) * sx[i + 3] * sw[j + 2];
                *o[3].add(j + 3) = reduce(c33) * sx[i + 3] * sw[j + 3];
                *o[4].add(j + 0) = reduce(c40) * sx[i + 4] * sw[j + 0];
                *o[4].add(j + 1) = reduce(c41) * sx[i + 4] * sw[j + 1];
                *o[4].add(j + 2) = reduce(c42) * sx[i + 4] * sw[j + 2];
                *o[4].add(j + 3) = reduce(c43) * sx[i + 4] * sw[j + 3];
                *o[5].add(j + 0) = reduce(c50) * sx[i + 5] * sw[j + 0];
                *o[5].add(j + 1) = reduce(c51) * sx[i + 5] * sw[j + 1];
                *o[5].add(j + 2) = reduce(c52) * sx[i + 5] * sw[j + 2];
                *o[5].add(j + 3) = reduce(c53) * sx[i + 5] * sw[j + 3];
                *o[6].add(j + 0) = reduce(c60) * sx[i + 6] * sw[j + 0];
                *o[6].add(j + 1) = reduce(c61) * sx[i + 6] * sw[j + 1];
                *o[6].add(j + 2) = reduce(c62) * sx[i + 6] * sw[j + 2];
                *o[6].add(j + 3) = reduce(c63) * sx[i + 6] * sw[j + 3];
                *o[7].add(j + 0) = reduce(c70) * sx[i + 7] * sw[j + 0];
                *o[7].add(j + 1) = reduce(c71) * sx[i + 7] * sw[j + 1];
                *o[7].add(j + 2) = reduce(c72) * sx[i + 7] * sw[j + 2];
                *o[7].add(j + 3) = reduce(c73) * sx[i + 7] * sw[j + 3];
            }
            j += 4;
        }
        while j < n {
            for rr in 0..8 {
                let mut acc = 0i32;
                for p in 0..K { acc += unsafe { *r[rr].add(p) as i32 * *w.as_ptr().add(j * K + p) as i32 }; }
                unsafe { *o[rr].add(j) = acc as f32 * sx[i + rr] * sw[j] };
            }
            j += 1;
        }
        i += 8;
    }
    if m8 < m {
        unsafe { matmul_i8_simd_k4::<K>(&a[m8 * K..], w, &sx[m8..], sw, m - m8, n, &mut out[m8 * n..]) };
    }
}

/// Specialised on the contraction size, which is what lets the four weight rows share
/// one moving pointer.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn matmul_i8_simd_k<const K: usize>(a: &[i16], w: &[i8], sx: &[f32], sw: &[f32], m: usize, n: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    let n4 = n / 4 * 4;
    let mut i = 0;
    while i < m {
        let arow = unsafe { a.as_ptr().add(i * K) };
        let orow = unsafe { out.as_mut_ptr().add(i * n) };
        let mut j = 0;
        while j < n4 {
            let mut acc0 = i32x4_splat(0);
            let mut acc1 = i32x4_splat(0);
            let mut acc2 = i32x4_splat(0);
            let mut acc3 = i32x4_splat(0);
            let mut xp = arow;
            let mut wp = unsafe { w.as_ptr().add(j * K) };
            let mut p = 0;
            while p + 32 <= K {
                let x0 = unsafe { v128_load(xp.cast()) };
                let x1 = unsafe { v128_load(xp.add(8).cast()) };
                let x2 = unsafe { v128_load(xp.add(16).cast()) };
                let x3 = unsafe { v128_load(xp.add(24).cast()) };
                acc0 = unsafe { dot4_i8(wp, x0, x1, x2, x3, acc0) };
                acc1 = unsafe { dot4_i8(wp.add(K), x0, x1, x2, x3, acc1) };
                acc2 = unsafe { dot4_i8(wp.add(2 * K), x0, x1, x2, x3, acc2) };
                acc3 = unsafe { dot4_i8(wp.add(3 * K), x0, x1, x2, x3, acc3) };
                xp = unsafe { xp.add(32) };
                wp = unsafe { wp.add(32) };
                p += 32;
            }
            while p < K {
                let xv = unsafe { v128_load(xp.cast()) };
                acc0 = unsafe { i32x4_add(acc0, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load(wp.cast()) }))) };
                acc1 = unsafe { i32x4_add(acc1, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load(wp.add(K).cast()) }))) };
                acc2 = unsafe { i32x4_add(acc2, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load(wp.add(2 * K).cast()) }))) };
                acc3 = unsafe { i32x4_add(acc3, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load(wp.add(3 * K).cast()) }))) };
                xp = unsafe { xp.add(8) };
                wp = unsafe { wp.add(8) };
                p += 8;
            }
            let reduce = |acc: v128| -> f32 {
                (i32x4_extract_lane::<0>(acc) + i32x4_extract_lane::<1>(acc)
                    + i32x4_extract_lane::<2>(acc) + i32x4_extract_lane::<3>(acc)) as f32
            };
            unsafe {
                *orow.add(j) = reduce(acc0) * sx[i] * sw[j];
                *orow.add(j + 1) = reduce(acc1) * sx[i] * sw[j + 1];
                *orow.add(j + 2) = reduce(acc2) * sx[i] * sw[j + 2];
                *orow.add(j + 3) = reduce(acc3) * sx[i] * sw[j + 3];
            }
            j += 4;
        }
        while j < n {
            let mut acc = 0i32;
            for p in 0..K {
                acc += unsafe { *arow.add(p) as i32 * *w.as_ptr().add(j * K + p) as i32 };
            }
            unsafe { *orow.add(j) = acc as f32 * sx[i] * sw[j] };
            j += 1;
        }
        i += 1;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn dot4_i8(i8w: *const i8, x0: core::arch::wasm32::v128, x1: core::arch::wasm32::v128, x2: core::arch::wasm32::v128, x3: core::arch::wasm32::v128, acc: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    // The four loads use constant offsets, so the address arithmetic happens once per
    // four k-chunks instead of once per load. Measured: this is what took the kernel
    // from 1.605 to 0.9 instructions/MAC (docs/VERDICT_ENGINE.md 5.1.7).
    let mut a = acc;
    a = i32x4_add(a, i32x4_dot_i16x8(x0, i16x8_extend_low_i8x16(unsafe { v128_load(i8w.cast()) })));
    a = i32x4_add(a, i32x4_dot_i16x8(x1, i16x8_extend_low_i8x16(unsafe { v128_load(i8w.add(8).cast()) })));
    a = i32x4_add(a, i32x4_dot_i16x8(x2, i16x8_extend_low_i8x16(unsafe { v128_load(i8w.add(16).cast()) })));
    a = i32x4_add(a, i32x4_dot_i16x8(x3, i16x8_extend_low_i8x16(unsafe { v128_load(i8w.add(24).cast()) })));
    a
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn matmul_i8_simd(a: &[i16], w: &[i8], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    let n4 = n / 4 * 4;
    let mut i = 0;
    while i < m {
        let arow = unsafe { a.as_ptr().add(i * k) };
        let orow = unsafe { out.as_mut_ptr().add(i * n) };
        let mut j = 0;
        while j < n4 {
            let mut acc0 = i32x4_splat(0);
            let mut acc1 = i32x4_splat(0);
            let mut acc2 = i32x4_splat(0);
            let mut acc3 = i32x4_splat(0);
            let w0 = unsafe { w.as_ptr().add(j * k) };
            let mut p0 = w0;
            let mut p1 = unsafe { w0.add(k) };
            let mut p2 = unsafe { w0.add(2 * k) };
            let mut p3 = unsafe { w0.add(3 * k) };
            let mut xp = arow;
            let mut p = 0;
            while p + 32 <= k {
                // SAFETY: `k % 8 == 0` and `p + 32 <= k` keep every 8-wide window inside
                // the row for both the activations and each of the four weight rows.
                let x0 = unsafe { v128_load(xp.cast()) };
                let x1 = unsafe { v128_load(xp.add(8).cast()) };
                let x2 = unsafe { v128_load(xp.add(16).cast()) };
                let x3 = unsafe { v128_load(xp.add(24).cast()) };
                acc0 = unsafe { dot4_i8(p0, x0, x1, x2, x3, acc0) };
                acc1 = unsafe { dot4_i8(p1, x0, x1, x2, x3, acc1) };
                acc2 = unsafe { dot4_i8(p2, x0, x1, x2, x3, acc2) };
                acc3 = unsafe { dot4_i8(p3, x0, x1, x2, x3, acc3) };
                xp = unsafe { xp.add(32) };
                p0 = unsafe { p0.add(32) };
                p1 = unsafe { p1.add(32) };
                p2 = unsafe { p2.add(32) };
                p3 = unsafe { p3.add(32) };
                p += 32;
            }
            while p < k {
                let xv = unsafe { v128_load(xp.cast()) };
                acc0 = unsafe { i32x4_add(acc0, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load(p0.cast()) }))) };
                acc1 = unsafe { i32x4_add(acc1, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load(p1.cast()) }))) };
                acc2 = unsafe { i32x4_add(acc2, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load(p2.cast()) }))) };
                acc3 = unsafe { i32x4_add(acc3, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load(p3.cast()) }))) };
                xp = unsafe { xp.add(8) };
                p0 = unsafe { p0.add(8) };
                p1 = unsafe { p1.add(8) };
                p2 = unsafe { p2.add(8) };
                p3 = unsafe { p3.add(8) };
                p += 8;
            }
            let reduce = |acc: v128| -> f32 {
                (i32x4_extract_lane::<0>(acc) + i32x4_extract_lane::<1>(acc)
                    + i32x4_extract_lane::<2>(acc) + i32x4_extract_lane::<3>(acc)) as f32
            };
            unsafe {
                *orow.add(j) = reduce(acc0) * sx[i] * sw[j];
                *orow.add(j + 1) = reduce(acc1) * sx[i] * sw[j + 1];
                *orow.add(j + 2) = reduce(acc2) * sx[i] * sw[j + 2];
                *orow.add(j + 3) = reduce(acc3) * sx[i] * sw[j + 3];
            }
            j += 4;
        }
        while j < n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += unsafe { *arow.add(p) as i32 * *w.as_ptr().add(j * k + p) as i32 };
            }
            unsafe { *orow.add(j) = acc as f32 * sx[i] * sw[j] };
            j += 1;
        }
        i += 1;
    }
}

// --- softmax ---------------------------------------------------------------------

/// In-place softmax over each row of a `[rows, cols]` buffer.
///
/// `exp` stays the same `f32::exp` call, so the values are unchanged; the max, the sum
/// of the exponentials and the final division are vectorised, which is what the fused
/// attention softmax was still spending its instructions on.
pub fn softmax_rows_inplace(data: &mut [f32], rows: usize, cols: usize) {
    assert!(data.len() >= rows * cols);
    for r in 0..rows {
        let row = &mut data[r * cols..r * cols + cols];
        #[cfg(target_arch = "wasm32")]
        if cols % 4 == 0 {
            // SAFETY: `cols % 4 == 0` keeps every 4-wide access inside the row.
            unsafe { softmax_row_simd(row) };
            continue;
        }
        softmax_row_scalar(row);
    }
}

/// Reference implementation, also the native path.
pub fn softmax_row_scalar(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for v in row.iter() {
        if *v > max {
            max = *v;
        }
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn softmax_row_simd(row: &mut [f32]) {
    use core::arch::wasm32::*;
    let n = row.len();
    let mut mv = f32x4_splat(f32::NEG_INFINITY);
    let mut p = 0;
    while p + 4 <= n {
        mv = f32x4_max(mv, unsafe { v128_load(row.as_ptr().add(p).cast()) });
        p += 4;
    }
    let m = f32x4_extract_lane::<0>(mv)
        .max(f32x4_extract_lane::<1>(mv))
        .max(f32x4_extract_lane::<2>(mv))
        .max(f32x4_extract_lane::<3>(mv));
    // `exp` is a libm call either way, so this loop stays scalar: only the arithmetic
    // around it is vectorised.
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let d = f32x4_splat(sum);
    let mut p = 0;
    while p + 4 <= n {
        let x = unsafe { v128_load(row.as_ptr().add(p).cast()) };
        unsafe { v128_store(row.as_mut_ptr().add(p).cast(), f32x4_div(x, d)) };
        p += 4;
    }
}

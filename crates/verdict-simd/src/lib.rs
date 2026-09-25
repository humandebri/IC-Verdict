//! Hand-written wasm SIMD integer kernels for the openJev dense matmuls.
//!
//! ## What is here
//!
//! `quantize_rows_i8` / `quantize_acts_i16` turn f32 weights and activations into the
//! i8 / i16 pair the kernel consumes, `matmul_i8` runs the blocked `i32x4.dot_i16x8`
//! kernel (with a scalar reference for odd shapes and for hosts), and
//! `softmax_rows_inplace` is the fused attention softmax.
//!
//! Measured in the canister on the real checkpoint: the int8 kernel runs at **0.780
//! instructions/MAC** against gemm's f32 2.501, which is what took a 120-token
//! decision from 37.1e9 to 14.6e9 instructions. Every number and every rejected
//! approach is recorded in `docs/VERDICT_ENGINE.md` 5.1.
//!
//! ## What was removed, and why
//!
//! This crate used to carry an f32 `f32x4` kernel family, a 16-byte alignment helper
//! (`AlignedF32`) and `simd_probe_*` codegen probes. Only the crate's own benchmarks
//! called them, and `AlignedF32::from_slice` silently returned shifted data on wasm32
//! (it aligned the pointer without moving the values). The model path never used any
//! of it, so the family and the alignment machinery are gone rather than fixed. The
//! int8 path needs no alignment precondition: wasm permits unaligned vector loads.
//!
//! ## Safety
//!
//! `simd128` instructions are accepted by the IC's Wasm validator and execute on ICP
//! (gemm's own SIMD kernels are in the deployed module). The intrinsics are `unsafe`
//! only because Rust requires that of `core::arch`; every call site is an `unsafe fn`
//! whose callers bound the pointers by slice length, and `matmul_i8` checks that the
//! slices cover the shape before dispatching. Weight steps read exactly eight bytes
//! (`v128_load64_zero`), never past the end of a row.
#![deny(unsafe_op_in_unsafe_fn)]

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
/// The scale uses the row's maximum magnitude. Production packs are quantised offline,
/// and avoiding clipping is the conservative choice for the 1000-case parity gate.
pub fn quantize_rows_i8(src: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(src.len() >= rows * cols, "input shorter than rows*cols");
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![1.0f32; rows];
    if rows == 0 || cols == 0 {
        return (q, scales);
    }
    for r in 0..rows {
        let row = &src[r * cols..r * cols + cols];
        let max = row.iter().fold(0.0f32,|a,v|a.max(v.abs()));
        let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
        scales[r] = scale;
        for (i, v) in row.iter().enumerate() {
            q[r * cols + i] = (v / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

/// Symmetric INT8 with one scale per `block` values in each row.
pub fn quantize_blocks_i8(src:&[f32],rows:usize,cols:usize,block:usize)->(Vec<i8>,Vec<f32>){
    assert!(block>0&&src.len()>=rows*cols,"invalid blocked quantisation shape");
    let blocks=cols.div_ceil(block);let mut q=vec![0i8;rows*cols];let mut scales=vec![1.0f32;rows*blocks];
    for r in 0..rows{for b in 0..blocks{let begin=b*block;let end=(begin+block).min(cols);let row=&src[r*cols+begin..r*cols+end];let max=row.iter().fold(0.0f32,|a,v|a.max(v.abs()));let scale=if max>0.0{max/127.0}else{1.0};scales[r*blocks+b]=scale;for(i,v)in row.iter().enumerate(){q[r*cols+begin+i]=(v/scale).round().clamp(-127.0,127.0)as i8;}}}
    (q,scales)
}

/// Block-scaled variant used by production packs. Activations retain one high-precision
/// i16 scale per input row; weight sums are rescaled after every block.
#[allow(clippy::too_many_arguments)] // Kernel shape and buffers are explicit.
pub fn matmul_i8_blocked(a:&[i16],w:&[i8],sx:&[f32],sw:&[f32],m:usize,k:usize,n:usize,block:usize,out:&mut[f32])->bool{
    assert!(block>0,"block must be nonzero");
    let blocks=k.div_ceil(block);
    let mk=m.checked_mul(k).expect("shape overflow");let nk=n.checked_mul(k).expect("shape overflow");
    let mn=m.checked_mul(n).expect("shape overflow");let nb=n.checked_mul(blocks).expect("shape overflow");
    assert!(a.len()>=mk&&w.len()>=nk&&sx.len()>=m&&sw.len()>=nb&&out.len()>=mn);
    if m==0||n==0||k==0{out[..mn].fill(0.0);return false;}
    #[cfg(target_arch="wasm32")]
    if block==32&&k.is_multiple_of(32){unsafe{matmul_i8_block32_simd(a,w,sx,sw,m,k,n,out)};return true;}
    #[cfg(not(target_arch="wasm32"))]
    {
        use rayon::prelude::*;
        out[..m*n].par_chunks_mut(n).enumerate().for_each(|(i,row)|{for j in 0..n{let mut total=0.0f32;for b in 0..blocks{let begin=b*block;let end=(begin+block).min(k);let mut acc=0i32;for p in begin..end{acc+=(a[i*k+p]as i32)*(w[j*k+p]as i32);}total+=acc as f32*sw[j*blocks+b];}row[j]=total*sx[i];}});
        false
    }
    #[cfg(target_arch="wasm32")]
    for i in 0..m{for j in 0..n{let mut total=0.0f32;for b in 0..blocks{let begin=b*block;let end=(begin+block).min(k);let mut acc=0i32;for p in begin..end{acc+=(a[i*k+p]as i32)*(w[j*k+p]as i32);}total+=acc as f32*sw[j*blocks+b];}out[i*n+j]=total*sx[i];}}
    #[cfg(target_arch="wasm32")]
    {false}
}

#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_i8_block32_simd(a:&[i16],w:&[i8],sx:&[f32],sw:&[f32],m:usize,k:usize,n:usize,out:&mut[f32]){
    use core::arch::wasm32::*;let blocks=k/32;
    for i in 0..m{for j in 0..n{let mut total=0.0f32;for b in 0..blocks{let ap=unsafe{a.as_ptr().add(i*k+b*32)};let wp=unsafe{w.as_ptr().add(j*k+b*32)};let mut acc=i32x4_splat(0);for p in [0usize,8,16,24]{let av=unsafe{v128_load(ap.add(p).cast())};let wv=i16x8_extend_low_i8x16(unsafe{v128_load64_zero(wp.add(p).cast())});acc=i32x4_add(acc,i32x4_dot_i16x8(av,wv));}let sum=i32x4_extract_lane::<0>(acc)+i32x4_extract_lane::<1>(acc)+i32x4_extract_lane::<2>(acc)+i32x4_extract_lane::<3>(acc);total+=sum as f32*sw[j*blocks+b];}out[i*n+j]=total*sx[i];}}
}

/// Measurement candidate only: production keeps the validated block32 kernel.
/// `tile` selects the number of input rows sharing each widened weight vector.
#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_block32_candidate(a:&[i16],w:&[i8],sx:&[f32],sw:&[f32],m:usize,k:usize,n:usize,out:&mut[f32],tile:usize)->bool{
    assert!(k>0&&k.is_multiple_of(32)&&[2,4,8].contains(&tile));
    let mk=m.checked_mul(k).expect("shape overflow");let nk=n.checked_mul(k).expect("shape overflow");
    let mn=m.checked_mul(n).expect("shape overflow");let nb=n.checked_mul(k/32).expect("shape overflow");
    assert!(a.len()>=mk&&w.len()>=nk&&sx.len()>=m&&sw.len()>=nb&&out.len()>=mn);
    if m==0||n==0{return false;}
    #[cfg(target_arch="wasm32")]
    {
        // SAFETY: shape checks above cover all rows and each 32-element block.
        unsafe{match tile{
            2=>block32_tile::<2>(a,w,sx,sw,m,k,n,out),
            4=>block32_tile::<4>(a,w,sx,sw,m,k,n,out),
            8=>block32_tile::<8>(a,w,sx,sw,m,k,n,out),_=>unreachable!(),
        }}
        true
    }
    #[cfg(not(target_arch="wasm32"))]
    {matmul_i8_blocked(a,w,sx,sw,m,k,n,32,out)}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // Row/column indices address multiple tile buffers.
unsafe fn block32_tile<const M:usize>(a:&[i16],w:&[i8],sx:&[f32],sw:&[f32],m:usize,k:usize,n:usize,out:&mut[f32]){
    use core::arch::wasm32::*;
    let blocks=k/32;
    for i in (0..m).step_by(M){for j in (0..n).step_by(4){
        let rows=M.min(m-i);let cols=4.min(n-j);let mut total=[[0f32;4];M];
        for b in 0..blocks{
            let mut acc=[[i32x4_splat(0);4];M];
            for p in [0usize,8,16,24]{
                let mut xs=[i32x4_splat(0);M];
                for r in 0..rows{xs[r]=unsafe{v128_load(a.as_ptr().add((i+r)*k+b*32+p).cast())};}
                for c in 0..cols{
                    let wv=i16x8_extend_low_i8x16(unsafe{v128_load64_zero(w.as_ptr().add((j+c)*k+b*32+p).cast())});
                    for r in 0..rows{acc[r][c]=i32x4_add(acc[r][c],i32x4_dot_i16x8(xs[r],wv));}
                }
            }
            for r in 0..rows{for c in 0..cols{
                let v=acc[r][c];let sum=i32x4_extract_lane::<0>(v)+i32x4_extract_lane::<1>(v)+i32x4_extract_lane::<2>(v)+i32x4_extract_lane::<3>(v);
                total[r][c]+=sum as f32*sw[(j+c)*blocks+b];
            }}
        }
        for r in 0..rows{for c in 0..cols{out[(i+r)*n+j+c]=total[r][c]*sx[i+r];}}
    }}
}

/// Measurement-only sliding-window core, [heads,tokens,dim]. Global attention
/// and production inference are deliberately not routed here before acceptance.
#[allow(clippy::too_many_arguments)]
pub fn local_attention_candidate(q:&[f32],k:&[f32],v:&[f32],heads:usize,t:usize,d:usize,distance:usize)->Vec<f32>{
    assert!(d>0&&q.len()==heads*t*d&&k.len()==q.len()&&v.len()==q.len());
    let mut out=vec![0f32;q.len()];let scale=(1.0/(d as f64).sqrt()) as f32;
    let mut scores=Vec::with_capacity(t.min(distance.saturating_mul(2).saturating_add(1)));
    for h in 0..heads{for i in 0..t{
        let lo=i.saturating_sub(distance);let hi=t.min(i.saturating_add(distance).saturating_add(1));
        scores.clear();
        for j in lo..hi{let mut dot=0f32;for c in 0..d{dot+=q[(h*t+i)*d+c]*k[(h*t+j)*d+c];}scores.push(dot*scale);}
        softmax_row_scalar(&mut scores);
        for (j,&weight) in (lo..hi).zip(&scores){for c in 0..d{out[(h*t+i)*d+c]+=weight*v[(h*t+j)*d+c];}}
    }}
    out
}

/// Activation quantisation into the i16 range.
///
/// The activations were already stored as i16, so using the full i16 range instead of
/// the i8 range costs nothing at inference and gives 64x finer resolution. The range is
/// bounded by the i32 accumulator: `range * 127 * k` must stay below `2^31`
/// (127 * 1152 = 146304, so anything up to ~14600 is safe).
pub const ACTIVATION_RANGE: f32 = 8192.0;

pub fn quantize_acts_i16(src: &[f32], rows: usize, cols: usize) -> (Vec<i16>, Vec<f32>) {
    assert!(src.len() >= rows * cols, "input shorter than rows*cols");
    let mut q = vec![0i16; rows * cols];
    let mut scales = vec![1.0f32; rows];
    if rows == 0 || cols == 0 {
        return (q, scales);
    }
    // The i32 accumulator has to hold `sum |a*w| <= range * 127 * cols`. 8192 is the
    // precision the kernel was measured at for k=768/1152 (0.780 instructions/MAC);
    // a larger contraction gets a smaller range instead of wrapping silently.
    let range = ACTIVATION_RANGE
        .min(((i32::MAX as f32) / (127.0 * cols as f32)).floor())
        .max(1.0);
    for r in 0..rows {
        let row = &src[r * cols..r * cols + cols];
        let max = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        // Dividing by the scale costs a full division per element and measured 167
        // instructions per element; multiplying by the reciprocal and rounding with
        // `f32x4.nearest` in blocks of eight is a small fraction of that. The scalar
        // path uses `round_ties_even` so both paths round identically.
        let inv_scale = if max > 0.0 { range / max } else { 1.0 };
        scales[r] = if max > 0.0 { max / range } else { 1.0 };
        let out = &mut q[r * cols..r * cols + cols];
        #[cfg(target_arch = "wasm32")]
        if cols.is_multiple_of(8) {
            // SAFETY: the loop runs while `p + 8 <= cols`, so both the 8-wide load and
            // the 8-wide store stay inside the row slices.
            unsafe { quantize_row_i16_simd(row, out, inv_scale) };
            continue;
        }
        for (i, v) in row.iter().enumerate() {
            out[i] = (v * inv_scale).round_ties_even().clamp(-range, range) as i16;
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


/// `out[m, n] = a[m, k] · w[n, k]ᵀ` with i16 activations and i8 weights.
///
/// Returns `true` when the wasm SIMD path ran; the scalar path is the reference used
/// by the native tests.
#[must_use]
#[allow(clippy::too_many_arguments)] // Hot kernel ABI keeps dimensions and buffers explicit.
pub fn matmul_i8(a: &[i16], w: &[i8], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) -> bool {
    let mk=m.checked_mul(k).expect("shape overflow");
    let nk=n.checked_mul(k).expect("shape overflow");
    let mn=m.checked_mul(n).expect("shape overflow");
    assert!(a.len() >= mk && w.len() >= nk && sx.len() >= m && sw.len() >= n && out.len() >= mn);
    #[cfg(target_arch = "wasm32")]
    {
        if k.is_multiple_of(8) {
            // SAFETY: see the module-level note; `k.is_multiple_of(8)` keeps every 8-wide load
            // inside one row and all pointers come from slices checked above.
            match k {
                768 => unsafe {
                    if n.is_multiple_of(16) { matmul_i8_simd_tile::<768,16,16>(a, w, sx, sw, m, n, out) }
                    else { matmul_i8_simd_tile::<768,16,8>(a, w, sx, sw, m, n, out) }
                },
                1152 => unsafe {
                    if n.is_multiple_of(16) { matmul_i8_simd_tile::<1152,16,16>(a, w, sx, sw, m, n, out) }
                    else { matmul_i8_simd_tile::<1152,16,8>(a, w, sx, sw, m, n, out) }
                },
                _ => unsafe { matmul_i8_simd(a, w, sx, sw, m, k, n, out) },
            }
            return true;
        }
    }
    matmul_i8_scalar(a, w, sx, sw, m, k, n, out);
    false
}

/// Reference implementation of the int8 kernel (also the native path).
#[allow(clippy::too_many_arguments)] // Scalar reference intentionally mirrors the SIMD ABI.
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
#[allow(clippy::too_many_arguments)]
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
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(0).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xa0, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xb0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(8).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xa1, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xb1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(16).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xa2, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xb2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(24).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xa3, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xb3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(K).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xa0, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xb0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(K + 8).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xa1, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xb1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(K + 16).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xa2, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xb2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(K + 24).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xa3, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xb3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(2 * K).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xa0, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xb0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(2 * K + 8).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xa1, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xb1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(2 * K + 16).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xa2, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xb2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(2 * K + 24).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xa3, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xb3, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(3 * K).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(xa0, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(xb0, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(3 * K + 8).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(xa1, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(xb1, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(3 * K + 16).cast())) };
                c03 = i32x4_add(c03, i32x4_dot_i16x8(xa2, wv));
                c13 = i32x4_add(c13, i32x4_dot_i16x8(xb2, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(3 * K + 24).cast())) };
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
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(0).cast())) };
                c00 = i32x4_add(c00, i32x4_dot_i16x8(xav, wv));
                c10 = i32x4_add(c10, i32x4_dot_i16x8(xbv, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(K).cast())) };
                c01 = i32x4_add(c01, i32x4_dot_i16x8(xav, wv));
                c11 = i32x4_add(c11, i32x4_dot_i16x8(xbv, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(2 * K).cast())) };
                c02 = i32x4_add(c02, i32x4_dot_i16x8(xav, wv));
                c12 = i32x4_add(c12, i32x4_dot_i16x8(xbv, wv));
                let wv = unsafe { i16x8_extend_low_i8x16(v128_load64_zero(wp.add(3 * K).cast())) };
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
    // The final row uses the same bounded eight-lane loads as the main tiles.
    if i < m {
        // SAFETY: i < m, and K is 768 or 1152; sliced buffers cover m-i rows.
        unsafe { matmul_i8_simd(&a[i * K..], w, &sx[i..], sw, m - i, K, n, &mut out[i * n..]) };
    }
}

/// Share loaded vectors across 16/8/4 rows and 16 columns. Other widths retain
/// eight-column tiles so small matrices do not fall into a large scalar tail.
/// K is 768 or 1152 (both multiples of 64); R is 16, 8, or 4; C is 16 or 8.
/// These are selected only by the dispatch and tail calls below. Literal
/// macro indices let LLVM retain accumulators in Wasm locals, not a memory array.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable="simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_i8_simd_tile<const K:usize,const R:usize,const C:usize>(a:&[i16],w:&[i8],sx:&[f32],sw:&[f32],m:usize,n:usize,out:&mut[f32]) {
    use core::arch::wasm32::*;
    let full=m/R*R;let nc=n/C*C;
    for i in (0..full).step_by(R) {
        for j in (0..nc).step_by(C) {
            let mut acc=[[i32x4_splat(0);16];16];
            for p in (0..K).step_by(64) {
                let mut x=[[i32x4_splat(0);8];16];
                macro_rules! load_rows {($($r:literal),*)=>{$(
                    if $r<R {
                    // SAFETY: i+r < full <= m and p+64 <= K.
                    let ap=unsafe{a.as_ptr().add((i+$r)*K+p)};
                    x[$r]=unsafe{[
                        v128_load(ap.cast()),v128_load(ap.add(8).cast()),
                        v128_load(ap.add(16).cast()),v128_load(ap.add(24).cast()),
                        v128_load(ap.add(32).cast()),v128_load(ap.add(40).cast()),
                        v128_load(ap.add(48).cast()),v128_load(ap.add(56).cast()),
                    ]};
                    }
                )*};}
                load_rows!(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
                macro_rules! accumulate_rows {($c:literal,$w0:ident,$w1:ident,$w2:ident,$w3:ident,$w4:ident,$w5:ident,$w6:ident,$w7:ident;$($r:literal),*)=>{$(
                    if $r<R {acc[$r][$c]=i32x4_add(i32x4_add(i32x4_add(i32x4_add(acc[$r][$c],
                        i32x4_dot_i16x8(x[$r][0],$w0)),i32x4_dot_i16x8(x[$r][1],$w1)),
                        i32x4_dot_i16x8(x[$r][2],$w2)),i32x4_dot_i16x8(x[$r][3],$w3));
                    acc[$r][$c]=i32x4_add(i32x4_add(i32x4_add(i32x4_add(acc[$r][$c],
                        i32x4_dot_i16x8(x[$r][4],$w4)),i32x4_dot_i16x8(x[$r][5],$w5)),
                        i32x4_dot_i16x8(x[$r][6],$w6)),i32x4_dot_i16x8(x[$r][7],$w7));}
                )*};}
                macro_rules! columns {($($c:literal),*)=>{$(
                    if $c<C {
                    // SAFETY: j+c < nc <= n; each load reads eight bytes.
                    let wp=unsafe{w.as_ptr().add((j+$c)*K+p)};
                    let w0=unsafe{i16x8_extend_low_i8x16(v128_load64_zero(wp.cast()))};
                    let w1=unsafe{i16x8_extend_low_i8x16(v128_load64_zero(wp.add(8).cast()))};
                    let w2=unsafe{i16x8_extend_low_i8x16(v128_load64_zero(wp.add(16).cast()))};
                    let w3=unsafe{i16x8_extend_low_i8x16(v128_load64_zero(wp.add(24).cast()))};
                    let w4=unsafe{i16x8_extend_low_i8x16(v128_load64_zero(wp.add(32).cast()))};
                    let w5=unsafe{i16x8_extend_low_i8x16(v128_load64_zero(wp.add(40).cast()))};
                    let w6=unsafe{i16x8_extend_low_i8x16(v128_load64_zero(wp.add(48).cast()))};
                    let w7=unsafe{i16x8_extend_low_i8x16(v128_load64_zero(wp.add(56).cast()))};
                    accumulate_rows!($c,w0,w1,w2,w3,w4,w5,w6,w7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
                    }
                )*};}
                columns!(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
            }
            let reduce=|v| (i32x4_extract_lane::<0>(v)+i32x4_extract_lane::<1>(v)+i32x4_extract_lane::<2>(v)+i32x4_extract_lane::<3>(v)) as f32;
            macro_rules! store_row {($r:literal;$($c:literal),*)=>{$(
                // SAFETY: full tile bounds and caller-checked scale/output slices.
                if $r<R && $c<C {unsafe{*out.as_mut_ptr().add((i+$r)*n+j+$c)=reduce(acc[$r][$c]) * *sx.get_unchecked(i+$r) * *sw.get_unchecked(j+$c)};}
            )*};}
            macro_rules! store_rows {($($r:literal),*)=>{$(store_row!($r;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);)*};}
            store_rows!(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        }
        for j in nc..n {for r in 0..R {
            let mut sum=0i32;
            for p in 0..K {sum+=unsafe{*a.get_unchecked((i+r)*K+p) as i32 * *w.get_unchecked(j*K+p) as i32};}
            unsafe{*out.get_unchecked_mut((i+r)*n+j)=sum as f32 * *sx.get_unchecked(i+r) * *sw.get_unchecked(j)};
        }}
    }
    if full<m {
        // SAFETY: each sliced buffer covers the remaining m-full rows.
        let a=&a[full*K..];let sx=&sx[full..];let out=&mut out[full*n..];
        let remaining=m-full;
        // A tile missing one row costs less than 8+4+2+1 / 4+2+1 / 2+1.
        // Pad only this quantized matrix product, never the model token sequence:
        // zero scratch rows cannot affect any real row's dot product or scale.
        if remaining==R-1 {
            let mut padded_a=vec![0i16;R*K];
            padded_a[..remaining*K].copy_from_slice(&a[..remaining*K]);
            let mut padded_sx=vec![1f32;R];
            padded_sx[..remaining].copy_from_slice(&sx[..remaining]);
            let mut padded_out=vec![0f32;R*n];
            // SAFETY: full R-row scratch buffers, unchanged checked weights/scales.
            unsafe{matmul_i8_simd_tile::<K,R,C>(&padded_a,w,&padded_sx,sw,R,n,&mut padded_out)};
            out[..remaining*n].copy_from_slice(&padded_out[..remaining*n]);
            return;
        }
        unsafe{if R==16 {matmul_i8_simd_tile::<K,8,C>(a,w,sx,sw,m-full,n,out)}
            else if R==8 {matmul_i8_simd_tile::<K,4,C>(a,w,sx,sw,m-full,n,out)}
            else {matmul_i8_simd_k2::<K>(a,w,sx,sw,m-full,n,out)}};
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
    a = i32x4_add(a, i32x4_dot_i16x8(x0, i16x8_extend_low_i8x16(unsafe { v128_load64_zero(i8w.cast()) })));
    a = i32x4_add(a, i32x4_dot_i16x8(x1, i16x8_extend_low_i8x16(unsafe { v128_load64_zero(i8w.add(8).cast()) })));
    a = i32x4_add(a, i32x4_dot_i16x8(x2, i16x8_extend_low_i8x16(unsafe { v128_load64_zero(i8w.add(16).cast()) })));
    a = i32x4_add(a, i32x4_dot_i16x8(x3, i16x8_extend_low_i8x16(unsafe { v128_load64_zero(i8w.add(24).cast()) })));
    a
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
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
                // SAFETY: `k.is_multiple_of(8)` and `p + 32 <= k` keep every 8-wide window inside
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
                acc0 = i32x4_add(acc0, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load64_zero(p0.cast()) })));
                acc1 = i32x4_add(acc1, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load64_zero(p1.cast()) })));
                acc2 = i32x4_add(acc2, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load64_zero(p2.cast()) })));
                acc3 = i32x4_add(acc3, i32x4_dot_i16x8(xv, i16x8_extend_low_i8x16(unsafe { v128_load64_zero(p3.cast()) })));
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
        if cols.is_multiple_of(4) {
            // SAFETY: `cols.is_multiple_of(4)` keeps every 4-wide access inside the row.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block32_matches_i64_reference_including_partial_blocks_and_tiles(){
        for (m,k,n) in [(1usize,16usize,1usize),(3,33,5),(7,64,9),(9,96,7)]{
            let a:Vec<i16>=(0..m*k).map(|i|(i%16385) as i16-8192).collect();
            let w:Vec<i8>=(0..n*k).map(|i|(i%255) as i16-127).map(|x|x as i8).collect();
            let sx=vec![0.00013;m];let sw:Vec<f32>=(0..n*k.div_ceil(32)).map(|i|0.001*(1+i%7) as f32).collect();
            let mut want=vec![0.;m*n];
            for i in 0..m{for j in 0..n{
                let mut total=0f32;
                for b in 0..k.div_ceil(32){let acc:i64=(b*32..((b+1)*32).min(k)).map(|p|a[i*k+p] as i64*w[j*k+p] as i64).sum();total+=acc as f32*sw[j*k.div_ceil(32)+b];}
                want[i*n+j]=total*sx[i];
            }}
            let mut got=vec![0.;m*n];let _=matmul_i8_blocked(&a,&w,&sx,&sw,m,k,n,32,&mut got);assert_eq!(got,want);
            if k.is_multiple_of(32){for tile in [2,4,8]{let _=matmul_i8_block32_candidate(&a,&w,&sx,&sw,m,k,n,&mut got,tile);assert_eq!(got,want);}}
        }
    }
    #[test]
    fn sliding_attention_boundaries_are_uniform_for_zero_scores(){
        for t in [1,3,8]{for distance in [0,1,4,20]{
            let q=vec![0.;2*t*4];let v:Vec<f32>=(0..2*t*4).map(|i|i as f32).collect();
            let got=local_attention_candidate(&q,&q,&v,2,t,4,distance);
            for h in 0..2{for i in 0..t{for c in 0..4{
                let lo=i.saturating_sub(distance);let hi=t.min(i+distance+1);
                let expected=(lo..hi).map(|j|v[(h*t+j)*4+c]).sum::<f32>()/(hi-lo) as f32;
                assert!((got[(h*t+i)*4+c]-expected).abs()<1e-4);
            }}}
        }}
    }
    #[test]
    fn block32_empty_output_and_empty_contraction(){
        let mut out=[1f32;6];let _=matmul_i8_blocked(&[],&[],&[1.;2],&[],2,0,3,32,&mut out);assert_eq!(out,[0.;6]);
        let _=matmul_i8_blocked(&[],&[],&[],&[],0,32,0,32,&mut []);
    }
    #[test]
    #[should_panic(expected="shape overflow")]
    fn block32_rejects_overflow_before_pointer_arithmetic(){
        let _=matmul_i8_blocked(&[],&[],&[],&[],usize::MAX,32,1,32,&mut []);
    }
    fn sample(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let a = (0..m * k).map(|i| ((i % 13) as f32) * 0.25 - 1.5).collect();
        let b = (0..k * n).map(|i| ((i % 7) as f32) * 0.5 - 1.5).collect();
        (a, b)
    }

    /// Independent reference: `out[i,j] = sx[i]*sw[j]*sum_p a[i,p]*w[j,p]`, accumulated
    /// in i64 so the test does not share the kernel's i32 range question.
    fn reference(aq: &[i16], wq: &[i8], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i64;
                for p in 0..k {
                    acc += aq[i * k + p] as i64 * wq[j * k + p] as i64;
                }
                out[i * n + j] = acc as f32 * sx[i] * sw[j];
            }
        }
        out
    }

    #[test]
    fn quantisation_is_bounded_and_dequantises_within_one_step() {
        let (m, k) = (4usize, 32usize);
        let (a, _) = sample(m, k, k);
        let (aq, asx) = quantize_acts_i16(&a, m, k);
        assert!(aq.iter().all(|v| (*v as f32).abs() <= ACTIVATION_RANGE));
        assert!(asx.iter().all(|s| *s > 0.0));
        for r in 0..m {
            for p in 0..k {
                let back = aq[r * k + p] as f32 * asx[r];
                assert!((back - a[r * k + p]).abs() <= asx[r], "row {r} step {p}");
            }
        }
    }

    #[test]
    fn int8_matmul_matches_an_independent_reference_over_shapes() {
        // Boundary shapes on purpose: m/n/k cover every blocking tail.
        for (m, k, n) in [(1usize, 8usize, 1usize), (3, 16, 5), (4, 32, 7), (7, 40, 9), (8, 64, 12), (13, 96, 3)] {
            let (a, b) = sample(m, k, n);
            let mut w = vec![0f32; n * k];
            for p in 0..k {
                for j in 0..n {
                    w[j * k + p] = b[p * n + j];
                }
            }
            let (aq, asx) = quantize_acts_i16(&a, m, k);
            let (wq, wsx) = quantize_rows_i8(&w, n, k);
            let mut got = vec![0f32; m * n];
            let _ = matmul_i8(&aq, &wq, &asx, &wsx, m, k, n, &mut got);
            let want = reference(&aq, &wq, &asx, &wsx, m, k, n);
            let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
            for (x, y) in got.iter().zip(want.iter()) {
                assert!((x - y).abs() / scale < 1e-4, "shape {m}x{k}x{n}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn model_width_tiles_and_tails_match_i64_exactly() {
        for k in [768,1152] {
            for m in [1,2,3,4,5,6,7,8,9,15,16,17,39,40,41,47,48,49] {
                for n in [1,3,4,5,7,8,9] {
                    let a:Vec<i16>=(0..m*k).map(|i|match i%5 {0=>8192,1=>-8192,_=>(i%16385) as i16-8192}).collect();
                    let w:Vec<i8>=(0..n*k).map(|i|match i%5 {0=>127,1=>-127,_=>((i%255) as i16-127) as i8}).collect();
                    let sx:Vec<f32>=(0..m).map(|i|0.00013*(1+i%7) as f32).collect();
                    let sw:Vec<f32>=(0..n).map(|i|0.0017*(1+i%3) as f32).collect();
                    let mut out=vec![f32::NAN;m*n];
                    let _=matmul_i8(&a,&w,&sx,&sw,m,k,n,&mut out);
                    let want=reference(&a,&w,&sx,&sw,m,k,n);
                    assert!(out.iter().zip(want).all(|(a,b)|a.to_bits()==b.to_bits()),"shape {m}x{k}x{n}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected="shape overflow")]
    fn row_kernel_rejects_overflow_before_pointer_dispatch() {
        let _=matmul_i8(&[],&[],&[],&[],usize::MAX,768,0,&mut []);
    }

    #[test]
    fn activation_range_shrinks_for_large_contractions() {
        // The accumulator is i32: `range * 127 * cols` must stay below i32::MAX.
        // k=1152 keeps the tuned range; a 16k contraction gets a smaller one.
        for cols in [1152usize, 4096, 16384] {
            let row: Vec<f32> = (0..cols).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
            let (q, scales) = quantize_acts_i16(&row, 1, cols);
            let bound = ((i32::MAX as f32) / (127.0 * cols as f32)).floor().max(1.0);
            assert!(q.iter().all(|v| (*v as f32).abs() <= bound.min(ACTIVATION_RANGE) + 1.0), "cols={cols}");
            assert!(scales[0] >= 1.0 / bound, "cols={cols}");
            // The worst-case product still fits: no silent wrap is possible.
            let worst = (bound.min(ACTIVATION_RANGE) * 127.0 * cols as f32) as f64;
            assert!(worst <= i32::MAX as f64 + 1.0, "cols={cols}: {worst}");
        }
    }

    #[test]
    fn degenerate_quantisation_inputs_do_not_panic() {
        let (q, scales) = quantize_rows_i8(&[], 0, 0);
        assert!(q.is_empty() && scales.is_empty());
        let (q, scales) = quantize_acts_i16(&[], 0, 0);
        assert!(q.is_empty() && scales.is_empty());
        // rows > 0 with cols == 0 is the case that used to underflow `cols - 1`.
        let (q, scales) = quantize_rows_i8(&[], 3, 0);
        assert!(q.is_empty() && scales.len() == 3);
    }

    #[test]
    #[should_panic(expected = "input shorter than rows*cols")]
    fn quantisation_rejects_a_short_slice() {
        let _ = quantize_rows_i8(&[1.0, 2.0], 2, 4);
    }

    #[test]
    fn matmul_i8_reports_which_path_ran() {
        let (m, k, n) = (2usize, 8usize, 2usize);
        let (aq, asx) = quantize_acts_i16(&[0.5f32; 16], m, k);
        let (wq, wsx) = quantize_rows_i8(&[0.25f32; 16], n, k);
        let mut out = vec![0f32; m * n];
        let simd = matmul_i8(&aq, &wq, &asx, &wsx, m, k, n, &mut out);
        #[cfg(not(target_arch = "wasm32"))]
        assert!(!simd);
        #[cfg(target_arch = "wasm32")]
        assert!(simd);
    }

    #[test]
    fn softmax_rows_match_the_scalar_reference_and_sum_to_one() {
        let (rows, cols) = (5usize, 16usize);
        let mut data: Vec<f32> = (0..rows * cols).map(|i| ((i % 11) as f32) * 0.3 - 1.0).collect();
        let mut want = data.clone();
        for r in 0..rows {
            softmax_row_scalar(&mut want[r * cols..r * cols + cols]);
        }
        softmax_rows_inplace(&mut data, rows, cols);
        for (x, y) in data.iter().zip(want.iter()) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
        for r in 0..rows {
            let sum: f32 = data[r * cols..r * cols + cols].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r} sums to {sum}");
        }
    }

    #[test]
    fn softmax_handles_degenerate_shapes() {
        // Empty rows: no work, no division by zero, no panic.
        let mut empty: Vec<f32> = Vec::new();
        softmax_rows_inplace(&mut empty, 3, 0);
        // A row of -inf is NaN in both paths (exp(-inf - -inf) = exp(NaN)); it must not
        // panic, and it must not fabricate a distribution.
        let mut degenerate = vec![f32::NEG_INFINITY; 4];
        softmax_rows_inplace(&mut degenerate, 1, 4);
        assert!(degenerate.iter().all(|v| v.is_nan()));
    }
}

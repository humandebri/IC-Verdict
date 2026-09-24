//! Run the real Wasm SIMD dispatch against an independent i64 reference.
//! Build with --release --target wasm32-unknown-unknown and instantiate in Node.
//! No IC imports, model weights, network, or native SIMD emulation are involved.

#[unsafe(no_mangle)]
pub extern "C" fn check() -> u32 {
    let mut cases = 0;
    for k in [8, 40, 768, 1152] {
        for m in (1..=17).chain([31, 32, 33]).chain(39..=56).chain([63, 64, 65]) {
            for n in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 96] {
                for pattern in 0..3 {
                    let a: Vec<i16> = (0..m * k)
                        .map(|i| if pattern == 0 { (i % 16385) as i16 - 8192 } else { 8192 })
                        .collect();
                    let w: Vec<i8> = (0..n * k)
                        .map(|i| match pattern {
                            0 => ((i % 255) as i16 - 127) as i8,
                            1 => 127,
                            _ => -127,
                        })
                        .collect();
                    let sx: Vec<f32> = (0..m).map(|i| 0.00013 * (1 + i % 7) as f32).collect();
                    let sw: Vec<f32> = (0..n).map(|i| 0.0017 * (1 + i % 3) as f32).collect();
                    let mut got = vec![f32::NAN; m * n];
                    let simd = verdict_simd::matmul_i8(&a, &w, &sx, &sw, m, k, n, &mut got);
                    assert!(simd, "this check must run as Wasm");
                    for i in 0..m {
                        for j in 0..n {
                            let sum: i64 = (0..k).map(|p| a[i * k + p] as i64 * w[j * k + p] as i64).sum();
                            let want = sum as f32 * sx[i] * sw[j];
                            assert_eq!(got[i * n + j].to_bits(), want.to_bits(), "shape {m}x{k}x{n}, pattern {pattern}, [{i},{j}]");
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    cases
}

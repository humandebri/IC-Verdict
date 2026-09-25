//! Local benchmark only: prepare and validate once, then run the production SIMD kernel.
use std::{cell::RefCell, hint::black_box};

#[link(wasm_import_module = "ic0")]
extern "C" {
    fn msg_arg_data_size() -> i32;
    fn msg_arg_data_copy(dst: i32, offset: i32, size: i32);
    fn performance_counter(kind: i32) -> i64;
    fn msg_reply_data_append(src: i32, size: i32);
    fn msg_reply();
}

struct Bench {
    m: usize, n: usize, k: usize,
    a: Vec<i16>, weights: Vec<Vec<i8>>, sx: Vec<f32>, sw: Vec<f32>, out: Vec<f32>, cursor: usize,
}
thread_local! { static BENCH: RefCell<Option<Bench>> = const { RefCell::new(None) }; }

fn args<const N: usize>() -> [u32; N] {
    let mut bytes = vec![0u8; N * 4];
    unsafe {
        assert_eq!(msg_arg_data_size() as usize, bytes.len());
        msg_arg_data_copy(bytes.as_mut_ptr() as i32, 0, bytes.len() as i32);
    }
    std::array::from_fn(|i| u32::from_le_bytes(bytes[i*4..i*4+4].try_into().unwrap()))
}
fn reply(values: &[u64]) {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    unsafe { msg_reply_data_append(bytes.as_ptr() as i32, bytes.len() as i32); msg_reply(); }
}

#[export_name = "canister_update prepare"]
pub extern "C" fn prepare() {
    let [m, n, k, banks] = args::<4>().map(|x| x as usize);
    assert!(m > 0 && m <= 512 && n > 0 && n <= 4096 && k > 0 && k <= 4096 && k % 8 == 0);
    assert!(banks > 0 && banks <= 32);
    let a: Vec<i16> = (0..m*k).map(|i| (i % 16385) as i16 - 8192).collect();
    let weights: Vec<Vec<i8>> = (0..banks).map(|b| (0..n*k).map(|i| (((i + b*31) % 255) as i16 - 127) as i8).collect()).collect();
    let sx = vec![0.00013; m];
    let sw = vec![0.0017; n];
    let mut out = vec![0.0; m*n];
    let mut exact = vec![0.0; m*n];
    assert!(verdict_simd::matmul_i8(&a, &weights[0], &sx, &sw, m, k, n, &mut out));
    verdict_simd::matmul_i8_scalar(&a, &weights[0], &sx, &sw, m, k, n, &mut exact);
    assert!(out.iter().zip(&exact).all(|(a,b)| a.to_bits() == b.to_bits()));
    BENCH.with(|s| *s.borrow_mut() = Some(Bench { m, n, k, a, weights, sx, sw, out, cursor: 0 }));
    reply(&[m as u64, n as u64, k as u64, banks as u64]);
}

#[export_name = "canister_update run"]
pub extern "C" fn run() {
    let [iterations] = args::<1>();
    assert!(iterations <= 64);
    let (kernel, checksum) = BENCH.with(|state| {
        let mut borrow = state.borrow_mut();
        let b = borrow.as_mut().expect("prepare first");
        let before = unsafe { performance_counter(0) };
        for _ in 0..iterations {
            let simd = verdict_simd::matmul_i8(black_box(&b.a), black_box(&b.weights[b.cursor]), &b.sx, &b.sw, b.m, b.k, b.n, &mut b.out);
            assert!(simd);
            black_box(&b.out);
            b.cursor = (b.cursor + 1) % b.weights.len();
        }
        let kernel = unsafe { performance_counter(0) } - before;
        let checksum = b.out.iter().fold(0u64, |s,v| s.wrapping_add(v.to_bits() as u64));
        (kernel as u64, checksum)
    });
    let total = unsafe { performance_counter(0) } as u64;
    reply(&[total, kernel, checksum]);
}

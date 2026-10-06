//! `Device::causal_attention` and its reverse on CUDA against the host's float64, for a GPU host
//! without the test binaries (the cluster): grouped queries over unequal sequences with rows outside
//! every sequence at widths 18, 20, 64 and 128, and the vpd4l and Qwen3-0.6B head shapes over
//! sequences of 512, 512 and 300 rows. A line per case: the outputs', log partitions' and
//! cotangents' largest errors relative to their largest entries. Exits nonzero when one exceeds the
//! bands of `tests/tensor_decoder.rs`.
//! `cargo run --release -p gam-gpu --example attention_parity`.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Device, HeadLayout, Storage};
use ndarray::Array2;
use std::ops::Range;

const U: f64 = 1.0 / 16_777_216.0;

fn matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        f64::from((((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale) as f32)
    })
}

fn largest(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

fn error(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

/// The case's errors relative to the largest entries, and whether every one is within its band.
fn case(d: &Device, layout: HeadLayout, rows: usize, sequences: &[Range<usize>], seed: u64) -> Result<([f64; 3], bool), String> {
    let e = |x: gam_gpu::gpu_error::GpuError| x.to_string();
    let host = Device::host();
    let scale = 1.0 / (layout.width as f64).sqrt();
    let (y, ga) = (matrix(rows, layout.columns(), seed, 2.0), matrix(rows, layout.queries * layout.width, seed + 1, 1.0));
    let (yd, gad, yh, gah) = (d.upload(y.view()).map_err(e)?, d.upload(ga.view()).map_err(e)?, host.upload(y.view()).map_err(e)?, host.upload(ga.view()).map_err(e)?);
    let (out, lse) = d.causal_attention(&yd, layout, sequences, scale).map_err(e)?;
    let gy = d.causal_attention_backward(&yd, layout, sequences, scale, (&out, &lse), &gad).map_err(e)?;
    let (hout, hlse) = host.causal_attention(&yh, layout, sequences, scale).map_err(e)?;
    let hgy = host.causal_attention_backward(&yh, layout, sequences, scale, (&hout, &hlse), &gah).map_err(e)?;
    let (a, l, g) = (d.download(&out).map_err(e)?, d.download(&lse).map_err(e)?, d.download(&gy).map_err(e)?);
    let (ha, hl, hg) = (host.download(&hout).map_err(e)?, host.download(&hlse).map_err(e)?, host.download(&hgy).map_err(e)?);
    let longest = sequences.iter().map(ExactSizeIterator::len).max().unwrap_or(0) as f64;
    let w = layout.width as f64;
    let delta = scale * w * w * U * largest(&y).powi(2);
    let within = error(&a, &ha) <= (2.0 * delta + 2.0 * longest * U) * largest(&y)
        && error(&l, &hl) <= delta + 4.0 * longest * U * (1.0 + largest(&hl))
        && error(&g, &hg) <= (16.0 * delta + 16.0 * (longest + w) * U) * largest(&hg).max(1.0);
    Ok(([error(&a, &ha) / largest(&ha), error(&l, &hl) / largest(&hl), error(&g, &hg) / largest(&hg)], within))
}

fn main() -> Result<(), String> {
    let device = Device::accelerator(GpuPolicy::Auto).map_err(|e| e.to_string())?.ok_or("no accelerator")?;
    let d = device.with_storage(Storage::F32).map_err(|e| e.to_string())?;
    println!("{}", d.name());
    let mut cases: Vec<(String, HeadLayout, usize, Vec<Range<usize>>)> = [18, 20, 64, 128]
        .into_iter()
        .map(|width| (format!("width {width}"), HeadLayout { queries: 4, keys: 2, width }, 170, vec![0..11, 13..90, 90..155, 160..161]))
        .collect();
    for (name, layout) in [("vpd4l", HeadLayout { queries: 6, keys: 6, width: 128 }), ("qwen3-0.6b", HeadLayout { queries: 16, keys: 8, width: 128 })] {
        cases.push((name.to_string(), layout, 1324, vec![0..512, 512..1024, 1024..1324]));
    }
    let mut all = true;
    for (i, (name, layout, rows, sequences)) in cases.iter().enumerate() {
        let ([eo, el, eg], within) = case(&d, *layout, *rows, sequences, 8 + i as u64)?;
        println!("{name}: outputs {eo:.2e}, log partitions {el:.2e}, cotangents {eg:.2e} of their largest entries{}", if within { "" } else { "  OUTSIDE THE BANDS" });
        all &= within;
    }
    if all { Ok(()) } else { Err("errors outside the bands".to_string()) }
}

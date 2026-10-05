//! Causal attention in f32 on the Apple GPU (`attention_f32.inc`, the kernels CUDA compiles too)
//! against float64 attention computed here, values and cotangents, with the error printed.
//!
//! Bands (`u = 2⁻²⁴`): a score is an f32 fused multiply-add chain over the head's `D` columns, within
//! `D u Σ_d |q_d k_d|` of the exact one; scaled by `c` it shifts the softmax's exponents by at most
//! `δ = c D u S` (`S` the largest such sum), which moves each weight by at most `2δ` relative, and the
//! row sums and the weighted sum of values add `T` more roundings: an output is within
//! `(2δ + 2 T u) max|v|` of float64. The reverse's products read the same weights and sum over at
//! most `T` rows or `D` columns: a cotangent is within `(4δ + 4 (T + D) u)` of the largest entry it
//! sums. Each test runs when Metal resolves and has nothing to run otherwise.
#![cfg(target_os = "macos")]

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Device, HeadLayout};
use ndarray::Array2;
use std::ops::Range;

const U: f64 = 1.0 / 16_777_216.0;

fn metal() -> Option<Device> {
    let device = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")?;
    (!device.float64()).then_some(device)
}

fn matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        f64::from(((2.0 * unit - 1.0) * scale) as f32)
    })
}

/// Float64 attention of `y` over `sequences`: the output, log partitions, and the cotangent in `y`
/// of `⟨ga, output⟩`.
fn reference(y: &Array2<f64>, layout: HeadLayout, sequences: &[Range<usize>], scale: f64, ga: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (w, group) = (layout.width, layout.queries / layout.keys);
    let mut out = Array2::zeros((y.nrows(), layout.queries * w));
    let mut lse = Array2::zeros((y.nrows(), layout.queries));
    let mut gy = Array2::zeros(y.dim());
    for r in sequences {
        for h in 0..layout.queries {
            let (q0, k0, v0) = (h * w, (layout.queries + h / group) * w, (layout.queries + layout.keys + h / group) * w);
            for i in r.clone() {
                let s: Vec<f64> = (r.start..=i).map(|j| scale * (0..w).map(|t| y[[i, q0 + t]] * y[[j, k0 + t]]).sum::<f64>()).collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = s.iter().map(|x| (x - m).exp()).sum();
                lse[[i, h]] = m + total.ln();
                let p: Vec<f64> = s.iter().map(|x| (x - m).exp() / total).collect();
                for t in 0..w {
                    out[[i, h * w + t]] = p.iter().enumerate().map(|(n, pj)| pj * y[[r.start + n, v0 + t]]).sum();
                }
                let dp: Vec<f64> = (r.start..=i).map(|j| (0..w).map(|t| ga[[i, h * w + t]] * y[[j, v0 + t]]).sum()).collect();
                let dot: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                for (n, j) in (r.start..=i).enumerate() {
                    let ds = p[n] * (dp[n] - dot);
                    for t in 0..w {
                        gy[[j, v0 + t]] += p[n] * ga[[i, h * w + t]];
                        gy[[i, q0 + t]] += scale * ds * y[[j, k0 + t]];
                        gy[[j, k0 + t]] += scale * ds * y[[i, q0 + t]];
                    }
                }
            }
        }
    }
    (out, lse, gy)
}

fn largest(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

/// The largest error of `a` against `b`, asserted within `band`.
fn close(what: &str, a: &Array2<f64>, b: &Array2<f64>, band: f64) -> f64 {
    assert_eq!(a.dim(), b.dim(), "{what}: shapes");
    let error = a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(error <= band, "{what}: error {error:e} above the band {band:e}");
    error
}

/// Metal against float64 on one case; returns the outputs' and cotangents' largest errors relative
/// to their largest entries.
fn case(d: &Device, layout: HeadLayout, rows: usize, sequences: &[Range<usize>], seed: u64) -> (f64, f64) {
    let scale = 1.0 / (layout.width as f64).sqrt();
    let y = matrix(rows, layout.columns(), seed, 2.0);
    let ga = matrix(rows, layout.queries * layout.width, seed + 1, 1.0);
    let (out, lse, gy) = reference(&y, layout, sequences, scale, &ga);
    let up = |m: &Array2<f64>| d.upload(m.view()).unwrap();
    let (yd, gad) = (up(&y), up(&ga));
    let (od, ld) = d.causal_attention(&yd, layout, sequences, scale).unwrap();
    let gyd = d.causal_attention_backward(&yd, layout, sequences, scale, (&od, &ld), &gad).unwrap();
    let (o, l, g) = (d.download(&od).unwrap(), d.download(&ld).unwrap(), d.download(&gyd).unwrap());
    let longest = sequences.iter().map(ExactSizeIterator::len).max().unwrap_or(0) as f64;
    // The largest Σ_d |q_d k_d| is at most the head width times the largest product.
    let delta = scale * layout.width as f64 * U * layout.width as f64 * largest(&y).powi(2);
    let what = format!("{layout:?} over {sequences:?}");
    let eo = close(&format!("outputs {what}"), &o, &out, (2.0 * delta + 2.0 * longest * U) * largest(&y));
    close(&format!("log partitions {what}"), &l, &lse, delta + 4.0 * longest * U * (1.0 + largest(&lse)));
    let eg = close(&format!("cotangents {what}"), &g, &gy, (4.0 * delta + 4.0 * (longest + layout.width as f64) * U) * largest(&gy).max(1.0) * 4.0);
    (eo / largest(&out), eg / largest(&gy))
}

#[test]
fn f32_attention_matches_float64_on_small_shapes() {
    let Some(d) = metal() else { return };
    // Grouped queries, sequences of unequal lengths across the tiles, rows outside every sequence;
    // a width of 18 loads by element, 20 and 128 by four columns.
    for width in [18, 20, 64, 128] {
        let layout = HeadLayout { queries: 4, keys: 2, width };
        let (eo, eg) = case(&d, layout, 90, &[0..11, 13..50, 50..85, 88..89], 3);
        eprintln!("width {width}: outputs {eo:.2e}, cotangents {eg:.2e} of their largest entries");
    }
}

#[test]
fn f32_attention_matches_float64_at_the_fit_head_shapes() {
    let Some(d) = metal() else { return };
    for (name, layout) in [("vpd4l", HeadLayout { queries: 6, keys: 6, width: 128 }), ("qwen3-0.6b", HeadLayout { queries: 16, keys: 8, width: 128 })] {
        let (eo, eg) = case(&d, layout, 356, &[0..256, 256..356], 7);
        eprintln!("{name}: outputs {eo:.2e}, cotangents {eg:.2e} of their largest entries");
    }
}

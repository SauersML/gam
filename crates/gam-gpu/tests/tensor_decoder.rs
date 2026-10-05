//! The decoder layer's fused operations (`Device::rms_gain`, `heads_rope`, `causal_attention`,
//! `swiglu`, `gelu_tanh` and their reverses) on CUDA against their host twins on the same inputs,
//! and the split products it runs in (`Arithmetic::Tf32x3`, `Arithmetic::Bf16x3`).
//!
//! The device computes in f32 and rounds what a product reads to bfloat16 where asked; the host
//! computes in float64 and rounds the same values alike. A bfloat16 output is within one bfloat16 rounding
//! (`2⁻⁸` relative) of the host's plus the f32 error before it; an f32 output within `2⁻¹⁶` of the
//! largest magnitude entering it (a chain of f32 operations and sums of at most a few thousand
//! terms). Attention's weights are rounded to bfloat16 on both sides, so a weight at a rounding tie
//! may round apart: its outputs are within `2⁻⁷` of `Σ |p| |v|`, the magnitude they sum. Each test
//! runs when a CUDA device resolves and has nothing to run otherwise, except the split products'
//! host test.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Arithmetic, Device, HeadLayout, Op, Storage, Tensor};
use ndarray::Array2;

fn cuda() -> Option<Device> {
    let wide = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault")?;
    Some(wide.with_storage(Storage::F32).expect("CUDA holds f32"))
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

fn largest(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

/// `a` within `band` of `b` entrywise.
fn close(what: &str, a: &Array2<f64>, b: &Array2<f64>, band: impl Fn(usize, usize) -> f64) {
    assert_eq!(a.dim(), b.dim(), "{what}: shapes");
    for ((i, j), x) in a.indexed_iter() {
        let allowed = band(i, j);
        assert!((x - b[[i, j]]).abs() <= allowed, "{what} ({i},{j}): {x} against {} (band {allowed:e})", b[[i, j]]);
    }
}

const SINGLE: f64 = 1.0 / 65_536.0;
const HALF: f64 = 1.0 / 256.0;

#[test]
fn the_rms_gain_and_its_reverse_match_the_host() {
    let Some(d) = cuda() else { return };
    let host = Device::host();
    let (x, g, gy) = (matrix(37, 300, 1, 2.0), matrix(1, 300, 2, 1.0), matrix(37, 300, 3, 1.0));
    for bf16 in [false, true] {
        let (dy, dk) = d.rms_gain(&d.upload(x.view()).unwrap(), &d.upload(g.view()).unwrap(), 1e-6, bf16).unwrap();
        let (hy, hk) = host.rms_gain(&host.upload(x.view()).unwrap(), &host.upload(g.view()).unwrap(), 1e-6, bf16).unwrap();
        let (y, k) = (host.download(&hy).unwrap(), host.download(&hk).unwrap());
        let band = if bf16 { HALF } else { SINGLE };
        close("rms gain", &d.download(&dy).unwrap(), &y, |i, j| band * y[[i, j]].abs() + SINGLE * largest(&y));
        close("rms scale", &d.download(&dk).unwrap(), &k, |i, _| SINGLE * k[[i, 0]]);
    }
    let (_, dk) = d.rms_gain(&d.upload(x.view()).unwrap(), &d.upload(g.view()).unwrap(), 1e-6, false).unwrap();
    let (_, hk) = host.rms_gain(&host.upload(x.view()).unwrap(), &host.upload(g.view()).unwrap(), 1e-6, false).unwrap();
    let base = matrix(37, 300, 4, 1.0);
    let mut dgx = d.upload(base.view()).unwrap();
    let mut hgx = host.upload(base.view()).unwrap();
    d.rms_gain_backward((&d.upload(x.view()).unwrap(), &d.upload(g.view()).unwrap(), &dk), &d.upload(gy.view()).unwrap(), &mut dgx).unwrap();
    host.rms_gain_backward((&host.upload(x.view()).unwrap(), &host.upload(g.view()).unwrap(), &hk), &host.upload(gy.view()).unwrap(), &mut hgx).unwrap();
    let expected = host.download(&hgx).unwrap();
    close("rms gain reverse", &d.download(&dgx).unwrap(), &expected, |_, _| SINGLE * largest(&expected) * 4.0);
}

/// Heads, projections, gains and rotation angles of a small grouped-query attention layer.
fn heads_case(rows: usize, normed: bool, half_split: bool) -> (HeadLayout, Array2<f64>, Option<Array2<f64>>, Array2<f64>, Array2<f64>, bool) {
    let layout = HeadLayout { queries: 4, keys: 2, width: 16 };
    let p = matrix(rows, layout.columns(), 5, 3.0);
    let gains = normed.then(|| matrix(layout.queries + layout.keys, layout.width, 6, 1.5));
    let planes = 6;
    let angles = Array2::from_shape_fn((rows, planes), |(r, plane)| (r % 9) as f64 * 10f64.powf(-(plane as f64) / planes as f64));
    (layout, p, gains, angles.mapv(f64::cos).mapv(|v| f64::from(v as f32)), angles.mapv(f64::sin).mapv(|v| f64::from(v as f32)), half_split)
}

#[test]
fn heads_rope_and_its_reverse_match_the_host() {
    let Some(d) = cuda() else { return };
    let host = Device::host();
    for normed in [false, true] {
        for half_split in [false, true] {
            let (layout, p, gains, cos, sin, half) = heads_case(21, normed, half_split);
            let gy = matrix(21, layout.columns(), 7, 1.0);
            let run = |dev: &Device| {
                let up = |m: &Array2<f64>| dev.upload(m.view()).unwrap();
                let (pt, ct, st) = (up(&p), up(&cos), up(&sin));
                let gt = gains.as_ref().map(up);
                let (y, k) = dev.heads_rope(&pt, layout, gt.as_ref().map(|g| (g, 1e-6)), Some((&ct, &st, half)), true).unwrap();
                let (y32, _) = dev.heads_rope(&pt, layout, gt.as_ref().map(|g| (g, 1e-6)), Some((&ct, &st, half)), false).unwrap();
                let gp = dev.heads_rope_backward(&pt, layout, gt.as_ref().zip(k.as_ref()), Some((&ct, &st, half)), &up(&gy)).unwrap();
                (dev.download(&y).unwrap(), dev.download(&y32).unwrap(), dev.download(&gp).unwrap())
            };
            let ((y, y32, gp), (hy, hy32, hgp)) = (run(&d), run(&host));
            let what = format!("heads (normed {normed}, rotate-half {half_split})");
            close(&what, &y, &hy, |i, j| HALF * hy[[i, j]].abs() + 4.0 * SINGLE * largest(&hy));
            close(&format!("{what} in f32"), &y32, &hy32, |_, _| 4.0 * SINGLE * largest(&hy32));
            close(&format!("{what} reverse"), &gp, &hgp, |_, _| 16.0 * SINGLE * largest(&hgp));
        }
    }
}

/// The device's attention and its reverse on `sequences` of `rows` rows against the host's on the
/// same bfloat16 heads; returns the largest absolute and relative differences of the outputs and
/// of the cotangents.
fn attention_case(d: &Device, layout: HeadLayout, rows: usize, sequences: &[std::ops::Range<usize>], scale: f64, seed: u64) -> [(f64, f64); 2] {
    let host = Device::host();
    // Bfloat16 projections, exactly as the attention reads them on both sides.
    let (y16, _) = d.heads_rope(&d.upload(matrix(rows, layout.columns(), seed, 2.0).view()).unwrap(), layout, None, None, true).unwrap();
    let y = d.download(&y16).unwrap();
    let ga = matrix(rows, layout.queries * layout.width, seed + 1, 1.0);
    let (out, lse) = d.causal_attention(&y16, layout, sequences, scale).unwrap();
    let (hy, hga) = (host.upload(y.view()).unwrap(), host.upload(ga.view()).unwrap());
    let (hout, hlse) = host.causal_attention(&hy, layout, sequences, scale).unwrap();
    let (a, ha) = (d.download(&out).unwrap(), host.download(&hout).unwrap());
    let values = largest(&y);
    let what = format!("{layout:?} over {sequences:?}");
    // Both sides round the weights and the outputs to bfloat16; a weight at a rounding tie may round
    // apart, and the device's row sums add its rounded weights (2⁻⁹ relative).
    close(&format!("attention {what}"), &a, &ha, |i, j| 2.0 * HALF * values + HALF * ha[[i, j]].abs());
    let (l, hl) = (d.download(&lse).unwrap(), host.download(&hlse).unwrap());
    close(&format!("log partitions {what}"), &l, &hl, |i, j| HALF + SINGLE * hl[[i, j]].abs());
    let gy = d.download(&d.causal_attention_backward(&y16, layout, sequences, scale, (&out, &lse), &d.upload(ga.view()).unwrap()).unwrap()).unwrap();
    let hgy = host.download(&host.causal_attention_backward(&hy, layout, sequences, scale, (&hout, &hlse), &hga).unwrap()).unwrap();
    // The reverse reads the cotangent and the weights' cotangent rounded to bfloat16.
    let band = 8.0 * HALF * largest(&hgy).max(values);
    close(&format!("attention reverse {what}"), &gy, &hgy, |_, _| band);
    let difference = |x: &Array2<f64>, h: &Array2<f64>| {
        let abs = x.iter().zip(h).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        (abs, abs / largest(h))
    };
    [difference(&a, &ha), difference(&gy, &hgy)]
}

#[test]
fn causal_attention_and_its_reverse_match_the_host() {
    let Some(d) = cuda() else { return };
    // Grouped queries, sequences of unequal lengths crossing the 64-row tiles, rows outside every
    // sequence; widths padded inside the kernels (16 and 20 columns: the latter loads by element)
    // and a full 128.
    for width in [16, 20, 128] {
        let layout = HeadLayout { queries: 4, keys: 2, width };
        let sequences = [0..11, 13..90, 90..155, 160..161];
        attention_case(&d, layout, 170, &sequences, 1.0 / (width as f64).sqrt(), 8);
    }
}

#[test]
fn causal_attention_matches_the_host_at_the_fit_shapes() {
    let Some(d) = cuda() else { return };
    // vpd4l: 6 heads of 128 with their own keys and values; Qwen3-0.6B: 16 query heads over 8
    // key-value heads of 128. Sequences of 512 rows and one shorter.
    for (name, layout) in [("vpd4l", HeadLayout { queries: 6, keys: 6, width: 128 }), ("qwen3-0.6b", HeadLayout { queries: 16, keys: 8, width: 128 })] {
        let sequences = [0..512, 512..1024, 1024..1324];
        let [(a_abs, a_rel), (g_abs, g_rel)] = attention_case(&d, layout, 1324, &sequences, 1.0 / 128f64.sqrt(), 20);
        eprintln!("{name}: outputs max abs {a_abs:.3e} (relative to the largest {a_rel:.3e}), cotangents max abs {g_abs:.3e} (relative {g_rel:.3e})");
    }
}

#[test]
fn the_mlp_activations_and_their_reverses_match_the_host() {
    let Some(d) = cuda() else { return };
    let host = Device::host();
    let (h, ga, gh) = (matrix(19, 64, 10, 4.0), matrix(19, 32, 11, 1.0), matrix(19, 64, 12, 1.0));
    let bias = matrix(1, 64, 13, 0.5);
    let up = |dev: &Device, m: &Array2<f64>| -> Tensor { dev.upload(m.view()).unwrap() };
    for bf16 in [true, false] {
        let a = d.download(&d.swiglu(&up(&d, &h), bf16).unwrap()).unwrap();
        let ha = host.download(&host.swiglu(&up(&host, &h), bf16).unwrap()).unwrap();
        let band = if bf16 { HALF } else { SINGLE };
        close("swiglu", &a, &ha, |i, j| band * ha[[i, j]].abs() + SINGLE * largest(&ha));
        let a = d.download(&d.gelu_tanh(&up(&d, &h), Some(&up(&d, &bias)), bf16).unwrap()).unwrap();
        let ha = host.download(&host.gelu_tanh(&up(&host, &h), Some(&up(&host, &bias)), bf16).unwrap()).unwrap();
        close("gelu", &a, &ha, |i, j| band * ha[[i, j]].abs() + SINGLE * largest(&ha));
    }
    let g = d.download(&d.swiglu_backward(&up(&d, &h), &up(&d, &ga)).unwrap()).unwrap();
    let hg = host.download(&host.swiglu_backward(&up(&host, &h), &up(&host, &ga)).unwrap()).unwrap();
    close("swiglu reverse", &g, &hg, |_, _| 4.0 * SINGLE * largest(&hg));
    let g = d.download(&d.gelu_tanh_backward(&up(&d, &h), Some(&up(&d, &bias)), &up(&d, &gh)).unwrap()).unwrap();
    let hg = host.download(&host.gelu_tanh_backward(&up(&host, &h), Some(&up(&host, &bias)), &up(&host, &gh)).unwrap()).unwrap();
    close("gelu reverse", &g, &hg, |_, _| 4.0 * SINGLE * largest(&hg));
}

/// `op(a) op(b)` in `arithmetic` on `device` (f32 operands, and `b` also as a bfloat16 copy where
/// `frozen`), and the float64 product's magnitude `Σ |a| |b|` per entry.
fn split_case(device: &Device, arithmetic: Arithmetic, frozen: bool) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let host = Device::host();
    let (rows, cols, inner) = (40, 56, 700);
    let (a, b) = (matrix(rows, inner, 31, 1.0), matrix(cols, inner, 37, 1.0));
    let b = if frozen { host.download(&host.bf16_copy(&host.upload(b.view()).unwrap()).unwrap()).unwrap() } else { b };
    let (da, db) = (device.upload(a.view()).unwrap(), device.upload(b.view()).unwrap());
    let db = if frozen && !device.is_host() { device.bf16_copy(&db).unwrap() } else { db };
    let mut c = device.zeros(rows, cols).unwrap();
    device.gemm(&mut c, 1.0, &da, Op::N, &db, Op::T, 0.0, arithmetic).unwrap();
    (device.download(&c).unwrap(), a.dot(&b.t()), a.mapv(f64::abs).dot(&b.mapv(f64::abs).t()))
}

/// The largest difference of `c` from `exact` relative to the magnitude summed.
fn worst(c: &Array2<f64>, exact: &Array2<f64>, magnitude: &Array2<f64>) -> f64 {
    c.iter().zip(exact).zip(magnitude).fold(0.0_f64, |m, ((x, e), g)| m.max((x - e).abs() / g))
}

/// A split product is within `2 u` of each term's magnitude plus f32's sums (`(k + 2) 2⁻²⁴`) of the
/// exact product, `u` its unit roundoff; one product of its parts' arithmetic is 64 times further
/// off at least.
#[test]
fn split_products_keep_f32_accuracy_on_the_host() {
    let host = Device::host();
    for arithmetic in [Arithmetic::Tf32x3, Arithmetic::Bf16x3] {
        let mut split = 0.0_f64;
        for frozen in [false, true] {
            let (c, exact, magnitude) = split_case(&host, arithmetic, frozen);
            let bound = 2.0 * arithmetic.unit_roundoff() + 702.0 * f64::from(f32::EPSILON) / 2.0;
            close(&format!("{arithmetic:?} (frozen {frozen})"), &c, &exact, |i, j| bound * magnitude[[i, j]]);
            split = split.max(worst(&c, &exact, &magnitude));
        }
        let single = arithmetic.split().unwrap();
        let (c, exact, magnitude) = split_case(&host, single, false);
        let alone = worst(&c, &exact, &magnitude);
        assert!(alone > 64.0 * split, "{single:?} alone is within {alone:e} of the magnitude, {arithmetic:?} within {split:e}");
    }
}

/// The CUDA split products against the host's, which round alike: within the bound of the exact
/// product either keeps.
#[test]
fn split_products_match_the_host_on_cuda() {
    let Some(d) = cuda() else { return };
    for arithmetic in [Arithmetic::Tf32x3, Arithmetic::Bf16x3] {
        for frozen in [false, true] {
            let (c, exact, magnitude) = split_case(&d, arithmetic, frozen);
            let bound = 2.0 * arithmetic.unit_roundoff() + 702.0 * f64::from(f32::EPSILON) / 2.0;
            close(&format!("CUDA {arithmetic:?} (frozen {frozen})"), &c, &exact, |i, j| bound * magnitude[[i, j]]);
        }
    }
}

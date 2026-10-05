//! The decoder layer's fused operations (`Device::rms_gain`, `heads_rope`, `causal_attention`,
//! `swiglu`, `gelu_tanh` and their reverses) on CUDA against their host twins on the same inputs.
//!
//! The device computes in f32 and rounds what a product reads to bfloat16; the host computes in
//! float64 and rounds the same values to bfloat16. A bfloat16 output is within one bfloat16 rounding
//! (`2⁻⁸` relative) of the host's plus the f32 error before it; an f32 output within `2⁻¹⁶` of the
//! largest magnitude entering it (a chain of f32 operations and sums of at most a few thousand
//! terms). Attention's weights are rounded to bfloat16 on both sides, so a weight at a rounding tie
//! may round apart: its outputs are within `2⁻⁷` of `Σ |p| |v|`, the magnitude they sum. Each test
//! runs when a CUDA device resolves and has nothing to run otherwise.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Device, HeadLayout, Storage, Tensor};
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
                let (y, k) = dev.heads_rope(&pt, layout, gt.as_ref().map(|g| (g, 1e-6)), Some((&ct, &st, half))).unwrap();
                let gp = dev.heads_rope_backward(&pt, layout, gt.as_ref().zip(k.as_ref()), Some((&ct, &st, half)), &up(&gy)).unwrap();
                (dev.download(&y).unwrap(), dev.download(&gp).unwrap())
            };
            let ((y, gp), (hy, hgp)) = (run(&d), run(&host));
            let what = format!("heads (normed {normed}, rotate-half {half_split})");
            close(&what, &y, &hy, |i, j| HALF * hy[[i, j]].abs() + 4.0 * SINGLE * largest(&hy));
            close(&format!("{what} reverse"), &gp, &hgp, |_, _| 16.0 * SINGLE * largest(&hgp));
        }
    }
}

#[test]
fn causal_attention_and_its_reverse_match_the_host() {
    let Some(d) = cuda() else { return };
    let host = Device::host();
    let layout = HeadLayout { queries: 4, keys: 2, width: 16 };
    let (blocks, length) = (3, 11);
    // Bfloat16 projections, exactly as the attention reads them on both sides.
    let (y16, _) = d.heads_rope(&d.upload(matrix(blocks * length, layout.columns(), 8, 2.0).view()).unwrap(), layout, None, None).unwrap();
    let y = d.download(&y16).unwrap();
    let ga = matrix(blocks * length, layout.queries * layout.width, 9, 1.0);
    let scale = 0.25;
    let a = d.download(&d.causal_attention(&y16, layout, blocks, scale).unwrap()).unwrap();
    let ha = host.download(&host.causal_attention(&host.upload(y.view()).unwrap(), layout, blocks, scale).unwrap()).unwrap();
    let values = largest(&y);
    close("attention", &a, &ha, |_, _| 2.0 * HALF * values);
    let gy = d.download(&d.causal_attention_backward(&y16, layout, blocks, scale, &d.upload(ga.view()).unwrap()).unwrap()).unwrap();
    let hgy = host.download(&host.causal_attention_backward(&host.upload(y.view()).unwrap(), layout, blocks, scale, &host.upload(ga.view()).unwrap()).unwrap()).unwrap();
    // The reverse reads the cotangent and the weights' cotangent rounded to bfloat16.
    close("attention reverse", &gy, &hgy, |_, _| 8.0 * HALF * largest(&hgy).max(values));
}

#[test]
fn the_mlp_activations_and_their_reverses_match_the_host() {
    let Some(d) = cuda() else { return };
    let host = Device::host();
    let (h, ga, gh) = (matrix(19, 64, 10, 4.0), matrix(19, 32, 11, 1.0), matrix(19, 64, 12, 1.0));
    let bias = matrix(1, 64, 13, 0.5);
    let up = |dev: &Device, m: &Array2<f64>| -> Tensor { dev.upload(m.view()).unwrap() };
    let a = d.download(&d.swiglu(&up(&d, &h)).unwrap()).unwrap();
    let ha = host.download(&host.swiglu(&up(&host, &h)).unwrap()).unwrap();
    close("swiglu", &a, &ha, |i, j| HALF * ha[[i, j]].abs() + SINGLE * largest(&ha));
    let g = d.download(&d.swiglu_backward(&up(&d, &h), &up(&d, &ga)).unwrap()).unwrap();
    let hg = host.download(&host.swiglu_backward(&up(&host, &h), &up(&host, &ga)).unwrap()).unwrap();
    close("swiglu reverse", &g, &hg, |_, _| 4.0 * SINGLE * largest(&hg));
    let a = d.download(&d.gelu_tanh(&up(&d, &h), Some(&up(&d, &bias))).unwrap()).unwrap();
    let ha = host.download(&host.gelu_tanh(&up(&host, &h), Some(&up(&host, &bias))).unwrap()).unwrap();
    close("gelu", &a, &ha, |i, j| HALF * ha[[i, j]].abs() + SINGLE * largest(&ha));
    let g = d.download(&d.gelu_tanh_backward(&up(&d, &h), Some(&up(&d, &bias)), &up(&d, &gh)).unwrap()).unwrap();
    let hg = host.download(&host.gelu_tanh_backward(&up(&host, &h), Some(&up(&host, &bias)), &up(&host, &gh)).unwrap()).unwrap();
    close("gelu reverse", &g, &hg, |_, _| 4.0 * SINGLE * largest(&hg));
}

//! The decoder layer's fused operations (`Device::rms_gain`, `heads_rope`, `swiglu`, `gelu_tanh`
//! and their reverses) on the Apple GPU (f32) against the host's float64 on the same inputs. Each
//! output is a short chain of f32 operations and sums of at most a few hundred terms, within `2⁻¹⁶`
//! of the largest magnitude entering it (the bands of `tests/tensor_decoder.rs`, CUDA's twin); a
//! value asked for in bfloat16 is within one bfloat16 rounding (`2⁻⁸` relative) more. Each test runs
//! when Metal resolves and has nothing to run otherwise.
#![cfg(target_os = "macos")]

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Device, HeadLayout, Tensor};
use ndarray::Array2;

const SINGLE: f64 = 1.0 / 65_536.0;
const HALF: f64 = 1.0 / 256.0;

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
        f64::from((((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale) as f32)
    })
}

fn largest(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

fn close(what: &str, a: &Array2<f64>, b: &Array2<f64>, band: impl Fn(usize, usize) -> f64) {
    assert_eq!(a.dim(), b.dim(), "{what}: shapes");
    for ((i, j), x) in a.indexed_iter() {
        let allowed = band(i, j);
        assert!((x - b[[i, j]]).abs() <= allowed, "{what} ({i},{j}): {x} against {} (band {allowed:e})", b[[i, j]]);
    }
}

#[test]
fn the_rms_gain_and_its_reverse_match_the_host() {
    let Some(d) = metal() else { return };
    let host = Device::host();
    let (x, g, gy, base) = (matrix(37, 300, 1, 2.0), matrix(1, 300, 2, 1.0), matrix(37, 300, 3, 1.0), matrix(37, 300, 4, 1.0));
    for bf16 in [false, true] {
        let run = |dev: &Device| {
            let up = |m: &Array2<f64>| dev.upload(m.view()).unwrap();
            let (xt, gt) = (up(&x), up(&g));
            let (y, k) = dev.rms_gain(&xt, &gt, 1e-6, bf16).unwrap();
            let mut gx = up(&base);
            dev.rms_gain_backward((&xt, &gt, &k), &up(&gy), &mut gx).unwrap();
            (dev.download(&y).unwrap(), dev.download(&k).unwrap(), dev.download(&gx).unwrap())
        };
        let ((y, k, gx), (hy, hk, hgx)) = (run(&d), run(&host));
        let band = if bf16 { HALF } else { SINGLE };
        close("rms gain", &y, &hy, |i, j| band * hy[[i, j]].abs() + SINGLE * largest(&hy));
        close("rms scale", &k, &hk, |i, _| SINGLE * hk[[i, 0]]);
        close("rms gain reverse", &gx, &hgx, |_, _| 4.0 * SINGLE * largest(&hgx));
    }
}

#[test]
fn heads_rope_and_its_reverse_match_the_host() {
    let Some(d) = metal() else { return };
    let host = Device::host();
    let layout = HeadLayout { queries: 4, keys: 2, width: 16 };
    let rows = 21;
    let (p, gy) = (matrix(rows, layout.columns(), 5, 3.0), matrix(rows, layout.columns(), 7, 1.0));
    let gains = matrix(layout.queries + layout.keys, layout.width, 6, 1.5);
    let planes = 6;
    let angles = Array2::from_shape_fn((rows, planes), |(r, plane)| (r % 9) as f64 * 10f64.powf(-(plane as f64) / planes as f64));
    let (cos, sin) = (angles.mapv(|a| f64::from(a.cos() as f32)), angles.mapv(|a| f64::from(a.sin() as f32)));
    for normed in [false, true] {
        for rotated in [None, Some(false), Some(true)] {
            let run = |dev: &Device| {
                let up = |m: &Array2<f64>| dev.upload(m.view()).unwrap();
                let (pt, ct, st, gt) = (up(&p), up(&cos), up(&sin), up(&gains));
                let rotation = rotated.map(|half| (&ct, &st, half));
                let (y, k) = dev.heads_rope(&pt, layout, normed.then_some((&gt, 1e-6)), rotation).unwrap();
                let norm: Option<(&Tensor, &Tensor)> = if normed { Some((&gt, k.as_ref().unwrap())) } else { None };
                let gp = dev.heads_rope_backward(&pt, layout, norm, rotation, &up(&gy)).unwrap();
                (dev.download(&y).unwrap(), dev.download(&gp).unwrap())
            };
            let ((y, gp), (hy, hgp)) = (run(&d), run(&host));
            let what = format!("heads (normed {normed}, rotation {rotated:?})");
            close(&what, &y, &hy, |_, _| 4.0 * SINGLE * largest(&hy));
            close(&format!("{what} reverse"), &gp, &hgp, |_, _| 16.0 * SINGLE * largest(&hgp));
        }
    }
}

#[test]
fn the_mlp_activations_and_their_reverses_match_the_host() {
    let Some(d) = metal() else { return };
    let host = Device::host();
    let (h, ga, gh, bias) = (matrix(19, 64, 10, 4.0), matrix(19, 32, 11, 1.0), matrix(19, 64, 12, 1.0), matrix(1, 64, 13, 0.5));
    let up = |dev: &Device, m: &Array2<f64>| -> Tensor { dev.upload(m.view()).unwrap() };
    for bf16 in [false, true] {
        let band = if bf16 { HALF } else { SINGLE };
        let a = d.download(&d.swiglu(&up(&d, &h), bf16).unwrap()).unwrap();
        let ha = host.download(&host.swiglu(&up(&host, &h), bf16).unwrap()).unwrap();
        close("swiglu", &a, &ha, |i, j| band * ha[[i, j]].abs() + SINGLE * largest(&ha));
        let a = d.download(&d.gelu_tanh(&up(&d, &h), Some(&up(&d, &bias)), bf16).unwrap()).unwrap();
        let ha = host.download(&host.gelu_tanh(&up(&host, &h), Some(&up(&host, &bias)), bf16).unwrap()).unwrap();
        close("gelu", &a, &ha, |i, j| band * ha[[i, j]].abs() + SINGLE * largest(&ha));
    }
    let g = d.download(&d.swiglu_backward(&up(&d, &h), &up(&d, &ga)).unwrap()).unwrap();
    let hg = host.download(&host.swiglu_backward(&up(&host, &h), &up(&host, &ga)).unwrap()).unwrap();
    close("swiglu reverse", &g, &hg, |_, _| 4.0 * SINGLE * largest(&hg));
    for b in [None, Some(&bias)] {
        let g = d.download(&d.gelu_tanh_backward(&up(&d, &h), b.map(|b| up(&d, b)).as_ref(), &up(&d, &gh)).unwrap()).unwrap();
        let hg = host.download(&host.gelu_tanh_backward(&up(&host, &h), b.map(|b| up(&host, b)).as_ref(), &up(&host, &gh)).unwrap()).unwrap();
        close("gelu reverse", &g, &hg, |_, _| 4.0 * SINGLE * largest(&hg));
    }
}

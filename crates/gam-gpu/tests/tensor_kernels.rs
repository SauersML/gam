//! The tensor operations of `gam_gpu::tensor`, each on a CUDA device against its host twin, and
//! the host's products against compensated references.
//!
//! The bands: a float64 product lies within `γ_{k+2}·(|A||B|)_{ij}` of the exact value on either
//! backend, so the two agree within twice that; an f32 product within its derived f32 band
//! (`precision_bounds::GemmBand`). Every other operation is one expression per entry, or a
//! per-row reduction of `n` terms: the backends agree within `γ_{n+8}` of the magnitude of what
//! is summed (they sum in different orders; `exp`, `log`, `tanh` and `erfc` are within a few
//! ulps on both). The device half runs when a CUDA device is present (a runtime probe) and has
//! nothing to run otherwise.

use gam_gpu::GpuPolicy;
use gam_gpu::precision_bounds::{DeviceArithmetic, GemmBand};
use gam_gpu::tensor::{Arithmetic, Device, Op, PointwiseLaw, Tensor};
use ndarray::{Array2, ArrayView1};

const U: f64 = f64::EPSILON / 2.0;

fn gamma(n: usize) -> f64 {
    n as f64 * U / (1.0 - n as f64 * U)
}

fn matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        (2.0 * unit - 1.0) * scale
    })
}

fn exact_dot(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let p = x * y;
        let pe = x.mul_add(y, -p);
        let s = sum + p;
        let bp = s - sum;
        let se = (sum - (s - bp)) + (p - bp);
        sum = s;
        compensation += pe + se;
    }
    sum + compensation
}

fn accelerator() -> Option<Device> {
    Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault")
}

#[test]
fn kl_remains_consistent_with_its_gradient_when_probabilities_underflow() {
    let mut devices = vec![Device::host()];
    devices.extend(accelerator());
    for device in devices {
        // The teacher assigns substantial mass to a class whose candidate probability underflows.
        let target = ndarray::array![[0.0, 1.0], [0.0, -1000.0], [0.0, 1.0]];
        let logits = ndarray::array![[0.0, -1000.0], [0.0, -1000.0], [0.0, -1000.0]];
        let flags = device.upload_indices(&[1, 1, 0]).expect("flags");
        let teacher = up(&device, &target);
        let mut gradient = up(&device, &logits);
        let kl = device.kl_rows(&teacher, &mut gradient, Some(&flags)).expect("kl");
        let p = 1.0 / (1.0 + (-1.0_f64).exp());
        let expected = p * (1000.0 + p.ln()) + (1.0 - p) * (1.0 - p).ln();
        assert!((kl[0] - expected).abs() < 1e-10, "{}: {} against {expected}", device.name(), kl[0]);
        assert_eq!(kl[1], 0.0, "identical distributions, including zero teacher mass");
        assert_eq!(kl[2], 0.0, "unscored loss");
        let gradient = down(&device, &gradient);
        assert!((gradient[[0, 0]] - p).abs() < 1e-14);
        assert!((gradient[[0, 1]] + p).abs() < 1e-14);
        assert!(gradient.row(1).iter().chain(gradient.row(2).iter()).all(|g| *g == 0.0));
        let step = 1e-3;
        let mut losses = Vec::new();
        for delta in [-step, step] {
            let mut perturbed = logits.clone();
            perturbed[[0, 1]] += delta;
            losses.push(device.kl_rows(&teacher, &mut up(&device, &perturbed), Some(&flags)).expect("perturbed kl")[0]);
        }
        let numeric = (losses[1] - losses[0]) / (2.0 * step);
        assert!((numeric - gradient[[0, 1]]).abs() < 1e-8, "loss derivative {numeric} disagrees with cotangent");
    }
}

fn up(device: &Device, m: &Array2<f64>) -> Tensor {
    device.upload(m.view()).expect("upload")
}

fn down(device: &Device, t: &Tensor) -> Array2<f64> {
    device.download(t).expect("download")
}

/// Entrywise within `band(i, j)`.
fn assert_within(what: &str, a: &Array2<f64>, b: &Array2<f64>, band: impl Fn(usize, usize) -> f64) {
    assert_eq!(a.dim(), b.dim(), "{what}: shapes");
    for ((i, j), x) in a.indexed_iter() {
        let allowed = band(i, j);
        assert!((x - b[[i, j]]).abs() <= allowed, "{what} ({i},{j}): {x} against {} (band {allowed:e})", b[[i, j]]);
    }
}

#[test]
fn host_products_lie_inside_their_bands() {
    let host = Device::host();
    let (a, b) = (matrix(9, 33, 3, 2.0), matrix(33, 7, 5, 1.5));
    let reference = Array2::from_shape_fn((9, 7), |(i, j)| exact_dot(a.row(i), b.column(j)));
    let magnitude = a.mapv(f64::abs).dot(&b.mapv(f64::abs));
    for (ta, tb) in [(Op::N, Op::N), (Op::T, Op::N), (Op::N, Op::T), (Op::T, Op::T)] {
        let left = if ta == Op::T { a.t().to_owned() } else { a.clone() };
        let right = if tb == Op::T { b.t().to_owned() } else { b.clone() };
        let mut c = host.zeros(9, 7).expect("zeros");
        host.gemm(&mut c, 1.0, &up(&host, &left), ta, &up(&host, &right), tb, 0.0, Arithmetic::F64).expect("gemm");
        assert_within("host f64 product", &down(&host, &c), &reference, |i, j| gamma(35) * magnitude[[i, j]]);
    }
    let mut c = host.zeros(9, 7).expect("zeros");
    host.gemm(&mut c, 1.0, &up(&host, &a), Op::N, &up(&host, &b), Op::N, 0.0, Arithmetic::F32).expect("gemm");
    let band = GemmBand::derive(DeviceArithmetic::F32, a.view(), b.view()).expect("an f32 band");
    assert_within("host f32 product", &down(&host, &c), &reference, |i, j| band.entry(i, j));
}

#[test]
fn device_operations_agree_with_the_host() {
    let Some(device) = accelerator() else { return };
    let host = Device::host();
    let both = |m: &Array2<f64>| (up(&host, m), up(&device, m));
    let (rows, cols, inner) = (96, 70, 300);
    let (a, b) = (matrix(rows, inner, 11, 1.0), matrix(inner, cols, 13, 1.0));
    let magnitude = a.mapv(f64::abs).dot(&b.mapv(f64::abs));
    for arithmetic in [Arithmetic::F64, Arithmetic::F32, Arithmetic::Tf32] {
        let (ha, da) = both(&a);
        let (hb, db) = both(&b);
        let (mut hc, mut dc) = (host.zeros(rows, cols).expect("zeros"), device.zeros(rows, cols).expect("zeros"));
        host.gemm(&mut hc, 1.0, &ha, Op::N, &hb, Op::N, 0.0, Arithmetic::F64).expect("host gemm");
        device.gemm(&mut dc, 1.0, &da, Op::N, &db, Op::N, 0.0, arithmetic).expect("device gemm");
        let relative = match arithmetic {
            Arithmetic::F64 => 2.0 * gamma(inner + 2),
            // Both operands rounded, then f32 sums: `(k + 2)` f32 roundings of the magnitude.
            other => 2.0 * (inner + 2) as f64 * other.unit_roundoff() + (inner + 2) as f64 * f64::from(f32::EPSILON),
        };
        assert_within(&format!("{arithmetic:?} product"), &down(&device, &dc), &down(&host, &hc), |i, j| relative * magnitude[[i, j]]);
    }
    // Strided-batched products: four blocks of a scores-shaped product.
    let (q, k) = (matrix(4 * 16, 8, 17, 1.0), matrix(4 * 16, 8, 19, 1.0));
    let (hq, dq) = both(&q);
    let (hk, dk) = both(&k);
    let (mut hs, mut ds) = (host.zeros(64, 16).expect("zeros"), device.zeros(64, 16).expect("zeros"));
    host.gemm_batched(4, &mut hs, 0.5, &hq, Op::N, &hk, Op::T, 0.0, Arithmetic::F64).expect("host batched");
    device.gemm_batched(4, &mut ds, 0.5, &dq, Op::N, &dk, Op::T, 0.0, Arithmetic::F64).expect("device batched");
    assert_within("batched product", &down(&device, &ds), &down(&host, &hs), |_, _| 2.0 * gamma(10) * 0.5 * 8.0);
    // Causal softmax of the scores, and its backward map.
    host.softmax_rows(&mut hs, true).expect("host softmax");
    device.softmax_rows(&mut ds, true).expect("device softmax");
    let alpha = down(&host, &hs);
    assert_within("causal softmax", &down(&device, &ds), &alpha, |_, _| gamma(16 + 8));
    let dalpha = matrix(64, 16, 23, 1.0);
    let (hd, dd) = both(&dalpha);
    let (hb, db) = (host.softmax_backward(&hs, &hd).expect("host"), device.softmax_backward(&ds, &dd).expect("device"));
    assert_within("softmax backward", &down(&device, &db), &down(&host, &hb), |_, _| 4.0 * gamma(16 + 8));
    // Elementwise maps, norms, laws, rotations, gathers.
    let x = matrix(rows, cols, 29, 3.0);
    let g = matrix(rows, cols, 31, 1.0);
    let (hx, dx) = both(&x);
    let (hg, dg) = both(&g);
    let scale = x.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for epsilon in [1e-6] {
        let pairs = [
            (host.rms_norm(&hx, epsilon), device.rms_norm(&dx, epsilon)),
            (host.rms_norm_backward(&hx, &hg, epsilon), device.rms_norm_backward(&dx, &dg, epsilon)),
            (host.rms_norm_tangent(&hx, &hg, epsilon), device.rms_norm_tangent(&dx, &dg, epsilon)),
        ];
        for (name, (h, d)) in ["rms", "rms backward", "rms tangent"].into_iter().zip(pairs) {
            let reference = down(&host, &h.expect("host"));
            let magnitude = reference.iter().fold(scale, |m, v| m.max(v.abs()));
            assert_within(name, &down(&device, &d.expect("device")), &reference, |_, _| gamma(cols + 16) * magnitude);
        }
    }
    let laws = [PointwiseLaw::Relu, PointwiseLaw::Identity, PointwiseLaw::Zero, PointwiseLaw::Silu, PointwiseLaw::Gelu, PointwiseLaw::GeluTanh];
    let codes: Vec<u32> = (0..cols).map(|c| laws[c % laws.len()].code()).collect();
    let (hcodes, dcodes) = (host.upload_indices(&codes).expect("codes"), device.upload_indices(&codes).expect("codes"));
    let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
    let values = down(&host, &host.law_values(&hx, &hcodes, c).expect("host laws"));
    assert_within("laws", &down(&device, &device.law_values(&dx, &dcodes, c).expect("device laws")), &values, |i, j| {
        16.0 * U * (x[[i, j]].abs() + 1.0)
    });
    let slopes = down(&host, &host.law_slopes(&hg, &hx, &hcodes, c).expect("host slopes"));
    assert_within("law slopes", &down(&device, &device.law_slopes(&dg, &dx, &dcodes, c).expect("device slopes")), &slopes, |i, j| {
        32.0 * U * g[[i, j]].abs() * (x[[i, j]].abs() + 1.0).powi(3)
    });
    let planes = 5;
    let angles = matrix(rows, planes, 37, 3.0);
    let (hcos, dcos) = both(&angles.mapv(f64::cos));
    let (hsin, dsin) = both(&angles.mapv(f64::sin));
    for (half, inverse) in [(true, false), (false, true)] {
        let h = down(&host, &host.rotate(&hx, &hcos, &hsin, half, inverse).expect("host rotate"));
        let d = down(&device, &device.rotate(&dx, &dcos, &dsin, half, inverse).expect("device rotate"));
        assert_eq!(h, d, "a rotation is one rounded expression per entry, the same on both");
    }
    let table = matrix(40, cols, 41, 1.0);
    let ids: Vec<u32> = (0..rows as u32).map(|r| (r * 7) % 40).collect();
    let (ht, dt) = both(&table);
    let gathered = down(&device, &device.gather_rows(&dt, &device.upload_indices(&ids).expect("ids")).expect("gather"));
    assert_eq!(gathered, down(&host, &host.gather_rows(&ht, &host.upload_indices(&ids).expect("ids")).expect("gather")));
    // In place and accumulating maps.
    let row = matrix(1, cols, 43, 1.0);
    let (hr, dr) = both(&row);
    let (mut hy, mut dy) = both(&g);
    host.axpy(&mut hy, 0.25, &hx).expect("axpy");
    device.axpy(&mut dy, 0.25, &dx).expect("axpy");
    host.add_row(&mut hy, -1.0, &hr).expect("add_row");
    device.add_row(&mut dy, -1.0, &dr).expect("add_row");
    host.scale_columns(&mut hy, &hx, &hr, true).expect("scale");
    device.scale_columns(&mut dy, &dx, &dr, true).expect("scale");
    host.hadamard(&mut hy, &hx, &hg, true).expect("hadamard");
    device.hadamard(&mut dy, &dx, &dg, true).expect("hadamard");
    assert_eq!(down(&device, &dy), down(&host, &hy), "elementwise maps round alike (no contraction)");
    // The KL rows, the sampled cotangent and the Fisher quadratic, with some rows unscored.
    let (classes, n) = (1000, 37);
    let logits = matrix(n, classes, 47, 6.0);
    let target = matrix(n, classes, 53, 6.0);
    let flags: Vec<u32> = (0..n as u32).map(|r| u32::from(r % 5 != 3)).collect();
    let (ht, dt) = both(&target);
    let (mut hl, mut dl) = both(&logits);
    let (hf, df) = (host.upload_indices(&flags).expect("flags"), device.upload_indices(&flags).expect("flags"));
    let hk = host.kl_rows(&ht, &mut hl, Some(&hf)).expect("host kl");
    let dk = device.kl_rows(&dt, &mut dl, Some(&df)).expect("device kl");
    for r in 0..n {
        assert!((hk[r] - dk[r]).abs() <= gamma(classes + 8) * 40.0, "kl row {r}: {} against {}", dk[r], hk[r]);
    }
    assert_within("kl cotangent", &down(&device, &dl), &down(&host, &hl), |_, _| gamma(classes + 8));
    let uniforms = matrix(n, 1, 59, 0.5).mapv(|v| v + 0.5);
    let (hu, du) = both(&uniforms);
    let (mut hs, mut ds) = both(&logits);
    host.sampled_cotangent(&mut hs, &hu, None).expect("host sampled");
    device.sampled_cotangent(&mut ds, &du, None).expect("device sampled");
    let (hs, ds) = (down(&host, &hs), down(&device, &ds));
    for r in 0..n {
        // The label is the one entry below zero; away from a tie at the uniform both pick it.
        let label = |m: &Array2<f64>| m.row(r).iter().position(|v| *v < -0.5).expect("a label");
        assert_eq!(label(&hs), label(&ds), "row {r}'s label");
    }
    assert_within("sampled cotangent", &ds, &hs, |_, _| gamma(classes + 8));
    let tangent = matrix(n, classes, 61, 1.0);
    let (hl, dl) = both(&logits);
    let (htan, dtan) = both(&tangent);
    let hq = host.softmax_quadratic(&hl, &htan).expect("host quadratic");
    let dq = device.softmax_quadratic(&dl, &dtan).expect("device quadratic");
    for r in 0..n {
        assert!((hq[r] - dq[r]).abs() <= gamma(2 * classes + 16) * 4.0, "quadratic row {r}");
    }
}

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
        let mut scratch = up(&device, &logits);
        assert_eq!(device.kl_score_rows(&teacher, &mut scratch, Some(&flags)).expect("score only"), kl);
        assert_eq!(down(&device, &scratch), logits);
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
        // Normalize before adding logarithms, so even a large common logit offset cancels.
        let shifted_teacher = up(&device, &target.mapv(|x| x + 1e15));
        let mut shifted_logits = up(&device, &logits.mapv(|x| x - 1e15));
        let shifted = device.kl_rows(&shifted_teacher, &mut shifted_logits, Some(&flags)).expect("shifted kl");
        assert_eq!(shifted, kl);
        assert_eq!(down(&device, &shifted_logits), gradient);
    }
}

fn up(device: &Device, m: &Array2<f64>) -> Tensor {
    device.upload(m.view()).expect("upload")
}

#[test]
fn grouped_mask_fisher_squares_sums_with_cross_terms() {
    let mut devices = vec![Device::host()];
    devices.extend(accelerator());
    for device in devices {
        let blocks = device.column_blocks(&[2, 1]).expect("blocks");
        let left = up(&device, &ndarray::array![[2.0, -2.0, 3.0], [1.0, 2.0, -1.0]]);
        let right = up(&device, &ndarray::array![[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let sums = device.block_products(&left, &right, &blocks).expect("reduce");
        assert_eq!(down(&device, &sums), ndarray::array![[0.0, 6.0], [11.0, -5.0]]);
        let mut fisher = device.zeros(2, 2).expect("fisher");
        device.hadamard(&mut fisher, &sums, &sums, true).expect("square sums");
        assert_eq!(down(&device, &fisher), ndarray::array![[0.0, 36.0], [121.0, 25.0]]);
        assert!(device.column_blocks(&[0]).is_err());
        assert!(device.block_products(&left, &right, &device.column_blocks(&[2]).expect("blocks")).is_err());
    }
}

#[test]
fn sampled_head_lookup_matches_full_pullback_in_both_orientations() {
    let mut devices = vec![Device::host()];
    devices.extend(accelerator());
    let logits = ndarray::array![[0.0, 1.0, -1000.0], [1.0, 0.0, 2.0], [0.0, 0.0, 0.0]];
    let head = ndarray::array![[2.0, -3.0], [0.5, 7.0], [-1.0, 4.0]];
    for device in devices {
        let flags = device.upload_indices(&[1, 0, 1]).expect("flags");
        for transposed in [false, true] {
            let weights = up(&device, &if transposed { head.t().to_owned() } else { head.clone() });
            let op = if transposed { Op::T } else { Op::N };
            let mut probabilities = up(&device, &logits);
            device.softmax_rows(&mut probabilities, false).expect("softmax");
            let mut mean = device.zeros(3, 2).expect("mean");
            device.gemm(&mut mean, 1.0, &probabilities, Op::N, &weights, op, 0.0, Arithmetic::F64).expect("mean projection");
            for u in [0.0, 0.2, 0.7, 1.0 - f64::EPSILON] {
                let uniforms = up(&device, &Array2::from_elem((3, 1), u));
                let shared = device.sampled_head_cotangent(&probabilities, &mean, &weights, transposed, &uniforms, Some(&flags)).expect("lookup");
                let mut cotangent = up(&device, &logits);
                device.sampled_cotangent(&mut cotangent, &uniforms, Some(&flags)).expect("sample");
                let mut reference = device.zeros(3, 2).expect("reference");
                device.gemm(&mut reference, 1.0, &cotangent, Op::N, &weights, op, 0.0, Arithmetic::F64).expect("pullback");
                assert_within("sampled head lookup", &down(&device, &shared), &down(&device, &reference), |_, _| 1e-13);
                assert!(down(&device, &shared).row(1).iter().all(|g| *g == 0.0));
            }
        }
    }
}

/// A drawn head sweep (`Device::head_log_partition_drawn`) on the host, a CUDA device in float64
/// and in f32 storage, and the Apple GPU: its partitions and expected rows are the undrawn sweep's;
/// each scored row's draw is its expected row less one class's head row, rounded once (an unscored
/// row's is zero); and over rows of one hidden vector against classes spanning several of CUDA's
/// swept chunks (the last narrower than a block), the draws of each of a few sets of classes fall
/// within five binomial standard errors of the set's softmax probability.
#[test]
fn a_drawn_head_sweep_draws_each_class_with_its_softmax_probability() {
    let (rows, classes, width) = (6000, 3940, 8);
    let mut devices = vec![Device::host()];
    devices.extend(accelerator());
    devices.extend(Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault").filter(|d| !d.float64()));
    // f32 values, so every device holds the same ones.
    let single = |m: Array2<f64>| m.mapv(|v| f64::from(v as f32));
    let vector = single(matrix(1, width, 5, 3.0));
    let hidden = Array2::from_shape_fn((rows, width), |(_, j)| vector[[0, j]]);
    let embedding = single(matrix(classes, width, 9, 0.7));
    let uniforms = single(matrix(rows, 1, 13, 0.5).mapv(|v| v + 0.5));
    let flags: Vec<u32> = (0..rows as u32).map(|r| u32::from(r % 7 != 3)).collect();
    let logits = embedding.dot(&vector.row(0));
    let top = logits.fold(f64::NEG_INFINITY, |a, b| a.max(*b));
    let weights = logits.mapv(|z| (z - top).exp());
    let q = &weights / weights.sum();
    let mut order: Vec<usize> = (0..classes).collect();
    order.sort_by(|a, b| q[*b].total_cmp(&q[*a]));
    let set = |name: String, member: &dyn Fn(usize) -> bool| (name, (0..classes).map(member).collect::<Vec<bool>>());
    let mut sets: Vec<(String, Vec<bool>)> = order[..4].iter().map(|&c| set(format!("class {c}"), &|k| k == c)).collect();
    sets.extend((0..5).map(|m| set(format!("classes {m} mod 5"), &|k| k % 5 == m)));
    sets.extend([(0, 1000), (1000, 2560), (2560, 3840), (3840, classes)].map(|(a, b)| set(format!("classes {a}..{b}"), &|k| a <= k && k < b)));
    for device in &devices {
        let arithmetic = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
        let (h, e, u) = (up(device, &hidden), up(device, &embedding), up(device, &uniforms));
        let f = device.upload_indices(&flags).expect("flags");
        let mut plain = device.zeros(rows, width).expect("zeros");
        let partitions = device.head_log_partition(&h, &e, false, Some(&f), Some(&mut plain), arithmetic).expect("the sweep");
        let (mut mean, mut draws) = (device.zeros(rows, width).expect("zeros"), device.zeros(rows, width).expect("zeros"));
        let drawn = device.head_log_partition_drawn(&h, (&e, false), Some(&f), &mut mean, (&u, &mut draws), arithmetic).expect("the drawn sweep");
        let name = device.name();
        assert_eq!(drawn, partitions, "{name}: the partitions");
        let (mean, draws) = (down(device, &mean), down(device, &draws));
        assert_eq!(mean, down(device, &plain), "{name}: the expected rows");
        let mut counts = vec![0usize; classes];
        for r in 0..rows {
            if flags[r] == 0 {
                assert!(draws.row(r).iter().all(|v| *v == 0.0), "{name} row {r}: an unscored row's draw");
                continue;
            }
            let gap = |c: usize| (0..width).map(|j| (mean[[r, j]] - draws[[r, j]] - embedding[[c, j]]).powi(2)).sum::<f64>();
            let label = (0..classes).min_by(|a, b| gap(*a).total_cmp(&gap(*b))).expect("a class");
            for j in 0..width {
                let exact = mean[[r, j]] - embedding[[label, j]];
                let band = if device.float64() { 0.0 } else { exact.abs() / 16_777_216.0 };
                assert!((draws[[r, j]] - exact).abs() <= band, "{name} row {r}: draw {} against {exact} less class {label}", draws[[r, j]]);
            }
            counts[label] += 1;
        }
        let scored = flags.iter().filter(|f| **f != 0).count() as f64;
        for (set, members) in &sets {
            let p: f64 = (0..classes).filter(|c| members[*c]).map(|c| q[c]).sum();
            let n = (0..classes).filter(|c| members[*c]).map(|c| counts[c]).sum::<usize>() as f64;
            let error = (scored * p * (1.0 - p)).sqrt();
            assert!((n - scored * p).abs() <= 5.0 * error + 1.0, "{name}: {set} drawn {n} times against {} (standard error {error})", scored * p);
        }
    }
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

#[test]
fn softmax_supports_rectangular_vocabulary_rows_and_causal_tile_offsets() {
    let mut devices = vec![Device::host()];
    devices.extend(accelerator());
    for d in devices {
        let mut logits = d.zeros(3, 7).expect("zeros");
        d.softmax_rows(&mut logits, false).expect("rectangular softmax");
        assert!(down(&d, &logits).iter().all(|p| (*p - 1.0 / 7.0).abs() < 1e-15));
        let mut tile = d.zeros(3, 7).expect("zeros");
        d.softmax_rows_offset(&mut tile, true, 2).expect("causal offset");
        let tile = down(&d, &tile);
        for ((r, c), p) in tile.indexed_iter() {
            let expected = if c <= r + 2 { 1.0 / (r + 3) as f64 } else { 0.0 };
            assert!((p - expected).abs() < 1e-15);
        }
    }
}

#[test]
fn proposal_gemm_workspace_handles_shape_changes_accumulation_and_overwrite() {
    let Some(d) = accelerator() else { return };
    for arithmetic in [Arithmetic::F32, Arithmetic::Tf32] {
        // Shrink and then grow independently in each dimension to exercise cached capacities.
        for (m, n, k) in [(17, 13, 31), (3, 5, 7), (23, 9, 19), (4, 21, 11)] {
            let a = matrix(m, k, 101, 0.5);
            let b = matrix(k, n, 103, 0.5);
            let (ad, bd) = (up(&d, &a), up(&d, &b));
            let reference = a.dot(&b);
            let mut c = up(&d, &Array2::from_elem((m, n), f64::NAN));
            d.gemm(&mut c, 1.0, &ad, Op::N, &bd, Op::N, 0.0, arithmetic).expect("overwrite");
            let tolerance = k as f64 * arithmetic.unit_roundoff() * 8.0;
            assert_within("overwrite stale output", &down(&d, &c), &reference, |_, _| tolerance);
            d.gemm(&mut c, 0.5, &ad, Op::N, &bd, Op::N, 0.25, arithmetic).expect("accumulate");
            assert_within("accumulate current output", &down(&d, &c), &(&reference * 0.75), |_, _| tolerance);
        }
    }
}

#[test]
fn the_device_eigendecomposition_agrees_with_the_host() {
    use gam_linalg::{decompose::eigh, roundoff::{SymmetricAssembly, symmetric_spectrum_rounding_band}};
    let Some(device) = accelerator() else { return };
    // A Gram matrix of rank 40 in 96 dimensions (eigenvalues at zero, as a removal's scaled Gram
    // has) and a full-rank one with a spread spectrum.
    for (rows, n, seed) in [(40, 96, 3), (400, 200, 5)] {
        let b = matrix(rows, n, seed, 1.0);
        let a = b.t().dot(&b);
        let Some((values, vectors)) = device.symmetric_eigh(a.view()).expect("the device decomposes") else { return };
        let host = eigh(a.view(), SymmetricAssembly::Mirrored, None).expect("the host decomposes");
        let band = symmetric_spectrum_rounding_band(values.as_slice().expect("contiguous"));
        assert!(values.windows(2).into_iter().all(|w| w[0] <= w[1]), "increasing eigenvalues");
        for (d, h) in values.iter().zip(&host.values) {
            assert!((d - h).abs() <= band + host.band, "eigenvalue {d} on the device, {h} on the host, bands {band} and {}", host.band);
        }
        // Orthonormal columns, and the matrix rebuilt from them, within the band.
        let gram = vectors.t().dot(&vectors);
        let unit = n as f64 * f64::EPSILON;
        for ((i, j), v) in gram.indexed_iter() {
            assert!((v - if i == j { 1.0 } else { 0.0 }).abs() <= unit, "column products ({i}, {j}) = {v}");
        }
        let rebuilt = vectors.dot(&ndarray::Array2::from_diag(&values)).dot(&vectors.t());
        for ((i, j), v) in rebuilt.indexed_iter() {
            assert!((v - a[[i, j]]).abs() <= band, "entry ({i}, {j}): {v} rebuilt, {} given", a[[i, j]]);
        }
    }
}

#[test]
fn the_split_gram_lies_within_its_bound_and_is_timed_against_the_float64_product() {
    use gam_gpu::tensor::Storage;
    let Some(device) = accelerator() else { return };
    let Ok(narrow) = device.with_storage(Storage::F32) else { return };
    let wide = device.with_storage(Storage::F64).expect("float64 storage beside f32");
    let slices = 9;
    // Entries spread over 2^-30 .. 2^5, exact zeros, and a zero column, rounded to f32 as the
    // device holds them.
    let (rows, cols) = (1000, 96);
    let spread = matrix(rows, cols, 11, 1.0);
    let a = Array2::from_shape_fn((rows, cols), |(k, j)| {
        let scale = 2f64.powi(5 - ((k * 7 + j * 13) % 36) as i32);
        if j == 5 || (k + j) % 17 == 0 { 0.0 } else { f64::from((spread[[k, j]] * scale) as f32) }
    });
    let mut c = wide.zeros(cols, cols).expect("sum");
    if !wide.gram_split(&mut c, &narrow.upload(a.view()).expect("upload"), slices).expect("split Gram") {
        return;
    }
    let c = wide.download(&c).expect("download");
    let exponent = |j: usize| {
        let m = a.column(j).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if m == 0.0 { 0 } else { m.log2().floor() as i32 + 1 }
    };
    for i in 0..cols {
        for j in 0..cols {
            let exact = exact_dot(a.column(i), a.column(j));
            let magnitude: f64 = a.column(i).iter().zip(a.column(j).iter()).map(|(x, y)| (x * y).abs()).sum();
            let bound = (slices + 4) as f64 * rows as f64 * 2f64.powi(exponent(i) + exponent(j) - 7 * slices as i32) + gamma(slices + 1) * magnitude;
            assert!((c[[i, j]] - exact).abs() <= bound, "entry ({i}, {j}): {} against {exact}, bound {bound}", c[[i, j]]);
        }
    }
    // One vpd4l MLP's batch, 4096 rows of 3072 functions: the split Gram against the float64
    // rank-k update of the same values.
    let (rows, cols) = (4096, 3072);
    let a = matrix(rows, cols, 17, 1.0).mapv(|v| f64::from(v as f32));
    let (narrow_a, wide_a) = (narrow.upload(a.view()).expect("upload"), wide.upload(a.view()).expect("upload"));
    let mut split = wide.zeros(cols, cols).expect("sum");
    let mut full = wide.zeros(cols, cols).expect("sum");
    wide.gram_split(&mut split, &narrow_a, slices).expect("warm");
    wide.gram_lower(&mut full, &wide_a, 1.0).expect("warm");
    assert_eq!(wide.download(&full).expect("sync").dim(), (cols, cols));
    let repeats = 5;
    let started = std::time::Instant::now();
    for _ in 0..repeats {
        wide.gram_split(&mut split, &narrow_a, slices).expect("split Gram");
    }
    assert_eq!(wide.download(&split).expect("sync").dim(), (cols, cols));
    let split_seconds = started.elapsed().as_secs_f64() / repeats as f64;
    let started = std::time::Instant::now();
    for _ in 0..repeats {
        let wide_now = wide.convert(&narrow_a).expect("widen");
        wide.gram_lower(&mut full, &wide_now, 1.0).expect("float64 Gram");
    }
    assert_eq!(wide.download(&full).expect("sync").dim(), (cols, cols));
    let full_seconds = started.elapsed().as_secs_f64() / repeats as f64;
    println!("split Gram {rows} x {cols} in {slices} slices: {:.2} ms; float64 rank-k update with the widening: {:.2} ms", split_seconds * 1e3, full_seconds * 1e3);
    // Both sums hold 1 + repeats products: their lower triangles agree within the two bounds.
    let (split, full) = (wide.download(&split).expect("split"), wide.download(&full).expect("full"));
    let norms: Vec<f64> = (0..cols).map(|j| a.column(j).dot(&a.column(j))).collect();
    for i in (0..cols).step_by(97) {
        for j in (0..=i).step_by(89) {
            let tolerance = (1 + repeats) as f64 * 2.0 * gamma(rows) * (norms[i] * norms[j]).sqrt();
            assert!((split[[i, j]] - full[[i, j]]).abs() <= tolerance, "entry ({i}, {j}): {} split, {} float64", split[[i, j]], full[[i, j]]);
        }
    }
}

/// The split Gram's time by slice count on one vpd4l MLP batch (4096 rows × 3072 functions), to
/// separate its int8 products from its float64 combines: `s` slices make `Σ_m (⌊m/2⌋ + 1)`
/// products over the shifts `m < s` and `s` combines.
#[test]
fn the_split_gram_is_timed_by_its_slices() {
    use gam_gpu::tensor::Storage;
    let Some(device) = accelerator() else { return };
    let Ok(narrow) = device.with_storage(Storage::F32) else { return };
    let wide = device.with_storage(Storage::F64).expect("float64 storage beside f32");
    let (rows, cols) = (4096, 3072);
    let a = narrow.upload(matrix(rows, cols, 23, 1.0).view()).expect("upload");
    let mut sum = wide.zeros(cols, cols).expect("sum");
    for slices in [1, 2, 3, 5, 9] {
        if !wide.gram_split(&mut sum, &a, slices).expect("warm") {
            return;
        }
        assert_eq!(wide.download(&sum).expect("sync").dim(), (cols, cols));
        let started = std::time::Instant::now();
        for _ in 0..5 {
            wide.gram_split(&mut sum, &a, slices).expect("split Gram");
        }
        assert_eq!(wide.download(&sum).expect("sync").dim(), (cols, cols));
        let products: usize = (0..slices).map(|m| m / 2 + 1).sum();
        println!("split Gram {rows} x {cols}, {slices} slices ({products} int8 products, {slices} combines): {:.2} ms", started.elapsed().as_secs_f64() / 5.0 * 1e3);
    }
}

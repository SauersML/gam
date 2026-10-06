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
use gam_gpu::tensor::{Arithmetic, Device, Op, PointwiseLaw, Tensor, fisher_sign};
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

/// The host, a CUDA device in float64 when one resolves, and the single-precision device (CUDA in
/// f32 storage, or the Apple GPU) when one does.
fn every_device() -> Vec<Device> {
    let mut devices = vec![Device::host()];
    devices.extend(accelerator());
    devices.extend(Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault").filter(|d| !d.float64()));
    devices
}

/// The Fisher probe's signs are one function of `(key, row, class)` on every device
/// ([`fisher_sign`]), so a probe is regenerated alike everywhere. On rows of equal probabilities
/// `2⁻¹²` over 4096 classes every step is exact in f32 (`√π = 2⁻⁶`, the sum of `±2⁻⁶`, and
/// `b = 2⁻⁶ ξ − 2⁻¹² (√π · ξ)`), so the host and CUDA (float64 and f32) give the probe made from the
/// counters on the host bit for bit. The Apple GPU's `sqrt` is within 4 ulps (`8u` relative) and
/// it sums in f32, so there each entry lies within `32u √π + π (n + 8) u Σ √π` of it (`n` the
/// classes). Everywhere each scored entry has the sign of its `ξ_c`, as
/// `|b_c − √π ξ_c| = √π |Σ ξ| / n < √π`.
/// Row `r` carries the signs of row `first + r`, so two calls on the rows split in two are the call
/// on all of them, and an unscored row is zero.
#[test]
fn fisher_probe_signs_are_the_same_on_every_device() {
    let (rows, classes, first, key) = (5, 4096, 70_000, 0xA5A5_5A5A_DEAD_BEEF_u64);
    let flags: Vec<u32> = (0..rows as u32).map(|r| u32::from(r != 2)).collect();
    let (pi, root) = (1.0 / classes as f64, 1.0 / 64.0);
    let signs: Vec<Vec<f64>> = (0..rows).map(|r| (0..classes).map(|c| fisher_sign(key, (first + r) as u64, c as u64)).collect()).collect();
    let exact = Array2::from_shape_fn((rows, classes), |(r, c)| root * signs[r][c] - pi * (root * signs[r].iter().sum::<f64>()));
    let probabilities = Array2::from_elem((rows, classes), pi);
    let single_unit = f64::from(f32::EPSILON) / 2.0;
    for device in every_device() {
        let name = device.name();
        let band = if cfg!(target_os = "macos") && !device.float64() {
            32.0 * single_unit * root + pi * (classes + 8) as f64 * single_unit * classes as f64 * root
        } else {
            0.0
        };
        let check = |what: &str, probe: &Array2<f64>, at: usize, scored: &dyn Fn(usize) -> bool| {
            for ((i, c), b) in probe.indexed_iter() {
                let r = at + i;
                if !scored(r) {
                    assert_eq!(*b, 0.0, "{name} {what} row {r}: an unscored row");
                    continue;
                }
                assert_eq!(b.signum(), signs[r][c], "{name} {what} ({r},{c}): the sign");
                assert!((b - exact[[r, c]]).abs() <= band, "{name} {what} ({r},{c}): {b} against {}", exact[[r, c]]);
            }
        };
        let mut whole = up(&device, &probabilities);
        device.fisher_probe_cotangent(&mut whole, (key, first), Some(&device.upload_indices(&flags).expect("flags"))).expect("the probe");
        check("whole", &down(&device, &whole), 0, &|r: usize| flags[r] != 0);
        for (at, n) in [(0, 2), (2, rows - 2)] {
            let mut part = up(&device, &Array2::from_elem((n, classes), pi));
            device.fisher_probe_cotangent(&mut part, (key, first + at), None).expect("the probe of a part");
            check("part", &down(&device, &part), at, &|_: usize| true);
        }
    }
}

/// A probe's outer products average to the softmax's Fisher matrix, on every device: with `π` the
/// stored probabilities, `L = diag √π − π √πᵀ` and `b = L ξ`, `E[b bᵀ] = L Lᵀ = diag π − (2 − Σ π) π πᵀ`
/// (`diag π − π πᵀ` when `Σ π = 1`), each of `R` rows of the same `π` with its own signs one draw.
/// For any `a`, `a · b` is a weighted sum of independent signs, so `E[(a · b)⁴] ≤ 3 E[(a · b)²]²` and
/// `Var(b_j b_k) ≤ E[b_j⁴]^½ E[b_k⁴]^½ ≤ 3 F_jj F_kk`: the mean over the rows lies within five of its
/// standard errors `√(3 F_jj F_kk / R)` of `F_jk`, for the rare class's `F_jj` a relative `5 √(3 / R)`,
/// where a label drawn from `π` would estimate `π_j (1 − π_j)` with relative variance
/// `(1 − 2π_j)² / (π_j (1 − π_j))` (`10⁴` here). The mean of `b` is zero within five standard errors
/// `√(F_jj / R)`.
#[test]
fn fisher_probes_average_to_the_softmax_fisher() {
    let rows = 20_000;
    // f32 values, so every device holds the same ones: one confident class and a rare one.
    let pi: Vec<f64> = [0.9_f32, 0.06, 0.03, 0.0099, 1e-4].iter().map(|p| f64::from(*p)).collect();
    let classes = pi.len();
    let total: f64 = pi.iter().sum();
    let fisher = Array2::from_shape_fn((classes, classes), |(j, k)| (if j == k { pi[j] } else { 0.0 }) - (2.0 - total) * pi[j] * pi[k]);
    let probabilities = Array2::from_shape_fn((rows, classes), |(_, c)| pi[c]);
    let n = rows as f64;
    for device in every_device() {
        let name = device.name();
        let mut t = up(&device, &probabilities);
        device.fisher_probe_cotangent(&mut t, (0x5EED, 11), None).expect("the probe");
        let b = down(&device, &t);
        let second = b.t().dot(&b) / n;
        for j in 0..classes {
            let mean = b.column(j).sum() / n;
            assert!(mean.abs() <= 5.0 * (fisher[[j, j]] / n).sqrt(), "{name}: the mean of class {j}'s entry, {mean:e}");
            for k in 0..classes {
                let error = (3.0 * fisher[[j, j]] * fisher[[k, k]] / n).sqrt();
                assert!((second[[j, k]] - fisher[[j, k]]).abs() <= 5.0 * error, "{name} ({j},{k}): {:e} against {:e} (standard error at most {error:e})", second[[j, k]], fisher[[j, k]]);
            }
        }
    }
}

/// `axpy_rows`, `axpy_rows_within` and `copy_rows_within` against the copies, axpys and writes they
/// replace, bit for bit.
fn row_moves_match_their_compositions(d: &Device) {
    let (a, b) = (up(d, &matrix(9, 13, 21, 2.0)), up(d, &matrix(7, 13, 22, 1.5)));
    let mut y = d.copy(&a).unwrap();
    d.axpy_rows(&mut y, 2, -0.375, (&b, 3), 4).unwrap();
    let mut expected = d.copy(&a).unwrap();
    let mut total = d.rows_of(&a, 2, 4).unwrap();
    d.axpy(&mut total, -0.375, &d.rows_of(&b, 3, 4).unwrap()).unwrap();
    d.set_rows(&mut expected, 2, &total).unwrap();
    assert_eq!(d.download(&y).unwrap(), d.download(&expected).unwrap(), "{} axpy_rows", d.name());
    let mut t = d.copy(&a).unwrap();
    d.axpy_rows_within(&mut t, 5, 1.0, 1, 3).unwrap();
    let mut expected = d.copy(&a).unwrap();
    let mut total = d.rows_of(&a, 5, 3).unwrap();
    d.axpy(&mut total, 1.0, &d.rows_of(&a, 1, 3).unwrap()).unwrap();
    d.set_rows(&mut expected, 5, &total).unwrap();
    assert_eq!(d.download(&t).unwrap(), d.download(&expected).unwrap(), "{} axpy_rows_within", d.name());
    let mut t = d.copy(&a).unwrap();
    d.copy_rows_within(&mut t, 0, 6, 3).unwrap();
    let mut expected = d.copy(&a).unwrap();
    d.set_rows(&mut expected, 0, &d.rows_of(&a, 6, 3).unwrap()).unwrap();
    assert_eq!(d.download(&t).unwrap(), d.download(&expected).unwrap(), "{} copy_rows_within", d.name());
    assert!(d.copy_rows_within(&mut t, 2, 3, 2).is_err(), "overlapping rows are refused");
}

#[test]
fn row_moves_are_the_copies_and_axpys_they_replace() {
    row_moves_match_their_compositions(&Device::host());
    if let Some(wide) = accelerator() {
        row_moves_match_their_compositions(&wide.with_storage(gam_gpu::tensor::Storage::F32).expect("CUDA holds f32"));
        row_moves_match_their_compositions(&wide);
    }
    if cfg!(target_os = "macos")
        && let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")
    {
        row_moves_match_their_compositions(&metal);
    }
}

/// `split_heads_bf16` and `softmax_backward_bf16` against the f32 operations and the `bf16_copy` of
/// their results, bit for bit, with and without a rotation.
fn bf16_outputs_match_their_copies(d: &Device) {
    let (blocks, length, heads, width, planes) = (2, 5, 3, 8, 3);
    let x = up(d, &matrix(blocks * length, heads * width + 4, 31, 2.0));
    let (cos, sin) = (up(d, &matrix(blocks * length, planes, 32, 1.0)), up(d, &matrix(blocks * length, planes, 33, 1.0)));
    for (what, turn) in [("unturned", None), ("interleaved", Some((&cos, &sin, false))), ("half split", Some((&cos, &sin, true)))] {
        for inverse in [false, true] {
            let split = d.split_heads_bf16(&x, 4, heads, width, blocks, turn, inverse).unwrap();
            let expected = d.bf16_copy(&d.split_heads(&x, 4, heads, width, blocks, turn, inverse).unwrap()).unwrap();
            assert_eq!(down(d, &split), down(d, &expected), "{} split {what}, inverse {inverse}", d.name());
            let (wide, half) = d.split_heads_both(&x, 4, heads, width, blocks, turn, inverse).unwrap();
            assert_eq!(down(d, &wide), down(d, &d.split_heads(&x, 4, heads, width, blocks, turn, inverse).unwrap()), "{} both splits' f32 {what}", d.name());
            assert_eq!(down(d, &half), down(d, &expected), "{} both splits' bfloat16 {what}", d.name());
        }
    }
    let (alpha, cot) = (up(d, &matrix(7, 19, 34, 1.0)), up(d, &matrix(7, 19, 35, 3.0)));
    let map = d.softmax_backward_bf16(&alpha, &cot).unwrap();
    let expected = d.bf16_copy(&d.softmax_backward(&alpha, &cot).unwrap()).unwrap();
    assert_eq!(down(d, &map), down(d, &expected), "{} softmax backward", d.name());
    for causal in [false, true] {
        let mut scores = up(d, &matrix(12, 6, 36, 2.0));
        let half = d.softmax_rows_bf16(&mut scores, causal).unwrap();
        let mut expected = up(d, &matrix(12, 6, 36, 2.0));
        d.softmax_rows(&mut expected, causal).unwrap();
        assert_eq!(down(d, &scores), down(d, &expected), "{} softmax, causal {causal}", d.name());
        assert_eq!(down(d, &half), down(d, &d.bf16_copy(&expected).unwrap()), "{} softmax's bfloat16 copy, causal {causal}", d.name());
    }
}

#[test]
fn bfloat16_outputs_are_the_copies_of_the_f32_ones() {
    bf16_outputs_match_their_copies(&Device::host());
    if let Some(wide) = accelerator() {
        bf16_outputs_match_their_copies(&wide.with_storage(gam_gpu::tensor::Storage::F32).expect("CUDA holds f32"));
    }
}

/// A probed head sweep (`Device::head_log_partition_probed`) on the host, a CUDA device in float64
/// and in f32 storage, and the Apple GPU, the head stored either way, over classes spanning several
/// of CUDA's swept chunks (the last narrower than a block): its partitions and expected rows are
/// the plain sweep's, an unscored row's probe is zero, and each scored row's probe
/// `Σ_c b_c e_c` (`b = √q ⊙ ξ − q D`, `D = √q · ξ`) lies within `ε ‖e‖_∞ Σ_c √q_c` of the probe
/// made in float64 on the host from the same signs, and within twice that of the device's own
/// unfused probe (the logits, their softmax, `Device::fisher_probe_cotangent` and the pullback).
/// `ε = 4δ + (4 s + 6n + 96) u`, `u` the device's unit roundoff, `n` the classes, `δ` the row's
/// logit error `(w + 2) u max_c Σ_j |h_j e_cj|` (`w` the width) and `s` the span of its logits: a
/// logit error `δ` moves `√q_c` by `δ √q_c`, `q_c` by `2δ q_c` and `D` by `δ Σ √q`, so
/// `Σ_c |Δb_c| |e_cj| ≤ 4δ ‖e‖_∞ Σ √q` (`Σ q = 1`, `|D| ≤ Σ √q`); an exponential's argument
/// rounds by `u s`, another such error; and the sums over the classes (`D`, the partition and the
/// pullback, `(n + 8) u` each), the exponentials and roots (`8u` each) and the last combination add
/// at most `(3n + 48) u` of `‖e‖_∞ Σ_c (√q_c + q_c |D|) ≤ 2 ‖e‖_∞ Σ √q`.
#[test]
fn a_probed_head_sweep_is_the_probe_of_its_softmax() {
    let (rows, classes, width, key) = (6000, 3940, 8, 0x0C0F_FEE5_u64);
    // f32 values, so every device holds the same ones.
    let single = |m: Array2<f64>| m.mapv(|v| f64::from(v as f32));
    let hidden = single(matrix(rows, width, 5, 3.0));
    let embedding = single(matrix(classes, width, 9, 0.7));
    let flags: Vec<u32> = (0..rows as u32).map(|r| u32::from(r % 7 != 3)).collect();
    let largest = embedding.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    // The probe in float64, and per row `Σ √q`, the logits' span and the largest `Σ_j |h_j e_cj|`.
    let mut exact = Array2::<f64>::zeros((rows, width));
    let (mut roots, mut spans, mut sizes) = (vec![0.0; rows], vec![0.0; rows], vec![0.0_f64; rows]);
    for r in 0..rows {
        let term = |c: usize, j: usize| hidden[[r, j]] * embedding[[c, j]];
        let z: Vec<f64> = (0..classes).map(|c| (0..width).map(|j| term(c, j)).sum::<f64>()).collect();
        let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let low = z.iter().copied().fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> = z.iter().map(|v| (v - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        let q: Vec<f64> = weights.iter().map(|w| w / total).collect();
        let signed: Vec<f64> = (0..classes).map(|c| q[c].sqrt() * fisher_sign(key, r as u64, c as u64)).collect();
        let dot: f64 = signed.iter().sum();
        for c in 0..classes {
            let b = signed[c] - q[c] * dot;
            for j in 0..width {
                exact[[r, j]] += b * embedding[[c, j]];
            }
            sizes[r] = sizes[r].max((0..width).map(|j| term(c, j).abs()).sum::<f64>());
        }
        roots[r] = q.iter().map(|p| p.sqrt()).sum();
        spans[r] = top - low;
    }
    for device in every_device() {
        let (arithmetic, u) = if device.float64() { (Arithmetic::F64, U) } else { (Arithmetic::F32, f64::from(f32::EPSILON) / 2.0) };
        let band = |r: usize| (4.0 * (width + 2) as f64 * u * sizes[r] + (4.0 * spans[r] + 6.0 * classes as f64 + 96.0) * u) * largest * roots[r];
        let (h, f) = (up(&device, &hidden), device.upload_indices(&flags).expect("flags"));
        for transposed in [false, true] {
            let name = format!("{} (transposed {transposed})", device.name());
            let e = up(&device, &if transposed { embedding.t().to_owned() } else { embedding.clone() });
            let mut plain = device.zeros(rows, width).expect("zeros");
            let partitions = device.head_log_partition(&h, &e, transposed, Some(&f), Some(&mut plain), arithmetic).expect("the sweep");
            let (mut mean, mut probed) = (device.zeros(rows, width).expect("zeros"), device.zeros(rows, width).expect("zeros"));
            let swept = device.head_log_partition_probed(&h, (&e, transposed), Some(&f), &mut mean, (key, &mut probed), arithmetic).expect("the probed sweep");
            assert_eq!(swept, partitions, "{name}: the partitions");
            assert_eq!(down(&device, &mean), down(&device, &plain), "{name}: the expected rows");
            // The same probe unfused on the device.
            let (into, back) = if transposed { (Op::N, Op::T) } else { (Op::T, Op::N) };
            let mut logits = device.zeros(rows, classes).expect("zeros");
            device.gemm(&mut logits, 1.0, &h, Op::N, &e, into, 0.0, arithmetic).expect("the logits");
            device.softmax_rows(&mut logits, false).expect("the softmax");
            device.fisher_probe_cotangent(&mut logits, (key, 0), Some(&f)).expect("the probe");
            let mut unfused = device.zeros(rows, width).expect("zeros");
            device.gemm(&mut unfused, 1.0, &logits, Op::N, &e, back, 0.0, arithmetic).expect("the pullback");
            let (probed, unfused) = (down(&device, &probed), down(&device, &unfused));
            for r in 0..rows {
                for j in 0..width {
                    let (swept, alone) = (probed[[r, j]], unfused[[r, j]]);
                    if flags[r] == 0 {
                        assert!(swept == 0.0 && alone == 0.0, "{name} row {r}: an unscored row's probe");
                        continue;
                    }
                    assert!((swept - exact[[r, j]]).abs() <= band(r), "{name} ({r},{j}): swept {swept} against {} (band {:e})", exact[[r, j]], band(r));
                    assert!((swept - alone).abs() <= 2.0 * band(r), "{name} ({r},{j}): swept {swept} against unfused {alone} (band {:e})", 2.0 * band(r));
                }
            }
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
    // The KL rows, the Fisher probe and the Fisher quadratic, with some rows unscored.
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
    // The probe of the host's softmax rows on both: the same signs, so each entry within the
    // summation band of `q (√q · ξ)` and the roundings of `√q` and the last two steps.
    let mut hq = up(&host, &logits);
    host.softmax_rows(&mut hq, false).expect("host softmax");
    let q = down(&host, &hq);
    let roots: Vec<f64> = q.rows().into_iter().map(|row| row.iter().map(|p| p.sqrt()).sum()).collect();
    let (mut hp, mut dp) = both(&q);
    host.fisher_probe_cotangent(&mut hp, (59, 3), Some(&hf)).expect("host probe");
    device.fisher_probe_cotangent(&mut dp, (59, 3), Some(&df)).expect("device probe");
    assert_within("fisher probe", &down(&device, &dp), &down(&host, &hp), |i, j| gamma(classes + 8) * (q[[i, j]].sqrt() + q[[i, j]] * roots[i]));
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

/// `nonzero_columns`, `gather_columns`, `scatter_columns` and `scatter_rows` on the host and the accelerator: the
/// columns nonzero on the rows asked for (a NaN counts, a column nonzero only on a row left out does
/// not), the gathered columns in order, and the columns written or added back (integers, so every
/// sum is exact in float32 too).
#[test]
fn column_reads_and_writes_agree_with_the_host() {
    let mut devices = vec![Device::host()];
    devices.extend(accelerator());
    devices.extend(Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault"));
    let (rows, cols) = (9, 13);
    let mut x = Array2::from_shape_fn((rows, cols), |(r, c)| if c % 3 == 0 { 0.0 } else { ((r * c) % 7) as f64 - 3.0 });
    x.column_mut(4).fill(0.0);
    x[[0, 4]] = 2.0;
    x[[5, 6]] = f64::NAN;
    x.column_mut(6).iter_mut().enumerate().for_each(|(r, v)| if r != 5 { *v = 0.0 });
    let every: Vec<u32> = (0..rows as u32).collect();
    let later: Vec<u32> = (1..rows as u32).collect();
    let nonzero = |rows: &[u32]| -> Vec<u32> { (0..cols as u32).filter(|&c| rows.iter().any(|&r| x[[r as usize, c as usize]] != 0.0)).collect() };
    assert!(nonzero(&every).contains(&4) && !nonzero(&later).contains(&4) && nonzero(&later).contains(&6));
    let ids: Vec<u32> = vec![5, 0, 12, 7];
    let values = Array2::from_shape_fn((rows, ids.len()), |(r, j)| (r + 2 * j) as f64 - 4.0);
    let target = Array2::from_shape_fn((rows, cols), |(r, c)| (r * 3 + c) as f64 % 5.0);
    for device in devices {
        assert_eq!(device.nonzero_columns(&up(&device, &x), &every).expect("nonzero"), nonzero(&every), "{}", device.name());
        assert_eq!(device.nonzero_columns(&up(&device, &x), &later).expect("nonzero"), nonzero(&later), "{}", device.name());
        let finite = x.mapv(|v| if v.is_nan() { 1.0 } else { v });
        let gathered = down(&device, &device.gather_columns(&up(&device, &finite), &device.upload_indices(&ids).expect("ids")).expect("gather"));
        for (j, &c) in ids.iter().enumerate() {
            assert_eq!(gathered.column(j), finite.column(c as usize), "{}: gathered column {j}", device.name());
        }
        for accumulate in [false, true] {
            let mut t = up(&device, &target);
            device.scatter_columns(&mut t, &device.upload_indices(&ids).expect("ids"), &up(&device, &values), accumulate).expect("scatter");
            let mut expected = target.clone();
            for (j, &c) in ids.iter().enumerate() {
                for r in 0..rows {
                    expected[[r, c as usize]] = if accumulate { target[[r, c as usize]] + values[[r, j]] } else { values[[r, j]] };
                }
            }
            assert_eq!(down(&device, &t), expected, "{}: scatter (accumulate {accumulate})", device.name());
            let row_ids: Vec<u32> = vec![8, 2, 5];
            let row_values = Array2::from_shape_fn((row_ids.len(), cols), |(i, c)| (i * 5 + c) as f64 - 6.0);
            let mut t = up(&device, &target);
            device.scatter_rows(&mut t, &device.upload_indices(&row_ids).expect("ids"), &up(&device, &row_values), accumulate).expect("scatter rows");
            let mut expected = target.clone();
            for (i, &r) in row_ids.iter().enumerate() {
                for c in 0..cols {
                    expected[[r as usize, c]] = if accumulate { target[[r as usize, c]] + row_values[[i, c]] } else { row_values[[i, c]] };
                }
            }
            assert_eq!(down(&device, &t), expected, "{}: scatter rows (accumulate {accumulate})", device.name());
        }
    }
}

/// The seeded sweep (`head_log_partition_seeded`, CUDA f32): its log partitions are the f32 sweep's
/// bit for bit, its expected rows within bfloat16 rounding of the f32 sweep's (each term `π e` with
/// both rounded, so within `2^-7 max |e|`), and its probe within `2^-6` of the f32 probe's norm.
#[test]
fn a_seeded_sweep_keeps_the_partitions_and_rounds_the_seeds() {
    let Some(wide) = accelerator().filter(|_| cfg!(target_os = "linux")) else { return };
    let d = wide.with_storage(gam_gpu::tensor::Storage::F32).expect("CUDA holds f32");
    let (rows, classes, width) = (37, 3001, 24);
    let single = |m: Array2<f64>| m.mapv(|v| f64::from(v as f32));
    let head_values = single(matrix(classes, width, 42, 0.5));
    let (hidden, head) = (up(&d, &single(matrix(rows, width, 41, 1.0))), up(&d, &head_values));
    let half = d.bf16_copy(&head).unwrap();
    let (mut expected, mut probed) = (d.zeros(rows, width).unwrap(), d.zeros(rows, width).unwrap());
    let partitions = d.head_log_partition_probed(&hidden, (&head, false), None, &mut expected, (7, &mut probed), Arithmetic::F32).unwrap();
    let (mut seeded_expected, mut seeded_probe) = (d.zeros(rows, width).unwrap(), d.zeros(rows, width).unwrap());
    let seeded = d.head_log_partition_seeded(&hidden, (&head, &half), None, &mut seeded_expected, Some((7, &mut seeded_probe)), Arithmetic::F32).unwrap();
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&seeded), bits(&partitions), "the log partitions");
    let largest = head_values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert_within("seeded expected rows", &down(&d, &seeded_expected), &down(&d, &expected), |_, _| 2f64.powi(-7) * largest);
    let (a, b) = (down(&d, &seeded_probe), down(&d, &probed));
    let norm = |m: &Array2<f64>| m.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm(&(&a - &b)) <= 2f64.powi(-6) * norm(&b), "seeded probe off by {} of {}", norm(&(&a - &b)), norm(&b));
}

//! The tensor operations of `gam_gpu::tensor` on the Apple GPU (f32) against their float64 host
//! twins, each within a band derived for f32 (`u = 2⁻²⁴`, `γ_n = nu/(1 − nu)`); the host's own
//! float64 error is below `2⁻²⁹` of every band and is absorbed in it.
//!
//! The device rounds every input to f32 (relative `u`) and each operation once more (no
//! contraction); a reduction of `n` terms is within `γ_n` of the magnitude it sums, in any order.
//! `exp`, `log`, `tanh` and `sqrt` (the safe math mode) are within 4 ulps, `erfc` within `1.2·10⁻⁷`
//! relative before its own roundings. A shift of logits by at most `δ` moves a softmax entry `p` by
//! at most `2δp` to first order, and a KL by at most `2δ` (the gradient `q − p` has `ℓ₁` norm at
//! most 2). Each test runs when Metal resolves (a runtime probe) and has nothing to run otherwise.
#![cfg(target_os = "macos")]

use gam_gpu::GpuPolicy;
use gam_gpu::gpu_error::GpuError;
use gam_gpu::precision_bounds::{DeviceArithmetic, GemmBand};
use gam_gpu::tensor::{Arithmetic, Device, Op, PointwiseLaw, Tensor};
use ndarray::Array2;

const U: f64 = 1.0 / 16_777_216.0;

fn gamma(n: usize) -> f64 {
    n as f64 * U / (1.0 - n as f64 * U)
}

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
        (2.0 * unit - 1.0) * scale
    })
}

/// `m` rounded to f32: what the device holds.
fn single(m: &Array2<f64>) -> Array2<f64> {
    m.mapv(|v| f64::from(v as f32))
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

fn largest(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

#[test]
fn metal_column_writes_preserve_bits_and_unselected_columns() {
    let Some(d) = metal() else { return };
    let x = single(&matrix(7, 9, 711, 2.0));
    let special = [0.0f32, -0.0, f32::from_bits(1), -f32::from_bits(1), f32::MAX, -f32::MAX];
    let part = Array2::from_shape_fn((7, 3), |(r, c)| f64::from(special[(r * 3 + c) % special.len()]));
    let mut target = up(&d, &x);
    let source = up(&d, &part);
    let mut expected = x.clone();
    for start in [2, 6, 0] {
        d.set_columns(&mut target, start, &source).expect("resident column copy");
        expected.slice_mut(ndarray::s![.., start..start + 3]).assign(&part);
        let actual = down(&d, &target);
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!((*a as f32).to_bits(), (*b as f32).to_bits());
        }
        let copied = down(&d, &d.columns_of(&target, start..start + 3).expect("resident column read"));
        for (a, b) in copied.iter().zip(&part) {
            assert_eq!((*a as f32).to_bits(), (*b as f32).to_bits());
        }
    }
    assert!(d.set_columns(&mut target, 7, &source).is_err());
    assert!(d.set_columns(&mut target, 0, &up(&d, &matrix(6, 3, 72, 1.0))).is_err());
    let empty = d.zeros(7, 0).expect("empty source");
    d.set_columns(&mut target, 9, &empty).expect("empty copy");
    assert!(d.columns_of(&target, 8..10).is_err());
    assert!(d.columns_of(&target, 3..3).is_err());
    assert_eq!(down(&d, &source), part);
}

#[test]
fn metal_is_single_precision_and_refuses_float64_products() {
    let Some(d) = metal() else { return };
    assert!(Device::host().float64());
    let (a, b) = (up(&d, &matrix(3, 4, 1, 1.0)), up(&d, &matrix(4, 2, 2, 1.0)));
    let mut c = d.zeros(3, 2).expect("zeros");
    let refused = d.gemm(&mut c, 1.0, &a, Op::N, &b, Op::N, 0.0, Arithmetic::F64);
    assert!(matches!(refused, Err(GpuError::NoDeviceKernel { .. })), "{refused:?}");
    // Moving values: f32-representable values round-trip exactly through every copy.
    let x = single(&matrix(7, 5, 3, 2.0));
    let t = up(&d, &x);
    assert_eq!(down(&d, &t), x);
    assert_eq!(down(&d, &d.copy(&t).expect("copy")), x);
    assert_eq!(down(&d, &d.rows_of(&t, 2, 3).expect("rows")), x.slice(ndarray::s![2..5, ..]).to_owned());
    let mut z = d.zeros(7, 5).expect("zeros");
    assert_eq!(down(&d, &z), Array2::<f64>::zeros((7, 5)));
    d.set_rows(&mut z, 4, &d.rows_of(&t, 0, 3).expect("rows")).expect("set rows");
    let mut expected = Array2::zeros((7, 5));
    expected.slice_mut(ndarray::s![4..7, ..]).assign(&x.slice(ndarray::s![0..3, ..]));
    assert_eq!(down(&d, &z), expected);
    assert!(d.memory().expect("memory").is_some());
}

#[test]
fn metal_products_lie_inside_the_f32_band() {
    let Some(d) = metal() else { return };
    let host = Device::host();
    let (m, n, k) = (37, 29, 300);
    let (a, b) = (matrix(m, k, 11, 1.0), matrix(k, n, 13, 1.0));
    let band = GemmBand::derive(DeviceArithmetic::F32, a.view(), b.view()).expect("an f32 band");
    let reference = a.dot(&b);
    for (ta, tb) in [(Op::N, Op::N), (Op::T, Op::N), (Op::N, Op::T), (Op::T, Op::T)] {
        for arithmetic in [Arithmetic::F32, Arithmetic::Tf32] {
            let left = if ta == Op::T { a.t().to_owned() } else { a.clone() };
            let right = if tb == Op::T { b.t().to_owned() } else { b.clone() };
            let mut c = d.zeros(m, n).expect("zeros");
            d.gemm(&mut c, 1.0, &up(&d, &left), ta, &up(&d, &right), tb, 0.0, arithmetic).expect("gemm");
            assert_within(&format!("{ta:?}{tb:?} {arithmetic:?} product"), &down(&d, &c), &reference, |i, j| band.entry(i, j));
        }
    }
    // `c ← α ab + β c`: the product's band times |α|, and three more roundings of
    // `|α| |a||b| + |β| |c|` (α's product, β's, the sum) with `c`'s own rounding.
    let c0 = matrix(m, n, 17, 2.0);
    let magnitude = a.mapv(f64::abs).dot(&b.mapv(f64::abs));
    let (alpha, beta) = (0.75, -0.5);
    let mut c = up(&d, &c0);
    d.gemm(&mut c, alpha, &up(&d, &a), Op::N, &up(&d, &b), Op::N, beta, Arithmetic::F32).expect("accumulate");
    let expected = &reference * alpha + &c0 * beta;
    assert_within("α ab + β c", &down(&d, &c), &expected, |i, j| {
        alpha.abs() * band.entry(i, j) + gamma(4) * (alpha.abs() * magnitude[[i, j]] + beta.abs() * c0[[i, j]].abs())
    });
    // An empty inner dimension scales `c` by β.
    let (empty_a, empty_b) = (d.zeros(m, 0).expect("zeros"), d.zeros(0, n).expect("zeros"));
    let mut c = up(&d, &single(&c0));
    d.gemm(&mut c, 1.0, &empty_a, Op::N, &empty_b, Op::N, 0.5, Arithmetic::F32).expect("empty");
    assert_eq!(down(&d, &c), single(&c0) * 0.5);
    // Strided-batched: four blocks of a scores-shaped product, each in its own band, times α.
    let (blocks, length, width) = (4, 16, 8);
    let (q, kk) = (matrix(blocks * length, width, 19, 1.0), matrix(blocks * length, width, 23, 1.0));
    let mut hs = host.zeros(blocks * length, length).expect("zeros");
    host.gemm_batched(blocks, &mut hs, 0.5, &up(&host, &q), Op::N, &up(&host, &kk), Op::T, 0.0, Arithmetic::F64).expect("host batched");
    let mut ds = d.zeros(blocks * length, length).expect("zeros");
    d.gemm_batched(blocks, &mut ds, 0.5, &up(&d, &q), Op::N, &up(&d, &kk), Op::T, 0.0, Arithmetic::F32).expect("device batched");
    let bands: Vec<GemmBand> = (0..blocks)
        .map(|i| {
            let rows = ndarray::s![i * length..(i + 1) * length, ..];
            GemmBand::derive(DeviceArithmetic::F32, q.slice(rows), kk.slice(rows).t()).expect("band")
        })
        .collect();
    assert_within("batched product", &down(&d, &ds), &down(&host, &hs), |i, j| 0.5 * bands[i / length].entry(i % length, j));
}

#[test]
fn metal_elementwise_maps_and_gathers_lie_inside_their_bands() {
    let Some(d) = metal() else { return };
    let host = Device::host();
    let (rows, cols) = (96, 70);
    let (x, g, row) = (matrix(rows, cols, 29, 3.0), matrix(rows, cols, 31, 1.0), matrix(1, cols, 43, 1.0));
    let both = |m: &Array2<f64>| (up(&host, m), up(&d, m));
    let ((hx, dx), (hg, dg), (hr, dr)) = (both(&x), both(&g), both(&row));
    let (mut hy, mut dy) = both(&g);
    host.axpy(&mut hy, 0.25, &hx).expect("axpy");
    d.axpy(&mut dy, 0.25, &dx).expect("axpy");
    host.add_row(&mut hy, -1.0, &hr).expect("add_row");
    d.add_row(&mut dy, -1.0, &dr).expect("add_row");
    host.scale_columns(&mut hy, &hx, &hr, true).expect("scale");
    d.scale_columns(&mut dy, &dx, &dr, true).expect("scale");
    host.hadamard(&mut hy, &hx, &hg, true).expect("hadamard");
    d.hadamard(&mut dy, &dx, &dg, true).expect("hadamard");
    // Four sums of at most M = Σ|terms| (four roundings), two products (two), the inputs' (one per
    // factor: at most 2u of each term): seven units of M, γ₈ M with the second-order terms.
    assert_within("elementwise maps", &down(&d, &dy), &down(&host, &hy), |i, j| {
        let (xv, gv, rv) = (x[[i, j]].abs(), g[[i, j]].abs(), row[[0, j]].abs());
        gamma(8) * (gv + 0.25 * xv + rv + xv * rv + xv * gv)
    });
    // Rotations: `c x_a − s x_b`, two products and a sum of rounded inputs.
    let planes = 5;
    let angles = matrix(rows, planes, 37, 3.0);
    let (cos, sin) = (angles.mapv(f64::cos), angles.mapv(f64::sin));
    let ((hc, dc), (hs, ds)) = (both(&cos), both(&sin));
    for (half, inverse) in [(true, false), (false, true)] {
        let h = down(&host, &host.rotate(&hx, &hc, &hs, half, inverse).expect("host rotate"));
        let r = down(&d, &d.rotate(&dx, &dc, &ds, half, inverse).expect("device rotate"));
        assert_within("rotation", &r, &h, |i, j| {
            let plane = if half { j % planes } else { j / 2 };
            let partner = if half { (j + planes) % (2 * planes) } else { j ^ 1 };
            if j >= 2 * planes {
                return U * x[[i, j]].abs();
            }
            gamma(5) * (cos[[i, plane]].abs() * x[[i, j]].abs() + sin[[i, plane]].abs() * x[[i, partner]].abs())
        });
    }
    // Gathers copy f32 values exactly; an id outside the table is refused.
    let table = matrix(40, cols, 41, 1.0);
    let ids: Vec<u32> = (0..rows as u32).map(|r| (r * 7) % 40).collect();
    let gathered = down(&d, &d.gather_rows(&up(&d, &table), &d.upload_indices(&ids).expect("ids")).expect("gather"));
    assert_eq!(gathered, single(&down(&host, &host.gather_rows(&up(&host, &table), &host.upload_indices(&ids).expect("ids")).expect("gather"))));
    assert!(d.gather_rows(&up(&d, &table), &d.upload_indices(&[40]).expect("ids")).is_err());
    // Laws and their slopes: the input's rounding moves a value by `|f'| u |x|`, the function
    // itself is within a few ulps (erfc's fit and its exp, about 16u of Φ), so 32u(|x| + 1)
    // bounds a value; a slope's derivative is at most (|x| + 1)³ in size on these laws.
    let laws = [PointwiseLaw::Relu, PointwiseLaw::Identity, PointwiseLaw::Zero, PointwiseLaw::Silu, PointwiseLaw::Gelu, PointwiseLaw::GeluTanh];
    let codes: Vec<u32> = (0..cols).map(|c| laws[c % laws.len()].code()).collect();
    let (hcodes, dcodes) = (host.upload_indices(&codes).expect("codes"), d.upload_indices(&codes).expect("codes"));
    let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
    let values = down(&host, &host.law_values(&hx, &hcodes, c).expect("host laws"));
    assert_within("laws", &down(&d, &d.law_values(&dx, &dcodes, c).expect("device laws")), &values, |i, j| 32.0 * U * (x[[i, j]].abs() + 1.0));
    let slopes = down(&host, &host.law_slopes(&hg, &hx, &hcodes, c).expect("host slopes"));
    assert_within("law slopes", &down(&d, &d.law_slopes(&dg, &dx, &dcodes, c).expect("device slopes")), &slopes, |i, j| {
        64.0 * U * g[[i, j]].abs() * (x[[i, j]].abs() + 1.0).powi(3)
    });
    // Root-mean-square norms: a row's reductions of `cols` terms and a dozen roundings around them,
    // of at most the largest of the values and the inputs' product.
    let scale = largest(&x) * largest(&g);
    let epsilon = 1e-6;
    let pairs = [
        (host.rms_norm(&hx, epsilon), d.rms_norm(&dx, epsilon)),
        (host.rms_norm_backward(&hx, &hg, epsilon), d.rms_norm_backward(&dx, &dg, epsilon)),
        (host.rms_norm_tangent(&hx, &hg, epsilon), d.rms_norm_tangent(&dx, &dg, epsilon)),
    ];
    for (name, (h, m)) in ["rms", "rms backward", "rms tangent"].into_iter().zip(pairs) {
        let reference = down(&host, &h.expect("host"));
        let magnitude = largest(&reference).max(scale);
        assert_within(name, &down(&d, &m.expect("device")), &reference, |_, _| gamma(cols + 16) * magnitude);
    }
}

#[test]
fn metal_row_reductions_lie_inside_their_bands() {
    let Some(d) = metal() else { return };
    let host = Device::host();
    let both = |m: &Array2<f64>| (up(&host, m), up(&d, m));
    // Causal softmax of square blocks and of a tile at an offset: `p (4u z_max + γ_{n+10})` (the
    // inputs' and the shift's roundings move the exponent by `2u z_max` each; exp, the sum and the
    // division the rest); entries past the causal edge are exactly zero.
    let (length, blocks) = (16, 4);
    let scores = matrix(blocks * length, length, 47, 4.0);
    let zmax = largest(&scores);
    for (rows, start) in [(blocks * length, None), (5, Some(7))] {
        let s = scores.slice(ndarray::s![..rows, ..]).to_owned();
        let ((mut hs, mut ds), causal) = (both(&s), true);
        match start {
            None => {
                host.softmax_rows(&mut hs, causal).expect("host softmax");
                d.softmax_rows(&mut ds, causal).expect("device softmax");
            }
            Some(start) => {
                host.softmax_rows_offset(&mut hs, causal, start).expect("host tile");
                d.softmax_rows_offset(&mut ds, causal, start).expect("device tile");
            }
        }
        let alpha = down(&host, &hs);
        assert_within("causal softmax", &down(&d, &ds), &alpha, |i, j| alpha[[i, j]] * (4.0 * U * zmax + gamma(length + 10)));
        // Its backward map: `α (d − Σ α d)`, the mean a sum of `n` products.
        let dalpha = matrix(rows, length, 53, 1.0);
        let (hd, dd) = both(&dalpha);
        let hb = down(&host, &host.softmax_backward(&hs, &hd).expect("host"));
        let db = down(&d, &d.softmax_backward(&ds, &dd).expect("device"));
        assert_within("softmax backward", &db, &hb, |i, j| {
            let mean: f64 = (0..length).map(|c| alpha[[i, c]] * dalpha[[i, c]].abs()).sum();
            alpha[[i, j]] * (dalpha[[i, j]].abs() + mean) * (4.0 * U * zmax + gamma(length + 14))
        });
    }
    // KL rows and their cotangent, some rows unscored, over a vocabulary of 1000 classes.
    let (classes, n) = (1000, 37);
    let (logits, target) = (matrix(n, classes, 59, 6.0), matrix(n, classes, 61, 6.0));
    let zmax = largest(&logits).max(largest(&target));
    let input = 4.0 * U * zmax + gamma(classes + 16);
    let flags: Vec<u32> = (0..n as u32).map(|r| u32::from(r % 5 != 3)).collect();
    let (ht, dt) = both(&target);
    let (mut hl, mut dl) = both(&logits);
    let hk = host.kl_rows(&ht, &mut hl, Some(&host.upload_indices(&flags).expect("flags"))).expect("host kl");
    let dk = d.kl_rows(&dt, &mut dl, Some(&d.upload_indices(&flags).expect("flags"))).expect("device kl");
    let softmax = |z: ndarray::ArrayView1<'_, f64>| {
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e = z.mapv(|v| (v - m).exp());
        let total = e.sum();
        e / total
    };
    for r in 0..n {
        let (p, q) = (softmax(target.row(r)), softmax(logits.row(r)));
        // Every term is within `input` of `p_c (|ln p_c| + |ln q_c| + 2 z_max + 1)`.
        let size: f64 = p.iter().zip(&q).map(|(a, b)| a * (a.ln().abs() + b.ln().abs() + 2.0 * zmax + 1.0)).sum();
        assert!((hk[r] - dk[r]).abs() <= input * size, "kl row {r}: {} against {} (band {:e})", dk[r], hk[r], input * size);
        if flags[r] == 0 {
            assert_eq!(dk[r], 0.0);
        }
    }
    let (hg, dg) = (down(&host, &hl), down(&d, &dl));
    let pq: Vec<(ndarray::Array1<f64>, ndarray::Array1<f64>)> = (0..n).map(|r| (softmax(target.row(r)), softmax(logits.row(r)))).collect();
    assert_within("kl cotangent", &dg, &hg, |i, j| input * (pq[i].0[j] + pq[i].1[j]));
    // Fisher probes of the same f32 probabilities (the host's softmax rows), with the same signs:
    // `√q` within 4 ulps (`8u`), `√q · ξ` summed in f32 (within `(8u + γ_n) Σ √q`), and the
    // product and the difference rounded: each entry within `16u (√q + q Σ √q) + γ_{n+8} q Σ √q`.
    let q = single(&Array2::from_shape_fn((n, classes), |(r, c)| pq[r].1[c]));
    let roots: Vec<f64> = q.rows().into_iter().map(|row| row.iter().map(|p| p.sqrt()).sum()).collect();
    let (mut hp, mut dp) = both(&q);
    host.fisher_probe_cotangent(&mut hp, (67, 3), Some(&host.upload_indices(&flags).expect("flags"))).expect("host probe");
    d.fisher_probe_cotangent(&mut dp, (67, 3), Some(&d.upload_indices(&flags).expect("flags"))).expect("device probe");
    assert_within("fisher probe", &down(&d, &dp), &down(&host, &hp), |i, j| {
        16.0 * U * (q[[i, j]].sqrt() + q[[i, j]] * roots[i]) + gamma(classes + 8) * q[[i, j]] * roots[i]
    });
    // The Fisher's quadratic per row: two passes of `n` terms of `q (|t| + |mean|)²`.
    let tangent = matrix(n, classes, 71, 1.0);
    let (htan, dtan) = both(&tangent);
    let (hl, dl) = both(&logits);
    let hq = host.softmax_quadratic(&hl, &htan).expect("host quadratic");
    let dq = d.softmax_quadratic(&dl, &dtan).expect("device quadratic");
    for r in 0..n {
        let q = &pq[r].1;
        let mean: f64 = q.iter().zip(tangent.row(r)).map(|(a, b)| a * b.abs()).sum();
        let size: f64 = q.iter().zip(tangent.row(r)).map(|(a, b)| a * (b.abs() + mean).powi(2)).sum();
        let band = 2.0 * (4.0 * U * zmax + gamma(2 * classes + 16)) * size;
        assert!((hq[r] - dq[r]).abs() <= band, "quadratic row {r}: {} against {} (band {band:e})", dq[r], hq[r]);
    }
}

#[test]
fn metal_grouped_products_match_the_host() {
    let Some(d) = metal() else { return };
    // Small integers: every product and sum is exact in f32.
    let blocks = d.column_blocks(&[2, 1]).expect("blocks");
    let left = up(&d, &ndarray::array![[2.0, -2.0, 3.0], [1.0, 2.0, -1.0]]);
    let right = up(&d, &ndarray::array![[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let sums = d.block_products(&left, &right, &blocks).expect("reduce");
    assert_eq!(down(&d, &sums), ndarray::array![[0.0, 6.0], [11.0, -5.0]]);
}

#[test]
fn metal_adam_sets_and_box_charge_match_the_host() {
    let Some(d) = metal() else { return };
    let host = Device::host();
    let (rows, cols) = (13, 9);
    let (w, m, v, g) = (matrix(rows, cols, 73, 1.0), matrix(rows, cols, 79, 0.1), matrix(rows, cols, 83, 0.01).mapv(f64::abs), matrix(rows, cols, 89, 0.3));
    let both = |x: &Array2<f64>| (up(&host, x), up(&d, x));
    let ((mut hw, mut dw), (mut hm, mut dm), (mut hv, mut dv), (hg, dg)) = (both(&w), both(&m), both(&v), both(&g));
    let (rate, betas) = (1e-2, (0.9, 0.999, 1e-8));
    host.adam(&mut hw, (&mut hm, &mut hv), &hg, rate, betas, 3).expect("host adam");
    d.adam(&mut dw, (&mut dm, &mut dv), &dg, rate, betas, 3).expect("device adam");
    let (c1, c2) = (1.0 - betas.0.powi(3), 1.0 - betas.1.powi(3));
    // A moment: four roundings of `β |m| + (1 − β) |g|`, and β's own rounding (`u β`), which
    // moves it by `u β |m − g|` (through both β and the f32 `1 − β`).
    let first = |i: usize, j: usize| gamma(4) * (betas.0 * m[[i, j]].abs() + (1.0 - betas.0) * g[[i, j]].abs()) + 2.0 * U * betas.0 * (m[[i, j]].abs() + g[[i, j]].abs());
    let second = |i: usize, j: usize| gamma(5) * (betas.1 * v[[i, j]] + (1.0 - betas.1) * g[[i, j]].powi(2)) + 2.0 * U * betas.1 * (v[[i, j]] + g[[i, j]].powi(2));
    assert_within("adam first moment", &down(&d, &dm), &down(&host, &hm), first);
    assert_within("adam second moment", &down(&d, &dv), &down(&host, &hv), second);
    let (hm, hv) = (down(&host, &hm), down(&host, &hv));
    // The weights: eight roundings of `|w| + step`, the first moment's error through the step, and
    // the second's halved through the square root (relative to `v`).
    assert_within("adam weights", &down(&d, &dw), &down(&host, &hw), |i, j| {
        let denominator = (hv[[i, j]] / c2).sqrt() + betas.2;
        let step = rate * (hm[[i, j]].abs() / c1) / denominator;
        gamma(8) * (w[[i, j]].abs() + step) + rate * (first(i, j) / c1) / denominator + step * (second(i, j) / (2.0 * hv[[i, j]]) + gamma(8))
    });
    // Sets on dyadic inputs (sizes, ratios, sums and codes all exact in f32): the host's sets.
    let dyadic = |rows: usize, cols: usize, seed: u64, levels: f64, unit: f64| matrix(rows, cols, seed, 1.0).mapv(|v| (v * levels).round() * unit);
    let reads = dyadic(rows, cols, 97, 8.0, 0.25);
    let metric = dyadic(rows, cols, 109, 16.0, 0.125).mapv(f64::abs);
    let bits = dyadic(1, cols, 101, 2.0, 1.0).mapv(|e| 2f64.powf(e));
    let left = dyadic(rows, 1, 103, 4.0, 0.25).mapv(f64::abs);
    let weight = dyadic(rows, 1, 107, 2.0, 1.0).mapv(|e| 2f64.powf(e - 1.0));
    let mask = Array2::from_shape_fn((rows, cols), |(r, c)| f64::from(u8::from((r + c) % 3 == 0)));
    let (mut hmask, mut dmask) = both(&mask);
    host.select_sets((&up(&host, &reads), &up(&host, &metric)), &up(&host, &bits), &up(&host, &left), &up(&host, &weight), &mut hmask).expect("host sets");
    d.select_sets((&up(&d, &reads), &up(&d, &metric)), &up(&d, &bits), &up(&d, &left), &up(&d, &weight), &mut dmask).expect("device sets");
    assert_eq!(down(&d, &dmask), down(&host, &hmask));
    // The box charge: `N` a sum of `cols` nonnegative terms of rounded inputs (γ_{cols+4} N), the
    // cotangent's and coefficient's few more roundings of their own terms.
    let (z, q, c0) = (matrix(rows, cols, 113, 2.0), matrix(rows, cols, 127, 1.0).mapv(f64::abs), matrix(rows, cols, 131, 1.0));
    let off = mask.mapv(|m| 1.0 - m);
    let ((hz, dz), (hm, dm), (hq, dq)) = (both(&z), both(&mask), both(&q));
    let ((mut hc, mut dc), (mut hk, mut dk)) = (both(&c0), both(&Array2::zeros((rows, cols))));
    let hn = host.box_charge(&hz, &hm, &hq, &mut hc, &mut hk).expect("host charge");
    let dn = d.box_charge(&dz, &dm, &dq, &mut dc, &mut dk).expect("device charge");
    let n: Vec<f64> = (0..rows).map(|r| (0..cols).map(|c| off[[r, c]] * z[[r, c]].abs() * q[[r, c]]).sum()).collect();
    for r in 0..rows {
        assert!((hn[r] - dn[r]).abs() <= 2.0 * gamma(cols + 4) * n[r] * n[r], "charge of row {r}: {} against {}", dn[r], hn[r]);
    }
    assert_within("charge cotangent", &down(&d, &dc), &down(&host, &hc), |r, c| gamma(cols + 8) * (c0[[r, c]].abs() + n[r] * q[[r, c]]));
    let hk = down(&host, &hk);
    assert_within("charge coefficient", &down(&d, &dk), &hk, |r, c| gamma(cols + 8) * hk[[r, c]].abs());
}

#[test]
fn metal_head_log_partition_matches_the_host() {
    let Some(d) = metal() else { return };
    let host = Device::host();
    // Logits `h·e` of 16 f32 terms are within `γ_16 Σ |h||e|` (`δ` below); a log partition moves
    // by at most `δ` and an expected head row `Σ q e` by `2δ max |e|` (the softmax's shift bound),
    // plus its own sum's `γ_classes` and the product's `γ_classes` rounding.
    let (rows, width, classes) = (7, 16, 300);
    let (hidden, head) = (single(&matrix(rows, width, 11, 1.0)), single(&matrix(classes, width, 12, 1.0)));
    let flags: Vec<u32> = (0..rows as u32).map(|r| u32::from(r != 3)).collect();
    let delta = gamma(width) * width as f64;
    for transposed in [false, true] {
        let stored = if transposed { head.t().to_owned() } else { head.clone() };
        let (mut he, mut de) = (host.zeros(rows, width).expect("zeros"), d.zeros(rows, width).expect("zeros"));
        let (hf, df) = (host.upload_indices(&flags).expect("flags"), d.upload_indices(&flags).expect("flags"));
        let hp = host.head_log_partition(&up(&host, &hidden), &up(&host, &stored), transposed, Some(&hf), Some(&mut he), Arithmetic::F32).expect("host");
        let dp = d.head_log_partition(&up(&d, &hidden), &up(&d, &stored), transposed, Some(&df), Some(&mut de), Arithmetic::F32).expect("metal");
        for (r, (a, b)) in dp.iter().zip(&hp).enumerate() {
            assert!((a - b).abs() <= delta + gamma(classes) * b.abs().max(1.0), "row {r}: {a} against {b}");
        }
        let band = 2.0 * delta * largest(&head) + 2.0 * gamma(classes) * largest(&head);
        assert_within("expected head rows", &down(&d, &de), &down(&host, &he), |_, _| band);
    }
}

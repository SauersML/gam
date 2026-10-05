//! The tensor operations in CUDA f32 storage (`Device::with_storage(Storage::F32)`) against their
//! float64 host twins on the same f32-representable inputs, each within a band in units of the f32
//! unit roundoff `u = 2⁻²⁴` (`γ_n = nu/(1 − nu)`); the host's own float64 error is below `2⁻²⁹` of
//! every band and is absorbed in it. And the float64 device, untouched: operands in mixed storage
//! are refused, conversions are exact both ways on f32 values.
//!
//! The device rounds each operation once (no contraction); a reduction of `n` terms summed in
//! float is within `γ_n` of the magnitude it sums, in any order. `expf`, `tanhf`, `sqrtf` and
//! `normcdff` are within 2 ulps (`4u` relative). A row reduction that leaves its row (a log
//! partition, a KL, an entropy) sums in double float terms each within `5u` of its value (its
//! exp, and its argument's rounding), rescaled in double: a log partition is within `6u`
//! (absolute), a probability within `11u` (relative), and a KL `Σ p (d − D)` (`d` the logit gap,
//! `D` the log partitions' difference) within `u (16 + 16 Σ p (|d − D| + |d|))`. Each test runs
//! when a CUDA device resolves (a runtime probe) and has nothing to run otherwise.

use gam_gpu::GpuPolicy;
use gam_gpu::gpu_error::GpuError;
use gam_gpu::tensor::{Arithmetic, Device, Op, PointwiseLaw, Storage, Tensor};
use ndarray::Array2;

const U: f64 = 1.0 / 16_777_216.0;

fn gamma(n: usize) -> f64 {
    n as f64 * U / (1.0 - n as f64 * U)
}

/// The CUDA device in f32 storage, and its float64 twin on the same stream.
fn cuda() -> Option<(Device, Device)> {
    let wide = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault")?;
    let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    Some((narrow, wide))
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

fn softmax(row: ndarray::ArrayView1<'_, f64>) -> Vec<f64> {
    let m = row.iter().fold(f64::NEG_INFINITY, |a, v| a.max(*v));
    let e: Vec<f64> = row.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

#[test]
fn f32_storage_moves_values_exactly_and_refuses_mixed_and_float64_work() {
    let Some((d, wide)) = cuda() else { return };
    assert!(!d.float64() && wide.float64() && d.storage() == Storage::F32);
    assert!(Device::host().with_storage(Storage::F32).is_err());
    let x = matrix(7, 5, 3, 2.0);
    let t = up(&d, &x);
    assert_eq!(t.storage(), Storage::F32);
    assert_eq!(t.bytes(), 7 * 5 * 4);
    assert_eq!(down(&d, &t), x);
    assert_eq!(down(&d, &d.copy(&t).expect("copy")), x);
    assert_eq!(down(&d, &d.rows_of(&t, 2, 3).expect("rows")), x.slice(ndarray::s![2..5, ..]).to_owned());
    assert_eq!(down(&d, &d.columns_of(&t, 1..4).expect("columns")), x.slice(ndarray::s![.., 1..4]).to_owned());
    let mut target = d.zeros(7, 5).expect("zeros");
    d.set_rows(&mut target, 0, &t).expect("set rows");
    d.set_columns(&mut target, 2, &d.columns_of(&t, 0..2).expect("columns")).expect("set columns");
    let mut expected = x.clone();
    for r in 0..7 {
        expected[[r, 2]] = x[[r, 0]];
        expected[[r, 3]] = x[[r, 1]];
    }
    assert_eq!(down(&d, &target), expected);
    let ids: Vec<u32> = vec![6, 0, 3, 3];
    let gathered = down(&d, &d.gather_rows(&t, &d.upload_indices(&ids).expect("ids")).expect("gather"));
    for (r, id) in ids.iter().enumerate() {
        assert_eq!(gathered.row(r), x.row(*id as usize));
    }
    let mut filled = d.copy(&t).expect("copy");
    d.fill_entries(&mut filled, &d.upload_indices(&[0, 7, 34]).expect("positions"), 2.5).expect("fill");
    let filled = down(&d, &filled);
    assert_eq!((filled[[0, 0]], filled[[1, 2]], filled[[6, 4]]), (2.5, 2.5, 2.5));
    // Conversions on the device: exact on f32 values both ways; narrowing rounds to nearest.
    let w = up(&wide, &x);
    let narrowed = d.convert(&w).expect("narrow");
    assert_eq!(narrowed.storage(), Storage::F32);
    assert_eq!(down(&d, &narrowed), x);
    let widened = wide.convert(&narrowed).expect("widen");
    assert_eq!(widened.storage(), Storage::F64);
    assert_eq!(down(&wide, &widened), x);
    let fine = Array2::from_elem((1, 3), 1.0 + f64::EPSILON * 3.0);
    assert_eq!(down(&d, &d.convert(&up(&wide, &fine)).expect("round")), Array2::from_elem((1, 3), 1.0));
    // Mixed storage is refused, never converted; so are float64 products and float64-only work.
    let mut sum = d.copy(&t).expect("copy");
    assert!(d.axpy(&mut sum, 1.0, &w).is_err());
    assert!(wide.axpy(&mut up(&wide, &x), 1.0, &t).is_err());
    let (a, b) = (up(&d, &matrix(3, 4, 1, 1.0)), up(&d, &matrix(4, 2, 2, 1.0)));
    let mut c = d.zeros(3, 2).expect("zeros");
    let refused = d.gemm(&mut c, 1.0, &a, Op::N, &b, Op::N, 0.0, Arithmetic::F64);
    assert!(matches!(refused, Err(GpuError::NoDeviceKernel { .. })), "{refused:?}");
    assert!(d.kl_proposal_rows(&t, &t).is_err());
    assert!(d.scaled_row_l2(&t, 0..5, 1.0).is_err());
}

/// Products straight on the f32 operands: within `(k + 2)` roundings in the operands' precision
/// plus `(k + 2)` f32 roundings of the accumulation, of `|A||B|`; strided-batched alike.
#[test]
fn f32_products_lie_inside_their_bands() {
    let Some((d, _)) = cuda() else { return };
    let host = Device::host();
    for (m, n, k) in [(96, 70, 300), (5, 3, 1), (33, 64, 1024)] {
        let (a, b) = (matrix(m, k, 11, 1.0), matrix(k, n, 13, 1.0));
        let magnitude = a.mapv(f64::abs).dot(&b.mapv(f64::abs));
        let reference = a.dot(&b);
        for arithmetic in [Arithmetic::F32, Arithmetic::Tf32] {
            let relative = 2.0 * (k + 2) as f64 * arithmetic.unit_roundoff() + (k + 2) as f64 * f64::from(f32::EPSILON);
            for (ta, tb) in [(Op::N, Op::N), (Op::T, Op::N), (Op::N, Op::T), (Op::T, Op::T)] {
                let left = if ta == Op::T { a.t().to_owned() } else { a.clone() };
                let right = if tb == Op::T { b.t().to_owned() } else { b.clone() };
                let mut c = up(&d, &Array2::from_elem((m, n), f64::NAN));
                d.gemm(&mut c, 1.0, &up(&d, &left), ta, &up(&d, &right), tb, 0.0, arithmetic).expect("product");
                assert_within(&format!("{arithmetic:?} {ta:?}{tb:?} product"), &down(&d, &c), &reference, |i, j| relative * magnitude[[i, j]]);
                d.gemm(&mut c, 0.5, &up(&d, &left), ta, &up(&d, &right), tb, 0.25, arithmetic).expect("accumulate");
                assert_within("accumulated product", &down(&d, &c), &(&reference * 0.75), |i, j| 2.0 * relative * magnitude[[i, j]]);
            }
        }
    }
    let (q, kk) = (matrix(4 * 16, 8, 17, 1.0), matrix(4 * 16, 8, 19, 1.0));
    let (mut hs, mut ds) = (host.zeros(64, 16).expect("zeros"), d.zeros(64, 16).expect("zeros"));
    host.gemm_batched(4, &mut hs, 0.5, &up(&host, &q), Op::N, &up(&host, &kk), Op::T, 0.0, Arithmetic::F64).expect("host batched");
    d.gemm_batched(4, &mut ds, 0.5, &up(&d, &q), Op::N, &up(&d, &kk), Op::T, 0.0, Arithmetic::F32).expect("device batched");
    assert_within("batched product", &down(&d, &ds), &down(&host, &hs), |_, _| gamma(10) * 0.5 * 8.0);
}

#[test]
fn f32_maps_norms_and_softmaxes_agree_with_the_host() {
    let Some((d, _)) = cuda() else { return };
    let host = Device::host();
    let both = |m: &Array2<f64>| (up(&host, m), up(&d, m));
    let (rows, cols) = (96, 70);
    let x = matrix(rows, cols, 29, 3.0);
    let g = matrix(rows, cols, 31, 1.0);
    let row = matrix(1, cols, 43, 1.0);
    let (hx, dx) = both(&x);
    let (hg, dg) = both(&g);
    let (hr, dr) = both(&row);
    // One rounding per elementwise expression (two for the accumulating forms).
    let mut hy = host.copy(&hg).expect("copy");
    let mut dy = d.copy(&dg).expect("copy");
    host.axpy(&mut hy, 0.25, &hx).expect("axpy");
    d.axpy(&mut dy, 0.25, &dx).expect("axpy");
    assert_within("axpy", &down(&d, &dy), &down(&host, &hy), |i, j| 2.0 * U * (g[[i, j]].abs() + 0.25 * x[[i, j]].abs()));
    let (mut hy, mut dy) = both(&g);
    host.add_row(&mut hy, -1.0, &hr).expect("add_row");
    d.add_row(&mut dy, -1.0, &dr).expect("add_row");
    assert_within("add_row", &down(&d, &dy), &down(&host, &hy), |i, j| U * (g[[i, j]].abs() + row[[0, j]].abs()));
    let (mut hy, mut dy) = both(&g);
    host.scale_columns(&mut hy, &hx, &hr, true).expect("scale");
    d.scale_columns(&mut dy, &dx, &dr, true).expect("scale");
    assert_within("scale_columns", &down(&d, &dy), &down(&host, &hy), |i, j| 2.0 * U * (g[[i, j]].abs() + (x[[i, j]] * row[[0, j]]).abs()));
    let (mut hy, mut dy) = both(&g);
    host.hadamard(&mut hy, &hx, &hg, true).expect("hadamard");
    d.hadamard(&mut dy, &dx, &dg, true).expect("hadamard");
    assert_within("hadamard", &down(&d, &dy), &down(&host, &hy), |i, j| 2.0 * U * (g[[i, j]].abs() + (x[[i, j]] * g[[i, j]]).abs()));
    // Norms: per row a float sum of `cols` squares, then a few roundings.
    let scale = x.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let epsilon = 1e-6;
    let pairs = [
        (host.rms_norm(&hx, epsilon), d.rms_norm(&dx, epsilon)),
        (host.rms_norm_backward(&hx, &hg, epsilon), d.rms_norm_backward(&dx, &dg, epsilon)),
        (host.rms_norm_tangent(&hx, &hg, epsilon), d.rms_norm_tangent(&dx, &dg, epsilon)),
    ];
    for (name, (h, dv)) in ["rms", "rms backward", "rms tangent"].into_iter().zip(pairs) {
        let reference = down(&host, &h.expect("host"));
        let magnitude = reference.iter().fold(scale, |m, v| m.max(v.abs()));
        assert_within(name, &down(&d, &dv.expect("device")), &reference, |_, _| gamma(cols + 16) * magnitude);
    }
    // Laws: each within a few ulps of its float evaluation, whose argument is exact.
    let laws = [PointwiseLaw::Relu, PointwiseLaw::Identity, PointwiseLaw::Zero, PointwiseLaw::Silu, PointwiseLaw::Gelu, PointwiseLaw::GeluTanh];
    let codes: Vec<u32> = (0..cols).map(|c| laws[c % laws.len()].code()).collect();
    let (hcodes, dcodes) = (host.upload_indices(&codes).expect("codes"), d.upload_indices(&codes).expect("codes"));
    let c = f64::from((std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2) as f32);
    let values = down(&host, &host.law_values(&hx, &hcodes, c).expect("host laws"));
    assert_within("laws", &down(&d, &d.law_values(&dx, &dcodes, c).expect("device laws")), &values, |i, j| 16.0 * U * (x[[i, j]].abs() + 1.0));
    let slopes = down(&host, &host.law_slopes(&hg, &hx, &hcodes, c).expect("host slopes"));
    assert_within("law slopes", &down(&d, &d.law_slopes(&dg, &dx, &dcodes, c).expect("device slopes")), &slopes, |i, j| {
        32.0 * U * g[[i, j]].abs() * (x[[i, j]].abs() + 1.0).powi(3)
    });
    // Rotations: two products and a sum per entry.
    let planes = 5;
    let angles = matrix(rows, planes, 37, 3.0);
    let (hcos, dcos) = both(&angles.mapv(|a| f64::from(a.cos() as f32)));
    let (hsin, dsin) = both(&angles.mapv(|a| f64::from(a.sin() as f32)));
    for (half, inverse) in [(true, false), (false, true)] {
        let h = down(&host, &host.rotate(&hx, &hcos, &hsin, half, inverse).expect("host rotate"));
        let r = down(&d, &d.rotate(&dx, &dcos, &dsin, half, inverse).expect("device rotate"));
        assert_within("rotate", &r, &h, |i, _| 3.0 * U * 2.0 * x.row(i).iter().fold(0.0_f64, |m, v| m.max(v.abs())));
    }
    // Softmaxes of attention scores, causal and not, and the backward map.
    let scores = matrix(64, 16, 23, 4.0);
    for causal in [false, true] {
        let (mut hs, mut ds) = both(&scores);
        host.softmax_rows(&mut hs, causal).expect("host softmax");
        d.softmax_rows(&mut ds, causal).expect("device softmax");
        assert_within("softmax", &down(&d, &ds), &down(&host, &hs), |_, _| gamma(16 + 8));
        let dalpha = matrix(64, 16, 41, 1.0);
        let (hd, dd) = both(&dalpha);
        let (hb, db) = (host.softmax_backward(&hs, &hd).expect("host"), d.softmax_backward(&ds, &dd).expect("device"));
        assert_within("softmax backward", &down(&d, &db), &down(&host, &hb), |_, _| 4.0 * gamma(16 + 8));
    }
    // Grouped products, the box charge and divided sums.
    let blocks = [3usize, 1, 7, 59];
    let (hb, db) = (host.column_blocks(&blocks).expect("blocks"), d.column_blocks(&blocks).expect("blocks"));
    let hp = down(&host, &host.block_products(&hx, &hg, &hb).expect("host blocks"));
    let dp = down(&d, &d.block_products(&dx, &dg, &db).expect("device blocks"));
    let offsets: Vec<usize> = std::iter::once(0).chain(blocks.iter().scan(0, |s, w| { *s += w; Some(*s) })).collect();
    assert_within("block products", &dp, &hp, |i, k| {
        gamma(blocks[k] + 1) * (offsets[k]..offsets[k + 1]).map(|c| (x[[i, c]] * g[[i, c]]).abs()).sum::<f64>()
    });
    let mask = matrix(rows, cols, 47, 1.0).mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
    let q = matrix(rows, cols, 53, 1.0).mapv(f64::abs);
    let (hm, dm) = both(&mask);
    let (hq, dq) = both(&q);
    let (mut hc, mut dc) = both(&g);
    let (mut hk, mut dk) = (host.zeros(rows, cols).expect("zeros"), d.zeros(rows, cols).expect("zeros"));
    let hn = host.box_charge(&hx, &hm, &hq, &mut hc, &mut hk).expect("host box");
    let dn = d.box_charge(&dx, &dm, &dq, &mut dc, &mut dk).expect("device box");
    for r in 0..rows {
        let norm = (0..cols).map(|c| (1.0 - mask[[r, c]]) * x[[r, c]].abs() * q[[r, c]]).sum::<f64>();
        assert!((hn[r] - dn[r]).abs() <= 8.0 * U * norm * norm + 1e-30, "box charge row {r}: {} against {}", dn[r], hn[r]);
    }
    assert_within("box cotangent", &down(&d, &dc), &down(&host, &hc), |i, j| 8.0 * U * (g[[i, j]].abs() + hn[i].sqrt() * 2.0 * q[[i, j]]));
    let argmax = d.argmax_rows(&dx).expect("argmax");
    assert_eq!(argmax, host.argmax_rows(&hx).expect("argmax"));
}

/// The vocabulary reductions: KL rows and their cotangent, softmax statistics, sampled cotangents,
/// the Fisher quadratic, and the swept head log partition, some rows unscored.
#[test]
fn f32_vocabulary_reductions_agree_with_the_host() {
    let Some((d, _)) = cuda() else { return };
    let host = Device::host();
    let both = |m: &Array2<f64>| (up(&host, m), up(&d, m));
    let (classes, n) = (5000, 37);
    let logits = matrix(n, classes, 47, 6.0);
    let target = matrix(n, classes, 53, 6.0).mapv(|v| f64::from((v + 40.0) as f32));
    let near = (&logits + &matrix(n, classes, 57, 0.01)).mapv(|v| f64::from(v as f32));
    let flags: Vec<u32> = (0..n as u32).map(|r| u32::from(r % 5 != 3)).collect();
    let (hf, df) = (host.upload_indices(&flags).expect("flags"), d.upload_indices(&flags).expect("flags"));
    for (name, teacher) in [("far", &target), ("near", &near)] {
        let (ht, dt) = both(teacher);
        let (mut hl, mut dl) = both(&logits);
        let hk = host.kl_rows(&ht, &mut hl, Some(&hf)).expect("host kl");
        let dk = d.kl_rows(&dt, &mut dl, Some(&df)).expect("device kl");
        let mut scratch = up(&d, &logits);
        assert_eq!(d.kl_score_rows(&dt, &mut scratch, Some(&df)).expect("score"), dk, "the score skips only the cotangent");
        assert_eq!(down(&d, &scratch), logits, "a score leaves its logits");
        for r in 0..n {
            if flags[r] == 0 {
                assert_eq!(dk[r], 0.0);
                continue;
            }
            let (p, qv) = (softmax(teacher.row(r)), softmax(logits.row(r)));
            let gap = (0..classes).map(|c| teacher[[r, c]] - logits[[r, c]]).collect::<Vec<_>>();
            let big_d = (0..classes).map(|c| p[c] * (gap[c] - (p[c].ln() - qv[c].ln()))).sum::<f64>();
            let spread: f64 = (0..classes).map(|c| p[c] * ((gap[c] - big_d).abs() + gap[c].abs())).sum();
            let band = U * (16.0 + 16.0 * spread);
            assert!((hk[r] - dk[r]).abs() <= band, "{name} kl row {r}: {} against {} (band {band:e})", dk[r], hk[r]);
        }
        let hg = down(&host, &hl);
        assert_within(&format!("{name} kl cotangent"), &down(&d, &dl), &hg, |i, j| {
            if flags[i] == 0 { 0.0 } else { 16.0 * U * (softmax(teacher.row(i))[j] + softmax(logits.row(i))[j]) + 1e-38 }
        });
    }
    // Softmax statistics: the log partition within 8u, Σ p log p within 16u (1 + Σ p |log p|).
    let (mut hs, mut ds) = both(&logits);
    let hstats = host.softmax_stats_rows(&mut hs, Some(&hf)).expect("host stats");
    let dstats = d.softmax_stats_rows(&mut ds, Some(&df)).expect("device stats");
    for r in 0..n {
        let entropy_scale = 1.0 + softmax(logits.row(r)).iter().map(|p| if *p > 0.0 { -p * p.ln() } else { 0.0 }).sum::<f64>();
        assert!((hstats[r][0] - dstats[r][0]).abs() <= 8.0 * U, "log partition row {r}: {:?} against {:?}", dstats[r], hstats[r]);
        assert!((hstats[r][1] - dstats[r][1]).abs() <= 16.0 * U * entropy_scale, "negative entropy row {r}");
    }
    let probabilities = down(&host, &hs);
    assert_within("softmax probabilities", &down(&d, &ds), &probabilities, |i, j| 16.0 * U * probabilities[[i, j]] + 1e-38);
    // Sampled cotangents: the same label away from ties at the uniform, q within 16u q.
    let uniforms = matrix(n, 1, 59, 0.5).mapv(|v| v + 0.5);
    let (hu, du) = both(&uniforms);
    let (mut hs, mut ds) = both(&logits);
    host.sampled_cotangent(&mut hs, &hu, None).expect("host sampled");
    d.sampled_cotangent(&mut ds, &du, None).expect("device sampled");
    let (hs, ds) = (down(&host, &hs), down(&d, &ds));
    for r in 0..n {
        let label = |m: &Array2<f64>| m.row(r).iter().position(|v| *v < -0.5).expect("a label");
        assert_eq!(label(&hs), label(&ds), "row {r}'s label");
    }
    assert_within("sampled cotangent", &ds, &hs, |i, j| 16.0 * U * (hs[[i, j]].abs() + if hs[[i, j]] < -0.5 { 1.0 } else { 0.0 }) + 1e-38);
    let tangent = matrix(n, classes, 61, 1.0);
    let (hl, dl) = both(&logits);
    let (htan, dtan) = both(&tangent);
    let hq = host.softmax_quadratic(&hl, &htan).expect("host quadratic");
    let dq = d.softmax_quadratic(&dl, &dtan).expect("device quadratic");
    for r in 0..n {
        assert!((hq[r] - dq[r]).abs() <= 32.0 * U, "quadratic row {r}: {} against {}", dq[r], hq[r]);
    }
}

/// The swept head log partition and its expected head row against the host's materialized ones:
/// each logit is an f32 product within `(width + 2)` roundings of `Σ |h||e|` (`δ` at most), so a log
/// partition is within `δ + 8u` and an expected row within `(2δ + (classes + 16) u) max |e|`.
#[test]
fn the_swept_head_log_partition_matches_the_materialized_one() {
    let Some((d, wide)) = cuda() else { return };
    let host = Device::host();
    for (rows, classes, width) in [(37, 3001, 64), (5, 70_000, 16), (1, 1, 8)] {
        let hidden = matrix(rows, width, 71, 1.0);
        let embedding = matrix(classes, width, 73, 1.0);
        let flags: Vec<u32> = (0..rows as u32).map(|r| u32::from(r % 4 != 2)).collect();
        let delta = (width + 2) as f64 * U * hidden.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * width as f64;
        let biggest = embedding.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for transposed in [false, true] {
            let head = if transposed { embedding.t().to_owned() } else { embedding.clone() };
            let hf = host.upload_indices(&flags).expect("flags");
            let mut hm = host.zeros(rows, width).expect("zeros");
            let hz = host.head_log_partition(&up(&host, &hidden), &up(&host, &head), transposed, Some(&hf), Some(&mut hm), Arithmetic::F64).expect("host");
            for device in [&d, &wide] {
                let df = device.upload_indices(&flags).expect("flags");
                let mut dm = up(device, &Array2::from_elem((rows, width), f64::NAN));
                let arithmetic = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
                let dz = device
                    .head_log_partition(&up(device, &hidden), &up(device, &head), transposed, Some(&df), Some(&mut dm), arithmetic)
                    .expect("device");
                // Float64 is held to the same bands in its own unit roundoff (2⁻⁵³ for u).
                let unit = if device.float64() { f64::EPSILON / 2.0 / U } else { 1.0 };
                for r in 0..rows {
                    if flags[r] == 0 {
                        assert_eq!(dz[r], 0.0);
                        continue;
                    }
                    let band = unit * (delta + 8.0 * U);
                    assert!((hz[r] - dz[r]).abs() <= band, "{} log partition row {r}: {} against {} (band {band:e})", device.name(), dz[r], hz[r]);
                }
                assert_within(&format!("{} expected head rows", device.name()), &down(device, &dm), &down(&host, &hm), |i, _| {
                    if flags[i] == 0 { 0.0 } else { unit * (2.0 * delta + (classes + 16) as f64 * U) * biggest }
                });
                let plain = device.head_log_partition(&up(device, &hidden), &up(device, &head), transposed, None, None, arithmetic).expect("no gradient");
                for r in 0..rows {
                    if flags[r] != 0 {
                        assert_eq!(plain[r], dz[r], "{} row {r}: the gradient does not move the partition", device.name());
                    }
                }
            }
        }
    }
}

#[test]
fn f32_adam_steps_agree_with_the_host() {
    let Some((d, _)) = cuda() else { return };
    let host = Device::host();
    let both = |m: &Array2<f64>| (up(&host, m), up(&d, m));
    let w = matrix(12, 9, 81, 1.0);
    let (mut hw, mut dw) = both(&w);
    let (mut hm, mut dm) = (host.zeros(12, 9).expect("zeros"), d.zeros(12, 9).expect("zeros"));
    let (mut hv, mut dv) = (host.zeros(12, 9).expect("zeros"), d.zeros(12, 9).expect("zeros"));
    for step in 1..=3 {
        let g = matrix(12, 9, 83 + step, 1.0);
        let (hg, dg) = both(&g);
        host.adam(&mut hw, (&mut hm, &mut hv), &hg, 1e-2, (0.9, 0.999, 1e-8), step).expect("host adam");
        d.adam(&mut dw, (&mut dm, &mut dv), &dg, 1e-2, (0.9, 0.999, 1e-8), step).expect("device adam");
    }
    // Each step moves a weight by at most the rate (Adam's normalized step), in a few roundings.
    assert_within("adam weights", &down(&d, &dw), &down(&host, &hw), |i, j| 3.0 * (16.0 * U * 1e-2 + U * (w[[i, j]].abs() + 0.03)));
    assert_within("adam first moment", &down(&d, &dm), &down(&host, &hm), |_, _| 16.0 * U);
}

/// A captured step (a product, a law through a temporary, an accumulation, an in-place softmax)
/// replays bitwise what running it directly does, in either storage; a host transfer inside a
/// capture is refused, records nothing, and the device works on after.
#[test]
fn a_captured_step_replays_what_running_it_does() {
    let Some((d, wide)) = cuda() else { return };
    for device in [&d, &wide] {
        let arithmetic = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
        let x_host = matrix(64, 32, 91, 1.0);
        let (x, w) = (up(device, &x_host), up(device, &matrix(32, 48, 92, 0.2)));
        let codes = device.upload_indices(&vec![PointwiseLaw::GeluTanh.code(); 48]).expect("codes");
        let c = 0.797_884_560_802_865_4;
        let step = |y: &mut Tensor, total: &mut Tensor, scores: &mut Tensor| -> Result<(), GpuError> {
            device.gemm(y, 1.0, &x, Op::N, &w, Op::N, 0.0, arithmetic)?;
            let activated = device.law_values(y, &codes, c)?;
            device.axpy(total, 0.5, &activated)?;
            device.softmax_rows(scores, false)?;
            device.hadamard(scores, total, total, true)
        };
        let fresh = || (device.zeros(64, 48).expect("y"), device.zeros(64, 48).expect("total"), up(device, &matrix(64, 48, 93, 3.0)));
        let (mut y, mut total, mut scores) = fresh();
        for _ in 0..3 {
            step(&mut y, &mut total, &mut scores).expect("direct step");
        }
        let (mut gy, mut gtotal, mut gscores) = fresh();
        device.synchronize().expect("idle");
        device.begin_capture().expect("capture");
        step(&mut gy, &mut gtotal, &mut gscores).expect("recorded step");
        let graph = device.end_capture().expect("graph");
        assert_eq!(down(device, &gtotal), Array2::<f64>::zeros((64, 48)), "{}: recording runs nothing", device.name());
        for _ in 0..3 {
            graph.launch().expect("replay");
        }
        assert_eq!(down(device, &gtotal), down(device, &total), "{}: replayed accumulation", device.name());
        assert_eq!(down(device, &gscores), down(device, &scores), "{}: replayed in-place maps", device.name());
        device.begin_capture().expect("capture");
        assert!(device.argmax_rows(&y).is_err(), "{}: a read-back inside a capture is refused", device.name());
        assert!(device.upload(x_host.view()).is_err(), "{}: so is an upload", device.name());
        assert!(device.begin_capture().is_err(), "{}: captures do not nest", device.name());
        device.end_capture().expect("the capture records on");
        assert!(device.end_capture().is_err(), "{}: one capture ends once", device.name());
        device.synchronize().expect("the stream works on");
        step(&mut y, &mut total, &mut scores).expect("a direct step after the capture");
        assert_eq!(device.argmax_rows(&y).expect("a read-back after it").len(), 64);
    }
    // The vocabulary reductions' device-resident forms: recorded in f32, replayed into float64
    // columns, equal to the downloading forms.
    let (rows, classes, width) = (24, 9000, 32);
    let (hidden, head) = (up(&d, &matrix(rows, width, 95, 1.0)), up(&d, &matrix(classes, width, 96, 1.0)));
    let (target, logits) = (up(&d, &matrix(rows, classes, 97, 5.0)), matrix(rows, classes, 98, 5.0));
    let mut direct_logits = up(&d, &logits);
    let kl = d.kl_rows(&target, &mut direct_logits, None).expect("kl");
    let mut direct_mean = d.zeros(rows, width).expect("zeros");
    let partitions = d.head_log_partition(&hidden, &head, false, None, Some(&mut direct_mean), Arithmetic::F32).expect("partition");
    let mut recorded_logits = up(&d, &logits);
    let (mut kl_column, mut partition_column) = (wide.zeros(rows, 1).expect("column"), wide.zeros(rows, 1).expect("column"));
    let mut mean = d.zeros(rows, width).expect("zeros");
    d.synchronize().expect("idle");
    d.begin_capture().expect("capture");
    d.kl_rows_into(&target, &mut recorded_logits, None, true, &mut kl_column).expect("recorded kl");
    d.head_log_partition_into(&hidden, &head, false, None, Some(&mut mean), &mut partition_column, Arithmetic::F32).expect("recorded partition");
    let graph = d.end_capture().expect("graph");
    graph.launch().expect("replay");
    assert_eq!(down(&wide, &kl_column).column(0).to_vec(), kl, "replayed KL rows");
    assert_eq!(down(&d, &recorded_logits), down(&d, &direct_logits), "replayed KL cotangent");
    assert_eq!(down(&wide, &partition_column).column(0).to_vec(), partitions, "replayed log partitions");
    assert_eq!(down(&d, &mean), down(&d, &direct_mean), "replayed expected head rows");
}

/// `x` rounded to bfloat16 (nearest, ties to even).
fn bf16(x: f64) -> f64 {
    let bits = (x as f32).to_bits();
    f64::from(f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16))
}

/// Products in bfloat16 (f32 operands rounded per call, or a bfloat16 copy read as it is) against
/// the host's products of the same rounded operands: only the f32 accumulation differs, within
/// `(k + 2)` f32 roundings of `|A||B|`. The swept head log partition on a bfloat16 head: its logits
/// alike (`δ`), its expected rows also rounding each chunk's weights and head rows (`2⁻⁷ max |e|`).
#[test]
fn bfloat16_products_match_the_host_on_rounded_operands() {
    let Some((d, wide)) = cuda() else { return };
    let host = Device::host();
    let (m, n, k) = (96, 70, 300);
    let (a, b) = (matrix(m, k, 21, 1.0), matrix(k, n, 22, 1.0));
    let mut hc = host.zeros(m, n).expect("zeros");
    host.gemm(&mut hc, 1.0, &up(&host, &a), Op::N, &up(&host, &b), Op::N, 0.0, Arithmetic::Bf16).expect("host");
    let reference = down(&host, &hc);
    let magnitude = a.mapv(|v| bf16(v).abs()).dot(&b.mapv(|v| bf16(v).abs()));
    let band = |i: usize, j: usize| (k + 2) as f64 * f64::from(f32::EPSILON) * magnitude[[i, j]];
    let (da, db) = (up(&d, &a), up(&d, &b));
    let mut c = d.zeros(m, n).expect("zeros");
    d.gemm(&mut c, 1.0, &da, Op::N, &db, Op::N, 0.0, Arithmetic::Bf16).expect("rounded per call");
    assert_within("bfloat16 product", &down(&d, &c), &reference, band);
    let frozen = d.bf16_copy(&up(&d, &b.t().to_owned())).expect("bfloat16 copy");
    assert_eq!((frozen.storage(), frozen.bytes()), (Storage::Bf16, k * n * 2));
    assert_eq!(down(&d, &frozen), b.t().mapv(bf16), "a bfloat16 copy rounds to nearest");
    assert_eq!(down(&d, &wide.bf16_copy(&up(&wide, &b)).expect("from float64")), b.mapv(bf16));
    let mut c = d.zeros(m, n).expect("zeros");
    d.gemm(&mut c, 1.0, &da, Op::N, &frozen, Op::T, 0.0, Arithmetic::Bf16).expect("frozen operand");
    assert_within("bfloat16 product on a frozen copy", &down(&d, &c), &reference, band);
    assert!(d.gemm(&mut c, 1.0, &da, Op::N, &frozen, Op::T, 0.0, Arithmetic::F32).is_err(), "a bfloat16 operand takes Bf16");
    assert!(d.axpy(&mut c, 1.0, &frozen).is_err(), "only products take bfloat16 copies");
    assert!(d.with_storage(Storage::Bf16).is_err(), "no device makes bfloat16 tensors");
    let (rows, classes, width) = (37, 20_011, 64);
    let hidden = matrix(rows, width, 23, 1.0);
    let embedding = matrix(classes, width, 24, 1.0);
    let mut hm = host.zeros(rows, width).expect("zeros");
    let hz = host.head_log_partition(&up(&host, &hidden), &up(&host, &embedding), false, None, Some(&mut hm), Arithmetic::Bf16).expect("host");
    let head = d.bf16_copy(&up(&d, &embedding)).expect("frozen head");
    let mut dm = d.zeros(rows, width).expect("zeros");
    let dz = d.head_log_partition(&up(&d, &hidden), &head, false, None, Some(&mut dm), Arithmetic::Bf16).expect("bfloat16 sweep");
    let delta = (width + 2) as f64 * f64::from(f32::EPSILON) * width as f64;
    for r in 0..rows {
        assert!((hz[r] - dz[r]).abs() <= delta + 8.0 * U, "bfloat16 log partition row {r}: {} against {}", dz[r], hz[r]);
    }
    assert_within("bfloat16 expected head rows", &down(&d, &dm), &down(&host, &hm), |_, _| 2.0 * delta + 2f64.powi(-7) + (classes + 16) as f64 * U);
}

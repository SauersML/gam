//! The factorized Gaussian posterior's device operations (`Device::reparameterize`,
//! `posterior_ivon`, `group_moments`, `group_divergence`): the host against the formulas entry by
//! entry, and every accelerator that resolves (CUDA in f32 storage with float64 group sums, the
//! Apple GPU in f32) against the host on the same inputs.
//!
//! An accelerator's value is a chain of at most 24 roundings, each within 4 ulps of f32 (`exp`,
//! `log`, `sqrt`, `cos` in the safe math modes; the rest exact to half an ulp), of terms no larger
//! than the largest magnitude entering it, so it is within `96 u` of that magnitude (`u = 2⁻²⁴`).
//! A group sum of `n` such terms adds `γ_n` of the summed magnitudes.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{BF16_TERM_RESOLUTION, Device, Indices, PosteriorStep, Storage, Tensor, TermLayout, posterior_normal};
use ndarray::Array2;

const U: f64 = 1.0 / 16_777_216.0;
const CHAIN: f64 = 96.0 * U;

fn matrix(rows: usize, cols: usize, seed: u64, scale: f64, shift: f64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        f64::from((shift + (2.0 * unit - 1.0) * scale) as f32)
    })
}

/// The test posterior: 6 × 40 entries in 9 groups (rows 0–2 by row, the rest by column pairs),
/// one group removed (`s = −∞`, `μ = 0`), the `(key, stream)` of its sample, its gradient and
/// Gauss–Newton factor, and its step.
struct Case {
    mean: Array2<f64>,
    log_sd: Array2<f64>,
    moments: [Array2<f64>; 2],
    sample: (u64, u64),
    gradient: Array2<f64>,
    factor: Array2<f64>,
    groups: Vec<u32>,
    count: usize,
    step: PosteriorStep,
}

fn case() -> Case {
    let (rows, cols) = (6, 40);
    let groups: Vec<u32> = (0..rows * cols).map(|i| if i / cols < 3 { (i / cols) as u32 } else { 3 + ((i % cols) / 8) as u32 }).collect();
    let mut mean = matrix(rows, cols, 1, 0.5, 0.0);
    let mut log_sd = matrix(rows, cols, 2, 0.5, -3.0);
    for (i, g) in groups.iter().enumerate() {
        if *g == 4 {
            mean[(i / cols, i % cols)] = 0.0;
            log_sd[(i / cols, i % cols)] = f64::NEG_INFINITY;
        }
    }
    // The gradient's momentum, and a positive curvature estimate.
    let moments = [matrix(rows, cols, 3, 0.1, 0.0), matrix(rows, cols, 4, 0.5, 1.0)];
    let step = PosteriorStep { gradient_scale: 1.5, factor_scale: 0.25, tokens: 50.0, rate: 0.1, beta1: 0.9, beta2: 0.999, step: 7 };
    let (gradient, factor) = (matrix(rows, cols, 7, 3.0, 0.0), matrix(rows, cols, 8, 2.0, 0.0));
    Case { mean, log_sd, moments, sample: (0x1234_5678_9abc_def0, 42), gradient, factor, groups, count: 8, step }
}

/// The formulas, entry by entry: the sample, the stepped posterior and moments, and the group sums
/// `(n, Σ μ² + σ², Σ 2s)` before and after the step, from the sums before.
fn reference(c: &Case) -> (Array2<f64>, Array2<f64>, Array2<f64>, [Array2<f64>; 2], Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut before = vec![[0.0; 3]; c.count];
    for (i, g) in c.groups.iter().enumerate() {
        let (mu, s) = (c.mean.as_slice().unwrap()[i], c.log_sd.as_slice().unwrap()[i]);
        if s > f64::NEG_INFINITY {
            let b = &mut before[*g as usize];
            b[0] += 1.0;
            b[1] += mu * mu + (2.0 * s).exp();
            b[2] += 2.0 * s;
        }
    }
    let variance: Vec<f64> = before.iter().map(|b| if b[0] > 0.0 { b[1] / b[0] } else { 0.0 }).collect();
    let (b1, b2, n) = (c.step.beta1, c.step.beta2, c.step.tokens);
    let c1 = 1.0 - b1.powi(c.step.step as i32);
    let mut theta = c.mean.clone();
    let (mut mean, mut log_sd, mut moments) = (c.mean.clone(), c.log_sd.clone(), c.moments.clone());
    let mut after = vec![[0.0; 3]; c.count];
    for (i, g) in c.groups.iter().enumerate() {
        let at = (i / c.mean.ncols(), i % c.mean.ncols());
        let e = f64::from(posterior_normal(c.sample.0, c.sample.1, i as u64));
        let (mu, s) = (c.mean[at], c.log_sd[at]);
        theta[at] = mu + s.exp() * e;
        if s == f64::NEG_INFINITY {
            continue;
        }
        let (sd, gr) = (s.exp(), c.step.gradient_scale * c.gradient[at]);
        let delta = 1.0 / (n * variance[*g as usize]);
        let momentum = b1 * moments[0][at] + (1.0 - b1) * gr;
        let estimate = c.step.factor_scale * c.factor[at] * c.factor[at];
        let (h, d) = (moments[1][at], estimate - moments[1][at]);
        let curvature = h + (1.0 - b2) * d + 0.5 * (1.0 - b2) * (1.0 - b2) * d * d / (h + delta);
        moments[0][at] = momentum;
        moments[1][at] = curvature;
        mean[at] = mu - (c.step.rate * (momentum / c1 + delta * mu) / (curvature + delta)).clamp(-sd, sd);
        log_sd[at] = -0.5 * (n * (curvature + delta)).ln();
        let a = &mut after[*g as usize];
        a[0] += 1.0;
        a[1] += mean[at] * mean[at] + (2.0 * log_sd[at]).exp();
        a[2] += 2.0 * log_sd[at];
    }
    (theta, mean, log_sd, moments, before, after)
}

/// The device's results on `c`: the sample, the stepped posterior and moments, the group sums
/// before and after the step, and the divergences from the sums before. `fit` holds the posterior,
/// its sample and gradient, `wide` the group sums.
fn run(fit: &Device, wide: &Device, c: &Case) -> (Array2<f64>, Array2<f64>, Array2<f64>, Vec<Array2<f64>>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let up = |d: &Device, m: &Array2<f64>| d.upload(m.view()).unwrap();
    let (mut mean, mut log_sd) = (up(fit, &c.mean), up(fit, &c.log_sd));
    let mut moments: Vec<Tensor> = c.moments.iter().map(|m| up(fit, m)).collect();
    let groups: Indices = fit.upload_indices(&c.groups).unwrap();
    let mut theta = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
    fit.reparameterize(&mut theta, (&mean, &log_sd), c.sample).unwrap();
    let mut sums = wide.zeros(c.count, 3).unwrap();
    fit.group_moments((&mean, &log_sd), &groups, &mut sums).unwrap();
    let before = wide.download(&sums).unwrap();
    let (mut variance, mut divergence) = (wide.zeros(c.count, 1).unwrap(), wide.zeros(c.count, 1).unwrap());
    wide.group_divergence(&mut sums, &mut variance, &mut divergence).unwrap();
    assert!(wide.download(&sums).unwrap().iter().all(|v| *v == 0.0), "the sums are zeroed");
    let (gradient, factor) = (up(fit, &c.gradient), up(fit, &c.factor));
    let [momentum, curvature] = &mut moments[..] else { unreachable!() };
    fit.posterior_ivon((&mut mean, &mut log_sd), [momentum, curvature], (&gradient, &factor), (&groups, &variance), &mut sums, &c.step).unwrap();
    let down = |t: &Tensor| fit.download(t).unwrap();
    let down_wide = |t: &Tensor| wide.download(t).unwrap();
    (fit.download(&theta).unwrap(), down(&mean), down(&log_sd), moments.iter().map(down).collect(), before, down_wide(&sums), down_wide(&divergence))
}

fn sums_of(rows: &[[f64; 3]]) -> Array2<f64> {
    Array2::from_shape_fn((rows.len(), 3), |(g, k)| rows[g][k])
}

/// `a` within `relative` of `b`'s largest magnitude (an entry `−∞` in both counts as equal).
fn close(what: &str, a: &Array2<f64>, b: &Array2<f64>, relative: f64) {
    let scale = b.iter().filter(|v| v.is_finite()).fold(0.0_f64, |m, v| m.max(v.abs()));
    for ((at, x), y) in a.indexed_iter().zip(b.iter()) {
        let equal = x == y || (x - y).abs() <= relative * scale;
        assert!(equal, "{what} {at:?}: {x} against {y} (band {:e})", relative * scale);
    }
}

#[test]
fn draws_are_standard_normal() {
    let n = 200_000u64;
    let draws: Vec<f64> = (0..n).map(|i| f64::from(posterior_normal(3, 9, i))).collect();
    let mean = draws.iter().sum::<f64>() / n as f64;
    let second = draws.iter().map(|x| x * x).sum::<f64>() / n as f64;
    // Six standard errors: the mean's is 1/√n, the second moment's √2/√n.
    let se = 1.0 / (n as f64).sqrt();
    assert!(mean.abs() < 6.0 * se, "mean {mean}");
    assert!((second - 1.0).abs() < 6.0 * 2f64.sqrt() * se, "second moment {second}");
    // Another stream, key or index draws anew.
    assert_ne!(posterior_normal(3, 9, 0), posterior_normal(3, 10, 0));
    assert_ne!(posterior_normal(3, 9, 0), posterior_normal(4, 9, 0));
    assert_ne!(posterior_normal(3, 9, 0), posterior_normal(3, 9, 1 << 32));
}

#[test]
fn the_host_steps_the_posterior_by_its_formulas() {
    let c = case();
    let host = Device::host();
    let (theta, mean, log_sd, moments, before, after, divergence) = run(&host, &host, &c);
    let (t, m, s, mo, b, a) = reference(&c);
    let exact = 1e-15;
    close("sample", &theta, &t, exact);
    close("mean", &mean, &m, exact);
    close("log sd", &log_sd, &s, exact);
    for (k, (x, y)) in moments.iter().zip(&mo).enumerate() {
        close(&format!("moment {k}"), x, y, exact);
    }
    close("sums before", &before, &sums_of(&b), exact);
    close("sums after", &after, &sums_of(&a), exact);
    for (g, row) in b.iter().enumerate() {
        let expected = if row[0] > 0.0 { 0.5 * (row[0] * (row[1] / row[0]).ln() - row[2]) } else { 0.0 };
        assert!((divergence[(g, 0)] - expected).abs() <= 1e-12 * expected.abs().max(1.0), "divergence {g}");
    }
    assert_eq!(b[4][0], 0.0, "the removed group is empty");
    assert!(theta.indexed_iter().all(|((r, col), v)| c.groups[r * 40 + col] != 4 || *v == 0.0), "a removed entry samples zero");
}

fn against_host(fit: &Device, wide: &Device) {
    let c = case();
    let host = Device::host();
    let (theta, mean, log_sd, moments, before, after, divergence) = run(fit, wide, &c);
    let (t, m, s, mo, b, a, d) = run(&host, &host, &c);
    // A group sums at most 40 entries; the step reads its variance (such a sum over its count),
    // and a second moment squares a gradient that carries the variance's error.
    let sum_band = CHAIN + 40.0 * U / (1.0 - 40.0 * U);
    close("sample", &theta, &t, CHAIN);
    close("mean", &mean, &m, 2.0 * sum_band);
    close("log sd", &log_sd, &s, 2.0 * sum_band);
    for (k, (x, y)) in moments.iter().zip(&mo).enumerate() {
        close(&format!("moment {k}"), x, y, 2.0 * sum_band);
    }
    close("sums before", &before, &b, sum_band);
    close("sums after", &after, &a, sum_band);
    // ½ (n ln v − Σ 2s) cancels terms as large as `n |ln v|` and `|Σ 2s|`.
    let cancelled = b.rows().into_iter().map(|r| r[0] * (r[1] / r[0].max(1.0)).ln().abs() + r[2].abs()).fold(0.0_f64, f64::max);
    for g in 0..c.count {
        assert!((divergence[(g, 0)] - d[(g, 0)]).abs() <= sum_band * cancelled, "divergence {g}: {} against {}", divergence[(g, 0)], d[(g, 0)]);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn the_apple_gpu_matches_the_host() {
    let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    against_host(&metal, &metal);
}

#[test]
fn cuda_matches_the_host() {
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    against_host(&narrow, &wide);
    against_host(&wide, &wide);
}

#[test]
fn a_bfloat16_sample_is_the_f32_sample_rounded() {
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    let fit = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    let c = case();
    let (mean, log_sd) = (fit.upload(c.mean.view()).unwrap(), fit.upload(c.log_sd.view()).unwrap());
    let mut single = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
    fit.reparameterize(&mut single, (&mean, &log_sd), c.sample).unwrap();
    let mut half = fit.bf16_copy(&single).unwrap();
    fit.reparameterize(&mut half, (&mean, &log_sd), c.sample).unwrap();
    // `bf16_copy` rounds to nearest, ties to even, as the bfloat16 sample does.
    let expected = fit.download(&fit.bf16_copy(&single).unwrap()).unwrap();
    assert_eq!(fit.download(&half).unwrap(), expected);
}

/// The bfloat16 nearest `x` (ties to even), as a float64.
fn bf16(x: f64) -> f64 {
    let bits = (x as f32).to_bits();
    f64::from(f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16))
}

/// On CUDA, f32 masters with a bfloat16 momentum (and a bfloat16 gradient, and a bfloat16 sample):
/// the stored momentum is the f32 step's value rounded once, and the masters, the f32 curvature and
/// the group sums are the f32 step's.
#[test]
fn cuda_bfloat16_momentum_is_the_f32_step_rounded() {
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    let fit = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    let half = wide.with_storage(Storage::Bf16).expect("CUDA holds bfloat16");
    let mut c = case();
    // Inputs a bfloat16 holds exactly, so both steps start from the same values.
    for m in &mut c.moments {
        m.mapv_inplace(bf16);
    }
    c.gradient.mapv_inplace(bf16);
    c.factor.mapv_inplace(bf16);
    let (theta, mean, log_sd, moments, _, after, _) = run(&fit, &wide, &c);
    let up = |d: &Device, m: &Array2<f64>| d.upload(m.view()).expect("upload");
    let down = |t: &Tensor| fit.download(t).expect("download");
    for gradient_storage in [&fit, &half] {
        let (mut m, mut s) = (up(&fit, &c.mean), up(&fit, &c.log_sd));
        let (mut momentum, mut curvature) = (up(&half, &c.moments[0]), up(&fit, &c.moments[1]));
        let groups = fit.upload_indices(&c.groups).expect("groups");
        let mut sample = half.zeros(c.mean.nrows(), c.mean.ncols()).expect("sample");
        fit.reparameterize(&mut sample, (&m, &s), c.sample).expect("bfloat16 sample");
        let mut sums = wide.zeros(c.count, 3).expect("sums");
        fit.group_moments((&m, &s), &groups, &mut sums).expect("group moments");
        let (mut variance, mut divergence) = (wide.zeros(c.count, 1).expect("variance"), wide.zeros(c.count, 1).expect("divergence"));
        wide.group_divergence(&mut sums, &mut variance, &mut divergence).expect("group divergence");
        let (gradient, factor) = (up(gradient_storage, &c.gradient), up(gradient_storage, &c.factor));
        fit.posterior_ivon((&mut m, &mut s), [&mut momentum, &mut curvature], (&gradient, &factor), (&groups, &variance), &mut sums, &c.step).expect("bfloat16 step");
        close("bfloat16 sample", &down(&sample), &theta.mapv(bf16), 2f64.powi(-8));
        close("mean", &down(&m), &mean, CHAIN);
        close("log sd", &down(&s), &log_sd, CHAIN);
        close("bfloat16 momentum", &down(&momentum), &moments[0].mapv(bf16), 2f64.powi(-8));
        close("curvature", &down(&curvature), &moments[1], CHAIN);
        close("sums after", &wide.download(&sums).expect("sums"), &after, CHAIN);
        // A bfloat16 copy widens back to the values it holds.
        assert_eq!(down(&fit.convert(&momentum).expect("widen")), down(&momentum));
    }
}

/// A posterior whose rows need 1, 2 and 3 bfloat16 terms (row r has `σ = 2^-(8r+2) |μ|`, so
/// `u_{r+1} |μ| ≤ σ < u_r |μ|` with `u = 2^-8, 2^-16, 2^-24`), and a removed row (`μ = 0`,
/// `s = −∞`). The means are bfloat16 values.
fn term_case() -> (Array2<f64>, Array2<f64>) {
    let mean = matrix(4, 24, 11, 1.0, 0.0).mapv(|m| bf16(if m.abs() < 0.05 { 0.5 } else { m }));
    let mut log_sd = Array2::zeros(mean.dim());
    for ((r, c), s) in log_sd.indexed_iter_mut() {
        *s = if r == 3 { f64::NEG_INFINITY } else { (mean[(r, c)].abs() * 2f64.powi(-(8 * r as i32 + 2))).ln() };
    }
    let mean = Array2::from_shape_fn(mean.dim(), |(r, c)| if r == 3 { 0.0 } else { mean[(r, c)] });
    (mean, log_sd)
}

/// The sum of a sample's terms, per entry, from `Separate` tensors.
fn term_sum(d: &Device, terms: &[Tensor]) -> Array2<f64> {
    terms.iter().map(|t| d.download(t).unwrap()).fold(None, |acc: Option<Array2<f64>>, t| Some(acc.map_or(t.clone(), |a| a + t))).unwrap()
}

/// Checks a device's terms: their count, each term a bfloat16 value, the stacked layout the
/// separate one side by side, and the sum within `u_K |θ|` of the f32 sample `theta`.
fn check_terms(d: &Device, (m, s): (&Array2<f64>, &Array2<f64>), theta: &Array2<f64>, rows_need: &[usize]) {
    for (r, need) in rows_need.iter().enumerate() {
        let row = |a: &Array2<f64>| d.upload(a.slice(ndarray::s![r..r + 1, ..])).unwrap();
        assert_eq!(d.term_count((&row(m), &row(s))).unwrap(), *need, "row {r}");
    }
    let (mean, log_sd) = (&d.upload(m.view()).unwrap(), &d.upload(s.view()).unwrap());
    assert_eq!(d.term_count((mean, log_sd)).unwrap(), 3);
    for count in 1..=3 {
        let separate = d.sample_terms((mean, log_sd), (7, 3), count, TermLayout::Separate).unwrap();
        let stacked = d.sample_terms((mean, log_sd), (7, 3), count, TermLayout::Stacked).unwrap();
        assert_eq!((separate.len(), stacked.len()), (count, 1));
        let side = d.download(&stacked[0]).unwrap();
        let cols = theta.ncols();
        for (k, term) in separate.iter().enumerate() {
            let values = d.download(term).unwrap();
            assert!(values.iter().all(|v| bf16(*v) == *v), "term {k} holds bfloat16 values");
            assert_eq!(side.slice(ndarray::s![.., k * cols..(k + 1) * cols]), values, "stacked term {k}");
        }
        let sum = term_sum(d, &separate);
        let u = BF16_TERM_RESOLUTION[count - 1];
        for ((at, a), b) in sum.indexed_iter().zip(theta.iter()) {
            assert!((a - b).abs() <= u * b.abs(), "{count} terms at {at:?}: {a} against {b}");
        }
    }
}

#[test]
fn the_host_writes_a_sample_as_bfloat16_terms_that_keep_its_noise() {
    let host = Device::host();
    let (m, s) = term_case();
    let theta = Array2::from_shape_fn(m.dim(), |(r, c)| {
        let i = (r * m.ncols() + c) as u64;
        f64::from((m[(r, c)] + s[(r, c)].exp() * f64::from(posterior_normal(7, 3, i))) as f32)
    });
    check_terms(&host, (&m, &s), &theta, &[1, 2, 3, 1]);
    // One bfloat16 loses the noise of a row that needs three: noise below half a bfloat16 step of
    // a bfloat16 mean rounds back to the mean, while three terms keep it.
    let (mean, log_sd) = (host.upload(m.view()).unwrap(), host.upload(s.view()).unwrap());
    let one = host.download(&host.sample_terms((&mean, &log_sd), (7, 3), 1, TermLayout::Separate).unwrap()[0]).unwrap();
    assert!(one.row(2).iter().zip(m.row(2)).all(|(t, mu)| t == mu), "the noise vanishes in one bfloat16");
    let three = term_sum(&host, &host.sample_terms((&mean, &log_sd), (7, 3), 3, TermLayout::Separate).unwrap());
    // (An f32 sample keeps noise above half an f32 step, 2^-24 |μ|: all but a draw |ε| < 2^-6.)
    let kept = three.row(2).iter().zip(m.row(2)).filter(|(t, mu)| t != mu).count();
    assert!(kept >= 20, "three terms keep the noise of {kept} of 24 entries");
}

#[test]
fn cuda_bfloat16_terms_match_its_f32_sample() {
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    let fit = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    let (m, s) = term_case();
    let (mean, log_sd) = (fit.upload(m.view()).unwrap(), fit.upload(s.view()).unwrap());
    let mut single = fit.zeros(m.nrows(), m.ncols()).unwrap();
    fit.reparameterize(&mut single, (&mean, &log_sd), (7, 3)).unwrap();
    check_terms(&fit, (&m, &s), &fit.download(&single).unwrap(), &[1, 2, 3, 1]);
}

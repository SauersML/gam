//! The factorized Gaussian posterior's device operations (`Device::reparameterize`, its block form,
//! `posterior_ivon`, `group_moments`, `group_divergence`): the host against the formulas entry by
//! entry, and every accelerator that resolves (CUDA in f32 storage with float64 group sums, the
//! Apple GPU in f32) against the host on the same inputs, with the groups held per row, per column
//! and per entry (`GroupMap`).
//!
//! An accelerator's value is a chain of at most 24 roundings, each within 4 ulps of f32 (`exp`,
//! `log`, `sqrt`, `cos` in the safe math modes; the rest exact to half an ulp), of terms no larger
//! than the largest magnitude entering it, so it is within `96 u` of that magnitude (`u = 2⁻²⁴`).
//! A group sum of `n` such terms adds `γ_n` of the summed magnitudes.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{ANTITHETIC, Device, GroupAxis, PosteriorStep, Storage, Tensor, posterior_normal};
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

/// A test posterior in 8 groups laid out by `axis`: 6 × 40 entries by row (rows 0–5), 130 × 16 by
/// column pairs (more rows than one column thread sums, `GroupMap`), or 6 × 40 by entry (rows 0–2 by
/// row, the rest by blocks of 8 columns); group 4 removed (`s = −∞`, `μ = 0`). With the `(key,
/// stream)` of its sample, IVON's state (the momentum and a positive curvature), its gradient,
/// Gauss–Newton factor and prior curvature of either sign, and its step (the seventh of a constant
/// `β₁`).
struct Case {
    mean: Array2<f64>,
    log_sd: Array2<f64>,
    moments: [Array2<f64>; 2],
    sample: (u64, u64),
    gradient: Array2<f64>,
    factor: Array2<f64>,
    prior: Array2<f64>,
    groups: Vec<u32>,
    count: usize,
    step: PosteriorStep,
}

const AXES: [GroupAxis; 3] = [GroupAxis::Rows, GroupAxis::Columns, GroupAxis::Entries];

fn case(axis: GroupAxis) -> Case {
    let (rows, cols) = if axis == GroupAxis::Columns { (130, 16) } else { (6, 40) };
    let group = |r: usize, c: usize| match axis {
        GroupAxis::Rows => r,
        GroupAxis::Columns => c / 2,
        GroupAxis::Entries if r < 3 => r,
        GroupAxis::Entries => 3 + c / 8,
    };
    let groups: Vec<u32> = (0..rows * cols).map(|i| group(i / cols, i % cols) as u32).collect();
    let mut mean = matrix(rows, cols, 1, 0.5, 0.0);
    let mut log_sd = matrix(rows, cols, 2, 0.5, -3.0);
    for (i, g) in groups.iter().enumerate() {
        if *g == 4 {
            mean[(i / cols, i % cols)] = 0.0;
            log_sd[(i / cols, i % cols)] = f64::NEG_INFINITY;
        }
    }
    let moments = [matrix(rows, cols, 3, 0.1, 0.0), matrix(rows, cols, 4, 0.5, 1.0)];
    let step = PosteriorStep { gradient_scale: 1.5, factor_scale: 0.25, tokens: 50.0, beta1: 0.9, beta2: 0.999, weight: PosteriorStep::constant_weight(0.9, 6) };
    let (gradient, factor, prior) = (matrix(rows, cols, 7, 3.0, 0.0), matrix(rows, cols, 8, 2.0, 0.0), matrix(rows, cols, 5, 0.5, 0.0));
    Case { mean, log_sd, moments, sample: (0x1234_5678_9abc_def0, 42), gradient, factor, prior, groups, count: 8, step }
}

/// A step's results: the sample, the posterior moved by the step's full direction and IVON's state
/// (the momentum, the curvature), the group sums `(n, Σ μ² + σ², Σ 2s)` before and after the step,
/// the divergences from the sums before, and the step's terms (groups × 5,
/// `Device::posterior_ivon`).
#[derive(Clone, PartialEq)]
struct Stepped {
    theta: Array2<f64>,
    mean: Array2<f64>,
    log_sd: Array2<f64>,
    moments: Vec<Array2<f64>>,
    before: Array2<f64>,
    after: Array2<f64>,
    divergence: Array2<f64>,
    terms: Array2<f64>,
}

/// The formulas, entry by entry, from the sums before the step: with `d₀` the direction before the
/// step and `d` the step's, the terms `d · d`, `u · d`, `Σ h⁺ d²` (`h⁺` the curvature where `h + δ > 0`, else 0), `(g + δ μ) · d₀` and
/// `Σ (h₀⁺ + δ) d₀²` per group.
fn reference(c: &Case) -> Stepped {
    let mut before = Array2::zeros((c.count, 3));
    for (i, g) in c.groups.iter().enumerate() {
        let (mu, s) = (c.mean.as_slice().unwrap()[i], c.log_sd.as_slice().unwrap()[i]);
        if s > f64::NEG_INFINITY {
            let g = *g as usize;
            before[(g, 0)] += 1.0;
            before[(g, 1)] += mu * mu + (2.0 * s).exp();
            before[(g, 2)] += 2.0 * s;
        }
    }
    let variance: Vec<f64> = (0..c.count).map(|g| if before[(g, 0)] > 0.0 { before[(g, 1)] / before[(g, 0)] } else { 0.0 }).collect();
    let divergence = Array2::from_shape_fn((c.count, 1), |(g, _)| if before[(g, 0)] > 0.0 { 0.5 * (before[(g, 0)] * variance[g].ln() - before[(g, 2)]) } else { 0.0 });
    let (b1, b2, n) = (c.step.beta1, c.step.beta2, c.step.tokens);
    // The momentum's bias correction before the step and after it.
    let (w0, w1) = (c.step.weight, b1 * c.step.weight + (1.0 - b1));
    let mut theta = c.mean.clone();
    let (mut mean, mut log_sd, mut moments) = (c.mean.clone(), c.log_sd.clone(), c.moments.clone());
    let (mut after, mut terms) = (Array2::zeros((c.count, 3)), Array2::zeros((c.count, 5)));
    for (i, g) in c.groups.iter().enumerate() {
        let at = (i / c.mean.ncols(), i % c.mean.ncols());
        let e = f64::from(posterior_normal(c.sample.0, c.sample.1, i as u64));
        let (mu, s) = (c.mean[at], c.log_sd[at]);
        theta[at] = mu + s.exp() * e;
        if s == f64::NEG_INFINITY {
            continue;
        }
        let g = *g as usize;
        let delta = 1.0 / (n * variance[g]);
        let (m0, h0) = (moments[0][at], moments[1][at]);
        // The curvature where the total precision `h + δ` is positive, else none.
        let usable = |h: f64| if h + delta > 0.0 { h } else { 0.0 };
        let held = usable(h0) + delta;
        let previous = (m0 / w0 + delta * mu) / held;
        let (gr, u) = (c.step.gradient_scale * c.gradient[at], c.factor[at]);
        let estimate = c.step.factor_scale * u * u + c.prior[at];
        let momentum = b1 * m0 + (1.0 - b1) * gr;
        let curvature = h0 + (1.0 - b2) * (estimate - h0);
        let positive = usable(curvature);
        let d = (momentum / w1 + delta * mu) / held;
        moments[0][at] = momentum;
        moments[1][at] = curvature;
        mean[at] = mu - d;
        log_sd[at] = -0.5 * (n * (positive + delta)).ln();
        after[(g, 0)] += 1.0;
        after[(g, 1)] += mean[at] * mean[at] + (2.0 * log_sd[at]).exp();
        after[(g, 2)] += 2.0 * log_sd[at];
        terms[(g, 0)] += d * d;
        terms[(g, 1)] += u * d;
        terms[(g, 2)] += (usable(h0) * d) * d;
        terms[(g, 3)] += (gr + delta * mu) * previous;
        terms[(g, 4)] += (held * previous) * previous;
    }
    Stepped { theta, mean, log_sd, moments: moments.to_vec(), before, after, divergence, terms }
}

/// The device's results on `c`: the sample, the step and its full move of the mean (an average
/// from zero with weight 1, the stepped mean itself, whose moments are the sums after the step).
/// `fit` holds the posterior, its sample and gradient, `wide` the group sums.
fn run(fit: &Device, wide: &Device, c: &Case) -> Stepped {
    let up = |d: &Device, m: &Array2<f64>| d.upload(m.view()).unwrap();
    let (mut mean, mut log_sd) = (up(fit, &c.mean), up(fit, &c.log_sd));
    let (mut momentum, mut curvature) = (up(fit, &c.moments[0]), up(fit, &c.moments[1]));
    let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
    let mut theta = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
    fit.reparameterize(&mut theta, (&mean, &log_sd), c.sample).unwrap();
    let mut sums = wide.zeros(c.count, 3).unwrap();
    fit.group_moments((&mean, &log_sd), &groups, &mut sums).unwrap();
    let before = wide.download(&sums).unwrap();
    let (mut variance, mut divergence) = (wide.zeros(c.count, 1).unwrap(), wide.zeros(c.count, 2).unwrap());
    wide.group_divergence(&mut sums, None, &mut variance, &mut divergence).unwrap();
    assert!(wide.download(&sums).unwrap().iter().all(|v| *v == 0.0), "the sums are zeroed");
    let (gradient, factor, prior) = (up(fit, &c.gradient), up(fit, &c.factor), up(fit, &c.prior));
    let (mut direction, mut terms) = (fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap(), wide.zeros(c.count, 5).unwrap());
    fit.posterior_ivon((&mean, &mut log_sd), [&mut momentum, &mut curvature], (Some(&gradient), Some(&factor), Some(&prior)), (&groups, &variance), (&mut direction, &mut terms), &c.step).unwrap();
    let mut average = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
    fit.posterior_finish((&mut mean, &direction, 1.0), (&mut average, 1.0), &log_sd, &groups, &mut sums).unwrap();
    let down = |t: &Tensor| fit.download(t).unwrap();
    let down_wide = |t: &Tensor| wide.download(t).unwrap();
    Stepped {
        theta: down(&theta),
        mean: down(&mean),
        log_sd: down(&log_sd),
        moments: vec![down(&momentum), down(&curvature)],
        before,
        after: down_wide(&sums),
        divergence: down_wide(&divergence),
        terms: down_wide(&terms),
    }
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
fn an_antithetic_key_draws_the_negated_noise_on_every_backend() {
    let key = 0x1234_5678_9ABC_DEF0_u64;
    for i in [0, 1, 77, 1 << 33] {
        assert_eq!(posterior_normal(key ^ ANTITHETIC, 9, i), -posterior_normal(key, 9, i));
    }
    let (mean, log_sd) = (Array2::from_elem((6, 7), 0.5), Array2::from_shape_fn((6, 7), |(r, c)| -1.0 + 0.1 * (r + c) as f64));
    let mut devices = vec![Device::host()];
    devices.extend(Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault"));
    for d in devices {
        let up = |a: &Array2<f64>| d.upload(a.view()).unwrap();
        let (m, s) = (up(&mean), up(&log_sd));
        let (mut plus, mut minus) = (d.zeros(6, 7).unwrap(), d.zeros(6, 7).unwrap());
        d.reparameterize(&mut plus, (&m, &s), (key, 4)).unwrap();
        d.reparameterize(&mut minus, (&m, &s), (key ^ ANTITHETIC, 4)).unwrap();
        // The pair's samples sum to twice the mean.
        let (plus, minus) = (d.download(&plus).unwrap(), d.download(&minus).unwrap());
        for (p, q) in plus.iter().zip(&minus) {
            assert!((p + q - 1.0).abs() < 1e-6, "{}: {p} and {q}", d.name());
        }
    }
}

#[test]
fn group_ids_are_held_per_row_per_column_or_per_entry() {
    let host = Device::host();
    for axis in AXES {
        let c = case(axis);
        assert_eq!(host.group_map(&c.groups, c.mean.dim()).unwrap().axis(), axis);
    }
    assert!(host.group_map(&[0, 1, 2], (2, 2)).is_err(), "ids for every entry");
}

#[test]
fn the_host_steps_the_posterior_by_its_formulas() {
    for axis in AXES {
        host_formulas(&case(axis));
    }
}

/// The host's step against the formulas, the fresh and own slope sums among its terms.
fn host_formulas(c: &Case) {
    let host = Device::host();
    let found = run(&host, &host, c);
    let expected = reference(c);
    let exact = 1e-15;
    let pairs = [
        ("sample", &found.theta, &expected.theta),
        ("mean", &found.mean, &expected.mean),
        ("log sd", &found.log_sd, &expected.log_sd),
        ("momentum", &found.moments[0], &expected.moments[0]),
        ("curvature", &found.moments[1], &expected.moments[1]),
        ("sums before", &found.before, &expected.before),
        ("sums after", &found.after, &expected.after),
        ("step terms", &found.terms, &expected.terms),
    ];
    for (what, x, y) in pairs {
        close(what, x, y, exact);
    }
    for g in 0..c.count {
        let (x, y) = (found.divergence[(g, 0)], expected.divergence[(g, 0)]);
        assert!((x - y).abs() <= 1e-12 * y.abs().max(1.0), "divergence {g}: {x} against {y}");
    }
    assert_eq!(expected.before[(4, 0)], 0.0, "the removed group is empty");
    assert!(expected.terms.row(4).iter().all(|t| *t == 0.0), "the removed group has no terms");
    assert!(found.theta.iter().zip(&c.groups).all(|(v, g)| *g != 4 || *v == 0.0), "a removed entry samples zero");
}

fn against_host(fit: &Device, wide: &Device) {
    for axis in AXES {
        against_host_on(fit, wide, &case(axis));
    }
}

fn against_host_on(fit: &Device, wide: &Device, c: &Case) {
    let host = Device::host();
    let found = run(fit, wide, c);
    let expected = run(&host, &host, c);
    // No device sum is atomic (`GroupMap`): a second run is the first bit for bit.
    assert!(run(fit, wide, c) == found, "the device's results repeat bit for bit");
    // A group sums at most `n` entries; the step reads its variance (such a sum over its count).
    let n = (0..c.count as u32).map(|g| c.groups.iter().filter(|h| **h == g).count()).max().unwrap() as f64;
    let sum_band = CHAIN + n * U / (1.0 - n * U);
    close("sample", &found.theta, &expected.theta, CHAIN);
    close("mean", &found.mean, &expected.mean, 2.0 * sum_band);
    close("log sd", &found.log_sd, &expected.log_sd, 2.0 * sum_band);
    for (k, (x, y)) in found.moments.iter().zip(&expected.moments).enumerate() {
        close(&format!("moment {k}"), x, y, 2.0 * sum_band);
    }
    close("sums before", &found.before, &expected.before, sum_band);
    close("sums after", &found.after, &expected.after, sum_band);
    // A term sums at most `n` products of signed factors, each within `2 sum_band` of the largest
    // product, so it is within `2 n sum_band` of the largest term.
    close("step terms", &found.terms, &expected.terms, 2.0 * n * sum_band);
    // ½ (n ln v − Σ 2s) cancels terms as large as `n |ln v|` and `|Σ 2s|`.
    let b = &expected.before;
    let cancelled = b.rows().into_iter().map(|r| r[0] * (r[1] / r[0].max(1.0)).ln().abs() + r[2].abs()).fold(0.0_f64, f64::max);
    for g in 0..c.count {
        let (x, y) = (found.divergence[(g, 0)], expected.divergence[(g, 0)]);
        assert!((x - y).abs() <= sum_band * cancelled, "divergence {g}: {x} against {y}");
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
    let c = case(GroupAxis::Entries);
    let (mean, log_sd) = (fit.upload(c.mean.view()).unwrap(), fit.upload(c.log_sd.view()).unwrap());
    let mut single = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
    fit.reparameterize(&mut single, (&mean, &log_sd), c.sample).unwrap();
    let mut half = fit.bf16_copy(&single).unwrap();
    fit.reparameterize(&mut half, (&mean, &log_sd), c.sample).unwrap();
    // `bf16_copy` rounds to nearest, ties to even, as the bfloat16 sample does.
    let expected = fit.download(&fit.bf16_copy(&single).unwrap()).unwrap();
    assert_eq!(fit.download(&half).unwrap(), expected);
}

/// `R` coordinates in one group whose data term per token is `½ h (θ − a)²` (`a` the coordinate's
/// value in `M`, its start) with gradient noise of standard deviation `s` per step, the group's
/// prior variance `v` held fixed, at `N = 2^16` tokens: `F / N = E_q[ℓ] + KL(q ‖ p) / N` is
/// stationary in `μ` at `μ* = h a / (h + δ)`, `δ = 1 / (N v)`. With `h = δ / 4` and `a = −33`,
/// `μ* = −6.6`, and the gradient's noise equals the full gradient's size at the start
/// (`s = δ |a|`), so the data term's pull near `μ*` is within the momentum's noise. Each step
/// draws `θ = μ + σ ε` and `g = h (θ − a) + s z` (`ε`, `z` standard normal), and the Gauss–Newton
/// factor `√h`, so the curvature stays `h`. The kernel takes IVON's full direction; each step
/// moves a tenth of it, as a line step whose measurement the test does not model would. Returns
/// each coordinate's average `μ` over the last half of `steps` steps. `fit` holds the posterior,
/// `wide` the group's variance and sums.
fn settled(fit: &Device, wide: &Device, steps: u64) -> (f64, Vec<f64>) {
    const R: usize = 64;
    let (tokens, v) = (65_536.0, 1.0);
    let delta = 1.0 / (tokens * v);
    let (h, a) = (0.25 * delta, -33.0);
    let s = delta * 33.0;
    let target = h * a / (h + delta);
    let up = |m: Array2<f64>| fit.upload(m.view()).unwrap();
    let (mut mean, mut log_sd) = (up(Array2::from_elem((1, R), a)), up(Array2::from_elem((1, R), -0.5 * (tokens * (h + delta)).ln())));
    let (mut momentum, mut curvature) = (up(Array2::zeros((1, R))), up(Array2::from_elem((1, R), h)));
    let groups = fit.group_map(&[0; R], (1, R)).unwrap();
    let variance = wide.upload(Array2::from_elem((1, 1), v).view()).unwrap();
    let factor = up(Array2::from_elem((1, R), h.sqrt()));
    let mut averages = vec![0.0; R];
    for t in 1..=steps {
        let (mu, sd) = (fit.download(&mean).unwrap(), fit.download(&log_sd).unwrap().mapv(f64::exp));
        if t > steps / 2 {
            for (total, m) in averages.iter_mut().zip(&mu) {
                *total += m / (steps - steps / 2) as f64;
            }
        }
        let gradient = Array2::from_shape_fn((1, R), |(_, i)| {
            let theta = mu[(0, i)] + sd[(0, i)] * f64::from(posterior_normal(11, t, i as u64));
            h * (theta - a) + s * f64::from(posterior_normal(12, t, i as u64))
        });
        let step = PosteriorStep { gradient_scale: 1.0, factor_scale: 1.0, tokens, beta1: 0.9, beta2: 1.0 - 1.0 / 64.0, weight: PosteriorStep::constant_weight(0.9, t - 1) };
        let (mut direction, mut sums) = (fit.zeros(1, R).unwrap(), wide.zeros(1, 5).unwrap());
        let gradient = up(gradient);
        fit.posterior_ivon((&mean, &mut log_sd), [&mut momentum, &mut curvature], (Some(&gradient), Some(&factor), None), (&groups, &variance), (&mut direction, &mut sums), &step).unwrap();
        mean = up(&mu - &(fit.download(&direction).unwrap() * 0.1));
    }
    (target, averages)
}

/// IVON's step's fixed point is `F`'s stationary point: its direction is linear in the bias-corrected
/// momentum, whose expectation at `μ*` is zero. Averaged over the last 2000 of 4000 steps and over
/// 64 coordinates, `μ` is within six standard errors (of that average over the coordinates) of
/// `μ* = −6.6`, and the standard error is small enough to tell `μ*` from the `−4.1` at which
/// shrinking the data momentum alone by its measured noise, with `δ μ` added exactly, settled.
fn settles_at_the_stationary_point(fit: &Device, wide: &Device) {
    let (target, averages) = settled(fit, wide, 4000);
    let n = averages.len() as f64;
    let mean = averages.iter().sum::<f64>() / n;
    let error = (averages.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
    assert!((target + 6.6).abs() < 1e-12, "μ* {target}");
    assert!(error < 0.2, "standard error {error}");
    assert!((mean - target).abs() < 6.0 * error, "μ settled at {mean} ± {error}, against μ* = {target}");
}

#[test]
fn a_coordinate_within_its_gradient_noise_settles_at_the_stationary_point() {
    let host = Device::host();
    settles_at_the_stationary_point(&host, &host);
    #[cfg(target_os = "macos")]
    if let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault") {
        settles_at_the_stationary_point(&metal, &metal);
    }
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        settles_at_the_stationary_point(&wide.with_storage(Storage::F32).expect("CUDA holds f32"), &wide);
        settles_at_the_stationary_point(&wide, &wide);
    }
}

/// A softmax regression's step on `fit` (sums on `wide`): logits `z_t = W x_t` over `K` classes at
/// `n` fixed inputs, and per draw the Gauss–Newton factor `u = Σ_t (e_{y_t} − p_t) x_tᵀ`, each label
/// `y_t` drawn from `p_t = softmax(z_t)`, so `E[u uᵀ] = G = Σ_t J_tᵀ F_t J_t` (`F_t` the softmax's
/// Fisher matrix `diag(p_t) − p_t p_tᵀ`). The step's direction `d` is the same for every draw
/// (with `β₂ = ½` the draw carries half the stepped curvature, so a direction formed from it would
/// differ draw by draw). Returns over `draws` draws the mean of `c (u · d)²` (`c` the factor
/// scale, `u · d` the step's terms), its standard error, and the exact `c dᵀ G d`.
fn direction_is_independent_of_the_probe(fit: &Device, wide: &Device, draws: usize) -> (f64, f64, f64) {
    let (k, dim, n) = (5, 4, 12);
    let w = matrix(k, dim, 21, 0.8, 0.0);
    let x = matrix(n, dim, 22, 1.0, 0.0);
    let p: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            let z: Vec<f64> = (0..k).map(|c| (0..dim).map(|j| w[(c, j)] * x[(t, j)]).sum()).collect();
            let top = z.iter().copied().fold(f64::MIN, f64::max);
            let e: Vec<f64> = z.iter().map(|v| (v - top).exp()).collect();
            let total: f64 = e.iter().sum();
            e.iter().map(|v| v / total).collect()
        })
        .collect();
    let ids: Vec<u32> = (0..k * dim).map(|i| (i / dim) as u32).collect();
    let groups = fit.group_map(&ids, (k, dim)).unwrap();
    let variance = wide.upload_vec(k, 1, (0..k).map(|g| 0.01 * (g + 1) as f64).collect()).unwrap();
    let step = PosteriorStep { gradient_scale: 1.5, factor_scale: 1.0 / n as f64, tokens: 50.0, beta1: 0.9, beta2: 0.5, weight: 0.5 };
    let (log_sd, momentum, curvature, gradient) = (matrix(k, dim, 2, 0.5, -3.0), matrix(k, dim, 3, 0.1, 0.0), matrix(k, dim, 4, 0.5, 1.0), matrix(k, dim, 7, 3.0, 0.0));
    let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
    let (mean, gradient) = (up(&w), up(&gradient));
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut first, mut total, mut squares) = (None::<Array2<f64>>, 0.0, 0.0);
    for _ in 0..draws {
        let mut u = Array2::<f64>::zeros((k, dim));
        for (t, pt) in p.iter().enumerate() {
            let r = uniform();
            let y = (0..k).find(|c| r < pt[..=*c].iter().sum::<f64>()).unwrap_or(k - 1);
            for (c, pc) in pt.iter().enumerate() {
                let seed = f64::from(u8::from(c == y)) - pc;
                for j in 0..dim {
                    u[(c, j)] += seed * x[(t, j)];
                }
            }
        }
        let (mut s, mut m, mut h) = (up(&log_sd), up(&momentum), up(&curvature));
        let (mut direction, mut sums) = (fit.zeros(k, dim).unwrap(), wide.zeros(k, 5).unwrap());
        fit.posterior_ivon((&mean, &mut s), [&mut m, &mut h], (Some(&gradient), Some(&up(&u)), None), (&groups, &variance), (&mut direction, &mut sums), &step).unwrap();
        let d = fit.download(&direction).unwrap();
        match &first {
            None => first = Some(d),
            Some(f) => assert_eq!(&d, f, "{}: the direction moved with the probe", fit.name()),
        }
        let along: f64 = wide.download(&sums).unwrap().column(1).sum();
        let x = step.factor_scale * along * along;
        total += x;
        squares += x * x;
    }
    let d = first.unwrap();
    // `dᵀ G d = Σ_t a_tᵀ F_t a_t`, `a_t = d x_t` the logits' move along `d`.
    let exact: f64 = p
        .iter()
        .enumerate()
        .map(|(t, pt)| {
            let a: Vec<f64> = (0..k).map(|c| (0..dim).map(|j| d[(c, j)] * x[(t, j)]).sum()).collect();
            let first: f64 = pt.iter().zip(&a).map(|(pc, ac)| pc * ac).sum();
            pt.iter().zip(&a).map(|(pc, ac)| pc * ac * ac).sum::<f64>() - first * first
        })
        .sum::<f64>()
        * step.factor_scale;
    let m = total / draws as f64;
    (m, ((squares / draws as f64 - m * m) / (draws as f64 - 1.0)).sqrt(), exact)
}

/// On the host over 40,000 label draws the mean of `c (u · d)²` is within five standard errors of
/// `c dᵀ G d`, the standard error under 2% of it: the curvature along the step's direction that
/// the step length reads is unbiased. Every accelerator keeps the direction fixed across draws.
#[test]
fn a_steps_direction_does_not_depend_on_its_curvature_probe() {
    let host = Device::host();
    let (mean, error, exact) = direction_is_independent_of_the_probe(&host, &host, 40_000);
    assert!((mean - exact).abs() <= 5.0 * error && error < 0.02 * exact, "c (u · d)² averages {mean} against c dᵀ G d = {exact} (standard error {error})");
    #[cfg(target_os = "macos")]
    if let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault") {
        direction_is_independent_of_the_probe(&metal, &metal, 4);
    }
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        direction_is_independent_of_the_probe(&wide.with_storage(Storage::F32).expect("CUDA holds f32"), &wide, 4);
    }
}

/// `c`'s sample written by `fit` into a block at (2, 4) of a larger tensor on `out` (`out`'s
/// storage) holds the whole sample entry for entry and leaves the rest as it was.
fn sample_in_a_block(fit: &Device, out: &Device) {
    let c = case(GroupAxis::Entries);
    let (mean, log_sd) = (fit.upload(c.mean.view()).unwrap(), fit.upload(c.log_sd.view()).unwrap());
    let (rows, cols) = c.mean.dim();
    let mut whole = out.zeros(rows, cols).unwrap();
    fit.reparameterize(&mut whole, (&mean, &log_sd), c.sample).unwrap();
    let mut stack = out.upload(Array2::from_elem((rows + 3, cols + 5), 0.5).view()).unwrap();
    fit.reparameterize_block(&mut stack, (2, 4), (&mean, &log_sd), c.sample).unwrap();
    let (whole, stack) = (out.download(&whole).unwrap(), out.download(&stack).unwrap());
    for ((r, k), v) in stack.indexed_iter() {
        let inside = (2..2 + rows).contains(&r) && (4..4 + cols).contains(&k);
        assert_eq!(*v, if inside { whole[(r - 2, k - 4)] } else { 0.5 }, "({r}, {k})");
    }
    assert!(fit.reparameterize_block(&mut out.zeros(rows, cols).unwrap(), (1, 0), (&mean, &log_sd), c.sample).is_err(), "a block past the tensor");
}

#[test]
fn a_sample_written_into_a_block_is_the_whole_sample() {
    let host = Device::host();
    sample_in_a_block(&host, &host);
    #[cfg(target_os = "macos")]
    if let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault") {
        sample_in_a_block(&metal, &metal);
    }
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        let fit = wide.with_storage(Storage::F32).expect("CUDA holds f32");
        sample_in_a_block(&fit, &fit);
        sample_in_a_block(&fit, &wide.with_storage(Storage::Bf16).expect("CUDA holds bfloat16"));
        sample_in_a_block(&wide, &wide);
    }
}

/// IVON's direction from a constant gradient without noise is its full step with the unfiltered
/// full gradient: from a zero momentum (`W = 0`) each step's bias-corrected momentum `m / W'` is the
/// gradient `g`, so `d = (g + δ μ) / (h_{t−1} + δ)` and `s = −½ ln(N (h_t + δ))` from the first step
/// on, `h_t = β₂ᵗ h₀` without a curvature estimate. (A filter of the gradient by its measured spread
/// made no step before a spread was measured, and shrank every step after.)
fn constant_gradient_steps_fully(fit: &Device, wide: &Device) {
    let (rows, cols, tokens, v) = (2, 5, 64.0, 0.5);
    let delta = 1.0 / (tokens * v);
    let (mean, gradient, start) = (matrix(rows, cols, 21, 0.5, 0.0), matrix(rows, cols, 22, 2.0, 0.0), matrix(rows, cols, 23, 0.25, 0.5));
    let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
    let (mu, mut log_sd) = (up(&mean), up(&Array2::from_elem((rows, cols), -1.0)));
    let (mut momentum, mut curvature, g) = (fit.zeros(rows, cols).unwrap(), up(&start), up(&gradient));
    let groups = fit.group_map(&vec![0; rows * cols], (rows, cols)).unwrap();
    let variance = wide.upload(Array2::from_elem((1, 1), v).view()).unwrap();
    let (beta1, beta2) = (0.9, 0.75);
    let mut weight = 0.0;
    for t in 1..=4 {
        let step = PosteriorStep { gradient_scale: 1.0, factor_scale: 1.0, tokens, beta1, beta2, weight };
        let (mut direction, mut sums) = (fit.zeros(rows, cols).unwrap(), wide.zeros(1, 5).unwrap());
        fit.posterior_ivon((&mu, &mut log_sd), [&mut momentum, &mut curvature], (Some(&g), None, None), (&groups, &variance), (&mut direction, &mut sums), &step).unwrap();
        weight = step.correction();
        let (held, h) = (start.mapv(|h0| beta2.powi(t - 1) * h0), start.mapv(|h0| beta2.powi(t) * h0));
        let expected = Array2::from_shape_fn((rows, cols), |at| (gradient[at] + delta * mean[at]) / (held[at] + delta));
        close(&format!("{} step {t}'s direction", fit.name()), &fit.download(&direction).unwrap(), &expected, CHAIN);
        close(&format!("{} step {t}'s log sd", fit.name()), &fit.download(&log_sd).unwrap(), &h.mapv(|h| -0.5 * (tokens * (h + delta)).ln()), CHAIN);
    }
}

#[test]
fn a_constant_gradient_moves_the_mean_by_ivon_s_full_step() {
    let host = Device::host();
    constant_gradient_steps_fully(&host, &host);
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        constant_gradient_steps_fully(&wide.with_storage(Storage::F32).expect("CUDA holds f32"), &wide);
        constant_gradient_steps_fully(&wide, &wide);
    }
    if cfg!(target_os = "macos")
        && let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")
    {
        constant_gradient_steps_fully(&metal, &metal);
    }
}

/// A prior curvature input joins the curvature's average, of either sign, and the deviations are
/// the stationary ones of the total precision `N h + 1 / v` wherever it is positive (the direction
/// the curvature before the step): `σ² = 1 / (N (h + δ))`, below `v` for `h > 0` and above it for
/// `−δ < h < 0`; where `h + δ ≤ 0` the curvature keeps the signed average and `σ² = 1 / (N δ) = v`,
/// the prior's. The prior input is chosen to put the stepped curvature at each of the three.
fn prior_curvature_enters_the_total_precision(fit: &Device, wide: &Device) {
    let (rows, cols, tokens, v) = (3, 4, 64.0, 0.5);
    let delta = 1.0 / (tokens * v);
    let (mean, gradient, factor) = (matrix(rows, cols, 31, 0.5, 0.0), matrix(rows, cols, 32, 1.0, 0.0), matrix(rows, cols, 33, 1.0, 0.0));
    let start = matrix(rows, cols, 35, 0.25, 0.5);
    let (beta2, square) = (0.5, 0.5);
    // Stepped curvatures above zero, between −δ and zero, and below −δ, and the prior inputs that
    // give them: `h = start + (1 − β₂)(c u² + r − start)`.
    let targets = [0.3, -0.5 * delta, -0.5, 0.1 * delta];
    let wanted = Array2::from_shape_fn((rows, cols), |(r, c)| targets[(r + c) % targets.len()]);
    let prior = Array2::from_shape_fn((rows, cols), |at| (wanted[at] - beta2 * start[at]) / (1.0 - beta2) - square * factor[at] * factor[at]);
    let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
    let (mu, mut log_sd) = (up(&mean), up(&Array2::from_elem((rows, cols), -1.0)));
    let (mut momentum, mut curvature) = (fit.zeros(rows, cols).unwrap(), up(&start));
    let groups = fit.group_map(&vec![0; rows * cols], (rows, cols)).unwrap();
    let variance = wide.upload(Array2::from_elem((1, 1), v).view()).unwrap();
    let step = PosteriorStep { gradient_scale: 1.0, factor_scale: square, tokens, beta1: 0.9, beta2, weight: 0.0 };
    let (mut direction, mut sums) = (fit.zeros(rows, cols).unwrap(), wide.zeros(1, 5).unwrap());
    let (g, u, r) = (up(&gradient), up(&factor), up(&prior));
    fit.posterior_ivon((&mu, &mut log_sd), [&mut momentum, &mut curvature], (Some(&g), Some(&u), Some(&r)), (&groups, &variance), (&mut direction, &mut sums), &step).unwrap();
    let h = Array2::from_shape_fn((rows, cols), |at| start[at] + (1.0 - beta2) * (square * factor[at] * factor[at] + prior[at] - start[at]));
    assert!(h.iter().any(|x| *x < -delta) && h.iter().any(|x| *x > -delta && *x < 0.0) && h.iter().any(|x| *x > 0.0), "curvatures in all three ranges");
    let usable = |x: f64| if x + delta > 0.0 { x } else { 0.0 };
    let positive = h.mapv(usable);
    let name = fit.name();
    close(&format!("{name} signed curvature"), &fit.download(&curvature).unwrap(), &h, CHAIN);
    close(&format!("{name} log sd"), &fit.download(&log_sd).unwrap(), &positive.mapv(|x| -0.5 * (tokens * (x + delta)).ln()), CHAIN);
    let expected = Array2::from_shape_fn((rows, cols), |at| (gradient[at] + delta * mean[at]) / (usable(start[at]) + delta));
    close(&format!("{name} direction"), &fit.download(&direction).unwrap(), &expected, CHAIN);
    for (x, s) in h.iter().zip(fit.download(&log_sd).unwrap().iter()) {
        let variance = (2.0 * s).exp();
        if *x + delta <= 0.0 {
            assert!((variance - v).abs() <= 1e-5 * v, "{name}: σ² {variance} at a total precision below zero, against v = {v}");
        } else if *x < 0.0 {
            assert!(variance > v, "{name}: σ² {variance} at a negative curvature of positive total precision, not above v = {v}");
        }
    }
}

#[test]
fn a_prior_curvature_joins_the_average_and_sigma_follows_the_total_precision() {
    let host = Device::host();
    prior_curvature_enters_the_total_precision(&host, &host);
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        prior_curvature_enters_the_total_precision(&wide.with_storage(Storage::F32).expect("CUDA holds f32"), &wide);
        prior_curvature_enters_the_total_precision(&wide, &wide);
    }
    if cfg!(target_os = "macos")
        && let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")
    {
        prior_curvature_enters_the_total_precision(&metal, &metal);
    }
}

/// An absent gradient, factor or prior curvature steps as a zero one, bit for bit: the momentum
/// still decays and the curvature still averages toward the estimates present.
fn absent_inputs_are_zero(fit: &Device, wide: &Device) {
    let c = case(GroupAxis::Entries);
    let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
    let step_with = |inputs: [Option<&Array2<f64>>; 3]| -> Vec<Array2<f64>> {
        let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
        let (mean, mut log_sd) = (up(&c.mean), up(&c.log_sd));
        let (mut momentum, mut curvature) = (up(&c.moments[0]), up(&c.moments[1]));
        let variance = wide.upload_vec(c.count, 1, (0..c.count).map(|g| 0.01 * (g + 1) as f64).collect()).unwrap();
        let [gradient, factor, prior] = inputs.map(|a| a.map(up));
        let (mut direction, mut terms) = (fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap(), wide.zeros(c.count, 5).unwrap());
        fit.posterior_ivon((&mean, &mut log_sd), [&mut momentum, &mut curvature], (gradient.as_ref(), factor.as_ref(), prior.as_ref()), (&groups, &variance), (&mut direction, &mut terms), &c.step).unwrap();
        vec![fit.download(&direction).unwrap(), fit.download(&log_sd).unwrap(), fit.download(&momentum).unwrap(), fit.download(&curvature).unwrap(), wide.download(&terms).unwrap()]
    };
    let zero = Array2::zeros(c.mean.dim());
    let given = [&c.gradient, &c.factor, &c.prior];
    for absent in 0..3 {
        let with_zero: [Option<&Array2<f64>>; 3] = std::array::from_fn(|k| Some(if k == absent { &zero } else { given[k] }));
        let without: [Option<&Array2<f64>>; 3] = std::array::from_fn(|k| (k != absent).then_some(given[k]));
        assert!(step_with(with_zero) == step_with(without), "{}: input {absent} absent", fit.name());
    }
    assert!(step_with([Some(&zero); 3]) == step_with([None; 3]), "{}: every input absent", fit.name());
}

#[test]
fn an_absent_input_is_a_zero_input() {
    let host = Device::host();
    absent_inputs_are_zero(&host, &host);
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        absent_inputs_are_zero(&wide.with_storage(Storage::F32).expect("CUDA holds f32"), &wide);
        absent_inputs_are_zero(&wide, &wide);
    }
    if cfg!(target_os = "macos")
        && let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")
    {
        absent_inputs_are_zero(&metal, &metal);
    }
}

/// `group_curvature`, `group_divergence` and `group_code_length` of the case on `fit` (sums on
/// `wide`): the curvature rows of the case's factor at its posterior, in row 1 of 2 the code length
/// of its groups (groups 4, removed, and 6 out of the explanation), their scales sent against
/// `reference` when one is given, and the groups' variances.
fn removal_sums(fit: &Device, wide: &Device, reference: Option<&[f64]>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let c = case(GroupAxis::Entries);
    let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
    let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
    let mut curvature = wide.zeros(c.count, 3).unwrap();
    fit.group_curvature((&up(&c.factor), &up(&c.mean), &up(&c.log_sd)), &groups, &mut curvature).unwrap();
    let mut moments = wide.zeros(c.count, 3).unwrap();
    fit.group_moments((&up(&c.mean), &up(&c.log_sd)), &groups, &mut moments).unwrap();
    let column = |v: Vec<f64>| wide.upload_vec(v.len(), 1, v).unwrap();
    let reference = reference.map(|r| column(r.to_vec()));
    let (mut variance, mut divergence) = (wide.zeros(c.count, 1).unwrap(), wide.zeros(c.count, 2).unwrap());
    wide.group_divergence(&mut moments, reference.as_ref(), &mut variance, &mut divergence).unwrap();
    let weight = column((0..c.count).map(|g| if g == 4 || g == 6 { 0.0 } else { 1.0 }).collect());
    let constant = column((0..c.count).map(|g| 0.5 * (g as f64 + 2.0).ln()).collect());
    let mut lengths = wide.zeros(2, 3).unwrap();
    wide.group_code_length(&divergence, (&weight, &constant), &mut lengths, 1).unwrap();
    (wide.download(&curvature).unwrap(), wide.download(&lengths).unwrap(), wide.download(&variance).unwrap())
}

/// The code lengths with each group's reference `2^(2 − k)` times its variance on the host
/// (`k = g mod 5`), so that `log2(S / n)` sits at the integer `k − 2` against its reference: groups
/// 1, 3 and 6 take the edge of exponent 0's bin and group 5 that of exponent −1's (a saved bit
/// worth 0.05 nats more than the divergence's rise there), the rest stay at `S / n`, every choice
/// far from a tie in f32.
fn removal_sums_against_host(fit: &Device, wide: &Device) {
    let host = Device::host();
    let count = case(GroupAxis::Entries).count;
    let (_, _, variance) = removal_sums(&host, &host, None);
    let initial: Vec<f64> = (0..count).map(|g| if variance[(g, 0)] > 0.0 { variance[(g, 0)] * 2f64.powi(-(g as i32 % 5) + 2) } else { 1.0 }).collect();
    let (curvature, lengths, chosen) = removal_sums(fit, wide, Some(&initial));
    let (expected_curvature, expected_lengths, expected_chosen) = removal_sums(&host, &host, Some(&initial));
    for (g, (v, w)) in chosen.iter().zip(&expected_chosen).enumerate() {
        assert!((v - w).abs() <= 4.0 * CHAIN * w.abs(), "group {g}: v {v} against {w}");
    }
    let moved: Vec<usize> = (0..count).filter(|g| expected_chosen[(*g, 0)] != variance[(*g, 0)]).collect();
    assert_eq!(moved, [1, 3, 5, 6], "the groups whose variance takes a cheaper bin's edge");
    let sum_band = CHAIN + 40.0 * U / (1.0 - 40.0 * U);
    // A dot product u_G . mu_G cancels terms as large as its summed magnitudes, at most 40 entries
    // of the largest magnitude.
    close("curvature sums", &curvature, &expected_curvature, 40.0 * sum_band);
    assert_eq!(expected_lengths[(0, 0)], 0.0, "row 0 untouched");
    assert_eq!(expected_lengths[(1, 0)], 6.0, "six groups in the explanation");
    // The divergences' cancellation (`n |ln v| + |Σ 2s|`, below 40 x 10 per group) bounds the code
    // length's difference; the scales' bits are integers and equal.
    let bound = sum_band * 40.0 * 10.0 * count as f64;
    assert!((lengths[(1, 1)] - expected_lengths[(1, 1)]).abs() <= bound, "code length {} against {}", lengths[(1, 1)], expected_lengths[(1, 1)]);
}

/// A step's terms (`posterior_ivon`'s sums) against the group curvatures of its direction on the
/// same device, for each axis's case: `d · d`, `u · d` and `Σ h₀⁺ d²` (`h₀` the curvature before
/// the step) each bit for bit its group curvature's column 1 (the same entries in the same order;
/// the case's curvature stays positive).
fn step_terms_match_the_curvatures(fit: &Device, wide: &Device) {
    for axis in AXES {
        let c = case(axis);
        let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
        let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
        let (mean, mut log_sd) = (up(&c.mean), up(&c.log_sd));
        let (mut momentum, mut curvature, before) = (up(&c.moments[0]), up(&c.moments[1]), up(&c.moments[1]));
        assert!(c.moments[1].iter().all(|h| *h > 0.0), "a positive curvature before the step");
        let (gradient, factor, prior) = (up(&c.gradient), up(&c.factor), up(&c.prior));
        let variance = wide.upload_vec(c.count, 1, (0..c.count).map(|g| 0.01 * (g + 1) as f64).collect()).unwrap();
        let (mut direction, mut sums) = (fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap(), wide.zeros(c.count, 5).unwrap());
        fit.posterior_ivon((&mean, &mut log_sd), [&mut momentum, &mut curvature], (Some(&gradient), Some(&factor), Some(&prior)), (&groups, &variance), (&mut direction, &mut sums), &c.step).unwrap();
        assert!(fit.download(&curvature).unwrap().iter().all(|h| *h > 0.0), "{} {axis:?}: a positive curvature", fit.name());
        let mut weighted = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
        fit.hadamard(&mut weighted, &before, &direction, false).unwrap();
        let terms = wide.download(&sums).unwrap();
        for (k, x) in [&direction, &factor, &weighted].into_iter().enumerate() {
            let mut part = wide.zeros(c.count, 3).unwrap();
            fit.group_curvature((x, &direction, &log_sd), &groups, &mut part).unwrap();
            let part = wide.download(&part).unwrap();
            for g in 0..c.count {
                assert_eq!(terms[(g, k)], part[(g, 1)], "{} {axis:?}: term {k} of group {g}", fit.name());
            }
        }
    }
}

/// `posterior_finish` against the `axpy`, `move_toward` and `group_moments` it replaces on the same
/// device, bit for bit, for each axis's case.
fn finish_matches_its_parts(fit: &Device, wide: &Device) {
    for axis in AXES {
        let c = case(axis);
        let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
        let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
        let (direction, log_sd) = (up(&matrix(c.mean.nrows(), c.mean.ncols(), 9, 0.05, 0.0)), up(&c.log_sd));
        let ((mut mean, mut average), mut sums) = ((up(&c.mean), up(&c.moments[0])), wide.zeros(c.count, 3).unwrap());
        fit.posterior_finish((&mut mean, &direction, 0.375), (&mut average, 0.0625), &log_sd, &groups, &mut sums).unwrap();
        let ((mut m, mut a), mut s) = ((up(&c.mean), up(&c.moments[0])), wide.zeros(c.count, 3).unwrap());
        fit.axpy(&mut m, -0.375, &direction).unwrap();
        fit.move_toward(&mut a, 0.0625, &m).unwrap();
        fit.group_moments((&a, &log_sd), &groups, &mut s).unwrap();
        assert_eq!(fit.download(&mean).unwrap(), fit.download(&m).unwrap(), "{} {axis:?}: the mean", fit.name());
        assert_eq!(fit.download(&average).unwrap(), fit.download(&a).unwrap(), "{} {axis:?}: the average", fit.name());
        assert_eq!(wide.download(&sums).unwrap(), wide.download(&s).unwrap(), "{} {axis:?}: the moments", fit.name());
    }
}

/// On CUDA every entry of a step is in the masters' storage: an f32 posterior's step refuses a
/// gradient in `other`'s storage (bfloat16).
fn refuses_another_storage(fit: &Device, other: &Device, wide: &Device) {
    let c = case(GroupAxis::Rows);
    let up = |d: &Device, a: &Array2<f64>| d.upload(a.view()).unwrap();
    let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
    let (mean, mut log_sd) = (up(fit, &c.mean), up(fit, &c.log_sd));
    let (mut momentum, mut curvature) = (up(fit, &c.moments[0]), up(fit, &c.moments[1]));
    let variance = wide.upload_vec(c.count, 1, vec![0.5; c.count]).unwrap();
    let (mut direction, mut sums) = (fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap(), wide.zeros(c.count, 5).unwrap());
    let gradient = up(other, &c.gradient);
    let step = fit.posterior_ivon((&mean, &mut log_sd), [&mut momentum, &mut curvature], (Some(&gradient), None, None), (&groups, &variance), (&mut direction, &mut sums), &c.step);
    assert!(step.is_err(), "a {:?} gradient beside {:?} masters", other.storage(), fit.storage());
}

#[test]
fn a_steps_terms_and_finish_are_the_operations_they_replace() {
    let host = Device::host();
    step_terms_match_the_curvatures(&host, &host);
    finish_matches_its_parts(&host, &host);
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
        let half = wide.with_storage(Storage::Bf16).expect("CUDA holds bfloat16");
        step_terms_match_the_curvatures(&narrow, &wide);
        step_terms_match_the_curvatures(&wide, &wide);
        finish_matches_its_parts(&narrow, &wide);
        finish_matches_its_parts(&wide, &wide);
        refuses_another_storage(&narrow, &half, &wide);
    }
    // On Linux the single-precision device is CUDA's f32 storage, whose sums are no float64 tensor.
    if cfg!(target_os = "macos")
        && let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")
    {
        step_terms_match_the_curvatures(&metal, &metal);
        finish_matches_its_parts(&metal, &metal);
    }
}

/// `move_toward` and `scaled` against the copies, zeros and `axpy`s they replace, bit for bit (zeros
/// among `x`, of either sign, for `scaled`).
fn moves_match_their_compositions(d: &Device) {
    let (x, y) = (d.upload(matrix(7, 33, 11, 2.0, 0.5).view()).unwrap(), d.upload(matrix(7, 33, 12, 1.5, -0.25).view()).unwrap());
    let mut signed = matrix(7, 33, 13, 1.0, 0.0);
    for (k, v) in signed.iter_mut().enumerate().filter(|(k, _)| k % 5 == 0) {
        *v = if k % 2 == 0 { -0.0 } else { 0.0 };
    }
    let signed = d.upload(signed.view()).unwrap();
    for alpha in [-1.0_f64, -0.375, 0.0, 1.0, 3.0] {
        let mut expected = d.zeros(7, 33).unwrap();
        d.axpy(&mut expected, alpha, &signed).unwrap();
        let bits = |t: &Tensor| d.download(t).unwrap().mapv(f64::to_bits);
        assert_eq!(bits(&d.scaled(alpha, &signed).unwrap()), bits(&expected), "{} scaled {alpha}", d.name());
    }
    for alpha in [-1.0_f64, -0.375, 0.1, 3.0] {
        let mut moved = d.copy(&y).unwrap();
        d.move_toward(&mut moved, alpha.abs() / 4.0, &x).unwrap();
        let mut difference = d.copy(&x).unwrap();
        d.axpy(&mut difference, -1.0, &y).unwrap();
        let mut expected = d.copy(&y).unwrap();
        d.axpy(&mut expected, alpha.abs() / 4.0, &difference).unwrap();
        assert_eq!(d.download(&moved).unwrap(), d.download(&expected).unwrap(), "{} move_toward {alpha}", d.name());
    }
}

#[test]
fn a_move_toward_is_the_copy_and_axpys_it_replaces() {
    moves_match_their_compositions(&Device::host());
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        moves_match_their_compositions(&wide.with_storage(Storage::F32).expect("CUDA holds f32"));
        moves_match_their_compositions(&wide);
    }
    if cfg!(target_os = "macos")
        && let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")
    {
        moves_match_their_compositions(&metal);
    }
}

/// CUDA's reduction over groups of one column each (four adjacent columns per block, or 32 from
/// 8192 such groups; group c is 7c mod cols, one column each while 7 does not divide cols) against the same entries transposed and grouped by rows, which the
/// per-segment reduction takes in the same order: every group sum, and every entry the kernels
/// write, bit for bit. 600 rows (two full passes of a block and part of a third) and `cols` columns
/// in a shuffled group order (full tiles of columns not adjacent in group order, and part of one),
/// the last four groups beyond the count, and removed entries.
fn single_columns_match_their_transposes(fit: &Device, wide: &Device, cols: usize) {
    let (rows, count) = (600, cols - 4);
    let group = |c: usize| ((c * 7) % cols) as u32;
    let ids: Vec<u32> = (0..rows * cols).map(|i| group(i % cols)).collect();
    let by_rows: Vec<u32> = (0..cols * rows).map(|i| group(i / rows)).collect();
    let (map, map_t) = (fit.group_map(&ids, (rows, cols)).unwrap(), fit.group_map(&by_rows, (cols, rows)).unwrap());
    assert_eq!((map.axis(), map_t.axis()), (GroupAxis::Columns, GroupAxis::Rows));
    let (mut mean, mut log_sd) = (matrix(rows, cols, 1, 0.5, 0.0), matrix(rows, cols, 2, 0.5, -3.0));
    for i in (0..rows * cols).filter(|i| i % 37 == 0) {
        mean[(i / cols, i % cols)] = 0.0;
        log_sd[(i / cols, i % cols)] = f64::NEG_INFINITY;
    }
    let (momentum, curvature) = (matrix(rows, cols, 3, 0.1, 0.0), matrix(rows, cols, 4, 0.5, 1.0));
    let (gradient, factor, prior) = (matrix(rows, cols, 7, 3.0, 0.0), matrix(rows, cols, 8, 2.0, 0.0), matrix(rows, cols, 5, 0.5, 0.0));
    let both = |a: &Array2<f64>| (fit.upload(a.view()).unwrap(), fit.upload(a.t().as_standard_layout().view()).unwrap());
    let back = |t: &Tensor| fit.download(t).unwrap().t().as_standard_layout().to_owned();
    let pair = |what: &str, (a, b): (&Tensor, &Tensor), d: &Device| assert_eq!(d.download(a).unwrap(), d.download(b).unwrap(), "{} {what}", fit.name());
    let ((m, mt), (s, st), (u, ut)) = (both(&mean), both(&log_sd), both(&factor));
    let sums = |columns: usize| (wide.zeros(count, columns).unwrap(), wide.zeros(count, columns).unwrap());
    let (mut x, mut y) = sums(3);
    fit.group_moments((&m, &s), &map, &mut x).unwrap();
    fit.group_moments((&mt, &st), &map_t, &mut y).unwrap();
    pair("moments", (&x, &y), wide);
    let (mut x, mut y) = sums(3);
    fit.group_curvature((&u, &m, &s), &map, &mut x).unwrap();
    fit.group_curvature((&ut, &mt, &st), &map_t, &mut y).unwrap();
    pair("curvature", (&x, &y), wide);
    let ((g, gt), (r, rt)) = (both(&gradient), both(&prior));
    let variance = wide.upload_vec(count, 1, (0..count).map(|g| 0.01 * (g + 1) as f64).collect()).unwrap();
    let step = case(GroupAxis::Rows).step;
    let (mut s, mut st) = both(&log_sd);
    let ((mut p, mut pt), (mut c, mut ct)) = (both(&momentum), both(&curvature));
    let (mut d, mut dt) = (fit.zeros(rows, cols).unwrap(), fit.zeros(cols, rows).unwrap());
    let (mut x, mut y) = sums(5);
    fit.posterior_ivon((&m, &mut s), [&mut p, &mut c], (Some(&g), Some(&u), Some(&r)), (&map, &variance), (&mut d, &mut x), &step).unwrap();
    fit.posterior_ivon((&mt, &mut st), [&mut pt, &mut ct], (Some(&gt), Some(&ut), Some(&rt)), (&map_t, &variance), (&mut dt, &mut y), &step).unwrap();
    pair("step terms", (&x, &y), wide);
    for (what, a, b) in [("direction", &d, &dt), ("log sd", &s, &st), ("momentum", &p, &pt), ("curvature", &c, &ct)] {
        assert_eq!(fit.download(a).unwrap(), back(b), "{} stepped {what}", fit.name());
    }
    let ((mut m, mut mt), (mut a, mut at)) = (both(&mean), both(&momentum));
    let (mut x, mut y) = sums(3);
    fit.posterior_finish((&mut m, &d, 0.375), (&mut a, 0.0625), &s, &map, &mut x).unwrap();
    fit.posterior_finish((&mut mt, &dt, 0.375), (&mut at, 0.0625), &st, &map_t, &mut y).unwrap();
    pair("finish sums", (&x, &y), wide);
    for (what, a, b) in [("finished mean", &m, &mt), ("average", &a, &at)] {
        assert_eq!(fit.download(a).unwrap(), back(b), "{} {what}", fit.name());
    }
}

#[test]
fn cuda_single_column_groups_reduce_as_their_transposes() {
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    for cols in [22, 8194] {
        single_columns_match_their_transposes(&narrow, &wide, cols);
        single_columns_match_their_transposes(&wide, &wide, cols);
    }
}

/// CUDA's reduction over groups of one entry each (one thread a group) against the same entries as
/// column 0 of an operator whose groups are its rows and whose column 1 is removed, which the
/// per-segment reduction takes a block a group: every group sum, and every entry of column 0 the
/// kernels write, bit for bit, for the posterior step, its finish, the moments and the curvature.
/// 3000 rows in a shuffled group order, the last ten beyond the count, and removed entries.
fn single_entries_match_their_rows(fit: &Device, wide: &Device) {
    let (rows, count) = (3000, 2990);
    let group = |r: usize| ((r * 7) % rows) as u32;
    let (map, map_2) = (fit.group_map(&(0..rows).map(group).collect::<Vec<_>>(), (rows, 1)).unwrap(), fit.group_map(&(0..2 * rows).map(|i| group(i / 2)).collect::<Vec<_>>(), (rows, 2)).unwrap());
    let widen = |a: &Array2<f64>, other: f64| Array2::from_shape_fn((rows, 2), |(r, c)| if c == 0 { a[(r, 0)] } else { other });
    let (mut mean, mut log_sd) = (matrix(rows, 1, 1, 0.5, 0.0), matrix(rows, 1, 2, 0.5, -3.0));
    for r in (0..rows).filter(|r| r % 37 == 0) {
        mean[(r, 0)] = 0.0;
        log_sd[(r, 0)] = f64::NEG_INFINITY;
    }
    let (momentum, curvature) = (matrix(rows, 1, 3, 0.1, 0.0), matrix(rows, 1, 4, 0.5, 1.0));
    let (gradient, factor, prior) = (matrix(rows, 1, 7, 3.0, 0.0), matrix(rows, 1, 8, 2.0, 0.0), matrix(rows, 1, 5, 0.5, 0.0));
    let both = |a: &Array2<f64>, other: f64| (fit.upload(a.view()).unwrap(), fit.upload(widen(a, other).view()).unwrap());
    let first = |t: &Tensor| fit.download(t).unwrap().column(0).to_owned().insert_axis(ndarray::Axis(1));
    let pair = |what: &str, (a, b): (&Tensor, &Tensor), d: &Device| assert_eq!(d.download(a).unwrap(), d.download(b).unwrap(), "{} {what}", fit.name());
    let ((m, m2), (s, s2), (u, u2)) = (both(&mean, 0.0), both(&log_sd, f64::NEG_INFINITY), both(&factor, 1.0));
    let sums = |columns: usize| (wide.zeros(count, columns).unwrap(), wide.zeros(count, columns).unwrap());
    let (mut x, mut y) = sums(3);
    fit.group_moments((&m, &s), &map, &mut x).unwrap();
    fit.group_moments((&m2, &s2), &map_2, &mut y).unwrap();
    pair("moments", (&x, &y), wide);
    let (mut x, mut y) = sums(3);
    fit.group_curvature((&u, &m, &s), &map, &mut x).unwrap();
    fit.group_curvature((&u2, &m2, &s2), &map_2, &mut y).unwrap();
    pair("curvature", (&x, &y), wide);
    let ((g, g2), (r, r2)) = (both(&gradient, 1.0), both(&prior, 1.0));
    let variance = wide.upload_vec(count, 1, (0..count).map(|g| 0.01 * (g + 1) as f64).collect()).unwrap();
    let step = case(GroupAxis::Rows).step;
    let ((mut s, mut s2), (mut p, mut p2), (mut c, mut c2)) = (both(&log_sd, f64::NEG_INFINITY), both(&momentum, 0.0), both(&curvature, 0.0));
    let (mut d, mut d2) = (fit.zeros(rows, 1).unwrap(), fit.zeros(rows, 2).unwrap());
    let (mut x, mut y) = sums(5);
    fit.posterior_ivon((&m, &mut s), [&mut p, &mut c], (Some(&g), Some(&u), Some(&r)), (&map, &variance), (&mut d, &mut x), &step).unwrap();
    fit.posterior_ivon((&m2, &mut s2), [&mut p2, &mut c2], (Some(&g2), Some(&u2), Some(&r2)), (&map_2, &variance), (&mut d2, &mut y), &step).unwrap();
    pair("step terms", (&x, &y), wide);
    for (what, a, b) in [("direction", &d, &d2), ("log sd", &s, &s2), ("momentum", &p, &p2), ("curvature", &c, &c2)] {
        assert_eq!(fit.download(a).unwrap(), first(b), "{} stepped {what}", fit.name());
    }
    let ((mut m, mut m2), (mut a, mut a2)) = (both(&mean, 0.0), both(&momentum, 0.0));
    let (mut x, mut y) = sums(3);
    fit.posterior_finish((&mut m, &d, 0.375), (&mut a, 0.0625), &s, &map, &mut x).unwrap();
    fit.posterior_finish((&mut m2, &d2, 0.375), (&mut a2, 0.0625), &s2, &map_2, &mut y).unwrap();
    pair("finish sums", (&x, &y), wide);
    for (what, a, b) in [("finished mean", &m, &m2), ("average", &a, &a2)] {
        assert_eq!(fit.download(a).unwrap(), first(b), "{} {what}", fit.name());
    }
}

#[test]
fn cuda_single_entry_groups_reduce_as_their_rows() {
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    single_entries_match_their_rows(&narrow, &wide);
    single_entries_match_their_rows(&wide, &wide);
}

/// Samples gathered and run together ([`Device::run_samples`]) against each written by its own
/// `reparameterize_block`, bit for bit: two operators whole and both into blocks of a stacked
/// output, on `fit` with outputs in `out`'s storage.
fn gathered_samples_match_their_own(fit: &Device, out: &Device) {
    let c = case(GroupAxis::Rows);
    let (rows, cols) = c.mean.dim();
    let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
    let operators = [(up(&c.mean), up(&c.log_sd)), (up(&c.moments[0]), up(&(&c.log_sd * 0.5)))];
    let key = 0x0123_4567_89ab_cdef;
    let mut whole: Vec<Tensor> = (0..2).map(|_| out.zeros(rows, cols).unwrap()).collect();
    let mut stack = out.zeros(2 * rows + 1, cols + 3).unwrap();
    let mut samples = fit.samples(key);
    for (i, ((mean, log_sd), theta)) in operators.iter().zip(&mut whole).enumerate() {
        // SAFETY: every tensor lives past run_samples below, and nothing reads them before it.
        unsafe { fit.add_sample(&mut samples, theta, (0, 0), (mean, log_sd), i as u64 + 3) }.unwrap();
    }
    for (i, (mean, log_sd)) in operators.iter().enumerate() {
        // SAFETY: as above.
        unsafe { fit.add_sample(&mut samples, &mut stack, (i * rows + 1, 2), (mean, log_sd), i as u64 + 3) }.unwrap();
    }
    fit.run_samples(samples).unwrap();
    let mut expected_stack = out.zeros(2 * rows + 1, cols + 3).unwrap();
    for (i, ((mean, log_sd), theta)) in operators.iter().zip(&whole).enumerate() {
        let mut expected = out.zeros(rows, cols).unwrap();
        fit.reparameterize_block(&mut expected, (0, 0), (mean, log_sd), (key, i as u64 + 3)).unwrap();
        assert_eq!(out.download(theta).unwrap(), out.download(&expected).unwrap(), "{} operator {i} whole", fit.name());
        fit.reparameterize_block(&mut expected_stack, (i * rows + 1, 2), (mean, log_sd), (key, i as u64 + 3)).unwrap();
    }
    assert_eq!(out.download(&stack).unwrap(), out.download(&expected_stack).unwrap(), "{} stacked blocks", fit.name());
}

#[test]
fn gathered_samples_are_the_ones_each_writes_alone() {
    let host = Device::host();
    gathered_samples_match_their_own(&host, &host);
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
        gathered_samples_match_their_own(&narrow, &narrow);
        gathered_samples_match_their_own(&narrow, &wide.with_storage(Storage::Bf16).expect("CUDA holds bfloat16"));
        gathered_samples_match_their_own(&wide, &wide);
    }
    if let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault") {
        gathered_samples_match_their_own(&metal, &metal);
    }
}

#[test]
fn removal_sums_on_the_host_match_their_formulas() {
    let host = Device::host();
    let c = case(GroupAxis::Entries);
    let (_, _, variance) = removal_sums(&host, &host, None);
    let initial: Vec<f64> = (0..c.count).map(|g| if variance[(g, 0)] > 0.0 { variance[(g, 0)] * 2f64.powi(3) } else { 1.0 }).collect();
    let (curvature, lengths, chosen) = removal_sums(&host, &host, Some(&initial));
    assert_eq!(chosen, variance, "every variance stays at S / n");
    for g in 0..c.count {
        let entries: Vec<usize> = (0..c.groups.len()).filter(|i| c.groups[*i] as usize == g).collect();
        let at = |a: &Array2<f64>, i: usize| a[(i / a.ncols(), i % a.ncols())];
        let live: Vec<usize> = entries.iter().copied().filter(|i| at(&c.log_sd, *i) != f64::NEG_INFINITY).collect();
        let dot: f64 = live.iter().map(|i| at(&c.factor, *i) * at(&c.mean, *i)).sum();
        let noise: f64 = live.iter().map(|i| at(&c.factor, *i).powi(2) * (2.0 * at(&c.log_sd, *i)).exp()).sum();
        assert_eq!(curvature[(g, 0)], live.len() as f64);
        assert!((curvature[(g, 1)] - dot).abs() <= 1e-12 * dot.abs().max(1.0) && (curvature[(g, 2)] - noise).abs() <= 1e-12 * noise, "group {g}");
    }
    // Every scale is 2^-3 of its start: the exponent -3, zigzag 5, plus one 6, an Elias delta
    // codeword of 2 + 2 x 1 + 1 = 5 bits. Exponent -2's bin codes in as many bits, and the edge of
    // exponent -1's (4 bits) lies 1.5 ln 2 from S / n, where the divergence of a group of at least
    // 24 entries has risen by 4.7 nats, more than the bit it saves: every variance stays at S / n.
    let mut moments = host.zeros(c.count, 3).unwrap();
    host.group_moments((&host.upload(c.mean.view()).unwrap(), &host.upload(c.log_sd.view()).unwrap()), &host.group_map(&c.groups, c.mean.dim()).unwrap(), &mut moments).unwrap();
    let (mut v, mut d) = (host.zeros(c.count, 1).unwrap(), host.zeros(c.count, 2).unwrap());
    host.group_divergence(&mut moments, None, &mut v, &mut d).unwrap();
    let d = host.download(&d).unwrap();
    let expected: f64 = (0..c.count).filter(|g| *g != 4 && *g != 6).map(|g| d[(g, 0)] + 0.5 * (g as f64 + 2.0).ln() + 5.0 * std::f64::consts::LN_2).sum();
    assert!((lengths[(1, 1)] - expected).abs() <= 1e-12 * expected.abs(), "{} against {expected}", lengths[(1, 1)]);
}

#[cfg(target_os = "macos")]
#[test]
fn removal_sums_on_the_apple_gpu_match_the_host() {
    let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    removal_sums_against_host(&metal, &metal);
}

#[test]
fn removal_sums_on_cuda_match_the_host() {
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    removal_sums_against_host(&narrow, &wide);
    removal_sums_against_host(&wide, &wide);
}

/// `resolved_counts` of a gated and an ungated layer on `fit` (sums on `wide`): values, slopes and
/// noises from the case's arrays, a dead column, and exact zeros in the values.
fn resolved_counts_on(fit: &Device, wide: &Device) -> [Array2<f64>; 2] {
    let c = case(GroupAxis::Entries);
    let mut value = c.mean.clone();
    value.row_mut(1).fill(0.0);
    let (slope, phi, noise_z, noise_y) = (c.gradient.clone(), c.factor.clone(), c.log_sd.mapv(|s| if s.is_finite() { (2.0 * s).exp() } else { 0.0 }), c.moments[0].clone());
    let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
    let alive = fit.upload_indices(&(0..value.ncols()).map(|j| u32::from(j % 7 != 3)).collect::<Vec<_>>()).unwrap();
    let mut gated = wide.zeros(1, 3).unwrap();
    fit.resolved_counts((&up(&value), &up(&slope), &up(&phi)), (&up(&noise_z), Some(&up(&noise_y))), &alive, &mut gated).unwrap();
    let mut plain = wide.zeros(1, 3).unwrap();
    fit.resolved_counts((&up(&value), &up(&slope), &up(&value)), (&up(&noise_z), None), &alive, &mut plain).unwrap();
    [wide.download(&gated).unwrap(), wide.download(&plain).unwrap()]
}

fn resolved_counts_against_host(fit: &Device, wide: &Device) {
    let host = Device::host();
    let expected = resolved_counts_on(&host, &host);
    // An entry within f32 rounding of its noise may fall either way: allow a few.
    for (k, (actual, expected)) in resolved_counts_on(fit, wide).iter().zip(&expected).enumerate() {
        assert_eq!(actual[(0, 0)], expected[(0, 0)], "layer {k}: entries counted");
        assert_eq!(actual[(0, 1)], expected[(0, 1)], "layer {k}: nonzero entries");
        assert!((actual[(0, 2)] - expected[(0, 2)]).abs() <= 2.0, "layer {k}: resolved {} against {}", actual[(0, 2)], expected[(0, 2)]);
    }
}

#[test]
fn resolved_counts_on_the_host_count_their_entries() {
    let host = Device::host();
    let [gated, plain] = resolved_counts_on(&host, &host);
    let c = case(GroupAxis::Entries);
    let alive = (0..c.mean.ncols()).filter(|j| j % 7 != 3).count() * c.mean.nrows();
    assert_eq!(gated[(0, 0)], alive as f64);
    assert_eq!(plain[(0, 0)], alive as f64);
    assert!(gated[(0, 1)] < alive as f64 && gated[(0, 2)] <= gated[(0, 1)] && plain[(0, 2)] <= plain[(0, 1)]);
}

#[cfg(target_os = "macos")]
#[test]
fn resolved_counts_on_the_apple_gpu_match_the_host() {
    let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    resolved_counts_against_host(&metal, &metal);
}

#[test]
fn resolved_counts_on_cuda_match_the_host() {
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") else { return };
    let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
    resolved_counts_against_host(&narrow, &wide);
    resolved_counts_against_host(&wide, &wide);
}

/// The bits of the Elias δ codeword of the signed index of `k` (`2k + 1` for `k ≥ 0`, `2|k|` for
/// `k < 0`) from `⌊log2⌋` of the index and of its length: the test's own form of
/// `scale_code_bits`.
fn elias_delta(k: i64) -> f64 {
    let x = if k >= 0 { 2 * k.unsigned_abs() + 1 } else { 2 * k.unsigned_abs() };
    let low = x.ilog2();
    f64::from(low + 2 * (low + 1).ilog2() + 1)
}

/// `group_prior` against every bin of exponent −40 to 40, each charged `KL + ln 2 · bits` at
/// `S / n` clamped to it: its charge is the least of theirs, for groups of 1 to 500 entries with
/// `S / n` from `2^-12` to `2^12` of the reference (every entry `μ² = σ² = S / 2n`), and its
/// divergence is `KL(q ‖ N(0, v I))` at its variance `v`, which is `S / n` or a bin's edge.
#[test]
fn the_prior_variance_is_the_least_charge_over_the_bins() {
    let ln2 = std::f64::consts::LN_2;
    for k in -2100_i64..=2100 {
        assert_eq!(gam_gpu::tensor::scale_code_bits(k), elias_delta(k), "exponent {k}");
    }
    let reference = 0.37;
    for n in [1.0, 3.0, 24.0, 40.0, 500.0] {
        for step in -240..=240 {
            let t = f64::from(step) / 20.0 + 0.013;
            let centre = reference * t.exp2();
            let log_variance = n * (0.5 * centre).ln();
            let divergence = |v: f64| 0.5 * (n * centre / v + n * v.ln() - n - log_variance);
            let (v, kl, bits) = gam_gpu::tensor::group_prior(n, n * centre, log_variance, Some(reference));
            let least = (-40_i64..=40)
                .map(|k| divergence(reference * t.clamp(k as f64 - 0.5, k as f64 + 0.5).exp2()) + ln2 * elias_delta(k))
                .fold(f64::INFINITY, f64::min);
            let charge = kl + ln2 * bits;
            assert!((charge - least).abs() <= 1e-9 * least.abs().max(1.0), "n {n}, t {t}: {charge} against the least {least}");
            assert!((kl - divergence(v)).abs() <= 1e-9 * kl.abs().max(1.0), "n {n}, t {t}: KL {kl} at v {v}");
            // The distance of `log2(v / v⁰)` from the nearest bin edge, a half-integer.
            let shifted = (v / reference).log2() - 0.5;
            let edge = (shifted - shifted.round()).abs();
            // `S / n` as `group_prior` forms it from the sums, `(n c) / n`, which need not be `c` itself.
            assert!(v == n * centre / n || edge <= 1e-12, "n {n}, t {t}: v {v} is neither S / n nor an edge");
        }
    }
}

/// Groups' sums `(n, S, Σ 2s)` (every entry `μ² = σ² = S / 2n`) with `log2(S / n)` at the given
/// power of two of their references, and an empty group last; returned with the references.
fn prior_case() -> (Array2<f64>, Vec<f64>) {
    // (n, log2 of S / n against the reference, the reference)
    let groups = [(40.0, 0.52, 1.0), (40.0, 1.0, 0.25), (8.0, 4.6, 3.0), (24.0, -0.53, 0.5), (24.0, -2.0, 1.5), (500.0, 0.52, 2.0), (1.0, 7.3, 0.125), (24.0, 0.0, 1.0)];
    let mut sums = Array2::zeros((groups.len() + 1, 3));
    let mut references = vec![1.0; groups.len() + 1];
    for (g, &(n, t, r)) in groups.iter().enumerate() {
        let centre: f64 = r * f64::exp2(t);
        sums[(g, 0)] = n;
        sums[(g, 1)] = n * centre;
        sums[(g, 2)] = n * (0.5 * centre).ln();
        references[g] = r;
    }
    (sums, references)
}

/// `group_divergence` of [`prior_case`] on `d`: the variances and the divergences with their bits.
fn priors_on(d: &Device) -> (Array2<f64>, Array2<f64>) {
    let (sums, references) = prior_case();
    let rows = sums.nrows();
    let mut s = d.upload(sums.view()).unwrap();
    let r = d.upload_vec(rows, 1, references).unwrap();
    let (mut v, mut divergence) = (d.zeros(rows, 1).unwrap(), d.zeros(rows, 2).unwrap());
    d.group_divergence(&mut s, Some(&r), &mut v, &mut divergence).unwrap();
    assert!(d.download(&s).unwrap().iter().all(|x| *x == 0.0), "{}: the sums are zeroed", d.name());
    (d.download(&v).unwrap(), d.download(&divergence).unwrap())
}

/// On the host, [`prior_case`]'s groups take the bins `group_prior` reasons out (its doc): a centre
/// just inside an expensive bin next to a cheaper one (groups 0, 3 and 5: 2^0.52 and 2^-0.53 of
/// their references, in bins of 4 bits) moves to the cheaper bin's edge and lowers the charge;
/// group 1's, at the middle of exponent 1's bin, still gains by the move to exponent 0's edge;
/// group 2's passes exponent 4's bin (a code as long as exponent 5's) for exponent 3's edge; group
/// 4's moves one bin; group 6's (one entry, seven bins out) stays, as does group 7's at its
/// reference; the empty group is zero.
#[test]
fn prior_variances_on_the_host_take_the_cheaper_bins() {
    let ln2 = std::f64::consts::LN_2;
    let (sums, references) = prior_case();
    let (v, divergence) = priors_on(&Device::host());
    // Per group, `log2(v / v⁰)` and the bits of its scale.
    let expected = [(0.5, 1.0), (0.5, 1.0), (3.5, 5.0), (-0.5, 1.0), (-1.5, 4.0), (0.5, 1.0), (7.3, 8.0), (0.0, 1.0)];
    for (g, &(at, bits)) in expected.iter().enumerate() {
        let (n, centre) = (sums[(g, 0)], sums[(g, 1)] / sums[(g, 0)]);
        assert!((v[(g, 0)] - references[g] * f64::exp2(at)).abs() <= 1e-12 * v[(g, 0)], "group {g}: v {}", v[(g, 0)]);
        assert_eq!(divergence[(g, 1)], bits, "group {g}: bits");
        let stays = g >= 6;
        assert_eq!(v[(g, 0)] == centre, stays, "group {g}: v {} against S / n {centre}", v[(g, 0)]);
        let at_centre = gam_gpu::tensor::group_prior(n, n * centre, sums[(g, 2)], None).1;
        let unmoved = at_centre + ln2 * gam_gpu::tensor::scale_code_bits((centre / references[g]).log2().round() as i64);
        let charge = divergence[(g, 0)] + ln2 * bits;
        if stays {
            assert!((charge - unmoved).abs() <= 1e-12 * unmoved.abs(), "group {g} stays");
        } else {
            assert!(charge < unmoved, "group {g}: the cheaper bin lowers the charge, {charge} against {unmoved}");
        }
    }
    assert_eq!((v[(8, 0)], divergence[(8, 0)], divergence[(8, 1)]), (0.0, 0.0, 0.0));
}

/// `group_divergence` with references on `d` against the host: the same bits, and the variances and
/// divergences within f32 rounding of the terms they cancel.
fn priors_against_host(d: &Device) {
    let (sums, _) = prior_case();
    let (v, divergence) = priors_on(d);
    let (hv, hdivergence) = priors_on(&Device::host());
    for g in 0..sums.nrows() {
        assert_eq!(divergence[(g, 1)], hdivergence[(g, 1)], "{} group {g}: bits", d.name());
        assert!((v[(g, 0)] - hv[(g, 0)]).abs() <= 4.0 * CHAIN * hv[(g, 0)], "{} group {g}: v {} against {}", d.name(), v[(g, 0)], hv[(g, 0)]);
        let (n, second) = (sums[(g, 0)], sums[(g, 1)]);
        let cancelled = if n > 0.0 { n * (second / n).ln().abs() + sums[(g, 2)].abs() + n } else { 0.0 };
        assert!((divergence[(g, 0)] - hdivergence[(g, 0)]).abs() <= 4.0 * CHAIN * cancelled, "{} group {g}: KL {} against {}", d.name(), divergence[(g, 0)], hdivergence[(g, 0)]);
    }
}

#[test]
fn prior_variances_on_every_accelerator_match_the_host() {
    if cfg!(target_os = "macos")
        && let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")
    {
        priors_against_host(&metal);
    }
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        priors_against_host(&wide);
    }
}

/// Two steps of the curvature draws' innovations (`Device::curvature_innovation`) on each axis's
/// case, with the second step's curvature, factor and prior curvature its own: the first stores
/// `e₋ = c u² + r − h` and adds nothing; the second stores `e` and adds `(Δy e₋, Δy², h²)` per group,
/// `Δy = e − (1 − K) e₋`; removed entries (group 4) and those of groups at or beyond the sums' rows
/// (7) are left. Returns the innovations and the sums.
fn innovations(fit: &Device, wide: &Device, c: &Case, gain: f64) -> (Array2<f64>, Array2<f64>) {
    let up = |m: &Array2<f64>| fit.upload(m.view()).unwrap();
    let (rows, cols) = c.mean.dim();
    let groups = fit.group_map(&c.groups, (rows, cols)).unwrap();
    let log_sd = up(&c.log_sd);
    let mut innovation = fit.zeros(rows, cols).unwrap();
    let mut sums = wide.zeros(c.count - 1, 3).unwrap();
    let scale = c.step.factor_scale;
    fit.curvature_innovation((Some(&up(&c.factor)), Some(&up(&c.prior))), (&up(&c.moments[1]), &log_sd), &mut innovation, &groups, &mut sums, (scale, gain, true)).unwrap();
    assert!(wide.download(&sums).unwrap().iter().all(|v| *v == 0.0), "a first step adds nothing");
    let (curvature, factor, prior) = (matrix(rows, cols, 11, 0.5, 1.0), matrix(rows, cols, 9, 2.0, 0.0), matrix(rows, cols, 10, 0.5, 0.0));
    fit.curvature_innovation((Some(&up(&factor)), None), (&up(&curvature), &log_sd), &mut innovation, &groups, &mut sums, (scale, gain, false)).unwrap();
    // The prior curvature enters a later step's draws as the first's did.
    fit.curvature_innovation((None, Some(&up(&prior))), (&up(&curvature), &log_sd), &mut innovation, &groups, &mut sums, (scale, gain, false)).unwrap();
    (fit.download(&innovation).unwrap(), wide.download(&sums).unwrap())
}

#[test]
fn the_curvature_innovations_follow_their_formulas_on_every_backend() {
    let host = Device::host();
    let gain = 0.125;
    for axis in AXES {
        let c = case(axis);
        let (rows, cols) = c.mean.dim();
        let (curvature, factor, prior) = (matrix(rows, cols, 11, 0.5, 1.0), matrix(rows, cols, 9, 2.0, 0.0), matrix(rows, cols, 10, 0.5, 0.0));
        let scale = c.step.factor_scale;
        let mut expected_sums = Array2::<f64>::zeros((c.count - 1, 3));
        let mut expected = Array2::<f64>::zeros((rows, cols));
        for (i, &g) in c.groups.iter().enumerate() {
            let at = (i / cols, i % cols);
            let g = g as usize;
            if g >= c.count - 1 || c.log_sd[at] == f64::NEG_INFINITY {
                continue;
            }
            let first = scale * c.factor[at] * c.factor[at] + c.prior[at] - c.moments[1][at];
            let second = scale * factor[at] * factor[at] - curvature[at];
            let third = prior[at] - curvature[at];
            for (before, after) in [(first, second), (second, third)] {
                let change = after - (1.0 - gain) * before;
                expected_sums[(g, 0)] += change * before;
                expected_sums[(g, 1)] += change * change;
                expected_sums[(g, 2)] += curvature[at] * curvature[at];
            }
            expected[at] = third;
        }
        let (innovation, sums) = innovations(&host, &host, &c, gain);
        close("host innovations", &innovation, &expected, 1e-15);
        close("host innovation sums", &sums, &expected_sums, 1e-12);
        let mut devices: Vec<(Device, Device)> = Vec::new();
        if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
            devices.push((wide.with_storage(Storage::F32).expect("CUDA holds f32"), wide));
        }
        if let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault") {
            devices.push((metal.clone(), metal));
        }
        // Each term is a chain within `CHAIN` of its magnitude; a group sums at most 260 of them
        // (the columns case's pairs of 130 rows), twice.
        for (fit, wide) in &devices {
            let (found, found_sums) = innovations(fit, wide, &c, gain);
            close(&format!("{} innovations", fit.name()), &found, &innovation, CHAIN);
            close(&format!("{} innovation sums", fit.name()), &found_sums, &sums, 520.0 * (CHAIN + 520.0 * U));
        }
    }
}

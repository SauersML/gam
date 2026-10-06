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
/// stream)` of its sample, its gradient and Gauss–Newton factor, and its step.
struct Case {
    mean: Array2<f64>,
    log_sd: Array2<f64>,
    moments: [Array2<f64>; 3],
    sample: (u64, u64),
    gradient: Array2<f64>,
    factor: Array2<f64>,
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
    // The gradient's momentum, a positive curvature estimate and the gradient's second moment.
    let moments = [matrix(rows, cols, 3, 0.1, 0.0), matrix(rows, cols, 4, 0.5, 1.0), matrix(rows, cols, 5, 0.01, 0.02)];
    let step = PosteriorStep { gradient_scale: 1.5, factor_scale: 0.25, tokens: 50.0, beta1: 0.9, beta2: 0.999, weights: PosteriorStep::constant_weights(0.9, 7) };
    let (gradient, factor) = (matrix(rows, cols, 7, 3.0, 0.0), matrix(rows, cols, 8, 2.0, 0.0));
    Case { mean, log_sd, moments, sample: (0x1234_5678_9abc_def0, 42), gradient, factor, groups, count: 8, step }
}

/// The formulas, entry by entry: the sample, the stepped posterior and moments, and the group sums
/// `(n, Σ μ² + σ², Σ 2s)` before and after the step, from the sums before.
fn reference(c: &Case) -> (Array2<f64>, Array2<f64>, Array2<f64>, [Array2<f64>; 3], Vec<[f64; 3]>, Vec<[f64; 3]>) {
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
    let t = 7;
    let c1 = 1.0 - b1.powi(t);
    // The momentum's effective number of gradients, and the factor from the moments' spread to its
    // variance.
    let effective = (1.0 + b1) * c1 * c1 / ((1.0 - b1) * (1.0 - b1.powi(2 * t)));
    let spread = 1.0 / (effective - 1.0);
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
        let gr = c.step.gradient_scale * c.gradient[at];
        let delta = 1.0 / (n * variance[*g as usize]);
        let momentum = b1 * moments[0][at] + (1.0 - b1) * gr;
        let power = b1 * moments[2][at] + (1.0 - b1) * gr * gr;
        let (m, noise) = (momentum / c1, (power / c1 - (momentum / c1).powi(2)).max(0.0) * spread);
        let full = m + delta * mu;
        let signal = if full * full > noise { full - noise / full } else { 0.0 };
        let estimate = c.step.factor_scale * c.factor[at] * c.factor[at];
        let (h, d) = (moments[1][at], estimate - moments[1][at]);
        let curvature = h + (1.0 - b2) * d;
        moments[0][at] = momentum;
        moments[1][at] = curvature;
        moments[2][at] = power;
        mean[at] = mu - signal / (curvature + delta);
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
    let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
    let mut theta = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
    fit.reparameterize(&mut theta, (&mean, &log_sd), c.sample).unwrap();
    let mut sums = wide.zeros(c.count, 3).unwrap();
    fit.group_moments((&mean, &log_sd), &groups, &mut sums).unwrap();
    let before = wide.download(&sums).unwrap();
    let (mut variance, mut divergence) = (wide.zeros(c.count, 1).unwrap(), wide.zeros(c.count, 1).unwrap());
    wide.group_divergence(&mut sums, &mut variance, &mut divergence).unwrap();
    assert!(wide.download(&sums).unwrap().iter().all(|v| *v == 0.0), "the sums are zeroed");
    let (gradient, factor) = (up(fit, &c.gradient), up(fit, &c.factor));
    let [momentum, curvature, power] = &mut moments[..] else { unreachable!() };
    let (mut direction, mut terms) = (fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap(), wide.zeros(c.count, 3).unwrap());
    fit.posterior_ivon((&mean, &mut log_sd), [momentum, curvature, power], (&gradient, &factor), (&groups, &variance), (&mut direction, &mut terms), &c.step).unwrap();
    // The full step, and an average from zero with weight 1 (the stepped mean itself), whose
    // moments are the sums after the step.
    let mut average = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
    fit.posterior_finish((&mut mean, &direction, 1.0), (&mut average, 1.0), &log_sd, &groups, &mut sums).unwrap();
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

fn host_formulas(c: &Case) {
    let host = Device::host();
    let (theta, mean, log_sd, moments, before, after, divergence) = run(&host, &host, c);
    let (t, m, s, mo, b, a) = reference(c);
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
    assert!(theta.iter().zip(&c.groups).all(|(v, g)| *g != 4 || *v == 0.0), "a removed entry samples zero");
}

fn against_host(fit: &Device, wide: &Device) {
    for axis in AXES {
        against_host_on(fit, wide, &case(axis));
    }
}

fn against_host_on(fit: &Device, wide: &Device, c: &Case) {
    let host = Device::host();
    let (theta, mean, log_sd, moments, before, after, divergence) = run(fit, wide, c);
    let (t, m, s, mo, b, a, d) = run(&host, &host, c);
    // No device sum is atomic (`GroupMap`): a second run is the first bit for bit.
    let first = (theta.clone(), mean.clone(), log_sd.clone(), moments.clone(), before.clone(), after.clone(), divergence.clone());
    assert!(run(fit, wide, c) == first, "the device's results repeat bit for bit");
    // A group sums at most `n` entries; the step reads its variance (such a sum over its count),
    // and a second moment squares a gradient that carries the variance's error.
    let n = (0..c.count as u32).map(|g| c.groups.iter().filter(|h| **h == g).count()).max().unwrap() as f64;
    let sum_band = CHAIN + n * U / (1.0 - n * U);
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
    let (mut momentum, mut curvature, mut power) = (up(Array2::zeros((1, R))), up(Array2::from_elem((1, R), h)), up(Array2::zeros((1, R))));
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
        let step = PosteriorStep { gradient_scale: 1.0, factor_scale: 1.0, tokens, beta1: 0.9, beta2: 1.0 - 1.0 / 64.0, weights: PosteriorStep::constant_weights(0.9, t) };
        let (mut direction, mut sums) = (fit.zeros(1, R).unwrap(), wide.zeros(1, 3).unwrap());
        fit.posterior_ivon((&mean, &mut log_sd), [&mut momentum, &mut curvature, &mut power], (&up(gradient), &factor), (&groups, &variance), (&mut direction, &mut sums), &step).unwrap();
        mean = up(&mu - &(fit.download(&direction).unwrap() * 0.1));
    }
    (target, averages)
}

/// The filtered step's fixed point is `F`'s stationary point: averaged over the last 2000 of 4000
/// steps and over 64 coordinates, `μ` is within six standard errors (of that average over the
/// coordinates) of `μ* = −6.6`, and the standard error is small enough to tell `μ*` from the
/// `−4.1` at which filtering the data momentum alone and adding `δ μ` exactly settled.
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
    for axis in AXES {
        bfloat16_momentum(&fit, &half, &wide, case(axis));
    }
}

fn bfloat16_momentum(fit: &Device, half: &Device, wide: &Device, mut c: Case) {
    // Inputs a bfloat16 holds exactly, so both steps start from the same values.
    for m in &mut c.moments {
        m.mapv_inplace(bf16);
    }
    c.gradient.mapv_inplace(bf16);
    c.factor.mapv_inplace(bf16);
    let (theta, mean, log_sd, moments, _, after, _) = run(fit, wide, &c);
    let up = |d: &Device, m: &Array2<f64>| d.upload(m.view()).expect("upload");
    let down = |t: &Tensor| fit.download(t).expect("download");
    for gradient_storage in [fit, half] {
        let (mut m, mut s) = (up(fit, &c.mean), up(fit, &c.log_sd));
        let (mut momentum, mut curvature, mut power) = (up(half, &c.moments[0]), up(fit, &c.moments[1]), up(fit, &c.moments[2]));
        let groups = fit.group_map(&c.groups, c.mean.dim()).expect("groups");
        let mut sample = half.zeros(c.mean.nrows(), c.mean.ncols()).expect("sample");
        fit.reparameterize(&mut sample, (&m, &s), c.sample).expect("bfloat16 sample");
        let mut sums = wide.zeros(c.count, 3).expect("sums");
        fit.group_moments((&m, &s), &groups, &mut sums).expect("group moments");
        let (mut variance, mut divergence) = (wide.zeros(c.count, 1).expect("variance"), wide.zeros(c.count, 1).expect("divergence"));
        wide.group_divergence(&mut sums, &mut variance, &mut divergence).expect("group divergence");
        let (gradient, factor) = (up(gradient_storage, &c.gradient), up(gradient_storage, &c.factor));
        let (mut direction, mut terms) = (fit.zeros(c.mean.nrows(), c.mean.ncols()).expect("direction"), wide.zeros(c.count, 3).expect("terms"));
        fit.posterior_ivon((&m, &mut s), [&mut momentum, &mut curvature, &mut power], (&gradient, &factor), (&groups, &variance), (&mut direction, &mut terms), &c.step).expect("bfloat16 step");
        let mut average = fit.zeros(c.mean.nrows(), c.mean.ncols()).expect("average");
        fit.posterior_finish((&mut m, &direction, 1.0), (&mut average, 1.0), &s, &groups, &mut sums).expect("finish");
        close("bfloat16 sample", &down(&sample), &theta.mapv(bf16), 2f64.powi(-8));
        close("mean", &down(&m), &mean, CHAIN);
        close("log sd", &down(&s), &log_sd, CHAIN);
        close("bfloat16 momentum", &down(&momentum), &moments[0].mapv(bf16), 2f64.powi(-8));
        close("curvature", &down(&curvature), &moments[1], CHAIN);
        close("gradient second moment", &down(&power), &moments[2], CHAIN);
        close("sums after", &wide.download(&sums).expect("sums"), &after, CHAIN);
        // A bfloat16 copy widens back to the values it holds.
        assert_eq!(down(&fit.convert(&momentum).expect("widen")), down(&momentum));
    }
}

/// `group_curvature` and `group_code_length` of the case on `fit` (sums on `wide`): the curvature
/// rows of the case's factor at its posterior, and in row 1 of 2 the code length of its groups
/// (groups 4, removed, and 6 out of the explanation; each group's starting variance `2^-k` times
/// its variance on the host, so that the scale's exponent is the integer `k` and not near a
/// rounding boundary).
fn removal_sums(fit: &Device, wide: &Device, initial: &[f64]) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let c = case(GroupAxis::Entries);
    let up = |a: &Array2<f64>| fit.upload(a.view()).unwrap();
    let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
    let mut curvature = wide.zeros(c.count, 3).unwrap();
    fit.group_curvature((&up(&c.factor), &up(&c.mean), &up(&c.log_sd)), &groups, &mut curvature).unwrap();
    let mut moments = wide.zeros(c.count, 3).unwrap();
    fit.group_moments((&up(&c.mean), &up(&c.log_sd)), &groups, &mut moments).unwrap();
    let (mut variance, mut divergence) = (wide.zeros(c.count, 1).unwrap(), wide.zeros(c.count, 1).unwrap());
    wide.group_divergence(&mut moments, &mut variance, &mut divergence).unwrap();
    let column = |v: Vec<f64>| wide.upload_vec(v.len(), 1, v).unwrap();
    let weight = column((0..c.count).map(|g| if g == 4 || g == 6 { 0.0 } else { 1.0 }).collect());
    let constant = column((0..c.count).map(|g| 0.5 * (g as f64 + 2.0).ln()).collect());
    let mut lengths = wide.zeros(2, 3).unwrap();
    wide.group_code_length((&divergence, &variance), (&weight, &constant, &column(initial.to_vec())), &mut lengths, 1).unwrap();
    (wide.download(&curvature).unwrap(), wide.download(&lengths).unwrap(), wide.download(&variance).unwrap())
}

fn removal_sums_against_host(fit: &Device, wide: &Device) {
    let host = Device::host();
    let count = case(GroupAxis::Entries).count;
    let (_, _, variance) = removal_sums(&host, &host, &vec![1.0; count]);
    let initial: Vec<f64> = (0..count).map(|g| if variance[(g, 0)] > 0.0 { variance[(g, 0)] * 2f64.powi(-(g as i32 % 5) + 2) } else { 1.0 }).collect();
    let (curvature, lengths, _) = removal_sums(fit, wide, &initial);
    let (expected_curvature, expected_lengths, _) = removal_sums(&host, &host, &initial);
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
/// same device, for each axis's case, the momentum in `momenta`'s storage and the gradient and the
/// factor in `gradients`': `d · d`, `u · d` and `Σ h' d²` each bit for bit its group curvature's
/// column 1 (the same entries in the same order).
fn step_terms_match_the_curvatures(fit: &Device, (momenta, gradients): (&Device, &Device), wide: &Device) {
    for axis in AXES {
        let c = case(axis);
        let up = |d: &Device, a: &Array2<f64>| d.upload(a.view()).unwrap();
        let groups = fit.group_map(&c.groups, c.mean.dim()).unwrap();
        let (mean, mut log_sd) = (up(fit, &c.mean), up(fit, &c.log_sd));
        let (mut momentum, mut curvature, mut power) = (up(momenta, &c.moments[0]), up(fit, &c.moments[1]), up(fit, &c.moments[2]));
        let (gradient, factor) = (up(gradients, &c.gradient), up(gradients, &c.factor));
        let variance = wide.upload_vec(c.count, 1, (0..c.count).map(|g| 0.01 * (g + 1) as f64).collect()).unwrap();
        let (mut direction, mut sums) = (fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap(), wide.zeros(c.count, 3).unwrap());
        fit.posterior_ivon((&mean, &mut log_sd), [&mut momentum, &mut curvature, &mut power], (&gradient, &factor), (&groups, &variance), (&mut direction, &mut sums), &c.step).unwrap();
        let mut weighted = fit.zeros(c.mean.nrows(), c.mean.ncols()).unwrap();
        fit.hadamard(&mut weighted, &curvature, &direction, false).unwrap();
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

#[test]
fn a_steps_terms_and_finish_are_the_operations_they_replace() {
    let host = Device::host();
    step_terms_match_the_curvatures(&host, (&host, &host), &host);
    finish_matches_its_parts(&host, &host);
    if let Some(wide) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        let narrow = wide.with_storage(Storage::F32).expect("CUDA holds f32");
        let half = wide.with_storage(Storage::Bf16).expect("CUDA holds bfloat16");
        step_terms_match_the_curvatures(&narrow, (&narrow, &narrow), &wide);
        step_terms_match_the_curvatures(&narrow, (&half, &narrow), &wide);
        step_terms_match_the_curvatures(&narrow, (&half, &half), &wide);
        step_terms_match_the_curvatures(&wide, (&wide, &wide), &wide);
        finish_matches_its_parts(&narrow, &wide);
        finish_matches_its_parts(&wide, &wide);
    }
    // On Linux the single-precision device is CUDA's f32 storage, whose sums are no float64 tensor.
    if cfg!(target_os = "macos")
        && let Some(metal) = Device::single_precision(GpuPolicy::Auto).expect("a probe that does not fault")
    {
        step_terms_match_the_curvatures(&metal, (&metal, &metal), &metal);
        finish_matches_its_parts(&metal, &metal);
    }
}

/// `move_toward` against the copy and `axpy`s it replaces, bit for bit.
fn moves_match_their_compositions(d: &Device) {
    let (x, y) = (d.upload(matrix(7, 33, 11, 2.0, 0.5).view()).unwrap(), d.upload(matrix(7, 33, 12, 1.5, -0.25).view()).unwrap());
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

/// CUDA's reduction over groups of one column each (eight adjacent columns per block) against the
/// same entries transposed and grouped by rows, which the per-segment reduction takes in the same
/// order: every group sum, and every entry the kernels write, bit for bit. 600 rows (two full
/// passes of a block and part of a third) and 22 columns in a shuffled group order (full tiles of
/// columns not adjacent in group order, and part of one), groups 18 to 21 beyond the count, and
/// removed entries; the momentum in `momenta`'s storage.
fn single_columns_match_their_transposes(fit: &Device, momenta: &Device, wide: &Device) {
    let (rows, cols, count) = (600, 22, 18);
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
    let (momentum, curvature, power) = (matrix(rows, cols, 3, 0.1, 0.0), matrix(rows, cols, 4, 0.5, 1.0), matrix(rows, cols, 5, 0.01, 0.02));
    let (gradient, factor) = (matrix(rows, cols, 7, 3.0, 0.0), matrix(rows, cols, 8, 2.0, 0.0));
    let both = |d: &Device, a: &Array2<f64>| (d.upload(a.view()).unwrap(), d.upload(a.t().as_standard_layout().view()).unwrap());
    let back = |t: &Tensor| fit.download(t).unwrap().t().as_standard_layout().to_owned();
    let pair = |what: &str, (a, b): (&Tensor, &Tensor), d: &Device| assert_eq!(d.download(a).unwrap(), d.download(b).unwrap(), "{} {what}", fit.name());
    let ((m, mt), (s, st), (u, ut)) = (both(fit, &mean), both(fit, &log_sd), both(fit, &factor));
    let sums = |columns: usize| (wide.zeros(count, columns).unwrap(), wide.zeros(count, columns).unwrap());
    let (mut x, mut y) = sums(3);
    fit.group_moments((&m, &s), &map, &mut x).unwrap();
    fit.group_moments((&mt, &st), &map_t, &mut y).unwrap();
    pair("moments", (&x, &y), wide);
    let (mut x, mut y) = sums(3);
    fit.group_curvature((&u, &m, &s), &map, &mut x).unwrap();
    fit.group_curvature((&ut, &mt, &st), &map_t, &mut y).unwrap();
    pair("curvature", (&x, &y), wide);
    let (g, gt) = both(fit, &gradient);
    let variance = wide.upload_vec(count, 1, (0..count).map(|g| 0.01 * (g + 1) as f64).collect()).unwrap();
    let step = case(GroupAxis::Rows).step;
    let (mut s, mut st) = both(fit, &log_sd);
    let ((mut p, mut pt), (mut c, mut ct), (mut q, mut qt)) = (both(momenta, &momentum), both(fit, &curvature), both(fit, &power));
    let (mut d, mut dt) = (fit.zeros(rows, cols).unwrap(), fit.zeros(cols, rows).unwrap());
    let (mut x, mut y) = sums(3);
    fit.posterior_ivon((&m, &mut s), [&mut p, &mut c, &mut q], (&g, &u), (&map, &variance), (&mut d, &mut x), &step).unwrap();
    fit.posterior_ivon((&mt, &mut st), [&mut pt, &mut ct, &mut qt], (&gt, &ut), (&map_t, &variance), (&mut dt, &mut y), &step).unwrap();
    pair("step terms", (&x, &y), wide);
    for (what, a, b) in [("direction", &d, &dt), ("log sd", &s, &st), ("momentum", &p, &pt), ("curvature", &c, &ct), ("power", &q, &qt)] {
        assert_eq!(fit.download(a).unwrap(), back(b), "{} stepped {what}", fit.name());
    }
    let ((mut m, mut mt), (mut a, mut at)) = (both(fit, &mean), both(fit, &momentum));
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
    let half = wide.with_storage(Storage::Bf16).expect("CUDA holds bfloat16");
    single_columns_match_their_transposes(&narrow, &narrow, &wide);
    single_columns_match_their_transposes(&narrow, &half, &wide);
    single_columns_match_their_transposes(&wide, &wide, &wide);
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
    let (_, _, variance) = removal_sums(&host, &host, &vec![1.0; c.count]);
    let initial: Vec<f64> = (0..c.count).map(|g| if variance[(g, 0)] > 0.0 { variance[(g, 0)] * 2f64.powi(3) } else { 1.0 }).collect();
    let (curvature, lengths, _) = removal_sums(&host, &host, &initial);
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
    // codeword of 2 + 2 x 1 + 1 = 5 bits.
    let mut moments = host.zeros(c.count, 3).unwrap();
    host.group_moments((&host.upload(c.mean.view()).unwrap(), &host.upload(c.log_sd.view()).unwrap()), &host.group_map(&c.groups, c.mean.dim()).unwrap(), &mut moments).unwrap();
    let (mut v, mut d) = (host.zeros(c.count, 1).unwrap(), host.zeros(c.count, 1).unwrap());
    host.group_divergence(&mut moments, &mut v, &mut d).unwrap();
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

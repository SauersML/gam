//! gam#3047 / gam#3030: the stacked first-stage sandwich of the conditional
//! latent calibration, pinned against the estimating system itself and audited
//! for calibration of the interval it feeds.
//!
//! Two arms:
//!
//!   - a pin of the inverse bread: perturb the score `z → z + tδ`, refit, and
//!     compare the Richardson-extrapolated central difference of the refit `θ̂₁`
//!     with `J⁻¹·∂ψ/∂t`. The mean stage is linear in `z`; the log-linear
//!     variance stage (gam#4019) is the root of a smooth score, polished here to
//!     that root, so the difference carries an `O(h⁴)` truncation and roundoff
//!     only; a reversed cross-stage block (the gam#3047 sign) misses by the
//!     whole of `2K·g_m`.
//!   - a Monte Carlo coverage audit of the second-stage slope interval the
//!     Murphy–Topel correction produces, on both variance stages, which fails on
//!     over-coverage exactly as on under-coverage: the reversed sign made the
//!     fired stage conservative, and the dropped constant stage (gam#3030) made
//!     the homoskedastic stage anti-conservative.

use super::{
    AUTO_Z_CONDITIONAL_RIDGE_REL, LatentZConditionalCalibration, build_intercept_basis,
    fit_conditional_latent_calibration, kolmogorov_upper_quantile,
    stacked_first_stage_inverse_bread,
};
use gam_math::probability::normal_cdf;
use gam_math::special::gauss_legendre;
use gam_test_support::calibration::{
    COVERAGE_FALSE_POSITIVE_RATE, COVERAGE_NOMINAL_LEVELS, COVERAGE_REPLICATIONS,
    CalibrationRng, CoverageClass, audit_coverage, standard_normal_quantile,
};
use ndarray::{Array1, Array2, s};

/// `E a² = 1/3` of `a ~ U(−1, 1)`.
const A_SECOND_MOMENT: f64 = 1.0 / 3.0;
/// `z = QUAD·a² + LIN·a + e`: the quadratic term makes the linear conditional
/// mean MISSPECIFIED, so the mean residual is correlated with `a²` and the
/// cross-stage bread `M_vm = −2 Σ w (û/v) B Aᵀ` is O(n) rather than ≈ 0.
const QUAD: f64 = 0.8;
const LIN: f64 = 0.3;
/// `e = ξ·(SCALE0 + SCALE1·a)` with `ξ = (U² − E U²)/sd(U²)`, `U ~ U(0, 1)`:
/// skewed (`E ξ³ > 0`, which is what feeds the meat's `Ω_mv ∝ E[û³]`) and
/// heteroskedastic, the regime both channels of the stacked sandwich serve.
/// `ξ` is BOUNDED on purpose. The slope interval's width is dominated by
/// `Var(v̂) = Var(u²)/n`, whose HC0 estimate is a fourth-moment average with
/// relative error `√(E u⁸)/(Var(u²)·√n)`; an exponential `ξ` (`E ξ⁸ ≈ 1.5e4`)
/// leaves that at 24% at `n = 2000`, and the audit then grades the
/// variance ESTIMATOR's slow convergence rather than the propagation this
/// file pins. Measured with `(Exp(1) − 1)`: 95% coverage 0.9335 at
/// `n = 32000` with the corrected propagation, whose empirical `Var(β̂)` it
/// matched to within the Monte Carlo error of `Var(v̂)`.
const SCALE0: f64 = 0.5;
const SCALE1: f64 = 0.2;
/// Moments of `U²`, `U ~ U(0, 1)`: `E U² = 1/3`, `E U⁴ = 1/5`.
const U2_MEAN: f64 = 1.0 / 3.0;
const U2_VAR: f64 = 1.0 / 5.0 - 1.0 / 9.0;
/// Second-stage model `y = BETA·ζ* + SIGMA·N(0, 1)`.
const BETA: f64 = 1.0;
const SIGMA: f64 = 0.5;

/// One draw of the first-stage design: the marginal block `a` (n × 1) and `z`.
fn draw_design(rng: &mut CalibrationRng, n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut a_block = Array2::<f64>::zeros((n, 1));
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let a = 2.0 * rng.uniform_open01() - 1.0;
        let u = rng.uniform_open01();
        let e = (u * u - U2_MEAN) / U2_VAR.sqrt() * (SCALE0 + SCALE1 * a);
        a_block[[i, 0]] = a;
        z[i] = QUAD * a * a + LIN * a + e;
    }
    (a_block, z)
}

/// Population pseudo-truth of the first stage under [`draw_design`].
///
/// The linear projection of `z` on `[1, a]` is `(QUAD·E a², LIN)`. The mean
/// residual is `u = QUAD(a² − E a²) + e`, so
/// `s(a) = E[u² | a] = QUAD²(a² − E a²)² + (SCALE0 + SCALE1·a)²`. The constant
/// stage's pseudo-truth is `log E u²`. The fired stage's is the root `γ*` of the
/// population log-linear score `E[B (s(a)·exp(−Bᵀγ) − 1)] = 0`, `B = [1, a]`:
/// `s` is not log-linear, so `γ*` is the Kullback-Leibler projection rather than
/// a closed form. Its integrand is a quartic times an exponential, which a
/// 64-point Gauss-Legendre rule integrates to roundoff, and the population
/// information is positive definite, so Newton converges from `(log E u², 0)`.
/// Returns `(mean, fired log-variance γ*, constant log-variance)`.
fn pseudo_truth() -> ([f64; 2], [f64; 2], f64) {
    let mean = [QUAD * A_SECOND_MOMENT, LIN];
    let (nodes, rule_weights) = gauss_legendre(64);
    // `a ~ U(−1, 1)` has density 1/2 on the rule's interval.
    let conditional_second_moment = |a: f64| {
        let centred = a * a - A_SECOND_MOMENT;
        QUAD * QUAD * centred * centred + (SCALE0 + SCALE1 * a).powi(2)
    };
    let second_moment = nodes
        .iter()
        .zip(rule_weights.iter())
        .map(|(&a, &w)| 0.5 * w * conditional_second_moment(a))
        .sum::<f64>();
    let rows: Vec<(f64, f64, f64)> = nodes
        .iter()
        .zip(rule_weights.iter())
        .map(|(&a, &w)| (a, 0.5 * w, conditional_second_moment(a)))
        .collect();
    let gamma = newton_log_linear_root([second_moment.ln(), 0.0], |gamma| {
        let mut score = [0.0; 2];
        let mut information = [[0.0; 2]; 2];
        for &(a, w, s) in &rows {
            let ratio = s * (-(gamma[0] + gamma[1] * a)).exp();
            let basis = [1.0, a];
            for j in 0..2 {
                score[j] += w * basis[j] * (ratio - 1.0);
                for k in 0..2 {
                    information[j][k] += w * ratio * basis[j] * basis[k];
                }
            }
        }
        (score, information)
    });
    (mean, gamma, second_moment.ln())
}

/// Newton's method on a two-parameter log-linear variance score from a start
/// inside its quadratic basin, run until a step stops contracting -- the point at
/// which roundoff, not the iteration, sets the error. `system(γ)` returns the
/// score and the (positive-definite) information `−∂score/∂γ`.
fn newton_log_linear_root(
    start: [f64; 2],
    system: impl Fn([f64; 2]) -> ([f64; 2], [[f64; 2]; 2]),
) -> [f64; 2] {
    let mut gamma = start;
    let mut previous = f64::INFINITY;
    loop {
        let (score, info) = system(gamma);
        let det = info[0][0] * info[1][1] - info[0][1] * info[1][0];
        let step = [
            (info[1][1] * score[0] - info[0][1] * score[1]) / det,
            (info[0][0] * score[1] - info[1][0] * score[0]) / det,
        ];
        let size = step[0].hypot(step[1]);
        if !(size < previous) {
            return gamma;
        }
        gamma = [gamma[0] + step[0], gamma[1] + step[1]];
        previous = size;
    }
}

/// The refit's `θ₁ = (mean_coeffs, variance stage)`, in the coordinates of
/// `theta1_cov`: the constant stage is `log homoskedastic_var`.
fn theta1(cal: &LatentZConditionalCalibration) -> Vec<f64> {
    let mut theta = cal.mean_coeffs.clone();
    if cal.log_var_coeffs.is_empty() {
        theta.push(cal.homoskedastic_var.ln());
    } else {
        theta.extend_from_slice(&cal.log_var_coeffs);
    }
    theta
}

/// [`theta1`] with the fired stage's `γ̂` polished to the exact root of its
/// score `Σ w B (û²/v − 1)` at the refit's own mean residuals `û`. The
/// production fit stops on its likelihood's rounding band, which leaves `γ̂`
/// within `O(√ε)` of the root -- statistically nothing, but a difference
/// quotient divides that by the step. Newton from there converges quadratically,
/// so the pin compares `J⁻¹` with the derivative of the root itself, which is
/// what `J⁻¹` claims to be.
fn theta1_at_root(
    cal: &LatentZConditionalCalibration,
    z: &Array1<f64>,
    weights: &Array1<f64>,
    a_block: &Array2<f64>,
) -> Vec<f64> {
    let mut theta = theta1(cal);
    if cal.log_var_coeffs.is_empty() {
        return theta;
    }
    let rows: Vec<(f64, f64, f64)> = (0..z.len())
        .map(|i| {
            let a = a_block[[i, 0]];
            let u = z[i] - LatentZConditionalCalibration::affine(&cal.mean_coeffs, a_block.row(i));
            (a, weights[i], u * u)
        })
        .collect();
    let gamma = newton_log_linear_root(
        [cal.log_var_coeffs[0], cal.log_var_coeffs[1]],
        |gamma| {
            let mut score = [0.0; 2];
            let mut information = [[0.0; 2]; 2];
            for &(a, w, u2) in &rows {
                let ratio = u2 * (-(gamma[0] + gamma[1] * a)).exp();
                let basis = [1.0, a];
                for j in 0..2 {
                    score[j] += w * basis[j] * (ratio - 1.0);
                    for k in 0..2 {
                        information[j][k] += w * ratio * basis[j] * basis[k];
                    }
                }
            }
            (score, information)
        },
    );
    let p = cal.mean_coeffs.len();
    theta[p..].copy_from_slice(&gamma);
    theta
}

/// Exact pin of the inverse bread on one variance stage. Returns the relative
/// error of the production `J⁻¹` and of the gam#3047 reversed-`K` mutant.
fn inverse_bread_pin(fit_variance: bool) -> (f64, f64) {
    let n = 400;
    let mut rng = CalibrationRng::new(0x3047_0001 + u64::from(fit_variance));
    let (a_block, z) = draw_design(&mut rng, n);
    let weights: Array1<f64> = (0..n).map(|_| 0.5 + rng.uniform_open01()).collect();
    let delta: Array1<f64> = (0..n).map(|_| rng.standard_normal()).collect();

    let cal = fit_conditional_latent_calibration(&z, &weights, a_block.view(), fit_variance)
        .expect("base fit");
    assert_eq!(cal.log_var_coeffs.is_empty(), !fit_variance);
    assert_eq!(cal.theta1_dim(), theta1(&cal).len());
    assert_eq!(cal.theta1_cov.dim(), (cal.theta1_dim(), cal.theta1_dim()));

    // The system at the base fit, built the way the calibration builds it.
    let basis = build_intercept_basis(a_block.view());
    let p = basis.ncols();
    let mut mean_normal = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let row = basis.row(i);
        for j in 0..p {
            for k in 0..p {
                mean_normal[[j, k]] += weights[i] * row[j] * row[k];
            }
        }
    }
    for j in 0..p {
        mean_normal[[j, j]] *= 1.0 + AUTO_Z_CONDITIONAL_RIDGE_REL;
    }
    let mean_residuals: Vec<f64> = (0..n)
        .map(|i| z[i] - LatentZConditionalCalibration::affine(&cal.mean_coeffs, a_block.row(i)))
        .collect();
    // The system at the base root: the polished `γ̂` and its fitted `v_i`.
    let base_theta = theta1_at_root(&cal, &z, &weights, &a_block);
    let (var_basis, var_fitted): (Array2<f64>, Vec<f64>) = if fit_variance {
        let gamma = &base_theta[p..];
        let fitted = (0..n)
            .map(|i| (gamma[0] + gamma[1] * a_block[[i, 0]]).exp())
            .collect();
        (basis.clone(), fitted)
    } else {
        (Array2::<f64>::ones((n, 1)), vec![cal.homoskedastic_var; n])
    };
    let q = var_basis.ncols();
    let j_inv = stacked_first_stage_inverse_bread(
        basis.view(),
        var_basis.view(),
        weights.view(),
        &mean_residuals,
        &var_fitted,
        &mean_normal,
    )
    .expect("inverse bread");

    // ∂ψ/∂t at t = 0: ψ^m = Σ w A û, ψ^v = Σ w B (û²/v − 1), û = z + tδ − Aᵀβ_m.
    let mut g = Array1::<f64>::zeros(p + q);
    for i in 0..n {
        for j in 0..p {
            g[j] += weights[i] * basis[[i, j]] * delta[i];
        }
        for j in 0..q {
            g[p + j] += 2.0 * weights[i] * mean_residuals[i] / var_fitted[i]
                * var_basis[[i, j]]
                * delta[i];
        }
    }
    let predicted = j_inv.dot(&g);

    // The gam#3047 mutant: the reversed sign of the cross-stage block `K`.
    let mut mutant = j_inv.clone();
    mutant.slice_mut(s![p.., ..p]).mapv_inplace(|entry| -entry);
    let mutant_predicted = mutant.dot(&g);

    // The refit root is smooth in t, so the central difference `D(h)` carries
    // an even series `D(h) = θ' + c₂h² + c₄h⁴ + …`, and the Richardson
    // combination `(4·D(h/2) − D(h))/3` cancels the `h²` term.
    let step = 1.0e-3;
    let refit = |t: f64| {
        let zt = &z + &(&delta * t);
        let cal_t = fit_conditional_latent_calibration(&zt, &weights, a_block.view(), fit_variance)
            .expect("perturbed fit");
        theta1_at_root(&cal_t, &zt, &weights, &a_block)
    };
    let central = |h: f64| -> Vec<f64> {
        let plus = refit(h);
        let minus = refit(-h);
        plus.iter()
            .zip(minus.iter())
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect()
    };
    let coarse = central(step);
    let fine = central(0.5 * step);
    let observed: Vec<f64> = fine
        .iter()
        .zip(coarse.iter())
        .map(|(f, c)| (4.0 * f - c) / 3.0)
        .collect();
    let scale = observed.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rel_err = |pred: &Array1<f64>| {
        observed
            .iter()
            .zip(pred.iter())
            .map(|(o, p)| (o - p) * (o - p))
            .sum::<f64>()
            .sqrt()
            / scale
    };
    (rel_err(&predicted), rel_err(&mutant_predicted))
}

/// The error budget of the pin. The Richardson-extrapolated central difference
/// at `h = 1e-3` leaves an `O(h⁴) ≈ 1e-12` truncation of an O(1) smooth root and
/// `ε/h ≈ 2e-13` of cancellation error per coordinate, amplified by the
/// conditioning of the O(n) normal systems; `√ε` bounds that with room (a numpy
/// replica of this pin measured 3.8e-12 fired and 2.2e-12 constant), and a
/// reversed `K` misses by O(1).
fn pin_tolerance() -> f64 {
    f64::EPSILON.sqrt()
}

#[test]
fn inverse_bread_matches_the_refit_on_the_fired_variance_stage_3047() {
    let (err, mutant_err) = inverse_bread_pin(true);
    assert!(
        err <= pin_tolerance(),
        "gam#3047: J⁻¹·∂ψ/∂t must reproduce the exact refit sensitivity of θ̂₁; relative error \
         {err:.3e}"
    );
    assert!(
        mutant_err > 1.0e3 * pin_tolerance(),
        "gam#3047: the reversed cross-stage block must be distinguishable from the refit on a \
         misspecified mean; mutant relative error {mutant_err:.3e}"
    );
}

#[test]
fn inverse_bread_matches_the_refit_on_the_constant_variance_stage_3030() {
    let (err, _) = inverse_bread_pin(false);
    assert!(
        err <= pin_tolerance(),
        "gam#3030: the constant variance stage log v̂ = log(Σwû²/Σw) must be propagated by the \
         same inverse bread; relative error {err:.3e}"
    );
}

/// Per-replication sample size of the coverage audit: large enough that the
/// constant-stage share of the slope variance (`≈ (κ_u − 1)/(4n)` against the
/// naive `σ²/n`) is a visible fraction of the interval, so dropping it (gam#3030)
/// moves coverage by far more than the Wilson half-width.
const COVERAGE_SAMPLE_SIZE: usize = 2000;

/// Coverage audit of the corrected second-stage slope interval.
///
/// Each replication fits the first stage, standardises `ζ̂`, fits the
/// no-intercept Gaussian slope `β̂ = Σζ̂y/Σζ̂²` with known noise variance (naive
/// `Vb = σ²/Σζ̂²`), and adds the Murphy–Topel term the calibration returns. The
/// truth `β = BETA` is exact on the pseudo-true `ζ*`. Returns the hit counts at
/// [`COVERAGE_NOMINAL_LEVELS`] and the two-sided Wald p-values.
fn slope_coverage(fit_variance: bool, seed: u64) -> ([usize; 3], Vec<f64>) {
    let n = COVERAGE_SAMPLE_SIZE;
    let (mean_star, log_var_star, constant_log_var_star) = pseudo_truth();
    let z_crit: Vec<f64> = COVERAGE_NOMINAL_LEVELS
        .iter()
        .map(|&c| standard_normal_quantile(0.5 + 0.5 * c))
        .collect();
    let weights = Array1::<f64>::ones(n);
    let mut rng = CalibrationRng::new(seed);
    let mut hits = [0_usize; 3];
    let mut p_values = Vec::with_capacity(COVERAGE_REPLICATIONS);
    for _ in 0..COVERAGE_REPLICATIONS {
        let (a_block, z) = draw_design(&mut rng, n);
        let y: Array1<f64> = (0..n)
            .map(|i| {
                let a = a_block[[i, 0]];
                let v = if fit_variance {
                    (log_var_star[0] + log_var_star[1] * a).exp()
                } else {
                    constant_log_var_star.exp()
                };
                let zeta_star = (z[i] - mean_star[0] - mean_star[1] * a) / v.sqrt();
                BETA * zeta_star + SIGMA * rng.standard_normal()
            })
            .collect();
        let cal = fit_conditional_latent_calibration(&z, &weights, a_block.view(), fit_variance)
            .expect("first-stage fit");
        let zeta = cal.apply(z.view(), a_block.view()).expect("ζ̂");
        let sxx = zeta.dot(&zeta);
        let beta_hat = zeta.dot(&y) / sxx;
        let sigma2 = SIGMA * SIGMA;
        let vb = sigma2 / sxx;
        // s_i = ∂(score_β)_i/∂ζ_i of ℓ_i = −(y_i − βζ_i)²/(2σ²).
        let sensitivity = Array2::from_shape_fn((n, 1), |(i, _)| {
            (y[i] - 2.0 * beta_hat * zeta[i]) / sigma2
        });
        let term = cal
            .generated_regressor_correction(
                sensitivity.view(),
                z.view(),
                a_block.view(),
                Array2::from_elem((1, 1), vb).view(),
            )
            .expect("Murphy–Topel term");
        let se = (vb + term[[0, 0]]).sqrt();
        let t = (beta_hat - BETA).abs() / se;
        for (hit, &crit) in hits.iter_mut().zip(z_crit.iter()) {
            if t <= crit {
                *hit += 1;
            }
        }
        p_values.push(2.0 * (1.0 - normal_cdf(t)));
    }
    (hits, p_values)
}

/// Kolmogorov–Smirnov distance of the p-values from U(0, 1).
fn ks_distance(mut p: Vec<f64>) -> f64 {
    p.sort_by(f64::total_cmp);
    let m = p.len() as f64;
    p.iter()
        .enumerate()
        .map(|(i, &u)| {
            let lo = u - i as f64 / m;
            let hi = (i + 1) as f64 / m - u;
            lo.max(hi)
        })
        .fold(0.0, f64::max)
}

fn assert_slope_interval_calibrated(fit_variance: bool, seed: u64, issue: &str) {
    let (hits, p_values) = slope_coverage(fit_variance, seed);
    for (&hit, &nominal) in hits.iter().zip(COVERAGE_NOMINAL_LEVELS.iter()) {
        let verdict = audit_coverage(hit, COVERAGE_REPLICATIONS, nominal);
        assert_eq!(
            verdict.class,
            CoverageClass::Calibrated,
            "{issue}: the corrected slope interval must cover at its nominal level in BOTH \
             directions -- over-coverage is a defect exactly as under-coverage is; nominal \
             {nominal}, empirical {:.4}, Wilson CI [{:.4}, {:.4}]",
            verdict.empirical,
            verdict.ci_lo,
            verdict.ci_hi
        );
    }
    let m = p_values.len() as f64;
    let root_m = m.sqrt();
    let ks_crit =
        kolmogorov_upper_quantile(COVERAGE_FALSE_POSITIVE_RATE) / (root_m + 0.12 + 0.11 / root_m);
    let ks = ks_distance(p_values);
    assert!(
        ks <= ks_crit,
        "{issue}: the slope Wald p-values must be U(0, 1) across the whole range; KS distance \
         {ks:.4} exceeds the level-{COVERAGE_FALSE_POSITIVE_RATE} critical value {ks_crit:.4}"
    );
}

#[test]
fn slope_interval_is_calibrated_on_the_fired_variance_stage_3047() {
    assert_slope_interval_calibrated(true, 0x3047_c0de, "gam#3047");
}

#[test]
fn slope_interval_is_calibrated_on_the_constant_variance_stage_3030() {
    assert_slope_interval_calibrated(false, 0x3030_c0de, "gam#3030");
}

/// gam#4019: under multiplicative heteroskedasticity `Var(z | a) = e^a` -- convex
/// in `a`, where the retired linear variance stage went negative on the lower
/// sixth of the span and was floored at `1e-3·Var(z)`, leaving `sd(ζ)` at 7-11
/// there -- the log-linear stage must recover the variance law and standardise
/// `ζ` on every stretch of the span.
///
/// Design `a ~ N(0, 1)`, `z = 0.3·a + e^{a/2}·ε`, `ε ~ N(0, 1)`: the mean is
/// linear and the variance log-linear with `γ* = (0, 1)` exactly. Every bound
/// is a Wald bound at a Bonferroni split of [`COVERAGE_FALSE_POSITIVE_RATE`]
/// over the two coefficients and the bins:
///
///   - `|γ̂_k − γ*_k| ≤ c·se_k` with `se` from the calibration's own stacked
///     sandwich, so the covariance the Murphy–Topel correction consumes is
///     audited on the same fit;
///   - per bin, the root mean square of `ζ̂` (which is 1 for the true `ζ`)
///     within `c·(1/√(2 n_b) + ½·max se(η̂(a)))`: the first term is the
///     sampling SD of a Gaussian sample's RMS, the second the first-order
///     effect `½·Δη` of the fitted log-variance's error on the scale, bounded
///     at the bin's extreme rows. A numpy replica of this construction exceeded
///     the combined bound on 1 of 200 seeds (familywise level 0.01), at 1.0015×.
#[test]
fn log_linear_variance_stage_standardises_zeta_under_multiplicative_heteroskedasticity_4019() {
    let n = 20_000;
    let mut rng = CalibrationRng::new(0x4019_0001);
    let mut a_block = Array2::<f64>::zeros((n, 1));
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let a = rng.standard_normal();
        a_block[[i, 0]] = a;
        z[i] = 0.3 * a + (0.5 * a).exp() * rng.standard_normal();
    }
    let weights = Array1::<f64>::ones(n);
    let cal = fit_conditional_latent_calibration(&z, &weights, a_block.view(), true)
        .expect("fired log-linear variance stage");
    assert_eq!(cal.log_var_coeffs.len(), 2);
    let p = cal.mean_coeffs.len();
    let v_gamma = cal.theta1_cov.slice(s![p.., p..]).to_owned();

    let edges = [f64::NEG_INFINITY, -2.0, -1.2, -0.8, 0.0, 1.0, f64::INFINITY];
    let bins = edges.len() - 1;
    let crit =
        standard_normal_quantile(1.0 - COVERAGE_FALSE_POSITIVE_RATE / (2.0 * (bins + 2) as f64));

    let gamma_star = [0.0, 1.0];
    for k in 0..2 {
        let se = v_gamma[[k, k]].sqrt();
        let wald = (cal.log_var_coeffs[k] - gamma_star[k]).abs() / se;
        assert!(
            wald <= crit,
            "gam#4019: log-variance coefficient {k} = {:.4} must recover the log-linear law's \
             {} within {crit:.3} sandwich SEs ({se:.4}); Wald {wald:.3}",
            cal.log_var_coeffs[k],
            gamma_star[k]
        );
    }

    let zeta = cal.apply(z.view(), a_block.view()).expect("ζ̂");
    let se_eta = |a: f64| {
        let c = [1.0, a];
        let mut q = 0.0_f64;
        for j in 0..2 {
            for k in 0..2 {
                q += c[j] * v_gamma[[j, k]] * c[k];
            }
        }
        q.sqrt()
    };
    for window in edges.windows(2) {
        let (lo, hi) = (window[0], window[1]);
        let rows: Vec<usize> = (0..n)
            .filter(|&i| a_block[[i, 0]] >= lo && a_block[[i, 0]] < hi)
            .collect();
        let n_b = rows.len() as f64;
        let rms = (rows.iter().map(|&i| zeta[i] * zeta[i]).sum::<f64>() / n_b).sqrt();
        let (a_min, a_max) = rows.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &i| {
            (l.min(a_block[[i, 0]]), h.max(a_block[[i, 0]]))
        });
        let tolerance = crit * (1.0 / (2.0 * n_b).sqrt() + 0.5 * se_eta(a_min).max(se_eta(a_max)));
        assert!(
            (rms - 1.0).abs() <= tolerance,
            "gam#4019: ζ̂ must be unit-scaled on a ∈ [{lo}, {hi}) ({n_b} rows); RMS {rms:.4}, \
             tolerance {tolerance:.4}"
        );
    }
}

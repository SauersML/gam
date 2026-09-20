//! gam#3047 / gam#3030: the stacked first-stage sandwich of the conditional
//! latent calibration, pinned against the estimating system itself and audited
//! for calibration of the interval it feeds.
//!
//! Two arms:
//!
//!   - an exact pin of the inverse bread: perturb the score `z → z + tδ`, refit,
//!     and compare the central difference of the refit `θ̂₁` with `J⁻¹·∂ψ/∂t`.
//!     The mean stage is linear in `z` and the variance stage quadratic, so a
//!     central difference is EXACT up to roundoff; a reversed cross-stage block
//!     (the gam#3047 sign) misses by the whole of `2K·g_m`.
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
use gam_test_support::calibration::{
    COVERAGE_FALSE_POSITIVE_RATE, COVERAGE_NOMINAL_LEVELS, COVERAGE_REPLICATIONS,
    CalibrationRng, CoverageClass, audit_coverage, standard_normal_quantile,
};
use ndarray::{Array1, Array2, s};

/// Moments of `a ~ U(−1, 1)`: `E a² = 1/3`, `E a⁴ = 1/5`, odd moments zero.
const A_SECOND_MOMENT: f64 = 1.0 / 3.0;
const A_FOURTH_MOMENT: f64 = 1.0 / 5.0;
/// `z = QUAD·a² + LIN·a + e`: the quadratic term makes the linear conditional
/// mean MISSPECIFIED, so the mean residual is correlated with `a²` and the
/// cross-stage bread `M_vm = −2 Σ w û B Aᵀ` is O(n) rather than ≈ 0.
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
/// `E[u² | a] = QUAD²(a² − E a²)² + (SCALE0 + SCALE1·a)²`, whose projection on
/// `[1, a]` has intercept `E u² = QUAD²·Var(a²) + SCALE0² + SCALE1²·E a²` and
/// slope `2·SCALE0·SCALE1` (the only odd part). The constant stage is the
/// intercept alone.
fn pseudo_truth() -> ([f64; 2], [f64; 2]) {
    let var_a2 = A_FOURTH_MOMENT - A_SECOND_MOMENT * A_SECOND_MOMENT;
    let mean = [QUAD * A_SECOND_MOMENT, LIN];
    let var = [
        QUAD * QUAD * var_a2 + SCALE0 * SCALE0 + SCALE1 * SCALE1 * A_SECOND_MOMENT,
        2.0 * SCALE0 * SCALE1,
    ];
    (mean, var)
}

/// The refit's `θ₁ = (mean_coeffs, variance stage)`, in the coordinates of
/// `theta1_cov`.
fn theta1(cal: &LatentZConditionalCalibration) -> Vec<f64> {
    let mut theta = cal.mean_coeffs.clone();
    if cal.var_coeffs.is_empty() {
        theta.push(cal.homoskedastic_var);
    } else {
        theta.extend_from_slice(&cal.var_coeffs);
    }
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
    assert_eq!(cal.var_coeffs.is_empty(), !fit_variance);
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
    let (var_basis, var_normal) = if fit_variance {
        (basis.clone(), mean_normal.clone())
    } else {
        (
            Array2::<f64>::ones((n, 1)),
            Array2::from_elem((1, 1), weights.sum()),
        )
    };
    let q = var_basis.ncols();
    let j_inv = stacked_first_stage_inverse_bread(
        basis.view(),
        var_basis.view(),
        weights.view(),
        &mean_residuals,
        &mean_normal,
        &var_normal,
    )
    .expect("inverse bread");

    // ∂ψ/∂t at t = 0: ψ^m = Σ w A û, ψ^v = Σ w B (û² − Bᵀβ_v), û = z + tδ − Aᵀβ_m.
    let mut g = Array1::<f64>::zeros(p + q);
    for i in 0..n {
        for j in 0..p {
            g[j] += weights[i] * basis[[i, j]] * delta[i];
        }
        for j in 0..q {
            g[p + j] += 2.0 * weights[i] * mean_residuals[i] * var_basis[[i, j]] * delta[i];
        }
    }
    let predicted = j_inv.dot(&g);

    // The gam#3047 mutant: the pre-fix `K = −M⁻¹·M_vm·M⁻¹`, which on the fired
    // stage (`N = M`) is exactly the reversed sign of `N⁻¹·M_vm·M⁻¹`.
    let mut mutant = j_inv.clone();
    mutant.slice_mut(s![p.., ..p]).mapv_inplace(|entry| -entry);
    let mutant_predicted = mutant.dot(&g);

    // The mean stage is linear in z and the variance stage quadratic, so the
    // central difference is exact; the step only sets the roundoff scale.
    let step = 1.0e-3;
    let refit = |t: f64| {
        let zt = &z + &(&delta * t);
        theta1(
            &fit_conditional_latent_calibration(&zt, &weights, a_block.view(), fit_variance)
                .expect("perturbed fit"),
        )
    };
    let plus = refit(step);
    let minus = refit(-step);
    let observed: Vec<f64> = plus
        .iter()
        .zip(minus.iter())
        .map(|(a, b)| (a - b) / (2.0 * step))
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

/// The roundoff budget of the exact pin. A central difference of an O(1)
/// quadratic at step `h = 1e-3` carries `ε/h ≈ 2e-13` of cancellation error
/// per coordinate, amplified by the conditioning of the O(n) normal systems;
/// `√ε` bounds that with room, and a reversed `K` misses by O(1).
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
        "gam#3030: the constant variance stage v̂ = Σwû²/Σw must be propagated by the same \
         inverse bread; relative error {err:.3e}"
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
    let (mean_star, var_star) = pseudo_truth();
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
                    var_star[0] + var_star[1] * a
                } else {
                    var_star[0]
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


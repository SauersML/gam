//! OBJECTIVE quality of gam's ALO (approximate leave-one-out) corrected linear
//! predictor on real binomial/logit data.
//!
//! The point of any LOO method is *honest out-of-sample prediction*: the
//! corrected predictor `eta_tilde_i` is what the model would have predicted for
//! observation `i` had it never seen `i`. The objective metric this test asserts
//! is therefore the **mean held-out binomial deviance (log-loss)** of the
//! corrected linear predictor against the real `DEATH_EVENT` responses:
//!
//!     loss(eta) = mean_i [ -2 * ( y_i*log p_i + (1-y_i)*log(1-p_i) ) ],
//!     p_i = logistic(eta_i).
//!
//! PRIMARY OBJECTIVE CLAIM (predictive honesty):
//!   * ALO's held-out log-loss must be *strictly larger* than the model's own
//!     in-sample log-loss — an LOO predictor that does not pay an honest
//!     out-of-sample penalty is not doing leave-one-out at all (it is just the
//!     in-sample fit relabelled). This is a property of LOO that holds for any
//!     correct implementation regardless of any reference tool.
//!   * NOTE: we deliberately do NOT assert "ALO beats the intercept-only
//!     baseline" on this cohort. The EXACT frozen-curvature LOO oracle itself
//!     scores WORSE than the intercept-only marginal rate here (the deterministically
//!     subsampled `ejection_fraction`-only smooth carries no out-of-sample edge
//!     over the marginal death rate), so demanding `loss_alo < loss_intercept`
//!     would assert predictive signal the data does not contain — a bar even a
//!     perfect oracle fails. The legitimate predictive claim is match-or-beat
//!     against that exact oracle (below).
//!
//! BASELINE TO MATCH-OR-BEAT (objective accuracy, not "same fit"):
//!   * exact FROZEN-CURVATURE (frozen-λ) brute-force LOO — the EXACT leave-one-out
//!     predictor ALO computes: smoothing parameters λ, the penalty block S(λ) AND
//!     the penalized Hessian H = XᵀWX + S(λ) are all held at the full fit's
//!     converged values. ALO holds the OFF-row curvature frozen at H and solves
//!     the dropped-row stationarity, reduced to the scalar fixed point
//!       η̃_i = η̂_i + h_i · ℓ_i'(η̃_i),   h_i = x_iᵀ H⁻¹ x_i,
//!       ℓ_i'(η) = μ(η) − y_i  (canonical logit: c_i = w_i/μ'(η̂_i) = 1),
//!     keeping only the held-out row's own likelihood curvature exact. We
//!     reconstruct THAT fixed point independently from gam's converged geometry
//!     (dense H⁻¹ leverage solve, Newton on the 1-D residual) and read η̃_i. This
//!     is the unimpeachable mathematical oracle ALO is derived from, so it is
//!     ground truth, not a peer tool.
//!
//!     ESTIMAND ALIGNMENT (two distinct refit notions, both rejected as oracles).
//!     (1) A per-fold re-selected-λ refit (`fit_from_formula` per fold) re-runs
//!     REML and re-selects λ on each n−1 subsample — a DIFFERENT estimand
//!     (LOO-with-λ-reselected); on a small penalized fit λ is volatile fold to
//!     fold, so it disagrees with ALO by ~20% in relative L2 for reasons that are
//!     NOT an ALO-algebra defect. (2) A frozen-λ but RE-CURVED nonlinear refit
//!     (rebuild Σ_{j≠i} μ_j(1−μ_j) x_j x_jᵀ + S at the dropped optimum) still
//!     differs from frozen-curvature ALO at second order in the O(p/n) off-row
//!     Hessian change — another genuine estimand gap (~1.5% rel L2 at n=60), not
//!     an algebra bug. ALO freezes that off-row curvature by construction, so the
//!     only quantity it can be held to element-wise is the frozen-CURVATURE LOO
//!     above.
//!
//! GROUND-TRUTH CORRECTNESS (kept — the frozen-curvature exact LOO is the EXACT
//! analytic quantity ALO computes, not a noisy peer-tool fit): the corrected
//! predictors must agree with it to solver round-off. A genuine error in the ALO
//! algebra would both blow up this agreement and degrade the predictive metric
//! above; keeping it pins down *where* a regression came from.
//!
//! We use Binomial/logit — the canonical GLM case. The logit link is canonical,
//! so the IRLS working weights equal the Fisher information and ALO's one-step
//! Newton correction is at its most accurate.
//!
//! Data: `heart_failure_clinical_records_dataset.csv` (299 real patients),
//! `DEATH_EVENT ~ s(ejection_fraction)`. Identical encoded data feeds the full
//! fit (for ALO) and every leave-one-out refit (for the exact oracle): the LOO
//! datasets are the full encoded design with exactly one row deleted, so basis,
//! family, link, and smoothing machinery are byte-for-byte the same in both arms.

use gam::data::EncodedDataset;
use gam::estimate::UnifiedFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::{TermCollectionSpec, build_term_collection_design};
use gam::test_support::reference::{max_abs_diff, pearson, relative_l2};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2, ArrayView1};
use std::path::Path;

const HEART_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

const FORMULA: &str = "DEATH_EVENT ~ s(ejection_fraction)";

/// Solve the SPD system `A x = b` by dense Cholesky (A = L Lᵀ). The frozen-λ
/// leave-i-out Hessian Σ_{j≠i} w_j x_j x_jᵀ + S(λ) is SPD (Fisher weights ≥ 0
/// plus a positive penalty/stabilisation ridge), so this is the exact solver for
/// the brute-force oracle. Panics on a non-positive pivot (a non-SPD Hessian is
/// itself a real defect worth failing on).
fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let p = a.nrows();
    assert_eq!(p, a.ncols(), "solve_spd requires a square matrix");
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                assert!(
                    sum > 0.0,
                    "leave-i-out Hessian not SPD at pivot {i}: {sum:.3e}"
                );
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Forward solve L y = b.
    let mut yv = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * yv[k];
        }
        yv[i] = sum / l[[i, i]];
    }
    // Back solve Lᵀ x = y.
    let mut xv = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        let mut sum = yv[i];
        for k in (i + 1)..p {
            sum -= l[[k, i]] * xv[k];
        }
        xv[i] = sum / l[[i, i]];
    }
    xv
}

/// Fit the binomial/logit smooth and return the fitted `UnifiedFitResult`
/// together with the frozen term spec needed to rebuild the design.
fn fit_binomial_logit(ds: &EncodedDataset) -> (UnifiedFitResult, TermCollectionSpec) {
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(FORMULA, ds, &cfg).expect("gam binomial/logit fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial/logit s(x)");
    };
    (fit.fit, fit.resolvedspec)
}

/// Evaluate the linear predictor (logit scale) of a fitted model at a single
/// design point `x` for the predictor column `pred_idx`, using the model's own
/// frozen spec. For the logit link, design·beta IS eta, exactly the quantity
/// exact LOO produces for the held-out point.
fn eta_at_point(
    beta: &Array1<f64>,
    spec: &TermCollectionSpec,
    n_headers: usize,
    pred_idx: usize,
    x: f64,
) -> f64 {
    let mut grid = Array2::<f64>::zeros((1, n_headers));
    grid[[0, pred_idx]] = x;
    let design =
        build_term_collection_design(grid.view(), spec).expect("rebuild design at held-out point");
    design.design.apply(beta)[0]
}

/// Mean held-out binomial deviance (log-loss) of a linear predictor `eta`
/// against binary responses `y`: `mean_i -2*(y log p + (1-y) log(1-p))`,
/// `p = logistic(eta)`. Lower is a more accurate probabilistic prediction.
/// Probabilities are clamped away from {0,1} so a single confident miss cannot
/// produce a non-finite loss.
fn mean_binomial_log_loss(eta: &[f64], y: &[f64]) -> f64 {
    assert_eq!(eta.len(), y.len(), "log-loss length mismatch");
    const EPS: f64 = 1e-12;
    let mut acc = 0.0;
    for (&e, &yi) in eta.iter().zip(y.iter()) {
        let p = (1.0 / (1.0 + (-e).exp())).clamp(EPS, 1.0 - EPS);
        acc += -2.0 * (yi * p.ln() + (1.0 - yi) * (1.0 - p).ln());
    }
    acc / eta.len() as f64
}


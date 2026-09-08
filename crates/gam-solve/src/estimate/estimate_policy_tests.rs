//! Estimate-policy unit tests.
//!
//! Declared only as `#[cfg(test)] mod estimate_policy_tests;` in
//! `estimate/mod.rs`; the inner attribute below states that scope in the file
//! itself, so the compiler enforces it rather than a naming convention.
#![cfg(test)]

use super::evaluation::{
    sas_effective_epsilon, sas_effective_epsilon_second, sas_log_delta_edge_barriercostgrad,
    sas_log_delta_edge_barriercostgradhess,
};
use super::external_options::resolve_external_family;
use super::optimizer::{
    external_reml_seed_config, freeze_lambda_search_nuisance_at_canonical_anchor,
    standard_reml_search_prefers_gradient_only,
};
use super::penalty::REML_SEED_SCREENING_RHO_CAP;
use super::prefit::{
    PrefitRegularityDiagnostic, detect_prefit_binomial_single_column_separation_in_design,
    detect_prefit_unpenalized_rank_deficiency_in_design, reject_prefit_binomial_separation,
    reject_prefit_unpenalized_rank_deficiency,
};
use super::reml::hyper::link_binomial_aux;
use super::*;
use crate::mixture_link::{
    sas_inverse_link_jet, sas_inverse_link_jetwith_param_partials, sas_link_complement,
};
use gam_linalg::utils::StableSolver;
use gam_problem::{
    InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily, SeedRiskProfile, StandardLink,
};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::atomic::Ordering;

#[test]
fn gaussian_external_reml_uses_one_analytic_seed() {
    // The profiled-Gaussian path scores its data-derived `initial.sp` and
    // summed-penalty diagonal candidates before constructing the outer
    // problem.  The generic lattice must not repeat that basin decision.
    let cfg = external_reml_seed_config(2, LinkFunction::Identity);
    assert_eq!(cfg.risk_profile, SeedRiskProfile::Gaussian);
    assert_eq!(cfg.max_seeds, 1);
    assert_eq!(cfg.seed_budget, 3);
    assert_eq!(cfg.over_smoothing_probe_rho, None);
}

#[test]
fn high_dimensional_gaussian_external_reml_does_not_restore_a_lattice() {
    // Coordinate count must not silently re-enable heuristic global shifts:
    // the coupled analytic candidates own the same decision at every k.
    let cfg = external_reml_seed_config(REML_SEED_SCREENING_RHO_CAP, LinkFunction::Identity);
    assert_eq!(cfg.risk_profile, SeedRiskProfile::Gaussian);
    assert_eq!(cfg.max_seeds, 1);
    assert_eq!(cfg.seed_budget, 3);
    assert_eq!(cfg.over_smoothing_probe_rho, None);
}

#[test]
fn high_dimensional_glm_external_reml_requests_arc_seed_pair() {
    let cfg = external_reml_seed_config(REML_SEED_SCREENING_RHO_CAP, LinkFunction::Logit);
    assert_eq!(cfg.risk_profile, SeedRiskProfile::GeneralizedLinear);
    assert_eq!(
        cfg.max_seeds, 2,
        "high-dimensional GLM REML must generate the alternate ARC startup basin"
    );
    assert_eq!(
        cfg.seed_budget, 2,
        "high-dimensional GLM REML must request both generated starts so ARC's GLM cap is not nullified"
    );
}

#[test]
fn generalized_external_reml_keeps_multistart_policy() {
    let cfg = external_reml_seed_config(2, LinkFunction::Logit);
    assert_eq!(cfg.risk_profile, SeedRiskProfile::GeneralizedLinear);
    assert!(cfg.max_seeds > 1);
    assert_eq!(
        cfg.seed_budget, 2,
        "GLM REML must request the alternate ARC startup basin"
    );
}

#[test]
fn profiled_gaussian_search_consumes_exact_outer_curvature() {
    assert!(
        !standard_reml_search_prefers_gradient_only(LinkFunction::Identity),
        "quadratic Gaussian identity REML must route its available exact Hessian into search"
    );
}

#[test]
fn non_gaussian_search_reserves_order_four_for_mint() {
    for link in [
        LinkFunction::Logit,
        LinkFunction::Probit,
        LinkFunction::Log,
    ] {
        assert!(
            standard_reml_search_prefers_gradient_only(link),
            "{link:?} must retain the optimize-3 / certify-4 derivative ceiling"
        );
    }
}

#[test]
fn constraint_matrix_internal_transform_equals_backtransform_composition() {
    // Conditioning: intercept at col 0, a centered+scaled col 1
    // (mean=0.37, scale=2.5), and a plain unconditioned col 2.
    let conditioning = ParametricColumnConditioning {
        intercept_idx: Some(0),
        columns: vec![(1, 0.37, 2.5)],
    };

    // Constraint matrix authored on the ORIGINAL (user-scale) coefficients.
    // Row 0/1 are a pure box on β1 (β1 ≥ ·, β1 ≤ ·) with a *zero* intercept
    // entry — the case the old `+mean·scale` bug still mangled via the scale
    // power. Row 2 genuinely touches the intercept column, exercising the
    // mean-mixing term that a single-coefficient box leaves at zero (so it
    // also pins the sign of that term).
    let a_orig = array![[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.5, -3.0],];
    let a_int = conditioning.transform_constraint_matrix_to_internal(&a_orig);

    // The defining invariant: A_int·β_int must equal A_orig·β_orig for the
    // β_orig the solver will actually report, i.e. A_int = A_orig·M where
    // β_orig = M·β_int = backtransform_beta(β_int). Anything else lets the
    // user-scale coefficient escape the box it satisfies internally.
    for beta_int in [
        array![0.3, 2.0, -1.5],
        array![-1.1, 4.7, 0.9],
        array![0.0, 1.0, 0.0],
    ] {
        let beta_orig = conditioning.backtransform_beta(&beta_int);
        let lhs_int = a_int.dot(&beta_int);
        let lhs_orig = a_orig.dot(&beta_orig);
        for k in 0..lhs_int.len() {
            assert!(
                (lhs_int[k] - lhs_orig[k]).abs() < 1e-12,
                "row {k}: internal constraint value {} != original-at-backtransform {} \
                 — A_int must equal A_orig·M",
                lhs_int[k],
                lhs_orig[k]
            );
        }
    }

    // Pin the box-escape mechanism directly: a pure `β1 ≤ ub` becomes
    // `(1/scale)·β1_int ≤ ub` internally, so the active-set row entry is
    // 1/scale (= 0.4), NOT scale (= 2.5, the old `1/scale²` escape).
    assert!(
        (a_int[[1, 1]] - (-1.0 / 2.5)).abs() < 1e-12,
        "internal box row entry is {}, expected -1/scale = -0.4",
        a_int[[1, 1]]
    );
    // The intercept column (M's identity column) and plain column are
    // carried through untouched.
    assert_eq!(a_int[[2, 0]], 1.0);
    assert_eq!(a_int[[2, 2]], -3.0);
}

/// `backtransform_covariance` must compute `M·Σ_int·Mᵀ` — the unique
/// congruence consistent with `β_orig = M·β_int`. The old implementation
/// computed `Mᵀ·Σ_int·M`, which silently swapped the conditioned slope's
/// variance with the intercept's whenever the parametric column was
/// centered or scaled.
#[test]
fn backtransform_covariance_uses_correct_basis_congruence() {
    // Intercept at col 0, plus two conditioned parametric columns to
    // exercise off-diagonal mixing (single column would only exercise the
    // diagonal swap symptom).
    let conditioning = ParametricColumnConditioning {
        intercept_idx: Some(0),
        columns: vec![(1, 0.7, 2.5), (2, -1.3, 0.4)],
    };

    // Build M explicitly so the congruence can be verified by direct
    // matrix algebra rather than re-derived inside the test.
    let mut m = Array2::<f64>::eye(3);
    m[[0, 1]] = -0.7 / 2.5;
    m[[0, 2]] = -(-1.3) / 0.4;
    m[[1, 1]] = 1.0 / 2.5;
    m[[2, 2]] = 1.0 / 0.4;

    // A non-trivial symmetric PD `Σ_int`. The off-diagonals matter:
    // they're exactly the entries `Mᵀ·Σ·M` mishandles vs `M·Σ·Mᵀ`.
    let sigma_int = array![[1.7, -0.4, 0.9], [-0.4, 2.1, -0.2], [0.9, -0.2, 3.0],];

    let expected = m.dot(&sigma_int).dot(&m.t());
    let actual = conditioning.backtransform_covariance(&sigma_int);

    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (actual[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "backtransform_covariance mismatch at ({i},{j}): \
                 got {}, expected {} = (M·Σ·Mᵀ)[{i},{j}]",
                actual[[i, j]],
                expected[[i, j]],
            );
        }
    }

    // Pin the user-visible symptom directly: a `y ~ x` Gaussian fit with
    // a non-zero-mean x. After conditioning, `Σ_int` is the
    // diag(σ²/n, σ²/Sxx_centered) covariance of the orthogonalized
    // (intercept, centered slope) coefficients. The raw-basis variances
    // (M·Σ·Mᵀ) must be the textbook OLS expressions:
    //   Var(intercept_raw) = σ² (1/n + x̄² / Sxx)
    //   Var(slope_raw)     = σ² / Sxx
    // Anything that reports `σ²/n` as the intercept variance is the old
    // bug — the conditioned-basis intercept variance leaking through.
    let one_x_only = ParametricColumnConditioning {
        intercept_idx: Some(0),
        columns: vec![(1, 5.0, 2.0)], // x̄ = 5, sd(x) = 2
    };
    let sigma_sq = 1.7;
    let n = 250.0;
    let sxx = (n - 1.0) * 4.0; // sd² · (n−1) for a sample with sd(x)=2
    let sigma_int_yx = array![
        [sigma_sq / n, 0.0],
        [0.0, sigma_sq / (sxx / 4.0)], // centered+scaled (divide by sd² for the conditioned scale)
    ];
    let cov_raw = one_x_only.backtransform_covariance(&sigma_int_yx);
    let expected_var_intercept = sigma_sq * (1.0 / n + 25.0 / sxx);
    let expected_var_slope = sigma_sq / sxx;
    assert!(
        (cov_raw[[0, 0]] - expected_var_intercept).abs() < 1e-10,
        "raw intercept variance: got {}, expected {} (= σ²(1/n + x̄²/Sxx))",
        cov_raw[[0, 0]],
        expected_var_intercept
    );
    assert!(
        (cov_raw[[1, 1]] - expected_var_slope).abs() < 1e-10,
        "raw slope variance: got {}, expected {} (= σ²/Sxx)",
        cov_raw[[1, 1]],
        expected_var_slope
    );
}

/// `backtransform_penalized_hessian` must compute `M⁻ᵀ·H_int·M⁻¹` —
/// derived from `L_int(β_int) = L_orig(M·β_int)` and the chain rule.
/// Together with `backtransform_covariance`, this preserves the exact
/// inverse pair `inv(H_orig) == Σ_orig` whenever `inv(H_int) == Σ_int`.
#[test]
fn backtransform_penalized_hessian_is_inverse_of_covariance_backtransform() {
    let conditioning = ParametricColumnConditioning {
        intercept_idx: Some(0),
        columns: vec![(1, 0.7, 2.5), (2, -1.3, 0.4)],
    };

    // Build M and M⁻¹ explicitly.
    let mut m = Array2::<f64>::eye(3);
    m[[0, 1]] = -0.7 / 2.5;
    m[[0, 2]] = -(-1.3) / 0.4;
    m[[1, 1]] = 1.0 / 2.5;
    m[[2, 2]] = 1.0 / 0.4;
    let mut m_inv = Array2::<f64>::eye(3);
    m_inv[[0, 1]] = 0.7;
    m_inv[[0, 2]] = -1.3;
    m_inv[[1, 1]] = 2.5;
    m_inv[[2, 2]] = 0.4;

    let h_int = array![[3.2, 0.5, -0.3], [0.5, 1.4, 0.2], [-0.3, 0.2, 2.0],];

    let expected = m_inv.t().dot(&h_int).dot(&m_inv);
    let actual = conditioning.backtransform_penalized_hessian(&h_int);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (actual[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "backtransform_penalized_hessian mismatch at ({i},{j}): \
                 got {}, expected {} = (M⁻ᵀ·H·M⁻¹)[{i},{j}]",
                actual[[i, j]],
                expected[[i, j]],
            );
        }
    }

    // And the covariance/Hessian back-transforms compose so that
    // `Σ_orig = inv(H_orig)` holds whenever `Σ_int = inv(H_int)`. Pick a
    // `Σ_int = inv(H_int)` (smoothly invertible above), back-transform
    // each, and confirm they are mutual inverses to working precision.
    let sigma_int = {
        // 3×3 inverse via cofactors — small enough to hand-roll.
        let det = h_int[[0, 0]] * (h_int[[1, 1]] * h_int[[2, 2]] - h_int[[1, 2]] * h_int[[2, 1]])
            - h_int[[0, 1]] * (h_int[[1, 0]] * h_int[[2, 2]] - h_int[[1, 2]] * h_int[[2, 0]])
            + h_int[[0, 2]] * (h_int[[1, 0]] * h_int[[2, 1]] - h_int[[1, 1]] * h_int[[2, 0]]);
        let mut inv = Array2::<f64>::zeros((3, 3));
        inv[[0, 0]] = (h_int[[1, 1]] * h_int[[2, 2]] - h_int[[1, 2]] * h_int[[2, 1]]) / det;
        inv[[0, 1]] = -(h_int[[0, 1]] * h_int[[2, 2]] - h_int[[0, 2]] * h_int[[2, 1]]) / det;
        inv[[0, 2]] = (h_int[[0, 1]] * h_int[[1, 2]] - h_int[[0, 2]] * h_int[[1, 1]]) / det;
        inv[[1, 0]] = -(h_int[[1, 0]] * h_int[[2, 2]] - h_int[[1, 2]] * h_int[[2, 0]]) / det;
        inv[[1, 1]] = (h_int[[0, 0]] * h_int[[2, 2]] - h_int[[0, 2]] * h_int[[2, 0]]) / det;
        inv[[1, 2]] = -(h_int[[0, 0]] * h_int[[1, 2]] - h_int[[0, 2]] * h_int[[1, 0]]) / det;
        inv[[2, 0]] = (h_int[[1, 0]] * h_int[[2, 1]] - h_int[[1, 1]] * h_int[[2, 0]]) / det;
        inv[[2, 1]] = -(h_int[[0, 0]] * h_int[[2, 1]] - h_int[[0, 1]] * h_int[[2, 0]]) / det;
        inv[[2, 2]] = (h_int[[0, 0]] * h_int[[1, 1]] - h_int[[0, 1]] * h_int[[1, 0]]) / det;
        inv
    };
    let cov_orig = conditioning.backtransform_covariance(&sigma_int);
    let h_orig = conditioning.backtransform_penalized_hessian(&h_int);
    let product = cov_orig.dot(&h_orig);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (product[[i, j]] - expected).abs() < 1e-10,
                "Σ_orig · H_orig should be identity at ({i},{j}): got {}",
                product[[i, j]]
            );
        }
    }
}

#[test]
fn prefit_binomial_detects_unpenalized_realized_design_separator() {
    let x = array![[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let w = Array1::ones(y.len());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let diagnostic = detect_prefit_binomial_single_column_separation_in_design(
        y.view(),
        w.view(),
        &design,
        &[true, true],
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "separation screen must complete without a layout error", e
        )
    })
    .expect("second column exactly separates the binary response");

    assert_eq!(diagnostic.column_index, 1);
    assert!(diagnostic.positive_above_threshold);
    assert_eq!(diagnostic.threshold, 0.0);
}

#[test]
fn prefit_binomial_screen_respects_penalties_and_fractional_responses() {
    let x = array![[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]];
    let binary_y = array![0.0, 0.0, 1.0, 1.0];
    let fractional_y = array![0.0, 0.25, 0.75, 1.0];
    let w = Array1::ones(binary_y.len());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));

    assert_eq!(
        detect_prefit_binomial_single_column_separation_in_design(
            binary_y.view(),
            w.view(),
            &design,
            &[true, false],
        )
        .unwrap_or_else(|e| panic!(
            "{} failed: {:?}",
            "separation screen must complete without a layout error", e
        )),
        None,
        "a separating column with effective quadratic penalty should not be pre-fit rejected"
    );
    assert_eq!(
        detect_prefit_binomial_single_column_separation_in_design(
            fractional_y.view(),
            w.view(),
            &design,
            &[true, true],
        )
        .unwrap_or_else(|e| panic!(
            "{} failed: {:?}",
            "separation screen must complete without a layout error", e
        )),
        None,
        "fractional binomial proportions are not exact binary separation"
    );
}

#[test]
fn prefit_binomial_logit_rejects_before_outer_solver() {
    let x = array![[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let w = Array1::ones(y.len());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let cfg = RemlConfig::external(
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        1e-7,
        false,
    );
    let err = reject_prefit_binomial_separation(&cfg, y.view(), w.view(), &design, &[])
        .expect_err("unpenalized exact separator should fail before REML/PIRLS");

    assert!(matches!(
        err,
        EstimationError::PrefitPerfectSeparationDetected {
            column_index: 1,
            positive_above_threshold: true,
            ..
        }
    ));
}

#[test]
fn prefit_binomial_probit_rejects_before_outer_solver() {
    let x = array![[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let w = Array1::ones(y.len());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let cfg = RemlConfig::external(
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Probit),
        )),
        1e-7,
        false,
    );
    let err = reject_prefit_binomial_separation(&cfg, y.view(), w.view(), &design, &[])
        .expect_err("unpenalized exact separator should fail before REML/PIRLS");

    assert!(matches!(
        err,
        EstimationError::PrefitPerfectSeparationDetected {
            column_index: 1,
            positive_above_threshold: true,
            ..
        }
    ));
}

#[test]
fn prefit_binomial_rejects_linear_combination_separator() {
    let x = array![
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, -1.0]
    ];
    let y = array![1.0, 1.0, 0.0, 0.0];
    let w = Array1::ones(y.len());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let cfg = RemlConfig::external(
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        1e-7,
        false,
    );
    let err = reject_prefit_binomial_separation(&cfg, y.view(), w.view(), &design, &[])
        .expect_err("x1 + x2 separates although neither coordinate separates alone");

    assert!(matches!(
        err,
        EstimationError::PrefitLinearSeparationDetected {
            num_unpenalized_columns: 3,
            ..
        }
    ));
}

#[test]
fn prefit_rank_check_detects_unpenalized_duplicate_column() {
    let x = array![
        [1.0, -2.0, -2.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0]
    ];
    let w = Array1::ones(x.nrows());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let diagnostic =
        detect_prefit_unpenalized_rank_deficiency_in_design(w.view(), &design, &[true, true, true])
            .unwrap_or_else(|e| {
                panic!(
                    "{} failed: {:?}",
                    "rank check should stream dense design", e
                )
            })
            .expect("duplicate unpenalized columns are rank deficient");

    match diagnostic {
        PrefitRegularityDiagnostic::RankDeficient {
            rank,
            num_unpenalized_columns,
            min_eigenvalue,
            tolerance,
            column_indices,
        } => {
            assert_eq!(rank, 2);
            assert_eq!(num_unpenalized_columns, 3);
            assert_eq!(column_indices, vec![0, 1, 2]);
            assert!(
                min_eigenvalue.abs() <= tolerance,
                "duplicate-column min eigenvalue should be at the rank tolerance"
            );
        }
        other => panic!("expected exact rank deficiency, got {other:?}"),
    }
}

#[test]
fn prefit_rank_check_ignores_alias_carried_only_by_penalized_column() {
    let x = array![
        [1.0, -2.0, -2.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0]
    ];
    let w = Array1::ones(x.nrows());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let diagnostic = detect_prefit_unpenalized_rank_deficiency_in_design(
        w.view(),
        &design,
        &[true, true, false],
    )
    .expect("rank check should stream dense design");

    assert_eq!(
        diagnostic, None,
        "aliasing that is removed from the unpenalized subspace by a penalty should not be pre-fit rejected"
    );
}

#[test]
fn prefit_rank_check_rejects_before_reml_state_construction() {
    let x = array![
        [1.0, -2.0, -2.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0]
    ];
    let w = Array1::ones(x.nrows());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let err = reject_prefit_unpenalized_rank_deficiency(w.view(), &design, &[])
        .expect_err("rank-deficient unpenalized design should fail before REML/PIRLS");

    assert!(matches!(
        err,
        EstimationError::PrefitRankDeficientDesignDetected {
            rank: 2,
            num_unpenalized_columns: 3,
            ..
        }
    ));
}

#[test]
fn prefit_rank_check_detects_near_degenerate_unpenalized_design() {
    // Two near-collinear columns (alias to ~1e-7 perturbation) keep full
    // numeric rank but blow the Gram condition number past the
    // near-degeneracy tolerance, so the fit would grind/diverge.
    let x = array![
        [1.0, -2.0, -2.0 + 1e-7],
        [1.0, -1.0, -1.0 - 1e-7],
        [1.0, 1.0, 1.0 + 1e-7],
        [1.0, 2.0, 2.0 - 1e-7]
    ];
    let w = Array1::ones(x.nrows());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let diagnostic =
        detect_prefit_unpenalized_rank_deficiency_in_design(w.view(), &design, &[true, true, true])
            .unwrap_or_else(|e| {
                panic!(
                    "{} failed: {:?}",
                    "rank check should stream dense design", e
                )
            })
            .expect("near-collinear unpenalized columns are near-degenerate");

    match diagnostic {
        PrefitRegularityDiagnostic::NearDegenerate {
            num_unpenalized_columns,
            condition_number,
            tolerance,
            column_indices,
            ..
        } => {
            assert_eq!(num_unpenalized_columns, 3);
            assert_eq!(column_indices, vec![0, 1, 2]);
            assert!(
                condition_number > tolerance,
                "near-degenerate Gram condition number {condition_number:.3e} should exceed tolerance {tolerance:.3e}"
            );
        }
        other => panic!("expected near-degenerate diagnostic, got {other:?}"),
    }

    let err = reject_prefit_unpenalized_rank_deficiency(w.view(), &design, &[])
        .expect_err("near-degenerate unpenalized design should fail before REML/PIRLS");
    assert!(matches!(
        err,
        EstimationError::PrefitNearDegenerateDesignDetected {
            num_unpenalized_columns: 3,
            ..
        }
    ));
}

#[test]
fn prefit_rank_check_accepts_well_conditioned_unpenalized_design() {
    let x = array![
        [1.0, -2.0, 4.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 4.0]
    ];
    let w = Array1::ones(x.nrows());
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x));
    let diagnostic =
        detect_prefit_unpenalized_rank_deficiency_in_design(w.view(), &design, &[true, true, true])
            .expect("rank check should stream dense design");
    assert_eq!(
        diagnostic, None,
        "a well-conditioned full-rank unpenalized design must not be pre-fit rejected"
    );
}

#[test]
fn sas_raw_epsilon_hessian_chain_rule_matches_chained_gradient_slope() {
    let raw0 = 1.3_f64;
    let (eps0, d1, d2) = sas_effective_epsilon_second(raw0);
    let g0 = array![0.4, -0.7, 0.2];
    let h_eff = array![[2.0, 0.3, -0.1], [0.3, 1.5, 0.25], [-0.1, 0.25, 0.8]];

    let analytic = h_eff[[0, 0]] * d1 * d1 + g0[0] * d2;
    let chained_grad = |raw: f64| {
        let (eps, deps_draw) = sas_effective_epsilon(raw);
        let delta = array![eps - eps0, 0.0, 0.0];
        let g_eff = &g0 + &h_eff.dot(&delta);
        g_eff[0] * deps_draw
    };
    let h = 1e-6;
    let fd = (chained_grad(raw0 + h) - chained_grad(raw0 - h)) / (2.0 * h);
    assert!(
        (analytic - fd).abs() < 2e-8,
        "SAS raw epsilon Hessian chain rule mismatch: analytic={analytic:.12e} fd={fd:.12e}"
    );
}

#[test]
fn sas_log_delta_barrier_hessian_matches_gradient_slope() {
    let raw = 2.25_f64;
    let (_, _, analytic_hess) = sas_log_delta_edge_barriercostgradhess(raw);
    let h = 1e-6;
    let (_, gp) = sas_log_delta_edge_barriercostgrad(raw + h);
    let (_, gm) = sas_log_delta_edge_barriercostgrad(raw - h);
    let fd = (gp - gm) / (2.0 * h);
    assert!(
        (analytic_hess - fd).abs() < 2e-9,
        "SAS log-delta barrier Hessian mismatch: analytic={analytic_hess:.12e} fd={fd:.12e}"
    );
}

fn decode_invariant_test_parts() -> UnifiedFitResultParts {
    let log_lambdas = array![0.2_f64.ln(), 0.8_f64.ln()];
    let lambdas = log_lambdas
        .mapv(|value| gam_problem::checked_exp_log_strength(value).expect("fixture log strength"));
    UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: array![0.25, -0.5],
            role: BlockRole::Mean,
            edf: 1.5,
            lambdas: lambdas.clone(),
        }],
        // One working row per training row. `WorkingGeometry` carries the
        // P-IRLS working weights and response, which are per-observation, so
        // `training_sample_size` is not free to disagree with their length —
        // `UnifiedFitResult::try_from_parts` refuses when it does. This fixture
        // declared 16 against a 3-row working geometry and predated that check,
        // which is why all eight tests sharing it died in the constructor with
        // `working row count 3 must match training_sample_size 16` before
        // reaching the invariant each was written to measure.
        training_sample_size: 3,
        log_lambdas,
        lambdas,
        likelihood_family: Some(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        )),
        likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
        log_likelihood_normalization: LogLikelihoodNormalization::Full,
        log_likelihood: -1.2,
        deviance: 2.4,
        reml_score: Some(0.7),
        stable_penalty_term: 0.3,
        penalized_objective: Some(2.2),
        used_device: false,
        outer_iterations: 3,
        outer_converged: true,
        outer_gradient_norm: Some(0.05),
        standard_deviation: 1.1,
        covariance_conditional: Some(array![[1.0, 0.1], [0.1, 2.0]]),
        covariance_corrected: Some(array![[1.2, 0.1], [0.1, 2.2]]),
        inference: Some(FitInference {
            edf_by_block: vec![0.6, 0.9],
            penalty_block_trace: vec![],
            edf_total: 1.5,
            smoothing_correction: Some(array![[0.2, 0.0], [0.0, 0.2]]),
            smoothing_correction_method: Some(
                crate::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                    active_rank: 1,
                    rho_dimension: 1,
                },
            ),
            // This fixture's primary method is already FirstOrderIdentifiedSubspace
            // (no cubature upgrade), so its retained "first-order" pair mirrors
            // the primary pair.
            smoothing_correction_first_order: Some(array![[0.2, 0.0], [0.0, 0.2]]),
            smoothing_correction_method_first_order: Some(
                crate::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                    active_rank: 1,
                    rho_dimension: 1,
                },
            ),
            penalized_hessian: array![[2.0, 0.1], [0.1, 3.0]].into(),
            reparam_qs: Some(array![[1.0, 0.0], [0.0, 1.0]]),
            // Coherent with this fixture's own scale: the family is
            // Gaussian/identity with `likelihood_scale: ProfiledGaussian` and
            // `standard_deviation: 1.1`, so the dispersion IS the profiled
            // estimate phi-hat = sigma-hat^2 = 1.21. `Dispersion::UNIT` here was
            // a `Known 1.0` placeholder that contradicted both, and production
            // refuses the pair on sight ("cached inference dispersion
            // Dispersion { source: Known, phi: 1.0 } disagrees with
            // family-resolved dispersion Dispersion { source: Estimated,
            // phi: 1.2100000000000002 }") — correctly, since a cached phi that
            // disagrees with the family is exactly the divergence the cache
            // exists to prevent.
            dispersion: Dispersion::estimated(1.1 * 1.1)
                .expect("profiled Gaussian phi-hat = sigma-hat^2 is a valid estimate"),
            beta_covariance: Some(array![[1.0, 0.1], [0.1, 2.0]].into()),
            beta_standard_errors: Some(array![1.0, 2.0_f64.sqrt()]),
            beta_covariance_corrected: Some(array![[1.2, 0.1], [0.1, 2.2]]),
            beta_standard_errors_corrected: Some(array![1.2_f64.sqrt(), 2.2_f64.sqrt()]),
            beta_covariance_frequentist: None,
            coefficient_influence: None,
            weighted_gram: None,
            bias_correction_beta: None,
            bias_correction_jacobian: None,
        }),
        fitted_link: FittedLinkState::Standard(None),
        geometry: Some(FitGeometry {
            coefficient_gauge: gam_problem::Gauge::identity(&[2]),
            penalized_hessian: array![[2.0, 0.1], [0.1, 3.0]].into(),
            constrained_posterior: None,
            working: Some(crate::model_types::WorkingGeometry {
                weights: array![1.0, 0.5, 0.75],
                response: array![0.1, 0.2, 0.3],
            }),
        }),
        block_states: Vec::new(),
        pirls_status: crate::pirls::PirlsStatus::Converged,
        max_abs_eta: 1.25,
        constraint_kkt: None,
        artifacts: FitArtifacts {
            criterion_certificate: Some(crate::model_types::OuterCriterionCertificate {
                stationarity: crate::model_types::OuterStationarityCertificate::AnalyticGradient {
                    grad_norm: 0.05,
                    projected_grad_norm: 0.05,
                    bound: 0.1,
                    rung: crate::model_types::CertifiedRung {
                        label: "solver-band".to_string(),
                        derived_standard: false,
                    },
                },
                curvature: crate::model_types::CurvatureEvidence::Measured { psd: true },
                lambdas_railed: Vec::new(),
                railed_facts: Vec::new(),
                curvature_floor: None,
            }),
            ..Default::default()
        },
        inner_cycles: 0,
    }
}

fn decode_invariant_test_fit() -> UnifiedFitResult {
    UnifiedFitResult::try_from_parts(decode_invariant_test_parts())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "construct decode invariant test fit", e))
}

#[test]
fn unified_fit_accepts_inference_hessian_in_rectangular_active_geometry_frame() {
    let mut parts = decode_invariant_test_parts();
    let active_hessian = array![[2.5]];
    let geometry = parts.geometry.as_mut().expect("fixture geometry");
    geometry.coefficient_gauge = gam_problem::Gauge::from_t(
        Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).unwrap(),
        &[2],
        &[1],
    );
    geometry.penalized_hessian = active_hessian.clone().into();
    parts
        .inference
        .as_mut()
        .expect("fixture inference")
        .penalized_hessian = active_hessian.clone().into();

    let fit = UnifiedFitResult::try_from_parts(parts)
        .expect("a saved two-coordinate beta may carry one-coordinate active geometry");
    assert_eq!(fit.beta.len(), 2);
    assert_eq!(
        fit.geometry
            .as_ref()
            .expect("saved geometry")
            .coefficient_gauge
            .reduced_total(),
        1
    );
    assert_eq!(
        fit.penalized_hessian().expect("active Hessian").dim(),
        (1, 1)
    );
}

#[test]
fn unified_fit_constructor_rejects_nonconverged_outer_state() {
    let mut parts = decode_invariant_test_parts();
    parts.outer_converged = false;
    let error = UnifiedFitResult::try_from_parts(parts)
        .expect_err("an outer checkpoint must not mint a fitted model");
    assert!(matches!(error, EstimationError::FitDidNotConverge { .. }));
}

#[test]
fn unified_fit_constructor_rejects_every_nonconverged_inner_state() {
    // Every non-converged terminal state remains a checkpoint and must never
    // mint a fit, including a near-stationary stalled state.
    for status in [
        crate::pirls::PirlsStatus::StalledAtValidMinimum,
        crate::pirls::PirlsStatus::MaxIterationsReached,
        crate::pirls::PirlsStatus::LmStepSearchExhausted,
        crate::pirls::PirlsStatus::Unstable,
    ] {
        let mut parts = decode_invariant_test_parts();
        parts.pirls_status = status;
        let error = UnifiedFitResult::try_from_parts(parts)
            .expect_err("a non-converged inner state must be rejected");
        assert!(
            matches!(error, EstimationError::FitDidNotConverge { .. }),
            "status {status:?} should surface FitDidNotConverge, got {error:?}"
        );
    }
}

#[test]
fn unified_fit_constructor_requires_outer_certificate_after_iterations() {
    let mut parts = decode_invariant_test_parts();
    parts.artifacts.criterion_certificate = None;
    let error = UnifiedFitResult::try_from_parts(parts)
        .expect_err("outer iterations without analytic evidence must be rejected");
    assert!(matches!(error, EstimationError::FitDidNotConverge { .. }));
}

#[test]
fn dispersion_phi_prefers_inference_then_falls_back_to_standard_deviation() {
    // With a cached `inference` block present, `dispersion_phi()` returns
    // the stored dispersion verbatim so it can never diverge from the φ̂
    // that scaled the covariances at fit time.
    let fit = decode_invariant_test_fit();
    let expected_cached = Dispersion::estimated(1.1 * 1.1).expect("valid phi-hat");
    assert_eq!(fit.dispersion(), Some(expected_cached));
    assert_eq!(fit.dispersion_phi().unwrap(), 1.1 * 1.1);

    // Deployment-saved models drop `inference` (see `core_saved_fit_result`,
    // which stores `inference: None`). `dispersion()` is then `None`, but
    // `dispersion_phi()` must still recover the Gaussian scale φ̂ = σ̂² from
    // the always-serialized `standard_deviation`. This is the code path the
    // unseen-level prior variance (#674) relies on.
    // The cached and fallback routes now agree numerically, because coherence
    // requires it — a cached phi that the family would not reproduce is exactly
    // what production refuses. What still distinguishes them, and what this
    // half of the test owns, is that `dispersion()` (the cached BLOCK) vanishes
    // with `inference` while `dispersion_phi()` (the recovered SCALE) does not.
    let mut stripped = fit.clone();
    stripped.inference = None;
    assert!(stripped.dispersion().is_none());
    let expected_phi = stripped.standard_deviation * stripped.standard_deviation;
    assert!(
        (stripped.dispersion_phi().unwrap() - expected_phi).abs() < 1e-12,
        "fallback φ̂ should equal σ̂² = {expected_phi}, got {}",
        stripped.dispersion_phi().unwrap()
    );

    // A fixed-scale family (Poisson) keeps φ̂ = 1 on the fallback path even
    // with a non-unit residual summary, so the unseen-level prior collapses
    // to the historical 1/λ for those families.
    let mut poisson = stripped.clone();
    poisson.likelihood_family = Some(LikelihoodSpec::new(
        ResponseFamily::Poisson,
        InverseLink::Standard(StandardLink::Log),
    ));
    // The scale metadata has to move WITH the family. Cloning the Gaussian
    // fixture and swapping only `likelihood_family` left `ProfiledGaussian`
    // beside a Poisson response, and the resolver refuses that pair on sight —
    // "family poisson requires exact FixedDispersion { phi: 1.0 } metadata, got
    // ProfiledGaussian" — so the `unwrap` below blew up before the assertion it
    // guards was ever reached. Poisson is a fixed-scale family; its metadata
    // says so.
    poisson.likelihood_scale = LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 };
    poisson.standard_deviation = 2.7;
    assert_eq!(poisson.dispersion_phi().unwrap(), 1.0);
}

#[test]
fn resolve_external_family_rejects_unsupported_firth_request() {
    let err = resolve_external_family(
        &LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ),
        Some(true),
    )
    .expect_err("Poisson fitting should reject unsupported Firth requests explicitly");
    assert!(
        err.to_string()
            .contains("requires a Binomial inverse link with a Fisher-weight jet"),
        "unexpected error: {err}"
    );
}

#[test]
fn resolve_external_family_accepts_constant_precision_beta_regression() {
    // Beta(logit) with a constant precision φ is a genuine-dispersion mean
    // family on par with Gamma/Tweedie/Negative-Binomial: the external GLM
    // route fits the mean while φ is estimated by the Pearson moment estimator
    // (betareg's default behavior). The route must accept it and surface the
    // φ-estimation contract via the EstimatedBetaPhi scale metadata.
    let (spec, firth) = resolve_external_family(
        &LikelihoodSpec::new(
            ResponseFamily::Beta { phi: 5.0 },
            InverseLink::Standard(StandardLink::Logit),
        ),
        None,
    )
    .expect("external-design policy must accept constant-precision beta regression");
    assert!(
        !firth,
        "beta regression does not request Firth bias reduction"
    );
    assert!(
        spec.scale.beta_phi_is_estimated(),
        "beta φ must be flagged for joint estimation, got {:?}",
        spec.scale
    );
}

#[test]
fn resolve_external_family_accepts_supported_nonlogit_firth_request() {
    let (_, firth) = resolve_external_family(
        &LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::CLogLog),
        ),
        Some(true),
    )
    .expect("CLogLog has a Fisher-weight jet");
    assert!(firth);
}

#[test]
fn fit_geometry_wire_schema_requires_explicit_coefficient_gauge() {
    let fit = decode_invariant_test_fit();
    let mut payload = serde_json::to_value(&fit).expect("serialize fit");
    payload["geometry"]
        .as_object_mut()
        .expect("serialized geometry object")
        .remove("coefficient_gauge");
    let error = serde_json::from_value::<UnifiedFitResult>(payload)
        .expect_err("an old geometry payload must not guess an identity gauge");
    assert!(error.to_string().contains("missing field `coefficient_gauge`"));
}

#[test]
fn unified_fit_decode_validation_rejects_beta_drift_from_blocks() {
    let fit = decode_invariant_test_fit();
    let mut payload = serde_json::to_value(&fit).expect("serialize fit");
    // `Array1<f64>` uses ndarray's own (versioned-sequence) serde format,
    // not a bare JSON array, so round-trip the drifted value through
    // serde_json to honour that schema while still corrupting the data.
    payload["beta"] =
        serde_json::to_value(Array1::from(vec![9.0_f64, 8.0_f64])).expect("serialize drifted beta");
    let decoded: UnifiedFitResult =
        serde_json::from_value(payload).expect("deserialize corrupted fit");
    let err = decoded
        .validate_numeric_finiteness()
        .expect_err("beta drift should fail validation");
    assert!(
        err.to_string()
            .contains("decoded beta must match coefficient blocks"),
        "unexpected error: {err}"
    );
}

#[test]
fn unified_fit_validation_rejects_edf_smoothing_parameter_drift() {
    let mut fit = decode_invariant_test_fit();
    fit.inference
        .as_mut()
        .expect("test fit has inference")
        .edf_by_block = vec![1.5];
    let err = fit
        .validate_numeric_finiteness()
        .expect_err("EDF entries should align with smoothing parameters");
    assert!(
        err.to_string()
            .contains("EDF smoothing-parameter count mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn unified_fit_validation_rejects_any_log_lambda_drift() {
    let mut fit = decode_invariant_test_fit();
    fit.log_lambdas[0] += 5e-14;
    let err = fit
        .validate_numeric_finiteness()
        .expect_err("canonical log strengths must not drift from physical strengths");
    assert!(err.to_string().contains("log_lambdas must equal"));
}

#[test]
fn unified_fit_validation_rejects_material_log_lambda_drift() {
    let mut fit = decode_invariant_test_fit();
    fit.log_lambdas[0] += 1e-4;
    let err = fit
        .validate_numeric_finiteness()
        .expect_err("material log-lambda drift should fail validation");
    assert!(
        err.to_string().contains("log_lambdas must equal"),
        "unexpected error: {err}"
    );
}

#[test]
fn unified_fit_decode_validation_rejects_geometry_drift_from_inference() {
    let fit = decode_invariant_test_fit();
    let mut payload = serde_json::to_value(&fit).expect("serialize fit");
    let drifted_hessian: Array2<f64> = array![[4.0, 0.0], [0.0, 5.0]];
    payload["geometry"]["penalized_hessian"] =
        serde_json::to_value(&drifted_hessian).expect("serialize drifted penalized Hessian");
    let decoded: UnifiedFitResult =
        serde_json::from_value(payload).expect("deserialize corrupted fit");
    let err = decoded
        .validate_numeric_finiteness()
        .expect_err("geometry drift should fail validation");
    assert!(
        err.to_string()
            .contains("geometry penalized Hessian must match inference.penalized_hessian"),
        "unexpected error: {err}"
    );
}

fn build_tiny_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let x1 = -1.5 + 3.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (2.1 * x1).sin();
    }
    x
}

fn one_penalty_non_intercept(p: usize) -> Vec<Array2<f64>> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    vec![s]
}

fn dense_penalty_test_inputs(
    s_list: &[Array2<f64>],
    p: usize,
    context: &str,
) -> (
    Vec<PenaltySpec>,
    Vec<gam_terms::construction::CanonicalPenalty>,
    Vec<usize>,
) {
    let penalty_specs = s_list
        .iter()
        .cloned()
        .map(PenaltySpec::Dense)
        .collect::<Vec<_>>();
    let (canonical_penalties, active_nullspace_dims) =
        gam_terms::construction::canonicalize_penalty_specs(
            &penalty_specs,
            &vec![1; penalty_specs.len()],
            p,
            context,
        )
        .expect("canonicalize dense penalties");
    (penalty_specs, canonical_penalties, active_nullspace_dims)
}

#[test]
fn sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19() {
    let seed = 19_u64;
    let n = 20usize;
    let x = build_tiny_design(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_non_intercept(x.ncols());

    let true_beta = array![-0.2, 0.9, -0.4];
    let eta_true = x.dot(&true_beta);
    let eps_true = 0.25;
    let ld_true = -0.20;
    let p = eta_true.mapv(|e| {
        sas_inverse_link_jet(e, eps_true, ld_true)
            .expect("finite SAS eta")
            .mu
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

    let opts = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Sas(
                crate::mixture_link::state_from_sasspec(SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                })
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "valid SAS initial state", e)),
            ),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: Some(SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        }),
        optimize_sas: true,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };

    let theta = array![0.10, 0.12, -0.18];
    let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
    assert!(effective_sas_link.is_some());
    let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
        &s_list,
        x.ncols(),
        "sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19",
    );
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
        &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x.clone())),
        &penalty_specs,
    );
    let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(x.clone()),
    ));
    let mut reml_state = RemlState::newwith_offset(
        y.view(),
        x_fit,
        w.view(),
        offset.view(),
        canonical_penalties.clone(),
        x.ncols(),
        &cfg,
        Some(active_nullspace_dims.clone()),
        None,
        None,
    )
    .expect("reml_state");
    let rho = theta.slice(s![..1]).to_owned();
    let (epsilon_eff, d_eps_d_raw) = sas_effective_epsilon(theta[1]);
    let sas_state = state_from_sasspec(SasLinkSpec {
        initial_epsilon: epsilon_eff,
        initial_log_delta: theta[2],
    })
    .expect("sas state");
    reml_state.set_link_states(None, Some(sas_state));

    let pirls_result = reml_state
        .obtain_eval_bundle(&rho)
        .map(|b| b.pirls_result.clone())
        .expect("pirls_result");
    let eta = &pirls_result.final_eta;
    let x_t = &pirls_result.x_transformed;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let du_vec: Vec<f64> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta[i],
                sas_state.epsilon,
                sas_state.log_delta,
            )
            .expect("finite SAS eta");
            let mu = jets.jet.mu;
            let aux = link_binomial_aux(i, eta[i], y[i], w[i].max(0.0), mu, 1.0 - mu)
                .expect("finite binomial row geometry");
            let d1 = jets.jet.d1;
            let dmu = jets.djet_depsilon.mu;
            let dd1 = jets.djet_depsilon.d1;
            aux.a2 * dmu * d1 + aux.a1 * dd1
        })
        .collect();
    let du_by_eps = Array1::from_vec(du_vec);
    let score_at = |raw_eps: f64| -> Array1<f64> {
        let (eps_eff, _) = sas_effective_epsilon(raw_eps);
        let sas_state = state_from_sasspec(SasLinkSpec {
            initial_epsilon: eps_eff,
            initial_log_delta: theta[2],
        })
        .expect("score sas state");
        let out_vec: Vec<f64> = (0..eta.len())
            .into_par_iter()
            .map(|i| {
                let jets = sas_inverse_link_jetwith_param_partials(
                    eta[i],
                    sas_state.epsilon,
                    sas_state.log_delta,
                )
                .expect("finite SAS eta");
                let mu = jets.jet.mu;
                let d1 = jets.jet.d1;
                let aux = link_binomial_aux(i, eta[i], y[i], w[i].max(0.0), mu, 1.0 - mu)
                    .expect("finite binomial row geometry");
                aux.a1 * d1
            })
            .collect();
        Array1::from_vec(out_vec)
    };
    let score_p = score_at(theta[1] + 1e-4 * (1.0 + theta[1].abs()));
    let score_m = score_at(theta[1] - 1e-4 * (1.0 + theta[1].abs()));
    let fd_du_raw = (&score_p - &score_m).mapv(|v| v / (2.0 * 1e-4 * (1.0 + theta[1].abs())));
    let du_raw = du_by_eps.mapv(|v| v * d_eps_d_raw);
    // `du/d(raw ε)` at FIXED η compares an analytic single-row jet channel to a
    // fixed-η central difference — no PIRLS re-solve, so there is no solver
    // noise floor. The two agree to ~1e-8; a 1e-5 bound is a meaningful guard
    // (still ~1000× the observed residual) that would catch a dropped ε-jet
    // channel without flaking (gam#855).
    gam_linalg::test_support::fd_checker::assert_matrix_derivativefd(
        &fd_du_raw.insert_axis(Axis(1)),
        &du_raw.insert_axis(Axis(1)),
        1e-5,
        "sas du / d raw epsilon at fixed eta",
    );
    let rhs = x_t.transpose_vector_multiply(&du_by_eps);
    let neg_du_deta_vec: Vec<f64> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta[i].clamp(-30.0, 30.0),
                sas_state.epsilon,
                sas_state.log_delta,
            )
            .expect("finite SAS eta");
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let d2 = jets.jet.d2;
            let aux = link_binomial_aux(i, eta[i], y[i], w[i].max(0.0), mu, 1.0 - mu)
                .expect("finite binomial row geometry");
            -(aux.a2 * d1 * d1 + aux.a1 * d2)
        })
        .collect();
    let neg_du_deta = Array1::from_vec(neg_du_deta_vec);
    let score_beta_jacobian = {
        let x_dense = x_t.to_dense();
        let diag_v = Array2::from_diag(&neg_du_deta);
        let mut j = x_dense.t().dot(&diag_v).dot(&x_dense);
        for ((r, c), v) in pirls_result.reparam_result.s_transformed.indexed_iter() {
            j[[r, c]] += v;
        }
        if pirls_result.ridge_passport.delta() > 0.0 {
            for d in 0..j.nrows() {
                j[[d, d]] += pirls_result.ridge_passport.delta();
            }
        }
        j
    };
    let factor = StableSolver::new()
        .factorize(&score_beta_jacobian)
        .expect("observed-jacobian factorization for dbeta");
    let mut dbeta_exact = rhs.clone();
    let mut dbeta_matrix = gam_linalg::faer_ndarray::array1_to_col_matmut(&mut dbeta_exact);
    factor.solve_in_place(dbeta_matrix.as_mut());
    assert!(dbeta_exact.iter().all(|value| value.is_finite()));
    dbeta_exact *= d_eps_d_raw;

    let fd_h = 1e-4 * (1.0 + theta[1].abs());
    let beta_at = |raw_eps: f64| -> (Array1<f64>, f64) {
        let mut state = RemlState::newwith_offset(
            y.view(),
            conditioning.apply_to_design(&DesignMatrix::Dense(
                gam_linalg::matrix::DenseDesignMatrix::from(x.clone()),
            )),
            w.view(),
            offset.view(),
            canonical_penalties.clone(),
            x.ncols(),
            &cfg,
            Some(active_nullspace_dims.clone()),
            None,
            None,
        )
        .expect("fd state");
        let (eps_eff, _) = sas_effective_epsilon(raw_eps);
        let sas_state = state_from_sasspec(SasLinkSpec {
            initial_epsilon: eps_eff,
            initial_log_delta: theta[2],
        })
        .expect("fd sas state");
        state.set_link_states(None, Some(sas_state));
        let pirls = state
            .obtain_eval_bundle(&rho)
            .map(|b| b.pirls_result.clone())
            .expect("fd pirls");
        (
            pirls.beta_transformed.as_ref().clone(),
            pirls.ridge_passport.delta(),
        )
    };
    let (beta_p, ridge_p) = beta_at(theta[1] + fd_h);
    let (beta_m, ridge_m) = beta_at(theta[1] - fd_h);
    let fd_beta = (&beta_p - &beta_m).mapv(|v| v / (2.0 * fd_h));

    // gam#855: the analytic composite `dβ/dε = J⁻¹·rhs` is the exact IFT
    // linearization at the converged β̂; the FD comparator re-runs PIRLS to
    // convergence at each perturbed ε. With the ε-derivative channel of the
    // SAS-reweighted IRLS system fully captured (the original report's missing
    // channel), the two agree to ~1e-9 here — the well-conditioned n=20 fit
    // takes NO stabilization ridge (`ridge_used == 0`), so the earlier
    // "adaptive-ridge contaminates the FD" rationale does not hold and a slack
    // relative bound would silently re-admit the dropped-channel regression
    // (its original signature was abs_diff ≈ 3.7e-3). An absolute 1e-5 bar is a
    // genuine guard: ~1e4× the observed residual yet ~370× tighter than the
    // original miss, and robust to cross-platform PIRLS-convergence jitter.
    // gam#855's precondition, stated as what it is about.
    //
    // This used to assert `ridge == 0`, which was the right instrument while δ
    // was EXCEPTIONAL — applied only where a bare Cholesky failed. Under that
    // selector a nonzero ridge here would have meant the analytic point and the
    // FD re-solves had been rescued differently, so they would linearize
    // different systems and the comparison below would be meaningless.
    //
    // δ is now applied unconditionally (#1575/#2519: a δ chosen by a
    // Cholesky-success predicate is a function of ρ, and made the outer
    // criterion jump by 0.5·ln(1e8) = 9.21 between neighbouring ρ). A CONSTANT
    // δ satisfies the precondition rather than violating it: the analytic
    // Jacobian and both FD re-solves all linearize `XᵀWX + S_λ + δI` with the
    // same δ.
    //
    // So the assertion now checks the property directly — the three points
    // agree — instead of checking a value that only implied it. This is
    // strictly stronger: it would still catch an adaptive ridge, which
    // `== 0` would also have caught, AND it catches a δ that differs between
    // the analytic point and a perturbed one, which `== 0` would not have
    // caught had δ ever been nonzero-but-equal.
    let ridge_0 = pirls_result.ridge_passport.delta();
    assert!(
        ridge_0 == ridge_p && ridge_0 == ridge_m,
        "the IFT Jacobian and the FD re-solves must linearize the SAME system, \
         so the stabilization ridge must not change across the perturbation \
         (gam#855): analytic δ={ridge_0:.3e}, δ(+h)={ridge_p:.3e}, δ(-h)={ridge_m:.3e}"
    );
    gam_linalg::test_support::fd_checker::assert_matrix_derivativefd(
        &fd_beta.insert_axis(Axis(1)),
        &dbeta_exact.insert_axis(Axis(1)),
        1e-5,
        "sas observed-jacobian dbeta / d raw epsilon",
    );
}

#[test]
fn sas_true_score_beta_jacobian_matchesfd_at_seed19() {
    let seed = 19_u64;
    let n = 20usize;
    let x = build_tiny_design(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_non_intercept(x.ncols());

    let true_beta = array![-0.2, 0.9, -0.4];
    let eta_true = x.dot(&true_beta);
    let eps_true = 0.25;
    let ld_true = -0.20;
    let p = eta_true.mapv(|e| {
        sas_inverse_link_jet(e, eps_true, ld_true)
            .expect("finite SAS eta")
            .mu
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

    let opts = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Sas(
                crate::mixture_link::state_from_sasspec(SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                })
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "valid SAS initial state", e)),
            ),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: Some(SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        }),
        optimize_sas: true,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };

    let theta = array![0.10, 0.12, -0.18];
    let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
    assert!(effective_sas_link.is_some());
    let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
        &s_list,
        x.ncols(),
        "sas_true_score_beta_jacobian_matchesfd_at_seed19",
    );
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
        &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x.clone())),
        &penalty_specs,
    );
    let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(x.clone()),
    ));
    let mut reml_state = RemlState::newwith_offset(
        y.view(),
        x_fit,
        w.view(),
        offset.view(),
        canonical_penalties,
        x.ncols(),
        &cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("reml_state");
    let rho = theta.slice(s![..1]).to_owned();
    let (epsilon_eff, _) = sas_effective_epsilon(theta[1]);
    let sas_state = state_from_sasspec(SasLinkSpec {
        initial_epsilon: epsilon_eff,
        initial_log_delta: theta[2],
    })
    .expect("sas state");
    reml_state.set_link_states(None, Some(sas_state));

    let pirls_result = reml_state
        .obtain_eval_bundle(&rho)
        .map(|b| b.pirls_result.clone())
        .expect("pirls_result");
    let beta0 = pirls_result.beta_transformed.as_ref().clone();
    let s_transformed = pirls_result.reparam_result.s_transformed.clone();
    let ridge = pirls_result.ridge_passport.delta();
    let x_dense = match &pirls_result.x_transformed {
        DesignMatrix::Dense(x_dense) => x_dense.to_dense(),
        DesignMatrix::Sparse(_) => {
            panic!("expected dense transformed design in seed-19 SAS test")
        }
    };

    let gradient_at = |beta: &Array1<f64>| -> Array1<f64> {
        let mut eta = offset.clone();
        eta += &x_dense.dot(beta);
        let mut u = Array1::<f64>::zeros(eta.len());
        for i in 0..eta.len() {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta[i].clamp(-30.0, 30.0),
                sas_state.epsilon,
                sas_state.log_delta,
            )
            .expect("finite SAS eta");
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let aux = link_binomial_aux(i, eta[i], y[i], w[i].max(0.0), mu, 1.0 - mu)
                .expect("finite binomial row geometry");
            u[i] = aux.a1 * d1;
        }
        let mut g = -x_dense.t().dot(&u);
        g += &s_transformed.dot(beta);
        if ridge > 0.0 {
            g += &beta.mapv(|v| ridge * v);
        }
        g
    };

    let mut analytic_j = Array2::<f64>::zeros((beta0.len(), beta0.len()));
    let mut eta0 = offset.clone();
    eta0 += &x_dense.dot(&beta0);
    let mut neg_du_deta = Array1::<f64>::zeros(eta0.len());
    for i in 0..eta0.len() {
        let jets = sas_inverse_link_jetwith_param_partials(
            eta0[i].clamp(-30.0, 30.0),
            sas_state.epsilon,
            sas_state.log_delta,
        )
        .expect("finite SAS eta");
        let mu = jets.jet.mu;
        let d1 = jets.jet.d1;
        let d2 = jets.jet.d2;
        let aux = link_binomial_aux(i, eta0[i], y[i], w[i].max(0.0), mu, 1.0 - mu)
            .expect("finite binomial row geometry");
        neg_du_deta[i] = -(aux.a2 * d1 * d1 + aux.a1 * d2);
    }
    let weighted_x = &x_dense * &neg_du_deta.insert_axis(Axis(1));
    analytic_j.assign(&x_dense.t().dot(&weighted_x));
    analytic_j += &s_transformed;
    if ridge > 0.0 {
        for j in 0..analytic_j.nrows() {
            analytic_j[[j, j]] += ridge;
        }
    }

    let mut fd_j = Array2::<f64>::zeros((beta0.len(), beta0.len()));
    for j in 0..beta0.len() {
        let h = 1e-5 * (1.0 + beta0[j].abs());
        let mut beta_p = beta0.clone();
        let mut beta_m = beta0.clone();
        beta_p[j] += h;
        beta_m[j] -= h;
        let g_p = gradient_at(&beta_p);
        let g_m = gradient_at(&beta_m);
        let fd_col = (&g_p - &g_m).mapv(|v| v / (2.0 * h));
        fd_j.column_mut(j).assign(&fd_col);
    }

    gam_linalg::test_support::fd_checker::assert_matrix_derivativefd(
        &fd_j,
        &analytic_j,
        2e-3,
        "sas true beta-score jacobian at seed-19",
    );
}

#[test]
fn sas_pirlshessian_matches_true_score_jacobian_at_seed19() {
    let seed = 19_u64;
    let n = 20usize;
    let x = build_tiny_design(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_non_intercept(x.ncols());

    let true_beta = array![-0.2, 0.9, -0.4];
    let eta_true = x.dot(&true_beta);
    let eps_true = 0.25;
    let ld_true = -0.20;
    let p = eta_true.mapv(|e| {
        sas_inverse_link_jet(e, eps_true, ld_true)
            .expect("finite SAS eta")
            .mu
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

    let opts = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Sas(
                crate::mixture_link::state_from_sasspec(SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                })
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "valid SAS initial state", e)),
            ),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: Some(SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        }),
        optimize_sas: true,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };

    let theta = array![0.10, 0.12, -0.18];
    let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
    assert!(effective_sas_link.is_some());
    let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
        &s_list,
        x.ncols(),
        "sas_pirlshessian_matches_true_score_jacobian_at_seed19",
    );
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
        &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x.clone())),
        &penalty_specs,
    );
    let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(x.clone()),
    ));
    let mut reml_state = RemlState::newwith_offset(
        y.view(),
        x_fit,
        w.view(),
        offset.view(),
        canonical_penalties,
        x.ncols(),
        &cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("reml_state");
    let rho = theta.slice(s![..1]).to_owned();
    let (epsilon_eff, _) = sas_effective_epsilon(theta[1]);
    let sas_state = state_from_sasspec(SasLinkSpec {
        initial_epsilon: epsilon_eff,
        initial_log_delta: theta[2],
    })
    .expect("sas state");
    reml_state.set_link_states(None, Some(sas_state));

    let pirls_result = reml_state
        .obtain_eval_bundle(&rho)
        .map(|b| b.pirls_result.clone())
        .expect("pirls_result");
    let beta0 = pirls_result.beta_transformed.as_ref().clone();
    let s_transformed = pirls_result.reparam_result.s_transformed.clone();
    let ridge = pirls_result.ridge_passport.delta();
    let x_dense = match &pirls_result.x_transformed {
        DesignMatrix::Dense(x_dense) => x_dense.to_dense(),
        DesignMatrix::Sparse(_) => {
            panic!("expected dense transformed design in seed-19 SAS test")
        }
    };

    let mut eta0 = offset.clone();
    eta0 += &x_dense.dot(&beta0);
    let mut neg_du_deta = Array1::<f64>::zeros(eta0.len());
    for i in 0..eta0.len() {
        let jets = sas_inverse_link_jetwith_param_partials(
            eta0[i].clamp(-30.0, 30.0),
            sas_state.epsilon,
            sas_state.log_delta,
        )
        .expect("finite SAS eta");
        let mu = jets.jet.mu;
        let d1 = jets.jet.d1;
        let d2 = jets.jet.d2;
        let aux = link_binomial_aux(i, eta0[i], y[i], w[i].max(0.0), mu, 1.0 - mu)
            .expect("finite binomial row geometry");
        neg_du_deta[i] = -(aux.a2 * d1 * d1 + aux.a1 * d2);
    }
    let weighted_x = &x_dense * &neg_du_deta.insert_axis(Axis(1));
    let mut true_jacobian = x_dense.t().dot(&weighted_x);
    true_jacobian += &s_transformed;
    if ridge > 0.0 {
        for j in 0..true_jacobian.nrows() {
            true_jacobian[[j, j]] += ridge;
        }
    }

    let pht_dense = pirls_result.penalized_hessian_transformed.to_dense();
    let max_abs_diff = true_jacobian
        .iter()
        .zip(pht_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_diff <= 2e-3,
        "expected PIRLS Hessian to match the true SAS score Jacobian, got max_abs_diff={max_abs_diff:.3e}"
    );
}

// The outer-REML binomial jet carries the stable complement `1 - mu` for SAS
// links so a saturating row is not refused. A consistent fully-saturated row
// (the complement itself underflows) has finite a1/a2 with variance -> 0, the
// zero-Fisher-weight limit; an inconsistent one is a typed refusal.
#[test]
fn link_binomial_aux_carries_sas_complement_and_zero_weights_saturated_rows() {
    let aux_at = |eta: f64, yi: f64, eps: f64, log_delta: f64| {
        let mu = sas_inverse_link_jetwith_param_partials(eta, eps, log_delta)
            .expect("finite SAS eta")
            .jet
            .mu;
        let omm = sas_link_complement(eta, eps, log_delta, mu);
        (mu, omm, link_binomial_aux(0, eta, yi, 1.0, mu, omm))
    };

    // Representable complement (probit-reduced SAS, delta=1, eps=0): `mu` rounds to
    // exactly 1.0 at eta=10 but `1 - mu = Phi(-10) ≈ 7.6e-24` is a representable
    // tail, so the carried complement keeps the variance strictly positive and the
    // likelihood derivative cancellation-free.
    let (mu, omm, res) = aux_at(10.0, 1.0, 0.0, 0.0);
    let aux = res.expect("representable-complement SAS row must be representable");
    assert_eq!(mu, 1.0, "mu rounds to 1.0 at eta=10 (probit-reduced SAS)");
    assert!(
        omm > 0.0 && aux.variance > 0.0,
        "carried complement keeps variance positive; omm={omm} variance={}",
        aux.variance
    );
    assert!(aux.a1.is_finite() && aux.a2.is_finite());

    // Fully saturated CONSISTENT rows: the complement underflows to 0 (u clamps to
    // SAS_U_CLAMP, z=sinh(50)≈2.6e21, Phi(-z)=0). The row is perfectly predicted —
    // a1/a2 stay finite and variance is 0 (the caller contributes zero Fisher
    // weight, the analytic eta -> ±inf limit).
    for (eta, yi) in [(30.0, 1.0), (-30.0, 0.0)] {
        let (_, _, res) = aux_at(eta, yi, 0.0, 12.0);
        let aux = res.expect("consistent saturated SAS row must be representable");
        assert!(
            aux.a1.is_finite() && aux.a2.is_finite(),
            "consistent saturated a1/a2 must be finite at eta={eta}"
        );
        assert!(
            aux.variance.is_finite() && aux.variance >= 0.0,
            "consistent saturated variance must be finite and >= 0 at eta={eta}; got {}",
            aux.variance
        );
    }

    // Fully saturated INCONSISTENT rows (response off the predicted boundary):
    // -inf log-likelihood, a typed refusal — never silently zero-weighted.
    for (eta, yi) in [(30.0, 0.0), (-30.0, 1.0)] {
        let (_, _, res) = aux_at(eta, yi, 0.0, 12.0);
        assert!(
            matches!(
                res,
                Err(EstimationError::PirlsRowGeometryUnrepresentable { .. })
            ),
            "inconsistent saturated SAS row must be a typed refusal at eta={eta}; got {res:?}"
        );
    }
}

/// Build a Beta-precision (φ estimated) outer state on a deterministic
/// fixture: `logit(μ) = 1.6·(x − ½)`, three design columns `[1, x, x²]`, one
/// dense penalty on the two non-intercept columns.
fn beta_precision_anchor_state<'a>(
    y: &'a Array1<f64>,
    w: &'a Array1<f64>,
    x: &Array2<f64>,
    cfg: &'a RemlConfig,
) -> RemlState<'a> {
    let p = x.ncols();
    let offset = Array1::<f64>::zeros(y.len());
    let mut s = Array2::<f64>::zeros((p, p));
    s[[1, 1]] = 1.0;
    s[[2, 2]] = 1.0;
    let canonical = gam_terms::construction::canonicalize_penalty_specs(
        &[crate::estimate::PenaltySpec::Dense(s)],
        &[1],
        p,
        "beta_precision_anchor_state",
    )
    .map(|(canonical, _)| canonical)
    .expect("canonicalize the anchor fixture penalty");
    RemlState::newwith_offset(
        y.view(),
        x.clone(),
        w.view(),
        offset.view(),
        canonical,
        p,
        cfg,
        Some(vec![1]),
        None,
        None,
    )
    .expect("build the Beta-precision anchor state")
}

fn beta_precision_anchor_fixture() -> (Array1<f64>, Array1<f64>, Array2<f64>, RemlConfig) {
    let n = 60usize;
    let mut y = Array1::<f64>::zeros(n);
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let xi = i as f64 / (n as f64 - 1.0);
        let eta = 1.6 * (xi - 0.5);
        let mu = 1.0 / (1.0 + (-eta).exp());
        // Deterministic alternating perturbation: enough conditional spread for
        // the Pearson precision to be a genuinely data-driven quantity.
        let wiggle = if i % 2 == 0 { 0.06 } else { -0.06 };
        y[i] = (mu + wiggle).clamp(0.02, 0.98);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = xi;
        x[[i, 2]] = xi * xi;
    }
    let w = Array1::<f64>::ones(n);
    let phi = 8.0;
    let likelihood = GlmLikelihoodSpec {
        spec: LikelihoodSpec::new(
            ResponseFamily::Beta { phi },
            InverseLink::Standard(StandardLink::Logit),
        ),
        scale: gam_problem::LikelihoodScaleMetadata::EstimatedBetaPhi { phi },
    };
    (y, w, x, RemlConfig::external(likelihood, 1e-8, false))
}

#[test]
fn cubature_pirls_uses_lambda_search_frozen_beta_precision_2632() {
    // Sigma-point fits are evaluations of the same profiled criterion as the
    // production lambda search. They must therefore inherit its frozen
    // likelihood nuisance parameter instead of silently re-profiling phi from
    // each off-trajectory point's seed eta.
    let (y, w, x, cfg) = beta_precision_anchor_fixture();
    let state = beta_precision_anchor_state(&y, &w, &x, &cfg);
    let frozen_phi = 11.0_f64;
    state
        .frozen_beta_phi
        .store(frozen_phi.to_bits(), Ordering::Relaxed);

    let result = state
        .execute_pirls_stateless_for_cubature(&array![0.0], None)
        .expect("the Beta cubature sigma-point fit must converge");
    let (realized_phi, estimated) = match result
        .likelihood
        .resolved_scale()
        .expect("the cubature result must retain valid Beta scale metadata")
    {
        gam_problem::ResolvedLikelihoodScale::BetaPrecision {
            precision,
            estimated,
        } => (precision, estimated),
        other => panic!("expected Beta precision after cubature PIRLS, got {other:?}"),
    };

    assert!(
        !estimated,
        "cubature PIRLS re-enabled Beta-precision estimation instead of using \
         the lambda-search freeze (#2632)"
    );
    assert_eq!(
        realized_phi.value().to_bits(),
        frozen_phi.to_bits(),
        "cubature PIRLS changed the lambda-search-frozen Beta precision: \
         expected {frozen_phi:.17e}, got {:.17e}",
        realized_phi.value()
    );
}

#[test]
fn lambda_search_nuisance_freeze_is_a_function_of_data_and_spec_alone_2363() {
    // #2363. The λ-search holds the estimated nuisance ψ fixed so that
    // `F(ρ) = REML(ρ, ψ)` is stationary in ρ (#1074 / #1477 / #2369). Which
    // value gets frozen therefore DEFINES the criterion the outer search
    // minimizes — so if it is captured at whatever solve the persistent
    // warm-start cache happened to steer the search into first, a cold machine
    // and a warm machine minimize different criteria and legitimately report
    // different fits (measured: the same Beta fit at REML −6.382e2 cold and
    // −7.408e2 warm).
    //
    // The contract this pins is the repair: ψ is anchored at the symmetric
    // reference ρ = 0 on a state with no warm start attached, so it is a
    // function of (data, model spec) alone — BITWISE identical no matter what
    // seed β or heuristic λ a caller (or a cache) supplies.
    let (y, w, x, cfg) = beta_precision_anchor_fixture();
    let resolved = cfg
        .likelihood
        .resolved_scale()
        .expect("the fixture declares an estimated Beta precision");
    assert!(
        matches!(
            resolved,
            gam_problem::ResolvedLikelihoodScale::BetaPrecision {
                estimated: true,
                ..
            }
        ),
        "fixture precondition: the freeze under test only exists for an ESTIMATED Beta precision"
    );
    let seed_config = external_reml_seed_config(1, LinkFunction::Logit);

    let pristine = beta_precision_anchor_state(&y, &w, &x, &cfg);
    freeze_lambda_search_nuisance_at_canonical_anchor(&pristine, &resolved, 1, None, &seed_config)
        .expect("the anchor must succeed on a pristine state");
    let anchored_bits = pristine.frozen_beta_phi.load(Ordering::Relaxed);
    assert_ne!(
        anchored_bits, 0,
        "the anchor must actually freeze a precision; an unfrozen λ-search re-profiles φ from \
         every trial's warm-start η and the outer criterion drifts with ρ"
    );
    let anchored_phi = f64::from_bits(anchored_bits);
    assert!(
        anchored_phi.is_finite() && anchored_phi > 0.0,
        "the anchored Beta precision must be a finite positive value; got {anchored_phi}"
    );

    // A caller-supplied warm β and a different heuristic λ are exactly what the
    // persistent cache donates. Neither may move the frozen value by one ulp.
    let seeded = beta_precision_anchor_state(&y, &w, &x, &cfg);
    let donated_beta = array![0.35, -1.7, 2.4];
    seeded.setwarm_start_original_beta(Some(donated_beta.view()));
    assert!(
        seeded.current_original_basis_beta().is_some(),
        "precondition: the donated warm β is installed before the anchor runs"
    );
    freeze_lambda_search_nuisance_at_canonical_anchor(
        &seeded,
        &resolved,
        1,
        Some(&[3.0]),
        &seed_config,
    )
    .expect("the anchor must succeed regardless of what a caller donated");
    assert_eq!(
        seeded.frozen_beta_phi.load(Ordering::Relaxed),
        anchored_bits,
        "the λ-search nuisance freeze must be BITWISE independent of the donated warm β and \
         heuristic λ: it defines the outer criterion, and a criterion that moves with cache \
         state is the #2363 defect"
    );

    // The predictor the anchor hands the search must be invariant too. It is
    // deliberately left in place (it is what suppresses the cache-dependent
    // persistent β restore), so it has to be the SAME β on both machines.
    let pristine_seed = pristine
        .current_original_basis_beta()
        .expect("the anchor leaves its own β as the search's warm start");
    let seeded_seed = seeded
        .current_original_basis_beta()
        .expect("the anchor leaves its own β as the search's warm start");
    assert_eq!(
        pristine_seed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        seeded_seed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "the predictor the anchor hands the search must not depend on the donated warm β either"
    );

    // And it must refuse to run at all once the on-disk session is open: from
    // that point the inner solve reloads the cached β, so an anchor taken there
    // would be cache-dependent again.
    let attached = beta_precision_anchor_state(&y, &w, &x, &cfg);
    let directory = tempfile::tempdir().expect("create explicit warm-start root");
    attached.attach_persistent_warm_start_store(
        crate::persistent_warm_start::configured_store(directory.path().join("warm")),
    );
    let refusal = freeze_lambda_search_nuisance_at_canonical_anchor(
        &attached,
        &resolved,
        1,
        None,
        &seed_config,
    );
    assert!(
        matches!(refusal, Err(EstimationError::InvalidInput(_))),
        "anchoring after the persistent layer is attached must be a typed refusal, not a \
         silently cache-dependent freeze; got {refusal:?}"
    );
    assert_eq!(
        attached.frozen_beta_phi.load(Ordering::Relaxed),
        0,
        "a refused anchor must not leave a partially-established freeze behind"
    );
}

/// One arm of the cold/prime/warm cache-invariance matrix: a complete
/// production fit through the external-design entry point, with the on-disk
/// persistent warm-start layer either structurally disabled or engaged.
fn cache_invariance_arm(
    family: &LikelihoodSpec,
    y: &Array1<f64>,
    w: &Array1<f64>,
    x: &Array2<f64>,
    persistent_warm_start_store: Option<gam_runtime::warm_start::ConfiguredWarmStartStore>,
) -> ExternalOptimResult {
    let offset = Array1::<f64>::zeros(y.len());
    let s_list: Vec<PenaltySpec> = one_penalty_non_intercept(x.ncols())
        .into_iter()
        .map(PenaltySpec::Dense)
        .collect();
    let opts = ExternalOptimOptions {
        family: family.clone(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: persistent_warm_start_store.clone(),
    };
    optimize_external_designwith_heuristic_lambdas_andwarm_start(
        y.view(),
        w.view(),
        x.clone(),
        offset.view(),
        s_list,
        None,
        None,
        &opts,
    )
    .unwrap_or_else(|error| {
        panic!(
            "a fit must succeed regardless of cache state (persist={}): \
             {error:?}",
            persistent_warm_start_store.is_some()
        )
    })
}

/// Signed ulp distance between two finite `f64`s, using the standard
/// monotone-ordering trick so the count is meaningful across zero.
fn ulp_distance(left: f64, right: f64) -> i128 {
    let order = |value: f64| -> i128 {
        let bits = value.to_bits() as i64;
        if bits < 0 {
            (i64::MIN - bits) as i128
        } else {
            bits as i128
        }
    };
    order(left) - order(right)
}

/// Describe the first coordinate at which two arms disagree BITWISE, or `None`
/// when every coordinate is bit-identical. The description carries the ulp
/// distance so a regression report says how far the value moved, not just that
/// it moved.
fn first_bitwise_gap(cold: &[f64], warm: &[f64]) -> Option<String> {
    if cold.len() != warm.len() {
        return Some(format!(
            "layout differs: cold len={} warm len={}",
            cold.len(),
            warm.len()
        ));
    }
    cold.iter()
        .zip(warm.iter())
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(index, (a, b))| {
            format!(
                "coordinate {index}: cold={a:.17e} warm={b:.17e} ulps={} |Δ|={:.3e}",
                ulp_distance(*a, *b),
                (a - b).abs()
            )
        })
}

/// Deterministic `n`-row covariate grid shared by the estimated-nuisance
/// cache-invariance fixtures.
fn nuisance_invariance_design(n: usize) -> (Array2<f64>, Vec<f64>) {
    let mut x = Array2::<f64>::zeros((n, 4));
    let mut grid = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = 2.0 * t - 1.0;
        x[[i, 2]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).cos();
        grid.push(t);
    }
    (x, grid)
}

#[test]
fn estimated_nuisance_fits_land_in_the_same_place_cold_and_warm_2363() {
    // #2363, the OUTCOME half of the contract the mechanism test above pins:
    // a fit must be a function of (data, model spec) alone. The persistent
    // warm-start cache may change how fast the search gets there — it donates a
    // ρ seed and a warm β — but never where it lands.
    //
    // The three families here are the ones whose outer criterion carries an
    // ESTIMATED nuisance frozen across the λ-search (Gamma shape, Tweedie φ,
    // Beta precision). Freezing a value the cache had steered the search into
    // made cold and warm machines minimize different criteria; this asserts
    // they now minimize the same one and report the same fit.
    //
    // Each family runs against one test-owned empty root: cold arms carry no
    // capability, the priming arm writes through the explicit capability, and
    // the warm arm reuses the same clone-shared store. No machine history or
    // concurrent lane can enter the attribution.
    //
    // The comparison is BITWISE on the criterion, on β and on the certified
    // log-λ. Approximate agreement is the wrong contract here: the defect this
    // guards agreed to three or four digits on two of the three families and was
    // still a different objective, so any tolerance wide enough to absorb solver
    // wobble is also wide enough to absorb the next instance of the bug.
    let n = 160usize;
    let (x, grid) = nuisance_invariance_design(n);
    let w = Array1::<f64>::ones(n);

    let gamma_y = Array1::from_iter(grid.iter().enumerate().map(|(i, t)| {
        let mu = (0.3 + 0.9 * (2.0 * std::f64::consts::PI * t).sin()).exp();
        mu * if i % 3 == 0 { 0.72 } else { 1.18 }
    }));
    let tweedie_y = Array1::from_iter(grid.iter().enumerate().map(|(i, t)| {
        let mu = (0.1 + 1.1 * (2.0 * std::f64::consts::PI * t).sin()).exp();
        if i % 5 == 0 {
            0.0
        } else {
            mu * if i % 2 == 0 { 0.62 } else { 1.31 }
        }
    }));
    let beta_y = Array1::from_iter(grid.iter().enumerate().map(|(i, t)| {
        let eta = 1.8 * (2.0 * std::f64::consts::PI * t).sin();
        let mu = 1.0 / (1.0 + (-eta).exp());
        (mu + if i % 2 == 0 { 0.07 } else { -0.07 }).clamp(0.02, 0.98)
    }));

    let cases: [(&str, LikelihoodSpec, &Array1<f64>); 3] = [
        (
            "gamma_estimated_shape",
            LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            ),
            &gamma_y,
        ),
        (
            "tweedie_estimated_phi",
            LikelihoodSpec::new(
                ResponseFamily::Tweedie { p: 1.5 },
                InverseLink::Standard(StandardLink::Log),
            ),
            &tweedie_y,
        ),
        (
            "beta_estimated_precision",
            LikelihoodSpec::new(
                ResponseFamily::Beta { phi: 8.0 },
                InverseLink::Standard(StandardLink::Logit),
            ),
            &beta_y,
        ),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (name, family, y) in &cases {
        let cache_directory = tempfile::tempdir().expect("create private cache root");
        let cache = crate::persistent_warm_start::configured_store(
            cache_directory.path().join("warm"),
        );
        let cold = cache_invariance_arm(family, y, &w, &x, None);
        // CONTROL, before the cache is involved at all: a second cold arm.
        //
        // Every difference below is attributed to the cache, and that
        // attribution is only sound if the fit is bitwise reproducible with the
        // cache held fixed. If cold-vs-cold already differs, the subject is not
        // cache invariance at all and no amount of work on the cache would
        // close it. Measured here rather than assumed, because this test's
        // whole claim is an attribution.
        let failures_before_case = failures.len();
        let cold_again = cache_invariance_arm(family, y, &w, &x, None);
        if let Some(gap) = first_bitwise_gap(
            cold.beta.as_slice().expect("contiguous cold β"),
            cold_again.beta.as_slice().expect("contiguous second cold β"),
        ) {
            failures.push(format!(
                "[{name}] CONTROL FAILED — two COLD fits differ, so the differences below \
                 are not attributable to the cache: {gap}"
            ));
        }
        if let Some(gap) = first_bitwise_gap(
            cold.log_lambdas.as_slice().expect("contiguous cold log-λ"),
            cold_again
                .log_lambdas
                .as_slice()
                .expect("contiguous second cold log-λ"),
        ) {
            failures.push(format!(
                "[{name}] CONTROL FAILED — two COLD fits' log-λ differ: {gap}"
            ));
        }
        drop(cache_invariance_arm(
            family,
            y,
            &w,
            &x,
            Some(cache.clone()),
        ));
        let warm = cache_invariance_arm(
            family,
            y,
            &w,
            &x,
            Some(cache.clone()),
        );

        if cold.outer_converged != warm.outer_converged {
            failures.push(format!(
                "[{name}] the convergence verdict depends on cache state: cold={} warm={}",
                cold.outer_converged, warm.outer_converged
            ));
        }
        // BITWISE, not "close". Numerically close state is not interchangeable
        // provenance for a profiled objective: the whole failure mode this issue
        // documents is two runs that agreed to several digits and still sat on
        // different criteria. An approximate bound would have to be set above
        // whatever wobble the current solver happens to produce, and would then
        // stop measuring the invariant and start measuring the wobble. The
        // reports carry the ulp distance so a regression says how far it moved.
        match (cold.reml_score, warm.reml_score) {
            (Some(cold_score), Some(warm_score)) => {
                if let Some(gap) = first_bitwise_gap(
                    std::slice::from_ref(&cold_score),
                    std::slice::from_ref(&warm_score),
                ) {
                    failures.push(format!(
                        "[{name}] the outer criterion depends on cache state: {gap}"
                    ));
                }
            }
            // Presence itself is part of the invariant: a fit that has a
            // criterion cold and none warm (or the reverse) is exactly the
            // cache-dependent provenance this guard exists to catch.
            (Some(_), None) | (None, Some(_)) => {
                let (cold_score, warm_score) = (cold.reml_score, warm.reml_score);
                failures.push(format!(
                    "[{name}] the outer criterion's EXISTENCE depends on cache state: \
                     cold={cold_score:?} warm={warm_score:?}"
                ));
            }
            // Neither arm reported a criterion: the absence agrees, so there is
            // no cache-dependent provenance to report here.
            (None, None) => {}
        }
        if let Some(gap) = first_bitwise_gap(
            cold.beta.as_slice().expect("contiguous cold β"),
            warm.beta.as_slice().expect("contiguous warm β"),
        ) {
            failures.push(format!(
                "[{name}] the coefficients depend on cache state: {gap}"
            ));
        }
        if let Some(gap) = first_bitwise_gap(
            cold.log_lambdas.as_slice().expect("contiguous cold log-λ"),
            warm.log_lambdas.as_slice().expect("contiguous warm log-λ"),
        ) {
            failures.push(format!(
                "[{name}] the certified log-λ depends on cache state: {gap}"
            ));
        }
        // PROVENANCE for whatever differed above: did the warm arm SEARCH from
        // the seeded optimum, or re-certify it in place?
        //
        // The cache hit logs `action=resume-and-recertify` and installs the
        // prior fit's ρ as `initial_rho` with `screen_initial_rho = false`
        // (`rho_optimizer/run.rs`, the `CacheSeedDecision::ExactFinal` arm),
        // plus the prior β as an inner seed. If that point is already
        // certified, the outer search has nothing to do and must return it
        // unchanged -- so a warm arm reporting ANY outer iterations is
        // re-deriving a point it was handed, and the ~1e-9 drift in the
        // certified log-λ follows from that rather than from the cache
        // carrying wrong state.
        //
        // Two readings, opposite fixes, and the iteration count separates them:
        // warm iterations == 0 means the seed was returned and the difference
        // is elsewhere; warm iterations > 0 means it searched. Reported only
        // when this case actually failed, so a green run stays quiet.
        if failures.len() > failures_before_case {
            failures.push(format!(
                "[{name}] PROVENANCE: cold iters={} |g|={:.6e} | warm iters={} |g|={:.6e} \
                 (a warm arm that re-certifies a seeded optimum in place must report 0 \
                 outer iterations; anything else re-derived a point it was given)",
                cold.iterations, cold.finalgrad_norm, warm.iterations, warm.finalgrad_norm,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "a warm cache changed WHERE the fit landed, not just how fast it got there:\n{}",
        failures.join("\n")
    );
}

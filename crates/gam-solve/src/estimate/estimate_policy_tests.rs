use super::evaluation::{
    sas_effective_epsilon, sas_effective_epsilon_second, sas_log_delta_edge_barriercostgrad,
    sas_log_delta_edge_barriercostgradhess,
};
use super::external_options::resolve_external_family;
use super::optimizer::external_reml_seed_config;
use super::penalty::REML_SEED_SCREENING_RHO_CAP;
use super::prefit::{
    PrefitRegularityDiagnostic, detect_prefit_binomial_single_column_separation_in_design,
    detect_prefit_unpenalized_rank_deficiency_in_design, reject_prefit_binomial_separation,
    reject_prefit_unpenalized_rank_deficiency,
};
use super::reml::hyper::link_binomial_aux;
use super::*;
use crate::mixture_link::{sas_inverse_link_jet, sas_inverse_link_jetwith_param_partials};
use gam_linalg::utils::{StableSolver, max_abs_diag};
use gam_problem::{
    InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily, SeedRiskProfile, StandardLink,
};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn gaussian_external_reml_seeds_over_smoothing_safety_net() {
    // #1074: low-dimensional Gaussian REML ranks several deterministic candidate
    // basins (`max_seeds > seed_budget`) AND fully solves TWO of them — the
    // flexible anchor (slot 0) and the over-smoothing probe (slot 1) — so
    // lowest-cost keep-best can escape the flexible over-fit basin a weak-signal
    // over-rich fit otherwise rails into. The probe is an absolute high-λ start.
    let cfg = external_reml_seed_config(2, LinkFunction::Identity);
    assert_eq!(cfg.risk_profile, SeedRiskProfile::Gaussian);
    assert!(
        cfg.max_seeds > cfg.seed_budget,
        "Gaussian REML should rank deterministic candidate basins before startup"
    );
    assert_eq!(
        cfg.seed_budget, 2,
        "Gaussian REML must fully solve both the flexible anchor and the over-smoothing \
         probe so keep-best can reject an over-fit basin"
    );
    assert_eq!(
        cfg.over_smoothing_probe_rho,
        Some(8.0),
        "Gaussian REML must seed an absolute high-λ over-smoothing probe (#1074)"
    );
}

#[test]
fn high_dimensional_gaussian_external_reml_keeps_over_smoothing_safety_net() {
    // #1074: a MULTI-TERM Gaussian model (e.g. `s(long,lat,bs="tp") + s(depth)`,
    // four penalty blocks) lands at/above the screening cap. It must STILL get
    // the over-smoothing safety net — a budget-2 multi-start with the high-λ
    // probe — or it descends into the flexible basin and over-fits the weak
    // signal (the #1074 quakes edf≈104 vs mgcv≈15 failure). The lattice stays
    // minimal (anchor + global shifts + probe, no exploratory seeds) to honour
    // the cap's perf intent, but the budget-2 keep-best coverage is preserved.
    let cfg = external_reml_seed_config(REML_SEED_SCREENING_RHO_CAP, LinkFunction::Identity);
    assert_eq!(cfg.risk_profile, SeedRiskProfile::Gaussian);
    assert!(
        cfg.max_seeds > cfg.seed_budget,
        "high-dimensional Gaussian REML must still rank the flexible and over-smoothing basins"
    );
    assert_eq!(
        cfg.seed_budget, 2,
        "high-dimensional Gaussian REML must fully solve both basins so keep-best can \
         reject the multi-term over-fit (#1074)"
    );
    assert_eq!(
        cfg.over_smoothing_probe_rho,
        Some(8.0),
        "the over-smoothing probe must survive past the screening cap for Gaussian (#1074)"
    );
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
    UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: array![0.25, -0.5],
            role: BlockRole::Mean,
            edf: 1.5,
            lambdas: array![0.2, 0.8],
        }],
        log_lambdas: array![0.2_f64.max(1e-300).ln(), 0.8_f64.max(1e-300).ln()],
        lambdas: array![0.2, 0.8],
        likelihood_family: Some(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        )),
        likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
        log_likelihood_normalization: LogLikelihoodNormalization::Full,
        log_likelihood: -1.2,
        deviance: 2.4,
        reml_score: 0.7,
        stable_penalty_term: 0.3,
        penalized_objective: 2.2,
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
            penalized_hessian: array![[2.0, 0.1], [0.1, 3.0]].into(),
            working_weights: array![1.0, 0.5, 0.75],
            working_response: array![0.1, 0.2, 0.3],
            reparam_qs: Some(array![[1.0, 0.0], [0.0, 1.0]]),
            dispersion: Dispersion::Known(1.0),
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
            penalized_hessian: array![[2.0, 0.1], [0.1, 3.0]].into(),
            working_weights: array![1.0, 0.5, 0.75],
            working_response: array![0.1, 0.2, 0.3],
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
                },
                hessian_psd: Some(true),
                lambdas_railed: Vec::new(),
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
    assert_eq!(fit.dispersion(), Some(Dispersion::Known(1.0)));
    assert_eq!(fit.dispersion_phi(), 1.0);

    // Deployment-saved models drop `inference` (see `core_saved_fit_result`,
    // which stores `inference: None`). `dispersion()` is then `None`, but
    // `dispersion_phi()` must still recover the Gaussian scale φ̂ = σ̂² from
    // the always-serialized `standard_deviation`. This is the code path the
    // unseen-level prior variance (#674) relies on.
    let mut stripped = fit.clone();
    stripped.inference = None;
    assert!(stripped.dispersion().is_none());
    let expected_phi = stripped.standard_deviation * stripped.standard_deviation;
    assert!(
        (stripped.dispersion_phi() - expected_phi).abs() < 1e-12,
        "fallback φ̂ should equal σ̂² = {expected_phi}, got {}",
        stripped.dispersion_phi()
    );

    // A fixed-scale family (Poisson) keeps φ̂ = 1 on the fallback path even
    // with a non-unit residual summary, so the unseen-level prior collapses
    // to the historical 1/λ for those families.
    let mut poisson = stripped.clone();
    poisson.likelihood_family = Some(LikelihoodSpec::new(
        ResponseFamily::Poisson,
        InverseLink::Standard(StandardLink::Log),
    ));
    poisson.standard_deviation = 2.7;
    assert_eq!(poisson.dispersion_phi(), 1.0);
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
fn unified_fit_validation_accepts_persisted_log_lambda_roundoff() {
    let mut fit = decode_invariant_test_fit();
    fit.log_lambdas[0] += 5e-14;
    assert!(
        fit.validate_numeric_finiteness().is_ok(),
        "sub-ulp persisted log-lambda roundoff should remain valid"
    );
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
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
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
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
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
                let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
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
    gam_test_support::assert_matrix_derivativefd(
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
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
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
        if pirls_result.ridge_used > 0.0 {
            for d in 0..j.nrows() {
                j[[d, d]] += pirls_result.ridge_used;
            }
        }
        j
    };
    let stable_solver = StableSolver::new("sas dbeta exact test");
    let mut dbeta_exact = stable_solver
        .solvevectorwithridge_retries(
            &score_beta_jacobian,
            &rhs,
            max_abs_diag(&score_beta_jacobian) * 1e-12,
        )
        .expect("observed-jacobian solve for dbeta");
    dbeta_exact *= d_eps_d_raw;

    let fd_h = 1e-4 * (1.0 + theta[1].abs());
    let beta_at = |raw_eps: f64| -> Array1<f64> {
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
        pirls.beta_transformed.as_ref().clone()
    };
    let beta_p = beta_at(theta[1] + fd_h);
    let beta_m = beta_at(theta[1] - fd_h);
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
    assert_eq!(
        pirls_result.ridge_used, 0.0,
        "well-conditioned n=20 SAS fit must take no stabilization ridge; \
         a nonzero ridge would mean the IFT Jacobian and the FD re-solve no \
         longer linearize the same system (gam#855)"
    );
    gam_test_support::assert_matrix_derivativefd(
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
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
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
    let ridge = pirls_result.ridge_used;
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
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
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
        let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
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

    gam_test_support::assert_matrix_derivativefd(
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
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
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
    let ridge = pirls_result.ridge_used;
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
        let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
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

#[test]
fn link_binomial_aux_stay_finite_for_saturated_sas_probabilities() {
    let saturated_cases = [
        (
            0.0,
            sas_inverse_link_jetwith_param_partials(-30.0, 0.0, 12.0)
                .expect("finite SAS eta")
                .jet
                .mu,
        ),
        (
            1.0,
            sas_inverse_link_jetwith_param_partials(30.0, 0.0, 12.0)
                .expect("finite SAS eta")
                .jet
                .mu,
        ),
    ];
    for (yi, mu) in saturated_cases {
        let aux = link_binomial_aux(yi, 1.0, mu);
        assert!(aux.a1.is_finite(), "a1 must be finite for yi={yi} mu={mu}");
        assert!(aux.a2.is_finite(), "a2 must be finite for yi={yi} mu={mu}");
        assert!(
            aux.variance.is_finite() && aux.variance > 0.0,
            "variance must be finite and positive for yi={yi} mu={mu}"
        );
    }
}

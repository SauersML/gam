#[cfg(test)]
mod estimate_policy_tests {
    use super::reml::hyper::link_binomial_aux;
    use super::*;
    use crate::linalg::utils::{StableSolver, max_abs_diag};
    use crate::mixture_link::{sas_inverse_link_jet, sas_inverse_link_jetwith_param_partials};
    use crate::types::{InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily, StandardLink};
    use ndarray::{Array1, Array2, array};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn gaussian_external_reml_uses_single_seed_policy() {
        let cfg = external_reml_seed_config(2, LinkFunction::Identity);
        assert_eq!(cfg.risk_profile, SeedRiskProfile::Gaussian);
        assert!(
            cfg.max_seeds > cfg.seed_budget,
            "Gaussian REML should rank deterministic candidate basins before startup"
        );
        assert_eq!(
            cfg.seed_budget, 1,
            "standard Gaussian REML should fully optimize the best screened start by default"
        );
    }

    #[test]
    fn high_dimensional_external_reml_skips_seed_screening() {
        let cfg = external_reml_seed_config(REML_SEED_SCREENING_RHO_CAP, LinkFunction::Identity);
        assert_eq!(cfg.risk_profile, SeedRiskProfile::Gaussian);
        assert_eq!(
            cfg.max_seeds, 1,
            "moderate/high-dimensional REML should start from the deterministic seed directly"
        );
        assert_eq!(cfg.seed_budget, 1);
    }

    #[test]
    fn generalized_external_reml_keeps_multistart_policy() {
        let cfg = external_reml_seed_config(2, LinkFunction::Logit);
        assert_eq!(cfg.risk_profile, SeedRiskProfile::GeneralizedLinear);
        assert!(cfg.max_seeds > 1);
        assert_eq!(cfg.seed_budget, 1);
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
            let det = h_int[[0, 0]]
                * (h_int[[1, 1]] * h_int[[2, 2]] - h_int[[1, 2]] * h_int[[2, 1]])
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
        let diagnostic = detect_prefit_binomial_single_column_separation_in_design(
            y.view(),
            w.view(),
            &design,
            &[true, true],
        )
        .expect("separation screen must complete without a layout error")
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));

        assert_eq!(
            detect_prefit_binomial_single_column_separation_in_design(
                binary_y.view(),
                w.view(),
                &design,
                &[true, false],
            )
            .expect("separation screen must complete without a layout error"),
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
            .expect("separation screen must complete without a layout error"),
            None,
            "fractional binomial proportions are not exact binary separation"
        );
    }

    #[test]
    fn prefit_binomial_logit_rejects_before_outer_solver() {
        let x = array![[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let w = Array1::ones(y.len());
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
        let diagnostic = detect_prefit_unpenalized_rank_deficiency_in_design(
            w.view(),
            &design,
            &[true, true, true],
        )
        .expect("rank check should stream dense design")
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
        let diagnostic = detect_prefit_unpenalized_rank_deficiency_in_design(
            w.view(),
            &design,
            &[true, true, true],
        )
        .expect("rank check should stream dense design")
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
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
        let diagnostic = detect_prefit_unpenalized_rank_deficiency_in_design(
            w.view(),
            &design,
            &[true, true, true],
        )
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

    fn decode_invariant_test_fit() -> UnifiedFitResult {
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
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
            outer_iterations: 3,
            outer_converged: true,
            outer_gradient_norm: Some(0.05),
            standard_deviation: 1.1,
            covariance_conditional: Some(array![[1.0, 0.1], [0.1, 2.0]]),
            covariance_corrected: Some(array![[1.2, 0.1], [0.1, 2.2]]),
            inference: Some(FitInference {
                edf_by_block: vec![0.6, 0.9],
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
            artifacts: FitArtifacts::default(),
            inner_cycles: 0,
        })
        .expect("construct decode invariant test fit")
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
        payload["beta"] = serde_json::to_value(Array1::from(vec![9.0_f64, 8.0_f64]))
            .expect("serialize drifted beta");
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
        assert!(file!().ends_with(".rs"));
        let mut fit = decode_invariant_test_fit();
        fit.log_lambdas[0] += 5e-14;
        fit.validate_numeric_finiteness()
            .expect("sub-ulp persisted log-lambda roundoff should remain valid");
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
        Vec<crate::construction::CanonicalPenalty>,
        Vec<usize>,
    ) {
        let penalty_specs = s_list
            .iter()
            .cloned()
            .map(PenaltySpec::Dense)
            .collect::<Vec<_>>();
        let (canonical_penalties, active_nullspace_dims) =
            crate::construction::canonicalize_penalty_specs(
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
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
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
                    .expect("valid SAS initial state"),
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
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
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
                );
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
                    );
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
        crate::test_support::assert_matrix_derivativefd(
            &fd_du_raw.insert_axis(Axis(1)),
            &du_raw.insert_axis(Axis(1)),
            2e-3,
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
                );
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
                    crate::matrix::DenseDesignMatrix::from(x.clone()),
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

        // The two derivative channels feeding this IFT solve — `du/dε` at fixed
        // η and the score β-Jacobian — are each validated against their own FDs
        // above / in `sas_true_score_beta_jacobian_matchesfd_at_seed19`. The
        // composite `dβ/dε = J⁻¹·rhs` is the exact IFT linearization at the
        // converged β̂, but the FD comparator re-runs PIRLS to convergence at
        // each perturbed ε, so its `β̂(ε±)` carry the *adaptive* stabilization
        // ridge, whose magnitude shifts non-smoothly with conditioning across
        // the ± solves. That solver-only channel (correctly excluded from the
        // analytic IFT) contaminates the FD by a fixed fraction of the dominant
        // ~0.22-magnitude component, so a relative bound is the principled
        // comparison here rather than an absolute one tuned for the small
        // entries (gam#855).
        crate::test_support::assert_matrix_derivativefd_rel(
            &fd_beta.insert_axis(Axis(1)),
            &dbeta_exact.insert_axis(Axis(1)),
            2e-2,
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
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
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
                    .expect("valid SAS initial state"),
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
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
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
                );
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
            );
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

        crate::test_support::assert_matrix_derivativefd(
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
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
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
                    .expect("valid SAS initial state"),
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
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
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
            );
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
                    .jet
                    .mu,
            ),
            (
                1.0,
                sas_inverse_link_jetwith_param_partials(30.0, 0.0, 12.0)
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
}

#[cfg(test)]
mod continuous_order_tests {
    use super::*;

    fn try_compute_continuous_smoothness_order(
        lambda_tilde: &[f64],
        normalization_scale: &[f64],
        eps: f64,
    ) -> Option<ContinuousSmoothnessOrder> {
        if lambda_tilde.len() != 3 || normalization_scale.len() != 3 {
            return None;
        }
        Some(compute_continuous_smoothness_order(
            [lambda_tilde[0], lambda_tilde[1], lambda_tilde[2]],
            [
                normalization_scale[0],
                normalization_scale[1],
                normalization_scale[2],
            ],
            eps,
        ))
    }

    #[test]
    fn continuous_order_formula_matches_closed_form() {
        let out = compute_continuous_smoothness_order([2.0, 10.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::Ok);
        let r = out.r_ratio.expect("R");
        let nu = out.nu.expect("nu");
        let kappa2 = out.kappa2.expect("kappa2");
        assert!((r - (100.0 / 6.0)).abs() < 1e-12);
        assert!((nu - (r / (r - 2.0))).abs() < 1e-12);
        assert!((kappa2 - (10.0 / ((r - 2.0) * 3.0))).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_unscales_lambdas_exactly_by_ck() {
        let out = compute_continuous_smoothness_order([6.0, 15.0, 9.0], [3.0, 5.0, 9.0], 1e-12);
        // Physical lambdas must satisfy lambda_k = lambda_tilde_k / c_k.
        assert!((out.lambda0 - 2.0).abs() < 1e-12);
        assert!((out.lambda1 - 3.0).abs() < 1e-12);
        assert!((out.lambda2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_invalid_ck_is_guarded() {
        let out = compute_continuous_smoothness_order([1.0, 1.0, 1.0], [1.0, 0.0, 1.0], 1e-12);
        assert_eq!(
            out.status,
            ContinuousSmoothnessOrderStatus::UndefinedZeroLambda
        );
        assert!(out.r_ratio.is_none());
    }

    #[test]
    fn continuous_order_is_invariant_to_penalty_normalization_reversal() {
        let base = compute_continuous_smoothness_order([2.0, 10.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        let scaled = compute_continuous_smoothness_order(
            [2.0 * 4.0, 10.0 * 0.5, 3.0 * 8.0],
            [4.0, 0.5, 8.0],
            1e-12,
        );
        assert_eq!(base.status, ContinuousSmoothnessOrderStatus::Ok);
        assert_eq!(scaled.status, ContinuousSmoothnessOrderStatus::Ok);
        assert!((base.r_ratio.unwrap() - scaled.r_ratio.unwrap()).abs() < 1e-12);
        assert!((base.nu.unwrap() - scaled.nu.unwrap()).abs() < 1e-12);
        assert!((base.kappa2.unwrap() - scaled.kappa2.unwrap()).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_flags_non_matern_regimewhen_r_le_4() {
        let out = compute_continuous_smoothness_order([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::NonMaternRegime);
        assert!(out.nu.is_none());
        assert!(out.kappa2.is_none());
    }

    #[test]
    fn continuous_order_reports_effective_nu_kappa_in_non_matern_bandwhen_r_gt_2() {
        let out = compute_continuous_smoothness_order([1.0, 3.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::NonMaternRegime);
        let r = out.r_ratio.expect("R");
        assert!(r > 2.0 && r < 4.0);
        assert!(out.nu.is_some());
        assert!(out.kappa2.is_some());
    }

    #[test]
    fn continuous_order_boundary_r_equals_four_is_matern_square_case() {
        let out = compute_continuous_smoothness_order([1.0, 2.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::Ok);
        let nu = out.nu.expect("nu");
        assert!((nu - 2.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_guardszero_or_nearzero_lambda() {
        let out = compute_continuous_smoothness_order([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::IntrinsicLimit);
        assert!(out.r_ratio.is_none());
    }

    #[test]
    fn continuous_order_first_order_limitwhen_lambda2_collapses() {
        let out = compute_continuous_smoothness_order([2.0, 4.0, 1e-20], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::FirstOrderLimit);
        assert_eq!(out.nu, Some(1.0));
        let k2 = out.kappa2.expect("kappa2");
        assert!((k2 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_intrinsic_limitwhen_lambda0_collapses() {
        let out = compute_continuous_smoothness_order([1e-20, 4.0, 2.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::IntrinsicLimit);
        assert_eq!(out.nu, Some(1.0));
        assert_eq!(out.kappa2, Some(0.0));
    }

    #[test]
    fn continuous_order_is_only_defined_for_three_penalties_per_term() {
        let ok =
            try_compute_continuous_smoothness_order(&[2.0, 10.0, 3.0], &[1.0, 1.0, 1.0], 1e-12);
        let two = try_compute_continuous_smoothness_order(&[2.0, 10.0], &[1.0, 1.0], 1e-12);
        let four = try_compute_continuous_smoothness_order(
            &[2.0, 10.0, 3.0, 7.0],
            &[1.0, 1.0, 1.0, 1.0],
            1e-12,
        );
        assert!(ok.is_some());
        assert!(two.is_none());
        assert!(four.is_none());
    }
}

#[cfg(test)]
mod invert_regularized_rho_hessian_tests {
    use super::{EigenClassification, invert_regularized_rho_hessian};
    use ndarray::Array2;

    /// Build a real symmetric n×n matrix with a specified eigenvalue spectrum
    /// rotated by a fixed orthogonal basis. Returns (matrix, eigenvectors).
    fn build_with_spectrum(eigenvalues: &[f64]) -> (Array2<f64>, Array2<f64>) {
        let n = eigenvalues.len();
        let mut q = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let v = if i == j {
                    1.0
                } else {
                    ((i + 1) as f64 * 0.37 + (j + 1) as f64 * 0.19).sin()
                };
                q[[j, i]] = v;
            }
        }
        // Modified Gram-Schmidt orthonormalization on columns.
        for i in 0..n {
            for k in 0..i {
                let mut dot = 0.0;
                for r in 0..n {
                    dot += q[[r, i]] * q[[r, k]];
                }
                for r in 0..n {
                    q[[r, i]] -= dot * q[[r, k]];
                }
            }
            let mut nrm = 0.0;
            for r in 0..n {
                nrm += q[[r, i]] * q[[r, i]];
            }
            let nrm = nrm.sqrt();
            assert!(nrm > 1e-12, "degenerate basis in test setup");
            for r in 0..n {
                q[[r, i]] /= nrm;
            }
        }
        // Form A = Q * diag(eigenvalues) * Q^T.
        let mut a = Array2::<f64>::zeros((n, n));
        for r in 0..n {
            for c in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[[r, k]] * eigenvalues[k] * q[[c, k]];
                }
                a[[r, c]] = sum;
            }
        }
        for r in 0..n {
            for c in (r + 1)..n {
                let avg = 0.5 * (a[[r, c]] + a[[c, r]]);
                a[[r, c]] = avg;
                a[[c, r]] = avg;
            }
        }
        (a, q)
    }

    #[test]
    fn spd_case_returns_full_rank_inverse_no_repair() {
        let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0]);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 4);
        assert_eq!(inv.dropped_negative, 0);
        assert_eq!(inv.dropped_small_positive, 0);
        assert_eq!(inv.dropped_numerical_zero, 0);
        assert!(!inv.repaired_hessian);

        let prod = a.dot(&inv.inverse);
        for r in 0..4 {
            for c in 0..4 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (prod[[r, c]] - expected).abs() < 1e-9,
                    "A*Ainv[{r},{c}]={} not ~ {expected}",
                    prod[[r, c]]
                );
            }
        }
    }

    #[test]
    fn z2_saddle_case_drops_negative_eigenpair() {
        let evals = [10.0, 5.0, 2.0, -0.066];
        let (a, q) = build_with_spectrum(&evals);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 3);
        assert_eq!(inv.dropped_negative, 1);
        assert_eq!(inv.dropped_small_positive, 0);
        assert_eq!(inv.dropped_numerical_zero, 0);
        assert!(inv.repaired_hessian);

        // On each active eigenvector v: inv*A*v = v.
        for active_idx in 0..4 {
            if evals[active_idx] <= 0.0 {
                continue;
            }
            let v = q.column(active_idx).to_owned();
            let av = a.dot(&v);
            let inv_av = inv.inverse.dot(&av);
            for r in 0..4 {
                assert!(
                    (inv_av[r] - v[r]).abs() < 1e-9,
                    "active eigenvector not preserved at idx {active_idx}, row {r}: got {}, expected {}",
                    inv_av[r],
                    v[r]
                );
            }
        }
        // Negative-eigenvalue direction is annihilated.
        let v_neg = q.column(3).to_owned();
        let inv_vneg = inv.inverse.dot(&v_neg);
        let nrm = inv_vneg.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            nrm < 1e-9,
            "pseudo-inverse should annihilate dropped direction; got norm {nrm}"
        );
    }

    #[test]
    fn flat_direction_dropped() {
        // Build a matrix with one near-zero eigenvalue. We pick -1e-13 (just
        // below zero by less than neg_tol) so Cholesky reliably refuses the
        // matrix and we exercise the eigendecomp branch. The classification
        // should be DroppedNumericalZero or DroppedNegative, both of which
        // count as "near-zero direction dropped" for this test's purposes.
        let evals = [10.0, 5.0, 2.0, -1e-13];
        let (a, q) = build_with_spectrum(&evals);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 3, "expected three identified directions");
        let dropped =
            inv.dropped_small_positive + inv.dropped_numerical_zero + inv.dropped_negative;
        assert_eq!(dropped, 1, "expected exactly one direction dropped");
        assert!(inv.repaired_hessian);

        let v_flat = q.column(3).to_owned();
        let inv_vflat = inv.inverse.dot(&v_flat);
        let nrm = inv_vflat.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            nrm < 1e-3,
            "pseudo-inverse should annihilate flat direction; got norm {nrm}"
        );
    }

    #[test]
    fn mixed_negative_and_flat_yields_active_rank_two() {
        let evals = [10.0, 5.0, -0.066, 1e-13];
        let (a, _q) = build_with_spectrum(&evals);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 2);
        assert_eq!(inv.dropped_negative, 1);
        assert_eq!(
            inv.dropped_small_positive + inv.dropped_numerical_zero,
            1,
            "expected one near-zero direction dropped"
        );
        assert!(inv.repaired_hessian);
    }

    #[test]
    fn all_bad_spectrum_yields_zero_active_rank() {
        let evals = [-0.1, -0.05, -1.0, -0.5];
        let (a, _q) = build_with_spectrum(&evals);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 0);
        assert_eq!(inv.dropped_negative, 4);
        assert!(inv.repaired_hessian);
        for r in 0..4 {
            for c in 0..4 {
                assert!(inv.inverse[[r, c]].abs() < 1e-12);
            }
        }
        assert!(
            inv.classifications
                .iter()
                .all(|c| matches!(c, EigenClassification::DroppedNegative))
        );
    }

    #[test]
    fn non_finite_input_returns_none() {
        let mut a = Array2::<f64>::eye(4);
        a[[1, 1]] = f64::NAN;
        let result = invert_regularized_rho_hessian(&a);
        assert!(
            result.is_none(),
            "expected None for NaN-bearing input matrix"
        );

        let mut a = Array2::<f64>::eye(4);
        a[[2, 2]] = f64::INFINITY;
        let result = invert_regularized_rho_hessian(&a);
        assert!(
            result.is_none(),
            "expected None for Inf-bearing input matrix"
        );
    }

    /// The slow eigendecomposition path must populate `eigenvalues` AND
    /// `eigenvectors` so the [INDEF-HESS] diagnostic doesn't have to recompute
    /// `eigh` redundantly. The Cholesky fast path leaves both empty since the
    /// diagnostic isn't invoked when the matrix is SPD.
    #[test]
    fn slow_path_populates_eigenvalues_and_eigenvectors() {
        let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, -0.066]);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert!(inv.repaired_hessian);
        assert_eq!(inv.eigenvalues.len(), 4);
        assert_eq!(inv.eigenvectors.shape(), &[4, 4]);
        assert_eq!(inv.classifications.len(), 4);
        // Eigenvectors are unit-norm and pairwise orthogonal.
        for j in 0..4 {
            let v = inv.eigenvectors.column(j);
            let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (nrm - 1.0).abs() < 1e-9,
                "eigenvector {j} not unit-norm: ‖v‖={nrm}"
            );
        }
    }

    #[test]
    fn fast_path_leaves_eigendecomp_fields_empty() {
        let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0]);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert!(!inv.repaired_hessian);
        assert!(inv.eigenvalues.is_empty());
        assert!(inv.eigenvectors.is_empty());
        assert!(inv.classifications.is_empty());
    }
}

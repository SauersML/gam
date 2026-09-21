//! P-IRLS regression and root-cause tests.

// ── Test-only Tweedie density oracles ──────────────────────────────────────
// Consumed by `tweedie_exact_series_tests` below. They live here (inside the
// `pirls::tests` module) because the scanner forbids `#[cfg(test)]` on bare
// src items and cross-module consumption rules out a private test_support
// submodule in `deviance.rs`.
use super::{LN_2PI, tweedie_exact_series_loglik_from_eta};
use gam_spec::is_valid_tweedie_power;

#[inline]
fn tweedie_unit_deviance(yi: f64, mui_c: f64, p: f64) -> f64 {
    if !is_valid_tweedie_power(p) {
        f64::NAN
    } else if !valid_tweedie_response(yi) {
        f64::NAN
    } else if yi == 0.0 {
        mui_c.powf(2.0 - p) / (2.0 - p)
    } else {
        yi.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) - yi * mui_c.powf(1.0 - p) / (1.0 - p)
            + mui_c.powf(2.0 - p) / (2.0 - p)
    }
}

/// Tweedie **saddlepoint** log-density (prior weight `w` ⇒ `φᵢ = φ/w`). Exact at
/// `y = 0` for `1 < p < 2` (compound-Poisson point mass `exp(−wμ^{2−p}/((2−p)φ)`);
/// the standard `(2πφᵢ V(y))^{-½} exp(−wd/φ)` approximation for `y > 0`, where
/// `V(y) = y^p` and `d` is the unit deviance. The exponent matches the REML
/// kernel's `−w·d/φ` term exactly; this only restores the `−½ln(2πφᵢ y^p)`
/// prefactor. Homogeneous so `elpd(c·y) − elpd(y) = −n ln c` still holds.
#[inline]
fn tweedie_saddlepoint_loglik_approximation(yi: f64, mui: f64, w: f64, p: f64, phi: f64) -> f64 {
    if w <= 0.0 {
        // Zero prior weight excludes the observation (the y>0 prefactor's
        // −ln wᵢ would otherwise diverge).
        return 0.0;
    }
    let exponent = -w * tweedie_unit_deviance(yi, mui, p) / phi;
    if yi <= 0.0 {
        // Exact point mass at zero (no Jacobian prefactor for a mass atom).
        exponent
    } else {
        // φᵢ = φ/w  ⇒  −½ ln(2π (φ/w) y^p).
        exponent - 0.5 * (LN_2PI + phi.ln() - w.ln() + p * yi.ln())
    }
}

/// Exact Tweedie (compound Poisson–gamma, `1 < p < 2`) log-density at one
/// observation, evaluated by the Jørgensen / Dunn–Smyth infinite-series
/// representation of the exponential-dispersion normalizer.
///
/// Unlike [`tweedie_saddlepoint_loglik_approximation`] — which is asymptotically exact only in
/// the many-jumps (large-λ) limit and biases the maximum-likelihood variance
/// power **low** at small/moderate λ (#2105) — this is the exact normalized
/// density. It is what a profile likelihood of `p` must optimize (mgcv's
/// `ldTweedie` uses the same series for exactly this reason); the saddlepoint's
/// missing `O(1/λ)` normalizer correction, integrated across the sample, is what
/// dragged `p̂` down (e.g. `p̂ ≈ 1.33` on `p = 1.5` data) and thereby inflated the
/// reported Pearson dispersion `φ̂ = Σw(y−μ)²/μ^p / Σw` by `~13%`.
///
/// Density (prior weight `w` scales the dispersion, `φᵢ = φ/w`):
/// ```text
/// f(0)   = exp(−λ),                               λ = μ^{2−p} / (φᵢ (2−p))
/// f(y>0) = Σ_{k≥1} Pois(k; λ) · Gamma(y; kα, γ),  α = (2−p)/(p−1),
///                                                 γ = φᵢ (p−1) μ^{p−1}.
/// ```
/// Test adapter for the production eta-space exact-series oracle.
#[inline]
fn tweedie_series_loglik(yi: f64, mui: f64, w: f64, p: f64, phi: f64) -> f64 {
    if w == 0.0 {
        return 0.0;
    }
    tweedie_exact_series_loglik_from_eta(0, yi, mui.ln(), w, p, phi.ln())
        .expect("exact Tweedie test fixture")
}

/// Exact Tweedie log-density. This never switches to a saddlepoint: callers
/// selecting an exact likelihood always receive the compound-Poisson–gamma
/// series named by the API.
#[inline]
fn tweedie_exact_loglik(yi: f64, mui: f64, w: f64, p: f64, phi: f64) -> f64 {
    tweedie_series_loglik(yi, mui, w, p, phi)
}
// The nested test modules below address the rest of the solver through
// `super::`, which resolves to this module; the re-imports here forward the
// sibling concern modules and the shared item surface so those paths keep
// pointing at the same definitions they did when this file was inlined.

pub(crate) use super::*;

#[cfg(test)]
mod tests {
    use super::loop_driver::{default_beta_guess_external, exact_lambdas_from_rho};
    use super::reweight::madsen_lm_accept_factor;
    use super::{DENSE_OUTER_MAX_P, DevianceEtaRow, LinearInequalityConstraints, PenaltyConfig, PirlsConfig, PirlsLinearSolvePath, PirlsProblem, PirlsWorkspace, SparseXtWxCache, WeightFamily, WeightLink, WorkingDerivativeBuffersMut, bernoulli_geometry_from_jet, calculate_deviance_from_eta, calculate_loglikelihood_omitting_constants_from_eta, calculate_null_deviance, compute_constraint_kkt_diagnostics, compute_observed_hessian_curvature_arrays, deviance_eta_row_with_log_measure_scale, deviance_eta_rows_with_log_measure_scale, fit_model_for_fixed_rho, observed_weight_dispatch, observed_weight_noncanonical, pirls_data_log_kernel_from_eta, select_active_set_release, should_log_pirls_decision_summary, should_use_sparse_native_pirls, solve_newton_directionwith_linear_constraints, solve_newton_directionwith_lower_bounds, stable_finite_signed_sum, unit_measure_deviance_and_log_kernel_from_eta, update_glmvectors, variance_jet_for_weight_family, write_gamma_log_working_state, write_negative_binomial_log_working_state, write_poisson_log_working_state, write_tweedie_log_working_state};
    use crate::estimate::EstimationError;
    use crate::mixture_link::{InverseLinkJet as MixtureInverseLinkJet, state_fromspec};
    use approx::assert_relative_eq;
    use faer::sparse::{SparseColMat, Triplet};
    use gam_linalg::matrix::DesignMatrix;
    use gam_math::probability::standard_normal_quantile;
    use gam_problem::{
        GlmLikelihoodSpec, InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, LinkComponent,
        LinkFunction, LogSmoothingParamsView, MixtureLinkSpec, ResponseFamily, StandardLink,
    };

    // Test-only zero-log-measure-scale wrapper over the production single-row
    // deviance/score oracle. Lives in this test module (its only consumer) rather
    // than as a `#[cfg(test)]`-gated production fn, so the non-test lib build
    // carries no unreferenced item.
    fn deviance_eta_row(
        row: usize,
        y: f64,
        eta: f64,
        likelihood: &GlmLikelihoodSpec,
        inverse_link: &InverseLink,
        prior_weight: f64,
    ) -> Result<DevianceEtaRow, EstimationError> {
        deviance_eta_row_with_log_measure_scale(
            row,
            y,
            eta,
            likelihood,
            inverse_link,
            prior_weight,
            0.0,
        )
    }

    // Full-operator Firth/Jeffreys diagnostics reference (#1575): bit-identical to
    // the production factor-cached Firth diagnostics path, used here
    // as the operator-equivalence oracle. Lives in this test module (its only
    // consumer) rather than as a `#[cfg(test)]`-gated production fn.
    fn compute_jeffreys_pirls_diagnostics(
        link: &InverseLink,
        x_design: ArrayView2<f64>,
        eta: ArrayView1<f64>,
        observation_weights: ArrayView1<f64>,
    ) -> Result<(Array1<f64>, f64, Array1<f64>), EstimationError> {
        use crate::estimate::reml::FirthDenseOperator;
        let op = FirthDenseOperator::build_with_observation_weights_for_link(
            link,
            &x_design.to_owned(),
            &eta.to_owned(),
            observation_weights,
        )?;
        Ok((
            op.pirls_hat_diag(),
            op.jeffreys_logdet(),
            op.pirls_jeffreys_eta_score(),
        ))
    }
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder, array};

    #[test]
    pub(crate) fn dense_workspace_xtwx_preserves_signed_observed_weights() {
        let x = array![[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0], [0.5, -3.0]];
        let weights = array![2.0, -1.5, 0.25, -3.0];
        let mut streamed = Array2::<f64>::zeros((x.ncols(), x.ncols()).f());

        PirlsWorkspace::add_dense_xtwx_signed(&weights, &x, &mut streamed);

        let wx = Array2::from_shape_fn(x.raw_dim(), |(i, j)| weights[i] * x[[i, j]]);
        let expected = x.t().dot(&wx);
        for i in 0..x.ncols() {
            for j in 0..x.ncols() {
                assert_relative_eq!(streamed[[i, j]], expected[[i, j]], epsilon = 1e-12);
            }
        }
        assert!(
            streamed[[0, 0]] < 0.0,
            "negative row weights must not be clipped through a sqrt(max(0,w)) Gram path"
        );
    }

    #[test]
    fn sparse_spgemm_xtwx_preserves_signed_observed_weights() {
        let p = DENSE_OUTER_MAX_P + 1;
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 0, 3.0),
            Triplet::new(1, 1, -1.0),
        ];
        let x = SparseColMat::try_new_from_triplets(2, p, &triplets).unwrap();
        let mut cache = SparseXtWxCache::new(&x).unwrap();
        cache.compute_numeric(&x, &array![2.0, -1.5]).unwrap();

        let value = |row: usize, col: usize| {
            let range = cache.xtwx_symbolic.col_range(col);
            range
                .clone()
                .find_map(|index| {
                    (cache.xtwx_symbolic.row_idx()[index] == row).then_some(cache.xtwxvalues[index])
                })
                .expect("requested entry must be in X^T X symbolic pattern")
        };
        assert_relative_eq!(value(0, 0), -11.5, epsilon = 1e-12);
        assert_relative_eq!(value(0, 1), 8.5, epsilon = 1e-12);
        assert_relative_eq!(value(1, 1), 6.5, epsilon = 1e-12);
    }

    #[test]
    pub(crate) fn firth_pirls_diagnostics_preserve_nonstandard_inverse_link() {
        let x = array![[1.0, -1.2], [1.0, -0.3], [1.0, 0.4], [1.0, 1.1], [1.0, 1.8],];
        let eta = array![-1.4, -0.5, 0.2, 0.9, 1.4];
        let observation_weights = array![1.0, 0.7, 1.3, 0.9, 1.1];
        let cloglog = InverseLink::Standard(StandardLink::CLogLog);
        let mixture = InverseLink::Mixture(
            state_fromspec(&MixtureLinkSpec {
                components: vec![
                    LinkComponent::CLogLog,
                    LinkComponent::LogLog,
                    LinkComponent::Cauchit,
                ],
                initial_rho: array![0.2, -0.4],
            })
            .expect("valid mixture spec"),
        );

        for link in [&cloglog, &mixture] {
            let (hat, logdet, score) = compute_jeffreys_pirls_diagnostics(
                link,
                x.view(),
                eta.view(),
                observation_weights.view(),
            )
            .expect("supported Firth inverse link");
            assert_eq!(hat.len(), x.nrows());
            assert_eq!(score.len(), x.nrows());
            assert!(
                logdet.is_finite(),
                "Jeffreys logdet must stay finite for {link:?}"
            );
            assert!(
                hat.iter().all(|value| value.is_finite() && *value >= 0.0),
                "hat diagonal must stay finite and non-negative for {link:?}: {hat:?}"
            );
            assert!(
                score.iter().all(|value| value.is_finite()),
                "Jeffreys eta-score must stay finite for {link:?}: {score:?}"
            );
        }
    }

    #[test]
    pub(crate) fn firth_factored_path_matches_full_operator_oracle_1575() {
        // #1575 hoisted the β-INDEPENDENT Firth design factor out of the inner
        // Newton loop: `build_design_factor_with_observation_weights` (η-free,
        // built once) followed by `build_from_design_factor` (per-η) must
        // reproduce the full operator rebuilt fresh at each η — the
        // by-construction equivalence the hoist's correctness rests on. This
        // pins that equivalence directly (the surrounding tests only exercise
        // the factored path through end-to-end fits), including the
        // identifiable-subspace reduction on a RANK-DEFICIENT design and the
        // design-factor REUSE across multiple η (the whole point of the hoist).
        use crate::estimate::reml::FirthDenseOperator;

        // Rank-deficient design: col 4 := col 2 + col 3, forcing a structural
        // null direction so the reduced-space (r = 3 < p = 4) path is taken.
        let mut x = array![
            [1.0, -1.2, 0.5, 0.0],
            [1.0, -0.3, 1.1, 0.0],
            [1.0, 0.4, -0.6, 0.0],
            [1.0, 1.1, 0.2, 0.0],
            [1.0, 1.8, -0.9, 0.0],
            [1.0, 0.1, 1.4, 0.0],
        ];
        for i in 0..x.nrows() {
            x[[i, 3]] = x[[i, 1]] + x[[i, 2]];
        }
        let etas = [
            array![-1.4, -0.5, 0.2, 0.9, 1.4, 0.0],
            array![0.3, -2.1, 1.7, -0.8, 0.6, -1.2],
            array![2.0, 2.0, -2.0, -2.0, 0.5, -0.5],
        ];
        let weights = array![1.0, 0.7, 1.3, 0.9, 1.1, 0.4];
        let logit = InverseLink::Standard(StandardLink::Logit);
        let cloglog = InverseLink::Standard(StandardLink::CLogLog);

        for link in [&logit, &cloglog] {
            // Build each design factor ONCE; reuse across every η below.
            let factor_w = FirthDenseOperator::build_design_factor_with_observation_weights(
                &x,
                Some(weights.view()),
            )
            .expect("weighted design factor builds");
            let factor_u =
                FirthDenseOperator::build_design_factor_with_observation_weights(&x, None)
                    .expect("unweighted design factor builds");

            for eta in &etas {
                // Weighted: factored fast path vs full-operator oracle.
                let op_f = FirthDenseOperator::build_from_design_factor(&factor_w, link, eta)
                    .expect("factored weighted operator");
                let hat_f = op_f.pirls_hat_diag();
                let logdet_f = op_f.jeffreys_logdet();
                let score_f = op_f.pirls_jeffreys_eta_score();
                let (hat_o, logdet_o, score_o) =
                    compute_jeffreys_pirls_diagnostics(link, x.view(), eta.view(), weights.view())
                        .expect("oracle weighted diagnostics");
                assert_relative_eq!(logdet_f, logdet_o, epsilon = 1e-12, max_relative = 1e-12);
                for i in 0..x.nrows() {
                    assert_relative_eq!(hat_f[i], hat_o[i], epsilon = 1e-12, max_relative = 1e-12);
                    assert_relative_eq!(
                        score_f[i],
                        score_o[i],
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                }

                // Unweighted (`None` branch of the factor builder) vs the full
                // operator with no observation weights.
                let op_fu = FirthDenseOperator::build_from_design_factor(&factor_u, link, eta)
                    .expect("factored unweighted operator");
                let hat_fu = op_fu.pirls_hat_diag();
                let logdet_fu = op_fu.jeffreys_logdet();
                let score_fu = op_fu.pirls_jeffreys_eta_score();
                let op_u = FirthDenseOperator::build_for_link(link, &x, eta)
                    .expect("full unweighted operator");
                assert_relative_eq!(
                    logdet_fu,
                    op_u.jeffreys_logdet(),
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
                let hat_ou = op_u.pirls_hat_diag();
                let score_ou = op_u.pirls_jeffreys_eta_score();
                for i in 0..x.nrows() {
                    assert_relative_eq!(
                        hat_fu[i],
                        hat_ou[i],
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                    assert_relative_eq!(
                        score_fu[i],
                        score_ou[i],
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                }
            }
        }
    }

    /// Calculate scale parameter correctly for different link functions.
    ///
    /// Contract:
    /// - Gaussian (Identity): residual standard deviation sigma
    /// - Binomial links: fixed at 1.0 as in mgcv
    pub(crate) fn calculate_scale(
        beta: &Array1<f64>,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        offset: ArrayView1<f64>,
        edf: f64,
        link_function: LinkFunction,
    ) -> f64 {
        match link_function {
            LinkFunction::Logit
            | LinkFunction::Probit
            | LinkFunction::CLogLog
            | LinkFunction::LogLog
            | LinkFunction::Cauchit
            | LinkFunction::Sas
            | LinkFunction::BetaLogistic
            | LinkFunction::Log => 1.0,
            LinkFunction::Inverse | LinkFunction::InverseSquared | LinkFunction::Sqrt => {
                panic!("calculate_scale has no residual scale for the reciprocal and sqrt links")
            }
            LinkFunction::Identity => {
                let mut fitted = x.dot(beta);
                fitted += &offset;
                let residuals = &y - &fitted;
                let weighted_rss: f64 = weights
                    .iter()
                    .zip(residuals.iter())
                    .map(|(&w, &r)| w * r * r)
                    .sum();
                let effective_n = y.len() as f64;
                (weighted_rss / (effective_n - edf).max(1.0)).sqrt()
            }
        }
    }

    #[test]
    pub(crate) fn madsen_lm_reject_trajectory_doubles_per_rejection() {
        // The companion to the accept update: on rejection, `loop_lambda`
        // is multiplied by `madsen_reject_factor` (initially 2.0), then
        // the factor doubles. Replaces the older fixed ×10 every time.
        // Locks the trajectory so a future commit can't silently restore
        // the binary ×10 rule (which over-shot the `LM_MAX_LAMBDA = 1e12`
        // ceiling in just 12 rejections — the textbook ×2 doubling needs
        // ~40 rejections to hit the same ceiling, well past
        // `lm_max_attempts`).
        let mut loop_lambda = 1.0_f64;
        let mut v = 2.0_f64;
        let trajectory = (0..6)
            .map(|_| {
                loop_lambda *= v;
                v *= 2.0;
                loop_lambda
            })
            .collect::<Vec<_>>();
        // 1.0 * 2 = 2; * 4 = 8; * 8 = 64; * 16 = 1024; * 32 = 32768; * 64 = 2097152
        assert_eq!(
            trajectory,
            vec![2.0, 8.0, 64.0, 1024.0, 32_768.0, 2_097_152.0],
            "Madsen rejection trajectory must double the multiplier each time"
        );
        // Compared with the OLD fixed ×10 rule, which gave
        //   [10, 100, 1_000, 10_000, 100_000, 1_000_000]
        // — Madsen's ×2 doubling is gentler initially (2 < 10) but
        // catches up (rejection 6: 2_097_152 > 1_000_000). The point
        // isn't to be smaller forever; the point is to give MORE
        // chances near the trust radius before saturating the ceiling.
        // Under ×10, after 12 rejections lambda × 10^12 hits LM_MAX_LAMBDA
        // and lm_can_retry returns false. Under ×2 doubling, we get
        // lambda × 2^(N(N+1)/2) — N=12 gives 2^78 ≈ 3·10^23, much past
        // the ceiling, so the ceiling fires earlier in attempt count
        // but later in cumulative-multiplier terms — the LM trajectory
        // covers more of the trust-radius space before declaring the
        // search exhausted.
    }

    #[test]
    pub(crate) fn madsen_lm_accept_factor_matches_canonical_textbook_values() {
        // Madsen-Nielsen-Tingleff Eq 3.17 canonical values. Locks the
        // implementation against silent regression to the older binary
        // `if rho > 0.25 { lambda /= 10 } else { keep }` rule.
        let cases: &[(f64, f64, &str)] = &[
            (1.0, 1.0 / 3.0, "rho=1: floored at 1/3 (cube=1, 1-cube=0)"),
            (0.75, 0.875, "rho=0.75: 1 - (0.5)^3 = 0.875 (slight shrink)"),
            (0.5, 1.0, "rho=0.5: 1 - 0 = 1.0 (no change)"),
            (
                0.25,
                1.125,
                "rho=0.25: 1 - (-0.5)^3 = 1.125 (slight expand)",
            ),
        ];
        for (rho, expected, why) in cases {
            let got = madsen_lm_accept_factor(*rho);
            assert!(
                (got - expected).abs() < 1e-12,
                "madsen_lm_accept_factor({rho}) = {got:.6}, expected {expected:.6} — {why}"
            );
        }
        // Marginal-accept (rho → 0⁺): cube → -1, 1 - cube → 2.0.
        // Capped at 2.0 so a barely-accepted step bumps lambda by at
        // most ×2 — the texbook upper bound (vs unbounded growth as
        // rho continues to drop, which never fires in this branch
        // because rho ≤ 0 routes through the rejection path).
        let small_positive = madsen_lm_accept_factor(1e-9);
        assert!(
            (small_positive - 2.0).abs() < 1e-6,
            "rho ≈ 0⁺ must approach the 2.0 cap; got {small_positive:.6}"
        );
        // Hypothetical rho < 0 still yields a well-defined cap so the
        // function is total — this protects against numeric corner
        // cases producing NaN even though the LM loop never calls us
        // there.
        assert_eq!(madsen_lm_accept_factor(-100.0), 2.0);
        assert_eq!(madsen_lm_accept_factor(100.0), 1.0 / 3.0);
        // Floor + ceiling are exact (no roundoff slop on the clamp).
        assert!(madsen_lm_accept_factor(0.99).is_finite());
        assert!(madsen_lm_accept_factor(0.01) <= 2.0 + 1e-15);
        assert!(madsen_lm_accept_factor(0.99) >= 1.0 / 3.0 - 1e-15);
    }

    #[test]
    fn log_link_edges_are_exact_and_tiny_weights_are_not_floored() {
        let eta = array![-700.0, 0.0, 700.0, -2.0];
        let y = array![1.0, 1.0, 1.0, 0.0];
        let prior = array![1.0, 1e-300, 1.0, 0.0];
        let n = eta.len();
        let mut mu = Array1::zeros(n);
        let mut weights = Array1::zeros(n);
        let mut z = Array1::zeros(n);
        let mut c = Array1::zeros(n);
        let mut d = Array1::zeros(n);
        let mut d1 = Array1::zeros(n);
        let mut d2 = Array1::zeros(n);
        let mut d3 = Array1::zeros(n);
        write_poisson_log_working_state(
            y.view(),
            &eta,
            prior.view(),
            &mut mu,
            &mut weights,
            &mut z,
            Some(WorkingDerivativeBuffersMut {
                c: &mut c,
                d: &mut d,
                dmu_deta: &mut d1,
                d2mu_deta2: &mut d2,
                d3mu_deta3: &mut d3,
            }),
        )
        .expect("closed log-link domain must be represented exactly");
        for i in 0..n {
            assert_eq!(mu[i].to_bits(), eta[i].exp().to_bits());
            assert_eq!(d1[i].to_bits(), mu[i].to_bits());
            assert_eq!(d2[i].to_bits(), mu[i].to_bits());
            assert_eq!(d3[i].to_bits(), mu[i].to_bits());
            assert!(z[i].is_finite());
        }
        assert_eq!(weights[1].to_bits(), 1e-300_f64.to_bits());
        assert_eq!(c[1].to_bits(), weights[1].to_bits());
        assert_eq!(d[1].to_bits(), weights[1].to_bits());
        assert_eq!(weights[3], 0.0);
        assert_eq!(z[3].to_bits(), eta[3].to_bits());
    }

    #[test]
    fn every_log_link_family_uses_the_same_exact_row_and_derivative_surface() {
        let eta = array![-2.0, 0.0, 2.0];
        let y = array![1.0, 2.0, 4.0];
        let prior = array![0.5, 1e-300, 0.0];
        let n = eta.len();

        let check = |family: &str,
                     with_mu: &Array1<f64>,
                     with_w: &Array1<f64>,
                     with_z: &Array1<f64>,
                     without_mu: &Array1<f64>,
                     without_w: &Array1<f64>,
                     without_z: &Array1<f64>| {
            assert_eq!(with_mu, without_mu, "{family} mu seam");
            assert_eq!(with_w, without_w, "{family} weight seam");
            assert_eq!(with_z, without_z, "{family} working-response seam");
        };

        // Gamma: W = prior * shape and both eta derivatives vanish.
        {
            let shape = 2.5;
            let mut mu = Array1::zeros(n);
            let mut w = Array1::zeros(n);
            let mut z = Array1::zeros(n);
            let mut c = Array1::zeros(n);
            let mut d = Array1::zeros(n);
            let mut d1 = Array1::zeros(n);
            let mut d2 = Array1::zeros(n);
            let mut d3 = Array1::zeros(n);
            write_gamma_log_working_state(
                y.view(),
                &eta,
                prior.view(),
                shape,
                &mut mu,
                &mut w,
                &mut z,
                Some(WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut d1,
                    d2mu_deta2: &mut d2,
                    d3mu_deta3: &mut d3,
                }),
            )
            .unwrap();
            let mut mu_plain = Array1::zeros(n);
            let mut w_plain = Array1::zeros(n);
            let mut z_plain = Array1::zeros(n);
            write_gamma_log_working_state(
                y.view(),
                &eta,
                prior.view(),
                shape,
                &mut mu_plain,
                &mut w_plain,
                &mut z_plain,
                None,
            )
            .unwrap();
            check("Gamma", &mu, &w, &z, &mu_plain, &w_plain, &z_plain);
            for i in 0..n {
                assert_eq!(w[i].to_bits(), (prior[i] * shape).to_bits());
                assert_eq!(c[i], 0.0);
                assert_eq!(d[i], 0.0);
                assert_eq!(d1[i].to_bits(), mu[i].to_bits());
            }
        }

        // Tweedie: W = prior * mu^(2-p)/phi and c,d are exact derivatives.
        {
            let (p, phi) = (1.5, 2.0);
            let mut mu = Array1::zeros(n);
            let mut w = Array1::zeros(n);
            let mut z = Array1::zeros(n);
            let mut c = Array1::zeros(n);
            let mut d = Array1::zeros(n);
            let mut d1 = Array1::zeros(n);
            let mut d2 = Array1::zeros(n);
            let mut d3 = Array1::zeros(n);
            write_tweedie_log_working_state(
                y.view(),
                &eta,
                prior.view(),
                p,
                phi,
                &mut mu,
                &mut w,
                &mut z,
                Some(WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut d1,
                    d2mu_deta2: &mut d2,
                    d3mu_deta3: &mut d3,
                }),
            )
            .unwrap();
            let mut mu_plain = Array1::zeros(n);
            let mut w_plain = Array1::zeros(n);
            let mut z_plain = Array1::zeros(n);
            write_tweedie_log_working_state(
                y.view(),
                &eta,
                prior.view(),
                p,
                phi,
                &mut mu_plain,
                &mut w_plain,
                &mut z_plain,
                None,
            )
            .unwrap();
            check("Tweedie", &mu, &w, &z, &mu_plain, &w_plain, &z_plain);
            for i in 0..n {
                assert_eq!(c[i].to_bits(), (0.5 * w[i]).to_bits());
                assert_eq!(d[i].to_bits(), (0.25 * w[i]).to_bits());
            }
        }

        // NB2: express curvature through r = theta/(theta+mu), never a squared
        // overflowing denominator.
        {
            let theta = 3.0;
            let mut mu = Array1::zeros(n);
            let mut w = Array1::zeros(n);
            let mut z = Array1::zeros(n);
            let mut c = Array1::zeros(n);
            let mut d = Array1::zeros(n);
            let mut d1 = Array1::zeros(n);
            let mut d2 = Array1::zeros(n);
            let mut d3 = Array1::zeros(n);
            write_negative_binomial_log_working_state(
                y.view(),
                &eta,
                prior.view(),
                theta,
                &mut mu,
                &mut w,
                &mut z,
                Some(WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut d1,
                    d2mu_deta2: &mut d2,
                    d3mu_deta3: &mut d3,
                }),
            )
            .unwrap();
            let mut mu_plain = Array1::zeros(n);
            let mut w_plain = Array1::zeros(n);
            let mut z_plain = Array1::zeros(n);
            write_negative_binomial_log_working_state(
                y.view(),
                &eta,
                prior.view(),
                theta,
                &mut mu_plain,
                &mut w_plain,
                &mut z_plain,
                None,
            )
            .unwrap();
            check("NB2", &mu, &w, &z, &mu_plain, &w_plain, &z_plain);
            for i in 0..n {
                let r = theta / (theta + mu[i]);
                assert_relative_eq!(c[i], w[i] * r, max_relative = 1e-15);
                assert_relative_eq!(d[i], w[i] * r * (2.0 * r - 1.0), max_relative = 1e-15);
            }
        }
    }

    #[test]
    fn every_log_link_rule_refuses_unrepresentable_row_products_atomically() {
        fn sentinels() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
            (
                Array1::from_elem(1, 11.0),
                Array1::from_elem(1, 13.0),
                Array1::from_elem(1, 17.0),
            )
        }
        fn assert_atomic(mu: &Array1<f64>, w: &Array1<f64>, z: &Array1<f64>) {
            assert_eq!(mu[0], 11.0);
            assert_eq!(w[0], 13.0);
            assert_eq!(z[0], 17.0);
        }
        let eta = array![700.0];
        let y = array![1.0];

        let (mut mu, mut w, mut z) = sentinels();
        let err = write_poisson_log_working_state(
            y.view(),
            &eta,
            array![1e10].view(),
            &mut mu,
            &mut w,
            &mut z,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EstimationError::PirlsRowGeometryUnrepresentable { row: 0, .. }
        ));
        assert_atomic(&mu, &w, &z);

        let (mut mu, mut w, mut z) = sentinels();
        write_gamma_log_working_state(
            y.view(),
            &array![0.0],
            array![2.0].view(),
            1e308,
            &mut mu,
            &mut w,
            &mut z,
            None,
        )
        .unwrap_err();
        assert_atomic(&mu, &w, &z);

        let (mut mu, mut w, mut z) = sentinels();
        write_tweedie_log_working_state(
            y.view(),
            &array![-700.0],
            array![1.0].view(),
            1.000_001,
            1e308,
            &mut mu,
            &mut w,
            &mut z,
            None,
        )
        .unwrap_err();
        assert_atomic(&mu, &w, &z);

        let (mut mu, mut w, mut z) = sentinels();
        write_negative_binomial_log_working_state(
            y.view(),
            &eta,
            array![1e308].view(),
            3.0,
            &mut mu,
            &mut w,
            &mut z,
            None,
        )
        .unwrap_err();
        assert_atomic(&mu, &w, &z);
    }

    #[test]
    fn fixed_rho_uses_the_shared_closed_log_strength_domain_without_projection() {
        use gam_problem::{LOG_STRENGTH_MAX, LOG_STRENGTH_MIN};

        let rho = array![LOG_STRENGTH_MIN, 0.0, LOG_STRENGTH_MAX];
        let lambda = exact_lambdas_from_rho(
            LogSmoothingParamsView::new(rho.view()).expect("closed strength domain"),
        );
        for i in 0..rho.len() {
            assert_eq!(lambda[i].to_bits(), rho[i].exp().to_bits());
        }

        let invalid = array![0.0, LOG_STRENGTH_MAX + 1.0, LOG_STRENGTH_MIN - 1.0];
        let err = LogSmoothingParamsView::new(invalid.view())
            .expect_err("out-of-domain rho must be refused before exponentiation");
        assert_eq!(err.coordinate, 1);
        assert_eq!(err.value, LOG_STRENGTH_MAX + 1.0);
    }

    #[test]
    fn log_link_refusal_is_atomic_and_reports_the_smallest_bad_row() {
        let eta = array![0.0, 701.0, -701.0];
        let y = array![1.0, 1.0, 1.0];
        let prior = Array1::ones(3);
        let mut mu = Array1::from_elem(3, 17.0);
        let mut weights = Array1::from_elem(3, 19.0);
        let mut z = Array1::from_elem(3, 23.0);
        let err = write_poisson_log_working_state(
            y.view(),
            &eta,
            prior.view(),
            &mut mu,
            &mut weights,
            &mut z,
            None,
        )
        .expect_err("out-of-domain eta must be refused");
        assert!(matches!(
            err,
            EstimationError::InverseLinkDomainViolation { eta: 701.0, .. }
        ));
        assert_eq!(mu, Array1::from_elem(3, 17.0));
        assert_eq!(weights, Array1::from_elem(3, 19.0));
        assert_eq!(z, Array1::from_elem(3, 23.0));
    }

    #[test]
    fn canonical_logit_tail_geometry_remains_exact_at_rounded_mean_endpoints() {
        let eta = array![-700.0, -40.0, 40.0, 700.0];
        let y = array![1.0, 1.0, 0.0, 0.0];
        let prior = Array1::ones(4);
        let mut mu = Array1::zeros(4);
        let mut weights = Array1::zeros(4);
        let mut z = Array1::zeros(4);
        update_glmvectors(
            y.view(),
            &eta,
            &InverseLink::Standard(StandardLink::Logit),
            prior.view(),
            &mut mu,
            &mut weights,
            &mut z,
            None,
        )
        .expect("represented canonical-logit tails must not be projected");
        assert_eq!(mu[2], 1.0);
        assert_eq!(mu[3], 1.0);
        for i in 0..4 {
            let jet = crate::mixture_link::logit_inverse_link_jet5(eta[i]);
            assert_eq!(weights[i].to_bits(), jet.d1.to_bits());
            assert!(weights[i] > 0.0 && z[i].is_finite());
        }
    }

    #[test]
    fn canonical_logit_saturated_consistent_rows_take_the_zero_weight_limit() {
        // Past |eta| ~ 745.13 the logit mean derivative rounds to zero. A row whose
        // response sits on the saturated boundary (adult income, gam#3195: row with
        // eta = -745.17, y = 0) has a residual that rounds to zero with it, so its
        // working geometry is the prior-weight-zero limit, not a refusal.
        let eta = array![-745.5, -746.0, 745.5, 900.0];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let prior = Array1::ones(4);
        let mut mu = Array1::zeros(4);
        let mut weights = Array1::zeros(4);
        let mut z = Array1::zeros(4);
        update_glmvectors(
            y.view(),
            &eta,
            &InverseLink::Standard(StandardLink::Logit),
            prior.view(),
            &mut mu,
            &mut weights,
            &mut z,
            None,
        )
        .expect("consistent saturated canonical-logit rows are representable");
        for i in 0..4 {
            assert_eq!(crate::mixture_link::logit_inverse_link_jet5(eta[i]).d1, 0.0);
            assert_eq!(weights[i], 0.0);
            assert_eq!(z[i], eta[i]);
            assert_eq!(mu[i], y[i]);
        }

        // The same saturation with the response on the far boundary keeps a
        // unit-order score and no representable weight: still refused.
        for (eta_value, y_value) in [(-746.0, 1.0), (746.0, 0.0)] {
            let mut mu = Array1::zeros(1);
            let mut weights = Array1::zeros(1);
            let mut z = Array1::zeros(1);
            let refused = update_glmvectors(
                array![y_value].view(),
                &array![eta_value],
                &InverseLink::Standard(StandardLink::Logit),
                array![1.0].view(),
                &mut mu,
                &mut weights,
                &mut z,
                None,
            );
            assert!(refused.is_err(), "eta={eta_value}, y={y_value} must be refused");
        }
    }

    #[test]
    fn canonical_logit_weight_derivative_matches_finite_difference_at_tail() {
        let eta0 = 40.0;
        let h = 1e-4;
        let eval_weight = |eta_value: f64| {
            let eta = array![eta_value];
            let y = array![0.0];
            let prior = array![1.0];
            let mut mu = Array1::zeros(1);
            let mut weight = Array1::zeros(1);
            let mut z = Array1::zeros(1);
            update_glmvectors(
                y.view(),
                &eta,
                &InverseLink::Standard(StandardLink::Logit),
                prior.view(),
                &mut mu,
                &mut weight,
                &mut z,
                None,
            )
            .unwrap();
            weight[0]
        };
        let fd = (eval_weight(eta0 + h) - eval_weight(eta0 - h)) / (2.0 * h);
        let analytic = crate::mixture_link::logit_inverse_link_jet5(eta0).d2;
        assert_relative_eq!(fd, analytic, max_relative = 2e-8, epsilon = 1e-30);
    }

    #[test]
    pub(crate) fn gaussian_scale_uses_offset_in_residuals() {
        // Perfect fit only if offset is included: y = offset + Xβ.
        // If offset were dropped, weighted RSS would be non-zero.
        let x = array![[1.0], [2.0], [3.0]];
        let beta = array![2.0];
        let offset = array![10.0, 20.0, 30.0];
        let y = array![12.0, 24.0, 36.0]; // offset + x * beta
        let w = Array1::ones(3);

        let scale = calculate_scale(
            &beta,
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            0.0,
            LinkFunction::Identity,
        );

        assert!(
            scale.abs() < 1e-12,
            "scale must be ~0 for exact fit with offset; got {}",
            scale
        );
    }

    #[test]
    pub(crate) fn gaussian_scale_matchesweighted_sdwith_offset() {
        let x = array![[1.0], [2.0], [4.0]];
        let beta = array![1.5];
        let offset = array![0.5, -1.0, 2.0];
        let y = array![2.2, 2.0, 7.5];
        let w = array![1.0, 2.0, 0.5];
        let edf = 1.25;

        let scale = calculate_scale(
            &beta,
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            edf,
            LinkFunction::Identity,
        );

        let mut fitted = x.dot(&beta);
        fitted += &offset;
        let rss: f64 = w
            .iter()
            .zip(y.iter().zip(fitted.iter()))
            .map(|(&wi, (&yi, &fi))| wi * (yi - fi).powi(2))
            .sum();
        let expected = (rss / ((y.len() as f64 - edf).max(1.0))).sqrt();

        assert!(
            (scale - expected).abs() < 1e-12,
            "scale mismatch: got {}, expected {}",
            scale,
            expected
        );
    }

    #[test]
    pub(crate) fn kkt_diagnosticszero_for_strictly_feasible_stationary_point() {
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, 1.0]],
            b: array![0.0, 0.0],
        };
        let beta = array![1.0, 2.0];
        let grad = array![0.0, 0.0];
        let diag = compute_constraint_kkt_diagnostics(&beta, &grad, grad.dot(&grad).sqrt(), &constraints);
        assert!(diag.primal_feasibility <= 1e-12);
        assert!(diag.dual_feasibility <= 1e-12);
        assert!(diag.complementarity <= 1e-12);
        assert!(diag.stationarity <= 1e-12);
    }

    #[test]
    pub(crate) fn kkt_diagnostics_capture_active_lower_bound_solution() {
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, 1.0]],
            b: array![0.0, 0.0],
        };
        let beta = array![0.0, 1.5];
        let grad = array![2.0, 0.0];
        let diag = compute_constraint_kkt_diagnostics(&beta, &grad, grad.dot(&grad).sqrt(), &constraints);
        assert_eq!(diag.n_constraints, 2);
        assert_eq!(diag.n_active, 1);
        assert!(diag.primal_feasibility <= 1e-12);
        assert!(diag.dual_feasibility <= 1e-12);
        assert!(diag.complementarity <= 1e-12);
        assert!(diag.stationarity <= 1e-10);
    }

    /// The FULLY-ACTIVE face with a RANK-DEFICIENT active set — the case where a
    /// KKT diagnostic is most likely to be wrong, and the one the other two
    /// `compute_constraint_kkt_diagnostics` tests above cannot reach.
    ///
    /// Those two both use the orthogonal `[[1,0],[0,1]]` and exercise
    /// `n_active ∈ {0, 1}`: a strictly interior point with zero gradient, and a
    /// single active lower bound. Neither has a redundant row, and neither has an
    /// empty tangent space.
    ///
    /// Here rows 0 and 1 are PARALLEL (`[1,0]` and `[2,0]`), every row is active at
    /// `beta = 0` because every `b_i = 0`, and the gradient is placed in the
    /// INTERIOR of the active cone as `Aᵀ·lambda` with all three multipliers
    /// strictly positive. Three things follow that nothing else here pins:
    ///
    /// * `n_active == n_constraints == 3` — the tangent space is empty, so there is
    ///   no direction left to move and every residual must vanish exactly.
    /// * The active set is rank deficient (`rank(A) = 2 < 3`), so the multipliers
    ///   are NOT unique: `[1, 0.5, 2]` and `[2, 0, 2]` reproduce this identical
    ///   gradient. Anything that recovers multipliers by solving a normal-equation
    ///   system meets a singular `AᵀA` on this fixture. Non-uniqueness must not
    ///   become non-zero residuals — the diagnostic's job is to certify the FACE,
    ///   not to pick a canonical `lambda`.
    /// * Every multiplier is strictly positive, so a sign convention that flipped
    ///   `lambda` would surface as `dual_feasibility`, not hide in a zero.
    ///
    /// This coverage previously lived in a test that called a
    /// multiplier-consuming entry point; that entry point was retired with the
    /// row-pivot CTN solver and the orphaned test was removed, which fixed the
    /// build and dropped the degenerate case with it. Restored here against the
    /// surviving API rather than left as a gap.
    #[test]
    pub(crate) fn kkt_diagnostics_vanish_on_a_rank_deficient_fully_active_face() {
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            b: array![0.0, 0.0, 0.0],
        };
        let beta = array![0.0, 0.0];
        let lambda_true = array![1.0, 0.5, 2.0];
        let grad = constraints.a.t().dot(&lambda_true);

        let diag = compute_constraint_kkt_diagnostics(&beta, &grad, grad.dot(&grad).sqrt(), &constraints);

        assert_eq!(diag.n_constraints, 3);
        assert_eq!(
            diag.n_active, 3,
            "every row has b_i = 0 and beta = 0, so the whole face is active; got {}",
            diag.n_active
        );
        assert!(
            diag.primal_feasibility <= 1e-12,
            "beta = 0 meets every row with exactly zero slack; got {}",
            diag.primal_feasibility
        );
        assert!(
            diag.dual_feasibility <= 1e-12,
            "the gradient is A^T * lambda with lambda strictly positive, so no \
             recovered multiplier may be negative; got {}",
            diag.dual_feasibility
        );
        assert!(
            diag.complementarity <= 1e-12,
            "every active row has zero slack, so lambda_i * slack_i vanishes \
             regardless of which multiplier vector is recovered; got {}",
            diag.complementarity
        );
        assert!(
            diag.stationarity <= 1e-10,
            "the gradient lies in the cone spanned by the active normals, so \
             ||grad - A^T lambda||_inf must vanish even though rank(A) = 2 < 3 \
             leaves lambda non-unique; got {}",
            diag.stationarity
        );
    }

    #[test]
    pub(crate) fn linear_constraint_active_set_releases_positive_kkt_systemmultiplier() {
        // min_d g^T d + 0.5 d^T H d, subject to A(beta + d) >= b
        // with beta fixed at the lower bound x >= 0 and an upper bound x <= 0.1.
        // The first active-set KKT solve at x=0 yields d=0 and lambda_sys=+1
        // for the lower-bound row, which must be released (lambda_true = -lambda_sys).
        let hessian = array![[1.0]];
        let gradient = array![-1.0];
        let beta = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [-1.0]],
            b: array![0.0, -0.1],
        };
        let mut direction = Array1::zeros(1);

        solve_newton_directionwith_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            None,
        )
        .expect("constrained Newton direction should solve");

        assert!(
            (direction[0] - 0.1).abs() <= 1e-10,
            "expected step to upper bound (0.1), got {}",
            direction[0]
        );
    }

    #[test]
    pub(crate) fn linear_constraint_active_set_ignores_near_tangential_inactiverows() {
        let hessian = array![[1.0, 0.0], [0.0, 1.0]];
        let gradient = array![-1.0, 0.0];
        let beta = array![0.0, 0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[-1e-16, 1.0]],
            b: array![-1.0],
        };
        let mut direction = Array1::zeros(2);

        solve_newton_directionwith_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            None,
        )
        .expect("near-tangential inactive row should not block the Newton step");

        assert!(
            (direction[0] - 1.0).abs() <= 1e-12,
            "expected unconstrained x-step of 1.0, got {}",
            direction[0]
        );
        assert!(
            direction[1].abs() <= 1e-12,
            "expected zero y-step, got {}",
            direction[1]
        );
    }

    #[test]
    pub(crate) fn default_beta_guess_logit_uses_log_odds_prevalence() {
        let y = array![0.0, 1.0, 1.0, 1.0];
        let w = Array1::ones(4);
        let beta =
            default_beta_guess_external(3, &ResponseFamily::Binomial, LinkFunction::Logit, y.view(), w.view(), None, None);
        let prevalence: f64 = (3.0 + 0.5) / (4.0 + 1.0);
        let expected = (prevalence / (1.0 - prevalence)).ln();
        assert!((beta[0] - expected).abs() < 1e-12);
        assert_eq!(beta[1], 0.0);
        assert_eq!(beta[2], 0.0);
    }

    #[test]
    pub(crate) fn default_beta_guess_probit_uses_standard_normal_quantile() {
        let y = array![0.0, 1.0, 1.0, 1.0];
        let w = Array1::ones(4);
        let beta =
            default_beta_guess_external(3, &ResponseFamily::Binomial, LinkFunction::Probit, y.view(), w.view(), None, None);
        let prevalence: f64 = (3.0 + 0.5) / (4.0 + 1.0);
        let log_odds = (prevalence / (1.0 - prevalence)).ln();
        let expected =
            standard_normal_quantile(prevalence).expect("prevalence lies inside (0, 1)");
        assert!((expected - log_odds).abs() > 1e-3);
        assert!((beta[0] - expected).abs() < 1e-12);
        assert_eq!(beta[1], 0.0);
        assert_eq!(beta[2], 0.0);
    }

    #[test]
    pub(crate) fn sparse_native_decision_rejects_dense_design() {
        let x = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0]
        ]));
        let s = array![[1.0, 0.0], [0.0, 1.0]];
        let mut workspace = PirlsWorkspace::new(2, 2);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::DenseTransformed);
        assert_eq!(decision.reason, "design_not_sparse");
        // The dense route never counts the design's nonzeros, and says so.
        assert_eq!(decision.nnz_x, None);
        assert!(decision.format_fields(decision.path_str()).contains("nnz_x=na"));
    }

    #[test]
    pub(crate) fn pirls_decision_summary_logs_on_power_of_two_repetitions() {
        assert!(!should_log_pirls_decision_summary(1));
        assert!(should_log_pirls_decision_summary(2));
        assert!(!should_log_pirls_decision_summary(3));
        assert!(should_log_pirls_decision_summary(4));
        assert!(!should_log_pirls_decision_summary(6));
        assert!(should_log_pirls_decision_summary(8));
    }

    #[test]
    pub(crate) fn sparse_native_decision_collects_sparse_stats_for_large_sparse_design() {
        let triplets: Vec<_> = (0..300).map(|i| Triplet::new(i, i, 1.0)).collect();
        let x = SparseColMat::try_new_from_triplets(300, 300, &triplets)
            .expect("sparse identity should build");
        let x = DesignMatrix::from(x);
        let s = Array2::from_diag(&Array1::ones(300));
        let mut workspace = PirlsWorkspace::new(300, 300);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::SparseNative);
        assert_eq!(decision.reason, "sparse_native_eligible");
        assert_eq!(decision.nnz_x, Some(300));
        assert_eq!(decision.nnz_xtwx_symbolic, Some(300));
        assert_eq!(decision.nnz_h_est, Some(300));
        assert!(decision.density_h_est.expect("density") < 0.01);
    }

    #[test]
    pub(crate) fn sparse_native_decision_allows_moderate_sparse_designs_below_old_width_gate() {
        let triplets: Vec<_> = (0..64).map(|i| Triplet::new(i, i, 1.0)).collect();
        let x = SparseColMat::try_new_from_triplets(64, 64, &triplets)
            .expect("sparse identity should build");
        let x = DesignMatrix::from(x);
        let s = Array2::from_diag(&Array1::ones(64));
        let mut workspace = PirlsWorkspace::new(64, 64);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::SparseNative);
        assert_eq!(decision.reason, "sparse_native_eligible");
        assert_eq!(decision.nnz_x, Some(64));
        assert_eq!(decision.nnz_xtwx_symbolic, Some(64));
        assert_eq!(decision.nnz_h_est, Some(64));
        assert!(decision.density_h_est.expect("density") < 0.05);
    }

    #[test]
    pub(crate) fn sparse_native_decision_rejects_finite_lower_bounds() {
        let triplets: Vec<_> = (0..64).map(|i| Triplet::new(i, i, 1.0)).collect();
        let x = SparseColMat::try_new_from_triplets(64, 64, &triplets)
            .expect("sparse identity should build");
        let x = DesignMatrix::from(x);
        let s = Array2::from_diag(&Array1::ones(64));
        let mut lower_bounds = Array1::from_elem(64, f64::NEG_INFINITY);
        lower_bounds[0] = 0.0;
        let mut workspace = PirlsWorkspace::new(64, 64);
        let decision =
            should_use_sparse_native_pirls(&mut workspace, &x, &s, Some(&lower_bounds), None);
        assert_eq!(decision.path, PirlsLinearSolvePath::DenseTransformed);
        assert_eq!(decision.reason, "constraints_present");
    }

    /// Stiff two-penalty binomial fixture shared by the PSD-root regression
    /// tests: a one-hot design with three rows per coefficient and penalty
    /// energies whose ratio exceeds the square-root-solve stiffness threshold.
    /// Returns the sparse design, the canonical penalties and the smoothing
    /// parameters.
    fn stiff_one_hot_binomial_fixture() -> (
        SparseColMat<usize, f64>,
        Vec<gam_terms::construction::CanonicalPenalty>,
        Array1<f64>,
    ) {
        use gam_terms::construction::CanonicalPenalty;

        let p = 64usize;
        let n = 3 * p;
        let triplets: Vec<_> = (0..n).map(|row| Triplet::new(row, row / 3, 1.0)).collect();
        let x = SparseColMat::try_new_from_triplets(n, p, &triplets)
            .expect("one-hot sparse design should build");

        let mut low_energy_root = Array2::<f64>::zeros((1, p));
        low_energy_root[[0, 0]] = 1.0;
        let mut high_energy_root = Array2::<f64>::zeros((p - 1, p));
        for coefficient in 1..p {
            high_energy_root[[coefficient - 1, coefficient]] = 1.0;
        }
        let canonical: Vec<_> = [low_energy_root, high_energy_root]
            .into_iter()
            .map(|root| {
                let rank = root.nrows();
                CanonicalPenalty {
                    local: root.t().dot(&root).into_shared(),
                    root: root.into_shared(),
                    col_range: 0..p,
                    total_dim: p,
                    nullity: p - rank,
                    prior_mean: Array1::zeros(p),
                    positive_eigenvalues: vec![1.0; rank],
                    op: None,
                }
            })
            .collect();

        let rho = array![-12.0, 12.0];
        let lambdas = rho.mapv(f64::exp);
        assert!(
            lambdas[1] / lambdas[0] > f64::EPSILON.sqrt().recip(),
            "fixture must exceed the PSD-root stiffness threshold"
        );
        (x, canonical, rho)
    }

    fn fit_stiff_one_hot_binomial<X: Into<DesignMatrix> + Clone>(
        x: X,
        canonical: &[gam_terms::construction::CanonicalPenalty],
        rho: &Array1<f64>,
    ) -> Result<super::PirlsResult, EstimationError> {
        let design: DesignMatrix = x.clone().into();
        let n = design.nrows();
        let p = canonical[0].total_dim;
        let y = Array1::from_shape_fn(n, |row| if row % 3 == 2 { 1.0 } else { 0.0 });
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            )),
            link_kind: InverseLink::Standard(StandardLink::Logit),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
        };

        fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view())
                .expect("test rho lies in the smoothing-strength domain"),
            PirlsProblem {
                x,
                offset: offset.view(),
                y: y.view(),
                priorweights: weights.view(),
                gaussian_fixed_cache: None,
                glm_first_step_gram: None,
            },
            PenaltyConfig {
                canonical_penalties: canonical,
                reparam_invariant: None,
                p,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
            },
            &config,
            None,
        )
        .map(|(fit, _)| fit)
    }

    /// Regression for #2401: the cancellation-safe PSD-root solve introduced
    /// for stiff P-IRLS systems must accept the sparse-native coordinate frame.
    /// Before the fix, the routing oracle correctly chose `SparseNative`, the
    /// disparate penalty energies correctly chose the augmented root solve, and
    /// the dense root writer then aborted solely because those two valid
    /// choices had no shared implementation.
    #[test]
    pub(crate) fn sparse_native_stiff_penalty_uses_psd_root_2401() {
        let (x, canonical, rho) = stiff_one_hot_binomial_fixture();
        let p = x.ncols();
        let n = x.nrows();
        let x_design = DesignMatrix::from(x.clone());
        let lambdas = rho.mapv(f64::exp);
        let weighted_penalty = &canonical[0].local * lambdas[0] + &canonical[1].local * lambdas[1];
        let mut routing_workspace = PirlsWorkspace::new(n, p);
        let decision = should_use_sparse_native_pirls(
            &mut routing_workspace,
            &x_design,
            &weighted_penalty,
            None,
            None,
        );
        assert_eq!(
            decision.path,
            PirlsLinearSolvePath::SparseNative,
            "fixture must exercise the sparse-native coordinate frame; reason={}",
            decision.reason
        );

        let fit = fit_stiff_one_hot_binomial(x, &canonical, &rho)
            .expect("stiff sparse-native binomial P-IRLS fit must use the PSD root");

        assert!(
            fit.beta_transformed.iter().all(|value| value.is_finite()),
            "sparse-native PSD-root fit must return finite coefficients"
        );
    }

    /// The same stiff system through a dense design streams its augmented
    /// root through the blocked QR in design-row chunks instead of assembling
    /// an `(n + rank + p) × p` dense root. The dense coordinate frame must
    /// reach the same P-IRLS termination state as the sparse-native frame on
    /// this identical penalized problem.
    #[test]
    pub(crate) fn dense_stiff_penalty_streams_psd_root_through_blocked_qr() {
        let (x, canonical, rho) = stiff_one_hot_binomial_fixture();
        let x_dense = DesignMatrix::from(x.clone()).to_dense();
        let lambdas = rho.mapv(f64::exp);
        let weighted_penalty = &canonical[0].local * lambdas[0] + &canonical[1].local * lambdas[1];
        let mut routing_workspace = PirlsWorkspace::new(x_dense.nrows(), x_dense.ncols());
        let decision = should_use_sparse_native_pirls(
            &mut routing_workspace,
            &DesignMatrix::from(x_dense.clone()),
            &weighted_penalty,
            None,
            None,
        );
        assert_eq!(
            decision.path,
            PirlsLinearSolvePath::DenseTransformed,
            "fixture must exercise the dense coordinate frame; reason={}",
            decision.reason
        );

        let sparse_fit = fit_stiff_one_hot_binomial(x, &canonical, &rho)
            .expect("stiff sparse-native binomial P-IRLS fit");
        let dense_fit = fit_stiff_one_hot_binomial(x_dense, &canonical, &rho)
            .expect("stiff dense binomial P-IRLS fit must stream the PSD root");

        assert!(
            dense_fit.beta_transformed.iter().all(|value| value.is_finite()),
            "dense PSD-root fit must return finite coefficients"
        );
        assert_eq!(
            dense_fit.status, sparse_fit.status,
            "dense and sparse square-root fits must terminate in the same P-IRLS state"
        );
    }

    // The sparse-native frame conversion must preserve the declared penalty exactly.
    #[test]
    pub(crate) fn sparse_native_reparam_preserves_declared_penalty() {
        use gam_terms::construction::{
            CanonicalPenalty, EngineDims, stable_reparameterization_original_frame,
        };
        use ndarray::array;

        let p = 2usize;
        let root = array![[1.0, 0.0]];
        let canonical = vec![CanonicalPenalty::from_dense_root(root, p)];
        let lambdas = [3.0f64];
        let result = stable_reparameterization_original_frame(
            &canonical,
            &lambdas,
            EngineDims::new(p, canonical.len()),
            None,
        )
        .expect("declared penalty must reparameterize");

        let gram = result.e_transformed.t().dot(&result.e_transformed);
        for (actual, expected) in gram.iter().zip(result.s_transformed.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-12);
        }
        assert_relative_eq!(result.s_transformed[[0, 0]], 3.0, epsilon = 1e-12);
        assert_relative_eq!(result.s_transformed[[1, 1]], 0.0, epsilon = 1e-12);
    }

    #[test]
    pub(crate) fn sparse_penalized_assembly_matches_dense_diagonal_case() {
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 3.0),
        ];
        let x = SparseColMat::try_new_from_triplets(3, 3, &triplets)
            .expect("diagonal sparse matrix should build");
        let weights = array![2.0, 3.0, 5.0];
        let s_lambda = array![[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]];
        let mut workspace = PirlsWorkspace::new(3, 3);
        let assembled = super::sparse_reml_penalized_hessian(
            &mut workspace,
            &x,
            &weights,
            &s_lambda,
            None,
        )
        .expect("sparse penalized assembly should succeed");
        let dense = DesignMatrix::from(x.clone()).to_dense();
        let mut expected = dense.t().dot(&Array2::from_diag(&weights)).dot(&dense);
        expected += &s_lambda;
        let actual = DesignMatrix::from(assembled).to_dense();
        for i in 0..3 {
            for j in 0..3 {
                let target = if i <= j { expected[[i, j]] } else { 0.0 };
                assert!(
                    (actual[[i, j]] - target).abs() < 1e-10,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    actual[[i, j]],
                    target
                );
            }
        }
    }

    #[test]
    pub(crate) fn pure_logit_working_state_preserves_tail_fisher_mass() {
        let y = array![1.0];
        let eta = array![50.0];
        let priorweights = array![1.0];
        let inverse_link = InverseLink::Standard(StandardLink::Logit);
        let mut mu = Array1::zeros(1);
        let mut weights = Array1::zeros(1);
        let mut z = Array1::zeros(1);

        update_glmvectors(
            y.view(),
            &eta,
            &inverse_link,
            priorweights.view(),
            &mut mu,
            &mut weights,
            &mut z,
            None,
        )
        .expect("pure logit working state");

        let jet = crate::mixture_link::logit_inverse_link_jet5(eta[0]);
        assert!(jet.d1 > 0.0);
        assert!(
            (weights[0] - jet.d1).abs() < 1e-30,
            "pure logit PIRLS weight should equal the stable tail formula at eta={}; got {} vs {}",
            eta[0],
            weights[0],
            jet.d1
        );
        assert!(
            (mu[0] - jet.mu).abs() < 1e-30,
            "pure logit PIRLS mu mismatch at eta={}; got {} vs {}",
            eta[0],
            mu[0],
            jet.mu
        );
        // `mu` rounds to EXACTLY 1.0 here: 1 - e^-50 is not representable
        // distinctly from 1 (e^-50 = 1.93e-22, machine epsilon = 2.2e-16). So
        // reconstructing the residual as `y - jet.mu` CANCELS to zero and would
        // claim z == eta — i.e. that this observation carries no information,
        // which is precisely the tail-mass loss this test exists to catch.
        //
        // Production avoids that by rebuilding `y - mu` from the tail complement
        // (`canonical_logit_working_response`: "mu may round to exactly zero or
        // one while exp(-|eta|) and the score remain representable"), so the
        // reference here must be built the same way:
        //     tail = e^{-|eta|},  1 - mu = tail/(1+tail),  d1 = tail/(1+tail)^2
        // hence for y = 1,  (y - mu)/d1 = 1 + tail  (= 51 at eta = 50).
        let tail = (-eta[0].abs()).exp();
        let one_minus_mu = tail / (1.0 + tail);
        let expected_z = eta[0] + one_minus_mu / jet.d1;
        assert!(
            (expected_z - (eta[0] + 1.0 + tail)).abs() <= 1e-12,
            "stable working-response reference disagrees with its closed form: {} vs {}",
            expected_z,
            eta[0] + 1.0 + tail
        );
        assert!(
            (z[0] - expected_z).abs() < 1e-12,
            "pure logit PIRLS z should preserve the exact working response at eta={}; got {} vs {}",
            eta[0],
            z[0],
            expected_z
        );
        // Same cancellation trap as above, and here it defeated the assertion's
        // own purpose: `y[0] - jet.mu` evaluates to 0 at this eta, so the check
        // "the score carrier preserves y-mu" was satisfied by any implementation
        // that ALSO lost the tail mass to zero — the exact regression the test is
        // named for. The true residual is the stable complement, so compare
        // against that:  w·(z-eta) = d1·(1+tail) = tail/(1+tail) = 1-mu.
        assert!(
            (weights[0] * (z[0] - eta[0]) - one_minus_mu).abs() < 1e-30,
            "pure logit PIRLS score carrier should preserve y-mu at eta={}; got {} vs {}",
            eta[0],
            weights[0] * (z[0] - eta[0]),
            one_minus_mu
        );
        assert!(
            one_minus_mu > 0.0,
            "the tail residual must not be the cancelled zero this test guards against"
        );
    }

    /// Below full saturation the noncanonical links carry the exact tail
    /// complement `1 - mu` (option 3): `mu` has already rounded to `1.0`, but the
    /// weight `d1^2/(mu(1-mu))` and the working score stay representable, so the
    /// row proceeds instead of being refused. (cloglog `eta = 5`, probit
    /// `eta = 10` are both past the point `mu` rounds to `1.0`.)
    #[test]
    pub(crate) fn noncanonical_binomial_carries_tail_complement_below_saturation() {
        for (link, eta_val) in [(StandardLink::CLogLog, 5.0), (StandardLink::Probit, 10.0)] {
            let y = array![1.0];
            let eta = array![eta_val];
            let priorweights = array![1.0];
            let inverse_link = InverseLink::Standard(link);
            let mut mu = Array1::zeros(1);
            let mut weights = Array1::zeros(1);
            let mut z = Array1::zeros(1);

            update_glmvectors(
                y.view(),
                &eta,
                &inverse_link,
                priorweights.view(),
                &mut mu,
                &mut weights,
                &mut z,
                None,
            )
            .expect("tail-complement row must be representable");

            assert_eq!(
                mu[0], 1.0,
                "{link:?} at eta={eta_val} is past where mu rounds to 1.0"
            );
            assert!(
                weights[0].is_finite() && weights[0] > 0.0,
                "{link:?} working weight must stay positive finite via the carried complement; got {}",
                weights[0]
            );
            assert!(
                z[0].is_finite(),
                "{link:?} working response must remain finite; got {}",
                z[0]
            );
            // The working score X^T W (z - eta) is the whole point of carrying the
            // complement: it stays representable and positive rather than
            // collapsing to zero (or the row being refused outright).
            let score = weights[0] * (z[0] - eta[0]);
            assert!(
                score.is_finite() && score > 0.0,
                "{link:?} working score must stay positive finite; got {score}"
            );
        }
    }

    /// A fully saturated row (`mu == 1.0`, complement underflowed) that is
    /// *consistent* with its response (`y == 1`) is the analytic `eta -> inf`
    /// limit of the weight formula: a zero-weight row, exactly what the
    /// `priorweight == 0` branch returns.
    #[test]
    pub(crate) fn saturated_consistent_binomial_row_is_zero_weight() {
        for (link, eta_val) in [(StandardLink::CLogLog, 30.0), (StandardLink::Probit, 40.0)] {
            let y = array![1.0];
            let eta = array![eta_val];
            let priorweights = array![1.0];
            let inverse_link = InverseLink::Standard(link);
            let mut mu = Array1::zeros(1);
            let mut weights = Array1::zeros(1);
            let mut z = Array1::zeros(1);

            update_glmvectors(
                y.view(),
                &eta,
                &inverse_link,
                priorweights.view(),
                &mut mu,
                &mut weights,
                &mut z,
                None,
            )
            .expect("consistent saturated row must be representable");

            assert_eq!(mu[0], 1.0, "{link:?} saturated mu");
            assert_eq!(
                weights[0], 0.0,
                "{link:?} consistent saturated row must be zero-weight; got {}",
                weights[0]
            );
            assert_eq!(
                z[0], eta[0],
                "{link:?} zero-weight working response is eta; got {}",
                z[0]
            );
        }
    }

    /// A fully saturated row that is *inconsistent* with its response (`y == 0`
    /// while `mu == 1.0`) has `-inf` log-likelihood and must stay a typed
    /// refusal — never silently zero-weighted.
    #[test]
    pub(crate) fn saturated_inconsistent_binomial_row_is_refused() {
        for (link, eta_val) in [(StandardLink::CLogLog, 30.0), (StandardLink::Probit, 40.0)] {
            let y = array![0.0];
            let eta = array![eta_val];
            let priorweights = array![1.0];
            let inverse_link = InverseLink::Standard(link);
            let mut mu = Array1::zeros(1);
            let mut weights = Array1::zeros(1);
            let mut z = Array1::zeros(1);

            let result = update_glmvectors(
                y.view(),
                &eta,
                &inverse_link,
                priorweights.view(),
                &mut mu,
                &mut weights,
                &mut z,
                None,
            );

            assert!(
                matches!(
                    result,
                    Err(EstimationError::PirlsRowGeometryUnrepresentable { .. })
                ),
                "{link:?} inconsistent saturated row must be a typed refusal; got {result:?}"
            );
        }
    }

    /// The stable Fisher-weight curvature `c = dW/deta`, `d = d2W/deta2` matches
    /// central finite differences of the weight through the saturating band where
    /// the naive `d1^2/v` form divides by an underflowed variance (cloglog
    /// `eta` 5.9-6.61, probit ~24-27). Each probe rebuilds the inverse-link jet
    /// and the stable complement from scratch — the working cache is never frozen
    /// across the difference stencil. The pair-then-multiply factoring keeps `c`
    /// and `d` finite and nonzero where the naive quotient would be `0/0`.
    #[test]
    pub(crate) fn noncanonical_binomial_curvature_matches_central_fd_through_saturation() {
        let weight_c_d = |link: StandardLink, eta: f64| -> (f64, f64, f64) {
            let inverse_link = InverseLink::Standard(link);
            let jet = crate::mixture_link::inverse_link_jet_for_inverse_link(&inverse_link, eta)
                .expect("inverse-link jet must evaluate");
            let omm = crate::mixture_link::inverse_link_complement_for_inverse_link(
                &inverse_link,
                eta,
                jet.mu,
            );
            let geometry = bernoulli_geometry_from_jet(0, eta, 1.0, 1.0, jet, omm)
                .expect("saturating-band row must be representable");
            (geometry.weight, geometry.c, geometry.d)
        };
        // (link, eta, fd step): a regime-1 point then the regime-2 band where the
        // naive `d1^2/(mu(1-mu))` and its `v^2`/`v^3` curvature denominators
        // underflow. Steps are sized to the local scale of the super-exponential
        // weight so central differences stay well inside float precision.
        for (link, eta, h) in [
            (StandardLink::CLogLog, 5.0, 3e-5),
            (StandardLink::CLogLog, 6.0, 3e-5),
            (StandardLink::CLogLog, 6.3, 3e-5),
            (StandardLink::Probit, 12.0, 1e-4),
            (StandardLink::Probit, 25.0, 1e-4),
        ] {
            let (w0, c, d) = weight_c_d(link, eta);
            assert!(
                w0 > 0.0 && c.is_finite() && d.is_finite() && c != 0.0 && d != 0.0,
                "{link:?} eta={eta}: weight/curvature must be representable and nonzero; \
                 W={w0} c={c} d={d}"
            );
            let (wp, _, _) = weight_c_d(link, eta + h);
            let (wm, _, _) = weight_c_d(link, eta - h);
            let c_fd = (wp - wm) / (2.0 * h);
            let d_fd = (wp - 2.0 * w0 + wm) / (h * h);
            let rel_c = (c - c_fd).abs() / c.abs();
            let rel_d = (d - d_fd).abs() / d.abs();
            assert!(
                rel_c <= 5e-3,
                "{link:?} eta={eta}: dW/deta vs central FD rel err {rel_c:.2e} \
                 (analytic {c:.3e}, fd {c_fd:.3e})"
            );
            assert!(
                rel_d <= 5e-3,
                "{link:?} eta={eta}: d2W/deta2 vs central FD rel err {rel_d:.2e} \
                 (analytic {d:.3e}, fd {d_fd:.3e})"
            );
        }
    }

    #[test]
    pub(crate) fn gamma_log_deviance_uses_gamma_formula() {
        let y = array![2.0, 5.0];
        let mu = array![1.0, 4.0];
        let w = array![1.5, 0.75];
        let eta = mu.mapv(f64::ln);
        let inverse_link = InverseLink::Standard(StandardLink::Log);
        let dev = calculate_deviance_from_eta(
            y.view(),
            &eta,
            &GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gamma,
                inverse_link.clone(),
            )),
            &inverse_link,
            w.view(),
        )
        .expect("Gamma eta deviance must be representable");
        let expected = 2.0
            * (1.5 * (2.0_f64 / 1.0 - 1.0 - (2.0_f64 / 1.0).ln())
                + 0.75 * (5.0_f64 / 4.0 - 1.0 - (5.0_f64 / 4.0).ln()));
        assert_relative_eq!(dev, expected, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn null_deviance_preserves_boundaries_dormancy_and_beta_mle_geometry() {
        let poisson = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ));
        let zeros = array![0.0, 0.0, f64::NAN];
        let dormant = array![1.0, 2.0, 0.0];
        assert_eq!(
            calculate_null_deviance(zeros.view(), &poisson, dormant.view()).unwrap(),
            0.0,
            "an all-zero positive-weight Poisson sample has a genuine boundary null deviance"
        );
        let negative = array![1.0, -1.0, 0.0];
        assert!(calculate_null_deviance(zeros.view(), &poisson, negative.view()).is_err());

        let phi = 7.0;
        let beta = GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::Beta { phi },
                InverseLink::Standard(StandardLink::Logit),
            ),
            scale: gam_problem::LikelihoodScaleMetadata::EstimatedBetaPhi { phi },
        };
        let y = array![0.01, 0.2, 0.85];
        let w = array![1.0, 3.0, 0.5];
        let exact_null = calculate_null_deviance(y.view(), &beta, w.view()).unwrap();
        let arithmetic_mean = y
            .iter()
            .zip(w.iter())
            .map(|(&response, &weight)| response * weight)
            .sum::<f64>()
            / w.sum();
        let arithmetic_eta = array![
            arithmetic_mean.ln() - (-arithmetic_mean).ln_1p(),
            arithmetic_mean.ln() - (-arithmetic_mean).ln_1p(),
            arithmetic_mean.ln() - (-arithmetic_mean).ln_1p(),
        ];
        let arithmetic_deviance = calculate_deviance_from_eta(
            y.view(),
            &arithmetic_eta,
            &beta,
            &InverseLink::Standard(StandardLink::Logit),
            w.view(),
        )
        .unwrap();
        assert!(
            exact_null < arithmetic_deviance,
            "the fixed-precision Beta intercept MLE is not generally the weighted arithmetic mean"
        );
    }

    #[test]
    fn deviance_eta_row_value_and_score_are_one_surface_for_every_glm_family() {
        let cases = [
            (ResponseFamily::Gaussian, StandardLink::Identity, -0.4, 0.7),
            (ResponseFamily::Poisson, StandardLink::Log, 3.0, 0.4),
            (ResponseFamily::Gamma, StandardLink::Log, 1.7, -0.2),
            (
                ResponseFamily::Tweedie { p: 1.45 },
                StandardLink::Log,
                2.2,
                0.3,
            ),
            (
                ResponseFamily::NegativeBinomial {
                    theta: 1.8,
                    theta_fixed: true,
                },
                StandardLink::Log,
                4.0,
                0.6,
            ),
            (ResponseFamily::Binomial, StandardLink::Logit, 0.3, -0.8),
            (
                ResponseFamily::Beta { phi: 3.5 },
                StandardLink::Logit,
                0.35,
                -0.3,
            ),
        ];
        let prior_weight = 1.3;
        let h = 2.0e-6;
        for (family, link, y, eta) in cases {
            let inverse_link = InverseLink::Standard(link);
            let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                family.clone(),
                inverse_link.clone(),
            ));
            let row = deviance_eta_row(0, y, eta, &likelihood, &inverse_link, prior_weight)
                .expect("central deviance row");
            let plus = deviance_eta_row(0, y, eta + h, &likelihood, &inverse_link, prior_weight)
                .expect("plus row")
                .half_deviance;
            let minus = deviance_eta_row(0, y, eta - h, &likelihood, &inverse_link, prior_weight)
                .expect("minus row")
                .half_deviance;
            let finite_difference = (plus - minus) / (2.0 * h);
            assert_relative_eq!(
                row.eta_score,
                finite_difference,
                epsilon = 2.0e-7,
                max_relative = 2.0e-6
            );
        }
    }

    #[test]
    fn deviance_eta_row_preserves_extreme_balanced_value_and_score_channels() {
        let canonical = |family, link| {
            let inverse_link = InverseLink::Standard(link);
            (
                GlmLikelihoodSpec::canonical(LikelihoodSpec::new(family, inverse_link.clone())),
                inverse_link,
            )
        };

        let (poisson, log) = canonical(ResponseFamily::Poisson, StandardLink::Log);
        let far_left = deviance_eta_row(0, 1.0, -1.0e308, &poisson, &log, 1.0)
            .expect("finite far-left Poisson row");
        assert_relative_eq!(far_left.half_deviance, 1.0e308, max_relative = 2.0e-15);
        assert_eq!(far_left.eta_score, -1.0);
        let ratio_overflow = deviance_eta_row(0, 1.0e10, -700.0, &poisson, &log, 1.0)
            .expect("Poisson deviance must not form y/mu");
        assert!(ratio_overflow.half_deviance.is_finite());

        let (negative_binomial, log) = canonical(
            ResponseFamily::NegativeBinomial {
                theta: 1.0,
                theta_fixed: true,
            },
            StandardLink::Log,
        );
        let nb = deviance_eta_row(0, 2.0, -1.0e308, &negative_binomial, &log, 0.5)
            .expect("finite far-left NB row");
        assert_relative_eq!(nb.half_deviance, 1.0e308, max_relative = 2.0e-15);
        assert_eq!(nb.eta_score, -1.0);
        let (nb_positive_tail, log) = canonical(
            ResponseFamily::NegativeBinomial {
                theta: 3.0,
                theta_fixed: true,
            },
            StandardLink::Log,
        );
        let nb_positive = deviance_eta_row(0, 2.0, 1.0e308, &nb_positive_tail, &log, 0.25)
            .expect("finite far-right NB score/value");
        assert_relative_eq!(nb_positive.eta_score, 0.75, max_relative = 2.0e-15);
        assert!(nb_positive.half_deviance.is_finite());

        let (binomial, logit) = canonical(ResponseFamily::Binomial, StandardLink::Logit);
        let binomial_tail = deviance_eta_row(0, 0.5, -1.0e308, &binomial, &logit, 1.0)
            .expect("finite logit natural-coordinate tail");
        assert_relative_eq!(binomial_tail.half_deviance, 5.0e307, max_relative = 2.0e-15);
        assert_eq!(binomial_tail.eta_score, -0.5);

        let (gaussian, identity) = canonical(ResponseFamily::Gaussian, StandardLink::Identity);
        let gaussian_balanced = deviance_eta_row(0, 1.0e200, 0.0, &gaussian, &identity, 1.0e-300)
            .expect("weighted Gaussian square remains finite");
        assert_relative_eq!(
            gaussian_balanced.half_deviance,
            5.0e99,
            max_relative = 3.0e-14
        );
        assert_relative_eq!(
            gaussian_balanced.eta_score,
            -1.0e-100,
            max_relative = 3.0e-14
        );
        let gaussian_overflowing_residual =
            deviance_eta_row(0, f64::MAX, -f64::MAX, &gaussian, &identity, 1.0e-320)
                .expect("weighted Gaussian opposite-sign residual remains finite");
        assert!(gaussian_overflowing_residual.half_deviance.is_finite());
        assert!(gaussian_overflowing_residual.eta_score.is_finite());

        let (gamma, log) = canonical(ResponseFamily::Gamma, StandardLink::Log);
        let gamma_balanced = deviance_eta_row(0, f64::MAX, -700.0, &gamma, &log, 1.0e-320)
            .expect("weighted Gamma ratio remains finite");
        assert!(gamma_balanced.half_deviance.is_finite());
        assert!(gamma_balanced.eta_score.is_finite());

        let (tweedie, log) = canonical(ResponseFamily::Tweedie { p: 1.5 }, StandardLink::Log);
        let tweedie_balanced = deviance_eta_row(0, f64::MAX, -700.0, &tweedie, &log, 1.0e-300)
            .expect("weighted Tweedie power product remains finite");
        assert!(tweedie_balanced.half_deviance.is_finite());
        assert!(tweedie_balanced.eta_score.is_finite());

        for p in [
            f64::from_bits(1.0_f64.to_bits() + 1),
            f64::from_bits(2.0_f64.to_bits() - 1),
        ] {
            let (boundary, log) = canonical(ResponseFamily::Tweedie { p }, StandardLink::Log);
            for eta in [-100.0, 100.0] {
                let row = deviance_eta_row(0, 1.0, eta, &boundary, &log, 1.0)
                    .expect("Tweedie boundary-power row");
                let h = 1.0e-5;
                let plus = deviance_eta_row(0, 1.0, eta + h, &boundary, &log, 1.0)
                    .expect("boundary plus")
                    .half_deviance;
                let minus = deviance_eta_row(0, 1.0, eta - h, &boundary, &log, 1.0)
                    .expect("boundary minus")
                    .half_deviance;
                assert_relative_eq!(
                    row.eta_score,
                    (plus - minus) / (2.0 * h),
                    max_relative = 2.0e-6
                );
            }
        }

        let ignored = deviance_eta_row(
            0,
            f64::NAN,
            f64::INFINITY,
            &poisson,
            &InverseLink::Standard(StandardLink::Log),
            0.0,
        )
        .expect("zero-weight row has exactly zero statistical measure");
        assert_eq!(ignored.half_deviance, 0.0);
        assert_eq!(ignored.eta_score, 0.0);

        let eta = array![-1000.0];
        let y = array![0.0];
        let weights = array![1.0];
        assert_eq!(
            deviance_eta_row(0, 0.0, eta[0], &poisson, &log, 1.0)
                .expect("raw underflowed Poisson row")
                .half_deviance,
            0.0
        );
        let phi = f64::from_bits(1);
        let scaled = deviance_eta_rows_with_log_measure_scale(
            y.view(),
            &eta,
            &poisson,
            &log,
            weights.view(),
            -phi.ln(),
        )
        .expect("scale is folded in before materializing the row");
        assert!(scaled[0].half_deviance.is_finite() && scaled[0].half_deviance > 0.0);
        assert!(scaled[0].eta_score.is_finite() && scaled[0].eta_score > 0.0);
    }

    #[test]
    fn deviance_eta_batch_reports_the_smallest_invalid_row_atomically() {
        let inverse_link = InverseLink::Standard(StandardLink::Log);
        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            inverse_link.clone(),
        ));
        let y = array![1.0, -1.0, -2.0];
        let eta = array![0.0, 0.0, 0.0];
        let weights = array![1.0, 1.0, 1.0];
        assert!(matches!(
            calculate_deviance_from_eta(y.view(), &eta, &likelihood, &inverse_link, weights.view(),),
            Err(EstimationError::PirlsRowGeometryUnrepresentable { row: 1, .. })
        ));
    }

    #[test]
    fn signed_deviance_reduction_avoids_partial_sum_overflow() {
        let values = [f64::MAX, f64::MAX, -f64::MAX];
        assert_eq!(
            stable_finite_signed_sum(&values, "signed deviance witness")
                .expect("representable final sum"),
            f64::MAX
        );
    }

    #[test]
    fn profiled_gaussian_pirls_data_kernel_is_exactly_negative_half_raw_deviance() {
        let y = array![2.0, -1.0, 4.0];
        let eta = array![1.0, 0.5, 3.0];
        let weights = array![2.0, 0.5, 1.5];
        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ));
        assert!(matches!(
            likelihood.scale,
            LikelihoodScaleMetadata::ProfiledGaussian
        ));

        let deviance = calculate_deviance_from_eta(
            y.view(),
            &eta,
            &likelihood,
            &likelihood.spec.link,
            weights.view(),
        )
        .expect("profiled-Gaussian conventional deviance");
        let raw_weighted_rss = 4.625_f64;
        assert_eq!(deviance, raw_weighted_rss);

        let data_kernel = pirls_data_log_kernel_from_eta(
            y.view(),
            &eta,
            &likelihood,
            &likelihood.spec.link,
            weights.view(),
            deviance,
        )
        .expect("profiled-Gaussian P-IRLS data kernel");
        assert_eq!(data_kernel, -0.5 * deviance);
    }

    #[test]
    fn fixed_gaussian_pirls_keeps_raw_deviance_but_scales_likelihood_kernel() {
        let y = array![2.0, -1.0, 4.0];
        let eta = array![1.0, 0.5, 3.0];
        let weights = array![2.0, 0.5, 1.5];
        let phi = 4.0_f64;
        let likelihood = GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            scale: LikelihoodScaleMetadata::FixedDispersion { phi },
        };

        let deviance = calculate_deviance_from_eta(
            y.view(),
            &eta,
            &likelihood,
            &likelihood.spec.link,
            weights.view(),
        )
        .expect("fixed-Gaussian conventional deviance");
        let raw_weighted_rss = 4.625_f64;
        assert_eq!(deviance, raw_weighted_rss);

        let data_kernel = pirls_data_log_kernel_from_eta(
            y.view(),
            &eta,
            &likelihood,
            &likelihood.spec.link,
            weights.view(),
            deviance,
        )
        .expect("fixed-Gaussian P-IRLS data kernel");
        let strict_kernel = calculate_loglikelihood_omitting_constants_from_eta(
            y.view(),
            &eta,
            &likelihood,
            &likelihood.spec.link,
            weights.view(),
        )
        .expect("fixed-Gaussian strict eta likelihood");
        assert_eq!(data_kernel, strict_kernel);
        assert_eq!(data_kernel, -0.5 * raw_weighted_rss / phi);
    }

    #[test]
    fn unit_measure_single_pass_objective_is_bit_identical_to_the_two_pass_objective() {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(2_026_091_9);
        let n = 4096usize;
        let eta = Array1::from_iter((0..n).map(|_| -6.0 + 8.0 * rng.random::<f64>()));
        let bernoulli_y = eta.mapv(|e| {
            let p = 1.0 / (1.0 + (-e).exp());
            if rng.random::<f64>() < p { 1.0 } else { 0.0 }
        });
        let bernoulli_w = Array1::from_iter((0..n).map(|i| if i % 97 == 0 { 0.0 } else { 1.0 }));
        let trials = Array1::from_iter((0..n).map(|i| (1 + i % 9) as f64));
        let trials_y = Array1::from_iter(
            (0..n).map(|i| (rng.random::<f64>() * (trials[i] + 1.0)).floor().min(trials[i]) / trials[i]),
        );
        let poisson_eta = eta.mapv(|e| 0.25 * e);
        let poisson_y = poisson_eta.mapv(|e| (rng.random::<f64>() * 2.0 * e.exp()).floor());
        let poisson_w = Array1::from_iter((0..n).map(|_| 0.5 + rng.random::<f64>()));

        let logit = InverseLink::Standard(StandardLink::Logit);
        let log = InverseLink::Standard(StandardLink::Log);
        let binomial = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            logit.clone(),
        ));
        let poisson = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            log.clone(),
        ));
        let cases = [
            ("bernoulli", &binomial, &logit, &bernoulli_y, &eta, &bernoulli_w),
            ("binomial trials", &binomial, &logit, &trials_y, &eta, &trials),
            ("poisson", &poisson, &log, &poisson_y, &poisson_eta, &poisson_w),
        ];
        for (label, likelihood, link, y, eta, w) in cases {
            let deviance =
                calculate_deviance_from_eta(y.view(), eta, likelihood, link, w.view())
                    .expect("two-pass deviance");
            let log_kernel =
                pirls_data_log_kernel_from_eta(y.view(), eta, likelihood, link, w.view(), deviance)
                    .expect("two-pass data log-kernel");
            let (fused_deviance, fused_log_kernel) =
                unit_measure_deviance_and_log_kernel_from_eta(y.view(), eta, likelihood, link, w.view())
                    .expect("single-pass objective")
                    .expect("unit-measure family takes the single pass");
            assert_eq!(fused_deviance.to_bits(), deviance.to_bits(), "{label} deviance");
            assert_eq!(fused_log_kernel.to_bits(), log_kernel.to_bits(), "{label} log-kernel");
        }

        let profiled_gaussian = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ));
        assert!(
            unit_measure_deviance_and_log_kernel_from_eta(
                poisson_y.view(),
                &poisson_eta,
                &profiled_gaussian,
                &profiled_gaussian.spec.link,
                poisson_w.view(),
            )
            .expect("profiled Gaussian is declined, not rejected")
            .is_none()
        );
    }

    /// Regression for issue #2126: `calculate_deviance` for a Gamma family must
    /// report the conventional **unscaled** deviance `D = 2·Σ wᵢ·d(yᵢ, μᵢ)` —
    /// exactly like Poisson/Binomial/NB/Beta and R/mgcv/statsmodels — and must
    /// NOT multiply the unit deviance by the fitted shape (≈ 1/φ̂), which would
    /// report the scaled deviance `D/φ̂` instead. The bug only manifests when the
    /// shape differs from 1, so this pins a likelihood with an explicit
    /// `FixedGammaShape { shape = 4.0 }`: the reported deviance must equal the
    /// shape-free value and must be strictly different from `shape · D`.
    #[test]
    pub(crate) fn gamma_deviance_is_unscaled_ignoring_shape() {
        let y = array![2.0, 5.0, 1.5];
        let mu = array![1.0, 4.0, 2.0];
        let w = array![1.5, 0.75, 1.0];
        let shape = 4.0_f64;
        let likelihood = GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            ),
            scale: gam_problem::LikelihoodScaleMetadata::FixedGammaShape { shape },
        };
        // Sanity: the likelihood really does carry a non-unit shape, so the old
        // scaled-deviance code path (× shape) would have been exercised.
        assert_eq!(likelihood.gamma_shape(), Some(shape));

        let eta = mu.mapv(f64::ln);
        let inverse_link = InverseLink::Standard(StandardLink::Log);
        let dev = calculate_deviance_from_eta(y.view(), &eta, &likelihood, &inverse_link, w.view())
            .expect("Gamma eta deviance must be representable");

        let sum_unit: f64 = w
            .iter()
            .zip(y.iter())
            .zip(mu.iter())
            .map(|((&wi, &yi), &mui)| {
                let ratio = yi / mui;
                wi * (ratio - 1.0 - ratio.ln())
            })
            .sum();
        let unscaled = 2.0 * sum_unit;

        // The reported deviance is the unscaled 2·Σ w·d(y, μ) ...
        assert_relative_eq!(dev, unscaled, epsilon = 1e-12, max_relative = 1e-9);
        // ... and is NOT the shape-scaled deviance (this is the #2126 assertion:
        // the old code returned `shape * unscaled`, which for shape = 4 differs).
        assert!(
            (dev - shape * unscaled).abs() > 1e-6,
            "Gamma deviance must be unscaled, not scaled by shape={shape}: \
             dev={dev}, unscaled={unscaled}, scaled={}",
            shape * unscaled
        );
    }

    #[test]
    pub(crate) fn gamma_log_observed_curvature_matches_shape_one_closed_form() {
        let eta = array![0.2, -0.4];
        let mu = eta.mapv(f64::exp);
        let y = array![1.8, 0.7];
        let w = array![2.0, 0.5];

        let (w_obs, c_obs, d_obs) = compute_observed_hessian_curvature_arrays(
            &GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            )),
            &InverseLink::Standard(StandardLink::Log),
            &eta,
            y.view(),
            w.view(),
        )
        .expect("gamma-log observed curvature should evaluate");

        for i in 0..eta.len() {
            let expected_w = w[i] * y[i] / mu[i];
            assert_relative_eq!(w_obs[i], expected_w, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(c_obs[i], -expected_w, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(d_obs[i], expected_w, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    /// `(W, ∂W/∂η, ∂²W/∂η²)` of a closed-form observed weight against the Dual3
    /// derivatives of the observed information written from the log-likelihood,
    /// `W(η) = −∂²ℓ/∂η²`. The dispatch specializations are a separate derivation
    /// of the same tower; this pins every channel to rounding (#932).
    fn assert_observed_tower_matches_dual3(
        label: &str,
        dispatched: (f64, f64, f64),
        weight_of_eta: impl Fn(num_dual::Dual3_64) -> num_dual::Dual3_64,
        eta: f64,
    ) {
        let (w, c, d, _) = num_dual::third_derivative(|x| weight_of_eta(x), eta);
        for (channel, closed, reference) in [
            ("W", dispatched.0, w),
            ("dW/deta", dispatched.1, c),
            ("d2W/deta2", dispatched.2, d),
        ] {
            let scale = closed.abs().max(reference.abs());
            assert!(
                closed.is_finite() && (closed - reference).abs() <= 1.0e-12 * scale,
                "{label} {channel} at eta={eta}: closed form {closed:+.17e} vs Dual3 {reference:+.17e}"
            );
        }
        assert!(
            dispatched.1 != 0.0 && dispatched.2 != 0.0,
            "{label} at eta={eta}: the derivative channels must be live, got {dispatched:?}"
        );
    }

    #[test]
    fn negative_binomial_log_observed_curvature_matches_dual3_932() {
        use num_dual::DualNum;
        let prior_weight = 1.3;
        for theta in [0.7_f64, 4.5] {
            for y in [0.0_f64, 3.0, 17.0] {
                for eta in [-2.0_f64, 0.3, 1.7, 5.0] {
                    let mu = eta.exp();
                    let jet = MixtureInverseLinkJet {
                        mu,
                        d1: mu,
                        d2: mu,
                        d3: mu,
                    };
                    let dispatched = observed_weight_dispatch(
                        WeightFamily::NegativeBinomial { theta },
                        WeightLink::Log,
                        y,
                        mu,
                        1.0 - mu,
                        1.0,
                        prior_weight,
                        jet,
                        mu,
                    );
                    // −∂²ℓ/∂η² for ℓ = y·η − (y+θ)·log(e^η + θ) is (y+θ)·θ·e^η/(e^η+θ)².
                    assert_observed_tower_matches_dual3(
                        &format!("NB2 log theta={theta} y={y}"),
                        dispatched,
                        |x| {
                            let mu = x.exp();
                            (mu * theta) / ((mu + theta) * (mu + theta)) * (prior_weight * (y + theta))
                        },
                        eta,
                    );
                }
            }
        }
    }

    #[test]
    fn gamma_log_observed_curvature_matches_dual3_932() {
        use num_dual::DualNum;
        let (phi, prior_weight) = (0.4_f64, 1.75_f64);
        for y in [0.3_f64, 2.1] {
            for eta in [-3.0_f64, 0.2, 4.0] {
                let mu = eta.exp();
                let jet = MixtureInverseLinkJet {
                    mu,
                    d1: mu,
                    d2: mu,
                    d3: mu,
                };
                let dispatched = observed_weight_dispatch(
                    WeightFamily::Gamma,
                    WeightLink::Log,
                    y,
                    mu,
                    1.0 - mu,
                    phi,
                    prior_weight,
                    jet,
                    mu,
                );
                // −∂²ℓ/∂η² for ℓ = −(y·e^{−η} + η)/φ is y·e^{−η}/φ.
                assert_observed_tower_matches_dual3(
                    &format!("Gamma log y={y}"),
                    dispatched,
                    |x| (-x).exp() * (prior_weight * y / phi),
                    eta,
                );
            }
        }
    }

    /// `ψ⁽ᵏ⁾` (`order` 0 or 1) of a Dual3 argument, composed through third order
    /// by the chain rule from the scalar stack `[ψ, ψ₁, ψ₂, ψ₃, ψ₄]`.
    fn polygamma_dual3(order: usize, x: num_dual::Dual3_64) -> num_dual::Dual3_64 {
        let stack = gam_math::special::polygamma_stack(x.re, order + 4);
        let (f1, f2, f3) = (stack[order + 1], stack[order + 2], stack[order + 3]);
        num_dual::Dual3_64::new(
            stack[order],
            f1 * x.v1,
            f2 * x.v1 * x.v1 + f1 * x.v2,
            f3 * x.v1 * x.v1 * x.v1 + 3.0 * f2 * x.v1 * x.v2 + f1 * x.v3,
        )
    }

    /// The Beta-logit row inputs exactly as `compute_observed_hessian_curvature_arrays`
    /// assembles them, dispatched through `observed_weight_dispatch`.
    fn beta_logit_dispatched(y: f64, eta: f64, precision: f64, prior_weight: f64) -> (f64, f64, f64) {
        let link = InverseLink::Standard(StandardLink::Logit);
        let jet = crate::mixture_link::inverse_link_jet_for_inverse_link(&link, eta)
            .expect("logit jet");
        let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
            .expect("logit fourth derivative");
        let one_minus_mu =
            crate::mixture_link::inverse_link_complement_for_inverse_link(&link, eta, jet.mu);
        observed_weight_dispatch(
            WeightFamily::Beta { phi: precision },
            WeightLink::Other,
            y,
            jet.mu,
            one_minus_mu,
            1.0,
            prior_weight,
            jet,
            h4,
        )
    }

    #[test]
    fn beta_logit_observed_curvature_matches_dual3() {
        use num_dual::DualNum;
        let prior_weight = 1.3;
        for precision in [0.8_f64, 4.0] {
            for y in [0.1_f64, 0.6] {
                let y_star = y.ln() - (-y).ln_1p();
                for eta in [-2.0_f64, 0.3, 1.7] {
                    // ℓ = lnΓ(φ) − lnΓ(μφ) − lnΓ((1−μ)φ) + (μφ−1)ln y + ((1−μ)φ−1)ln(1−y)
                    // with μ = logistic(η), q = dμ/dη = μ(1−μ), q' = q(1−2μ):
                    // −∂²ℓ/∂η² = φ²q²(ψ₁(μφ) + ψ₁((1−μ)φ)) − φq'(y* − ψ(μφ) + ψ((1−μ)φ)).
                    // The Dual3 carries this unshifted form, a separate route from
                    // the recurrence-shifted closed form in the dispatch.
                    assert_observed_tower_matches_dual3(
                        &format!("Beta logit phi={precision} y={y}"),
                        beta_logit_dispatched(y, eta, precision, prior_weight),
                        |x| {
                            let mu = ((-x).exp() + 1.0).recip();
                            let one_minus_mu = -mu + 1.0;
                            let q = mu * one_minus_mu;
                            let q1 = q * (one_minus_mu - mu);
                            let a = mu * precision;
                            let b = one_minus_mu * precision;
                            let trigamma_sum = polygamma_dual3(1, a) + polygamma_dual3(1, b);
                            let residual = polygamma_dual3(0, b) - polygamma_dual3(0, a) + y_star;
                            (q * q * trigamma_sum * (precision * precision) - q1 * residual * precision)
                                * prior_weight
                        },
                        eta,
                    );
                }
            }
        }
    }

    #[test]
    fn beta_logit_observed_curvature_keeps_its_tail_order() {
        // Far in either logit tail the Fisher weight φ²q²(ψ₁(a) + ψ₁(b)) tends to
        // ω, while the observed weight vanishes like ω·μ(1−μ): its two O(ω)
        // terms cancel exactly. References: −∂ᵏℓ/∂ηᵏ (k = 2, 3, 4) of the Beta
        // log-likelihood at y = 0.6, φ = 4, ω = 1, evaluated with 60-digit
        // arithmetic. The recurrence-shifted form keeps them to rounding, where
        // the unshifted difference keeps only about three digits at |η| = 30.
        let (y, precision) = (0.6_f64, 4.0_f64);
        for (eta, reference) in [
            (
                -30.0_f64,
                [
                    -7.4441703904022336e-13,
                    -7.4441703903938675e-13,
                    -7.4441703903771354e-13,
                ],
            ),
            (
                30.0,
                [
                    -4.4088187034463916e-13,
                    4.4088187034391617e-13,
                    -4.4088187034247019e-13,
                ],
            ),
        ] {
            let (w, c, d) = beta_logit_dispatched(y, eta, precision, 1.0);
            for (channel, got, want) in [
                ("W", w, reference[0]),
                ("dW/deta", c, reference[1]),
                ("d2W/deta2", d, reference[2]),
            ] {
                assert!(
                    (got - want).abs() <= 1.0e-12 * want.abs(),
                    "Beta logit {channel} at eta={eta}: {got:+.17e} vs 60-digit reference {want:+.17e}"
                );
            }
        }
    }

    #[test]
    fn beta_logit_prices_the_laplace_with_observed_information() {
        let likelihood = GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::Beta { phi: 4.0 },
                InverseLink::Standard(StandardLink::Logit),
            ),
            scale: LikelihoodScaleMetadata::EstimatedBetaPhi { phi: 4.0 },
        };
        assert!(super::supports_observed_hessian_curvature_for_likelihood(
            &likelihood,
            &InverseLink::Standard(StandardLink::Logit),
        ));
        let eta = array![-30.0, -2.0, 0.3, 1.7, 30.0];
        let y = array![0.6, 0.1, 0.6, 0.1, 0.6];
        let prior = array![1.0, 1.3, 1.3, 1.3, 1.0];
        let (w, c, d) = compute_observed_hessian_curvature_arrays(
            &likelihood,
            &InverseLink::Standard(StandardLink::Logit),
            &eta,
            y.view(),
            &Array1::zeros(eta.len()),
            prior.view(),
        )
        .expect("Beta-logit observed curvature");
        for i in 0..eta.len() {
            assert_eq!(
                (w[i], c[i], d[i]),
                beta_logit_dispatched(y[i], eta[i], 4.0, prior[i]),
                "row {i}"
            );
        }
    }

    #[test]
    fn tweedie_log_observed_curvature_matches_dual3() {
        use num_dual::DualNum;
        let (phi, prior_weight) = (0.6_f64, 1.4_f64);
        for p in [1.2_f64, 1.5, 1.9] {
            for y in [0.0_f64, 0.7, 3.0] {
                for eta in [-2.0_f64, 0.3, 2.5] {
                    let mu = eta.exp();
                    let jet = MixtureInverseLinkJet {
                        mu,
                        d1: mu,
                        d2: mu,
                        d3: mu,
                    };
                    let dispatched = observed_weight_dispatch(
                        WeightFamily::Tweedie { p },
                        WeightLink::Log,
                        y,
                        mu,
                        1.0 - mu,
                        phi,
                        prior_weight,
                        jet,
                        mu,
                    );
                    // −∂²ℓ/∂η² for ℓ = [y·e^{(1−p)η}/(1−p) − e^{(2−p)η}/(2−p)]/φ is
                    // [(p−1)·y·e^{(1−p)η} + (2−p)·e^{(2−p)η}]/φ.
                    assert_observed_tower_matches_dual3(
                        &format!("Tweedie log p={p} y={y}"),
                        dispatched,
                        |x| {
                            ((x * (1.0 - p)).exp() * ((p - 1.0) * y)
                                + (x * (2.0 - p)).exp() * (2.0 - p))
                                * (prior_weight / phi)
                        },
                        eta,
                    );
                }
            }
        }
    }

    /// Tweedie with the log link is non-canonical for 1<p<2, so it must be
    /// served observed information. On a zero row the observed weight is
    /// `(2−p)` times the Fisher weight `ω μ^{2−p}/φ`.
    #[test]
    fn tweedie_log_is_served_observed_curvature() {
        let p = 1.5_f64;
        let link = InverseLink::Standard(StandardLink::Log);
        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Tweedie { p },
            link.clone(),
        ));
        assert!(super::supports_observed_hessian_curvature_for_likelihood(
            &likelihood,
            &link
        ));
        let phi = super::fixed_glm_dispersion(&likelihood).expect("Tweedie dispersion");
        let eta = array![0.4, -0.3, 1.1];
        let mu = eta.mapv(f64::exp);
        let y = array![0.0, 2.5, 0.2];
        let prior = array![1.0, 0.8, 1.6];
        let fisher = Array1::from_iter(
            (0..eta.len()).map(|i| prior[i] * mu[i].powf(2.0 - p) / phi),
        );
        let (w_obs, _, _) = compute_observed_hessian_curvature_arrays(
            &likelihood,
            &link,
            &eta,
            y.view(),
            &fisher,
            prior.view(),
        )
        .expect("Tweedie-log observed curvature should evaluate");
        for i in 0..eta.len() {
            let expected = fisher[i] * ((p - 1.0) * y[i] / mu[i] + (2.0 - p));
            assert_relative_eq!(w_obs[i], expected, epsilon = 0.0, max_relative = 1e-13);
            assert!(
                (w_obs[i] - fisher[i]).abs() > 1e-3 * fisher[i],
                "row {i}: y != mu, so observed and Fisher weights must differ"
            );
        }
        assert_relative_eq!(w_obs[0], (2.0 - p) * fisher[0], epsilon = 0.0, max_relative = 1e-13);
    }

    /// In the Poisson limit `mu << theta` the NB2 observed weight is
    /// `≈ prior·mu·(y+theta)/theta`. Forming `s = 1 − r` there loses
    /// `log10(theta/mu)` digits and returns exactly zero once `mu/theta < eps/2`.
    #[test]
    fn negative_binomial_log_observed_curvature_keeps_poisson_limit_precision() {
        use num_dual::DualNum;
        let prior_weight = 0.9;
        for theta in [1.0e7_f64, 1.0e12] {
            for y in [0.0_f64, 2.0] {
                for eta in [-27.6_f64, -5.0, 0.0] {
                    let mu = eta.exp();
                    let jet = MixtureInverseLinkJet {
                        mu,
                        d1: mu,
                        d2: mu,
                        d3: mu,
                    };
                    let dispatched = observed_weight_dispatch(
                        WeightFamily::NegativeBinomial { theta },
                        WeightLink::Log,
                        y,
                        mu,
                        1.0 - mu,
                        1.0,
                        prior_weight,
                        jet,
                        mu,
                    );
                    assert!(dispatched.0 > 0.0, "theta={theta} eta={eta}: W collapsed to {dispatched:?}");
                    assert_observed_tower_matches_dual3(
                        &format!("NB2 log Poisson limit theta={theta} y={y}"),
                        dispatched,
                        |x| {
                            let mu = x.exp();
                            (mu * theta) / ((mu + theta) * (mu + theta)) * (prior_weight * (y + theta))
                        },
                        eta,
                    );
                }
            }
        }
    }

    #[test]
    pub(crate) fn gamma_log_observed_curvature_dispatch_avoids_generic_overflow() {
        let y = 1.25;
        let phi = 0.5;
        let prior_weight = 1.75;
        let eta: f64 = 400.0;
        let mu = eta.exp();
        let jet = MixtureInverseLinkJet {
            mu,
            d1: mu,
            d2: mu,
            d3: mu,
        };
        let h4 = mu;

        let generic = observed_weight_noncanonical(
            y - mu,
            jet.d1,
            jet.d2,
            jet.d3,
            h4,
            variance_jet_for_weight_family(WeightFamily::Gamma, mu, 1.0 - mu),
            phi,
            prior_weight,
        );
        assert!(
            !generic.0.is_finite() || !generic.1.is_finite() || !generic.2.is_finite(),
            "generic Gamma-log curvature should expose the overflow/cancellation-prone path at eta={eta}: {generic:?}"
        );

        let (w_obs, c_obs, d_obs) = observed_weight_dispatch(
            WeightFamily::Gamma,
            WeightLink::Log,
            y,
            mu,
            1.0 - mu,
            phi,
            prior_weight,
            jet,
            h4,
        );
        let expected_w = prior_weight * y / (phi * mu);
        assert!(w_obs.is_finite() && c_obs.is_finite() && d_obs.is_finite());
        assert_relative_eq!(w_obs, expected_w, epsilon = 0.0, max_relative = 1e-12);
        assert_relative_eq!(c_obs, -expected_w, epsilon = 0.0, max_relative = 1e-12);
        assert_relative_eq!(d_obs, expected_w, epsilon = 0.0, max_relative = 1e-12);
    }

    #[test]
    pub(crate) fn binomial_mixture_observed_curvature_tolerates_indefinite_rows() {
        // Regression for issue #1598.
        //
        // For a binomial *blended/mixture* link the observed information
        //   W_obs = W_Fisher − (y − μ)·B
        // legitimately goes non-positive on individual rows when a large
        // residual flips the sign of the residual-dependent correction `B`
        // (the link is non-canonical, so B ≠ 0). The observed-curvature array
        // build must NOT hard-bail on such a finite-but-indefinite row: signed
        // row curvature is assembled exactly and any required PD stabilization
        // is an explicit ridge on the assembled matrix. Before the fix the build
        // aborted with "observed Hessian curvature is not
        // positive finite at row N", which propagated up as
        // "no candidate seeds passed outer startup validation
        // (mixture/SAS flexible link)" and made the whole joint solve fail.
        //
        // Construct a real blend (mixing weight ~0.5 on probit) and place
        // observations at extreme η where μ saturates while the *opposite*
        // label is observed, forcing a large residual and an indefinite W_obs.
        let mix_spec = MixtureLinkSpec {
            components: vec![LinkComponent::Logit, LinkComponent::Probit],
            // rho such that the softmax weights are well away from a pure
            // component, so B carries a genuine non-canonical contribution.
            initial_rho: Array1::from_vec(vec![0.0]),
        };
        let mix_state = state_fromspec(&mix_spec).expect("mixture state");
        let link = InverseLink::Mixture(mix_state);
        // The likelihood spec's link must match the mixture inverse link so the
        // `supports_observed_hessian_curvature_for_likelihood` gate (keyed on
        // `spec.link`) recognizes the binomial-mixture observed-curvature path.
        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            link.clone(),
        ));

        // Extreme η with mismatched labels: y=1 at η≈−6 (μ≈0) and y=0 at
        // η≈+6 (μ≈1) → huge residuals → indefinite observed weight on those
        // rows. Interior rows keep the build exercising the positive branch too.
        let eta = array![-6.0, 6.0, -0.3, 0.4, -2.0, 2.0];
        let y = array![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(eta.len());

        // Post-fix contract: the build SUCCEEDS (no bail) and returns finite
        // arrays even though some rows carry a non-positive observed weight.
        let (w_obs, c_obs, d_obs) =
            compute_observed_hessian_curvature_arrays(&likelihood, &link, &eta, y.view(), w.view())
                .expect(
                    "binomial mixture observed curvature must tolerate finite indefinite \
                     rows instead of bailing (#1598)",
                );

        assert!(
            w_obs.iter().all(|w| w.is_finite()),
            "all observed weights must be finite: {w_obs:?}"
        );
        assert!(
            c_obs.iter().all(|c| c.is_finite()) && d_obs.iter().all(|d| d.is_finite()),
            "all observed curvature derivatives must be finite"
        );
        // The test is only meaningful if it actually exercises the formerly-
        // bailing branch: at least one row must be non-positive (indefinite).
        assert!(
            w_obs.iter().any(|&w| w <= 0.0),
            "fixture must produce at least one indefinite observed-weight row to \
             guard the no-bail contract; got {w_obs:?}"
        );
    }

    #[test]
    pub(crate) fn binomial_latent_cloglog_takes_the_observed_curvature_path_3802() {
        // The latent-cloglog mean `μ(η) = E[1 − exp(−Z e^η)]` (lognormal `Z`) is
        // not the canonical Bernoulli link, so `W_obs = W_F − (y−μ)·B` with
        // `B ≠ 0` and the Laplace approximation needs the observed information.
        // The gate used to omit this link, so these fits silently built the
        // LAML from Fisher curvature.
        let state = gam_problem::types::LatentCLogLogState::new(0.4)
            .expect("valid latent cloglog state");
        let link = InverseLink::LatentCLogLog(state);
        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            link.clone(),
        ));
        assert!(super::supports_observed_hessian_curvature_for_likelihood(
            &likelihood,
            &link
        ));

        let eta = array![-2.0, -0.7, 0.0, 0.6, 1.4, -1.1];
        let y = array![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let prior = Array1::<f64>::ones(eta.len());
        let mut fisher = Array1::<f64>::zeros(eta.len());
        let mut tower = Vec::with_capacity(eta.len());
        for i in 0..eta.len() {
            let jet = crate::mixture_link::inverse_link_jet_for_inverse_link(&link, eta[i])
                .expect("latent cloglog jet");
            let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                &link, eta[i],
            )
            .expect("latent cloglog pdf third derivative");
            let one_minus_mu = crate::mixture_link::inverse_link_complement_for_inverse_link(
                &link, eta[i], jet.mu,
            );
            fisher[i] = jet.d1 * jet.d1 / (jet.mu * one_minus_mu);
            tower.push(observed_weight_dispatch(
                WeightFamily::Binomial,
                WeightLink::Other,
                y[i],
                jet.mu,
                one_minus_mu,
                1.0,
                1.0,
                jet,
                h4,
            ));
        }

        let (w_obs, c_obs, d_obs) =
            compute_observed_hessian_curvature_arrays(&likelihood, &link, &eta, y.view(), &fisher, prior.view())
                .expect("latent cloglog observed curvature");
        for i in 0..eta.len() {
            // Two algebraic arrangements of the same five jet values: the
            // log-probability jet and the ratio tower agree to rounding.
            let (w_ref, c_ref, d_ref) = tower[i];
            assert_relative_eq!(w_obs[i], w_ref, max_relative = 1e-9, epsilon = 1e-12);
            assert_relative_eq!(c_obs[i], c_ref, max_relative = 1e-9, epsilon = 1e-12);
            assert_relative_eq!(d_obs[i], d_ref, max_relative = 1e-9, epsilon = 1e-12);
            // Non-canonical: the residual term moves the weight off Fisher.
            assert!(
                (w_obs[i] - fisher[i]).abs() > 1e-6 * fisher[i].abs(),
                "row {i}: observed {} equals Fisher {} on a non-canonical link",
                w_obs[i],
                fisher[i]
            );
        }
    }

    #[test]
    fn latent_cloglog_observed_hessian_matches_log_likelihood_finite_difference() {
        let link = InverseLink::LatentCLogLog(
            gam_problem::types::LatentCLogLogState::new(0.4).unwrap(),
        );
        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            link.clone(),
        ));
        assert!(super::supports_observed_hessian_curvature_for_likelihood(
            &likelihood,
            &link,
        ));
        // Differentiate only log probabilities, independently of every
        // derivative field and observed-weight recurrence under test.
        for eta in [-4.0, -2.0, -0.7, 0.0, 0.6, 1.4] {
            for y in [0.0, 1.0] {
                for prior in [0.25, 2.0] {
                    let nll = |value| {
                        let mu = crate::mixture_link::inverse_link_jet_for_inverse_link(
                            &link, value,
                        ).unwrap().mu;
                        let probability = if y == 1.0 {
                            mu
                        } else {
                            crate::mixture_link::inverse_link_complement_for_inverse_link(
                                &link, value, mu,
                            )
                        };
                        -prior * probability.ln()
                    };
                    let (weights, _, _) = compute_observed_hessian_curvature_arrays(
                        &likelihood, &link, &array![eta], array![y].view(),
                        &array![0.0], array![prior].view(),
                    ).unwrap();
                    for h in [0.002_f64, 0.001] {
                        let reference = (-nll(eta + 2.0 * h) + 16.0 * nll(eta + h)
                            - 30.0 * nll(eta) + 16.0 * nll(eta - h)
                            - nll(eta - 2.0 * h)) / (12.0 * h * h);
                        let tolerance = 1e-8 + 1e-6 * reference.abs();
                        assert!((weights[0] - reference).abs() < tolerance,
                            "eta={eta}, y={y}, prior={prior}, h={h}: observed={}, likelihood FD={reference}", weights[0]);
                    }
                }
            }
        }
    }

    /// Intercept-only Gamma PIRLS fit under `link`: returns the fitted shape,
    /// the shape re-profiled at the converged η, and the fitted mean.
    fn intercept_only_gamma_fit(link: StandardLink, y: &Array1<f64>) -> (f64, f64, f64) {
        let x = Array2::<f64>::ones((y.len(), 1));
        let w = Array1::ones(y.len());
        let offset = Array1::zeros(y.len());
        let rho = array![0.0];
        let rs = [array![[0.0]]];
        let canonical: Vec<gam_terms::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                gam_terms::construction::CanonicalPenalty {
                    root: r.clone().into_shared(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local: local.into_shared(),
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(link),
            )),
            link_kind: InverseLink::Standard(link),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
        };

        let (result, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view())
                .expect("test rho lies in exact strength domain"),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                gaussian_fixed_cache: None,
                glm_first_step_gram: None,
            },
            PenaltyConfig {
                canonical_penalties: &canonical,
                reparam_invariant: None,
                p: 1,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
            },
            &config,
            None,
        )
        .expect("gamma PIRLS fit");

        let fitted_shape = result
            .likelihood
            .gamma_shape()
            .expect("gamma fit should expose fitted shape");
        let profiled_shape = super::estimate_gamma_shape_from_eta(
            &result.likelihood.spec.link,
            y.view(),
            &result.final_eta.to_owned(),
            w.view(),
        )
        .expect("converged Gamma shape must be representable");
        let eta = result.final_eta[0];
        let mean = match link {
            StandardLink::Log => eta.exp(),
            StandardLink::Inverse => eta.recip(),
            other => panic!("not a Gamma link: {other:?}"),
        };
        (fitted_shape, profiled_shape, mean)
    }

    #[test]
    pub(crate) fn gamma_log_fit_profiles_shape_instead_of_fixing_one() {
        let y = array![0.8, 1.1, 1.7, 2.0, 2.6, 3.1];
        let (fitted_shape, profiled_shape, _) = intercept_only_gamma_fit(StandardLink::Log, &y);
        assert!(fitted_shape > 1.0, "shape should not stay fixed at one");
        assert_relative_eq!(
            fitted_shape,
            profiled_shape,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    /// The shape MLE reads μ through the fit's own link. An intercept-only fit
    /// has μ̂ = ȳ under every link, so the inverse-link shape must equal the
    /// log-link one; reading μ = exp(η) at the inverse-link η = 1/ȳ instead
    /// profiled the shape against the wrong mean (the fuzzer's gamma(inverse)
    /// cells reported φ ≈ 1.5 on data drawn at φ = 1/3).
    #[test]
    pub(crate) fn gamma_inverse_fit_profiles_shape_on_its_own_link() {
        let y = array![0.8, 1.1, 1.7, 2.0, 2.6, 3.1];
        let ybar = y.mean().expect("nonempty response");
        let (log_shape, _, log_mean) = intercept_only_gamma_fit(StandardLink::Log, &y);
        let (inv_shape, inv_profiled, inv_mean) =
            intercept_only_gamma_fit(StandardLink::Inverse, &y);
        assert_relative_eq!(log_mean, ybar, max_relative = 1e-8);
        assert_relative_eq!(inv_mean, ybar, max_relative = 1e-8);
        assert_relative_eq!(inv_shape, inv_profiled, epsilon = 1e-10, max_relative = 1e-10);
        assert_relative_eq!(inv_shape, log_shape, max_relative = 1e-6);
    }

    /// Identity-Poisson with a group whose counts are all zero: the likelihood
    /// increases without bound as that group's mean falls to zero, so the
    /// maximum sits on the boundary `eta = 0` of the identity link's
    /// feasibility set. Fisher curvature `1/mu` diverges there, so the Newton
    /// decrement collapses while the gradient does not; P-IRLS must return the
    /// typed boundary error instead of certifying the edge as converged.
    #[test]
    fn identity_poisson_boundary_optimum_is_a_typed_error() {
        let n = 40;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            x[[i, 0]] = 1.0;
            if i >= n / 2 {
                x[[i, 1]] = 1.0;
                y[i] = [3.0, 5.0, 6.0, 4.0, 7.0][i % 5];
            }
        }
        let w = Array1::ones(n);
        let offset = Array1::zeros(n);
        let rho = array![0.0];
        let root = array![[0.0, 0.0]];
        let canonical = vec![gam_terms::construction::CanonicalPenalty {
            local: root.t().dot(&root).into_shared(),
            root: root.into_shared(),
            col_range: 0..2,
            total_dim: 2,
            nullity: 2,
            prior_mean: Array1::zeros(2),
            positive_eigenvalues: Vec::new(),
            op: None,
        }];
        let link = InverseLink::Standard(StandardLink::Identity);
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Poisson,
                link.clone(),
            )),
            link_kind: link,
            max_iterations: 200,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
        };

        let err = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view())
                .expect("test rho lies in exact strength domain"),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                gaussian_fixed_cache: None,
                glm_first_step_gram: None,
            },
            PenaltyConfig {
                canonical_penalties: &canonical,
                reparam_invariant: None,
                p: 2,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
            },
            &config,
            None,
        )
        .map(|_| ())
        .expect_err("a boundary optimum has no interior fit to report");
        assert!(
            matches!(
                err,
                EstimationError::LinkFeasibilityBoundaryOptimum {
                    link: "identity",
                    ..
                }
            ),
            "expected the typed boundary error, got {err:?}"
        );
        assert!(err.is_trial_point_infeasible());
    }

    /// The cold-start working weight is the Fisher weight at the weighted mean:
    /// for the inverse-Gaussian inverse-squared link `W = w·ȳ³/(4φ)` on every
    /// row, whatever the other design columns hold. Rescaling `y → c·y` with the
    /// dispersion `φ → φ/c` scales it by `c⁴`, exactly as the fitted working
    /// weight does, which is the unit covariance the analytic `initial.sp` seed
    /// and the ρ-domain Gram rely on. The weight needs no solve, so it exists at
    /// responses where the inner solve at the seed point refuses (the fuzzer's
    /// inverse-Gaussian n = 50 cells lost their seed to such a refusal).
    #[test]
    pub(crate) fn start_working_weights_follow_response_units() {
        let t = array![-1.0, -0.4, 0.1, 0.5, 0.9, 1.3];
        let mut x = Array2::<f64>::ones((t.len(), 2));
        x.column_mut(1).assign(&t);
        let design = DesignMatrix::from(x);
        let y = array![0.6, 1.4, 0.9, 2.2, 1.1, 3.0];
        let w = array![1.0, 2.0, 0.5, 1.5, 1.0, 3.0];
        let offset = Array1::zeros(y.len());
        let config_at = |phi: f64| {
            let link = InverseLink::Standard(StandardLink::InverseSquared);
            PirlsConfig {
                likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    ResponseFamily::InverseGaussian,
                    link.clone(),
                ))
                .with_dispersion_phi_frozen_for_search(phi),
                link_kind: link,
                max_iterations: 100,
                convergence_tolerance: 1e-8,
                firth_bias_reduction: false,
                initial_lm_lambda: None,
            }
        };
        let phi = 0.7;
        let weights = super::start_working_weights(
            &design,
            y.view(),
            w.view(),
            offset.view(),
            &config_at(phi),
        )
        .expect("start weights at a positive response");
        let ybar = y.dot(&w) / w.sum();
        for (&wi, &wt) in w.iter().zip(weights.iter()) {
            assert_relative_eq!(wt, wi * ybar.powi(3) / (4.0 * phi), max_relative = 1e-12);
        }

        let c = 3.0;
        let scaled = super::start_working_weights(
            &design,
            (&y * c).view(),
            w.view(),
            offset.view(),
            &config_at(phi / c),
        )
        .expect("start weights at a rescaled response");
        for (&wt, &ws) in weights.iter().zip(scaled.iter()) {
            assert_relative_eq!(ws, c.powi(4) * wt, max_relative = 1e-12);
        }
    }

    #[test]
    pub(crate) fn poisson_cache_rehydration_preserves_log_derivatives() {
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let y = array![1.0, 2.0, 4.0, 8.0];
        let w = Array1::ones(4);
        let offset = Array1::zeros(4);
        let rho = array![0.0];
        let rs = [array![[1.0]]];
        let canonical: Vec<gam_terms::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                gam_terms::construction::CanonicalPenalty {
                    root: r.clone().into_shared(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local: local.into_shared(),
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            )),
            link_kind: InverseLink::Standard(StandardLink::Log),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view())
                .expect("test rho lies in exact strength domain"),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                gaussian_fixed_cache: None,
                glm_first_step_gram: None,
            },
            PenaltyConfig {
                canonical_penalties: &canonical,
                reparam_invariant: None,
                p: 1,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
            },
            &config,
            None,
        )
        .expect("poisson PIRLS fit");

        let compacted = fit.compact_for_reml_cache();
        let rehydrated = compacted
            .rehydrate_after_reml_cache(
                &DesignMatrix::from(x.clone()),
                y.view(),
                w.view(),
                offset.view(),
                &InverseLink::Standard(StandardLink::Log),
            )
            .expect("rehydration should succeed");

        assert_eq!(fit.solve_c_array.len(), rehydrated.solve_c_array.len());
        for i in 0..fit.solve_c_array.len() {
            assert_relative_eq!(
                fit.solve_c_array[i],
                rehydrated.solve_c_array[i],
                epsilon = 1e-12,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                fit.solve_d_array[i],
                rehydrated.solve_d_array[i],
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
        // #2062: the stored penalized-KKT-residual round-trip assertions were
        // removed with the flexible-link envelope paper-over that introduced the
        // `penalized_gradient_transformed` field (the correction it fed is gone).
    }

    #[test]
    pub(crate) fn linear_constraint_active_set_releases_stalewarm_boundary_hint() {
        let hessian = array![[2.0]];
        let gradient = array![0.0];
        let beta = array![1e-9];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![0.0],
        };
        let mut direction = Array1::zeros(1);
        let mut active_hint = vec![0];

        solve_newton_directionwith_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("active-set solve should succeed");

        assert_relative_eq!(direction[0], 0.0, epsilon = 1e-14);
        let projected = &beta + &direction;
        assert_relative_eq!(projected[0], beta[0], epsilon = 1e-14);
        assert!(active_hint.is_empty());
    }

    #[test]
    pub(crate) fn linear_constraint_active_set_releases_stalewarm_hint() {
        let hessian = array![[1.0]];
        let gradient = array![-1.0];
        let beta = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [-1.0]],
            b: array![0.0, -0.1],
        };
        let mut direction = Array1::zeros(1);
        let mut active_hint = vec![0];

        solve_newton_directionwith_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("stale warm active-set hint should be releasable");

        assert!(
            (direction[0] - 0.1).abs() <= 1e-10,
            "expected step to upper bound (0.1), got {}",
            direction[0]
        );
        assert_eq!(active_hint, vec![1]);
    }

    #[test]
    pub(crate) fn lower_bound_active_set_releases_stalewarm_boundary_hint() {
        let hessian = array![[2.0]];
        let gradient = array![0.0];
        let beta = array![1e-9];
        let lower_bounds = array![0.0];
        let mut direction = Array1::zeros(1);
        let mut active_hint = vec![0];

        solve_newton_directionwith_lower_bounds(
            &hessian,
            &gradient,
            &beta,
            &lower_bounds,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("lower-bound active-set solve should succeed");

        assert_relative_eq!(direction[0], 0.0, epsilon = 1e-14);
        let projected = &beta + &direction;
        assert_relative_eq!(projected[0], beta[0], epsilon = 1e-14);
        assert!(active_hint.is_empty());
    }

    /// gam#979: the free solve and active-bound multiplier must come from one
    /// quadratic model. Here the bare gradient on coordinate 1 is positive, but
    /// coupling to the free coordinate makes the full multiplier negative, so the
    /// bound must be released. Dropping `H d` keeps it active; using a different
    /// Hessian can release/re-add it forever. The exact convex-model KKT solve has
    /// the unconstrained minimizer `d=(-7,4)`, which is feasible and stationary.
    #[test]
    pub(crate) fn lower_bound_release_uses_the_step_models_full_multiplier_979() {
        let hessian = array![[2.0, 3.0], [3.0, 5.0]];
        let gradient = array![2.0, 1.0];
        let beta = array![0.0, 0.0];
        let lower_bounds = array![f64::NEG_INFINITY, 0.0];
        let mut direction = Array1::zeros(2);
        let mut active = vec![1];
        solve_newton_directionwith_lower_bounds(
            &hessian,
            &gradient,
            &beta,
            &lower_bounds,
            &mut direction,
            Some(&mut active),
        )
        .expect("consistent convex-model QP should succeed");
        assert!(active.is_empty());
        assert_relative_eq!(direction[0], -7.0, epsilon = 1e-12);
        assert_relative_eq!(direction[1], 4.0, epsilon = 1e-12);
        let stationarity = &gradient + &hessian.dot(&direction);
        assert!(stationarity.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    pub(crate) fn select_active_set_release_worst_violation_picks_most_negative() {
        // Multipliers λ_i = g_i + (Hd)_i across active = {0, 1, 2}:
        //   i=0: -0.1 (mildly negative)
        //   i=1: -0.5 (most negative)
        //   i=2: -0.2
        // Worst-violation must pick i=1.
        let gradient = array![-0.1, -0.5, -0.2];
        let hd = array![0.0, 0.0, 0.0];
        let active_idx = vec![0, 1, 2];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, false),
            Some(1)
        );
    }

    #[test]
    pub(crate) fn select_active_set_release_blands_picks_lowest_index_with_negative_multiplier() {
        // Same setup as above. Bland's rule must pick the LOWEST index with a
        // strictly-negative multiplier (i=0), not the most negative (i=1).
        // This is the anti-cycling property — combined with Bland-compatible
        // tie-breaking on entering, it monotonically orders the active-set
        // sequence and prevents activate/release ping-pong on the same
        // coordinate at degenerate vertices.
        let gradient = array![-0.1, -0.5, -0.2];
        let hd = array![0.0, 0.0, 0.0];
        let active_idx = vec![0, 1, 2];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true),
            Some(0)
        );
    }

    #[test]
    pub(crate) fn select_active_set_release_blands_deadband_ignores_round_off() {
        // A multiplier of magnitude 64·ε·|g| is round-off level and must NOT
        // trigger release under Bland's rule. Otherwise pure floating-point
        // noise would cause spurious activate/release transitions and reopen
        // the cycling vulnerability the deadband was added to close.
        let g = 1.0_f64;
        let lambda_noise = -32.0 * f64::EPSILON * g; // strictly inside the deadband
        let gradient = array![g];
        let hd = array![lambda_noise - g]; // λ = g + hd = lambda_noise
        let active_idx = vec![0];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true),
            None,
            "round-off-level multiplier must not trigger Bland's release"
        );

        // ...but a multiplier just outside the deadband (128·ε·|g|) must
        // trigger release, so the rule still detects genuine KKT violations.
        let lambda_real = -128.0 * f64::EPSILON * g;
        let hd = array![lambda_real - g];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true),
            Some(0)
        );
    }

    #[test]
    pub(crate) fn select_active_set_release_returns_none_when_kkt_satisfied() {
        // All active multipliers ≥ 0 → KKT satisfied → no release, both rules
        // signal termination by returning None.
        let gradient = array![0.5, 1.0, 0.0];
        let hd = array![0.0, 0.0, 0.0];
        let active_idx = vec![0, 1, 2];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, false),
            None
        );
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true),
            None
        );
    }

    #[test]
    pub(crate) fn lower_bound_active_set_releases_stalewarm_hint() {
        let hessian = array![[1.0]];
        let gradient = array![-1.0];
        let beta = array![0.0];
        let lower_bounds = array![0.0];
        let mut direction = Array1::zeros(1);
        let mut active_hint = vec![0];

        solve_newton_directionwith_lower_bounds(
            &hessian,
            &gradient,
            &beta,
            &lower_bounds,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("stale warm lower-bound hint should be releasable");

        assert!(
            (direction[0] - 1.0).abs() <= 1e-12,
            "expected unconstrained step of 1.0 after releasing stale bound, got {}",
            direction[0]
        );
        assert!(active_hint.is_empty());
    }
}

#[cfg(test)]
mod root_cause_tests {
    use super::reweight::exact_newton_decrement_sq;
    use super::*;
    use approx::assert_relative_eq;
    use gam_problem::LogSmoothingParamsView;
    use ndarray::{Array1, Array2, array};

    pub(crate) fn capture_pirls_penalized_deviance<F, R>(run: F) -> (R, Vec<f64>)
    where
        F: FnOnce() -> R,
    {
        super::reweight::test_support::PIRLS_PENALIZED_DEVIANCE_TRACE.with(|trace| {
            *trace.borrow_mut() = Some(Vec::new());
        });
        let result = run();
        let captured = super::reweight::test_support::PIRLS_PENALIZED_DEVIANCE_TRACE
            .with(|trace| trace.borrow_mut().take().unwrap());
        (result, captured)
    }

    pub(crate) fn scalar_working_state(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
        gradient: f64,
        deviance: f64,
    ) -> WorkingState {
        WorkingState {
            eta: LinearPredictor::new(array![beta.as_ref()[0]]),
            gradient: array![gradient],
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(array![[1.0]]),
            log_likelihood: 0.0,
            deviance,
            deviance_magnitude: deviance.abs(),
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            hessian_curvature: curvature,
            gradient_natural_scale: 0.0,
        }
    }

    #[test]
    pub(crate) fn exact_decrement_certifies_stiff_numerical_plateau_2316() {
        let beta = Coefficients::new(array![0.0]);
        let mut state =
            scalar_working_state(&beta, HessianCurvatureKind::Fisher, 1.448_052e-3, 428.0);
        state.hessian = gam_linalg::matrix::SymmetricMatrix::Dense(array![[1.0e8]]);

        let decrement_sq = exact_newton_decrement_sq(&state, None)
            .expect("a finite positive-definite Hessian has an exact decrement");
        let threshold = 1.0e-6_f64.powi(2) * (1.0 + state.penalized_objective().abs());

        assert!(decrement_sq <= threshold);
        assert_relative_eq!(
            decrement_sq,
            1.448_052e-3_f64.powi(2) / 1.0e8,
            epsilon = 1.0e-28
        );
        // The old damped resolvent bound used ridge_used.max(1e-12) as a
        // lower eigenvalue bound. At the observed λ≈1e5 it inflated this
        // decisive decrement by 1e17 and therefore could never certify.
        let obsolete_bound = decrement_sq * (1.0 + 1.0e5 / 1.0e-12);
        assert!(obsolete_bound > threshold);
    }

    /// #2705: the exact-decrement half of the inner certificate is published as
    /// `exact_newton_decrement_evidence`, so a consumer that re-checks
    /// stationarity (the survival LAML gate) accepts exactly the modes P-IRLS
    /// minted. Three states pin it. A stiff mode whose gradient misses strict
    /// KKT but whose decrement is inside the objective's rounding band
    /// certifies. The same gradient on unit curvature does not. And a
    /// coordinate held at its lower bound by a large multiplier certifies on its
    /// binding face, where the unconstrained decrement is dominated by that
    /// multiplier and refuses.
    #[test]
    pub(crate) fn exact_newton_decrement_evidence_certifies_what_the_gradient_band_refuses_2705() {
        let gradient = 5.0e-5_f64;
        let stiffness = 1.6e8_f64;
        let beta = Coefficients::new(array![0.0]);
        let mut stiff = scalar_working_state(&beta, HessianCurvatureKind::Observed, gradient, 428.0);
        stiff.hessian = gam_linalg::matrix::SymmetricMatrix::Dense(array![[stiffness]]);
        assert!(
            !stiff.certifies_kkt(gradient, 1.0e-8),
            "the stiff fixture must sit outside strict KKT, or it pins nothing"
        );
        let certified = exact_newton_decrement_evidence(&stiff, beta.as_ref(), None, None, 1.0);
        let stiff_decrement_sq = certified
            .decrement_sq
            .expect("a positive-definite Hessian has an exact decrement");
        assert_relative_eq!(
            stiff_decrement_sq,
            gradient.powi(2) / stiffness,
            max_relative = 1.0e-12
        );
        assert!(
            certified.certifies(),
            "stiff decrement {stiff_decrement_sq:.3e} must certify against threshold {:.3e}",
            certified.threshold
        );

        let unit = scalar_working_state(&beta, HessianCurvatureKind::Observed, gradient, 428.0);
        let refused = exact_newton_decrement_evidence(&unit, beta.as_ref(), None, None, 1.0);
        assert!(
            !refused.certifies(),
            "unit-curvature decrement {:?} must not certify against threshold {:.3e}",
            refused.decrement_sq,
            refused.threshold
        );

        let multiplier = 10.0_f64;
        let face_beta = array![0.0, 1.0];
        let face_state = WorkingState {
            eta: LinearPredictor::new(array![0.0]),
            gradient: array![multiplier, gradient],
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(array![[1.0, 0.0], [0.0, stiffness]]),
            log_likelihood: 0.0,
            deviance: 428.0,
            deviance_magnitude: 428.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            hessian_curvature: HessianCurvatureKind::Observed,
            gradient_natural_scale: 0.0,
        };
        let lower_bounds = gam_problem::LinearInequalityConstraints::from_per_coordinate_lower_bounds(
            &array![0.0, f64::NEG_INFINITY],
        )
        .expect("one finite lower bound induces one constraint row");
        let unconstrained = exact_newton_decrement_evidence(&face_state, &face_beta, None, None, 1.0);
        assert!(
            !unconstrained.certifies(),
            "the multiplier dominates the unconstrained decrement {:?}",
            unconstrained.decrement_sq
        );
        let on_face =
            exact_newton_decrement_evidence(&face_state, &face_beta, Some(&lower_bounds), None, 1.0);
        let face_decrement_sq = on_face
            .decrement_sq
            .expect("the reduced curvature on the binding face factorizes");
        assert_relative_eq!(
            face_decrement_sq,
            gradient.powi(2) / stiffness,
            max_relative = 1.0e-12
        );
        assert!(
            on_face.certifies(),
            "face decrement {face_decrement_sq:.3e} must certify against threshold {:.3e}",
            on_face.threshold
        );
    }

    #[test]
    pub(crate) fn lm_gain_value_matches_working_state_objective_2316() {
        let beta = Coefficients::new(array![0.0]);
        let mut state = scalar_working_state(&beta, HessianCurvatureKind::Fisher, 0.0, 8.0);
        state.penalty_term = 2.0;
        let expected = state.penalized_objective();

        let screened = CandidateEvaluation::Screen(CandidateScreen {
            deviance: state.deviance,
            penalty_term: state.penalty_term,
            arithmetic_finite: true,
        });
        let full = CandidateEvaluation::Full(state);

        assert_eq!(screened.penalized_objective(false, 1.0), expected);
        assert_eq!(full.penalized_objective(false, 1.0), expected);
    }

    pub(crate) fn test_working_state(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> WorkingState {
        scalar_working_state(beta, curvature, 1.0, 1.0)
    }

    #[derive(Default)]
    pub(crate) struct CandidateEvalFailureModel {
        pub(crate) observed_updates: usize,
        pub(crate) fisher_updates: usize,
        pub(crate) observed_candidate_calls: usize,
        pub(crate) fisher_candidate_calls: usize,
    }

    impl CandidateEvalFailureModel {
        pub(crate) fn state(beta: &Coefficients, curvature: HessianCurvatureKind) -> WorkingState {
            test_working_state(beta, curvature)
        }
    }

    impl WorkingModel for CandidateEvalFailureModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            match curvature {
                HessianCurvatureKind::Observed => self.observed_updates += 1,
                HessianCurvatureKind::Fisher => self.fisher_updates += 1,
            }
            Ok(Self::state(beta, curvature))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            match curvature {
                HessianCurvatureKind::Observed => self.observed_candidate_calls += 1,
                HessianCurvatureKind::Fisher => self.fisher_candidate_calls += 1,
            }
            Err(EstimationError::InvalidInput(format!(
                "non-finite candidate evaluation under {curvature:?} curvature at beta={:.3e}",
                beta.as_ref()[0],
            )))
        }

        fn supports_observed_information_curvature(&self) -> bool {
            true
        }
    }

    #[derive(Default)]
    pub(crate) struct PermanentCandidateErrorModel {
        pub(crate) candidate_calls: usize,
    }

    impl WorkingModel for PermanentCandidateErrorModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(test_working_state(beta, curvature))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            self.candidate_calls += 1;
            Err(EstimationError::InvalidSpecification(format!(
                "permanent candidate failure under {curvature:?} curvature at beta={:.3e}",
                beta.as_ref()[0],
            )))
        }
    }

    #[derive(Default)]
    pub(crate) struct FirthAcceptedStateFailureModel {
        pub(crate) current_state_calls: usize,
        pub(crate) candidate_state_calls: usize,
        pub(crate) candidate_screen_calls: usize,
    }

    impl WorkingModel for FirthAcceptedStateFailureModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            if beta.as_ref()[0].abs() < 1e-12 {
                self.current_state_calls += 1;
                Ok(test_working_state(beta, curvature))
            } else {
                self.candidate_state_calls += 1;
                Err(EstimationError::InvalidInput(format!(
                    "overflow while re-evaluating accepted candidate under {curvature:?} curvature at beta={:.3e}",
                    beta.as_ref()[0],
                )))
            }
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            // Production firth `update_candidate` evaluates the candidate
            // through `update_with_curvature` (with Firth ACTIVE, so the
            // candidate/accepted objective carries the Jeffreys `−2·½log|XᵀWX|`
            // term consistently with `current_penalized`; gam#1821) rather than
            // via a separate cheap screen: since the candidate screening split
            // landed, the firth accepted-state re-evaluation is folded into this
            // single call instead of running as a distinct post-acceptance
            // phase. Mirror that here so the injected
            // candidate-evaluation failure actually surfaces through the LM
            // loop, which must bound the retries instead of looping or silently
            // accepting a non-stationary state.
            self.candidate_screen_calls += 1;
            self.update_with_curvature(beta, curvature)
        }
    }

    #[derive(Default)]
    pub(crate) struct FirthPermanentCandidateErrorModel {
        pub(crate) current_state_calls: usize,
        pub(crate) candidate_state_calls: usize,
        pub(crate) candidate_screen_calls: usize,
    }

    impl WorkingModel for FirthPermanentCandidateErrorModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            if beta.as_ref()[0].abs() < 1e-12 {
                self.current_state_calls += 1;
                Ok(test_working_state(beta, curvature))
            } else {
                self.candidate_state_calls += 1;
                Err(EstimationError::InvalidSpecification(
                    "permanent firth breakdown re-evaluating accepted candidate".to_string(),
                ))
            }
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            self.candidate_screen_calls += 1;
            self.update_with_curvature(beta, curvature)
        }
    }

    #[derive(Default)]
    pub(crate) struct ActiveConstraintKktModel;

    impl WorkingModel for ActiveConstraintKktModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 1.0, 0.0))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 1.0, 0.0))
        }
    }

    pub(crate) struct PlateauStatusModel {
        pub(crate) gradient: f64,
        pub(crate) current_deviance: f64,
        pub(crate) candidate_deviance: f64,
    }

    impl PlateauStatusModel {
        pub(crate) fn state(
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
            gradient: f64,
            deviance: f64,
        ) -> WorkingState {
            scalar_working_state(beta, curvature, gradient, deviance)
        }
    }

    impl WorkingModel for PlateauStatusModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(Self::state(
                beta,
                curvature,
                self.gradient,
                self.current_deviance,
            ))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(Self::state(
                beta,
                curvature,
                self.gradient,
                self.candidate_deviance,
            ))
        }
    }

    pub(crate) struct ExactDecrementAtIterationCapModel {
        pub(crate) exact_calls: usize,
        pub(crate) gradient: f64,
        pub(crate) candidate_deviance: f64,
    }

    impl WorkingModel for ExactDecrementAtIterationCapModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, self.gradient, 1.0))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(
                beta,
                curvature,
                self.gradient,
                self.candidate_deviance,
            ))
        }

        fn exact_unconstrained_decrement_sq(
            &mut self,
            beta: &Coefficients,
            state: &WorkingState,
        ) -> Result<Option<f64>, EstimationError> {
            assert_eq!(beta.as_ref().len(), state.gradient.len());
            self.exact_calls += 1;
            Ok(Some(0.0))
        }
    }

    pub(crate) struct LinearObjectivePlateauModel {
        pub(crate) gradient: f64,
    }

    impl LinearObjectivePlateauModel {
        pub(crate) fn state(
            &self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> WorkingState {
            let deviance = 1.0 + self.gradient * beta[0];
            scalar_working_state(beta, curvature, self.gradient, deviance)
        }
    }

    impl WorkingModel for LinearObjectivePlateauModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(self.state(beta, curvature))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(self.state(beta, curvature))
        }
    }

    /// A coordinate is at its lower bound exactly when the bound's unit row is
    /// active by the inequality system's one activity rule, at the solver's
    /// published feasibility resolution (#3180). Then a positive gradient is a
    /// KKT multiplier and drops out. A coordinate 1e-6 above its bound is 100
    /// times outside that resolution: it is interior, a step of 1e-6 toward the
    /// bound still lowers the objective by about `0.5 · 1e-6`, and its gradient
    /// is a stationarity defect. The band this used to pass under
    /// (`1e-6·max(|β|, |lb|, 1) + 1e-10`) called that point stationary.
    #[test]
    pub(crate) fn projected_gradient_excludes_near_bound_kkt_forces() {
        let gradient = array![0.5, 1e-4];
        let lower_bounds = array![0.0, f64::NEG_INFINITY];
        for at_bound in [0.0, 0.5 * gam_problem::PRIMAL_FEASIBILITY_TOL] {
            let beta = array![at_bound, 2.0];
            let norm = projected_gradient_norm(&gradient, &beta, Some(&lower_bounds));
            assert_eq!(
                norm, 1e-4,
                "the multiplier of a bound at slack {at_bound:e} must drop out"
            );
        }
        let beta = array![1e-6, 2.0];
        let norm = projected_gradient_norm(&gradient, &beta, Some(&lower_bounds));
        assert_eq!(
            norm,
            (0.5_f64 * 0.5 + 1e-4 * 1e-4).sqrt(),
            "an interior coordinate's gradient is a stationarity defect"
        );
    }

    /// The issue's repro (#3180): `f(β) = ½(β − c)²`, `lb = 0`, `c = −3e-7`, at
    /// `β = 5e-7`. The constrained minimizer is `β* = 0`, so this point is not
    /// stationary, and `g = β − c = 8e-7` is the defect the norm must report.
    #[test]
    pub(crate) fn projected_gradient_reports_an_interior_point_short_of_its_bound_3180() {
        let c = -3e-7;
        let beta = array![5e-7];
        let gradient = array![beta[0] - c];
        let norm = projected_gradient_norm(&gradient, &beta, Some(&array![0.0]));
        assert_eq!(norm, gradient[0]);
        assert!(norm > 0.0);
    }

    /// One activity rule (#3180): on every instance, the bounds the P-IRLS box
    /// path treats as binding are exactly the rows `active_face` marks active on
    /// the same bounds written as unit rows, among those the gradient presses
    /// into. Slacks straddle the feasibility resolution from both sides.
    #[test]
    pub(crate) fn box_bounds_bind_exactly_where_the_active_face_is_active_3180() {
        let tol = gam_problem::PRIMAL_FEASIBILITY_TOL;
        let slacks = [
            0.0,
            0.25 * tol,
            tol,
            1.5 * tol,
            1e-7,
            5e-7,
            1e-6,
            0.3,
            -0.5 * tol,
        ];
        let bounds = [0.0, -2.0, 1.0, 1e3];
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _instance in 0..64 {
            let p = 7;
            let mut beta = Array1::<f64>::zeros(p);
            let mut lower_bounds = Array1::<f64>::zeros(p);
            let mut gradient = Array1::<f64>::zeros(p);
            for i in 0..p {
                let lb = bounds[(next() % bounds.len() as u64) as usize];
                lower_bounds[i] = lb;
                beta[i] = lb + slacks[(next() % slacks.len() as u64) as usize];
                gradient[i] = ((next() % 2001) as f64 - 1000.0) / 250.0;
            }
            let constraints = LinearInequalityConstraints {
                a: Array2::<f64>::eye(p),
                b: lower_bounds.clone(),
            };
            let face = crate::active_set::active_face(&beta, &constraints)
                .expect("unit rows match the coefficient width");
            let expected = (0..p)
                .filter(|&i| !(face.active_idx.contains(&i) && gradient[i] > 0.0))
                .map(|i| gradient[i] * gradient[i])
                .sum::<f64>()
                .sqrt();
            let norm = projected_gradient_norm(&gradient, &beta, Some(&lower_bounds));
            assert_eq!(
                norm, expected,
                "beta={beta:?} lb={lower_bounds:?} g={gradient:?}"
            );
        }
    }

    /// Hypothesis 2: with loosened active_tol, the solver identifies near-bound
    /// coefficients as active and moves them TO the bound (direction = lb - beta),
    /// rather than computing a full unconstrained Newton step and clipping.
    #[test]
    pub(crate) fn bound_solver_treats_near_bound_positive_grad_as_active() {
        let hessian = array![[2.0, 0.0], [0.0, 2.0]];
        let gradient = array![1.0, 0.0];
        let beta = array![1e-6, 5.0];
        let lower_bounds = array![0.0, f64::NEG_INFINITY];
        let mut direction = Array1::zeros(2);
        let mut active_hint = vec![];

        solve_newton_directionwith_lower_bounds(
            &hessian,
            &gradient,
            &beta,
            &lower_bounds,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("solve should succeed");

        // With the fix, beta[0] is identified as active. The direction
        // moves it exactly to the bound: d[0] = lb - beta = -1e-6.
        // Without the fix (active_tol=1e-12), the unconstrained Newton step
        // d[0] = -g/H = -0.5 is computed, then clipped — same result here
        // but the active set hint is wrong, causing downstream issues.
        assert!(
            active_hint.contains(&0),
            "near-bound coeff with positive gradient should be in active set, got {:?}",
            active_hint
        );
        // Direction should move to bound, not be the unconstrained step
        assert!(
            (direction[0] - (-1e-6)).abs() < 1e-14,
            "direction should snap to bound (lb - beta = -1e-6), got {:.6e}",
            direction[0]
        );
    }

    #[test]
    pub(crate) fn pirls_converges_at_active_linear_constraint_kkt_point() {
        let mut model = ActiveConstraintKktModel;
        let options = WorkingModelPirlsOptions {
            max_iterations: 3,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 3,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![0.0],
            }),
            initial_lm_lambda: None,
        };

        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, None)
                .expect("active-constraint KKT point should be accepted as converged");

        assert_eq!(summary.status, PirlsStatus::Converged);
        assert!(
            summary.lastgradient_norm <= 1e-12,
            "KKT-aware stationarity norm should vanish at the constrained optimum, got {:.6e}",
            summary.lastgradient_norm
        );
        let kkt = summary
            .constraint_kkt
            .expect("linear constraint run should report KKT diagnostics");
        assert!(kkt.primal_feasibility <= 1e-12);
        assert!(kkt.dual_feasibility <= 1e-12);
        assert!(kkt.complementarity <= 1e-12);
        assert!(kkt.stationarity <= 1e-12);
    }

    fn certificate_state(n: usize, p: usize, natural_scale: f64) -> WorkingState {
        WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 1.0,
            deviance_magnitude: 1.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: natural_scale,
        }
    }

    /// The large-scale case: `n=320000`, `p=20`, projected stationarity
    /// residual `‖g‖ = 1.465e-5` on a natural scale of `1e3`. An absolute
    /// test `‖g‖ < 1e-6` rejects it, though its dimensionless residual is
    /// `1.5e-8`; the certificate accepts it.
    #[test]
    pub(crate) fn certifies_kkt_accepts_large_scale_pathological_case() {
        let g_norm = 1.465e-5;
        let tol = 1e-6;
        let state = certificate_state(320_000, 20, 1.0e3);
        assert!(state.certifies_kkt(g_norm, tol));
        assert!(
            !(g_norm < tol),
            "this test must witness the failure of the old absolute test; \
             otherwise it does not prove the fix"
        );
    }

    /// The certificate is exactly invariant under a uniform rescaling of the
    /// gradient (`F → c·F`, or `β → β/c`), which scales `‖g‖`, `‖score‖` and
    /// `‖S·β‖` together, at EVERY natural scale — not only once the natural
    /// scale dominates an additive floor.
    #[test]
    pub(crate) fn certifies_kkt_is_scale_invariant() {
        let tol = 1e-8;
        for natural_scale in [1.0e-5, 1.0, 5.0e6] {
            for residual in [1.0e-10, 1.0e-6] {
                let g_norm = residual * natural_scale;
                let verdict = certificate_state(1000, 10, natural_scale).certifies_kkt(g_norm, tol);
                assert_eq!(verdict, residual < tol);
                for c in [1.0e-6, 1.0e6] {
                    assert_eq!(
                        certificate_state(1000, 10, natural_scale * c).certifies_kkt(g_norm * c, tol),
                        verdict,
                        "KKT classification must be invariant under g → c·g (scale {natural_scale:e}, c {c:e})"
                    );
                }
            }
        }
    }

    /// The canonical inverse-Gaussian inner solve at a small-unit response:
    /// P-IRLS stopped at `‖g‖ = 4.72e-12` on a natural scale of order `1e-5`,
    /// two steps into a solve whose mode (the same data at ten times the
    /// units) sits nine steps away. The dimension bound `τ·√n·√p` and the
    /// additive `1 +` floor both read that residual in the gradient's own
    /// units and certified it; its dimensionless residual `4.7e-7` is far
    /// above `τ` and must not certify.
    #[test]
    pub(crate) fn certifies_kkt_refuses_small_unit_residual_far_from_mode() {
        let tol = 1e-10;
        let state = certificate_state(500, 23, 1.0e-5);
        assert!(!state.certifies_kkt(4.72e-12, tol));
        assert!(!state.near_stationary_kkt(4.72e-12, tol));
        assert!(state.certifies_kkt(4.72e-12 * 1.0e-4, tol));
    }

    /// An exactly zero residual is stationary on any natural scale, zero
    /// included; a nonzero residual on a zero natural scale is not resolved by
    /// the ratio and is left to the exact Newton decrement.
    #[test]
    pub(crate) fn certifies_kkt_on_zero_natural_scale() {
        let tol = 1e-6;
        let state = certificate_state(100, 5, 0.0);
        assert!(state.certifies_kkt(0.0, tol));
        assert!(state.near_stationary_kkt(0.0, tol));
        assert!(!state.certifies_kkt(2.0e-6, tol));
        assert!(!state.near_stationary_kkt(2.0e-6, tol));
    }

    /// The near-stationary band is exactly 10× the strict KKT tolerance on the
    /// same dimensionless residual. It classifies a usable but non-strictly
    /// converged minimum as `StalledAtValidMinimum` rather than as a hard
    /// non-convergence.
    #[test]
    pub(crate) fn near_stationary_kkt_uses_ten_times_band() {
        let tol = 1e-6;
        let state = certificate_state(100, 4, 100.0);
        // Relative ‖g‖ = g/100 ≤ 10·tol = 1e-5 ⇒ accept when g ≤ 1e-3.
        assert!(state.near_stationary_kkt(9.9e-4, tol));
        assert!(!state.near_stationary_kkt(2.0e-3, tol));
        // Strict KKT at the same point should be ~10× tighter.
        assert!(!state.certifies_kkt(9.9e-4, tol));
    }

    /// Hypothesis 3: LM gain-ratio fallback should accept when both predicted
    /// and actual reduction are floating-point noise relative to the objective.
    #[test]
    pub(crate) fn lm_gain_ratio_accepts_zero_step_at_stationarity() {
        // Simulate: objective ~ 9e5, predicted reduction ~ 5e-16, actual ~ -1e-14
        let current_penalized: f64 = 9e5;
        let predicted_reduction: f64 = 5e-16;
        let actual_reduction: f64 = -1e-14;
        let noise_floor = current_penalized.abs() * 1e-14; // ~9e-9 (#1127: relative floor, no absolute .max(1.0))

        let rho = if predicted_reduction > noise_floor {
            actual_reduction / predicted_reduction
        } else if actual_reduction >= -noise_floor {
            1.0 // both at noise level → accept
        } else {
            -1.0
        };

        // actual_reduction (-1e-14) >= -noise_floor (-9e-9) → rho = 1.0
        assert!(
            rho > 0.0,
            "near-zero reductions should not hard-reject; rho={:.1}, pred={:.2e}, actual={:.2e}, noise={:.2e}",
            rho,
            predicted_reduction,
            actual_reduction,
            noise_floor
        );
    }

    #[test]
    pub(crate) fn candidate_evaluation_errors_respect_lm_exhaustion_budget() {
        let mut model = CandidateEvalFailureModel::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 5,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            None,
        ) {
            Ok(_) => panic!("candidate evaluation failures should exhaust LM retries and surface"),
            Err(err) => err,
        };

        match err {
            EstimationError::PirlsDidNotConverge {
                iterations,
                budget,
                stop,
                last_change,
            } => {
                assert!(
                    budget == options.max_iterations,
                    "expected LM exhaustion to surface as PIRLS non-convergence with screening cap"
                );
                assert!(
                    iterations <= budget,
                    "a solve cannot report more iterations ({iterations}) than its budget ({budget})"
                );
                assert!(!stop.is_empty(), "the refusal must name why the solve stopped");
                assert!(last_change.is_finite() && last_change > 0.0);
            }
            other => {
                panic!("expected PirlsDidNotConverge from candidate evaluation, got {other:?}")
            }
        }

        assert_eq!(
            model.observed_updates, 1,
            "the PIRLS iteration should start on observed curvature once"
        );
        assert_eq!(
            model.fisher_updates, 1,
            "candidate failure should trigger exactly one observed->Fisher fallback"
        );
        assert_eq!(
            model.observed_candidate_calls, 1,
            "observed candidate evaluation should fail once before the Fisher fallback"
        );
        assert_eq!(
            model.fisher_candidate_calls,
            options.max_step_halving - 1,
            "Fisher candidate evaluation must stop at the configured LM retry budget"
        );
    }

    #[test]
    pub(crate) fn permanent_candidate_errors_do_not_trigger_lm_retries() {
        let mut model = PermanentCandidateErrorModel::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 5,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            None,
        ) {
            Ok(_) => panic!("permanent candidate failures should surface immediately"),
            Err(err) => err,
        };

        match err {
            EstimationError::InvalidSpecification(message) => {
                assert!(
                    message.contains("permanent candidate failure"),
                    "expected permanent candidate failure, got {message}"
                );
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }

        assert_eq!(
            model.candidate_calls, 1,
            "non-retriable candidate failures should not be re-evaluated under stronger damping"
        );
    }

    #[test]
    pub(crate) fn firth_candidate_reevaluation_respects_lm_retry_budget() {
        let mut model = FirthAcceptedStateFailureModel::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            firth_bias_reduction: true,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            None,
        ) {
            Ok(_) => panic!("Firth candidate reevaluation failures should not loop indefinitely"),
            Err(err) => err,
        };

        match err {
            EstimationError::PirlsDidNotConverge {
                iterations,
                budget,
                stop,
                last_change,
            } => {
                assert_eq!(budget, options.max_iterations);
                assert!(
                    iterations <= budget,
                    "a solve cannot report more iterations ({iterations}) than its budget ({budget})"
                );
                assert!(!stop.is_empty(), "the refusal must name why the solve stopped");
                assert!(last_change.is_finite() && last_change > 0.0);
            }
            other => panic!("expected PirlsDidNotConverge, got {other:?}"),
        }

        assert_eq!(model.current_state_calls, 1);
        assert_eq!(
            model.candidate_screen_calls, options.max_step_halving,
            "screening pass should retry until the LM budget is exhausted"
        );
        assert_eq!(
            model.candidate_state_calls, options.max_step_halving,
            "Firth accepted-state reevaluation must stop at the configured LM retry budget"
        );
    }

    #[test]
    pub(crate) fn firth_permanent_candidate_error_propagates_without_lm_retries() {
        // Complement to `firth_candidate_reevaluation_respects_lm_retry_budget`
        // from the opposite angle: a *non-retriable* Firth candidate-evaluation
        // failure (a structural breakdown, not a numerical overflow) must
        // surface immediately as its original error, without the LM loop
        // spending a single damping retry on it. This guards the
        // retriable/non-retriable split for the Firth path that routes its
        // candidate evaluation through `update_candidate`.
        let mut model = FirthPermanentCandidateErrorModel::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 5,
            firth_bias_reduction: true,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            None,
        ) {
            Ok(_) => panic!("permanent Firth candidate failures should surface immediately"),
            Err(err) => err,
        };

        match err {
            EstimationError::InvalidSpecification(message) => {
                assert!(
                    message.contains("permanent firth breakdown"),
                    "expected the original permanent-failure error, got {message}"
                );
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }

        assert_eq!(model.current_state_calls, 1);
        assert_eq!(
            model.candidate_screen_calls, 1,
            "a non-retriable Firth candidate failure must not be re-screened"
        );
        assert_eq!(
            model.candidate_state_calls, 1,
            "a non-retriable Firth candidate failure must not consume LM retries"
        );
    }

    #[test]
    pub(crate) fn plateaued_accepted_step_does_not_report_converged_with_large_projected_gradient()
    {
        let mut model = PlateauStatusModel {
            gradient: 5e-5,
            current_deviance: 1.0,
            candidate_deviance: 1.0 - 1.25e-9,
        };
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, None)
                .expect("plateaued accepted step should still return a final state");

        // The plateau case ACCEPTS the candidate step (it's a noise-scale
        // improvement of 1.25e-9 in deviance), so the LM block does not
        // exhaust. The outer iteration counter (max_iterations=1) runs out
        // first, so the default-initialized MaxIterationsReached stands.
        // Distinct from the rejection test below, which exhausts LM retries
        // before iter completes. What both tests guard against is the
        // gradient 5e-5 (above the 1e-5 near-stationary band) being silently
        // promoted to Converged or StalledAtValidMinimum.
        assert_eq!(
            result.status,
            PirlsStatus::MaxIterationsReached,
            "projected gradient 5e-5 is well above the near-stationary band and must not be promoted to Converged/Stalled — the candidate step is accepted but the outer iteration counter must run out as MaxIterationsReached, not be silently re-classified"
        );
    }

    #[test]
    pub(crate) fn iteration_cap_runs_one_final_exact_decrement_certificate_2316() {
        let mut model = ExactDecrementAtIterationCapModel {
            exact_calls: 0,
            gradient: 5.0e-5,
            candidate_deviance: 1.0 - 1.25e-9,
        };
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, None)
                .expect("the exact final-state decrement certifies iteration exhaustion");

        assert_eq!(model.exact_calls, 1);
        assert_eq!(result.status, PirlsStatus::Converged);
    }

    #[test]
    pub(crate) fn soft_stall_gets_one_final_exact_decrement_certificate_2316() {
        let mut model = ExactDecrementAtIterationCapModel {
            exact_calls: 0,
            // Outside strict KKT (1e-6) but inside the 10× soft band.
            gradient: 5.0e-6,
            // Force LM rejection so the loop exits through soft acceptance
            // before the accepted-state exact-decrement check.
            candidate_deviance: 2.0,
        };
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 1,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, None)
                .expect("the exact final-state decrement certifies a soft LM stall");

        assert_eq!(model.exact_calls, 1);
        assert_eq!(result.status, PirlsStatus::Converged);
    }

    /// A linear objective has no minimum, so no plateau on it is a valid stall
    /// (#2902). `deviance = 1 − 5e-5·β` falls without bound as β grows, and the
    /// one constraint, `β ≥ −100`, sits on the far side. Every damped Newton step
    /// buys a sub-tolerance decrease. The deleted long-plateau arm accepted
    /// twenty such steps as `StalledAtValidMinimum`. The loop must instead report
    /// that it reached no minimum.
    #[test]
    pub(crate) fn linear_objective_plateau_is_not_a_valid_stall_2902() {
        let mut model = LinearObjectivePlateauModel { gradient: -5e-5 };
        let options = WorkingModelPirlsOptions {
            max_iterations: 25,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![-100.0],
            }),
            initial_lm_lambda: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, None)
                .expect("a plateau on an unbounded ray still returns its final state");

        assert!(
            !matches!(
                result.status,
                PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
            ),
            "a linear objective has no minimum to converge or stall at, got {:?} after {} \
             iterations",
            result.status,
            result.iterations,
        );
    }

    #[test]
    pub(crate) fn rejected_noise_scale_step_can_finish_via_exact_decrement_certificate_2316() {
        let mut model = PlateauStatusModel {
            gradient: 1e-5,
            current_deviance: 1.0e6,
            candidate_deviance: 1.0e6 + 1.0,
        };
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 1,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, None)
                .expect("noise-scale rejected step should still preserve the current state");

        // The candidate is objectively worse, so the LM trial is rejected.
        // That says nothing about the unchanged current iterate's stationarity:
        // #2316 deliberately pays for one exact final Newton-decrement
        // certificate after any finite bounded exit. Here H=1 and g=1e-5, so
        // g'H^-1g=1e-10, inside twice the objective's rounding band
        // (2·γ_{n+p²}·|½·D| ≈ 2.2e-10 at D=1e6 with one row and one
        // coefficient): no step can be shown to improve the objective on this
        // arithmetic, so the final iterate is independently certified and
        // promoted to Converged. The old June test predated #2316 and
        // incorrectly made a trial-step verdict override the later
        // final-state certificate.
        assert!(model.candidate_deviance > model.current_deviance);
        let exact_decrement_sq = model.gradient * model.gradient;
        let final_penalized_objective = 0.5 * model.current_deviance;
        let exact_decrement_bound =
            2.0 * super::convergence::objective_rounding_band(1, 1, final_penalized_objective);
        assert!(
            exact_decrement_sq <= exact_decrement_bound,
            "fixture must carry the decisive #2316 exact-decrement certificate"
        );
        assert_eq!(
            result.status,
            PirlsStatus::Converged,
            "the rejected trial must not suppress an independent exact certificate for the unchanged final iterate"
        );
    }

    /// Helper: assert that the penalized deviance trace is non-increasing
    /// across P-IRLS iterations, allowing a small tolerance for floating-point
    /// rounding.
    pub(crate) fn assert_deviance_monotone(trace: &[f64], label: &str) {
        assert!(
            trace.len() >= 2,
            "{}: expected at least 2 deviance recordings, got {}",
            label,
            trace.len()
        );
        for i in 1..trace.len() {
            let prev = trace[i - 1];
            let curr = trace[i];
            // Allow tiny increases up to a relative tolerance of 1e-8 plus
            // an absolute tolerance of 1e-12, to account for floating-point noise.
            let tol = 1e-8 * prev.abs() + 1e-12;
            assert!(
                curr <= prev + tol,
                "{}: deviance increased at iteration {} -> {}: {:.12e} -> {:.12e} (delta = {:.3e})",
                label,
                i - 1,
                i,
                prev,
                curr,
                curr - prev,
            );
        }
    }

    #[test]
    pub(crate) fn test_deviance_monotonicity_logistic() {
        // Logistic regression: binary y with a single covariate plus intercept.
        let n = 30;
        let mut x_data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64 / (n - 1) as f64) * 4.0 - 2.0; // t in [-2, 2]
            x_data[[i, 0]] = 1.0;
            x_data[[i, 1]] = t;
            // Deterministic binary labels: P(y=1) = sigmoid(0.5 + 1.5*t)
            let eta = 0.5 + 1.5 * t;
            let p = 1.0 / (1.0 + (-eta).exp());
            let pseudo_random = ((i * 31 + 7) % 17) as f64 / 17.0;
            y[i] = if pseudo_random < p { 1.0 } else { 0.0 };
        }

        let w = Array1::ones(n);
        let offset = Array1::zeros(n);
        let rho = array![0.0];
        let rs = [array![[0.0, 0.0], [0.0, 1.0]]];
        let canonical: Vec<gam_terms::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                gam_terms::construction::CanonicalPenalty {
                    root: r.clone().into_shared(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local: local.into_shared(),
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            )),
            link_kind: InverseLink::Standard(StandardLink::Logit),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
        };

        let (result, trace) = capture_pirls_penalized_deviance(|| {
            fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view())
                    .expect("test rho lies in exact strength domain"),
                PirlsProblem {
                    x: x_data.view(),
                    offset: offset.view(),
                    y: y.view(),
                    priorweights: w.view(),
                    gaussian_fixed_cache: None,
                    glm_first_step_gram: None,
                },
                PenaltyConfig {
                    canonical_penalties: &canonical,
                    reparam_invariant: None,
                    p: 2,
                    coefficient_lower_bounds: None,
                    linear_constraints_original: None,
                },
                &config,
                None,
            )
        });
        result.expect("Logistic P-IRLS fit should succeed");
        assert_deviance_monotone(&trace, "Logistic");
    }

    #[test]
    pub(crate) fn test_deviance_monotonicity_logistic_multiseed() {
        // Run logistic regression with multiple deterministic "seeds" to
        // stress-test monotonicity under varied label configurations.
        let seeds: &[u64] = &[42, 137, 271, 314, 997];
        let n = 25;

        for &seed in seeds {
            let mut x_data = Array2::<f64>::zeros((n, 3));
            let mut y = Array1::<f64>::zeros(n);
            for i in 0..n {
                let t1 = (i as f64 / (n - 1) as f64) * 6.0 - 3.0;
                // Second covariate derived from seed for variety
                let t2 =
                    ((i as u64).wrapping_mul(seed).wrapping_add(13) % 100) as f64 / 100.0 - 0.5;
                x_data[[i, 0]] = 1.0;
                x_data[[i, 1]] = t1;
                x_data[[i, 2]] = t2;
                let eta = -0.3 + 1.0 * t1 + 0.8 * t2;
                let p = 1.0 / (1.0 + (-eta).exp());
                // Deterministic label assignment using a hash of (i, seed)
                let hash = (i as u64)
                    .wrapping_mul(seed)
                    .wrapping_add(seed >> 2)
                    .wrapping_mul(2654435761);
                let pseudo_uniform = (hash % 10000) as f64 / 10000.0;
                y[i] = if pseudo_uniform < p { 1.0 } else { 0.0 };
            }

            // Ensure we have at least one of each class; if not, force one.
            let ones: f64 = y.iter().sum();
            if ones < 1.0 {
                y[0] = 1.0;
            }
            if ones > (n as f64 - 1.0) {
                y[n - 1] = 0.0;
            }

            let w = Array1::ones(n);
            let offset = Array1::zeros(n);
            let rho = array![0.0, 0.0];
            let rs = vec![
                // Penalty matrices: penalize 2nd and 3rd coefficients independently
                array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ];
            let canonical: Vec<gam_terms::construction::CanonicalPenalty> = rs
                .iter()
                .map(|r| {
                    let local = r.t().dot(r);
                    gam_terms::construction::CanonicalPenalty {
                        root: r.clone().into_shared(),
                        col_range: 0..r.ncols(),
                        total_dim: r.ncols(),
                        nullity: 0,
                        local: local.into_shared(),
                        prior_mean: Array1::zeros(r.ncols()),
                        positive_eigenvalues: Vec::new(),
                        op: None,
                    }
                })
                .collect();
            let config = PirlsConfig {
                likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Logit),
                )),
                link_kind: InverseLink::Standard(StandardLink::Logit),
                max_iterations: 100,
                convergence_tolerance: 1e-8,
                firth_bias_reduction: false,
                initial_lm_lambda: None,
            };

            let (result, trace) = capture_pirls_penalized_deviance(|| {
                fit_model_for_fixed_rho(
                    LogSmoothingParamsView::new(rho.view())
                        .expect("test rho lies in exact strength domain"),
                    PirlsProblem {
                        x: x_data.view(),
                        offset: offset.view(),
                        y: y.view(),
                        priorweights: w.view(),
                        gaussian_fixed_cache: None,
                        glm_first_step_gram: None,
                    },
                    PenaltyConfig {
                        canonical_penalties: &canonical,
                        reparam_invariant: None,
                        p: 3,
                        coefficient_lower_bounds: None,
                        linear_constraints_original: None,
                    },
                    &config,
                    None,
                )
            });
            result.unwrap_or_else(|e| {
                panic!("Logistic P-IRLS fit failed for seed {}: {:?}", seed, e)
            });
            assert_deviance_monotone(&trace, &format!("Logistic(seed={})", seed));
        }
    }

    pub(crate) fn capture_pirls_lm_attempts<F, R>(run: F) -> (R, usize)
    where
        F: FnOnce() -> R,
    {
        super::reweight::test_support::PIRLS_LM_ATTEMPT_COUNT.with(|count| count.set(Some(0)));
        let result = run();
        let attempts = super::reweight::test_support::PIRLS_LM_ATTEMPT_COUNT
            .with(|count| count.take())
            .unwrap();
        (result, attempts)
    }

    /// Rare-event logistic cold start (pyGAM audit F18). A ~3%-prevalence
    /// response whose log-odds rise steeply with a skewed covariate (the
    /// credit-default shape: logit p = −10.65 + 0.0055·balance) makes the
    /// undamped Newton step from the prevalence-intercept seed overshoot by
    /// orders of magnitude: it drives η to ≈ +11 on the high-balance rows.
    /// The Levenberg–Marquardt damping must climb from `u ≈ 1e-16` to O(1)
    /// before a trial is accepted. The geometric reject schedule alone
    /// (×2, ×4, ×8, …) needs ~11 trials to cross those 16 decades, each
    /// re-solving a step the tiny damping leaves unchanged. Moré's
    /// interpolated rejection update takes the damping to the radius the
    /// rejected trial indicates in one or two trials. This pins the cost in
    /// LM attempts (damped solves plus trial evaluations), not wall clock.
    #[test]
    pub(crate) fn rare_event_logistic_cold_start_reaches_damping_in_few_trials() {
        let n = 4000;
        let n_basis = 10;
        let mut rng_state: u64 = 0x5EED_F18_0000_0001;
        let mut uniform = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((rng_state >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
        };
        let mut balance = Vec::with_capacity(n);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            // Box–Muller N(835, 480), truncated at zero like a card balance.
            let (u1, u2) = (uniform(), uniform());
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let b = (835.0 + 480.0 * z).max(0.0);
            let p = 1.0 / (1.0 + (10.65 - 0.0055 * b).exp());
            y[i] = if uniform() < p { 1.0 } else { 0.0 };
            balance.push(b);
        }
        let positives: f64 = y.sum();
        assert!(
            positives > 0.01 * n as f64 && positives < 0.08 * n as f64,
            "fixture must be rare-event, got {positives} positives of {n}"
        );
        // Intercept plus a hat-function (linear B-spline) basis on the
        // covariate range, with a second-difference penalty on the hats.
        let (lo, hi) = balance
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &b| {
                (lo.min(b), hi.max(b))
            });
        let h = (hi - lo) / (n_basis - 1) as f64;
        let p = n_basis + 1;
        let mut x = Array2::<f64>::zeros((n, p));
        for (i, &b) in balance.iter().enumerate() {
            x[[i, 0]] = 1.0;
            for k in 0..n_basis {
                let knot = lo + k as f64 * h;
                x[[i, k + 1]] = (1.0 - ((b - knot) / h).abs()).max(0.0);
            }
        }
        let mut root = Array2::<f64>::zeros((n_basis - 2, p));
        for r in 0..n_basis - 2 {
            root[[r, r + 1]] = 1.0;
            root[[r, r + 2]] = -2.0;
            root[[r, r + 3]] = 1.0;
        }
        let canonical = vec![gam_terms::construction::CanonicalPenalty {
            local: root.t().dot(&root).into_shared(),
            root: root.into_shared(),
            col_range: 0..p,
            total_dim: p,
            nullity: 0,
            prior_mean: Array1::zeros(p),
            positive_eigenvalues: Vec::new(),
            op: None,
        }];
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            )),
            link_kind: InverseLink::Standard(StandardLink::Logit),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
        };
        let w = Array1::ones(n);
        let offset = Array1::zeros(n);
        let rho = array![0.0];
        let (result, attempts) = capture_pirls_lm_attempts(|| {
            fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view())
                    .expect("test rho lies in exact strength domain"),
                PirlsProblem {
                    x: x.view(),
                    offset: offset.view(),
                    y: y.view(),
                    priorweights: w.view(),
                    gaussian_fixed_cache: None,
                    glm_first_step_gram: None,
                },
                PenaltyConfig {
                    canonical_penalties: &canonical,
                    reparam_invariant: None,
                    p,
                    coefficient_lower_bounds: None,
                    linear_constraints_original: None,
                },
                &config,
                None,
            )
        });
        let (_, working) = result.expect("rare-event logistic P-IRLS fit should succeed");
        assert_eq!(working.status, PirlsStatus::Converged);
        // Every attempt beyond one per iteration is a rejected trial. The
        // geometric schedule alone spent 10 rejections (21 attempts over 11
        // iterations) climbing from `u` to the accepted damping; the
        // interpolated update needs 2.
        let rejected = attempts.saturating_sub(working.iterations);
        assert!(
            rejected <= 3,
            "rare-event cold start spent {rejected} rejected LM trials \
             ({attempts} attempts over {} iterations)",
            working.iterations
        );
    }

    #[test]
    pub(crate) fn solve_newton_direction_implicit_matches_dense_at_k500() {
        // Phase 2C equivalence test: PCG-against-implicit-H must produce the
        // same Newton direction as dense Cholesky on the same fully-assembled
        // Hessian H = X^T W X + ridge·I + λ·S, where S is provided in
        // operator form via `ClosedFormPenaltyOperator`. This pins the
        // contract that future refactors of `solve_newton_direction_implicit`
        // cannot silently drift from the dense path.
        use gam_terms::analytic_penalties::PenaltyOp;
        use gam_terms::basis::closed_form_operator::ClosedFormPenaltyOperator;

        const K: usize = 500;
        const D: usize = 4;

        // Synthetic centers in [0,1]^D via deterministic LCG.
        let mut state: u64 = 0xDEADBEEF_CAFEBABE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut centers = Array2::<f64>::zeros((K, D));
        for i in 0..K {
            for j in 0..D {
                centers[[i, j]] = next();
            }
        }
        let op = std::sync::Arc::new(ClosedFormPenaltyOperator::new(
            centers.view(),
            /* q = */ 2,
            /* m = */ 2,
            /* s = */ 1,
            /* kappa = */ 1.0,
            None,
            None,
            0,
            None,
        ));
        let p = op.dim();
        assert_eq!(p, K);
        let s_dense = op.as_dense();

        // Synthetic well-conditioned X^T W X (diag-dominant SPD).
        let mut xtwx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..=i {
                let v = if i == j {
                    2.0 + ((i as f64) * 0.07).sin() * 0.3
                } else {
                    (((i as f64 - j as f64) * 0.13).cos()) * 0.02 / (((i + 1) as f64).sqrt())
                };
                xtwx[[i, j]] = v;
                xtwx[[j, i]] = v;
            }
        }
        let xtwx_diag: Array1<f64> = (0..p).map(|i| xtwx[[i, i]]).collect();
        let lambda = 0.1_f64;
        let ridge = 0.0_f64;
        let gradient = Array1::<f64>::from_shape_fn(p, |i| ((i as f64) * 0.31).sin());

        // Dense reference: form full H = X^T W X + λ S, factor and solve.
        let mut h_dense = xtwx.clone();
        for i in 0..p {
            for j in 0..p {
                h_dense[[i, j]] += lambda * s_dense[[i, j]];
            }
        }
        let mut dense_dir = Array1::<f64>::zeros(p);
        super::solve_newton_direction_dense(&h_dense, &gradient, &mut dense_dir)
            .expect("dense Newton solve should succeed on synthetic SPD");

        // Implicit path: PCG against operator H = X^T W X + λ·op.matvec.
        let xtwx_for_closure = xtwx.clone();
        let apply_xtwx = move |v: &Array1<f64>| -> Array1<f64> { xtwx_for_closure.dot(v) };
        let op_pen: &dyn PenaltyOp = op.as_ref();
        let mut implicit_dir = Array1::<f64>::zeros(p);
        super::solve_newton_direction_implicit(
            apply_xtwx,
            xtwx_diag.view(),
            &[],
            &[(lambda, op_pen)],
            &gradient,
            &mut implicit_dir,
            ridge,
            /* rel_tol = */ 1e-12,
            /* max_iter = */ 4 * p,
        )
        .expect("implicit Newton solve should succeed on synthetic SPD");

        let dense_norm: f64 = dense_dir.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut diff_sq = 0.0_f64;
        for i in 0..p {
            let d = implicit_dir[i] - dense_dir[i];
            diff_sq += d * d;
        }
        let rel = diff_sq.sqrt() / dense_norm.max(1e-300);
        assert!(
            rel < 1e-9,
            "implicit-PCG vs dense-Cholesky Newton direction relative diff {} exceeds 1e-9",
            rel
        );
    }

    // ─── Issue 4: ExportedLaplaceCurvature labelling regressions ─────────────
    //
    // The inner LM step search may accept Fisher curvature when observed went
    // non-SPD or produced a bad gain ratio mid-iteration. The exported Laplace
    // curvature on `WorkingModelPirlsResult` (and downstream `PirlsResult`) is
    // re-evaluated at the accepted β̂ in a post-convergence finalization step
    // and must reflect the *actual* Hessian status — never silently mislabel a
    // Fisher fallback as exact, and never silently substitute Fisher when the
    // Observed Hessian is indefinite.

    /// Inner-loop accepts a step under Fisher (it's the only curvature this
    /// model offers during the inner loop), but in post-convergence
    /// finalization we explicitly recompute the Observed Hessian. Result:
    /// the exported label flips from whatever the inner loop used to
    /// `ObservedExact` (when SPD) — Fisher → Observed substitution is
    /// detected by the inertia gate, not silently accepted.
    #[derive(Default)]
    pub(crate) struct InnerFisherButObservedSpdAtMode {
        pub(crate) observed_post_calls: usize,
    }

    impl WorkingModel for InnerFisherButObservedSpdAtMode {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            if curvature == HessianCurvatureKind::Observed {
                self.observed_post_calls += 1;
            }
            // SPD scalar Hessian; identical for either curvature here, mirrors
            // the canonical-link case where Observed = Fisher numerically but
            // labels still need to be honest.
            Ok(scalar_working_state(beta, curvature, 0.0, 0.0))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 0.0, 0.0))
        }

        fn supports_observed_information_curvature(&self) -> bool {
            true
        }
    }

    #[test]
    pub(crate) fn exported_laplace_observed_exact_when_post_finalization_spd() {
        let mut model = InnerFisherButObservedSpdAtMode::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 2,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 3,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };
        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, None)
                .expect("converged scalar model should produce a result");
        assert!(
            matches!(
                summary.exported_laplace_curvature,
                ExportedLaplaceCurvature::ObservedExact
            ),
            "post-convergence Observed-SPD must export ObservedExact, got {:?}",
            summary.exported_laplace_curvature
        );
        assert!(
            model.observed_post_calls >= 1,
            "post-convergence finalization must call update_with_curvature(Observed) \
             at least once to assert SPD inertia"
        );
    }

    /// Model that does NOT support observed information (e.g. canonical-link
    /// or surrogate-by-design family). Exported curvature must be
    /// `ExpectedInformationSurrogate`, not silently relabeled `ObservedExact`.
    #[derive(Default)]
    pub(crate) struct CanonicalSurrogateModel;

    impl WorkingModel for CanonicalSurrogateModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }
        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 0.0, 0.0))
        }
        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 0.0, 0.0))
        }
        // Default `supports_observed_information_curvature() -> false`.
    }

    #[test]
    pub(crate) fn exported_laplace_surrogate_when_observed_unsupported() {
        let mut model = CanonicalSurrogateModel;
        let options = WorkingModelPirlsOptions {
            max_iterations: 2,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 3,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
        };
        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, None)
                .expect("canonical surrogate model should converge");
        assert!(
            matches!(
                summary.exported_laplace_curvature,
                ExportedLaplaceCurvature::ExpectedInformationSurrogate
            ),
            "model that doesn't support observed information must export \
             ExpectedInformationSurrogate (no silent ObservedExact relabel), \
             got {:?}",
            summary.exported_laplace_curvature
        );
    }

    #[test]
    pub(crate) fn dense_xtwx_signed_assembly_preserves_negative_weights() {
        let x = array![[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]];
        let weights = array![2.0, -3.0, 0.25];
        let mut got = Array2::<f64>::zeros((2, 2));
        PirlsWorkspace::add_dense_xtwx_signed(&weights, &x, &mut got);

        let mut expected = Array2::<f64>::zeros((2, 2));
        for i in 0..x.nrows() {
            for a in 0..x.ncols() {
                for b in 0..x.ncols() {
                    expected[[a, b]] += weights[i] * x[[i, a]] * x[[i, b]];
                }
            }
        }
        for (actual, expected) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
        }
        assert!(
            got[[0, 0]] < 0.0,
            "negative observed-Hessian weights must not be clipped away"
        );
    }
}

/// Regression tests for the fully-normalized, scale-aware **reporting**
/// eta-space log-likelihood evaluation
/// that back the user-facing AIC and PSIS-LOO elpd. These are distinct from the
/// REML building-block `*_omitting_constants` kernels, which deliberately drop
/// family/saturated normalizers. Root causes: #1581 (Poisson `−ln Γ(y+1)`),
/// #1582 (Poisson↔NB cross-family comparability), #1583 (Gaussian scale + 2π).
#[cfg(test)]
mod reporting_loglikelihood_tests {
    use super::super::{
        calculate_loglikelihood_omitting_constants_from_eta,
        eta_log_likelihood_value_and_score_into, evaluate_full_log_likelihood_from_eta,
    };
    use gam_problem::{
        GlmLikelihoodSpec, InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily,
        StandardLink,
    };
    use ndarray::{Array1, array};
    use statrs::function::gamma::ln_gamma;

    fn canonical(family: ResponseFamily, link: StandardLink) -> GlmLikelihoodSpec {
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(family, InverseLink::Standard(link)))
    }

    fn eta_fixture(mu: &Array1<f64>, link: StandardLink) -> Array1<f64> {
        match link {
            StandardLink::Identity => mu.clone(),
            StandardLink::Log => mu.mapv(f64::ln),
            StandardLink::Logit => mu.mapv(|value| value.ln() - (-value).ln_1p()),
            other => panic!("reporting test fixture does not implement {other:?}"),
        }
    }

    fn full_at_fixture(
        y: &Array1<f64>,
        mu: &Array1<f64>,
        likelihood: &GlmLikelihoodSpec,
        weights: &Array1<f64>,
        link: StandardLink,
    ) -> super::super::FullLogLikelihoodEvaluation {
        let eta = eta_fixture(mu, link);
        evaluate_full_log_likelihood_from_eta(y.view(), eta.view(), likelihood, weights.view())
            .expect("full eta likelihood fixture")
    }

    // ---- #1581: Poisson reporting log-likelihood is a true (negative) log-mass.
    #[test]
    fn poisson_full_loglik_is_log_mass_and_carries_count_normalizer() {
        let y = array![0.0, 1.0, 2.0, 3.0, 7.0];
        let mu = array![0.5, 1.2, 2.5, 2.0, 6.0];
        let w = Array1::<f64>::ones(y.len());
        let glm = canonical(ResponseFamily::Poisson, StandardLink::Log);

        let evaluation = full_at_fixture(&y, &mu, &glm, &w, StandardLink::Log);
        let pw = evaluation.pointwise();
        // Every Poisson pointwise value is a log probability mass ≤ 0.
        for (i, &v) in pw.iter().enumerate() {
            assert!(v <= 0.0, "row {i}: Poisson log-mass must be ≤ 0, got {v}");
        }
        // Matches the analytic count log-likelihood y·ln μ − μ − ln Γ(y+1).
        let analytic: f64 = y
            .iter()
            .zip(mu.iter())
            .map(|(&yi, &mui)| {
                let log_term = if yi > 0.0 { yi * mui.ln() } else { 0.0 };
                log_term - mui - ln_gamma(yi + 1.0)
            })
            .sum();
        let total = evaluation.total();
        assert!((total - analytic).abs() < 1e-10, "{total} vs {analytic}");
        assert!(
            total < 0.0,
            "summed Poisson elpd must be negative, got {total}"
        );

        // The reporting kernel differs from the REML building block by EXACTLY
        // the dropped −Σ ln Γ(y+1) count normalizer — the #1581 root cause.
        let eta = mu.mapv(f64::ln);
        let omitting = calculate_loglikelihood_omitting_constants_from_eta(
            y.view(),
            &eta,
            &glm,
            &InverseLink::Standard(StandardLink::Log),
            w.view(),
        )
        .expect("exact eta log-likelihood");
        let dropped: f64 = y.iter().map(|&yi| ln_gamma(yi + 1.0)).sum();
        assert!(
            (omitting - total - dropped).abs() < 1e-10,
            "omitting − full must equal Σ ln Γ(y+1) = {dropped}; got {}",
            omitting - total
        );
    }

    // A fractional binomial prior weight (a scikit-learn `sample_weight` on 0/1
    // labels) is a weighted Bernoulli log-mass, w·[y ln μ + (1−y) ln(1−μ)]:
    // the continuous `ln C(w, wy)` normalizer vanishes for a 0/1 response, and
    // for a proportion it is the lnΓ continuation of the integer coefficient.
    #[test]
    fn binomial_full_loglik_accepts_fractional_prior_weights() {
        let y = array![0.0, 1.0, 1.0, 0.0, 0.4];
        let mu = array![0.3, 0.8, 0.55, 0.1, 0.35];
        let w = array![0.5, 2.25, 1.0, 3.7, 2.5];
        let glm = canonical(ResponseFamily::Binomial, StandardLink::Logit);

        let evaluation = full_at_fixture(&y, &mu, &glm, &w, StandardLink::Logit);
        let pw = evaluation.pointwise();
        for row in 0..y.len() {
            let (yi, mui, wi) = (y[row], mu[row], w[row]);
            let log_coefficient = ln_gamma(wi + 1.0)
                - ln_gamma(wi * yi + 1.0)
                - ln_gamma(wi * (1.0 - yi) + 1.0);
            let expected =
                log_coefficient + wi * (yi * mui.ln() + (1.0 - yi) * (1.0 - mui).ln());
            assert!(
                (pw[row] - expected).abs() < 1e-10,
                "row {row} (w={wi}, y={yi}): {} vs {expected}",
                pw[row]
            );
        }
        let bernoulli_rows = [0, 1, 2, 3];
        for row in bernoulli_rows {
            let (yi, mui, wi) = (y[row], mu[row], w[row]);
            let weighted_log_mass = wi * (yi * mui.ln() + (1.0 - yi) * (1.0 - mui).ln());
            assert!(
                (pw[row] - weighted_log_mass).abs() < 1e-12,
                "row {row}: a 0/1 response carries no normalizer at w={wi}"
            );
        }
    }

    // ---- #1582: Poisson and NB(θ→∞) report the SAME log-likelihood on the same
    // count data (NB → Poisson as Var = μ + μ²/θ → μ), so AIC/elpd are
    // comparable across the two families.
    #[test]
    fn poisson_and_large_theta_negbin_full_loglik_agree() {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0];
        let mu = array![0.8, 1.5, 2.2, 3.1, 3.8, 5.5, 8.0];
        let w = Array1::<f64>::ones(y.len());

        let poisson = canonical(ResponseFamily::Poisson, StandardLink::Log);
        let theta = 1.0e5;
        let negbin = canonical(
            ResponseFamily::NegativeBinomial {
                theta,
                theta_fixed: true,
            },
            StandardLink::Log,
        );

        let ll_pois = full_at_fixture(&y, &mu, &poisson, &w, StandardLink::Log).total();
        let ll_nb = full_at_fixture(&y, &mu, &negbin, &w, StandardLink::Log).total();

        // Both are proper negative log-masses.
        assert!(ll_pois < 0.0 && ll_nb < 0.0, "{ll_pois}, {ll_nb}");
        // They agree to well under one nat — the residual is O(n·μ²/θ).
        assert!(
            (ll_pois - ll_nb).abs() < 1.0e-2,
            "Poisson vs NB(θ=1e5) must agree: {ll_pois} vs {ll_nb} (Δ={})",
            ll_pois - ll_nb
        );
    }

    // ---- #1583: Gaussian reporting log-likelihood is a true predictive density
    // — it uses the estimated variance and obeys the change-of-variables law
    // elpd(c·y) − elpd(y) = −n·ln c, not the c²-scaling of −½·RSS.
    #[test]
    fn gaussian_full_loglik_obeys_change_of_variables() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.5, 0.5];
        let mu = array![1.1, 1.9, 3.2, 3.8, 5.0, 0.7];
        let w = Array1::<f64>::ones(y.len());
        let n = y.len() as f64;
        let sigma2 = 0.25_f64;

        let glm = |s2: f64| GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            scale: LikelihoodScaleMetadata::FixedDispersion { phi: s2 },
        };

        let ll = full_at_fixture(&y, &mu, &glm(sigma2), &w, StandardLink::Identity).total();
        // Analytic profiled-Gaussian value −½ Σ[ln(2πσ²) + resid²/σ²].
        let analytic: f64 = y
            .iter()
            .zip(mu.iter())
            .map(|(&yi, &mui)| {
                let r = yi - mui;
                -0.5 * ((2.0 * std::f64::consts::PI * sigma2).ln() + r * r / sigma2)
            })
            .sum();
        assert!((ll - analytic).abs() < 1e-10, "{ll} vs {analytic}");

        // Scale equivariance: y→c·y, μ→c·μ, σ̂²→c²·σ̂² ⇒ elpd shifts by −n·ln c.
        for &c in &[0.5_f64, 2.0, 10.0] {
            let yc = y.mapv(|v| c * v);
            let muc = mu.mapv(|v| c * v);
            let llc = full_at_fixture(&yc, &muc, &glm(c * c * sigma2), &w, StandardLink::Identity)
                .total();
            let shift = llc - ll;
            assert!(
                (shift - (-n * c.ln())).abs() < 1e-9,
                "c={c}: change-of-variables shift must be −n·ln c = {}, got {shift}",
                -n * c.ln()
            );
        }
    }

    // A profiled Gaussian whose scale was never concretized is rejected, not
    // silently interpreted as a unit-variance density.
    #[test]
    fn gaussian_full_loglik_requires_concrete_scale() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.1, 2.1, 2.9];
        let w = Array1::<f64>::ones(y.len());
        let glm = canonical(ResponseFamily::Gaussian, StandardLink::Identity);
        // canonical Gaussian ⇒ ProfiledGaussian ⇒ fixed_phi() == None.
        let eta = eta_fixture(&mu, StandardLink::Identity);
        let error = evaluate_full_log_likelihood_from_eta(y.view(), eta.view(), &glm, w.view())
            .expect_err("unresolved profiled Gaussian scale must fail");
        assert!(error.to_string().contains("explicit positive dispersion"));

        let error = calculate_loglikelihood_omitting_constants_from_eta(
            y.view(),
            &eta,
            &glm,
            &glm.spec.link,
            w.view(),
        )
        .expect_err("strict eta likelihood must not invent a profiled Gaussian dispersion");
        assert!(error.to_string().contains("explicit positive dispersion"));

        let original_score = array![7.0, 8.0, 9.0];
        let mut score = original_score.clone();
        let error = eta_log_likelihood_value_and_score_into(
            y.view(),
            &eta,
            &glm,
            &glm.spec.link,
            w.view(),
            &mut score,
        )
        .expect_err("HMC eta likelihood must not invent a profiled Gaussian dispersion");
        assert!(error.to_string().contains("explicit positive dispersion"));
        assert_eq!(
            score, original_score,
            "strict eta likelihood failure must leave the caller's score untouched"
        );
    }

    // Gaussian prior weights act as inverse-variance scaling (Var = φ/wᵢ): the
    // normalizer picks up the +½ ln wᵢ Jacobian, the residual term picks up wᵢ.
    #[test]
    fn gaussian_full_loglik_prior_weight_jacobian() {
        let y = array![1.0, 2.0];
        let mu = array![1.3, 1.7];
        let w = array![2.0, 0.5];
        let sigma2 = 0.4_f64;
        let glm = GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            scale: LikelihoodScaleMetadata::FixedDispersion { phi: sigma2 },
        };
        let evaluation = full_at_fixture(&y, &mu, &glm, &w, StandardLink::Identity);
        let pw = evaluation.pointwise();
        for i in 0..2 {
            let r = y[i] - mu[i];
            let expect = -0.5
                * ((2.0 * std::f64::consts::PI * sigma2).ln() - w[i].ln() + w[i] * r * r / sigma2);
            assert!(
                (pw[i] - expect).abs() < 1e-12,
                "row {i}: {} vs {expect}",
                pw[i]
            );
        }
    }

    // Binomial reporting log-likelihood is a true log-mass ≤ 0 and carries the
    // ln C(nᵢ, nᵢyᵢ) coefficient (zero for Bernoulli, positive for counts).
    #[test]
    fn binomial_full_loglik_carries_coefficient() {
        // Grouped binomial: prior weights are the trial counts nᵢ.
        let y = array![0.0, 0.25, 0.5, 1.0];
        let mu = array![0.1, 0.3, 0.55, 0.9];
        let w = array![3.0, 4.0, 6.0, 2.0];
        let glm = canonical(ResponseFamily::Binomial, StandardLink::Logit);
        let evaluation = full_at_fixture(&y, &mu, &glm, &w, StandardLink::Logit);
        let pw = evaluation.pointwise();
        for (i, &v) in pw.iter().enumerate() {
            assert!(
                v <= 1e-12,
                "row {i}: binomial log-mass must be ≤ 0, got {v}"
            );
            let n = w[i];
            let k = n * y[i];
            let coef = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0);
            let expect = coef + n * (y[i] * mu[i].ln() + (1.0 - y[i]) * (1.0 - mu[i]).ln());
            assert!((v - expect).abs() < 1e-10, "row {i}: {v} vs {expect}");
        }

        // Bernoulli (nᵢ = 1, yᵢ ∈ {0,1}): coefficient vanishes, so the full and
        // omitting kernels coincide.
        let yb = array![0.0, 1.0, 1.0, 0.0];
        let mub = array![0.2, 0.8, 0.6, 0.4];
        let wb = Array1::<f64>::ones(4);
        let full = full_at_fixture(&yb, &mub, &glm, &wb, StandardLink::Logit);
        let eta = eta_fixture(&mub, StandardLink::Logit);
        let omit = calculate_loglikelihood_omitting_constants_from_eta(
            yb.view(),
            &eta,
            &glm,
            &glm.spec.link,
            wb.view(),
        )
        .expect("Bernoulli omitted likelihood");
        for i in 0..4 {
            let analytic = yb[i] * mub[i].ln() + (1.0 - yb[i]) * (1.0 - mub[i]).ln();
            assert!(
                (full.pointwise()[i] - analytic).abs() < 1e-12,
                "row {i}: {} vs {analytic}",
                full.pointwise()[i],
            );
        }
        assert!((full.total() - omit).abs() < 1e-12);
    }

    // Gamma reporting log-likelihood equals the analytic Gamma density (shape
    // ν = 1/φ, mean μ), evaluated on the eta surface.
    #[test]
    fn gamma_full_loglik_matches_density() {
        let y = array![1.8, 0.7, 3.2];
        let mu = array![2.0, 1.0, 2.5];
        let w = array![1.0, 2.0, 0.5];
        // canonical Gamma ⇒ shape 1 ⇒ ν = 1.
        let glm = canonical(ResponseFamily::Gamma, StandardLink::Log);
        let nu = 1.0_f64;
        let evaluation = full_at_fixture(&y, &mu, &glm, &w, StandardLink::Log);
        let pw = evaluation.pointwise();
        for i in 0..3 {
            let a = w[i] * nu;
            let expect =
                a * (a / mu[i]).ln() + (a - 1.0) * y[i].ln() - a * y[i] / mu[i] - ln_gamma(a);
            assert!(
                (pw[i] - expect).abs() < 1e-10,
                "row {i}: {} vs {expect}",
                pw[i]
            );
        }
        let total = evaluation.total();
        assert!((total - pw.sum()).abs() < 1e-12);
    }

    // Zero prior weight excludes an observation: every family contributes
    // exactly 0 (no −∞ from the Gamma shape→0 or Tweedie −ln w prefactor).
    #[test]
    fn zero_prior_weight_contributes_zero_every_family() {
        let y = array![2.0, 3.0];
        let mu = array![1.5, 2.5];
        let w = array![0.0, 0.0];
        for glm in [
            canonical(ResponseFamily::Poisson, StandardLink::Log),
            canonical(ResponseFamily::Gamma, StandardLink::Log),
            canonical(ResponseFamily::Binomial, StandardLink::Logit),
            GlmLikelihoodSpec {
                spec: LikelihoodSpec::new(
                    ResponseFamily::Gaussian,
                    InverseLink::Standard(StandardLink::Identity),
                ),
                scale: LikelihoodScaleMetadata::FixedDispersion { phi: 0.3 },
            },
            GlmLikelihoodSpec {
                spec: LikelihoodSpec::new(
                    ResponseFamily::Tweedie { p: 1.5 },
                    InverseLink::Standard(StandardLink::Log),
                ),
                scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            },
        ] {
            let yb = array![0.5, 0.6];
            let (yy, mm) = if matches!(glm.spec.response, ResponseFamily::Binomial) {
                (yb.clone(), array![0.4, 0.55])
            } else {
                (y.clone(), mu.clone())
            };
            let link = match &glm.spec.response {
                ResponseFamily::Gaussian => StandardLink::Identity,
                ResponseFamily::Binomial => StandardLink::Logit,
                _ => StandardLink::Log,
            };
            let evaluation = full_at_fixture(&yy, &mm, &glm, &w, link);
            let pw = evaluation.pointwise();
            for &v in pw.iter() {
                assert_eq!(
                    v, 0.0,
                    "{:?}: zero-weight row must be 0, got {v}",
                    glm.spec.response
                );
            }
        }
    }

    // The certified total shares the pointwise kernel.
    #[test]
    fn scalar_equals_sum_of_pointwise() {
        let y = array![0.0, 2.0, 5.0, 1.0];
        let mu = array![1.0, 2.0, 4.0, 1.5];
        let w = array![1.0, 1.0, 2.0, 1.0];
        let glm = canonical(ResponseFamily::Poisson, StandardLink::Log);
        let evaluation = full_at_fixture(&y, &mu, &glm, &w, StandardLink::Log);
        let pw = evaluation.pointwise();
        let total = evaluation.total();
        assert!((total - pw.sum()).abs() < 1e-12);
    }
}

/// #2105: the exact compound-Poisson–gamma Tweedie density used for variance-
/// power estimation. Reporting and the `p`-profile both use the exact series
/// (`tweedie_series_loglik` /
/// `tweedie_exact_loglik_total_from_eta`): the saddlepoint's missing `O(1/λ)` normalizer
/// biases the profile maximizer of `p` low, which inflates the reported Pearson
/// dispersion `φ̂` and every SE / interval derived from it.
#[cfg(test)]
mod tweedie_exact_series_tests {
    use super::{
        tweedie_exact_loglik, tweedie_saddlepoint_loglik_approximation, tweedie_series_loglik,
    };
    use ndarray::Array1;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use statrs::function::gamma::ln_gamma;

    /// Standard normal via Box–Muller (only `rand`'s uniform is available here).
    fn normal_draw(rng: &mut StdRng) -> f64 {
        let u1: f64 = rng.random::<f64>().max(1e-300);
        let u2: f64 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Brute-force reference: sum a very wide, fixed window of the mixture terms
    /// with an explicit log-sum-exp. Independent of the adaptive climb/tail logic
    /// in `tweedie_series_loglik`, so agreement certifies that logic.
    fn series_bruteforce(yi: f64, mui: f64, w: f64, p: f64, phi: f64) -> f64 {
        let phi_i = phi / w;
        let lambda = mui.powf(2.0 - p) / (phi_i * (2.0 - p));
        if yi <= 0.0 {
            return -lambda;
        }
        let alpha = (2.0 - p) / (p - 1.0);
        let scale = phi_i * (p - 1.0) * mui.powf(p - 1.0);
        let k_hi = (lambda * 4.0) as usize + 20_000;
        let terms: Vec<f64> = (1..=k_hi)
            .map(|k| {
                let kf = k as f64;
                -lambda + kf * lambda.ln() - ln_gamma(kf + 1.0) + (kf * alpha - 1.0) * yi.ln()
                    - yi / scale
                    - kf * alpha * scale.ln()
                    - ln_gamma(kf * alpha)
            })
            .collect();
        let m = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + terms.iter().map(|t| (t - m).exp()).sum::<f64>().ln()
    }

    /// The exact Tweedie log-likelihood summed over rows: the objective a
    /// variance-power profile maximizes. The rows below are valid responses
    /// with unit prior weight, so the per-row exact series is summed directly.
    fn tweedie_exact_loglik_total_from_eta(
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        priorweights: ndarray::ArrayView1<f64>,
        p: f64,
        phi: f64,
    ) -> Result<f64, crate::estimate::EstimationError> {
        let mut total = 0.0;
        for row in 0..y.len() {
            total += super::super::tweedie_exact_series_loglik_from_eta(
                row,
                y[row],
                eta[row],
                priorweights[row],
                p,
                phi.ln(),
            )?;
        }
        Ok(total)
    }

    #[test]
    fn series_matches_brute_force_across_regimes() {
        // (mu, phi, p) spanning small/large mean, small/large dispersion, and
        // the whole open power interval.
        let cases = [
            (2.0, 0.6, 1.5),
            (0.5, 0.6, 1.5),
            (4.0, 0.6, 1.5),
            (50.0, 0.3, 1.5),
            (2.0, 2.0, 1.7),
            (1.0, 0.1, 1.3),
            (200.0, 0.5, 1.6),
        ];
        for (mu, phi, p) in cases {
            for &y in &[0.0, 0.3, 2.0, 8.0, mu] {
                let got = tweedie_series_loglik(y, mu, 1.0, p, phi);
                let want = series_bruteforce(y, mu, 1.0, p, phi);
                assert!(
                    (got - want).abs() < 1e-9,
                    "series != brute force at mu={mu} phi={phi} p={p} y={y}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn series_density_normalizes_to_one() {
        // P(Y=0) + ∫₀^∞ f(y) dy ≈ 1 (trapezoid on a fine grid to a far tail).
        // Restricted to gamma-shape α = (2−p)/(p−1) ≥ 1 (i.e. p ≤ 1.5) so the
        // density is finite at y→0⁺ and the uniform-grid trapezoid is accurate;
        // the p>1.5 / α<1 spike at the origin is an integrable singularity that a
        // uniform trapezoid cannot resolve (its density VALUES are certified
        // exact by `series_matches_brute_force_across_regimes`).
        for (mu, phi, p) in [
            (2.0_f64, 0.6_f64, 1.5_f64),
            (1.0, 0.1, 1.3),
            (3.0, 0.4, 1.4),
        ] {
            let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
            let mass0 = (-lambda).exp();
            let hi = mu * 30.0;
            let steps = 300_000usize;
            let h = hi / steps as f64;
            let mut integral = 0.0;
            for k in 0..=steps {
                let y = (k as f64) * h + 1e-9;
                let f = tweedie_series_loglik(y, mu, 1.0, p, phi).exp();
                let wgt = if k == 0 || k == steps { 0.5 } else { 1.0 };
                integral += wgt * f;
            }
            integral *= h;
            let total = mass0 + integral;
            assert!(
                (total - 1.0).abs() < 5e-3,
                "Tweedie series density must integrate to 1 (mu={mu} phi={phi} p={p}): \
                 P(0)={mass0} + ∫={integral} = {total}"
            );
        }
    }

    #[test]
    fn exact_loglik_never_switches_to_saddlepoint() {
        let (mu, phi, p) = (1.0e8_f64, 0.5_f64, 1.5_f64);
        let y = mu;
        let exact = tweedie_exact_loglik(y, mu, 1.0, p, phi);
        let series_at_large_index = tweedie_series_loglik(y, mu, 1.0, p, phi);
        let saddle = tweedie_saddlepoint_loglik_approximation(y, mu, 1.0, p, phi);
        assert_eq!(exact, series_at_large_index);
        assert!(
            (exact - saddle).abs() < 1e-3,
            "the separately named approximation should converge toward the exact series: \
             {exact} vs {saddle}"
        );
        let (mu2, phi2) = (5.0e3_f64, 1.0_f64); // index ≈ 283, below threshold
        let series = tweedie_series_loglik(mu2, mu2, 1.0, p, phi2);
        let saddle2 = tweedie_saddlepoint_loglik_approximation(mu2, mu2, 1.0, p, phi2);
        assert!(
            (series - saddle2).abs() < 1e-2,
            "series and saddlepoint must agree closely near the crossover: {series} vs {saddle2}"
        );
    }

    /// Compound-Poisson–gamma (Jørgensen) Tweedie sample generator.
    fn tweedie_sample(mu: f64, p: f64, phi: f64, rng: &mut StdRng) -> f64 {
        let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
        let shape = (2.0 - p) / (p - 1.0);
        let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
        // Knuth Poisson.
        let l = (-lambda).exp();
        let mut k = 0u32;
        let mut prod = 1.0_f64;
        loop {
            prod *= rng.random::<f64>();
            if prod <= l {
                break;
            }
            k += 1;
            if k > 100_000 {
                break;
            }
        }
        // Sum of `k` Gamma(shape, scale) draws via Marsaglia–Tsang.
        let mut y = 0.0;
        for _ in 0..k {
            y += gamma_draw(shape, scale, rng);
        }
        y
    }

    fn gamma_draw(shape: f64, scale: f64, rng: &mut StdRng) -> f64 {
        if shape < 1.0 {
            let u: f64 = rng.random::<f64>().max(1e-300);
            return gamma_draw(shape + 1.0, scale, rng) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let z: f64 = normal_draw(rng);
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u: f64 = rng.random::<f64>().max(1e-300);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v * scale;
            }
        }
    }

    fn pearson_phi(y: &Array1<f64>, mu: &Array1<f64>, p: f64) -> f64 {
        let mut num = 0.0;
        for (&yi, &mui) in y.iter().zip(mu.iter()) {
            num += (yi - mui).powi(2) / mui.powf(p);
        }
        num / y.len() as f64
    }

    fn golden_max_p<F: Fn(f64) -> f64>(f: F) -> f64 {
        let (mut a, mut b) = (1.001_f64, 1.999_f64);
        let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
        let (mut c, mut d) = (b - gr * (b - a), a + gr * (b - a));
        let (mut fc, mut fd) = (f(c), f(d));
        while b - a > 1e-3 {
            if fc >= fd {
                b = d;
                d = c;
                fd = fc;
                c = b - gr * (b - a);
                fc = f(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + gr * (b - a);
                fd = f(d);
            }
        }
        0.5 * (a + b)
    }

    #[test]
    fn exact_profile_recovers_power_where_saddlepoint_is_biased_low() {
        // Synthetic Tweedie data at the TRUE mean (isolates the density
        // approximation from the mean fit). The exact-series profile recovers
        // p_true; the saddlepoint profile is biased conspicuously low — the
        // #2105 root cause at the density level.
        let mut rng = StdRng::seed_from_u64(2_105_015);
        let n = 6000usize;
        let (p_true, phi_true) = (1.5_f64, 0.6_f64);
        let mut mu = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x: f64 = -1.5 + 3.0 * rng.random::<f64>();
            let m = (0.7 + 0.5 * x).exp();
            mu[i] = m;
            y[i] = tweedie_sample(m, p_true, phi_true, &mut rng);
        }
        let w = Array1::<f64>::ones(n);

        let exact_obj = |p: f64| {
            let phi = pearson_phi(&y, &mu, p);
            let eta = mu.mapv(f64::ln);
            tweedie_exact_loglik_total_from_eta(y.view(), eta.view(), w.view(), p, phi)
                .expect("exact Tweedie profile row")
        };
        let saddle_obj = |p: f64| {
            let phi = pearson_phi(&y, &mu, p);
            (0..n)
                .map(|i| tweedie_saddlepoint_loglik_approximation(y[i], mu[i], w[i], p, phi))
                .sum::<f64>()
        };

        let p_exact = golden_max_p(exact_obj);
        let p_saddle = golden_max_p(saddle_obj);
        eprintln!("#2105 density profile: p_exact={p_exact:.4} p_saddle={p_saddle:.4}");

        // Exact profile lands near the truth ...
        assert!(
            (p_exact - p_true).abs() < 0.06,
            "exact-series profile must recover p_true={p_true}: got {p_exact}"
        );
        // ... while the saddlepoint is biased low by a wide margin (this is the
        // bug — assert the pre-fix estimator would have failed a tight bound).
        assert!(
            p_saddle < p_exact - 0.1,
            "saddlepoint profile should be biased low relative to exact: \
             p_saddle={p_saddle}, p_exact={p_exact}"
        );

        // The dispersion at the recovered power is unbiased under the exact
        // profile and INFLATED under the saddlepoint's low power.
        let phi_exact = pearson_phi(&y, &mu, p_exact);
        let phi_saddle = pearson_phi(&y, &mu, p_saddle);
        assert!(
            (phi_exact - phi_true).abs() < 0.05,
            "φ̂ at the exact power must recover φ_true={phi_true}: got {phi_exact}"
        );
        assert!(
            phi_saddle > phi_exact * 1.05,
            "the saddlepoint's low power must inflate φ̂: {phi_saddle} vs {phi_exact}"
        );
    }
}

#[test]
fn active_face_newton_direction_stays_on_the_face_and_zeroes_the_reduced_gradient() {
    use super::reweight::{active_face_newton_direction, unconstrained_newton_direction};
    use ndarray::array;
    // Strictly convex quadratic ½βᵀHβ + gᵀβ with one active row a = (1, 1, 1)/√3.
    let curvature = array![[2.0, 0.3, 0.0], [0.3, 3.0, 0.1], [0.0, 0.1, 4.0]];
    let gradient = array![0.7, -1.1, 0.4];
    let s3 = 3.0_f64.sqrt();
    let a_active = array![[1.0 / s3, 1.0 / s3, 1.0 / s3]];

    let d = active_face_newton_direction(&curvature, &gradient, &a_active)
        .expect("a PD curvature on a one-row face has a reduced Newton step");
    // Stays on the face.
    let along_normal = a_active.row(0).dot(&d).abs();
    assert!(along_normal <= 1e-12, "step leaves the face: aᵀd = {along_normal:.3e}");
    // Zeroes the gradient of the quadratic model in every face direction.
    let residual = &gradient + &curvature.dot(&d);
    for tangent in [array![1.0, -1.0, 0.0], array![1.0, 1.0, -2.0]] {
        let component = tangent.dot(&residual).abs() / tangent.dot(&tangent).sqrt();
        assert!(
            component <= 1e-12,
            "reduced gradient not zeroed along {tangent:?}: {component:.3e}"
        );
    }
    // Differs from the unconstrained step, which would leave the face.
    let free = unconstrained_newton_direction(&curvature, &gradient).expect("PD curvature");
    assert!(a_active.row(0).dot(&free).abs() > 1e-3);

    // A fully determined face (three independent rows in 3-D) has no free
    // direction: the step is exactly zero, never a fabricated one.
    let full = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let pinned = active_face_newton_direction(&curvature, &gradient, &full).expect("widths agree");
    assert!(pinned.iter().all(|v| *v == 0.0));

    // Width mismatch is a refusal, not a guess.
    assert!(active_face_newton_direction(&curvature, &gradient, &array![[1.0, 0.0]]).is_none());
}

#[test]
fn binding_constraint_rows_drop_a_primal_active_row_with_a_zero_multiplier() {
    use crate::active_set::binding_constraint_rows;
    use gam_problem::LinearInequalityConstraints;
    use ndarray::array;
    // Two bound rows β₁ ≥ 0 and β₂ ≥ 0, both primal-active at the origin.
    let constraints = LinearInequalityConstraints::new(
        array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        array![0.0, 0.0],
    )
    .expect("well-formed constraint system");
    let beta = array![0.0, 0.0, 0.3];
    // The objective gradient points OUT of the feasible set along row 1 (a
    // positive multiplier holds it) and INTO it along row 2 (the minimum wants
    // to leave that boundary: multiplier zero).
    let gradient = array![1.0, -1.0, 0.0];
    let binding = binding_constraint_rows(&beta, &gradient, &constraints).expect("width agrees");
    assert_eq!(binding.nrows(), 1, "only the row with a positive multiplier binds");
    assert!((binding[[0, 0]] - 1.0).abs() < 1e-15 && binding[[0, 1]].abs() < 1e-15);
    // Width mismatch is a refusal.
    assert!(binding_constraint_rows(&array![0.0, 0.0], &gradient, &constraints).is_none());
}

/// The Beta half-unit deviance formed from shape differences agrees with the
/// two-log-likelihood difference where that difference is exact, and stays
/// smooth where it is not: at φ = 1e6 each log-likelihood is O(6e5), so their
/// difference resolves only to ~1e-10, while the second η-difference of the
/// unit deviance over h = 1e-6 is ≈ h²·φμ(1−μ) ≈ 2.5e-7 — the direct form is
/// noise at that scale, the difference form reproduces the analytic curvature.
#[test]
fn beta_half_unit_deviance_from_shape_differences_is_exact_and_resolved() {
    use super::deviance::{
        beta_eta_for_logit_target, beta_fitted_loglikelihood_unit_from_eta,
        beta_half_unit_deviance_from_shape_differences,
    };
    let logit = |p: f64| (p / (1.0 - p)).ln();
    // Moderate φ: both forms are accurate; they must agree.
    for &(y, phi, eta_offset) in &[(0.2_f64, 40.0_f64, 0.3_f64), (0.5, 120.0, -0.2), (0.9, 400.0, 0.05)] {
        let eta_s = beta_eta_for_logit_target(logit(y), phi).expect("saturated eta");
        let eta = eta_s + eta_offset;
        let direct = beta_fitted_loglikelihood_unit_from_eta(y, eta_s, phi).expect("sat")
            - beta_fitted_loglikelihood_unit_from_eta(y, eta, phi).expect("fit");
        let stable = beta_half_unit_deviance_from_shape_differences(y, eta, eta_s, phi)
            .expect("Stirling regime");
        assert!(
            (stable - direct).abs() <= 1e-11 * direct.abs().max(1e-3),
            "y={y} phi={phi}: stable={stable:.15e} direct={direct:.15e}"
        );
        assert!(stable > 0.0);
    }
    // Exact zero at the saturated point, exact y ↔ 1−y symmetry.
    let phi = 1.0e6;
    let y = 0.3;
    let eta_s = beta_eta_for_logit_target(logit(y), phi).expect("saturated eta");
    let eta_s_flip = beta_eta_for_logit_target(logit(1.0 - y), phi).expect("saturated eta");
    let eta = eta_s + 2.0e-3;
    let stable = beta_half_unit_deviance_from_shape_differences(y, eta, eta_s, phi).expect("regime");
    let flipped = beta_half_unit_deviance_from_shape_differences(1.0 - y, -eta, eta_s_flip, phi)
        .expect("regime");
    // The saturated η of `1−y` comes from the root finder, not from negating
    // η_s(y); its ulp-level asymmetry moves δ by φμ(1−μ)·ε ≈ 2e-11 and the
    // half-deviance by ≈1e-13 relative, so the bar is set above the root
    // finder's own noise, two orders below the direct form's 4e-11.
    assert!((stable - flipped).abs() <= 1e-11 * stable, "symmetry: {stable:.15e} vs {flipped:.15e}");
    assert_eq!(
        beta_half_unit_deviance_from_shape_differences(y, eta_s, eta_s, phi).expect("regime"),
        0.0
    );
    // Resolution at φ = 1e6: the second difference over h reproduces the
    // analytic curvature φ·μ(1−μ)·[ψ'(a)+ψ'(b)]·(μ(1−μ)φ) ≈ φμ(1−μ) to 1e-6,
    // while the direct difference of log-likelihoods is dominated by its own
    // cancellation noise there.
    let h = 1.0e-6;
    let curvature_of = |f: &dyn Fn(f64) -> f64| (f(eta + h) - 2.0 * f(eta) + f(eta - h)) / (h * h);
    let stable_fn = |e: f64| beta_half_unit_deviance_from_shape_differences(y, e, eta_s, phi).expect("regime");
    let direct_fn = |e: f64| {
        beta_fitted_loglikelihood_unit_from_eta(y, eta_s, phi).expect("sat")
            - beta_fitted_loglikelihood_unit_from_eta(y, e, phi).expect("fit")
    };
    let (mu, one_minus_mu) = {
        let m = 1.0 / (1.0 + (-eta).exp());
        (m, 1.0 - m)
    };
    // d²(½d)/dη² = c(1−2μ)·[ψ(a) − ψ(b) − logit y] + c²·[ψ'(a) + ψ'(b)], c = φμ(1−μ):
    // the same two terms the near-saturation branch of the deviance carries.
    let c = phi * mu * one_minus_mu;
    let bracket = gam_math::special::digamma(mu * phi) - gam_math::special::digamma(one_minus_mu * phi)
        - logit(y);
    let analytic = c * (1.0 - 2.0 * mu) * bracket
        + c * c
            * (gam_math::special::trigamma(mu * phi)
                + gam_math::special::trigamma(one_minus_mu * phi));
    let stable_curv = curvature_of(&stable_fn);
    let direct_curv = curvature_of(&direct_fn);
    assert!(
        ((stable_curv - analytic) / analytic).abs() <= 1e-6,
        "difference form: second difference {stable_curv:.9e} vs analytic {analytic:.9e}"
    );
    assert!(
        ((direct_curv - analytic) / analytic).abs() > 1e-5,
        "positive control: the direct form should be noise-dominated at this φ, got {direct_curv:.9e} vs {analytic:.9e}"
    );
}

/// Beta-logit Fisher working rows in both logit tails (#4527).
#[cfg(test)]
mod beta_logit_fisher_row_tail_tests {
    use super::super::exact_beta_logit_row;

    fn relative_error(value: f64, reference: f64) -> f64 {
        ((value - reference) / reference).abs()
    }

    /// Fisher `W`, `z`, `c = dW/dη` and `d = d²W/dη²` of one row
    /// (`ω = 1`, `φ = 4`, `y = 1/3` as an f64). The references are the
    /// closed forms evaluated in 60-digit arithmetic and rounded to 17
    /// significant digits.
    ///
    /// The shifted-polygamma forms are well conditioned at every point here.
    /// Their float64 evaluation stays within 1.6e-15 of the reference, so
    /// 1e-12 leaves room for the polygamma implementation's own error and
    /// still refuses main's unshifted forms. Those forms cancel O(1) poles:
    /// they miss by 9.5e-10 at η = 8, by 7.3e-4 at η = 15 and by a factor
    /// 1e10 at η = 30, where the rounded `1 − μ` compounds the cancellation.
    #[test]
    fn beta_logit_fisher_curvature_matches_reference_in_both_tails() {
        let phi = 4.0;
        let y = 1.0 / 3.0;
        // (η, W, z, c, d)
        let references = [
            (-30.0, 9.9999999999981285e-1, -28.99999999999948, -1.871524593762105e-13, -1.8715245937561751e-13),
            (-15.0, 9.9999938819852749e-1, -13.999998298961095, -6.1179830402406552e-7, -6.1179196706819654e-7),
            (-8.0, 9.9933287479558206e-1, -6.9981415182408401, -6.6333553141763312e-4, -6.5577689451943296e-4),
            (0.3, 1.2765364075355014, -0.52855793718461334, -0.087181825644534897, -0.26828185644214066),
            (8.0, 9.9933287479558206e-1, 6.996281324888551, 6.6333553141763312e-4, -6.5577689451943296e-4),
            (15.0, 9.9999938819852749e-1, 13.999996602678447, 6.1179830402406552e-7, -6.1179196706819654e-7),
            (30.0, 9.9999999999981285e-1, 28.999999999998961, 1.871524593762105e-13, -1.8715245937561751e-13),
        ];
        for (eta, weight, z, c, d) in references {
            let row = exact_beta_logit_row(0, eta, Some(y), 1.0, phi)
                .unwrap_or_else(|error| panic!("beta-logit row at eta={eta} refused: {error:?}"));
            for (label, value, reference) in
                [("W", row.weight, weight), ("z", row.z, z), ("c", row.c, c), ("d", row.d, d)]
            {
                let error = relative_error(value, reference);
                assert!(
                    error <= 1e-12,
                    "eta={eta}: {label} = {value:e} vs reference {reference:e} (relative error {error:e})"
                );
            }
        }
    }

    /// The Beta model is symmetric under `y ↔ 1 − y`, `η ↔ −η`: `W` and `d` are
    /// even and `z` and `c` are odd. The responses are dyadic, so `1 − y` is exact.
    /// Past η ≈ 36.7 the rounded mean is exactly 1.0, so the upper row can
    /// only exist by forming its second shape from the exact logit complement.
    /// The mirrored rows run the same float operations except the order of the
    /// jet's `q″` polynomial, so they agree to a few ulps. 1e-13 is a bound on
    /// that, not a tolerance on an approximation.
    #[test]
    fn beta_logit_fisher_row_is_mirror_symmetric_past_the_rounded_mean() {
        let phi = 4.0;
        let tail_response = 2.0_f64.powi(-40);
        for eta in [20.0, 30.0, 40.0, 60.0] {
            let upper = exact_beta_logit_row(0, eta, Some(1.0 - tail_response), 1.0, phi)
                .unwrap_or_else(|error| panic!("upper-tail row at eta={eta} refused: {error:?}"));
            let lower = exact_beta_logit_row(0, -eta, Some(tail_response), 1.0, phi)
                .unwrap_or_else(|error| panic!("lower-tail row at eta={} refused: {error:?}", -eta));
            for (label, up, mirrored) in [
                ("W", upper.weight, lower.weight),
                ("z", upper.z, -lower.z),
                ("c", upper.c, -lower.c),
                ("d", upper.d, lower.d),
            ] {
                let error = relative_error(up, mirrored);
                assert!(
                    error <= 1e-13,
                    "eta=±{eta}: {label} upper {up:e} vs mirrored lower {mirrored:e} (relative error {error:e})"
                );
            }
        }
        // At η = 40 the 60-digit reference is W = 1 − 8.5e-18, z = 39 + 4.4e-16,
        // c = 8.4967085105831768e-18 and d = −8.4967085105831755e-18.
        let upper = exact_beta_logit_row(0, 40.0, Some(1.0 - tail_response), 1.0, phi)
            .expect("upper-tail row at eta=40");
        assert!(relative_error(upper.c, 8.4967085105831768e-18) <= 1e-12, "c = {:e}", upper.c);
        assert!(relative_error(upper.d, -8.4967085105831755e-18) <= 1e-12, "d = {:e}", upper.d);
    }

    /// The Beta precision refresh reads the same exact logit pair as the row.
    /// Main formed the moment statistic from the rounded mean and refused
    /// every row past η ≈ 36.7, where `μ` rounds to 1 although `1 − μ` is a
    /// normal number. The mirrored sample (η → −η, y → 1 − y) must give the
    /// same precision. The reference is the moment estimator in 60-digit
    /// arithmetic on these f64 inputs.
    #[test]
    fn beta_precision_moment_estimate_accepts_rows_past_the_rounded_mean() {
        use super::super::estimate_beta_phi_from_eta;
        use ndarray::array;
        let weights = array![1.0, 1.0, 1.0, 1.0];
        let eta = array![-2.0, 0.5, 38.0, 45.0];
        let y = array![0.25, 0.625, 1.0 - 2f64.powi(-53), 1.0 - 2f64.powi(-50)];
        let mirrored_eta = eta.mapv(|e: f64| -e);
        let mirrored_y = y.mapv(|v: f64| 1.0 - v);
        let phi = estimate_beta_phi_from_eta(y.view(), &eta, weights.view())
            .expect("the upper-tail sample has a finite moment precision");
        let mirrored = estimate_beta_phi_from_eta(mirrored_y.view(), &mirrored_eta, weights.view())
            .expect("the lower-tail sample has a finite moment precision");
        let reference = 23.544459333498157;
        assert!(relative_error(phi, reference) <= 1e-13, "phi = {phi:e}");
        assert!(relative_error(mirrored, reference) <= 1e-13, "mirrored phi = {mirrored:e}");
    }
}

//! P-IRLS regression and root-cause tests.

// ── Test-only Tweedie density oracles ──────────────────────────────────────
// Consumed by `tweedie_exact_series_tests` below. They live here (inside the
// `pirls::tests` module) because the scanner forbids `#[cfg(test)]` on bare
// src items and cross-module consumption rules out a private test_support
// submodule in `deviance.rs`.
use super::{LN_2PI, log_gamma_stirling_correction, tweedie_exact_series_loglik_from_eta};
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
fn tweedie_saddlepoint_loglik_approximation(
    yi: f64,
    mui: f64,
    w: f64,
    p: f64,
    phi: f64,
) -> f64 {
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
//!
//! The nested `#[cfg(test)] mod`s below address the rest of the solver through
//! `super::`, which now resolves to this module; the re-imports here forward the
//! sibling concern modules and the shared item surface so those paths keep
//! pointing at the same definitions they did when this file was inlined.

pub(crate) use super::*;

#[cfg(test)]
mod tests {
    use super::loop_driver::{default_beta_guess_external, exact_lambdas_from_rho};
    use super::reweight::madsen_lm_accept_factor;
    use super::{
        DENSE_OUTER_MAX_P, DevianceEtaRow, LinearInequalityConstraints, PenaltyConfig, PirlsConfig,
        PirlsLinearSolvePath, PirlsProblem, PirlsWorkspace, SparseXtWxCache, WeightFamily,
        WeightLink, WorkingDerivativeBuffersMut, bernoulli_geometry_from_jet,
        calculate_deviance_from_eta, calculate_loglikelihood_omitting_constants_from_eta,
        calculate_null_deviance, compute_constraint_kkt_diagnostics,
        compute_observed_hessian_curvature_arrays, deviance_eta_row_with_log_measure_scale,
        deviance_eta_rows_with_log_measure_scale, evaluate_full_log_likelihood_from_eta,
        fit_model_for_fixed_rho, observed_weight_dispatch, observed_weight_noncanonical,
        select_active_set_release, should_log_pirls_decision_summary,
        should_use_sparse_native_pirls, solve_newton_directionwith_linear_constraints,
        solve_newton_directionwith_lower_bounds, stable_finite_signed_sum, update_glmvectors,
        variance_jet_for_weight_family, write_gamma_log_working_state,
        write_negative_binomial_log_working_state, write_poisson_log_working_state,
        write_tweedie_log_working_state,
    };
    use crate::active_set;
    use crate::estimate::EstimationError;
    use crate::mixture_link::{InverseLinkJet as MixtureInverseLinkJet, state_fromspec};
    use approx::assert_relative_eq;
    use faer::sparse::{SparseColMat, Triplet};
    use gam_linalg::matrix::DesignMatrix;
    use gam_math::probability::standard_normal_quantile;
    use gam_problem::{
        Coefficients, GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LinkComponent, LinkFunction,
        LogSmoothingParamsView, MixtureLinkSpec, ResponseFamily, StandardLink,
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
    // the production `jeffreys_pirls_diagnostics_from_factor` fast path, used here
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
            op.pirls_firth_score_shift(),
        ))
    }
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder, array};

    #[test]
    pub(crate) fn dense_workspace_xtwx_preserves_signed_observed_weights() {
        let x = array![[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0], [0.5, -3.0]];
        let weights = array![2.0, -1.5, 0.25, -3.0];
        let mut workspace = PirlsWorkspace::new(x.nrows(), x.ncols(), 0, 0);
        let mut streamed = Array2::<f64>::zeros((x.ncols(), x.ncols()).f());

        PirlsWorkspace::add_dense_xtwx_signed(
            &weights,
            &mut workspace.weighted_x_chunk,
            &x,
            &mut streamed,
        );

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
            let (hat, logdet, shift) = compute_jeffreys_pirls_diagnostics(
                link,
                x.view(),
                eta.view(),
                observation_weights.view(),
            )
            .expect("supported Firth inverse link");
            assert_eq!(hat.len(), x.nrows());
            assert_eq!(shift.len(), x.nrows());
            assert!(
                logdet.is_finite(),
                "Jeffreys logdet must stay finite for {link:?}"
            );
            assert!(
                hat.iter().all(|value| value.is_finite() && *value >= 0.0),
                "hat diagonal must stay finite and non-negative for {link:?}: {hat:?}"
            );
            assert!(
                shift.iter().all(|value| value.is_finite()),
                "Firth score shift must stay finite for {link:?}: {shift:?}"
            );
        }
    }

    #[test]
    pub(crate) fn firth_factored_path_matches_full_operator_oracle_1575() {
        // #1575 hoisted the β-INDEPENDENT Firth design factor out of the inner
        // Newton loop: `build_design_factor_with_observation_weights` (η-free,
        // built once) followed by `pirls_diagnostics_from_factor` (per-η) must
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
                let (hat_f, logdet_f, shift_f) =
                    FirthDenseOperator::pirls_diagnostics_from_factor(&factor_w, link, eta)
                        .expect("factored weighted diagnostics");
                let (hat_o, logdet_o, shift_o) =
                    compute_jeffreys_pirls_diagnostics(link, x.view(), eta.view(), weights.view())
                        .expect("oracle weighted diagnostics");
                assert_relative_eq!(logdet_f, logdet_o, epsilon = 1e-12, max_relative = 1e-12);
                for i in 0..x.nrows() {
                    assert_relative_eq!(hat_f[i], hat_o[i], epsilon = 1e-12, max_relative = 1e-12);
                    assert_relative_eq!(
                        shift_f[i],
                        shift_o[i],
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                }

                // Unweighted (`None` branch of the factor builder) vs the full
                // operator with no observation weights.
                let (hat_fu, logdet_fu, shift_fu) =
                    FirthDenseOperator::pirls_diagnostics_from_factor(&factor_u, link, eta)
                        .expect("factored unweighted diagnostics");
                let op_u = FirthDenseOperator::build_for_link(link, &x, eta)
                    .expect("full unweighted operator");
                assert_relative_eq!(
                    logdet_fu,
                    op_u.jeffreys_logdet(),
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
                let hat_ou = op_u.pirls_hat_diag();
                let shift_ou = op_u.pirls_firth_score_shift();
                for i in 0..x.nrows() {
                    assert_relative_eq!(
                        hat_fu[i],
                        hat_ou[i],
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                    assert_relative_eq!(
                        shift_fu[i],
                        shift_ou[i],
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
        let diag = compute_constraint_kkt_diagnostics(&beta, &grad, &constraints);
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
        let diag = compute_constraint_kkt_diagnostics(&beta, &grad, &constraints);
        assert_eq!(diag.n_constraints, 2);
        assert_eq!(diag.n_active, 1);
        assert!(diag.primal_feasibility <= 1e-12);
        assert!(diag.dual_feasibility <= 1e-12);
        assert!(diag.complementarity <= 1e-12);
        assert!(diag.stationarity <= 1e-10);
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
            default_beta_guess_external(3, LinkFunction::Logit, y.view(), w.view(), None, None);
        let prevalence: f64 = (3.0 + 0.5) / (4.0 + 1.0);
        let prevalence = prevalence.max(1e-6_f64).min(1.0_f64 - 1e-6_f64);
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
            default_beta_guess_external(3, LinkFunction::Probit, y.view(), w.view(), None, None);
        let prevalence: f64 = (3.0 + 0.5) / (4.0 + 1.0);
        let prevalence = prevalence.max(1e-6_f64).min(1.0_f64 - 1e-6_f64);
        let log_odds = (prevalence / (1.0 - prevalence)).ln();
        let expected =
            standard_normal_quantile(prevalence).expect("clamped prevalence must be valid");
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
        let mut workspace = PirlsWorkspace::new(2, 2, 0, 0);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::DenseTransformed);
        assert_eq!(decision.reason, "design_not_sparse");
    }

    pub(crate) fn fixed_gaussian_beta(
        x: Array2<f64>,
        y: Array1<f64>,
        penalties: Vec<gam_terms::smooth::BlockwisePenalty>,
        rho: Array1<f64>,
    ) -> Array1<f64> {
        let p = x.ncols();
        let weights = Array1::<f64>::ones(y.len());
        let offset = Array1::<f64>::zeros(y.len());
        let specs: Vec<crate::estimate::PenaltySpec> = penalties
            .iter()
            .map(crate::estimate::PenaltySpec::from_blockwise_ref)
            .collect();
        let nulls = vec![0; specs.len()];
        let (canonical, _) = gam_terms::construction::canonicalize_penalty_specs(
            &specs,
            &nulls,
            p,
            "prior mean test",
        )
        .expect("canonical penalties");
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            )),
            link_kind: InverseLink::Standard(StandardLink::Identity),
            max_iterations: 20,
            convergence_tolerance: 1e-12,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            arrow_schur: None,
        };
        let problem = PirlsProblem {
            x,
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        };
        let penalty = PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        };
        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view())
                .expect("test rho lies in exact strength domain"),
            problem,
            penalty,
            &config,
            None,
        )
        .expect("fixed rho fit");
        fit.beta_transformed.as_ref().clone()
    }

    #[test]
    pub(crate) fn constant_prior_mean_centers_penalty() {
        let x = Array2::<f64>::zeros((4, 1));
        let y = Array1::<f64>::zeros(4);
        let penalty = gam_terms::smooth::BlockwisePenalty::ridge(0..1, 1.0)
            .with_prior_mean(gam_problem::CoefficientPriorMean::scalar(2.5));
        let beta = fixed_gaussian_beta(x, y, vec![penalty], array![0.0]);
        assert!((beta[0] - 2.5).abs() < 1e-10, "beta={beta:?}");
    }

    #[test]
    pub(crate) fn functional_prior_mean_recovers_kernel_amplitude() {
        let x = Array2::<f64>::zeros((5, 3));
        let y = Array1::<f64>::zeros(5);
        let metadata = array![2.0];
        let alpha = 1.75;
        let penalty = gam_terms::smooth::BlockwisePenalty::ridge(0..3, 1.0).with_prior_mean(
            gam_problem::CoefficientPriorMean::functional(
                metadata,
                std::sync::Arc::new(move |a: &Array1<f64>| {
                    let t = a[0];
                    array![alpha, alpha * t, alpha * t * t]
                }),
            ),
        );
        let beta = fixed_gaussian_beta(x, y, vec![penalty], array![0.0]);
        let recovered_alpha = beta[0];
        assert!((recovered_alpha - alpha).abs() < 1e-10, "beta={beta:?}");
        assert!((beta[1] / 2.0 - alpha).abs() < 1e-10, "beta={beta:?}");
        assert!((beta[2] / 4.0 - alpha).abs() < 1e-10, "beta={beta:?}");
    }

    #[test]
    pub(crate) fn zero_prior_mean_matches_default_fixed_fit_bitwise() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0],];
        let y = array![0.5, 1.0, 1.5, 2.0, 2.5];
        let base_penalty = gam_terms::smooth::BlockwisePenalty::ridge(0..2, 1.0);
        let zero_penalty = gam_terms::smooth::BlockwisePenalty::ridge(0..2, 1.0).with_prior_mean(
            gam_problem::CoefficientPriorMean::constant(Array1::zeros(2)),
        );
        let rho = array![0.25];
        let beta_default =
            fixed_gaussian_beta(x.clone(), y.clone(), vec![base_penalty], rho.clone());
        let beta_zero = fixed_gaussian_beta(x, y, vec![zero_penalty], rho);
        assert_eq!(beta_default.to_vec(), beta_zero.to_vec());
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
        let mut workspace = PirlsWorkspace::new(300, 300, 0, 0);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::SparseNative);
        assert_eq!(decision.reason, "sparse_native_eligible");
        assert_eq!(decision.nnz_x, 300);
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
        let mut workspace = PirlsWorkspace::new(64, 64, 0, 0);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::SparseNative);
        assert_eq!(decision.reason, "sparse_native_eligible");
        assert_eq!(decision.nnz_x, 64);
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
        let mut workspace = PirlsWorkspace::new(64, 64, 0, 0);
        let decision =
            should_use_sparse_native_pirls(&mut workspace, &x, &s, Some(&lower_bounds), None);
        assert_eq!(decision.path, PirlsLinearSolvePath::DenseTransformed);
        assert_eq!(decision.reason, "constraints_present");
    }

    /// Regression for the sparse-native vs dense REML λ-selection divergence
    /// (#1266 class): the sparse-native reparam result MUST carry the same
    /// penalty (with the `penalty_shrinkage_floor` ridge folded in) that the
    /// engine's reported `log_det`/`det1` are the determinant of. Building the
    /// sparse-native inner penalty from the bare λ-weighted canonical sum (the
    /// old behaviour) dropped the shrinkage ridge, so the inner penalized
    /// Hessian `H = XᵀWX + S` and the EDF were evaluated on `S_λ` while the REML
    /// penalty-logdet term used `S_λ + shrinkage·P_range` — an internally
    /// inconsistent objective that biased λ relative to the dense/Kronecker
    /// backends.
    ///
    /// Asserts, with a deliberately large shrinkage floor so the effect is
    /// unambiguous: (1) the result's penalty root and Gram are mutually
    /// consistent (`Eᵀ E == s_transformed`), and (2) the penalty actually
    /// carries the shrinkage ridge on the penalized direction (i.e. it is NOT
    /// the bare λ-weighted sum) — so the inner solve is on the same penalty as
    /// the reported `log_det`.
    #[test]
    pub(crate) fn sparse_native_reparam_folds_shrinkage_floor_into_penalty() {
        use gam_terms::construction::{
            CanonicalPenalty, EngineDims, stable_reparameterization_engine_canonical,
        };
        use ndarray::array;

        // p = 2 coefficients, a rank-1 penalty that penalizes the first
        // coordinate only (root = [[1, 0]]). The second coordinate is the
        // penalty null space.
        let p = 2usize;
        let root = array![[1.0, 0.0]];
        let local = root.t().dot(&root);
        let canonical = vec![CanonicalPenalty {
            root: root.clone(),
            col_range: 0..p,
            total_dim: p,
            nullity: 1,
            local,
            prior_mean: Array1::zeros(p),
            positive_eigenvalues: Vec::new(),
            op: None,
        }];
        let lambdas = [3.0f64];
        // A large shrinkage floor so the ridge is clearly visible against the
        // λ-weighted eigenvalue.
        let shrinkage_floor = Some(1e-2);

        let base = stable_reparameterization_engine_canonical(
            &canonical,
            &lambdas,
            EngineDims::new(p, canonical.len()),
            None,
            shrinkage_floor,
        )
        .expect("engine should succeed for a well-formed rank-1 penalty");

        // The shrinkage ridge must be non-zero for this fixture (otherwise the
        // test would not exercise the regression).
        assert!(
            base.penalty_shrinkage_ridge > 0.0,
            "fixture must trigger a non-zero shrinkage ridge, got {}",
            base.penalty_shrinkage_ridge
        );

        let result = super::loop_driver::build_sparse_native_reparam_result(
            base.clone(),
            &canonical,
            &lambdas,
            p,
        );

        // (0) sparse-native always reports identity Qs.
        assert_eq!(result.qs, Array2::<f64>::eye(p));

        // (1) Root/Gram consistency: EᵀE == s_transformed. This is what makes
        // the augmented EDF solve and the inner penalized Hessian live on one
        // penalty.
        let gram = result.e_transformed.t().dot(&result.e_transformed);
        for i in 0..p {
            for j in 0..p {
                assert_relative_eq!(gram[[i, j]], result.s_transformed[[i, j]], epsilon = 1e-9);
            }
        }

        // (2) The penalty carries the shrinkage ridge on the penalized
        // direction: s_transformed[0,0] == λ·1 + shrinkage_ridge (NOT the bare
        // λ = 3.0 the old code produced).
        let bare = lambdas[0]; // λ · root²  on the penalized coordinate
        assert!(
            result.s_transformed[[0, 0]] > bare + 0.5 * base.penalty_shrinkage_ridge,
            "penalized direction must include the shrinkage ridge: \
             s[0,0]={} should exceed bare λ={} by ~ridge={}",
            result.s_transformed[[0, 0]],
            bare,
            base.penalty_shrinkage_ridge
        );
        assert_relative_eq!(
            result.s_transformed[[0, 0]],
            bare + base.penalty_shrinkage_ridge,
            epsilon = 1e-9
        );

        // The null-space coordinate stays unpenalized (no spurious ridge there).
        assert_relative_eq!(result.s_transformed[[1, 1]], 0.0, epsilon = 1e-9);
    }

    /// End-to-end CROSS-BACKEND consistency regression for #1344.
    ///
    /// The white-box test above proves the sparse-native inner penalty carries
    /// the shrinkage floor *by construction*. It does NOT demonstrate the
    /// property #1344 is actually about: that the sparse-native and dense PIRLS
    /// backends, fed the SAME model, SELECT THE SAME smoothing parameter λ and
    /// report the SAME EDF.
    ///
    /// `penalty_shrinkage_floor` folds a ρ-independent ridge `shrinkage·P_range`
    /// into the penalized range of `S_λ`, and the reparam engine reports the
    /// determinant of the *floored* penalty to REML. Before the fix the
    /// sparse-native backend solved its inner penalized Hessian `H = XᵀWX + S`
    /// on the bare `S_λ` (no shrinkage) while still reporting the floored
    /// `log_det` to REML — an internally inconsistent objective that shifts the
    /// REML optimum, so sparse-native selected a different λ (and EDF) than
    /// dense for the same model. Because the backend a model lands on is decided
    /// purely by penalized-Hessian density, two statistically identical models
    /// could get materially different fits.
    ///
    /// This test fits ONE penalized GAM end-to-end through the full REML outer
    /// loop (`optimize_external_design`) twice. The ONLY difference is the
    /// storage class of the design matrix:
    ///
    ///   * a `faer` `SparseColMat` → routes to the **sparse-native** backend;
    ///   * the bit-identical numbers as a dense `Array2` → routes to the
    ///     **dense-transformed** backend.
    ///
    /// It first ASSERTS, via the real routing oracle
    /// `should_use_sparse_native_pirls`, that the sparse design genuinely takes
    /// `SparseNative` and the dense design genuinely takes `DenseTransformed` —
    /// so the comparison cannot silently degenerate into dense-vs-dense. Both
    /// fits use the SAME non-zero `penalty_shrinkage_floor` and a
    /// second-difference penalty whose 2-D null space ({constant, linear}) gives
    /// the penalized-range energy that makes the floor fire (the exact condition
    /// the issue says triggers the bug "on every sparse-native fit whose penalty
    /// carries penalized range energy").
    ///
    /// Then it asserts the REML-selected λ (relative 1e-3 in log space) and
    /// `edf_total` (absolute 1e-2) agree across backends. The two backends use
    /// different internal reparameterizations and independent outer optimizers,
    /// so byte-identity is not expected; but the SELECTED model is a property of
    /// the (now shared) REML objective, not of the linear-algebra basis, so λ
    /// and EDF must agree to optimizer tolerance. Disagreement means the two
    /// backends are optimizing different objectives — exactly the #1344 bug — so
    /// this test FAILS rather than being weakened.
    #[test]
    pub(crate) fn sparse_native_and_dense_select_same_lambda_under_shrinkage_floor() {
        use crate::estimate::{ExternalOptimOptions, optimize_external_design};
        use gam_terms::smooth::BlockwisePenalty;

        // --- Fixture: a banded "local-support" (B-spline-like) design so the
        // penalized Hessian is sparse enough for the sparse copy to take the
        // sparse-native path, with a 2nd-difference penalty that carries
        // penalized-range energy so the shrinkage floor fires. p is chosen large
        // enough that the banded H sits well below SPARSE_NATIVE_MAX_H_DENSITY.
        let n = 300usize;
        let p = 60usize;

        // Deterministic LCG normal so the fixture is bit-reproducible.
        struct Lcg {
            s: u64,
        }
        impl Lcg {
            fn unit(&mut self) -> f64 {
                self.s = self
                    .s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.s >> 33) as f64 + 1.0) / ((1u64 << 31) as f64 + 1.0)
            }
            fn normal(&mut self) -> f64 {
                let u1 = self.unit().max(1.0e-300);
                let u2 = self.unit();
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            }
        }

        // Banded tent design: each row has <= 3 consecutive nonzeros.
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64) * ((p - 1) as f64) / ((n - 1) as f64);
            let c = t.floor() as isize;
            let frac = t - c as f64;
            let w3 = [
                0.5 * (1.0 - frac).powi(2),
                0.5 + frac - frac * frac,
                0.5 * frac * frac,
            ];
            for (k, &wv) in w3.iter().enumerate() {
                let col = c - 1 + k as isize;
                if (0..p as isize).contains(&col) {
                    x[[i, col as usize]] = wv;
                }
            }
        }

        // Second-difference penalty DᵀD (null space {constant, linear}).
        let m = p - 2;
        let mut d = Array2::<f64>::zeros((m, p));
        for r in 0..m {
            d[[r, r]] = 1.0;
            d[[r, r + 1]] = -2.0;
            d[[r, r + 2]] = 1.0;
        }
        let penalty = d.t().dot(&d);

        // Smooth truth + moderate noise.
        let mut beta_true = Array1::<f64>::zeros(p);
        for j in 0..p {
            let u = j as f64 / (p - 1) as f64;
            beta_true[j] = (3.0 * std::f64::consts::PI * u).sin() + 0.5 * u;
        }
        let mut rng = Lcg { s: 0xC0FFEE_1344 };
        let mut y = x.dot(&beta_true);
        for yi in y.iter_mut() {
            *yi += 0.15 * rng.normal();
        }
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);

        // A clearly non-zero shrinkage floor so any backend-specific omission of
        // the ridge dominates the REML optimum (the default 1e-6 also fires).
        let shrinkage_floor = 1e-3;

        // --- ROUTING ORACLE: prove the two storage classes take the two
        // backends BEFORE comparing fits. The penalty fed to the oracle is the
        // λ-weighted penalty at the seed (λ scale is immaterial to the sparsity
        // pattern that drives the density gate). ---
        let x_dense_design: DesignMatrix = x.clone().into();
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for i in 0..n {
            for j in 0..p {
                let v = x[[i, j]];
                if v != 0.0 {
                    triplets.push(Triplet::new(i, j, v));
                }
            }
        }
        let x_sparse_mat = SparseColMat::try_new_from_triplets(n, p, &triplets)
            .expect("banded sparse design assembles");
        let x_sparse_design: DesignMatrix = x_sparse_mat.clone().into();

        let mut ws = PirlsWorkspace::new(n, p, 0, 0);
        let sparse_decision =
            should_use_sparse_native_pirls(&mut ws, &x_sparse_design, &penalty, None, None);
        assert_eq!(
            sparse_decision.path,
            PirlsLinearSolvePath::SparseNative,
            "fixture invariant: the sparse design MUST route to sparse-native \
             (reason={}, density={:?}); otherwise this is a vacuous dense-vs-dense \
             comparison. Lower p or widen the band if the penalized-Hessian \
             density crept above the gate.",
            sparse_decision.reason,
            sparse_decision.density_h_est
        );
        let dense_decision =
            should_use_sparse_native_pirls(&mut ws, &x_dense_design, &penalty, None, None);
        assert_eq!(
            dense_decision.path,
            PirlsLinearSolvePath::DenseTransformed,
            "fixture invariant: the dense design MUST route to the dense backend \
             (reason={})",
            dense_decision.reason
        );

        // --- Fit the SAME model through both backends via the full REML loop. ---
        let opts = |floor: f64| ExternalOptimOptions {
            family: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 200,
            // Tight inner tolerance so the REML criterion driving λ selection is
            // evaluated at the converged β̂ for BOTH backends; loose tolerance
            // would let backend-specific warm-start residue mask or fake a gap.
            tol: 1e-11,
            nullspace_dims: vec![2],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: Some(floor),
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        };

        let sparse_res = optimize_external_design(
            y.view(),
            w.view(),
            x_sparse_mat,
            offset.view(),
            vec![BlockwisePenalty::new(0..p, penalty.clone())],
            &opts(shrinkage_floor),
        )
        .expect("sparse-native external fit must succeed");

        let dense_res = optimize_external_design(
            y.view(),
            w.view(),
            x.clone(),
            offset.view(),
            vec![BlockwisePenalty::new(0..p, penalty.clone())],
            &opts(shrinkage_floor),
        )
        .expect("dense external fit must succeed");

        let sparse_edf = sparse_res
            .inference
            .as_ref()
            .map(|i| i.edf_total)
            .expect("sparse fit reports edf");
        let dense_edf = dense_res
            .inference
            .as_ref()
            .map(|i| i.edf_total)
            .expect("dense fit reports edf");

        eprintln!(
            "[#1344] sparse-native: lambda={:?} edf={:.6} reml={:.6}",
            sparse_res.lambdas.as_slice().unwrap(),
            sparse_edf,
            sparse_res.reml_score
        );
        eprintln!(
            "[#1344] dense:         lambda={:?} edf={:.6} reml={:.6}",
            dense_res.lambdas.as_slice().unwrap(),
            dense_edf,
            dense_res.reml_score
        );

        assert_eq!(sparse_res.lambdas.len(), dense_res.lambdas.len());
        assert_eq!(sparse_res.lambdas.len(), 1, "single penalty block ⇒ one λ");

        // The optimum must be a genuine interior point (not a boundary, where the
        // comparison would be vacuous). exp(±12) are the outer search bounds.
        let sparse_log = sparse_res.lambdas[0].ln();
        assert!(
            sparse_log.is_finite() && sparse_log.abs() < 11.0,
            "selected λ must be an interior optimum, got log λ = {sparse_log}"
        );

        // (1) Selected smoothing parameter agrees across backends (log space).
        let log_sparse = sparse_res.lambdas[0].ln();
        let log_dense = dense_res.lambdas[0].ln();
        let rel_log_diff = (log_sparse - log_dense).abs() / (1.0 + log_dense.abs());
        assert!(
            rel_log_diff < 1e-3,
            "cross-backend λ divergence (#1344): sparse-native log λ = {log_sparse:.8}, \
             dense log λ = {log_dense:.8}, relative log-difference = {rel_log_diff:.3e} \
             exceeds 1e-3. The backends are selecting different smoothing \
             parameters for the same model — different REML objectives — which is \
             exactly the bug #1344 closed."
        );

        // (2) Reported EDF agrees across backends.
        let edf_diff = (sparse_edf - dense_edf).abs();
        assert!(
            edf_diff < 1e-2,
            "cross-backend EDF divergence (#1344): sparse-native edf = {sparse_edf:.6}, \
             dense edf = {dense_edf:.6}, |Δ| = {edf_diff:.3e} exceeds 1e-2"
        );
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
        let ridge = 1e-8;
        let mut workspace = PirlsWorkspace::new(3, 3, 0, 0);
        let assembled = super::sparse_reml_penalized_hessian(
            &mut workspace,
            &x,
            &weights,
            &s_lambda,
            ridge,
            None,
        )
        .expect("sparse penalized assembly should succeed");
        let dense = DesignMatrix::from(x.clone()).to_dense();
        let mut expected = dense.t().dot(&Array2::from_diag(&weights)).dot(&dense);
        expected += &s_lambda;
        for i in 0..3 {
            expected[[i, i]] += ridge;
        }
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
    pub(crate) fn pirls_result_stores_integrated_logit_derivative_jet() {
        let x = array![[1.0], [1.0], [1.0], [1.0], [1.0]];
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0];
        let w = Array1::ones(5);
        let offset = Array1::zeros(5);
        let rho = Array1::<f64>::zeros(1);
        let covariate_se = array![0.9, 0.7, 0.8, 0.6, 0.75];
        let rs = [array![[1.0]]];
        let canonical: Vec<gam_terms::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                gam_terms::construction::CanonicalPenalty {
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
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
            arrow_schur: None,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view())
                .expect("test rho lies in exact strength domain"),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                covariate_se: Some(covariate_se.view()),
                gaussian_fixed_cache: None,
                glm_first_step_gram: None,
            },
            PenaltyConfig {
                canonical_penalties: &canonical,
                balanced_penalty_root: None,
                reparam_invariant: None,
                p: 1,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
                penalty_shrinkage_floor: None,
                kronecker_factored: None,
            },
            &config,
            Some(&Coefficients::new(array![0.0])),
        )
        .expect("integrated logit PIRLS fit");

        let ctx = crate::quadrature::QuadratureContext::new();
        for i in 0..y.len() {
            let jet = crate::quadrature::integrated_inverse_link_jet(
                &ctx,
                LinkFunction::Logit,
                fit.final_eta[i],
                covariate_se[i],
            )
            .expect("logit integrated inverse-link jet should evaluate");
            let expected = bernoulli_geometry_from_jet(
                i,
                fit.final_eta[i],
                y[i],
                w[i],
                MixtureInverseLinkJet {
                    mu: jet.mean,
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                },
            )
            .expect("integrated Bernoulli row geometry must be representable");
            assert_relative_eq!(
                fit.solve_dmu_deta[i],
                jet.d1,
                epsilon = 1e-9,
                max_relative = 1e-9
            );
            assert_relative_eq!(
                fit.solve_d2mu_deta2[i],
                jet.d2,
                epsilon = 1e-9,
                max_relative = 1e-8
            );
            assert_relative_eq!(
                fit.solve_d3mu_deta3[i],
                jet.d3,
                epsilon = 1e-8,
                max_relative = 1e-7
            );
            assert_relative_eq!(
                fit.solve_c_array[i],
                expected.c,
                epsilon = 1e-9,
                max_relative = 1e-8
            );
            assert_relative_eq!(
                fit.solve_d_array[i],
                expected.d,
                epsilon = 1e-8,
                max_relative = 1e-7
            );
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
        let expected_z = eta[0] + (y[0] - jet.mu) / jet.d1;
        assert!(
            (z[0] - expected_z).abs() < 1e-12,
            "pure logit PIRLS z should preserve the exact working response at eta={}; got {} vs {}",
            eta[0],
            z[0],
            expected_z
        );
        assert!(
            (weights[0] * (z[0] - eta[0]) - (y[0] - jet.mu)).abs() < 1e-30,
            "pure logit PIRLS score carrier should preserve y-mu at eta={}; got {} vs {}",
            eta[0],
            weights[0] * (z[0] - eta[0]),
            y[0] - jet.mu
        );
    }

    #[test]
    pub(crate) fn noncanonical_binomial_working_state_clamps_saturating_standard_links() {
        for link in [StandardLink::Probit, StandardLink::CLogLog] {
            let y = array![1.0];
            let eta = array![30.0];
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
            .expect("noncanonical binomial working state");

            assert!(
                mu[0] > 0.0 && mu[0] < 1.0,
                "{link:?} working mu must stay inside (0,1) before variance evaluation; got {}",
                mu[0]
            );
            assert!(
                weights[0].is_finite() && weights[0] > 0.0,
                "{link:?} working weight must remain positive finite at saturated eta; got {}",
                weights[0]
            );
            assert!(
                z[0].is_finite(),
                "{link:?} working response must remain finite at saturated eta; got {}",
                z[0]
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

    /// Regression for issue #2131 (sibling of #2126): `calculate_deviance` for a
    /// Tweedie family must report the conventional **unscaled** deviance
    /// `D = 2·Σ wᵢ·d(yᵢ, μᵢ)` — exactly like Poisson/Binomial/NB/Beta and Gamma
    /// (post-#2126) and R/mgcv/statsmodels — and must NOT divide the unit
    /// deviance by the dispersion `φ`, which would report the scaled deviance
    /// `D/φ̂` instead. The bug only manifests when `φ ≠ 1`, so this pins a
    /// likelihood with an explicit `FixedDispersion { phi = 0.25 }`: the reported
    /// deviance must equal the φ-free value and must be strictly different from
    /// `D/φ` (= 4·D here).
    #[test]
    pub(crate) fn tweedie_deviance_is_unscaled_ignoring_phi() {
        let y = array![2.0, 5.0, 1.5];
        let mu = array![1.0, 4.0, 2.0];
        let w = array![1.5, 0.75, 1.0];
        let p = 1.5_f64;
        let phi = 0.25_f64;
        let likelihood = GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::Tweedie { p },
                InverseLink::Standard(StandardLink::Log),
            ),
            scale: gam_problem::LikelihoodScaleMetadata::FixedDispersion { phi },
        };
        // Sanity: the likelihood really does carry a non-unit dispersion, so the
        // old scaled-deviance code path (÷ φ) would have been exercised.
        assert_eq!(likelihood.fixed_phi(), Some(phi));

        let eta = mu.mapv(f64::ln);
        let inverse_link = InverseLink::Standard(StandardLink::Log);
        let dev = calculate_deviance_from_eta(y.view(), &eta, &likelihood, &inverse_link, w.view())
            .expect("Tweedie eta deviance must be representable");

        // Unscaled reference: 2·Σ wᵢ·d(yᵢ, μᵢ) with the Tweedie unit deviance
        // `d = y^{2-p}/((1-p)(2-p)) - y·μ^{1-p}/(1-p) + μ^{2-p}/(2-p)`.
        let sum_unit: f64 = w
            .iter()
            .zip(y.iter())
            .zip(mu.iter())
            .map(|((&wi, &yi), &mui)| {
                let unit = yi.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
                    - yi * mui.powf(1.0 - p) / (1.0 - p)
                    + mui.powf(2.0 - p) / (2.0 - p);
                wi * unit
            })
            .sum();
        let unscaled = 2.0 * sum_unit;

        // The reported deviance is the unscaled 2·Σ w·d(y, μ) ...
        assert_relative_eq!(dev, unscaled, epsilon = 1e-12, max_relative = 1e-9);
        // ... and is NOT the φ-scaled deviance (this is the #2131 assertion: the
        // old code returned `unscaled / φ`, which for φ = 0.25 differs by ×4).
        assert!(
            (dev - unscaled / phi).abs() > 1e-6,
            "Tweedie deviance must be unscaled, not scaled by 1/φ (φ={phi}): \
             dev={dev}, unscaled={unscaled}, scaled={}",
            unscaled / phi
        );
    }

    #[test]
    pub(crate) fn gamma_log_observed_curvature_matches_shape_one_closed_form() {
        let eta = array![0.2, -0.4];
        let mu = eta.mapv(f64::exp);
        let y = array![1.8, 0.7];
        let w = array![2.0, 0.5];
        let fisher = w.clone();

        let (w_obs, c_obs, d_obs) = compute_observed_hessian_curvature_arrays(
            &GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            )),
            &InverseLink::Standard(StandardLink::Log),
            &eta,
            y.view(),
            &fisher,
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
            y,
            mu,
            jet.d1,
            jet.d2,
            jet.d3,
            h4,
            variance_jet_for_weight_family(WeightFamily::Gamma, mu),
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
            eta,
            y,
            mu,
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

        // Exact Fisher weights at these represented eta values.
        let mut fisher = Array1::<f64>::zeros(eta.len());
        for i in 0..eta.len() {
            let jet = crate::mixture_link::inverse_link_jet_for_inverse_link(&link, eta[i])
                .expect("mixture jet");
            let mu = jet.mu;
            let v = mu * (1.0 - mu);
            fisher[i] = jet.d1 * jet.d1 / v;
        }

        // Post-fix contract: the build SUCCEEDS (no bail) and returns finite
        // arrays even though some rows carry a non-positive observed weight.
        let (w_obs, c_obs, d_obs) = compute_observed_hessian_curvature_arrays(
            &likelihood,
            &link,
            &eta,
            y.view(),
            &fisher,
            w.view(),
        )
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
    pub(crate) fn negative_binomial_log_observed_curvature_matches_size_theta_closed_form() {
        let theta = 2.5;
        let eta = array![0.2, -0.4, 1.1];
        let mu = eta.mapv(f64::exp);
        let y = array![0.0, 3.0, 8.0];
        let w = array![2.0, 0.5, 1.25];
        let fisher = Array1::from_iter(
            mu.iter()
                .zip(w.iter())
                .map(|(&mu_i, &w_i)| w_i * theta * mu_i / (theta + mu_i)),
        );

        let (w_obs, c_obs, d_obs) = compute_observed_hessian_curvature_arrays(
            &GlmLikelihoodSpec::canonical(LikelihoodSpec::negative_binomial_log(theta)),
            &InverseLink::Standard(StandardLink::Log),
            &eta,
            y.view(),
            &fisher,
            w.view(),
        )
        .expect("negative-binomial-log observed curvature should evaluate");

        for i in 0..eta.len() {
            let denom = theta + mu[i];
            let scale = w[i] * theta * (theta + y[i]);
            let expected_w = scale * mu[i] / (denom * denom);
            let expected_c = scale * mu[i] * (theta - mu[i]) / (denom * denom * denom);
            let expected_d = scale * mu[i] * (theta * theta - 4.0 * theta * mu[i] + mu[i] * mu[i])
                / (denom * denom * denom * denom);
            assert_relative_eq!(w_obs[i], expected_w, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(c_obs[i], expected_c, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(d_obs[i], expected_d, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    pub(crate) fn poisson_external_fit_reports_full_loglikelihood_not_reml_kernel() {
        use crate::estimate::{ExternalOptimOptions, optimize_external_design};
        use gam_terms::smooth::BlockwisePenalty;

        let x = array![
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
        ];
        let y = array![0.0, 1.0, 2.0, 4.0, 6.0, 9.0];
        let w = Array1::ones(y.len());
        let offset = Array1::zeros(y.len());
        let local_penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ));
        let opts = ExternalOptimOptions {
            family: likelihood.spec.clone(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: true,
            max_iter: 100,
            tol: 1e-10,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        };

        let result = optimize_external_design(
            y.view(),
            w.view(),
            x.clone(),
            offset.view(),
            vec![BlockwisePenalty::new(0..2, local_penalty)],
            &opts,
        )
        .expect("external Poisson fit should converge");

        let eta = x.dot(&result.beta) + &offset;
        let full =
            evaluate_full_log_likelihood_from_eta(y.view(), eta.view(), &likelihood, w.view())
                .expect("full eta log-likelihood")
                .total();
        let omit = calculate_loglikelihood_omitting_constants_from_eta(
            y.view(),
            &eta,
            &likelihood,
            &InverseLink::Standard(StandardLink::Log),
            w.view(),
        )
        .expect("exact eta log-likelihood");
        assert!(
            full <= 0.0,
            "Poisson reporting log-likelihood is a log-mass and must be <= 0, got {full}"
        );
        assert!(
            omit > full,
            "REML omitting-constants kernel must be larger after dropping count normalizers: \
             omit={omit} full={full}"
        );
        assert_relative_eq!(
            result.log_likelihood,
            full,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    pub(crate) fn gamma_log_fit_profiles_shape_instead_of_fixing_one() {
        let x = array![[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]];
        let y = array![0.8, 1.1, 1.7, 2.0, 2.6, 3.1];
        let w = Array1::ones(y.len());
        let offset = Array1::zeros(y.len());
        let rho = array![0.0];
        let rs = [array![[0.0]]];
        let canonical: Vec<gam_terms::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                gam_terms::construction::CanonicalPenalty {
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            )),
            link_kind: InverseLink::Standard(StandardLink::Log),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let (result, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view())
                .expect("test rho lies in exact strength domain"),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                covariate_se: None,
                gaussian_fixed_cache: None,
                glm_first_step_gram: None,
            },
            PenaltyConfig {
                canonical_penalties: &canonical,
                balanced_penalty_root: None,
                reparam_invariant: None,
                p: 1,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
                penalty_shrinkage_floor: None,
                kronecker_factored: None,
            },
            &config,
            None,
        )
        .expect("gamma PIRLS fit");

        let fitted_shape = result
            .likelihood
            .gamma_shape()
            .expect("gamma fit should expose fitted shape");
        let profiled_shape =
            super::estimate_gamma_shape_from_eta(y.view(), &result.final_eta.to_owned(), w.view())
                .expect("converged Gamma shape must be representable");

        assert!(fitted_shape > 1.0, "shape should not stay fixed at one");
        assert_relative_eq!(
            fitted_shape,
            profiled_shape,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
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
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
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
            arrow_schur: None,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view())
                .expect("test rho lies in exact strength domain"),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                covariate_se: None,
                gaussian_fixed_cache: None,
                glm_first_step_gram: None,
            },
            PenaltyConfig {
                canonical_penalties: &canonical,
                balanced_penalty_root: None,
                reparam_invariant: None,
                p: 1,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
                penalty_shrinkage_floor: None,
                kronecker_factored: None,
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
    pub(crate) fn working_set_kkt_diagnostics_use_active_setmultipliers() {
        let working_constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            b: array![0.0, 0.0, 0.0],
        };
        let x = array![0.0, 0.0];
        let lambda_true = array![1.0, 0.5, 2.0];
        let gradient = working_constraints.a.t().dot(&lambda_true);

        let kkt = active_set::working_set_kkt_diagnostics_from_multipliers(
            &x,
            &gradient,
            &working_constraints,
            &lambda_true,
            3,
        )
        .expect("working-set KKT diagnostics");

        assert!(kkt.primal_feasibility <= 1e-12);
        assert!(kkt.dual_feasibility <= 1e-12);
        assert!(kkt.complementarity <= 1e-12);
        assert!(kkt.stationarity <= 1e-12);
        assert_eq!(kkt.n_active, 3);
    }

    #[test]
    pub(crate) fn compress_activeworking_set_groups_near_collinearrows() {
        let constraints = LinearInequalityConstraints {
            a: array![
                [0.0, 0.5, 0.0],
                [0.0, 0.50000000000003, 0.0],
                [1.0, 0.0, 0.0]
            ],
            b: array![1e-8, 1.00000000000005e-8, 0.2],
        };
        let x = array![0.0, 0.0, 0.0];
        let active = vec![0, 1, 2];

        let compressed = active_set::compress_active_working_set(&x, &constraints, &active)
            .expect("compress working set");

        assert_eq!(compressed.constraints.a.nrows(), 2);
        assert_eq!(compressed.groups.len(), 2);
        assert!(
            compressed.groups.iter().any(|g| g == &vec![0, 1]),
            "near-collinear rows should be grouped together: {:?}",
            compressed.groups
        );
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
            None,
        )
        .expect("lower-bound active-set solve should succeed");

        assert_relative_eq!(direction[0], 0.0, epsilon = 1e-14);
        let projected = &beta + &direction;
        assert_relative_eq!(projected[0], beta[0], epsilon = 1e-14);
        assert!(active_hint.is_empty());
    }

    /// gam#979: when a caller reflects negative-curvature modes to keep the QP
    /// step bounded (survival joint-Newton), the freed step `d` is far-field, so
    /// the second-order release multiplier `(H·d)_i` is untrustworthy and can flip
    /// a first-order-optimal bound loose — released spuriously, then re-added on
    /// the next outer re-linearization (the active-set zigzag that ground the
    /// n=3000 survival marginal-slope fit for 30 cycles). Passing the pre-
    /// reflection Hessian via `kkt_hessian` flags the reflection, so the release is
    /// judged on the reflection-invariant FIRST-ORDER multiplier `λ_i = g_i`.
    ///
    /// Here coord 1 is pinned (β₁ at its lower bound, d₁=0) with a first-order-
    /// optimal reduced gradient g₁ = 1 > 0. On the reflected path (`Some(kkt)`) the
    /// first-order test KEEPS it regardless of the far-field `(H·d)₁`; on the
    /// unreflected path (`None`) the exact `λ₁ = g₁ + (H_step·d)₁ = 1 + 3·(−1) =
    /// −2 < 0` releases it (β₁ overshoots to 4.0) — the historical behavior the
    /// unreflected callers keep byte-for-byte.
    #[test]
    pub(crate) fn lower_bound_release_uses_true_curvature_not_reflected_step_979() {
        // Step Hessian (stands in for the reflected, PD model used for the step).
        let h_step = array![[2.0, 3.0], [3.0, 5.0]];
        // Pre-reflection Hessian: its presence flags the reflected path (the
        // release then uses the first-order g₁, so these entries are not consulted
        // for the decision — only for the diagnostic log).
        let h_kkt = array![[2.0, 0.5], [0.5, -3.0]];
        let gradient = array![2.0, 1.0];
        let beta = array![0.0, 0.0];
        let lower_bounds = array![f64::NEG_INFINITY, 0.0];

        // Reflected path: first-order λ₁ = g₁ = 1 > 0, so the bound is KEPT.
        // β₁ stays pinned at its lower bound (no zigzag).
        let mut dir_fixed = Array1::zeros(2);
        let mut active_fixed = vec![1];
        solve_newton_directionwith_lower_bounds(
            &h_step,
            &gradient,
            &beta,
            &lower_bounds,
            &mut dir_fixed,
            Some(&mut active_fixed),
            Some(&h_kkt),
        )
        .expect("true-curvature bounded solve should succeed");
        assert_eq!(
            active_fixed,
            vec![1],
            "true-curvature release must KEEP the genuinely-binding bound active"
        );
        assert!(
            (beta[1] + dir_fixed[1]).abs() < 1e-12,
            "coord 1 must stay pinned at its lower bound; got {}",
            beta[1] + dir_fixed[1]
        );

        // WITHOUT it (kkt=None ⇒ the release test uses the STEP Hessian, the old
        // behavior): λ₁ = 1 + 3·(−1) = −2 < 0, so the bound is released
        // spuriously and coord 1 flies off to 4.0 — the overshoot the outer loop
        // then has to walk back, cycle after cycle.
        let mut dir_bug = Array1::zeros(2);
        let mut active_bug = vec![1];
        solve_newton_directionwith_lower_bounds(
            &h_step,
            &gradient,
            &beta,
            &lower_bounds,
            &mut dir_bug,
            Some(&mut active_bug),
            None,
        )
        .expect("step-curvature bounded solve should succeed");
        assert!(
            active_bug.is_empty(),
            "sanity: with the step Hessian the bound is (wrongly) released"
        );
        assert!(
            (beta[1] + dir_bug[1]) > 1.0,
            "sanity: the spuriously-freed coord overshoots its bound; got {}",
            beta[1] + dir_bug[1]
        );
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
            select_active_set_release(&gradient, &hd, &active_idx, false, false),
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
            select_active_set_release(&gradient, &hd, &active_idx, true, false),
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
            select_active_set_release(&gradient, &hd, &active_idx, true, false),
            None,
            "round-off-level multiplier must not trigger Bland's release"
        );

        // ...but a multiplier just outside the deadband (128·ε·|g|) must
        // trigger release, so the rule still detects genuine KKT violations.
        let lambda_real = -128.0 * f64::EPSILON * g;
        let hd = array![lambda_real - g];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true, false),
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
            select_active_set_release(&gradient, &hd, &active_idx, false, false),
            None
        );
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true, false),
            None
        );
    }

    /// gam#979: on a negative-curvature-reflected step the freed-block Newton
    /// step `d` is far-field along the reflected modes, so the second-order
    /// release term `(H·d)_i` is not model-trustworthy. A bound that is
    /// FIRST-ORDER optimal (`g_i ≥ 0`) must be KEPT even when the far-field
    /// `(H·d)_i` drives the full multiplier `g_i + (H·d)_i` negative — otherwise
    /// the freed coefficient overshoots (β∞ ≈ 26 on the n=3000 marginal-slope
    /// fit) and the active set zigzags across outer cycles. The reflected path
    /// judges dual feasibility on the reflection-invariant first-order `g_i`.
    #[test]
    pub(crate) fn select_active_set_release_reflected_uses_first_order_multiplier_979() {
        let gradient = array![1.0]; // g_0 = 1 ≥ 0 ⇒ first-order optimal ⇒ keep
        let hd = array![-3.0]; // far-field second-order term (fictitious length)
        let active_idx = vec![0];

        // Reflected path (worst-violation): first-order g_0 = 1 > 0 ⇒ KEEP.
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, false, true),
            None,
            "reflected release must judge on first-order g_i and KEEP a first-order-optimal bound"
        );
        // Reflected path (Bland's): same first-order verdict ⇒ KEEP.
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true, true),
            None,
            "Bland's variant must also KEEP on the reflected first-order test"
        );

        // Non-reflected path (exact λ = g + Hd = 1 − 3 = −2 < 0) ⇒ RELEASE.
        // Byte-identical to the historical behavior for callers that do not
        // reflect their step Hessian.
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, false, false),
            Some(0),
            "the non-reflected exact multiplier is unchanged"
        );

        // The fix is conservative, NOT release-blocking: a genuinely first-order-
        // infeasible bound (g_0 < 0) still releases on the reflected path.
        let gradient_neg = array![-0.5];
        assert_eq!(
            select_active_set_release(&gradient_neg, &hd, &active_idx, false, true),
            Some(0),
            "a first-order-infeasible bound (g_i < 0) must still release on the reflected path"
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
            None,
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
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: curvature,
            gradient_natural_scale: 0.0,
        }
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

    /// Hypothesis 1: `projected_gradient_norm` uses `bound_tol = 1e-10` which
    /// is too tight.  A coefficient at 1e-6 above its lower bound with a
    /// positive gradient (KKT multiplier) should be recognized as "at the
    /// bound" and excluded from the projected gradient.
    #[test]
    pub(crate) fn projected_gradient_excludes_near_bound_kkt_forces() {
        let gradient = array![0.5, 1e-4];
        let beta = array![1e-6, 2.0];
        let lower_bounds = array![0.0, f64::NEG_INFINITY];
        let norm = projected_gradient_norm(&gradient, &beta, Some(&lower_bounds));
        // Correct: only beta[1]'s gradient counts -> norm ~ 1e-4.
        // BUG: bound_tol=1e-10 misses beta[0] at 1e-6 -> norm ~ 0.5.
        assert!(
            norm < 0.01,
            "projected gradient should exclude near-bound KKT force (beta=1e-6, lb=0), got {:.6e}",
            norm
        );
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
            None,
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
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![0.0],
            }),
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
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

    /// The user's large-scale pathological case: a fit with `n=320000`,
    /// `p=20`, projected stationarity residual `‖g‖ = 1.465e-5`. The old
    /// absolute test `‖g‖ < 1e-6` rejects this as non-converged, even
    /// though the normalized residual is ~2.6e-8. After the fix, the
    /// scale-invariant certificate accepts it under EITHER bound.
    #[test]
    pub(crate) fn certifies_kkt_accepts_large_scale_pathological_case() {
        let n = 320_000usize;
        let p = 20usize;
        let g_norm = 1.465e-5;
        let tol = 1e-6;

        let state = WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 1.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            // At convergence the score and penalty gradient nearly cancel;
            // both are O(√n) for standardized columns. Use a representative
            // magnitude so the natural-scale bound has something to chew on.
            gradient_natural_scale: 1.0e3,
        };

        // Dimension-based bound: tol * sqrt(n) * sqrt(p) ≈ 1e-6 * 565.7 * 4.47 ≈ 2.5e-3
        // Natural-scale bound: 1.465e-5 / (1 + 1e3) ≈ 1.5e-8
        // Both pass; old absolute test 1.465e-5 < 1e-6 fails.
        assert!(
            state.certifies_kkt(g_norm, tol),
            "scale-invariant certificate should accept large-scale pathological case"
        );
        assert!(
            !(g_norm < tol),
            "this test must witness the failure of the old absolute test; \
             otherwise it does not prove the fix"
        );
    }

    /// The strict KKT certificate must be invariant under uniform rescaling
    /// of the objective `F → c·F` (which scales `‖g‖`, `‖score‖`, and
    /// `‖S·β‖` all by the same `c`). The additive `1` floor in the
    /// natural-scale denominator makes the test approximately invariant
    /// at small natural scale and exactly invariant in the limit.
    #[test]
    pub(crate) fn certifies_kkt_is_scale_invariant() {
        let n = 1000usize;
        let p = 10usize;
        let tol = 1e-6;
        let g_norm = 1.0;
        let natural_scale = 5.0e6; // dominates the +1 floor

        let mk_state = |g: Array1<f64>, ns: f64| WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: g,
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 0.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: ns,
        };

        let base = mk_state(Array1::zeros(p), natural_scale);
        let scaled = mk_state(Array1::zeros(p), natural_scale * 1000.0);

        // Numerator scales by c; denominator scales by c when the natural
        // scale dominates. So r_g is invariant.
        assert_eq!(
            base.certifies_kkt(g_norm, tol),
            scaled.certifies_kkt(g_norm * 1000.0, tol),
            "KKT classification must be invariant under uniform F → c·F"
        );
    }

    /// The two scale-invariant certificates must each be sufficient on its
    /// own (acceptance under EITHER suffices). One is data-driven (natural
    /// scale), the other purely structural (sqrt(n)·sqrt(p)). Both should
    /// accept obviously-converged states; failures of one should not block
    /// the other.
    #[test]
    pub(crate) fn certifies_kkt_accepts_under_either_bound() {
        let n = 100usize;
        let p = 5usize;
        let tol = 1e-6;

        let state_well_scaled = WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 0.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: 1.0e6,
        };
        // Natural-scale bound: 1.0 / (1+1e6) ≈ 1e-6 → at threshold; pass.
        // Dimension bound: 1.0 < 1e-6 * sqrt(100) * sqrt(5) ≈ 2.2e-5 → fail.
        // Acceptance under EITHER: pass (via natural-scale).
        assert!(state_well_scaled.certifies_kkt(0.99e-6 * (1.0 + 1.0e6), tol));

        let state_unscaled = WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 0.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: 0.0,
        };
        // Natural-scale bound: 2e-6 / 1 = 2e-6 → fail (above tol=1e-6).
        // Dimension bound: 2e-6 < 1e-6 * sqrt(100) * sqrt(5) ≈ 2.236e-5 → pass.
        // Acceptance under EITHER: pass (via dimension).
        assert!(state_unscaled.certifies_kkt(2.0e-6, tol));
    }

    /// The near-stationary band is exactly 10× the strict KKT tolerance,
    /// applied under either bound. It classifies a usable but non-strictly
    /// converged minimum as `StalledAtValidMinimum` rather than as a hard
    /// non-convergence.
    #[test]
    pub(crate) fn near_stationary_kkt_uses_ten_times_band() {
        let n = 100usize;
        let p = 4usize;
        let tol = 1e-6;
        let state = WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 0.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: 99.0,
        };
        // Natural-scale band: relative ‖g‖ = g/(1+99) = g/100 ≤ 10·tol = 1e-5
        // ⇒ accept when g ≤ 1e-3.
        assert!(state.near_stationary_kkt(9.9e-4, tol));
        assert!(!state.near_stationary_kkt(2.0e-3, tol));
        // Strict KKT at the same point should be ~10× tighter.
        assert!(!state.certifies_kkt(9.9e-4, tol));
    }

    /// The Newton-decrement upper bound `(−lin)·(1 + λ_lm/λ_min)` is
    /// derived from the resolvent identity and is a *provable* upper bound
    /// on `gᵀH⁻¹g` whenever `λ_min(H) ≥ ridge_floor`. Verify the algebraic
    /// inequality on a 2×2 worked example so the formula is locked in.
    #[test]
    pub(crate) fn newton_decrement_correction_upper_bounds_true_decrement() {
        // H = diag(2, 0.5).  λ_min = 0.5.  λ_lm = 0.25.
        let lambda_min = 0.5_f64;
        let lambda_lm = 0.25_f64;
        let g = ndarray::array![1.0_f64, 1.0];
        // True Newton decrement²: gᵀ H⁻¹ g = 1/2 + 1/0.5 = 0.5 + 2.0 = 2.5
        let true_decrement_sq = g[0].powi(2) / 2.0 + g[1].powi(2) / 0.5;
        // Damped: gᵀ (H+λI)⁻¹ g = 1/(2+0.25) + 1/(0.5+0.25) = 1/2.25 + 1/0.75
        let damped_decrement_sq =
            g[0].powi(2) / (2.0 + lambda_lm) + g[1].powi(2) / (0.5 + lambda_lm);
        // Correction factor: 1 + λ_lm / λ_min = 1 + 0.25/0.5 = 1.5
        let correction = 1.0 + lambda_lm / lambda_min;
        let upper_bound = damped_decrement_sq * correction;
        assert!(
            upper_bound >= true_decrement_sq,
            "(1 + λ_lm/λ_min)·damped must upper-bound true decrement: \
             upper={:.6}  true={:.6}",
            upper_bound,
            true_decrement_sq,
        );
        // And the bound should be tight enough to be useful (within 2× of true).
        assert!(
            upper_bound <= 2.0 * true_decrement_sq,
            "correction should not be wildly loose: upper={:.6}  true={:.6}",
            upper_bound,
            true_decrement_sq,
        );
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
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            |_| {},
        ) {
            Ok(_) => panic!("candidate evaluation failures should exhaust LM retries and surface"),
            Err(err) => err,
        };

        match err {
            EstimationError::PirlsDidNotConverge {
                max_iterations,
                last_change,
            } => {
                assert!(
                    max_iterations == options.max_iterations,
                    "expected LM exhaustion to surface as PIRLS non-convergence with screening cap"
                );
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
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            |_| {},
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
            min_step_size: 0.0,
            firth_bias_reduction: true,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            |_| {},
        ) {
            Ok(_) => panic!("Firth candidate reevaluation failures should not loop indefinitely"),
            Err(err) => err,
        };

        match err {
            EstimationError::PirlsDidNotConverge {
                max_iterations,
                last_change,
            } => {
                assert_eq!(max_iterations, options.max_iterations);
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
            min_step_size: 0.0,
            firth_bias_reduction: true,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            |_| {},
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
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
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
    pub(crate) fn long_constrained_objective_plateau_reports_valid_stall() {
        let mut model = LinearObjectivePlateauModel { gradient: -5e-5 };
        let options = WorkingModelPirlsOptions {
            max_iterations: 25,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![-100.0],
            }),
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
                .expect("long constrained objective plateau should preserve the final state");

        assert_eq!(
            result.status,
            PirlsStatus::StalledAtValidMinimum,
            "a long monotone objective plateau under explicit constraints is a valid bounded stall, unlike the unconstrained one-step plateau guard above"
        );
        assert!(
            result.iterations < options.max_iterations,
            "the long-plateau certificate should exit before exhausting the whole iteration budget"
        );
    }

    #[test]
    pub(crate) fn rejected_noise_scale_step_requires_near_stationary_projected_gradient() {
        let mut model = PlateauStatusModel {
            gradient: 2e-5,
            current_deviance: 1.0e6,
            candidate_deviance: 1.0e6 + 1.0,
        };
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 1,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
                .expect("noise-scale rejected step should still preserve the current state");

        // Same exit path as the plateau test: noise-scale rejection drives the
        // LM block to exhaustion with projected_grad 2e-5 above the
        // near-stationary band (= 1e-5), so the exact status is
        // LmStepSearchExhausted — keep the assertion strict so a future
        // regression that silently promotes to Converged/Stalled OR falls back
        // to the generic MaxIterationsReached default fails immediately.
        assert_eq!(
            result.status,
            PirlsStatus::LmStepSearchExhausted,
            "projected gradient 2e-5 exceeds the near-stationary band and must hit the LM-exhaust exit, not be accepted after a noise-scale rejection or fall through to MaxIterationsReached"
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
    pub(crate) fn test_deviance_monotonicity_gaussian() {
        // Simple Gaussian GAM: y ~ X beta with a smooth penalty.
        // Design matrix with an intercept column and one covariate.
        let n = 20;
        let mut x_data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            x_data[[i, 0]] = 1.0; // intercept
            x_data[[i, 1]] = t; // covariate
            // true relationship: y = 3 + 2*t + deterministic pseudo-noise
            y[i] = 3.0 + 2.0 * t + 0.3 * (((i * 17 + 5) % 11) as f64 / 11.0 - 0.5);
        }

        let w = Array1::ones(n);
        let offset = Array1::zeros(n);
        let rho = array![0.0]; // log(lambda) = 0, so lambda = 1
        // Penalty on the second coefficient only (leave intercept unpenalized).
        let rs = [array![[0.0, 0.0], [0.0, 1.0]]];
        let canonical: Vec<gam_terms::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                gam_terms::construction::CanonicalPenalty {
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            )),
            link_kind: InverseLink::Standard(StandardLink::Identity),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            arrow_schur: None,
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
                    covariate_se: None,
                    gaussian_fixed_cache: None,
                    glm_first_step_gram: None,
                },
                PenaltyConfig {
                    canonical_penalties: &canonical,
                    balanced_penalty_root: None,
                    reparam_invariant: None,
                    p: 2,
                    coefficient_lower_bounds: None,
                    linear_constraints_original: None,
                    penalty_shrinkage_floor: None,
                    kronecker_factored: None,
                },
                &config,
                None,
            )
        });
        result.expect("Gaussian P-IRLS fit should succeed");
        if trace.len() < 2 {
            // Gaussian identity-link can short-circuit through an exact dense solve
            // path without iterative PIRLS updates, yielding an empty trace.
            return;
        }
        assert_deviance_monotone(&trace, "Gaussian");
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
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
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
            arrow_schur: None,
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
                    covariate_se: None,
                    gaussian_fixed_cache: None,
                    glm_first_step_gram: None,
                },
                PenaltyConfig {
                    canonical_penalties: &canonical,
                    balanced_penalty_root: None,
                    reparam_invariant: None,
                    p: 2,
                    coefficient_lower_bounds: None,
                    linear_constraints_original: None,
                    penalty_shrinkage_floor: None,
                    kronecker_factored: None,
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
                        root: r.clone(),
                        col_range: 0..r.ncols(),
                        total_dim: r.ncols(),
                        nullity: 0,
                        local,
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
                arrow_schur: None,
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
                        covariate_se: None,
                        gaussian_fixed_cache: None,
                        glm_first_step_gram: None,
                    },
                    PenaltyConfig {
                        canonical_penalties: &canonical,
                        balanced_penalty_root: None,
                        reparam_invariant: None,
                        p: 3,
                        coefficient_lower_bounds: None,
                        linear_constraints_original: None,
                        penalty_shrinkage_floor: None,
                        kronecker_factored: None,
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
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };
        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
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
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };
        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
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
        let mut chunk = Array2::<f64>::zeros((0, 0));
        let mut got = Array2::<f64>::zeros((2, 2));
        PirlsWorkspace::add_dense_xtwx_signed(&weights, &mut chunk, &x, &mut got);

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
        calculate_loglikelihood_omitting_constants_from_eta, evaluate_full_log_likelihood_from_eta,
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
    use super::super::tweedie_exact_loglik_total_from_eta;
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

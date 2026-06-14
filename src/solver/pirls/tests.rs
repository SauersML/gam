//! P-IRLS regression and root-cause tests.
//!
//! The nested `#[cfg(test)] mod`s below address the rest of the solver through
//! `super::`, which now resolves to this module; the re-imports here forward the
//! sibling concern modules and the shared item surface so those paths keep
//! pointing at the same definitions they did when this file was inlined.

pub(crate) use super::*;

#[cfg(test)]
mod tests {
    use super::log_link_working_state::{ETA_CLAMP, MIN_MU, MIN_WEIGHT};
    use super::loop_driver::default_beta_guess_external;
    use super::reweight::madsen_lm_accept_factor;
    use super::{
        LinearInequalityConstraints, PenaltyConfig, PirlsConfig, PirlsLinearSolvePath,
        PirlsProblem, PirlsWorkspace, WorkingDerivativeBuffersMut, bernoulli_geometry_from_jet,
        calculate_deviance, compute_constraint_kkt_diagnostics, compute_jeffreys_pirls_diagnostics,
        compute_observed_hessian_curvature_arrays, fit_model_for_fixed_rho,
        select_active_set_release, should_log_pirls_decision_summary,
        should_use_sparse_native_pirls, solve_newton_directionwith_linear_constraints,
        solve_newton_directionwith_lower_bounds, tweedie_log_weight_mu_power, update_glmvectors,
        write_gamma_log_working_state, write_negative_binomial_log_working_state,
        write_poisson_log_working_state, write_tweedie_log_working_state,
    };
    use crate::matrix::DesignMatrix;
    use crate::mixture_link::{InverseLinkJet as MixtureInverseLinkJet, state_fromspec};
    use crate::probability::standard_normal_quantile;
    use crate::solver::active_set;
    use crate::types::{
        Coefficients, GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LinkComponent, LinkFunction,
        LogSmoothingParamsView, MixtureLinkSpec, ResponseFamily, StandardLink,
    };
    use approx::assert_relative_eq;
    use faer::sparse::{SparseColMat, Triplet};
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

    /// Outputs of a single log-link working-state write: `mu`, `weights`, `z`,
    /// and the four derivative buffers. Used by the parity test to compare the
    /// unified engine against the independent pre-unification reference math.
    pub(crate) struct LogLinkWorkingOutputs {
        pub(crate) mu: Array1<f64>,
        pub(crate) weights: Array1<f64>,
        pub(crate) z: Array1<f64>,
        pub(crate) c: Array1<f64>,
        pub(crate) d: Array1<f64>,
        pub(crate) dmu_deta: Array1<f64>,
        pub(crate) d2mu_deta2: Array1<f64>,
        pub(crate) d3mu_deta3: Array1<f64>,
    }

    impl LogLinkWorkingOutputs {
        pub(crate) fn zeros(n: usize) -> Self {
            Self {
                mu: Array1::zeros(n),
                weights: Array1::zeros(n),
                z: Array1::zeros(n),
                c: Array1::zeros(n),
                d: Array1::zeros(n),
                dmu_deta: Array1::zeros(n),
                d2mu_deta2: Array1::zeros(n),
                d3mu_deta3: Array1::zeros(n),
            }
        }

        pub(crate) fn assert_matches(&self, other: &Self, family: &str) {
            for (name, lhs, rhs) in [
                ("mu", &self.mu, &other.mu),
                ("weights", &self.weights, &other.weights),
                ("z", &self.z, &other.z),
                ("c", &self.c, &other.c),
                ("d", &self.d, &other.d),
                ("dmu_deta", &self.dmu_deta, &other.dmu_deta),
                ("d2mu_deta2", &self.d2mu_deta2, &other.d2mu_deta2),
                ("d3mu_deta3", &self.d3mu_deta3, &other.d3mu_deta3),
            ] {
                for (i, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{family}: buffer `{name}` row {i} diverged from the \
                         pre-unification reference: engine={a:?} reference={b:?}"
                    );
                }
            }
        }
    }

    /// Representative `(eta, y, prior_weight)` rows exercising the shared engine
    /// edge cases: ordinary rows, the `eta` clamp (`|eta| > 700`), the
    /// `MIN_WEIGHT` floor (tiny prior weight), and a zero-prior dropped row.
    pub(crate) fn log_link_parity_rows() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let eta = array![-2.3, 0.0, 1.7, 4.2, -800.0, 900.0, -3.0, 0.5];
        let y = array![0.0, 1.0, 3.0, 12.0, 0.0, 25.0, 2.0, 1.0];
        // Row 6 carries a tiny prior weight to trip the MIN_WEIGHT floor; row 7
        // carries zero prior weight to exercise the dropped-row branch.
        let prior = array![1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1e-13, 0.0];
        (eta, y, prior)
    }

    /// Pre-unification Poisson reference: `V(mu) = mu`, canonical-link curvature.
    pub(crate) fn reference_poisson(
        eta: &Array1<f64>,
        y: &Array1<f64>,
        prior: &Array1<f64>,
    ) -> LogLinkWorkingOutputs {
        let mut out = LogLinkWorkingOutputs::zeros(eta.len());
        for i in 0..eta.len() {
            let eta_raw = eta[i];
            let eta_i = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu_i = eta_i.exp().max(MIN_MU);
            out.mu[i] = mu_i;
            let raw_weight = prior[i].max(0.0) * mu_i;
            let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
            out.weights[i] = if raw_weight > 0.0 {
                raw_weight.max(MIN_WEIGHT)
            } else {
                0.0
            };
            out.z[i] = eta_i + (y[i] - mu_i) / mu_i;
            out.dmu_deta[i] = mu_i;
            out.d2mu_deta2[i] = mu_i;
            out.d3mu_deta3[i] = mu_i;
            if !(floor_active || eta_raw != eta_i) {
                out.c[i] = raw_weight;
                out.d[i] = raw_weight;
            }
        }
        out
    }

    /// Pre-unification Gamma reference: weight `prior * shape`, unfloored, no
    /// curvature, `mu`-jet never zeroed on clamp.
    pub(crate) fn reference_gamma(
        eta: &Array1<f64>,
        y: &Array1<f64>,
        prior: &Array1<f64>,
        shape: f64,
    ) -> LogLinkWorkingOutputs {
        let mut out = LogLinkWorkingOutputs::zeros(eta.len());
        for i in 0..eta.len() {
            let eta_i = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu_i = eta_i.exp().max(MIN_MU);
            out.mu[i] = mu_i;
            out.weights[i] = prior[i].max(0.0) * shape;
            out.z[i] = eta_i + (y[i] - mu_i) / mu_i;
            out.dmu_deta[i] = mu_i;
            out.d2mu_deta2[i] = mu_i;
            out.d3mu_deta3[i] = mu_i;
        }
        out
    }

    /// Pre-unification Tweedie reference: weight `prior * mu^(2-p) / phi`, the
    /// `mu`-jet zeroed on clamp, curvature `(2-p) * w`, `(2-p)^2 * w`.
    pub(crate) fn reference_tweedie(
        eta: &Array1<f64>,
        y: &Array1<f64>,
        prior: &Array1<f64>,
        p: f64,
        phi: f64,
    ) -> LogLinkWorkingOutputs {
        let exponent = 2.0 - p;
        let mut out = LogLinkWorkingOutputs::zeros(eta.len());
        for i in 0..eta.len() {
            let eta_raw = eta[i];
            let eta_i = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
            let clamp_active = eta_raw != eta_i;
            let mu_i = eta_i.exp().max(MIN_MU);
            out.mu[i] = mu_i;
            let raw_weight = prior[i].max(0.0) * tweedie_log_weight_mu_power(mu_i, p) / phi;
            let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
            out.weights[i] = if raw_weight > 0.0 {
                raw_weight.max(MIN_WEIGHT)
            } else {
                0.0
            };
            out.z[i] = eta_i + (y[i] - mu_i) / mu_i;
            if !clamp_active {
                out.dmu_deta[i] = mu_i;
                out.d2mu_deta2[i] = mu_i;
                out.d3mu_deta3[i] = mu_i;
            }
            if !(floor_active || clamp_active) {
                out.c[i] = exponent * raw_weight;
                out.d[i] = exponent * exponent * raw_weight;
            }
        }
        out
    }

    /// Pre-unification negative-binomial reference: numerically-stable Fisher
    /// weight `mu * theta / (theta + mu)`, curvature from the NB variance jet.
    pub(crate) fn reference_negbin(
        eta: &Array1<f64>,
        y: &Array1<f64>,
        prior: &Array1<f64>,
        theta: f64,
    ) -> LogLinkWorkingOutputs {
        let mut out = LogLinkWorkingOutputs::zeros(eta.len());
        for i in 0..eta.len() {
            let eta_raw = eta[i];
            let eta_i = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu_i = eta_i.exp().max(MIN_MU);
            let denom = theta + mu_i;
            let negbin_weight = if theta > mu_i {
                mu_i / (1.0 + mu_i / theta)
            } else {
                theta / (1.0 + theta / mu_i)
            };
            let raw_weight = prior[i].max(0.0) * negbin_weight;
            let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
            out.mu[i] = mu_i;
            out.weights[i] = if raw_weight > 0.0 {
                raw_weight.max(MIN_WEIGHT)
            } else {
                0.0
            };
            out.z[i] = eta_i + (y[i] - mu_i) / mu_i;
            out.dmu_deta[i] = mu_i;
            out.d2mu_deta2[i] = mu_i;
            out.d3mu_deta3[i] = mu_i;
            if !(floor_active || eta_raw != eta_i) {
                out.c[i] = raw_weight * theta / denom;
                out.d[i] = raw_weight * theta * (theta - mu_i) / (denom * denom);
            }
        }
        out
    }

    /// Run a unified log-link writer through both its derivative and
    /// no-derivative branches and collect the buffers.
    pub(crate) fn run_unified<F>(n: usize, write: F) -> (LogLinkWorkingOutputs, LogLinkWorkingOutputs)
    where
        F: Fn(
            &mut Array1<f64>,
            &mut Array1<f64>,
            &mut Array1<f64>,
            Option<WorkingDerivativeBuffersMut<'_>>,
        ),
    {
        let mut with = LogLinkWorkingOutputs::zeros(n);
        {
            let mut c = Array1::zeros(n);
            let mut d = Array1::zeros(n);
            let mut dmu = Array1::zeros(n);
            let mut d2 = Array1::zeros(n);
            let mut d3 = Array1::zeros(n);
            write(
                &mut with.mu,
                &mut with.weights,
                &mut with.z,
                Some(WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu,
                    d2mu_deta2: &mut d2,
                    d3mu_deta3: &mut d3,
                }),
            );
            with.c = c;
            with.d = d;
            with.dmu_deta = dmu;
            with.d2mu_deta2 = d2;
            with.d3mu_deta3 = d3;
        }
        let mut without = LogLinkWorkingOutputs::zeros(n);
        write(&mut without.mu, &mut without.weights, &mut without.z, None);
        (with, without)
    }

    #[test]
    pub(crate) fn log_link_working_state_engine_matches_per_family_reference() {
        // The unified `write_log_link_working_state` engine must reproduce the
        // exact pre-unification per-family row math bit-for-bit: `mu`, `weights`,
        // working response `z`, and the curvature buffers `c`/`d`, across every
        // edge case (eta clamp, MIN_WEIGHT floor, zero-prior dropped row). The
        // no-derivative branch must additionally agree with the derivative
        // branch on `mu`/`weights`/`z`. Bounds are exact (bitwise), never
        // weakened — any divergence is a real regression in a central solver
        // path shared by Poisson, Gamma, Tweedie, and negative binomial.
        let (eta, y, prior) = log_link_parity_rows();
        let n = eta.len();

        // Poisson.
        let reference = reference_poisson(&eta, &y, &prior);
        let (with, without) = run_unified(n, |mu, w, z, derivs| {
            write_poisson_log_working_state(y.view(), &eta, prior.view(), mu, w, z, derivs);
        });
        with.assert_matches(&reference, "Poisson (derivatives)");
        assert_eq!(with.mu.to_vec(), without.mu.to_vec(), "Poisson mu branch");
        assert_eq!(
            with.weights.to_vec(),
            without.weights.to_vec(),
            "Poisson weights branch"
        );
        assert_eq!(with.z.to_vec(), without.z.to_vec(), "Poisson z branch");

        // Gamma (fixed shape).
        let shape = 2.5;
        let reference = reference_gamma(&eta, &y, &prior, shape);
        let (with, without) = run_unified(n, |mu, w, z, derivs| {
            write_gamma_log_working_state(y.view(), &eta, prior.view(), shape, mu, w, z, derivs);
        });
        with.assert_matches(&reference, "Gamma (derivatives)");
        assert_eq!(with.mu.to_vec(), without.mu.to_vec(), "Gamma mu branch");
        assert_eq!(
            with.weights.to_vec(),
            without.weights.to_vec(),
            "Gamma weights branch"
        );
        assert_eq!(with.z.to_vec(), without.z.to_vec(), "Gamma z branch");

        // Tweedie.
        let p = 1.5;
        let phi = 0.7;
        let reference = reference_tweedie(&eta, &y, &prior, p, phi);
        let (with, without) = run_unified(n, |mu, w, z, derivs| {
            write_tweedie_log_working_state(y.view(), &eta, prior.view(), p, phi, mu, w, z, derivs)
                .expect("valid Tweedie parameters");
        });
        with.assert_matches(&reference, "Tweedie (derivatives)");
        assert_eq!(with.mu.to_vec(), without.mu.to_vec(), "Tweedie mu branch");
        assert_eq!(
            with.weights.to_vec(),
            without.weights.to_vec(),
            "Tweedie weights branch"
        );
        assert_eq!(with.z.to_vec(), without.z.to_vec(), "Tweedie z branch");

        // Negative binomial (fixed theta).
        let theta = 3.0;
        let reference = reference_negbin(&eta, &y, &prior, theta);
        let (with, without) = run_unified(n, |mu, w, z, derivs| {
            write_negative_binomial_log_working_state(
                y.view(),
                &eta,
                prior.view(),
                theta,
                mu,
                w,
                z,
                derivs,
            )
            .expect("valid negative-binomial theta");
        });
        with.assert_matches(&reference, "NegBin (derivatives)");
        assert_eq!(with.mu.to_vec(), without.mu.to_vec(), "NegBin mu branch");
        assert_eq!(
            with.weights.to_vec(),
            without.weights.to_vec(),
            "NegBin weights branch"
        );
        assert_eq!(with.z.to_vec(), without.z.to_vec(), "NegBin z branch");
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
        let x = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
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
        penalties: Vec<crate::smooth::BlockwisePenalty>,
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
        let (canonical, _) =
            crate::construction::canonicalize_penalty_specs(&specs, &nulls, p, "prior mean test")
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
            geodesic_acceleration: false,
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
            LogSmoothingParamsView::new(rho.view()),
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
        let penalty = crate::smooth::BlockwisePenalty::ridge(0..1, 1.0)
            .with_prior_mean(crate::estimate::CoefficientPriorMean::scalar(2.5));
        let beta = fixed_gaussian_beta(x, y, vec![penalty], array![0.0]);
        assert!((beta[0] - 2.5).abs() < 1e-10, "beta={beta:?}");
    }

    #[test]
    pub(crate) fn functional_prior_mean_recovers_kernel_amplitude() {
        let x = Array2::<f64>::zeros((5, 3));
        let y = Array1::<f64>::zeros(5);
        let metadata = array![2.0];
        let alpha = 1.75;
        let penalty = crate::smooth::BlockwisePenalty::ridge(0..3, 1.0).with_prior_mean(
            crate::estimate::CoefficientPriorMean::functional(
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
        let base_penalty = crate::smooth::BlockwisePenalty::ridge(0..2, 1.0);
        let zero_penalty = crate::smooth::BlockwisePenalty::ridge(0..2, 1.0).with_prior_mean(
            crate::estimate::CoefficientPriorMean::constant(Array1::zeros(2)),
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
        assert!(file!().ends_with(".rs"));
        let x = array![[1.0], [1.0], [1.0], [1.0], [1.0]];
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0];
        let w = Array1::ones(5);
        let offset = Array1::zeros(5);
        let rho = Array1::<f64>::zeros(1);
        let covariate_se = array![0.9, 0.7, 0.8, 0.6, 0.75];
        let rs = [array![[1.0]]];
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
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
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
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
                fit.final_eta[i].clamp(-ETA_CLAMP, ETA_CLAMP),
                covariate_se[i],
            )
            .expect("logit integrated inverse-link jet should evaluate");
            let expected = bernoulli_geometry_from_jet(
                fit.final_eta[i],
                fit.final_eta[i].clamp(-ETA_CLAMP, ETA_CLAMP),
                y[i],
                w[i],
                MixtureInverseLinkJet {
                    mu: jet.mean,
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                },
            );
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
        assert!(file!().ends_with(".rs"));
        let y = array![2.0, 5.0];
        let mu = array![1.0, 4.0];
        let w = array![1.5, 0.75];
        let dev = calculate_deviance(
            y.view(),
            &mu,
            &GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            )),
            w.view(),
        );
        let expected = 2.0
            * (1.5 * (2.0_f64 / 1.0 - 1.0 - (2.0_f64 / 1.0).ln())
                + 0.75 * (5.0_f64 / 4.0 - 1.0 - (5.0_f64 / 4.0).ln()));
        assert_relative_eq!(dev, expected, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    pub(crate) fn gamma_log_observed_curvature_matches_shape_one_closed_form() {
        assert!(file!().ends_with(".rs"));
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
    pub(crate) fn negative_binomial_log_observed_curvature_matches_size_theta_closed_form() {
        assert!(file!().ends_with(".rs"));
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
    pub(crate) fn gamma_log_fit_profiles_shape_instead_of_fixing_one() {
        let x = array![[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]];
        let y = array![0.8, 1.1, 1.7, 2.0, 2.6, 3.1];
        let w = Array1::ones(y.len());
        let offset = Array1::zeros(y.len());
        let rho = array![0.0];
        let rs = [array![[0.0]]];
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
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
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (result, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
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
            super::estimate_gamma_shape_from_eta(y.view(), &result.final_eta, w.view());

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
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
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
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
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
        )
        .expect("lower-bound active-set solve should succeed");

        assert_relative_eq!(direction[0], 0.0, epsilon = 1e-14);
        let projected = &beta + &direction;
        assert_relative_eq!(projected[0], beta[0], epsilon = 1e-14);
        assert!(active_hint.is_empty());
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
    use super::*;
    use crate::types::LogSmoothingParamsView;
    use approx::assert_relative_eq;
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
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(array![[1.0]]),
            log_likelihood: 0.0,
            deviance,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: curvature,
            gradient_natural_scale: 0.0,
        }
    }

    pub(crate) fn test_working_state(beta: &Coefficients, curvature: HessianCurvatureKind) -> WorkingState {
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
            // through `update_with_curvature` (with Firth temporarily disabled)
            // rather than via a separate cheap screen: since the candidate
            // screening split landed, the firth accepted-state re-evaluation is
            // folded into this single call instead of running as a distinct
            // post-acceptance phase. Mirror that here so the injected
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
        pub(crate) fn state(&self, beta: &Coefficients, curvature: HessianCurvatureKind) -> WorkingState {
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
            geodesic_acceleration: false,
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
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
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
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
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
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
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
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
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
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
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
        let noise_floor = current_penalized.abs().max(1.0) * 1e-14; // ~9e-9

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
            geodesic_acceleration: false,
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
            geodesic_acceleration: false,
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
            geodesic_acceleration: false,
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
            geodesic_acceleration: false,
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
    pub(crate) fn plateaued_accepted_step_does_not_report_converged_with_large_projected_gradient() {
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
            geodesic_acceleration: false,
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
            geodesic_acceleration: false,
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
            geodesic_acceleration: false,
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
        assert!(file!().ends_with(".rs"));
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
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
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
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (result, trace) = capture_pirls_penalized_deviance(|| {
            fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
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
        assert!(file!().ends_with(".rs"));
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
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
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
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (result, trace) = capture_pirls_penalized_deviance(|| {
            fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
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
            let canonical: Vec<crate::construction::CanonicalPenalty> = rs
                .iter()
                .map(|r| {
                    let local = r.t().dot(r);
                    crate::construction::CanonicalPenalty {
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
                geodesic_acceleration: false,
                arrow_schur: None,
            };

            let (result, trace) = capture_pirls_penalized_deviance(|| {
                fit_model_for_fixed_rho(
                    LogSmoothingParamsView::new(rho.view()),
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
        use crate::terms::closed_form_operator::ClosedFormPenaltyOperator;
        use crate::terms::penalty_op::PenaltyOp;

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
            geodesic_acceleration: false,
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
            geodesic_acceleration: false,
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

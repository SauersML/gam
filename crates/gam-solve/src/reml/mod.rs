use self::inner_strategy::GeometryBackendKind;
use super::*;
use crate::pirls::assemble_and_factor_sparse_penalized_system;
use gam_linalg::sparse_exact::SparseExactFactor;
use gam_problem::OuterEval;
use gam_problem::SasLinkState;
use gam_terms::basis::LocalDesignJacobianProvider;
use ndarray::{Array1, Array2, s};
use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::{Arc, RwLock};

pub mod assembly;
pub mod atoms;
pub(crate) mod continuation;
pub(crate) mod eval;
mod firth;
#[cfg(test)]
mod gaussian_sufficient_statistics_tests;
#[cfg(test)]
mod glm_outer_hessian_fd_tests;
pub(super) mod hyper;
mod inner_strategy;
// #1521 carve: promoted `pub(crate)` -> `pub` so the extracted
// `gam-custom-family` crate reaches the Jeffreys-subspace items it consumes.
pub mod jeffreys_subspace;
pub(crate) mod laml_logdet;
pub mod outer_eval;
pub mod penalty_logdet;
pub mod per_atom_efs;
pub mod reml_outer_engine;
pub mod reparameterized_inner;
mod rho_key;
mod sparse_exact_penalty;

pub(crate) use sparse_exact_penalty::sparse_penalty_block_count_from_canonical;

pub(crate) const PERSISTENT_LATENT_VALUES_CACHE_CAPACITY: usize = 8;

#[derive(Debug)]
pub(crate) struct PersistentLatentValuesCache {
    pub(crate) entries: HashMap<String, Array2<f64>>,
    pub(crate) lru: VecDeque<String>,
    pub(crate) capacity: usize,
}

impl Default for PersistentLatentValuesCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            capacity: PERSISTENT_LATENT_VALUES_CACHE_CAPACITY,
        }
    }
}

impl PersistentLatentValuesCache {
    pub(crate) fn lookup(
        &mut self,
        key: &str,
        n_obs: usize,
        latent_dim: usize,
    ) -> Option<Array2<f64>> {
        let values = self.entries.get(key)?;
        if values.dim() != (n_obs, latent_dim) {
            return None;
        }
        let values = values.clone();
        self.touch(key.to_string());
        Some(values)
    }

    pub(crate) fn insert(&mut self, key: String, values: Array2<f64>) {
        if values.iter().any(|value| !value.is_finite()) {
            return;
        }
        self.entries.insert(key.clone(), values);
        self.touch(key);
        while self.entries.len() > self.capacity {
            let Some(evicted) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&evicted);
        }
    }

    pub(crate) fn touch(&mut self, key: String) {
        if let Some(index) = self.lru.iter().position(|queued| queued == &key) {
            self.lru.remove(index);
        }
        self.lru.push_back(key);
    }
}

/// Cached state from the most recent successful PIRLS solve, populated by
/// `updatewarm_start_from` and consumed by the IFT-based warm-start
/// predictor (`RemlState::predict_warm_start_beta_ift_with_outcome`).
/// See the field doc on `RemlState::ift_warm_start_cache` for the math.
#[derive(Clone)]
pub(crate) struct IftWarmStartCache {
    /// β at the converged solve, in ORIGINAL basis. Mirror of
    /// `warm_start_beta` stashed alongside the H factor for atomic
    /// consistency under concurrent reads (the predictor needs both
    /// β and H from the SAME solve; reading them from two locks risks
    /// a torn pair if a fresh solve lands between reads).
    pub beta_original: ndarray::Array1<f64>,
    /// ρ at which the solve occurred. Mirror of `warm_start_rho`,
    /// stashed for the same atomic-consistency reason.
    pub rho: ndarray::Array1<f64>,
    /// Penalized Hessian H_pen at the converged β, in TRANSFORMED basis.
    /// The IFT predictor factors this on demand; basis transforms run in
    /// transformed basis for numerical stability.
    pub penalized_hessian_transformed: gam_linalg::matrix::SymmetricMatrix,
    /// Reparameterization matrix qs converting between transformed
    /// (column) basis and original basis: `β_orig = qs · β_tfd`,
    /// `H_orig = qs · H_tfd · qs^T`.
    pub qs: ndarray::Array2<f64>,
    /// True when the PIRLS result was already in original basis
    /// (`OriginalSparseNative`) — in which case `qs` is the identity
    /// and the IFT predictor can skip the basis-conversion ops.
    pub frame_was_original: bool,
    /// Per-penalty precomputation `S_k · β_cur[cp.col_range]`,
    /// indexed in lockstep with `RemlObjectiveState::canonical_penalties`.
    /// Each entry is the local-block mat-vec the IFT predictor would
    /// otherwise recompute on every predict call. With H_pen factor
    /// caching (commit ec18559d) the per-call cost dropped from
    /// `O(p³)` Cholesky to `O(p²) ≈ k · O(block²)` rhs construction;
    /// at large-scale CTN (p ≈ several thousand) that's several ms
    /// per predict call still being paid. By stashing `S_k · β_cur`
    /// at cache-write time the predictor's per-call work drops to
    /// just the `Δρ_k · e^{ρ_k} · sb_block` accumulation, which is
    /// `O(p)` rather than `O(p²)`.
    ///
    /// `None` when the cache predates this commit's writer hook (e.g.,
    /// transient state during invalidation); the predictor falls back
    /// to recomputing the mat-vec when this is `None` or the length
    /// mismatches `canonical_penalties.len()`.
    pub lambda_s_beta_blocks: Option<Vec<ndarray::Array1<f64>>>,
}

#[cfg(test)]
mod tests {
    use super::atoms::CriterionAtom;
    use super::{
        DirectionalHyperParam, EvalCacheManager, EvalShared, HyperDesignDerivative,
        HyperPenaltyDerivative, ImplicitDerivLevel, RemlConfig, RemlState,
    };
    use crate::estimate::EstimationError;
    use crate::pirls::PirlsCoordinateFrame;
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;
    use gam_linalg::matrix::symmetrize_in_place;
    use gam_problem::{
        GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink,
    };
    use gam_problem::{HessianValue, OuterEval};
    use gam_terms::basis::ImplicitDesignPsiDerivative;
    use ndarray::{Array1, Array2, array, s};
    use std::sync::Arc;

    /// Shorthand for the canonical Binomial-Logit `GlmLikelihoodSpec` used by
    /// the REML test fixtures.
    pub(crate) fn binomial_logit_glm_spec() -> GlmLikelihoodSpec {
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ))
    }

    /// Shorthand for the canonical Gaussian-Identity `GlmLikelihoodSpec` used
    /// by the REML test fixtures.
    pub(crate) fn gaussian_identity_glm_spec() -> GlmLikelihoodSpec {
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ))
    }

    impl DirectionalHyperParam {
        pub(super) fn new(
            x_tau_original: Array2<f64>,
            penalty_first_components: Vec<(usize, Array2<f64>)>,
            x_tau_tau_original: Option<Vec<Option<Array2<f64>>>>,
            penaltysecond_components: Option<Vec<Option<Vec<(usize, Array2<f64>)>>>>,
        ) -> Result<Self, EstimationError> {
            let x_tau_tau_original = x_tau_tau_original.map(|rows| {
                rows.into_iter()
                    .map(|entry| entry.map(HyperDesignDerivative::from))
                    .collect::<Vec<_>>()
            });
            let penalty_first_components = penalty_first_components
                .into_iter()
                .map(|(idx, matrix)| (idx, HyperPenaltyDerivative::from(matrix)))
                .collect();
            let penaltysecond_components = penaltysecond_components.map(|rows| {
                rows.into_iter()
                    .map(|row| {
                        row.map(|components| {
                            components
                                .into_iter()
                                .map(|(idx, matrix)| (idx, HyperPenaltyDerivative::from(matrix)))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>()
            });
            Self::new_compact(
                HyperDesignDerivative::from(x_tau_original),
                penalty_first_components,
                x_tau_tau_original,
                penaltysecond_components,
            )
        }

        pub(super) fn single_penalty(
            penalty_index: usize,
            x_tau_original: Array2<f64>,
            s_tau_original: Array2<f64>,
            x_tau_tau_original: Option<Vec<Option<Array2<f64>>>>,
            s_tau_tau_original: Option<Vec<Option<Array2<f64>>>>,
        ) -> Result<Self, EstimationError> {
            let penaltysecond_components = s_tau_tau_original.map(|rows| {
                rows.into_iter()
                    .map(|mat| mat.map(|mat| vec![(penalty_index, mat)]))
                    .collect::<Vec<_>>()
            });
            Self::new(
                x_tau_original,
                vec![(penalty_index, s_tau_original)],
                x_tau_tau_original,
                penaltysecond_components,
            )
        }
    }

    /// Common shape for the design-motion + penalty-motion REML test fixtures
    /// (Gaussian-identity and binomial-logit at present): both carry the
    /// same `(y, w, X, S0, cfg, ρ)` plus a perturbation pair, and need the
    /// same three helpers (`state`, `state_perturbed`, `fd_directional_gradient`).
    /// Per-fixture `new()` constructors fill the fields with family-specific
    /// data; the helpers below are shared via default impls so every fixture
    /// pays the boilerplate once.
    trait LogitDesignMotionFixture {
        fn y(&self) -> &Array1<f64>;
        fn w(&self) -> &Array1<f64>;
        fn x(&self) -> &Array2<f64>;
        fn s0(&self) -> &Array2<f64>;
        fn cfg(&self) -> &RemlConfig;
        fn rho(&self) -> &Array1<f64>;

        fn state(&self) -> RemlState<'_> {
            build_logit_state(self.y(), self.w(), self.x(), self.s0(), self.cfg())
        }

        fn state_perturbed(
            &self,
            x_tau: &Array2<f64>,
            s_tau: &Array2<f64>,
            eps: f64,
        ) -> (RemlState<'_>, RemlState<'_>) {
            let x_plus = self.x() + &x_tau.mapv(|v| eps * v);
            let x_minus = self.x() - &x_tau.mapv(|v| eps * v);
            let s_plus = self.s0() + &s_tau.mapv(|v| eps * v);
            let s_minus = self.s0() - &s_tau.mapv(|v| eps * v);
            (
                build_logit_state(self.y(), self.w(), &x_plus, &s_plus, self.cfg()),
                build_logit_state(self.y(), self.w(), &x_minus, &s_minus, self.cfg()),
            )
        }

        /// Central FD approximation to the directional cost derivative at ρ.
        fn fd_directional_gradient(&self, x_tau: &Array2<f64>, s_tau: &Array2<f64>) -> f64 {
            let h = 2e-5;
            let (state_plus, state_minus) = self.state_perturbed(x_tau, s_tau, h);
            let v_plus = state_plus.compute_cost(self.rho()).expect("cost+");
            let v_minus = state_minus.compute_cost(self.rho()).expect("cost-");
            (v_plus - v_minus) / (2.0 * h)
        }
    }

    pub(crate) fn build_logit_state<'a>(
        y: &'a Array1<f64>,
        w: &'a Array1<f64>,
        x: &Array2<f64>,
        s: &Array2<f64>,
        cfg: &'a RemlConfig,
    ) -> RemlState<'a> {
        use crate::estimate::PenaltySpec;
        let p = x.ncols();
        let offset = Array1::<f64>::zeros(y.len());
        let spec = PenaltySpec::Dense(s.clone());
        let canonical =
            gam_terms::construction::canonicalize_penalty_specs(&[spec], &[1], p, "test")
                .map(|(canonical, _)| canonical)
                .expect("canonicalize");
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
        .expect("state")
    }

    fn bundle_with_inner_kkt_residual(bundle: &EvalShared, residual: Array1<f64>) -> EvalShared {
        let mut pirls_result = bundle.pirls_result.as_ref().clone();
        pirls_result.lastgradient_norm = residual.dot(&residual).sqrt();
        pirls_result.penalized_gradient_transformed = residual;
        let mut cloned = bundle.clone();
        cloned.pirls_result = Arc::new(pirls_result);
        cloned
    }

    fn evaluate_synthetic_psi_value_without_inner_kkt(
        state: &RemlState<'_>,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> f64 {
        let mode = super::reml_outer_engine::EvalMode::ValueOnly;
        let mut assembly = state
            .build_auto_assembly(rho, bundle, mode, true, false)
            .expect("uncorrected synthetic-psi assembly");
        assert!(
            matches!(
                &assembly.dispersion,
                super::reml_outer_engine::DispersionHandling::Fixed { .. }
            ),
            "the #2305 bridge fixture must exercise the fixed-dispersion LAML identity"
        );
        let p_dim = assembly.beta.len();
        assembly.ext_coords = vec![super::reml_outer_engine::HyperCoord {
            a: 0.0,
            g: Array1::zeros(p_dim),
            drift: super::reml_outer_engine::HyperCoordDrift::none(),
            ld_s: 0.0,
            b_depends_on_beta: false,
            is_penalty_like: false,
            firth_g: None,
            tk_eta_fixed: None,
            tk_x_fixed: None,
        }];
        state
            .assemble_and_evaluate(rho, bundle, mode, assembly)
            .expect("uncorrected synthetic-psi value")
            .cost
    }

    #[test]
    fn psi_value_bridge_corrects_nonstationary_inner_mode_2305() {
        // The residual correction implemented by the unified evaluator is the
        // fixed-dispersion LAML identity. Gaussian-identity uses profiled-scale
        // REML and deliberately does not enter that gate, so use a genuine
        // fixed-dispersion design-moving model here.
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.4],
            [1.0, -0.2, 0.7],
            [1.0, 0.3, -0.5],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, 0.6],
        ];
        let s = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.15], [0.0, 0.15, 0.8]];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-12, false);
        let state = build_logit_state(&y, &w, &x, &s, &cfg);
        let rho = array![0.2];
        let base = state.obtain_eval_bundle(&rho).expect("base inner mode");
        let residual = array![0.18, -0.11, 0.07];
        let capped = bundle_with_inner_kkt_residual(&base, residual);

        let corrected = state
            .evaluate_unified_value_only_with_synthetic_ext_count(&rho, &capped, 1, false)
            .expect("psi value with inner-KKT correction");
        let exact_kkt_assumption =
            evaluate_synthetic_psi_value_without_inner_kkt(&state, &rho, &capped);
        let residual_energy = corrected
            .ift_residual_energy
            .expect("design-moving bridge must attach the nonstationary KKT residual");

        assert!(
            residual_energy.abs() > 1e-10,
            "the synthetic nonstationary residual must produce a material fixed-dispersion \
             IFT correction, got residual_energy={residual_energy:.12e}"
        );
        assert_eq!(
            corrected.cost,
            exact_kkt_assumption - residual_energy,
            "the psi value bridge must apply exactly the correction reported by the unified \
             evaluator: corrected={:.12e}, exact-kkt={exact_kkt_assumption:.12e}, \
             residual-energy={residual_energy:.12e}",
            corrected.cost,
        );
    }

    #[test]
    fn psi_value_bridge_correction_vanishes_at_stationarity_2305() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.6, -0.4],
            [1.0, -0.2, 0.7],
            [1.0, 0.3, -0.5],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, 0.6],
        ];
        let s = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.15], [0.0, 0.15, 0.8]];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-12, false);
        let state = build_logit_state(&y, &w, &x, &s, &cfg);
        let rho = array![0.2];
        let base = state.obtain_eval_bundle(&rho).expect("base inner mode");
        let stationary = bundle_with_inner_kkt_residual(&base, Array1::zeros(x.ncols()));

        let corrected = state
            .evaluate_unified_value_only_with_synthetic_ext_count(&rho, &stationary, 1, false)
            .expect("stationary psi value with correction enabled");
        let exact_kkt_assumption =
            evaluate_synthetic_psi_value_without_inner_kkt(&state, &rho, &stationary);

        assert_eq!(
            corrected.ift_residual_energy,
            Some(0.0),
            "the design-moving bridge must attach the residual, whose correction is exactly \
             zero at stationarity"
        );
        assert_eq!(
            corrected.cost, exact_kkt_assumption,
            "the generic inner-KKT correction must be exactly zero at a stationary inner mode"
        );
    }

    #[test]
    fn repeated_penalty_ranges_keep_analytic_outer_hessian() {
        let y = array![0.2, -0.1, 0.3, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![[1.0, -0.7], [1.0, -0.2], [1.0, 0.3], [1.0, 0.9]];
        let offset = Array1::<f64>::zeros(y.len());
        let cfg = RemlConfig::external(gaussian_identity_glm_spec(), 1e-10, false);
        let p = x.ncols();
        let canonical = vec![
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[0.0, 1.0]], p),
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[1.0, 0.0]], p),
        ];
        let state = RemlState::newwith_offset(
            y.view(),
            x,
            w.view(),
            offset.view(),
            canonical,
            p,
            &cfg,
            Some(vec![1, 1]),
            None,
            None,
        )
        .expect("state");

        assert!(
            state.analytic_outer_hessian_enabled(),
            "double-penalty-style repeated coefficient ranges must still route to exact Hessian"
        );
    }

    /// The IFT trust radius belongs to the `RemlState` that measured it, not
    /// to the ADDRESS that state happened to occupy.
    ///
    /// Its predecessor (a quality-history step-cap controller) lived in a
    /// process-global `HashMap` keyed by `self as *const _ as usize`. Every
    /// constructing caller makes the state a stack local, so two fits reached
    /// from the same call site land on the same frame slot — and with no
    /// `Drop` to evict the entry, the second fit read the first fit's history.
    /// That is a silent cross-fit dependence: the same data fitted twice in
    /// one process takes a different IFT step schedule the second time.
    ///
    /// The probe builds a state, measures a radius and stages a step on it,
    /// drops it, and builds a second state in the same slot. The second state
    /// must be unmeasured: no radius and no staged step.
    #[test]
    fn adaptive_ift_controller_does_not_survive_into_the_next_state_in_the_same_slot() {
        use super::outer_eval::WarmStartPredictionSource;

        fn probe(record: bool) -> (usize, Option<f64>, bool) {
            let y = array![0.2, -0.1, 0.3, 0.0];
            let w = Array1::<f64>::ones(y.len());
            let x = array![[1.0, -0.7], [1.0, -0.2], [1.0, 0.3], [1.0, 0.9]];
            let offset = Array1::<f64>::zeros(y.len());
            let cfg = RemlConfig::external(gaussian_identity_glm_spec(), 1e-10, false);
            let p = x.ncols();
            let canonical = vec![gam_terms::construction::CanonicalPenalty::from_dense_root(
                array![[0.0, 1.0]],
                p,
            )];
            let state = RemlState::newwith_offset(
                y.view(),
                x,
                w.view(),
                offset.view(),
                canonical,
                p,
                &cfg,
                Some(vec![1]),
                None,
                None,
            )
            .expect("state");
            let address = &state as *const RemlState<'_> as usize;
            if record {
                // A unit step whose prediction missed by 10 against a flat
                // miss of 1: the measured radius is 1·1/10.
                let radius = state.record_warm_start_prediction_error(
                    WarmStartPredictionSource::Ift,
                    1.0,
                    10.0,
                    1.0,
                );
                assert_eq!(radius, Some(0.1));
                state.stage_warm_start_prediction_step(WarmStartPredictionSource::Ift, 0.05);
                return (address, radius, true);
            }
            let radius = state.warm_start_trust_radius(WarmStartPredictionSource::Ift);
            let staged = state.take_staged_warm_start_prediction_step().is_some();
            (address, radius, staged)
        }

        let (first_address, _, _) = probe(true);
        let (second_address, radius, staged) = probe(false);

        assert_eq!(
            first_address, second_address,
            "the probe only exercises the defect when the second state reuses the first's slot; \
             both calls are to the same fn at the same depth, so the frame offsets must coincide"
        );
        assert_eq!(
            radius, None,
            "a freshly built state must be unmeasured, not carry the radius the previous state \
             at this address measured"
        );
        assert!(
            !staged,
            "a freshly built state must not inherit the previous state's staged prediction step"
        );
    }

    /// The warm-start trust radius is derived from one measurement, not
    /// tuned: at step s the predictor's miss is E ≈ c₂s² and the flat seed's
    /// miss is F ≈ c₁s, so the step at which they tie is R = s·F/E. A
    /// predictor that hit exactly has no finite radius; a solve where both
    /// seeds were exact, or where the inputs are not finite, measures
    /// nothing; and the flat seed has no radius to measure (gam#2902).
    #[test]
    fn warm_start_trust_radius_is_the_step_where_the_predictor_ties_the_flat_seed() {
        use super::outer_eval::WarmStartPredictionSource::{Flat, Ift, TangentLine};

        let y = array![0.2, -0.1, 0.3, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![[1.0, -0.7], [1.0, -0.2], [1.0, 0.3], [1.0, 0.9]];
        let offset = Array1::<f64>::zeros(y.len());
        let cfg = RemlConfig::external(gaussian_identity_glm_spec(), 1e-10, false);
        let p = x.ncols();
        let canonical = vec![gam_terms::construction::CanonicalPenalty::from_dense_root(
            array![[0.0, 1.0]],
            p,
        )];
        let state = RemlState::newwith_offset(
            y.view(),
            x,
            w.view(),
            offset.view(),
            canonical,
            p,
            &cfg,
            Some(vec![1]),
            None,
            None,
        )
        .expect("state");

        assert_eq!(state.warm_start_trust_radius(Ift), None);
        assert_eq!(state.record_warm_start_prediction_error(Ift, 1.0, 0.5, 1.0), Some(2.0));
        assert_eq!(state.warm_start_trust_radius(Ift), Some(2.0));
        // The tangent radius is its own measurement.
        assert_eq!(state.warm_start_trust_radius(TangentLine), None);
        assert_eq!(
            state.record_warm_start_prediction_error(TangentLine, 3.0, 2.0, 0.5),
            Some(0.75)
        );
        assert_eq!(state.warm_start_trust_radius(Ift), Some(2.0));
        // An exact prediction leaves no step at which the flat seed wins.
        assert_eq!(
            state.record_warm_start_prediction_error(Ift, 0.5, 0.0, 1.0),
            Some(f64::INFINITY)
        );
        // Both seeds exact, or a non-finite input, measure nothing and keep
        // the previous radius.
        assert_eq!(state.record_warm_start_prediction_error(Ift, 0.5, 0.0, 0.0), None);
        assert_eq!(state.record_warm_start_prediction_error(Ift, f64::NAN, 1.0, 1.0), None);
        assert_eq!(state.record_warm_start_prediction_error(Ift, 1.0, -1.0, 1.0), None);
        assert_eq!(state.warm_start_trust_radius(Ift), Some(f64::INFINITY));
        assert_eq!(state.record_warm_start_prediction_error(Flat, 1.0, 0.5, 1.0), None);
        assert_eq!(state.warm_start_trust_radius(Flat), None);
        state.clear_warm_start_adaptive_signals();
        assert_eq!(state.warm_start_trust_radius(Ift), None);
        assert_eq!(state.warm_start_trust_radius(TangentLine), None);
    }

    #[test]
    fn canonical_logit_firth_keeps_exact_tk_hessian_beyond_row_pair_scale() {
        // n²·p = 1.12e8: ten times the row-pair budget that used to send this
        // fit's outer curvature to BFGS. The exact TK Hessian has a design-tensor
        // route now (#2900), so the planner keeps it.
        let n = 2_000usize;
        let p = 28usize;
        let y = Array1::from_iter((0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }));
        let w = Array1::<f64>::ones(n);
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            x[[i, 0]] = 1.0;
            for j in 1..p {
                x[[i, j]] = ((j as f64) * std::f64::consts::TAU * t).sin()
                    + 0.25 * (((j + 1) as f64) * std::f64::consts::TAU * t).cos();
            }
        }
        let mut s = Array2::<f64>::zeros((p, p));
        for j in 1..p {
            s[[j, j]] = 1.0;
        }
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-10, true);
        let state = build_logit_state(&y, &w, &x, &s, &cfg);

        assert!(
            state.analytic_outer_hessian_enabled(),
            "canonical-logit Firth fits keep exact TK Hessian curvature at row-pair scale"
        );
    }

    #[test]
    fn canonical_logit_firth_keeps_exact_tk_hessian_for_small_separation_guards() {
        let n = 40usize;
        let p = 6usize;
        let y = Array1::from_iter((0..n).map(|i| if i >= n / 2 { 1.0 } else { 0.0 }));
        let w = Array1::<f64>::ones(n);
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64) / (n - 1) as f64;
            x[[i, 0]] = 1.0;
            for j in 1..p {
                x[[i, j]] = t.powi(j as i32);
            }
        }
        let mut s = Array2::<f64>::zeros((p, p));
        for j in 1..p {
            s[[j, j]] = 1.0;
        }
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-10, true);
        let state = build_logit_state(&y, &w, &x, &s, &cfg);

        assert!(
            state.analytic_outer_hessian_enabled(),
            "small Firth rescue fits should keep exact TK Hessian curvature"
        );
    }

    #[test]
    fn nonlogit_firth_keeps_tk_value_and_gradient() {
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.0, 0.3],
            [1.0, -0.7, -0.2],
            [1.0, -0.3, 0.4],
            [1.0, 0.0, -0.5],
            [1.0, 0.2, 0.6],
            [1.0, 0.6, -0.4],
            [1.0, 0.9, 0.2],
            [1.0, 1.3, -0.1],
        ];
        let s = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.1], [0.0, 0.1, 0.7]];
        let rho = array![0.15];

        for link in [StandardLink::Probit, StandardLink::CLogLog] {
            let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(link),
            ));
            let cfg = RemlConfig::external(likelihood, 1e-9, true).with_max_iterations(500);
            let state = build_logit_state(&y, &w, &x, &s, &cfg);
            assert!(
                state.analytic_outer_hessian_enabled(),
                "{link:?} Firth must carry its analytic TK outer Hessian (#3203)"
            );

            let bundle = state
                .obtain_eval_bundle(&rho)
                .expect("non-logit Firth bundle");
            let atom = state
                .tierney_kadane_terms(
                    &rho,
                    &bundle,
                    super::reml_outer_engine::EvalMode::ValueAndGradient,
                    &[],
                )
                .expect("non-logit TK correction");
            let value = CriterionAtom::value(&atom);
            let gradient = atom.gradient().expect("TK gradient");
            assert!(
                value.is_finite() && value.abs() > 1e-12,
                "{link:?} must receive a material finite TK correction, got {value}"
            );
            assert_eq!(gradient.len(), rho.len());
            assert!(
                gradient.iter().all(|entry| entry.is_finite()),
                "{link:?} TK gradient must be finite: {gradient:?}"
            );
        }
    }

    pub(crate) fn poisson_log_glm_spec() -> GlmLikelihoodSpec {
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ))
    }

    /// Regression (issue #893): for a fixed-dispersion family a uniform prior
    /// weight `w = c` is *exact* `c`-fold row replication. The two encodings must
    /// therefore present a byte-identical LAML smoothing-selection surface — both
    /// the cost `V(ρ)` and its gradient `∇V(ρ)` — because every term (penalised
    /// deviance `D_p`, the working cross-product `XᵀWX`, the log-determinants)
    /// is a sum of per-observation contributions that is identical whether a row
    /// carries weight `c` or is stacked `c` times. This locks the *surface*
    /// invariant that #893 ultimately reduces to: when the surfaces coincide,
    /// the only remaining requirement for `λ̂(w=c) = λ̂(c×)` is that the outer
    /// optimiser resolve the shared optimum (handled by the tightened outer
    /// tolerance in `workflow.rs`). A regression that reintroduced a
    /// row-count-vs-weight-sum asymmetry into the inner solve or the cost would
    /// break this directly, independent of the optimiser tolerance.
    #[test]
    pub(crate) fn fixed_dispersion_laml_surface_is_replication_invariant() {
        let n = 200usize;
        let p = 8usize;
        let c = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            let tau = std::f64::consts::TAU;
            x[[i, 0]] = 1.0;
            x[[i, 1]] = t;
            x[[i, 2]] = (tau * t).sin();
            x[[i, 3]] = (tau * t).cos();
            x[[i, 4]] = (2.0 * tau * t).sin();
            x[[i, 5]] = (2.0 * tau * t).cos();
            x[[i, 6]] = (3.0 * tau * t).sin();
            x[[i, 7]] = (3.0 * tau * t).cos();
            let eta = 0.3 + 0.9 * (1.4 * (t - 0.5)).sin();
            // Deterministic non-negative integer counts near exp(eta).
            y[i] = (eta.exp() + 0.5 * ((i as f64) * 2.399_963).sin())
                .round()
                .max(0.0);
        }
        let mut s = Array2::<f64>::zeros((p, p));
        for j in 1..p {
            s[[j, j]] = 1.0;
        }

        // Replicated design (c literal copies of each row).
        let mut x_rep = Array2::<f64>::zeros((n * c, p));
        let mut y_rep = Array1::<f64>::zeros(n * c);
        for r in 0..c {
            for i in 0..n {
                let row = r * n + i;
                for j in 0..p {
                    x_rep[[row, j]] = x[[i, j]];
                }
                y_rep[row] = y[i];
            }
        }

        let w_weighted = Array1::<f64>::from_elem(n, c as f64);
        let w_rep = Array1::<f64>::ones(n * c);

        let cfg = RemlConfig::external(poisson_log_glm_spec(), 1e-10, false);
        let st_w = build_logit_state(&y, &w_weighted, &x, &s, &cfg);
        let st_r = build_logit_state(&y_rep, &w_rep, &x_rep, &s, &cfg);

        for &rho in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0] {
            let r = Array1::from_elem(1, rho);
            let cw = st_w.compute_cost(&r).expect("weighted cost");
            let cr = st_r.compute_cost(&r).expect("replicated cost");
            let gw = st_w.compute_gradient(&r).expect("weighted grad");
            let gr = st_r.compute_gradient(&r).expect("replicated grad");
            // Costs and gradients must coincide to optimiser precision; the only
            // admissible difference is f64 summation order over n vs c·n rows.
            assert!(
                (cw - cr).abs() <= 1e-9 * (1.0 + cw.abs()),
                "LAML cost differs between w=c and c× replication at rho={rho}: \
                 cost_w={cw:.12e} cost_r={cr:.12e} diff={:.3e}",
                cw - cr
            );
            assert!(
                (gw[0] - gr[0]).abs() <= 1e-9 * (1.0 + gw[0].abs()),
                "LAML gradient differs between w=c and c× replication at rho={rho}: \
                 g_w={:.12e} g_r={:.12e} diff={:.3e}",
                gw[0],
                gr[0],
                gw[0] - gr[0]
            );
        }
    }

    /// Regression (issue #893): the geometric-mean log-weight ρ-anchor
    /// ([`RemlState::rho_weight_anchor`]) is a *profiled*-dispersion construct
    /// (issue #877). For a fixed-dispersion family the optimum does not slide by
    /// `log c` under a weight rescale in a way the prior should track, and a
    /// nonzero anchor would evaluate the regularising ρ-prior at *different*
    /// coordinates for the `w=c` vs `c×` encodings — breaking the very
    /// equivalence #893 requires. The anchor must therefore be exactly `0` for a
    /// fixed-dispersion family and the geometric mean for Gaussian-identity.
    #[test]
    pub(crate) fn rho_weight_anchor_is_zero_for_fixed_dispersion() {
        let n = 50usize;
        let p = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            x[[i, 0]] = 1.0;
            x[[i, 1]] = t;
            x[[i, 2]] = t * t;
            y[i] = (1.0 + (3.0 * t).sin()).round().max(0.0);
        }
        let mut s = Array2::<f64>::zeros((p, p));
        s[[2, 2]] = 1.0;
        // All weights = c: geometric-mean log-weight = ln(c) ≠ 0.
        let c = 4.0_f64;
        let w = Array1::<f64>::from_elem(n, c);

        let cfg_pois = RemlConfig::external(poisson_log_glm_spec(), 1e-10, false);
        let st_pois = build_logit_state(&y, &w, &x, &s, &cfg_pois);
        assert_eq!(
            st_pois.rho_weight_anchor(),
            0.0,
            "fixed-dispersion (Poisson) anchor must be 0, not the geometric-mean log-weight"
        );

        let cfg_gauss = RemlConfig::external(gaussian_identity_glm_spec(), 1e-10, false);
        let st_gauss = build_logit_state(&y, &w, &x, &s, &cfg_gauss);
        assert!(
            (st_gauss.rho_weight_anchor() - c.ln()).abs() <= 1e-12,
            "Gaussian-identity (profiled) anchor must be the geometric-mean log-weight ln(c)={:.6}, got {:.6}",
            c.ln(),
            st_gauss.rho_weight_anchor()
        );
    }

    pub(crate) fn beta_original_from_bundle(bundle: &EvalShared) -> Array1<f64> {
        let pr = bundle.pirls_result.as_ref();
        match pr.coordinate_frame {
            PirlsCoordinateFrame::OriginalSparseNative => pr.beta_transformed.as_ref().clone(),
            PirlsCoordinateFrame::TransformedQs => {
                pr.reparam_result.qs.dot(pr.beta_transformed.as_ref())
            }
        }
    }

    pub(crate) fn compute_joint_hypercostgradienthessian(
        state: &RemlState<'_>,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError> {
        let (cost, gradient, hessian) = state.compute_joint_hyper_eval_with_order(
            theta,
            rho_dim,
            hyper_dirs,
            crate::rho_optimizer::OuterEvalOrder::ValueGradientHessian,
        )?;
        Ok((
            cost,
            gradient,
            hessian
                .materialize_dense()
                .map_err(|error| EstimationError::RemlOptimizationFailed(error.to_string()))?
                .ok_or_else(|| {
                    EstimationError::RemlOptimizationFailed(
                        "joint hyper Hessian requested but unavailable".to_string(),
                    )
                })?,
        ))
    }

    pub(crate) fn h_original_from_bundle(bundle: &EvalShared) -> Array2<f64> {
        let pr = bundle.pirls_result.as_ref();
        match pr.coordinate_frame {
            PirlsCoordinateFrame::OriginalSparseNative => bundle.h_total.as_ref().clone(),
            PirlsCoordinateFrame::TransformedQs => {
                let qs = &pr.reparam_result.qs;
                let tmp = gam_linalg::faer_ndarray::fast_ab(qs, bundle.h_total.as_ref());
                gam_linalg::faer_ndarray::fast_abt(&tmp, qs)
            }
        }
    }

    pub(crate) fn single_directional_tau_gradient(
        state: &RemlState<'_>,
        rho: &Array1<f64>,
        hyper: DirectionalHyperParam,
    ) -> Result<f64, EstimationError> {
        let mut theta = Array1::<f64>::zeros(rho.len() + 1);
        theta.slice_mut(s![..rho.len()]).assign(rho);
        let (_, gradient, _) = state.compute_joint_hyper_eval_with_order(
            &theta,
            rho.len(),
            &[hyper],
            crate::rho_optimizer::OuterEvalOrder::ValueAndGradient,
        )?;
        Ok(gradient[rho.len()])
    }

    pub(crate) fn fd_directional_tau_cost_gradient(
        y: &Array1<f64>,
        w: &Array1<f64>,
        x: &Array2<f64>,
        s0: &Array2<f64>,
        cfg: &RemlConfig,
        rho: &Array1<f64>,
        x_tau: &Array2<f64>,
        s_tau: &Array2<f64>,
    ) -> f64 {
        let h = 2e-5;
        let x_plus = x + &x_tau.mapv(|v| h * v);
        let x_minus = x - &x_tau.mapv(|v| h * v);
        let s_plus = s0 + &s_tau.mapv(|v| h * v);
        let s_minus = s0 - &s_tau.mapv(|v| h * v);
        let state_plus = build_logit_state(y, w, &x_plus, &s_plus, cfg);
        let state_minus = build_logit_state(y, w, &x_minus, &s_minus, cfg);
        let v_plus = state_plus.compute_cost(rho).expect("cost+");
        let v_minus = state_minus.compute_cost(rho).expect("cost-");
        (v_plus - v_minus) / (2.0 * h)
    }

    pub(crate) fn directional_tau_hessian_fd_reference(
        y: &Array1<f64>,
        w: &Array1<f64>,
        x: &Array2<f64>,
        s0: &Array2<f64>,
        cfg: &RemlConfig,
        rho: &Array1<f64>,
        hyper_dirs: &[DirectionalHyperParam],
        x_tau_mats: &[Array2<f64>],
        s_tau_mats: &[Array2<f64>],
    ) -> Array2<f64> {
        assert_eq!(hyper_dirs.len(), x_tau_mats.len());
        assert_eq!(hyper_dirs.len(), s_tau_mats.len());

        const TARGET_PHYSICAL_STEP: f64 = 1e-5;

        let n_dirs = hyper_dirs.len();
        let mut h_ttfd = Array2::<f64>::zeros((n_dirs, n_dirs));
        for j in 0..n_dirs {
            let direction_scale = x_tau_mats[j]
                .iter()
                .chain(s_tau_mats[j].iter())
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            let h = if direction_scale > 0.0 {
                TARGET_PHYSICAL_STEP / direction_scale
            } else {
                TARGET_PHYSICAL_STEP
            };

            let x_plus = x + &x_tau_mats[j].mapv(|v| h * v);
            let x_minus = x - &x_tau_mats[j].mapv(|v| h * v);
            let s_plus = s0 + &s_tau_mats[j].mapv(|v| h * v);
            let s_minus = s0 - &s_tau_mats[j].mapv(|v| h * v);

            let state_plus = build_logit_state(y, w, &x_plus, &s_plus, cfg);
            let state_minus = build_logit_state(y, w, &x_minus, &s_minus, cfg);
            for i in 0..n_dirs {
                let g_plus =
                    single_directional_tau_gradient(&state_plus, rho, hyper_dirs[i].clone())
                        .expect("g+ for FD");
                let g_minus =
                    single_directional_tau_gradient(&state_minus, rho, hyper_dirs[i].clone())
                        .expect("g- for FD");
                h_ttfd[[i, j]] = (g_plus - g_minus) / (2.0 * h);
            }
        }
        symmetrize_in_place(&mut h_ttfd);
        h_ttfd
    }

    #[test]
    pub(crate) fn eval_cache_manager_stores_first_order_outer_eval() {
        let cache = EvalCacheManager::new(0);
        let rho = array![0.25, -0.0];
        let rho_key = super::rho_key::sanitized_rhokey(&rho);
        let eval = OuterEval {
            cost: 3.5,
            gradient: array![1.0, -2.0],
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        };

        cache.store_outer_eval(&rho_key, &eval);

        let cached = cache
            .cached_outer_eval(&rho_key)
            .expect("first-order outer eval should be cached");
        assert_eq!(cached.cost, eval.cost);
        assert_eq!(cached.gradient, eval.gradient);
        assert!(matches!(cached.hessian, HessianValue::Unavailable));

        cache.invalidate_eval_bundle();
        assert!(
            cache.cached_outer_eval(&rho_key).is_none(),
            "invalidating the bundle should clear the outer-eval cache too"
        );
    }

    /// #1575 multi-slot outer-eval cache correctness oracle.
    ///
    /// A memoization is only safe if a hit returns *exactly* what the miss path
    /// stored. This pins three properties of the bounded LRU:
    ///   1. round-trip fidelity — a hit is `f64::to_bits`-identical in cost AND
    ///      every gradient component to the value stored on the miss path;
    ///   2. no aliasing — distinct rho-keys never return each other's eval;
    ///   3. honest eviction — once an evicted key is requested again it MISSES
    ///      (so the caller recomputes) rather than returning a stale neighbour.
    #[test]
    pub(crate) fn outer_eval_lru_hit_is_bit_identical_and_evicts_honestly_1575() {
        use super::OUTER_EVAL_LRU_CAPACITY;

        // Helper: a deterministic OuterEval whose bits encode `seed`, so any
        // cross-key contamination is detectable bit-for-bit.
        let make_eval = |seed: f64| OuterEval {
            cost: (seed * std::f64::consts::PI).sin() / 3.0 - seed,
            gradient: array![seed, -seed * 2.0, seed.recip()],
            hessian: HessianValue::Unavailable,
            inner_beta_hint: Some(array![seed + 0.5, seed - 0.5]),
        };
        let bits_eq = |a: &OuterEval, b: &OuterEval| -> bool {
            a.cost.to_bits() == b.cost.to_bits()
                && a.gradient.len() == b.gradient.len()
                && a.gradient
                    .iter()
                    .zip(b.gradient.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
        };

        let cache = EvalCacheManager::new(0);

        // (1) Round-trip fidelity: store at rho_a, then a forced hit must equal
        // the stored eval bit-for-bit (the "hit == miss" guarantee).
        let rho_a = array![0.25, -1.5];
        let key_a = super::rho_key::sanitized_rhokey(&rho_a);
        let eval_a = make_eval(0.25);
        cache.store_outer_eval(&key_a, &eval_a);
        let hit_a = cache
            .cached_outer_eval(&key_a)
            .expect("stored rho_a must hit");
        assert!(
            bits_eq(&hit_a, &eval_a),
            "cache hit must be bit-identical (cost+gradient) to the stored miss-path eval"
        );
        assert_eq!(
            hit_a.inner_beta_hint.as_ref().map(|b| b.to_vec()),
            eval_a.inner_beta_hint.as_ref().map(|b| b.to_vec()),
            "inner_beta_hint must round-trip unchanged"
        );

        // (2) No aliasing: a second, distinct rho must return its OWN eval, and
        // the first key must still return the first eval untouched.
        let rho_b = array![0.25, -1.4999999999999998];
        let key_b = super::rho_key::sanitized_rhokey(&rho_b);
        assert_ne!(key_a, key_b, "the two rho-keys must differ");
        let eval_b = make_eval(7.0);
        cache.store_outer_eval(&key_b, &eval_b);
        assert!(
            bits_eq(
                &cache.cached_outer_eval(&key_b).expect("rho_b must hit"),
                &eval_b
            ),
            "rho_b must return its own eval, not rho_a's"
        );
        assert!(
            bits_eq(
                &cache
                    .cached_outer_eval(&key_a)
                    .expect("rho_a must still hit"),
                &eval_a
            ),
            "rho_a must be unaffected by the rho_b insert"
        );

        // (3) Honest eviction: overflow the LRU with fresh keys. The
        // least-recently-used entry must be evicted and then MISS (forcing a
        // recompute), while a still-resident key returns its exact stored bits.
        let cache = EvalCacheManager::new(0);
        let mut keys = Vec::new();
        let mut evals = Vec::new();
        for i in 0..OUTER_EVAL_LRU_CAPACITY {
            let rho = array![i as f64, -(i as f64)];
            let key = super::rho_key::sanitized_rhokey(&rho);
            let eval = make_eval(i as f64 + 0.123);
            cache.store_outer_eval(&key, &eval);
            keys.push(key);
            evals.push(eval);
        }
        // Cache is exactly full; key[0] is the least-recently-used.
        assert_eq!(
            cache
                .outer_eval_lru
                .read()
                .expect("outer-eval LRU lock is poisoned: a writer panicked while holding it")
                .entries
                .len(),
            OUTER_EVAL_LRU_CAPACITY
        );
        // One more distinct key evicts the LRU (key[0]).
        let rho_overflow = array![999.0, -999.0];
        let key_overflow = super::rho_key::sanitized_rhokey(&rho_overflow);
        let eval_overflow = make_eval(42.0);
        cache.store_outer_eval(&key_overflow, &eval_overflow);
        assert_eq!(
            cache
                .outer_eval_lru
                .read()
                .expect("outer-eval LRU lock is poisoned: a writer panicked while holding it")
                .entries
                .len(),
            OUTER_EVAL_LRU_CAPACITY,
            "capacity must stay bounded"
        );
        assert!(
            cache.cached_outer_eval(&keys[0]).is_none(),
            "the least-recently-used key must be evicted and now MISS (recompute), not return stale"
        );
        assert!(
            bits_eq(
                &cache
                    .cached_outer_eval(&keys[1])
                    .expect("a still-resident key must hit"),
                &evals[1]
            ),
            "a still-resident key must return its exact stored bits"
        );
        assert!(
            bits_eq(
                &cache
                    .cached_outer_eval(&key_overflow)
                    .expect("the freshest key must hit"),
                &eval_overflow
            ),
            "the freshest key must hit with its own eval"
        );
    }

    #[test]
    pub(crate) fn reset_outer_seed_state_clears_pirls_cache() {
        // Build a minimal logit RemlState, populate the cross-call PIRLS LRU
        // by evaluating the outer objective at one rho, then verify that
        // reset_outer_seed_state wipes that LRU (alongside the eval bundle
        // and warm-start signals). This pins down the cross-attempt
        // cleanup contract that a budget-bump retry relies on.
        let y = array![0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.5, -0.4],
            [1.0, 0.0, 0.7],
            [1.0, 0.4, -0.3],
            [1.0, 0.9, 0.1],
            [1.0, 1.3, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.1, 0.15], [0.0, 0.15, 0.8],];
        let rho = array![0.0];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-10, false);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);

        // Trigger a full outer eval so execute_pirls_if_needed inserts at
        // least one entry into the cross-call PIRLS LRU.
        state
            .compute_outer_eval_with_order(
                &rho,
                crate::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            )
            .expect("outer eval should succeed");

        let populated_len = state
            .cache_manager
            .pirls_cache
            .read()
            .expect("PIRLS cache lock is poisoned: a writer panicked while holding it")
            .map
            .len();
        assert!(
            populated_len > 0,
            "evaluating the outer objective should populate the PIRLS LRU, got {populated_len}"
        );

        state.reset_outer_seed_state();

        let cleared_len = state
            .cache_manager
            .pirls_cache
            .read()
            .expect("PIRLS cache lock is poisoned: a writer panicked while holding it")
            .map
            .len();
        assert_eq!(
            cleared_len, 0,
            "reset_outer_seed_state must clear the cross-call PIRLS LRU; got {cleared_len} entries"
        );
    }

    fn gaussian_reset_fixture() -> (Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>) {
        let y = array![0.3, -0.2, 0.9, 0.4, -0.7, 1.1, 0.2, -0.4];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.5, -0.4],
            [1.0, 0.0, 0.7],
            [1.0, 0.4, -0.3],
            [1.0, 0.9, 0.1],
            [1.0, 1.3, -0.6],
            [1.0, -0.8, 0.5],
            [1.0, 0.6, 0.9],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.1, 0.15], [0.0, 0.15, 0.8],];
        (y, w, x, s0)
    }

    #[test]
    pub(crate) fn gaussian_reset_keeps_the_seed_independent_inner_solve() {
        // A Gaussian-identity inner mode is one direct solve, so the solve an
        // outer reset used to discard is the one the next evaluation rebuilt.
        // The reset keeps it: re-evaluating the same ρ runs no new solve, and
        // returns exactly what a state that never saw another ρ returns.
        let (y, w, x, s0) = gaussian_reset_fixture();
        let cfg = RemlConfig::external(gaussian_identity_glm_spec(), 1e-10, false);
        let order = crate::rho_optimizer::OuterEvalOrder::ValueAndGradient;
        let (rho_seed, rho) = (array![-1.5], array![0.7]);
        let solves = |state: &RemlState<'_>| {
            state
                .arena
                .inner_pirls_solve_count
                .load(std::sync::atomic::Ordering::Relaxed)
        };

        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state
            .compute_outer_eval_with_order(&rho_seed, order)
            .expect("seed eval");
        state.compute_outer_eval_with_order(&rho, order).expect("eval");
        let before_reset = solves(&state);
        state.reset_outer_seed_state();
        let reused = state
            .compute_outer_eval_with_order(&rho, order)
            .expect("eval after reset");
        assert_eq!(
            solves(&state),
            before_reset,
            "the reset must keep the Gaussian inner solve at ρ, not rebuild it"
        );

        let fresh_state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let fresh = fresh_state
            .compute_outer_eval_with_order(&rho, order)
            .expect("fresh eval");
        assert_eq!(reused.cost.to_bits(), fresh.cost.to_bits());
        assert_eq!(
            reused.gradient.mapv(f64::to_bits),
            fresh.gradient.mapv(f64::to_bits)
        );
    }

    #[test]
    pub(crate) fn gaussian_block_correction_decision_needs_no_evaluation() {
        // Laplace is exact for the Gaussian-identity model, so the #784
        // admission deferred to the certified optimum declines there without
        // an evaluation, and the optimum's inner solve is left in place.
        let (y, w, x, s0) = gaussian_reset_fixture();
        let cfg = RemlConfig::external(gaussian_identity_glm_spec(), 1e-10, false);
        let rho = array![0.7];
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.defer_block_correction_admission();
        state.compute_cost_and_gradient(&rho).expect("eval");
        let solves = || {
            state
                .arena
                .inner_pirls_solve_count
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        let before = solves();
        assert!(
            !state
                .decide_block_correction_admission(&rho, None)
                .expect("decision"),
            "the correction never applies to a Gaussian-identity fit"
        );
        assert_eq!(solves(), before, "the decision must not run an inner solve");
        assert!(!state.block_correction_admission_deferred());
        state.compute_cost_and_gradient(&rho).expect("eval");
        assert_eq!(
            solves(),
            before,
            "the optimum's inner solve must survive the decision"
        );
    }

    #[test]
    pub(crate) fn reset_outer_seed_state_preserves_frozen_negbin_theta_1448() {
        // #1448 regression: the NB outer θ↔λ alternation loop
        // (solver/estimate/optimizer.rs) re-runs the ρ search after each θ
        // refresh by (a) re-freezing the λ-search θ at θ_final into
        // `frozen_negbin_theta`, then (b) calling `reset_outer_seed_state()` to
        // drop the caches keyed to the old θ. Step (b) MUST NOT clear the freeze
        // set in step (a): the capture in `solve_for_unified_rho` only writes the
        // frozen slot when it is 0, so if the reset zeroed it the next round would
        // re-derive θ from the seed η and the loop would never reach the (ρ, θ)
        // joint fixed point — silently regressing #1448 back to a single
        // freeze→refresh pass.
        //
        // This pins the load-bearing distinction between `reset_outer_seed_state`
        // (alternation-round reset, freeze SURVIVES) and the surface-refresh reset
        // (new design, freeze re-zeroed). The end-to-end convergence on a real NB
        // fit is exercised by the public-API path; here we lock the invariant the
        // loop depends on, next to `reset_outer_seed_state_clears_pirls_cache`.
        use std::sync::atomic::Ordering;

        let y = array![0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.5, -0.4],
            [1.0, 0.0, 0.7],
            [1.0, 0.4, -0.3],
            [1.0, 0.9, 0.1],
            [1.0, 1.3, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.1, 0.15], [0.0, 0.15, 0.8],];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-10, false);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);

        // Simulate the alternation loop's re-freeze step: pin θ_final.
        let theta_final_bits = 2.5_f64.to_bits();
        state
            .frozen_negbin_theta
            .store(theta_final_bits, Ordering::Relaxed);
        assert_eq!(
            state.frozen_negbin_theta.load(Ordering::Relaxed),
            theta_final_bits,
            "precondition: the re-freeze stores θ_final into the frozen slot"
        );

        // The alternation loop's per-round reset.
        state.reset_outer_seed_state();

        assert_eq!(
            state.frozen_negbin_theta.load(Ordering::Relaxed),
            theta_final_bits,
            "reset_outer_seed_state (alternation-round reset) must PRESERVE the \
             re-frozen NB θ; clearing it would defeat the #1448 θ↔λ alternation \
             (the next ρ search would re-derive θ from the seed and never reach \
             the joint fixed point)"
        );
    }

    #[test]
    pub(crate) fn implicit_hyper_design_derivative_respects_full_model_embedding() {
        let operator = ImplicitDesignPsiDerivative::new(
            array![1.0, 2.0, 3.0, 4.0],
            array![0.5, -1.0, 1.5, 2.0],
            array![0.1, 0.2, 0.3, 0.4],
            array![[1.0, 0.2], [0.5, 0.1], [1.5, 0.3], [2.0, 0.4]],
            None,
            None,
            2,
            2,
            1,
            2,
        );
        let local = operator
            .materialize_first(0)
            .expect("materialized first derivative");
        assert_eq!(
            local.ncols(),
            3,
            "operator-local derivative should stay smooth-local"
        );

        let implicit = HyperDesignDerivative::from_implicit(
            Arc::new(operator),
            ImplicitDerivLevel::First(0),
            1..4,
            5,
        );
        let embedded = HyperDesignDerivative::from_embedded(local.clone(), 1..4, 5);

        assert_eq!(implicit.nrows(), embedded.nrows());
        assert_eq!(implicit.ncols(), 5);
        assert_eq!(implicit.materialize(), embedded.materialize());

        let u = array![7.0, 1.5, -2.0, 0.25, -3.0];
        let v = array![0.75, -1.25];
        assert_eq!(
            implicit.forward_mul_original(&u).expect("implicit forward"),
            embedded.forward_mul_original(&u).expect("embedded forward")
        );
        assert_eq!(
            implicit
                .transpose_mul_original(&v)
                .expect("implicit transpose"),
            embedded
                .transpose_mul_original(&v)
                .expect("embedded transpose")
        );

        let qs = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ];
        assert_eq!(
            implicit
                .transformed(&qs, None)
                .expect("implicit transformed"),
            embedded
                .transformed(&qs, None)
                .expect("embedded transformed")
        );
        let u_transformed = array![1.0, -0.5, 2.0];
        assert_eq!(
            implicit
                .transformed_forward_mul(&qs, None, &u_transformed)
                .expect("implicit transformed forward"),
            embedded
                .transformed_forward_mul(&qs, None, &u_transformed)
                .expect("embedded transformed forward")
        );
        assert_eq!(
            implicit
                .transformed_transpose_mul(&qs, None, &v)
                .expect("implicit transformed transpose"),
            embedded
                .transformed_transpose_mul(&qs, None, &v)
                .expect("embedded transformed transpose")
        );
    }

    #[test]
    pub(crate) fn directional_hyper_identities_match_finite_differences_logit() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9],];

        // Use one directional hyperparameter τ with a penalty perturbation:
        // S(τ) = S + τ S_τ.
        // Keep X_τ = 0 so this identity test remains valid in both non-Firth
        // and Firth-logit modes.
        let x_tau = Array2::<f64>::zeros(x.raw_dim());
        let s_tau = array![[0.0, 0.0, 0.0], [0.0, 0.25, 0.04], [0.0, 0.04, 0.15],];
        let hyper =
            DirectionalHyperParam::single_penalty(0, x_tau.clone(), s_tau.clone(), None, None)
                .expect("single-penalty hyper direction");
        let rho = array![0.0];

        // Tight inner tolerance: the envelope theorem requires an exact inner
        // P-IRLS optimum; 1e-10 leaves enough residual gradient to cause ~12%
        // V_tau mismatch on this small (n=8) logistic problem.
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-14, false);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let bundle = state.obtain_eval_bundle(&rho).expect("bundle");
        let pr = bundle.pirls_result.as_ref();

        let beta = beta_original_from_bundle(&bundle);
        let h_orig = h_original_from_bundle(&bundle);
        let u = &pr.solveweights * &(&pr.solveworking_response - &pr.final_eta);

        // B from implicit solve:
        //   H B = X_τ^T g - X^T W(X_τ β̂) - S_τ β̂.
        let x_tau_beta = gam_linalg::faer_ndarray::fast_av(&x_tau, &beta);
        let weighted_x_tau_beta = &pr.finalweights * &x_tau_beta;
        let rhs = gam_linalg::faer_ndarray::fast_atv(&x_tau, &u)
            - gam_linalg::faer_ndarray::fast_atv(&x, &weighted_x_tau_beta)
            - s_tau.dot(&beta);
        let chol = h_orig.cholesky(Side::Lower).expect("chol(H)");
        let b_analytic = chol.solvevec(&rhs);

        // H_τ from exact total derivative:
        //   H_τ = X_τ^T W X + X^T W X_τ + X^T W_τ X + S_τ,
        // with W_τ provided by the family directional curvature callback.
        let eta_dot = &x_tau_beta + &gam_linalg::faer_ndarray::fast_av(&x, &b_analytic);
        let w_direction = crate::pirls::directionalworking_curvature_from_c_array(
            &pr.solve_c_array.to_owned(),
            &eta_dot,
        );
        let wx = RemlState::row_scale(&x, &pr.finalweights.to_owned());
        let wx_tau = RemlState::row_scale(&x_tau, &pr.finalweights.to_owned());
        let mut xwtau_x = x.clone();
        match w_direction {
            crate::pirls::DirectionalWorkingCurvature::Diagonal(diag) => {
                xwtau_x = RemlState::row_scale(&xwtau_x, &diag);
            }
        }
        let mut h_tau_analytic = gam_linalg::faer_ndarray::fast_atb(&x_tau, &wx);
        h_tau_analytic += &gam_linalg::faer_ndarray::fast_atb(&x, &wx_tau);
        h_tau_analytic += &gam_linalg::faer_ndarray::fast_atb(&x, &xwtau_x);
        h_tau_analytic += &s_tau;

        // Fit-block stationarity cancellation:
        //   -ℓ_β^T B + β̂^T S B = 0.
        // Here S is the effective penalty in the inner Hessian surface:
        //   S = H - X^T W X.
        let ell_beta = gam_linalg::faer_ndarray::fast_atv(&x, &u);
        let s_eff = &h_orig - &gam_linalg::faer_ndarray::fast_atb(&x, &wx);
        let cancellation = -ell_beta.dot(&b_analytic) + beta.dot(&s_eff.dot(&b_analytic));

        // Finite differences in τ against re-fit objective and mode.
        let h = 2e-5;
        let x_plus = &x + &(x_tau.mapv(|v| h * v));
        let x_minus = &x - &(x_tau.mapv(|v| h * v));
        let s_plus = &s0 + &(s_tau.mapv(|v| h * v));
        let s_minus = &s0 - &(s_tau.mapv(|v| h * v));

        let state_plus = build_logit_state(&y, &w, &x_plus, &s_plus, &cfg);
        let state_minus = build_logit_state(&y, &w, &x_minus, &s_minus, &cfg);
        let bundle_plus = state_plus.obtain_eval_bundle(&rho).expect("bundle+");
        let bundle_minus = state_minus.obtain_eval_bundle(&rho).expect("bundle-");
        let beta_plus = beta_original_from_bundle(&bundle_plus);
        let beta_minus = beta_original_from_bundle(&bundle_minus);
        let bfd = (&beta_plus - &beta_minus).mapv(|v| v / (2.0 * h));

        let h_plus = h_original_from_bundle(&bundle_plus);
        let h_minus = h_original_from_bundle(&bundle_minus);
        let h_taufd = (&h_plus - &h_minus).mapv(|v| v / (2.0 * h));

        let v_plus = state_plus.compute_cost(&rho).expect("cost+");
        let v_minus = state_minus.compute_cost(&rho).expect("cost-");
        let v_taufd = (v_plus - v_minus) / (2.0 * h);

        let v_tau_analytic = single_directional_tau_gradient(&state, &rho, hyper.clone())
            .expect("analytic directional gradient");

        let b_num = (&b_analytic - &bfd).mapv(|v| v * v).sum().sqrt();
        let b_den = bfd.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let b_rel = b_num / b_den;
        for i in 0..b_analytic.len() {
            assert_eq!(
                b_analytic[i].signum(),
                bfd[i].signum(),
                "B sign mismatch at i={i}: analytic={} fd={}",
                b_analytic[i],
                bfd[i]
            );
        }
        assert!(
            b_rel < 2e-2,
            "B implicit solve mismatch vs FD: rel={b_rel:.3e}, num={b_num:.3e}, den={b_den:.3e}"
        );

        let dh_num = (&h_tau_analytic - &h_taufd).mapv(|v| v * v).sum().sqrt();
        let dh_den = h_taufd.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let dh_rel = dh_num / dh_den;
        for i in 0..h_tau_analytic.nrows() {
            for j in 0..h_tau_analytic.ncols() {
                assert_eq!(
                    h_tau_analytic[[i, j]].signum(),
                    h_taufd[[i, j]].signum(),
                    "H_tau sign mismatch at ({i},{j}): analytic={} fd={}",
                    h_tau_analytic[[i, j]],
                    h_taufd[[i, j]]
                );
            }
        }
        assert!(
            dh_rel < 3e-2,
            "H_tau mismatch vs FD: rel={dh_rel:.3e}, num={dh_num:.3e}, den={dh_den:.3e}"
        );

        let v_abs = (v_tau_analytic - v_taufd).abs();
        let v_rel = v_abs / v_taufd.abs().max(1e-10);
        assert_eq!(
            v_tau_analytic.signum(),
            v_taufd.signum(),
            "V_tau sign mismatch: analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );
        assert!(
            v_rel < 2e-2,
            "V_tau mismatch vs FD: rel={v_rel:.3e}, abs={v_abs:.3e}, analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );

        assert!(
            cancellation.abs() < 1e-10,
            "stationarity cancellation failed: | -ell_beta^T B + beta^T S B | = {:.3e}",
            cancellation.abs()
        );
    }

    #[test]
    pub(crate) fn firth_exacthessian_includes_analytic_tk_second_derivatives() {
        // Rank-deficient X: the 4th column is 2x the 2nd column.
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.4, -2.4],
            [1.0, -0.9, -0.1, -1.8],
            [1.0, -0.6, 0.3, -1.2],
            [1.0, -0.2, -0.4, -0.4],
            [1.0, 0.1, 0.5, 0.2],
            [1.0, 0.4, -0.6, 0.8],
            [1.0, 0.8, 0.2, 1.6],
            [1.0, 1.1, -0.3, 2.2],
            [1.0, 1.4, 0.7, 2.8],
            [1.0, 1.7, -0.2, 3.4],
        ];
        let s0 = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.2, 0.0],
            [0.0, 0.2, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ];
        let s1 = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, -0.1, 0.0],
            [0.0, -0.1, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.3],
        ];
        let offset = Array1::<f64>::zeros(y.len());
        // Rank-deficient Firth logit needs more inner iterations to converge
        // tightly enough for the envelope-theorem derivative tests.
        let cfg =
            RemlConfig::external(binomial_logit_glm_spec(), 1e-9, true).with_max_iterations(500);
        let p = x.ncols();
        use crate::estimate::PenaltySpec;
        let specs = vec![PenaltySpec::Dense(s0), PenaltySpec::Dense(s1)];
        let canonical =
            gam_terms::construction::canonicalize_penalty_specs(&specs, &[1, 1], p, "test")
                .map(|(canonical, _)| canonical)
                .expect("canonicalize");
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            w.view(),
            offset.view(),
            canonical,
            p,
            &cfg,
            Some(vec![1, 1]),
            None,
            None,
        )
        .expect("state");
        let rho = array![0.1, -0.2];
        assert!(
            state.analytic_outer_hessian_enabled(),
            "Firth logit should no longer disable analytic outer Hessian planning"
        );
        let outer = state
            .compute_outer_eval_with_order(
                &rho,
                crate::rho_optimizer::OuterEvalOrder::ValueGradientHessian,
            )
            .expect("outer Hessian eval should succeed");
        assert!(
            outer.hessian.is_analytic(),
            "outer planner should request and return an analytic Hessian"
        );
        let bundle = state.obtain_eval_bundle(&rho).expect("exact firth bundle");
        let h_dense = state
            .compute_lamlhessian_exact_from_bundle(&rho, &bundle)
            .expect("Firth exact Hessian should include analytic TK second derivatives");
        assert_eq!(h_dense.raw_dim(), ndarray::Ix2(2, 2));
        assert!(
            h_dense.iter().all(|value| value.is_finite()),
            "Hessian should be finite: {h_dense:?}"
        );
    }

    #[test]
    pub(crate) fn firth_outer_hessian_matches_gradient_finite_difference_with_tk_terms() {
        assert_firth_outer_hessian_matches_gradient_finite_difference(
            binomial_logit_glm_spec(),
            2.0e-3,
        );
    }

    /// #3203: non-canonical Firth links take `f = d⁴W_obs/dη⁴` from the
    /// six-order Bernoulli log jet; the analytic TK outer ρ-Hessian must match
    /// the central difference of the analytic gradient. The central-difference
    /// truncation `δ²|∇³V|/6` is ~1e-10 at δ = 2e-5 and the inner-solve residual
    /// at tol 1e-9 keeps every entry within 4e-8 of the difference, while
    /// dropping the `f` term moves at least one entry per link by ≥ 1e-4, so the
    /// 1e-6 band certifies the `f` carrier itself rather than only c/d/e.
    #[test]
    fn noncanonical_firth_outer_hessian_matches_gradient_finite_difference_3203() {
        for link in [
            StandardLink::Probit,
            StandardLink::CLogLog,
            StandardLink::LogLog,
            StandardLink::Cauchit,
        ] {
            assert_firth_outer_hessian_matches_gradient_finite_difference(
                GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(link),
                )),
                1.0e-6,
            );
        }
    }

    fn assert_firth_outer_hessian_matches_gradient_finite_difference(
        likelihood: GlmLikelihoodSpec,
        rel_tol: f64,
    ) {
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.0, 0.3],
            [1.0, -0.7, -0.2],
            [1.0, -0.3, 0.4],
            [1.0, 0.0, -0.5],
            [1.0, 0.2, 0.6],
            [1.0, 0.6, -0.4],
            [1.0, 0.9, 0.2],
            [1.0, 1.3, -0.1],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.1], [0.0, 0.1, 0.7],];
        let s1 = array![[0.0, 0.0, 0.0], [0.0, 0.4, -0.05], [0.0, -0.05, 0.9],];
        let link_label = format!("{:?}", likelihood);
        let cfg = RemlConfig::external(likelihood, 1e-9, true).with_max_iterations(500);
        let p_dim = x.ncols();
        use crate::estimate::PenaltySpec;
        let specs = vec![PenaltySpec::Dense(s0), PenaltySpec::Dense(s1)];
        let canonical =
            gam_terms::construction::canonicalize_penalty_specs(&specs, &[1, 1], p_dim, "test")
                .map(|(canonical, _)| canonical)
                .expect("canonicalize");
        let offset = Array1::<f64>::zeros(y.len());
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            w.view(),
            offset.view(),
            canonical,
            p_dim,
            &cfg,
            Some(vec![1, 1]),
            None,
            None,
        )
        .expect("state");
        assert!(
            state.analytic_outer_hessian_enabled(),
            "{link_label}: Firth must carry its analytic outer Hessian"
        );
        let rho = array![0.15, -0.25];
        let eval = state
            .compute_outer_eval_with_order(
                &rho,
                crate::rho_optimizer::OuterEvalOrder::ValueGradientHessian,
            )
            .expect("analytic Hessian eval");
        let h = match eval.hessian {
            HessianValue::Dense(hessian) => hessian,
            HessianValue::Operator(_) | HessianValue::Unavailable => {
                panic!("expected dense analytic Hessian")
            }
        };
        let delta = 2.0e-5;
        for col in 0..rho.len() {
            let mut rp = rho.clone();
            let mut rm = rho.clone();
            rp[col] += delta;
            rm[col] -= delta;
            let gp = state
                .compute_outer_eval_with_order(
                    &rp,
                    crate::rho_optimizer::OuterEvalOrder::ValueAndGradient,
                )
                .expect("plus grad")
                .gradient;
            let gm = state
                .compute_outer_eval_with_order(
                    &rm,
                    crate::rho_optimizer::OuterEvalOrder::ValueAndGradient,
                )
                .expect("minus grad")
                .gradient;
            for row in 0..rho.len() {
                let fd = (gp[row] - gm[row]) / (2.0 * delta);
                let an = h[[row, col]];
                let rel = (fd - an).abs() / fd.abs().max(an.abs()).max(1e-6);
                assert!(
                    rel < rel_tol,
                    "{link_label}: Hessian mismatch ({row},{col}): analytic={an:.9e}, fd={fd:.9e}, rel={rel:.3e}"
                );
            }
        }
    }

    #[test]
    pub(crate) fn firthgradient_lives_in_design_column_space_under_rank_deficiency() {
        // Rank-deficient design: col4 = 2*col2.
        let x = array![
            [1.0, -1.2, 0.4, -2.4],
            [1.0, -0.9, -0.1, -1.8],
            [1.0, -0.6, 0.3, -1.2],
            [1.0, -0.2, -0.4, -0.4],
            [1.0, 0.1, 0.5, 0.2],
            [1.0, 0.4, -0.6, 0.8],
            [1.0, 0.8, 0.2, 1.6],
            [1.0, 1.1, -0.3, 2.2],
        ];
        let beta = array![0.1, -0.2, 0.3, 0.05];
        let eta = x.dot(&beta);
        let op = super::RemlState::build_firth_dense_operator_for_link(
            &gam_problem::InverseLink::Standard(gam_problem::StandardLink::Logit),
            &x,
            &eta,
            ndarray::Array1::ones(x.nrows()).view(),
        )
        .expect("firth operator");

        // Exact reduced-space Firth gradient:
        //   gradPhi = 0.5 Xᵀ (w' ⊙ h), with h = diag(X_r K_r X_rᵀ).
        let gradphi = 0.5 * x.t().dot(&(&op.w1 * &op.h_diag));

        // Check (I - QQᵀ) gradPhi ≈ 0.
        let q = &op.q_basis;
        let proj = q.dot(&q.t().dot(&gradphi));
        let resid = &gradphi - &proj;
        let rel =
            resid.mapv(|v| v * v).sum().sqrt() / gradphi.mapv(|v| v * v).sum().sqrt().max(1e-12);
        assert!(
            rel < 1e-10,
            "Firth gradient should lie in Col(Xᵀ): rel residual={rel:.3e}"
        );
    }

    #[test]
    pub(crate) fn firth_logit_directional_hypergradient_accepts_penalty_only_with_full_tk_gradient()
    {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.1, 0.8],];
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            Array2::<f64>::zeros((x.nrows(), x.ncols())),
            array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.03], [0.0, 0.03, 0.12],],
            None,
            None,
        )
        .expect("single-penalty hyper direction");
        let rho = array![0.0];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let gradient = single_directional_tau_gradient(&state, &rho, hyper)
            .expect("Firth penalty-only directional gradient should use analytic TK propagation");
        assert!(gradient.is_finite(), "gradient={gradient}");
        let fd = fd_directional_tau_cost_gradient(
            &y,
            &w,
            &x,
            &s0,
            &cfg,
            &rho,
            &Array2::<f64>::zeros((x.nrows(), x.ncols())),
            &array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.03], [0.0, 0.03, 0.12],],
        );
        let rel = (gradient - fd).abs() / gradient.abs().max(fd.abs()).max(1.0e-10);
        assert!(
            rel < 1.0e-3,
            "Firth penalty-only directional gradient mismatch: analytic={gradient:.12e}, fd={fd:.12e}, rel={rel:.3e}"
        );

        let efs_hyper = DirectionalHyperParam::single_penalty(
            0,
            Array2::<f64>::zeros((x.nrows(), x.ncols())),
            array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.03], [0.0, 0.03, 0.12],],
            None,
            None,
        )
        .expect("single-penalty EFS hyper direction");
        let efs = state
            .compute_efs_steps_with_psi_ext(&rho, &[efs_hyper])
            .expect("Firth penalty-only EFS should use analytic TK propagation");
        assert!(efs.cost.is_finite(), "efs cost={}", efs.cost);
    }

    /// Regression for gam#1821: the analytic ρ-gradient of the Firth-corrected
    /// LAML cost must equal the central finite difference of that SAME cost,
    /// evaluated END-TO-END through the inner P-IRLS solve (`compute_cost` /
    /// `compute_gradient`), for a genuinely Firth-active (near-separable) fit.
    ///
    /// This exercises the branch the earlier operator-level Firth FD tests never
    /// touched: the LM line-search that produces β̂. The dense LAML gradient uses
    /// the envelope identity, which holds ONLY when β̂ satisfies the *Firth*-KKT
    /// stationarity `∇(−ℓ+½βᵀSβ) = ∇Φ`. When `GamWorkingModel::update_candidate`
    /// built line-search candidates with Firth disabled, the candidate/accepted
    /// `WorkingState` dropped the `−2·½log|XᵀWX|` Jeffreys term, the objective
    /// the line search compared (candidate vs `current_penalized`) was
    /// inconsistent, and — because the accepted state IS the candidate and
    /// convergence is certified on `accepted_state.gradient` — the inner solve
    /// settled at the ordinary penalized MLE (`∇(−ℓ+½βᵀSβ)=0`) instead. At that
    /// wrong mode the envelope breaks and the analytic ρ-gradient disagrees with
    /// the FD of the cost by `O((∇Φ)ᵀ∂β̂/∂ρ)` — a large (percent-level) desync
    /// that appears ONLY under `firth_bias_reduction`. A tight FD-vs-analytic
    /// bound therefore fails iff the inner solve regresses off the Firth mode.
    #[test]
    pub(crate) fn firth_logit_rho_gradient_matches_finite_difference_through_inner_solve() {
        // Near-separable n=3 logit: Firth is materially active (the ordinary
        // penalized MLE and the Firth-penalized mode differ enough that any
        // envelope break shows up far above the 1e-4 tolerance).
        let x = array![[1.0, -6.0], [1.0, 0.2], [1.0, 5.8]];
        let y = array![0.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        // Full-rank identity penalty on both coefficients (single ρ).
        let s0 = array![[1.0, 0.0], [0.0, 1.0]];
        // Tight inner tolerance so β̂(ρ ± δ) and β̂(ρ) are all at the converged
        // Firth-KKT mode; otherwise the FD would capture β̂'s residual motion.
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-12, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let delta = 1e-4_f64;
        for &rho in &[-0.6_f64, -0.3, 0.0, 0.3, 0.6] {
            let r = array![rho];
            let analytic = state
                .compute_gradient(&r)
                .expect("Firth LAML ρ-gradient should evaluate")[0];
            let cost_plus = state
                .compute_cost(&array![rho + delta])
                .expect("Firth LAML cost(ρ+δ) should evaluate");
            let cost_minus = state
                .compute_cost(&array![rho - delta])
                .expect("Firth LAML cost(ρ−δ) should evaluate");
            let fd = (cost_plus - cost_minus) / (2.0 * delta);
            let rel = (fd - analytic).abs() / fd.abs().max(1e-3);
            assert!(
                analytic.is_finite() && fd.is_finite(),
                "non-finite Firth ρ-gradient at rho={rho:+.3}: fd={fd:+.6e}, analytic={analytic:+.6e}"
            );
            assert!(
                rel < 1e-4,
                "Firth ρ-gradient FD desync at rho={rho:+.3}: fd={fd:+.6e}, analytic={analytic:+.6e}, rel={rel:.3e} (>= 1e-4). \
                 The inner P-IRLS likely converged off the Firth-KKT mode (gam#1821)."
            );
        }
    }

    /// Clamped B-spline design matrix by the Cox–de Boor recursion. A point
    /// equal to the last knot belongs to the last non-degenerate span, so each
    /// row sums to one on the closed interval.
    fn clamped_bspline_design(x: &Array1<f64>, knots: &[f64], degree: usize) -> Array2<f64> {
        let k = knots.len() - degree - 1;
        let last_span = knots.len() - degree - 2;
        let mut design = Array2::<f64>::zeros((x.len(), k));
        for (row, &xv) in x.iter().enumerate() {
            let span = (degree..=last_span)
                .find(|&j| knots[j] <= xv && xv < knots[j + 1])
                .unwrap_or(last_span);
            let mut basis = vec![0.0_f64; knots.len() - 1];
            basis[span] = 1.0;
            for order in 1..=degree {
                for i in 0..knots.len() - 1 - order {
                    let left_den = knots[i + order] - knots[i];
                    let right_den = knots[i + order + 1] - knots[i + 1];
                    let left = if left_den > 0.0 {
                        (xv - knots[i]) / left_den * basis[i]
                    } else {
                        0.0
                    };
                    let right = if right_den > 0.0 {
                        (knots[i + order + 1] - xv) / right_den * basis[i + 1]
                    } else {
                        0.0
                    };
                    basis[i] = left + right;
                }
            }
            for col in 0..k {
                design[[row, col]] = basis[col];
            }
        }
        design
    }

    /// #3507: the Firth LAML outer criterion is bounded in ρ, including through
    /// the pitchfork of the Firth mode.
    ///
    /// Fixture: a perfectly separated step `y = 1{x > ½}` on a design that is
    /// symmetric under `x ↦ 1 − x`, fitted by five clamped cubic B-splines with
    /// a second-difference penalty (rank r = 3, unpenalized linear null space).
    /// The symmetric stationary point of `F = −ℓ + ½βᵀS_λβ − ½log|XᵀWX|` has
    /// `λ_min(H_F) < 0` for ρ ≲ −9.3, so the inner mode is the asymmetric
    /// branch and `λ_min(H_F) → 0` at the bifurcation.
    ///
    /// The derived criterion is
    /// `V = F(β̂) + ½log|H_0| − ½tr(H_0⁻¹HΦ) − TK(H_0) − ½log|S_λ|₊`, with
    /// `H_0 = XᵀWX + S_λ ≻ 0` everywhere. So:
    /// * V is finite at every ρ, including across the pitchfork. The former
    ///   criterion (`½log|H_F| + TK(H_F)`) blows up there: 5.3e3 at ρ = −9.25
    ///   on this fixture, against 34.9 at ρ = −20.
    /// * Lower edge (λ → 0): β̂ tends to the unpenalized Firth fit, which is
    ///   finite for full-rank X. Every term except `−½log|S_λ|₊ = −½rρ + c` is
    ///   `O(λ)`. So `dV/dρ → −r/2` and V → +∞, the largest value on the scan.
    /// * Upper edge (λ → ∞): `½log|H_0|` grows as `½rρ` and cancels the
    ///   pseudo-determinant. β̂ tends to the Firth fit in the null space, so V
    ///   reaches a plateau, with `dV/dρ = O(1/λ)`.
    ///
    /// Tolerances:
    /// * At ρ = −20 the `O(λ)` corrections are `λ·tr(H_0⁻¹S)`-sized, about
    ///   2e-5·r/2 here. The 1e-2·r/2 band leaves room for the inner-solve
    ///   residual at tol 1e-9.
    /// * At ρ = 20 the slope is `O(1/λ) ≈ 2e-9`, so the same band bounds both
    ///   the gradient and the last cost step, `ΔV ≤ band·Δρ`.
    #[test]
    fn firth_laml_outer_criterion_bounded_across_rho_edges_through_pitchfork_3507() {
        let n = 64usize;
        let x_pts = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y = x_pts.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        let w = Array1::<f64>::ones(n);
        let degree = 3usize;
        let knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let x = clamped_bspline_design(&x_pts, &knots, degree);
        let k = x.ncols();
        let mut d2 = Array2::<f64>::zeros((k - 2, k));
        for row in 0..k - 2 {
            d2[[row, row]] = 1.0;
            d2[[row, row + 1]] = -2.0;
            d2[[row, row + 2]] = 1.0;
        }
        let s = d2.t().dot(&d2);
        let half_rank = (k - 2) as f64 / 2.0;
        let cfg =
            RemlConfig::external(binomial_logit_glm_spec(), 1e-9, true).with_max_iterations(500);
        let state = build_logit_state(&y, &w, &x, &s, &cfg);

        // Both edges, plus a quarter-step walk through the pitchfork near −9.3.
        let grid = [
            -20.0, -16.0, -12.0, -10.0, -9.75, -9.5, -9.25, -9.0, -8.75, -8.5, -8.0, -6.0, -4.0,
            -2.0, 0.0, 4.0, 8.0, 12.0, 16.0, 20.0,
        ];
        let mut costs = Vec::with_capacity(grid.len());
        let mut grads = Vec::with_capacity(grid.len());
        for &rho in &grid {
            let r = array![rho];
            let cost = state
                .compute_cost(&r)
                .unwrap_or_else(|e| panic!("Firth LAML cost at rho={rho:+.2} failed: {e:?}"));
            let grad = state
                .compute_gradient(&r)
                .unwrap_or_else(|e| panic!("Firth LAML gradient at rho={rho:+.2} failed: {e:?}"))
                [0];
            assert!(
                cost.is_finite() && grad.is_finite(),
                "Firth LAML must be finite at rho={rho:+.2}: cost={cost:?}, grad={grad:?}"
            );
            costs.push(cost);
            grads.push(grad);
        }
        let table: Vec<String> = grid
            .iter()
            .zip(costs.iter().zip(grads.iter()))
            .map(|(rho, (c, g))| format!("rho={rho:+6.2} V={c:+.6} dV={g:+.6e}"))
            .collect();
        let table = table.join("\n");
        let band = 1e-2 * half_rank;

        // Lower edge: coercive at the pseudo-determinant rate.
        assert!(
            (grads[0] + half_rank).abs() <= band,
            "lower-edge slope must be −r/2 = {:+.3}, got {:+.6e}\n{table}",
            -half_rank,
            grads[0]
        );
        let (argmax, _) = costs
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, (i, &c)| {
                if c > best.1 { (i, c) } else { best }
            });
        assert_eq!(
            argmax, 0,
            "the lower edge must carry the largest criterion value (V → +∞ as ρ → −∞); \
             an interior maximum is the unbounded-curvature failure of #3507\n{table}"
        );

        // Upper edge: plateau.
        let last = grid.len() - 1;
        assert!(
            grads[last].abs() <= band,
            "upper-edge slope must vanish, got {:+.6e}\n{table}",
            grads[last]
        );
        assert!(
            (costs[last] - costs[last - 1]).abs() <= band * (grid[last] - grid[last - 1]),
            "upper-edge criterion must plateau\n{table}"
        );
    }

    #[test]
    pub(crate) fn firth_logit_directional_hypergradient_accepts_design_moving_with_full_tk_gradient()
     {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.1, 0.8],];
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            Array2::from_elem((x.nrows(), x.ncols()), 1e-3),
            Array2::<f64>::zeros((x.ncols(), x.ncols())),
            None,
            None,
        )
        .expect("single-penalty hyper direction");
        let rho = array![0.0];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let gradient = single_directional_tau_gradient(&state, &rho, hyper)
            .expect("Firth design-moving directional gradient should use analytic TK propagation");
        assert!(gradient.is_finite(), "gradient={gradient}");
        let x_tau = Array2::from_elem((x.nrows(), x.ncols()), 1e-3);
        let s_tau = Array2::<f64>::zeros((x.ncols(), x.ncols()));
        let fd = fd_directional_tau_cost_gradient(&y, &w, &x, &s0, &cfg, &rho, &x_tau, &s_tau);
        let rel = (gradient - fd).abs() / gradient.abs().max(fd.abs()).max(1.0e-10);
        assert!(
            rel < 2.0e-2,
            "Firth design-moving directional gradient mismatch: analytic={gradient:.12e}, fd={fd:.12e}, rel={rel:.3e}"
        );
    }

    #[test]
    pub(crate) fn firth_logit_hybrid_efs_accepts_full_tk_psi_gradient() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.1, 0.8],];
        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(
                0,
                Array2::from_shape_fn((x.nrows(), x.ncols()), |(i, j)| {
                    1e-3 * ((i + 1) as f64) * ((j + 2) as f64)
                }),
                Array2::<f64>::zeros((x.ncols(), x.ncols())),
                None,
                None,
            )
            .expect("design-moving hyper direction"),
        ];
        let rho = array![0.0];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);

        let full = state
            .evaluate_unified_with_psi_ext(
                &rho,
                None,
                crate::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient,
                &hyper_dirs,
            )
            .expect("full Firth psi gradient should use analytic TK propagation");
        assert!(full.cost.is_finite(), "full cost={}", full.cost);
        let full_grad = full.gradient.expect("gradient should be present");
        assert!(
            full_grad.iter().all(|value| value.is_finite()),
            "full gradient={full_grad:?}"
        );

        let efs = state
            .compute_efs_steps_with_psi_ext(&rho, &hyper_dirs)
            .expect("hybrid EFS should use analytic TK propagation");
        assert!(efs.cost.is_finite(), "efs cost={}", efs.cost);
    }

    #[test]
    pub(crate) fn joint_hyperhessianwires_mixed_blocks() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9],];
        let cfg =
            RemlConfig::external(binomial_logit_glm_spec(), 1e-10, false).with_max_iterations(500);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];
        let theta = array![0.0, 0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(
                0,
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
                None,
                None,
            )
            .expect("single-penalty hyper direction"),
            DirectionalHyperParam::single_penalty(
                0,
                Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                Array2::<f64>::zeros((x.ncols(), x.ncols())),
                None,
                None,
            )
            .expect("single-penalty hyper direction"),
        ];

        let (_, _, h) =
            compute_joint_hypercostgradienthessian(&state, &theta, rho.len(), &hyper_dirs)
                .expect("joint hyper cost+gradient+hessian");
        assert_eq!(h.nrows(), theta.len());
        assert_eq!(h.ncols(), theta.len());
        assert!(h.iter().all(|v| v.is_finite()));
        for i in 0..h.nrows() {
            for j in 0..i {
                let diff = (h[[i, j]] - h[[j, i]]).abs();
                assert!(
                    diff < 1e-6,
                    "joint hessian asymmetry at ({i},{j}): {diff:.3e}"
                );
            }
        }
        // Mixed block must be nontrivial for at least one supplied direction.
        let mixed_0 = h[[0, 1]];
        let mixed_1 = h[[0, 2]];
        assert!(
            mixed_0.is_finite() && mixed_1.is_finite(),
            "mixed blocks must be finite"
        );
    }

    #[test]
    pub(crate) fn joint_tau_tau_linear_dirs_matchfd_reference_away_fromzero_psi() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9],];
        let cfg =
            RemlConfig::external(binomial_logit_glm_spec(), 1e-10, false).with_max_iterations(500);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];
        let psi = array![0.7, -0.4];
        let theta = array![rho[0], psi[0], psi[1]];
        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(
                0,
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
                None,
                None,
            )
            .expect("linear tau direction"),
            DirectionalHyperParam::single_penalty(
                0,
                Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                Array2::<f64>::zeros((x.ncols(), x.ncols())),
                None,
                None,
            )
            .expect("linear tau direction"),
        ];

        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, rho.len(), &hyper_dirs)
                .expect("joint hyper cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![rho.len().., rho.len()..]).to_owned();

        // FD via physical perturbation of design/penalty matrices (matching
        // the V_tau FD pattern).  For column j we perturb X and S₀ along
        // direction j, build fresh states, and evaluate the τ-gradient for
        // every direction i at those perturbed states.
        let x_tau_mats: Vec<Array2<f64>> = vec![
            Array2::<f64>::zeros((x.nrows(), x.ncols())),
            Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
        ];
        let s_tau_mats: Vec<Array2<f64>> = vec![
            array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15]],
            Array2::<f64>::zeros((x.ncols(), x.ncols())),
        ];

        let h_ttfd = directional_tau_hessian_fd_reference(
            &y,
            &w,
            &x,
            &s0,
            &cfg,
            &rho,
            &hyper_dirs,
            &x_tau_mats,
            &s_tau_mats,
        );

        let num = (&h_tt_analytic - &h_ttfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_ttfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "linear-dir joint tau-tau block deviates from FD reference away from zero psi: rel={rel:.3e}, analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    #[test]
    pub(crate) fn joint_hypervalidation_rejects_out_of_boundssecond_order_penalty_index() {
        // The hyper direction declares a second-order penalty derivative
        // against base penalty index 1, but the configured ρ vector has
        // dimension 1 (so only index 0 is valid).  The pair-callback
        // builder in `build_tau_penalty_derivative_data` is responsible for
        // validating both first- and second-order penalty indices against
        // `rho.len()`; this test pins that contract.
        //
        // We deliberately keep `firth_bias_reduction = true` here so the
        // call site exercises the full Firth/Tierney–Kadane outer pipeline:
        // PIRLS + ext-coord construction + pair-callback assembly.  With
        // analytic c/d propagation now wired in
        // `tk_direct_gradient_from_cd_and_design`, there is no longer any
        // FD-fallback rejection on this path, so the out-of-bounds error
        // fired by the pair-callback builder is the first failure the
        // joint evaluator surfaces — and that is exactly what we want this
        // test to assert.
        let y = array![0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -0.5, 0.2],
            [1.0, -0.1, -0.3],
            [1.0, 0.4, 0.6],
            [1.0, 0.9, -0.2],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.1, 0.8],];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-10, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let theta = array![0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam::new(
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                vec![(0, Array2::<f64>::zeros((x.ncols(), x.ncols())))],
                None,
                Some(vec![Some(vec![(1, Array2::<f64>::eye(x.ncols()))])]),
            )
            .expect("hyper direction with invalid second-order penalty index"),
        ];

        let msg = match compute_joint_hypercostgradienthessian(&state, &theta, 1, &hyper_dirs) {
            Ok(_) => panic!("invalid second-order penalty index should be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("out of bounds") || msg.contains("penalty_index"),
            "unexpected validation error: {msg}"
        );
    }

    #[test]
    pub(crate) fn joint_tau_tau_analytic_matchesfd_reference() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9],];
        let cfg =
            RemlConfig::external(binomial_logit_glm_spec(), 1e-10, false).with_max_iterations(500);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];
        let psi = array![0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(
                0,
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
                None,
                None,
            )
            .expect("single-penalty hyper direction"),
            DirectionalHyperParam::single_penalty(
                0,
                Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                Array2::<f64>::zeros((x.ncols(), x.ncols())),
                None,
                None,
            )
            .expect("single-penalty hyper direction"),
        ];

        let theta = {
            let mut t = Array1::<f64>::zeros(rho.len() + psi.len());
            t.slice_mut(s![..rho.len()]).assign(&rho);
            t.slice_mut(s![rho.len()..]).assign(&psi);
            t
        };
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, rho.len(), &hyper_dirs)
                .expect("joint hyper cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![rho.len().., rho.len()..]).to_owned();
        assert_eq!(h_tt_analytic.nrows(), hyper_dirs.len());
        assert_eq!(h_tt_analytic.ncols(), hyper_dirs.len());

        // FD via physical perturbation of design/penalty matrices (matching
        // the V_tau FD pattern).  For column j we perturb X and S₀ along
        // direction j, build fresh states, and evaluate the τ-gradient for
        // every direction i at those perturbed states.
        let x_tau_mats: Vec<Array2<f64>> = vec![
            Array2::<f64>::zeros((x.nrows(), x.ncols())),
            Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
        ];
        let s_tau_mats: Vec<Array2<f64>> = vec![
            array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15]],
            Array2::<f64>::zeros((x.ncols(), x.ncols())),
        ];

        let h_ttfd = directional_tau_hessian_fd_reference(
            &y,
            &w,
            &x,
            &s0,
            &cfg,
            &rho,
            &hyper_dirs,
            &x_tau_mats,
            &s_tau_mats,
        );

        let num = (&h_tt_analytic - &h_ttfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_ttfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "analytic tau-tau block deviates from FD reference: rel={rel:.3e}, analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    // ── Profiled Gaussian REML coverage for design-moving τ-directions ──
    //
    // The existing directional-hyper tests all use BinomialLogit, which has
    // DispersionHandling::Fixed.  These tests validate the profiled Gaussian
    // path (DispersionHandling::ProfiledGaussian) with design-moving
    // τ-directions, where the profiled scale φ̂ = D_p/(n−M) depends on ρ
    // and the envelope-theorem rescaling by (n−M)/D_p must be correct.

    /// Shared test fixture for profiled Gaussian REML tests.
    pub(crate) struct GaussianRemlFixture {
        pub(crate) y: Array1<f64>,
        pub(crate) w: Array1<f64>,
        pub(crate) x: Array2<f64>,
        pub(crate) s0: Array2<f64>,
        pub(crate) cfg: RemlConfig,
        pub(crate) rho: Array1<f64>,
        /// Design-moving τ-direction (non-zero X_τ, zero S_τ).
        pub(crate) x_tau_design: Array2<f64>,
        /// Penalty-only τ-direction (zero X_τ, non-zero S_τ).
        pub(crate) s_tau_penalty: Array2<f64>,
    }

    impl GaussianRemlFixture {
        pub(crate) fn new() -> Self {
            let y = array![0.5, 1.2, -0.3, 0.8, 1.1, -0.6, 0.9, 0.1, -0.2, 0.7];
            let x = array![
                [1.0, -1.2, 0.3],
                [1.0, -0.8, -0.4],
                [1.0, -0.3, 0.7],
                [1.0, 0.1, -0.9],
                [1.0, 0.5, 0.2],
                [1.0, 0.9, -0.1],
                [1.0, 1.3, 0.8],
                [1.0, 1.7, -0.6],
                [1.0, -0.5, 0.5],
                [1.0, 0.3, -0.3],
            ];
            Self {
                w: Array1::<f64>::ones(y.len()),
                y,
                x: x.clone(),
                s0: array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9]],
                cfg: RemlConfig::external(gaussian_identity_glm_spec(), 1e-14, false),
                rho: array![0.0],
                x_tau_design: array![
                    [0.0, 1e-3, -2e-3],
                    [0.0, -3e-3, 1e-3],
                    [0.0, 2e-3, 0.5e-3],
                    [0.0, -1e-3, 3e-3],
                    [0.0, 0.5e-3, -1e-3],
                    [0.0, 1.5e-3, 2e-3],
                    [0.0, -2e-3, -0.5e-3],
                    [0.0, 3e-3, 1e-3],
                    [0.0, -0.5e-3, 2e-3],
                    [0.0, 1e-3, -1.5e-3],
                ],
                s_tau_penalty: array![[0.0, 0.0, 0.0], [0.0, 0.25, 0.04], [0.0, 0.04, 0.15]],
            }
        }
    }

    impl LogitDesignMotionFixture for GaussianRemlFixture {
        fn y(&self) -> &Array1<f64> {
            &self.y
        }
        fn w(&self) -> &Array1<f64> {
            &self.w
        }
        fn x(&self) -> &Array2<f64> {
            &self.x
        }
        fn s0(&self) -> &Array2<f64> {
            &self.s0
        }
        fn cfg(&self) -> &RemlConfig {
            &self.cfg
        }
        fn rho(&self) -> &Array1<f64> {
            &self.rho
        }
    }

    #[test]
    pub(crate) fn profiled_gaussian_design_moving_gradient_matches_fd() {
        let f = GaussianRemlFixture::new();
        let state = f.state();
        let s_tau = Array2::<f64>::zeros((3, 3));
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            f.x_tau_design.clone(),
            s_tau.clone(),
            None,
            None,
        )
        .expect("design-moving hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_taufd = f.fd_directional_gradient(&f.x_tau_design, &s_tau);

        let v_rel = (v_tau_analytic - v_taufd).abs() / v_taufd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Gaussian REML design-moving V_tau mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );
    }

    #[test]
    pub(crate) fn profiled_gaussian_penalty_only_gradient_matches_fd() {
        let f = GaussianRemlFixture::new();
        let state = f.state();
        let x_tau = Array2::<f64>::zeros(f.x.raw_dim());
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            x_tau.clone(),
            f.s_tau_penalty.clone(),
            None,
            None,
        )
        .expect("penalty-only hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_taufd = f.fd_directional_gradient(&x_tau, &f.s_tau_penalty);

        let v_rel = (v_tau_analytic - v_taufd).abs() / v_taufd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Gaussian REML penalty-only V_tau mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );
    }

    #[test]
    pub(crate) fn profiled_gaussian_joint_hessian_matches_fd() {
        // Validate the ττ Hessian block under profiled Gaussian REML with
        // both a penalty-only and a design-moving direction.
        let f = GaussianRemlFixture::new();
        let x_tau_0 = Array2::<f64>::zeros(f.x.raw_dim());
        let s_tau_0 = f.s_tau_penalty.clone();
        let x_tau_1 = f.x_tau_design.clone();
        let s_tau_1 = Array2::<f64>::zeros((3, 3));

        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(0, x_tau_0.clone(), s_tau_0.clone(), None, None)
                .expect("penalty-only direction"),
            DirectionalHyperParam::single_penalty(0, x_tau_1.clone(), s_tau_1.clone(), None, None)
                .expect("design-moving direction"),
        ];

        let state = f.state();
        let mut theta = Array1::<f64>::zeros(f.rho.len() + hyper_dirs.len());
        theta.slice_mut(s![..f.rho.len()]).assign(&f.rho);
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, f.rho.len(), &hyper_dirs)
                .expect("joint cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![f.rho.len().., f.rho.len()..]).to_owned();

        // Finite-difference Hessian: perturb each direction, re-evaluate
        // gradient of all directions at perturbed states.
        let x_tau_mats = vec![x_tau_0.clone(), x_tau_1.clone()];
        let s_tau_mats = vec![s_tau_0.clone(), s_tau_1.clone()];
        let h_ttfd = directional_tau_hessian_fd_reference(
            &f.y,
            &f.w,
            &f.x,
            &f.s0,
            &f.cfg,
            &f.rho,
            &hyper_dirs,
            &x_tau_mats,
            &s_tau_mats,
        );

        let num = (&h_tt_analytic - &h_ttfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_ttfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "Gaussian REML tau-tau Hessian mismatch: rel={rel:.3e}, \
             analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    // ── Non-Gaussian + design-motion: IFT Hessian-drift coverage ────────
    //
    // For non-Gaussian links (logit, probit, cloglog, ...), H = X'W(η)X + S
    // depends on β̂ through η = Xβ̂.  When ψ moves the design, the total
    // Hessian drift dH/dψ includes an IFT contribution from dβ̂/dψ:
    //
    //   dH/dψ = [explicit at fixed β] + X' diag(c ⊙ X(-v_i)) X
    //
    // where v_i = H⁻¹ g_i.  The standard GLM path handles this via
    // `hessian_derivative_correction(v_i)`.  This test validates that the
    // gradient is correct for logit + design-moving ψ, which would fail if
    // the IFT correction were missing.

    #[test]
    pub(crate) fn logit_design_moving_gradient_matches_fd() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
            [1.0, -0.5, 0.5],
            [1.0, 0.3, -0.3],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9]];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-14, false);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];

        // Design-moving direction with non-zero X_τ.
        let x_tau = array![
            [0.0, 1e-3, -2e-3],
            [0.0, -3e-3, 1e-3],
            [0.0, 2e-3, 0.5e-3],
            [0.0, -1e-3, 3e-3],
            [0.0, 0.5e-3, -1e-3],
            [0.0, 1.5e-3, 2e-3],
            [0.0, -2e-3, -0.5e-3],
            [0.0, 3e-3, 1e-3],
            [0.0, -0.5e-3, 2e-3],
            [0.0, 1e-3, -1.5e-3],
        ];
        let s_tau = Array2::<f64>::zeros((3, 3));
        let hyper =
            DirectionalHyperParam::single_penalty(0, x_tau.clone(), s_tau.clone(), None, None)
                .expect("design-moving hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &rho, hyper)
            .expect("analytic directional gradient");

        let h = 2e-5;
        let x_plus = &x + &x_tau.mapv(|v| h * v);
        let x_minus = &x - &x_tau.mapv(|v| h * v);
        let state_plus = build_logit_state(&y, &w, &x_plus, &s0, &cfg);
        let state_minus = build_logit_state(&y, &w, &x_minus, &s0, &cfg);
        let v_plus = state_plus.compute_cost(&rho).expect("cost+");
        let v_minus = state_minus.compute_cost(&rho).expect("cost-");
        let v_taufd = (v_plus - v_minus) / (2.0 * h);

        let v_rel = (v_tau_analytic - v_taufd).abs() / v_taufd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Logit REML design-moving V_tau mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );
    }

    #[test]
    pub(crate) fn logit_design_moving_hessian_matches_fd() {
        // Hessian-level validation for logit + design-motion.
        // The IFT correction enters the trace term through
        // hessian_derivative_correction(v_i), so the Hessian is the most
        // sensitive test of whether the correction is applied correctly.
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
            [1.0, -0.5, 0.5],
            [1.0, 0.3, -0.3],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9]];
        let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-14, false);
        let rho = array![0.0];

        // Two directions: one penalty-only, one design-moving.
        let x_tau_0 = Array2::<f64>::zeros(x.raw_dim());
        let s_tau_0 = array![[0.0, 0.0, 0.0], [0.0, 0.25, 0.04], [0.0, 0.04, 0.15]];
        let x_tau_1 = array![
            [0.0, 1e-3, -2e-3],
            [0.0, -3e-3, 1e-3],
            [0.0, 2e-3, 0.5e-3],
            [0.0, -1e-3, 3e-3],
            [0.0, 0.5e-3, -1e-3],
            [0.0, 1.5e-3, 2e-3],
            [0.0, -2e-3, -0.5e-3],
            [0.0, 3e-3, 1e-3],
            [0.0, -0.5e-3, 2e-3],
            [0.0, 1e-3, -1.5e-3],
        ];
        let s_tau_1 = Array2::<f64>::zeros((3, 3));

        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(0, x_tau_0.clone(), s_tau_0.clone(), None, None)
                .expect("penalty-only direction"),
            DirectionalHyperParam::single_penalty(0, x_tau_1.clone(), s_tau_1.clone(), None, None)
                .expect("design-moving direction"),
        ];

        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let mut theta = Array1::<f64>::zeros(rho.len() + hyper_dirs.len());
        theta.slice_mut(s![..rho.len()]).assign(&rho);
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, rho.len(), &hyper_dirs)
                .expect("joint cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![rho.len().., rho.len()..]).to_owned();

        let x_tau_mats = vec![x_tau_0.clone(), x_tau_1.clone()];
        let s_tau_mats = vec![s_tau_0.clone(), s_tau_1.clone()];
        let h_ttfd = directional_tau_hessian_fd_reference(
            &y,
            &w,
            &x,
            &s0,
            &cfg,
            &rho,
            &hyper_dirs,
            &x_tau_mats,
            &s_tau_mats,
        );

        let num = (&h_tt_analytic - &h_ttfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_ttfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "Logit REML design-moving tau-tau Hessian mismatch: rel={rel:.3e}, \
             analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    // ── Larger non-Gaussian + design-motion fixture (n=30, p=5) ────────
    //
    // Validates the IFT correction (hessian_derivative_correction) at a
    // scale large enough that the correction is numerically non-trivial:
    // with n=30 and p=5, the logistic Hessian W(η) is far from identity
    // and the IFT term dβ̂/dψ contributes meaningfully.

    /// Shared test fixture for binomial-logit REML with design-moving
    /// ψ-coordinates, n=30, p=5.
    pub(crate) struct BinomialLogitDesignMotionFixture {
        pub(crate) y: Array1<f64>,
        pub(crate) w: Array1<f64>,
        pub(crate) x: Array2<f64>,
        pub(crate) s0: Array2<f64>,
        pub(crate) cfg: RemlConfig,
        pub(crate) rho: Array1<f64>,
        /// Design-moving τ-direction: non-zero X_τ, zero S_τ.
        pub(crate) x_tau_design: Array2<f64>,
        /// Penalty-only τ-direction: zero X_τ, non-zero S_τ.
        pub(crate) s_tau_penalty: Array2<f64>,
    }

    impl BinomialLogitDesignMotionFixture {
        pub(crate) fn new() -> Self {
            // Binary response with roughly balanced classes.
            let y = array![
                1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
            ];
            // Design matrix: intercept + 4 covariate columns with varied magnitudes.
            let x = array![
                [1.0, -1.50, 0.42, 0.88, -0.31],
                [1.0, -1.12, -0.65, 0.14, 1.23],
                [1.0, -0.80, 1.10, -0.53, 0.07],
                [1.0, -0.55, -0.22, 1.40, -0.90],
                [1.0, -0.30, 0.73, -1.05, 0.44],
                [1.0, -0.05, -1.33, 0.60, 0.81],
                [1.0, 0.18, 0.55, -0.27, -1.15],
                [1.0, 0.42, -0.90, 1.12, 0.33],
                [1.0, 0.70, 1.28, -0.78, -0.56],
                [1.0, 0.95, -0.18, 0.45, 1.40],
                [1.0, 1.20, 0.66, -1.30, -0.02],
                [1.0, 1.45, -1.05, 0.22, 0.68],
                [1.0, -1.35, 0.90, 0.55, -0.43],
                [1.0, -0.98, -0.40, -0.88, 1.05],
                [1.0, -0.62, 1.42, 0.30, -0.70],
                [1.0, -0.28, -0.77, -1.18, 0.52],
                [1.0, 0.05, 0.15, 0.95, -1.35],
                [1.0, 0.33, -1.20, -0.40, 0.18],
                [1.0, 0.60, 0.82, 1.25, -0.85],
                [1.0, 0.88, -0.50, -0.65, 1.10],
                [1.0, 1.15, 1.05, 0.10, -0.22],
                [1.0, -1.22, -0.95, 0.72, 0.90],
                [1.0, -0.75, 0.38, -1.42, 0.15],
                [1.0, -0.42, -1.15, 0.50, -1.08],
                [1.0, -0.10, 0.60, -0.15, 0.75],
                [1.0, 0.25, -0.28, 1.05, -0.48],
                [1.0, 0.52, 1.35, -0.92, 0.30],
                [1.0, 0.80, -0.70, 0.38, 1.20],
                [1.0, 1.08, 0.48, -0.60, -0.95],
                [1.0, 1.35, -0.55, 0.85, 0.42]
            ];
            // Penalty matrix: zero on intercept, SPD on remaining 4 columns.
            let s0 = array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.40, 0.15, 0.05, -0.10],
                [0.0, 0.15, 1.10, -0.20, 0.08],
                [0.0, 0.05, -0.20, 0.95, 0.12],
                [0.0, -0.10, 0.08, 0.12, 1.25]
            ];
            let cfg = RemlConfig::external(binomial_logit_glm_spec(), 1e-14, false);
            // Design-moving direction: perturb covariate columns, leave
            // intercept untouched.
            let x_tau_design = array![
                [0.0, 1.2e-3, -0.8e-3, 0.5e-3, -1.5e-3],
                [0.0, -2.0e-3, 1.4e-3, -0.3e-3, 0.9e-3],
                [0.0, 0.6e-3, -1.1e-3, 1.8e-3, -0.4e-3],
                [0.0, -1.3e-3, 0.7e-3, -1.0e-3, 2.1e-3],
                [0.0, 0.9e-3, -0.5e-3, 0.2e-3, -0.8e-3],
                [0.0, -0.4e-3, 1.8e-3, -1.5e-3, 0.3e-3],
                [0.0, 1.5e-3, -1.3e-3, 0.8e-3, -1.1e-3],
                [0.0, -0.7e-3, 0.4e-3, -2.0e-3, 1.6e-3],
                [0.0, 2.2e-3, -0.9e-3, 1.3e-3, -0.6e-3],
                [0.0, -1.0e-3, 1.6e-3, -0.7e-3, 0.5e-3],
                [0.0, 0.3e-3, -2.1e-3, 1.1e-3, -1.8e-3],
                [0.0, -1.8e-3, 0.2e-3, -0.4e-3, 1.3e-3],
                [0.0, 1.1e-3, -1.5e-3, 2.0e-3, -0.2e-3],
                [0.0, -0.5e-3, 0.9e-3, -1.2e-3, 0.7e-3],
                [0.0, 1.7e-3, -0.3e-3, 0.6e-3, -2.0e-3],
                [0.0, -1.4e-3, 1.1e-3, -0.9e-3, 0.4e-3],
                [0.0, 0.8e-3, -1.7e-3, 1.5e-3, -0.1e-3],
                [0.0, -0.2e-3, 0.6e-3, -1.8e-3, 1.0e-3],
                [0.0, 1.4e-3, -0.4e-3, 0.3e-3, -1.3e-3],
                [0.0, -0.9e-3, 2.0e-3, -0.5e-3, 0.8e-3],
                [0.0, 0.5e-3, -1.0e-3, 1.6e-3, -0.7e-3],
                [0.0, -2.1e-3, 0.3e-3, -0.8e-3, 1.5e-3],
                [0.0, 0.7e-3, -1.8e-3, 0.9e-3, -0.3e-3],
                [0.0, -0.6e-3, 1.3e-3, -2.2e-3, 1.1e-3],
                [0.0, 1.9e-3, -0.7e-3, 0.4e-3, -0.9e-3],
                [0.0, -1.1e-3, 0.5e-3, -1.4e-3, 2.2e-3],
                [0.0, 0.4e-3, -1.6e-3, 1.2e-3, -0.5e-3],
                [0.0, -1.6e-3, 0.8e-3, -0.1e-3, 0.6e-3],
                [0.0, 1.3e-3, -2.2e-3, 0.7e-3, -1.4e-3],
                [0.0, -0.3e-3, 1.0e-3, -1.6e-3, 1.8e-3]
            ];
            // Penalty-only direction: non-zero S_τ, symmetric, zero on intercept.
            let s_tau_penalty = array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.30, 0.05, -0.02, 0.04],
                [0.0, 0.05, 0.22, 0.03, -0.01],
                [0.0, -0.02, 0.03, 0.18, 0.06],
                [0.0, 0.04, -0.01, 0.06, 0.26]
            ];
            Self {
                w: Array1::<f64>::ones(y.len()),
                y,
                x,
                s0,
                cfg,
                rho: array![0.0],
                x_tau_design,
                s_tau_penalty,
            }
        }
    }

    impl LogitDesignMotionFixture for BinomialLogitDesignMotionFixture {
        fn y(&self) -> &Array1<f64> {
            &self.y
        }
        fn w(&self) -> &Array1<f64> {
            &self.w
        }
        fn x(&self) -> &Array2<f64> {
            &self.x
        }
        fn s0(&self) -> &Array2<f64> {
            &self.s0
        }
        fn cfg(&self) -> &RemlConfig {
            &self.cfg
        }
        fn rho(&self) -> &Array1<f64> {
            &self.rho
        }
    }

    // ── n=30, p=5 binomial-logit design-motion gradient tests ────────

    #[test]
    pub(crate) fn binomial_logit_n30_design_moving_gradient_matches_fd() {
        // Pure design-motion: X_τ ≠ 0, S_τ = 0.
        // The IFT correction is essential here: because the family is
        // binomial-logit, the working weights W(η) depend on β̂, so
        // when X moves with ψ, the implicit derivative dβ̂/dψ enters
        // the total Hessian drift.  Without hessian_derivative_correction
        // the analytic gradient would disagree with FD.
        let f = BinomialLogitDesignMotionFixture::new();
        let state = f.state();
        let s_tau = Array2::<f64>::zeros((5, 5));
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            f.x_tau_design.clone(),
            s_tau.clone(),
            None,
            None,
        )
        .expect("design-moving hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_tau_fd = f.fd_directional_gradient(&f.x_tau_design, &s_tau);

        let v_rel = (v_tau_analytic - v_tau_fd).abs() / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Binomial-logit n=30 design-moving gradient mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );
    }

    #[test]
    pub(crate) fn binomial_logit_n30_penalty_only_gradient_matches_fd() {
        // Penalty-only direction: X_τ = 0, S_τ ≠ 0.
        // Serves as a baseline: the IFT correction should still be
        // present (since H depends on β̂ through W(η)), but the
        // explicit X_τ contribution is zero.
        let f = BinomialLogitDesignMotionFixture::new();
        let state = f.state();
        let x_tau = Array2::<f64>::zeros(f.x.raw_dim());
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            x_tau.clone(),
            f.s_tau_penalty.clone(),
            None,
            None,
        )
        .expect("penalty-only hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_tau_fd = f.fd_directional_gradient(&x_tau, &f.s_tau_penalty);

        let v_rel = (v_tau_analytic - v_tau_fd).abs() / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Binomial-logit n=30 penalty-only gradient mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );
    }

    #[test]
    pub(crate) fn binomial_logit_n30_joint_design_penalty_gradient_matches_fd() {
        // Joint direction: both X_τ ≠ 0 and S_τ ≠ 0 simultaneously.
        // This is the hardest case: the analytic gradient must correctly
        // combine the explicit penalty drift, the explicit design drift,
        // and the IFT Hessian-drift correction.
        let f = BinomialLogitDesignMotionFixture::new();
        let state = f.state();
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            f.x_tau_design.clone(),
            f.s_tau_penalty.clone(),
            None,
            None,
        )
        .expect("joint design+penalty hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_tau_fd = f.fd_directional_gradient(&f.x_tau_design, &f.s_tau_penalty);

        let v_rel = (v_tau_analytic - v_tau_fd).abs() / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Binomial-logit n=30 joint design+penalty gradient mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );
    }

    #[test]
    pub(crate) fn binomial_logit_n30_design_moving_hessian_matches_fd() {
        // Hessian-level validation with two τ-directions: one
        // penalty-only and one design-moving.  The ττ Hessian block is
        // the most sensitive test of the IFT correction because errors
        // in the correction accumulate quadratically in the trace term.
        let f = BinomialLogitDesignMotionFixture::new();
        let x_tau_0 = Array2::<f64>::zeros(f.x.raw_dim());
        let s_tau_0 = f.s_tau_penalty.clone();
        let x_tau_1 = f.x_tau_design.clone();
        let s_tau_1 = Array2::<f64>::zeros((5, 5));

        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(0, x_tau_0.clone(), s_tau_0.clone(), None, None)
                .expect("penalty-only direction"),
            DirectionalHyperParam::single_penalty(0, x_tau_1.clone(), s_tau_1.clone(), None, None)
                .expect("design-moving direction"),
        ];

        let state = f.state();
        let mut theta = Array1::<f64>::zeros(f.rho.len() + hyper_dirs.len());
        theta.slice_mut(s![..f.rho.len()]).assign(&f.rho);
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, f.rho.len(), &hyper_dirs)
                .expect("joint cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![f.rho.len().., f.rho.len()..]).to_owned();

        let x_tau_mats = vec![x_tau_0.clone(), x_tau_1.clone()];
        let s_tau_mats = vec![s_tau_0.clone(), s_tau_1.clone()];
        let h_tt_fd = directional_tau_hessian_fd_reference(
            &f.y,
            &f.w,
            &f.x,
            &f.s0,
            &f.cfg,
            &f.rho,
            &hyper_dirs,
            &x_tau_mats,
            &s_tau_mats,
        );

        let num = (&h_tt_analytic - &h_tt_fd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_tt_fd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "Binomial-logit n=30 tau-tau Hessian mismatch: rel={rel:.3e}, \
             analytic={h_tt_analytic:?}, fd={h_tt_fd:?}"
        );
    }

    #[test]
    pub(crate) fn binomial_logit_n30_nonzero_rho_design_moving_gradient_matches_fd() {
        // Validate at a non-trivial smoothing parameter ρ = log(λ) = 1.5,
        // so the penalty term λS is scaled up and the balance between
        // likelihood and penalty is different from ρ=0.
        let f = BinomialLogitDesignMotionFixture::new();
        let rho = array![1.5];
        let s_tau = Array2::<f64>::zeros((5, 5));

        let state = f.state();
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            f.x_tau_design.clone(),
            s_tau.clone(),
            None,
            None,
        )
        .expect("design-moving hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &rho, hyper)
            .expect("analytic directional gradient");

        // FD at the shifted ρ: perturb X, re-solve inner, evaluate cost.
        let h = 2e-5;
        let (state_plus, state_minus) = f.state_perturbed(&f.x_tau_design, &s_tau, h);
        let v_plus = state_plus.compute_cost(&rho).expect("cost+");
        let v_minus = state_minus.compute_cost(&rho).expect("cost-");
        let v_tau_fd = (v_plus - v_minus) / (2.0 * h);

        let v_rel = (v_tau_analytic - v_tau_fd).abs() / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Binomial-logit n=30 rho=1.5 design-moving gradient mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );
    }

    #[test]
    pub(crate) fn binomial_logit_n30_rank_deficient_hessian_matches_cost_fd() {
        // Regression lock for the rank-deficient LAML pseudo-logdet: the cost
        // is the intrinsic `½ log|H_pen|₊` and the trace kernel the spectral
        // `H_pen⁺`, exact for every drift direction (#901). Since #2901 V22
        // both come from one operator on H's identified subspace.
        //
        // The sibling `binomial_logit_n30_design_moving_hessian_matches_fd`
        // passes pre- AND post-fix because its FD reference differentiates
        // the *analytic gradient* — any self-consistent (if wrong) gradient
        // kernel gives a self-consistent Hessian under re-differentiation,
        // so that test cannot distinguish full-space from projected traces.
        // It passed under the buggy kernel because the same leakage entered
        // both sides of the ratio and cancelled.
        //
        // Here we FD-differentiate `compute_cost` TWICE and compare against
        // the analytic Hessian.  Central second differences expose every
        // disagreement between `½ log|U_Sᵀ H U_S|_+` (used by the cost) and
        // `½ tr(G_ε(H) · Ḣ)` / `−½ tr(G_ε Ḣ_i G_ε Ḣ_j)` (the full-space
        // traces that the gradient and Hessian used before the projection
        // fix).  Under the buggy kernel the IFT correction
        // `D_β H[v] = X' diag(c ⊙ X v) X` leaks onto `null(S)` — X's
        // all-ones intercept column sits there — and that leakage enters
        // the analytic Hessian but not the cost's projected logdet.
        //
        // Direction mix chosen to maximise the null-space leakage pathway:
        //   τ_0 = penalty-only (X_τ = 0, S_τ ≠ 0)  → v_0 = H⁻¹(−S_τ β̂) is
        //         concentrated in range(S_+), but `D_β H[v_0]` has rows and
        //         columns on the intercept because `X[:,0] = 1_n`.
        //   τ_1 = design-moving (X_τ ≠ 0 on non-intercept columns, S_τ = 0)
        //         → `v_1` also picks up the intercept via `X'WX_τβ̂`, and
        //         the base drift `X_τᵀWX + XᵀWX_τ` straddles range(S_+) /
        //         null(S).
        // Both pure directions AND the mixed partial load the Schur correction,
        // so any of the three entries can catch a regression.
        let f = BinomialLogitDesignMotionFixture::new();
        let x_tau_0 = Array2::<f64>::zeros(f.x.raw_dim());
        let s_tau_0 = f.s_tau_penalty.clone();
        let x_tau_1 = f.x_tau_design.clone();
        let s_tau_1 = Array2::<f64>::zeros((5, 5));

        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(0, x_tau_0.clone(), s_tau_0.clone(), None, None)
                .expect("penalty-only direction"),
            DirectionalHyperParam::single_penalty(0, x_tau_1.clone(), s_tau_1.clone(), None, None)
                .expect("design-moving direction"),
        ];

        // Analytic Hessian block.
        let state = f.state();
        let mut theta = Array1::<f64>::zeros(f.rho.len() + hyper_dirs.len());
        theta.slice_mut(s![..f.rho.len()]).assign(&f.rho);
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, f.rho.len(), &hyper_dirs)
                .expect("joint cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![f.rho.len().., f.rho.len()..]).to_owned();

        // Cost-level FD reference.  Central second differences give O(h²)
        // accuracy; the step is sized so the physical perturbation on X / S
        // stays near `1e-5` (same scale as the gradient tests).
        const TARGET_PHYSICAL_STEP: f64 = 1e-5;
        let x_tau_mats = [&x_tau_0, &x_tau_1];
        let s_tau_mats = [&s_tau_0, &s_tau_1];
        let steps: [f64; 2] = {
            let mut steps = [0.0; 2];
            for (j, step) in steps.iter_mut().enumerate() {
                let scale = x_tau_mats[j]
                    .iter()
                    .chain(s_tau_mats[j].iter())
                    .fold(0.0_f64, |acc, value| acc.max(value.abs()));
                *step = if scale > 0.0 {
                    TARGET_PHYSICAL_STEP / scale
                } else {
                    TARGET_PHYSICAL_STEP
                };
            }
            steps
        };

        // Evaluate `compute_cost` at `(a · τ_0, b · τ_1)` multipliers.
        let eval_cost = |a: f64, b: f64| -> f64 {
            let x_eval = &f.x
                + &x_tau_mats[0].mapv(|v| a * steps[0] * v)
                + &x_tau_mats[1].mapv(|v| b * steps[1] * v);
            let s_eval = &f.s0
                + &s_tau_mats[0].mapv(|v| a * steps[0] * v)
                + &s_tau_mats[1].mapv(|v| b * steps[1] * v);
            let st = build_logit_state(&f.y, &f.w, &x_eval, &s_eval, &f.cfg);
            st.compute_cost(&f.rho).expect("cost eval")
        };

        let v_00 = eval_cost(0.0, 0.0);
        let v_p0 = eval_cost(1.0, 0.0);
        let v_m0 = eval_cost(-1.0, 0.0);
        let v_0p = eval_cost(0.0, 1.0);
        let v_0m = eval_cost(0.0, -1.0);
        let v_pp = eval_cost(1.0, 1.0);
        let v_pm = eval_cost(1.0, -1.0);
        let v_mp = eval_cost(-1.0, 1.0);
        let v_mm = eval_cost(-1.0, -1.0);

        let h00_fd = (v_p0 - 2.0 * v_00 + v_m0) / (steps[0] * steps[0]);
        let h11_fd = (v_0p - 2.0 * v_00 + v_0m) / (steps[1] * steps[1]);
        let h01_fd = (v_pp - v_pm - v_mp + v_mm) / (4.0 * steps[0] * steps[1]);

        let h_tt_fd = array![[h00_fd, h01_fd], [h01_fd, h11_fd]];

        let num = (&h_tt_analytic - &h_tt_fd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_tt_fd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;

        assert!(
            rel < 3e-3,
            "Binomial-logit n=30 rank-deficient Hessian vs cost-FD mismatch: rel={rel:.3e}, \
             analytic={h_tt_analytic:?}, fd={h_tt_fd:?}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RemlGeometry {
    DenseSpectral,
    SparseExactSpd,
}

trait PenalizedGeometry {
    fn backend_kind(&self) -> GeometryBackendKind;
}

#[derive(Clone)]
pub(crate) enum DerivativeMatrixStorage {
    Dense(Array2<f64>),
    Zero(ZeroDerivativeMatrix),
    Embedded(EmbeddedDerivativeMatrix),
    Implicit(ImplicitDerivativeOp),
    LatentCoord(LatentCoordDerivativeOp),
}

/// Mechanical surface every `DerivativeMatrixStorage` variant must expose so
/// the `HyperDesignDerivative` / `HyperPenaltyDerivative` wrappers can dispatch
/// with a single per-call `storage_dispatch!`. Each backend owns its variant's
/// substantive math; the wrappers contain only one-line routing.
///
/// `design_*` variants treat the backend as an X-style operator (rows index
/// data, columns index coefficients); `penalty_*` variants treat the backend
/// as a square `p×p` penalty in the global coefficient frame. The Embedded
/// case is the only variant whose two views genuinely differ (local rows vs
/// total_dim square), which is why the two role-specific methods both live in
/// one trait rather than two parallel traits.
trait DerivativeStorageBackend {
    fn resident_byte_count(&self) -> usize;
    fn design_nrows(&self) -> usize;
    fn design_ncols(&self) -> usize;
    fn penalty_dim(&self) -> usize;
    fn uses_implicit_storage(&self) -> bool;
    fn any_nonzero(&self) -> bool;
    fn materialize(&self) -> Array2<f64>;
    fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )>;
    fn implicit_axis_count_hint(&self) -> Option<usize>;
    fn design_forward_mul_original(&self, u: &Array1<f64>) -> Result<Array1<f64>, EstimationError>;
    fn design_transpose_mul_original(
        &self,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError>;
    fn design_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError>;
    /// Default materialises through `design_transformed` then `.dot(u)`;
    /// implicit/latent-coordinate backends override with a direct-operator
    /// path that skips the dense materialisation.
    fn design_transformed_forward_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        Ok(self.design_transformed(qs, free_basis_opt)?.dot(u))
    }
    /// Default materialises through `design_transformed` then `.t().dot(v)`;
    /// implicit/latent-coordinate backends override with a direct path.
    fn design_transformed_transpose_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        Ok(self.design_transformed(qs, free_basis_opt)?.t().dot(v))
    }
    fn penalty_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError>;
    fn penalty_scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError>;
}

/// Fans `expr` over the four `DerivativeMatrixStorage` variants in one place
/// so every wrapper method is a single dispatch line — the compiler enforces
/// exhaustiveness here, so adding a new variant produces one hard error at
/// this site rather than a silent miss in any of the (currently 16) ladders.
macro_rules! storage_dispatch {
    ($scrutinee:expr, $backend:ident => $body:expr) => {
        match $scrutinee {
            DerivativeMatrixStorage::Dense($backend) => $body,
            DerivativeMatrixStorage::Zero($backend) => $body,
            DerivativeMatrixStorage::Embedded($backend) => $body,
            DerivativeMatrixStorage::Implicit($backend) => $body,
            DerivativeMatrixStorage::LatentCoord($backend) => $body,
        }
    };
}

#[derive(Clone)]
pub(crate) struct ZeroDerivativeMatrix {
    rows: usize,
    cols: usize,
}

impl ZeroDerivativeMatrix {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
}

/// Which derivative level the implicit operator should compute.
#[derive(Clone, Copy, Debug)]
pub enum ImplicitDerivLevel {
    /// ∂X/∂ψ_d
    First(usize),
    /// ∂²X/∂ψ_d²
    SecondDiag(usize),
    /// ∂²X/∂ψ_d∂ψ_e
    SecondCross(usize, usize),
}

/// Lazy implicit operator storage: delegates matvecs to the
/// `ImplicitDesignPsiDerivative` and materializes dense form only on demand.
#[derive(Clone)]
pub(crate) struct ImplicitDerivativeOp {
    pub(crate) operator: std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
    pub(crate) level: ImplicitDerivLevel,
    pub(crate) global_range: Range<usize>,
    pub(crate) total_dim: usize,
    /// Cached dense materialization (lazy, populated on first call to ops that need the full matrix).
    ///
    /// Rayon-safe: `materialize_local` calls `materialize_first` / `_second_diag`
    /// / `_second_cross` on the implicit basis-derivative operator, which for
    /// streaming bases dispatches `(0..nc).into_par_iter().for_each(...)`. A plain
    /// `std::sync::OnceLock` here would deadlock if `materialize_dense` were first
    /// called concurrently from inside another rayon par_iter — racing workers
    /// would park on the OnceLock's OS condvar, leaving the leader's nested
    /// par_iter without workers. `RayonSafeOnce` runs init lock-free.
    pub(crate) cached_dense: std::sync::Arc<gam_runtime::resource::RayonSafeOnce<Array2<f64>>>,
}

#[derive(Clone)]
pub(crate) struct LatentCoordDerivativeOp {
    pub(crate) operator: std::sync::Arc<gam_terms::basis::LatentCoordDesignDerivative>,
    pub(crate) flat_axis: usize,
    pub(crate) global_range: Range<usize>,
    pub(crate) total_dim: usize,
    pub(crate) cached_dense: std::sync::Arc<gam_runtime::resource::RayonSafeOnce<Array2<f64>>>,
}

impl LatentCoordDerivativeOp {
    pub(crate) fn materialize_local(&self) -> Array2<f64> {
        self.operator.materialize_axis(self.flat_axis).expect(
            "radial scalar evaluation failed during latent-coordinate derivative materialization",
        )
    }

    pub(crate) fn materialize_dense(&self) -> &Array2<f64> {
        self.cached_dense.get_or_compute(|| {
            let local = self.materialize_local();
            let mut out = Array2::<f64>::zeros((local.nrows(), self.total_dim));
            out.slice_mut(s![.., self.global_range.clone()])
                .assign(&local);
            out
        })
    }

    pub(crate) fn nrows(&self) -> usize {
        self.operator.n_data()
    }

    pub(crate) fn ncols(&self) -> usize {
        self.total_dim
    }

    pub(crate) fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        let local = self
            .operator
            .transpose_mul_axis(self.flat_axis, &v.view())
            .expect(
                "radial scalar evaluation failed during latent-coordinate derivative transpose_mul",
            );
        let mut out = Array1::<f64>::zeros(self.total_dim);
        out.slice_mut(s![self.global_range.clone()]).assign(&local);
        out
    }

    pub(crate) fn forward_mul(&self, u: &Array1<f64>) -> Array1<f64> {
        let u_local = u.slice(s![self.global_range.clone()]).to_owned();
        self.operator
            .forward_mul_axis(self.flat_axis, &u_local.view())
            .expect(
                "radial scalar evaluation failed during latent-coordinate derivative forward_mul",
            )
    }
}

impl ImplicitDerivativeOp {
    pub(crate) fn materialize_local(&self) -> Array2<f64> {
        match self.level {
            ImplicitDerivLevel::First(axis) => self.operator.materialize_first(axis).expect(
                "radial scalar evaluation failed during implicit derivative materialization",
            ),
            ImplicitDerivLevel::SecondDiag(axis) => {
                self.operator.materialize_second_diag(axis).expect(
                    "radial scalar evaluation failed during implicit derivative materialization",
                )
            }
            ImplicitDerivLevel::SecondCross(d, e) => {
                self.operator.materialize_second_cross(d, e).expect(
                    "radial scalar evaluation failed during implicit derivative materialization",
                )
            }
        }
    }

    pub(crate) fn materialize_dense(&self) -> &Array2<f64> {
        self.cached_dense.get_or_compute(|| {
            let local = self.materialize_local();
            let mut out = Array2::<f64>::zeros((local.nrows(), self.total_dim));
            out.slice_mut(s![.., self.global_range.clone()])
                .assign(&local);
            out
        })
    }

    pub(crate) fn nrows(&self) -> usize {
        self.operator.n_data()
    }

    pub(crate) fn ncols(&self) -> usize {
        self.total_dim
    }

    pub(crate) fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        let local = match self.level {
            ImplicitDerivLevel::First(axis) => self
                .operator
                .transpose_mul(axis, &v.view())
                .expect("radial scalar evaluation failed during implicit derivative transpose_mul"),
            ImplicitDerivLevel::SecondDiag(axis) => self
                .operator
                .transpose_mul_second_diag(axis, &v.view())
                .expect("radial scalar evaluation failed during implicit derivative transpose_mul"),
            ImplicitDerivLevel::SecondCross(d, e) => self
                .operator
                .transpose_mul_second_cross(d, e, &v.view())
                .expect("radial scalar evaluation failed during implicit derivative transpose_mul"),
        };
        let mut out = Array1::<f64>::zeros(self.total_dim);
        out.slice_mut(s![self.global_range.clone()]).assign(&local);
        out
    }

    pub(crate) fn forward_mul(&self, u: &Array1<f64>) -> Array1<f64> {
        let u_local = u.slice(s![self.global_range.clone()]).to_owned();
        match self.level {
            ImplicitDerivLevel::First(axis) => self
                .operator
                .forward_mul(axis, &u_local.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul"),
            ImplicitDerivLevel::SecondDiag(axis) => self
                .operator
                .forward_mul_second_diag(axis, &u_local.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul"),
            ImplicitDerivLevel::SecondCross(d, e) => self
                .operator
                .forward_mul_second_cross(d, e, &u_local.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul"),
        }
    }
}

#[derive(Clone)]
pub(crate) struct EmbeddedDerivativeMatrix {
    pub(crate) local: Array2<f64>,
    pub(crate) global_range: Range<usize>,
    pub(crate) total_dim: usize,
}

impl EmbeddedDerivativeMatrix {
    pub(crate) fn new(local: Array2<f64>, global_range: Range<usize>, total_dim: usize) -> Self {
        Self {
            local,
            global_range,
            total_dim,
        }
    }
}

impl DerivativeStorageBackend for Array2<f64> {
    fn resident_byte_count(&self) -> usize {
        self.len().saturating_mul(std::mem::size_of::<f64>())
    }
    fn design_nrows(&self) -> usize {
        Array2::nrows(self)
    }
    fn design_ncols(&self) -> usize {
        Array2::ncols(self)
    }
    fn penalty_dim(&self) -> usize {
        Array2::nrows(self)
    }
    fn uses_implicit_storage(&self) -> bool {
        false
    }
    fn any_nonzero(&self) -> bool {
        self.iter().any(|v| *v != 0.0)
    }
    fn materialize(&self) -> Array2<f64> {
        self.clone()
    }
    fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        None
    }
    fn implicit_axis_count_hint(&self) -> Option<usize> {
        None
    }

    fn design_forward_mul_original(&self, u: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        if Array2::ncols(self) != u.len() {
            crate::bail_invalid_estim!(
                "dense hyper design derivative forward_mul_original width mismatch: matrix={}x{}, vector={}",
                Array2::nrows(self),
                Array2::ncols(self),
                u.len()
            );
        }
        Ok(self.dot(u))
    }

    fn design_transpose_mul_original(
        &self,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if Array2::nrows(self) != v.len() {
            crate::bail_invalid_estim!(
                "dense hyper design derivative transpose_mul_original height mismatch: matrix={}x{}, vector={}",
                Array2::nrows(self),
                Array2::ncols(self),
                v.len()
            );
        }
        Ok(self.t().dot(v))
    }

    fn design_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        Ok(gam_linalg::matrix::DenseRightProductView::new(self)
            .with_factor(qs)
            .with_optional_factor(free_basis_opt)
            .materialize())
    }

    fn penalty_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        let mut transformed = qs.t().dot(self).dot(qs);
        if let Some(z) = free_basis_opt {
            transformed = z.t().dot(&transformed).dot(z);
        }
        Ok(transformed)
    }

    fn penalty_scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError> {
        if target.raw_dim() != self.raw_dim() {
            crate::bail_invalid_estim!(
                "dense hyper penalty derivative shape mismatch: target={}x{}, matrix={}x{}",
                target.nrows(),
                target.ncols(),
                Array2::nrows(self),
                Array2::ncols(self)
            );
        }
        target.scaled_add(amp, self);
        Ok(())
    }
}

impl DerivativeStorageBackend for ZeroDerivativeMatrix {
    fn resident_byte_count(&self) -> usize {
        0
    }
    fn design_nrows(&self) -> usize {
        self.rows
    }
    fn design_ncols(&self) -> usize {
        self.cols
    }
    fn penalty_dim(&self) -> usize {
        self.cols
    }
    fn uses_implicit_storage(&self) -> bool {
        false
    }
    fn any_nonzero(&self) -> bool {
        false
    }
    fn materialize(&self) -> Array2<f64> {
        Array2::<f64>::zeros((self.rows, self.cols))
    }
    fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        None
    }
    fn implicit_axis_count_hint(&self) -> Option<usize> {
        None
    }

    fn design_forward_mul_original(&self, u: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        if self.cols != u.len() {
            crate::bail_invalid_estim!(
                "zero hyper design derivative forward_mul_original width mismatch: matrix={}x{}, vector={}",
                self.rows,
                self.cols,
                u.len()
            );
        }
        Ok(Array1::<f64>::zeros(self.rows))
    }

    fn design_transpose_mul_original(
        &self,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if self.rows != v.len() {
            crate::bail_invalid_estim!(
                "zero hyper design derivative transpose_mul_original height mismatch: matrix={}x{}, vector={}",
                self.rows,
                self.cols,
                v.len()
            );
        }
        Ok(Array1::<f64>::zeros(self.cols))
    }

    fn design_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        if self.cols != qs.nrows() {
            crate::bail_invalid_estim!(
                "zero design derivative width mismatch: total_cols={}, qs rows={}",
                self.cols,
                qs.nrows()
            );
        }
        let cols = free_basis_opt.map_or(qs.ncols(), |z| z.ncols());
        Ok(Array2::<f64>::zeros((self.rows, cols)))
    }

    fn design_transformed_forward_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if self.cols != qs.nrows() {
            crate::bail_invalid_estim!(
                "zero design derivative width mismatch: total_cols={}, qs rows={}",
                self.cols,
                qs.nrows()
            );
        }
        let cols = free_basis_opt.map_or(qs.ncols(), |z| z.ncols());
        if u.len() != cols {
            crate::bail_invalid_estim!(
                "zero design derivative transformed forward width mismatch: expected {}, vector={}",
                cols,
                u.len()
            );
        }
        Ok(Array1::<f64>::zeros(self.rows))
    }

    fn design_transformed_transpose_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if self.rows != v.len() {
            crate::bail_invalid_estim!(
                "zero design derivative transpose height mismatch: matrix rows={}, vector={}",
                self.rows,
                v.len()
            );
        }
        if self.cols != qs.nrows() {
            crate::bail_invalid_estim!(
                "zero design derivative width mismatch: total_cols={}, qs rows={}",
                self.cols,
                qs.nrows()
            );
        }
        let cols = free_basis_opt.map_or(qs.ncols(), |z| z.ncols());
        Ok(Array1::<f64>::zeros(cols))
    }

    fn penalty_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        if self.cols != qs.nrows() {
            crate::bail_invalid_estim!(
                "zero penalty derivative width mismatch: total_dim={}, qs rows={}",
                self.cols,
                qs.nrows()
            );
        }
        let cols = free_basis_opt.map_or(qs.ncols(), |z| z.ncols());
        Ok(Array2::<f64>::zeros((cols, cols)))
    }

    fn penalty_scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError> {
        // Zero penalty derivative: `amp · 0 = 0`, so `amp` scales nothing and
        // `target` is left unchanged. Validate it is finite so a bad scale
        // surfaces here rather than silently no-op'ing on a NaN/inf amplitude.
        if !amp.is_finite() {
            crate::bail_invalid_estim!(
                "zero hyper penalty derivative received non-finite amp={amp}"
            );
        }
        if target.nrows() != self.cols || target.ncols() != self.cols {
            crate::bail_invalid_estim!(
                "zero hyper penalty derivative shape mismatch: target={}x{}, expected {}x{}",
                target.nrows(),
                target.ncols(),
                self.cols,
                self.cols
            );
        }
        Ok(())
    }
}

impl DerivativeStorageBackend for EmbeddedDerivativeMatrix {
    fn resident_byte_count(&self) -> usize {
        self.local.len().saturating_mul(std::mem::size_of::<f64>())
    }
    fn design_nrows(&self) -> usize {
        self.local.nrows()
    }
    fn design_ncols(&self) -> usize {
        self.total_dim
    }
    fn penalty_dim(&self) -> usize {
        self.total_dim
    }
    fn uses_implicit_storage(&self) -> bool {
        false
    }
    fn any_nonzero(&self) -> bool {
        self.local.iter().any(|v| *v != 0.0)
    }
    fn materialize(&self) -> Array2<f64> {
        let mut dense = Array2::<f64>::zeros((self.local.nrows(), self.total_dim));
        dense
            .slice_mut(s![.., self.global_range.clone()])
            .assign(&self.local);
        dense
    }
    fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        None
    }
    fn implicit_axis_count_hint(&self) -> Option<usize> {
        None
    }

    fn design_forward_mul_original(&self, u: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        if self.total_dim != u.len() {
            crate::bail_invalid_estim!(
                "embedded hyper design derivative forward_mul_original width mismatch: total_dim={}, vector={}",
                self.total_dim,
                u.len()
            );
        }
        let u_local = u.slice(s![self.global_range.clone()]).to_owned();
        Ok(self.local.dot(&u_local))
    }

    fn design_transpose_mul_original(
        &self,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if self.local.nrows() != v.len() {
            crate::bail_invalid_estim!(
                "embedded hyper design derivative transpose_mul_original height mismatch: local_rows={}, vector={}",
                self.local.nrows(),
                v.len()
            );
        }
        let mut out = Array1::<f64>::zeros(self.total_dim);
        let pulled = self.local.t().dot(v);
        out.slice_mut(s![self.global_range.clone()]).assign(&pulled);
        Ok(out)
    }

    fn design_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        if self.total_dim != qs.nrows() {
            crate::bail_invalid_estim!(
                "embedded design derivative width mismatch: total_cols={}, qs rows={}",
                self.total_dim,
                qs.nrows()
            );
        }
        let qs_local = qs.slice(s![self.global_range.clone(), ..]);
        let mut transformed = self.local.dot(&qs_local);
        if let Some(z) = free_basis_opt {
            transformed = transformed.dot(z);
        }
        Ok(transformed)
    }

    fn penalty_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        if self.total_dim != qs.nrows() {
            crate::bail_invalid_estim!(
                "embedded penalty derivative width mismatch: total_dim={}, qs rows={}",
                self.total_dim,
                qs.nrows()
            );
        }
        let qs_local = qs.slice(s![self.global_range.clone(), ..]);
        let mut transformed = qs_local.t().dot(&self.local).dot(&qs_local);
        if let Some(z) = free_basis_opt {
            transformed = z.t().dot(&transformed).dot(z);
        }
        Ok(transformed)
    }

    fn penalty_scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError> {
        if target.nrows() != self.total_dim || target.ncols() != self.total_dim {
            crate::bail_invalid_estim!(
                "embedded hyper penalty derivative shape mismatch: target={}x{}, expected {}x{}",
                target.nrows(),
                target.ncols(),
                self.total_dim,
                self.total_dim
            );
        }
        target
            .slice_mut(s![self.global_range.clone(), self.global_range.clone()])
            .scaled_add(amp, &self.local);
        Ok(())
    }
}

impl DerivativeStorageBackend for ImplicitDerivativeOp {
    fn resident_byte_count(&self) -> usize {
        0
    }
    fn design_nrows(&self) -> usize {
        self.nrows()
    }
    fn design_ncols(&self) -> usize {
        self.ncols()
    }
    fn penalty_dim(&self) -> usize {
        self.nrows()
    }
    fn uses_implicit_storage(&self) -> bool {
        true
    }
    fn any_nonzero(&self) -> bool {
        true
    }
    fn materialize(&self) -> Array2<f64> {
        self.materialize_dense().clone()
    }
    fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        match self.level {
            ImplicitDerivLevel::First(axis) => Some((self.operator.clone(), axis)),
            _ => None,
        }
    }
    fn implicit_axis_count_hint(&self) -> Option<usize> {
        Some(self.operator.n_axes())
    }

    fn design_forward_mul_original(&self, u: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        if self.ncols() != u.len() {
            crate::bail_invalid_estim!(
                "implicit hyper design derivative forward_mul_original width mismatch: operator_cols={}, vector={}",
                self.ncols(),
                u.len()
            );
        }
        Ok(self.forward_mul(u))
    }

    fn design_transpose_mul_original(
        &self,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if self.nrows() != v.len() {
            crate::bail_invalid_estim!(
                "implicit hyper design derivative transpose_mul_original height mismatch: operator_rows={}, vector={}",
                self.nrows(),
                v.len()
            );
        }
        Ok(self.transpose_mul(v))
    }

    fn design_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        let dense = self.materialize_dense();
        Ok(gam_linalg::matrix::DenseRightProductView::new(dense)
            .with_factor(qs)
            .with_optional_factor(free_basis_opt)
            .materialize())
    }

    fn design_transformed_forward_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let mut right = if let Some(z) = free_basis_opt {
            z.dot(u)
        } else {
            u.clone()
        };
        right = qs.dot(&right);
        Ok(self.forward_mul(&right))
    }

    fn design_transformed_transpose_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let mut pulled = qs.t().dot(&self.transpose_mul(v));
        if let Some(z) = free_basis_opt {
            pulled = z.t().dot(&pulled);
        }
        Ok(pulled)
    }

    fn penalty_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        let dense = self.materialize_dense();
        let mut transformed = qs.t().dot(dense).dot(qs);
        if let Some(z) = free_basis_opt {
            transformed = z.t().dot(&transformed).dot(z);
        }
        Ok(transformed)
    }

    fn penalty_scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError> {
        let dense = self.materialize_dense();
        if target.raw_dim() != dense.raw_dim() {
            crate::bail_invalid_estim!(
                "implicit hyper penalty derivative shape mismatch: target={}x{}, matrix={}x{}",
                target.nrows(),
                target.ncols(),
                dense.nrows(),
                dense.ncols()
            );
        }
        target.scaled_add(amp, dense);
        Ok(())
    }
}

impl DerivativeStorageBackend for LatentCoordDerivativeOp {
    fn resident_byte_count(&self) -> usize {
        0
    }
    fn design_nrows(&self) -> usize {
        self.nrows()
    }
    fn design_ncols(&self) -> usize {
        self.ncols()
    }
    fn penalty_dim(&self) -> usize {
        self.nrows()
    }
    fn uses_implicit_storage(&self) -> bool {
        true
    }
    fn any_nonzero(&self) -> bool {
        true
    }
    fn materialize(&self) -> Array2<f64> {
        self.materialize_dense().clone()
    }
    fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        None
    }
    fn implicit_axis_count_hint(&self) -> Option<usize> {
        Some(self.operator.n_axes())
    }

    fn design_forward_mul_original(&self, u: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        if self.ncols() != u.len() {
            crate::bail_invalid_estim!(
                "latent-coordinate hyper design derivative forward_mul_original width mismatch: operator_cols={}, vector={}",
                self.ncols(),
                u.len()
            );
        }
        Ok(self.forward_mul(u))
    }

    fn design_transpose_mul_original(
        &self,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if self.nrows() != v.len() {
            crate::bail_invalid_estim!(
                "latent-coordinate hyper design derivative transpose_mul_original height mismatch: operator_rows={}, vector={}",
                self.nrows(),
                v.len()
            );
        }
        Ok(self.transpose_mul(v))
    }

    fn design_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        let dense = self.materialize_dense();
        Ok(gam_linalg::matrix::DenseRightProductView::new(dense)
            .with_factor(qs)
            .with_optional_factor(free_basis_opt)
            .materialize())
    }

    fn design_transformed_forward_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let mut right = if let Some(z) = free_basis_opt {
            z.dot(u)
        } else {
            u.clone()
        };
        right = qs.dot(&right);
        Ok(self.forward_mul(&right))
    }

    fn design_transformed_transpose_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let mut pulled = qs.t().dot(&self.transpose_mul(v));
        if let Some(z) = free_basis_opt {
            pulled = z.t().dot(&pulled);
        }
        Ok(pulled)
    }

    fn penalty_transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        let dense = self.materialize_dense();
        let mut transformed = qs.t().dot(dense).dot(qs);
        if let Some(z) = free_basis_opt {
            transformed = z.t().dot(&transformed).dot(z);
        }
        Ok(transformed)
    }

    fn penalty_scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError> {
        let dense = self.materialize_dense();
        if target.raw_dim() != dense.raw_dim() {
            crate::bail_invalid_estim!(
                "latent-coordinate hyper penalty derivative shape mismatch: target={}x{}, matrix={}x{}",
                target.nrows(),
                target.ncols(),
                dense.nrows(),
                dense.ncols()
            );
        }
        target.scaled_add(amp, dense);
        Ok(())
    }
}

#[derive(Clone)]
pub struct HyperDesignDerivative {
    pub(crate) storage: DerivativeMatrixStorage,
}

impl HyperDesignDerivative {
    pub fn zero(nrows: usize, ncols: usize) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Zero(ZeroDerivativeMatrix::new(nrows, ncols)),
        }
    }

    pub fn from_embedded(
        local: Array2<f64>,
        global_range: Range<usize>,
        total_cols: usize,
    ) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Embedded(EmbeddedDerivativeMatrix::new(
                local,
                global_range,
                total_cols,
            )),
        }
    }

    pub fn from_implicit(
        operator: std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        level: ImplicitDerivLevel,
        global_range: Range<usize>,
        total_cols: usize,
    ) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Implicit(ImplicitDerivativeOp {
                operator,
                level,
                global_range,
                total_dim: total_cols,
                cached_dense: std::sync::Arc::new(gam_runtime::resource::RayonSafeOnce::new()),
            }),
        }
    }

    pub fn from_latent_coord(
        operator: std::sync::Arc<gam_terms::basis::LatentCoordDesignDerivative>,
        flat_axis: usize,
        global_range: Range<usize>,
        total_cols: usize,
    ) -> Self {
        Self {
            storage: DerivativeMatrixStorage::LatentCoord(LatentCoordDerivativeOp {
                operator,
                flat_axis,
                global_range,
                total_dim: total_cols,
                cached_dense: std::sync::Arc::new(gam_runtime::resource::RayonSafeOnce::new()),
            }),
        }
    }

    pub(crate) fn resident_byte_count(&self) -> usize {
        storage_dispatch!(&self.storage, b => b.resident_byte_count())
    }

    pub(crate) fn nrows(&self) -> usize {
        storage_dispatch!(&self.storage, b => b.design_nrows())
    }

    pub(crate) fn ncols(&self) -> usize {
        storage_dispatch!(&self.storage, b => b.design_ncols())
    }

    pub(crate) fn uses_implicit_storage(&self) -> bool {
        storage_dispatch!(&self.storage, b => b.uses_implicit_storage())
    }

    /// The global columns this derivative can be nonzero in, or `None` for a
    /// dense matrix, which records no support.
    pub(crate) fn column_support(&self) -> Option<Range<usize>> {
        match &self.storage {
            DerivativeMatrixStorage::Dense(_) => None,
            DerivativeMatrixStorage::Zero(_) => Some(0..0),
            DerivativeMatrixStorage::Embedded(backend) => Some(backend.global_range.clone()),
            DerivativeMatrixStorage::Implicit(backend) => Some(backend.global_range.clone()),
            DerivativeMatrixStorage::LatentCoord(backend) => Some(backend.global_range.clone()),
        }
    }

    /// `X_τ[rows, ·] · factor`. Stored values multiply directly, and an implicit
    /// derivative forms its rows one chunk at a time from the term's radial
    /// jets. `None` only for a latent-coordinate derivative, whose rows exist
    /// only through its matvecs.
    pub(crate) fn dense_rows_times(
        &self,
        rows: Range<usize>,
        factor: &Array2<f64>,
    ) -> Option<Array2<f64>> {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => Some(gam_linalg::faer_ndarray::fast_ab(
                &dense.slice(s![rows, ..]),
                factor,
            )),
            DerivativeMatrixStorage::Zero(_) => {
                Some(Array2::<f64>::zeros((rows.len(), factor.ncols())))
            }
            DerivativeMatrixStorage::Embedded(backend) => Some(gam_linalg::faer_ndarray::fast_ab(
                &backend.local.slice(s![rows, ..]),
                &factor.slice(s![backend.global_range.clone(), ..]),
            )),
            DerivativeMatrixStorage::Implicit(backend) => {
                let local = match backend.level {
                    ImplicitDerivLevel::First(axis) => backend.operator.row_chunk_first(axis, rows),
                    ImplicitDerivLevel::SecondDiag(axis) => {
                        backend.operator.row_chunk_second_diag(axis, rows)
                    }
                    ImplicitDerivLevel::SecondCross(d, e) => {
                        backend.operator.row_chunk_second_cross(d, e, rows)
                    }
                }
                .expect("radial scalar evaluation failed during implicit derivative row chunk");
                Some(gam_linalg::faer_ndarray::fast_ab(
                    &local,
                    &factor.slice(s![backend.global_range.clone(), ..]),
                ))
            }
            DerivativeMatrixStorage::LatentCoord(_) => None,
        }
    }

    pub(crate) fn materialize(&self) -> Array2<f64> {
        storage_dispatch!(&self.storage, b => b.materialize())
    }

    pub(crate) fn any_nonzero(&self) -> bool {
        storage_dispatch!(&self.storage, b => b.any_nonzero())
    }

    pub(crate) fn forward_mul_original(
        &self,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        storage_dispatch!(&self.storage, b => b.design_forward_mul_original(u))
    }

    pub(crate) fn transpose_mul_original(
        &self,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        storage_dispatch!(&self.storage, b => b.design_transpose_mul_original(v))
    }

    pub(crate) fn transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        storage_dispatch!(&self.storage, b => b.design_transformed(qs, free_basis_opt))
    }

    pub(crate) fn transformed_forward_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        storage_dispatch!(&self.storage, b => b.design_transformed_forward_mul(qs, free_basis_opt, u))
    }

    pub(crate) fn transformed_transpose_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        storage_dispatch!(&self.storage, b => b.design_transformed_transpose_mul(qs, free_basis_opt, v))
    }

    /// If this derivative uses implicit storage at the first-derivative level,
    /// return the shared implicit operator and the axis index.
    ///
    /// Returns `None` for dense/embedded storage or for second-derivative levels.
    pub(crate) fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        storage_dispatch!(&self.storage, b => b.implicit_first_axis_info())
    }

    pub(crate) fn implicit_axis_count_hint(&self) -> Option<usize> {
        storage_dispatch!(&self.storage, b => b.implicit_axis_count_hint())
    }
}

impl From<Array2<f64>> for HyperDesignDerivative {
    fn from(value: Array2<f64>) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Dense(value),
        }
    }
}

#[derive(Clone)]
pub struct HyperPenaltyDerivative {
    pub(crate) storage: DerivativeMatrixStorage,
}

impl HyperPenaltyDerivative {
    pub fn from_embedded(local: Array2<f64>, global_range: Range<usize>, total_dim: usize) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Embedded(EmbeddedDerivativeMatrix::new(
                local,
                global_range,
                total_dim,
            )),
        }
    }

    pub(crate) fn resident_byte_count(&self) -> usize {
        storage_dispatch!(&self.storage, b => b.resident_byte_count())
    }

    pub(crate) fn nrows(&self) -> usize {
        storage_dispatch!(&self.storage, b => b.penalty_dim())
    }

    pub(crate) fn ncols(&self) -> usize {
        self.nrows()
    }

    pub(crate) fn scaled_materialize(&self, amp: f64) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.nrows(), self.ncols()));
        self.scaled_add_to(&mut out, amp)
            .expect("scaled materialize uses matching target shape");
        out
    }

    pub(crate) fn transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        storage_dispatch!(&self.storage, b => b.penalty_transformed(qs, free_basis_opt))
    }

    pub(crate) fn scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError> {
        storage_dispatch!(&self.storage, b => b.penalty_scaled_add_to(target, amp))
    }
}

impl From<Array2<f64>> for HyperPenaltyDerivative {
    fn from(value: Array2<f64>) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Dense(value),
        }
    }
}

#[derive(Clone)]
pub struct PenaltyDerivativeComponent {
    pub penalty_index: usize,
    pub matrix: HyperPenaltyDerivative,
}

#[derive(Clone)]
pub struct DirectionalHyperParam {
    pub(crate) x_tau_original: HyperDesignDerivative,
    // Canonical penalty representation: every tau direction is decomposed into
    // base-penalty derivatives. There is no separate "assembled total" path.
    pub(crate) penalty_first_components: Vec<PenaltyDerivativeComponent>,
    // Optional pairwise second hyper-derivatives against all tau directions.
    // If provided, each vector must have length psi_dim and hold an optional
    // X_{tau_i,tau_j} entry in original coordinates.
    pub(crate) x_tau_tau_original: Option<Vec<Option<HyperDesignDerivative>>>,
    // Pairwise second derivatives are stored in the same canonical base-penalty
    // decomposition as the first derivatives.
    pub(crate) penaltysecond_components: Option<Vec<Option<Vec<PenaltyDerivativeComponent>>>>,
    pub(crate) penaltysecond_component_provider: Option<
        std::sync::Arc<
            dyn Fn(usize) -> Result<Option<Vec<PenaltyDerivativeComponent>>, EstimationError>
                + Send
                + Sync
                + 'static,
        >,
    >,
    /// Whether this coordinate is penalty-like (B_i = ∂H/∂τ_i is PSD).
    /// True for τ (penalty scaling) coordinates; false for ψ (design-moving,
    /// anisotropic length-scale) coordinates. Controls EFS eligibility.
    pub(crate) is_penalty_like: bool,
}

impl DirectionalHyperParam {
    pub(crate) fn resident_byte_count(&self) -> usize {
        let mut bytes = self.x_tau_original.resident_byte_count();
        for component in &self.penalty_first_components {
            bytes = bytes.saturating_add(component.matrix.resident_byte_count());
        }
        if let Some(entries) = self.x_tau_tau_original.as_ref() {
            for entry in entries.iter().flatten() {
                bytes = bytes.saturating_add(entry.resident_byte_count());
            }
        }
        if let Some(rows) = self.penaltysecond_components.as_ref() {
            for components in rows.iter().flatten() {
                for component in components {
                    bytes = bytes.saturating_add(component.matrix.resident_byte_count());
                }
            }
        }
        bytes
    }

    pub(crate) fn canonicalize_penalty_components(
        components: Vec<(usize, HyperPenaltyDerivative)>,
    ) -> Result<Vec<PenaltyDerivativeComponent>, EstimationError> {
        let mut out: Vec<PenaltyDerivativeComponent> = Vec::with_capacity(components.len());
        for (penalty_index, matrix) in components {
            if out.iter().any(|c| c.penalty_index == penalty_index) {
                crate::bail_invalid_estim!(
                    "duplicate penalty derivative component for penalty {}",
                    penalty_index
                );
            }
            out.push(PenaltyDerivativeComponent {
                penalty_index,
                matrix,
            });
        }
        Ok(out)
    }

    pub fn new_compact(
        x_tau_original: HyperDesignDerivative,
        penalty_first_components: Vec<(usize, HyperPenaltyDerivative)>,
        x_tau_tau_original: Option<Vec<Option<HyperDesignDerivative>>>,
        penaltysecond_components: Option<Vec<Option<Vec<(usize, HyperPenaltyDerivative)>>>>,
    ) -> Result<Self, EstimationError> {
        let is_penalty_like = !x_tau_original.any_nonzero();
        let penalty_first_components =
            Self::canonicalize_penalty_components(penalty_first_components)?;
        let penaltysecond_components = match penaltysecond_components {
            Some(rows) => {
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    out.push(match row {
                        Some(components) => {
                            Some(Self::canonicalize_penalty_components(components)?)
                        }
                        None => None,
                    });
                }
                Some(out)
            }
            None => None,
        };
        Ok(Self {
            x_tau_original,
            penalty_first_components,
            x_tau_tau_original,
            penaltysecond_components,
            penaltysecond_component_provider: None,
            is_penalty_like,
        })
    }

    /// Mark this coordinate as non-penalty-like (design-moving).
    /// EFS will skip it; use Newton/BFGS for these coordinates.
    pub fn not_penalty_like(mut self) -> Self {
        self.is_penalty_like = false;
        self
    }

    pub fn with_penaltysecond_component_provider(
        mut self,
        provider: std::sync::Arc<
            dyn Fn(usize) -> Result<Option<Vec<PenaltyDerivativeComponent>>, EstimationError>
                + Send
                + Sync
                + 'static,
        >,
    ) -> Self {
        self.penaltysecond_component_provider = Some(provider);
        self
    }

    pub(crate) fn x_tau_dense(&self) -> Array2<f64> {
        self.x_tau_original.materialize()
    }

    pub(crate) fn transformed_x_tau(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        self.x_tau_original.transformed(qs, free_basis_opt)
    }

    pub(crate) fn x_tau_tau_entry_at(&self, j: usize) -> Option<HyperDesignDerivative> {
        self.x_tau_tau_original
            .as_ref()
            .and_then(|rows| rows.get(j))
            .and_then(|entry| entry.clone())
    }

    /// Whether this coordinate's design derivative uses implicit storage at the
    /// first-derivative level.
    pub(crate) fn has_implicit_operator(&self) -> bool {
        self.x_tau_original.uses_implicit_storage()
    }

    /// Extract the implicit design derivative operator and axis, if available.
    pub(crate) fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        self.x_tau_original.implicit_first_axis_info()
    }

    pub(crate) fn implicit_axis_count_hint(&self) -> Option<usize> {
        self.x_tau_original.implicit_axis_count_hint()
    }

    pub(crate) fn penalty_first_components(&self) -> &[PenaltyDerivativeComponent] {
        &self.penalty_first_components
    }

    pub(crate) fn penalty_total_at(
        &self,
        rho: &Array1<f64>,
        p: usize,
    ) -> Result<Array2<f64>, EstimationError> {
        let mut out = Array2::<f64>::zeros((p, p));
        for component in &self.penalty_first_components {
            if component.matrix.nrows() != p || component.matrix.ncols() != p {
                crate::bail_invalid_estim!(
                    "S_tau shape mismatch for penalty {}: expected {}x{}, got {}x{}",
                    component.penalty_index,
                    p,
                    p,
                    component.matrix.nrows(),
                    component.matrix.ncols()
                );
            }
            if component.penalty_index >= rho.len() {
                crate::bail_invalid_estim!(
                    "penalty_index {} out of bounds for rho dimension {}",
                    component.penalty_index,
                    rho.len()
                );
            }
            let lambda = gam_problem::checked_exp_log_strength(rho[component.penalty_index])
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            component.matrix.scaled_add_to(&mut out, lambda)?;
        }
        Ok(out)
    }

    pub(crate) fn penaltysecond_components_for(
        &self,
        j: usize,
    ) -> Result<Option<Vec<PenaltyDerivativeComponent>>, EstimationError> {
        if let Some(components) = self
            .penaltysecond_components
            .as_ref()
            .and_then(|rows| rows.get(j))
            .and_then(|row| row.clone())
        {
            return Ok(Some(components));
        }
        if let Some(provider) = self.penaltysecond_component_provider.as_ref() {
            return provider(j);
        }
        Ok(None)
    }

    pub(crate) fn penaltysecond_componentrows(
        &self,
    ) -> Option<&[Option<Vec<PenaltyDerivativeComponent>>]> {
        self.penaltysecond_components.as_deref()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SparseRemlDecision {
    pub(crate) geometry: RemlGeometry,
    pub(crate) reason: &'static str,
    pub(crate) p: usize,
    /// Design nonzeros, or `None` when the design has no sparse representation
    /// to count them from. Not the same statement as "the design has none":
    /// `design_not_sparse` fires whenever `as_sparse()` returns `None`, which
    /// includes a lazy operator wrapping a genuinely sparse matrix — and this
    /// field used to render that case as `nnz_x=0`, a measurement-shaped zero
    /// for a design measured at 583,674 nonzeros one gate later.
    pub(crate) nnz_x: Option<usize>,
    pub(crate) nnz_h_upper_est: Option<usize>,
    pub(crate) density_h_upper_est: Option<f64>,
}

#[derive(Clone)]
pub(crate) struct SparseExactEvalData {
    pub(crate) factor: Arc<SparseExactFactor>,
    pub(crate) takahashi: Option<Arc<gam_linalg::sparse_exact::TakahashiInverse>>,
    /// The upper-triangular penalized Hessian `factor` factors, so trace
    /// kernels can read its sparsity pattern.
    pub(crate) hessian: Arc<faer::sparse::SparseColMat<usize, f64>>,
    pub(crate) logdet_h: f64,
    pub(crate) logdet_s_pos: f64,
    pub(crate) penalty_rank: usize,
    pub(crate) det1_values: Arc<Array1<f64>>,
}

#[derive(Clone)]
pub struct FirthDenseOperator {
    // Exact Firth/Jeffreys objects on the identifiable subspace.
    //
    // Let X in R^{n×p} potentially be rank-deficient with rank r.
    // With optional fixed observation weights a_i >= 0 we define A = diag(a),
    // choose an orthonormal coefficient-space basis Q for the identifiable
    // subspace of A^{1/2} X, and set:
    //   X_r := A^{1/2} X Q          (A = I when no fixed observation weights),
    //   W   := diag(w), with w_i = mu_i (1 - mu_i), 0 < w_i <= 1/4 for finite logit eta,
    //   I_r := X_rᵀ W X_r,
    //   S_r := X_rᵀ X_r.
    //
    // Firth term is represented as:
    //   Phi(beta) = 0.5 log |I_r(beta)| - 0.5 log |S_r|,
    // which is exactly
    //   0.5 log |Uᵀ W U|
    // for the canonical orthonormalized identifiable design
    //   U = X_r S_r^{-1/2}.
    // This removes the raw-basis term from explicit reduced designs while
    // keeping the same identifiable-subspace hat matrix and beta derivatives,
    // because S_r is fixed with respect to beta.
    //
    // Mapping back to the full p-space uses:
    //   I_+^dagger = Q I_r^{-1} Qᵀ.
    //
    // We store reduced-space factors so all derivatives can be evaluated exactly
    // without materializing dense n×n matrices M = X K Xᵀ or P = M⊙M.
    pub(crate) x_dense: Array2<f64>,
    pub(crate) x_dense_t: Array2<f64>,
    // Orthonormal coefficient-space basis for the identifiable subspace,
    // built from the retained eigenspace of (A^{1/2} X)ᵀ(A^{1/2} X).
    pub(crate) q_basis: Array2<f64>,
    // Reduced identifiable design. With fixed observation weights a_i this is
    // diag(sqrt(a_i)) X Q; otherwise it is X Q.
    pub(crate) x_reduced: Array2<f64>,
    // Optional fixed case-weight square roots used when the Jeffreys/Firth
    // operator is formed from Xᵀ diag(case_weight ⊙ w(η)) X rather than
    // Xᵀ diag(w(η)) X. The exact directional tau derivatives must project and
    // row-scale with the same weights so the reduced Fisher, hat diagonals,
    // and tau kernels all live on one consistent identifiable subspace.
    pub(crate) observation_weight_sqrt: Option<Array1<f64>>,
    // I_r^{-1}
    pub(crate) k_reduced: Array2<f64>,
    // diag(S_r^{-1}) with S_r = X_rᵀ X_r. In the current canonical reduced
    // basis this completely characterizes the metric inverse, because Q
    // diagonalizes the design Gram. It is used to remove the reduced-coordinate
    // basis term from Phi_tau when the design moves.
    pub(crate) x_metric_reduced_inv_diag: Array1<f64>,
    // 0.5 (log|I_r| - log|S_r|) at the current eta.
    pub(crate) half_log_det: f64,
    // h = diag(M), M = X_r K_r X_r'
    pub(crate) h_diag: Array1<f64>,
    // Logistic Fisher-weight eta-derivatives: w', w'', w''', w'''' as n-vectors.
    pub(crate) w: Array1<f64>,
    pub(crate) w1: Array1<f64>,
    pub(crate) w2: Array1<f64>,
    pub(crate) w3: Array1<f64>,
    pub(crate) w4: Array1<f64>,
    // B = diag(w') X used in D Hphi and D^2 Hphi contractions.
    pub(crate) b_base: Array2<f64>,
    // Cached invariant contraction P*B where P = (X_r K_r X_r') ⊙ (X_r K_r X_r').
    // This avoids recomputing the same O(n r^2 p) block in every directional call.
    pub(crate) p_b_base: Array2<f64>,
}

/// β-independent (design-only) factor of the Firth/Jeffreys operator.
///
/// Everything stored here depends ONLY on the fixed design `X` and the fixed
/// prior/observation weights `a_i` — NOT on the current linear predictor `η`
/// (i.e. NOT on β). For a single inner PIRLS solve the design and prior weights
/// are constant while `η` changes every Newton iteration, so this factor can be
/// built once per solve and reused, hoisting the O(n·p²) Gram, the O(p³)
/// identifiable-subspace eigendecomposition, and the two n×p design clones out
/// of the per-iteration hot path (#1575).
///
/// The β-dependent remainder (Fisher weights `w(η)`, reduced Fisher
/// `I_r = X_rᵀ W X_r`, its inverse `K_r`, the hat diagonal `h`, and the
/// half-log-determinant) is rebuilt per iteration from this factor via
/// [`FirthDenseOperator::build_from_design_factor`]. The design-only work stays
/// hoisted while the full per-state operator supplies both PIRLS diagnostics
/// and the exact Jeffreys coefficient curvature.
#[derive(Clone)]
pub(crate) struct FirthDesignFactor {
    // Raw design and its transpose (the operator stores owned copies).
    pub(crate) x_dense: Array2<f64>,
    pub(crate) x_dense_t: Array2<f64>,
    // Orthonormal identifiable-subspace basis Q of (A^{1/2} X)ᵀ(A^{1/2} X).
    pub(crate) q_basis: Array2<f64>,
    // Reduced identifiable design X_r = A^{1/2} X Q.
    pub(crate) x_reduced: Array2<f64>,
    // Fixed case-weight square roots (sqrt(a_i)), if any.
    pub(crate) observation_weight_sqrt: Option<Array1<f64>>,
    // Retained positive spectrum of the design Gram = S_r diagonal.
    pub(crate) metric_spectrum: Array1<f64>,
    // diag(S_r^{-1}); precomputed reciprocal of `metric_spectrum`.
    pub(crate) x_metric_reduced_inv_diag: Array1<f64>,
    // rank r = ncols(q_basis); n = nrows(x_dense).
    pub(crate) r: usize,
    pub(crate) n: usize,
}

/// The Jeffreys log-density `½ log|I(β)|` of a fixed design under one inverse
/// link, as a function of `η = Xβ` alone.
///
/// It holds the β-independent [`FirthDesignFactor`] and evaluates the same
/// identifiable-subspace value [`FirthDenseOperator`] carries, without the
/// Fisher inverse, hat diagonal, and weight derivatives only the gradient
/// needs: the per-state cost is `X_rᵀ W X_r` and one factorization. A
/// Metropolis ratio `|I(β')|^½ / |I(β)|^½` needs nothing more.
pub struct JeffreysHalfLogDet {
    pub(crate) factor: FirthDesignFactor,
    pub(crate) link: InverseLink,
}

#[derive(Clone)]
pub(crate) struct FirthDirection {
    pub(crate) deta: Array1<f64>,
    pub(crate) g_u_reduced: Array2<f64>,
    pub(crate) a_u_reduced: Array2<f64>,
    pub(crate) dh: Array1<f64>,
    // B_u = diag(w'' ⊙ δη_u) X is represented by the row-scaling vector only.
    pub(crate) b_uvec: Array1<f64>,
}

/// Shared contractions of `D H_φ[u]` against one symmetric `Π`, built by
/// `FirthDenseOperator::hphi_direction_trace_kernel`.
pub(crate) struct FirthHphiTraceKernel {
    /// `ℓ = diag(X Π Xᵀ)`.
    pub(crate) leverage: Array1<f64>,
    /// `v = ((M⊙M)⊙(X Π Xᵀ)) w'`.
    pub(crate) hadamard_w1: Array1<f64>,
    /// `R = Zᵀ diag(w') (M⊙X Π Xᵀ) diag(w') Z` in reduced coordinates.
    pub(crate) reduced: Array2<f64>,
}

#[derive(Clone)]
pub(crate) struct FirthTauPartialKernel {
    pub(super) deta_partial: Array1<f64>,
    pub(crate) dotw1: Array1<f64>,
    pub(crate) dotw2: Array1<f64>,
    pub(crate) dot_h_partial: Array1<f64>,
    // Reduced design drift X_{tau,r} = X_tau Q used in exact design-moving
    // Hadamard-Gram contractions.
    pub(crate) x_tau_reduced: Array2<f64>,
    pub(super) dot_i_partial: Array2<f64>,
    // Reduced Fisher inverse drift:
    //   dot(K_r) = -K_r dot(I_r) K_r
    // where dot(I_r) includes explicit X_tau and weight drift at beta-fixed.
    pub(crate) dot_k_reduced: Array2<f64>,
}

#[derive(Clone)]
pub(crate) struct FirthTauExactKernel {
    pub(crate) gphi_tau: Array1<f64>,
    pub(crate) phi_tau_partial: f64,
    pub(crate) tau_kernel: Option<FirthTauPartialKernel>,
}

/// Pair-level (τ_i × τ_j) exact Firth bundle at fixed β.
///
/// Mirrors `FirthTauExactKernel` but for the 2nd-order cross
/// derivatives:
///   Phi_{τ_i τ_j}|β  (scalar, `phi_tau_tau_partial`)
///   (gphi)_{τ_i τ_j}|β (p-vector, `gphi_tau_tau`)
///
/// Carries an optional `tau_tau_kernel` so pair callbacks can chain
/// into Primitive A (`hphi_tau_tau_partial_apply`) for the operator-
/// valued Hessian 2nd drift without recomputing shared reduced Grams.
///
#[derive(Clone)]
pub(crate) struct FirthTauTauExactKernel {
    pub(super) phi_tau_tau_partial: f64,
    pub(super) gphi_tau_tau: Array1<f64>,
    pub(super) tau_tau_kernel: Option<FirthTauTauPartialKernel>,
}

/// Prepared state for `∂²H_φ/∂τ_i ∂τ_j |_β` (Primitive A).
///
/// Carries both τ-direction reduced designs, their η̇ vectors, and the
/// reduced-coordinate drifts (İ, K̇, ḣ) for i and j so the apply step can
/// form M̈_{ij}, K̈_{ij}, ḧ_{ij}, Γ̈_{ij}, and B̈_{ij} matrix-free.  Fields
/// are filled in by 13b; kept with a neutral internal shape so downstream
/// pair callbacks can hold the kernel across the pair dispatch.
///
/// Wired into the pair-callback's `b_operator` via
/// `FirthAugmentedPairHyperOperator`, and produced by both
/// `hphi_tau_tau_partial_prepare_from_partials` and
/// `exact_tau_tau_kernel` (the scalar/p-vector companion).
#[derive(Clone, Default)]
pub(crate) struct FirthTauTauPartialKernel {
    pub(super) x_tau_i_reduced: Array2<f64>,
    pub(super) x_tau_j_reduced: Array2<f64>,
    pub(super) deta_i_partial: Array1<f64>,
    pub(super) deta_j_partial: Array1<f64>,
    pub(super) dot_h_i_partial: Array1<f64>,
    pub(super) dot_h_j_partial: Array1<f64>,
    pub(super) dot_k_i_reduced: Array2<f64>,
    pub(super) dot_k_j_reduced: Array2<f64>,
    pub(super) dot_i_i_partial: Array2<f64>,
    pub(super) dot_i_j_partial: Array2<f64>,
    pub(super) x_tau_tau_reduced: Option<Array2<f64>>,
    pub(super) deta_ij_partial: Option<Array1<f64>>,
}

/// Prepared state for `D_β((H_φ)_τ|_β)[v]` (Primitive B).
///
/// Carries the τ-kernel pieces (x_tau_reduced, İ, K̇, ḣ), the
/// β-direction quantities (δη_v, A_v, dh_v, b-chain), and the mixed
/// β-τ pieces (D_β(K̇_τ)[v], D_β(ḣ_τ)[v], δη_{τ,v}) so the apply
/// step collapses to the 9-term β-τ expansion without recomputing
/// shared reduced Grams.
#[derive(Clone, Default)]
pub(crate) struct FirthTauBetaPartialKernel {
    pub(super) x_tau_reduced: Array2<f64>,
    pub(super) deta_partial: Array1<f64>,
    pub(super) dot_h_partial: Array1<f64>,
    pub(super) dot_i_partial: Array2<f64>,
    pub(super) dot_k_reduced: Array2<f64>,
    pub(super) deta_v: Array1<f64>,
    pub(super) deta_tau_v: Array1<f64>,
    pub(super) a_v_reduced: Array2<f64>,
    pub(super) dh_v: Array1<f64>,
    pub(super) b_vvec: Array1<f64>,
    pub(super) d_beta_dot_k: Array2<f64>,
    pub(super) d_beta_dot_h: Array1<f64>,
}

/// The predicate a dense criterion builder decided `½log|H|₊`'s rank with
/// (#2959 D1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CriterionRankPredicate {
    /// `H`'s identified subspace: the eigenvalues above `p·ε·‖H‖₂`, never fewer
    /// than `rank(S_λ)` ([`reml_outer_engine::DenseSpectralOperator::identified_rank`]).
    IdentifiedSubspace,
    /// A structural rank taken from the unscaled design and penalty roots (Firth).
    StructuralRank,
    /// The root `B = [√W·X; √λ_k R_k]`'s singular values, every mode priced
    /// (#2644, gam#2735).
    RootScale,
}

/// The coefficient frame a criterion rank decision was taken in (#2959 D1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CriterionFrame {
    /// PIRLS's transformed frame: the matrix is the bundle's `h_total`.
    Transformed,
    /// The original frame: the matrix is `Qs·h_total·Qsᵀ` with the transformed
    /// barrier swapped for the original-basis one.
    Original,
}

/// The rank decision the dense criterion priced `½log|H|₊` with at one
/// evaluation point, published by the builder that priced it (#2959 D1).
///
/// The identified-rank certificate (#2901 V22) certifies this decision. It used
/// to re-rank PIRLS's stabilized Hessian with its own call to the predicate, so
/// wherever a builder priced a different rank (the root upgrade pricing a mode the
/// assembled band masks) the certificate vouched for a rank the criterion never
/// used.
pub(crate) struct CriterionRankDecision {
    pub(crate) predicate: CriterionRankPredicate,
    pub(crate) frame: CriterionFrame,
    /// `rank(S_λ)`, the floor the predicate was taken at.
    pub(crate) penalty_rank: usize,
    /// The matrix the decision was taken on, in `frame`.
    pub(crate) hessian: Arc<Array2<f64>>,
    /// The operator the criterion priced: its raw eigenpairs and active set.
    pub(crate) operator: Arc<reml_outer_engine::DenseSpectralOperator>,
}

impl CriterionRankDecision {
    /// The number of eigenpairs the criterion priced.
    pub(crate) fn priced_rank(&self) -> usize {
        self.operator
            .active_mask
            .iter()
            .filter(|&&active| active)
            .count()
    }
}

/// What the observation-row fields of a bundle's `pirls_result` describe.
///
/// Every coefficient-space quantity (β̂, `H`, deviance, score, the REML value
/// and its ρ-derivatives) is exact under both variants. The variants differ
/// only in whether `final_eta`, `finalmu` and the working response are this
/// mode's fitted rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BundleRows {
    /// The rows were realised from the design at this bundle's mode.
    Observed,
    /// Fixed-design Gaussian identity: the fit was solved from the
    /// `XᵀWX`, `XᵀW(y−offset)`, `(y−offset)ᵀW(y−offset)` sufficient
    /// statistics, and the row fields are the ρ-invariant carrier shared by
    /// every such solve. Only pure-ρ criterion evaluations may consume it.
    SufficientStatistics,
}

/// Holds the state for the outer REML optimization and supplies cost and
/// gradient evaluations to the `opt` optimizer.
///
/// The `cache` field uses `RefCell` to enable interior mutability. This is a crucial
/// performance optimization. The `cost_andgrad` closure required by the BFGS
/// optimizer takes an immutable reference `&self`. However, we want to cache the
/// results of the expensive P-IRLS computation to avoid re-calculating the fit
/// for the same `rho` vector, which can happen during the line search.
/// `RefCell` allows us to mutate the cache through a `&self` reference,
/// making this optimization possible while adhering to the optimizer's API.
#[derive(Clone)]
pub(crate) struct EvalShared {
    pub(crate) key: Option<Vec<u64>>,
    pub(crate) pirls_result: Arc<PirlsResult>,
    /// Whether `pirls_result`'s rows are fitted rows or the sufficient-statistic
    /// carrier. `obtain_eval_bundle` serves only `Observed` bundles; the pure-ρ
    /// outer criterion accepts either.
    pub(crate) rows: BundleRows,
    /// The routing verdict this bundle was built under, carried WITH the
    /// quantities it was decided from (#2465 instance 4). The bundle used to
    /// hold the bare `RemlGeometry` label, so every consumer that reported
    /// `backend {:?}` reported a two-valued enum and nothing that could
    /// falsify it: `select_reml_geometry` measures a penalized-Hessian
    /// density against `SPARSE_HESSIAN_MAX_DENSITY` on one of its routes
    /// and never measures it on the others, and the label is identical
    /// across all of them. Storing the decision rather than its outcome makes a
    /// bundle unrepresentable without the basis for its own label.
    pub(crate) geometry: SparseRemlDecision,
    /// The exact H_total matrix used for LAML cost computation.
    /// For Firth: effective Hessian minus hphi (plus any barrier curvature).
    /// For non-Firth: the effective Hessian itself (plus any barrier curvature).
    pub(crate) h_total: Arc<Array2<f64>>,
    pub(crate) sparse_exact: Option<Arc<SparseExactEvalData>>,
    pub(crate) firth_dense_operator: Option<Arc<FirthDenseOperator>>,
    /// Cached FirthDenseOperator built from the original (non-reparameterized)
    /// design matrix, for use by the sparse evaluation path.
    pub(crate) firth_dense_operator_original: Option<Arc<FirthDenseOperator>>,
    /// The ONE original-frame penalty pseudo-logdet factorization for this
    /// evaluation point (#931 atom discipline). `log|Σ λ_k S_k|₊`'s VALUE,
    /// ρ-derivatives, τ/ψ components, and ρ×τ cross blocks are all
    /// contractions of this single eigendecomposition; the ρ-side criterion
    /// assembly (`dense_penalty_logdet_derivs`, the sparse det2 path) and the
    /// original-basis hyper-coordinate builders share it through
    /// [`EvalShared::penalty_pseudologdet_original`]. Building a second
    /// factorization of the same Sλ for the same evaluation point is the
    /// objective↔gradient desync surface (#748/#752/#901) this cell removes:
    /// the ridge and positive-eigenspace threshold are decided exactly once.
    /// (The transformed-frame pair-callback path builds its own object — it
    /// factorizes the canonical-TRANSFORMED, possibly constraint-projected
    /// penalties, a genuinely different matrix, not a duplicate of this one.)
    pub(crate) penalty_pseudologdet: std::sync::OnceLock<Arc<penalty_logdet::PenaltyPseudologdet>>,
    /// Root-scale value, inverse and derivatives for this evaluation point.
    ///
    /// `None` inside the cell means the upgrade was examined and declined (the
    /// assembled spectrum already resolves the criterion, or the root could not
    /// be shown to reproduce this `H`); an empty cell means it has not been
    /// examined yet. Caching it here is what keeps the extra `O(n·p²)` Gram to
    /// once per ρ rather than once per value/gradient/Hessian call at that ρ.
    pub(crate) root_scale_hessian_operator:
        std::sync::OnceLock<Option<Arc<reml_outer_engine::DenseSpectralOperator>>>,
    /// The rank decision the dense criterion priced at this evaluation point,
    /// published by its builder (#2959 D1). Shared across the bundle's clones, so
    /// the copy the cache keeps carries what the evaluation's own copy decided.
    /// Empty until a builder prices a spectral operator here: the value-only
    /// Cholesky and the sparse route publish none.
    pub(crate) criterion_rank_decision: Arc<std::sync::OnceLock<Arc<CriterionRankDecision>>>,
    /// The penalty components the criterion APPLIES, `S̃_k = Π S_k Π`, in the
    /// ORIGINAL coefficient frame (#2454).
    ///
    /// `RemlState::canonical_penalties` are the penalties as the term layer
    /// built them. The penalty the model is actually fitted under is the
    /// reparameterization's split-projected `S̃(λ) = Π(Σ_k λ_k S_k)Π`: that is
    /// what the inner solve minimizes, what `H` contains, and what the reported
    /// penalty energy `‖Eβ̂‖²` measures. `log|S|₊`, its rank, its ρ- and
    /// ψ-derivatives, and the `M_p` that sets the REML denominator must all be
    /// taken on THAT penalty, or the two halves of the LAML ratio saturate at
    /// different rates and the criterion acquires an asymptotic ρ-slope of
    /// `½(rank(S̃) − rank(S))` — unbounded below, no interior optimum.
    ///
    /// `Π` is λ-invariant (it comes from the balanced penalty, not the weighted
    /// one), so this set is a function of the inner solve's reparameterization
    /// alone and is computed once per bundle. It is the same `Arc` as the input
    /// when the split declares nothing null or the projection falls below the
    /// roots' own representation noise, which is the ordinary case.
    pub(crate) applied_canonical_penalties:
        std::sync::OnceLock<Arc<Vec<gam_terms::construction::CanonicalPenalty>>>,
    /// Per-evaluation-point cache of the canonical penalty score vectors
    /// `S_k β̂` evaluated at this bundle's inner mode `β̂ =
    /// pirls_result.beta_transformed` (unscaled by λ_k). These depend ONLY
    /// on the inner solution carried by this bundle and the `RemlState`'s
    /// fixed `canonical_penalties` — never on which ρ-coordinate or eval
    /// mode the assembly is running — so they are computed exactly once per
    /// inner solution and shared by every assemble call that reuses the
    /// bundle (cost + gradient evaluations at the same ρ, EFS, synthetic-ext
    /// value probes). Exact hoist, not an approximation: every consumer sees
    /// literally the same vectors it previously recomputed. Initialized via
    /// plain ndarray matvecs (no rayon inside the `OnceLock` closure — the
    /// `get_or_init`+`into_par_iter` deadlock trap does not apply).
    pub(crate) penalty_scores_at_mode: std::sync::OnceLock<Arc<Vec<Array1<f64>>>>,
    /// Per-evaluation-point cache of the #784 block-local Laplace-to-sampling
    /// correction. The correction is a deterministic function of ONLY this
    /// bundle's converged inner state (`pirls_result`, `h_total`), the
    /// `RemlState`'s fixed `canonical_penalties`, and the bundle's ρ, so its
    /// value and gradient are identical for the value-only, value+gradient,
    /// and value+gradient+Hessian assemble calls that share this bundle at a
    /// single ρ, and are computed once per inner solution (exact hoist — #784,
    /// #1082). Its ρ-Hessian is a second pass over the quadrature nodes, paid
    /// only by an evaluation that asks for the Hessian; an entry computed
    /// without it serves every call that does not.
    pub(crate) block_local_correction: BlockLocalCorrectionCell,
}

/// The [`EvalShared::block_local_correction`] slot. Unlike the bundle's
/// `OnceLock` caches it can be upgraded once, from an entry computed without
/// the ρ-Hessian to one with it; a clone copies the entry, as a cloned
/// `OnceLock` does.
#[derive(Default)]
pub(crate) struct BlockLocalCorrectionCell(std::sync::Mutex<Option<BlockLocalCorrectionCache>>);

impl BlockLocalCorrectionCell {
    fn slot(&self) -> std::sync::MutexGuard<'_, Option<BlockLocalCorrectionCache>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The cached entry for `n_ext`, when it serves an evaluation that does or
    /// does not (`want_hessian`) ask for the ρ-Hessian.
    pub(crate) fn get(
        &self,
        n_ext: usize,
        want_hessian: bool,
    ) -> Option<BlockLocalCorrectionCache> {
        self.slot()
            .as_ref()
            .filter(|entry| entry.n_ext == n_ext && (entry.carries_hessian || !want_hessian))
            .cloned()
    }

    /// Store `entry` unless the slot already holds one that serves at least
    /// as much. Racing writers built from identical inputs, so either is
    /// correct.
    pub(crate) fn store(&self, entry: BlockLocalCorrectionCache) {
        let mut slot = self.slot();
        let keep = slot.as_ref().is_some_and(|held| {
            held.n_ext == entry.n_ext && (held.carries_hessian || !entry.carries_hessian)
        });
        if !keep {
            *slot = Some(entry);
        }
    }
}

impl Clone for BlockLocalCorrectionCell {
    fn clone(&self) -> Self {
        Self(std::sync::Mutex::new(self.slot().clone()))
    }
}

/// One bundle's #784 block-local correction, as
/// [`EvalShared::block_local_correction`] holds it.
#[derive(Clone)]
pub(crate) struct BlockLocalCorrectionCache {
    /// The external-coordinate count the terms were laid out for. With ψ
    /// coordinates present the seam declines (the cheap zero), and `n_ext` is
    /// fixed for a fit, so one entry suffices.
    pub(crate) n_ext: usize,
    pub(crate) terms: Arc<outer_eval::TkCorrectionTerms>,
    /// Whether `terms` was computed for an evaluation that asked for the
    /// ρ-Hessian. Its `hessian` is then the correction's exact ρ-Hessian, or
    /// `None` because the correction declined or has none on this fit.
    pub(crate) carries_hessian: bool,
    /// The #2623 ρ-block audit record for the SAME computation, so a later
    /// assemble call at this ρ that reads the cache can re-publish it. Without
    /// that, the audit window — which is cleared at the start of every
    /// assemble call — would report the second and third calls at an engaged ρ
    /// as DECLINED, and an FD row asserting engagement would fail on a fit
    /// where the splice ran. `None` when the splice declined or when the audit
    /// was disarmed (the production case, which allocates nothing).
    pub(crate) audit: Option<crate::estimate::outer_eval_capture::QuadratureMarginalAudit>,
    /// The certified paired-rule error of the quadrature `terms` was integrated
    /// with, in `V`'s own units, so a later assemble at this ρ that reads the
    /// cache publishes the same `band_f` term the computing one did (#3004).
    /// `0.0` when the splice declined. Unlike `audit` this is carried whether or
    /// not the ρ-block audit is armed: the certificate's band must exist wherever
    /// the correction does.
    pub(crate) quadrature_error: f64,
}

/// The penalty components the criterion APPLIES, `S̃_k = Π S_k Π`, for an inner
/// solve that ran under `reparam` (#2454).
///
/// A free function rather than a method, because the sparse-exact lane has to
/// build this BEFORE its `EvalShared` exists — and building it twice from two
/// places is exactly the split this whole issue is about. `EvalShared` caches
/// the result of this call; the sparse builder seeds that cache with the same
/// object it used itself.
///
/// The identity `Arc` comes back whenever the split declares nothing null, the
/// penalty list is empty, or every projection falls below the roots' own
/// representation noise — so the ordinary fit allocates nothing and keeps each
/// penalty's `op` handle, block chart and cached spectrum.
pub(crate) fn applied_canonical_penalties_for(
    reparam: &gam_terms::construction::ReparamResult,
    canonical_penalties: &Arc<Vec<gam_terms::construction::CanonicalPenalty>>,
) -> Result<Arc<Vec<gam_terms::construction::CanonicalPenalty>>, EstimationError> {
    let split = reparam.null_split();
    if split.declared_null_dim() == 0 || canonical_penalties.is_empty() {
        return Ok(Arc::clone(canonical_penalties));
    }
    let projected = canonical_penalties
        .iter()
        .map(|penalty| {
            split.projected_canonical(penalty, gam_terms::construction::PenaltyFrame::Original)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            EstimationError::LayoutError(format!(
                "projecting the canonical penalties onto the reparameterization's penalized \
                 subspace failed: {error}"
            ))
        })?;
    // `projected_canonical` returns `None` when the projection is below the
    // root's own noise, so an all-`None` result IS the identity and is handed
    // back as the original `Arc` rather than as a copy.
    if projected.iter().all(Option::is_none) {
        return Ok(Arc::clone(canonical_penalties));
    }
    Ok(Arc::new(
        projected
            .into_iter()
            .zip(canonical_penalties.iter())
            .map(|(projected, penalty)| projected.unwrap_or_else(|| penalty.clone()))
            .collect(),
    ))
}

impl EvalShared {
    pub(crate) fn matches(&self, key: &Option<Vec<u64>>) -> bool {
        match (&self.key, key) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// Publish the rank decision the builder priced at this point. The first
    /// builder to price a spectral operator here decides; every later build at
    /// the same point prices the same matrix with the same predicate.
    pub(crate) fn publish_criterion_rank_decision(
        &self,
        decision: impl FnOnce() -> CriterionRankDecision,
    ) {
        self.criterion_rank_decision
            .get_or_init(|| Arc::new(decision()));
    }

    /// The rank decision a builder published at this point, if one priced a
    /// spectral operator here.
    pub(crate) fn criterion_rank_decision(&self) -> Option<Arc<CriterionRankDecision>> {
        self.criterion_rank_decision.get().map(Arc::clone)
    }

    /// Lazily build — once per evaluation point — the original-frame
    /// [`PenaltyPseudologdet`](penalty_logdet::PenaltyPseudologdet) of
    /// `Σ λ_k S_k` and hand every caller the SAME factorization.
    ///
    /// This is the #931 port of the penalty-logdet term: value, ρ-first /
    /// ρ-second derivatives, τ-gradient components, τ×τ and ρ×τ Hessian
    /// blocks are all projections of one eigendecomposition, so no pair of
    /// consumers can disagree about the positive-eigenspace threshold.
    ///
    /// `lambdas` must be the λ = exp(ρ) vector of this bundle's evaluation
    /// point and `p` the original-basis coefficient dimension; on a cache
    /// hit both are checked against the stored object where representable.
    /// The penalty components the criterion APPLIES, `S̃_k = Π S_k Π`, in the
    /// ORIGINAL frame — see [`EvalShared::applied_canonical_penalties`] the
    /// field for why every criterion consumer must read these rather than the
    /// term layer's raw blocks (#2454).
    ///
    /// Computed once per bundle and shared. Returns the SAME `Arc` as the input
    /// whenever the split declares nothing null or the projection falls below
    /// the roots' own representation noise, so the ordinary path allocates
    /// nothing and keeps every penalty's `op` handle and block chart.
    pub(crate) fn applied_canonical_penalties(
        &self,
        canonical_penalties: &Arc<Vec<gam_terms::construction::CanonicalPenalty>>,
    ) -> Result<Arc<Vec<gam_terms::construction::CanonicalPenalty>>, EstimationError> {
        if let Some(applied) = self.applied_canonical_penalties.get() {
            return Ok(Arc::clone(applied));
        }
        let applied = applied_canonical_penalties_for(
            &self.pirls_result.reparam_result,
            canonical_penalties,
        )?;
        match self.applied_canonical_penalties.set(Arc::clone(&applied)) {
            Ok(()) => Ok(applied),
            Err(_) => Ok(Arc::clone(
                self.applied_canonical_penalties
                    .get()
                    .expect("OnceLock set raced, so it is initialized"),
            )),
        }
    }

    ///
    /// `spectra` must be the state's [`RemlState::penalty_unit_spectra`]; it
    /// serves the factorization whenever the applied penalties are the
    /// canonical list itself, which is the ordinary path.
    pub(crate) fn penalty_pseudologdet_original(
        &self,
        canonical_penalties: &Arc<Vec<gam_terms::construction::CanonicalPenalty>>,
        spectra: &penalty_logdet::PenaltyUnitSpectra,
        lambdas: &[f64],
        p: usize,
    ) -> Result<Arc<penalty_logdet::PenaltyPseudologdet>, EstimationError> {
        if let Some(pld) = self.penalty_pseudologdet.get() {
            if pld.dim() != p {
                return Err(EstimationError::LayoutError(format!(
                    "shared penalty pseudo-logdet frame mismatch: cached p={}, requested p={}",
                    pld.dim(),
                    p
                )));
            }
            return Ok(Arc::clone(pld));
        }
        // `log|S|₊` is one half of the LAML ratio `½(log|H| − log|S|₊)`; the
        // other half carries the split-projected penalty, so this one must too.
        let applied = self.applied_canonical_penalties(canonical_penalties)?;
        let pld = if spectra.serves(&applied) {
            penalty_logdet::PenaltyPseudologdet::from_penalty_spectra(spectra, lambdas, p)
        } else {
            penalty_logdet::PenaltyPseudologdet::from_penalties(&applied, lambdas, p)
        };
        let pld = Arc::new(pld.map_err(EstimationError::InvalidInput)?);
        match self.penalty_pseudologdet.set(Arc::clone(&pld)) {
            Ok(()) => Ok(pld),
            // A concurrent caller initialized the cell first; both objects
            // were built from identical inputs — return the canonical winner
            // so every consumer holds literally the same factorization.
            Err(_) => Ok(Arc::clone(
                self.penalty_pseudologdet
                    .get()
                    .expect("OnceLock set raced, so it is initialized"),
            )),
        }
    }
}

impl PenalizedGeometry for EvalShared {
    fn backend_kind(&self) -> GeometryBackendKind {
        match self.geometry.geometry {
            RemlGeometry::DenseSpectral => GeometryBackendKind::DenseSpectral,
            RemlGeometry::SparseExactSpd => GeometryBackendKind::SparseExactSpd,
        }
    }
}

impl SparseRemlDecision {
    /// One-line rendering of the quantities this routing verdict was decided
    /// from, for emission beside the `backend=`/`geometry=` label itself
    /// (#2465). `nnz_x`/`nnz_h_est`/`density_h_est` read `na` on the routes
    /// that decide before the corresponding structure is measured — that
    /// absence is itself the basis, and it is what tells
    /// `penalized_hessian_too_dense` (a density genuinely measured above the
    /// threshold) apart from `design_not_sparse` and its four siblings, which
    /// never measure one.
    pub(crate) fn basis(&self) -> String {
        format!(
            "reason={} p={} nnz_x={} nnz_h_est={} density_h_est={} threshold={:.4}",
            self.reason,
            self.p,
            self.nnz_x
                .map(|value| value.to_string())
                .unwrap_or_else(|| "na".to_string()),
            self.nnz_h_upper_est
                .map(|value| value.to_string())
                .unwrap_or_else(|| "na".to_string()),
            self.density_h_upper_est
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "na".to_string()),
            RemlState::SPARSE_HESSIAN_MAX_DENSITY,
        )
    }
}

/// LRU cache keyed by sanitized ρ vectors that holds compacted PIRLS results
/// for warm-starting outer line searches and revisited evaluations.
///
/// Eviction is byte-budgeted rather than entry-count-budgeted: each entry
/// records its own estimated footprint (the surviving n-length vectors plus
/// the two p×p Hessians plus per-entry overhead) and the cache evicts in
/// LRU order until the running total fits under the budget. The newest entry
/// is always kept, alone if it alone exceeds the budget, so the next
/// evaluation at its ρ (a gradient after a value) reuses the solve: the cache
/// holds at most the larger of its budget and one solve, which the evaluation
/// that produced it held anyway.
pub(crate) struct PirlsLruCache {
    // Stored tuple: (compacted result, last-touched clock, estimated bytes).
    pub(crate) map: HashMap<Vec<u64>, (Arc<PirlsResult>, u64, usize)>,
    pub(crate) byte_budget: usize,
    pub(crate) current_bytes: usize,
    pub(crate) clock: u64,
}

impl PirlsLruCache {
    pub(crate) fn new(byte_budget: usize) -> Self {
        Self {
            map: HashMap::new(),
            byte_budget: byte_budget.max(1),
            current_bytes: 0,
            clock: 0,
        }
    }

    pub(crate) fn get(&mut self, key: &Vec<u64>) -> Option<Arc<PirlsResult>> {
        if let Some(entry) = self.map.get_mut(key) {
            self.clock += 1;
            entry.1 = self.clock;
            Some(entry.0.clone())
        } else {
            None
        }
    }

    pub(crate) fn insert(&mut self, key: Vec<u64>, value: Arc<PirlsResult>) {
        self.clock += 1;
        let bytes = pirls_result_cache_bytes(&value);
        if let Some((_, _, prev_bytes)) = self.map.remove(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(prev_bytes);
        }
        while !self.map.is_empty() && self.current_bytes + bytes > self.byte_budget {
            let evict_key = self
                .map
                .iter()
                .min_by_key(|(_, (_, ts, _))| *ts)
                .map(|(k, _)| k.clone());
            match evict_key {
                Some(k) => {
                    if let Some((_, _, evict_bytes)) = self.map.remove(&k) {
                        self.current_bytes = self.current_bytes.saturating_sub(evict_bytes);
                    }
                }
                None => break,
            }
        }
        self.current_bytes += bytes;
        self.map.insert(key, (value, self.clock, bytes));
    }

    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.current_bytes = 0;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct PenaltySubspaceCacheKey {
    pub(crate) penalty_matrix_fingerprint: u64,
}

pub(crate) struct PenaltySubspaceCache {
    pub(crate) entry: Option<(PenaltySubspaceCacheKey, Arc<outer_eval::PenaltySubspace>)>,
}

impl PenaltySubspaceCache {
    pub(crate) fn new() -> Self {
        Self { entry: None }
    }

    pub(crate) fn get(
        &self,
        key: &PenaltySubspaceCacheKey,
    ) -> Option<Arc<outer_eval::PenaltySubspace>> {
        self.entry
            .as_ref()
            .filter(|(cached_key, _)| cached_key == key)
            .map(|(_, value)| value.clone())
    }

    pub(crate) fn insert(
        &mut self,
        key: PenaltySubspaceCacheKey,
        value: Arc<outer_eval::PenaltySubspace>,
    ) {
        self.entry = Some((key, value));
    }

    pub(crate) fn clear(&mut self) {
        self.entry = None;
    }
}

impl PenaltySubspaceCacheKey {
    /// Build a cache key from the transformed-E matrix. `E` is hashed by exact
    /// f64 bits (column-major), so the key is bit-exact and avoids float-Hash
    /// issues. Two calls at the same `E` yield equal keys.
    pub(crate) fn from_inputs(e_transformed: &ndarray::Array2<f64>) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        e_transformed.nrows().hash(&mut hasher);
        e_transformed.ncols().hash(&mut hasher);
        for value in e_transformed.iter() {
            value.to_bits().hash(&mut hasher);
        }
        Self {
            penalty_matrix_fingerprint: hasher.finish(),
        }
    }
}

/// Byte budget of one fit's PIRLS result cache.
///
/// A cache hit saves one P-IRLS solve, and a solve costs passes over the
/// design, so the memo may hold as many bytes as the dense design it
/// memoizes and no more: at any `n` and any number of outer evaluations it is
/// one further design-sized store, or the newest solve alone where one solve
/// outgrows the design (see [`PirlsLruCache::insert`]). The
/// host bound is the governor's stationary per-operation ceiling rather than
/// live availability, so which evaluations hit the cache never depends on
/// what else the machine is doing (SPEC-20).
///
/// A fixed budget was pyGAM audit speed F11: 128 MiB regardless of the
/// design, so a Poisson fit at n = 1e5 with an 8.8 MB design pinned 133 MB of
/// cached solves.
pub(crate) fn pirls_cache_byte_budget(x: &DesignMatrix) -> usize {
    let host_ceiling =
        gam_runtime::resource::MemoryGovernor::global().single_materialization_cap_bytes();
    gam_runtime::resource::dense_f64_bytes(x.nrows(), x.ncols())
        .map_or(host_ceiling, |design_bytes| design_bytes.min(host_ceiling))
}

/// Estimate the in-cache footprint of a (compacted) PIRLS result.
///
/// Mirrors what `compact_for_reml_cache` keeps:
/// * six surviving n-length f64 arrays (final_eta, solveweights,
///   solveworking_response, solvemu, solve_c_array, solve_d_array);
/// * the p-length coefficient vector;
/// * the two p×p Hessians (dense or CSC sparse);
/// * the `ReparamResult` payload, the dominant term beyond n: besides
///   `s_transformed`, `qs` and `e_transformed` it carries one rotated penalty
///   per smoothing coordinate, each with a dense p×p Gram, so it grows as K·p²;
/// * the p-length penalized gradient, the Firth hat diagonal and any
///   transformed inequality constraints.
/// A small constant overhead absorbs scalar fields, enum discriminants, and
/// the HashMap entry. Every array the entry owns is counted: an entry that
/// under-reports lets the cache hold many times its byte budget (pyGAM audit
/// speed F1: forty coordinates at p = 221 put 16 MB of rotated penalties in
/// each entry against a 2.5 MB estimate, so the 128 MiB cache pinned 1.3 GB).
pub(crate) fn pirls_result_cache_bytes(result: &PirlsResult) -> usize {
    use std::mem::size_of;
    let n_array_elems = result.final_eta.len()
        + result.solveweights.len()
        + result.solveworking_response.len()
        + result.solvemu.len()
        + result.solve_c_array.len()
        + result.solve_d_array.len();
    let p = result.beta_transformed.0.len();
    let pen_h = symmetric_matrix_cache_bytes(&result.penalized_hessian_transformed);
    let stab_h = symmetric_matrix_cache_bytes(&result.stabilizedhessian_transformed);
    let firth = match &result.firth {
        crate::pirls::FirthDiagnostics::Inactive => 0,
        crate::pirls::FirthDiagnostics::Active { hat_diag, .. } => hat_diag.len(),
    };
    let constraints = result
        .linear_constraints_transformed
        .as_ref()
        .map_or(0, |constraints| constraints.a.len() + constraints.b.len());
    let small_arrays = n_array_elems
        + p
        + result.penalized_gradient_transformed.len()
        + firth
        + constraints;
    small_arrays * size_of::<f64>()
        + pen_h
        + stab_h
        + result.reparam_result.resident_bytes()
        + 1024
}

pub(crate) fn symmetric_matrix_cache_bytes(m: &gam_linalg::matrix::SymmetricMatrix) -> usize {
    use gam_linalg::matrix::SymmetricMatrix;
    use std::mem::size_of;
    match m {
        SymmetricMatrix::Dense(a) => a.len() * size_of::<f64>(),
        SymmetricMatrix::Sparse(s) => {
            // CSC sparse: f64 values + usize row indices + usize column pointers.
            let (symbolic, values) = s.parts();
            values.len() * (size_of::<f64>() + size_of::<usize>())
                + std::mem::size_of_val(symbolic.col_ptr())
        }
    }
}

/// Capacity (number of distinct rho-points) of the outer-eval reuse LRU.
///
/// Sized to comfortably span a binomial seed grid's local revisit window
/// (baseline + isotropic shifts + per-axis refinements) plus a few
/// line-search trial points without unbounded growth. Each slot holds one
/// `OuterEval` (a scalar cost, a length-k gradient, an optional inner-beta
/// hint, and a usually-`Unavailable` Hessian), so the footprint is tiny.
pub(crate) const OUTER_EVAL_LRU_CAPACITY: usize = 8;

/// Bounded least-recently-used cache of converged outer REML evaluations,
/// keyed by sanitized rho-bits plus the active inner-solve caps.
///
/// CORRECTNESS: the key starts with `Vec<u64>` of `f64::to_bits` (with ±0
/// canonicalized) and appends the outer P-IRLS cap. Every
/// other input to `OuterEval` — design matrix, prior weights, offset, penalty
/// structure, link/SAS/mixture state, Firth/Jeffreys configuration, and the
/// rho-prior — is immutable for the lifetime of the state that owns the cache.
/// Inner fidelity is not immutable: the search-time cap deliberately changes it.
/// Including that cap prevents a full-fidelity terminal request from
/// replaying a coarse search entry at the same rho. Distinct rho/fidelity
/// states never alias because lookups compare the full key vector.
pub(crate) struct OuterEvalLru {
    capacity: usize,
    /// Front = least-recently-used, back = most-recently-used.
    entries: std::collections::VecDeque<(Vec<u64>, OuterEval)>,
}

impl OuterEvalLru {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: std::collections::VecDeque::new(),
        }
    }

    /// Returns a clone of the eval stored under `key`, if present, promoting it
    /// to most-recently-used. A miss returns `None` so the caller recomputes —
    /// never a stale value from a different key.
    pub(crate) fn get(&mut self, key: &[u64]) -> Option<OuterEval> {
        let pos = self.entries.iter().position(|(k, _)| k.as_slice() == key)?;
        let entry = self.entries.remove(pos)?;
        let eval = entry.1.clone();
        self.entries.push_back(entry);
        Some(eval)
    }

    /// Inserts (or refreshes) the eval for `key` as most-recently-used,
    /// evicting the least-recently-used entry once capacity is exceeded.
    pub(crate) fn insert(&mut self, key: Vec<u64>, eval: OuterEval) {
        if let Some(pos) = self
            .entries
            .iter()
            .position(|(k, _)| k.as_slice() == key.as_slice())
        {
            self.entries.remove(pos);
        }
        self.entries.push_back((key, eval));
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Centralized cache/memoization owner for REML evaluations.
///
/// This keeps cache-key identity, bundle reuse, and invalidation policy out of
/// the math kernels so objective/derivative routines can stay algebra-focused.
pub(crate) struct EvalCacheManager {
    pub(crate) pirls_cache: RwLock<PirlsLruCache>,
    pub(crate) penalty_subspace_cache: RwLock<PenaltySubspaceCache>,
    pub(crate) current_eval_bundle: RwLock<Option<EvalShared>>,
    /// Bounded multi-slot LRU of converged outer evaluations keyed by the
    /// sanitized rho-bits (#1575).
    ///
    /// For a frozen `RemlState` (fixed design, prior weights, offset, penalty
    /// structure, link state, Firth/Jeffreys configuration, and rho-prior — all
    /// of which are immutable for the lifetime of the state that owns this
    /// manager and therefore the lifetime of the cache), the remaining
    /// result-determining input is `rho`: every inner solve runs at one
    /// fidelity (#3536), so a value and a gradient request at one rho share an
    /// inner mode. Line searches and certification probes revisit earlier
    /// rho-points; this LRU returns the stored cost/gradient for them instead
    /// of re-running a full n-sized P-IRLS.
    pub(crate) outer_eval_lru: RwLock<OuterEvalLru>,
    pub(crate) pirls_cache_enabled: AtomicBool,
}

impl EvalCacheManager {
    /// `pirls_cache_byte_budget` is the fit's PIRLS result-cache budget,
    /// derived once per fit by [`pirls_cache_byte_budget`].
    pub(crate) fn new(pirls_cache_byte_budget: usize) -> Self {
        Self {
            pirls_cache: RwLock::new(PirlsLruCache::new(pirls_cache_byte_budget)),
            penalty_subspace_cache: RwLock::new(PenaltySubspaceCache::new()),
            current_eval_bundle: RwLock::new(None),
            outer_eval_lru: RwLock::new(OuterEvalLru::new(OUTER_EVAL_LRU_CAPACITY)),
            pirls_cache_enabled: AtomicBool::new(true),
        }
    }

    /// Memoizing wrapper for `PenaltySubspace` construction.
    ///
    /// The penalty-subspace eigendecomposition is shape-invariant: any two
    /// outer evaluations at the same `E_transformed` produce bit-identical
    /// subspaces. The single-slot cache amortizes consecutive fixed-S queries
    /// (rank, logdet, trace) within a single outer iter.
    pub(super) fn cached_penalty_subspace<F>(
        &self,
        e_transformed: &ndarray::Array2<f64>,
        build: F,
    ) -> Result<Arc<outer_eval::PenaltySubspace>, EstimationError>
    where
        F: FnOnce() -> Result<outer_eval::PenaltySubspace, EstimationError>,
    {
        let key = PenaltySubspaceCacheKey::from_inputs(e_transformed);
        if let Some(hit) = self
            .penalty_subspace_cache
            .read()
            .expect("penalty-subspace cache lock is poisoned: a writer panicked while holding it")
            .get(&key)
        {
            return Ok(hit);
        }
        let value = Arc::new(build()?);
        self.penalty_subspace_cache
            .write()
            .expect("penalty-subspace cache lock is poisoned: a writer panicked while holding it")
            .insert(key, value.clone());
        Ok(value)
    }

    /// The cached bundle at `key`, whichever rows it carries; callers that
    /// read rows check [`EvalShared::rows`].
    pub(crate) fn cached_eval_bundle(&self, key: &Option<Vec<u64>>) -> Option<EvalShared> {
        let guard = self
            .current_eval_bundle
            .read()
            .expect("current eval bundle lock is poisoned: a writer panicked while holding it");
        let bundle: &EvalShared = guard.as_ref()?;
        bundle.matches(key).then(|| bundle.clone())
    }

    pub(crate) fn store_eval_bundle(&self, bundle: EvalShared) {
        *self
            .current_eval_bundle
            .write()
            .expect("current eval bundle lock is poisoned: a writer panicked while holding it") =
            Some(bundle);
    }

    pub(crate) fn cached_outer_eval(&self, key: &Option<Vec<u64>>) -> Option<OuterEval> {
        let key = key.as_ref()?;
        // The LRU is the authoritative multi-slot store; it always contains the
        // most-recently-stored eval too (kept in sync by `store_outer_eval`), so
        // a single LRU probe subsumes the old single-slot fast path while also
        // serving revisited (non-immediate) rho-points. `get` is a tiny linear
        // scan (capacity is `OUTER_EVAL_LRU_CAPACITY`) that promotes the hit to
        // most-recently-used; hence the write lock.
        self.outer_eval_lru
            .write()
            .expect("outer-eval LRU lock is poisoned: a writer panicked while holding it")
            .get(key)
    }

    pub(crate) fn store_outer_eval(&self, key: &Option<Vec<u64>>, eval: &OuterEval) {
        if let Some(key) = key.clone() {
            self.outer_eval_lru
                .write()
                .expect("outer-eval LRU lock is poisoned: a writer panicked while holding it")
                .insert(key, eval.clone());
        }
    }

    pub(crate) fn invalidate_eval_bundle(&self) {
        self.current_eval_bundle
            .write()
            .expect("current eval bundle lock is poisoned: a writer panicked while holding it")
            .take();
        self.outer_eval_lru
            .write()
            .expect("outer-eval LRU lock is poisoned: a writer panicked while holding it")
            .clear();
    }

    pub(crate) fn clear_eval_and_factor_caches(&self) {
        self.invalidate_eval_bundle();
        self.penalty_subspace_cache
            .write()
            .expect("penalty-subspace cache lock is poisoned: a writer panicked while holding it")
            .clear();
    }
}

/// Reusable scratch/runtime memory that should not be part of mathematical
/// state invariants.
pub(crate) struct RemlArena {
    pub(crate) cost_eval_count: RwLock<u64>,
    /// Number of *actual* full-n inner P-IRLS solves performed (#1575).
    ///
    /// Distinct from `cost_eval_count`, which counts every outer cost/gradient
    /// REQUEST including single-slot cache hits and prior short-circuits. This
    /// counts only the cache-missing `prepare_eval_bundlewithkey` calls — i.e.
    /// the genuinely expensive `O(n·p²)` inner solves the #1575 slowdown is
    /// about ("~150 outer cost evals each running a full n-sized P-IRLS"). A
    /// healthy warm-started fit performs roughly 2 inner solves per outer
    /// cost-eval (one value, one gradient/Hessian), so a large ratio between
    /// the two signals broken warm-starting or duplicate solving. This is pure
    /// observability: it never feeds back into the optimization and changes no
    /// fitted value.
    pub(crate) inner_pirls_solve_count: AtomicU64,
    pub(crate) lastgradient_used_stochastic_fallback: AtomicBool,
}

impl RemlArena {
    pub(crate) fn new() -> Self {
        Self {
            cost_eval_count: RwLock::new(0),
            inner_pirls_solve_count: AtomicU64::new(0),
            lastgradient_used_stochastic_fallback: AtomicBool::new(false),
        }
    }

    /// Run post-convergence work without charging it to the search.
    ///
    /// `outer_cost_evals` and `inner_pirls_solves` report the work the
    /// smoothing-parameter search did (#1575), and work guards read them as
    /// such. Inference run at the converged `ρ̂` afterwards (the #938 Tier-0
    /// diagnostic and the tiers it selects) evaluates the same criterion, so
    /// both counters are put back to their values at entry once it returns: a
    /// fit that requests that inference reports the same search as one that
    /// does not.
    pub(crate) fn without_charging_the_search<T>(&self, work: impl FnOnce() -> T) -> T {
        let cost_evals = *self
            .cost_eval_count
            .read()
            .expect("cost-eval counter lock is never held across a panic");
        let inner_solves = self
            .inner_pirls_solve_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let result = work();
        *self
            .cost_eval_count
            .write()
            .expect("cost-eval counter lock is never held across a panic") = cost_evals;
        self.inner_pirls_solve_count
            .store(inner_solves, std::sync::atomic::Ordering::Relaxed);
        result
    }
}

/// The ρ at which a fit decides the #784 block-local correction's admission,
/// which [`RemlState::block_correction_admission`] then freezes for the fit
/// (#2748, #1082).
///
/// A fixed-ρ evaluation publishes the ρ it is handed, so it decides there, at
/// its first evaluation whose skewness verdict engages. A search publishes its
/// certified optimum, and deciding at whichever ρ it evaluates first made the
/// fitted criterion a function of the start. On
/// `gam_tensor_te_2d_poisson_matches_mgcv` the verdict engaged at the first
/// evaluation (`max|γ| = 0.238` against `τ = 0.126`) and not at the Laplace
/// optimum (`0.066`), so that latch integrated an `m = 8` block at every one of
/// 132 later inner solutions for a correction the published ρ declines (job
/// 1246493). A search therefore prices the Laplace criterion, decides once at
/// its certified optimum, and either declines there or latches and continues
/// the corrected search from that optimum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockCorrectionDecision {
    /// Decided at the first evaluation whose verdict engages (a fixed-ρ caller).
    AtFirstEngagedEvaluation,
    /// A search is running on the Laplace criterion: the correction is zero and
    /// nothing latches.
    DeferredToOptimum,
    /// The next evaluation, at the search's certified optimum, decides.
    DecidingAtOptimum,
    /// The verdict declined at the certified optimum: zero for the fit.
    DeclinedAtOptimum,
    /// Admitted at the certified optimum and latched.
    AdmittedAtOptimum,
}

/// The #784 block quadrature latched beside the admission (#2623): whether the
/// block marginal is integrated axis by axis with the analytic mixed-axis term or as
/// one tensor rule over the whole block, and each piece's rule. Every later
/// evaluation integrates on these rules, so the criterion is one fixed rule's value
/// at every ρ and its gradient is that rule's derivative (#2748).
/// `hessian_refusal` is the mathematical reason `Δ_b` has no closed-form ρ-Hessian
/// on this fit, or `None` when the correction carries its exact ρ-Hessian into the
/// criterion.
///
/// The block itself is latched as its SPECTRAL POSITIONS: the ranks, in the
/// ascending eigenvalue order of the penalized Hessian, of the directions the
/// admission integrated (a rank, so it does not depend on which order the
/// criterion's eigensolver returns its pairs in). Each later ρ takes the eigenvectors at those
/// positions, so axis `r`'s rule stays attached to the direction it was
/// certified on, and the block moves with ρ as continuously as the
/// eigenvectors at those positions do (continuously away from an eigenvalue
/// coincidence with a neighbouring position, steeply near an avoided one).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BlockQuadratureLatch {
    pub(crate) block_positions: Vec<usize>,
    pub(crate) pieces: Vec<LatchedPieceRule>,
    pub(crate) axis_split: bool,
    pub(crate) hessian_refusal: Option<String>,
}

/// One latched piece of a #784 block. The variant is the rule, so a consumer that
/// rebuilds a piece's nodes (a second-order pass, an audit) must name which rule it
/// rebuilds rather than read a node count as a Gauss–Hermite order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LatchedPieceRule {
    /// A tensor Gauss–Hermite rule of these per-axis orders, with the paired-rule
    /// errors measured at admission: the certificate every later evaluation at those
    /// orders carries, since the paired error switches nothing once they are latched.
    /// A one-axis piece whose target is truncated (a positive-domain link) is
    /// integrated here, by the truncated-normal transport of its Gauss–Hermite rule.
    GaussHermite {
        axis_orders: Vec<usize>,
        certified_axis_errors: Vec<f64>,
    },
    /// One axis, integrated by the composite Gauss–Kronrod rule on the partition the
    /// admission adapted, in the logistic image of the standardized axis `z = √λ·t`
    /// oriented so its standardized skewness is positive.
    Composite {
        breakpoints: Vec<gam_problem::laplace_sampler_contract::AxisBreakpoint>,
    },
}

pub(crate) struct RemlState<'a> {
    pub(crate) y: ArrayView1<'a, f64>,
    pub(crate) x: DesignMatrix,
    pub(crate) weights: ArrayView1<'a, f64>,
    pub(crate) offset: Array1<f64>,
    /// Canonicalized block-local penalties with pre-computed roots.
    /// This is the single canonical penalty representation — no full-width
    /// `rank × p` roots are stored separately.
    pub(crate) canonical_penalties: Arc<Vec<gam_terms::construction::CanonicalPenalty>>,
    pub(crate) reparam_invariant: ReparamInvariant,
    pub(crate) sparse_penalty_block_count: usize,
    pub(crate) p: usize,
    pub(crate) config: Arc<RemlConfig>,
    pub(crate) runtime_mixture_link_state: Option<gam_problem::MixtureLinkState>,
    pub(crate) runtime_sas_link_state: Option<SasLinkState>,
    pub(crate) nullspace_dims: Vec<usize>,
    pub(crate) coefficient_lower_bounds: Option<Array1<f64>>,
    pub(crate) linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    /// Explicit prior on log smoothing parameters used by the REML/LAML objective.
    pub(crate) rho_prior: gam_problem::RhoPrior,

    pub(crate) cache_manager: EvalCacheManager,
    pub(crate) arena: RemlArena,
    pub(crate) warm_start_beta: RwLock<Option<Coefficients>>,
    /// Two-point ρ-trajectory used for second-order warm-start
    /// extrapolation: when the outer optimizer asks for a fit at a new
    /// ρ, we have `β(ρ_k)` (in `warm_start_beta`) and `β(ρ_{k-1})` (in
    /// `prev_warm_start_beta`). The implicit β(ρ) trajectory is locally
    /// linear under the FOC ∇F(β,ρ)=0, so a tangent-line prediction
    /// `β_predict(ρ_new) = β_k + α · (β_k − β_{k-1})` where α is the
    /// projection of `(ρ_new − ρ_k)` onto `(ρ_k − ρ_{k-1})` gives a
    /// better seed than the flat `β_k` alone — replacing PIRLS warm-
    /// start "use last β as-is" with a real tangent-prediction step.
    pub(crate) warm_start_rho: RwLock<Option<Array1<f64>>>,
    pub(crate) prev_warm_start_beta: RwLock<Option<Coefficients>>,
    pub(crate) prev_warm_start_rho: RwLock<Option<Array1<f64>>>,
    /// The #784 block-local correction's ADMISSION, frozen for the fit: `0` is
    /// "not yet admitted anywhere", `m + 1` is "admitted, with a block of `m`
    /// curvature-heavy directions" (#2748).
    ///
    /// The correction used to be admitted or declined by predicates evaluated
    /// at ρ — a skewness threshold `τ(n_eff)`, a quadrature-resolution test,
    /// and a block-dimension cap. A criterion term switched on and off by a
    /// function of ρ is not a function of ρ: it has a jump of the FULL `|Δ_b|`
    /// across every switching surface, and the outer search descends onto one,
    /// because the ON region's minimum is on its boundary. Measured on
    /// `haberman_5yr` (#2748): `max|γ| = 0.125` against `τ = 0.125` at the
    /// refused checkpoint, `V` jumping `1.744282e2 → 1.744593e2` — exactly
    /// `Δ_b = 3.1144e-2` — between two adjacent line-search trial points while
    /// `|g| = 2.045e-2`. No sufficient-decrease test can pass across a jump
    /// larger than `c₁·α·|gᵀd|` for any `α`, so `StepSizeTooSmall after 50
    /// attempt(s)` is the only reachable outcome.
    ///
    /// Freezing the admission is what makes the spliced objective a function.
    /// It also keeps the spliced GRADIENT exact: the four channels differentiate
    /// `Δ_b` at a FIXED block, and any admission that depends on ρ contributes
    /// a `∂Δ_b/∂(admission)·∂(admission)/∂ρ` term the channels do not carry —
    /// the same objective↔gradient desync this site already declines the splice
    /// over for ψ coordinates and for the Beta family.
    ///
    /// Latched on FIRST admission rather than at the first evaluation, so a fit
    /// whose correction never engages is bit-identical to the pre-#2748 fit and
    /// a fit that engaged consistently keeps the same block it always had; only
    /// the fits that were toggling change.
    ///
    /// WHERE that admission is decided is [`Self::block_correction_decision`]
    /// (#1082).
    pub(crate) block_correction_admission: AtomicUsize,
    /// The ρ at which [`Self::block_correction_admission`] is decided (#1082); see
    /// [`BlockCorrectionDecision`].
    pub(crate) block_correction_decision: std::sync::Mutex<BlockCorrectionDecision>,
    /// The per-axis Gauss–Hermite orders latched beside
    /// [`Self::block_correction_admission`] (#2623). They are selected once, at
    /// admission, as the smallest orders whose paired differences resolve
    /// `min(|Δ_b|, 1/n_eff²)`, and held for the fit, so the nodes, and with them
    /// the value, gradient and moments, are one measure at every ρ. Whether the
    /// block is integrated axis by axis is latched with them, for the same reason.
    pub(crate) block_correction_axis_orders: std::sync::Mutex<Option<BlockQuadratureLatch>>,
    /// The certificate's own value resolution `band_f` at the ρ where the
    /// correction's admission is decided (#3004): the error the evaluated `V`
    /// already carries there, from its channels, its inner factor and its inner
    /// mode's residual.
    ///
    /// It is the ORDER TARGET. The quadrature exists to remove a term of the
    /// criterion, and the criterion is read only through a certificate that
    /// cannot distinguish two values closer than `band_f`, so a rule resolving
    /// `Δ_b` below `band_f` buys no decision anywhere. `None` until the
    /// admission supplies it, and a correction that reaches its order search
    /// without one DECLINES: its own error would then be charged into `band_f`
    /// against a target nothing derived.
    pub(crate) block_correction_value_band: std::sync::Mutex<Option<f64>>,
    /// Adaptive IFT step-cap controller, the hypergradient budget controller,
    /// and the two mode-response caches.
    ///
    /// These four lived in process-global `HashMap`s keyed by
    /// `self as *const _ as usize` — the state's own ADDRESS. An address is not
    /// an identity: it is reused the moment the state is dropped, so a fit
    /// whose `RemlState` landed on a recycled slot inherited the previous
    /// fit's quality history, budget history, and cached mode-response
    /// columns. `RemlState` is a stack local in every constructing caller
    /// (`estimate/evaluation.rs`, `estimate/optimizer.rs`, …), so consecutive
    /// fits on one thread reuse the same frame address near-deterministically,
    /// and there is no `Drop` to evict the dead entry. Two fits running
    /// concurrently in one process shared the tables outright.
    ///
    /// Holding them as fields makes the lifetime exactly right by
    /// construction: the state dies with its owner, concurrent fits cannot
    /// see each other, and no key can alias. They sit beside
    /// `ift_warm_start_cache` / `ift_cached_factor`, which were already
    /// per-state interior-mutability fields — these were the odd ones out.
    pub(crate) warm_start_trust: std::sync::Mutex<outer_eval::WarmStartTrustState>,
    pub(crate) ift_mode_response_slot:
        std::sync::Mutex<Option<outer_eval::IftModeResponseRuntimeCache>>,
    pub(crate) ift_joint_mode_response_slot:
        std::sync::Mutex<Option<outer_eval::IftJointModeResponseRuntimeCache>>,
    pub(crate) warm_start_enabled: AtomicBool,
    /// Inner-PIRLS feedback signal driven by `execute_pirls_if_needed` after
    /// each inner solve: the iteration count at which the inner Newton
    /// stopped, and whether it converged. The outer bridges read
    /// `last_inner_converged` (through `InnerProgressFeedback`) so a
    /// stationarity certificate is never issued at a non-converged inner
    /// mode; `last_inner_iters == 0` means no solve has reported yet. The
    /// outer never uses them to limit the inner iteration budget (#3536).
    /// Default 0 / false.
    pub(crate) last_inner_iters: Arc<AtomicUsize>,
    pub(crate) last_inner_converged: Arc<AtomicBool>,

    /// Cached state from the most recent successful PIRLS solve, used by
    /// the IFT-based warm-start predictor.
    ///
    /// The implicit-function theorem applied to the FOC ∇_β F(β,ρ)=0
    /// gives `dβ/dρ_k = -H_pen^{-1} · (e^{ρ_k} · S_k · β)`. A first-order
    /// Taylor predictor reads
    /// `β_predict(ρ_new) = β_cur − Σ_k Δρ_k · H_pen^{-1} · (e^{ρ_cur_k} · S_k · β_cur)`.
    /// This is a strict superset of the tangent-line predictor's
    /// requirements: works after a single successful solve (tangent-line
    /// needs two prior fits), and gives the EXACT first-order Jacobian
    /// of the implicit β(ρ) trajectory rather than a secant proxy along one
    /// ρ-direction.
    ///
    /// Populated in `updatewarm_start_from` when PIRLS converges; cleared
    /// on failure, on `reset_surface`, and on link-state changes.
    pub(crate) ift_warm_start_cache: RwLock<Option<IftWarmStartCache>>,

    /// Persisted Levenberg-Marquardt damping coefficient from the most
    /// recent successful PIRLS solve, bit-packed into an `AtomicU64`
    /// (`f64::to_bits` low 64 bits). Read at the start of
    /// `execute_pirls_if_needed` and written into the
    /// `PirlsConfig::initial_lm_lambda` hint so the inner Newton seeds
    /// `λ_LM` near the damping the previous solve discovered, instead
    /// of cold-starting at `1e-6` and burning 4-6 halving steps to
    /// recover. `0` (the default) signals "no hint"; the inner solver
    /// clamps any positive hint into `[1e-6, 1e-3]` so a stale value
    /// cannot destabilize the next solve. Reset on `reset_surface` and
    /// on failed solves.
    pub(crate) last_pirls_lm_lambda: Arc<AtomicU64>,

    /// Negative-Binomial overdispersion `theta` frozen for the smoothing-
    /// parameter (λ) search (#1082), bit-packed `f64` (`f64::to_bits`). `0`
    /// (the default) signals "not yet frozen". On the first
    /// λ-search inner solve of an estimated-θ NB fit, the maximum-likelihood θ
    /// at the solve's converged η is computed once and stored here; every subsequent
    /// λ-search evaluation pins the inner solve to this value via
    /// `GlmLikelihoodSpec::with_negbin_theta_frozen_for_search`, so the REML
    /// criterion `F(ρ) = REML(ρ, θ_frozen)` is a stationary function of ρ and
    /// the outer optimizer converges instead of chasing the per-eval θ drift
    /// that the estimated path injects. The single final reported fit still
    /// ML-refreshes θ at the converged η. Reset on `reset_surface`.
    pub(crate) frozen_negbin_theta: Arc<AtomicU64>,

    /// Tweedie exponential-dispersion `phi` frozen for the smoothing-parameter
    /// (λ) search (#1477), bit-packed `f64` (`f64::to_bits`). `0` (the default)
    /// signals "not yet frozen". On the first λ-search inner solve
    /// of an estimated-φ Tweedie fit, the converged-η Pearson `phî` is captured
    /// once and stored here; every subsequent λ-search evaluation pins the inner
    /// solve to this value via
    /// `GlmLikelihoodSpec::with_tweedie_phi_frozen_for_search`, so the REML
    /// criterion `F(ρ) = REML(ρ, φ_frozen)` is a stationary function of ρ. The
    /// Tweedie LAML omits the `phi`-dependent saddlepoint normalizer, so a `phi`
    /// drifting with each warm-start η lets the criterion reward dispersion
    /// inflation and rail a double-penalty null-space `λ` to the box bound (the
    /// #1477 boundary blow-up). The single final reported fit still
    /// Pearson-refreshes `phi` at the converged η. Reset on `reset_surface`.
    pub(crate) frozen_tweedie_phi: Arc<AtomicU64>,

    /// Gamma shape `k = 1/φ` frozen for the smoothing-parameter (λ) search
    /// (#1074), bit-packed `f64` (`f64::to_bits`). `0` (the default) signals
    /// "not yet frozen". On the first λ-search inner solve of an
    /// estimated-shape Gamma fit, the seed's converged-η MLE `k̂` is captured
    /// once and stored here; every subsequent λ-search evaluation pins the inner
    /// solve to this value via
    /// `GlmLikelihoodSpec::with_gamma_shape_frozen_for_search`, so the REML
    /// criterion `F(ρ) = REML(ρ, k_frozen)` is a stationary function of ρ. With
    /// `k` estimated the inner solver re-derives it from each warm-start η, and
    /// because the Gamma working weight is `W = prior·k` and the
    /// omitting-constants log-likelihood is `−k·½D`, a `k` swinging with η makes
    /// BOTH the curvature `H = k·XᵀX + λS` and the data-fit `k·½D` jump with ρ —
    /// the criterion grows deterministic spikes that floor the projected
    /// gradient and rail `λ` to the over-smoothed corner (the #1074 te/Gamma
    /// tensor under-recovery). The single final reported fit still ML-refreshes
    /// `k` at the converged η. Reset on `reset_surface`.
    pub(crate) frozen_gamma_shape: Arc<AtomicU64>,

    /// Beta-regression precision `phi` frozen for the smoothing-parameter (λ)
    /// search (#2369), bit-packed `f64` (`f64::to_bits`). `0` (the default)
    /// signals "not yet frozen". On the first λ-search inner solve
    /// of an estimated-φ Beta fit, the seed's converged-η Pearson `phî` is
    /// captured once and stored here; every subsequent λ-search evaluation pins
    /// the inner solve to this value via
    /// `GlmLikelihoodSpec::with_beta_phi_frozen_for_search`, so the REML
    /// criterion `F(ρ) = REML(ρ, φ_frozen)` is a stationary function of ρ. With
    /// `phi` estimated the inner solver re-derives it from each warm-start η, and
    /// because the Beta precision does not factor out of the digamma mean score
    /// `∂ℓ/∂β = φ·Σ xᵢ(y*ᵢ − μ*ᵢ)`, a `phi` swinging with η makes both β̂(ρ) and
    /// the REML data-fit / log-det terms jump with ρ while the analytic outer
    /// gradient holds `phi` fixed — the projected gradient floors above tolerance
    /// and the optimizer refuses ("NOT STATIONARY"), the family-unusable #2369
    /// signature, identical to the sibling Gamma/Tweedie/NB drift. The single
    /// final reported fit still Pearson-refreshes `phi` at the converged η. Reset
    /// on `reset_surface`.
    pub(crate) frozen_beta_phi: Arc<AtomicU64>,

    /// Gaussian (non-identity link) / inverse Gaussian dispersion `phi` frozen
    /// for the λ search, bit-packed `f64`; `0` means "not yet frozen". Captured
    /// once as the converged-η MLE `Σwd/Σw` and applied via
    /// `GlmLikelihoodSpec::with_dispersion_phi_frozen_for_search`, exactly like
    /// [`Self::frozen_tweedie_phi`]; the final reported fit refreshes it.
    pub(crate) frozen_dispersion_phi: Arc<AtomicU64>,

    /// Last observed IFT-prediction residual (`‖β_converged − β_predicted‖
    /// / ‖β_converged‖`) from the most recent solve where
    /// the predictor was actually consumed. Bit-packed `f64` (low 64
    /// bits via `f64::to_bits`).
    ///
    /// "No signal yet" is encoded as a NaN bit-pattern
    /// (`IFT_RESIDUAL_NO_SIGNAL_BITS`). The original `0` sentinel
    /// collided with `f64::to_bits(0.0) == 0` — a true residual of
    /// exactly 0 (degenerate but mathematically possible if every
    /// β_predicted_i matched β_converged_i to bit-equality) would
    /// have been indistinguishable from "predictor never reported".
    /// NaN's self-inequality makes the sentinel unambiguous: any
    /// stored finite non-negative value is genuine signal.
    ///
    /// Read by the outer loop's inner-iteration cap schedule and narrated in
    /// the `[IFT-QUALITY]` bench log. The predictor's step admission does not
    /// read it: that is the derived trust radius in `warm_start_trust`.
    /// Reset on `reset_surface` and on failed solves.
    pub(crate) last_ift_prediction_residual: Arc<AtomicU64>,

    /// Cached Cholesky factorization of `IftWarmStartCache::penalized_hessian_transformed`.
    /// Lazily computed on the first IFT predict call after a fresh
    /// `updatewarm_start_from`, then reused by every subsequent
    /// predict call until the IFT cache is invalidated. At large-scale
    /// scale where p can reach several thousand, the dense Cholesky
    /// is O(p³)/3 — multiple seconds per refactor — so caching saves
    /// real wall time across the typical 5-10 IFT predict calls per
    /// outer fit. Reset jointly with `ift_warm_start_cache` (on
    /// reset_surface, on link-state changes, on failed PIRLS solves,
    /// and whenever a new H_pen replaces the cached one).
    pub(crate) ift_cached_factor: RwLock<Option<Arc<dyn gam_linalg::matrix::FactorizedSystem>>>,

    /// Precomputed `(XᵀWX, XᵀW(y − offset))` for the Gaussian + Identity
    /// outer REML loop, populated once before the outer optimizer when the
    /// family / link / constraint preconditions hold and the design supports
    /// the Identity short-circuit at `pirls.rs:6237`. When present, each
    /// inner `solve_penalized_least_squares_implicit` reads these matrices
    /// instead of restreaming the O(N·p²) GEMM and O(N·p) matvec per outer
    /// iteration — the penalty `λ·S` is still added per-λ.
    ///
    /// Invalidated jointly with the design in `reset_surface`.
    pub(crate) gaussian_fixed_cache: RwLock<Option<Arc<crate::pirls::GaussianFixedCache>>>,
    /// Once-built row placeholders for fixed-design Gaussian value-only
    /// evaluations. These rows depend only on the immutable
    /// `(offset, y, weights, link)` surface, so every distinct rho probe can
    /// share them while its exact deviance/score comes from the Gaussian
    /// sufficient statistics. Full derivative/final-fit evaluations never
    /// consume this slot and continue to materialize their exact `X beta`
    /// rows (#2435).
    pub(crate) gaussian_cost_only_frozen_rows:
        RwLock<Option<Arc<crate::pirls::GaussianFrozenRows>>>,
    /// Conditioned-frame exact ψ-derivatives `(∂XᵀWX/∂ψ, ∂XᵀW(y−offset)/∂ψ)`
    /// for the SINGLE design-moving spatial hyperparameter (#1033b), assembled
    /// n-free from the certified Chebyshev ψ-Gram tensor and installed beside
    /// `gaussian_fixed_cache` at the same in-window trial. When present the
    /// Gaussian-identity ψ-gradient HyperCoord (`a_j`, `g_j`, dense `B_j`) is
    /// formed from these k×k objects instead of realizing and contracting the
    /// n×k ∂X/∂ψ slab — retiring the second per-trial n-pass. Lives in the
    /// SAME conditioned column frame as `gaussian_fixed_cache.xtwx_orig`, so
    /// the hyper-coord builder transforms it by the per-eval Qs/free-basis the
    /// same way it transforms the streamed Gram. Invalidated with the design.
    pub(crate) gaussian_psi_gram_deriv:
        RwLock<Option<Arc<(ndarray::Array2<f64>, ndarray::Array1<f64>)>>>,
    /// Conditioned-frame exact ψ-derivative pair `(∂XᵀWX/∂ψ, ∂XᵀW(y−offset)/∂ψ)`
    /// for the SINGLE design-moving spatial hyperparameter in the GLM (frozen-W)
    /// lane (#1033 / #1111), assembled n-free from
    /// [`crate::glm_sufficient_lane::FrozenWeightGramTensor::gradient_pair_if_sound`]
    /// and installed beside `glm_first_step_gram` at the same in-window
    /// drift-OK trial. When present, the GLM ψ-gradient HyperCoord serves its
    /// envelope `a_j` and score `g_j` from these k×k objects instead of
    /// realizing and contracting the n×k ∂X/∂ψ slab — the second per-trial
    /// n-pass. Unlike the Gaussian lane the Hessian curvature `B_j` is NOT
    /// served from the tensor: for a GLM the per-trial `B_j` term
    /// `X_τᵀWX + XᵀWX_τ` is irreducibly n-dependent (the moving working weight
    /// `W` does not factor out of a frozen-W k×k object), so `B_j` keeps the
    /// exact streamed slab (#1033). Lives in the SAME conditioned column frame
    /// as `glm_first_step_gram` / `gaussian_fixed_cache.xtwx_orig`, so the
    /// hyper-coord builder transforms it by the per-eval Qs/free-basis the same
    /// way. NOT family-gated (the GLM lane's own slot). Invalidated with the
    /// design.
    pub(crate) glm_psi_gram_deriv:
        RwLock<Option<Arc<(ndarray::Array2<f64>, ndarray::Array1<f64>)>>>,
    /// Frozen-weight first-Fisher-step data-fit Gram `XᵀWX` for the GLM
    /// design-moving ψ-sweep (#1111 / #1033 mechanism (c)), in the conditioned
    /// (original / `x_fit`) column frame — the SAME frame as
    /// `gaussian_fixed_cache.xtwx_orig`.
    ///
    /// Assembled n-free per in-window ψ-trial from the certified frozen-weight
    /// Chebyshev tensor ([`crate::glm_sufficient_lane::FrozenWeightGramTensor`])
    /// and installed only when the trial's converged working weight has not
    /// drifted past tolerance from the frozen snapshot. When present, the GLM
    /// inner P-IRLS serves its FIRST Fisher-scoring iteration's `XᵀWX` from this
    /// cache instead of restreaming the O(N·p²) weighted cross-product — the
    /// dominant per-trial n-term in a large-n Poisson/Binomial κ-sweep. The
    /// penalty `Sλ` is still added per-λ on top, and every subsequent inner
    /// iteration restreams the true (moving) `W`, so the converged β̂ is
    /// unchanged; only the first-iteration Gram build is elided. Unlike
    /// `gaussian_fixed_cache` this is NOT family-gated — it is the GLM lane's
    /// own slot, consumed once per inner solve. Invalidated with the design in
    /// `reset_surface`.
    pub(crate) glm_first_step_gram: RwLock<Option<Arc<ndarray::Array2<f64>>>>,
    /// Previous successful non-Gaussian fixed-design data-fit Gram `XᵀWX` in
    /// the conditioned original frame, keyed to `warm_start_beta`.
    ///
    /// When the next outer trial uses a flat warm start, its first PIRLS
    /// curvature build evaluates at the same `η = Xβ` as the previous converged
    /// solve, so the Hessian weights and `XᵀWX` are identical. Reusing this
    /// original-frame Gram skips one dense `O(n·p²)` pass per warm-started
    /// trial while still letting later PIRLS iterations restream the moving
    /// weights. IFT/tangent-predicted starts do not consume this cache.
    ///
    /// The cached dense `p×p` Gram is coupled to a [`MemoryGovernor`] ledger
    /// reservation ([`gam_runtime::resource::Governed`]) so a long-lived cache
    /// entry never holds dense bytes off-ledger; a reservation refusal under
    /// joint pressure skips caching (the streamed path remains available).
    pub(crate) flat_glm_first_step_gram:
        RwLock<Option<gam_runtime::resource::Governed<Arc<ndarray::Array2<f64>>>>>,
    /// Stable disk-cache key for the current realized REML surface. Computed
    /// lazily because it hashes the row-chunked design and data vectors.
    pub(crate) persistent_warm_start_key: RwLock<Option<String>>,
    pub(crate) persistent_latent_values_fingerprint: Option<u64>,
    pub(crate) persistent_latent_values_cache: RwLock<PersistentLatentValuesCache>,
    pub(crate) analytic_penalty_registry_fingerprint: u64,
    /// Ensures the process attempts at most one disk restore per surface.
    pub(crate) persistent_warm_start_loaded: AtomicBool,
    /// Scoped counter disabling all disk persistence from canonical anchors and
    /// cost-only posterior/probe evaluations. In-memory warm starts still
    /// update; loads, writes, and eviction sweeps are suppressed together so a
    /// probe cannot inherit cache history or mutate it.
    pub(crate) persistent_warm_start_store_suppression: AtomicUsize,
    /// Caller-configured cross-process warm-start capability.
    ///
    /// Empty by default: in-memory warm starts remain on, while no ambient
    /// filesystem root is discovered. The estimate constructor attaches the
    /// explicit capability at most once after nuisance anchoring; clones of the
    /// capability share its lazy/opened store.
    pub(crate) persistent_warm_start_store:
        std::sync::OnceLock<gam_runtime::warm_start::ConfiguredWarmStartStore>,
    /// #1033: memoized fit-invariant O(n) response/weight scalars.
    ///
    /// `gaussian_weight_log_sum_half` (`½·Σ log wᵢ`),
    /// `gaussian_dp_floor_scale` (the weighted null deviance `D₀`), the
    /// positive-weight observation count, and `rho_weight_anchor`
    /// (`mean(log wᵢ)`) are pure functions of the borrowed `(y, weights)` —
    /// fields `reset_surface` NEVER reassigns — so they are constant for the
    /// whole life of the `RemlState`. Memoizing them once per fit keeps every
    /// subsequent outer trial in coefficient space. Plain scalar closures, no
    /// rayon inside `get_or_init` (no deadlock trap).
    pub(crate) gaussian_weight_log_sum_half_cache: std::sync::OnceLock<f64>,
    pub(crate) gaussian_dp_floor_scale_cache: std::sync::OnceLock<f64>,
    pub(crate) positive_weight_observation_count_cache: std::sync::OnceLock<usize>,
    pub(crate) rho_weight_anchor_cache: std::sync::OnceLock<f64>,
    /// The data half `R_G` (`R_GᵀR_G = XᵀWX`) of the root-scale Hessian
    /// operator, kept for the weights it was formed at. Keyed to `x`, so
    /// `reset_surface` clears it.
    pub(crate) data_root_cache: laml_logdet::DataRootCache,
    /// The λ-free half of every evaluation's `log|Sλ|₊` factorization — each
    /// penalty's unit root and each block's structural rank — for the penalty
    /// list in `canonical_penalties`. See [`Self::penalty_unit_spectra`].
    pub(crate) penalty_unit_spectra: RwLock<Option<Arc<penalty_logdet::PenaltyUnitSpectra>>>,
}

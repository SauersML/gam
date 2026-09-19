use super::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub(crate) const TK_BLOCK_SIZE: usize = 128;

pub(crate) const ADAPTIVE_KKT_ETA: f64 = 0.1;

pub(crate) const ADAPTIVE_KKT_FLOOR_REML_DIVISOR: f64 = 100.0;

pub(crate) const IFT_QUALITY_HISTORY_CAP: usize = 5;

/// Rolling-quality bands and step-cap adjustment factors for the IFT step-cap
/// controller (`record_ift_prediction_quality`). `quality` is the relative
/// prediction residual averaged over the last [`IFT_QUALITY_HISTORY_CAP`]
/// predictions; below `GROW` the linearization is reliably excellent and the cap
/// is loosened, above `SHRINK` it is tightened, in between it is held. A rolling
/// quality at or above `FLAT_FALLBACK` flips the predictor to flat warm-start.
pub(crate) const IFT_QUALITY_GROW_BAND: f64 = 1e-3;

pub(crate) const IFT_QUALITY_SHRINK_BAND: f64 = 1e-1;

pub(crate) const IFT_QUALITY_FLAT_FALLBACK_BAND: f64 = 0.5;

pub(crate) const IFT_STEP_CAP_GROW_FACTOR: f64 = 1.5;

pub(crate) const IFT_STEP_CAP_SHRINK_FACTOR: f64 = 0.5;

// KKT residual acceptance tolerances for the active-set inner solver.
// Primal/dual/complementarity are checked at 1e-7 (matches the inner
// barrier-stopping tolerance used in PIRLS); stationarity uses a looser
// 5e-6 because the gradient is scaled by penalised Hessian curvature
// that can carry an extra ~order of magnitude of roundoff at convergence.
// The gradient-unit channels (dual, complementarity, stationarity) are judged
// relative to `max(1, ‖g‖∞)` as well as absolutely, through
// `active_set::exceeds_at_gradient_scale`, so the verdict does not depend on
// the response's units.
pub(crate) const KKT_TOL_PRIMAL: f64 = 1e-7;

pub(crate) const KKT_TOL_DUAL: f64 = 1e-7;

pub(crate) const KKT_TOL_COMP: f64 = 1e-7;

pub(crate) const KKT_TOL_STAT: f64 = 5e-6;

pub(crate) fn transformed_penalty_matvec(
    penalty: &gam_terms::construction::CanonicalPenalty,
    beta: &Array1<f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(beta.len());
    let beta_block = beta.slice(ndarray::s![penalty.col_range.clone()]);
    let centered = &beta_block - &penalty.prior_mean;
    let local = penalty.local.dot(&centered);
    out.slice_mut(ndarray::s![penalty.col_range.clone()])
        .assign(&local);
    out
}

impl EvalShared {
    /// Canonical penalty scores `S_k β̂` at this bundle's inner mode
    /// `β̂ = pirls_result.beta_transformed`, computed once per inner solution
    /// and shared by every assemble call on the same bundle (exact hoist —
    /// see the field doc on `penalty_scores_at_mode`).
    ///
    /// `canonical_penalties` must be the penalty slice in the SAME coordinate
    /// frame as `beta_transformed` — i.e. the reparameterized set
    /// `pirls_result.reparam_result.canonical_transformed`, not the owning
    /// `RemlState`'s original-frame `canonical_penalties` (gam#2623: the
    /// original-frame contraction sign-inverted the spliced ρ-gradient).
    /// On a cache hit the stored length is checked against it so a count
    /// mismatch fails loudly instead of silently feeding stale scores.
    pub(crate) fn canonical_penalty_scores_at_mode(
        &self,
        canonical_penalties: &[gam_terms::construction::CanonicalPenalty],
    ) -> Result<Arc<Vec<Array1<f64>>>, EstimationError> {
        if let Some(scores) = self.penalty_scores_at_mode.get() {
            if scores.len() != canonical_penalties.len() {
                return Err(EstimationError::LayoutError(format!(
                    "shared penalty-score cache mismatch: cached {} score vectors, \
                     requested {} canonical penalties",
                    scores.len(),
                    canonical_penalties.len()
                )));
            }
            return Ok(Arc::clone(scores));
        }
        let beta_hat = self.pirls_result.beta_transformed.as_ref();
        let scores = Arc::new(
            canonical_penalties
                .iter()
                .map(|pen| transformed_penalty_matvec(pen, beta_hat))
                .collect::<Vec<_>>(),
        );
        match self.penalty_scores_at_mode.set(Arc::clone(&scores)) {
            Ok(()) => Ok(scores),
            // A concurrent caller initialized the cell first; both vectors
            // were built from identical inputs (same β̂, same penalties) —
            // return the canonical winner so every consumer holds literally
            // the same allocation.
            Err(_) => Ok(Arc::clone(
                self.penalty_scores_at_mode
                    .get()
                    .expect("OnceLock set raced, so it is initialized"),
            )),
        }
    }
}

pub(crate) struct PenaltySubspace {
    pub(crate) evals: Array1<f64>,
    pub(crate) rank: usize,
}

#[derive(Default)]
pub(crate) struct IftQualityRuntimeState {
    pub(crate) quality_history: Vec<f64>,
    pub(crate) next_step_cap: Option<f64>,
    pub(crate) fallback_next_flat: bool,
}

#[derive(Clone)]
pub(crate) struct IftModeResponseRuntimeCache {
    pub(crate) rho: Array1<f64>,
    pub(crate) rho_mode_response_cols: Option<Array2<f64>>,
    pub(crate) ext_mode_response_cols: Option<Array2<f64>>,
}

#[derive(Clone)]
pub(crate) struct IftJointModeResponseRuntimeCache {
    pub(crate) theta: Array1<f64>,
    pub(crate) rho_dim: usize,
    pub(crate) beta_original: Array1<f64>,
    pub(crate) mode_response_cols: Array2<f64>,
    pub(crate) active_constraints: bool,
}

pub(crate) fn joint_ift_cache_matches_theta(
    cache: &IftJointModeResponseRuntimeCache,
    theta: &Array1<f64>,
    new_rho: &Array1<f64>,
) -> bool {
    if cache.theta.len() <= cache.rho_dim
        || theta.len() != cache.theta.len()
        || new_rho.len() != cache.rho_dim
    {
        return false;
    }
    for i in 0..cache.rho_dim {
        if theta[i].to_bits() != new_rho[i].to_bits() {
            return false;
        }
    }
    for i in cache.rho_dim..theta.len() {
        if theta[i].to_bits() != cache.theta[i].to_bits() {
            return false;
        }
    }
    true
}

/// The canonical model-domain upper face owned by the current outer problem.
///
/// This is deliberately not an `Array1<f64>` in the TLS slot. Temporary
/// active-set search boxes have the same representation, and storing one there
/// changes the derivative problem by pretending a frozen search coordinate is
/// an active model upper bound. The private constructor keeps that semantic
/// distinction at the state boundary; callers can only install it through the
/// explicitly model-domain-named recording function below.
#[derive(Clone)]
pub(crate) struct OuterRhoModelUpperBounds {
    values: Array1<f64>,
}

impl OuterRhoModelUpperBounds {
    fn from_model_domain(upper: &Array1<f64>) -> Option<Self> {
        (!upper.is_empty() && upper.iter().all(|value| value.is_finite())).then(|| Self {
            values: upper.clone(),
        })
    }

    pub(crate) fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }
}

thread_local! {
    pub(crate) static IFT_LATEST_OUTER_THETA: std::cell::RefCell<Option<Array1<f64>>> =
        const { std::cell::RefCell::new(None) };

    static IFT_CURRENT_OUTER_RHO_MODEL_UPPER_BOUNDS:
        std::cell::RefCell<Option<OuterRhoModelUpperBounds>> =
            const { std::cell::RefCell::new(None) };
}

pub(crate) fn record_current_outer_theta_for_ift(theta: &Array1<f64>) {
    let value = if theta.is_empty() || theta.iter().any(|v| !v.is_finite()) {
        None
    } else {
        Some(theta.clone())
    };
    IFT_LATEST_OUTER_THETA.with(|slot| *slot.borrow_mut() = value);
}

pub(crate) fn record_current_outer_rho_model_upper_bounds_for_ift(upper: &Array1<f64>) {
    let value = OuterRhoModelUpperBounds::from_model_domain(upper);
    IFT_CURRENT_OUTER_RHO_MODEL_UPPER_BOUNDS.with(|slot| *slot.borrow_mut() = value);
}

pub(crate) fn current_outer_rho_model_upper_bounds_for_ift() -> Option<OuterRhoModelUpperBounds> {
    IFT_CURRENT_OUTER_RHO_MODEL_UPPER_BOUNDS.with(|slot| slot.borrow().clone())
}

pub(crate) fn latest_outer_theta_for_ift() -> Option<Array1<f64>> {
    IFT_LATEST_OUTER_THETA.with(|slot| slot.borrow().clone())
}

#[inline]
pub(crate) fn compute_gradient_for_tk(mode: super::reml_outer_engine::EvalMode) -> bool {
    mode != super::reml_outer_engine::EvalMode::ValueOnly
}

pub(crate) fn hash_array_view(hasher: &mut Fingerprinter, values: ndarray::ArrayView1<'_, f64>) {
    hasher.write_usize(values.len());
    for &value in values {
        hasher.write_f64(value);
    }
}

pub(crate) fn hash_array2<S: ndarray::Data<Elem = f64>>(
    hasher: &mut Fingerprinter,
    values: &ndarray::ArrayBase<S, ndarray::Ix2>,
) {
    hasher.write_usize(values.nrows());
    hasher.write_usize(values.ncols());
    for &value in values {
        hasher.write_f64(value);
    }
}

pub(crate) fn hash_aux_prior_strength(
    hasher: &mut Fingerprinter,
    strength: gam_terms::latent::AuxPriorStrength,
) {
    use gam_terms::latent::AuxPriorStrength;
    match strength {
        AuxPriorStrength::Auto => hasher.write_str("auto"),
        AuxPriorStrength::Fixed(value) => {
            hasher.write_str("fixed");
            hasher.write_f64(value);
        }
    }
}

pub(in crate::estimate) fn latent_id_mode_cache_fingerprint(
    id_mode: &gam_terms::latent::LatentIdMode,
) -> u64 {
    use gam_terms::latent::{AuxPriorFamily, LatentIdMode};
    let mut hasher = Fingerprinter::new();
    hasher.write_str("latent-id-mode-cache-v1");
    match id_mode {
        LatentIdMode::AuxPrior {
            u,
            family,
            strength,
        } => {
            hasher.write_str("aux-prior");
            hash_array2(&mut hasher, u);
            match family {
                AuxPriorFamily::Ridge => hasher.write_str("ridge"),
                AuxPriorFamily::Linear => hasher.write_str("linear"),
            }
            hash_aux_prior_strength(&mut hasher, *strength);
        }
        LatentIdMode::AuxPriorDimSelection {
            u,
            family,
            strength,
            ..
        } => {
            hasher.write_str("aux-prior-dim-selection");
            hash_array2(&mut hasher, u);
            match family {
                AuxPriorFamily::Ridge => hasher.write_str("ridge"),
                AuxPriorFamily::Linear => hasher.write_str("linear"),
            }
            hash_aux_prior_strength(&mut hasher, *strength);
        }
        LatentIdMode::DimSelection { .. } => hasher.write_str("dim-selection"),
        LatentIdMode::IsometryToReference {
            reference,
            strength,
        } => {
            hasher.write_str("isometry-to-reference");
            hash_array2(&mut hasher, reference);
            hash_aux_prior_strength(&mut hasher, *strength);
        }
        LatentIdMode::AuxOutcome { head, .. } => {
            use gam_terms::decoders::behavioral_head::AuxOutcomeFamily;
            hasher.write_str("aux-outcome");
            match head.family() {
                AuxOutcomeFamily::Binomial => hasher.write_str("binomial"),
                AuxOutcomeFamily::Multinomial { n_classes } => {
                    hasher.write_str("multinomial");
                    hasher.write_usize(n_classes);
                }
            }
            hasher.write_usize(head.n_obs());
            hasher.write_f64(head.effective_labeled_count());
        }
        LatentIdMode::None => hasher.write_str("none"),
    }
    hasher.finish_u64()
}

pub(crate) fn hash_array3(hasher: &mut Fingerprinter, values: &ndarray::Array3<f64>) {
    let (a, b, c) = values.dim();
    hasher.write_usize(a);
    hasher.write_usize(b);
    hasher.write_usize(c);
    for &value in values {
        hasher.write_f64(value);
    }
}

pub(crate) fn hash_psi_slice(
    hasher: &mut Fingerprinter,
    target: &gam_terms::analytic_penalties::PsiSlice,
) {
    hasher.write_usize(target.range.start);
    hasher.write_usize(target.range.end);
    match target.latent_dim {
        Some(latent_dim) => {
            hasher.write_bool(true);
            hasher.write_usize(latent_dim);
        }
        None => hasher.write_bool(false),
    }
}

pub(crate) fn hash_scalar_weight_schedule(
    hasher: &mut Fingerprinter,
    schedule: &gam_terms::analytic_penalties::ScalarWeightSchedule,
) {
    use gam_problem::schedule::ScheduleKind;

    hasher.write_f64(schedule.w_start);
    hasher.write_f64(schedule.w_end);
    match &schedule.kind {
        ScheduleKind::Geometric { rate } => {
            hasher.write_str("geometric");
            hasher.write_f64(*rate);
        }
        ScheduleKind::Linear { steps } => {
            hasher.write_str("linear");
            hasher.write_usize(*steps);
        }
        ScheduleKind::ReciprocalIter => hasher.write_str("reciprocal-iter"),
    }
    hasher.write_usize(schedule.iter_count);
}

pub(crate) fn hash_weight_schedule_option(
    hasher: &mut Fingerprinter,
    schedule: &Option<gam_terms::analytic_penalties::ScalarWeightSchedule>,
) {
    match schedule {
        Some(schedule) => {
            hasher.write_bool(true);
            hash_scalar_weight_schedule(hasher, schedule);
        }
        None => hasher.write_bool(false),
    }
}

pub(crate) fn hash_gumbel_temperature_schedule(
    hasher: &mut Fingerprinter,
    schedule: &gam_problem::schedule::GumbelTemperatureSchedule,
) {
    use gam_problem::schedule::ScheduleKind;

    hasher.write_f64(schedule.tau_start);
    hasher.write_f64(schedule.tau_min);
    match &schedule.decay {
        ScheduleKind::Geometric { rate } => {
            hasher.write_str("geometric");
            hasher.write_f64(*rate);
        }
        ScheduleKind::Linear { steps } => {
            hasher.write_str("linear");
            hasher.write_usize(*steps);
        }
        ScheduleKind::ReciprocalIter => hasher.write_str("reciprocal-iter"),
    }
    hasher.write_usize(schedule.iter_count);
}

pub(crate) fn hash_gumbel_schedule_option(
    hasher: &mut Fingerprinter,
    schedule: &Option<gam_problem::schedule::GumbelTemperatureSchedule>,
) {
    match schedule {
        Some(schedule) => {
            hasher.write_bool(true);
            hash_gumbel_temperature_schedule(hasher, schedule);
        }
        None => hasher.write_bool(false),
    }
}

pub(crate) fn hash_isometry_reference(
    hasher: &mut Fingerprinter,
    reference: &gam_terms::analytic_penalties::IsometryReference,
) {
    use gam_terms::analytic_penalties::IsometryReference;

    match reference {
        IsometryReference::Euclidean => hasher.write_str("euclidean"),
        IsometryReference::UserSupplied(values) => {
            hasher.write_str("user-supplied");
            hash_array2(hasher, values.as_ref());
        }
    }
}

pub(crate) fn hash_weight_field(
    hasher: &mut Fingerprinter,
    field: &gam_terms::analytic_penalties::WeightField,
) {
    use gam_terms::analytic_penalties::WeightField;

    match field {
        WeightField::Identity => hasher.write_str("identity"),
        WeightField::Factored { u, rank, p_out } => {
            hasher.write_str("factored");
            hash_array2(hasher, u.as_ref());
            hasher.write_usize(*rank);
            hasher.write_usize(*p_out);
        }
    }
}

pub(crate) fn hash_sparsity_kind(
    hasher: &mut Fingerprinter,
    kind: gam_terms::analytic_penalties::SparsityKind,
) {
    use gam_terms::analytic_penalties::SparsityKind;

    match kind {
        SparsityKind::SmoothedL1 { eps } => {
            hasher.write_str("smoothed-l1");
            hasher.write_f64(eps);
        }
        SparsityKind::Hoyer => hasher.write_str("hoyer"),
        SparsityKind::Log { delta } => {
            hasher.write_str("log");
            hasher.write_f64(delta);
        }
    }
}

pub(crate) fn hash_difference_op_kind(
    hasher: &mut Fingerprinter,
    kind: &gam_terms::analytic_penalties::DifferenceOpKind,
) {
    use gam_terms::analytic_penalties::DifferenceOpKind;

    match kind {
        DifferenceOpKind::ForwardDiff1D => hasher.write_str("forward-diff-1d"),
        DifferenceOpKind::GraphEdges(edges) => {
            hasher.write_str("graph-edges");
            hasher.write_usize(edges.len());
            for &(from, to) in edges {
                hasher.write_usize(from);
                hasher.write_usize(to);
            }
        }
    }
}

pub(crate) fn hash_groups(hasher: &mut Fingerprinter, groups: &[Vec<usize>]) {
    hasher.write_usize(groups.len());
    for group in groups {
        hasher.write_usize(group.len());
        for &axis in group {
            hasher.write_usize(axis);
        }
    }
}

pub(crate) fn hash_analytic_penalty_kind(
    hasher: &mut Fingerprinter,
    penalty: &gam_terms::analytic_penalties::AnalyticPenaltyKind,
) {
    use gam_terms::analytic_penalties::{AnalyticPenaltyKind, PenaltyConcavity};

    hasher.write_str(penalty.name());
    hasher.write_str(&format!("{:?}", penalty.tier()));
    hasher.write_usize(penalty.rho_count());
    match penalty {
        AnalyticPenaltyKind::Isometry(p) => {
            hasher.write_str("isometry");
            hash_psi_slice(hasher, &p.target);
            hash_isometry_reference(hasher, &p.reference);
            hasher.write_usize(p.rho_index);
            hasher.write_usize(p.p_out);
            hash_weight_field(hasher, &p.weight);
            hasher.write_f64(p.scalar_weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
            // The `jacobian_cache` / `jacobian_second_cache` /
            // `third_decoder_derivative` slots are interior-mutable
            // (`RwLock<Option<Arc<…>>>`), lazily populated, and θ-DEPENDENT:
            // the SAE/IFT driver calls `refresh_caches` each outer step so the
            // cached J / H / K reflect the Jacobian at the *current* outer θ.
            // They are NOT part of this penalty's identity — they are a pure
            // (recomputable) function of the basis + θ, and the basis identity
            // is already captured exactly by `duchon_radial_source` (below) for
            // the Duchon path and by the hashed design matrix / latent
            // fingerprint for the SAE path. Hashing the live cache snapshot made
            // the persistent warm-start key non-reproducible across otherwise
            // identical fits: a cold fit opens its session with the slots empty
            // (`None`), while a repeat fit sees them populated from the prior
            // run's converged θ, so the key drifted and the `skip-outer-
            // validation` warm hit was lost (issue #1048). The stored payload is
            // the converged (ρ, β) — equivalence to recomputing is unaffected by
            // dropping these derived snapshots from the key, so we deliberately
            // do NOT hash them.
            match p.duchon_radial_source.as_ref() {
                Some(source) => {
                    hasher.write_bool(true);
                    hash_array2(hasher, source.centers.as_ref());
                    hash_array2(hasher, source.radial_coefficients.as_ref());
                    match source.length_scale {
                        Some(length_scale) => {
                            hasher.write_bool(true);
                            hasher.write_f64(length_scale);
                        }
                        None => hasher.write_bool(false),
                    }
                    hasher.write_str(&format!("{:?}", source.nullspace_order));
                }
                None => hasher.write_bool(false),
            }
        }
        AnalyticPenaltyKind::Sparsity(p) => {
            hasher.write_str("sparsity");
            hasher.write_str(&format!("{:?}", p.target_tier));
            hash_sparsity_kind(hasher, p.kind);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
            hasher.write_bool(p.learns_smoothing());
        }
        AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => {
            hasher.write_str("softmax-assignment-sparsity");
            hasher.write_usize(p.k_atoms);
            hasher.write_f64(p.temperature);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::OrderedBetaBernoulli(p) => {
            hasher.write_str("ibp-assignment");
            hasher.write_usize(p.k_max);
            hasher.write_f64(p.alpha);
            hasher.write_f64(p.tau);
            hash_gumbel_schedule_option(hasher, &p.temperature_schedule);
            hasher.write_bool(p.learnable_alpha);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::Ard(p) => {
            hasher.write_str("ard");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.latent_dim);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
            hasher.write_usize(p.rho_indices.len());
            for &idx in &p.rho_indices {
                hasher.write_usize(idx);
            }
            hasher.write_f64(p.n_eff);
        }
        AnalyticPenaltyKind::TopKActivation(p) => {
            hasher.write_str("topk-activation");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.k);
            hasher.write_usize(p.latent_dim);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::SmoothThreshold(p) => {
            hasher.write_str("smooth_threshold");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.latent_dim);
            hash_array_view(hasher, p.thresholds.view());
            hasher.write_f64(p.weight);
            hasher.write_f64(p.smoothing_eps);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::TotalVariation(p) => {
            hasher.write_str("total-variation");
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hash_difference_op_kind(hasher, &p.difference_op);
            hasher.write_f64(p.smoothing_eps);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::NuclearNorm(p) => {
            hasher.write_str("nuclear-norm");
            hash_psi_slice(hasher, &p.target);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_f64(p.smoothing_eps);
            match p.max_rank {
                Some(max_rank) => {
                    hasher.write_bool(true);
                    hasher.write_usize(max_rank);
                }
                None => hasher.write_bool(false),
            }
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::BlockSparsity(p) => {
            hasher.write_str("block-sparsity");
            hash_psi_slice(hasher, &p.target);
            hash_groups(hasher, &p.groups);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_f64(p.smoothing_eps);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::MechanismSparsity(p) => {
            hasher.write_str("mechanism-sparsity");
            hash_psi_slice(hasher, &p.target);
            hash_groups(hasher, &p.feature_groups);
            hasher.write_f64(p.weight);
            hasher.write_f64(p.smoothing_eps);
            hasher.write_f64(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            match &p.weight_schedule {
                Some(schedule) => {
                    hasher.write_bool(true);
                    hash_scalar_weight_schedule(hasher, schedule.as_ref());
                }
                None => hasher.write_bool(false),
            }
        }
        AnalyticPenaltyKind::RowPrecisionPrior(p) => {
            hasher.write_str("row-precision-prior");
            hash_array3(hasher, &p.lambda_per_row);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_psi_slice(hasher, &p.target);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::IvaeRidgeMeanGauge(p) => {
            hasher.write_str("ivae-ridge-mean-gauge");
            hash_array2(hasher, &p.aux);
            hash_array2(hasher, &p.ridge_inv);
            hasher.write_f64(p.ridge_eps);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_psi_slice(hasher, &p.target);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
            hasher.write_str("parametric-row-precision-prior");
            hash_array2(hasher, &p.aux);
            hash_array_view(hasher, p.log_alpha.view());
            hash_array_view(hasher, p.raw_beta.view());
            hash_array2(hasher, &p.mu);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hash_psi_slice(hasher, &p.target);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::ScadMcp(p) => {
            hasher.write_str("scad-mcp");
            hash_psi_slice(hasher, &p.target);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_f64(p.gamma);
            hasher.write_f64(p.smoothing_eps);
            match p.variant {
                PenaltyConcavity::Mcp => hasher.write_str("mcp"),
                PenaltyConcavity::Scad => hasher.write_str("scad"),
            }
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::BlockOrthogonality(p) => {
            hasher.write_str("block-orthogonality");
            hash_psi_slice(hasher, &p.target);
            hash_groups(hasher, &p.groups);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::DecoderIncoherence(p) => {
            hasher.write_str("decoder-incoherence");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.block_sizes.len());
            for &m in &p.block_sizes {
                hasher.write_usize(m);
            }
            hasher.write_usize(p.p_out);
            hasher.write_usize(p.k_atoms);
            hasher.write_usize(p.pairs.len());
            for &(j, k, w) in &p.pairs {
                hasher.write_usize(j);
                hasher.write_usize(k);
                hasher.write_f64(w);
            }
            hasher.write_f64(p.weight);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::Orthogonality(p) => {
            hasher.write_str("orthogonality");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.latent_dim);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::NestedPrefix(p) => {
            hasher.write_str("nested-prefix");
            hash_psi_slice(hasher, &p.target);
            hasher.write_str(&format!("{:?}", p.target_tier));
            hasher.write_usize(p.prefix_sizes.len());
            for &m in &p.prefix_sizes {
                hasher.write_usize(m);
            }
            hasher.write_usize(p.shell_weights.len());
            for &w in &p.shell_weights {
                hasher.write_f64(w);
            }
            hasher.write_f64(p.eps);
            hasher.write_usize(p.rho_indices.len());
            for &idx in &p.rho_indices {
                hasher.write_usize(idx);
            }
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::Monotonicity(p) => {
            hasher.write_str("monotonicity");
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_f64(p.direction);
            hasher.write_f64(p.smoothing_eps);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::SheafConsistency(p) => {
            hasher.write_str("sheaf-consistency");
            hasher.write_f64(p.weight());
            let dims = p.stalk_dims();
            hasher.write_usize(dims.len());
            for &d in dims {
                hasher.write_usize(d);
            }
        }
        AnalyticPenaltyKind::HarmonicRoughness(p) => {
            hasher.write_str("harmonic-roughness");
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            // Per-period diagonal weights are the penalty's identity: the
            // resolved operator tiles them across the `n_eff` rows, so two
            // penalties with equal (weight, n_eff, rho_index) but different
            // row_weights are distinct and must not share a warm-start key.
            hasher.write_usize(p.row_weights.len());
            for &w in &p.row_weights {
                hasher.write_f64(w);
            }
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
    }
}

pub(crate) fn analytic_penalty_registry_fingerprint(
    registry: &gam_terms::analytic_penalties::AnalyticPenaltyRegistry,
) -> u64 {
    let mut hasher = Fingerprinter::new();
    hasher.write_str("analytic-penalty-registry-v1");
    hasher.write_usize(registry.penalties.len());
    for penalty in &registry.penalties {
        hash_analytic_penalty_kind(&mut hasher, penalty);
    }
    hasher.finish_u64()
}

pub(crate) fn hash_design_matrix(
    hasher: &mut Fingerprinter,
    design: &DesignMatrix,
) -> Result<(), String> {
    // Stream the design through fixed-byte row blocks so a large-scale design
    // is never fully materialized just to fingerprint it. Target ~8 MiB of
    // working set per chunk, with a row-count floor of 1 (always make progress)
    // and a ceiling so a very narrow design does not request an unbounded chunk.
    // Imported, not transcribed (#2704). The BYTE TARGET is the shared
    // quantity; the `[1, 4096]` band below is NOT shared and must not be
    // unified with the BLAS-3 tiling bands — a fingerprint pass wants a
    // progress guarantee under `step_by` and a bounded buffer, not L2/L3
    // residency.
    const HASH_CHUNK_TARGET_BYTES: usize = gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES;
    const HASH_CHUNK_MIN_ROWS: usize = 1;
    const HASH_CHUNK_MAX_ROWS: usize = 4096;
    let n = design.nrows();
    let p = design.ncols();
    hasher.write_usize(n);
    hasher.write_usize(p);
    let bytes_per_row = p.saturating_mul(std::mem::size_of::<f64>()).max(1);
    let chunk_rows =
        (HASH_CHUNK_TARGET_BYTES / bytes_per_row).clamp(HASH_CHUNK_MIN_ROWS, HASH_CHUNK_MAX_ROWS);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| format!("persistent warm-start design hash failed: {e}"))?;
        hash_array2(hasher, &chunk);
    }
    Ok(())
}

pub(crate) fn hash_canonical_penalties(
    hasher: &mut Fingerprinter,
    penalties: &[gam_terms::construction::CanonicalPenalty],
) {
    hasher.write_usize(penalties.len());
    for penalty in penalties {
        hasher.write_usize(penalty.col_range.start);
        hasher.write_usize(penalty.col_range.end);
        hasher.write_usize(penalty.total_dim);
        hasher.write_usize(penalty.nullity);
        hash_array2(hasher, &penalty.root);
        hash_array2(hasher, &penalty.local);
        hash_array_view(hasher, penalty.prior_mean.view());
        hasher.write_usize(penalty.positive_eigenvalues.len());
        for &value in &penalty.positive_eigenvalues {
            hasher.write_f64(value);
        }
        hasher.write_bool(penalty.op.is_some());
    }
}

pub(crate) fn finite_positive_from_bits(bits: u64) -> Option<f64> {
    if bits == 0 {
        return None;
    }
    let value = f64::from_bits(bits);
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

pub(crate) fn finite_nonnegative_from_bits(bits: u64) -> Option<f64> {
    let value = f64::from_bits(bits);
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

pub(crate) fn finite_nonnegative_bits_or_no_signal(value: Option<f64>) -> u64 {
    value
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(f64::to_bits)
        .unwrap_or(IFT_RESIDUAL_NO_SIGNAL_BITS)
}

#[derive(Clone)]
pub(crate) struct TkCorrectionTerms {
    pub(crate) value: f64,
    pub(crate) gradient: Option<Array1<f64>>,
    pub(crate) hessian: Option<Array2<f64>>,
}

pub(crate) struct TkSharedIntermediates {
    pub(crate) h_diag: Array1<f64>,
    pub(crate) x_m: Array1<f64>,
    pub(crate) y: Array1<f64>,
    pub(crate) active_blocks: Vec<TkActiveBlock>,
    /// `Some` when the row-pair sums run through the `TkRowPairTensor` route.
    pub(crate) row_pair_tensor: Option<TkRowPairTensor>,
}

/// How the Tierney-Kadane row-pair sums `Σ_ij c_i c_j K_ij^m (…)`, with
/// `K_ij = x_iᵀH⁻¹x_j`, are evaluated. Both routes are exact and agree to
/// roundoff; they differ only in work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TkRowPairRoute {
    /// The blocked row-pair gram: `O(active²·p)`.
    RowPairs,
    /// Contraction through `T = Σ_j c_j x_j⊗x_j⊗x_j`: `O((active + n)·p³)` time
    /// and `p³` working memory.
    Tensor,
}

impl TkRowPairRoute {
    /// The route with less leading work for `n` rows, `active` of them with a
    /// nonzero weight, and `p` columns: the row-pair gram forms `active²·p`
    /// products, the tensor route `active·p³` to build `T` and `n·p³` to
    /// contract it against every row.
    pub(crate) fn predicted(n: usize, active: usize, p: usize) -> Self {
        let p_cubed = p.saturating_mul(p).saturating_mul(p);
        let row_pairs = active.saturating_mul(active).saturating_mul(p);
        let tensor = active.saturating_add(n).saturating_mul(p_cubed);
        if tensor < row_pairs {
            Self::Tensor
        } else {
            Self::RowPairs
        }
    }

    /// The route with less leading work for the ρ-Hessian block of
    /// `¹⁄₁₂ Σ_ij c_i c_j K_ij³` over `n` rows, `p` columns and `k` smoothing
    /// coordinates. The row-pair jets form `n²·(1 + k + k²)·p` products. The
    /// tensor route forms `n·(1 + k)·p³` to build its tensors and
    /// `n·((1 + 2k)·p³ + k²·p²)` to contract them.
    pub(crate) fn predicted_rho_hessian(n: usize, p: usize, k: usize) -> Self {
        let p_squared = p.saturating_mul(p);
        let p_cubed = p_squared.saturating_mul(p);
        let jet_width = k.saturating_mul(k).saturating_add(k).saturating_add(1);
        let row_pairs = n.saturating_mul(n).saturating_mul(jet_width).saturating_mul(p);
        let per_row = k
            .saturating_mul(3)
            .saturating_add(2)
            .saturating_mul(p_cubed)
            .saturating_add(k.saturating_mul(k).saturating_mul(p_squared));
        let tensor = n.saturating_mul(per_row);
        if tensor < row_pairs {
            Self::Tensor
        } else {
            Self::RowPairs
        }
    }
}

/// Per-row contractions of `T = Σ_j c_j x_j⊗x_j⊗x_j` against `z_i = H⁻¹x_i`.
pub(crate) struct TkRowPairTensor {
    /// Row `i` is `r_i = T[z_i, z_i, ·] = Σ_j c_j K_ij² x_j`.
    pub(crate) r: Array2<f64>,
    /// `s_i = r_iᵀz_i = Σ_j c_j K_ij³`.
    pub(crate) s: Array1<f64>,
    /// `H⁻¹` as a dense matrix.
    pub(crate) h_inv: Array2<f64>,
    /// The ledger charge for `r` and `s`.
    pub(crate) _reservation: gam_runtime::resource::MemoryReservation,
}

pub(crate) struct TkActiveBlock {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) entries: Vec<(usize, f64)>,
}

/// Family-dependent derivative context shared by all assembly builders.
///
/// Both `build_dense_derivative_context` and `build_sparse_derivative_context`
/// return this, eliminating the tuple-order mismatch that previously existed
/// between the two paths.
pub(crate) struct DerivativeContext {
    pub(crate) deriv_provider: Box<dyn super::reml_outer_engine::HessianDerivativeProvider>,
    pub(crate) dispersion: super::reml_outer_engine::DispersionHandling,
    pub(crate) log_likelihood: f64,
    pub(crate) firth_op: Option<std::sync::Arc<super::FirthDenseOperator>>,
    pub(crate) barrier_config: Option<super::reml_outer_engine::BarrierConfig>,
}

/// Project a `GlmLikelihoodSpec` onto a `LikelihoodSpec` for pattern matching
/// on the `(response, link)` form used elsewhere in the codebase.
#[inline]
pub(crate) fn reml_spec(likelihood: &GlmLikelihoodSpec) -> LikelihoodSpec {
    likelihood.spec.clone()
}

#[inline]
pub(crate) fn reml_is_gaussian_identity(likelihood: &GlmLikelihoodSpec) -> bool {
    reml_spec(likelihood).is_gaussian_identity()
}

/// Inverse link of a Binomial family for which a Fisher-weight jet exists, i.e.
/// the links the link-general Jeffreys term can regularize. This includes
/// standard `{Logit, Probit, CLogLog}` and stateful links whose fourth/fifth
/// inverse-link derivatives are available, including mixture LogLog/Cauchit
/// components. Returns `None` for any other response or link.
#[inline]
pub(crate) fn reml_jeffreys_supported_link(likelihood: &GlmLikelihoodSpec) -> Option<InverseLink> {
    let spec = reml_spec(likelihood);
    if !matches!(spec.response, ResponseFamily::Binomial) {
        return None;
    }
    if spec.link.has_fisher_weight_jet() {
        Some(spec.link.clone())
    } else {
        None
    }
}

/// Resolve whether the Jeffreys/Firth term should be assembled on the REML path
/// and, if so, the inverse link to evaluate the Fisher weight with.
///
/// The Jeffreys term is assembled iff the caller requested Firth bias reduction
/// (`firth_bias_reduction`) on a Binomial inverse link that exposes a
/// Fisher-weight jet. This MUST agree with the inner P-IRLS Firth activation in
/// `loop_driver.rs`: the outer analytic derivatives (`H`, `u`, IFT) and the
/// converged inner mode have to be derivatives of the SAME penalized objective.
/// Arming the outer term while the inner mode is non-Firth (or vice-versa)
/// desyncs the two by exactly the Jeffreys score/curvature contribution and
/// breaks the τ-τ Hessian-vs-FD and stationarity-cancellation identities
/// (#825). Unsupported links return `None` instead of pretending they are Logit.
#[inline]
pub(crate) fn reml_robust_jeffreys_link(config: &RemlConfig) -> Option<InverseLink> {
    if !config.firth_bias_reduction {
        return None;
    }
    reml_jeffreys_supported_link(&config.likelihood)
}

/// `upper`/`tail_prob` calibrating the proper distribution prior derived for an
/// unset smoothing coordinate. The tail statement `P(d > upper) = tail_prob` on the
/// marginal-SD distance scale `d = exp(−ρ/2)` calibrates the exponential rate
/// `θ = −ln(tail_prob)/upper`. We use `upper = 10`, `tail_prob = 0.01`
/// ⇒ `θ = −ln(0.01)/10 ≈ 0.4605`.
pub(crate) const RHO_DISTRIBUTION_PC_UPPER: f64 = 10.0;

pub(crate) const RHO_DISTRIBUTION_PC_TAIL_PROB: f64 = 0.01;

#[inline]
pub(crate) fn reml_fixed_glm_dispersion(
    likelihood: &GlmLikelihoodSpec,
) -> Result<f64, EstimationError> {
    use gam_problem::ResolvedLikelihoodScale as Scale;
    let resolved = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let phi = match resolved {
        // These likelihoods carry their complete scale in the family geometry:
        // NB through theta, Beta through its precision-dependent likelihood and
        // Hessian. Treating Beta precision as EDM dispersion double-scales EFS.
        Scale::Unit | Scale::NegativeBinomial { .. } | Scale::BetaPrecision { .. } => 1.0,
        Scale::FixedGaussian { phi } | Scale::Tweedie { phi, .. } => phi.value(),
        Scale::Gamma { .. } => resolved
            .gamma_phi()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        Scale::ProfiledGaussian => {
            return Err(EstimationError::InvalidInput(
                "profiled Gaussian has no fixed REML dispersion".to_string(),
            ));
        }
        Scale::Unspecified => {
            return Err(EstimationError::InvalidInput(
                "family has no scalar GLM dispersion".to_string(),
            ));
        }
    };
    if phi.is_finite() && phi > 0.0 {
        Ok(phi)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{} REML dispersion must be finite and positive; got {phi}",
            likelihood.spec.response.name()
        )))
    }
}

// `MIN_IMPORTANCE_ESS_FRACTION = 0.10` used to live here: the minimum
// importance-sampling effective-sample fraction below which the #784 block-local
// sampled marginalization was declined, on the stated grounds that a smaller
// fraction meant "the Monte-Carlo estimate would be noisier than the Laplace
// error it corrects".
//
// That reasoning was right and the quantity was wrong. `ESS/S` is a relative
// EFFICIENCY of the importance weights; the noise of the estimate is
// `se(Δ_b) = sqrt(1/ESS − 1/S)`, which depends on `S` as well and is what has to
// be compared against anything. Measured on a `geo_latlon`-shaped fit the weights
// are near-uniform (`ESS = 508/512`), so the fraction never fired at all, while
// `se` exceeded `|Δ_b|` outright as the search converged (gam#2584). The gate now
// tests the paired-rule certificate in `block_local_quadrature_correction_compute`, which
// needs no fraction: `se ≥ |Δ_b|` is the estimate failing to distinguish its own
// correction from zero.

/// Block-local non-Gaussian-remainder target for the adaptive Laplace-to-
/// sampling fallback (issue #784).
///
/// Implements the engine-local HMC I/O block target for the standard-GAM
/// GLM inner loop. The fallback sampler asks this target, for each whitened
/// block displacement `t` (coordinates in the curvature-heavy H-eigenvector
/// subspace `V_b`), for the non-Gaussian remainder
///
///   ΔF(t) = F(β̂ + δ) − F(β̂) − ½ δᵀ H δ,   δ = V_b t,  s = X δ,
///   F(β)  = Σ_i ψ_i(x_iᵀβ) + ½ βᵀ S(ρ) β,   ψ_i = D_i/(2φ),
///
/// with `H = Xᵀ W X + S(ρ)` and `W_i = ψ_i''(η̂_i)`. The penalty is exactly
/// quadratic, so its curvature cancels against `S(ρ)` in `H` and only its
/// linear term `(S β̂)·δ` survives. The mode condition `∇F(β̂) = 0` is
/// `S β̂ = −Xᵀ ψ'(η̂)`, so that linear term is `−ψ'(η̂)·s`, and the remainder is
/// the per-row Taylor remainder of the likelihood beyond second order:
///
///   ΔF(t) = Σ_i [ψ_i(η̂_i + s_i) − ψ_i(η̂_i) − ψ_i'(η̂_i) s_i − ½ W_i s_i²].
///
/// On the exact mode this equals the form that keeps `(S β̂)·δ`, and so do its
/// total ρ-derivatives. The two differ in what they do with the mode's
/// rounding, and the difference decides whether the outer search can converge.
/// `(S β̂)·δ` rebuilt from `λ_k (S_k β̂)` is a sum over penalties that the outer
/// search drives to `λ = e^{27}` and beyond, where `S_k β̂` itself is a
/// rounding residue of `‖β̂‖`. Measured on a near-separated binomial CV fold
/// (n = 160, `y ~ s(x) + s(z)`): `λ = 8.4e11` times a residue near `3e-15`
/// put a spurious linear term of `±4e-4` along a block direction whose
/// curvature is `0.035`, while the inner solve's own KKT residual along that
/// direction was `1e-12` to `1e-17`. The quadrature's first-order response to a
/// linear term is `−E_p[t]`, about 1.66 there, so `Δ_b` moved by up to `1.6e-3`
/// between evaluations at one ρ, which no line search can descend through;
/// the fit ended with `|Pg| = 3e-2` and `StepSizeTooSmall`. The row form is a
/// sum over bounded per-row terms with no `λ` in it, so it has no such channel.
///
/// Firth: the Jeffreys term `J = −½ log|I(β)|` is part of `F`, so the mode
/// condition carries `∇J`, and `H` carries `∇²J`. Its linear and quadratic
/// Taylor terms cancel in `ΔF` exactly as the penalty's do. What this target
/// leaves out is the Jeffreys remainder beyond second order, which is
/// `O(n^{-3/2})` next to the likelihood remainder's `O(n^{-1/2})`. The
/// penalty-score form instead left out `∇J·δ`, a first-order term, and
/// measured `Δ_b ≈ 9.3` on the same fold's Firth retry.
///
/// `D/2` and its η-score come from one fallible family row oracle; `φ = 1` for
/// families whose reported deviance already carries the likelihood scale
/// (including fixed-scale Gaussian and Beta), and is the EDM dispersion for
/// unscaled Gamma/Tweedie deviance. With δ held fixed in coefficient space ρ
/// does not appear in `ΔF` explicitly: it enters only through the mode `β̂(ρ)`
/// (via `η̂`, `ψ'(η̂)` and `W`), which the mode-motion channel differentiates.
pub(crate) struct Gam784BlockTarget<'t> {
    /// `X_t` (transformed-basis dense design, matching `h_total`/`solve_c_array`).
    pub(crate) x_transformed: &'t Array2<f64>,
    /// Block eigenvectors `V_b` (columns), shape `p × m`.
    pub(crate) block_vecs: Array2<f64>,
    /// Block curvatures `λ_r` (the `H_total` eigenvalues), length `m`.
    pub(crate) block_lambdas: Array1<f64>,
    /// Mode linear predictor η̂ = X_t β̂.
    pub(crate) eta_hat: Array1<f64>,
    /// Per-row observed weights `W_i` (the likelihood Hessian diagonal).
    pub(crate) weights_obs: Array1<f64>,
    /// `ln|W_i|` (`-inf` for exact zero), certified once at target creation.
    pub(crate) weights_obs_log_abs: Array1<f64>,
    /// Response y and prior weights for the deviance.
    pub(crate) y: Array1<f64>,
    pub(crate) prior_weights: Array1<f64>,
    /// Family/link spec for the deviance and the inverse link.
    pub(crate) likelihood: GlmLikelihoodSpec,
    pub(crate) inverse_link: InverseLink,
    /// Divisor converting reported half-deviance to negative log-likelihood.
    pub(crate) phi: f64,
    /// Penalty scores `S_k β̂` per canonical penalty (unscaled by λ_k).
    /// Shared from the eval bundle's once-per-inner-solution cache
    /// (`EvalShared::canonical_penalty_scores_at_mode`). `ΔF` does not read
    /// them; the evaluator's mode response `dβ̂/dρ_k = −H⁻¹ λ_k S_k β̂` does.
    pub(crate) penalty_scores: Arc<Vec<Array1<f64>>>,
    /// TRANSFORMED-frame canonical penalties — the same coordinate frame as
    /// `x_transformed`, `block_vecs` and the mode β̂ they are contracted
    /// against. These are `pirls_result.reparam_result.canonical_transformed`,
    /// the documented single source of truth for penalty roots in the
    /// transformed frame. Contracting the ORIGINAL-frame penalties here made
    /// the spliced ρ-gradient wrong by 4.6e-2 to 1.0 relative — sign-inverted
    /// with nine orders of error on the worst cells — whenever the stable
    /// reparameterization was far from identity (gam#2623).
    pub(crate) penalties: &'t [gam_terms::construction::CanonicalPenalty],
    /// `λ_k = e^{ρ_k}` per canonical penalty, aligned with `penalty_scores`.
    pub(crate) lambdas: Vec<f64>,
    /// Certified `D(eta_hat)/(2 phi)` on the exact row surface.
    pub(crate) base_scaled_half_deviance: f64,
    /// Its per-row eta gradient `ψ'(η̂)`: the linear Taylor term of `ΔF`, and
    /// the base the sampler moment channels are measured against.
    pub(crate) base_neg_score_at_mode: Array1<f64>,
    /// `Σ_i |D_i(eta_hat)/(2 phi)|`, the absolute sum the base half-deviance
    /// accumulated, for [`BlockExcessTarget::excess_rounding_band`].
    pub(crate) base_absolute_half_deviance: f64,
}

impl Gam784BlockTarget<'_> {
    /// Map a whitened block displacement `t` to the coefficient displacement
    /// `δ = V_b t` and the per-row score `s = X_t δ`.
    pub(crate) fn displacement(&self, t: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let delta = self.block_vecs.dot(t);
        let s = gam_linalg::faer_ndarray::fast_av(self.x_transformed, &delta);
        (delta, s)
    }

    /// Evaluate `D(eta)/(2 phi)` and its full per-row eta gradient atomically.
    /// The row oracle owns both channels, including canonical logit tails and
    /// log-coordinate Bregman algebra for every log-link family.  An invalid
    /// row therefore invalidates the entire draw; no zero-score surrogate is
    /// ever paired with a different objective value.
    pub(crate) fn likelihood_surface_at(
        &self,
        eta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>), EstimationError> {
        if !(self.phi.is_finite() && self.phi > 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "#784 likelihood scale must be finite and positive; got {}",
                self.phi
            )));
        }
        let rows = crate::pirls::deviance_eta_rows_with_log_measure_scale(
            self.y.view(),
            eta,
            &self.likelihood,
            &self.inverse_link,
            self.prior_weights.view(),
            -self.phi.ln(),
        )?;
        let half_values: Vec<f64> = rows.iter().map(|row| row.half_deviance).collect();
        let half_deviance =
            crate::pirls::stable_finite_signed_sum(&half_values, "#784 scaled half-deviance")?;
        let mut score = Array1::<f64>::zeros(rows.len());
        for (i, row) in rows.into_iter().enumerate() {
            let value = row.eta_score;
            if !value.is_finite() {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "scaled deviance eta score",
                    eta: eta[i],
                    value,
                });
            }
            score[i] = value;
        }
        Ok((half_deviance, score))
    }

    pub(crate) fn neg_score_at(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.likelihood_surface_at(eta).map(|(_, score)| score)
    }

    /// `ΔF` from the displaced scaled half-deviance at `η̂ + s`: the row
    /// Taylor remainder `ψ(η̂ + s) − ψ(η̂) − ψ'(η̂)·s − ½ Σ_i W_i s_i²`.
    fn remainder_at(
        &self,
        displaced_scaled_half_deviance: f64,
        s: ndarray::ArrayView1<'_, f64>,
    ) -> Result<f64, EstimationError> {
        let curv = self.observed_quadratic(s)?;
        let value_diff = displaced_scaled_half_deviance - self.base_scaled_half_deviance;
        Ok(value_diff - self.base_neg_score_at_mode.dot(&s) - 0.5 * curv)
    }

    /// `sum_i W_i s_i^2` on an exponent-scaled signed surface.  Squaring `s_i`
    /// before multiplying by a tiny `W_i` can overflow even when the weighted
    /// term is finite; scaling every term by the largest log magnitude avoids
    /// that false refusal.  One deterministic Neumaier pass preserves signed
    /// observed-curvature cancellation.
    pub(crate) fn observed_quadratic(
        &self,
        s: ndarray::ArrayView1<'_, f64>,
    ) -> Result<f64, EstimationError> {
        if s.len() != self.weights_obs.len() {
            return Err(EstimationError::InvalidInput(format!(
                "#784 observed quadratic length mismatch: scores={}, weights={}",
                s.len(),
                self.weights_obs.len()
            )));
        }
        let mut max_log = f64::NEG_INFINITY;
        for i in 0..s.len() {
            if !s[i].is_finite() {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "#784 displacement score",
                    eta: self.eta_hat[i],
                    value: s[i],
                });
            }
            let log_term = if self.weights_obs[i] == 0.0 || s[i] == 0.0 {
                f64::NEG_INFINITY
            } else {
                self.weights_obs_log_abs[i] + 2.0 * s[i].abs().ln()
            };
            max_log = max_log.max(log_term);
        }
        if max_log == f64::NEG_INFINITY {
            return Ok(0.0);
        }
        let mut sum = 0.0_f64;
        let mut compensation = 0.0_f64;
        for i in 0..s.len() {
            if self.weights_obs[i] == 0.0 || s[i] == 0.0 {
                continue;
            }
            let log_term = self.weights_obs_log_abs[i] + 2.0 * s[i].abs().ln();
            let term = self.weights_obs[i].signum() * (log_term - max_log).exp();
            let next = sum + term;
            compensation += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
        }
        let normalized = sum + compensation;
        if normalized == 0.0 {
            return Ok(0.0);
        }
        let value = normalized.signum() * (max_log + normalized.abs().ln()).exp();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(EstimationError::InvalidInput(
                "#784 signed observed quadratic is outside f64 range".to_string(),
            ))
        }
    }
}

impl BlockExcessTarget for Gam784BlockTarget<'_> {
    fn block_dim(&self) -> usize {
        self.block_lambdas.len()
    }

    fn rho_dim(&self) -> usize {
        self.lambdas.len()
    }

    fn block_curvatures(&self) -> &Array1<f64> {
        &self.block_lambdas
    }

    fn excess(&self, t: &Array1<f64>) -> f64 {
        let (_delta, s) = self.displacement(t);
        let eta_disp = &self.eta_hat + &s;
        let Ok((scaled_half_deviance, _score)) = self.likelihood_surface_at(&eta_disp) else {
            return f64::INFINITY;
        };
        self.remainder_at(scaled_half_deviance, s.view())
            .unwrap_or(f64::INFINITY)
    }

    /// The rounding band of [`Self::excess`] at `t`, from what its sums accumulate:
    /// - the displaced and base scaled half-deviances, compensated sums over `n`
    ///   rows. Each row oracle rounds in fewer than `n` operations, so each sum
    ///   carries at most `accumulation_band(n, Σ|row|)`.
    /// - the linear Taylor term `ψ'(η̂)·s`, an inner product over `n` terms.
    /// - the observed quadratic `Σ_i W_i s_i²`, a compensated sum over `n` terms.
    /// - the design product `s = X_t V_b t`, whose entries round within
    ///   `γ_{p(m+1)}·Σ_j |x_ij|·‖δ‖∞`. That moves the displaced surface by at most
    ///   `|ψ'(η̂_i + s_i)|` times it, the linear term by `|ψ'(η̂_i)|` times it and
    ///   the quadratic by `|W_i|·|s_i|` times it.
    ///
    /// A row surface that does not evaluate returns `+∞`, which no bar passes.
    fn excess_rounding_band(&self, t: &Array1<f64>) -> f64 {
        let (delta, s) = self.displacement(t);
        let eta_disp = &self.eta_hat + &s;
        let Ok(rows) = crate::pirls::deviance_eta_rows_with_log_measure_scale(
            self.y.view(),
            &eta_disp,
            &self.likelihood,
            &self.inverse_link,
            self.prior_weights.view(),
            -self.phi.ln(),
        ) else {
            return f64::INFINITY;
        };
        let n = rows.len();
        let displaced_absolute: f64 = rows.iter().map(|row| row.half_deviance.abs()).sum();
        let deviance_band = gam_linalg::roundoff::accumulation_band(n, displaced_absolute)
            + gam_linalg::roundoff::accumulation_band(n, self.base_absolute_half_deviance);
        let p = delta.len();
        let linear_absolute: f64 = self
            .base_neg_score_at_mode
            .iter()
            .zip(s.iter())
            .map(|(score, value)| (score * value).abs())
            .sum();
        let linear_band = gam_linalg::roundoff::accumulation_band(n, linear_absolute);
        let curvature_absolute: f64 = self
            .weights_obs
            .iter()
            .zip(s.iter())
            .map(|(weight, value)| weight.abs() * value * value)
            .sum();
        let curvature_band = gam_linalg::roundoff::accumulation_band(n, curvature_absolute);
        let delta_max = delta.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let design_growth =
            gam_linalg::roundoff::accumulation_growth(p * (self.block_lambdas.len() + 1));
        let mut design_band = 0.0_f64;
        for ((((design_row, row), base_score), weight), value) in self
            .x_transformed
            .rows()
            .into_iter()
            .zip(rows.iter())
            .zip(self.base_neg_score_at_mode.iter())
            .zip(self.weights_obs.iter())
            .zip(s.iter())
        {
            let row_absolute: f64 = design_row.iter().map(|entry| entry.abs()).sum();
            let entry_band = design_growth * row_absolute * delta_max;
            design_band += (row.eta_score.abs() + base_score.abs() + weight.abs() * value.abs())
                * entry_band;
        }
        deviance_band + linear_band + 0.5 * curvature_band + design_band
    }

    /// Zero: with `δ` held fixed in coefficient space, ρ reaches `ΔF` only
    /// through the mode `β̂(ρ)` (see the type's derivation), and that motion is
    /// the evaluator's mode channel, not an explicit one.
    fn excess_rho_gradient(&self, t: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            t.len(),
            self.block_dim(),
            "#784 block displacement length must match the block dimension"
        );
        Array1::<f64>::zeros(self.lambdas.len())
    }

    fn displaced_neg_score(&self, t: &Array1<f64>) -> Result<Array1<f64>, String> {
        let (_delta, s) = self.displacement(t);
        self.neg_score_at(&(&self.eta_hat + &s))
            .map_err(|error| error.to_string())
    }

    fn base_neg_score(&self) -> Result<Array1<f64>, String> {
        Ok(self.base_neg_score_at_mode.clone())
    }

    /// One node of [`Self::excess_with_displaced_neg_score_batch`] (and of
    /// [`Self::excess_batch`], which runs it) holds its columns of `Δ = V_b·T` (p) and
    /// `S = X_t·Δ` (n), each at most twice because `fast_ab`'s small-shape route forms
    /// the product before assigning it; its result entry, displaced score (n) and
    /// excess-only result; and its draw transients: `η̂ + s` (n), the
    /// row oracle's `Result` rows and the certified rows (n each) and the half-deviance
    /// values (n). The transients count for every node, not per worker: the row sweep
    /// is itself a parallel collect, so a worker blocked in it can start another node's
    /// draw.
    fn node_working_bytes(&self) -> Option<usize> {
        let n = self.eta_hat.len();
        let p = self.block_vecs.nrows();
        let row_bytes = std::mem::size_of::<Result<crate::pirls::DevianceEtaRow, EstimationError>>()
            + std::mem::size_of::<crate::pirls::DevianceEtaRow>();
        p.checked_mul(2)?
            .checked_add(n.checked_mul(5)?)?
            .checked_add(1)?
            .checked_mul(std::mem::size_of::<f64>())?
            .checked_add(n.checked_mul(row_bytes)?)?
            .checked_add(std::mem::size_of::<(f64, Option<Array1<f64>>)>())
    }

    /// Fused excess + displaced score sharing one design matvec `s = X_t δ`
    /// and one atomic row-oracle sweep at `η̂ + s`. Each row's value and score
    /// are evaluated together on the same unprojected surface (#784, #1082).
    fn excess_with_displaced_neg_score(&self, t: &Array1<f64>) -> (f64, Option<Array1<f64>>) {
        let (_delta, s) = self.displacement(t);
        let eta_disp = &self.eta_hat + &s;
        let Ok((scaled_half_deviance, ngs)) = self.likelihood_surface_at(&eta_disp) else {
            return (f64::INFINITY, None);
        };
        let Ok(excess) = self.remainder_at(scaled_half_deviance, s.view()) else {
            return (f64::INFINITY, None);
        };
        if excess.is_finite() {
            (excess, Some(ngs))
        } else {
            (excess, None)
        }
    }

    /// Batched excess + displaced score over all importance draws (#784/#1082
    /// hot path). The per-draw cost of [`Self::excess_with_displaced_neg_score`]
    /// is dominated by the design matvec `s = X_t · δ` (O(n·p)), repeated
    /// `n_draws` times (up to 4096). Those matvecs share the SAME design `X_t`
    /// and the SAME block frame `V_b`, so they batch into two dense matrix–matrix
    /// products (BLAS-3) instead of `n_draws` matrix–vector products (BLAS-2):
    ///
    /// ```text
    ///   Δ = V_b · T            (p × n_draws)   T = draws (m × n_draws)
    ///   S = X_t · Δ            (n × n_draws)   one big GEMM, the win
    /// ```
    ///
    /// Column `s` of `S` is exactly `fast_av(X_t, V_b · t_s)` — the same vector
    /// the serial path forms — and everything downstream (the inverse-link jet
    /// sweep, deviance, linear Taylor and curvature terms) is then computed
    /// per-column with byte-for-byte the same arithmetic as the serial
    /// `excess_with_displaced_neg_score`. Only the matvec→GEMM reassociation can
    /// perturb `S` (faer reduces the inner `p`-sum the same way per output
    /// element regardless of the RHS column count, so this is at the level of
    /// floating-point reassociation, not a different estimator).
    fn excess_with_displaced_neg_score_batch(
        &self,
        draws: &Array2<f64>,
    ) -> Vec<(f64, Option<Array1<f64>>)> {
        let m = self.block_lambdas.len();
        let n = self.eta_hat.len();
        let n_draws = draws.ncols();
        assert_eq!(
            draws.nrows(),
            m,
            "posterior displacement draw rows must match smoothing block count"
        );

        // δ-columns: Δ = V_b · T  (p × n_draws). Cheap (O(p·m·n_draws)) and kept
        // identical to the serial `block_vecs.dot(t)` per column.
        let delta_all = gam_linalg::faer_ndarray::fast_ab(&self.block_vecs, draws);
        // s-columns: S = X_t · Δ  (n × n_draws). THE batched matvec — one GEMM
        // replacing `n_draws` separate `fast_av(x_transformed, δ_s)` calls.
        let s_all = gam_linalg::faer_ndarray::fast_ab(self.x_transformed, &delta_all);

        // Parallelise over DRAWS, which is where the independent work is.
        //
        // This loop used to be serial, and each iteration called
        // `likelihood_surface_at` -> `deviance_eta_rows_with_log_measure_scale`,
        // which fans out over the `n` rows with `into_par_iter()`. So the outer
        // dimension (many genuinely independent draws) ran on one thread while
        // the inner one (a single sweep of ~1e3 cheap rows) paid a fork/join
        // round on every draw. That is the wrong level: the row sweep is tens of
        // microseconds of arithmetic, and scheduling it costs the same order.
        //
        // Measured on the geo_latlon fuzz family (n=960, p=11, binomial-logit),
        // where this sampler is 45% of the profile: wall clock RISES with core
        // count -- 205.0s at 1 core, 264.3s at 4, 304.3s at 16, 555.9s at 32.
        // More cores made it 2.7x slower, because every additional worker is
        // another thief splitting a sweep too small to be worth splitting.
        //
        // Order is preserved: this is an indexed map into a `Vec`, so draw `s`
        // still lands at position `s` and the result is bit-identical to the
        // serial loop.
        let out: Vec<(f64, Option<Array1<f64>>)> = (0..n_draws)
            .into_par_iter()
            .map(|sidx| {
                let s_col = s_all.column(sidx);
                let mut eta_disp = self.eta_hat.clone();
                for i in 0..n {
                    eta_disp[i] += s_col[i];
                }
                let Ok((scaled_half_deviance, ngs)) = self.likelihood_surface_at(&eta_disp) else {
                    return (f64::INFINITY, None);
                };
                let Ok(excess) = self.remainder_at(scaled_half_deviance, s_col) else {
                    return (f64::INFINITY, None);
                };
                if excess.is_finite() {
                    (excess, Some(ngs))
                } else {
                    (excess, None)
                }
            })
            .collect();
        out
    }

    fn excess_batch(&self, nodes: &Array2<f64>) -> Vec<f64> {
        // Preserve the BLAS-3 displacement path for the coarse rule. The row
        // oracle currently produces value and score together atomically; drop
        // the unused score here without falling back to one BLAS-2 matvec per
        // node.
        self.excess_with_displaced_neg_score_batch(nodes)
            .into_iter()
            .map(|(excess, _)| excess)
            .collect()
    }
}

#[cfg(test)]
mod exact_deviance_state_cache_tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn observed_quadratic_scales_before_squaring_and_preserves_sign() {
        let x = Array2::<f64>::zeros((2, 1));
        let weights_obs = array![1.0e-320_f64, -1.0e-320_f64];
        let weights_obs_log_abs = weights_obs.mapv(|weight| weight.abs().ln());
        let target = Gam784BlockTarget {
            x_transformed: &x,
            block_vecs: Array2::zeros((1, 1)),
            block_lambdas: array![1.0],
            eta_hat: array![0.0, 0.0],
            weights_obs,
            weights_obs_log_abs,
            y: array![0.0, 0.0],
            prior_weights: array![1.0, 1.0],
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            )),
            inverse_link: InverseLink::Standard(StandardLink::Log),
            phi: 1.0,
            penalty_scores: Arc::new(Vec::new()),
            penalties: &[],
            lambdas: Vec::new(),
            base_scaled_half_deviance: 0.0,
            base_neg_score_at_mode: array![0.0, 0.0],
            base_absolute_half_deviance: 0.0,
        };
        let s = array![1.0e200, 5.0e199];
        let observed = target
            .observed_quadratic(s.view())
            .expect("weighted quadratic");
        let first = (target.weights_obs[0].ln() + 2.0 * s[0].ln()).exp();
        let second = (target.weights_obs[1].abs().ln() + 2.0 * s[1].ln()).exp();
        let expected = first - second;
        approx::assert_relative_eq!(observed, expected, max_relative = 2.0e-14);
    }

    /// `ΔF` is the definition `F(β̂+δ) − F(β̂) − ½ δᵀ H δ` at an exact mode, and
    /// reads nothing of the penalty.
    ///
    /// The mode is exact by construction: a Poisson-log likelihood at an
    /// arbitrary `β̂`, and the rank-one penalty `S = g gᵀ/(−g·β̂)` for
    /// `g = Xᵀ ψ'(η̂)`, which gives `S β̂ = −g`. The target is handed penalty
    /// scores that are a stiff `λ = 1e12` times a residue of `3e-15` — what a
    /// railed smoothing parameter does to `S_k β̂` in floating point — and must
    /// match the definition regardless. Rebuilding `(S β̂)·δ` from those scores
    /// put `3e-3·δ` into `ΔF`, and that noise stopped the outer search on a
    /// near-separated binomial CV fold.
    #[test]
    fn remainder_is_the_taylor_definition_at_an_exact_mode_and_ignores_penalty_rounding() {
        let x = array![
            [1.0, 0.2],
            [1.0, -0.7],
            [1.0, 1.3],
            [1.0, 0.4],
            [1.0, -1.1],
            [1.0, 0.9]
        ];
        let y = array![3.0, 0.0, 6.0, 2.0, 1.0, 4.0];
        let beta_hat = array![0.3, 0.1];
        let eta_hat = x.dot(&beta_hat);
        let mu = eta_hat.mapv(f64::exp);
        let psi = |eta: &Array1<f64>| -> f64 {
            eta.iter()
                .zip(y.iter())
                .map(|(&e, &yi): (&f64, &f64)| {
                    let m = e.exp();
                    let log_ratio = if yi > 0.0 { yi * (yi.ln() - e) } else { 0.0 };
                    log_ratio - (yi - m)
                })
                .sum()
        };
        let g = x.t().dot(&(&mu - &y));
        let curvature = -g.dot(&beta_hat);
        assert!(curvature > 0.0, "fixture needs g·β̂ < 0 for a PSD penalty");
        let s_pen = {
            let mut s_pen = Array2::<f64>::zeros((2, 2));
            for a in 0..2 {
                for b in 0..2 {
                    s_pen[(a, b)] = g[a] * g[b] / curvature;
                }
            }
            s_pen
        };
        let mode_residual = &g + &s_pen.dot(&beta_hat);
        assert!(mode_residual.iter().all(|r| r.abs() < 1.0e-12));

        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ));
        let inverse_link = InverseLink::Standard(StandardLink::Log);
        let prior_weights = Array1::<f64>::ones(6);
        let base_rows = crate::pirls::deviance_eta_rows_with_log_measure_scale(
            y.view(),
            &eta_hat,
            &likelihood,
            &inverse_link,
            prior_weights.view(),
            0.0,
        )
        .expect("base rows");
        let base_half: Vec<f64> = base_rows.iter().map(|row| row.half_deviance).collect();
        let base_scaled_half_deviance: f64 = base_half.iter().sum();
        let base_absolute_half_deviance: f64 = base_half.iter().map(|v| v.abs()).sum();
        let base_neg_score_at_mode = Array1::from_iter(base_rows.iter().map(|row| row.eta_score));
        let weights_obs = mu.clone();
        let weights_obs_log_abs = weights_obs.mapv(f64::ln);
        let target = Gam784BlockTarget {
            x_transformed: &x,
            block_vecs: Array2::eye(2),
            block_lambdas: array![1.0, 1.0],
            eta_hat: eta_hat.clone(),
            weights_obs,
            weights_obs_log_abs,
            y: y.clone(),
            prior_weights,
            likelihood,
            inverse_link,
            phi: 1.0,
            penalty_scores: Arc::new(vec![array![3.0e-15, -3.0e-15]]),
            penalties: &[],
            lambdas: vec![1.0e12],
            base_scaled_half_deviance,
            base_neg_score_at_mode,
            base_absolute_half_deviance,
        };

        let h = x.t().dot(&Array2::from_diag(&mu).dot(&x)) + &s_pen;
        let objective = |beta: &Array1<f64>| psi(&x.dot(beta)) + 0.5 * beta.dot(&s_pen.dot(beta));
        let draws = array![[0.4, -0.25, 0.05], [-0.3, 0.6, 0.02]];
        let batch = target.excess_with_displaced_neg_score_batch(&draws);
        for (column, (batched, _)) in draws.columns().into_iter().zip(batch) {
            let t = column.to_owned();
            let definition = objective(&(&beta_hat + &t)) - objective(&beta_hat) - 0.5 * t.dot(&h.dot(&t));
            let excess = target.excess(&t);
            let band = target.excess_rounding_band(&t);
            assert!(
                (excess - definition).abs() <= 1.0e-12 * definition.abs().max(1.0) + band,
                "ΔF {excess:.17e} against the definition {definition:.17e} at t = {t:?}"
            );
            assert!((excess - batched).abs() <= band, "batched ΔF {batched:.17e} against {excess:.17e}");
            assert!(target.excess_rho_gradient(&t).iter().all(|&v| v == 0.0));
        }
    }
}

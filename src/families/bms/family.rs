use super::*;

use crate::families::fnv1a::Fnv1a;

#[derive(Clone)]
pub(super) struct BernoulliMarginalSlopeFamily {
    pub(super) y: Arc<Array1<f64>>,
    pub(super) weights: Arc<Array1<f64>>,
    pub(super) z: Arc<Array1<f64>>,
    pub(super) latent_measure: LatentMeasureKind,
    pub(super) gaussian_frailty_sd: Option<f64>,
    pub(super) base_link: InverseLink,
    pub(super) marginal_design: DesignMatrix,
    pub(super) logslope_design: DesignMatrix,
    pub(super) score_warp: Option<DeviationRuntime>,
    pub(super) link_dev: Option<DeviationRuntime>,
    /// Resource policy controlling materialization decisions for psi design
    /// resolution and other size-sensitive helpers invoked during exact-Newton
    /// joint psi calculus. Threaded from the fit entry point so large-scale
    /// runs pick up the caller's analytic-operator preference instead of an
    /// inline default.
    pub(super) policy: gam_runtime::resource::ResourcePolicy,
    /// Fit-lifetime byte-limited LRU for de-nested cubic cell moments. The key
    /// is the exact bit pattern of `(c0, c1, c2, c3, left, right)`, so reuse
    /// across PIRLS cycles is safe only for byte-identical cells while LRU
    /// eviction never changes numerical results.
    pub(super) cell_moment_lru: Arc<exact_kernel::CellMomentLruCache>,
    pub(super) cell_moment_cache_stats: Arc<exact_kernel::CellMomentCacheStats>,
    /// Per-row warm-start cache for the scalar intercept root-finder
    /// (`solve_row_intercept_base`). The intercept `a` is solved per row at
    /// every inner PIRLS iteration; without a warm start, each call burns
    /// ~10–20 root-solver iterations re-deriving the same answer from the
    /// closed-form rigid/affine seed. Across consecutive PIRLS iterations β
    /// moves only a little, so the previous iter's converged `a` is an
    /// excellent initial guess and typically lets the root-solver finish in
    /// 1–2 iterations.
    ///
    /// Slots are initialised to `NaN` (sentinel for "not yet solved") and
    /// overwritten with the converged intercept on every successful call.
    /// Set to `None` for unit-test fixtures that build a
    /// `BernoulliMarginalSlopeFamily` directly without running the full fit
    /// pipeline; production paths go through `make_family` which initialises
    /// the cache to length-`n` NaN.
    pub(super) intercept_warm_starts: Option<Arc<BernoulliInterceptWarmStartCache>>,
    /// Per-fit counter of outer rho-gradient evaluations. Increments
    /// on every call to `batched_outer_gradient_terms`. Drives the
    /// two-phase auto-subsample schedule: while
    /// `count < BMS_AUTO_SUBSAMPLE_PHASE1_BUDGET` and
    /// `auto_outer_subsample` is enabled, the family installs a
    /// stratified Horvitz–Thompson mask. Once the budget is
    /// exhausted, every subsequent eval reverts to full data so the
    /// final BFGS iterations satisfy the user's tight `outer_tol`
    /// without paying any noise floor.
    ///
    /// Each new fit constructs a fresh family (the counter starts at
    /// zero), so the schedule resets per fit without any cross-fit
    /// leakage. Atomic so the field is safe to clone via Arc.
    pub(super) auto_subsample_phase_counter: Arc<std::sync::atomic::AtomicUsize>,
    /// Last ρ vector at which the auto-subsample phase counter was
    /// bumped. BFGS line searches re-call `batched_outer_gradient_terms`
    /// at the same ρ during step-size retries; without this guard the
    /// per-call increment burns the Phase-1 budget on those retries
    /// instead of on distinct outer iterations. We bump the counter only
    /// when the incoming ρ differs (L2) from the last one we saw, so the
    /// budget exactly counts distinct outer steps. The mutex is the
    /// minimal coordination needed: the counter+last_rho pair must be
    /// updated atomically so two threads cannot both decide "new ρ" and
    /// double-bump.
    pub(super) auto_subsample_last_rho: Arc<Mutex<Option<Array1<f64>>>>,
}

/// Number of outer-gradient evaluations the auto-subsample schedule
/// spends in Phase 1 (stratified subsample, ≈ 1 % gradient noise).
/// After this many calls the family reverts to full data for all
/// remaining outer evaluations, so BFGS/ARC can drive `‖∇‖` below the
/// user's tight `outer_tol`. The budget is sized so that BFGS can
/// reduce a generic ρ-gradient by ≈ 2–3 decades in Phase 1 (typical
/// L-BFGS rate of one decade per ~5 iterations on a noisy gradient,
/// stalling at the noise floor) before switching to exact Phase-2
/// polish.
#[derive(Clone)]
pub(super) struct BernoulliInterceptPredictorWarmStart {
    pub(super) intercept: f64,
    pub(super) primary_point: Vec<f64>,
    pub(super) intercept_primary_deriv: Vec<f64>,
}

/// Per-row warm-start cache for the scalar intercept root-finder.
///
/// Each slot stores `(value, beta_tag)` where `beta_tag` is a 64-bit hash of
/// the per-row state that uniquely determines the intercept root. Reads return
/// `Some(a)` only when the caller's tag matches the stored tag AND the stored
/// value is finite. This makes the cache transactional with respect to
/// trust-region trials and subsampled probes: a rejected trial at β_A and an
/// accepted full-data eval at β_B key under distinct tags, so writes from one
/// cannot poison reads from the other.
///
/// The "never written" sentinel is `beta_tag == 0`. Tag helpers
/// (`hash_intercept_warm_start_key_*`) remap `0` to `1` so the sentinel can
/// never collide with a real key.
///
/// Memory ordering: the writer stores `value` with `Relaxed` and then `tag`
/// with `Release`; the reader loads `tag` with `Acquire`, reads `value` with
/// `Relaxed`, and re-checks `tag` with `Acquire`. The double-check detects a
/// torn read where another thread interleaved a tag bump between the value
/// read and the second tag load.
pub(super) struct BernoulliInterceptWarmStartCache {
    pub(super) intercept_value: Vec<AtomicU64>,
    pub(super) intercept_tag: Vec<AtomicU64>,
    pub(super) predictors: Vec<Mutex<Option<BernoulliInterceptPredictorWarmStart>>>,
}

impl BernoulliInterceptWarmStartCache {
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.intercept_value.len()
    }

    /// Return the cached intercept iff the slot's stored `beta_tag` matches
    /// the caller's `beta_tag` and the stored value is finite.
    #[inline]
    pub(super) fn load_tagged(&self, row: usize, beta_tag: u64) -> Option<f64> {
        let value_slot = self.intercept_value.get(row)?;
        let tag_slot = self.intercept_tag.get(row)?;
        let tag_before = tag_slot.load(Ordering::Acquire);
        if tag_before != beta_tag {
            return None;
        }
        let bits = value_slot.load(Ordering::Relaxed);
        let tag_after = tag_slot.load(Ordering::Acquire);
        if tag_after != beta_tag {
            return None;
        }
        let value = f64::from_bits(bits);
        value.is_finite().then_some(value)
    }

    /// Stamp the slot with the converged intercept under `beta_tag`.
    #[inline]
    pub(super) fn store_tagged(&self, row: usize, value: f64, beta_tag: u64) {
        if let (Some(value_slot), Some(tag_slot)) =
            (self.intercept_value.get(row), self.intercept_tag.get(row))
        {
            // Invalidate before writing the new value so an interleaved
            // reader cannot see the new tag paired with the old value.
            tag_slot.store(0, Ordering::Release);
            value_slot.store(value.to_bits(), Ordering::Relaxed);
            tag_slot.store(beta_tag, Ordering::Release);
        }
    }

    /// CAS-install `(value, beta_tag)` into a slot only if the tag slot is
    /// still the "never written" sentinel (`0`). Returns `Ok(())` if the seed
    /// was installed, `Err(prev_tag)` if some prior write already populated
    /// the slot (in which case the caller should keep the existing entry).
    #[inline]
    pub(super) fn compare_exchange_unseeded(
        &self,
        row: usize,
        value: f64,
        beta_tag: u64,
    ) -> Result<(), u64> {
        let value_slot = self.intercept_value.get(row).ok_or(0u64)?;
        let tag_slot = self.intercept_tag.get(row).ok_or(0u64)?;
        match tag_slot.compare_exchange(0, beta_tag, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                // We own the tag; publish the value. A late reader that loads
                // `tag == beta_tag` and then `value == NaN` will reject via
                // `is_finite()` and fall back to the closed-form seed.
                value_slot.store(value.to_bits(), Ordering::Relaxed);
                Ok(())
            }
            Err(prev) => Err(prev),
        }
    }

    pub(super) fn predictor_seed(&self, row: usize, current_point: &[f64]) -> Option<f64> {
        let warm = self.predictors.get(row)?.lock().ok()?.as_ref().cloned()?;
        if warm.primary_point.len() != current_point.len()
            || warm.intercept_primary_deriv.len() != current_point.len()
            || !warm.intercept.is_finite()
        {
            return None;
        }
        let correction = warm
            .intercept_primary_deriv
            .iter()
            .zip(current_point.iter().zip(warm.primary_point.iter()))
            .map(|(a_u, (new, old))| a_u * (new - old))
            .sum::<f64>();
        let seed = warm.intercept + correction;
        seed.is_finite().then_some(seed)
    }

    pub(super) fn store_predictor(
        &self,
        row: usize,
        intercept: f64,
        primary_point: Vec<f64>,
        intercept_primary_deriv: Vec<f64>,
    ) {
        if !intercept.is_finite()
            || primary_point.iter().any(|value| !value.is_finite())
            || intercept_primary_deriv
                .iter()
                .any(|value| !value.is_finite())
        {
            return;
        }
        let Some(slot) = self.predictors.get(row) else {
            return;
        };
        if let Ok(mut guard) = slot.lock() {
            *guard = Some(BernoulliInterceptPredictorWarmStart {
                intercept,
                primary_point,
                intercept_primary_deriv,
            });
        }
    }
}

pub(super) fn new_intercept_warm_start_cache(n: usize) -> Arc<BernoulliInterceptWarmStartCache> {
    Arc::new(BernoulliInterceptWarmStartCache {
        intercept_value: (0..n).map(|_| AtomicU64::new(f64::NAN.to_bits())).collect(),
        intercept_tag: (0..n).map(|_| AtomicU64::new(0)).collect(),
        predictors: (0..n).map(|_| Mutex::new(None)).collect(),
    })
}

/// FNV-1a 64-bit hash of per-row state determining the empirical-grid rigid
/// intercept root. The root depends only on `(marginal.q, slope)` (the grid
/// nodes/weights are immutable per `latent_measure`), so hashing these two
/// scalars is sufficient to distinguish trust-region trials at different β.
/// Returned tag is guaranteed non-zero (zero is remapped to one) so the
/// cache's "never written" sentinel cannot collide with a real key.
#[inline]
pub(super) fn hash_intercept_warm_start_key_rigid(marginal_q: f64, slope: f64) -> u64 {
    let mut hash = Fnv1a::new();
    // Domain separator for the rigid (empirical-grid) cache stream.
    hash.mix_byte(0xb1);
    hash.mix_f64(marginal_q);
    hash.mix_f64(slope);
    hash.finish_nonzero()
}

/// FNV-1a 64-bit hash of per-row state determining the FLEX intercept root.
/// The root depends on `(marginal_eta, slope, beta_h, beta_w)`: under
/// link-deviation and score-warp the joint β vector enters via the link
/// basis evaluated at the intercept, so trials at different β at the same
/// row produce different roots and must NOT share a cache slot.
#[inline]
pub(super) fn hash_intercept_warm_start_key_flex(
    marginal_eta: f64,
    slope: f64,
    beta_h: Option<&Array1<f64>>,
    beta_w: Option<&Array1<f64>>,
) -> u64 {
    let mut hash = Fnv1a::new();
    // Domain separator for the FLEX cache stream so it cannot collide with
    // a rigid-stream hash that happens to produce the same scalar bits.
    hash.mix_byte(0xb2);
    hash.mix_f64(marginal_eta);
    hash.mix_f64(slope);
    hash.mix_opt_beta(0xc1, beta_h);
    hash.mix_opt_beta(0xc2, beta_w);
    hash.finish_nonzero()
}

#[derive(Clone, Default)]
pub(super) struct ThetaHints {
    pub(super) marginal_beta: Option<Array1<f64>>,
    pub(super) logslope_beta: Option<Array1<f64>>,
    pub(super) score_warp_beta: Option<Array1<f64>>,
    pub(super) link_dev_beta: Option<Array1<f64>>,
}

// `pub` (was `pub(crate)`) so the cross-crate `gam-predict` saved-runtime tests
// can drive this builder after the #1521 split moved them out of `gam` (#1567).
pub fn build_score_warp_deviation_block_from_seed(
    seed: &Array1<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<DeviationPrepared, String> {
    build_deviation_block_from_knots_and_design_seed(seed, seed, cfg)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BernoulliMarginalLinkMap {
    /// The marginal linear predictor η this map was expanded about. The q-stack
    /// `[q, q1, ..]` are the derivatives of `q(·) = Φ⁻¹(Φ(·))` at this η, so the
    /// generic-jet path (#932 cutover) must seed its axis-0 primary here for the
    /// composition's expansion point to be honest.
    pub eta: f64,
    pub mu: f64,
    pub mu1: f64,
    pub mu2: f64,
    pub mu3: f64,
    pub mu4: f64,
    pub q: f64,
    pub q1: f64,
    pub q2: f64,
    pub q3: f64,
    pub q4: f64,
}

#[inline]
pub(super) fn clamp_bernoulli_link_probability(probability: f64) -> f64 {
    probability.clamp(
        BERNOULLI_LINK_PROBABILITY_EPS,
        1.0 - BERNOULLI_LINK_PROBABILITY_EPS,
    )
}

pub(crate) fn bernoulli_marginal_slope_eta_from_probability(
    base_link: &InverseLink,
    probability: f64,
    context: &str,
) -> Result<f64, String> {
    require_probit_marginal_slope_link(base_link, context)?;
    let target = clamp_bernoulli_link_probability(probability);
    standard_normal_quantile(target)
        .map_err(|e| format!("{context} failed to invert probit probability {target}: {e}"))
}

pub(crate) fn bernoulli_marginal_link_map(
    base_link: &InverseLink,
    eta: f64,
) -> Result<BernoulliMarginalLinkMap, String> {
    require_probit_marginal_slope_link(base_link, "bernoulli marginal-slope")?;
    let raw_mu = normal_cdf(eta);
    let mu = clamp_bernoulli_link_probability(raw_mu);
    let q = standard_normal_quantile(mu).map_err(|e| {
        format!("bernoulli marginal-slope probit target inversion failed at mu={mu}: {e}")
    })?;
    if raw_mu <= BERNOULLI_LINK_PROBABILITY_EPS || raw_mu >= 1.0 - BERNOULLI_LINK_PROBABILITY_EPS {
        return Ok(BernoulliMarginalLinkMap {
            eta,
            mu,
            mu1: 0.0,
            mu2: 0.0,
            mu3: 0.0,
            mu4: 0.0,
            q,
            q1: 0.0,
            q2: 0.0,
            q3: 0.0,
            q4: 0.0,
        });
    }
    let phi_eta = normal_pdf(eta);
    let phi_q = normal_pdf(q);
    if !phi_q.is_finite() || phi_q <= 0.0 {
        return Err(format!(
            "bernoulli marginal-slope internal probit density must be positive, got phi(q)={phi_q} at eta={eta}, q={q}"
        ));
    }
    let mu1 = phi_eta;
    let mu2 = -eta * phi_eta;
    let mu3 = (eta * eta - 1.0) * phi_eta;
    let mu4 = -(eta.powi(3) - 3.0 * eta) * phi_eta;
    let q1 = mu1 / phi_q;
    let q1_sq = q1 * q1;
    let q1_cu = q1_sq * q1;
    let q1_q = q1_sq * q1_sq;
    let q2 = mu2 / phi_q + q * q1_sq;
    let q3 = mu3 / phi_q + 3.0 * q * q1 * q2 - (q * q - 1.0) * q1_cu;
    let q4 = mu4 / phi_q + (q.powi(3) - 3.0 * q) * q1_q + 4.0 * q * q1 * q3 + 3.0 * q * q2 * q2
        - 6.0 * (q * q - 1.0) * q1_sq * q2;
    Ok(BernoulliMarginalLinkMap {
        eta,
        mu,
        mu1,
        mu2,
        mu3,
        mu4,
        q,
        q1,
        q2,
        q3,
        q4,
    })
}

impl BernoulliMarginalLinkMap {
    /// The marginal linear predictor η this map was expanded about — the seed
    /// value for the generic-jet path's axis-0 primary.
    #[inline]
    pub(crate) fn eta_value(&self) -> f64 {
        self.eta
    }
}

pub(super) fn require_probit_marginal_slope_link(
    base_link: &InverseLink,
    context: &str,
) -> Result<(), String> {
    if matches!(base_link, InverseLink::Standard(StandardLink::Probit)) {
        Ok(())
    } else {
        Err(format!(
            "{context} requires link(type=probit); non-probit marginal-slope base links are not supported by the calibrated de-nested probit kernel"
        ))
    }
}

pub(crate) fn build_link_deviation_block_from_knots_design_seed_and_weights(
    knot_seed: &Array1<f64>,
    design_seed: &Array1<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<DeviationPrepared, String> {
    build_deviation_block_from_knots_and_design_seed(knot_seed, design_seed, cfg)
}

pub(super) fn build_deviation_block_from_knots_and_design_seed(
    knot_seed: &Array1<f64>,
    design_seed: &Array1<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<DeviationPrepared, String> {
    if cfg.degree != 3 {
        return Err(format!(
            "structural deviation runtime is cubic; degree must be 3, got {}",
            cfg.degree
        ));
    }
    let penalty_orders = resolve_deviation_operator_orders(cfg)?;
    let knots =
        initializewiggle_knots_from_seed(knot_seed.view(), cfg.degree, cfg.num_internal_knots)?;
    // The smoothness-null-space drop must remove the union of null spaces
    // across all configured penalties, which (for nested null spaces of
    // increasing-order derivative penalties) equals the largest order's
    // null space. Thus we drop polynomials of degree < max_order.
    let max_penalty_order = penalty_orders.iter().copied().max().ok_or_else(|| {
        "deviation block requires at least one positive function-penalty derivative order"
            .to_string()
    })?;
    let runtime = DeviationRuntime::try_new(knots, cfg.monotonicity_eps, max_penalty_order)?;
    let design = runtime.design(design_seed)?;
    let p = design.ncols();
    if p == 0 {
        return Err("structural deviation basis has no free derivative controls".to_string());
    }
    let mut block = ParameterBlockInput {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
        offset: Array1::zeros(design_seed.len()),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: None,
        initial_beta: Some(Array1::zeros(p)),
    };
    for order in penalty_orders {
        append_deviation_function_penalty(&mut block, &runtime, order)?;
    }
    if cfg.double_penalty {
        append_deviation_function_penalty(&mut block, &runtime, 0)?;
    }
    Ok(DeviationPrepared { block, runtime })
}

pub(super) fn resolve_deviation_operator_orders(
    cfg: &DeviationBlockConfig,
) -> Result<Vec<usize>, String> {
    let mut orders = Vec::new();
    let requested = if cfg.penalty_orders.is_empty() {
        std::slice::from_ref(&cfg.penalty_order)
    } else {
        cfg.penalty_orders.as_slice()
    };
    for &order in requested {
        if order == 0 {
            continue;
        }
        if order > cfg.degree {
            return Err(format!(
                "deviation function penalty derivative order {order} exceeds basis degree {}",
                cfg.degree
            ));
        }
        if !orders.contains(&order) {
            orders.push(order);
        }
    }
    if orders.is_empty() {
        return Err(
            "deviation block requires at least one positive function-penalty derivative order"
                .to_string(),
        );
    }
    Ok(orders)
}

pub(super) fn append_deviation_function_penalty(
    block: &mut ParameterBlockInput,
    runtime: &DeviationRuntime,
    derivative_order: usize,
) -> Result<(), String> {
    let (penalty, nullity) =
        runtime.integrated_derivative_penalty_with_nullity(derivative_order)?;
    block
        .penalties
        .push(crate::model_types::PenaltySpec::Dense(penalty));
    block.nullspace_dims.push(nullity);
    Ok(())
}

// Cross-block identifiability for the BMS family's parametric and flex
// blocks. Each deviation block's basis is individually orthogonal to its
// own smoothness-penalty null space (`smoothness_nullspace_orthogonal_complement`
// inside `DeviationRuntime::try_new`), but that only makes each block
// identifiable in isolation. Two flex blocks of overlapping argument
// classes (or a flex block whose column span at training rows reproduces
// parametric features) leave a near-null direction in the joint penalised
// Hessian: a linear combination of `β` across blocks produces zero net
// η-contribution at training rows yet costs only the (penalised) basis
// norm. Newton steps blow up along that direction and the inner solver
// either drifts indefinitely along the null mode or, at large scale,
// breaks the constrained QP active-set iteration once `W = p(1−p)` further
// degrades the data Hessian.
//
// The principled fix is the standard GAM identifiability convention
// (Wood, §5.4 / mgcv `gam.side`) generalised to multi-anchor unions:
// reparameterise each later block so its column span at training rows is
// orthogonal — in the W-metric — to the union of every earlier block's
// column span. Stack the parametric anchors into `A` (n × d) and project
// the candidate basis `C` (n × p_c) onto the W-orthogonal complement of
// span(A): `C̃ = (I − P_A^{(W)}) C` with `P_A^{(W)} = A (AᵀWA)⁻¹ AᵀW`.
// Keep the columns of `C̃ V` whose `C̃ᵀ W C̃` eigenvalues are above the
// numerical noise floor; absorb the projection into a residual `M = R K_w V`
// stored alongside the runtime so each evaluated row computes
// `design_row = pure_span_row · V − n_row · M`. The joint design then has
// full numerical column rank under the W inner product (the actual row
// metric of the Hessian build at large scale), `σ_min(joint H + S) ≥
// λ_min(S₊)` regardless of how β shifts the linear-predictor distribution,
// and the soft "+∞" / divergence-detection / trust-region-collapse-as-KKT
// scaffolding in the inner solver becomes vestigial rather than
// load-bearing.
//
// Subtle but important: the *old* algorithm computed `T = null(AᵀC)` —
// candidate directions whose Gram with the anchor is exactly zero.
// `null(AᵀC) ≠ ∅` is NOT the same as `span(C) ⊆ span(A)`. Counterexample:
// `A = [e₁]`, `C = [e₁ + e₂]`. Then `AᵀC = [1] ≠ 0` so `null(AᵀC) = ∅`
// and the old algorithm declared "fully aliased", even though
// `(I − P_A) C = [e₂]` carries a full independent direction. The
// residualisation theorem is `span(C) ⊆ span(A) ⇔ (I − P_A) C = 0`, and
// that is what this code actually tests.
//
// `install_compiled_flex_block_into_runtime` is a thin wrapper:
//
//   1. `build_bms_flex_block_context` — densifies anchors, stacks N_train,
//      and assembles the `BernoulliDenseDesignOperator` + `BlockOrder` vectors

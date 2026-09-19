use super::*;

use crate::fnv1a::Fnv1a;
use crate::latent_anchor::{
    AnchorGrid, AnchorGridOwned, AnchorRootCache, AnchorRowContext, AnchorSolveCounts,
};

#[derive(Clone)]
pub(super) struct BernoulliMarginalSlopeFamily {
    pub(super) y: Arc<Array1<f64>>,
    pub(super) weights: Arc<Array1<f64>>,
    pub(super) z: Arc<Array1<f64>>,
    pub(super) latent_measure: LatentMeasureKind,
    pub(super) gaussian_frailty_sd: Option<f64>,
    pub(super) base_link: InverseLink,
    pub(super) marginal_design: DesignMatrix,
    pub(super) slope_design: DesignMatrix,
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
    /// Fit-lifetime pools of runtime-sized jet workspaces for the empirical
    /// FLEX third and fourth contractions and the third trace. Idle
    /// workspaces are charged to the governor and freed with the fit's
    /// families (gam#2989).
    pub(super) jet_scratch: Arc<super::hessian_paths::JetScratch>,
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
    /// `count < AUTO_OUTER_PHASE1_BUDGET` and
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
    /// Whether this member's Jeffreys/Firth prior is armed. A fit arms it only
    /// on the unarmed route's own evidence: `fit_bernoulli_marginal_slope_terms`
    /// runs the whole route through `arm_on_evidence` (#979, #3164).
    pub(super) jeffreys_armed: bool,
    /// The residual genetic repair block (gam#2924): `K` conditionally centred
    /// features with constant coefficients entering the genetic drive beside
    /// the score, and the joint `(z, r)` covariance the anchor integrates.
    /// `Some` routes every rigid-path consumer through the residual row kernel
    /// (`RowKernel<2+K>`); `None` is the two-primary family unchanged.
    pub(super) residual: Option<Arc<super::residual_repair::ResidualBlockRuntime>>,
    /// This member's own search state in a parallel multistart (gnomon#2359).
    /// `None` outside a multistart.
    pub(super) search: Option<Arc<BmsSearchMember>>,
}

/// One outer search's own state in a parallel multistart (gnomon#2359).
///
/// The lane's memory reading drives its row-primary cache decision, and its pins
/// are charged to the lane. The same-β stores hold this search's builds only,
/// where a fit on its own keeps them process-wide. An exact cache carries the
/// row intercept roots its builder's warm starts converged to, and a hit skips
/// the root solves that seed a search's next warm start, so a shared store let a
/// search read other searches' roots and lose its own builds to their evictions,
/// and its result depended on which searches ran beside it.
pub(super) struct BmsSearchMember {
    pub(super) lane: Arc<gam_runtime::resource::SearchLaneBudget>,
    pub(super) exact_caches: Mutex<super::cell_moment_assembly::SharedExactCacheStore>,
    pub(super) rigid_tensors: Mutex<super::row_kernel::SharedRigidTensorStore>,
}

impl BmsSearchMember {
    pub(super) fn new(lane: Arc<gam_runtime::resource::SearchLaneBudget>) -> Self {
        Self {
            lane,
            exact_caches: Mutex::new(super::cell_moment_assembly::SharedExactCacheStore::empty()),
            rigid_tensors: Mutex::new(super::row_kernel::SharedRigidTensorStore::empty()),
        }
    }
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
    /// The law the rigid empirical intercept anchors on, with the root slots it
    /// is solved in; `None` for a cache built without one, whose rows solve cold.
    pub(super) anchor_law: Option<BernoulliAnchorLaw>,
}

impl BernoulliInterceptWarmStartCache {
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.intercept_value.len()
    }

    /// A cache of the same rows on the same law holding nothing. Each multistart
    /// search starts from one, as a freshly built family does (gnomon#2359), so no
    /// search reads another's warm starts or root slots.
    pub(super) fn empty_like(&self) -> Arc<Self> {
        Arc::new(warm_start_cache(
            self.len(),
            self.anchor_law.as_ref().map(BernoulliAnchorLaw::empty_like),
        ))
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

/// The per-row intercept warm-start cache, carrying the fit's latent law and the
/// root slots its rigid empirical intercepts are solved in (gam#2926).
pub(super) fn new_intercept_warm_start_cache_on_law(
    latent_measure: &LatentMeasureKind,
    n: usize,
) -> Result<Arc<BernoulliInterceptWarmStartCache>, String> {
    let anchor_law = BernoulliAnchorLaw::from_kind(latent_measure, n)?;
    Ok(Arc::new(warm_start_cache(n, anchor_law)))
}

fn warm_start_cache(
    n: usize,
    anchor_law: Option<BernoulliAnchorLaw>,
) -> BernoulliInterceptWarmStartCache {
    BernoulliInterceptWarmStartCache {
        intercept_value: (0..n).map(|_| AtomicU64::new(f64::NAN.to_bits())).collect(),
        intercept_tag: (0..n).map(|_| AtomicU64::new(0)).collect(),
        predictors: (0..n).map(|_| Mutex::new(None)).collect(),
        anchor_law,
    }
}

/// The latent law the rigid empirical intercept anchors on, materialised once
/// per fit — one grid for a global law, one per training row for a local
/// mixture — with the roots solved on it (gam#2926, gam#2943 part 3(iii)).
///
/// The calibration `Σ_k w_k Φ(a + b·u_k) = Φ(q)` at the OBSERVED slope `b` is
/// the anchoring equation of [`crate::latent_anchor`]: with `Φ(−x) = 1 − Φ(x)` and
/// weights summing to one it reads `Σ_k w_k Φ(−(a + b·u_k)) = Φ(−q)`. So `a(q, b)`
/// is that equation's root, and one slot per row keeps it across the outer
/// search: a bitwise repeat of a row's equation is answered from the slot, and
/// any other equation is solved from the closed form (gam#2983), so every root
/// is the one its equation defines.
pub(super) struct BernoulliAnchorLaw {
    /// Immutable once built, so a search member shares it (gnomon#2359).
    grids: Arc<BernoulliAnchorGrids>,
    roots: AnchorRootCache,
}

enum BernoulliAnchorGrids {
    Global(AnchorGridOwned),
    PerRow(Vec<LocalRowLaw>),
}

/// One training row's local mixture, combined once: the grid every consumer of
/// the row reads, its log-weights for the anchor, and the mixture it was combined
/// from, which a lookup must name to be served.
struct LocalRowLaw {
    grid: EmpiricalZGrid,
    log_weights: Vec<f64>,
    mixture: Vec<(usize, f64)>,
}

impl BernoulliMarginalSlopeFamily {
    /// Row `row`'s empirical latent grid. A local mixture is served from the grid
    /// the fit's anchor law combined once, instead of being combined again on
    /// every call; any other measure, or a row whose mixture is not the one the
    /// law combined, is read from the measure itself.
    pub(super) fn training_row_grid(
        &self,
        row: usize,
    ) -> Result<Option<std::borrow::Cow<'_, EmpiricalZGrid>>, String> {
        if let LatentMeasureKind::LocalEmpirical {
            train_row_mixtures, ..
        } = &self.latent_measure
            && let Some(law) = self
                .intercept_warm_starts
                .as_ref()
                .and_then(|cache| cache.anchor_law.as_ref())
            && let Some(mixture) = train_row_mixtures.get(row)
            && let Some(grid) = law.local_row_grid(row, mixture)
        {
            return Ok(Some(std::borrow::Cow::Borrowed(grid)));
        }
        self.latent_measure.empirical_grid_for_training_row(row)
    }
}

impl BernoulliAnchorLaw {
    /// Materialise `kind` for `n` training rows; `None` for the standard-normal
    /// law, whose anchor is the closed form.
    pub(super) fn from_kind(kind: &LatentMeasureKind, n: usize) -> Result<Option<Self>, String> {
        let grids = match kind {
            LatentMeasureKind::StandardNormal => return Ok(None),
            LatentMeasureKind::GlobalEmpirical { grid } => {
                BernoulliAnchorGrids::Global(AnchorGridOwned::from_grid(grid))
            }
            LatentMeasureKind::LocalEmpirical {
                train_row_mixtures, ..
            } => BernoulliAnchorGrids::PerRow(
                (0..n)
                    .into_par_iter()
                    .map(|row| {
                        let missing = || {
                            format!(
                                "bernoulli marginal-slope local latent law produced no grid for \
                                 row {row}"
                            )
                        };
                        let grid = kind
                            .empirical_grid_for_training_row(row)?
                            .ok_or_else(missing)?
                            .into_owned();
                        let mixture = train_row_mixtures.get(row).ok_or_else(missing)?;
                        Ok(LocalRowLaw {
                            log_weights: grid.weights.iter().map(|w| w.ln()).collect(),
                            grid,
                            mixture: mixture.iter().copied().collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            ),
        };
        let shared_across_rows = matches!(grids, BernoulliAnchorGrids::Global(_));
        Ok(Some(Self {
            grids: Arc::new(grids),
            roots: AnchorRootCache::new(n, 1, shared_across_rows),
        }))
    }

    /// The same law with empty root slots, for a search that must not read
    /// another search's roots (gnomon#2359).
    fn empty_like(&self) -> Self {
        Self {
            grids: Arc::clone(&self.grids),
            roots: self.roots.empty_like(),
        }
    }

    /// Row `row`'s law with the roots solved on it. `nodes` is the grid the
    /// caller read for the row; a grid that is not this law's is refused, so no
    /// row is ever answered with another law's roots.
    pub(super) fn row_context(
        &self,
        row: usize,
        nodes: &[f64],
    ) -> Result<AnchorRowContext<'_>, String> {
        let grid = match self.grids.as_ref() {
            BernoulliAnchorGrids::Global(grid) => grid.view(),
            BernoulliAnchorGrids::PerRow(rows) => {
                let law = rows.get(row).ok_or_else(|| {
                    format!("bernoulli marginal-slope anchor law has no grid for row {row}")
                })?;
                AnchorGrid {
                    nodes: &law.grid.nodes,
                    weights: &law.grid.weights,
                    log_weights: &law.log_weights,
                }
            }
        };
        let bits = |values: &[f64]| {
            (
                values.len(),
                values.first().map(|value| value.to_bits()),
                values.last().map(|value| value.to_bits()),
            )
        };
        if bits(grid.nodes) != bits(nodes) {
            return Err(format!(
                "bernoulli marginal-slope anchor: row {row} read a grid of {} nodes that is not \
                 the law of {} nodes its root slots were built on",
                nodes.len(),
                grid.nodes.len()
            ));
        }
        Ok(AnchorRowContext {
            grid,
            roots: Some(&self.roots),
        })
    }

    /// Row `row`'s combined local grid, when this law combined it from exactly
    /// `mixture`.
    fn local_row_grid(&self, row: usize, mixture: &[(usize, f64)]) -> Option<&EmpiricalZGrid> {
        match self.grids.as_ref() {
            BernoulliAnchorGrids::PerRow(rows) => rows
                .get(row)
                .filter(|law| law.mixture.as_slice() == mixture)
                .map(|law| &law.grid),
            BernoulliAnchorGrids::Global(_) => None,
        }
    }

    /// What the root slots have answered with so far.
    pub(super) fn counts(&self) -> AnchorSolveCounts {
        self.roots.counts()
    }
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
    pub(super) slope_beta: Option<Array1<f64>>,
    pub(super) residual_beta: Option<Array1<f64>>,
    pub(super) score_warp_beta: Option<Array1<f64>>,
    pub(super) link_dev_beta: Option<Array1<f64>>,
}

pub(crate) fn build_score_warp_deviation_block_from_seed(
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

/// The probit marginal link at `η`. The marginal index is `q = Φ⁻¹(Φ(η)) = η`
/// exactly, on every finite `η`: no probability is formed and inverted, so the
/// likelihood keeps its exact slope and curvature in the marginal coefficients
/// however deep into a tail a trial point walks (gam#2978). Every anchored
/// route solves its calibration from `q` in log space on the smaller tail
/// (`latent_anchor::solve_anchor`); `μ = Φ(η)` and its derivative stack are
/// the exact probability for the routes that integrate in probability space.
pub(crate) fn bernoulli_marginal_link_map(
    base_link: &InverseLink,
    eta: f64,
) -> Result<BernoulliMarginalLinkMap, String> {
    require_probit_marginal_slope_link(base_link, "bernoulli marginal-slope")?;
    if !eta.is_finite() {
        return Err(format!(
            "bernoulli marginal-slope marginal index is non-finite: eta={eta}"
        ));
    }
    let phi_eta = normal_pdf(eta);
    let mu1 = phi_eta;
    let mu2 = -eta * phi_eta;
    let mu3 = (eta * eta - 1.0) * phi_eta;
    let mu4 = -(eta.powi(3) - 3.0 * eta) * phi_eta;
    Ok(BernoulliMarginalLinkMap {
        eta,
        mu: normal_cdf(eta),
        mu1,
        mu2,
        mu3,
        mu4,
        q: eta,
        q1: 1.0,
        q2: 0.0,
        q3: 0.0,
        q4: 0.0,
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
    // Clamped ends: the BMS anchored-cubic SAVED-MODEL replay reconstructs the
    // deviation on that convention, so its knots cannot move before it reads the
    // ramp definition (gam#2695).
    let knots = gam_terms::basis::initializewiggle_knots_from_seed(
        knot_seed.view(),
        cfg.degree,
        cfg.num_internal_knots,
    )?;
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
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design)),
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

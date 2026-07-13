//! The `SurvivalMarginalSlopeFamily` data container itself: its fields, the
//! intercept warm-start cache, per-fit hint state, and the small accessor /
//! flex-block-routing methods that read the family's own configuration
//! (which optional blocks are active, where each block's coefficients live).

use super::*;

/// The time block has one beta vector but THREE design matrices (entry, exit,
/// derivative-at-exit). The ParameterBlockSpec uses the exit design as its
/// "official" design, so block_states[0].eta = design_exit @ beta + offset_exit.
/// This eta is NOT used in the likelihood computation — row_neglog_directional
/// recomputes all 3 linear predictors from beta_time directly. The exit-design
/// eta exists only to satisfy the CustomFamily/PIRLS interface; ExactNewton
/// blocks do not use eta for working response/weights.
#[derive(Clone)]
pub(crate) struct SurvivalMarginalSlopeFamily {
    pub(crate) n: usize,
    pub(crate) event: Arc<Array1<f64>>,
    pub(crate) weights: Arc<Array1<f64>>,
    pub(crate) z: Arc<Array2<f64>>,
    pub(crate) score_covariance: MarginalSlopeCovariance,
    pub(crate) gaussian_frailty_sd: Option<f64>,
    pub(crate) derivative_guard: f64,
    /// Time block: 3 designs sharing one beta vector.
    /// Stored as DesignMatrix to support sparse local-support bases at
    /// large scale (B-spline/I-spline rows have only degree+1 nonzeros).
    pub(crate) design_entry: DesignMatrix,
    pub(crate) design_exit: DesignMatrix,
    pub(crate) design_derivative_exit: DesignMatrix,
    pub(crate) offset_entry: Arc<Array1<f64>>,
    pub(crate) offset_exit: Arc<Array1<f64>>,
    pub(crate) derivative_offset_exit: Arc<Array1<f64>>,
    /// Baseline covariate block: contributes additively to q0 and q1, but not qd1.
    pub(crate) marginal_design: DesignMatrix,
    /// The log-slope coefficient design, its physical channels in current
    /// coordinates, and their baseline + smooth offsets. This is the sole
    /// source of truth for both scalar and per-score log-slope geometry.
    pub(crate) logslope_layout: LogslopeLayout,
    pub(crate) score_warp: Option<DeviationRuntime>,
    pub(crate) link_dev: Option<DeviationRuntime>,
    /// Absorbed Stage-1 influence columns `Z̃_infl` at the training rows
    /// (`n × p₁`), residualized against the marginal location span in the
    /// rigid-pilot row metric (#461, design §3). When `Some`, the family hosts a
    /// dedicated additive absorber block whose coefficient `γ` shifts the
    /// de-nested observed index `η₁` by `+Z̃_infl[row,:]·γ` (sibling of the
    /// per-row calibration intercept — un-`c(g)`-scaled, unlike the marginal
    /// block which enters the time-quantile location through `q·c(g)`). The
    /// block carries a fixed small ridge and is dropped at predict. `None` ⇒ raw
    /// `z` with no CTN Stage-1; the free-warp `score_warp` is the fallback basis.
    pub(crate) influence_absorber: Option<Array2<f64>>,
    pub(crate) time_linear_constraints: Option<LinearInequalityConstraints>,
    pub(crate) time_wiggle_knots: Option<Array1<f64>>,
    pub(crate) time_wiggle_degree: Option<usize>,
    pub(crate) time_wiggle_ncols: usize,
    /// Per-row cache of the previous PIRLS iter's converged intercepts. Two
    /// slots per row: `[entry_q0, exit_q1]`. Across consecutive PIRLS
    /// iterations β changes only a little, so the previously-converged `a` is
    /// an excellent initial guess for the calibration root and typically lets
    /// the solver finish in ~1–2 iterations versus the rigid closed-form seed
    /// which can be many bracket-expansion steps away. Slots are initialised
    /// to `NaN` (sentinel for "not yet solved") and overwritten with the
    /// converged intercept on every successful call.
    ///
    /// Set to `None` for unit-test fixtures that build a
    /// `SurvivalMarginalSlopeFamily` directly without running the full fit
    /// pipeline; production paths go through `make_family` which initialises
    /// the cache to length-`n`. When `None`, the solver behaves exactly as it
    /// did before the warm-start machinery was added (closed-form rigid seed).
    pub(crate) intercept_warm_starts: Option<Arc<SurvivalInterceptWarmStartCache>>,
    /// Per-fit counter of outer evaluations. Increments on each distinct
    /// outer step (detected via the concatenated-beta proxy stored in
    /// `auto_subsample_last_rho`). Drives the same two-phase
    /// auto-subsample schedule used by `BernoulliMarginalSlopeFamily`:
    /// the first `SURVIVAL_MGS_AUTO_SUBSAMPLE_PHASE1_BUDGET` evaluations
    /// install a stratified Horvitz-Thompson mask (Phase 1, ≈ 1 %
    /// gradient noise); subsequent evaluations revert to full data
    /// (Phase 2). The counter resets per fit because each fit
    /// constructs a fresh family.
    pub(crate) auto_subsample_phase_counter: Arc<AtomicUsize>,
    /// Companion to `auto_subsample_phase_counter`. Stores the
    /// concatenated-beta vector seen at the most recent counter bump.
    /// Survival entry points (`*_workspace_with_options`) do not receive
    /// the outer ρ directly, so we use the joint coefficient vector as
    /// a stable per-outer-eval key. Within a single outer eval all
    /// downstream calls share the same betas, so retries don't bump the
    /// counter; across outer evals the betas change so the counter
    /// increments cleanly.
    pub(crate) auto_subsample_last_rho: Arc<Mutex<Option<Array1<f64>>>>,
}

/// Number of outer evaluations the survival auto-subsample schedule
/// spends in Phase 1 before reverting to full data. Mirrors the BMS
/// budget so the two families share an empirical noise-floor schedule.
pub(crate) const SURVIVAL_MGS_AUTO_SUBSAMPLE_PHASE1_BUDGET: usize = 12;

/// Discriminates the two intercept slots per row: the entry-time intercept
/// (solved against `q0`) and the exit-time intercept (solved against `q1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurvivalInterceptSlotKind {
    Entry = 0,
    Exit = 1,
}

/// Per-row warm-start storage for the survival calibration root solver.
///
/// Two slots per row (entry intercept against `q0`, exit intercept against
/// `q1`). Each slot stores the converged intercept `a` alongside a
/// `beta_tag: u64` — a 64-bit hash of the joint coefficient vector at the
/// time of write. Reads return `Some(a)` only when the caller's tag matches
/// the stored tag AND the stored value is finite. This makes the cache
/// transactional with respect to trust-region trials and subsampled probes:
/// a rejected trial at β_A and an accepted full-data eval at β_B key under
/// distinct tags, so writes from one cannot poison reads from the other.
///
/// The "never written" sentinel is `beta_tag == 0`. Callers compute their
/// tag with `hash_intercept_warm_start_key` and remap `0` to `1` so that the
/// sentinel can never collide with a real key. Two consecutive evaluations
/// at the same β share the same tag and reuse the cached root.
///
/// Memory ordering: the writer stores `value` with `Relaxed` and then `tag`
/// with `Release`. The reader loads `tag` with `Acquire`, reads `value`
/// with `Relaxed`, and re-checks `tag` with `Acquire`. The double-check
/// detects a torn read where another thread interleaved a tag bump between
/// the value read and the second tag load.
pub(crate) struct SurvivalInterceptWarmStartCache {
    pub(crate) entry_value: Vec<std::sync::atomic::AtomicU64>,
    pub(crate) entry_tag: Vec<std::sync::atomic::AtomicU64>,
    pub(crate) exit_value: Vec<std::sync::atomic::AtomicU64>,
    pub(crate) exit_tag: Vec<std::sync::atomic::AtomicU64>,
}

impl SurvivalInterceptWarmStartCache {
    #[inline]
    pub(crate) fn slots_for(
        &self,
        kind: SurvivalInterceptSlotKind,
    ) -> (
        &[std::sync::atomic::AtomicU64],
        &[std::sync::atomic::AtomicU64],
    ) {
        match kind {
            SurvivalInterceptSlotKind::Entry => (&self.entry_value, &self.entry_tag),
            SurvivalInterceptSlotKind::Exit => (&self.exit_value, &self.exit_tag),
        }
    }

    /// Return the cached intercept iff the slot's stored `beta_tag` matches
    /// the caller's `beta_tag` and the stored value is finite. Otherwise
    /// returns `None` (cache miss — caller falls back to closed-form seed).
    #[inline]
    pub(crate) fn load(
        &self,
        row: usize,
        kind: SurvivalInterceptSlotKind,
        beta_tag: u64,
    ) -> Option<f64> {
        let (values, tags) = self.slots_for(kind);
        let value_slot = values.get(row)?;
        let tag_slot = tags.get(row)?;
        let tag_before = tag_slot.load(std::sync::atomic::Ordering::Acquire);
        if tag_before != beta_tag {
            return None;
        }
        let bits = value_slot.load(std::sync::atomic::Ordering::Relaxed);
        let tag_after = tag_slot.load(std::sync::atomic::Ordering::Acquire);
        if tag_after != beta_tag {
            return None;
        }
        let value = f64::from_bits(bits);
        value.is_finite().then_some(value)
    }

    /// Stamp the slot with the converged intercept under `beta_tag`. Concurrent
    /// writers from different trials race; the last writer wins, which is fine
    /// because every reader gates on its own tag and only accepts a match.
    #[inline]
    pub(crate) fn store(&self, row: usize, kind: SurvivalInterceptSlotKind, a: f64, beta_tag: u64) {
        let (values, tags) = self.slots_for(kind);
        if let (Some(value_slot), Some(tag_slot)) = (values.get(row), tags.get(row)) {
            // Invalidate before writing the new value so an interleaved
            // reader cannot see the new tag paired with the old value.
            tag_slot.store(0, std::sync::atomic::Ordering::Release);
            value_slot.store(a.to_bits(), std::sync::atomic::Ordering::Relaxed);
            tag_slot.store(beta_tag, std::sync::atomic::Ordering::Release);
        }
    }
}

pub(crate) fn new_intercept_warm_start_cache(n: usize) -> Arc<SurvivalInterceptWarmStartCache> {
    Arc::new(SurvivalInterceptWarmStartCache {
        entry_value: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()))
            .collect(),
        entry_tag: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect(),
        exit_value: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()))
            .collect(),
        exit_tag: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect(),
    })
}

/// FNV-1a 64-bit hash of the joint coefficient slices `(beta_h, beta_w)`.
/// Returned tag is guaranteed non-zero (zero is remapped to one) so that
/// the cache's "never written" sentinel cannot collide with a real key.
/// At 64 bits, false collisions across distinct β are astronomically rare;
/// on a miss we just re-solve from the closed-form seed.
#[inline]
pub(crate) fn hash_intercept_warm_start_key(
    beta_h: Option<&Array1<f64>>,
    beta_w: Option<&Array1<f64>>,
) -> u64 {
    let mut hash = Fnv1a::new();
    hash.mix_opt_beta(0xa1, beta_h);
    hash.mix_opt_beta(0xa2, beta_w);
    hash.finish_nonzero()
}

#[derive(Clone, Default)]
pub(crate) struct ThetaHints {
    pub(crate) time_beta: Option<Array1<f64>>,
    pub(crate) marginal_beta: Option<Array1<f64>>,
    pub(crate) logslope_beta: Option<Array1<f64>>,
    pub(crate) score_warp_beta: Option<Array1<f64>>,
    pub(crate) link_dev_beta: Option<Array1<f64>>,
    pub(crate) influence_beta: Option<Array1<f64>>,
}

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn time_derivative_lower_bound(&self) -> f64 {
        assert!(
            self.derivative_guard.is_finite() && self.derivative_guard > 0.0,
            "survival marginal-slope derivative guard must be finite and positive: derivative_guard={}",
            self.derivative_guard
        );
        self.derivative_guard
    }

    pub(crate) fn flex_active(&self) -> bool {
        // The absorbed influence block (#461) rides the dynamic-Q primary-jet
        // path (it adds the `o_infl` primary coordinate), so it counts as "flex"
        // for dispatch purposes even when no score_warp / link_dev is present —
        // the rigid closed-form row kernel has no `o_infl` channel.
        self.score_warp.is_some() || self.link_dev.is_some() || self.influence_absorber.is_some()
    }

    pub(crate) fn effective_flex_active(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<bool, String> {
        if self.score_warp.is_some() && self.flex_score_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival score-warp block state".to_string(),
            }
            .into());
        }
        if self.link_dev.is_some() && self.flex_link_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival link-deviation block state".to_string(),
            }
            .into());
        }
        if self.influence_absorber.is_some() && self.flex_influence_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival influence-absorber block state".to_string(),
            }
            .into());
        }
        Ok(self.flex_active())
    }

    pub(crate) fn flex_score_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.score_warp.is_none() {
            return Ok(None);
        }
        block_states
            .get(3)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival score-warp block state".to_string())
    }

    pub(crate) fn flex_link_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.link_dev.is_none() {
            return Ok(None);
        }
        let idx = if self.score_warp.is_some() { 4 } else { 3 };
        block_states
            .get(idx)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival link-deviation block state".to_string())
    }

    /// Coefficient `γ` of the absorbed Stage-1 influence block (#461). The
    /// absorber is the trailing block, so its index is `3 + score_warp? +
    /// link_dev?`. `None` when no influence Jacobian was installed.
    pub(crate) fn flex_influence_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.influence_absorber.is_none() {
            return Ok(None);
        }
        let idx = 3 + usize::from(self.score_warp.is_some()) + usize::from(self.link_dev.is_some());
        block_states
            .get(idx)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival influence-absorber block state".to_string())
    }

    /// Per-row absorbed-influence index offset `o_infl[row] = Z̃_infl[row,:]·γ`.
    /// Returns `0.0` when no absorber is installed (the additive shift vanishes),
    /// so callers can fold it unconditionally into the de-nested observed `η₁`.
    pub(crate) fn influence_index_offset(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        let (Some(z_tilde), Some(gamma)) = (
            self.influence_absorber.as_ref(),
            self.flex_influence_beta(block_states)?,
        ) else {
            return Ok(0.0);
        };
        if gamma.len() != z_tilde.ncols() {
            return Err(format!(
                "survival influence-absorber β length {} != Z̃_infl columns {}",
                gamma.len(),
                z_tilde.ncols()
            ));
        }
        Ok(z_tilde.row(row).dot(gamma))
    }
}

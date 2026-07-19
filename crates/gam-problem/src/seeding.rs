#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedRiskProfile {
    Gaussian,
    /// Gaussian location-scale keeps Gaussian's lowest-REML keep-best policy,
    /// but its non-profiled log-scale predictor has the same capped-screening
    /// over-smoothing risk as multi-parameter likelihoods.
    GaussianLocationScale,
    GeneralizedLinear,
    Survival,
}

impl SeedRiskProfile {
    #[inline]
    pub const fn anchor_rho_shift(self) -> f64 {
        match self {
            Self::Gaussian | Self::GaussianLocationScale => 0.0,
            Self::GeneralizedLinear => 1.0,
            Self::Survival => 2.0,
        }
    }

    #[inline]
    pub const fn baseline_centers(self) -> &'static [f64] {
        match self {
            Self::Gaussian | Self::GaussianLocationScale => &[0.0, -3.0, 3.0, -6.0, 6.0],
            Self::GeneralizedLinear => &[0.0, 2.0, 4.0, -2.0],
            Self::Survival => &[0.0, 2.0, 4.0, 6.0],
        }
    }

    #[inline]
    pub const fn global_shifts(self) -> &'static [f64] {
        match self {
            Self::Gaussian | Self::GaussianLocationScale => &[-2.0, 2.0, -4.0, 4.0],
            Self::GeneralizedLinear => &[0.0, 2.0, 4.0, -1.0, -2.0, -4.0],
            Self::Survival => &[0.0, 2.0, 4.0, 6.0, -2.0, -4.0],
        }
    }

    #[inline]
    pub const fn exploratory_amplitude(self) -> f64 {
        match self {
            Self::Gaussian | Self::GaussianLocationScale => 2.0,
            Self::GeneralizedLinear => 2.5,
            Self::Survival => 3.0,
        }
    }

    #[inline]
    pub const fn promotes_interior_seed_extremes(self) -> bool {
        // Plain Gaussian REML's profiled-scale basin does NOT exhibit the
        // capped-screening over-smoothing bias the other profiles do, so it does
        // not need the flexible slot-0 promotion for its own sake. But a
        // weak-signal Gaussian fit on an over-rich spatial basis has the
        // OPPOSITE failure: REML descends from the heuristic anchor into the
        // flexible (low-λ) basin and over-fits (#1074 quakes: edf≈104 vs mgcv≈15,
        // held-out R²≈0.02), because the heavily-penalized basin is a separate
        // attractor never seeded/solved. Promoting the heaviest INTERIOR seed to
        // the second full-budget slot (paired with the Gaussian over-smoothing
        // probe and seed_budget≥2 in `external_reml_seed_config`) lets the
        // multi-start SEE that basin; Gaussian's lowest-cost keep-best
        // (`uses_lowest_cost_keep_best`) then adopts it only when it scores a
        // strictly lower REML, so this can never worsen a flexible fit.
        matches!(
            self,
            Self::Gaussian | Self::GaussianLocationScale | Self::GeneralizedLinear | Self::Survival
        )
    }

    #[inline]
    pub const fn uses_parsimonious_keep_best(self) -> bool {
        matches!(self, Self::GeneralizedLinear | Self::Survival)
    }

    #[inline]
    pub const fn uses_lowest_cost_keep_best(self) -> bool {
        matches!(self, Self::Gaussian | Self::GaussianLocationScale)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SeedConfig {
    pub bounds: (f64, f64),
    pub max_seeds: usize,
    /// Maximum number of seed starts to run in heuristic order.
    pub seed_budget: usize,
    /// Initial inner-iteration cap used while ranking candidate seeds.
    pub screen_max_inner_iterations: usize,
    pub risk_profile: SeedRiskProfile,
    /// Number of trailing dimensions that are auxiliary parameters rather than
    /// log-smoothing parameters.
    pub num_auxiliary_trailing: usize,
    /// Optional absolute over-smoothing probe on every smoothing dimension.
    pub over_smoothing_probe_rho: Option<f64>,
}

impl Default for SeedConfig {
    fn default() -> Self {
        Self {
            bounds: (-12.0, 12.0),
            max_seeds: 12,
            seed_budget: 2,
            screen_max_inner_iterations: 3,
            risk_profile: SeedRiskProfile::GeneralizedLinear,
            num_auxiliary_trailing: 0,
            over_smoothing_probe_rho: None,
        }
    }
}

/// A validated, finite, ordered ρ (log-λ) seed interval `[lo, hi]`.
///
/// Every seed clamp in the outer-optimizer prepass and the candidate lattice
/// derives a trial ρ and pins it into a single uniform box assembled from two
/// *independently-owned* constants — the outer ρ lower wall
/// (`options.rho_lower_bound`) and an over-smoothing ceiling (`RHO_BOUND` or an
/// effective-df crossing). When those constants drift apart the interval inverts
/// (`lo > hi`): the #2370 disease, where an edf-ceiling that used to equal
/// `-rho_lower_bound` was moved by #2356 and the emitted upper bound dropped
/// below the lower one. The historical response — *silently swapping* the pair
/// (`normalize_seed_bounds`) — does not make the fit correct; it makes the
/// optimizer solve a *different, silently substituted* box and return a model as
/// if nothing were wrong. That is strictly worse than the panic it replaced: a
/// panic is loud, a silently-wrong λ-box is not.
///
/// This type makes the inverted state unrepresentable. It is constructed only
/// through [`OrderedRhoBounds::new`], which refuses an inverted or non-finite
/// interval with the same typed [`EstimationError::InvalidInput`] the outer
/// entry (`run_outer_uncertified`) now enforces (#2379 / #2370). Every downstream
/// clamp then operates on an interval that is ordered *by construction*, so
/// `f64::clamp`'s `min <= max` precondition can never be violated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderedRhoBounds {
    lo: f64,
    hi: f64,
}

impl OrderedRhoBounds {
    /// Validate and wrap a `[lo, hi]` ρ interval. Refuses (rather than silently
    /// reorders) an inverted (`lo > hi`) or non-finite interval, naming both
    /// endpoints. `lo == hi` is a valid degenerate single-point box.
    pub fn new(lo: f64, hi: f64) -> Result<Self, crate::estimation_error::EstimationError> {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(crate::estimation_error::EstimationError::InvalidInput(format!(
                "seed ρ-box is inverted or non-finite: lower={lo}, upper={hi}; an \
                 inverted box means the ρ lower wall and the over-smoothing ceiling \
                 have drifted apart (cf. #2370) — refusing rather than silently \
                 reordering the interval (#2379)"
            )));
        }
        Ok(Self { lo, hi })
    }

    /// The (validated) lower endpoint.
    #[inline]
    pub fn lower(self) -> f64 {
        self.lo
    }

    /// The (validated) upper endpoint.
    #[inline]
    pub fn upper(self) -> f64 {
        self.hi
    }

    /// Clamp `value` into `[lo, hi]`. Infallible: the interval is ordered by
    /// construction, so `f64::clamp`'s `min <= max` precondition always holds.
    #[inline]
    pub fn clamp(self, value: f64) -> f64 {
        value.clamp(self.lo, self.hi)
    }

    /// Raise the upper endpoint to at least `floor`, preserving orderedness.
    ///
    /// The criterion-ranked prepass widens its over-smoothing bound to the full
    /// range the outer optimizer can reach (`RHO_BOUND`) so a genuinely large λ
    /// seed is not clipped to the seed band. This only ever *raises* `hi`, so the
    /// interval stays valid by construction. A non-finite `floor` is ignored to
    /// preserve the finiteness invariant (callers pass the finite `RHO_BOUND`).
    #[inline]
    pub fn with_upper_at_least(self, floor: f64) -> Self {
        if floor.is_finite() && floor > self.hi {
            Self {
                lo: self.lo,
                hi: floor,
            }
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── anchor_rho_shift ───────────────────────────────────────────────────────

    #[test]
    fn anchor_rho_shift_gaussian_is_zero() {
        assert_eq!(SeedRiskProfile::Gaussian.anchor_rho_shift(), 0.0);
        assert_eq!(
            SeedRiskProfile::GaussianLocationScale.anchor_rho_shift(),
            0.0
        );
    }

    #[test]
    fn anchor_rho_shift_generalized_linear_is_one() {
        assert_eq!(SeedRiskProfile::GeneralizedLinear.anchor_rho_shift(), 1.0);
    }

    #[test]
    fn anchor_rho_shift_survival_is_two() {
        assert_eq!(SeedRiskProfile::Survival.anchor_rho_shift(), 2.0);
    }

    // ── promotes_interior_seed_extremes ───────────────────────────────────────

    #[test]
    fn promotes_interior_extremes_for_all_profiles() {
        // #1074: plain Gaussian was originally excluded (its profiled-scale REML
        // basin has no capped-screening over-smoothing bias), but a weak-signal
        // Gaussian fit on an over-rich basis has the OPPOSITE failure — it
        // descends into the flexible (low-λ) basin and over-fits. Promoting the
        // heaviest interior seed to the second full-budget slot (paired with the
        // over-smoothing probe + `seed_budget ≥ 2`) lets the multi-start SEE the
        // heavily-penalized basin; Gaussian's lowest-cost keep-best then adopts
        // it only when it scores a strictly lower REML, so this can never worsen
        // a flexible fit. Every risk profile now promotes the interior extremes.
        assert!(SeedRiskProfile::Gaussian.promotes_interior_seed_extremes());
        assert!(SeedRiskProfile::GaussianLocationScale.promotes_interior_seed_extremes());
        assert!(SeedRiskProfile::GeneralizedLinear.promotes_interior_seed_extremes());
        assert!(SeedRiskProfile::Survival.promotes_interior_seed_extremes());
    }

    // ── keep-best policy flags ────────────────────────────────────────────────

    #[test]
    fn parsimonious_keep_best_only_for_glm_and_survival() {
        assert!(!SeedRiskProfile::Gaussian.uses_parsimonious_keep_best());
        assert!(!SeedRiskProfile::GaussianLocationScale.uses_parsimonious_keep_best());
        assert!(SeedRiskProfile::GeneralizedLinear.uses_parsimonious_keep_best());
        assert!(SeedRiskProfile::Survival.uses_parsimonious_keep_best());
    }

    #[test]
    fn lowest_cost_keep_best_only_for_gaussian_variants() {
        assert!(SeedRiskProfile::Gaussian.uses_lowest_cost_keep_best());
        assert!(SeedRiskProfile::GaussianLocationScale.uses_lowest_cost_keep_best());
        assert!(!SeedRiskProfile::GeneralizedLinear.uses_lowest_cost_keep_best());
        assert!(!SeedRiskProfile::Survival.uses_lowest_cost_keep_best());
    }

    // ── OrderedRhoBounds (#2379) ──────────────────────────────────────────────
    // The validated-interval type that REPLACES the silent swap on every seed
    // clamp: an inverted box must be a typed refusal, never a reordered interval.

    #[test]
    fn ordered_rho_bounds_accepts_ordered_interval() {
        let b = OrderedRhoBounds::new(-12.0, 12.0).expect("ordered interval is valid");
        assert_eq!(b.lower(), -12.0);
        assert_eq!(b.upper(), 12.0);
    }

    #[test]
    fn ordered_rho_bounds_accepts_degenerate_point_interval() {
        // lo == hi is a valid single-point box (matches opt::Bounds, which uses
        // `lower > upper` as the inversion test).
        let b = OrderedRhoBounds::new(2.0, 2.0).expect("point interval is valid");
        assert_eq!(b.clamp(5.0), 2.0);
        assert_eq!(b.clamp(-5.0), 2.0);
    }

    #[test]
    fn ordered_rho_bounds_refuses_inverted_interval_with_typed_error() {
        // This is the #2379 contract: an inverted seed-bound pair reaching the
        // seed path is a typed refusal, NOT a silently reordered box. The exact
        // scenario from #2370 — lower = -10 (the ρ lower wall) above an
        // independently-derived edf ceiling of -11.855.
        let err = OrderedRhoBounds::new(-10.0, -11.855)
            .expect_err("an inverted box must be refused, not swapped");
        match err {
            crate::estimation_error::EstimationError::InvalidInput(msg) => {
                // Both endpoints are named, so a drift is diagnosable from the error.
                assert!(msg.contains("-10"), "error names the lower bound: {msg}");
                assert!(msg.contains("-11.855"), "error names the upper bound: {msg}");
                assert!(msg.contains("invert"), "error explains the inversion: {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn ordered_rho_bounds_refuses_non_finite_interval() {
        assert!(OrderedRhoBounds::new(f64::NAN, 12.0).is_err());
        assert!(OrderedRhoBounds::new(-12.0, f64::INFINITY).is_err());
        assert!(OrderedRhoBounds::new(f64::NEG_INFINITY, 12.0).is_err());
    }

    #[test]
    fn ordered_rho_bounds_clamp_respects_both_ends() {
        let b = OrderedRhoBounds::new(-3.0, 5.0).unwrap();
        assert_eq!(b.clamp(1.0), 1.0);
        assert_eq!(b.clamp(-10.0), -3.0);
        assert_eq!(b.clamp(100.0), 5.0);
    }

    #[test]
    fn ordered_rho_bounds_with_upper_only_raises_and_stays_ordered() {
        let b = OrderedRhoBounds::new(-12.0, 8.0).unwrap();
        // Widening to a larger ceiling raises the upper endpoint.
        let widened = b.with_upper_at_least(30.0);
        assert_eq!(widened.lower(), -12.0);
        assert_eq!(widened.upper(), 30.0);
        // Widening to a floor already below `hi` is a no-op (never lowers `hi`).
        let unchanged = b.with_upper_at_least(2.0);
        assert_eq!(unchanged.upper(), 8.0);
        // A non-finite floor is ignored so the finiteness invariant is preserved.
        let finite = b.with_upper_at_least(f64::INFINITY);
        assert!(finite.upper().is_finite());
        assert_eq!(finite.upper(), 8.0);
    }
}

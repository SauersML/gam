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
            Self::Gaussian
                | Self::GaussianLocationScale
                | Self::GeneralizedLinear
                | Self::Survival
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

#[inline]
pub fn normalize_seed_bounds(bounds: (f64, f64)) -> (f64, f64) {
    if bounds.0 <= bounds.1 {
        bounds
    } else {
        (bounds.1, bounds.0)
    }
}

#[inline]
pub fn clamp_seed_rho_to_bounds(value: f64, bounds: (f64, f64)) -> f64 {
    let (lo, hi) = normalize_seed_bounds(bounds);
    value.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── anchor_rho_shift ───────────────────────────────────────────────────────

    #[test]
    fn anchor_rho_shift_gaussian_is_zero() {
        assert_eq!(SeedRiskProfile::Gaussian.anchor_rho_shift(), 0.0);
        assert_eq!(SeedRiskProfile::GaussianLocationScale.anchor_rho_shift(), 0.0);
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

    // ── normalize_seed_bounds ─────────────────────────────────────────────────

    #[test]
    fn normalize_already_ordered_bounds_unchanged() {
        assert_eq!(normalize_seed_bounds((-3.0, 5.0)), (-3.0, 5.0));
    }

    #[test]
    fn normalize_reversed_bounds_swaps() {
        assert_eq!(normalize_seed_bounds((5.0, -3.0)), (-3.0, 5.0));
    }

    #[test]
    fn normalize_equal_bounds_unchanged() {
        assert_eq!(normalize_seed_bounds((2.0, 2.0)), (2.0, 2.0));
    }

    // ── clamp_seed_rho_to_bounds ──────────────────────────────────────────────

    #[test]
    fn clamp_within_bounds_returns_value() {
        assert_eq!(clamp_seed_rho_to_bounds(1.0, (-3.0, 5.0)), 1.0);
    }

    #[test]
    fn clamp_below_lo_returns_lo() {
        assert_eq!(clamp_seed_rho_to_bounds(-10.0, (-3.0, 5.0)), -3.0);
    }

    #[test]
    fn clamp_above_hi_returns_hi() {
        assert_eq!(clamp_seed_rho_to_bounds(100.0, (-3.0, 5.0)), 5.0);
    }

    #[test]
    fn clamp_normalizes_reversed_bounds_before_clamping() {
        // bounds (5, -3) normalizes to (-3, 5); 100 clamps to 5
        assert_eq!(clamp_seed_rho_to_bounds(100.0, (5.0, -3.0)), 5.0);
    }
}

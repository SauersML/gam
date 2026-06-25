#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedRiskProfile {
    Gaussian,
    GeneralizedLinear,
    Survival,
}

impl SeedRiskProfile {
    #[inline]
    pub const fn anchor_rho_shift(self) -> f64 {
        match self {
            Self::Gaussian => 0.0,
            Self::GeneralizedLinear => 1.0,
            Self::Survival => 2.0,
        }
    }

    #[inline]
    pub const fn baseline_centers(self) -> &'static [f64] {
        match self {
            Self::Gaussian => &[0.0, -3.0, 3.0, -6.0, 6.0],
            Self::GeneralizedLinear => &[0.0, 2.0, 4.0, -2.0],
            Self::Survival => &[0.0, 2.0, 4.0, 6.0],
        }
    }

    #[inline]
    pub const fn global_shifts(self) -> &'static [f64] {
        match self {
            Self::Gaussian => &[-2.0, 2.0, -4.0, 4.0],
            Self::GeneralizedLinear => &[0.0, 2.0, 4.0, -1.0, -2.0, -4.0],
            Self::Survival => &[0.0, 2.0, 4.0, 6.0, -2.0, -4.0],
        }
    }

    #[inline]
    pub const fn exploratory_amplitude(self) -> f64 {
        match self {
            Self::Gaussian => 2.0,
            Self::GeneralizedLinear => 2.5,
            Self::Survival => 3.0,
        }
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

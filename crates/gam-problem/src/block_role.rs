use serde::{Deserialize, Serialize};

/// Role of a coefficient block within a multi-parameter model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockRole {
    /// Single-parameter GAM (standard GLM/GAM mean model).
    Mean,
    /// Location parameter in GAMLSS / survival location-scale.
    Location,
    /// Scale (log-sigma) parameter in GAMLSS / survival location-scale.
    Scale,
    /// Time/baseline hazard block in survival models.
    Time,
    /// Threshold block in survival models.
    Threshold,
    /// Link-wiggle correction block.
    LinkWiggle,
}

impl BlockRole {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Location => "location",
            Self::Scale => "scale",
            Self::Time => "time",
            Self::Threshold => "threshold",
            Self::LinkWiggle => "link-wiggle",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mean() {
        assert_eq!(BlockRole::Mean.name(), "mean");
    }

    #[test]
    fn name_location() {
        assert_eq!(BlockRole::Location.name(), "location");
    }

    #[test]
    fn name_scale() {
        assert_eq!(BlockRole::Scale.name(), "scale");
    }

    #[test]
    fn name_time() {
        assert_eq!(BlockRole::Time.name(), "time");
    }

    #[test]
    fn name_threshold() {
        assert_eq!(BlockRole::Threshold.name(), "threshold");
    }

    #[test]
    fn name_link_wiggle() {
        assert_eq!(BlockRole::LinkWiggle.name(), "link-wiggle");
    }

    #[test]
    fn eq_reflexive() {
        assert_eq!(BlockRole::Mean, BlockRole::Mean);
        assert_ne!(BlockRole::Mean, BlockRole::Scale);
    }
}

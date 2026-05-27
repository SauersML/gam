//! Declarative `return Err(...)` shorthands for the repetitive `bail_*`
//! patterns across the crate. Each macro expands to the equivalent
//! `return Err(Variant(...))` so it can be used inside any function
//! whose error type matches.
//!
//! Naming: `bail_<variant>_<type-shortcode>!`. Type shortcodes:
//!   estim   → EstimationError
//!   basis   → BasisError
//!   gamlss  → GamlssError
//!   tnorm   → TransformationNormalError
//!   surv    → SurvivalError
//!   sls     → SurvivalLocationScaleError
//!   custom  → CustomFamilyError

#[macro_export]
macro_rules! bail_invalid_estim {
    ($lit:literal) => {
        return Err($crate::solver::estimate::EstimationError::InvalidInput($lit.to_string()))
    };
    ($($arg:tt)*) => {
        return Err($crate::solver::estimate::EstimationError::InvalidInput(format!($($arg)*)))
    };
}

#[macro_export]
macro_rules! bail_invalid_basis {
    ($lit:literal) => {
        return Err($crate::terms::basis::BasisError::InvalidInput($lit.to_string()))
    };
    ($($arg:tt)*) => {
        return Err($crate::terms::basis::BasisError::InvalidInput(format!($($arg)*)))
    };
}

#[macro_export]
macro_rules! bail_dim_basis {
    ($lit:literal) => {
        return Err($crate::terms::basis::BasisError::DimensionMismatch($lit.to_string()))
    };
    ($($arg:tt)*) => {
        return Err($crate::terms::basis::BasisError::DimensionMismatch(format!($($arg)*)))
    };
}

#[macro_export]
macro_rules! bail_invalid_gamlss {
    ($lit:literal) => {
        return Err($crate::families::gamlss::GamlssError::InvalidInput { reason: $lit.to_string() })
    };
    ($($arg:tt)*) => {
        return Err($crate::families::gamlss::GamlssError::InvalidInput { reason: format!($($arg)*) })
    };
}

#[macro_export]
macro_rules! bail_dim_gamlss {
    ($lit:literal) => {
        return Err($crate::families::gamlss::GamlssError::DimensionMismatch { reason: $lit.to_string() })
    };
    ($($arg:tt)*) => {
        return Err($crate::families::gamlss::GamlssError::DimensionMismatch { reason: format!($($arg)*) })
    };
}

#[macro_export]
macro_rules! bail_invalid_tnorm {
    ($lit:literal) => {
        return Err($crate::families::transformation_normal::TransformationNormalError::InvalidInput { reason: $lit.to_string() })
    };
    ($($arg:tt)*) => {
        return Err($crate::families::transformation_normal::TransformationNormalError::InvalidInput { reason: format!($($arg)*) })
    };
}

#[macro_export]
macro_rules! bail_invalid_surv {
    ($lit:literal) => {
        return Err($crate::families::survival::SurvivalError::InvalidInput { reason: $lit.to_string() })
    };
    ($($arg:tt)*) => {
        return Err($crate::families::survival::SurvivalError::InvalidInput { reason: format!($($arg)*) })
    };
}

#[macro_export]
macro_rules! bail_dim_sls {
    ($lit:literal) => {
        return Err($crate::families::survival_location_scale::SurvivalLocationScaleError::DimensionMismatch { reason: $lit.to_string() })
    };
    ($($arg:tt)*) => {
        return Err($crate::families::survival_location_scale::SurvivalLocationScaleError::DimensionMismatch { reason: format!($($arg)*) })
    };
}

#[macro_export]
macro_rules! bail_dim_custom {
    ($lit:literal) => {
        return Err($crate::families::custom_family::CustomFamilyError::DimensionMismatch { reason: $lit.to_string() })
    };
    ($($arg:tt)*) => {
        return Err($crate::families::custom_family::CustomFamilyError::DimensionMismatch { reason: format!($($arg)*) })
    };
}

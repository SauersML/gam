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
macro_rules! impl_reason_error_boilerplate {
    ($type:ident { $($variant:ident),+ $(,)? }) => {
        impl ::std::fmt::Display for $type {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                match self {
                    $(Self::$variant { reason })|+ => f.write_str(reason),
                }
            }
        }

        impl ::std::error::Error for $type {}

        impl From<$type> for String {
            fn from(err: $type) -> String {
                err.to_string()
            }
        }
    };
}

#[macro_export]
macro_rules! bail_invalid_estim {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::model_types::EstimationError::InvalidInput(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::model_types::EstimationError::InvalidInput($msg))
    };
}

#[macro_export]
macro_rules! bail_invalid_basis {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::terms::basis::BasisError::InvalidInput(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::terms::basis::BasisError::InvalidInput($msg))
    };
}

#[macro_export]
macro_rules! bail_dim_basis {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::terms::basis::BasisError::DimensionMismatch(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::terms::basis::BasisError::DimensionMismatch($msg))
    };
}

#[macro_export]
macro_rules! bail_invalid_gamlss {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::families::gamlss::GamlssError::InvalidInput { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::families::gamlss::GamlssError::InvalidInput { reason: $msg })
    };
}

#[macro_export]
macro_rules! bail_dim_gamlss {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::families::gamlss::GamlssError::DimensionMismatch { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::families::gamlss::GamlssError::DimensionMismatch { reason: $msg })
    };
}

#[macro_export]
macro_rules! bail_invalid_tnorm {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::families::transformation_normal::TransformationNormalError::InvalidInput { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::families::transformation_normal::TransformationNormalError::InvalidInput { reason: $msg })
    };
}

#[macro_export]
macro_rules! bail_invalid_surv {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::families::survival::SurvivalError::InvalidInput { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::families::survival::SurvivalError::InvalidInput { reason: $msg })
    };
}

#[macro_export]
macro_rules! bail_dim_sls {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::families::survival::location_scale::SurvivalLocationScaleError::DimensionMismatch { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::families::survival::location_scale::SurvivalLocationScaleError::DimensionMismatch { reason: $msg })
    };
}

#[macro_export]
macro_rules! bail_dim_custom {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::families::custom_family::CustomFamilyError::DimensionMismatch { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::families::custom_family::CustomFamilyError::DimensionMismatch { reason: $msg })
    };
}

//! Declarative `return Err(...)` shorthands for the repetitive `bail_*`
//! patterns whose error types live in **this** crate (`gam-models`), plus the
//! shared `impl_reason_error_boilerplate!` derive used by every
//! `{ reason: String }`-shaped error enum defined here.
//!
//! The `bail_*` shorthands for error types that were relocated to the neutral
//! `gam-problem` crate (`bail_invalid_estim!`, `bail_dim_custom!`) are *not*
//! redefined here — they are re-exported from `gam-problem` at the crate root so
//! `crate::bail_invalid_estim!` / `crate::bail_dim_custom!` continue to resolve.
//!
//! Naming: `bail_<variant>_<type-shortcode>!`. Type shortcodes:
//!   tnorm   → TransformationNormalError
//!   surv    → SurvivalError
//!   sls     → SurvivalLocationScaleError

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
macro_rules! bail_invalid_tnorm {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::transformation_normal::TransformationNormalError::InvalidInput { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::transformation_normal::TransformationNormalError::InvalidInput { reason: $msg })
    };
}

#[macro_export]
macro_rules! bail_invalid_surv {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::survival::SurvivalError::InvalidInput { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::survival::SurvivalError::InvalidInput { reason: $msg })
    };
}

#[macro_export]
macro_rules! bail_dim_sls {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::survival::location_scale::SurvivalLocationScaleError::DimensionMismatch { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::survival::location_scale::SurvivalLocationScaleError::DimensionMismatch { reason: $msg })
    };
}

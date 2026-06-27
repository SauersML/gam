//! Declarative `return Err(...)` shorthands for the repetitive `bail_*`
//! patterns whose error types live in **this** crate (`gam-models`).
//!
//! The shared `impl_reason_error_boilerplate!` derive used by every
//! `{ reason: String }`-shaped error enum was carved down into the base kernel
//! crate (`gam-model-kernels`) under #1521; it is brought back into crate-wide
//! textual scope via `#[macro_use] extern crate gam_model_kernels;` at the crate
//! root so the unqualified call sites here resolve unchanged.
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

//! The shared `impl_reason_error_boilerplate!` derive used by every
//! `{ reason: String }`-shaped error enum across the `gam-model-kernels` and
//! `gam-models` crates. The macro is pure, dependency-free boilerplate
//! (`Display` / `Error` / `From<_> for String`), so it lives in this base
//! kernel crate and is `#[macro_export]`ed back up to `gam-models` (which
//! brings it into crate-wide textual scope via `#[macro_use] extern crate
//! gam_model_kernels;`).

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

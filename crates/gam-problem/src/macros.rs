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
        return Err($crate::EstimationError::InvalidInput(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::EstimationError::InvalidInput($msg))
    };
}

#[macro_export]
macro_rules! bail_invalid_basis {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::BasisError::InvalidInput(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::BasisError::InvalidInput($msg))
    };
}

#[macro_export]
macro_rules! bail_dim_basis {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::BasisError::DimensionMismatch(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::BasisError::DimensionMismatch($msg))
    };
}

#[macro_export]
macro_rules! bail_dim_custom {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::CustomFamilyError::DimensionMismatch { reason: format!($fmt $(, $($arg)*)?) })
    };
    ($msg:expr $(,)?) => {
        return Err($crate::CustomFamilyError::DimensionMismatch { reason: $msg })
    };
}

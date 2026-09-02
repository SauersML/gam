//! JSON encoding for quantities whose mathematical domain includes `+∞`.
//!
//! ## The gap this closes
//!
//! Several solver contracts are *extended* reals: a one-sided box wall, a
//! saturated rail, an unbounded upper limit. In memory `f64` represents them
//! exactly — `f64::INFINITY` is the value, not a sentinel. JSON has no literal
//! for it, and `serde_json` resolves that by writing the JSON literal `null`
//! *silently*. The value round-trips into a `Vec<f64>` field as
//!
//! ```text
//! invalid type: null, expected f64
//! ```
//!
//! at load time, arbitrarily far from the fit that produced it (#2601: every
//! shape-constrained fit that retains a constraint face writes one `+∞` upper
//! limit per half-line coordinate, and none of those models could be reloaded).
//!
//! ## The encoding
//!
//! `null` **is** the encoding of `+∞` here, chosen because it is what a
//! bounded-above/unbounded-above distinction means in JSON ("there is no upper
//! limit") and because it is byte-identical to what `serde_json` has already
//! been writing — so a model saved before this module existed reads back with
//! the meaning it always had, rather than needing a migration.
//!
//! Values that are **not** in the domain (`NaN`, `−∞` for an upper limit) are
//! deliberately NOT laundered: they serialize through the ordinary `f64` path,
//! where `null` would be written and the structural guard in
//! [`crate::serde_finite`] refuses the payload with the offending field's path.
//! Encoding them as `null` too would make the codec lossy in the one direction
//! that matters — it would turn a bug into a legitimate-looking `+∞`.

use serde::de::{Deserializer, Visitor};
use serde::ser::{SerializeSeq, Serializer};
use std::fmt;

/// `Vec<f64>` whose `+∞` entries are written as JSON `null` and read back as
/// `+∞`.
///
/// Use as `#[serde(with = "gam_problem::serde_extended_real::vec_f64")]`.
pub mod vec_f64 {
    use super::*;

    pub fn serialize<S>(values: &[f64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(values.len()))?;
        for value in values {
            if *value == f64::INFINITY {
                seq.serialize_element(&Option::<f64>::None)?;
            } else {
                // Finite values take the ordinary numeric encoding; anything
                // else (NaN, −∞) is passed through unmodified so the write-side
                // finiteness guard can name it instead of this codec hiding it.
                seq.serialize_element(value)?;
            }
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ExtendedRealVecVisitor;

        impl<'de> Visitor<'de> for ExtendedRealVecVisitor {
            type Value = Vec<f64>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a sequence of numbers, with null for +infinity")
            }

        }

        deserializer.deserialize_seq(ExtendedRealVecVisitor)
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Limits {
        #[serde(default, with = "super::vec_f64")]
        upper: Vec<f64>,
    }

    #[test]
    fn positive_infinity_round_trips_through_null() {
        let value = Limits {
            upper: vec![f64::INFINITY, 2.5, f64::INFINITY],
        };
        let json = serde_json::to_string(&value).expect("serialize");
        assert_eq!(json, r#"{"upper":[null,2.5,null]}"#);
        let back: Limits = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, value);
    }

    #[test]
    fn a_model_written_before_this_codec_reads_back_with_its_original_meaning() {
        // Exactly the bytes `serde_json` produced for `vec![f64::INFINITY; 3]`
        // before this module existed — the #2601 payload.
        let back: Limits = serde_json::from_str(r#"{"upper":[null,null,null]}"#).expect("legacy");
        assert_eq!(back.upper, vec![f64::INFINITY; 3]);
    }

    #[test]
    fn empty_and_absent_are_both_the_legacy_all_half_lines_encoding() {
        let back: Limits = serde_json::from_str(r#"{"upper":[]}"#).expect("empty");
        assert!(back.upper.is_empty());
        let back: Limits = serde_json::from_str(r#"{}"#).expect("absent");
        assert!(back.upper.is_empty());
    }

    #[test]
    fn finite_values_keep_the_ordinary_numeric_encoding() {
        let value = Limits {
            upper: vec![0.5, -3.0, 1e300],
        };
        let json = serde_json::to_string(&value).expect("serialize");
        assert_eq!(json, r#"{"upper":[0.5,-3.0,1e+300]}"#);
        let back: Limits = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, value);
    }

    #[test]
    fn nan_is_not_laundered_into_an_unbounded_limit() {
        // NaN is outside the domain; the codec must NOT encode it the way `+∞`
        // is encoded, or a defect would read back as a legitimate half-line.
        // It falls through to the ordinary f64 path, where the write-side
        // finiteness guard (`serde_finite`) names the field.
        let value = Limits {
            upper: vec![f64::NAN],
        };
        assert!(
            gam_problem_serde_finite_rejects(&value),
            "the structural guard must refuse a NaN limit"
        );
    }

    fn gam_problem_serde_finite_rejects<T: serde::Serialize>(value: &T) -> bool {
        crate::serde_finite::ensure_serialized_floats_are_finite(value).is_err()
    }

    #[test]
    fn an_infinite_limit_is_not_flagged_by_the_finiteness_guard() {
        // The codec turns `+∞` into `null` BEFORE the guard sees a float, so a
        // legitimate half-line is not mistaken for a defect.
        let value = Limits {
            upper: vec![f64::INFINITY, 1.0],
        };
        assert!(crate::serde_finite::ensure_serialized_floats_are_finite(&value).is_ok());
    }
}

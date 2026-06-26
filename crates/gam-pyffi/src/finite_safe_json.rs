//! Non-finite-safe JSON transport for the f64-bearing prediction payloads.
//!
//! JSON has no representation for the non-finite IEEE-754 values, and
//! `serde_json` silently serializes `NaN` / `±∞` as the bare literal `null`.
//! The survival prediction payload legitimately carries non-finite values: a
//! saturated Royston-Parmar tail has cumulative hazard `Λ(t) = exp(η) = +∞`
//! (with `S(t) = 0`), and a genuine modelling `NaN` should surface *as* `NaN`
//! for debugging rather than disappear behind an opaque parse failure.
//!
//! The default `null` encoding round-trips asymmetrically — it serializes
//! happily, but the typed `f64` deserializer then rejects `null` with
//! `invalid type: null, expected f64 at line 1 column …` (the #1564 "null in
//! serialized survival prediction payload" failure). The asymmetry, not the
//! non-finite value itself, is the bug: a value the producer is allowed to emit
//! must be a value the consumer can parse.
//!
//! This module makes the boundary *total*. Every f64 is encoded losslessly:
//! finite values stay JSON numbers (via the crate's bit-exact `float_roundtrip`
//! parser), while `+∞ / -∞ / NaN` become the JSON strings `"Infinity" /
//! "-Infinity" / "NaN"`. The matching deserializer accepts a number OR one of
//! those tokens (case-insensitively, also tolerating `inf` / `-inf`), so the
//! round-trip is total over all of `f64`.
//!
//! The producer struct (`SurvivalPredictionPayload`, serialize-only) and the
//! consumer struct (`SurvivalPredictionJsonPayload`, deserialize-only) sit on
//! opposite ends of one in-process boundary, so the adapters below are
//! intentionally one-directional: the producer's required fields use the
//! `serialize`-only `vec` / `matrix` / `map` adapters, while the consumer's
//! tolerant `Option` fields use the matching `opt_*` `deserialize` adapters.
//! The encoded form is identical either way, so the boundary stays symmetric
//! even though no single struct round-trips through both directions.
//!
//! The payload is an in-process, predict-time transport (engine → Python, never
//! persisted in the saved model), so changing its non-finite encoding has no
//! cross-version compatibility surface.

use serde::de::{Deserializer, Error as DeError, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// A single `f64` that survives a JSON round-trip even when it is non-finite.
#[derive(Clone, Copy)]
struct SafeF64(f64);

impl Serialize for SafeF64 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let x = self.0;
        if x.is_finite() {
            serializer.serialize_f64(x)
        } else if x.is_nan() {
            serializer.serialize_str("NaN")
        } else if x > 0.0 {
            serializer.serialize_str("Infinity")
        } else {
            serializer.serialize_str("-Infinity")
        }
    }
}

impl<'de> Deserialize<'de> for SafeF64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SafeF64Visitor;
        impl Visitor<'_> for SafeF64Visitor {
            type Value = SafeF64;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(
                    "a finite number or one of the tokens \"Infinity\", \"-Infinity\", \"NaN\"",
                )
            }

            fn visit_f64<E>(self, v: f64) -> Result<SafeF64, E> {
                Ok(SafeF64(v))
            }
            fn visit_f32<E>(self, v: f32) -> Result<SafeF64, E> {
                Ok(SafeF64(f64::from(v)))
            }
            fn visit_i64<E>(self, v: i64) -> Result<SafeF64, E> {
                Ok(SafeF64(v as f64))
            }
            fn visit_u64<E>(self, v: u64) -> Result<SafeF64, E> {
                Ok(SafeF64(v as f64))
            }
            fn visit_i128<E>(self, v: i128) -> Result<SafeF64, E> {
                Ok(SafeF64(v as f64))
            }
            fn visit_u128<E>(self, v: u128) -> Result<SafeF64, E> {
                Ok(SafeF64(v as f64))
            }

            fn visit_str<E: DeError>(self, v: &str) -> Result<SafeF64, E> {
                let token = v.trim();
                if token.eq_ignore_ascii_case("infinity")
                    || token.eq_ignore_ascii_case("inf")
                    || token.eq_ignore_ascii_case("+infinity")
                    || token.eq_ignore_ascii_case("+inf")
                {
                    Ok(SafeF64(f64::INFINITY))
                } else if token.eq_ignore_ascii_case("-infinity")
                    || token.eq_ignore_ascii_case("-inf")
                {
                    Ok(SafeF64(f64::NEG_INFINITY))
                } else if token.eq_ignore_ascii_case("nan") {
                    Ok(SafeF64(f64::NAN))
                } else {
                    // A numeric value that arrived as a JSON string still
                    // parses (defensive — the encoder never emits this form).
                    token
                        .parse::<f64>()
                        .map(SafeF64)
                        .map_err(|_| E::custom(format!("invalid float token: {token:?}")))
                }
            }
        }
        deserializer.deserialize_any(SafeF64Visitor)
    }
}

/// Borrowed `&[f64]` that serializes element-wise through [`SafeF64`] without
/// an intermediate allocation.
struct SafeSlice<'a>(&'a [f64]);

impl Serialize for SafeSlice<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for &value in self.0 {
            seq.serialize_element(&SafeF64(value))?;
        }
        seq.end()
    }
}

fn unwrap_vec(raw: Vec<SafeF64>) -> Vec<f64> {
    raw.into_iter().map(|x| x.0).collect()
}

fn unwrap_matrix(raw: Vec<Vec<SafeF64>>) -> Vec<Vec<f64>> {
    raw.into_iter().map(unwrap_vec).collect()
}

/// Serialize a required `Vec<f64>` field — `#[serde(with = "…::vec")]`.
pub mod vec {
    use super::{SafeSlice, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        SafeSlice(value).serialize(serializer)
    }
}

/// Serialize a required `Vec<Vec<f64>>` field — `#[serde(with = "…::matrix")]`.
pub mod matrix {
    use super::{SafeSlice, SerializeSeq, Serializer};

    pub fn serialize<S: Serializer>(value: &[Vec<f64>], serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(value.len()))?;
        for row in value {
            seq.serialize_element(&SafeSlice(row))?;
        }
        seq.end()
    }
}

/// Serialize a required `BTreeMap<String, Vec<f64>>` — `#[serde(with = "…::map")]`.
pub mod map {
    use super::{BTreeMap, SafeSlice, SerializeMap, Serializer};

    pub fn serialize<S: Serializer>(
        value: &BTreeMap<String, Vec<f64>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(value.len()))?;
        for (key, values) in value {
            map.serialize_entry(key, &SafeSlice(values))?;
        }
        map.end()
    }
}

/// Serialize/deserialize an `Option<Vec<f64>>` — `#[serde(with = "…::opt_vec")]`.
pub mod opt_vec {
    use super::{Deserialize, Deserializer, SafeF64, SafeSlice, Serializer, unwrap_vec};

    pub fn serialize<S: Serializer>(
        value: &Option<Vec<f64>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(values) => serializer.serialize_some(&SafeSlice(values)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<f64>>, D::Error> {
        Ok(Option::<Vec<SafeF64>>::deserialize(deserializer)?.map(unwrap_vec))
    }
}

/// Serialize/deserialize an `Option<Vec<Vec<f64>>>` — `#[serde(with = "…::opt_matrix")]`.
pub mod opt_matrix {
    use super::{Deserialize, Deserializer, SafeF64, Serialize, Serializer, matrix, unwrap_matrix};

    pub fn serialize<S: Serializer>(
        value: &Option<Vec<Vec<f64>>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(rows) => serializer.serialize_some(&MatrixRef(rows)),
            None => serializer.serialize_none(),
        }
    }

    /// Adapter so `serialize_some` reuses the borrow-only [`matrix`] encoder.
    struct MatrixRef<'a>(&'a [Vec<f64>]);

    impl Serialize for MatrixRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            matrix::serialize(self.0, serializer)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<Vec<f64>>>, D::Error> {
        Ok(Option::<Vec<Vec<SafeF64>>>::deserialize(deserializer)?.map(unwrap_matrix))
    }
}

/// Deserialize an `Option<BTreeMap<String, Vec<f64>>>` — `#[serde(with = "…::opt_map")]`.
pub mod opt_map {
    use super::{BTreeMap, Deserialize, Deserializer, SafeF64, unwrap_vec};

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<BTreeMap<String, Vec<f64>>>, D::Error> {
        let raw = Option::<BTreeMap<String, Vec<SafeF64>>>::deserialize(deserializer)?;
        Ok(raw.map(|columns| {
            columns
                .into_iter()
                .map(|(key, values)| (key, unwrap_vec(values)))
                .collect()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the real `SurvivalPredictionPayload` (serialize-only producer):
    /// required fields plus the optional uncertainty fields.
    #[derive(Serialize)]
    struct ProducerProbe {
        #[serde(with = "vec")]
        times: Vec<f64>,
        #[serde(with = "matrix")]
        cumulative_hazard: Vec<Vec<f64>>,
        #[serde(with = "vec")]
        linear_predictor: Vec<f64>,
        #[serde(with = "map")]
        columns: BTreeMap<String, Vec<f64>>,
        #[serde(skip_serializing_if = "Option::is_none", with = "opt_matrix")]
        survival_se: Option<Vec<Vec<f64>>>,
        #[serde(skip_serializing_if = "Option::is_none", with = "opt_vec")]
        eta_se: Option<Vec<f64>>,
    }

    /// Mirrors the real `SurvivalPredictionJsonPayload` (deserialize-only,
    /// tolerant `Option` consumer).
    #[derive(Deserialize)]
    struct ConsumerProbe {
        #[serde(default, with = "opt_vec")]
        times: Option<Vec<f64>>,
        #[serde(default, with = "opt_matrix")]
        cumulative_hazard: Option<Vec<Vec<f64>>>,
        #[serde(default, with = "opt_vec")]
        linear_predictor: Option<Vec<f64>>,
        #[serde(default, with = "opt_map")]
        columns: Option<BTreeMap<String, Vec<f64>>>,
        #[serde(default, with = "opt_matrix")]
        survival_se: Option<Vec<Vec<f64>>>,
        #[serde(default, with = "opt_vec")]
        eta_se: Option<Vec<f64>>,
    }

    #[test]
    fn nonfinite_values_round_trip_losslessly() {
        let mut columns = BTreeMap::new();
        columns.insert("survival_prob".to_string(), vec![1.0, 0.0]);
        let probe = ProducerProbe {
            times: vec![0.5, 1.0, 2.0],
            cumulative_hazard: vec![
                vec![0.0, 1.5, f64::INFINITY],
                vec![f64::NEG_INFINITY, f64::NAN, 7.0],
            ],
            linear_predictor: vec![1.99, f64::INFINITY, -3.0],
            columns,
            survival_se: Some(vec![vec![f64::INFINITY, 0.1]]),
            eta_se: Some(vec![f64::NAN, 0.25]),
        };

        let json = serde_json::to_string(&probe).expect("serialize must not fail");
        // The default serde_json encoding would have emitted `null` here; ours
        // emits explicit tokens so the parse can recover the value.
        assert!(json.contains("\"Infinity\""), "json: {json}");
        assert!(json.contains("\"-Infinity\""), "json: {json}");
        assert!(json.contains("\"NaN\""), "json: {json}");
        assert!(!json.contains("null"), "no bare nulls expected: {json}");

        let back: ConsumerProbe = serde_json::from_str(&json).expect("parse must succeed");
        let cum = back.cumulative_hazard.expect("cumulative_hazard present");
        assert!(cum[0][2].is_infinite() && cum[0][2] > 0.0);
        assert!(cum[1][0].is_infinite() && cum[1][0] < 0.0);
        assert!(cum[1][1].is_nan());
        assert!(back.linear_predictor.expect("lp present")[1].is_infinite());
        assert!(back.survival_se.expect("se present")[0][0].is_infinite());
        assert!(back.eta_se.expect("eta_se present")[0].is_nan());
        assert_eq!(back.times.expect("times present"), vec![0.5, 1.0, 2.0]);
        assert_eq!(back.columns.expect("columns present")["survival_prob"], vec![
            1.0, 0.0
        ]);
    }

    #[test]
    fn plain_numbers_and_missing_optional_fields_still_parse() {
        // Finite-only payloads keep the ordinary JSON-number wire form, and a
        // payload that omits the optional uncertainty fields parses to `None`.
        let json = r#"{"cumulative_hazard":[[0.0,1.0]],"linear_predictor":[0.5],"columns":{"a":[1.0]}}"#;
        let back: ConsumerProbe = serde_json::from_str(json).expect("plain numbers parse");
        assert_eq!(back.cumulative_hazard.unwrap(), vec![vec![0.0, 1.0]]);
        assert_eq!(back.linear_predictor.unwrap(), vec![0.5]);
        assert!(back.survival_se.is_none());
        assert!(back.eta_se.is_none());
        assert!(back.times.is_none());
    }
}

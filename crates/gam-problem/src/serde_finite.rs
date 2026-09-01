//! Structural non-finite-float guard for anything that implements [`Serialize`].
//!
//! ## Why this exists
//!
//! `serde_json` renders `f64::NAN` and `±f64::INFINITY` as the JSON literal
//! `null` — JSON has no encoding for them and `serde_json`'s serializer takes
//! the lossy branch silently. A persisted model carrying one non-finite scalar
//! therefore *writes* without complaint and only fails on the way back in, as
//!
//! ```text
//! invalid type: null, expected f64
//! ```
//!
//! — an error that names neither the field nor the fit that produced it, and
//! that surfaces arbitrarily far from the computation at fault (#2601).
//!
//! ## Why it is structural rather than a field list
//!
//! The pre-existing guards (`ensure_finite_scalar`, `validate_all_finite`, and
//! the hand-maintained `FittedModel::validate_numeric_finiteness`) each name one
//! field. A hand-maintained enumeration over a struct with hundreds of optional
//! numeric fields cannot stay complete: every new field is opted OUT by default,
//! so the guard silently stops covering the payload as the payload grows. That
//! is exactly how #2601's `null` reached a saved model.
//!
//! [`ensure_serialized_floats_are_finite`] instead walks the value through
//! `serde`'s own data model — the same traversal the JSON writer performs — so
//! *every* float that would be written is checked, by construction, with no
//! per-field opt-in. It tracks the struct-field / map-key / sequence-index path
//! as it descends, so the error names the offending scalar the way the
//! scalar-at-a-time guards do:
//!
//! ```text
//! payload.fit_result.blocks[3].edf must be finite, got NaN
//! ```
//!
//! The walk allocates nothing per scalar; only the current path (bounded by the
//! nesting depth) and the borrowed field names are held.

use serde::ser::{
    Impossible, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant,
    SerializeTuple, SerializeTupleStruct, SerializeTupleVariant, Serializer,
};
use std::fmt::{self, Display, Write as _};

/// A non-finite float found at `path` while walking a serializable value.
#[derive(Debug, Clone, PartialEq)]
pub struct NonFiniteFloat {
    /// Dotted / indexed path to the offending scalar, e.g.
    /// `payload.fit_result.blocks[3].edf`. Empty when the value serialized is
    /// itself a bare float.
    pub path: String,
    /// The offending value, widened to `f64` (`f32` inputs keep their class:
    /// a `f32::NAN` reports as `NaN`).
    pub value: f64,
}

impl Display for NonFiniteFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            write!(f, "value must be finite, got {}", self.value)
        } else {
            write!(f, "{} must be finite, got {}", self.path, self.value)
        }
    }
}

impl std::error::Error for NonFiniteFloat {}

/// Error channel of the walking serializer.
#[derive(Debug)]
enum WalkError {
    NonFinite(NonFiniteFloat),
    Custom(String),
}

impl Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WalkError::NonFinite(found) => Display::fmt(found, f),
            WalkError::Custom(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for WalkError {}

impl serde::ser::Error for WalkError {
    fn custom<T: Display>(msg: T) -> Self {
        WalkError::Custom(msg.to_string())
    }
}

/// Cursor over the serde data model carrying the path to the current value.
struct FloatWalker {
    path: String,
}

impl FloatWalker {
    /// Append `.name` (or `name` at the root) and return the previous length so
    /// the caller can truncate back after descending.
    fn push_field(&mut self, name: &str) -> usize {
        let restore = self.path.len();
        if !self.path.is_empty() {
            self.path.push('.');
        }
        self.path.push_str(name);
        restore
    }

    fn check(&self, value: f64) -> Result<(), WalkError> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(WalkError::NonFinite(NonFiniteFloat {
                path: self.path.clone(),
                value,
            }))
        }
    }
}

/// Render a map key into the path. Only string and integer keys are
/// representable in JSON objects, which is the format this guard protects; any
/// other key shape falls back to a positional index so the path stays useful.
struct KeyRenderer;

impl KeyRenderer {

}

impl Serializer for KeyRenderer {
    type Ok = String;
    type Error = WalkError;
    type SerializeSeq = Impossible<String, WalkError>;
    type SerializeTuple = Impossible<String, WalkError>;
    type SerializeTupleStruct = Impossible<String, WalkError>;
    type SerializeTupleVariant = Impossible<String, WalkError>;
    type SerializeMap = Impossible<String, WalkError>;
    type SerializeStruct = Impossible<String, WalkError>;
    type SerializeStructVariant = Impossible<String, WalkError>;

}

impl<'a> Serializer for &'a mut FloatWalker {
    type Ok = ();
    type Error = WalkError;
    type SerializeSeq = SeqWalker<'a>;
    type SerializeTuple = SeqWalker<'a>;
    type SerializeTupleStruct = SeqWalker<'a>;
    type SerializeTupleVariant = VariantSeqWalker<'a>;
    type SerializeMap = MapWalker<'a>;
    type SerializeStruct = StructWalker<'a>;
    type SerializeStructVariant = StructWalker<'a>;

    // The integer widths and `char` carry no finiteness verdict, and saying so
    // once per width states that decision fourteen times over. The narrow widths
    // widen losslessly into the widest one of their signedness — the shape
    // `KeyRenderer` already uses above — so "an integer is not a float" is
    // decided in one place per signedness, and a future verdict (a range check,
    // say) has one place to live.

}

/// What opened a compound, named for the length-coverage assertion below.
#[derive(Clone, Copy)]
struct Origin {
    /// The type name serde supplied, or a literal for the anonymous compounds
    /// (a bare sequence or map has no name in the data model).
    type_name: &'static str,
    /// `Some((variant, variant_index))` when the compound is an enum variant.
    variant: Option<(&'static str, u32)>,
}

impl Origin {
    fn plain(type_name: &'static str) -> Self {
        Origin {
            type_name,
            variant: None,
        }
    }

    fn variant(type_name: &'static str, variant_index: u32, variant: &'static str) -> Self {
        Origin {
            type_name,
            variant: Some((variant, variant_index)),
        }
    }
}

impl Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.variant {
            Some((variant, variant_index)) => write!(
                f,
                "`{}::{variant}` (variant #{variant_index})",
                self.type_name
            ),
            None => f.write_str(self.type_name),
        }
    }
}

/// Sequence / tuple cursor: elements are addressed by position.
struct SeqWalker<'a> {
    walker: &'a mut FloatWalker,
    index: usize,
    announced: Option<usize>,
    origin: Origin,
}

impl<'a> SeqWalker<'a> {
    fn new(walker: &'a mut FloatWalker, announced: Option<usize>, origin: Origin) -> Self {
        SeqWalker {
            walker,
            index: 0,
            announced,
            origin,
        }
    }

}

impl SerializeSeq for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn end(self) -> Result<(), WalkError> {
        self.finish();
        Ok(())
    }
}

impl SerializeTuple for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn end(self) -> Result<(), WalkError> {
        SerializeSeq::end(self)
    }
}

impl SerializeTupleStruct for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn end(self) -> Result<(), WalkError> {
        SerializeSeq::end(self)
    }
}

/// A tuple variant additionally owns the pushed variant-name segment.
struct VariantSeqWalker<'a> {
    seq: SeqWalker<'a>,
    restore: usize,
}

impl SerializeTupleVariant for VariantSeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn end(self) -> Result<(), WalkError> {
        self.seq.finish();
        self.seq.walker.pop_to(self.restore);
        Ok(())
    }
}

/// Map cursor: the key is rendered into the path, then the value is walked.
struct MapWalker<'a> {
    walker: &'a mut FloatWalker,
    /// Set while a key has been consumed and its value not yet walked.
    restore: Option<usize>,
    announced: Option<usize>,
    entries: usize,
    origin: Origin,
}

impl SerializeMap for MapWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_key<T>(&mut self, key: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        // A key that cannot be rendered as a scalar is not JSON-encodable at
        // all; fall back to a positional marker so the walk (and its float
        // verdict) still completes.
        let rendered = key
            .serialize(KeyRenderer)
            .unwrap_or_else(|err| format!("<unrenderable key: {err}>"));
        self.restore = Some(self.walker.push_field(&rendered));
        self.entries += 1;
        Ok(())
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        let outcome = value.serialize(&mut *self.walker);
        if let Some(restore) = self.restore.take() {
            self.walker.pop_to(restore);
        }
        outcome
    }

    fn end(self) -> Result<(), WalkError> {
        assert_announced_len(&self.origin, self.announced, self.entries);
        Ok(())
    }
}

/// Struct cursor: fields are addressed by name.
struct StructWalker<'a> {
    walker: &'a mut FloatWalker,
    /// `Some` for a struct *variant*, whose variant-name segment must be popped
    /// when the compound ends.
    restore: Option<usize>,
    announced: usize,
    fields: usize,
    origin: Origin,
}

impl SerializeStruct for StructWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn end(self) -> Result<(), WalkError> {
        assert_announced_len(&self.origin, Some(self.announced), self.fields);
        if let Some(restore) = self.restore {
            self.walker.pop_to(restore);
        }
        Ok(())
    }
}

impl SerializeStructVariant for StructWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn end(self) -> Result<(), WalkError> {
        SerializeStruct::end(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use std::collections::BTreeMap;

    #[derive(Serialize)]
    struct Leaf {
        edf: f64,
        name: String,
    }

    #[derive(Serialize)]
    struct Root {
        blocks: Vec<Leaf>,
        scale: Option<f64>,
        counts: Vec<u32>,
        by_term: BTreeMap<String, f64>,
    }

    fn root() -> Root {
        Root {
            blocks: vec![
                Leaf {
                    edf: 1.0,
                    name: "a".to_string(),
                },
                Leaf {
                    edf: 2.0,
                    name: "b".to_string(),
                },
            ],
            scale: Some(0.5),
            counts: vec![1, 2, 3],
            by_term: BTreeMap::from([("s(x)".to_string(), 3.25)]),
        }
    }

    #[test]
    fn all_finite_payload_passes() {
        assert!(ensure_serialized_floats_are_finite(&root()).is_ok());
    }

    #[test]
    fn nested_sequence_element_reports_indexed_path() {
        let mut value = root();
        value.blocks[1].edf = f64::NAN;
        let err = ensure_serialized_floats_are_finite(&value).unwrap_err();
        assert_eq!(err.path, "blocks[1].edf");
        assert!(err.value.is_nan());
        assert!(
            err.to_string().contains("blocks[1].edf must be finite"),
            "message should name the path: {err}"
        );
    }

    #[test]
    fn optional_scalar_reports_its_field() {
        let mut value = root();
        value.scale = Some(f64::INFINITY);
        let err = ensure_serialized_floats_are_finite(&value).unwrap_err();
        assert_eq!(err.path, "scale");
        assert_eq!(err.value, f64::INFINITY);
    }

    #[test]
    fn map_value_reports_its_key() {
        let mut value = root();
        value
            .by_term
            .insert("s(z)".to_string(), f64::NEG_INFINITY);
        let err = ensure_serialized_floats_are_finite(&value).unwrap_err();
        assert_eq!(err.path, "by_term.s(z)");
    }

    #[test]
    fn none_is_not_a_non_finite_float() {
        let mut value = root();
        value.scale = None;
        assert!(ensure_serialized_floats_are_finite(&value).is_ok());
    }

    #[test]
    fn bare_scalar_has_empty_path() {
        let err = ensure_serialized_floats_are_finite(&f64::NAN).unwrap_err();
        assert!(err.path.is_empty());
        assert!(err.to_string().starts_with("value must be finite"));
    }

    #[test]
    fn f32_non_finite_is_caught_and_widened() {
        #[derive(Serialize)]
        struct Small {
            w: f32,
        }
        let err = ensure_serialized_floats_are_finite(&Small { w: f32::NAN }).unwrap_err();
        assert_eq!(err.path, "w");
        assert!(err.value.is_nan());
    }

    #[test]
    fn struct_variant_and_newtype_variant_paths_are_reported() {
        #[derive(Serialize)]
        enum Node {
            Scale { phi: f64 },
            Raw(f64),
        }
        #[derive(Serialize)]
        struct Holder {
            node: Node,
        }
        let err =
            ensure_serialized_floats_are_finite(&Holder { node: Node::Scale { phi: f64::NAN } })
                .unwrap_err();
        assert_eq!(err.path, "node.Scale.phi");
        let err = ensure_serialized_floats_are_finite(&Holder {
            node: Node::Raw(f64::INFINITY),
        })
        .unwrap_err();
        assert_eq!(err.path, "node.Raw");
    }

    #[test]
    fn path_state_is_restored_after_each_branch() {
        // A finite branch visited BEFORE the offending one must not leave its
        // segments on the path (the truncate-on-exit contract).
        #[derive(Serialize)]
        struct Two {
            first: Vec<Leaf>,
            second: f64,
        }
        let err = ensure_serialized_floats_are_finite(&Two {
            first: vec![Leaf {
                edf: 1.0,
                name: "ok".to_string(),
            }],
            second: f64::NAN,
        })
        .unwrap_err();
        assert_eq!(err.path, "second");
    }

    #[test]
    fn enum_keyed_map_reports_the_variant_as_its_key() {
        // A unit-variant key is written as the bare variant name, so that is the
        // segment the path must carry.
        #[derive(Serialize, PartialEq, Eq, PartialOrd, Ord)]
        enum Term {
            Linear,
            Smooth,
        }
        #[derive(Serialize)]
        struct Holder {
            by_term: BTreeMap<Term, f64>,
        }
        let err = ensure_serialized_floats_are_finite(&Holder {
            by_term: BTreeMap::from([(Term::Linear, 1.0), (Term::Smooth, f64::NAN)]),
        })
        .unwrap_err();
        assert_eq!(err.path, "by_term.Smooth");
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "announced 2 elements but emitted 1")]
    fn a_compound_that_under_emits_its_announced_length_is_caught() {
        // The coverage claim is only as good as serde's length contract: a
        // compound that emits fewer elements than it announced has subtrees the
        // walk never visited, and would otherwise pass by silence.
        struct UnderEmitting;
        impl Serialize for UnderEmitting {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut seq = serializer.serialize_seq(Some(2))?;
                seq.serialize_element(&1.0f64)?;
                seq.end()
            }
        }
        // `assert_announced_len` panics before this returns, which is what
        // `#[should_panic]` catches. Asserting on the result anyway keeps the
        // test honest: were that assertion ever removed, the under-emitting
        // walk would return `Ok` and this would fail rather than pass silently.
        assert!(
            ensure_serialized_floats_are_finite(&UnderEmitting).is_err(),
            "a compound emitting fewer elements than it announced must not pass"
        );
    }

    #[test]
    fn walk_agrees_with_what_serde_json_would_write() {
        // The contract: the guard rejects exactly the payloads whose JSON
        // rendering contains a `null` that came from a float. Verify on a value
        // that serde_json silently lossy-renders.
        let mut value = root();
        value.blocks[0].edf = f64::NAN;
        let json = serde_json::to_string(&value).expect("serde_json renders NaN as null");
        assert!(
            json.contains("\"edf\":null"),
            "precondition: serde_json writes NaN as null, got {json}"
        );
        assert!(ensure_serialized_floats_are_finite(&value).is_err());
    }
}

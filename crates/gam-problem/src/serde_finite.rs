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

/// Walk `value` through `serde`'s data model and fail on the FIRST non-finite
/// `f32`/`f64` that a serializer would emit, reporting its path.
///
/// This is the write-side counterpart of the load-side type error: it converts
/// "a `null` will silently appear in the output" into a typed refusal at the
/// point of origin.
pub fn ensure_serialized_floats_are_finite<T>(value: &T) -> Result<(), NonFiniteFloat>
where
    T: Serialize + ?Sized,
{
    let mut walker = FloatWalker { path: String::new() };
    match value.serialize(&mut walker) {
        Ok(()) => Ok(()),
        Err(WalkError::NonFinite(found)) => Err(found),
        // `Custom` can only arise from a `Serialize` impl that itself reports an
        // error (e.g. a map with an unrepresentable key). Such a value cannot be
        // serialized to JSON either, so there is no float verdict to give and
        // the writer downstream will surface the same failure with its own
        // message. Treat it as "nothing non-finite found here".
        Err(WalkError::Custom(_)) => Ok(()),
    }
}

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

    fn push_index(&mut self, index: usize) -> usize {
        let restore = self.path.len();
        // `fmt::Write` for `String` never returns `Err`, so this names an
        // invariant of the sink rather than hiding a failure mode. Do not
        // "remove the Result" by writing `push_str(&index.to_string())`: that
        // allocates once per sequence element and breaks this module's
        // no-allocation-per-scalar promise.
        write!(self.path, "[{index}]").expect("`String`'s `fmt::Write` impl is infallible");
        restore
    }

    fn pop_to(&mut self, restore: usize) {
        self.path.truncate(restore);
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
    /// A key serde offered as a compound value. JSON object keys are strings,
    /// so there is no spelling for it; name the shape that was offered so the
    /// `<key>` placeholder that lands in the path can be traced back to the type
    /// that produced it.
    fn not_a_scalar(shape: impl Display) -> WalkError {
        WalkError::Custom(format!("map key is not a scalar: {shape}"))
    }

    /// Spelling of an enum variant used as a map key. `variant` is what the JSON
    /// writer emits as the object key, so it is what the path segment must be —
    /// but `derive` is not the only source of `Serialize` impls, and an empty
    /// name would splice an invisible segment into the path. Fall back to the
    /// enum and the variant index, which serde always supplies.
    fn variant_key(name: &'static str, variant_index: u32, variant: &'static str) -> String {
        if variant.is_empty() {
            format!("{name}#{variant_index}")
        } else {
            variant.to_string()
        }
    }
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

    fn serialize_str(self, value: &str) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_bool(self, value: bool) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_i64(self, value: i64) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_i128(self, value: i128) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_u64(self, value: u64) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_u128(self, value: u128) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_i8(self, value: i8) -> Result<String, WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_i16(self, value: i16) -> Result<String, WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_i32(self, value: i32) -> Result<String, WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_u8(self, value: u8) -> Result<String, WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_u16(self, value: u16) -> Result<String, WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_u32(self, value: u32) -> Result<String, WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_f32(self, value: f32) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_f64(self, value: f64) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_char(self, value: char) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<String, WalkError> {
        // Byte strings have no JSON object-key spelling. The length is the one
        // property that distinguishes two such keys in the path without
        // rendering an unbounded blob into it.
        Ok(format!("<{} bytes>", value.len()))
    }

    fn serialize_none(self) -> Result<String, WalkError> {
        Ok("null".to_string())
    }

    fn serialize_some<T>(self, value: &T) -> Result<String, WalkError>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<String, WalkError> {
        Ok("null".to_string())
    }

    fn serialize_unit_struct(self, name: &'static str) -> Result<String, WalkError> {
        Ok(name.to_string())
    }

    fn serialize_unit_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<String, WalkError> {
        Ok(Self::variant_key(name, variant_index, variant))
    }

    fn serialize_newtype_struct<T>(self, name: &'static str, value: &T) -> Result<String, WalkError>
    where
        T: Serialize + ?Sized,
    {
        // A newtype struct is transparent in JSON: the key is the inner value's
        // spelling. If the inner value has no spelling, name the wrapper — that
        // is the type the caller wrote, and the only one they can act on.
        value
            .serialize(self)
            .map_err(|inner| WalkError::Custom(format!("inside newtype struct `{name}`: {inner}")))
    }

    fn serialize_newtype_variant<T>(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<String, WalkError>
    where
        T: Serialize + ?Sized,
    {
        // Two entries keyed by the same variant but carrying different payloads
        // are distinct keys, so the payload belongs in the spelling. Dropping it
        // (as this did) collapsed them onto one path segment.
        let inner = value.serialize(KeyRenderer)?;
        Ok(format!(
            "{}({inner})",
            Self::variant_key(name, variant_index, variant)
        ))
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, WalkError> {
        Err(Self::not_a_scalar(match len {
            Some(len) => format!("a sequence of {len} elements"),
            None => "a sequence of unannounced length".to_string(),
        }))
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, WalkError> {
        Err(Self::not_a_scalar(format_args!("a {len}-tuple")))
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, WalkError> {
        Err(Self::not_a_scalar(format_args!(
            "tuple struct `{name}` with {len} fields"
        )))
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, WalkError> {
        Err(Self::not_a_scalar(format_args!(
            "tuple variant `{name}::{variant}` (variant #{variant_index}) with {len} fields"
        )))
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, WalkError> {
        Err(Self::not_a_scalar(match len {
            Some(len) => format!("a map of {len} entries"),
            None => "a map of unannounced length".to_string(),
        }))
    }

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, WalkError> {
        Err(Self::not_a_scalar(format_args!(
            "struct `{name}` with {len} fields"
        )))
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, WalkError> {
        Err(Self::not_a_scalar(format_args!(
            "struct variant `{name}::{variant}` (variant #{variant_index}) with {len} fields"
        )))
    }
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

    fn serialize_f64(self, value: f64) -> Result<(), WalkError> {
        self.check(value)
    }

    fn serialize_f32(self, value: f32) -> Result<(), WalkError> {
        // Widen for the verdict AND the message: `f32::NAN as f64` is still
        // NaN and `f32::INFINITY as f64` is still infinite, so the class is
        // preserved exactly.
        self.check(f64::from(value))
    }

    // The integer widths and `char` carry no finiteness verdict, and saying so
    // once per width states that decision fourteen times over. The narrow widths
    // widen losslessly into the widest one of their signedness — the shape
    // `KeyRenderer` already uses above — so "an integer is not a float" is
    // decided in one place per signedness, and a future verdict (a range check,
    // say) has one place to live.

    fn serialize_bool(self, _: bool) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_i8(self, value: i8) -> Result<(), WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_i16(self, value: i16) -> Result<(), WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_i32(self, value: i32) -> Result<(), WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_i64(self, value: i64) -> Result<(), WalkError> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i128(self, _: i128) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_u8(self, value: u8) -> Result<(), WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_u16(self, value: u16) -> Result<(), WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_u32(self, value: u32) -> Result<(), WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_u64(self, value: u64) -> Result<(), WalkError> {
        self.serialize_u128(u128::from(value))
    }

    fn serialize_u128(self, _: u128) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_char(self, value: char) -> Result<(), WalkError> {
        // JSON writes a `char` as the one-character string it encodes to.
        self.serialize_str(value.encode_utf8(&mut [0u8; 4]))
    }

    fn serialize_str(self, _: &str) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_bytes(self, _: &[u8]) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_none(self) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_some<T>(self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_unit_struct(self, _: &'static str) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_newtype_struct<T>(self, _: &'static str, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        // Externally tagged enums serialize as `{"Variant": payload}`, so the
        // variant name IS a path segment in the emitted JSON — and an empty one
        // would splice an invisible segment into the reported path.
        assert!(
            !variant.is_empty(),
            "`{name}` variant #{variant_index} has an empty name; \
             its path segment would be invisible"
        );
        let restore = self.push_field(variant);
        let outcome = value.serialize(&mut *self);
        self.pop_to(restore);
        outcome
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<SeqWalker<'a>, WalkError> {
        Ok(SeqWalker::new(self, len, Origin::plain("a sequence")))
    }

    fn serialize_tuple(self, len: usize) -> Result<SeqWalker<'a>, WalkError> {
        Ok(SeqWalker::new(self, Some(len), Origin::plain("a tuple")))
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<SeqWalker<'a>, WalkError> {
        Ok(SeqWalker::new(self, Some(len), Origin::plain(name)))
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<VariantSeqWalker<'a>, WalkError> {
        let origin = Origin::variant(name, variant_index, variant);
        let restore = self.push_field(variant);
        Ok(VariantSeqWalker {
            seq: SeqWalker::new(self, Some(len), origin),
            restore,
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<MapWalker<'a>, WalkError> {
        Ok(MapWalker {
            walker: self,
            restore: None,
            announced: len,
            entries: 0,
            origin: Origin::plain("a map"),
        })
    }

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<StructWalker<'a>, WalkError> {
        Ok(StructWalker {
            walker: self,
            restore: None,
            announced: len,
            fields: 0,
            origin: Origin::plain(name),
        })
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<StructWalker<'a>, WalkError> {
        let origin = Origin::variant(name, variant_index, variant);
        let restore = self.push_field(variant);
        Ok(StructWalker {
            walker: self,
            restore: Some(restore),
            announced: len,
            fields: 0,
            origin,
        })
    }

    fn collect_str<T>(self, _: &T) -> Result<(), WalkError>
    where
        T: Display + ?Sized,
    {
        Ok(())
    }

    fn is_human_readable(&self) -> bool {
        // The format this guard protects is JSON. Types whose `Serialize` impl
        // branches on this (e.g. compact binary encodings) must be walked in the
        // same shape the JSON writer will use, or the guard would inspect a
        // different set of floats than the one persisted.
        true
    }
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

/// serde's contract: a compound that announces a length emits exactly that many
/// elements. The guard's whole claim — *every* float the writer emits is
/// checked — rests on the walk seeing the same elements the writer will, so a
/// compound that emits a different number than it announced has subtrees the
/// walk never visited. That is precisely the silent coverage loss this module
/// exists to prevent (#2601), so announced lengths are checked, not ignored.
fn assert_announced_len(origin: &Origin, announced: Option<usize>, emitted: usize) {
    if let Some(announced) = announced {
        assert_eq!(
            emitted, announced,
            "{origin} announced {announced} elements but emitted {emitted}"
        );
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

    fn finish(&self) {
        assert_announced_len(&self.origin, self.announced, self.index);
    }
}

impl SerializeSeq for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        let restore = self.walker.push_index(self.index);
        let outcome = value.serialize(&mut *self.walker);
        self.walker.pop_to(restore);
        self.index += 1;
        outcome
    }

    fn end(self) -> Result<(), WalkError> {
        self.finish();
        Ok(())
    }
}

impl SerializeTuple for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<(), WalkError> {
        SerializeSeq::end(self)
    }
}

impl SerializeTupleStruct for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        SerializeSeq::serialize_element(self, value)
    }

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

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        SerializeSeq::serialize_element(&mut self.seq, value)
    }

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

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        let restore = self.walker.push_field(key);
        let outcome = value.serialize(&mut *self.walker);
        self.walker.pop_to(restore);
        self.fields += 1;
        outcome
    }

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

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        SerializeStruct::serialize_field(self, key, value)
    }

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

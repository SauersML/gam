//! Bounded immutable native codewords for ordinary saved-artifact replay.
//!
//! The cache budget counts packed capacities and decoded numeric buffers, not caller-owned
//! source arrays. Construction additionally uses one operator's temporary real/index arrays
//! and decoded buffers. Complete artifact messages, envelope copies, metadata, allocator and
//! execution buffers are outside this cache budget. No candidate coefficients enter the cache.
use crate::{artifact::Artifact, operator_program::{Declarations, NativeOperatorCodec, NativeOperatorCodecStats, NativeOperatorCodecUsage, OperatorBody}};
use std::time::Instant;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Preflight {
    pub packed_bytes: usize,
    pub decoded_numeric_bytes: usize,
    pub caller_source_numeric_bytes: usize,
    pub largest_operator_numeric_bytes: usize,
    pub seconds: f64,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct Timings {
    pub f32_seconds: f64,
    pub encode_seconds: f64,
    pub decode_seconds: f64,
    pub reencode_seconds: f64,
}
pub struct Canonical {
    pub decoded: Artifact,
    pub bytes: Vec<u8>,
    pub timings: Timings,
}
/// Construct once per native source and reuse across grafts and saved-byte replay.
pub struct CanonicalArtifactCache {
    codec: NativeOperatorCodec,
    preflight: Preflight,
}
impl CanonicalArtifactCache {
    /// Refuse non-f32 native literals. This does not round or duplicate caller source arrays.
    /// The exact size pass itself has per-operator lattice temporaries, reported separately.
    pub fn new(native: &Artifact, budget_bytes: usize) -> Result<Self, String> {
        let started = Instant::now();
        if !native.has_f32_literals() { return Err("canonical native cache requires f32 literals".into()); }
        let source = native.message_program()?;
        let numeric = |body: &OperatorBody| -> Result<usize, String> {
            let (values, mask) = match body {
                OperatorBody::Identity => (0, 0),
                OperatorBody::Diagonal { values, .. } => (values.len(), 0),
                OperatorBody::Dense { values, present, .. } => (values.len(), present.len()),
                OperatorBody::LowRank { left, right, .. } => (left.len().checked_add(right.len()).ok_or("numeric overflow")?, 0),
            };
            values.checked_mul(8).and_then(|n| n.checked_add(mask)).ok_or_else(|| "numeric overflow".into())
        };
        let mut preflight = Preflight { packed_bytes: 0, decoded_numeric_bytes: 0, caller_source_numeric_bytes: 0, largest_operator_numeric_bytes: 0, seconds: 0. };
        for op in &source.operators {
            let bytes = numeric(&op.body)?;
            preflight.decoded_numeric_bytes = preflight.decoded_numeric_bytes.checked_add(bytes).ok_or("numeric overflow")?;
            preflight.largest_operator_numeric_bytes = preflight.largest_operator_numeric_bytes.max(bytes);
        }
        for op in &native.program.operators {
            preflight.caller_source_numeric_bytes = preflight.caller_source_numeric_bytes.checked_add(numeric(&op.body)?).ok_or("source numeric overflow")?;
        }
        if preflight.decoded_numeric_bytes > budget_bytes { return Err("native cache numeric preflight exceeds budget".into()); }
        for op in &source.operators {
            let (a, b) = op.code_bits().map_err(|e| e.to_string())?;
            let bytes = usize::try_from(a.checked_add(b).ok_or("code length overflow")?.div_ceil(8)).map_err(|e| e.to_string())?;
            preflight.packed_bytes = preflight.packed_bytes.checked_add(bytes).ok_or("packed overflow")?;
            if preflight.packed_bytes.checked_add(preflight.decoded_numeric_bytes).is_none_or(|n| n > budget_bytes) {
                return Err("native cache exact packed/numeric preflight exceeds budget".into());
            }
        }
        preflight.seconds = started.elapsed().as_secs_f64();
        let codec = NativeOperatorCodec::new(&source, budget_bytes).map_err(|e| e.to_string())?;
        let stats = codec.stats();
        if stats.encoded_capacity_bytes != preflight.packed_bytes || stats.decoded_numeric_bytes != preflight.decoded_numeric_bytes {
            return Err("native cache allocation differs from exact preflight".into());
        }
        Ok(Self { codec, preflight })
    }
    pub fn preflight(&self) -> &Preflight { &self.preflight }
    pub fn stats(&self) -> NativeOperatorCodecStats { self.codec.stats() }
    pub fn usage(&self) -> NativeOperatorCodecUsage { self.codec.usage() }
    /// Check canonical ordinary byte equality and executable indices after f32 replay.
    pub fn canonical(&self, artifact: &Artifact) -> Result<Canonical, String> {
        let started = Instant::now();
        let f32 = artifact.f32_literals()?;
        let f32_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let bytes = f32.to_bytes_with_native_codec(&self.codec)?;
        let encode_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let decoded = Artifact::from_bytes_with_native_codec(&bytes, &artifact.program.declarations, &self.codec)?;
        let decode_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        if decoded.to_bytes_with_native_codec(&self.codec)? != bytes { return Err("noncanonical candidate wire replay".into()); }
        let reencode_seconds = started.elapsed().as_secs_f64();
        if decoded.program.nodes != artifact.program.nodes || decoded.program.output != artifact.program.output
            || decoded.program.operators.len() != artifact.program.operators.len()
            || decoded.program.rules.len() != artifact.program.rules.len()
            || decoded.program.rules.iter().zip(&artifact.program.rules).any(|(a,b)| a.nodes != b.nodes || a.output != b.output || a.inputs != b.inputs)
            || decoded.program.operators.iter().zip(&artifact.program.operators).any(|(a,b)| a.rows != b.rows || a.cols != b.cols)
            || decoded.places != artifact.places {
            return Err("wire replay changed executable reference indices".into());
        }
        Ok(Canonical { decoded, bytes, timings: Timings { f32_seconds, encode_seconds, decode_seconds, reencode_seconds } })
    }
    /// Parse the entire ordinary envelope, bindings, controls and derived bodies. Never trusts
    /// a source pointer for saved input: substitution requires exact codeword bytes.
    pub fn decode_saved(&self, bytes: &[u8], declarations: &Declarations) -> Result<Artifact, String> {
        Artifact::from_bytes_with_native_codec(bytes, declarations, &self.codec)
    }
}

#[cfg(test)]
#[path = "canonical_artifact_tests.rs"]
mod tests;

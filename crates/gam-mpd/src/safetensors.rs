//! Read a `.safetensors` checkpoint into binary64 arrays (#2951).
//!
//! The format is an 8-byte little-endian header length `n`, `n` bytes of JSON mapping each
//! tensor name to its `dtype`, `shape` and `data_offsets` `[begin, end)` into the byte buffer that
//! follows, and that buffer, little-endian and C-ordered. The optional `__metadata__` entry is a
//! string map. Every stored float type read here widens to binary64 exactly: `F32`, `F16` and
//! `BF16` (the upper half of an `F32`) are subsets of binary64, so a native tensor enters the
//! program as the very reals the source multiplies by. A tensor of another element type (a `U8`
//! causal-mask buffer, an `I64` position table) is listed with its type and size and refused only
//! when it is read as reals: a checkpoint is not rejected for the buffers it carries.
//!
//! The file is memory-mapped; a read copies one tensor out, and its binary64 copy is reserved on
//! gam-runtime's memory governor before it is allocated.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gam_runtime::resource::{Governed, MemoryGovernor, MemoryReservationError};
use memmap2::Mmap;
use ndarray::{Array1, Array2};

/// A stored element type this reader widens to binary64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredFloat {
    F64,
    F32,
    F16,
    Bf16,
}

/// A stored element type: a float this reader widens, or another type of the format (`BOOL`,
/// `U8`, `I8`, `F8_*`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`), kept by name and size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoredType {
    Float(StoredFloat),
    Other { name: String, bytes: usize },
}

impl StoredType {
    fn parse(name: &str) -> Option<Self> {
        let other = |bytes| Some(Self::Other { name: name.to_string(), bytes });
        match name {
            "F64" => Some(Self::Float(StoredFloat::F64)),
            "F32" => Some(Self::Float(StoredFloat::F32)),
            "F16" => Some(Self::Float(StoredFloat::F16)),
            "BF16" => Some(Self::Float(StoredFloat::Bf16)),
            "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => other(1),
            "I16" | "U16" => other(2),
            "I32" | "U32" => other(4),
            "I64" | "U64" => other(8),
            _ => None,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Self::Float(float) => float.bytes(),
            Self::Other { bytes, .. } => *bytes,
        }
    }
}

impl StoredFloat {
    fn bytes(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// The element at the start of `raw`, widened exactly.
    fn widen(self, raw: &[u8]) -> f64 {
        match self {
            Self::F64 => f64::from_le_bytes(raw[..8].try_into().expect("eight bytes")),
            Self::F32 => f64::from(f32::from_le_bytes(raw[..4].try_into().expect("four bytes"))),
            Self::Bf16 => f64::from(f32::from_bits(u32::from(u16::from_le_bytes([raw[0], raw[1]])) << 16)),
            Self::F16 => {
                // sign, 5 exponent bits (bias 15), 10 fraction bits; every value is an exact
                // binary64 product of an integer and a power of two.
                let half = u16::from_le_bytes([raw[0], raw[1]]);
                let sign = if half >> 15 == 1 { -1.0 } else { 1.0 };
                let (exponent, fraction) = (i32::from((half >> 10) & 0x1f), f64::from(half & 0x3ff));
                match exponent {
                    0 => sign * fraction * 2f64.powi(-24),
                    31 if fraction == 0.0 => sign * f64::INFINITY,
                    31 => f64::NAN,
                    _ => sign * (1024.0 + fraction) * 2f64.powi(exponent - 25),
                }
            }
        }
    }
}

/// One tensor's entry in the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    pub dtype: StoredType,
    pub shape: Vec<usize>,
    begin: usize,
    end: usize,
}

#[derive(Debug)]
pub enum SafetensorsError {
    Io { path: PathBuf, error: std::io::Error },
    /// The header length, JSON or an entry does not describe the file.
    Header(String),
    /// A tensor stored in a type this reader does not widen, read as reals, or a type the format
    /// does not define.
    UnsupportedType { tensor: String, dtype: String },
    Missing(String),
    /// A tensor whose shape is not the one the caller needs.
    Shape { tensor: String, expected: Vec<usize>, found: Vec<usize> },
    Memory(MemoryReservationError),
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Header(message) => write!(f, "safetensors header: {message}"),
            Self::UnsupportedType { tensor, dtype } => write!(f, "{tensor} is stored as {dtype}, not F64, F32, F16 or BF16"),
            Self::Missing(tensor) => write!(f, "no tensor {tensor}"),
            Self::Shape { tensor, expected, found } => write!(f, "{tensor} has shape {found:?}, expected {expected:?}"),
            Self::Memory(error) => write!(f, "safetensors read: {error}"),
        }
    }
}

impl std::error::Error for SafetensorsError {}

/// A memory-mapped `.safetensors` file and its parsed header.
pub struct SafetensorsFile {
    mmap: Arc<Mmap>,
    data_start: usize,
    tensors: BTreeMap<String, TensorEntry>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self, SafetensorsError> {
        let io = |error| SafetensorsError::Io { path: path.to_path_buf(), error };
        let file = File::open(path).map_err(io)?;
        // SAFETY: the mapping is read-only and lives as long as `self`; a checkpoint is not
        // rewritten while it is being read.
        let mmap = unsafe { Mmap::map(&file) }.map_err(io)?;
        let header = |message: String| SafetensorsError::Header(message);
        let length = mmap
            .get(..8)
            .map(|raw| u64::from_le_bytes(raw.try_into().expect("eight bytes")))
            .ok_or_else(|| header("file shorter than its 8-byte header length".into()))?;
        let data_start = usize::try_from(length)
            .ok()
            .and_then(|length| length.checked_add(8))
            .filter(|&start| start <= mmap.len())
            .ok_or_else(|| header(format!("header length {length} exceeds the file")))?;
        let json: serde_json::Value =
            serde_json::from_slice(&mmap[8..data_start]).map_err(|error| header(error.to_string()))?;
        let entries = json.as_object().ok_or_else(|| header("not a JSON object".into()))?;
        let data_len = mmap.len() - data_start;
        let mut tensors = BTreeMap::new();
        for (name, entry) in entries {
            if name == "__metadata__" {
                continue;
            }
            let dtype_name = entry["dtype"].as_str().ok_or_else(|| header(format!("{name}: no dtype")))?;
            let dtype = StoredType::parse(dtype_name).ok_or_else(|| SafetensorsError::UnsupportedType {
                tensor: name.clone(),
                dtype: dtype_name.to_string(),
            })?;
            let integers = |key: &str| -> Result<Vec<usize>, SafetensorsError> {
                entry[key]
                    .as_array()
                    .ok_or_else(|| header(format!("{name}: no {key}")))?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| usize::try_from(value).ok())
                            .ok_or_else(|| header(format!("{name}: {key} holds a non-integer")))
                    })
                    .collect()
            };
            let shape = integers("shape")?;
            let offsets = integers("data_offsets")?;
            let [begin, end] = offsets[..] else {
                return Err(header(format!("{name}: data_offsets is not a pair")));
            };
            let bytes = shape
                .iter()
                .try_fold(dtype.bytes(), |acc, &axis| acc.checked_mul(axis))
                .ok_or_else(|| header(format!("{name}: size overflows")))?;
            if begin > end || end > data_len || end - begin != bytes {
                return Err(header(format!(
                    "{name}: offsets [{begin}, {end}) do not hold {shape:?} {dtype_name} in {data_len} bytes"
                )));
            }
            tensors.insert(name.clone(), TensorEntry { dtype, shape, begin, end });
        }
        Ok(Self { mmap: Arc::new(mmap), data_start, tensors })
    }

    /// Every tensor, by name.
    pub fn tensors(&self) -> &BTreeMap<String, TensorEntry> {
        &self.tensors
    }

    fn entry(&self, name: &str) -> Result<&TensorEntry, SafetensorsError> {
        self.tensors.get(name).ok_or_else(|| SafetensorsError::Missing(name.to_string()))
    }

    fn widened(&self, name: &str, entry: &TensorEntry) -> Result<Vec<f64>, SafetensorsError> {
        let float = match &entry.dtype {
            StoredType::Float(float) => *float,
            StoredType::Other { name: dtype, .. } => {
                return Err(SafetensorsError::UnsupportedType { tensor: name.to_string(), dtype: dtype.clone() });
            }
        };
        Ok(self.mmap[self.data_start + entry.begin..self.data_start + entry.end]
            .chunks_exact(float.bytes())
            .map(|raw| float.widen(raw))
            .collect())
    }

    /// The two-axis tensor `name`, which must be `rows × cols`, widened exactly to binary64.
    pub fn matrix(
        &self,
        governor: &MemoryGovernor,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Governed<Array2<f64>>, SafetensorsError> {
        let entry = self.entry(name)?;
        if entry.shape != [rows, cols] {
            return Err(SafetensorsError::Shape { tensor: name.into(), expected: vec![rows, cols], found: entry.shape.clone() });
        }
        let reservation = governor
            .try_reserve_dense_f64(rows, cols, "safetensors matrix")
            .map_err(SafetensorsError::Memory)?;
        let values = Array2::from_shape_vec((rows, cols), self.widened(name, entry)?).expect("the header's shape holds the data");
        Ok(reservation.bind(values))
    }

    /// The two-axis float tensor `name`, which must be `rows × cols`, where the file stores it
    /// ([`Stored`]): nothing is read or widened until its values are asked for.
    pub fn stored(&self, name: &str, rows: usize, cols: usize) -> Result<Stored, SafetensorsError> {
        let entry = self.entry(name)?;
        if entry.shape != [rows, cols] {
            return Err(SafetensorsError::Shape { tensor: name.into(), expected: vec![rows, cols], found: entry.shape.clone() });
        }
        let float = match &entry.dtype {
            StoredType::Float(float) => *float,
            StoredType::Other { name: dtype, .. } => return Err(SafetensorsError::UnsupportedType { tensor: name.to_string(), dtype: dtype.clone() }),
        };
        Ok(Stored { map: Arc::clone(&self.mmap), start: self.data_start + entry.begin, float, rows, cols })
    }

    /// The one-axis tensor `name`, which must hold `len` entries, widened exactly to binary64.
    pub fn vector(&self, name: &str, len: usize) -> Result<Array1<f64>, SafetensorsError> {
        let entry = self.entry(name)?;
        if entry.shape != [len] {
            return Err(SafetensorsError::Shape { tensor: name.into(), expected: vec![len], found: entry.shape.clone() });
        }
        Ok(Array1::from_vec(self.widened(name, entry)?))
    }
}

/// A matrix's reals where a safetensors file stores them: `rows × cols` row-major elements of
/// `float` from byte `start` of the file's memory map. Reading widens them exactly to binary64
/// ([`Stored::matrix`]), or to f32 for the stored types it holds exactly ([`Stored::f32_values`]).
#[derive(Clone)]
pub struct Stored {
    map: Arc<Mmap>,
    start: usize,
    float: StoredFloat,
    rows: usize,
    cols: usize,
}

impl Stored {
    /// Its shape, rows × cols.
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn bytes(&self) -> &[u8] {
        &self.map[self.start..self.start + self.rows * self.cols * self.float.bytes()]
    }

    /// Its elements in row-major order, widened exactly to binary64.
    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        self.bytes().chunks_exact(self.float.bytes()).map(|raw| self.float.widen(raw))
    }

    /// The matrix, widened exactly to binary64.
    #[must_use]
    pub fn matrix(&self) -> Array2<f64> {
        Array2::from_shape_vec((self.rows, self.cols), self.values().collect()).expect("the stored shape holds the data")
    }

    /// Its elements in row-major order as f32, when that holds every stored value exactly (stored
    /// F32, F16 or BF16).
    #[must_use]
    pub fn f32_values(&self) -> Option<Vec<f32>> {
        (self.float != StoredFloat::F64).then(|| self.values().map(|v| v as f32).collect())
    }

    /// Rows `rows` of it, where the file stores them (a row range is contiguous).
    #[must_use]
    pub fn rows(&self, rows: std::ops::Range<usize>) -> Option<Self> {
        (rows.start <= rows.end && rows.end <= self.rows).then(|| Self { start: self.start + rows.start * self.cols * self.float.bytes(), rows: rows.len(), ..self.clone() })
    }
}

impl PartialEq for Stored {
    /// The same reals stored the same way.
    fn eq(&self, other: &Self) -> bool {
        self.dim() == other.dim() && self.float == other.float && self.bytes() == other.bytes()
    }
}

impl fmt::Debug for Stored {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stored({}x{} {:?})", self.rows, self.cols, self.float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_governor;

    /// A file with an `F32` matrix, a `BF16` vector, an `F16` vector, a `U8` mask buffer and
    /// metadata, written byte for byte.
    fn fixture(dir: &Path) -> PathBuf {
        let matrix: [f32; 6] = [1.0, -2.5, 0.1, 3.0e-39, f32::MAX, -0.0];
        let vector: [u16; 3] = [0x3f80, 0xc040, 0x0001]; // 1, -3, the least bf16 subnormal
        let mut data = Vec::new();
        matrix.iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
        vector.iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
        data.extend_from_slice(&[1, 0, 1, 1]);
        let half: [u16; 3] = [0x3c00, 0xc200, 0x0001]; // 1, -3, the least f16 subnormal 2^-24
        half.iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "m": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
            "v": {"dtype": "BF16", "shape": [3], "data_offsets": [24, 30]},
            "mask": {"dtype": "U8", "shape": [2, 2], "data_offsets": [30, 34]},
            "h": {"dtype": "F16", "shape": [3], "data_offsets": [34, 40]},
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        let path = dir.join("fixture.safetensors");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn widens_every_stored_value_exactly() {
        let dir = std::env::temp_dir().join(format!("gam-safetensors-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = SafetensorsFile::open(&fixture(&dir)).unwrap();
        let m = file.matrix(test_governor(), "m", 2, 3).unwrap();
        let expected = [1.0f32, -2.5, 0.1, 3.0e-39, f32::MAX, -0.0].map(f64::from);
        assert!(m.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()));
        let v = file.vector("v", 3).unwrap();
        assert_eq!(v.to_vec(), vec![1.0, -3.0, f64::from(f32::from_bits(0x0001_0000))]);
        assert!(matches!(file.matrix(test_governor(), "m", 3, 2), Err(SafetensorsError::Shape { .. })));
        assert!(matches!(file.vector("absent", 1), Err(SafetensorsError::Missing(_))));
        assert_eq!(file.vector("h", 3).unwrap().to_vec(), vec![1.0, -3.0, 2f64.powi(-24)]);
        // The mask buffer is listed, and refused only when read as reals.
        assert_eq!(file.tensors()["mask"].dtype, StoredType::Other { name: "U8".to_string(), bytes: 1 });
        assert!(matches!(file.matrix(test_governor(), "mask", 2, 2), Err(SafetensorsError::UnsupportedType { .. })));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn refuses_offsets_that_do_not_hold_the_shape() {
        let dir = std::env::temp_dir().join(format!("gam-safetensors-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = serde_json::json!({"m": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 20]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 24]);
        let path = dir.join("bad.safetensors");
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(SafetensorsFile::open(&path), Err(SafetensorsError::Header(_))));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

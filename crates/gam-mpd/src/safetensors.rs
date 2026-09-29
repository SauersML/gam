//! Read a `.safetensors` checkpoint into binary64 arrays (#2951).
//!
//! The format is an 8-byte little-endian header length `n`, `n` bytes of JSON mapping each
//! tensor name to its `dtype`, `shape` and `data_offsets` `[begin, end)` into the byte buffer that
//! follows, and that buffer, little-endian and C-ordered. The optional `__metadata__` entry is a
//! string map. Every stored float type read here widens to binary64 exactly: `F32` and `BF16`
//! (the upper half of an `F32`) are subsets of binary64, so a native tensor enters the program as
//! the very reals the source multiplies by.
//!
//! The file is memory-mapped; a read copies one tensor out, and its binary64 copy is reserved on
//! gam-runtime's memory governor before it is allocated.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

use gam_runtime::resource::{Governed, MemoryGovernor, MemoryReservationError};
use memmap2::Mmap;
use ndarray::{Array1, Array2};

/// A stored element type this reader widens to binary64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredFloat {
    F64,
    F32,
    Bf16,
}

impl StoredFloat {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "F64" => Some(Self::F64),
            "F32" => Some(Self::F32),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }

    /// The element at the start of `raw`, widened exactly.
    fn widen(self, raw: &[u8]) -> f64 {
        match self {
            Self::F64 => f64::from_le_bytes(raw[..8].try_into().expect("eight bytes")),
            Self::F32 => f64::from(f32::from_le_bytes(raw[..4].try_into().expect("four bytes"))),
            Self::Bf16 => f64::from(f32::from_bits(u32::from(u16::from_le_bytes([raw[0], raw[1]])) << 16)),
        }
    }
}

/// One tensor's entry in the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    pub dtype: StoredFloat,
    pub shape: Vec<usize>,
    begin: usize,
    end: usize,
}

#[derive(Debug)]
pub enum SafetensorsError {
    Io { path: PathBuf, error: std::io::Error },
    /// The header length, JSON or an entry does not describe the file.
    Header(String),
    /// A tensor stored in a type this reader does not widen.
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
            Self::UnsupportedType { tensor, dtype } => write!(f, "{tensor} is stored as {dtype}, not F64, F32 or BF16"),
            Self::Missing(tensor) => write!(f, "no tensor {tensor}"),
            Self::Shape { tensor, expected, found } => write!(f, "{tensor} has shape {found:?}, expected {expected:?}"),
            Self::Memory(error) => write!(f, "safetensors read: {error}"),
        }
    }
}

impl std::error::Error for SafetensorsError {}

/// A memory-mapped `.safetensors` file and its parsed header.
pub struct SafetensorsFile {
    mmap: Mmap,
    data_start: usize,
    tensors: BTreeMap<String, TensorEntry>,
    metadata: BTreeMap<String, String>,
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
        let mut metadata = BTreeMap::new();
        for (name, entry) in entries {
            if name == "__metadata__" {
                for (key, value) in entry.as_object().into_iter().flatten() {
                    metadata.insert(key.clone(), value.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            let dtype_name = entry["dtype"].as_str().ok_or_else(|| header(format!("{name}: no dtype")))?;
            let dtype = StoredFloat::parse(dtype_name).ok_or_else(|| SafetensorsError::UnsupportedType {
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
        Ok(Self { mmap, data_start, tensors, metadata })
    }

    /// Every tensor, by name.
    pub fn tensors(&self) -> &BTreeMap<String, TensorEntry> {
        &self.tensors
    }

    /// The header's `__metadata__` strings.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    fn entry(&self, name: &str) -> Result<&TensorEntry, SafetensorsError> {
        self.tensors.get(name).ok_or_else(|| SafetensorsError::Missing(name.to_string()))
    }

    fn widened(&self, entry: &TensorEntry) -> Vec<f64> {
        self.mmap[self.data_start + entry.begin..self.data_start + entry.end]
            .chunks_exact(entry.dtype.bytes())
            .map(|raw| entry.dtype.widen(raw))
            .collect()
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
        let values = Array2::from_shape_vec((rows, cols), self.widened(entry)).expect("the header's shape holds the data");
        Ok(reservation.bind(values))
    }

    /// The one-axis tensor `name`, which must hold `len` entries, widened exactly to binary64.
    pub fn vector(&self, name: &str, len: usize) -> Result<Array1<f64>, SafetensorsError> {
        let entry = self.entry(name)?;
        if entry.shape != [len] {
            return Err(SafetensorsError::Shape { tensor: name.into(), expected: vec![len], found: entry.shape.clone() });
        }
        Ok(Array1::from_vec(self.widened(entry)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_governor;

    /// A file with an `F32` matrix, a `BF16` vector and metadata, written byte for byte.
    fn fixture(dir: &Path) -> PathBuf {
        let matrix: [f32; 6] = [1.0, -2.5, 0.1, 3.0e-39, f32::MAX, -0.0];
        let vector: [u16; 3] = [0x3f80, 0xc040, 0x0001]; // 1, -3, the least bf16 subnormal
        let mut data = Vec::new();
        matrix.iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
        vector.iter().for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "m": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
            "v": {"dtype": "BF16", "shape": [3], "data_offsets": [24, 30]},
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
        assert_eq!(file.metadata()["format"], "pt");
        let m = file.matrix(test_governor(), "m", 2, 3).unwrap();
        let expected = [1.0f32, -2.5, 0.1, 3.0e-39, f32::MAX, -0.0].map(f64::from);
        assert!(m.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()));
        let v = file.vector("v", 3).unwrap();
        assert_eq!(v.to_vec(), vec![1.0, -3.0, f64::from(f32::from_bits(0x0001_0000))]);
        assert!(matches!(file.matrix(test_governor(), "m", 3, 2), Err(SafetensorsError::Shape { .. })));
        assert!(matches!(file.vector("absent", 1), Err(SafetensorsError::Missing(_))));
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

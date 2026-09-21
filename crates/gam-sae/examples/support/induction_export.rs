//! The registry export of mpd-induction's benchmark, as the induction examples read it (#2951).
//!
//! `bench/mpd_induction_2951.py registry` writes `registry.json`, one little-endian `<f4` `.npy` per
//! stored tensor in its trained dtype, and a `<f8` `.npy` of the native logits. [`load_export`]
//! widens every tensor to binary64 (exact for every `f32`) and registers it in a
//! [`TensorRegistry`], with its aliases and every discovered use site.
//! `torch.nn.functional.linear` applies `x Wᵀ`, the identity orientation; an index or an addition
//! reads the stored values; any other op is refused. Like `npy_header.rs`, this is not an example
//! target: it is compiled only where it is `#[path]`-included.

use crate::npy_header::{NpyFloat, parse_npy_float_header, parse_npy_header};
use gam_sae::parameter_decomposition::lift::{TensorId, TensorRegistry, TieOrientation, UseMap, UseSiteId};
use ndarray::{Array2, ArrayD, IxDyn};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The fields of `registry.json` the examples read; the others are ignored.
#[derive(Deserialize)]
pub struct Export {
    pub stage: String,
    pub checkpoint: u64,
    pub trained_dtype: String,
    pub sequences: usize,
    pub tokens: Vec<Vec<i64>>,
    pub config: Config,
    pub tensors: Vec<Tensor>,
    pub use_sites: Vec<Use>,
    pub files: BTreeMap<String, ExportFile>,
}

#[derive(Deserialize)]
pub struct Config {
    pub vocab: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_head: usize,
}

#[derive(Deserialize)]
pub struct Tensor {
    pub tensor_id: String,
    pub aliases: Vec<String>,
    pub shape: Vec<usize>,
}

#[derive(Deserialize)]
pub struct Use {
    pub tensor_id: String,
    pub ordinal: usize,
    pub transposed: bool,
    pub op: String,
}

#[derive(Deserialize)]
pub struct ExportFile {
    pub path: String,
}

/// A registry export with its tensors widened to binary64 and registered.
pub struct LoadedExport {
    pub export: Export,
    pub registry: TensorRegistry,
    pub values: BTreeMap<String, ArrayD<f64>>,
}

impl LoadedExport {
    /// One stored matrix, by tensor id.
    pub fn weight(&self, id: &str) -> Result<Array2<f64>, String> {
        self.values
            .get(id)
            .ok_or_else(|| format!("the registry export holds no {id}"))?
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map(|view| view.to_owned())
            .map_err(|error| format!("{id}: {error}"))
    }
}

/// The bytes of one exported array, found by its file name inside `dir`, so an export can move.
fn read_array(
    files: &BTreeMap<String, ExportFile>,
    dir: &Path,
    array_id: &str,
) -> Result<(PathBuf, Vec<u8>), String> {
    let file = files
        .get(array_id)
        .ok_or_else(|| format!("the manifest names no file for {array_id}"))?;
    let name = Path::new(&file.path)
        .file_name()
        .ok_or_else(|| format!("{}: no file name", file.path))?;
    let path = dir.join(name);
    let bytes = std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok((path, bytes))
}

/// The data bytes of `count` elements of `width` bytes, refusing a truncated or padded file.
fn payload<'a>(
    bytes: &'a [u8],
    data_off: usize,
    count: usize,
    width: usize,
    path: &Path,
) -> Result<&'a [u8], String> {
    let end = count
        .checked_mul(width)
        .and_then(|size| size.checked_add(data_off))
        .ok_or_else(|| format!("{}: size overflow", path.display()))?;
    if end != bytes.len() {
        return Err(format!(
            "{}: {} bytes, expected {end}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(&bytes[data_off..end])
}

/// One stored tensor: a two-axis `<f4` array, widened to binary64.
fn stored_tensor(export: &Export, dir: &Path, tensor: &Tensor) -> Result<ArrayD<f64>, String> {
    let (path, bytes) = read_array(&export.files, dir, &tensor.tensor_id)?;
    let (rows, cols, width, is_f4, data_off) = parse_npy_header(&bytes, &path)?;
    if !is_f4 {
        return Err(format!("{}: expected the trained <f4 values", path.display()));
    }
    if [rows, cols][..] != tensor.shape[..] {
        return Err(format!(
            "{}: shape ({rows}, {cols}), but registry.json records {:?}",
            path.display(),
            tensor.shape
        ));
    }
    let values = payload(&bytes, data_off, rows * cols, width, &path)?
        .chunks_exact(width)
        .map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[rows, cols]), values)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// One executed array: a two-axis `<f8` array of the declared shape.
pub fn float64_array(
    files: &BTreeMap<String, ExportFile>,
    dir: &Path,
    array_id: &str,
    expected: (usize, usize),
) -> Result<Array2<f64>, String> {
    let (path, bytes) = read_array(files, dir, array_id)?;
    let header = parse_npy_float_header(&bytes, &path)?;
    if header.float != NpyFloat::F8 {
        return Err(format!(
            "{}: expected <f8 values from the binary64 executor",
            path.display()
        ));
    }
    let [rows, cols] = header.shape[..] else {
        return Err(format!(
            "{}: expected two axes, got {:?}",
            path.display(),
            header.shape
        ));
    };
    if (rows, cols) != expected {
        return Err(format!(
            "{}: shape ({rows}, {cols}), expected {expected:?}",
            path.display()
        ));
    }
    let values = payload(&bytes, header.data_off, rows * cols, header.float.bytes(), &path)?
        .chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// What a discovered use site does with its tensor, for the ops this export produces.
fn use_map(site: &Use) -> Result<UseMap, String> {
    match (site.op.as_str(), site.transposed) {
        ("torch.nn.functional.linear", false) => Ok(UseMap::Linear(TieOrientation::Identity)),
        ("torch.Tensor.__getitem__" | "torch.Tensor.add", false) => Ok(UseMap::Stored),
        (op, transposed) => Err(format!(
            "use site {}#{}: no use map is declared for op {op} (transposed: {transposed})",
            site.tensor_id, site.ordinal
        )),
    }
}

/// Reads `registry.json` in `dir`, refusing an export of another stage, dtype or token layout,
/// and registers every tensor, alias and use site.
pub fn load_export(dir: &Path) -> Result<LoadedExport, String> {
    let text = std::fs::read_to_string(dir.join("registry.json"))
        .map_err(|error| format!("read registry.json in {}: {error}", dir.display()))?;
    let export: Export = serde_json::from_str(&text).map_err(|error| format!("registry.json: {error}"))?;
    if export.stage != "registry" {
        return Err(format!("stage {:?}; expected registry", export.stage));
    }
    if export.trained_dtype != "float32" {
        return Err(format!(
            "trained dtype {:?}; the tensors are read as <f4",
            export.trained_dtype
        ));
    }
    if export.tokens.len() != export.sequences
        || export.tokens.iter().any(|row| row.len() != export.config.seq_len)
    {
        return Err("registry.json's token rows are not sequences x T".to_string());
    }
    let mut registry = TensorRegistry::default();
    let mut values = BTreeMap::new();
    for tensor in &export.tensors {
        let stored = stored_tensor(&export, dir, tensor)?;
        registry
            .register_storage(TensorId(tensor.tensor_id.clone()), stored.view())
            .map_err(|error| error.to_string())?;
        values.insert(tensor.tensor_id.clone(), stored);
    }
    for tensor in &export.tensors {
        for alias in &tensor.aliases {
            registry
                .register_alias(TensorId(alias.clone()), TensorId(tensor.tensor_id.clone()))
                .map_err(|error| error.to_string())?;
        }
    }
    for site in &export.use_sites {
        let storage = TensorId(site.tensor_id.clone());
        registry
            .register_use_site(UseSiteId::read(&storage, site.ordinal), storage, use_map(site)?)
            .map_err(|error| error.to_string())?;
    }
    Ok(LoadedExport {
        export,
        registry,
        values,
    })
}

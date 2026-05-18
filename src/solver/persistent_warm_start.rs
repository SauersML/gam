use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_VERSION: u32 = 1;
const MAX_ENTRY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
const MAX_ENTRIES: usize = 4096;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PersistentWarmStartRecord {
    pub version: u32,
    pub key: String,
    pub package_version: String,
    pub created_unix_secs: u64,
    pub updated_unix_secs: u64,
    pub n_rows: usize,
    pub n_cols: usize,
    pub rho: Vec<f64>,
    pub beta: Vec<f64>,
    pub prev_rho: Option<Vec<f64>>,
    pub prev_beta: Option<Vec<f64>>,
    pub last_inner_iters: usize,
    pub last_inner_converged: bool,
    pub last_pirls_lm_lambda: Option<f64>,
    pub last_ift_prediction_residual: Option<f64>,
    pub last_pirls_accept_rho: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PersistentBlockWarmStartRecord {
    pub version: u32,
    pub key: String,
    pub package_version: String,
    pub created_unix_secs: u64,
    pub updated_unix_secs: u64,
    pub n_rows: usize,
    pub block_names: Vec<String>,
    pub block_dims: Vec<usize>,
    pub rho: Vec<f64>,
    pub block_beta: Vec<Vec<f64>>,
    pub active_sets: Vec<Option<Vec<usize>>>,
}

impl PersistentBlockWarmStartRecord {
    pub(crate) fn new(
        key: String,
        n_rows: usize,
        block_names: Vec<String>,
        block_dims: Vec<usize>,
    ) -> Self {
        let now = unix_secs_now();
        Self {
            version: CACHE_VERSION,
            key,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            created_unix_secs: now,
            updated_unix_secs: now,
            n_rows,
            block_names,
            block_dims,
            rho: Vec::new(),
            block_beta: Vec::new(),
            active_sets: Vec::new(),
        }
    }

    pub(crate) fn is_compatible(
        &self,
        key: &str,
        n_rows: usize,
        block_names: &[String],
        block_dims: &[usize],
        rho_len: usize,
    ) -> bool {
        self.version == CACHE_VERSION
            && self.key == key
            && self.package_version == env!("CARGO_PKG_VERSION")
            && self.n_rows == n_rows
            && self.block_names == block_names
            && self.block_dims == block_dims
            && self.rho.len() == rho_len
            && self.rho.iter().all(|v| v.is_finite())
            && self.block_beta.len() == block_dims.len()
            && self
                .block_beta
                .iter()
                .zip(block_dims.iter())
                .all(|(beta, dim)| beta.len() == *dim && beta.iter().all(|v| v.is_finite()))
            && self.active_sets.len() == block_dims.len()
    }
}

impl PersistentWarmStartRecord {
    pub(crate) fn new(key: String, n_rows: usize, n_cols: usize) -> Self {
        let now = unix_secs_now();
        Self {
            version: CACHE_VERSION,
            key,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            created_unix_secs: now,
            updated_unix_secs: now,
            n_rows,
            n_cols,
            rho: Vec::new(),
            beta: Vec::new(),
            prev_rho: None,
            prev_beta: None,
            last_inner_iters: 0,
            last_inner_converged: false,
            last_pirls_lm_lambda: None,
            last_ift_prediction_residual: None,
            last_pirls_accept_rho: None,
        }
    }

    pub(crate) fn is_compatible(&self, key: &str, n_rows: usize, n_cols: usize) -> bool {
        self.version == CACHE_VERSION
            && self.key == key
            && self.package_version == env!("CARGO_PKG_VERSION")
            && self.n_rows == n_rows
            && self.n_cols == n_cols
            && self.rho.iter().all(|v| v.is_finite())
            && self.beta.len() == n_cols
            && self.beta.iter().all(|v| v.is_finite())
            && self
                .prev_rho
                .as_ref()
                .is_none_or(|rho| rho.len() == self.rho.len() && rho.iter().all(|v| v.is_finite()))
            && self
                .prev_beta
                .as_ref()
                .is_none_or(|beta| beta.len() == n_cols && beta.iter().all(|v| v.is_finite()))
    }
}

pub(crate) struct StableHasher {
    state: u64,
}

impl StableHasher {
    pub(crate) fn new() -> Self {
        Self {
            state: 0xcbf2_9ce4_8422_2325,
        }
    }

    pub(crate) fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub(crate) fn write_str(&mut self, value: &str) {
        self.write_usize(value.len());
        self.write_bytes(value.as_bytes());
    }

    pub(crate) fn write_usize(&mut self, value: usize) {
        self.write_bytes(&(value as u64).to_le_bytes());
    }

    pub(crate) fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub(crate) fn write_bool(&mut self, value: bool) {
        self.write_bytes(&[u8::from(value)]);
    }

    pub(crate) fn write_f64(&mut self, value: f64) {
        let normalized = if value == 0.0 { 0.0 } else { value };
        self.write_u64(normalized.to_bits());
    }

    pub(crate) fn finish_hex(&self) -> String {
        format!("{:016x}", self.state)
    }
}

pub(crate) fn load_record(key: &str) -> Option<PersistentWarmStartRecord> {
    let path = cache_file_path(key)?;
    let metadata = fs::metadata(&path).ok()?;
    if metadata.len() > MAX_ENTRY_BYTES {
        let _ = fs::remove_file(path);
        return None;
    }
    let mut file = File::open(&path).ok()?;
    let mut raw = String::with_capacity(metadata.len() as usize);
    file.read_to_string(&mut raw).ok()?;
    match serde_json::from_str::<PersistentWarmStartRecord>(&raw) {
        Ok(record) => Some(record),
        Err(_) => {
            let _ = fs::remove_file(path);
            None
        }
    }
}

pub(crate) fn load_block_record(key: &str) -> Option<PersistentBlockWarmStartRecord> {
    let path = cache_file_path(key)?;
    let metadata = fs::metadata(&path).ok()?;
    if metadata.len() > MAX_ENTRY_BYTES {
        let _ = fs::remove_file(path);
        return None;
    }
    let mut file = File::open(&path).ok()?;
    let mut raw = String::with_capacity(metadata.len() as usize);
    file.read_to_string(&mut raw).ok()?;
    match serde_json::from_str::<PersistentBlockWarmStartRecord>(&raw) {
        Ok(record) => Some(record),
        Err(_) => {
            let _ = fs::remove_file(path);
            None
        }
    }
}

pub(crate) fn store_record(record: &PersistentWarmStartRecord) -> Result<(), String> {
    store_json_record(&record.key, record)
}

pub(crate) fn store_block_record(record: &PersistentBlockWarmStartRecord) -> Result<(), String> {
    store_json_record(&record.key, record)
}

fn store_json_record<T: Serialize>(key: &str, record: &T) -> Result<(), String> {
    let dir = cache_dir()?;
    fs::create_dir_all(&dir).map_err(|e| {
        format!(
            "failed to create warm-start cache dir '{}': {e}",
            dir.display()
        )
    })?;
    let path = cache_file_path_in_dir(&dir, key);
    let bytes = serde_json::to_vec(record)
        .map_err(|e| format!("failed to encode warm-start cache record: {e}"))?;
    if bytes.len() as u64 > MAX_ENTRY_BYTES {
        return Ok(());
    }

    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        key,
        std::process::id() as u64 ^ unix_nanos_now()
    ));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| {
                format!(
                    "failed to create warm-start cache temp file '{}': {e}",
                    tmp.display()
                )
            })?;
        file.write_all(&bytes).map_err(|e| {
            format!(
                "failed to write warm-start cache temp file '{}': {e}",
                tmp.display()
            )
        })?;
        file.sync_all().map_err(|e| {
            format!(
                "failed to sync warm-start cache temp file '{}': {e}",
                tmp.display()
            )
        })?;
    }
    fs::rename(&tmp, &path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!(
            "failed to install warm-start cache record '{}': {e}",
            path.display()
        )
    })?;
    sync_dir_best_effort(&dir);
    prune_cache_dir(&dir);
    Ok(())
}

fn cache_file_path(key: &str) -> Option<PathBuf> {
    cache_dir()
        .ok()
        .map(|dir| cache_file_path_in_dir(&dir, key))
}

fn cache_file_path_in_dir(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.json"))
}

fn cache_dir() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir()
        .map_err(|e| format!("failed to resolve current directory for warm-start cache: {e}"))?;
    Ok(cwd
        .join(".cache")
        .join("gamfit")
        .join("warm-start")
        .join("v1"))
}

fn unix_secs_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn unix_nanos_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn sync_dir_best_effort(dir: &Path) {
    if let Ok(file) = File::open(dir) {
        let _ = file.sync_all();
    }
}

fn prune_cache_dir(dir: &Path) {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return;
    };
    let mut entries = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            if path.extension().and_then(|s| s.to_str()) == Some("tmp") {
                let _ = fs::remove_file(path);
            }
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        entries.push((path, metadata.len(), modified));
    }
    let mut total: u64 = entries.iter().map(|(_, len, _)| *len).sum();
    if total <= MAX_TOTAL_BYTES && entries.len() <= MAX_ENTRIES {
        return;
    }
    entries.sort_by_key(|(_, _, modified)| *modified);
    for (path, len, _) in entries {
        if total <= MAX_TOTAL_BYTES && cache_entry_count(dir) <= MAX_ENTRIES {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
        }
    }
}

fn cache_entry_count(dir: &Path) -> usize {
    fs::read_dir(dir)
        .ok()
        .map(|it| {
            it.flatten()
                .filter(|entry| entry.path().extension().and_then(|s| s.to_str()) == Some("json"))
                .count()
        })
        .unwrap_or(0)
}

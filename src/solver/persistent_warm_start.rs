use crate::cache::{EntryKind, Fingerprinter, StoreOptions, WarmStartStore};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_VERSION: u32 = 1;
const MAX_ENTRY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
const CACHE_TTL_SECS: u64 = 60 * 60 * 24 * 365 * 10;

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
    load_json_record(key)
}

pub(crate) fn load_block_record(key: &str) -> Option<PersistentBlockWarmStartRecord> {
    load_json_record(key)
}

pub(crate) fn store_record(record: &PersistentWarmStartRecord) -> Result<(), String> {
    store_json_record(&record.key, record)
}

pub(crate) fn store_block_record(record: &PersistentBlockWarmStartRecord) -> Result<(), String> {
    store_json_record(&record.key, record)
}

fn store_json_record<T: Serialize>(key: &str, record: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec(record)
        .map_err(|e| format!("failed to encode warm-start cache record: {e}"))?;
    if bytes.len() as u64 > MAX_ENTRY_BYTES {
        return Ok(());
    }
    let Some(store) = persistent_store() else {
        return Ok(());
    };
    store
        .save(
            &fingerprint_for_key(key),
            &bytes,
            None,
            None,
            EntryKind::Checkpoint,
        )
        .map(|_| ())
        .map_err(|e| format!("failed to persist warm-start cache record: {e}"))
}

fn load_json_record<T: for<'de> Deserialize<'de>>(key: &str) -> Option<T> {
    let store = persistent_store()?;
    let entry = store.lookup(&fingerprint_for_key(key)).ok().flatten()?;
    if entry.payload.len() as u64 > MAX_ENTRY_BYTES {
        return None;
    }
    serde_json::from_slice(&entry.payload).ok()
}

fn persistent_store() -> Option<WarmStartStore> {
    let base = dirs::cache_dir()?;
    let root = base.join("gam").join("warm").join("v1");
    WarmStartStore::open(
        root,
        StoreOptions {
            size_budget_bytes: MAX_TOTAL_BYTES,
            ttl: Duration::from_secs(CACHE_TTL_SECS),
        },
    )
    .ok()
}

fn fingerprint_for_key(key: &str) -> crate::cache::Fingerprint {
    let mut fp = Fingerprinter::new();
    fp.absorb_str(b"warm-start-key", key);
    fp.finalize()
}

/// Open a [`crate::cache::Session`] for outer-iterate (rho-axis) checkpoints.
///
/// Uses a different fingerprint tag than [`fingerprint_for_key`] so the
/// outer-iterate keyspace is disjoint from the inner beta-record keyspace —
/// the two layers persist different payload shapes and must not alias.
pub(crate) fn open_outer_session(key: &str) -> Option<std::sync::Arc<crate::cache::Session>> {
    let store = persistent_store()?;
    let mut fp = Fingerprinter::new();
    fp.absorb_str(b"outer-iterate-key", key);
    let fp = fp.finalize();
    Some(std::sync::Arc::new(crate::cache::Session::open(store, fp)))
}

fn unix_secs_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

use gam_runtime::warm_start::{EntryKind, Fingerprinter, StoreOptions, WarmStartStore};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_VERSION: u32 = 1;
const MAX_ENTRY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
const CACHE_TTL_SECS: u64 = 60 * 60 * 24 * 365 * 10;

/// String tag identifying the on-disk cache schema, embedded directly in
/// cache keys.
///
/// The leading `schema2-` prefix is bumped manually only when the
/// serialized cache layout changes in a way that makes prior entries
/// unsafe to consume (struct fields added/removed, optimization
/// invariants altered, payload semantics shift). This is **deliberately
/// separate** from `CARGO_PKG_VERSION` so a routine library version bump
/// does NOT invalidate every user's warm-start cache.
pub fn cache_schema_tag() -> String {
    // Bumped from `schema2-` → `schema3-` when the three hand-written
    // hashers (`Fingerprinter`, `StableHasher`, `CacheDigestBuilder`) were
    // unified onto `Fingerprinter`. Prior on-disk warm-start entries are
    // walled off into the `schema2-` keyspace and cold-start once; this
    // is the intentional consequence of the unification, documented in
    // the commit that performs it. See `src/warm_start/key.rs` for the new
    // canonical hasher API.
    // Bumped to `v2` when the persistent warm-start key stopped hashing the
    // θ-dependent, lazily-refreshed isometry Jacobian cache slots
    // (`jacobian_cache` / `jacobian_second_cache` / `third_decoder_derivative`).
    // Those snapshots made the key non-reproducible across identical repeat
    // fits, so the outer `skip-outer-validation` warm hit was lost (#1048). The
    // bump walls off any entries written under the old, drifting keys; they are
    // simply never matched (and TTL-evicted) rather than aliasing.
    // Bumped to `v3` when the descriptor-indexed cross-fit `FitArtifact`
    // keyspace ("fit-artifact-key") was introduced alongside the existing
    // inner/outer warm-start records. The bump walls off any entries written
    // under the old layouts so a mixed-schema store never aliases a legacy
    // payload into the new artifact reader (and vice versa).
    "schema3-unified-fingerprinter-v3".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistentWarmStartRecord {
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
pub struct PersistentBlockInnerSummary {
    pub log_likelihood: f64,
    pub penalty_value: f64,
    pub cycles: usize,
    pub converged: bool,
    pub block_logdet_h: f64,
    pub block_logdet_s: f64,
}

impl PersistentBlockInnerSummary {
    fn is_valid(&self) -> bool {
        self.log_likelihood.is_finite()
            && self.penalty_value.is_finite()
            && self.block_logdet_h.is_finite()
            && self.block_logdet_s.is_finite()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistentBlockWarmStartRecord {
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
    #[serde(default)]
    pub inner: Option<PersistentBlockInnerSummary>,
}

impl PersistentBlockWarmStartRecord {
    pub fn new(
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
            inner: None,
        }
    }

    pub fn is_compatible(
        &self,
        key: &str,
        n_rows: usize,
        block_names: &[String],
        block_dims: &[usize],
        rho_len: usize,
    ) -> bool {
        self.version == CACHE_VERSION
            && self.key == key
            // Note: `package_version` is no longer required to match. A
            // library version bump that doesn't change the cache schema
            // (the common case for patch / minor releases) should NOT
            // invalidate users' on-disk warm-start caches. Schema-breaking
            // changes bump the `schemaN-` prefix in `cache_schema_tag()`,
            // which is encoded in the cache key itself.
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
            && self.inner.as_ref().is_none_or(|inner| inner.is_valid())
    }
}

impl PersistentWarmStartRecord {
    pub fn new(key: String, n_rows: usize, n_cols: usize) -> Self {
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

    pub fn is_compatible(&self, key: &str, n_rows: usize, n_cols: usize) -> bool {
        self.version == CACHE_VERSION
            && self.key == key
            // Note: `package_version` is no longer required to match. A
            // library version bump that doesn't change the cache schema
            // (the common case for patch / minor releases) should NOT
            // invalidate users' on-disk warm-start caches. Schema-breaking
            // changes bump the `schemaN-` prefix in `cache_schema_tag()`,
            // which is encoded in the cache key itself.
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

pub fn load_record(key: &str) -> Option<PersistentWarmStartRecord> {
    load_json_record(key)
}

pub fn load_block_record(key: &str) -> Option<PersistentBlockWarmStartRecord> {
    load_json_record(key)
}

pub fn store_record(record: &PersistentWarmStartRecord) -> Result<(), String> {
    store_json_record(&record.key, record)
}

pub fn store_block_record(record: &PersistentBlockWarmStartRecord) -> Result<(), String> {
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
    let mut fp = Fingerprinter::new();
    fp.absorb_str(b"warm-start-key", key);
    store
        .save(&fp.finalize(), &bytes, None, None, EntryKind::Checkpoint)
        .map(|_| ())
        .map_err(|e| format!("failed to persist warm-start cache record: {e}"))
}

fn load_json_record<T: for<'de> Deserialize<'de>>(key: &str) -> Option<T> {
    let store = persistent_store()?;
    let mut fp = Fingerprinter::new();
    fp.absorb_str(b"warm-start-key", key);
    let entry = store.lookup(&fp.finalize()).ok().flatten()?;
    if entry.payload.len() as u64 > MAX_ENTRY_BYTES {
        return None;
    }
    serde_json::from_slice(&entry.payload).ok()
}

/// Anchor the warm-start cache under the platform temp directory.
///
/// Reading `XDG_CACHE_HOME` / `HOME` / `LOCALAPPDATA` (the canonical
/// `dirs::cache_dir()` fallbacks) requires `env::var_os`, which is banned
/// in this crate (see the build-script tripwire scan and the
/// `feedback_no_env_vars` policy memo). `std::env::temp_dir()` resolves
/// platform-conventional locations through OS-level primitives without
/// going through `env::var`, so we route the persistent warm-start
/// checkpoint root there instead. The directory is durable across
/// processes within a single boot, and `WarmStartStore::open` falls back
/// to `None` if the path is unwritable.
fn persistent_store() -> Option<WarmStartStore> {
    // Memoize the store process-wide. The root (`temp_dir()/gam/warm/v1`) is
    // constant within a process, so a single instance suffices — and reusing
    // it is essential, not just an optimization: `WarmStartStore` carries the
    // per-store directory-scan / metadata cache and the eviction-throttle
    // counters that #1114 added. Reconstructing the store on every save/lookup
    // (as this used to) handed each fit an empty cache and a zeroed throttle,
    // so every operation re-walked the cache root and re-read every metadata
    // JSON from disk — the syscall storm that made several quality tests look
    // hung. Clones returned here share the cache and throttle via `Arc`.
    static STORE: OnceLock<Option<WarmStartStore>> = OnceLock::new();
    STORE
        .get_or_init(|| {
            let root = std::env::temp_dir().join("gam").join("warm").join("v1");
            WarmStartStore::open(
                root,
                StoreOptions {
                    size_budget_bytes: MAX_TOTAL_BYTES,
                    ttl: Duration::from_secs(CACHE_TTL_SECS),
                },
            )
            .ok()
        })
        .clone()
}

/// Open a [`gam_runtime::warm_start::Session`] for outer-iterate (rho-axis) checkpoints.
///
/// Uses a different fingerprint tag than the inner `warm-start-key`
/// absorption (see [`load_json_record`]) so the outer-iterate keyspace
/// is disjoint from the inner beta-record keyspace —
/// the two layers persist different payload shapes and must not alias.
pub(crate) fn open_outer_session(
    key: &str,
) -> Option<std::sync::Arc<gam_runtime::warm_start::Session>> {
    let store = persistent_store()?;
    let mut fp = Fingerprinter::new();
    fp.absorb_str(b"outer-iterate-key", key);
    let fp = fp.finalize();
    Some(std::sync::Arc::new(gam_runtime::warm_start::Session::open(
        store, fp,
    )))
}

/// Persist a descriptor-indexed cross-fit [`FitArtifact`] under the
/// `fit-artifact-key` keyspace, keyed by the descriptor's structural key (so
/// an LOSO fold of the same model retrieves a prior full-data fit). The
/// schema tag is folded into the key so legacy layouts are walled off.
///
/// Best-effort: encoding / store failures are swallowed (a warm-start
/// artifact is never required), oversize payloads are dropped.
pub fn store_fit_artifact(
    artifact: &crate::warm_start_artifact::FitArtifact,
) -> Result<(), String> {
    if !artifact.is_usable() {
        // Never persist a non-finite / wrong-schema artifact: it could only
        // ever be rejected on load anyway.
        return Ok(());
    }
    let bytes = serde_json::to_vec(artifact)
        .map_err(|e| format!("failed to encode fit-artifact record: {e}"))?;
    if bytes.len() as u64 > MAX_ENTRY_BYTES {
        return Ok(());
    }
    let Some(store) = persistent_store() else {
        return Ok(());
    };
    let key = artifact.descriptor.descriptor_key().to_hex();
    let mut fp = Fingerprinter::new();
    fp.absorb_str(b"fit-artifact-key", &cache_schema_tag());
    fp.absorb_str(b"fit-artifact-descriptor", &key);
    store
        .save(&fp.finalize(), &bytes, None, None, EntryKind::Checkpoint)
        .map(|_| ())
        .map_err(|e| format!("failed to persist fit-artifact record: {e}"))
}

/// Load the newest valid cross-fit [`FitArtifact`] whose descriptor key
/// matches `descriptor_key_hex` (the hex of [`crate::warm_start_artifact::FitDescriptor::descriptor_key`]).
///
/// Uses `lookup_latest` (newest-valid) rather than objective-ranked lookup:
/// descriptor-key matches can be different folds / row sets whose objectives
/// are not on a common scale, so "lowest objective" is the wrong rule.
/// Returns `None` (cold fallback) on any miss or non-finite payload.
pub fn load_fit_artifact_by_descriptor(
    descriptor_key_hex: &str,
) -> Option<crate::warm_start_artifact::FitArtifact> {
    let store = persistent_store()?;
    let mut fp = Fingerprinter::new();
    fp.absorb_str(b"fit-artifact-key", &cache_schema_tag());
    fp.absorb_str(b"fit-artifact-descriptor", descriptor_key_hex);
    let entry = store.lookup_latest(&fp.finalize()).ok().flatten()?;
    if entry.payload.len() as u64 > MAX_ENTRY_BYTES {
        return None;
    }
    let artifact: crate::warm_start_artifact::FitArtifact =
        serde_json::from_slice(&entry.payload).ok()?;
    // Finite-guard on the way out: a corrupt payload must cold-fallback,
    // never poison a fit.
    artifact.is_usable().then_some(artifact)
}

fn unix_secs_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod warm_start_artifact_tests {
    use super::*;
    use crate::warm_start_artifact::{
        FIT_ARTIFACT_SCHEMA, FitArtifact, FitDescriptor, GlobalFitSummary, ResponseSig,
        SerializableBasisMeta, TermArtifact, TermRole, term_identity_from_block,
    };

    fn sample_artifact(family: &str, var: &str, rho: Vec<f64>) -> FitArtifact {
        // Block-layer identity (the surviving, fold-invariant identity API):
        // the block name carries the variable, with one unlabeled penalty.
        let block_name = format!("s({var})");
        let id = term_identity_from_block(TermRole::Mean, &block_name, &[None], &[1], 10);
        FitArtifact {
            schema: FIT_ARTIFACT_SCHEMA,
            created_unix_secs: unix_secs_now(),
            descriptor: FitDescriptor {
                family_kind: family.to_string(),
                term_identities: vec![id],
                response_signature: ResponseSig {
                    family_kind: family.to_string(),
                    n_response_channels: 1,
                },
                row_population: None,
            },
            terms: vec![TermArtifact {
                identity: id,
                role: TermRole::Mean,
                basis_meta: SerializableBasisMeta {
                    kind: "block-spec".to_string(),
                    degree: None,
                    num_knots: None,
                    n_centers: Some(8),
                    nullspace_order: None,
                    matern_nu: None,
                    periodic: false,
                },
                joint_null_rotation: None,
                raw_beta: vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
                rho_for_term: rho,
            }],
            global: GlobalFitSummary {
                outer_objective: -42.0,
                converged: true,
                n_rows: 500,
            },
        }
    }

    #[test]
    fn artifact_round_trips_on_disk_by_descriptor() {
        // Use a unique family-kind tag so this test's descriptor key is
        // disjoint from any other run's keyspace (the store is process-shared
        // under the temp dir).
        let family = format!("test-roundtrip-{}", unix_secs_now());
        let artifact = sample_artifact(&family, "x", vec![2.5]);
        let key_hex = artifact.descriptor.descriptor_key().to_hex();

        // If the platform temp dir is unwritable, the store is None and the
        // round-trip is a no-op; only assert when persistence is available.
        if persistent_store().is_none() {
            return;
        }
        store_fit_artifact(&artifact).expect("store fit artifact");
        let loaded = load_fit_artifact_by_descriptor(&key_hex)
            .expect("artifact must be retrievable by descriptor key");
        assert_eq!(loaded.schema, artifact.schema);
        assert_eq!(loaded.terms.len(), 1);
        assert_eq!(loaded.terms[0].identity, artifact.terms[0].identity);
        assert_eq!(loaded.terms[0].rho_for_term, vec![2.5]);
        assert_eq!(loaded.terms[0].raw_beta, artifact.terms[0].raw_beta);
        assert_eq!(
            loaded.descriptor.descriptor_key(),
            artifact.descriptor.descriptor_key()
        );
    }

    #[test]
    fn loso_fold_descriptor_matches_full_data_artifact() {
        let family = format!("test-loso-{}", unix_secs_now());
        // Full-data fit on 1000 rows.
        let mut full = sample_artifact(&family, "x", vec![1.7]);
        full.descriptor.row_population =
            Some(crate::warm_start_artifact::RowPopulationTag {
                n_rows: 1000,
                label: Some("full".to_string()),
            });
        full.global.n_rows = 1000;
        let full_key = full.descriptor.descriptor_key().to_hex();

        // LOSO fold: same term identities, fewer rows. Its descriptor key
        // must equal the full-data key, so the load hits the stored artifact.
        let fold = sample_artifact(&family, "x", vec![1.7]);
        let fold_key = fold.descriptor.descriptor_key().to_hex();
        assert_eq!(
            full_key, fold_key,
            "fold and full descriptor keys must match"
        );

        if persistent_store().is_none() {
            return;
        }
        store_fit_artifact(&full).expect("store full-data artifact");
        let loaded = load_fit_artifact_by_descriptor(&fold_key)
            .expect("LOSO fold must retrieve the full-data artifact");
        assert_eq!(loaded.terms[0].rho_for_term, vec![1.7]);
    }
}

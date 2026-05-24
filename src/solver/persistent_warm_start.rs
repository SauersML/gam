use crate::cache::{EntryKind, Fingerprinter, StoreOptions, WarmStartStore};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_VERSION: u32 = 1;
/// On-disk cache schema version.
///
/// Bumped manually only when the serialized cache layout changes in a
/// way that makes prior entries unsafe to consume (struct fields added/
/// removed, optimization invariants altered, payload semantics shift).
///
/// This is **deliberately separate** from `CARGO_PKG_VERSION` so a
/// routine library version bump does NOT invalidate every user's
/// warm-start cache. A user upgrading from 0.1.X to 0.1.(X+1) should
/// see their existing checkpoints still load; only a deliberate schema
/// change (e.g., reshaping how ρ is encoded) bumps this constant.
pub(crate) const CACHE_SCHEMA_VERSION: u32 = 2;
const MAX_ENTRY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
const CACHE_TTL_SECS: u64 = 60 * 60 * 24 * 365 * 10;

/// String form of [`CACHE_SCHEMA_VERSION`] for direct use in cache keys.
pub(crate) fn cache_schema_tag() -> String {
    format!("schema{CACHE_SCHEMA_VERSION}-arrow-schur-streaming-v1")
}

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
pub(crate) struct PersistentBlockInnerSummary {
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
    #[serde(default)]
    pub inner: Option<PersistentBlockInnerSummary>,
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
            inner: None,
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
            // Note: `package_version` is no longer required to match. A
            // library version bump that doesn't change the cache schema
            // (the common case for patch / minor releases) should NOT
            // invalidate users' on-disk warm-start caches. Schema-breaking
            // changes bump `CACHE_SCHEMA_VERSION` which is encoded in
            // the cache key itself.
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
            // Note: `package_version` is no longer required to match. A
            // library version bump that doesn't change the cache schema
            // (the common case for patch / minor releases) should NOT
            // invalidate users' on-disk warm-start caches. Schema-breaking
            // changes bump `CACHE_SCHEMA_VERSION` which is encoded in
            // the cache key itself.
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

    pub(crate) fn finish_u64(&self) -> u64 {
        self.state
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
    let root = std::env::temp_dir().join("gam").join("warm").join("v1");
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

/// Look up a single outer-iterate payload by string key without opening
/// a long-lived session.
///
/// Used by the workflow dispatcher to fetch a *seed* from a near-match
/// (data-independent) key. The session's own [`crate::cache::Session::preload`]
/// stashes the returned entry so the next [`crate::cache::Session::try_load`]
/// returns it ahead of the (empty) exact-key store lookup. Save writes
/// always go to the session's own key — the prefix lookup is read-only.
pub(crate) fn lookup_outer_iterate_payload(seed_key: &str) -> Option<crate::cache::CachedEntry> {
    let store = persistent_store()?;
    let mut fp = Fingerprinter::new();
    fp.absorb_str(b"outer-iterate-key", seed_key);
    // Seed-prefix entries can represent related but non-identical fits
    // (different folds, diseases, or row sets). Their objectives are not on a
    // common scale, so "lowest objective" is the wrong selection rule here.
    // Prefer the newest valid seed; exact-key resume still uses objective-
    // ranked lookup through Session::try_load().
    store.lookup_latest(&fp.finalize()).ok().flatten()
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

// ---------------------------------------------------------------------------
// Piece 3 — IFT warm-starting of the latent field
//
// See `proposals/latent_coord.md` §2 / §4 and
// `proposals/composition_engine.md` §6 (the ψ-machinery mapping). The
// existing β-IFT predictor lives in
// `crate::solver::reml::runtime::predict_warm_start_beta_ift_with_outcome`
// and operates on the dense `H_pen` Cholesky factor cached after the
// inner solve. The latent IFT predictor below is the row-local analogue
// for the arrow-structured (t, β) system: per-row `H_tt^(i)` factors
// already produced by
// [`crate::solver::arrow_schur::solve_arrow_newton_step`] are reused to
// apply the per-row sensitivity in O(d²) ops per row, independent of K.
//
// Math recap (proposal §2.3 + composition_engine §7):
//
//   At convergence, g_t^(i)(t̂_i, β̂, ρ) = 0  for every row i.
//   Differentiating implicitly w.r.t. β:
//       H_tt^(i) · ∂t̂_i/∂β = -H_tβ^(i)        (the cross-block).
//   Hence for a candidate β-shift Δβ:
//       Δt_i ≈ -(H_tt^(i))⁻¹ · (H_tβ^(i) · Δβ).
//
//   For a ρ-shift (or any external hyper-axis η) the driver supplies
//   the per-row partial g_t^(i) shift δg_t^(i); the predictor returns
//       Δt_i ≈ -(H_tt^(i))⁻¹ · δg_t^(i).
//
// Both forms reuse the per-row Cholesky factor cached at the last
// accepted Newton step — no per-row re-factor, no `Schur⁻¹` formation,
// O(N · d²) total apply cost.
// ---------------------------------------------------------------------------

use crate::solver::arrow_schur::ArrowFactorCache;
use ndarray::{Array1, ArrayView1};

/// Outcome of an [`ift_warm_start_latent`] call.
#[derive(Debug, Clone)]
pub enum LatentIftOutcome {
    /// Predictor applied a finite shift; `delta_t` is the flat row-major
    /// increment of length `N · d` (sum of the β-coupled and ρ-coupled
    /// contributions).
    Applied { delta_t: Array1<f64> },
    /// Predictor declined (no cache, dim mismatch, or non-finite input).
    /// Caller should fall back to a from-scratch latent inner solve.
    Noop,
}

/// Predict the analytic shift in the latent field `t̂` induced by a
/// shape-coefficient change `Δβ` and/or an external `δg_t` shift
/// (typically coming from a ρ perturbation that the driver has already
/// resolved into per-row gradient shifts via the analytic-penalty
/// registry).
///
/// This is the row-local analogue of
/// `RemlState::predict_warm_start_beta_ift_with_outcome`: it consumes
/// the per-row factor cache produced by
/// [`crate::solver::arrow_schur::solve_arrow_newton_step`] at the last
/// accepted Newton step and applies the IFT first-order predictor
///
/// ```text
///   Δt_i  =  -(H_tt^(i))⁻¹ · (H_tβ^(i) Δβ + δg_t^(i))
/// ```
///
/// per row. Both sources can be omitted (`None`) — at least one must be
/// supplied for the predictor to do useful work.
///
/// # Damping invariant
///
/// The IFT formula inverts the *undamped* inner Hessian `H_tt^(i)` — the LM
/// ridge `ρ_t·I` is part of the Newton damping, not the implicit-function
/// theorem derivation. The cache produced by
/// [`crate::solver::arrow_schur::solve_arrow_newton_step`] stores both the
/// damped Newton factor and the undamped IFT factor; this predictor
/// dispatches into `predict_delta_t_from_delta_*` which select the latter.
/// Callers MUST NOT hand-construct an `ArrowFactorCache` with missing
/// undamped factors.
///
/// The result is meant to be applied to the existing
/// [`crate::terms::latent_coord::LatentCoordValues`] block via
/// `retract_flat_delta(Δt)` before the next inner Newton
/// launch — so the inner solver starts at a configuration already
/// consistent with the new `(β, ρ)`, requiring far fewer iterations
/// than a cold restart.
///
/// # Integration with [`crate::solver::latent_inner::LatentInnerSolver`]
///
/// The intended pipeline is:
///
///   1. inner solve at `(β₀, ρ₀)` produces
///      [`crate::solver::latent_inner::LatentInnerOutcome::factor_cache`];
///   2. REML outer loop proposes `(β₁, ρ₁)`;
///   3. caller invokes `ift_warm_start_latent(cache, Some(β₁ − β₀), δg_t)`
///      to obtain `Δt`;
///   4. caller updates the latent field by `Δt` and launches the next
///      inner Newton — typically converging in 1–2 iterations.
pub fn ift_warm_start_latent(
    cache: &ArrowFactorCache,
    delta_beta: Option<ArrayView1<'_, f64>>,
    delta_gt: Option<ArrayView1<'_, f64>>,
) -> LatentIftOutcome {
    let n = cache.n_rows();
    let d = cache.d;
    let k = cache.k;

    if delta_beta.is_none() && delta_gt.is_none() {
        return LatentIftOutcome::Noop;
    }
    if let Some(db) = delta_beta.as_ref() {
        if db.len() != k {
            return LatentIftOutcome::Noop;
        }
        if db.iter().any(|v| !v.is_finite()) {
            return LatentIftOutcome::Noop;
        }
    }
    if let Some(dg) = delta_gt.as_ref() {
        if dg.len() != n * d {
            return LatentIftOutcome::Noop;
        }
        if dg.iter().any(|v| !v.is_finite()) {
            return LatentIftOutcome::Noop;
        }
    }
    if cache.undamped_factor_count() != n {
        return LatentIftOutcome::Noop;
    }
    if delta_beta.is_some() && !cache.htbeta_available() {
        return LatentIftOutcome::Noop;
    }
    for factor in cache
        .htt_factors
        .iter()
        .chain(cache.undamped_factors_iter())
    {
        if factor.dim() != (d, d) {
            return LatentIftOutcome::Noop;
        }
    }
    // Sum the two contributions on the RHS, then per-row solve.
    let delta_t = cache.predict_delta_t_combined(delta_beta, delta_gt);
    LatentIftOutcome::Applied { delta_t }
}

fn unix_secs_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

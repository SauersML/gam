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
pub(crate) const CACHE_SCHEMA_VERSION: u32 = 1;
const MAX_ENTRY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
const CACHE_TTL_SECS: u64 = 60 * 60 * 24 * 365 * 10;

/// String form of [`CACHE_SCHEMA_VERSION`] for direct use in cache keys.
pub(crate) fn cache_schema_tag() -> String {
    format!("schema{CACHE_SCHEMA_VERSION}")
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
    /// `reason` is a stable string ID for diagnostics; the
    /// `#[allow(dead_code)]` is required because the crate-wide
    /// `deny(dead_code)` lint cannot see that this field is consumed
    /// only by opt-in verbose IFT logging (Piece 2 driver integration,
    /// not yet wired into the production REML loop).
    Noop {
        #[allow(dead_code)]
        reason: &'static str,
    },
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
/// Callers MUST NOT hand-construct an `ArrowFactorCache` that leaves
/// `htt_factors_undamped` empty — both arrays must be populated.
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
    let n = cache.htt_factors.len();
    let d = cache.d;
    let k = cache.k;

    if delta_beta.is_none() && delta_gt.is_none() {
        return LatentIftOutcome::Noop {
            reason: "no-perturbation: both delta_beta and delta_gt are None",
        };
    }
    if let Some(db) = delta_beta.as_ref() {
        if db.len() != k {
            return LatentIftOutcome::Noop {
                reason: "delta_beta length mismatch with cached K",
            };
        }
        if db.iter().any(|v| !v.is_finite()) {
            return LatentIftOutcome::Noop {
                reason: "delta_beta contains non-finite entries",
            };
        }
    }
    if let Some(dg) = delta_gt.as_ref() {
        if dg.len() != n * d {
            return LatentIftOutcome::Noop {
                reason: "delta_gt length mismatch with cached N·d",
            };
        }
        if dg.iter().any(|v| !v.is_finite()) {
            return LatentIftOutcome::Noop {
                reason: "delta_gt contains non-finite entries",
            };
        }
    }

    // Sum the two contributions on the RHS, then per-row solve.
    let mut delta_t = Array1::<f64>::zeros(n * d);
    if let Some(db) = delta_beta {
        let part = cache.predict_delta_t_from_delta_beta(db);
        for i in 0..delta_t.len() {
            delta_t[i] += part[i];
        }
    }
    if let Some(dg) = delta_gt {
        let part = cache.predict_delta_t_from_delta_gt(dg);
        for i in 0..delta_t.len() {
            delta_t[i] += part[i];
        }
    }
    LatentIftOutcome::Applied { delta_t }
}

// ---------------------------------------------------------------------------
// Riemannian retraction wrapper (additive)
// ---------------------------------------------------------------------------

/// Apply an IFT-predicted Euclidean tangent `delta_t` through per-row
/// retractions, returning the *new* latent field as a flat row-major array.
///
/// The predicted `δt_i` produced by [`ift_warm_start_latent`] is a tangent
/// vector at the current `t_i`. When the per-row latent coordinate lives on
/// a non-trivial manifold (S¹, S², torus, interval), the additive update
/// `t_i + δt_i` exits the manifold; we must apply the retraction
/// `R_{t_i}(P_{T_{t_i}}(δt_i))`.
///
/// For `manifolds[i] = None` or `Some(ManifoldKind::Euclidean(_))` the
/// behavior collapses to plain addition (bit-equivalent to the pre-Riemannian
/// path). The vector-transport-aware Taylor extension for small `Δβ`
/// reuses the parallel-transport approximation embedded in each
/// `Manifold::vector_transport` implementation — see
/// [`crate::solver::riemannian`].
#[allow(dead_code)]
pub fn apply_ift_retraction(
    point_flat: &Array1<f64>,
    delta_t: &Array1<f64>,
    row_ambient_dim: usize,
    manifolds: &[Option<crate::solver::riemannian::ManifoldKind>],
) -> Array1<f64> {
    crate::solver::arrow_schur::apply_per_row_retraction(
        point_flat,
        delta_t,
        row_ambient_dim,
        manifolds,
    )
}

// ---------------------------------------------------------------------------
// IFT cascade through (u, β, ρ) — materialize ∂(β*, u*)/∂ρ from the cache.
//
// Per `proposals/arrow_schur_evidence.md` §2.4 / §2.6 / §7:
//
//   ∂β*/∂ρ_a = -A⁻¹ q_a               (q_a = ∂g_red/∂ρ_a)
//   ∂u*/∂ρ_a = -H_uu⁻¹ G_{u,ρ_a} - H_uu⁻¹ H_uβ ∂β*/∂ρ_a.
//
// Both per-row solves use the UNDAMPED row factors
// (`htt_factors_undamped`), per proposal §1.7. The β solve uses the
// cached Schur Cholesky `schur_factor`. The math is equivalent to two
// applications of the existing predictors:
//
//   ∂u*/∂ρ_a contribution from ∂β*/∂ρ_a -> predict_delta_t_from_delta_beta
//   ∂u*/∂ρ_a contribution from G_{u,ρ_a} -> predict_delta_t_from_delta_gt
//
// and we reuse those predictors verbatim to keep the damping invariant
// in a single place.
// ---------------------------------------------------------------------------

/// Sensitivity bundle for one batch of `ρ` perturbations.
///
/// `beta_rho[:, a]` is `∂β*/∂ρ_a`; `u_rho[:, a]` is `∂u*/∂ρ_a` flat
/// row-major. Shapes follow the contract in proposal §5.3.
#[allow(dead_code)] // INTEGRATION-HOOK(evidence): consumed by solver/evidence::evidence_grad_rho when REML driver routes IFT cascade through this entry point.
#[derive(Debug, Clone)]
pub struct ArrowIftCascade {
    pub beta_rho: ndarray::Array2<f64>,
    pub u_rho: ndarray::Array2<f64>,
}

/// Materialize `∂(β*, u*)/∂ρ` from a converged arrow-Schur cache.
///
/// `schur_rhs_rho` has shape `K × R`, with column `a` equal to
/// `q_a = ∂g_red/∂ρ_a`.
///
/// `gu_rho` has shape `(N·d) × R`, with column `a` equal to the
/// per-row `G_{u,ρ_a}` stacked row-major.
///
/// Returns `None` if the cache lacks an undamped Schur factor (the
/// InexactPCG path; see proposal §6.5 — PCG mode cannot supply this
/// exactly without an explicit estimator).
///
/// This function does not mutate the cache and does not rebuild the
/// design (proposal §5.3 operational contract).
#[allow(dead_code)] // INTEGRATION-HOOK(evidence): see ArrowIftCascade.
pub fn ift_cascade_through_rho(
    cache: &ArrowFactorCache,
    schur_rhs_rho: ndarray::ArrayView2<'_, f64>,
    gu_rho: ndarray::ArrayView2<'_, f64>,
) -> Option<ArrowIftCascade> {
    let k = cache.k;
    let n = cache.htt_factors_undamped.len();
    let d = cache.d;
    debug_assert_eq!(schur_rhs_rho.nrows(), k);
    debug_assert_eq!(gu_rho.nrows(), n * d);
    let r = schur_rhs_rho.ncols();
    debug_assert_eq!(gu_rho.ncols(), r);

    // ∂β*/∂ρ — requires the Schur Cholesky factor. PCG caches fall
    // through `None` here, which the caller must handle (proposal §6.5).
    let beta_rho = crate::solver::evidence::ift_dbeta_drho(cache, schur_rhs_rho)?;

    // ∂u*/∂ρ — assemble per-column using the existing predictors.
    // The predictor sign convention is:
    //
    //   predict_delta_t_from_delta_beta(Δβ)  =  -H_uu⁻¹ H_uβ · Δβ
    //   predict_delta_t_from_delta_gt(δg_t)  =  -H_uu⁻¹ · δg_t
    //
    // and `∂u*/∂ρ_a = -H_uu⁻¹ G_{u,ρ_a} - H_uu⁻¹ H_uβ · ∂β*/∂ρ_a`, so
    // we sum the two predictor outputs directly (no further negation).
    let mut u_rho = ndarray::Array2::<f64>::zeros((n * d, r));
    let mut tmp_gu = Array1::<f64>::zeros(n * d);
    let mut tmp_db = Array1::<f64>::zeros(k);
    for a in 0..r {
        for row in 0..n * d {
            tmp_gu[row] = gu_rho[[row, a]];
        }
        for row in 0..k {
            tmp_db[row] = beta_rho[[row, a]];
        }
        let part_db = cache.predict_delta_t_from_delta_beta(tmp_db.view());
        let part_gu = cache.predict_delta_t_from_delta_gt(tmp_gu.view());
        for row in 0..n * d {
            u_rho[[row, a]] = part_db[row] + part_gu[row];
        }
    }

    Some(ArrowIftCascade { beta_rho, u_rho })
}

fn unix_secs_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct ResourcePolicy {
    pub max_single_materialization_bytes: usize,
    pub max_operator_cache_bytes: usize,
    pub max_spatial_distance_cache_bytes: usize,
    pub max_owned_data_cache_bytes: usize,
    pub row_chunk_target_bytes: usize,
    pub derivative_storage_mode: DerivativeStorageMode,
}

pub const SPATIAL_DISTANCE_CACHE_MAX_BYTES: usize = 512 * 1024 * 1024;
pub const SPATIAL_DISTANCE_CACHE_SINGLE_ENTRY_MAX_BYTES: usize = 256 * 1024 * 1024;
pub const OWNED_DATA_CACHE_MAX_ENTRIES: usize = 2;

/// Auto-strict triggers for [`ResourcePolicy::for_problem`].
///
/// Tuned for biobank-scale problems where dense materialization of any
/// design factor would itself be tens of GiB. Below these thresholds we
/// stay on `default_library` so small-data and ad-hoc fits keep working
/// without operator implementations.
pub const STRICT_POLICY_NROWS_THRESHOLD: usize = 100_000;
pub const STRICT_POLICY_P_THRESHOLD: usize = 5_000;

/// Hints that flip strict mode on regardless of n/p — used when a code path
/// is structurally operator-only and any dense fallback would be a bug.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProblemHints {
    pub marginal_slope_biobank_active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivativeStorageMode {
    /// Production exact-math: operator-backed, no dense fallback.
    AnalyticOperatorRequired,
    /// Allow dense materialization if under the single-materialization budget.
    MaterializeIfSmall,
    /// Dense materialization only permitted for diagnostic code paths.
    DiagnosticsOnly,
}

#[derive(Clone, Debug)]
pub struct MaterializationPolicy {
    pub max_single_dense_bytes: usize,
    pub max_cached_dense_bytes: usize,
    pub row_chunk_target_bytes: usize,
    pub allow_operator_materialization: bool,
    pub allow_diagnostic_materialization: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum MatrixMaterializationError {
    #[error(
        "{context}: dense materialization of {nrows}x{ncols} requires {bytes} bytes (limit {limit_bytes})"
    )]
    TooLarge {
        context: &'static str,
        nrows: usize,
        ncols: usize,
        bytes: usize,
        limit_bytes: usize,
    },

    #[error("{context}: operator does not implement chunked row access")]
    MissingRowChunk { context: &'static str },

    #[error("{context}: row materialization failed: {reason}")]
    RowMaterializationFailed {
        context: &'static str,
        reason: String,
    },

    #[error("{context}: materialization forbidden by policy (mode={mode:?})")]
    Forbidden {
        context: &'static str,
        mode: DerivativeStorageMode,
    },
}

pub trait ResidentBytes {
    fn resident_bytes(&self) -> usize;
}

impl ResourcePolicy {
    /// Conservative default suitable for general-purpose use.
    ///
    /// Uses `MaterializeIfSmall`: dense materialization is allowed only when the
    /// matrix fits under `max_single_materialization_bytes`. This lets small-data
    /// families that lack an implicit operator work out of the box, while
    /// biobank-scale problems error out and force the analytic-operator path.
    /// Set `derivative_storage_mode = AnalyticOperatorRequired` explicitly to
    /// reject all dense fallback.
    pub const fn default_library() -> Self {
        Self {
            max_single_materialization_bytes: 256 * 1024 * 1024, // 256 MiB
            max_operator_cache_bytes: 1024 * 1024 * 1024,        // 1 GiB
            max_spatial_distance_cache_bytes: SPATIAL_DISTANCE_CACHE_MAX_BYTES,
            max_owned_data_cache_bytes: 512 * 1024 * 1024, // 512 MiB
            row_chunk_target_bytes: 8 * 1024 * 1024,       // 8 MiB per chunk
            derivative_storage_mode: DerivativeStorageMode::MaterializeIfSmall,
        }
    }

    /// Strict mode that rejects every dense fallback. Use when you intend to
    /// run only on operator-backed bases (biobank-scale Duchon/TPS, exact
    /// GAMLSS marginal slope, CTN, etc.).
    pub const fn analytic_operator_required() -> Self {
        Self {
            max_single_materialization_bytes: 256 * 1024 * 1024,
            max_operator_cache_bytes: 1024 * 1024 * 1024,
            max_spatial_distance_cache_bytes: SPATIAL_DISTANCE_CACHE_MAX_BYTES,
            max_owned_data_cache_bytes: 512 * 1024 * 1024,
            row_chunk_target_bytes: 8 * 1024 * 1024,
            derivative_storage_mode: DerivativeStorageMode::AnalyticOperatorRequired,
        }
    }

    /// Auto-derive the resource policy from the shape of the problem rather
    /// than from an explicit CLI flag. The library refuses to silently
    /// densify operator-backed designs once the problem is large enough that
    /// a hidden dense fallback would blow real-world memory budgets, but
    /// keeps the permissive default for ordinary small-data fits so that
    /// non-operator bases still work out of the box.
    ///
    /// Strict mode (`AnalyticOperatorRequired`) is selected when ANY of:
    ///   * `n_rows >= STRICT_POLICY_NROWS_THRESHOLD` (biobank scale by row count)
    ///   * `p_estimate >= STRICT_POLICY_P_THRESHOLD` (biobank scale by coefficient count)
    ///   * `hints.marginal_slope_biobank_active` (the GAMLSS marginal-slope
    ///     biobank path is in play; the corresponding operators MUST stay
    ///     matrix-free regardless of n)
    pub const fn for_problem(n_rows: usize, p_estimate: usize, hints: ProblemHints) -> Self {
        let strict = n_rows >= STRICT_POLICY_NROWS_THRESHOLD
            || p_estimate >= STRICT_POLICY_P_THRESHOLD
            || hints.marginal_slope_biobank_active;
        if strict {
            Self::analytic_operator_required()
        } else {
            Self::default_library()
        }
    }

    /// Permissive mode for small-data usage and tests.
    pub const fn permissive_small_data() -> Self {
        Self {
            max_single_materialization_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
            max_operator_cache_bytes: 2 * 1024 * 1024 * 1024,
            max_spatial_distance_cache_bytes: SPATIAL_DISTANCE_CACHE_MAX_BYTES,
            max_owned_data_cache_bytes: 512 * 1024 * 1024,
            row_chunk_target_bytes: 64 * 1024 * 1024,
            derivative_storage_mode: DerivativeStorageMode::MaterializeIfSmall,
        }
    }

    pub const fn material_policy(&self) -> MaterializationPolicy {
        MaterializationPolicy {
            max_single_dense_bytes: self.max_single_materialization_bytes,
            max_cached_dense_bytes: self.max_operator_cache_bytes,
            row_chunk_target_bytes: self.row_chunk_target_bytes,
            allow_operator_materialization: matches!(
                self.derivative_storage_mode,
                DerivativeStorageMode::MaterializeIfSmall
            ),
            allow_diagnostic_materialization: !matches!(
                self.derivative_storage_mode,
                DerivativeStorageMode::AnalyticOperatorRequired
            ),
        }
    }
}

/// Returns how many rows to stream per chunk so that each chunk uses approximately
/// `target_bytes` given a row width of `cols` f64 entries.
pub const fn rows_for_target_bytes(target_bytes: usize, cols: usize) -> usize {
    let raw_bytes_per_row = cols.saturating_mul(std::mem::size_of::<f64>());
    let bytes_per_row = if raw_bytes_per_row == 0 {
        1
    } else {
        raw_bytes_per_row
    };
    let rows = target_bytes / bytes_per_row;
    if rows == 0 { 1 } else { rows }
}

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// Byte-limited LRU cache with an optional entry cap.
///
/// Unlike an entry-count-limited LRU, this cache tracks the resident byte cost
/// of each value (via [`ResidentBytes`]) and evicts the least-recently-used
/// entries until the total resident bytes fit under `max_bytes`. This is the
/// correct policy for biobank-scale payloads where a single cache entry (e.g.
/// an n*K distance matrix) can itself be multiple gigabytes and an entry-count
/// cap would silently blow the memory budget. Small entry caps are still useful
/// for payloads with known shape, such as owned PC data matrices shared across
/// model blocks.
pub struct ByteLruCache<K: Eq + Hash + Clone, V> {
    inner: Mutex<ByteLruInner<K, V>>,
    max_bytes: usize,
    max_entries: Option<usize>,
}

struct ByteLruInner<K, V> {
    map: HashMap<K, (V, usize)>, // (value, byte_charge)
    order: VecDeque<K>,
    resident_bytes: usize,
}

impl<K: Eq + Hash + Clone, V: Clone + ResidentBytes> ByteLruCache<K, V> {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(ByteLruInner {
                map: HashMap::new(),
                order: VecDeque::new(),
                resident_bytes: 0,
            }),
            max_bytes,
            max_entries: None,
        }
    }

    pub fn with_max_entries(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            inner: Mutex::new(ByteLruInner {
                map: HashMap::new(),
                order: VecDeque::new(),
                resident_bytes: 0,
            }),
            max_bytes,
            max_entries: Some(max_entries),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        // recover from poison
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let v = g.map.get(key)?.0.clone();
        // move to back (most-recently-used)
        if let Some(pos) = g.order.iter().position(|k| k == key) {
            let k = g.order.remove(pos).unwrap();
            g.order.push_back(k);
        }
        Some(v)
    }

    pub fn insert(&self, key: K, value: V) {
        let charge = value.resident_bytes();
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());

        // If already present, remove the old entry first so resident bytes stay
        // accurate and the LRU ordering reflects this insertion.
        if let Some((_old, old_charge)) = g.map.remove(&key) {
            g.resident_bytes = g.resident_bytes.saturating_sub(old_charge);
            if let Some(pos) = g.order.iter().position(|k| k == &key) {
                g.order.remove(pos);
            }
        }

        if charge > self.max_bytes {
            // Too large to cache; skip insertion.
            return;
        }

        if let Some(max_entries) = self.max_entries {
            if max_entries == 0 {
                return;
            }
            while g.map.len() >= max_entries {
                if let Some(evict_key) = g.order.pop_front() {
                    if let Some((_v, c)) = g.map.remove(&evict_key) {
                        g.resident_bytes = g.resident_bytes.saturating_sub(c);
                    }
                } else {
                    break;
                }
            }
        }

        while g.resident_bytes + charge > self.max_bytes {
            if let Some(evict_key) = g.order.pop_front() {
                if let Some((_v, c)) = g.map.remove(&evict_key) {
                    g.resident_bytes = g.resident_bytes.saturating_sub(c);
                }
            } else {
                break;
            }
        }

        g.map.insert(key.clone(), (value, charge));
        g.order.push_back(key);
        g.resident_bytes = g.resident_bytes.saturating_add(charge);
    }

    pub fn resident_bytes(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .resident_bytes
    }

    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .map
            .len()
    }

    pub fn clear(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.map.clear();
        g.order.clear();
        g.resident_bytes = 0;
    }
}

impl<K: Eq + Hash + Clone, V: Clone + ResidentBytes> std::fmt::Debug for ByteLruCache<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteLruCache")
            .field("resident_bytes", &self.resident_bytes())
            .field("max_bytes", &self.max_bytes)
            .field("max_entries", &self.max_entries)
            .finish()
    }
}

/// Byte-accounting for `Arc<Array2<f64>>`.
///
/// Reports the full dense footprint of the owned array. Multiple `Arc`s
/// pointing to the same allocation will each report the full size; this is
/// the conservative accounting the caches want because a single residency in
/// the cache is what we are budgeting for.
impl ResidentBytes for Arc<ndarray::Array2<f64>> {
    fn resident_bytes(&self) -> usize {
        std::mem::size_of::<f64>()
            .saturating_mul(self.nrows())
            .saturating_mul(self.ncols())
    }
}

/// Lazy-init cache safe to call from inside rayon par_iter.
///
/// `std::sync::OnceLock::get_or_init` parks racing threads on an OS
/// condition variable until the leader's init closure finishes. If the
/// leader's init closure itself dispatches a nested `into_par_iter`, the
/// parked threads are now unavailable as rayon workers, and the leader
/// blocks waiting for chunks that no one can service. Classic deadlock.
///
/// `RayonSafeOnce` removes the trap by computing the value *outside* any
/// lock. Concurrent racers may produce duplicate values; the first to
/// publish wins, the rest drop their result. No thread ever parks waiting
/// for another thread's init to finish, so nested rayon par_iter inside
/// the init closure is safe.
///
/// Use this in place of `OnceLock` whenever the init closure transitively
/// runs rayon work *and* the cache may be entered concurrently from
/// inside another rayon par_iter. The redundant-work cost on first race
/// is the price for never deadlocking; in practice the loser threads
/// throw away one round of work and steady-state is identical to
/// `OnceLock`.
pub struct RayonSafeOnce<T> {
    slot: std::sync::OnceLock<T>,
}

impl<T> RayonSafeOnce<T> {
    pub const fn new() -> Self {
        Self {
            slot: std::sync::OnceLock::new(),
        }
    }

    /// Returns the cached value if already populated.
    #[inline]
    pub fn get(&self) -> Option<&T> {
        self.slot.get()
    }

    /// Returns the cached value, computing it if absent.
    ///
    /// The init closure runs WITHOUT holding any lock — calls from
    /// concurrent rayon workers may all run it, and all but the first
    /// to call `set` discard their result. This is the contract that
    /// keeps nested `into_par_iter` inside `init` from deadlocking on
    /// other workers parked on a `OnceLock`.
    pub fn get_or_init<F>(&self, init: F) -> &T
    where
        F: FnOnce() -> T,
    {
        if let Some(v) = self.slot.get() {
            return v;
        }
        let candidate = init();
        drop(self.slot.set(candidate));
        self.slot
            .get()
            .expect("RayonSafeOnce slot populated by set() above")
    }

    /// Fallible variant of `get_or_init`.
    pub fn get_or_try_init<F, E>(&self, init: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if let Some(v) = self.slot.get() {
            return Ok(v);
        }
        let candidate = init()?;
        drop(self.slot.set(candidate));
        Ok(self
            .slot
            .get()
            .expect("RayonSafeOnce slot populated by set() above"))
    }
}

impl<T> Default for RayonSafeOnce<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for RayonSafeOnce<T> {
    fn clone(&self) -> Self {
        let cloned = Self::new();
        if let Some(value) = self.slot.get() {
            drop(cloned.slot.set(value.clone()));
        }
        cloned
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for RayonSafeOnce<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RayonSafeOnce")
            .field("slot", &self.slot.get())
            .finish()
    }
}

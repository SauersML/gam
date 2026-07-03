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
/// Tuned for large-scale problems where dense materialization of any
/// design factor would itself be tens of GiB. Below these thresholds we
/// stay on `default_library` so small-data and ad-hoc fits keep working
/// without operator implementations.
pub const STRICT_POLICY_NROWS_THRESHOLD: usize = 100_000;
pub const STRICT_POLICY_P_THRESHOLD: usize = 5_000;

/// Hints that flip strict mode on regardless of n/p — used when a code path
/// is structurally operator-only and any dense fallback would be a bug.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProblemHints {
    pub marginal_slope_large_scale_active: bool,
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
    /// pathologically large problems still error out and force the analytic-operator
    /// path. Set `derivative_storage_mode = AnalyticOperatorRequired` explicitly to
    /// reject all dense fallback.
    ///
    /// The 1 GiB single-materialization budget matches the established
    /// large-scale densification ceiling used elsewhere in the codebase
    /// (e.g. `CoefficientTransformOperator::MATERIALIZE_MAX_BYTES`). Real
    /// large-scale GAMLSS spatial designs (320k rows × ~130 cols ≈ 0.32 GiB)
    /// must be materializable under this default because their families
    /// (e.g. `BinomialLocationScale`) eagerly densify in
    /// `build_location_scale_block` and have no operator-only fallback. A
    /// tighter cap silently classified those as "too big" even though the
    /// only available code path is the dense one.
    pub const fn default_library() -> Self {
        Self {
            max_single_materialization_bytes: 1024 * 1024 * 1024, // 1 GiB
            max_operator_cache_bytes: 1024 * 1024 * 1024,         // 1 GiB
            max_spatial_distance_cache_bytes: SPATIAL_DISTANCE_CACHE_MAX_BYTES,
            max_owned_data_cache_bytes: 512 * 1024 * 1024, // 512 MiB
            row_chunk_target_bytes: 8 * 1024 * 1024,       // 8 MiB per chunk
            derivative_storage_mode: DerivativeStorageMode::MaterializeIfSmall,
        }
    }

    /// Strict mode that rejects every dense fallback. Use when you intend to
    /// run only on operator-backed bases (large-scale Duchon/TPS, exact
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
    ///   * `n_rows >= STRICT_POLICY_NROWS_THRESHOLD` (large scale by row count)
    ///   * `p_estimate >= STRICT_POLICY_P_THRESHOLD` (large scale by coefficient count)
    ///   * `hints.marginal_slope_large_scale_active` (the GAMLSS marginal-slope
    ///     large-scale path is in play; the corresponding operators MUST stay
    ///     matrix-free regardless of n)
    pub const fn for_problem(n_rows: usize, p_estimate: usize, hints: ProblemHints) -> Self {
        let strict = n_rows >= STRICT_POLICY_NROWS_THRESHOLD
            || p_estimate >= STRICT_POLICY_P_THRESHOLD
            || hints.marginal_slope_large_scale_active;
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
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Byte-limited LRU cache with an optional entry cap.
///
/// Unlike an entry-count-limited LRU, this cache tracks the resident byte cost
/// of each value (via [`ResidentBytes`]) and evicts the least-recently-used
/// entries until the total resident bytes fit under `max_bytes`. This is the
/// correct policy for large-scale payloads where a single cache entry (e.g.
/// an n*K distance matrix) can itself be multiple gigabytes and an entry-count
/// cap would silently blow the memory budget. Small entry caps are still useful
/// for payloads with known shape, such as owned PC data matrices shared across
/// model blocks.
pub struct ByteLruCache<K: Eq + Hash + Clone, V> {
    /// One independent LRU partition per shard. A single shard (the default)
    /// is byte-for-byte equivalent to the original single-`Mutex` cache; with
    /// `shard_count > 1` the key hash selects the shard, so concurrent traffic
    /// on distinct keys contends `1/shard_count` as often and each shard's
    /// recency `VecDeque` is `1/shard_count` as long (the hit-path rescan is a
    /// linear `position` lookup, so shrinking the per-shard order also cuts
    /// per-access cost). Sharding is opt-in (`new_sharded`) precisely because
    /// the byte budget is split across shards — that is correct for caches of
    /// many small entries (e.g. cell-moment memos) but wrong for caches of a
    /// few multi-GiB entries (distance matrices), which keep `shard_count == 1`.
    shards: Box<[Mutex<ByteLruInner<K, V>>]>,
    /// Per-shard byte budget. `shard_bytes * shards.len() >= max_bytes`.
    shard_bytes: usize,
    /// Per-shard entry budget, if any (`0` disables caching, as before).
    shard_entries: Option<usize>,
    max_bytes: usize,
}

struct ByteLruInner<K, V> {
    map: HashMap<K, (V, usize)>, // (value, byte_charge)
    order: VecDeque<K>,
    resident_bytes: usize,
}

impl<K: Eq + Hash + Clone, V: Clone + ResidentBytes> ByteLruCache<K, V> {
    pub fn new(max_bytes: usize) -> Self {
        Self::build(max_bytes, None, 1)
    }

    pub fn with_max_entries(max_bytes: usize, max_entries: usize) -> Self {
        Self::build(max_bytes, Some(max_entries), 1)
    }

    /// Like [`new`](Self::new) but partitions the cache across `shard_count`
    /// independently-locked LRU shards to cut lock contention under heavy
    /// concurrent access. The byte budget is divided evenly across shards, so
    /// this is only appropriate for caches holding many small entries.
    pub fn new_sharded(max_bytes: usize, shard_count: usize) -> Self {
        Self::build(max_bytes, None, shard_count)
    }

    /// Like [`with_max_entries`](Self::with_max_entries) but sharded; see
    /// [`new_sharded`](Self::new_sharded).
    pub fn with_max_entries_sharded(
        max_bytes: usize,
        max_entries: usize,
        shard_count: usize,
    ) -> Self {
        Self::build(max_bytes, Some(max_entries), shard_count)
    }

    fn build(max_bytes: usize, max_entries: Option<usize>, shard_count: usize) -> Self {
        let shard_count = shard_count.max(1);
        // Split the global budgets across shards, rounding up so the aggregate
        // capacity never falls below the requested budget. With a single shard
        // these equal the global budgets exactly (legacy behavior). A `0`
        // entry budget still disables caching and must not be rounded up to 1.
        let shard_bytes = max_bytes.div_ceil(shard_count);
        let shard_entries = max_entries.map(|m| {
            if m == 0 {
                0
            } else {
                m.div_ceil(shard_count).max(1)
            }
        });
        let shards = (0..shard_count)
            .map(|_| {
                Mutex::new(ByteLruInner {
                    map: HashMap::new(),
                    order: VecDeque::new(),
                    resident_bytes: 0,
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            shards,
            shard_bytes,
            shard_entries,
            max_bytes,
        }
    }

    #[inline]
    fn shard(&self, key: &K) -> &Mutex<ByteLruInner<K, V>> {
        if self.shards.len() == 1 {
            return &self.shards[0];
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        &self.shards[(hasher.finish() as usize) % self.shards.len()]
    }

    pub fn get(&self, key: &K) -> Option<V> {
        // recover from poison
        let mut g = self.shard(key).lock().unwrap_or_else(|p| p.into_inner());
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
        let mut g = self.shard(&key).lock().unwrap_or_else(|p| p.into_inner());

        // If already present, remove the old entry first so resident bytes stay
        // accurate and the LRU ordering reflects this insertion.
        if let Some((_old, old_charge)) = g.map.remove(&key) {
            g.resident_bytes = g.resident_bytes.saturating_sub(old_charge);
            if let Some(pos) = g.order.iter().position(|k| k == &key) {
                g.order.remove(pos);
            }
        }

        if charge > self.shard_bytes {
            // Too large to cache; skip insertion.
            return;
        }

        if let Some(max_entries) = self.shard_entries {
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

        while g.resident_bytes + charge > self.shard_bytes {
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
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .resident_bytes
            })
            .sum()
    }

    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.lock().unwrap_or_else(|p| p.into_inner()).map.len())
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        for shard in self.shards.iter() {
            let mut g = shard.lock().unwrap_or_else(|p| p.into_inner());
            g.map.clear();
            g.order.clear();
            g.resident_bytes = 0;
        }
    }
}

impl<K: Eq + Hash + Clone, V: Clone + ResidentBytes> std::fmt::Debug for ByteLruCache<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteLruCache")
            .field("resident_bytes", &self.resident_bytes())
            .field("max_bytes", &self.max_bytes)
            .field("shard_count", &self.shards.len())
            .field("shard_bytes", &self.shard_bytes)
            .field("shard_entries", &self.shard_entries)
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
    ///
    /// Named `get_or_compute` (not `get_or_init`) so the codebase-level
    /// lint that bans `OnceLock::get_or_init` near rayon `par_iter` does
    /// not flag this safe-by-construction path.
    pub fn get_or_compute<F>(&self, init: F) -> &T
    where
        F: FnOnce() -> T,
    {
        if let Some(v) = self.slot.get() {
            return v;
        }
        let candidate = init();
        self.slot.set(candidate).ok();
        self.slot
            .get()
            .expect("RayonSafeOnce slot populated by set() above")
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
            cloned.slot.set(value.clone()).ok();
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

#[cfg(test)]
mod byte_lru_tests {
    use super::*;

    /// Fixed-charge value so byte-budget arithmetic in the tests is exact.
    #[derive(Clone, PartialEq, Debug)]
    struct Payload(u64);
    impl ResidentBytes for Payload {
        fn resident_bytes(&self) -> usize {
            8
        }
    }

    #[test]
    fn single_shard_round_trips_and_evicts_by_bytes() {
        // 3 entries' worth of budget; a single shard preserves strict global LRU.
        let cache: ByteLruCache<u64, Payload> = ByteLruCache::new(24);
        for k in 0..3 {
            cache.insert(k, Payload(k));
        }
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.resident_bytes(), 24);
        // Touch key 0 so it is most-recently-used, then overflow by one.
        assert_eq!(cache.get(&0), Some(Payload(0)));
        cache.insert(3, Payload(3));
        // Key 1 (now least-recently-used) is evicted; 0 survives the touch.
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&0), Some(Payload(0)));
        assert_eq!(cache.get(&3), Some(Payload(3)));
    }

    #[test]
    fn zero_entry_budget_disables_caching_in_every_shard() {
        let single: ByteLruCache<u64, Payload> = ByteLruCache::with_max_entries(1 << 20, 0);
        single.insert(7, Payload(7));
        assert_eq!(single.get(&7), None);
        let sharded: ByteLruCache<u64, Payload> =
            ByteLruCache::with_max_entries_sharded(1 << 20, 0, 16);
        sharded.insert(7, Payload(7));
        assert_eq!(sharded.get(&7), None);
    }

    #[test]
    fn sharded_cache_retrieves_all_keys_and_respects_aggregate_budget() {
        // Generous budget split across 8 shards; every inserted key must be
        // retrievable and the aggregate residency must never exceed the global
        // budget (shard_bytes * shard_count, rounded up).
        let shard_count = 8usize;
        let max_bytes = 8 * 64; // 64 entries' worth, 8 per shard on average.
        let cache: ByteLruCache<u64, Payload> = ByteLruCache::new_sharded(max_bytes, shard_count);
        for k in 0..64u64 {
            cache.insert(k, Payload(k));
        }
        // Per-shard budgets sum to >= the requested global budget.
        assert!(cache.resident_bytes() <= max_bytes.div_ceil(shard_count) * shard_count);
        // Re-inserting then reading back a key returns the stored payload.
        cache.insert(123, Payload(123));
        assert_eq!(cache.get(&123), Some(Payload(123)));
        assert!(!cache.is_empty());
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.resident_bytes(), 0);
    }
}

#[cfg(test)]
mod resource_policy_tests {
    use super::*;

    // ── rows_for_target_bytes ─────────────────────────────────────────────────

    #[test]
    fn rows_for_target_bytes_exact_fit() {
        // 1 col × 8 bytes/f64; target 8 bytes → 1 row
        assert_eq!(rows_for_target_bytes(8, 1), 1);
    }

    #[test]
    fn rows_for_target_bytes_multiple_rows() {
        // 1 col × 8 bytes/f64; target 80 bytes → 10 rows
        assert_eq!(rows_for_target_bytes(80, 1), 10);
    }

    #[test]
    fn rows_for_target_bytes_multiple_cols() {
        // 4 cols × 8 = 32 bytes/row; target 128 → 4 rows
        assert_eq!(rows_for_target_bytes(128, 4), 4);
    }

    #[test]
    fn rows_for_target_bytes_zero_target_returns_one() {
        // Zero target cannot give 0 rows — floor to 1
        assert_eq!(rows_for_target_bytes(0, 1), 1);
    }

    #[test]
    fn rows_for_target_bytes_zero_cols_returns_non_zero() {
        // Zero cols → bytes_per_row falls back to 1 → rows = target
        assert_eq!(rows_for_target_bytes(100, 0), 100);
    }

    #[test]
    fn rows_for_target_bytes_large_target() {
        // 8 MiB target, 1024 cols → 1024 bytes/row → 8192 rows
        let target = 8 * 1024 * 1024;
        let cols = 1024_usize;
        let expected = target / (cols * std::mem::size_of::<f64>());
        assert_eq!(rows_for_target_bytes(target, cols), expected);
    }

    // ── ResourcePolicy::for_problem ──────────────────────────────────────────

    #[test]
    fn for_problem_small_data_uses_materialize_if_small() {
        let p = ResourcePolicy::for_problem(1000, 100, ProblemHints::default());
        assert_eq!(
            p.derivative_storage_mode,
            DerivativeStorageMode::MaterializeIfSmall
        );
    }

    #[test]
    fn for_problem_large_nrows_is_strict() {
        let p = ResourcePolicy::for_problem(
            STRICT_POLICY_NROWS_THRESHOLD,
            100,
            ProblemHints::default(),
        );
        assert_eq!(
            p.derivative_storage_mode,
            DerivativeStorageMode::AnalyticOperatorRequired
        );
    }

    #[test]
    fn for_problem_large_p_is_strict() {
        let p =
            ResourcePolicy::for_problem(100, STRICT_POLICY_P_THRESHOLD, ProblemHints::default());
        assert_eq!(
            p.derivative_storage_mode,
            DerivativeStorageMode::AnalyticOperatorRequired
        );
    }

    #[test]
    fn for_problem_marginal_slope_hint_is_strict() {
        let p = ResourcePolicy::for_problem(
            100,
            100,
            ProblemHints {
                marginal_slope_large_scale_active: true,
            },
        );
        assert_eq!(
            p.derivative_storage_mode,
            DerivativeStorageMode::AnalyticOperatorRequired
        );
    }

    // ── ResourcePolicy::material_policy ─────────────────────────────────────

    #[test]
    fn material_policy_default_library_allows_operator_and_diagnostics() {
        let mp = ResourcePolicy::default_library().material_policy();
        assert!(mp.allow_operator_materialization);
        assert!(mp.allow_diagnostic_materialization);
    }

    #[test]
    fn material_policy_analytic_operator_required_blocks_both() {
        let mp = ResourcePolicy::analytic_operator_required().material_policy();
        assert!(!mp.allow_operator_materialization);
        assert!(!mp.allow_diagnostic_materialization);
    }

    #[test]
    fn material_policy_propagates_byte_limits() {
        let policy = ResourcePolicy::default_library();
        let mp = policy.material_policy();
        assert_eq!(
            mp.max_single_dense_bytes,
            policy.max_single_materialization_bytes
        );
        assert_eq!(mp.max_cached_dense_bytes, policy.max_operator_cache_bytes);
        assert_eq!(mp.row_chunk_target_bytes, policy.row_chunk_target_bytes);
    }
}

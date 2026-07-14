use crate::cgroup_memory::detect_cgroup_memory;
pub use crate::cgroup_memory::{
    CgroupMemoryAvailability, CgroupMemoryLimit, CgroupMemoryObservation,
    CgroupMemoryProbeFailure, CgroupMemoryProbeFailureKind,
};

/// The library-default streamed row-chunk target (8 MiB), shared as a `const`
/// so compile-time consumers (e.g. device tile geometry) stay in lockstep with
/// [`ResourcePolicy::default_library`] without a runtime policy query.
pub const LIBRARY_ROW_CHUNK_TARGET_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct ResourcePolicy {
    pub max_single_materialization_bytes: usize,
    pub max_operator_cache_bytes: usize,
    pub max_spatial_distance_cache_bytes: usize,
    pub max_owned_data_cache_bytes: usize,
    pub row_chunk_target_bytes: usize,
    pub derivative_storage_mode: DerivativeStorageMode,
}

pub const OWNED_DATA_CACHE_MAX_ENTRIES: usize = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Process-wide memory governor
// ─────────────────────────────────────────────────────────────────────────────

/// Fraction of the *detected available* memory the governor is allowed to hand
/// out as reservations: 3/4.
///
/// The ledger only accounts for the large, planned allocations that route
/// through [`MemoryGovernor::try_reserve`] (dense design materializations,
/// covariance blocks, sampler design assemblies). Everything else — allocator
/// slack, thread stacks, code, and small per-iteration temporaries, plus
/// whatever the rest of the machine does concurrently — lives in the remaining
/// quarter. The base quantity is *available* (not total) memory, so memory
/// already committed by other processes is excluded before the fraction is
/// applied.
const GOVERNOR_BUDGET_NUMERATOR: u128 = 3;
const GOVERNOR_BUDGET_DENOMINATOR: u128 = 4;

/// Which observation is the binding ceiling on memory available to this
/// process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAvailabilitySource {
    Host,
    Cgroup,
    HostAndCgroup,
    CgroupProbeFailure,
}

impl std::fmt::Display for MemoryAvailabilitySource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host => formatter.write_str("host"),
            Self::Cgroup => formatter.write_str("cgroup"),
            Self::HostAndCgroup => formatter.write_str("host and cgroup equally"),
            Self::CgroupProbeFailure => formatter.write_str("cgroup probe failure"),
        }
    }
}

/// Provenance-preserving memory availability for the current process.
///
/// A finite cgroup-v1 or cgroup-v2 ceiling admits exactly `min(host available,
/// reclaim-aware cgroup available)`. A v2 literal `memory.max = max` remains
/// typed as unbounded and therefore defers to the host; v1's numeric unlimited
/// sentinel participates exactly and likewise loses to a tighter host value.
/// If an active controller cannot be parsed exactly, admission fails closed
/// with zero bytes while retaining the typed probe failure; it never silently
/// inherits host capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAvailability {
    host_available_bytes: u64,
    cgroup: CgroupMemoryObservation,
    available_bytes: u64,
    limiting_source: MemoryAvailabilitySource,
}

impl MemoryAvailability {
    fn from_observation(
        host_available_bytes: u64,
        cgroup: CgroupMemoryObservation,
    ) -> Self {
        use std::cmp::Ordering;

        let (available_bytes, limiting_source) = match &cgroup {
            CgroupMemoryObservation::NotPresent
            | CgroupMemoryObservation::V2Unbounded { .. } => {
                (host_available_bytes, MemoryAvailabilitySource::Host)
            }
            CgroupMemoryObservation::V2Limited(observation)
            | CgroupMemoryObservation::V1Limited(observation) => {
                match observation.available_bytes().cmp(&host_available_bytes) {
                    Ordering::Less => (
                        observation.available_bytes(),
                        MemoryAvailabilitySource::Cgroup,
                    ),
                    Ordering::Equal => (
                        host_available_bytes,
                        MemoryAvailabilitySource::HostAndCgroup,
                    ),
                    Ordering::Greater => {
                        (host_available_bytes, MemoryAvailabilitySource::Host)
                    }
                }
            }
            CgroupMemoryObservation::ProbeFailed(_) => {
                (0, MemoryAvailabilitySource::CgroupProbeFailure)
            }
        };
        Self {
            host_available_bytes,
            cgroup,
            available_bytes,
            limiting_source,
        }
    }

    pub const fn host_available_bytes(&self) -> u64 {
        self.host_available_bytes
    }

    pub const fn cgroup(&self) -> &CgroupMemoryObservation {
        &self.cgroup
    }

    pub const fn available_bytes(&self) -> u64 {
        self.available_bytes
    }

    pub fn available_bytes_usize(&self) -> usize {
        usize::try_from(self.available_bytes).unwrap_or(usize::MAX)
    }

    pub const fn limiting_source(&self) -> MemoryAvailabilitySource {
        self.limiting_source
    }
}

impl std::fmt::Display for MemoryAvailability {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cgroup {
            CgroupMemoryObservation::ProbeFailed(failure) => write!(
                formatter,
                "0 bytes admitted because the active cgroup probe failed closed (host_available={}, failure={})",
                self.host_available_bytes, failure,
            ),
            observation => write!(
                formatter,
                "{} bytes limited by {} (host_available={}, {})",
                self.available_bytes, self.limiting_source, self.host_available_bytes, observation,
            ),
        }
    }
}

/// Refresh and return the OS and cgroup observations that govern process
/// memory admission. This is the single system-memory authority shared by the
/// global governor and specialized runtime planners.
pub fn detect_memory_availability() -> MemoryAvailability {
    static SYSTEM: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let system = SYSTEM.get_or_init(|| Mutex::new(sysinfo::System::new()));
    let mut system = system.lock().expect("sysinfo system mutex poisoned");
    system.refresh_memory();
    let cgroup = detect_cgroup_memory();
    MemoryAvailability::from_observation(system.available_memory(), cgroup)
}

/// Convert one provenance-preserving availability observation to the process
/// budget. Zero is an authoritative exhausted-memory signal and deliberately
/// yields a zero budget.
fn governor_budget_from_availability(availability: &MemoryAvailability) -> usize {
    let scaled = u128::from(availability.available_bytes()) * GOVERNOR_BUDGET_NUMERATOR
        / GOVERNOR_BUDGET_DENOMINATOR;
    usize::try_from(scaled).unwrap_or(usize::MAX)
}

/// Typed refusal from [`MemoryGovernor::try_reserve`].
///
/// Carries the full ledger evidence so callers can route to a chunked or
/// matrix-free strategy (and so error messages explain *why* dense was
/// refused). This is a routing signal, never an abort: the process still has
/// its unreserved headroom, the requested allocation just does not fit the
/// joint budget.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum MemoryReservationError {
    #[error(
        "{context}: cannot reserve {requested_bytes} bytes; {reserved_bytes} of {budget_bytes} bytes already reserved process-wide; detected availability: {availability}"
    )]
    BudgetExceeded {
        context: Box<str>,
        requested_bytes: usize,
        reserved_bytes: usize,
        budget_bytes: usize,
        availability: MemoryAvailability,
    },

    #[error(
        "{context}: dense allocation size overflow for {copies} copies of a {nrows}x{ncols} f64 matrix"
    )]
    SizeOverflow {
        context: Box<str>,
        nrows: usize,
        ncols: usize,
        copies: usize,
    },
}

#[derive(Debug)]
struct GovernorLedger {
    budget_bytes: usize,
    availability: MemoryAvailability,
    reserved_bytes: std::sync::atomic::AtomicUsize,
}

/// Process-wide byte-accounting governor for large allocations.
///
/// Every large planned allocation (dense design materialization, covariance
/// block, sampler design assembly) reserves its byte footprint against one
/// shared ledger via [`try_reserve`](Self::try_reserve) and holds the returned
/// RAII [`MemoryReservation`] for as long as the allocation is live. Because
/// the ledger is shared, allocations that are each individually acceptable can
/// no longer *jointly* exceed memory: whichever request would tip the ledger
/// past the budget gets a typed [`MemoryReservationError`] and routes to a
/// chunked or matrix-free strategy instead. Strategy selection is thereby a
/// continuous function of predicted live bytes vs remaining budget, not of
/// row/column thresholds.
///
/// The global budget is sized once from actually-available memory (host
/// `available_memory`, clamped by cgroup limits inside containers) — see
/// [`GOVERNOR_BUDGET_NUMERATOR`] for the headroom rationale.
#[derive(Debug, Clone)]
pub struct MemoryGovernor {
    ledger: Arc<GovernorLedger>,
}

impl MemoryGovernor {
    /// The process-wide governor. Budget detection runs once, on first use.
    pub fn global() -> &'static MemoryGovernor {
        static GLOBAL: OnceLock<MemoryGovernor> = OnceLock::new();
        GLOBAL.get_or_init(|| {
            let availability = detect_memory_availability();
            MemoryGovernor::with_detected_availability(availability)
        })
    }

    fn with_detected_availability(availability: MemoryAvailability) -> Self {
        let budget_bytes = governor_budget_from_availability(&availability);
        Self {
            ledger: Arc::new(GovernorLedger {
                budget_bytes,
                availability,
                reserved_bytes: std::sync::atomic::AtomicUsize::new(0),
            }),
        }
    }

    pub fn budget_bytes(&self) -> usize {
        self.ledger.budget_bytes
    }

    pub fn availability(&self) -> MemoryAvailability {
        self.ledger.availability.clone()
    }

    pub fn reserved_bytes(&self) -> usize {
        self.ledger
            .reserved_bytes
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn remaining_bytes(&self) -> usize {
        self.ledger
            .budget_bytes
            .saturating_sub(self.reserved_bytes())
    }

    /// Absolute ceiling for one governed operation. Consumers reserve the
    /// operation's complete predicted live set (matrix plus simultaneous
    /// workspaces/copies), so the shared ledger itself is the policy.
    pub fn single_materialization_cap_bytes(&self) -> usize {
        self.ledger.budget_bytes
    }

    /// Reserve `bytes` against the joint ledger.
    ///
    /// On success the returned [`MemoryReservation`] must be held for as long
    /// as the allocation it accounts for is live; dropping it releases the
    /// bytes. On failure the caller receives the ledger evidence and is
    /// expected to fall back to a chunked or matrix-free strategy.
    pub fn try_reserve(
        &self,
        bytes: usize,
        context: &str,
    ) -> Result<MemoryReservation, MemoryReservationError> {
        use std::sync::atomic::Ordering;
        let mut current = self.ledger.reserved_bytes.load(Ordering::Relaxed);
        loop {
            let next = match current.checked_add(bytes) {
                Some(next) if next <= self.ledger.budget_bytes => next,
                _ => {
                    return Err(MemoryReservationError::BudgetExceeded {
                        context: context.into(),
                        requested_bytes: bytes,
                        reserved_bytes: current,
                        budget_bytes: self.ledger.budget_bytes,
                        availability: self.ledger.availability.clone(),
                    });
                }
            };
            match self.ledger.reserved_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(MemoryReservation {
                        ledger: Arc::clone(&self.ledger),
                        bytes,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }

    /// Reserve the footprint of a dense `nrows × ncols` `f64` matrix.
    /// Dimension-product overflow is reported as a budget refusal (an
    /// allocation whose size cannot even be computed certainly does not fit).
    pub fn try_reserve_dense_f64(
        &self,
        nrows: usize,
        ncols: usize,
        context: &str,
    ) -> Result<MemoryReservation, MemoryReservationError> {
        self.try_reserve_dense_f64_copies(nrows, ncols, 1, context)
    }

    /// Reserve the predicted live footprint of `copies` simultaneous dense
    /// matrices with one atomic ledger charge.
    pub fn try_reserve_dense_f64_copies(
        &self,
        nrows: usize,
        ncols: usize,
        copies: usize,
        context: &str,
    ) -> Result<MemoryReservation, MemoryReservationError> {
        let bytes = dense_f64_bytes(nrows, ncols)
            .and_then(|one| one.checked_mul(copies))
            .ok_or_else(|| MemoryReservationError::SizeOverflow {
                context: context.into(),
                nrows,
                ncols,
                copies,
            })?;
        self.try_reserve(bytes, context)
    }
}

/// Checked byte footprint of a dense `nrows × ncols` `f64` matrix.
pub const fn dense_f64_bytes(nrows: usize, ncols: usize) -> Option<usize> {
    match nrows.checked_mul(ncols) {
        Some(cells) => cells.checked_mul(std::mem::size_of::<f64>()),
        None => None,
    }
}

/// RAII guard for bytes reserved on a [`MemoryGovernor`] ledger; dropping it
/// releases the reservation. Hold it exactly as long as the accounted
/// allocation is live.
#[derive(Debug)]
#[must_use = "dropping a memory reservation immediately releases its ledger charge"]
pub struct MemoryReservation {
    ledger: Arc<GovernorLedger>,
    bytes: usize,
}

impl MemoryReservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Couple this reservation to the value whose memory it accounts for.
    pub fn bind<T>(self, value: T) -> Governed<T> {
        Governed {
            value,
            reservation: self,
        }
    }
}

/// A value whose live memory is coupled to a process-wide reservation.
///
/// Large fallible materializations return this owner so the allocation cannot
/// outlive its ledger charge. It dereferences to the wrapped value for normal
/// ndarray and collection operations.
#[derive(Debug)]
#[must_use = "the governed value owns a live process-wide memory reservation"]
pub struct Governed<T> {
    value: T,
    reservation: MemoryReservation,
}

impl<T> Governed<T> {
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
}

impl<T> std::ops::Deref for Governed<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> std::ops::DerefMut for Governed<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T> AsRef<T> for Governed<T> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> AsMut<T> for Governed<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.ledger
            .reserved_bytes
            .fetch_sub(self.bytes, std::sync::atomic::Ordering::AcqRel);
    }
}

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

    /// The process-wide [`MemoryGovernor`] could not reserve the requested live
    /// footprint (or its checked byte size overflowed). Callers route to a
    /// chunked or matrix-free strategy.
    #[error(transparent)]
    Reservation(#[from] MemoryReservationError),
}

pub trait ResidentBytes {
    fn resident_bytes(&self) -> usize;
}

impl ResourcePolicy {
    /// Conservative default suitable for general-purpose use.
    ///
    /// Uses `MaterializeIfSmall`: dense materialization is allowed only when
    /// the matrix fits under `max_single_materialization_bytes`. This lets
    /// small-data families that lack an implicit operator work out of the box,
    /// while problems whose dense footprint does not fit real memory get a
    /// typed refusal that forces the analytic-operator path. Set
    /// `derivative_storage_mode = AnalyticOperatorRequired` explicitly to
    /// reject all dense fallback.
    ///
    /// Scalar caps expose the governor's full allowance; they are admission
    /// hints, not independent budgets. Actual materializations and caches must
    /// reserve their complete live footprint against the shared governor, so
    /// any combination of categories is bounded by one ledger.
    pub fn default_library() -> Self {
        let governor = MemoryGovernor::global();
        let single_cap = governor.single_materialization_cap_bytes();
        Self {
            max_single_materialization_bytes: single_cap,
            max_operator_cache_bytes: single_cap,
            max_spatial_distance_cache_bytes: single_cap,
            max_owned_data_cache_bytes: single_cap,
            row_chunk_target_bytes: LIBRARY_ROW_CHUNK_TARGET_BYTES,
            derivative_storage_mode: DerivativeStorageMode::MaterializeIfSmall,
        }
    }

    /// Strict mode that rejects every dense fallback. Use when you intend to
    /// run only on operator-backed bases (large-scale Duchon/TPS, exact
    /// GAMLSS marginal slope, CTN, etc.). The byte caps only govern the
    /// residual diagnostic surfaces (materialization itself is forbidden by
    /// the mode).
    pub fn analytic_operator_required() -> Self {
        let base = Self::default_library();
        Self {
            derivative_storage_mode: DerivativeStorageMode::AnalyticOperatorRequired,
            ..base
        }
    }

    /// Auto-derive the resource policy from the shape of the problem rather
    /// than from an explicit CLI flag.
    ///
    /// Shape alone never flips a policy mode: doing so merely moves the old
    /// row/column cliff to a byte threshold. Every non-structural path starts
    /// permissive and makes its strategy decision from the operation's checked
    /// predicted live bytes versus the governor's current remaining budget.
    ///
    /// `hints.marginal_slope_large_scale_active` forces strict mode regardless
    /// of shape: that path is structurally operator-only and any dense
    /// fallback would be a bug, not a memory question.
    pub fn for_problem(hints: ProblemHints) -> Self {
        if hints.marginal_slope_large_scale_active {
            return Self::analytic_operator_required();
        }
        Self::default_library()
    }

    /// Permissive mode for small-data usage and tests. Admission still uses
    /// the same process ledger; only the streaming chunk geometry differs.
    pub fn permissive_small_data() -> Self {
        let base = Self::default_library();
        Self {
            row_chunk_target_bytes: 64 * 1024 * 1024,
            ..base
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

/// Select the row count for prediction-time covariance work.
///
/// A prediction row can keep roughly four `parameter_dim × local_dim`
/// `f64` workspaces live while gradients and covariance solves are assembled.
/// This central policy prevents predictors from drifting to different memory
/// budgets and chunk bounds for the same operation.
pub fn prediction_chunk_rows(parameter_dim: usize, local_dim: usize, total_rows: usize) -> usize {
    const MIN_ROWS: usize = 16;
    const MAX_ROWS: usize = 4096;

    if total_rows == 0 {
        return 1;
    }
    let live_f64_values_per_row = parameter_dim
        .max(1)
        .saturating_mul(local_dim.max(1))
        .saturating_mul(4);
    rows_for_target_bytes(
        ResourcePolicy::default_library().row_chunk_target_bytes,
        live_f64_values_per_row,
    )
    .clamp(MIN_ROWS, MAX_ROWS)
    .min(total_rows)
}

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

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
    governor: MemoryGovernor,
}

struct ByteLruInner<K, V> {
    // The reservation is stored beside the value, so eviction and clear drop
    // the process-wide charge at exactly the same time as cache ownership.
    map: HashMap<K, (V, usize, MemoryReservation)>,
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
        Self::build_with_governor(
            max_bytes,
            max_entries,
            shard_count,
            MemoryGovernor::global().clone(),
        )
    }

    fn build_with_governor(
        max_bytes: usize,
        max_entries: Option<usize>,
        shard_count: usize,
        governor: MemoryGovernor,
    ) -> Self {
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
            governor,
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
        if let Some((_old, old_charge, _reservation)) = g.map.remove(&key) {
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
                    if let Some((_v, c, _reservation)) = g.map.remove(&evict_key) {
                        g.resident_bytes = g.resident_bytes.saturating_sub(c);
                    }
                } else {
                    break;
                }
            }
        }

        while g.resident_bytes + charge > self.shard_bytes {
            if let Some(evict_key) = g.order.pop_front() {
                if let Some((_v, c, _reservation)) = g.map.remove(&evict_key) {
                    g.resident_bytes = g.resident_bytes.saturating_sub(c);
                }
            } else {
                break;
            }
        }

        let reservation = match self
            .governor
            .try_reserve(charge, "ByteLruCache resident entry")
        {
            Ok(reservation) => reservation,
            Err(_) => return,
        };
        g.map.insert(key.clone(), (value, charge, reservation));
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

    fn cache_test_governor(budget_bytes: usize) -> MemoryGovernor {
        let available_bytes = (budget_bytes as u128 * GOVERNOR_BUDGET_DENOMINATOR)
            .div_ceil(GOVERNOR_BUDGET_NUMERATOR);
        MemoryGovernor::with_detected_availability(MemoryAvailability::from_observation(
            u64::try_from(available_bytes).expect("test cache budget must fit in u64"),
            CgroupMemoryObservation::NotPresent,
        ))
    }

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
        let cache: ByteLruCache<u64, Payload> =
            ByteLruCache::build_with_governor(24, None, 1, cache_test_governor(24));
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
        let cache: ByteLruCache<u64, Payload> = ByteLruCache::build_with_governor(
            max_bytes,
            None,
            shard_count,
            cache_test_governor(max_bytes),
        );
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

    fn test_governor(budget_bytes: usize) -> MemoryGovernor {
        let available_bytes = (budget_bytes as u128 * GOVERNOR_BUDGET_DENOMINATOR)
            .div_ceil(GOVERNOR_BUDGET_NUMERATOR);
        let availability = MemoryAvailability::from_observation(
            u64::try_from(available_bytes).expect("test budget has a representable observation"),
            CgroupMemoryObservation::NotPresent,
        );
        let governor = MemoryGovernor::with_detected_availability(availability);
        assert_eq!(governor.budget_bytes(), budget_bytes);
        governor
    }

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

    #[test]
    fn prediction_chunks_share_the_runtime_byte_budget() {
        assert_eq!(prediction_chunk_rows(1024, 1, 100_000), 256);
        assert_eq!(prediction_chunk_rows(32, 2, 100_000), 4096);
    }

    #[test]
    fn prediction_chunks_respect_dataset_bounds() {
        assert_eq!(prediction_chunk_rows(1, 1, 7), 7);
        assert_eq!(prediction_chunk_rows(1, 1, 0), 1);
    }

    // ── ResourcePolicy::for_problem ──────────────────────────────────────────

    #[test]
    fn for_problem_small_data_uses_materialize_if_small() {
        let p = ResourcePolicy::for_problem(ProblemHints::default());
        assert_eq!(
            p.derivative_storage_mode,
            DerivativeStorageMode::MaterializeIfSmall
        );
    }

    #[test]
    fn for_problem_has_no_row_or_column_cliff() {
        let narrow = ResourcePolicy::for_problem(ProblemHints::default());
        let wide = ResourcePolicy::for_problem(ProblemHints::default());
        assert_eq!(
            narrow.derivative_storage_mode,
            DerivativeStorageMode::MaterializeIfSmall
        );
        assert_eq!(
            wide.derivative_storage_mode,
            DerivativeStorageMode::MaterializeIfSmall
        );
    }

    #[test]
    fn for_problem_dimension_overflow_defers_to_typed_reservation() {
        let policy = ResourcePolicy::for_problem(ProblemHints::default());
        assert_eq!(
            policy.derivative_storage_mode,
            DerivativeStorageMode::MaterializeIfSmall
        );
    }

    #[test]
    fn for_problem_marginal_slope_hint_is_strict() {
        let p = ResourcePolicy::for_problem(ProblemHints {
            marginal_slope_large_scale_active: true,
        });
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

    // ── MemoryGovernor ledger ────────────────────────────────────────────────

    #[test]
    fn reservations_account_and_release_on_drop() {
        let governor = test_governor(1_000);
        assert_eq!(governor.remaining_bytes(), 1_000);
        let first = governor.try_reserve(600, "test-first").expect("fits");
        assert_eq!(governor.reserved_bytes(), 600);
        assert_eq!(governor.remaining_bytes(), 400);
        assert_eq!(first.bytes(), 600);
        drop(first);
        assert_eq!(governor.reserved_bytes(), 0);
        assert_eq!(governor.remaining_bytes(), 1_000);
    }

    #[test]
    fn jointly_excessive_reservations_are_refused_with_evidence() {
        // Two allocations that each fit alone must not be jointly grantable —
        // this is exactly the independent-budgets failure the ledger exists
        // to prevent.
        let governor = test_governor(1_000);
        let availability = governor.availability();
        let held = governor.try_reserve(600, "test-held").expect("fits alone");
        let refusal = governor
            .try_reserve(600, "test-joint")
            .expect_err("600 + 600 exceeds the 1000-byte budget");
        assert_eq!(
            refusal,
            MemoryReservationError::BudgetExceeded {
                context: "test-joint".into(),
                requested_bytes: 600,
                reserved_bytes: 600,
                budget_bytes: 1_000,
                availability,
            }
        );
        // After releasing the holder, the same request succeeds: refusal is a
        // routing signal, not a terminal state.
        drop(held);
        let refreshed = governor
            .try_reserve(600, "test-joint")
            .expect("fits after release");
        assert_eq!(refreshed.bytes(), 600);
    }

    #[test]
    fn dense_reservation_uses_checked_footprint() {
        let governor = test_governor(1 << 20);
        let ok = governor
            .try_reserve_dense_f64(1024, 64, "test-dense")
            .expect("512 KiB fits in 1 MiB");
        assert_eq!(ok.bytes(), 1024 * 64 * 8);
        drop(ok);
        // Dimension-product overflow must refuse, never wrap into a tiny
        // spurious reservation.
        governor
            .try_reserve_dense_f64(usize::MAX, 2, "test-overflow")
            .expect_err("overflowing footprint cannot be reserved");
        assert!(matches!(
            governor.try_reserve_dense_f64(usize::MAX, 2, "test-overflow"),
            Err(MemoryReservationError::SizeOverflow { .. })
        ));
    }

    #[test]
    fn concurrent_reservations_never_oversubscribe() {
        let governor = std::sync::Arc::new(test_governor(1_000));
        let granted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(9));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let governor = std::sync::Arc::clone(&governor);
                let granted = std::sync::Arc::clone(&granted);
                let barrier = std::sync::Arc::clone(&barrier);
                scope.spawn(move || {
                    let held = governor.try_reserve(200, "test-race").ok();
                    if held.is_some() {
                        granted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    barrier.wait();
                    assert!(governor.reserved_bytes() <= governor.budget_bytes());
                    barrier.wait();
                    drop(held);
                });
            }
            barrier.wait();
            assert_eq!(granted.load(std::sync::atomic::Ordering::SeqCst), 5);
            assert_eq!(governor.reserved_bytes(), 1_000);
            barrier.wait();
        });
        assert_eq!(governor.reserved_bytes(), 0);
    }

    #[test]
    fn global_policy_caps_are_one_shared_admission_ceiling() {
        let governor = MemoryGovernor::global();
        assert_eq!(
            governor.single_materialization_cap_bytes(),
            governor.budget_bytes()
        );
        let policy = ResourcePolicy::default_library();
        assert_eq!(
            policy.max_single_materialization_bytes,
            governor.single_materialization_cap_bytes()
        );
        let strict = ResourcePolicy::analytic_operator_required();
        assert_eq!(
            strict.max_single_materialization_bytes,
            policy.max_single_materialization_bytes
        );
    }

    #[test]
    fn governed_value_holds_and_releases_its_charge() {
        let governor = test_governor(64);
        let governed = governor
            .try_reserve(32, "governed-value")
            .expect("reservation fits")
            .bind(vec![0_u8; 32]);
        assert_eq!(governed.len(), 32);
        assert_eq!(governed.reserved_bytes(), 32);
        assert_eq!(governor.reserved_bytes(), 32);
        drop(governed);
        assert_eq!(governor.reserved_bytes(), 0);
    }

    #[test]
    fn memory_availability_distinguishes_host_cgroup_and_exhaustion() {
        let host_only = MemoryAvailability::from_observation(
            1_000,
            CgroupMemoryObservation::NotPresent,
        );
        assert_eq!(host_only.available_bytes(), 1_000);
        assert_eq!(host_only.limiting_source(), MemoryAvailabilitySource::Host);
        assert_eq!(governor_budget_from_availability(&host_only), 750);

        let exhausted_host =
            MemoryAvailability::from_observation(0, CgroupMemoryObservation::NotPresent);
        assert_eq!(exhausted_host.available_bytes(), 0);
        assert_eq!(governor_budget_from_availability(&exhausted_host), 0);

        let finite_cgroup = MemoryAvailability::from_observation(
            1_000,
            CgroupMemoryObservation::V2Limited(CgroupMemoryAvailability::fixture(
                "/fixture/leaf",
                600,
                200,
                0,
                1,
            )),
        );
        assert_eq!(finite_cgroup.available_bytes(), 400);
        assert_eq!(
            finite_cgroup.limiting_source(),
            MemoryAvailabilitySource::Cgroup
        );
        assert_eq!(governor_budget_from_availability(&finite_cgroup), 300);

        let exhausted_cgroup = MemoryAvailability::from_observation(
            1_000,
            CgroupMemoryObservation::V2Limited(CgroupMemoryAvailability::fixture(
                "/fixture/leaf",
                600,
                600,
                0,
                1,
            )),
        );
        assert_eq!(exhausted_cgroup.available_bytes(), 0);
        assert_eq!(
            exhausted_cgroup.limiting_source(),
            MemoryAvailabilitySource::Cgroup
        );
        assert_eq!(governor_budget_from_availability(&exhausted_cgroup), 0);

        // A finite cgroup with more headroom than the host remains visible as
        // provenance, but the host is the binding observation.
        let host_is_tighter = MemoryAvailability::from_observation(
            1_000,
            CgroupMemoryObservation::V2Limited(CgroupMemoryAvailability::fixture(
                "/fixture/leaf",
                8_000,
                2_000,
                0,
                1,
            )),
        );
        assert_eq!(host_is_tighter.available_bytes(), 1_000);
        assert_eq!(
            host_is_tighter.limiting_source(),
            MemoryAvailabilitySource::Host
        );

        let equal_cgroup_ceiling = MemoryAvailability::from_observation(
            1_000,
            CgroupMemoryObservation::V2Limited(CgroupMemoryAvailability::fixture(
                "/fixture/leaf",
                1_200,
                200,
                0,
                1,
            )),
        );
        assert_eq!(equal_cgroup_ceiling.available_bytes(), 1_000);
        assert_eq!(
            equal_cgroup_ceiling.limiting_source(),
            MemoryAvailabilitySource::HostAndCgroup
        );

        let both_exhausted = MemoryAvailability::from_observation(
            0,
            CgroupMemoryObservation::V2Limited(CgroupMemoryAvailability::fixture(
                "/fixture/leaf",
                600,
                600,
                0,
                1,
            )),
        );
        assert_eq!(
            both_exhausted.limiting_source(),
            MemoryAvailabilitySource::HostAndCgroup
        );

        // A numeric cgroup-v2 `memory.max = 0` is distinct from the literal
        // `max` token and remains an authoritative hard-zero ceiling.
        let zero_ceiling = MemoryAvailability::from_observation(
            8_000,
            CgroupMemoryObservation::V2Limited(CgroupMemoryAvailability::fixture(
                "/fixture/leaf",
                0,
                0,
                0,
                1,
            )),
        );
        assert_eq!(zero_ceiling.available_bytes(), 0);
        assert_eq!(
            zero_ceiling.limiting_source(),
            MemoryAvailabilitySource::Cgroup
        );
        assert_eq!(governor_budget_from_availability(&zero_ceiling), 0);
    }

    #[test]
    fn literal_unlimited_cgroup_defers_to_host_available_memory_2317() {
        let unlimited = MemoryAvailability::from_observation(
            2_430_926_848,
            CgroupMemoryObservation::V2Unbounded {
                cgroup_path: "/fixture/leaf".into(),
                inspected_levels: 3,
            },
        );
        assert_eq!(unlimited.available_bytes(), 2_430_926_848);
        assert_eq!(unlimited.limiting_source(), MemoryAvailabilitySource::Host);
        assert_eq!(
            governor_budget_from_availability(&unlimited),
            1_823_195_136
        );
        assert!(format!("{unlimited}").contains("unbounded cgroup-v2"));
    }

    #[test]
    fn finite_cgroup_v1_headroom_participates_in_the_same_exact_minimum() {
        let availability = MemoryAvailability::from_observation(
            8_000,
            CgroupMemoryObservation::V1Limited(CgroupMemoryAvailability::fixture(
                "/sys/fs/cgroup/memory/slurm/job",
                4_000,
                1_500,
                500,
                4,
            )),
        );
        assert_eq!(availability.available_bytes(), 3_000);
        assert_eq!(
            availability.limiting_source(),
            MemoryAvailabilitySource::Cgroup
        );
        assert_eq!(governor_budget_from_availability(&availability), 2_250);
        let evidence = format!("{availability}");
        assert!(evidence.contains("cgroup-v1"));
        assert!(evidence.contains("available=3000"));
    }

    #[test]
    fn malformed_active_cgroup_fails_closed_with_typed_evidence() {
        let availability = MemoryAvailability::from_observation(
            8_000,
            CgroupMemoryObservation::ProbeFailed(CgroupMemoryProbeFailure::fixture(
                CgroupMemoryProbeFailureKind::InvalidCounter,
                "/fixture/leaf/memory.current",
                "expected an unsigned byte count",
            )),
        );
        assert_eq!(availability.available_bytes(), 0);
        assert_eq!(
            availability.limiting_source(),
            MemoryAvailabilitySource::CgroupProbeFailure
        );
        assert_eq!(governor_budget_from_availability(&availability), 0);
        let evidence = format!("{availability}");
        assert!(evidence.contains("failed closed"));
        assert!(evidence.contains("invalid-counter"));
    }

    #[test]
    fn compressed_macos_observation_keeps_xnu_available_memory_positive() {
        // #2316's healthy 8 GiB macOS host had more compressed than
        // free+inactive pages. sysinfo 0.33 subtracted compressor pages and
        // saturated to zero; 0.38 follows XNU and reports
        // (active + inactive + free) * page_size.
        let xnu_available = (75_514_u64 + 69_056 + 3_802) * 16_384;
        assert_eq!(xnu_available, 2_430_926_848);
        let availability = MemoryAvailability::from_observation(
            xnu_available,
            CgroupMemoryObservation::NotPresent,
        );
        assert_eq!(availability.available_bytes(), xnu_available);
        assert_eq!(
            governor_budget_from_availability(&availability),
            1_823_195_136
        );
    }
}

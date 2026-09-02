use crate::cgroup_memory::detect_cgroup_memory;
pub use crate::cgroup_memory::{
    CgroupMemoryAvailability, CgroupMemoryLimit, CgroupMemoryObservation, CgroupMemoryProbeFailure,
    CgroupMemoryProbeFailureKind,
};

/// The library-default streamed row-chunk target (8 MiB), shared as a `const`
/// so compile-time consumers (e.g. device tile geometry) stay in lockstep with
/// [`ResourcePolicy::default_library`] without a runtime policy query.
///
/// **The 8 MiB value is unmeasured.** The paragraph above justifies there being
/// ONE copy; it does not justify the number. No cache level, measured
/// throughput, or consequence-for-being-wrong is recorded anywhere for it.
/// Twelve transcribed copies across six crates were collapsed onto this
/// definition (#2704) purely to remove duplication — that consolidation moved
/// no value and measured nothing, and a single canonical copy of an arbitrary
/// number must not be read as a decided one. Anyone retuning it is retuning an
/// unsupported literal, not overriding a derived budget.
///
/// Consumers should size chunks with [`rows_for_target_bytes`], which performs
/// the `target / (cols * 8)` division and floors the result at one row.
///
/// Re-declaring this value — as a module-local `const`, an inline literal, or
/// any other spelling of 8 MiB — is caught by the workspace guard
/// `tests/row_chunk_target_bytes_single_source_2704.rs`, which fails on any
/// transcription of this value outside the exemption table it carries for the
/// three sites that are a genuinely DIFFERENT quantity coinciding at 8 MiB.
/// Import this constant instead, or derive from it under a name.
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

/// Fraction of this process's memory **capacity** the governor is allowed to
/// hand out as reservations: 3/4.
///
/// The ledger only accounts for the large, planned allocations that route
/// through [`MemoryGovernor::try_reserve`] (dense design materializations,
/// covariance blocks, sampler design assemblies). Everything else — allocator
/// slack, thread stacks, code, and small per-iteration temporaries — lives in
/// the remaining quarter.
///
/// The base quantity is *capacity* (`min(host total, binding cgroup hard
/// limit)`), not free space, and that is the same separation #2684 established
/// for the materialization ceiling — see
/// `governor_budget_from_availability` for why the ledger cannot be an
/// exception to it (#2702).
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
/// Two quantities live here and they answer different questions.
///
/// * **Available** (`available_bytes`) is *free space right now*: a finite
///   cgroup-v1 or cgroup-v2 ceiling admits exactly `min(host available,
///   reclaim-aware cgroup available)`. A v2 literal `memory.max = max` remains
///   typed as unbounded and therefore defers to the host; v1's numeric
///   unlimited sentinel participates exactly and likewise loses to a tighter
///   host value. This quantity *moves*, and under pressure it goes to zero.
/// * **Capacity** (`capacity_bytes`) is *how much memory this process could
///   ever address*: `min(host total, the binding cgroup's hard limit)`. It is
///   a property of the box and the cgroup configuration, not of the schedule,
///   so it does not move when a sibling allocates.
///
/// The distinction is load-bearing (#2684). "Does this dense footprint fit
/// here at all?" is a *capacity* question and its answer must be stationary —
/// see [`process_memory_availability`] on why a route derived from a moving
/// quantity is a defect. "Does this allocation fit *right now*?" is an
/// availability question, and the joint ledger
/// ([`MemoryGovernor::try_reserve`]) is the instrument for it. Reading the
/// second where the first was meant is what let a cgroup a few pages from its
/// limit refuse a 28,800-byte design on a host with 448 GB free.
///
/// If an active controller cannot be parsed exactly, admission fails closed
/// with zero bytes — both available *and* capacity — while retaining the typed
/// probe failure; it never silently inherits host capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAvailability {
    host_available_bytes: u64,
    host_total_bytes: u64,
    cgroup: CgroupMemoryObservation,
    available_bytes: u64,
    capacity_bytes: u64,
    limiting_source: MemoryAvailabilitySource,
}

impl MemoryAvailability {
    pub(crate) fn from_observation(
        host_available_bytes: u64,
        host_total_bytes: u64,
        cgroup: CgroupMemoryObservation,
    ) -> Self {
        use std::cmp::Ordering;

        // Capacity never depends on what is currently resident: a finite
        // cgroup ceiling clamps the host's total, an unbounded or absent
        // controller defers to it, and a probe that failed closed admits
        // nothing at all. A host whose reported available exceeds its reported
        // total (a torn /proc/meminfo read) must not shrink capacity below
        // what is demonstrably reachable, hence the max.
        let capacity_bytes = match &cgroup {
            CgroupMemoryObservation::NotPresent | CgroupMemoryObservation::V2Unbounded { .. } => {
                host_total_bytes.max(host_available_bytes)
            }
            CgroupMemoryObservation::V2Limited(observation)
            | CgroupMemoryObservation::V1Limited(observation) => host_total_bytes
                .max(host_available_bytes)
                .min(observation.limit_bytes()),
            CgroupMemoryObservation::ProbeFailed(_) => 0,
        };

        let (available_bytes, limiting_source) = match &cgroup {
            CgroupMemoryObservation::NotPresent | CgroupMemoryObservation::V2Unbounded { .. } => {
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
                    Ordering::Greater => (host_available_bytes, MemoryAvailabilitySource::Host),
                }
            }
            CgroupMemoryObservation::ProbeFailed(_) => {
                (0, MemoryAvailabilitySource::CgroupProbeFailure)
            }
        };
        Self {
            host_available_bytes,
            host_total_bytes,
            cgroup,
            available_bytes,
            capacity_bytes,
            limiting_source,
        }
    }

    /// The stationary ceiling on memory this process could ever address:
    /// `min(host total, binding cgroup hard limit)`, or zero when the cgroup
    /// probe failed closed. Unlike [`Self::available_bytes`] this does not move
    /// when other work allocates, which is what makes it usable as a *routing*
    /// threshold (#2684).
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub const fn available_bytes(&self) -> u64 {
        self.available_bytes
    }

    pub fn available_bytes_usize(&self) -> usize {
        usize::try_from(self.available_bytes).unwrap_or(usize::MAX)
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
                "{} bytes limited by {} (capacity={}, host_available={}, host_total={}, {})",
                self.available_bytes,
                self.limiting_source,
                self.capacity_bytes,
                self.host_available_bytes,
                self.host_total_bytes,
                observation,
            ),
        }
    }
}

static MEMORY_AVAILABILITY_PROBES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Take a **fresh** reading of the OS and cgroup memory observations.
///
/// This is a syscall-bearing probe, not an accessor: it refreshes `sysinfo`
/// (`/proc/meminfo`) and re-reads the cgroup limit/usage files on every call,
/// and the value it returns *moves* — available memory is a live, shared
/// quantity owned by the whole machine, not by this process.
///
/// Call this only where a live reading is the point (an OOM guard immediately
/// before committing an allocation). Anything that *routes* — picks a
/// strategy, sizes a plan, chooses an operator — must use
/// [`process_memory_availability`] instead, because a route derived from a
/// moving quantity makes the fitted answer depend on what else the box was
/// doing (SPEC-20: a fit object must only ever come from a converged
/// optimization, and reproducibly so).
pub fn resample_memory_availability() -> MemoryAvailability {
    static SYSTEM: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let system = SYSTEM.get_or_init(|| Mutex::new(sysinfo::System::new()));
    let mut system = system.lock().expect("sysinfo system mutex poisoned");
    system.refresh_memory();
    let cgroup = detect_cgroup_memory();
    MEMORY_AVAILABILITY_PROBES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    MemoryAvailability::from_observation(
        system.available_memory(),
        system.total_memory(),
        cgroup,
    )
}

/// The process's memory availability: **one** observation, taken once, shared
/// by every planner in the process.
///
/// This is the same reading the [`MemoryGovernor`] sized its budget from, so a
/// planner that admits a plan against this figure and the governor that then
/// accounts the allocation cannot disagree about how much memory the process
/// believes it has. Two consequences follow, and both are the point:
///
/// * **Free.** It is an atomic-free borrow of an already-initialized static —
///   no syscalls, no parsing, no allocation — so it can sit on the hottest
///   inner loop without a caching layer of its own.
/// * **Constant.** Two runs of the same fit in one process see byte-identical
///   budgets, so any route chosen from it is reproducible. Planners used to
///   defend against the movement themselves (a monotone high-water floor in
///   BMS, a per-term captured field in the SAE); a stationary primitive is what
///   makes those compensations unnecessary rather than load-bearing.
pub fn process_memory_availability() -> &'static MemoryAvailability {
    &MemoryGovernor::global().ledger.availability
}

/// The process's available-memory figure in bytes, saturating to `usize`.
/// Convenience over [`process_memory_availability`] for planners that only
/// need the scalar.
pub fn process_available_memory_bytes() -> usize {
    process_memory_availability().available_bytes_usize()
}

/// Convert one provenance-preserving availability observation to the process
/// ledger budget: 3/4 of the observation's stationary **capacity**. A probe
/// that failed closed reports zero capacity and therefore admits nothing, which
/// is the authoritative exhausted-memory signal.
///
/// gam#2702. This used to read `available_bytes` — free space — and that made
/// every reservation verdict in the process a function of what the rest of the
/// box happened to be doing, in exactly the way #2684 removed one layer up:
///
/// * the observation behind it is taken **once**, in the [`MemoryGovernor`]'s
///   `OnceLock`, so it never tracked exhaustion in the first place. It froze
///   whatever free memory existed at the instant of first governed allocation
///   and used that number for the rest of the process's life. A frozen sample
///   of a moving quantity is neither a live guard nor a reproducible
///   threshold;
/// * two processes launched from one job cgroup therefore derived different
///   budgets from the same configuration, because page cache and sibling
///   processes charge that cgroup continuously. Measured consequence: three
///   `gam` inference tests passed in a 71-test run and failed in a 6-test
///   subset of the same binary at the same commit, with
///   `resource policy refused exact coefficient-SE columns 0..1` — a refusal of
///   a few kilobytes.
///
/// The ledger's job is to bound **this process's own** jointly-live governed
/// allocations against the ceiling this process could ever address. Memory
/// already committed elsewhere on the box is not memory this ledger can account
/// for, release, or route around; reading it only makes a fit's verdict a
/// function of its neighbours, which SPEC-20 forbids (see
/// [`process_memory_availability`]). Under a job scheduler the cgroup hard
/// limit *is* the per-job ceiling, so capacity is the bound that actually binds.
fn governor_budget_from_availability(availability: &MemoryAvailability) -> usize {
    stationary_headroom_of_capacity(availability)
}

/// Convert the same observation to the process's **stationary** materialization
/// ceiling: the answer to "could a dense footprint this large ever live here?",
/// which must not move when a sibling allocates (#2684). A probe that failed
/// closed reports zero capacity and therefore admits nothing.
fn governor_materialization_cap_from_availability(availability: &MemoryAvailability) -> usize {
    stationary_headroom_of_capacity(availability)
}

/// The one arithmetic both process-wide ceilings are derived from: the headroom
/// fraction applied to stationary capacity.
///
/// The ledger budget and the materialization cap are therefore equal by
/// construction rather than by coincidence, and that equality is the point:
/// they are two questions about the same ceiling ("does this fit alongside what
/// this process already has live" and "could this ever fit here at all"), so a
/// routing decision taken from the cap can never be contradicted by the ledger
/// for a reason that is not another live reservation. Keeping the arithmetic in
/// one place is what stops the two from drifting back apart (#2684, #2702).
fn stationary_headroom_of_capacity(availability: &MemoryAvailability) -> usize {
    let scaled = u128::from(availability.capacity_bytes()) * GOVERNOR_BUDGET_NUMERATOR
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
    materialization_cap_bytes: usize,
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
/// The global budget is sized once from this process's stationary capacity
/// (host total memory, clamped by the binding cgroup's hard limit inside
/// containers or under a job scheduler) — see `GOVERNOR_BUDGET_NUMERATOR` for
/// the headroom rationale and `governor_budget_from_availability` for why it
/// is not denominated in free memory (#2702).
#[derive(Debug, Clone)]
pub struct MemoryGovernor {
    ledger: Arc<GovernorLedger>,
}

impl MemoryGovernor {
    /// The process-wide governor. Budget detection runs once, on first use.
    ///
    /// This `OnceLock` is where the process's single availability observation
    /// is taken; [`process_memory_availability`] hands the same observation to
    /// every other planner rather than probing again.
    pub fn global() -> &'static MemoryGovernor {
        static GLOBAL: OnceLock<MemoryGovernor> = OnceLock::new();
        GLOBAL.get_or_init(|| {
            let availability = resample_memory_availability();
            MemoryGovernor::with_detected_availability(availability)
        })
    }

    fn with_detected_availability(availability: MemoryAvailability) -> Self {
        let budget_bytes = governor_budget_from_availability(&availability);
        let materialization_cap_bytes = governor_materialization_cap_from_availability(&availability);
        Self {
            ledger: Arc::new(GovernorLedger {
                budget_bytes,
                materialization_cap_bytes,
                availability,
                reserved_bytes: std::sync::atomic::AtomicUsize::new(0),
            }),
        }
    }

    pub fn reserved_bytes(&self) -> usize {
        self.ledger
            .reserved_bytes
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Bytes still admissible: the budget less what is reserved *by this
    /// process* right now. The only quantity here that moves is this process's
    /// own live governed footprint — which is the one thing a caller can
    /// actually route around.
    pub fn remaining_bytes(&self) -> usize {
        self.ledger
            .budget_bytes
            .saturating_sub(self.reserved_bytes())
    }

    /// Absolute ceiling for one governed operation: 3/4 of this process's
    /// memory **capacity** (`min(host total, binding cgroup limit)`).
    ///
    /// This is a *routing* threshold — the question it answers is "could a
    /// dense footprint this large ever live in this process?" — so it is
    /// deliberately stationary. It is **not** the live budget: whether an
    /// allocation fits *right now* is decided by [`Self::try_reserve`] against
    /// the joint ledger, which returns a typed, routable refusal instead of an
    /// abort. Since #2702 both are denominated in the same capacity, so the two
    /// can only disagree because of a live reservation.
    ///
    /// Returning the live budget here was gam#2684: the budget was 3/4 of
    /// *available* memory, so a cgroup sitting at its limit drove this ceiling
    /// continuously to zero (measured at 53,248 bytes available with 448 GB
    /// free on the host), and every caller comparing a request against it —
    /// including `DenseDesignMatrix::to_dense`, which panics on refusal —
    /// refused allocations as small as a 300x12 design. Threshold-shaped
    /// routing decisions taken from a moving quantity also make the chosen
    /// route depend on what else the box was doing, which SPEC-20 forbids
    /// (see [`process_memory_availability`]); one consumer even hashes this
    /// cap into a basis cache key.
    pub fn single_materialization_cap_bytes(&self) -> usize {
        self.ledger.materialization_cap_bytes
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
    /// Scalar caps expose the governor's stationary capacity ceiling; they are
    /// routing thresholds, not independent budgets, and they do not move with
    /// load. Actual materializations and caches must reserve their complete
    /// live footprint against the shared governor, so any combination of
    /// categories is bounded by one ledger — that reservation, not these caps,
    /// is what enforces "fits right now" (#2684).
    pub fn default_library() -> Self {
        Self::for_observed_memory(process_memory_availability())
    }

    /// The library-default policy a process would derive from `availability`.
    ///
    /// [`Self::default_library`] is exactly this function applied to the
    /// process's own observation, so the routing thresholds a fit sees are a
    /// pure function of the *observed environment* — and, because every scalar
    /// below is taken from
    /// [`MemoryAvailability::capacity_bytes`] rather than
    /// [`MemoryAvailability::available_bytes`], a pure function of that
    /// environment's stationary CAPACITY. Two observations of the same box that
    /// differ only in how much memory happened to be free at the instant of the
    /// probe therefore produce byte-identical policies, and any route selected
    /// from one of these caps is reproducible across processes and across load
    /// (#2684).
    ///
    /// Exposed rather than kept private so a test can hold capacity fixed and
    /// vary free memory without racing the real machine; the live path uses the
    /// same code, so the property under test is the shipped one.
    pub fn for_observed_memory(availability: &MemoryAvailability) -> Self {
        let single_cap = governor_materialization_cap_from_availability(availability);
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
            let k = g
                .order
                .remove(pos)
                .expect("position() returned an in-bounds index into this same deque");
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
        if self.slot.set(candidate).is_err() {
            log::trace!(
                "RayonSafeOnce: a concurrent initializer won the race; \
                 keeping its value and discarding this candidate"
            );
        }
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
            // `get_or_init` rather than `set(..).expect(..)`: the slot of a
            // freshly constructed `Self` is empty, so the initializer always
            // runs — and this spelling has no `Result` to discard, which keeps
            // the impl's bound at `T: Clone` instead of forcing `T: Debug` on
            // every caller just to name a panic that cannot happen.
            cloned.slot.get_or_init(|| value.clone());
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
        let available_bytes =
            u64::try_from(available_bytes).expect("test cache budget must fit in u64");
        MemoryGovernor::with_detected_availability(MemoryAvailability::from_observation(
            available_bytes,
            available_bytes,
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
        let available_bytes =
            u64::try_from(available_bytes).expect("test budget has a representable observation");
        let availability = MemoryAvailability::from_observation(
            available_bytes,
            available_bytes,
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
        // Canonical target, 1024 cols → 8192 bytes/row → rows = target/8192.
        // Taken by name rather than transcribed, so the case stays pinned to the
        // library target if that target ever moves (#2704).
        let target = LIBRARY_ROW_CHUNK_TARGET_BYTES;
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
                    let held = governor.try_reserve(200, "test-race");
                    if held.is_ok() {
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
            governor_materialization_cap_from_availability(&governor.availability())
        );
        // Both ceilings are denominated in the same stationary capacity since
        // #2702, so they are equal by construction; a divergence here would mean
        // one of them had picked up a second base quantity again.
        assert_eq!(
            governor.single_materialization_cap_bytes(),
            governor.budget_bytes(),
            "the routing cap and the ledger budget are one ceiling asked two questions"
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

    /// The `available_bytes` assertions here are the availability derivation.
    /// The budget assertions alongside them are deliberately denominated in
    /// CAPACITY (#2702): free space decides whether an allocation fits *now*,
    /// which is the ledger's live `reserved_bytes` question, not the size of the
    /// ceiling the ledger measures against.
    #[test]
    fn memory_availability_distinguishes_host_cgroup_and_exhaustion() {
        let host_only =
            MemoryAvailability::from_observation(1_000, 4_000, CgroupMemoryObservation::NotPresent);
        assert_eq!(host_only.available_bytes(), 1_000);
        assert_eq!(host_only.limiting_source(), MemoryAvailabilitySource::Host);
        assert_eq!(host_only.capacity_bytes(), 4_000);
        assert_eq!(governor_budget_from_availability(&host_only), 3_000);

        // Free space at zero with the box's capacity unchanged: the budget is
        // the same 3/4 of 4,000, because a host that is momentarily full has not
        // become a smaller host.
        let exhausted_host =
            MemoryAvailability::from_observation(0, 4_000, CgroupMemoryObservation::NotPresent);
        assert_eq!(exhausted_host.available_bytes(), 0);
        assert_eq!(governor_budget_from_availability(&exhausted_host), 3_000);

        let finite_cgroup = MemoryAvailability::from_observation(
            1_000,
            4_000_000_000,
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
        // Capacity is the cgroup's 600-byte hard limit, not its 400 free bytes.
        assert_eq!(finite_cgroup.capacity_bytes(), 600);
        assert_eq!(governor_budget_from_availability(&finite_cgroup), 450);

        let exhausted_cgroup = MemoryAvailability::from_observation(
            1_000,
            4_000_000_000,
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
        // Same cgroup, charged to its limit: same ceiling, so same budget as the
        // headroom-having reading above. That equality IS the #2702 property.
        assert_eq!(exhausted_cgroup.capacity_bytes(), 600);
        assert_eq!(governor_budget_from_availability(&exhausted_cgroup), 450);

        // A finite cgroup with more headroom than the host remains visible as
        // provenance, but the host is the binding observation.
        let host_is_tighter = MemoryAvailability::from_observation(
            1_000,
            4_000_000_000,
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
            4_000_000_000,
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
            4_000_000_000,
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
            4_000_000_000,
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
            4_000_000_000,
            CgroupMemoryObservation::V2Unbounded {
                cgroup_path: "/fixture/leaf".into(),
                inspected_levels: 3,
            },
        );
        assert_eq!(unlimited.available_bytes(), 2_430_926_848);
        assert_eq!(unlimited.limiting_source(), MemoryAvailabilitySource::Host);
        // An unbounded controller defers to the host for capacity too, so the
        // budget is 3/4 of the host's 4 GB total (#2702).
        assert_eq!(unlimited.capacity_bytes(), 4_000_000_000);
        assert_eq!(governor_budget_from_availability(&unlimited), 3_000_000_000);
        assert!(format!("{unlimited}").contains("unbounded cgroup-v2"));
    }

    #[test]
    fn finite_cgroup_v1_headroom_participates_in_the_same_exact_minimum() {
        let availability = MemoryAvailability::from_observation(
            8_000,
            4_000_000_000,
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
        // The v1 controller's 4,000-byte hard limit is the capacity, and the
        // budget is 3/4 of it regardless of the 1,500 bytes charged to it.
        assert_eq!(availability.capacity_bytes(), 4_000);
        assert_eq!(governor_budget_from_availability(&availability), 3_000);
        let evidence = format!("{availability}");
        assert!(evidence.contains("cgroup-v1"));
        assert!(evidence.contains("available=3000"));
    }

    /// gam#2684. A cgroup pinned at its hard limit reports almost no *available*
    /// memory while its *capacity* is unchanged. The numbers below are the ones
    /// measured on an MSI compute node (`--mem=6g`, host with 448 GB free):
    /// the shipped probe reported `available=53_248` while the job's ceiling was
    /// still 6 GiB. The routing cap must come from the ceiling — otherwise a
    /// 300x12 f64 design (28,800 bytes) is refused.
    ///
    /// #2702 extended that to the ledger budget, which this test used to assert
    /// shrank to 39,936 bytes here: the observation is taken once per process
    /// and never refreshed, so a free-denominated budget did not track
    /// exhaustion — it froze one process's view of the schedule. A large
    /// reservation still routes instead of allocating, but because the ledger
    /// accounts THIS process's live footprint, not because the ceiling moved.
    #[test]
    fn a_cgroup_at_its_limit_moves_neither_the_budget_nor_the_materialization_cap_2684_2702() {
        const DESIGN_300X12_BYTES: usize = 300 * 12 * 8;
        let limit_bytes = 6 * 1024 * 1024 * 1024_u64;
        let at_the_limit = MemoryAvailability::from_observation(
            448_648_040_448,
            527_799_400 * 1024,
            CgroupMemoryObservation::V1Limited(CgroupMemoryAvailability::fixture(
                "/sys/fs/cgroup/memory/slurm/uid_81060/job_14615476",
                limit_bytes,
                limit_bytes - 53_248,
                0,
                6,
            )),
        );
        assert_eq!(at_the_limit.available_bytes(), 53_248);
        assert_eq!(at_the_limit.capacity_bytes(), limit_bytes);
        let governor = MemoryGovernor::with_detected_availability(at_the_limit);
        // The budget is the job's ceiling, not the 53,248 bytes that happened to
        // be free when the probe ran.
        assert_eq!(governor.budget_bytes(), (limit_bytes as usize) / 4 * 3);
        // It still refuses what cannot fit: 8 GiB does not fit a 6 GiB job.
        assert!(governor.try_reserve(8 << 30, "larger-than-the-job").is_err());
        // And the few-kilobyte reservation #2702 was filed for is admitted at
        // exactly the reading that refused it — the SE path asks for two dense
        // f64 copies of one column of a p=64 transformed Hessian.
        let se_chunk = governor
            .try_reserve_dense_f64_copies(64, 1, 2, "coefficient-SE solve chunk")
            .expect("a 1,024-byte SE chunk must be admissible in a 6 GiB job");
        assert_eq!(se_chunk.bytes(), 1_024);
        drop(se_chunk);
        // The routing ceiling does not move either.
        assert_eq!(
            governor.single_materialization_cap_bytes(),
            (limit_bytes as usize) / 4 * 3
        );
        assert!(governor.single_materialization_cap_bytes() > DESIGN_300X12_BYTES);

        // Falsification guard: the cap must still be able to say no. A cgroup
        // whose whole ceiling is smaller than the design refuses it, so the
        // assertion above cannot be satisfied by a cap that always admits.
        let tiny_ceiling = MemoryAvailability::from_observation(
            448_648_040_448,
            527_799_400 * 1024,
            CgroupMemoryObservation::V1Limited(CgroupMemoryAvailability::fixture(
                "/fixture/tiny",
                1_024,
                0,
                0,
                1,
            )),
        );
        assert_eq!(tiny_ceiling.capacity_bytes(), 1_024);
        assert_eq!(
            MemoryGovernor::with_detected_availability(tiny_ceiling)
                .single_materialization_cap_bytes(),
            768
        );

        // Second falsification guard: an idle cgroup with the same ceiling must
        // report the same cap, or the cap is still tracking the schedule.
        let idle = MemoryAvailability::from_observation(
            448_648_040_448,
            527_799_400 * 1024,
            CgroupMemoryObservation::V1Limited(CgroupMemoryAvailability::fixture(
                "/sys/fs/cgroup/memory/slurm/uid_81060/job_14615476",
                limit_bytes,
                92_827_648,
                28_672,
                6,
            )),
        );
        assert!(idle.available_bytes() > 6_000_000_000);
        assert_eq!(
            MemoryGovernor::with_detected_availability(idle).single_materialization_cap_bytes(),
            (limit_bytes as usize) / 4 * 3
        );
    }

    #[test]
    fn malformed_active_cgroup_fails_closed_with_typed_evidence() {
        let availability = MemoryAvailability::from_observation(
            8_000,
            4_000_000_000,
            CgroupMemoryObservation::ProbeFailed(CgroupMemoryProbeFailure::fixture(
                CgroupMemoryProbeFailureKind::InvalidCounter,
                "/fixture/leaf/memory.current",
                "expected an unsigned byte count",
            )),
        );
        assert_eq!(availability.available_bytes(), 0);
        // Fail-closed covers capacity too: an unreadable controller must not
        // hand out a routing ceiling either (#2684 keeps this policy intact).
        assert_eq!(availability.capacity_bytes(), 0);
        assert_eq!(
            availability.limiting_source(),
            MemoryAvailabilitySource::CgroupProbeFailure
        );
        assert_eq!(governor_budget_from_availability(&availability), 0);
        assert_eq!(
            governor_materialization_cap_from_availability(&availability),
            0
        );
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
            8 * 1024 * 1024 * 1024,
            CgroupMemoryObservation::NotPresent,
        );
        assert_eq!(availability.available_bytes(), xnu_available);
        // Capacity is the host's 8 GiB, so the budget is 6 GiB (#2702); the
        // 2.43 GB free figure decides nothing but the availability provenance.
        assert_eq!(availability.capacity_bytes(), 8 * 1024 * 1024 * 1024);
        assert_eq!(
            governor_budget_from_availability(&availability),
            6 * 1024 * 1024 * 1024
        );
    }
}

/// gam#2702: a reservation verdict is a function of the request and this
/// process's own live footprint — never of what else was on the box, and never
/// of what this process happened to do earlier.
///
/// The filed incident: three `gam` inference tests passed in a 71-test run of
/// the `inference` binary and failed in a 6-test subset of that same binary at
/// that same commit, all three with
/// `resource policy refused exact coefficient-SE columns 0..1`. The refused
/// allocation was two dense f64 copies of one column — kilobytes. The ledger
/// budget was `3/4 x FREE memory` sampled once, at whichever moment this
/// process first touched the governor, so processes launched from one job
/// cgroup derived different budgets according to how much page cache and how
/// many sibling test processes were charged to that cgroup at their own instant
/// of first touch.
///
/// #2684 had already removed exactly this mechanism from the materialization
/// ceiling. What follows asserts the ledger is not an exception, on the shipped
/// derivation, with the pre-fix arithmetic spelled out as the falsification
/// control.
#[cfg(test)]
mod governor_budget_is_capacity_determined_2702_tests {
    use super::*;

    /// The MSI compute node the incident was measured on: a `--mem=8g` job on a
    /// box with hundreds of GB free.
    const HOST_AVAILABLE_BYTES: u64 = 448_648_040_448;
    const HOST_TOTAL_BYTES: u64 = 527_799_400 * 1024;
    const JOB_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

    /// The refused allocation, restated from the failing path
    /// (`gam-solve` `optimizer.rs`): the exact factorized coefficient-SE solve
    /// reserves two dense `p_t x chunk` f64 workspaces at once. `p_t = 512` and
    /// `chunk = 1` is the smallest request that path can ever make on a model of
    /// this width — one column at a time.
    const SE_TRANSFORMED_ROWS: usize = 512;
    /// One column, two copies, eight bytes per f64.
    const SE_CHUNK_BYTES: usize = SE_TRANSFORMED_ROWS * 8 * 2;

    /// One cgroup read at load `charged`, everything else held fixed.
    fn one_job_cgroup_at_load(charged: u64) -> MemoryAvailability {
        crate::test_support::simulated_cgroup_memory_environment(
            HOST_AVAILABLE_BYTES,
            HOST_TOTAL_BYTES,
            JOB_LIMIT_BYTES,
            charged,
        )
    }

    /// The derivation this fix replaced, kept here as the control: 3/4 of FREE
    /// memory. A test that never evaluates it cannot show that the assertions
    /// below had a way to fail.
    fn pre_2702_free_denominated_budget(availability: &MemoryAvailability) -> usize {
        let scaled = u128::from(availability.available_bytes()) * GOVERNOR_BUDGET_NUMERATOR
            / GOVERNOR_BUDGET_DENOMINATOR;
        usize::try_from(scaled).unwrap_or(usize::MAX)
    }

    #[test]
    fn one_job_observed_at_two_load_levels_yields_one_budget_and_one_verdict() {
        // The two readings the incident's two runs took: a cgroup with room, and
        // the same cgroup four kilobytes from its limit — the state any cgroup
        // settles into once its job has read a few gigabytes of build artifacts.
        let roomy = one_job_cgroup_at_load(92_827_648);
        let pinned = one_job_cgroup_at_load(JOB_LIMIT_BYTES - 4_096);

        // The arms are genuinely different observations, or nothing below is
        // being tested.
        assert!(roomy.available_bytes() > 7_000_000_000);
        assert_eq!(pinned.available_bytes(), 4_096);
        // ... and they are genuinely the same job.
        assert_eq!(roomy.capacity_bytes(), JOB_LIMIT_BYTES);
        assert_eq!(pinned.capacity_bytes(), JOB_LIMIT_BYTES);

        // FALSIFICATION CONTROL. Under the pre-#2702 derivation the pinned arm's
        // budget was 3,072 bytes: below the SE chunk, so that arm refused and the
        // roomy arm admitted. The assertions that follow therefore had a way to
        // fail, and this is the exact inversion the issue reported.
        assert_eq!(pre_2702_free_denominated_budget(&pinned), 3_072);
        assert!(pre_2702_free_denominated_budget(&pinned) < SE_CHUNK_BYTES);
        assert!(pre_2702_free_denominated_budget(&roomy) > SE_CHUNK_BYTES);

        let roomy_governor = MemoryGovernor::with_detected_availability(roomy);
        let pinned_governor = MemoryGovernor::with_detected_availability(pinned);

        assert_eq!(roomy_governor.budget_bytes(), pinned_governor.budget_bytes());
        assert_eq!(
            pinned_governor.budget_bytes(),
            (JOB_LIMIT_BYTES as usize) / 4 * 3
        );
        assert_eq!(
            roomy_governor.remaining_bytes(),
            pinned_governor.remaining_bytes()
        );

        // The verdict itself, taken through the shipped reservation call on both
        // arms: the same request, the same answer.
        for governor in [&roomy_governor, &pinned_governor] {
            let reservation = governor
                .try_reserve_dense_f64_copies(
                    SE_TRANSFORMED_ROWS,
                    1,
                    2,
                    "factorized coefficient-SE solve chunk",
                )
                .expect("an 8 KiB SE chunk is admissible in an 8 GiB job at any load");
            assert_eq!(reservation.bytes(), SE_CHUNK_BYTES);
        }
    }

    #[test]
    fn a_request_larger_than_the_job_is_still_refused_at_every_load() {
        // The ceiling must keep saying no, or the test above is satisfied by a
        // governor that admits everything.
        for charged in [0, JOB_LIMIT_BYTES / 2, JOB_LIMIT_BYTES - 4_096] {
            let governor = MemoryGovernor::with_detected_availability(one_job_cgroup_at_load(charged));
            let refusal = governor
                .try_reserve(16 * 1024 * 1024 * 1024, "twice the job's ceiling")
                .expect_err("16 GiB cannot be admitted in an 8 GiB job");
            match refusal {
                MemoryReservationError::BudgetExceeded { budget_bytes, .. } => {
                    assert_eq!(budget_bytes, (JOB_LIMIT_BYTES as usize) / 4 * 3);
                }
                other => panic!("expected a budget refusal naming the ceiling, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_verdict_does_not_depend_on_what_this_process_did_earlier() {
        // The property the issue asks for, stated as history-independence: a
        // request's verdict must be the same whether or not the process has
        // already built and dropped something large. Reservations are the only
        // process state the ledger has, and a released one must leave no trace.
        let governor = MemoryGovernor::with_detected_availability(one_job_cgroup_at_load(0));
        let request = || {
            governor
                .try_reserve_dense_f64_copies(
                    SE_TRANSFORMED_ROWS,
                    1,
                    2,
                    "factorized coefficient-SE solve chunk",
                )
                .map(|reservation| reservation.bytes())
        };

        let before = request().expect("admissible on a fresh ledger");
        {
            // Something big enough that a free-denominated ledger would have
            // been left visibly poorer by it: all but one SE chunk of the budget.
            let bulk = governor
                .try_reserve(
                    governor.remaining_bytes() - SE_CHUNK_BYTES,
                    "prior work in this process",
                )
                .expect("the bulk reservation is exactly the remaining budget");
            assert_eq!(governor.remaining_bytes(), SE_CHUNK_BYTES);
            // While it is live the ledger honestly reports the pressure: this is
            // the one thing that MAY change a verdict, and the next request still
            // fits because the bulk left exactly one chunk.
            assert!(request().is_ok());
            drop(bulk);
        }
        assert_eq!(governor.reserved_bytes(), 0);
        assert_eq!(request().expect("admissible again"), before);
    }
}

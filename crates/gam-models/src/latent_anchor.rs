//! The declared-law anchor of the marginal-slope index (gam#2923), owned here
//! for every family that anchors on a declared latent law.
//!
//! Survival writes the anchoring equation below directly. The Bernoulli
//! intercept `Σ_k w_k Φ(a + b·u_k) = Φ(q)` is the same equation under
//! `(q, α, b) → (−q, −a, −b)`, so `a(q, b) = −α(−q, −b)` with sign-mapped
//! derivatives. A family keeps its roots in [`AnchorRootCache`] slots: survival
//! two per row (entry and exit), a family with one location channel one.
//!
//! The family's defining identity is that `q(t, a)` is the MARGINAL survival
//! index: `E_z[Φ(−η) | a] = Φ(−q)`. The closed-form lowering
//! `η = q·√(1 + rᵀΣr) + rᵀz` is that identity solved under `z | a ~ N(0, Σ)`
//! and nothing else. On a declared finite law `F_a` — nodes `u_k`, weights
//! `w_k` on the scalar the slope projects the score onto — the identity is
//! instead the anchoring equation
//!
//! ```text
//!     Σ_k w_k Φ(−(α + b·u_k)) = Φ(−q),        b = s_f·g,
//! ```
//!
//! whose left side is continuous, strictly decreasing in `α` and runs from 1
//! to 0, so `α(q, b)` exists and is unique for every `(q, b)`
//! (`Descent.Portability.MarginalAnchor.exists_unique_anchor`, stated for the
//! logit link; the argument is link-agnostic). `α` replaces `q·c` in the row
//! index, and its derivatives come from implicit differentiation of the same
//! equation (`anchor_deriv_eq`): with `G(α, q, b) = Σ_k w_k Φ(−(α + b u_k)) −
//! Φ(−q)`,
//!
//! ```text
//!     α_θ = −G_θ / G_α = −Σ_k w_k φ(η_k) ∂_θ(b u_k) / Σ_k w_k φ(η_k),   η_k = α + b u_k,
//! ```
//!
//! and the second and third orders likewise.
//!
//! This module owns two things and nothing else: the root itself
//! ([`solve_anchor`], and [`solve_anchor_in_slot`] through a row's root
//! slot), and its Taylor table through order five ([`AnchorTaylor`], and
//! [`anchor_taylor_in_slot`] through the same slot), whose orders through
//! three are the implicit derivatives the direct order-two lowering reads
//! ([`AnchorDerivatives`]) and whose lift onto any [`JetField`] serves the
//! higher-order carriers (gam#2928). The Gaussian closed form is the special
//! case of both: on a Gaussian
//! quadrature grid the root is `q·√(1 + b²)` to quadrature tolerance, which is
//! what `anchor_tests::gaussian_grid_reproduces_the_closed_form` pins.

use std::sync::atomic::{AtomicU64, Ordering};
use gam_linalg::utils::splitmix64_hash;
use gam_math::nested_dual::JetField;
use gam_math::probability::{
    normal_cdf, normal_cdf_and_pdf, normal_logcdf, normal_pdf,
    signed_probit_logcdf_and_mills_ratio,
};
use gam_math::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_math::special::logaddexp;
use crate::cubic_cell_kernel::{
    DenestedCubicCell, NON_AFFINE_VALUE_TERM_OPERATIONS, TERMINAL_GL_ORDER,
    visit_cell_log_quadrature,
};
use crate::marginal_slope_shared::eval_coeff4_at;
use smallvec::SmallVec;

/// `|log-residual|` at which the anchoring equation counts as solved. The
/// residual is `log Σ_k w_k Φ(∓η_k) − log Φ(∓q)` on whichever tail is the
/// smaller, i.e. the RELATIVE error of the anchored marginal probability on
/// the side where relative error is the meaningful one. Where one float step
/// of `α` moves the residual by more than this, or the residual's own rounding
/// exceeds it, a root is held to that instead ([`anchor_residual_resolution`]).
pub(crate) const ANCHOR_LOG_RESIDUAL_TOL: f64 = 1e-13;

/// A finite law on the projected score, borrowed from wherever the family
/// materialised it: nodes ascending, positive weights summing to one, and the
/// log-weights the solver's log-space residual reads.
#[derive(Clone, Copy)]
pub(crate) struct AnchorGrid<'a> {
    pub(crate) nodes: &'a [f64],
    pub(crate) weights: &'a [f64],
    pub(crate) log_weights: &'a [f64],
}

impl<'a> AnchorGrid<'a> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// An owned finite law: what the family materialises once per fit (one for a
/// global law, one per row for a local one) and hands out as [`AnchorGrid`]s.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AnchorGridOwned {
    pub(crate) nodes: Vec<f64>,
    pub(crate) weights: Vec<f64>,
    pub(crate) log_weights: Vec<f64>,
}

impl AnchorGridOwned {
    pub(crate) fn from_grid(grid: &crate::bms::EmpiricalZGrid) -> Self {
        Self::new(grid.nodes.clone(), grid.weights.clone())
    }

    pub(crate) fn new(nodes: Vec<f64>, weights: Vec<f64>) -> Self {
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        Self {
            nodes,
            weights,
            log_weights,
        }
    }

    #[inline]
    pub(crate) fn view(&self) -> AnchorGrid<'_> {
        AnchorGrid {
            nodes: &self.nodes,
            weights: &self.weights,
            log_weights: &self.log_weights,
        }
    }
}

/// What a row hands the anchored frame: its law, and the law's root slots
/// (gam#2928).
#[derive(Clone, Copy)]
pub(crate) struct AnchorRowContext<'a> {
    pub(crate) grid: AnchorGrid<'a>,
    /// The roots already solved on this law, when the caller keeps them.
    pub(crate) roots: Option<&'a AnchorRootCache>,
}

/// The Gaussian closed form, which is the anchor's seed and — on a Gaussian
/// law — its value.
#[inline]
pub(crate) fn gaussian_anchor(q: f64, observed_slope: f64) -> f64 {
    q * (1.0 + observed_slope * observed_slope).sqrt()
}

/// Entries in a global law's cross-row root table: a power of two.
const SHARED_ROOT_SLOTS: usize = 256;

/// The Taylor coefficients a [`RootSlot`] carries beside `α`: `c[i][j]` for
/// `1 ≤ i + j ≤ 5` ([`AnchorTaylor::stored_entries`]).
const TABLE_COEFFICIENTS: usize = 20;

/// One stored anchor: the exact inputs of the equation it solved, its root,
/// and — once a consumer that reads them has differentiated that root — its
/// Taylor table, published under a sequence lock so a reader sees one
/// completed write or nothing.
#[derive(Debug)]
struct RootSlot {
    sequence: AtomicU64,
    q: AtomicU64,
    slope: AtomicU64,
    root: AtomicU64,
    /// The table's coefficients at `root` in [`AnchorTaylor::stored_entries`]
    /// order, `NaN` while no consumer has differentiated it.
    table: [AtomicU64; TABLE_COEFFICIENTS],
}

/// A completed read of a [`RootSlot`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct StoredAnchor {
    q: u64,
    slope: u64,
    root: f64,
    taylor: Option<AnchorTaylor>,
}

/// The inputs and root of one completed write of a [`RootSlot`], without its
/// table: what deciding a hit and seeding a solve read (gam#2928).
#[derive(Clone, Copy, Debug, PartialEq)]
struct StoredRoot {
    /// The sequence the write left, so a later table read can tell whether it
    /// still reads the same write.
    sequence: u64,
    q: u64,
    slope: u64,
    root: f64,
}

impl RootSlot {
    fn empty() -> Self {
        let nan = f64::NAN.to_bits();
        Self {
            sequence: AtomicU64::new(0),
            q: AtomicU64::new(nan),
            slope: AtomicU64::new(nan),
            root: AtomicU64::new(nan),
            table: std::array::from_fn(|_| AtomicU64::new(nan)),
        }
    }

    /// The stored inputs and root from one completed write, or `None` while a
    /// writer holds the slot or finished a write during the read. Most slot
    /// calls only compare inputs or read the root, so the table is not loaded
    /// here (gam#2928).
    fn read_root(&self) -> Option<StoredRoot> {
        let before = self.sequence.load(Ordering::Acquire);
        if before & 1 == 1 {
            return None;
        }
        let q = self.q.load(Ordering::Acquire);
        let slope = self.slope.load(Ordering::Acquire);
        let root = f64::from_bits(self.root.load(Ordering::Acquire));
        (self.sequence.load(Ordering::Acquire) == before).then_some(StoredRoot {
            sequence: before,
            q,
            slope,
            root,
        })
    }

    /// The table the write `stored` read published, or `None` while that write
    /// published none or another write has begun since.
    fn read_table(&self, stored: &StoredRoot) -> Option<AnchorTaylor> {
        let mut coefficients = [[0.0; TAYLOR_SLOTS]; TAYLOR_SLOTS];
        coefficients[0][0] = stored.root;
        for (cell, (i, j)) in self.table.iter().zip(AnchorTaylor::stored_entries()) {
            coefficients[i][j] = f64::from_bits(cell.load(Ordering::Acquire));
        }
        if self.sequence.load(Ordering::Acquire) != stored.sequence {
            return None;
        }
        coefficients
            .iter()
            .flatten()
            .all(|value| value.is_finite())
            .then_some(AnchorTaylor { coefficients })
    }

    /// The stored anchor from one completed write, with its table where the
    /// same write published one, or `None` while a writer holds the slot.
    fn read(&self) -> Option<StoredAnchor> {
        let stored = self.read_root()?;
        Some(StoredAnchor {
            q: stored.q,
            slope: stored.slope,
            root: stored.root,
            taylor: self.read_table(&stored),
        })
    }

    /// Publish a solved anchor, with its table where the writer has one. A
    /// writer that finds the slot held skips it: the slot is a cache, and the
    /// other writer's anchor serves as well.
    fn write(&self, anchor: &StoredAnchor) {
        let before = self.sequence.load(Ordering::Acquire);
        if before & 1 == 1
            || self
                .sequence
                .compare_exchange(before, before + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return;
        }
        self.q.store(anchor.q, Ordering::Release);
        self.slope.store(anchor.slope, Ordering::Release);
        self.root.store(anchor.root.to_bits(), Ordering::Release);
        match &anchor.taylor {
            Some(taylor) => {
                for (cell, (i, j)) in self.table.iter().zip(AnchorTaylor::stored_entries()) {
                    cell.store(taylor.coefficients[i][j].to_bits(), Ordering::Release);
                }
            }
            None => {
                let nan = f64::NAN.to_bits();
                for cell in &self.table {
                    cell.store(nan, Ordering::Release);
                }
            }
        }
        self.sequence.store(before + 2, Ordering::Release);
    }
}

/// What a law's root slots answered with (gam#2943 part 3(iii)). A hit answers
/// from a stored root of bitwise the same equation; every other equation is
/// solved from the closed form (gam#2983). Evaluations are residual
/// evaluations, one per Halley or bisection step.
#[derive(Debug, Default)]
struct AnchorSolveCounters {
    hits: AtomicU64,
    cold_solves: AtomicU64,
    cold_evaluations: AtomicU64,
}

/// A snapshot of a law's [`AnchorSolveCounters`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AnchorSolveCounts {
    pub(crate) hits: u64,
    pub(crate) cold_solves: u64,
    pub(crate) cold_evaluations: u64,
}

impl AnchorSolveCounters {
    fn snapshot(&self) -> AnchorSolveCounts {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        AnchorSolveCounts {
            hits: load(&self.hits),
            cold_solves: load(&self.cold_solves),
            cold_evaluations: load(&self.cold_evaluations),
        }
    }
}

impl std::fmt::Display for AnchorSolveCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let per_solve = if self.cold_solves == 0 {
            0.0
        } else {
            self.cold_evaluations as f64 / self.cold_solves as f64
        };
        write!(
            f,
            "{} hits, {} solves from the closed form ({per_solve:.2} residual evaluations each)",
            self.hits, self.cold_solves,
        )
    }
}

/// The roots already solved on one law (gam#2928): `slots_per_row` slots per
/// training row — one per location channel the family anchors, two for
/// survival's entry and exit — and, for a global law, a cross-row table.
///
/// A stored root answers only for bitwise the inputs it was solved for. The
/// residual reads nothing else: `q`, the OBSERVED slope `b` (frailty scale
/// already applied), and the row's grid, which is fixed for the life of the
/// law that owns this cache — a re-estimated law is a new law with a new
/// cache. A hash selects a cross-row entry; equality of the stored inputs
/// decides a hit, and a root stored under any other inputs answers nothing,
/// not even as a seed (gam#2983). Every stored root was solved from the
/// closed form, so it is the one root its equation defines, and a hit from
/// any slot, whichever row or thread published it, is bitwise what a solve
/// would return. The cross-row
/// table keeps its own entries per slot kind: every row entering at the time
/// origin asks one entry equation, and the row-specific exit equation every
/// row publishes at each iterate would otherwise evict it (gam#2928).
#[derive(Debug)]
pub(crate) struct AnchorRootCache {
    /// Row `r`'s slot `s` at `r·slots_per_row + s`.
    slots: Vec<RootSlot>,
    slots_per_row: usize,
    /// `Some` for a global law, on which every row solves the same equation
    /// for the same inputs; a per-row law gives each row its own equation.
    /// Slot kind `s`'s entries at `s·SHARED_ROOT_SLOTS + hash`.
    shared: Option<Vec<RootSlot>>,
    shared_index: fn(u64, u64) -> u64,
    counters: AnchorSolveCounters,
}

impl AnchorRootCache {
    pub(crate) fn new(rows: usize, slots_per_row: usize, shared_across_rows: bool) -> Self {
        Self::with_shared_index(rows, slots_per_row, shared_across_rows, |q, slope| {
            splitmix64_hash(q ^ splitmix64_hash(slope))
        })
    }

    /// [`Self::new`] with the cross-row table indexed by `shared_index`.
    fn with_shared_index(
        rows: usize,
        slots_per_row: usize,
        shared_across_rows: bool,
        shared_index: fn(u64, u64) -> u64,
    ) -> Self {
        let slots = |count: usize| (0..count).map(|_| RootSlot::empty()).collect::<Vec<_>>();
        Self {
            slots: slots(rows * slots_per_row),
            slots_per_row,
            shared: shared_across_rows.then(|| slots(slots_per_row * SHARED_ROOT_SLOTS)),
            shared_index,
            counters: AnchorSolveCounters::default(),
        }
    }

    /// A cache of the same shape holding nothing: what a search that must not
    /// read another search's roots starts from (gnomon#2359).
    pub(crate) fn empty_like(&self) -> Self {
        Self::with_shared_index(
            self.slots.len() / self.slots_per_row.max(1),
            self.slots_per_row,
            self.shared.is_some(),
            self.shared_index,
        )
    }

    /// What the slots have answered with so far.
    pub(crate) fn counts(&self) -> AnchorSolveCounts {
        self.counters.snapshot()
    }

    /// Row `row`'s slot `slot`; `None` for a slot the cache does not keep.
    fn slot(&self, row: usize, slot: usize) -> Option<&RootSlot> {
        if slot >= self.slots_per_row {
            return None;
        }
        self.slots.get(row * self.slots_per_row + slot)
    }

    /// Slot kind `slot`'s cross-row entry for `(q, slope)`; `None` on a per-row
    /// law or for a slot the cache does not keep.
    fn shared_slot(&self, slot: usize, q: u64, slope: u64) -> Option<&RootSlot> {
        if slot >= self.slots_per_row {
            return None;
        }
        let table = self.shared.as_ref()?;
        let entry = (self.shared_index)(q, slope) as usize & (SHARED_ROOT_SLOTS - 1);
        table.get(slot * SHARED_ROOT_SLOTS + entry)
    }
}

/// [`solve_anchor`] through the law's root slots (gam#2928).
///
/// Every consumer of a row at one coefficient iterate — the value, the direct
/// order-two lowering, and each directional jet the outer derivatives seed —
/// asks for the root of bitwise the same equation, so the row's slot answers
/// without evaluating the residual. On a global law, rows asking the same
/// equation — every row entering at the time origin shares its entry index —
/// find one row's root in the cross-row table. An equation no slot holds is
/// solved from the closed form, which depends only on `(q, b)` (gam#2983). A
/// solve accepts any `α` whose residual meets [`anchor_residual_resolution`],
/// so where it stops depends on where it started: seeded from a slot that
/// other rows and threads publish into, the root depended on the order they
/// ran in, and 12-thread fits of one input diverged run to run (truth2370,
/// 57% of warm-seeded roots off the closed-form root by up to 6e5 ulps).
/// Seeded from the closed form, the root is a function of its equation alone,
/// whatever the slots hold and however many threads fill them.
pub(crate) fn solve_anchor_in_slot(
    q: f64,
    observed_slope: f64,
    context: AnchorRowContext<'_>,
    row: usize,
    slot: usize,
) -> Result<f64, String> {
    anchor_in_slot(q, observed_slope, context, row, slot, false).map(|anchor| anchor.root)
}

/// [`AnchorTaylor`] through the law's root slots (gam#2928).
///
/// A slot whose stored inputs are bitwise `(q, b)` and that holds a table
/// answers with it: it is what [`AnchorTaylor::at`] returned at the stored
/// root, so every consumer of the row at one iterate — the direct order-two
/// lowering, the value path and each directional jet — reads the same bits it
/// would compute. Otherwise the root comes through the path of
/// [`solve_anchor_in_slot`], is differentiated once, and is published with its
/// table.
pub(crate) fn anchor_taylor_in_slot(
    q: f64,
    observed_slope: f64,
    context: AnchorRowContext<'_>,
    row: usize,
    slot: usize,
) -> Result<AnchorTaylor, String> {
    let anchor = anchor_in_slot(q, observed_slope, context, row, slot, true)?;
    match anchor.taylor {
        Some(taylor) => Ok(taylor),
        None => AnchorTaylor::at(anchor.root, q, observed_slope, context.grid),
    }
}

/// [`AnchorDerivatives`] through the law's root slots: the orders through
/// three of [`anchor_taylor_in_slot`]'s table.
pub(crate) fn anchor_derivatives_in_slot(
    q: f64,
    observed_slope: f64,
    context: AnchorRowContext<'_>,
    row: usize,
    slot: usize,
) -> Result<AnchorDerivatives, String> {
    anchor_taylor_in_slot(q, observed_slope, context, row, slot).map(|taylor| taylor.derivatives())
}

/// The slot path both entries share: the anchor for `(q, b)`, carrying
/// derivatives whenever `differentiate` asks for them.
///
/// The row's own slot is asked first, by its inputs and root alone: its table
/// is loaded only where a consumer differentiates, and the cross-row table is
/// hashed only where the row's slot does not answer (gam#2928).
fn anchor_in_slot(
    q: f64,
    observed_slope: f64,
    context: AnchorRowContext<'_>,
    row: usize,
    slot: usize,
    differentiate: bool,
) -> Result<StoredAnchor, String> {
    let inputs = (q.to_bits(), observed_slope.to_bits());
    let differentiated = |root: f64| -> Result<Option<AnchorTaylor>, String> {
        if differentiate {
            AnchorTaylor::at(root, q, observed_slope, context.grid).map(Some)
        } else {
            Ok(None)
        }
    };
    let Some(cache) = context.roots else {
        let root = solve_anchor(q, observed_slope, context.grid)?;
        return Ok(StoredAnchor {
            q: inputs.0,
            slope: inputs.1,
            root,
            taylor: differentiated(root)?,
        });
    };
    let own = cache.slot(row, slot);
    let own_root = own
        .and_then(RootSlot::read_root)
        .filter(|stored| stored.root.is_finite());
    let publish = |anchor: &StoredAnchor| {
        if let Some(own) = own {
            own.write(anchor);
        }
        if let Some(shared) = cache.shared_slot(slot, inputs.0, inputs.1) {
            shared.write(anchor);
        }
    };
    let counters = &cache.counters;
    let cold = || -> Result<f64, String> {
        let (root, evaluations) =
            solve_anchor_counted(q, observed_slope, context.grid, gaussian_anchor(q, observed_slope))?;
        counters.cold_solves.fetch_add(1, Ordering::Relaxed);
        counters
            .cold_evaluations
            .fetch_add(evaluations as u64, Ordering::Relaxed);
        Ok(root)
    };
    if let (Some(own), Some(stored)) = (own, own_root)
        && (stored.q, stored.slope) == inputs
    {
        counters.hits.fetch_add(1, Ordering::Relaxed);
        let taylor = if differentiate { own.read_table(&stored) } else { None };
        if taylor.is_some() || !differentiate {
            return Ok(StoredAnchor {
                q: inputs.0,
                slope: inputs.1,
                root: stored.root,
                taylor,
            });
        }
        let anchor = StoredAnchor {
            q: inputs.0,
            slope: inputs.1,
            root: stored.root,
            taylor: differentiated(stored.root)?,
        };
        publish(&anchor);
        return Ok(anchor);
    }
    let shared_match = cache
        .shared_slot(slot, inputs.0, inputs.1)
        .and_then(RootSlot::read)
        .filter(|stored| stored.root.is_finite() && (stored.q, stored.slope) == inputs);
    let root = match shared_match {
        Some(stored) if stored.taylor.is_some() || !differentiate => {
            counters.hits.fetch_add(1, Ordering::Relaxed);
            if let Some(own) = own {
                own.write(&stored);
            }
            return Ok(stored);
        }
        Some(stored) => {
            counters.hits.fetch_add(1, Ordering::Relaxed);
            stored.root
        }
        None => cold()?,
    };
    let anchor = StoredAnchor {
        q: inputs.0,
        slope: inputs.1,
        root,
        taylor: differentiated(root)?,
    };
    publish(&anchor);
    Ok(anchor)
}

/// Solve the anchoring equation for `α` at the marginal index `q` and the
/// OBSERVED slope `b` (frailty scale already applied).
///
/// Bracketed Newton–Halley on a log-space residual: with `S(α) = Σ_k w_k
/// Φ(−η_k)` the residual is `log S − log Φ(−q)` when `Φ(−q) ≤ ½` and
/// `log(1 − S) − log Φ(q)` otherwise, so the tolerance is always a relative
/// error on the smaller of the two marginal tails. Both sides are strictly
/// monotone in `α`, the log-sum-exp keeps the sum finite where every node has
/// underflowed, and the solve searches from the Gaussian closed-form seed.
///
/// The solve is a safeguarded Halley iteration on a sign bracket (gam#2928):
/// Halley's step where it agrees with Newton's, a doubling search until both
/// sides of the root are seen, bisection whenever a step leaves the bracket or
/// fails to halve. It returns only an `α` whose residual meets
/// [`anchor_residual_resolution`] — the tolerance, or where coarser the
/// residual one float step of `α` resolves or the residual's own rounding —
/// and refuses otherwise. It does
/// not stop on bracket width, so no returned root carries a residual above
/// that criterion.
pub(crate) fn solve_anchor(q: f64, observed_slope: f64, grid: AnchorGrid<'_>) -> Result<f64, String> {
    solve_anchor_from(q, observed_slope, grid, gaussian_anchor(q, observed_slope))
}

/// [`solve_anchor`] from an explicit seed.
pub(crate) fn solve_anchor_from(
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    seed: f64,
) -> Result<f64, String> {
    solve_anchor_counted(q, observed_slope, grid, seed).map(|(root, _)| root)
}

/// [`solve_anchor_from`], with the residual evaluations the solve spent.
fn solve_anchor_counted(
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    seed: f64,
) -> Result<(f64, usize), String> {
    solve_anchor_with(q, observed_slope, grid, seed, anchor_log_residual)
}

/// The residual a solve reads: `(F, F′, F″)` at `(α, b, grid, survival side,
/// log target)`.
type AnchorResidual = fn(f64, f64, AnchorGrid<'_>, bool, f64) -> Result<(f64, f64, f64), String>;

/// [`solve_anchor_from`] on an explicit residual form, so the log-space form
/// can be solved on exactly the production path as an oracle.
fn solve_anchor_with(
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    seed: f64,
    residual: AnchorResidual,
) -> Result<(f64, usize), String> {
    if !(q.is_finite() && observed_slope.is_finite() && seed.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor requires finite q, slope and seed, got q={q}, b={observed_slope}, seed={seed}"
        ));
    }
    let survival_side = q >= 0.0;
    let log_target = smaller_tail_log_target(q);
    let rounding = anchor_residual_rounding(log_target, grid.len());
    solve_log_tail_root(
        seed,
        survival_side,
        |alpha| {
            let (value, first, second) =
                residual(alpha, observed_slope, grid, survival_side, log_target)?;
            Ok(LogTailResidual {
                value,
                first,
                second,
                rounding,
            })
        },
        "survival marginal-slope anchor",
        || format!("q={q}, b={observed_slope}"),
    )
}

/// One evaluation of a log-tail calibration residual: `(F, F′, F″)` at `α` and
/// the rounding `F` carries there ([`anchor_residual_rounding`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct LogTailResidual {
    pub(crate) value: f64,
    pub(crate) first: f64,
    pub(crate) second: f64,
    pub(crate) rounding: f64,
}

/// A calibration `P(a) = E[Φ(η(a, Z))]` at one intercept, read on its smaller
/// tail (gam#3216, gam#3333) and held in log form (gam#3639).
///
/// `T = E[Φ(∓η)]` is `1 − P` on the survival side (`q ≥ 0`) and `P` on the
/// complement side, summed from positive terms and never formed as `1 − P`.
/// It is kept as `log T`, and the slopes as ratios to it: `P′/T` and, where
/// evaluated, `P″/T`. Every consumer reads a ratio — the log residual's
/// `F′ = ∓P′/T`, and the implicit derivatives `a_u = −P_u/P′` in whatever unit
/// the calibration is divided by ([`Self::density_in_unit`]) — so nothing
/// needs `T` or `P′` themselves, which both underflow past `|q| ≈ 37.5` while
/// their ratio stays of order `|q|`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CalibrationTail {
    pub(crate) log_tail: f64,
    pub(crate) density_ratio: f64,
    pub(crate) density_slope_ratio: Option<f64>,
    /// The terms summed into `T`, for its rounding.
    pub(crate) summands: usize,
    /// A bound on `T`'s error beyond the correctly rounded sum of its terms
    /// (gam#3216, from PR #3374), RELATIVE to `T`: the sum of the cell
    /// integrators' own bounds over `T` on the de-nested route, zero on a
    /// finite law, whose terms are each a positive probability to its own
    /// relative accuracy. It is relative because `T` itself is held only as
    /// `log T` (gam#3639), and the residual reads exactly this ratio.
    pub(crate) tail_rounding_ratio: f64,
}

impl CalibrationTail {
    /// The log form of a tail summed in linear space: `log T`, `P′/T` and
    /// `P″/T`, each one rounded operation on the sums. Every route whose tail
    /// is summed linearly — the standard-normal cells, and a finite law above
    /// [`LINEAR_RESIDUAL_FLOOR`] — reads it through here, so its log residual
    /// is bitwise the one the linear sums gave before the log form.
    ///
    /// `tail_rounding` is the route's own absolute bound on `T`, carried as
    /// the ratio the residual reads; a route whose terms are each a positive
    /// probability to their own relative accuracy passes `0.0`.
    pub(crate) fn from_linear(
        tail: f64,
        density: f64,
        density_slope: Option<f64>,
        summands: usize,
        tail_rounding: f64,
    ) -> Self {
        Self {
            log_tail: tail.ln(),
            density_ratio: density / tail,
            density_slope_ratio: density_slope.map(|slope| slope / tail),
            summands,
            tail_rounding_ratio: tail_rounding / tail,
        }
    }

    /// The residual `log T − log Φ(∓q)` with its slopes: `T′ = ∓P′`,
    /// `F′ = T′/T`, and `F″ = T″/T − F′²` where `P″` was evaluated, else zero
    /// so the solve takes Newton's step. Its rounding is the tail's own bound
    /// relative to it plus the logarithms' ([`anchor_residual_rounding`]).
    ///
    /// A tail that underflowed to zero is `log T = −∞` (gam#3216, from PR
    /// #3374): the residual keeps its sign, which is all the bracket of
    /// [`solve_log_tail_root`] reads from it, with no finite slope and an
    /// unbounded rounding, and the solve never accepts it. A far probe of the
    /// bracket search therefore narrows the bracket instead of ending the
    /// solve. Only a non-finite log tail, bound or slope, or a negative bound,
    /// is refused; the caller checks `P′ > 0` at the root it accepts.
    pub(crate) fn log_residual(
        &self,
        survival_side: bool,
        log_target: f64,
    ) -> Result<LogTailResidual, String> {
        if self.log_tail == f64::NEG_INFINITY {
            return Ok(LogTailResidual {
                value: f64::NEG_INFINITY,
                first: f64::NAN,
                second: f64::NAN,
                rounding: f64::INFINITY,
            });
        }
        if !(self.log_tail.is_finite()
            && self.tail_rounding_ratio.is_finite()
            && self.tail_rounding_ratio >= 0.0
            && self.density_ratio.is_finite())
        {
            return Err(format!(
                "calibration tail has no resolvable log residual: log T={:e}, P'/T={:e}, \
                 tail_rounding/T={:e}",
                self.log_tail, self.density_ratio, self.tail_rounding_ratio
            ));
        }
        let sign = if survival_side { -1.0 } else { 1.0 };
        let first = sign * self.density_ratio;
        let second = match self.density_slope_ratio {
            Some(slope) if slope.is_finite() => sign * slope - first * first,
            Some(slope) => {
                return Err(format!("calibration tail has a non-finite P''/T={slope:e}"));
            }
            None => 0.0,
        };
        Ok(LogTailResidual {
            value: self.log_tail - log_target,
            first,
            second,
            rounding: self.tail_rounding_ratio
                + anchor_residual_rounding(log_target, self.summands),
        })
    }

    /// `P′/U` for a calibration divided by the unit `U = exp(log_unit)`:
    /// `(P′/T)·exp(log T − log U)`, finite wherever `T` and `U` are within
    /// the exponential's range of each other. At a root on the unit
    /// [`CalibrationUnit`] fixes, `T = U` up to the root's residual, so the
    /// exponent is that residual and the result is `P′/T` to within it; its
    /// only rounding beyond the ratio's own is the exponential's argument,
    /// `ε·(|log T| + |log U|)` absolute.
    #[inline]
    pub(crate) fn density_in_unit(&self, log_unit: f64) -> f64 {
        self.density_ratio * (self.log_tail - log_unit).exp()
    }
}

/// The running sums of a finite-law calibration tail
/// `T = Σ_k w_k Φ(∓η_k)`, `P′ = Σ w φ(η)·η_a`, `P″ = Σ w φ(η)·(η_aa − η·η_a²)`
/// (gam#3639), in linear space or by streaming log-sum-exp.
///
/// The linear pass is bitwise the sums every finite-law route formed before
/// the log form. The log pass holds each term as
/// `t_k = log w_k + log Φ(∓η_k)` against the running maximum `m`, adding
/// `e_k = exp(t_k − m)` to the tail and `e_k·ρ_k·η_a` and
/// `e_k·ρ_k·(η_aa − η·η_a²)` to the slopes, with `ρ_k = φ(η_k)/Φ(∓η_k)` the
/// Mills ratio from the same `erfcx` evaluation as `log Φ`; a new maximum
/// rescales all three sums by `exp(m_old − m_new) ≤ 1`. Then
/// `log T = m + log Σ e`, `P′/T = Σ e·ρ·η_a / Σ e` and likewise `P″/T`, so no
/// quantity ever leaves the range of its ratio to `T`. Nothing is stored per
/// node, so a pass allocates nothing.
pub(crate) struct CalibrationTailSum {
    survival_side: bool,
    log_space: bool,
    log_max: f64,
    tail: f64,
    density: f64,
    density_slope: f64,
}

impl CalibrationTailSum {
    fn new(survival_side: bool, log_space: bool) -> Self {
        Self {
            survival_side,
            log_space,
            log_max: f64::NEG_INFINITY,
            tail: 0.0,
            density: 0.0,
            density_slope: 0.0,
        }
    }

    /// Add node `k`: weight `w_k`, index `η_k` and its first two intercept
    /// derivatives.
    #[inline]
    pub(crate) fn push(&mut self, weight: f64, eta: f64, eta_a: f64, eta_aa: f64) {
        if !self.log_space {
            let tail_arg = if self.survival_side { -eta } else { eta };
            let curvature = eta_aa - eta * eta_a * eta_a;
            let pdf = normal_pdf(eta);
            self.tail += weight * normal_cdf(tail_arg);
            self.density += weight * pdf * eta_a;
            self.density_slope += weight * pdf * curvature;
            return;
        }
        self.push_log_weight(weight.ln(), eta, eta_a, eta_aa);
    }

    /// [`Self::push`] with the node's weight given as its logarithm, for a
    /// law whose weights are themselves below `f64::MIN_POSITIVE` (the
    /// log-space cells of [`sum_denested_cells_in_log_space`], gam#4504). A
    /// linear sum takes `exp(log_weight)`.
    #[inline]
    fn push_log_weight(&mut self, log_weight: f64, eta: f64, eta_a: f64, eta_aa: f64) {
        if !self.log_space {
            self.push(log_weight.exp(), eta, eta_a, eta_aa);
            return;
        }
        let tail_arg = if self.survival_side { -eta } else { eta };
        let curvature = eta_aa - eta * eta_a * eta_a;
        let (log_cdf, mills) = signed_probit_logcdf_and_mills_ratio(tail_arg);
        let term = log_weight + log_cdf;
        if term == f64::NEG_INFINITY {
            // A zero weight or a tail that is exactly zero contributes nothing.
            return;
        }
        if term > self.log_max {
            let rescale = (self.log_max - term).exp();
            self.tail *= rescale;
            self.density *= rescale;
            self.density_slope *= rescale;
            self.log_max = term;
        }
        let scaled = (term - self.log_max).exp();
        let scaled_density = scaled * mills;
        self.tail += scaled;
        self.density += scaled_density * eta_a;
        self.density_slope += scaled_density * curvature;
    }

    fn finish(self, summands: usize) -> CalibrationTail {
        if !self.log_space {
            // A finite law's terms are each a positive probability to their
            // own relative accuracy, so the sum carries no bound beyond the
            // correctly rounded one (gam#3216, from PR #3374).
            return CalibrationTail::from_linear(
                self.tail,
                self.density,
                Some(self.density_slope),
                summands,
                0.0,
            );
        }
        CalibrationTail {
            log_tail: self.log_max + self.tail.ln(),
            density_ratio: self.density / self.tail,
            density_slope_ratio: Some(self.density_slope / self.tail),
            summands,
            tail_rounding_ratio: 0.0,
        }
    }
}

/// A finite-law calibration tail on `survival_side` from `visit`, which pushes
/// every node of the law once into the sum it is handed (gam#3639).
///
/// The tail is summed in linear space; where it falls below
/// [`LINEAR_RESIDUAL_FLOOR`] or leaves the finite range, `visit` runs once more
/// into a log-space sum — the same rule and floor the declared-law anchor uses
/// ([`anchor_log_residual`]), so the two forms agree to rounding wherever both
/// are finite and the log form alone serves the deep tail. A `visit` that
/// records per-node values must start its record afresh on each call.
pub(crate) fn sum_calibration_tail<E>(
    survival_side: bool,
    summands: usize,
    mut visit: impl FnMut(&mut CalibrationTailSum) -> Result<(), E>,
) -> Result<CalibrationTail, E> {
    let mut linear = CalibrationTailSum::new(survival_side, false);
    visit(&mut linear)?;
    if linear.tail.is_finite() && linear.tail >= LINEAR_RESIDUAL_FLOOR {
        return Ok(linear.finish(summands));
    }
    let mut log_space = CalibrationTailSum::new(survival_side, true);
    visit(&mut log_space)?;
    Ok(log_space.finish(summands))
}

/// Whether a de-nested cells calibration tail summed in linear probability
/// still holds its value (gam#4504): the sum and its bound are finite, and the
/// sum is large enough that the underflow of its node terms sits below its
/// own rounding. A node term below `f64::MIN_POSITIVE` is flushed to zero or
/// held as a subnormal, which moves the sum by less than `f64::MIN_POSITIVE`,
/// so `summands` terms move it by at most `summands·MIN_POSITIVE`; the sum
/// keeps its relative accuracy while that is at most `u·T`. Past it — for a
/// standard-normal row, from about `|q| ≈ 36` — the cells are summed in log
/// units by [`sum_denested_cells_in_log_space`]. An infinite bound (an affine
/// cell's bivariate-normal orthant difference with no digits left) takes the
/// log route too.
pub(crate) fn linear_cell_tail_is_representable(
    tail: f64,
    tail_rounding: f64,
    summands: usize,
) -> bool {
    tail.is_finite()
        && tail_rounding.is_finite()
        && UNIT_ROUNDOFF * tail >= summands as f64 * f64::MIN_POSITIVE
}

/// A de-nested cells calibration tail summed in log units (gam#4504), the
/// standard-normal latent law's counterpart of the log pass of
/// [`sum_calibration_tail`], taken where [`linear_cell_tail_is_representable`]
/// fails.
///
/// Each cell is the finite law of its log-space quadrature
/// ([`visit_cell_log_quadrature`]): nodes `z_j` with log weights `log ω_j`,
/// index `η(z_j)` and intercept derivatives `η_a(z_j)`, `η_aa(z_j)` from the
/// cell's coefficient partials, all pushed into one streaming log-sum-exp
/// across the cells. `log T`, `P′/T` and `P″/T` therefore come out as ratios,
/// and nothing underflows with the tail. Each window is chosen on the tail's
/// own integrand: the cell negated on the survival side.
///
/// Its bound relative to `T` is the per-term Wilkinson factor the linear
/// value of a curved cell charges, plus every cell's dropped-mass bound over
/// `T`. Each node term's exponent is formed at the magnitude of `log T`,
/// whose rounding the residual already charges ([`anchor_residual_rounding`]).
///
/// `cells` yields `(cell, ∂c/∂a, ∂²c/∂a²)`. `density_slope` says whether `P″`
/// is read, as on the linear route it replaces.
pub(crate) fn sum_denested_cells_in_log_space(
    survival_side: bool,
    density_slope: bool,
    cells: impl IntoIterator<Item = (DenestedCubicCell, [f64; 4], [f64; 4])>,
) -> Result<CalibrationTail, String> {
    let mut sum = CalibrationTailSum::new(survival_side, true);
    let mut log_dropped = f64::NEG_INFINITY;
    let mut summands = 0usize;
    for (cell, dc_da, d2c_da2) in cells {
        let tail_cell = if survival_side { cell.negated() } else { cell };
        let dropped = visit_cell_log_quadrature(tail_cell, |z, log_weight| {
            sum.push_log_weight(
                log_weight,
                cell.eta(z),
                eval_coeff4_at(&dc_da, z),
                eval_coeff4_at(&d2c_da2, z),
            );
        })?;
        summands += TERMINAL_GL_ORDER;
        log_dropped = logaddexp(log_dropped, dropped);
    }
    let tail = sum.finish(summands);
    Ok(CalibrationTail {
        density_slope_ratio: if density_slope {
            tail.density_slope_ratio
        } else {
            None
        },
        tail_rounding_ratio: accumulation_growth(
            TERMINAL_GL_ORDER + NON_AFFINE_VALUE_TERM_OPERATIONS,
        ) + (log_dropped - tail.log_tail).exp(),
        ..tail
    })
}

/// The unit a probit calibration constraint `Σ_k w_k Φ(η_k) − Φ(q)` is
/// written in (gam#3639): the smaller marginal tail `N = Φ(−|q|)`, fixed at
/// the row's `q`.
///
/// Divided by `N`, the constraint on the survival side (`q ≥ 0`) is
/// `1 − Σ_k w_k Φ(−η_k)/N` (using `Σ w = 1`), node value `−Φ(−η)/N` and
/// marginal value `−1`; on the complement side it is `Σ_k w_k Φ(η_k)/N − 1`.
/// Both are the original constraint times the constant `1/N`, so the root and
/// every implicit derivative `a_u = −F_u/F_a`, which is homogeneous of degree
/// zero in `F`, are unchanged; only the representation moves, from
/// probabilities and densities that underflow together past `|q| ≈ 37.5` to
/// ratios of order `|q|`. The derivative channels are
/// `φ(η)/N · (1, −η, η² − 1, −(η³ − 3η))`, with
///
/// ```text
///     φ(η)/N = ρ_q · exp(½(q − η)(q + η)),     ρ_q = φ(q)/N,
/// ```
///
/// the Mills ratio at `q` from the same evaluation as `log N`. Its relative
/// rounding is `ε·(1 + |q² − η²|/2)` beyond `ρ_q`'s own, so it is exact at the
/// marginal (`η = q` gives `ρ_q` itself) and nowhere pays `ε·η²/2` the way
/// `exp(log φ(η) − log N)` would. At a root, each node's weighted unit
/// density obeys `w_k φ(η_k)/N = ρ(∓η_k)·w_k Φ(∓η_k)/N ≤ ρ(∓η_k)` (each
/// term of `T = N` is at most `N`), so it stays of order `|η_k| + 1`. The
/// node value `Φ(∓η)/N` is `exp(log Φ(∓η) − log N)`, rounding
/// `ε·(|log Φ(∓η)| + |log N|)` absolute in its exponent; at `|q| = 39` that
/// is `≈ 3.4e-13` relative, below the root's own residual resolution there
/// (`≈ 1.4e-12`, [`anchor_residual_rounding`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CalibrationUnit {
    q: f64,
    survival_side: bool,
    log_unit: f64,
    mills: f64,
}

impl CalibrationUnit {
    pub(crate) fn new(q: f64) -> Self {
        // `Φ(−|q|)`: bitwise `smaller_tail_log_target(q)`, which is the log
        // target every tail residual on this row is solved against.
        let (log_unit, mills) = signed_probit_logcdf_and_mills_ratio(-q.abs());
        Self {
            q,
            survival_side: q >= 0.0,
            log_unit,
            mills,
        }
    }

    /// `log N`.
    #[inline]
    pub(crate) fn log_unit(&self) -> f64 {
        self.log_unit
    }

    /// `φ(η)/N`.
    #[inline]
    pub(crate) fn density(&self, eta: f64) -> f64 {
        self.mills * (0.5 * (self.q - eta) * (self.q + eta)).exp()
    }

    /// The unary Taylor stack of one node's term of the unit constraint:
    /// `(∓Φ(∓η)/N, φ(η)/N·(1, −η, η² − 1, −(η³ − 3η)))`.
    #[inline]
    pub(crate) fn cdf_stack(&self, eta: f64) -> [f64; 5] {
        let value = if self.survival_side {
            -(normal_logcdf(-eta) - self.log_unit).exp()
        } else {
            (normal_logcdf(eta) - self.log_unit).exp()
        };
        let density = self.density(eta);
        let eta2 = eta * eta;
        [
            value,
            density,
            -eta * density,
            (eta2 - 1.0) * density,
            -(eta2 - 3.0) * eta * density,
        ]
    }

    /// The marginal's stack, `cdf_stack(q)`: value exactly `∓1` and first
    /// derivative exactly `ρ_q`.
    #[inline]
    pub(crate) fn marginal_stack(&self) -> [f64; 5] {
        self.cdf_stack(self.q)
    }
}

/// `log Φ(∓q)`, the log of the smaller marginal tail: the survival tail
/// `Φ(−q)` when `q ≥ 0`, the complement `Φ(q)` otherwise.
#[inline]
pub(crate) fn smaller_tail_log_target(q: f64) -> f64 {
    if q >= 0.0 {
        normal_logcdf(-q)
    } else {
        normal_logcdf(q)
    }
}

/// The root of a calibration residual written on the smaller marginal tail,
/// `F(α) = log T(α) − log Φ(∓q)`, from `seed` (gam#2928, gam#3333).
///
/// `residual` returns `(F, F′, F″)` at `α` with the rounding `F` carries there
/// ([`anchor_residual_rounding`]); `F″ = 0` reduces every Halley step to
/// Newton's. `survival_side` is `q ≥ 0`: there `T` is the survival tail and
/// `F` strictly decreases in `α`, on the complement side it strictly
/// increases.
///
/// The solve is a safeguarded Halley iteration on a sign bracket: Halley's
/// step where it agrees with Newton's, a doubling search until both sides of
/// the root are seen, bisection whenever a step leaves the bracket or fails to
/// halve. It returns only an `α` whose residual meets
/// [`anchor_residual_resolution`], finished by one Newton step from there, and
/// refuses otherwise; it never stops on a bracket width. So the returned root
/// is a function of the equation alone, to rounding, whatever the seed.
/// Returns the root and the residual evaluations spent.
pub(crate) fn solve_log_tail_root(
    seed: f64,
    survival_side: bool,
    mut residual: impl FnMut(f64) -> Result<LogTailResidual, String>,
    label: &str,
    context: impl Fn() -> String,
) -> Result<(f64, usize), String> {
    if !seed.is_finite() {
        return Err(format!("{label} requires a finite seed, got {seed} ({})", context()));
    }
    let increasing = !survival_side;
    // The sign bracket: `below < root < above` once each side has been seen.
    let mut below = f64::NEG_INFINITY;
    let mut above = f64::INFINITY;
    let mut alpha = seed;
    let mut step_cap = (0.25 * (1.0 + seed.abs())).max(1.0);
    let mut last_step = f64::INFINITY;
    for evaluation in 0..ANCHOR_SOLVE_MAX_EVALUATIONS {
        let LogTailResidual {
            value,
            first,
            second,
            rounding,
        } = residual(alpha)?;
        // A residual that is not finite (an underflowed tail's `−∞`) carries a
        // sign for the bracket and nothing to accept.
        if value.is_finite() && value.abs() <= anchor_residual_resolution(alpha, first, rounding) {
            // The accepted `α` still carries a residual up to the tolerance,
            // and where it stops depends on the seed, so it moves irregularly
            // with the coefficients: summed over 3e5 rows it floored the inner
            // solver's KKT residual at ~3e-11 against its 1e-11 target, and
            // the joint Newton spun its whole cycle budget (gam#2928). Newton's
            // step from the accepted point costs no evaluation and removes it
            // to second order; it is taken only where Newton's model holds
            // (`|F·F″| ≤ F′²`, the rule the step below uses) and inside the
            // bracket, else the accepted point stands.
            let polished = alpha - value / first;
            let in_model = (value * second).abs() <= first * first;
            return Ok((
                if in_model && polished.is_finite() && polished > below && polished < above {
                    polished
                } else {
                    alpha
                },
                evaluation + 1,
            ));
        }
        let root_is_above = if increasing { value < 0.0 } else { value > 0.0 };
        if root_is_above {
            below = alpha;
        } else {
            above = alpha;
        }
        let toward = if root_is_above { 1.0 } else { -1.0 };
        // Halley's step is Newton's divided by `1 − L/2`, `L = F·F″/F′²`, and
        // is taken only where that correction is a model of the residual,
        // `|L| ≤ 1`; else Newton's. A step that does not point at the root is
        // replaced by the capped search step. On a plateau — the marginal tail
        // near one, `F′` vanishing while `F` does not — `L` is huge and
        // Halley's step collapses to `2F′/F″`, a fraction of a unit per
        // evaluation however far away the root is: a trial at `b = 8.7`
        // seeded 55 away crawled 206 evaluations to it (gam#2928). Newton's
        // step there is the doubling search, or bisection once bracketed.
        let newton = -value / first;
        let halley = -2.0 * value * first / (2.0 * first * first - value * second);
        let mut step = if halley.is_finite() && (value * second).abs() <= first * first {
            halley
        } else {
            newton
        };
        if !(step.is_finite() && step * toward > 0.0) {
            step = toward * step_cap;
        }
        let next = if below.is_finite() && above.is_finite() {
            let midpoint = below + 0.5 * (above - below);
            let candidate = alpha + step;
            // Bisect whenever the step leaves the bracket or is not at most
            // half the step before it, so the bracket reaches adjacent floats
            // in bounded work however the curvature behaves.
            let next = if candidate > below && candidate < above && step.abs() <= 0.5 * last_step {
                candidate
            } else {
                midpoint
            };
            last_step = (next - alpha).abs();
            if !(next > below && next < above) {
                return Err(format!(
                    "{label} bracketed its root between adjacent floats \
                     [{below:e}, {above:e}] without resolving the residual ({})",
                    context()
                ));
            }
            next
        } else if step.abs() > step_cap {
            // One side unseen: search toward it with a doubling cap.
            let capped = alpha + toward * step_cap;
            step_cap *= 2.0;
            capped
        } else {
            alpha + step
        };
        alpha = next;
    }
    Err(format!(
        "{label} did not resolve its residual in {ANCHOR_SOLVE_MAX_EVALUATIONS} \
         evaluations ({}, seed={seed}, bracket [{below:e}, {above:e}])",
        context()
    ))
}

/// Residual evaluations a solve may spend before it refuses: enough for a
/// doubling search across the whole `f64` range and a bisection down to
/// adjacent floats.
const ANCHOR_SOLVE_MAX_EVALUATIONS: usize = 256;

/// The `|F|` a root is held to (gam#2928): [`ANCHOR_LOG_RESIDUAL_TOL`], or the
/// coarser of what the arithmetic can resolve — the residual one
/// floating-point step of `α` moves, `|F′(α)|·ulp(α)`, and the residual's own
/// rounding ([`anchor_residual_rounding`]). The float nearest the root meets
/// it, so a solve stops on this criterion or refuses; it never stops on a
/// bracket width.
#[inline]
pub(crate) fn anchor_residual_resolution(alpha: f64, first: f64, rounding: f64) -> f64 {
    let magnitude = alpha.abs();
    ANCHOR_LOG_RESIDUAL_TOL
        .max(first.abs() * (magnitude.next_up() - magnitude))
        .max(rounding)
}

/// The rounding the residual `log T − log Φ(∓q)` carries on a law of `nodes`
/// nodes (gam#2941): two logarithms of magnitude about `|log Φ(∓q)|`, each
/// formed by at most four rounded operations at that magnitude, and `nodes`
/// terms summed. Deep in a tail it is the binding floor: at `q = −89.4` both
/// logarithms sit near −4000 and a residual cannot be resolved below about
/// 3e-12, however close `α` is to the root.
#[inline]
pub(crate) fn anchor_residual_rounding(log_target: f64, nodes: usize) -> f64 {
    f64::EPSILON * (nodes as f64 + 8.0 * (1.0 + log_target.abs()))
}

/// The smallest marginal tail the linear-space residual sums directly. Every
/// node term is a positive probability, so the sum keeps each term's relative
/// accuracy; below this floor a term can have underflowed against a sum it is
/// no longer negligible in, and the log-space form takes over.
const LINEAR_RESIDUAL_FLOOR: f64 = 1e-250;

/// `(F, F′, F″)` of the log-space anchoring residual at `α` (gam#2928).
///
/// The residual is always `log T(α) − log Φ(∓q)` with `T` the marginal on the
/// SMALLER tail — `Σ_k w_k Φ(−η_k)` against `Φ(−q) ≤ ½` when `q ≥ 0`,
/// `Σ_k w_k Φ(η_k)` against `Φ(q) ≤ ½` otherwise — so acceptance at
/// `|F| ≤ 1e-13` is a relative error on that tail on both forms, and nothing
/// subtracts nearly equal survival probabilities where `S` sits near one.
/// `T` is summed in linear space whenever it sits above
/// [`LINEAR_RESIDUAL_FLOOR`]: every node term is a positive probability from
/// the table route [`normal_cdf_and_pdf`], so the sum keeps each term's
/// relative accuracy, at one exponential per node for both `Φ` and `φ` where
/// the log-space form pays a scaled complementary error function, its
/// logarithm and a log-sum-exp exponential.
/// The two forms agree to rounding, and deep tails keep the log-space form.
fn anchor_log_residual(
    alpha: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    survival_side: bool,
    log_target: f64,
) -> Result<(f64, f64, f64), String> {
    // T = Σ_k w_k Φ(∓η_k), and the density sums Σ w φ(η), Σ w η φ(η).
    let mut tail = 0.0;
    let mut density = 0.0;
    let mut density_eta = 0.0;
    for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
        let eta = alpha + observed_slope * u;
        // φ is even, and the table's φ is bitwise normal_pdf at ±η.
        let (cdf, pdf) = normal_cdf_and_pdf(if survival_side { -eta } else { eta });
        tail += w * cdf;
        let weighted_pdf = w * pdf;
        density += weighted_pdf;
        density_eta += weighted_pdf * eta;
    }
    if !(tail.is_finite() && tail >= LINEAR_RESIDUAL_FLOOR) {
        return anchor_log_residual_in_log_space(
            alpha,
            observed_slope,
            grid,
            survival_side,
            log_target,
        );
    }
    let value = tail.ln() - log_target;
    let ratio_sum = density / tail;
    let ratio_eta_sum = density_eta / tail;
    // Survival side: T′/T = −Σ w φ / T, (log T)″ = Σ w η φ / T − (T′/T)².
    // Complement side: T′/T = Σ w φ / T, (log T)″ = −Σ w η φ / T − (T′/T)².
    let (first, second) = if survival_side {
        (-ratio_sum, ratio_eta_sum - ratio_sum * ratio_sum)
    } else {
        (ratio_sum, -ratio_eta_sum - ratio_sum * ratio_sum)
    };
    if !(value.is_finite() && first.is_finite() && first != 0.0 && second.is_finite()) {
        return anchor_log_residual_in_log_space(
            alpha,
            observed_slope,
            grid,
            survival_side,
            log_target,
        );
    }
    Ok((value, first, second))
}

/// [`anchor_log_residual`] with the marginal tail formed by log-sum-exp, for
/// tails below [`LINEAR_RESIDUAL_FLOOR`].
fn anchor_log_residual_in_log_space(
    alpha: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    survival_side: bool,
    log_target: f64,
) -> Result<(f64, f64, f64), String> {
    let m = grid.len();
    // Per node, from one shared `erfcx`/`erfc` evaluation: the log term
    // `log w_k + log Φ(∓η_k)` and the Mills ratio `φ(η_k) / Φ(∓η_k)`
    // (gam#2928).
    // Inline storage, filled rather than zeroed: a law of up to 128 nodes
    // touches no allocator and no per-evaluation memset.
    let mut terms: SmallVec<[f64; 128]> = SmallVec::with_capacity(m);
    let mut mills: SmallVec<[f64; 128]> = SmallVec::with_capacity(m);
    let mut log_max = f64::NEG_INFINITY;
    for k in 0..m {
        let eta = alpha + observed_slope * grid.nodes[k];
        let (log_cdf, mills_ratio) =
            signed_probit_logcdf_and_mills_ratio(if survival_side { -eta } else { eta });
        let term = grid.log_weights[k] + log_cdf;
        terms.push(term);
        mills.push(mills_ratio);
        if term > log_max {
            log_max = term;
        }
    }
    if !log_max.is_finite() {
        return Err(format!(
            "survival marginal-slope anchor residual is non-finite at α={alpha}, b={observed_slope}"
        ));
    }
    // log Σ_k w_k Φ(∓η_k) by log-sum-exp; each term is overwritten by its
    // scaled weight `e_k = exp(term_k − max)`.
    let mut sum = 0.0;
    for term in terms.iter_mut() {
        *term = (*term - log_max).exp();
        sum += *term;
    }
    let log_sum = log_max + sum.ln();
    let value = log_sum - log_target;
    // Density ratios `m_k = w_k φ(η_k) / S = (e_k / Σ_j e_j)·φ(η_k)/Φ(∓η_k)`:
    // the log-sum-exp weight times the Mills ratio, with no second
    // exponential per node.
    let mut ratio_sum = 0.0;
    let mut ratio_eta_sum = 0.0;
    for k in 0..m {
        let eta = alpha + observed_slope * grid.nodes[k];
        let ratio = terms[k] / sum * mills[k];
        ratio_sum += ratio;
        ratio_eta_sum += ratio * eta;
    }
    // Survival side: S′ = −Σ w φ, S″ = Σ w η φ.
    // Complement side: C′ = Σ w φ, C″ = −Σ w η φ.
    let (first, second) = if survival_side {
        (-ratio_sum, ratio_eta_sum - ratio_sum * ratio_sum)
    } else {
        (ratio_sum, -ratio_eta_sum - ratio_sum * ratio_sum)
    };
    // `F′` is mathematically non-zero on both sides; where every density ratio
    // underflowed the monotone solver only needs its sign to bracket.
    let first = if first == 0.0 {
        if survival_side {
            -f64::MIN_POSITIVE
        } else {
            f64::MIN_POSITIVE
        }
    } else {
        first
    };
    if !(value.is_finite() && first.is_finite() && second.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor residual state is non-finite: F={value}, F′={first}, F″={second} at α={alpha}"
        ));
    }
    Ok((value, first, second))
}

/// The anchor and its derivatives in `(q, b)` through order three: the orders
/// of its Taylor table ([`AnchorTaylor::derivatives`]) the direct order-two
/// lowering of the anchored frame reads. Third order is the highest it reads:
/// the exit-time rate feature is `α̇₁ = α_q(q₁, b)·q̇₁`, whose curvature in the
/// primaries is a third derivative of `α`. Everything above that comes through
/// the table's lift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AnchorDerivatives {
    pub(crate) alpha: f64,
    pub(crate) a_q: f64,
    pub(crate) a_b: f64,
    pub(crate) a_qq: f64,
    pub(crate) a_qb: f64,
    pub(crate) a_bb: f64,
    pub(crate) a_qqq: f64,
    pub(crate) a_qqb: f64,
    pub(crate) a_qbb: f64,
    pub(crate) a_bbb: f64,
}

/// The law's density weights at an anchor, normalized in log space (gam#2941):
/// `ω_k = w_k φ(η_k) / Σ_j w_j φ(η_j)` with `η_k = α + b·u_k`.
///
/// Every implicit derivative of the anchor is a ratio of density sums — to
/// each other and to `φ(q)` — so it reads the weights only through `ω` and the
/// scale only through the density ratio `φ(q)/Σ w φ(η)`. Deep in either tail
/// every `φ(η_k)` underflows while those ratios are finite. Each exponent is
/// formed against the heaviest node `*` as `log w_k − log w_* − ½(η_k − η_*)(η_k
/// + η_*)`, whose difference `η_k − η_* = b·(u_k − u_*)` carries no cancellation,
/// so a weight keeps its digits where `½η²` alone would round them away.
pub(crate) struct AnchorDensity {
    weights: SmallVec<[f64; 128]>,
    /// `log w_*`, `η_*` and `log Σ_k exp(·)` of the shifted sum, so the density
    /// ratio `φ(q)/D` at any `q` is one exponential of the same pass.
    heaviest_log_weight: f64,
    heaviest_eta: f64,
    log_relative_sum: f64,
}

impl AnchorDensity {
    pub(crate) fn at(alpha: f64, observed_slope: f64, grid: AnchorGrid<'_>) -> Result<Self, String> {
        let m = grid.len();
        let mut heaviest = None;
        let mut heaviest_log_density = f64::NEG_INFINITY;
        for k in 0..m {
            let eta = alpha + observed_slope * grid.nodes[k];
            let log_density = grid.log_weights[k] - 0.5 * eta * eta;
            if log_density > heaviest_log_density {
                heaviest_log_density = log_density;
                heaviest = Some(k);
            }
        }
        let Some(heaviest) = heaviest else {
            return Err(format!(
                "survival marginal-slope anchor density has no finite node at α={alpha}, b={observed_slope}"
            ));
        };
        let heaviest_node = grid.nodes[heaviest];
        let heaviest_log_weight = grid.log_weights[heaviest];
        let heaviest_eta = alpha + observed_slope * heaviest_node;
        let mut weights = SmallVec::with_capacity(m);
        let mut sum = 0.0;
        for k in 0..m {
            let eta = alpha + observed_slope * grid.nodes[k];
            let exponent = (grid.log_weights[k] - heaviest_log_weight)
                - 0.5 * (observed_slope * (grid.nodes[k] - heaviest_node)) * (eta + heaviest_eta);
            let relative = exponent.exp();
            weights.push(relative);
            sum += relative;
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err(format!(
                "survival marginal-slope anchor density weights are not normalizable: Σ={sum:e} at α={alpha}, b={observed_slope}"
            ));
        }
        for weight in weights.iter_mut() {
            *weight /= sum;
        }
        Ok(Self {
            weights,
            heaviest_log_weight,
            heaviest_eta,
            log_relative_sum: sum.ln(),
        })
    }

    /// `ω_k`, in the law's node order.
    #[inline]
    pub(crate) fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// `log(φ(q)/D)` with `D = Σ_k w_k φ(η_k)`: the `√(2π)` cancels, and the
    /// difference of squares is formed against the heaviest node as the
    /// weights were.
    #[inline]
    pub(crate) fn log_density_ratio(&self, q: f64) -> f64 {
        -self.heaviest_log_weight
            - self.log_relative_sum
            - 0.5 * (q - self.heaviest_eta) * (q + self.heaviest_eta)
    }

}

/// The anchor's first partials `(α_q, α_b)` at a solved root, by implicit
/// differentiation of the calibration residual. At the root
/// `Σ_k w_k Φ(∓η_k) = Φ(∓q)` with `η_k = α + b·u_k`, so with the density mass
/// `D = Σ_k w_k φ(η_k)`:
///
/// ```text
///   α_q = φ(q) / D,        α_b = −Σ_k w_k φ(η_k) u_k / D.
/// ```
///
/// Both sides of the marginal give these same two expressions. They are the
/// order-one entries of the anchor's Taylor table
/// ([`AnchorTaylor::derivatives`]) without the table: a prediction node
/// reads `α` and these two partials alone, and the table's order-five lift
/// per law node was the whole cost of a posterior rule's node. Formed against
/// the heaviest law node as [`AnchorDensity::at`] forms its weights, so a tail
/// where every `φ(η_k)` underflows keeps finite ratios; `φ(q)/D` is then
/// `exp(−½q² − log D)` with `log D` read off the same shifted sum.
pub(crate) fn anchor_first_derivatives(
    alpha: f64,
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
) -> Result<(f64, f64), String> {
    let m = grid.len();
    let mut heaviest = None;
    let mut heaviest_log_density = f64::NEG_INFINITY;
    for k in 0..m {
        let eta = alpha + observed_slope * grid.nodes[k];
        let log_density = grid.log_weights[k] - 0.5 * eta * eta;
        if log_density > heaviest_log_density {
            heaviest_log_density = log_density;
            heaviest = Some(k);
        }
    }
    let Some(heaviest) = heaviest else {
        return Err(format!(
            "survival marginal-slope anchor density has no finite node at α={alpha}, b={observed_slope}"
        ));
    };
    let heaviest_node = grid.nodes[heaviest];
    let heaviest_log_weight = grid.log_weights[heaviest];
    let heaviest_eta = alpha + observed_slope * heaviest_node;
    let mut sum = 0.0;
    let mut node_sum = 0.0;
    for k in 0..m {
        let node = grid.nodes[k];
        let eta = alpha + observed_slope * node;
        let exponent = (grid.log_weights[k] - heaviest_log_weight)
            - 0.5 * (observed_slope * (node - heaviest_node)) * (eta + heaviest_eta);
        let relative = exponent.exp();
        sum += relative;
        node_sum += relative * node;
    }
    if !(sum.is_finite() && sum > 0.0 && node_sum.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor density weights are not normalizable: Σ={sum:e} at α={alpha}, b={observed_slope}"
        ));
    }
    // `log D = log w_* − ½η_*² + log Σ`, and `φ(q)/D` cancels the `√(2π)`.
    let a_q = (-0.5 * q * q - heaviest_log_weight + 0.5 * heaviest_eta * heaviest_eta - sum.ln()).exp();
    let a_b = -node_sum / sum;
    if !(a_q.is_finite() && a_b.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor partials are not finite at α={alpha}, q={q}, b={observed_slope}: α_q={a_q}, α_b={a_b}"
        ));
    }
    Ok((a_q, a_b))
}

/// The order of the anchor's Taylor table. The order-≤4 carriers read `α`
/// through order four and the exit rate `α̇₁ = α_q·q̇₁` through order four,
/// which is `α` through order five.
const ANCHOR_TAYLOR_ORDER: usize = 5;

/// Coefficient slots per axis of the table: degrees `0..=ANCHOR_TAYLOR_ORDER`.
const TAYLOR_SLOTS: usize = ANCHOR_TAYLOR_ORDER + 1;

/// Slots per axis of the order-six table the exact fifth likelihood derivative
/// reads (gam#2945): the event row's `log α̇₁ = log(α_q·q̇₁)` through order five
/// is `α` through order six. It is built on demand and never stored, so a
/// root slot keeps its order-five table.
pub(crate) const ANCHOR_SIXTH_SLOTS: usize = 7;

/// `n!` for `n ≤ 6`: the factor between a Taylor coefficient and the
/// derivative stack a carrier composes.
const FACTORIAL: [f64; ANCHOR_SIXTH_SLOTS] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0];

/// Slices of `δα^r` the order-five solve reads: degree `e` for `2 ≤ r ≤ e ≤ 5`.
const POWER_SLICES: usize = 10;

/// Slices of `δα^r` the order-six solve reads: degree `e` for `2 ≤ r ≤ e ≤ 6`.
const SIXTH_POWER_SLICES: usize = 15;

/// Where the degree-`e` part of `δα^r` sits among a solve's power slices
/// (`2 ≤ r ≤ e`): degrees in order, powers ascending within each.
#[inline]
fn power_slice(r: usize, degree: usize) -> usize {
    (degree - 1) * (degree - 2) / 2 + (r - 2)
}

/// `[G, G′, …, G⁽⁶⁾](x)` for `G(x) = log F(x) = log Φ(−x)`: `G^{(n)}(x) =
/// (−1)^n·(log Φ)^{(n)}(−x)`, from the tail-exact log-CDF derivatives (the left
/// tail's continued fraction, the right tail's log-magnitude recurrence), so a
/// node or target deep in either tail keeps every digit (gam#3639).
#[inline]
fn log_survival_derivatives(x: f64) -> [f64; ANCHOR_SIXTH_SLOTS] {
    let log_cdf = gam_math::probability::normal_logcdf_derivatives_through_sixth(-x);
    std::array::from_fn(|n| if n % 2 == 0 { log_cdf[n] } else { -log_cdf[n] })
}

/// The log-mass derivatives of the side the anchoring equation is solved on,
/// the side [`solve_anchor`] reads: `log Φ(−x)` for `q ≥ 0`, and for `q < 0`
/// the complement `log Φ(x)`. The weights sum to one, so `Σ_k w_k Φ(−η_k) =
/// Φ(−q)` and `Σ_k w_k Φ(η_k) = Φ(q)` are the same equation. Only the smaller
/// side has digits in a tail: at `q = −39`, `log Φ(39) = −1e-333` is `0` in
/// binary64 and every partial of the survival side underflows with it
/// (gam#3639).
#[inline]
fn anchor_side_log_derivatives(x: f64, complement: bool) -> [f64; ANCHOR_SIXTH_SLOTS] {
    if complement {
        gam_math::probability::normal_logcdf_derivatives_through_sixth(x)
    } else {
        log_survival_derivatives(x)
    }
}

/// The anchor's Taylor table at a solved root (gam#2928): `c[i][j] =
/// ∂_q^i ∂_b^j α / (i!·j!)` through total order five.
///
/// Implicit differentiation of `Ψ(α, b) = log Σ_k w_k F(α + b u_k) = log F(q)`,
/// `F(x) = Φ(−x)`, degree by degree: the logarithm of the anchoring equation,
/// whose partials stay `O(1)` in either tail where `F`'s own grow like `η^{n−1}`
/// (gam#3639; [`anchor_taylor_coefficients`] derives `Ψ`'s Taylor polynomial from
/// the nodes' centred exponents). With `A_d` the degree-`d` part of `δα` and
/// `T_n` the Taylor coefficients of `log F` at `q`, the degree-`d` part of the
/// expanded equation is
///
/// ```text
///     Ψ_{10}·A_d = T_d·δq^d − Σ_{(r,s) ∉ {(0,0),(1,0)}} Ψ_{rs}·[δα^r]_{d−s}·δb^s / (r!·s!),
/// ```
///
/// whose right side reads only `A_1, …, A_{d−1}`: `[δα^r]_e` for `r ≥ 2` is
/// built from parts below degree `e`, and `r = 1` enters only with `s ≥ 1`.
/// Five steps on homogeneous polynomials of at most six terms solve the table
/// through order five. Its orders through three are the implicit derivatives
/// the direct order-two lowering reads ([`AnchorTaylor::derivatives`]);
/// `anchor_tests` pins them against the closed-form implicit derivatives, and
/// the table's lift against Newton's iteration in the jet algebra.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AnchorTaylor {
    coefficients: [[f64; TAYLOR_SLOTS]; TAYLOR_SLOTS],
}

/// The implicit-differentiation solve of [`AnchorTaylor`] through total order
/// `SLOTS − 1`, with `SLICES` the number of power slices that order reads.
///
/// The order-six solve (gam#2945) runs the order-five one's arithmetic
/// unchanged and adds degree six after it, so its orders through five are the
/// order-five table's bits.
/// The bits of the anchor's own scale, `1 + |α|`, that every coefficient of
/// the linear-space table must be accurate to for the table to be published
/// from it. The degree-by-degree solve forms each coefficient as a sum whose
/// terms are Hermite moments, `F^{(n)}(η) = (−1)^n He_{n−1}(η) φ(η)`, growing
/// like `η^{n−1}` while the coefficient is `O(1)` or smaller: the sum cancels,
/// and the rounding it carries is `ε` times the sum of the terms' magnitudes,
/// propagated here through every product and quotient of the solve. The bound
/// is absolute on the anchor's scale because that is how the table is read:
/// a row derivative is a Faà di Bruno sum of the coefficients with `O(1)`
/// weights, and the certificates that read it judge that sum against its own
/// rounding, so an absolute error at this level below `1 + |α|` is below
/// every band the table feeds. Where the linear solve does not reach it the
/// table is solved in logarithms instead ([`anchor_taylor_coefficients_log`],
/// gam#3639), whose coefficients are `O(1)` at every order: at a tail index
/// the linear sum cancels `He_4(39)/5! ≈ 2e4` down to `5e-10` — thirteen
/// orders, no bits — and the log table is the only one with digits; inside
/// the law's bulk the linear solve reaches this at a twentieth of the log
/// table's arithmetic, which is most of a marginal-slope fit's time (gnomon
/// biobank, 16k rows: 5451 against 1328 CPU-seconds).
const LINEAR_TABLE_MIN_BITS: i32 = 32;

/// The anchor's Taylor table in `(q, b)`: the linear-space solve where it keeps
/// [`LINEAR_TABLE_MIN_BITS`] bits of every coefficient, the log-space solve
/// where it does not.
fn anchor_taylor_coefficients<const SLOTS: usize, const SLICES: usize>(
    alpha: f64,
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
) -> Result<[[f64; SLOTS]; SLOTS], String> {
    const { assert!(SLICES == (SLOTS - 1) * (SLOTS - 2) / 2) };
    // The linear solve of a degree is the same arithmetic at every order, so
    // the published order-five table decides on its five degrees and an
    // order-six table extends it by construction (gam#2945): where only its
    // sixth degree lacks bits, that degree alone is read from the log table.
    let (linear, failed) =
        anchor_taylor_coefficients_linear::<SLOTS, SLICES>(alpha, q, observed_slope, grid)?;
    match failed {
        None => Ok(linear),
        Some(degree) if degree >= TAYLOR_SLOTS => {
            let log = anchor_taylor_coefficients_log::<SLOTS, SLICES>(alpha, q, observed_slope, grid)?;
            let mut table = linear;
            for (i, row) in table.iter_mut().enumerate() {
                for (j, slot) in row.iter_mut().enumerate() {
                    if i + j >= degree {
                        *slot = log[i][j];
                    }
                }
            }
            Ok(table)
        }
        Some(_) => anchor_taylor_coefficients_log::<SLOTS, SLICES>(alpha, q, observed_slope, grid),
    }
}

/// The Hermite-moment (linear-space) table: `moments[n][s] = Σ_k ω_k u_k^s
/// F^{(n)}(η_k)` under the law's density weights `ω_k` and the target
/// `F^{(n)}(q)·φ(q)/D`, both homogeneous of degree zero in the density so they
/// stay finite where every node's density underflows (gam#2941). Beside the
/// table, the lowest total degree whose coefficients the solve does not reach
/// [`LINEAR_TABLE_MIN_BITS`] at, or `None` when every degree does; degree one
/// where the marginal tail is below the linear residual floor or the target
/// ratio is not a finite positive number. The table's entries at and above a
/// failing degree are not to be read.
fn anchor_taylor_coefficients_linear<const SLOTS: usize, const SLICES: usize>(
    alpha: f64,
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
) -> Result<([[f64; SLOTS]; SLOTS], Option<usize>), String> {
    const { assert!(SLICES == (SLOTS - 1) * (SLOTS - 2) / 2) };
    let refused = [[0.0_f64; SLOTS]; SLOTS];
    if smaller_tail_log_target(q) < LINEAR_RESIDUAL_FLOOR.ln() {
        return Ok((refused, Some(1)));
    }
    let density = AnchorDensity::at(alpha, observed_slope, grid)?;
    // Every sum carries its magnitude beside it: the rounding of a floating
    // sum is `ε` times the sum of its terms' magnitudes, and the bound below
    // propagates that through every product and quotient of the solve, so the
    // bits each coefficient keeps are the bits of the arithmetic that formed
    // it, not of its last sum alone.
    let mut moments = [[0.0_f64; SLOTS]; SLOTS];
    let mut moments_magnitude = [[0.0_f64; SLOTS]; SLOTS];
    for (&u, &omega) in grid.nodes.iter().zip(density.weights().iter()) {
        let stack = survival_cdf_derivative_stack::<SLOTS>(alpha + observed_slope * u, omega);
        let mut power = 1.0;
        for s in 0..SLOTS {
            for n in s.max(1)..SLOTS {
                let term = stack[n] * power;
                moments[n][s] += term;
                moments_magnitude[n][s] += term.abs();
            }
            power *= u;
        }
    }
    let h_alpha = moments[1][0];
    if !(h_alpha.is_finite() && h_alpha < 0.0) {
        return Ok((refused, Some(1)));
    }
    let h_alpha_relative_error = f64::EPSILON * moments_magnitude[1][0] / h_alpha.abs();
    // `φ(q)/D`, the density ratio the target is normalized by, from the pass
    // that formed the weights.
    let ratio = density.log_density_ratio(q).exp();
    if !(ratio.is_finite() && ratio > 0.0) {
        return Ok((refused, Some(1)));
    }
    let target = survival_cdf_derivative_stack::<SLOTS>(q, ratio);
    let mut parts = [[0.0_f64; SLOTS]; SLOTS];
    let mut parts_error = [[0.0_f64; SLOTS]; SLOTS];
    let mut powers = [[0.0_f64; SLOTS]; SLICES];
    let mut powers_error = [[0.0_f64; SLOTS]; SLICES];
    let error_floor = 2.0_f64.powi(-LINEAR_TABLE_MIN_BITS) * (1.0 + alpha.abs());
    for degree in 1..SLOTS {
        for r in 2..=degree {
            let mut power = [0.0_f64; SLOTS];
            let mut power_error = [0.0_f64; SLOTS];
            for j in 1..=degree + 1 - r {
                let lower = degree - j;
                let (right, right_error) = if r == 2 {
                    (&parts[lower], &parts_error[lower])
                } else {
                    let slice = power_slice(r - 1, lower);
                    (&powers[slice], &powers_error[slice])
                };
                for i1 in 0..=j {
                    for i2 in 0..=lower {
                        let product = parts[j][i1] * right[i2];
                        power[i1 + i2] += product;
                        power_error[i1 + i2] += parts_error[j][i1] * right[i2].abs()
                            + parts[j][i1].abs() * right_error[i2]
                            + f64::EPSILON * product.abs();
                    }
                }
            }
            powers[power_slice(r, degree)] = power;
            powers_error[power_slice(r, degree)] = power_error;
        }
        let mut rest = [0.0_f64; SLOTS];
        let mut rest_error = [0.0_f64; SLOTS];
        rest[0] = moments[degree][degree] / FACTORIAL[degree];
        rest_error[0] = f64::EPSILON * moments_magnitude[degree][degree] / FACTORIAL[degree];
        let target_term = target[degree] / FACTORIAL[degree];
        rest[degree] -= target_term;
        rest_error[degree] += f64::EPSILON * target_term.abs();
        for s in 0..degree {
            let lower = degree - s;
            for r in 1..=lower {
                if (r, s) == (1, 0) {
                    continue;
                }
                let scale = moments[r + s][s] / (FACTORIAL[r] * FACTORIAL[s]);
                let scale_error =
                    f64::EPSILON * moments_magnitude[r + s][s] / (FACTORIAL[r] * FACTORIAL[s]);
                let (source, source_error) = if r == 1 {
                    (&parts[lower], &parts_error[lower])
                } else {
                    let slice = power_slice(r, lower);
                    (&powers[slice], &powers_error[slice])
                };
                for i in 0..=lower {
                    let term = scale * source[i];
                    rest[i] += term;
                    rest_error[i] += scale_error * source[i].abs()
                        + scale.abs() * source_error[i]
                        + f64::EPSILON * term.abs();
                }
            }
        }
        for i in 0..=degree {
            let part = -rest[i] / h_alpha;
            let error = rest_error[i] / h_alpha.abs() + part.abs() * h_alpha_relative_error;
            if !(error <= error_floor) {
                let mut coefficients = [[0.0_f64; SLOTS]; SLOTS];
                for (lower, part) in parts.iter().enumerate().take(degree).skip(1) {
                    for i in 0..=lower {
                        coefficients[i][lower - i] = part[i];
                    }
                }
                coefficients[0][0] = alpha;
                return Ok((coefficients, Some(degree)));
            }
            parts[degree][i] = part;
            parts_error[degree][i] = error;
        }
    }
    let mut coefficients = [[0.0_f64; SLOTS]; SLOTS];
    for (degree, part) in parts.iter().enumerate().skip(1) {
        for i in 0..=degree {
            coefficients[i][degree - i] = part[i];
        }
    }
    coefficients[0][0] = alpha;
    if !coefficients.iter().flatten().all(|c| c.is_finite()) {
        return Ok((refused, Some(1)));
    }
    Ok((coefficients, None))
}

/// `F^{(n)}(x)·density` for `F(x) = Φ(−x)`, `n < SLOTS`: the Hermite moments
/// `F^{(n)} = (−1)^n He_{n−1}(x) φ(x)` with `φ(x)` replaced by the caller's
/// normalized density.
#[inline]
fn survival_cdf_derivative_stack<const SLOTS: usize>(x: f64, density: f64) -> [f64; SLOTS] {
    const { assert!(SLOTS <= ANCHOR_SIXTH_SLOTS) };
    let x2 = x * x;
    let hermite = [
        0.0,
        -1.0,
        x,
        1.0 - x2,
        x2 * x - 3.0 * x,
        -(x2 * x2 - 6.0 * x2 + 3.0),
        (x2 * x2 - 10.0 * x2 + 15.0) * x,
    ];
    std::array::from_fn(|n| hermite[n] * density)
}

/// The log-space table (gam#3639), for a tail index the linear table has no
/// bits at.
fn anchor_taylor_coefficients_log<const SLOTS: usize, const SLICES: usize>(
    alpha: f64,
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
) -> Result<[[f64; SLOTS]; SLOTS], String> {
    const { assert!(SLICES == (SLOTS - 1) * (SLOTS - 2) / 2) };
    // The equation is solved in logarithms, `Ψ(α, b) = log Σ_k w_k F(α + b u_k) =
    // log F(q)`, and each node's exponent is centred on its mean under the
    // law's own tail probabilities before it is exponentiated (gam#3639).
    //
    // Why: in `F` itself every partial is a Hermite moment, `F^{(n)}(η) =
    // (−1)^n He_{n−1}(η)·φ(η)`, which grows like `η^{n−1}` while `α`'s own
    // derivatives are tiny in a tail (`α ≈ q` plus a smooth correction). At
    // `q = 39` the degree-five equation cancels terms of size `He_4(39)/5! ≈
    // 2e4` down to `∂⁵α/5! ≈ 5e-10`, thirteen orders of magnitude, so the
    // order-five coefficient was mostly rounding: the row NLL's fifth derivative
    // came out `1.6e-5` against an exact `1.22e-7` (90-digit reference), an
    // error larger than every term of its Faà di Bruno sum. `log F` has
    // derivatives that grow at most linearly (`(log F)′ = −λ ≈ −η`, the rest
    // bounded), and centring removes that common slope, so every coefficient
    // the solve reads is `O(1)`.
    //
    // With `G = log F` and `ε_k = δα + u_k·δb`, `Ψ(α + δα, b + δb) − Ψ(α, b) =
    // log E_p[exp(Y_k)]`, `Y_k = Σ_n G^{(n)}(η_k)·ε_k^n / n!`, under the weights
    // `p_k ∝ w_k F(η_k)`. Writing `Ȳ = E_p[Y]` and `Z = Y − Ȳ`, it is `Ȳ +
    // log(1 + E_p[exp(Z) − 1])`, whose `exp` and `log` series act on
    // polynomials without a large common term. `moments[r+s][s]` then holds
    // `∂_α^r ∂_b^s Ψ` and `target[n]` holds `G^{(n)}(q)`, the inputs the
    // degree-by-degree solve below reads for any equation `Ψ(α, b) = T(q)`.
    let degree_limit = SLOTS - 1;
    let mut stacks: SmallVec<[[f64; ANCHOR_SIXTH_SLOTS]; 128]> =
        SmallVec::with_capacity(grid.len());
    let mut log_mass: SmallVec<[f64; 128]> = SmallVec::with_capacity(grid.len());
    let mut heaviest = f64::NEG_INFINITY;
    let complement = q < 0.0;
    for (k, &u) in grid.nodes.iter().enumerate() {
        let eta = alpha + observed_slope * u;
        let stack = anchor_side_log_derivatives(eta, complement);
        let mass = grid.log_weights[k] + stack[0];
        heaviest = heaviest.max(mass);
        stacks.push(stack);
        log_mass.push(mass);
    }
    if !heaviest.is_finite() {
        return Err(format!(
            "survival marginal-slope anchor has no node of finite tail mass at α={alpha}, b={observed_slope}"
        ));
    }
    let mut probabilities: SmallVec<[f64; 128]> =
        log_mass.iter().map(|mass| (mass - heaviest).exp()).collect();
    let total: f64 = probabilities.iter().sum();
    for probability in probabilities.iter_mut() {
        *probability /= total;
    }
    // `Y_k`'s coefficient of `δα^i δb^j` is `G^{(i+j)}(η_k)·u_k^j / (i!·j!)`.
    let exponent = |stack: &[f64; ANCHOR_SIXTH_SLOTS], u: f64| -> [[f64; SLOTS]; SLOTS] {
        let mut poly = [[0.0_f64; SLOTS]; SLOTS];
        let mut u_power = 1.0;
        for j in 0..SLOTS {
            for i in 0..SLOTS - j {
                if i + j >= 1 {
                    poly[i][j] = stack[i + j] * u_power / (FACTORIAL[i] * FACTORIAL[j]);
                }
            }
            u_power *= u;
        }
        poly
    };
    let multiply = |left: &[[f64; SLOTS]; SLOTS], right: &[[f64; SLOTS]; SLOTS]| {
        let mut product = [[0.0_f64; SLOTS]; SLOTS];
        for i1 in 0..SLOTS {
            for j1 in 0..SLOTS - i1 {
                if left[i1][j1] == 0.0 {
                    continue;
                }
                for i2 in 0..SLOTS - i1 - j1 {
                    for j2 in 0..SLOTS - i1 - j1 - i2 {
                        product[i1 + i2][j1 + j2] += left[i1][j1] * right[i2][j2];
                    }
                }
            }
        }
        product
    };
    let mut mean = [[0.0_f64; SLOTS]; SLOTS];
    for ((stack, &u), &probability) in stacks.iter().zip(grid.nodes).zip(&probabilities) {
        let poly = exponent(stack, u);
        for i in 0..SLOTS {
            for j in 0..SLOTS - i {
                mean[i][j] += probability * poly[i][j];
            }
        }
    }
    // `E_p[exp(Z) − 1] = Σ_{m ≥ 2} E_p[Z^m] / m!`: `E_p[Z] = 0`.
    let mut excess = [[0.0_f64; SLOTS]; SLOTS];
    for ((stack, &u), &probability) in stacks.iter().zip(grid.nodes).zip(&probabilities) {
        let mut centred = exponent(stack, u);
        for i in 0..SLOTS {
            for j in 0..SLOTS - i {
                centred[i][j] -= mean[i][j];
            }
        }
        let mut power = centred;
        for order in 2..=degree_limit {
            power = multiply(&power, &centred);
            let weight = probability / FACTORIAL[order];
            for i in 0..SLOTS {
                for j in 0..SLOTS - i {
                    excess[i][j] += weight * power[i][j];
                }
            }
        }
    }
    // `log(1 + W) = Σ_{m ≥ 1} (−1)^{m+1} W^m / m`; `W` starts at degree two, so
    // `m ≤ degree_limit / 2` reaches every degree the table holds.
    let mut log_poly = mean;
    let mut power = excess;
    for order in 1..=degree_limit / 2 {
        if order > 1 {
            power = multiply(&power, &excess);
        }
        let sign = if order % 2 == 1 { 1.0 } else { -1.0 };
        for i in 0..SLOTS {
            for j in 0..SLOTS - i {
                log_poly[i][j] += sign * power[i][j] / order as f64;
            }
        }
    }
    let mut moments = [[0.0_f64; SLOTS]; SLOTS];
    for r in 0..SLOTS {
        for s in 0..SLOTS - r {
            if r + s >= 1 {
                moments[r + s][s] = log_poly[r][s] * FACTORIAL[r] * FACTORIAL[s];
            }
        }
    }
    // The survival side decreases in `α`, the complement side increases.
    let h_alpha = moments[1][0];
    let expected_sign = if complement { 1.0 } else { -1.0 };
    if !(h_alpha.is_finite() && h_alpha * expected_sign > 0.0) {
        return Err(format!(
            "survival marginal-slope anchor has no finite gradient: Ψ_α={h_alpha:e} at α={alpha}, q={q}, b={observed_slope}"
        ));
    }
    let target_stack = anchor_side_log_derivatives(q, complement);
    let target: [f64; SLOTS] = std::array::from_fn(|n| if n == 0 { 0.0 } else { target_stack[n] });
    // parts[d][i]: the coefficient of δq^i·δb^(d−i) in δα. powers holds the
    // degree-e part of δα^r for 2 ≤ r ≤ e only — the slices the solve reads —
    // at `power_slice(r, e)`, so an order-five table zeroes 480 bytes of
    // scratch instead of the whole 6·6·6 cube (gam#2928).
    let mut parts = [[0.0_f64; SLOTS]; SLOTS];
    let mut powers = [[0.0_f64; SLOTS]; SLICES];
    for degree in 1..SLOTS {
        // [δα^r]_degree = Σ_j A_j·[δα^(r−1)]_(degree−j) reads parts below
        // `degree` only.
        for r in 2..=degree {
            let mut power = [0.0_f64; SLOTS];
            for j in 1..=degree + 1 - r {
                let lower = degree - j;
                let right = if r == 2 { &parts[lower] } else { &powers[power_slice(r - 1, lower)] };
                for i1 in 0..=j {
                    for i2 in 0..=lower {
                        power[i1 + i2] += parts[j][i1] * right[i2];
                    }
                }
            }
            powers[power_slice(r, degree)] = power;
        }
        // Everything of degree `degree` in the expanded equation except
        // `H_10·A_degree`: the slope moment `H_{0,degree}·δb^degree / degree!`,
        // the target's `δq^degree` term, and `H_rs·[δα^r]_(degree−s)·δb^s`.
        let mut rest = [0.0_f64; SLOTS];
        rest[0] = moments[degree][degree] / FACTORIAL[degree];
        rest[degree] -= target[degree] / FACTORIAL[degree];
        for s in 0..degree {
            let lower = degree - s;
            for r in 1..=lower {
                if (r, s) == (1, 0) {
                    continue;
                }
                let scale = moments[r + s][s] / (FACTORIAL[r] * FACTORIAL[s]);
                let source = if r == 1 { &parts[lower] } else { &powers[power_slice(r, lower)] };
                for i in 0..=lower {
                    rest[i] += scale * source[i];
                }
            }
        }
        for i in 0..=degree {
            parts[degree][i] = -rest[i] / h_alpha;
        }
    }
    let mut coefficients = [[0.0_f64; SLOTS]; SLOTS];
    for (degree, part) in parts.iter().enumerate().skip(1) {
        for i in 0..=degree {
            coefficients[i][degree - i] = part[i];
        }
    }
    coefficients[0][0] = alpha;
    if !coefficients.iter().flatten().all(|c| c.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor Taylor table is non-finite at α={alpha}, q={q}, b={observed_slope}"
        ));
    }
    Ok(coefficients)
}

/// The anchor's Taylor table through total order six at an already solved
/// anchor, `c[i][j] = ∂_q^i ∂_b^j α / (i!·j!)` for `i + j ≤ 6` (gam#2945).
///
/// The exact fifth likelihood derivative of an event row reads
/// `log α_q(q₁, b)` through order five, which is `α` through order six. Only
/// that kernel reads the sixth order, so the table is solved on demand at the
/// root its slot returns and never published: orders through five are the
/// slot's order-five table bit for bit ([`anchor_taylor_coefficients`]).
pub(crate) fn anchor_taylor_through_sixth(
    alpha: f64,
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
) -> Result<[[f64; ANCHOR_SIXTH_SLOTS]; ANCHOR_SIXTH_SLOTS], String> {
    anchor_taylor_coefficients::<ANCHOR_SIXTH_SLOTS, SIXTH_POWER_SLICES>(
        alpha,
        q,
        observed_slope,
        grid,
    )
}

impl AnchorTaylor {
    /// The table at an already solved anchor.
    pub(crate) fn at(
        alpha: f64,
        q: f64,
        observed_slope: f64,
        grid: AnchorGrid<'_>,
    ) -> Result<Self, String> {
        anchor_taylor_coefficients::<TAYLOR_SLOTS, POWER_SLICES>(alpha, q, observed_slope, grid)
            .map(|coefficients| Self { coefficients })
    }

    /// `c[i][j] = ∂_q^i ∂_b^j α / (i!·j!)` for `i + j ≤ 5`, zero above.
    #[inline]
    pub(crate) fn coefficients(&self) -> &[[f64; TAYLOR_SLOTS]; TAYLOR_SLOTS] {
        &self.coefficients
    }

    /// The anchor and its implicit derivatives through order three, from the
    /// table: `∂_q^i ∂_b^j α = i!·j!·c[i][j]`.
    pub(crate) fn derivatives(&self) -> AnchorDerivatives {
        let c = &self.coefficients;
        AnchorDerivatives {
            alpha: c[0][0],
            a_q: c[1][0],
            a_b: c[0][1],
            a_qq: 2.0 * c[2][0],
            a_qb: c[1][1],
            a_bb: 2.0 * c[0][2],
            a_qqq: 6.0 * c[3][0],
            a_qqb: 2.0 * c[2][1],
            a_qbb: 2.0 * c[1][2],
            a_bbb: 6.0 * c[0][3],
        }
    }

    /// `∂_q^i ∂_b^j α = i!·j!·c[i][j]` for any `i + j ≤ 5`. The table is
    /// solved in density-normalized units, so every partial it holds stays
    /// finite where each node's density underflows; a consumer that reads the
    /// anchor's fifth order from here needs no linear-probability Jacobian
    /// (gam#3639).
    pub(crate) fn partial(&self, i: usize, j: usize) -> f64 {
        assert!(
            i + j <= ANCHOR_TAYLOR_ORDER,
            "anchor Taylor table holds partials through total order {ANCHOR_TAYLOR_ORDER}, asked for ({i}, {j})"
        );
        FACTORIAL[i] * FACTORIAL[j] * self.coefficients[i][j]
    }

    /// `(i, j)` of each coefficient a [`RootSlot`] stores beside the root, by
    /// total degree: `c[d][0], c[d−1][1], …, c[0][d]` for `d = 1..=5`.
    fn stored_entries() -> impl Iterator<Item = (usize, usize)> {
        (1..TAYLOR_SLOTS).flat_map(|degree| (0..=degree).rev().map(move |i| (i, degree - i)))
    }

    /// `α` over any carrier: the table composed with `(q − q*, b − b*)`.
    /// `delta_b` is the observed-slope carrier with its value channel zero.
    #[inline]
    pub(crate) fn lift<T: JetField>(&self, q: &T, delta_b: &T) -> T {
        self.lift_q_derivative_of_order(q, delta_b, 0)
    }

    /// `∂α/∂q` over any carrier, from the same table one order higher in `q`.
    #[inline]
    pub(crate) fn lift_q_derivative<T: JetField>(&self, q: &T, delta_b: &T) -> T {
        self.lift_q_derivative_of_order(q, delta_b, 1)
    }

    /// `∂_q^shift α` by Horner's rule in `δb` over univariate compositions in
    /// `q`: `Σ_j p_j(q)·δb^j`, with `p_j`'s derivative stack
    /// `(i + shift)!·c[i + shift][j]`. Carriers truncate at order four, so the
    /// `j = 4` slice is the last one any carrier reads, and the value channel
    /// is `shift!·c[shift][0]` exactly because `δb` carries a zero value.
    fn lift_q_derivative_of_order<T: JetField>(&self, q: &T, delta_b: &T, shift: usize) -> T {
        let slice = |j: usize| -> [f64; 5] {
            std::array::from_fn(|i| {
                let source = i + shift;
                if source + j <= ANCHOR_TAYLOR_ORDER {
                    FACTORIAL[source] * self.coefficients[source][j]
                } else {
                    0.0
                }
            })
        };
        let mut acc = q.compose_unary(slice(4));
        for j in (0..4).rev() {
            acc = q.compose_unary(slice(j)).add(&delta_b.mul(&acc));
        }
        acc
    }
}

#[cfg(test)]
mod anchor_tests {
    use crate::test_support::{gauss_hermite_probabilists, skewed_grid};
    use super::*;
    use gam_math::jet_scalar::{JetScalar, Order2};
    use gam_math::jet_tower::Tower4;
    use gam_math::probability::{normal_cdf, normal_pdf};

    /// The linear-space table is published where it keeps its bits and agrees
    /// with the log-space table there to those bits; at a tail index it has
    /// none and steps aside for the log table (gam#3639).
    #[test]
    fn linear_table_keeps_its_bits_in_the_bulk_and_steps_aside_in_the_tail() {
        let grid = skewed_grid();
        let mut published_linear = 0;
        for q in [-3.0_f64, -1.2, -0.3, 0.0, 0.4, 1.5, 3.0] {
            for observed_slope in [0.05_f64, 0.4, 1.3] {
                let alpha = solve_anchor(q, observed_slope, grid.view()).unwrap();
                let log_table = anchor_taylor_coefficients_log::<TAYLOR_SLOTS, POWER_SLICES>(
                    alpha,
                    q,
                    observed_slope,
                    grid.view(),
                )
                .unwrap();
                let (linear, failed) = anchor_taylor_coefficients_linear::<TAYLOR_SLOTS, POWER_SLICES>(
                    alpha,
                    q,
                    observed_slope,
                    grid.view(),
                )
                .unwrap();
                if failed.is_some() {
                    continue;
                }
                published_linear += 1;
                let bar = 2.0_f64.powi(-LINEAR_TABLE_MIN_BITS) * (1.0 + alpha.abs());
                for i in 0..TAYLOR_SLOTS {
                    for j in 0..TAYLOR_SLOTS - i {
                        let (a, b) = (linear[i][j], log_table[i][j]);
                        assert!(
                            (a - b).abs() <= bar,
                            "q={q} b={observed_slope} coefficient ({i}, {j}): linear {a:e} vs log {b:e}"
                        );
                    }
                }
                let published = anchor_taylor_coefficients::<TAYLOR_SLOTS, POWER_SLICES>(
                    alpha,
                    q,
                    observed_slope,
                    grid.view(),
                )
                .unwrap();
                for i in 0..TAYLOR_SLOTS {
                    for j in 0..TAYLOR_SLOTS - i {
                        assert_eq!(
                            published[i][j].to_bits(),
                            linear[i][j].to_bits(),
                            "the published table is the linear one where it has its bits ({i}, {j})"
                        );
                    }
                }
            }
        }
        assert!(published_linear >= 15, "the bulk publishes the linear table ({published_linear} of 21)");
        let q = 39.0;
        let observed_slope = 0.3;
        let alpha = solve_anchor(q, observed_slope, grid.view()).unwrap();
        assert!(
            anchor_taylor_coefficients_linear::<TAYLOR_SLOTS, POWER_SLICES>(alpha, q, observed_slope, grid.view())
                .unwrap()
                .1
                .is_some(),
            "a tail index has no linear-space bits"
        );
        // An order-six table extends the published order-five table bitwise on
        // the bulk point whichever path its sixth degree takes.
        let (q, observed_slope) = (0.4, 0.4);
        let alpha = solve_anchor(q, observed_slope, grid.view()).unwrap();
        let fifth = AnchorTaylor::at(alpha, q, observed_slope, grid.view()).unwrap();
        let sixth = anchor_taylor_through_sixth(alpha, q, observed_slope, grid.view()).unwrap();
        for i in 0..TAYLOR_SLOTS {
            for j in 0..TAYLOR_SLOTS - i {
                assert_eq!(fifth.coefficients()[i][j].to_bits(), sixth[i][j].to_bits(), "({i}, {j})");
            }
        }
    }

    /// The implicit first partials of the anchor agree with the order-one
    /// entries of its Taylor table on a skewed law, on both sides of the
    /// marginal and far into the tails, where every node density underflows.
    #[test]
    fn implicit_first_partials_match_the_taylor_table() {
        let grid = skewed_grid();
        for q in [-9.0_f64, -3.5, -1.0, -0.2, 0.0, 0.3, 1.7, 4.0, 8.5] {
            for observed_slope in [0.0_f64, 0.05, 0.4, 1.3, 2.8] {
                let alpha = solve_anchor(q, observed_slope, grid.view()).unwrap();
                let table = AnchorTaylor::at(alpha, q, observed_slope, grid.view())
                    .unwrap()
                    .derivatives();
                let (a_q, a_b) = anchor_first_derivatives(alpha, q, observed_slope, grid.view()).unwrap();
                let tol = |reference: f64| 1e-10 * reference.abs().max(1e-3);
                assert!(
                    (a_q - table.a_q).abs() <= tol(table.a_q),
                    "q={q} b={observed_slope}: a_q {a_q} vs table {}",
                    table.a_q
                );
                assert!(
                    (a_b - table.a_b).abs() <= tol(table.a_b),
                    "q={q} b={observed_slope}: a_b {a_b} vs table {}",
                    table.a_b
                );
            }
        }
    }

    /// Whatever seed a solve starts from, it returns the same root to
    /// rounding: a warm start from a neighbouring node's root is a work
    /// choice, not a value choice.
    #[test]
    fn warm_started_solve_returns_the_seed_free_root() {
        let grid = skewed_grid();
        for q in [-6.0_f64, -1.5, 0.4, 3.0] {
            for observed_slope in [0.1_f64, 0.9, 2.2] {
                let cold = solve_anchor(q, observed_slope, grid.view()).unwrap();
                for seed in [cold + 0.3, cold - 0.25, cold + 1.5, 0.0] {
                    let warm = solve_anchor_from(q, observed_slope, grid.view(), seed).unwrap();
                    assert!(
                        (warm - cold).abs() <= 1e-12 * (1.0 + cold.abs()),
                        "q={q} b={observed_slope} seed={seed}: {warm} vs {cold}"
                    );
                }
            }
        }
    }

    // ── The standard-normal cells route in log units (gam#4504) ──

    /// The affine cells `(α, β)` of the line cut at `splits`, whose values sum
    /// to `Φ(α/√(1 + β²))` exactly, each with `∂c/∂a = (1, 0, 0, 0)` and
    /// `∂²c/∂a² = 0`.
    fn affine_line_cells(
        alpha: f64,
        beta: f64,
        splits: &[f64],
    ) -> Vec<(DenestedCubicCell, [f64; 4], [f64; 4])> {
        let edges: Vec<f64> = std::iter::once(f64::NEG_INFINITY)
            .chain(splits.iter().copied())
            .chain(std::iter::once(f64::INFINITY))
            .collect();
        edges
            .windows(2)
            .map(|edge| {
                let cell = DenestedCubicCell {
                    left: edge[0],
                    right: edge[1],
                    c0: alpha,
                    c1: beta,
                    c2: 0.0,
                    c3: 0.0,
                };
                (cell, [1.0, 0.0, 0.0, 0.0], [0.0; 4])
            })
            .collect()
    }

    /// gam#4504: past `|q| ≈ 37.5` every cell of a standard-normal row is below
    /// `f64::MIN_POSITIVE`, so the linear cell sum is zero and its log tail
    /// `−∞`. The log-space cells hold the tail there. On affine cells cut
    /// anywhere along the line the exact tail is `Φ(h)`, `h = α/√(1 + β²)`, with
    /// `P′/T = λ(h)/s` and `P″/T = ∓h·λ(h)/s²` (`λ = φ/Φ`, `s = √(1 + β²)`),
    /// so at `h = −40` and `h = −60`, on both sides and three partitions, the log
    /// route must match those to its own charged bound plus the residual's
    /// rounding at that `log T`.
    ///
    /// Control: the same cells through the linear evaluator
    /// ([`crate::cubic_cell_kernel::evaluate_cell_moments`]) fail
    /// [`linear_cell_tail_is_representable`], and their log tail misses
    /// `log Φ(h)` by more than that band. At `h = −20`, where the linear sum
    /// holds, the linear sum passes the predicate, and both routes match the
    /// exact tail to their bands.
    #[test]
    fn log_space_cells_hold_the_exact_tail_where_the_linear_sum_underflows_4504() {
        use crate::cubic_cell_kernel::evaluate_cell_moments;
        let partitions: [&[f64]; 3] = [&[], &[-2.0, 0.5, 3.0], &[-45.0, -1.0, 0.0, 2.5, 30.0, 41.0]];
        for (h, linear_holds) in [(-20.0_f64, true), (-40.0, false), (-60.0, false)] {
            let (log_reference, mills) = signed_probit_logcdf_and_mills_ratio(h);
            for beta in [0.35_f64, 3.0, -2.0] {
                let s = beta.hypot(1.0);
                for splits in partitions {
                    for survival_side in [false, true] {
                        // The complement side reads `Φ(α/s)` at `α = h·s`; the
                        // survival side reads `Φ(−α/s)` at `α = −h·s`.
                        let sign = if survival_side { -1.0 } else { 1.0 };
                        let cells = affine_line_cells(sign * h * s, sign * beta, splits);
                        let label = format!(
                            "h={h} beta={beta} cells={} survival={survival_side}",
                            cells.len()
                        );
                        let tail = sum_denested_cells_in_log_space(survival_side, true, cells.clone())
                            .expect("log-space cells");
                        let band = tail.tail_rounding_ratio
                            + anchor_residual_rounding(log_reference, tail.summands);
                        assert!(
                            (tail.log_tail - log_reference).abs() <= band,
                            "{label}: log T {:+.17e} against log Φ(h) {log_reference:+.17e} (band {band:e})",
                            tail.log_tail
                        );
                        let density_reference = mills / s;
                        assert!(
                            (tail.density_ratio - density_reference).abs() <= band * density_reference,
                            "{label}: P'/T {:+.17e} against λ(h)/s {density_reference:+.17e}",
                            tail.density_ratio
                        );
                        let slope_reference = -sign * h * mills / (s * s);
                        let slope = tail.density_slope_ratio.expect("P''/T is read");
                        assert!(
                            (slope - slope_reference).abs() <= band * slope_reference.abs(),
                            "{label}: P''/T {slope:+.17e} against ∓h·λ(h)/s² {slope_reference:+.17e}"
                        );

                        let linear = cells.iter().try_fold(
                            (0.0, 0.0),
                            |(value, rounding), (cell, _, _)| {
                                evaluate_cell_moments(
                                    if survival_side { cell.negated() } else { *cell },
                                    4,
                                )
                                .map(|state| (value + state.value, rounding + state.value_rounding))
                            },
                        );
                        match (linear_holds, linear) {
                            (true, Ok((linear, linear_rounding))) => {
                                let linear_log = linear.ln();
                                let linear_band = linear_rounding / linear
                                    + anchor_residual_rounding(log_reference, tail.summands);
                                assert!(
                                    linear_cell_tail_is_representable(
                                        linear,
                                        linear_rounding,
                                        tail.summands
                                    ) && (linear_log - log_reference).abs() <= linear_band,
                                    "{label}: the linear sum holds here: log T {linear_log:+.17e} \
                                     against {log_reference:+.17e} (band {linear_band:e})"
                                );
                            }
                            (true, Err(error)) => panic!("{label}: linear cell: {error}"),
                            (false, Ok((linear, linear_rounding))) => {
                                let linear_log = linear.ln();
                                assert!(
                                    !linear_cell_tail_is_representable(
                                        linear,
                                        linear_rounding,
                                        tail.summands
                                    ) && !((linear_log - log_reference).abs() <= band),
                                    "{label}: control: the linear sum {linear:e} (log \
                                     {linear_log:e}) should have lost log Φ(h) = {log_reference:e}"
                                );
                            }
                            // A linear cell that refuses has lost the value as well.
                            (false, Err(_)) => {}
                        }
                    }
                }
            }
        }
    }

    // ── The smaller-tail log residual: its own rounding and underflow (gam#3216, PR #3374) ──

    /// The survival-side calibration `T(α) = Φ(−α)` with root `α = q`, perturbed
    /// by the factor `exp(±r)` on the side of the root it is evaluated on and
    /// carrying `bound` as its rounding.
    fn perturbed_survival_tail(alpha: f64, q: f64, r: f64, bound: f64) -> CalibrationTail {
        let exact = normal_cdf(-alpha);
        let above_target = exact.ln() > smaller_tail_log_target(q);
        let factor = if above_target { r.exp() } else { (-r).exp() };
        CalibrationTail::from_linear(exact * factor, normal_pdf(alpha), None, 1, bound * exact)
    }

    /// gam#3216 (a), from PR #3374: the tail's own rounding bound enters the
    /// residual's resolution. A tail that is right only to a relative bound
    /// `r` far above the logarithms' rounding can put the residual's sign
    /// change across a jump of `2r`, as a sum of cell integrals that is not
    /// correctly rounded can: here `T(α) = Φ(−α)·exp(±r)`, whose residual is at
    /// least `r` on every float. Charged its bound, the solve accepts a float
    /// at the jump; with the bound dropped it bisects down to adjacent floats
    /// and refuses the root.
    #[test]
    fn a_tail_bound_above_the_logarithms_rounding_is_resolvable_3216() {
        let q = 5.0;
        let log_target = smaller_tail_log_target(q);
        let r = 1e-10;
        assert!(r > ANCHOR_LOG_RESIDUAL_TOL.max(anchor_residual_rounding(log_target, 1)));
        let solve = |bound: f64| {
            solve_log_tail_root(
                q + 3.0,
                true,
                |alpha| perturbed_survival_tail(alpha, q, r, bound).log_residual(true, log_target),
                "perturbed survival tail",
                String::new,
            )
        };
        let refused = solve(0.0).expect_err("uncharged, no float resolves the residual");
        assert!(refused.contains("adjacent floats"), "{refused}");
        // `|T − Φ(−α)| ≤ (e^r − 1)·Φ(−α)`, the bound the tail declares.
        let (root, _) = solve(r.exp_m1()).expect("charged, the jump resolves");
        // A float at the jump carries a residual within the tail's bound, so the
        // root is within that bound's `α`-width of the true one.
        let first = normal_pdf(q) / normal_cdf(-q);
        assert!(
            (root - q).abs() * first <= 2.0 * r.exp_m1(),
            "root {root} vs {q}: residual width {:e} against the bound {:e}",
            (root - q).abs() * first,
            r.exp_m1()
        );
    }

    /// gam#3216 (b), from PR #3374: a probe whose tail underflowed keeps its
    /// residual's sign. With `T(α) = Φ(−α)` and the root at `α = 5`, a solve
    /// seeded where `Φ(−α)` rounds to zero reads `log T = −∞`, which says the
    /// root lies below, narrows its bracket and reaches the root; the old
    /// residual refused the seed and ended the solve.
    #[test]
    fn a_far_probe_whose_tail_underflows_keeps_the_bracket_3216() {
        let q = 5.0;
        let seed = 60.0;
        let log_target = smaller_tail_log_target(q);
        assert_eq!(normal_cdf(-seed), 0.0, "the seed's tail must underflow");
        let tail = |alpha: f64| perturbed_survival_tail(alpha, q, 0.0, 0.0);
        let (root, evaluations) = solve_log_tail_root(
            seed,
            true,
            |alpha| tail(alpha).log_residual(true, log_target),
            "underflowing survival tail",
            String::new,
        )
        .expect("the underflowed seed narrows the bracket instead of ending the solve");
        let at_root = tail(root)
            .log_residual(true, log_target)
            .expect("finite residual at the root");
        let root_resolution = 4.0 * at_root.rounding.max(ANCHOR_LOG_RESIDUAL_TOL) / at_root.first.abs();
        assert!(
            (root - q).abs() <= root_resolution,
            "root {root} vs {q} after {evaluations} evaluations (resolution {root_resolution:e})"
        );
        let underflowed = tail(seed).log_residual(true, log_target).expect("a sign, not an error");
        assert!(underflowed.value == f64::NEG_INFINITY && underflowed.rounding == f64::INFINITY);
    }

    // ── The closed-form implicit derivatives the table's orders ≤ 3 are pinned against ──

    /// Which of `(q, b)` an implicit-derivative index names.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Var {
        Q,
        B,
    }

    impl AnchorDerivatives {
        /// The implicit derivatives through order three in closed form, at an
        /// already solved anchor: implicit differentiation of `G(α, q, b) =
        /// Σ_k w_k Φ(−(α + b u_k)) − Φ(−q) = 0` written out order by order.
        pub(crate) fn at(
            alpha: f64,
            q: f64,
            observed_slope: f64,
            grid: AnchorGrid<'_>,
        ) -> Result<Self, String> {
            // Moments of the law against the derivatives of `F(x) = Φ(−x)`:
            //   F′ = −φ, F″ = xφ, F‴ = (1 − x²)φ,
            // each weighted by `u_k^j` for the `b`-derivatives. Every implicit
            // derivative is homogeneous of degree zero in `G`, so `G` is divided
            // through by `Σ_k w_k φ(η_k)` (gam#2941): the moments read the
            // normalized weights `ω_k`, and `φ(q)` becomes the finite ratio `ρ`.
            let density = AnchorDensity::at(alpha, observed_slope, grid)?;
            let mut p = [0.0_f64; 4];
            let mut s2 = [0.0_f64; 4];
            let mut s3 = [0.0_f64; 4];
            for (&u, &omega) in grid.nodes.iter().zip(density.weights().iter()) {
                let eta = alpha + observed_slope * u;
                let mut power = 1.0;
                for j in 0..4 {
                    p[j] += omega * power;
                    s2[j] += omega * eta * power;
                    s3[j] += omega * (1.0 - eta * eta) * power;
                    power *= u;
                }
            }
            // `ρ = φ(q) / Σ_k w_k φ(η_k)`, formed about the heaviest node `*` with the
            // quadratic terms as the product `(q − η_*)(q + η_*)`: at a tail root `q`
            // and `η_*` agree to many digits and their squares would not.
            let (heaviest_log_weight, heaviest_eta) = grid
                .nodes
                .iter()
                .zip(grid.log_weights)
                .map(|(&u, &log_weight)| (log_weight, alpha + observed_slope * u))
                .max_by(|left, right| {
                    (left.0 - 0.5 * left.1 * left.1).total_cmp(&(right.0 - 0.5 * right.1 * right.1))
                })
                .expect("the grid has a node");
            let log_relative_sum = grid
                .nodes
                .iter()
                .zip(grid.log_weights)
                .map(|(&u, &log_weight)| {
                    let eta = alpha + observed_slope * u;
                    ((log_weight - heaviest_log_weight)
                        - 0.5 * (eta - heaviest_eta) * (eta + heaviest_eta))
                        .exp()
                })
                .sum::<f64>()
                .ln();
            let rho = (-heaviest_log_weight
                - log_relative_sum
                - 0.5 * (q - heaviest_eta) * (q + heaviest_eta))
                .exp();
            // Partial derivatives of G by order. In `q` alone: `−F^{(n)}(q) =
            // −(−1)^n He_{n−1}(q) φ(q)`; in `(α, b)` the law's moments above, by
            // total order and by the number of `b`'s. Order zero is never read.
            let g_q = [0.0, rho, -q * rho, -(1.0 - q * q) * rho];
            let g_ab = [[0.0; 4], [-p[0], -p[1], -p[2], -p[3]], s2, s3];
            // `na` in α, `nq` in q, `nb` in b; mixed q/(α, b) derivatives vanish.
            let g = |na: usize, nq: usize, nb: usize| -> f64 {
                if nq > 0 {
                    if na + nb > 0 { 0.0 } else { g_q[nq] }
                } else {
                    g_ab[na + nb][nb]
                }
            };
            let g_alpha = g(1, 0, 0);
            if !(g_alpha.is_finite() && g_alpha < 0.0) {
                return Err(format!(
                    "survival marginal-slope anchor has no finite gradient: G_α={g_alpha:e} at α={alpha}, q={q}, b={observed_slope}"
                ));
            }
            let idx = |v: Var| -> (usize, usize) {
                match v {
                    Var::Q => (1, 0),
                    Var::B => (0, 1),
                }
            };
            // G with `na` α-derivatives and the listed (q, b) variables.
            let gp = |na: usize, vars: &[Var]| -> f64 {
                let mut nq = 0;
                let mut nb = 0;
                for v in vars {
                    let (dq, db) = idx(*v);
                    nq += dq;
                    nb += db;
                }
                g(na, nq, nb)
            };
            let first = |x: Var| -> f64 { -gp(0, &[x]) / g_alpha };
            let a1 = |x: Var| first(x);
            let second = |x: Var, y: Var| -> f64 {
                -(gp(0, &[x, y]) + gp(1, &[x]) * a1(y) + gp(1, &[y]) * a1(x) + gp(2, &[]) * a1(x) * a1(y))
                    / g_alpha
            };
            let third = |x: Var, y: Var, z: Var| -> f64 {
                let ax = a1(x);
                let ay = a1(y);
                let az = a1(z);
                let axy = second(x, y);
                let axz = second(x, z);
                let ayz = second(y, z);
                -(gp(0, &[x, y, z])
                    + gp(1, &[x, y]) * az
                    + (gp(1, &[x, z]) + gp(2, &[x]) * az) * ay
                    + gp(1, &[x]) * ayz
                    + (gp(1, &[y, z]) + gp(2, &[y]) * az) * ax
                    + gp(1, &[y]) * axz
                    + (gp(2, &[z]) + gp(3, &[]) * az) * ax * ay
                    + gp(2, &[]) * (axz * ay + ax * ayz)
                    + (gp(1, &[z]) + gp(2, &[]) * az) * axy)
                    / g_alpha
            };
            let out = Self {
                alpha,
                a_q: first(Var::Q),
                a_b: first(Var::B),
                a_qq: second(Var::Q, Var::Q),
                a_qb: second(Var::Q, Var::B),
                a_bb: second(Var::B, Var::B),
                a_qqq: third(Var::Q, Var::Q, Var::Q),
                a_qqb: third(Var::Q, Var::Q, Var::B),
                a_qbb: third(Var::Q, Var::B, Var::B),
                a_bbb: third(Var::B, Var::B, Var::B),
            };
            let implicit = [
                out.a_q, out.a_b, out.a_qq, out.a_qb, out.a_bb, out.a_qqq, out.a_qqb, out.a_qbb, out.a_bbb,
            ];
            if !implicit.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "survival marginal-slope anchor derivatives are not representable at α={alpha}, q={q}, b={observed_slope}: {implicit:?}"
                ));
            }
            Ok(out)
        }
    }

    // ── The jet-algebra oracle the Taylor table is pinned against ──────────

    /// `[Φ(−x), F′, F″, F‴, F⁗]` with `F(x) = Φ(−x)`: `F^{(n)} = (−1)^n He_{n−1}(x) φ(x)`.
    fn survival_cdf_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        [
            normal_cdf(-x),
            -pdf,
            x * pdf,
            (1.0 - x * x) * pdf,
            (x * x * x - 3.0 * x) * pdf,
        ]
    }

    /// `[F′, F″, F‴, F⁗, F⁽⁵⁾]` of the same `F`: the stack of `−φ`.
    fn survival_cdf_prime_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        let x2 = x * x;
        [
            -pdf,
            x * pdf,
            (1.0 - x2) * pdf,
            (x2 * x - 3.0 * x) * pdf,
            -(x2 * x2 - 6.0 * x2 + 3.0) * pdf,
        ]
    }

    /// `[φ, φ′, φ″, φ‴, φ⁗]`: `φ^{(n)} = (−1)^n He_n(x) φ(x)`.
    fn pdf_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        let x2 = x * x;
        [
            pdf,
            -x * pdf,
            (x2 - 1.0) * pdf,
            -(x2 * x - 3.0 * x) * pdf,
            (x2 * x2 - 6.0 * x2 + 3.0) * pdf,
        ]
    }

    /// `[1/x, −1/x², 2/x³, −6/x⁴, 24/x⁵]`.
    fn reciprocal_stack(x: f64) -> [f64; 5] {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        [inv, -inv2, 2.0 * inv3, -6.0 * inv3 * inv, 24.0 * inv3 * inv2]
    }

    /// The anchor as a jet in whatever `q` and `b` carry, by Newton's
    /// iteration in the truncated jet algebra: from the exact real root one
    /// step is exact through first order, two through third, three through
    /// seventh. The value channel is pinned back to the solver's root.
    /// `alpha_star` MUST be the converged root at `(q.value(), b.value())`.
    fn anchor_jet<T: JetField>(alpha_star: f64, q: &T, b: &T, grid: AnchorGrid<'_>) -> T {
        let mut alpha = q.constant_like(alpha_star);
        for _ in 0..3 {
            // G = Σ_k w_k F(η_k) − F(q),   G_α = Σ_k w_k F′(η_k).
            let mut g = q.compose_unary(survival_cdf_stack(q.value())).neg();
            let mut g_alpha = q.constant_like(0.0);
            for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
                let eta = alpha.add(&b.scale(u));
                let eta_value = eta.value();
                g = g.add(&eta.compose_unary(survival_cdf_stack(eta_value)).scale(w));
                g_alpha =
                    g_alpha.add(&eta.compose_unary(survival_cdf_prime_stack(eta_value)).scale(w));
            }
            let step = g.mul(&g_alpha.compose_unary(reciprocal_stack(g_alpha.value())));
            alpha = alpha.sub(&step);
        }
        alpha.with_value(alpha_star)
    }

    /// `∂α/∂q = φ(q) / Σ_k w_k φ(η_k)` as a jet, at an anchor jet already
    /// lifted by [`anchor_jet`].
    fn anchor_q_derivative_jet<T: JetField>(alpha: &T, q: &T, b: &T, grid: AnchorGrid<'_>) -> T {
        let phi_q = q.compose_unary(pdf_stack(q.value()));
        let mut density = q.constant_like(0.0);
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let eta = alpha.add(&b.scale(u));
            density = density.add(&eta.compose_unary(pdf_stack(eta.value())).scale(w));
        }
        phi_q.mul(&density.compose_unary(reciprocal_stack(density.value())))
    }

    /// `∂α/∂b = −Σ_k w_k φ(η_k) u_k / Σ_k w_k φ(η_k)` as a jet: the factor a
    /// follow-up-varying slope's rate `ḃ₁` enters `α̇₁` with.
    fn anchor_b_derivative_jet<T: JetField>(alpha: &T, q: &T, b: &T, grid: AnchorGrid<'_>) -> T {
        let mut density = q.constant_like(0.0);
        let mut moment = q.constant_like(0.0);
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let eta = alpha.add(&b.scale(u));
            let pdf = eta.compose_unary(pdf_stack(eta.value()));
            density = density.add(&pdf.scale(w));
            moment = moment.add(&pdf.scale(w * u));
        }
        moment.mul(&density.compose_unary(reciprocal_stack(density.value()))).neg()
    }

    fn owned(nodes: Vec<f64>, weights: Vec<f64>) -> AnchorGridOwned {
        AnchorGridOwned::new(nodes, weights)
    }

    /// Solve the anchor and differentiate it.
    fn solve_derivatives(q: f64, b: f64, grid: AnchorGrid<'_>) -> Result<AnchorDerivatives, String> {
        AnchorDerivatives::at(solve_anchor(q, b, grid)?, q, b, grid)
    }

    fn marginal(alpha: f64, b: f64, grid: AnchorGrid<'_>) -> f64 {
        grid.nodes
            .iter()
            .zip(grid.weights.iter())
            .map(|(&u, &w)| w * normal_cdf(-(alpha + b * u)))
            .sum()
    }

    #[test]
    fn gauss_hermite_law_integrates_moments() {
        for m in [2usize, 5, 16, 33, 65] {
            let (nodes, weights) = gauss_hermite_probabilists(m).expect("grid");
            assert_eq!(nodes.len(), m);
            let total: f64 = weights.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "m={m} total={total}");
            for k in 1..m {
                assert!(nodes[k] > nodes[k - 1], "m={m} nodes must ascend");
            }
            let mean: f64 = nodes.iter().zip(&weights).map(|(u, w)| u * w).sum();
            let var: f64 = nodes.iter().zip(&weights).map(|(u, w)| u * u * w).sum();
            assert!(mean.abs() < 1e-12, "m={m} mean={mean}");
            assert!((var - 1.0).abs() < 1e-10, "m={m} var={var}");
            if m >= 4 {
                let fourth: f64 = nodes.iter().zip(&weights).map(|(u, w)| u.powi(4) * w).sum();
                assert!((fourth - 3.0).abs() < 1e-9, "m={m} fourth={fourth}");
            }
        }
    }

    /// The anchoring equation is solved: the anchored marginal survival is
    /// `Φ(−q)` to the log-residual tolerance, across both tails and slopes of
    /// both signs, on a law that is nothing like Gaussian.
    #[test]
    fn anchor_solves_the_anchoring_equation_on_a_skewed_law() {
        let grid = skewed_grid();
        for &q in &[-3.5, -1.2, -0.3, 0.0, 0.4, 1.7, 3.2, 5.5] {
            for &b in &[-1.3, -0.4, 0.0, 0.25, 0.9, 2.1] {
                let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                let lhs = marginal(alpha, b, grid.view());
                let rhs = normal_cdf(-q);
                let smaller = rhs.min(1.0 - rhs);
                assert!(
                    (lhs - rhs).abs() <= 1e-10 * smaller.max(1e-300),
                    "q={q} b={b}: Σ w Φ(−η) = {lhs:e} vs Φ(−q) = {rhs:e}"
                );
            }
        }
    }

    /// Gaussian is the special case: on a Gauss–Hermite law the anchor is the
    /// closed form `q·√(1 + b²)` to quadrature tolerance, and its derivatives
    /// are the closed form's.
    #[test]
    fn gaussian_grid_reproduces_the_closed_form() {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("grid");
        let grid = owned(nodes, weights);
        for &q in &[-2.5, -0.7, 0.0, 0.6, 2.2] {
            for &b in &[-1.1, -0.3, 0.0, 0.45, 1.6] {
                let d = solve_derivatives(q, b, grid.view()).expect("anchor");
                let c = (1.0 + b * b).sqrt();
                assert!((d.alpha - q * c).abs() < 1e-9, "q={q} b={b}: α={} vs {}", d.alpha, q * c);
                assert!((d.a_q - c).abs() < 1e-8, "q={q} b={b}: α_q={} vs {c}", d.a_q);
                assert!((d.a_b - q * b / c).abs() < 1e-8, "q={q} b={b}: α_b={} vs {}", d.a_b, q * b / c);
                assert!(d.a_qq.abs() < 1e-8, "q={q} b={b}: α_qq={}", d.a_qq);
                assert!((d.a_qb - b / c).abs() < 1e-7, "q={q} b={b}: α_qb={} vs {}", d.a_qb, b / c);
                assert!((d.a_bb - q / (c * c * c)).abs() < 1e-7, "q={q} b={b}: α_bb={} vs {}", d.a_bb, q / c.powi(3));
            }
        }
    }

    /// The declared grid has to be sized to the drive `b`: a Gauss–Hermite law
    /// reproduces the closed form `q·√(1 + b²)` only to a tolerance that
    /// shrinks with its node count, and at slopes of 2–4 a 16-node law is
    /// visibly wrong where 64 nodes are at 1e-4 and 128 at round-off.
    #[test]
    fn gauss_hermite_grid_size_sets_the_closed_form_tolerance() {
        let mut previous = f64::INFINITY;
        for (m, bound) in [(16_usize, 5e-1), (64, 2e-3), (128, 1e-5)] {
            let (nodes, weights) = gauss_hermite_probabilists(m).expect("grid");
            let grid = owned(nodes, weights);
            let mut worst = 0.0_f64;
            for &b in &[2.0, 3.0, 4.0] {
                for &q in &[-1.0, 0.5, 2.0] {
                    let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                    worst = worst.max((alpha - q * (1.0 + b * b).sqrt()).abs());
                }
            }
            eprintln!(
                "[2923 grid] m={m}: max |α − q√(1+b²)| over b ∈ {{2, 3, 4}}, q ∈ {{−1, 0.5, 2}} = {worst:.3e}"
            );
            assert!(worst < bound, "m={m}: {worst:.3e} exceeds {bound:.1e}");
            assert!(
                worst < previous,
                "m={m}: the tolerance must shrink with the node count ({worst:.3e} vs {previous:.3e})"
            );
            previous = worst;
        }
    }

    /// Every closed-form implicit derivative through order three matches a
    /// central difference of the solved anchor on the skewed law.
    #[test]
    fn implicit_derivatives_match_finite_differences() {
        let grid = skewed_grid();
        let solve = |q: f64, b: f64| solve_anchor(q, b, grid.view()).expect("anchor");
        // Central differences carry an `h²` truncation term; at `1e-4` it sits
        // below the tolerances while round-off on the closed-form derivatives
        // (each exact to ~1e-13) stays near `1e-9`.
        let h = 1e-4;
        for &q in &[-1.4, -0.2, 0.5, 1.9] {
            for &b in &[-0.8, 0.1, 0.7, 1.5] {
                let d = solve_derivatives(q, b, grid.view()).expect("derivatives");
                let f = |dq: f64, db: f64| solve(q + dq, b + db);
                // First order: five-point stencils.
                let fd_q = (-f(2.0 * h, 0.0) + 8.0 * f(h, 0.0) - 8.0 * f(-h, 0.0) + f(-2.0 * h, 0.0)) / (12.0 * h);
                let fd_b = (-f(0.0, 2.0 * h) + 8.0 * f(0.0, h) - 8.0 * f(0.0, -h) + f(0.0, -2.0 * h)) / (12.0 * h);
                assert!((d.a_q - fd_q).abs() < 1e-8 * (1.0 + fd_q.abs()), "q={q} b={b}: α_q {} vs fd {fd_q}", d.a_q);
                assert!((d.a_b - fd_b).abs() < 1e-8 * (1.0 + fd_b.abs()), "q={q} b={b}: α_b {} vs fd {fd_b}", d.a_b);
                // Second order from the closed-form first derivatives.
                let dq = |dq: f64, db: f64| solve_derivatives(q + dq, b + db, grid.view()).expect("d");
                let fd_qq = (dq(h, 0.0).a_q - dq(-h, 0.0).a_q) / (2.0 * h);
                let fd_qb = (dq(0.0, h).a_q - dq(0.0, -h).a_q) / (2.0 * h);
                let fd_bq = (dq(h, 0.0).a_b - dq(-h, 0.0).a_b) / (2.0 * h);
                let fd_bb = (dq(0.0, h).a_b - dq(0.0, -h).a_b) / (2.0 * h);
                assert!((d.a_qq - fd_qq).abs() < 1e-6 * (1.0 + fd_qq.abs()), "q={q} b={b}: α_qq {} vs fd {fd_qq}", d.a_qq);
                assert!((d.a_qb - fd_qb).abs() < 1e-6 * (1.0 + fd_qb.abs()), "q={q} b={b}: α_qb {} vs fd {fd_qb}", d.a_qb);
                assert!((d.a_qb - fd_bq).abs() < 1e-6 * (1.0 + fd_bq.abs()), "q={q} b={b}: α_qb {} vs fd(b,q) {fd_bq}", d.a_qb);
                assert!((d.a_bb - fd_bb).abs() < 1e-6 * (1.0 + fd_bb.abs()), "q={q} b={b}: α_bb {} vs fd {fd_bb}", d.a_bb);
                // Third order from the closed-form second derivatives.
                let fd_qqq = (dq(h, 0.0).a_qq - dq(-h, 0.0).a_qq) / (2.0 * h);
                let fd_qqb = (dq(0.0, h).a_qq - dq(0.0, -h).a_qq) / (2.0 * h);
                let fd_qbb = (dq(0.0, h).a_qb - dq(0.0, -h).a_qb) / (2.0 * h);
                let fd_bbb = (dq(0.0, h).a_bb - dq(0.0, -h).a_bb) / (2.0 * h);
                let fd_qbq = (dq(h, 0.0).a_qb - dq(-h, 0.0).a_qb) / (2.0 * h);
                assert!((d.a_qqq - fd_qqq).abs() < 1e-5 * (1.0 + fd_qqq.abs()), "q={q} b={b}: α_qqq {} vs fd {fd_qqq}", d.a_qqq);
                assert!((d.a_qqb - fd_qqb).abs() < 1e-5 * (1.0 + fd_qqb.abs()), "q={q} b={b}: α_qqb {} vs fd {fd_qqb}", d.a_qqb);
                assert!((d.a_qqb - fd_qbq).abs() < 1e-5 * (1.0 + fd_qbq.abs()), "q={q} b={b}: α_qqb {} vs fd(qb,q) {fd_qbq}", d.a_qqb);
                assert!((d.a_qbb - fd_qbb).abs() < 1e-5 * (1.0 + fd_qbb.abs()), "q={q} b={b}: α_qbb {} vs fd {fd_qbb}", d.a_qbb);
                assert!((d.a_bbb - fd_bbb).abs() < 1e-5 * (1.0 + fd_bbb.abs()), "q={q} b={b}: α_bbb {} vs fd {fd_bbb}", d.a_bbb);
            }
        }
    }

    /// The jet lift carries the same derivatives as the closed form through
    /// order two (`Order2<2>`) and through order four (`Tower4<2>`), and its
    /// value channel is bitwise the solver's root.
    #[test]
    fn anchor_jet_matches_closed_form_and_finite_differences() {
        let grid = skewed_grid();
        for &q in &[-1.1, 0.3, 1.6] {
            for &b in &[-0.6, 0.2, 1.2] {
                let d = solve_derivatives(q, b, grid.view()).expect("derivatives");
                let qj = Order2::<2>::variable(q, 0);
                let bj = Order2::<2>::variable(b, 1);
                let aj = anchor_jet(d.alpha, &qj, &bj, grid.view());
                assert_eq!(aj.value().to_bits(), d.alpha.to_bits(), "value channel is the root");
                let tower = aj.0;
                assert!((tower.g[0] - d.a_q).abs() < 1e-10 * (1.0 + d.a_q.abs()), "jet α_q {} vs {}", tower.g[0], d.a_q);
                assert!((tower.g[1] - d.a_b).abs() < 1e-10 * (1.0 + d.a_b.abs()), "jet α_b {} vs {}", tower.g[1], d.a_b);
                assert!((tower.h[0][0] - d.a_qq).abs() < 1e-9 * (1.0 + d.a_qq.abs()), "jet α_qq {} vs {}", tower.h[0][0], d.a_qq);
                assert!((tower.h[0][1] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "jet α_qb {} vs {}", tower.h[0][1], d.a_qb);
                assert!((tower.h[1][1] - d.a_bb).abs() < 1e-9 * (1.0 + d.a_bb.abs()), "jet α_bb {} vs {}", tower.h[1][1], d.a_bb);

                let qt = Tower4::<2>::variable(q, 0);
                let bt = Tower4::<2>::variable(b, 1);
                let at = anchor_jet(d.alpha, &qt, &bt, grid.view());
                assert!((at.t3[0][0][0] - d.a_qqq).abs() < 1e-8 * (1.0 + d.a_qqq.abs()), "tower α_qqq {} vs {}", at.t3[0][0][0], d.a_qqq);
                assert!((at.t3[0][0][1] - d.a_qqb).abs() < 1e-8 * (1.0 + d.a_qqb.abs()), "tower α_qqb {} vs {}", at.t3[0][0][1], d.a_qqb);
                assert!((at.t3[0][1][1] - d.a_qbb).abs() < 1e-8 * (1.0 + d.a_qbb.abs()), "tower α_qbb {} vs {}", at.t3[0][1][1], d.a_qbb);
                assert!((at.t3[1][1][1] - d.a_bbb).abs() < 1e-8 * (1.0 + d.a_bbb.abs()), "tower α_bbb {} vs {}", at.t3[1][1][1], d.a_bbb);
                // Fourth order against a central difference of the third.
                let h = 1e-4;
                let third = |dq: f64, db: f64| solve_derivatives(q + dq, b + db, grid.view()).expect("d");
                let fd_qqqq = (third(h, 0.0).a_qqq - third(-h, 0.0).a_qqq) / (2.0 * h);
                let fd_qqqb = (third(0.0, h).a_qqq - third(0.0, -h).a_qqq) / (2.0 * h);
                let fd_qqbb = (third(0.0, h).a_qqb - third(0.0, -h).a_qqb) / (2.0 * h);
                let fd_qbbb = (third(0.0, h).a_qbb - third(0.0, -h).a_qbb) / (2.0 * h);
                let fd_bbbb = (third(0.0, h).a_bbb - third(0.0, -h).a_bbb) / (2.0 * h);
                assert!((at.t4[0][0][0][0] - fd_qqqq).abs() < 1e-5 * (1.0 + fd_qqqq.abs()), "tower α_qqqq {} vs fd {fd_qqqq}", at.t4[0][0][0][0]);
                assert!((at.t4[0][0][0][1] - fd_qqqb).abs() < 1e-5 * (1.0 + fd_qqqb.abs()), "tower α_qqqb {} vs fd {fd_qqqb}", at.t4[0][0][0][1]);
                assert!((at.t4[0][0][1][1] - fd_qqbb).abs() < 1e-5 * (1.0 + fd_qqbb.abs()), "tower α_qqbb {} vs fd {fd_qqbb}", at.t4[0][0][1][1]);
                assert!((at.t4[0][1][1][1] - fd_qbbb).abs() < 1e-5 * (1.0 + fd_qbbb.abs()), "tower α_qbbb {} vs fd {fd_qbbb}", at.t4[0][1][1][1]);
                assert!((at.t4[1][1][1][1] - fd_bbbb).abs() < 1e-5 * (1.0 + fd_bbbb.abs()), "tower α_bbbb {} vs fd {fd_bbbb}", at.t4[1][1][1][1]);

                // The q- and b-derivative jets agree with the closed form through order two.
                let aq = anchor_q_derivative_jet(&aj, &qj, &bj, grid.view()).0;
                assert!((aq.v - d.a_q).abs() < 1e-10 * (1.0 + d.a_q.abs()));
                assert!((aq.g[0] - d.a_qq).abs() < 1e-9 * (1.0 + d.a_qq.abs()), "α_q jet g[q] {} vs {}", aq.g[0], d.a_qq);
                assert!((aq.g[1] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "α_q jet g[b] {} vs {}", aq.g[1], d.a_qb);
                assert!((aq.h[0][0] - d.a_qqq).abs() < 1e-8 * (1.0 + d.a_qqq.abs()), "α_q jet h[qq] {} vs {}", aq.h[0][0], d.a_qqq);
                assert!((aq.h[0][1] - d.a_qqb).abs() < 1e-8 * (1.0 + d.a_qqb.abs()), "α_q jet h[qb] {} vs {}", aq.h[0][1], d.a_qqb);
                assert!((aq.h[1][1] - d.a_qbb).abs() < 1e-8 * (1.0 + d.a_qbb.abs()), "α_q jet h[bb] {} vs {}", aq.h[1][1], d.a_qbb);
                let ab = anchor_b_derivative_jet(&aj, &qj, &bj, grid.view()).0;
                assert!((ab.v - d.a_b).abs() < 1e-10 * (1.0 + d.a_b.abs()));
                assert!((ab.g[0] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "α_b jet g[q] {} vs {}", ab.g[0], d.a_qb);
                assert!((ab.g[1] - d.a_bb).abs() < 1e-9 * (1.0 + d.a_bb.abs()), "α_b jet g[b] {} vs {}", ab.g[1], d.a_bb);
                assert!((ab.h[1][1] - d.a_bbb).abs() < 1e-8 * (1.0 + d.a_bbb.abs()), "α_b jet h[bb] {} vs {}", ab.h[1][1], d.a_bbb);
            }
        }
    }

    /// The Taylor table is the closed-form implicit derivatives through order
    /// three, and its lift onto the full order-four tower is Newton's iteration
    /// in the jet algebra, for `α` and for `α_q` — which reads the table's
    /// fifth order (gam#2928).
    #[test]
    fn taylor_table_matches_the_closed_form_and_the_jet_oracle() {
        let grid = skewed_grid();
        for &q in &[-2.1, -0.6, 0.3, 1.6, 3.4] {
            for &b in &[-1.2, -0.2, 0.0, 0.7, 1.9] {
                let d = solve_derivatives(q, b, grid.view()).expect("derivatives");
                let taylor = AnchorTaylor::at(d.alpha, q, b, grid.view()).expect("table");
                let c = &taylor.coefficients;
                let close = |left: f64, right: f64, tol: f64, what: &str| {
                    assert!(
                        (left - right).abs() <= tol * (1.0 + right.abs()),
                        "q={q} b={b} {what}: {left:+.15e} vs {right:+.15e}"
                    );
                };
                close(c[1][0], d.a_q, 1e-12, "table α_q");
                close(c[0][1], d.a_b, 1e-12, "table α_b");
                close(2.0 * c[2][0], d.a_qq, 1e-11, "table α_qq");
                close(c[1][1], d.a_qb, 1e-11, "table α_qb");
                close(2.0 * c[0][2], d.a_bb, 1e-11, "table α_bb");
                close(6.0 * c[3][0], d.a_qqq, 1e-10, "table α_qqq");
                close(2.0 * c[2][1], d.a_qqb, 1e-10, "table α_qqb");
                close(2.0 * c[1][2], d.a_qbb, 1e-10, "table α_qbb");
                close(6.0 * c[0][3], d.a_bbb, 1e-10, "table α_bbb");

                let qt = Tower4::<2>::variable(q, 0);
                let bt = Tower4::<2>::variable(b, 1);
                let delta_b = bt.compose_unary([0.0, 1.0, 0.0, 0.0, 0.0]);
                let lifted = taylor.lift(&qt, &delta_b);
                let oracle = anchor_jet(d.alpha, &qt, &bt, grid.view());
                assert_eq!(lifted.value().to_bits(), d.alpha.to_bits(), "the lift's value is the root");
                let lifted_q = taylor.lift_q_derivative(&qt, &delta_b);
                let oracle_q = anchor_q_derivative_jet(&oracle, &qt, &bt, grid.view());
                for (what, left, right) in [("α", &lifted, &oracle), ("α_q", &lifted_q, &oracle_q)] {
                    close(left.value(), right.value(), 1e-12, &format!("{what} value"));
                    for i in 0..2 {
                        close(left.g[i], right.g[i], 1e-11, &format!("{what} g[{i}]"));
                        for j in 0..2 {
                            close(left.h[i][j], right.h[i][j], 1e-10, &format!("{what} h[{i}][{j}]"));
                            for k in 0..2 {
                                close(left.t3[i][j][k], right.t3[i][j][k], 1e-9, &format!("{what} t3[{i}][{j}][{k}]"));
                                for l in 0..2 {
                                    close(
                                        left.t4[i][j][k][l],
                                        right.t4[i][j][k][l],
                                        1e-8,
                                        &format!("{what} t4[{i}][{j}][{k}][{l}]"),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// A row's slot answers only for bitwise the inputs its root was solved
    /// for: a sentinel stored under `(q, b)` comes back for exactly `(q, b)`,
    /// one bit off in either input is a different equation and is solved, and
    /// a root stored under other inputs, near or far, seeds nothing: every
    /// solve through the slot is bitwise the closed-form-seeded root, and is
    /// published under its own inputs (gam#2983).
    #[test]
    fn anchor_slot_answers_only_its_own_equation_and_seeds_nothing_2983() {
        let grid = skewed_grid();
        let cache = AnchorRootCache::new(4, 2, false);
        let context = AnchorRowContext {
            grid: grid.view(),
            roots: Some(&cache),
        };
        let slot = 1;
        let (q, b) = (0.7_f64, 1.1_f64);

        let cold = solve_anchor_in_slot(q, b, context, 2, slot).expect("cold solve");
        assert_eq!(cold.to_bits(), solve_anchor(q, b, grid.view()).expect("closed form").to_bits());
        // Stored under exactly these inputs the slot IS the answer: a sentinel
        // comes back, which no solve could produce.
        let own = cache.slot(2, slot).expect("row 2 slot");
        let sentinel = |q: f64, b: f64, root: f64| StoredAnchor {
            q: q.to_bits(),
            slope: b.to_bits(),
            root,
            taylor: None,
        };
        own.write(&sentinel(q, b, 123.0));
        assert_eq!(solve_anchor_in_slot(q, b, context, 2, slot).expect("reuse"), 123.0);
        for (nudged_q, nudged_b) in [
            (f64::from_bits(q.to_bits() + 1), b),
            (q, f64::from_bits(b.to_bits() + 1)),
        ] {
            own.write(&sentinel(q, b, 123.0));
            let nudged = solve_anchor_in_slot(nudged_q, nudged_b, context, 2, slot).expect("nudged");
            let fresh = solve_anchor(nudged_q, nudged_b, grid.view()).expect("fresh nudged");
            assert_eq!(
                nudged.to_bits(),
                fresh.to_bits(),
                "a one-bit-different equation must be solved: {nudged} vs {fresh}"
            );
        }

        // An iterate walk, each equation a small step from the last, through a
        // slot holding the previous equation's root and table: every root is
        // bitwise the closed-form-seeded root of its own equation. A root the
        // slot seeded would be one of the many floats its residual accepts, set
        // by where the seed sat; warm-seeded, 57% of truth2370's anchor roots
        // were off the closed-form root (gam#2983).
        let mut previous = (q, b, cold);
        for step in 1..=64 {
            let (next_q, next_b) = (q + 3e-4 * f64::from(step), b - 4e-4 * f64::from(step));
            let table = AnchorTaylor::at(previous.2, previous.0, previous.1, grid.view()).expect("table");
            cache.slot(1, slot).expect("row 1 slot").write(&StoredAnchor {
                q: previous.0.to_bits(),
                slope: previous.1.to_bits(),
                root: previous.2,
                taylor: Some(table),
            });
            let through_slot = solve_anchor_in_slot(next_q, next_b, context, 1, slot).expect("slot solve");
            let fresh = solve_anchor(next_q, next_b, grid.view()).expect("fresh solve");
            assert_eq!(
                through_slot.to_bits(),
                fresh.to_bits(),
                "step {step}: the slot solve {through_slot:+.17e} is not the closed-form root {fresh:+.17e}"
            );
            assert_eq!(
                cache
                    .slot(1, slot)
                    .and_then(RootSlot::read)
                    .map(|stored| (stored.q, stored.slope, stored.root.to_bits())),
                Some((next_q.to_bits(), next_b.to_bits(), fresh.to_bits())),
                "step {step}: the solve is published under its own inputs"
            );
            previous = (next_q, next_b, through_slot);
        }

        cache.slot(3, slot).expect("row 3 slot").write(&sentinel(5.0, 0.2, 1e6));
        let poisoned = solve_anchor_in_slot(-0.4, 0.9, context, 3, slot).expect("poisoned slot");
        let reference = solve_anchor(-0.4, 0.9, grid.view()).expect("reference");
        assert_eq!(poisoned.to_bits(), reference.to_bits(), "poisoned {poisoned} vs reference {reference}");
    }

    /// A point-mass law at `u₀ = 0` anchors at `α = q` for every slope, so
    /// `α_q = 1` and every other derivative vanishes, and at `α = q` the
    /// normalized tables form each of those from identical operands: the
    /// implicit-derivative table must be exactly `(1, 0, …, 0)` and the Taylor
    /// table `c[1][0] = 1` with the rest zero up to the rounding of a degree-five
    /// polynomial in `q`. Deep in both tails, where `φ(q)` and `φ(η)` have long
    /// underflowed and the linear-space moments refused (gam#2941). Off the
    /// origin the solve must land on the closed form `q − b·u₀` within its
    /// certified error.
    #[test]
    fn point_mass_law_anchor_tables_are_the_closed_form_in_both_tails() {
        let origin = owned(vec![0.0], vec![1.0]);
        let shifted = owned(vec![0.37], vec![1.0]);
        let qs = [
            -90.0, -84.0, -80.0, -70.55899843438496, -12.0, 0.0, 12.0, 70.55899843438496, 80.0, 84.0, 90.0,
        ];
        let bs = [-25.0, -21.12209971696722, -1.0, 0.0, 1.0, 21.12209971696722, 25.0];
        for &q in &qs {
            for &b in &bs {
                let d = AnchorDerivatives::at(q, q, b, origin.view())
                    .unwrap_or_else(|e| panic!("q={q} b={b}: derivatives refused: {e}"));
                assert_eq!(
                    [d.a_q, d.a_b, d.a_qq, d.a_qb, d.a_bb, d.a_qqq, d.a_qqb, d.a_qbb, d.a_bbb],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "q={q} b={b}: point-mass derivatives"
                );
                let taylor = AnchorTaylor::at(q, q, b, origin.view())
                    .unwrap_or_else(|e| panic!("q={q} b={b}: Taylor table refused: {e}"));
                for i in 0..TAYLOR_SLOTS {
                    for j in 0..TAYLOR_SLOTS - i {
                        let expected = match (i, j) {
                            (0, 0) => q,
                            (1, 0) => 1.0,
                            _ => 0.0,
                        };
                        let rounding = 64.0 * f64::EPSILON * (1.0 + q.abs()).powi((i + j) as i32 + 1);
                        assert!(
                            (taylor.coefficients[i][j] - expected).abs() <= rounding,
                            "q={q} b={b}: Taylor c[{i}][{j}] = {:+.17e}, expected {expected} (rounding {rounding:.3e})",
                            taylor.coefficients[i][j]
                        );
                    }
                }

                let u0 = 0.37;
                let alpha = solve_anchor(q, b, shifted.view())
                    .unwrap_or_else(|e| panic!("q={q} b={b}: point-mass solve refused: {e}"));
                let survival_side = q >= 0.0;
                let log_target = if survival_side {
                    normal_logcdf(-q)
                } else {
                    normal_logcdf(q)
                };
                let (_, first, _) = anchor_log_residual(alpha, b, shifted.view(), survival_side, log_target)
                    .expect("point-mass residual");
                let certified = anchor_residual_resolution(
                    alpha,
                    first,
                    anchor_residual_rounding(log_target, shifted.view().len()),
                ) / first.abs()
                    + 2.0 * f64::EPSILON * (q.abs() + (b * u0).abs());
                assert!(
                    (alpha - (q - b * u0)).abs() <= certified,
                    "q={q} b={b}: point-mass root {alpha:+.17e} vs closed form {:+.17e} (certified {certified:.3e})",
                    q - b * u0
                );
                let shifted_derivatives = AnchorDerivatives::at(alpha, q, b, shifted.view())
                    .unwrap_or_else(|e| panic!("q={q} b={b}: shifted derivatives refused: {e}"));
                assert!(
                    (shifted_derivatives.a_b + u0).abs() <= 4.0 * f64::EPSILON,
                    "q={q} b={b}: α_b {} vs −u₀",
                    shifted_derivatives.a_b
                );
            }
        }
    }

    /// Deep in both tails of a skewed and a Gauss–Hermite law — `|q|` to 80,
    /// `|b|` to 25, and the 2768 fixture's refused point — the anchor solves and
    /// its derivatives are finite and are central differences of the solved
    /// anchor (gam#2941). The bound is the difference's own error: the Taylor
    /// remainder `h²/6·|α‴|` (α‴ from the table at the stencil's three points)
    /// plus each stencil root's certified error `resolution/|F′|` over `h`, plus
    /// the table's rounding.
    #[test]
    fn anchor_derivatives_match_differences_deep_in_both_tails() {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("grid");
        let gaussian = owned(nodes, weights);
        let skewed = skewed_grid();
        // The 32 scores ((7i) mod 32)/8 − 2 of gnomon's delayed-entry calibration
        // fixture at equal weight (a permutation of −2 + k/8, taken ascending):
        // every startup seed there was refused at the first point below.
        let calibrate = owned((0..32).map(|k| k as f64 / 8.0 - 2.0).collect(), vec![1.0 / 32.0; 32]);
        let points = [
            (-83.97022428333848, -0.09658543743257131),
            (-70.55899843438496, -21.12209971696722),
            (-84.0, -3.0),
            (-84.0, 12.0),
            (-90.0, -25.0),
            (-90.0, 25.0),
            (90.0, -25.0),
            (90.0, 4.0),
            (-80.0, -25.0),
            (-80.0, 25.0),
            (80.0, -25.0),
            (80.0, 25.0),
            (-40.0, 7.0),
            (55.0, -12.0),
        ];
        // The calibration point really is one where the linear-space moments
        // underflowed: at its anchor `Σ_k w_k φ(η_k)` rounds to zero.
        let (refused_q, refused_b) = (-83.97022428333848, -0.09658543743257131);
        let refused_alpha = solve_anchor(refused_q, refused_b, calibrate.view()).expect("calibration anchor");
        let linear_density: f64 = calibrate
            .nodes
            .iter()
            .zip(calibrate.weights.iter())
            .map(|(&u, &w)| w * normal_pdf(refused_alpha + refused_b * u))
            .sum();
        assert_eq!(
            linear_density, 0.0,
            "the calibration point must be one where Σ w φ(η) underflows (α={refused_alpha})"
        );
        for (label, grid) in [("skewed", &skewed), ("gaussian", &gaussian), ("calibrate", &calibrate)] {
            let m = grid.view().len() as f64;
            let widest_node = grid.nodes.iter().fold(0.0_f64, |widest, u| widest.max(u.abs()));
            let solve = |q: f64, b: f64| -> (f64, f64, AnchorDerivatives) {
                let alpha = solve_anchor(q, b, grid.view())
                    .unwrap_or_else(|e| panic!("{label} q={q} b={b}: solve refused: {e}"));
                let survival_side = q >= 0.0;
                let log_target = if survival_side {
                    normal_logcdf(-q)
                } else {
                    normal_logcdf(q)
                };
                let (_, first, _) = anchor_log_residual(alpha, b, grid.view(), survival_side, log_target)
                    .expect("residual");
                let certified = anchor_residual_resolution(
                    alpha,
                    first,
                    anchor_residual_rounding(log_target, grid.view().len()),
                ) / first.abs();
                let derivatives = AnchorDerivatives::at(alpha, q, b, grid.view())
                    .unwrap_or_else(|e| panic!("{label} q={q} b={b}: derivatives refused: {e}"));
                (alpha, certified, derivatives)
            };
            for &(q, b) in &points {
                let (alpha, _, d) = solve(q, b);
                // The table's rounding: its log-density ratio forms products of
                // indices as large as `|q| + |α| + |b|·max|u|`.
                let scale = 1.0 + q.abs() + alpha.abs() + b.abs() * widest_node;
                let table_rounding = (m + 4.0) * f64::EPSILON * scale * scale;
                // Each stencil is divided by its representable width, so the
                // rounding of `q ± h` is not an error of the difference.
                let (q_up, q_down) = (q + 1e-3 * (1.0 + q.abs()), q - 1e-3 * (1.0 + q.abs()));
                let (up, up_error, d_up) = solve(q_up, b);
                let (down, down_error, d_down) = solve(q_down, b);
                let width_q = q_up - q_down;
                let fd_q = (up - down) / width_q;
                let third_q = d.a_qqq.abs().max(d_up.a_qqq.abs()).max(d_down.a_qqq.abs());
                let bound_q = width_q * width_q / 24.0 * third_q
                    + (up_error + down_error) / width_q
                    + table_rounding * (1.0 + d.a_q.abs());
                assert!(
                    (d.a_q - fd_q).abs() <= bound_q,
                    "{label} q={q} b={b}: α_q {:+.17e} vs difference {fd_q:+.17e} (bound {bound_q:.3e})",
                    d.a_q
                );
                let (b_up, b_down) = (b + 1e-3 * (1.0 + b.abs()), b - 1e-3 * (1.0 + b.abs()));
                let (right, right_error, d_right) = solve(q, b_up);
                let (left, left_error, d_left) = solve(q, b_down);
                let width_b = b_up - b_down;
                let fd_b = (right - left) / width_b;
                let third_b = d.a_bbb.abs().max(d_right.a_bbb.abs()).max(d_left.a_bbb.abs());
                let bound_b = width_b * width_b / 24.0 * third_b
                    + (right_error + left_error) / width_b
                    + table_rounding * (1.0 + d.a_b.abs());
                assert!(
                    (d.a_b - fd_b).abs() <= bound_b,
                    "{label} q={q} b={b}: α_b {:+.17e} vs difference {fd_b:+.17e} (bound {bound_b:.3e})",
                    d.a_b
                );
                let taylor = AnchorTaylor::at(d.alpha, q, b, grid.view())
                    .unwrap_or_else(|e| panic!("{label} q={q} b={b}: Taylor table refused: {e}"));
                assert!(
                    taylor.coefficients.iter().flatten().all(|c| c.is_finite()),
                    "{label} q={q} b={b}: Taylor table not finite"
                );
            }
        }
    }

    /// A slot keeps derivatives for exactly the equation its root solved
    /// (gam#2928): a root-only consumer stores none, the first differentiating
    /// consumer adds `AnchorTaylor::at` of that root and the order-two lowering
    /// reads its derivatives from it, a sentinel table stored under `(q, b)`
    /// answers for `(q, b)` alone, and a one-bit change in either input is
    /// solved and differentiated afresh. A table stored under other inputs,
    /// sound or poisoned, seeds nothing: the slot's root is bitwise the
    /// closed-form-seeded root (gam#2983).
    #[test]
    fn anchor_slot_keeps_the_table_of_exactly_its_own_equation() {
        let grid = skewed_grid();
        let cache = AnchorRootCache::new(3, 2, false);
        let context = AnchorRowContext {
            grid: grid.view(),
            roots: Some(&cache),
        };
        let slot = 1;
        let (q, b) = (0.9_f64, 1.3_f64);

        let root = solve_anchor_in_slot(q, b, context, 0, slot).expect("root");
        let stored = cache.slot(0, slot).and_then(RootSlot::read).expect("stored root");
        assert!(stored.taylor.is_none(), "a root-only consumer stores no table");
        let taylor = anchor_taylor_in_slot(q, b, context, 0, slot).expect("table");
        assert_eq!(taylor, AnchorTaylor::at(root, q, b, grid.view()).expect("reference"));
        assert_eq!(
            anchor_derivatives_in_slot(q, b, context, 0, slot).expect("derivatives"),
            taylor.derivatives(),
            "the order-two lowering reads the published table"
        );
        assert_eq!(
            cache.slot(0, slot).and_then(RootSlot::read).and_then(|stored| stored.taylor),
            Some(taylor),
            "the differentiated root is published with its table"
        );

        let mut sentinel = taylor;
        sentinel.coefficients[2][0] = 42.0;
        let own = cache.slot(0, slot).expect("row 0 slot");
        let store = |taylor: AnchorTaylor| StoredAnchor {
            q: q.to_bits(),
            slope: b.to_bits(),
            root,
            taylor: Some(taylor),
        };
        own.write(&store(sentinel));
        assert_eq!(anchor_taylor_in_slot(q, b, context, 0, slot).expect("hit"), sentinel);
        for (nudged_q, nudged_b) in [
            (f64::from_bits(q.to_bits() + 1), b),
            (q, f64::from_bits(b.to_bits() + 1)),
        ] {
            own.write(&store(sentinel));
            let fresh = anchor_taylor_in_slot(nudged_q, nudged_b, context, 0, slot).expect("nudged");
            assert_ne!(fresh, sentinel, "a one-bit-different equation is not a hit");
            assert_eq!(
                fresh,
                AnchorTaylor::at(fresh.derivatives().alpha, nudged_q, nudged_b, grid.view())
                    .expect("nudged reference")
            );
        }

        // A table stored under other inputs, sound or poisoned, seeds nothing
        // (gam#2983): the slot's root is bitwise the closed-form-seeded root.
        // The nearby equations are where a sound table's series would land a
        // seed inside the residual resolution, so a solve started there could
        // stop on other bits. The far one is where the series outruns itself.
        let mut poisoned = taylor;
        poisoned.coefficients[1][0] = 1e6;
        poisoned.coefficients[0][1] = -1e6;
        let others = (1..=16)
            .map(|step| (q + 3e-4 * f64::from(step), b - 4e-4 * f64::from(step)))
            .chain(std::iter::once((1.05_f64, 1.21_f64)));
        for (other_q, other_b) in others {
            let cold = solve_anchor(other_q, other_b, grid.view()).expect("cold");
            for (row, table) in [(1, taylor), (2, poisoned)] {
                cache.slot(row, slot).expect("slot").write(&store(table));
                let through_slot =
                    solve_anchor_in_slot(other_q, other_b, context, row, slot).expect("slot solve");
                assert_eq!(
                    through_slot.to_bits(),
                    cold.to_bits(),
                    "({other_q}, {other_b}) row {row}: the slot solve {through_slot:+.17e} is not the \
                     closed-form root {cold:+.17e}"
                );
            }
        }
    }

    /// Rows asking a global law the same equation share one root through the
    /// cross-row table. A hash only selects an entry: with every equation
    /// forced onto one entry by a colliding test hasher, a different equation
    /// is never a hit. A per-row law keeps no table at all.
    #[test]
    fn anchor_cross_row_table_decides_hits_by_exact_inputs() {
        let grid = skewed_grid();
        let cache = AnchorRootCache::with_shared_index(4, 2, true, |_, _| 7);
        let context = AnchorRowContext {
            grid: grid.view(),
            roots: Some(&cache),
        };
        let slot = 0;
        let (q, b) = (-1.3_f64, 0.8_f64);

        let first = solve_anchor_in_slot(q, b, context, 0, slot).expect("row 0");
        // A sentinel stored for (q, b) reaches row 1 asking the same equation
        // without a solve, and is published into row 1's own slot.
        cache
            .shared_slot(slot, q.to_bits(), b.to_bits())
            .expect("global law table")
            .write(&StoredAnchor {
                q: q.to_bits(),
                slope: b.to_bits(),
                root: 77.0,
                taylor: None,
            });
        assert_eq!(solve_anchor_in_slot(q, b, context, 1, slot).expect("row 1"), 77.0);
        assert_eq!(
            cache
                .slot(1, slot)
                .and_then(RootSlot::read)
                .map(|stored| (stored.q, stored.slope, stored.root)),
            Some((q.to_bits(), b.to_bits(), 77.0))
        );

        // A different equation hashed onto the same entry is solved on its own.
        let (other_q, other_b) = (0.4_f64, -0.5_f64);
        let colliding = solve_anchor_in_slot(other_q, other_b, context, 2, slot).expect("row 2");
        let reference = solve_anchor(other_q, other_b, grid.view()).expect("reference");
        assert_eq!(colliding.to_bits(), reference.to_bits(), "a colliding entry is never a hit");
        assert!((first - reference).abs() > 1e-3, "the two equations have different roots");

        // Each slot kind keeps its own entries (gam#2928): a sentinel this kind
        // holds for (q, b) is no hit for the other kind asking the same
        // equation onto the same hash, and the other kind's publish does not
        // evict it.
        let other_kind = 1;
        cache
            .shared_slot(slot, q.to_bits(), b.to_bits())
            .expect("this kind's table")
            .write(&StoredAnchor {
                q: q.to_bits(),
                slope: b.to_bits(),
                root: 55.0,
                taylor: None,
            });
        let other = solve_anchor_in_slot(q, b, context, 3, other_kind).expect("other kind");
        assert_eq!(other.to_bits(), first.to_bits(), "the other kind solves afresh");
        assert_eq!(
            cache
                .shared_slot(slot, q.to_bits(), b.to_bits())
                .and_then(RootSlot::read)
                .map(|stored| stored.root),
            Some(55.0),
            "the other kind's publish left this kind's entry alone"
        );

        assert!(
            AnchorRootCache::new(4, 2, false).shared_slot(slot, q.to_bits(), b.to_bits()).is_none(),
            "a per-row law keeps no cross-row table"
        );
    }

    /// The linear-space residual is the log-space residual to rounding wherever
    /// its tail clears the floor, on both sides and across slopes, and a tail
    /// below the floor is handed to the log-space form unchanged (gam#2928).
    #[test]
    fn linear_space_residual_matches_the_log_space_residual() {
        let grid = skewed_grid();
        let mut compared = 0usize;
        for &q in &[-6.0, -2.2, -0.4, 0.0, 0.9, 3.1, 7.5] {
            let survival_side = q >= 0.0;
            let log_target = if survival_side {
                normal_logcdf(-q)
            } else {
                normal_logcdf(q)
            };
            for &b in &[-2.5, -0.7, 0.0, 0.35, 1.6, 4.0] {
                for &offset in &[-3.0, -0.5, 0.0, 0.4, 2.5] {
                    let alpha = gaussian_anchor(q, b) + offset;
                    let linear = anchor_log_residual(alpha, b, grid.view(), survival_side, log_target)
                        .expect("linear-space residual");
                    let logged =
                        anchor_log_residual_in_log_space(alpha, b, grid.view(), survival_side, log_target)
                            .expect("log-space residual");
                    // Rounding, not a picked tolerance: the log-space form rounds each
                    // node's log term — about |log T| in size where it carries weight —
                    // before exponentiating it, and either form sums m node terms, so
                    // the density moments agree to this relative rounding and F, a
                    // difference of two logarithms, to this absolute rounding.
                    let log_tail = logged.0 + log_target;
                    let rounding = (grid.view().len() as f64 + 4.0)
                        * f64::EPSILON
                        * (1.0 + log_tail.abs() + log_target.abs());
                    assert!(
                        (linear.0 - logged.0).abs() <= rounding,
                        "q={q} b={b} α={alpha} F: linear {:+.17e} vs log {:+.17e} (rounding {rounding:.3e})",
                        linear.0,
                        logged.0
                    );
                    assert!(
                        (linear.1 - logged.1).abs() <= rounding * logged.1.abs(),
                        "q={q} b={b} α={alpha} F′: linear {:+.17e} vs log {:+.17e} (rounding {rounding:.3e})",
                        linear.1,
                        logged.1
                    );
                    // F″ = ±Σ m η − (Σ m)² subtracts nearly equal numbers in the tails
                    // (both near 170 against F″ ≈ −1 at q = −6, b = −2.5), which
                    // amplifies either form's rounding a hundredfold. What each form
                    // computes is the moment Σ m η = ±(F″ + F′²): compare it before the
                    // subtraction, allowing the recovery's own rounding.
                    let moment = |r: (f64, f64, f64)| r.2 + r.1 * r.1;
                    let recovery = 2.0
                        * f64::EPSILON
                        * (linear.2.abs() + linear.1 * linear.1 + logged.2.abs() + logged.1 * logged.1);
                    assert!(
                        (moment(linear) - moment(logged)).abs() <= rounding * moment(logged).abs() + recovery,
                        "q={q} b={b} α={alpha} Σ m η: linear {:+.17e} vs log {:+.17e} (rounding {rounding:.3e})",
                        moment(linear),
                        moment(logged)
                    );
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 7 * 6 * 5);
        // A tail far below the floor (every node past Φ(−44) ≈ 1e-423, which
        // underflows): the two forms are the same evaluation.
        let (q, b, alpha) = (38.0, 0.5, 45.0);
        let log_target = normal_logcdf(-q);
        let deep = anchor_log_residual(alpha, b, grid.view(), true, log_target).expect("deep tail");
        let reference =
            anchor_log_residual_in_log_space(alpha, b, grid.view(), true, log_target).expect("deep reference");
        assert_eq!(
            (deep.0.to_bits(), deep.1.to_bits(), deep.2.to_bits()),
            (reference.0.to_bits(), reference.1.to_bits(), reference.2.to_bits()),
            "below the floor the residual is the log-space form"
        );
    }

    /// Roots solved on the linear-space residual meet the solve's acceptance
    /// criterion measured on the log-space form, and so do the log-space form's
    /// own roots, across `q ∈ [−8, 8]` and slopes to ±4 on both laws, with both
    /// marginal tails `Φ(−q) < 1e-6` and `Φ(−q) > 1 − 1e-6` in the set and shown
    /// to run the linear form (gam#2928). The two roots then differ by at most
    /// what those residuals allow: `F` is monotone, so by the mean value theorem
    /// `|α_lin − α_log| ≤ (|F(α_lin)| + |F(α_log)|) / min |F′|` over the
    /// interval between them. Nothing here is a hand-picked bound: the one
    /// allowance is the rounding of the reference evaluation itself, a sum of
    /// `m` positive terms and a logarithm, `(m + 4)·ε·(1 + |log target|)`.
    #[test]
    fn linear_space_roots_match_the_log_space_roots_across_both_tails() {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("grid");
        let gaussian = owned(nodes, weights);
        let skewed = skewed_grid();
        let qs = [-8.0, -6.0, -4.8, -3.0, -1.0, -0.2, 0.0, 0.3, 1.5, 3.3, 4.8, 6.5, 8.0];
        assert!(qs.iter().any(|&q| normal_cdf(-q) < 1e-6), "a small survival tail is in the set");
        assert!(qs.iter().any(|&q| normal_cdf(-q) > 1.0 - 1e-6), "a small event tail is in the set");
        let mut worst_gap = 0.0_f64;
        let mut worst_bound_use = 0.0_f64;
        let mut linear_tail_cases = 0usize;
        for (label, grid) in [("gaussian", &gaussian), ("skewed", &skewed)] {
            for &q in &qs {
                let survival_side = q >= 0.0;
                let log_target = if survival_side {
                    normal_logcdf(-q)
                } else {
                    normal_logcdf(q)
                };
                let noise = (grid.view().len() as f64 + 4.0) * f64::EPSILON * (1.0 + log_target.abs());
                for &b in &[-4.0, -1.6, -0.5, 0.0, 0.5, 1.6, 4.0] {
                    let seed = gaussian_anchor(q, b);
                    let (linear, _) = solve_anchor_with(q, b, grid.view(), seed, anchor_log_residual)
                        .unwrap_or_else(|e| panic!("{label} q={q} b={b}: linear solve: {e}"));
                    let (logged, _) =
                        solve_anchor_with(q, b, grid.view(), seed, anchor_log_residual_in_log_space)
                            .unwrap_or_else(|e| panic!("{label} q={q} b={b}: log solve: {e}"));
                    let reference = |alpha: f64| {
                        anchor_log_residual_in_log_space(alpha, b, grid.view(), survival_side, log_target)
                            .expect("reference residual")
                    };
                    let (f_linear, d_linear, _) = reference(linear);
                    let (f_logged, d_logged, _) = reference(logged);
                    assert!(
                        f_linear.abs()
                            <= anchor_residual_resolution(
                                linear,
                                d_linear,
                                anchor_residual_rounding(log_target, grid.view().len())
                            ) + noise,
                        "{label} q={q} b={b}: linear root {linear:+.17e} has reference residual {f_linear:.3e}"
                    );
                    assert!(
                        f_logged.abs()
                            <= anchor_residual_resolution(
                                logged,
                                d_logged,
                                anchor_residual_rounding(log_target, grid.view().len())
                            ),
                        "{label} q={q} b={b}: log root {logged:+.17e} has reference residual {f_logged:.3e}"
                    );
                    let (_, d_mid, _) = reference(linear + 0.5 * (logged - linear));
                    let min_slope = d_linear.abs().min(d_logged.abs()).min(d_mid.abs());
                    let bound = (f_linear.abs() + f_logged.abs() + 2.0 * noise) / min_slope;
                    let gap = (linear - logged).abs();
                    assert!(
                        gap <= bound,
                        "{label} q={q} b={b}: linear root {linear:+.17e} vs log root {logged:+.17e}: gap {gap:.3e} exceeds the residual bound {bound:.3e}"
                    );
                    worst_gap = worst_gap.max(gap / logged.abs().max(1.0));
                    if bound > 0.0 {
                        worst_bound_use = worst_bound_use.max(gap / bound);
                    }
                    let target_tail = normal_cdf(-q).min(normal_cdf(q));
                    if target_tail < 1e-6 {
                        let tail: f64 = grid
                            .nodes
                            .iter()
                            .zip(grid.weights.iter())
                            .map(|(&u, &w)| {
                                let eta = linear + b * u;
                                w * normal_cdf(if q >= 0.0 { -eta } else { eta })
                            })
                            .sum();
                        assert!(
                            tail >= LINEAR_RESIDUAL_FLOOR,
                            "{label} q={q} b={b}: the tail case must run the linear form (T={tail:e})"
                        );
                        linear_tail_cases += 1;
                    }
                }
            }
        }
        eprintln!(
            "[2928 residual] max |α_linear − α_log|/max(1, |α|) = {worst_gap:.3e}, max gap/bound = {worst_bound_use:.3} over {} solves; {linear_tail_cases} tail solves on the linear form",
            2 * qs.len() * 7
        );
        assert!(linear_tail_cases >= 2 * 4 * 7, "every tail case ran the linear form");
    }

    /// The solve must not refuse anywhere a line search can wander: wide
    /// slopes, indices deep in either tail, on both laws.
    #[test]
    fn anchor_solves_across_the_whole_trial_range() {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("grid");
        let gaussian = owned(nodes, weights);
        let skewed = skewed_grid();
        let mut failures = Vec::new();
        for (label, grid) in [("gaussian", &gaussian), ("skewed", &skewed)] {
            for &q in &[-12.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 12.0] {
                for &b in &[-8.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 8.0, 20.0] {
                    if let Err(reason) = solve_anchor(q, b, grid.view()) {
                        failures.push(format!("{label} q={q} b={b}: {reason}"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "anchor solves refused:\n{}", failures.join("\n"));
    }

    /// Deep tails: the log-space residual keeps the solve finite and exact
    /// where the linear-space marginal has long underflowed.
    #[test]
    fn anchor_survives_the_deep_tails() {
        let grid = skewed_grid();
        for &q in &[-7.5, 7.5] {
            for &b in &[-1.5, 0.0, 1.5] {
                let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                assert!(alpha.is_finite());
                // On the survival side the anchored survival must equal Φ(−q)
                // relatively; on the complement side the anchored failure
                // probability must equal Φ(q) relatively.
                let survival = marginal(alpha, b, grid.view());
                if q >= 0.0 {
                    let target = normal_cdf(-q);
                    assert!(((survival - target) / target).abs() < 1e-9, "q={q} b={b}: {survival:e} vs {target:e}");
                } else {
                    let failure: f64 = grid
                        .nodes
                        .iter()
                        .zip(grid.weights.iter())
                        .map(|(&u, &w)| w * normal_cdf(alpha + b * u))
                        .sum();
                    let target = normal_cdf(q);
                    assert!(((failure - target) / target).abs() < 1e-9, "q={q} b={b}: {failure:e} vs {target:e}");
                }
            }
        }
    }

    /// gam#2945: the order-six table the exact fifth likelihood derivative solves on demand
    /// extends the order-five table a root slot publishes. Its orders through five are that
    /// table's bits, so the fifth derivative reads the same anchor the value path and every
    /// carrier read, and its sixth order is not the zero a truncated table would leave.
    #[test]
    fn sixth_order_table_extends_the_published_table_bitwise_2945() {
        let grid = skewed_grid();
        for &q in &[-2.1, -0.6, 0.3, 1.6, 3.4] {
            for &b in &[-1.2, -0.2, 0.0, 0.7, 1.9] {
                let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                let published = AnchorTaylor::at(alpha, q, b, grid.view()).expect("order-five table");
                let sixth =
                    anchor_taylor_through_sixth(alpha, q, b, grid.view()).expect("order-six table");
                for i in 0..TAYLOR_SLOTS {
                    for j in 0..TAYLOR_SLOTS - i {
                        assert_eq!(
                            sixth[i][j].to_bits(),
                            published.coefficients[i][j].to_bits(),
                            "q={q} b={b}: c[{i}][{j}] = {:+.17e} against the published {:+.17e}",
                            sixth[i][j],
                            published.coefficients[i][j]
                        );
                    }
                }
                let top = ANCHOR_SIXTH_SLOTS - 1;
                assert!(
                    (0..=top).any(|i| sixth[i][top - i] != 0.0),
                    "q={q} b={b}: the skewed law's anchor has no sixth-order term"
                );
            }
        }
    }
}

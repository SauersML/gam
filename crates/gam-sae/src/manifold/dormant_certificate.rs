//! Dormant-capacity convergence certificate for `K ≫ intrinsic rank` (audit §9 /
//! §34 / §35-priority-4).
//!
//! # The certificate the block lane cannot satisfy
//!
//! The Tier-1 block lane
//! (`crate::sparse_dict::block::fit_block_sparse_dictionary`) certifies a fit by
//! replaying one full alternation and requiring that the gauge-invariant
//! Grassmann-projector displacement of **every** block frame fall under a
//! tolerance (`crate::sparse_dict::block::BlockSparseConvergence::frame_residual`
//! is a max over ALL `G` blocks). In the same alternation, the AuxK lane
//! deliberately identifies the DEAD blocks (zero utilisation) and reseeds their
//! frames from the worst-reconstructed residual rows. The two requirements are
//! mutually exclusive by construction: a policy whose job is to keep moving unused
//! frames can never let the unused frames stop moving.
//!
//! At `K ≫ intrinsic rank` this is not a corner case, it is the regime. On a
//! rank-`r` corpus with `K` capacity slots, at most `~r/b` blocks can carry code
//! mass; the remaining `K/b − ~r/b` slots are *necessarily* unoccupied. For such a
//! slot the reconstruction objective does not merely have a shallow minimum in the
//! frame — it does not depend on the frame at all. Let `N_eff,k = 0`: the block's
//! contribution to `Σ_i ‖x_i − Σ_g z_{ig} D_g‖²` is identically zero for EVERY
//! `D_k ∈ Gr(b,P)`. The decoder of an unoccupied slot is a continuum of exactly
//! equivalent optima, i.e. it is **unidentified**. Demanding that an unidentified
//! parameter reach a fixed point is an ill-posed stopping condition, and no
//! tolerance (however loose) makes it well-posed. It also explains the observed
//! failure mode: forcing more epochs to close the frame residual keeps re-seeding
//! dormant frames on residual rows and erodes explained variance.
//!
//! # The well-posed stopping condition
//!
//! `K` is CAPACITY, not model size. The fitted model size is `K_active`, and the
//! object that must recur is the **ledger** (which slots are occupied), not the
//! full capacity state. This module certifies exactly that:
//!
//! 1. **Active fixed point** — every ACTIVE atom satisfies its continuous
//!    fixed-point/KKT condition (frame projector, shared scale, routing, criterion)
//!    under one replayed alternation.
//! 2. **Dormant exclusion** — dormant atoms are explicitly marked and EXCLUDED from
//!    the recurrence; their frames are free to move (that is what revival does) and
//!    their motion is reported but never gates the verdict.
//! 3. **No profitable birth** — no proposed birth clears its evidence/MDL threshold:
//!    neither a residual-row block birth
//!    (`crate::sparse_dict::block::block_birth_evidence_margin`, a deviance-minus-
//!    rank-charge margin in nats) nor a linear-community curved promotion
//!    ([`super::curve_promotion::propose_curve_promotion`], a `dl_old − dl_new`
//!    saving in bits). Both are *margins*: profitable iff strictly positive.
//! 4. **No profitable structural move** — no merge, demotion, or death strictly
//!    improves the model on the same ledger currency.
//! 5. **Ledger recurrence** — the active/dormant partition is unchanged by the
//!    replayed alternation.
//!
//! Then the fit is converged with `K_active` fitted components and a dormant
//! reservoir of `K − K_active` free slots.
//!
//! # Pure producer
//!
//! [`certify_dormant_capacity`] reads a candidate iterate and its replay image and
//! returns a typed verdict. It mutates no fit state, holds no fit state, and is not
//! wired into any alternation loop: it is the *checkable evidence* a driver may
//! attach to a fit, in the same sense as [`super::curve_promotion`]'s proposal
//! producer.

use ndarray::ArrayView2;

/// Rule that separates OCCUPIED capacity slots from DORMANT ones.
///
/// The rationale is identifiability, not a tuned cutoff. A block's decoder is a
/// point `D_k ∈ Gr(b,P)` — a `b`-dimensional subspace. The reconstruction
/// objective sees `D_k` only through the block's code second moment
/// `C_k = Σ_i z_{ik} z_{ik}ᵀ` (`b×b`), so the frame is pinned by the data only when
/// `C_k` can be full rank, which requires at least `b` effectively-contributing
/// rows. With fewer than `b` effective rows, `C_k` is singular for every frame and
/// a positive-dimensional family of frames attains the same objective value: the
/// slot is unidentified, i.e. dormant, and requiring it to stop moving is the
/// ill-posed condition this module exists to remove.
///
/// "Effective rows" is the participation number of the block's gate mass (see
/// [`effective_occupancy`]), not a raw firing count, so a slot held alive by a
/// single large gate and a spray of numerically-zero ones is correctly dormant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyThreshold {
    /// `N_eff,k ≥ frame_dim`: the exact count at which the block's `b×b` code Gram
    /// can first be nonsingular, hence the exact count at which the frame becomes
    /// identified. `frame_dim` is the block size `b` of the dictionary under test —
    /// it is read from the model, never chosen.
    FrameIdentifiability { frame_dim: usize },
}

impl OccupancyThreshold {
    /// Minimum effective rows an occupied slot must carry.
    pub fn min_effective_rows(&self) -> f64 {
        match *self {
            OccupancyThreshold::FrameIdentifiability { frame_dim } => frame_dim as f64,
        }
    }
}

/// The active/dormant LEDGER of a capacity-`K` dictionary. `active ∪ dormant` is
/// exactly the slot set and the two are disjoint; both are ascending.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomOccupancy {
    /// Occupied slots `A = {k : N_eff,k ≥ threshold}` — the FITTED model.
    pub active: Vec<usize>,
    /// Unoccupied slots — free CAPACITY, unidentified by the objective.
    pub dormant: Vec<usize>,
    /// Per-slot effective occupancy `N_eff,k` (participation number of gate mass).
    pub effective_rows: Vec<f64>,
    /// The rule that produced the partition.
    pub threshold: OccupancyThreshold,
}

impl AtomOccupancy {
    /// Total capacity `K` (in slots).
    pub fn capacity(&self) -> usize {
        self.effective_rows.len()
    }

    /// `K_active`: the fitted model size.
    pub fn n_active(&self) -> usize {
        self.active.len()
    }

}

/// Scalar fixed-point residuals of the ACTIVE state under one replayed alternation.
/// These are the displacements the block lane already measures — explained
/// variance, shared scale `γ`, and the exposed routing — every one of which is a
/// function of the occupied support alone (a dormant slot contributes no code, no
/// reconstruction, and no gate, so it cannot move any of them). They enter the
/// active-KKT check unchanged; only the FRAME residual has to be restricted, which
/// this module does itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct ActiveStateResiduals {
    /// Relative displacement of the fit criterion (e.g. explained variance).
    pub criterion: f64,
    /// Relative displacement of the shared tied scale `γ`.
    pub gamma: f64,
    /// Gauge-invariant displacement of the exposed routing (selected-slot gates).
    pub routing: f64,
}

impl ActiveStateResiduals {
    fn worst(&self) -> f64 {
        self.criterion.max(self.gamma).max(self.routing)
    }

    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("criterion", self.criterion),
            ("gamma", self.gamma),
            ("routing", self.routing),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "certify_dormant_capacity: active {name} residual must be finite and \
                     non-negative, got {value}"
                ));
            }
        }
        Ok(())
    }
}

/// Everything the certificate reads. Nothing here is owned or mutated.
#[derive(Clone, Copy, Debug)]
pub struct DormantCapacityInputs<'a> {
    /// Candidate fixed-point frames, `K×P` with slot `g` occupying rows
    /// `[g·frame_dim, (g+1)·frame_dim)`.
    pub frames: ArrayView2<'a, f32>,
    /// The frames produced by ONE replayed alternation from `frames` (same layout).
    pub replayed_frames: ArrayView2<'a, f32>,
    /// Block size `b`: rows per capacity slot.
    pub frame_dim: usize,
    /// Ledger of the candidate iterate (from its own routing).
    pub occupancy: &'a AtomOccupancy,
    /// Ledger of the REPLAY image (from the replayed routing). Condition 5 asks
    /// these two to agree.
    pub replayed_occupancy: &'a AtomOccupancy,
    /// Scalar active-state displacements under that same replay.
    pub active_residuals: ActiveStateResiduals,
    /// Evidence/MDL margins of every proposed BIRTH — residual-row block births and
    /// linear-community curved promotions alike. Profitable iff strictly positive.
    /// An empty slice means the birth lane proposed nothing.
    pub birth_margins: &'a [f64],
    /// Improvement margins of every proposed MERGE / demotion / death move.
    /// Profitable iff strictly positive.
    pub structural_margins: &'a [f64],
    /// Fixed-point tolerance for the ACTIVE continuous conditions.
    pub tolerance: f64,
}

/// Why a capacity state is not a fixed point. Every variant carries the evidence.
#[derive(Clone, Debug, PartialEq)]
pub enum NotConvergedReason {
    /// An ACTIVE atom is still moving: the worst active continuous residual (frame
    /// projector, criterion, `γ`, routing) exceeds the tolerance.
    ActiveFixedPointOpen { residual: f64, tolerance: f64 },
    /// A proposed birth clears its evidence/MDL threshold.
    ProfitableBirth { margin: f64 },
    /// A merge, demotion, or death strictly improves the model.
    ProfitableStructuralMove { margin: f64 },
    /// The active/dormant ledger did not recur under the replay: slots entered or
    /// left the fitted model.
    LedgerChanged {
        entered: Vec<usize>,
        left: Vec<usize>,
    },
    /// The supplied occupancy is not a partition of the capacity set (an internal
    /// contradiction: the dormant exclusion cannot even be stated).
    OccupancyNotAPartition { capacity: usize },
}

/// Typed verdict of the dormant-capacity certificate.
#[derive(Clone, Debug, PartialEq)]
pub enum CapacityVerdict {
    /// Conditions 1–5 hold: `K_active` is the fitted model size and the remaining
    /// `K − K_active` slots are free capacity.
    Converged,
    NotConverged(NotConvergedReason),
}

/// The checkable certificate: the five conditions, their evidence, and the verdict.
#[derive(Clone, Debug)]
pub struct DormantCapacityCertificate {
    /// The ledger the verdict is about (`K_active` = `occupancy.n_active()`).
    pub occupancy: AtomOccupancy,
    /// Condition 1: every ACTIVE atom's continuous fixed-point/KKT residual is
    /// within tolerance.
    pub active_kkt_ok: bool,
    /// Condition 2: the ledger is a genuine partition and the recurrence check
    /// consulted only active slots — no dormant frame entered any gating residual.
    pub dormant_excluded: bool,
    /// Condition 3.
    pub no_profitable_birth: bool,
    /// Condition 4.
    pub no_profitable_structural_move: bool,
    /// Condition 5.
    pub ledger_recurs: bool,
    /// Max gauge-invariant Grassmann-projector displacement over ACTIVE slots. This
    /// is the residual that gates the verdict.
    pub active_frame_residual: f64,
    /// Max projector displacement over DORMANT slots. REPORTED ONLY — revival moves
    /// this freely and it can never change the verdict. A large value alongside
    /// `Converged` is the expected signature of a live revival policy at `K ≫ rank`,
    /// not a defect.
    pub dormant_frame_residual: f64,
    /// Worst ACTIVE continuous residual actually compared against the tolerance
    /// (frame projector, criterion, `γ`, routing).
    pub active_residual: f64,
    /// The tolerance it was compared against.
    pub tolerance: f64,
    pub verdict: CapacityVerdict,
}

impl DormantCapacityCertificate {
    /// `K_active`: the fitted model size (dormant slots are capacity, not model).
    pub fn n_active(&self) -> usize {
        self.occupancy.n_active()
    }

}

/// Gauge-invariant Grassmann-projector displacement of ONE capacity slot between
/// two frame sets: `‖D Dᵀ − E Eᵀ‖_F` read from `b×b` frame overlaps and normalised
/// by the measured projector norms, so identical stored frames give exactly zero
/// and any `O(b)` change of basis on either side leaves it unchanged. Same
/// invariant the block lane's whole-dictionary residual is built from — restricted
/// here to a single slot so the max can range over the ACTIVE set only.
fn slot_projector_residual(
    previous: ArrayView2<'_, f32>,
    next: ArrayView2<'_, f32>,
    slot: usize,
    frame_dim: usize,
) -> f64 {
    let base = slot * frame_dim;
    let mut previous_norm2 = 0.0_f64;
    let mut next_norm2 = 0.0_f64;
    let mut overlap = 0.0_f64;
    for left in 0..frame_dim {
        for right in 0..frame_dim {
            let mut previous_dot = 0.0_f64;
            let mut next_dot = 0.0_f64;
            let mut cross_dot = 0.0_f64;
            for column in 0..previous.ncols() {
                previous_dot += previous[[base + left, column]] as f64
                    * previous[[base + right, column]] as f64;
                next_dot +=
                    next[[base + left, column]] as f64 * next[[base + right, column]] as f64;
                cross_dot +=
                    previous[[base + left, column]] as f64 * next[[base + right, column]] as f64;
            }
            previous_norm2 += previous_dot * previous_dot;
            next_norm2 += next_dot * next_dot;
            overlap += cross_dot * cross_dot;
        }
    }
    let scale = previous_norm2 + next_norm2;
    let distance2 = (scale - 2.0 * overlap).max(0.0);
    if scale == 0.0 {
        if distance2 == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (distance2 / scale).sqrt()
    }
}

/// Certify a `K ≫ intrinsic-rank` capacity state against conditions 1–5.
///
/// The core of the fix is in what is NOT checked: the frame-projector recurrence
/// ranges over the ACTIVE slots only. Dormant frames are unidentified by the
/// objective and are reseeded on purpose by the AuxK revival policy; their motion
/// is measured, reported as [`DormantCapacityCertificate::dormant_frame_residual`],
/// and then ignored. A verdict of [`CapacityVerdict::Converged`] is therefore
/// invariant under any change whatsoever to the dormant frames.
pub fn certify_dormant_capacity(
    inputs: DormantCapacityInputs<'_>,
) -> Result<DormantCapacityCertificate, String> {
    let DormantCapacityInputs {
        frames,
        replayed_frames,
        frame_dim,
        occupancy,
        replayed_occupancy,
        active_residuals,
        birth_margins,
        structural_margins,
        tolerance,
    } = inputs;

    if frame_dim == 0 {
        return Err("certify_dormant_capacity: frame_dim (block size b) must be >= 1".to_string());
    }
    if frames.dim() != replayed_frames.dim() {
        return Err(format!(
            "certify_dormant_capacity: frames {:?} and replayed frames {:?} must share the K×P shape",
            frames.dim(),
            replayed_frames.dim()
        ));
    }
    if frames.nrows() % frame_dim != 0 {
        return Err(format!(
            "certify_dormant_capacity: {} decoder rows is not a whole number of {frame_dim}-row slots",
            frames.nrows()
        ));
    }
    let capacity = frames.nrows() / frame_dim;
    if occupancy.capacity() != capacity || replayed_occupancy.capacity() != capacity {
        return Err(format!(
            "certify_dormant_capacity: ledgers report {} / {} slots but the decoder has {capacity}",
            occupancy.capacity(),
            replayed_occupancy.capacity()
        ));
    }
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(format!(
            "certify_dormant_capacity: tolerance must be finite and > 0, got {tolerance}"
        ));
    }
    active_residuals.validate()?;
    for (name, margins) in [("birth", birth_margins), ("structural", structural_margins)] {
        if margins.iter().any(|m| !m.is_finite()) {
            return Err(format!(
                "certify_dormant_capacity: every {name} margin must be finite"
            ));
        }
    }

    // Condition 2 — the ledger must actually BE a partition before "excluded from
    // the recurrence" means anything. Checked, not assumed.
    let mut covered = vec![0usize; capacity];
    for &slot in occupancy.active.iter().chain(occupancy.dormant.iter()) {
        if slot >= capacity {
            return Err(format!(
                "certify_dormant_capacity: ledger slot {slot} exceeds capacity {capacity}"
            ));
        }
        covered[slot] += 1;
    }
    let dormant_excluded = covered.iter().all(|&count| count == 1)
        && occupancy.dormant.iter().all(|&slot| {
            // The gating residual below ranges over `active` only; this asserts the
            // two sets cannot overlap, i.e. no dormant frame can reach it.
            occupancy.active.binary_search(&slot).is_err()
        });

    // Condition 1 — continuous fixed point on the ACTIVE support only. The max over
    // dormant slots is computed too, but exclusively for reporting.
    let mut active_frame_residual = 0.0_f64;
    for &slot in &occupancy.active {
        active_frame_residual = active_frame_residual.max(slot_projector_residual(
            frames,
            replayed_frames,
            slot,
            frame_dim,
        ));
    }
    let mut dormant_frame_residual = 0.0_f64;
    for &slot in &occupancy.dormant {
        dormant_frame_residual = dormant_frame_residual.max(slot_projector_residual(
            frames,
            replayed_frames,
            slot,
            frame_dim,
        ));
    }
    let active_residual = active_frame_residual.max(active_residuals.worst());
    let active_kkt_ok = active_residual <= tolerance;

    // Conditions 3 and 4 — no proposed move is profitable on its own currency.
    let best_birth = birth_margins
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let no_profitable_birth = best_birth <= 0.0;
    let best_structural = structural_margins
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let no_profitable_structural_move = best_structural <= 0.0;

    // Condition 5 — the LEDGER recurs. Slot identity matters (a slot that entered
    // the model is a new component), but the dormant frames' CONTENT does not.
    let entered: Vec<usize> = replayed_occupancy
        .active
        .iter()
        .filter(|slot| occupancy.active.binary_search(slot).is_err())
        .cloned()
        .collect();
    let left: Vec<usize> = occupancy
        .active
        .iter()
        .filter(|slot| replayed_occupancy.active.binary_search(slot).is_err())
        .cloned()
        .collect();
    let ledger_recurs = entered.is_empty() && left.is_empty();

    let verdict = if !dormant_excluded {
        CapacityVerdict::NotConverged(NotConvergedReason::OccupancyNotAPartition { capacity })
    } else if !active_kkt_ok {
        CapacityVerdict::NotConverged(NotConvergedReason::ActiveFixedPointOpen {
            residual: active_residual,
            tolerance,
        })
    } else if !no_profitable_birth {
        CapacityVerdict::NotConverged(NotConvergedReason::ProfitableBirth { margin: best_birth })
    } else if !no_profitable_structural_move {
        CapacityVerdict::NotConverged(NotConvergedReason::ProfitableStructuralMove {
            margin: best_structural,
        })
    } else if !ledger_recurs {
        CapacityVerdict::NotConverged(NotConvergedReason::LedgerChanged { entered, left })
    } else {
        CapacityVerdict::Converged
    };

    Ok(DormantCapacityCertificate {
        occupancy: occupancy.clone(),
        active_kkt_ok,
        dormant_excluded,
        no_profitable_birth,
        no_profitable_structural_move,
        ledger_recurs,
        active_frame_residual,
        dormant_frame_residual,
        active_residual,
        tolerance,
        verdict,
    })
}


//! Dormant-capacity convergence certificate for `K ≫ intrinsic rank` (audit §9 /
//! §34 / §35-priority-4).
//!
//! # The certificate the block lane cannot satisfy
//!
//! The Tier-1 block lane
//! ([`crate::sparse_dict::block::fit_block_sparse_dictionary`]) certifies a fit by
//! replaying one full alternation and requiring that the gauge-invariant
//! Grassmann-projector displacement of **every** block frame fall under a
//! tolerance ([`crate::sparse_dict::block::BlockSparseConvergence::frame_residual`]
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
//!    ([`crate::sparse_dict::block::block_birth_evidence_margin`], a deviance-minus-
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

    /// Size of the dormant reservoir `K − K_active`.
    pub fn n_dormant(&self) -> usize {
        self.dormant.len()
    }

    /// Whether slot `k` is dormant (unidentified by the reconstruction objective).
    pub fn is_dormant(&self, slot: usize) -> bool {
        self.dormant.binary_search(&slot).is_ok()
    }
}

/// Effective occupancy `N_eff,k` of every capacity slot from a sparse block
/// routing: `blocks[N,k]` (which slots fired) and `gates[N,k]` (their group-ℓ₂
/// presence), the exact pair the block lane exposes on its fit.
///
/// `N_eff,k = (Σ_i g_{ik})² / Σ_i g_{ik}²` — the participation number (inverse
/// Simpson / Kish effective sample size) of the slot's gate mass. It equals the
/// firing count when the gates are equal, is invariant to the overall code scale
/// `γ` (numerator and denominator are both degree-2 homogeneous), and — unlike a
/// raw count — refuses to count rows whose gate is negligible against the slot's
/// own mass. A slot with no gate mass gets `N_eff = 0`.
pub fn effective_occupancy(
    blocks: ArrayView2<'_, u32>,
    gates: ArrayView2<'_, f32>,
    capacity: usize,
) -> Result<Vec<f64>, String> {
    if blocks.dim() != gates.dim() {
        return Err(format!(
            "effective_occupancy: blocks {:?} and gates {:?} must have the same N×k shape",
            blocks.dim(),
            gates.dim()
        ));
    }
    let mut sum = vec![0.0_f64; capacity];
    let mut sum_sq = vec![0.0_f64; capacity];
    for row in 0..blocks.nrows() {
        for slot in 0..blocks.ncols() {
            let index = blocks[[row, slot]] as usize;
            if index >= capacity {
                return Err(format!(
                    "effective_occupancy: routed block {index} exceeds capacity {capacity}"
                ));
            }
            let gate = gates[[row, slot]] as f64;
            if !gate.is_finite() {
                return Err("effective_occupancy: routing gates must be finite".to_string());
            }
            let gate = gate.abs();
            if gate == 0.0 {
                // Padded slot of a row with fewer than `k` live blocks.
                continue;
            }
            sum[index] += gate;
            sum_sq[index] += gate * gate;
        }
    }
    Ok((0..capacity)
        .map(|k| {
            if sum_sq[k] <= 0.0 {
                0.0
            } else {
                sum[k] * sum[k] / sum_sq[k]
            }
        })
        .collect())
}

/// Partition capacity slots into the active model and the dormant reservoir.
pub fn classify_occupancy(
    effective_rows: &[f64],
    threshold: OccupancyThreshold,
) -> Result<AtomOccupancy, String> {
    let minimum = threshold.min_effective_rows();
    if !(minimum.is_finite() && minimum > 0.0) {
        return Err(format!(
            "classify_occupancy: threshold must demand a positive effective-row count, got {minimum}"
        ));
    }
    let mut active = Vec::new();
    let mut dormant = Vec::new();
    for (slot, &n_eff) in effective_rows.iter().enumerate() {
        if !n_eff.is_finite() || n_eff < 0.0 {
            return Err(format!(
                "classify_occupancy: slot {slot} has invalid effective occupancy {n_eff}"
            ));
        }
        if n_eff >= minimum {
            active.push(slot);
        } else {
            dormant.push(slot);
        }
    }
    Ok(AtomOccupancy {
        active,
        dormant,
        effective_rows: effective_rows.to_vec(),
        threshold,
    })
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

impl CapacityVerdict {
    /// Whether the verdict certifies a fixed point.
    pub fn is_converged(&self) -> bool {
        matches!(self, CapacityVerdict::Converged)
    }
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

    /// Size of the dormant reservoir.
    pub fn n_dormant(&self) -> usize {
        self.occupancy.n_dormant()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    const P: usize = 16;
    const B: usize = 2;
    const CAPACITY: usize = 100; // K = 200 atoms of capacity …
    const N_PLANES: usize = 3; // … on a corpus of intrinsic rank 6 (3 planes × b=2).
    const ROWS_PER_PLANE: usize = 80;

    /// Signal planes live on ambient coordinate pairs (0,1), (2,3), (4,5); all
    /// remaining capacity is parked on coordinate pairs the corpus has zero energy
    /// in, so those slots receive no gate mass at all.
    fn capacity_decoder() -> Array2<f32> {
        let mut decoder = Array2::<f32>::zeros((CAPACITY * B, P));
        let free = P - 2 * N_PLANES;
        for slot in 0..CAPACITY {
            let (first, second) = if slot < N_PLANES {
                (2 * slot, 2 * slot + 1)
            } else {
                // Deterministic parking pairs of DISTINCT signal-free coordinates.
                let low = (slot * 3) % free;
                let high = (low + 1 + slot % (free - 1)) % free;
                (2 * N_PLANES + low, 2 * N_PLANES + high)
            };
            decoder[[slot * B, first]] = 1.0;
            decoder[[slot * B + 1, second]] = 1.0;
        }
        decoder
    }

    /// Rank-6 corpus: `ROWS_PER_PLANE` rows on each of the three signal planes.
    fn corpus() -> Array2<f32> {
        let n = N_PLANES * ROWS_PER_PLANE;
        let mut x = Array2::<f32>::zeros((n, P));
        for plane in 0..N_PLANES {
            for i in 0..ROWS_PER_PLANE {
                let row = plane * ROWS_PER_PLANE + i;
                let theta = i as f32 * 0.7 + plane as f32;
                let radius = 1.0 + (i % 5) as f32 * 0.25;
                x[[row, 2 * plane]] = radius * theta.cos();
                x[[row, 2 * plane + 1]] = radius * theta.sin();
            }
        }
        x
    }

    /// Top-1 block routing by the group-ℓ₂ gate `‖x_i D_gᵀ‖₂` — the same presence
    /// signal the block lane routes on.
    fn route_top1(x: &Array2<f32>, decoder: &Array2<f32>) -> (Array2<u32>, Array2<f32>) {
        let n = x.nrows();
        let mut blocks = Array2::<u32>::zeros((n, 1));
        let mut gates = Array2::<f32>::zeros((n, 1));
        for row in 0..n {
            let mut best_slot = 0usize;
            let mut best_gate = -1.0_f32;
            for slot in 0..CAPACITY {
                let mut energy = 0.0_f32;
                for axis in 0..B {
                    let mut dot = 0.0_f32;
                    for column in 0..P {
                        dot += x[[row, column]] * decoder[[slot * B + axis, column]];
                    }
                    energy += dot * dot;
                }
                let gate = energy.sqrt();
                if gate > best_gate {
                    best_gate = gate;
                    best_slot = slot;
                }
            }
            blocks[[row, 0]] = best_slot as u32;
            gates[[row, 0]] = best_gate;
        }
        (blocks, gates)
    }

    fn ledger(x: &Array2<f32>, decoder: &Array2<f32>) -> AtomOccupancy {
        let (blocks, gates) = route_top1(x, decoder);
        let n_eff = effective_occupancy(blocks.view(), gates.view(), CAPACITY)
            .expect("routing is well formed");
        classify_occupancy(
            &n_eff,
            OccupancyThreshold::FrameIdentifiability { frame_dim: B },
        )
        .expect("identifiability threshold is positive")
    }

    fn inputs<'a>(
        frames: &'a Array2<f32>,
        replayed: &'a Array2<f32>,
        occupancy: &'a AtomOccupancy,
        replayed_occupancy: &'a AtomOccupancy,
        birth_margins: &'a [f64],
    ) -> DormantCapacityInputs<'a> {
        DormantCapacityInputs {
            frames: frames.view(),
            replayed_frames: replayed.view(),
            frame_dim: B,
            occupancy,
            replayed_occupancy,
            active_residuals: ActiveStateResiduals::default(),
            birth_margins,
            structural_margins: &[],
            tolerance: 1.0e-6,
        }
    }

    /// AUDIT §34 decisive test: at `K ≫ intrinsic rank` the certificate reports a
    /// rank-sized active model plus a large dormant reservoir, certifies Converged,
    /// and — the decisive assertion — that verdict does NOT depend on the dormant
    /// frames having stopped moving: replacing every dormant frame with a completely
    /// different subspace (exactly what AuxK revival does each epoch) leaves the
    /// verdict and the gating residual bit-identical.
    #[test]
    fn dormant_capacity_certificate_ignores_dormant_frame_motion_at_k_far_above_rank() {
        let x = corpus();
        let decoder = capacity_decoder();
        let occupancy = ledger(&x, &decoder);

        // K = 200 atoms of capacity; the corpus identifies exactly the three planes.
        assert_eq!(occupancy.capacity(), CAPACITY);
        assert_eq!(occupancy.active, vec![0, 1, 2]);
        assert_eq!(occupancy.n_dormant(), CAPACITY - N_PLANES);
        for &slot in &occupancy.active {
            assert!(
                occupancy.effective_rows[slot] > B as f64,
                "active slot {slot} must clear the frame-identifiability count"
            );
        }
        for &slot in &occupancy.dormant {
            assert_eq!(
                occupancy.effective_rows[slot], 0.0,
                "dormant slot {slot} carries no gate mass on a rank-6 corpus"
            );
        }

        // The alternation is at its ACTIVE fixed point: the replay reproduces the
        // frames and the ledger, and no move is profitable.
        let quiet = certify_dormant_capacity(inputs(
            &decoder,
            &decoder,
            &occupancy,
            &occupancy,
            &[-3.5, -0.25],
        ))
        .expect("well-formed certificate inputs");
        assert_eq!(quiet.verdict, CapacityVerdict::Converged);
        assert!(quiet.active_kkt_ok && quiet.dormant_excluded && quiet.ledger_recurs);
        assert!(quiet.no_profitable_birth && quiet.no_profitable_structural_move);
        assert_eq!(quiet.n_active(), N_PLANES);
        assert_eq!(quiet.n_dormant(), CAPACITY - N_PLANES);
        assert_eq!(quiet.dormant_frame_residual, 0.0);

        // DECISIVE: reseed EVERY dormant frame onto an unrelated subspace (a revival
        // sweep). The dormant projectors move maximally; the verdict must not.
        let mut revived = decoder.clone();
        for &slot in &occupancy.dormant {
            for axis in 0..B {
                for column in 0..P {
                    revived[[slot * B + axis, column]] = 0.0;
                }
                // Park the revived frame on two signal-free coordinates chosen
                // differently from the seed, so the projector genuinely changes.
                let shifted = 2 * N_PLANES + ((slot + 2 * axis + 4) % (P - 2 * N_PLANES));
                revived[[slot * B + axis, shifted]] = 1.0;
            }
        }
        // The revived dormant frames really are different subspaces …
        let moved = certify_dormant_capacity(inputs(
            &decoder,
            &revived,
            &occupancy,
            &occupancy,
            &[-3.5, -0.25],
        ))
        .expect("well-formed certificate inputs");
        assert!(
            moved.dormant_frame_residual > 0.1,
            "the test must actually move the dormant frames (residual {})",
            moved.dormant_frame_residual
        );
        // … and the certificate is completely indifferent to that motion.
        assert_eq!(moved.verdict, CapacityVerdict::Converged);
        assert_eq!(moved.active_frame_residual, quiet.active_frame_residual);
        assert_eq!(moved.active_residual, quiet.active_residual);
        assert_eq!(moved.n_active(), N_PLANES);

        // CONTROL — the certificate is not vacuously permissive: move an ACTIVE
        // frame by the same construction and it refuses.
        let mut active_moved = decoder.clone();
        for column in 0..P {
            active_moved[[0, column]] = 0.0;
        }
        active_moved[[0, 7]] = 1.0;
        let refused = certify_dormant_capacity(inputs(
            &decoder,
            &active_moved,
            &occupancy,
            &occupancy,
            &[-3.5, -0.25],
        ))
        .expect("well-formed certificate inputs");
        assert!(!refused.active_kkt_ok);
        assert!(matches!(
            refused.verdict,
            CapacityVerdict::NotConverged(NotConvergedReason::ActiveFixedPointOpen { .. })
        ));
    }

    /// Conditions 3–5 are load-bearing: a profitable birth, a profitable structural
    /// move, and a ledger that fails to recur each refuse the certificate even
    /// though the ACTIVE continuous fixed point holds exactly.
    #[test]
    fn dormant_capacity_certificate_refuses_profitable_moves_and_open_ledger() {
        let x = corpus();
        let decoder = capacity_decoder();
        let occupancy = ledger(&x, &decoder);

        let birth = certify_dormant_capacity(inputs(
            &decoder,
            &decoder,
            &occupancy,
            &occupancy,
            &[-1.0, 4.75],
        ))
        .expect("well-formed certificate inputs");
        assert!(!birth.no_profitable_birth);
        assert_eq!(
            birth.verdict,
            CapacityVerdict::NotConverged(NotConvergedReason::ProfitableBirth { margin: 4.75 })
        );

        let mut structural = inputs(&decoder, &decoder, &occupancy, &occupancy, &[]);
        let margins = [0.5_f64];
        structural.structural_margins = &margins;
        let merged = certify_dormant_capacity(structural).expect("well-formed certificate inputs");
        assert!(!merged.no_profitable_structural_move);
        assert_eq!(
            merged.verdict,
            CapacityVerdict::NotConverged(NotConvergedReason::ProfitableStructuralMove {
                margin: 0.5
            })
        );

        // A dormant slot that woke up (its N_eff cleared the identifiability count)
        // is a CHANGED ledger — that, and not dormant frame motion, is the recurrence
        // failure the certificate is supposed to catch.
        let mut woken = occupancy.clone();
        woken.effective_rows[7] = 32.0;
        let woken = classify_occupancy(&woken.effective_rows, woken.threshold)
            .expect("identifiability threshold is positive");
        let open =
            certify_dormant_capacity(inputs(&decoder, &decoder, &occupancy, &woken, &[-1.0]))
                .expect("well-formed certificate inputs");
        assert!(!open.ledger_recurs);
        assert_eq!(
            open.verdict,
            CapacityVerdict::NotConverged(NotConvergedReason::LedgerChanged {
                entered: vec![7],
                left: vec![],
            })
        );
    }
}

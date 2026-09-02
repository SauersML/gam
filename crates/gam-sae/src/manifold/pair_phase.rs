//! Pairwise PHASE-COUPLING screen over accepted circle-atoms (report F4) — the
//! joint-dependence blind spot the energy screen ([`super::pair_kappa`]) documents
//! but cannot close, plus the "phase circuit" causal observable (report App D).
//!
//! # What the energy screen misses, and why phase closes it
//!
//! The pairwise ENERGY screen adjudicates a pair on the normalised energy
//! cross-moment `ρ = E[r_A²·r_B²] / (E[r_A²]·E[r_B²])`. It is a SECOND-order
//! statistic in the per-row radii, so it sees only PRESENCE (energy) coupling.
//! [`super::tests_joint_vs_cascade_2131`] pins three distinct joint dependencies a
//! cascade can split across two frames and shows exactly which the energy screen
//! catches:
//!
//!   * **case 2 — gated torus** (shared presence gate, independent angles):
//!     `ρ = 1/q > 1`. The energy screen FIRES. Its home tail.
//!   * **case 3 — two DENSE circles with a correlated PHASE law** (`θ_B ≈ θ_A + φ`
//!     at `q = 1`): presence is constant so every `r² ≡ 1`, the energy
//!     cross-moment is BLIND to the phase law ⇒ `ρ ≈ 1`, NO fire. A genuine
//!     inter-atom dependence, invisible to any energy-only screen.
//!   * **case 1 — a single circle SPLIT across two dense frames**: complementary
//!     energies `r_A² + r_B² ≈ const` ⇒ `ρ ≈ 1/2`, the LOWER tail. The `ρ > 1`
//!     merge screen does not adjudicate it.
//!
//! This module closes case 3 with a PHASE statistic and case 1 with a lower-tail
//! FUSE-RACE proposal.
//!
//! # The phase statistic
//!
//! For a co-firing circle pair, read each row's in-plane angle
//! `θ_·= atan2(p₂, p₁)` from the projection onto the atom's certified 2-plane (the
//! same projection [`super::pair_kappa`] squares to get `r²`; here we keep the
//! angle). With per-row weights `w_n` (the gate product — mass on rows where BOTH
//! atoms are present) the coupling statistic on harmonic `h` is the WEIGHTED MEAN
//! RESULTANT LENGTH of the phase difference,
//!
//! ```text
//! T_h = |Σ_n w_n · e^{i·h·(θ_A,n − θ_B,n)}| / Σ_n w_n ,   h = 1, 2,
//! ```
//!
//! plus the orientation-REVERSING channel on the phase SUM
//!
//! ```text
//! T_sum = |Σ_n w_n · e^{i·(θ_A,n + θ_B,n)}| / Σ_n w_n .
//! ```
//!
//!   * `T₁` fires on a **rotation coupling** `θ_B = θ_A + φ` (a torus density on a
//!     shifted diagonal) — the case-3 phase law.
//!   * `T₂` fires on a **reflection / diameter coupling** `θ_B = ±θ_A + φ mod π`
//!     (antipodal identification: the h=2 harmonic is invariant to a π flip).
//!   * `T_sum` fires on an **orientation-reversing coupling** `θ_B = −θ_A + φ` (a
//!     mirror law), which `T₁` on the difference cannot see (its difference angle
//!     `2θ_A − φ` still winds).
//!
//! Each `T ∈ [0, 1]`: `1` is a rigid phase lock, `0` is no coupling. Under
//! independence `T` is the resultant of a random walk, `E[T] = O(1/√N_eff)` — NOT
//! zero at finite sample, and NOT the parametric Rayleigh value once the two
//! per-column spectra are coloured. So we do NOT lean on the Rayleigh null.
//!
//! # Calibration — the standing phase-randomized null, not Rayleigh
//!
//! The null is drawn from the standing battery's phase-randomised surrogate
//! ([`crate::null_battery::phase_randomized_surrogate`]): it re-randomises each
//! ambient column's Fourier phases along token order, preserving that column's
//! one-dimensional POWER SPECTRUM exactly while destroying any coherent
//! cross-column phase relationship. Re-projecting the surrogate through the SAME
//! two atom bases and recomputing `T_h` gives the null law of the statistic AT THE
//! OBSERVED SAMPLE SIZE AND SPECTRUM. The screen's `z` and `p` are read off that
//! empirical null, so a coloured spectrum or a small `N_eff` inflates the null
//! `T` and is automatically discounted — the exact failure a parametric Rayleigh
//! null would miss. A spike-in power harness (plant `θ_B = θ_A + φ` at a known
//! concentration, confirm detection) lives in the tests.
//!
//! # Multiplicity — e-BH over the pair × channel ledger
//!
//! A screen over `K` atoms tests `O(K²)` pairs on 3 channels each. Each channel's
//! exact Monte-Carlo screen (`B` phase-randomised surrogate draws plus the
//! observed statistic form `B + 1` exchangeable values under the null) yields the
//! valid permutation e-value `e = (B+1)·1{no surrogate ≥ observed}`
//! ([`permutation_e_value`]): `E_0[e] = (B+1)·P(observed is the strict maximum) =
//! 1` under exchangeability, reaching its maximum `B + 1` exactly when the
//! observed statistic beats every surrogate. The family is controlled with e-BH
//! ([`ebh_reject`]): FDR ≤ α with NO independence assumption across the (heavily
//! dependent) pair statistics — the property a p-value BH could not give here.
//!
//! **Budget.** A Monte-Carlo screen with `B` draws resolves an e-value no larger
//! than `B + 1`, and e-BH rejects a family of `m` = (pairs × 3 channels) entries
//! only when the largest e clears `m/(α·k)` for some `k` — so at minimum
//! `B + 1 ≥ m/α`. The replicate budget MUST be sized to the family; it cannot be
//! a fixed round number. (The former `½·p^{−½}` calibrator was worse still: it
//! capped `e` at `½√(B+1) ≈ 7` for `B = 200`, below the `1/α = 20` a SINGLE
//! rejection needs, so the screen could never fire at its declared level. The
//! reciprocal `(B+1)/(1+#{≥})` is not a fix either — it is not an e-value, its
//! null mean being the harmonic number `H_{B+1} ≈ ln B`, not `≤ 1`.)
//!
//! # The verdict, at zero reconstruction cost
//!
//! A firing `T` on a DENSE pair (both circles already fully reconstructed by their
//! marginals) means a joint `d = 2` torus coordinate would capture a real density
//! the two 1-D charts cannot — and it costs NOTHING in reconstruction EV (the
//! marginals are unchanged). The screen therefore proposes a torus coordinate on
//! positive phase evidence. The lower-tail case-1 fragmentation instead triggers a
//! FUSE-RACE ([`fuse_race_candidate`]): a single fused 2-plane candidate whose
//! terminal joint fit adjudicates against keeping two atoms.

use ndarray::Array2;

/// Which phase harmonic / channel a coupling statistic measures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseChannel {
    /// `h = 1` on the phase DIFFERENCE `θ_A − θ_B`: a rotation coupling
    /// `θ_B = θ_A + φ` (the case-3 diagonal torus density).
    Difference1,
    /// `h = 2` on the phase difference: a reflection / diameter coupling,
    /// invariant to an antipodal `π` flip of either angle.
    Difference2,
    /// `h = 1` on the phase SUM `θ_A + θ_B`: an orientation-REVERSING coupling
    /// `θ_B = −θ_A + φ` that the difference channels cannot see.
    Sum1,
}

impl PhaseChannel {
    pub fn as_str(self) -> &'static str {
        match self {
            PhaseChannel::Difference1 => "difference_h1",
            PhaseChannel::Difference2 => "difference_h2",
            PhaseChannel::Sum1 => "sum_h1",
        }
    }

    /// The three channels the screen always evaluates, in ledger order.
    pub fn all() -> [PhaseChannel; 3] {
        [
            PhaseChannel::Difference1,
            PhaseChannel::Difference2,
            PhaseChannel::Sum1,
        ]
    }

}

/// One channel's screen result: observed resultant, its Monte-Carlo null
/// summary against the phase-randomised surrogate, and the calibrated e-value.
#[derive(Clone, Debug)]
pub struct ChannelVerdict {
    pub channel: PhaseChannel,
    /// Observed weighted mean resultant length `T ∈ [0, 1]`.
    pub resultant: f64,
    /// Kish effective sample size the resultant was formed on.
    pub n_eff: f64,
    /// Mean and sd of `T` under the phase-randomised null.
    pub null_mean: f64,
    pub null_sd: f64,
    /// Standardised excess over the null, `(T − null_mean) / null_sd`.
    pub z: f64,
    /// Exact upper-tail Monte-Carlo p-value `(1 + #{T_null ≥ T}) / (B + 1)`.
    pub p_value: f64,
    /// Valid permutation e-value `(B+1)·1{no surrogate ≥ T}` (feeds the e-BH
    /// ledger); `E_0[e] = 1`, maximal value `B + 1`. See `permutation_e_value`.
    pub e_value: f64,
}

/// The phase-coupling verdict for one atom pair, across all three channels.
#[derive(Clone, Debug)]
pub struct PhaseVerdict {
    pub atom_a: usize,
    pub atom_b: usize,
    /// Rows on which both atoms are present (the weight support).
    pub n_co_active: usize,
    /// Per-channel results, in [`PhaseChannel::all`] order.
    pub channels: Vec<ChannelVerdict>,
    /// The channel with the largest e-value (strongest coupling evidence).
    pub best_channel: PhaseChannel,
    /// Largest e-value across channels (the pair's ledger entry).
    pub best_e_value: f64,
    /// Smallest per-channel p-value across channels.
    pub best_p_value: f64,
    /// `ρ`-style lower-tail evidence for a SPLIT single circle: the energy
    /// cross-moment `E[r_A²r_B²]/(E r_A² · E r_B²)` restricted to co-active rows.
    /// `< 1` with complementary energies flags a fragmentation (case 1).
    pub energy_rho: f64,
    /// Coefficient of variation of `r_A² + r_B²` on co-active rows. Near-zero
    /// (constant total energy) is the complementarity signature of a split chart.
    pub total_energy_cv: f64,
    /// True ⇒ a joint `d = 2` torus coordinate is proposed on positive phase
    /// evidence (set by [`screen_all_pairs_phase`] after the e-BH ledger).
    pub torus_proposed: bool,
    /// True ⇒ the pair reads as a lower-tail SPLIT single circle: a fuse-race
    /// candidate is worth building for the terminal joint fit to adjudicate.
    pub fuse_race_proposed: bool,
}

/// Number of phase-randomised surrogate draws the null is estimated on. Matches
/// the order of the standing battery's replicate budget; the exact-null p-value
/// floor is `1/(B+1)`.
pub const PHASE_NULL_REPLICATES: usize = 200;

/// e-BH (Wang & Ramdas 2022) at FDR level `alpha` over a family of e-values.
/// Returns the indices of the rejected hypotheses (the discoveries). Valid with
/// NO independence assumption across the e-values — the property that lets the
/// dependent pair statistics share one ledger. Sort descending, find the largest
/// `k` with the `k`-th largest e-value `≥ m/(α·k)`, reject those `k`.
pub fn ebh_reject(e_values: &[f64], alpha: f64) -> Vec<usize> {
    let m = e_values.len();
    if m == 0 || !(alpha > 0.0) {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| {
        e_values[j]
            .partial_cmp(&e_values[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut k_star = 0usize;
    for rank in 1..=m {
        let e = e_values[order[rank - 1]];
        if e >= (m as f64) / (alpha * rank as f64) {
            k_star = rank;
        }
    }
    order.into_iter().take(k_star).collect()
}

/// A residual inter-factor coupling the separation did not remove: the pair, its
/// strongest phase channel, and that channel's e-value (its e-BH ledger entry).
#[derive(Clone, Copy, Debug)]
pub struct ResidualCoupling {
    pub atom_a: usize,
    pub atom_b: usize,
    pub channel: PhaseChannel,
    pub e_value: f64,
}

/// The pairwise phase-coupling screen for a candidate circle factorization — the
/// DUAL READING of the same e-BH ledger the phase-fusion screen uses.
///
/// [`screen_all_pairs_phase`] proposes a torus BINDING on POSITIVE coupling
/// evidence: an e-BH discovery over the (pair × channel) surrogate-null ledger.
/// The separation problem (#2111) reads the SAME ledger for whether ANY residual
/// coupling survives. On a dense product-of-circles torus, ring-ness (a
/// second-order, radial signal) is degenerate: every 2-plane inside the span of
/// two circles is equally "ring-like", so no second-order score can split the
/// product into its circle factors. The identifying signal is the joint PHASE
/// law: a product of independent circles factorises, so every cross-phase
/// resultant `T_h` sits in the phase-randomised surrogate null.
///
/// **What non-rejection does and does NOT mean.** Family-wise non-rejection is
/// evidence of ABSENCE OF DETECTED COUPLING at the given budget — it is NOT a
/// certificate of independence. A hypothesis test controls the false-DISCOVERY
/// side; it never certifies a null. Failing to reject can equally mean the
/// coupling is real but the phase-randomised power / replicate budget was too low
/// to see it (recall the budget bound `B + 1 ≥ m/α` in the module header — below
/// it the ledger CANNOT reject, so `no_coupling_detected` is then vacuous). Read
/// this struct as "the screen found no phase coupling it could act on", never as
/// "the factors are proven independent".
#[derive(Clone, Debug)]
pub struct PhaseCouplingScreen {
    /// True ⇒ no (pair × channel) coupling cleared the e-BH surrogate-null ledger
    /// at `alpha`: NO evidence of pairwise phase coupling was found at this
    /// replicate budget. This is absence of a detected effect, NOT a proof of
    /// independence (see the struct docs; check the budget bound before relying on
    /// a "clean" screen).
    pub no_coupling_detected: bool,
    /// The couplings that DID clear the ledger (e-BH discoveries). Empty iff
    /// `no_coupling_detected`. Each names a pair whose phase law did not factorise
    /// — the caller re-separates it if a rotation within the pair's 4-plane can
    /// null the coupling (a whitened-basis BLEND, the #2111 45° saddle), or fuses
    /// it if the coupling is rotation-invariant (a GENUINE torus density the
    /// marginals miss).
    pub residual_couplings: Vec<ResidualCoupling>,
    /// The full per-pair verdicts (all pairs in `a < b` order), for the
    /// fuse-vs-reseparate adjudication and diagnostics.
    pub verdicts: Vec<PhaseVerdict>,
}

/// A fused single-atom 2-plane candidate for the case-1 lower-tail race: one circle
/// whose diameters were split across two frames is re-expressed as ONE 2-plane
/// spanning the top-two energy directions of the union of the two atoms' ambient
/// columns. The terminal joint fit adjudicates this against keeping two atoms.
#[derive(Clone, Debug)]
pub struct FuseRaceCandidate {
    /// The union ambient columns the two planes touched.
    pub support_columns: Vec<usize>,
    /// The fused `p × 2` basis (top-two principal directions on the union
    /// support), embedded back into the full ambient width `p`.
    pub basis: Array2<f64>,
    /// Fraction of the union-support energy the fused 2-plane captures. A genuine
    /// single split circle sits near `1.0` (the circle IS 2-dimensional); a true
    /// pair of independent circles leaks energy to the discarded directions.
    pub captured_energy_fraction: f64,
    /// The two atoms this candidate would fuse (a [`StructureMove::Fusion`]).
    pub atom_a: usize,
    pub atom_b: usize,
}

// ---------------------------------------------------------------------------
// App D — the PHASE CIRCUIT (causal half).
//
// A firing phase screen is CORRELATIONAL: it certifies that two atoms share a
// phase law. A phase CIRCUIT is the CAUSAL upgrade — a measured transfer law
// `A_BA` such that steering `θ_A` by `Δ` moves `θ_B` by a PREDICTED amount, with a
// dose-response. The pulled-back chart-to-chart operator machinery already exists
// ([`crate::chart_transfer`]); here we (1) fit the SO(2)-valued transfer operator
// from co-firing angles, (2) certify it (isometry + Lie-equivariance defects,
// polar transfer angle), and (3) score an INTERVENTION SHARD: steer `θ_A += Δ`,
// push through `A_BA`, compare the predicted `Δθ_B` to the observed response. A
// certified circuit = a transfer law whose predicted dose matches the intervention
// with slope ≈ 1 and small residual.
// ---------------------------------------------------------------------------

/// Certificate for a candidate phase circuit: the measured transfer law plus its
/// isometry / equivariance defects and the intervention dose-response.
#[derive(Clone, Debug)]
pub struct PhaseCircuitCertificate {
    /// Polar SO(2) transfer angle `dθ_B/dθ_A` when the operator rotates
    /// (`det > 0`); `None` when it reflects/collapses (`det ≤ 0`) — reported, not
    /// folded into a spurious angle.
    pub transfer_angle: Option<f64>,
    /// Sign of the operator determinant: `+1` orientation-preserving (rotation),
    /// `−1` orientation-reversing (mirror circuit), `0` collapse.
    pub orientation: i8,
    /// Frobenius `‖AᵀA − I‖`: zero for an isometric (pure-rotation) transport.
    pub transport_defect: f64,
    /// Frobenius `‖A·G − G·A‖` against the shared SO(2) generator: zero when the
    /// transfer commutes with rotation (an equivariant phase law).
    pub equivariance_defect: f64,
    /// Slope of observed `Δθ_B` on predicted `Δθ_B` across the intervention shard
    /// (through the origin). `≈ 1` for a faithful circuit.
    pub dose_slope: f64,
    /// Fraction of intervention-response variance the predicted dose explains.
    pub dose_r2: f64,
    /// True ⇒ a certified phase circuit: a near-orthogonal transfer whose
    /// through-origin dose slope sits in the identity band and is significantly
    /// nonzero — the presence test `|β̂| > t_{n−1}(1 − α/2)·SE` at
    /// `PHASE_SCREEN_ALPHA` (the transfer really tracks the prediction, not
    /// noise), the Student-`t` critical value at the fit's `n − 1` residual dof.
    pub certified: bool,
}

#[cfg(test)]
mod tests {
    include!("pair_phase_tests.rs");
}

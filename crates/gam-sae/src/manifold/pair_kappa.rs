//! Pairwise κ merge-proposal screen over accepted manifold atoms (Part-2
//! statistical-debt closure).
//!
//! The ISA birth producer ([`super::isa_seed`]) certifies ONE plane at a time on
//! the fourth-moment contrast `(κ − 2)²`, with analytic anchors `κ = 1` (dense
//! circle), `κ = 2` (Gaussian blend), `κ = 1/q` (gated circle). Birth is a
//! per-atom decision; nothing looks at PAIRS of accepted atoms to ask whether two
//! atoms are really two faces of ONE structure. This screen closes that: for each
//! co-activating pair it computes a JOINT fourth-moment statistic on the pair's
//! shared rows and adjudicates
//!
//!   * "TWO INDEPENDENT structures" — keep both (two genuinely separate circles),
//!   * "ONE structure" — a product/torus or a single curved set split across two
//!     atoms — a MERGE proposal.
//!
//! THE STATISTIC. Take the two atoms' per-row in-plane radii `r_A²`, `r_B²`
//! (energy each atom reconstructs on that row). The load-bearing quantity is the
//! NORMALISED ENERGY CROSS-MOMENT
//!
//! ```text
//! ρ = E[r_A²·r_B²] / (E[r_A²]·E[r_B²]).
//! ```
//!
//! This is a joint fourth-order moment (two squared amplitudes), the pairwise
//! analogue of the single-plane κ. Its meaning is an analytic anchor, exactly like
//! the ISA κ anchors:
//!
//!   * TWO INDEPENDENT structures ⇒ the two energies are independent ⇒
//!     `E[r_A²r_B²] = E[r_A²]·E[r_B²]` EXACTLY ⇒ `ρ = 1`. This is a SHARP null, not
//!     a tuned value: independence of the presence/energy of two separate charts
//!     forces `ρ = 1` regardless of their individual κ's, amplitudes, or gates.
//!   * ONE BOUND structure (a torus/product with a SHARED presence gate `q`, or a
//!     single curved set both atoms co-activate on) ⇒ the two energies are
//!     PRESENCE-COUPLED: both are large together (both present) or both ~0 (both
//!     absent) ⇒ `E[r_A²r_B²] = E[r_A²]·E[r_B²]/q > E[r_A²]·E[r_B²]` ⇒ `ρ = 1/q > 1`.
//!     A gated torus split into its two circle factors has co-gated presence, so
//!     `ρ = 1/q` — the binding the merge screen must catch.
//!
//! So `ρ − 1` is the pairwise contrast: zero under independence, positive under
//! binding. (A FULLY DENSE product — both circles present on every row — has
//! `q = 1 ⇒ ρ = 1`: it is geometrically indistinguishable from two independent
//! dense circles, and correctly NOT flagged; there is no information favouring a
//! merge over keeping two atoms. The screen fires only on POSITIVE evidence of
//! binding, which requires a shared gate `q < 1`.)
//!
//! THE OFF-ROWS ARE LOAD-BEARING — the moments are taken over ALL rows the two
//! atoms share as a domain (the full `n`), NOT over the co-active intersection.
//! The presence coupling lives in the JOINT on/off pattern: on rows where BOTH
//! atoms are active, each radius is ~constant (`r_A² ≈ a²`), so the cross-moment
//! there is `≈ E[r_A²]·E[r_B²]` and `ρ → 1` for ANY pair — conditioning on
//! co-activation destroys exactly the signal the screen needs. Evaluating over
//! all rows keeps the absent rows (`r² ≈ 0`) that carry the coupling: co-gated
//! atoms are jointly zero and jointly large together (`ρ = 1/q`), independent
//! atoms factor (`ρ = 1`).
//!
//! THE THRESHOLD IS DERIVED, NOT TUNED. Under the independence null, `ρ̂` is a
//! ratio of sample moments; first-order, `ρ̂ − 1 ≈ meanᵢ (Uᵢ−1)(Vᵢ−1)` with
//! `U = r_A²/E[r_A²]`, `V = r_B²/E[r_B²]` (the ratio's estimated denominators
//! cancel most of the numerator's fluctuation — the two subtracted terms of the
//! ratio-estimator influence function). Under independence the factors are
//! independent, so the delta-method leading-order variance is
//! `Var(ρ̂) ≈ (κ_A − 1)(κ_B − 1)/N` on the `N` rows (`κ_A = E[r_A⁴]/E[r_A²]²` is
//! each atom's own single-plane κ — the fourth-moment spread the ISA producer
//! already measures). The pair is flagged as ONE structure iff
//! `ρ̂ − 1 > z·√((κ_A − 1)(κ_B − 1)/N)` at the conventional `z = 3` — the SAME
//! level the ISA sub-sample floor is derived at ([`super::isa_seed`]
//! `ISA_SUBSAMPLE_FLOOR`). No magic ε.

use crate::coactivation_conditionality::{
    CoactivationConditionality, VaryingCoefficientConfig, estimate_on_rows,
    residual_gate_activities,
};
use ndarray::{Array1, ArrayView2};

use super::isa_seed::IsaPlaneCandidate;

/// Per-row projected energies of one atom on its own 2-plane, plus the per-row
/// active gate. Recomputed from the raw data and the atom's basis so the screen is
/// self-contained (no reliance on stored radii).
struct PlaneEnergies {
    /// `r_i² = ‖P_plane·(x_i − mean)‖²` per row.
    r2: Vec<f64>,
    /// Whether row `i` clears the atom's own noise floor (active).
    active: Vec<bool>,
}

/// The verdict for one atom pair.
#[derive(Clone, Debug)]
pub struct PairVerdict {
    /// Indices of the two atoms in the accepted set.
    pub atom_a: usize,
    pub atom_b: usize,
    /// Rows the statistic is evaluated over (the full shared domain `n`).
    pub n_rows: usize,
    /// Rows on which BOTH atoms are active (a diagnostic; NOT the statistic
    /// domain — see the module header on why conditioning on these destroys the
    /// coupling signal).
    pub n_co_active: usize,
    /// Observed normalised energy cross-moment `ρ̂`.
    pub rho: f64,
    /// Each atom's own single-plane κ over all rows (drives the null SE).
    pub kappa_a: f64,
    pub kappa_b: f64,
    /// Independence-null standard error of `ρ̂`, `√((κ_A − 1)(κ_B − 1)/N)`.
    pub rho_se: f64,
    /// `z`-score of `ρ̂ − 1` against the independence null.
    pub z: f64,
    /// Context-conditional residual-gate stability report. This is the structure
    /// search gate; the pooled coupling certificate protects the weighted
    /// Pearson statistic, while `rho` remains fourth-moment evidence and cannot
    /// by itself propose fusion.
    pub conditionality: Option<CoactivationConditionality>,
    /// True iff the residual-gate weighted-Pearson coupling has enough local KL
    /// margin to dominate continuous-context by-smooth drift.
    pub conditional_stable: bool,
    /// True ⇒ the pair reads as ONE bound structure (a merge proposal).
    pub merge_proposed: bool,
}

/// The evidence level the screen fires at — `z = 3`, the same level the ISA
/// sub-sample floor is derived at. Not a model threshold: the significance of a
/// standard-normal contrast, shared with the birth certificate's design edge.
const PAIR_Z: f64 = 3.0;

/// The minimum row count for the delta-method SE to be trustworthy. Unlike the
/// ISA κ certificate's `ISA_SUBSAMPLE_FLOOR` ([`super::isa_seed`]) — a
/// RESOLUTION bound at the gated `q = 0.43` design edge — this is only a
/// first-order-expansion validity floor: the `z = 3` contrast already
/// self-scales its SE with `N`, so all the floor must exclude is the small-`N`
/// regime where the delta-method variance itself is untrustworthy. Below it a
/// pair cannot be adjudicated (returned with `merge_proposed = false`,
/// `z = 0`).
const PAIR_ROW_FLOOR: usize = 500;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn lcg_normal(s: &mut u64) -> f64 {
        let u1 = lcg(s).max(1e-12);
        let u2 = lcg(s);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Build an [`IsaPlaneCandidate`] for an axis-aligned circle on ambient dims
    /// `(d0, d1)`, active on the given row mask. Only the fields the screen reads
    /// (`basis`, `gate_logits`) need to be faithful; the rest are placeholders.
    fn axis_candidate(p: usize, d0: usize, d1: usize, active: &[bool]) -> IsaPlaneCandidate {
        let n = active.len();
        let mut basis = Array2::<f64>::zeros((p, 2));
        basis[[d0, 0]] = 1.0;
        basis[[d1, 1]] = 1.0;
        let gate_logits: Vec<f64> = active
            .iter()
            .map(|&a| if a { 0.0 } else { f64::NEG_INFINITY })
            .collect();
        IsaPlaneCandidate {
            basis,
            amplitudes: [1.0, 1.0],
            phases_turns: Array2::<f64>::zeros((n, 1)),
            gate_logits,
            kappa: 1.0,
            q_hat: active.iter().filter(|&&a| a).count() as f64 / n as f64,
        }
    }

    /// TWO INDEPENDENT GATED CIRCLES — planted on disjoint ambient dims with
    /// INDEPENDENT per-row presence gates (q = 0.5 each). The energies are
    /// independent ⇒ ρ ≈ 1 ⇒ NO merge flag.
    #[test]
    fn two_independent_circles_not_flagged() {
        let mut s = 0x9A1_u64;
        let n = 6000usize;
        let p = 8usize;
        let qa = 0.5;
        let qb = 0.5;
        let mut data = Array2::<f64>::zeros((n, p));
        let mut act_a = vec![false; n];
        let mut act_b = vec![false; n];
        for i in 0..n {
            // INDEPENDENT presence draws.
            if lcg(&mut s) < qa {
                act_a[i] = true;
                let th = std::f64::consts::TAU * lcg(&mut s);
                data[[i, 0]] += th.cos();
                data[[i, 1]] += th.sin();
            }
            if lcg(&mut s) < qb {
                act_b[i] = true;
                let th = std::f64::consts::TAU * lcg(&mut s);
                data[[i, 2]] += th.cos();
                data[[i, 3]] += th.sin();
            }
            for j in 0..p {
                data[[i, j]] += 0.02 * lcg_normal(&mut s);
            }
        }
        let mean = Array1::<f64>::zeros(p);
        let ca = axis_candidate(p, 0, 1, &act_a);
        let cb = axis_candidate(p, 2, 3, &act_b);
        let v = screen_pair(data.view(), &mean, 0, 1, &ca, &cb);
        assert!(
            !v.merge_proposed,
            "two INDEPENDENT circles must NOT be flagged: ρ={:.4} z={:.3}",
            v.rho, v.z
        );
        assert!(
            (v.rho - 1.0).abs() < 0.15,
            "independent ρ must sit near the null 1.0; got {:.4}",
            v.rho
        );
    }

    /// GATED TORUS SPLIT INTO TWO ATOMS — a product of two circles with a SHARED
    /// presence gate (q = 0.4): when the torus is present, BOTH circle factors are
    /// active (independent angles); when absent, neither. Co-gated presence ⇒
    /// ρ ≈ 1/q ≈ 2.5 ≫ 1 ⇒ MERGE flagged.
    #[test]
    fn gated_torus_split_is_flagged() {
        let mut s = 0x7013_u64;
        let n = 6000usize;
        let p = 8usize;
        let q = 0.4;
        let mut data = Array2::<f64>::zeros((n, p));
        let mut act = vec![false; n];
        for i in 0..n {
            // SHARED presence: one gate drives BOTH factors.
            if lcg(&mut s) < q {
                act[i] = true;
                let ta = std::f64::consts::TAU * lcg(&mut s);
                let tb = std::f64::consts::TAU * lcg(&mut s); // independent angle
                data[[i, 0]] += ta.cos();
                data[[i, 1]] += ta.sin();
                data[[i, 2]] += tb.cos();
                data[[i, 3]] += tb.sin();
            }
            for j in 0..p {
                data[[i, j]] += 0.02 * lcg_normal(&mut s);
            }
        }
        let mean = Array1::<f64>::zeros(p);
        let ca = axis_candidate(p, 0, 1, &act);
        let cb = axis_candidate(p, 2, 3, &act);
        let v = screen_pair(data.view(), &mean, 0, 1, &ca, &cb);
        assert!(
            v.merge_proposed,
            "a gated torus split into two atoms MUST be flagged: ρ={:.4} z={:.3}",
            v.rho, v.z
        );
        // ρ must land near the co-gated anchor 1/q = 2.5, well above the null.
        assert!(
            v.rho > 1.5,
            "co-gated torus ρ must be ≫ 1 (anchor 1/q≈2.5); got {:.4}",
            v.rho
        );
    }

    /// screen_all_pairs on a three-atom set (independent A, and a co-gated B–C
    /// torus) must return exactly the {B,C} proposal.
    #[test]
    fn screen_all_pairs_selects_only_bound_pair() {
        let mut s = 0xC0FFEE_u64;
        let n = 6000usize;
        let p = 12usize;
        let q_iso = 0.5;
        let q_tor = 0.4;
        let mut data = Array2::<f64>::zeros((n, p));
        let mut act_a = vec![false; n];
        let mut act_bc = vec![false; n];
        for i in 0..n {
            if lcg(&mut s) < q_iso {
                act_a[i] = true;
                let th = std::f64::consts::TAU * lcg(&mut s);
                data[[i, 0]] += th.cos();
                data[[i, 1]] += th.sin();
            }
            if lcg(&mut s) < q_tor {
                act_bc[i] = true;
                let tb = std::f64::consts::TAU * lcg(&mut s);
                let tc = std::f64::consts::TAU * lcg(&mut s);
                data[[i, 2]] += tb.cos();
                data[[i, 3]] += tb.sin();
                data[[i, 4]] += tc.cos();
                data[[i, 5]] += tc.sin();
            }
            for j in 0..p {
                data[[i, j]] += 0.02 * lcg_normal(&mut s);
            }
        }
        let mean = Array1::<f64>::zeros(p);
        let cands = vec![
            axis_candidate(p, 0, 1, &act_a),  // 0: independent
            axis_candidate(p, 2, 3, &act_bc), // 1: torus factor
            axis_candidate(p, 4, 5, &act_bc), // 2: torus factor
        ];
        let flags = screen_all_pairs(data.view(), &mean, &cands);
        assert_eq!(flags.len(), 1, "exactly one bound pair expected");
        assert!(
            flags[0].atom_a == 1 && flags[0].atom_b == 2,
            "the flagged pair must be the co-gated torus factors (1,2); got ({},{})",
            flags[0].atom_a,
            flags[0].atom_b
        );
    }
}

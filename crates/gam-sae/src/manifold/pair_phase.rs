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
//! A screen over `K` atoms tests `O(K²)` pairs on 3 channels each. Each Monte-Carlo
//! `p` is turned into an e-value with the admissible calibrator `e = ½·p^{−½}`
//! (`∫₀¹ ½ p^{−½} dp = 1`, decreasing in `p`), and the family is controlled with
//! e-BH ([`ebh_reject`]): FDR ≤ α with NO independence assumption across the
//! (heavily dependent) pair statistics — the property a p-value BH could not give
//! here.
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

use ndarray::{Array1, Array2, ArrayView2};

use super::isa_seed::IsaPlaneCandidate;
use crate::null_battery::phase_randomized_surrogate;
use gam_solve::structure_search::StructureMove;

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

    /// The signed per-row phase this channel forms a resultant of, given the two
    /// atoms' in-plane angles. The `h = 2` channel doubles the difference.
    fn row_angle(self, theta_a: f64, theta_b: f64) -> f64 {
        match self {
            PhaseChannel::Difference1 => theta_a - theta_b,
            PhaseChannel::Difference2 => 2.0 * (theta_a - theta_b),
            PhaseChannel::Sum1 => theta_a + theta_b,
        }
    }
}

/// Per-row in-plane angle and squared radius of one atom on its own 2-plane, plus
/// the per-row gate. Recomputed from the raw data and the atom's basis so the
/// screen is self-contained (no reliance on stored `phases_turns`).
struct PlanePhases {
    /// `θ_i = atan2(p₂, p₁)` per row (radians, `(−π, π]`).
    theta: Vec<f64>,
    /// `r_i² = p₁² + p₂²` per row (energy on the plane; feeds the fuse-race).
    r2: Vec<f64>,
    /// Whether row `i` clears the atom's own gate (present).
    active: Vec<bool>,
}

/// Project every row of `data` onto the candidate's 2-plane, recovering
/// `(θ, r², active)`. Mirrors [`super::pair_kappa`]'s projection but keeps the
/// angle rather than only the squared radius.
fn plane_phases(
    data: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
    cand: &IsaPlaneCandidate,
) -> PlanePhases {
    let (n, p) = data.dim();
    let mut theta = vec![0.0_f64; n];
    let mut r2 = vec![0.0_f64; n];
    let mut active = vec![false; n];
    for i in 0..n {
        let (mut p1, mut p2) = (0.0_f64, 0.0_f64);
        for j in 0..p {
            let ri = data[[i, j]] - mean[j];
            p1 += ri * cand.basis[[j, 0]];
            p2 += ri * cand.basis[[j, 1]];
        }
        theta[i] = p2.atan2(p1);
        r2[i] = p1 * p1 + p2 * p2;
        active[i] = cand.gate_logits[i].is_finite();
    }
    PlanePhases { theta, r2, active }
}

/// The weighted mean resultant length `T` and Kish effective sample size for one
/// channel, given the per-row angles and non-negative weights. Returns `(T, n_eff)`
/// with `T = 0`, `n_eff = 0` on empty weight.
fn resultant(
    channel: PhaseChannel,
    theta_a: &[f64],
    theta_b: &[f64],
    w: &[f64],
) -> (f64, f64) {
    let (mut cx, mut sx, mut wsum, mut wsq) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..theta_a.len() {
        let wi = w[i];
        if wi <= 0.0 {
            continue;
        }
        let ang = channel.row_angle(theta_a[i], theta_b[i]);
        cx += wi * ang.cos();
        sx += wi * ang.sin();
        wsum += wi;
        wsq += wi * wi;
    }
    if wsum <= 0.0 {
        return (0.0, 0.0);
    }
    let t = ((cx * cx + sx * sx).sqrt() / wsum).min(1.0);
    let n_eff = wsum * wsum / wsq;
    (t, n_eff)
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
    /// Admissible-calibrator e-value `½·p^{−½}` (feeds the e-BH ledger).
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

/// Complementary-energy tolerance for the case-1 fuse-race: the coefficient of
/// variation of `r_A² + r_B²` on co-active rows must sit below this for the total
/// energy to read as CONSTANT (one circle's radius split across two frames).
const FUSE_TOTAL_ENERGY_CV_MAX: f64 = 0.20;

/// Upper bound on the co-active energy cross-moment `ρ` for the fuse-race: a split
/// single circle has anti-correlated diameters, `ρ ≈ 1/2 < 1`. The screen requires
/// `ρ` strictly below the independence null to call fragmentation.
const FUSE_RHO_MAX: f64 = 0.85;

/// Admissible p-to-e calibrator `e = ½·p^{−½}` (`κ = ½`): `∫₀¹ κ p^{κ−1} dp = 1`
/// and it is decreasing in `p`, so it is a valid e-value for any valid p-value.
fn calibrate_e_value(p: f64) -> f64 {
    let p = p.clamp(f64::MIN_POSITIVE, 1.0);
    0.5 * p.powf(-0.5)
}

/// The four ambient dims spanned by the two axis-recoverable planes, if each plane
/// is axis-aligned (one non-zero basis entry per column). Returns `None` for a
/// non-axis plane, in which case the null uses the full ambient width.
fn plane_support_columns(cand: &IsaPlaneCandidate) -> Option<[usize; 2]> {
    let p = cand.basis.nrows();
    let mut cols = [usize::MAX; 2];
    for c in 0..2 {
        let mut hit = None;
        for j in 0..p {
            if cand.basis[[j, c]].abs() > 1e-9 {
                if hit.is_some() {
                    return None;
                }
                hit = Some(j);
            }
        }
        cols[c] = hit?;
    }
    Some(cols)
}

/// Build the reduced-column submatrix and remapped bases so the (expensive)
/// phase-randomised surrogate runs on only the ambient columns the two planes
/// touch. Falls back to the full matrix when either plane is not axis-aligned.
fn reduced_null_inputs(
    data: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
    cand_a: &IsaPlaneCandidate,
    cand_b: &IsaPlaneCandidate,
) -> Option<(Array2<f64>, Array1<f64>, IsaPlaneCandidate, IsaPlaneCandidate)> {
    let sa = plane_support_columns(cand_a)?;
    let sb = plane_support_columns(cand_b)?;
    let mut cols: Vec<usize> = Vec::new();
    for &c in sa.iter().chain(sb.iter()) {
        if !cols.contains(&c) {
            cols.push(c);
        }
    }
    let n = data.nrows();
    let pr = cols.len();
    let mut sub = Array2::<f64>::zeros((n, pr));
    for (jr, &jc) in cols.iter().enumerate() {
        for i in 0..n {
            sub[[i, jr]] = data[[i, jc]];
        }
    }
    let mut sub_mean = Array1::<f64>::zeros(pr);
    for (jr, &jc) in cols.iter().enumerate() {
        sub_mean[jr] = mean[jc];
    }
    let remap = |cand: &IsaPlaneCandidate, supp: [usize; 2]| -> IsaPlaneCandidate {
        let mut basis = Array2::<f64>::zeros((pr, 2));
        for c in 0..2 {
            let jr = cols.iter().position(|&x| x == supp[c]).unwrap_or(0);
            basis[[jr, c]] = cand.basis[[supp[c], c]];
        }
        IsaPlaneCandidate {
            basis,
            amplitudes: cand.amplitudes,
            phases_turns: cand.phases_turns.clone(),
            gate_logits: cand.gate_logits.clone(),
            kappa: cand.kappa,
            q_hat: cand.q_hat,
        }
    };
    Some((sub, sub_mean, remap(cand_a, sa), remap(cand_b, sb)))
}

/// Screen one atom pair for phase coupling. `data` is the ambient matrix the atoms
/// were certified on, `mean` its column mean. The three channel statistics are
/// calibrated against `replicates` phase-randomised surrogate draws seeded from
/// `seed`; `torus_proposed` is left `false` here and set by the family-level e-BH
/// ledger in [`screen_all_pairs_phase`]. A single-pair caller can read
/// `best_p_value` directly.
pub fn screen_pair_phase(
    data: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
    atom_a: usize,
    atom_b: usize,
    cand_a: &IsaPlaneCandidate,
    cand_b: &IsaPlaneCandidate,
    replicates: usize,
    seed: u64,
) -> Result<PhaseVerdict, String> {
    let pa = plane_phases(data, mean, cand_a);
    let pb = plane_phases(data, mean, cand_b);
    let n = pa.theta.len();
    // Weights: mass on rows where BOTH atoms are present (the gate product).
    let w: Vec<f64> = (0..n)
        .map(|i| if pa.active[i] && pb.active[i] { 1.0 } else { 0.0 })
        .collect();
    let n_co_active = w.iter().filter(|&&x| x > 0.0).count();

    // Observed per-channel resultants.
    let channels_all = PhaseChannel::all();
    let observed: Vec<(f64, f64)> = channels_all
        .iter()
        .map(|&ch| resultant(ch, &pa.theta, &pb.theta, &w))
        .collect();

    // Null: phase-randomised surrogate of the ambient columns the planes touch,
    // re-projected through the SAME bases. One surrogate feeds all three channels.
    let (null_data, null_mean, ncand_a, ncand_b) =
        reduced_null_inputs(data, mean, cand_a, cand_b).unwrap_or_else(|| {
            (
                data.to_owned(),
                mean.clone(),
                copy_candidate(cand_a),
                copy_candidate(cand_b),
            )
        });
    let mut null_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(replicates); channels_all.len()];
    for rep in 0..replicates {
        let rep_seed = mix_seed(seed, rep as u64);
        let surrogate = phase_randomized_surrogate(null_data.view(), rep_seed)?;
        let sa = plane_phases(surrogate.view(), &null_mean, &ncand_a);
        let sb = plane_phases(surrogate.view(), &null_mean, &ncand_b);
        for (ci, &ch) in channels_all.iter().enumerate() {
            let (t, _) = resultant(ch, &sa.theta, &sb.theta, &w);
            null_samples[ci].push(t);
        }
    }

    let mut channels = Vec::with_capacity(channels_all.len());
    for (ci, &ch) in channels_all.iter().enumerate() {
        let (t, n_eff) = observed[ci];
        let samples = &null_samples[ci];
        let b = samples.len();
        let mean_null = samples.iter().sum::<f64>() / b.max(1) as f64;
        let var = samples
            .iter()
            .map(|x| (x - mean_null) * (x - mean_null))
            .sum::<f64>()
            / (b.saturating_sub(1)).max(1) as f64;
        let sd = var.sqrt();
        let exceed = samples.iter().filter(|&&x| x >= t).count();
        let p_value = (1 + exceed) as f64 / (b + 1) as f64;
        let z = if sd > 0.0 { (t - mean_null) / sd } else { 0.0 };
        channels.push(ChannelVerdict {
            channel: ch,
            resultant: t,
            n_eff,
            null_mean: mean_null,
            null_sd: sd,
            z,
            p_value,
            e_value: calibrate_e_value(p_value),
        });
    }

    // Best (strongest-coupling) channel.
    let (best_idx, best_e) = channels
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.e_value))
        .fold((0usize, f64::NEG_INFINITY), |acc, (i, e)| {
            if e > acc.1 { (i, e) } else { acc }
        });
    let best_channel = channels[best_idx].channel;
    let best_p_value = channels
        .iter()
        .map(|c| c.p_value)
        .fold(f64::INFINITY, f64::min);

    // Lower-tail fuse-race diagnostics on the co-active rows: energy cross-moment
    // and the coefficient of variation of the total energy.
    let (energy_rho, total_energy_cv) = coactive_energy_stats(&pa.r2, &pb.r2, &w);
    let fuse_race_proposed =
        n_co_active >= 2 && energy_rho < FUSE_RHO_MAX && total_energy_cv < FUSE_TOTAL_ENERGY_CV_MAX;

    Ok(PhaseVerdict {
        atom_a,
        atom_b,
        n_co_active,
        channels,
        best_channel,
        best_e_value: best_e,
        best_p_value,
        energy_rho,
        total_energy_cv,
        torus_proposed: false,
        fuse_race_proposed,
    })
}

/// Energy cross-moment `ρ` and the CV of the total energy on co-active rows.
/// A single circle split across two frames has `r_A² + r_B² ≈ const` (low CV) and
/// anti-correlated diameters (`ρ < 1`).
fn coactive_energy_stats(r2a: &[f64], r2b: &[f64], w: &[f64]) -> (f64, f64) {
    let (mut ma, mut mb, mut cross, mut wsum) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..r2a.len() {
        if w[i] <= 0.0 {
            continue;
        }
        ma += r2a[i];
        mb += r2b[i];
        cross += r2a[i] * r2b[i];
        wsum += 1.0;
    }
    if wsum < 2.0 || ma <= 0.0 || mb <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    ma /= wsum;
    mb /= wsum;
    cross /= wsum;
    let rho = cross / (ma * mb);
    // CV of the total energy r_A² + r_B².
    let mut mt = 0.0_f64;
    for i in 0..r2a.len() {
        if w[i] > 0.0 {
            mt += r2a[i] + r2b[i];
        }
    }
    mt /= wsum;
    let mut vt = 0.0_f64;
    for i in 0..r2a.len() {
        if w[i] > 0.0 {
            let d = (r2a[i] + r2b[i]) - mt;
            vt += d * d;
        }
    }
    vt /= wsum;
    let cv = if mt > 0.0 { vt.sqrt() / mt } else { f64::NAN };
    (rho, cv)
}

/// Field-by-field copy of a candidate ([`IsaPlaneCandidate`] does not derive
/// `Clone`), used to own the null inputs in the non-axis fallback.
fn copy_candidate(c: &IsaPlaneCandidate) -> IsaPlaneCandidate {
    IsaPlaneCandidate {
        basis: c.basis.clone(),
        amplitudes: c.amplitudes,
        phases_turns: c.phases_turns.clone(),
        gate_logits: c.gate_logits.clone(),
        kappa: c.kappa,
        q_hat: c.q_hat,
    }
}

fn mix_seed(seed: u64, rep: u64) -> u64 {
    let mut x = seed ^ rep.wrapping_mul(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x
}

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

/// Screen every co-firing atom pair for phase coupling and mark `torus_proposed`
/// via a single e-BH ledger at FDR level `alpha` over ALL (pair × channel)
/// e-values. Returns the full per-pair verdict vector (all pairs, in `a<b` order),
/// with `torus_proposed = true` exactly on the pairs whose best channel is an e-BH
/// discovery. `replicates`/`seed` drive the phase-randomised null.
pub fn screen_all_pairs_phase(
    data: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
    candidates: &[IsaPlaneCandidate],
    replicates: usize,
    seed: u64,
    alpha: f64,
) -> Result<Vec<PhaseVerdict>, String> {
    let mut verdicts = Vec::new();
    let mut ledger_e: Vec<f64> = Vec::new();
    // (verdict index) for each ledger entry — one entry per pair × CHANNEL.
    // The per-pair maximum over channels is NOT an e-value (for three null
    // channels E[max e] = 8/5 > 1), so feeding `best_e_value` to e-BH would
    // void the FDR guarantee the module header states. Every calibrated
    // channel e-value enters the ledger under its own entry, exactly the
    // pair × channel family the header prices; a pair is proposed when ANY of
    // its channel entries is rejected.
    let mut ledger_owner: Vec<usize> = Vec::new();
    for a in 0..candidates.len() {
        for b in (a + 1)..candidates.len() {
            let pair_seed = mix_seed(seed, ((a as u64) << 20) ^ b as u64);
            let v = screen_pair_phase(
                data,
                mean,
                a,
                b,
                &candidates[a],
                &candidates[b],
                replicates,
                pair_seed,
            )?;
            if v.n_co_active >= 2 {
                for ch in &v.channels {
                    ledger_e.push(ch.e_value);
                    ledger_owner.push(verdicts.len());
                }
            }
            verdicts.push(v);
        }
    }
    for idx in ebh_reject(&ledger_e, alpha) {
        verdicts[ledger_owner[idx]].torus_proposed = true;
    }
    Ok(verdicts)
}

/// The phase screen's contribution to the structure-search proposal stream: a
/// [`StructureMove::Fusion`] for every pair the screen binds. Two proposal kinds
/// share the Fusion move (both assert a `BindingEdge` the terminal joint fit
/// adjudicates):
///
///   * **torus binding** — a dense pair with a certified phase law (the e-BH
///     `torus_proposed` discovery): a joint `d = 2` torus coordinate at zero
///     reconstruction cost.
///   * **fuse-race** — a lower-tail split single circle (`fuse_race_proposed`):
///     the union 2-plane raced against keeping two atoms.
///
/// Returned in `(a < b)` order, torus bindings before fuse-races, each pair once.
/// This is the producer a [`crate::structure_harvest`] call-site enqueues; it is
/// kept here (beside the screen) so the hot harvester file need only thread the
/// accepted [`IsaPlaneCandidate`] set and ambient matrix and call this.
pub fn phase_fusion_moves(
    data: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
    candidates: &[IsaPlaneCandidate],
    replicates: usize,
    seed: u64,
    alpha: f64,
) -> Result<Vec<StructureMove>, String> {
    let verdicts = screen_all_pairs_phase(data, mean, candidates, replicates, seed, alpha)?;
    let mut moves = Vec::new();
    let mut seen: Vec<(usize, usize)> = Vec::new();
    for v in verdicts.iter().filter(|v| v.torus_proposed) {
        seen.push((v.atom_a, v.atom_b));
        moves.push(StructureMove::Fusion {
            a: v.atom_a,
            b: v.atom_b,
        });
    }
    for v in verdicts.iter().filter(|v| v.fuse_race_proposed) {
        if !seen.contains(&(v.atom_a, v.atom_b)) {
            moves.push(StructureMove::Fusion {
                a: v.atom_a,
                b: v.atom_b,
            });
        }
    }
    Ok(moves)
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

/// Build the union-plane fuse-race candidate for a pair flagged by the lower-tail
/// diagnostics. Computes the `2×2`… here the union support is up to 4 columns; we
/// take the top-two eigenvectors of the co-active energy covariance on that
/// support as the fused 2-plane. Returns `None` if the pair is not axis-recoverable
/// or has too few co-active rows.
pub fn fuse_race_candidate(
    data: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
    atom_a: usize,
    atom_b: usize,
    cand_a: &IsaPlaneCandidate,
    cand_b: &IsaPlaneCandidate,
) -> Option<FuseRaceCandidate> {
    let sa = plane_support_columns(cand_a)?;
    let sb = plane_support_columns(cand_b)?;
    let mut cols: Vec<usize> = Vec::new();
    for &c in sa.iter().chain(sb.iter()) {
        if !cols.contains(&c) {
            cols.push(c);
        }
    }
    let d = cols.len();
    if d < 2 {
        return None;
    }
    let pa = plane_phases(data, mean, cand_a);
    let pb = plane_phases(data, mean, cand_b);
    let n = data.nrows();
    let active: Vec<bool> = (0..n).map(|i| pa.active[i] && pb.active[i]).collect();
    let n_act = active.iter().filter(|&&x| x).count();
    if n_act < 2 {
        return None;
    }
    // Co-active energy covariance on the union support (centred by `mean`).
    let mut cov = Array2::<f64>::zeros((d, d));
    for i in 0..n {
        if !active[i] {
            continue;
        }
        let mut v = vec![0.0_f64; d];
        for (r, &c) in cols.iter().enumerate() {
            v[r] = data[[i, c]] - mean[c];
        }
        for r in 0..d {
            for s in 0..d {
                cov[[r, s]] += v[r] * v[s];
            }
        }
    }
    cov.mapv_inplace(|x| x / n_act as f64);
    let total: f64 = (0..d).map(|r| cov[[r, r]]).sum();
    let (evecs, evals) = symmetric_eig_jacobi(&cov);
    // Top-two eigenpairs.
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap_or(std::cmp::Ordering::Equal));
    let captured = if total > 0.0 {
        (evals[order[0]].max(0.0) + evals[order[1]].max(0.0)) / total
    } else {
        0.0
    };
    let p = data.ncols();
    let mut basis = Array2::<f64>::zeros((p, 2));
    for k in 0..2 {
        let col = order[k];
        for (r, &c) in cols.iter().enumerate() {
            basis[[c, k]] = evecs[[r, col]];
        }
    }
    Some(FuseRaceCandidate {
        support_columns: cols,
        basis,
        captured_energy_fraction: captured,
        atom_a,
        atom_b,
    })
}

/// Symmetric-eigendecomposition by cyclic Jacobi rotations, for the small
/// (`d ≤ 4`) union-support covariance of the fuse-race. Returns `(eigenvectors as
/// columns, eigenvalues)`.
fn symmetric_eig_jacobi(a: &Array2<f64>) -> (Array2<f64>, Vec<f64>) {
    let d = a.nrows();
    let mut m = a.clone();
    let mut v = Array2::<f64>::eye(d);
    for _ in 0..64 {
        let mut off = 0.0_f64;
        for p in 0..d {
            for q in (p + 1)..d {
                off += m[[p, q]] * m[[p, q]];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = m[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[[p, p]];
                let aqq = m[[q, q]];
                let phi = 0.5 * (2.0 * apq).atan2(app - aqq);
                let (c, s) = (phi.cos(), phi.sin());
                for k in 0..d {
                    let mkp = m[[k, p]];
                    let mkq = m[[k, q]];
                    m[[k, p]] = c * mkp + s * mkq;
                    m[[k, q]] = -s * mkp + c * mkq;
                }
                for k in 0..d {
                    let mpk = m[[p, k]];
                    let mqk = m[[q, k]];
                    m[[p, k]] = c * mpk + s * mqk;
                    m[[q, k]] = -s * mpk + c * mqk;
                }
                for k in 0..d {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp + s * vkq;
                    v[[k, q]] = -s * vkp + c * vkq;
                }
            }
        }
    }
    let evals: Vec<f64> = (0..d).map(|i| m[[i, i]]).collect();
    (v, evals)
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

use crate::chart_transfer::{certify_square_transfer, so2_polar_angle};

/// The SO(2) circle generator `[[0,−1],[1,0]]` — the Lie generator both charts'
/// rotation actions share, used for the equivariance defect.
fn so2_generator() -> Array2<f64> {
    let mut g = Array2::<f64>::zeros((2, 2));
    g[[0, 1]] = -1.0;
    g[[1, 0]] = 1.0;
    g
}

/// Least-squares SO(2)-ish transfer operator `A_BA` mapping atom A's unit phase
/// vector `(cos θ_A, sin θ_A)` to atom B's `(cos θ_B, sin θ_B)`, weighted by `w`.
/// `A = (Σ w u_B u_Aᵀ)(Σ w u_A u_Aᵀ)⁻¹`. For a rotation law `θ_B = θ_A + φ` this
/// recovers the rotation by `φ`; for a reversal `θ_B = −θ_A + φ` it recovers a
/// reflection (negative determinant), which the certificate reports honestly.
pub fn phase_transfer_operator(
    theta_a: &[f64],
    theta_b: &[f64],
    w: &[f64],
) -> Result<Array2<f64>, String> {
    if theta_a.len() != theta_b.len() || theta_a.len() != w.len() {
        return Err("phase_transfer_operator: length mismatch".to_string());
    }
    let mut cross = Array2::<f64>::zeros((2, 2)); // Σ w u_B u_Aᵀ
    let mut gram = Array2::<f64>::zeros((2, 2)); // Σ w u_A u_Aᵀ
    let mut wsum = 0.0_f64;
    for i in 0..theta_a.len() {
        let wi = w[i];
        if wi <= 0.0 {
            continue;
        }
        let ua = [theta_a[i].cos(), theta_a[i].sin()];
        let ub = [theta_b[i].cos(), theta_b[i].sin()];
        for r in 0..2 {
            for c in 0..2 {
                cross[[r, c]] += wi * ub[r] * ua[c];
                gram[[r, c]] += wi * ua[r] * ua[c];
            }
        }
        wsum += wi;
    }
    if wsum <= 0.0 {
        return Err("phase_transfer_operator: no positive-weight rows".to_string());
    }
    let det = gram[[0, 0]] * gram[[1, 1]] - gram[[0, 1]] * gram[[1, 0]];
    let scale = (gram[[0, 0]].abs() * gram[[1, 1]].abs()).max(1e-300);
    if !det.is_finite() || det.abs() <= f64::EPSILON.sqrt() * scale {
        return Err("phase_transfer_operator: singular input angle gram (θ_A not exciting)".to_string());
    }
    let inv = {
        let mut m = Array2::<f64>::zeros((2, 2));
        m[[0, 0]] = gram[[1, 1]] / det;
        m[[1, 1]] = gram[[0, 0]] / det;
        m[[0, 1]] = -gram[[0, 1]] / det;
        m[[1, 0]] = -gram[[1, 0]] / det;
        m
    };
    Ok(cross.dot(&inv))
}

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
    /// True ⇒ a certified phase circuit: rotation transport with small defects and
    /// a dose-response that tracks the prediction (slope near 1, high `R²`).
    pub certified: bool,
}

/// Predict atom B's phase after steering atom A to `theta_a_steered`, by pushing
/// the unit vector through the transfer operator and re-reading the angle.
pub fn predicted_theta_b(op: ArrayView2<'_, f64>, theta_a_steered: f64) -> f64 {
    let ua = [theta_a_steered.cos(), theta_a_steered.sin()];
    let vb0 = op[[0, 0]] * ua[0] + op[[0, 1]] * ua[1];
    let vb1 = op[[1, 0]] * ua[0] + op[[1, 1]] * ua[1];
    vb1.atan2(vb0)
}

/// Signed circular difference `a − b` wrapped to `(−π, π]`.
fn wrap_pi(x: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut y = x % tau;
    if y > std::f64::consts::PI {
        y -= tau;
    } else if y <= -std::f64::consts::PI {
        y += tau;
    }
    y
}

/// Thresholds for certifying a phase circuit.
const CIRCUIT_TRANSPORT_DEFECT_MAX: f64 = 0.35;
const CIRCUIT_DOSE_SLOPE_LO: f64 = 0.6;
const CIRCUIT_DOSE_SLOPE_HI: f64 = 1.4;
const CIRCUIT_DOSE_R2_MIN: f64 = 0.5;

/// Certify a phase circuit from the co-firing angles and an intervention shard.
///
/// The operator is fit from `(theta_a, theta_b, w)`. The intervention shard is the
/// caller-supplied `(predicted_dtheta_b, observed_dtheta_b)` pairs: for each shard
/// row the caller steered `θ_A += Δ`, read the PREDICTED `Δθ_B` from
/// [`predicted_theta_b`], and measured the OBSERVED `Δθ_B` from the frozen model's
/// re-encoded B coordinate. This function scores the dose-response and combines it
/// with the operator's transport/equivariance defects into a certificate.
pub fn certify_phase_circuit(
    theta_a: &[f64],
    theta_b: &[f64],
    w: &[f64],
    predicted_dtheta_b: &[f64],
    observed_dtheta_b: &[f64],
) -> Result<PhaseCircuitCertificate, String> {
    let op = phase_transfer_operator(theta_a, theta_b, w)?;
    let det = op[[0, 0]] * op[[1, 1]] - op[[0, 1]] * op[[1, 0]];
    let orientation = if det > 1e-9 {
        1
    } else if det < -1e-9 {
        -1
    } else {
        0
    };
    let transfer_angle = so2_polar_angle(op.view()).ok();
    let g = so2_generator();
    let cert = certify_square_transfer(op.view(), g.view(), g.view())?;

    // Dose-response: regress observed Δθ_B on predicted Δθ_B through the origin,
    // wrapping both to (−π, π]. Slope near 1 with high R² = faithful transfer.
    if predicted_dtheta_b.len() != observed_dtheta_b.len() {
        return Err("certify_phase_circuit: dose shard length mismatch".to_string());
    }
    let (mut sxx, mut sxy, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..predicted_dtheta_b.len() {
        let x = wrap_pi(predicted_dtheta_b[i]);
        let y = wrap_pi(observed_dtheta_b[i]);
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    let dose_slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    // R² of the through-origin fit: 1 − SS_res/SS_tot with SS_tot = Σ y².
    let ss_res = syy - 2.0 * dose_slope * sxy + dose_slope * dose_slope * sxx;
    let dose_r2 = if syy > 0.0 {
        (1.0 - ss_res / syy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let certified = orientation == 1
        && cert.transport_defect <= CIRCUIT_TRANSPORT_DEFECT_MAX
        && dose_slope >= CIRCUIT_DOSE_SLOPE_LO
        && dose_slope <= CIRCUIT_DOSE_SLOPE_HI
        && dose_r2 >= CIRCUIT_DOSE_R2_MIN;

    Ok(PhaseCircuitCertificate {
        transfer_angle,
        orientation,
        transport_defect: cert.transport_defect,
        equivariance_defect: cert.equivariance_defect,
        dose_slope,
        dose_r2,
        certified,
    })
}

/// Read the co-firing in-plane angles and gate-product weights for one atom pair
/// from raw data — the input a caller needs to fit [`phase_transfer_operator`] and
/// run an intervention shard against a real model. Returns `(θ_A, θ_B, w)`.
pub fn coactive_angles(
    data: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
    cand_a: &IsaPlaneCandidate,
    cand_b: &IsaPlaneCandidate,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let pa = plane_phases(data, mean, cand_a);
    let pb = plane_phases(data, mean, cand_b);
    let n = pa.theta.len();
    let w: Vec<f64> = (0..n)
        .map(|i| if pa.active[i] && pb.active[i] { 1.0 } else { 0.0 })
        .collect();
    (pa.theta, pb.theta, w)
}

#[cfg(test)]
mod tests {
    include!("pair_phase_tests.rs");
}

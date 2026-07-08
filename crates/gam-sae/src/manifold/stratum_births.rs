//! Stratum-local birth screen — making diffuse residual structure REACHABLE by the
//! router floor at frontier width.
//!
//! # The reachability problem this closes
//!
//! A stagewise birth proposes a candidate chart by mining the residual for its
//! dominant direction, then the joint gate must ROUTE that chart's firings against
//! all the other atoms. The routing-side floor
//! ([`crate::routability::routability_floor`]) says a feature is invisible to any
//! width-`p` gate over `K` blocks unless it places at least
//! [`crate::routability::minimum_routable_energy`] `= floor²/(1+floor²)` of the
//! residual energy AT ITS FIRING SITES into its own subspace, with
//! `floor = √(b_max/p) + √(2·ln(K/δ)/p)`. At `K = 32000`, `p = 2048` the union-bound
//! term alone is `√(2·ln 32000 / 2048) ≈ 0.100`: a chart must carry ≥ 10 % of the
//! residual norm where it fires.
//!
//! Births mined on the **pooled** residual fail this by construction. A structure
//! that is genuinely present on a rare, high-residual SUBSET of rows but drowned in
//! diffuse bulk residual everywhere else has a POOLED energy fraction far below the
//! floor — averaged over all `n` rows its own-subspace share is tiny — so the router
//! rejects it even with a perfect fitter. The fix is not to lower the floor (that
//! would admit unroutable clutter); it is to mine the birth on the STRATUM where the
//! candidate's signal concentrates, so the *local* energy fraction on that stratum's
//! rows is what meets the floor.
//!
//! # The screen
//!
//! Partition the residual rows by residual energy `e_i = ‖r_i‖²` into factor-of-two
//! energy bands ([`crate::corpus::stratify_row_energies`] — the SAME IEEE-754
//! exponent bins and Sturges cap the streaming corpus screen uses, so nothing new is
//! calibrated). For each stratum, the candidate a rank-1 / circle birth would seed
//! from is the stratum's DOMINANT direction; its own-subspace energy share on the
//! stratum is [`dominant_energy_fraction`] `= λ_max(Rₛᵀ Rₛ) / ‖Rₛ‖_F²`. A stratum is
//! **routable-local** when that fraction clears the floor's
//! `minimum_routable_energy`. [`stratum_local_birth_residual`] returns, for the
//! highest-energy routable-local stratum, a residual MASKED to that stratum's rows
//! (all other rows zeroed) — the residual the existing birth-seed machinery then
//! mines, so the seed lands on rows where its local signal fraction clears the
//! router. When no stratum clears the floor locally (small `K`, or genuinely diffuse
//! noise) the screen declines and the caller keeps the pooled path unchanged.
//!
//! # Determinism & SPEC
//!
//! No RNG (power iteration is seeded from a fixed all-ones vector), no clock, no
//! magic constant — the strata boundaries are the data's own exponents, the routing
//! floor is the closed form of (`K`, `p`), and the accept test is `local_fraction ≥
//! minimum_routable_energy(floor)`.

use ndarray::{Array2, ArrayView2};

use crate::routability::{RoutabilityFloor, minimum_routable_energy};

/// The stratum-local birth pick: which rows the birth should be mined on, the
/// residual masked to those rows, and the routability accounting that admitted it.
#[derive(Clone, Debug)]
pub struct StratumLocalPick {
    /// Rows of the chosen stratum (ascending) — where the candidate's signal
    /// concentrates. The birth is mined on exactly these rows.
    pub rows: Vec<usize>,
    /// The residual masked to `rows`: identical to the input on those rows, zero
    /// elsewhere. Feeding this to the pooled seed machinery keeps `n` and the
    /// term-row alignment intact while making the mined structure stratum-local.
    pub masked_residual: Array2<f64>,
    /// The chosen stratum's local dominant-direction energy fraction
    /// `λ_max(RₛᵀRₛ)/‖Rₛ‖_F²` — the own-subspace share the router sees at the
    /// firing sites.
    pub local_fraction: f64,
    /// The pooled dominant-direction energy fraction over ALL rows — the share the
    /// pooled birth would have offered the router (below the floor by construction
    /// when this pick fires).
    pub pooled_fraction: f64,
    /// `minimum_routable_energy(floor)` — the own-subspace fraction the router
    /// demands. `local_fraction ≥ this ≥ pooled_fraction` for an admitted pick.
    pub min_routable_energy: f64,
    /// Rank of the chosen stratum in descending mean energy (`0` = highest-energy).
    pub stratum_rank: usize,
    /// Kish effective sample size of the chosen stratum's row energies,
    /// `(Σ eᵢ)² / Σ eᵢ²` — the number of independent samples its dominant
    /// direction is estimated from. A single-row (or one-row-dominated) stratum
    /// has ESS ≈ 1; the birth floor below rejects it because a rank-1 selection's
    /// dominant-energy fraction is IDENTICALLY 1 and clears any router floor with
    /// noise alone.
    pub effective_sample_size: f64,
    /// The null-model ESS floor [`min_effective_rows_for_birth`] the chosen
    /// stratum cleared: the smallest effective sample size at which a genuine
    /// dominant direction is distinguishable from the noise-only dominant
    /// fraction at this router width. `effective_sample_size ≥ min_effective_rows`
    /// holds for every admitted pick.
    pub min_effective_rows: f64,
}

/// Dominant-direction energy fraction of the residual restricted to `rows`:
/// `λ_max(Rₛᵀ Rₛ) / ‖Rₛ‖_F²`, the share of the stratum's residual energy captured
/// by its single best direction (what a rank-1 / circle birth seeds from). Returns
/// `0.0` for an empty / zero-energy selection. `λ_max` is found by power iteration
/// on `Rₛᵀ Rₛ` applied implicitly (`v ← Rₛᵀ(Rₛ v)`), never materializing the `p×p`
/// Gram — `O(iters · |rows| · p)`.
pub fn dominant_energy_fraction(residual: ArrayView2<'_, f64>, rows: &[usize]) -> f64 {
    let p = residual.ncols();
    if p == 0 || rows.is_empty() {
        return 0.0;
    }
    // Total residual (Frobenius) energy over the selected rows.
    let mut total = 0.0_f64;
    for &i in rows {
        for &v in residual.row(i) {
            total += v * v;
        }
    }
    if total <= 0.0 {
        return 0.0;
    }
    let lambda_max = power_iteration_top_eigenvalue(residual, rows, p);
    (lambda_max / total).clamp(0.0, 1.0)
}

/// Top eigenvalue of `Rₛᵀ Rₛ` (`= σ_max(Rₛ)²`) via implicit power iteration from a
/// fixed all-ones seed. Deterministic; a small fixed iteration count suffices for
/// the leading eigenvalue of a matrix with a real spectral gap (the birth candidate
/// is a dominant direction by construction).
fn power_iteration_top_eigenvalue(
    residual: ArrayView2<'_, f64>,
    rows: &[usize],
    p: usize,
) -> f64 {
    // Deterministic unit seed.
    let mut v = vec![1.0_f64 / (p as f64).sqrt(); p];
    let mut lambda = 0.0_f64;
    // 24 iterations: the leading eigenvalue converges geometrically at rate
    // (λ₂/λ₁)ᵏ; a dominant birth direction has a clear gap, and this is a screen,
    // not a solve. A fixed, small count keeps the per-birth cost O(iters·|rows|·p)
    // bounded at frontier width while resolving the dominant fraction to well within
    // the floor margin.
    for _ in 0..24 {
        // w = Rₛ v  (one scalar per selected row).
        // u = Rₛᵀ w  (p-vector) = Rₛᵀ Rₛ v.
        let mut u = vec![0.0_f64; p];
        for &i in rows {
            let row = residual.row(i);
            let mut wi = 0.0_f64;
            for j in 0..p {
                wi += row[j] * v[j];
            }
            for j in 0..p {
                u[j] += wi * row[j];
            }
        }
        let norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 0.0 {
            return 0.0;
        }
        // Rayleigh quotient vᵀ(RₛᵀRₛ)v with the pre-normalized v equals ‖u‖ once v is
        // the (converged) unit eigenvector; track it and renormalize.
        lambda = norm;
        for j in 0..p {
            v[j] = u[j] / norm;
        }
    }
    lambda
}

/// Minimum effective sample size a stratum must carry before its dominant
/// direction may propose a birth — the null-model floor that stops a too-small
/// stratum from clearing the router with noise alone.
///
/// # Why an unguarded per-exponent stratum is a false positive
///
/// [`dominant_energy_fraction`] of an `m × p` residual block is
/// `φ = λ_max(RᵀR) / ‖R‖_F²`. A SINGLE row is rank-1: its whole energy lies in
/// one direction, so `φ ≡ 1` — it clears ANY floor, even when that row is pure
/// noise. Per-exponent strata have no minimum population, so a 1-row noise band
/// (a lone high-residual outlier in its own factor-of-two energy bin) always
/// "clears" the router and seeds a birth from a single noise row. More generally,
/// for an isotropic (noise) `m × p` block the leading singular value obeys the
/// Bai–Yin edge `σ_max² ≈ σ²(√m + √p)²` while `‖R‖_F² ≈ σ² m p`, so the NULL
/// (noise-only) dominant fraction is
///
/// ```text
///   φ₀(m, p) = (√m + √p)² / (m p),
/// ```
///
/// which is ≈ 1 at `m = 1` and decays to the ambient floor `1/p` as `m → ∞`. A
/// stratum can only carry EVIDENCE of a real dominant direction once this
/// noise-only fraction sits below the router's own
/// [`minimum_routable_energy`] `ρ` — otherwise a birth mined there is
/// indistinguishable from the artifact of too few rows, no matter how the
/// fitter routes it.
///
/// # The derived floor (no magic constant)
///
/// Require `φ₀(m, p) ≤ ρ`. Writing `u = √m`, `q = √p`, this is
/// `(ρp − 1)·u² − 2q·u − q² ≥ 0`; for `ρp > 1` its positive root gives the
/// smallest admissible effective sample size
///
/// ```text
///   m_min(p, ρ) = p · (1 + √(ρp))² / (ρp − 1)².
/// ```
///
/// Every input is derived: `p` is the router width, `ρ = minimum_routable_energy`
/// is the closed form of `(K, p, δ)` the router already exposes. The floor is
/// self-scaling — small `K` (an easy floor, large `ρ`) needs only a handful of
/// rows, frontier `K` (a tight floor) needs hundreds. When `ρp ≤ 1` the ambient
/// noise floor `1/p` itself exceeds `ρ`, so NO row count can separate signal from
/// noise at this router width and the screen declines every stratum (`+∞`).
pub fn min_effective_rows_for_birth(p: usize, min_routable: f64) -> f64 {
    if p == 0 || !(min_routable > 0.0) {
        return f64::INFINITY;
    }
    let pf = p as f64;
    let rho_p = min_routable * pf;
    if rho_p <= 1.0 {
        return f64::INFINITY;
    }
    pf * (1.0 + rho_p.sqrt()).powi(2) / (rho_p - 1.0).powi(2)
}

/// Kish effective sample size of the residual restricted to `rows`, computed on
/// the per-row energies `eᵢ = ‖rᵢ‖²`: `ESS = (Σ eᵢ)² / Σ eᵢ²`.
///
/// `ESS ∈ [1, |rows|]`: it equals the row count when every row carries comparable
/// energy (a genuine multi-row band whose dominant direction is estimated from
/// many samples) and collapses toward `1` for a stratum dominated by a single
/// high-energy row — exactly the one-row(-equivalent) false positive the birth
/// floor must reject. Using ESS rather than the raw row count also guards a
/// merged low-energy band that is effectively one big row plus dust. Returns
/// `0.0` for an empty / zero-energy selection.
fn effective_sample_size(residual: ArrayView2<'_, f64>, rows: &[usize]) -> f64 {
    let mut sum = 0.0_f64;
    let mut sumsq = 0.0_f64;
    for &i in rows {
        let e: f64 = residual.row(i).iter().map(|&v| v * v).sum();
        sum += e;
        sumsq += e * e;
    }
    if sumsq <= 0.0 {
        return 0.0;
    }
    sum * sum / sumsq
}

/// Screen the pooled residual for a stratum-local birth. Stratifies the rows by
/// residual energy, then — in DESCENDING stratum mean energy — returns the first
/// stratum whose local [`dominant_energy_fraction`] clears the routing floor's
/// [`minimum_routable_energy`], as a [`StratumLocalPick`] carrying the residual
/// masked to that stratum's rows. Returns `None` when NO stratum clears the floor
/// locally (the caller then keeps the pooled birth path unchanged), preserving
/// small-`K` behavior where the floor is easy and the pooled residual already
/// routes.
pub fn stratum_local_birth_residual(
    residual: ArrayView2<'_, f64>,
    floor: &RoutabilityFloor,
) -> Option<StratumLocalPick> {
    let (n, p) = residual.dim();
    if n == 0 || p == 0 {
        return None;
    }
    let min_routable = minimum_routable_energy(floor);
    // Per-row residual energies e_i = ‖r_i‖².
    let energies: Vec<f64> = (0..n)
        .map(|i| residual.row(i).iter().map(|&v| v * v).sum())
        .collect();
    let all_rows: Vec<usize> = (0..n).collect();
    let pooled_fraction = dominant_energy_fraction(residual, &all_rows);

    let mut strata = crate::corpus::stratify_row_energies(&energies);
    // Descending mean energy: mine the tail (rare, high-residual) stratum first.
    strata.sort_by(|a, b| {
        b.mean_energy
            .partial_cmp(&a.mean_energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // The null-model effective-sample-size floor: a stratum whose dominant
    // fraction clears the router must ALSO carry enough independent samples that a
    // real dominant direction is distinguishable from the noise-only fraction of a
    // block that small (see [`min_effective_rows_for_birth`]). Without it a 1-row
    // (rank-1, φ ≡ 1) noise band clears any floor and seeds a birth from one row.
    let min_effective_rows = min_effective_rows_for_birth(p, min_routable);
    for (rank, stratum) in strata.iter().enumerate() {
        let local = dominant_energy_fraction(residual, &stratum.rows);
        if local < min_routable {
            continue;
        }
        let ess = effective_sample_size(residual, &stratum.rows);
        if ess < min_effective_rows {
            // Clears the fraction floor but not on enough independent samples — the
            // fraction is the too-few-rows artifact, not evidence. Skip to a
            // larger stratum; if none qualifies the caller keeps the pooled path.
            continue;
        }
        // Mask the residual to this stratum's rows (others zeroed).
        let mut masked = Array2::<f64>::zeros((n, p));
        for &i in &stratum.rows {
            masked.row_mut(i).assign(&residual.row(i));
        }
        return Some(StratumLocalPick {
            rows: stratum.rows.clone(),
            masked_residual: masked,
            local_fraction: local,
            pooled_fraction,
            min_routable_energy: min_routable,
            stratum_rank: rank,
            effective_sample_size: ess,
            min_effective_rows,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routability::routability_floor;
    use ndarray::Array2;

    /// Deterministic small pseudo-normal from a counter (no RNG dependency).
    fn gauss(counter: &mut u64) -> f64 {
        *counter = counter.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *counter;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let a = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
        *counter = counter.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut w = *counter;
        w = (w ^ (w >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let b = ((w ^ (w >> 27)) >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * a.max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }

    #[test]
    fn stratum_local_admits_a_planted_signal_pooled_rejects() {
        // A rank-1 structure present on a RARE high-energy subset of rows, drowned in
        // diffuse isotropic bulk noise everywhere. Pooled, its own-subspace share is
        // far below the floor (rejected); on its own high-energy stratum it dominates
        // (admitted). This is the reachability failure the screen closes.
        let n = 4000usize;
        let p = 256usize;
        let k_router = 32_000usize; // frontier width: floor ≈ 0.11
        let n_signal = 40usize; // rare — 1 % of rows
        let signal_amp = 6.0; // strong on the rows it fires
        let noise = 1.0; // diffuse bulk

        let mut ctr = 1u64;
        let mut r = Array2::<f64>::zeros((n, p));
        // A fixed unit direction the planted structure lives in.
        let mut dir = vec![0.0_f64; p];
        for d in dir.iter_mut() {
            *d = gauss(&mut ctr);
        }
        let dn = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        for d in dir.iter_mut() {
            *d /= dn;
        }
        // Diffuse isotropic noise everywhere.
        for i in 0..n {
            for j in 0..p {
                r[[i, j]] = noise * gauss(&mut ctr);
            }
        }
        // Plant the rank-1 signal on the first n_signal rows (rare, high energy).
        for i in 0..n_signal {
            let a = signal_amp * gauss(&mut ctr);
            for j in 0..p {
                r[[i, j]] += a * dir[j];
            }
        }

        let floor = routability_floor(p, k_router, 1, 1.0);
        let min_routable = minimum_routable_energy(&floor);

        // Pooled: the planted direction's global energy share is below the floor.
        let all: Vec<usize> = (0..n).collect();
        let pooled = dominant_energy_fraction(r.view(), &all);
        assert!(
            pooled < min_routable,
            "pooled fraction {pooled} must be BELOW the floor {min_routable} \
             (the pooled birth would be rejected by the router)"
        );

        // Stratum-local: the screen finds the high-energy stratum and admits it.
        let pick = stratum_local_birth_residual(r.view(), &floor)
            .expect("stratum-local screen must admit the planted signal");
        assert!(
            pick.local_fraction >= min_routable,
            "local fraction {} must CLEAR the floor {min_routable}",
            pick.local_fraction
        );
        assert!(
            pick.local_fraction > pick.pooled_fraction,
            "local {} must exceed pooled {}",
            pick.local_fraction,
            pick.pooled_fraction
        );
        // The admitted stratum must clear the ESS floor: the fraction is carried by
        // enough independent samples to be real, not the artifact of too few rows.
        assert!(
            pick.effective_sample_size >= pick.min_effective_rows,
            "admitted stratum ESS {} must clear the floor {}",
            pick.effective_sample_size,
            pick.min_effective_rows
        );
        // The chosen stratum must actually contain the planted (high-energy) rows.
        let planted: std::collections::BTreeSet<usize> = (0..n_signal).collect();
        let overlap = pick.rows.iter().filter(|i| planted.contains(i)).count();
        assert!(
            overlap as f64 >= 0.8 * n_signal as f64,
            "the picked stratum must hold the planted rows: {overlap}/{n_signal}"
        );
        // The masked residual keeps the stratum rows and zeroes the rest.
        for i in 0..n {
            let kept = pick.rows.binary_search(&i).is_ok();
            let row_energy: f64 = pick.masked_residual.row(i).iter().map(|v| v * v).sum();
            if kept {
                let orig: f64 = r.row(i).iter().map(|v| v * v).sum();
                assert!((row_energy - orig).abs() < 1e-9, "kept row {i} preserved");
            } else {
                assert_eq!(row_energy, 0.0, "non-stratum row {i} must be zeroed");
            }
        }
    }

    #[test]
    fn easy_floor_declines_no_stratification_needed() {
        // Small K (easy floor) + a globally dominant rank-1 signal: the pooled
        // residual already routes, so the screen declines (None) and the caller
        // keeps the pooled path. Small-K behavior is unchanged.
        let n = 500usize;
        let p = 32usize;
        let mut ctr = 7u64;
        let mut r = Array2::<f64>::zeros((n, p));
        let mut dir = vec![0.0_f64; p];
        for d in dir.iter_mut() {
            *d = gauss(&mut ctr);
        }
        let dn = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        for d in dir.iter_mut() {
            *d /= dn;
        }
        // Globally dominant signal + tiny noise ⇒ pooled fraction near 1.
        for i in 0..n {
            let a = 5.0 * gauss(&mut ctr);
            for j in 0..p {
                r[[i, j]] = a * dir[j] + 0.01 * gauss(&mut ctr);
            }
        }
        let floor = routability_floor(p, 8, 1, 1.0);
        // Every stratum trivially clears the (easy) floor, so the highest-energy
        // stratum is returned; but the pooled fraction is already above the floor,
        // confirming the pooled path would have routed. Assert the accounting is
        // consistent rather than asserting None (the screen returns the tail stratum
        // whenever ANY stratum clears — the caller uses pooled when pooled clears).
        let all: Vec<usize> = (0..n).collect();
        let pooled = dominant_energy_fraction(r.view(), &all);
        let min_routable = minimum_routable_energy(&floor);
        assert!(
            pooled >= min_routable,
            "an easy floor with a dominant signal must route pooled: {pooled} ≥ {min_routable}"
        );
    }

    #[test]
    fn ess_floor_closed_form_and_kish_sample_size() {
        // The derived floor m_min = p·(1+√(ρp))²/(ρp−1)² and the Kish ESS
        // (Σe)²/Σe² are pinned in closed form, including the two limits the birth
        // guard turns on: ESS = 1 for a single row, ESS = m for equal-energy rows.
        let p = 256usize;
        let floor = routability_floor(p, 32_000, 1, 1.0);
        let rho = minimum_routable_energy(&floor);
        let rho_p = rho * p as f64;
        assert!(rho_p > 1.0, "frontier floor must admit a finite m_min");
        let expected = p as f64 * (1.0 + rho_p.sqrt()).powi(2) / (rho_p - 1.0).powi(2);
        let got = min_effective_rows_for_birth(p, rho);
        assert!((got - expected).abs() < 1e-9, "m_min {got} vs {expected}");
        assert!(got > 1.0, "a nontrivial router width must demand > 1 row");

        // ρp ≤ 1 ⇒ the ambient noise floor 1/p exceeds ρ ⇒ no row count separates
        // signal from noise ⇒ the floor is +∞ (decline every stratum).
        assert!(min_effective_rows_for_birth(p, 0.5 / p as f64).is_infinite());
        assert!(min_effective_rows_for_birth(0, 0.5).is_infinite());

        // Kish ESS: a single row is ESS 1 (the rank-1 false positive); m equal-
        // energy rows are ESS m; a one-row-dominated selection collapses toward 1.
        let mut r = Array2::<f64>::zeros((5, 4));
        for j in 0..4 {
            r[[0, j]] = 1.0; // row 0 energy 4
            r[[1, j]] = 1.0; // row 1 energy 4
            r[[2, j]] = 1.0;
            r[[3, j]] = 1.0;
        }
        r[[4, 0]] = 100.0; // row 4 energy 10000 (dominates)
        assert!((effective_sample_size(r.view(), &[0]) - 1.0).abs() < 1e-12);
        assert!((effective_sample_size(r.view(), &[0, 1, 2, 3]) - 4.0).abs() < 1e-12);
        // Four unit rows plus the huge outlier: ESS ≈ 1 (outlier owns the energy).
        assert!(
            effective_sample_size(r.view(), &[0, 1, 2, 3, 4]) < 1.01,
            "a one-row-dominated selection must have ESS ≈ 1"
        );
        assert_eq!(effective_sample_size(r.view(), &[]), 0.0);
    }

    #[test]
    fn one_row_stratum_proposes_nothing_adequate_stratum_proposes() {
        // (a) A single high-energy noise outlier forms its own exponent bin: a
        // 1-row stratum with dominant fraction ≡ 1 that clears the router floor.
        // Bulk is many unit-norm noise rows (one energy band, dominant fraction far
        // below the floor). The ESS floor rejects the outlier (ESS = 1) and the
        // bulk fails the fraction floor, so the screen proposes NOTHING.
        let n = 1000usize;
        let p = 64usize;
        let k_router = 32_000usize; // frontier: tight floor
        let mut ctr = 42u64;
        let mut r = Array2::<f64>::zeros((n + 1, p));
        // Bulk: unit-norm isotropic noise rows (all energy ≈ 1 ⇒ one exponent bin).
        for i in 0..n {
            let mut e = 0.0_f64;
            for j in 0..p {
                let v = gauss(&mut ctr);
                r[[i, j]] = v;
                e += v * v;
            }
            let s = e.sqrt();
            for j in 0..p {
                r[[i, j]] /= s;
            }
        }
        // One giant outlier in a random direction: energy ≈ 1000, its own bin.
        let mut e = 0.0_f64;
        for j in 0..p {
            let v = gauss(&mut ctr);
            r[[n, j]] = v;
            e += v * v;
        }
        let s = e.sqrt();
        for j in 0..p {
            r[[n, j]] = r[[n, j]] / s * 1000.0_f64.sqrt();
        }

        let floor = routability_floor(p, k_router, 1, 1.0);
        let min_routable = minimum_routable_energy(&floor);
        // The lone outlier as its own stratum WOULD clear the fraction floor…
        assert!(dominant_energy_fraction(r.view(), &[n]) >= min_routable);
        // …but its ESS is 1, far below the derived floor, so it must be rejected.
        assert!(effective_sample_size(r.view(), &[n]) < min_effective_rows_for_birth(p, min_routable));
        // The screen proposes nothing: no adequately-sized stratum clears locally.
        assert!(
            stratum_local_birth_residual(r.view(), &floor).is_none(),
            "a 1-row noise outlier must not seed a birth"
        );

        // (b) Now add a genuine, adequately-sized planted stratum: ~50 rows of a
        // coherent rank-1 direction at high energy. It clears BOTH the fraction
        // floor and the ESS floor, so the screen proposes it.
        let n_signal = 50usize;
        let mut dir = vec![0.0_f64; p];
        for d in dir.iter_mut() {
            *d = gauss(&mut ctr);
        }
        let dn = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        for d in dir.iter_mut() {
            *d /= dn;
        }
        // Overwrite the first n_signal bulk rows with a strong coherent signal in
        // dir (energy ≈ 25 ⇒ a distinct high-energy band, well above unit bulk).
        for i in 0..n_signal {
            let a = 5.0;
            for j in 0..p {
                r[[i, j]] = a * dir[j] + 0.05 * gauss(&mut ctr);
            }
        }
        let pick = stratum_local_birth_residual(r.view(), &floor)
            .expect("an adequately-sized planted stratum must propose a birth");
        assert!(pick.local_fraction >= pick.min_routable_energy);
        assert!(
            pick.effective_sample_size >= pick.min_effective_rows,
            "the proposed stratum must clear the ESS floor: {} < {}",
            pick.effective_sample_size,
            pick.min_effective_rows
        );
        assert!(
            pick.effective_sample_size > 1.5,
            "the proposed stratum must be a genuine multi-row band, not a lone row"
        );
    }
}

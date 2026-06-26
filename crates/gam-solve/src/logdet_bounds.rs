//! #1011 — deterministic two-sided enclosures for a block-SPD log-determinant.
//!
//! For the bordered-arrow evidence at frontier atom counts, the dense border
//! Schur factor is the scaling wall. This module computes CERTIFIED bounds
//! `lower ≤ log|S| − log|D| ≤ upper` from exact moments of the
//! block-preconditioned residual, with no randomness and no estimator
//! variance — an enclosure, refinable until it is tighter than the consuming
//! decision's margin (topology-race Δ, EFS step tolerance, …).
//!
//! Math (derivation on issue #1011): with `D = blockdiag(S_11..S_KK)`,
//! `S_ii = L_i L_iᵀ`, and `E = D^{-1/2}(S − D)D^{-1/2}`:
//! * `I + E = D^{-1/2} S D^{-1/2} ≻ 0` ⇒ every eigenvalue `λ_a > −1`;
//! * `E` has ZERO diagonal blocks ⇒ `tr E = Σ λ_a = 0`;
//! * `p₂ = tr E² = Σ_{i≠j} ‖Ẽ_ij‖_F²` and
//!   `p₃ = tr E³ = Σ_{i≠j≠k≠i} tr(Ẽ_ij Ẽ_jk Ẽ_ki)` are EXACT block
//!   contractions of `Ẽ_ij = L_i⁻¹ S_ij L_j⁻ᵀ` — never forming `E` densely;
//! * a spectral-radius certificate `ρ = min(√p₂, max_i Σ_{j≠i} ‖Ẽ_ij‖_F)`
//!   (block Gershgorin via `‖·‖₂ ≤ ‖·‖_F`), required `< 1`.
//!
//! Per-eigenvalue inequalities, valid for ALL `λ > −1` (alternating-series
//! remainder for `λ ≥ 0`, monotone tail for `λ < 0`):
//! `log(1+λ) ≤ λ − λ²/2 + λ³/3`, and on `[−ρ, ρ]` the cubic remainder obeys
//! `R(λ) ≥ −ρ²λ²/(4(1−ρ))`. Summing with `Σλ = 0`:
//!
//! ```text
//! order 3:  upper = −p₂/2 + p₃/3
//!           lower = upper − ρ²·p₂ / (4(1−ρ))
//! order 2:  upper = −p₂/2 + ρ·p₂/3          (λ³ ≤ ρλ² for λ≥0; λ³<0≤ρλ² else)
//!           lower = −p₂/2 − ρ·p₂ / (3(1−ρ))
//! ```
//!
//! The gap scales as `ρ·p₂` (order 2) / `ρ²·p₂` (order 3): preconditioner
//! quality drives certainty, and absorbing the worst off-diagonal pair into
//! `D` is the refinement step when the gap is too wide. `ρ ≥ 1` is an
//! explicit refusal (`Err`), never a silent fallback.

use faer::Side;
use ndarray::Array2;

use gam_linalg::faer_ndarray::FaerCholesky;
use gam_linalg::triangular::forward_substitution_lower_matrix;

/// A certified enclosure of `log|S|` for a block-partitioned SPD matrix.
#[derive(Debug, Clone)]
pub struct LogdetEnclosure {
    /// Exact `log|D| = Σ_i log|S_ii|` from the per-block Cholesky factors.
    pub block_diag_logdet: f64,
    /// Certified lower bound on `log|S|` (i.e. `block_diag_logdet + correction_lower`).
    pub lower: f64,
    /// Certified upper bound on `log|S|`.
    pub upper: f64,
    /// The spectral-radius certificate used (`< 1` or this struct would not exist).
    pub rho: f64,
    /// Exact second moment `tr(E²)`.
    pub p2: f64,
    /// Exact third moment `tr(E³)` when the order-3 enclosure was requested.
    pub p3: Option<f64>,
}

impl LogdetEnclosure {
    /// Width of the enclosure — compare against the consuming decision's margin.
    pub fn gap(&self) -> f64 {
        self.upper - self.lower
    }

    /// The enclosure's certified point value when (and only when) the gap is
    /// already below the consuming decision's margin. Below margin a single
    /// `f64` is a lie — the caller must escalate — so this returns the explicit
    /// [`MarginVerdict`] rather than ever fabricating one.
    ///
    /// `decision_margin` is the smallest spread in the consumer's verdict that
    /// matters: the topology-race candidate gap Δ, the EFS step tolerance, an
    /// Armijo slack. A `Decided` value is the enclosure midpoint, which is
    /// within `gap/2 ≤ margin/2` of the truth — tighter than the decision can
    /// resolve, so the verdict is identical to the one the exact logdet would
    /// have produced.
    /// Whether a bare enclosure `gap` is resolved more tightly than a consumer's
    /// `decision_margin` — the predicate behind [`Self::decide_within_margin`],
    /// exposed for consumers that hold only the gap (e.g. the EFS engine, which
    /// receives the cost's enclosure width through `EfsEval`).
    pub fn gap_resolves_margin(gap: f64, decision_margin: f64) -> bool {
        decision_margin.is_finite()
            && decision_margin > 0.0
            && gap.is_finite()
            && gap < decision_margin
    }

    pub fn decide_within_margin(&self, decision_margin: f64) -> MarginVerdict {
        let gap = self.gap();
        if decision_margin.is_finite() && decision_margin > 0.0 && gap < decision_margin {
            MarginVerdict::Decided {
                value: 0.5 * (self.lower + self.upper),
                gap,
                decision_margin,
            }
        } else {
            MarginVerdict::InsufficientMargin {
                gap,
                decision_margin,
            }
        }
    }
}

/// The shared decision-margin contract between an enclosure-valued quantity and
/// its consumer (the topology race, the EFS outer step, the coreset race
/// transfer — all declare a margin and inherit this verdict).
///
/// `Decided` means the enclosure is strictly tighter than the consumer's
/// decision margin, so its midpoint is interchangeable with the exact value for
/// that decision. `InsufficientMargin` is the honesty escalation: the consumer
/// must refine (more moments, pair absorption, a larger coreset) or fall back
/// to the exact dense path — never decide on a point value that the enclosure
/// does not actually pin down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarginVerdict {
    Decided {
        value: f64,
        gap: f64,
        decision_margin: f64,
    },
    InsufficientMargin {
        gap: f64,
        decision_margin: f64,
    },
}

impl MarginVerdict {
    /// The certified point value when the margin closed, else `None` — the
    /// caller that pattern-matches `None` must escalate to the exact path.
    pub fn decided_value(&self) -> Option<f64> {
        match self {
            MarginVerdict::Decided { value, .. } => Some(*value),
            MarginVerdict::InsufficientMargin { .. } => None,
        }
    }

    pub fn is_decided(&self) -> bool {
        matches!(self, MarginVerdict::Decided { .. })
    }
}

/// Refine a block-preconditioned log-determinant enclosure until its gap falls
/// below `decision_margin`, climbing the deterministic ladder:
///
/// 1. order-2 moments (`tr E²` only) — gap `∝ ρ·p₂`;
/// 2. order-3 moments (`+ tr E³`)   — gap `∝ ρ²·p₂`;
/// 3. **pair absorption**: merge the strongest off-diagonal pair `(i, j)` (the
///    largest `‖Ẽ_ij‖_F`, the term that dominates `ρ`) into a single joint
///    diagonal block, shrinking the residual `E` and `ρ`, then recurse.
///
/// Each rung is an EXACT enclosure (it always contains the truth); the ladder
/// only tightens the gap. When absorption can no longer split a pair (the whole
/// matrix has collapsed to one block, i.e. the exact dense logdet) or the
/// `max_absorptions` budget is spent without closing the margin, the result is
/// [`MarginVerdict::InsufficientMargin`] — the consumer then takes the exact
/// dense path. `decision_margin` is the consumer's declared margin.
pub fn refine_logdet_enclosure_to_margin(
    diag: &[Array2<f64>],
    off: &[(usize, usize, Array2<f64>)],
    decision_margin: f64,
    max_absorptions: usize,
) -> Result<(LogdetEnclosure, MarginVerdict), String> {
    // Rung 1 & 2: moments at the current partition.
    let order2 = block_preconditioned_logdet_enclosure(diag, off, false)?;
    if order2.decide_within_margin(decision_margin).is_decided() {
        let verdict = order2.decide_within_margin(decision_margin);
        return Ok((order2, verdict));
    }
    let order3 = block_preconditioned_logdet_enclosure(diag, off, true)?;
    let verdict3 = order3.decide_within_margin(decision_margin);
    if verdict3.is_decided() {
        return Ok((order3, verdict3));
    }

    // Rung 3: absorb the strongest off-diagonal pair and recurse. Each call to
    // `block_preconditioned_logdet_enclosure` is independent and exact, so the
    // recursion only ever sharpens; the worst it can do is reach the single
    // dense block (the exact logdet, gap 0) when the budget allows.
    if max_absorptions == 0 || off.is_empty() {
        return Ok((order3, verdict3));
    }
    let (merged_diag, merged_off) = absorb_strongest_pair(diag, off)?;
    // Absorption strictly reduces the block count, so the recursion terminates.
    refine_logdet_enclosure_to_margin(
        &merged_diag,
        &merged_off,
        decision_margin,
        max_absorptions - 1,
    )
}

/// Merge the two atoms joined by the strongest off-diagonal block (largest
/// `‖S_ij‖_F` in the *whitened* metric, the term that dominates the spectral
/// radius) into a single joint diagonal block, returning the smaller partition.
///
/// The new joint block stacks the two diagonals and the coupling on its
/// off-diagonal; every other block is re-indexed into the contracted layout.
/// The dense matrix `S` is unchanged — only the *partition* coarsens — so the
/// resulting enclosure is over the same `log|S|`, just with a smaller residual
/// `E` (the absorbed pair contributes 0 to the new `E`).
fn absorb_strongest_pair(
    diag: &[Array2<f64>],
    off: &[(usize, usize, Array2<f64>)],
) -> Result<(Vec<Array2<f64>>, Vec<(usize, usize, Array2<f64>)>), String> {
    let k = diag.len();
    // Pick the pair with the largest whitened Frobenius coupling. We whiten by
    // the per-block Cholesky factors so the ranking matches the contribution to
    // ρ, not the raw (un-preconditioned) block norm.
    let mut lowers: Vec<Array2<f64>> = Vec::with_capacity(k);
    for (i, s_ii) in diag.iter().enumerate() {
        let factor = s_ii
            .cholesky(Side::Lower)
            .map_err(|e| format!("absorb_strongest_pair: block {i} is not SPD: {e:?}"))?;
        lowers.push(factor.lower_triangular());
    }
    let mut best: Option<(usize, f64)> = None; // (index into off, whitened ‖·‖_F)
    for (idx, (i, j, s_ij)) in off.iter().enumerate() {
        let e_ij = whitened_off_block(&lowers[*i], &lowers[*j], s_ij);
        let f = frobenius_sq(&e_ij).sqrt();
        match best {
            None => best = Some((idx, f)),
            Some((_, bf)) if f > bf => best = Some((idx, f)),
            _ => {}
        }
    }
    let (best_idx, _) =
        best.ok_or_else(|| "absorb_strongest_pair: no off-diagonal pair to absorb".to_string())?;
    let (a, b, _) = &off[best_idx];
    let (a, b) = (*a.min(b), *a.max(b));
    let (ma, mb) = (diag[a].nrows(), diag[b].nrows());
    let merged_dim = ma + mb;

    // Assemble the joint diagonal block [[S_aa, S_ab], [S_abᵀ, S_bb]].
    let mut joint = Array2::<f64>::zeros((merged_dim, merged_dim));
    joint.slice_mut(ndarray::s![..ma, ..ma]).assign(&diag[a]);
    joint.slice_mut(ndarray::s![ma.., ma..]).assign(&diag[b]);
    if let Some((_, _, s_ab)) = off.iter().find(|(i, j, _)| (*i, *j) == (a, b)) {
        joint.slice_mut(ndarray::s![..ma, ma..]).assign(s_ab);
        let s_ba = s_ab.t().to_owned();
        joint.slice_mut(ndarray::s![ma.., ..ma]).assign(&s_ba);
    }

    // New block ordering: the joint block takes index 0, every other old block
    // `g ∉ {a, b}` keeps its relative order at a shifted index.
    let mut old_to_new = vec![0usize; k];
    let mut new_diag: Vec<Array2<f64>> = Vec::with_capacity(k - 1);
    new_diag.push(joint);
    let mut next = 1usize;
    for g in 0..k {
        if g == a || g == b {
            old_to_new[g] = 0;
        } else {
            old_to_new[g] = next;
            new_diag.push(diag[g].clone());
            next += 1;
        }
    }

    // Re-index the surviving off-diagonal blocks. A block touching `a` or `b`
    // becomes a (joint, g) coupling; we must place its rows into the correct
    // half of the joint block's row span. Blocks between two survivors carry
    // over unchanged (up to re-index). The absorbed (a, b) pair is dropped — it
    // now lives inside the joint diagonal block.
    let mut new_off: Vec<(usize, usize, Array2<f64>)> = Vec::new();
    // Accumulate joint↔survivor couplings keyed by the survivor's NEW index, so
    // the `a`- and `b`-halves of a survivor that couples to BOTH are stacked
    // into one (merged_dim × m_g) block.
    use std::collections::BTreeMap;
    let mut joint_couplings: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
    for (i, j, s_ij) in off {
        let (i, j) = (*i, *j);
        if (i, j) == (a, b) {
            continue; // absorbed into the joint diagonal
        }
        let touches_a = i == a || j == a;
        let touches_b = i == b || j == b;
        if touches_a || touches_b {
            // Identify the survivor `g` and the oriented block S_{joint-half, g}.
            let g = if i == a || i == b { j } else { i };
            let mg = diag[g].nrows();
            let entry = joint_couplings
                .entry(old_to_new[g])
                .or_insert_with(|| Array2::<f64>::zeros((merged_dim, mg)));
            // Orient the stored block so its rows index the absorbed atom and
            // columns index `g` (S_{absorbed, g}); transpose when the stored
            // upper-triangle entry has `g` first.
            let s_half = if i == g {
                s_ij.t().to_owned()
            } else {
                s_ij.clone()
            };
            let row_off = if i == a || j == a { 0 } else { ma };
            entry
                .slice_mut(ndarray::s![row_off..row_off + s_half.nrows(), ..])
                .assign(&s_half);
        } else {
            // Survivor–survivor block: re-index, preserving i<j orientation.
            let (ni, nj) = (old_to_new[i], old_to_new[j]);
            let (ni, nj, block) = if ni < nj {
                (ni, nj, s_ij.clone())
            } else {
                (nj, ni, s_ij.t().to_owned())
            };
            new_off.push((ni, nj, block));
        }
    }
    for (g_new, block) in joint_couplings {
        // joint index is 0 < g_new always.
        new_off.push((0, g_new, block));
    }
    Ok((new_diag, new_off))
}

/// `Ẽ_ij = L_i⁻¹ · S_ij · L_j⁻ᵀ`: forward-solve on the left, then on the
/// right via the transpose identity `(X L_j⁻ᵀ)ᵀ = L_j⁻¹ Xᵀ`.
fn whitened_off_block(l_i: &Array2<f64>, l_j: &Array2<f64>, s_ij: &Array2<f64>) -> Array2<f64> {
    let x = forward_substitution_lower_matrix(l_i, s_ij);
    let xt = x.t().to_owned();
    forward_substitution_lower_matrix(l_j, &xt).t().to_owned()
}

fn frobenius_sq(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum()
}

/// Certified two-sided enclosure of `log|S|` for a block-SPD matrix given as
/// per-atom diagonal blocks plus upper-triangle off-diagonal blocks
/// (`off[(i, j)]` with `i < j`; `S_ji = S_ijᵀ` by symmetry). Pass
/// `use_third_moment = true` for the order-3 enclosure (extra `O(triples)`
/// work, gap `∝ ρ²p₂` instead of `∝ ρp₂`).
///
/// Errors when a diagonal block is not SPD or when the spectral-radius
/// certificate fails (`ρ ≥ 1`): the caller must refine the partition
/// (absorb the offending pair into one joint diagonal block) — the bound
/// machinery never silently degrades.
pub fn block_preconditioned_logdet_enclosure(
    diag: &[Array2<f64>],
    off: &[(usize, usize, Array2<f64>)],
    use_third_moment: bool,
) -> Result<LogdetEnclosure, String> {
    let k = diag.len();
    if k == 0 {
        return Err("block_preconditioned_logdet_enclosure: no diagonal blocks".to_string());
    }
    // Exact per-block factors and log|D|.
    let mut lowers: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut block_diag_logdet = 0.0_f64;
    for (i, s_ii) in diag.iter().enumerate() {
        if s_ii.nrows() != s_ii.ncols() {
            return Err(format!(
                "block_preconditioned_logdet_enclosure: diagonal block {i} is not square"
            ));
        }
        let factor = s_ii.cholesky(Side::Lower).map_err(|e| {
            format!("block_preconditioned_logdet_enclosure: block {i} is not SPD: {e:?}")
        })?;
        let l = factor.lower_triangular();
        for d in 0..l.nrows() {
            block_diag_logdet += 2.0 * l[[d, d]].ln();
        }
        lowers.push(l);
    }
    // Whitened off-diagonal blocks (upper triangle), p₂, and block-Gershgorin
    // row sums.
    let mut whitened: Vec<(usize, usize, Array2<f64>)> = Vec::with_capacity(off.len());
    let mut p2 = 0.0_f64;
    let mut row_sums = vec![0.0_f64; k];
    for (i, j, s_ij) in off {
        let (i, j) = (*i, *j);
        if i >= j || j >= k {
            return Err(format!(
                "block_preconditioned_logdet_enclosure: off-block ({i},{j}) must satisfy i<j<K={k}"
            ));
        }
        if s_ij.nrows() != lowers[i].nrows() || s_ij.ncols() != lowers[j].nrows() {
            return Err(format!(
                "block_preconditioned_logdet_enclosure: off-block ({i},{j}) shape mismatch"
            ));
        }
        let e_ij = whitened_off_block(&lowers[i], &lowers[j], s_ij);
        let f2 = frobenius_sq(&e_ij);
        // E_ij and its transpose E_ji contribute equally to tr(E²).
        p2 += 2.0 * f2;
        let f = f2.sqrt();
        row_sums[i] += f;
        row_sums[j] += f;
        whitened.push((i, j, e_ij));
    }
    let gershgorin = row_sums.iter().fold(0.0_f64, |a, &b| a.max(b));
    let rho = p2.sqrt().min(gershgorin);
    if !(rho < 1.0) {
        return Err(format!(
            "block_preconditioned_logdet_enclosure: spectral-radius certificate failed \
             (ρ = {rho:.6} ≥ 1); refine the block partition (absorb the strongest \
             off-diagonal pair into the preconditioner) and retry"
        ));
    }

    // Optional exact third moment over ordered distinct triples. Lookup
    // returns the (a,b) whitened block, transposing the stored upper-triangle
    // entry when needed.
    let p3 = if use_third_moment {
        let get = |a: usize, b: usize| -> Option<Array2<f64>> {
            for (i, j, e) in &whitened {
                if *i == a && *j == b {
                    return Some(e.clone());
                }
                if *i == b && *j == a {
                    return Some(e.t().to_owned());
                }
            }
            None
        };
        let mut acc = 0.0_f64;
        for a in 0..k {
            for b in 0..k {
                if b == a {
                    continue;
                }
                let Some(e_ab) = get(a, b) else { continue };
                for c in 0..k {
                    if c == a || c == b {
                        continue;
                    }
                    let (Some(e_bc), Some(e_ca)) = (get(b, c), get(c, a)) else {
                        continue;
                    };
                    // tr(E_ab · E_bc · E_ca)
                    let prod = e_ab.dot(&e_bc);
                    for r in 0..prod.nrows() {
                        for s in 0..prod.ncols() {
                            acc += prod[[r, s]] * e_ca[[s, r]];
                        }
                    }
                }
            }
        }
        Some(acc)
    } else {
        None
    };

    let (corr_lower, corr_upper) = match p3 {
        Some(p3) => {
            let upper = -p2 / 2.0 + p3 / 3.0;
            let lower = upper - rho * rho * p2 / (4.0 * (1.0 - rho));
            (lower, upper)
        }
        None => {
            let upper = -p2 / 2.0 + rho * p2 / 3.0;
            let lower = -p2 / 2.0 - rho * p2 / (3.0 * (1.0 - rho));
            (lower, upper)
        }
    };
    Ok(LogdetEnclosure {
        block_diag_logdet,
        lower: block_diag_logdet + corr_lower,
        upper: block_diag_logdet + corr_upper,
        rho,
        p2,
        p3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic block-SPD fixture: strong SPD diagonal blocks, weak
    /// off-diagonal coupling (so the certificate ρ < 1 holds), assembled
    /// densely for the oracle.
    fn fixture(
        k: usize,
        m: usize,
        coupling: f64,
    ) -> (
        Vec<Array2<f64>>,
        Vec<(usize, usize, Array2<f64>)>,
        Array2<f64>,
    ) {
        let dim = k * m;
        let mut dense = Array2::<f64>::zeros((dim, dim));
        let mut diag = Vec::new();
        let mut off = Vec::new();
        for i in 0..k {
            let mut d = Array2::<f64>::zeros((m, m));
            for r in 0..m {
                for c in 0..m {
                    let v = if r == c {
                        3.0 + 0.4 * (i as f64) + 0.2 * (r as f64)
                    } else {
                        0.3 * ((r + 2 * c + i) as f64 * 0.7).sin()
                    };
                    d[[r, c]] = v;
                }
            }
            // Symmetrize and make diagonally dominant ⇒ SPD.
            let mut sym = Array2::<f64>::zeros((m, m));
            for r in 0..m {
                for c in 0..m {
                    sym[[r, c]] = 0.5 * (d[[r, c]] + d[[c, r]]);
                }
                sym[[r, r]] += 1.0;
            }
            for r in 0..m {
                for c in 0..m {
                    dense[[i * m + r, i * m + c]] = sym[[r, c]];
                }
            }
            diag.push(sym);
        }
        for i in 0..k {
            for j in (i + 1)..k {
                let mut b = Array2::<f64>::zeros((m, m));
                for r in 0..m {
                    for c in 0..m {
                        b[[r, c]] =
                            coupling * ((r as f64) - (c as f64) + (i + j) as f64 * 0.31).cos();
                    }
                }
                for r in 0..m {
                    for c in 0..m {
                        dense[[i * m + r, j * m + c]] = b[[r, c]];
                        dense[[j * m + c, i * m + r]] = b[[r, c]];
                    }
                }
                off.push((i, j, b));
            }
        }
        (diag, off, dense)
    }

    fn dense_logdet(s: &Array2<f64>) -> f64 {
        let l = s
            .cholesky(Side::Lower)
            .expect("oracle fixture must be SPD")
            .lower_triangular();
        (0..l.nrows()).map(|d| 2.0 * l[[d, d]].ln()).sum()
    }

    /// Containment: the enclosure must contain the dense truth at both
    /// orders, and the order-3 gap must not exceed the order-2 gap.
    #[test]
    fn enclosure_contains_dense_truth_and_order3_tightens() {
        let (diag, off, dense) = fixture(4, 3, 0.08);
        let truth = dense_logdet(&dense);
        let e2 =
            block_preconditioned_logdet_enclosure(&diag, &off, false).expect("order-2 enclosure");
        let e3 =
            block_preconditioned_logdet_enclosure(&diag, &off, true).expect("order-3 enclosure");
        assert!(
            e2.lower <= truth && truth <= e2.upper,
            "order-2 enclosure [{}, {}] must contain dense log|S| = {}",
            e2.lower,
            e2.upper,
            truth
        );
        assert!(
            e3.lower <= truth && truth <= e3.upper,
            "order-3 enclosure [{}, {}] must contain dense log|S| = {}",
            e3.lower,
            e3.upper,
            truth
        );
        assert!(
            e3.gap() <= e2.gap() + 1e-12,
            "order-3 gap {} must not exceed order-2 gap {}",
            e3.gap(),
            e2.gap()
        );
        // The enclosure is non-vacuous: the correction is genuinely bounded
        // away from the trivial ±∞ and the gap shrinks with ρ²p₂.
        assert!(e3.gap() < 0.5 * e2.gap() + 1e-9 || e2.gap() < 1e-9);
    }

    /// The block-diagonal case is exact: zero coupling ⇒ enclosure collapses
    /// to the exact log|D| at width 0.
    #[test]
    fn zero_coupling_is_exact() {
        let (diag, _off, dense) = fixture(3, 2, 0.0);
        let truth = dense_logdet(&dense);
        let e = block_preconditioned_logdet_enclosure(&diag, &[], true).expect("enclosure");
        assert!((e.lower - truth).abs() < 1e-10 && (e.upper - truth).abs() < 1e-10);
        assert!(e.gap() < 1e-12);
    }

    /// Strong coupling must REFUSE (ρ ≥ 1), never emit a wrong enclosure.
    #[test]
    fn failed_radius_certificate_refuses() {
        let (diag, off, _dense) = fixture(3, 2, 5.0);
        let err = block_preconditioned_logdet_enclosure(&diag, &off, false)
            .expect_err("ρ ≥ 1 must refuse");
        assert!(err.contains("spectral-radius certificate failed"));
    }

    /// The margin verdict refuses to fabricate a point value when the gap
    /// exceeds the declared decision margin, and decides (with a midpoint that
    /// the dense truth brackets) when it is below.
    #[test]
    fn margin_verdict_is_honest_about_the_gap() {
        let (diag, off, dense) = fixture(4, 3, 0.08);
        let truth = dense_logdet(&dense);
        let e = block_preconditioned_logdet_enclosure(&diag, &off, true).expect("enclosure");
        // A margin tighter than the gap must escalate, never decide.
        let tight = e.gap() * 0.5;
        assert!(!e.decide_within_margin(tight).is_decided());
        assert!(e.decide_within_margin(tight).decided_value().is_none());
        // A margin wider than the gap decides, and the midpoint is within
        // gap/2 of the truth — i.e. interchangeable for that decision.
        let loose = e.gap() * 2.0 + 1e-9;
        let verdict = e.decide_within_margin(loose);
        assert!(verdict.is_decided());
        let value = verdict.decided_value().expect("decided");
        assert!((value - truth).abs() <= 0.5 * e.gap() + 1e-12);
    }

    /// Pair absorption preserves the dense `log|S|` it encloses while shrinking
    /// the residual: the absorbed pair drops out of `E`, so the gap can only
    /// tighten, and the enclosure still brackets the truth.
    #[test]
    fn pair_absorption_preserves_truth_and_tightens() {
        let (diag, off, dense) = fixture(4, 3, 0.14);
        let truth = dense_logdet(&dense);
        let before =
            block_preconditioned_logdet_enclosure(&diag, &off, true).expect("pre-absorption");
        let (mdiag, moff) = absorb_strongest_pair(&diag, &off).expect("absorb");
        assert_eq!(
            mdiag.len(),
            diag.len() - 1,
            "one fewer block after absorption"
        );
        let after =
            block_preconditioned_logdet_enclosure(&mdiag, &moff, true).expect("post-absorption");
        assert!(
            after.lower <= truth && truth <= after.upper,
            "absorbed enclosure [{}, {}] must still contain log|S| = {truth}",
            after.lower,
            after.upper
        );
        assert!(
            after.gap() <= before.gap() + 1e-9,
            "absorption must not widen the gap ({} vs {})",
            after.gap(),
            before.gap()
        );
    }

    /// The refinement ladder closes a margin that order-3 alone cannot, by
    /// absorbing pairs — and the decided value brackets the dense truth.
    #[test]
    fn refinement_ladder_closes_margin_via_absorption() {
        let (diag, off, dense) = fixture(5, 2, 0.16);
        let truth = dense_logdet(&dense);
        let order3 =
            block_preconditioned_logdet_enclosure(&diag, &off, true).expect("order-3 enclosure");
        // Choose a margin between the dense-exact (0) and the order-3 gap so the
        // ladder must climb past moments into absorption.
        let margin = order3.gap() * 0.5;
        let (enc, verdict) =
            refine_logdet_enclosure_to_margin(&diag, &off, margin, 8).expect("ladder");
        assert!(
            verdict.is_decided(),
            "ladder must close the margin via absorption"
        );
        assert!(
            enc.lower <= truth && truth <= enc.upper,
            "refined enclosure [{}, {}] must contain log|S| = {truth}",
            enc.lower,
            enc.upper
        );
        assert!(enc.gap() < margin);
    }
}

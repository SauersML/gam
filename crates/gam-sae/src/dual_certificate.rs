//! Global-optimality **dual certificate** for the sparse-dictionary lanes,
//! read through the BLASSO / super-resolution lens.
//!
//! # The convex program the lane approximates
//!
//! A sparse-dictionary row solve is the fixed-support case of the
//! Beurling-LASSO (BLASSO) / atomic-norm program
//!
//! ```text
//!     min_{μ}  ½‖x − Σ_k ∫ γ_k dμ_k‖²  +  λ Σ_k |μ_k|(M_k)
//! ```
//!
//! over atomic measures `μ_k` on each atom's parameter manifold `M_k`. The
//! collapsed linear lane ([`crate::sparse_dict`]) selects a support (the routed
//! atoms) and places point masses on it; the block lane does the same over the
//! group-ℓ₂ blocks. In both cases the routed support fixes *which* atoms carry
//! mass and the code solve fixes *how much*.
//!
//! # The dual certificate
//!
//! At any candidate solution with residual `r = x − x̂`, convex duality attaches
//! a **dual polynomial** whose value at atom `k` is
//!
//! ```text
//!     η_k = ⟨r, d_k⟩ / λ            (linear atom)
//!     η_g = ‖r D_gᵀ‖₂ / λ          (block atom — the *gate of the residual*)
//! ```
//!
//! The candidate is a **global optimum of the convex program** iff the dual
//! polynomial is feasible — `sup_k |η_k| ≤ 1` — and *saturates* (`|η_k| = 1`)
//! exactly on the active support. Feasibility gives (a) a measured global
//! optimality certificate sitting beside the first-order LAML audit in
//! [`crate::certificates`], and (b) a **threshold-free birth trigger**: any atom
//! or location with `η > 1` strictly decreases the objective, so it is a
//! principled residual-mining candidate. The threshold `1` is *derived* from
//! convex duality (SPEC rule 19), not a tuned coherence knob — it is the exact
//! point of indifference where a new atom would carry the same dual value as the
//! active support.
//!
//! # Scale-free implementation (no exposed λ)
//!
//! The lanes never expose `λ` — they solve the support least-squares with only a
//! tiny Tikhonov ridge, i.e. the `λ → 0` face of the program. That has one
//! consequence we make explicit rather than paper over: at an unpenalised
//! support solve the residual is orthogonal to the active atoms
//! (`⟨r, d_k⟩ = ρ c_k ≈ 0` for active `k`), so the residual's *active* dual
//! values collapse to the ridge and cannot serve as the `λ` scale. The faithful
//! scale-free surrogate is the realised **atomic mass** `|c_k|` (`‖z_g‖₂` for a
//! block): the amount of mass the solution already placed on the weakest active
//! atom is exactly `|μ_k|(M_k)` for that atom, and BLASSO saturation says the
//! implied penalty equals that mass. We therefore read, per row `i`,
//!
//! ```text
//!     implied λ_i       = min_{k ∈ S}  active_gate_k        (min active mass)
//!     off_support gate  = |⟨r_i, d_k⟩| (linear) / γ‖r_i D_gᵀ‖₂ (block), k ∉ S
//!     optimality_ratio  = max_{k ∉ S} off_gate_k  /  implied λ_i
//! ```
//!
//! `optimality_ratio ≤ 1` certifies the greedy support is **dual-feasible** at
//! the implied λ (no off-support atom carries a dual value above the weakest
//! active mass, so the support is a valid BLASSO support and no birth improves
//! it). `optimality_ratio > 1` identifies a **strictly improving birth
//! candidate**: an off-support atom whose optimal newly-added code would exceed
//! the least-used active atom's mass. The off-support gate is *exactly* the
//! optimal one-atom code of a candidate against the current residual, so the
//! ratio is a clean greedy-optimality statement and its threshold is the derived
//! `1`, never tuned.
//!
//! All dot products and norms are f64-accumulated from f32 inputs; the residual
//! correlation profile is folded to a running max/argmax per row, so the
//! `N×K` correlation matrix is never materialised (only the data-size `N×P`
//! reconstruction the lanes already expose is formed).

use crate::sparse_dict::{
    BlockSparseFit, SparseDictFit, block_gates, block_projections_row,
    reconstruct_block_sparse_rows, reconstruct_sparse_rows,
};
use ndarray::{ArrayView2, ArrayView3};
use std::collections::HashSet;
use std::f64::consts::TAU;

/// Relative slack on the derived unit threshold, absorbing the f32-input /
/// f64-accumulation rounding of the two dot-product paths (residual and code).
/// It shifts only the exact `= 1` boundary; a genuine birth (`η ≫ 1`) or a
/// genuine certificate (`η ≪ 1`) is unaffected.
const CERT_REL_SLACK: f64 = 1.0e-4;

/// Floor on the implied λ so a degenerate row with no live active mass yields a
/// large-but-finite `η` instead of a non-finite one. Set to the f32 rounding
/// unit; a real active support is orders of magnitude above it.
const LAMBDA_FLOOR: f64 = f32::EPSILON as f64;

/// Quantile levels reported for the per-row optimality-ratio distribution.
const RATIO_QUANTILES: [f64; 4] = [0.5, 0.9, 0.99, 1.0];

/// A measured global-optimality dual certificate over a fitted dictionary.
#[derive(Clone, Debug)]
pub struct DualCertificateReport {
    /// Rows the certificate was evaluated over.
    pub n_rows: usize,
    /// Fraction of rows whose greedy support is dual-feasible
    /// (`optimality_ratio ≤ 1 + slack`) — the certified global optima.
    pub frac_certified: f64,
    /// `(quantile, value)` of the per-row optimality-ratio distribution
    /// (`RATIO_QUANTILES`). The `1.0` entry is the worst row.
    pub optimality_ratio_quantiles: Vec<(f64, f64)>,
    /// Top strictly-improving `(row, atom, η)` birth candidates, `η > 1`, sorted
    /// by descending `η` and truncated to the caller's budget. Empty when every
    /// row is certified.
    pub birth_candidates: Vec<(usize, u32, f64)>,
}

/// Per-row certificate scratch: the implied λ (min active mass), the strongest
/// off-support dual value and the atom that carries it, and the derived ratio.
struct RowCertificate {
    optimality_ratio: f64,
    birth: Option<(u32, f64)>,
}

/// BLASSO dual birth ratio for a residual harmonic circle code.
///
/// `residual_coeffs[h] = (c_{h+1}, s_{h+1})` stores the Fourier residual after
/// subtracting the current measure on one block. The dual polynomial is
/// `η(t) = <r, u(t)> / λ` with `u(t)` the atom signature
/// `(cos 2πht, sin 2πht)_{h=1..H}`, and — like the linear and block lanes — λ is
/// read in MASS units, so the numerator must be too: the optimal new spike's
/// amplitude is the matched filter `<r, u(t)> / ‖u(t)‖² = <r, u(t)> / H`
/// (`‖u(t)‖² = H`; equivalently, unit-normalising the atoms to `u/√H` rescales
/// the active mass to `a√H` and lands on the same `1/H`). Without it η is
/// inflated by `H` relative to the other lanes' `optimal-new-mass / weakest-
/// active-mass` convention and the derived unit threshold means `a_new > a/H`,
/// not `a_new > a`. This returns `sup_t η(t)` against the active measure mass;
/// values above `1` are the threshold-free multiplicity/birth trigger from
/// convex duality.
pub fn harmonic_dual_birth_eta(residual_coeffs: &[(f64, f64)], active_mass: f64) -> f64 {
    if residual_coeffs.is_empty() {
        return 0.0;
    }
    let lambda = active_mass.max(LAMBDA_FLOOR);
    let (t, _curvature) = harmonic_dual_argmax(residual_coeffs);
    let matched_amplitude =
        harmonic_dual_value(residual_coeffs, t).max(0.0) / residual_coeffs.len() as f64;
    matched_amplitude / lambda
}

fn harmonic_dual_value(coeffs: &[(f64, f64)], t: f64) -> f64 {
    let mut acc = 0.0;
    for (h, &(c_h, s_h)) in coeffs.iter().enumerate() {
        let phase = TAU * (h + 1) as f64 * t;
        let (sin_h, cos_h) = phase.sin_cos();
        acc += c_h * cos_h + s_h * sin_h;
    }
    acc
}

fn harmonic_dual_derivative(coeffs: &[(f64, f64)], t: f64) -> f64 {
    let mut acc = 0.0;
    for (h, &(c_h, s_h)) in coeffs.iter().enumerate() {
        let omega = TAU * (h + 1) as f64;
        let phase = omega * t;
        let (sin_h, cos_h) = phase.sin_cos();
        acc += omega * (-c_h * sin_h + s_h * cos_h);
    }
    acc
}

fn harmonic_dual_second_derivative(coeffs: &[(f64, f64)], t: f64) -> f64 {
    let mut acc = 0.0;
    for (h, &(c_h, s_h)) in coeffs.iter().enumerate() {
        let omega = TAU * (h + 1) as f64;
        let phase = omega * t;
        let (sin_h, cos_h) = phase.sin_cos();
        acc += omega * omega * (-c_h * cos_h - s_h * sin_h);
    }
    acc
}

fn harmonic_dual_argmax(coeffs: &[(f64, f64)]) -> (f64, f64) {
    let harmonics = coeffs.len();
    let grid = 4 * harmonics.max(1);
    let mut best_t = 0.0;
    let mut best_value = f64::NEG_INFINITY;
    for idx in 0..grid {
        let t = idx as f64 / grid as f64;
        let value = harmonic_dual_value(coeffs, t);
        if value > best_value {
            best_value = value;
            best_t = t;
        }
    }

    let tolerance = f64::EPSILON.sqrt();
    let iteration_cap = 64;
    let mut t = best_t;
    let mut converged = false;
    for _step_idx in 0..iteration_cap {
        let second = harmonic_dual_second_derivative(coeffs, t);
        if second.abs() <= f64::MIN_POSITIVE {
            break;
        }
        let step = harmonic_dual_derivative(coeffs, t) / second;
        t -= step;
        if step.abs() <= tolerance * (1.0 + t.abs()) {
            converged = true;
            break;
        }
    }

    let polished_t = t.rem_euclid(1.0);
    if converged && harmonic_dual_value(coeffs, polished_t) >= best_value {
        (
            polished_t,
            harmonic_dual_second_derivative(coeffs, polished_t),
        )
    } else {
        (best_t, harmonic_dual_second_derivative(coeffs, best_t))
    }
}

/// Assemble a [`DualCertificateReport`] from per-row certificates.
fn assemble_report(rows: Vec<RowCertificate>, max_candidates: usize) -> DualCertificateReport {
    let n_rows = rows.len();
    let threshold = 1.0 + CERT_REL_SLACK;

    let mut ratios: Vec<f64> = Vec::with_capacity(n_rows);
    let mut certified = 0usize;
    let mut births: Vec<(usize, u32, f64)> = Vec::new();
    for (row_idx, rc) in rows.iter().enumerate() {
        ratios.push(rc.optimality_ratio);
        if rc.optimality_ratio <= threshold {
            certified += 1;
        }
        if let Some((atom, eta)) = rc.birth {
            if eta > threshold {
                births.push((row_idx, atom, eta));
            }
        }
    }

    let frac_certified = if n_rows == 0 {
        1.0
    } else {
        certified as f64 / n_rows as f64
    };

    let optimality_ratio_quantiles = quantiles(&mut ratios, &RATIO_QUANTILES);

    // Strongest strictly-improving births first; deterministic tie-break by
    // (row, atom) so the truncation is reproducible.
    births.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });
    births.truncate(max_candidates);

    DualCertificateReport {
        n_rows,
        frac_certified,
        optimality_ratio_quantiles,
        birth_candidates: births,
    }
}

/// Nearest-rank quantiles of `values` at the requested probabilities.
fn quantiles(values: &mut [f64], probs: &[f64]) -> Vec<(f64, f64)> {
    if values.is_empty() {
        return probs.iter().map(|&p| (p, 0.0)).collect();
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    probs
        .iter()
        .map(|&p| {
            let clamped = p.clamp(0.0, 1.0);
            // Nearest-rank: rank ∈ [1, n], index = rank − 1.
            let rank = (clamped * n as f64).ceil().max(1.0) as usize;
            let idx = rank.min(n) - 1;
            (p, values[idx])
        })
        .collect()
}

/// Global-optimality dual certificate for the collapsed **linear** lane.
///
/// For each row of `data` (`N×P`, the rows the `fit` encodes), reconstructs the
/// residual from the fitted sparse routing, then folds the residual's dual value
/// `|⟨r, d_k⟩|` over the whole dictionary — skipping the row's active support —
/// into a running max/argmax, and forms the scale-free `optimality_ratio`
/// against the weakest active code mass. See the module docs for the BLASSO
/// derivation and the `λ → 0` deviation.
pub fn sparse_dict_dual_certificate(
    data: ArrayView2<'_, f32>,
    fit: &SparseDictFit,
    max_candidates: usize,
) -> Result<DualCertificateReport, String> {
    sparse_route_dual_certificate(
        data,
        fit.decoder.view(),
        fit.indices.view(),
        fit.codes.view(),
        max_candidates,
    )
}

/// Global-optimality dual certificate for a fixed-width sparse linear route.
///
/// This is the diagnostic core for callers that own a frozen dictionary and
/// its route, but not the optimizer state that produced a [`SparseDictFit`].
/// Keeping the route as `indices[N,s]` / `codes[N,s]` avoids both an `N×K`
/// expansion and the invalid practice of fabricating convergence evidence just
/// to call a fit-oriented diagnostic.
pub fn sparse_route_dual_certificate(
    data: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    indices: ArrayView2<'_, u32>,
    codes: ArrayView2<'_, f32>,
    max_candidates: usize,
) -> Result<DualCertificateReport, String> {
    let (k, p) = decoder.dim();
    if k == 0 {
        return Err("sparse_route_dual_certificate: dictionary has no atoms".to_string());
    }
    if data.ncols() != p {
        return Err(format!(
            "sparse_route_dual_certificate: data has P={} columns but the decoder has P={p}",
            data.ncols()
        ));
    }
    let n = data.nrows();
    if indices.nrows() != n || codes.nrows() != n {
        return Err(format!(
            "sparse_route_dual_certificate: routing has {} rows but data has {n}",
            indices.nrows()
        ));
    }
    let s = indices.ncols();
    if codes.ncols() != s {
        return Err(format!(
            "sparse_route_dual_certificate: indices width {s} != codes width {}",
            codes.ncols()
        ));
    }

    let recon = reconstruct_sparse_rows(decoder, indices, codes)?;
    let mut rows: Vec<RowCertificate> = Vec::with_capacity(n);
    let mut residual: Vec<f64> = vec![0.0; p];
    let mut active: HashSet<u32> = HashSet::new();

    for i in 0..n {
        // Row residual r = x − x̂ (f64), and the active support (live codes only).
        for c in 0..p {
            residual[c] = data[[i, c]] as f64 - recon[[i, c]] as f64;
        }
        active.clear();
        let mut min_active_mass = f64::INFINITY;
        for j in 0..s {
            let code = codes[[i, j]] as f64;
            if code == 0.0 {
                continue;
            }
            active.insert(indices[[i, j]]);
            let mass = code.abs();
            if mass < min_active_mass {
                min_active_mass = mass;
            }
        }
        let implied_lambda = if min_active_mass.is_finite() {
            min_active_mass.max(LAMBDA_FLOOR)
        } else {
            LAMBDA_FLOOR
        };

        // Fold the off-support residual dual value |⟨r, d_k⟩| to a running max.
        let mut max_off_gate = 0.0f64;
        let mut argmax_atom: Option<u32> = None;
        for (atom_idx, atom) in decoder.outer_iter().enumerate() {
            if active.contains(&(atom_idx as u32)) {
                continue;
            }
            let mut dot = 0.0f64;
            for c in 0..p {
                dot += residual[c] * atom[c] as f64;
            }
            let gate = dot.abs();
            if gate > max_off_gate {
                max_off_gate = gate;
                argmax_atom = Some(atom_idx as u32);
            }
        }

        let optimality_ratio = max_off_gate / implied_lambda;
        let birth = argmax_atom.map(|a| (a, max_off_gate / implied_lambda));
        rows.push(RowCertificate {
            optimality_ratio,
            birth,
        });
    }

    Ok(assemble_report(rows, max_candidates))
}

/// Global-optimality dual certificate for the **block** lane.
///
/// The block dual value is the *gate of the residual* `γ‖r D_gᵀ‖₂` — the same
/// group-ℓ₂ presence the router ranks blocks by, evaluated at the residual
/// instead of the data. Off-support blocks are folded to a running max/argmax
/// (reported as the block's leading atom index `g·b`); the implied λ is the
/// weakest active gate `‖z_g‖₂`.
pub fn block_dual_certificate(
    data: ArrayView2<'_, f32>,
    fit: &BlockSparseFit,
    max_candidates: usize,
) -> Result<DualCertificateReport, String> {
    block_route_dual_certificate_scaled(
        data,
        fit.decoder.view(),
        fit.blocks.view(),
        fit.codes.view(),
        fit.block_size,
        fit.gamma as f64,
        max_candidates,
    )
}

/// Global-optimality dual certificate for a fixed-width sparse block route.
///
/// The route stores `blocks[N,s]` and signed within-block `codes[N,s,b]`.
/// Presence is derived exactly as `‖code‖₂`, so a redundant gate matrix and a
/// synthetic [`BlockSparseFit`] are unnecessary. For an external frozen route
/// the residual dual uses the decoder-coordinate scale (`dual_scale = 1`); the
/// fitted-object convenience [`block_dual_certificate`] supplies its learned
/// tied-encoder scale internally.
pub fn block_route_dual_certificate(
    data: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    block_size: usize,
    max_candidates: usize,
) -> Result<DualCertificateReport, String> {
    block_route_dual_certificate_scaled(
        data,
        decoder,
        blocks,
        codes,
        block_size,
        1.0,
        max_candidates,
    )
}

fn block_route_dual_certificate_scaled(
    data: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    block_size: usize,
    dual_scale: f64,
    max_candidates: usize,
) -> Result<DualCertificateReport, String> {
    let (k, p) = decoder.dim();
    let b = block_size;
    if b == 0 || k == 0 {
        return Err("block_route_dual_certificate: empty dictionary or block size".to_string());
    }
    if k % b != 0 {
        return Err(format!(
            "block_route_dual_certificate: decoder has K={k} rows, not a multiple of block size {b}"
        ));
    }
    if !(dual_scale.is_finite() && dual_scale > 0.0) {
        return Err(format!(
            "block_route_dual_certificate: dual scale must be finite and positive, got {dual_scale}"
        ));
    }
    let n_blocks = k / b;
    if data.ncols() != p {
        return Err(format!(
            "block_route_dual_certificate: data has P={} columns but the decoder has P={p}",
            data.ncols()
        ));
    }
    let n = data.nrows();
    if blocks.nrows() != n || codes.shape()[0] != n {
        return Err(format!(
            "block_route_dual_certificate: routing has {} rows but data has {n}",
            blocks.nrows()
        ));
    }
    let topk = blocks.ncols();
    if codes.shape() != [n, topk, b] {
        return Err(format!(
            "block_route_dual_certificate: codes shape {:?} does not match blocks {:?} and block size {b}",
            codes.shape(),
            blocks.dim()
        ));
    }

    let recon = reconstruct_block_sparse_rows(decoder, blocks, codes, b)?;
    let mut rows: Vec<RowCertificate> = Vec::with_capacity(n);
    let mut residual = ndarray::Array1::<f32>::zeros(p);
    let mut active: HashSet<u32> = HashSet::new();

    for i in 0..n {
        for c in 0..p {
            residual[c] = data[[i, c]] - recon[[i, c]];
        }
        active.clear();
        let mut min_active_gate = f64::INFINITY;
        for j in 0..topk {
            let mut gate2 = 0.0_f64;
            for r in 0..b {
                let code = codes[[i, j, r]] as f64;
                gate2 += code * code;
            }
            let gate = gate2.sqrt();
            if gate == 0.0 {
                continue;
            }
            active.insert(blocks[[i, j]]);
            if gate < min_active_gate {
                min_active_gate = gate;
            }
        }
        let implied_lambda = if min_active_gate.is_finite() {
            min_active_gate.max(LAMBDA_FLOOR)
        } else {
            LAMBDA_FLOOR
        };

        // Residual block gates γ‖r D_gᵀ‖₂ over every block, off-support max.
        let w = block_projections_row(residual.view(), decoder, n_blocks, b);
        let residual_gates = block_gates(w.view());
        let mut max_off_gate = 0.0f64;
        let mut argmax_block: Option<u32> = None;
        for (g, &rg) in residual_gates.iter().enumerate() {
            if active.contains(&(g as u32)) {
                continue;
            }
            let gate = dual_scale * rg as f64;
            if gate > max_off_gate {
                max_off_gate = gate;
                argmax_block = Some(g as u32);
            }
        }

        let optimality_ratio = max_off_gate / implied_lambda;
        // Report the block's leading atom index so the birth candidate is a
        // dictionary row, consistent with the linear lane.
        let birth = argmax_block.map(|g| (g * b as u32, optimality_ratio));
        rows.push(RowCertificate {
            optimality_ratio,
            birth,
        });
    }

    Ok(assemble_report(rows, max_candidates))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_dict::{BlockSparseFit, DecoderSolveStats, ScoreRouteStats, SparseDictFit};
    use ndarray::{Array2, Array3};

    /// Tiny deterministic LCG so the synthetic dictionaries need no rng crate in
    /// the test and are bit-reproducible across runs.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_f32(&mut self) -> f32 {
            // Numerical Recipes LCG, upper bits → (−1, 1).
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
            2.0 * u - 1.0
        }
    }

    fn unit_rows(k: usize, p: usize, seed: u64) -> Array2<f32> {
        let mut rng = Lcg::new(seed);
        let mut d = Array2::<f32>::zeros((k, p));
        for r in 0..k {
            let mut norm = 0.0f64;
            for c in 0..p {
                let v = rng.next_f32();
                d[[r, c]] = v;
                norm += (v as f64) * (v as f64);
            }
            let inv = 1.0 / norm.sqrt().max(1e-12);
            for c in 0..p {
                d[[r, c]] = (d[[r, c]] as f64 * inv) as f32;
            }
        }
        d
    }

    /// Make decoder rows `a` and `b` exactly orthonormal (Gram–Schmidt `b` off
    /// `a`), so a two-atom support has a clean residual geometry for the tests.
    fn orthonormalize_pair(d: &mut Array2<f32>, a: usize, b: usize) {
        let p = d.ncols();
        let dot: f64 = (0..p).map(|c| d[[a, c]] as f64 * d[[b, c]] as f64).sum();
        let mut norm = 0.0f64;
        for c in 0..p {
            let v = d[[b, c]] as f64 - dot * d[[a, c]] as f64;
            d[[b, c]] = v as f32;
            norm += v * v;
        }
        let inv = 1.0 / norm.sqrt();
        for c in 0..p {
            d[[b, c]] = (d[[b, c]] as f64 * inv) as f32;
        }
    }

    fn make_fit(decoder: Array2<f32>, indices: Array2<u32>, codes: Array2<f32>) -> SparseDictFit {
        let active = indices.ncols();
        SparseDictFit {
            decoder,
            indices,
            codes,
            explained_variance: 1.0,
            epochs: 0,
            converged: true,
            active,
            score_route_stats: ScoreRouteStats::default(),
            decoder_solve_stats: DecoderSolveStats::default(),
        }
    }

    /// Solve the ridge-regularised active-set least-squares codes for one row on
    /// a *given* support, packed fixed width — the same `(Gᵃ + ρI) c = Dᵃ x`
    /// system the lane's own `codes` solver forms, done small and in-test so the
    /// certificate is exercised against genuine LS-on-support codes.
    fn codes_on_support(
        row: ndarray::ArrayView1<'_, f32>,
        decoder: ArrayView2<'_, f32>,
        support: &[u32],
        s: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let m = support.len();
        let p = row.len();
        let ridge = 1.0e-6f64;
        let mut gram = vec![0.0f64; m * m];
        let mut rhs = vec![0.0f64; m];
        for i in 0..m {
            let di = decoder.row(support[i] as usize);
            let mut proj = 0.0f64;
            for c in 0..p {
                proj += di[c] as f64 * row[c] as f64;
            }
            rhs[i] = proj;
            for j in 0..m {
                let dj = decoder.row(support[j] as usize);
                let mut g = 0.0f64;
                for c in 0..p {
                    g += di[c] as f64 * dj[c] as f64;
                }
                gram[i * m + j] = g;
            }
            gram[i * m + i] += ridge;
        }
        let sol = solve_dense(&mut gram, &mut rhs, m);
        let mut indices = Vec::with_capacity(s);
        let mut codes = Vec::with_capacity(s);
        for i in 0..m.min(s) {
            indices.push(support[i]);
            codes.push(sol[i] as f32);
        }
        while indices.len() < s {
            indices.push(support[0]);
            codes.push(0.0);
        }
        (indices, codes)
    }

    /// Gaussian elimination with partial pivoting for the tiny in-test systems.
    fn solve_dense(a: &mut [f64], b: &mut [f64], m: usize) -> Vec<f64> {
        for col in 0..m {
            let mut pivot = col;
            for r in (col + 1)..m {
                if a[r * m + col].abs() > a[pivot * m + col].abs() {
                    pivot = r;
                }
            }
            if pivot != col {
                for c in 0..m {
                    a.swap(col * m + c, pivot * m + c);
                }
                b.swap(col, pivot);
            }
            let diag = a[col * m + col];
            for r in (col + 1)..m {
                let factor = a[r * m + col] / diag;
                for c in col..m {
                    a[r * m + c] -= factor * a[col * m + c];
                }
                b[r] -= factor * b[col];
            }
        }
        let mut x = vec![0.0f64; m];
        for r in (0..m).rev() {
            let mut acc = b[r];
            for c in (r + 1)..m {
                acc -= a[r * m + c] * x[c];
            }
            x[r] = acc / a[r * m + r];
        }
        x
    }

    #[test]
    fn exact_recovery_certifies_every_row() {
        let (k, p, s, n) = (64usize, 32usize, 2usize, 40usize);
        let decoder = unit_rows(k, p, 0xA11CE);
        let mut rng = Lcg::new(0xD1C7);

        let mut data = Array2::<f32>::zeros((n, p));
        let mut indices = Array2::<u32>::zeros((n, s));
        let mut codes = Array2::<f32>::zeros((n, s));
        for i in 0..n {
            let a0 = (i * 7 + 1) as u32 % k as u32;
            let mut a1 = (i * 13 + 5) as u32 % k as u32;
            if a1 == a0 {
                a1 = (a1 + 1) % k as u32;
            }
            let c0 = rng.next_f32() * 2.0 + 2.1; // bounded away from 0
            let c1 = rng.next_f32() * 2.0 - 2.1;
            for c in 0..p {
                data[[i, c]] += c0 * decoder[[a0 as usize, c]] + c1 * decoder[[a1 as usize, c]];
            }
            let (idx, cds) = codes_on_support(data.row(i), decoder.view(), &[a0, a1], s);
            for j in 0..s {
                indices[[i, j]] = idx[j];
                codes[[i, j]] = cds[j];
            }
        }

        let fit = make_fit(decoder, indices, codes);
        let report = sparse_dict_dual_certificate(data.view(), &fit, 8).unwrap();
        assert_eq!(report.n_rows, n);
        assert!(
            report.frac_certified >= 1.0 - 1e-9,
            "exact recovery must certify every row, got {}",
            report.frac_certified
        );
        assert!(
            report.birth_candidates.is_empty(),
            "no births at the exact optimum, got {:?}",
            report.birth_candidates
        );
        // Worst-row ratio is far below the unit threshold.
        let worst = report.optimality_ratio_quantiles.last().unwrap().1;
        assert!(
            worst < 1e-2,
            "worst ratio {worst} should be ~0 at exact recovery"
        );
    }

    #[test]
    fn dropped_atom_surfaces_as_birth_candidate() {
        let (k, p, s) = (64usize, 32usize, 2usize);
        let mut decoder = unit_rows(k, p, 0xBEEF);
        // Clean orthonormal support {0, 1} so the dropped-atom residual is exactly
        // its own contribution.
        orthonormalize_pair(&mut decoder, 0, 1);

        // Row = 5·d0 + 1·d1; keep atom 1, drop atom 0. Re-solve codes on {1}.
        let (c_drop, c_keep) = (5.0f32, 1.0f32);
        let mut row = ndarray::Array1::<f32>::zeros(p);
        for c in 0..p {
            row[c] = c_drop * decoder[[0, c]] + c_keep * decoder[[1, c]];
        }
        let n = 1usize;
        let mut data = Array2::<f32>::zeros((n, p));
        data.row_mut(0).assign(&row);

        // Broken support: only atom 1.
        let (idx, cds) = codes_on_support(row.view(), decoder.view(), &[1], s);
        let mut indices = Array2::<u32>::zeros((n, s));
        let mut codes = Array2::<f32>::zeros((n, s));
        for j in 0..s {
            indices[[0, j]] = idx[j];
            codes[[0, j]] = cds[j];
        }

        let fit = make_fit(decoder, indices, codes);
        let report = sparse_dict_dual_certificate(data.view(), &fit, 8).unwrap();
        assert_eq!(
            report.frac_certified, 0.0,
            "broken support must not certify"
        );
        let birth = report
            .birth_candidates
            .iter()
            .find(|(r, a, _)| *r == 0 && *a == 0)
            .expect("dropped atom 0 must appear as a birth candidate");
        assert!(
            birth.2 > 1.0,
            "dropped atom must have dual value η > 1, got {}",
            birth.2
        );
        // Its contribution (5) dwarfs the kept mass (1), so η ≈ 5.
        assert!(
            birth.2 > 3.0,
            "expected η ≈ 5 for the dropped atom, got {}",
            birth.2
        );
    }

    #[test]
    fn small_noise_stays_certified() {
        let (k, p, s, n) = (64usize, 32usize, 2usize, 40usize);
        let decoder = unit_rows(k, p, 0x5EED);
        let mut rng = Lcg::new(0x0FF);
        let mut noise = Lcg::new(0x213);

        let mut data = Array2::<f32>::zeros((n, p));
        let mut indices = Array2::<u32>::zeros((n, s));
        let mut codes = Array2::<f32>::zeros((n, s));
        for i in 0..n {
            let a0 = (i * 7 + 1) as u32 % k as u32;
            let mut a1 = (i * 13 + 5) as u32 % k as u32;
            if a1 == a0 {
                a1 = (a1 + 1) % k as u32;
            }
            let c0 = rng.next_f32() * 2.0 + 2.1;
            let c1 = rng.next_f32() * 2.0 - 2.1;
            for c in 0..p {
                data[[i, c]] += c0 * decoder[[a0 as usize, c]]
                    + c1 * decoder[[a1 as usize, c]]
                    + 1.0e-3 * noise.next_f32();
            }
            let (idx, cds) = codes_on_support(data.row(i), decoder.view(), &[a0, a1], s);
            for j in 0..s {
                indices[[i, j]] = idx[j];
                codes[[i, j]] = cds[j];
            }
        }

        let fit = make_fit(decoder, indices, codes);
        let report = sparse_dict_dual_certificate(data.view(), &fit, 8).unwrap();
        assert!(
            report.frac_certified >= 0.95,
            "small noise should stay certified, got frac {}",
            report.frac_certified
        );
    }

    #[test]
    fn block_exact_reconstruction_certifies() {
        // Identity decoder: 4 orthonormal blocks of size 2 in R^8.
        let (n_blocks, b, p) = (4usize, 2usize, 8usize);
        let k = n_blocks * b;
        let mut decoder = Array2::<f32>::zeros((k, p));
        for r in 0..k {
            decoder[[r, r]] = 1.0;
        }
        let topk = 2usize;
        let n = 6usize;
        let mut rng = Lcg::new(0xB10C);

        let mut data = Array2::<f32>::zeros((n, p));
        let mut blocks = Array2::<u32>::zeros((n, topk));
        let mut gates = Array2::<f32>::zeros((n, topk));
        let mut codes = Array3::<f32>::zeros((n, topk, b));
        for i in 0..n {
            let g0 = (i as u32) % n_blocks as u32;
            let mut g1 = (i as u32 + 2) % n_blocks as u32;
            if g1 == g0 {
                g1 = (g1 + 1) % n_blocks as u32;
            }
            for (slot, g) in [g0, g1].into_iter().enumerate() {
                let mut norm2 = 0.0f64;
                for r in 0..b {
                    let z = rng.next_f32() * 2.0 + 1.5; // nonzero mass
                    codes[[i, slot, r]] = z;
                    norm2 += (z as f64) * (z as f64);
                    // Identity frame: atom (g*b+r) is axis (g*b+r).
                    data[[i, (g as usize) * b + r]] += z;
                }
                blocks[[i, slot]] = g;
                gates[[i, slot]] = norm2.sqrt() as f32;
            }
        }

        let fit = BlockSparseFit {
            decoder,
            blocks,
            gates,
            codes,
            gamma: 1.0,
            block_utilization: vec![0.0; n_blocks],
            block_stable_rank: vec![0.0; n_blocks],
            matryoshka_prefix_losses: Vec::new(),
            explained_variance: 1.0,
            epochs: 0,
            converged: true,
            block_topk: topk,
            block_size: b,
        };
        let report = block_dual_certificate(data.view(), &fit, 8).unwrap();
        assert_eq!(report.n_rows, n);
        assert!(
            report.frac_certified >= 1.0 - 1e-9,
            "exact block reconstruction must certify every row, got {}",
            report.frac_certified
        );
        assert!(report.birth_candidates.is_empty());
    }

    #[test]
    fn block_dropped_block_surfaces_as_birth() {
        let (n_blocks, b, p) = (4usize, 2usize, 8usize);
        let k = n_blocks * b;
        let mut decoder = Array2::<f32>::zeros((k, p));
        for r in 0..k {
            decoder[[r, r]] = 1.0;
        }
        let topk = 2usize;
        let n = 1usize;

        // True content in blocks 0 (heavy) and 1 (light); keep only block 1.
        let mut data = Array2::<f32>::zeros((n, p));
        // block 0 gets mass (4,3) → ‖z‖=5; block 1 gets (1,0) → ‖z‖=1.
        data[[0, 0]] = 4.0;
        data[[0, 1]] = 3.0;
        data[[0, 2]] = 1.0;

        let mut blocks = Array2::<u32>::zeros((n, topk));
        let mut gates = Array2::<f32>::zeros((n, topk));
        let mut codes = Array3::<f32>::zeros((n, topk, b));
        // Only block 1 active (z = (1,0)); slot 1 padded (gate 0).
        blocks[[0, 0]] = 1;
        gates[[0, 0]] = 1.0;
        codes[[0, 0, 0]] = 1.0;

        let fit = BlockSparseFit {
            decoder,
            blocks,
            gates,
            codes,
            gamma: 1.0,
            block_utilization: vec![0.0; n_blocks],
            block_stable_rank: vec![0.0; n_blocks],
            matryoshka_prefix_losses: Vec::new(),
            explained_variance: 0.0,
            epochs: 0,
            converged: false,
            block_topk: topk,
            block_size: b,
        };
        let report = block_dual_certificate(data.view(), &fit, 8).unwrap();
        assert_eq!(report.frac_certified, 0.0);
        // Dropped block 0 → leading atom index 0, η ≈ ‖(4,3)‖/1 = 5.
        let birth = report
            .birth_candidates
            .iter()
            .find(|(r, a, _)| *r == 0 && *a == 0)
            .expect("dropped block 0 must surface as a birth candidate");
        assert!(
            birth.2 > 3.0,
            "expected η ≈ 5 for the dropped block, got {}",
            birth.2
        );
    }
}

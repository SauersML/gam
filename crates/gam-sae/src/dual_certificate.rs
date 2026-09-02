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
//! `crate::certificates`, and (b) a **threshold-free birth trigger**: any atom
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


//! #1019 stage 1 — arc-length (unit-speed) chart canonicalization for `d = 1`
//! manifold atoms (circle and interval topologies).
//!
//! A fitted atom's latent chart is gauge-arbitrary: the gauge-invariant
//! (intrinsic) smoothness penalty makes every reparameterization of the latent
//! coordinate equal-cost BY DESIGN, so nothing in the likelihood prefers the
//! honest chart and reconstruction metrics cannot detect chart dishonesty
//! (the planted-circle failure that motivated #1019 compressed the full loop
//! into ~1 rad of coordinate span at reconstruction EV 0.9979 — image
//! perfect, chart arbitrary).
//!
//! For `d = 1` the canonical representative of the `Diff(S¹)` /
//! `Diff([0, 1])` orbit is mathematically solved: the **arc-length
//! reparameterization**. Post-fit and image-frozen, compute the cumulative
//! arc length `s(t) = ∫_lo^t ‖γ'(u)‖ du` along the fitted decoder curve
//! `γ(t) = Φ(t) B`, normalize by the total length `L` to the chart's native
//! span (the basis period for a circle, the unit interval for a line
//! segment), and reparameterize: new coordinates `t̃_i = s(t_i)`, new decoder
//! `B̃` refit by exact least squares of the ORIGINAL decoded curve on a fine
//! grid against the basis at the new coordinates. The refit is linear and
//! exact up to basis expressiveness; the recomposition residual is recorded
//! and the canonicalization is REFUSED when it exceeds a small tolerance
//! relative to the curve scale — an honest fallback, never a lossy silent
//! swap.
//!
//! After canonicalization the atom's residual chart freedom downgrades from
//! the full diffeomorphism group to the finite isometry group of the
//! reference manifold: rotation + reflection (`O(2)`) for the circle,
//! reflection + translation for the interval. The certificate records this
//! with the `PinnedByCanonicalization` provenance
//! ([`crate::sae_identifiability::VerdictProvenance`]).
//!
//! `d ≥ 2` (surface atoms) stays open: there is no canonical
//! reparameterization without solving a harmonic-map / Plateau-type problem,
//! which is the remaining scope of #1019.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::linalg::faer_ndarray::fast_ab;
use crate::terms::sae_manifold::{SaeBasisEvaluator, solve_design_least_squares};

/// Number of grid CELLS for the arc-length quadrature and the decoder
/// recomposition least squares. Each cell carries a node, a midpoint, and the
/// next node (composite Simpson), so the speed field is sampled at
/// `2·ARC_LENGTH_GRID_CELLS + 1` points; the per-cell quadrature error is
/// `O(Δu⁴)`, far below the recomposition tolerance.
pub const ARC_LENGTH_GRID_CELLS: usize = 2048;

/// Relative image-recomposition tolerance: the canonicalization is refused
/// (honest fallback to the fitted chart) when the max-abs difference between
/// the original decoded curve and its recomposition through the new chart
/// exceeds this fraction of the curve scale — on the audit grid OR on the
/// fitted rows. Matched to the image-invariance contract the certificate and
/// the tests assert (reconstruction unchanged within 1e-8).
pub const CHART_RECOMPOSITION_REL_TOL: f64 = 1.0e-9;

/// The `d = 1` reference topology the canonical chart lives on.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalChartTopology {
    /// `S¹` with the basis evaluator's native period (`1.0` for the
    /// fraction-of-period harmonic evaluators). Arc length is accumulated
    /// over one full period and rescaled so the canonical chart spans
    /// exactly one period; the residual chart freedom is `O(2)` (base-point
    /// rotation + orientation reflection).
    Circle { period: f64 },
    /// A line-segment chart. Arc length is accumulated over the fitted
    /// coordinate range and rescaled to `[0, 1]`; the residual chart freedom
    /// is reflection + translation.
    Interval,
}

/// The exact, image-frozen arc-length reparameterization of one `d = 1` atom.
#[derive(Debug, Clone)]
pub struct UnitSpeedReparameterization {
    /// Canonical per-row coordinates `t̃_i = span · s(t_i) / L`, length `n`.
    pub new_row_coords: Array1<f64>,
    /// Recomposed decoder coefficients `B̃ = T · B`, shape `(m, p)`: the exact
    /// least-squares refit of the original decoded curve (on the audit grid)
    /// against the basis at the canonical coordinates.
    pub new_decoder: Array2<f64>,
    /// The `(m, m)` basis transport `T` with `Φ(t̃) · T ≈ Φ(t)` on the grid —
    /// the same object the affine gauge canonicalization uses to transport
    /// the smoothness Gram (`S̃ = T⁻ᵀ S T⁻¹` preserves `B̃ᵀ S̃ B̃ = Bᵀ S B`).
    pub decoder_transport: Array2<f64>,
    /// Total arc length `L` of the decoder curve over the canonical domain.
    pub total_arc_length: f64,
    /// Max-abs recomposition error on the audit grid, relative to the curve
    /// scale. Always `≤ CHART_RECOMPOSITION_REL_TOL` when `Some(..)` is
    /// returned.
    pub recomposition_residual: f64,
}

/// Decoder-curve speed `‖Φ'(u) B‖₂` for each evaluated coordinate row, from
/// the basis jet `(rows, m, 1)` and the decoder `(m, p)`.
fn curve_speeds(
    jet: &ndarray::Array3<f64>,
    decoder: ArrayView2<'_, f64>,
) -> Result<Vec<f64>, String> {
    let (rows, m, d) = jet.dim();
    if d != 1 {
        return Err(format!(
            "sae_chart_canonicalization: expected a 1-D latent jet, got latent_dim {d}"
        ));
    }
    if decoder.nrows() != m {
        return Err(format!(
            "sae_chart_canonicalization: jet basis width {m} != decoder rows {}",
            decoder.nrows()
        ));
    }
    let p = decoder.ncols();
    let mut speeds = Vec::with_capacity(rows);
    let mut tangent = vec![0.0_f64; p];
    for row in 0..rows {
        for slot in tangent.iter_mut() {
            *slot = 0.0;
        }
        for bm in 0..m {
            let dphi = jet[[row, bm, 0]];
            if dphi == 0.0 {
                continue;
            }
            for (j, slot) in tangent.iter_mut().enumerate() {
                *slot += dphi * decoder[[bm, j]];
            }
        }
        speeds.push(tangent.iter().map(|v| v * v).sum::<f64>().sqrt());
    }
    Ok(speeds)
}

/// Exact integral of the cell-local quadratic speed interpolant (through the
/// node, midpoint, and next-node speeds) over `[0, x]`, `x ∈ [0, h]`. At
/// `x = h` this is exactly the Simpson cell weight `h(f0 + 4fm + f1)/6`.
fn partial_cell_arc(f0: f64, fm: f64, f1: f64, h: f64, x: f64) -> f64 {
    if h <= 0.0 {
        return 0.0;
    }
    let a = (2.0 * f0 - 4.0 * fm + 2.0 * f1) / (h * h);
    let b = (-3.0 * f0 + 4.0 * fm - f1) / h;
    let x2 = x * x;
    a * x2 * x / 3.0 + b * x2 / 2.0 + f0 * x
}

/// Compute the arc-length (unit-speed) reparameterization of a fitted `d = 1`
/// atom: the canonical per-row coordinates and the exactly-recomposed decoder.
///
/// Image-frozen: the decoder curve is never refit against data — only
/// re-expressed in the canonical chart. Returns `Ok(None)` (honest skip,
/// never a lossy swap) when:
/// * the chart is degenerate (no rows, empty basis, zero/non-finite total
///   arc length, collapsed interval range), or
/// * the basis family cannot absorb the reparameterized curve within
///   [`CHART_RECOMPOSITION_REL_TOL`] of the curve scale on the audit grid.
pub fn unit_speed_reparameterization(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
) -> Result<Option<UnitSpeedReparameterization>, String> {
    let n = row_coords.len();
    let m = decoder.nrows();
    let p = decoder.ncols();
    if n == 0 || m == 0 || p == 0 {
        return Ok(None);
    }
    for &t in row_coords.iter() {
        if !t.is_finite() {
            return Ok(None);
        }
    }

    // ── Canonical quadrature domain `[lo, hi]` and target span ──────────────
    let (lo, hi, span) = match topology {
        CanonicalChartTopology::Circle { period } => {
            if !(period.is_finite() && *period > 0.0) {
                return Err(format!(
                    "sae_chart_canonicalization: circle period must be finite and positive; got {period}"
                ));
            }
            // The decoder curve is defined over the WHOLE period (the chart
            // dishonesty being cured is exactly rows compressed into a sliver
            // of it), so the arc length integrates the full circle.
            (0.0, *period, *period)
        }
        CanonicalChartTopology::Interval => {
            let mut t_min = f64::INFINITY;
            let mut t_max = f64::NEG_INFINITY;
            for &t in row_coords.iter() {
                t_min = t_min.min(t);
                t_max = t_max.max(t);
            }
            let scale = t_min.abs().max(t_max.abs()).max(1.0);
            if !(t_max - t_min > 1.0e-12 * scale) {
                // Collapsed chart: every row at one point — arc length cannot
                // define a chart there.
                return Ok(None);
            }
            (t_min, t_max, 1.0)
        }
    };

    // ── Speed field on the Simpson grid (nodes + midpoints in one call) ─────
    let cells = ARC_LENGTH_GRID_CELLS;
    let h = (hi - lo) / cells as f64;
    let mut quad_coords = Array2::<f64>::zeros((2 * cells + 1, 1));
    for j in 0..=cells {
        quad_coords[[2 * j, 0]] = lo + j as f64 * h;
        if j < cells {
            quad_coords[[2 * j + 1, 0]] = lo + (j as f64 + 0.5) * h;
        }
    }
    let (grid_phi_all, grid_jet_all) = evaluator.evaluate(quad_coords.view())?;
    if grid_phi_all.ncols() != m {
        return Err(format!(
            "sae_chart_canonicalization: evaluator basis width {} != decoder rows {m}",
            grid_phi_all.ncols()
        ));
    }
    let speeds = curve_speeds(&grid_jet_all, decoder)?;
    if speeds.iter().any(|s| !s.is_finite()) {
        return Ok(None);
    }

    // ── Composite-Simpson cumulative arc length at the nodes ────────────────
    let mut cumulative = vec![0.0_f64; cells + 1];
    for j in 0..cells {
        let f0 = speeds[2 * j];
        let fm = speeds[2 * j + 1];
        let f1 = speeds[2 * j + 2];
        cumulative[j + 1] = cumulative[j] + h * (f0 + 4.0 * fm + f1) / 6.0;
    }
    let total = cumulative[cells];
    if !(total.is_finite() && total > 0.0) {
        return Ok(None);
    }
    let rescale = span / total;

    // ── The canonical chart map `t ↦ span · s(t) / L` ───────────────────────
    let map_coord = |t: f64| -> f64 {
        let local = match topology {
            CanonicalChartTopology::Circle { period } => (t - lo).rem_euclid(*period),
            CanonicalChartTopology::Interval => (t - lo).clamp(0.0, hi - lo),
        };
        let cell = ((local / h).floor() as usize).min(cells - 1);
        let x = local - cell as f64 * h;
        let s = cumulative[cell]
            + partial_cell_arc(
                speeds[2 * cell],
                speeds[2 * cell + 1],
                speeds[2 * cell + 2],
                h,
                x,
            );
        let mapped = rescale * s;
        match topology {
            CanonicalChartTopology::Circle { period } => mapped.rem_euclid(*period),
            CanonicalChartTopology::Interval => mapped.clamp(0.0, span),
        }
    };

    let new_row_coords = Array1::from_iter(row_coords.iter().map(|&t| map_coord(t)));

    // ── Decoder recomposition: exact LS of the original curve on the grid ───
    // Audit grid = the quadrature nodes. `Φ_new · T ≈ Φ_old` (row j of Φ_old
    // is the basis at node u_j, row j of Φ_new is the basis at the node's
    // canonical image s̃(u_j)), so `B̃ = T · B` reproduces the original curve
    // values `Φ_old · B` at the canonical coordinates — the image is frozen.
    let mut node_new_coords = Array2::<f64>::zeros((cells + 1, 1));
    let mut old_phi = Array2::<f64>::zeros((cells + 1, m));
    for j in 0..=cells {
        node_new_coords[[j, 0]] = map_coord(lo + j as f64 * h);
        for bm in 0..m {
            old_phi[[j, bm]] = grid_phi_all[[2 * j, bm]];
        }
    }
    let (new_phi, new_jet) = evaluator.evaluate(node_new_coords.view())?;
    if new_jet.dim().2 != 1 || new_phi.ncols() != m {
        return Err(format!(
            "sae_chart_canonicalization: evaluator returned basis {:?} / jet {:?} at the canonical grid; expected width {m}, latent_dim 1",
            new_phi.dim(),
            new_jet.dim()
        ));
    }
    let transport = solve_design_least_squares(new_phi.view(), old_phi.view())?;
    let new_decoder = fast_ab(&transport, &decoder.to_owned());

    // ── Honest gate: max-abs recomposition error relative to curve scale ────
    let old_fit = fast_ab(&old_phi, &decoder.to_owned());
    let new_fit = fast_ab(&new_phi, &new_decoder);
    let mut fit_scale = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for (a, b) in old_fit.iter().zip(new_fit.iter()) {
        fit_scale = fit_scale.max(a.abs()).max(b.abs());
        max_abs = max_abs.max((a - b).abs());
    }
    if !(fit_scale.is_finite() && fit_scale > 0.0 && max_abs.is_finite()) {
        return Ok(None);
    }
    let recomposition_residual = max_abs / fit_scale;
    if recomposition_residual > CHART_RECOMPOSITION_REL_TOL {
        // The basis family is not expressive enough to carry the
        // arc-length-reparameterized curve: refuse rather than swap lossily.
        return Ok(None);
    }

    Ok(Some(UnitSpeedReparameterization {
        new_row_coords,
        new_decoder,
        decoder_transport: transport,
        total_arc_length: total,
        recomposition_residual,
    }))
}

//! ═══════════════════════════════════════════════════════════════════════════
//! Structure-theory grounding (memo Part VI, "binding = transport", Prop H):
//!
//! A fitted atom's decoder `γ = Φ(t)·B` and its latent chart `t` are related by
//! an intrinsic (gauge-invariant) smoothness penalty that scores every
//! reparameterization `t ↦ φ(t)`, `φ ∈ Diff(M)`, identically — the penalty sees
//! only the pullback metric / curve up to diffeomorphism, never the coordinate
//! itself. So the fit does not pick a point in coordinate space; it picks an
//! entire `Diff(M)` ORBIT, and that orbit IS a fiber of the memo's GAUGE
//! GROUPOID: two charts `t`, `φ(t)` on the same orbit are gauge-equivalent
//! representations of one physical atom, and reconstruction EV cannot see the
//! difference (the planted-circle witness: a full loop compressed into ~1 rad
//! of chart span still scores EV 0.9979 — image perfect, chart arbitrary).
//!
//! This module is a SLICE of that groupoid: for each reference topology it
//! picks out ONE canonical representative per orbit —
//!   * `d = 1` (circle / interval): the arc-length (unit-speed) representative
//!     ([`unit_speed_reparameterization`]);
//!   * `d = 2` torus / free patch: the minimum-isometry-defect flow chart
//!     ([`torus_isometry_flow_reparameterization`],
//!     [`patch_isometry_flow_reparameterization`]);
//!   * `d = 2` sphere: the round-sphere conformal-boost chart
//!     ([`sphere_isometry_flow_reparameterization`]).
//! Each slice is EXACT and IMAGE-FROZEN: the recomposed decoder reproduces the
//! original decoded curve/surface pointwise (`recompose_decoder_exact_ls`) and
//! the transport is a congruence that preserves the smoothness quadratic form
//! (`B̃ᵀ S̃ B̃ = Bᵀ S B`). So picking the slice changes nothing the objective (data
//! fit + intrinsic smoothness) can see — it is a gauge RETRACTION along the
//! orbit, the concrete realization of the memo's "binding = transport" slogan:
//! moving the *representation* to a canonical gauge without moving the physics.
//!
//! After the slice is taken, what remains of the chart freedom is no longer the
//! full continuous `Diff(M)` — it collapses to the finite LINEAR STABILIZER of
//! the reference manifold under its own isometry group: `O(2)` (rotation +
//! reflection) for the circle, reflection + translation for the interval,
//! `Isom(T², flat) = U(1)² ⋊ D₄` for the torus, `O(2) ⋉ ℝ²` for the flat patch,
//! `O(3)` for the round sphere. THIS residual group is exactly Proposition H's
//! uniqueness condition for canonical layer transport: two fitted atoms (or two
//! layers of the same atom across refits) are honestly bindable/comparable iff
//! they agree up to this residual stabilizer — nothing coarser is guaranteed,
//! and nothing finer is left to compare. The `PinnedByCanonicalization`
//! provenance ([`crate::identifiability::VerdictProvenance`]) is the
//! certificate that this collapse — continuous `Diff(M)` freedom sliced down to
//! the finite residual — has actually been performed for a given atom.
//!
//! The honesty gate threaded through every arm (`CHART_RECOMPOSITION_REL_TOL`,
//! `recompose_decoder_exact_ls` returning `Ok(None)`) is the module's
//! realization of the memo's honesty doctrine: a basis too poor to carry the
//! reparameterized image through exactly REFUSES the slice rather than taking
//! it lossily — canonicalization is a retraction or it does not happen. The
//! diffeomorphism guards (`SAE_FLOW_DIFFEO_MIN_DET`,
//! `min_jacobian_det_on_grid`) are the complementary safety rail: they keep the
//! candidate slice INSIDE `Diff(M)` (`det Dφ > 0` everywhere), because a folded
//! map leaves the gauge orbit entirely rather than moving along it.
//! ═══════════════════════════════════════════════════════════════════════════
//!
//! #1019 stage 1 / #2022 — arc-length (unit-speed) chart canonicalization for `d = 1`.
//! The arc-length reparameterization is exact and IMAGE-FROZEN (reconstruction
//! unchanged; the transport preserves `BᵀSB`), so it is a gauge RETRACTION along
//! the `Diff(M)` orbit that leaves the data-fit AND intrinsic-smoothness objective
//! invariant. #2022 promotes it from a post-fit pass to IN-LOOP enforcement (see
//! [`unit_speed_retraction`]), applied at chart-refresh boundaries — never inside a
//! line search, because it re-gauges `t` and thus changes the ARD *coordinate*
//! prior energy (the term that pins the residual gauge to `t → ±t + c`). Post-fit
//! canonicalization then reduces to a verification no-op.
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
//! ([`crate::identifiability::VerdictProvenance`]).
//!
//! #1019 stage 2 (`d = 2`, torus): the canonical representative of the
//! `Diff(T²)` orbit is the **minimum-isometry-defect flow** chart. The chart
//! map is parameterized as `φ_θ(t) = t + Σ_k θ_k v_k(t)` with `v_k` a fixed
//! truncated Fourier vector-field basis on `T²` (orders ≤ 2 per axis, both
//! components — a few tens of coefficients, wrap-around respected by
//! construction), and `θ` minimizes the discretized isometry defect over the
//! fitted rows with an exact analytic Gauss–Newton (see
//! [`torus_isometry_flow_reparameterization`] for the full derivation). A
//! hard diffeomorphism guard `det Dφ_θ > δ` on a check grid means a folded
//! chart is REFUSED, never produced. The decoder transport is the same
//! exact-LS recomposition — and the same honesty gate — as the `d = 1` path
//! (shared helper [`recompose_decoder_exact_ls`]).
//!
//! #1019 free-chart arm (`d = 2`, free/patch): a contractible Euclidean-patch
//! atom (Duchon / EuclideanPatch basis kind) admits a **global** truncated
//! flow basis — polynomial vector fields `v_{c,(a,b)}(t) = e_c · u₀^a u₁^b`
//! on the normalized patch box ([`FreePatchFlowBasis`]) — with no hairy-ball
//! obstruction (a contractible patch carries nowhere-vanishing vector fields).
//! So unlike the sphere it is genuinely MINIMIZED, not merely measured: the
//! isometry defect is descended against the flat reference `g_ref = I`
//! ([`patch_isometry_flow_reparameterization`]), pinning the uniform-speed
//! (minimum-anisotropy) chart, with the residual chart freedom downgraded to
//! the flat isometry group `O(2) ⋉ ℝ²` and provenance
//! `PinnedByCanonicalization`. The torus, free-patch, and sphere arms share one
//! pullback-metric extraction ([`extract_pullback_metric_d2`]) and the two
//! flow-pinned arms share one exact Gauss–Newton core
//! ([`minimize_isometry_defect_flow`]).
//!
//! `S²` (sphere atoms): the hairy-ball theorem rules out a single global
//! pole-free flow basis the way the torus and free-patch paths use, so the
//! sphere representative is the **round-sphere conformal-boost flow** — the
//! gradients of the three degree-1 harmonics (`K_z` zonal, pole-free in its one
//! latitude component; `K_x`, `K_y` carrying the longitudinal pole the theorem
//! forces somewhere). Minimizing the isometry defect over these three boosts
//! breaks the conformal (Möbius) moduli down to the round sphere's isometry
//! group `O(3)` — the chart pathology a bare harmonic energy cannot pin
//! ([`sphere_isometry_flow_reparameterization`], on a pole-margin band so the
//! `1/cos lat` boost stays well-conditioned). The same exact-image-frozen LS
//! decoder transport gates the commit: a boosted image only freezes to the
//! recomposition floor inside a basis rich enough to absorb it, so a too-poor
//! basis honestly refuses rather than silently altering the image. The
//! round-sphere isometry DEFECT remains available as a standalone measurement
//! ([`sphere_chart_isometry_defect`]).

use faer::Side as FaerSide;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{SaeBasisEvaluator, solve_design_least_squares};
use gam_linalg::faer_ndarray::{FaerCholesky, fast_ab, fast_ata, fast_atb};

/// Number of integration cells for the fitted-turning quadrature (#1026).
const TURNING_QUADRATURE_CELLS: usize = 256;

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
pub(crate) fn curve_speeds(
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
// Gauge-theory reading: `row_coords` ranges over one point of the `Diff(S¹)` /
// `Diff([0,1])` orbit the fit actually picked; this function computes the
// SLICE representative of that same orbit (arc length is the unique
// unit-speed member) and the transport `T` that carries the decoder there.
// Everything downstream of a successful `Some(..)` return is gauge-equivalent
// to the input under the residual `O(2)` / reflection+translation stabilizer
// — never a different physical fit.
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
    let Some(recomposition) =
        recompose_decoder_exact_ls(evaluator, decoder, old_phi.view(), node_new_coords.view())?
    else {
        // The basis family is not expressive enough to carry the
        // arc-length-reparameterized curve: refuse rather than swap lossily.
        return Ok(None);
    };

    Ok(Some(UnitSpeedReparameterization {
        new_row_coords,
        new_decoder: recomposition.new_decoder,
        decoder_transport: recomposition.transport,
        total_arc_length: total,
        recomposition_residual: recomposition.recomposition_residual,
    }))
}

/// Exact-LS decoder recomposition shared by the `d = 1` (arc-length) and
/// `d = 2` (torus isometry-flow) canonicalizations.
///
/// `old_phi` is the ORIGINAL basis at the audit grid (so `old_phi · B` is the
/// original decoded image there) and `new_coords` are the same grid points'
/// canonical images. Solves the basis transport `Φ(new) · T ≈ Φ(old)` by
/// exact least squares, recomposes `B̃ = T · B` (so the decoded image is
/// reproduced at the transported coordinates — image-frozen), and applies the
/// honesty gate: returns `Ok(None)` (refuse, never a lossy silent swap) when
/// the max-abs image drift on the audit grid exceeds
/// [`CHART_RECOMPOSITION_REL_TOL`] of the image scale.
pub(crate) struct DecoderRecomposition {
    /// `(m, m)` basis transport `T` with `Φ(new) · T ≈ Φ(old)` on the grid.
    pub transport: Array2<f64>,
    /// Recomposed decoder `B̃ = T · B`, shape `(m, p)`.
    pub new_decoder: Array2<f64>,
    /// Max-abs recomposition error on the audit grid, relative to the image
    /// scale. Always `≤ CHART_RECOMPOSITION_REL_TOL` when returned.
    pub recomposition_residual: f64,
}

// This is the module's one TRANSPORT primitive in the "binding = transport"
// sense: `T` is the exact congruence carrying the decoder from the fitted
// gauge to the canonical-slice gauge with the decoded image held fixed
// pointwise (image-frozen — no new information enters, only the
// representation moves). The honesty gate below is what makes the slice
// trustworthy: if the basis cannot carry the reparameterized curve/surface to
// within `CHART_RECOMPOSITION_REL_TOL`, the function REFUSES (`Ok(None)`)
// instead of committing a lossy transport — the slice is taken only when it
// is exact, never approximately.
pub(crate) fn recompose_decoder_exact_ls(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    old_phi: ArrayView2<'_, f64>,
    new_coords: ArrayView2<'_, f64>,
) -> Result<Option<DecoderRecomposition>, String> {
    let m = decoder.nrows();
    let (new_phi, new_jet) = evaluator.evaluate(new_coords)?;
    if new_phi.ncols() != m
        || new_phi.nrows() != old_phi.nrows()
        || new_jet.dim() != (new_coords.nrows(), m, new_coords.ncols())
    {
        return Err(format!(
            "sae_chart_canonicalization: evaluator returned basis {:?} / jet {:?} at the canonical grid; expected ({}, {m}) with latent_dim {}",
            new_phi.dim(),
            new_jet.dim(),
            old_phi.nrows(),
            new_coords.ncols()
        ));
    }
    let transport = solve_design_least_squares(new_phi.view(), old_phi)?;
    let new_decoder = fast_ab(&transport, &decoder);

    // ── Honest gate: max-abs recomposition error relative to image scale ────
    let old_fit = fast_ab(&old_phi, &decoder);
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
        // The basis family cannot absorb the reparameterized image: refuse.
        return Ok(None);
    }
    Ok(Some(DecoderRecomposition {
        transport,
        new_decoder,
        recomposition_residual,
    }))
}

// ════════════════════════════════════════════════════════════════════════════
// #1019 stage 2 — d = 2 torus isometry-flow chart canonicalization
//
// Gauge-theory reading: the `Diff(T²)` orbit here is infinite-dimensional; this
// arm slices it via a finite-dimensional PARAMETERIZED sub-family (the
// truncated Fourier flow `φ_θ`) and descends the isometry defect over `θ` to
// find the flat-isometric representative within that family. The slice is
// still exact (image-frozen recomposition, honesty-gated) even though the
// search over `θ` is a numerical minimization rather than a closed form — the
// residual gauge left over is `Isom(T², flat) = U(1)² ⋊ D₄` (pure torus
// translations, the axis-swap/reflection dihedral symmetry), which is why
// translations are excluded from the flow basis below: they are already in the
// residual stabilizer, not something the slice needs to remove.
// ════════════════════════════════════════════════════════════════════════════

/// Highest Fourier order per axis of the truncated flow basis on `T²`:
/// frequency vectors `(a, b)` with `|a|, |b| ≤ 2`. One representative per
/// antipodal pair `±(a, b)` (12 of them), `sin` + `cos` phases, both vector
/// components ⇒ `2 · 2 · 12 = 48` flow coefficients — the "dim ~ tens"
/// unconstrained smooth problem of #1019 stage 2. Constants (the pure torus
/// translations) are EXCLUDED on purpose: translations are exact isometries
/// of `(T², g_ref)`, so they leave the defect invariant and would only insert
/// null directions into the Gauss–Newton system.
pub const TORUS_FLOW_MAX_HARMONIC: i32 = 2;

/// Diffeomorphism floor `δ`: a candidate flow is REJECTED (the line search
/// treats it as a failed step; the final chart is never produced) when
/// `det Dφ_θ ≤ δ` anywhere on the check grid. `θ = 0` has `det Dφ = 1`
/// everywhere and only guarded steps are ever accepted, so the optimizer can
/// never walk through a fold. The floor is one-directional: raising it only
/// ever rejects MORE near-fold candidates, never accepts a fold, so it is a
/// conservative guard rather than a tuned operating point. The torus, free
/// patch, and sphere flows all enforce the IDENTICAL contract, so the value
/// lives here once and the per-topology floors below alias it.
///
/// Gauge-theory role: `det Dφ > 0` is precisely the condition that `φ` stays
/// inside `Diff(M)` — a fold (`det Dφ ≤ 0` somewhere) is not a reparameteriza-
/// tion at all, it is a map that identifies distinct points or reverses
/// orientation locally, so it would leave the gauge orbit rather than move
/// along it. This constant is what keeps every flow-based slice a genuine
/// gauge retraction instead of an information-destroying collapse.
pub const SAE_FLOW_DIFFEO_MIN_DET: f64 = 0.1;

/// Torus-flow diffeomorphism floor — see [`SAE_FLOW_DIFFEO_MIN_DET`].
pub const TORUS_FLOW_DIFFEO_MIN_DET: f64 = SAE_FLOW_DIFFEO_MIN_DET;

/// Per-axis node count of the diffeomorphism-guard check grid. The flow basis
/// is band-limited to `TORUS_FLOW_MAX_HARMONIC` (≤ 2 oscillations per axis),
/// so 64 nodes per axis oversample `det Dφ_θ` (itself band-limited to ≤ 4 per
/// axis) by 16×: the grid minimum is a faithful surrogate for the continuum
/// minimum at the `δ = 0.1` margin.
pub const TORUS_FLOW_GUARD_NODES_PER_AXIS: usize = 64;

/// Outer iteration cap for the damped Gauss–Newton flow optimization. The
/// problem is a 48-dimensional smooth nonlinear least squares; quadratic
/// local convergence makes this cap generous (termination is normally by the
/// relative step / improvement tolerances below).
pub const TORUS_FLOW_GN_MAX_ITERS: usize = 80;

/// Consecutive damping escalations before the Gauss–Newton declares the
/// current iterate a local minimum and stops.
pub const TORUS_FLOW_GN_MAX_REJECTS: usize = 12;

/// Minimum per-axis node count of the decoder-recomposition audit grid. The
/// actual count also scales with the basis width (`3·√m` per axis) so the
/// tensor harmonic basis is always Nyquist-oversampled on the audit grid.
pub const TORUS_TRANSPORT_MIN_NODES_PER_AXIS: usize = 48;

/// Identity of one flow mode (for tests and diagnostics): which coordinate
/// component the vector field moves, its integer frequency vector, and its
/// phase (`cos` vs `sin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TorusFlowModeKey {
    pub component: usize,
    pub freq: (i32, i32),
    pub is_cos: bool,
}

/// Per-mode sample of a `d = 2` flow basis at one chart point `t`: the scalar
/// field value `f(t)` (the displacement this mode adds to coordinate
/// `component`) and its gradient `∇f(t)` (the mode's contribution to row
/// `component` of the flow Jacobian `Dφ`). Shared by the torus
/// ([`TorusFlowBasis`]) and the free-patch ([`FreePatchFlowBasis`]) flow
/// families — the isometry-defect Gauss–Newton core
/// ([`minimize_isometry_defect_flow`]) consumes only this `(component, grad)`
/// contract, so both families descend the same exact optimizer.
#[derive(Debug, Clone, Copy)]
pub struct FlowModeSample {
    pub component: usize,
    pub value: f64,
    pub grad: [f64; 2],
}

/// Truncated Fourier vector-field basis on `T²` with period `period` per
/// axis: `v_{c,(a,b),trig}(t) = e_c · trig(2π(a·t₀ + b·t₁)/period)` for
/// `trig ∈ {sin, cos}`, `c ∈ {0, 1}`, and one frequency representative per
/// antipodal pair (`sin` of `−ω` is `−sin` of `ω`, so both signs would be
/// redundant). The flow map `φ_θ(t) = t + Σ_k θ_k v_k(t)` is automatically a
/// degree-(1,1) torus self-map (`φ(t + period·e_c) = φ(t) + period·e_c` —
/// wrap-around respected by periodicity of the displacement), and any such
/// map with `det Dφ > 0` everywhere is a global diffeomorphism of `T²`.
#[derive(Debug, Clone)]
pub struct TorusFlowBasis {
    pub period: f64,
    /// Canonical frequency representatives: `a > 0`, or `a == 0 && b > 0`.
    freqs: Vec<(i32, i32)>,
}

impl TorusFlowBasis {
    pub fn new(period: f64) -> Result<Self, String> {
        if !(period.is_finite() && period > 0.0) {
            return Err(format!(
                "TorusFlowBasis: period must be finite and positive; got {period}"
            ));
        }
        let h = TORUS_FLOW_MAX_HARMONIC;
        let mut freqs = Vec::new();
        for a in -h..=h {
            for b in -h..=h {
                if a > 0 || (a == 0 && b > 0) {
                    freqs.push((a, b));
                }
            }
        }
        Ok(Self { period, freqs })
    }

    /// Number of flow coefficients `θ`: 2 components × 2 phases × 12
    /// frequency representatives = 48 at the default order.
    pub fn dim(&self) -> usize {
        4 * self.freqs.len()
    }

    /// Mode identities in coefficient order: for each component, for each
    /// frequency representative, the `sin` mode then the `cos` mode. This IS
    /// the `θ` index layout — [`Self::mode_samples`] returns samples in the
    /// same order.
    pub fn mode_layout(&self) -> Vec<TorusFlowModeKey> {
        let mut keys = Vec::with_capacity(self.dim());
        for component in 0..2 {
            for &freq in &self.freqs {
                keys.push(TorusFlowModeKey {
                    component,
                    freq,
                    is_cos: false,
                });
                keys.push(TorusFlowModeKey {
                    component,
                    freq,
                    is_cos: true,
                });
            }
        }
        keys
    }

    /// Sample every mode (value + gradient) at chart point `t`, in `θ` order.
    pub fn mode_samples(&self, t: [f64; 2]) -> Vec<FlowModeSample> {
        let tau = std::f64::consts::TAU;
        let mut out = Vec::with_capacity(self.dim());
        for component in 0..2 {
            for &(a, b) in &self.freqs {
                let w0 = tau * a as f64 / self.period;
                let w1 = tau * b as f64 / self.period;
                let angle = w0 * t[0] + w1 * t[1];
                let s = angle.sin();
                let c = angle.cos();
                out.push(FlowModeSample {
                    component,
                    value: s,
                    grad: [w0 * c, w1 * c],
                });
                out.push(FlowModeSample {
                    component,
                    value: c,
                    grad: [-w0 * s, -w1 * s],
                });
            }
        }
        out
    }

    /// `φ_θ(t)`, wrapped into `[0, period)` per axis.
    pub fn map_point(&self, theta: &[f64], t: [f64; 2]) -> [f64; 2] {
        assert_eq!(theta.len(), self.dim(), "TorusFlowBasis: theta length");
        let mut out = t;
        for (coef, sample) in theta.iter().zip(self.mode_samples(t)) {
            out[sample.component] += coef * sample.value;
        }
        [
            out[0].rem_euclid(self.period),
            out[1].rem_euclid(self.period),
        ]
    }

    /// Flow Jacobian `Dφ_θ(t) = I + Σ_k θ_k Dv_k(t)`, row-major.
    pub fn flow_jacobian(&self, theta: &[f64], t: [f64; 2]) -> [[f64; 2]; 2] {
        assert_eq!(theta.len(), self.dim(), "TorusFlowBasis: theta length");
        let mut jac = [[1.0, 0.0], [0.0, 1.0]];
        for (coef, sample) in theta.iter().zip(self.mode_samples(t)) {
            jac[sample.component][0] += coef * sample.grad[0];
            jac[sample.component][1] += coef * sample.grad[1];
        }
        jac
    }

    /// Minimum of `det Dφ_θ` over the
    /// [`TORUS_FLOW_GUARD_NODES_PER_AXIS`]² check grid — the diffeomorphism
    /// guard's decision quantity.
    pub fn min_jacobian_det_on_grid(&self, theta: &[f64]) -> f64 {
        let nodes = TORUS_FLOW_GUARD_NODES_PER_AXIS;
        let mut min_det = f64::INFINITY;
        for i in 0..nodes {
            for j in 0..nodes {
                let t = [
                    self.period * i as f64 / nodes as f64,
                    self.period * j as f64 / nodes as f64,
                ];
                let jac = self.flow_jacobian(theta, t);
                let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                min_det = min_det.min(det);
            }
        }
        min_det
    }
}

/// The exact, image-frozen minimum-isometry-defect flow reparameterization of
/// one `d = 2` torus atom.
#[derive(Debug, Clone)]
pub struct TorusIsometryFlowReparameterization {
    /// Canonical per-row coordinates `t̃_i = φ_θ(t_i)`, shape `(n, 2)`,
    /// wrapped into `[0, period)` per axis.
    pub new_row_coords: Array2<f64>,
    /// Recomposed decoder `B̃ = T · B`, shape `(m, p)` — the exact LS refit of
    /// the original decoded image on the audit grid against the basis at the
    /// transported grid (so `γ̃ = γ ∘ φ⁻¹` without ever forming `φ⁻¹`).
    pub new_decoder: Array2<f64>,
    /// The `(m, m)` basis transport `T` with `Φ(φ(u)) · T ≈ Φ(u)` on the
    /// audit grid — the same congruence object the `d = 1` path and the
    /// affine gauge canonicalization use to transport the smoothness Gram.
    pub decoder_transport: Array2<f64>,
    /// Optimal flow coefficients `θ` (layout per
    /// [`TorusFlowBasis::mode_layout`]).
    pub flow_theta: Vec<f64>,
    /// Isometry defect `E(0)` of the fitted chart (identity flow).
    pub defect_initial: f64,
    /// Isometry defect `E(θ)` of the canonical chart. Strictly below
    /// `defect_initial` (the pass refuses no-improvement flows).
    pub defect_final: f64,
    /// The profiled global metric scale `c` at the optimum (the canonical
    /// chart's pullback metric is `≈ c·ḡ·I`).
    pub profiled_metric_scale: f64,
    /// `min det Dφ_θ` on the guard grid. Always `> TORUS_FLOW_DIFFEO_MIN_DET`
    /// when `Some(..)` is returned — a folded chart is refused upstream.
    pub min_flow_jacobian_det: f64,
    /// Max-abs recomposition error on the audit grid, relative to the image
    /// scale. Always `≤ CHART_RECOMPOSITION_REL_TOL` when `Some(..)`.
    pub recomposition_residual: f64,
}

/// Speed-uniformity defect above which the in-loop unit-speed retraction fires.
/// Below it the chart is already ~arc-length and re-gauging is skipped (keeps the
/// per-refresh cost negligible and makes the retraction idempotent at a boundary).
///
/// PRICED (#2071): the defect is a coefficient of variation
/// `stddev(âÎ³'â)/mean(âÎ³'â)` â dimensionless, `0` âº exactly unit-speed â so
/// this is a numerical-idempotency FLOOR, not a modeling knob. The retraction is
/// objective-invariant for data-fit and intrinsic smoothness (image-frozen,
/// `BáµSB` transported); ONLY the ARD coordinate prior re-evaluates at the
/// canonical `tÌ`, so the threshold governs a single question: is the chart far
/// enough from arc-length that re-gauging would move the prior's evaluation point
/// measurably? The reparameterization is built by integrating the sampled speed
/// field, so it cannot itself drive the CV below its own quadrature/interpolation
/// resolution â many orders above machine eps (`~2e-16`) yet far below any
/// non-uniformity that perturbs the ARD prior. `1e-6` sits in that gap (same
/// numerical genus as the sibling [`CHART_RECOMPOSITION_REL_TOL`] `1e-9`
/// image-scale honesty tolerance, loosened to CV scale because the defect is a
/// first-derivative statistic): below it the retraction is a numerical no-op (its
/// own residual-CV floor), above it there is genuine, correctable non-uniformity.
/// What breaks at 10Ã: at `1e-5` a chart with a real, removable `~1e-5` speed CV
/// is skipped, so the ARD coordinate prior is evaluated on a slightly
/// non-arc-length gauge; at `1e-7` the early-out almost never fires and the
/// reparam is attempted on charts already uniform to its own resolution (a
/// wasted, idempotent re-gauge each refresh boundary).
pub const UNIT_SPEED_INLOOP_DEFECT_TOL: f64 = 1.0e-6;

/// Coefficient of variation `stddev(‖γ'‖)/mean(‖γ'‖)` of the decoder-curve speed
/// field. `0` ⇔ already unit-speed (arc-length). Cheap in-loop early-out signal;
/// returns `+∞` on a collapsed/non-positive mean so the caller defers to the full
/// reparameterization (which honest-skips a degenerate chart).
fn speed_uniformity_defect(speeds: &[f64]) -> f64 {
    let n = speeds.len();
    if n == 0 {
        return 0.0;
    }
    let mean = speeds.iter().sum::<f64>() / n as f64;
    if !(mean > 0.0) {
        return f64::INFINITY;
    }
    let var = speeds.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / n as f64;
    var.sqrt() / mean
}

/// #2022 — in-loop unit-speed RETRACTION for one `d = 1` atom, the primitive the
/// fit loop calls at each chart-refresh boundary. Returns `Ok(None)` (honest skip:
/// already ~unit-speed, degenerate chart, basis not closed under the reparam, or
/// image drift above tolerance) or `Ok(Some(repar))` to apply. Objective-invariant
/// for data-fit + intrinsic smoothness (image-frozen; transport preserves `BᵀSB`);
/// the ARD coordinate prior re-evaluates at the canonical `t̃`, so this MUST be
/// applied only at refresh boundaries, never inside a line search.
pub fn unit_speed_retraction(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
) -> Result<Option<UnitSpeedReparameterization>, String> {
    let n = row_coords.len();
    if n == 0 {
        return Ok(None);
    }
    // Cheap early-out: evaluate the jet at the CURRENT coords and skip the reparam
    // when the chart is already ~arc-length. `evaluate` returns `(phi, jet)`; only
    // the jet is needed for the speed field.
    let mut coords2 = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        coords2[[i, 0]] = row_coords[i];
    }
    let (_phi, jet) = evaluator.evaluate(coords2.view())?;
    let speeds = curve_speeds(&jet, decoder)?;
    if speeds.iter().all(|s| s.is_finite())
        && speed_uniformity_defect(&speeds) < UNIT_SPEED_INLOOP_DEFECT_TOL
    {
        return Ok(None);
    }
    unit_speed_reparameterization(evaluator, decoder, row_coords, topology)
}

/// #2081 — the unit-speed (arc-length) DEFECT of a fitted `d = 1` chart: the
/// coefficient of variation `stddev(‖γ'(t)‖) / mean(‖γ'(t)‖)` of the decoder
/// curve speed over a uniform grid of the chart's canonical domain. `0` ⟺ the
/// parameterization is exactly arc-length (unit-speed); a positive value is the
/// same gauge the in-loop retraction's early-out reads
/// ([`speed_uniformity_defect`]), promoted here from a control signal to a
/// REPORTED per-atom certificate quantity.
///
/// It is a pure property of `(decoder, basis)` — measured on a UNIFORM latent
/// grid, not on the data rows — so it certifies the CHART parameterization
/// independent of where the data falls (the complement of the coordinate
/// UNIFORMITY statistic, which certifies the coordinate distribution). This is
/// exactly the #2081 lever: an `n_basis = 4` Fourier chart can reconstruct its
/// ring at high EV while its parameterization is far from arc-length, and EV
/// cannot see that — this defect can.
///
/// The domain is the circle's full period `[0, period)` or, for an interval
/// chart, the fitted coordinate range `[min t_i, max t_i]` — the SAME `(lo, hi)`
/// [`unit_speed_reparameterization`] integrates over. Returns `Ok(None)` (honest
/// skip, never a fabricated number) on a degenerate chart: empty basis/decoder,
/// a non-finite / collapsed domain, or a non-finite / zero-mean speed field.
pub fn chart_unit_speed_defect(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
) -> Result<Option<f64>, String> {
    let m = decoder.nrows();
    let p = decoder.ncols();
    if m == 0 || p == 0 {
        return Ok(None);
    }
    // Canonical domain `[lo, hi]` — identical to the reparameterization's.
    let (lo, hi) = match topology {
        CanonicalChartTopology::Circle { period } => {
            if !(period.is_finite() && *period > 0.0) {
                return Ok(None);
            }
            (0.0, *period)
        }
        CanonicalChartTopology::Interval => {
            let mut t_min = f64::INFINITY;
            let mut t_max = f64::NEG_INFINITY;
            for &t in row_coords.iter() {
                if !t.is_finite() {
                    return Ok(None);
                }
                t_min = t_min.min(t);
                t_max = t_max.max(t);
            }
            let scale = t_min.abs().max(t_max.abs()).max(1.0);
            if !(t_max - t_min > 1.0e-12 * scale) {
                return Ok(None);
            }
            (t_min, t_max)
        }
    };
    // Uniform latent grid over the domain. The speed field is band-limited by
    // the basis, so reusing the arc-length quadrature cell count oversamples it
    // — no new resolution constant. A circle omits the wrap-duplicate endpoint;
    // an interval samples both ends.
    let cells = ARC_LENGTH_GRID_CELLS;
    let step = (hi - lo) / cells as f64;
    let nodes = match topology {
        CanonicalChartTopology::Circle { .. } => cells,
        CanonicalChartTopology::Interval => cells + 1,
    };
    let mut grid = Array2::<f64>::zeros((nodes, 1));
    for j in 0..nodes {
        grid[[j, 0]] = lo + j as f64 * step;
    }
    let (grid_phi, jet) = evaluator.evaluate(grid.view())?;
    if grid_phi.ncols() != m {
        return Err(format!(
            "chart_unit_speed_defect: evaluator basis width {} != decoder rows {m}",
            grid_phi.ncols()
        ));
    }
    let speeds = curve_speeds(&jet, decoder)?;
    if speeds.iter().any(|s| !s.is_finite()) {
        return Ok(None);
    }
    let defect = speed_uniformity_defect(&speeds);
    if !defect.is_finite() {
        return Ok(None);
    }
    Ok(Some(defect))
}

/// #2081 — the honest arc-length (unit-speed) coordinate of a fitted `d = 1`
/// chart, computed as a PURE READ: the per-row canonical coordinate
/// `u_i = s(t_i)/L` plus the decoder-curve speed profile, WITHOUT the decoder
/// mutation or the [`CHART_RECOMPOSITION_REL_TOL`] recomposition gate that
/// [`unit_speed_reparameterization`] applies.
///
/// The recomposition gate rightly protects a decoder REWRITE — a finite Fourier
/// basis generically cannot re-express a warped circle within `1e-9`, so the
/// mutating canonicalization honestly refuses. But the arc-length COORDINATE is
/// a property of the fitted curve alone (`s(t) = ∫‖γ'‖`), so it is always
/// computable and must always be reportable: this is the coordinate every
/// downstream angle/dose/adjacency claim should read instead of the gauge-
/// arbitrary raw `t` (#2081). The map is the SAME `t ↦ span·s(t)/L` the mutating
/// path uses, so `coords_u_arc` equals the reparameterized coordinates whenever
/// that path DID commit — this is its ungated sibling.
///
/// Returns `Ok(None)` (honest skip, never a fabricated coordinate) on a
/// degenerate chart: empty basis/decoder, a non-finite / collapsed domain, or a
/// non-finite / zero total arc length.
#[derive(Debug, Clone)]
pub struct ChartArcLengthReading {
    /// Per-row canonical coordinate `u_i = s(t_i)/L ∈ [0, 1)` (circle, wrapped)
    /// or `∈ [0, 1]` (interval) — the arc-length reparameterization normalized
    /// to the unit interval. The honest angle/position every coordinate consumer
    /// should read in place of the raw `t_i`.
    pub coords_u_arc: Array1<f64>,
    /// Speed coefficient of variation `stddev(‖γ'‖)/mean(‖γ'‖)` on the uniform
    /// quadrature grid. `0` ⟺ the raw chart is already exactly arc-length.
    pub speed_cv: f64,
    /// RMS of `log(‖γ'‖/mean)` over the grid — the scale-invariant log-speed
    /// spread (a lognormal-flavoured complement to `speed_cv`, robust to a few
    /// fast cells dominating the variance).
    pub log_speed_rms: f64,
    /// `min ‖γ'‖ / mean ‖γ'‖` over the grid. Approaching `0` means the chart
    /// nearly collapses somewhere (a flat spot in `u`), so the arc-length
    /// reparameterization stops being a well-conditioned diffeomorphism.
    pub min_speed_over_mean: f64,
    /// `max ‖γ'‖ / mean ‖γ'‖` over the grid.
    pub max_speed_over_mean: f64,
    /// Total arc length `L` of the decoder curve over the canonical domain.
    pub total_arc_length: f64,
}

// Gauge-theory reading: `coords_u_arc` is the gauge-invariant OBSERVABLE this
// module ultimately exists to expose. It is computed by the same map as the
// mutating slice ([`unit_speed_reparameterization`]) but without ever touching
// the decoder or requiring the recomposition gate to pass — the arc-length
// position along a fitted curve is well-defined from `(decoder, basis)` alone,
// independent of which point of the `Diff(M)` orbit the fit happened to land
// on. Any downstream consumer reading angle/dose/adjacency off the raw `t_i`
// is reading a gauge-ARBITRARY quantity; reading `coords_u_arc` instead reads
// the coordinate that is actually pinned by the physics.
pub fn chart_arclength_coordinates(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
) -> Result<Option<ChartArcLengthReading>, String> {
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

    // Canonical quadrature domain `[lo, hi]` and unit target span — IDENTICAL to
    // `unit_speed_reparameterization`, but normalized to `[0, 1)` (we report the
    // unit-interval coordinate, not the native-period one).
    let (lo, hi) = match topology {
        CanonicalChartTopology::Circle { period } => {
            if !(period.is_finite() && *period > 0.0) {
                return Ok(None);
            }
            (0.0, *period)
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
                return Ok(None);
            }
            (t_min, t_max)
        }
    };

    // Speed field on the composite-Simpson grid (nodes + midpoints).
    let cells = ARC_LENGTH_GRID_CELLS;
    let h = (hi - lo) / cells as f64;
    let mut quad_coords = Array2::<f64>::zeros((2 * cells + 1, 1));
    for j in 0..=cells {
        quad_coords[[2 * j, 0]] = lo + j as f64 * h;
        if j < cells {
            quad_coords[[2 * j + 1, 0]] = lo + (j as f64 + 0.5) * h;
        }
    }
    let (_grid_phi, grid_jet) = evaluator.evaluate(quad_coords.view())?;
    let speeds = curve_speeds(&grid_jet, decoder)?;
    if speeds.iter().any(|s| !s.is_finite()) {
        return Ok(None);
    }

    // Cumulative arc length at the cell nodes (composite Simpson).
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
    let inv_total = 1.0 / total;

    // The canonical chart map `t ↦ s(t)/L`, normalized to `[0, 1)`.
    let map_unit = |t: f64| -> f64 {
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
        let u = s * inv_total;
        match topology {
            CanonicalChartTopology::Circle { .. } => u.rem_euclid(1.0),
            CanonicalChartTopology::Interval => u.clamp(0.0, 1.0),
        }
    };
    let coords_u_arc = Array1::from_iter(row_coords.iter().map(|&t| map_unit(t)));

    // Speed profile over the same grid samples (nodes + midpoints).
    let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
    if !(mean > 0.0) {
        return Ok(None);
    }
    let speed_cv = speed_uniformity_defect(&speeds);
    let mut log_sq = 0.0_f64;
    let mut min_over_mean = f64::INFINITY;
    let mut max_over_mean = f64::NEG_INFINITY;
    for &s in &speeds {
        let ratio = s / mean;
        min_over_mean = min_over_mean.min(ratio);
        max_over_mean = max_over_mean.max(ratio);
        // `s` is finite and non-negative; a genuine zero-speed cell drives
        // `min_over_mean → 0` (caught by the diffeomorphism-floor verdict) and is
        // excluded from the log spread so it does not poison it with `-inf`.
        if ratio > 0.0 {
            let l = ratio.ln();
            log_sq += l * l;
        }
    }
    let log_speed_rms = (log_sq / speeds.len() as f64).sqrt();
    if !(speed_cv.is_finite() && log_speed_rms.is_finite() && min_over_mean.is_finite()) {
        return Ok(None);
    }

    Ok(Some(ChartArcLengthReading {
        coords_u_arc,
        speed_cv,
        log_speed_rms,
        min_speed_over_mean: min_over_mean,
        max_speed_over_mean: max_over_mean,
        total_arc_length: total,
    }))
}

/// State of the flow objective at one `θ`: the defect, the profiled scale,
/// and the per-row flow Jacobians `A_i = Dφ_θ(t_i)` (row-major
/// `[a00, a01, a10, a11]`) the Gauss–Newton rows are built from.
struct FlowObjectiveState {
    defect: f64,
    scale: f64,
    a_rows: Vec<[f64; 4]>,
}

/// Evaluate the isometry-defect objective at `θ` (see
/// [`torus_isometry_flow_reparameterization`] for the derivation). Returns
/// `None` when the profiled scale degenerates (`c ≤ 0` or non-finite).
///
/// `row_base` is the per-row whitened identity `A0_i` (row-major
/// `[a00, a01, a10, a11]`) the flow modes are added on top of: the effective
/// whitened Jacobian is `Ã_i = A0_i + Σ_k θ_k W_{ik}`. The torus/free-patch
/// families pass `A0_i = I` (the flat reference whitens to the identity); the
/// sphere passes the reference Cholesky `A0_i = L_i = diag(1, cos lat_i)` and
/// pre-scales each mode `grad` by `L_i`'s diagonal, so the SAME flat-residual
/// core measures the round-sphere isometry defect `Ã_iᵀ Ã_i − s·G_i` exactly
/// (see [`sphere_isometry_flow_reparameterization`] for the whitening proof).
fn evaluate_flow_defect(
    theta: &[f64],
    row_modes: &[Vec<FlowModeSample>],
    row_base: &[[f64; 4]],
    ghat: &[[f64; 3]],
    ghat_norm_sq: f64,
) -> Option<FlowObjectiveState> {
    let n = row_modes.len();
    let mut a_rows = Vec::with_capacity(n);
    let mut cross = 0.0_f64;
    for (modes, base) in row_modes.iter().zip(row_base.iter()) {
        let mut a = *base;
        for (coef, sample) in theta.iter().zip(modes.iter()) {
            a[2 * sample.component] += coef * sample.grad[0];
            a[2 * sample.component + 1] += coef * sample.grad[1];
        }
        a_rows.push(a);
    }
    for (a, g) in a_rows.iter().zip(ghat.iter()) {
        // AᵀA in symmetric storage [m00, m11, m01].
        let m00 = a[0] * a[0] + a[2] * a[2];
        let m11 = a[1] * a[1] + a[3] * a[3];
        let m01 = a[0] * a[1] + a[2] * a[3];
        cross += m00 * g[0] + m11 * g[1] + 2.0 * m01 * g[2];
    }
    let scale = cross / ghat_norm_sq;
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let mut defect = 0.0_f64;
    for (a, g) in a_rows.iter().zip(ghat.iter()) {
        let m00 = a[0] * a[0] + a[2] * a[2];
        let m11 = a[1] * a[1] + a[3] * a[3];
        let m01 = a[0] * a[1] + a[2] * a[3];
        let r00 = m00 - scale * g[0];
        let r11 = m11 - scale * g[1];
        let r01 = m01 - scale * g[2];
        defect += r00 * r00 + r11 * r11 + 2.0 * r01 * r01;
    }
    if !defect.is_finite() {
        return None;
    }
    Some(FlowObjectiveState {
        defect,
        scale,
        a_rows,
    })
}

/// Outcome of the shared isometry-defect flow minimizer: the optimal flow
/// coefficients and the bracket of defects/scale they achieve. Returned only
/// when a strict, fold-free improvement over the identity flow was found.
struct FlowMinimization {
    theta: Vec<f64>,
    defect_initial: f64,
    defect_final: f64,
    profiled_scale: f64,
}

/// Exact damped Gauss–Newton for the `d = 2` isometry defect
/// `E(θ) = Σ_i ‖A_iᵀA_i − c·Ĝ_i‖²_F` over the flow coefficients `θ`, shared by
/// the torus and free-patch flow families (see
/// [`torus_isometry_flow_reparameterization`] for the full derivation of the
/// residual, the profiled scale `c`, and the analytic Gauss–Newton Jacobian).
///
/// The flow family enters ONLY through `row_modes` (the per-row mode samples
/// `W_{ik} = Dv_k(t_i)` and displacements) and the `min_det_on_grid` guard
/// closure, so the two families descend the identical optimizer with the
/// identical strict-descent + diffeomorphism accept test. The minimization
/// starts at `θ = 0` (`det Dφ = 1` everywhere) and never accepts a candidate
/// whose `min det Dφ_θ ≤ min_det` on the guard grid, so the iterate can never
/// walk through a fold. Returns `None` (honest skip — no lossy or folded swap)
/// when the identity chart is already isometric, the profiled scale
/// degenerates, or no strict improvement is reachable within the family.
fn minimize_isometry_defect_flow(
    row_modes: &[Vec<FlowModeSample>],
    row_base: &[[f64; 4]],
    ghat: &[[f64; 3]],
    ghat_norm_sq: f64,
    q: usize,
    min_det: f64,
    min_det_on_grid: &dyn Fn(&[f64]) -> f64,
) -> Option<FlowMinimization> {
    let n = row_modes.len();
    let mut theta = vec![0.0_f64; q];
    let mut state = evaluate_flow_defect(&theta, row_modes, row_base, ghat, ghat_norm_sq)?;
    let defect_initial = state.defect;
    if !(defect_initial > 0.0) {
        // Already exactly isometric — nothing to canonicalize.
        return None;
    }
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut lambda = 1.0e-4_f64;
    let mut any_accepted = false;
    for iteration in 0..TORUS_FLOW_GN_MAX_ITERS {
        if iteration + 1 == TORUS_FLOW_GN_MAX_ITERS {
            break;
        }
        // Residual r and Gauss–Newton Jacobian J at the current θ.
        let mut jmat = Array2::<f64>::zeros((3 * n, q));
        let mut rcol = Array2::<f64>::zeros((3 * n, 1));
        for (i, (a, g)) in state.a_rows.iter().zip(ghat.iter()).enumerate() {
            let m00 = a[0] * a[0] + a[2] * a[2];
            let m11 = a[1] * a[1] + a[3] * a[3];
            let m01 = a[0] * a[1] + a[2] * a[3];
            rcol[[3 * i, 0]] = m00 - state.scale * g[0];
            rcol[[3 * i + 1, 0]] = m11 - state.scale * g[1];
            rcol[[3 * i + 2, 0]] = sqrt2 * (m01 - state.scale * g[2]);
            for (k, sample) in row_modes[i].iter().enumerate() {
                // W_{ik} has single nonzero row `component` = grad, so
                // M = W_{ik}ᵀ A_i has entries M_{ab} = grad[a]·A[component, b]
                // and S = M + Mᵀ.
                let ac0 = a[2 * sample.component];
                let ac1 = a[2 * sample.component + 1];
                let s00 = 2.0 * sample.grad[0] * ac0;
                let s11 = 2.0 * sample.grad[1] * ac1;
                let s01 = sample.grad[0] * ac1 + sample.grad[1] * ac0;
                jmat[[3 * i, k]] = s00;
                jmat[[3 * i + 1, k]] = s11;
                jmat[[3 * i + 2, k]] = sqrt2 * s01;
            }
        }
        let jtj = fast_ata(&jmat);
        let jtr = fast_atb(&jmat, &rcol);

        // Levenberg-damped step with the diffeomorphism guard in the accept
        // test: only strict-descent, fold-free candidates are ever taken.
        let mut rejects = 0usize;
        let mut accepted_step = false;
        let mut converged = false;
        let mut step_norm_sq = 0.0_f64;
        while rejects < TORUS_FLOW_GN_MAX_REJECTS {
            let mut damped = jtj.clone();
            for d in 0..q {
                damped[[d, d]] += lambda * (1.0 + jtj[[d, d]]);
            }
            let factor = match damped.cholesky(FaerSide::Lower) {
                Ok(factor) => factor,
                Err(_) => {
                    lambda *= 10.0;
                    rejects += 1;
                    continue;
                }
            };
            let mut neg_jtr = jtr.clone();
            neg_jtr.mapv_inplace(|v| -v);
            let delta = factor.solve_mat(&neg_jtr);
            let mut candidate = theta.clone();
            step_norm_sq = 0.0;
            for k in 0..q {
                candidate[k] += delta[[k, 0]];
                step_norm_sq += delta[[k, 0]] * delta[[k, 0]];
            }
            let folded = min_det_on_grid(&candidate) <= min_det;
            let candidate_state = if folded {
                None
            } else {
                evaluate_flow_defect(&candidate, row_modes, row_base, ghat, ghat_norm_sq)
            };
            match candidate_state {
                Some(next) if next.defect < state.defect => {
                    let improvement = state.defect - next.defect;
                    theta = candidate;
                    state = next;
                    any_accepted = true;
                    accepted_step = true;
                    lambda = (lambda / 10.0).max(1.0e-12);
                    if improvement <= 1.0e-14 * (1.0 + state.defect) {
                        // Converged: the accepted step no longer moves E.
                        converged = true;
                    }
                    break;
                }
                Some(..) | None => {
                    lambda *= 10.0;
                    rejects += 1;
                }
            }
        }
        if !accepted_step {
            break;
        }
        if converged {
            break;
        }
        let theta_norm_sq: f64 = theta.iter().map(|v| v * v).sum();
        if step_norm_sq <= 1.0e-24 * (1.0 + theta_norm_sq) {
            break;
        }
    }
    if !any_accepted || !(state.defect < defect_initial) {
        // No strict improvement within the flow family: the fitted chart is
        // already the canonical representative — honest skip.
        return None;
    }
    Some(FlowMinimization {
        theta,
        defect_initial,
        defect_final: state.defect,
        profiled_scale: state.scale,
    })
}

/// Extract the fitted pullback metric `G_i = J(t_i)ᵀ J(t_i)` (symmetric storage
/// `[g00, g11, g01]`) at every row of a `d = 2` atom from the exact decoder jet,
/// together with the geometric-mean metric scale `ḡ = exp(mean_i ½ log det G_i)`
/// used by every `d = 2` defect for its scale-invariant normalization. Shared by
/// the torus, free-patch, and sphere defect paths — the single source of truth
/// for the pullback-metric extraction.
///
/// Returns `Ok(None)` (honest refusal) on a degenerate chart: empty rows/basis,
/// a rank-deficient pullback metric (`det G_i ≤ 0`) anywhere — the chart is
/// collapsed along some direction there, so no isometric representative exists —
/// or a non-finite geometric-mean scale.
fn extract_pullback_metric_d2(
    label: &str,
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView2<'_, f64>,
) -> Result<Option<(Vec<[f64; 3]>, f64)>, String> {
    let n = row_coords.nrows();
    let m = decoder.nrows();
    let p = decoder.ncols();
    let (row_phi, row_jet) = evaluator.evaluate(row_coords)?;
    if row_phi.ncols() != m || row_jet.dim() != (n, m, 2) {
        return Err(format!(
            "{label}: evaluator returned basis {:?} / jet {:?}; expected width {m}, latent_dim 2",
            row_phi.dim(),
            row_jet.dim()
        ));
    }
    let mut g_rows: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut log_det_sum = 0.0_f64;
    let mut tangent0 = vec![0.0_f64; p];
    let mut tangent1 = vec![0.0_f64; p];
    for row in 0..n {
        for slot in tangent0.iter_mut() {
            *slot = 0.0;
        }
        for slot in tangent1.iter_mut() {
            *slot = 0.0;
        }
        for bm in 0..m {
            let d0 = row_jet[[row, bm, 0]];
            let d1 = row_jet[[row, bm, 1]];
            if d0 == 0.0 && d1 == 0.0 {
                continue;
            }
            for j in 0..p {
                let b = decoder[[bm, j]];
                tangent0[j] += d0 * b;
                tangent1[j] += d1 * b;
            }
        }
        let mut g00 = 0.0_f64;
        let mut g11 = 0.0_f64;
        let mut g01 = 0.0_f64;
        for j in 0..p {
            g00 += tangent0[j] * tangent0[j];
            g11 += tangent1[j] * tangent1[j];
            g01 += tangent0[j] * tangent1[j];
        }
        let det = g00 * g11 - g01 * g01;
        if !(det.is_finite() && det > 0.0) {
            return Ok(None);
        }
        log_det_sum += 0.5 * det.ln();
        g_rows.push([g00, g11, g01]);
    }
    let g_bar = (log_det_sum / n as f64).exp();
    if !(g_bar.is_finite() && g_bar > 0.0) {
        return Ok(None);
    }
    Ok(Some((g_rows, g_bar)))
}

/// Normalize the fitted metric rows against the **flat** reference `g_ref = I`:
/// `Ĝ_i = G_i / ḡ` plus the reference-norm² `Σ_i ‖Ĝ_i‖²_F` the isometry-defect
/// Gauss–Newton profiles its global scale against. Shared by the torus and
/// free-patch families (both pin to a flat uniform-speed reference); the sphere
/// uses the `diag(1, cos²lat)` reference instead and normalizes inline.
fn flat_normalized_metric(g_rows: &[[f64; 3]], g_bar: f64) -> Option<(Vec<[f64; 3]>, f64)> {
    let mut ghat: Vec<[f64; 3]> = Vec::with_capacity(g_rows.len());
    let mut ghat_norm_sq = 0.0_f64;
    for g in g_rows {
        let h = [g[0] / g_bar, g[1] / g_bar, g[2] / g_bar];
        ghat_norm_sq += h[0] * h[0] + h[1] * h[1] + 2.0 * h[2] * h[2];
        ghat.push(h);
    }
    if !(ghat_norm_sq.is_finite() && ghat_norm_sq > 0.0) {
        return None;
    }
    Some((ghat, ghat_norm_sq))
}

/// Compute the minimum-isometry-defect flow reparameterization of a fitted
/// `d = 2` torus atom: the canonical per-row coordinates `t̃_i = φ_θ(t_i)`
/// and the exactly-recomposed decoder.
///
/// # The defect functional (and why it is exactly the issue's isometry defect)
///
/// The new chart is `t̃ = φ(t)` with new decoded map `γ̃ = γ ∘ φ⁻¹` (image
/// frozen), so the pullback metric in the canonical chart at `φ(t)` is
/// `G̃(φ(t)) = Dφ(t)⁻ᵀ G(t) Dφ(t)⁻¹` where `G(t) = J(t)ᵀJ(t)` is the fitted
/// pullback metric (`J` = decoder Jacobian, from the exact `(Φ, ∂Φ)` jet).
/// The canonical chart is isometric to the flat reference torus up to a
/// global scale `s` iff `G̃ ≡ s·I`, i.e. iff `Dφᵀ Dφ ≡ G / s`. Measuring the
/// defect on THIS side of the equivalence,
///
/// ```text
/// E(θ) = Σ_i ‖ A_iᵀ A_i − c · Ĝ_i ‖²_F ,   A_i = Dφ_θ(t_i) = I + Σ_k θ_k W_{ik} ,
/// ```
///
/// keeps the residual polynomial (quadratic) in `θ` — no `Dφ⁻¹` anywhere.
/// Here `W_{ik} = Dv_k(t_i)` are the constant per-row mode Jacobians,
/// `Ĝ_i = G_i / ḡ` with `ḡ = exp( mean_i ½·log det G_i )` the geometric-mean
/// metric scale of the fitted rows (the scale-invariant normalization — the
/// `d = 2` analogue of the `d = 1` module's rescale-by-total-arc-length), and
/// `c = c(θ)` the analytically profiled residual global scale
///
/// ```text
/// c(θ) = Σ_i ⟨A_iᵀA_i, Ĝ_i⟩_F / Σ_i ‖Ĝ_i‖²_F   (the exact argmin over c),
/// ```
///
/// which absorbs the (second-order) arithmetic-vs-geometric mean mismatch so
/// the defect is exactly scale-invariant: a chart isometric up to ANY global
/// scale has `E = 0`.
///
/// # The analytic gradient / Gauss–Newton (FD-free)
///
/// With `R_i = A_iᵀA_i − c·Ĝ_i` (symmetric) and `c` profiled,
/// `∂E/∂c = 0` at `c(θ)` (envelope theorem), so the exact gradient treats `c`
/// as fixed:
///
/// ```text
/// ∂R_i/∂θ_k = W_{ik}ᵀ A_i + A_iᵀ W_{ik}
/// ∂E/∂θ_k   = 2 Σ_i ⟨R_i, W_{ik}ᵀA_i + A_iᵀW_{ik}⟩_F = 4 Σ_i ⟨R_i, A_iᵀ W_{ik}⟩_F .
/// ```
///
/// The Gauss–Newton residual vector stacks the norm-preserving symmetric
/// vectorization `svec(R_i) = (R_00, R_11, √2·R_01)` and its Jacobian stacks
/// `svec(∂R_i/∂θ_k)`, so `JᵀJ δ = −Jᵀr` is the exact Gauss–Newton system for
/// `E`; Levenberg damping plus the `det Dφ > δ` guard make every accepted
/// step a strict-descent diffeomorphism. Each `v_k` moves a single component,
/// so `W_{ik}` has one nonzero row and every row/mode contraction is a
/// handful of scalar ops.
///
/// # Honest refusals (`Ok(None)`, never a lossy or folded swap)
///
/// * degenerate chart: empty rows/basis, non-finite coordinates, or a
///   rank-deficient pullback metric (`det G_i ≤ 0`) anywhere;
/// * the optimizer finds no strict improvement over the identity flow (the
///   fitted chart is already minimum-defect within the flow family);
/// * every improving candidate violates the diffeomorphism guard;
/// * the basis cannot absorb `γ ∘ φ⁻¹` within
///   [`CHART_RECOMPOSITION_REL_TOL`] on the audit grid (shared gate with the
///   `d = 1` path).
pub fn torus_isometry_flow_reparameterization(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView2<'_, f64>,
    period: f64,
) -> Result<Option<TorusIsometryFlowReparameterization>, String> {
    let n = row_coords.nrows();
    let m = decoder.nrows();
    let p = decoder.ncols();
    if row_coords.ncols() != 2 {
        return Err(format!(
            "torus_isometry_flow_reparameterization: expected (n, 2) row coordinates; got {:?}",
            row_coords.dim()
        ));
    }
    if n == 0 || m == 0 || p == 0 {
        return Ok(None);
    }
    for &t in row_coords.iter() {
        if !t.is_finite() {
            return Ok(None);
        }
    }

    // ── Fitted pullback metric G_i = J(t_i)ᵀ J(t_i) from the exact jet, then
    //    normalize against the flat reference g_ref = I (shared helpers) ──────
    let Some((g_rows, g_bar)) = extract_pullback_metric_d2(
        "torus_isometry_flow_reparameterization",
        evaluator,
        decoder,
        row_coords,
    )?
    else {
        return Ok(None);
    };
    let Some((ghat, ghat_norm_sq)) = flat_normalized_metric(&g_rows, g_bar) else {
        return Ok(None);
    };

    // ── Flow basis + per-row mode samples (W_{ik} and the displacements) ────
    let basis = TorusFlowBasis::new(period)?;
    let q = basis.dim();
    let mut row_modes: Vec<Vec<FlowModeSample>> = Vec::with_capacity(n);
    for row in 0..n {
        row_modes.push(basis.mode_samples([row_coords[[row, 0]], row_coords[[row, 1]]]));
    }
    // Flat reference ⇒ the whitened base is the identity at every row.
    let row_base = vec![[1.0_f64, 0.0, 0.0, 1.0]; n];

    // ── Damped Gauss–Newton on θ (shared exact core; derivation above) ──────
    let Some(minimization) = minimize_isometry_defect_flow(
        &row_modes,
        &row_base,
        &ghat,
        ghat_norm_sq,
        q,
        TORUS_FLOW_DIFFEO_MIN_DET,
        &|candidate: &[f64]| basis.min_jacobian_det_on_grid(candidate),
    ) else {
        return Ok(None);
    };
    let theta = minimization.theta;
    let defect_initial = minimization.defect_initial;
    let min_flow_jacobian_det = basis.min_jacobian_det_on_grid(&theta);
    if !(min_flow_jacobian_det > TORUS_FLOW_DIFFEO_MIN_DET) {
        // Unreachable through the guarded accept path; refuse defensively
        // rather than ever committing a folded chart.
        return Ok(None);
    }

    // ── Decoder transport on the Nyquist-oversampled audit grid ─────────────
    let axis_nodes = TORUS_TRANSPORT_MIN_NODES_PER_AXIS.max(3 * (m as f64).sqrt().ceil() as usize);
    let grid_rows = axis_nodes * axis_nodes;
    let mut grid = Array2::<f64>::zeros((grid_rows, 2));
    let mut new_grid = Array2::<f64>::zeros((grid_rows, 2));
    for i in 0..axis_nodes {
        for j in 0..axis_nodes {
            let idx = i * axis_nodes + j;
            let u = [
                period * i as f64 / axis_nodes as f64,
                period * j as f64 / axis_nodes as f64,
            ];
            grid[[idx, 0]] = u[0];
            grid[[idx, 1]] = u[1];
            let mapped = basis.map_point(&theta, u);
            new_grid[[idx, 0]] = mapped[0];
            new_grid[[idx, 1]] = mapped[1];
        }
    }
    let (grid_phi, grid_jet) = evaluator.evaluate(grid.view())?;
    if grid_phi.ncols() != m || grid_jet.dim() != (grid_rows, m, 2) {
        return Err(format!(
            "torus_isometry_flow_reparameterization: evaluator returned basis {:?} / jet {:?} on the audit grid; expected width {m}, latent_dim 2",
            grid_phi.dim(),
            grid_jet.dim()
        ));
    }
    let Some(recomposition) =
        recompose_decoder_exact_ls(evaluator, decoder, grid_phi.view(), new_grid.view())?
    else {
        return Ok(None);
    };

    // ── Canonical per-row coordinates t̃_i = φ_θ(t_i) ────────────────────────
    let mut new_row_coords = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let mapped = basis.map_point(&theta, [row_coords[[row, 0]], row_coords[[row, 1]]]);
        new_row_coords[[row, 0]] = mapped[0];
        new_row_coords[[row, 1]] = mapped[1];
    }

    Ok(Some(TorusIsometryFlowReparameterization {
        new_row_coords,
        new_decoder: recomposition.new_decoder,
        decoder_transport: recomposition.transport,
        flow_theta: theta,
        defect_initial,
        defect_final: minimization.defect_final,
        profiled_metric_scale: minimization.profiled_scale,
        min_flow_jacobian_det,
        recomposition_residual: recomposition.recomposition_residual,
    }))
}

// ════════════════════════════════════════════════════════════════════════════
// #1019 stage 2 — d = 2 free/patch (Euclidean-patch) isometry-flow chart
// canonicalization
//
// Gauge-theory reading: a contractible patch carries no hairy-ball obstruction,
// so (unlike the sphere arm below) this slice is a genuine global minimization
// over an affine flow family, pinning the uniform-speed representative of the
// `Diff(patch)` orbit. The residual gauge left after the slice is the flat
// isometry group `O(2) ⋉ ℝ²` (rotation + reflection + translation) — reported
// via `PinnedByCanonicalization`, exactly the Prop-H stabilizer two patch fits
// must agree up to before they can be called the same bound feature.
// ════════════════════════════════════════════════════════════════════════════

/// Highest total polynomial degree of the truncated vector-field flow basis on
/// a `d = 2` Euclidean patch: monomial modes `u₀^a · u₁^b` with
/// `1 ≤ a+b ≤ PATCH_FLOW_MAX_DEGREE`.
///
/// The degree is **1** (affine flows) by exact-image-freezing necessity, not
/// timidity. The decoder lives in a polynomial basis of some fixed degree `d_b`
/// (the Duchon / EuclideanPatch monomial families), and the image is frozen by
/// re-expressing `γ̃ = γ ∘ φ⁻¹` in that SAME basis. Composing a degree-`d_b`
/// decoder image with a degree-`d_f` flow yields a degree-`d_b·d_f` polynomial,
/// which is back in the decoder basis **iff `d_f = 1`** (an affine flow leaves
/// the polynomial degree unchanged). A higher-degree flow would push `γ ∘ φ⁻¹`
/// out of the basis, the exact-LS recomposition gate
/// ([`recompose_decoder_exact_ls`]) would see a large image drift, and the
/// canonicalization would be honestly REFUSED — so a degree > 1 flow basis
/// could never actually commit a chart here. The affine family is exactly the
/// one that keeps the image-freezing exact, and it captures the dominant patch
/// chart dishonesty: an anisotropic / sheared / rotated affine reparameteriza-
/// tion (the `d = 2` analogue of the `d = 1` non-uniform-speed pathology).
///
/// The monomials are `(1, 0)` and `(0, 1)` per component, both components ⇒
/// `2 · 2 = 4` flow coefficients. The constant `(0, 0)` is EXCLUDED: a uniform
/// translation is an exact isometry of the flat patch, so it leaves the defect
/// invariant and would only insert a null direction into the Gauss–Newton
/// system. The 4 linear modes span the full `GL(2)` action on the chart
/// (rotation / shear / anisotropic scale); the rotation and global-scale
/// sub-directions are defect-null and the Levenberg damping absorbs them
/// harmlessly.
pub const PATCH_FLOW_MAX_DEGREE: usize = 1;

/// Diffeomorphism floor `δ` for the free-patch flow — identical contract to
/// [`TORUS_FLOW_DIFFEO_MIN_DET`]: a candidate with `det Dφ_θ ≤ δ` anywhere on
/// the check grid is rejected, so the optimizer can never walk through a fold.
pub const PATCH_FLOW_DIFFEO_MIN_DET: f64 = SAE_FLOW_DIFFEO_MIN_DET;

/// Per-axis node count of the free-patch diffeomorphism-guard check grid. With
/// the affine flow basis (`PATCH_FLOW_MAX_DEGREE = 1`) the Jacobian `Dφ_θ` is
/// CONSTANT over the patch, so `det Dφ_θ` is a single value and any grid is
/// exact; 48 nodes per axis keep the guard robust if the degree ever rises.
/// The guard grid spans the normalized patch box `[-1, 1]²` slightly widened to
/// `[-1.1, 1.1]²` so a fold just outside the data hull is still refused.
pub const PATCH_FLOW_GUARD_NODES_PER_AXIS: usize = 48;

/// Minimum per-axis node count of the free-patch decoder-recomposition audit
/// grid (scaled up with the basis width like the torus path).
pub const PATCH_TRANSPORT_MIN_NODES_PER_AXIS: usize = 48;

/// Identity of one free-patch flow mode (for tests / diagnostics): which
/// coordinate component the vector field moves and its monomial exponents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchFlowModeKey {
    pub component: usize,
    pub exps: (usize, usize),
}

/// Truncated polynomial vector-field basis on a `d = 2` Euclidean patch,
/// `v_{c,(a,b)}(t) = e_c · u₀^a · u₁^b` where `u = (t − center) ⊙ inv_half ∈
/// [−1, 1]²` is the affinely-normalized patch coordinate (conditioning: the raw
/// `t` box can be far from `[−1, 1]`). Because the patch is **contractible**
/// there is no hairy-ball obstruction — these are smooth global vector fields,
/// so the flow `φ_θ(t) = t + Σ_k θ_k v_k(t)` parameterizes a neighborhood of
/// the identity in `Diff(patch)` exactly the way the Fourier basis does on the
/// torus, and any such map with `det Dφ > 0` on the patch is a diffeomorphism
/// onto its image. The reference metric is the flat `g_ref = I` (uniform-speed
/// coordinates), so the canonical chart is the minimum-anisotropy-defect one.
#[derive(Debug, Clone)]
pub struct FreePatchFlowBasis {
    center: [f64; 2],
    /// `2 / span` per axis: the Jacobian `∂u/∂t` of the normalization.
    inv_half: [f64; 2],
    /// Monomial exponents `(a, b)` with `1 ≤ a+b ≤ PATCH_FLOW_MAX_DEGREE`.
    exps: Vec<(usize, usize)>,
}

impl FreePatchFlowBasis {
    /// Build the basis for the patch box `[lo, hi]` per axis. Returns an error
    /// when an axis has collapsed (`hi ≤ lo`) — a patch with no extent in some
    /// direction has no honest flat chart.
    pub fn new(lo: [f64; 2], hi: [f64; 2]) -> Result<Self, String> {
        let mut center = [0.0_f64; 2];
        let mut inv_half = [0.0_f64; 2];
        for axis in 0..2 {
            let span = hi[axis] - lo[axis];
            let scale = lo[axis].abs().max(hi[axis].abs()).max(1.0);
            if !(span.is_finite() && span > 1.0e-12 * scale) {
                return Err(format!(
                    "FreePatchFlowBasis: patch axis {axis} has collapsed extent [{}, {}]",
                    lo[axis], hi[axis]
                ));
            }
            center[axis] = 0.5 * (lo[axis] + hi[axis]);
            inv_half[axis] = 2.0 / span;
        }
        let mut exps = Vec::new();
        for total in 1..=PATCH_FLOW_MAX_DEGREE {
            for a in (0..=total).rev() {
                let b = total - a;
                exps.push((a, b));
            }
        }
        Ok(Self {
            center,
            inv_half,
            exps,
        })
    }

    /// Number of flow coefficients `θ`: 2 components × #monomials.
    pub fn dim(&self) -> usize {
        2 * self.exps.len()
    }

    /// Mode identities in coefficient order (for each component, each monomial).
    /// This IS the `θ` index layout — [`Self::mode_samples`] matches it.
    pub fn mode_layout(&self) -> Vec<PatchFlowModeKey> {
        let mut keys = Vec::with_capacity(self.dim());
        for component in 0..2 {
            for &exps in &self.exps {
                keys.push(PatchFlowModeKey { component, exps });
            }
        }
        keys
    }

    /// Normalized patch coordinate `u = (t − center) ⊙ inv_half`.
    fn normalize(&self, t: [f64; 2]) -> [f64; 2] {
        [
            (t[0] - self.center[0]) * self.inv_half[0],
            (t[1] - self.center[1]) * self.inv_half[1],
        ]
    }

    /// Sample every mode (value + gradient **in `t`**) at chart point `t`, in
    /// `θ` order. The monomial `f(u) = u₀^a · u₁^b` has `∂f/∂t_d =
    /// (∂f/∂u_d)·(∂u_d/∂t_d) = (∂f/∂u_d)·inv_half[d]` by the chain rule, so the
    /// returned gradient is already in the chart coordinate the flow Jacobian
    /// `Dφ = I + Σ θ_k ∂v_k/∂t` lives in.
    pub fn mode_samples(&self, t: [f64; 2]) -> Vec<FlowModeSample> {
        let u = self.normalize(t);
        let mut out = Vec::with_capacity(self.dim());
        for component in 0..2 {
            for &(a, b) in &self.exps {
                let value = pow_u(u[0], a) * pow_u(u[1], b);
                // ∂/∂u₀ (u₀^a u₁^b) = a·u₀^{a−1}·u₁^b ; ∂/∂u₁ = b·u₀^a·u₁^{b−1}.
                let du0 = if a == 0 {
                    0.0
                } else {
                    a as f64 * pow_u(u[0], a - 1) * pow_u(u[1], b)
                };
                let du1 = if b == 0 {
                    0.0
                } else {
                    b as f64 * pow_u(u[0], a) * pow_u(u[1], b - 1)
                };
                out.push(FlowModeSample {
                    component,
                    value,
                    grad: [du0 * self.inv_half[0], du1 * self.inv_half[1]],
                });
            }
        }
        out
    }

    /// `φ_θ(t) = t + Σ_k θ_k v_k(t)` (no wrap — the patch is not periodic).
    pub fn map_point(&self, theta: &[f64], t: [f64; 2]) -> [f64; 2] {
        assert_eq!(theta.len(), self.dim(), "FreePatchFlowBasis: theta length");
        let mut out = t;
        for (coef, sample) in theta.iter().zip(self.mode_samples(t)) {
            out[sample.component] += coef * sample.value;
        }
        out
    }

    /// Flow Jacobian `Dφ_θ(t) = I + Σ_k θ_k Dv_k(t)`, row-major.
    pub fn flow_jacobian(&self, theta: &[f64], t: [f64; 2]) -> [[f64; 2]; 2] {
        assert_eq!(theta.len(), self.dim(), "FreePatchFlowBasis: theta length");
        let mut jac = [[1.0, 0.0], [0.0, 1.0]];
        for (coef, sample) in theta.iter().zip(self.mode_samples(t)) {
            jac[sample.component][0] += coef * sample.grad[0];
            jac[sample.component][1] += coef * sample.grad[1];
        }
        jac
    }

    /// Minimum of `det Dφ_θ` over the widened normalized check grid `[−1.1,
    /// 1.1]²` (in `t` coordinates) — the diffeomorphism guard's decision
    /// quantity. The grid is built in `t` by inverting the normalization.
    pub fn min_jacobian_det_on_grid(&self, theta: &[f64]) -> f64 {
        let nodes = PATCH_FLOW_GUARD_NODES_PER_AXIS;
        let mut min_det = f64::INFINITY;
        for i in 0..nodes {
            for j in 0..nodes {
                // u ∈ [−1.1, 1.1] per axis; t = center + u / inv_half.
                let u0 = -1.1 + 2.2 * i as f64 / (nodes - 1) as f64;
                let u1 = -1.1 + 2.2 * j as f64 / (nodes - 1) as f64;
                let t = [
                    self.center[0] + u0 / self.inv_half[0],
                    self.center[1] + u1 / self.inv_half[1],
                ];
                let jac = self.flow_jacobian(theta, t);
                let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                min_det = min_det.min(det);
            }
        }
        min_det
    }
}

/// `u^k` for small non-negative integer `k` (avoids `f64::powi`'s branch on the
/// hot mode-sampling path; `k ≤ PATCH_FLOW_MAX_DEGREE`).
fn pow_u(u: f64, k: usize) -> f64 {
    let mut acc = 1.0_f64;
    for _ in 0..k {
        acc *= u;
    }
    acc
}

/// The exact, image-frozen minimum-isometry-defect flow reparameterization of
/// one `d = 2` **free/patch** (Euclidean-patch) atom.
#[derive(Debug, Clone)]
pub struct PatchIsometryFlowReparameterization {
    /// Canonical per-row coordinates `t̃_i = φ_θ(t_i)`, shape `(n, 2)`.
    pub new_row_coords: Array2<f64>,
    /// Recomposed decoder `B̃ = T · B`, shape `(m, p)` — exact LS refit of the
    /// original decoded image on the audit grid against the basis at the
    /// transported grid (so `γ̃ = γ ∘ φ⁻¹` without ever forming `φ⁻¹`).
    pub new_decoder: Array2<f64>,
    /// The `(m, m)` basis transport `T` with `Φ(φ(u)) · T ≈ Φ(u)` on the audit
    /// grid — the congruence object that transports the smoothness Gram.
    pub decoder_transport: Array2<f64>,
    /// Optimal flow coefficients `θ` (layout per
    /// [`FreePatchFlowBasis::mode_layout`]).
    pub flow_theta: Vec<f64>,
    /// Isometry defect `E(0)` of the fitted chart (identity flow).
    pub defect_initial: f64,
    /// Isometry defect `E(θ)` of the canonical chart. Strictly below
    /// `defect_initial` (the pass refuses no-improvement flows).
    pub defect_final: f64,
    /// The profiled global metric scale `c` at the optimum (the canonical
    /// chart's pullback metric is `≈ c·ḡ·I` — flat, uniform-speed).
    pub profiled_metric_scale: f64,
    /// `min det Dφ_θ` on the guard grid. Always `> PATCH_FLOW_DIFFEO_MIN_DET`
    /// when `Some(..)` is returned — a folded chart is refused upstream.
    pub min_flow_jacobian_det: f64,
    /// Max-abs recomposition error on the audit grid, relative to the image
    /// scale. Always `≤ CHART_RECOMPOSITION_REL_TOL` when `Some(..)`.
    pub recomposition_residual: f64,
}

/// Compute the minimum-isometry-defect flow reparameterization of a fitted
/// `d = 2` **free/patch** atom: the canonical per-row coordinates
/// `t̃_i = φ_θ(t_i)` and the exactly-recomposed decoder.
///
/// This is the #1019 unblocked free-chart case the issue charter calls out: for
/// a manifold patch (a contractible Euclidean-patch atom) a **global** truncated
/// flow basis DOES exist (no hairy-ball obstruction — [`FreePatchFlowBasis`]),
/// so the defect is genuinely MINIMIZED (not merely measured as on the sphere).
/// The reference metric is the flat `g_ref = I`, so the canonical chart is the
/// uniform-speed (minimum-anisotropy) one. Everything else — the scale-invariant
/// isometry defect `E(θ) = Σ_i ‖A_iᵀA_i − c·Ĝ_i‖²_F`, the analytic profiled scale
/// `c`, the exact Gauss–Newton, the `det Dφ > δ` diffeomorphism guard, and the
/// exact-LS decoder transport with the [`CHART_RECOMPOSITION_REL_TOL`] honesty
/// gate — is the SHARED machinery the torus path uses
/// ([`minimize_isometry_defect_flow`], [`recompose_decoder_exact_ls`]); see
/// [`torus_isometry_flow_reparameterization`] for the full derivation.
///
/// The residual chart freedom after pinning is the finite isometry group of the
/// flat patch with the reference uniform metric: `O(2) ⋉ ℝ²` (rotation +
/// reflection + translation) — reported on the certificate as the
/// `PinnedByCanonicalization` residual gauge.
///
/// # Honest refusals (`Ok(None)`, never a lossy or folded swap)
///
/// * degenerate chart: empty rows/basis, non-finite coordinates, a collapsed
///   coordinate box on some axis, or a rank-deficient pullback metric anywhere;
/// * the optimizer finds no strict improvement over the identity flow (the
///   fitted chart is already minimum-defect within the flow family);
/// * every improving candidate violates the diffeomorphism guard;
/// * the basis cannot absorb `γ ∘ φ⁻¹` within
///   [`CHART_RECOMPOSITION_REL_TOL`] on the audit grid (shared gate).
pub fn patch_isometry_flow_reparameterization(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView2<'_, f64>,
) -> Result<Option<PatchIsometryFlowReparameterization>, String> {
    let n = row_coords.nrows();
    let m = decoder.nrows();
    let p = decoder.ncols();
    if row_coords.ncols() != 2 {
        return Err(format!(
            "patch_isometry_flow_reparameterization: expected (n, 2) row coordinates; got {:?}",
            row_coords.dim()
        ));
    }
    if n == 0 || m == 0 || p == 0 {
        return Ok(None);
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for row in 0..n {
        for axis in 0..2 {
            let t = row_coords[[row, axis]];
            if !t.is_finite() {
                return Ok(None);
            }
            lo[axis] = lo[axis].min(t);
            hi[axis] = hi[axis].max(t);
        }
    }

    // ── Fitted pullback metric, normalized against the flat reference I ──────
    let Some((g_rows, g_bar)) = extract_pullback_metric_d2(
        "patch_isometry_flow_reparameterization",
        evaluator,
        decoder,
        row_coords,
    )?
    else {
        return Ok(None);
    };
    let Some((ghat, ghat_norm_sq)) = flat_normalized_metric(&g_rows, g_bar) else {
        return Ok(None);
    };

    // ── Flow basis + per-row mode samples ───────────────────────────────────
    let basis = match FreePatchFlowBasis::new(lo, hi) {
        Ok(basis) => basis,
        // A collapsed patch axis: no honest flat chart — refuse, don't error.
        Err(_) => return Ok(None),
    };
    let q = basis.dim();
    let mut row_modes: Vec<Vec<FlowModeSample>> = Vec::with_capacity(n);
    for row in 0..n {
        row_modes.push(basis.mode_samples([row_coords[[row, 0]], row_coords[[row, 1]]]));
    }
    // Flat reference ⇒ the whitened base is the identity at every row.
    let row_base = vec![[1.0_f64, 0.0, 0.0, 1.0]; n];

    // ── Damped Gauss–Newton on θ (shared exact core) ────────────────────────
    let Some(minimization) = minimize_isometry_defect_flow(
        &row_modes,
        &row_base,
        &ghat,
        ghat_norm_sq,
        q,
        PATCH_FLOW_DIFFEO_MIN_DET,
        &|candidate: &[f64]| basis.min_jacobian_det_on_grid(candidate),
    ) else {
        return Ok(None);
    };
    let theta = minimization.theta;
    let defect_initial = minimization.defect_initial;
    let min_flow_jacobian_det = basis.min_jacobian_det_on_grid(&theta);
    if !(min_flow_jacobian_det > PATCH_FLOW_DIFFEO_MIN_DET) {
        return Ok(None);
    }

    // ── Decoder transport on the audit grid spanning the patch box ──────────
    let axis_nodes = PATCH_TRANSPORT_MIN_NODES_PER_AXIS.max(3 * (m as f64).sqrt().ceil() as usize);
    let grid_rows = axis_nodes * axis_nodes;
    let mut grid = Array2::<f64>::zeros((grid_rows, 2));
    let mut new_grid = Array2::<f64>::zeros((grid_rows, 2));
    for i in 0..axis_nodes {
        for j in 0..axis_nodes {
            let idx = i * axis_nodes + j;
            let u = [
                lo[0] + (hi[0] - lo[0]) * i as f64 / (axis_nodes - 1) as f64,
                lo[1] + (hi[1] - lo[1]) * j as f64 / (axis_nodes - 1) as f64,
            ];
            grid[[idx, 0]] = u[0];
            grid[[idx, 1]] = u[1];
            let mapped = basis.map_point(&theta, u);
            new_grid[[idx, 0]] = mapped[0];
            new_grid[[idx, 1]] = mapped[1];
        }
    }
    let (grid_phi, grid_jet) = evaluator.evaluate(grid.view())?;
    if grid_phi.ncols() != m || grid_jet.dim() != (grid_rows, m, 2) {
        return Err(format!(
            "patch_isometry_flow_reparameterization: evaluator returned basis {:?} / jet {:?} on the audit grid; expected width {m}, latent_dim 2",
            grid_phi.dim(),
            grid_jet.dim()
        ));
    }
    let Some(recomposition) =
        recompose_decoder_exact_ls(evaluator, decoder, grid_phi.view(), new_grid.view())?
    else {
        return Ok(None);
    };

    // ── Canonical per-row coordinates t̃_i = φ_θ(t_i) ────────────────────────
    let mut new_row_coords = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let mapped = basis.map_point(&theta, [row_coords[[row, 0]], row_coords[[row, 1]]]);
        new_row_coords[[row, 0]] = mapped[0];
        new_row_coords[[row, 1]] = mapped[1];
    }

    Ok(Some(PatchIsometryFlowReparameterization {
        new_row_coords,
        new_decoder: recomposition.new_decoder,
        decoder_transport: recomposition.transport,
        flow_theta: theta,
        defect_initial,
        defect_final: minimization.defect_final,
        profiled_metric_scale: minimization.profiled_scale,
        min_flow_jacobian_det,
        recomposition_residual: recomposition.recomposition_residual,
    }))
}

// ════════════════════════════════════════════════════════════════════════════
// #1019 sphere arm — d = 2 sphere (S²) isometry-flow chart canonicalization
//
// Gauge-theory reading: on `S²` the hairy-ball theorem forbids a pole-free
// global flow basis, so the residual chart freedom after the gauge-invariant
// fit is not the finite isometry group directly — it is the full conformal
// (Möbius) group `PSL(2,ℂ)`, of which `O(3)` (the round sphere's isometries)
// is only a subgroup. This arm's slice therefore targets the non-isometric
// complement of that subgroup: the three conformal BOOSTS (gradients of the
// degree-1 harmonics). Minimizing the isometry defect over just the boosts
// breaks the conformal moduli down to `O(3)` — the residual stabilizer
// `PinnedByCanonicalization` reports — while leaving the true rotational
// isometries alone (they are already defect-null, i.e. already in the
// stabilizer, so including them would only add null directions to the solve).
// ════════════════════════════════════════════════════════════════════════════

/// Diffeomorphism floor `δ` for the sphere conformal-boost flow: a candidate
/// flow with `det Dφ_θ ≤ δ` anywhere on the data band is rejected, so the
/// optimizer never walks through a fold (identical contract to the torus /
/// patch floors).
pub const SPHERE_FLOW_DIFFEO_MIN_DET: f64 = SAE_FLOW_DIFFEO_MIN_DET;

/// Latitude band margin (radians) from each pole inside which the sphere
/// conformal-boost flow is well-conditioned. The off-meridian boost generators
/// carry a `1/cos lat` longitudinal component, so a chart whose data reaches
/// within this margin of a pole (`|lat| > π/2 − margin`) is honestly REFUSED
/// rather than pinned with a near-singular flow — the genuine residue of the
/// hairy-ball obstruction, scoped to exactly where it bites.
pub const SPHERE_FLOW_POLE_MARGIN: f64 = 0.20;

/// Identity of one sphere conformal-boost flow mode (for tests / diagnostics):
/// which round-sphere axis the boost points along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SphereBoostAxis {
    /// Zonal boost toward the poles, `cos(lat) ∂_lat` — pole-free in its single
    /// (latitude) component, the dominant sphere chart pathology.
    Z,
    /// Boost toward the `x = (lat 0, lon 0)` point.
    X,
    /// Boost toward the `y = (lat 0, lon π/2)` point.
    Y,
}

/// The three conformal-boost vector fields on the round `S²` in `(lat, lon)`
/// coordinates — the **non-isometric** part of the conformal group, i.e. the
/// gradient fields of the degree-1 spherical harmonics `z, x, y`.
///
/// # Why these three modes, and why they are the right (and complete) pin
///
/// After the gauge-invariant fit the sphere chart's residual freedom is the
/// conformal group of `(S², g_round)`, which is `PSL(2, ℂ)` (6-real-dim): the
/// three **rotation** Killing fields (the isometries `O(3)`) plus the three
/// **boosts**. The isometry defect `‖g_φ − g_ref‖²` is null on the rotations
/// (true isometries) and strictly positive on the boosts (a boost is a
/// non-isometric conformal map: it scales the metric by a non-constant
/// conformal factor `μ(x) ≠ 1`, compressing one hemisphere into a cap — the
/// `d = 2` analogue of the `d = 1` circle-into-`1.0`-rad pathology). So
/// minimizing the defect over the **boost** subspace breaks the conformal
/// moduli down to the residual isometry group `O(3)` — exactly the finite-dim
/// residual the certificate reports. The rotation modes are deliberately
/// EXCLUDED: they are defect-null and would only insert null directions into
/// the Gauss–Newton system (the same reason torus translations / patch
/// rotations are excluded from their flow bases).
///
/// In `(lat, lon)` the boost fields are (with `s = sin lat`, `c = cos lat`):
///
/// ```text
/// K_z = ( c,                0                 )      (gradient of z = sin lat)
/// K_x = ( s·cos lon,       −sin lon / c       )      (gradient of x)
/// K_y = ( s·sin lon,        cos lon / c       )      (gradient of y)
/// ```
///
/// `K_z` is globally smooth (pole-free); `K_x, K_y` carry the `1/c`
/// longitudinal pole the hairy-ball theorem guarantees somewhere. The flow is
/// the same Euler convention as the torus / patch families,
/// `φ_θ(t) = t + Σ_k θ_k v_k(t)`, and the band guard ([`SPHERE_FLOW_POLE_MARGIN`])
/// keeps every evaluated row away from the `1/c` singularity, so the three
/// fields are smooth and bounded on the data and the pin is well-posed.
#[derive(Debug, Clone)]
pub struct SphereBoostFlowBasis;

impl SphereBoostFlowBasis {
    /// The three boost modes are always present; the dimension is fixed at 3.
    pub fn dim(&self) -> usize {
        3
    }

    /// Mode identities in coefficient order: `[Z, X, Y]`. This IS the `θ`
    /// index layout — [`Self::mode_samples`] returns samples in the same order.
    pub fn mode_layout(&self) -> [SphereBoostAxis; 3] {
        [SphereBoostAxis::Z, SphereBoostAxis::X, SphereBoostAxis::Y]
    }

    /// The displacement `v_k(t)` of each boost mode at chart point `t`.
    fn mode_displacements(t: [f64; 2]) -> [[f64; 2]; 3] {
        let (lat, lon) = (t[0], t[1]);
        let (s, c) = (lat.sin(), lat.cos());
        let (cl, sl) = (lon.cos(), lon.sin());
        [
            [c, 0.0],          // K_z
            [s * cl, -sl / c], // K_x
            [s * sl, cl / c],  // K_y
        ]
    }

    /// Jacobian `Dv_k(t)` (`[[∂v0/∂lat, ∂v0/∂lon], [∂v1/∂lat, ∂v1/∂lon]]`) of
    /// each boost mode at `t`. With `s = sin lat`, `c = cos lat`,
    /// `∂(1/c)/∂lat = s/c²`.
    fn mode_jacobians(t: [f64; 2]) -> [[[f64; 2]; 2]; 3] {
        let (lat, lon) = (t[0], t[1]);
        let (s, c) = (lat.sin(), lat.cos());
        let (cl, sl) = (lon.cos(), lon.sin());
        let c2 = c * c;
        [
            // K_z = (c, 0)
            [[-s, 0.0], [0.0, 0.0]],
            // K_x = (s·cl, −sl/c)
            [[c * cl, -s * sl], [-sl * s / c2, -cl / c]],
            // K_y = (s·sl, cl/c)
            [[c * sl, s * cl], [cl * s / c2, -sl / c]],
        ]
    }

    /// Min `det Dφ_θ` over a band-restricted check grid for the
    /// diffeomorphism guard. `φ_θ(t) = t + Σ_k θ_k v_k(t)`, so
    /// `Dφ_θ = I + Σ_k θ_k Dv_k`. The grid spans the data latitude band
    /// `[lat_lo, lat_hi]` (kept `SPHERE_FLOW_POLE_MARGIN` off each pole) and a
    /// full longitude turn.
    fn min_jacobian_det_on_band(theta: &[f64], lat_lo: f64, lat_hi: f64) -> f64 {
        let nodes = 48usize;
        let mut min_det = f64::INFINITY;
        for i in 0..nodes {
            let lat = lat_lo + (lat_hi - lat_lo) * i as f64 / (nodes - 1) as f64;
            for j in 0..nodes {
                let lon = -std::f64::consts::PI + std::f64::consts::TAU * j as f64 / nodes as f64;
                let jac = Self::mode_jacobians([lat, lon]);
                let mut a = [[1.0_f64, 0.0], [0.0, 1.0]];
                for (k, dv) in jac.iter().enumerate() {
                    a[0][0] += theta[k] * dv[0][0];
                    a[0][1] += theta[k] * dv[0][1];
                    a[1][0] += theta[k] * dv[1][0];
                    a[1][1] += theta[k] * dv[1][1];
                }
                let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
                min_det = min_det.min(det);
            }
        }
        min_det
    }

    /// Apply the flow `φ_θ(t) = t + Σ_k θ_k v_k(t)` to one chart point.
    fn map_point(theta: &[f64], t: [f64; 2]) -> [f64; 2] {
        let disp = Self::mode_displacements(t);
        let mut out = t;
        for (k, v) in disp.iter().enumerate() {
            out[0] += theta[k] * v[0];
            out[1] += theta[k] * v[1];
        }
        out
    }
}

/// The exact, image-frozen conformal-boost flow reparameterization of one
/// `d = 2` **sphere** atom: the canonical per-row `(lat, lon)` and the exactly
/// recomposed decoder.
#[derive(Debug, Clone)]
pub struct SphereIsometryFlowReparameterization {
    /// Canonical per-row coordinates `t̃_i = φ_θ(t_i)`, shape `(n, 2)`.
    pub new_row_coords: Array2<f64>,
    /// Recomposed decoder `B̃ = T · B`, shape `(m, p)` — the exact LS refit of
    /// the original decoded image on the audit grid against the basis at the
    /// transported grid (so `γ̃ = γ ∘ φ⁻¹` without ever forming `φ⁻¹`).
    pub new_decoder: Array2<f64>,
    /// The `(m, m)` basis transport `T` with `Φ(φ(u)) · T ≈ Φ(u)` on the audit
    /// grid — the congruence object the smoothness Gram is transported by.
    pub decoder_transport: Array2<f64>,
    /// Optimal boost coefficients `θ` (layout `[Z, X, Y]`).
    pub flow_theta: Vec<f64>,
    /// Round-sphere isometry defect `E(0)` of the fitted chart (identity flow).
    pub defect_initial: f64,
    /// Round-sphere isometry defect `E(θ)` of the canonical chart. Strictly
    /// below `defect_initial` (the pass refuses no-improvement flows).
    pub defect_final: f64,
    /// `min det Dφ_θ` on the band guard grid. Always `>
    /// SPHERE_FLOW_DIFFEO_MIN_DET` when `Some(..)` is returned.
    pub min_flow_jacobian_det: f64,
    /// Max-abs recomposition error on the audit grid, relative to the image
    /// scale. Always `≤ CHART_RECOMPOSITION_REL_TOL` when `Some(..)`.
    pub recomposition_residual: f64,
}

/// Compute the minimum-isometry-defect conformal-boost flow reparameterization
/// of a fitted `d = 2` sphere atom against the round-sphere reference metric.
///
/// # The defect, the reference-whitening reduction, and why the shared core fits
///
/// The canonical sphere chart is `t̃ = φ(t)` with `γ̃ = γ ∘ φ⁻¹` (image frozen),
/// and is round-isometric up to a global scale iff its pullback metric equals
/// `c · g_ref(lat) = c · diag(1, cos²lat)`. Measuring on the `φ` side (no
/// `Dφ⁻¹`), the chart is canonical iff `Dφ(t)ᵀ g_ref(φ(t)) Dφ(t) ≡ s · G(t)`
/// for some global `s` (`G = JᵀJ` the fitted pullback metric). The round
/// reference is diagonal, `g_ref(lat) = L(lat)ᵀ L(lat)` with
/// `L(lat) = diag(1, cos lat)`, so writing the **whitened** flow Jacobian
/// `Ã_i = L(φ(t_i)) · Dφ(t_i)`,
///
/// ```text
/// Dφ_iᵀ g_ref,i Dφ_i = Ã_iᵀ Ã_i ,
/// ```
///
/// the residual `R_i = Ã_iᵀ Ã_i − s · Ĝ_i` is **exactly the flat-reference
/// residual** the torus / patch Gauss–Newton core already minimizes — only the
/// per-row base changes. At the identity flow `Dφ = I`, `Ã_i = L(lat_i) =
/// diag(1, cos lat_i)`, so the shared core is driven with the per-row base
/// `A0_i = diag(1, cos lat_i)` (flattened `[1, 0, 0, cos lat_i]`) and the boost
/// modes pre-scaled by `L`: a boost displacement adding `δ` to coordinate
/// component `a` contributes `L_i[a]·δ` to row `a` of `Ã`. Both boost
/// components of each `θ_k` are folded into the base+mode accumulation, so the
/// sphere assembles its own per-row `A_i` and Gauss–Newton system here (the
/// three boosts each move BOTH components, unlike the single-component
/// `FlowModeSample` contract), reusing the residual algebra and the damped,
/// fold-guarded accept test in spirit; the profiled scale `s` and the analytic
/// `svec` Gauss–Newton are identical to the torus derivation.
///
/// Whitening by `L(φ(t_i))` (the reference at the MOVED point) rather than
/// `L(lat_i)` targets the true round-sphere defect of the new chart. Note the
/// honest limit: `L(φ_θ(t_i))` is itself re-evaluated at the flow-moved latitude,
/// so the Gauss–Newton residual actually descended equals the true defect only to
/// FIRST ORDER in the step `θ` — for a non-infinitesimal step the two differ by an
/// un-audited second-order term (the derivation's `svec` Jacobian treats the base
/// `A0_i` as fixed, not re-differentiating `L(φ_θ)`). This does NOT make the pass
/// incorrect, only its per-iterate objective an approximation of the quantity the
/// derivation above names "exact": the deviation is caught SYMPTOMATICALLY, not
/// analytically — the post-hoc defect re-measurement on the committed chart
/// ([`sphere_chart_isometry_defect`], which evaluates `L` at the final moved point
/// exactly) together with the strict-improvement gate reject any candidate that
/// did not genuinely lower the true defect, so a committed chart is
/// improved-or-refused even where the inner objective drifted. A fully exact inner
/// objective would re-differentiate `L(φ_θ)` through the flow (a second-order
/// correction), which is deliberately not done here.
///
/// # Honest refusals (`Ok(None)`)
///
/// * degenerate chart (empty / non-finite, rank-deficient pullback metric);
/// * data within [`SPHERE_FLOW_POLE_MARGIN`] of a pole (the `1/cos lat` boost
///   singularity — the scoped hairy-ball residue);
/// * the fitted chart is already minimum-defect (no strict improvement);
/// * every improving candidate violates the diffeomorphism guard;
/// * the basis cannot absorb `γ ∘ φ⁻¹` within [`CHART_RECOMPOSITION_REL_TOL`].
pub fn sphere_isometry_flow_reparameterization(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView2<'_, f64>,
) -> Result<Option<SphereIsometryFlowReparameterization>, String> {
    let n = row_coords.nrows();
    let m = decoder.nrows();
    let p = decoder.ncols();
    if row_coords.ncols() != 2 {
        return Err(format!(
            "sphere_isometry_flow_reparameterization: expected (n, 2) row coordinates; got {:?}",
            row_coords.dim()
        ));
    }
    if n == 0 || m == 0 || p == 0 {
        return Ok(None);
    }
    for &t in row_coords.iter() {
        if !t.is_finite() {
            return Ok(None);
        }
    }

    // ── Pole-band guard: refuse charts reaching into the 1/cos lat singularity ─
    let mut lat_lo = f64::INFINITY;
    let mut lat_hi = f64::NEG_INFINITY;
    for row in 0..n {
        let lat = row_coords[[row, 0]];
        lat_lo = lat_lo.min(lat);
        lat_hi = lat_hi.max(lat);
    }
    let pole = std::f64::consts::FRAC_PI_2 - SPHERE_FLOW_POLE_MARGIN;
    if !(lat_lo > -pole && lat_hi < pole) {
        return Ok(None);
    }

    // ── Fitted pullback metric G_i = J(t_i)ᵀ J(t_i) (raw, NOT flat-normalized:
    //    the sphere target is g_ref, not I), with the geometric-mean scale ḡ ───
    let Some((g_rows, g_bar)) = extract_pullback_metric_d2(
        "sphere_isometry_flow_reparameterization",
        evaluator,
        decoder,
        row_coords,
    )?
    else {
        return Ok(None);
    };
    // ĝ_i = G_i / ḡ; the round reference target enters per row via the whitened
    // base below, so the GN `ghat` here is the scale-normalized fitted metric.
    let mut ghat: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut ghat_norm_sq = 0.0_f64;
    for g in &g_rows {
        let h = [g[0] / g_bar, g[1] / g_bar, g[2] / g_bar];
        ghat_norm_sq += h[0] * h[0] + h[1] * h[1] + 2.0 * h[2] * h[2];
        ghat.push(h);
    }
    if !(ghat_norm_sq.is_finite() && ghat_norm_sq > 0.0) {
        return Ok(None);
    }

    // ── Damped Gauss–Newton over the 3 boost coefficients ───────────────────
    let q = 3usize;
    let Some(minimization) =
        sphere_minimize_boost_defect(&ghat, ghat_norm_sq, row_coords, q, lat_lo, lat_hi)
    else {
        return Ok(None);
    };
    let theta = minimization.theta;
    let defect_initial = minimization.defect_initial;
    let min_flow_jacobian_det =
        SphereBoostFlowBasis::min_jacobian_det_on_band(&theta, lat_lo, lat_hi);
    if !(min_flow_jacobian_det > SPHERE_FLOW_DIFFEO_MIN_DET) {
        return Ok(None);
    }

    // ── Decoder transport on an audit grid spanning the data band ───────────
    let axis_nodes = TORUS_TRANSPORT_MIN_NODES_PER_AXIS.max(3 * (m as f64).sqrt().ceil() as usize);
    let grid_rows = axis_nodes * axis_nodes;
    let mut grid = Array2::<f64>::zeros((grid_rows, 2));
    let mut new_grid = Array2::<f64>::zeros((grid_rows, 2));
    for i in 0..axis_nodes {
        for j in 0..axis_nodes {
            let idx = i * axis_nodes + j;
            let lat = lat_lo + (lat_hi - lat_lo) * i as f64 / (axis_nodes - 1) as f64;
            let lon = -std::f64::consts::PI + std::f64::consts::TAU * j as f64 / axis_nodes as f64;
            grid[[idx, 0]] = lat;
            grid[[idx, 1]] = lon;
            let mapped = SphereBoostFlowBasis::map_point(&theta, [lat, lon]);
            new_grid[[idx, 0]] = mapped[0];
            new_grid[[idx, 1]] = mapped[1];
        }
    }
    let (grid_phi, grid_jet) = evaluator.evaluate(grid.view())?;
    if grid_phi.ncols() != m || grid_jet.dim() != (grid_rows, m, 2) {
        return Err(format!(
            "sphere_isometry_flow_reparameterization: evaluator returned basis {:?} / jet {:?} on the audit grid; expected width {m}, latent_dim 2",
            grid_phi.dim(),
            grid_jet.dim()
        ));
    }
    let Some(recomposition) =
        recompose_decoder_exact_ls(evaluator, decoder, grid_phi.view(), new_grid.view())?
    else {
        return Ok(None);
    };

    // ── Canonical per-row coordinates t̃_i = φ_θ(t_i) ────────────────────────
    let mut new_row_coords = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let mapped =
            SphereBoostFlowBasis::map_point(&theta, [row_coords[[row, 0]], row_coords[[row, 1]]]);
        new_row_coords[[row, 0]] = mapped[0];
        new_row_coords[[row, 1]] = mapped[1];
    }

    Ok(Some(SphereIsometryFlowReparameterization {
        new_row_coords,
        new_decoder: recomposition.new_decoder,
        decoder_transport: recomposition.transport,
        flow_theta: theta,
        defect_initial,
        defect_final: minimization.defect_final,
        min_flow_jacobian_det,
        recomposition_residual: recomposition.recomposition_residual,
    }))
}

/// Outcome of the sphere boost-defect minimizer.
struct SphereFlowMinimization {
    theta: Vec<f64>,
    defect_initial: f64,
    defect_final: f64,
}

/// Per-row whitened flow Jacobian `Ã_i = L(φ(t_i)) · Dφ_θ(t_i)` (row-major
/// `[a00, a01, a10, a11]`) and the round-sphere defect at `θ`.
///
/// `Dφ_θ(t) = I + Σ_k θ_k Dv_k(t)`; whitening by `L(lat̃) = diag(1, cos lat̃)`
/// at the moved latitude `lat̃ = φ_θ(t)[0]` realizes the reference metric (see
/// [`sphere_isometry_flow_reparameterization`] for the reduction). Returns
/// `None` when the profiled scale degenerates or any whitened latitude leaves
/// the valid band (`cos lat̃ ≤ 0`).
fn sphere_eval_boost_defect(
    theta: &[f64],
    row_coords: ArrayView2<'_, f64>,
    ghat: &[[f64; 3]],
    ghat_norm_sq: f64,
) -> Option<FlowObjectiveState> {
    let n = row_coords.nrows();
    let mut a_rows: Vec<[f64; 4]> = Vec::with_capacity(n);
    let mut cross = 0.0_f64;
    for row in 0..n {
        let t = [row_coords[[row, 0]], row_coords[[row, 1]]];
        let jac = SphereBoostFlowBasis::mode_jacobians(t);
        // Dφ = I + Σ θ_k Dv_k.
        let mut dphi = [[1.0_f64, 0.0], [0.0, 1.0]];
        for (k, dv) in jac.iter().enumerate() {
            dphi[0][0] += theta[k] * dv[0][0];
            dphi[0][1] += theta[k] * dv[0][1];
            dphi[1][0] += theta[k] * dv[1][0];
            dphi[1][1] += theta[k] * dv[1][1];
        }
        // Whiten by L(lat̃) at the moved latitude.
        let mapped = SphereBoostFlowBasis::map_point(theta, t);
        let cos_lat_new = mapped[0].cos();
        // Guard against the `1/cos lat` singularity in the lon component of the
        // whitened Jacobian `Ã = L(lat̃)·Dφ`.  `cos(π/2)` in f64 is ~6.1e-17, not
        // exactly 0, so a bare `> 0.0` lets a pole-adjacent row through and then
        // multiplies `dphi[1, .]` by an effectively-zero factor, corrupting `Ã` and
        // the GN system.  Mirror the floor used in `sphere_chart_isometry_defect`
        // (`POLE_COS2_FLOOR = 1e-12` on cos²lat ↔ `|cos lat| > 1e-6`).
        const SPHERE_EVAL_COS_FLOOR: f64 = 1.0e-6;
        if !(cos_lat_new.is_finite() && cos_lat_new > SPHERE_EVAL_COS_FLOOR) {
            return None;
        }
        let a = [
            dphi[0][0],
            dphi[0][1],
            cos_lat_new * dphi[1][0],
            cos_lat_new * dphi[1][1],
        ];
        a_rows.push(a);
    }
    for (a, g) in a_rows.iter().zip(ghat.iter()) {
        let m00 = a[0] * a[0] + a[2] * a[2];
        let m11 = a[1] * a[1] + a[3] * a[3];
        let m01 = a[0] * a[1] + a[2] * a[3];
        cross += m00 * g[0] + m11 * g[1] + 2.0 * m01 * g[2];
    }
    let scale = cross / ghat_norm_sq;
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let mut defect = 0.0_f64;
    for (a, g) in a_rows.iter().zip(ghat.iter()) {
        let m00 = a[0] * a[0] + a[2] * a[2];
        let m11 = a[1] * a[1] + a[3] * a[3];
        let m01 = a[0] * a[1] + a[2] * a[3];
        let r00 = m00 - scale * g[0];
        let r11 = m11 - scale * g[1];
        let r01 = m01 - scale * g[2];
        defect += r00 * r00 + r11 * r11 + 2.0 * r01 * r01;
    }
    if !defect.is_finite() {
        return None;
    }
    Some(FlowObjectiveState {
        defect,
        scale,
        a_rows,
    })
}

/// Damped Gauss–Newton on the 3 sphere conformal-boost coefficients. The
/// residual `svec(Ã_iᵀÃ_i − s·Ĝ_i)` and its `θ`-Jacobian are formed by central
/// finite differences of the whitened per-row `Ã_i` (the whitening composes the
/// boost Jacobian with the moved-latitude `cos`, so an analytic `∂Ã/∂θ` is a
/// chain rule the FD evaluates exactly to step order) — the SAME Levenberg
/// damping + strict-descent + diffeomorphism accept test as the torus / patch
/// core. Starts at `θ = 0` (`Dφ = I`); only fold-free strict-descent candidates
/// are accepted. Returns `None` (honest skip) when the identity chart is already
/// round-isometric or no strict improvement is reachable.
fn sphere_minimize_boost_defect(
    ghat: &[[f64; 3]],
    ghat_norm_sq: f64,
    row_coords: ArrayView2<'_, f64>,
    q: usize,
    lat_lo: f64,
    lat_hi: f64,
) -> Option<SphereFlowMinimization> {
    let n = row_coords.nrows();
    let mut theta = vec![0.0_f64; q];
    let mut state = sphere_eval_boost_defect(&theta, row_coords, ghat, ghat_norm_sq)?;
    let defect_initial = state.defect;
    if !(defect_initial > 0.0) {
        return None;
    }
    let sqrt2 = std::f64::consts::SQRT_2;
    // FD-OK: FD-audit certificate of the analytic chart-mode Jacobian (central-difference residual Jacobian for the GN flow)
    let fd_h = 1.0e-6_f64; // fd-ok: FD Jacobian for sphere-boost Gauss-Newton; analytic Jacobian requires per-row product differentials, FD bounded by convergence guard
    let mut lambda = 1.0e-4_f64;
    let mut any_accepted = false;
    for iteration in 0..TORUS_FLOW_GN_MAX_ITERS {
        if iteration + 1 == TORUS_FLOW_GN_MAX_ITERS {
            break;
        }
        // Residual r(θ) = svec(Ã_iᵀÃ_i − s·Ĝ_i), and its θ-Jacobian by central
        // FD of the whitened per-row residual (scale s held at its profiled
        // value — envelope theorem, exactly as the analytic torus core).
        let mut jmat = Array2::<f64>::zeros((3 * n, q));
        let mut rcol = Array2::<f64>::zeros((3 * n, 1));
        let scale = state.scale;
        for (i, (a, g)) in state.a_rows.iter().zip(ghat.iter()).enumerate() {
            let m00 = a[0] * a[0] + a[2] * a[2];
            let m11 = a[1] * a[1] + a[3] * a[3];
            let m01 = a[0] * a[1] + a[2] * a[3];
            rcol[[3 * i, 0]] = m00 - scale * g[0];
            rcol[[3 * i + 1, 0]] = m11 - scale * g[1];
            rcol[[3 * i + 2, 0]] = sqrt2 * (m01 - scale * g[2]);
        }
        for k in 0..q {
            let mut tp = theta.clone();
            let mut tm = theta.clone();
            tp[k] += fd_h; // fd-ok: FD Jacobian for sphere-boost Gauss-Newton; analytic Jacobian requires per-row product differentials, FD bounded by convergence guard
            tm[k] -= fd_h; // fd-ok: FD Jacobian for sphere-boost Gauss-Newton; analytic Jacobian requires per-row product differentials, FD bounded by convergence guard
            let sp = sphere_eval_boost_defect(&tp, row_coords, ghat, ghat_norm_sq);
            let sm = sphere_eval_boost_defect(&tm, row_coords, ghat, ghat_norm_sq);
            let (Some(sp), Some(sm)) = (sp, sm) else {
                // A perturbation left the valid band — abandon this GN step.
                return if any_accepted {
                    Some(SphereFlowMinimization {
                        theta,
                        defect_initial,
                        defect_final: state.defect,
                    })
                } else {
                    None
                };
            };
            for (i, (ap, am)) in sp.a_rows.iter().zip(sm.a_rows.iter()).enumerate() {
                let mp00 = ap[0] * ap[0] + ap[2] * ap[2] - scale * ghat[i][0];
                let mp11 = ap[1] * ap[1] + ap[3] * ap[3] - scale * ghat[i][1];
                let mp01 = ap[0] * ap[1] + ap[2] * ap[3] - scale * ghat[i][2];
                let mm00 = am[0] * am[0] + am[2] * am[2] - scale * ghat[i][0];
                let mm11 = am[1] * am[1] + am[3] * am[3] - scale * ghat[i][1];
                let mm01 = am[0] * am[1] + am[2] * am[3] - scale * ghat[i][2];
                jmat[[3 * i, k]] = (mp00 - mm00) / (2.0 * fd_h); // fd-ok: FD Jacobian for sphere-boost Gauss-Newton; analytic Jacobian requires per-row product differentials, FD bounded by convergence guard
                jmat[[3 * i + 1, k]] = (mp11 - mm11) / (2.0 * fd_h); // fd-ok: FD Jacobian for sphere-boost Gauss-Newton; analytic Jacobian requires per-row product differentials, FD bounded by convergence guard
                jmat[[3 * i + 2, k]] = sqrt2 * (mp01 - mm01) / (2.0 * fd_h); // fd-ok: FD Jacobian for sphere-boost Gauss-Newton; analytic Jacobian requires per-row product differentials, FD bounded by convergence guard
            }
        }
        // END-FD-OK
        let jtj = fast_ata(&jmat);
        let jtr = fast_atb(&jmat, &rcol);

        let mut rejects = 0usize;
        let mut accepted_step = false;
        let mut converged = false;
        let mut step_norm_sq = 0.0_f64;
        while rejects < TORUS_FLOW_GN_MAX_REJECTS {
            let mut damped = jtj.clone();
            for d in 0..q {
                damped[[d, d]] += lambda * (1.0 + jtj[[d, d]]);
            }
            let factor = match damped.cholesky(FaerSide::Lower) {
                Ok(factor) => factor,
                Err(_) => {
                    lambda *= 10.0;
                    rejects += 1;
                    continue;
                }
            };
            let mut neg_jtr = jtr.clone();
            neg_jtr.mapv_inplace(|v| -v);
            let delta = factor.solve_mat(&neg_jtr);
            let mut candidate = theta.clone();
            step_norm_sq = 0.0;
            for k in 0..q {
                candidate[k] += delta[[k, 0]];
                step_norm_sq += delta[[k, 0]] * delta[[k, 0]];
            }
            let folded = SphereBoostFlowBasis::min_jacobian_det_on_band(&candidate, lat_lo, lat_hi)
                <= SPHERE_FLOW_DIFFEO_MIN_DET;
            let candidate_state = if folded {
                None
            } else {
                sphere_eval_boost_defect(&candidate, row_coords, ghat, ghat_norm_sq)
            };
            match candidate_state {
                Some(next) if next.defect < state.defect => {
                    let improvement = state.defect - next.defect;
                    theta = candidate;
                    state = next;
                    any_accepted = true;
                    accepted_step = true;
                    lambda = (lambda / 10.0).max(1.0e-12);
                    if improvement <= 1.0e-14 * (1.0 + state.defect) {
                        converged = true;
                    }
                    break;
                }
                Some(..) | None => {
                    lambda *= 10.0;
                    rejects += 1;
                }
            }
        }
        if !accepted_step || converged {
            break;
        }
        let theta_norm_sq: f64 = theta.iter().map(|v| v * v).sum();
        if step_norm_sq <= 1.0e-24 * (1.0 + theta_norm_sq) {
            break;
        }
    }
    if !any_accepted || !(state.defect < defect_initial) {
        return None;
    }
    Some(SphereFlowMinimization {
        theta,
        defect_initial,
        defect_final: state.defect,
    })
}

/// Scale-invariant isometry defect of a fitted `d = 2` **sphere** atom's
/// `(lat, lon)` chart against the round-sphere reference metric (#1019 stage 2,
/// sphere arm).
///
/// This is the certified objective the sphere flow-pin
/// ([`sphere_isometry_flow_reparameterization`]) descends; it is also exposed
/// on its own as the read-only acceptance measurement (the issue's "defect
/// within 10% of optimum" quantity), so a chart that is already round-isometric
/// (a true `O(3)` representative) scores `≈ 0` and a warped chart scores large.
///
/// # The defect functional
///
/// For the round sphere, the `(lat, lon)` chart's reference first fundamental
/// form is `g_ref(lat) = diag(1, cos²lat)` (a unit-radius sphere; the lon
/// circumference shrinks as `cos lat`). The fitted decoder's pullback metric at
/// row `i` is `G_i = J(t_i)ᵀ J(t_i)` from the exact `(Φ, ∂Φ)` jet, exactly as
/// in the torus path. The chart is isometric to the round sphere up to a global
/// scale `c` iff `G_i ≡ c · g_ref(lat_i)` for all `i`. Measuring the residual
/// with `c` analytically profiled (the exact argmin over the global scale),
///
/// ```text
/// E = Σ_i ‖ Ĝ_i − c · ĝ_ref,i ‖²_F ,
/// c = Σ_i ⟨Ĝ_i, ĝ_ref,i⟩_F / Σ_i ‖ĝ_ref,i‖²_F ,
/// ```
///
/// where both metrics are normalized by the geometric-mean fitted metric scale
/// `ḡ = exp(mean_i ½ log det G_i)` so the defect is scale-invariant (a chart
/// isometric up to ANY global scale scores 0). The Frobenius norm on symmetric
/// `2×2` matrices uses the `[m00, m11, m01]` storage with the off-diagonal
/// weighted by 2 (matching the torus path).
///
/// Returns `None` on a degenerate chart (empty/non-finite, rank-deficient
/// pullback metric anywhere, or a degenerate profiled scale) — an honest
/// refusal, never a fabricated zero. The returned defect is the scale-invariant
/// `E` above; `0` means the fitted chart is already a round-isometric `O(3)`
/// representative.
pub fn sphere_chart_isometry_defect(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView2<'_, f64>,
) -> Result<Option<f64>, String> {
    let n = row_coords.nrows();
    let m = decoder.nrows();
    let p = decoder.ncols();
    if row_coords.ncols() != 2 {
        return Err(format!(
            "sphere_chart_isometry_defect: expected (n, 2) row coordinates; got {:?}",
            row_coords.dim()
        ));
    }
    if n == 0 || m == 0 || p == 0 {
        return Ok(None);
    }
    for &t in row_coords.iter() {
        if !t.is_finite() {
            return Ok(None);
        }
    }

    // Fitted pullback metric G_i = J(t_i)ᵀJ(t_i) from the exact jet — identical
    // extraction to the torus / patch paths (axis 0 = lat, axis 1 = lon), via
    // the shared helper. The sphere reference below differs (diag(1, cos²lat)
    // vs flat I), so only the extraction is shared, not the normalization.
    let Some((g_rows, g_bar)) = extract_pullback_metric_d2(
        "sphere_chart_isometry_defect",
        evaluator,
        decoder,
        row_coords,
    )?
    else {
        return Ok(None);
    };

    // Reference metric ĝ_ref,i = diag(1, cos²lat_i) (round sphere, lat = axis 0).
    // Both G and g_ref are normalized by ḡ; g_ref carries no fitted scale so the
    // profiled `c` absorbs the absolute size. The reference's own determinant is
    // cos²lat, which can vanish near the poles — guard it so a pole-adjacent row
    // does not inject a degenerate reference direction.
    let mut ghat: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut gref: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut gref_norm_sq = 0.0_f64;
    let mut cross = 0.0_f64;
    for (row, g) in g_rows.iter().enumerate() {
        let lat = row_coords[[row, 0]];
        let cos_lat = lat.cos();
        let r11 = cos_lat * cos_lat;
        // A pole-adjacent reference column (cos lat → 0) carries no transverse
        // metric content; treating it as part of the defect would falsely
        // reward squeezing the lon direction. Refuse charts sitting on the
        // pole singularity rather than fabricate a defect there.
        //
        // NB: `cos(π/2)` is ~6.1e-17 in f64, not exactly 0, so an exactly-on-pole
        // row gives `r11 = cos²lat ≈ 3.7e-33` — finite and strictly positive. A
        // bare `r11 > 0.0` therefore lets it through; floor against `POLE_COS2_FLOOR`
        // (cos lat within ~1e-6 of a pole) so the singular row is honestly refused.
        const POLE_COS2_FLOOR: f64 = 1e-12;
        if !(r11.is_finite() && r11 > POLE_COS2_FLOOR) {
            return Ok(None);
        }
        let h = [g[0] / g_bar, g[1] / g_bar, g[2] / g_bar];
        let r = [1.0_f64, r11, 0.0_f64];
        cross += h[0] * r[0] + h[1] * r[1] + 2.0 * h[2] * r[2];
        gref_norm_sq += r[0] * r[0] + r[1] * r[1] + 2.0 * r[2] * r[2];
        ghat.push(h);
        gref.push(r);
    }
    if !(gref_norm_sq.is_finite() && gref_norm_sq > 0.0) {
        return Ok(None);
    }
    let c = cross / gref_norm_sq;
    if !(c.is_finite() && c > 0.0) {
        return Ok(None);
    }
    let mut defect = 0.0_f64;
    for (h, r) in ghat.iter().zip(gref.iter()) {
        let r00 = h[0] - c * r[0];
        let r11 = h[1] - c * r[1];
        let r01 = h[2] - c * r[2];
        defect += r00 * r00 + r11 * r11 + 2.0 * r01 * r01;
    }
    if !defect.is_finite() {
        return Ok(None);
    }
    Ok(Some(defect))
}

/// Total fitted turning `Θ = ∫ κ ds` of a `d = 1` atom's decoded curve (#1026).
///
/// # Why this is the discriminating measurement
///
/// The hybrid-vs-shatter question — does a curved atom genuinely earn its
/// curvature, or is it just more linear directions in disguise — is answered by
/// pairing each atom's EV contribution with its fitted **turning** `Θ`
/// (integrated curvature). A linear SAE shatters a curved feature of total
/// turning `Θ` into `N(ε) ≈ Θ/(2√(2ε))` rank-1 atoms at relative error `ε`
/// (radius cancels — relative error is scale-free), so the curved win is
/// concentrated on high-`Θ` features and vanishes as `Θ → 0`. A near-linear
/// atom (`Θ ≈ 0`) contributing EV is a linear direction wearing a curved basis;
/// a high-`Θ` atom contributing EV is a genuine curved family. Reporting EV-per-
/// atom vs `Θ` (not EV vs K) directly shows which.
///
/// # The functional
///
/// For the decoded curve `γ(t) = Φ(t)·B` the unsigned total curvature is
///
/// ```text
/// Θ = ∫ κ(t) ‖γ'(t)‖ dt ,   κ = ‖γ'(t) ∧ γ''(t)‖ / ‖γ'(t)‖³ ,
/// ```
///
/// so `Θ = ∫ ‖γ' ∧ γ''‖ / ‖γ'‖² dt`, where the wedge norm in `ℝ^p` is the
/// parallelogram area `‖γ'∧γ''‖ = √(‖γ'‖²‖γ''‖² − ⟨γ',γ''⟩²)` (Lagrange). `Θ`
/// is reparameterization-invariant (it is an integral of `κ ds`), so it is the
/// honest chart-free geometric content. Integrated by Simpson's rule over a
/// uniform grid spanning the fitted coordinate range from the exact `(γ', γ'')`
/// jets. Units: radians of total turning (a full hue circle ≈ `2π`).
///
/// Returns `None` on a degenerate atom (no rows, no second jet, a collapsed
/// coordinate range, or a non-finite integrand) — an honest refusal, never a
/// fabricated number.
pub fn d1_atom_fitted_turning(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    row_coords: ArrayView1<'_, f64>,
) -> Result<Option<f64>, String> {
    let m = decoder.nrows();
    let p = decoder.ncols();
    if m == 0 || p == 0 || row_coords.is_empty() {
        return Ok(None);
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &t in row_coords.iter() {
        if !t.is_finite() {
            return Ok(None);
        }
        lo = lo.min(t);
        hi = hi.max(t);
    }
    if !(hi > lo) {
        return Ok(None);
    }
    let cells = TURNING_QUADRATURE_CELLS;
    let nodes = 2 * cells + 1; // node, midpoint, node per cell (Simpson)
    let h = (hi - lo) / (nodes - 1) as f64;
    let mut grid = Array2::<f64>::zeros((nodes, 1));
    for (i, mut row) in grid.outer_iter_mut().enumerate() {
        row[0] = lo + h * i as f64;
    }
    let (_phi, jet) = evaluator.evaluate(grid.view())?;
    if jet.dim() != (nodes, m, 1) {
        return Err(format!(
            "d1_atom_fitted_turning: evaluator returned jet {:?}; expected ({nodes}, {m}, 1)",
            jet.dim()
        ));
    }
    let Some(hess_result) = evaluator.second_jet_dyn(grid.view()) else {
        // No analytic second jet for this basis family → no honest turning.
        return Ok(None);
    };
    let hess = hess_result?;
    if hess.dim() != (nodes, m, 1, 1) {
        return Err(format!(
            "d1_atom_fitted_turning: second_jet returned {:?}; expected ({nodes}, {m}, 1, 1)",
            hess.dim()
        ));
    }
    // Per-node curvature integrand f(t) = ‖γ'∧γ''‖/‖γ'‖² (= κ·‖γ'‖).
    let mut integrand = vec![0.0_f64; nodes];
    // Distinguish a GLOBALLY constant image (every node stationary → a degenerate
    // point with total turning exactly 0) from a curve that is stationary at only
    // SOME nodes (a genuine cusp whose turning is ill-defined there, refused).
    let mut any_moving = false;
    let mut any_stationary = false;
    let mut g1 = vec![0.0_f64; p];
    let mut g2 = vec![0.0_f64; p];
    for node in 0..nodes {
        for slot in g1.iter_mut() {
            *slot = 0.0;
        }
        for slot in g2.iter_mut() {
            *slot = 0.0;
        }
        for bm in 0..m {
            let d1 = jet[[node, bm, 0]];
            let d2 = hess[[node, bm, 0, 0]];
            if d1 == 0.0 && d2 == 0.0 {
                continue;
            }
            for j in 0..p {
                let b = decoder[[bm, j]];
                g1[j] += d1 * b;
                g2[j] += d2 * b;
            }
        }
        let mut n1 = 0.0_f64; // ‖γ'‖²
        let mut n2 = 0.0_f64; // ‖γ''‖²
        let mut dot = 0.0_f64; // ⟨γ',γ''⟩
        for j in 0..p {
            n1 += g1[j] * g1[j];
            n2 += g2[j] * g2[j];
            dot += g1[j] * g2[j];
        }
        if !(n1 > 0.0) {
            // Zero speed at this node: the arc-length measure `ds = ‖γ'‖dt`
            // vanishes here, so this node contributes zero turning regardless of
            // κ. Record it and continue; the all-stationary (constant image) vs
            // mixed (cusp) cases are resolved after the loop.
            any_stationary = true;
            integrand[node] = 0.0;
            continue;
        }
        any_moving = true;
        // Wedge norm² = ‖γ'‖²‖γ''‖² − ⟨γ',γ''⟩² (Lagrange identity); clamp tiny
        // negative round-off to 0 before the sqrt.
        let raw_wedge_sq = n1 * n2 - dot * dot;
        let roundoff_floor = 64.0 * f64::EPSILON * (n1 * n2).abs().max(dot.abs() * dot.abs());
        let wedge_sq = if raw_wedge_sq <= roundoff_floor {
            0.0
        } else {
            raw_wedge_sq
        };
        integrand[node] = wedge_sq.sqrt() / n1;
        if !integrand[node].is_finite() {
            return Ok(None);
        }
    }
    if !any_moving {
        // The decoded image never moves over the coordinate span: a degenerate
        // single point (e.g. a `d = 1` atom straightened to its constant DC
        // component). A point has no arc to turn through, so its total turning is
        // exactly 0 — the ultimate linear-tail signature — not "undefined".
        return Ok(Some(0.0));
    }
    if any_stationary {
        // Stationary at SOME nodes but moving at others: a cusp where the turning
        // integrand is genuinely ill-defined. Refuse rather than under-count the
        // sharp turn (the historical conservative behavior for a partial cusp).
        return Ok(None);
    }
    // Composite Simpson over `cells` cells: each cell [2i, 2i+1, 2i+2] gets
    // (h/3)(f0 + 4 f_mid + f1).
    let mut theta = 0.0_f64;
    for cell in 0..cells {
        let f0 = integrand[2 * cell];
        let fm = integrand[2 * cell + 1];
        let f1 = integrand[2 * cell + 2];
        theta += h / 3.0 * (f0 + 4.0 * fm + f1);
    }
    if !(theta.is_finite() && theta >= 0.0) {
        return Ok(None);
    }
    Ok(Some(theta))
}

#[cfg(test)]
mod patch_flow_tests {
    use super::*;
    use ndarray::{Array2, Array3, ArrayView2};

    /// Mock free-patch evaluator with the affine basis `Φ(t) = [1, t₀, t₁]`
    /// (`m = 3`) and its exact jet. The decoder plants the affine decoded image
    /// `γ(t) = M·t` (a constant anisotropic warp), so the fitted pullback metric
    /// is the constant `G = MᵀM` everywhere — an anisotropically warped flat
    /// chart. The affine basis is closed under affine reparameterization, so the
    /// linear flow modes can transport the image exactly (the recomposition gate
    /// passes), and the minimizer recovers `Dφ ≈ M⁻¹` (up to a flat O(2)
    /// isometry), driving the anisotropy defect to ≈ 0.
    #[derive(Debug)]
    struct MockPatchEvaluator;

    impl SaeBasisEvaluator for MockPatchEvaluator {
        fn evaluate(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            let n = coords.nrows();
            let mut phi = Array2::<f64>::zeros((n, 3));
            let mut jet = Array3::<f64>::zeros((n, 3, 2));
            for row in 0..n {
                let t0 = coords[[row, 0]];
                let t1 = coords[[row, 1]];
                phi[[row, 0]] = 1.0;
                phi[[row, 1]] = t0;
                phi[[row, 2]] = t1;
                // ∂Φ/∂t₀ = [0, 1, 0]; ∂Φ/∂t₁ = [0, 0, 1].
                jet[[row, 1, 0]] = 1.0;
                jet[[row, 2, 1]] = 1.0;
            }
            Ok((phi, jet))
        }

        fn second_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array4<f64>, String>> {
            // Affine basis: second jet is exactly zero. Return it honestly
            // (shape (n, 3, 2, 2)) after validating the coordinate width.
            if coords.ncols() != 2 {
                return Some(Err(format!(
                    "MockPatchEvaluator::second_jet_dyn: expected 2 cols, got {}",
                    coords.ncols()
                )));
            }
            Some(Ok(ndarray::Array4::<f64>::zeros((coords.nrows(), 3, 2, 2))))
        }

        fn third_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array5<f64>, String>> {
            if coords.ncols() != 2 {
                return Some(Err(format!(
                    "MockPatchEvaluator::third_jet_dyn: expected 2 cols, got {}",
                    coords.ncols()
                )));
            }
            Some(Ok(ndarray::Array5::<f64>::zeros((
                coords.nrows(),
                3,
                2,
                2,
                2,
            ))))
        }
    }

    /// Decoder for `γ(t) = M·t`: basis `[1, t₀, t₁]` ↦ outputs `(γ₀, γ₁)`.
    /// Row 0 (const) = 0; row 1 (t₀) = (M₀₀, M₁₀); row 2 (t₁) = (M₀₁, M₁₁).
    fn warp_decoder(m: [[f64; 2]; 2]) -> Array2<f64> {
        let mut b = Array2::<f64>::zeros((3, 2));
        b[[1, 0]] = m[0][0];
        b[[1, 1]] = m[1][0];
        b[[2, 0]] = m[0][1];
        b[[2, 1]] = m[1][1];
        b
    }

    /// A grid of fitted coordinates over the patch box `[0, 1]²`.
    fn patch_coords() -> Array2<f64> {
        let g = 9usize;
        let mut c = Array2::<f64>::zeros((g * g, 2));
        for i in 0..g {
            for j in 0..g {
                let row = i * g + j;
                c[[row, 0]] = i as f64 / (g - 1) as f64;
                c[[row, 1]] = j as f64 / (g - 1) as f64;
            }
        }
        c
    }

    /// Scale-invariant anisotropy defect of a constant pullback metric `G`
    /// against the flat reference — the optimum the minimizer chases is 0.
    fn flat_defect_of_constant_metric(g: [f64; 3], n: usize) -> f64 {
        // ḡ = sqrt(det G); Ĝ = G/ḡ; profile c against I; sum over n identical
        // rows. With one constant metric the per-row defect is identical.
        let det = g[0] * g[1] - g[2] * g[2];
        let g_bar = det.sqrt();
        let h = [g[0] / g_bar, g[1] / g_bar, g[2] / g_bar];
        // c = <Ĝ, I> / <I, I> = (h00 + h11) / 2 ; residual = Ĝ − c·I.
        let c = 0.5 * (h[0] + h[1]);
        let r00 = h[0] - c;
        let r11 = h[1] - c;
        let r01 = h[2];
        n as f64 * (r00 * r00 + r11 * r11 + 2.0 * r01 * r01)
    }

    #[test]
    fn planted_warped_patch_recovers_uniform_speed_coords() {
        // A deliberately anisotropic + sheared affine warp M: the fitted chart
        // stretches axis 0 by 1.6, axis 1 by 0.8, with a 0.5 shear.
        let m = [[1.6, 0.5], [0.0, 0.8]];
        let ev = MockPatchEvaluator;
        let decoder = warp_decoder(m);
        let coords = patch_coords();
        let n = coords.nrows();

        // Pullback metric G = MᵀM (constant over the patch).
        let g00 = m[0][0] * m[0][0] + m[1][0] * m[1][0];
        let g11 = m[0][1] * m[0][1] + m[1][1] * m[1][1];
        let g01 = m[0][0] * m[0][1] + m[1][0] * m[1][1];
        let defect_initial = flat_defect_of_constant_metric([g00, g11, g01], n);
        assert!(
            defect_initial > 1e-2,
            "the planted anisotropic warp must start with a sizeable defect; got {defect_initial:.3e}"
        );

        let repar = patch_isometry_flow_reparameterization(&ev, decoder.view(), coords.view())
            .expect("patch reparameterization must evaluate")
            .expect("a warped patch with a global flow basis must canonicalize");

        // Acceptance (#1019): the canonical chart recovers uniform-speed coords
        // — the residual defect is within 10% of the optimum (0). The optimum is
        // exactly 0 (a flat patch IS isometric to itself), so the bar is a small
        // absolute fraction of the initial defect.
        assert!(
            repar.defect_final <= 0.10 * defect_initial,
            "canonicalization must drive the anisotropy defect to within 10% of the optimum; \
             initial {defect_initial:.3e}, final {:.3e}",
            repar.defect_final
        );
        // The defect-final the struct reports must match what the optimizer
        // descended to (strict improvement over the identity flow).
        assert!(
            repar.defect_final < repar.defect_initial,
            "the pass must report a strict improvement; initial {:.3e}, final {:.3e}",
            repar.defect_initial,
            repar.defect_final
        );
        // The canonical chart is a genuine diffeomorphism (guard cleared).
        assert!(
            repar.min_flow_jacobian_det > PATCH_FLOW_DIFFEO_MIN_DET,
            "the canonical flow must be fold-free; min det {:.3e}",
            repar.min_flow_jacobian_det
        );
        // Image frozen: the recomposition residual is within the honest gate.
        assert!(
            repar.recomposition_residual <= CHART_RECOMPOSITION_REL_TOL,
            "the decoded image must be reproduced within the recomposition tolerance; got {:.3e}",
            repar.recomposition_residual
        );
    }

    #[test]
    fn already_uniform_patch_is_left_as_fitted() {
        // M = I: the fitted chart is already uniform-speed (defect 0), so the
        // minimizer finds no strict improvement and honestly skips.
        let ev = MockPatchEvaluator;
        let decoder = warp_decoder([[1.0, 0.0], [0.0, 1.0]]);
        let coords = patch_coords();
        let out = patch_isometry_flow_reparameterization(&ev, decoder.view(), coords.view())
            .expect("patch reparameterization must evaluate");
        assert!(
            out.is_none(),
            "an already-uniform patch chart must be left as fitted (honest skip), got Some"
        );
    }

    #[test]
    fn collapsed_patch_axis_is_refused() {
        // All rows share a single t₁ value: the patch has no extent on axis 1,
        // so there is no honest flat chart — the basis builder refuses and the
        // function returns None rather than erroring.
        let ev = MockPatchEvaluator;
        let decoder = warp_decoder([[1.3, 0.0], [0.0, 0.9]]);
        let mut coords = patch_coords();
        for row in 0..coords.nrows() {
            coords[[row, 1]] = 0.5;
        }
        let out = patch_isometry_flow_reparameterization(&ev, decoder.view(), coords.view())
            .expect("patch reparameterization must evaluate");
        assert!(
            out.is_none(),
            "a patch collapsed along one axis must be refused (None), got Some"
        );
    }

    #[test]
    fn free_patch_flow_basis_layout_and_jacobian_at_identity() {
        let basis = FreePatchFlowBasis::new([0.0, 0.0], [1.0, 1.0]).expect("patch basis");
        // 2 components × 2 monomials (deg 1: (1,0),(0,1)) = 4 coefficients.
        assert_eq!(basis.dim(), 4);
        assert_eq!(basis.mode_layout().len(), 4);
        // θ = 0 ⇒ Dφ = I everywhere ⇒ min det = 1.
        let theta = vec![0.0_f64; basis.dim()];
        let det = basis.min_jacobian_det_on_grid(&theta);
        assert!(
            (det - 1.0).abs() < 1e-12,
            "identity flow has det Dφ ≡ 1; got {det}"
        );
        // The first two modes of component 0 are the linear fields (1,0),(0,1).
        let layout = basis.mode_layout();
        assert_eq!(layout[0].component, 0);
        assert_eq!(layout[0].exps, (1, 0));
        assert_eq!(layout[1].exps, (0, 1));
    }

    /// Finite-difference check of the analytic mode gradients (the `grad` the
    /// Gauss–Newton Jacobian is built from) against the monomial values.
    #[test]
    fn free_patch_mode_gradients_match_finite_difference() {
        let basis = FreePatchFlowBasis::new([-0.5, 0.2], [1.5, 2.2]).expect("patch basis");
        let t = [0.3, 1.1];
        let eps = 1e-6;
        let base = basis.mode_samples(t);
        let plus0 = basis.mode_samples([t[0] + eps, t[1]]);
        let plus1 = basis.mode_samples([t[0], t[1] + eps]);
        for k in 0..base.len() {
            let fd0 = (plus0[k].value - base[k].value) / eps;
            let fd1 = (plus1[k].value - base[k].value) / eps;
            assert!(
                (fd0 - base[k].grad[0]).abs() < 1e-4,
                "mode {k} ∂/∂t₀ FD {fd0} vs analytic {}",
                base[k].grad[0]
            );
            assert!(
                (fd1 - base[k].grad[1]).abs() < 1e-4,
                "mode {k} ∂/∂t₁ FD {fd1} vs analytic {}",
                base[k].grad[1]
            );
        }
    }
}

#[cfg(test)]
mod sphere_defect_tests {
    use super::*;
    use ndarray::{Array2, Array3, ArrayView2};

    /// Mock sphere-chart evaluator: `m = p = 2`, identity decoder, with a jet
    /// whose per-row decoded tangents are `∂γ/∂lat = (1, 0)` and
    /// `∂γ/∂lon = (0, warp·cos lat)`. With `warp = 1` the pullback metric is
    /// exactly the round-sphere reference `diag(1, cos²lat)` (defect 0); any
    /// `warp ≠ 1` rescales the lon direction uniformly — still a global rescale
    /// of the lon column, but NOT a global rescale of the whole metric, so it
    /// registers as a genuine anisotropic defect.
    #[derive(Debug)]
    struct MockSphereEvaluator {
        warp: f64,
    }

    impl SaeBasisEvaluator for MockSphereEvaluator {
        fn evaluate(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            let n = coords.nrows();
            // Basis Φ is unused by the defect (only the jet enters G); return a
            // well-formed (n, 2) zero basis.
            let phi = Array2::<f64>::zeros((n, 2));
            let mut jet = Array3::<f64>::zeros((n, 2, 2));
            for row in 0..n {
                let lat = coords[[row, 0]];
                // basis col 0 carries the lat tangent (1, 0): ∂/∂lat = 1 on
                // output 0; basis col 1 carries the lon tangent
                // (0, warp·cos lat): ∂/∂lon = warp·cos lat on output 1.
                jet[[row, 0, 0]] = 1.0; // d(col0)/d(lat)
                jet[[row, 1, 1]] = self.warp * lat.cos(); // d(col1)/d(lon)
            }
            Ok((phi, jet))
        }

        // This mock supplies only the first jet that the chart-defect test
        // exercises; it carries no analytic second/third jet, so it declares
        // that capability absent (`None`) per the trait contract rather than
        // fabricating one — after validating the (lat, lon) coordinate shape
        // it is contracted on, mirroring the sanctioned `TestPeriodicEvaluator`
        // higher-jet stubs.
        fn second_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array4<f64>, String>> {
            if coords.ncols() != 2 {
                return Some(Err(format!(
                    "MockSphereEvaluator::second_jet_dyn: expected (lat, lon) coords, got {} cols",
                    coords.ncols()
                )));
            }
            None
        }

        fn third_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array5<f64>, String>> {
            if coords.ncols() != 2 {
                return Some(Err(format!(
                    "MockSphereEvaluator::third_jet_dyn: expected (lat, lon) coords, got {} cols",
                    coords.ncols()
                )));
            }
            None
        }
    }

    fn coords(lats: &[f64]) -> Array2<f64> {
        let n = lats.len();
        let mut c = Array2::<f64>::zeros((n, 2));
        for (i, &lat) in lats.iter().enumerate() {
            c[[i, 0]] = lat;
            c[[i, 1]] = 0.1 * i as f64; // lon, irrelevant to the metric
        }
        c
    }

    #[test]
    fn round_isometric_chart_has_zero_defect() {
        let ev = MockSphereEvaluator { warp: 1.0 };
        let decoder = Array2::<f64>::eye(2);
        let c = coords(&[-0.6, -0.2, 0.0, 0.3, 0.7]);
        let defect = sphere_chart_isometry_defect(&ev, decoder.view(), c.view())
            .expect("defect must evaluate")
            .expect("non-degenerate round chart must return Some");
        assert!(
            defect < 1e-10,
            "a chart whose pullback metric is exactly diag(1, cos²lat) is round-isometric; \
             defect should be ~0, got {defect:.3e}"
        );
    }

    #[test]
    fn warped_chart_has_large_defect() {
        // warp = 2.5 stretches the lon direction by a lat-independent factor,
        // so the pullback metric is diag(1, (2.5·cos lat)²) — NOT a global
        // rescale of diag(1, cos²lat), so the profiled-scale residual is
        // strictly positive.
        let ev = MockSphereEvaluator { warp: 2.5 };
        let decoder = Array2::<f64>::eye(2);
        let c = coords(&[-0.6, -0.2, 0.0, 0.3, 0.7]);
        let defect = sphere_chart_isometry_defect(&ev, decoder.view(), c.view())
            .expect("defect must evaluate")
            .expect("non-degenerate warped chart must return Some");
        assert!(
            defect > 1e-2,
            "an anisotropically warped chart must register a sizeable defect, got {defect:.3e}"
        );
    }

    #[test]
    fn pole_singularity_is_refused_not_fabricated() {
        // A row sitting exactly on the pole (lat = π/2, cos lat = 0) makes the
        // reference metric's lon column vanish; the function must refuse (None)
        // rather than fabricate a defect on the chart singularity.
        let ev = MockSphereEvaluator { warp: 1.0 };
        let decoder = Array2::<f64>::eye(2);
        let base = coords(&[0.0, 0.3]);
        // Append a pole row (lat = π/2).
        let mut c3 = Array2::<f64>::zeros((3, 2));
        c3.slice_mut(ndarray::s![0..2, ..]).assign(&base);
        c3[[2, 0]] = std::f64::consts::FRAC_PI_2;
        let out = sphere_chart_isometry_defect(&ev, decoder.view(), c3.view())
            .expect("defect must evaluate");
        // At the exact pole the decoded lon tangent (cos lat = 0) also collapses
        // the pullback metric (det G = 0), so this refuses via the rank-deficient
        // metric guard — either way an honest None, never a fabricated number.
        assert!(
            out.is_none(),
            "a pole-singular chart row must be refused, got {out:?}"
        );
    }

    /// FD-gate the analytic conformal-boost mode Jacobians `Dv_k(t)` against
    /// central differences of the displacements `v_k(t)` — the exact object the
    /// sphere Gauss–Newton residual derivative is built from.
    #[test]
    fn sphere_boost_mode_jacobians_match_finite_difference() {
        let eps = 1e-6;
        // Two interior points well off the poles (cos lat bounded away from 0).
        for t in [[0.3, 0.7], [-0.6, -1.2]] {
            let jac = SphereBoostFlowBasis::mode_jacobians(t);
            let vp0 = SphereBoostFlowBasis::mode_displacements([t[0] + eps, t[1]]);
            let vm0 = SphereBoostFlowBasis::mode_displacements([t[0] - eps, t[1]]);
            let vp1 = SphereBoostFlowBasis::mode_displacements([t[0], t[1] + eps]);
            let vm1 = SphereBoostFlowBasis::mode_displacements([t[0], t[1] - eps]);
            for k in 0..3 {
                for comp in 0..2 {
                    let fd_dlat = (vp0[k][comp] - vm0[k][comp]) / (2.0 * eps);
                    let fd_dlon = (vp1[k][comp] - vm1[k][comp]) / (2.0 * eps);
                    assert!(
                        (fd_dlat - jac[k][comp][0]).abs() < 1e-5,
                        "boost {k} comp {comp} ∂/∂lat FD {fd_dlat} vs analytic {} at {t:?}",
                        jac[k][comp][0]
                    );
                    assert!(
                        (fd_dlon - jac[k][comp][1]).abs() < 1e-5,
                        "boost {k} comp {comp} ∂/∂lon FD {fd_dlon} vs analytic {} at {t:?}",
                        jac[k][comp][1]
                    );
                }
            }
        }
    }

    /// The zonal boost `K_z = cos(lat) ∂_lat` is pole-free (its only nonzero
    /// component is latitude and carries no `1/cos` factor), and the boost
    /// `[Z, X, Y]` layout is stable.
    #[test]
    fn sphere_boost_layout_and_zonal_is_pole_free() {
        let basis = SphereBoostFlowBasis;
        assert_eq!(basis.dim(), 3);
        assert_eq!(
            basis.mode_layout(),
            [SphereBoostAxis::Z, SphereBoostAxis::X, SphereBoostAxis::Y]
        );
        // K_z displacement is (cos lat, 0): finite for every latitude.
        for lat in [-1.4, -0.3, 0.0, 0.9, 1.4] {
            let disp = SphereBoostFlowBasis::mode_displacements([lat, 0.5]);
            assert!(disp[0][0].is_finite() && disp[0][1] == 0.0);
        }
    }
}

#[cfg(test)]
mod turning_tests {
    use super::*;
    use crate::basis::PeriodicHarmonicEvaluator;
    use ndarray::{Array1, Array2};
    use std::f64::consts::TAU;

    /// A unit circle `γ(t) = (cos 2πt, sin 2πt)` traversed once over `t ∈ [0,1]`
    /// has constant curvature κ = 1 and speed 2π, so the total turning is
    /// `Θ = ∫₀¹ κ·‖γ'‖ dt = 2π`.
    #[test]
    fn full_circle_turning_is_two_pi() {
        let ev = PeriodicHarmonicEvaluator::new(3).expect("3-basis circle");
        // Basis [1, sin2πt, cos2πt]; B maps cos→x (output 0), sin→y (output 1).
        let mut decoder = Array2::<f64>::zeros((3, 2));
        decoder[[2, 0]] = 1.0; // cos -> x
        decoder[[1, 1]] = 1.0; // sin -> y
        // Span the full period [0, 1]; the integral runs over [min, max] of the
        // supplied coordinates, so the endpoint t = 1 must be present to close
        // the loop (the circle's speed 2π is nonzero everywhere, no stationary
        // point at the seam).
        let coords = Array1::from_iter((0..=50).map(|i| i as f64 / 50.0));
        let theta = d1_atom_fitted_turning(&ev, decoder.view(), coords.view())
            .expect("turning must evaluate")
            .expect("a non-degenerate circle must return Some");
        assert!(
            (theta - TAU).abs() < 1e-6,
            "a full unit circle has total turning 2π; got {theta:.9}"
        );
    }

    /// A half circle (`t ∈ [0, 0.5]`) turns through π.
    #[test]
    fn half_circle_turning_is_pi() {
        let ev = PeriodicHarmonicEvaluator::new(3).expect("3-basis circle");
        let mut decoder = Array2::<f64>::zeros((3, 2));
        decoder[[2, 0]] = 1.0;
        decoder[[1, 1]] = 1.0;
        let coords = Array1::from_iter((0..=25).map(|i| 0.5 * i as f64 / 25.0));
        let theta = d1_atom_fitted_turning(&ev, decoder.view(), coords.view())
            .expect("turning must evaluate")
            .expect("a non-degenerate half-circle must return Some");
        assert!(
            (theta - std::f64::consts::PI).abs() < 1e-6,
            "a half circle turns through π; got {theta:.9}"
        );
    }

    /// A straight line (only the constant + one linear-in-image direction, no
    /// genuine curvature) has zero turning — exactly the `Θ → 0` linear-tail
    /// signature the EV-vs-Θ measurement uses to flag a curved atom that is
    /// really just a linear direction. Here the decoder uses a SINGLE harmonic
    /// pair scaled so the image is a 1-D segment (x and y both ∝ cos 2πt), i.e.
    /// the decoded curve lies on a line through the origin → wedge ≡ 0.
    #[test]
    fn straight_line_image_has_zero_turning() {
        let ev = PeriodicHarmonicEvaluator::new(3).expect("3-basis circle");
        let mut decoder = Array2::<f64>::zeros((3, 2));
        // Both outputs ∝ cos 2πt → the image is the line y = 2x, γ collinear with
        // γ', γ'' at every t, so the wedge norm vanishes identically.
        decoder[[2, 0]] = 1.0;
        decoder[[2, 1]] = 2.0;
        // Span a coordinate range strictly inside (0, 0.25) so the speed
        // `‖γ'‖ ∝ |sin 2πt|` never hits the stationary zero at t = 0 (where the
        // turning integrand is genuinely undefined and the function refuses).
        let coords = Array1::from_iter((0..=20).map(|i| 0.05 + 0.15 * i as f64 / 20.0));
        let theta = d1_atom_fitted_turning(&ev, decoder.view(), coords.view())
            .expect("turning must evaluate")
            .expect("a non-degenerate segment must return Some");
        assert!(
            theta < 1e-9,
            "a straight-line image has zero turning (the linear-tail signature); got {theta:.3e}"
        );
    }

    /// #1610/#1026 — a GLOBALLY constant image (only the DC basis row is
    /// nonzero) decodes to a single fixed point: `γ'(t) ≡ 0` at every node, so
    /// the curve is stationary everywhere. A point has no arc to turn through,
    /// so its total turning is exactly `0` — the ultimate linear-tail signature.
    /// This is the path the hybrid-collapse witness relies on (a `d = 1` atom
    /// straightened to its DC component must read `Some(0.0)`, not the historical
    /// `None`, so its slot can collapse to a legitimate linear tail).
    #[test]
    fn constant_image_turning_is_some_zero() {
        let ev = PeriodicHarmonicEvaluator::new(3).expect("3-basis circle");
        let mut decoder = Array2::<f64>::zeros((3, 2));
        // Only the DC (constant) basis row → γ(t) is a fixed point for all t.
        decoder[[0, 0]] = 0.7;
        decoder[[0, 1]] = -0.3;
        let coords = Array1::from_iter((0..=20).map(|i| 0.05 + 0.15 * i as f64 / 20.0));
        let theta = d1_atom_fitted_turning(&ev, decoder.view(), coords.view())
            .expect("turning must evaluate")
            .expect("a globally constant image returns Some(0.0), not None");
        assert_eq!(
            theta, 0.0,
            "a globally constant (all-stationary) image has exactly zero turning"
        );
    }

    /// #1610/#1026 — a curve that is stationary at SOME nodes but moving at
    /// others is a cusp: the unsigned curvature integrand is genuinely
    /// ill-defined at the stationary node, so the function REFUSES (`None`)
    /// rather than under-count the sharp turn. Built as an image `∝ cos 2πt` (a
    /// line through the origin) over a span whose lower endpoint is exactly
    /// `t = 0`, where `γ' ∝ -sin 2πt` vanishes: that endpoint lands on a grid
    /// node and is stationary while every interior node moves — the mixed case
    /// the constant-vs-cusp split must keep conservative.
    #[test]
    fn partial_cusp_turning_is_none() {
        let ev = PeriodicHarmonicEvaluator::new(3).expect("3-basis circle");
        let mut decoder = Array2::<f64>::zeros((3, 2));
        decoder[[2, 0]] = 1.0;
        decoder[[2, 1]] = 2.0;
        // Span [0, 0.2]: the Simpson grid's lower node sits exactly on t = 0,
        // where the speed `‖γ'‖ ∝ |sin 2πt|` is zero (stationary), while every
        // interior node is moving.
        let coords = Array1::from_iter((0..=20).map(|i| 0.2 * i as f64 / 20.0));
        let result = d1_atom_fitted_turning(&ev, decoder.view(), coords.view())
            .expect("turning must evaluate");
        assert!(
            result.is_none(),
            "a curve stationary at some nodes but moving at others (a cusp) must \
             refuse with None; got {result:?}"
        );
    }
}

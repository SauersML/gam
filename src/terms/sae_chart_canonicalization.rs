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
//! `S²` (sphere atoms) is the remaining #1019 stage-2 gap, refused on
//! purpose: by the hairy-ball theorem every smooth tangent vector field on
//! `S²` has zeros, so there is **no global pole-free flow basis** with which
//! to parameterize `Diff(S²)` the way the torus path does. Canonicalizing
//! sphere charts needs a genuinely different representative (harmonic-map /
//! Plateau-type); sphere atoms are left on their fitted charts.

use faer::Side as FaerSide;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::linalg::faer_ndarray::{FaerCholesky, fast_ab, fast_ata, fast_atb};
use crate::terms::sae_manifold::{SaeBasisEvaluator, solve_design_least_squares};

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
/// never walk through a fold.
pub const TORUS_FLOW_DIFFEO_MIN_DET: f64 = 0.1;

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
fn evaluate_flow_defect(
    theta: &[f64],
    row_modes: &[Vec<FlowModeSample>],
    ghat: &[[f64; 3]],
    ghat_norm_sq: f64,
) -> Option<FlowObjectiveState> {
    let n = row_modes.len();
    let mut a_rows = Vec::with_capacity(n);
    let mut cross = 0.0_f64;
    for modes in row_modes {
        let mut a = [1.0_f64, 0.0, 0.0, 1.0];
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
    ghat: &[[f64; 3]],
    ghat_norm_sq: f64,
    q: usize,
    min_det: f64,
    min_det_on_grid: &dyn Fn(&[f64]) -> f64,
) -> Option<FlowMinimization> {
    let n = row_modes.len();
    let mut theta = vec![0.0_f64; q];
    let mut state = evaluate_flow_defect(&theta, row_modes, ghat, ghat_norm_sq)?;
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
                evaluate_flow_defect(&candidate, row_modes, ghat, ghat_norm_sq)
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

    // ── Damped Gauss–Newton on θ (shared exact core; derivation above) ──────
    let Some(minimization) = minimize_isometry_defect_flow(
        &row_modes,
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

/// Scale-invariant isometry defect of a fitted `d = 2` **sphere** atom's
/// `(lat, lon)` chart against the round-sphere reference metric (#1019 stage 2,
/// sphere arm).
///
/// # Why a measurement and not a full flow-pinning (the hairy-ball obstruction)
///
/// The torus path canonicalizes by minimizing the isometry defect over a global
/// truncated Fourier vector-field flow `φ_θ` and committing the minimizer. The
/// sphere has **no global pole-free flow basis** (every smooth tangent field on
/// `S²` has a zero — the hairy-ball theorem), and the `(lat, lon)` chart is
/// itself singular at the poles, so the torus-style flow-pin construction does
/// not transfer. What DOES transfer exactly is the *measurement*: the issue's
/// acceptance criterion is "the planted warped sphere-patch recovers
/// uniform-speed coords (defect within 10% of optimum)", and the defect itself
/// is well-defined at the fitted chart with no flow basis at all. Reporting it
/// turns the honest "left as fitted" sphere arm into a *measurable* one — a
/// fitted chart that is already round-isometric (a true `O(3)` representative)
/// scores `≈ 0`, while a warped chart scores large, so the diagnostic
/// quantifies exactly the chart dishonesty #1019 exists to expose. The pinning
/// (committing a minimizing harmonic-map flow with pole-aware basis) is the
/// remaining seam; this is the certified defect functional it would minimize.
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
    let Some((g_rows, g_bar)) =
        extract_pullback_metric_d2("sphere_chart_isometry_defect", evaluator, decoder, row_coords)?
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
        if !(r11.is_finite() && r11 > 0.0) {
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
            // Zero speed at a node: the curve is momentarily stationary in this
            // chart; the curvature integrand is undefined there. A genuinely
            // collapsed curve has no honest turning.
            return Ok(None);
        }
        // Wedge norm² = ‖γ'‖²‖γ''‖² − ⟨γ',γ''⟩² (Lagrange identity); clamp tiny
        // negative round-off to 0 before the sqrt.
        let wedge_sq = (n1 * n2 - dot * dot).max(0.0);
        integrand[node] = wedge_sq.sqrt() / n1;
        if !integrand[node].is_finite() {
            return Ok(None);
        }
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
}

#[cfg(test)]
mod turning_tests {
    use super::*;
    use crate::terms::sae::basis::PeriodicHarmonicEvaluator;
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
}

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

/// Per-mode sample of the flow basis at one chart point `t`: the scalar field
/// value `f(t)` (the displacement this mode adds to coordinate `component`)
/// and its gradient `∇f(t)` (the mode's contribution to row `component` of
/// the flow Jacobian `Dφ`).
#[derive(Debug, Clone, Copy)]
pub struct TorusFlowModeSample {
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
    pub fn mode_samples(&self, t: [f64; 2]) -> Vec<TorusFlowModeSample> {
        let tau = std::f64::consts::TAU;
        let mut out = Vec::with_capacity(self.dim());
        for component in 0..2 {
            for &(a, b) in &self.freqs {
                let w0 = tau * a as f64 / self.period;
                let w1 = tau * b as f64 / self.period;
                let angle = w0 * t[0] + w1 * t[1];
                let s = angle.sin();
                let c = angle.cos();
                out.push(TorusFlowModeSample {
                    component,
                    value: s,
                    grad: [w0 * c, w1 * c],
                });
                out.push(TorusFlowModeSample {
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
    row_modes: &[Vec<TorusFlowModeSample>],
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

    // ── Fitted pullback metric G_i = J(t_i)ᵀ J(t_i) from the exact jet ──────
    let (row_phi, row_jet) = evaluator.evaluate(row_coords)?;
    if row_phi.ncols() != m || row_jet.dim() != (n, m, 2) {
        return Err(format!(
            "torus_isometry_flow_reparameterization: evaluator returned basis {:?} / jet {:?}; expected width {m}, latent_dim 2",
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
            // Rank-deficient pullback metric: the chart is collapsed along
            // some direction at this row — no isometric representative exists.
            return Ok(None);
        }
        log_det_sum += 0.5 * det.ln();
        g_rows.push([g00, g11, g01]);
    }
    // Geometric-mean metric scale ḡ (scale-invariant normalization).
    let g_bar = (log_det_sum / n as f64).exp();
    if !(g_bar.is_finite() && g_bar > 0.0) {
        return Ok(None);
    }
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

    // ── Flow basis + per-row mode samples (W_{ik} and the displacements) ────
    let basis = TorusFlowBasis::new(period)?;
    let q = basis.dim();
    let mut row_modes: Vec<Vec<TorusFlowModeSample>> = Vec::with_capacity(n);
    for row in 0..n {
        row_modes.push(basis.mode_samples([row_coords[[row, 0]], row_coords[[row, 1]]]));
    }

    // ── Damped Gauss–Newton on θ (derivation in the function docs) ──────────
    let mut theta = vec![0.0_f64; q];
    let Some(mut state) = evaluate_flow_defect(&theta, &row_modes, &ghat, ghat_norm_sq) else {
        return Ok(None);
    };
    let defect_initial = state.defect;
    if !(defect_initial > 0.0) {
        // Already exactly isometric — nothing to canonicalize.
        return Ok(None);
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
            let folded =
                basis.min_jacobian_det_on_grid(&candidate) <= TORUS_FLOW_DIFFEO_MIN_DET;
            let candidate_state = if folded {
                None
            } else {
                evaluate_flow_defect(&candidate, &row_modes, &ghat, ghat_norm_sq)
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
                        rejects = TORUS_FLOW_GN_MAX_REJECTS;
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
        let theta_norm_sq: f64 = theta.iter().map(|v| v * v).sum();
        if step_norm_sq <= 1.0e-24 * (1.0 + theta_norm_sq) {
            break;
        }
    }
    if !any_accepted || !(state.defect < defect_initial) {
        // No strict improvement within the flow family: the fitted chart is
        // already the canonical representative — honest skip.
        return Ok(None);
    }
    let min_flow_jacobian_det = basis.min_jacobian_det_on_grid(&theta);
    if !(min_flow_jacobian_det > TORUS_FLOW_DIFFEO_MIN_DET) {
        // Unreachable through the guarded accept path; refuse defensively
        // rather than ever committing a folded chart.
        return Ok(None);
    }

    // ── Decoder transport on the Nyquist-oversampled audit grid ─────────────
    let axis_nodes =
        TORUS_TRANSPORT_MIN_NODES_PER_AXIS.max(3 * (m as f64).sqrt().ceil() as usize);
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
        defect_final: state.defect,
        profiled_metric_scale: state.scale,
        min_flow_jacobian_det,
        recomposition_residual: recomposition.recomposition_residual,
    }))
}

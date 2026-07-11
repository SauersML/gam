//! Kantorovich-certified encode atlas (issue #1010).
//!
//! Encoding a row `x ∈ ℝᵖ` against a frozen multi-atom dictionary is the joint
//! coordinate problem
//!
//! ```text
//! min_{t_1,…,t_K}  ½‖x − Σ_k z_k B_kᵀΦ_k(t_k)‖² + Σ_k prior_k(t_k).
//! ```
//!
//! [`joint_encode_refine_row`] solves that objective with the shared residual.
//! The per-atom atlas below is an initializer and a standalone single-atom
//! projection facility; composing its independent optima is not a multi-atom
//! encode. With the amplitude `z_k` and decoder block `B_k` held fixed, Newton on
//! a single-atom field `F(t) = ∇f_k(t)` converges from a start `t₀` into the unique
//! root in a certified ball whenever the **Newton–Kantorovich** quantity
//!
//! ```text
//! h = β · η · L ≤ ½,    β = ‖F'(t₀)⁻¹‖,   η = ‖F'(t₀)⁻¹ F(t₀)‖,
//! ```
//!
//! where `L` is a Lipschitz constant of `F'` (the Hessian of `f_k`) on a region
//! containing the Newton iterates. `h` is CHECKABLE per row in `O(q³)`
//! (`q = latent_dim`, tiny), so each fast-path encode carries its own
//! exactness certificate.
//!
//! ## The closed-form Hessian-Lipschitz constant `L`
//!
//! Write `m(t) = z·BᵀΦ(t) ∈ ℝᵖ` (the reconstruction) and `r(t) = m(t) − x`.
//! Then `f = ½‖r‖² + prior` and, differentiating three times,
//!
//! ```text
//! ∇³f = 3·sym(J_mᵀ : ∇²m) + ⟨r, ∇³m⟩ + ∇³prior,
//! ```
//!
//! so an operator-norm bound on the chart is
//!
//! ```text
//! L ≤ 3·‖J_m‖·‖∇²m‖ + ‖r‖·‖∇³m‖ + L_prior,
//! ```
//!
//! with `‖∂^g m‖ ≤ |z|·(Σ_m ‖B_{m,:}‖)·B_g`, where `B_g = sup_chart max_m
//! ‖∂^g Φ_m‖` is the per-column jet sup of the basis family — closed form per
//! family ([`BasisHessianLipschitz`]). `‖r‖` is bounded by `‖x‖ +
//! |z|·(Σ_m‖B_{m,:}‖)·B_0`. The ARD/von-Mises prior `L_prior` is a closed-form
//! constant from the prior strength. Every bound is conservative (an
//! over-estimate of `L` only SHRINKS the certified radius — it can never
//! certify a row that does not converge).
//!
//! ## Pipeline
//!
//! 1. **Offline, per atom** ([`EncodeAtlas::build`]): chart centers `t_c` on the
//!    atom's coordinate grid (the SHAPE_BAND grid idiom), each with a certified
//!    Newton radius `R_c` solved from the Kantorovich inequality at the
//!    worst-case in-chart start.
//! 2. **Online, per row** ([`EncodeAtlas::certified_encode_row`]): route to the
//!    nearest chart, start from its distilled IFT predictor, take one or two
//!    Newton steps, then the `h ≤ ½` check AT the start point is the per-row
//!    certificate.
//! 3. **Uncertified tail**: rows whose start fails `h ≤ ½` are FLAGGED (counted
//!    in [`EncodeResult::encode_uncertified_count`]) and must be routed by the
//!    caller to the existing exact multi-start solve. No approximation enters
//!    silently.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use opt::constants::{ARMIJO_C1, BACKTRACK_CONTRACTION};
use opt::{AcceptedStep, BacktrackConfig, backtracking_line_search};

use crate::candidate_index::{AtomFrameSketch, SaeCandidateIndex, auto_candidate_budget};
use crate::manifold::{
    AffineCoordinateEvaluator, CylinderHarmonicEvaluator, DuchonCoordinateEvaluator,
    EuclideanPatchEvaluator, PeriodicHarmonicEvaluator, SaeBasisEvaluator, SaeManifoldAtom,
    SphereChartEvaluator, TorusHarmonicEvaluator,
};
use gam_linalg::faer_ndarray::FaerEigh;

use faer::Side;

/// The Kantorovich convergence threshold `h ≤ ½`. Below this the Newton
/// iteration is guaranteed to converge quadratically into the unique root in
/// the certified ball; at or above it the start is uncertified.
pub const KANTOROVICH_THRESHOLD: f64 = 0.5;

/// Row count at or above which the corpus-rate certified-encode batch
/// (`certified_encode_batch` / `certified_encode_with_index`) fans its
/// per-row encodes out over rayon. Below this the per-row Newton + chart
/// routing is cheap enough that the fan-out overhead does not pay; matched to
/// the same order as the arrow-Schur `SCHUR_MATVEC_PARALLEL_ROW_MIN` gate so
/// short batches inside an outer atom-level fan-out stay sequential.
pub(crate) const ENCODE_BATCH_PARALLEL_ROW_MIN: usize = 256;

/// Minimum frame alignment `‖Uₖᵀd‖/‖d‖ ∈ [0,1]` the routed atom must have for an
/// index-routed encode to be attempted at all — a FIT-QUALITY floor, NOT a
/// routing-correctness gate (#1026, corrected by the #1777 exact-routing path).
///
/// Routing itself is now EXACT: the index-routed encode picks the atom via
/// [`SaeCandidateIndex::route_exact`], which returns the GLOBAL argmax of the
/// routing score (the universal-bound LSH fast path, else a full-scan fallback) —
/// so there is no "missed-better-ungathered-atom" hole left for this constant to
/// patch. What remains is a different, honest question: even the globally-best
/// atom may align only weakly with a row (no atom in the dictionary fits it). A
/// finite alignment below this floor means the best available atom is a poor fit,
/// so the row is flagged and routed to the exact multi-start fallback rather than
/// encoded against an atom it barely belongs to. (Previously this same comparison
/// double-served as a recall proxy; that role is gone — `route_exact` guarantees
/// recall — leaving only the fit-quality role described here.)
pub(crate) const CANDIDATE_ROUTING_MIN_ALIGNMENT: f64 = 0.5;

/// Number of nearest charts the CERTIFIED encode refines in before returning the
/// lowest-reconstruction-error certified result. A single nearest chart is not
/// globally sound where the decoded manifold folds near itself (both competing
/// basins' charts reconstruct near the fold, so both rank among the nearest by
/// ambient distance); refining the top few captures the global basin. For a
/// unimodal atom all candidates converge to the same root, so K>1 is a no-op.
pub(crate) const CERTIFIED_ROUTING_TOPK: usize = 4;

/// Newton refinement convergence floor. Once a refinement step's length `‖δ‖`
/// falls below this (relative to the coordinate scale `1 + ‖t‖`), the iterate has
/// reached the certified root to f64 resolution: applying the step cannot move `t`
/// meaningfully, and the remaining fixed-budget steps only re-accumulate round-off.
/// Stopping there is STRICTLY more accurate than draining a fixed step budget on a
/// well-conditioned quadratic Newton tail, and it removes that tail's per-step
/// `evaluate` + `second_jet` cost (the dominant per-row encode work). The batched
/// and per-row encodes share this rule, so they stay bit-identical.
pub(crate) const NEWTON_REFINE_CONVERGED_EPS: f64 = 1.0e-12;

/// Global-minimum short-circuit floor for top-K certified routing. The
/// reconstruction error `‖x − z·m(t)‖` is bounded below by 0, so a certified
/// candidate whose residual already sits at the ambient noise floor
/// (`≤ this · (1 + ‖x‖)`) is provably the global optimum over the charts — no
/// competing chart can reach a strictly lower residual. The remaining candidates'
/// refinement is then skipped. Conservative (a genuine second basin of the same
/// target reconstructs the SAME point, so returning the first is a valid encode).
pub(crate) const CERTIFIED_GLOBAL_MIN_RECON_FLOOR: f64 = 1.0e-11;

/// A chart region on an atom's latent coordinate: a center `t_c` plus a
/// certified in-chart radius. Over the ball `‖t − t_c‖ ≤ radius` the jet sup
/// bounds returned by [`BasisHessianLipschitz`] hold, so the Kantorovich
/// constant `L` computed from them is valid for any start in the ball.
///
/// For radial (Duchon) families the chart also carries the minimum kernel-center
/// distance `exclusion_r_min` (a lower bound on `‖t − c_k‖` over the chart) that
/// bounds the otherwise-singular `1/r` radial tails (issue #1010).
#[derive(Debug, Clone)]
pub struct ChartRegion {
    /// Chart center coordinate `t_c` (length = latent_dim).
    pub center: Array1<f64>,
    /// In-chart radius in the coordinate metric.
    pub radius: f64,
    /// For radial (Duchon) families: a lower bound on `‖t − c_k‖` over the
    /// chart, across every kernel center `c_k`. `None` for non-radial families.
    pub exclusion_r_min: Option<f64>,
    /// For radial (Duchon) families: an upper bound on `‖t − c_k‖` over the
    /// chart, across every kernel center `c_k`. `None` for non-radial families.
    pub radial_r_max: Option<f64>,
}

impl ChartRegion {
    pub fn new(center: Array1<f64>, radius: f64) -> Self {
        Self {
            center,
            radius,
            exclusion_r_min: None,
            radial_r_max: None,
        }
    }

    pub fn with_radial_bounds(mut self, r_min: f64, r_max: f64) -> Self {
        self.exclusion_r_min = Some(r_min);
        self.radial_r_max = Some(r_max);
        self
    }

    /// A jet-sup certificate is only meaningful over a genuine region. Even
    /// families whose bounds are manifold-global constants (the sup over any
    /// chart equals the global sup) must refuse a malformed chart rather than
    /// certify garbage geometry.
    pub(crate) fn assert_valid(&self) {
        assert!(
            self.radius.is_finite()
                && self.radius >= 0.0
                && self.center.iter().all(|c| c.is_finite()),
            "ChartRegion must have a finite center and a finite non-negative radius"
        );
    }
}

/// Per-column sup-norm bounds on the first three coordinate jets of a basis
/// family `Φ(t)`, valid over a stated [`ChartRegion`] (issue #1010). These are
/// the analytic ingredients of the Hessian-Lipschitz constant `L` — see the
/// module docs for the assembly. `value_sup` bounds `max_m |Φ_m|`,
/// `jacobian_sup`/`hessian_sup`/`third_sup` bound `max_m ‖∂^g Φ_m‖`.
pub trait BasisHessianLipschitz {
    fn value_sup(&self, chart: &ChartRegion) -> f64;
    fn jacobian_sup(&self, chart: &ChartRegion) -> f64;
    fn hessian_sup(&self, chart: &ChartRegion) -> f64;
    fn third_sup(&self, chart: &ChartRegion) -> f64;
}

/// Sup over the circle of the `g`-th derivative of any single harmonic column
/// of a `num_basis`-wide Fourier basis `[1, sin(2π h t), cos(2π h t), …]`:
/// `(2π·H)^g` for the top harmonic `H = (num_basis − 1)/2`. The constant column
/// contributes `0` for `g ≥ 1`, so the top harmonic dominates; the bound is
/// global (the trig magnitudes are `≤ 1` everywhere, independent of the chart).
pub(crate) fn harmonic_jet_sup(num_basis: usize, order: u32) -> f64 {
    let top_harmonic = num_basis.saturating_sub(1) / 2;
    let omega = std::f64::consts::TAU * top_harmonic as f64;
    omega.powi(order as i32)
}

impl BasisHessianLipschitz for PeriodicHarmonicEvaluator {
    fn value_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        1.0
    }
    fn jacobian_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        harmonic_jet_sup(self.num_basis, 1)
    }
    fn hessian_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        harmonic_jet_sup(self.num_basis, 2)
    }
    fn third_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        harmonic_jet_sup(self.num_basis, 3)
    }
}

impl BasisHessianLipschitz for TorusHarmonicEvaluator {
    /// Tensor product of per-axis circle harmonics. A torus basis column is a
    /// product of single-axis harmonics, each bounded as in the circle case.
    /// The `g`-th coordinate jet routes `g` derivative operators across the
    /// `latent_dim` factors (Leibniz); each routing contributes a product of
    /// per-axis derivative magnitudes. A per-column sup is therefore bounded by
    /// the top single-axis frequency to the `g`-th power times the number of
    /// such routings (`latent_dim^g`, the count of operator-to-axis maps).
    fn value_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        1.0
    }
    fn jacobian_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        torus_jet_sup(self.num_harmonics, self.latent_dim, 1)
    }
    fn hessian_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        torus_jet_sup(self.num_harmonics, self.latent_dim, 2)
    }
    fn third_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        torus_jet_sup(self.num_harmonics, self.latent_dim, 3)
    }
}

/// Per-column `g`-th jet sup for the torus harmonic basis: `(2π·H)^g ·
/// latent_dim^g`, where `H = num_harmonics` is the top per-axis frequency and
/// `latent_dim^g` over-counts the Leibniz routings of `g` operators across the
/// product factors (a conservative bound — each routing's per-axis magnitude is
/// `≤ (2π H)^{#ops on that axis}`, and the products telescope to `(2π H)^g`).
pub(crate) fn torus_jet_sup(num_harmonics: usize, latent_dim: usize, order: u32) -> f64 {
    let omega = std::f64::consts::TAU * num_harmonics as f64;
    omega.powi(order as i32) * (latent_dim as f64).powi(order as i32)
}

impl BasisHessianLipschitz for SphereChartEvaluator {
    /// The 7-column lat/lon chart `[1, x, y, z, xy, yz, xz]` with
    /// `x = cos(lat)cos(lon)`, `y = cos(lat)sin(lon)`, `z = sin(lat)`. Each of
    /// `x, y, z` is a product of two unit-frequency trig factors, so its `g`-th
    /// coordinate jet is a sum of `2^g` products of `{sin,cos}` (each `≤ 1`):
    /// magnitude `≤ 2^g` for `g ≥ 1`, `≤ 1` for `g = 0`. The bilinear columns
    /// `xy, yz, xz` are products of two such coordinates; by Leibniz over the
    /// product, their `g`-th jet is bounded by `Σ_{i=0}^{g} C(g,i)·(2^i)·(2^{g−i})
    /// = (2+2)^g = 4^g` (using `‖∂^i u‖ ≤ 2^i`, `|u| ≤ 1`). The bilinear columns
    /// dominate, so the per-column sup is `4^g` (`g ≥ 1`). Bounds are global
    /// constants — the chart box `lat ∈ [-π/2, π/2]` does not enlarge them.
    fn value_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        1.0
    }
    fn jacobian_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        4.0
    }
    fn hessian_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        16.0
    }
    fn third_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        64.0
    }
}

impl BasisHessianLipschitz for AffineCoordinateEvaluator {
    /// The affine basis `[1, t₁, …, t_d]` is degree ≤ 1: its first jet has unit
    /// columns, and all second and third jets vanish. The value sup is
    /// `max(1, ‖t‖)` over the chart, bounded by `1 + ‖t_c‖ + radius`.
    fn value_sup(&self, chart: &ChartRegion) -> f64 {
        let center_norm = chart.center.dot(&chart.center).sqrt();
        1.0 + center_norm + chart.radius
    }
    fn jacobian_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        1.0
    }
    fn hessian_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        0.0
    }
    fn third_sup(&self, chart: &ChartRegion) -> f64 {
        chart.assert_valid();
        0.0
    }
}

impl BasisHessianLipschitz for EuclideanPatchEvaluator {
    /// Monomials of total degree ≤ `max_degree` in `t ∈ ℝ^d`. Over the ball of
    /// radius `R` about `t_c`, each coordinate is bounded by `ρ = ‖t_c‖∞ + R`.
    /// A monomial `t^α` with `|α| = q` has `g`-th partials bounded (crudely) by
    /// the descending-factorial coefficient `q·(q−1)···(q−g+1) ≤ q^g` times
    /// `ρ^{max(q−g,0)}`, and there are at most `d^g` partial routings, so the
    /// per-column `g`-th jet sup is `≤ d^g · D^g · ρ^{max(D−g,0)}` with
    /// `D = max_degree`. Conservative; D is small for patch evaluators.
    fn value_sup(&self, chart: &ChartRegion) -> f64 {
        let rho = patch_rho(chart);
        let d = self.max_degree as i32;
        rho.powi(d).max(1.0)
    }
    fn jacobian_sup(&self, chart: &ChartRegion) -> f64 {
        patch_jet_sup(self.latent_dim, self.max_degree, chart, 1)
    }
    fn hessian_sup(&self, chart: &ChartRegion) -> f64 {
        patch_jet_sup(self.latent_dim, self.max_degree, chart, 2)
    }
    fn third_sup(&self, chart: &ChartRegion) -> f64 {
        patch_jet_sup(self.latent_dim, self.max_degree, chart, 3)
    }
}

impl BasisHessianLipschitz for CylinderHarmonicEvaluator {
    /// Cylinder `S¹ × ℝ` product basis `Φ_{c,l} = c(t₀)·l(t₁)`, the circle
    /// (periodic harmonic) factor on axis 0 crossed with the monomial line
    /// factor on axis 1. Because the two factors depend on disjoint coordinates,
    /// the order-`g` coordinate jet in any cell is exactly
    /// `c^{(k₀)}(t₀)·l^{(k₁)}(t₁)` with `k₀ + k₁ = g`, so the per-column sup is
    /// the max over the split `k₀ + k₁ = g` of the product of the two per-axis
    /// per-order sups: the circle factor contributes `1` at order 0 and
    /// `(2π·H)^{k₀}` at order `k₀ ≥ 1` (trig magnitudes `≤ 1`); the line factor
    /// contributes the monomial-patch sup `D^{k₁}·ρ^{max(D−k₁,0)}` (`D = line
    /// degree`, `ρ = ‖t_c‖∞ + radius`). Bounds are global in the periodic axis
    /// and chart-local in the line axis.
    fn value_sup(&self, chart: &ChartRegion) -> f64 {
        cylinder_jet_sup(self.circle_harmonics, self.line_degree, chart, 0)
    }
    fn jacobian_sup(&self, chart: &ChartRegion) -> f64 {
        cylinder_jet_sup(self.circle_harmonics, self.line_degree, chart, 1)
    }
    fn hessian_sup(&self, chart: &ChartRegion) -> f64 {
        cylinder_jet_sup(self.circle_harmonics, self.line_degree, chart, 2)
    }
    fn third_sup(&self, chart: &ChartRegion) -> f64 {
        cylinder_jet_sup(self.circle_harmonics, self.line_degree, chart, 3)
    }
}

/// Per-column order-`g` jet sup of the cylinder product basis: the max over
/// `k₀ + k₁ = g` of `circle_axis_sup(k₀) · line_axis_sup(k₁)`, where the circle
/// axis sup is `(2π·H)^{k₀}` (`1` at `k₀ = 0`) and the line axis sup is the
/// monomial-patch bound `D^{k₁}·ρ^{max(D−k₁,0)}` (`1` at `k₁ = 0`). See the
/// [`CylinderHarmonicEvaluator`] doc comment for the derivation.
pub(crate) fn cylinder_jet_sup(
    circle_harmonics: usize,
    line_degree: usize,
    chart: &ChartRegion,
    order: u32,
) -> f64 {
    let omega = std::f64::consts::TAU * circle_harmonics as f64;
    let big_d = line_degree as f64;
    let rho = patch_rho(chart);
    let mut best = 0.0_f64;
    for k0 in 0..=order {
        let k1 = order - k0;
        let circle = if k0 == 0 { 1.0 } else { omega.powi(k0 as i32) };
        let line = if k1 == 0 {
            rho.powi(line_degree as i32).max(1.0)
        } else {
            let residual = line_degree.saturating_sub(k1 as usize) as i32;
            // `.max(1.0)` as in `patch_jet_sup`: for ρ < 1 a lower-degree line
            // monomial dominates the k1-th derivative, so the bare `ρ^residual`
            // underestimates the line-factor sup. The value case (k1==0) already
            // clamps; this completes it for the derivative orders.
            big_d.powi(k1 as i32) * rho.powi(residual).max(1.0)
        };
        best = best.max(circle * line);
    }
    best
}

/// Sup-norm radius `ρ = ‖t_c‖∞ + radius` of the chart (the coordinate magnitude
/// bound used by the monomial-patch jet bounds).
pub(crate) fn patch_rho(chart: &ChartRegion) -> f64 {
    let center_inf = chart
        .center
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    center_inf + chart.radius
}

/// Per-column `g`-th jet sup for a monomial patch of max degree `D` in `d`
/// coordinates over the chart: `d^g · D^g · ρ^{max(D−g,0)}` (see the
/// [`EuclideanPatchEvaluator`] doc comment for the derivation).
pub(crate) fn patch_jet_sup(
    latent_dim: usize,
    max_degree: usize,
    chart: &ChartRegion,
    order: u32,
) -> f64 {
    let d = latent_dim as f64;
    let big_d = max_degree as f64;
    let rho = patch_rho(chart);
    let residual_degree = max_degree.saturating_sub(order as usize) as i32;
    // `.max(1.0)`: for ρ < 1 (small charts near the origin) the g-th jet sup is NOT
    // dominated by the max-degree monomial `t^D` (whose g-th derivative ~ ρ^{D-g}
    // shrinks with ρ) but by a LOWER-degree monomial whose g-th derivative is a
    // larger constant — e.g. {1,t,t²}'s jacobian sup is the linear term's constant
    // `1`, which exceeds `2ρ` when ρ < ½. Without the clamp the bound underestimates
    // the true sup (numerically: D=3, ρ=0.1, g=1 → formula 0.03 vs true 1.0), which
    // would make the certificate's Lipschitz `L` too small → a FALSE certificate.
    // `D^g · max(ρ^{D-g}, 1)` upper-bounds `max_{q∈[g,D]} (q!/(q-g)!)·ρ^{q-g}` for
    // all ρ (the `q=g` term gives `g! ≤ D^g`, the `q=D` term gives `≤ D^g·ρ^{D-g}`).
    d.powi(order as i32) * big_d.powi(order as i32) * rho.powi(residual_degree).max(1.0)
}

impl BasisHessianLipschitz for DuchonCoordinateEvaluator {
    /// Radial-kernel basis `Φ_m(t) = φ(r_m)`, `r_m = ‖t − c_m‖`, plus a
    /// polynomial nullspace block. For the cubic Duchon kernel `φ(r) = r³` the
    /// radial derivatives are `φ' = 3r²`, `φ'' = 6r`, `φ''' = 6`. The chain rule
    /// to coordinate jets introduces `1/r` factors through the unit radial
    /// direction `u = (t − c)/r` and the projector `(I − uuᵀ)/r`, so over a
    /// chart the jets are bounded by combining the radial-derivative magnitudes
    /// at the worst-case radius with the inverse-radius tail at the chart's
    /// EXCLUSION radius `r_min` (the closest a chart point gets to any center):
    ///
    /// ```text
    /// ‖∇φ‖    ≤ |φ'|                              ≤ 3 r_max²
    /// ‖∇²φ‖   ≤ |φ''| + |φ'|/r                    ≤ 6 r_max + 3 r_max²/r_min
    /// ‖∇³φ‖   ≤ |φ'''| + 3|φ''|/r + 3|φ'|/r²      ≤ 6 + 18 r_max/r_min + 9 r_max²/r_min²
    /// ```
    ///
    /// (the `1/r`, `1/r²` tails are bounded by `1/r_min`, `1/r_min²`). The
    /// polynomial nullspace block is degree ≤ `order`; its jets are bounded like
    /// the monomial patch with `D = order`. The per-column sup is the max of the
    /// kernel and polynomial bounds. The `r³` kernel is itself `C²` (no
    /// singularity) so these tails are conservative but finite for any
    /// `r_min > 0`; the atlas refines charts to keep `r_min` bounded away from 0.
    fn value_sup(&self, chart: &ChartRegion) -> f64 {
        let r_max = chart.radial_r_max.unwrap_or(chart.radius);
        let poly = duchon_poly_jet_sup(self.centers.ncols(), self.order_degree(), chart, 0);
        (r_max.powi(3)).max(poly)
    }
    fn jacobian_sup(&self, chart: &ChartRegion) -> f64 {
        let r_max = chart.radial_r_max.unwrap_or(chart.radius);
        let kernel = 3.0 * r_max * r_max;
        let poly = duchon_poly_jet_sup(self.centers.ncols(), self.order_degree(), chart, 1);
        kernel.max(poly)
    }
    fn hessian_sup(&self, chart: &ChartRegion) -> f64 {
        let r_max = chart.radial_r_max.unwrap_or(chart.radius);
        let r_min = chart
            .exclusion_r_min
            .unwrap_or(chart.radius)
            .max(f64::MIN_POSITIVE);
        let kernel = 6.0 * r_max + 3.0 * r_max * r_max / r_min;
        let poly = duchon_poly_jet_sup(self.centers.ncols(), self.order_degree(), chart, 2);
        kernel.max(poly)
    }
    fn third_sup(&self, chart: &ChartRegion) -> f64 {
        let r_max = chart.radial_r_max.unwrap_or(chart.radius);
        let r_min = chart
            .exclusion_r_min
            .unwrap_or(chart.radius)
            .max(f64::MIN_POSITIVE);
        let kernel = 6.0 + 18.0 * r_max / r_min + 9.0 * r_max * r_max / (r_min * r_min);
        let poly = duchon_poly_jet_sup(self.centers.ncols(), self.order_degree(), chart, 3);
        kernel.max(poly)
    }
}

/// Polynomial-block degree of a Duchon nullspace order, used to bound the
/// nullspace columns like a monomial patch.
trait DuchonOrderDegree {
    fn order_degree(&self) -> usize;
}

impl DuchonOrderDegree for DuchonCoordinateEvaluator {
    fn order_degree(&self) -> usize {
        match self.order {
            gam_terms::basis::DuchonNullspaceOrder::Zero => 0,
            gam_terms::basis::DuchonNullspaceOrder::Linear => 1,
            gam_terms::basis::DuchonNullspaceOrder::Degree(d) => d,
        }
    }
}

/// Per-column `g`-th jet sup of the Duchon polynomial nullspace block, treated
/// as a monomial patch of degree `order_degree`.
pub(crate) fn duchon_poly_jet_sup(
    latent_dim: usize,
    order_degree: usize,
    chart: &ChartRegion,
    order: u32,
) -> f64 {
    if order_degree == 0 {
        return if order == 0 { 1.0 } else { 0.0 };
    }
    patch_jet_sup(latent_dim, order_degree, chart, order)
}

/// Decoder magnitude `Σ_m ‖B_{m,:}‖₂` of an atom's frozen decoder block: the
/// factor that converts a per-column `Φ`-jet sup `B_g` into a reconstruction
/// jet sup `‖∂^g m‖ ≤ |z|·decoder_row_norm_sum·B_g`.
pub(crate) fn decoder_row_norm_sum(decoder: ArrayView2<'_, f64>) -> f64 {
    let mut acc = 0.0;
    for row in decoder.rows() {
        acc += row.dot(&row).sqrt();
    }
    acc
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ReconstructionJetSups {
    pub(crate) value: f64,
    pub(crate) jacobian: f64,
    pub(crate) hessian: f64,
    pub(crate) third: f64,
}

pub(crate) fn pair_trig_decoder_sup(
    sin_row: ArrayView1<'_, f64>,
    cos_row: ArrayView1<'_, f64>,
) -> f64 {
    let aa = sin_row.dot(&sin_row);
    let bb = cos_row.dot(&cos_row);
    let ab = sin_row.dot(&cos_row);
    let trace = aa + bb;
    let disc = ((aa - bb) * (aa - bb) + 4.0 * ab * ab).sqrt();
    (0.5 * (trace + disc)).sqrt()
}

/// Per-harmonic reconstruction jet sups of a periodic atom. `decoder` MUST be
/// the FULL-width decoder on the standard `[1, sin 2πt, cos 2πt, …]` inner
/// basis: the row pairing below identifies rows `(2h−1, 2h)` as the harmonic-`h`
/// `(sin, cos)` pair and prices them at `ω = 2πh`. A #1117 rank-reduced decoder
/// `B̃ = Qᵀ B` has NO such row meaning (each reduced row mixes all harmonics),
/// so callers re-expand through `Q` first — see [`reconstruction_jet_sups`].
pub(crate) fn periodic_reconstruction_jet_sups(
    decoder: ArrayView2<'_, f64>,
) -> ReconstructionJetSups {
    let mut value = 0.0;
    let mut jacobian = 0.0;
    let mut hessian = 0.0;
    let mut third = 0.0;
    if decoder.nrows() > 0 {
        value += decoder.row(0).dot(&decoder.row(0)).sqrt();
    }
    let harmonics = decoder.nrows().saturating_sub(1) / 2;
    for h in 1..=harmonics {
        let sin_idx = 2 * h - 1;
        let cos_idx = 2 * h;
        let amp = pair_trig_decoder_sup(decoder.row(sin_idx), decoder.row(cos_idx));
        let omega = std::f64::consts::TAU * h as f64;
        value += amp;
        jacobian += omega * amp;
        hessian += omega.powi(2) * amp;
        third += omega.powi(3) * amp;
    }
    for row in (1 + 2 * harmonics)..decoder.nrows() {
        let amp = decoder.row(row).dot(&decoder.row(row)).sqrt();
        value += amp;
        let omega = std::f64::consts::TAU * harmonics.max(1) as f64;
        jacobian += omega * amp;
        hessian += omega.powi(2) * amp;
        third += omega.powi(3) * amp;
    }
    ReconstructionJetSups {
        value,
        jacobian,
        hessian,
        third,
    }
}

pub(crate) fn reconstruction_jet_sups(
    atom: &SaeManifoldAtom,
    sups: JetSups,
) -> ReconstructionJetSups {
    // `sups` bounds the FULL-width family (see `family_jet_sups`), so the
    // decoder it pairs with must be the full-width pre-image `B = Q B̃` when the
    // atom was #1117 rank-reduced. The reconstruction is identical through that
    // frame (`Φ̃ B̃ = Φ (Q B̃)`), so the bound is exact-in-structure; pairing the
    // full sups with the reduced `B̃` instead would price periodic rows at the
    // wrong harmonic and mismatch the family the sups were taken over.
    let full_decoder = atom
        .reduced_column_map
        .is_some()
        .then(|| atom.full_width_decoder());
    let decoder = full_decoder
        .as_ref()
        .map_or_else(|| atom.decoder_coefficients.view(), |b| b.view());
    if matches!(atom.basis_kind, crate::manifold::SaeAtomBasisKind::Periodic) {
        periodic_reconstruction_jet_sups(decoder)
    } else {
        let decoder_norm_sum = decoder_row_norm_sum(decoder);
        ReconstructionJetSups {
            value: decoder_norm_sum * sups.value,
            jacobian: decoder_norm_sum * sups.jacobian,
            hessian: decoder_norm_sum * sups.hessian,
            third: decoder_norm_sum * sups.third,
        }
    }
}

/// The Hessian-Lipschitz constant `L` of the per-row encode objective `f_k` on
/// a chart, assembled in closed form from the basis jet sups and the decoder /
/// amplitude / target magnitudes. See the module docs for the derivation:
///
/// ```text
/// L ≤ 3·‖J_m‖·‖∇²m‖ + ‖r‖·‖∇³m‖ + L_prior,
/// ‖∂^g m‖ ≤ |z|·S_B·B_g,   S_B = Σ_m ‖B_{m,:}‖,
/// ‖r‖ ≤ ‖x‖ + |z|·S_B·B_0,
/// ```
///
/// `prior_lipschitz` is the caller-supplied closed-form `L_prior` of the
/// ARD/von-Mises coordinate prior (`0.0` if no prior is active on the encode).
pub(crate) fn hessian_lipschitz_constant(
    recon_sups: ReconstructionJetSups,
    amplitude: f64,
    target_norm: f64,
    prior_lipschitz: f64,
) -> f64 {
    let z = amplitude.abs();
    let m_jac = z * recon_sups.jacobian;
    let m_hess = z * recon_sups.hessian;
    let m_third = z * recon_sups.third;
    let recon_value = z * recon_sups.value;
    let r_norm = target_norm + recon_value;
    3.0 * m_jac * m_hess + r_norm * m_third + prior_lipschitz
}

/// One offline-certified chart: a center, its Kantorovich constants, and the
/// certified Newton-convergence radius `R_c` solved from `h = β·η·L ≤ ½` at the
/// worst-case in-chart start.
#[derive(Debug, Clone)]
pub struct CertifiedChart {
    pub region: ChartRegion,
    /// Closed-form Hessian-Lipschitz constant `L` over the chart.
    pub lipschitz: f64,
    /// `β = ‖F'(t_c)⁻¹‖` at the chart center (worst-case in-chart start uses
    /// the center's curvature; the radius is solved so the certificate holds for
    /// any start in the ball).
    pub beta_center: f64,
    /// Certified Newton radius: starts within `radius` of `t_c` satisfy `h ≤ ½`.
    pub certified_radius: f64,
    /// Distilled amortized-encoder Jacobian for this chart (#1026 ladder item 3).
    ///
    /// The exact encode map `x ↦ t` solves `F(t; x) = J_m(t)ᵀ(m(t) − x) = 0`. By
    /// the implicit function theorem its derivative at the converged root is
    /// `dt/dx = −(∂_t F)⁻¹ (∂_x F) = H⁻¹ J_m` (since `∂_x F = −J_m`), so the
    /// first-order Taylor expansion of the encode map about this chart's center
    /// `t_c` is the closed-form AFFINE predictor
    ///
    /// ```text
    /// t(x) ≈ t_c + (1/z) · A₁ · (x − z · m₁(t_c)),   A₁ = (J₁ᵀJ₁ + ridge·I)⁻¹ J₁,
    /// ```
    ///
    /// with `J₁ = Bᵀ J_Φ(t_c)` and `m₁(t_c) = BᵀΦ(t_c)` the AMPLITUDE-1
    /// reconstruction jets (the amplitude `z` factors out analytically, so the
    /// stored Jacobian is amplitude-free). This is the DISTILLED amortized
    /// encoder of the #1026 thread: the per-row Hessian factorization + Newton
    /// iteration is moved OFFLINE into this `d × p` matrix, leaving a single
    /// `O(d·p)` mat-vec online — no per-row eigendecomposition, no second-jet
    /// evaluation. The Kantorovich certificate is still evaluated AT the
    /// predicted start, so the amortized prediction is trusted iff `h ≤ ½` and an
    /// uncertified row still routes to the exact multi-start solve (the encoder
    /// approximates inference, the certificate keeps it honest — the thread's
    /// "encoder + certificate-gated exact fallback" deployment). `None` when the
    /// center's Gauss–Newton block is singular (no certifiable amortization).
    pub amortized_jacobian: Option<Array2<f64>>,
    /// Amplitude-1 chart-center reconstruction `m₁(t_c) = BᵀΦ(t_c)` (length `p`),
    /// the anchor the amortized predictor expands the encode map around.
    pub recon_center: Array1<f64>,
    /// Precomputed affine-predictor CONSTANT term `base = t_c − A₁·m₁(t_c)` (length
    /// `d`), so the online amortized encode of a row `x` at amplitude `z` is the
    /// single mat-vec `t̂ = base + (1/z)·A₁·x` with NO per-row `A₁·m₁` recompute.
    /// Hoisting this atom-static term offline is what lets the massive-K index-routed
    /// fast paths run a single allocation-free pass over rows (rather than a per-atom
    /// GEMM sub-batch that degenerates to one row per group when `K ≫ N`). `None`
    /// exactly when `amortized_jacobian` is `None` (singular Gauss–Newton block).
    pub amortized_base: Option<Array1<f64>>,
}

/// The per-atom encode atlas: a set of certified charts covering the atom's
/// coordinate domain, plus the decoder/amplitude scaling needed to recompute a
/// per-row certificate online.
#[derive(Debug, Clone)]
pub struct AtomEncodeAtlas {
    pub atom_index: usize,
    pub latent_dim: usize,
    pub decoder_norm_sum: f64,
    pub charts: Vec<CertifiedChart>,
}

/// Result of a certified encode over a batch of rows, carrying the honesty
/// flag: how many rows could NOT be certified and were flagged for the exact
/// multi-start fallback (issue #1010 — no approximation enters silently).
#[derive(Debug, Clone)]
pub struct EncodeResult {
    /// Per-row encoded latent coordinates (`n_rows × latent_dim`).
    pub coords: Array2<f64>,
    /// Per-row certificate: `true` ⇒ the row's start satisfied `h ≤ ½` and the
    /// 1–2 Newton steps are exact-into-the-certified-ball; `false` ⇒ flagged.
    pub certified: Vec<bool>,
    /// Count of rows that could not be certified. These ride the payload so the
    /// caller routes them to the exact multi-start encode — honesty, never
    /// silent. Equals `certified.iter().filter(|c| !**c).count()`.
    pub encode_uncertified_count: usize,
}

/// Result of solving the frozen dictionary's joint coordinate objective over a
/// batch. `converged[row]` is a numerical first-order stationarity verdict for
/// the shared-residual objective, not a Newton--Kantorovich certificate.
#[derive(Debug, Clone)]
pub struct JointEncodeResult {
    /// Per-atom coordinate blocks, each shaped `n_rows × latent_dim_k`.
    pub coords: Vec<Array2<f64>>,
    /// Joint row solve reached the first-order tolerance.
    pub converged: Vec<bool>,
    /// Exact tally of `false` entries in `converged`.
    pub unconverged_count: usize,
}

impl JointEncodeResult {
    pub(crate) fn new(coords: Vec<Array2<f64>>, converged: Vec<bool>) -> Self {
        let unconverged_count = converged.iter().filter(|ok| !**ok).count();
        Self {
            coords,
            converged,
            unconverged_count,
        }
    }
}

impl EncodeResult {
    pub(crate) fn from_rows(coords: Array2<f64>, certified: Vec<bool>) -> Self {
        let encode_uncertified_count = certified.iter().filter(|c| !**c).count();
        Self {
            coords,
            certified,
            encode_uncertified_count,
        }
    }
}

/// The honest cost breakdown of the encode tax (reviewer condition #3). Every
/// (row, atom) encode lands in exactly one of three tiers, in ascending cost:
///
///   1. **amortized-certified** — the one-mat-vec distilled predictor's start
///      already satisfies the Kantorovich `h ≤ ½` certificate. Cheapest.
///   2. **Newton-rescued** — the amortized start is uncertified, but the
///      certified IFT-warm-start Newton encode lands a certified root. Middling.
///   3. **multi-start fallback** — neither certifies, so the row rides the exact
///      multi-start solve. This is the true cost MULTIPLIER at scale, and its
///      fraction GROWS with atom similarity / co-activation interference (the
///      per-row joint `(t, a)` landscape multiplies basins that no per-atom
///      certificate covers). Reporting it is what keeps the encode-tax story
///      honest — an SAE's one-matmul encode has no analogue of this tail.
///
/// The tiers partition the (row, atom) grid: `amortized_certified +
/// newton_rescued + multistart_fallback == n_rows · n_atoms`.
#[derive(Debug, Clone, Default)]
pub struct FallbackTelemetry {
    pub n_rows: usize,
    pub n_atoms: usize,
    /// (row, atom) encodes certified by the cheap amortized predictor.
    pub amortized_certified: usize,
    /// (row, atom) encodes the amortized predictor missed but the certified
    /// Newton warm-start rescued.
    pub newton_rescued: usize,
    /// (row, atom) encodes neither tier certified — routed to the exact
    /// multi-start solve.
    pub multistart_fallback: usize,
}

impl FallbackTelemetry {
    /// Total (row, atom) encodes accounted for.
    #[must_use]
    pub fn total(&self) -> usize {
        self.n_rows * self.n_atoms
    }

    /// Fraction of encodes the cheap amortized predictor certified outright.
    #[must_use]
    pub fn amortized_fraction(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.amortized_certified as f64 / t as f64
        }
    }

    /// Fraction of encodes rescued by the certified Newton warm-start.
    #[must_use]
    pub fn newton_fraction(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.newton_rescued as f64 / t as f64
        }
    }

    /// Fraction of encodes that fell through to the exact multi-start solve —
    /// the encode-tax cost multiplier.
    #[must_use]
    pub fn multistart_fraction(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.multistart_fallback as f64 / t as f64
        }
    }

    /// Fold another atom's tallies into this one (n_rows is shared across atoms;
    /// n_atoms and the tier counts accumulate). Lets a caller aggregate the
    /// per-atom telemetry of [`EncodeAtlas::encode_atom_with_fallback_telemetry`]
    /// into one dictionary-wide breakdown.
    pub fn accumulate(&mut self, other: &FallbackTelemetry) {
        self.n_rows = other.n_rows;
        self.n_atoms += other.n_atoms;
        self.amortized_certified += other.amortized_certified;
        self.newton_rescued += other.newton_rescued;
        self.multistart_fallback += other.multistart_fallback;
    }
}

/// Per-row Kantorovich certificate at a start `t₀` for one atom encode.
#[derive(Debug, Clone, Copy)]
pub struct RowCertificate {
    pub beta: f64,
    pub eta: f64,
    pub lipschitz: f64,
    /// `h = β·η·L`. The row is certified iff `h ≤ ½`.
    pub h: f64,
}

impl RowCertificate {
    pub fn certified(&self) -> bool {
        self.h.is_finite() && self.h <= KANTOROVICH_THRESHOLD
    }
}

#[derive(Debug, Clone)]
struct CertifiedEncodeProbe {
    coord: Array1<f64>,
    final_cert: RowCertificate,
}

/// Canonical flat-axis polynomial degree of a cylinder `S¹ × ℝ` atom — the
/// degree the topology-race builder ([`gam_solve::structure_harvest`]) uses
/// for the line axis (`CylinderHarmonicEvaluator::new(_, 2)`). The encode atlas
/// recovers the circle harmonic count from the basis width using this degree, so
/// the two must agree.
pub(crate) const SAE_CYLINDER_LINE_DEGREE: usize = 2;

/// Build a basis-family handle for one atom from its [`SaeManifoldAtom`]. The
/// atlas needs to evaluate the jet sups, which live on the concrete evaluator
/// types; the atom carries the evaluator as `Arc<dyn SaeBasisEvaluator>`, so we
/// reconstruct the family bound from the atom's basis kind + width + centers.
///
/// The width used is the FULL inner-basis width [`SaeManifoldAtom::full_basis_size`],
/// never the stored (possibly #1117 rank-reduced) [`SaeManifoldAtom::basis_size`].
/// After [`SaeManifoldAtom::reduce_basis_to_subspace`] the live columns are
/// Q-mixtures `Φ̃ = Φ Q` of the fixed-width family, so a family rebuilt at the
/// REDUCED width bounds the wrong function space — a 5-wide periodic atom reduced
/// to `r = 3` would be bounded as a single-harmonic family (under-estimating the
/// `g`-th jet by `2^g`), and a degree-2 patch reduced to `r = 2` would be bounded
/// as affine (`L = 0` ⇒ every start "certifies": a FALSE Kantorovich
/// certificate, violating the module invariant that every bound over-estimates).
/// The sups returned here bound the FULL-width family; [`reconstruction_jet_sups`]
/// pairs them with the full-width decoder pre-image `B = Q B̃`, against which the
/// reconstruction is IDENTICAL (`Φ̃ B̃ = Φ (Q B̃)`), so the certificate frame never
/// sees the reduction and stays sound.
pub(crate) fn family_jet_sups(
    atom: &SaeManifoldAtom,
    chart: &ChartRegion,
) -> Result<JetSups, String> {
    use crate::manifold::SaeAtomBasisKind::*;
    let m = atom.full_basis_size();
    let d = atom.latent_dim;
    let sups = match &atom.basis_kind {
        Periodic => {
            let ev = PeriodicHarmonicEvaluator::new(m)?;
            JetSups::from_family(&ev, chart)
        }
        Torus => {
            // Torus basis width is `(2H+1)^d`; recover the per-axis harmonic
            // count `H` from `axis_m = m^(1/d)` rather than a sum formula.
            let axis_m = integer_root(m, d.max(1));
            let num_harmonics = axis_m.saturating_sub(1) / 2;
            let ev = TorusHarmonicEvaluator::new(d, num_harmonics.max(1))?;
            JetSups::from_family(&ev, chart)
        }
        Sphere => {
            let ev = SphereChartEvaluator;
            JetSups::from_family(&ev, chart)
        }
        Cylinder => {
            // Cylinder width is `(2H+1)·(D+1)` with the canonical flat-axis
            // degree `D = SAE_CYLINDER_LINE_DEGREE` (the harvest convention).
            // Recover the per-axis circle harmonic count `H` from
            // `2H+1 = m/(D+1)`.
            let ml = SAE_CYLINDER_LINE_DEGREE + 1;
            if d != 2 || ml == 0 || m % ml != 0 {
                return Err(format!(
                    "EncodeAtlas: Cylinder atom requires latent_dim == 2 and width divisible by {ml}; got dim={d}, m={m}"
                ));
            }
            let axis_mc = m / ml;
            let h = axis_mc.saturating_sub(1) / 2;
            let ev = CylinderHarmonicEvaluator::new(h.max(1), SAE_CYLINDER_LINE_DEGREE)?;
            JetSups::from_family(&ev, chart)
        }
        Mobius => {
            return Err(
                "EncodeAtlas: Mobius jet bounds require its persisted harmonic and width \
                 degrees; use the atom's exact analytic jets"
                    .to_string(),
            );
        }
        Linear | EuclideanPatch | Poincare => {
            // The patch width fixes max_degree implicitly; bound by a degree that
            // covers the column count (conservative). Degree d-patch column count
            // grows fast; we recover the smallest degree whose patch is ≥ m.
            // Poincare atoms use the same tangent-coordinate polynomial decoder;
            // their intrinsic smoothness differs in the penalty, not in Phi(t).
            let degree = euclidean_patch_degree(d, m);
            let ev = EuclideanPatchEvaluator::new(d, degree)?;
            JetSups::from_family(&ev, chart)
        }
        Duchon => {
            // UNSOUND — DO NOT TRUST for a certificate (F2/F3). This bound
            // hard-codes cubic `φ(r) = r³` radial jets and a single origin center
            // (`duchon_centers_from_atom`), but the real Duchon kernel is the
            // polyharmonic `c·r^(2m−d)` (with log variants) over the atom's
            // data-placed centers — so it can UNDER-estimate L and issue a FALSE
            // certificate (the module's own warning). The atom does not expose its
            // real order / center matrix / scaling to this crate, so no sound bound
            // is available. `build_atom_atlas_from_centers` therefore REFUSES to
            // certify Duchon atoms (emits uncertified charts → exact-encode
            // fallback) and never reaches this arm; it is retained only so the
            // family dispatch is total. If a future change threads the real
            // order/centers here, replace this with the true `φ_{m,d}` jet bounds.
            let centers = duchon_centers_from_atom(atom);
            let conservative_m = m.max(1);
            let ev = DuchonCoordinateEvaluator::new(centers, conservative_m)?;
            JetSups::from_family(&ev, chart)
        }
        Precomputed(name) => {
            return Err(format!(
                "EncodeAtlas: precomputed basis '{name}' has no closed-form jet sup; route to exact encode"
            ));
        }
        // A finite-set (indicator) atom is piecewise constant — it has no
        // continuous jet to bound, so there is no Kantorovich chart; route to the
        // exact encode like any other non-differentiable basis.
        FiniteSet => {
            return Err(
                "EncodeAtlas: finite-set (indicator) basis has no closed-form jet sup; \
                 route to exact encode"
                    .to_string(),
            );
        }
    };
    Ok(sups)
}

/// Smallest monomial-patch degree whose column count covers `m` basis columns.
pub(crate) fn euclidean_patch_degree(latent_dim: usize, m: usize) -> usize {
    // Column count of a degree-D patch in d vars is C(d+D, D). Grow D until it
    // covers m; cap at m so a degenerate width still terminates.
    let mut degree = 0usize;
    while patch_column_count(latent_dim, degree) < m && degree < m {
        degree += 1;
    }
    degree
}

/// Largest integer `a` with `a^k ≤ n` (the floor of the `k`-th root). Used to
/// recover the per-axis harmonic width `axis_m` from a torus basis width
/// `m = axis_m^d`.
pub(crate) fn integer_root(n: usize, k: usize) -> usize {
    if k == 0 {
        return 1;
    }
    if k == 1 {
        return n;
    }
    let mut a = 1usize;
    loop {
        let next = a + 1;
        let mut pow: u128 = 1;
        let mut overflow = false;
        for _ in 0..k {
            pow = pow.saturating_mul(next as u128);
            if pow > n as u128 {
                overflow = true;
                break;
            }
        }
        if overflow {
            return a;
        }
        a = next;
    }
}

pub(crate) fn patch_column_count(latent_dim: usize, degree: usize) -> usize {
    // C(d + D, D)
    let mut num = 1u128;
    let mut den = 1u128;
    for i in 1..=degree {
        num *= (latent_dim + i) as u128;
        den *= i as u128;
    }
    (num / den) as usize
}

/// Recover Duchon centers from an atom: when the evaluator is unavailable the
/// atlas falls back to the atom's own latent-coordinate hull as the center set,
/// which only affects the radial-tail bound conservatively.
pub(crate) fn duchon_centers_from_atom(atom: &SaeManifoldAtom) -> Array2<f64> {
    // One center at the origin in latent_dim space is a sound conservative
    // default: the chart's own r_min / r_max bracket the true radial range.
    Array2::<f64>::zeros((1, atom.latent_dim.max(1)))
}

/// The four per-column jet sups of a basis family over a chart.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JetSups {
    pub(crate) value: f64,
    pub(crate) jacobian: f64,
    pub(crate) hessian: f64,
    pub(crate) third: f64,
}

impl JetSups {
    pub(crate) fn from_family<B: BasisHessianLipschitz>(family: &B, chart: &ChartRegion) -> Self {
        Self {
            value: family.value_sup(chart),
            jacobian: family.jacobian_sup(chart),
            hessian: family.hessian_sup(chart),
            third: family.third_sup(chart),
        }
    }
}

/// Evaluate one atom's encode objective gradient `F(t) = ∇f_k(t)` and the FULL
/// Hessian `F'(t) = ∇²f_k(t)` at a single coordinate `t`, for a single target
/// row `x` and fixed amplitude `z`. With `m(t) = z·BᵀΦ(t)`, `r = m − x`,
/// `J_m = z·Bᵀ J_Φ`:
///
/// ```text
/// g_t[a]   = J_m[a] · r                                  (= ∇f)
/// H_tt[a,b] = J_m[a] · J_m[b] + r · ∂²m/∂t_a∂t_b         (= ∇²f, FULL Hessian)
/// ```
///
/// The certificate uses the FULL Hessian rather than the Gauss-Newton block
/// `J_mᵀ J_m`. This is the principled choice for Newton–Kantorovich: the
/// theorem certifies convergence of Newton on `F = ∇f` to the unique nearby
/// ROOT of `∇f`, but a root of `∇f` can be a maximum. The full Hessian is
/// positive-definite exactly on the genuine-minimum basin, so requiring
/// `λ_min(H) > 0` (finite `β`) is what flags a start that would otherwise let
/// Gauss-Newton march into the wrong root (e.g. the circle antipode, a local
/// max where `∇f = 0` but the full curvature is negative). The residual term
/// needs the basis second jet `∂²Φ/∂t²`; an evaluator without one returns
/// `None`, and the row is flagged (no silent Gauss-Newton fallback).
///
/// The Hessian returned is the TRUE `∇²f_k` — no Levenberg ridge is added
/// (F2). The Kantorovich certificate (`row_certificate`) and its `λ_min(H) > 0`
/// saddle gate must see the genuine field: a ridged `H + λI` certifies neither
/// the original objective nor a consistently regularized one (a
/// locally-constant reconstruction has `H = 0`, whose ridged `λI` would falsely
/// certify a non-isolated, non-unique root). Ridge stays only in the
/// UNCERTIFIED amortized predictor (`center_amortized_jacobian`/`center_beta`).
pub fn encode_grad_hess(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    t: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
) -> Result<Option<(Array1<f64>, Array2<f64>)>, String> {
    // The bare Euclidean, prior-free objective — the historical field, bit-identical
    // to the metric-free encode (see [`encode_grad_hess_core`], `EncodeObjective`).
    encode_grad_hess_core(
        atom,
        evaluator,
        t,
        x,
        amplitude,
        &EncodeObjective::euclidean(),
    )
}

/// The TRUE per-row encode objective's non-Euclidean ingredients (F3), so the
/// Kantorovich certificate certifies the SAME functional the fit optimized `t`
/// against — not a bare Euclidean stand-in.
///
/// The fit's per-row data loss is generalized least squares `½ rᵀ M_n r`
/// (`r = m(t) − x`) under a per-row output metric `M_n = U_n U_nᵀ`
/// ([`gam_problem::RowMetric`]), plus a per-axis latent coordinate prior
/// `Σ_a ArdAxisPrior(α_a, t_a)` (the ARD Gaussian / von-Mises energy the fit
/// placed on the coordinate). The metric-free encode drops BOTH, so its certified
/// root solves a different problem whenever a non-identity metric or an active
/// prior is present. `EncodeObjective` threads them back in:
///
/// * `metric_factor` — the per-row factor `U_n ∈ ℝ^{p×rank}`; `None` ⇒ `M = I`.
///   Whitening reduces to `M r = U(Uᵀr)` and `Jᵀ M J = (UᵀJ)ᵀ(UᵀJ)`.
/// * `prior_alpha` — per-axis ARD precision `α_a` (the caller folds in any row
///   weight); `None`/`0` ⇒ no prior on that axis. The period is read from the
///   atom's basis kind so a periodic axis uses the von-Mises energy.
/// * `metric_norm_bound` — a GLOBAL upper bound `max_n ‖M_n‖` used to scale the
///   offline chart Lipschitz (the per-row `M_n` enters `β, η` online; the
///   certificate needs a valid `L` upper bound, and `L_data` scales by `‖M‖`).
///
/// [`EncodeObjective::euclidean`] (all `None`, bound `1`) reproduces the metric-
/// free field bit-for-bit, so every existing caller is unchanged.
#[derive(Clone, Copy)]
pub struct EncodeObjective<'a> {
    /// Per-row output-metric factor `U ∈ ℝ^{p×rank}` (`M = U Uᵀ`). `None` ⇒ `I`.
    pub metric_factor: Option<ArrayView2<'a, f64>>,
    /// Per-axis ARD precision `α_a` (row weight folded in). `None` ⇒ no prior.
    pub prior_alpha: Option<&'a [f64]>,
    /// Global bound `max_n ‖M_n‖` scaling the offline chart Lipschitz. `1.0` for
    /// the Euclidean objective.
    pub metric_norm_bound: f64,
}

impl<'a> EncodeObjective<'a> {
    /// The bare Euclidean, prior-free objective — every code path is bit-identical
    /// to the metric-free encode under this value.
    pub fn euclidean() -> Self {
        Self {
            metric_factor: None,
            prior_alpha: None,
            metric_norm_bound: 1.0,
        }
    }

    /// Closed-form Lipschitz contribution of the latent prior's third derivative.
    /// The Gaussian (non-periodic) prior is quadratic, so `prior''' ≡ 0`; the
    /// von-Mises (periodic) prior has `hess = α cos(κt)` ⇒ `third = −α κ sin(κt)`,
    /// bounded by `α·κ` with `κ = 2π/period`. Summed over axes (a conservative
    /// bound on the diagonal third-order tensor's operator norm — over-estimating
    /// `L` only shrinks the certified radius, never certifies a divergent start).
    fn prior_lipschitz(&self, atom: &SaeManifoldAtom) -> f64 {
        let Some(alpha) = self.prior_alpha else {
            return 0.0;
        };
        let mut l = 0.0;
        for axis in 0..atom.latent_dim.min(alpha.len()) {
            if let Some(period) = latent_axis_period(atom, axis) {
                let kappa = std::f64::consts::TAU / period;
                l += alpha[axis].abs() * kappa;
            }
        }
        l
    }

    /// The chart's Kantorovich Lipschitz for the TRUE objective: the stored
    /// Euclidean data-term bound `data_lipschitz` scaled by the global metric
    /// operator-norm bound (`½ rᵀM r`'s Hessian-Lipschitz is `‖M‖·L_data`), plus
    /// the prior's third-derivative bound. Reduces to `data_lipschitz` exactly for
    /// [`Self::euclidean`] (`1·L + 0`).
    fn effective_lipschitz(&self, atom: &SaeManifoldAtom, data_lipschitz: f64) -> f64 {
        self.metric_norm_bound * data_lipschitz + self.prior_lipschitz(atom)
    }
}

/// Apply the per-row output metric `M = U Uᵀ` to a residual/tangent vector:
/// `M v = U (Uᵀ v)`, `U ∈ ℝ^{p×rank}`. `O(p·rank)`, never the dense `p×p`.
fn apply_row_metric(u: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Array1<f64> {
    let utv = u.t().dot(&v); // Uᵀ v ∈ ℝ^rank
    u.dot(&utv) // U (Uᵀ v) ∈ ℝ^p
}

/// Maximum Gauss--Newton iterations for the frozen-dictionary joint row solve.
const JOINT_ENCODE_MAX_ITER: usize = 64;
const JOINT_ENCODE_GRAD_TOL: f64 = 1.0e-10;
const JOINT_ENCODE_STEP_TOL: f64 = 1.0e-12;

/// Floor on the Levenberg--Marquardt damping `λ` for the joint row solve. The
/// damping is carried and decayed *across* outer Gauss--Newton iterations
/// (warm-started at this value on entry, then raised on rejection and lowered on
/// acceptance), so it is a stateful trust-region parameter rather than an
/// `escalate_ridge` schedule — it is deliberately kept hand-rolled. This is the
/// initial `λ` seed at the smallest scale that still perturbs the Hessian.
const JOINT_ENCODE_DAMPING_FLOOR: f64 = 1.0e-10;

/// Multiplicative growth applied to the LM damping `λ` whenever a damped step is
/// rejected (unfactorable, non-descent, or Armijo-exhausted). Numerically equal
/// to [`opt::constants::RIDGE_GROWTH`], but this loop is stateful across outer
/// iterations and is intentionally not routed through `escalate_ridge`.
const JOINT_ENCODE_DAMPING_GROWTH: f64 = 10.0;

/// Multiplicative decay applied to the LM damping `λ` after an accepted step, so
/// the next outer iteration starts from a looser trust region. Kept above the
/// `f64::EPSILON · diag_scale` floor at the use site.
const JOINT_ENCODE_DAMPING_DECAY: f64 = 3.0;

/// Maximum number of damping-escalation attempts within a single outer
/// Gauss--Newton iteration before the row is declared non-improvable.
const JOINT_ENCODE_DAMPING_MAX_ATTEMPTS: usize = 12;

/// Maximum number of Armijo backtracking halvings for the inner line search on
/// each damped step. Passed as `max_steps` to the shared
/// [`opt::backtracking_line_search`] primitive.
const JOINT_ENCODE_ARMIJO_MAX_STEPS: usize = 24;

fn joint_data_value_grad_hess(
    jac: ArrayView2<'_, f64>,
    residual: ArrayView1<'_, f64>,
    metric_factor: Option<ArrayView2<'_, f64>>,
) -> (f64, Array1<f64>, Array2<f64>) {
    let q = jac.nrows();
    let p = jac.ncols();
    let weighted_residual = match metric_factor.as_ref() {
        Some(u) => apply_row_metric(u.view(), residual),
        None => residual.to_owned(),
    };
    let value = 0.5 * residual.dot(&weighted_residual);
    let grad = jac.dot(&weighted_residual);
    let weighted_jac = match metric_factor.as_ref() {
        Some(u) => {
            let mut out = Array2::<f64>::zeros((q, p));
            for axis in 0..q {
                out.row_mut(axis)
                    .assign(&apply_row_metric(u.view(), jac.row(axis)));
            }
            out
        }
        None => jac.to_owned(),
    };
    let hess = jac.dot(&weighted_jac.t());
    (value, grad, hess)
}

/// Value, exact gradient, and positive-semidefinite Gauss--Newton curvature for
/// the shared-residual multi-atom objective. The gradient is exact; the PSD
/// curvature is used only to choose a descent step, so damping changes neither
/// the objective nor its stationary points.
fn joint_encode_value_grad_hess(
    atoms: &[SaeManifoldAtom],
    coords: &[Array1<f64>],
    x: ArrayView1<'_, f64>,
    amplitudes: ArrayView1<'_, f64>,
    metric_factor: Option<ArrayView2<'_, f64>>,
) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
    let k_atoms = atoms.len();
    if coords.len() != k_atoms || amplitudes.len() != k_atoms {
        return Err(format!(
            "joint encode: {} atoms require {} coordinate blocks and amplitudes; got {} and {}",
            k_atoms,
            k_atoms,
            coords.len(),
            amplitudes.len()
        ));
    }
    let p = x.len();
    if let Some(u) = metric_factor.as_ref() {
        if u.nrows() != p {
            return Err(format!(
                "joint encode: metric factor has {} rows but target has {p} outputs",
                u.nrows()
            ));
        }
    }

    let mut offsets = Vec::with_capacity(k_atoms + 1);
    offsets.push(0usize);
    for (atom_idx, atom) in atoms.iter().enumerate() {
        if atom.output_dim() != p {
            return Err(format!(
                "joint encode: atom {atom_idx} output_dim {} != target width {p}",
                atom.output_dim()
            ));
        }
        if coords[atom_idx].len() != atom.latent_dim {
            return Err(format!(
                "joint encode: atom {atom_idx} coordinate length {} != latent_dim {}",
                coords[atom_idx].len(),
                atom.latent_dim
            ));
        }
        offsets.push(offsets[atom_idx] + atom.latent_dim);
    }
    let q = *offsets.last().unwrap_or(&0);
    let mut recon = Array1::<f64>::zeros(p);
    let mut jac = Array2::<f64>::zeros((q, p));

    for (atom_idx, atom) in atoms.iter().enumerate() {
        let z = amplitudes[atom_idx];
        if !z.is_finite() {
            return Err(format!("joint encode: amplitude[{atom_idx}] is not finite"));
        }
        let Some(evaluator) = atom.basis_evaluator.as_ref() else {
            return Err(format!(
                "joint encode: atom {atom_idx} has no basis evaluator for its live coordinate block"
            ));
        };
        let d = atom.latent_dim;
        let m = atom.basis_size();
        let coord = coords[atom_idx]
            .view()
            .to_shape((1, d))
            .map_err(|e| format!("joint encode: atom {atom_idx} coordinate reshape: {e}"))?
            .to_owned();
        let (phi, dphi) = evaluator.evaluate(coord.view())?;
        if phi.dim() != (1, m) || dphi.dim() != (1, m, d) {
            return Err(format!(
                "joint encode: atom {atom_idx} evaluator returned phi {:?}, jet {:?}; expected (1,{m}) and (1,{m},{d})",
                phi.dim(),
                dphi.dim()
            ));
        }
        let start = offsets[atom_idx];
        for basis_col in 0..m {
            let phi_v = phi[[0, basis_col]];
            for out in 0..p {
                let b = atom.decoder_coefficients[[basis_col, out]];
                recon[out] += z * phi_v * b;
                for axis in 0..d {
                    jac[[start + axis, out]] += z * dphi[[0, basis_col, axis]] * b;
                }
            }
        }
    }

    let residual = &recon - &x;
    let (mut value, mut grad, mut hess) =
        joint_data_value_grad_hess(jac.view(), residual.view(), metric_factor.clone());

    for (atom_idx, atom) in atoms.iter().enumerate() {
        let Some(alpha) = atom.ard_precisions.as_deref() else {
            continue;
        };
        let start = offsets[atom_idx];
        for axis in 0..atom.latent_dim.min(alpha.len()) {
            if alpha[axis] == 0.0 {
                continue;
            }
            let prior = crate::manifold::ArdAxisPrior::eval(
                alpha[axis],
                coords[atom_idx][axis],
                latent_axis_period(atom, axis),
            );
            value += prior.value;
            grad[start + axis] += prior.grad;
            hess[[start + axis, start + axis]] += prior.hess.max(0.0);
        }
    }
    Ok((value, grad, hess))
}

fn joint_encode_damped_step(
    hess: ArrayView2<'_, f64>,
    grad: ArrayView1<'_, f64>,
    damping: f64,
) -> Result<Option<Array1<f64>>, String> {
    let q = grad.len();
    if q == 0 {
        return Ok(Some(Array1::zeros(0)));
    }
    let mut system = Array2::<f64>::zeros((q, q));
    for i in 0..q {
        for j in 0..q {
            system[[i, j]] = 0.5 * (hess[[i, j]] + hess[[j, i]]);
        }
        system[[i, i]] += damping;
    }
    let (evals, evecs) = system
        .eigh(Side::Lower)
        .map_err(|e| format!("joint encode: damped eigensolve failed: {e:?}"))?;
    if evals.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return Ok(None);
    }
    let mut step = Array1::<f64>::zeros(q);
    for (col, &lambda) in evals.iter().enumerate() {
        let v = evecs.column(col);
        let coefficient = -v.dot(&grad) / lambda;
        for row in 0..q {
            step[row] += coefficient * v[row];
        }
    }
    if step.iter().any(|v| !v.is_finite()) {
        Ok(None)
    } else {
        Ok(Some(step))
    }
}

fn joint_encode_add_step(
    atoms: &[SaeManifoldAtom],
    coords: &[Array1<f64>],
    step: ArrayView1<'_, f64>,
    scale: f64,
) -> Vec<Array1<f64>> {
    let mut out = Vec::with_capacity(atoms.len());
    let mut offset = 0usize;
    for (atom_idx, atom) in atoms.iter().enumerate() {
        let mut next = coords[atom_idx].clone();
        for axis in 0..atom.latent_dim {
            next[axis] += scale * step[offset + axis];
            if let Some(period) = latent_axis_period(atom, axis) {
                next[axis] = next[axis].rem_euclid(period);
            }
        }
        offset += atom.latent_dim;
        out.push(next);
    }
    out
}

/// Refine one row against all co-active atoms using the shared reconstruction
/// residual. Independent atlas projections may be supplied as starts, but every
/// accepted step and the convergence test use
/// `Σ_k z_k B_kᵀΦ_k(t_k) - x`, including all cross-atom Jacobian blocks.
pub(crate) fn joint_encode_refine_row(
    atoms: &[SaeManifoldAtom],
    initial_coords: &[Array1<f64>],
    x: ArrayView1<'_, f64>,
    amplitudes: ArrayView1<'_, f64>,
    metric_factor: Option<ArrayView2<'_, f64>>,
) -> Result<(Vec<Array1<f64>>, bool), String> {
    let mut coords = initial_coords.to_vec();
    let q: usize = atoms.iter().map(|a| a.latent_dim).sum();
    if q == 0 {
        return Ok((coords, true));
    }
    let target_scale = 1.0 + x.dot(&x).sqrt();
    let mut damping = JOINT_ENCODE_DAMPING_FLOOR;

    for _ in 0..JOINT_ENCODE_MAX_ITER {
        let (value, grad, hess) =
            joint_encode_value_grad_hess(atoms, &coords, x, amplitudes, metric_factor.clone())?;
        let grad_norm = grad.dot(&grad).sqrt();
        if grad_norm <= JOINT_ENCODE_GRAD_TOL * target_scale {
            return Ok((coords, true));
        }
        let diag_scale = (0..q)
            .map(|i| hess[[i, i]].abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        damping = damping.max(f64::EPSILON * diag_scale);

        let mut accepted = None;
        for _ in 0..JOINT_ENCODE_DAMPING_MAX_ATTEMPTS {
            let Some(step) = joint_encode_damped_step(hess.view(), grad.view(), damping)? else {
                damping *= JOINT_ENCODE_DAMPING_GROWTH;
                continue;
            };
            let directional = grad.dot(&step);
            if !(directional.is_finite() && directional < 0.0) {
                damping *= JOINT_ENCODE_DAMPING_GROWTH;
                continue;
            }
            // Armijo backtracking on the shared reconstruction objective, migrated
            // onto the shared `opt` primitive with bit-for-bit-identical semantics:
            // initial step 1.0, `BACKTRACK_CONTRACTION` (×0.5) contraction,
            // `JOINT_ENCODE_ARMIJO_MAX_STEPS` (24) trials, and the exact
            // sufficient-decrease test `F(t + s·d) ≤ F(t) + c₁·s·∇F·d`
            // (c₁ = `ARMIJO_C1` = 1e-4, no roundoff cushion — the pre-migration
            // loop had none). `trial(s)` evaluates the candidate coords (always
            // well defined, so never `Ok(None)`) and threads them through the
            // payload so the accepted trial is returned without recomputation.
            let base_value = value;
            let step_unit_norm = step.dot(&step).sqrt();
            let line_search = backtracking_line_search::<Vec<Array1<f64>>, String>(
                BacktrackConfig {
                    initial_step: 1.0,
                    contraction: BACKTRACK_CONTRACTION,
                    max_steps: JOINT_ENCODE_ARMIJO_MAX_STEPS,
                },
                |line_scale| {
                    let candidate = joint_encode_add_step(atoms, &coords, step.view(), line_scale);
                    let (candidate_value, _, _) = joint_encode_value_grad_hess(
                        atoms,
                        &candidate,
                        x,
                        amplitudes,
                        metric_factor.clone(),
                    )?;
                    Ok(Some((candidate_value, candidate)))
                },
                |line_scale, candidate_value| {
                    candidate_value <= base_value + ARMIJO_C1 * line_scale * directional
                },
            )?;
            if let Some(AcceptedStep {
                step: line_scale,
                payload: candidate,
                ..
            }) = line_search
            {
                accepted = Some((candidate, line_scale * step_unit_norm));
            }
            if accepted.is_some() {
                damping = (damping / JOINT_ENCODE_DAMPING_DECAY).max(f64::EPSILON * diag_scale);
                break;
            }
            damping *= JOINT_ENCODE_DAMPING_GROWTH;
        }
        let Some((next, step_norm)) = accepted else {
            return Ok((coords, false));
        };
        coords = next;
        if step_norm <= JOINT_ENCODE_STEP_TOL * target_scale {
            let (_, final_grad, _) =
                joint_encode_value_grad_hess(atoms, &coords, x, amplitudes, metric_factor.clone())?;
            let converged =
                final_grad.dot(&final_grad).sqrt() <= JOINT_ENCODE_GRAD_TOL * target_scale;
            return Ok((coords, converged));
        }
    }
    let (_, final_grad, _) =
        joint_encode_value_grad_hess(atoms, &coords, x, amplitudes, metric_factor)?;
    let converged = final_grad.dot(&final_grad).sqrt() <= JOINT_ENCODE_GRAD_TOL * target_scale;
    Ok((coords, converged))
}

/// Objective-aware gradient/Hessian of the certified encode field (F3). With
/// [`EncodeObjective::euclidean`] this is bit-for-bit the historical metric-free
/// field; with a metric it whitens the residual through `M = U Uᵀ`, and with a
/// prior it adds the ARD/von-Mises gradient and (diagonal) Hessian.
pub(crate) fn encode_grad_hess_core(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    t: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    objective: &EncodeObjective<'_>,
) -> Result<Option<(Array1<f64>, Array2<f64>)>, String> {
    let d = atom.latent_dim;
    let p = atom.output_dim();
    let m = atom.basis_size();
    let coords = t.to_shape((1, d)).map_err(|e| e.to_string())?.to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    if phi.dim() != (1, m) {
        return Err(format!(
            "encode_grad_hess: evaluator returned phi {:?}, expected (1, {m})",
            phi.dim()
        ));
    }
    let decoder = &atom.decoder_coefficients;
    // Reconstruction m(t) = z · Bᵀ Φ(t)  ∈ ℝᵖ.
    let mut recon = Array1::<f64>::zeros(p);
    for basis_col in 0..m {
        let phi_v = phi[[0, basis_col]];
        if phi_v == 0.0 {
            continue;
        }
        for out in 0..p {
            recon[out] += amplitude * phi_v * decoder[[basis_col, out]];
        }
    }
    let residual = &recon - &x;
    // J_m[axis] = z · Bᵀ (∂Φ/∂t_axis)  ∈ ℝᵖ.
    let mut jm = Array2::<f64>::zeros((d, p));
    for axis in 0..d {
        for basis_col in 0..m {
            let dphi = jet[[0, basis_col, axis]];
            if dphi == 0.0 {
                continue;
            }
            for out in 0..p {
                jm[[axis, out]] += amplitude * dphi * decoder[[basis_col, out]];
            }
        }
    }
    // The full-Hessian residual term needs ∂²Φ/∂t². No second jet ⇒ no
    // certificate (flag), never a silent Gauss-Newton substitute.
    let second = match evaluator.second_jet_dyn(coords.view()) {
        Some(result) => result?,
        None => return Ok(None),
    };
    // F3 — metric whitening. The certified objective is `½ rᵀ M r`, so the field
    // reads the M-weighted residual `M r` and M-weighted image tangents `M J_m[a]`.
    // With no metric (`None`) `wr` aliases `residual` and `jb` aliases `jm.row(b)`,
    // so the assembly below is the historical Euclidean field bit-for-bit.
    let mr_owned;
    let wr: &Array1<f64> = match objective.metric_factor {
        Some(u) => {
            mr_owned = apply_row_metric(u, residual.view());
            &mr_owned
        }
        None => &residual,
    };
    let mjm: Option<Vec<Array1<f64>>> = objective
        .metric_factor
        .map(|u| (0..d).map(|a| apply_row_metric(u, jm.row(a))).collect());
    // (M-weighted) residual · decoder-row is INDEPENDENT of the (a,b) axes; hoist it
    // to one O(m·p) pass so the per-axis curvature term is a cheap O(m) dot.
    let mut rd = vec![0.0_f64; m];
    for (basis_col, rd_col) in rd.iter_mut().enumerate() {
        let mut dot = 0.0;
        for out in 0..p {
            dot += wr[out] * decoder[[basis_col, out]];
        }
        *rd_col = dot;
    }
    // g_t[a] = J_m[a] · (M r) ;  H_tt[a,b] = J_m[a]·(M J_m[b]) + (M r)·∂²m/∂t_a∂t_b.
    // The full Hessian is symmetric (Gauss-Newton block + symmetric second jet), so
    // compute the upper triangle and mirror — half the curvature work.
    let mut g = Array1::<f64>::zeros(d);
    let mut h = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        let ja = jm.row(a);
        g[a] = ja.dot(wr);
        for b in a..d {
            // Gauss-Newton block `J_aᵀ M J_b` (Euclidean: `J_aᵀ J_b`).
            let mut hab = match &mjm {
                Some(v) => ja.dot(&v[b]),
                None => ja.dot(&jm.row(b)),
            };
            // (M-weighted) residual · second-jet curvature: (M r) · ∂²m_{ab},
            // ∂²m_{ab}[out] = z · Σ_basis (∂²Φ/∂t_a∂t_b) · B[basis, out].
            let mut curv = 0.0;
            for basis_col in 0..m {
                let d2phi = second[[0, basis_col, a, b]];
                if d2phi == 0.0 {
                    continue;
                }
                curv += amplitude * d2phi * rd[basis_col];
            }
            hab += curv;
            h[[a, b]] = hab;
            h[[b, a]] = hab;
        }
    }
    // F3 — latent coordinate prior. The fit placed an ARD (Gaussian) / von-Mises
    // (periodic) prior on `t`; its energy enters the certified objective, so its
    // gradient and (per-axis diagonal) Hessian enter the Newton field. Absent for
    // `EncodeObjective::euclidean` (no allocation, no arithmetic).
    if let Some(alpha) = objective.prior_alpha {
        for axis in 0..d.min(alpha.len()) {
            let alpha_axis = alpha[axis];
            if alpha_axis == 0.0 {
                continue;
            }
            let pr = crate::manifold::ArdAxisPrior::eval(
                alpha_axis,
                t[axis],
                latent_axis_period(atom, axis),
            );
            g[axis] += pr.grad;
            h[[axis, axis]] += pr.hess;
        }
    }
    // NO ridge: the certificate must use the TRUE Hessian (F2). See the doc above.
    Ok(Some((g, h)))
}

/// Operator-norm of `H⁻¹` (i.e. `β = 1/λ_min(H)`) and the Newton step
/// `δ = −H⁻¹ g` with `η = ‖δ‖`, from a symmetric PSD `H` and gradient `g`.
/// Returns `None` when `H` is numerically singular (λ_min ≤ 0) — an
/// uncertifiable start.
pub(crate) fn beta_eta_newton(
    h: ArrayView2<'_, f64>,
    g: ArrayView1<'_, f64>,
) -> Result<Option<(f64, f64, Array1<f64>)>, String> {
    // Closed-form fast paths for the tiny latent dims that dominate SAE atoms
    // (`d = 1, 2`), avoiding a faer eigendecomposition + its heap allocations on
    // the hottest per-row Newton inner loop. `β = 1/λ_min(H)`, `δ = −H⁻¹g`, and the
    // `λ_min ≤ 0` gate are computed directly; the symmetric-`H` reads mirror
    // `eigh(Side::Lower)` (which uses the lower triangle) exactly.
    let d = h.nrows();
    if d == 1 {
        // A coordinate direction is certifiable only when the actual derivative
        // F'(t) has strictly positive, resolved curvature.
        let h00 = h[[0, 0]];
        if !(h00.is_finite() && h00 > 0.0) {
            return Ok(None);
        }
        let delta0 = -g[0] / h00;
        let mut delta = Array1::<f64>::zeros(1);
        delta[0] = delta0;
        return Ok(Some((1.0 / h00, delta0.abs(), delta)));
    }
    if d == 2 {
        // Symmetric H = [[a, b], [b, c]] read from the lower triangle.
        let a = h[[0, 0]];
        let b = h[[1, 0]];
        let c = h[[1, 1]];
        let tr = a + c;
        let det = a * c - b * b;
        // λ_min = ½(tr − √((a−c)² + 4b²)), λ_max = ½(tr + √…); ≥ 0 ⇒ PSD.
        let disc = ((a - c) * (a - c) + 4.0 * b * b).max(0.0).sqrt();
        let lambda_min = 0.5 * (tr - disc);
        let lambda_max = 0.5 * (tr + disc);
        let max_abs = lambda_min.abs().max(lambda_max.abs());
        if !(lambda_min.is_finite() && lambda_max.is_finite() && max_abs > 0.0) {
            return Ok(None);
        }
        let floor = gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR * max_abs;
        // Healthy PD block — smallest eigenvalue clears the numerical null band —
        // keeps the closed-form fast path. A negative or unresolved eigenvalue is
        // not invertible as F'(t), so ordinary Kantorovich cannot certify it.
        if lambda_min > floor {
            // δ = −H⁻¹g with H⁻¹ = [[c, −b], [−b, a]] / det (det = λ_min·λ_max > 0).
            let inv_det = 1.0 / det;
            let g0 = g[0];
            let g1 = g[1];
            let d0 = -(c * g0 - b * g1) * inv_det;
            let d1 = -(a * g1 - b * g0) * inv_det;
            if !(d0.is_finite() && d1.is_finite()) {
                return Ok(None);
            }
            let mut delta = Array1::<f64>::zeros(2);
            delta[0] = d0;
            delta[1] = d1;
            let eta = (d0 * d0 + d1 * d1).sqrt();
            return Ok(Some((1.0 / lambda_min, eta, delta)));
        }
        return Ok(None);
    }
    beta_eta_newton_positive_definite(h, g)
}

/// Spectral path for `d ≥ 3`. Ordinary Newton--Kantorovich requires the inverse
/// of the actual derivative `F'(t) = H`; quotienting or replacing a null
/// eigenvalue changes that derivative and therefore cannot certify the returned
/// iteration. We consequently accept only a numerically resolved positive-
/// definite Hessian and report every singular/indefinite start as uncertified.
fn beta_eta_newton_positive_definite(
    h: ArrayView2<'_, f64>,
    g: ArrayView1<'_, f64>,
) -> Result<Option<(f64, f64, Array1<f64>)>, String> {
    let d = h.nrows();
    // Symmetrise defensively before the eigendecomposition (the assembled Hessian
    // is symmetric only up to reduction order), mirroring the evidence routine.
    let mut sym = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let v = 0.5 * (h[[i, j]] + h[[j, i]]);
            if !v.is_finite() {
                return Ok(None);
            }
            sym[[i, j]] = v;
        }
    }
    let (vals, vecs) = sym
        .eigh(Side::Lower)
        .map_err(|e| format!("beta_eta_newton: eigh failed: {e:?}"))?;
    let max_abs = vals.iter().fold(
        0.0_f64,
        |acc, &v| if v.is_finite() { acc.max(v.abs()) } else { acc },
    );
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return Ok(None);
    }
    let floor = gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR * max_abs;
    if vals
        .iter()
        .any(|&lambda| !lambda.is_finite() || lambda <= floor)
    {
        return Ok(None);
    }
    let lambda_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    if !(lambda_min.is_finite() && lambda_min > 0.0) {
        return Ok(None);
    }
    let beta = 1.0 / lambda_min;
    // Newton step δ = −H⁻¹g via the eigendecomposition of the actual Hessian.
    let mut delta = Array1::<f64>::zeros(d);
    for (col, &lam) in vals.iter().enumerate() {
        let vi = vecs.column(col);
        let coeff = vi.dot(&g) / lam;
        for row in 0..d {
            delta[row] -= coeff * vi[row];
        }
    }
    if delta.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    let eta = delta.dot(&delta).sqrt();
    Ok(Some((beta, eta, delta)))
}

/// Compute the per-row Kantorovich certificate for encoding target row `x`
/// against atom `atom` at start coordinate `t₀`, with fixed amplitude `z` and
/// the chart's closed-form Lipschitz constant `lipschitz`. Returns the
/// certificate AND the Newton step `δ = −H⁻¹ g` so the caller can advance.
pub fn row_certificate(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    t0: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    lipschitz: f64,
) -> Result<(RowCertificate, Array1<f64>), String> {
    // Euclidean, prior-free objective — bit-identical to the metric-free encode.
    row_certificate_core(
        atom,
        evaluator,
        t0,
        x,
        amplitude,
        lipschitz,
        &EncodeObjective::euclidean(),
    )
}

/// Objective-aware [`row_certificate`] (F3): the certificate `h = β·η·L` is
/// computed from the TRUE objective's gradient/Hessian ([`encode_grad_hess_core`])
/// so it certifies the metric- and prior-weighted field. `lipschitz` must already
/// be the objective's effective bound ([`EncodeObjective::effective_lipschitz`]).
pub(crate) fn row_certificate_core(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    t0: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    lipschitz: f64,
    objective: &EncodeObjective<'_>,
) -> Result<(RowCertificate, Array1<f64>), String> {
    let uncertified = || {
        (
            RowCertificate {
                beta: f64::INFINITY,
                eta: f64::INFINITY,
                lipschitz,
                h: f64::INFINITY,
            },
            Array1::<f64>::zeros(atom.latent_dim),
        )
    };
    // No second jet ⇒ no full Hessian ⇒ uncertifiable (flag).
    let Some((g, h)) = encode_grad_hess_core(atom, evaluator, t0, x, amplitude, objective)? else {
        return Ok(uncertified());
    };
    match beta_eta_newton(h.view(), g.view())? {
        Some((beta, eta, delta)) => {
            let cert = RowCertificate {
                beta,
                eta,
                lipschitz,
                h: beta * eta * lipschitz,
            };
            Ok((cert, delta))
        }
        // Indefinite / negative-curvature full Hessian: the start is at or past
        // a basin boundary (a max/saddle of f), not the minimum basin — flag.
        None => Ok(uncertified()),
    }
}

fn uncertified_certificate(lipschitz: f64) -> RowCertificate {
    RowCertificate {
        beta: f64::INFINITY,
        eta: f64::INFINITY,
        lipschitz,
        h: f64::INFINITY,
    }
}

fn refine_certified_start(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    mut t: Array1<f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    lipschitz: f64,
    newton_steps: usize,
    initial_cert: RowCertificate,
    mut delta: Array1<f64>,
    chart_center: ArrayView1<'_, f64>,
    chart_radius: f64,
    objective: &EncodeObjective<'_>,
) -> Result<Option<CertifiedEncodeProbe>, String> {
    assert!(initial_cert.certified());
    let mut final_cert = initial_cert;
    for _ in 0..newton_steps {
        // Convergence early-exit: the pending Newton step is below the coordinate
        // ULP scale, so `t + δ == t` to f64 resolution — the certified root is
        // reached and the remaining fixed-budget steps would only re-accumulate
        // round-off. This is where the well-conditioned quadratic Newton tail's
        // redundant `evaluate` + `second_jet` work is eliminated.
        if delta.dot(&delta).sqrt() <= NEWTON_REFINE_CONVERGED_EPS * (1.0 + t.dot(&t).sqrt()) {
            break;
        }
        let next = &t + &delta;
        // SOUNDNESS GUARD — same containment rule as `certify_with_basin_warmup`:
        // `lipschitz` is only a valid Hessian-Lipschitz bound inside this chart's
        // ball for the chart-local families. A refine iterate that leaves the ball
        // would have its certificate recomputed below with an `L` that no longer
        // bounds the true geometry there, so `h ≤ ½` would NOT imply Kantorovich
        // convergence — and the in-hand certificate at the previous iterate is
        // itself suspect (its guarantee needs `L` valid on the Newton sequence's
        // ball, part of which now lies outside the chart). Refuse and flag for the
        // exact fallback, exactly as the warm-up does. Wrap-aware distance, so
        // periodic-seam iterates are measured in the true manifold geometry.
        if latent_coordinate_distance(atom, next.view(), chart_center) > chart_radius {
            return Ok(None);
        }
        t = next;
        let (cert, next_delta) = row_certificate_core(
            atom,
            evaluator,
            t.view(),
            x,
            amplitude,
            lipschitz,
            objective,
        )?;
        if !cert.certified() {
            return Ok(None);
        }
        final_cert = cert;
        delta = next_delta;
    }
    Ok(Some(CertifiedEncodeProbe {
        coord: t,
        final_cert,
    }))
}

/// Certify an encode probe from `t_start`, navigating into the Kantorovich basin
/// first if needed (#1154/#1026). The Kantorovich quantity `h = β·η·L` scales with
/// amplitude through `L`, so at unit amplitude a positive-definite chart-center /
/// distilled start can sit OUTSIDE the certified ball (`h > ½`). Rather than
/// flagging it uncertified immediately — which made the encoder certify ZERO
/// held-out rows at amplitude 1.0 and fall back to the exact solve for everything —
/// take plain Newton steps toward the root, re-certifying at each iterate, while
/// the Kantorovich quantity `h = β·η·L` keeps CONTRACTING toward the ½ bound. The
/// certificate at the landing point is a full Kantorovich guarantee from there
/// (`h ≤ ½` ⇒ Newton converges to the in-ball root), so this only ever WIDENS the
/// certified set; it never certifies a non-convergent start.
///
/// Termination is the natural Newton stopping rule — there is no arbitrary step
/// budget. The warm-up stops and flags for the exact fallback when either the start
/// is not steppable (indefinite / non-finite Hessian — at or past a basin boundary)
/// or a step fails to reduce `h` (the iterate is not approaching a certifiable
/// in-chart root: its root lies outside this chart's valid Lipschitz region, or the
/// start was past the basin — empirically the rows that miss *plateau*, so more
/// steps cannot help; the lever there is denser charts, not more iterations). On
/// success the start is refined `newton_steps` further by [`refine_certified_start`].
/// Minimum per-step *multiplicative* decrease of the Kantorovich `h` the basin
/// warm-up requires to keep stepping (FIX #4). A tiny geometric floor: a
/// continued step must contract `h` by at least this fraction. Chosen small so
/// it never bites a genuinely converging row (Newton in the Kantorovich regime
/// is at least geometric and quadratic once `h < 1`, contracting `h` far faster
/// than `1/64` per step) while still forcing termination on a plateau.
///
/// PRICED (#2071): the value only sets the termination bound
/// `N < ln(2·h₀) / −ln(1 − c)` (below) — it is a floor on "real progress", not a
/// tuning parameter, so any small `c ∈ (0, 1)` is correct; `1/64 = 2^-6` is priced
/// for its bound. What breaks at 10×: at `c = 1/6.4` the floor starts rejecting
/// slow-but-genuine geometric contractions near the `½` certifiable boundary
/// (false plateaus → premature exact fallback); at `c = 1/640` the plateau bound
/// loosens ~10× (`N` grows from a few hundred to a few thousand
/// `row_certificate` solves on a pathological row). `1/64` keeps the bound at a
/// few hundred while leaving a wide margin below Newton's actual contraction.
const WARMUP_MIN_MULTIPLICATIVE_DECREASE: f64 = 1.0 / 64.0;

/// Kantorovich-quadratic acceptance coefficient for the basin warm-up (FIX #4).
/// Once `h < 1` a converging Newton step contracts quadratically (`h_new ≲ κ·h²`);
/// accepting that path in addition to the geometric floor makes the
/// "genuinely-converging rows are untouched" guarantee explicit. Kept `< 1` so
/// the quadratic path is itself a strict contraction (for `h < 1`,
/// `κ·h² < κ·h < h`), which preserves the termination bound below.
///
/// PRICED (#2071): the only constraint the termination proof imposes is `κ < 1`
/// (so the quadratic branch stays a strict contraction and cannot defeat the
/// geometric bound); `0.5` is the natural centre of `(0, 1)`, a factor-2 margin
/// below the `κ = 1` boundary. What breaks at 10×: `κ = 5` violates `κ < 1` and
/// the quadratic branch could ACCEPT an expanding step (`κ·h² > h` for
/// `h > 1/κ`), breaking termination; `κ = 0.05` merely tightens the quadratic
/// acceptance (fewer rows take the quadratic branch, more fall to the geometric
/// floor) with no correctness effect. Anything in `(0, 1)` is sound; `0.5` maximises
/// the margin.
const WARMUP_QUADRATIC_KAPPA: f64 = 0.5;

/// Sufficient-decrease test for the basin-warmup loop (FIX #4).
///
/// Returns `true` while the warm-up should keep stepping. The previous exit rule
/// (`h_new >= h_prev` — strict decrease or quit) never fires for an `h`-sequence
/// that decreases *monotonically toward a limit above ½*: the increments fall
/// below one ulp long before `h` crosses the certifiable ½ bound, so a single
/// pathological row could spin ~1e15 full `row_certificate` solves (Hessian
/// build + solve) on the encode hot path. We instead require genuine
/// *multiplicative* progress each step, which matches Newton's actual behavior:
/// a healthy contraction clears the geometric floor by a wide margin (and the
/// quadratic path once `h < 1`), while a plateau (`h_new/h_prev → 1`) satisfies
/// neither branch and flags to the exact fallback.
///
/// Termination bound: the warm-up loop runs only while the row is uncertified
/// (`h_prev > ½`), and every continued step contracts `h` by at least the factor
/// `(1 − WARMUP_MIN_MULTIPLICATIVE_DECREASE)` (the quadratic branch, for `h < 1`,
/// contracts by `κ·h_prev < κ < 1`, i.e. even harder). Hence after `N` continued
/// steps `h ≤ (1 − c)^N · h₀`, and since the loop stops once `h ≤ ½` we get
/// `N < ln(2·h₀) / −ln(1 − c)` — a few hundred iterations at most, versus
/// unbounded before. No arbitrary fixed step cap is imposed; the bound is a
/// consequence of the contraction requirement, in keeping with the function's
/// no-magic-budget design.
fn warmup_progress_sufficient(h_new: f64, h_prev: f64) -> bool {
    if !(h_new.is_finite() && h_prev.is_finite()) {
        return false;
    }
    // Geometric floor: a real Newton contraction easily clears this.
    if h_new <= (1.0 - WARMUP_MIN_MULTIPLICATIVE_DECREASE) * h_prev {
        return true;
    }
    // Kantorovich-quadratic path (only once `h < 1`, where it is a strict
    // contraction): makes the no-regression guarantee for converging rows
    // explicit without ever admitting a non-contracting (plateau) step.
    h_prev < 1.0 && h_new <= WARMUP_QUADRATIC_KAPPA * h_prev * h_prev
}

/// Whether the basin warm-up must REJECT the just-taken step (flag to the exact
/// fallback). FIX (F6): a step that just crossed into the certified region
/// (`h ≤ ½`) is ALWAYS accepted, even if its multiplicative decrease was below
/// the progress floor (e.g. `h: 0.501 → 0.499`, a legitimate but tiny cross of
/// the ½ bound). Only a step that is STILL uncertified AND failed to make
/// sufficient multiplicative progress is a plateau that must flag. The old code
/// ran the progress test unconditionally, so it could reject an in-hand
/// certificate and push the row to the exact fallback for nothing (a false
/// negative).
fn warmup_should_reject(next_certified: bool, h_new: f64, h_prev: f64) -> bool {
    !next_certified && !warmup_progress_sufficient(h_new, h_prev)
}

fn certify_with_basin_warmup(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    t_start: Array1<f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    lipschitz: f64,
    newton_steps: usize,
    chart_center: ArrayView1<'_, f64>,
    chart_radius: f64,
    objective: &EncodeObjective<'_>,
) -> Result<Option<CertifiedEncodeProbe>, String> {
    // SOUNDNESS GUARD: `lipschitz` is the chart's Hessian-Lipschitz sup, which is
    // only a valid bound over this chart's ball `‖t − center‖ ≤ radius` for the
    // chart-local families (`EuclideanPatch`/`Linear`/`Poincare` monomial patches,
    // `Cylinder` line axis, `Duchon` radial kernels). If a warm-up iterate leaves
    // that ball, `row_certificate` would compute `h = β·η·L` with an `L` that no
    // longer bounds the true geometry there, so `h ≤ ½` would NOT imply Kantorovich
    // convergence — a false certificate. (The `h`-contraction check does NOT catch
    // this: `h` can decrease monotonically toward an out-of-chart root the whole
    // way.) So we keep every certified iterate inside the chart; a row whose root is
    // outside this chart flags for the exact fallback — its lever is a denser grid,
    // not a step using an invalid `L`. Global-`L` families (periodic/torus/sphere)
    // route their points to charts whose centers are near the root, so the guard
    // rarely trips for them, and where it does the row was out-of-chart anyway.
    let in_chart = |t: &Array1<f64>| -> bool {
        // Wrap-aware chart containment. A raw Euclidean latent distance
        // `Σ(tᵢ − centerᵢ)²` mis-measures separation across the wrap seam of a
        // periodic axis: an iterate at `t = 0.99` against a chart centered at
        // `0.01` reads distance `0.98` instead of the true circle distance
        // `0.02`, so it is wrongly rejected at the start check and at every step
        // check — silently disabling the amortized encoder for a phase-localized
        // band of rows on every periodic / torus / cylinder-angle /
        // sphere-longitude chart and pushing that band to the multi-start
        // fallback. `latent_coordinate_distance` folds each axis onto its
        // `latent_axis_period`, so the containment ball is measured in the true
        // manifold geometry — exactly the geometry the chart's Lipschitz bound
        // `L` is valid over. This only ever moves points that are genuinely
        // near the center (small wrapped distance) back INTO the chart where
        // `L` holds, so it widens acceptance without weakening the soundness
        // guard (an iterate truly far from the center on the circle still fails).
        latent_coordinate_distance(atom, t.view(), chart_center) <= chart_radius
    };
    let mut t = t_start;
    // The distilled / chart-center start must itself be in-chart for its certificate
    // to be valid; a bad IFT prediction landing outside the chart is uncertifiable.
    if !in_chart(&t) {
        return Ok(None);
    }
    let (mut cert, mut delta) = row_certificate_core(
        atom,
        evaluator,
        t.view(),
        x,
        amplitude,
        lipschitz,
        objective,
    )?;
    while !cert.certified() {
        // Not steppable (indefinite / non-finite Hessian): flag.
        if !(cert.h.is_finite() && cert.beta.is_finite() && cert.eta.is_finite()) {
            return Ok(None);
        }
        let prev_h = cert.h;
        let next = &t + &delta;
        // Refuse to step where the chart's `L` is no longer valid (see guard above).
        if !in_chart(&next) {
            return Ok(None);
        }
        t = next;
        let (next_cert, next_delta) = row_certificate_core(
            atom,
            evaluator,
            t.view(),
            x,
            amplitude,
            lipschitz,
            objective,
        )?;
        cert = next_cert;
        delta = next_delta;
        // The warm-up only helps while h keeps *multiplicatively* contracting
        // toward ½. A plain strict-decrease test (`h >= prev_h`) never fires for
        // a sequence that decreases monotonically toward a limit above ½, so it
        // could spin ~1e15 `row_certificate` solves for one row; require genuine
        // multiplicative progress instead (bounded to a few hundred steps, see
        // `warmup_progress_sufficient`). Once a step fails that bar the iterate is
        // not converging to a certifiable in-chart root — flag for the exact
        // fallback (no arbitrary step budget; the bound falls out of the
        // contraction requirement).
        //
        // F6: only enforce the progress bar while STILL uncertified. If this step
        // just crossed `h ≤ ½` (e.g. 0.501 → 0.499, a decrease too small to clear
        // the multiplicative floor) the point is already certified — the loop
        // guard `while !cert.certified()` will exit and refine it. Running the
        // progress test unconditionally would falsely reject an in-hand certificate
        // and push the row to the exact fallback for nothing. See
        // [`warmup_should_reject`].
        if warmup_should_reject(cert.certified(), cert.h, prev_h) {
            return Ok(None);
        }
    }
    refine_certified_start(
        atom,
        evaluator,
        t,
        x,
        amplitude,
        lipschitz,
        newton_steps,
        cert,
        delta,
        chart_center,
        chart_radius,
        objective,
    )
}

fn kantorovich_root_radius(cert: RowCertificate) -> f64 {
    if !cert.certified() || !(cert.eta.is_finite() && cert.eta >= 0.0) {
        return f64::INFINITY;
    }
    if cert.eta == 0.0 {
        return 0.0;
    }
    if !(cert.h.is_finite() && cert.h >= 0.0) {
        return f64::INFINITY;
    }
    let h = cert.h.min(KANTOROVICH_THRESHOLD);
    let discriminant = (1.0 - 2.0 * h).max(0.0).sqrt();
    let radius = 2.0 * cert.eta / (1.0 + discriminant);
    if radius.is_finite() {
        radius
    } else {
        f64::INFINITY
    }
}

fn distilled_probe_tolerance(
    amortized: &CertifiedEncodeProbe,
    cold: &CertifiedEncodeProbe,
    amplitude: f64,
    x: ArrayView1<'_, f64>,
) -> f64 {
    let certified_radius =
        kantorovich_root_radius(amortized.final_cert) + kantorovich_root_radius(cold.final_cert);
    let coord_scale = amortized.coord.dot(&amortized.coord).sqrt()
        + cold.coord.dot(&cold.coord).sqrt()
        + x.dot(&x).sqrt()
        + amplitude.abs()
        + 1.0;
    certified_radius + 1024.0 * f64::EPSILON * coord_scale
}

fn latent_coordinate_distance(
    atom: &SaeManifoldAtom,
    lhs: ArrayView1<'_, f64>,
    rhs: ArrayView1<'_, f64>,
) -> f64 {
    let mut acc = 0.0;
    for axis in 0..lhs.len().min(rhs.len()) {
        let mut diff = (lhs[axis] - rhs[axis]).abs();
        if let Some(period) = latent_axis_period(atom, axis) {
            let wrapped = diff.rem_euclid(period);
            diff = wrapped.min(period - wrapped);
        }
        acc += diff * diff;
    }
    acc.sqrt()
}

fn latent_axis_period(atom: &SaeManifoldAtom, axis: usize) -> Option<f64> {
    use crate::manifold::SaeAtomBasisKind::*;
    match &atom.basis_kind {
        Periodic | Torus => Some(1.0),
        Cylinder if axis == 0 => Some(1.0),
        Sphere if axis == 1 => Some(std::f64::consts::TAU),
        _ => None,
    }
}

/// Configuration for [`EncodeAtlas`] construction and online encode. All fields
/// are explicit; the atlas never reads global state and adds no CLI flags.
#[derive(Debug, Clone, Copy)]
pub struct AtlasConfig {
    /// Grid resolution per latent axis for offline chart centers (the
    /// SHAPE_BAND grid idiom).
    pub grid_resolution: usize,
    /// Levenberg ridge floor added to the per-row Gauss-Newton Hessian.
    pub ridge: f64,
    /// Number of online Newton refinement steps after a certified start (1 or 2
    /// per issue #1010).
    pub newton_steps: usize,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 16,
            ridge: 1.0e-9,
            newton_steps: 2,
        }
    }
}

/// The encode atlas: per-atom certified charts plus the online certified-encode
/// driver (issue #1010).
#[derive(Debug, Clone)]
pub struct EncodeAtlas {
    pub atoms: Vec<AtomEncodeAtlas>,
    pub config: AtlasConfig,
}

impl EncodeAtlas {
    /// Build the offline atlas over a frozen dictionary: for each atom, lay down
    /// chart centers on the atom's coordinate grid and certify a Newton radius
    /// from the Kantorovich inequality at the worst-case in-chart start.
    ///
    /// `amplitude_bound[k]` is the per-atom bound on `|z_k|` used to scale the
    /// reconstruction jets (the offline `L` must hold for the largest amplitude
    /// the encode can produce); `target_norm_bound` bounds `‖x‖` over the data.
    pub fn build(
        atoms: &[SaeManifoldAtom],
        amplitude_bound: &[f64],
        target_norm_bound: f64,
        config: AtlasConfig,
    ) -> Result<Self, String> {
        if amplitude_bound.len() != atoms.len() {
            return Err(format!(
                "EncodeAtlas::build: amplitude_bound length {} != atom count {}",
                amplitude_bound.len(),
                atoms.len()
            ));
        }
        let mut atom_atlases = Vec::with_capacity(atoms.len());
        for (k, atom) in atoms.iter().enumerate() {
            let atlas =
                Self::build_atom_atlas(k, atom, amplitude_bound[k], target_norm_bound, &config)?;
            atom_atlases.push(atlas);
        }
        Ok(Self {
            atoms: atom_atlases,
            config,
        })
    }

    pub(crate) fn build_atom_atlas(
        atom_index: usize,
        atom: &SaeManifoldAtom,
        amplitude_bound: f64,
        target_norm_bound: f64,
        config: &AtlasConfig,
    ) -> Result<AtomEncodeAtlas, String> {
        let centers = chart_center_grid(atom, config.grid_resolution);
        // Half the inter-center spacing is the natural in-chart radius so the
        // charts tile the grid without gaps; refined below if the certificate
        // fails at that radius. One uniform radius for the regular grid.
        let nominal_radius = chart_nominal_radius(atom, config.grid_resolution);
        let radii = vec![nominal_radius; centers.nrows()];
        Self::build_atom_atlas_from_centers(
            atom_index,
            atom,
            centers.view(),
            &radii,
            amplitude_bound,
            target_norm_bound,
            config,
        )
    }

    /// Build a per-atom atlas from EXPLICIT chart centers with a per-center
    /// nominal radius — the geometry-agnostic core shared by the regular-grid
    /// [`Self::build_atom_atlas`] and the data-driven [`Self::build_data_driven`].
    /// Every chart is certified identically (Kantorovich radius from the in-chart
    /// curvature at its center); only the center PLACEMENT and per-center radius
    /// differ. `radii[c]` is the nominal in-chart radius for `centers[c]`.
    pub(crate) fn build_atom_atlas_from_centers(
        atom_index: usize,
        atom: &SaeManifoldAtom,
        centers: ArrayView2<'_, f64>,
        radii: &[f64],
        amplitude_bound: f64,
        target_norm_bound: f64,
        config: &AtlasConfig,
    ) -> Result<AtomEncodeAtlas, String> {
        let d = atom.latent_dim;
        if centers.ncols() != d {
            return Err(format!(
                "build_atom_atlas_from_centers: centers have {} cols but atom latent_dim is {d}",
                centers.ncols()
            ));
        }
        if radii.len() != centers.nrows() {
            return Err(format!(
                "build_atom_atlas_from_centers: {} radii != {} centers",
                radii.len(),
                centers.nrows()
            ));
        }
        // Full-width frame (matches `family_jet_sups` / `reconstruction_jet_sups`):
        // the atlas's stored decoder scaling must pair with the full-width family
        // sups, so a #1117 rank-reduced atom contributes `Σ‖(Q B̃)_{m,:}‖`, not the
        // reduced-row sum. Identical for an un-reduced atom.
        let decoder_norm_sum = decoder_row_norm_sum(atom.full_width_decoder().view());
        let mut charts = Vec::with_capacity(centers.nrows());
        // HONEST REFUSAL for Duchon atoms (F2/F3): the closed-form Hessian-Lipschitz
        // bound available here (`family_jet_sups` Duchon arm) hard-codes cubic-r³
        // jets and a single origin center, but the real Duchon kernel is the
        // polyharmonic `c·r^(2m−d)` (with log variants) over data-placed centers.
        // That bound can UNDER-estimate L → a FALSE certificate (the module's own
        // warning; underestimating L is the dangerous direction). The atom does not
        // expose its real order/centers/scaling to this crate, so no sound bound is
        // available — refuse rather than fabricate. Every Duchon chart is emitted
        // UNCERTIFIED (`certified_radius = 0`, no amortized predictor), so routing
        // skips it and every Duchon row flags for the exact multi-start encode.
        let duchon_uncertifiable =
            matches!(atom.basis_kind, crate::manifold::SaeAtomBasisKind::Duchon);
        for c in 0..centers.nrows() {
            let center = centers.row(c).to_owned();
            let nominal_radius = radii[c];
            let region = chart_region(atom, center.clone(), nominal_radius);
            if duchon_uncertifiable {
                charts.push(CertifiedChart {
                    region,
                    lipschitz: f64::INFINITY,
                    beta_center: f64::INFINITY,
                    certified_radius: 0.0,
                    amortized_jacobian: None,
                    recon_center: Array1::<f64>::zeros(atom.output_dim()),
                    amortized_base: None,
                });
                continue;
            }
            let sups = family_jet_sups(atom, &region)?;
            let recon_sups = reconstruction_jet_sups(atom, sups);
            let lipschitz =
                hessian_lipschitz_constant(recon_sups, amplitude_bound, target_norm_bound, 0.0);
            // β at the chart center bounds the worst-case in-chart curvature
            // (the Gauss-Newton Hessian is continuous; the certified radius is
            // solved so the certificate is robust to the start within the ball).
            let beta_center = match center_beta(atom, &center, config.ridge) {
                Some(b) => b,
                None => {
                    // Degenerate center curvature: no certifiable chart here, and
                    // no amortized Jacobian (the same singular Gauss–Newton block).
                    charts.push(CertifiedChart {
                        region,
                        lipschitz,
                        beta_center: f64::INFINITY,
                        certified_radius: 0.0,
                        amortized_jacobian: None,
                        recon_center: Array1::<f64>::zeros(atom.output_dim()),
                        amortized_base: None,
                    });
                    continue;
                }
            };
            // Distill the amortized-encoder Jacobian at this center (#1026 ladder
            // item 3): the IFT derivative of the encode map, precomputed offline
            // so the online encode is one mat-vec. A finite `beta_center` (above)
            // means the Gauss–Newton block is non-singular, so this succeeds
            // alongside it; the pair travels together on the chart.
            let (amortized_jacobian, recon_center) =
                match center_amortized_jacobian(atom, &center, config.ridge) {
                    Some((a1, m1)) => (Some(a1), m1),
                    None => (None, Array1::<f64>::zeros(atom.output_dim())),
                };
            // Certified radius from h = β·η·L ≤ ½ with η ≤ R (Newton step length
            // is bounded by the start distance to the root, itself ≤ chart
            // radius at worst): R_c = ½ / (β·L), capped at the nominal radius.
            let certified_radius = if lipschitz > 0.0 && beta_center.is_finite() {
                (0.5 / (beta_center * lipschitz)).min(region.radius)
            } else {
                region.radius
            };
            // Precompute the affine-predictor constant `base = t_c − A₁·m₁` (atom-
            // static), so the online encode is a single `base + (1/z)·A₁·x` mat-vec.
            let amortized_base = amortized_jacobian
                .as_ref()
                .map(|a1| &center - &a1.dot(&recon_center));
            charts.push(CertifiedChart {
                region,
                lipschitz,
                beta_center,
                certified_radius,
                amortized_jacobian,
                recon_center,
                amortized_base,
            });
        }
        Ok(AtomEncodeAtlas {
            atom_index,
            latent_dim: d,
            decoder_norm_sum,
            charts,
        })
    }

    /// Build the atlas with DATA-DRIVEN chart placement: instead of a dense
    /// `resolution^d` product grid (exponential in latent dim `d`, so the regular
    /// [`Self::build`] is forced to coarse, poorly-certified charts for `d ≥ 3`),
    /// place a bounded number of charts AT the data's own latent coordinates. The
    /// chart count is then `O(max_charts)` regardless of `d`, and every chart sits
    /// where data actually lands (small in-chart residual → certifies), so
    /// higher-dimensional atoms — which reconstruct real activations far better per
    /// parameter — become affordable and well-covered.
    ///
    /// `coords[k]` is atom `k`'s `n × d_k` latent coordinates (the seed coords, or
    /// a previous encode's output). Charts are chosen by greedy farthest-point
    /// sampling over those coords (deterministic, coverage-maximizing), capped at
    /// `max_charts`. Each chart's nominal radius is half the distance to its
    /// nearest neighbor center, so the charts tile the local data density. The
    /// per-chart Kantorovich certification is IDENTICAL to the regular grid — only
    /// the center placement differs.
    pub fn build_data_driven(
        atoms: &[SaeManifoldAtom],
        coords: &[Array2<f64>],
        amplitude_bound: &[f64],
        target_norm_bound: f64,
        max_charts: usize,
        config: AtlasConfig,
    ) -> Result<Self, String> {
        if amplitude_bound.len() != atoms.len() || coords.len() != atoms.len() {
            return Err(format!(
                "build_data_driven: amplitude_bound {} / coords {} must match atom count {}",
                amplitude_bound.len(),
                coords.len(),
                atoms.len()
            ));
        }
        let mut atom_atlases = Vec::with_capacity(atoms.len());
        for (k, atom) in atoms.iter().enumerate() {
            let (centers, radii) =
                data_driven_chart_centers(atom, coords[k].view(), max_charts.max(1))?;
            let atlas = Self::build_atom_atlas_from_centers(
                k,
                atom,
                centers.view(),
                &radii,
                amplitude_bound[k],
                target_norm_bound,
                &config,
            )?;
            atom_atlases.push(atlas);
        }
        Ok(Self {
            atoms: atom_atlases,
            config,
        })
    }

    fn refine_certified_encode_start(
        &self,
        atom: &SaeManifoldAtom,
        evaluator: &dyn SaeBasisEvaluator,
        chart: &CertifiedChart,
        t: Array1<f64>,
        x: ArrayView1<'_, f64>,
        amplitude: f64,
        objective: &EncodeObjective<'_>,
    ) -> Result<(Array1<f64>, RowCertificate), String> {
        // Certify from the warm start, navigating into the Kantorovich basin first
        // if the unit-amplitude start has h > ½ (see `certify_with_basin_warmup`).
        // The Lipschitz is the objective's EFFECTIVE bound (F3): the stored
        // Euclidean data-term `L` scaled by the metric operator-norm bound plus the
        // prior's third-derivative bound. Reduces to `chart.lipschitz` exactly for
        // the Euclidean objective, so the metric-free path is unchanged.
        let lipschitz = objective.effective_lipschitz(atom, chart.lipschitz);
        let Some(probe) = certify_with_basin_warmup(
            atom,
            evaluator,
            t,
            x,
            amplitude,
            lipschitz,
            self.config.newton_steps,
            chart.region.center.view(),
            chart.region.radius,
            objective,
        )?
        else {
            return Ok((
                Array1::<f64>::zeros(atom.latent_dim),
                uncertified_certificate(chart.lipschitz),
            ));
        };
        // F5: pair the REFINED coordinate with the certificate evaluated AT that
        // refined landing point (`final_cert`), not the pre-refinement
        // `initial_cert`. Returning `initial_cert` describes a different (earlier)
        // iterate's β/η/h — its Kantorovich root-radius overstates the refined
        // point's distance to the root.
        Ok((probe.coord, probe.final_cert))
    }

    /// Online certified encode of one target row `x` against one atom `k` with
    /// fixed amplitude `z`. Routes to the nearest chart, starts from that chart's
    /// distilled IFT warm start, runs `config.newton_steps` Newton steps, and
    /// returns the encoded coordinate with its certificate. An uncertified start
    /// (no chart, no distilled Jacobian, non-positive amplitude, or `h > ½`)
    /// flags the row for the exact multi-start caller.
    pub fn certified_encode_row(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        x: ArrayView1<'_, f64>,
        amplitude: f64,
    ) -> Result<(Array1<f64>, RowCertificate), String> {
        // The bare Euclidean, prior-free objective — bit-identical to the metric-
        // free certified encode.
        self.certified_encode_row_with_objective(
            atom,
            atom_index,
            x,
            amplitude,
            &EncodeObjective::euclidean(),
        )
    }

    /// [`Self::certified_encode_row`] against the TRUE encode objective (F3): the
    /// Newton–Kantorovich certificate is computed under the fit's per-row output
    /// metric and latent coordinate prior ([`EncodeObjective`]), so the certified
    /// root is the minimizer of the SAME generalized-least-squares-plus-prior
    /// functional the fit used — not a bare Euclidean stand-in that certifies a
    /// different problem. The metric operator-norm bound scales the offline chart
    /// Lipschitz; the per-row metric and prior enter `β, η` and the candidate-
    /// ranking SSE guard online. `EncodeObjective::euclidean()` reproduces the
    /// metric-free path exactly.
    pub fn certified_encode_row_with_objective(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        x: ArrayView1<'_, f64>,
        amplitude: f64,
        objective: &EncodeObjective<'_>,
    ) -> Result<(Array1<f64>, RowCertificate), String> {
        let atom_atlas = self
            .atoms
            .get(atom_index)
            .ok_or_else(|| format!("certified_encode_row: atom {atom_index} not in atlas"))?;
        let d = atom.latent_dim;
        // A per-row metric factor `U` must be `p × rank` (`M = U Uᵀ` acts on the
        // p-dim output). A shape mismatch is a caller bug — surface it rather than
        // silently certifying a wrong (or panicking) whitening.
        if let Some(u) = objective.metric_factor {
            if u.nrows() != atom.output_dim() {
                return Err(format!(
                    "certified_encode_row_with_objective: metric factor has {} rows but atom output_dim is {}",
                    u.nrows(),
                    atom.output_dim()
                ));
            }
        }
        // A missing basis evaluator means the amortized/cold predictor cannot fire
        // for this atom (e.g. a frozen-baseline or first-build atom that never
        // attached a distilled evaluator). That is exactly the "cannot certify"
        // state — flag the row uncertified (zeros coords, ∞ certificate) so the
        // upstream exact multi-start solve owns it, never a hard error that aborts
        // the whole criterion. Mirrors the no-chart / singular-Jacobian branches.
        let Some(evaluator) = atom.basis_evaluator.as_ref().cloned() else {
            return Ok((
                Array1::<f64>::zeros(d),
                RowCertificate {
                    beta: f64::INFINITY,
                    eta: f64::INFINITY,
                    lipschitz: f64::INFINITY,
                    h: f64::INFINITY,
                },
            ));
        };

        // Route to the nearest chart centers by AMBIENT reconstruction distance.
        // A single nearest chart is NOT globally sound on self-approaching atoms:
        // where the decoded manifold folds near itself (two distant latent points
        // map near the same output), the nearest-center chart can certify into the
        // locally-worse basin while another chart holds the GLOBAL minimum (both
        // branches' charts reconstruct near the crossing, so both are near in
        // ambient distance). The certificate is honest about LOCAL convergence but
        // cannot see the better far basin. So we refine in the top-K nearest charts
        // and keep the lowest-reconstruction-error CERTIFIED result. For a unimodal
        // atom every candidate chart converges to the same root, so this is a no-op
        // (first-wins tie → the nearest chart), preserving the existing behavior.
        let candidates = nearest_charts_topk(atom_atlas, x, amplitude, CERTIFIED_ROUTING_TOPK);
        if candidates.is_empty() {
            return Ok((
                Array1::<f64>::zeros(d),
                RowCertificate {
                    beta: f64::INFINITY,
                    eta: f64::INFINITY,
                    lipschitz: f64::INFINITY,
                    h: f64::INFINITY,
                },
            ));
        }
        // Best CERTIFIED result by reconstruction error, plus the nearest chart's
        // result as the uncertified fallback (preserving the prior return when no
        // candidate certifies — the nearest chart owns the flagged row).
        let mut best: Option<(Array1<f64>, RowCertificate, f64)> = None;
        let mut nearest_fallback: Option<(Array1<f64>, RowCertificate)> = None;
        for chart_idx in candidates {
            let chart = &atom_atlas.charts[chart_idx];
            let Some(t) = amortized_warm_start(chart, x, amplitude) else {
                if nearest_fallback.is_none() {
                    nearest_fallback = Some((
                        Array1::<f64>::zeros(d),
                        uncertified_certificate(chart.lipschitz),
                    ));
                }
                continue;
            };
            let (coord, cert) = self.refine_certified_encode_start(
                atom,
                evaluator.as_ref(),
                chart,
                t,
                x,
                amplitude,
                objective,
            )?;
            if nearest_fallback.is_none() {
                nearest_fallback = Some((coord.clone(), cert.clone()));
            }
            if cert.certified() {
                let err = encode_reconstruction_error_core(
                    atom,
                    evaluator.as_ref(),
                    coord.view(),
                    x,
                    amplitude,
                    objective,
                );
                if best.as_ref().map(|(_, _, e)| err < *e).unwrap_or(true) {
                    best = Some((coord, cert, err));
                }
                // Global-minimum short-circuit: reconstruction error ≥ 0, so a
                // certified candidate already at the ambient noise floor is provably
                // the global optimum over the remaining charts — stop refining them.
                if let Some((_, _, e)) = best.as_ref() {
                    if *e <= CERTIFIED_GLOBAL_MIN_RECON_FLOOR * (1.0 + x.dot(&x).sqrt()) {
                        break;
                    }
                }
            }
        }
        match best {
            Some((coord, cert, _)) => Ok((coord, cert)),
            None => Ok(nearest_fallback.unwrap_or_else(|| {
                (
                    Array1::<f64>::zeros(d),
                    RowCertificate {
                        beta: f64::INFINITY,
                        eta: f64::INFINITY,
                        lipschitz: f64::INFINITY,
                        h: f64::INFINITY,
                    },
                )
            })),
        }
    }

    /// Amortized (distilled) encode of one target row `x` against one atom `k`
    /// with fixed amplitude `z` (#1026 ladder item 3).
    ///
    /// Routes to the nearest chart, then predicts the latent coordinate in CLOSED
    /// FORM from that chart's precomputed implicit-function-theorem Jacobian:
    ///
    /// ```text
    /// t̂ = t_c + (1/z) · A₁ · (x − z · m₁(t_c)),
    /// ```
    ///
    /// a single `O(d·p)` mat-vec — no per-row Hessian factorization or
    /// eigendecomposition, which is the amortization. The Kantorovich
    /// certificate is then evaluated AT the predicted start `t̂` with the chart's
    /// closed-form Lipschitz constant. A prediction is accepted only when that
    /// certificate holds, an independent cold chart-center probe also certifies,
    /// and the two refined coordinates agree within the two probes' final
    /// Kantorovich root-radius bounds. This keeps the distilled path honest
    /// without letting the exact probe reuse the distilled warm start it is
    /// auditing. A chart without a distilled Jacobian (singular Gauss–Newton
    /// block) flags the row.
    pub fn amortized_encode_row(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        x: ArrayView1<'_, f64>,
        amplitude: f64,
    ) -> Result<(Array1<f64>, RowCertificate), String> {
        // Euclidean, prior-free objective — bit-identical to the metric-free path.
        self.amortized_encode_row_with_objective(
            atom,
            atom_index,
            x,
            amplitude,
            &EncodeObjective::euclidean(),
        )
    }

    /// [`Self::amortized_encode_row`] against the TRUE encode objective (F3): the
    /// distilled predictor's warm start is Euclidean (its `A₁` is the Euclidean
    /// Gauss–Newton block), but BOTH Kantorovich probes certify under the supplied
    /// metric + prior objective, with the chart Lipschitz taken as the objective's
    /// effective bound. So the fast path is preserved (one mat-vec warm start) while
    /// the certificate — and therefore the trust/fallback decision — is honest about
    /// the metric-and-prior objective the fit optimized. `EncodeObjective::euclidean`
    /// reproduces the metric-free distilled path exactly.
    pub fn amortized_encode_row_with_objective(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        x: ArrayView1<'_, f64>,
        amplitude: f64,
        objective: &EncodeObjective<'_>,
    ) -> Result<(Array1<f64>, RowCertificate), String> {
        let atom_atlas = self
            .atoms
            .get(atom_index)
            .ok_or_else(|| format!("amortized_encode_row: atom {atom_index} not in atlas"))?;
        let d = atom.latent_dim;
        let uncertified = || {
            (
                Array1::<f64>::zeros(d),
                RowCertificate {
                    beta: f64::INFINITY,
                    eta: f64::INFINITY,
                    lipschitz: f64::INFINITY,
                    h: f64::INFINITY,
                },
            )
        };
        // A missing basis evaluator means the distilled predictor cannot fire for
        // this atom — flag the row uncertified (the exact upstream solve owns it)
        // rather than erroring, exactly as the no-chart / singular-Jacobian /
        // non-positive-amplitude branches below do. Never a silent wrong encode,
        // never a hard abort of the criterion.
        let Some(evaluator) = atom.basis_evaluator.as_ref().cloned() else {
            return Ok(uncertified());
        };
        let Some((chart_idx, _)) = nearest_chart(atom_atlas, x, amplitude) else {
            return Ok(uncertified());
        };
        let chart = &atom_atlas.charts[chart_idx];
        // Closed-form predicted start t̂ = t_c + (1/z)·A₁·(x − z·m₁). `None` when
        // the chart's Gauss–Newton block was singular (no distilled Jacobian, so
        // the amortized predictor cannot fire) or the amplitude is not strictly
        // positive and finite (a near-inactive atom, where the amplitude-divided
        // map is undefined) — either way flag for the exact fallback, never a
        // silent wrong encode.
        let Some(t_hat) = amortized_warm_start(chart, x, amplitude) else {
            return Ok(uncertified());
        };
        // Effective Kantorovich Lipschitz for the TRUE objective (F3): the stored
        // Euclidean data-term bound scaled by the metric operator-norm bound plus
        // the prior's third-derivative bound. Reduces to `chart.lipschitz` exactly
        // for `EncodeObjective::euclidean`, so the metric-free distilled path is
        // unchanged.
        let lipschitz = objective.effective_lipschitz(atom, chart.lipschitz);
        // Evaluate the SAME Kantorovich certificate at the predicted start. The
        // amortized prediction is trusted only if this certificate holds AND an
        // independent cold chart-center probe certifies and agrees below the
        // two probes' final Kantorovich root-radius bounds. This avoids the
        // self-referential gate where the "exact" probe is warm-started by the
        // same distilled prediction it is supposed to audit.
        let Some(amortized_probe) = certify_with_basin_warmup(
            atom,
            evaluator.as_ref(),
            t_hat,
            x,
            amplitude,
            lipschitz,
            self.config.newton_steps,
            chart.region.center.view(),
            chart.region.radius,
            objective,
        )?
        else {
            return Ok((Array1::<f64>::zeros(d), uncertified_certificate(lipschitz)));
        };

        let cold_start = chart.region.center.clone();
        let Some(cold_probe) = certify_with_basin_warmup(
            atom,
            evaluator.as_ref(),
            cold_start,
            x,
            amplitude,
            lipschitz,
            self.config.newton_steps,
            chart.region.center.view(),
            chart.region.radius,
            objective,
        )?
        else {
            return Ok((amortized_probe.coord, uncertified_certificate(lipschitz)));
        };

        let gap =
            latent_coordinate_distance(atom, amortized_probe.coord.view(), cold_probe.coord.view());
        let tolerance = distilled_probe_tolerance(&amortized_probe, &cold_probe, amplitude, x);
        if !(gap.is_finite() && gap <= tolerance) {
            return Ok((amortized_probe.coord, uncertified_certificate(lipschitz)));
        }
        // F5: return the certificate at the refined landing coordinate
        // (`final_cert`), consistent with the coord actually returned and with the
        // `distilled_probe_tolerance` gate above (which already reads `final_cert`
        // via `kantorovich_root_radius`). `initial_cert` is a stale earlier iterate.
        Ok((amortized_probe.coord, amortized_probe.final_cert))
    }

    /// Batched amortized (distilled) encode over many rows against one atom
    /// (#1026 ladder item 3, corpus-rate). Each row uses the closed-form
    /// per-chart Jacobian predictor and carries its own Kantorovich certificate;
    /// uncertified rows are flagged in [`EncodeResult::encode_uncertified_count`]
    /// for the exact multi-start fallback. Row-independent against the frozen
    /// dictionary, so the batch fans out over rows (deterministic row-order
    /// assembly, bit-identical run-to-run), staying sequential inside a rayon
    /// worker to avoid nested oversubscription.
    pub fn amortized_encode_batch(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        targets: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
    ) -> Result<EncodeResult, String> {
        let n = targets.nrows();
        if amplitudes.len() != n {
            return Err(format!(
                "amortized_encode_batch: amplitudes len {} != rows {n}",
                amplitudes.len()
            ));
        }
        let d = atom.latent_dim;
        let encode_rows =
            |range: std::ops::Range<usize>| -> Result<Vec<(Array1<f64>, bool)>, String> {
                range
                    .map(|row| {
                        let (t, cert) = self.amortized_encode_row(
                            atom,
                            atom_index,
                            targets.row(row),
                            amplitudes[row],
                        )?;
                        Ok((t, cert.certified()))
                    })
                    .collect()
            };
        let rows: Vec<(Array1<f64>, bool)> =
            if n >= ENCODE_BATCH_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
                use rayon::prelude::*;
                const CHUNK: usize = 256;
                let n_chunks = n.div_ceil(CHUNK);
                let chunked: Vec<Vec<(Array1<f64>, bool)>> = (0..n_chunks)
                    .into_par_iter()
                    .map(|c| {
                        let start = c * CHUNK;
                        let end = (start + CHUNK).min(n);
                        encode_rows(start..end)
                    })
                    .collect::<Result<_, _>>()?;
                chunked.into_iter().flatten().collect()
            } else {
                encode_rows(0..n)?
            };
        let mut coords = Array2::<f64>::zeros((n, d));
        let mut certified = Vec::with_capacity(n);
        for (row, (t, cert)) in rows.into_iter().enumerate() {
            coords.row_mut(row).assign(&t);
            certified.push(cert);
        }
        Ok(EncodeResult::from_rows(coords, certified))
    }

    /// Encode one atom's rows through the full three-tier fallback cascade and
    /// report the cost breakdown ([`FallbackTelemetry`], reviewer condition #3).
    ///
    /// Each row is tried cheapest-first: the amortized one-mat-vec predictor,
    /// then (if uncertified) the certified IFT-warm-start Newton encode, then (if
    /// still uncertified) it is counted for the exact multi-start solve. The
    /// returned coords carry the best CERTIFIED encode reached; a multi-start row
    /// keeps the Newton iterate as its (uncertified) coordinate so the caller can
    /// still decode it, exactly as [`super::SaeManifoldTerm::amortized_encode_target`]
    /// does — the honesty flag rides `certified`.
    ///
    /// This is the instrumented analogue of [`Self::amortized_encode_batch`] +
    /// the per-row Newton rescue: it does the SAME work, and additionally counts
    /// which tier certified each encode so the multi-start-fallback fraction (the
    /// encode-tax multiplier) is measurable.
    pub fn encode_atom_with_fallback_telemetry(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        targets: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
    ) -> Result<(EncodeResult, FallbackTelemetry), String> {
        let n = targets.nrows();
        if amplitudes.len() != n {
            return Err(format!(
                "encode_atom_with_fallback_telemetry: amplitudes len {} != rows {n}",
                amplitudes.len()
            ));
        }
        let amortized = self.amortized_encode_batch(atom, atom_index, targets, amplitudes)?;
        let mut coords = amortized.coords;
        let mut certified = amortized.certified;
        let mut telemetry = FallbackTelemetry {
            n_rows: n,
            n_atoms: 1,
            ..FallbackTelemetry::default()
        };
        for row in 0..n {
            if certified[row] {
                telemetry.amortized_certified += 1;
                continue;
            }
            // The amortized predictor missed: try the certified Newton warm-start.
            let (t, cert) =
                self.certified_encode_row(atom, atom_index, targets.row(row), amplitudes[row])?;
            // Keep the Newton iterate regardless (it is a better start than the
            // amortized one even when uncertified, and a multi-start row still
            // needs a decodable coordinate).
            coords.row_mut(row).assign(&t);
            if cert.certified() {
                certified[row] = true;
                telemetry.newton_rescued += 1;
            } else {
                telemetry.multistart_fallback += 1;
            }
        }
        Ok((EncodeResult::from_rows(coords, certified), telemetry))
    }

    /// Batched certified encode over many rows against one atom (the #988
    /// throughput consumer). Each row carries its own certificate; uncertified
    /// rows are flagged in [`EncodeResult::encode_uncertified_count`] for the
    /// exact multi-start fallback.
    pub fn certified_encode_batch(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        targets: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
    ) -> Result<EncodeResult, String> {
        let n = targets.nrows();
        if amplitudes.len() != n {
            return Err(format!(
                "certified_encode_batch: amplitudes len {} != rows {n}",
                amplitudes.len()
            ));
        }
        let d = atom.latent_dim;
        // Per-row encode is independent against a frozen dictionary (#1010), so
        // the corpus-rate batch fans out over rows (#1026 amortized-encoder leg /
        // #977 Stage-3 corpus encode). Each row produces an owned `(t, certified)`
        // pair; results are assembled back in row order so the output is
        // bit-identical run-to-run regardless of thread scheduling. Stay
        // sequential inside a rayon worker (e.g. when an outer atom-level fan-out
        // owns the pool) to avoid nested oversubscription. The first row that
        // fails to encode propagates its error deterministically.
        let encode_rows =
            |range: std::ops::Range<usize>| -> Result<Vec<(Array1<f64>, bool)>, String> {
                range
                    .map(|row| {
                        let (t, cert) = self.certified_encode_row(
                            atom,
                            atom_index,
                            targets.row(row),
                            amplitudes[row],
                        )?;
                        Ok((t, cert.certified()))
                    })
                    .collect()
            };
        let rows: Vec<(Array1<f64>, bool)> =
            if n >= ENCODE_BATCH_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
                use rayon::prelude::*;
                const CHUNK: usize = 256;
                let n_chunks = n.div_ceil(CHUNK);
                let chunked: Vec<Vec<(Array1<f64>, bool)>> = (0..n_chunks)
                    .into_par_iter()
                    .map(|c| {
                        let start = c * CHUNK;
                        let end = (start + CHUNK).min(n);
                        encode_rows(start..end)
                    })
                    .collect::<Result<_, _>>()?;
                chunked.into_iter().flatten().collect()
            } else {
                encode_rows(0..n)?
            };
        let mut coords = Array2::<f64>::zeros((n, d));
        let mut certified = Vec::with_capacity(n);
        for (row, (t, cert)) in rows.into_iter().enumerate() {
            coords.row_mut(row).assign(&t);
            certified.push(cert);
        }
        Ok(EncodeResult::from_rows(coords, certified))
    }

    /// Batched GEMM "fast" amortized encode — the traditional-encoder forward
    /// pass, WITH manifolds. For every row this applies the SAME closed-form
    /// affine predictor as [`amortized_warm_start`]
    /// (`t̂ = t_c + (1/z)·A₁·(x − z·m₁)`), but routed and applied as batched
    /// matrix products instead of a per-row loop wrapped in the Kantorovich
    /// certificate + basin warmup. NO per-row certificate is taken: this is the
    /// speed mode (the certified `*_encode_*` paths remain the accuracy mode).
    ///
    /// Cost is GEMM-bound: one `(n × p)·(p × d)` decode-distance product for
    /// nearest-chart routing (skipped for single-chart atoms) plus, per chart,
    /// one `(n_c × p)·(p × d)` predictor product — i.e. `≈ X·Wᵀ`, exactly a
    /// dense SAE encoder's forward map.
    ///
    /// Degenerate rows are handled exactly as `amortized_warm_start` flags them
    /// (returns `None` ⇒ zeroed coord here): a missing basis evaluator, a chart
    /// whose Gauss–Newton block was singular (`amortized_jacobian == None`), or a
    /// non-finite / non-positive amplitude. Those rows are zeroed (never a panic,
    /// never a silent wrong encode), and their indices are returned in the
    /// `valid` mask so the caller can route them to the exact path if desired.
    ///
    /// Returns `(coords, valid)` where `coords` is `n × d` and `valid[row]` is
    /// `true` iff the amortized predictor fired for that row.
    pub fn amortized_encode_batch_fast(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        x: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
    ) -> Result<(Array2<f64>, Vec<bool>), String> {
        let n = x.nrows();
        let p = atom.output_dim();
        let d = atom.latent_dim;
        if x.ncols() != p {
            return Err(format!(
                "amortized_encode_batch_fast: x has {} cols but atom output dim is {p}",
                x.ncols()
            ));
        }
        if amplitudes.len() != n {
            return Err(format!(
                "amortized_encode_batch_fast: amplitudes len {} != rows {n}",
                amplitudes.len()
            ));
        }
        let atom_atlas = self.atoms.get(atom_index).ok_or_else(|| {
            format!("amortized_encode_batch_fast: atom {atom_index} not in atlas")
        })?;
        let mut coords = Array2::<f64>::zeros((n, d));
        let mut valid = vec![false; n];

        // A missing basis evaluator means this atom never had a well-formed atlas
        // built here — treat every row as uncertified (zeroed), exactly like the
        // per-row `amortized_encode_row` no-evaluator branch. (The predictor below
        // uses only cached atlas data, so no evaluator call is made online.)
        if atom.basis_evaluator.is_none() {
            return Ok((coords, valid));
        }

        // ── Routing recon-centers (cached, no online basis evaluation). ────────
        // Routing sends a row to the chart whose center reconstruction
        // `m(t_c) = BᵀΦ(t_c)` is closest in ‖·‖². Those center reconstructions are
        // OFFLINE-cached in `chart.recon_center` (bit-identical to re-evaluating the
        // basis at the fixed centers — same φ·decoder accumulation). Gather the cache
        // instead of calling `evaluator.evaluate` on every invocation: that per-call
        // chart-center evaluation was the dominant per-atom-group overhead at massive
        // K, where N rows scatter across many atoms into tiny groups so a fixed
        // per-call cost is amortized over only a handful of rows. This is what keeps
        // the fast index-routed encode near-flat as K grows.
        let valid_charts: Vec<usize> = (0..atom_atlas.charts.len())
            .filter(|&c| atom_atlas.charts[c].certified_radius > 0.0)
            .collect();
        if valid_charts.is_empty() {
            return Ok((coords, valid));
        }
        // recon_centers (C × p): the cached m(t_c) for each certifiable chart.
        let mut recon_centers = Array2::<f64>::zeros((valid_charts.len(), p));
        for (ci, &c) in valid_charts.iter().enumerate() {
            recon_centers
                .row_mut(ci)
                .assign(&atom_atlas.charts[c].recon_center);
        }
        // Per-chart routing key: route_idx[row] = argmin_c ‖x_row − z_row·r_c‖²
        // (F1: the TRUE objective at the row's amplitude `z_row`, where `r_c` is
        // the amplitude-1 center reconstruction). First chart wins on a tie (strict
        // `<`), matching `nearest_chart`. A non-finite amplitude routes to chart 0
        // (moot — the predictor below skips the row anyway).
        //
        // Score the DIRECT squared distance `Σ_j (z·r_cj − x_j)²`, accumulated in
        // the same element order as the per-row `nearest_chart`, NOT the algebraic
        // expansion `z²‖r_c‖² − 2z·(x·r_c)`. The two are equal in exact arithmetic,
        // but the expansion subtracts two O(‖x‖²) quantities and loses ~‖x‖²·ε of
        // precision to cancellation — enough to FLIP the argmin between two charts
        // whose reconstructions coincide to rounding (e.g. period-wrapped torus-seam
        // charts, whose latent centers differ by a full period yet reconstruct to
        // within 1e-14). On such a near-tie the expansion and `nearest_chart`
        // disagree, and the two seam charts predict coords a full period apart — so
        // the batched fast-encode would diverge from the per-row warm-start it must
        // reproduce bit-for-bit (the `fast_encode_matches_per_row_warm_start`
        // contract). Computing the same direct distance in the same order keeps this
        // path byte-identical to `nearest_chart`, so routing (and the tie-break)
        // agrees on every row. `recon_centers` is still the cached offline
        // `m₁(t_c)` — no basis re-evaluation, which was the fast path's real cost.
        let route_idx: Vec<usize> = if valid_charts.len() == 1 {
            vec![0usize; n]
        } else {
            (0..n)
                .map(|row| {
                    let z = amplitudes[row];
                    // F4 — ACTIVITY GATE BEFORE ROUTING: an inactive (z = 0) or
                    // non-finite-amplitude row is skipped by the predictor loop
                    // below (`amp.abs() > 0.0`), so its chart never matters. Routing
                    // it anyway is the dense `O(n·K·C·p)` waste this path incurs at
                    // massive K, where each atom is active on only a sparse handful of
                    // the N rows yet the routing scan still touches every (row, chart)
                    // pair. Gate the amplitude here so the `C·p` distance scan runs
                    // only for the atom's genuinely-active rows; the chart-0 sentinel
                    // is moot for the gated rows (they are dropped downstream).
                    if !(z.is_finite() && z.abs() > 0.0) {
                        return 0usize;
                    }
                    let x_row = x.row(row);
                    let mut best_c = 0usize;
                    let mut best_d = f64::INFINITY;
                    for c in 0..valid_charts.len() {
                        let mut dist = 0.0;
                        for (r, xv) in recon_centers.row(c).iter().zip(x_row.iter()) {
                            let diff = z * r - xv;
                            dist += diff * diff;
                        }
                        if dist < best_d {
                            best_d = dist;
                            best_c = c;
                        }
                    }
                    best_c
                })
                .collect()
        };

        // ── Per-chart batched affine predictor. ───────────────────────────────
        // For rows routed to chart `c` with finite jacobian `A₁` (d × p) and
        // center reconstruction `m₁` (= `chart.recon_center`), the predictor is
        //   t̂ = t_c − A₁·m₁ + (1/z)·(A₁·x).
        // `t_c − A₁·m₁` is a per-chart constant `base`; `A₁·x` is a d-vector of
        // per-row dot products. Instead of gathering routed rows into a fresh
        // `X_c` (n_c × p) buffer and running a GEMM into a second `U` (n_c × d)
        // buffer — two allocations plus a full copy of the routed rows, per chart —
        // fuse the gather straight into the multiply: stream each source row of `x`
        // once (it is contiguous) and dot it against `A₁`'s rows, writing the
        // predicted coord directly. Zero per-chart heap traffic; the inverse
        // amplitude is hoisted to one reciprocal per row.
        //
        // Precompute each valid chart's `(A₁, base)` once (charts with a singular
        // Gauss–Newton block carry no `A₁`, so their routed rows stay
        // zeroed/uncertified — same as `amortized_warm_start` returning `None`).
        struct ChartPredictor<'a> {
            a1: &'a Array2<f64>,
            base: &'a Array1<f64>,
        }
        let predictors: Vec<Option<ChartPredictor<'_>>> = valid_charts
            .iter()
            .map(|&c| {
                let chart = &atom_atlas.charts[c];
                // `base = t_c − A₁·m₁` is precomputed offline in the atlas; reuse it
                // (both are `Some` together — singular G-N block ⇒ both `None`).
                match (
                    chart.amortized_jacobian.as_ref(),
                    chart.amortized_base.as_ref(),
                ) {
                    (Some(a1), Some(base)) => Some(ChartPredictor { a1, base }),
                    _ => None,
                }
            })
            .collect();

        for row in 0..n {
            let Some(pred) = predictors[route_idx[row]].as_ref() else {
                continue;
            };
            let amp = amplitudes[row];
            if !(amp.is_finite() && amp.abs() > 0.0) {
                continue;
            }
            let inv_z = 1.0 / amp;
            let x_row = x.row(row);
            let mut coord_row = coords.row_mut(row);
            for axis in 0..d {
                // (A₁·x)[axis] = A₁ row `axis` (contiguous, length p) · x_row.
                coord_row[axis] = pred.base[axis] + pred.a1.row(axis).dot(&x_row) * inv_z;
            }
            valid[row] = true;
        }
        Ok((coords, valid))
    }

    /// Fast batched FULL forward pass against one atom: encode → decode, the
    /// manifold analogue of a traditional SAE's `x̂ = z·D` (decoder `D`, code `z`).
    ///
    /// A traditional SAE decodes with one GEMM. The manifold SAE's reconstruction
    /// is `m(t̂) = z·Φ(t̂)·B` (module header) — the SAME GEMM `Φ·B`, but the code
    /// `Φ(t̂)` is the curved chart basis evaluated at the encoded latent coordinate
    /// rather than a flat one-hot. So the fast forward is exactly:
    ///   1. [`amortized_encode_batch_fast`] → per-row latent coords `t̂` (one
    ///      routing GEMM + one affine GEMM per chart — a traditional `W·x+b`);
    ///   2. ONE batched basis evaluation `Φ(t̂)` (the manifold-curvature step a
    ///      flat SAE doesn't have — `n×m`);
    ///   3. ONE GEMM `recon = Φ(t̂)·B` (`(n×m)·(m×p)` — a traditional decoder
    ///      `z·D`), then the per-row amplitude scale `z`.
    ///
    /// Rows the encoder could not certify-predict (no evaluator / singular
    /// Gauss–Newton block / non-finite-or-zero amplitude) are returned as a ZERO
    /// reconstruction and flagged `false` in the valid-mask — never a silent wrong
    /// decode. The reconstruction of a valid row equals, bit-for-bit up to GEMM
    /// reassociation, `z·(Φ(t̂_row)·B)` with `t̂` from the per-row predictor.
    pub fn amortized_reconstruct_batch_fast(
        &self,
        atom: &SaeManifoldAtom,
        atom_index: usize,
        x: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
    ) -> Result<(Array2<f64>, Vec<bool>), String> {
        let n = x.nrows();
        let p = atom.output_dim();
        // Step 1: batched encode → latent coords (reuses the fast routing+affine).
        let (coords, valid) = self.amortized_encode_batch_fast(atom, atom_index, x, amplitudes)?;
        let mut recon = Array2::<f64>::zeros((n, p));
        // A missing evaluator means no row could encode — every row is zeroed and
        // already flagged `false` by the encode; nothing to decode.
        let Some(evaluator) = atom.basis_evaluator.as_ref().cloned() else {
            return Ok((recon, valid));
        };
        // Step 2: ONE batched basis evaluation Φ(t̂) over all rows (n × m). Invalid
        // rows carry coords = 0 (the chart-origin); we still evaluate them in the
        // batch for a single GEMM, then zero their reconstruction below — the basis
        // is finite at the origin so this cannot poison the valid rows' GEMM.
        let (phi, _jet) = evaluator
            .evaluate(coords.view())
            .map_err(|err| format!("amortized_reconstruct_batch_fast: basis eval: {err}"))?;
        // Step 3: ONE GEMM recon = Φ·B (n × p), then per-row amplitude scale z.
        // m(t̂) = z·Φ(t̂)·B, matching the module header and `fill_decoded_row`'s
        // `Φ·decoder` accumulation (the amplitude is applied once here).
        let decoded = phi.dot(&atom.decoder_coefficients); // (n × p), amplitude-1
        for row in 0..n {
            if !valid[row] {
                continue; // stays zeroed — uncertified, like warm_start `None`.
            }
            let z = amplitudes[row];
            for col in 0..p {
                recon[[row, col]] = z * decoded[[row, col]];
            }
        }
        Ok((recon, valid))
    }

    /// LSH-routed certified encode (issue #1010 step 2 + 3): for each target
    /// row, the existing [`SaeCandidateIndex`] (#985/#994) proposes the
    /// best-aligned atom by frame alignment to the row direction; the row is then
    /// encoded against THAT atom's certified chart atlas. This is the production
    /// routing path. Atom selection is EXACT (#1777): [`SaeCandidateIndex::route_exact`]
    /// returns the global argmax of the routing score (the universal-bound LSH fast
    /// path, else a full-scan fallback) — never a silently-missed ungathered atom —
    /// and the atlas does the in-atom nearest-chart routing and the per-row
    /// Kantorovich certificate.
    ///
    /// `atoms[id]` must be aligned with the atlas's `atoms[id]` (same dictionary
    /// order the atlas was built from and the sketch/index were built over).
    /// A row over an empty dictionary, or whose globally-best atom aligns below the
    /// fit-quality floor, is flagged uncertified — it routes to the exact
    /// multi-start fallback, never a silent wrong encode.
    pub fn certified_encode_with_index<S: AtomFrameSketch + Sync>(
        &self,
        atoms: &[SaeManifoldAtom],
        index: &SaeCandidateIndex,
        sketch: &S,
        targets: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
        latent_dim: usize,
    ) -> Result<EncodeResult, String> {
        let n = targets.nrows();
        if amplitudes.len() != n {
            return Err(format!(
                "certified_encode_with_index: amplitudes len {} != rows {n}",
                amplitudes.len()
            ));
        }
        let budget = auto_candidate_budget(atoms.len().max(1));
        // LSH-routed per-row encode is independent across rows (sublinear atom
        // selection + frozen-dictionary in-atom Newton), so the corpus-rate batch
        // fans out over rows (#1026 amortized-encoder/routing leg / #977 Stage-3).
        // `None` coords (no LSH candidate) carry through as a zeroed row flagged
        // uncertified — identical to the sequential semantics. Results assemble
        // back in row order (bit-identical run-to-run); the first encode error
        // propagates deterministically. Stay sequential inside a rayon worker to
        // avoid nested oversubscription.
        let encode_rows =
            |range: std::ops::Range<usize>| -> Result<Vec<Option<(Array1<f64>, bool)>>, String> {
                range
                    .map(|row| {
                        // EXACT routing (#1777): pick the GLOBAL argmax of the
                        // routing score over the whole dictionary, not merely the
                        // best LSH-gathered candidate. `route_exact` certifies the
                        // sublinear gather against the universal `[0,1]` alignment
                        // bound and falls back to a full scan otherwise, so the
                        // returned atom is guaranteed to be the globally-best — no
                        // silently-missed ungathered atom.
                        let Some(route) =
                            index.route_exact(sketch, targets.row(row), budget, true)
                        else {
                            // Empty dictionary: flag for the exact fallback.
                            return Ok(None);
                        };
                        let best_atom = route.atom;
                        // Fit-quality floor: even the globally-best atom may align
                        // only weakly with this row (no atom fits it). A finite
                        // alignment below the floor — or a NaN, the zero-norm
                        // ‖d‖ = 0 row — flags for the exact multi-start fallback
                        // rather than encoding against a poorly-fitting atom. This is
                        // a quality gate, not a routing-correctness gate; routing is
                        // already exact. See CANDIDATE_ROUTING_MIN_ALIGNMENT.
                        if !route.alignment.is_finite()
                            || route.alignment < CANDIDATE_ROUTING_MIN_ALIGNMENT
                        {
                            return Ok(None);
                        }
                        let atom = atoms.get(best_atom).ok_or_else(|| {
                            format!(
                                "certified_encode_with_index: proposed atom {best_atom} out of range"
                            )
                        })?;
                        let (t, cert) = self.certified_encode_row(
                            atom,
                            best_atom,
                            targets.row(row),
                            amplitudes[row],
                        )?;
                        // Heterogeneous-atom dictionaries with different latent_dim
                        // per atom are not supported by the batched API: the caller
                        // declares one shared `latent_dim` for the output tensor.
                        // Silently zeroing the coord row while recording a
                        // certified=true flag would produce corrupted
                        // reconstructions downstream — error loudly instead.
                        if t.len() != latent_dim {
                            return Err(format!(
                                "certified_encode_with_index: atom {best_atom} returned t.len()={} \
                                 but declared latent_dim={latent_dim}; heterogeneous-dim \
                                 dictionaries are not supported by this batched encode path",
                                t.len()
                            ));
                        }
                        Ok(Some((t, cert.certified())))
                    })
                    .collect()
            };
        let rows: Vec<Option<(Array1<f64>, bool)>> =
            if n >= ENCODE_BATCH_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
                use rayon::prelude::*;
                const CHUNK: usize = 256;
                let n_chunks = n.div_ceil(CHUNK);
                let chunked: Vec<Vec<Option<(Array1<f64>, bool)>>> = (0..n_chunks)
                    .into_par_iter()
                    .map(|c| {
                        let start = c * CHUNK;
                        let end = (start + CHUNK).min(n);
                        encode_rows(start..end)
                    })
                    .collect::<Result<_, _>>()?;
                chunked.into_iter().flatten().collect()
            } else {
                encode_rows(0..n)?
            };
        let mut coords = Array2::<f64>::zeros((n, latent_dim));
        let mut certified = Vec::with_capacity(n);
        for (row, slot) in rows.into_iter().enumerate() {
            match slot {
                Some((t, cert)) => {
                    coords.row_mut(row).assign(&t);
                    certified.push(cert);
                }
                None => certified.push(false),
            }
        }
        Ok(EncodeResult::from_rows(coords, certified))
    }

    /// LSH-routed AMORTIZED (distilled) encode — the production token-rate
    /// encoder of #1026 ladder item 3. Identical routing to
    /// [`Self::certified_encode_with_index`] (LSH proposes the best-aligned atom,
    /// the atlas routes to the in-atom nearest chart), but the in-atom encode is
    /// the closed-form per-chart Jacobian predictor + certificate gate of
    /// [`Self::amortized_encode_row`] rather than the certified Newton-refinement
    /// path.
    /// This is the deployment path: the distilled affine map produces the encode
    /// in one mat-vec, the Kantorovich certificate decides trust-or-fallback per
    /// row, and uncertified rows (the adversarial tail the thread expects to
    /// concentrate on rare tokens) are flagged for the exact multi-start solve —
    /// compute goes where the questions are. Row-independent against the frozen
    /// dictionary, so the batch fans out over rows with deterministic row-order
    /// assembly (bit-identical run-to-run).
    pub fn amortized_encode_with_index<S: AtomFrameSketch + Sync>(
        &self,
        atoms: &[SaeManifoldAtom],
        index: &SaeCandidateIndex,
        sketch: &S,
        targets: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
        latent_dim: usize,
    ) -> Result<EncodeResult, String> {
        let n = targets.nrows();
        if amplitudes.len() != n {
            return Err(format!(
                "amortized_encode_with_index: amplitudes len {} != rows {n}",
                amplitudes.len()
            ));
        }
        let budget = auto_candidate_budget(atoms.len().max(1));
        let encode_rows =
            |range: std::ops::Range<usize>| -> Result<Vec<Option<(Array1<f64>, bool)>>, String> {
                range
                    .map(|row| {
                        // EXACT routing (#1777): global argmax of the routing score,
                        // not just the best LSH-gathered candidate (see
                        // certified_encode_with_index for the full rationale).
                        let Some(route) =
                            index.route_exact(sketch, targets.row(row), budget, true)
                        else {
                            return Ok(None);
                        };
                        let best_atom = route.atom;
                        // Fit-quality floor (not a routing-correctness gate; routing
                        // is exact): even the globally-best atom may fit a row poorly,
                        // and a NaN alignment is the zero-norm ‖d‖ = 0 row. Either way
                        // flag for the exact multi-start fallback. See
                        // CANDIDATE_ROUTING_MIN_ALIGNMENT.
                        if !route.alignment.is_finite()
                            || route.alignment < CANDIDATE_ROUTING_MIN_ALIGNMENT
                        {
                            return Ok(None);
                        }
                        let atom = atoms.get(best_atom).ok_or_else(|| {
                            format!(
                                "amortized_encode_with_index: proposed atom {best_atom} out of range"
                            )
                        })?;
                        let (t, cert) = self.amortized_encode_row(
                            atom,
                            best_atom,
                            targets.row(row),
                            amplitudes[row],
                        )?;
                        if t.len() != latent_dim {
                            return Err(format!(
                                "amortized_encode_with_index: atom {best_atom} returned t.len()={} \
                                 but declared latent_dim={latent_dim}; heterogeneous-dim \
                                 dictionaries are not supported by this batched encode path",
                                t.len()
                            ));
                        }
                        Ok(Some((t, cert.certified())))
                    })
                    .collect()
            };
        let rows: Vec<Option<(Array1<f64>, bool)>> =
            if n >= ENCODE_BATCH_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
                use rayon::prelude::*;
                const CHUNK: usize = 256;
                let n_chunks = n.div_ceil(CHUNK);
                let chunked: Vec<Vec<Option<(Array1<f64>, bool)>>> = (0..n_chunks)
                    .into_par_iter()
                    .map(|c| {
                        let start = c * CHUNK;
                        let end = (start + CHUNK).min(n);
                        encode_rows(start..end)
                    })
                    .collect::<Result<_, _>>()?;
                chunked.into_iter().flatten().collect()
            } else {
                encode_rows(0..n)?
            };
        let mut coords = Array2::<f64>::zeros((n, latent_dim));
        let mut certified = Vec::with_capacity(n);
        for (row, slot) in rows.into_iter().enumerate() {
            match slot {
                Some((t, cert)) => {
                    coords.row_mut(row).assign(&t);
                    certified.push(cert);
                }
                None => certified.push(false),
            }
        }
        Ok(EncodeResult::from_rows(coords, certified))
    }

    /// LSH-routed FAST amortized encode over the WHOLE dictionary — the
    /// multi-atom, corpus-rate analogue of [`Self::amortized_encode_with_index`].
    ///
    /// `amortized_encode_with_index` routes per row, then runs the per-row
    /// closed-form predictor + Kantorovich certificate + cold cross-check on each
    /// row independently. This fast variant keeps the SAME per-row EXACT routing
    /// (`index.route_exact` + the fit-quality floor), but replaces the per-row
    /// predictor with the GEMM-batched [`Self::amortized_encode_batch_fast`]:
    /// it GROUPS rows by their global-argmax atom and runs one batched affine-
    /// predictor pass per atom-group (a routing GEMM + a predictor GEMM each),
    /// reproducing a traditional SAE's whole-dictionary `W·x+b` throughput. No
    /// per-row certificate — this is the speed mode validated as accuracy-parity
    /// with the certified solve (`fast_forward_is_accuracy_parity_with_certified`).
    ///
    /// Returns the per-row latent coords and a valid-mask: `false` for a row with
    /// an empty dictionary, a sub-threshold/NaN routing alignment, or one the batched
    /// predictor could not fire on (no evaluator / singular Gauss–Newton block /
    /// non-finite-or-zero amplitude). Each row is written exactly once (disjoint
    /// per-atom groups), so the result is independent of group iteration order.
    pub fn amortized_encode_with_index_fast<S: AtomFrameSketch + Sync>(
        &self,
        atoms: &[SaeManifoldAtom],
        index: &SaeCandidateIndex,
        sketch: &S,
        targets: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
        latent_dim: usize,
    ) -> Result<(Array2<f64>, Vec<bool>), String> {
        let n = targets.nrows();
        if amplitudes.len() != n {
            return Err(format!(
                "amortized_encode_with_index_fast: amplitudes len {} != rows {n}",
                amplitudes.len()
            ));
        }
        let budget = auto_candidate_budget(atoms.len().max(1));
        let mut coords = Array2::<f64>::zeros((n, latent_dim));
        let mut valid = vec![false; n];
        // ── Single allocation-free pass: route each row, apply the CACHED predictor.
        //
        // Routing sublinearity (massive-K, K≈32k): the certified path uses
        // `route_exact`, whose universal-bound LSH certificate only fires at the
        // alignment ceiling (≈1.0); for any real dictionary (`alignment < 1`) it
        // falls back to `brute_force_best_atom` — an O(K) full scan PER ROW, making
        // the encode O(N·K). This SPEED path takes the LSH gather's best-aligned atom
        // directly (`propose` scores only ~budget candidates → O(log K)); a rare miss
        // is caught by the fit-quality floor + the downstream certificate/exact
        // fallback (the documented speed/accuracy tradeoff).
        //
        // Allocation: the predictor uses only OFFLINE-cached atlas data (per-chart
        // `recon_center` for routing + `amortized_jacobian`/`amortized_base` for the
        // `t̂ = base + (1/z)·A₁·x` mat-vec), so NO per-row or per-atom heap buffer is
        // allocated. This replaces the old per-atom-group GEMM sub-batch — which at
        // `K ≫ N` degenerated to ONE row per group, so its per-group buffers (x_sub,
        // recon-centers, predictors) dominated and made the "fast" path allocation-
        // bound. Now the only per-row allocation is inside `index.propose`.
        for row in 0..n {
            let dir = targets.row(row);
            let proposal = index.propose(sketch, dir, budget, true);
            let Some(&best_atom) = proposal.proposed.first() else {
                continue; // nothing gathered (empty dictionary / probe-dim mismatch)
            };
            // Fit-quality floor: the best gathered atom still fits this row poorly,
            // or the alignment is NaN (zero-norm row) — flag for the exact fallback.
            let alignment = sketch.alignment(best_atom, dir);
            if !alignment.is_finite() || alignment < CANDIDATE_ROUTING_MIN_ALIGNMENT {
                continue;
            }
            let atom = atoms.get(best_atom).ok_or_else(|| {
                format!("amortized_encode_with_index_fast: proposed atom {best_atom} out of range")
            })?;
            if atom.latent_dim != latent_dim {
                return Err(format!(
                    "amortized_encode_with_index_fast: atom {best_atom} latent_dim {} != declared \
                     {latent_dim}; heterogeneous-dim dictionaries are not supported by this path",
                    atom.latent_dim
                ));
            }
            let Some(atom_atlas) = self.atoms.get(best_atom) else {
                continue; // no atlas for this atom → predictor cannot fire (zeroed)
            };
            if amortized_predict_row(
                atom_atlas,
                dir,
                amplitudes[row],
                latent_dim,
                coords.row_mut(row),
            ) {
                valid[row] = true;
            }
        }
        Ok((coords, valid))
    }

    /// LSH-routed FAST full forward over the WHOLE dictionary: encode → decode,
    /// the multi-atom analogue of [`Self::amortized_reconstruct_batch_fast`]. Same
    /// sublinear per-row routing + per-atom grouping as
    /// [`Self::amortized_encode_with_index_fast`], but each group is run through
    /// the batched reconstruct (`m(t̂) = z·Φ(t̂)·B`) so the result is the per-row
    /// reconstruction in the ambient space. Rows that do not route/predict decode
    /// to an exact zero reconstruction and are flagged `false`.
    pub fn amortized_reconstruct_with_index_fast<S: AtomFrameSketch + Sync>(
        &self,
        atoms: &[SaeManifoldAtom],
        index: &SaeCandidateIndex,
        sketch: &S,
        targets: ArrayView2<'_, f64>,
        amplitudes: ArrayView1<'_, f64>,
    ) -> Result<(Array2<f64>, Vec<bool>), String> {
        let n = targets.nrows();
        let p = targets.ncols();
        if amplitudes.len() != n {
            return Err(format!(
                "amortized_reconstruct_with_index_fast: amplitudes len {} != rows {n}",
                amplitudes.len()
            ));
        }
        let budget = auto_candidate_budget(atoms.len().max(1));
        // SUBLINEAR routing for the SPEED-mode full forward — mirror
        // `amortized_encode_with_index_fast`: take the LSH gather's best-aligned
        // atom (O(budget) candidates) instead of route_exact's O(K) full-scan
        // certification, keeping the whole fast encode→decode sublinear in K at
        // K=32k. The gather's best is the exact argmax on the vast majority of rows;
        // rare misses are caught by the fit-quality floor + downstream fallback.
        let mut groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for row in 0..n {
            let dir = targets.row(row);
            let proposal = index.propose(sketch, dir, budget, true);
            let Some(&best_atom) = proposal.proposed.first() else {
                continue; // nothing gathered (empty dictionary / probe-dim mismatch)
            };
            let alignment = sketch.alignment(best_atom, dir);
            if !alignment.is_finite() || alignment < CANDIDATE_ROUTING_MIN_ALIGNMENT {
                continue;
            }
            groups.entry(best_atom).or_default().push(row);
        }

        let mut recon = Array2::<f64>::zeros((n, p));
        let mut valid = vec![false; n];
        for (atom_idx, rows_here) in groups {
            let atom = atoms.get(atom_idx).ok_or_else(|| {
                format!(
                    "amortized_reconstruct_with_index_fast: proposed atom {atom_idx} out of range"
                )
            })?;
            if atom.output_dim() != p {
                return Err(format!(
                    "amortized_reconstruct_with_index_fast: atom {atom_idx} output_dim {} != target \
                     dim {p}",
                    atom.output_dim()
                ));
            }
            let mut x_sub = Array2::<f64>::zeros((rows_here.len(), p));
            let mut amp_sub = Array1::<f64>::zeros(rows_here.len());
            for (i, &row) in rows_here.iter().enumerate() {
                x_sub.row_mut(i).assign(&targets.row(row));
                amp_sub[i] = amplitudes[row];
            }
            let (sub_recon, sub_valid) = self.amortized_reconstruct_batch_fast(
                atom,
                atom_idx,
                x_sub.view(),
                amp_sub.view(),
            )?;
            for (i, &row) in rows_here.iter().enumerate() {
                if sub_valid[i] {
                    recon.row_mut(row).assign(&sub_recon.row(i));
                    valid[row] = true;
                }
            }
        }
        Ok((recon, valid))
    }
}

/// Offline `β = 1/λ_min(H_GN)` at a chart center from the Gauss-Newton block
/// `H_GN = J_mᵀ J_m` (residual-free). The offline `β` bounds the curvature the
/// online certificate sees: charts are placed where the encode lands, so the
/// representative residual is small and `H_GN` is the dominant, residual-free
/// curvature estimate. (The online per-row certificate still uses the FULL
/// Hessian; this is only the offline radius-sizing curvature.) Returns `None`
/// for a degenerate center (`λ_min ≤ 0`), which marks an uncertifiable chart.
pub(crate) fn center_beta(atom: &SaeManifoldAtom, center: &Array1<f64>, ridge: f64) -> Option<f64> {
    let evaluator = atom.basis_evaluator.as_ref()?.clone();
    let d = atom.latent_dim;
    let p = atom.output_dim();
    let m = atom.basis_size();
    let coords = center.view().to_shape((1, d)).ok()?.to_owned();
    let (_phi, jet) = evaluator.evaluate(coords.view()).ok()?;
    let decoder = &atom.decoder_coefficients;
    // J_m[axis] = Bᵀ (∂Φ/∂t_axis) ∈ ℝᵖ (amplitude-1; curvature scales with z²
    // and is absorbed conservatively by the amplitude-bounded Lipschitz term).
    let mut jm = Array2::<f64>::zeros((d, p));
    for axis in 0..d {
        for basis_col in 0..m {
            let dphi = jet[[0, basis_col, axis]];
            if dphi == 0.0 {
                continue;
            }
            for out in 0..p {
                jm[[axis, out]] += dphi * decoder[[basis_col, out]];
            }
        }
    }
    let mut h = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for b in 0..d {
            h[[a, b]] = jm.row(a).dot(&jm.row(b));
        }
        h[[a, a]] += ridge;
    }
    let (vals, _vecs) = h.eigh(Side::Lower).ok()?;
    let lambda_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    if lambda_min.is_finite() && lambda_min > 0.0 {
        Some(1.0 / lambda_min)
    } else {
        None
    }
}

/// #1154 — the amortized encoder's closed-form warm-start coordinate for one
/// row `x` against one chart at amplitude `z`:
///
/// ```text
/// t̂ = t_c + (1/z) · A₁ · (x − z · m₁(t_c)),
/// ```
///
/// a single `O(d·p)` mat-vec from the chart's precomputed IFT Jacobian `A₁` and
/// center reconstruction `m₁`. Returns `None` when the chart carries no
/// distilled Jacobian (singular Gauss–Newton block) or the amplitude is not
/// strictly positive and finite (a near-inactive atom, where the
/// amplitude-divided map is undefined) — in those cases the caller starts from
/// the chart center instead. Shared by the amortized encode (where `t̂` is the
/// prediction) and the exact certified encode (where `t̂` is the Newton
/// warm-start that then refines to stationarity, Design A).
pub(crate) fn amortized_warm_start(
    chart: &CertifiedChart,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
) -> Option<Array1<f64>> {
    let a1 = chart.amortized_jacobian.as_ref()?;
    if !(amplitude.is_finite() && amplitude.abs() > 0.0) {
        return None;
    }
    let d = a1.nrows();
    let mut t_hat = chart.region.center.clone();
    for (out_idx, &m1_out) in chart.recon_center.iter().enumerate().take(a1.ncols()) {
        let resid = x[out_idx] - amplitude * m1_out;
        for axis in 0..d {
            t_hat[axis] += a1[[axis, out_idx]] * resid / amplitude;
        }
    }
    Some(t_hat)
}

/// Single-row amortized predictor against ONE atom's cached atlas, writing the
/// encoded latent coordinate DIRECTLY into `out` (length `d`) with NO heap
/// allocation. Routes to the nearest certifiable chart by cached center-
/// reconstruction distance `‖x − m(t_c)‖²` (matching `amortized_encode_batch_fast`'s
/// per-chart routing), then applies that chart's precomputed affine predictor
/// `t̂ = base + (1/z)·A₁·x` (`base = t_c − A₁·m₁` is `chart.amortized_base`).
///
/// Returns `false` — leaving `out` at its incoming (zeroed) value — for exactly the
/// rows `amortized_encode_batch_fast` would flag: no certifiable chart, a nearest
/// chart with no distilled predictor (singular Gauss–Newton block), or an unusable
/// amplitude. This is the per-row core of the allocation-free massive-K fast encode.
pub(crate) fn amortized_predict_row(
    atom_atlas: &AtomEncodeAtlas,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    d: usize,
    mut out: ndarray::ArrayViewMut1<'_, f64>,
) -> bool {
    if !(amplitude.is_finite() && amplitude.abs() > 0.0) {
        return false;
    }
    // Nearest certifiable chart by the TRUE objective ‖x − z·recon_center‖²
    // (F1; `z = amplitude`, finite and non-zero by the guard above). First-wins on
    // ties (strict `<`) — same amplitude-scaled argmin as
    // `amortized_encode_batch_fast`'s route_idx.
    let mut best_ci: Option<usize> = None;
    let mut best_dist = f64::INFINITY;
    for (ci, chart) in atom_atlas.charts.iter().enumerate() {
        if chart.certified_radius <= 0.0 {
            continue;
        }
        let mut dist = 0.0;
        for (r, xv) in chart.recon_center.iter().zip(x.iter()) {
            let diff = amplitude * r - xv;
            dist += diff * diff;
        }
        if dist < best_dist {
            best_dist = dist;
            best_ci = Some(ci);
        }
    }
    let Some(ci) = best_ci else {
        return false;
    };
    let chart = &atom_atlas.charts[ci];
    // The nearest chart must carry a distilled predictor; otherwise flag (zeroed),
    // exactly as the per-chart `None` branch of `amortized_encode_batch_fast`.
    let (Some(a1), Some(base)) = (
        chart.amortized_jacobian.as_ref(),
        chart.amortized_base.as_ref(),
    ) else {
        return false;
    };
    let inv_z = 1.0 / amplitude;
    for axis in 0..d {
        out[axis] = base[axis] + a1.row(axis).dot(&x) * inv_z;
    }
    true
}

/// The amplitude-1 distilled amortized-encoder Jacobian at a chart center
/// (#1026 ladder item 3). Returns `(A₁, m₁)` where `m₁ = BᵀΦ(t_c) ∈ ℝᵖ` is the
/// amplitude-1 center reconstruction and `A₁ = (J₁ᵀJ₁ + ridge·I)⁻¹ J₁ ∈ ℝ^{d×p}`
/// is the implicit-function-theorem derivative of the encode map `x ↦ t`
/// (Gauss–Newton block — the residual-free, dominant curvature exactly as the
/// offline radius-sizing `β`). With these, the online encode of a row `x` at
/// amplitude `z` is the closed-form affine prediction
/// `t = t_c + (1/z)·A₁·(x − z·m₁)` — one mat-vec, no per-row factorization.
/// `None` when the basis has no jet or the Gauss–Newton block is singular (no
/// certifiable amortization), matching `center_beta`'s gate so a chart with a
/// finite `β` always carries a Jacobian and vice versa.
pub(crate) fn center_amortized_jacobian(
    atom: &SaeManifoldAtom,
    center: &Array1<f64>,
    ridge: f64,
) -> Option<(Array2<f64>, Array1<f64>)> {
    let evaluator = atom.basis_evaluator.as_ref()?.clone();
    let d = atom.latent_dim;
    let p = atom.output_dim();
    let m = atom.basis_size();
    let coords = center.view().to_shape((1, d)).ok()?.to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view()).ok()?;
    let decoder = &atom.decoder_coefficients;
    // m₁(t_c) = BᵀΦ(t_c) ∈ ℝᵖ (amplitude-1 center reconstruction).
    let mut recon = Array1::<f64>::zeros(p);
    for basis_col in 0..m {
        let phi_v = phi[[0, basis_col]];
        if phi_v == 0.0 {
            continue;
        }
        for out in 0..p {
            recon[out] += phi_v * decoder[[basis_col, out]];
        }
    }
    // J₁[axis] = Bᵀ (∂Φ/∂t_axis) ∈ ℝᵖ (amplitude-1; z factors out analytically).
    let mut jm = Array2::<f64>::zeros((d, p));
    for axis in 0..d {
        for basis_col in 0..m {
            let dphi = jet[[0, basis_col, axis]];
            if dphi == 0.0 {
                continue;
            }
            for out in 0..p {
                jm[[axis, out]] += dphi * decoder[[basis_col, out]];
            }
        }
    }
    // H_GN = J₁ J₁ᵀ + ridge·I ∈ ℝ^{d×d}.
    let mut h = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for b in 0..d {
            h[[a, b]] = jm.row(a).dot(&jm.row(b));
        }
        h[[a, a]] += ridge;
    }
    let (vals, vecs) = h.eigh(Side::Lower).ok()?;
    let lambda_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    if !(lambda_min.is_finite() && lambda_min > 0.0) {
        return None;
    }
    // A₁ = H_GN⁻¹ J₁ via the eigendecomposition: H⁻¹ = Σ_i (1/λᵢ) vᵢ vᵢᵀ, so
    // A₁[:, out] = Σ_i (vᵢ · J₁[:, out]) / λᵢ · vᵢ. Column-by-column keeps it the
    // d×p Jacobian (one SPD solve reused across all p output channels).
    let mut a1 = Array2::<f64>::zeros((d, p));
    for out in 0..p {
        let jcol = jm.column(out);
        for (i, &lam) in vals.iter().enumerate() {
            if !(lam.is_finite() && lam > 0.0) {
                return None;
            }
            let vi = vecs.column(i);
            let coeff = vi.dot(&jcol) / lam;
            for row in 0..d {
                a1[[row, out]] += coeff * vi[row];
            }
        }
    }
    Some((a1, recon))
}

/// Route a target row to the nearest chart of an atom by reconstruction
/// distance: the chart whose center reconstruction `m(t_c)` is closest to `x`.
/// Returns the chart index and the distance, or `None` when the atom has no
/// charts.
pub(crate) fn nearest_chart(
    atom_atlas: &AtomEncodeAtlas,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
) -> Option<(usize, f64)> {
    if atom_atlas.charts.is_empty() {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for (idx, chart) in atom_atlas.charts.iter().enumerate() {
        if chart.certified_radius <= 0.0 {
            continue;
        }
        // F1: score the TRUE encode objective `‖x − z·m(t_c)‖²` at the row's
        // amplitude `z`, NOT the amplitude-1 `‖x − m(t_c)‖²`. The distilled
        // `chart.recon_center` is the amplitude-1 reconstruction `m₁(t_c)`; the
        // reconstruction actually compared against `x` is `z·m₁(t_c)`. Routing on
        // the amplitude-1 centers picks the wrong chart whenever `z ≠ 1` (e.g. a
        // small `z` makes a large-norm center reconstruct near a small `x`).
        // Distance accumulated in place, no temporary array.
        let mut dist = 0.0;
        for (r, xv) in chart.recon_center.iter().zip(x.iter()) {
            let diff = amplitude * r - xv;
            dist += diff * diff;
        }
        if best.map(|(_, b)| dist < b).unwrap_or(true) {
            best = Some((idx, dist));
        }
    }
    best
}

/// The `k` charts whose CENTER reconstruction `m(t_c)` is nearest to `x` in
/// ambient ‖·‖², returned as chart indices sorted by increasing distance (ties
/// broken by chart index — deterministic). Only certifiable charts
/// (`certified_radius > 0`) are considered, exactly like [`nearest_chart`], whose
/// single result is `nearest_charts_topk(.., 1)[0]`. Used by the certified encode
/// to refine the global basin on self-approaching atoms (see
/// [`CERTIFIED_ROUTING_TOPK`]).
pub(crate) fn nearest_charts_topk(
    atom_atlas: &AtomEncodeAtlas,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    k: usize,
) -> Vec<usize> {
    if atom_atlas.charts.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for (idx, chart) in atom_atlas.charts.iter().enumerate() {
        if chart.certified_radius <= 0.0 {
            continue;
        }
        // `m₁(t_c) = BᵀΦ(t_c)` is an OFFLINE per-chart constant already distilled
        // into `chart.recon_center` at build time (bit-for-bit the same φ·decoder
        // accumulation this used to recompute). Reuse it instead of re-evaluating
        // the basis at a fixed center for every row — that re-eval was the encode's
        // dominant per-row cost (charts × rows basis evals, each allocating the φ/jet
        // arrays). F1: score the amplitude-scaled `‖x − z·m₁(t_c)‖²` (the true
        // objective), not the amplitude-1 `‖m₁(t_c) − x‖²`. Computed in place.
        let mut dist = 0.0;
        for (r, xv) in chart.recon_center.iter().zip(x.iter()) {
            let diff = amplitude * r - xv;
            dist += diff * diff;
        }
        scored.push((idx, dist));
    }
    // Sort by distance, then chart index for a deterministic, first-wins order
    // consistent with `nearest_chart`'s strict-`<` tie rule.
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.into_iter().take(k).map(|(idx, _)| idx).collect()
}

/// Reconstruction error `‖x − z·m(t)‖` of an encoded coordinate `t` — the
/// criterion the certified encode minimizes over its candidate charts to pick the
/// GLOBAL basin. `m(t) = Bᵀ Φ(t)` is the amplitude-1 reconstruction; `z` is the
/// amplitude. A non-finite reconstruction returns `+∞` so it never wins.
pub fn encode_reconstruction_error(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    coord: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
) -> f64 {
    // Bare Euclidean residual norm — bit-identical to the metric-free encode.
    encode_reconstruction_error_core(
        atom,
        evaluator,
        coord,
        x,
        amplitude,
        &EncodeObjective::euclidean(),
    )
}

/// Objective-aware reconstruction error (F3): the WHITENED residual norm
/// `‖M^{1/2} r‖ = ‖Uᵀ r‖` when a metric is active, so the candidate-ranking /
/// warm-start SSE guard measures error in the SAME metric the certified objective
/// minimizes — an unwhitened `‖r‖₂` guard would rank candidates by a different
/// functional than the one being certified. With no metric this is `‖r‖₂`, exactly
/// the historical guard.
pub(crate) fn encode_reconstruction_error_core(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    coord: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    objective: &EncodeObjective<'_>,
) -> f64 {
    let d = atom.latent_dim;
    let p = atom.output_dim();
    let m = atom.basis_size();
    let coords = match coord.to_shape((1, d)) {
        Ok(c) => c.to_owned(),
        Err(_) => return f64::INFINITY,
    };
    let Ok((phi, _jet)) = evaluator.evaluate(coords.view()) else {
        return f64::INFINITY;
    };
    let mut residual = Array1::<f64>::zeros(p);
    for out in 0..p {
        let mut recon = 0.0;
        for basis_col in 0..m {
            recon += phi[[0, basis_col]] * atom.decoder_coefficients[[basis_col, out]];
        }
        residual[out] = x[out] - amplitude * recon;
    }
    // `½ rᵀ M r = ½‖Uᵀr‖²` under `M = U Uᵀ`; the guard reports the metric norm
    // `‖Uᵀr‖`. Euclidean (`None`) accumulates `Σ r²` in the SAME element order as
    // the historical loop, so the metric-free guard is bit-for-bit unchanged.
    let err2 = match objective.metric_factor {
        Some(u) => {
            let utr = u.t().dot(&residual);
            utr.dot(&utr)
        }
        None => {
            let mut e = 0.0;
            for out in 0..p {
                e += residual[out] * residual[out];
            }
            e
        }
    };
    if err2.is_finite() {
        err2.sqrt()
    } else {
        f64::INFINITY
    }
}

/// Maximum number of chart centers laid down per atom (the SHAPE_BAND grid
/// point cap; mirrors `SHAPE_BAND_MAX_POINTS` in the atom band machinery).
pub(crate) const SHAPE_BAND_MAX_POINTS: usize = 512;

/// Lay down chart centers on an atom's coordinate grid (the SHAPE_BAND grid
/// idiom): a regular grid spanning the compact latent domain for periodic /
/// sphere / torus atoms, and a strided cover of the latent axes for unbounded
/// (Duchon / Euclidean) atoms.
///
/// Periodic / torus latents are fractions of one period, so the per-axis grid
/// spans `[0, 1)`; the sphere chart spans `lat ∈ [−π/2, π/2]`, `lon ∈ [−π, π)`.
/// These conventions match the basis evaluators (the fraction-of-period circle
/// harmonic and the lat/lon sphere chart).
/// Squared coordinate distance between two latent points under the atom's chart
/// geometry: per-axis WRAPPED distance `min(|a−b|, period−|a−b|)` on periodic
/// (circle) axes — period 1 to match `chart_center_grid`'s `[0,1)` torus tiling
/// — and plain difference on line axes. Used to place + size data-driven charts.
pub(crate) fn coord_dist_sq(
    atom: &SaeManifoldAtom,
    a: ArrayView1<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> f64 {
    // Per-axis period comes from the ONE canonical source `latent_axis_period` —
    // the SAME convention the certified-encode `in_chart` guard uses (via
    // `latent_coordinate_distance`): period-1 fraction axes for periodic/torus and
    // the cylinder angle, period-2π on the sphere LONGITUDE (axis 1), and
    // NON-periodic otherwise — including the sphere LATITUDE (axis 0), which ranges
    // over [−π/2, π/2] and must NOT wrap. A former local `periodic_axis` closure
    // wrapped BOTH sphere axes at period 1, so two genuinely-far radian longitudes
    // (e.g. 0 and 3 rad) collapsed to distance 0 — corrupting the farthest-point
    // placement AND the nearest-neighbour radii in `data_driven_chart_centers` for
    // sphere atoms, and disagreeing with the metric the encode's `in_chart`
    // soundness guard enforces. Delegating keeps the two metrics identical.
    let mut acc = 0.0;
    for axis in 0..a.len().min(b.len()) {
        let mut d = (a[axis] - b[axis]).abs();
        if let Some(period) = latent_axis_period(atom, axis) {
            let wrapped = d.rem_euclid(period);
            d = wrapped.min(period - wrapped);
        }
        acc += d * d;
    }
    acc
}

/// Greedy farthest-point sampling of up to `max_charts` chart centers from the
/// atom's latent `coords` (n × d), with each center's nominal radius set to half
/// the distance to its nearest neighbor center (floored, so a singleton/coincident
/// cluster still gets a usable ball). Deterministic: seeds from row 0, then
/// repeatedly adds the coord maximally far (under [`coord_dist_sq`]) from the
/// chosen set — coverage-maximizing and reproducible run-to-run.
pub(crate) fn data_driven_chart_centers(
    atom: &SaeManifoldAtom,
    coords: ArrayView2<'_, f64>,
    max_charts: usize,
) -> Result<(Array2<f64>, Vec<f64>), String> {
    let n = coords.nrows();
    let d = coords.ncols();
    if d != atom.latent_dim {
        return Err(format!(
            "data_driven_chart_centers: coords have {d} cols but atom latent_dim is {}",
            atom.latent_dim
        ));
    }
    if n == 0 {
        return Ok((Array2::<f64>::zeros((0, d)), Vec::new()));
    }
    let k = max_charts.min(n);
    // Farthest-point sampling: maintain each row's distance to the nearest chosen
    // center, add the row with the maximum such distance each step.
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    chosen.push(0);
    let mut nearest_sq: Vec<f64> = (0..n)
        .map(|r| coord_dist_sq(atom, coords.row(r), coords.row(0)))
        .collect();
    while chosen.len() < k {
        // Pick the row farthest from the current center set (first-wins tie).
        let mut best = 0usize;
        let mut best_d = -1.0;
        for r in 0..n {
            if nearest_sq[r] > best_d {
                best_d = nearest_sq[r];
                best = r;
            }
        }
        if best_d <= 0.0 {
            break; // all remaining rows coincide with a chosen center.
        }
        chosen.push(best);
        for r in 0..n {
            let dr = coord_dist_sq(atom, coords.row(r), coords.row(best));
            if dr < nearest_sq[r] {
                nearest_sq[r] = dr;
            }
        }
    }
    let m = chosen.len();
    let mut centers = Array2::<f64>::zeros((m, d));
    for (i, &row) in chosen.iter().enumerate() {
        centers.row_mut(i).assign(&coords.row(row));
    }
    // Per-center radius = half the nearest-OTHER-center distance, floored so a
    // coincident pair still yields a positive ball, capped at 0.5 (the largest
    // meaningful half-period on a unit circle).
    let mut radii = vec![0.0_f64; m];
    for i in 0..m {
        let mut nn = f64::INFINITY;
        for j in 0..m {
            if i == j {
                continue;
            }
            let dsq = coord_dist_sq(atom, centers.row(i), centers.row(j));
            if dsq < nn {
                nn = dsq;
            }
        }
        let r = if nn.is_finite() { 0.5 * nn.sqrt() } else { 0.5 };
        radii[i] = r.max(1.0e-3).min(0.5);
    }
    Ok((centers, radii))
}

pub(crate) fn chart_center_grid(atom: &SaeManifoldAtom, resolution: usize) -> Array2<f64> {
    use crate::manifold::SaeAtomBasisKind::*;
    let d = atom.latent_dim;
    match &atom.basis_kind {
        Periodic | Torus => regular_product_grid(d, resolution, 0.0, 1.0, false),
        // Cylinder `S¹ × ℝ`: axis 0 is the periodic circle `[0, 1)` (no
        // endpoint, like the harmonic axes); axis 1 is the unbounded line,
        // covered by a strided unit box `[-0.5, 0.5]` about the origin (like the
        // Euclidean patch). The certified radius refines each chart; out-of-cover
        // line starts route to the exact fallback honestly.
        Cylinder if d == 2 => cylinder_chart_center_grid(resolution),
        Cylinder => regular_product_grid(d, resolution, -0.5, 0.5, true),
        Mobius if d == 2 => mobius_chart_center_grid(resolution),
        Mobius => regular_product_grid(d, resolution, -1.0, 1.0, true),
        Sphere if d == 2 => sphere_latlon_grid(resolution),
        Linear | Sphere | Duchon | EuclideanPatch | Poincare | Precomputed(_) | FiniteSet => {
            // Unbounded / non-compact latents (and the finite-set index axis): a
            // strided cover of a unit box about the origin per axis. The certified
            // radius refines each chart; out-of-cover starts route to the exact
            // fallback honestly.
            regular_product_grid(d, resolution, -0.5, 0.5, true)
        }
    }
}

/// A regular `resolution`-per-axis product grid over `[lo, hi]^d`, capped at
/// [`SHAPE_BAND_MAX_POINTS`] total points (the per-axis resolution is reduced
/// until the product fits). When `include_endpoint` the last grid point sits at
/// `hi`; otherwise the axis is treated as periodic and stops one step short.
/// Per-axis resolution actually used by [`regular_product_grid`] after the
/// [`SHAPE_BAND_MAX_POINTS`] product cap. Chart radii must be derived from THIS
/// (not the raw `resolution`), otherwise for `resolution^d > SHAPE_BAND_MAX_POINTS`
/// the grid spacing is coarser than the radius and the charts leave gaps.
pub(crate) fn capped_per_axis(d: usize, resolution: usize) -> usize {
    let mut per_axis = resolution.max(2);
    while per_axis.saturating_pow(d as u32) > SHAPE_BAND_MAX_POINTS && per_axis > 2 {
        per_axis -= 1;
    }
    per_axis
}

pub(crate) fn regular_product_grid(
    d: usize,
    resolution: usize,
    lo: f64,
    hi: f64,
    include_endpoint: bool,
) -> Array2<f64> {
    if d == 0 {
        return Array2::<f64>::zeros((1, 0));
    }
    let per_axis = capped_per_axis(d, resolution);
    let total = per_axis.saturating_pow(d as u32).max(1);
    let denom = if include_endpoint {
        (per_axis.max(2) - 1) as f64
    } else {
        per_axis as f64
    };
    let mut grid = Array2::<f64>::zeros((total, d));
    let mut idx = vec![0usize; d];
    for flat in 0..total {
        for axis in 0..d {
            let frac = idx[axis] as f64 / denom;
            grid[[flat, axis]] = lo + (hi - lo) * frac;
        }
        for axis in (0..d).rev() {
            idx[axis] += 1;
            if idx[axis] < per_axis {
                break;
            }
            idx[axis] = 0;
        }
    }
    grid
}

/// Lat/lon sphere chart grid: `lat ∈ [−π/2, π/2]`, `lon ∈ [−π, π)`, matching
/// the [`crate::manifold::SphereChartEvaluator`] convention.
pub(crate) fn sphere_latlon_grid(resolution: usize) -> Array2<f64> {
    use std::f64::consts::PI;
    // Per-axis cap derived from the shared point budget rather than a hardcoded
    // literal (#2071): the largest r with r² ≤ SHAPE_BAND_MAX_POINTS. Equals 22
    // at the current budget (22²=484≤512), but now tracks the budget if it
    // changes instead of silently desyncing from the sibling grid arms.
    let r_cap = SHAPE_BAND_MAX_POINTS.isqrt();
    let r = resolution.max(2).min(r_cap);
    let mut grid = Array2::<f64>::zeros((r * r, 2));
    for i in 0..r {
        let lat = -PI / 2.0 + PI * (i as f64 + 0.5) / r as f64;
        for j in 0..r {
            let lon = -PI + 2.0 * PI * (j as f64) / r as f64;
            grid[[i * r + j, 0]] = lat;
            grid[[i * r + j, 1]] = lon;
        }
    }
    grid
}

/// Cylinder `S¹ × ℝ` chart-center grid: axis 0 sweeps the periodic circle over
/// one period `[0, 1)` (no endpoint, matching the harmonic axis), axis 1 strides
/// a unit box `[−0.5, 0.5]` about the origin on the unbounded line (with
/// endpoint). Capped at [`SHAPE_BAND_MAX_POINTS`] total centers.
pub(crate) fn cylinder_chart_center_grid(resolution: usize) -> Array2<f64> {
    let mut per_axis = resolution.max(2);
    while per_axis * per_axis > SHAPE_BAND_MAX_POINTS && per_axis > 2 {
        per_axis -= 1;
    }
    let total = per_axis * per_axis;
    let line_denom = (per_axis.max(2) - 1) as f64;
    let mut grid = Array2::<f64>::zeros((total, 2));
    for i in 0..per_axis {
        // Periodic axis 0: stop one step short of the period.
        let circle = i as f64 / per_axis as f64;
        for j in 0..per_axis {
            // Line axis 1: include the endpoint of the unit box.
            let line = -0.5 + (j as f64) / line_denom;
            grid[[i * per_axis + j, 0]] = circle;
            grid[[i * per_axis + j, 1]] = line;
        }
    }
    grid
}

/// Möbius double-cover chart grid: angle `s ∈ [0, 2)` and bounded width
/// `w ∈ [-1, 1]`. The deck identification is encoded by the basis, so the
/// atlas covers the ordinary cylindrical chart without duplicating a seam.
pub(crate) fn mobius_chart_center_grid(resolution: usize) -> Array2<f64> {
    let mut per_axis = resolution.max(2);
    while per_axis * per_axis > SHAPE_BAND_MAX_POINTS && per_axis > 2 {
        per_axis -= 1;
    }
    let mut grid = Array2::<f64>::zeros((per_axis * per_axis, 2));
    let width_denom = (per_axis - 1) as f64;
    for i in 0..per_axis {
        let angle = 2.0 * i as f64 / per_axis as f64;
        for j in 0..per_axis {
            let width = -1.0 + 2.0 * j as f64 / width_denom;
            grid[[i * per_axis + j, 0]] = angle;
            grid[[i * per_axis + j, 1]] = width;
        }
    }
    grid
}

/// Nominal in-chart radius: half the inter-center grid spacing, so charts tile
/// the domain. For compact latents this is the grid step; for unbounded latents
/// a unit default that the certified radius refines.
pub(crate) fn chart_nominal_radius(atom: &SaeManifoldAtom, resolution: usize) -> f64 {
    use crate::manifold::SaeAtomBasisKind::*;
    match &atom.basis_kind {
        Periodic | Torus => 0.5 / (capped_per_axis(atom.latent_dim, resolution) as f64),
        // Must use the SAME capped per-axis count `sphere_latlon_grid` lays the
        // centers on: the coarsest tiling step is the longitude half-spacing `π/r`
        // with `r` the grid's per-axis count. Deriving the radius from the RAW
        // resolution makes it smaller than the grid spacing once the grid caps,
        // leaving gaps between charts so rows in the gaps spuriously route to the
        // exact fallback (the hazard `capped_per_axis` documents for the regular
        // grid). The cap is DERIVED, not a literal (#2071): `sphere_latlon_grid`
        // uses `SHAPE_BAND_MAX_POINTS.isqrt()` — the largest `r` with `r² ≤`
        // the band-point budget — so the two stay in lockstep if the budget moves
        // (a hardcoded `22` here silently desyncs the moment the budget changes).
        Sphere => {
            let r_cap = SHAPE_BAND_MAX_POINTS.isqrt();
            std::f64::consts::PI / (resolution.max(2).min(r_cap) as f64)
        }
        // Cylinder charts tile two heterogeneous axes (a `[0,1)` periodic step
        // and a unit-box line step); the chart radius is a single scalar, so we
        // take the tighter (periodic) step `0.5/res` to keep every chart valid
        // on both axes. The certified Kantorovich radius refines it per chart.
        Cylinder => 0.5 / (capped_per_axis(atom.latent_dim, resolution) as f64),
        // Angle spacing is `2/r` and width spacing is `2/(r-1)`; their
        // half-spacings are `1/r` and `1/(r-1)`, so the angular axis is the
        // conservative scalar chart radius.
        Mobius => 1.0 / (capped_per_axis(atom.latent_dim, resolution) as f64),
        Linear | Duchon | EuclideanPatch | Poincare | Precomputed(_) | FiniteSet => {
            1.0 / (resolution.max(2) as f64)
        }
    }
}

/// Build the [`ChartRegion`] for a center, attaching the radial r_min / r_max
/// bracket for Duchon atoms (the chart's distance range to the kernel centers).
pub(crate) fn chart_region(
    atom: &SaeManifoldAtom,
    center: Array1<f64>,
    radius: f64,
) -> ChartRegion {
    use crate::manifold::SaeAtomBasisKind::*;
    let region = ChartRegion::new(center.clone(), radius);
    match &atom.basis_kind {
        Duchon => {
            // r ranges over [‖t_c‖ − radius, ‖t_c‖ + radius] about the single
            // origin-anchored center used by the conservative radial bound.
            //
            // The lower bound must be `max(0, center_norm − radius)` — NOT floored
            // at `radius`. When the chart contains the kernel center
            // (`center_norm < radius`, true r_min = 0), flooring at `radius`
            // would give a finite, NON-CONSERVATIVE `r_min`, causing the
            // hessian_sup / third_sup formulas (which divide by r_min) to
            // underestimate the Lipschitz constant and potentially grant a false
            // Kantorovich certificate. Flooring at `f64::MIN_POSITIVE` instead
            // correctly drives the formulas toward ∞, producing a very large L
            // that will NEVER certify (rows route to the exact multi-start
            // fallback) — conservative and sound.
            let center_norm = center.dot(&center).sqrt();
            let r_min = (center_norm - radius).max(f64::MIN_POSITIVE);
            let r_max = center_norm + radius;
            region.with_radial_bounds(r_min, r_max)
        }
        // Cylinder has no radial kernel block (it is a harmonic × polynomial
        // tensor, not a Duchon radial basis), so it needs no radial r_min/r_max.
        Periodic | Sphere | Torus | Cylinder | Mobius | Linear | EuclideanPatch | Poincare
        | Precomputed(_) | FiniteSet => region,
    }
}

/// Per-atom ambient tangents at the given coords: for atom `k` and row `i`, the
/// `d_k × p` matrix whose axis-`a` row is `∂m_k/∂t_a = (∂Φ/∂t_a)·B_k`, the image
/// tangent the joint Hessian couples through. `None` for an atom with no basis
/// evaluator (its coordinate is not differentiable, so it carries no coupling).
fn atom_row_tangents(
    atom: &SaeManifoldAtom,
    coords: ArrayView2<'_, f64>,
) -> Result<Option<Vec<Array2<f64>>>, String> {
    let Some(evaluator) = atom.basis_evaluator.as_ref() else {
        return Ok(None);
    };
    let n = coords.nrows();
    let d = atom.latent_dim;
    let p = atom.output_dim();
    let (_phi, jet) = evaluator.evaluate(coords)?; // jet: (n, M, d)
    let m = jet.shape()[1];
    let b = &atom.decoder_coefficients; // (M, p)
    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        let mut tan = Array2::<f64>::zeros((d, p));
        for axis in 0..d {
            for out_col in 0..p {
                let mut acc = 0.0;
                for basis_col in 0..m {
                    acc += jet[[row, basis_col, axis]] * b[[basis_col, out_col]];
                }
                tan[[axis, out_col]] = acc;
            }
        }
        out.push(tan);
    }
    Ok(Some(out))
}

/// Smallest eigenvalue of the symmetric `d × d` Gauss–Newton curvature block
/// `z² · T Tᵀ` (`T` is `d × p`). `d = 1` is the scalar fast path; general `d`
/// uses a symmetric eigensolve.
fn min_curvature_eigenvalue(tan: &Array2<f64>, z: f64) -> Result<f64, String> {
    let d = tan.nrows();
    if d == 0 {
        return Ok(0.0);
    }
    let z2 = z * z;
    if d == 1 {
        let row = tan.row(0);
        return Ok(z2 * row.dot(&row));
    }
    let mut gram = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for bx in 0..d {
            gram[[a, bx]] = z2 * tan.row(a).dot(&tan.row(bx));
        }
    }
    let (evals, _vecs) = gram
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("min_curvature_eigenvalue: eigh failed: {e:?}"))?;
    Ok(evals.iter().copied().fold(f64::INFINITY, f64::min))
}

/// Frobenius norm of the cross-coupling block `z_k z_j · T_k T_jᵀ`
/// (`T_k` is `d_k × p`, `T_j` is `d_j × p`). Frobenius upper-bounds the operator
/// norm, so a dominance decision made against it is SOUND.
fn cross_block_frobenius(tan_k: &Array2<f64>, tan_j: &Array2<f64>, zk: f64, zj: f64) -> f64 {
    let mut acc = 0.0;
    for a in 0..tan_k.nrows() {
        for bx in 0..tan_j.nrows() {
            let dot = tan_k.row(a).dot(&tan_j.row(bx));
            acc += dot * dot;
        }
    }
    (zk * zj).abs() * acc.sqrt()
}

/// The JOINT (multi-atom) encode-fallback fraction: the share of rows whose
/// per-row joint reconstruction problem across CO-ACTIVE atoms is NOT covered by
/// the composition of the per-atom certificates, so the row genuinely needs the
/// exact multi-start solve (reviewer condition #3 — the honest encode-tax cost
/// multiplier at scale).
///
/// The per-atom Kantorovich certificate certifies each atom's coordinate encode
/// IN ISOLATION — a block-diagonal view of the joint Hessian. The joint problem
/// couples co-active atoms through the off-diagonal blocks
/// `H_kj = z_k z_j J_k(t_k)ᵀ J_j(t_j)` (tangent-image inner products). When an
/// atom's own curvature block fails to dominate its coupling to the rest —
/// Gershgorin: `λ_min(H_kk) ≤ Σ_{j≠k} ‖H_kj‖` — the block-diagonal certificate
/// no longer implies a joint root, a second basin can open, and the row must go
/// to multi-start. This fraction GROWS with atom-image similarity and
/// co-activation: no per-atom certificate covers the joint problem.
///
/// The curvature block uses the Gauss–Newton form `z_k² J_kᵀ J_k` (exact for
/// flat/linear atoms, where the residual-curvature term vanishes identically);
/// the off-diagonal is measured in Frobenius norm, which upper-bounds the
/// operator norm, so a row DECLARED dominant is genuinely dominant and the
/// fraction never under-reports the multi-start need. Rows with fewer than two
/// co-active atoms have no cross blocks and are never counted as fallbacks.
///
/// `amplitude_floor` is the mass above which an atom counts as co-active; pass a
/// small positive value (a domain threshold on the assignment mass, not a solver
/// knob).
pub fn joint_encode_fallback_fraction(
    atoms: &[SaeManifoldAtom],
    coords: &[Array2<f64>],
    amplitudes: ArrayView2<'_, f64>,
    amplitude_floor: f64,
) -> Result<f64, String> {
    let k_atoms = atoms.len();
    let (n, amp_k) = amplitudes.dim();
    if amp_k != k_atoms {
        return Err(format!(
            "joint_encode_fallback_fraction: amplitudes have {amp_k} cols but {k_atoms} atoms"
        ));
    }
    if coords.len() != k_atoms {
        return Err(format!(
            "joint_encode_fallback_fraction: {} coord blocks but {k_atoms} atoms",
            coords.len()
        ));
    }
    if n == 0 || k_atoms == 0 {
        return Ok(0.0);
    }
    // F5 — FILTER BEFORE MATERIALIZE: the dense form built the full `n × K`
    // tangent tensor (`atom_row_tangents` over every atom, all N rows) BEFORE the
    // per-row activity filter, which OOMs at `K = 32k` even though each row couples
    // only through its co-active atoms. Cross-coupling exists only among a row's
    // co-active, differentiable atoms (`z > floor`), so materialize just that row's
    // active tangents, lazily, per row. `atom_row_tangents` is row-wise, so the
    // single-row slice is bit-identical to the batched evaluate the dense path
    // indexed — the fallback fraction is unchanged; only the peak allocation drops
    // from `O(n·K·d·p)` to the max active-set size per row.
    for (atom_idx, coord) in coords.iter().enumerate() {
        if coord.nrows() != n {
            return Err(format!(
                "joint_encode_fallback_fraction: coord block {atom_idx} has {} rows, expected {n}",
                coord.nrows()
            ));
        }
    }
    let mut fallback_rows = 0usize;
    for row in 0..n {
        // Gather this row's co-active, differentiable atoms (evaluator-less atoms
        // carry no coupling — same `is_some()` predicate the dense path applied via
        // `tangents[k].is_some()`).
        let active: Vec<usize> = (0..k_atoms)
            .filter(|&k| {
                amplitudes[[row, k]] > amplitude_floor && atoms[k].basis_evaluator.is_some()
            })
            .collect();
        if active.len() < 2 {
            continue; // no cross blocks: the per-atom certificate composes trivially
        }
        // Materialize ONLY this row's active tangents (one `d × p` block per active
        // atom), never the dense tensor. An atom that unexpectedly yields no tangent
        // (evaluator present but non-differentiable) simply drops out of the coupling,
        // exactly as a `None` block did in the dense path.
        let mut tans: Vec<Array2<f64>> = Vec::with_capacity(active.len());
        let mut zs: Vec<f64> = Vec::with_capacity(active.len());
        for &k in &active {
            let coord_row = coords[k].row(row).insert_axis(Axis(0)); // (1, d)
            if let Some(mut block) = atom_row_tangents(&atoms[k], coord_row)? {
                tans.push(block.pop().expect("single-row tangents carry one block"));
                zs.push(amplitudes[[row, k]]);
            }
        }
        if tans.len() < 2 {
            continue;
        }
        let mut row_needs_multistart = false;
        for (ki, tan_k) in tans.iter().enumerate() {
            let zk = zs[ki];
            let lam_min = min_curvature_eigenvalue(tan_k, zk)?;
            let mut coupling = 0.0;
            for (ji, tan_j) in tans.iter().enumerate() {
                if ji == ki {
                    continue;
                }
                coupling += cross_block_frobenius(tan_k, tan_j, zk, zs[ji]);
            }
            if lam_min <= coupling {
                row_needs_multistart = true;
                break;
            }
        }
        if row_needs_multistart {
            fallback_rows += 1;
        }
    }
    Ok(fallback_rows as f64 / n as f64)
}

#[cfg(test)]
mod encode_fix_tests {
    //! Unit tests for the two `certify_with_basin_warmup` fixes:
    //!   FIX #3 — wrap-aware chart containment (`in_chart` now measures latent
    //!            distance on the atom's periodic geometry via
    //!            [`latent_coordinate_distance`]).
    //!   FIX #4 — multiplicative sufficient-decrease bound on the warm-up loop
    //!            ([`warmup_progress_sufficient`]).
    use super::*;
    use crate::manifold::SaeAtomBasisKind;
    use ndarray::{Array1, Array2, Array3};

    /// Minimal atom carrying only a `basis_kind`; the wrap-aware metric reads no
    /// other field, so a tiny well-formed basis/decoder/penalty suffices.
    fn tiny_atom(kind: SaeAtomBasisKind, latent_dim: usize) -> SaeManifoldAtom {
        let m = 2usize;
        let phi = Array2::<f64>::eye(m);
        let jet = Array3::<f64>::zeros((m, m, latent_dim));
        let dec = Array2::<f64>::from_elem((m, 1), 0.5);
        let smooth = Array2::<f64>::eye(m);
        SaeManifoldAtom::new_with_provided_function_gram(
            "tiny", kind, latent_dim, phi, jet, dec, smooth,
        )
        .expect("tiny atom builds")
    }

    #[test]
    fn joint_normal_equations_use_the_shared_multi_atom_residual() {
        // u1=(1,0), u2=(1,1), x=(2,1). At t=(0,0), the shared residual is -x.
        // The joint normal equations recover the unique coefficients (1,1);
        // independent projections would instead produce (2,1.5).
        let jac = ndarray::array![[1.0_f64, 0.0], [1.0, 1.0]];
        let residual = ndarray::array![-2.0_f64, -1.0];
        let (_value, grad, hess) = joint_data_value_grad_hess(jac.view(), residual.view(), None);
        let step = joint_encode_damped_step(hess.view(), grad.view(), 1.0e-15)
            .expect("joint system factors")
            .expect("joint system is positive definite");
        assert!(
            (step[0] - 1.0).abs() < 1.0e-12,
            "first coefficient={}",
            step[0]
        );
        assert!(
            (step[1] - 1.0).abs() < 1.0e-12,
            "second coefficient={}",
            step[1]
        );
        let recon0 = step[0] + step[1];
        let recon1 = step[1];
        assert!((recon0 - 2.0).abs() < 1.0e-12 && (recon1 - 1.0).abs() < 1.0e-12);
    }

    /// FIX #3: a point on the far side of the wrap seam of a periodic axis is now
    /// IN-CHART under the wrap-aware metric, where the old raw-Euclidean sum used
    /// by `in_chart` rejected it. This is the exact predicate `in_chart` evaluates
    /// (`latent_coordinate_distance(atom, t, center) <= radius`).
    #[test]
    fn wrap_aware_containment_accepts_seam_point() {
        let atom = tiny_atom(SaeAtomBasisKind::Periodic, 1);
        let t = Array1::from(vec![0.99f64]);
        let center = Array1::from(vec![0.01f64]);
        let radius = 0.05;

        // Precondition: the OLD raw-Euclidean latent distance is 0.98 — far
        // outside the 0.05 ball, so the old `in_chart` REJECTED this iterate.
        let raw = (t[0] - center[0]).abs();
        assert!(
            raw > radius,
            "precondition: raw Euclidean distance {raw} must exceed the chart radius {radius}"
        );

        // The wrap-aware metric sees the true circle distance 0.02 (period 1.0),
        // so the seam point is now correctly IN-CHART.
        let d = latent_coordinate_distance(&atom, t.view(), center.view());
        assert!(
            (d - 0.02).abs() < 1e-12,
            "wrap-aware distance across the seam must be 0.02, got {d}"
        );
        assert!(
            d <= radius,
            "wrap-aware seam point must now be in-chart (d={d} <= r={radius})"
        );
    }

    /// Soundness guard for FIX #3: on a NON-periodic axis the metric must NOT
    /// wrap — the same coordinates stay at their full Euclidean separation and
    /// (correctly) remain OUT of the small ball. This preserves the
    /// never-issue-a-false-certificate invariant for flat-patch families.
    #[test]
    fn wrap_aware_containment_does_not_wrap_flat_axis() {
        let atom = tiny_atom(SaeAtomBasisKind::EuclideanPatch, 1);
        let t = Array1::from(vec![0.99f64]);
        let center = Array1::from(vec![0.01f64]);
        let d = latent_coordinate_distance(&atom, t.view(), center.view());
        assert!(
            (d - 0.98).abs() < 1e-12,
            "a flat (non-periodic) axis must keep the full 0.98 distance, got {d}"
        );
        assert!(
            d > 0.05,
            "flat-axis point correctly stays out of a 0.05 ball"
        );
    }

    /// FIX #4: a genuinely converging (Kantorovich-quadratic) `h`-sequence is
    /// untouched — every step clears the sufficient-decrease bar, so no
    /// converging row is regressed to the fallback.
    #[test]
    fn converging_h_sequence_is_never_flagged() {
        // Quadratic Newton contraction h_{k+1} = h_k^2 from an uncertified start.
        let mut h = 0.9f64;
        while h > KANTOROVICH_THRESHOLD {
            let h_next = h * h;
            assert!(
                warmup_progress_sufficient(h_next, h),
                "converging step {h} -> {h_next} must be accepted (no regression)"
            );
            h = h_next;
        }
    }

    /// FIX #4: a monotone `h`-sequence decreasing toward a limit ABOVE ½ (the
    /// pathological plateau that the old strict-decrease rule spun on until the
    /// increments fell below one ulp) now terminates in a BOUNDED, small number
    /// of `warmup_progress_sufficient` steps — flag-to-fallback rather than loop.
    #[test]
    fn plateau_h_sequence_terminates_bounded() {
        let limit = 0.65f64; // plateau limit strictly above ½: never certifies
        let ratio = 0.8f64; // geometric approach; ratio -> 1 as h -> limit
        let mut h = 2.0f64; // uncertified start
        let start = h;
        let mut accepted = 0usize;
        let mut iters = 0usize;
        loop {
            iters += 1;
            // Hard ceiling: proves boundedness. The OLD strict-decrease rule
            // would run ~1e15 steps here (h strictly decreases every step toward
            // 0.65 and only stalls below one ulp), so tripping this assert would
            // signal a regression to the unbounded behavior.
            assert!(
                iters < 10_000,
                "warm-up must terminate in a bounded number of steps, not loop"
            );
            let h_next = limit + (h - limit) * ratio; // monotone decreasing > limit
            if !warmup_progress_sufficient(h_next, h) {
                break; // flag to the exact fallback
            }
            accepted += 1;
            h = h_next;
        }
        // It made real progress before flagging (a warm-up, not an instant bail)…
        assert!(
            accepted >= 1,
            "expected the warm-up to accept at least one contracting step first"
        );
        // …and it flagged while still strictly above the plateau limit — exactly
        // where the OLD rule would have kept spinning (h is still shrinking).
        assert!(
            h > limit && h < start,
            "flagged mid-descent (limit < h={h} < start={start}); old rule would not stop here"
        );
        // Termination bound was tiny, not astronomical.
        assert!(iters < 200, "plateau flagged in {iters} steps (bounded)");
    }

    /// FIX #4: non-finite `h` (indefinite / blown-up Hessian) is treated as
    /// insufficient progress — the row flags rather than being accepted.
    #[test]
    fn non_finite_h_flags() {
        assert!(!warmup_progress_sufficient(f64::NAN, 0.9));
        assert!(!warmup_progress_sufficient(f64::INFINITY, 0.9));
        assert!(!warmup_progress_sufficient(0.5, f64::NAN));
    }

    /// BUG (sphere data-driven placement): `coord_dist_sq` must measure sphere
    /// coordinates in RADIANS via the canonical `latent_axis_period` — longitude
    /// (axis 1) wraps at 2π, latitude (axis 0) does NOT wrap — not at unit period
    /// on both axes. The old period-1 wrap collapsed genuinely-far longitudes to
    /// distance 0, corrupting `data_driven_chart_centers` for sphere atoms.
    #[test]
    fn coord_dist_sq_sphere_uses_radian_metric_not_unit_period() {
        let atom = tiny_atom(SaeAtomBasisKind::Sphere, 2);
        // Longitude differs by 3.0 rad: circle distance min(3, 2π−3)=3
        // (2π−3 ≈ 3.283) ⇒ dist² = 9. The old period-1 wrap read 3 mod 1 = 0.
        let a = Array1::from(vec![0.0f64, 0.0]);
        let b = Array1::from(vec![0.0f64, 3.0]);
        let dsq = coord_dist_sq(&atom, a.view(), b.view());
        assert!(
            (dsq - 9.0).abs() < 1e-9,
            "sphere longitude dist² must be 3²=9 (2π radian period), got {dsq}"
        );
        // Latitude (axis 0) must NOT wrap: 1.2 rad apart ⇒ dist² = 1.44.
        let c = Array1::from(vec![0.0f64, 0.0]);
        let e = Array1::from(vec![1.2f64, 0.0]);
        let dsq_lat = coord_dist_sq(&atom, c.view(), e.view());
        assert!(
            (dsq_lat - 1.44).abs() < 1e-9,
            "sphere latitude must not wrap; 1.2²=1.44, got {dsq_lat}"
        );
        // Must agree with the certified-encode metric (single source of truth).
        let dc = latent_coordinate_distance(&atom, a.view(), b.view());
        assert!(
            (dc * dc - dsq).abs() < 1e-9,
            "coord_dist_sq must equal latent_coordinate_distance² ({} vs {dsq})",
            dc * dc
        );
    }

    /// No-regression: periodic / torus axes still wrap at unit period.
    #[test]
    fn coord_dist_sq_torus_still_wraps_unit_period() {
        let atom = tiny_atom(SaeAtomBasisKind::Torus, 1);
        let a = Array1::from(vec![0.02f64]);
        let b = Array1::from(vec![0.98f64]);
        // Wrapped circle distance min(0.96, 0.04) = 0.04 ⇒ dist² = 0.0016.
        let dsq = coord_dist_sq(&atom, a.view(), b.view());
        assert!(
            (dsq - 0.0016).abs() < 1e-12,
            "torus unit-period wrap; got {dsq}"
        );
    }

    /// BUG (sphere chart tiling): `chart_nominal_radius` for the sphere must use
    /// the SAME capped per-axis count `SHAPE_BAND_MAX_POINTS.isqrt()` as
    /// `sphere_latlon_grid`, so the radius covers the (capped) longitude
    /// half-spacing `π/r_cap` and the charts tile without gaps for
    /// `resolution > r_cap` (raw π/40 would leave gaps). `r_cap` is derived from
    /// the band-point budget (#2071), so this test tracks it rather than pinning 22.
    #[test]
    fn chart_nominal_radius_sphere_covers_capped_grid_spacing() {
        let atom = tiny_atom(SaeAtomBasisKind::Sphere, 2);
        let r_cap = SHAPE_BAND_MAX_POINTS.isqrt(); // 22 at the current 512-point budget
        let resolution = r_cap * 2; // any resolution past the cap exercises the gap hazard
        let lon_half_spacing = std::f64::consts::PI / r_cap as f64;
        let r = chart_nominal_radius(&atom, resolution);
        assert!(
            r >= lon_half_spacing - 1e-12,
            "sphere radius {r} must cover the capped lon half-spacing {lon_half_spacing} \
             (no gaps for resolution>r_cap); raw π/{resolution}={} would gap",
            std::f64::consts::PI / resolution as f64
        );
    }

    // ---------------------------------------------------------------------
    // F1: amplitude-aware chart routing.
    // ---------------------------------------------------------------------

    /// A single certifiable chart with the given amplitude-1 center reconstruction.
    fn chart_with_recon(recon: Vec<f64>) -> CertifiedChart {
        CertifiedChart {
            region: ChartRegion::new(Array1::zeros(1), 1.0),
            lipschitz: 1.0,
            beta_center: 1.0,
            certified_radius: 1.0,
            amortized_jacobian: None,
            recon_center: Array1::from(recon),
            amortized_base: None,
        }
    }

    fn atlas_two_charts(m1: f64, m2: f64) -> AtomEncodeAtlas {
        AtomEncodeAtlas {
            atom_index: 0,
            latent_dim: 1,
            decoder_norm_sum: 1.0,
            charts: vec![chart_with_recon(vec![m1]), chart_with_recon(vec![m2])],
        }
    }

    /// F1 counterexample (module review): chart centers reconstruct (amplitude 1)
    /// to `m₁ = 1` and `m₂ = 10`. A row `x = 1` at amplitude `z = 0.1`
    /// reconstructs as `z·m`, so chart 2 is EXACT (`0.1·10 = 1 = x`) and chart 1 is
    /// wrong (`0.1·1 = 0.1`, error `0.9`). Amplitude-blind routing on the
    /// amplitude-1 centers picks chart 1 (`|1−1| = 0 < |10−1| = 9`); the
    /// amplitude-aware fix picks chart 2.
    #[test]
    fn f1_routing_scores_amplitude_scaled_reconstruction() {
        let atlas = atlas_two_charts(1.0, 10.0);
        let x = Array1::from(vec![1.0]);

        let (idx, _) = nearest_chart(&atlas, x.view(), 0.1).expect("routes");
        assert_eq!(
            idx, 1,
            "z=0.1: must route to the m=10 chart (z·m=1 exact), not m=1"
        );

        let ranked = nearest_charts_topk(&atlas, x.view(), 0.1, 2);
        assert_eq!(ranked[0], 1, "z=0.1: nearest chart is the m=10 chart");

        // Negative amplitude: z=-0.1 makes z·m₂ = -1 (error 2) and z·m₁ = -0.1
        // (error 1.1), so chart 1 is now nearer — the sign is respected.
        let (idxn, _) = nearest_chart(&atlas, x.view(), -0.1).expect("routes");
        assert_eq!(idxn, 0, "z=-0.1: sign-aware routing prefers the m=1 chart");

        // No-regression at z=1: recovers the amplitude-1 argmin (m=1 chart exact).
        let (idx1, _) = nearest_chart(&atlas, x.view(), 1.0).expect("routes");
        assert_eq!(idx1, 0, "z=1 recovers the amplitude-1 nearest (m=1) chart");
        assert_eq!(nearest_charts_topk(&atlas, x.view(), 1.0, 1)[0], 0);
    }

    // ---------------------------------------------------------------------
    // F2: the certificate uses the TRUE (un-ridged) Hessian.
    // ---------------------------------------------------------------------

    /// A basis with constant `Φ`, zero Jacobian, and zero second jet: the
    /// reconstruction is locally CONSTANT, so the true encode Hessian is `0`
    /// (singular) and the coordinate is non-unique.
    #[derive(Debug)]
    struct ConstantPhi {
        m: usize,
        d: usize,
    }
    impl SaeBasisEvaluator for ConstantPhi {
        fn evaluate(
            &self,
            coords: ndarray::ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            let n = coords.nrows();
            Ok((
                Array2::ones((n, self.m)),
                Array3::zeros((n, self.m, self.d)),
            ))
        }
        fn second_jet_dyn(
            &self,
            coords: ndarray::ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array4<f64>, String>> {
            let n = coords.nrows();
            Some(Ok(ndarray::Array4::zeros((n, self.m, self.d, self.d))))
        }
        fn third_jet_dyn(
            &self,
            coords: ndarray::ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array5<f64>, String>> {
            if coords.ncols() != self.d {
                return Some(Err(format!(
                    "ConstantPhi::third_jet_dyn: expected d = {}, got {} coords",
                    self.d,
                    coords.ncols()
                )));
            }
            None
        }
    }

    /// F2: a locally-constant reconstruction has a SINGULAR true Hessian (`H = 0`),
    /// so the point is NOT a genuine isolated minimum and must NOT be certified.
    /// The old code ridged the certified Hessian (`H → ridge·I`, PD), faking
    /// `β = 1/ridge`, `η = 0`, `h = 0 ≤ ½` — a FALSE certificate. With the true
    /// Hessian (`encode_grad_hess` no longer adds ridge) `beta_eta_newton` sees
    /// `λ_min = 0` and refuses.
    #[test]
    fn f2_certificate_uses_true_hessian_refuses_singular_field() {
        let atom = tiny_atom(SaeAtomBasisKind::EuclideanPatch, 1);
        let eval = ConstantPhi {
            m: atom.basis_size(),
            d: 1,
        };
        let t0 = Array1::from(vec![0.0]);
        let x = Array1::from(vec![0.5]);

        let (g, h) = encode_grad_hess(&atom, &eval, t0.view(), x.view(), 1.0)
            .expect("encode_grad_hess runs")
            .expect("second jet present ⇒ Some");
        assert!(
            h.iter().all(|&v| v == 0.0),
            "the TRUE Hessian of a constant reconstruction is 0 — no ridge is added \
             to the certified field; got {h:?}"
        );
        assert!(
            g.iter().all(|&v| v == 0.0),
            "gradient is 0 at a flat reconstruction"
        );

        let (cert, _) = row_certificate(&atom, &eval, t0.view(), x.view(), 1.0, 1.0)
            .expect("row_certificate runs");
        assert!(
            !cert.certified(),
            "a singular true Hessian must NOT be certified (the old ridged H falsely did)"
        );
        assert!(
            !cert.beta.is_finite(),
            "β must be ∞ (uncertifiable), never the ridge-faked 1/ridge; got {}",
            cert.beta
        );
    }

    // ---------------------------------------------------------------------
    // F3: Duchon atoms are refused (honest refusal, never a fabricated bound).
    // ---------------------------------------------------------------------

    /// F3: `build_atom_atlas_from_centers` must emit ONLY uncertified charts for a
    /// Duchon atom — the closed-form bound available here would fabricate cubic-r³
    /// jets and an origin center for a polyharmonic `c·r^(2m−d)` kernel over
    /// data-placed centers, risking an under-estimate of `L` (false certificate).
    /// The refusal routes every Duchon row to the exact multi-start encode. A
    /// non-Duchon control atom is NOT auto-zeroed by this guard.
    #[test]
    fn f3_duchon_atoms_are_uncertifiable() {
        let atom = tiny_atom(SaeAtomBasisKind::Duchon, 1);
        let centers = ndarray::array![[0.0_f64], [0.3], [0.7]];
        let radii = vec![0.1_f64, 0.1, 0.1];
        let atlas = EncodeAtlas::build_atom_atlas_from_centers(
            0,
            &atom,
            centers.view(),
            &radii,
            1.0,
            1.0,
            &AtlasConfig::default(),
        )
        .expect("duchon atlas builds (uncertified)");
        assert_eq!(atlas.charts.len(), 3, "one chart per center");
        for (i, chart) in atlas.charts.iter().enumerate() {
            assert_eq!(
                chart.certified_radius, 0.0,
                "duchon chart {i} must be uncertified (refused), got r={}",
                chart.certified_radius
            );
            assert!(
                chart.amortized_jacobian.is_none(),
                "duchon chart {i} must carry no amortized predictor"
            );
        }
    }

    // ---------------------------------------------------------------------
    // F6: an already-certified warm-up step is never rejected by the progress rule.
    // ---------------------------------------------------------------------

    /// F6: a step that just crossed into the certified region (`h ≤ ½`) is accepted
    /// even when its multiplicative decrease was below the progress floor
    /// (`0.501 → 0.499`). Only a STILL-uncertified plateau step is rejected. The
    /// old unconditional progress test rejected the `0.501 → 0.499` cross — a false
    /// negative that forced an unnecessary exact-solve fallback.
    #[test]
    fn f6_certified_step_not_rejected_by_progress_rule() {
        // 0.501 → 0.499: below the 1/64 multiplicative floor, and NOT the quadratic
        // path, so `warmup_progress_sufficient` is false…
        assert!(
            !warmup_progress_sufficient(0.499, 0.501),
            "precondition: this tiny decrease fails the progress bar"
        );
        // …but because the step is now CERTIFIED it must NOT be rejected (F6).
        assert!(
            !warmup_should_reject(true, 0.499, 0.501),
            "a certified step must be accepted regardless of the progress floor"
        );
        // A still-uncertified plateau step IS rejected (the guard still bites).
        assert!(
            warmup_should_reject(false, 0.499, 0.501),
            "an uncertified sub-floor step is a plateau and must flag to fallback"
        );
        // A certified step that ALSO made big progress is accepted (sanity).
        assert!(!warmup_should_reject(true, 0.1, 0.9));
        // A genuinely-contracting uncertified step is accepted (no regression).
        assert!(!warmup_should_reject(false, 0.4, 0.9));
    }

    // A singular Hessian is not invertible as the derivative F'(t). Replacing
    // its null eigenvalue by an arbitrary stiffness describes a different map,
    // so it must never produce an ordinary Kantorovich certificate.
    #[test]
    fn beta_eta_newton_refuses_rank1_null_2x2() {
        let h = ndarray::array![[4.0_f64, 0.0], [0.0, 0.0]];
        let g = Array1::from(vec![4.0_f64, 0.0]);
        assert!(beta_eta_newton(h.view(), g.view()).expect("runs").is_none());
    }

    /// Refusal is independent of whether the observed gradient happens to have a
    /// small projection onto the null direction: that observation cannot turn a
    /// singular derivative into the derivative required by the theorem.
    #[test]
    fn beta_eta_newton_refuses_null_with_projected_gradient() {
        let h = ndarray::array![[4.0_f64, 0.0], [0.0, 0.0]];
        let g = Array1::from(vec![4.0_f64, 1.0e-3]);
        assert!(beta_eta_newton(h.view(), g.view()).expect("runs").is_none());
    }

    /// A genuinely INDEFINITE 2×2 (one negative eigenvalue) is STILL refused —
    /// deflation must not manufacture a certificate at a saddle/max of the encode
    /// objective (the pre-existing negative-curvature guard is preserved).
    #[test]
    fn beta_eta_newton_refuses_genuine_negative_curvature_2x2() {
        let h = ndarray::array![[4.0_f64, 0.0], [0.0, -2.0]];
        let g = Array1::from(vec![1.0_f64, 1.0]);
        let out = beta_eta_newton(h.view(), g.view()).expect("runs");
        assert!(
            out.is_none(),
            "a negative-curvature (indefinite) start must NOT certify — it is at/past a \
             basin boundary; deflating it to +1 would be a false certificate"
        );
    }

    /// The general eigen path applies the same refusal to a rank-deficient PSD
    /// block instead of silently changing its spectrum.
    #[test]
    fn beta_eta_newton_refuses_rank_deficient_3x3() {
        let h = ndarray::array![[4.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let g = Array1::from(vec![4.0_f64, 1.0, 0.0]);
        assert!(beta_eta_newton(h.view(), g.view()).expect("runs").is_none());
    }

    /// A healthy positive-definite 2×2 is UNCHANGED by the fix: it keeps the
    /// closed-form fast path, β = 1/λ_min, and the Newton step solves H·δ = −g.
    #[test]
    fn beta_eta_newton_healthy_pd_unchanged() {
        let h = ndarray::array![[4.0_f64, 1.0], [1.0, 3.0]];
        let g = Array1::from(vec![2.0_f64, -1.0]);
        let (beta, eta, delta) = beta_eta_newton(h.view(), g.view())
            .expect("runs")
            .expect("a PD block certifies");
        // λ_min = ½(7 − √5); β = 1/λ_min.
        let lambda_min = 0.5 * (7.0 - 5.0_f64.sqrt());
        assert!((beta - 1.0 / lambda_min).abs() < 1e-9, "β={beta}");
        // Newton normal equations: H·δ = −g.
        let hd0 = h[[0, 0]] * delta[0] + h[[0, 1]] * delta[1];
        let hd1 = h[[1, 0]] * delta[0] + h[[1, 1]] * delta[1];
        assert!((hd0 + g[0]).abs() < 1e-9 && (hd1 + g[1]).abs() < 1e-9);
        assert!((eta - delta.dot(&delta).sqrt()).abs() < 1e-12);
    }
}

#[cfg(test)]
mod joint_fallback_tests {
    //! Reviewer condition #3 — the multi-start-fallback fraction is an honest
    //! encode-tax number that GROWS with atom-image similarity / co-activation
    //! interference. These tests drive `joint_encode_fallback_fraction` over a
    //! similarity sweep on flat (linear) atoms, where the Gauss–Newton curvature
    //! block is EXACT (no residual-curvature term), so the Gershgorin dominance
    //! decision is the true joint certificate and the curve is analytic.
    use super::*;
    use crate::manifold::SaeAtomBasisKind;
    use ndarray::Array2;
    use std::sync::Arc;

    /// Deterministic LCG (no `rand` dependency) for reproducible amplitudes.
    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// Build `k` degree-1 (flat) atoms in `R^p` whose image TANGENT directions
    /// have common pairwise cosine `rho`: `dir_k = √rho · e0 + √(1−rho) · e_{k+1}`
    /// with `{e0, e1, …}` orthonormal. Requires `p ≥ k + 1`. Each atom's decoder
    /// puts the tangent on the degree-1 monomial row, so `∂m/∂t = dir_k` exactly.
    fn similar_linear_atoms(k: usize, p: usize, rho: f64) -> Vec<SaeManifoldAtom> {
        assert!(p >= k + 1, "need p >= k+1 orthonormal directions");
        let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("degree-1 patch"));
        // A single coordinate row is enough to pin the (constant) tangent; the
        // diagnostic re-evaluates the jet at whatever coords it is given.
        let coord = Array2::<f64>::zeros((1, 1));
        let (phi, jet) = evaluator.evaluate(coord.view()).expect("evaluate");
        let m = phi.ncols();
        (0..k)
            .map(|atom_idx| {
                let mut dec = Array2::<f64>::zeros((m, p));
                // Row 1 is the degree-1 monomial (`∂φ/∂t = 1` there); place the
                // unit tangent direction on it.
                dec[[1, 0]] = rho.sqrt();
                dec[[1, atom_idx + 1]] = (1.0 - rho).sqrt();
                SaeManifoldAtom::new_with_provided_function_gram(
                    "lin",
                    SaeAtomBasisKind::EuclideanPatch,
                    1,
                    phi.clone(),
                    jet.clone(),
                    dec,
                    Array2::<f64>::eye(m),
                )
                .expect("atom builds")
                .with_basis_second_jet(evaluator.clone())
            })
            .collect()
    }

    /// The joint multi-start-fallback fraction is MONOTONE NON-DECREASING in
    /// atom-image similarity, is exactly zero when the atoms are orthogonal, and
    /// is strictly positive once the images are strongly aligned — the honest
    /// encode-tax cost multiplier the reviewer asks for.
    #[test]
    fn joint_fallback_fraction_rises_with_atom_similarity() {
        let k = 4usize;
        let p = 8usize;
        let n = 400usize;
        // Per-row amplitudes: a spread of masses so that at intermediate
        // similarity SOME rows (those with an atom whose mass is dominated by its
        // co-active neighbours) tip out of Gershgorin dominance while others stay
        // in — a smooth curve rather than a step.
        let mut rng = Lcg(42);
        let amplitudes = Array2::from_shape_fn((n, k), |_| 0.2 + 1.3 * rng.unit());
        // Coordinates are irrelevant for flat atoms (constant tangent), but the
        // diagnostic still evaluates the jet at them.
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let floor = 1.0e-9;

        let sweep = [0.0_f64, 0.2, 0.4, 0.6, 0.8, 0.95];
        let mut fractions = Vec::new();
        for &rho in &sweep {
            let atoms = similar_linear_atoms(k, p, rho);
            let frac = joint_encode_fallback_fraction(&atoms, &coords, amplitudes.view(), floor)
                .expect("joint fallback fraction computes");
            assert!(
                (0.0..=1.0).contains(&frac),
                "fallback fraction must be a probability, got {frac} at rho={rho}"
            );
            fractions.push(frac);
        }
        eprintln!(
            "[ENCODE-FALLBACK-SWEEP] similarity rho={:?} -> joint multistart fraction={:?}",
            sweep, fractions
        );
        // Orthogonal atoms: the block-diagonal per-atom certificates compose, no
        // row needs multi-start.
        assert!(
            fractions[0] == 0.0,
            "orthogonal atoms must need no multi-start fallback, got {}",
            fractions[0]
        );
        // Monotone non-decreasing across the similarity sweep.
        for w in fractions.windows(2) {
            assert!(
                w[1] >= w[0] - 1.0e-12,
                "fallback fraction must not DECREASE as similarity rises: {:?}",
                fractions
            );
        }
        // Strongly-aligned images force a materially higher fallback fraction —
        // the effect is real, not a rounding wobble.
        assert!(
            *fractions.last().unwrap() > 0.25,
            "strong atom similarity must drive a substantial multi-start tail; curve={fractions:?}"
        );
    }

    /// A single co-active atom per row has no cross blocks, so the joint fallback
    /// fraction is zero regardless of how curved or ill-conditioned the atom is —
    /// the joint tax is purely a CO-ACTIVATION phenomenon.
    #[test]
    fn joint_fallback_zero_without_coactivation() {
        let k = 3usize;
        let p = 8usize;
        let n = 50usize;
        let atoms = similar_linear_atoms(k, p, 0.9); // highly similar, but…
        // …only ONE atom active per row (block-diagonal one-hot amplitudes).
        let mut amplitudes = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            amplitudes[[row, row % k]] = 1.0;
        }
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let frac = joint_encode_fallback_fraction(&atoms, &coords, amplitudes.view(), 1.0e-9)
            .expect("computes");
        assert!(
            frac == 0.0,
            "no co-activation ⇒ no joint fallback, got {frac}"
        );
    }

    /// The per-atom fallback tiers PARTITION the (row, atom) grid and the tier
    /// fractions are consistent probabilities; `accumulate` folds per-atom
    /// telemetry into a dictionary-wide breakdown that preserves the partition.
    #[test]
    fn fallback_telemetry_tiers_partition_and_accumulate() {
        let a = FallbackTelemetry {
            n_rows: 100,
            n_atoms: 1,
            amortized_certified: 70,
            newton_rescued: 20,
            multistart_fallback: 10,
        };
        assert_eq!(
            a.amortized_certified + a.newton_rescued + a.multistart_fallback,
            a.total(),
            "the three tiers must partition the (row, atom) grid"
        );
        assert!((a.amortized_fraction() - 0.70).abs() < 1e-12);
        assert!((a.newton_fraction() - 0.20).abs() < 1e-12);
        assert!((a.multistart_fraction() - 0.10).abs() < 1e-12);
        assert!(
            (a.amortized_fraction() + a.newton_fraction() + a.multistart_fraction() - 1.0).abs()
                < 1e-12,
            "the tier fractions must sum to one"
        );

        let b = FallbackTelemetry {
            n_rows: 100,
            n_atoms: 1,
            amortized_certified: 40,
            newton_rescued: 30,
            multistart_fallback: 30,
        };
        let mut agg = a.clone();
        agg.accumulate(&b);
        assert_eq!(agg.n_rows, 100, "n_rows is shared across atoms");
        assert_eq!(agg.n_atoms, 2, "accumulate sums the atom count");
        assert_eq!(agg.multistart_fallback, 40);
        assert_eq!(
            agg.amortized_certified + agg.newton_rescued + agg.multistart_fallback,
            agg.total(),
            "the partition is preserved under accumulation"
        );
    }
}

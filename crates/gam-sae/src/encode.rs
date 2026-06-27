//! Kantorovich-certified encode atlas (issue #1010).
//!
//! Encoding a row `x ∈ ℝᵖ` against a FROZEN SAE dictionary is, per atom `k`,
//! the coordinate-only Newton problem
//!
//! ```text
//! min_t  f_k(t) = ½‖x − z_k · B_kᵀ Φ_k(t)‖² + prior_k(t),
//! ```
//!
//! with the amplitude `z_k` and decoder block `B_k` held fixed (the encode
//! freezes the dictionary; only the latent coordinate `t` moves). Newton on
//! `F(t) = ∇f_k(t)` converges quadratically from a start `t₀` into the unique
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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use gam_linalg::faer_ndarray::FaerEigh;
use crate::candidate_index::{
    AtomFrameSketch, SaeCandidateIndex, auto_candidate_budget,
};
use crate::manifold::{
    AffineCoordinateEvaluator, CylinderHarmonicEvaluator, DuchonCoordinateEvaluator,
    EuclideanPatchEvaluator, PeriodicHarmonicEvaluator, SaeBasisEvaluator, SaeManifoldAtom,
    SphereChartEvaluator, TorusHarmonicEvaluator,
};

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

/// Minimum frame alignment `‖Uₖᵀd‖/‖d‖ ∈ [0,1]` the best LSH-gathered atom must
/// have for an index-routed encode to be TRUSTED (#1026). The in-atom Kantorovich
/// certificate attests Newton convergence *within* the chosen atom; it does NOT
/// attest that the LSH gather found the globally-correct atom. Because alignment
/// is capped at 1, a HIGH best-alignment leaves no room for a materially-better
/// ungathered atom (the gather is trustworthy), whereas a LOW best-alignment means
/// the true best atom was likely missed by LSH recall. Rows whose best gathered
/// atom aligns below this are flagged uncertified and routed to the exact
/// full-scan fallback — closing the silent-mis-routing hole where a non-empty but
/// wrong gather certified against a suboptimal atom. Erring toward flagging only
/// ever adds exact-fallback work (correct, slower); it never certifies a mis-route.
pub(crate) const CANDIDATE_ROUTING_MIN_ALIGNMENT: f64 = 0.5;

/// Bounded plain-Newton warm-up steps the certified encoder takes from an
/// uncertified chart-center start to navigate INTO the Kantorovich basin before
/// flagging the row for the exact fallback (#1154/#1026). The Kantorovich
/// quantity `h = β·η·L` scales with amplitude through `L`, so at unit amplitude a
/// chart-center start can be positive-definite yet have `h > ½`; a few Newton
/// steps reach an iterate where `h ≤ ½` and the encode is certified-exact from
/// there. Bounded so a genuinely non-convergent row still falls back quickly; each
/// step is one grad/Hessian eval plus a `d×d` solve (`d` = latent dim).
pub(crate) const SAE_ENCODE_BASIN_WARMUP_STEPS: usize = 8;

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
            big_d.powi(k1 as i32) * rho.powi(residual)
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
    d.powi(order as i32) * big_d.powi(order as i32) * rho.powi(residual_degree)
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
    if matches!(
        atom.basis_kind,
        crate::manifold::SaeAtomBasisKind::Periodic
    ) {
        periodic_reconstruction_jet_sups(atom.decoder_coefficients.view())
    } else {
        let decoder_norm_sum = decoder_row_norm_sum(atom.decoder_coefficients.view());
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
    initial_cert: RowCertificate,
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
pub(crate) fn family_jet_sups(
    atom: &SaeManifoldAtom,
    chart: &ChartRegion,
) -> Result<JetSups, String> {
    use crate::manifold::SaeAtomBasisKind::*;
    let m = atom.basis_size();
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
            // The atom carries the basis kind but not the nullspace order, and
            // the certificate needs an UPPER bound on L. The kernel-tail bound
            // (cubic r³ coefficients vs the chart's r_min/r_max) is independent
            // of the constructed order; the polynomial-block bound grows with the
            // order, so we construct with a conservative order whose polynomial
            // degree upper-bounds any nullspace the atom's basis width can hold.
            // Constructing with `m = basis_size` maps to `Degree(basis_size − 1)`
            // — an over-estimate that keeps the Lipschitz bound sound.
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
pub(crate) fn encode_grad_hess(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    t: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    ridge: f64,
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
    // g_t[axis] = J_m[axis] · r ;  H_tt[a,b] = J_m[a]·J_m[b] + r·∂²m/∂t_a∂t_b.
    let mut g = Array1::<f64>::zeros(d);
    let mut h = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        let ja = jm.row(a);
        g[a] = ja.dot(&residual);
        for b in 0..d {
            // Gauss-Newton block.
            let mut hab = ja.dot(&jm.row(b));
            // Residual · second-jet curvature: r · ∂²m_{ab},
            // ∂²m_{ab}[out] = z · Σ_basis (∂²Φ/∂t_a∂t_b) · B[basis, out].
            let mut curv = 0.0;
            for basis_col in 0..m {
                let d2phi = second[[0, basis_col, a, b]];
                if d2phi == 0.0 {
                    continue;
                }
                let mut dot = 0.0;
                for out in 0..p {
                    dot += residual[out] * decoder[[basis_col, out]];
                }
                curv += amplitude * d2phi * dot;
            }
            hab += curv;
            h[[a, b]] = hab;
        }
    }
    for a in 0..d {
        h[[a, a]] += ridge;
    }
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
    let (vals, vecs) = h
        .eigh(Side::Lower)
        .map_err(|e| format!("beta_eta_newton: eigh failed: {e:?}"))?;
    let lambda_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    if !(lambda_min.is_finite() && lambda_min > 0.0) {
        return Ok(None);
    }
    let beta = 1.0 / lambda_min;
    // Newton step δ = −H⁻¹ g via the eigendecomposition: δ = −Σ_i (vᵢᵀg/λᵢ) vᵢ.
    let d = h.nrows();
    let mut delta = Array1::<f64>::zeros(d);
    for (col, &lam) in vals.iter().enumerate() {
        if lam <= 0.0 {
            return Ok(None);
        }
        let vi = vecs.column(col);
        let coeff = vi.dot(&g) / lam;
        for row in 0..d {
            delta[row] -= coeff * vi[row];
        }
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
    ridge: f64,
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
    let Some((g, h)) = encode_grad_hess(atom, evaluator, t0, x, amplitude, ridge)? else {
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
    ridge: f64,
    newton_steps: usize,
    initial_cert: RowCertificate,
    mut delta: Array1<f64>,
) -> Result<Option<CertifiedEncodeProbe>, String> {
    assert!(initial_cert.certified());
    let mut final_cert = initial_cert;
    for _ in 0..newton_steps {
        t = &t + &delta;
        let (cert, next_delta) =
            row_certificate(atom, evaluator, t.view(), x, amplitude, lipschitz, ridge)?;
        if !cert.certified() {
            return Ok(None);
        }
        final_cert = cert;
        delta = next_delta;
    }
    Ok(Some(CertifiedEncodeProbe {
        coord: t,
        initial_cert,
        final_cert,
    }))
}

/// Certify an encode probe from `t_start`, navigating into the Kantorovich basin
/// first if needed (#1154/#1026). The Kantorovich quantity `h = β·η·L` scales with
/// amplitude through `L`, so at unit amplitude a positive-definite chart-center /
/// distilled start can sit OUTSIDE the certified ball (`h > ½`). Rather than
/// flagging it uncertified immediately — which made the encoder certify ZERO
/// held-out rows at amplitude 1.0 and fall back to the exact solve for everything —
/// take up to [`SAE_ENCODE_BASIN_WARMUP_STEPS`] plain Newton steps toward the root,
/// re-certifying at each iterate. The certificate at the landing point is a full
/// Kantorovich guarantee from there (`h ≤ ½` ⇒ Newton converges to the in-ball
/// root), so this only ever WIDENS the certified set; it never certifies a
/// non-convergent start. An indefinite Hessian (`β`/`η` non-finite — at/past a
/// basin boundary) is not steppable, so we stop and return `None` (flag). On
/// success the start is then refined `newton_steps` further by
/// [`refine_certified_start`].
fn certify_with_basin_warmup(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    t_start: Array1<f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    lipschitz: f64,
    ridge: f64,
    newton_steps: usize,
) -> Result<Option<CertifiedEncodeProbe>, String> {
    let mut t = t_start;
    let (mut cert, mut delta) =
        row_certificate(atom, evaluator, t.view(), x, amplitude, lipschitz, ridge)?;
    let mut warmups = 0usize;
    while !cert.certified() {
        if warmups >= SAE_ENCODE_BASIN_WARMUP_STEPS
            || !(cert.beta.is_finite() && cert.eta.is_finite())
        {
            return Ok(None);
        }
        t = &t + &delta;
        let (next_cert, next_delta) =
            row_certificate(atom, evaluator, t.view(), x, amplitude, lipschitz, ridge)?;
        cert = next_cert;
        delta = next_delta;
        warmups += 1;
    }
    refine_certified_start(
        atom,
        evaluator,
        t,
        x,
        amplitude,
        lipschitz,
        ridge,
        newton_steps,
        cert,
        delta,
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
        let d = atom.latent_dim;
        let decoder_norm_sum = decoder_row_norm_sum(atom.decoder_coefficients.view());
        let centers = chart_center_grid(atom, config.grid_resolution);
        // Half the inter-center spacing is the natural in-chart radius so the
        // charts tile the grid without gaps; refined below if the certificate
        // fails at that radius.
        let nominal_radius = chart_nominal_radius(atom, config.grid_resolution);
        let mut charts = Vec::with_capacity(centers.nrows());
        for c in 0..centers.nrows() {
            let center = centers.row(c).to_owned();
            let region = chart_region(atom, center.clone(), nominal_radius);
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
            charts.push(CertifiedChart {
                region,
                lipschitz,
                beta_center,
                certified_radius,
                amortized_jacobian,
                recon_center,
            });
        }
        Ok(AtomEncodeAtlas {
            atom_index,
            latent_dim: d,
            decoder_norm_sum,
            charts,
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
    ) -> Result<(Array1<f64>, RowCertificate), String> {
        // Certify from the warm start, navigating into the Kantorovich basin first
        // if the unit-amplitude start has h > ½ (see `certify_with_basin_warmup`).
        let Some(probe) = certify_with_basin_warmup(
            atom,
            evaluator,
            t,
            x,
            amplitude,
            chart.lipschitz,
            self.config.ridge,
            self.config.newton_steps,
        )?
        else {
            return Ok((
                Array1::<f64>::zeros(atom.latent_dim),
                uncertified_certificate(chart.lipschitz),
            ));
        };
        Ok((probe.coord, probe.initial_cert))
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
        let atom_atlas = self
            .atoms
            .get(atom_index)
            .ok_or_else(|| format!("certified_encode_row: atom {atom_index} not in atlas"))?;
        let d = atom.latent_dim;
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

        // Route to the nearest chart center (ambient routing happens upstream via
        // sae_candidate_index; here we pick the in-atom chart). Without a chart we
        // cannot certify — flag immediately.
        let Some((chart_idx, _)) = nearest_chart(atom_atlas, x, atom, evaluator.as_ref()) else {
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
        let chart = &atom_atlas.charts[chart_idx];
        let Some(t) = amortized_warm_start(chart, x, amplitude) else {
            return Ok((
                Array1::<f64>::zeros(d),
                uncertified_certificate(chart.lipschitz),
            ));
        };
        self.refine_certified_encode_start(atom, evaluator.as_ref(), chart, t, x, amplitude)
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
        let Some((chart_idx, _)) = nearest_chart(atom_atlas, x, atom, evaluator.as_ref()) else {
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
            chart.lipschitz,
            self.config.ridge,
            self.config.newton_steps,
        )?
        else {
            return Ok((
                Array1::<f64>::zeros(d),
                uncertified_certificate(chart.lipschitz),
            ));
        };

        let cold_start = chart.region.center.clone();
        let Some(cold_probe) = certify_with_basin_warmup(
            atom,
            evaluator.as_ref(),
            cold_start,
            x,
            amplitude,
            chart.lipschitz,
            self.config.ridge,
            self.config.newton_steps,
        )?
        else {
            return Ok((
                amortized_probe.coord,
                uncertified_certificate(chart.lipschitz),
            ));
        };

        let gap =
            latent_coordinate_distance(atom, amortized_probe.coord.view(), cold_probe.coord.view());
        let tolerance = distilled_probe_tolerance(&amortized_probe, &cold_probe, amplitude, x);
        if !(gap.is_finite() && gap <= tolerance) {
            return Ok((
                amortized_probe.coord,
                uncertified_certificate(chart.lipschitz),
            ));
        }
        Ok((amortized_probe.coord, amortized_probe.initial_cert))
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

    /// LSH-routed certified encode (issue #1010 step 2 + 3): for each target
    /// row, the existing [`SaeCandidateIndex`] (#985/#994) proposes the
    /// best-aligned atom by frame alignment to the row direction; the row is then
    /// encoded against THAT atom's certified chart atlas. This is the production
    /// routing path — the LSH does sublinear atom selection, the atlas does the
    /// in-atom nearest-chart routing and the per-row Kantorovich certificate.
    ///
    /// `atoms[id]` must be aligned with the atlas's `atoms[id]` (same dictionary
    /// order the atlas was built from and the sketch/index were built over).
    /// A row with no LSH proposal (empty bucket) is flagged uncertified — it
    /// routes to the exact multi-start fallback, never a silent wrong encode.
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
                        // The row direction is the (unit-tolerant) target; the LSH
                        // ranks atoms by how much of that direction lies in each
                        // atom's column space. `propose` returns the top-`budget`
                        // atom ids by exact frame alignment.
                        let proposal = index.propose(sketch, targets.row(row), budget, true);
                        let Some(&best_atom) = proposal.proposed.first() else {
                            // No LSH candidate: flag for the exact fallback.
                            return Ok(None);
                        };
                        // Routing-confidence gate (#1026): the in-atom certificate
                        // attests convergence WITHIN best_atom, not that best_atom is
                        // the globally-correct atom. A non-empty but low-alignment
                        // gather likely missed the true best atom (LSH recall < 1) —
                        // flag it for the exact fallback rather than silently
                        // certifying a mis-route. See CANDIDATE_ROUTING_MIN_ALIGNMENT.
                        if sketch.alignment(best_atom, targets.row(row))
                            < CANDIDATE_ROUTING_MIN_ALIGNMENT
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
                        let proposal = index.propose(sketch, targets.row(row), budget, true);
                        let Some(&best_atom) = proposal.proposed.first() else {
                            return Ok(None);
                        };
                        // Routing-confidence gate (#1026): flag low-alignment gathers
                        // for the exact fallback so a missed-true-best LSH route is
                        // never silently certified. See CANDIDATE_ROUTING_MIN_ALIGNMENT
                        // and certified_encode_with_index for the full rationale.
                        if sketch.alignment(best_atom, targets.row(row))
                            < CANDIDATE_ROUTING_MIN_ALIGNMENT
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
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
) -> Option<(usize, f64)> {
    if atom_atlas.charts.is_empty() {
        return None;
    }
    let d = atom.latent_dim;
    let p = atom.output_dim();
    let m = atom.basis_size();
    let mut best: Option<(usize, f64)> = None;
    for (idx, chart) in atom_atlas.charts.iter().enumerate() {
        if chart.certified_radius <= 0.0 {
            continue;
        }
        let coords = match chart.region.center.view().to_shape((1, d)) {
            Ok(c) => c.to_owned(),
            Err(_) => continue,
        };
        let Ok((phi, _jet)) = evaluator.evaluate(coords.view()) else {
            continue;
        };
        // m(t_c) = Bᵀ Φ(t_c) (amplitude-1; routing is scale-tolerant).
        let mut recon = Array1::<f64>::zeros(p);
        for basis_col in 0..m {
            let phi_v = phi[[0, basis_col]];
            if phi_v == 0.0 {
                continue;
            }
            for out in 0..p {
                recon[out] += phi_v * atom.decoder_coefficients[[basis_col, out]];
            }
        }
        let diff = &recon - &x;
        let dist = diff.dot(&diff);
        if best.map(|(_, b)| dist < b).unwrap_or(true) {
            best = Some((idx, dist));
        }
    }
    best
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
        Sphere if d == 2 => sphere_latlon_grid(resolution),
        Linear | Sphere | Duchon | EuclideanPatch | Poincare | Precomputed(_) => {
            // Unbounded / non-compact latents: a strided cover of a unit box
            // about the origin per axis. The certified radius refines each chart;
            // out-of-cover starts route to the exact fallback honestly.
            regular_product_grid(d, resolution, -0.5, 0.5, true)
        }
    }
}

/// A regular `resolution`-per-axis product grid over `[lo, hi]^d`, capped at
/// [`SHAPE_BAND_MAX_POINTS`] total points (the per-axis resolution is reduced
/// until the product fits). When `include_endpoint` the last grid point sits at
/// `hi`; otherwise the axis is treated as periodic and stops one step short.
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
    let mut per_axis = resolution.max(2);
    while per_axis.saturating_pow(d as u32) > SHAPE_BAND_MAX_POINTS && per_axis > 2 {
        per_axis -= 1;
    }
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
    let r = resolution.max(2).min(22); // 22² = 484 ≤ SHAPE_BAND_MAX_POINTS.
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

/// Nominal in-chart radius: half the inter-center grid spacing, so charts tile
/// the domain. For compact latents this is the grid step; for unbounded latents
/// a unit default that the certified radius refines.
pub(crate) fn chart_nominal_radius(atom: &SaeManifoldAtom, resolution: usize) -> f64 {
    use crate::manifold::SaeAtomBasisKind::*;
    match &atom.basis_kind {
        Periodic | Torus => 0.5 / (resolution.max(2) as f64),
        Sphere => std::f64::consts::PI / (resolution.max(2) as f64),
        // Cylinder charts tile two heterogeneous axes (a `[0,1)` periodic step
        // and a unit-box line step); the chart radius is a single scalar, so we
        // take the tighter (periodic) step `0.5/res` to keep every chart valid
        // on both axes. The certified Kantorovich radius refines it per chart.
        Cylinder => 0.5 / (resolution.max(2) as f64),
        Linear | Duchon | EuclideanPatch | Poincare | Precomputed(_) => {
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
        Periodic | Sphere | Torus | Cylinder | Linear | EuclideanPatch | Poincare
        | Precomputed(_) => region,
    }
}

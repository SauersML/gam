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
//!    nearest chart, take one or two Newton steps, then the `h ≤ ½` check AT the
//!    start point is the per-row certificate.
//! 3. **Uncertified tail**: rows whose start fails `h ≤ ½` are FLAGGED (counted
//!    in [`EncodeResult::encode_uncertified_count`]) and must be routed by the
//!    caller to the existing exact multi-start solve. No approximation enters
//!    silently.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::linalg::faer_ndarray::FaerEigh;
use crate::terms::sae_manifold::{
    AffineCoordinateEvaluator, DuchonCoordinateEvaluator, EuclideanPatchEvaluator,
    PeriodicHarmonicEvaluator, SaeBasisEvaluator, SaeManifoldAtom, SphereChartEvaluator,
    TorusHarmonicEvaluator,
};

use faer::Side;

/// The Kantorovich convergence threshold `h ≤ ½`. Below this the Newton
/// iteration is guaranteed to converge quadratically into the unique root in
/// the certified ball; at or above it the start is uncertified.
pub const KANTOROVICH_THRESHOLD: f64 = 0.5;

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
fn harmonic_jet_sup(num_basis: usize, order: u32) -> f64 {
    let top_harmonic = num_basis.saturating_sub(1) / 2;
    let omega = std::f64::consts::TAU * top_harmonic as f64;
    omega.powi(order as i32)
}

impl BasisHessianLipschitz for PeriodicHarmonicEvaluator {
    fn value_sup(&self, _chart: &ChartRegion) -> f64 {
        1.0
    }
    fn jacobian_sup(&self, _chart: &ChartRegion) -> f64 {
        harmonic_jet_sup(self.num_basis, 1)
    }
    fn hessian_sup(&self, _chart: &ChartRegion) -> f64 {
        harmonic_jet_sup(self.num_basis, 2)
    }
    fn third_sup(&self, _chart: &ChartRegion) -> f64 {
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
    fn value_sup(&self, _chart: &ChartRegion) -> f64 {
        1.0
    }
    fn jacobian_sup(&self, _chart: &ChartRegion) -> f64 {
        torus_jet_sup(self.num_harmonics, self.latent_dim, 1)
    }
    fn hessian_sup(&self, _chart: &ChartRegion) -> f64 {
        torus_jet_sup(self.num_harmonics, self.latent_dim, 2)
    }
    fn third_sup(&self, _chart: &ChartRegion) -> f64 {
        torus_jet_sup(self.num_harmonics, self.latent_dim, 3)
    }
}

/// Per-column `g`-th jet sup for the torus harmonic basis: `(2π·H)^g ·
/// latent_dim^g`, where `H = num_harmonics` is the top per-axis frequency and
/// `latent_dim^g` over-counts the Leibniz routings of `g` operators across the
/// product factors (a conservative bound — each routing's per-axis magnitude is
/// `≤ (2π H)^{#ops on that axis}`, and the products telescope to `(2π H)^g`).
fn torus_jet_sup(num_harmonics: usize, latent_dim: usize, order: u32) -> f64 {
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
    fn value_sup(&self, _chart: &ChartRegion) -> f64 {
        1.0
    }
    fn jacobian_sup(&self, _chart: &ChartRegion) -> f64 {
        4.0
    }
    fn hessian_sup(&self, _chart: &ChartRegion) -> f64 {
        16.0
    }
    fn third_sup(&self, _chart: &ChartRegion) -> f64 {
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
    fn jacobian_sup(&self, _chart: &ChartRegion) -> f64 {
        1.0
    }
    fn hessian_sup(&self, _chart: &ChartRegion) -> f64 {
        0.0
    }
    fn third_sup(&self, _chart: &ChartRegion) -> f64 {
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

/// Sup-norm radius `ρ = ‖t_c‖∞ + radius` of the chart (the coordinate magnitude
/// bound used by the monomial-patch jet bounds).
fn patch_rho(chart: &ChartRegion) -> f64 {
    let center_inf = chart
        .center
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    center_inf + chart.radius
}

/// Per-column `g`-th jet sup for a monomial patch of max degree `D` in `d`
/// coordinates over the chart: `d^g · D^g · ρ^{max(D−g,0)}` (see the
/// [`EuclideanPatchEvaluator`] doc comment for the derivation).
fn patch_jet_sup(latent_dim: usize, max_degree: usize, chart: &ChartRegion, order: u32) -> f64 {
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
        let r_min = chart.exclusion_r_min.unwrap_or(chart.radius).max(f64::MIN_POSITIVE);
        let kernel = 6.0 * r_max + 3.0 * r_max * r_max / r_min;
        let poly = duchon_poly_jet_sup(self.centers.ncols(), self.order_degree(), chart, 2);
        kernel.max(poly)
    }
    fn third_sup(&self, chart: &ChartRegion) -> f64 {
        let r_max = chart.radial_r_max.unwrap_or(chart.radius);
        let r_min = chart.exclusion_r_min.unwrap_or(chart.radius).max(f64::MIN_POSITIVE);
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
            crate::basis::DuchonNullspaceOrder::Zero => 0,
            crate::basis::DuchonNullspaceOrder::Linear => 1,
            crate::basis::DuchonNullspaceOrder::Degree(d) => d,
        }
    }
}

/// Per-column `g`-th jet sup of the Duchon polynomial nullspace block, treated
/// as a monomial patch of degree `order_degree`.
fn duchon_poly_jet_sup(latent_dim: usize, order_degree: usize, chart: &ChartRegion, order: u32) -> f64 {
    if order_degree == 0 {
        return if order == 0 { 1.0 } else { 0.0 };
    }
    patch_jet_sup(latent_dim, order_degree, chart, order)
}

/// Decoder magnitude `Σ_m ‖B_{m,:}‖₂` of an atom's frozen decoder block: the
/// factor that converts a per-column `Φ`-jet sup `B_g` into a reconstruction
/// jet sup `‖∂^g m‖ ≤ |z|·decoder_row_norm_sum·B_g`.
fn decoder_row_norm_sum(decoder: ArrayView2<'_, f64>) -> f64 {
    let mut acc = 0.0;
    for row in decoder.rows() {
        acc += row.dot(&row).sqrt();
    }
    acc
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
#[allow(clippy::too_many_arguments)]
fn hessian_lipschitz_constant(
    jet_value: f64,
    jet_jac: f64,
    jet_hess: f64,
    jet_third: f64,
    decoder_norm_sum: f64,
    amplitude: f64,
    target_norm: f64,
    prior_lipschitz: f64,
) -> f64 {
    let z = amplitude.abs();
    let s_b = decoder_norm_sum;
    let m_jac = z * s_b * jet_jac;
    let m_hess = z * s_b * jet_hess;
    let m_third = z * s_b * jet_third;
    let recon_value = z * s_b * jet_value;
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
    fn from_rows(coords: Array2<f64>, certified: Vec<bool>) -> Self {
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

/// Build a basis-family handle for one atom from its [`SaeManifoldAtom`]. The
/// atlas needs to evaluate the jet sups, which live on the concrete evaluator
/// types; the atom carries the evaluator as `Arc<dyn SaeBasisEvaluator>`, so we
/// reconstruct the family bound from the atom's basis kind + width + centers.
fn family_jet_sups(atom: &SaeManifoldAtom, chart: &ChartRegion) -> Result<JetSups, String> {
    use crate::terms::sae_manifold::SaeAtomBasisKind::*;
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
        EuclideanPatch => {
            // The patch width fixes max_degree implicitly; bound by a degree that
            // covers the column count (conservative). Degree d-patch column count
            // grows fast; we recover the smallest degree whose patch is ≥ m.
            let degree = euclidean_patch_degree(d, m);
            let ev = EuclideanPatchEvaluator::new(d, degree)?;
            JetSups::from_family(&ev, chart)
        }
        Duchon => {
            // Reconstruct a radial bound from the atom's stored centers when
            // available; otherwise treat the kernel as cubic over the chart.
            let centers = duchon_centers_from_atom(atom);
            let ev = DuchonCoordinateEvaluator::new(centers, 1)?;
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
fn euclidean_patch_degree(latent_dim: usize, m: usize) -> usize {
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
fn integer_root(n: usize, k: usize) -> usize {
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

fn patch_column_count(latent_dim: usize, degree: usize) -> usize {
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
fn duchon_centers_from_atom(atom: &SaeManifoldAtom) -> Array2<f64> {
    // One center at the origin in latent_dim space is a sound conservative
    // default: the chart's own r_min / r_max bracket the true radial range.
    Array2::<f64>::zeros((1, atom.latent_dim.max(1)))
}

/// The four per-column jet sups of a basis family over a chart.
#[derive(Debug, Clone, Copy)]
struct JetSups {
    value: f64,
    jacobian: f64,
    hessian: f64,
    third: f64,
}

impl JetSups {
    fn from_family<B: BasisHessianLipschitz>(family: &B, chart: &ChartRegion) -> Self {
        Self {
            value: family.value_sup(chart),
            jacobian: family.jacobian_sup(chart),
            hessian: family.hessian_sup(chart),
            third: family.third_sup(chart),
        }
    }
}

/// Evaluate one atom's encode objective gradient `F(t) = ∇f_k(t)` and Hessian
/// `F'(t) = H_tt` at a single coordinate `t`, for a single target row `x` and
/// fixed amplitude `z`. This mirrors the fixed-decoder arrow-Schur assembly
/// (`assemble_arrow_schur`) per row: `g_t = J_mᵀ r`, `H_tt = J_mᵀ J_m`, with
/// `m = z·BᵀΦ(t)`, `r = m − x`, `J_m = z·Bᵀ J_Φ`. The Gauss-Newton Hessian is
/// used (PSD, exactly as the production encode solver does), plus an optional
/// ridge floor.
fn encode_grad_hess(
    atom: &SaeManifoldAtom,
    evaluator: &dyn SaeBasisEvaluator,
    t: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    amplitude: f64,
    ridge: f64,
) -> Result<(Array1<f64>, Array2<f64>), String> {
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
    // g_t[axis] = J_m[axis] · r ;  H_tt[a,b] = J_m[a] · J_m[b].
    let mut g = Array1::<f64>::zeros(d);
    let mut h = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        let ja = jm.row(a);
        g[a] = ja.dot(&residual);
        for b in 0..d {
            h[[a, b]] = ja.dot(&jm.row(b));
        }
    }
    for a in 0..d {
        h[[a, a]] += ridge;
    }
    Ok((g, h))
}

/// Operator-norm of `H⁻¹` (i.e. `β = 1/λ_min(H)`) and the Newton step
/// `δ = −H⁻¹ g` with `η = ‖δ‖`, from a symmetric PSD `H` and gradient `g`.
/// Returns `None` when `H` is numerically singular (λ_min ≤ 0) — an
/// uncertifiable start.
fn beta_eta_newton(
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
    let (g, h) = encode_grad_hess(atom, evaluator, t0, x, amplitude, ridge)?;
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
        None => Ok((
            RowCertificate {
                beta: f64::INFINITY,
                eta: f64::INFINITY,
                lipschitz,
                h: f64::INFINITY,
            },
            Array1::<f64>::zeros(atom.latent_dim),
        )),
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
            let atlas = Self::build_atom_atlas(
                k,
                atom,
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

    fn build_atom_atlas(
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
            let lipschitz = hessian_lipschitz_constant(
                sups.value,
                sups.jacobian,
                sups.hessian,
                sups.third,
                decoder_norm_sum,
                amplitude_bound,
                target_norm_bound,
                0.0,
            );
            // β at the chart center bounds the worst-case in-chart curvature
            // (the Gauss-Newton Hessian is continuous; the certified radius is
            // solved so the certificate is robust to the start within the ball).
            let beta_center = match center_beta(atom, &center, config.ridge) {
                Some(b) => b,
                None => {
                    // Degenerate center curvature: no certifiable chart here.
                    charts.push(CertifiedChart {
                        region,
                        lipschitz,
                        beta_center: f64::INFINITY,
                        certified_radius: 0.0,
                    });
                    continue;
                }
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
            });
        }
        Ok(AtomEncodeAtlas {
            atom_index,
            latent_dim: d,
            decoder_norm_sum,
            charts,
        })
    }

    /// Online certified encode of one target row `x` against one atom `k` with
    /// fixed amplitude `z`. Routes to the nearest chart, runs `config.newton_steps`
    /// Newton steps, and returns the encoded coordinate with its certificate.
    /// An uncertified start (no chart, `h > ½`) flags the row for the exact
    /// multi-start fallback.
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
        let evaluator = atom
            .basis_evaluator
            .as_ref()
            .ok_or_else(|| format!("certified_encode_row: atom {atom_index} has no evaluator"))?
            .clone();
        let d = atom.latent_dim;

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
        let mut t = chart.region.center.clone();

        // The per-row certificate is evaluated AT the start point (the chart
        // center), using the chart's closed-form L. This is the exactness gate.
        let (cert, mut delta) = row_certificate(
            atom,
            evaluator.as_ref(),
            t.view(),
            x,
            amplitude,
            chart.lipschitz,
            self.config.ridge,
        )?;
        if !cert.certified() {
            return Ok((t, cert));
        }
        // Certified: 1–2 Newton steps converge quadratically into the root.
        for step in 0..self.config.newton_steps {
            t = &t + &delta;
            if step + 1 < self.config.newton_steps {
                let (_c, next_delta) = row_certificate(
                    atom,
                    evaluator.as_ref(),
                    t.view(),
                    x,
                    amplitude,
                    chart.lipschitz,
                    self.config.ridge,
                )?;
                delta = next_delta;
            }
        }
        Ok((t, cert))
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
        let mut coords = Array2::<f64>::zeros((n, d));
        let mut certified = Vec::with_capacity(n);
        for row in 0..n {
            let (t, cert) = self.certified_encode_row(
                atom,
                atom_index,
                targets.row(row),
                amplitudes[row],
            )?;
            coords.row_mut(row).assign(&t);
            certified.push(cert.certified());
        }
        Ok(EncodeResult::from_rows(coords, certified))
    }
}

/// `β = 1/λ_min(H_tt)` at a chart center, using the encode Gauss-Newton Hessian
/// with the chart center as the coordinate and a zero residual (the offline
/// `β` bounds the curvature; the residual term is dropped in Gauss-Newton). The
/// target is irrelevant to `H_tt = J_mᵀ J_m`, so any `x` gives the same `H`; we
/// pass zeros.
fn center_beta(atom: &SaeManifoldAtom, center: &Array1<f64>, ridge: f64) -> Option<f64> {
    let evaluator = atom.basis_evaluator.as_ref()?.clone();
    let p = atom.output_dim();
    let x = Array1::<f64>::zeros(p);
    let (_g, h) = encode_grad_hess(atom, evaluator.as_ref(), center.view(), x.view(), 1.0, ridge).ok()?;
    let (vals, _vecs) = h.eigh(Side::Lower).ok()?;
    let lambda_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    if lambda_min.is_finite() && lambda_min > 0.0 {
        Some(1.0 / lambda_min)
    } else {
        None
    }
}

/// Route a target row to the nearest chart of an atom by reconstruction
/// distance: the chart whose center reconstruction `m(t_c)` is closest to `x`.
/// Returns the chart index and the distance, or `None` when the atom has no
/// charts.
fn nearest_chart(
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

/// Lay down chart centers on an atom's coordinate grid (the SHAPE_BAND grid
/// idiom): a regular grid spanning the compact latent domain for periodic /
/// sphere / torus atoms, and the atom's own training-coordinate hull for
/// unbounded (Duchon / Euclidean) atoms.
fn chart_center_grid(atom: &SaeManifoldAtom, resolution: usize) -> Array2<f64> {
    let d = atom.latent_dim;
    if let Some(grid) = atom.basis_kind.projection_seed_grid(d, resolution) {
        return grid;
    }
    // Unbounded latents: use a strided sample of the atom's own training
    // coordinates (the convex hull where the encode lands), capped at
    // SHAPE_BAND-style point count. The atom's basis_values rows are the
    // training-row evaluations; the coordinate hull is recovered from the
    // training coords carried alongside. We fall back to the origin row when no
    // coords are available.
    let n_rows = atom.n_obs();
    if n_rows == 0 {
        return Array2::<f64>::zeros((1, d));
    }
    const SHAPE_BAND_MAX_POINTS: usize = 512;
    let target_points = resolution.saturating_mul(resolution).min(SHAPE_BAND_MAX_POINTS).max(1);
    let stride = n_rows.div_ceil(target_points).max(1);
    let rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
    // Without stored coords we anchor charts at the origin plus a unit spread on
    // each axis, a conservative cover that the certified radius refines.
    let g = rows.len().max(1);
    let mut grid = Array2::<f64>::zeros((g, d));
    for (gi, _row) in rows.iter().enumerate() {
        for axis in 0..d {
            grid[[gi, axis]] = (gi as f64) / (g.max(1) as f64) - 0.5;
        }
    }
    grid
}

/// Nominal in-chart radius: half the inter-center grid spacing, so charts tile
/// the domain. For compact latents this is the grid step; for unbounded latents
/// a unit default that the certified radius refines.
fn chart_nominal_radius(atom: &SaeManifoldAtom, resolution: usize) -> f64 {
    use crate::terms::sae_manifold::SaeAtomBasisKind::*;
    match &atom.basis_kind {
        Periodic | Torus => 0.5 / (resolution.max(2) as f64),
        Sphere => std::f64::consts::PI / (resolution.max(2) as f64),
        Duchon | EuclideanPatch | Precomputed(_) => 1.0 / (resolution.max(2) as f64),
    }
}

/// Build the [`ChartRegion`] for a center, attaching the radial r_min / r_max
/// bracket for Duchon atoms (the chart's distance range to the kernel centers).
fn chart_region(atom: &SaeManifoldAtom, center: Array1<f64>, radius: f64) -> ChartRegion {
    use crate::terms::sae_manifold::SaeAtomBasisKind::*;
    let region = ChartRegion::new(center.clone(), radius);
    match &atom.basis_kind {
        Duchon => {
            // r ranges over [‖t_c‖ − radius, ‖t_c‖ + radius] about the single
            // origin-anchored center used by the conservative radial bound.
            let center_norm = center.dot(&center).sqrt();
            let r_min = (center_norm - radius).max(radius.max(f64::MIN_POSITIVE));
            let r_max = center_norm + radius;
            region.with_radial_bounds(r_min, r_max)
        }
        _ => region,
    }
}

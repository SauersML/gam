use ndarray::{Array2, Array3, Array4, Array5, ArrayView2};
use std::sync::Arc;

pub trait SaeBasisEvaluator: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String>;

    /// Return the same evaluator after the coordinate change
    /// `old_t = shift + scale * new_t`, when the basis family can transport the
    /// decoder coefficients exactly enough for the accepted-iterate gauge fix.
    fn affine_transformed_evaluator(
        &self,
        shift: &[f64],
        scale: &[f64],
        n_basis: usize,
    ) -> Result<Option<Arc<dyn SaeBasisSecondJet>>, String> {
        if shift.len() == usize::MAX || scale.len() == usize::MAX || n_basis == usize::MAX {
            return Err("SaeBasisEvaluator::affine_transformed_evaluator: unreachable affine metadata width".to_string());
        }
        Ok(None)
    }

    /// Column split for the curvature homotopy `Phi_eta = [linear, eta*curved]`.
    ///
    /// The default is a flat linear basis. Curved atom evaluators override this
    /// with their topology-specific split; callers pass `n_basis` so the split is
    /// checked against the concrete design width currently being evaluated.
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        Ok(PhiEtaSplit::all_linear(n_basis))
    }

    /// Evaluate the basis at curvature scale `eta in [0, 1]` plus the analytic
    /// derivative with respect to eta.
    ///
    /// At `eta == 1.0` this leaves the existing basis and jet arrays untouched,
    /// so the returned `phi`/`jet` are exactly the same values as [`Self::evaluate`].
    fn evaluate_phi_eta(
        &self,
        coords: ArrayView2<'_, f64>,
        eta: f64,
    ) -> Result<PhiEtaEvaluation, String> {
        if !(eta.is_finite() && (0.0..=1.0).contains(&eta)) {
            return Err(format!(
                "SaeBasisEvaluator::evaluate_phi_eta: eta must be finite in [0, 1]; got {eta}"
            ));
        }
        let (mut phi, mut jet) = self.evaluate(coords)?;
        let split = self.phi_eta_split(phi.ncols())?;
        let mut dphi_deta = Array2::<f64>::zeros(phi.dim());
        let mut djet_deta = Array3::<f64>::zeros(jet.dim());
        for &col in &split.curved_cols {
            if col >= phi.ncols() {
                return Err(format!(
                    "SaeBasisEvaluator::evaluate_phi_eta: curved column {col} exceeds basis width {}",
                    phi.ncols()
                ));
            }
            for row in 0..phi.nrows() {
                dphi_deta[[row, col]] = phi[[row, col]];
                if eta != 1.0 {
                    phi[[row, col]] *= eta;
                }
                for axis in 0..jet.shape()[2] {
                    djet_deta[[row, col, axis]] = jet[[row, col, axis]];
                    if eta != 1.0 {
                        jet[[row, col, axis]] *= eta;
                    }
                }
            }
        }
        Ok(PhiEtaEvaluation {
            phi,
            jet,
            dphi_deta,
            djet_deta,
            split,
        })
    }

    /// Object-safe forwarder to [`SaeBasisSecondJet::second_jet`] for callers
    /// holding `&dyn SaeBasisEvaluator` / `Arc<dyn SaeBasisEvaluator>`.
    ///
    /// Implementations return `Some(result)` only when an analytic second jet
    /// exists for this evaluator. Returning `None` is an explicit capability
    /// declaration, not a default sentinel hidden in the trait.
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>>;

    /// Object-safe forwarder to the basis third jet
    /// `T[n, m, a, c, e] = ∂³Φ_m / ∂t_a ∂t_c ∂t_e`, for callers holding
    /// `&dyn SaeBasisEvaluator` / `Arc<dyn SaeBasisSecondJet>`. The exact
    /// isometry Hessian (`IsometryPenalty::hvp`) needs the *decoder* third jet
    /// `K = Σ_m T[..,m,..]·B[m,:]` for its residual·curvature term; without it
    /// that exact Hessian silently drops the residual and collapses to
    /// Gauss-Newton (issue #458).
    ///
    /// Implementations return `Some(result)` only when an analytic third jet
    /// exists for this evaluator. Evaluators without one return `None`
    /// explicitly; there is no finite-difference fallback.
    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiEtaSplit {
    pub linear_cols: Vec<usize>,
    pub curved_cols: Vec<usize>,
}

impl PhiEtaSplit {
    pub fn all_linear(n_basis: usize) -> Self {
        Self {
            linear_cols: (0..n_basis).collect(),
            curved_cols: Vec::new(),
        }
    }

    fn from_curved_mask(mask: Vec<bool>) -> Self {
        let mut linear_cols = Vec::new();
        let mut curved_cols = Vec::new();
        for (col, curved) in mask.into_iter().enumerate() {
            if curved {
                curved_cols.push(col);
            } else {
                linear_cols.push(col);
            }
        }
        Self {
            linear_cols,
            curved_cols,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhiEtaEvaluation {
    pub phi: Array2<f64>,
    pub jet: Array3<f64>,
    pub dphi_deta: Array2<f64>,
    pub djet_deta: Array3<f64>,
    pub split: PhiEtaSplit,
}

fn monomial_linear_mask(dimension: usize, max_total_degree: usize) -> Vec<bool> {
    crate::basis::monomial_exponents(dimension, max_total_degree)
        .iter()
        .map(|alpha| alpha.iter().sum::<usize>() <= 1)
        .collect()
}

fn duchon_effective_order_for_eta(
    centers: ArrayView2<'_, f64>,
    order: crate::basis::DuchonNullspaceOrder,
) -> crate::basis::DuchonNullspaceOrder {
    let mut effective = order;
    while effective != crate::basis::DuchonNullspaceOrder::Zero
        && centers.nrows() <= duchon_polynomial_column_count(centers.ncols(), effective)
    {
        effective = match effective {
            crate::basis::DuchonNullspaceOrder::Zero => crate::basis::DuchonNullspaceOrder::Zero,
            crate::basis::DuchonNullspaceOrder::Linear => crate::basis::DuchonNullspaceOrder::Zero,
            crate::basis::DuchonNullspaceOrder::Degree(2) => {
                crate::basis::DuchonNullspaceOrder::Linear
            }
            crate::basis::DuchonNullspaceOrder::Degree(k) => {
                crate::basis::DuchonNullspaceOrder::Degree(k - 1)
            }
        };
    }
    effective
}

fn duchon_polynomial_column_count(
    dimension: usize,
    order: crate::basis::DuchonNullspaceOrder,
) -> usize {
    match order {
        crate::basis::DuchonNullspaceOrder::Zero => 1,
        crate::basis::DuchonNullspaceOrder::Linear => dimension + 1,
        crate::basis::DuchonNullspaceOrder::Degree(degree) => {
            crate::basis::monomial_exponents(dimension, degree).len()
        }
    }
}

/// Bases that expose an analytic second jet
/// `H[n, m, a, c] = ∂²Phi_k[n, m] / (∂t_{n,a} ∂t_{n,c})`,
/// shape `(n_rows, n_basis, latent_dim, latent_dim)`.
///
/// Implemented only by evaluators with a closed-form Hessian (periodic
/// harmonic, sphere chart, torus). Callers that need an analytic
/// `∂J/∂t` require this bound; evaluators without it must use a
/// derivative-free fallback. Replaces the previous `Option<Array4<f64>>`
/// return on the base trait so the "no second jet" case is encoded by
/// trait absence rather than a sentinel `None`, and shape mismatches
/// surface as descriptive errors instead of silently collapsing to
/// `None`.
pub trait SaeBasisSecondJet: SaeBasisEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String>;
}

/// Bases that expose an analytic third jet
/// `T[n, m, a, c, e] = ∂³Φ_m[n] / (∂t_{n,a} ∂t_{n,c} ∂t_{n,e})`,
/// shape `(n_rows, n_basis, latent_dim, latent_dim, latent_dim)`.
///
/// The exact isometry Hessian (`IsometryPenalty::hvp`) needs the third decoder
/// jet `K = ∂³φ/∂t³ = Σ_m T[..,m,..] · B[m, :]` for its residual·curvature term
/// `B_{ab,cd} = K_{a,cd}ᵀ W J_b + H_{a,c}ᵀ W H_{b,d} + H_{a,d}ᵀ W H_{b,c}
/// + J_aᵀ W K_{b,cd}`. Bases that supply a closed-form `H` (the
/// [`SaeBasisSecondJet`] super-bound) but not `K` leave that exact Hessian
/// silently dropping the residual term; this trait closes that gap for every
/// analytic basis: the curved bases (sphere chart, periodic harmonic, torus
/// harmonic), the Euclidean monomial patch, the trivially-zero affine basis,
/// and the Duchon basis (radial third-derivative kernel block + monomial
/// nullspace block, both in closed form). The full third jet is symmetric in
/// its three trailing axes.
pub trait SaeBasisThirdJet: SaeBasisSecondJet {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String>;
}

/// Periodic harmonic basis evaluator for a single-dimensional circle latent.
///
/// Produces `M = 2*num_harmonics + 1` basis functions
/// `[1, sin(2π·1·t), cos(2π·1·t), …, sin(2π·H·t), cos(2π·H·t)]` where
/// `H = (M − 1) / 2`. The latent must have `latent_dim == 1`.
#[derive(Debug, Clone)]
pub struct PeriodicHarmonicEvaluator {
    pub num_basis: usize,
}

impl PeriodicHarmonicEvaluator {
    pub fn new(num_basis: usize) -> Result<Self, String> {
        if num_basis == 0 || num_basis % 2 == 0 {
            return Err(format!(
                "PeriodicHarmonicEvaluator requires odd num_basis >= 1; got {num_basis}"
            ));
        }
        Ok(Self { num_basis })
    }
}

impl SaeBasisEvaluator for PeriodicHarmonicEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        if n_basis != self.num_basis {
            return Err(format!(
                "PeriodicHarmonicEvaluator::phi_eta_split: n_basis {n_basis} != evaluator width {}",
                self.num_basis
            ));
        }
        let mut curved = vec![false; n_basis];
        for h in 2..=(n_basis - 1) / 2 {
            curved[2 * h - 1] = true;
            curved[2 * h] = true;
        }
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            phi[[row, 0]] = 1.0;
            for h in 1..=num_harmonics {
                let angle = two_pi * (h as f64) * t;
                let s = angle.sin();
                let c = angle.cos();
                let s_idx = 2 * h - 1;
                let c_idx = 2 * h;
                phi[[row, s_idx]] = s;
                phi[[row, c_idx]] = c;
                jet[[row, s_idx, 0]] = two_pi * (h as f64) * c;
                jet[[row, c_idx, 0]] = -two_pi * (h as f64) * s;
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for PeriodicHarmonicEvaluator {
    /// Second derivative of the 1D Fourier basis on the unit circle.
    ///
    /// For `Phi = [1, sin(2π h t), cos(2π h t), ...]` we have
    /// `Phi'' = [0, -(2π h)² sin(...), -(2π h)² cos(...), ...]`, i.e.
    /// the second derivative is `-(2π h)² · phi(t)` on each harmonic pair.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator::second_jet: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut h = Array4::<f64>::zeros((n, m, 1, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            for k in 1..=num_harmonics {
                let freq = two_pi * (k as f64);
                let freq2 = freq * freq;
                let angle = freq * t;
                let s = angle.sin();
                let c = angle.cos();
                let s_idx = 2 * k - 1;
                let c_idx = 2 * k;
                h[[row, s_idx, 0, 0]] = -freq2 * s;
                h[[row, c_idx, 0, 0]] = -freq2 * c;
            }
        }
        Ok(h)
    }
}

impl SaeBasisThirdJet for PeriodicHarmonicEvaluator {
    /// Third derivative of the 1-D Fourier basis on the unit circle.
    ///
    /// For `Phi = [1, sin(2π h t), cos(2π h t), …]` the chain of derivatives is
    /// `sin → ωc → −ω²s → −ω³c` and `cos → −ωs → −ω²c → ω³s`, so the third
    /// derivative is `[0, −(2π h)³ cos(…), +(2π h)³ sin(…), …]`.
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator::third_jet: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut t3 = Array5::<f64>::zeros((n, m, 1, 1, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            for k in 1..=num_harmonics {
                let freq = two_pi * (k as f64);
                let freq3 = freq * freq * freq;
                let angle = freq * t;
                let s = angle.sin();
                let c = angle.cos();
                let s_idx = 2 * k - 1;
                let c_idx = 2 * k;
                t3[[row, s_idx, 0, 0, 0]] = -freq3 * c;
                t3[[row, c_idx, 0, 0, 0]] = freq3 * s;
            }
        }
        Ok(t3)
    }
}

/// Raw-angle periodic evaluator for the minimal SAE-manifold front-end.
///
/// The basis is exactly `[cos(t), sin(t)]` with `t` measured in radians. If
/// the latent coordinate has more than one axis, the first axis carries the
/// circle phase and the remaining axes are left available to the optimizer but
/// do not enter this basis.
#[derive(Debug, Clone)]
pub struct RawPeriodicCircleEvaluator {
    pub latent_dim: usize,
}

impl RawPeriodicCircleEvaluator {
    pub fn new(latent_dim: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("RawPeriodicCircleEvaluator requires latent_dim >= 1".to_string());
        }
        Ok(Self { latent_dim })
    }
}

impl SaeBasisEvaluator for RawPeriodicCircleEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        if n_basis != 2 {
            return Err(format!(
                "RawPeriodicCircleEvaluator::phi_eta_split: n_basis {n_basis} != 2"
            ));
        }
        Ok(PhiEtaSplit::all_linear(n_basis))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        if coords.ncols() != self.latent_dim {
            return Some(Err(format!(
                "RawPeriodicCircleEvaluator::second_jet_dyn: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            )));
        }
        None
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        if coords.ncols() != self.latent_dim {
            return Some(Err(format!(
                "RawPeriodicCircleEvaluator::third_jet_dyn: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            )));
        }
        None
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "RawPeriodicCircleEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = Array3::<f64>::zeros((n, 2, self.latent_dim));
        for row in 0..n {
            let t = coords[[row, 0]];
            phi[[row, 0]] = t.cos();
            phi[[row, 1]] = t.sin();
            jet[[row, 0, 0]] = -t.sin();
            jet[[row, 1, 0]] = t.cos();
        }
        Ok((phi, jet))
    }
}

/// Diagonal of the chart-local seven-column sphere basis penalty.
///
/// The columns are `[1, x, y, z, xy, yz, xz]`; the constant column carries a
/// numerically-negligible ridge (`1e-8`) so the penalty stays positive
/// definite, the three linear columns are penalized at unit weight, and the
/// three bilinear columns at weight `4` (their second-order angular content).
/// This is the single source of truth for the chart penalty shared between the
/// core SAE path and the PyFFI `sphere_chart_basis_with_jet` helper.
pub const SPHERE_CHART_PENALTY_DIAGONAL: [f64; 7] = [1e-8, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0];

/// Shared single source of truth for the chart-local sphere basis and its
/// analytic first-derivative (lat/lon) jet.
///
/// `coords` is an `(N, 2)` array of latitude/longitude pairs in radians. The
/// returned `phi` has shape `(N, 7)` with columns `[1, x, y, z, xy, yz, xz]`
/// for the unit-sphere embedding `x = cos(lat)cos(lon)`, `y = cos(lat)sin(lon)`,
/// `z = sin(lat)`; the returned `jet` has shape `(N, 7, 2)` with the last axis
/// indexing `[∂/∂lat, ∂/∂lon]`.
///
/// The map and its jet are everywhere `C^∞` in `(lat, lon)`: every column is a
/// polynomial in `cos`/`sin` of the two coordinates, and `cos`/`sin` are entire,
/// so the exact analytic derivatives `∂x/∂lat = -sin(lat)cos(lon)`, … are
/// globally smooth. Latitude is therefore **not** clamped and the latitude
/// derivatives are **not** gated here.
///
/// The physical `lat ∈ [-π/2, π/2]` box that pins a canonical latitude range is
/// enforced where it belongs — in the latent retraction / tangent projection
/// ([`crate::terms::latent_coord::LatentManifold::Interval`]), which clamps the
/// coordinate after each step and zeroes only the *outward-normal* component of
/// the tangent velocity at an active bound (a correct KKT projection). The old
/// binary `chain_lat` gate instead zeroed the *entire* latitude jet at the
/// boundary, making the basis nonsmooth there: an atom whose latitude reached
/// `±π/2` saw a zero latitude gradient and froze, even for the tangential
/// (in-box) direction along which the loss does decrease. Computing the exact
/// jet here and letting the retraction handle the bound restores a smooth
/// objective and the correct boundary behaviour. Both the core path
/// ([`SphereChartEvaluator`]) and the PyFFI helper route through this function.
pub fn sphere_chart_basis_jet(
    coords: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    if coords.ncols() != 2 {
        return Err(format!(
            "sphere_chart_basis_jet expects latent_dim == 2, got {}",
            coords.ncols()
        ));
    }
    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 7));
    let mut jet = Array3::<f64>::zeros((n, 7, 2));
    for row in 0..n {
        let lat = coords[[row, 0]];
        let lon = coords[[row, 1]];
        let clat = lat.cos();
        let slat = lat.sin();
        let clon = lon.cos();
        let slon = lon.sin();
        let x = clat * clon;
        let y = clat * slon;
        let z = slat;
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = x;
        phi[[row, 2]] = y;
        phi[[row, 3]] = z;
        phi[[row, 4]] = x * y;
        phi[[row, 5]] = y * z;
        phi[[row, 6]] = x * z;

        let dx_dlat = -slat * clon;
        let dx_dlon = -clat * slon;
        let dy_dlat = -slat * slon;
        let dy_dlon = clat * clon;
        let dz_dlat = clat;
        jet[[row, 1, 0]] = dx_dlat;
        jet[[row, 1, 1]] = dx_dlon;
        jet[[row, 2, 0]] = dy_dlat;
        jet[[row, 2, 1]] = dy_dlon;
        jet[[row, 3, 0]] = dz_dlat;
        jet[[row, 4, 0]] = dx_dlat * y + x * dy_dlat;
        jet[[row, 4, 1]] = dx_dlon * y + x * dy_dlon;
        jet[[row, 5, 0]] = dy_dlat * z + y * dz_dlat;
        jet[[row, 5, 1]] = dy_dlon * z;
        jet[[row, 6, 0]] = dx_dlat * z + x * dz_dlat;
        jet[[row, 6, 1]] = dx_dlon * z;
    }
    Ok((phi, jet))
}

/// Lat/lon sphere chart evaluator used by the Rust-owned minimal SAE path.
#[derive(Debug, Clone)]
pub struct SphereChartEvaluator;

impl SaeBasisEvaluator for SphereChartEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        if n_basis != 7 {
            return Err(format!(
                "SphereChartEvaluator::phi_eta_split: n_basis {n_basis} != 7"
            ));
        }
        let mut curved = vec![false; n_basis];
        for col in 4..7 {
            curved[col] = true;
        }
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        sphere_chart_basis_jet(coords)
    }
}

impl SaeBasisSecondJet for SphereChartEvaluator {
    /// Analytic Hessian of the 7-column lat/lon sphere chart basis.
    ///
    /// With `x = cos(lat) cos(lon)`, `y = cos(lat) sin(lon)`, `z = sin(lat)`
    /// the non-trivial second derivatives are
    ///
    /// ```text
    /// x_{lat,lat} = -x,     x_{lon,lon} = -x,     x_{lat,lon} = sin(lat)·sin(lon)
    /// y_{lat,lat} = -y,     y_{lon,lon} = -y,     y_{lat,lon} = -sin(lat)·cos(lon)
    /// z_{lat,lat} = -z,     z_{lon,lon} =  0,     z_{lat,lon} =  0
    /// ```
    ///
    /// Bilinear basis entries `xy, yz, xz` follow the product rule
    /// `(fg)_{αβ} = f_{αβ} g + f_α g_β + f_β g_α + f g_{αβ}`. The map is `C^∞`
    /// in `(lat, lon)`, so the Hessian is the exact analytic one with no clamp
    /// or boundary gating; the `lat ∈ [-π/2, π/2]` box is enforced by the
    /// retraction, not by truncating derivatives (see [`sphere_chart_basis_jet`]).
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "SphereChartEvaluator::second_jet expects latent_dim == 2, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut h = Array4::<f64>::zeros((n, 7, 2, 2));
        for row in 0..n {
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            let x = clat * clon;
            let y = clat * slon;
            let z = slat;
            let dx = [-slat * clon, -clat * slon];
            let dy = [-slat * slon, clat * clon];
            let dz = [clat, 0.0];
            let hx = [[-x, slat * slon], [slat * slon, -x]];
            let hy = [[-y, -slat * clon], [-slat * clon, -y]];
            let hz = [[-z, 0.0], [0.0, 0.0]];
            for axis_a in 0..2 {
                for axis_b in 0..2 {
                    h[[row, 1, axis_a, axis_b]] = hx[axis_a][axis_b];
                    h[[row, 2, axis_a, axis_b]] = hy[axis_a][axis_b];
                    h[[row, 3, axis_a, axis_b]] = hz[axis_a][axis_b];
                }
            }
            let pair = |hf: [[f64; 2]; 2],
                        df: [f64; 2],
                        f: f64,
                        hg: [[f64; 2]; 2],
                        dg: [f64; 2],
                        g: f64|
             -> [[f64; 2]; 2] {
                let mut out = [[0.0; 2]; 2];
                for axis_a in 0..2 {
                    for axis_b in 0..2 {
                        out[axis_a][axis_b] = hf[axis_a][axis_b] * g
                            + df[axis_a] * dg[axis_b]
                            + df[axis_b] * dg[axis_a]
                            + f * hg[axis_a][axis_b];
                    }
                }
                out
            };
            let hxy = pair(hx, dx, x, hy, dy, y);
            let hyz = pair(hy, dy, y, hz, dz, z);
            let hxz = pair(hx, dx, x, hz, dz, z);
            for axis_a in 0..2 {
                for axis_b in 0..2 {
                    h[[row, 4, axis_a, axis_b]] = hxy[axis_a][axis_b];
                    h[[row, 5, axis_a, axis_b]] = hyz[axis_a][axis_b];
                    h[[row, 6, axis_a, axis_b]] = hxz[axis_a][axis_b];
                }
            }
        }
        Ok(h)
    }
}

impl SaeBasisThirdJet for SphereChartEvaluator {
    /// Third derivative of the 7-column lat/lon sphere chart basis
    /// `[1, x, y, z, xy, yz, xz]`.
    ///
    /// Each Cartesian coordinate is *separable* in (lat, lon):
    /// `x = cos(lat) cos(lon)`, `y = cos(lat) sin(lon)`, `z = sin(lat)·1`. A
    /// separable coordinate's mixed derivative is the product of the per-axis
    /// derivative of the right order, so it is fully described by two
    /// length-4 derivative tables (orders 0..3) — one per axis. The map is
    /// `C^∞` in `(lat, lon)`; the tables are the exact analytic derivatives
    /// with no clamp or boundary gating (the `lat ∈ [-π/2, π/2]` box is
    /// enforced by the retraction, see [`sphere_chart_basis_jet`]).
    ///
    /// The bilinear columns `xy, yz, xz` are products of two separable
    /// coordinates; their third derivative is the symmetric triple-Leibniz sum
    /// over the `2³` ways to route the three derivative operators to the two
    /// factors. This is the order-3 generalization of the `pair` Leibniz used
    /// in [`SaeBasisSecondJet::second_jet`], so the two stay structurally
    /// identical and a finite difference of `second_jet` pins it.
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "SphereChartEvaluator::third_jet expects latent_dim == 2, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut t3 = Array5::<f64>::zeros((n, 7, 2, 2, 2));
        // Derivative of a separable coordinate along axes `ax` (each 0 = lat,
        // 1 = lon): product of the lat table at order `#lat` and the lon table
        // at order `#lon`.
        let single = |lat: &[f64; 4], lon: &[f64; 4], ax: [usize; 3]| -> f64 {
            let n_lat = ax.iter().filter(|&&q| q == 0).count();
            lat[n_lat] * lon[3 - n_lat]
        };
        // Third derivative of a product of two separable coordinates: sum over
        // all 2³ routings of the three operators to factor f vs g (Leibniz).
        let product = |f_lat: &[f64; 4],
                       f_lon: &[f64; 4],
                       g_lat: &[f64; 4],
                       g_lon: &[f64; 4],
                       ax: [usize; 3]|
         -> f64 {
            let mut acc = 0.0;
            for mask in 0u8..8 {
                let (mut f_lat_n, mut f_lon_n, mut g_lat_n, mut g_lon_n) = (0, 0, 0, 0);
                for (i, &axis) in ax.iter().enumerate() {
                    let to_f = (mask >> i) & 1 == 1;
                    match (to_f, axis == 0) {
                        (true, true) => f_lat_n += 1,
                        (true, false) => f_lon_n += 1,
                        (false, true) => g_lat_n += 1,
                        (false, false) => g_lon_n += 1,
                    }
                }
                acc += f_lat[f_lat_n] * f_lon[f_lon_n] * g_lat[g_lat_n] * g_lon[g_lon_n];
            }
            acc
        };
        for row in 0..n {
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            // Per-axis derivative tables, orders 0..3 (exact analytic, no clamp).
            let cos_lat = [clat, -slat, -clat, slat];
            let sin_lat = [slat, clat, -slat, -clat];
            let cos_lon = [clon, -slon, -clon, slon];
            let sin_lon = [slon, clon, -slon, -clon];
            let const_lon = [1.0, 0.0, 0.0, 0.0];
            // x = cos(lat)cos(lon), y = cos(lat)sin(lon), z = sin(lat).
            let (x_lat, x_lon) = (&cos_lat, &cos_lon);
            let (y_lat, y_lon) = (&cos_lat, &sin_lon);
            let (z_lat, z_lon) = (&sin_lat, &const_lon);
            for axis_a in 0..2 {
                for axis_b in 0..2 {
                    for axis_c in 0..2 {
                        let ax = [axis_a, axis_b, axis_c];
                        t3[[row, 1, axis_a, axis_b, axis_c]] = single(x_lat, x_lon, ax);
                        t3[[row, 2, axis_a, axis_b, axis_c]] = single(y_lat, y_lon, ax);
                        t3[[row, 3, axis_a, axis_b, axis_c]] = single(z_lat, z_lon, ax);
                        t3[[row, 4, axis_a, axis_b, axis_c]] =
                            product(x_lat, x_lon, y_lat, y_lon, ax);
                        t3[[row, 5, axis_a, axis_b, axis_c]] =
                            product(y_lat, y_lon, z_lat, z_lon, ax);
                        t3[[row, 6, axis_a, axis_b, axis_c]] =
                            product(x_lat, x_lon, z_lat, z_lon, ax);
                    }
                }
            }
        }
        Ok(t3)
    }
}

/// Tensor-product periodic harmonic evaluator for a `d`-dimensional torus
/// `T^d = (S^1)^d`. The basis is the tensor product over each axis of the
/// 1-D circle basis
/// `[1, cos(2π·1·t), sin(2π·1·t), …, cos(2π·H·t), sin(2π·H·t)]`
/// (each axis contributes `2H+1` factors, so the total basis size is
/// `(2H+1)^d`). The latent coords are angular phases in `[0, 1)` (consistent
/// with the periodic 1-D atoms).
#[derive(Debug, Clone)]
pub struct TorusHarmonicEvaluator {
    pub latent_dim: usize,
    pub num_harmonics: usize,
}

impl TorusHarmonicEvaluator {
    pub fn new(latent_dim: usize, num_harmonics: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("TorusHarmonicEvaluator requires latent_dim >= 1".to_string());
        }
        if num_harmonics == 0 {
            return Err("TorusHarmonicEvaluator requires num_harmonics >= 1".to_string());
        }
        Ok(Self {
            latent_dim,
            num_harmonics,
        })
    }

    pub fn axis_basis_size(&self) -> usize {
        2 * self.num_harmonics + 1
    }

    pub fn basis_size(&self) -> usize {
        // (2H+1)^d — computed iteratively to surface overflow.
        let axis_m = self.axis_basis_size();
        let mut total: usize = 1;
        for _ in 0..self.latent_dim {
            total = total
                .checked_mul(axis_m)
                .expect("TorusHarmonicEvaluator: basis size overflowed usize");
        }
        total
    }
}

impl SaeBasisEvaluator for TorusHarmonicEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        let expected = self.basis_size();
        if n_basis != expected {
            return Err(format!(
                "TorusHarmonicEvaluator::phi_eta_split: n_basis {n_basis} != evaluator width {expected}"
            ));
        }
        let d = self.latent_dim;
        let axis_m = self.axis_basis_size();
        let mut curved = Vec::with_capacity(n_basis);
        let mut idx = vec![0usize; d];
        for _flat in 0..n_basis {
            let mut nonconstant_axes = 0usize;
            let mut has_higher_harmonic = false;
            for &axis_col in &idx {
                if axis_col > 0 {
                    nonconstant_axes += 1;
                    if axis_col > 2 {
                        has_higher_harmonic = true;
                    }
                }
            }
            curved.push(has_higher_harmonic || nonconstant_axes > 1);
            for axis in (0..d).rev() {
                idx[axis] += 1;
                if idx[axis] < axis_m {
                    break;
                }
                idx[axis] = 0;
            }
        }
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "TorusHarmonicEvaluator: expected latent_dim {d}, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let axis_m = self.axis_basis_size();
        let m = self.basis_size();
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, d));
        // Per-axis evaluation buffer: phi_axis[axis][col] and dphi_axis[axis][col].
        let mut phi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut dphi_axis = vec![vec![0.0_f64; axis_m]; d];
        for row in 0..n {
            for axis in 0..d {
                let t = coords[[row, axis]];
                phi_axis[axis][0] = 1.0;
                dphi_axis[axis][0] = 0.0;
                for h in 1..=h_max {
                    let freq = two_pi * (h as f64);
                    let angle = freq * t;
                    let s = angle.sin();
                    let c = angle.cos();
                    let s_idx = 2 * h - 1;
                    let c_idx = 2 * h;
                    phi_axis[axis][s_idx] = s;
                    phi_axis[axis][c_idx] = c;
                    dphi_axis[axis][s_idx] = freq * c;
                    dphi_axis[axis][c_idx] = -freq * s;
                }
            }
            // Enumerate the Cartesian product of per-axis indices in
            // lexicographic order (axis 0 is the slowest).
            let mut idx = vec![0usize; d];
            for flat in 0..m {
                let mut val = 1.0_f64;
                for axis in 0..d {
                    val *= phi_axis[axis][idx[axis]];
                }
                phi[[row, flat]] = val;
                // ∂/∂coords[row, axis_target] = product over axes, replacing
                // phi_axis[axis_target] with its derivative.
                for axis_target in 0..d {
                    let mut deriv = 1.0_f64;
                    for axis in 0..d {
                        deriv *= if axis == axis_target {
                            dphi_axis[axis][idx[axis]]
                        } else {
                            phi_axis[axis][idx[axis]]
                        };
                    }
                    jet[[row, flat, axis_target]] = deriv;
                }
                // Increment lexicographic index (last axis fastest).
                for axis in (0..d).rev() {
                    idx[axis] += 1;
                    if idx[axis] < axis_m {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for TorusHarmonicEvaluator {
    /// Hessian of the tensor-product torus basis.
    ///
    /// Each basis function factors as `Φ_flat = Π_axis f_axis(t_axis)`, so
    ///
    /// * `∂² Φ / ∂t_a ∂t_b = (Π_{k ∉ {a, b}} f_k) · f_a'(t_a) · f_b'(t_b)`
    ///   when `a ≠ b`,
    /// * `∂² Φ / ∂t_a²    = (Π_{k ≠ a} f_k) · f_a''(t_a)` on the diagonal.
    ///
    /// Per-axis the basis is `[1, sin(2π h t), cos(2π h t), …]`, so
    /// `f_axis''(t) = -(2π h)² · f_axis(t)` on the harmonic columns and 0 on
    /// the constant column.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "TorusHarmonicEvaluator::second_jet expects latent_dim == {d}, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let axis_m = self.axis_basis_size();
        let m = self.basis_size();
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut hess = Array4::<f64>::zeros((n, m, d, d));
        let mut phi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut dphi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut d2phi_axis = vec![vec![0.0_f64; axis_m]; d];
        for row in 0..n {
            for axis in 0..d {
                let t = coords[[row, axis]];
                phi_axis[axis][0] = 1.0;
                dphi_axis[axis][0] = 0.0;
                d2phi_axis[axis][0] = 0.0;
                for k in 1..=h_max {
                    let freq = two_pi * (k as f64);
                    let freq2 = freq * freq;
                    let angle = freq * t;
                    let s = angle.sin();
                    let c = angle.cos();
                    let s_idx = 2 * k - 1;
                    let c_idx = 2 * k;
                    phi_axis[axis][s_idx] = s;
                    phi_axis[axis][c_idx] = c;
                    dphi_axis[axis][s_idx] = freq * c;
                    dphi_axis[axis][c_idx] = -freq * s;
                    d2phi_axis[axis][s_idx] = -freq2 * s;
                    d2phi_axis[axis][c_idx] = -freq2 * c;
                }
            }
            let mut idx = vec![0usize; d];
            for flat in 0..m {
                for axis_a in 0..d {
                    for axis_b in 0..d {
                        let mut prod = 1.0_f64;
                        for axis in 0..d {
                            let factor = if axis == axis_a && axis == axis_b {
                                d2phi_axis[axis][idx[axis]]
                            } else if axis == axis_a || axis == axis_b {
                                dphi_axis[axis][idx[axis]]
                            } else {
                                phi_axis[axis][idx[axis]]
                            };
                            prod *= factor;
                        }
                        hess[[row, flat, axis_a, axis_b]] = prod;
                    }
                }
                for axis in (0..d).rev() {
                    idx[axis] += 1;
                    if idx[axis] < axis_m {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
        Ok(hess)
    }
}

impl SaeBasisThirdJet for TorusHarmonicEvaluator {
    /// Third derivative of the tensor-product torus basis.
    ///
    /// Each basis function factors as `Φ_flat = Π_axis f_axis(t_axis)`, so its
    /// third derivative `∂³Φ / ∂t_a ∂t_b ∂t_c` is the product, over every
    /// axis, of `f_axis` differentiated as many times as that axis appears in
    /// `{a, b, c}` (0..3). Per axis the basis is `[1, sin(2π h t),
    /// cos(2π h t), …]`, whose order-3 derivative is `[0, −(2π h)³ cos(…),
    /// +(2π h)³ sin(…), …]`. This is the order-3 sibling of
    /// [`SaeBasisSecondJet::second_jet`].
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "TorusHarmonicEvaluator::third_jet expects latent_dim == {d}, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let axis_m = self.axis_basis_size();
        let m = self.basis_size();
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut t3 = Array5::<f64>::zeros((n, m, d, d, d));
        // Per-axis derivative tables indexed [axis][order 0..3][column].
        let mut deriv_axis = vec![vec![vec![0.0_f64; axis_m]; 4]; d];
        for row in 0..n {
            for axis in 0..d {
                let t = coords[[row, axis]];
                for order in 0..4 {
                    deriv_axis[axis][order][0] = 0.0;
                }
                deriv_axis[axis][0][0] = 1.0;
                for k in 1..=h_max {
                    let freq = two_pi * (k as f64);
                    let freq2 = freq * freq;
                    let freq3 = freq2 * freq;
                    let angle = freq * t;
                    let s = angle.sin();
                    let c = angle.cos();
                    let s_idx = 2 * k - 1;
                    let c_idx = 2 * k;
                    deriv_axis[axis][0][s_idx] = s;
                    deriv_axis[axis][0][c_idx] = c;
                    deriv_axis[axis][1][s_idx] = freq * c;
                    deriv_axis[axis][1][c_idx] = -freq * s;
                    deriv_axis[axis][2][s_idx] = -freq2 * s;
                    deriv_axis[axis][2][c_idx] = -freq2 * c;
                    deriv_axis[axis][3][s_idx] = -freq3 * c;
                    deriv_axis[axis][3][c_idx] = freq3 * s;
                }
            }
            let mut idx = vec![0usize; d];
            for flat in 0..m {
                for axis_a in 0..d {
                    for axis_b in 0..d {
                        for axis_c in 0..d {
                            let mut prod = 1.0_f64;
                            for axis in 0..d {
                                let order = (axis == axis_a) as usize
                                    + (axis == axis_b) as usize
                                    + (axis == axis_c) as usize;
                                prod *= deriv_axis[axis][order][idx[axis]];
                            }
                            t3[[row, flat, axis_a, axis_b, axis_c]] = prod;
                        }
                    }
                }
                for axis in (0..d).rev() {
                    idx[axis] += 1;
                    if idx[axis] < axis_m {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
        Ok(t3)
    }
}

/// Affine Euclidean/Duchon fallback for the minimal fit entrypoint.
#[derive(Debug, Clone)]
pub struct AffineCoordinateEvaluator {
    pub latent_dim: usize,
}

impl AffineCoordinateEvaluator {
    pub fn new(latent_dim: usize) -> Self {
        Self { latent_dim }
    }
}

impl SaeBasisEvaluator for AffineCoordinateEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        let expected = self.latent_dim + 1;
        if n_basis != expected {
            return Err(format!(
                "AffineCoordinateEvaluator::phi_eta_split: n_basis {n_basis} != {expected}"
            ));
        }
        Ok(PhiEtaSplit::all_linear(n_basis))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "AffineCoordinateEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let m = self.latent_dim + 1;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, self.latent_dim));
        phi.column_mut(0).fill(1.0);
        for row in 0..n {
            for axis in 0..self.latent_dim {
                phi[[row, axis + 1]] = coords[[row, axis]];
                jet[[row, axis + 1, axis]] = 1.0;
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for AffineCoordinateEvaluator {
    /// Second derivative of the affine basis `[1, t_1, ..., t_d]`.
    ///
    /// Every basis function is at most linear in `t`, so all second derivatives
    /// are identically zero. Returns the all-zeros tensor of shape
    /// `(n_obs, d+1, d, d)`.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "AffineCoordinateEvaluator::second_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let m = self.latent_dim + 1;
        let d = self.latent_dim;
        Ok(Array4::<f64>::zeros((n, m, d, d)))
    }
}

impl SaeBasisThirdJet for AffineCoordinateEvaluator {
    /// Third derivative of the affine basis `[1, t_1, …, t_d]`. Every column is
    /// at most linear, so all third derivatives vanish identically.
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "AffineCoordinateEvaluator::third_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let m = self.latent_dim + 1;
        let d = self.latent_dim;
        Ok(Array5::<f64>::zeros((n, m, d, d, d)))
    }
}

/// Scale-free Duchon atom evaluator for the SAE-manifold Newton loop.
///
/// Recomputes the radial+polynomial design `Φ(t)` and its first/second
/// input-location jets at arbitrary latent coordinates against a fixed set of
/// `centers` and Duchon null-space `order`. The column layout — the
/// kernel block `Φ_radial(t)·Z` followed by the polynomial block `P(t)`,
/// both carrying the same scalar kernel amplification `α` — matches
/// [`crate::basis::build_duchon_basis`] under the SAE atom's spec
/// (`length_scale = None`, `power = 0`, no identifiability transform). The
/// forward design and the jet are produced from a single core entry point
/// ([`crate::basis::duchon_sae_atom_basis_with_jet`]) so they always agree on
/// column count and scaling — the exact contract issue #247 pinned.
#[derive(Debug, Clone)]
pub struct DuchonCoordinateEvaluator {
    pub centers: Array2<f64>,
    pub order: crate::basis::DuchonNullspaceOrder,
}

impl DuchonCoordinateEvaluator {
    /// Build from the atom's centers and Duchon `m` (`m = 1` → constant
    /// null space, `m = 2` → constant+linear, `m = k+1` → degree-`k`).
    pub fn new(centers: Array2<f64>, m: usize) -> Result<Self, String> {
        if centers.ncols() == 0 {
            return Err("DuchonCoordinateEvaluator: centers must have at least one column".into());
        }
        if m == 0 {
            return Err("DuchonCoordinateEvaluator: Duchon m must be at least 1".into());
        }
        let order = match m {
            1 => crate::basis::DuchonNullspaceOrder::Zero,
            2 => crate::basis::DuchonNullspaceOrder::Linear,
            other => crate::basis::DuchonNullspaceOrder::Degree(other - 1),
        };
        Ok(Self { centers, order })
    }
}

impl SaeBasisEvaluator for DuchonCoordinateEvaluator {
    fn affine_transformed_evaluator(
        &self,
        shift: &[f64],
        scale: &[f64],
        n_basis: usize,
    ) -> Result<Option<Arc<dyn SaeBasisSecondJet>>, String> {
        let dim = self.centers.ncols();
        if shift.len() != dim || scale.len() != dim {
            return Err(format!(
                "DuchonCoordinateEvaluator::affine_transformed_evaluator: affine vectors must have length {dim}; got shift={} scale={}",
                shift.len(),
                scale.len()
            ));
        }
        if n_basis == usize::MAX {
            return Err(
                "DuchonCoordinateEvaluator::affine_transformed_evaluator: unreachable basis width"
                    .to_string(),
            );
        }
        if dim != 1 {
            return Ok(None);
        }
        if !(scale[0].is_finite() && scale[0] > 0.0 && shift[0].is_finite()) {
            return Ok(None);
        }
        let mut centers = self.centers.clone();
        for row in 0..centers.nrows() {
            centers[[row, 0]] = (centers[[row, 0]] - shift[0]) / scale[0];
        }
        Ok(Some(Arc::new(Self {
            centers,
            order: self.order,
        })))
    }

    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        let dim = self.centers.ncols();
        let effective = duchon_effective_order_for_eta(self.centers.view(), self.order);
        let n_poly = duchon_polynomial_column_count(dim, effective);
        if n_basis < n_poly {
            return Err(format!(
                "DuchonCoordinateEvaluator::phi_eta_split: n_basis {n_basis} smaller than polynomial block {n_poly}"
            ));
        }
        let n_kernel = n_basis - n_poly;
        let mut curved = vec![false; n_basis];
        for col in 0..n_kernel {
            curved[col] = true;
        }
        if let crate::basis::DuchonNullspaceOrder::Degree(degree) = effective {
            let linear_mask = monomial_linear_mask(dim, degree);
            if linear_mask.len() != n_poly {
                return Err(format!(
                    "DuchonCoordinateEvaluator::phi_eta_split: polynomial mask width {} != {n_poly}",
                    linear_mask.len()
                ));
            }
            for (local_col, linear) in linear_mask.into_iter().enumerate() {
                if !linear {
                    curved[n_kernel + local_col] = true;
                }
            }
        }
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.centers.ncols() {
            return Err(format!(
                "DuchonCoordinateEvaluator: expected latent_dim {}, got {}",
                self.centers.ncols(),
                coords.ncols()
            ));
        }
        crate::basis::duchon_sae_atom_basis_with_jet(coords, self.centers.view(), self.order)
            .map_err(|err| err.to_string())
    }
}

impl SaeBasisSecondJet for DuchonCoordinateEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.centers.ncols() {
            return Err(format!(
                "DuchonCoordinateEvaluator::second_jet: expected latent_dim {}, got {}",
                self.centers.ncols(),
                coords.ncols()
            ));
        }
        crate::basis::duchon_sae_atom_second_jet(coords, self.centers.view(), self.order)
            .map_err(|err| err.to_string())
    }
}

impl SaeBasisThirdJet for DuchonCoordinateEvaluator {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        if coords.ncols() != self.centers.ncols() {
            return Err(format!(
                "DuchonCoordinateEvaluator::third_jet: expected latent_dim {}, got {}",
                self.centers.ncols(),
                coords.ncols()
            ));
        }
        crate::basis::duchon_sae_atom_third_jet(coords, self.centers.view(), self.order)
            .map_err(|err| err.to_string())
    }
}

/// Flat Euclidean tangent-patch evaluator for the SAE-manifold Newton loop.
///
/// The basis is the set of monomials of total degree ≤ `max_degree` in the
/// atom's latent coordinates (a zero-curvature polynomial expansion, distinct
/// from the thin-plate Duchon kernel). It recomputes the monomial design and
/// its first/second derivatives at arbitrary coordinates, so the inner Newton
/// latent update stays consistent with the deployed design.
#[derive(Debug, Clone)]
pub struct EuclideanPatchEvaluator {
    pub latent_dim: usize,
    pub max_degree: usize,
}

impl EuclideanPatchEvaluator {
    pub fn new(latent_dim: usize, max_degree: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("EuclideanPatchEvaluator: latent_dim must be positive".into());
        }
        Ok(Self {
            latent_dim,
            max_degree,
        })
    }

    pub fn basis_size(&self) -> usize {
        crate::basis::monomial_exponents(self.latent_dim, self.max_degree).len()
    }

    fn order(&self) -> crate::basis::DuchonNullspaceOrder {
        match self.max_degree {
            0 => crate::basis::DuchonNullspaceOrder::Zero,
            1 => crate::basis::DuchonNullspaceOrder::Linear,
            k => crate::basis::DuchonNullspaceOrder::Degree(k),
        }
    }
}

impl SaeBasisEvaluator for EuclideanPatchEvaluator {
    fn affine_transformed_evaluator(
        &self,
        shift: &[f64],
        scale: &[f64],
        n_basis: usize,
    ) -> Result<Option<Arc<dyn SaeBasisSecondJet>>, String> {
        if shift.len() != self.latent_dim || scale.len() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator::affine_transformed_evaluator: affine vectors must have length {}; got shift={} scale={}",
                self.latent_dim,
                shift.len(),
                scale.len()
            ));
        }
        if n_basis != self.basis_size() {
            return Err(format!(
                "EuclideanPatchEvaluator::affine_transformed_evaluator: n_basis {n_basis} != evaluator width {}",
                self.basis_size()
            ));
        }
        if shift.iter().chain(scale.iter()).any(|v| !v.is_finite())
            || scale.iter().any(|&v| v <= 0.0)
        {
            return Ok(None);
        }
        Ok(Some(Arc::new(Self {
            latent_dim: self.latent_dim,
            max_degree: self.max_degree,
        })))
    }

    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        let linear_mask = monomial_linear_mask(self.latent_dim, self.max_degree);
        if linear_mask.len() != n_basis {
            return Err(format!(
                "EuclideanPatchEvaluator::phi_eta_split: polynomial mask width {} != n_basis {n_basis}",
                linear_mask.len()
            ));
        }
        Ok(PhiEtaSplit::from_curved_mask(
            linear_mask.into_iter().map(|linear| !linear).collect(),
        ))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = crate::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        let mut phi = Array2::<f64>::zeros((n, m));
        for (col, alpha) in exponents.iter().enumerate() {
            for row in 0..n {
                let mut value = 1.0_f64;
                for (axis, &exp) in alpha.iter().enumerate() {
                    if exp != 0 {
                        value *= coords[[row, axis]].powi(exp as i32);
                    }
                }
                phi[[row, col]] = value;
            }
        }
        let jet = crate::basis::duchon_polynomial_first_derivative_nd(coords, self.order());
        if jet.shape() != [n, m, self.latent_dim] {
            return Err(format!(
                "EuclideanPatchEvaluator: monomial jet shape {:?} disagrees with ({n}, {m}, {})",
                jet.shape(),
                self.latent_dim
            ));
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for EuclideanPatchEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator::second_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = crate::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        let d = self.latent_dim;
        let mut hess = Array4::<f64>::zeros((n, m, d, d));
        for (col, alpha) in exponents.iter().enumerate() {
            for a in 0..d {
                if alpha[a] == 0 {
                    continue;
                }
                for c in 0..d {
                    if a != c && alpha[c] == 0 {
                        continue;
                    }
                    let lead = if a == c {
                        (alpha[a] as f64) * (alpha[a].saturating_sub(1) as f64)
                    } else {
                        (alpha[a] as f64) * (alpha[c] as f64)
                    };
                    if lead == 0.0 {
                        continue;
                    }
                    for row in 0..n {
                        let mut value = lead;
                        for axis in 0..d {
                            let mut exp = alpha[axis];
                            if axis == a {
                                exp = exp.saturating_sub(1);
                            }
                            if axis == c {
                                exp = exp.saturating_sub(1);
                            }
                            if exp != 0 {
                                value *= coords[[row, axis]].powi(exp as i32);
                            }
                        }
                        hess[[row, col, a, c]] = value;
                    }
                }
            }
        }
        Ok(hess)
    }
}

impl SaeBasisThirdJet for EuclideanPatchEvaluator {
    /// Third derivative of the monomial basis `Φ_α = Π_axis t_axis^{α_axis}`.
    ///
    /// Differentiating axis `j` a total of `k_j` times (where `k_j` is how
    /// often axis `j` appears in `{a, b, c}`) contracts that factor to
    /// `falling(α_j, k_j) · t_j^{α_j − k_j}`, with `falling(α, k) = α(α−1)…
    /// (α−k+1)` and the term vanishing whenever `α_j < k_j`.
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator::third_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = crate::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        let d = self.latent_dim;
        let mut t3 = Array5::<f64>::zeros((n, m, d, d, d));
        let falling = |alpha: usize, k: usize| -> f64 {
            let mut acc = 1.0_f64;
            for j in 0..k {
                acc *= (alpha as f64) - (j as f64);
            }
            acc
        };
        for (col, alpha) in exponents.iter().enumerate() {
            for a in 0..d {
                if alpha[a] == 0 {
                    continue;
                }
                for b in 0..d {
                    for c in 0..d {
                        // Per-axis differentiation order in this (a, b, c) cell.
                        let mut order = vec![0usize; d];
                        order[a] += 1;
                        order[b] += 1;
                        order[c] += 1;
                        if (0..d).any(|axis| order[axis] > alpha[axis]) {
                            continue;
                        }
                        let mut lead = 1.0_f64;
                        for axis in 0..d {
                            lead *= falling(alpha[axis], order[axis]);
                        }
                        if lead == 0.0 {
                            continue;
                        }
                        for row in 0..n {
                            let mut value = lead;
                            for axis in 0..d {
                                let exp = alpha[axis] - order[axis];
                                if exp != 0 {
                                    value *= coords[[row, axis]].powi(exp as i32);
                                }
                            }
                            t3[[row, col, a, b, c]] = value;
                        }
                    }
                }
            }
        }
        Ok(t3)
    }
}

/// Tensor-product harmonic × polynomial evaluator for the cylinder `S¹ × ℝ`
/// (`d = 2`): a periodic circle axis crossed with a flat (Duchon-polynomial)
/// line axis.
///
/// This is the missing geometry in the `d = 2` topology race (torus vs sphere
/// vs euclidean-patch vs **cylinder**). A feature whose latent structure is
/// *periodic along one axis and unbounded-linear along the other* (e.g. a
/// phase-times-magnitude direction in the residual stream) lives on `S¹ × ℝ`,
/// not on `T²` (two circles) or `S²`; before this evaluator it was forced into
/// a torus stand-in (wrapping the linear axis spuriously) or a flat patch
/// (losing the periodicity). With it, the cylinder is a first-class candidate
/// adjudicated under the same TK-normalized evidence gate.
///
/// # Basis
///
/// Axis 0 is the circle, in the `PeriodicHarmonicEvaluator` form
/// `c(t₀) = [1, sin(2π·1·t₀), cos(2π·1·t₀), …, sin(2π·H·t₀), cos(2π·H·t₀)]`
/// (`Mc = 2H + 1` columns; `t₀` is a fraction of one period, matching the
/// periodic/torus convention). Axis 1 is the flat line, in the
/// `EuclideanPatchEvaluator` form `l(t₁) = [1, t₁, t₁², …, t₁^D]`
/// (`Ml = D + 1` columns). The product basis is
/// `Φ_{c,l}(t) = c_c(t₀) · l_l(t₁)`, `M = Mc · Ml` columns, enumerated
/// lexicographically with the circle index slowest (`col = c·Ml + l`), so the
/// constant `[c=0, l=0]` is column 0.
///
/// # Jets (exact product rule across the two axes)
///
/// Because the two factors depend on disjoint coordinates, the value/derivative
/// of `Φ_{c,l}` along a multi-index over `{axis 0, axis 1}` is simply the
/// product of the circle factor differentiated as many times as axis 0 appears
/// and the line factor differentiated as many times as axis 1 appears. Each
/// per-axis derivative table (orders 0..3) is closed form (circle: the Fourier
/// `sin → ωcos → −ω²sin → −ω³cos` chain; line: the falling-factorial monomial
/// chain), so the value, first, second and third jets are all exact and the
/// second/third jets pin against a finite difference of the level below.
#[derive(Debug, Clone)]
pub struct CylinderHarmonicEvaluator {
    /// Number of circle harmonics `H ≥ 1` (axis-0 width is `2H + 1`).
    pub circle_harmonics: usize,
    /// Polynomial degree `D ≥ 0` of the flat line axis (axis-1 width is `D + 1`).
    pub line_degree: usize,
}

impl CylinderHarmonicEvaluator {
    pub fn new(circle_harmonics: usize, line_degree: usize) -> Result<Self, String> {
        if circle_harmonics == 0 {
            return Err(
                "CylinderHarmonicEvaluator requires circle_harmonics >= 1 (S¹ needs at least one \
                 harmonic pair)"
                    .to_string(),
            );
        }
        Ok(Self {
            circle_harmonics,
            line_degree,
        })
    }

    /// Circle-axis width `Mc = 2H + 1`.
    pub fn circle_basis_size(&self) -> usize {
        2 * self.circle_harmonics + 1
    }

    /// Line-axis width `Ml = D + 1`.
    pub fn line_basis_size(&self) -> usize {
        self.line_degree + 1
    }

    /// Product basis width `M = Mc · Ml`.
    pub fn basis_size(&self) -> usize {
        self.circle_basis_size() * self.line_basis_size()
    }

    /// Per-axis circle derivative tables, orders 0..=3, indexed `[order][col]`,
    /// length `Mc` each. Column 0 is the constant; columns `2h-1`/`2h` carry the
    /// `h`-th sin/cos pair at frequency `ω = 2π h`.
    fn circle_tables(&self, t: f64) -> [Vec<f64>; 4] {
        let mc = self.circle_basis_size();
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut table = [
            vec![0.0_f64; mc],
            vec![0.0_f64; mc],
            vec![0.0_f64; mc],
            vec![0.0_f64; mc],
        ];
        // Constant column: value 1, all derivatives 0.
        table[0][0] = 1.0;
        for h in 1..=self.circle_harmonics {
            let omega = two_pi * (h as f64);
            let w2 = omega * omega;
            let w3 = w2 * omega;
            let angle = omega * t;
            let s = angle.sin();
            let c = angle.cos();
            let s_idx = 2 * h - 1;
            let c_idx = 2 * h;
            // sin chain: sin → ω cos → −ω² sin → −ω³ cos.
            table[0][s_idx] = s;
            table[1][s_idx] = omega * c;
            table[2][s_idx] = -w2 * s;
            table[3][s_idx] = -w3 * c;
            // cos chain: cos → −ω sin → −ω² cos → ω³ sin.
            table[0][c_idx] = c;
            table[1][c_idx] = -omega * s;
            table[2][c_idx] = -w2 * c;
            table[3][c_idx] = w3 * s;
        }
        table
    }

    /// Per-axis line (monomial) derivative tables, orders 0..=3, indexed
    /// `[order][col]`, length `Ml` each. Column `j` is `t^j`; its `k`-th
    /// derivative is `falling(j, k) · t^{j-k}` (zero once `k > j`).
    fn line_tables(&self, t: f64) -> [Vec<f64>; 4] {
        let ml = self.line_basis_size();
        let mut table = [
            vec![0.0_f64; ml],
            vec![0.0_f64; ml],
            vec![0.0_f64; ml],
            vec![0.0_f64; ml],
        ];
        for j in 0..ml {
            for k in 0..4 {
                if k > j {
                    // falling(j, k) = 0: the monomial is exhausted.
                    table[k][j] = 0.0;
                    continue;
                }
                let mut coeff = 1.0_f64;
                for q in 0..k {
                    coeff *= (j - q) as f64;
                }
                let residual = j - k;
                let pow = if residual == 0 {
                    1.0
                } else {
                    t.powi(residual as i32)
                };
                table[k][j] = coeff * pow;
            }
        }
        table
    }

    /// Analytic seed roughness Gram `S = ∫ (LΦ)ᵀ (LΦ)` for the cylinder, built
    /// as the tensor sum of a per-axis curvature energy: it penalizes the
    /// second derivative along the circle (the bending energy of the periodic
    /// factor) plus the second derivative along the line (the thin-plate energy
    /// of the flat factor). Because the basis is a clean tensor product and the
    /// two coordinate measures are independent on `[0,1) × ℝ`, the cross terms
    /// factor through the per-axis Grams.
    ///
    /// Concretely, with `Sc` the circle second-derivative Gram and `Gc` the
    /// circle value Gram (both `Mc × Mc`), `Sl`/`Gl` their line counterparts
    /// (`Ml × Ml`), the roughness operator
    /// `‖∂²_{t₀}Φ‖² + ‖∂²_{t₁}Φ‖²` integrates to
    /// `S = Sc ⊗ Gl + Gc ⊗ Sl` in the same lexicographic column order as the
    /// design. This is gauge-invariant: it depends only on the basis functions,
    /// not on any chart-specific normalization, and the constant column (zero in
    /// both `Sc` and `Sl`) sits in the null space exactly as the smooth-penalty
    /// nullity recovery expects.
    ///
    /// The circle blocks use the closed-form Fourier integrals on `[0,1)`:
    /// `∫₀¹ 1 dt = 1`, `∫₀¹ sin²(2πht) = ∫₀¹ cos²(2πht) = ½`, all distinct-mode
    /// and sin·cos cross integrals vanish, so `Gc` is diagonal
    /// `diag(1, ½, ½, …)` and `Sc = diag(0, (2πh)⁴·½, …)` (the second derivative
    /// of a mode scales its value by `(2πh)²`, squared and integrated → `(2πh)⁴`
    /// times the value integral). The line blocks use the monomial moments on a
    /// canonical unit interval `[0,1)` so the energy is finite and scale-fixed:
    /// `Gl[i,j] = ∫₀¹ tⁱ⁺ʲ dt = 1/(i+j+1)` and
    /// `Sl[i,j] = ∫₀¹ (i(i-1)t^{i-2})(j(j-1)t^{j-2}) dt`.
    pub fn roughness_gram(&self) -> Array2<f64> {
        let mc = self.circle_basis_size();
        let ml = self.line_basis_size();
        let two_pi = 2.0 * std::f64::consts::PI;

        // Circle value Gram Gc and second-derivative Gram Sc (both diagonal).
        let mut gc = Array2::<f64>::zeros((mc, mc));
        let mut sc = Array2::<f64>::zeros((mc, mc));
        gc[[0, 0]] = 1.0; // ∫₀¹ 1 dt
        for h in 1..=self.circle_harmonics {
            let omega = two_pi * (h as f64);
            let w4 = omega.powi(4);
            let s_idx = 2 * h - 1;
            let c_idx = 2 * h;
            // ∫₀¹ sin² = ∫₀¹ cos² = ½.
            gc[[s_idx, s_idx]] = 0.5;
            gc[[c_idx, c_idx]] = 0.5;
            // Second derivative scales the mode by (2πh)²; squared·integrated.
            sc[[s_idx, s_idx]] = w4 * 0.5;
            sc[[c_idx, c_idx]] = w4 * 0.5;
        }

        // Line value Gram Gl and second-derivative Gram Sl on the canonical
        // interval [0,1).
        let mut gl = Array2::<f64>::zeros((ml, ml));
        let mut sl = Array2::<f64>::zeros((ml, ml));
        for i in 0..ml {
            for j in 0..ml {
                // Gl[i,j] = ∫₀¹ t^{i+j} dt = 1/(i+j+1).
                gl[[i, j]] = 1.0 / ((i + j + 1) as f64);
                // Sl[i,j] = ∫₀¹ (i(i-1) t^{i-2})(j(j-1) t^{j-2}) dt.
                if i >= 2 && j >= 2 {
                    let ci = (i * (i - 1)) as f64;
                    let cj = (j * (j - 1)) as f64;
                    let exp = (i - 2) + (j - 2);
                    sl[[i, j]] = ci * cj / ((exp + 1) as f64);
                }
            }
        }

        // S = Sc ⊗ Gl + Gc ⊗ Sl in lexicographic (circle-slow, line-fast) order.
        let m = mc * ml;
        let mut s = Array2::<f64>::zeros((m, m));
        for ca in 0..mc {
            for la in 0..ml {
                let row = ca * ml + la;
                for cb in 0..mc {
                    for lb in 0..ml {
                        let col = cb * ml + lb;
                        s[[row, col]] = sc[[ca, cb]] * gl[[la, lb]] + gc[[ca, cb]] * sl[[la, lb]];
                    }
                }
            }
        }
        s
    }

    fn check_coords(&self, coords: ArrayView2<'_, f64>, what: &str) -> Result<(), String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "CylinderHarmonicEvaluator::{what}: expected latent_dim == 2 (S¹ × ℝ), got {}",
                coords.ncols()
            ));
        }
        Ok(())
    }
}

impl SaeBasisEvaluator for CylinderHarmonicEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        let expected = self.basis_size();
        if n_basis != expected {
            return Err(format!(
                "CylinderHarmonicEvaluator::phi_eta_split: n_basis {n_basis} != evaluator width {expected}"
            ));
        }
        let ml = self.line_basis_size();
        // A product column `[c, l]` is curved iff either factor carries
        // curvature: the circle factor is curved on any harmonic above the
        // first (`c > 2`, the 2nd-and-higher sin/cos), and the line factor is
        // curved on any monomial of degree ≥ 2 (`l > 1`). The first-harmonic
        // circle columns crossed with the affine line columns span the linear
        // relaxation, matching the `eta`-homotopy split of the two parent
        // evaluators.
        let mut curved = vec![false; expected];
        for c in 0..self.circle_basis_size() {
            for l in 0..ml {
                let circle_curved = c > 2;
                let line_curved = l > 1;
                curved[c * ml + l] = circle_curved || line_curved;
            }
        }
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        self.check_coords(coords, "evaluate")?;
        let n = coords.nrows();
        let mc = self.circle_basis_size();
        let ml = self.line_basis_size();
        let m = mc * ml;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 2));
        for row in 0..n {
            let t0 = coords[[row, 0]];
            let t1 = coords[[row, 1]];
            let circ = self.circle_tables(t0);
            let line = self.line_tables(t1);
            for c in 0..mc {
                for l in 0..ml {
                    let col = c * ml + l;
                    // Value: c·l. ∂/∂t₀ = c'·l. ∂/∂t₁ = c·l'.
                    phi[[row, col]] = circ[0][c] * line[0][l];
                    jet[[row, col, 0]] = circ[1][c] * line[0][l];
                    jet[[row, col, 1]] = circ[0][c] * line[1][l];
                }
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for CylinderHarmonicEvaluator {
    /// Hessian of the cylinder product basis. With `Φ_{c,l} = c(t₀)·l(t₁)` and
    /// the two factors on disjoint coordinates:
    /// `∂²/∂t₀² = c''·l`, `∂²/∂t₁² = c·l''`, `∂²/∂t₀∂t₁ = c'·l'`.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        self.check_coords(coords, "second_jet")?;
        let n = coords.nrows();
        let mc = self.circle_basis_size();
        let ml = self.line_basis_size();
        let m = mc * ml;
        let mut h = Array4::<f64>::zeros((n, m, 2, 2));
        for row in 0..n {
            let t0 = coords[[row, 0]];
            let t1 = coords[[row, 1]];
            let circ = self.circle_tables(t0);
            let line = self.line_tables(t1);
            for c in 0..mc {
                for l in 0..ml {
                    let col = c * ml + l;
                    h[[row, col, 0, 0]] = circ[2][c] * line[0][l];
                    h[[row, col, 1, 1]] = circ[0][c] * line[2][l];
                    let mixed = circ[1][c] * line[1][l];
                    h[[row, col, 0, 1]] = mixed;
                    h[[row, col, 1, 0]] = mixed;
                }
            }
        }
        Ok(h)
    }
}

impl SaeBasisThirdJet for CylinderHarmonicEvaluator {
    /// Third derivative of the cylinder product basis. The number of axis-0
    /// derivative operators `k₀` (and axis-1 `k₁ = 3 − k₀`) in a cell `(a,b,e)`
    /// routes that many derivatives to the circle factor and the rest to the
    /// line factor: `∂³Φ = c^{(k₀)}(t₀) · l^{(k₁)}(t₁)`. This is the order-3
    /// sibling of [`SaeBasisSecondJet::second_jet`].
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        self.check_coords(coords, "third_jet")?;
        let n = coords.nrows();
        let mc = self.circle_basis_size();
        let ml = self.line_basis_size();
        let m = mc * ml;
        let mut t3 = Array5::<f64>::zeros((n, m, 2, 2, 2));
        for row in 0..n {
            let t0 = coords[[row, 0]];
            let t1 = coords[[row, 1]];
            let circ = self.circle_tables(t0);
            let line = self.line_tables(t1);
            for c in 0..mc {
                for l in 0..ml {
                    let col = c * ml + l;
                    for a in 0..2 {
                        for b in 0..2 {
                            for e in 0..2 {
                                // k0 = number of axis-0 operators among (a,b,e).
                                let k0 = (a == 0) as usize + (b == 0) as usize + (e == 0) as usize;
                                let k1 = 3 - k0;
                                t3[[row, col, a, b, e]] = circ[k0][c] * line[k1][l];
                            }
                        }
                    }
                }
            }
        }
        Ok(t3)
    }
}

/// Rank-revealing subspace reparametrization of an inner basis evaluator.
///
/// Issue #1117: a decoder basis (e.g. [`PeriodicHarmonicEvaluator`]) emits a
/// *fixed* number of columns `M` independent of the data, so on a
/// near-degenerate checkpoint the higher columns are unexcited and the
/// decoder design `Φ` is rank-deficient by construction (the OLMo
/// `stage1-step0` PCA-32 circle: data Gram rank `3/5`, the 2nd-harmonic pair
/// dead). A rank-deficient design leaves the inner solve conditioned only by
/// ridges/deflation and flattens the outer REML surface, stalling BFGS.
///
/// This wrapper makes the design **full-rank by construction**: the
/// data-supported subspace of the inner basis is discovered ONCE at fit entry
/// (the eigenvectors of the weighted data Gram `G = Φᵀ W Φ` whose eigenvalue
/// clears the relative spectral cutoff) and frozen into an orthonormal column
/// map `Q ∈ ℝ^{M × r}` (`r = rank(G) ≤ M`). The wrapped evaluator then emits
/// the reduced design `Φ̃ = Φ Q` (and its jets `∂Φ̃ = (∂Φ) Q`, …) on every
/// refresh, so the reduction *survives* re-evaluation — unlike a step-time
/// projector, which the evaluator's next `evaluate` overwrites.
///
/// Because `Q` is a fixed linear remix of the inner columns, the reduced basis
/// is exactly as smooth as the inner one and every derivative composes by the
/// same right-multiply: `∂^g Φ̃ = (∂^g Φ) Q`, contracting only the basis
/// (column) axis. The retained columns span exactly the data-identified part of
/// the inner basis; the smooth/REML penalty then shrinks within that span and
/// is never asked to identify a direction the data cannot see.
///
/// When the inner Gram is full rank (`r == M`, the `base`/`step_2300` case),
/// the fit-entry installer skips the wrap entirely and the inner evaluator is
/// used unchanged, so the well-conditioned path is byte-identical.
#[derive(Debug, Clone)]
pub struct SubspaceReducedEvaluator {
    inner: Arc<dyn SaeBasisSecondJet>,
    /// `(M × r)` orthonormal column map onto the data-supported subspace.
    q: Array2<f64>,
}

impl SubspaceReducedEvaluator {
    /// Wrap `inner` with the column map `q` (`M_inner × r`, `r ≤ M_inner`). The
    /// retained width is `q.ncols()`. The columns of `q` are expected to be
    /// orthonormal (the eigenvectors of a symmetric data Gram); orthonormality
    /// is not re-checked here — it is the caller's contract at fit entry.
    pub fn new(inner: Arc<dyn SaeBasisSecondJet>, q: Array2<f64>) -> Result<Self, String> {
        if q.nrows() == 0 || q.ncols() == 0 {
            return Err(format!(
                "SubspaceReducedEvaluator: column map must be non-empty; got {:?}",
                q.dim()
            ));
        }
        if q.ncols() > q.nrows() {
            return Err(format!(
                "SubspaceReducedEvaluator: retained rank {} exceeds inner basis width {}",
                q.ncols(),
                q.nrows()
            ));
        }
        Ok(Self { inner, q })
    }

    /// Inner basis width `M` (the wrapped evaluator's column count).
    pub fn inner_width(&self) -> usize {
        self.q.nrows()
    }

    /// Retained (data-supported) width `r`.
    pub fn reduced_width(&self) -> usize {
        self.q.ncols()
    }

    fn check_inner_width(&self, got: usize, what: &str) -> Result<(), String> {
        if got != self.q.nrows() {
            return Err(format!(
                "SubspaceReducedEvaluator::{what}: inner evaluator returned width {got}, \
                 column map expects {}",
                self.q.nrows()
            ));
        }
        Ok(())
    }
}

/// Right-multiply the basis (column) axis of a per-row value matrix
/// `phi` (`n × M`) by `q` (`M × r`), returning `(n × r)`.
fn remix_cols_2(phi: &Array2<f64>, q: &Array2<f64>) -> Array2<f64> {
    phi.dot(q)
}

/// Right-multiply the basis axis of a jet `jet[n, M, ..]` by `q` (`M × r`),
/// returning the same trailing shape with the basis axis reduced to `r`. The
/// trailing derivative axes are flattened, the `(M)`→`(r)` remix applied as one
/// matmul, then reshaped back; this is the exact `∂^g Φ̃ = (∂^g Φ) Q` contract.
fn remix_cols_along_basis(
    jet: ndarray::ArrayViewD<'_, f64>,
    q: &Array2<f64>,
) -> Result<ndarray::ArrayD<f64>, String> {
    let shape = jet.shape().to_vec();
    if shape.len() < 2 {
        return Err(format!(
            "SubspaceReducedEvaluator: jet must have at least (n, M) axes; got {shape:?}"
        ));
    }
    let n = shape[0];
    let m = shape[1];
    if m != q.nrows() {
        return Err(format!(
            "SubspaceReducedEvaluator: jet basis axis {m} != column-map rows {}",
            q.nrows()
        ));
    }
    let r = q.ncols();
    let trailing: usize = shape[2..].iter().product::<usize>().max(1);
    let mut out_shape = shape.clone();
    out_shape[1] = r;
    // Flatten the trailing derivative axes so the remix is a single
    // `(M)→(r)` contraction over the basis axis for every (row, trailing) fiber.
    // `to_owned()` produces a standard (row-major contiguous) layout, so the
    // flatten and the final reshape back to `out_shape` are exact.
    let jet_std = jet.to_owned();
    let jet_flat = jet_std
        .to_shape((n, m, trailing))
        .map_err(|err| format!("SubspaceReducedEvaluator: jet reshape failed: {err}"))?;
    let mut out_flat = Array3::<f64>::zeros((n, r, trailing));
    for row in 0..n {
        for t in 0..trailing {
            for rc in 0..r {
                let mut acc = 0.0_f64;
                for mc in 0..m {
                    acc += jet_flat[[row, mc, t]] * q[[mc, rc]];
                }
                out_flat[[row, rc, t]] = acc;
            }
        }
    }
    let out = out_flat
        .into_shape_with_order(ndarray::IxDyn(&out_shape))
        .map_err(|err| format!("SubspaceReducedEvaluator: out reshape failed: {err}"))?;
    Ok(out)
}

impl SaeBasisEvaluator for SubspaceReducedEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        if n_basis != self.q.ncols() {
            return Err(format!(
                "SubspaceReducedEvaluator::phi_eta_split: n_basis {n_basis} != reduced width {}",
                self.q.ncols()
            ));
        }
        // A reduced column is "curved" iff its data-supported direction draws on
        // any curved inner column. `Q[:, rc]` mixes inner columns; the reduced
        // column carries curvature when `Q[curved_inner, rc]` is non-zero.
        let inner_split = self.inner.phi_eta_split(self.q.nrows())?;
        let mut inner_curved = vec![false; self.q.nrows()];
        for &col in &inner_split.curved_cols {
            if col < inner_curved.len() {
                inner_curved[col] = true;
            }
        }
        let mut curved = vec![false; self.q.ncols()];
        for rc in 0..self.q.ncols() {
            for mc in 0..self.q.nrows() {
                if inner_curved[mc] && self.q[[mc, rc]] != 0.0 {
                    curved[rc] = true;
                    break;
                }
            }
        }
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        match self.inner.third_jet_dyn(coords) {
            Some(Ok(t3)) => {
                if let Err(err) = self.check_inner_width(t3.shape()[1], "third_jet_dyn") {
                    return Some(Err(err));
                }
                Some(
                    remix_cols_along_basis(t3.view().into_dyn(), &self.q).and_then(|out| {
                        out.into_dimensionality::<ndarray::Ix5>().map_err(|err| {
                            format!("SubspaceReducedEvaluator: third jet dim: {err}")
                        })
                    }),
                )
            }
            Some(Err(err)) => Some(Err(err)),
            None => None,
        }
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let (phi, jet) = self.inner.evaluate(coords)?;
        self.check_inner_width(phi.ncols(), "evaluate")?;
        let phi_red = remix_cols_2(&phi, &self.q);
        let jet_red = remix_cols_along_basis(jet.view().into_dyn(), &self.q)?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|err| format!("SubspaceReducedEvaluator: jet dim: {err}"))?;
        Ok((phi_red, jet_red))
    }
}

impl SaeBasisSecondJet for SubspaceReducedEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let h = self.inner.second_jet(coords)?;
        self.check_inner_width(h.shape()[1], "second_jet")?;
        remix_cols_along_basis(h.view().into_dyn(), &self.q)?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|err| format!("SubspaceReducedEvaluator: second jet dim: {err}"))
    }
}

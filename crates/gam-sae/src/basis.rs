use ndarray::{Array2, Array3, Array4, Array5, ArrayView2};
use std::sync::Arc;

pub trait SaeBasisEvaluator: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String>;

    /// Evaluate `Φ` and its first-location jet DIRECTLY into caller-owned
    /// buffers, avoiding the fresh `(N, M)` Φ + `(N, M, d)` jet allocation
    /// [`Self::evaluate`] performs on every call.
    ///
    /// The inner Newton loop re-evaluates each atom's basis on every
    /// line-search trial (`crate::manifold::atom::SaeManifoldAtom::refresh_basis`);
    /// there the term already owns correctly-shaped Φ / jet arrays, so reusing
    /// them removes the per-trial allocation churn (multiple trials per Newton
    /// iteration under LM / backtracking). `phi` must be shaped
    /// `(coords.nrows(), M)` and `jet` `(coords.nrows(), M, coords.ncols())`
    /// for this evaluator's basis width `M`; implementations return a
    /// descriptive `Err` on any mismatch — the same shape guard `refresh_basis`
    /// previously applied to the freshly-allocated arrays.
    ///
    /// The default forwards to [`Self::evaluate`] and copies, so evaluators
    /// that have not specialized it stay correct at the cost of one extra copy;
    /// the hot evaluators override it to fill in place.
    fn evaluate_into(
        &self,
        phi: &mut Array2<f64>,
        jet: &mut Array3<f64>,
        coords: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let (new_phi, new_jet) = self.evaluate(coords)?;
        if new_phi.dim() != phi.dim() {
            return Err(format!(
                "SaeBasisEvaluator::evaluate_into: evaluator returned Φ {:?}, target buffer {:?}",
                new_phi.dim(),
                phi.dim()
            ));
        }
        if new_jet.dim() != jet.dim() {
            return Err(format!(
                "SaeBasisEvaluator::evaluate_into: evaluator returned jet {:?}, target buffer {:?}",
                new_jet.dim(),
                jet.dim()
            ));
        }
        phi.assign(&new_phi);
        jet.assign(&new_jet);
        Ok(())
    }

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

    /// Column split for the curvature homotopy `Phi_eta = [base, eta*curved]`.
    ///
    /// `base` columns are held FIXED as `eta` walks `0 → 1`; `curved` columns are
    /// scaled by `eta`. The `eta = 0` endpoint is therefore the *base-topology*
    /// relaxation — the atom evaluated on its base (η-invariant) columns only —
    /// NOT a linear/affine model: for the harmonic and sphere-chart bases the
    /// base block already carries extrinsic curvature (a first-harmonic
    /// `[sin, cos]` pair traces a circle, the sphere chart's `[x, y, z]` block
    /// traces the unit sphere). "Base" names "does not scale with η", never
    /// "curvature-free". See [`PhiEtaSplit`].
    ///
    /// The default is a flat (genuinely affine) monomial-style basis where every
    /// column is a base column. Curved atom evaluators override this with their
    /// topology-specific split; callers pass `n_basis` so the split is checked
    /// against the concrete design width currently being evaluated.
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        Ok(PhiEtaSplit::all_base(n_basis))
    }

    /// Per-factor basis sizes `(M₁, M₂)` of a `d = 2` TENSOR-PRODUCT evaluator
    /// whose fused basis is the Kronecker product of two factor bases in
    /// row-major column order `flat = j·M₂ + k` (the within-atom carve / #993
    /// convention). `M₁·M₂` equals the fused basis width. `None` for evaluators
    /// that are not a two-factor product (single-axis, sphere chart, monomial
    /// patch) — the within-atom functional-ANOVA carve is only defined on a
    /// genuine product manifold, so a `None` here is the honest "no factor
    /// split" signal, never a guess.
    fn factor_basis_sizes(&self) -> Option<(usize, usize)> {
        None
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

/// Curvature-homotopy column split `Phi_eta = [base, eta*curved]`.
///
/// `curved_cols` are the columns scaled by the dial `eta`; `base_cols` are the
/// columns held fixed across the whole walk. "Base" means "η-invariant", NOT
/// "linear/curvature-free": for the harmonic and sphere-chart bases the base
/// block already embeds curvature (first-harmonic `[sin, cos]`, the sphere
/// chart's `[x, y, z]`). The `eta = 0` endpoint is thus the base-topology
/// relaxation, not an affine/Eckart-Young linear model. The genuine
/// low-rank (Eckart-Young / PCA) certificate lives in
/// `crate::manifold::outer_objective::linear_span_anchor` and is a rank
/// ceiling that bounds every `eta`, independent of this split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiEtaSplit {
    pub base_cols: Vec<usize>,
    pub curved_cols: Vec<usize>,
}

impl PhiEtaSplit {
    pub fn all_base(n_basis: usize) -> Self {
        Self {
            base_cols: (0..n_basis).collect(),
            curved_cols: Vec::new(),
        }
    }

    fn from_curved_mask(mask: Vec<bool>) -> Self {
        let mut base_cols = Vec::new();
        let mut curved_cols = Vec::new();
        for (col, curved) in mask.into_iter().enumerate() {
            if curved {
                curved_cols.push(col);
            } else {
                base_cols.push(col);
            }
        }
        Self {
            base_cols,
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
    gam_terms::basis::monomial_exponents(dimension, max_total_degree)
        .iter()
        .map(|alpha| alpha.iter().sum::<usize>() <= 1)
        .collect()
}

fn duchon_effective_order_for_eta(
    centers: ArrayView2<'_, f64>,
    order: gam_terms::basis::DuchonNullspaceOrder,
) -> gam_terms::basis::DuchonNullspaceOrder {
    let mut effective = order;
    while effective != gam_terms::basis::DuchonNullspaceOrder::Zero
        && centers.nrows() <= duchon_polynomial_column_count(centers.ncols(), effective)
    {
        effective = match effective {
            gam_terms::basis::DuchonNullspaceOrder::Zero => {
                gam_terms::basis::DuchonNullspaceOrder::Zero
            }
            gam_terms::basis::DuchonNullspaceOrder::Linear => {
                gam_terms::basis::DuchonNullspaceOrder::Zero
            }
            gam_terms::basis::DuchonNullspaceOrder::Degree(2) => {
                gam_terms::basis::DuchonNullspaceOrder::Linear
            }
            gam_terms::basis::DuchonNullspaceOrder::Degree(k) => {
                gam_terms::basis::DuchonNullspaceOrder::Degree(k - 1)
            }
        };
    }
    effective
}

fn duchon_polynomial_column_count(
    dimension: usize,
    order: gam_terms::basis::DuchonNullspaceOrder,
) -> usize {
    match order {
        gam_terms::basis::DuchonNullspaceOrder::Zero => 1,
        gam_terms::basis::DuchonNullspaceOrder::Linear => dimension + 1,
        gam_terms::basis::DuchonNullspaceOrder::Degree(degree) => {
            gam_terms::basis::monomial_exponents(dimension, degree).len()
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
/// [`PeriodicHarmonicEvaluator::new`] accepts the **total odd basis width**
/// `M`, not the number of non-constant harmonics. It emits
/// `[1, sin(2π·1·t), cos(2π·1·t), …, sin(2π·H·t), cos(2π·H·t)]`, where
/// `H = (M − 1) / 2`. Thus `new(5)` means `M = 5` (two harmonics), while
/// `new(11)` means `M = 11` (five harmonics). The latent must have
/// `latent_dim == 1`.
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
        // Base (η-invariant) columns: constant `1` plus the FIRST-harmonic
        // `[sin 2πt, cos 2πt]`. These are not linear — the first-harmonic pair
        // already traces the unit circle — but they define the base topology the
        // homotopy relaxes onto; only harmonics ≥ 2 scale with `eta`.
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
        // Single source of truth: allocate the correctly-shaped buffers and let
        // the in-place path fill them, so `evaluate` and `evaluate_into` can
        // never numerically diverge.
        let n = coords.nrows();
        let m = self.num_basis;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 1));
        self.evaluate_into(&mut phi, &mut jet, coords)?;
        Ok((phi, jet))
    }

    fn evaluate_into(
        &self,
        phi: &mut Array2<f64>,
        jet: &mut Array3<f64>,
        coords: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        if phi.dim() != (n, m) {
            return Err(format!(
                "PeriodicHarmonicEvaluator::evaluate_into: Φ buffer {:?} != ({n}, {m})",
                phi.dim()
            ));
        }
        if jet.dim() != (n, m, 1) {
            return Err(format!(
                "PeriodicHarmonicEvaluator::evaluate_into: jet buffer {:?} != ({n}, {m}, 1)",
                jet.dim()
            ));
        }
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        // The constant column carries a zero jet and is never written in the
        // harmonic loop below, so clear both buffers to erase any stale
        // (reused-workspace) contents before filling.
        phi.fill(0.0);
        jet.fill(0.0);
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
        Ok(())
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

/// One real spherical-harmonic column `Y_l^m`, fixed at construction.
///
/// Each column factors as `R_{l,m}(lat) · T_m(lon)` — a pure-latitude
/// amplitude times a pure-longitude phase — so its full jet in `(lat, lon)` is
/// the separable outer product of the two per-axis derivative tables. The
/// latitude amplitude is `R_{l,m}(lat) = N_{l,m} · cos(lat)^{|m|} · Q(sin lat)`,
/// where `Q = d^{|m|}/du^{|m|} P_l(u)` is the `|m|`-th `u`-derivative of the
/// Legendre polynomial (the associated-Legendre function's polynomial part) and
/// `N_{l,m}` is the orthonormalization constant. The `(-1)^m` Condon–Shortley
/// phase is intentionally dropped: it is a per-column sign the decoder absorbs
/// and it has no effect on the span, orthonormality, or reconstruction.
#[derive(Debug, Clone)]
struct SphHarmonicColumn {
    /// Spherical-harmonic degree `l`.
    degree: usize,
    /// Signed order `m ∈ [-l, l]`: `m ≥ 0` uses `cos(m·lon)`, `m < 0` uses
    /// `sin(|m|·lon)`.
    m: i64,
    /// `|m|`, the `cos(lat)` power and the Legendre `u`-derivative order.
    am: usize,
    /// Orthonormalization constant `N_{l,m}` (includes the `√2` for `m ≠ 0`).
    norm: f64,
    /// `Q = d^{|m|}/du^{|m|} P_l(u)`, ascending powers of `u = sin(lat)`.
    assoc: Vec<f64>,
    /// `true` iff `l ≥ 2` — the curved (η-dialed) refinement above the base
    /// monopole+dipole sphere embedding.
    curved: bool,
}

/// Real orthonormal spherical-harmonic evaluator on `S^2`, charted by
/// `(lat, lon)` with `x = cos(lat)cos(lon)`, `y = cos(lat)sin(lon)`,
/// `z = sin(lat)`.
///
/// This is the rotation-covariant basis the fixed 7-column
/// the removed fixed seven-column `(lat, lon)` chart was not: its columns are
/// the `(degree+1)²` real
/// spherical harmonics `Y_l^m`, `l = 0..=degree`, `m = -l..=l`, an orthonormal
/// basis for every band-limited field on the sphere. The degree-2 chart spans
/// only `[1, x, y, z, xy, yz, xz]` — the monopole, the dipole, and *three of the
/// five* quadrupoles — so it cannot represent `x²−y²`, `3z²−1`, or any `l ≥ 3`
/// content; a genuinely sphere-class field with higher-degree structure is
/// out of its span. This evaluator closes that resolution gap, with the working
/// degree chosen by [`select_spherical_harmonic_degree`] under the same
/// spectral-noise-floor bandwidth doctrine the torus/circle already use.
///
/// **Smoothness / poles.** Every column is a polynomial in `sin(lat)`,
/// `cos(lat)`, `sin(m·lon)`, `cos(m·lon)`, all entire, so the map and its jets
/// are globally `C^∞` in `(lat, lon)` — latitude is never clamped here. The
/// pole gauge degeneracy is intrinsic to the lat/lon chart, not this basis: at
/// `cos(lat) = 0` every `m ≠ 0` harmonic and its longitude derivative vanish, so
/// all longitudes collapse to one physical point. That is handled exactly where
/// [`AmbientSphereHarmonicEvaluator`] handles it — by the retraction / tangent projection
/// enforcing the `lat ∈ [-π/2, π/2]` box and the pole-seam seeding (issue
/// #1890) — never by truncating a derivative.
#[derive(Debug, Clone)]
pub struct SphericalHarmonicEvaluator {
    degree: usize,
    columns: Vec<SphHarmonicColumn>,
}

/// Spectral identity of one real spherical-harmonic column.
///
/// The evaluator's public column order is triangular in `(degree, order)`:
/// degree `l` contributes orders `-l..=l`.  Carrying this identity beside the
/// analytic evaluator lets quotient constructions restrict the cover's exact
/// eigenspaces without re-deriving indexing conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalHarmonicMode {
    pub degree: usize,
    pub order: i64,
    /// Positive-Laplacian eigenvalue `l(l+1)` on the unit sphere.
    pub laplace_eigenvalue: f64,
    /// Diagonal `L²(S²)` Gram entry.  The real harmonics are orthonormal.
    pub l2_gram_weight: f64,
}

/// Legendre polynomials `P_0..=P_degree` as ascending-power coefficient vectors,
/// via the exact three-term recurrence `l·P_l = (2l-1)·u·P_{l-1} − (l-1)·P_{l-2}`.
fn legendre_polynomials(degree: usize) -> Vec<Vec<f64>> {
    let mut polys: Vec<Vec<f64>> = Vec::with_capacity(degree + 1);
    polys.push(vec![1.0]);
    if degree >= 1 {
        polys.push(vec![0.0, 1.0]);
    }
    for l in 2..=degree {
        let prev = &polys[l - 1];
        let prev2 = &polys[l - 2];
        let mut next = vec![0.0_f64; l + 1];
        // (2l-1)·u·P_{l-1}
        for (k, &coeff) in prev.iter().enumerate() {
            next[k + 1] += (2 * l - 1) as f64 * coeff;
        }
        // −(l-1)·P_{l-2}
        for (k, &coeff) in prev2.iter().enumerate() {
            next[k] -= (l - 1) as f64 * coeff;
        }
        let inv = 1.0 / l as f64;
        for c in next.iter_mut() {
            *c *= inv;
        }
        polys.push(next);
    }
    polys
}

/// `order`-th derivative of an ascending-power polynomial (exact).
fn polynomial_derivative(coeffs: &[f64], order: usize) -> Vec<f64> {
    let mut c = coeffs.to_vec();
    for _ in 0..order {
        if c.len() <= 1 {
            return vec![0.0];
        }
        c = (1..c.len()).map(|k| k as f64 * c[k]).collect();
    }
    c
}

/// Orthonormalization constant `N_{l,m} = √((2l+1)/(4π) · (l−|m|)!/(l+|m|)!)`,
/// times `√2` for `m ≠ 0` (the real-harmonic combination of `±m`).
fn spherical_harmonic_norm(l: usize, m: i64) -> f64 {
    let am = m.unsigned_abs() as usize;
    let mut ratio = 1.0_f64; // (l-am)! / (l+am)!
    for k in (l - am + 1)..=(l + am) {
        ratio /= k as f64;
    }
    let base = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI) * ratio).sqrt();
    if m == 0 {
        base
    } else {
        base * std::f64::consts::SQRT_2
    }
}

/// One-dimensional order-3 jet `[f, f', f'', f''']` product (Leibniz).
#[inline]
fn sph_jet_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0],
        a[1] * b[0] + a[0] * b[1],
        a[2] * b[0] + 2.0 * a[1] * b[1] + a[0] * b[2],
        a[3] * b[0] + 3.0 * a[2] * b[1] + 3.0 * a[1] * b[2] + a[0] * b[3],
    ]
}

impl SphericalHarmonicEvaluator {
    pub fn new(degree: usize) -> Result<Self, String> {
        let side = degree.checked_add(1).ok_or_else(|| {
            "SphericalHarmonicEvaluator: basis width overflowed usize".to_string()
        })?;
        let basis_size = side.checked_mul(side).ok_or_else(|| {
            "SphericalHarmonicEvaluator: basis width overflowed usize".to_string()
        })?;
        let polys = legendre_polynomials(degree);
        let mut columns = Vec::with_capacity(basis_size);
        for l in 0..=degree {
            for m in -(l as i64)..=(l as i64) {
                let am = m.unsigned_abs() as usize;
                columns.push(SphHarmonicColumn {
                    degree: l,
                    m,
                    am,
                    norm: spherical_harmonic_norm(l, m),
                    assoc: polynomial_derivative(&polys[l], am),
                    curved: l >= 2,
                });
            }
        }
        Ok(Self { degree, columns })
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn basis_size(&self) -> usize {
        self.columns.len()
    }

    /// Exact `(l,m)` and Laplace-eigenvalue metadata in evaluator column order.
    pub fn spectral_modes(&self) -> Vec<SphericalHarmonicMode> {
        self.columns
            .iter()
            .map(|column| {
                let degree = column.degree as f64;
                SphericalHarmonicMode {
                    degree: column.degree,
                    order: column.m,
                    laplace_eigenvalue: degree * (degree + 1.0),
                    l2_gram_weight: 1.0,
                }
            })
            .collect()
    }

    /// Latitude amplitude jet `[R, R', R'', R''']` of a column via order-3 jet
    /// arithmetic in `lat`: `R = N · cos(lat)^{|m|} · Q(sin lat)`.
    fn lat_table(&self, col: &SphHarmonicColumn, lat: f64) -> [f64; 4] {
        let (s, c) = lat.sin_cos();
        let slat = [s, c, -s, -c];
        let clat = [c, -s, -c, s];
        let mut pow = [1.0, 0.0, 0.0, 0.0];
        for _ in 0..col.am {
            pow = sph_jet_mul(pow, clat);
        }
        // Horner evaluation of Q at slat.
        let mut acc = [0.0, 0.0, 0.0, 0.0];
        for &coeff in col.assoc.iter().rev() {
            acc = sph_jet_mul(acc, slat);
            acc[0] += coeff;
        }
        let mut r = sph_jet_mul(pow, acc);
        for value in r.iter_mut() {
            *value *= col.norm;
        }
        r
    }

    /// Longitude phase jet `[T, T', T'', T''']`: `cos(m·lon)` for `m ≥ 0`,
    /// `sin(|m|·lon)` for `m < 0`.
    fn lon_table(m: i64, lon: f64) -> [f64; 4] {
        if m == 0 {
            return [1.0, 0.0, 0.0, 0.0];
        }
        let mf = m.unsigned_abs() as f64;
        let (s, c) = (mf * lon).sin_cos();
        if m > 0 {
            [c, -mf * s, -mf * mf * c, mf * mf * mf * s]
        } else {
            [s, mf * c, -mf * mf * s, -mf * mf * mf * c]
        }
    }

    fn check_coords(&self, coords: ArrayView2<'_, f64>, what: &str) -> Result<(), String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "SphericalHarmonicEvaluator::{what}: expected latent_dim == 2 (lat, lon), got {}",
                coords.ncols()
            ));
        }
        Ok(())
    }
}

impl SaeBasisEvaluator for SphericalHarmonicEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        let expected = self.basis_size();
        if n_basis != expected {
            return Err(format!(
                "SphericalHarmonicEvaluator::phi_eta_split: n_basis {n_basis} != evaluator width {expected}"
            ));
        }
        // Base (η-invariant) block: monopole + dipole `l ≤ 1` (the base sphere
        // embedding). The quadrupole and higher harmonics `l ≥ 2` are the
        // η-dialed curvature refinement, mirroring the fixed sphere chart.
        let curved = self
            .columns
            .iter()
            .map(|col| col.curved)
            .collect::<Vec<_>>();
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    /// The `(l, m)` triangular indexing is not a clean per-axis tensor product.
    fn factor_basis_sizes(&self) -> Option<(usize, usize)> {
        None
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
        let m = self.basis_size();
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 2));
        for row in 0..n {
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            for (col_idx, col) in self.columns.iter().enumerate() {
                let r = self.lat_table(col, lat);
                let t = Self::lon_table(col.m, lon);
                phi[[row, col_idx]] = r[0] * t[0];
                jet[[row, col_idx, 0]] = r[1] * t[0];
                jet[[row, col_idx, 1]] = r[0] * t[1];
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for SphericalHarmonicEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        self.check_coords(coords, "second_jet")?;
        let n = coords.nrows();
        let m = self.basis_size();
        let mut h = Array4::<f64>::zeros((n, m, 2, 2));
        for row in 0..n {
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            for (col_idx, col) in self.columns.iter().enumerate() {
                let r = self.lat_table(col, lat);
                let t = Self::lon_table(col.m, lon);
                // Separable column: ∂^{p+q}/∂lat^p ∂lon^q = R^{(p)} · T^{(q)}.
                h[[row, col_idx, 0, 0]] = r[2] * t[0];
                h[[row, col_idx, 0, 1]] = r[1] * t[1];
                h[[row, col_idx, 1, 0]] = r[1] * t[1];
                h[[row, col_idx, 1, 1]] = r[0] * t[2];
            }
        }
        Ok(h)
    }
}

impl SaeBasisThirdJet for SphericalHarmonicEvaluator {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        self.check_coords(coords, "third_jet")?;
        let n = coords.nrows();
        let m = self.basis_size();
        let mut t3 = Array5::<f64>::zeros((n, m, 2, 2, 2));
        for row in 0..n {
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            for (col_idx, col) in self.columns.iter().enumerate() {
                let r = self.lat_table(col, lat);
                let t = Self::lon_table(col.m, lon);
                for axis_a in 0..2 {
                    for axis_b in 0..2 {
                        for axis_c in 0..2 {
                            // Separable: the mixed derivative depends only on how
                            // many of the three operators are latitude (axis 0).
                            let n_lat = (axis_a == 0) as usize
                                + (axis_b == 0) as usize
                                + (axis_c == 0) as usize;
                            t3[[row, col_idx, axis_a, axis_b, axis_c]] = r[n_lat] * t[3 - n_lat];
                        }
                    }
                }
            }
        }
        Ok(t3)
    }
}

/// One column of [`AmbientSphereHarmonicEvaluator`], pre-differentiated in `z`.
#[derive(Debug, Clone)]
struct AmbientSphereColumn {
    /// Spherical-harmonic degree `l`.
    degree: usize,
    /// Signed order `m`. `m >= 0` selects the `Re[w^{|m|}]` phase, `m < 0` the
    /// `Im[w^{|m|}]` phase — the same convention as the chart evaluator's
    /// `cos(m·lon)` / `sin(|m|·lon)`.
    m: i64,
    /// `|m|`, the power of `w = x + iy` this column carries.
    am: usize,
    /// Orthonormalization constant `N_{l,m}` (includes the `√2` for `m != 0`).
    norm: f64,
    /// `d^r/dz^r Q_l^{|m|}` for `r = 0..=3`, ascending-power coefficients.
    /// Third order is the deepest jet any consumer asks for, so all four are
    /// materialized once at construction and the hot path never differentiates.
    assoc_derivatives: [Vec<f64>; 4],
    /// `true` iff `l >= 2` — the curved (η-dialed) refinement above the base
    /// monopole+dipole embedding, matching [`SphericalHarmonicEvaluator`].
    curved: bool,
}

/// `Re[w^k]` and `Im[w^k]` for `w = x + iy`, by direct recurrence.
#[inline]
fn ambient_complex_power(x: f64, y: f64, k: usize) -> (f64, f64) {
    let (mut re, mut im) = (1.0_f64, 0.0_f64);
    for _ in 0..k {
        let next_re = re * x - im * y;
        let next_im = re * y + im * x;
        re = next_re;
        im = next_im;
    }
    (re, im)
}

/// Horner evaluation of an ascending-power coefficient vector.
#[inline]
fn ambient_poly_eval(coefficients: &[f64], z: f64) -> f64 {
    let mut acc = 0.0_f64;
    for &coefficient in coefficients.iter().rev() {
        acc = acc * z + coefficient;
    }
    acc
}

/// `∂^p_x ∂^q_y A_m(x, y)` where `A_m = Re[w^{|m|}]` for `m >= 0` and
/// `Im[w^{|m|}]` for `m < 0`.
///
/// One identity covers every order: `∂^p_x ∂^q_y w^a = i^q · a^{↓(p+q)} ·
/// w^{a−p−q}`, because `∂_x w = 1` and `∂_y w = i`. So the derivative is a
/// falling factorial, a lower power of `w`, and a rotation by `q` quarter
/// turns — no case analysis per jet order, and exact at every point including
/// the poles, where a `(lat, lon)` chart's longitude derivative collapses.
#[inline]
fn ambient_angular_partial(am: usize, sine_phase: bool, x: f64, y: f64, p: usize, q: usize) -> f64 {
    let order = p + q;
    if order > am {
        // `w^a` is a polynomial of degree `a`; past that every partial is zero.
        return 0.0;
    }
    let mut falling = 1.0_f64;
    for step in 0..order {
        falling *= (am - step) as f64;
    }
    let (re, im) = ambient_complex_power(x, y, am - order);
    // Multiply by `i^q`: rotate the pair by `q` quarter turns.
    let (re, im) = match q % 4 {
        0 => (re, im),
        1 => (-im, re),
        2 => (-re, -im),
        _ => (im, -re),
    };
    falling * if sine_phase { im } else { re }
}

/// Real spherical harmonics on `S²` in **ambient Cartesian coordinates**
/// `u = (x, y, z)` — the pole-free sibling of [`SphericalHarmonicEvaluator`].
///
/// # Why ambient
///
/// `S²` admits no global 2-D chart, so every `(lat, lon)` parameterization pays
/// for that at the poles, in four distinct ways:
///
///   * **The pole is a wall, not a point.** A `lat ∈ [-π/2, π/2]` latent is an
///     `Interval`, whose retraction clamps and whose tangent projection zeroes
///     the outward velocity at the bound. On the sphere the pole is an ordinary
///     interior point you walk straight through.
///   * **The trust-region metric is wrong by `cos²(lat)`.** The round metric is
///     `dlat² + cos²(lat)·dlon²`, but a product latent's metric weights are
///     per-axis constants, so a fixed coordinate step is a large geodesic move
///     at the equator and a vanishing one near a pole.
///   * **Longitude is gauge at the poles.** `cos(lat) = 0` collapses every
///     longitude to one physical point and kills the longitude jet, leaving a
///     singular `H_tt` block that a coordinate prior can still push around.
///   * **A fixed low-degree chart block need not be rotation-covariant.**
///
/// Parameterizing by the ambient unit vector removes all four. The latent
/// manifold is [`gam_terms::latent::LatentManifold::Sphere`], whose retraction
/// `(u+ξ)/‖u+ξ‖` is globally smooth with no cut and no boundary, whose tangent
/// projection `v − (u·v)u` is exact, and whose uniform ambient metric restricts
/// to *precisely* the round metric on the tangent space. Rotation covariance is
/// automatic: each degree-`l` block spans a full `SO(3)` irrep.
///
/// # The same function space, exactly
///
/// The columns are the identical real harmonics, not an approximation of them.
/// Writing `w = x + iy`, the chart's longitude factor times its `cos(lat)`
/// power is a *polynomial*:
///
/// ```text
/// cos(lat)^{|m|} · cos(|m|·lon) = Re[w^{|m|}]
/// cos(lat)^{|m|} · sin(|m|·lon) = Im[w^{|m|}]
/// ```
///
/// and the Legendre factor `Q_l^{|m|}(sin lat)` is just `Q_l^{|m|}(z)`. So
/// column `(l, m)` is `N_{l,m} · Q_l^{|m|}(z) · A_m(x, y)`, a polynomial in
/// `(x, y, z)` whose derivatives of every order are finite EVERYWHERE, poles
/// included. `ambient_sphere_matches_chart_on_the_sphere` pins that equality
/// against [`SphericalHarmonicEvaluator`] to 1e-12, so this evaluator inherits
/// its orthonormality and spectral metadata rather than re-deriving them.
///
/// # Ambient jets are deliberately unprojected
///
/// The polynomial is the harmonic ambient extension off the sphere, so its
/// ambient gradient carries a radial component that is not tangential data.
/// Removing it is the *manifold's* job, not the basis's:
/// `LatentManifold::project_to_tangent` and `riemannian_hessian_matrix` (with
/// its `add_normal_pinning`) do exactly that, and they are the single authority
/// for it. Projecting here as well would double-apply the correction. The jets
/// below are therefore the raw polynomial derivatives.
///
/// Coordinates are *not* required to be exactly unit-norm: line-search trials
/// legitimately evaluate slightly off the sphere before the retraction pulls
/// them back, and the polynomial is well-defined there. Only on `‖u‖ = 1` do
/// the columns coincide with the orthonormal harmonics.
#[derive(Debug, Clone)]
pub struct AmbientSphereHarmonicEvaluator {
    degree: usize,
    columns: Vec<AmbientSphereColumn>,
}

impl AmbientSphereHarmonicEvaluator {
    pub fn new(degree: usize) -> Result<Self, String> {
        let side = degree.checked_add(1).ok_or_else(|| {
            "AmbientSphereHarmonicEvaluator: basis width overflowed usize".to_string()
        })?;
        let basis_size = side.checked_mul(side).ok_or_else(|| {
            "AmbientSphereHarmonicEvaluator: basis width overflowed usize".to_string()
        })?;
        let polys = legendre_polynomials(degree);
        let mut columns = Vec::with_capacity(basis_size);
        for l in 0..=degree {
            for m in -(l as i64)..=(l as i64) {
                let am = m.unsigned_abs() as usize;
                let assoc = polynomial_derivative(&polys[l], am);
                columns.push(AmbientSphereColumn {
                    degree: l,
                    m,
                    am,
                    norm: spherical_harmonic_norm(l, m),
                    assoc_derivatives: [
                        polynomial_derivative(&assoc, 0),
                        polynomial_derivative(&assoc, 1),
                        polynomial_derivative(&assoc, 2),
                        polynomial_derivative(&assoc, 3),
                    ],
                    curved: l >= 2,
                });
            }
        }
        Ok(Self { degree, columns })
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn basis_size(&self) -> usize {
        self.columns.len()
    }

    /// Upper bound on `|∂^g Φ_col|` over the unit sphere, for EVERY column and
    /// every mixed partial of total order `g`: `column_jet_bound() · degree^g`.
    ///
    /// Derived, not tabulated. Each column is `N · Q(z) · A_m(x, y)`, and `Q`
    /// and `A` depend on DISJOINT variables, so a mixed partial factors exactly
    /// with no Leibniz cross-terms:
    ///
    /// ```text
    ///   ∂^p_x ∂^q_y ∂^r_z [N · Q(z) · A_m(x,y)]  =  N · Q^{(r)}(z) · ∂^p_x ∂^q_y A_m
    /// ```
    ///
    /// On `‖u‖ = 1` both factors are bounded by a falling factorial capped at the
    /// degree: `|Q^{(r)}| ≤ ‖Q‖₁ · degree^r` since every power in `Q` is at most
    /// `degree`, and `|∂^p_x ∂^q_y A_m| ≤ (|m|)^{p+q} ≤ degree^{p+q}` from the
    /// identity `∂^p_x ∂^q_y w^a = i^q a^{↓(p+q)} w^{a−p−q}` with `|w| ≤ 1`.
    /// Multiplying gives `N · ‖Q‖₁ · degree^g`, independent of the point — so the
    /// bound is global and needs no chart region.
    pub fn column_jet_bound(&self) -> f64 {
        self.columns
            .iter()
            .map(|column| {
                let coefficient_norm: f64 = column.assoc_derivatives[0]
                    .iter()
                    .map(|coefficient| coefficient.abs())
                    .sum();
                column.norm * coefficient_norm
            })
            .fold(0.0_f64, f64::max)
    }

    /// Exact `(l, m)` and Laplace-eigenvalue metadata in evaluator column order,
    /// in the same layout and units as [`SphericalHarmonicEvaluator`] — the two
    /// evaluators are the same basis in different coordinates, so a quotient
    /// construction can consume either cover with one character table.
    pub fn spectral_modes(&self) -> Vec<SphericalHarmonicMode> {
        self.columns
            .iter()
            .map(|column| {
                let degree = column.degree as f64;
                SphericalHarmonicMode {
                    degree: column.degree,
                    order: column.m,
                    laplace_eigenvalue: degree * (degree + 1.0),
                    l2_gram_weight: 1.0,
                }
            })
            .collect()
    }

    /// The mixed partial `∂^p_x ∂^q_y ∂^r_z` of one column at `(x, y, z)`.
    ///
    /// The column is separable as `N · Q^{(r)}(z) · ∂^p_x ∂^q_y A_m(x, y)`, so
    /// every jet order routes through this one function and no derivative rule
    /// is written twice.
    #[inline]
    fn partial(
        &self,
        column: &AmbientSphereColumn,
        x: f64,
        y: f64,
        z: f64,
        p: usize,
        q: usize,
        r: usize,
    ) -> f64 {
        // `r` counts the z-axes among at most three requested axes, so it is
        // structurally in `0..=3` and indexes the materialized derivatives
        // directly; a wider jet order would be a caller bug and panics here.
        let radial = ambient_poly_eval(&column.assoc_derivatives[r], z);
        if radial == 0.0 {
            return 0.0;
        }
        let angular = ambient_angular_partial(column.am, column.m < 0, x, y, p, q);
        column.norm * radial * angular
    }

    /// Count how many of the requested axes are `x`, `y`, and `z`.
    #[inline]
    fn axis_counts(axes: &[usize]) -> (usize, usize, usize) {
        let mut counts = (0_usize, 0_usize, 0_usize);
        for &axis in axes {
            match axis {
                0 => counts.0 += 1,
                1 => counts.1 += 1,
                _ => counts.2 += 1,
            }
        }
        counts
    }

    fn check_coords(&self, coords: ArrayView2<'_, f64>, what: &str) -> Result<(), String> {
        if coords.ncols() != 3 {
            return Err(format!(
                "AmbientSphereHarmonicEvaluator::{what}: expected ambient dim == 3 (x, y, z), got {}",
                coords.ncols()
            ));
        }
        Ok(())
    }
}

impl SaeBasisEvaluator for AmbientSphereHarmonicEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        let expected = self.basis_size();
        if n_basis != expected {
            return Err(format!(
                "AmbientSphereHarmonicEvaluator::phi_eta_split: n_basis {n_basis} != evaluator width {expected}"
            ));
        }
        // Base (η-invariant) block: monopole + dipole `l <= 1`, i.e. the base
        // sphere embedding itself. `l >= 2` is the η-dialed refinement, matching
        // the chart evaluator so the homotopy means the same thing either way.
        let curved = self
            .columns
            .iter()
            .map(|column| column.curved)
            .collect::<Vec<_>>();
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    /// The `(l, m)` triangular indexing is not a clean per-axis tensor product.
    fn factor_basis_sizes(&self) -> Option<(usize, usize)> {
        None
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
        let m = self.basis_size();
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 3));
        for row in 0..n {
            let (x, y, z) = (coords[[row, 0]], coords[[row, 1]], coords[[row, 2]]);
            for (col_idx, column) in self.columns.iter().enumerate() {
                phi[[row, col_idx]] = self.partial(column, x, y, z, 0, 0, 0);
                jet[[row, col_idx, 0]] = self.partial(column, x, y, z, 1, 0, 0);
                jet[[row, col_idx, 1]] = self.partial(column, x, y, z, 0, 1, 0);
                jet[[row, col_idx, 2]] = self.partial(column, x, y, z, 0, 0, 1);
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for AmbientSphereHarmonicEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        self.check_coords(coords, "second_jet")?;
        let n = coords.nrows();
        let m = self.basis_size();
        let mut h = Array4::<f64>::zeros((n, m, 3, 3));
        for row in 0..n {
            let (x, y, z) = (coords[[row, 0]], coords[[row, 1]], coords[[row, 2]]);
            for (col_idx, column) in self.columns.iter().enumerate() {
                for axis_a in 0..3 {
                    for axis_b in 0..3 {
                        let (p, q, r) = Self::axis_counts(&[axis_a, axis_b]);
                        h[[row, col_idx, axis_a, axis_b]] =
                            self.partial(column, x, y, z, p, q, r);
                    }
                }
            }
        }
        Ok(h)
    }
}

impl SaeBasisThirdJet for AmbientSphereHarmonicEvaluator {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        self.check_coords(coords, "third_jet")?;
        let n = coords.nrows();
        let m = self.basis_size();
        let mut t3 = Array5::<f64>::zeros((n, m, 3, 3, 3));
        for row in 0..n {
            let (x, y, z) = (coords[[row, 0]], coords[[row, 1]], coords[[row, 2]]);
            for (col_idx, column) in self.columns.iter().enumerate() {
                for axis_a in 0..3 {
                    for axis_b in 0..3 {
                        for axis_c in 0..3 {
                            let (p, q, r) = Self::axis_counts(&[axis_a, axis_b, axis_c]);
                            t3[[row, col_idx, axis_a, axis_b, axis_c]] =
                                self.partial(column, x, y, z, p, q, r);
                        }
                    }
                }
            }
        }
        Ok(t3)
    }
}

/// Tensor-product periodic harmonic evaluator for a `d`-dimensional torus
/// `T^d = (S^1)^d`. The basis is the tensor product over each axis of the
/// 1-D circle basis, stored **sine-first** within each harmonic (matching the
/// `evaluate` layout below, `s_idx = 2h-1`, `c_idx = 2h`, and the
/// `PeriodicHarmonicEvaluator` convention):
/// `[1, sin(2π·1·t), cos(2π·1·t), …, sin(2π·H·t), cos(2π·H·t)]`
/// (each axis contributes `2H+1` factors, so the total basis size is
/// `(2H+1)^d`). The latent coords are angular phases in `[0, 1)` (consistent
/// with the periodic 1-D atoms).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealHarmonicComponent {
    Constant,
    Sine { harmonic: usize },
    Cosine { harmonic: usize },
}

impl RealHarmonicComponent {
    pub fn harmonic(self) -> usize {
        match self {
            Self::Constant => 0,
            Self::Sine { harmonic } | Self::Cosine { harmonic } => harmonic,
        }
    }

    /// Sign under reflection of the corresponding circle coordinate.
    pub fn reflection_sign(self) -> i8 {
        match self {
            Self::Sine { .. } => -1,
            Self::Constant | Self::Cosine { .. } => 1,
        }
    }
}

/// Spectral identity of one tensor-product real Fourier column.
#[derive(Debug, Clone, PartialEq)]
pub struct TorusHarmonicMode {
    pub components: Vec<RealHarmonicComponent>,
    /// Positive-Laplacian eigenvalue `sum_a k_a^2`, up to the common `(2π)^2`
    /// chart-unit scale absorbed by the smoothing strength.
    pub laplace_eigenvalue: f64,
    /// Diagonal Gram entry under normalized Haar measure on the torus.  Each
    /// sine/cosine factor contributes `1/2`; constant factors contribute one.
    pub l2_gram_weight: f64,
}

#[derive(Debug, Clone)]
pub struct TorusHarmonicEvaluator {
    latent_dim: usize,
    num_harmonics: usize,
    axis_basis_size: usize,
    basis_size: usize,
}

impl TorusHarmonicEvaluator {
    pub fn new(latent_dim: usize, num_harmonics: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("TorusHarmonicEvaluator requires latent_dim >= 1".to_string());
        }
        if num_harmonics == 0 {
            return Err("TorusHarmonicEvaluator requires num_harmonics >= 1".to_string());
        }
        let axis_width = num_harmonics
            .checked_mul(2)
            .and_then(|twice| twice.checked_add(1))
            .ok_or_else(|| {
                "TorusHarmonicEvaluator: per-axis basis width overflowed usize".to_string()
            })?;
        let basis_size = (0..latent_dim)
            .try_fold(1usize, |width, _| width.checked_mul(axis_width))
            .ok_or_else(|| {
                "TorusHarmonicEvaluator: tensor basis width overflowed usize".to_string()
            })?;
        Ok(Self {
            latent_dim,
            num_harmonics,
            axis_basis_size: axis_width,
            basis_size,
        })
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn num_harmonics(&self) -> usize {
        self.num_harmonics
    }

    pub fn axis_basis_size(&self) -> usize {
        self.axis_basis_size
    }

    pub fn basis_size(&self) -> usize {
        self.basis_size
    }

    /// Real Fourier component represented by one per-axis column index.
    pub fn axis_component(axis_column: usize) -> RealHarmonicComponent {
        if axis_column == 0 {
            RealHarmonicComponent::Constant
        } else {
            let harmonic = axis_column.div_ceil(2);
            if axis_column % 2 == 1 {
                RealHarmonicComponent::Sine { harmonic }
            } else {
                RealHarmonicComponent::Cosine { harmonic }
            }
        }
    }

    /// Exact tensor Fourier modes and Laplace eigenvalues in evaluator column
    /// order (last axis fastest).
    pub fn spectral_modes(&self) -> Vec<TorusHarmonicMode> {
        let axis_m = self.axis_basis_size();
        let mut index = vec![0usize; self.latent_dim];
        let mut modes = Vec::with_capacity(self.basis_size());
        for _ in 0..self.basis_size() {
            let components: Vec<RealHarmonicComponent> =
                index.iter().copied().map(Self::axis_component).collect();
            let laplace_eigenvalue = components
                .iter()
                .map(|component| {
                    let harmonic = component.harmonic() as f64;
                    harmonic * harmonic
                })
                .sum();
            let l2_gram_weight = components.iter().fold(1.0, |weight, component| {
                if *component == RealHarmonicComponent::Constant {
                    weight
                } else {
                    0.5 * weight
                }
            });
            modes.push(TorusHarmonicMode {
                components,
                laplace_eigenvalue,
                l2_gram_weight,
            });
            for axis in (0..self.latent_dim).rev() {
                index[axis] += 1;
                if index[axis] < axis_m {
                    break;
                }
                index[axis] = 0;
            }
        }
        modes
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
        // Base (η-invariant) columns: the constant plus each single-axis
        // first-harmonic `[sin, cos]` (one non-constant axis, harmonic ≤ 1).
        // These embed the per-axis circles — the torus base topology — so they
        // are curved, not linear; only higher harmonics and cross-axis products
        // scale with `eta`. `eta = 0` is thus the base-topology relaxation.
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

    /// A `d = 2` torus is the tensor product of two equal circle factors, each
    /// of width `axis_basis_size()` (constant-leading: column 0 ≡ 1). Higher-
    /// dimensional tori are not a TWO-factor product, so they report `None`.
    fn factor_basis_sizes(&self) -> Option<(usize, usize)> {
        if self.latent_dim == 2 {
            let m = self.axis_basis_size();
            Some((m, m))
        } else {
            None
        }
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        // Single source of truth: allocate correctly-shaped buffers and fill
        // them through the in-place path (see `evaluate_into`).
        let n = coords.nrows();
        let m = self.basis_size();
        let d = self.latent_dim;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, d));
        self.evaluate_into(&mut phi, &mut jet, coords)?;
        Ok((phi, jet))
    }

    fn evaluate_into(
        &self,
        phi: &mut Array2<f64>,
        jet: &mut Array3<f64>,
        coords: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
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
        if phi.dim() != (n, m) {
            return Err(format!(
                "TorusHarmonicEvaluator::evaluate_into: Φ buffer {:?} != ({n}, {m})",
                phi.dim()
            ));
        }
        if jet.dim() != (n, m, d) {
            return Err(format!(
                "TorusHarmonicEvaluator::evaluate_into: jet buffer {:?} != ({n}, {m}, {d})",
                jet.dim()
            ));
        }
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        // Every `(row, flat)` Φ entry and every `(row, flat, axis)` jet entry is
        // written unconditionally in the product loops below, so no pre-clear is
        // needed — stale workspace contents cannot survive.
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
        Ok(())
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

/// One-dimensional real character of an involutive deck transformation.
///
/// This type is deliberately narrower than a signed-permutation action.  It
/// certifies that one cover column `f` is itself an eigenvector of pullback by
/// the non-identity deck map `g`: `g*f = f` or `g*f = -f`.  The exact Reynolds
/// projector is therefore
///
/// `P_Gamma f = (f + g*f) / 2 = f` for `Trivial`, and `0` for `Sign`.
///
/// A basis that is merely permuted by `g` cannot use this representation: for
/// example, swapping two columns retains their orbit sum, not either column.
/// Such an action requires an explicit restriction matrix `Q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeckInvolutionCharacter {
    Trivial,
    Sign,
}

impl DeckInvolutionCharacter {
    /// Eigenvalue of the exact group average `(I + g*) / 2` on this column.
    fn reynolds_projector_eigenvalue(self) -> u8 {
        match self {
            Self::Trivial => 1,
            Self::Sign => 0,
        }
    }
}

/// Spectral metadata for every cover column, in cover evaluator order.
///
/// Requiring a full character table, rather than accepting a caller-selected
/// mask, makes the group average the single authority for quotient width,
/// column restriction, Gram entries, penalty eigenvalues, and nullity.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CoverSpectralCharacter {
    deck_character: DeckInvolutionCharacter,
    laplace_eigenvalue: f64,
    l2_gram_weight: f64,
    curved_if_retained: bool,
}

/// Exact diagonal-character restriction of an analytic cover evaluator.
///
/// Static column selection is exact only because each supported cover column
/// is a simultaneous one-dimensional character of the deck involution.  Mere
/// signed permutation of a basis is insufficient; non-diagonal actions require
/// orbit sums or a dense restriction matrix `Q`.  Values and analytic
/// first/second/third jets are restricted from one cover evaluator, while the
/// same complete character table determines width, Gram, spectral penalties,
/// and null space.
#[derive(Debug, Clone)]
pub struct QuotientSpectralEvaluator {
    quotient_name: String,
    cover: Arc<dyn SaeBasisThirdJet>,
    cover_width: usize,
    cover_columns: Vec<usize>,
    laplace_eigenvalues: Vec<f64>,
    l2_gram_weights: Vec<f64>,
    curved_columns: Vec<bool>,
}

/// Number of invariant real spherical harmonics on `RP²` through even cover
/// degree `2H`: `sum_{r=0}^H (4r+1) = (H+1)(2H+1)`.
pub fn projective_plane_basis_size(harmonic_order: usize) -> Result<usize, String> {
    if harmonic_order == 0 {
        return Err("projective_plane_basis_size requires harmonic_order >= 1".to_string());
    }
    harmonic_order
        .checked_add(1)
        .and_then(|left| {
            harmonic_order
                .checked_mul(2)
                .and_then(|twice| twice.checked_add(1))
                .and_then(|right| left.checked_mul(right))
        })
        .ok_or_else(|| "projective_plane_basis_size overflowed usize".to_string())
}

/// Number of invariant real torus harmonics on the flat Klein quotient through
/// equal per-axis order `H`: `1 + 2 floor(H/2) + H + 2H²`.
pub fn klein_bottle_basis_size(num_harmonics: usize) -> Result<usize, String> {
    if num_harmonics == 0 {
        return Err("klein_bottle_basis_size requires num_harmonics >= 1".to_string());
    }
    let cross_width = num_harmonics
        .checked_mul(num_harmonics)
        .and_then(|square| square.checked_mul(2))
        .ok_or_else(|| "klein_bottle_basis_size overflowed usize".to_string())?;
    1usize
        .checked_add(2 * (num_harmonics / 2))
        .and_then(|width| width.checked_add(num_harmonics))
        .and_then(|width| width.checked_add(cross_width))
        .ok_or_else(|| "klein_bottle_basis_size overflowed usize".to_string())
}

impl QuotientSpectralEvaluator {
    /// Build a quotient from a cover and its complete diagonal character table.
    ///
    /// `characters[cover_column]` must certify the pullback eigencharacter of
    /// that exact cover column.  The Reynolds projector derives the retained
    /// columns; callers cannot supply an already-filtered mask.  A connected
    /// closed cover and quotient have one zero-eigenvalue mode (the constant),
    /// whose character is trivial and which remains in the homotopy base.
    fn from_diagonal_involution_characters(
        quotient_name: impl Into<String>,
        cover: Arc<dyn SaeBasisThirdJet>,
        cover_width: usize,
        characters: Vec<CoverSpectralCharacter>,
    ) -> Result<Self, String> {
        let quotient_name = quotient_name.into();
        if quotient_name.trim().is_empty() {
            return Err("QuotientSpectralEvaluator requires a non-empty quotient name".to_string());
        }
        if cover_width == 0 {
            return Err(format!(
                "QuotientSpectralEvaluator[{quotient_name}]: cover width must be positive"
            ));
        }
        if characters.len() != cover_width {
            return Err(format!(
                "QuotientSpectralEvaluator[{quotient_name}]: diagonal character table width {} != cover width {cover_width}",
                characters.len()
            ));
        }

        let mut cover_columns = Vec::with_capacity(cover_width);
        let mut laplace_eigenvalues = Vec::with_capacity(cover_width);
        let mut l2_gram_weights = Vec::with_capacity(cover_width);
        let mut curved_columns = Vec::with_capacity(cover_width);
        let mut nullity = 0usize;
        for (cover_column, mode) in characters.into_iter().enumerate() {
            if !(mode.laplace_eigenvalue.is_finite() && mode.laplace_eigenvalue >= 0.0) {
                return Err(format!(
                    "QuotientSpectralEvaluator[{quotient_name}]: cover column {cover_column} has invalid Laplace eigenvalue {}",
                    mode.laplace_eigenvalue
                ));
            }
            if !(mode.l2_gram_weight.is_finite() && mode.l2_gram_weight > 0.0) {
                return Err(format!(
                    "QuotientSpectralEvaluator[{quotient_name}]: cover column {cover_column} has invalid L2 Gram weight {}",
                    mode.l2_gram_weight
                ));
            }
            if mode.laplace_eigenvalue == 0.0 {
                if mode.deck_character != DeckInvolutionCharacter::Trivial {
                    return Err(format!(
                        "QuotientSpectralEvaluator[{quotient_name}]: the constant cover mode at column {cover_column} must have trivial deck character"
                    ));
                }
                nullity += 1;
                if mode.curved_if_retained {
                    return Err(format!(
                        "QuotientSpectralEvaluator[{quotient_name}]: the constant null mode cannot be curvature-scaled"
                    ));
                }
            }
            if mode.deck_character.reynolds_projector_eigenvalue() == 1 {
                cover_columns.push(cover_column);
                laplace_eigenvalues.push(mode.laplace_eigenvalue);
                l2_gram_weights.push(mode.l2_gram_weight);
                curved_columns.push(mode.curved_if_retained);
            }
        }
        if nullity != 1 {
            return Err(format!(
                "QuotientSpectralEvaluator[{quotient_name}]: connected closed quotient requires exactly one constant null mode; found {nullity}"
            ));
        }

        Ok(Self {
            quotient_name,
            cover,
            cover_width,
            cover_columns,
            laplace_eigenvalues,
            l2_gram_weights,
            curved_columns,
        })
    }

    /// Real harmonics on `RP² = S²/{u ~ -u}` through quotient order `H`.
    ///
    /// Antipodal parity is `Y_lm(-u) = (-1)^l Y_lm(u)`, hence precisely the
    /// even degrees `l = 0, 2, ..., 2H` survive.
    pub fn projective_plane(harmonic_order: usize) -> Result<Self, String> {
        if harmonic_order == 0 {
            return Err(
                "QuotientSpectralEvaluator::projective_plane requires harmonic_order >= 1"
                    .to_string(),
            );
        }
        let max_degree = harmonic_order.checked_mul(2).ok_or_else(|| {
            "QuotientSpectralEvaluator::projective_plane: maximum cover degree overflowed usize"
                .to_string()
        })?;
        let cover_side = max_degree.checked_add(1).ok_or_else(|| {
            "QuotientSpectralEvaluator::projective_plane: cover width overflowed usize".to_string()
        })?;
        cover_side.checked_mul(cover_side).ok_or_else(|| {
            "QuotientSpectralEvaluator::projective_plane: cover width overflowed usize".to_string()
        })?;
        let expected_width = projective_plane_basis_size(harmonic_order)?;

        let cover = SphericalHarmonicEvaluator::new(max_degree)?;
        let cover_width = cover.basis_size();
        let modes = cover.spectral_modes();
        Self::projective_plane_from_cover(
            "projective-plane",
            Arc::new(cover),
            cover_width,
            modes,
            expected_width,
        )
    }

    /// `RP²` on the AMBIENT spherical cover — the pole-free sibling of
    /// [`Self::projective_plane`].
    ///
    /// The construction is *identical*: same antipodal deck involution, same
    /// Reynolds projector, same width theorem. Only the cover's COORDINATES
    /// change, from `(lat, lon)` to the ambient unit vector — and that is
    /// legitimate precisely because [`AmbientSphereHarmonicEvaluator`] is the
    /// same basis in different coordinates, pinned to 1e-12 by
    /// `ambient_sphere_matches_chart_on_the_sphere`. Its `spectral_modes()`
    /// therefore carry the same `(degree, order)` in the same column order, so
    /// the character table transfers unchanged rather than being re-derived.
    ///
    /// The quotient is also stated more honestly here. `RP² = S²/{u ~ -u}`, and
    /// the antipodal map IS the ambient `u -> -u`; in the chart the same map is
    /// the awkward `(lat, lon) -> (-lat, lon + π)`, whose Killing directions the
    /// chart cannot even evaluate at its own poles.
    pub fn projective_plane_ambient(harmonic_order: usize) -> Result<Self, String> {
        if harmonic_order == 0 {
            return Err(
                "QuotientSpectralEvaluator::projective_plane_ambient requires harmonic_order >= 1"
                    .to_string(),
            );
        }
        let max_degree = harmonic_order.checked_mul(2).ok_or_else(|| {
            "QuotientSpectralEvaluator::projective_plane_ambient: maximum cover degree overflowed usize"
                .to_string()
        })?;
        let expected_width = projective_plane_basis_size(harmonic_order)?;
        let cover = AmbientSphereHarmonicEvaluator::new(max_degree)?;
        let cover_width = cover.basis_size();
        let modes = cover.spectral_modes();
        Self::projective_plane_from_cover(
            "projective-plane-ambient",
            Arc::new(cover),
            cover_width,
            modes,
            expected_width,
        )
    }

    /// Shared `RP²` construction over either spherical cover.
    ///
    /// Antipodal parity is `Y_lm(-u) = (-1)^l Y_lm(u)`, hence precisely the even
    /// degrees survive — a statement about the HARMONICS, not about the
    /// coordinates they are written in, which is why one implementation serves
    /// both covers.
    fn projective_plane_from_cover(
        quotient_name: &str,
        cover: Arc<dyn SaeBasisThirdJet>,
        cover_width: usize,
        modes: Vec<SphericalHarmonicMode>,
        expected_width: usize,
    ) -> Result<Self, String> {
        let characters = modes
            .into_iter()
            .map(|mode| CoverSpectralCharacter {
                deck_character: if mode.degree % 2 == 0 {
                    DeckInvolutionCharacter::Trivial
                } else {
                    DeckInvolutionCharacter::Sign
                },
                laplace_eigenvalue: mode.laplace_eigenvalue,
                l2_gram_weight: mode.l2_gram_weight,
                // `l = 0,2` contains the constant and Veronese embedding; only
                // higher even degrees are the curvature refinement.
                curved_if_retained: mode.degree > 2,
            })
            .collect::<Vec<_>>();
        let evaluator =
            Self::from_diagonal_involution_characters(quotient_name, cover, cover_width, characters)?;
        if evaluator.basis_size() != expected_width {
            return Err(format!(
                "QuotientSpectralEvaluator::{quotient_name}: group average produced width {}, expected {expected_width}",
                evaluator.basis_size()
            ));
        }
        Ok(evaluator)
    }

    /// Real harmonics on the flat Klein bottle
    /// `T²/{(theta, phi) ~ (theta + 1/2, -phi)}` through order `H` per axis.
    /// `H >= 2` is required because the standard smooth `R⁴` Klein embedding
    /// uses theta harmonics one and two; its constant plus six coordinate modes
    /// form the seven-column homotopy base.
    pub fn klein_bottle(num_harmonics: usize) -> Result<Self, String> {
        if num_harmonics < 2 {
            return Err(
                "QuotientSpectralEvaluator::klein_bottle requires num_harmonics >= 2 for the standard R4 embedding"
                    .to_string(),
            );
        }
        let expected_width = klein_bottle_basis_size(num_harmonics)?;

        let cover = TorusHarmonicEvaluator::new(2, num_harmonics)?;
        let cover_width = cover.basis_size();
        let mut characters = Vec::with_capacity(cover_width);
        for (cover_column, mode) in cover.spectral_modes().into_iter().enumerate() {
            let [theta_component, phi_component] = mode.components.as_slice() else {
                return Err(format!(
                    "QuotientSpectralEvaluator::klein_bottle: torus cover mode {cover_column} did not have two factors"
                ));
            };
            let theta_harmonic = theta_component.harmonic();
            let phi_harmonic = phi_component.harmonic();
            let half_turn_sign = if theta_harmonic % 2 == 0 { 1 } else { -1 };
            let deck_character = if half_turn_sign * phi_component.reflection_sign() == 1 {
                DeckInvolutionCharacter::Trivial
            } else {
                DeckInvolutionCharacter::Sign
            };
            // Standard R4 Klein embedding in this unit-period chart:
            //   {1,
            //    sin/cos(4πθ),
            //    sin/cos(4πθ) cos(2πφ),
            //    sin/cos(2πθ) sin(2πφ)}.
            // These are exactly one constant plus six coordinate columns.
            let embedding_mode = (theta_harmonic == 0 && phi_harmonic == 0)
                || (theta_harmonic == 2 && phi_harmonic == 0)
                || (theta_harmonic == 2
                    && phi_harmonic == 1
                    && phi_component.reflection_sign() == 1)
                || (theta_harmonic == 1
                    && phi_harmonic == 1
                    && phi_component.reflection_sign() == -1);
            characters.push(CoverSpectralCharacter {
                deck_character,
                laplace_eigenvalue: mode.laplace_eigenvalue,
                l2_gram_weight: mode.l2_gram_weight,
                curved_if_retained: !embedding_mode,
            });
        }
        let evaluator = Self::from_diagonal_involution_characters(
            "klein-bottle",
            Arc::new(cover),
            cover_width,
            characters,
        )?;
        if evaluator.basis_size() != expected_width {
            return Err(format!(
                "QuotientSpectralEvaluator::klein_bottle: group average produced width {}, expected {expected_width}",
                evaluator.basis_size()
            ));
        }
        Ok(evaluator)
    }

    pub fn basis_size(&self) -> usize {
        self.cover_columns.len()
    }

    /// Exact spectral penalty `G · diag(lambda_laplace^power)`.
    ///
    /// The Gram factor matters for the unnormalized real Fourier columns: it
    /// makes the generalized penalty eigenvalues exactly the cover Laplacian's
    /// eigenvalues rather than silently changing them by column convention.
    pub fn spectral_penalty(&self, power: u32) -> Result<Array2<f64>, String> {
        if power == 0 {
            return Err(format!(
                "QuotientSpectralEvaluator[{}]::spectral_penalty requires power >= 1",
                self.quotient_name
            ));
        }
        let exponent = i32::try_from(power).map_err(|_| {
            format!(
                "QuotientSpectralEvaluator[{}]::spectral_penalty power {power} exceeds i32::MAX",
                self.quotient_name
            )
        })?;
        let mut penalty = Array2::<f64>::zeros((self.basis_size(), self.basis_size()));
        for column in 0..self.basis_size() {
            let value =
                self.l2_gram_weights[column] * self.laplace_eigenvalues[column].powi(exponent);
            if !value.is_finite() {
                return Err(format!(
                    "QuotientSpectralEvaluator[{}]::spectral_penalty overflowed at quotient column {column}",
                    self.quotient_name
                ));
            }
            penalty[[column, column]] = value;
        }
        Ok(penalty)
    }

    fn validate_cover_value_jet(
        &self,
        phi: &Array2<f64>,
        jet: &Array3<f64>,
        n_rows: usize,
        latent_dim: usize,
    ) -> Result<(), String> {
        if phi.dim() != (n_rows, self.cover_width) {
            return Err(format!(
                "QuotientSpectralEvaluator[{}]: cover Phi shape {:?} != ({n_rows}, {})",
                self.quotient_name,
                phi.dim(),
                self.cover_width
            ));
        }
        if jet.dim() != (n_rows, self.cover_width, latent_dim) {
            return Err(format!(
                "QuotientSpectralEvaluator[{}]: cover jet shape {:?} != ({n_rows}, {}, {latent_dim})",
                self.quotient_name,
                jet.dim(),
                self.cover_width
            ));
        }
        Ok(())
    }
}

impl SaeBasisEvaluator for QuotientSpectralEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let n_rows = coords.nrows();
        let latent_dim = coords.ncols();
        let mut phi = Array2::<f64>::zeros((n_rows, self.basis_size()));
        let mut jet = Array3::<f64>::zeros((n_rows, self.basis_size(), latent_dim));
        self.evaluate_into(&mut phi, &mut jet, coords)?;
        Ok((phi, jet))
    }

    fn evaluate_into(
        &self,
        phi: &mut Array2<f64>,
        jet: &mut Array3<f64>,
        coords: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let n_rows = coords.nrows();
        let latent_dim = coords.ncols();
        let quotient_width = self.basis_size();
        if phi.dim() != (n_rows, quotient_width) {
            return Err(format!(
                "QuotientSpectralEvaluator[{}]::evaluate_into: Phi buffer {:?} != ({n_rows}, {quotient_width})",
                self.quotient_name,
                phi.dim()
            ));
        }
        if jet.dim() != (n_rows, quotient_width, latent_dim) {
            return Err(format!(
                "QuotientSpectralEvaluator[{}]::evaluate_into: jet buffer {:?} != ({n_rows}, {quotient_width}, {latent_dim})",
                self.quotient_name,
                jet.dim()
            ));
        }

        let (cover_phi, cover_jet) = self.cover.evaluate(coords)?;
        self.validate_cover_value_jet(&cover_phi, &cover_jet, n_rows, latent_dim)?;
        for (quotient_column, &cover_column) in self.cover_columns.iter().enumerate() {
            for row in 0..n_rows {
                phi[[row, quotient_column]] = cover_phi[[row, cover_column]];
                for axis in 0..latent_dim {
                    jet[[row, quotient_column, axis]] = cover_jet[[row, cover_column, axis]];
                }
            }
        }
        Ok(())
    }

    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        if n_basis != self.basis_size() {
            return Err(format!(
                "QuotientSpectralEvaluator[{}]::phi_eta_split: n_basis {n_basis} != evaluator width {}",
                self.quotient_name,
                self.basis_size()
            ));
        }
        Ok(PhiEtaSplit::from_curved_mask(self.curved_columns.clone()))
    }

    /// A quotient mask couples cover factors and is not a tensor-product basis.
    fn factor_basis_sizes(&self) -> Option<(usize, usize)> {
        None
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }
}

impl SaeBasisSecondJet for QuotientSpectralEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let n_rows = coords.nrows();
        let latent_dim = coords.ncols();
        let cover_hessian = self.cover.second_jet(coords)?;
        if cover_hessian.dim() != (n_rows, self.cover_width, latent_dim, latent_dim) {
            return Err(format!(
                "QuotientSpectralEvaluator[{}]: cover second-jet shape {:?} != ({n_rows}, {}, {latent_dim}, {latent_dim})",
                self.quotient_name,
                cover_hessian.dim(),
                self.cover_width
            ));
        }
        let mut hessian = Array4::<f64>::zeros((n_rows, self.basis_size(), latent_dim, latent_dim));
        for (quotient_column, &cover_column) in self.cover_columns.iter().enumerate() {
            for row in 0..n_rows {
                for axis_a in 0..latent_dim {
                    for axis_b in 0..latent_dim {
                        hessian[[row, quotient_column, axis_a, axis_b]] =
                            cover_hessian[[row, cover_column, axis_a, axis_b]];
                    }
                }
            }
        }
        Ok(hessian)
    }
}

impl SaeBasisThirdJet for QuotientSpectralEvaluator {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        let n_rows = coords.nrows();
        let latent_dim = coords.ncols();
        let cover_third = self.cover.third_jet(coords)?;
        if cover_third.dim() != (n_rows, self.cover_width, latent_dim, latent_dim, latent_dim) {
            return Err(format!(
                "QuotientSpectralEvaluator[{}]: cover third-jet shape {:?} != ({n_rows}, {}, {latent_dim}, {latent_dim}, {latent_dim})",
                self.quotient_name,
                cover_third.dim(),
                self.cover_width
            ));
        }
        let mut third = Array5::<f64>::zeros((
            n_rows,
            self.basis_size(),
            latent_dim,
            latent_dim,
            latent_dim,
        ));
        for (quotient_column, &cover_column) in self.cover_columns.iter().enumerate() {
            for row in 0..n_rows {
                for axis_a in 0..latent_dim {
                    for axis_b in 0..latent_dim {
                        for axis_c in 0..latent_dim {
                            third[[row, quotient_column, axis_a, axis_b, axis_c]] =
                                cover_third[[row, cover_column, axis_a, axis_b, axis_c]];
                        }
                    }
                }
            }
        }
        Ok(third)
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
/// [`gam_terms::basis::build_duchon_basis`] under the SAE atom's spec
/// (`length_scale = None`, `power = 0`, no identifiability transform). The
/// forward design and the jet are produced from a single core entry point
/// ([`gam_terms::basis::duchon_sae_atom_basis_with_jet`]) so they always agree on
/// column count and scaling — the exact contract issue #247 pinned.
#[derive(Debug, Clone)]
pub struct DuchonCoordinateEvaluator {
    pub centers: Array2<f64>,
    pub order: gam_terms::basis::DuchonNullspaceOrder,
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
            1 => gam_terms::basis::DuchonNullspaceOrder::Zero,
            2 => gam_terms::basis::DuchonNullspaceOrder::Linear,
            other => gam_terms::basis::DuchonNullspaceOrder::Degree(other - 1),
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
        if let gam_terms::basis::DuchonNullspaceOrder::Degree(degree) = effective {
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
        gam_terms::basis::duchon_sae_atom_basis_with_jet(coords, self.centers.view(), self.order)
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
        gam_terms::basis::duchon_sae_atom_second_jet(coords, self.centers.view(), self.order)
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
        gam_terms::basis::duchon_sae_atom_third_jet(coords, self.centers.view(), self.order)
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
        gam_terms::basis::monomial_exponents(self.latent_dim, self.max_degree).len()
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
        // Single source of truth: allocate correctly-shaped buffers and fill
        // them through the in-place path (see `evaluate_into`).
        let n = coords.nrows();
        let m = self.basis_size();
        let d = self.latent_dim;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, d));
        self.evaluate_into(&mut phi, &mut jet, coords)?;
        Ok((phi, jet))
    }

    fn evaluate_into(
        &self,
        phi: &mut Array2<f64>,
        jet: &mut Array3<f64>,
        coords: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "EuclideanPatchEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = gam_terms::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        if phi.dim() != (n, m) {
            return Err(format!(
                "EuclideanPatchEvaluator::evaluate_into: Φ buffer {:?} != ({n}, {m})",
                phi.dim()
            ));
        }
        if jet.dim() != (n, m, d) {
            return Err(format!(
                "EuclideanPatchEvaluator::evaluate_into: jet buffer {:?} != ({n}, {m}, {d})",
                jet.dim()
            ));
        }
        // The jet is nonzero only where a monomial's axis exponent is positive,
        // so most entries are left untouched by the loops below; clear both
        // buffers to erase any stale (reused-workspace) contents first.
        phi.fill(0.0);
        jet.fill(0.0);
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
            // Monomial first derivative, written to match
            // `gam_terms::basis::duchon_polynomial_first_derivative_nd` operation
            // for operation (same factor ordering) so the value is bit-identical.
            for axis in 0..d {
                let a_axis = alpha[axis];
                if a_axis == 0 {
                    continue;
                }
                for row in 0..n {
                    let mut value = a_axis as f64;
                    for a in 0..d {
                        let exp_a = if a == axis { a_axis - 1 } else { alpha[a] };
                        if exp_a != 0 {
                            value *= coords[[row, a]].powi(exp_a as i32);
                        }
                    }
                    jet[[row, col, axis]] = value;
                }
            }
        }
        Ok(())
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
        let exponents = gam_terms::basis::monomial_exponents(self.latent_dim, self.max_degree);
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
        let exponents = gam_terms::basis::monomial_exponents(self.latent_dim, self.max_degree);
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
        // reference interval [0,1).
        //
        // NOTE: despite the cylinder being declared `S¹ × ℝ` (unbounded line
        // axis) in the latent-manifold spec, this line-factor roughness is a
        // *compact* [0,1) reference-domain measure, NOT an intrinsic roughness
        // integrated over all of ℝ (a uniform ∫_ℝ tⁱ⁺ʲ dt would diverge for
        // polynomials). It is a canonical reference penalty: the line
        // coordinate's scale and origin therefore matter through this reference
        // interval — the same monomial coefficients carry a different penalty
        // if the line axis is rescaled or shifted relative to [0,1).
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
        // A product column `[c, l]` is curved (η-dialed) iff either factor
        // carries dialed curvature: the circle factor on any harmonic above the
        // first (`c > 2`, the 2nd-and-higher sin/cos), the line factor on any
        // monomial of degree ≥ 2 (`l > 1`). The first-harmonic circle columns
        // crossed with the affine line columns are the base (η-invariant) block
        // and span the base-topology relaxation — the base circle columns still
        // embed the unit circle, so this block is not linear — matching the
        // `eta`-homotopy split of the two parent evaluators.
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

    /// A cylinder `S¹ × ℝ` is the tensor product of the circle factor
    /// (`circle_basis_size()`, the slow axis 0) and the line factor
    /// (`line_basis_size()`, the fast axis 1), both constant-leading.
    fn factor_basis_sizes(&self) -> Option<(usize, usize)> {
        Some((self.circle_basis_size(), self.line_basis_size()))
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

/// Möbius-band harmonic basis on the DOUBLE-COVER chart (#2240).
///
/// The band is charted by `(s, w)` with `s ∈ [0, 2)` an angle on the
/// double-cover circle (period 2) and `w ∈ [-1, 1]` the width coordinate. The
/// deck transformation of the double cover is `σ(s, w) = (s + 1, -w)`; the
/// Möbius band is the quotient, and functions on the band are exactly the
/// σ-invariant functions on the cylinder cover. With the tensor harmonics
/// `T_k(s)·w^m` (`T_k ∈ {cos(πks), sin(πks)}`, period-2 modes), invariance is
/// `(-1)^{k+m} = 1`, so the basis keeps exactly the columns with `k + m`
/// EVEN: `(k even, m even) ∪ (k odd, m odd)`.
///
/// This gives the band its defining behavior with NO new manifold machinery:
/// the optimizer retracts on an ordinary smooth cylinder (`Circle{period 2} ×
/// Interval[-1, 1]`), every basis column is C^∞ there, and a point and its
/// deck-twin `(s+1, -w)` produce IDENTICAL basis rows — the half-twist lives
/// in the parity culling, not in a seam. Odd-`m` (width-odd) structure is
/// forced to carry a half-period angular factor, which is precisely the
/// non-orientability a torus or flat-patch chart cannot express.
#[derive(Debug, Clone)]
pub struct MobiusHarmonicEvaluator {
    /// Circle harmonics `H ≥ 1` on the double-cover angle (mode `k ≤ H`).
    pub circle_harmonics: usize,
    /// Width monomial degree `D ≥ 1` (`w^m`, `m ≤ D`).
    pub width_degree: usize,
    /// Admitted `(circle_col, width_power)` pairs (deck-invariant columns),
    /// circle-column-slow / width-power-fast, fixed at construction.
    columns: Vec<(usize, usize)>,
}

impl MobiusHarmonicEvaluator {
    pub fn new(circle_harmonics: usize, width_degree: usize) -> Result<Self, String> {
        if circle_harmonics == 0 {
            return Err(
                "MobiusHarmonicEvaluator requires circle_harmonics >= 1 (the band core needs \
                 at least the half-period harmonic pair)"
                    .to_string(),
            );
        }
        if width_degree == 0 {
            return Err(
                "MobiusHarmonicEvaluator requires width_degree >= 1: with no width-odd \
                 columns the deck-invariant basis degenerates to a plain circle"
                    .to_string(),
            );
        }
        let mc = 2 * circle_harmonics + 1;
        let mut columns = Vec::new();
        for c in 0..mc {
            let k = Self::circle_mode(c);
            for m in 0..=width_degree {
                if (k + m) % 2 == 0 {
                    columns.push((c, m));
                }
            }
        }
        Ok(Self {
            circle_harmonics,
            width_degree,
            columns,
        })
    }

    /// Angular mode `k` of circle column `c` (col 0 = constant, cols
    /// `2h-1`/`2h` = the sin/cos pair of mode `h`).
    fn circle_mode(c: usize) -> usize {
        c.div_ceil(2)
    }

    pub fn basis_size(&self) -> usize {
        self.columns.len()
    }

    /// Circle derivative tables on the DOUBLE-COVER angle, orders 0..=3:
    /// mode `k` has frequency `ω = πk` (period 2), so odd modes are the
    /// half-period harmonics the quotient demands.
    fn circle_tables(&self, s: f64) -> [Vec<f64>; 4] {
        let mc = 2 * self.circle_harmonics + 1;
        let pi = std::f64::consts::PI;
        let mut table = [
            vec![0.0_f64; mc],
            vec![0.0_f64; mc],
            vec![0.0_f64; mc],
            vec![0.0_f64; mc],
        ];
        table[0][0] = 1.0;
        for h in 1..=self.circle_harmonics {
            let omega = pi * (h as f64);
            let w2 = omega * omega;
            let w3 = w2 * omega;
            let angle = omega * s;
            let sv = angle.sin();
            let cv = angle.cos();
            let s_idx = 2 * h - 1;
            let c_idx = 2 * h;
            table[0][s_idx] = sv;
            table[1][s_idx] = omega * cv;
            table[2][s_idx] = -w2 * sv;
            table[3][s_idx] = -w3 * cv;
            table[0][c_idx] = cv;
            table[1][c_idx] = -omega * sv;
            table[2][c_idx] = -w2 * cv;
            table[3][c_idx] = w3 * sv;
        }
        table
    }

    /// Width (monomial) derivative tables on `[-1, 1]`, orders 0..=3.
    fn width_tables(&self, w: f64) -> [Vec<f64>; 4] {
        let mw = self.width_degree + 1;
        let mut table = [
            vec![0.0_f64; mw],
            vec![0.0_f64; mw],
            vec![0.0_f64; mw],
            vec![0.0_f64; mw],
        ];
        for j in 0..mw {
            for k in 0..4 {
                if k > j {
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
                    w.powi(residual as i32)
                };
                table[k][j] = coeff * pow;
            }
        }
        table
    }

    /// Analytic seed roughness Gram `S = Sc ⊗ Gw + Gc ⊗ Sw` restricted to the
    /// admitted deck-invariant columns. On the double-cover measure
    /// `[0, 2) × [-1, 1]` the circle Grams are diagonal (`∫₀² 1 = 2`,
    /// `∫₀² sin²(πks) = ∫₀² cos²(πks) = 1`, all cross terms vanish over the
    /// full period; the second derivative scales a mode by `(πk)²`), and the
    /// width Grams are the even-moment tables `∫₋₁¹ w^{i+j} dw`
    /// (`= 2/(i+j+1)` for `i+j` even, `0` odd). The constant column sits in
    /// the null space exactly as the smooth-penalty nullity recovery expects.
    pub fn roughness_gram(&self) -> Array2<f64> {
        let pi = std::f64::consts::PI;
        let m = self.columns.len();
        let moment = |exp: usize| -> f64 {
            if exp % 2 == 0 {
                2.0 / ((exp + 1) as f64)
            } else {
                0.0
            }
        };
        let mut s = Array2::<f64>::zeros((m, m));
        for (row, &(c_a, m_a)) in self.columns.iter().enumerate() {
            for (col, &(c_b, m_b)) in self.columns.iter().enumerate() {
                if c_a != c_b {
                    // Distinct circle columns are orthogonal in BOTH circle
                    // Grams over the full double-cover period.
                    continue;
                }
                let k = Self::circle_mode(c_a) as f64;
                let gc = if c_a == 0 { 2.0 } else { 1.0 };
                let sc = if c_a == 0 {
                    0.0
                } else {
                    (pi * k).powi(4) * 1.0
                };
                let gw = moment(m_a + m_b);
                let sw = if m_a >= 2 && m_b >= 2 {
                    ((m_a * (m_a - 1)) as f64) * ((m_b * (m_b - 1)) as f64) * moment(m_a + m_b - 4)
                } else {
                    0.0
                };
                s[[row, col]] = sc * gw + gc * sw;
            }
        }
        s
    }

    fn check_coords(&self, coords: ArrayView2<'_, f64>, what: &str) -> Result<(), String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "MobiusHarmonicEvaluator::{what}: expected latent_dim == 2 (double-cover \
                 angle × width), got {}",
                coords.ncols()
            ));
        }
        Ok(())
    }
}

impl SaeBasisEvaluator for MobiusHarmonicEvaluator {
    fn phi_eta_split(&self, n_basis: usize) -> Result<PhiEtaSplit, String> {
        let expected = self.basis_size();
        if n_basis != expected {
            return Err(format!(
                "MobiusHarmonicEvaluator::phi_eta_split: n_basis {n_basis} != evaluator width {expected}"
            ));
        }
        // Base (η-invariant) block: the constant, the half-period first
        // harmonic pair crossed with the affine width — the band's core
        // embedding. Higher angular modes (k ≥ 2) or width curvature (m ≥ 2)
        // are the η-dialed refinement, mirroring the cylinder's split.
        let curved = self
            .columns
            .iter()
            .map(|&(c, m)| Self::circle_mode(c) >= 2 || m >= 2)
            .collect::<Vec<_>>();
        Ok(PhiEtaSplit::from_curved_mask(curved))
    }

    /// The parity culling breaks the clean tensor-product factorization, so
    /// the Möbius basis does not expose per-axis factor sizes.
    fn factor_basis_sizes(&self) -> Option<(usize, usize)> {
        None
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
        let m = self.basis_size();
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 2));
        for row in 0..n {
            let circ = self.circle_tables(coords[[row, 0]]);
            let width = self.width_tables(coords[[row, 1]]);
            for (col, &(c, wm)) in self.columns.iter().enumerate() {
                phi[[row, col]] = circ[0][c] * width[0][wm];
                jet[[row, col, 0]] = circ[1][c] * width[0][wm];
                jet[[row, col, 1]] = circ[0][c] * width[1][wm];
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for MobiusHarmonicEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        self.check_coords(coords, "second_jet")?;
        let n = coords.nrows();
        let m = self.basis_size();
        let mut h = Array4::<f64>::zeros((n, m, 2, 2));
        for row in 0..n {
            let circ = self.circle_tables(coords[[row, 0]]);
            let width = self.width_tables(coords[[row, 1]]);
            for (col, &(c, wm)) in self.columns.iter().enumerate() {
                h[[row, col, 0, 0]] = circ[2][c] * width[0][wm];
                h[[row, col, 1, 1]] = circ[0][c] * width[2][wm];
                let mixed = circ[1][c] * width[1][wm];
                h[[row, col, 0, 1]] = mixed;
                h[[row, col, 1, 0]] = mixed;
            }
        }
        Ok(h)
    }
}

impl SaeBasisThirdJet for MobiusHarmonicEvaluator {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        self.check_coords(coords, "third_jet")?;
        let n = coords.nrows();
        let m = self.basis_size();
        let mut t3 = Array5::<f64>::zeros((n, m, 2, 2, 2));
        for row in 0..n {
            let circ = self.circle_tables(coords[[row, 0]]);
            let width = self.width_tables(coords[[row, 1]]);
            for (col, &(c, wm)) in self.columns.iter().enumerate() {
                for a in 0..2 {
                    for b in 0..2 {
                        for e in 0..2 {
                            let k0 = (a == 0) as usize + (b == 0) as usize + (e == 0) as usize;
                            let k1 = 3 - k0;
                            t3[[row, col, a, b, e]] = circ[k0][c] * width[k1][wm];
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

/// F2 finite-set (discrete anchor) basis: the INDICATOR / one-hot design over a
/// fixed set of `anchors` anchors. The latent coordinate `t` (a single flat axis)
/// is read as a categorical assignment — `t` is snapped to its nearest anchor
/// index `round(t)` clamped to `[0, anchors)` — and the design row is the one-hot
/// vector selecting that anchor. This is the honest model for cluster-like
/// structure (weekdays as a finite point set, not an occupied circle): unlike
/// every continuous evaluator here, the decoded map is piecewise CONSTANT in `t`,
/// so its first, second, and third jets are identically zero (the derivative of a
/// step is zero a.e.; the anchor assignment moves by re-labelling, not by a
/// tangent step). The design width equals `anchors`; the rank charge the race
/// prices is `anchors − 1` ([`crate::manifold::finite_set_rank_charge`]), one
/// anchor being the reference contrast.
#[derive(Debug, Clone)]
pub struct AnchorIndicatorEvaluator {
    pub anchors: usize,
}

impl AnchorIndicatorEvaluator {
    pub fn new(anchors: usize) -> Result<Self, String> {
        if anchors < 2 {
            return Err(format!(
                "AnchorIndicatorEvaluator requires anchors >= 2 (a finite set of at \
                 least two points); got {anchors}"
            ));
        }
        Ok(Self { anchors })
    }
}

impl SaeBasisEvaluator for AnchorIndicatorEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        // The indicator design is piecewise constant, so every jet order is zero.
        let n = coords.nrows();
        Some(Ok(Array5::<f64>::zeros((n, self.anchors, 1, 1, 1))))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "AnchorIndicatorEvaluator: expected latent_dim == 1 (a single \
                 categorical axis), got {d}"
            ));
        }
        let m = self.anchors;
        let mut phi = Array2::<f64>::zeros((n, m));
        // The jet is identically zero: a one-hot indicator is constant between
        // anchors, so ∂Φ/∂t = 0 a.e.
        let jet = Array3::<f64>::zeros((n, m, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            if !t.is_finite() {
                return Err("AnchorIndicatorEvaluator: non-finite coordinate".to_string());
            }
            // Snap to the nearest anchor index, clamped into range.
            let idx = t.round().clamp(0.0, (m - 1) as f64) as usize;
            phi[[row, idx]] = 1.0;
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for AnchorIndicatorEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let n = coords.nrows();
        Ok(Array4::<f64>::zeros((n, self.anchors, 1, 1)))
    }
}

impl SaeBasisThirdJet for AnchorIndicatorEvaluator {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        let n = coords.nrows();
        Ok(Array5::<f64>::zeros((n, self.anchors, 1, 1, 1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// Deterministic splitmix64 stream in `[0, 1)` for reproducible fixtures.
    fn uniform_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Rodrigues rotation of a 3-vector about a (not necessarily unit) axis.
    fn rotate_vector(v: [f64; 3], axis: [f64; 3], theta: f64) -> [f64; 3] {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let k = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        let (s, c) = theta.sin_cos();
        let kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
        let cross = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];
        [
            v[0] * c + cross[0] * s + k[0] * kv * (1.0 - c),
            v[1] * c + cross[1] * s + k[1] * kv * (1.0 - c),
            v[2] * c + cross[2] * s + k[2] * kv * (1.0 - c),
        ]
    }

    /// Relative residual of least-squares projecting every column of `target`
    /// onto the column span of `phi`. Zero exactly when the target lies in the
    /// span, so it reads as a direct test of span CLOSURE.
    fn span_residual(phi: &Array2<f64>, target: &Array2<f64>) -> f64 {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;
        let p = phi.ncols();
        let q = target.ncols();
        let mut gram = Array2::<f64>::zeros((p, p));
        let mut rhs = Array2::<f64>::zeros((p, q));
        for row in 0..phi.nrows() {
            for a in 0..p {
                for b in 0..p {
                    gram[[a, b]] += phi[[row, a]] * phi[[row, b]];
                }
                for c in 0..q {
                    rhs[[a, c]] += phi[[row, a]] * target[[row, c]];
                }
            }
        }
        let scale = gram.diag().iter().copied().fold(0.0_f64, f64::max);
        for d in gram.diag_mut().iter_mut() {
            *d += scale * 64.0 * f64::EPSILON;
        }
        let coefficients = gram.cholesky(Side::Lower).unwrap().solve_mat(&rhs);
        let fitted = phi.dot(&coefficients);
        let (mut residual, mut total) = (0.0_f64, 0.0_f64);
        for row in 0..target.nrows() {
            for c in 0..q {
                residual += (target[[row, c]] - fitted[[row, c]]).powi(2);
                total += target[[row, c]].powi(2);
            }
        }
        (residual / total.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// Deterministic uniform sample of unit 3-vectors, returned in BOTH
    /// parameterizations so chart and ambient evaluators see the same points.
    fn sphere_sample(seed: u64, n: usize) -> (Array2<f64>, Array2<f64>) {
        let mut rng = uniform_stream(seed);
        let mut chart = Array2::<f64>::zeros((n, 2));
        let mut ambient = Array2::<f64>::zeros((n, 3));
        for row in 0..n {
            let lat = (2.0 * rng() - 1.0).asin();
            let lon = 2.0 * std::f64::consts::PI * rng();
            chart[[row, 0]] = lat;
            chart[[row, 1]] = lon;
            ambient[[row, 0]] = lat.cos() * lon.cos();
            ambient[[row, 1]] = lat.cos() * lon.sin();
            ambient[[row, 2]] = lat.sin();
        }
        (chart, ambient)
    }

    /// The ambient evaluator is not a NEW basis — it is the same real harmonics
    /// written in `(x, y, z)`. On the sphere the two must agree to rounding, and
    /// that equality is what lets the ambient form inherit the chart's
    /// orthonormality and spectral metadata instead of re-deriving them.
    #[test]
    fn ambient_sphere_matches_chart_on_the_sphere() {
        let (chart_coords, ambient_coords) = sphere_sample(0x5B4E, 512);
        let chart = SphericalHarmonicEvaluator::new(3).unwrap();
        let ambient = AmbientSphereHarmonicEvaluator::new(3).unwrap();
        assert_eq!(chart.basis_size(), ambient.basis_size());
        let (phi_chart, _) = chart.evaluate(chart_coords.view()).unwrap();
        let (phi_ambient, _) = ambient.evaluate(ambient_coords.view()).unwrap();
        let mut worst = 0.0_f64;
        for row in 0..phi_chart.nrows() {
            for col in 0..phi_chart.ncols() {
                worst = worst.max((phi_chart[[row, col]] - phi_ambient[[row, col]]).abs());
            }
        }
        assert!(
            worst <= 1.0e-12,
            "ambient and chart spherical harmonics disagree on S²: max |Δ| = {worst:.3e}"
        );
    }

    /// The ambient jets are exact polynomial derivatives, so they must match
    /// central differences in the AMBIENT coordinates — including the second
    /// jet, which the Newton/Schur assembly consumes directly.
    #[test]
    fn ambient_sphere_jets_match_finite_differences() {
        let (_, ambient_coords) = sphere_sample(0xDEF7, 24);
        let ambient = AmbientSphereHarmonicEvaluator::new(3).unwrap();
        let (_, jet) = ambient.evaluate(ambient_coords.view()).unwrap();
        let hessian = ambient.second_jet(ambient_coords.view()).unwrap();
        let h = 1.0e-5_f64;
        let (mut worst_first, mut worst_second) = (0.0_f64, 0.0_f64);
        for row in 0..ambient_coords.nrows() {
            let base = [
                ambient_coords[[row, 0]],
                ambient_coords[[row, 1]],
                ambient_coords[[row, 2]],
            ];
            for axis_a in 0..3 {
                let (mut plus, mut minus) = (base, base);
                plus[axis_a] += h;
                minus[axis_a] -= h;
                let (phi_plus, jet_plus) =
                    ambient.evaluate(at_coords(plus).view()).unwrap();
                let (phi_minus, jet_minus) =
                    ambient.evaluate(at_coords(minus).view()).unwrap();
                for col in 0..ambient.basis_size() {
                    let fd = (phi_plus[[0, col]] - phi_minus[[0, col]]) / (2.0 * h);
                    worst_first = worst_first.max((fd - jet[[row, col, axis_a]]).abs());
                }
                // Second jet: difference the analytic FIRST jet, so the check is
                // independent of the Hessian's own derivation.
                for col in 0..ambient.basis_size() {
                    for axis_b in 0..3 {
                        let fd = (jet_plus[[0, col, axis_b]] - jet_minus[[0, col, axis_b]])
                            / (2.0 * h);
                        worst_second =
                            worst_second.max((fd - hessian[[row, col, axis_b, axis_a]]).abs());
                    }
                }
            }
        }
        assert!(
            worst_first <= 1.0e-6,
            "ambient first jet disagrees with finite differences: {worst_first:.3e}"
        );
        assert!(
            worst_second <= 1.0e-5,
            "ambient second jet disagrees with finite differences: {worst_second:.3e}"
        );
    }

    /// Pack one ambient point into an `(1, 3)` coordinate array.
    fn at_coords(point: [f64; 3]) -> Array2<f64> {
        let mut single = Array2::<f64>::zeros((1, 3));
        for axis in 0..3 {
            single[[0, axis]] = point[axis];
        }
        single
    }

    /// THE POLE TEST. At `u = (0, 0, ±1)` the `(lat, lon)` chart's longitude jet
    /// is identically zero — every longitude names the same physical point, so
    /// `H_tt`'s longitude block is singular there and the coordinate carries no
    /// information. The ambient parameterization has no such point: the pole is
    /// an ordinary unit vector, and the two tangent directions at it (`x` and
    /// `y`) carry full first-order signal.
    #[test]
    fn ambient_sphere_has_no_pole_degeneracy() {
        let ambient = AmbientSphereHarmonicEvaluator::new(2).unwrap();
        let chart = SphericalHarmonicEvaluator::new(2).unwrap();
        for &pole_z in &[1.0_f64, -1.0_f64] {
            let (_, jet) = ambient.evaluate(at_coords([0.0, 0.0, pole_z]).view()).unwrap();
            let mut chart_coords = Array2::<f64>::zeros((1, 2));
            chart_coords[[0, 0]] = pole_z * std::f64::consts::FRAC_PI_2;
            chart_coords[[0, 1]] = 0.9;
            let (_, chart_jet) = chart.evaluate(chart_coords.view()).unwrap();

            let mut chart_longitude = 0.0_f64;
            for col in 0..chart.basis_size() {
                chart_longitude = chart_longitude.max(chart_jet[[0, col, 1]].abs());
            }
            assert!(
                chart_longitude <= 1.0e-12,
                "chart longitude jet should collapse at the pole (that is the defect); got {chart_longitude:.3e}"
            );

            let mut ambient_tangential = 0.0_f64;
            for col in 0..ambient.basis_size() {
                for axis in 0..2 {
                    ambient_tangential = ambient_tangential.max(jet[[0, col, axis]].abs());
                    assert!(
                        jet[[0, col, axis]].is_finite(),
                        "ambient jet must stay finite at the pole"
                    );
                }
            }
            assert!(
                ambient_tangential >= 0.1,
                "ambient basis must carry tangential signal AT the pole; got {ambient_tangential:.3e}"
            );
        }
    }

    /// THE ROTATION TEST, and the reason the fixed chart block is not a sphere
    /// atom. A basis on `S²` deserves the name only if its SPAN is closed under
    /// `SO(3)`: physics does not care where we put the pole, so a rotated copy
    /// of a representable field must still be representable. Each degree-`l`
    /// harmonic block is an `SO(3)` irrep, so the ambient basis is closed
    /// exactly. The fixed 7-column chart `[1, x, y, z, xy, yz, xz]` holds only
    /// three of the five `l = 2` harmonics, so rotating `xy` produces `x²−y²`
    /// outside its span — its achievable fit depends on the arbitrary
    /// orientation of the chart's pole relative to the data.
    #[test]
    fn ambient_sphere_span_is_rotation_closed_and_a_truncated_block_is_not() {
        let (chart_coords, ambient_coords) = sphere_sample(0x120F, 900);
        let axis = [1.0, 1.0, 1.0];
        let theta = 0.7_f64;

        let mut rotated_ambient = Array2::<f64>::zeros(ambient_coords.dim());
        let mut rotated_chart = Array2::<f64>::zeros(chart_coords.dim());
        for row in 0..ambient_coords.nrows() {
            let turned = rotate_vector(
                [
                    ambient_coords[[row, 0]],
                    ambient_coords[[row, 1]],
                    ambient_coords[[row, 2]],
                ],
                axis,
                theta,
            );
            for a in 0..3 {
                rotated_ambient[[row, a]] = turned[a];
            }
            rotated_chart[[row, 0]] = turned[2].clamp(-1.0, 1.0).asin();
            rotated_chart[[row, 1]] = turned[1].atan2(turned[0]);
        }

        let ambient = AmbientSphereHarmonicEvaluator::new(2).unwrap();
        let (phi, _) = ambient.evaluate(ambient_coords.view()).unwrap();
        let (phi_rotated, _) = ambient.evaluate(rotated_ambient.view()).unwrap();
        let ambient_residual = span_residual(&phi, &phi_rotated);
        assert!(
            ambient_residual <= 1.0e-8,
            "ambient harmonic span must be closed under SO(3); residual {ambient_residual:.3e}"
        );

        // NEGATIVE CONTROL, synthesised rather than borrowed. Drop two of the five
        // `l = 2` columns to reproduce exactly the defect the removed `(lat, lon)`
        // chart had — it carried three of the five quadrupoles — and confirm such
        // a span is measurably NOT rotation-closed. Built here rather than by
        // keeping the chart alive, so the control tests the PROPERTY (an
        // incomplete degree block breaks covariance) rather than one obsolete
        // type, and cannot rot with it.
        let keep: Vec<usize> = (0..phi.ncols()).filter(|&c| c != 7 && c != 8).collect();
        let truncate = |full: &Array2<f64>| -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((full.nrows(), keep.len()));
            for (target, &source) in keep.iter().enumerate() {
                for row in 0..full.nrows() {
                    out[[row, target]] = full[[row, source]];
                }
            }
            out
        };
        let partial_residual = span_residual(&truncate(&phi), &truncate(&phi_rotated));
        assert!(
            partial_residual >= 1.0e-3,
            "a degree-2 block missing two of its five harmonics must NOT be \
             rotation-closed; if this passes the control is vacuous and the \
             positive assertion above proves nothing. residual {partial_residual:.3e}"
        );
    }

    /// THE LAW, stated infinitesimally — and the exact condition that decides
    /// which identifiability path a sphere atom is allowed to use.
    ///
    /// A basis on a manifold must be covariant under that manifold's own
    /// isometries. Sliding a point along a Killing field is a motion the
    /// geometry cannot distinguish, so it must not change WHICH functions the
    /// atom can represent. Differentiating the finite statement:
    ///
    /// ```text
    ///     d/ds Φ(flow_K(p, s))|_{s=0}  =  J(p) · K(p)   must lie in span{Φ}
    /// ```
    ///
    /// `crate::identifiability::exact_orbit_fields` needs precisely this to
    /// certify an atom on the EXACT-ORBIT path, and records that the legacy
    /// chart fails it: *"the sphere's legacy chart basis is not closed under
    /// ambient rotations, so sphere atoms remain on the frame path."* The
    /// fallback is documented a few lines above as unable to represent three
    /// `SO(3)` generators — so today `RP²` gets exact `SO(3)` Killing fields
    /// while the sphere, whose group those generators ARE, does not.
    ///
    /// The ambient basis satisfies the law by construction: each degree-`l`
    /// block is an `SO(3)` irrep, so `J·K` is a rotation within that block.
    /// Asserting it here is what licenses moving sphere atoms onto the
    /// exact-orbit path and DELETING that special case, rather than keeping a
    /// weaker certificate alive to accommodate a broken chart.
    #[test]
    fn ambient_sphere_basis_is_closed_under_its_own_killing_fields() {
        let (_, ambient) = sphere_sample(0x50F3, 700);
        let evaluator = AmbientSphereHarmonicEvaluator::new(3).unwrap();
        let (phi, jet) = evaluator.evaluate(ambient.view()).unwrap();
        let width = evaluator.basis_size();
        let n = ambient.nrows();

        // Killing fields of `S²` in ambient coordinates: `K_a(u) = e_a × u`,
        // the generators of `Isom(S²) = O(3)` that `AtomTopology::Sphere`
        // already declares.
        for axis in 0..3 {
            let mut derivative = Array2::<f64>::zeros((n, width));
            for row in 0..n {
                let u = [ambient[[row, 0]], ambient[[row, 1]], ambient[[row, 2]]];
                let k = match axis {
                    0 => [0.0, -u[2], u[1]],
                    1 => [u[2], 0.0, -u[0]],
                    _ => [-u[1], u[0], 0.0],
                };
                for col in 0..width {
                    let mut acc = 0.0_f64;
                    for a in 0..3 {
                        acc += jet[[row, col, a]] * k[a];
                    }
                    derivative[[row, col]] = acc;
                }
            }
            let residual = span_residual(&phi, &derivative);
            assert!(
                residual <= 1.0e-9,
                "Killing generator {axis} carries the ambient sphere basis OUT of its own \
                 span (relative residual {residual:.3e}). The atom's representable function \
                 space would then depend on where the coordinate origin was placed, and the \
                 atom could not be certified on the exact-orbit path."
            );
        }
    }

    /// Directional derivative of every basis column along a coordinate-space
    /// vector field: `(J · K)[row, col] = Σ_a jet[row, col, a] · K[row, a]`.
    fn derivative_along_field(
        jet: &Array3<f64>,
        field: &Array2<f64>,
    ) -> Array2<f64> {
        let (n, width, d) = jet.dim();
        let mut out = Array2::<f64>::zeros((n, width));
        for row in 0..n {
            for col in 0..width {
                let mut acc = 0.0_f64;
                for axis in 0..d {
                    acc += jet[[row, col, axis]] * field[[row, axis]];
                }
                out[[row, col]] = acc;
            }
        }
        out
    }

    /// A constant coordinate-space field — a translation generator, which is
    /// what every flat chart's continuous isometries are.
    fn constant_field(n: usize, d: usize, axis: usize) -> Array2<f64> {
        let mut field = Array2::<f64>::zeros((n, d));
        for row in 0..n {
            field[[row, axis]] = 1.0;
        }
        field
    }

    /// THE LAW, applied to EVERY analytic topology in the menu.
    ///
    /// `AtomTopology` declares each topology's isometry group, and
    /// `identifiability::exact_orbit_fields` materialises those generators as
    /// exact Killing fields — but nothing ever checked the atom's BASIS against
    /// them. That gap is how a topology got to claim `S²` while its latent was a
    /// cylinder (#2602): every individual piece looked right, and no invariant
    /// tied the basis to the geometry it was supposed to represent.
    ///
    /// The law is one line. Sliding a point along a Killing field is a motion the
    /// geometry cannot distinguish, so it cannot change WHICH functions the atom
    /// can represent:
    ///
    /// ```text
    ///     J(p) · K(p)  ∈  span{Φ}   for every Killing generator K.
    /// ```
    ///
    /// Asserting it per-topology turns "the basis matches the manifold" from a
    /// property maintained by review into one maintained by the suite.
    #[test]
    fn every_topology_basis_is_closed_under_its_declared_killing_fields() {
        let n = 400usize;
        let mut rng = uniform_stream(0xC0FFEE);

        // --- flat charts: the continuous isometries are TRANSLATIONS ---------
        // Fraction-of-period coordinates for the periodic families; the Möbius
        // cover circle has period 2 and its width is a bounded interval.
        struct FlatCase {
            name: &'static str,
            evaluator: Box<dyn SaeBasisEvaluator>,
            dim: usize,
            /// Axes whose translation IS a continuous isometry.
            symmetric_axes: &'static [usize],
            /// Axes whose translation is NOT — asserted to FAIL closure, so the
            /// test cannot pass by a basis that is accidentally translation
            /// invariant in every direction.
            asymmetric_axes: &'static [usize],
            span: fn(usize, f64) -> f64,
        }
        fn unit_span(axis: usize, u: f64) -> f64 {
            // Identity on EVERY axis -- that is exactly what makes this the
            // translation-invariant control that `mobius_span` is contrasted
            // against. The shared `fn(usize, f64) -> f64` pointer type requires
            // the parameter, so it is consumed here rather than hidden behind an
            // underscore: an underscore makes "required by a signature" and
            // "forgotten" look identical, which is what the ban scanner objects
            // to, and it aborts the ROOT build for the whole workspace.
            std::hint::black_box(axis);
            u
        }
        fn mobius_span(axis: usize, u: f64) -> f64 {
            if axis == 0 { 2.0 * u } else { 2.0 * u - 1.0 }
        }
        let cases: Vec<FlatCase> = vec![
            FlatCase {
                name: "periodic S1",
                evaluator: Box::new(PeriodicHarmonicEvaluator::new(7).unwrap()),
                dim: 1,
                symmetric_axes: &[0],
                asymmetric_axes: &[],
                span: unit_span,
            },
            FlatCase {
                name: "flat torus T2",
                evaluator: Box::new(TorusHarmonicEvaluator::new(2, 3).unwrap()),
                dim: 2,
                symmetric_axes: &[0, 1],
                asymmetric_axes: &[],
                span: unit_span,
            },
            FlatCase {
                name: "cylinder S1 x R",
                evaluator: Box::new(CylinderHarmonicEvaluator::new(3, 2).unwrap()),
                dim: 2,
                // Both factors translate: the circle shifts, the line slides.
                symmetric_axes: &[0, 1],
                asymmetric_axes: &[],
                span: unit_span,
            },
            FlatCase {
                name: "mobius band",
                evaluator: Box::new(MobiusHarmonicEvaluator::new(3, 2).unwrap()),
                dim: 2,
                // Only the cover circle. Sliding the WIDTH is not an isometry —
                // the band has a boundary, and the deck flips `w`.
                symmetric_axes: &[0],
                asymmetric_axes: &[1],
                span: mobius_span,
            },
            FlatCase {
                name: "klein bottle",
                evaluator: Box::new(QuotientSpectralEvaluator::klein_bottle(3).unwrap()),
                dim: 2,
                // `Isom` carries the first cover-circle translation only. The
                // second axis contributes a DISCRETE Z2 deck, not a second
                // shift: invariance `f(θ+½, −φ) = f(θ, φ)` makes `∂_φ f`
                // ANTI-invariant, so it provably leaves the retained span.
                symmetric_axes: &[0],
                asymmetric_axes: &[1],
                span: unit_span,
            },
            FlatCase {
                name: "euclidean patch",
                evaluator: Box::new(EuclideanPatchEvaluator::new(2, 2).unwrap()),
                dim: 2,
                symmetric_axes: &[0, 1],
                asymmetric_axes: &[],
                span: |_, u| 4.0 * u - 2.0,
            },
        ];

        for case in &cases {
            let mut coords = Array2::<f64>::zeros((n, case.dim));
            for row in 0..n {
                for axis in 0..case.dim {
                    coords[[row, axis]] = (case.span)(axis, rng());
                }
            }
            let (phi, jet) = case.evaluator.evaluate(coords.view()).unwrap();
            for &axis in case.symmetric_axes {
                let field = constant_field(n, case.dim, axis);
                let derivative = derivative_along_field(&jet, &field);
                let residual = span_residual(&phi, &derivative);
                assert!(
                    residual <= 1.0e-8,
                    "{}: translation along axis {axis} is a declared isometry, but it \
                     carries the basis OUT of its own span (residual {residual:.3e})",
                    case.name
                );
            }
            for &axis in case.asymmetric_axes {
                let field = constant_field(n, case.dim, axis);
                let derivative = derivative_along_field(&jet, &field);
                let residual = span_residual(&phi, &derivative);
                assert!(
                    residual >= 1.0e-6,
                    "{}: axis {axis} is NOT a declared isometry, so closure here would \
                     mean the basis carries a symmetry the manifold does not have \
                     (residual {residual:.3e})",
                    case.name
                );
            }
        }

        // --- the sphere and its quotient: `Isom = O(3)`, `K_a(u) = e_a × u` ---
        let (_, ambient) = sphere_sample(0x511E, n);
        let spherical: Vec<(&str, Box<dyn SaeBasisEvaluator>)> = vec![
            (
                "ambient sphere",
                Box::new(AmbientSphereHarmonicEvaluator::new(2).unwrap()),
            ),
            (
                "ambient RP2",
                Box::new(QuotientSpectralEvaluator::projective_plane_ambient(1).unwrap()),
            ),
        ];
        for (name, evaluator) in &spherical {
            let (phi, jet) = evaluator.evaluate(ambient.view()).unwrap();
            for generator in 0..3 {
                let mut field = Array2::<f64>::zeros((n, 3));
                for row in 0..n {
                    let u = [
                        ambient[[row, 0]],
                        ambient[[row, 1]],
                        ambient[[row, 2]],
                    ];
                    let k = match generator {
                        0 => [0.0, -u[2], u[1]],
                        1 => [u[2], 0.0, -u[0]],
                        _ => [-u[1], u[0], 0.0],
                    };
                    for axis in 0..3 {
                        field[[row, axis]] = k[axis];
                    }
                }
                let derivative = derivative_along_field(&jet, &field);
                let residual = span_residual(&phi, &derivative);
                assert!(
                    residual <= 1.0e-8,
                    "{name}: SO(3) generator {generator} carries the basis out of its own \
                     span (residual {residual:.3e})"
                );
            }
        }
    }

    /// The analytic Hessian and third jet must match finite differences of the
    /// lower jet (the Newton solver consumes them). Order-1 FD of a lower
    /// analytic derivative is well-conditioned, unlike a raw high-order FD of the
    /// value, so this pins every entry: `second_jet` against a central
    /// difference of the `evaluate` jet, and `third_jet` against a central
    /// difference of `second_jet`.
    #[test]
    fn spherical_harmonic_jets_match_finite_differences() {
        let evaluator = SphericalHarmonicEvaluator::new(4).unwrap();
        let coords =
            Array2::from_shape_vec((4, 2), vec![0.3, 0.7, -0.9, 2.1, 1.2, -1.3, -0.1, 4.0])
                .unwrap();
        let h = 1e-5;
        let m = evaluator.basis_size();
        let hess = evaluator.second_jet(coords.view()).unwrap();
        let third = evaluator.third_jet(coords.view()).unwrap();

        let shifted = |axis: usize, step: f64| -> Array2<f64> {
            let mut c = coords.clone();
            for row in 0..c.nrows() {
                c[[row, axis]] += step;
            }
            c
        };

        // second_jet vs central FD of the analytic first jet.
        let mut max_h_err = 0.0_f64;
        for axis in 0..2 {
            let (_, jp) = evaluator.evaluate(shifted(axis, h).view()).unwrap();
            let (_, jm) = evaluator.evaluate(shifted(axis, -h).view()).unwrap();
            for row in 0..coords.nrows() {
                for col in 0..m {
                    for other in 0..2 {
                        let fd = (jp[[row, col, other]] - jm[[row, col, other]]) / (2.0 * h);
                        max_h_err = max_h_err.max((hess[[row, col, other, axis]] - fd).abs());
                    }
                }
            }
        }
        assert!(
            max_h_err < 1e-4,
            "spherical-harmonic Hessian must match FD of the first jet; max err {max_h_err}"
        );

        // third_jet vs central FD of the analytic Hessian.
        let mut max_t_err = 0.0_f64;
        for axis in 0..2 {
            let hp = evaluator.second_jet(shifted(axis, h).view()).unwrap();
            let hm = evaluator.second_jet(shifted(axis, -h).view()).unwrap();
            for row in 0..coords.nrows() {
                for col in 0..m {
                    for a in 0..2 {
                        for b in 0..2 {
                            let fd = (hp[[row, col, a, b]] - hm[[row, col, a, b]]) / (2.0 * h);
                            max_t_err = max_t_err.max((third[[row, col, a, b, axis]] - fd).abs());
                        }
                    }
                }
            }
        }
        assert!(
            max_t_err < 1e-4,
            "spherical-harmonic third jet must match FD of the Hessian; max err {max_t_err}"
        );
    }

    #[test]
    fn quotient_and_cover_constructors_enforce_minimum_order_and_checked_widths() {
        assert!(projective_plane_basis_size(0).is_err());
        assert!(klein_bottle_basis_size(0).is_err());
        assert!(QuotientSpectralEvaluator::projective_plane(0).is_err());
        assert!(QuotientSpectralEvaluator::klein_bottle(0).is_err());
        assert!(QuotientSpectralEvaluator::klein_bottle(1).is_err());

        let projective_plane = QuotientSpectralEvaluator::projective_plane(1).unwrap();
        assert_eq!(projective_plane.basis_size(), 6);
        assert_eq!(
            projective_plane.basis_size(),
            projective_plane_basis_size(1).unwrap()
        );
        let klein_bottle = QuotientSpectralEvaluator::klein_bottle(2).unwrap();
        assert_eq!(klein_bottle.basis_size(), 13);
        assert_eq!(
            klein_bottle.basis_size(),
            klein_bottle_basis_size(2).unwrap()
        );
        assert_eq!(
            klein_bottle
                .phi_eta_split(klein_bottle.basis_size())
                .unwrap()
                .base_cols
                .len(),
            7
        );

        assert!(projective_plane_basis_size(usize::MAX).is_err());
        assert!(klein_bottle_basis_size(usize::MAX).is_err());
        assert!(QuotientSpectralEvaluator::projective_plane(usize::MAX).is_err());
        assert!(QuotientSpectralEvaluator::klein_bottle(usize::MAX).is_err());
        assert!(SphericalHarmonicEvaluator::new(usize::MAX).is_err());
        assert!(TorusHarmonicEvaluator::new(1, usize::MAX).is_err());
        assert!(TorusHarmonicEvaluator::new(usize::BITS as usize, 1).is_err());
    }

    #[test]
    fn mobius_basis_is_invariant_under_deck_transform() {
        let evaluator = MobiusHarmonicEvaluator::new(3, 2).unwrap();
        let coords = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, -0.8, 0.17, -0.3, 0.51, 0.0, 0.88, 0.4, 1.41, 0.9],
        )
        .unwrap();
        let mut twins = coords.clone();
        for row in 0..twins.nrows() {
            twins[[row, 0]] += 1.0;
            twins[[row, 1]] = -twins[[row, 1]];
        }
        let (phi, _) = evaluator.evaluate(coords.view()).unwrap();
        let (phi_twin, _) = evaluator.evaluate(twins.view()).unwrap();
        let max_error = (&phi - &phi_twin)
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_error <= 32.0 * f64::EPSILON,
            "every basis column must descend to the Möbius quotient; max deck error {max_error}"
        );
    }

    /// The in-place `evaluate_into` must reproduce `evaluate` EXACTLY, even when
    /// the target workspace arrives pre-loaded with garbage — the hot Newton loop
    /// reuses one buffer across trials, so any entry the in-place path forgets to
    /// (re)write would carry a stale value forward. Prefilling with a sentinel and
    /// asserting bit-equality against a fresh `evaluate` pins that.
    fn assert_into_matches_evaluate(eval: &dyn SaeBasisEvaluator, coords: &Array2<f64>) {
        let (phi_ref, jet_ref) = eval.evaluate(coords.view()).expect("evaluate");
        let mut phi = Array2::<f64>::from_elem(phi_ref.dim(), 999.0);
        let mut jet = Array3::<f64>::from_elem(jet_ref.dim(), 999.0);
        eval.evaluate_into(&mut phi, &mut jet, coords.view())
            .expect("evaluate_into");
        assert_eq!(
            phi, phi_ref,
            "evaluate_into Φ must equal evaluate Φ exactly"
        );
        assert_eq!(
            jet, jet_ref,
            "evaluate_into jet must equal evaluate jet exactly"
        );
    }

    /// Reusing ONE workspace across two different coordinate sets (the line-search
    /// cadence) must leave no contamination: the second fill must match a fresh
    /// `evaluate` on the second coordinates.
    fn assert_workspace_reuse(
        eval: &dyn SaeBasisEvaluator,
        coords_a: &Array2<f64>,
        coords_b: &Array2<f64>,
    ) {
        let (phi_a, jet_a) = eval.evaluate(coords_a.view()).expect("evaluate a");
        let mut phi = Array2::<f64>::zeros(phi_a.dim());
        let mut jet = Array3::<f64>::zeros(jet_a.dim());
        eval.evaluate_into(&mut phi, &mut jet, coords_a.view())
            .expect("into a");
        assert_eq!(phi, phi_a);
        assert_eq!(jet, jet_a);
        let (phi_b_ref, jet_b_ref) = eval.evaluate(coords_b.view()).expect("evaluate b");
        eval.evaluate_into(&mut phi, &mut jet, coords_b.view())
            .expect("into b (reused workspace)");
        assert_eq!(
            phi, phi_b_ref,
            "reused workspace Φ must not carry stale data"
        );
        assert_eq!(
            jet, jet_b_ref,
            "reused workspace jet must not carry stale data"
        );
    }

    #[test]
    fn periodic_harmonic_evaluate_into_matches() {
        let eval = PeriodicHarmonicEvaluator::new(5).unwrap();
        let coords_a = Array2::from_shape_vec((4, 1), vec![0.10, 0.35, 0.60, 0.85]).unwrap();
        let coords_b = Array2::from_shape_vec((4, 1), vec![0.20, 0.45, 0.70, 0.05]).unwrap();
        assert_into_matches_evaluate(&eval, &coords_a);
        assert_workspace_reuse(&eval, &coords_a, &coords_b);
    }

    #[test]
    fn euclidean_patch_evaluate_into_matches() {
        let eval = EuclideanPatchEvaluator::new(2, 2).unwrap();
        let coords_a =
            Array2::from_shape_vec((4, 2), vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8])
                .unwrap();
        let coords_b =
            Array2::from_shape_vec((4, 2), vec![-0.3, 0.9, 0.2, -0.1, 0.5, 0.5, -0.7, 0.3])
                .unwrap();
        assert_into_matches_evaluate(&eval, &coords_a);
        assert_workspace_reuse(&eval, &coords_a, &coords_b);
    }

    #[test]
    fn torus_harmonic_evaluate_into_matches() {
        let eval = TorusHarmonicEvaluator::new(2, 2).unwrap();
        let coords_a =
            Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
        let coords_b =
            Array2::from_shape_vec((4, 2), vec![0.9, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
                .unwrap();
        assert_into_matches_evaluate(&eval, &coords_a);
        assert_workspace_reuse(&eval, &coords_a, &coords_b);
    }

    #[test]
    fn default_evaluate_into_matches_for_unspecialized_evaluator() {
        // `AmbientSphereHarmonicEvaluator` does not override `evaluate_into`, so
        // this pins the allocate-and-copy DEFAULT trait method against `evaluate`
        // — the role the removed chart evaluator used to play, for the same
        // reason.
        let eval = AmbientSphereHarmonicEvaluator::new(2).unwrap();
        let coords_a =
            Array2::from_shape_vec((3, 3), vec![0.2, 0.5, -0.4, 1.1, 0.9, -0.7, 0.3, -0.2, 0.8])
                .unwrap();
        let coords_b =
            Array2::from_shape_vec((3, 3), vec![-0.1, 0.3, 0.6, -0.9, -0.5, 0.8, 0.4, 0.1, -0.6])
                .unwrap();
        assert_into_matches_evaluate(&eval, &coords_a);
        assert_workspace_reuse(&eval, &coords_a, &coords_b);
    }

    #[test]
    fn evaluate_into_rejects_mismatched_buffer() {
        let eval = PeriodicHarmonicEvaluator::new(5).unwrap();
        let coords = Array2::from_shape_vec((4, 1), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        // Wrong Φ width (4 columns instead of 5) must be rejected, not silently
        // written past — the shape guard `refresh_basis` relied on.
        let mut phi = Array2::<f64>::zeros((4, 4));
        let mut jet = Array3::<f64>::zeros((4, 5, 1));
        assert!(
            eval.evaluate_into(&mut phi, &mut jet, coords.view())
                .is_err()
        );
    }
}

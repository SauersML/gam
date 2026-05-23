//! `LatentCoord` — per-row latent coordinates as a first-class gamfit parameter.
//!
//! See `proposals/latent_coord.md` for the full design.
//!
//! ## Summary
//!
//! `LatentCoordValues` is the structural sibling of [`SpatialLogKappaCoords`]
//! (see [`crate::terms::smooth`]). Both store a flat `Array1<f64>` that the
//! REML/IFT outer loop treats as *design-moving, non-penalty-like*
//! hyper-coordinates. `SpatialLogKappaCoords` holds one (or a handful of) `ψ`
//! per spatial term — log-anisotropies of the kernel. `LatentCoordValues`
//! holds an `N × d` matrix of per-row latent coordinates `t_n ∈ ℝ^d`.
//!
//! For a Duchon (or any radial) basis:
//!
//! ```text
//! Φ_{n,k} = φ(‖t_n − c_k‖),
//! ∂Φ_{n,k}/∂t_n = φ'(r_{nk}) · (t_n − c_k) / r_{nk}.
//! ```
//!
//! The radial-gradient `φ'(r)` is the *same scalar* the `ψ` machinery already
//! computes via [`crate::basis::duchon_radial_jets`]; the chain rule
//! `(t_n − c_k)/r_{nk}` is what differs between "differentiate against the
//! kernel scale ψ" and "differentiate against the first kernel argument t".
//! Everything downstream of `HyperDesignDerivative::from_implicit` (matrix-free
//! Newton, IFT cache, persistent warm-start, REML/LAML evaluation) is reused
//! verbatim.
//!
//! ## Gauge fixing
//!
//! The bare data-fit `½‖y − Φ(t)β‖²` is invariant under any diffeomorphism
//! `t ↦ φ(t)` (absorb into a re-fit β), so the inner Hessian in the latent
//! block is singular and IFT breaks. [`LatentIdMode`] enumerates the
//! gauge-fix penalties exposed at the configuration layer:
//!
//! * [`LatentIdMode::AuxPrior`] — iVAE-style auxiliary-conditional prior
//!   `R_id(t,u) = ½ μ ‖t − ĥ(u)‖²` where `ĥ` is a small ridge / linear map
//!   fit internally against the auxiliary `u`. `μ` is REML-selectable like a
//!   smoothing parameter. This is the principled identifiability fix
//!   (Khemakhem et al. 2020).
//! * [`LatentIdMode::DimSelection`] — ARD on each latent axis. One ridge
//!   penalty per axis; REML drives unused axes' precision to infinity only
//!   after `AuxPrior` or a future isometry prior fixes the gauge.
//! * [`LatentIdMode::None`] — no gauge fix. Useful only as an explicit
//!   opt-out; the caller is responsible for separately providing a unique
//!   inner minimum (e.g. via a custom penalty).
//!
//! `IsometryToReference` is deferred to a follow-up (see proposal §4(b)).

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

/// Choice of auxiliary-prior conditional mean estimator `ĥ(u)`.
///
/// `Ridge` is the cheap default that closes form (one `K_u × K_u` solve);
/// `Linear` is equivalent to `Ridge` with zero ridge and is intended for
/// auxiliaries `u` that are already low-dimensional and well-conditioned.
#[derive(Debug, Clone, Copy)]
pub enum AuxPriorFamily {
    /// Ridge regression `t ≈ U · A` with a small diagonal regularizer.
    /// The default ridge strength is `1e-6 · trace(UᵀU)/p`, which is
    /// numerically benign and never under-constrains the fit when
    /// `n_obs > p`.
    Ridge,
    /// Plain linear projection (no ridge). Errors out at construction if
    /// `UᵀU` is singular.
    Linear,
}

/// Strength of the auxiliary-prior identifiability penalty.
///
/// `Auto` defers the choice to REML — the strength is added to the outer
/// vector as one extra `ρ`-axis (one log-precision per `LatentCoord`). When
/// the caller supplies an explicit `Fixed(μ)` the strength is held constant
/// throughout the fit; useful for warm-starts and reproducibility.
#[derive(Debug, Clone, Copy)]
pub enum AuxPriorStrength {
    Auto,
    Fixed(f64),
}

/// Identifiability / gauge-fix mode for a [`LatentCoordValues`] block.
///
/// `AuxPrior` is currently the only standalone gauge-fixing mode; see the
/// module docstring. `DimSelection` must be paired with `AuxPrior` (or a
/// future isometry mode) by higher-level assembly before fitting.
#[derive(Debug, Clone)]
pub enum LatentIdMode {
    /// Conditional Gaussian prior `p(t | u)` with mean `ĥ(u)` fit by
    /// `family`. The penalty contribution is
    /// `R_id = ½ μ · ‖t − ĥ(u)‖²`. `u` has shape `(n_obs, p)`.
    AuxPrior {
        u: Array2<f64>,
        family: AuxPriorFamily,
        strength: AuxPriorStrength,
    },
    /// ARD over latent axes. One ridge penalty per latent axis; the per-axis
    /// log-precision joins the outer ρ vector. `init_log_precision` seeds
    /// the per-axis ρ — a vector of length `d`. `None` defaults to a flat
    /// zero seed (precision = 1 on every axis).
    DimSelection {
        init_log_precision: Option<Array1<f64>>,
    },
    /// No gauge fix. Inner Hessian is rank-deficient; results are not
    /// uniquely defined. Intended only for the explicit "I supply my own
    /// gauge constraint via the smoothing penalty" pathway.
    None,
}

impl LatentIdMode {
    /// Fixes the audit finding that ARD/DimSelection alone is rotation
    /// symmetric and therefore not a standalone identifiability mode.
    pub fn is_identifiable(&self) -> bool {
        matches!(self, Self::AuxPrior { .. })
    }

    fn reject_dim_selection_alone(&self) {
        if matches!(self, Self::DimSelection { .. }) {
            panic!(
                "LatentIdMode::DimSelection is not a standalone gauge fix; pair ARD with AuxPrior or Isometry"
            );
        }
    }
}

/// Carrier for the `∂Φ/∂t` chain-rule input, dispatched on basis kind by
/// [`LatentCoordValues::design_gradient_wrt_t_dispatch`].
///
/// * [`InputLocationDerivative::Radial`] is the *radial-kernel* path: the
///   caller supplies the scalar `φ'(r_{n,k})` matrix together with the
///   center coordinates, and the chain rule
///   `∂Φ/∂t = φ'(r) · (t − c) / r` is applied internally. This covers every
///   isotropic radial basis — Duchon (any nullspace order), Matérn (every
///   supported half-integer ν), and anything else whose pointwise
///   gradient is radial. Helpers:
///   [`crate::terms::basis::duchon_radial_first_derivative_nd`],
///   [`crate::terms::basis::matern_radial_first_derivative_nd`].
/// * [`InputLocationDerivative::Jet`] is the *pre-computed jet* path: the
///   caller has already assembled a closed-form `(N, K, d)` tensor for a
///   basis whose chain rule is not a simple radial scalar times a unit
///   vector. Sphere kernels carry the tangent-direction times `K'(cos γ)`;
///   periodic-cyclic B-splines carry the closed-form cardinal derivative;
///   tensor-product B-splines carry the product-rule mix. Helpers:
///   [`crate::terms::basis::sphere_first_derivative_nd`],
///   [`crate::terms::basis::periodic_bspline_first_derivative_nd`],
///   [`crate::terms::basis::bspline_tensor_first_derivative`].
///
/// The dispatch is an enum rather than a trait because each path's
/// arguments differ structurally (radial bases reuse `φ'(r)` shared with
/// the ψ-chain machinery; jet bases ship the full tensor). All chain rules
/// are analytic and closed-form; no autodiff, no finite differences.
pub enum InputLocationDerivative<'a> {
    /// Radial-kernel chain rule. The chain rule `(t − c)/r` is reconstructed
    /// internally from the scalar `φ'(r_{n,k})` matrix and the center
    /// coordinates.
    Radial {
        kernel_first_derivative: ArrayView2<'a, f64>,
        centers: ArrayView2<'a, f64>,
    },
    /// Pre-computed analytic `(n_obs, n_centers, latent_dim)` jet.
    Jet(ArrayView3<'a, f64>),
}

/// Per-row latent coordinates `t ∈ ℝ^{N × d}` stored as a flat
/// row-major `Array1<f64>` of length `n_obs * latent_dim`.
///
/// The flat-`Array1` layout mirrors [`crate::terms::smooth::SpatialLogKappaCoords`]
/// so the same `HyperDesignDerivative::from_implicit` / `DirectionalHyperParam`
/// outer plumbing can consume it without modification.
#[derive(Debug, Clone)]
pub struct LatentCoordValues {
    /// Flattened (n_obs, latent_dim) latent matrix, row-major
    /// (so `values[n * d + k] = t_n[k]`).
    values: Array1<f64>,
    /// Number of rows `N`.
    n_obs: usize,
    /// Number of latent dimensions `d`.
    latent_dim: usize,
    /// Identifiability / gauge-fix mode.
    id_mode: LatentIdMode,
}

impl LatentCoordValues {
    /// Construct from a dense `(n_obs, latent_dim)` matrix.
    pub fn from_matrix(matrix: ArrayView2<'_, f64>, id_mode: LatentIdMode) -> Self {
        id_mode.reject_dim_selection_alone();
        let n_obs = matrix.nrows();
        let latent_dim = matrix.ncols();
        let mut values = Array1::<f64>::zeros(n_obs * latent_dim);
        for n in 0..n_obs {
            for k in 0..latent_dim {
                values[n * latent_dim + k] = matrix[[n, k]];
            }
        }
        Self {
            values,
            n_obs,
            latent_dim,
            id_mode,
        }
    }

    /// Construct directly from a flat (`n_obs * latent_dim`) array.
    pub fn from_flat(
        values: Array1<f64>,
        n_obs: usize,
        latent_dim: usize,
        id_mode: LatentIdMode,
    ) -> Self {
        id_mode.reject_dim_selection_alone();
        debug_assert_eq!(
            values.len(),
            n_obs * latent_dim,
            "LatentCoordValues::from_flat: length {} != n_obs * latent_dim = {}",
            values.len(),
            n_obs * latent_dim
        );
        Self {
            values,
            n_obs,
            latent_dim,
            id_mode,
        }
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Total length of the flat value array (= `n_obs * latent_dim`).
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn id_mode(&self) -> &LatentIdMode {
        &self.id_mode
    }

    /// View the flat value array.
    pub fn as_flat(&self) -> &Array1<f64> {
        &self.values
    }

    /// View row `n` as a length-`d` slice.
    pub fn row(&self, n: usize) -> &[f64] {
        let start = n * self.latent_dim;
        let end = start + self.latent_dim;
        &self.values.as_slice().expect("contiguous")[start..end]
    }

    /// Materialize as a dense `(n_obs, latent_dim)` matrix view.
    /// Useful when handing `t` to a row-major basis evaluator
    /// (e.g. `build_duchon_basis`).
    pub fn as_matrix(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n_obs, self.latent_dim));
        for n in 0..self.n_obs {
            for k in 0..self.latent_dim {
                out[[n, k]] = self.values[n * self.latent_dim + k];
            }
        }
        out
    }

    /// Mutable write back of the flat value array, e.g. after a Newton step.
    pub fn set_flat(&mut self, flat: ArrayView1<'_, f64>) {
        debug_assert_eq!(flat.len(), self.values.len());
        self.values.assign(&flat);
    }

    /// Apply this latent block back to a `TermCollectionSpec`-style covariate
    /// table: returns the `(N, d)` materialized matrix that downstream basis
    /// evaluators (Duchon, Matérn, ...) take as their feature input.
    ///
    /// This mirrors [`crate::terms::smooth::SpatialLogKappaCoords::apply_tospec`],
    /// but the carrier on the spec side is the data-row covariate block rather
    /// than the per-term `length_scale`. The spec-mutation is handled at the
    /// call site (the consuming term needs to know which columns of its
    /// feature view to overwrite).
    pub fn apply_tospec(&self) -> Array2<f64> {
        self.as_matrix()
    }

    /// Compute `∂Φ/∂t` for a radial-kernel design Φ — the original
    /// Duchon/Matérn path. See [`Self::design_gradient_wrt_t_dispatch`] for
    /// the basis-agnostic dispatch entry point.
    ///
    /// `kernel_first_derivative` is `φ'(r_{n,k})` for each (row, center)
    /// pair, shape `(n_obs, n_centers)`. `centers` is `(n_centers, d)`.
    /// Returns a `(n_obs, n_centers, d)` jet whose `(n, k, a)` entry is
    /// `∂Φ_{n,k} / ∂t_{n,a} = φ'(r_{n,k}) · (t_{n,a} − c_{k,a}) / r_{n,k}`.
    ///
    /// At `r = 0` the unit vector `(t − c)/r` is undefined; convention is
    /// to set the jet to zero there (the kernel is smooth at the origin
    /// for the supported Duchon orders, so `φ'(0) = 0` anyway and the
    /// product is the right limit).
    pub fn design_gradient_wrt_t(
        &self,
        kernel_first_derivative: ArrayView2<'_, f64>,
        centers: ArrayView2<'_, f64>,
    ) -> Array3<f64> {
        let n_obs = self.n_obs;
        let d = self.latent_dim;
        let n_centers = centers.nrows();
        debug_assert_eq!(centers.ncols(), d);
        debug_assert_eq!(kernel_first_derivative.shape(), &[n_obs, n_centers]);
        let mut jet = Array3::<f64>::zeros((n_obs, n_centers, d));
        for n in 0..n_obs {
            let t_n = self.row(n);
            for k in 0..n_centers {
                // r = ‖t_n − c_k‖
                let mut r2 = 0.0_f64;
                for a in 0..d {
                    let delta = t_n[a] - centers[[k, a]];
                    r2 += delta * delta;
                }
                let r = r2.sqrt();
                if r == 0.0 {
                    continue;
                }
                let phi_prime = kernel_first_derivative[[n, k]];
                if phi_prime == 0.0 {
                    continue;
                }
                let scale = phi_prime / r;
                for a in 0..d {
                    jet[[n, k, a]] = scale * (t_n[a] - centers[[k, a]]);
                }
            }
        }
        jet
    }

    /// Compute `∂Φ/∂t` for an arbitrary supported basis kind, by dispatching
    /// to the right closed-form chain rule.
    ///
    /// All radial-kernel bases (Duchon, Matérn) reduce to the same shape
    /// `φ'(r)`-times-unit-vector chain that `design_gradient_wrt_t` already
    /// implements. Non-radial bases (sphere, periodic-cyclic B-spline, tensor
    /// B-spline) carry their own analytic `(N, K, d)` jet — the caller
    /// pre-builds that jet using the matching `*_first_derivative_nd` helper
    /// in [`crate::terms::basis`] and passes it in via
    /// [`InputLocationDerivative::Jet`].
    ///
    /// This is the single entry point the outer optimizer should call; it
    /// stays in lock-step with the kernel-parameter ψ chain rule that
    /// `SpatialLogKappaCoords` uses (re-pointed at the first kernel argument
    /// rather than at the anisotropy ψ).
    pub fn design_gradient_wrt_t_dispatch(
        &self,
        input: InputLocationDerivative<'_>,
    ) -> Array3<f64> {
        match input {
            InputLocationDerivative::Radial {
                kernel_first_derivative,
                centers,
            } => self.design_gradient_wrt_t(kernel_first_derivative, centers),
            InputLocationDerivative::Jet(jet) => {
                // The non-radial helpers already produce a (N, K, d) tensor
                // in the same layout `contract_gradient` consumes. Return a
                // copy so the caller owns the data and is decoupled from the
                // source array's lifetime.
                jet.to_owned()
            }
        }
    }

    /// Contract a downstream gradient `∂L/∂Φ ∈ ℝ^(n_obs × n_centers)` and a
    /// `design_gradient_wrt_t` jet into a flat `∂L/∂t ∈ ℝ^(n_obs * d)`.
    ///
    /// This is the N-D generalization of
    /// `gam_pyffi::contract_position_gradient` (1-D), used inside the
    /// `_backward` pyffi entry point.
    pub fn contract_gradient(
        grad_phi: ArrayView2<'_, f64>,
        jet: &Array3<f64>,
    ) -> Array1<f64> {
        let n_obs = jet.shape()[0];
        let n_centers = jet.shape()[1];
        let d = jet.shape()[2];
        debug_assert_eq!(grad_phi.shape(), &[n_obs, n_centers]);
        let mut grad_t = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                let mut acc = 0.0_f64;
                for k in 0..n_centers {
                    acc += grad_phi[[n, k]] * jet[[n, k, a]];
                }
                grad_t[n * d + a] = acc;
            }
        }
        grad_t
    }
}

/// Auxiliary-prior penalty contribution: returns the per-row reference
/// coordinates `ĥ(u_n)` shape `(n_obs, d)` and the effective strength `μ`.
///
/// `t_target` is broadcast across the inner ridge of `½ μ · ‖t − t_target‖²`,
/// which the call site folds into the Y-stack via a virtual-row augmentation
/// (`y' = [y; √μ · t_target]`, `X' = [X; √μ · I_d ⊗ row-block]`). This
/// keeps the inner solver Gaussian-closed-form.
///
/// For `AuxPriorFamily::Ridge` the conditional mean is the closed-form ridge
/// regression `(UᵀU + ε I)⁻¹ UᵀT` evaluated at each row's `u_n`. For
/// `Linear` the ridge is zero (which raises if `UᵀU` is singular).
pub fn aux_prior_targets(
    t: ArrayView2<'_, f64>,
    u: ArrayView2<'_, f64>,
    family: AuxPriorFamily,
) -> Result<Array2<f64>, String> {
    let n_obs = t.nrows();
    let d = t.ncols();
    if u.nrows() != n_obs {
        return Err(format!(
            "aux_prior_targets: u has {} rows but t has {}",
            u.nrows(),
            n_obs
        ));
    }
    let p = u.ncols();
    if p == 0 {
        return Err("aux_prior_targets: auxiliary u must have at least one column".into());
    }
    // gram = UᵀU  (p × p)
    let mut gram = Array2::<f64>::zeros((p, p));
    for n in 0..n_obs {
        for i in 0..p {
            for j in 0..p {
                gram[[i, j]] += u[[n, i]] * u[[n, j]];
            }
        }
    }
    let ridge_eps = match family {
        AuxPriorFamily::Ridge => {
            let trace: f64 = (0..p).map(|i| gram[[i, i]]).sum();
            (1e-6 * trace / p as f64).max(1e-12)
        }
        AuxPriorFamily::Linear => 0.0,
    };
    for i in 0..p {
        gram[[i, i]] += ridge_eps;
    }
    // rhs = UᵀT  (p × d)
    let mut rhs = Array2::<f64>::zeros((p, d));
    for n in 0..n_obs {
        for i in 0..p {
            for k in 0..d {
                rhs[[i, k]] += u[[n, i]] * t[[n, k]];
            }
        }
    }
    let coeffs = solve_spd(gram.view(), rhs.view())?;
    // targets = U · coeffs  (n_obs × d)
    let mut targets = Array2::<f64>::zeros((n_obs, d));
    for n in 0..n_obs {
        for k in 0..d {
            let mut acc = 0.0_f64;
            for i in 0..p {
                acc += u[[n, i]] * coeffs[[i, k]];
            }
            targets[[n, k]] = acc;
        }
    }
    Ok(targets)
}

/// Lightweight Cholesky-based SPD solve. Keeps this module dependency-free
/// from the broader faer-wrapping surface; matrices here are tiny
/// (`p × p` with p = aux-feature count, typically O(10)).
fn solve_spd(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err("solve_spd: A must be square".into());
    }
    if b.nrows() != n {
        return Err("solve_spd: RHS row count mismatch".into());
    }
    // In-place Cholesky factorization. We pay the O(n³) copy + O(n³) factor
    // up front; n is tiny in the auxiliary-prior path.
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "solve_spd: non-positive pivot {sum} at index {i} \
                         (matrix is not positive definite)"
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Solve L y = b, then Lᵀ x = y, column by column.
    let d = b.ncols();
    let mut out = Array2::<f64>::zeros((n, d));
    for col in 0..d {
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = b[[i, col]];
            for k in 0..i {
                sum -= l[[i, k]] * y[k];
            }
            y[i] = sum / l[[i, i]];
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= l[[k, i]] * out[[k, col]];
            }
            out[[i, col]] = sum / l[[i, i]];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn from_matrix_roundtrip() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let lc = LatentCoordValues::from_matrix(m.view(), LatentIdMode::None);
        assert_eq!(lc.n_obs(), 3);
        assert_eq!(lc.latent_dim(), 2);
        let back = lc.as_matrix();
        assert_eq!(back, m);
    }

    #[test]
    fn row_access() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let lc = LatentCoordValues::from_matrix(m.view(), LatentIdMode::None);
        assert_eq!(lc.row(0), &[1.0, 2.0]);
        assert_eq!(lc.row(1), &[3.0, 4.0]);
    }
}

//! Constant-curvature (`M_κ`) smooth term: basis + penalty over the
//! κ-stereographic chart (#944, stage 3 step 1).
//!
//! The term is the κ-generic sibling of the intrinsic-S² Wahba smooth
//! (`sphere_spec.rs` / `build_spherical_spline_basis`): a reproducing-kernel
//! basis on a center set, with the kernel Gram on the centers as the RKHS
//! roughness penalty and a coefficient-space sum-to-zero constraint for
//! identifiability. Where the Wahba smooth hard-codes S² (lat/lon chart,
//! Legendre kernels), this term takes the geometry from
//! [`crate::geometry::constant_curvature::ConstantCurvature`] at an explicit
//! curvature κ, so one construction covers the whole interpolation
//! `S^d(1/√κ) → ℝ^d → H^d(1/√−κ)` through κ = 0.
//!
//! # Kernel
//!
//! `K_κ(x, y) = exp(−d_κ(x, y) / ℓ)` — the geodesic-exponential kernel, where
//! `d_κ` is the exact constant-curvature geodesic distance in the
//! κ-stereographic chart. The geodesic distance is a kernel of conditionally
//! negative type on all three constant-curvature space forms (Schoenberg 1942
//! for `S^d`; classical CND of `‖·‖` on `ℝ^d`; Faraut–Harzallah 1974 for
//! `H^d`), so `exp(−c·d_κ)` is positive definite for every `c > 0` and every
//! κ — the Gram on distinct centers is strictly PD, which is exactly what the
//! RKHS penalty construction needs. At κ = 0 the chart carries the doubled
//! gauge (`metric 4δ`, `d_0(x, y) = 2‖x − y‖`), so the κ = 0 term is the
//! Euclidean exponential (Matérn-½) kernel smooth with effective Euclidean
//! range `ℓ/2`.
//!
//! # κ-differentiability contract (what the ψ-channel stage consumes)
//!
//! Every κ-moving piece of this construction is differentiable in κ via the
//! exact κ-jets landed in stage 2, and every κ-FIXED piece is documented as
//! such so the later ψ-channel wiring (`∂X/∂κ`, `∂S/∂κ` into the LAML outer
//! gradient, Matérn iso-κ optimizer as the template) needs no new calculus:
//!
//! - **Centers are κ-fixed.** Center selection runs in chart coordinates
//!   (farthest-point / k-means / user-provided) and deliberately does NOT
//!   consult κ, so `∂(centers)/∂κ ≡ 0` and the design moves with κ only
//!   through the kernel. A κ-dependent center rule would add an
//!   uncontrolled, non-smooth term to the design drift.
//! - **The length scale ℓ is κ-fixed.** The auto-initialized ℓ is derived
//!   from chart-coordinate (κ = 0 gauge) center spacing only, and an
//!   explicit user ℓ is a constant. `∂ℓ/∂κ ≡ 0`.
//! - **The constraint transform `z` is κ-fixed.** Uniform coefficient
//!   weights; at fit time the global identifiability pipeline composes the
//!   parametric orthogonalization onto it and the result is FROZEN
//!   (mirroring `SphericalSplineIdentifiability::FrozenTransform`, #532), so
//!   the predict/ψ-trial rebuild replays the same `z` verbatim.
//! - **The kernel has exact κ-jets.** `∂K/∂κ` and `∂²K/∂κ²` follow from
//!   `distance_kappa_jet` (Tower4-exact, FD-gated) by the chain rule — see
//!   [`constant_curvature_kernel_kappa_jets`]. Therefore:
//!   `∂X_raw/∂κ = ∂K(data, centers)/∂κ`, realized design drift
//!   `∂X/∂κ = (∂K/∂κ)·z`, and penalty drift `∂S_raw/∂κ = zᵀ(∂K(centers,
//!   centers)/∂κ)z` are all available in closed form from this module today.
//!   (The penalty handed to the optimizer is Frobenius-normalized; the
//!   ψ-channel must route its κ-derivative through the same normalization
//!   rule — `normalize_penaltywith_psi_derivatives` is the existing seam.)
//! - **Available but not yet consumed:** `log_map_kappa_jet` /
//!   `exp_map_kappa_jet` cover future geodesic/normal-coordinate basis
//!   variants (e.g. tangent-space designs); the distance jet is the only one
//!   this kernel construction needs.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::geometry::constant_curvature::{ConstantCurvature, distance_kappa_jet};

use super::{
    BasisBuildResult, BasisError, BasisMetadata, BasisPsiDerivativeBundle,
    BasisPsiDerivativeResult, BasisPsiSecondDerivativeResult, CenterStrategy, PenaltyCandidate,
    PenaltyInfo, PenaltySource, filter_active_penalty_candidates_with_ops, normalize_penalty,
    normalize_penaltywith_psi_derivatives, select_centers_by_strategy,
    weighted_coefficient_sum_to_zero_transform,
};

/// Realized-design identifiability policy for the constant-curvature smooth.
/// Mirrors [`super::SphericalSplineIdentifiability`] (#532): the fit-time
/// center-space sum-to-zero `z` gets the parametric orthogonalization composed
/// onto it by the global identifiability pipeline, and the composed transform
/// is frozen here so predict-time (and future per-ψ-trial) rebuilds replay it
/// verbatim instead of recomputing `z` from the centers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ConstantCurvatureIdentifiability {
    /// Fit-time default: uniform-weight coefficient sum-to-zero over the
    /// centers (`Σ_j α_j = 0`), then global parametric residualization.
    #[default]
    CenterSumToZero,
    /// Predict-time replay: the frozen composed transform captured at fit
    /// time. `transform.nrows()` equals the number of centers.
    FrozenTransform { transform: Array2<f64> },
}

/// Constant-curvature smooth configuration (`curv(x, z, kappa = …)`).
///
/// The chart inputs are the raw feature columns interpreted as
/// κ-stereographic chart coordinates: any finite point for κ ≥ 0, the open
/// ball `‖x‖ < 1/√(−κ)` for κ < 0. The default κ = 0 reproduces a Euclidean
/// exponential-kernel smooth (in the doubled κ = 0 chart gauge), so the term
/// is safe to use as a drop-in flat smooth until κ becomes a fitted
/// ψ-coordinate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantCurvatureBasisSpec {
    /// Center/knot selection strategy in chart coordinates. Deliberately
    /// κ-independent (see the module-level κ-contract).
    pub center_strategy: CenterStrategy,
    /// Sectional curvature κ of the latent/feature geometry. Fixed at build
    /// time; the later ψ-channel stage promotes it to a fitted outer
    /// coordinate consuming this module's exact κ-jets.
    pub kappa: f64,
    /// Geodesic kernel range ℓ in `K_κ = exp(−d_κ/ℓ)`. The `0.0` sentinel
    /// requests the κ-independent auto initialization
    /// ([`realized_constant_curvature_length_scale`]); the realized value is
    /// persisted in [`BasisMetadata::ConstantCurvature`] and frozen back into
    /// the spec for predict-time replay.
    pub length_scale: f64,
    /// Add the ridge-like shrinkage penalty alongside the RKHS Gram penalty.
    pub double_penalty: bool,
    /// Realized-design identifiability policy (see type docs).
    #[serde(default)]
    pub identifiability: ConstantCurvatureIdentifiability,
}

impl Default for ConstantCurvatureBasisSpec {
    fn default() -> Self {
        Self {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 50 },
            kappa: 0.0,
            length_scale: 0.0,
            double_penalty: true,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        }
    }
}

/// Validate that every row of `points` is finite and inside the
/// κ-stereographic chart (`1 + κ‖x‖² > 0`; automatic for κ ≥ 0, the open-ball
/// constraint for κ < 0).
fn validate_chart_points(
    points: ArrayView2<'_, f64>,
    kappa: f64,
    what: &str,
) -> Result<(), BasisError> {
    for (i, row) in points.outer_iter().enumerate() {
        let mut nx2 = 0.0_f64;
        for &v in row.iter() {
            if !v.is_finite() {
                crate::bail_invalid_basis!(
                    "constant-curvature {what} row {i} has a non-finite coordinate"
                );
            }
            nx2 += v * v;
        }
        if 1.0 + kappa * nx2 <= 0.0 {
            crate::bail_invalid_basis!(
                "constant-curvature {what} row {i} lies outside the κ-stereographic chart \
                 (need 1 + κ·‖x‖² > 0; got κ = {kappa}, ‖x‖² = {nx2}); for κ < 0 the chart is \
                 the open ball ‖x‖ < 1/√(−κ)"
            );
        }
    }
    Ok(())
}

/// `K_κ(data, centers)` — the geodesic-exponential kernel matrix
/// `exp(−d_κ(x_i, c_j)/ℓ)`.
pub fn constant_curvature_kernel_matrix(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kappa: f64,
    length_scale: f64,
) -> Result<Array2<f64>, BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "constant-curvature kernel dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    if !(length_scale.is_finite() && length_scale > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature kernel needs a positive finite length_scale; got {length_scale}"
        );
    }
    validate_chart_points(data, kappa, "data")?;
    validate_chart_points(centers, kappa, "centers")?;
    let manifold = ConstantCurvature::new(data.ncols(), kappa);
    let mut out = Array2::<f64>::zeros((data.nrows(), centers.nrows()));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| -> Result<(), BasisError> {
            for (j, c) in centers.outer_iter().enumerate() {
                let d = manifold.distance(data.row(i), c).map_err(|e| {
                    BasisError::InvalidInput(format!(
                        "constant-curvature distance failed at (row {i}, center {j}): {e}"
                    ))
                })?;
                row[j] = (-d / length_scale).exp();
            }
            Ok(())
        })?;
    Ok(out)
}

/// `(K, ∂K/∂κ, ∂²K/∂κ²)` of the raw (pre-constraint) kernel matrix — the
/// ψ-channel hook. Exact: rides `distance_kappa_jet` (Tower4, FD-gated in
/// `geometry::constant_curvature`) through the chain rule for
/// `K = exp(−d/ℓ)` at κ-FIXED ℓ and centers (see the module κ-contract):
///
/// ```text
///   ∂K/∂κ  = −(d′/ℓ) · K
///   ∂²K/∂κ² = ((d′/ℓ)² − d″/ℓ) · K
/// ```
///
/// The realized design/penalty drifts follow by the κ-fixed transforms:
/// `∂X/∂κ = (∂K/∂κ)·z` and `∂S_raw/∂κ = zᵀ(∂K/∂κ)z` (centers×centers), with
/// the Frobenius penalty normalization differentiated by the existing
/// `normalize_penaltywith_psi_derivatives` seam.
pub fn constant_curvature_kernel_kappa_jets(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kappa: f64,
    length_scale: f64,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "constant-curvature kernel-jet dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    if !(length_scale.is_finite() && length_scale > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature kernel jets need a positive finite length_scale; got {length_scale}"
        );
    }
    validate_chart_points(data, kappa, "data")?;
    validate_chart_points(centers, kappa, "centers")?;
    let manifold = ConstantCurvature::new(data.ncols(), kappa);
    let n = data.nrows();
    let m = centers.nrows();
    let mut value = Array2::<f64>::zeros((n, m));
    let mut dk = Array2::<f64>::zeros((n, m));
    let mut dkk = Array2::<f64>::zeros((n, m));
    let rows: Vec<(usize, Vec<(f64, f64, f64)>)> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(usize, Vec<(f64, f64, f64)>), BasisError> {
            let mut row = Vec::with_capacity(m);
            for (j, c) in centers.outer_iter().enumerate() {
                let (d, d1, d2) = distance_kappa_jet(&manifold, data.row(i), c).map_err(|e| {
                    BasisError::InvalidInput(format!(
                        "constant-curvature distance κ-jet failed at (row {i}, center {j}): {e}"
                    ))
                })?;
                let k = (-d / length_scale).exp();
                let g = d1 / length_scale;
                row.push((k, -g * k, (g * g - d2 / length_scale) * k));
            }
            Ok((i, row))
        })
        .collect::<Result<Vec<_>, BasisError>>()?;
    for (i, row) in rows {
        for (j, (k, k1, k2)) in row.into_iter().enumerate() {
            value[(i, j)] = k;
            dk[(i, j)] = k1;
            dkk[(i, j)] = k2;
        }
    }
    Ok((value, dk, dkk))
}

/// Resolve the realized kernel range ℓ. An explicit positive `spec_length_scale`
/// is used verbatim; the `0.0` sentinel auto-initializes from the median
/// pairwise CHART distance among the centers, doubled to match the κ = 0
/// chart gauge (`d_0 = 2‖Δ‖`).
///
/// κ-contract: the auto rule reads chart coordinates only — it never consults
/// κ — so the realized ℓ is a κ-CONSTANT and contributes no `∂ℓ/∂κ` term to
/// the design drift.
pub fn realized_constant_curvature_length_scale(
    centers: ArrayView2<'_, f64>,
    spec_length_scale: f64,
) -> Result<f64, BasisError> {
    if spec_length_scale.is_finite() && spec_length_scale > 0.0 {
        return Ok(spec_length_scale);
    }
    if spec_length_scale != 0.0 {
        crate::bail_invalid_basis!(
            "constant-curvature length_scale must be positive (or 0.0 for auto); got {spec_length_scale}"
        );
    }
    let m = centers.nrows();
    if m < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: m });
    }
    let mut dists: Vec<f64> = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            let mut s = 0.0_f64;
            for k in 0..centers.ncols() {
                let dlt = centers[(i, k)] - centers[(j, k)];
                s += dlt * dlt;
            }
            dists.push(2.0 * s.sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).expect("finite chart distances"));
    let median = dists[dists.len() / 2];
    if !(median.is_finite() && median > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature auto length_scale failed: centers are degenerate \
             (median pairwise chart distance = {median})"
        );
    }
    Ok(median)
}

/// Build the constant-curvature reproducing-kernel smooth: realized design
/// `K_κ(data, centers)·z`, RKHS penalty `zᵀ K_κ(centers, centers) z`, and the
/// replayable [`BasisMetadata::ConstantCurvature`]. Structure mirrors the
/// Wahba S² builder (`build_spherical_spline_basis`); geometry comes from
/// `ConstantCurvature` at the spec's fixed κ.
pub fn build_constant_curvature_basis(
    data: ArrayView2<'_, f64>,
    spec: &ConstantCurvatureBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    if data.ncols() == 0 {
        crate::bail_invalid_basis!("constant-curvature smooth needs at least one feature column");
    }
    if !spec.kappa.is_finite() {
        crate::bail_invalid_basis!("constant-curvature smooth needs a finite kappa");
    }
    validate_chart_points(data, spec.kappa, "data")?;
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    validate_chart_points(centers.view(), spec.kappa, "centers")?;
    let length_scale = realized_constant_curvature_length_scale(centers.view(), spec.length_scale)?;
    let raw_penalty =
        constant_curvature_kernel_matrix(centers.view(), centers.view(), spec.kappa, length_scale)?;
    // Realized-design constraint transform: uniform coefficient sum-to-zero at
    // fit time; the frozen composed `z · z_parametric` at predict time (#532
    // pattern — see ConstantCurvatureIdentifiability).
    let z = match &spec.identifiability {
        ConstantCurvatureIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != centers.nrows() {
                crate::bail_dim_basis!(
                    "frozen constant-curvature identifiability transform mismatch: {} centers but transform has {} rows",
                    centers.nrows(),
                    transform.nrows()
                );
            }
            transform.clone()
        }
        ConstantCurvatureIdentifiability::CenterSumToZero => {
            let weights = Array1::<f64>::ones(centers.nrows());
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };
    let penalty = z.t().dot(&raw_penalty).dot(&z);
    let raw_design =
        constant_curvature_kernel_matrix(data, centers.view(), spec.kappa, length_scale)?;
    let design = crate::matrix::DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        raw_design.dot(&z),
    ));
    let (penalty_norm, c_primary) = normalize_penalty(&((&penalty + &penalty.t()) * 0.5));
    let mut candidates = vec![PenaltyCandidate {
        matrix: penalty_norm,
        nullspace_dim_hint: 0,
        source: PenaltySource::Primary,
        normalization_scale: c_primary,
        kronecker_factors: None,
        op: None,
    }];
    if spec.double_penalty {
        let ridge = Array2::<f64>::eye(design.ncols());
        let (ridge_norm, c_ridge) = normalize_penalty(&ridge);
        candidates.push(PenaltyCandidate {
            matrix: ridge_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: c_ridge,
            kronecker_factors: None,
            op: None,
        });
    }
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::ConstantCurvature {
            centers,
            kappa: spec.kappa,
            length_scale,
            constraint_transform: Some(z),
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

/// Symmetrize `M` in place to `(M + Mᵀ)/2` (the realized penalty is built from
/// the symmetric kernel Gram; the κ-derivative blocks inherit the same exact
/// symmetrization the value path applies before normalization).
fn symmetrize(m: &Array2<f64>) -> Array2<f64> {
    (m + &m.t()) * 0.5
}

/// Map a single primary-penalty κ-derivative onto the active penalty list by
/// source — the constant-curvature analogue of the Matérn double-penalty
/// derivative selector. The RKHS Gram is the only κ-moving penalty; the
/// double-penalty ridge `I` is κ-independent, so its derivative is exactly
/// zero. Any other source would mean the basis grew a penalty whose κ-movement
/// is unaccounted for, so we refuse loudly rather than silently drop a term.
fn active_constant_curvature_penalty_derivatives(
    penaltyinfo: &[PenaltyInfo],
    primary_derivative: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, BasisError> {
    penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| match &info.source {
            PenaltySource::Primary => Ok(primary_derivative.clone()),
            PenaltySource::DoublePenaltyNullspace => {
                Ok(Array2::<f64>::zeros(primary_derivative.raw_dim()))
            }
            other => Err(BasisError::InvalidInput(format!(
                "unexpected constant-curvature penalty source in κ-derivative path: {other:?}"
            ))),
        })
        .collect()
}

/// κ-derivative bundle for the constant-curvature smooth — the ψ-channel hook
/// that lets κ join the outer LAML/REML optimization as one signed,
/// design-moving coordinate (#944 stage 3 final wiring).
///
/// The outer optimizer's ψ-coordinate here is the **raw, signed curvature κ
/// itself** (NOT `log κ` as for the Matérn kernel scale): κ = 0 must be a
/// reachable interior point of the `S^d ← ℝ^d → H^d` family, which `log κ`
/// cannot represent. So this returns `∂·/∂κ` and `∂²·/∂κ²` directly, and the
/// outer assembly treats the coordinate as `ψ = κ` with `∂/∂ψ = ∂/∂κ`.
///
/// Every κ-fixed piece (centers, length scale ℓ, the center-space constraint
/// transform `z`) is held constant exactly as documented in the module
/// κ-contract, so the design moves with κ only through the geodesic-exponential
/// kernel and:
///
/// ```text
///   X = K(data, centers)·z          ⇒  ∂X/∂κ  = (∂K_dc/∂κ)·z,
///                                       ∂²X/∂κ² = (∂²K_dc/∂κ²)·z
///   S_raw = symm(zᵀ K(centers,centers) z)
///                                   ⇒  ∂S_raw/∂κ  = symm(zᵀ(∂K_cc/∂κ)z), etc.
/// ```
///
/// and the Frobenius penalty normalization is differentiated with the exact
/// quotient rules through the shared `normalize_penaltywith_psi_derivatives`
/// seam — identical to how the Matérn operator penalties propagate their
/// normalization. The double-penalty ridge `I` is κ-independent (zero
/// derivative).
///
/// Mirrors [`build_constant_curvature_basis`] so the realized design and
/// penalties whose κ-derivatives this returns are byte-for-byte the same
/// construction the value path produced (same centers, same ℓ, same `z`).
pub fn build_constant_curvature_basis_kappa_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &ConstantCurvatureBasisSpec,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    if data.ncols() == 0 {
        crate::bail_invalid_basis!("constant-curvature smooth needs at least one feature column");
    }
    if !spec.kappa.is_finite() {
        crate::bail_invalid_basis!("constant-curvature smooth needs a finite kappa");
    }
    validate_chart_points(data, spec.kappa, "data")?;
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    validate_chart_points(centers.view(), spec.kappa, "centers")?;
    let length_scale = realized_constant_curvature_length_scale(centers.view(), spec.length_scale)?;

    // κ-fixed constraint transform `z`, resolved exactly as the value builder.
    let z = match &spec.identifiability {
        ConstantCurvatureIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != centers.nrows() {
                crate::bail_dim_basis!(
                    "frozen constant-curvature identifiability transform mismatch: {} centers but transform has {} rows",
                    centers.nrows(),
                    transform.nrows()
                );
            }
            transform.clone()
        }
        ConstantCurvatureIdentifiability::CenterSumToZero => {
            let weights = Array1::<f64>::ones(centers.nrows());
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };

    // Design κ-jets: X = K(data, centers)·z, so the κ-derivatives are the
    // kernel κ-jets right-multiplied by the κ-fixed `z`.
    let (_k_dc, dk_dc, dkk_dc) =
        constant_curvature_kernel_kappa_jets(data, centers.view(), spec.kappa, length_scale)?;
    let design_first = dk_dc.dot(&z);
    let design_second_diag = dkk_dc.dot(&z);

    // Penalty κ-jets: S_raw = symm(zᵀ K(centers,centers) z). Rebuild the value
    // penalty (and its normalization constant) from the SAME path the value
    // builder used so the quotient-rule normalization derivatives are exact.
    let (k_cc, dk_cc, dkk_cc) = constant_curvature_kernel_kappa_jets(
        centers.view(),
        centers.view(),
        spec.kappa,
        length_scale,
    )?;
    let s_raw = symmetrize(&z.t().dot(&k_cc).dot(&z));
    let s_raw_first = symmetrize(&z.t().dot(&dk_cc).dot(&z));
    let s_raw_second = symmetrize(&z.t().dot(&dkk_cc).dot(&z));
    let (_s_norm, s_norm_first, s_norm_second, _c) =
        normalize_penaltywith_psi_derivatives(&s_raw, &s_raw_first, &s_raw_second);

    // Align the single primary-penalty derivative with the realized active
    // penalty list (primary always; ridge only when double_penalty, and
    // κ-independent). Rebuild the realized basis once to read `penaltyinfo`.
    let base = build_constant_curvature_basis(data, spec)?;
    let penalties_derivative =
        active_constant_curvature_penalty_derivatives(&base.penaltyinfo, &s_norm_first)?;
    let penaltiessecond_derivative =
        active_constant_curvature_penalty_derivatives(&base.penaltyinfo, &s_norm_second)?;

    Ok(BasisPsiDerivativeBundle {
        first: BasisPsiDerivativeResult {
            design_derivative: design_first,
            penalties_derivative,
            implicit_operator: None,
        },
        second: BasisPsiSecondDerivativeResult {
            designsecond_derivative: design_second_diag,
            penaltiessecond_derivative,
            implicit_operator: None,
        },
        implicit_operator: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::faer_ndarray::FaerEigh;

    // Diagnostic (#1059 follow-up): show that a κ-FROZEN chart-scale length
    // makes the geodesic-exponential kernel COLLAPSE toward the constant
    // function as κ grows positive (sphere distances compress), which is the
    // degenerate optimum the REML criterion rails to. For a fixed center set we
    // print, per κ, the median geodesic distance and the kernel "spread"
    // 1 − mean(offdiag K). A collapsing kernel ⇒ spread → 0 as κ ↑.
    #[test]
    fn kernel_spread_collapses_with_kappa_at_frozen_length_scale() {
        // 8 centers in a disk of radius 0.45 (inside every κ∈[-2,2] chart).
        let centers = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        // Frozen ℓ: the κ=0 chart-scale auto rule (median 2‖Δ‖).
        let ell_frozen = realized_constant_curvature_length_scale(centers.view(), 0.0).unwrap();

        let spread = |kappa: f64, ell: f64| -> f64 {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .unwrap();
            let m = k.nrows();
            let mut s = 0.0;
            let mut cnt = 0.0;
            for i in 0..m {
                for j in 0..m {
                    if i != j {
                        s += k[(i, j)];
                        cnt += 1.0;
                    }
                }
            }
            1.0 - s / cnt
        };

        let s_neg = spread(-2.0, ell_frozen);
        let s_zero = spread(0.0, ell_frozen);
        let s_pos = spread(2.0, ell_frozen);
        eprintln!(
            "[κ-collapse] frozen ℓ={ell_frozen:.4}: spread κ=-2 {s_neg:.4} | κ=0 {s_zero:.4} | κ=+2 {s_pos:.4}"
        );

        // The degenerate signature: positive κ collapses the kernel toward the
        // constant (spread shrinks), so the criterion can buy cheap EDF by
        // pushing κ up — this is the unidentifiability we are fixing.
        assert!(
            s_pos < s_zero && s_zero < s_neg,
            "expected kernel spread to shrink with κ at frozen ℓ: κ=-2 {s_neg} κ=0 {s_zero} κ=+2 {s_pos}"
        );

        // Decompose the κ-monotone REML Occam term. The realized penalty is the
        // Frobenius-normalized centered Gram S~ = S_raw/‖S_raw‖_F with
        // S_raw = symm(zᵀ K z); the REML evidence carries +½ log|S~|_+ over its
        // range. Print log det₊(S~) per κ to see whether the penalty-normalization
        // Occam term (not just the modest kernel-spread shift) is what rails κ.
        let weights = Array1::<f64>::ones(centers.nrows());
        let z = weighted_coefficient_sum_to_zero_transform(weights.view()).unwrap();
        let logdet_norm_penalty = |kappa: f64, ell: f64| -> f64 {
            let k =
                constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell).unwrap();
            let s_raw = symmetrize(&z.t().dot(&k).dot(&z));
            let (s_norm, _c) = normalize_penalty(&s_raw);
            let sym = symmetrize(&s_norm);
            let (evals, _v) = FaerEigh::eigh(&sym, faer::Side::Lower).unwrap();
            let max = evals.iter().cloned().fold(0.0_f64, f64::max);
            let tol = max * 1e-9;
            evals
                .iter()
                .filter(|&&e| e > tol)
                .map(|&e| e.ln())
                .sum::<f64>()
        };
        let l_neg = logdet_norm_penalty(-2.0, ell_frozen);
        let l_zero = logdet_norm_penalty(0.0, ell_frozen);
        let l_pos = logdet_norm_penalty(2.0, ell_frozen);
        eprintln!(
            "[κ-collapse] log|S~|_+ (frozen ℓ): κ=-2 {l_neg:.4} | κ=0 {l_zero:.4} | κ=+2 {l_pos:.4}"
        );

        // GEODESIC-SCALED ℓ removes the κ-dependence of the kernel resolution:
        // set ℓ(κ) = median geodesic distance d_κ among centers. Then the spread
        // should be ~κ-invariant. Print the geodesic-ℓ spread per κ.
        let geo_median_ell = |kappa: f64| -> f64 {
            let m = centers.nrows();
            let manifold = ConstantCurvature::new(centers.ncols(), kappa);
            let mut dists = Vec::with_capacity(m * (m - 1) / 2);
            for i in 0..m {
                for j in (i + 1)..m {
                    dists.push(manifold.distance(centers.row(i), centers.row(j)).unwrap());
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dists[dists.len() / 2]
        };
        let gs_neg = spread(-2.0, geo_median_ell(-2.0));
        let gs_zero = spread(0.0, geo_median_ell(0.0));
        let gs_pos = spread(2.0, geo_median_ell(2.0));
        let gl_neg = logdet_norm_penalty(-2.0, geo_median_ell(-2.0));
        let gl_zero = logdet_norm_penalty(0.0, geo_median_ell(0.0));
        let gl_pos = logdet_norm_penalty(2.0, geo_median_ell(2.0));
        eprintln!(
            "[κ-collapse] geodesic ℓ: spread κ=-2 {gs_neg:.4} | κ=0 {gs_zero:.4} | κ=+2 {gs_pos:.4}"
        );
        eprintln!(
            "[κ-collapse] geodesic ℓ: log|S~|_+ κ=-2 {gl_neg:.4} | κ=0 {gl_zero:.4} | κ=+2 {gl_pos:.4}"
        );
    }
}

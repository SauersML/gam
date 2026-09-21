//! Intrinsic S² (sphere) smooth specification types.
//!
//! These configuration types own the user-facing description of an intrinsic
//! sphere smooth (construction method, reproducing kernel, penalty order, and
//! realized-design identifiability policy). The sibling `sphere_kernels` and
//! `sphere_spectral` modules consume them to assemble the basis and penalty.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::CenterStrategy;

/// Which intrinsic S² smooth construction to use.
///
/// - `Wahba`: reproducing-kernel basis on a center set selected by
///   `center_strategy`. The kernel function itself is chosen via
///   `wahba_kernel` (the `H^m(S²)` Sobolev kernel Σ [l(l+1)]^{-m} P_l, in
///   closed form or truncated at a finite Legendre degree).
/// - `Harmonic`: real spherical-harmonic truncation up to `max_degree` with
///   the Laplace-Beltrami eigenvalue penalty `[l(l+1)]^m`. Basis
///   dimension is `max_degree * (max_degree + 2)`; centers are ignored.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SphereMethod {
    #[default]
    Wahba,
    Harmonic,
}

/// Which reproducing kernel to use for `SphereMethod::Wahba`.
///
/// Both variants are the reproducing kernel of `H^m(S²)` under the
/// Laplace–Beltrami inner product, `penalty_order m ∈ {1, 2, 3, 4}`:
///
/// `K_m(γ) = (1/4π) Σ_{l ≥ 1} (2l+1) · [l(l+1)]^{-m} · P_l(cos γ)`,
///
/// whose penalty quadratic form recovers `‖f‖²_{H^m} = Σ_l [l(l+1)]^m · |f̂_l|²`,
/// the same Laplace–Beltrami eigenvalue penalty `SphereMethod::Harmonic`
/// applies. They differ only in how the series is evaluated.
///
/// - **Sobolev** (default): for m=1, 2, 3 via the closed forms from
///   Beatson & zu Castell (2018) "Thinplate Splines on the Sphere"
///   (SIGMA 14 (2018), 083) using elementary functions plus the
///   di/trilogarithm. For m=4 the spectral Legendre series.
/// - **SobolevTruncated**: the series cut at a finite degree `lmax`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SphereWahbaKernel {
    /// True Sobolev `H^m(S²)` reproducing kernel via closed-form
    /// (elementary + Li_n) for m=1,2,3 and a deep (L=4096..96) spectral
    /// series for m=4. This is the user-facing default.
    #[default]
    Sobolev,
    /// Finite truncated-spectral Sobolev kernel evaluated by the same
    /// Legendre 3-term recurrence the GPU `s2_wahba_legendre_colmajor`
    /// kernel runs in registers:
    ///
    /// `K_L^{Sobolev}(γ) = Σ_{ℓ=1..L} c_ℓ · P_ℓ(cos γ)`,
    /// `c_0 = 0`, `c_ℓ = (2ℓ+1) / (4π · [ℓ(ℓ+1)]^m)`.
    ///
    /// This is the explicit parity target for the GPU kernel: at any
    /// finite `lmax` the closed-form (or m=4-spectral-at-fixed-deep-L)
    /// variants differ by the documented truncation error, while the GPU
    /// matches this variant exactly to roundoff. Single-source: both CPU
    /// and GPU use the same Legendre recurrence and the same c_ℓ array.
    SobolevTruncated {
        /// Largest Legendre degree retained (≥ 1). Practical range
        /// 5..200; GPU kernel uses LMAX as a compile-time `#define`.
        lmax: u16,
    },
}

/// Intrinsic S² (sphere) smooth configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphericalSplineBasisSpec {
    /// Center/knot selection strategy in latitude/longitude coordinates
    /// (used only when `method == Wahba`).
    pub center_strategy: CenterStrategy,
    /// Sphere roughness penalty order m ∈ {1, 2, 3, 4}.
    ///
    /// - **Wahba method**: the reproducing-kernel norm is
    ///   `Σ_l [l(l+1)]^m · |f̂_l|²` — the canonical `H^m(S²)` Sobolev
    ///   norm. `m=2` is the usual curvature (thin-plate analogue) penalty.
    /// - **Harmonic method**: raises the Laplace-Beltrami eigenvalue
    ///   penalty to `[l(l+1)]^m` (same as the Sobolev kernel norm above).
    pub penalty_order: usize,
    /// Add a ridge-like shrinkage penalty.
    pub double_penalty: bool,
    /// Interpret latitude/longitude in radians instead of degrees. Default is
    /// degrees (Earth/data-frame convention); set true for radians.
    #[serde(default)]
    pub radians: bool,
    /// Construction method. Default Wahba (reproducing kernel); Harmonic uses
    /// real spherical harmonics with a Laplace-Beltrami eigenvalue penalty
    /// (Wood §5.6.2).
    #[serde(default)]
    pub method: SphereMethod,
    /// Maximum spherical-harmonic degree L when `method == Harmonic`. Basis
    /// dimension is `L * (L + 2)`. Ignored for Wahba.
    #[serde(default)]
    pub max_degree: Option<usize>,
    /// When `method == Wahba`, how the `H^m(S²)` reproducing kernel is
    /// evaluated. See `SphereWahbaKernel` docs.
    ///
    /// Deserialised specs that omit this field fall back to the standard
    /// `Default` (Sobolev). Saved models that pre-date this field are not
    /// supported — they must be re-fit with the current encoding.
    #[serde(default)]
    pub wahba_kernel: SphereWahbaKernel,
    /// Realized-design identifiability policy for the Wahba sphere (#532).
    ///
    /// The finite-center Wahba kernel design `K(data, centers) · z` can span a
    /// near-constant direction on the data rows even though the continuous
    /// kernel omits the l=0 mode.
    /// In any model with a parametric intercept this collides with the global
    /// intercept — the #531 constant-vs-intercept rank-1 collision class. The
    /// fix (mirroring `MaternIdentifiability::FrozenTransform`) lets the global
    /// identifiability pipeline compose a parametric-orthogonalization onto `z`
    /// and freeze the composed transform here, so the predict-time rebuild
    /// reuses the exact fit-time realized transform instead of silently dropping
    /// the orthogonalization.
    #[serde(default)]
    pub identifiability: SphericalSplineIdentifiability,
    /// `true` when nobody chose the harmonic `max_degree`: it is the formula
    /// default's starting resolution, which the standard formula workflow
    /// refines from the converged fit's own evidence. An explicit degree or
    /// `k=` is `false` and honoured verbatim.
    #[serde(default)]
    pub adaptive_degree: bool,
}

/// Realized-design identifiability policy for the Wahba spherical spline (#532).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SphericalSplineIdentifiability {
    /// Fit-time default: keep the raw center coefficient chart, then let the
    /// global identifiability pipeline residualize the realized design against
    /// the parametric block.
    #[default]
    CenterSumToZero,
    /// Predict-time replay: use this frozen realized-design transform directly
    /// (the composed raw-chart/parametric transform captured at fit time).
    /// `transform.nrows()` equals the raw Wahba basis width after its
    /// finite-center kernel / low-degree harmonic decomposition;
    /// `transform.ncols()` is the constrained smooth dimension.
    FrozenTransform { transform: Array2<f64> },
}

impl Default for SphericalSplineBasisSpec {
    fn default() -> Self {
        Self {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 50 },
            penalty_order: 2,
            double_penalty: true,
            radians: false,
            method: SphereMethod::Wahba,
            max_degree: None,
            wahba_kernel: SphereWahbaKernel::Sobolev,
            identifiability: SphericalSplineIdentifiability::CenterSumToZero,
            adaptive_degree: false,
        }
    }
}

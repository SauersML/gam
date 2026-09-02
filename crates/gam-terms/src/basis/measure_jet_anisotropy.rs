//! Learned ambient anisotropy for the measure-jet energy.
//!
//! The isotropic measure-jet energy [`super::measure_jet_energy_form`] treats
//! the ambient coordinates with a Euclidean local Gram: the Gaussian kernel
//! weight is `exp(−‖δ‖²/2ε²)` and the local affine features are `δ/ε` with
//! `δ = x_j − x_i`. This module generalizes that Euclidean inner product to a
//! learned Mahalanobis metric
//!
//! ```text
//!   A = L Lᵀ,        Ā = A / det(A)^(1/d)      (det-normalized, det Ā = 1),
//! ```
//!
//! parametrized by the lower-triangular Cholesky factor `L` (d×d). The metric
//! enters every local block through the SINGLE substitution
//!
//! ```text
//!   ⟨u, v⟩  ↦  uᵀ Ā v ,
//! ```
//!
//! which is realized exactly by transforming the centers once with the
//! det-normalized factor `M = L / det(L)^(1/d)` (so `M Mᵀ = Ā`, `det M = 1`):
//!
//! ```text
//!   ‖δ M‖²       = δ Ā δᵀ           (metric squared distance → kernel),
//!   (δ/ε)M       = metric local affine features,
//!   Y = X M      (transformed row centers; E_A(X) ≡ E_I(Y)).
//! ```
//!
//! Because the local affine residual projects each block's center values onto
//! `span{1, local affine coords}` and `M` is invertible, the projection is
//! reparametrization-invariant: the metric reaches the energy ONLY through the
//! kernel weights `w` and the (linearly transformed) features. With `Ā = I`
//! (`M = I`, `Y = X`) the construction collapses to the isotropic energy
//! bit-for-bit — that is the contract the first oracle test pins.
//!
//! To learn `L` by REML the energy needs exact first and second derivatives
//! `∂E/∂L_ij`, `∂²E/∂L_ij∂L_kl`. They are produced from the SAME local block
//! walk as the value (no second assembly that could drift from the first),
//! by carrying, per requested `L`-direction, the exact first/second
//! directional derivatives of every metric-dependent block quantity — the
//! transformed features, the Gaussian weights, the weighted mean, `B`, `G`,
//! `G⁺` and the residual — through the closed-form product/chain rules.
//!
//! All ∂/∂L jets are FD-gated in this module's tests against central
//! differences of the energy (rel tol `5e-5`, step `h = 1e-4`, the
//! second-difference-optimal step mirroring `measure_jet_smooth`'s own jet
//! gates).

use ndarray::Array2;

/// A single requested derivative direction in `L`-space: the lower-triangular
/// entry `(i, j)` with `i >= j`. The zeroth-order "direction" (the value
/// itself) is handled separately; this names the active first-order channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LIndex {
    /// Row of the lower-triangular factor entry (`>= col`).
    pub row: usize,
    /// Column of the lower-triangular factor entry (`<= row`).
    pub col: usize,
}

/// The anisotropic energy together with its exact first and second jets with
/// respect to the lower-triangular Cholesky factor entries of `L`.
///
/// `indices[a]` names the `(row, col)` of the `a`-th active lower-triangular
/// entry (column-major over the lower triangle: for each column `j`, rows
/// `j..d`). `d_first[a] = ∂Q/∂L_{indices[a]}`, and `d_second[(a, b)]` (stored
/// for the full pair grid, symmetric in `a, b`) is
/// `∂²Q/∂L_{indices[a]}∂L_{indices[b]}`.
pub struct MeasureJetAnisotropyJets {
    /// The det-normalized anisotropic energy form (m×m, symmetric PSD).
    pub q: Array2<f64>,
    /// Active lower-triangular `L`-entry indices, in the derivative order.
    pub indices: Vec<LIndex>,
    /// First derivatives `∂Q/∂L_a`, one m×m form per active index.
    pub d_first: Vec<Array2<f64>>,
    /// Second derivatives `∂²Q/∂L_a∂L_b`, indexed by `a*n + b` over the
    /// `n = indices.len()` active entries (full symmetric grid).
    pub d_second: Vec<Array2<f64>>,
}

impl MeasureJetAnisotropyJets {
    /// Number of active lower-triangular derivative channels.
    #[inline]
    pub fn n_active(&self) -> usize {
        self.indices.len()
    }

    /// Borrow the second-derivative form `∂²Q/∂L_a∂L_b`.
    #[inline]
    pub fn second(&self, a: usize, b: usize) -> &Array2<f64> {
        &self.d_second[a * self.indices.len() + b]
    }
}

// ----------------------------------------------------------------------------
// Det-normalized factor M = L / det(L)^(1/d) and its exact L-jets.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Per-block algebra and its exact L-jets.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Top-level energy and L-jets.
// ----------------------------------------------------------------------------


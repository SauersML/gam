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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{measure_jet_band, measure_jet_energy_form};
    use ndarray::array;

    pub(crate) fn band_for(centers: &Array2<f64>) -> MeasureJetBand {
        measure_jet_band(centers.view(), 0).expect("band")
    }

    pub(crate) fn two_cluster_centers() -> (ndarray::Array2<f64>, ndarray::Array1<f64>) {
        (
            ndarray::array![
                [-0.8, -0.6],
                [-0.7, -0.5],
                [-0.6, -0.7],
                [0.8, 0.6],
                [0.7, 0.5],
                [0.6, 0.7]
            ],
            ndarray::array![0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )
    }

    /// Oracle (1): with `L = I` (so `Ā = I`, `M = I`) the anisotropic energy
    /// reproduces the isotropic `measure_jet_energy_form` bit-for-bit. The
    /// metric reaches the energy ONLY through the kernel and the (identity)
    /// feature transform, both of which are arithmetically the isotropic path
    /// when `M = I`.
    #[test]
    pub(crate) fn identity_metric_reproduces_isotropic_bit_for_bit() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let (s0, a0) = (1.3, 0.8);
        let l = Array2::<f64>::eye(2);
        let q_aniso = measure_jet_anisotropy_energy_form(
            centers.view(),
            masses.view(),
            &band,
            s0,
            a0,
            l.view(),
        )
        .expect("aniso energy");
        let q_iso = measure_jet_energy_form(centers.view(), masses.view(), &band, s0, a0, 1e-3)
            .expect("iso energy");
        assert_eq!(q_aniso.dim(), q_iso.dim());
        for (a, b) in q_aniso.iter().zip(q_iso.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "Ā = I must reproduce the isotropic energy bit-for-bit: {a} vs {b}"
            );
        }
    }

    /// Oracle (2): every `∂Q/∂L_ij` and `∂²Q/∂L_ij∂L_kl` matches central
    /// finite differences of the energy. Step `h = 1e-4` (the
    /// second-difference-optimal step mirroring `measure_jet_smooth`'s jet
    /// gate), rel tol `5e-5`. A non-identity, non-symmetric lower-triangular
    /// `L` exercises every active channel and the off-diagonal coupling.
    #[test]
    pub(crate) fn l_jets_match_finite_differences() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let (s0, a0) = (1.3, 0.8);
        // A genuinely anisotropic, full lower-triangular factor.
        let l0 = array![[1.30, 0.00], [-0.45, 0.80]];
        let jets = measure_jet_anisotropy_energy_form_with_jets(
            centers.view(),
            masses.view(),
            &band,
            s0,
            a0,
            l0.view(),
        )
        .expect("jets");

        // Base value must equal a plain re-evaluation bit-for-bit.
        let q_plain = measure_jet_anisotropy_energy_form(
            centers.view(),
            masses.view(),
            &band,
            s0,
            a0,
            l0.view(),
        )
        .expect("plain");
        for (a, b) in jets.q.iter().zip(q_plain.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "value drift {a} vs {b}");
        }

        let eval = |l: &Array2<f64>| {
            measure_jet_anisotropy_energy_form(
                centers.view(),
                masses.view(),
                &band,
                s0,
                a0,
                l.view(),
            )
            .expect("energy")
        };
        let perturb = |idx: LIndex, delta: f64| {
            let mut l = l0.clone();
            l[(idx.row, idx.col)] += delta;
            l
        };

        let h = 1e-4;
        let n = jets.n_active();

        // First derivatives via the central two-point stencil.
        for a in 0..n {
            let ia = jets.indices[a];
            let plus = eval(&perturb(ia, h));
            let minus = eval(&perturb(ia, -h));
            let fd = (&plus - &minus) / (2.0 * h);
            let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
            for (an, fdv) in jets.d_first[a].iter().zip(fd.iter()) {
                assert!(
                    (an - fdv).abs() <= 5e-5 * scale,
                    "∂Q/∂L[{},{}] mismatch: analytic {an:.6e} vs FD {fdv:.6e} (scale {scale:.3e})",
                    ia.row,
                    ia.col
                );
            }
        }

        // Diagonal second derivatives via the three-point stencil.
        for a in 0..n {
            let ia = jets.indices[a];
            let plus = eval(&perturb(ia, h));
            let center = eval(&l0);
            let minus = eval(&perturb(ia, -h));
            let fd = (&(&plus + &minus) - &(&center * 2.0)) / (h * h);
            let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
            for (an, fdv) in jets.second(a, a).iter().zip(fd.iter()) {
                assert!(
                    (an - fdv).abs() <= 5e-5 * scale,
                    "∂²Q/∂L[{},{}]² mismatch: analytic {an:.6e} vs FD {fdv:.6e} (scale {scale:.3e})",
                    ia.row,
                    ia.col
                );
            }
        }

        // Cross second derivatives via the four-point stencil.
        for a in 0..n {
            let ia = jets.indices[a];
            for b in (a + 1)..n {
                let ib = jets.indices[b];
                let mut lpp = l0.clone();
                lpp[(ia.row, ia.col)] += h;
                lpp[(ib.row, ib.col)] += h;
                let mut lpm = l0.clone();
                lpm[(ia.row, ia.col)] += h;
                lpm[(ib.row, ib.col)] -= h;
                let mut lmp = l0.clone();
                lmp[(ia.row, ia.col)] -= h;
                lmp[(ib.row, ib.col)] += h;
                let mut lmm = l0.clone();
                lmm[(ia.row, ia.col)] -= h;
                lmm[(ib.row, ib.col)] -= h;
                let pp = eval(&lpp);
                let pm = eval(&lpm);
                let mp = eval(&lmp);
                let mm = eval(&lmm);
                let fd = (&(&pp - &pm) - &(&mp - &mm)) / (4.0 * h * h);
                let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
                for (an, fdv) in jets.second(a, b).iter().zip(fd.iter()) {
                    assert!(
                        (an - fdv).abs() <= 5e-5 * scale,
                        "∂²Q/∂L[{},{}]∂L[{},{}] mismatch: analytic {an:.6e} vs FD {fdv:.6e} (scale {scale:.3e})",
                        ia.row,
                        ia.col,
                        ib.row,
                        ib.col
                    );
                }
                // Symmetry of the second-derivative grid.
                for (sab, sba) in jets.second(a, b).iter().zip(jets.second(b, a).iter()) {
                    assert!((sab - sba).abs() <= 1e-12 * (1.0 + sab.abs()));
                }
            }
        }
    }

    /// Oracle (3): det-normalization invariance — scaling `L` by any `c > 0`
    /// leaves the energy unchanged, because `Ā = (c L)(c L)ᵀ / det(c² L Lᵀ)^(1/d)
    /// = L Lᵀ / det(L Lᵀ)^(1/d)`. The whole point of the normalization is that
    /// only the SHAPE of the metric, not its overall scale, is learned.
    #[test]
    pub(crate) fn det_normalization_is_scale_invariant() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let (s0, a0) = (1.1, 0.9);
        let l0 = array![[0.90, 0.00], [0.35, 1.40]];
        let q_ref = measure_jet_anisotropy_energy_form(
            centers.view(),
            masses.view(),
            &band,
            s0,
            a0,
            l0.view(),
        )
        .expect("ref");
        for &c in &[0.25_f64, 0.5, 2.0, 7.5] {
            let lc = &l0 * c;
            let q_c = measure_jet_anisotropy_energy_form(
                centers.view(),
                masses.view(),
                &band,
                s0,
                a0,
                lc.view(),
            )
            .expect("scaled");
            let scale = q_ref.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            assert!(scale > 0.0, "energy is identically zero");
            for (a, b) in q_c.iter().zip(q_ref.iter()) {
                assert!(
                    (a - b).abs() <= 1e-10 * scale,
                    "scale c = {c} changed the normalized energy: {a:.6e} vs {b:.6e}"
                );
            }
        }
    }

    /// The energy must annihilate constants at every metric (the local affine
    /// projection still kills the constant exactly), mirroring the isotropic
    /// contract.
    #[test]
    pub(crate) fn anisotropic_energy_annihilates_constants() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let l = array![[1.20, 0.00], [-0.30, 0.95]];
        let q = measure_jet_anisotropy_energy_form(
            centers.view(),
            masses.view(),
            &band,
            1.5,
            1.0,
            l.view(),
        )
        .expect("energy");
        let m = q.nrows();
        let ones = Array1::<f64>::ones(m);
        let qv = q.dot(&ones);
        let scale = q.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(scale > 0.0, "energy is identically zero");
        for (i, v) in qv.iter().enumerate() {
            assert!(
                v.abs() <= 1e-10 * scale,
                "Q·1 leak at row {i}: {v:.3e} vs scale {scale:.3e}"
            );
        }
    }
}

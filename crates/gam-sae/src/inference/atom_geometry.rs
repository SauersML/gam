//! Per-atom **geometry lens** (#2091): an *additive*, read-only report on the
//! actual SHAPE of each fitted `SaeManifoldTerm`
//! atom — the thing that distinguishes a *manifold* SAE from a linear one.
//!
//! # Why this exists
//!
//! [`atom_lens`](crate::inference::atom_lens) answers *"is this atom used?"*
//! (presence vs behavioral coupling). It says nothing about whether the atom is a
//! HEALTHY curved feature or a degenerate one. For a manifold SAE that is the
//! decisive question: an atom's decoder curve `g_k(t) = Φ_k(t) B_k` is supposed to
//! trace a genuine circle / torus / sphere, but two silent failure modes waste it:
//!
//! * **collapse** — the curve populates FEWER ambient dimensions than its topology
//!   affords (a "circle" that has flattened onto a single line). This is the
//!   *single-atom* fingerprint of the K≥2 co-collapse the
//!   `structural_coherence_collapse_detected` guard chases pairwise — but measured
//!   per atom in `O(K)` rather than `O(K²)`, so it stays affordable at deployment
//!   width where the all-pairs scan cannot run.
//! * **latent linearity** — a curved-topology atom whose decoded points barely
//!   leave their leading principal axis is, up to noise, a straight direction: its
//!   curved basis is dead weight a plain linear atom would carry for free.
//!
//! # What it reports, per atom
//!
//! Everything is derived from the atom's own fitted state (`basis_values` = `Φ_k`,
//! the analytic `basis_jacobian` = `∂Φ_k/∂t`, and `decoder_coefficients` = `B_k`)
//! evaluated over the rows where the atom is active (assignment mass above
//! [`SAE_TRUST_ACTIVE_MASS_FLOOR`]). No finite differences, no autodiff: the
//! tangents are the model's own analytic jets.
//!
//! * **effective_output_dim** — participation ratio of the decoded-point spectrum,
//!   `(Σλ)² / Σλ²`. The effective number of ambient dimensions the atom's curve
//!   actually populates (`≈ 2` for a healthy circle, `≈ 1` for one collapsed to a
//!   line).
//! * **ideal_curve_dim** / **degeneracy** — the natural ambient dimension the
//!   topology affords, and `1 − eff/ideal` clamped to `[0, 1]`: how far BELOW its
//!   topology the atom has collapsed. `None` for topologies without a clean
//!   intrinsic embedding dimension (`Duchon`, `Precomputed`).
//! * **nonlinearity** — `1 − λ_max/Σλ`, the fraction of decoded-point variance OFF
//!   the leading principal axis (`0` = a straight direction, `≈ ½` = a circle).
//! * **tangent_speed_mean** / **speed_cv** — the mean physical (amplitude-scaled)
//!   arc speed `‖a · g_k'(t)‖` over active rows and its coefficient of variation. A
//!   low CV is the arc-length-uniform parameterisation a canonicalised chart wants;
//!   a high CV means the latent coordinate bunches the curve.
//!
//! # Cost — why it scales
//!
//! The decoded-point cloud is never materialised. Its centred second moment is
//! `a² Bᵀ S_φ B` (`p × p`) whose non-zero spectrum equals that of the `M × M`
//! matrix `a² S_φ^{½} (B Bᵀ) S_φ^{½}`, with `S_φ` the mass-weighted basis-feature
//! covariance and `M` the (small) basis width. Both `B Bᵀ` and `S_φ` are `M × M`;
//! the tangent speeds reuse the same `B Bᵀ` Gram. Per atom the work is
//! `O(n_active·M² + M²·p + M³)` with `O(M²)` memory — linear in `K`, independent of
//! how large the corpus or output dimension grow, so it never approaches the memory
//! wall.
//!
//! # Read-only / no loss contact
//!
//! Like `atom_lens`, nothing here feeds back into any loss, criterion, penalty,
//! or optimizer state. It is a pure read of the fitted term.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, s};

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;

use crate::inference::atom_lens::SAE_TRUST_ACTIVE_MASS_FLOOR;
use crate::manifold::{SaeAtomBasisKind, SaeManifoldTerm};

/// Eigenvalues below this fraction of the largest are treated as numerical dust
/// and dropped from the participation ratio / variance fractions. Relative, so it
/// is scale-free (the spectrum's absolute scale is `a²`-dependent and irrelevant to
/// the reported *ratios*).
const SPECTRUM_REL_FLOOR: f64 = 1.0e-9;

/// Tolerance, in *dimensions*, by which an atom may fall short of its topology's
/// natural ambient dimension before it reads as collapsed: an atom is healthy while
/// `effective_output_dim ≥ ideal_curve_dim − ½`. Half a dimension is the natural
/// "within one populated axis of full rank" band; it is topology-relative, not a
/// tuned scalar (for a circle it flags `eff < 1.5`, for a straight line `eff < 0.5`
/// = effectively dead).
const DEGENERACY_DIM_TOLERANCE: f64 = 0.5;

/// A curved-topology atom whose `nonlinearity` is below this reads as *effectively
/// linear* — its decoded points sit within this fraction of their variance on a
/// single axis, indistinguishable from a straight direction. Mirrors the
/// `VALIDITY_DIVERGENCE_FRACTION` (`0.1`) "within 10% of the linear picture"
/// convention the steering dosimetry already uses.
const EFFECTIVELY_LINEAR_NONLINEARITY_FLOOR: f64 = 0.1;

/// One atom's geometry entry. Every geometric field is `Option`: it degrades to
/// `None` (never an error, never a silent zero) when the atom is active on fewer
/// than two rows, when its decoded cloud is a single point, or when the small
/// symmetric eigensolve does not converge.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomGeometryEntry {
    /// The atom's name (mirrors [`crate::manifold::SaeManifoldAtom::name`]).
    pub name: String,
    /// The atom's topology tag, as its round-trip name (`"periodic"`, `"sphere"`,
    /// `"linear"`, …).
    pub topology: String,
    /// Latent (intrinsic) dimension `d_k`.
    pub latent_dim: usize,
    /// Number of rows on which the atom is active (assignment mass above
    /// [`SAE_TRUST_ACTIVE_MASS_FLOOR`]).
    pub n_active: usize,
    /// Participation ratio of the decoded-point spectrum — the effective number of
    /// ambient dimensions the curve populates over its active rows.
    pub effective_output_dim: Option<f64>,
    /// The natural ambient dimension the atom's topology affords (a circle → 2, a
    /// linear axis → 1, …), capped at what the basis width can express. `None` for
    /// topologies without a clean intrinsic embedding dimension.
    pub ideal_curve_dim: Option<f64>,
    /// `1 − effective_output_dim / ideal_curve_dim`, clamped to `[0, 1]`: how far
    /// BELOW its topology the atom has collapsed. `None` when either input is
    /// unavailable. This is the per-atom, `O(K)` co-collapse fingerprint.
    pub degeneracy: Option<f64>,
    /// `1 − λ_max / Σλ`: the fraction of decoded-point variance off the leading
    /// principal axis. `0` ⇒ a straight direction, `≈ ½` ⇒ a circle.
    pub nonlinearity: Option<f64>,
    /// Mean physical arc speed `‖a · g_k'(t)‖` over the atom's active rows.
    pub tangent_speed_mean: Option<f64>,
    /// Fourier energy fractions of the decoded curve, PERIODIC basis only.
    ///
    /// The basis rows are `[1, sin(2π·1·t), cos(2π·1·t), …, sin(2π·H·t),
    /// cos(2π·H·t)]`, so the decoded curve's harmonic-`h` component is
    /// `sin(2πht)·β_{2h−1} + cos(2πht)·β_{2h}` and its mean-square energy under
    /// the uniform coordinate measure is `½·E_h` with `E_h = ‖β_{2h−1}‖² +
    /// ‖β_{2h}‖²` (rows of the decoder). Entry `h−1` holds `E_h / Σ_g E_g` for
    /// `h = 1..H`; the constant row is excluded (it is the atom's offset, not
    /// shape). `None` for non-periodic bases or when all non-constant rows
    /// carry zero energy.
    pub harmonic_energy_fractions: Option<Vec<f64>>,
    /// `argmax_h E_h` (1-based). A dominant `h = 2` is the 180°-wraparound
    /// signature: the decoded curve is invariant under `t ↦ t + ½`.
    pub dominant_harmonic: Option<usize>,
    /// `E_2 / E_1` — the classic second-harmonic diagnostic. `None` when `H <
    /// 2` or `E_1 = 0`.
    pub second_harmonic_ratio: Option<f64>,
    /// Coefficient of variation of the arc speed across active rows: `0` ⇒ a
    /// perfectly arc-length-uniform parameterisation, larger ⇒ the latent
    /// coordinate bunches the curve.
    pub speed_cv: Option<f64>,
}

impl AtomGeometryEntry {

}

/// The geometry lens over every atom of a fitted SAE-manifold term, in atom order.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomGeometryReport {
    /// One entry per atom, in atom order.
    pub atoms: Vec<AtomGeometryEntry>,
}

impl AtomGeometryReport {

}

#[cfg(test)]
mod tests {
    use super::*;
    // `Array1`/`Array2` (and the view/`s!` items) arrive via `use super::*`; only
    // `Array3` is not already in the parent's imports.
    use ndarray::Array3;

    /// Build a planted circle atom: `Φ = [1, cos θ, sin θ]`, decoder mapping the two
    /// harmonics onto an orthonormal 2-plane of a `p = 4` output, `n` rows evenly
    /// spread around the circle, all fully active.
    fn planted_circle(
        n: usize,
        radius: f64,
    ) -> (Array2<f64>, Array3<f64>, Array2<f64>, Array1<f64>) {
        let m = 3usize;
        let p = 4usize;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jac = Array3::<f64>::zeros((n, m, 1));
        for row in 0..n {
            let theta = std::f64::consts::TAU * (row as f64) / (n as f64);
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = theta.cos();
            phi[[row, 2]] = theta.sin();
            // ∂/∂θ of [1, cos, sin] = [0, -sin, cos].
            jac[[row, 0, 0]] = 0.0;
            jac[[row, 1, 0]] = -theta.sin();
            jac[[row, 2, 0]] = theta.cos();
        }
        // Decoder: harmonic 1 → output channel 0, harmonic 2 → output channel 1,
        // each scaled by `radius`. Constant harmonic → no output (centred curve).
        let mut dec = Array2::<f64>::zeros((m, p));
        dec[[1, 0]] = radius;
        dec[[2, 1]] = radius;
        let masses = Array1::<f64>::ones(n);
        (phi, jac, dec, masses)
    }

    #[test]
    fn healthy_circle_reads_two_dimensional_and_curved() {
        let (phi, jac, dec, masses) = planted_circle(64, 2.0);
        let entry = atom_geometry_entry_from_parts(
            "c".into(),
            &SaeAtomBasisKind::Periodic,
            phi.view(),
            jac.view(),
            dec.view(),
            masses.view(),
        );
        let eff = entry.effective_output_dim.expect("circle has a spectrum");
        assert!(
            (eff - 2.0).abs() < 1.0e-6,
            "circle effective dim ≈ 2, got {eff}"
        );
        assert_eq!(entry.ideal_curve_dim, Some(2.0));
        assert!(
            entry.degeneracy.unwrap() < 1.0e-6,
            "healthy circle is not degenerate"
        );
        assert!(!entry.is_collapsed());
        // Two equal principal axes ⇒ nonlinearity ≈ ½.
        let nl = entry.nonlinearity.unwrap();
        assert!(
            (nl - 0.5).abs() < 1.0e-6,
            "circle nonlinearity ≈ ½, got {nl}"
        );
        assert!(!entry.is_effectively_linear());
        // Pure first-harmonic curve: all Fourier energy on h = 1.
        let fracs = entry
            .harmonic_energy_fractions
            .as_ref()
            .expect("periodic atom carries a harmonic spectrum");
        assert_eq!(fracs.len(), 1);
        assert!((fracs[0] - 1.0).abs() < 1.0e-12);
        assert_eq!(entry.dominant_harmonic, Some(1));
        assert_eq!(entry.second_harmonic_ratio, None);
        // Constant-speed parameterisation ⇒ CV ≈ 0; speed = a·radius·1 = 2.
        assert!(
            entry.speed_cv.unwrap() < 1.0e-6,
            "circle arc speed is uniform"
        );
        assert!((entry.tangent_speed_mean.unwrap() - 2.0).abs() < 1.0e-6);
    }

    /// Mixed two-harmonic curve `γ(θ) = r₁·e₁(θ) ⊕ r₂·e₂(2θ)` on disjoint output
    /// planes: the spectrum splits `E_h ∝ r_h²`, the 180°-wraparound harmonic
    /// dominates when `r₂ > r₁`, and the second-harmonic ratio is `(r₂/r₁)²` —
    /// the certified version of the InceptionV1 curve-detector harmonics.
    #[test]
    fn two_harmonic_curve_spectrum_splits_by_energy() {
        let n = 96usize;
        let m = 5usize;
        let p = 4usize;
        let (r1, r2) = (1.0_f64, 2.0_f64);
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jac = Array3::<f64>::zeros((n, m, 1));
        for row in 0..n {
            let theta = std::f64::consts::TAU * (row as f64) / (n as f64);
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = theta.sin();
            phi[[row, 2]] = theta.cos();
            phi[[row, 3]] = (2.0 * theta).sin();
            phi[[row, 4]] = (2.0 * theta).cos();
            jac[[row, 1, 0]] = theta.cos();
            jac[[row, 2, 0]] = -theta.sin();
            jac[[row, 3, 0]] = 2.0 * (2.0 * theta).cos();
            jac[[row, 4, 0]] = -2.0 * (2.0 * theta).sin();
        }
        let mut dec = Array2::<f64>::zeros((m, p));
        dec[[1, 0]] = r1;
        dec[[2, 1]] = r1;
        dec[[3, 2]] = r2;
        dec[[4, 3]] = r2;
        let masses = Array1::<f64>::ones(n);
        let entry = atom_geometry_entry_from_parts(
            "h2".into(),
            &SaeAtomBasisKind::Periodic,
            phi.view(),
            jac.view(),
            dec.view(),
            masses.view(),
        );
        let fracs = entry
            .harmonic_energy_fractions
            .as_ref()
            .expect("periodic atom carries a harmonic spectrum");
        assert_eq!(fracs.len(), 2);
        let e1 = 2.0 * r1 * r1;
        let e2 = 2.0 * r2 * r2;
        assert!((fracs[0] - e1 / (e1 + e2)).abs() < 1.0e-12);
        assert!((fracs[1] - e2 / (e1 + e2)).abs() < 1.0e-12);
        assert_eq!(entry.dominant_harmonic, Some(2));
        let ratio = entry
            .second_harmonic_ratio
            .expect("both harmonics carry energy");
        assert!((ratio - (r2 / r1).powi(2)).abs() < 1.0e-12);
    }

    #[test]
    fn collapsed_circle_reads_one_dimensional_and_degenerate() {
        // Both harmonics decode onto the SAME output channel ⇒ the "circle" is a
        // single line: effective dim ≈ 1, degeneracy high, flagged collapsed.
        let (phi, jac, mut dec, masses) = planted_circle(64, 2.0);
        dec.fill(0.0);
        dec[[1, 0]] = 2.0;
        dec[[2, 0]] = 2.0; // second harmonic onto channel 0 as well
        let entry = atom_geometry_entry_from_parts(
            "c".into(),
            &SaeAtomBasisKind::Periodic,
            phi.view(),
            jac.view(),
            dec.view(),
            masses.view(),
        );
        let eff = entry.effective_output_dim.unwrap();
        assert!(eff < 1.5, "collapsed circle effective dim < 1.5, got {eff}");
        assert!(
            entry.degeneracy.unwrap() > 0.2,
            "collapse must post material degeneracy"
        );
        assert!(
            entry.is_collapsed(),
            "a circle flattened to a line is collapsed"
        );
    }
    #[test]
    fn inactive_atom_degrades_to_none_not_zero() {
        let (phi, jac, dec, _) = planted_circle(64, 1.0);
        let masses = Array1::<f64>::zeros(64); // atom active nowhere
        let entry = atom_geometry_entry_from_parts(
            "dead".into(),
            &SaeAtomBasisKind::Periodic,
            phi.view(),
            jac.view(),
            dec.view(),
            masses.view(),
        );
        assert_eq!(entry.n_active, 0);
        assert_eq!(entry.effective_output_dim, None);
        assert_eq!(entry.degeneracy, None);
        assert_eq!(entry.tangent_speed_mean, None);
        assert!(!entry.is_collapsed() && !entry.is_effectively_linear());
    }
}

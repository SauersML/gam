//! Regression test for issue #1042: an explicit all-zero `aniso_log_scales`
//! must reduce to the plain *isotropic* Matérn kernel, not a data-driven
//! anisotropic one.
//!
//! `aniso_log_scales = [0, 0]` is the natural way to ask `matern_basis` for the
//! isotropic kernel: the centered contrasts `ψ_a = η_a − mean(η)` are identically
//! zero, the metric weights `exp(2 ψ_a) = 1`, the radius is plain Euclidean
//! `‖t − c‖`, and the design must equal both the `aniso_log_scales = None` design
//! and the closed-form isotropic Matérn.
//!
//! The historical bug (`auto_seed_aniso_contrasts`, then named
//! `maybe_initialize_aniso_contrasts`, fired inside the Matérn forward design
//! build) discarded an *exactly* all-zero vector and replaced it with
//! geometry-derived contrasts from the spread of the center cloud, producing an
//! anisotropic design — and a **jump discontinuity at η = 0**: `[1e-9, -1e-9]`
//! was honored verbatim (≈ isotropic) while `[0, 0]` jumped to the data-driven
//! kernel. The fix makes the Matérn forward design a pure, continuous function
//! of the supplied η; geometry seeding remains a separate, Duchon-only concern.
//!
//! These tests build via the production `build_matern_basis` with
//! `identifiability = None` (bare kernel columns) so the design is exactly the
//! kernel matrix `Φ_{n,k} = φ(r)`.

use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu,
    build_matern_basis_literal_aniso,
};
use ndarray::{Array2, ArrayView2, array};

/// Raw (un-projected) Matérn kernel design for the supplied anisotropy, built
/// through the **public** forward entry (`build_matern_basis_literal_aniso`, the
/// one the `matern_basis` FFI uses) which honors an explicit all-zero η as the
/// isotropic metric rather than the κ-optimizer's geometry-seeding sentinel.
fn forward_kernel(
    points: ArrayView2<'_, f64>,
    centers: &Array2<f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso: Option<&[f64]>,
) -> Array2<f64> {
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        length_scale,
        nu,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::None,
        aniso_log_scales: aniso.map(<[f64]>::to_vec),
        nullspace_shrinkage_survived: None,
    };
    build_matern_basis_literal_aniso(points, &spec)
        .expect("forward Matérn kernel should build")
        .design
        .to_dense()
}

/// Closed-form isotropic Matérn-3/2 reference, `r = Euclidean distance`:
/// `φ(r) = (1 + √3 r/ℓ) · exp(−√3 r/ℓ)`.
fn isotropic_matern_32(
    points: ArrayView2<'_, f64>,
    centers: &Array2<f64>,
    length_scale: f64,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((points.nrows(), centers.nrows()));
    for i in 0..points.nrows() {
        for j in 0..centers.nrows() {
            let mut r2 = 0.0_f64;
            for a in 0..centers.ncols() {
                let d = points[[i, a]] - centers[[j, a]];
                r2 += d * d;
            }
            let s = (3.0_f64).sqrt() * r2.sqrt() / length_scale;
            out[[i, j]] = (1.0 + s) * (-s).exp();
        }
    }
    out
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// The core #1042 guard: on a center cloud with *very different per-axis
/// spreads* (exactly the geometry the discarded override exploited), the
/// explicit all-zero `aniso_log_scales = [0, 0]` design must equal the
/// closed-form isotropic Matérn-3/2 and the `None` design.
#[test]
fn explicit_zero_aniso_reduces_to_isotropic_matern() {
    // Axis 0 spans ~[0, 40], axis 1 spans ~[0, 0.5]: an 80× spread ratio, so the
    // old geometry seed produced a strongly anisotropic metric.
    let centers = array![
        [0.0, 0.0],
        [12.0, 0.18],
        [27.0, 0.05],
        [40.0, 0.42],
        [6.0, 0.31],
    ];
    let points = array![
        [3.0, 0.10],
        [19.0, 0.27],
        [33.0, 0.04],
        [8.0, 0.45],
        [25.0, 0.20],
        [38.0, 0.38],
    ];
    let ls = 1.0;
    let nu = MaternNu::ThreeHalves;

    let reference = isotropic_matern_32(points.view(), &centers, ls);
    let design_none = forward_kernel(points.view(), &centers, ls, nu, None);
    let design_zero = forward_kernel(points.view(), &centers, ls, nu, Some(&[0.0, 0.0]));
    // Near-zero contrasts were already honored verbatim (no discontinuity there).
    let design_eps = forward_kernel(points.view(), &centers, ls, nu, Some(&[1e-9, -1e-9]));

    // The `None` design defines isotropic ground truth; confirm it matches the
    // closed form to roundoff (sanity check on the reference itself).
    let none_vs_ref = max_abs_diff(&design_none, &reference);
    assert!(
        none_vs_ref < 1e-12,
        "isotropic (None) design should equal closed-form Matérn-3/2; max|diff| = {none_vs_ref:.3e}"
    );

    // The honored near-zero path is essentially isotropic (continuity below η=0).
    let eps_vs_ref = max_abs_diff(&design_eps, &reference);
    assert!(
        eps_vs_ref < 1e-7,
        "near-zero contrast [1e-9,-1e-9] should be ≈ isotropic; max|diff| = {eps_vs_ref:.3e}"
    );

    // The fix: explicit [0,0] must equal the isotropic kernel, NOT a data-driven
    // anisotropic one (the bug measured max|diff| ≈ 0.13–0.65 here).
    let zero_vs_ref = max_abs_diff(&design_zero, &reference);
    assert!(
        zero_vs_ref < 1e-12,
        "explicit aniso_log_scales=[0,0] must reduce to the isotropic Matérn kernel, but the \
         design differs by max|diff| = {zero_vs_ref:.3e} (the all-zero request was overridden by \
         data-driven anisotropy from the center cloud). For comparison the honored near-zero \
         contrast [1e-9,-1e-9] is {eps_vs_ref:.3e} from isotropic."
    );

    // ...and it must equal the `None` design exactly (same code path semantics).
    let zero_vs_none = max_abs_diff(&design_zero, &design_none);
    assert!(
        zero_vs_none < 1e-12,
        "explicit aniso=[0,0] design must equal the None design; max|diff| = {zero_vs_none:.3e}"
    );
}

/// Continuity through η = 0: the design must vary smoothly as the contrasts
/// shrink to zero — `[ε, −ε]` for ε ∈ {1e-3, 1e-6, 1e-9, 0} forms a monotone,
/// vanishing sequence of deviations from isotropic, with NO jump at exactly 0.
#[test]
fn matern_design_is_continuous_through_zero_anisotropy() {
    let centers = array![[0.0, 0.0], [10.0, 0.2], [25.0, 0.5], [4.0, 0.35]];
    let points = array![[2.0, 0.1], [15.0, 0.3], [22.0, 0.45]];
    let ls = 1.3;
    let nu = MaternNu::FiveHalves;

    let reference = forward_kernel(points.view(), &centers, ls, nu, None);

    let mut prev = f64::INFINITY;
    for &eps in &[1e-3, 1e-6, 1e-9] {
        let design = forward_kernel(points.view(), &centers, ls, nu, Some(&[eps, -eps]));
        let gap = max_abs_diff(&design, &reference);
        assert!(
            gap < prev,
            "deviation from isotropic must shrink monotonically as η→0: gap({eps:.0e}) = {gap:.3e} \
             not < previous {prev:.3e}"
        );
        prev = gap;
    }

    // The limit point η = exactly 0 must continue the sequence to ≈ 0, not jump.
    let design_zero = forward_kernel(points.view(), &centers, ls, nu, Some(&[0.0, 0.0]));
    let gap_zero = max_abs_diff(&design_zero, &reference);
    assert!(
        gap_zero < prev,
        "design at exactly η=0 must continue the vanishing sequence (no jump discontinuity): \
         gap(0) = {gap_zero:.3e} not < gap(1e-9) = {prev:.3e}"
    );
    assert!(
        gap_zero < 1e-12,
        "design at exactly η=0 must equal isotropic to roundoff; gap = {gap_zero:.3e}"
    );
}

/// The same guarantee across smoothness ν: explicit [0,0] is isotropic for
/// ν ∈ {3/2, 5/2, 7/2, 9/2}. (ν=1/2 has a singular Laplacian at collocation for
/// d>1 and is not buildable with these centers, independent of anisotropy.)
#[test]
fn explicit_zero_aniso_is_isotropic_across_nu() {
    let centers = array![[0.0, 0.0], [30.0, 0.3], [15.0, 0.1], [45.0, 0.5]];
    let points = array![[5.0, 0.2], [22.0, 0.05], [40.0, 0.4]];
    let ls = 2.0;
    for nu in [
        MaternNu::ThreeHalves,
        MaternNu::FiveHalves,
        MaternNu::SevenHalves,
        MaternNu::NineHalves,
    ] {
        let design_none = forward_kernel(points.view(), &centers, ls, nu, None);
        let design_zero = forward_kernel(points.view(), &centers, ls, nu, Some(&[0.0, 0.0]));
        let gap = max_abs_diff(&design_zero, &design_none);
        assert!(
            gap < 1e-12,
            "explicit aniso=[0,0] must be isotropic for ν={nu:?}; max|diff| = {gap:.3e}"
        );
    }
}

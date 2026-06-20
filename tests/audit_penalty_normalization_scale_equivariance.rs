//! Audit: every 1-D B-spline smooth ships a Frobenius-NORMALIZED penalty into
//! the REML objective, and its design + penalty are SCALE-EQUIVARIANT.
//!
//! Root-cause class behind #1364 / #1365 / #1266: smooth-basis penalty operators
//! must enter the REML objective on a common (unit-Frobenius) scale, because the
//! shipped design penalty is `β'(S/c)β` and the smoothing parameter `λ`
//! multiplies that normalized block (see the comment block at
//! `src/terms/smooth/design_construction.rs:2898-2920`). A basis that ships its
//! penalty *un-normalized* (`normalization_scale = 1.0` on a raw operator) puts
//! `λ` on a basis-dependent scale; the outer `λ`-search heuristics then mis-
//! calibrate and the smooth under- or over-smooths relative to a normalized
//! basis like `bs="cr"`.
//!
//! This test certifies the invariant directly at construction time for the
//! single-penalty 1-D B-spline paths (open `bs="ps"` and cyclic `bs="cc"`),
//! across penalty orders, without needing a fitted model:
//!   1. the active penalty block is Frobenius-normalized (‖S‖_F ≈ 1), and
//!   2. the design columns are invariant under a pure rescaling of the abscissa
//!      (a B-spline is invariant to an affine reparam of x when knots scale).

use gam::terms::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotPlacement, BSplineKnotSpec,
    OneDimensionalBoundary, build_bspline_basis_1d,
};
use ndarray::Array1;

fn data() -> Array1<f64> {
    let n = 200usize;
    Array1::from_iter((0..n).map(|i| {
        let t = i as f64 / (n as f64 - 1.0);
        -3.0 + 6.0 * (0.4 * t + 0.6 * t * t)
    }))
}

fn frob(m: &ndarray::Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn open_spec(penalty_order: usize) -> BSplineBasisSpec {
    BSplineBasisSpec {
        degree: 3,
        penalty_order,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(20),
            placement: BSplineKnotPlacement::Uniform,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: Default::default(),
    }
}

fn cyclic_spec(penalty_order: usize) -> BSplineBasisSpec {
    let x = data();
    let (lo, hi) = (
        x.iter().cloned().fold(f64::INFINITY, f64::min),
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    BSplineBasisSpec {
        degree: 3,
        penalty_order,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(20),
            placement: BSplineKnotPlacement::Uniform,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Cyclic { start: lo, end: hi },
        boundary_conditions: Default::default(),
    }
}

#[test]
fn single_penalty_bspline_penalties_are_frobenius_normalized() {
    let x = data();
    // (label, spec-builder) for the single-penalty 1-D B-spline smooths.
    let cases: Vec<(&str, BSplineBasisSpec)> = vec![
        ("ps open order=2", open_spec(2)),
        ("ps open order=1", open_spec(1)),
        ("ps open order=3", open_spec(3)),
        ("cc cyclic order=2", cyclic_spec(2)),
    ];

    for (label, spec) in cases {
        let built = build_bspline_basis_1d(x.view(), &spec)
            .unwrap_or_else(|e| panic!("{label}: basis build failed: {e:?}"));
        assert!(
            !built.penalties.is_empty(),
            "{label}: expected at least one active penalty"
        );
        // The single active wiggliness block must be Frobenius-normalized so its
        // λ shares the unit-Frobenius scale used by cr / duchon / tensor.
        let n = frob(&built.penalties[0]);
        assert!(
            (n - 1.0).abs() < 1e-9,
            "{label}: penalty is NOT Frobenius-normalized: ‖S‖_F = {n:.6e} (expected 1.0). \
             An un-normalized penalty puts λ on a basis-dependent scale and mis-calibrates \
             REML's λ-search (the #1364/#1365 defect class)."
        );
    }
}

#[test]
fn single_penalty_bspline_design_is_scale_equivariant() {
    let x = data();
    for (label, spec) in [
        ("ps open order=2", open_spec(2)),
        ("cc cyclic order=2", cyclic_spec(2)),
    ] {
        // Build on x and on 100*x. With the cyclic boundary the period scales
        // too, so rebuild the spec at scale for the cyclic case.
        let base = build_bspline_basis_1d(x.view(), &spec).unwrap();
        let base_d = base.design.as_dense_ref().expect("dense").to_owned();

        let xc = x.mapv(|v| v * 100.0);
        let spec_c = match spec.boundary {
            OneDimensionalBoundary::Cyclic { start, end } => {
                let mut s = spec.clone();
                s.boundary = OneDimensionalBoundary::Cyclic {
                    start: start * 100.0,
                    end: end * 100.0,
                };
                s
            }
            _ => spec.clone(),
        };
        let scaled = build_bspline_basis_1d(xc.view(), &spec_c).unwrap();
        let scaled_d = scaled.design.as_dense_ref().expect("dense");

        let drift = (scaled_d - &base_d)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            drift < 1e-9,
            "{label}: B-spline design is NOT scale-equivariant: max column drift {drift:.3e} \
             under x -> 100x (expected < 1e-9)."
        );
    }
}

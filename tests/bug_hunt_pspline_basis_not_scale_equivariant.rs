//! Bug hunt: a 1-D P-spline (`bs="ps"`) basis + difference penalty is NOT
//! scale-equivariant under a pure rescaling of the covariate.
//!
//! A P-spline places its knots on the data range and penalizes *differences of
//! adjacent B-spline coefficients* (a unitless, coefficient-space penalty). The
//! B-spline value at a point depends only on the point's position relative to
//! the (data-derived) knots, which are themselves a linear function of the
//! covariate. Therefore, if we rescale `x -> c*x`, the auto-placed knots scale
//! by the same `c`, every B-spline column is UNCHANGED at the corresponding
//! data points, and the difference penalty is literally the same matrix.
//!
//! Consequently the design matrix `B` evaluated at the data and the penalty
//! `S` must both be invariant to `c` (to rounding). If they are not, the whole
//! penalized REML problem `(B, S, y)` is scale-dependent and the fit drifts
//! purely from the abscissa magnitude -- the same defect class as the thin-plate
//! translation/rescaling bugs (#1215 / #1269 / #1271), but on the `bs="ps"`
//! path that those fixes left untouched.
//!
//! Observed at the Python `gamfit.fit` level: fitting `y ~ s(x, bs="ps")` on
//! `x` vs `100*x` converges to entirely different REML `rho` (4.35 vs -1.99)
//! and a ~1.5% prediction drift. This test localizes the leak to the basis /
//! penalty construction itself.

use gam::terms::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotPlacement, BSplineKnotSpec,
    OneDimensionalBoundary, build_bspline_basis_1d,
};
use ndarray::Array1;

fn ps_spec() -> BSplineBasisSpec {
    // Mirror the default `bs="ps"` smooth: cubic, 2nd-order difference penalty,
    // automatic uniform knot placement, weighted sum-to-zero identifiability.
    BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
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

fn sample_x() -> Array1<f64> {
    // Deterministic, sorted, irregular spacing.
    let n = 200usize;
    Array1::from_iter((0..n).map(|i| {
        let t = i as f64 / (n as f64 - 1.0);
        // map to [-3, 3] with a mild nonuniform warp
        -3.0 + 6.0 * (0.5 * t + 0.5 * t * t)
    }))
}

#[test]
fn pspline_design_and_penalty_are_scale_equivariant() {
    let x = sample_x();
    let scales = [1.0_f64, 10.0, 100.0, 1000.0];

    let build = |c: f64| {
        let xc: Array1<f64> = x.mapv(|v| v * c);
        build_bspline_basis_1d(xc.view(), &ps_spec()).expect("ps basis builds")
    };

    let base = build(1.0);
    let base_design = base.design.as_dense_ref().expect("dense design").to_owned();
    assert!(
        !base.active_penalties.is_empty(),
        "expected at least one penalty"
    );
    let base_pen = base.active_penalties[0].matrix.clone();

    let mut worst_design = 0.0_f64;
    let mut worst_pen = 0.0_f64;
    let mut worst_scale = 1.0_f64;

    for &c in &scales[1..] {
        let r = build(c);
        let d = r.design.as_dense_ref().expect("dense design");
        assert_eq!(
            d.dim(),
            base_design.dim(),
            "design shape changed with scale"
        );

        // The B-spline columns at the (correspondingly scaled) data points must
        // be identical to those at scale 1 -- a B-spline is invariant to an
        // affine reparam of its abscissa when the knots scale with it.
        let dd = (d - &base_design)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        let pen = &r.active_penalties[0].matrix;
        assert_eq!(
            pen.dim(),
            base_pen.dim(),
            "penalty shape changed with scale"
        );
        let dp = (pen - &base_pen)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));

        if dd > worst_design {
            worst_design = dd;
            worst_scale = c;
        }
        worst_pen = worst_pen.max(dp);
    }

    eprintln!(
        "ps scale-equivariance: worst |design drift| = {worst_design:.3e} (at scale {worst_scale}), \
         worst |penalty drift| = {worst_pen:.3e}"
    );

    // A correctly scale-equivariant P-spline basis + difference penalty would
    // match to rounding (~1e-12). Anything larger means the abscissa magnitude
    // is leaking into the model geometry.
    assert!(
        worst_design < 1e-9,
        "P-spline design matrix is NOT scale-equivariant: worst column drift {worst_design:.3e} \
         at scale {worst_scale} (expected < 1e-9). The abscissa magnitude leaks into the basis, \
         so the penalized REML problem (B, S, y) is scale-dependent and the fit drifts purely \
         from rescaling x."
    );
    assert!(
        worst_pen < 1e-9,
        "P-spline difference penalty is NOT scale-equivariant: worst entry drift {worst_pen:.3e} \
         (expected < 1e-9)."
    );
}

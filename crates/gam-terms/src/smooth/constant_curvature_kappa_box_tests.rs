//! gam#2716 / gam#2687: the κ search box is derived over the configuration the
//! constant-curvature basis EVALUATES, not over `data` alone.
//!
//! The kernel is `K_κ(data, centers)`, and the RKHS penalty adds
//! `K_κ(centers, centers)`. `validate_chart_points` checks the chart gauge on
//! data **and** centers; `ConstantCurvature::distance` is called once per
//! evaluated pair. So the radius the box is denominated in is
//! `R = max‖p‖` over `data ∪ centers`. Before #2716 it was taken over `data`.
//!
//! Every assertion below is stated against the *geometry* — the shipped
//! `ConstantCurvature` distance and chart gauge at the box's own endpoints —
//! rather than against a recomputation of the bound's formula, so a formula that
//! agreed with itself but disagreed with the kernel would still fail here.

use crate::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    constant_curvature_center_chart_radius2, constant_curvature_data_chart_radius2,
    constant_curvature_kernel_matrix,
};
use crate::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    constant_curvature_kappa_bounds, constant_curvature_kappa_chart_fraction,
};
use gam_geometry::manifolds::constant_curvature::ConstantCurvature;
use ndarray::{Array2, array};

fn spec_with(strategy: CenterStrategy, dim: usize) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "curv".to_string(),
            basis: SmoothBasisSpec::ConstantCurvature {
                feature_cols: (0..dim).collect(),
                spec: ConstantCurvatureBasisSpec {
                    center_strategy: strategy,
                    kappa: 0.0,
                    kappa_fixed: false,
                    length_scale: 0.0,
                    length_scale_fixed: false,
                    double_penalty: false,
                    identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
                },
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
    }
}

/// A ring of `n` points of radius `r` in the first two coordinates of `dim`.
fn ring(n: usize, r: f64, dim: usize) -> Array2<f64> {
    let mut data = Array2::<f64>::zeros((n, dim));
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64) / (n as f64);
        data[(i, 0)] = r * theta.cos();
        data[(i, 1)] = r * theta.sin();
    }
    data
}

/// Every data-driven strategy selects data rows verbatim or convex combinations
/// of them, so `max‖c‖ ≤ max‖x‖` and the box must be **bit-identical** to its
/// pre-#2716 value `±F/max‖x‖²`. This is the compatibility half of the fix: the
/// repair may only move a configuration whose centers can leave the data hull.
#[test]
fn data_driven_center_strategies_keep_the_pre_2716_box_bit_for_bit() {
    let data = ring(64, 0.6, 2);
    // The pre-#2716 expression, evaluated here rather than quoted, so this is a
    // statement about the endpoint and not about `0.6·0.6` rounding.
    let max_r2 = data
        .outer_iter()
        .map(|row| row.dot(&row))
        .fold(0.0, f64::max);
    let expected = constant_curvature_kappa_chart_fraction() / max_r2;
    for strategy in [
        CenterStrategy::FarthestPoint { num_centers: 8 },
        CenterStrategy::KMeans {
            num_centers: 8,
            max_iter: 10,
        },
        CenterStrategy::EqualMass { num_centers: 8 },
        CenterStrategy::EqualMassCovarRepresentative { num_centers: 8 },
        CenterStrategy::Auto(Box::new(CenterStrategy::FarthestPoint { num_centers: 8 })),
    ] {
        let spec = spec_with(strategy.clone(), 2);
        let (lo, hi) = constant_curvature_kappa_bounds(data.view(), &spec, 0);
        assert_eq!(
            (lo, hi),
            (-expected, expected),
            "{strategy:?}: the κ box must not move for a strategy whose centers \
             are inside the data hull"
        );
    }
}

/// #2716's measured configuration: data at `max‖x‖ = 0.3`, three USER-PROVIDED
/// centers at `‖c‖ = 0.8` (ratio 2.67 > 2). The fold over the evaluated pairs is
/// at `κ = 1/(0.3·0.8) = 4.1667`; the pre-#2716 box's upper end was
/// `0.5/0.3² = 5.5556`, i.e. **1.333× PAST it**, and the box therefore contained
/// both `κ = 5.5556` and its involution twin `κ = 3.125`.
#[test]
fn user_provided_centers_beyond_twice_the_data_radius_no_longer_put_the_box_past_the_fold_2716() {
    let data = ring(32, 0.3, 2);
    let centers = array![[0.8, 0.0], [-0.8, 0.0], [0.0, 0.8]];
    let spec = spec_with(CenterStrategy::UserProvided(centers.clone()), 2);
    let (lo, hi) = constant_curvature_kappa_bounds(data.view(), &spec, 0);

    let fold = 1.0 / (0.3 * 0.8);
    let old_hi = constant_curvature_kappa_chart_fraction() / (0.3 * 0.3);
    assert!(
        old_hi > fold,
        "the configuration this test is about: the pre-#2716 upper end {old_hi} \
         must be past the fold {fold}"
    );
    // The box is denominated in `R = max(R_x, R_c) = 0.8`, so it is strictly
    // inside the data × center fold (1/0.24) and is exactly the half-margin to
    // the center × center fold (1/0.64) the penalty Gram also evaluates.
    assert!(
        (hi - constant_curvature_kappa_chart_fraction() / 0.64).abs() <= 1e-12,
        "the upper end must be F/max(R_x, R_c)²: got {hi}, expected {}",
        constant_curvature_kappa_chart_fraction() / 0.64
    );
    assert!(
        hi < fold,
        "and it must be strictly inside the data × center fold {fold}"
    );
    assert_eq!(lo, -hi, "the window stays symmetric: one radius, two walls");

    // Stated against the geometry, not the formula. Note that `D` itself cannot
    // witness the fold from one side: for an exactly anti-aligned pair it is the
    // perfect square `(1 − κ‖x‖‖c‖)²`, so it returns to POSITIVE values past the
    // fold — 0.1111 at the pre-#2716 upper end. What witnesses the fold is the
    // involution `κ ↦ 1/(κ‖x‖²‖c‖²)`, which fixes the pair's scale-free geodesic
    // separation: a box that contains a κ and its twin is doubly covered.
    let x = array![0.3, 0.0];
    let c = array![-0.8, 0.0];
    let rho2 = x.dot(&x) * c.dot(&c);
    let denom = |k: f64| 1.0 + 2.0 * k * x.dot(&c) + k * k * rho2;
    let floor = (1.0_f64 - constant_curvature_kappa_chart_fraction()).powi(2);
    assert!(
        denom(hi) >= floor - 1e-12,
        "at the box's upper end {hi} the pair gauge D = {} must stay at or above \
         (1−F)² = {floor}",
        denom(hi)
    );
    let antipodal_fraction = |k: f64| {
        ConstantCurvature::new(2, k)
            .distance(x.view(), c.view())
            .map(|d| d * k.sqrt() / std::f64::consts::PI)
    };
    let old_twin = 1.0 / (old_hi * rho2);
    assert!(
        old_twin > 0.0 && old_twin < old_hi,
        "the twin {old_twin} of the pre-#2716 upper end must itself be inside the \
         old box (0, {old_hi}] — that is what 'doubly covered' means"
    );
    let (f_end, f_twin) = (
        antipodal_fraction(old_hi).expect("past the fold the chart still evaluates"),
        antipodal_fraction(old_twin).expect("the twin is interior"),
    );
    assert!(
        (f_end - f_twin).abs() <= 1e-12,
        "the defect: the pre-#2716 box contained BOTH κ = {old_hi} and κ = {old_twin}, \
         which give the extreme pair an identical scale-free geometry \
         ({f_end} vs {f_twin})"
    );
    // The repaired box is not doubly covered: every κ it admits has its twin
    // strictly outside it.
    let new_twin = 1.0 / (hi * rho2);
    assert!(
        new_twin > hi,
        "the repaired box's upper end {hi} must sit BELOW its own involution fixed \
         point, so the twin {new_twin} is outside the box"
    );

    // The κ<0 half, which #2716 did not name: the chart gauge is per POINT and
    // the centers are points too. At the new lower end every center is still
    // inside the chart; at the pre-#2716 lower end the farthest center is not,
    // and `validate_chart_points` turns that into a hard basis-build error
    // rather than a rail.
    let old_lo = -constant_curvature_kappa_chart_fraction() / (0.3 * 0.3);
    assert!(
        1.0 + lo * 0.64 > 0.0,
        "at the box's lower end {lo} the farthest center (‖c‖² = 0.64) must stay \
         inside the chart; gauge = {}",
        1.0 + lo * 0.64
    );
    assert!(
        1.0 + old_lo * 0.64 <= 0.0,
        "the defect: at the pre-#2716 lower end {old_lo} the farthest center is \
         OUTSIDE the chart; gauge = {}",
        1.0 + old_lo * 0.64
    );
    // …and the kernel agrees, through the shipped builder rather than by algebra.
    assert!(
        constant_curvature_kernel_matrix(data.view(), centers.view(), lo, 1.0).is_ok(),
        "the kernel must build at the box's own lower end"
    );
    assert!(
        constant_curvature_kernel_matrix(data.view(), centers.view(), old_lo, 1.0).is_err(),
        "the defect: the kernel could not build at the pre-#2716 lower end"
    );
    // The fold is a REFUSAL in the shipped distance, so the box's upper end
    // being inside it is checkable end to end as well.
    let manifold_at_fold = ConstantCurvature::new(2, fold);
    assert!(
        manifold_at_fold.distance(x.view(), c.view()).is_err(),
        "the fold must be where the shipped distance refuses"
    );
    assert!(
        ConstantCurvature::new(2, hi)
            .distance(x.view(), c.view())
            .is_ok(),
        "the box's upper end must be strictly inside it"
    );
}

/// `UniformGrid` centers are the Cartesian product of per-axis linspaces over
/// the data's BOUNDING BOX, so a corner center sits at radius up to `√d·max‖x‖`.
/// That crosses the `‖c‖ ≥ 2·max‖x‖` threshold at `d ≥ 4` with no user input at
/// all — #2716 flagged this as an unmeasured hypothesis; here it is measured.
#[test]
fn uniform_grid_corner_centers_leave_the_hull_and_move_the_box_2716() {
    // The 2d coordinate poles of a radius-1 ball in d = 5: every axis spans
    // [−1, 1], so the grid's corner is at radius √5 = 2.236 > 2 while the data
    // radius is 1.
    let dim = 5usize;
    let mut data = Array2::<f64>::zeros((2 * dim, dim));
    for axis in 0..dim {
        data[(2 * axis, axis)] = 1.0;
        data[(2 * axis + 1, axis)] = -1.0;
    }
    let spec = spec_with(CenterStrategy::UniformGrid { points_per_dim: 3 }, dim);
    let (lo, hi) = constant_curvature_kappa_bounds(data.view(), &spec, 0);

    let corner_r2 = dim as f64;
    // The nearest fold over the pairs the basis evaluates: a data point against
    // a corner center (1/(1·√d)), and — via the RKHS penalty Gram — a corner
    // center against the opposite corner (1/d, the binding one).
    let data_center_fold = 1.0 / corner_r2.sqrt();
    let center_center_fold = 1.0 / corner_r2;
    let old_hi = constant_curvature_kappa_chart_fraction() / 1.0;
    assert!(
        old_hi > data_center_fold && old_hi > center_center_fold,
        "d = {dim}: the pre-#2716 upper end {old_hi} must be past BOTH corner \
         folds ({data_center_fold}, {center_center_fold})"
    );
    let expected = constant_curvature_kappa_chart_fraction() / corner_r2;
    assert!(
        (hi - expected).abs() <= 1e-12 && (lo + expected).abs() <= 1e-12,
        "the box must be denominated in the CORNER radius, the largest evaluated \
         point: got [{lo}, {hi}], expected ±{expected}"
    );
    assert!(
        hi < center_center_fold,
        "and it must be strictly inside the binding fold {center_center_fold}"
    );
}

/// A degenerate configuration (every point and every center at the origin)
/// gives every κ the same design, so κ is not identified and the window is
/// unbounded, on both radii since either one can be the degenerate side. The κ
/// profile refuses the non-finite bracket instead of searching a flat
/// criterion. A floor on `R²` had manufactured a finite window here.
#[test]
fn degenerate_radii_leave_the_window_unbounded() {
    let data = Array2::<f64>::zeros((8, 2));
    for strategy in [
        CenterStrategy::FarthestPoint { num_centers: 4 },
        CenterStrategy::UserProvided(Array2::<f64>::zeros((3, 2))),
        CenterStrategy::UniformGrid { points_per_dim: 2 },
    ] {
        let spec = spec_with(strategy.clone(), 2);
        let (lo, hi) = constant_curvature_kappa_bounds(data.view(), &spec, 0);
        assert_eq!(
            (lo, hi),
            (f64::NEG_INFINITY, f64::INFINITY),
            "{strategy:?}: an all-origin configuration has no radius to bound κ by"
        );
    }
}

/// κ has units of 1/length², so the window is `±F/R²` at every radius. A cloud
/// recorded in units 1e-5 of another has a window 1e10 times wider, not one
/// clipped at a floor on `R²`.
#[test]
fn the_window_scales_as_the_inverse_squared_radius() {
    let f = constant_curvature_kappa_chart_fraction();
    let feature_cols = [0usize, 1usize];
    for radius in [1.0e-5_f64, 1.0e-3, 1.0, 1.0e3] {
        let data = ring(32, radius, 2);
        let spec = spec_with(CenterStrategy::FarthestPoint { num_centers: 6 }, 2);
        let (lo, hi) = constant_curvature_kappa_bounds(data.view(), &spec, 0);
        let r2 = constant_curvature_data_chart_radius2(data.view(), &feature_cols);
        assert!(
            (hi * r2 / f - 1.0).abs() <= 1.0e-12 && (lo * r2 / f + 1.0).abs() <= 1.0e-12,
            "radius {radius}: the window must be ±F/R², got [{lo}, {hi}] for R² = {r2}"
        );
    }
}

/// Data far from the origin with centers near it (the mirror of #2716's
/// configuration). The evaluated-pair fold is then at `1/(R_x·R_c)`, which is
/// *larger* than `1/R_x²` — so a bound taken pair-wise would legitimately WIDEN
/// here. It must not, because the box has to be freeze-invariant: this exact
/// configuration is what a frozen data-driven center set looks like.
#[test]
fn centers_inside_the_data_hull_never_widen_the_box() {
    let data = ring(16, 2.0, 2);
    let centers = array![[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]];
    let spec = spec_with(CenterStrategy::UserProvided(centers), 2);
    let (lo, hi) = constant_curvature_kappa_bounds(data.view(), &spec, 0);
    let data_only = constant_curvature_kappa_chart_fraction() / 4.0;
    assert!(
        (hi - data_only).abs() <= 1e-12 && (lo + data_only).abs() <= 1e-12,
        "centers inside the hull must leave the box at the data radius, got [{lo}, {hi}]"
    );
    assert!(
        hi < constant_curvature_kappa_chart_fraction() / (2.0 * 0.5),
        "and it must stay strictly inside the wider data × center fold, which a \
         pair-wise bound would have handed the optimizer"
    );
}

/// gam#2687 read the symmetric window as the κ<0 chart constraint mirrored onto
/// "a branch that has no such constraint". The branch does have one, and this
/// pins it per strategy: **each wall retreats by the same fraction `F` from a
/// wall its OWN branch has**, and both walls are the shipped geometry's, not
/// this file's algebra.
///
/// * κ < 0 — the PER-POINT conformal gauge `λ = 1 + κR²`; at `κ_min` it has lost
///   exactly `F` of its flat value, `λ = 1 − F`.
/// * κ > 0 — the PER-PAIR Möbius denominator, `D = (1 − κR²)²` for an
///   anti-aligned pair at the box radius; at `κ_max`, `√D = 1 − F`.
///
/// **What this test can and cannot discriminate.** It cannot separate the two
/// DERIVATIONS numerically: both land on `F/R²`, which is gam#2687's resolution
/// (`19edc9a2d`: "no number changed; what changed is the derivation"). Asserting
/// `λ(κ_max) = 1 + F` as a foil would be scoring the observation against itself,
/// since that holds at the same κ. What it does discriminate is whether the
/// positive wall is still tied to a wall that EXISTS: the shipped
/// `ConstantCurvature::distance` must refuse at `κ = 1/R²` and accept at `κ_max`,
/// on every center strategy. That fails if the positive end is ever widened off
/// the fold (the `9.5/R²` this issue proposed), or if `R` stops being taken over
/// `data ∪ centers` so that the fold it retreats from is not the evaluated one.
#[test]
fn each_wall_retreats_by_f_from_its_own_branchs_gauge_never_the_others_2687() {
    let data = ring(48, 0.7, 2);
    let f = constant_curvature_kappa_chart_fraction();
    for strategy in [
        CenterStrategy::FarthestPoint { num_centers: 6 },
        CenterStrategy::KMeans {
            num_centers: 6,
            max_iter: 10,
        },
        CenterStrategy::EqualMass { num_centers: 6 },
        CenterStrategy::UniformGrid { points_per_dim: 3 },
        CenterStrategy::UserProvided(array![[1.4, 0.0], [-1.4, 0.0], [0.0, 0.2]]),
    ] {
        let spec = spec_with(strategy.clone(), 2);
        let (lo, hi) = constant_curvature_kappa_bounds(data.view(), &spec, 0);
        let feature_cols = [0usize, 1usize];
        let r2 = constant_curvature_data_chart_radius2(data.view(), &feature_cols).max(
            constant_curvature_center_chart_radius2(data.view(), &feature_cols, &strategy),
        );

        // κ < 0: the per-POINT gauge at the lower wall.
        let lambda = 1.0 + lo * r2;
        assert!(
            (lambda - (1.0 - f)).abs() <= 1e-12,
            "{strategy:?}: κ_min = {lo} must leave the per-point chart gauge at \
             1 − F = {}, got λ = {lambda}",
            1.0 - f
        );

        // κ > 0: the per-PAIR gauge at the upper wall, evaluated on the shipped
        // geometry — an anti-aligned pair at the box radius, which is the worst
        // case the box is denominated in.
        let r = r2.sqrt();
        let x = array![r, 0.0];
        let c = array![-r, 0.0];
        let d_pair = 1.0 + 2.0 * hi * x.dot(&c) + hi * hi * x.dot(&x) * c.dot(&c);
        assert!(
            (d_pair.sqrt() - (1.0 - f)).abs() <= 1e-9,
            "{strategy:?}: κ_max = {hi} must leave the per-pair Möbius gauge at \
             1 − F = {}, got √D = {}",
            1.0 - f,
            d_pair.sqrt()
        );

        // The wall itself, through the shipped distance rather than through the
        // algebra above: it refuses AT the fold and accepts at the box's end.
        assert!(
            ConstantCurvature::new(2, 1.0 / r2)
                .distance(x.view(), c.view())
                .is_err(),
            "{strategy:?}: the κ>0 branch must have a wall of its own at 1/R² = {}",
            1.0 / r2
        );
        assert!(
            ConstantCurvature::new(2, hi)
                .distance(x.view(), c.view())
                .is_ok(),
            "{strategy:?}: and the box's upper end {hi} must be strictly inside it"
        );
        // The per-point guard, end to end, at both walls: `validate_chart_points`
        // runs on data AND centers inside the kernel build.
        for kappa in [lo, hi] {
            assert!(
                constant_curvature_kernel_matrix(data.view(), data.view(), kappa, 1.0).is_ok(),
                "{strategy:?}: the per-point guard must accept at the box end {kappa}"
            );
        }
    }
}

/// The regression this file exists to prevent, and the one that made the
/// pair-wise bound untenable: `freeze_term_collection_from_design` rewrites a
/// fitted term's strategy as `UserProvided(realized centers)`, so the SAME
/// geometry is described data-driven before the fit and user-provided after it.
/// The box must not move across that re-description — measured on the #944
/// coverage fixture, a `κ_max` that read the realized center radius moved from
/// 1.412031543260163 (fit) to 1.4127975943783915 (inference), which put κ̂ off
/// its own bound, had it classified interior, and refused the fit as a
/// non-stationary point estimate.
#[test]
fn freezing_a_data_driven_center_set_to_user_provided_does_not_move_the_box() {
    let data = ring(40, 0.6, 2);
    // Whatever a data-driven strategy selects is a subset of the rows, and its
    // radius is at most the data radius. Take the extreme case (a center AT the
    // data radius) and a typical one (all centers well inside), and a set that
    // deliberately misses the outermost ring point.
    for realized in [
        array![[0.6, 0.0], [-0.6, 0.0], [0.0, 0.0]],
        array![[0.3, 0.1], [-0.2, 0.25], [0.0, 0.0]],
        array![[0.05, 0.0], [-0.05, 0.0], [0.0, 0.0]],
    ] {
        let before = constant_curvature_kappa_bounds(
            data.view(),
            &spec_with(CenterStrategy::FarthestPoint { num_centers: 3 }, 2),
            0,
        );
        let after = constant_curvature_kappa_bounds(
            data.view(),
            &spec_with(CenterStrategy::UserProvided(realized.clone()), 2),
            0,
        );
        assert_eq!(
            before, after,
            "freezing {realized:?} must not move the κ box: {before:?} -> {after:?}"
        );
    }
}

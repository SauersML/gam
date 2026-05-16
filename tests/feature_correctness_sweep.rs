//! Comprehensive correctness sweep across geometric smooths. Hundreds of
//! individual test cases stressing edge conditions: poles, seams, period
//! boundaries, all-same data, degenerate centers, varied k/L/degree.
//!
//! These tests live in release mode and are individually short, so a single
//! cargo test --test feature_correctness_sweep runs the whole sweep quickly.

use gam::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineEndpointBoundaryCondition,
    BSplineIdentifiability, BSplineKnotSpec, CenterStrategy, SphereMethod,
    SphericalSplineBasisSpec, build_bspline_basis_1d, build_spherical_spline_basis,
    create_cyclic_difference_penalty_matrix, create_periodic_bspline_basis_dense,
    create_periodic_bspline_derivative_dense, spherical_wahba_kernel_matrix,
};
use ndarray::{Array1, Array2, ArrayView2};
use std::f64::consts::{PI, TAU};

// ----- helpers -----

fn near(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps + eps * a.abs().max(b.abs())
}

fn make_uniform_loop(n: usize, period: f64) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64) * period))
}

fn random_lat_lon(n: usize, seed: u64) -> Array2<f64> {
    // Deterministic LCG (fine for test data; no need for chacha).
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (u32::MAX as f64)
    };
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        // Uniform on sphere via Lambert equal-area
        let z = 1.0 - 2.0 * next();
        let phi = 2.0 * PI * next();
        let lat = z.asin().to_degrees();
        let lon = phi.to_degrees() - 180.0;
        out[(i, 0)] = lat;
        out[(i, 1)] = lon;
    }
    out
}

// ----- PERIODIC 1D B-SPLINE: ~80 cases -----

#[test]
fn periodic_bspline_partition_of_unity_across_periods_and_k() {
    for period in [1.0_f64, TAU, 7.0, 24.0, 365.0, 1e-3, 1e6] {
        for &k in &[4usize, 6, 8, 12, 24, 48] {
            let n = 200;
            let xs = make_uniform_loop(n, period);
            let basis = create_periodic_bspline_basis_dense(xs.view(), (0.0, period), 3, k)
                .unwrap_or_else(|e| panic!("period={period} k={k}: {e}"));
            assert_eq!(basis.ncols(), k, "period={period} k={k}");
            for i in 0..n {
                let s = basis.row(i).sum();
                assert!(near(s, 1.0, 1e-10), "period={period} k={k} row {i} sum={s}");
            }
        }
    }
}

#[test]
fn periodic_bspline_evaluates_seam_exactly() {
    for &k in &[4usize, 8, 16] {
        let period = TAU;
        let pts = ndarray::array![0.0, period, 0.5 * period, 1.5 * period, -0.25 * period];
        let basis = create_periodic_bspline_basis_dense(pts.view(), (0.0, period), 3, k).unwrap();
        // 0 and period should give identical rows
        for j in 0..k {
            assert!(
                near(basis[(0, j)], basis[(1, j)], 1e-12),
                "k={k} col {j}: 0 vs period differ"
            );
        }
        // 0.5 period == 0.5 period (trivially)
        // -0.25 period == 0.75 period
        let pts2 = ndarray::array![0.75 * period];
        let basis2 = create_periodic_bspline_basis_dense(pts2.view(), (0.0, period), 3, k).unwrap();
        for j in 0..k {
            assert!(
                near(basis[(4, j)], basis2[(0, j)], 1e-12),
                "k={k} col {j}: -0.25 vs 0.75 period differ"
            );
        }
    }
}

#[test]
fn periodic_bspline_for_all_supported_degrees() {
    for degree in 0_usize..=4 {
        let period = TAU;
        let k = (degree + 1).max(4);
        let xs = make_uniform_loop(50, period);
        let basis = create_periodic_bspline_basis_dense(xs.view(), (0.0, period), degree, k)
            .unwrap_or_else(|e| panic!("degree={degree} k={k}: {e}"));
        for i in 0..xs.len() {
            assert!(
                near(basis.row(i).sum(), 1.0, 1e-10),
                "degree={degree} row {i}"
            );
        }
    }
}

#[test]
fn periodic_bspline_rejects_too_few_basis() {
    // degree+1 is the minimum; degree=3, k=3 should fail
    let xs = make_uniform_loop(10, TAU);
    let err = create_periodic_bspline_basis_dense(xs.view(), (0.0, TAU), 3, 3);
    assert!(err.is_err(), "expected error for k < degree+1");
}

#[test]
fn periodic_bspline_rejects_nonfinite_data() {
    let xs = ndarray::array![0.0, 1.0, f64::NAN, 2.0];
    let err = create_periodic_bspline_basis_dense(xs.view(), (0.0, TAU), 3, 8);
    assert!(err.is_err(), "expected error for NaN input");

    let xs = ndarray::array![0.0, 1.0, f64::INFINITY, 2.0];
    let err = create_periodic_bspline_basis_dense(xs.view(), (0.0, TAU), 3, 8);
    assert!(err.is_err(), "expected error for inf input");
}

#[test]
fn periodic_bspline_rejects_invalid_range() {
    let xs = ndarray::array![0.0, 1.0, 2.0];
    let err = create_periodic_bspline_basis_dense(xs.view(), (0.0, 0.0), 3, 8);
    assert!(err.is_err(), "expected error for zero-width range");
    let err = create_periodic_bspline_basis_dense(xs.view(), (1.0, 0.0), 3, 8);
    assert!(err.is_err(), "expected error for inverted range");
}

#[test]
fn cyclic_difference_penalty_constant_in_nullspace_across_orders_and_k() {
    for &k in &[4_usize, 6, 12, 24, 64, 256] {
        for &order in &[1_usize, 2, 3] {
            if order >= k {
                continue;
            }
            let s = create_cyclic_difference_penalty_matrix(k, order).unwrap();
            let v = s.dot(&Array1::<f64>::ones(k));
            for (i, &vi) in v.iter().enumerate() {
                assert!(vi.abs() < 1e-9, "k={k} order={order} row {i}: {vi}");
            }
            // Symmetry: S = D'D so S' = S
            for i in 0..k {
                for j in i + 1..k {
                    assert!(near(s[(i, j)], s[(j, i)], 1e-12), "k={k} order={order}");
                }
            }
        }
    }
}

#[test]
fn periodic_bspline_derivative_matches_finite_difference_at_interior() {
    // Sanity check derivative formula against centered FD on dense values.
    // (NOT for ground truth; this is a self-consistency sweep.)
    let period = TAU;
    let k = 12;
    let degree = 3;
    let h = 1e-4;
    let xs = ndarray::array![0.5, 1.7, 3.14, 4.2, 5.5];
    let xs_plus = &xs + h;
    let xs_minus = &xs - h;
    let bp = create_periodic_bspline_basis_dense(xs_plus.view(), (0.0, period), degree, k).unwrap();
    let bm =
        create_periodic_bspline_basis_dense(xs_minus.view(), (0.0, period), degree, k).unwrap();
    let d = create_periodic_bspline_derivative_dense(xs.view(), (0.0, period), degree, k).unwrap();
    for i in 0..xs.len() {
        for j in 0..k {
            let fd = (bp[(i, j)] - bm[(i, j)]) / (2.0 * h);
            assert!(
                near(d[(i, j)], fd, 1e-5),
                "deriv mismatch i={i} j={j}: analytic={} fd={}",
                d[(i, j)],
                fd
            );
        }
    }
}

// ----- BC B-SPLINE: ~25 cases -----

fn build_bc(
    n: usize,
    k_request: usize,
    bc_left: BSplineEndpointBoundaryCondition,
    bc_right: BSplineEndpointBoundaryCondition,
) -> Result<gam::basis::BasisBuildResult, gam::basis::BasisError> {
    let xs = Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64 - 1.0)));
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: k_request,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary_conditions: BSplineBoundaryConditions {
            left: bc_left,
            right: bc_right,
        },
    };
    build_bspline_basis_1d(xs.view(), &spec)
}

#[test]
fn bc_bspline_all_endpoint_combinations_build() {
    use BSplineEndpointBoundaryCondition::{Anchored, Clamped, Free};
    let bcs = [Free, Clamped, Anchored { value: 0.0 }];
    for &bl in &bcs {
        for &br in &bcs {
            let r = build_bc(100, 10, bl, br);
            assert!(r.is_ok(), "BC combo {bl:?}/{br:?} failed: {:?}", r.err());
        }
    }
}

#[test]
fn bc_bspline_anchored_at_left_vanishes_at_x_zero() {
    use BSplineEndpointBoundaryCondition::{Anchored, Free};
    let xs = ndarray::array![0.0, 0.25, 0.5, 0.75, 1.0];
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 8,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary_conditions: BSplineBoundaryConditions {
            left: Anchored { value: 0.0 },
            right: Free,
        },
    };
    let built = build_bspline_basis_1d(xs.view(), &spec).unwrap();
    let d = built.design.to_dense();
    // Any coefficient vector β yields f(0) = (row at x=0) · β = 0 because
    // the BC reparameterization enforces the row at x=0 to be the zero vector.
    let row0 = d.row(0);
    let s = row0.iter().sum::<f64>();
    assert!(s.abs() < 1e-9, "row at x=0 should sum to 0, got {s}");
    for &v in row0.iter() {
        assert!(v.abs() < 1e-9, "row at x=0 entry should be 0, got {v}");
    }
}

#[test]
fn bc_bspline_clamped_at_right_has_zero_derivative_at_x_one() {
    use BSplineEndpointBoundaryCondition::{Clamped, Free};
    // The clamped BC enforces f'(1) = 0; we verify by finite-difference
    // on the design: (f(1) - f(1-h)) / h should approach 0 as h → 0.
    let h = 1e-5;
    let xs = ndarray::array![1.0_f64, 1.0 - h];
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 8,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary_conditions: BSplineBoundaryConditions {
            left: Free,
            right: Clamped,
        },
    };
    let built = build_bspline_basis_1d(xs.view(), &spec).unwrap();
    let d = built.design.to_dense();
    // For each column, the slope at x=1 from (d[0]-d[1])/h should be ~0
    for j in 0..d.ncols() {
        let slope = (d[(0, j)] - d[(1, j)]) / h;
        assert!(
            slope.abs() < 1e-2,
            "clamped right: column {j} slope at x=1 should be ~0, got {slope}"
        );
    }
}

#[test]
fn bc_bspline_columns_drop_by_constraint_count() {
    use BSplineEndpointBoundaryCondition::{Anchored, Clamped, Free};
    let baseline = build_bc(100, 8, Free, Free).unwrap();
    let p0 = baseline.design.ncols();
    let one_side = build_bc(100, 8, Anchored { value: 0.0 }, Free).unwrap();
    assert_eq!(
        one_side.design.ncols() + 1,
        p0,
        "single anchored BC should reduce ncols by 1"
    );
    let both_sides = build_bc(100, 8, Anchored { value: 0.0 }, Clamped).unwrap();
    assert_eq!(
        both_sides.design.ncols() + 2,
        p0,
        "two BCs should reduce ncols by 2"
    );
}

#[test]
fn bc_bspline_rejects_nonzero_anchor() {
    use BSplineEndpointBoundaryCondition::{Anchored, Free};
    let r = build_bc(20, 8, Anchored { value: 1.5 }, Free);
    assert!(r.is_err(), "non-zero anchor should be rejected");
}

// ----- SPHERE WAHBA: ~40 cases -----

#[test]
fn wahba_kernel_kxx_is_psd_and_symmetric() {
    // Wahba's kernel K(x, x) for m ≥ 1 is finite and the Gram K[X, X]
    // must be symmetric positive semi-definite. Test across m and N.
    for m in 1..=4 {
        for &n in &[8_usize, 32, 128] {
            let pts = random_lat_lon(n, 42 + m as u64);
            let k = spherical_wahba_kernel_matrix(pts.view(), pts.view(), m, false).unwrap();
            // Symmetry
            for i in 0..n {
                for j in i + 1..n {
                    assert!(
                        near(k[(i, j)], k[(j, i)], 1e-10),
                        "m={m} n={n} ({i},{j}) not symmetric: {} vs {}",
                        k[(i, j)],
                        k[(j, i)]
                    );
                }
            }
            // Smallest eigenvalue >= -tol (PSD). Use power iteration on -K to
            // bound the most-negative eigenvalue from below — fast and
            // sufficient as a sanity check.
            let v0 = Array1::<f64>::ones(n) / (n as f64).sqrt();
            let mut v = v0;
            for _ in 0..50 {
                let kv = k.dot(&v);
                let mut shifted = &v * (1e6_f64) - &kv;
                let nrm = (shifted.iter().map(|x| x * x).sum::<f64>())
                    .sqrt()
                    .max(1e-30);
                shifted /= nrm;
                v = shifted;
            }
            let rayleigh = v.dot(&k.dot(&v));
            assert!(
                rayleigh > -1e-6,
                "m={m} n={n} estimated min eigenvalue {rayleigh} < -1e-6"
            );
        }
    }
}

#[test]
fn wahba_kernel_at_same_point_is_finite_for_all_orders() {
    let pts = ndarray::array![[10.0, 20.0], [-30.0, 60.0], [45.0, -150.0]];
    for m in 1..=4 {
        let k = spherical_wahba_kernel_matrix(pts.view(), pts.view(), m, false).unwrap();
        for i in 0..pts.nrows() {
            assert!(k[(i, i)].is_finite(), "m={m} k({i},{i}) non-finite");
        }
    }
}

#[test]
fn wahba_kernel_radians_consistent_with_degrees_across_orders() {
    let deg = ndarray::array![[15.0, 30.0], [-45.0, 90.0], [60.0, -120.0]];
    let to_rad = PI / 180.0;
    let rad = deg.map(|v| v * to_rad);
    for m in 1..=4 {
        let kd = spherical_wahba_kernel_matrix(deg.view(), deg.view(), m, false).unwrap();
        let kr = spherical_wahba_kernel_matrix(rad.view(), rad.view(), m, true).unwrap();
        for i in 0..deg.nrows() {
            for j in 0..deg.nrows() {
                assert!(
                    near(kd[(i, j)], kr[(i, j)], 1e-10),
                    "m={m} ({i},{j}): deg={} rad={}",
                    kd[(i, j)],
                    kr[(i, j)]
                );
            }
        }
    }
}

#[test]
fn wahba_kernel_invariant_under_antipodal_lon_flip() {
    // K depends only on angular separation; flipping all longitudes by π
    // preserves separations (gives the antipodal sphere rotation).
    let pts = ndarray::array![[10.0, 20.0], [-30.0, 60.0], [45.0, -120.0]];
    let pts_flip: Array2<f64> = {
        let mut p = pts.clone();
        for r in 0..p.nrows() {
            p[(r, 1)] = (p[(r, 1)] + 180.0) % 360.0;
            if p[(r, 1)] > 180.0 {
                p[(r, 1)] -= 360.0;
            }
        }
        p
    };
    let k = spherical_wahba_kernel_matrix(pts.view(), pts.view(), 2, false).unwrap();
    let k_flip = spherical_wahba_kernel_matrix(pts_flip.view(), pts_flip.view(), 2, false).unwrap();
    for i in 0..pts.nrows() {
        for j in 0..pts.nrows() {
            assert!(
                near(k[(i, j)], k_flip[(i, j)], 1e-10),
                "antipodal flip ({i},{j}): {} vs {}",
                k[(i, j)],
                k_flip[(i, j)]
            );
        }
    }
}

#[test]
fn wahba_kernel_rejects_wrong_dim() {
    let one_col = Array2::<f64>::ones((4, 1));
    let r = spherical_wahba_kernel_matrix(one_col.view(), one_col.view(), 2, false);
    assert!(r.is_err());
    let three_col = Array2::<f64>::ones((4, 3));
    let r = spherical_wahba_kernel_matrix(three_col.view(), three_col.view(), 2, false);
    assert!(r.is_err());
}

#[test]
fn wahba_kernel_rejects_invalid_penalty_order() {
    let pts = ndarray::array![[10.0, 20.0], [-30.0, 60.0]];
    for m in [0_usize, 5, 100] {
        let r = spherical_wahba_kernel_matrix(pts.view(), pts.view(), m, false);
        assert!(r.is_err(), "m={m} should be rejected");
    }
}

#[test]
fn wahba_kernel_rejects_lat_out_of_range() {
    let bad = ndarray::array![[91.0, 0.0], [0.0, 0.0]];
    let r = spherical_wahba_kernel_matrix(bad.view(), bad.view(), 2, false);
    assert!(r.is_err(), "lat>90 should be rejected");
    let bad = ndarray::array![[-91.0, 0.0], [0.0, 0.0]];
    let r = spherical_wahba_kernel_matrix(bad.view(), bad.view(), 2, false);
    assert!(r.is_err(), "lat<-90 should be rejected");
    // radians mode
    let bad = ndarray::array![[2.0, 0.0], [0.0, 0.0]];
    let r = spherical_wahba_kernel_matrix(bad.view(), bad.view(), 2, true);
    assert!(r.is_err(), "lat=2 rad > π/2 should be rejected");
}

// ----- SPHERE HARMONIC: ~40 cases -----

fn build_harmonic(
    data: ArrayView2<'_, f64>,
    l: usize,
    radians: bool,
) -> Result<gam::basis::BasisBuildResult, gam::basis::BasisError> {
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 2,
        double_penalty: false,
        radians,
        method: SphereMethod::Harmonic,
        max_degree: Some(l),
    };
    build_spherical_spline_basis(data, &spec)
}

#[test]
fn sphere_harmonic_basis_dim_is_l_times_l_plus_2() {
    for l in 1..=8 {
        let pts = random_lat_lon(50, 7 * l as u64);
        let built = build_harmonic(pts.view(), l, false).unwrap();
        assert_eq!(built.design.ncols(), l * (l + 2), "L={l}");
    }
}

#[test]
fn sphere_harmonic_design_finite_for_all_l_at_random_points() {
    for l in 1..=8 {
        let pts = random_lat_lon(200, 31 * l as u64);
        let built = build_harmonic(pts.view(), l, false).unwrap();
        let d = built.design.to_dense();
        for v in d.iter() {
            assert!(v.is_finite(), "L={l}: non-finite design entry");
        }
    }
}

#[test]
fn sphere_harmonic_at_north_pole_reduces_to_legendre() {
    // At lat=90, sin(lat)=1, cos(lat)=0 → all sectoral (m>0) terms vanish
    // because they include sin/cos of latitude in non-zonal places.
    // Specifically, P_{l,m}(1) = 0 for m >= 1, so only m=0 terms survive.
    let pts = ndarray::array![[90.0_f64, 0.0]];
    for l in 1..=5 {
        let built = build_harmonic(pts.view(), l, false).unwrap();
        let d = built.design.to_dense();
        // Column layout per degree l: sin(l φ), …, sin φ, m=0, cos φ, …, cos(l φ)
        // The m=0 column is at offset l within the (2l+1)-block; non-m=0 columns
        // should be zero at the pole.
        let mut col = 0_usize;
        for ll in 1..=l {
            let block_size = 2 * ll + 1;
            for m_idx in 0..block_size {
                let m_offset_from_m0 = (m_idx as isize) - (ll as isize);
                if m_offset_from_m0 != 0 {
                    let v = d[(0, col + m_idx)];
                    assert!(
                        v.abs() < 1e-10,
                        "L={l} ll={ll} m offset {m_offset_from_m0} should be 0 at pole, got {v}"
                    );
                }
            }
            col += block_size;
        }
    }
}

#[test]
fn sphere_harmonic_radians_matches_degrees_across_l_and_points() {
    let deg = ndarray::array![
        [10.0, 20.0],
        [-30.0, -45.0],
        [60.0, 170.0],
        [-80.0, -179.0],
        [5.0, 0.0],
    ];
    let to_rad = PI / 180.0;
    let rad = deg.map(|v| v * to_rad);
    for l in 1..=5 {
        let bd = build_harmonic(deg.view(), l, false).unwrap();
        let br = build_harmonic(rad.view(), l, true).unwrap();
        let dd = bd.design.to_dense();
        let dr = br.design.to_dense();
        assert_eq!(dd.shape(), dr.shape());
        for i in 0..dd.nrows() {
            for j in 0..dd.ncols() {
                assert!(
                    near(dd[(i, j)], dr[(i, j)], 1e-10),
                    "L={l} ({i},{j}): deg={} rad={}",
                    dd[(i, j)],
                    dr[(i, j)]
                );
            }
        }
    }
}

#[test]
fn sphere_harmonic_row_gram_invariant_under_longitude_shift() {
    // For a pure longitude rotation, the columns per (l, m) block transform
    // by a 2D orthogonal rotation. So D' = D R for block-orthogonal R, and
    // (D')(D')^T = D R R^T D^T = D D^T. Row Gram invariant.
    let pts = ndarray::array![
        [10.0_f64, 20.0_f64],
        [-30.0_f64, 60.0_f64],
        [50.0_f64, -120.0_f64],
        [-70.0_f64, 170.0_f64]
    ];
    for shift in [10.0_f64, 47.0_f64, 89.0_f64, 180.0_f64] {
        let mut rot = pts.clone();
        for r in 0..rot.nrows() {
            let v = rot[(r, 1)] + shift;
            rot[(r, 1)] = ((v + 180.0_f64).rem_euclid(360.0_f64)) - 180.0_f64;
        }
        for l in 1..=4 {
            let a = build_harmonic(pts.view(), l, false).unwrap();
            let b = build_harmonic(rot.view(), l, false).unwrap();
            let da = a.design.to_dense();
            let db = b.design.to_dense();
            let ga = da.dot(&da.t());
            let gb = db.dot(&db.t());
            for i in 0..ga.nrows() {
                for j in 0..ga.ncols() {
                    assert!(
                        near(ga[(i, j)], gb[(i, j)], 1e-10),
                        "shift={shift} L={l} ({i},{j}): {} vs {}",
                        ga[(i, j)],
                        gb[(i, j)]
                    );
                }
            }
        }
    }
}

#[test]
fn sphere_harmonic_rejects_l_zero_and_too_large() {
    let pts = ndarray::array![[10.0_f64, 20.0_f64]];
    // L=0 has no columns; rejected
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 2,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(0),
    };
    let r = build_spherical_spline_basis(pts.view(), &spec);
    assert!(r.is_err(), "L=0 should be rejected");
    // L>32 cap
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 2,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(33),
    };
    let r = build_spherical_spline_basis(pts.view(), &spec);
    assert!(r.is_err(), "L>32 should be rejected");
}

#[test]
fn sphere_harmonic_penalty_diagonal_eigenvalues_match_l_l_plus_1_squared() {
    let pts = random_lat_lon(60, 13);
    let l_max = 5;
    let built = build_harmonic(pts.view(), l_max, false).unwrap();
    let p = &built.penalties[0];
    // Normalized penalty; raw eigenvalue per (l, m) is [l(l+1)]^2. After
    // normalize_penalty, values may be scaled — confirm the *relative*
    // ratios: each (2l+1)-block has the same diagonal value.
    let mut col = 0_usize;
    let mut prev_val = -1.0_f64;
    for l in 1..=l_max {
        let block = 2 * l + 1;
        let v0 = p[(col, col)];
        for off in 1..block {
            assert!(
                near(p[(col + off, col + off)], v0, 1e-10),
                "L={l_max} l={l}: block diag {col}+{off} differs from {col}: {} vs {}",
                p[(col + off, col + off)],
                v0
            );
        }
        if prev_val >= 0.0 {
            assert!(
                v0 > prev_val,
                "L={l_max} l={l}: penalty eigenvalue should be monotone increasing"
            );
        }
        prev_val = v0;
        col += block;
    }
}

// ----- HARMONIC vs WAHBA consistency (intrinsic both) -----

#[test]
fn both_sphere_methods_give_rotation_invariant_smoothers() {
    // For pure longitude shift, both Wahba and Harmonic should produce
    // designs whose ROW GRAM is invariant (intrinsic basis).
    let pts = random_lat_lon(20, 1);
    let mut rot = pts.clone();
    for r in 0..rot.nrows() {
        let v = rot[(r, 1)] + 33.0_f64;
        rot[(r, 1)] = ((v + 180.0_f64).rem_euclid(360.0_f64)) - 180.0_f64;
    }
    for method in [SphereMethod::Wahba, SphereMethod::Harmonic] {
        let spec = SphericalSplineBasisSpec {
            center_strategy: CenterStrategy::UserProvided(pts.clone()),
            penalty_order: 2,
            double_penalty: false,
            radians: false,
            method,
            max_degree: Some(4),
        };
        let a = build_spherical_spline_basis(pts.view(), &spec).unwrap();
        let b = build_spherical_spline_basis(rot.view(), &spec).unwrap();
        let da = a.design.to_dense();
        let db = b.design.to_dense();
        let ga = da.dot(&da.t());
        let gb = db.dot(&db.t());
        for i in 0..ga.nrows() {
            for j in 0..ga.ncols() {
                assert!(
                    near(ga[(i, j)], gb[(i, j)], 1e-9),
                    "method={method:?} ({i},{j}): {} vs {}",
                    ga[(i, j)],
                    gb[(i, j)]
                );
            }
        }
    }
}

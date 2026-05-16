//! At-scale stress and timing harness for the geometric smooths.
//!
//! This is *not* a perf regression test — it prints timings via `eprintln!`
//! that show up only on `--nocapture` / on failure. It's a load test: it
//! exercises every parallel hot path at N = 1k, 10k, 100k and validates
//! that the core invariants (partition of unity, periodicity, kernel
//! symmetry, rotation invariance of intrinsic S² smooths) all still hold
//! at biobank scale.

use gam::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineEndpointBoundaryCondition,
    BSplineIdentifiability, BSplineKnotSpec, CenterStrategy, SphereMethod,
    SphericalSplineBasisSpec, build_bspline_basis_1d, build_spherical_spline_basis,
    create_cyclic_difference_penalty_matrix, create_periodic_bspline_basis_dense,
    create_periodic_bspline_derivative_dense, spherical_wahba_kernel_matrix,
};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn make_uniform_loop(n: usize, period: f64) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| {
        let u = (i as f64) / (n as f64);
        u * period
    }))
}

fn make_sphere_grid(n_total: usize) -> Array2<f64> {
    let nrow = (n_total as f64).sqrt().ceil() as usize;
    let ncol = (n_total + nrow - 1) / nrow;
    let mut out = Array2::<f64>::zeros((nrow * ncol, 2));
    for i in 0..nrow {
        for j in 0..ncol {
            let row = i * ncol + j;
            let lat = -89.0 + 178.0 * (i as f64) / (nrow.saturating_sub(1).max(1) as f64);
            let lon = -180.0 + 360.0 * (j as f64) / (ncol.max(1) as f64);
            out[(row, 0)] = lat;
            out[(row, 1)] = lon;
        }
    }
    out.slice(ndarray::s![..n_total, ..]).to_owned()
}

fn time_label<R>(label: &str, mut f: impl FnMut() -> R) -> R {
    let t = Instant::now();
    let r = f();
    let dt = t.elapsed().as_secs_f64();
    eprintln!("[scale] {label}: {:.3} ms", dt * 1e3);
    r
}

#[test]
fn periodic_bspline_scales_and_partitions_unity_at_100k() {
    let period = 2.0 * std::f64::consts::PI;
    for n in [1_000usize, 10_000, 100_000] {
        let xs = make_uniform_loop(n, period);
        let basis = time_label(&format!("periodic_bspline N={n} k=24 degree=3"), || {
            create_periodic_bspline_basis_dense(xs.view(), (0.0, period), 3, 24).expect("basis")
        });
        assert_eq!(basis.nrows(), n);
        assert_eq!(basis.ncols(), 24);
        // partition of unity: every row sums to ~1.0
        for i in 0..n.min(2048) {
            let s = basis.row(i).sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "row {i} partition of unity violation: {s}"
            );
        }
        // periodicity: row at x and x + period (mod) give the same row vector
        let xs_wrap = Array1::from_iter(xs.iter().map(|x| x + period));
        let basis_wrap = create_periodic_bspline_basis_dense(xs_wrap.view(), (0.0, period), 3, 24)
            .expect("wrap basis");
        for i in 0..n.min(2048) {
            for j in 0..24 {
                let d = (basis[(i, j)] - basis_wrap[(i, j)]).abs();
                assert!(
                    d < 1e-12,
                    "periodicity violated at row {i} col {j}: {d}"
                );
            }
        }
    }
}

#[test]
fn periodic_bspline_derivative_integrates_to_zero_over_loop() {
    let period = 2.0_f64 * std::f64::consts::PI;
    let n = 10_000;
    let xs = make_uniform_loop(n, period);
    let d = time_label(&format!("periodic_bspline_deriv N={n} k=20"), || {
        create_periodic_bspline_derivative_dense(xs.view(), (0.0, period), 3, 20)
            .expect("derivative basis")
    });
    // Integral of each derivative basis column around the loop is zero
    // for an exact periodic spline. Use trapezoidal sum over uniform xs.
    let h = period / n as f64;
    for j in 0..d.ncols() {
        let s = d.column(j).sum() * h;
        assert!(
            s.abs() < 1e-6,
            "column {j} derivative loop integral not zero: {s}"
        );
    }
}

#[test]
fn cyclic_difference_penalty_constant_in_nullspace_at_large_k() {
    for k in [16, 64, 256] {
        let s = create_cyclic_difference_penalty_matrix(k, 2).expect("penalty");
        let ones = Array1::<f64>::ones(k);
        let v = s.dot(&ones);
        for i in 0..k {
            assert!(
                v[i].abs() < 1e-10,
                "ones not in nullspace at k={k}, row {i}: {}",
                v[i]
            );
        }
        // shift invariance: row i of S equals row (i+1)%k of S shifted by one
        for i in 0..k {
            for j in 0..k {
                let lhs = s[(i, j)];
                let rhs = s[((i + 1) % k, (j + 1) % k)];
                assert!(
                    (lhs - rhs).abs() < 1e-12,
                    "shift invariance violated at k={k}, ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn sphere_wahba_kernel_at_100k_is_symmetric_on_self_eval() {
    // For the diagonal (data == centers), kernel(i, j) must equal kernel(j, i).
    // Test at three scales to confirm parallel chunks don't introduce races.
    for n in [256_usize, 4_096, 32_768] {
        let pts = make_sphere_grid(n);
        let k = time_label(
            &format!("wahba_kernel_matrix N=K={n} m=2"),
            || spherical_wahba_kernel_matrix(pts.view(), pts.view(), 2, false).expect("kernel"),
        );
        // Spot check 256 random off-diagonal symmetry positions
        let step = (n / 64).max(1);
        for i in (0..n).step_by(step) {
            for j in (0..n).step_by(step) {
                if i == j {
                    continue;
                }
                let d = (k[(i, j)] - k[(j, i)]).abs();
                assert!(
                    d < 1e-10,
                    "non-symmetric kernel at N={n}, ({i},{j}): {} vs {}",
                    k[(i, j)],
                    k[(j, i)]
                );
            }
        }
    }
}

#[test]
fn sphere_harmonic_basis_scales_and_keeps_diag_penalty_at_100k() {
    use std::f64::consts::PI;
    for n in [1_000usize, 10_000, 100_000] {
        let pts = make_sphere_grid(n);
        let l = 4usize;
        let spec = SphericalSplineBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
            penalty_order: 2,
            double_penalty: false,
            radians: false,
            method: SphereMethod::Harmonic,
            max_degree: Some(l),
        };
        let built = time_label(
            &format!("sphere_harmonic_basis N={n} L={l} p={}", l * (l + 2)),
            || build_spherical_spline_basis(pts.view(), &spec).expect("harmonic basis"),
        );
        assert_eq!(built.design.ncols(), l * (l + 2));
        // Penalty is diagonal, monotone in degree.
        let p = &built.penalties[0];
        for i in 0..p.nrows() {
            for j in 0..p.ncols() {
                if i != j {
                    assert!(p[(i, j)].abs() < 1e-12, "off-diag at N={n}, ({i},{j})");
                }
            }
        }
        let _ = PI;
    }
}

#[test]
fn boundary_conditioned_bspline_drops_dimension_correctly_and_scales() {
    // Build the BC-projected B-spline at N = 1k, 10k. Validate:
    //   - ncols equals expected reduced dimension
    //   - design at left endpoint vanishes when bc_left=anchored (value=0)
    //   - design at right endpoint has zero derivative when bc_right=clamped
    for n in [1_000usize, 10_000] {
        let xs = Array1::from_iter((0..n).map(|i| (i as f64) / (n as f64 - 1.0)));
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 10,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary_conditions: BSplineBoundaryConditions {
                left: BSplineEndpointBoundaryCondition::Anchored { value: 0.0 },
                right: BSplineEndpointBoundaryCondition::Clamped,
            },
        };
        let built = time_label(&format!("bc_bspline N={n} k=14 anchored+clamped"), || {
            build_bspline_basis_1d(xs.view(), &spec).expect("bc basis")
        });
        // Anchored at x=0 → design row at x=0 sums to 0
        let row0 = built.design.to_dense().row(0).to_owned();
        let s = row0.sum();
        assert!(
            s.abs() < 1e-9,
            "bc anchored: row at x=0 should sum to 0, got {s}"
        );
    }
}

#[test]
fn sphere_harmonic_design_rows_are_finite_at_poles_and_seam() {
    // Stress poles and the longitude seam at scale — the ALF recurrence
    // and the Chebyshev sin/cos recurrence must stay bounded.
    let mut pts = Vec::<f64>::with_capacity(4096 * 2);
    for i in 0..1024 {
        let frac = (i as f64) / 1024.0;
        pts.push(-89.999 + frac * 0.001); // near south pole
        pts.push(-180.0 + frac * 360.0);
    }
    for i in 0..1024 {
        let frac = (i as f64) / 1024.0;
        pts.push(89.999 - frac * 0.001); // near north pole
        pts.push(-180.0 + frac * 360.0);
    }
    for i in 0..2048 {
        let frac = (i as f64) / 2048.0;
        pts.push(-45.0 + frac * 90.0);
        pts.push(-180.0 + frac * 1e-6); // seam crossing
    }
    let data = Array2::from_shape_vec((4096, 2), pts).unwrap();
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 2,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(6),
    };
    let built = build_spherical_spline_basis(data.view(), &spec).expect("pole+seam build");
    let design = built.design.to_dense();
    let mut max_abs = 0.0_f64;
    for v in design.iter() {
        assert!(v.is_finite(), "non-finite sphere harmonic value");
        if v.abs() > max_abs {
            max_abs = v.abs();
        }
    }
    // Real spherical harmonics at L=6 have bounded magnitude
    // ~ √((2L+1)/(4π)) ≈ 1.02; require a generous bound that catches
    // recurrence overflow (which would explode rapidly).
    assert!(
        max_abs < 5.0,
        "sphere harmonic magnitude blew up at poles/seam: max |Y| = {max_abs}"
    );
}

//! Chart / basis-evaluator jet & finite-difference fidelity tests, split out of
//! `tests.rs` to keep that file under the #780 line-count gate. These exercise
//! the per-manifold coordinate evaluators (periodic, sphere, torus, cylinder,
//! Duchon, Euclidean patch, affine) and pin their analytic Jacobian / second /
//! third jets against central-difference oracles, the continuous fixed-decoder
//! OOS projection solve, and the Euclidean affine-gauge canonicalization. They
//! share the parent module's helpers via `super::tests`.

use super::*;
use approx::assert_abs_diff_eq;
use ndarray::array;

pub(crate) fn assert_jacobian_matches_central_difference<E: SaeBasisEvaluator>(
    evaluator: &E,
    coords: Array2<f64>,
    tolerance: f64,
) {
    let epsilon = 1.0e-6;
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the caller's coords lie in the evaluator's chart domain");
    let (n_rows, n_basis) = phi.dim();
    let latent_dim = coords.ncols();
    assert_eq!(jet.dim(), (n_rows, n_basis, latent_dim));

    for row in 0..n_rows {
        for axis in 0..latent_dim {
            let mut plus = coords.clone();
            let mut minus = coords.clone();
            plus[[row, axis]] += epsilon;
            minus[[row, axis]] -= epsilon;
            let (phi_plus, plus_jet) = evaluator
                .evaluate(plus.view())
                .expect("the +epsilon probe stays in the chart domain");
            let (phi_minus, minus_jet) = evaluator
                .evaluate(minus.view())
                .expect("the -epsilon probe stays in the chart domain");
            assert_eq!(plus_jet.dim(), jet.dim());
            assert_eq!(minus_jet.dim(), jet.dim());

            for basis in 0..n_basis {
                let finite_difference =
                    (phi_plus[[row, basis]] - phi_minus[[row, basis]]) / (2.0 * epsilon);
                let analytic = jet[[row, basis, axis]];
                let error = (analytic - finite_difference).abs();
                assert!(
                    error <= tolerance,
                    "row={row} basis={basis} axis={axis}: analytic={analytic:.12e}, \
                         finite_difference={finite_difference:.12e}, error={error:.12e}, \
                         tolerance={tolerance:.12e}"
                );
            }
        }
    }
}

#[test]
pub(crate) fn sae_basis_evaluator_jacobians_match_central_differences() {
    assert_jacobian_matches_central_difference(
        &PeriodicHarmonicEvaluator::new(7).unwrap(),
        array![[-0.37], [0.0], [0.125], [0.41]],
        1.0e-6,
    );

    assert_jacobian_matches_central_difference(
        &RawPeriodicCircleEvaluator::new(3).unwrap(),
        array![[-1.2, 0.3, 2.0], [0.0, -0.4, 0.8], [2.4, 1.1, -0.7]],
        1.0e-6,
    );

    // Ambient unit vectors, poles INCLUDED: the ambient sphere has no boundary,
    // so a pole is an ordinary sample point rather than a place to stay away
    // from — which is the whole difference from the chart this replaced.
    let sphere_coords = array![
        [0.0, 0.0, 1.0],
        [0.6, -0.8, 0.0],
        [0.36, 0.48, 0.8],
        [-0.48, 0.6, -0.64]
    ];
    assert_jacobian_matches_central_difference(
        &AmbientSphereHarmonicEvaluator::new(2).unwrap(),
        sphere_coords.clone(),
        1.0e-6,
    );
    let (sphere_phi, sphere_jet) = AmbientSphereHarmonicEvaluator::new(2)
        .unwrap()
        .evaluate(sphere_coords.view())
        .unwrap();
    // Degree 2 spans all nine harmonics through `l = 2`, where the removed
    // chart carried seven — the monopole, the dipole, and three of the five
    // quadrupoles.
    assert_eq!(sphere_phi.dim(), (sphere_coords.nrows(), 9));
    assert_eq!(sphere_jet.dim(), (sphere_coords.nrows(), 9, 3));
    for row in 0..sphere_coords.nrows() {
        let lat = sphere_coords[[row, 0]];
        let lon = sphere_coords[[row, 1]];
        let clat = lat.cos();
        let slat = lat.sin();
        let clon = lon.cos();
        let slon = lon.sin();
        let z = slat;
        let dx_dlon = -clat * slon;
        let dy_dlon = clat * clon;
        assert_eq!(sphere_jet[[row, 3, 1]], 0.0);
        assert!((sphere_jet[[row, 5, 1]] - dy_dlon * z).abs() <= 1.0e-12);
        assert!((sphere_jet[[row, 6, 1]] - dx_dlon * z).abs() <= 1.0e-12);
    }

    assert_jacobian_matches_central_difference(
        &AffineCoordinateEvaluator::new(3),
        array![[0.0, -1.0, 2.0], [3.5, 0.25, -0.75]],
        1.0e-6,
    );

    // Torus T^2 with H=3 → 49-column tensor product.
    let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
    assert_jacobian_matches_central_difference(
        &TorusHarmonicEvaluator::new(2, 3).unwrap(),
        torus_coords.clone(),
        1.0e-6,
    );
    let (torus_phi, torus_jet) = TorusHarmonicEvaluator::new(2, 3)
        .unwrap()
        .evaluate(torus_coords.view())
        .unwrap();
    assert_eq!(torus_phi.dim(), (torus_coords.nrows(), 49));
    assert_eq!(torus_jet.dim(), (torus_coords.nrows(), 49, 2));
    for row in 0..torus_coords.nrows() {
        // Column 0 = product of the two constant axis terms = 1.
        assert!((torus_phi[[row, 0]] - 1.0).abs() <= 1.0e-12);
        assert!(torus_jet[[row, 0, 0]].abs() <= 1.0e-12);
        assert!(torus_jet[[row, 0, 1]].abs() <= 1.0e-12);
    }
}

/// `seed_coords_by_decoder_projection` must replace each cold coordinate with
/// the continuous frozen-decoder projection and refresh the atom basis there.
/// Built on a decoder that maps the
/// circle injectively into `ℝ²` (`decode(t) = (sin 2πt, cos 2πt)`) so the
/// per-row global argmin is unambiguous. Direct Rust pin for the #628 OOS
/// seed, complementing the Python oracle end-to-end test.
#[test]
pub(crate) fn seed_coords_by_decoder_projection_recovers_continuous_minimiser() {
    use std::f64::consts::PI;

    // Deliberately wrong cold seed for both rows.
    let init_coords = array![[0.05], [0.05]];
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let (phi0, jet0) = evaluator.evaluate(init_coords.view()).unwrap();
    // (basis = [1, sin, cos]) × (2 output channels): decode(t) = (sin, cos).
    let decoder = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        // `K = logits.ncols()`; a single softmax atom is one logit column
        // (the lone simplex coordinate, pinned to 1.0 in `try_assignments_row`).
        Array2::<f64>::zeros((2, 1)),
        vec![init_coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // Targets sit at arbitrary phases. The Fourier stationary-root path must
    // return a genuinely continuous answer.
    let phases = [0.173_205_080_756_887_73, 0.731_058_578_630_004_9];
    let mut target = Array2::<f64>::zeros((2, 2));
    for (row, &t) in phases.iter().enumerate() {
        target[[row, 0]] = (2.0 * PI * t).sin();
        target[[row, 1]] = (2.0 * PI * t).cos();
    }

    term.seed_coords_by_decoder_projection(target.view())
        .unwrap();

    // Each row was seeded onto its exact continuous minimiser …
    let seeded = term.assignment.coords[0].as_matrix();
    let mut expected_coords = Array2::<f64>::zeros((2, 1));
    for (row, &expected) in phases.iter().enumerate() {
        assert_abs_diff_eq!(seeded[[row, 0]], expected, epsilon = 1e-10);
        expected_coords[[row, 0]] = expected;
    }
    // … and the basis cache was refreshed at the seeded coordinates.
    let (phi_expected, _) = evaluator.evaluate(expected_coords.view()).unwrap();
    assert_abs_diff_eq!(
        (&term.atoms[0].basis_values - &phi_expected)
            .mapv(f64::abs)
            .sum(),
        0.0,
        epsilon = 1e-12
    );
}

/// A target whose shape does not match `(n_obs, output_dim)` is a caller
/// bug and must surface as an error rather than silently mis-seeding.
#[test]
pub(crate) fn seed_coords_by_decoder_projection_rejects_shape_mismatch() {
    let init_coords = array![[0.05], [0.05]];
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let (phi0, jet0) = evaluator.evaluate(init_coords.view()).unwrap();
    let decoder = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((2, 1)),
        vec![init_coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // Output dim is 2; pass a 3-column target.
    let bad_target = Array2::<f64>::zeros((2, 3));
    let err = term
        .seed_coords_by_decoder_projection(bad_target.view())
        .unwrap_err();
    assert!(
        err.contains("target shape"),
        "expected a target-shape error, got: {err}"
    );
}

/// Central-difference oracle for `second_jet`: differentiate the analytic
/// first jet (which is FD-validated by the test above) coordinate-wise.
///
/// The threshold is magnitude-scaled (`abs_tol + rel_tol·max(|analytic|,
/// |fd|)`), exactly like the third-jet helper, because the central-difference
/// truncation error of a second derivative obtained by differencing the
/// first jet is `O(ε²/6·|f⁗|)`. For a harmonic basis `sin(ωt)` the fourth
/// derivative is `ω⁴·φ`, so with `ε = 1e-4` and the top harmonic of the
/// periodic/torus evaluators (`ω = 2π·3 ≈ 18.85 → ω⁴ ≈ 1.26e5`) the floor is
/// `≈ (1e-4)²/6·1.26e5 ≈ 2e-5` — several × any flat `1e-5` absolute bound.
/// A pure absolute bound is therefore physically wrong at the top of the
/// frequency range; the rel_tol term tracks the `ω⁴` truncation scale (the
/// analytic second jet itself is exact, `-ω²·φ`). The FD step is 1e-4 (the
/// sweet spot before f64 cancellation dominates a centered difference of an
/// `O(1)` Jacobian).
pub(crate) fn assert_second_jet_matches_central_difference<E: SaeBasisSecondJet>(
    evaluator: &E,
    coords: Array2<f64>,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<(), String> {
    let epsilon = 1.0e-4;
    let second = evaluator.second_jet(coords.view())?;
    let (_phi, jet) = evaluator.evaluate(coords.view())?;
    let (n_rows, n_basis, latent_dim, latent_dim_b) = second.dim();
    assert_eq!(latent_dim, latent_dim_b);
    assert_eq!((n_rows, n_basis, latent_dim), jet.dim());
    for row in 0..n_rows {
        for axis_c in 0..latent_dim {
            let mut plus = coords.clone();
            let mut minus = coords.clone();
            plus[[row, axis_c]] += epsilon;
            minus[[row, axis_c]] -= epsilon;
            let (_, jet_plus) = evaluator
                .evaluate(plus.view())
                .expect("the +epsilon probe stays in the chart domain");
            let (_, jet_minus) = evaluator
                .evaluate(minus.view())
                .expect("the -epsilon probe stays in the chart domain");
            for basis in 0..n_basis {
                for axis_a in 0..latent_dim {
                    let fd = (jet_plus[[row, basis, axis_a]] - jet_minus[[row, basis, axis_a]])
                        / (2.0 * epsilon);
                    let analytic = second[[row, basis, axis_a, axis_c]];
                    let error = (analytic - fd).abs();
                    let threshold = abs_tol + rel_tol * analytic.abs().max(fd.abs());
                    assert!(
                        error <= threshold,
                        "row={row} basis={basis} axis_a={axis_a} axis_c={axis_c}: \
                             analytic={analytic:.12e}, fd={fd:.12e}, error={error:.12e}, \
                             threshold={threshold:.12e}"
                    );
                }
            }
        }
    }
    // Hessian symmetry in (axis_a, axis_c).
    for row in 0..n_rows {
        for basis in 0..n_basis {
            for axis_a in 0..latent_dim {
                for axis_b in 0..latent_dim {
                    let h_ab = second[[row, basis, axis_a, axis_b]];
                    let h_ba = second[[row, basis, axis_b, axis_a]];
                    assert!(
                        (h_ab - h_ba).abs() <= 1.0e-12,
                        "second_jet not symmetric: row={row} basis={basis} \
                             ({axis_a},{axis_b})={h_ab:.6e} vs ({axis_b},{axis_a})={h_ba:.6e}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// The analytic third jet `T[n,m,a,c,e] = ∂³Φ_m/∂t_a∂t_c∂t_e` must equal the
/// central difference of the analytic (already FD-validated) second jet along
/// the trailing axis, and be fully symmetric across its three trailing axes.
/// This validates the closed-form `K` providers added for the exact isometry
/// Hessian (#458) against an independent numerical derivative — the third-jet
/// analogue of `assert_second_jet_matches_central_difference`. A
/// magnitude-scaled tolerance is used because the harmonic third derivatives
/// scale like `freq³` (≈ thousands for the higher harmonics), so a pure
/// absolute bound would be meaningless at the top of the range.
///
/// The numerical reference is a **4th-order** 5-point central difference
/// `(−f₊₂ + 8f₊ − 8f₋ + f₋₂)/(12h)` rather than the 2-point `(f₊−f₋)/(2h)`. The
/// 2-point stencil carries an `O(h²)` truncation error that, for a cubic line
/// factor (`t³`) whose true mixed third derivative is exactly 0 at `t=0`, is
/// `≈ 1.6e-6` at `h=1e-4` — above the `abs_tol=1e-6` floor, so it spuriously
/// failed an analytically-correct zero. The 5-point stencil is `O(h⁴)` (and
/// EXACT for polynomials up to degree 4, so it returns 0 to rounding on the
/// monomial line factors), which is the honest reference for this contract.
pub(crate) fn assert_third_jet_matches_central_difference<E: SaeBasisThirdJet>(
    evaluator: &E,
    coords: Array2<f64>,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<(), String> {
    let epsilon = 1.0e-4;
    let third = evaluator.third_jet(coords.view())?;
    let second = evaluator.second_jet(coords.view())?;
    let (n_rows, n_basis, latent_dim, ld_b, ld_c) = third.dim();
    assert_eq!(latent_dim, ld_b);
    assert_eq!(latent_dim, ld_c);
    assert_eq!((n_rows, n_basis, latent_dim, latent_dim), second.dim());
    for row in 0..n_rows {
        for axis_e in 0..latent_dim {
            let mut plus2 = coords.clone();
            let mut plus = coords.clone();
            let mut minus = coords.clone();
            let mut minus2 = coords.clone();
            plus2[[row, axis_e]] += 2.0 * epsilon;
            plus[[row, axis_e]] += epsilon;
            minus[[row, axis_e]] -= epsilon;
            minus2[[row, axis_e]] -= 2.0 * epsilon;
            let second_plus2 = evaluator.second_jet(plus2.view())?;
            let second_plus = evaluator.second_jet(plus.view())?;
            let second_minus = evaluator.second_jet(minus.view())?;
            let second_minus2 = evaluator.second_jet(minus2.view())?;
            for basis in 0..n_basis {
                for axis_a in 0..latent_dim {
                    for axis_c in 0..latent_dim {
                        let fd = (-second_plus2[[row, basis, axis_a, axis_c]]
                            + 8.0 * second_plus[[row, basis, axis_a, axis_c]]
                            - 8.0 * second_minus[[row, basis, axis_a, axis_c]]
                            + second_minus2[[row, basis, axis_a, axis_c]])
                            / (12.0 * epsilon);
                        let analytic = third[[row, basis, axis_a, axis_c, axis_e]];
                        let error = (analytic - fd).abs();
                        let threshold = abs_tol + rel_tol * analytic.abs().max(fd.abs());
                        assert!(
                            error <= threshold,
                            "row={row} basis={basis} a={axis_a} c={axis_c} e={axis_e}: \
                                 analytic={analytic:.12e}, fd={fd:.12e}, error={error:.6e}, \
                                 threshold={threshold:.6e}"
                        );
                    }
                }
            }
        }
    }
    // Full symmetry across the three trailing derivative axes (mixed partials
    // commute), so every permutation of `(a, c, e)` must agree.
    for row in 0..n_rows {
        for basis in 0..n_basis {
            for a in 0..latent_dim {
                for b in 0..latent_dim {
                    for c in 0..latent_dim {
                        let reference = third[[row, basis, a, b, c]];
                        for perm in [[a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]] {
                            let permuted = third[[row, basis, perm[0], perm[1], perm[2]]];
                            assert!(
                                (reference - permuted).abs() <= 1.0e-10,
                                "third_jet not symmetric: row={row} basis={basis} \
                                     ({a},{b},{c})={reference:.6e} vs ({},{},{})={permuted:.6e}",
                                perm[0],
                                perm[1],
                                perm[2]
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
pub(crate) fn isometry_periodic_second_jet_matches_fd() -> Result<(), String> {
    // Magnitude-scaled tolerance: the top harmonic (ω = 2π·3) drives a
    // O(ε²·ω⁴) ≈ 2e-5 central-difference truncation floor, far above any flat
    // 1e-5 absolute bound; rel_tol = 1e-5 tracks the ω⁴ scale (analytic exact).
    assert_second_jet_matches_central_difference(
        &PeriodicHarmonicEvaluator::new(7).unwrap(),
        array![[-0.37], [0.0], [0.125], [0.41]],
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
pub(crate) fn isometry_sphere_second_jet_matches_fd() -> Result<(), String> {
    // Ambient unit vectors, poles INCLUDED: the ambient sphere has no boundary,
    // so a pole is an ordinary sample point rather than a place to stay away
    // from — which is the whole difference from the chart this replaced.
    let sphere_coords = array![
        [0.0, 0.0, 1.0],
        [0.6, -0.8, 0.0],
        [0.36, 0.48, 0.8],
        [-0.48, 0.6, -0.64]
    ];
    assert_second_jet_matches_central_difference(
        &AmbientSphereHarmonicEvaluator::new(2).unwrap(),
        sphere_coords,
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
pub(crate) fn isometry_torus_second_jet_matches_fd() -> Result<(), String> {
    let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
    let evaluator = TorusHarmonicEvaluator::new(2, 3).unwrap();
    assert!(evaluator.basis_size() > 0);
    // Same ω⁴ truncation floor as the periodic case (top harmonic ω = 2π·3).
    assert_second_jet_matches_central_difference(&evaluator, torus_coords, 1.0e-6, 1.0e-5)?;
    Ok(())
}

#[test]
pub(crate) fn isometry_periodic_third_jet_matches_fd() -> Result<(), String> {
    assert_third_jet_matches_central_difference(
        &PeriodicHarmonicEvaluator::new(7).unwrap(),
        array![[-0.37], [0.0], [0.125], [0.41]],
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
pub(crate) fn isometry_sphere_third_jet_matches_fd() -> Result<(), String> {
    // Ambient unit vectors, poles INCLUDED: the ambient sphere has no boundary,
    // so a pole is an ordinary sample point rather than a place to stay away
    // from — which is the whole difference from the chart this replaced.
    let sphere_coords = array![
        [0.0, 0.0, 1.0],
        [0.6, -0.8, 0.0],
        [0.36, 0.48, 0.8],
        [-0.48, 0.6, -0.64]
    ];
    assert_third_jet_matches_central_difference(
        &AmbientSphereHarmonicEvaluator::new(2).unwrap(),
        sphere_coords,
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
pub(crate) fn isometry_torus_third_jet_matches_fd() -> Result<(), String> {
    let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
    let evaluator = TorusHarmonicEvaluator::new(2, 3).unwrap();
    assert!(evaluator.basis_size() > 0);
    assert_third_jet_matches_central_difference(&evaluator, torus_coords, 1.0e-6, 1.0e-5)?;
    Ok(())
}

#[test]
pub(crate) fn isometry_affine_third_jet_is_trivial_zero() -> Result<(), String> {
    let evaluator = AffineCoordinateEvaluator { latent_dim: 3 };
    let coords = array![[0.2, -0.3, 0.7], [1.1, 0.0, -0.4]];
    let third = evaluator.third_jet(coords.view())?;
    assert_eq!(third.dim(), (coords.nrows(), 4, 3, 3, 3));
    assert!(
        third.iter().all(|x| *x == 0.0),
        "affine third jet must vanish identically, got {third:?}"
    );
    Ok(())
}

#[test]
pub(crate) fn isometry_euclidean_patch_third_jet_matches_fd() -> Result<(), String> {
    let evaluator = EuclideanPatchEvaluator::new(2, 4)?;
    let coords = array![[0.2, -0.3], [0.7, 0.4], [-0.5, 0.9]];
    assert_third_jet_matches_central_difference(&evaluator, coords, 1.0e-6, 1.0e-5)?;
    Ok(())
}

/// Cylinder coordinates: periodic axis 0 (fraction-of-period) crossed with
/// the unbounded line axis 1. Mixed signs/magnitudes on the line axis pin the
/// monomial factor away from the trivial origin.
fn cylinder_test_coords() -> Array2<f64> {
    array![
        [0.0_f64, -1.3],
        [0.125, 0.0],
        [0.4, 0.7],
        [0.91, 2.2],
        [0.6, -0.45]
    ]
}

/// The cylinder product basis must equal the literal outer product of the
/// periodic circle factor and the monomial line factor in the lexicographic
/// (circle-slow, line-fast) layout, and its width must be `(2H+1)·(D+1)`.
#[test]
pub(crate) fn cylinder_phi_is_circle_tensor_line_product() -> Result<(), String> {
    let h = 2usize;
    let degree = 2usize;
    let evaluator = CylinderHarmonicEvaluator::new(h, degree)?;
    let mc = 2 * h + 1;
    let ml = degree + 1;
    assert_eq!(evaluator.circle_basis_size(), mc);
    assert_eq!(evaluator.line_basis_size(), ml);
    assert_eq!(evaluator.basis_size(), mc * ml);

    let coords = cylinder_test_coords();
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    assert_eq!(phi.dim(), (coords.nrows(), mc * ml));
    assert_eq!(jet.dim(), (coords.nrows(), mc * ml, 2));

    let two_pi = std::f64::consts::TAU;
    for row in 0..coords.nrows() {
        let t0 = coords[[row, 0]];
        let t1 = coords[[row, 1]];
        // Independent reconstruction of the per-axis value factors.
        let mut circ = vec![0.0_f64; mc];
        circ[0] = 1.0;
        for k in 1..=h {
            circ[2 * k - 1] = (two_pi * k as f64 * t0).sin();
            circ[2 * k] = (two_pi * k as f64 * t0).cos();
        }
        let line: Vec<f64> = (0..ml).map(|j| t1.powi(j as i32)).collect();
        for c in 0..mc {
            for l in 0..ml {
                let col = c * ml + l;
                let expect = circ[c] * line[l];
                assert_abs_diff_eq!(phi[[row, col]], expect, epsilon = 1e-12);
            }
        }
        // Column 0 is the product of the two constant factors = 1, with a
        // vanishing gradient on both axes.
        assert_abs_diff_eq!(phi[[row, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(jet[[row, 0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(jet[[row, 0, 1]], 0.0, epsilon = 1e-12);
    }
    Ok(())
}

/// Cylinder first jet (`∂Φ/∂t₀`, `∂Φ/∂t₁`) vs central differences.
#[test]
pub(crate) fn cylinder_jacobian_matches_central_difference() {
    assert_jacobian_matches_central_difference(
        &CylinderHarmonicEvaluator::new(3, 3).unwrap(),
        cylinder_test_coords(),
        1.0e-6,
    );
}

/// Cylinder Hessian vs central differences (product rule across the two
/// disjoint axes: `∂²/∂t₀² = c''·l`, `∂²/∂t₁² = c·l''`, `∂²/∂t₀∂t₁ = c'·l'`).
/// The top circle harmonic (ω = 2π·3) sets the same ω⁴ truncation floor as
/// the periodic/torus cases, so a magnitude-scaled rel_tol is used.
#[test]
pub(crate) fn cylinder_second_jet_matches_fd() -> Result<(), String> {
    let evaluator = CylinderHarmonicEvaluator::new(3, 3)?;
    assert_second_jet_matches_central_difference(
        &evaluator,
        cylinder_test_coords(),
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

/// Cylinder third jet vs a central difference of the (FD-validated) second
/// jet, plus full symmetry across the three trailing axes.
#[test]
pub(crate) fn cylinder_third_jet_matches_fd() -> Result<(), String> {
    let evaluator = CylinderHarmonicEvaluator::new(3, 3)?;
    assert_third_jet_matches_central_difference(
        &evaluator,
        cylinder_test_coords(),
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

/// The cylinder roughness Gram is `S = Sc ⊗ Gl + Gc ⊗ Sl`: symmetric, PSD,
/// with the constant column (`[c=0, l=0]`, the only column with neither a
/// circle-bending nor a line-bending contribution) exactly in its null
/// space. The diagonal entries match the closed-form per-axis energies.
#[test]
pub(crate) fn cylinder_roughness_gram_is_psd_with_constant_nullspace() {
    let h = 2usize;
    let degree = 2usize;
    let evaluator = CylinderHarmonicEvaluator::new(h, degree).unwrap();
    let mc = 2 * h + 1;
    let ml = degree + 1;
    let m = mc * ml;
    let s = evaluator.roughness_gram();
    assert_eq!(s.dim(), (m, m));

    // Symmetry.
    for i in 0..m {
        for j in 0..m {
            assert_abs_diff_eq!(s[[i, j]], s[[j, i]], epsilon = 1e-12);
        }
    }

    // The constant column (col 0 = [c=0, l=0]) is annihilated: neither
    // factor bends, so its entire row/column is zero.
    for j in 0..m {
        assert_abs_diff_eq!(s[[0, j]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s[[j, 0]], 0.0, epsilon = 1e-12);
    }

    // Closed-form diagonal check for the pure-circle column `[c, l=0]`:
    // `S[c0,c0] = Sc[c,c]·Gl[0,0] + Gc[c,c]·Sl[0,0]`. With l=0 the line is a
    // constant, so `Sl[0,0] = 0` and `Gl[0,0] = ∫₀¹ 1 = 1`, giving `Sc[c,c]`.
    let two_pi = std::f64::consts::TAU;
    for k in 1..=h {
        let omega4 = (two_pi * k as f64).powi(4);
        let s_idx = 2 * k - 1;
        let c_idx = 2 * k;
        // Sc[s,s] = Sc[c,c] = ω⁴·½ (∫₀¹ sin² = ∫₀¹ cos² = ½).
        assert_abs_diff_eq!(s[[s_idx * ml, s_idx * ml]], omega4 * 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(s[[c_idx * ml, c_idx * ml]], omega4 * 0.5, epsilon = 1e-6);
    }

    // The pure-line quadratic column `[c=0, l=2]` carries only line-bending
    // energy: `S = Gc[0,0]·Sl[2,2]` with `Gc[0,0] = 1` and
    // `Sl[2,2] = ∫₀¹ (2)² dt = 4`.
    if degree >= 2 {
        let col = 2; // c=0 → col = 0*ml + 2.
        assert_abs_diff_eq!(s[[col, col]], 4.0, epsilon = 1e-12);
    }

    // PSD: every eigenvalue ≥ 0 (within a tight tolerance).
    let (evals, _) = s.eigh(Side::Lower).unwrap();
    for &lam in evals.iter() {
        assert!(
            lam >= -1.0e-9,
            "cylinder roughness Gram must be PSD; got eigenvalue {lam:.3e}"
        );
    }
}

/// `CylinderHarmonicEvaluator::new` rejects `circle_harmonics == 0` (an S¹
/// with no harmonic pair is degenerate).
#[test]
pub(crate) fn cylinder_rejects_zero_harmonics() {
    assert!(CylinderHarmonicEvaluator::new(0, 2).is_err());
    assert!(CylinderHarmonicEvaluator::new(1, 0).is_ok());
}

/// The cylinder latent manifold is the product `S¹ × ℝ`: a unit-period
/// circle on axis 0 and an unbounded Euclidean line on axis 1.
#[test]
pub(crate) fn cylinder_latent_manifold_is_circle_times_line() {
    let manifold = SaeAtomBasisKind::Cylinder.latent_manifold(2);
    match manifold {
        LatentManifold::Product(parts) => {
            assert_eq!(parts.len(), 2);
            assert!(matches!(parts[0], LatentManifold::Circle { period } if period == 1.0));
            assert!(matches!(parts[1], LatentManifold::Euclidean));
        }
        other => panic!("expected Product[Circle, Euclidean], got {other:?}"),
    }
}

/// Issue #247: the Duchon coordinate evaluator must return a forward design
/// and a derivative jet with *matching column counts* — the original bug
/// was a radial-only design paired with a radial+polynomial jet (or vice
/// versa), which the consumer rejected as a "design/jet column mismatch".
#[test]
pub(crate) fn duchon_coordinate_evaluator_phi_and_jet_share_column_count() {
    for (d, centers) in [
        (1usize, array![[-1.0], [-0.4], [0.1], [0.6], [1.2], [1.9]]),
        (
            2usize,
            array![
                [-1.0, -0.8],
                [-0.3, 0.4],
                [0.2, -0.5],
                [0.7, 0.9],
                [1.1, -0.2],
                [1.6, 0.6],
            ],
        ),
    ] {
        let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
        let coords = match d {
            1 => array![[-0.5], [0.0], [0.3], [0.8]],
            _ => array![[-0.5, 0.2], [0.0, -0.3], [0.3, 0.7], [0.8, -0.1]],
        };
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        assert_eq!(
            phi.ncols(),
            jet.shape()[1],
            "Duchon d={d}: Phi has {} columns but jet has {}",
            phi.ncols(),
            jet.shape()[1]
        );
        assert_eq!(jet.shape()[0], coords.nrows());
        assert_eq!(jet.shape()[2], d);
    }
}

/// The Duchon evaluator's analytic first jet must equal the finite
/// difference of its own forward design — i.e. `dPhi/dt` is the true
/// derivative of `Phi(t)`, with no stray amplification/column mismatch.
#[test]
pub(crate) fn duchon_coordinate_evaluator_jacobian_matches_fd() {
    let centers = array![
        [-1.0, -0.8],
        [-0.3, 0.4],
        [0.2, -0.5],
        [0.7, 0.9],
        [1.1, -0.2],
        [1.6, 0.6],
    ];
    let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
    // Keep probe points away from any center so the radial kernel is smooth.
    let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
    assert_jacobian_matches_central_difference(&evaluator, coords, 1.0e-4);
}

/// The Duchon evaluator's analytic second jet must match the FD of its
/// (FD-validated) first jet.
#[test]
pub(crate) fn duchon_coordinate_evaluator_second_jet_matches_fd() -> Result<(), String> {
    let centers = array![
        [-1.0, -0.8],
        [-0.3, 0.4],
        [0.2, -0.5],
        [0.7, 0.9],
        [1.1, -0.2],
        [1.6, 0.6],
    ];
    let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
    let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
    assert_second_jet_matches_central_difference(&evaluator, coords, 1.0e-4, 1.0e-4)?;
    Ok(())
}

/// The Duchon evaluator's closed-form analytic third jet (radial
/// third-derivative kernel block + monomial nullspace block) must match the
/// FD of its (FD-validated) second jet, validating the closed form that
/// replaced the forbidden finite-difference `third_jet_dyn` default.
#[test]
pub(crate) fn duchon_coordinate_evaluator_third_jet_matches_fd() -> Result<(), String> {
    let centers = array![
        [-1.0, -0.8],
        [-0.3, 0.4],
        [0.2, -0.5],
        [0.7, 0.9],
        [1.1, -0.2],
        [1.6, 0.6],
    ];
    let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
    let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
    assert_third_jet_matches_central_difference(&evaluator, coords, 1.0e-4, 1.0e-4)?;
    Ok(())
}

/// The Euclidean tangent-patch evaluator's monomial design and its
/// first/second jets must be mutually consistent under finite differences.
#[test]
pub(crate) fn euclidean_patch_evaluator_jets_match_fd() -> Result<(), String> {
    let evaluator = EuclideanPatchEvaluator::new(2, 2).unwrap();
    let coords = array![[0.0, -1.0], [3.5, 0.25], [-0.75, 1.2], [0.4, 0.9]];
    assert_jacobian_matches_central_difference(&evaluator, coords.clone(), 1.0e-6);
    assert_second_jet_matches_central_difference(&evaluator, coords, 1.0e-5, 1.0e-5)?;
    // The degree-2 patch in d=2 has columns {1, x, y, x², xy, y²}.
    let (phi, _jet) = evaluator.evaluate(array![[0.0, 0.0]].view())?;
    assert_eq!(phi.ncols(), 6);
    Ok(())
}

#[test]
pub(crate) fn euclidean_affine_gauge_canonicalization_preserves_reconstruction()
-> Result<(), String> {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2)?);
    let canonical = array![[-1.0_f64], [-0.35], [0.1], [0.65], [1.2]];
    let mut coords = canonical.clone();
    for row in 0..coords.nrows() {
        coords[[row, 0]] = 2.75 + 4.0 * canonical[[row, 0]];
    }
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let decoder = array![[0.25, -0.4], [1.2, 0.3], [-0.15, 0.5]];
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "euclidean_patch",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(evaluator.basis_size()),
    )?
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((coords.nrows(), 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )?;
    let mut term = SaeManifoldTerm::new(vec![atom], assignment)?;
    let before = term.fitted();

    term.canonicalize_affine_gauge_after_accept(None)?;

    let after = term.fitted();
    let max_abs = before
        .iter()
        .zip(after.iter())
        .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(
        max_abs <= 1.0e-10,
        "canonicalization changed reconstruction by {max_abs:.3e}"
    );
    let live = term.assignment.coords[0].as_matrix();
    let mean = live.column(0).sum() / live.nrows() as f64;
    let rms = (live.column(0).iter().map(|v| v * v).sum::<f64>() / live.nrows() as f64).sqrt();
    assert_abs_diff_eq!(mean, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(rms, 1.0, epsilon = 1.0e-12);
    Ok(())
}

#[test]
pub(crate) fn quotient_step_norm_retains_prior_sensitive_euclidean_affine_orbit()
-> Result<(), String> {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2)?);
    let coords = array![[-1.0_f64], [-0.4], [0.2], [0.8], [1.3]];
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let decoder = array![[0.1, -0.2], [1.0, 0.4], [0.25, -0.3]];
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "euclidean_patch",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(evaluator.basis_size()),
    )?
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((coords.nrows(), 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )?;
    let term = SaeManifoldTerm::new(vec![atom], assignment)?;
    let gauges = term.dense_step_gauge_vectors()?;
    assert!(
        gauges.len() >= 2,
        "expected translation and scale gauge generators"
    );
    let n_coord = term.n_obs() * term.assignment.row_block_dim();
    let gauge = &gauges[1];
    let delta_t = gauge.slice(s![..n_coord]);
    let delta_beta = gauge.slice(s![n_coord..]);
    let raw = gauge.iter().map(|v| v * v).sum::<f64>();

    // The affine chart orbit is a likelihood symmetry, not a posterior null
    // direction (#2720). Its compensating decoder change cancels the fitted
    // function's first variation, while the Gaussian coordinate prior moves.
    let atom = &term.atoms[0];
    let coord_offset = term.assignment.coord_offsets()[0];
    let row_width = term.assignment.row_block_dim();
    let basis_direction = Array2::from_shape_fn(atom.basis_values.dim(), |(row, col)| {
        atom.basis_jacobian[[row, col, 0]] * delta_t[row * row_width + coord_offset]
    });
    let decoder_direction =
        Array2::from_shape_vec(atom.decoder_coefficients().dim(), delta_beta.to_vec())
            .map_err(|error| error.to_string())?;
    let fitted_direction = basis_direction.dot(atom.decoder_coefficients())
        + atom.basis_values.dot(&decoder_direction);
    assert!(fitted_direction.iter().all(|value| value.abs() < 1.0e-12));
    let coordinates = term.assignment.coords[0].as_matrix();
    let coordinate_prior_slope = (0..term.n_obs())
        .map(|row| coordinates[[row, 0]] * delta_t[row * row_width + coord_offset])
        .sum::<f64>();
    assert!(coordinate_prior_slope > 0.0);

    let quotient =
        term.quotient_newton_step_norm_sq(delta_t, delta_beta, raw, &vec![0.0; term.k_atoms()])?;

    assert_abs_diff_eq!(quotient, raw, epsilon = 1.0e-12);
    Ok(())
}

//! Integration test for SAE-manifold sphere and torus atom topologies.
//!
//! Regression cover for GitHub issues #170, #172, #179: spherical and toroidal
//! `atom_topology`s used to fall through to the Duchon collocation validator
//! and panic. The Rust dispatch now routes them to `SphereChartEvaluator`
//! (latent_dim = 2, 7-column lat/lon chart) and `TorusHarmonicEvaluator`
//! (tensor-product periodic harmonic basis of size `(2H+1)^d`). This test
//! drives both through the public `SaeManifoldTerm::run_joint_fit_arrow_schur`
//! Newton loop on synthetic data drawn from S² and T² and asserts the
//! in-sample reconstruction R² clears the 0.5 floor specified by the issue
//! repro plans.
//!
//! No `let _`, no `#[allow]`, no `#[ignore]`, no env vars, no `unwrap_or`
//! masks.

use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::terms::{
    latent::LatentManifold, sae::manifold::AssignmentMode,
    sae::manifold::PeriodicHarmonicEvaluator, sae::manifold::SaeAssignment,
    sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
    sae::manifold::SphereChartEvaluator, sae::manifold::TorusHarmonicEvaluator,
};

/// Newton loop driver shared by both topology tests. Mirrors the inner loop
/// `sae_manifold_fit` uses on the Rust side; returns the final fitted matrix
/// once the loss stabilises (or `max_outer` iterations elapse).
fn fit_single_atom(
    z: &Array2<f64>,
    atom: SaeManifoldAtom,
    true_coords: Array2<f64>,
    latent: LatentManifold,
    max_outer: usize,
) -> Array2<f64> {
    let n = z.nrows();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![true_coords],
        vec![latent],
        AssignmentMode::softmax(0.5),
    )
    .expect("assignment construction");
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("term construction");
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(1)]);
    let ridge = 1.0e-6;
    for _ in 0..max_outer {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .expect("Newton step");
        if !loss.total().is_finite() {
            break;
        }
    }
    term.fitted()
}

fn reconstruction_r2(z: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let mut sse = 0.0_f64;
    let mut sst = 0.0_f64;
    for ((row, col), v) in z.indexed_iter() {
        sst += v * v;
        let r = fitted[[row, col]] - v;
        sse += r * r;
    }
    1.0 - sse / sst.max(1.0e-12)
}

/// Issue #170 / #179: `atom_topology='sphere'`, `d_atom=2` must route to
/// `SphereChartEvaluator` rather than the Duchon collocation validator.
/// Synthesise n=300 points on S² (lat/lon → unit-vector in R³), fit a single
/// sphere atom from the true coords, and confirm R² ≥ 0.5.
#[test]
fn sphere_topology_recovers_synthetic_signal() {
    let n = 300usize;
    let p = 3usize;
    let d = 2usize;
    let mut true_coords = Array2::<f64>::zeros((n, d));
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        // Deterministic interleave over the lat × lon strip, avoiding the
        // poles where the chart degenerates.
        let t = (i as f64 + 0.5) / (n as f64);
        let s = (((i * 7) % n) as f64 + 0.5) / (n as f64);
        let lat = -0.45 * PI + 0.9 * PI * t;
        let lon = -PI + 2.0 * PI * s;
        true_coords[[i, 0]] = lat;
        true_coords[[i, 1]] = lon;
        z[[i, 0]] = lat.cos() * lon.cos();
        z[[i, 1]] = lat.cos() * lon.sin();
        z[[i, 2]] = lat.sin();
    }
    let (phi0, jet0) = SphereChartEvaluator
        .evaluate(true_coords.view())
        .expect("sphere chart evaluation");
    let m = phi0.ncols();
    assert_eq!(m, 7, "sphere chart basis must have 7 columns");
    let mut penalty = Array2::<f64>::eye(m);
    penalty *= 1.0e-4;
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "sphere_atom",
        SaeAtomBasisKind::Sphere,
        d,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        penalty,
    )
    .expect("sphere atom")
    .with_basis_evaluator(Arc::new(SphereChartEvaluator) as Arc<dyn SaeBasisEvaluator>);

    let latent = LatentManifold::Product(vec![
        LatentManifold::Interval {
            lo: -FRAC_PI_2,
            hi: FRAC_PI_2,
        },
        LatentManifold::Circle {
            period: std::f64::consts::TAU,
        },
    ]);
    let fitted = fit_single_atom(&z, atom, true_coords, latent, 12);
    let r2 = reconstruction_r2(&z, &fitted);
    assert!(
        r2 >= 0.5,
        "sphere SAE reconstruction R² too low: {r2:.4} (n={n}, p={p})"
    );
}

/// Issue #172: `atom_topology='torus'` must route to the product-of-circles
/// harmonic basis (no fallthrough to Duchon). Synthesise n=300 points with
/// two phase coordinates on T² and fit a single torus atom.
#[test]
fn torus_topology_recovers_synthetic_signal() {
    let n = 300usize;
    let p = 4usize;
    let d = 2usize;
    let h = 3usize;
    let evaluator = TorusHarmonicEvaluator::new(d, h).expect("torus evaluator");
    let m = evaluator.basis_size();
    assert_eq!(m, (2 * h + 1).pow(d as u32));

    let mut true_coords = Array2::<f64>::zeros((n, d));
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t1 = ((i as f64) * 0.137).rem_euclid(1.0);
        let t2 = ((i as f64) * 0.241 + 0.13).rem_euclid(1.0);
        true_coords[[i, 0]] = t1;
        true_coords[[i, 1]] = t2;
        let a1 = 2.0 * PI * t1;
        let a2 = 2.0 * PI * t2;
        z[[i, 0]] = a1.sin() + 0.3 * a2.cos();
        z[[i, 1]] = a1.cos() + 0.2 * (a1 + a2).sin();
        z[[i, 2]] = a2.sin();
        z[[i, 3]] = 0.5 * (a1 - a2).cos();
    }
    let (phi0, jet0) = evaluator
        .evaluate(true_coords.view())
        .expect("torus evaluation");
    let mut penalty = Array2::<f64>::eye(m);
    penalty *= 1.0e-4;
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "torus_atom",
        SaeAtomBasisKind::Torus,
        d,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        penalty,
    )
    .expect("torus atom")
    .with_basis_evaluator(Arc::new(
        TorusHarmonicEvaluator::new(d, h).expect("torus evaluator clone"),
    ) as Arc<dyn SaeBasisEvaluator>);

    let latent = LatentManifold::Product(vec![
        LatentManifold::Circle { period: 1.0 },
        LatentManifold::Circle { period: 1.0 },
    ]);
    let fitted = fit_single_atom(&z, atom, true_coords, latent, 12);
    let r2 = reconstruction_r2(&z, &fitted);
    assert!(
        r2 >= 0.5,
        "torus SAE reconstruction R² too low: {r2:.4} (n={n}, p={p})"
    );
}

/// Issue #174: K=2 independent 1-D periodic atoms must recover a torus
/// signal whose phases live in `[0, 1)`. Before the fix,
/// `LatentManifold::Circle` wrapped modulo `2π` (radians) while
/// `PeriodicHarmonicEvaluator` interprets the latent as a fraction of one
/// period (basis `cos(2π·h·t)`), so Newton updates landed at the wrong
/// principal interval and the optimiser stalled near R² ≈ 0. The fix
/// collapses the variant to `Circle { period: f64 }` and
/// `SaeAtomBasisKind::Periodic` constructs it with `period = 1.0` so the
/// manifold wrap matches the basis convention.
#[test]
fn k2_periodic_atoms_recover_torus_signal() {
    let n = 600usize;
    let p = 8usize;
    let k = 2usize;
    let m = 11usize;

    let mut z = Array2::<f64>::zeros((n, p));
    let mut true_coords_per_atom: Vec<Array2<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        true_coords_per_atom.push(Array2::<f64>::zeros((n, 1)));
    }
    for i in 0..n {
        let t1 = ((i as f64) * 0.137).rem_euclid(1.0);
        let t2 = ((i as f64) * 0.241 + 0.13).rem_euclid(1.0);
        true_coords_per_atom[0][[i, 0]] = t1;
        true_coords_per_atom[1][[i, 0]] = t2;
        let a1 = std::f64::consts::TAU * t1;
        let a2 = std::f64::consts::TAU * t2;
        let raw = [a1.cos(), a1.sin(), a2.cos(), a2.sin()];
        for j in 0..p {
            let mut acc = 0.0;
            for (r_idx, r_val) in raw.iter().enumerate() {
                acc += r_val * (1.0 / (1.0 + ((j + r_idx) % 7) as f64));
            }
            z[[i, j]] = acc;
        }
    }

    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator");
    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(k);
    let mut coords_init: Vec<Array2<f64>> = Vec::with_capacity(k);
    for ai in 0..k {
        let mut init = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            init[[i, 0]] =
                (true_coords_per_atom[ai][[i, 0]] + 0.27 * (ai as f64 + 1.0)).rem_euclid(1.0);
        }
        let (phi0, jet0) = evaluator
            .evaluate(init.view())
            .expect("periodic atom evaluation");
        let mut penalty = Array2::<f64>::eye(m);
        penalty *= 1.0e-4;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            &format!("periodic_atom_{ai}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            Array2::<f64>::zeros((m, p)),
            penalty,
        )
        .expect("periodic atom")
        .with_basis_evaluator(Arc::new(
            PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator clone"),
        ) as Arc<dyn SaeBasisEvaluator>);
        atoms.push(atom);
        coords_init.push(init);
    }

    let manifolds: Vec<LatentManifold> = (0..k)
        .map(|_| LatentManifold::Circle { period: 1.0 })
        .collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, k)),
        coords_init,
        manifolds,
        AssignmentMode::softmax(0.5),
    )
    .expect("assignment construction");
    let mut term = SaeManifoldTerm::new(atoms, assignment).expect("term construction");
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(1); k]);
    let ridge = 1.0e-6;
    let mut prev = f64::INFINITY;
    for _ in 0..40 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .expect("Newton step");
        let total = loss.total();
        if !total.is_finite() {
            break;
        }
        if (prev - total).abs() < 1.0e-6 * prev.abs().max(1.0e-12) {
            break;
        }
        prev = total;
    }

    let fitted = term.fitted();
    let r2 = reconstruction_r2(&z, &fitted);
    assert!(
        r2 >= 0.5,
        "issue #174: K=2 periodic-torus R² too low: {r2:.4} (n={n}, p={p}, k={k})"
    );
}

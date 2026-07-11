use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    latent::LatentManifold, sae::manifold::AssignmentMode, sae::manifold::SaeAssignment,
    sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};

enum CurveKind {
    Line,
    Parabola,
}

fn synthetic_curve(kind: CurveKind) -> (Array2<f64>, Array2<f64>) {
    let n = 150usize;
    let p = 8usize;
    let mut coords = Array2::<f64>::zeros((n, 1));
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let u = -1.0 + 2.0 * row as f64 / (n as f64 - 1.0);
        coords[[row, 0]] = 2.5 + 3.0 * u;
        let q = match kind {
            CurveKind::Line => 0.0,
            CurveKind::Parabola => u * u - 1.0 / 3.0,
        };
        for col in 0..p {
            let linear_loading = 0.35 + 0.07 * col as f64;
            let curve_loading = -0.18 + 0.05 * col as f64;
            let offset = 0.08 * ((col % 3) as f64 - 1.0);
            let phase = (row * (col + 3)) as f64;
            let noise = 0.04 * (phase.sin() + 0.5 * (0.37 * phase).cos());
            z[[row, col]] = offset + linear_loading * u + curve_loading * q + noise;
        }
    }
    (coords, z)
}

fn explained_variance(observed: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let n = observed.nrows();
    let p = observed.ncols();
    let mut means = vec![0.0_f64; p];
    for row in 0..n {
        for col in 0..p {
            means[col] += observed[[row, col]];
        }
    }
    for mean in &mut means {
        *mean /= n as f64;
    }
    let mut sst = 0.0_f64;
    let mut sse = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let centered = observed[[row, col]] - means[col];
            let residual = observed[[row, col]] - fitted[[row, col]];
            sst += centered * centered;
            sse += residual * residual;
        }
    }
    1.0 - sse / sst.max(1.0e-12)
}

fn fit_euclidean_curve(kind: CurveKind) -> f64 {
    let (coords, z) = synthetic_curve(kind);
    let n = z.nrows();
    let p = z.ncols();
    let evaluator = EuclideanPatchEvaluator::new(1, 2).expect("euclidean evaluator");
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("initial euclidean basis");
    let m = phi.ncols();
    let smooth_penalty =
        gam::basis::create_difference_penalty_matrix(m, 2, None).expect("roughness penalty");
    let atom = SaeManifoldAtom::new(
        "euclidean_curve",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p)),
        smooth_penalty,
    )
    .expect("euclidean atom")
    .with_basis_second_jet(Arc::new(evaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let mut rho = SaeManifoldRho::new(0.0, (0.01_f64).ln(), vec![Array1::<f64>::zeros(1)]);
    let ridge = 1.0e-6;
    let loss = term
        .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 96, 1.0, ridge, ridge)
        .expect("fixed-rho inner fit should converge");
    assert!(
        loss.total().is_finite(),
        "fixed-rho inner fit returned non-finite loss: {}",
        loss.total()
    );
    let (criterion, criterion_loss) = term
        .reml_criterion(z.view(), &rho, None, 16, 1.0, ridge, ridge)
        .expect("REML fixed-rho convergence check should accept the inner optimum");
    assert!(
        criterion.is_finite() && criterion_loss.total().is_finite(),
        "REML criterion should be finite after the inner solve: {criterion}"
    );
    explained_variance(&z, &term.try_fitted().expect("fitted values"))
}

#[test]
fn euclidean_line_and_parabola_inner_solves_converge_with_high_ev() {
    let line_ev = fit_euclidean_curve(CurveKind::Line);
    assert!(line_ev > 0.95, "line EV too low: {line_ev:.6}");
    let parabola_ev = fit_euclidean_curve(CurveKind::Parabola);
    assert!(parabola_ev > 0.95, "parabola EV too low: {parabola_ev:.6}");
}

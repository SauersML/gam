//! #3684 — the pitchfork soft mode of the planted K=1 Euclidean-patch arc, measured at converged
//! states on both branches of the bifurcation surface: the soft pencil eigenvalue `μ`, the cubic
//! `c = D³f[v,v,v]`, the bare quartic `D⁴f[v,v,v,v]`, the profiled quartic
//! `κ = D⁴f[v⁴] − 3·wᵀA⊥⁺w` with `w = D³f[v,v,·]`, the normal-form ratio `c²/(3μκ)` and the
//! Laplace validity scale `κ/μ²`, each read off finite differences of the inner gradient along the
//! soft eigenvector at the converged state.

use super::tests_orbit_gradient_2234::{displaced, inner_gradient};
use super::*;
use crate::assignment::{AssignmentMode, SaeAssignment};

fn planted_arc_objective() -> SaeManifoldOuterObjective {
    use crate::manifold::tests::deterministic_circle_noise;
    use gam_linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
    let (n, p, sigma) = (256usize, 4usize, 0.02);
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
        -1.0 + 2.0 * (row as f64) / (n as f64 - 1.0)
    });
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        let t = coords[[row, 0]];
        deterministic_circle_noise(col, 0) * t
            + deterministic_circle_noise(col, 1) * t * t
            + sigma * deterministic_circle_noise(row, col)
    });
    let evaluator = Arc::new(
        crate::basis::EuclideanPatchEvaluator::new(1, 2)
            .expect("dim 1, degree 2 is a valid Euclidean patch"),
    );
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the planted coordinates lie in the Euclidean patch domain");
    let decoder = fast_ata(&phi)
        .cholesky(faer::Side::Lower)
        .expect("the planted arc's patch Gram is positive definite")
        .solve_mat(&fast_atb(&phi, &target));
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "arc",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("phi, jet, decoder and gram were built with matching shapes")
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("one logit column, coordinate block and manifold");
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("one atom over the same rows");
    term.gpu_policy = gam_gpu::GpuPolicy::Off;
    let seed_rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, seed_rho, 40, 1.0, 1.0e-6, 1.0e-6)
}

fn joint(vector: &SaeArrowVector) -> Array1<f64> {
    Array1::from_iter(vector.t.iter().chain(vector.beta.iter()).copied())
}

/// `log[(1/√2π)∫exp(−η²/2 − ε·η⁴/24)dη]`: the symmetric-branch soft integral over its Laplace
/// value, in the Laplace-scaled coordinate `η = √μ·y`, `ε = κ/μ²`.
fn symmetric_branch_correction(epsilon: f64) -> f64 {
    let half_width = 12.0_f64.min(6.0 * (24.0 / epsilon).powf(0.25));
    let intervals = 20_000usize;
    let step = 2.0 * half_width / intervals as f64;
    let mut sum = 0.0;
    for index in 0..=intervals {
        let eta = -half_width + step * index as f64;
        let weight = if index == 0 || index == intervals {
            1.0
        } else if index % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += weight * (-(eta * eta) / 2.0 - epsilon * eta.powi(4) / 24.0).exp();
    }
    (sum * step / 3.0 / (2.0 * std::f64::consts::PI).sqrt()).ln()
}

/// `log[√(z/2π)∫exp(−(z/8)(ξ²−1)²)dξ]`: the broken-branch double well (both Z2 images) over the
/// Laplace value of the one well the inner solve converged to, `z = μ·y₀² = 3μ²/κ`.
fn broken_branch_correction(z: f64) -> f64 {
    let half_width = (1.0 + (10_368.0 / z).sqrt()).sqrt();
    let intervals = ((40.0 * half_width * z.sqrt()).ceil() as usize).clamp(20_000, 20_000_000);
    let intervals = intervals + intervals % 2;
    let step = 2.0 * half_width / intervals as f64;
    let mut sum = 0.0;
    for index in 0..=intervals {
        let xi = -half_width + step * index as f64;
        let weight = if index == 0 || index == intervals {
            1.0
        } else if index % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let well = xi * xi - 1.0;
        sum += weight * (-(z / 8.0) * well * well).exp();
    }
    (sum * step / 3.0 * (z / (2.0 * std::f64::consts::PI)).sqrt()).ln()
}

#[test]
fn soft_mode_quartic_across_the_pitchfork_3684() {
    let centre = [2.1697, 2.8027];
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let offsets = [-3.0e-2, -1.0e-2, -1.0e-3, -1.0e-4, 1.0e-4, 1.0e-3, 1.0e-2, 3.0e-2];
    let steps = [3.0e-1, 1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3];
    for &s in &offsets {
        let mut objective = planted_arc_objective();
        let flat = Array1::from(vec![centre[0] + s * inv_sqrt2, centre[1] + s * inv_sqrt2]);
        let rho = objective.baseline_rho.from_flat(flat.view()).expect("a flat ρ in layout");
        objective
            .evaluate_outer_criterion_route(&rho, true, false)
            .expect("the dense route converges the state");
        let target = objective.target.clone();
        let mut state = objective.term.clone();
        let coords = state.assignment.coords[0].as_matrix().to_owned();
        let decoder = state.atoms[0].decoder_coefficients().clone();
        eprintln!(
            "SOFT3684 s={s:+.1e} coords mean={:.6e} min={:.6e} max={:.6e} frame_none={} decoder={:?}",
            coords.mean().unwrap_or(f64::NAN),
            coords.iter().copied().fold(f64::INFINITY, f64::min),
            coords.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            state.atoms[0].decoder_frame.is_none(),
            decoder.iter().map(|value| format!("{value:.5e}")).collect::<Vec<_>>(),
        );
        let criterion = state
            .penalized_quasi_laplace_criterion_with_geometry(
                target.view(),
                &rho,
                None,
                0,
                1.0,
                1.0e-6,
                1.0e-6,
                true,
            )
            .expect("the criterion prices the converged state");
        let cost = criterion.0;
        let geometry = criterion.3.expect("the dense criterion hands out the block it priced");
        let block = &geometry.block;
        let total_t = geometry.total_t;
        let mu = block.eigenvalues[0];
        let vectors = &block.eigenvectors;
        let soft = vectors.column(0).to_owned();
        let v = SaeArrowVector {
            t: soft.slice(s![..total_t]).to_owned(),
            beta: soft.slice(s![total_t..]).to_owned(),
        };
        eprintln!(
            "SOFT3684 s={s:+.1e} cost={cost:.10e} dim={} total_t={total_t} mu={mu:.6e} mu1={:.6e} floor0={:.6e} |v_t|={:.4e} |v_beta|={:.4e} v_beta={:?}",
            block.eigenvalues.len(),
            block.eigenvalues[1],
            block.rank_floor(0),
            v.t.dot(&v.t).sqrt(),
            v.beta.dot(&v.beta).sqrt(),
            v.beta.iter().map(|value| format!("{value:.4e}")).collect::<Vec<_>>(),
        );
        let gradient_at = |scale: f64| -> Array1<f64> {
            joint(&inner_gradient(&displaced(&state, &v, scale), target.view(), &rho))
        };
        let g0 = gradient_at(0.0);
        for &h in &steps {
            let gp = gradient_at(h);
            let gm = gradient_at(-h);
            let gp2 = gradient_at(2.0 * h);
            let gm2 = gradient_at(-2.0 * h);
            let hessian_v = (&gp - &gm) / (2.0 * h);
            let w = (&gp - &(2.0 * &g0) + &gm) / (h * h);
            let fourth = (&gp2 - &(2.0 * &gp) + &(2.0 * &gm) - &gm2) / (2.0 * h * h * h);
            let curvature = soft.dot(&hessian_v);
            let cubic = soft.dot(&w);
            let bare_quartic = soft.dot(&fourth);
            let mut profile = 0.0;
            for index in 1..block.eigenvalues.len() {
                let component = vectors.column(index).dot(&w);
                profile += component * component / block.eigenvalues[index];
            }
            let kappa = bare_quartic - 3.0 * profile;
            let normal_form = cubic * cubic / (3.0 * mu * kappa);
            let epsilon = kappa / (mu * mu);
            let (sym, brk) = if kappa > 0.0 {
                (symmetric_branch_correction(epsilon), broken_branch_correction(3.0 / epsilon))
            } else {
                (f64::NAN, f64::NAN)
            };
            eprintln!(
                "SOFT3684 s={s:+.1e} h={h:.1e} vAv={curvature:.6e} vAv/mu={:.6e} c={cubic:.6e} D4={bare_quartic:.6e} wA+w={profile:.6e} kappa={kappa:.6e} c2/(3mu*kappa)={normal_form:.6e} kappa/mu2={epsilon:.6e} |g0|={:.3e} delta_sym={sym:.6e} delta_brk={brk:.6e} cost-delta_sym={:.10e} cost-delta_brk={:.10e}",
                curvature / mu,
                g0.dot(&g0).sqrt(),
                cost - sym,
                cost - brk,
            );
        }
    }
}

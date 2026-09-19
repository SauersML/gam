#![cfg(test)]
//! #2933 F39 — an embedded sphere coordinate prices its tangent response.
//!
//! An `S²` coordinate stores three ambient numbers but moves along two tangent
//! directions: the assembly projects its gradient and majorizer onto `T_t S` and
//! pins the normal only so the ambient factorization stays invertible, and the
//! retraction removes any normal motion. The response oracle of
//! [`super::tests_fitted_response_edf_2933`] re-solves the perturbed fit to its
//! tangent-projected root, so it measures the tangent response independently of
//! the operator. A divergence that reads the raw ambient jets, or an operator
//! whose residual curvature is ambient, counts part of each normal direction as
//! fitted data and misses it.
use super::tests_fitted_response_edf_2933::{
    ROOT_GRADIENT_CEILING, assert_prices_resolved_response, polish_to_root, resolved_response,
    root_norm, trace_and_residual_dof,
};
use super::tests_fitted_response_frames_2933::rademacher_quadratic_form_variance;
use super::*;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};

/// Fit from the term's current state, hold the gates the fit's last assembly
/// installed, then certify the root.
fn fit_and_certify(
    label: &str,
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &mut SaeManifoldRho,
) -> ArrowFactorCache {
    term.run_joint_fit_arrow_schur(target.view(), rho, None, 80, 1.0, 1.0e-7, 1.0e-7)
        .unwrap_or_else(|error| panic!("{label}: the fixture fits: {error}"));
    if !term.streaming_gates_frozen {
        let gates = term.collapse_prevention_gates();
        term.declare_collapse_prevention_gates(&gates);
    }
    let (cache, trajectory) = polish_to_root(term, target.view(), rho);
    let norm = root_norm(&trajectory);
    assert!(
        norm <= ROOT_GRADIENT_CEILING,
        "{label}: the fixture's root stalled at ‖g‖={norm:.3e}"
    );
    cache
}

fn sphere_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, ArrowFactorCache) {
    let n = 10usize;
    let p = 3usize;
    let evaluator = Arc::new(
        AmbientSphereHarmonicEvaluator::new(1).expect("degree one is a valid sphere basis"),
    );
    let coords = Array2::<f64>::from_shape_fn((n, 3), |(row, axis)| {
        let t = (row as f64 + 0.5) / n as f64;
        let lat = -0.9 + 1.8 * t;
        let lon = 0.3 + 5.1 * t;
        [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()][axis]
    });
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("unit coordinates evaluate");
    let width = phi.ncols();
    let decoder = Array2::<f64>::from_shape_fn((width, p), |(basis, out)| {
        0.2 + 0.6 * ((2 * basis + 3 * out) as f64 * 0.8 + 0.4).sin()
    });
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        for out in 0..p {
            target[[row, out]] += 0.05 * (1.3 * row as f64 + 2.1 * out as f64).cos();
        }
    }
    let penalty =
        Array2::<f64>::from_diag(&Array1::from_shape_fn(width, |basis| 1.0 + basis as f64));
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "sphere".to_string(),
        SaeAtomBasisKind::Sphere,
        3,
        phi,
        jet,
        decoder,
        penalty,
    )
    .expect("sphere atom: basis width, latent dimension and decoder shape agree")
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Sphere { dim: 3 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment: one logit column and one coordinate block");
    let mut term =
        SaeManifoldTerm::new(vec![atom], assignment).expect("the atom matches its block");
    let mut rho = SaeManifoldRho::new(0.0, -1.0, vec![Array1::<f64>::zeros(3)]);
    let cache = fit_and_certify("sphere", &mut term, &target, &mut rho);
    (term, target, rho, cache)
}

/// The exact spectral divergence, its residual dof, and the residual dof the
/// dispersion prices must be the re-solved tangent response's.
#[test]
fn sphere_coordinates_price_their_tangent_response_2933_f39() {
    let (term, target, rho, cache) = sphere_state();
    assert_prices_resolved_response("sphere", &term, &target, &rho, &cache);
}

/// The matrix-free route prices the same tangent response: the output-space
/// residual dof within four standard errors of `‖I − R‖²_F`, and the divergence
/// read off the same probes within four standard errors of `tr R`. Both standard
/// errors are taken from the re-solved response itself
/// ([`rademacher_quadratic_form_variance`]), at the probe count the estimator
/// chose, and that count must meet its own stopping rule: the residual dof's
/// Monte Carlo variance within the `2ν` sampling variance the dispersion carries.
#[test]
fn sphere_hutchinson_response_brackets_the_tangent_response_2933_f39() {
    let (mut term, target, rho, cache) = sphere_state();
    let response = resolved_response(&term, &target, &rho);
    let (trace, residual_dof) = trace_and_residual_dof(&response);
    term.host_available_bytes = 0;
    let priced = term
        .fitted_response_divergence(target.view(), &rho, &cache)
        .expect("the matrix-free estimator solves where the dense route is refused");
    let FittedResponseDivergenceEstimator::Hutchinson { likelihood, .. } = priced.estimator else {
        panic!(
            "without an admitted eigensystem the divergence must be the Hutchinson estimate, got \
             {:?}",
            priced.estimator
        );
    };
    let probes = likelihood.probes;
    let complement = Array2::<f64>::eye(response.nrows()) - &response;
    let gram = complement.t().dot(&complement);
    let residual_dof_standard_error =
        (rademacher_quadratic_form_variance(&gram) / probes as f64).sqrt();
    let divergence_standard_error =
        (rademacher_quadratic_form_variance(&response) / probes as f64).sqrt();
    eprintln!(
        "[#2933 F39 sphere Hutchinson] probes={probes} divergence={:.9e} (sample se {:.3e}, \
         resolved se {divergence_standard_error:.3e}) resolved tr R={trace:.9e}; residual dof \
         {:.9e} (sample se {:.3e}, resolved se {residual_dof_standard_error:.3e}) resolved \
         ‖I−R‖²={residual_dof:.9e}",
        priced.divergence,
        likelihood.divergence_standard_error,
        priced.likelihood_residual_dof,
        likelihood.residual_dof_standard_error
    );
    assert!(
        probes >= 2
            && likelihood.residual_dof_standard_error.powi(2) <= 2.0 * likelihood.residual_dof,
        "the probes stopped at {probes} with a residual-dof Monte Carlo variance {:.3e} above \
         the 2ν̂ = {:.3e} the dispersion carries",
        likelihood.residual_dof_standard_error.powi(2),
        2.0 * likelihood.residual_dof
    );
    assert!(
        (priced.likelihood_residual_dof - residual_dof).abs() <= 4.0 * residual_dof_standard_error,
        "the Hutchinson residual dof {} is more than four standard errors \
         ({residual_dof_standard_error:.3e}) from the re-solved ‖I − R‖²_F {residual_dof}",
        priced.likelihood_residual_dof
    );
    assert!(
        (priced.divergence - trace).abs() <= 4.0 * divergence_standard_error,
        "the Hutchinson divergence {} is more than four standard errors \
         ({divergence_standard_error:.3e}) from the re-solved tangent response {trace}",
        priced.divergence
    );
}

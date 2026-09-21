#![cfg(test)]
//! #2933 F30–F32. The tempered posterior expectation keeps its mean
//! displacement; the rank-charge audit fills every field from one evaluated
//! state; the MP edge's false-rank rate under a fitted noise-only null is
//! measured against the derived conditional law; and a stratum names the
//! boundary a crossing changes, which the outer objective publishes as its
//! criterion rank (#3436).

use super::tests::{TestPeriodicEvaluator, periodic_basis};
use super::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use super::wbic_audit::rank_charge_stratum;
use super::*;
use gam_linalg::utils::splitmix64;
use gam_solve::rho_optimizer::{CriterionRank, OuterEvalOrder, OuterObjective};
use ndarray::array;

fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(f64::MIN_POSITIVE);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Trapezoid rule for `E[½h(w−ŵ)²]` under `exp(−½βh(w−ŵ)² − ½τw²)`. It evaluates
/// only that density, on a window reaching twelve curvature standard deviations
/// past both `0` and `ŵ` (the tempered mean lies between them), where the rule is
/// spectrally accurate for a Gaussian integrand.
fn tempered_excess_by_quadrature_1d(h: f64, tau: f64, mle: f64, beta: f64) -> f64 {
    let log_density = |w: f64| -0.5 * beta * h * (w - mle).powi(2) - 0.5 * tau * w * w;
    let curvature_sd = (beta * h + tau).sqrt().recip();
    let lower = mle.min(0.0) - 12.0 * curvature_sd;
    let upper = mle.max(0.0) + 12.0 * curvature_sd;
    let steps = 20_000usize;
    let width = (upper - lower) / steps as f64;
    let grid = |i: usize| lower + i as f64 * width;
    let peak = (0..=steps)
        .map(|i| log_density(grid(i)))
        .fold(f64::NEG_INFINITY, f64::max);
    let mut mass = 0.0_f64;
    let mut moment = 0.0_f64;
    for i in 0..=steps {
        let w = grid(i);
        let end_weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let density = end_weight * (log_density(w) - peak).exp();
        mass += density;
        moment += density * 0.5 * h * (w - mle).powi(2);
    }
    moment / mass
}

fn two_by_two_extreme_eigenvalues(matrix: &Array2<f64>) -> (f64, f64) {
    let mid = 0.5 * (matrix[[0, 0]] + matrix[[1, 1]]);
    let radius =
        (0.25 * (matrix[[0, 0]] - matrix[[1, 1]]).powi(2) + matrix[[0, 1]].powi(2)).sqrt();
    (mid - radius, mid + radius)
}

fn half_quadratic(matrix: &Array2<f64>, center: &Array1<f64>, w: [f64; 2]) -> f64 {
    let d0 = w[0] - center[0];
    let d1 = w[1] - center[1];
    0.5 * (matrix[[0, 0]] * d0 * d0 + 2.0 * matrix[[0, 1]] * d0 * d1 + matrix[[1, 1]] * d1 * d1)
}

/// Tensor trapezoid rule for `E[½(w−ŵ)ᵀH(w−ŵ)]` under
/// `exp(−½β(w−ŵ)ᵀH(w−ŵ) − ½(w−μ₀)ᵀΛ(w−μ₀))`. The window is centred on `ŵ` and
/// reaches twelve standard deviations of the flattest curvature direction past
/// the largest possible mean shift `‖Λ‖·‖μ₀−ŵ‖/λ_min(βH+Λ)`.
fn tempered_excess_by_quadrature_2d(
    information: &Array2<f64>,
    prior_precision: &Array2<f64>,
    prior_mean: &Array1<f64>,
    minimizer: &Array1<f64>,
    beta: f64,
) -> f64 {
    let precision = information * beta + prior_precision;
    let lambda_min = two_by_two_extreme_eigenvalues(&precision).0;
    let prior_norm = two_by_two_extreme_eigenvalues(prior_precision).1;
    let offset = ((minimizer[0] - prior_mean[0]).powi(2)
        + (minimizer[1] - prior_mean[1]).powi(2))
    .sqrt();
    let half_width = 12.0 * lambda_min.sqrt().recip() + prior_norm * offset / lambda_min;
    let steps = 1600usize;
    let width = 2.0 * half_width / steps as f64;
    let coordinate = |axis: usize, i: usize| minimizer[axis] - half_width + i as f64 * width;
    let log_density = |w: [f64; 2]| {
        -beta * half_quadratic(information, minimizer, w)
            - half_quadratic(prior_precision, prior_mean, w)
    };
    let reference = log_density([minimizer[0], minimizer[1]]);
    let mut mass = 0.0_f64;
    let mut moment = 0.0_f64;
    for i in 0..=steps {
        let first = coordinate(0, i);
        let first_weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        for j in 0..=steps {
            let w = [first, coordinate(1, j)];
            let second_weight = if j == 0 || j == steps { 0.5 } else { 1.0 };
            let density = first_weight * second_weight * (log_density(w) - reference).exp();
            mass += density;
            moment += density * half_quadratic(information, minimizer, w);
        }
    }
    moment / mass
}

#[test]
fn tempered_gaussian_excess_loss_keeps_the_posterior_mean_displacement_2933() {
    // The audit's counterexample: h = τ = β = 1, ŵ = 2. The variance-only soft
    // count gives 0.25; the tempered posterior expectation is 0.75.
    let unit = array![[1.0]];
    let excess = tempered_gaussian_excess_loss(
        unit.view(),
        unit.view(),
        array![0.0].view(),
        array![2.0].view(),
        1.0,
    )
    .expect("proper tempered posterior");
    assert!((excess.posterior_spread - 0.25).abs() < 1.0e-14, "{excess:?}");
    assert!((excess.mean_displacement - 0.5).abs() < 1.0e-14, "{excess:?}");
    assert!((excess.total() - 0.75).abs() < 1.0e-14, "{excess:?}");

    // Closed form and an independent quadrature across temperatures, including
    // Watanabe's β = 1/ln n, and across prior strengths.
    for &(h, tau, mle) in &[(1.0, 1.0, 2.0), (40.0, 0.3, -1.7), (0.2, 5.0, 0.6)] {
        for beta in [1.0 / 1000.0_f64.ln(), 1.0 / 10.0_f64.ln(), 0.5, 1.0, 3.0] {
            let excess = tempered_gaussian_excess_loss(
                array![[h]].view(),
                array![[tau]].view(),
                array![0.0].view(),
                array![mle].view(),
                beta,
            )
            .expect("proper tempered posterior");
            let precision = beta * h + tau;
            let mean = beta * h * mle / precision;
            let closed = 0.5 * h * ((mean - mle).powi(2) + precision.recip());
            let quadrature = tempered_excess_by_quadrature_1d(h, tau, mle, beta);
            assert!(
                (excess.total() - closed).abs() <= 1.0e-12 * closed,
                "h={h} tau={tau} mle={mle} beta={beta}: primitive {excess:?} vs closed form {closed}"
            );
            assert!(
                (excess.total() - quadrature).abs() <= 1.0e-9 * quadrature,
                "h={h} tau={tau} mle={mle} beta={beta}: primitive {excess:?} vs quadrature {quadrature}"
            );
        }
    }

    // Two dimensions, correlated information, a prior that is flat along the
    // first axis and centred off zero along the second.
    let information = array![[3.0, 1.0], [1.0, 2.0]];
    let prior_precision = array![[0.0, 0.0], [0.0, 4.0]];
    let prior_mean = array![0.0, 0.5];
    let minimizer = array![1.0, -2.0];
    for beta in [0.4, 1.3] {
        let excess = tempered_gaussian_excess_loss(
            information.view(),
            prior_precision.view(),
            prior_mean.view(),
            minimizer.view(),
            beta,
        )
        .expect("proper tempered posterior");
        let quadrature = tempered_excess_by_quadrature_2d(
            &information,
            &prior_precision,
            &prior_mean,
            &minimizer,
            beta,
        );
        assert!(
            (excess.total() - quadrature).abs() <= 1.0e-9 * quadrature,
            "beta={beta}: primitive {excess:?} vs quadrature {quadrature}"
        );
    }
    let excess = tempered_gaussian_excess_loss(
        information.view(),
        prior_precision.view(),
        prior_mean.view(),
        minimizer.view(),
        0.4,
    )
    .expect("proper tempered posterior");
    assert!(
        excess.mean_displacement > 0.5 * excess.total(),
        "the displacement term must be material in this fixture: {excess:?}"
    );

    // An improper tempered posterior is refused, not priced.
    let zero = array![[0.0]];
    assert!(
        tempered_gaussian_excess_loss(
            zero.view(),
            zero.view(),
            array![0.0].view(),
            array![1.0].view(),
            1.0
        )
        .is_err()
    );
}

fn resolved_single_atom_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 24usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let decoder = array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]];
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
        target[[row, 1]] += 1.0e-3 * (0.29 * row as f64).cos();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .expect("the periodic fixture has matching basis, jet, decoder and Gram shapes")
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("one softmax block over one circle coordinate is a valid assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("one atom with one matching assignment block is a valid term");
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![250.0_f64.ln()]]);
    (term, target, rho)
}

#[test]
fn rank_charge_audit_fills_every_field_from_one_state_2933() {
    let (mut term, target, rho) = resolved_single_atom_state();
    let loss = term
        .loss(target.view(), &rho)
        .expect("the fixture loss is finite");
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("the fixture arrow system assembles");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("the resolved fixture has a positive definite evidence factor");

    let audits = term
        .rank_charge_audit(target.view(), &rho, &loss, &cache)
        .expect("the plain single-atom state is auditable");
    assert_eq!(audits.len(), 1);
    let audit = &audits[0];
    assert_eq!(
        (
            audit.atom,
            audit.basis_dim,
            audit.output_dim,
            audit.storage_dim,
            audit.intrinsic_dim
        ),
        (0, 3, 2, 6, 1)
    );

    // Tied to the priced state: the same dispersion, the same DOF the criterion
    // charges, and the criterion's charge arithmetic.
    let residual = term
        .reconstruction_residual(target.view(), &rho)
        .expect("fitted and target shapes match");
    let dispersion = term
        .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
        .expect("the fixture dispersion is finite and positive")
        .raw_output_noise_variance;
    assert_eq!(audit.dispersion.to_bits(), dispersion.to_bits());
    let mut grams = term.empty_decoder_gram_accumulator();
    term.accumulate_decoder_gram(&mut grams)
        .expect("the decoder Gram accumulates on the CPU fixture");
    let n_eff = term.per_atom_effective_sample_size();
    let priced = term
        .rank_dof_from_grams(&grams, &n_eff, &rho, dispersion)
        .expect("the priced rank-charge DOF exists at the fixture state");
    assert_eq!(audit.stratum.production_dof().to_bits(), priced.dof[0].to_bits());
    assert_eq!(
        priced.chargeable_rank,
        vec![audit.stratum.production_chargeable_rank()]
    );
    assert_eq!(
        audit.stratum.production_charge().to_bits(),
        (0.5 * priced.dof[0] * n_eff[0].max(1.0).ln()).to_bits()
    );
    assert!((audit.stratum.effective_sample_size() - 24.0).abs() <= 1.0e-12);
    assert!((audit.lambda_smooth - 0.8).abs() <= 1.0e-12);
    assert!((audit.inverse_temperature - 24.0_f64.ln().recip()).abs() <= 1.0e-15);
    assert_eq!(audit.stratum.mp_reconstruction_rank(), 2);
    assert_eq!(audit.stratum.production_chargeable_rank(), 2);
    let energies = audit.stratum.reconstruction_energies();
    let edge = audit.stratum.mp_reconstruction_rank_edge();
    assert_eq!(energies.len(), 2);
    assert!(energies.iter().all(|&energy| energy > edge));

    // Two directions above the edge: the nearest is the smaller one, and its
    // crossing drops the charge by one unit of ½·edf·ln N_eff.
    let boundary = audit
        .nearest_mp_boundary
        .expect("two reconstruction directions");
    assert_eq!(boundary.direction, 1);
    assert_eq!(boundary.signed_gap, energies[1] - edge);
    let unit_jump = 0.5 * audit.stratum.basis_edf() * 24.0_f64.ln();
    assert!((boundary.charge_jump + unit_jump).abs() <= 1.0e-12 * unit_jump);

    // Conditional noise law: a positive expected total energy, and a top-energy
    // bound no smaller than the average over the two nonzero energies.
    assert!(audit.noise_null.expected_total_energy > 0.0);
    assert_eq!(
        audit.noise_null.output_noise,
        OutputNoiseSpectrum::isotropic(dispersion, 2)
    );
    let edge_false_rank_bound = audit
        .noise_null
        .false_rank_probability_bound(edge)
        .expect("the MP edge is a finite non-negative threshold");
    assert_eq!(
        audit.mp_false_rank_probability_bound.to_bits(),
        edge_false_rank_bound.to_bits()
    );
    assert!((0.0..=1.0).contains(&audit.mp_false_rank_probability_bound));
    assert!(
        audit.noise_null.top_energy_expectation_bound
            >= 0.5 * audit.noise_null.expected_total_energy
    );

    // The tempered posterior against an independent route: accumulate the row
    // design and the raw target (K = 1, so the target is the atom's conditional
    // response), solve the normal equations per channel, and hand the stacked
    // (m·p)-dimensional system to the general tempered-Gaussian primitive.
    let atom = &term.atoms[0];
    let phi = &atom.basis_values;
    let gates = term.assignment.assignments();
    let (n, m) = phi.dim();
    let p = target.ncols();
    let mut row_gram = Array2::<f64>::zeros((m, m));
    let mut row_score = Array2::<f64>::zeros((m, p));
    for row in 0..n {
        let gate = gates[[row, 0]];
        for i in 0..m {
            for j in 0..m {
                row_gram[[i, j]] += gate * gate * phi[[row, i]] * phi[[row, j]];
            }
            for c in 0..p {
                row_score[[i, c]] += gate * phi[[row, i]] * target[[row, c]];
            }
        }
    }
    let channel_minimizers = row_gram
        .cholesky(Side::Lower)
        .expect("resolved design")
        .solve_mat(&row_score);
    let dim = m * p;
    let penalty = atom.smooth_penalty();
    let mut information = Array2::<f64>::zeros((dim, dim));
    let mut prior_precision = Array2::<f64>::zeros((dim, dim));
    let mut minimizer = Array1::<f64>::zeros(dim);
    for c in 0..p {
        for i in 0..m {
            minimizer[c * m + i] = channel_minimizers[[i, c]];
            for j in 0..m {
                information[[c * m + i, c * m + j]] = row_gram[[i, j]] / dispersion;
                prior_precision[[c * m + i, c * m + j]] =
                    audit.lambda_smooth * penalty[[i, j]] / dispersion;
            }
        }
    }
    let reference = tempered_gaussian_excess_loss(
        information.view(),
        prior_precision.view(),
        Array1::<f64>::zeros(dim).view(),
        minimizer.view(),
        audit.inverse_temperature,
    )
    .expect("proper tempered posterior");
    let posterior = audit.tempered_posterior;
    assert_eq!(
        posterior.inverse_temperature.to_bits(),
        audit.inverse_temperature.to_bits()
    );
    assert!(
        (posterior.posterior_spread - reference.posterior_spread).abs()
            <= 1.0e-10 * reference.posterior_spread,
        "audit {posterior:?} vs stacked reference {reference:?}"
    );
    assert!(
        (posterior.mean_displacement - reference.mean_displacement).abs()
            <= 1.0e-7 * reference.mean_displacement,
        "audit {posterior:?} vs stacked reference {reference:?}"
    );
    assert!(
        posterior.mean_displacement > 1.0e-3 * posterior.total(),
        "the displacement term must be material in this fixture: {posterior:?}"
    );
    assert_eq!(
        audit.production_minus_tempered.to_bits(),
        (audit.stratum.production_charge() - posterior.total()).to_bits()
    );

    // Where the conditional model is not the data fit, the audit refuses.
    let mut weighted = resolved_single_atom_state().0;
    weighted
        .set_row_loss_weights((0..n).map(|row| if row % 2 == 0 { 1.5 } else { 0.5 }).collect())
        .expect("alternating 1.5/0.5 weights are finite, positive and mean one");
    let refusal = weighted
        .rank_charge_audit(target.view(), &rho, &loss, &cache)
        .expect_err("weighted rows are refused");
    assert!(refusal.contains("row loss weights"), "{refusal}");
}

#[test]
fn rank_charge_audit_reports_the_tangent_dimension_of_an_ambient_sphere_2933() {
    // Tangent dimensions written out from the geometry, not read from the
    // implementation: S¹ is 1; S² and RP² are 2 whether stored as an ambient unit
    // vector or a (lat, lon) chart; T², the cylinder and the Möbius cover are 2;
    // a flat patch keeps its width.
    for (kind, latent_dim, expected) in [
        (SaeAtomBasisKind::Periodic, 1, 1),
        (SaeAtomBasisKind::Sphere, 3, 2),
        (SaeAtomBasisKind::ProjectivePlane, 3, 2),
        (SaeAtomBasisKind::ProjectivePlane, 2, 2),
        (SaeAtomBasisKind::Torus, 2, 2),
        (SaeAtomBasisKind::Cylinder, 2, 2),
        (SaeAtomBasisKind::Mobius, 2, 2),
        (SaeAtomBasisKind::EuclideanPatch, 4, 4),
    ] {
        assert_eq!(
            kind.latent_manifold(latent_dim).intrinsic_dim(latent_dim),
            expected,
            "{kind:?} at coordinate width {latent_dim}"
        );
    }

    // One ambient S² atom on Fibonacci points: three stored coordinates per row,
    // two dimensions.
    let n = 40usize;
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let coords = Array2::from_shape_fn((n, 3), |(row, axis)| {
        let z = 1.0 - (2.0 * row as f64 + 1.0) / n as f64;
        let radius = (1.0 - z * z).sqrt();
        let angle = golden_angle * row as f64;
        match axis {
            0 => radius * angle.cos(),
            1 => radius * angle.sin(),
            _ => z,
        }
    });
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(AmbientSphereHarmonicEvaluator::new(1).expect("degree-one sphere basis"));
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the sphere basis evaluates on unit vectors");
    let width = phi.ncols();
    let decoder = Array2::from_shape_fn((width, 2), |(basis, out)| {
        0.6 * (0.37 * (7 * basis + 3 * out + 1) as f64).sin()
    });
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
        target[[row, 1]] += 1.0e-3 * (0.29 * row as f64).cos();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "sphere",
        SaeAtomBasisKind::Sphere,
        3,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(width),
    )
    .expect("the sphere fixture has matching basis, jet, decoder and Gram shapes")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Sphere { dim: 3 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("one softmax block over one ambient sphere coordinate is a valid assignment");
    let mut term = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("one atom with one matching assignment block is a valid term");
    let rho = SaeManifoldRho::new(
        0.0,
        0.8_f64.ln(),
        vec![array![250.0_f64.ln(), 250.0_f64.ln(), 250.0_f64.ln()]],
    );
    let loss = term
        .loss(target.view(), &rho)
        .expect("the sphere fixture loss is finite");
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("the sphere fixture arrow system assembles");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("the sphere fixture has a positive definite evidence factor");
    let audits = term
        .rank_charge_audit(target.view(), &rho, &loss, &cache)
        .expect("the plain single-sphere state is auditable");
    assert_eq!(audits.len(), 1);
    let audit = &audits[0];
    assert_eq!(term.atoms[0].latent_dim(), 3);
    assert_eq!(
        (
            audit.basis_dim,
            audit.output_dim,
            audit.storage_dim,
            audit.intrinsic_dim
        ),
        (4, 2, 8, 2)
    );
}

#[test]
fn mp_edge_false_rank_rate_under_a_fitted_noise_only_null_2933() {
    let n = 160usize;
    let m = 5usize;
    let p = 6usize;
    let dispersion = 0.7_f64;
    let lambda = 3.0_f64;
    let draws = 3000usize;
    let phi = Array2::from_shape_fn((n, m), |(row, col)| {
        let t = (row as f64 + 0.5) / n as f64;
        (std::f64::consts::PI * col as f64 * t).cos()
    });
    let mut difference = Array2::<f64>::zeros((m - 2, m));
    for i in 0..m - 2 {
        difference[[i, i]] = 1.0;
        difference[[i, i + 1]] = -2.0;
        difference[[i, i + 2]] = 1.0;
    }
    let penalty = difference.t().dot(&difference);
    let mut state = 0x2933_3032_2933_3032_u64;
    // Nonuniform gates, then the same gates rescaled.
    for gate_scale in [1.0_f64, 0.4] {
        let gates = Array1::from_shape_fn(n, |row| {
            gate_scale * (0.25 + 0.75 * ((row * 37) % n) as f64 / n as f64)
        });
        let design = Array2::from_shape_fn((n, m), |(row, col)| gates[row] * phi[[row, col]]);
        let gram = design.t().dot(&design);
        let n_eff = gates.iter().map(|gate| gate * gate).sum::<f64>();
        let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p as f64, dispersion)
            .expect("positive occupancy, width and dispersion give a finite edge");
        let null = conditional_noise_null(
            &gram,
            n_eff,
            p,
            OutputNoiseSpectrum::isotropic(dispersion, p),
            lambda,
            Some(&penalty),
        )
        .expect("conditional noise-only law");
        let ridge = (&gram + &(&penalty * lambda))
            .cholesky(Side::Lower)
            .expect("penalized Gram is positive definite");
        let noise_sd = dispersion.sqrt();
        let mut totals = Vec::with_capacity(draws);
        let mut top_sum = 0.0_f64;
        let mut mp_false_ranks = 0usize;
        let mut chargeable_false_ranks = 0usize;
        for _ in 0..draws {
            let noise =
                Array2::from_shape_simple_fn((n, p), || noise_sd * standard_normal(&mut state));
            // The fitted decoder of a noise-only target under the fixed design.
            let decoder = ridge.solve_mat(&design.t().dot(&noise));
            let stratum = rank_charge_stratum(
                &gram,
                &decoder,
                n_eff,
                p as f64,
                dispersion,
                lambda,
                Some(&penalty),
            )
            .expect("noise decoder stratum");
            totals.push(stratum.reconstruction_energies().iter().sum::<f64>());
            top_sum += stratum.top_reconstruction_energy();
            mp_false_ranks += usize::from(stratum.mp_reconstruction_rank() > 0);
            chargeable_false_ranks += usize::from(stratum.production_chargeable_rank() > 0);
        }
        let count = draws as f64;
        let mean_total = totals.iter().sum::<f64>() / count;
        let variance =
            totals.iter().map(|total| (total - mean_total).powi(2)).sum::<f64>() / (count - 1.0);
        let standard_error = (variance / count).sqrt();
        let mean_top = top_sum / count;
        let mp_rate = mp_false_ranks as f64 / count;
        let chargeable_rate = chargeable_false_ranks as f64 / count;
        eprintln!(
            "#2933 F32 fitted noise-only null: gate_scale={gate_scale} N_eff={n_eff:.4} \
             edge={edge:.6e} expected_total={:.6e} mc_total={mean_total:.6e} (se {standard_error:.2e}) \
             top_bound={:.6e} mc_top={mean_top:.6e} mp_false_rank_rate={mp_rate:.4} \
             chargeable_false_rank_rate={chargeable_rate:.4}",
            null.expected_total_energy, null.top_energy_expectation_bound
        );
        assert!(
            standard_error < 0.05 * null.expected_total_energy,
            "the Monte Carlo sample must resolve the expected energy"
        );
        assert!(
            (mean_total - null.expected_total_energy).abs() <= 4.0 * standard_error,
            "gate_scale={gate_scale}: mc total energy {mean_total} (se {standard_error}) vs the \
             exact conditional expectation {}",
            null.expected_total_energy
        );
        assert!(
            mean_top <= null.top_energy_expectation_bound,
            "gate_scale={gate_scale}: mc top energy {mean_top} exceeds its bound {}",
            null.top_energy_expectation_bound
        );
        // Markov's inequality on the top energy bounds the MP false-rank rate.
        let markov = (null.top_energy_expectation_bound / edge).min(1.0);
        assert!(
            mp_rate <= markov + 4.0 * (markov * (1.0 - markov) / count).sqrt(),
            "gate_scale={gate_scale}: MP false-rank rate {mp_rate} vs Markov bound {markov}"
        );
        // Every alive noise decoder is charged at least rank one (#2258).
        assert_eq!(chargeable_false_ranks, draws);
    }
}

/// Bernstein's one-sided excess: a binomial count of `count` trials with success
/// probability at most `probability` exceeds `count·probability + t` with
/// probability at most `exp(−t² / (2(count·p(1−p) + t/3)))`. Returns the rate
/// excess `t / count` at which that tail probability is `exp(−log_inverse_level)`.
fn bernstein_rate_excess(count: usize, probability: f64, log_inverse_level: f64) -> f64 {
    let variance = count as f64 * probability * (1.0 - probability);
    let third = log_inverse_level / 3.0;
    (third + (third * third + 2.0 * variance * log_inverse_level).sqrt()) / count as f64
}

/// Thresholds at the `quantiles` of a pilot sample of top energies.
fn pilot_thresholds(pilot: &[f64], quantiles: &[f64]) -> Vec<f64> {
    let mut sorted = pilot.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantiles
        .iter()
        .map(|&quantile| {
            let index = ((sorted.len() as f64 * quantile).floor() as usize).min(sorted.len() - 1);
            sorted[index]
        })
        .collect()
}

#[test]
fn conditional_false_rank_probability_bound_covers_the_noise_only_tail_2933() {
    let n = 160usize;
    let m = 5usize;
    let pilot_draws = 1000usize;
    let draws = 3000usize;
    // Each assertion may fail by chance with probability at most e^-14 ≈ 8e-7.
    let log_inverse_level = 14.0_f64;
    let dispersion = 0.7_f64;
    let phi = Array2::from_shape_fn((n, m), |(row, col)| {
        let t = (row as f64 + 0.5) / n as f64;
        (std::f64::consts::PI * col as f64 * t).cos()
    });
    let mut difference = Array2::<f64>::zeros((m - 2, m));
    for i in 0..m - 2 {
        difference[[i, i]] = 1.0;
        difference[[i, i + 1]] = -2.0;
        difference[[i, i + 2]] = 1.0;
    }
    let penalty = difference.t().dot(&difference);
    let gates = Array1::from_shape_fn(n, |row| 0.25 + 0.75 * ((row * 37) % n) as f64 / n as f64);
    let design = Array2::from_shape_fn((n, m), |(row, col)| gates[row] * phi[[row, col]]);
    let gram = design.t().dot(&design);
    let n_eff = gates.iter().map(|gate| gate * gate).sum::<f64>();
    let quantiles = [0.5, 0.9, 0.99, 0.999];
    let mut state = 0x2933_3032_b0_u64;
    // (label, output width, equicorrelation c, smoothing λ). Row noise is
    // N(0, R[(1 − c)I + c·11ᵀ]): trace p·R, largest eigenvalue R(1 − c + c·p).
    for (label, p, correlation, lambda) in [
        ("isotropic unpenalized", 6usize, 0.0_f64, 0.0_f64),
        ("isotropic", 6, 0.0, 3.0),
        ("isotropic heavily smoothed", 6, 0.0, 300.0),
        ("equicorrelated", 12, 0.9, 3.0),
    ] {
        let covariance_spectrum = OutputNoiseSpectrum {
            trace: p as f64 * dispersion,
            largest_eigenvalue: dispersion * (1.0 - correlation + correlation * p as f64),
        };
        let law = conditional_noise_null(
            &gram,
            n_eff,
            p,
            covariance_spectrum,
            lambda,
            Some(&penalty),
        )
        .expect("conditional noise-only law");
        // What the audit assumes from the scalar raw dispersion alone.
        let scalar_law = conditional_noise_null(
            &gram,
            n_eff,
            p,
            OutputNoiseSpectrum::isotropic(dispersion, p),
            lambda,
            Some(&penalty),
        )
        .expect("scalar conditional noise-only law");
        let ridge = (&gram + &(&penalty * lambda))
            .cholesky(Side::Lower)
            .expect("penalized Gram is positive definite");
        let top_energy = |state: &mut u64| {
            let mut noise = Array2::<f64>::zeros((n, p));
            for row in 0..n {
                let shared = standard_normal(state);
                for col in 0..p {
                    noise[[row, col]] = dispersion.sqrt()
                        * ((1.0 - correlation).sqrt() * standard_normal(state)
                            + correlation.sqrt() * shared);
                }
            }
            let decoder = ridge.solve_mat(&design.t().dot(&noise));
            rank_charge_stratum(
                &gram,
                &decoder,
                n_eff,
                p as f64,
                dispersion,
                lambda,
                Some(&penalty),
            )
            .expect("noise decoder stratum")
            .top_reconstruction_energy()
        };
        // Thresholds come from an independent pilot sample, so each tail count
        // below is binomial.
        let pilot: Vec<f64> = (0..pilot_draws).map(|_| top_energy(&mut state)).collect();
        let thresholds = pilot_thresholds(&pilot, &quantiles);
        let tops: Vec<f64> = (0..draws).map(|_| top_energy(&mut state)).collect();
        let mut informative = false;
        let mut scalar_law_violated = false;
        for &threshold in &thresholds {
            let rate = tops.iter().filter(|&&top| top > threshold).count() as f64 / draws as f64;
            let bound = law
                .false_rank_probability_bound(threshold)
                .expect("finite threshold");
            let scalar_bound = scalar_law
                .false_rank_probability_bound(threshold)
                .expect("finite threshold");
            eprintln!(
                "#2933 F32 conditional false-rank bound [{label}]: N_eff={n_eff:.4} \
                 tau={threshold:.6e} mc_rate={rate:.4} bound={bound:.6e} \
                 scalar_R_bound={scalar_bound:.6e}"
            );
            assert!(
                rate <= bound + bernstein_rate_excess(draws, bound, log_inverse_level),
                "[{label}] tau={threshold}: noise-only rate {rate} exceeds the conditional \
                 bound {bound}"
            );
            informative |= rate > 0.0 && bound < 1.0;
            scalar_law_violated |=
                rate > scalar_bound + bernstein_rate_excess(draws, scalar_bound, log_inverse_level);
        }
        assert!(
            informative,
            "[{label}]: the bound must be below one at a threshold noise actually crosses"
        );
        if correlation > 0.0 {
            assert!(
                scalar_law_violated,
                "[{label}]: correlated output noise must exceed what the scalar-R law permits"
            );
        }
    }
}

fn decoder_with_energies(edge: f64, first: f64, second: f64, p: usize) -> Array2<f64> {
    let mut decoder = Array2::<f64>::zeros((2, p));
    decoder[[0, 0]] = (first * edge).sqrt();
    decoder[[1, 1]] = (second * edge).sqrt();
    decoder
}

#[test]
fn rank_charge_stratum_names_the_mp_edge_branch_2933() {
    let n_eff = 50.0_f64;
    let p = 3usize;
    let dispersion = 1.0_f64;
    // With G = N_eff·I the reconstruction energies are the squared decoder
    // singular values.
    let gram = Array2::<f64>::eye(2) * n_eff;
    let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p as f64, dispersion)
        .expect("positive occupancy, width and dispersion give a finite edge");
    let stratum_at = |first: f64, second: f64| {
        rank_charge_stratum(
            &gram,
            &decoder_with_energies(edge, first, second, p),
            n_eff,
            p as f64,
            dispersion,
            0.0,
            None,
        )
        .expect("stratum")
    };
    let log_n = n_eff.ln();

    // One direction above the edge and one just below it.
    let resolved = stratum_at(1.05, 0.99);
    assert_eq!(
        (resolved.mp_reconstruction_rank(), resolved.production_chargeable_rank()),
        (1, 1)
    );
    let boundary = resolved.nearest_mp_boundary().expect("two directions");
    assert_eq!(boundary.direction, 1);
    assert!((resolved.reconstruction_energies()[1] - 0.99 * edge).abs() <= 1.0e-9 * edge);
    assert!(boundary.signed_gap < 0.0);
    let unit_jump = 0.5 * resolved.basis_edf() * log_n;
    assert!((boundary.charge_jump - unit_jump).abs() <= 1.0e-12 * unit_jump);
    // Inside the stratum the charge does not move with the decoder.
    let same_stratum = stratum_at(1.08, 0.97);
    assert_eq!(
        same_stratum.production_charge().to_bits(),
        resolved.production_charge().to_bits()
    );
    // Across the boundary it jumps by exactly the reported amount.
    let crossed = stratum_at(1.05, 1.01);
    assert_eq!(crossed.production_chargeable_rank(), 2);
    assert!(
        (crossed.production_charge() - resolved.production_charge() - boundary.charge_jump).abs()
            <= 1.0e-12 * unit_jump
    );

    // The promotion boundary: the only counted direction falling below the edge
    // leaves the chargeable rank at one, so that crossing adds no charge.
    let single = stratum_at(1.01, 0.5);
    let boundary = single.nearest_mp_boundary().expect("two directions");
    assert_eq!(boundary.direction, 0);
    assert!(boundary.signed_gap > 0.0);
    assert_eq!(boundary.charge_jump, 0.0);
    let promoted = stratum_at(0.99, 0.5);
    assert_eq!(
        (promoted.mp_reconstruction_rank(), promoted.production_chargeable_rank()),
        (0, 1)
    );
    assert_eq!(
        promoted.production_charge().to_bits(),
        single.production_charge().to_bits()
    );

    // The value path prices the branch the stratum names.
    let decoder = decoder_with_energies(edge, 1.05, 0.99, p);
    let priced = realised_rank_charge_dof(&gram, &decoder, n_eff, p as f64, dispersion, 0.0, None)
        .expect("the diagonal fixture prices a finite DOF");
    assert_eq!(priced.to_bits(), resolved.production_dof().to_bits());
}

/// #3436 — a reconstruction energy at its Marchenko--Pastur edge is a stratum
/// boundary of the criterion, and the outer objective says which side a value
/// was priced on.
///
/// The rank charge `Σ_k ½·r_k·edf_k·log N_eff,k` is piecewise constant in the
/// chargeable ranks `r_k`, and `r_k` counts the energies `μ_j > e(R)` with
/// `e(R) = R·(1+√(p/N_eff))²`. Holding the state and moving the dispersion across
/// `R* = μ_min/(1+√(p/N_eff))²` by one part in 10⁹ moves the edge across the
/// smaller energy and nothing else: `r` drops from 2 to 1 and the priced DOF by
/// exactly one `edf`. Two values that far apart in `R` are on two smooth pieces,
/// so the per-atom ranks they publish must differ, which is what makes the outer
/// line search refuse to compare them.
#[test]
fn an_energy_at_its_mp_edge_splits_the_rank_charge_into_two_strata_3436() {
    let (mut term, target, rho) = resolved_single_atom_state();
    let loss = term
        .loss(target.view(), &rho)
        .expect("the fixture loss is finite");
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("the fixture arrow system assembles");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("the resolved fixture has a positive definite evidence factor");
    let audit = term
        .rank_charge_audit(target.view(), &rho, &loss, &cache)
        .expect("the plain single-atom state is auditable")
        .remove(0);
    let mut grams = term.empty_decoder_gram_accumulator();
    term.accumulate_decoder_gram(&mut grams)
        .expect("the decoder Gram accumulates on the CPU fixture");
    let n_eff = term.per_atom_effective_sample_size();
    let smallest_energy = audit
        .stratum
        .reconstruction_energies()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let unit_edge = crate::null_battery::mp_reconstruction_rank_edge(
        audit.stratum.effective_sample_size(),
        audit.output_dim as f64,
        1.0,
    )
    .expect("positive occupancy and width give a finite edge");
    let edge_dispersion = smallest_energy / unit_edge;
    assert!(edge_dispersion.is_finite() && edge_dispersion > 0.0);

    let price_at = |dispersion: f64| {
        term.rank_dof_from_grams(&grams, &n_eff, &rho, dispersion)
            .expect("the rank charge prices at a positive dispersion")
    };
    let below = price_at(edge_dispersion * (1.0 - 1.0e-9));
    let above = price_at(edge_dispersion * (1.0 + 1.0e-9));
    assert_eq!(below.chargeable_rank, vec![2]);
    assert_eq!(above.chargeable_rank, vec![1]);
    let edf = audit.stratum.basis_edf();
    assert!(
        (below.dof[0] - above.dof[0] - edf).abs() <= 1.0e-12 * edf,
        "the crossing moves the DOF by one edf: below {} above {} edf {edf}",
        below.dof[0],
        above.dof[0]
    );
    let below_stratum = CriterionRank::per_component(below.chargeable_rank.clone());
    let above_stratum = CriterionRank::per_component(above.chargeable_rank.clone());
    assert_ne!(below_stratum, above_stratum);
    // Each side is one stratum: a second price on the same side names the same one.
    assert_eq!(
        CriterionRank::per_component(price_at(edge_dispersion * (1.0 - 1.0e-6)).chargeable_rank),
        below_stratum
    );
    assert_eq!(
        CriterionRank::per_component(price_at(edge_dispersion * (1.0 + 1.0e-6)).chargeable_rank),
        above_stratum
    );
}

/// #3436 — every outer lane publishes the rank-charge branch of the value it
/// returned, read off the state that value was priced on: the value lane, the
/// gradient lane that differentiates the value lane's handed-off state, and
/// nothing after a reset.
#[test]
fn the_outer_objective_publishes_the_branch_its_value_was_priced_on_3436() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let rho_flat = rho.flat_coordinates();
    let audit = || {
        SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .for_installed_state_audit()
    };

    // The priced branch of the fixture's own state, through the dense route.
    let mut fresh = audit();
    let route_rho = fresh.baseline_rho.clone();
    fresh
        .evaluate_outer_criterion_route(&route_rho, true, false)
        .expect("the route prices the fixture at its own rho");
    let expected = CriterionRank::per_component(
        fresh
            .term
            .priced_rank_stratum
            .as_ref()
            .expect("a priced value records its branch")
            .to_vec(),
    );
    assert_eq!(expected.components().len(), fresh.term.k_atoms());

    let mut objective = audit();
    assert_eq!(objective.criterion_rank(), None);
    let cost = objective
        .eval_cost(&rho_flat)
        .expect("the value lane prices the fixture at its own rho");
    assert!(cost.is_finite());
    assert_eq!(objective.criterion_rank(), Some(expected.clone()));
    let sample = objective
        .eval(&rho_flat)
        .expect("the gradient lane differentiates the handed-off state");
    assert_eq!(sample.cost.to_bits(), cost.to_bits());
    assert_eq!(objective.criterion_rank(), Some(expected.clone()));
    let probe = objective
        .eval_with_order(&rho_flat, OuterEvalOrder::Value)
        .expect("the line-search value order prices the fixture");
    assert!(probe.cost.is_finite());
    assert_eq!(objective.criterion_rank(), Some(expected));
    objective.reset();
    assert_eq!(objective.criterion_rank(), None);
}

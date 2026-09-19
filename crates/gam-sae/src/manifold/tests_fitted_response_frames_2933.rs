#![cfg(test)]
//! #2933 F39 — a learned Grassmann decoder frame prices its integrated response.
//!
//! A framed decoder `B = C·Uᵀ` is fitted by alternating the border Newton solve in
//! `C` with the closed-form polar refresh of `U`, so perturbing the data moves the
//! frame as well as the coordinates. The dispersion used to hold every frame at
//! its fitted orientation and charge its `r·(p − r)` tangent dimensions as fully
//! determined directions, an upper bound rather than the response. The oracle here
//! re-solves each perturbed fit to the joint fixed point of both blocks and
//! differences every fitted scalar, so it measures the frame-integrated response
//! without reading the operator.
use super::construction::{
    FittedResponseFrame, ShapeInformationRoute, exact_a_pencil_decompositions_on_this_thread,
    hutchinson_residual_dof_resolved,
};
use super::tests_fitted_response_edf_2933::{
    ROOT_GRADIENT_CEILING, polish_to_root, root_norm, trace_and_residual_dof,
};
use super::*;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};

/// Central-difference step on a target entry of an order-one target.
const FD_STEP: f64 = 1.0e-4;

/// Relative agreement required between a priced quantity and its re-solved value.
const RELATIVE_TOLERANCE: f64 = 1.0e-4;

/// Largest number of polish-then-refresh alternations one re-solve may take.
const FRAME_ALTERNATIONS: usize = 60;

/// A frame refresh is at its fixed point once it moves no decoder coefficient by
/// more than this against order-one decoders, far below the `FD_STEP ×
/// RELATIVE_TOLERANCE` resolution the response comparison is held to.
const FRAME_FIXED_POINT_TOLERANCE: f64 = 1.0e-12;

/// One periodic atom `m(t) = B·[1, sin 2πt, cos 2πt]` in `p = 12` outputs whose
/// decoder lies in the span of the first `rank` output axes, so the fit activates
/// a rank-`rank` frame on that span (the #2933 F35 fixture). `off_span_constant`
/// adds a constant along output axis 2 at rank 2, a component outside the frame
/// span. It does not bind the rank constraint: the joint root absorbs it (job 1280343
/// measured `‖∇_B L·U⊥‖_F = 1.58e-4` against `‖∇_B L‖_F = 1.34`), so the bilinear
/// cross curvature `E` is not material in any arm of this fixture.
fn framed_circle(
    rank: usize,
    off_span_constant: f64,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    framed_circle_in(12, rank, off_span_constant)
}

/// [`framed_circle`] in `p` outputs: the decoder, the target's signal and its
/// perturbations live on the same first three output axes, and the remaining axes
/// carry no data, so only the frame's complement grows with `p`.
fn framed_circle_in(
    p: usize,
    rank: usize,
    off_span_constant: f64,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (n, m) = (24usize, 3usize);
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).expect("periodic basis"));
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("periodic jets");
    let mut decoder = Array2::<f64>::zeros((m, p));
    decoder[[1, 0]] = 0.9;
    decoder[[1, 1]] = 0.2;
    decoder[[2, 0]] = -0.1;
    decoder[[2, 1]] = 0.8;
    if rank == 3 {
        decoder[[0, 2]] = 0.35;
    }
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        let x = row as f64;
        target[[row, 0]] += 0.02 * (1.7 * x).sin();
        target[[row, 1]] += 0.02 * (1.3 * x).cos();
        if rank == 3 {
            target[[row, 2]] += 0.02 * (0.9 * x).sin();
        } else {
            target[[row, 2]] += off_span_constant;
        }
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "framed_circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("atom shapes agree")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment shapes agree");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(
        0.0,
        0.8_f64.ln(),
        vec![Array1::from_vec(vec![250.0_f64.ln()])],
    );
    (term, target, rho)
}

/// Alternate the exact-A polish of the border and coordinates with the polar
/// frame refresh until the refresh is at its fixed point and the polish is at its
/// root. Returns the evidence factorization at that root, the root's gradient
/// norm and the last refresh's largest decoder change.
fn framed_root(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (ArrowFactorCache, f64, f64) {
    let mut change = f64::INFINITY;
    for _ in 0..FRAME_ALTERNATIONS {
        let (cache, trajectory) = polish_to_root(term, target.view(), rho);
        let norm = root_norm(&trajectory);
        if change <= FRAME_FIXED_POINT_TOLERANCE {
            return (cache, norm, change);
        }
        let before: Vec<Array2<f64>> = term
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients().to_owned())
            .collect();
        term.refresh_active_frames_from_data(target.view())
            .expect("the polar frame refresh runs at every alternation");
        change = term
            .atoms
            .iter()
            .zip(&before)
            .flat_map(|(atom, previous)| {
                atom.decoder_coefficients()
                    .iter()
                    .zip(previous.iter())
                    .map(|(now, then)| (now - then).abs())
                    .collect::<Vec<f64>>()
            })
            .fold(0.0_f64, f64::max);
    }
    let (cache, trajectory) = polish_to_root(term, target.view(), rho);
    (cache, root_norm(&trajectory), change)
}

/// `R = ∂f̂/∂y` over the `n·p` scalars by central differences of framed fits
/// re-solved to their joint fixed points under the base term's declared gates.
fn resolved_framed_response(
    base: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Array2<f64> {
    let (n, p) = target.dim();
    let gates = base.collapse_prevention_gates();
    let mut response = Array2::<f64>::zeros((n * p, n * p));
    for row in 0..n {
        for col in 0..p {
            let mut fitted = Vec::with_capacity(2);
            for sign in [1.0_f64, -1.0] {
                let mut term = base.clone();
                term.declare_collapse_prevention_gates(&gates);
                let mut perturbed = target.clone();
                perturbed[[row, col]] += sign * FD_STEP;
                let (_, norm, change) = framed_root(&mut term, &perturbed, rho);
                assert!(
                    norm <= ROOT_GRADIENT_CEILING && change <= FRAME_FIXED_POINT_TOLERANCE,
                    "the framed re-solve at entry ({row}, {col}), side {sign} stopped at \
                     ‖g‖={norm:.3e}, frame change {change:.3e}"
                );
                fitted.push(
                    term.try_fitted_for_rho(rho)
                        .expect("the re-solved framed fit reconstructs"),
                );
            }
            for (index, (plus, minus)) in fitted[0].iter().zip(fitted[1].iter()).enumerate() {
                response[[index, row * p + col]] = (plus - minus) / (2.0 * FD_STEP);
            }
        }
    }
    response
}

fn assert_agrees(label: &str, value: f64, resolved: f64) {
    let gap = (value - resolved).abs();
    assert!(
        gap <= RELATIVE_TOLERANCE * resolved.abs().max(1.0),
        "{label}: {value:.9e} against the re-solved response {resolved:.9e} (gap {gap:.3e})"
    );
}

/// #2933 F39 — at rank 2 on a 3-column basis the frame orientation carries a
/// response the fixed-frame divergence omits: with the target in the frame span,
/// and with an off-span constant, which the joint root absorbs. At rank 3
/// the frame is a pure factorization gauge. In each, the divergence, its residual
/// dof and the residual dof the dispersion prices must be the re-solved response's,
/// priced integrated over the frame, with no count charged for it. Each arm prints
/// `‖∇_B L·U⊥‖_F` at the root, the normal gradient the cross curvature `E` is built
/// from, so a record can say whether `E` was material in that arm, and the elapsed
/// time of the oracle, of the frame-integrated divergence and of the fixed-frame
/// geometry the criterion already forms, so the integrated route's extra cost is
/// measured.
#[test]
fn learned_frames_price_their_integrated_response_2933_f39() {
    for (label, rank, off_span_constant) in [
        ("rank 2 in span", 2usize, 0.0_f64),
        ("rank 2 with an absorbed off-span constant", 2, 0.3),
        ("rank 3 gauge", 3, 0.0),
    ] {
        let (mut term, target, rho) = framed_circle(rank, off_span_constant);
        term.recompute_joint_shape_uncertainty(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .unwrap_or_else(|error| panic!("{label}: the framed fixture fits: {error}"));
        let frame_rank = term.atoms[0]
            .decoder_frame
            .as_ref()
            .map(|frame| frame.rank());
        assert_eq!(
            frame_rank,
            Some(rank),
            "{label}: the fit must activate a frame carrying the decoder's rank"
        );
        let gates = term.collapse_prevention_gates();
        term.declare_collapse_prevention_gates(&gates);
        let (cache, norm, change) = framed_root(&mut term, &target, &rho);
        assert!(
            norm <= ROOT_GRADIENT_CEILING && change <= FRAME_FIXED_POINT_TOLERANCE,
            "{label}: the framed fixture stopped at ‖g‖={norm:.3e}, frame change {change:.3e}"
        );
        let oracle_started = std::time::Instant::now();
        let (trace, residual_dof) =
            trace_and_residual_dof(&resolved_framed_response(&term, &target, &rho));
        let oracle_seconds = oracle_started.elapsed().as_secs_f64();
        let integrated_started = std::time::Instant::now();
        let priced = term
            .fitted_response_divergence(target.view(), &rho, &cache)
            .expect("the framed state admits a fitted-response divergence");
        let integrated_seconds = integrated_started.elapsed().as_secs_f64();
        let fixed_frame_started = std::time::Instant::now();
        term.materialize_exact_stationarity_geometry(&rho, target.view(), &cache)
            .expect("the framed state materializes its fixed-frame geometry");
        let fixed_frame_seconds = fixed_frame_started.elapsed().as_secs_f64();
        // The data gradient of the single ungated atom is `∇_B L = Φᵀ·r` (the softmax
        // gate of one atom is identically one and no row carries a weight). Its
        // smoothness gradient `λ·S·B` has no normal part, because `B·U⊥ = 0`.
        let frame = term.atoms[0]
            .decoder_frame
            .as_ref()
            .expect("the framed state carries its frame")
            .frame()
            .to_owned();
        let residual_at_root = term
            .reconstruction_residual(target.view(), &rho)
            .expect("the framed state has a residual");
        let data_gradient = term.atoms[0].basis_values.t().dot(&residual_at_root);
        let normal_gradient = &data_gradient - &data_gradient.dot(&frame).dot(&frame.t());
        let frobenius = |matrix: &Array2<f64>| matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
        eprintln!(
            "[#2933 F39 frames {label}] ‖∇_B L·U⊥‖_F={:.3e} of ‖∇_B L‖_F={:.3e}; oracle \
             {oracle_seconds:.2} s, integrated divergence {integrated_seconds:.3} s, fixed-frame \
             geometry {fixed_frame_seconds:.3} s",
            frobenius(&normal_gradient),
            frobenius(&data_gradient)
        );
        let loss = term.loss(target.view(), &rho).expect("the framed state has a loss");
        let residual = term
            .reconstruction_residual(target.view(), &rho)
            .expect("the framed state has a residual");
        let dispersion = term
            .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
            .expect("the framed state prices a dispersion");
        let priced_residual_dof = 2.0 * loss.data_fit / dispersion.raw_output_noise_variance;
        eprintln!(
            "[#2933 F39 frames {label}] N={} resolved tr R={trace:.9e} ‖I−R‖²={residual_dof:.9e}; \
             divergence {:.9e} ({:?}, {:?}), residual dof {:.9e}, dispersion RSS/φ \
             {priced_residual_dof:.9e}, frame dimension {}",
            target.len(),
            priced.divergence,
            priced.estimator,
            priced.frame_conditioning,
            priced.likelihood_residual_dof,
            term.grassmann_evidence_dimension()
        );
        assert!(
            trace > 1.0 && residual_dof > 1.0,
            "{label}: the re-solved response (tr R {trace}, ‖I−R‖² {residual_dof}) must be \
             material and leave residual dof"
        );
        assert_agrees(&format!("{label} divergence"), priced.divergence, trace);
        assert_agrees(
            &format!("{label} residual dof"),
            priced.likelihood_residual_dof,
            residual_dof,
        );
        assert_agrees(
            &format!("{label} residual dof the dispersion prices"),
            priced_residual_dof,
            residual_dof,
        );
        assert_eq!(
            priced.frame_conditioning,
            SaeFrameConditioning::MarginalOverLearnedFrames,
            "{label}: a small framed fit must integrate its frames"
        );
    }
}

/// `Var(zᵀMz)` for a Rademacher `z`: `2(‖S‖²_F − Σᵢ Sᵢᵢ²)` with `S = (M + Mᵀ)/2`, the
/// variance of one probe of `tr M`.
pub(super) fn rademacher_quadratic_form_variance(matrix: &Array2<f64>) -> f64 {
    let symmetric = (matrix + &matrix.t()) * 0.5;
    let frobenius = symmetric.iter().map(|value| value * value).sum::<f64>();
    let diagonal = (0..symmetric.nrows())
        .map(|index| symmetric[[index, index]] * symmetric[[index, index]])
        .sum::<f64>();
    2.0 * (frobenius - diagonal)
}

/// #2933 F39 — the probe count bounds the Monte Carlo variance from above instead
/// of reading it off the sample. The stream is a Hutchinson-shaped term with a
/// known law, `X = (ν/6)·χ²₆`: mean `ν = 100` and variance `ν²/3`, a residual dof
/// whose `2ν` target needs about `ν/6 ≈ 17` probes. Its first two draws, 30 and 40,
/// sit close together and low. The rule that compares their sample variance with
/// the target stops there, more than four target deviations `√(2ν)` from `ν`. The
/// upper confidence bound keeps probing, and where it stops the mean is within four
/// target deviations of `ν`.
#[test]
fn the_probe_count_keeps_probing_past_an_under_read_variance_2933_f39() {
    const MEAN: f64 = 100.0;
    const CHI_SQUARED_DEGREES: usize = 6;
    let target_deviation = (2.0 * MEAN).sqrt();
    let mut values = vec![30.0_f64, 40.0];
    let early_mean = 0.5 * (values[0] + values[1]);
    let early_variance = (values[0] - values[1]).powi(2) / 2.0;
    assert!(
        early_variance / 2.0 <= 2.0 * early_mean
            && (early_mean - MEAN).abs() > 4.0 * target_deviation,
        "the fixture's first two draws must stop the sample-variance rule far from ν: mean \
         {early_mean}, sample variance {early_variance}"
    );
    assert!(
        !hutchinson_residual_dof_resolved(&values).expect("the rule evaluates"),
        "two close draws must not bound the probe variance within the target"
    );
    let mut state = 0x2933_F39D_u64;
    let mut uniform = || {
        let bits = gam_linalg::utils::splitmix64(&mut state);
        ((bits >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let stopped_at = loop {
        // `−2 ln U` of a uniform `U` is a χ²₂ draw, so three of them sum to χ²₆.
        let chi_squared: f64 = (0..CHI_SQUARED_DEGREES / 2)
            .map(|_| -2.0 * uniform().ln())
            .sum();
        values.push(MEAN / CHI_SQUARED_DEGREES as f64 * chi_squared);
        if hutchinson_residual_dof_resolved(&values).expect("the rule evaluates") {
            break values.len();
        }
        assert!(
            values.len() < 100_000,
            "the rule never bounded the variance of a finite-variance stream"
        );
    };
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    eprintln!(
        "[#2933 F39 probe count] the sample-variance rule stops at 2 with mean {early_mean}; the \
         bound stops at {stopped_at} with mean {mean:.3} against ν = {MEAN} (target deviation \
         {target_deviation:.3})"
    );
    assert!(
        (mean - MEAN).abs() <= 4.0 * target_deviation,
        "where the bound stops, the mean {mean} is more than four target deviations \
         ({target_deviation:.3}) from ν = {MEAN}"
    );
}

/// The rank-2 framed fixture of [`learned_frames_price_their_integrated_response_2933_f39`] in
/// `p` outputs at its joint root, with the frame active and its evidence factor.
fn framed_state(
    p: usize,
    off_span_constant: f64,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, ArrowFactorCache) {
    let (mut term, target, rho) = framed_circle_in(p, 2, off_span_constant);
    term.recompute_joint_shape_uncertainty(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
        .expect("the framed fixture fits");
    assert_eq!(
        term.atoms[0].decoder_frame.as_ref().map(|frame| frame.rank()),
        Some(2),
        "the fit must activate a rank-2 frame"
    );
    let gates = term.collapse_prevention_gates();
    term.declare_collapse_prevention_gates(&gates);
    let (cache, norm, change) = framed_root(&mut term, &target, &rho);
    assert!(
        norm <= ROOT_GRADIENT_CEILING && change <= FRAME_FIXED_POINT_TOLERANCE,
        "the framed fixture stopped at ‖g‖={norm:.3e}, frame change {change:.3e}"
    );
    (term, target, rho, cache)
}

/// The largest carried host reading at which [`sae_exact_stationarity_admitted`]
/// refuses `dim`. The predicate only grows with the reading, so bisection over it finds
/// the edge.
fn largest_refusing_host(dim: usize) -> usize {
    let (mut refused, mut admitted) = (0usize, usize::MAX / 2);
    assert!(
        !sae_exact_stationarity_admitted(dim, refused) && sae_exact_stationarity_admitted(dim, admitted),
        "the exact-stationarity admission at dimension {dim} must refuse an empty host and admit \
         an unbounded one"
    );
    while admitted - refused > 1 {
        let reading = refused + (admitted - refused) / 2;
        if sae_exact_stationarity_admitted(dim, reading) {
            admitted = reading;
        } else {
            refused = reading;
        }
    }
    refused
}

/// #2933 F39 — a host too small for the dense frame-integrated eigensystem changes
/// how the response is computed, never what is estimated. The carried host reading is
/// one byte below the least one that admits the dense route's resident blocks at their
/// exact dimension `t + Σ M_k·p`, and the fixture is wide enough (`p = 128`) that those
/// blocks exceed the tiny-plan relaxation. The refusal therefore comes from the budget
/// itself, and every smaller allocation the probe route makes is still admitted. The
/// divergence must still integrate the frames, now by output-space probes. The probes
/// must meet their own stopping rule, and land within four of the `√(2ν)` sampling
/// deviations that rule holds them to of the dense route's residual dof at the
/// admitting host. Restoring the dense route whatever the host, or the flip to the
/// fixed-frame response on a refusal, turns this red.
#[test]
fn a_refused_dense_integrated_route_still_integrates_the_frames_2933_f39() {
    let (mut term, target, rho, cache) = framed_state(128, 0.0);
    let dense = term
        .fitted_response_divergence(target.view(), &rho, &cache)
        .expect("the admitting host prices the integrated response");
    assert_eq!(
        (dense.estimator, dense.frame_conditioning),
        (
            FittedResponseDivergenceEstimator::ExactSpectral,
            SaeFrameConditioning::MarginalOverLearnedFrames
        ),
        "the default host must admit the dense frame-integrated route"
    );
    let dense_dim = sae_exact_stationarity_dim(cache.delta_t_len(), term.beta_dim());
    assert!(
        sae_exact_stationarity_resident_bytes(dense_dim) > SAE_DIRECT_ALWAYS_ADMIT_BYTES,
        "the dense route's {} resident bytes at dimension {dense_dim} must exceed the tiny-plan \
         relaxation, so its refusal is the budget's",
        sae_exact_stationarity_resident_bytes(dense_dim)
    );
    term.host_available_bytes = largest_refusing_host(dense_dim);
    let probed = term
        .fitted_response_divergence(target.view(), &rho, &cache)
        .expect("the refusing host prices the integrated response by probes");
    eprintln!(
        "[#2933 F39 route] dense dim {dense_dim}, host {} bytes: dense ν={:.9e} tr R={:.9e}; \
         probed ν={:.9e} tr R={:.9e} ({:?}, {:?})",
        term.host_available_bytes,
        dense.likelihood_residual_dof,
        dense.divergence,
        probed.likelihood_residual_dof,
        probed.divergence,
        probed.estimator,
        probed.frame_conditioning
    );
    assert_eq!(
        probed.frame_conditioning,
        SaeFrameConditioning::MarginalOverLearnedFrames,
        "a refused dense route must not change the estimand to the fixed-frame response"
    );
    let FittedResponseDivergenceEstimator::Hutchinson { likelihood, raw } = probed.estimator else {
        panic!(
            "a host below the dense route's resident blocks must price the frames by probes, got \
             {:?}",
            probed.estimator
        );
    };
    assert!(raw.is_none(), "the fixture carries no whitening metric");
    assert!(
        likelihood.probes >= 2
            && likelihood.residual_dof_standard_error.powi(2) <= 2.0 * likelihood.residual_dof,
        "the probes stopped at {} with a residual-dof Monte Carlo variance {:.3e} above 2ν̂ = \
         {:.3e}",
        likelihood.probes,
        likelihood.residual_dof_standard_error.powi(2),
        2.0 * likelihood.residual_dof
    );
    let sampling_deviation = (2.0 * dense.likelihood_residual_dof).sqrt();
    assert!(
        (probed.likelihood_residual_dof - dense.likelihood_residual_dof).abs()
            <= 4.0 * sampling_deviation,
        "the probed residual dof {:.9e} is more than four sampling deviations \
         ({sampling_deviation:.3e}) from the dense {:.9e}",
        probed.likelihood_residual_dof,
        dense.likelihood_residual_dof
    );
}

/// #2933 F39 — the probe route and the dense route price one operator. Probing the
/// lifted evidence factor with every canonical output direction reproduces the dense
/// frame-integrated `ν = Σ‖e − Re‖²` and `tr R = Σ eᵀRe` exactly, with no statistical
/// multiple: each probe's solve resolves its Ritz pairs to `√ε` relative (the Krylov
/// solve's resolution tolerance), so the sums agree to `√ε` of the sum of the probe
/// values' magnitudes. Both of the F39 rank-2 arms are probed, the target in the frame
/// span and with an off-span constant, and each prints `‖∇_B L·U⊥‖_F`, the normal
/// gradient the cross curvature `E` is built from. Neither arm makes `E` material (job
/// 1280343), so this pin does not cover `E`.
#[test]
fn canonical_probes_of_the_lifted_factor_reproduce_the_dense_integrated_response_2933_f39() {
    for (label, off_span_constant) in [("in span", 0.0_f64), ("absorbed off-span constant", 0.3)] {
        let (term, target, rho, cache) = framed_state(12, off_span_constant);
        let dense = term
            .fitted_response_divergence(target.view(), &rho, &cache)
            .expect("the admitting host prices the integrated response");
        assert_eq!(
            dense.estimator,
            FittedResponseDivergenceEstimator::ExactSpectral,
            "{label}: the default host must admit the dense frame-integrated route"
        );
        let operator = term
            .frame_integrated_response_operator(&rho, target.view(), None)
            .expect("the lifted evidence factor builds at the framed state");
        let (n, p) = target.dim();
        let (mut residual_dof, mut residual_magnitude) = (0.0_f64, 0.0_f64);
        let (mut divergence, mut divergence_magnitude) = (0.0_f64, 0.0_f64);
        for row in 0..n {
            for col in 0..p {
                let mut z = vec![vec![0.0_f64; p]; n];
                z[row][col] = 1.0;
                let (residual, trace) = operator
                    .probe(FittedResponseFrame::Likelihood, &z)
                    .unwrap_or_else(|error| {
                        panic!("{label}: the canonical probe ({row}, {col}) solves: {error}")
                    });
                residual_dof += residual;
                residual_magnitude += residual.abs();
                divergence += trace;
                divergence_magnitude += trace.abs();
            }
        }
        let frame = term.atoms[0]
            .decoder_frame
            .as_ref()
            .expect("the framed state carries its frame")
            .frame()
            .to_owned();
        let residual_at_root = term
            .reconstruction_residual(target.view(), &rho)
            .expect("the framed state has a residual");
        let data_gradient = term.atoms[0].basis_values.t().dot(&residual_at_root);
        let normal_gradient = &data_gradient - &data_gradient.dot(&frame).dot(&frame.t());
        let resolution = f64::EPSILON.sqrt();
        eprintln!(
            "[#2933 F39 canonical {label}] ‖∇_B L·U⊥‖_F={:.3e}; ν {residual_dof:.12e} against \
             dense {:.12e} (gap {:.3e}, bar {:.3e}); tr R {divergence:.12e} against dense \
             {:.12e} (gap {:.3e}, bar {:.3e})",
            normal_gradient.iter().map(|v| v * v).sum::<f64>().sqrt(),
            dense.likelihood_residual_dof,
            (residual_dof - dense.likelihood_residual_dof).abs(),
            resolution * residual_magnitude,
            dense.divergence,
            (divergence - dense.divergence).abs(),
            resolution * divergence_magnitude
        );
        assert!(
            (residual_dof - dense.likelihood_residual_dof).abs() <= resolution * residual_magnitude,
            "{label}: canonical probes give ν = {residual_dof:.12e}, the dense route {:.12e}",
            dense.likelihood_residual_dof
        );
        assert!(
            (divergence - dense.divergence).abs() <= resolution * divergence_magnitude,
            "{label}: canonical probes give tr R = {divergence:.12e}, the dense route {:.12e}",
            dense.divergence
        );
    }
}

/// #2933 F33/F35 — a shape report on the frame-marginal route decomposes the
/// frame-integrated pencil once, for the dispersion's divergence and the covariance
/// together. Forming it again for either consumer counts two.
#[test]
fn a_frame_marginal_shape_report_decomposes_its_pencil_once_2933_f39() {
    let (term, target, rho, cache) = framed_state(12, 0.0);
    let loss = term.loss(target.view(), &rho).expect("the framed state has a loss");
    let residual = term
        .reconstruction_residual(target.view(), &rho)
        .expect("the framed state has a residual");
    let entered = exact_a_pencil_decompositions_on_this_thread();
    let route = term
        .shape_information_route(&rho, target.view(), None, &cache)
        .expect("the framed state has a shape information route");
    assert!(
        matches!(route, ShapeInformationRoute::FrameMarginal(_)),
        "the default host must admit the frame-marginal covariance"
    );
    let dispersion = term
        .reconstruction_dispersion_with_geometry(
            &loss,
            &cache,
            &rho,
            residual.view(),
            Some(route.held_response_geometry()),
        )
        .expect("the frame-marginal report prices a dispersion");
    let information = term
        .shape_information(&route, &rho, target.view(), &cache)
        .expect("the frame-marginal report inverts its information");
    let decompositions = exact_a_pencil_decompositions_on_this_thread() - entered;
    eprintln!(
        "[#2933 F39 report] {decompositions} pencil decompositions; raw noise variance {:.9e}, \
         observed-information covariance {}",
        dispersion.raw_output_noise_variance,
        matches!(information, SaeShapeInformation::ObservedInformation(_))
    );
    assert_eq!(
        decompositions, 1,
        "one frame-marginal report must decompose its pencil once, for the dispersion and the \
         covariance together"
    );
}

/// #2933 F39 — a dense evaluation of a framed state forms its fitted-response divergence
/// once, for the value's rank charge and the gradient's rank-charge derivative together.
/// The value leaves its dispersion on the spectral block it hands the gradient. Reading
/// it there must decompose nothing. Forming it again, the derivative with no block, must
/// decompose the frame-integrated pencil, which is the counter's positive control. Both
/// derivatives must agree bit for bit, because they price one dispersion at one state.
#[test]
fn a_dense_evaluation_forms_its_fitted_response_once_for_value_and_gradient_2933_f39() {
    let (mut term, target, rho, _root_cache) = framed_state(12, 0.0);
    let (_value, loss, priced) = term
        .penalized_quasi_laplace_criterion_priced_with_lane(
            target.view(),
            &rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
            true,
            None,
        )
        .expect("the framed state prices a dense criterion at its own rho");
    let (cache, geometry) = priced.expect("the dense route hands its spectral block on");
    let entered = exact_a_pencil_decompositions_on_this_thread();
    let shared = term
        .production_rank_charge_derivative(target.view(), &rho, &loss, &cache, Some(&geometry))
        .expect("the rank-charge derivative reads the value's block");
    let read = exact_a_pencil_decompositions_on_this_thread() - entered;
    let formed = term
        .production_rank_charge_derivative(target.view(), &rho, &loss, &cache, None)
        .expect("the rank-charge derivative forms its own dispersion");
    let forming = exact_a_pencil_decompositions_on_this_thread() - entered - read;
    let bits = |values: &Array1<f64>| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    eprintln!(
        "[#2933 F39 once per state] decompositions: reading the value's dispersion {read}, \
         forming it again {forming}"
    );
    assert!(
        forming > 0,
        "control: forming the framed dispersion must decompose the frame-integrated pencil"
    );
    assert_eq!(
        read, 0,
        "the gradient's rank charge must read the value's dispersion, not form it again"
    );
    assert!(
        bits(&shared.direct_rho) == bits(&formed.direct_rho)
            && bits(&shared.theta.t) == bits(&formed.theta.t)
            && bits(&shared.theta.beta) == bits(&formed.theta.beta),
        "the derivative off the value's dispersion must equal the one that forms it, bit for bit"
    );
}

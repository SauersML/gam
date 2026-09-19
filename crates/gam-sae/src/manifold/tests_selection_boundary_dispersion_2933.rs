//! #2933 F37/F38 — the reconstruction dispersion is conditional on the fitted
//! routing, so it carries no observed-margin selection charge, and its scale
//! equation is explicit.
//!
//! F37. For fixed candidates `±a` and `y ~ N(μ, σ²)`, the selection `ŷ = a·sign(y)`
//! has search degrees of freedom `Cov(ŷ, y)/σ² = (2a/σ)·φ(μ/σ)`. The charge the
//! dispersion used to add, `(2a/σ)·φ(y/σ)` per row, has expectation
//! `(2a/σ)·φ(μ/(√2σ))/√2`: `0.564190` against `0.797885` at `μ = 0`, and more
//! than the truth once `|μ| > σ·√(2 ln 2)`. A softmax gate is a smooth map of its
//! logits, so a winner crossing an assignment weight of `0.9` must not move the
//! dispersion.
//!
//! F38. The removed charge scaled as `h(σ) = σ⁻¹·exp(−m²/2σ²)`, which increases
//! for `σ < |m|`, so the single "monotone" fixed-point pass it justified was
//! neither monotone nor a root. Neither `ν = ‖I − R‖²_F` nor any other term of the
//! dispersion depends on `φ`, so at a fixed fitted state the dispersion is linear in
//! the residual sum of squares and equals `RSS/ν`, the root of its scale equation,
//! without a seed.
//!
//! The production assertions compare the dispersion across states that differ
//! only in what the removed charge read: the frozen-routing flag, a `0.9`
//! assignment crossing, and the residual scale against fixed margins. At the
//! parent commit each of them moves the dispersion. The root assertion reads `ν`
//! from the fitted response itself, so a denominator other than `ν`, such as the
//! historical `N − tr R`, fails it while staying linear in the RSS. The fixtures
//! have no row metric, so the raw and likelihood frames coincide and both scales
//! are checked.
//!
//! The selection is conditioned on, not estimated, since no statistic of one draw
//! estimates the search degrees of freedom without bias. The label test requires
//! every noise scale and shape report to name
//! `SaeSelectionConditioning::ConditionalOnFittedRouting`.

use super::tests::{TestPeriodicEvaluator, periodic_basis};
use super::*;
use gam_terms::latent::LatentManifold;
use ndarray::array;

/// Half the distance between the two decoded centers.
const HALF_GAP: f64 = 1.0;

fn standard_normal_density(u: f64) -> f64 {
    (-0.5 * u * u).exp() / std::f64::consts::TAU.sqrt()
}

/// Composite Simpson rule on `[lo, hi]` for an integrand that is smooth there.
fn simpson(lo: f64, hi: f64, f: &dyn Fn(f64) -> f64) -> f64 {
    let intervals = 20_000_usize;
    let h = (hi - lo) / intervals as f64;
    let mut sum = f(lo) + f(hi);
    for i in 1..intervals {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * f(lo + h * i as f64);
    }
    sum * h / 3.0
}

/// `E f(Y)` for `Y ~ N(μ, σ²)`, with `f = below` on `y < 0` and `f = above` on
/// `y > 0`. Splitting at the selection boundary keeps the rule's accuracy when
/// `f` jumps there. Both callers keep `0` inside `μ ± 14σ`.
fn gaussian_expectation(
    mu: f64,
    sigma: f64,
    below: &dyn Fn(f64) -> f64,
    above: &dyn Fn(f64) -> f64,
) -> f64 {
    let density = |y: f64| standard_normal_density((y - mu) / sigma) / sigma;
    simpson(mu - 14.0 * sigma, 0.0, &|y: f64| below(y) * density(y))
        + simpson(0.0, mu + 14.0 * sigma, &|y: f64| above(y) * density(y))
}

/// Evenly spaced targets on `(−2.5, 2.5)`, symmetric about the selection boundary.
/// An even count keeps every target off the boundary.
fn symmetric_targets(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| -2.5 + 5.0 * (i as f64 + 0.5) / n as f64)
        .collect()
}

/// The two scales of a dispersion, raw frame first.
fn scales(dispersion: SaeReconstructionDispersion) -> [(&'static str, f64); 2] {
    [
        ("raw output noise variance", dispersion.raw_output_noise_variance),
        ("likelihood dispersion", dispersion.likelihood_dispersion),
    ]
}

/// Two atoms whose decoded curves are the constants `+HALF_GAP` and `−HALF_GAP`
/// on one output channel, one row per target. Each row's routing logits prefer
/// the atom on the target's side of zero by `logit_gap`.
fn two_center_term(
    targets: &[f64],
    mode: AssignmentMode,
    logit_gap: f64,
    log_lambda_smooth: f64,
) -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>) {
    let n = targets.len();
    // At coordinate zero the von Mises ARD curvature is at its maximum, so the
    // coordinate block is positive definite although a constant curve gives the
    // coordinate no data curvature.
    let coords = Array2::<f64>::zeros((n, 1));
    let constant_atom = |name: &str, level: f64| {
        let (phi, jet) = periodic_basis(&coords);
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            array![[level], [0.0], [0.0]],
            Array2::<f64>::eye(3),
        )
        .expect("a three-column basis, latent dimension one and a 3x1 decoder agree")
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atoms = vec![
        constant_atom("upper", HALF_GAP),
        constant_atom("lower", -HALF_GAP),
    ];
    let logits = Array2::from_shape_fn((n, 2), |(row, atom)| {
        if (atom == 0) == (targets[row] > 0.0) {
            logit_gap
        } else {
            0.0
        }
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone(), coords],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        mode,
    )
    .expect("two logit columns, two coordinate blocks and two manifolds agree");
    let term = SaeManifoldTerm::new(atoms, assignment)
        .expect("each atom's basis width matches its assignment block");
    let rho = SaeManifoldRho::new(0.0, log_lambda_smooth, vec![array![0.0], array![0.0]])
        .for_assignment(&term.assignment);
    let target = Array2::from_shape_fn((n, 1), |(row, _)| targets[row]);
    (term, rho, target)
}

/// The loss, the factor cache of one undamped assembly at the stored state, and
/// the reconstruction residual: everything the dispersion reads.
fn dispersion_inputs(
    term: &mut SaeManifoldTerm,
    rho: &SaeManifoldRho,
    target: &Array2<f64>,
) -> (SaeManifoldLoss, ArrowFactorCache, Array2<f64>) {
    let loss = term
        .loss(target.view(), rho)
        .expect("the two-center state has a finite loss");
    let system = term
        .assemble_arrow_schur(target.view(), rho, None)
        .expect("the two-center state assembles");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("the ARD, smoothness and assignment priors make the system positive definite");
    let residual = term
        .reconstruction_residual(target.view(), rho)
        .expect("fitted and target shapes agree");
    (loss, cache, residual)
}

/// F37 on a hard support. The exact two-center calculation integrated over
/// repeated data shows why the observed-margin charge was not search degrees of
/// freedom. The production dispersion of a fixed TopK support must then be the
/// same whether or not the routing that chose it is flagged frozen.
#[test]
fn topk_dispersion_is_conditional_on_the_fitted_support_2933_f37() {
    let (a, sigma) = (HALF_GAP, 1.0_f64);
    // Cov(a·sign(Y), Y)/σ², by quadrature of its definition.
    let search_dof = |mu: f64| {
        gaussian_expectation(mu, sigma, &|y: f64| -a * (y - mu), &|y: f64| {
            a * (y - mu)
        }) / (sigma * sigma)
    };
    let plug_in = |y: f64| (2.0 * a / sigma) * standard_normal_density(y / sigma);
    let plug_in_mean = |mu: f64| gaussian_expectation(mu, sigma, &plug_in, &plug_in);
    let (boundary_dof, boundary_plug_in) = (search_dof(0.0), plug_in_mean(0.0));
    let far_mean = 1.5 * sigma;
    let (far_dof, far_plug_in) = (search_dof(far_mean), plug_in_mean(far_mean));
    eprintln!(
        "[#2933 F37 two-center] mu=0: search dof {boundary_dof:.12} plug-in mean \
         {boundary_plug_in:.12}; mu=1.5 sigma: search dof {far_dof:.12} plug-in mean {far_plug_in:.12}"
    );
    assert!(
        (boundary_dof - 2.0 * a / sigma * standard_normal_density(0.0)).abs() < 1e-9
            && (boundary_dof - 0.797_884_560_803).abs() < 1e-9,
        "search dof at the boundary is (2a/sigma)·phi(0) = 0.797885; quadrature gave {boundary_dof}"
    );
    assert!(
        (boundary_plug_in - 0.564_189_583_548).abs() < 1e-9
            && (boundary_plug_in / boundary_dof - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9,
        "the plug-in's mean at the boundary is 1/sqrt(2) of the search dof; got {boundary_plug_in} \
         against {boundary_dof}"
    );
    let far_ratio =
        std::f64::consts::FRAC_1_SQRT_2 * (far_mean * far_mean / (4.0 * sigma * sigma)).exp();
    assert!(
        (far_plug_in / far_dof - far_ratio).abs() < 1e-8 && far_plug_in > far_dof,
        "past sigma·sqrt(2 ln 2) the plug-in over-counts, so no rescaling repairs it: \
         ratio {} against {far_ratio}",
        far_plug_in / far_dof
    );

    let targets = symmetric_targets(40);
    let (mut term, rho, target) =
        two_center_term(&targets, AssignmentMode::top_k_support(1), 1.0, 0.0);
    let (loss, cache, residual) = dispersion_inputs(&mut term, &rho, &target);
    let routed = term
        .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
        .expect("the TopK two-center dispersion is defined");
    // Freezing the same logits changes no gate, no residual and no factor.
    term.assignment.frozen_logits = Some(term.assignment.logits.clone());
    let frozen = term
        .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
        .expect("the frozen-routing two-center dispersion is defined");
    let rss = 2.0 * loss.data_fit;
    for ((label, routed_scale), (_, frozen_scale)) in scales(routed).into_iter().zip(scales(frozen))
    {
        let implied_search_dof = rss / frozen_scale - rss / routed_scale;
        eprintln!(
            "[#2933 F37 TopK] {label}: routed {routed_scale:.12e} frozen {frozen_scale:.12e} \
             implied search dof {implied_search_dof:.6} over {} rows",
            targets.len()
        );
        assert!(
            (routed_scale - frozen_scale).abs() <= 1e-12 * frozen_scale,
            "{label}: a fixed support's dispersion must not depend on whether the routing that \
             chose it is flagged frozen: routed {routed_scale}, frozen {frozen_scale}, implied \
             search dof {implied_search_dof}"
        );
    }
}

/// F37 on a smooth gate. The winner's softmax weight is `0.9` at a logit gap of
/// `ln 9`, where the removed charge switched on. A move of `2e-7` in the gap must
/// change the dispersion no faster than its own slope on either side allows.
#[test]
fn saturated_softmax_dispersion_is_continuous_in_its_logits_2933_f37() {
    let targets = symmetric_targets(40);
    let mode = AssignmentMode::softmax(1.0);
    let dispersion_at = |logit_gap: f64| {
        let (mut term, rho, target) = two_center_term(&targets, mode, logit_gap, 0.0);
        let (loss, cache, residual) = dispersion_inputs(&mut term, &rho, &target);
        term.reconstruction_dispersion(&loss, &cache, &rho, residual.view())
            .expect("the softmax two-center dispersion is defined")
            .likelihood_dispersion
    };
    let crossing = 9.0_f64.ln();
    let step = 1.0e-7;
    let below = dispersion_at(crossing - step);
    let above = dispersion_at(crossing + step);
    // Secant slopes wholly on one side of the crossing, so a jump at the crossing
    // cannot enter the bound.
    let secant =
        |near: f64, far: f64| (dispersion_at(far) - dispersion_at(near)).abs() / (far - near).abs();
    let slope =
        secant(crossing - 0.05, crossing - 0.1).max(secant(crossing + 0.05, crossing + 0.1));
    let jump = (above - below).abs();
    let smooth_bound = 10.0 * slope * 2.0 * step + 1e-12 * below;
    eprintln!(
        "[#2933 F37 softmax] below {below:.12e} above {above:.12e} jump {jump:.3e} \
         side slope {slope:.3e} bound {smooth_bound:.3e}"
    );
    assert!(
        jump <= smooth_bound,
        "the dispersion of a smooth softmax state jumped by {jump:e} across a winner weight of \
         0.9 (below {below}, above {above}); ten times the side slope over the move allows \
         {smooth_bound:e}"
    );
}

/// F38. The removed charge's σ-derivative is positive below the margin, which
/// refutes the monotone one-pass claim. At a fixed fitted state, with the margins
/// held fixed, the dispersion must be linear in the RSS and must be the root of
/// `φ·ν = RSS` with `ν = ‖I − R‖²_F` read from the fitted response, for every
/// residual scale, with many and with few residual degrees of freedom.
#[test]
fn dispersion_is_the_explicit_root_of_its_scale_equation_2933_f38() {
    let margin = 2.0_f64;
    let charge = |sigma: f64| (-(margin * margin) / (2.0 * sigma * sigma)).exp() / sigma;
    let analytic_slope = charge(1.0) * (margin * margin - 1.0);
    let h = 1.0e-5;
    let central_difference = (charge(1.0 + h) - charge(1.0 - h)) / (2.0 * h);
    eprintln!(
        "[#2933 F38 charge] h'(1) at m=2: analytic {analytic_slope:.12} central \
         {central_difference:.12}"
    );
    assert!(
        (analytic_slope - 0.406_005_849_709_838).abs() < 1e-12
            && (central_difference - analytic_slope).abs() < 1e-9
            && analytic_slope > 0.0,
        "h(sigma) = exp(-m^2/2sigma^2)/sigma increases at sigma=1 < m=2: analytic \
         {analytic_slope}, central difference {central_difference}"
    );

    let cases: [(&str, Vec<f64>, f64); 2] = [
        ("forty rows", symmetric_targets(40), 0.0),
        ("three rows with few residual dof", vec![-1.5, 0.75, 2.0], -4.0),
    ];
    for (case, targets, log_lambda_smooth) in cases {
        let (mut term, rho, target) = two_center_term(
            &targets,
            AssignmentMode::top_k_support(1),
            1.0,
            log_lambda_smooth,
        );
        let (loss, cache, residual) = dispersion_inputs(&mut term, &rho, &target);
        let base = term
            .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
            .expect("the TopK two-center dispersion is defined");
        // The fixture has no decoder frames, so each frame's ν is the response's
        // own residual dof, raw frame first as in `scales`.
        let response = term
            .fitted_response_divergence(target.view(), &rho, &cache)
            .expect("the TopK two-center fitted response is defined");
        let nus = [response.raw_residual_dof, response.likelihood_residual_dof];
        let rss = 2.0 * loss.data_fit;
        for scale in [1.0_f64, 0.25, 0.5, 2.0, 4.0] {
            let mut scaled = loss;
            scaled.data_fit *= scale * scale;
            let rescaled = term
                .reconstruction_dispersion(&scaled, &cache, &rho, residual.view())
                .expect("the rescaled dispersion is defined");
            let scaled_rss = scale * scale * rss;
            for (((label, base_scale), (_, phi)), nu) in
                scales(base).into_iter().zip(scales(rescaled)).zip(nus)
            {
                let linearity_residual = phi * (rss / base_scale) - scaled_rss;
                let equation_residual = phi * nu - scaled_rss;
                eprintln!(
                    "[#2933 F38 {case}] {label}: nu {nu:.6} (N = {}) scale {scale}: phi \
                     {phi:.12e} linearity residual {linearity_residual:.3e} scale-equation \
                     residual {equation_residual:.3e}",
                    targets.len()
                );
                assert!(
                    linearity_residual.abs() <= 1e-12 * scaled_rss,
                    "{case}, {label}: at residual scale {scale} the dispersion {phi} is not linear \
                     in the RSS at fixed margins: it misses base·scale² by {linearity_residual:e} \
                     (RSS {scaled_rss})"
                );
                assert!(
                    equation_residual.abs() <= 1e-12 * scaled_rss,
                    "{case}, {label}: at residual scale {scale} the dispersion {phi} misses its \
                     scale equation phi·nu = RSS by {equation_residual:e} (nu {nu}, RSS \
                     {scaled_rss})"
                );
            }
        }
    }
}

/// F37 refusal. The factored and unfactored dispersions, the shape report
/// assembled on the production route and its unavailable twin must each name the
/// fitted routing they hold fixed.
#[test]
fn dispersion_and_shape_reports_name_the_routing_they_condition_on_2933_f37() {
    let targets = symmetric_targets(40);
    let (mut term, rho, target) =
        two_center_term(&targets, AssignmentMode::top_k_support(1), 1.0, 0.0);
    let (loss, cache, residual) = dispersion_inputs(&mut term, &rho, &target);
    let factored = term
        .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
        .expect("the TopK two-center dispersion is defined");
    let unfactored = term
        .unfactored_reconstruction_dispersion(target.view(), &rho)
        .expect("the unfactored two-center dispersion is defined");
    let route = term
        .shape_information_route(&rho, target.view(), None, &cache)
        .expect("the two-center state has a shape information route");
    let information = term
        .shape_information(&route, &rho, target.view(), &cache)
        .expect("the two-center shape information is defined");
    let assembled = term
        .assemble_shape_uncertainty(&information, factored)
        .expect("the two-center shape report assembles");
    let unavailable = term.unavailable_shape_uncertainty(
        unfactored,
        SaeShapeCovarianceUnavailable::NoDenseObservedInformation,
    );
    eprintln!(
        "[#2933 F37 label] assembled operator {}, unavailable operator {}",
        assembled.operator.as_str(),
        unavailable.operator.as_str()
    );
    let conditioning = SaeSelectionConditioning::ConditionalOnFittedRouting;
    for (label, dispersion) in [
        ("factored dispersion", factored),
        ("unfactored dispersion", unfactored),
    ] {
        assert_eq!(
            dispersion.selection_conditioning, conditioning,
            "the {label} must name the fitted routing it holds fixed"
        );
    }
    for (label, report) in [
        ("assembled shape report", &assembled),
        ("unavailable shape report", &unavailable),
    ] {
        assert_eq!(
            report.selection_conditioning, conditioning,
            "the {label} must name the fitted routing it holds fixed"
        );
        assert_eq!(
            report.dispersion.selection_conditioning, conditioning,
            "the dispersion of the {label} must name the fitted routing it holds fixed"
        );
    }
}

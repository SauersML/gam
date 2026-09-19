//! #2935 / #2933 — the accepted-iterate affine gauge and a constant-curvature atom's
//! κ family.
//!
//! `canonicalize_atom_affine_gauge` (fit_drivers.rs) admits `Poincare` atoms. It
//! re-charts the latent coordinates by `t' = (t − shift)/scale`, solves the basis
//! transport `T` with `Φ(t')·T = Φ(t)`, re-expresses the decoder as `T·B`, and installs
//! `S' = (T⁻¹)ᵀ S T⁻¹` beside `∂S'/∂κ = (T⁻¹)ᵀ (∂S/∂κ) T⁻¹`. A constant-curvature
//! atom's basis is a monomial patch in the tangent coordinate and carries no κ, so `T`
//! is κ-independent and the whole family moves by the same congruence:
//!
//! ```text
//!   S'(κ) = (T⁻¹)ᵀ S(κ) T⁻¹     for every κ
//! ```
//!
//! Every trial κ runs `prepare_constant_curvature`, which rebuilds `S(κ)` and `∂S/∂κ`
//! from the atom's geometry plan. The plan keeps its reference rows and κ in the
//! declared chart, so it also carries the gauge's decoder transport and moves every
//! Gram it builds by the same congruence. A plan left in the old chart installed the
//! OLD-chart Gram on an atom whose decoder is `T·B`, which mispriced the penalty energy
//! itself, not only its derivative.
//!
//! The fixture's coordinates are deliberately off-canonical (non-zero mean, rms far
//! from 1), so the gauge must act. #2935's anchor never exercised it (job 1160070: the
//! coordinates were not re-centred).
//! - `affine_gauge_installs_the_congruence_of_the_gram_and_its_kappa_derivative_2935`:
//!   the gauge moves the coordinates to mean 0 and rms 1, and the installed `S` and
//!   `∂S/∂κ` are the congruence of the pre-gauge pair.
//! - `trial_curvature_after_the_affine_gauge_installs_the_transported_gram_2935`: a
//!   trial κ on the gauged atom installs `(T⁻¹)ᵀ S(κ) T⁻¹` and the matching `∂S/∂κ`.
//! - `the_gauged_plan_persists_its_transport_and_rebuilds_the_installed_gram_2935`: the
//!   gauged plan survives a serialization round trip and rebuilds the installed Gram,
//!   so a saved model's plan prices the decoder it is saved with.
#![cfg(test)]
use super::outer_objective::{solve_basis_transport, transport_smooth_penalty_for_decoder};
use super::*;
use ndarray::{Array2, ArrayView2};

const KAPPA: f64 = 0.3;

/// One constant-curvature tangent-chart atom on deliberately non-canonical coordinates
/// (a golden-angle spiral with radii 0.55 to 1.05, so the per-axis mean is nonzero and
/// the rms is far from 1), so `canonicalize_affine_gauge_after_accept` cannot skip it.
fn off_canonical_curvature_term() -> (SaeManifoldTerm, Array2<f64>) {
    let n = 24usize;
    let p = 3usize;
    let coords = Array2::from_shape_fn((n, 2), |(row, axis)| {
        let angle = std::f64::consts::TAU * 0.618_033_988_75 * row as f64;
        let radius = 0.55 + 0.5 * (row as f64 + 0.5) / n as f64;
        if axis == 0 {
            radius * angle.cos()
        } else {
            radius * angle.sin()
        }
    });
    let plan = SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Poincare,
        2,
        SaeBasisResolution::Polynomial {
            degree: SAE_EUCLIDEAN_PATCH_MAX_DEGREE,
        },
        SaeReferenceMetricPlan::ConstantCurvatureChart {
            kappa: KAPPA,
            reference_coords: coords.clone(),
        },
    )
    .expect("constant-curvature plan");
    let bundle = plan
        .evaluate_bundle(coords.view())
        .expect("reference rows lie inside the chart");
    let m = bundle.basis_values.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
        0.4 * ((1 + basis_col + 3 * out_col) as f64).sin()
    });
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "curvature",
        SaeAtomBasisKind::Poincare,
        2,
        bundle.basis_values,
        bundle.basis_jacobian,
        decoder,
        bundle.reference_penalty,
    )
    .expect("atom from the plan's own bundle")
    .with_basis_second_jet(bundle.evaluator)
    .with_geometry_plan(plan)
    .expect("installed Gram is the plan's Gram");
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords.clone()],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("one coordinate block for one atom");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("single-atom term");
    (term, coords)
}

/// Per-axis mean and root-mean-square about that mean. The fixture's single atom
/// takes every row with weight one, so these are the gauge's own weighted moments.
fn axis_moments(coords: ArrayView2<'_, f64>) -> (Vec<f64>, Vec<f64>) {
    let n = coords.nrows() as f64;
    coords
        .columns()
        .into_iter()
        .map(|column| {
            let mean = column.sum() / n;
            let rms = (column.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / n).sqrt();
            (mean, rms)
        })
        .unzip()
}

/// The basis the atom's evaluator produces at `coords`: the values the gauge solves
/// its transport against.
fn evaluate_basis(atom: &SaeManifoldAtom, coords: ArrayView2<'_, f64>) -> Array2<f64> {
    atom.basis_second_jet
        .as_ref()
        .expect("the fixture atom carries its second-jet evaluator")
        .evaluate(coords)
        .expect("the evaluator accepts the chart coordinates")
        .0
}

fn kappa_derivative(atom: &SaeManifoldAtom) -> Array2<f64> {
    atom.smooth_penalty_kappa_derivative()
        .expect("∂S/∂κ sits at the live basis width")
        .expect("a constant-curvature atom carries ∂S/∂κ")
        .clone()
}

/// The Gram and `∂S/∂κ` a trial curvature installs on `atom`: prepared from the
/// atom's own geometry plan and committed on a copy, as the outer curvature install
/// does.
fn curvature_trial(atom: &SaeManifoldAtom, kappa: f64) -> (Array2<f64>, Array2<f64>) {
    let prepared = atom
        .prepare_constant_curvature(kappa)
        .expect("the trial curvature lies inside the chart domain");
    let mut trial = atom.clone();
    trial.commit_prepared_constant_curvature(prepared);
    (trial.smooth_penalty().clone(), kappa_derivative(&trial))
}

/// `(T⁻¹)ᵀ M T⁻¹`, the congruence the gauge applies to the Gram and to `∂S/∂κ`.
fn congruence(transport: &Array2<f64>, gram: &Array2<f64>) -> Array2<f64> {
    transport_smooth_penalty_for_decoder(transport.view(), gram.view())
        .expect("the basis transport is square and invertible")
}

/// Gram tolerance mirroring `with_geometry_plan`: square-root machine resolution,
/// scaled by the larger matrix's magnitude and the width.
fn gram_tolerance(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let scale = a
        .iter()
        .chain(b.iter())
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    f64::EPSILON.sqrt() * scale * a.nrows().max(1) as f64
}

fn max_abs_difference(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// After the gauge acts, the installed `S` and `∂S/∂κ` are the exact basis congruence
/// of the pre-gauge pair, and the congruence materially moved `S`.
#[test]
fn affine_gauge_installs_the_congruence_of_the_gram_and_its_kappa_derivative_2935() {
    let (mut term, old_coords) = off_canonical_curvature_term();
    let old_atom = term.atoms[0].clone();
    let old_phi = evaluate_basis(&old_atom, old_coords.view());
    let old_penalty = old_atom.smooth_penalty().clone();
    let old_derivative = kappa_derivative(&old_atom);
    let (old_mean, old_rms) = axis_moments(old_coords.view());

    term.canonicalize_affine_gauge_after_accept(None)
        .expect("the gauge admits Poincare atoms");

    let new_coords = term.assignment.coords[0].as_matrix();
    let (mean, rms) = axis_moments(new_coords.view());
    let new_phi = evaluate_basis(&term.atoms[0], new_coords.view());
    let transport = solve_basis_transport(new_phi.view(), old_phi.view())
        .expect("a monomial re-chart has an exact basis transport");
    let expected_penalty = congruence(&transport, &old_penalty);
    let expected_derivative = congruence(&transport, &old_derivative);
    let installed_penalty = term.atoms[0].smooth_penalty().clone();
    let installed_derivative = kappa_derivative(&term.atoms[0]);
    let penalty_gap = max_abs_difference(&installed_penalty, &expected_penalty);
    let penalty_tolerance = gram_tolerance(&installed_penalty, &expected_penalty);
    let derivative_gap = max_abs_difference(&installed_derivative, &expected_derivative);
    let derivative_tolerance = gram_tolerance(&installed_derivative, &expected_derivative);
    let chart_moved = max_abs_difference(&expected_penalty, &old_penalty);
    println!(
        "[#2935 gauge] pre (mean {old_mean:?}, rms {old_rms:?}) -> post (mean {mean:?}, rms {rms:?}); \
         S moved {chart_moved:.3e}; S gap {penalty_gap:.3e} (tolerance {penalty_tolerance:.3e}); \
         ∂S/∂κ gap {derivative_gap:.3e} (tolerance {derivative_tolerance:.3e})"
    );
    assert!(
        old_mean
            .iter()
            .zip(old_rms.iter())
            .any(|(m, r)| m.abs() > 1.0e-3 || (r - 1.0).abs() > 1.0e-3),
        "the fixture was already canonical, so the gauge would skip it"
    );
    assert!(
        mean.iter().all(|m| m.abs() <= 1.0e-9),
        "the gauge did not centre the coordinates ({mean:?})"
    );
    assert!(
        rms.iter().all(|r| (r - 1.0).abs() <= 1.0e-9),
        "the gauge did not unit-scale the coordinates ({rms:?})"
    );
    assert!(
        chart_moved > 1.0e3 * penalty_tolerance,
        "the congruence must materially change S for the transport checks to mean anything \
         ({chart_moved:.3e} vs tolerance {penalty_tolerance:.3e})"
    );
    assert!(
        penalty_gap <= penalty_tolerance,
        "the installed S is not the congruence of the pre-gauge Gram (gap {penalty_gap:.3e}, \
         tolerance {penalty_tolerance:.3e})"
    );
    assert!(
        derivative_gap <= derivative_tolerance,
        "the installed ∂S/∂κ is not the congruence of the pre-gauge derivative (gap \
         {derivative_gap:.3e}, tolerance {derivative_tolerance:.3e})"
    );
}

/// A trial curvature on the gauged atom must install the old chart's `S(κ)` and
/// `∂S/∂κ` carried across the re-chart by the gauge's congruence. Rebuilding them from
/// the plan's old-chart reference rows discards that congruence.
#[test]
fn trial_curvature_after_the_affine_gauge_installs_the_transported_gram_2935() {
    let (mut term, old_coords) = off_canonical_curvature_term();
    let old_atom = term.atoms[0].clone();
    let old_phi = evaluate_basis(&old_atom, old_coords.view());

    term.canonicalize_affine_gauge_after_accept(None)
        .expect("the gauge admits Poincare atoms");

    let new_coords = term.assignment.coords[0].as_matrix();
    let new_phi = evaluate_basis(&term.atoms[0], new_coords.view());
    let transport = solve_basis_transport(new_phi.view(), old_phi.view())
        .expect("a monomial re-chart has an exact basis transport");
    let step = 1.0e-3_f64;
    for kappa in [KAPPA - step, KAPPA + step] {
        let (old_penalty, old_derivative) = curvature_trial(&old_atom, kappa);
        let (installed_penalty, installed_derivative) = curvature_trial(&term.atoms[0], kappa);
        let expected_penalty = congruence(&transport, &old_penalty);
        let expected_derivative = congruence(&transport, &old_derivative);
        let penalty_gap = max_abs_difference(&installed_penalty, &expected_penalty);
        let penalty_tolerance = gram_tolerance(&installed_penalty, &expected_penalty);
        let derivative_gap = max_abs_difference(&installed_derivative, &expected_derivative);
        let derivative_tolerance = gram_tolerance(&installed_derivative, &expected_derivative);
        let chart_moved = max_abs_difference(&expected_penalty, &old_penalty);
        println!(
            "[#2935 trial κ={kappa:.4}] S moved {chart_moved:.3e}; S gap {penalty_gap:.3e} \
             (tolerance {penalty_tolerance:.3e}); ∂S/∂κ gap {derivative_gap:.3e} (tolerance \
             {derivative_tolerance:.3e})"
        );
        assert!(
            chart_moved > 1.0e3 * penalty_tolerance,
            "trial κ = {kappa}: the congruence must materially change S for the check to mean \
             anything ({chart_moved:.3e} vs tolerance {penalty_tolerance:.3e})"
        );
        assert!(
            penalty_gap <= penalty_tolerance,
            "trial κ = {kappa}: the installed S(κ) is not the old chart's Gram carried by the \
             gauge's congruence (gap {penalty_gap:.3e}, tolerance {penalty_tolerance:.3e})"
        );
        assert!(
            derivative_gap <= derivative_tolerance,
            "trial κ = {kappa}: the installed ∂S/∂κ is not the old chart's derivative carried \
             by the gauge's congruence (gap {derivative_gap:.3e}, tolerance \
             {derivative_tolerance:.3e})"
        );
    }
}

/// A gauged plan is model data: a serialization round trip must keep its decoder
/// transport, and the plan must rebuild the Gram installed beside the decoder it is
/// saved with. A plan that rebuilt the old chart's Gram would price a reloaded model's
/// decoder `T·B` with the penalty of its pre-image `B`.
#[test]
fn the_gauged_plan_persists_its_transport_and_rebuilds_the_installed_gram_2935() {
    let (mut term, _) = off_canonical_curvature_term();
    let old_penalty = term.atoms[0].smooth_penalty().clone();

    term.canonicalize_affine_gauge_after_accept(None)
        .expect("the gauge admits Poincare atoms");

    let atom = &term.atoms[0];
    let plan = atom
        .geometry_plan()
        .expect("the gauged atom keeps its geometry plan")
        .clone();
    let json = serde_json::to_string(&plan).expect("a geometry plan serializes");
    let reloaded: SaeAtomGeometryPlan =
        serde_json::from_str(&json).expect("a serialized geometry plan reloads through its validator");
    let installed_penalty = atom.smooth_penalty().clone();
    let installed_derivative = kappa_derivative(atom);
    let rebuilt_penalty = reloaded
        .build_reference_penalty()
        .expect("the reloaded plan builds its Gram");
    let rebuilt_derivative = reloaded
        .build_reference_penalty_kappa_derivative()
        .expect("the reloaded plan builds its curvature derivative")
        .expect("a constant-curvature plan carries ∂S/∂κ");
    let penalty_gap = max_abs_difference(&rebuilt_penalty, &installed_penalty);
    let penalty_tolerance = gram_tolerance(&rebuilt_penalty, &installed_penalty);
    let derivative_gap = max_abs_difference(&rebuilt_derivative, &installed_derivative);
    let derivative_tolerance = gram_tolerance(&rebuilt_derivative, &installed_derivative);
    let chart_moved = max_abs_difference(&installed_penalty, &old_penalty);
    println!(
        "[#2935 persisted plan] S moved {chart_moved:.3e}; rebuilt S gap {penalty_gap:.3e} \
         (tolerance {penalty_tolerance:.3e}); rebuilt ∂S/∂κ gap {derivative_gap:.3e} \
         (tolerance {derivative_tolerance:.3e}); round trip equal {}",
        reloaded == plan
    );
    assert!(
        chart_moved > 1.0e3 * penalty_tolerance,
        "the gauge must materially change S for the rebuild check to mean anything \
         ({chart_moved:.3e} vs tolerance {penalty_tolerance:.3e})"
    );
    assert_eq!(
        reloaded, plan,
        "a serialization round trip must keep the gauged plan, decoder transport included"
    );
    assert!(
        penalty_gap <= penalty_tolerance,
        "the reloaded plan does not rebuild the installed Gram (gap {penalty_gap:.3e}, \
         tolerance {penalty_tolerance:.3e})"
    );
    assert!(
        derivative_gap <= derivative_tolerance,
        "the reloaded plan does not rebuild the installed ∂S/∂κ (gap {derivative_gap:.3e}, \
         tolerance {derivative_tolerance:.3e})"
    );
}

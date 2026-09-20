#![cfg(test)]
//! #2933 F02 — a basis third jet is Analytic, CertifiedZero or Unavailable, and
//! only the first two may feed an exact observed-information derivative.
//!
//! The exact-A θ-adjoint differentiates `A = JᵀMJ + Σ(Mr)·∂²f + …`, whose
//! residual term moves with the coordinates through `⟨Mr, ∂³f⟩`. An evaluator
//! that exposes `∂²φ` but not `∂³φ` used to have that leg priced as zero, so a
//! `sin`-type basis returned an incomplete gradient labelled exact. These gates
//! pin the refusal on both producers and the production channel, keep the
//! certified-zero fast path for an affine basis, and check that the rank
//! reducer installed at fit entry forwards the declaration unchanged.

use super::construction::{AtomThirdJet, ThirdJetUnavailable};
use super::tests::*;
use super::tests_isometry_exact_hvp_majorizer_457::build_isometry_atom_for_evaluator;
use crate::manifold::tests_dense_solver_oracles::DeflatedArrowSolver;
use super::*;
use gam_solve::arrow_schur::ArrowSchurError;
use ndarray::{Array4, Array5, array};

/// `[1, sin 2πt, cos 2πt]` with its analytic second jet and no third jet. The
/// function's third derivative is `(2π)³·[0, −cos 2πt, sin 2πt]`, not zero.
#[derive(Debug)]
struct SecondOrderOnlyPeriodic;

impl SaeBasisEvaluator for SecondOrderOnlyPeriodic {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<SaeBasisThirdJetCapability, String> {
        if coords.ncols() != 1 {
            return Err(format!(
                "SecondOrderOnlyPeriodic: expected latent_dim 1, got {}",
                coords.ncols()
            ));
        }
        Ok(SaeBasisThirdJetCapability::Unavailable)
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        TestPeriodicEvaluator.evaluate(coords)
    }
}

impl SaeBasisSecondJet for SecondOrderOnlyPeriodic {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        TestPeriodicEvaluator
            .second_jet_dyn(coords)
            .ok_or_else(|| "TestPeriodicEvaluator exposes its second jet".to_string())?
    }

    fn jet_ball_bound(
        &self,
        center: ndarray::ArrayView1<'_, f64>,
        radius: f64,
    ) -> Result<crate::basis::SaeBasisJetBallCapability, String> {
        Ok(crate::basis::SaeBasisJetBallCapability::Unavailable(format!(
            "SecondOrderOnlyPeriodic is a third-jet capability test basis and declares no bound \
             on the ball of radius {radius} around {center}"
        )))
    }
}

/// The same periodic basis falsely certifying a zero third jet. It is the
/// positive control: it measures, on this fixture, the gradient a silent zero
/// would have returned.
#[derive(Debug)]
struct FalselyCertifiedPeriodic;

impl SaeBasisEvaluator for FalselyCertifiedPeriodic {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        TestPeriodicEvaluator.second_jet_dyn(coords)
    }

    fn third_jet_dyn(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<SaeBasisThirdJetCapability, String> {
        if coords.ncols() != 1 {
            return Err(format!(
                "FalselyCertifiedPeriodic: expected latent_dim 1, got {}",
                coords.ncols()
            ));
        }
        Ok(SaeBasisThirdJetCapability::CertifiedZero)
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        TestPeriodicEvaluator.evaluate(coords)
    }
}

/// The affine line `[1, t]`. It either certifies its zero third jet or hands back
/// the materialized zero tensor, so the two declarations can be compared.
#[derive(Debug)]
struct AffineLine {
    materialize_zero_jet: bool,
}

impl SaeBasisEvaluator for AffineLine {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<SaeBasisThirdJetCapability, String> {
        if coords.ncols() != 1 {
            return Err(format!("AffineLine: expected latent_dim 1, got {}", coords.ncols()));
        }
        if self.materialize_zero_jet {
            Ok(SaeBasisThirdJetCapability::Analytic(Array5::<f64>::zeros((
                coords.nrows(),
                2,
                1,
                1,
                1,
            ))))
        } else {
            Ok(SaeBasisThirdJetCapability::CertifiedZero)
        }
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != 1 {
            return Err(format!("AffineLine: expected latent_dim 1, got {}", coords.ncols()));
        }
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = Array3::<f64>::zeros((n, 2, 1));
        for row in 0..n {
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = coords[[row, 0]];
            jet[[row, 1, 0]] = 1.0;
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for AffineLine {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != 1 {
            return Err(format!("AffineLine: expected latent_dim 1, got {}", coords.ncols()));
        }
        Ok(Array4::<f64>::zeros((coords.nrows(), 2, 1, 1)))
    }

    fn jet_ball_bound(
        &self,
        center: ndarray::ArrayView1<'_, f64>,
        radius: f64,
    ) -> Result<crate::basis::SaeBasisJetBallCapability, String> {
        Ok(crate::basis::SaeBasisJetBallCapability::Unavailable(format!(
            "AffineLine is a third-jet capability test basis and declares no bound \
             on the ball of radius {radius} around {center}"
        )))
    }
}

/// One softmax atom on 24 rows. The target is the planted decoded image plus a
/// residual of amplitude 0.25, so `Mr` is far from zero and the residual third
/// leg is live wherever `∂³φ` is.
fn one_atom_fixture(
    evaluator: Arc<dyn SaeBasisEvaluator>,
    kind: SaeAtomBasisKind,
    manifold: LatentManifold,
    decoder: Array2<f64>,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 24usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the fixture coordinates lie in the evaluator's domain");
    let m = phi.ncols();
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        target[[row, 0]] += 0.25 * (0.37 * row as f64).sin();
        target[[row, 1]] += 0.25 * (0.29 * row as f64).cos();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "capability",
        kind,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("atom fixture shapes agree by construction")
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![manifold],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment fixture shapes agree by construction");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("one-atom term");
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![250.0_f64.ln()]]);
    (term, target, rho)
}

fn periodic_fixture(
    evaluator: Arc<dyn SaeBasisEvaluator>,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    one_atom_fixture(
        evaluator,
        SaeAtomBasisKind::Periodic,
        LatentManifold::Circle { period: 1.0 },
        array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]],
    )
}

fn affine_fixture(materialize_zero_jet: bool) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    one_atom_fixture(
        Arc::new(AffineLine {
            materialize_zero_jet,
        }),
        SaeAtomBasisKind::Linear,
        LatentManifold::Euclidean,
        array![[0.30, -0.10], [1.20, 0.90]],
    )
}

/// Both exact-A producers with the residual target, on one Newton-step cache:
/// the dense reconstruction and the from-probes channel at full-basis probes
/// (exact `S⁻¹e_j`, so no stochastic error).
fn exact_a_theta_adjoints(
    fixture: (SaeManifoldTerm, Array2<f64>, SaeManifoldRho),
) -> (
    SaeManifoldTerm,
    Result<SaeArrowVector, String>,
    Result<SaeArrowVector, String>,
) {
    let (mut term, target, rho) = fixture;
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("assemble the fixture's arrow system");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("factor the fixture's majorizer");
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| cache.schur_inverse_apply(v.view()).expect("S^-1 probe"))
        .collect();
    let solver = DeflatedArrowSolver::plain(&cache);
    let inv = term
        .materialize_joint_inverse(&cache, &solver)
        .expect("joint inverse");
    let dense =
        term.logdet_theta_adjoint_dense(&rho, &cache, &inv, true, true, Some(target.view()));
    let probed = term.logdet_theta_adjoint_from_probes(
        &rho,
        &cache,
        &probes,
        &sinv,
        EvidenceOperator::ExactObservedInformation,
        Some(target.view()),
    );
    (term, dense, probed)
}

fn worst_gap(x: &SaeArrowVector, y: &SaeArrowVector) -> f64 {
    x.t.iter()
        .zip(y.t.iter())
        .chain(x.beta.iter().zip(y.beta.iter()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs(x: &SaeArrowVector) -> f64 {
    x.t.iter()
        .chain(x.beta.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
}

fn unavailable_refusal() -> String {
    ThirdJetUnavailable {
        atom: "capability".to_string(),
    }
    .to_string()
}

/// The `sin`-type evaluator must make both exact-A θ-adjoint producers refuse.
///
/// Premises, each free to fail:
/// 1. the analytic twin's third jet is the central difference of its second jet
///    (magnitude `(2π)³ ≈ 248`), so it is a trustworthy oracle;
/// 2. on this fixture the falsely certified zero moves Γ away from the analytic
///    twin by a resolvable amount, so a silent zero here is a wrong gradient and
///    not a harmless one.
#[test]
fn second_order_only_evaluator_refuses_the_exact_a_theta_adjoint_2933() {
    let coords = Array2::from_shape_fn((7, 1), |(row, _)| 0.05 + 0.13 * row as f64);
    let third = match TestPeriodicEvaluator
        .third_jet_dyn(coords.view())
        .expect("analytic third jet")
    {
        SaeBasisThirdJetCapability::Analytic(t3) => t3,
        other => panic!("TestPeriodicEvaluator must declare an analytic third jet, got {other:?}"),
    };
    let step = 1.0e-5;
    let mut plus = coords.clone();
    let mut minus = coords.clone();
    plus.mapv_inplace(|t| t + step);
    minus.mapv_inplace(|t| t - step);
    let second_plus = TestPeriodicEvaluator
        .second_jet_dyn(plus.view())
        .expect("second jet")
        .expect("second jet");
    let second_minus = TestPeriodicEvaluator
        .second_jet_dyn(minus.view())
        .expect("second jet")
        .expect("second jet");
    let mut worst_fd = 0.0_f64;
    let mut largest = 0.0_f64;
    for row in 0..coords.nrows() {
        for basis in 0..3 {
            let fd = (second_plus[[row, basis, 0, 0]] - second_minus[[row, basis, 0, 0]])
                / (2.0 * step);
            let analytic = third[[row, basis, 0, 0, 0]];
            worst_fd = worst_fd.max((analytic - fd).abs());
            largest = largest.max(analytic.abs());
        }
    }
    println!("[#2933 F02] analytic third jet: max|T| = {largest:.6e}, worst |T - fd| = {worst_fd:.6e}");
    assert!(
        largest > 100.0 && worst_fd <= 1.0e-5,
        "premise 1: the analytic periodic third jet must match the central difference of its \
         second jet (max|T| = {largest:.6e}, worst gap = {worst_fd:.6e})"
    );

    let (_, dense_analytic, probes_analytic) =
        exact_a_theta_adjoints(periodic_fixture(Arc::new(TestPeriodicEvaluator)));
    let dense_analytic = dense_analytic.expect("analytic twin: dense exact-A adjoint");
    let probes_analytic = probes_analytic.expect("analytic twin: from-probes exact-A adjoint");
    let (_, dense_zeroed, probes_zeroed) =
        exact_a_theta_adjoints(periodic_fixture(Arc::new(FalselyCertifiedPeriodic)));
    let dense_zeroed = dense_zeroed.expect("falsely certified twin: dense exact-A adjoint");
    let probes_zeroed = probes_zeroed.expect("falsely certified twin: from-probes exact-A adjoint");
    let scale = max_abs(&dense_analytic);
    let dense_separation = worst_gap(&dense_zeroed, &dense_analytic);
    let probes_separation = worst_gap(&probes_zeroed, &probes_analytic);
    println!(
        "[#2933 F02] silent-zero control: dense |zeroed - analytic| = {dense_separation:.6e}, \
         from-probes |zeroed - analytic| = {probes_separation:.6e}, max|Γ| = {scale:.6e}"
    );
    assert!(
        dense_separation > 1.0e-3 * scale && probes_separation > 1.0e-3 * scale,
        "premise 2: a zero residual third leg must visibly change Γ on this fixture \
         (dense {dense_separation:.6e}, from-probes {probes_separation:.6e}, max|Γ| {scale:.6e})"
    );

    let (_, dense, probes) =
        exact_a_theta_adjoints(periodic_fixture(Arc::new(SecondOrderOnlyPeriodic)));
    assert_eq!(
        dense.err(),
        Some(unavailable_refusal()),
        "the dense exact-A θ-adjoint must refuse an evaluator without a third jet"
    );
    assert_eq!(
        probes.err(),
        Some(unavailable_refusal()),
        "the from-probes exact-A θ-adjoint must refuse an evaluator without a third jet"
    );
}

/// The production channel `dense_exact_a_logdet_channels` refuses the same
/// evaluator, while the analytic twin on the identical fixed-state cache
/// succeeds, so the refusal is the capability and not the fixture.
#[test]
fn production_exact_a_logdet_channels_refuse_an_unavailable_third_jet_2933() {
    let channels = |evaluator: Arc<dyn SaeBasisEvaluator>| -> Result<SaeArrowVector, String> {
        let (mut term, target, rho) = periodic_fixture(evaluator);
        let (_value, loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                0,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("fixed-state criterion cache");
        let geometry = term.materialize_dense_exact_a_geometry(&rho, target.view(), &cache)?;
        let rank_charge = term.production_rank_charge_derivative(
            target.view(),
            &rho,
            &loss,
            &cache,
            Some(&geometry),
        )?;
        term.dense_exact_a_logdet_channels(
            target.view(),
            &rho,
            &cache,
            &geometry,
            &rank_charge.theta,
        )
        .map(|channels| channels.theta_adjoint)
    };
    let analytic = channels(Arc::new(TestPeriodicEvaluator))
        .expect("the analytic twin admits the exact-A channels");
    assert!(
        max_abs(&analytic) > 1.0e-6,
        "the analytic twin's θ-adjoint must be non-trivial, got max|Γ| = {:.6e}",
        max_abs(&analytic)
    );
    assert_eq!(
        channels(Arc::new(SecondOrderOnlyPeriodic)).err(),
        Some(unavailable_refusal()),
        "the production exact-A channels must refuse an evaluator without a third jet"
    );
}

/// An affine basis certifies its zero third jet: no tensor is materialized, and
/// both producers return exactly what the materialized zero tensor returns.
#[test]
fn certified_zero_affine_evaluator_keeps_the_fast_zero_path_2933() {
    let (certified_term, dense_certified, probes_certified) =
        exact_a_theta_adjoints(affine_fixture(false));
    let (materialized_term, dense_materialized, probes_materialized) =
        exact_a_theta_adjoints(affine_fixture(true));
    assert!(
        matches!(
            certified_term.atom_third_jets().expect("certified jets")[..],
            [AtomThirdJet::CertifiedZero]
        ),
        "a certified-zero evaluator must not materialize a third-jet tensor"
    );
    assert!(
        matches!(
            materialized_term.atom_third_jets().expect("materialized jets")[..],
            [AtomThirdJet::Analytic(_)]
        ),
        "the materialized twin must carry its analytic tensor"
    );
    let dense_certified = dense_certified.expect("certified zero: dense exact-A adjoint");
    let probes_certified = probes_certified.expect("certified zero: from-probes exact-A adjoint");
    let dense_materialized = dense_materialized.expect("materialized zero: dense exact-A adjoint");
    let probes_materialized =
        probes_materialized.expect("materialized zero: from-probes exact-A adjoint");
    let scale = max_abs(&dense_certified);
    println!(
        "[#2933 F02] affine fast path: max|Γ| = {scale:.6e}, dense gap = {:.3e}, \
         from-probes gap = {:.3e}",
        worst_gap(&dense_certified, &dense_materialized),
        worst_gap(&probes_certified, &probes_materialized)
    );
    assert!(
        scale > 1.0e-6,
        "the affine θ-adjoint must be non-trivial for the equality below to mean anything"
    );
    assert_eq!(
        worst_gap(&dense_certified, &dense_materialized),
        0.0,
        "the certified-zero dense adjoint must equal the materialized-zero one"
    );
    assert_eq!(
        worst_gap(&probes_certified, &probes_materialized),
        0.0,
        "the certified-zero from-probes adjoint must equal the materialized-zero one"
    );
}

/// The rank reducer installed at fit entry remixes basis columns linearly, so it
/// must forward Unavailable and CertifiedZero unchanged and remix an analytic jet.
#[test]
fn subspace_reduction_forwards_the_third_jet_capability_2933() {
    let coords = Array2::from_shape_fn((5, 1), |(row, _)| 0.1 + 0.17 * row as f64);
    let q_periodic = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let unavailable =
        SubspaceReducedEvaluator::new(Arc::new(SecondOrderOnlyPeriodic), q_periodic.clone())
            .expect("reduce the second-order-only basis");
    assert!(
        matches!(
            unavailable.third_jet_dyn(coords.view()),
            Ok(SaeBasisThirdJetCapability::Unavailable)
        ),
        "a reduced second-order-only basis must stay Unavailable"
    );
    let certified = SubspaceReducedEvaluator::new(
        Arc::new(AffineLine {
            materialize_zero_jet: false,
        }),
        array![[1.0], [0.0]],
    )
    .expect("reduce the affine basis");
    assert!(
        matches!(
            certified.third_jet_dyn(coords.view()),
            Ok(SaeBasisThirdJetCapability::CertifiedZero)
        ),
        "a reduced affine basis must stay CertifiedZero"
    );
    let inner = PeriodicHarmonicEvaluator::new(3).expect("periodic basis");
    let inner_third = <PeriodicHarmonicEvaluator as SaeBasisThirdJet>::third_jet(&inner, coords.view())
        .expect("inner analytic third jet");
    let reduced = SubspaceReducedEvaluator::new(Arc::new(inner), q_periodic)
        .expect("reduce the periodic basis");
    let reduced_third = match reduced.third_jet_dyn(coords.view()) {
        Ok(SaeBasisThirdJetCapability::Analytic(t3)) => t3,
        other => panic!("a reduced analytic basis must stay Analytic, got {other:?}"),
    };
    assert_eq!(reduced_third.dim(), (coords.nrows(), 2, 1, 1, 1));
    let largest = inner_third
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(largest > 1.0, "the periodic third jet must be non-trivial, got {largest:.6e}");
    for row in 0..coords.nrows() {
        for col in 0..2 {
            assert_eq!(
                reduced_third[[row, col, 0, 0, 0]],
                inner_third[[row, col, 0, 0, 0]],
                "the identity-prefix remix must keep inner column {col} at row {row}"
            );
        }
    }
}

/// The exact isometry Hessian reads the same declaration through the cache
/// refresh, and production must refuse an unavailable third jet there too.
///
/// Premise, free to fail: with no decoder third jet installed, the exact
/// isometry Hessian's evaluation precondition refuses by naming K, while the
/// analytic twin's hvp is nonzero. So a penalty built on an unavailable jet has
/// no exact curvature to carry.
///
/// Positive control of that guard: `hvp` itself, called on the unavailable twin,
/// must stop by naming K. A zero default reinstated inside the method leaves the
/// precondition intact, so only this call can see it return a vector instead.
#[test]
fn isometry_penalty_refuses_an_unavailable_third_jet_2933() {
    let coords = Array2::from_shape_fn((12, 1), |(row, _)| (row as f64 + 0.3) / 12.0);
    let direction: Array1<f64> = (0..12).map(|row| (0.7 * row as f64).cos()).collect();
    let rho = array![0.0_f64];
    let max_abs = |hv: Array1<f64>| hv.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let refreshed_hvp = |evaluator: Arc<dyn SaeBasisSecondJet>| -> (
        bool,
        Result<f64, String>,
        Result<f64, String>,
    ) {
        let (atom, penalty, target_flat) = build_isometry_atom_for_evaluator(
            evaluator,
            SaeAtomBasisKind::Periodic,
            &coords,
            2,
            0.91,
        );
        refresh_isometry_caches_from_atom(&penalty, &atom, coords.view())
            .expect("the isometry caches refresh from the atom");
        let max_hv = penalty
            .evaluation_state_precondition(
                gam_terms::analytic_penalties::IsometryEvaluationOrder::Hessian,
                target_flat.len(),
            )
            .map(|()| max_abs(penalty.hvp(target_flat.view(), rho.view(), direction.view())));
        let direct_hv = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            penalty.hvp(target_flat.view(), rho.view(), direction.view())
        }))
        .map(|hv| max_abs(hv))
        .map_err(|payload| {
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|reason| reason.to_string()))
                .unwrap_or_else(|| "non-string panic payload".to_string())
        });
        (penalty.third_decoder_derivative().is_some(), max_hv, direct_hv)
    };
    let (analytic_k, analytic_hv, analytic_direct) =
        refreshed_hvp(Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic basis")));
    let (unavailable_k, unavailable_hv, unavailable_direct) =
        refreshed_hvp(Arc::new(SecondOrderOnlyPeriodic));
    println!(
        "[#2933 F02] isometry hvp: analytic K = {analytic_k}, max|Hv| = {analytic_hv:?}, direct \
         hvp = {analytic_direct:?}; unavailable K = {unavailable_k}, max|Hv| = \
         {unavailable_hv:?}, direct hvp = {unavailable_direct:?}"
    );
    assert!(
        analytic_direct.as_ref().is_ok_and(|max_hv| *max_hv > 1.0e-6),
        "positive control: with K installed, hvp evaluates a nonzero exact product \
         (direct hvp = {analytic_direct:?})"
    );
    assert!(
        unavailable_direct
            .as_ref()
            .is_err_and(|reason| reason.contains("K = ∂H/∂t")),
        "hvp on an unavailable third jet must stop by naming K, not return a vector \
         (direct hvp = {unavailable_direct:?})"
    );
    assert!(
        analytic_k && analytic_hv.as_ref().is_ok_and(|max_hv| *max_hv > 1.0e-6),
        "premise: the analytic twin installs K and has a nonzero exact isometry hvp \
         (K = {analytic_k}, max|Hv| = {analytic_hv:?})"
    );
    assert!(
        !unavailable_k
            && unavailable_hv
                .as_ref()
                .is_err_and(|reason| reason.contains("K = ∂H/∂t")),
        "premise: with no K the exact isometry Hessian refuses by naming K \
         (K = {unavailable_k}, max|Hv| = {unavailable_hv:?})"
    );

    let corrected = |evaluator: Arc<dyn SaeBasisEvaluator>| {
        let (term, _target, _rho) = periodic_fixture(evaluator);
        let iso = Arc::new(IsometryPenalty::new_euclidean(
            PsiSlice::full(term.assignment.coords[0].len(), Some(1)),
            2,
        ));
        term.corrected_isometry_penalty(&iso, 0, &term.assignment.coords[0])
    };
    match corrected(Arc::new(TestPeriodicEvaluator)) {
        Ok(AnalyticPenaltyKind::Isometry(penalty)) => assert!(
            penalty.third_decoder_derivative().is_some(),
            "the analytic twin's corrected isometry penalty must carry K"
        ),
        Ok(_) => panic!("corrected_isometry_penalty must return an isometry penalty"),
        Err(err) => panic!("the analytic twin must admit the isometry penalty, got {err}"),
    }
    match corrected(Arc::new(SecondOrderOnlyPeriodic)) {
        Err(ArrowSchurError::SchurFactorFailed { reason }) => assert!(
            reason.contains("declares its basis third jet unavailable"),
            "the isometry refusal must name the missing third jet, got: {reason}"
        ),
        Ok(_) => panic!(
            "an isometry penalty on an evaluator without a third jet must be refused, not built \
             with a zero-default exact Hessian"
        ),
        Err(other) => panic!("expected the third-jet capability refusal, got {other}"),
    }
}

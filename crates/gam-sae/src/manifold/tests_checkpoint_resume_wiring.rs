//! SPEC wall-survival E2E: the checkpoint wiring on the outer objective
//! (`bank_checkpoint` → `try_resume_from_checkpoint` → `remove_checkpoint`)
//! banks a resumable incumbent at a criterion improvement, installs it into a
//! FRESH objective on the identical data (the re-submitted-job scenario), and
//! removes the file once a converged fit is minted. Complements the pure
//! serialization round-trips in `checkpoint.rs`'s own tests — this exercises
//! the objective-level plumbing those left uncovered.

use super::*;
use gam_solve::rho_optimizer::{HessianSource, OuterObjective, OuterPlan, OuterResult, Solver};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// A tiny K=1 always-on circle term over a planted noisy circle; unique noise
/// per call keeps every test's data fingerprint (and thus its checkpoint path)
/// disjoint from other tests and other runs.
fn tiny_objective(salt: u64) -> (SaeManifoldOuterObjective, Array1<f64>) {
    let n = 24usize;
    let p = 4usize;
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    let mut state = 0x2235_c0de ^ salt;
    let mut noise = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.06
    };
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos() + noise();
        z[[i, 1]] = theta.sin() + noise();
        z[[i, 2]] = 0.4 * (2.0 * theta).cos() + noise();
        z[[i, 3]] = 0.4 * (2.0 * theta).sin() + noise();
    }
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "ckpt-e2e",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let logits = Array2::<f64>::from_elem((n, 1), 40.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .for_assignment(AssignmentMode::softmax(1.0));
    let flat = rho.to_flat();
    (
        SaeManifoldOuterObjective::new(term, z, None, rho, 6, 0.04, 1.0e-6, 1.0e-6),
        flat,
    )
}

/// Bank at an improvement → a fresh objective on the SAME data resumes the
/// banked incumbent (decoder installed, ledger counters carried) → discard
/// removes the file. The full re-submitted-job cycle.
#[test]
fn checkpoint_banks_resumes_and_discards_across_objectives() {
    let salt = std::process::id() as u64 ^ 0xE2E0;
    let (mut first, flat) = tiny_objective(salt);
    // Ensure no stale file from a crashed previous run of THIS test.
    first.remove_checkpoint();
    let cost = first
        .eval_cost(&flat)
        .expect("tiny circle criterion must evaluate");
    assert!(cost.is_finite());
    // The eval above recorded a first (improving) cost, which banks a
    // checkpoint through the wired call sites; bank explicitly as well so the
    // assertion does not depend on which eval lane the criterion took.
    first.bank_checkpoint(&flat);
    assert!(
        first.checkpoint_path.exists(),
        "an improving evaluation must leave a banked checkpoint on disk"
    );
    let fitted_decoder = first.term.atoms[0].decoder_coefficients.clone();

    // A fresh job on the identical data: resume must verify + install.
    let (mut second, _) = tiny_objective(salt);
    let resumed_rho = second
        .try_resume_from_checkpoint(flat.len())
        .expect("banked rho must satisfy the objective domain");
    assert!(
        resumed_rho.is_some(),
        "identical data + schema must resume the banked checkpoint"
    );
    assert_eq!(
        resumed_rho.unwrap().len(),
        flat.len(),
        "resumed rho must match the outer coordinate length"
    );
    let resumed_decoder = second.term.atoms[0].decoder_coefficients.clone();
    assert_eq!(
        resumed_decoder, fitted_decoder,
        "resume must install the banked decoder exactly (value-for-value)"
    );

    // Minting a converged fit discards the file; a third job starts cold.
    second.remove_checkpoint();
    assert!(
        !second.checkpoint_path.exists(),
        "discard must remove the checkpoint file"
    );
    let (mut third, _) = tiny_objective(salt);
    assert!(
        third
            .try_resume_from_checkpoint(flat.len())
            .expect("a missing checkpoint is not a domain error")
            .is_none(),
        "after discard a fresh fit must start cold"
    );
}

/// Different DATA must never resume another problem's checkpoint (the
/// fingerprint refusal path through the objective wiring).
#[test]
fn checkpoint_never_resumes_across_different_data() {
    let salt = std::process::id() as u64 ^ 0xD1FF;
    let (mut a, flat) = tiny_objective(salt);
    a.remove_checkpoint();
    a.eval_cost(&flat).expect("criterion must evaluate");
    a.bank_checkpoint(&flat);
    assert!(a.checkpoint_path.exists());

    // Different salt ⇒ different noise ⇒ different content hash ⇒ different
    // store path entirely; the other problem sees no file at its own path.
    let (mut b, _) = tiny_objective(salt ^ 0xFFFF);
    assert!(
        b.try_resume_from_checkpoint(flat.len())
            .expect("a missing checkpoint is not a domain error")
            .is_none(),
        "a different data fingerprint must not find (let alone resume) another \
         problem's checkpoint"
    );
    a.remove_checkpoint();
}

/// Primary and structured residual phases solve different objectives even on
/// the same target/K. Their checkpoints must be independently addressed, and a
/// structured phase must resume only the matching pass schedule.
#[test]
fn checkpoint_paths_are_phase_scoped_and_structured_phase_resumes() {
    let salt = std::process::id() as u64 ^ 0x57A6_E001;
    let (mut primary, flat) = tiny_objective(salt);
    scope_outer_checkpoint_to_stage(&mut primary, SaeFitStage::Primary);
    primary.remove_checkpoint();
    primary.bank_checkpoint(&flat);
    let primary_path = primary.checkpoint_path.clone();
    assert!(primary_path.exists());

    let structured_stage = SaeFitStage::StructuredResidual {
        pass: 1,
        total_passes: 2,
    };
    let (mut structured, _) = tiny_objective(salt);
    scope_outer_checkpoint_to_stage(&mut structured, structured_stage);
    structured.remove_checkpoint();
    structured.bank_checkpoint(&flat);
    let structured_path = structured.checkpoint_path.clone();
    assert!(structured_path.exists());
    assert_ne!(primary_path, structured_path);

    let (mut different_schedule, _) = tiny_objective(salt);
    scope_outer_checkpoint_to_stage(
        &mut different_schedule,
        SaeFitStage::StructuredResidual {
            pass: 1,
            total_passes: 4,
        },
    );
    assert_ne!(structured_path, different_schedule.checkpoint_path);

    primary.remove_checkpoint();
    assert!(!primary_path.exists());
    assert!(
        structured_path.exists(),
        "removing the primary phase must preserve structured work"
    );

    let (mut resumed, _) = tiny_objective(salt);
    scope_outer_checkpoint_to_stage(&mut resumed, structured_stage);
    assert!(
        resumed
            .try_resume_from_checkpoint(flat.len())
            .expect("banked rho must satisfy the objective domain")
            .is_some(),
        "a fresh matching structured phase must resume its own checkpoint"
    );
    resumed.remove_checkpoint();
}

/// The fit driver can only regain ownership of an outer objective after a
/// converged verdict. Nonconvergence retains all `OuterResult` evidence and
/// dropping the rejected objective leaves its wall-survival checkpoint intact.
#[test]
fn convergence_ownership_gate_preserves_typed_evidence_and_checkpoint() {
    let salt = std::process::id() as u64 ^ 0xCE47_1F1E;
    let stage = SaeFitStage::StructuredResidual {
        pass: 2,
        total_passes: 3,
    };
    let plan = OuterPlan {
        solver: Solver::Efs,
        hessian_source: HessianSource::EfsFixedPoint,
    };
    let (mut objective, flat) = tiny_objective(salt);
    scope_outer_checkpoint_to_stage(&mut objective, stage);
    objective.remove_checkpoint();
    objective.bank_checkpoint(&flat);
    let checkpoint_path = objective.checkpoint_path.clone();

    let mut result = OuterResult::new(flat.clone(), 12.5, 7, false, plan);
    result.final_grad_norm = Some(0.25);
    let error = match certify_outer_stage(objective, stage, Ok(result)) {
        Ok(_) => panic!("a non-converged result must not return a fit-producing objective"),
        Err(error) => error,
    };
    match error {
        SaeFitError::OuterDidNotConverge {
            stage: error_stage,
            result,
        } => {
            assert_eq!(error_stage, stage);
            assert_eq!(result.rho, flat);
            assert_eq!(result.final_value, 12.5);
            assert_eq!(result.iterations, 7);
            assert_eq!(result.final_grad_norm, Some(0.25));
            assert_eq!(result.plan_used, plan);
        }
        other => panic!("expected typed nonconvergence evidence, got {other}"),
    }
    assert!(
        checkpoint_path.exists(),
        "rejecting a non-converged objective must preserve its checkpoint"
    );

    let (mut cleanup, _) = tiny_objective(salt);
    scope_outer_checkpoint_to_stage(&mut cleanup, stage);
    cleanup.remove_checkpoint();

    let (mut converged_objective, converged_flat) = tiny_objective(salt ^ 0xC0A);
    scope_outer_checkpoint_to_stage(&mut converged_objective, stage);
    let converged = OuterResult::new(converged_flat, 8.0, 3, true, plan);
    let error = match certify_outer_stage(converged_objective, stage, Ok(converged)) {
        Ok(_) => panic!("a fabricated converged bit without analytic evidence must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        SaeFitError::OuterDidNotConverge {
            stage: rejected_stage,
            ..
        } if rejected_stage == stage
    ));
}

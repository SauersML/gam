//! #2996: a warm start filed from an inner result keeps none of its joint
//! Hessian workspace alive.
//!
//! `CustomOuterState` files a `ConstrainedWarmStart` in `warm_cache`,
//! `reset_warm_cache`, `pending_first_order_mode`, `walk_iterate`,
//! `value_probe` and one per ended walk in `walk_endpoints`. A BMS workspace
//! owns an `Arc` of the exact eval cache (the n-row row-primary blocks and cell
//! vectors). When the filed mode carried the workspace, every one of those
//! slots kept a whole cache alive past the two-slot exact-cache store, so
//! the live set grew with the number of walks. The outer working-set charge
//! counts a fixed number of live caches, and that count holds only while no
//! warm start retains one.
use super::*;
use crate::warm_start::cached_inner_mode_from_result;

#[test]
fn filed_warm_start_retains_no_joint_workspace_2996() {
    let beta = Array1::from_vec(vec![0.25, -0.5]);
    let rho = Array1::from_vec(vec![0.0]);
    let workspace: Arc<dyn ExactNewtonJointHessianWorkspace> = Arc::new(CountingHessianWorkspace {
        dense_calls: Arc::new(AtomicUsize::new(0)),
        matvec_calls: Arc::new(AtomicUsize::new(0)),
        source_preference: JointHessianSourcePreference::Dense,
    });
    let alive = Arc::downgrade(&workspace);
    let inner = BlockwiseInnerResult {
        cone_normalizer: None,
        solved_inner_tol: 1e-6,
        block_states: vec![ParameterBlockState {
            beta: beta.clone(),
            eta: Array1::zeros(1),
        }],
        terminal_working_sets: None,
        terminal_likelihood_score: None,
        active_sets: vec![None],
        log_likelihood: 0.0,
        penalty_value: 0.0,
        cycles: 1,
        converged: true,
        terminal_convergence_state: None,
        terminal_carrying_block: None,
        block_logdet_h: Some(0.0),
        block_logdet_s: Some(0.0),
        s_lambdas: vec![Array2::eye(2)],
        joint_workspace: Some(workspace),
        kkt_residual: None,
        active_constraints: None,
        objective_state: crate::assembly::InnerObjectiveState::unaugmented(&[rho.clone()], None),
    };

    let filed = ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: vec![beta.clone()],
        active_sets: vec![None],
        cached_inner: Some(cached_inner_mode_from_result(&inner)),
    };
    let walk_endpoints = vec![(vec![0u64], filed.clone()), (vec![1u64], filed.clone())];
    assert_eq!(alive.strong_count(), 1, "only the inner result owns the workspace");
    drop(inner);

    assert!(
        alive.upgrade().is_none(),
        "a filed warm start kept the joint workspace (and with it an exact eval cache) alive"
    );
    let kept = walk_endpoints[1].1.cached_inner.as_ref().expect("cached mode filed");
    assert_eq!(kept.cycles, 1);
    assert_eq!(filed.block_beta[0], beta);
}

use gam::custom_family::{
    BlockWorkingSet, CustomFamily, CustomFamilyBlockPsiDerivative,
    ExactNewtonJointPsiSecondOrderTerms, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, build_psi_hyper_coords, build_psi_pair_callbacks,
};
use gam::linalg::matrix::DesignMatrix;
use gam::matrix::DenseDesignMatrix;
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

#[derive(Clone)]
struct TinyFamily;
impl CustomFamily for TinyFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n = block_states[0].eta.len();
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: Array1::zeros(n),
                working_weights: Array1::ones(n),
            }],
        })
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        synced_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let _ = (
            synced_states.len(),
            specs.len(),
            derivative_blocks.len(),
            psi_i,
            psi_j,
        );
        Ok(None)
    }
}

fn one_block_spec() -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "b".into(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0, 0.0], [0.0, 1.0]])),
        offset: array![0.0, 0.0],
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(2))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: None,
    }
}

#[test]
fn bug_hunt_build_psi_hyper_coords_covers_each_penalty_axis_once_without_duplicates() {
    let deriv = CustomFamilyBlockPsiDerivative::new(
        Some(0),
        Array2::zeros((2, 2)),
        Array2::eye(2),
        None,
        None,
        None,
        None,
    );
    let coords = build_psi_hyper_coords(
        &TinyFamily,
        &[],
        &[one_block_spec()],
        &[vec![deriv]],
        &array![0.5, -0.1],
        &[0.0],
        &[1],
        None,
        false,
        None,
    )
    .expect("HyperCoord coverage should include every penalty axis exactly once with no gaps or duplicates");

    assert_eq!(
        coords.len(),
        2,
        "HyperCoord coverage should include every penalty axis exactly once with no gaps or duplicates"
    );
}

#[test]
fn bug_hunt_build_psi_pair_callbacks_out_of_registry_returns_documented_sentinel_instead_of_panicking()
 {
    let deriv = CustomFamilyBlockPsiDerivative::new(
        Some(0),
        Array2::zeros((2, 2)),
        Array2::eye(2),
        None,
        None,
        None,
        None,
    );
    let (ext_ext, _rho_ext) = build_psi_pair_callbacks(
        &TinyFamily,
        &[],
        &[one_block_spec()],
        Arc::new(vec![vec![deriv]]),
        &array![0.1, 0.2],
        &[0.0],
        &[1],
        None,
        None,
    )
    .expect("Pair callback registration should succeed for all declared coordinate pairs");

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ext_ext(99, 99);
    }))
    .is_err();

    assert!(
        !panicked,
        "Calling outside the registered (psi_i, psi_j) set should return an error/sentinel rather than panic"
    );
}

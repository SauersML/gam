use gam::terms::gated_decoder::GatedSAEDecoder;
use gam::terms::latent_coord::{LatentCoordValues, LatentIdMode};
use gam::terms::sae_manifold::{
    AssignmentMode, GumbelTemperatureSchedule, SaeAssignment, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm, ScheduleKind,
};
use ndarray::{Array1, Array2, Array3, array};

#[test]
fn latent_coord_assignment_decode_roundtrip_matches_dictionary_atom() {
    let coords =
        LatentCoordValues::from_matrix(array![[0.3, -0.7], [1.2, 0.5]].view(), LatentIdMode::None);
    let logits = array![[20.0, -20.0], [-20.0, 20.0]];
    let assignment = SaeAssignment::from_blocks_with_mode(
        logits,
        vec![coords.as_matrix(), coords.as_matrix()],
        AssignmentMode::softmax(1e-3),
    )
    .expect("test setup must construct assignment");

    let atom0 = SaeManifoldAtom::new(
        "a0",
        gam::terms::sae_manifold::SaeAtomBasisKind::Precomputed("dict0".into()),
        2,
        array![[1.0], [1.0]],
        Array3::zeros((2, 1, 2)),
        array![[2.0, -1.0]],
        array![[1.0]],
    )
    .expect("test setup must construct atom0");
    let atom1 = SaeManifoldAtom::new(
        "a1",
        gam::terms::sae_manifold::SaeAtomBasisKind::Precomputed("dict1".into()),
        2,
        array![[1.0], [1.0]],
        Array3::zeros((2, 1, 2)),
        array![[-3.0, 4.0]],
        array![[1.0]],
    )
    .expect("test setup must construct atom1");

    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).expect("term should build");
    let fitted = term.try_fitted().expect("fitted should evaluate");

    assert!(
        (fitted[[0, 0]] - 2.0).abs() < 1e-6
            && (fitted[[0, 1]] + 1.0).abs() < 1e-6
            && (fitted[[1, 0]] + 3.0).abs() < 1e-6
            && (fitted[[1, 1]] - 4.0).abs() < 1e-6,
        "Assignment-then-decode should round-trip each row back to its selected dictionary atom even at random latent_dim."
    );
}

#[test]
fn sae_assignment_modes_softmax_ibp_jumprelu_follow_documented_behavior() {
    let coord_blocks = vec![
        array![[0.0], [0.0]],
        array![[0.0], [0.0]],
        array![[0.0], [0.0]],
    ];

    let soft = SaeAssignment::from_blocks_with_mode(
        array![[1.0, 0.0, -1.0], [3.0, 2.0, 1.0]],
        coord_blocks.clone(),
        AssignmentMode::softmax(0.7),
    )
    .expect("softmax assignment should build");
    for row in 0..soft.n_obs() {
        let w = soft
            .try_assignments_row(row)
            .expect("softmax row should evaluate");
        let s: f64 = w.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-10,
            "Softmax assignments must be normalized and sum to one for every row."
        );
    }

    let ibp = SaeAssignment::from_blocks_with_mode(
        array![[0.0, 0.0, 0.0]],
        vec![array![[0.0]], array![[0.0]], array![[0.0]]],
        AssignmentMode::ibp_map(1.0, 0.1, false),
    )
    .expect("ibp assignment should build");
    let ibp_row = ibp.try_assignments_row(0).expect("ibp row should evaluate");
    assert!(
        ibp_row[0] > ibp_row[1] && ibp_row[1] > ibp_row[2],
        "IBPMap should reflect stick-breaking prior mass so earlier atoms are more likely than later atoms when logits are tied."
    );

    let jump = SaeAssignment::from_blocks_with_mode(
        array![[0.2, 0.6, -2.0]],
        vec![array![[0.0]], array![[0.0]], array![[0.0]]],
        AssignmentMode::jumprelu(0.5, 0.5),
    )
    .expect("jumprelu assignment should build");
    let jump_row = jump
        .try_assignments_row(0)
        .expect("jumprelu row should evaluate");
    assert!(
        jump_row[0] == 0.0 && jump_row[2] == 0.0 && jump_row[1] > 0.0,
        "JumpReLU should return sparse assignments with only logits above threshold receiving non-zero activation."
    );
}

#[test]
fn gated_sae_decoder_reconstructs_dictionary_atom_at_zero_residual() {
    let decoder =
        GatedSAEDecoder::new(Array2::eye(3), Array2::eye(3)).expect("decoder should build");
    let x = array![1.2, -0.8, 0.4];
    let y = decoder.decode_row(x.view()).expect("decode must succeed");
    assert!(
        (y[0] - x[0]).abs() < 1e-12 && (y[1] - x[1]).abs() < 1e-12 && (y[2] - x[2]).abs() < 1e-12,
        "With zero training residual and identity dictionary, gate activation plus linear decode should reproduce the original dictionary atom exactly."
    );
}

#[test]
fn update_ard_reml_matches_documented_map_rule() {
    let coords = array![[1.0, 2.0], [3.0, 4.0]];
    let assignment = SaeAssignment::from_blocks_with_mode(
        array![[0.0], [0.0]],
        vec![coords.clone()],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment should build");
    let atom = SaeManifoldAtom::new(
        "a",
        gam::terms::sae_manifold::SaeAtomBasisKind::Precomputed("dict".into()),
        2,
        array![[1.0], [1.0]],
        Array3::zeros((2, 1, 2)),
        array![[1.0]],
        array![[1.0]],
    )
    .expect("atom should build");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term should build");
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::zeros(2)]);
    term.update_ard_reml(&mut rho)
        .expect("ARD update should succeed");

    let n = 2.0;
    let expected0 = (n / (1.0_f64.powi(2) + 3.0_f64.powi(2))).ln();
    let expected1 = (n / (2.0_f64.powi(2) + 4.0_f64.powi(2))).ln();
    assert!(
        (rho.log_ard[0][0] - expected0).abs() < 1e-12
            && (rho.log_ard[0][1] - expected1).abs() < 1e-12,
        "ARD update must apply the documented MAP precision rule alpha_j = n / sum_i t_ij^2 for each latent axis."
    );
}

#[test]
fn temperature_schedule_is_applied_each_iteration_and_near_zero_behaves_like_argmax() {
    let logits = array![[2.0, 1.0, -4.0]];
    let mut term = {
        let assignment = SaeAssignment::from_blocks_with_mode(
            logits.clone(),
            vec![array![[0.0]], array![[0.0]], array![[0.0]]],
            AssignmentMode::softmax(2.0),
        )
        .expect("assignment should build");
        let atoms = (0..3)
            .map(|k| {
                SaeManifoldAtom::new(
                    format!("a{k}"),
                    gam::terms::sae_manifold::SaeAtomBasisKind::Precomputed("dict".into()),
                    1,
                    array![[1.0]],
                    Array3::zeros((1, 1, 1)),
                    array![[1.0]],
                    array![[1.0]],
                )
                .expect("atom should build")
            })
            .collect::<Vec<_>>();
        SaeManifoldTerm::new(atoms, assignment).expect("term should build")
    };

    let schedule = GumbelTemperatureSchedule::new(2.0, 1e-6, ScheduleKind::Geometric { rate: 0.1 })
        .expect("schedule should build");
    term.set_temperature_schedule(schedule)
        .expect("schedule should apply");
    assert!(
        (term.assignment.mode.temperature() - 2.0).abs() < 1e-12,
        "When a temperature schedule is provided, the assignment mode should immediately use the schedule temperature for the current iteration."
    );

    let near_zero = SaeAssignment::from_blocks_with_mode(
        logits,
        vec![array![[0.0]], array![[0.0]], array![[0.0]]],
        AssignmentMode::softmax(1e-6),
    )
    .expect("near-zero temperature assignment should build");
    let w = near_zero
        .try_assignments_row(0)
        .expect("assignments should evaluate");
    assert!(
        w[0] > 1.0 - 1e-8 && w[1] < 1e-8 && w[2] < 1e-8,
        "As temperature approaches zero, assignment should collapse to argmax on the largest logit."
    );
}

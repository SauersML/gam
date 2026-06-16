use gam::solver::evidence::{
    TopologyCandidate, TopologyKind, TopologyScoreScale, TopologySelectOptions, select_topology,
};

#[test]
fn select_topology_keeps_input_order_for_identical_scores_and_same_complexity() {
    let candidates = vec![
        TopologyCandidate {
            kind: TopologyKind::Flat,
            negative_log_evidence: 100.0,
            effective_dim: 10.0,
            n_obs: 50,
            converged: true,
            exclusion_reason: None,
        },
        TopologyCandidate {
            kind: TopologyKind::Flat,
            negative_log_evidence: 100.0,
            effective_dim: 10.0,
            n_obs: 50,
            converged: true,
            exclusion_reason: None,
        },
        TopologyCandidate {
            kind: TopologyKind::Periodic,
            negative_log_evidence: 100.0,
            effective_dim: 10.0,
            n_obs: 50,
            converged: true,
            exclusion_reason: None,
        },
    ];

    let selected = select_topology(
        &candidates,
        TopologySelectOptions {
            tie_tolerance: 0.0,
            score_scale: TopologyScoreScale::PerObservation,
        },
    );

    assert_eq!(
        selected.ranking.len(),
        candidates.len(),
        "Topology selection should preserve the input candidate count when nothing is rejected."
    );
    for (idx, candidate) in candidates.iter().enumerate() {
        assert!(
            selected.ranking[idx].kind == candidate.kind
                && (selected.ranking[idx].negative_log_evidence - candidate.negative_log_evidence)
                    .abs()
                    < 1e-12
                && (selected.ranking[idx].effective_dim - candidate.effective_dim).abs() < 1e-12
                && selected.ranking[idx].n_obs == candidate.n_obs,
            "Topology selection should keep a deterministic stable ordering when normalized scores are identical."
        );
    }
}

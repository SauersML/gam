//! #2899 P10 — the topology race fits every candidate and ranks the evidence each fitted
//! model publishes, in one Rust owner.

use crate::topology_selection::{
    TopologyCandidate, TopologyCandidateSource, TopologySelectionRequest, auto_topology_formula,
    select_topology,
};
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::inference::saved_summary::saved_model_summary;
use gam_solve::topology_formula::CandidateTopology;
use gam_solve::{TopologySelectionScoreKind, TopologySelectionScoreScale};

/// A periodic signal in an angle on `[0, 2π)` with a small deterministic residual.
fn angle_dataset(n: usize) -> EncodedDataset {
    let headers = vec!["y".to_string(), "theta".to_string()];
    let records = (0..n)
        .map(|row| {
            let theta = std::f64::consts::TAU * (row as f64 + 0.5) / n as f64;
            let y = theta.sin() + 0.4 * (2.0 * theta).cos() + 0.05 * (7.3 * row as f64).sin();
            csv::StringRecord::from(vec![format!("{y:.17e}"), format!("{theta:.17e}")])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode the angle dataset")
}

fn circle() -> CandidateTopology {
    CandidateTopology::PeriodicSplineCurve {
        n_knots: 12,
        degree: 3,
        penalty_order: 2,
        double_penalty: None,
    }
}

fn sphere() -> CandidateTopology {
    CandidateTopology::Sphere {
        n_centers: 20,
        penalty_order: 2,
        kernel: "sobolev".to_string(),
        radians: false,
        double_penalty: None,
    }
}

#[test]
fn the_auto_formula_smooths_every_non_response_column() {
    let dataset = angle_dataset(8);
    let (formula, dim) = auto_topology_formula(&dataset, "y").expect("formula");
    assert_eq!(formula, "y ~ s(theta, type=AUTO)");
    assert_eq!(dim, 1);
    assert!(auto_topology_formula(&dataset, "y ~ theta").is_err());
    assert!(auto_topology_formula(&dataset, "missing").is_err());
}

/// Every ranked candidate's evidence is read off its own fitted model's summary, and a
/// candidate that cannot take the predictors' dimension is a failed outcome when the
/// caller named it and not a candidate at all in a default portfolio.
#[test]
fn candidates_rank_on_their_own_summaries_and_misfits_follow_the_source() {
    let dataset = angle_dataset(60);
    let run = |source| {
        select_topology(
            &dataset,
            TopologySelectionRequest {
                formula: "y ~ s(theta, type=AUTO)",
                candidates: vec![
                    TopologyCandidate { name: "circle".to_string(), topology: circle() },
                    TopologyCandidate {
                        name: "circle_coarse".to_string(),
                        topology: CandidateTopology::PeriodicSplineCurve {
                            n_knots: 6,
                            degree: 3,
                            penalty_order: 2,
                            double_penalty: None,
                        },
                    },
                    TopologyCandidate { name: "sphere".to_string(), topology: sphere() },
                ],
                source,
                latent: None,
                score_kind: TopologySelectionScoreKind::Reml,
                score_scale: TopologySelectionScoreScale::Raw,
                config_json: None,
            },
        )
        .expect("the race runs")
    };

    let explicit = run(TopologyCandidateSource::Explicit);
    assert_eq!(explicit.result.ranked.len(), 2);
    assert_eq!(explicit.result.failed.len(), 1);
    assert_eq!(explicit.result.failed[0].name, "sphere");
    assert_eq!(explicit.result.failed[0].stage.as_str(), "assembly");
    for row in &explicit.result.ranked {
        let (_, model) = explicit
            .fits
            .iter()
            .find(|(name, _)| name == &row.name)
            .expect("every ranked candidate returns its model");
        let summary = saved_model_summary(model).expect("summary");
        assert_eq!(Some(row.raw_reml), summary.raw_reml_score);
        assert_eq!(row.score, row.raw_reml, "raw scale ranks the raw criterion");
        assert_eq!(Some(row.effective_dim), summary.edf_total);
        assert_eq!(row.basis_size, summary.coefficients.len());
        assert_eq!(row.n_obs, 60);
    }
    assert!(explicit.result.ranked[0].score <= explicit.result.ranked[1].score);
    assert_eq!(explicit.result.winner_index, Some(0));

    let defaults = run(TopologyCandidateSource::Defaults);
    assert!(defaults.result.failed.is_empty(), "{:?}", defaults.result.failed);
    assert_eq!(defaults.result.ranked.len(), 2);
}

#[test]
fn duplicate_names_and_a_lone_candidate_are_refused() {
    let dataset = angle_dataset(12);
    let request = |candidates| TopologySelectionRequest {
        formula: "y ~ s(theta, type=AUTO)",
        candidates,
        source: TopologyCandidateSource::Explicit,
        latent: None,
        score_kind: TopologySelectionScoreKind::Reml,
        score_scale: TopologySelectionScoreScale::Raw,
        config_json: None,
    };
    let duplicate = vec![
        TopologyCandidate { name: "circle".to_string(), topology: circle() },
        TopologyCandidate { name: "circle".to_string(), topology: circle() },
    ];
    assert!(select_topology(&dataset, request(duplicate)).is_err());
    let lone = vec![TopologyCandidate { name: "circle".to_string(), topology: circle() }];
    assert!(select_topology(&dataset, request(lone)).is_err());
}

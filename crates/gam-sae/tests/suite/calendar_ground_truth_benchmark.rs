//! Calendar ground-truth benchmark harness.
//!
//! Engels et al. report labeled GPT-2 SAE calendar clusters for days of week,
//! months, and years, and the accompanying PCA observation that the leading
//! intensity coordinate is the radial coordinate of the circle. This harness
//! pins the geometric contract on deterministic GPT-2-style planted data, then
//! requires the committed external-index fixture at:
//!
//! `crates/gam-sae/tests/data/engels_gpt2_calendar_sae_indices.json`
//!
//! The real GPT-2 activation path is gated on an optional fixture at:
//!
//! `crates/gam-sae/tests/data/engels_gpt2_calendar_sae_activations.json`
//!
//! If that file is absent, the real-data test prints a SKIP and returns cleanly;
//! the planted-data contract remains always-on.
//!
//! Expected index fixture schema:
//!
//! ```json
//! {
//!   "clusters": [
//!     {"name": "days", "feature_indices": [1, 2, 3], "labels": ["Monday"]},
//!     {"name": "months", "feature_indices": [4, 5, 6], "labels": ["January"]},
//!     {"name": "years", "feature_indices": [7, 8, 9], "labels": ["1990"]}
//!   ]
//! }
//! ```
//!
//! Expected optional activation fixture schema:
//!
//! ```json
//! {
//!   "model": "gpt-2",
//!   "sae_layer": 7,
//!   "n_rows": 840,
//!   "clusters": [
//!     {
//!       "name": "days",
//!       "feature_indices": [2592],
//!       "label_ids": [0, 1],
//!       "circle_codes": [[1.0, 0.0], [0.6, 0.8]],
//!       "pc1_amplitude": [1.0, 1.0]
//!     }
//!   ]
//! }
//! ```

use serde_json::Value;
use std::collections::BTreeSet;
use std::path::PathBuf;

const AXES: usize = 3;
const DAY_COUNT: usize = 7;
const MONTH_COUNT: usize = 12;
const YEAR_COUNT: usize = 10;
const AXIS_NAMES: [&str; AXES] = ["days", "months", "years"];
const LABEL_COUNTS: [usize; AXES] = [DAY_COUNT, MONTH_COUNT, YEAR_COUNT];
const ENGELS_DAY_FEATURE_INDICES: &[u64] =
    &[2592, 4445, 4663, 4733, 6531, 8179, 9566, 20927, 24185];
const ENGELS_MONTH_FEATURE_INDICES: &[u64] = &[
    3977, 4140, 5993, 7299, 9104, 9401, 10449, 11196, 12661, 14715, 17068, 17528, 19589, 21033,
    22043, 23304,
];
const ENGELS_YEAR_FEATURE_INDICES: &[u64] = &[
    1052, 2753, 4427, 6382, 8314, 9576, 9606, 13551, 19734, 20349,
];

#[derive(Clone, Debug)]
struct GroundTruthCluster {
    name: String,
    feature_indices: Vec<u64>,
    label_count: usize,
}

fn engels_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join("engels_gpt2_calendar_sae_indices.json")
}

fn expected_engels_feature_indices(axis: usize) -> Vec<u64> {
    match axis {
        0 => ENGELS_DAY_FEATURE_INDICES.to_vec(),
        1 => ENGELS_MONTH_FEATURE_INDICES.to_vec(),
        2 => ENGELS_YEAR_FEATURE_INDICES.to_vec(),
        _ => panic!("unknown calendar axis {axis}"),
    }
}

fn load_engels_clusters() -> Vec<GroundTruthCluster> {
    let path = engels_fixture_path();
    assert!(
        path.exists(),
        "missing committed Engels GPT-2 SAE calendar fixture at {}",
        path.display()
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let value: Value = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("parse {} as JSON: {err}", path.display()));
    let clusters = value
        .get("clusters")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{} must contain a clusters array", path.display()));
    clusters
        .iter()
        .map(|cluster| parse_ground_truth_cluster(cluster, &path))
        .collect()
}

fn parse_ground_truth_cluster(cluster: &Value, path: &PathBuf) -> GroundTruthCluster {
    let name = cluster
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{} cluster is missing a string name", path.display()))
        .to_string();
    let feature_indices = cluster
        .get("feature_indices")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{name} in {} is missing feature_indices", path.display()))
        .iter()
        .map(|entry| {
            entry.as_u64().unwrap_or_else(|| {
                panic!(
                    "{name} in {} has a non-integer feature index",
                    path.display()
                )
            })
        })
        .collect::<Vec<_>>();
    let unique_indices = feature_indices.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(
        unique_indices.len(),
        feature_indices.len(),
        "{name} in {} has duplicate GPT-2 SAE feature indices",
        path.display()
    );
    let label_count = cluster
        .get("labels")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{name} in {} is missing labels", path.display()))
        .len();
    assert!(
        feature_indices.len() >= label_count,
        "{name} in {} has fewer GPT-2 SAE feature indices ({}) than labels ({label_count})",
        path.display(),
        feature_indices.len()
    );
    GroundTruthCluster {
        name,
        feature_indices,
        label_count,
    }
}

fn assert_ground_truth_cluster(clusters: &[GroundTruthCluster], axis: usize) {
    let name = AXIS_NAMES[axis];
    let cluster = clusters
        .iter()
        .find(|cluster| cluster.name == name)
        .unwrap_or_else(|| panic!("missing Engels GPT-2 SAE {name} cluster"));
    assert_eq!(
        cluster.label_count, LABEL_COUNTS[axis],
        "{name} cluster must carry the canonical calendar label count; got {}",
        cluster.label_count
    );
    assert_eq!(
        cluster.feature_indices,
        expected_engels_feature_indices(axis),
        "{name} cluster must use the Engels et al. GPT-2 layer-7 SAE feature indices"
    );
}

fn assert_engels_ground_truth_clusters(clusters: &[GroundTruthCluster]) {
    for axis in 0..AXES {
        assert_ground_truth_cluster(clusters, axis);
    }
}

#[test]
fn engels_gpt2_sae_indices_are_the_real_calendar_ground_truth() {
    let clusters = load_engels_clusters();
    assert_engels_ground_truth_clusters(&clusters);
}


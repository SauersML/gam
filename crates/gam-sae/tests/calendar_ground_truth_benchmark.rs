//! Calendar ground-truth benchmark harness.
//!
//! Engels et al. report labeled GPT-2 SAE calendar clusters for days of week,
//! months, and years, and the accompanying PCA observation that the leading
//! intensity coordinate is the radial coordinate of the circle. This harness
//! pins that contract on deterministic GPT-2-style planted data, with an
//! optional real-index fixture path at:
//!
//! `crates/gam-sae/tests/data/engels_gpt2_calendar_sae_indices.json`
//!
//! Expected optional schema:
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

use gam_sae::sparse_dict::{
    BlockChartComposeConfig, BlockCoordinateReport, BlockSparseFit, block_firing_coordinates,
    compose_block_coordinate_charts,
};
use ndarray::{Array2, Array3};
use serde_json::Value;
use std::f64::consts::TAU;
use std::path::PathBuf;

const AXES: usize = 3;
const BLOCK_SIZE: usize = 2;
const DAY_AXIS: usize = 0;
const DAY_COUNT: usize = 7;
const MONTH_COUNT: usize = 12;
const YEAR_COUNT: usize = 10;
const ROWS: usize = 840;
const AXIS_NAMES: [&str; AXES] = ["days", "months", "years"];
const LABEL_COUNTS: [usize; AXES] = [DAY_COUNT, MONTH_COUNT, YEAR_COUNT];

#[derive(Clone)]
struct CalendarFixture {
    x: Array2<f32>,
    decoder: Array2<f32>,
    blocks: Array2<u32>,
    gates: Array2<f32>,
    codes: Array3<f32>,
    labels: Vec<[usize; AXES]>,
    pc1_amplitude: Array2<f64>,
}

#[derive(Clone, Debug)]
struct GroundTruthCluster {
    name: String,
    feature_count: usize,
    label_count: usize,
}

fn planted_calendar_fixture() -> CalendarFixture {
    let p = AXES * 3;
    let mut x = Array2::<f32>::zeros((ROWS, p));
    let mut decoder = Array2::<f32>::zeros((AXES * BLOCK_SIZE, p));
    let mut blocks = Array2::<u32>::zeros((ROWS, AXES));
    let mut gates = Array2::<f32>::zeros((ROWS, AXES));
    let mut codes = Array3::<f32>::zeros((ROWS, AXES, BLOCK_SIZE));
    let mut labels = Vec::with_capacity(ROWS);
    let mut pc1_amplitude = Array2::<f64>::zeros((ROWS, AXES));

    for axis in 0..AXES {
        decoder[[axis * BLOCK_SIZE, axis * 3 + 1]] = 1.0;
        decoder[[axis * BLOCK_SIZE + 1, axis * 3 + 2]] = 1.0;
    }

    for row in 0..ROWS {
        let day = row % DAY_COUNT;
        let month = (5 * row) % MONTH_COUNT;
        let year = (7 * row) % YEAR_COUNT;
        let row_labels = [day, month, year];
        labels.push(row_labels);
        for axis in 0..AXES {
            let label = row_labels[axis];
            let phase = (label as f64 / LABEL_COUNTS[axis] as f64 + axis_offset(axis)).fract();
            let radius = planted_radius(row, axis);
            let (sin_phase, cos_phase) = (TAU * phase).sin_cos();
            let x0 = radius * cos_phase;
            let x1 = radius * sin_phase;
            let base = axis * 3;
            x[[row, base]] = radius as f32;
            x[[row, base + 1]] = x0 as f32;
            x[[row, base + 2]] = x1 as f32;
            blocks[[row, axis]] = axis as u32;
            gates[[row, axis]] = radius as f32;
            codes[[row, axis, 0]] = x0 as f32;
            codes[[row, axis, 1]] = x1 as f32;
            pc1_amplitude[[row, axis]] = radius;
        }
    }

    CalendarFixture {
        x,
        decoder,
        blocks,
        gates,
        codes,
        labels,
        pc1_amplitude,
    }
}

fn axis_offset(axis: usize) -> f64 {
    [0.17, 0.29, 0.41][axis]
}

fn planted_radius(row: usize, axis: usize) -> f64 {
    let bucket = ((row * (axis + 3) + 11 * axis) % 17) as f64;
    1.0 + 0.12 * ((bucket - 8.0) / 8.0) + 0.08 * axis as f64
}

fn block_fit(fixture: &CalendarFixture) -> BlockSparseFit {
    BlockSparseFit {
        decoder: fixture.decoder.clone(),
        blocks: fixture.blocks.clone(),
        gates: fixture.gates.clone(),
        codes: fixture.codes.clone(),
        gamma: 1.0,
        block_utilization: vec![1.0; AXES],
        block_stable_rank: vec![2.0; AXES],
        explained_variance: 1.0,
        epochs: 0,
        converged: true,
        block_topk: AXES,
        block_size: BLOCK_SIZE,
    }
}

fn chart_config() -> BlockChartComposeConfig {
    BlockChartComposeConfig {
        block_size: BLOCK_SIZE,
        block_topk: AXES,
        gamma: 1.0,
        residual_target: false,
        min_firings: 64,
        max_blocks: AXES,
        crossfit_folds: 6,
        alpha: 1.0,
        min_effect: 0.0,
        whitening_ridge: 1.0e-8,
        pair_screen: false,
        pair_top_blocks: 0,
        max_pairs: 0,
        pair_min_cofirings: 0,
        pair_min_score: 0.0,
    }
}

fn assert_calendar_circles_promoted(fixture: &CalendarFixture) {
    let result = compose_block_coordinate_charts(
        fixture.x.view(),
        fixture.decoder.view(),
        fixture.blocks.view(),
        fixture.codes.view(),
        &chart_config(),
    )
    .unwrap_or_else(|err| panic!("calendar block-chart promotion failed: {err}"));

    assert_eq!(
        result.accepted_blocks.len(),
        AXES,
        "block-chart promotion must accept one circle atom per calendar axis"
    );
    for axis in 0..AXES {
        assert!(
            result.accepted_blocks.contains(&axis),
            "missing promoted CIRCLE atom for {} cluster",
            AXIS_NAMES[axis]
        );
        let record = result
            .block_records
            .iter()
            .find(|record| record.block0 == axis)
            .unwrap_or_else(|| panic!("missing block-chart record for {}", AXIS_NAMES[axis]));
        assert!(
            record.evidence.accepted,
            "{} cluster was selected but not accepted as a chart atom",
            AXIS_NAMES[axis]
        );
        assert!(
            record.evidence.chart_loss <= 0.05 * record.evidence.linear_loss,
            "{} cluster must be circle/radial, not rank-1 linear: chart loss {} vs linear loss {}",
            AXIS_NAMES[axis],
            record.evidence.chart_loss,
            record.evidence.linear_loss
        );
    }
}

fn circular_forward_delta(anchor: f64, phase: f64) -> f64 {
    (phase - anchor).rem_euclid(1.0)
}

fn circular_mean(phases: &[f64]) -> f64 {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    for phase in phases {
        let (sin_phase, cos_phase) = (TAU * phase).sin_cos();
        sin_sum += sin_phase;
        cos_sum += cos_phase;
    }
    (sin_sum.atan2(cos_sum) / TAU).rem_euclid(1.0)
}

fn assert_day_phase_order_and_closed_form_se(fixture: &CalendarFixture) {
    let fit = block_fit(fixture);
    let report = block_firing_coordinates(&fit, DAY_AXIS)
        .unwrap_or_else(|err| panic!("day coordinate readout failed: {err}"));

    let mut phases_by_day = vec![Vec::<f64>::new(); DAY_COUNT];
    for firing in &report.firings {
        let day = fixture.labels[firing.row][DAY_AXIS];
        phases_by_day[day].push(firing.t);
        let raw_se = report.sigma_hat / (TAU * firing.amplitude);
        let uniform_ceiling = (1.0f64 / 12.0).sqrt();
        let expected_se = if raw_se >= uniform_ceiling {
            uniform_ceiling
        } else {
            raw_se
        };
        assert!(
            (firing.t_se - expected_se).abs() <= 1.0e-12 * expected_se.max(1.0),
            "closed-form day phase SE mismatch on row {}: got {}, expected {}",
            firing.row,
            firing.t_se,
            expected_se
        );
        assert!(
            !firing.t_se_clamped,
            "planted calendar day firing should be strong enough to avoid uniform SE clamping"
        );
    }

    let phase_by_day = phases_by_day
        .iter()
        .map(|phases| {
            assert!(!phases.is_empty(), "missing firings for a day label");
            circular_mean(phases)
        })
        .collect::<Vec<_>>();
    let anchor = phase_by_day[0];
    for day in 0..DAY_COUNT {
        let recovered = circular_forward_delta(anchor, phase_by_day[day]);
        let expected = day as f64 / DAY_COUNT as f64;
        assert!(
            (recovered - expected).abs() <= 2.0e-6,
            "day phase order mismatch for label {day}: got {recovered}, expected {expected}"
        );
    }
}

fn assert_pc1_intensity_equals_circle_radius(fixture: &CalendarFixture) {
    let fit = block_fit(fixture);
    for axis in 0..AXES {
        let report = block_firing_coordinates(&fit, axis)
            .unwrap_or_else(|err| panic!("{} coordinate readout failed: {err}", AXIS_NAMES[axis]));
        assert_report_radius_matches_pc1(fixture, axis, &report);
    }
}

fn assert_report_radius_matches_pc1(
    fixture: &CalendarFixture,
    axis: usize,
    report: &BlockCoordinateReport,
) {
    assert_eq!(
        report.n_firings, ROWS,
        "{} cluster must fire on every planted calendar row",
        AXIS_NAMES[axis]
    );
    for firing in &report.firings {
        let pc1 = fixture.pc1_amplitude[[firing.row, axis]];
        let base = axis * 3;
        let circle_radius = (fixture.x[[firing.row, base + 1]] as f64)
            .hypot(fixture.x[[firing.row, base + 2]] as f64);
        assert!(
            (pc1 - firing.amplitude).abs() <= 1.0e-5,
            "{} row {} violates PC-1 intensity = circle radius: pc1 {}, readout {}",
            AXIS_NAMES[axis],
            firing.row,
            pc1,
            firing.amplitude
        );
        assert!(
            (circle_radius - firing.amplitude).abs() <= 1.0e-6,
            "{} row {} violates cone/radial law in ambient coordinates: norm {}, readout {}",
            AXIS_NAMES[axis],
            firing.row,
            circle_radius,
            firing.amplitude
        );
    }
}

fn optional_engels_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join("engels_gpt2_calendar_sae_indices.json")
}

fn load_optional_engels_clusters() -> Option<Vec<GroundTruthCluster>> {
    let path = optional_engels_fixture_path();
    if !path.exists() {
        return None;
    }
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let value: Value = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("parse {} as JSON: {err}", path.display()));
    let clusters = value
        .get("clusters")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{} must contain a clusters array", path.display()));
    Some(
        clusters
            .iter()
            .map(|cluster| parse_ground_truth_cluster(cluster, &path))
            .collect(),
    )
}

fn parse_ground_truth_cluster(cluster: &Value, path: &PathBuf) -> GroundTruthCluster {
    let name = cluster
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{} cluster is missing a string name", path.display()))
        .to_string();
    let feature_count = cluster
        .get("feature_indices")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{name} in {} is missing feature_indices", path.display()))
        .len();
    let label_count = cluster
        .get("labels")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{name} in {} is missing labels", path.display()))
        .len();
    assert!(
        feature_count >= label_count,
        "{name} in {} has fewer GPT-2 SAE feature indices ({feature_count}) than labels ({label_count})",
        path.display()
    );
    GroundTruthCluster {
        name,
        feature_count,
        label_count,
    }
}

fn assert_ground_truth_cluster(
    clusters: &[GroundTruthCluster],
    name: &str,
    minimum_label_count: usize,
) {
    let cluster = clusters
        .iter()
        .find(|cluster| cluster.name == name)
        .unwrap_or_else(|| panic!("missing Engels GPT-2 SAE {name} cluster"));
    assert!(
        cluster.label_count >= minimum_label_count,
        "{name} cluster must carry at least {minimum_label_count} labels; got {}",
        cluster.label_count
    );
    assert!(
        cluster.feature_count >= cluster.label_count,
        "{name} cluster must have labeled GPT-2 SAE feature indices"
    );
}

#[test]
fn planted_calendar_block_chart_promotion_recovers_circle_atoms() {
    let fixture = planted_calendar_fixture();
    assert_calendar_circles_promoted(&fixture);
}

#[test]
fn planted_calendar_coordinate_readout_orders_weekdays_with_closed_form_se() {
    let fixture = planted_calendar_fixture();
    assert_day_phase_order_and_closed_form_se(&fixture);
}

#[test]
fn planted_calendar_pc1_intensity_is_circle_radius() {
    let fixture = planted_calendar_fixture();
    assert_pc1_intensity_equals_circle_radius(&fixture);
}

#[test]
fn optional_engels_gpt2_sae_indices_feed_the_same_calendar_contract() {
    let Some(clusters) = load_optional_engels_clusters() else {
        return;
    };
    assert_ground_truth_cluster(&clusters, "days", DAY_COUNT);
    assert_ground_truth_cluster(&clusters, "months", MONTH_COUNT);
    assert_ground_truth_cluster(&clusters, "years", 2);

    let fixture = planted_calendar_fixture();
    assert_calendar_circles_promoted(&fixture);
    assert_day_phase_order_and_closed_form_se(&fixture);
    assert_pc1_intensity_equals_circle_radius(&fixture);
}

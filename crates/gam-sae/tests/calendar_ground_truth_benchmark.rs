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

use gam_sae::sparse_dict::{
    BlockChartComposeConfig, BlockCoordinateReport, BlockSparseFit, block_firing_coordinates,
    compose_block_coordinate_charts,
};
use gam_sae::manifold::{
    GraphCompressionKind, LearnedGraphAtom, OccupancyLaw, graph_edge_rank_charge,
};
use gam_sae::structure_harvest::graph_birth_candidate_for_structure_search;
use ndarray::{Array2, Array3};
use serde_json::Value;
use std::collections::BTreeSet;
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
const DAY_ORDER_SE_MULTIPLIER: f64 = 1.0;
const ENGELS_DAY_FEATURE_INDICES: &[u64] =
    &[2592, 4445, 4663, 4733, 6531, 8179, 9566, 20927, 24185];
const ENGELS_MONTH_FEATURE_INDICES: &[u64] = &[
    3977, 4140, 5993, 7299, 9104, 9401, 10449, 11196, 12661, 14715, 17068, 17528, 19589, 21033,
    22043, 23304,
];
const ENGELS_YEAR_FEATURE_INDICES: &[u64] =
    &[1052, 2753, 4427, 6382, 8314, 9576, 9606, 13551, 19734, 20349];

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
    feature_indices: Vec<u64>,
    label_count: usize,
}

struct RealClusterData {
    axis: usize,
    feature_indices: Vec<u64>,
    label_ids: Vec<usize>,
    circle_codes: Vec<[f64; 2]>,
    pc1_amplitude: Vec<f64>,
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

fn calendar_fixture_from_real_clusters(n_rows: usize, clusters: Vec<RealClusterData>) -> CalendarFixture {
    assert_eq!(
        clusters.len(),
        AXES,
        "real Engels activation fixture must contain days/months/years clusters"
    );
    let p = AXES * 3;
    let mut x = Array2::<f32>::zeros((n_rows, p));
    let mut decoder = Array2::<f32>::zeros((AXES * BLOCK_SIZE, p));
    let mut blocks = Array2::<u32>::zeros((n_rows, AXES));
    let mut gates = Array2::<f32>::zeros((n_rows, AXES));
    let mut codes = Array3::<f32>::zeros((n_rows, AXES, BLOCK_SIZE));
    let mut labels = vec![[0usize; AXES]; n_rows];
    let mut pc1_amplitude = Array2::<f64>::zeros((n_rows, AXES));

    for axis in 0..AXES {
        decoder[[axis * BLOCK_SIZE, axis * 3 + 1]] = 1.0;
        decoder[[axis * BLOCK_SIZE + 1, axis * 3 + 2]] = 1.0;
    }

    for cluster in clusters {
        assert_eq!(
            cluster.feature_indices,
            expected_engels_feature_indices(cluster.axis),
            "{} real activation cluster is not keyed by the Engels GPT-2 SAE feature indices",
            AXIS_NAMES[cluster.axis]
        );
        assert_eq!(
            cluster.label_ids.len(),
            n_rows,
            "{} label_ids length must match n_rows",
            AXIS_NAMES[cluster.axis]
        );
        assert_eq!(
            cluster.circle_codes.len(),
            n_rows,
            "{} circle_codes length must match n_rows",
            AXIS_NAMES[cluster.axis]
        );
        assert_eq!(
            cluster.pc1_amplitude.len(),
            n_rows,
            "{} pc1_amplitude length must match n_rows",
            AXIS_NAMES[cluster.axis]
        );
        for row in 0..n_rows {
            let label = cluster.label_ids[row];
            assert!(
                label < LABEL_COUNTS[cluster.axis],
                "{} row {} has label id {} outside 0..{}",
                AXIS_NAMES[cluster.axis],
                row,
                label,
                LABEL_COUNTS[cluster.axis]
            );
            let [x0, x1] = cluster.circle_codes[row];
            assert!(
                x0.is_finite() && x1.is_finite(),
                "{} row {} has a non-finite circle code",
                AXIS_NAMES[cluster.axis],
                row
            );
            let pc1 = cluster.pc1_amplitude[row];
            assert!(
                pc1.is_finite() && pc1 >= 0.0,
                "{} row {} has a non-finite or negative PC-1 amplitude",
                AXIS_NAMES[cluster.axis],
                row
            );
            let axis = cluster.axis;
            let base = axis * 3;
            let radius = x0.hypot(x1);
            labels[row][axis] = label;
            x[[row, base]] = pc1 as f32;
            x[[row, base + 1]] = x0 as f32;
            x[[row, base + 2]] = x1 as f32;
            blocks[[row, axis]] = axis as u32;
            gates[[row, axis]] = radius as f32;
            codes[[row, axis, 0]] = x0 as f32;
            codes[[row, axis, 1]] = x1 as f32;
            pc1_amplitude[[row, axis]] = pc1;
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
        matryoshka_prefix_losses: Vec::new(),
        explained_variance: 1.0,
        epochs: 0,
        convergence: gam_sae::sparse_dict::BlockSparseConvergence::trivially_converged(),
        block_topk: AXES,
        block_size: BLOCK_SIZE,
    }
}

fn chart_config() -> BlockChartComposeConfig {
    BlockChartComposeConfig {
        block_size: BLOCK_SIZE,
        block_tile: BLOCK_SIZE,
        block_topk: AXES,
        gamma: 1.0,
        residual_target: false,
        min_firings: 64,
        max_blocks: AXES,
        crossfit_folds: 6,
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
        result.selected_chart_blocks.len(),
        AXES,
        "block-chart promotion must accept one circle atom per calendar axis"
    );
    for axis in 0..AXES {
        assert!(
            result.selected_chart_blocks.contains(&axis),
            "missing promoted CIRCLE atom for {} cluster",
            AXIS_NAMES[axis]
        );
        let record = result
            .block_records
            .iter()
            .find(|record| record.block0 == axis)
            .unwrap_or_else(|| panic!("missing block-chart record for {}", AXIS_NAMES[axis]));
        assert!(
            record.evidence.selected_by_bic,
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
        assert_axis_circle_topology(fixture, axis);
    }
}

fn ideal_calendar_axis_anchor_embeddings(axis: usize) -> Array2<f64> {
    let anchors = LABEL_COUNTS[axis];
    Array2::<f64>::from_shape_fn((anchors, 2), |(label, col)| {
        let phase = (label as f64 / anchors as f64 + axis_offset(axis)).fract();
        let (sin_phase, cos_phase) = (TAU * phase).sin_cos();
        if col == 0 { cos_phase } else { sin_phase }
    })
}

fn recovered_calendar_axis_anchor_embeddings(fixture: &CalendarFixture, axis: usize) -> Array2<f64> {
    let fit = block_fit(fixture);
    let report = block_firing_coordinates(&fit, axis)
        .unwrap_or_else(|err| panic!("{} coordinate readout failed: {err}", AXIS_NAMES[axis]));
    let anchors = LABEL_COUNTS[axis];
    let mut phases_by_label = vec![Vec::<f64>::new(); anchors];
    for firing in &report.firings {
        let label = fixture.labels[firing.row][axis];
        phases_by_label[label].push(firing.t);
    }
    Array2::<f64>::from_shape_fn((anchors, 2), |(label, col)| {
        let phases = &phases_by_label[label];
        assert!(
            !phases.is_empty(),
            "{} recovered topology is missing label {}",
            AXIS_NAMES[axis],
            label
        );
        let phase = circular_mean(phases);
        let (sin_phase, cos_phase) = (TAU * phase).sin_cos();
        if col == 0 { cos_phase } else { sin_phase }
    })
}

fn calendar_axis_rows(fixture: &CalendarFixture, axis: usize) -> Vec<f64> {
    let anchors = LABEL_COUNTS[axis];
    let n_rows = fixture.labels.len();
    fixture
        .labels
        .iter()
        .enumerate()
        .map(|(row, labels)| {
            let jitter_rank = row % anchors;
            let jitter = jitter_rank as f64 / (n_rows * anchors * anchors) as f64;
            (labels[axis] as f64 / anchors as f64 + jitter).rem_euclid(1.0)
        })
        .collect()
}

fn assert_axis_circle_topology(fixture: &CalendarFixture, axis: usize) {
    let embeddings = recovered_calendar_axis_anchor_embeddings(fixture, axis);
    let rows = calendar_axis_rows(fixture, axis);
    let candidate_edges =
        LearnedGraphAtom::knn_candidate_edges(embeddings.view()).expect("calendar kNN graph");
    let n_eff = rows.len() as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let precisions = vec![1.0; candidate_edges.len()];
    let deltas = vec![charge * 1.4; candidate_edges.len()];
    let candidate =
        graph_birth_candidate_for_structure_search(embeddings.view(), &rows, n_eff, &precisions, &deltas)
            .unwrap_or_else(|err| {
                panic!("{} graph birth candidate failed: {err}", AXIS_NAMES[axis])
            });
    let readout = &candidate.selection.topology;

    assert!(
        candidate.selection.selected,
        "{} cluster must be selected as a graph atom by summed edge charge",
        AXIS_NAMES[axis]
    );
    assert_eq!(readout.b0, 1, "{} graph Betti b0", AXIS_NAMES[axis]);
    assert_eq!(readout.b1, 1, "{} graph Betti b1", AXIS_NAMES[axis]);
    assert_eq!(
        candidate.selection.occupancy,
        OccupancyLaw::Discrete {
            anchors: LABEL_COUNTS[axis]
        },
        "{} calendar occupancy must be atomic/discrete",
        AXIS_NAMES[axis]
    );
    assert_eq!(
        candidate.selection.compression.kind,
        GraphCompressionKind::Circle,
        "{} graph can certify a CIRCLE atom only after Betti (1,1) selection",
        AXIS_NAMES[axis]
    );
}

fn assert_calendar_graph_atoms_selected(fixture: &CalendarFixture) {
    for axis in 0..AXES {
        assert_axis_circle_topology(fixture, axis);
    }
}

fn assert_calendar_continuous_control_is_not_atomic() {
    let embeddings = ideal_calendar_axis_anchor_embeddings(DAY_AXIS);
    let rows = (0..ROWS)
        .map(|row| row as f64 / ROWS as f64)
        .collect::<Vec<_>>();
    let candidate_edges =
        LearnedGraphAtom::knn_candidate_edges(embeddings.view()).expect("continuous kNN graph");
    let n_eff = rows.len() as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let precisions = vec![1.0; candidate_edges.len()];
    let deltas = vec![charge * 1.4; candidate_edges.len()];
    let candidate = graph_birth_candidate_for_structure_search(
        embeddings.view(),
        &rows,
        n_eff,
        &precisions,
        &deltas,
    )
    .expect("continuous graph birth candidate");

    assert!(
        matches!(
            candidate.selection.occupancy,
            OccupancyLaw::Uniform | OccupancyLaw::Continuous
        ),
        "dense phase control must remain continuous/uniform, got {:?}",
        candidate.selection.occupancy
    );
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

fn circular_abs_error(a: f64, b: f64) -> f64 {
    let delta = (a - b + 0.5).rem_euclid(1.0) - 0.5;
    delta.abs()
}

fn closed_form_phase_se_from_coordinate_rs(sigma_hat: f64, amplitude: f64) -> (f64, bool) {
    let raw_se = if amplitude > 0.0 {
        sigma_hat / (TAU * amplitude)
    } else {
        f64::INFINITY
    };
    let uniform_ceiling = (1.0f64 / 12.0).sqrt();
    if raw_se >= uniform_ceiling {
        (uniform_ceiling, true)
    } else {
        (raw_se, false)
    }
}

fn label_mean_se(standard_errors: &[f64]) -> f64 {
    assert!(
        !standard_errors.is_empty(),
        "cannot compute a label mean SE from an empty label bucket"
    );
    let variance_sum = standard_errors.iter().map(|se| se * se).sum::<f64>();
    variance_sum.sqrt() / standard_errors.len() as f64
}

fn assert_day_phase_order_and_closed_form_se(fixture: &CalendarFixture) {
    let fit = block_fit(fixture);
    let report = block_firing_coordinates(&fit, DAY_AXIS)
        .unwrap_or_else(|err| panic!("day coordinate readout failed: {err}"));

    let mut phases_by_day = vec![Vec::<f64>::new(); DAY_COUNT];
    let mut phase_se_by_day = vec![Vec::<f64>::new(); DAY_COUNT];
    for firing in &report.firings {
        let day = fixture.labels[firing.row][DAY_AXIS];
        phases_by_day[day].push(firing.t);
        phase_se_by_day[day].push(firing.t_se);
        let (expected_se, expected_clamped) =
            closed_form_phase_se_from_coordinate_rs(report.sigma_hat, firing.amplitude);
        assert!(
            (firing.t_se - expected_se).abs() <= 1.0e-12 * expected_se.max(1.0),
            "closed-form day phase SE mismatch on row {}: got {}, expected {}",
            firing.row,
            firing.t_se,
            expected_se
        );
        assert_eq!(
            firing.t_se_clamped,
            expected_clamped,
            "closed-form day phase SE clamp flag mismatch on row {}",
            firing.row
        );
    }

    let phase_by_day = phases_by_day
        .iter()
        .map(|phases| {
            assert!(!phases.is_empty(), "missing firings for a day label");
            circular_mean(phases)
        })
        .collect::<Vec<_>>();
    let se_by_day = phase_se_by_day
        .iter()
        .map(|standard_errors| label_mean_se(standard_errors))
        .collect::<Vec<_>>();

    let forward_error = calibrated_day_order_error(&phase_by_day, &se_by_day, 1);
    let reverse_error = calibrated_day_order_error(&phase_by_day, &se_by_day, -1);
    let (orientation, calibrated) = if forward_error <= reverse_error {
        (1, calibrate_day_phases(&phase_by_day, 1))
    } else {
        (-1, calibrate_day_phases(&phase_by_day, -1))
    };
    for day in 0..DAY_COUNT {
        let recovered = calibrated[day];
        let expected = day as f64 / DAY_COUNT as f64;
        let error = circular_abs_error(recovered, expected);
        let se_bound = DAY_ORDER_SE_MULTIPLIER * se_by_day[day].max(1.0e-12);
        assert!(
            error <= se_bound,
            "day phase order mismatch for label {day}: got {recovered}, expected {expected}, \
             circular error {error}, closed-form mean SE bound {se_bound}, orientation {orientation}"
        );
    }
}

fn calibrate_day_phases(phase_by_day: &[f64], orientation: i32) -> Vec<f64> {
    let anchor = phase_by_day[0];
    phase_by_day
        .iter()
        .map(|phase| {
            if orientation >= 0 {
                circular_forward_delta(anchor, *phase)
            } else {
                circular_forward_delta(*phase, anchor)
            }
        })
        .collect()
}

fn calibrated_day_order_error(phase_by_day: &[f64], se_by_day: &[f64], orientation: i32) -> f64 {
    let calibrated = calibrate_day_phases(phase_by_day, orientation);
    calibrated
        .iter()
        .enumerate()
        .map(|(day, recovered)| {
            let expected = day as f64 / DAY_COUNT as f64;
            circular_abs_error(*recovered, expected) / se_by_day[day].max(1.0e-12)
        })
        .fold(0.0, f64::max)
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
        report.n_firings,
        fixture.x.nrows(),
        "{} cluster must fire on every calendar row",
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

fn engels_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join("engels_gpt2_calendar_sae_indices.json")
}

fn engels_activation_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join("engels_gpt2_calendar_sae_activations.json")
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
                panic!("{name} in {} has a non-integer feature index", path.display())
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

fn axis_for_cluster_name(name: &str) -> usize {
    AXIS_NAMES
        .iter()
        .position(|axis_name| *axis_name == name)
        .unwrap_or_else(|| panic!("unknown Engels GPT-2 SAE calendar cluster {name}"))
}

fn assert_ground_truth_cluster(clusters: &[GroundTruthCluster], axis: usize) {
    let name = AXIS_NAMES[axis];
    let cluster = clusters
        .iter()
        .find(|cluster| cluster.name == name)
        .unwrap_or_else(|| panic!("missing Engels GPT-2 SAE {name} cluster"));
    assert_eq!(
        cluster.label_count,
        LABEL_COUNTS[axis],
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

fn load_optional_engels_real_fixture(clusters: &[GroundTruthCluster]) -> Option<CalendarFixture> {
    let path = engels_activation_fixture_path();
    if !path.exists() {
        println!(
            "[calendar_ground_truth] SKIP real GPT-2 activation path: {} not present; \
             planted-data path still ran",
            path.display()
        );
        return None;
    }
    Some(load_engels_real_fixture(&path, clusters))
}

fn load_engels_real_fixture(path: &PathBuf, clusters: &[GroundTruthCluster]) -> CalendarFixture {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let value: Value = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("parse {} as JSON: {err}", path.display()));
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{} is missing string model", path.display()));
    assert_eq!(model, "gpt-2", "{} must be a GPT-2 activation fixture", path.display());
    let sae_layer = value
        .get("sae_layer")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{} is missing integer sae_layer", path.display()));
    assert_eq!(
        sae_layer,
        7,
        "{} must carry the Engels GPT-2 layer-7 SAE activations",
        path.display()
    );
    let n_rows = value
        .get("n_rows")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{} is missing integer n_rows", path.display()))
        as usize;
    assert!(
        n_rows >= 64,
        "{} must contain at least 64 rows for chart evidence, got {n_rows}",
        path.display()
    );
    let cluster_values = value
        .get("clusters")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{} must contain a clusters array", path.display()));
    let mut real_clusters = Vec::with_capacity(AXES);
    for axis in 0..AXES {
        let name = AXIS_NAMES[axis];
        let cluster_value = cluster_values
            .iter()
            .find(|cluster| cluster.get("name").and_then(Value::as_str) == Some(name))
            .unwrap_or_else(|| panic!("{} is missing real activation cluster {name}", path.display()));
        let ground_truth = clusters
            .iter()
            .find(|cluster| cluster.name == name)
            .unwrap_or_else(|| panic!("missing committed ground-truth cluster {name}"));
        let real_cluster = parse_real_cluster_data(cluster_value, path, n_rows);
        assert_eq!(
            real_cluster.axis,
            axis,
            "{} real activation cluster axis mismatch",
            AXIS_NAMES[axis]
        );
        assert_eq!(
            real_cluster.feature_indices,
            ground_truth.feature_indices,
            "{} real activation cluster feature indices do not match the committed Engels fixture",
            AXIS_NAMES[axis]
        );
        real_clusters.push(real_cluster);
    }
    calendar_fixture_from_real_clusters(n_rows, real_clusters)
}

fn parse_real_cluster_data(cluster: &Value, path: &PathBuf, n_rows: usize) -> RealClusterData {
    let name = cluster
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{} real cluster is missing a name", path.display()));
    let axis = axis_for_cluster_name(name);
    let feature_indices = parse_u64_array(cluster, "feature_indices", path, name);
    let label_ids = parse_usize_array(cluster, "label_ids", path, name);
    let circle_codes = parse_circle_codes(cluster, path, name);
    let pc1_amplitude = parse_f64_array(cluster, "pc1_amplitude", path, name);
    assert_eq!(
        label_ids.len(),
        n_rows,
        "{name} label_ids length must equal n_rows in {}",
        path.display()
    );
    assert_eq!(
        circle_codes.len(),
        n_rows,
        "{name} circle_codes length must equal n_rows in {}",
        path.display()
    );
    assert_eq!(
        pc1_amplitude.len(),
        n_rows,
        "{name} pc1_amplitude length must equal n_rows in {}",
        path.display()
    );
    RealClusterData {
        axis,
        feature_indices,
        label_ids,
        circle_codes,
        pc1_amplitude,
    }
}

fn parse_u64_array(cluster: &Value, field: &str, path: &PathBuf, name: &str) -> Vec<u64> {
    cluster
        .get(field)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{name} in {} is missing {field}", path.display()))
        .iter()
        .map(|entry| {
            entry.as_u64().unwrap_or_else(|| {
                panic!("{name} in {} has a non-integer {field} entry", path.display())
            })
        })
        .collect()
}

fn parse_usize_array(cluster: &Value, field: &str, path: &PathBuf, name: &str) -> Vec<usize> {
    parse_u64_array(cluster, field, path, name)
        .into_iter()
        .map(|value| value as usize)
        .collect()
}

fn parse_f64_array(cluster: &Value, field: &str, path: &PathBuf, name: &str) -> Vec<f64> {
    cluster
        .get(field)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{name} in {} is missing {field}", path.display()))
        .iter()
        .map(|entry| {
            entry.as_f64().unwrap_or_else(|| {
                panic!("{name} in {} has a non-float {field} entry", path.display())
            })
        })
        .collect()
}

fn parse_circle_codes(cluster: &Value, path: &PathBuf, name: &str) -> Vec<[f64; 2]> {
    cluster
        .get("circle_codes")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{name} in {} is missing circle_codes", path.display()))
        .iter()
        .enumerate()
        .map(|(row, entry)| {
            let pair = entry.as_array().unwrap_or_else(|| {
                panic!("{name} row {row} in {} has a non-array circle code", path.display())
            });
            assert_eq!(
                pair.len(),
                2,
                "{name} row {row} in {} must have a 2D circle code",
                path.display()
            );
            let x0 = pair[0].as_f64().unwrap_or_else(|| {
                panic!("{name} row {row} in {} has a non-float first circle coordinate", path.display())
            });
            let x1 = pair[1].as_f64().unwrap_or_else(|| {
                panic!("{name} row {row} in {} has a non-float second circle coordinate", path.display())
            });
            [x0, x1]
        })
        .collect()
}

#[test]
fn planted_calendar_block_chart_promotion_recovers_circle_atoms() {
    let fixture = planted_calendar_fixture();
    assert_calendar_circles_promoted(&fixture);
}

#[test]
fn planted_calendar_structure_search_selects_graph_atoms() {
    let fixture = planted_calendar_fixture();
    assert_calendar_graph_atoms_selected(&fixture);
    assert_calendar_continuous_control_is_not_atomic();
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
fn engels_gpt2_sae_indices_are_the_real_calendar_ground_truth() {
    let clusters = load_engels_clusters();
    assert_engels_ground_truth_clusters(&clusters);
}

#[test]
fn engels_gpt2_real_sae_calendar_contract_when_activations_present() {
    let clusters = load_engels_clusters();
    assert_engels_ground_truth_clusters(&clusters);
    let Some(fixture) = load_optional_engels_real_fixture(&clusters) else {
        return;
    };
    assert_calendar_circles_promoted(&fixture);
    assert_calendar_graph_atoms_selected(&fixture);
    assert_day_phase_order_and_closed_form_se(&fixture);
    assert_pc1_intensity_equals_circle_radius(&fixture);
}

//! Run mpd-induction's S1 stage receipt for every linear use against torch's executed stages (#2951).
//!
//! `mpd_induction_registry --run EXECUTE_DIR --settings SETTINGS_JSON --out REPORT_JSON`
//!
//! This is the example half of the direct-lane receipt runner. `EXECUTE_DIR` is what
//! `bench/mpd_induction_2951.py execute` writes:
//! * `execute.json`, naming the registry export it read (the harvest), the declared settings and the
//!   executor's dtype, device and TF32 flag;
//! * the edit's two factors;
//! * one `<f8` `.npy` per stage, holding every setting's rows in setting order.
//!
//! The registry export holds `registry.json`, one little-endian `<f4` `.npy` per stored tensor in its
//! trained dtype, and a `<f8` `.npy` of the native logits.
//!
//! The example:
//! * refuses a settings file other than the one `execute.json` declares, and exports with different
//!   checkpoints or token rows;
//! * widens every tensor to binary64 and registers it, with its aliases and every discovered use
//!   site, through `support/induction_export.rs`;
//! * checks that the export's edit factors are a layer-0 head's output block of the registered
//!   `W_O.0` (negated) and its unit columns, and builds each setting's [`ParameterEditRecord`] at
//!   `W_O.0#0`;
//! * wraps each (setting, sequence) of torch's executed rows as an [`ExecutedExperiment`] and refuses
//!   any block whose shape the registry does not declare;
//! * runs the stage receipt of every linear use the export covers:
//!   * `W_O.0#0` and `W_O.1#0`, from the mixed head outputs to the write;
//!   * `W_U#0`, from the last residual to the logits.
//!
//!   Each receipt runs the native kernel on torch's input rows: `apply_anchored_linear` on the rows
//!   an edit reaches, `native_linear` on the rest. [`compare_stage`] decides agreement against the
//!   derived bands;
//! * runs each layer's attention stages on torch's residual rows `resid_pre.l`. The source adds
//!   learned absolute positions, so no plane is rotated:
//!   * `scores.l`: `x W_Qᵀ`, `x W_Kᵀ` and `x W_Vᵀ` run through `native_linear`, each with its
//!     `affine_stage_band` as the rows' radius, then [`RotaryCausalAttention::attend_projected`].
//!     [`compare_stage`] decides it over the causal entries, with the score radius on both sides;
//!   * `pattern.l`: [`RotaryCausalAttention::weights_at_scores`] at torch's scores, measured against
//!     torch's pattern. Torch's `exp` and the native one share no derivation, so no band covers
//!     their difference (#2951 correction C2), as in `mlp_block_receipt`;
//!   * `mixed.l`: [`RotaryCausalAttention::mix_at_weights`] at torch's pattern, taken as exact,
//!     over the value rows and their radius, so the verdict is conditional on torch's pattern;
//!   * `resid_post.l`: torch's `resid_pre.l + write.l`, one IEEE addition, must equal it bit for bit;
//!   * the whole layer: [`NativeAttentionLayer::execute`] on `resid_pre.l` under the setting's reads
//!     (layer 0's output read carries the edit), measured against `resid_post.l`.
//!
//!   The attention radii are `attention`'s: first order in the input radii.
//!
//!   Positive controls, per (setting, layer):
//!   * each certified stage compared against torch's rows of the next sequence must be refuted at
//!     every sequence;
//!   * under an edited setting, layer 0 run with its stored output read must sit further from torch's
//!     edited `resid_post.0` than the edited layer does.
//!
//! The tallies go to `REPORT_JSON`. A certified refutation, a residual that is not bit-identical,
//! or a failed control exits with an error after the report is written.

use gam_sae::inference::intervention_shard::ExperimentUnit;
use gam_sae::parameter_decomposition::apply::{
    FactorView, FactoredEdit, apply_anchored_linear, native_linear,
};
use gam_sae::parameter_decomposition::attention::{
    AttentionGeometry, ProjectedRows, RotaryCausalAttention, RotaryEmbedding, RotaryPairing,
};
use gam_sae::parameter_decomposition::block::{
    AttentionLayerReads, AttentionProjection, NativeAttentionLayer, ProjectionRead, ScopedEdit,
};
use gam_sae::parameter_decomposition::lift::{
    ExecutedExperiment, ForwardRoundoff, LiftError, ParameterExperiment, ParameterReadout, TensorId,
    UseSiteId,
};
use gam_sae::parameter_decomposition::occurrence::{EditScope, ParameterEditRecord, PositionScope};
use gam_sae::parameter_decomposition::receipts::{
    ExternalExecution, ReceiptRefusal, StageAgreement, affine_stage_band, compare_stage,
    factored_edit_stage_band,
};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, s};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[path = "support/npy_header.rs"]
mod npy_header;
#[path = "support/induction_export.rs"]
mod induction_export;
use induction_export::{ExportFile, float64_array, load_export};

const USAGE: &str = "usage: mpd_induction_registry --run EXECUTE_DIR --settings SETTINGS_JSON --out REPORT_JSON";

/// The fields of `execute.json` this example reads.
#[derive(Deserialize)]
struct Execute {
    stage: String,
    harvest: String,
    declared: serde_json::Value,
    checkpoint: u64,
    sequences: usize,
    tokens: Vec<Vec<i64>>,
    edit: EditDeclaration,
    settings: Vec<Setting>,
    rows_per_setting: BTreeMap<String, usize>,
    external_dtype: String,
    external_device: String,
    tf32_matmul: bool,
    files: BTreeMap<String, ExportFile>,
}

#[derive(Deserialize)]
struct EditDeclaration {
    tensor_id: String,
    ordinal: usize,
    head: usize,
    left: String,
    right: String,
}

#[derive(Deserialize)]
struct Setting {
    name: String,
    positions: Option<Vec<usize>>,
    substituted: Vec<String>,
}

/// One certified stage's agreement counts over the export's sequences, and its worst entry.
#[derive(Serialize)]
struct StageReport {
    agrees: usize,
    refutes: usize,
    unresolved: usize,
    worst_sequence: usize,
    worst_witness: (usize, usize),
    worst_discrepancy: f64,
    worst_band: f64,
    worst_ratio: f64,
}

/// One (setting, use) of the linear receipts.
#[derive(Serialize)]
struct UseReport {
    setting: String,
    use_site: String,
    sequences: usize,
    #[serde(flatten)]
    stage: StageReport,
}

/// The largest `|external − native|` entry of one measured stage over the export's sequences.
#[derive(Serialize)]
struct MeasuredReport {
    largest: f64,
    sequence: usize,
    witness: (usize, usize),
    /// The native evaluation's own radius at the witness, where it has one. It is reported beside
    /// the measured discrepancy and never folded into a band (#2951 correction C2).
    native_radius: Option<f64>,
}

/// One (setting, layer) of the attention receipts.
#[derive(Serialize)]
struct AttentionReport {
    setting: String,
    layer: usize,
    sequences: usize,
    scores: StageReport,
    pattern: MeasuredReport,
    mixed: StageReport,
    /// Entries where torch's `resid_pre + write` is not bit-identical to its `resid_post`.
    residual_mismatches: usize,
    layer_output: MeasuredReport,
    /// Positive controls: each certified stage against torch's rows of the next sequence must be refuted.
    shifted_scores_refuted: usize,
    shifted_mixed_refuted: usize,
    /// Under an edited setting at layer 0, the layer with its stored output read, against torch's
    /// edited `resid_post`: the edit must move the output further than the edited layer's own discrepancy.
    unedited_output: Option<MeasuredReport>,
}

#[derive(Serialize)]
struct Report {
    checkpoint: u64,
    teacher_fingerprint: String,
    external_dtype: String,
    external_device: String,
    tf32_matmul: bool,
    receipts: Vec<UseReport>,
    attention: Vec<AttentionReport>,
}

/// The value following `name` on the command line.
fn flag<'a>(args: &'a [String], name: &str) -> Result<&'a str, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
        .ok_or_else(|| format!("missing {name}; {USAGE}"))
}

/// The native stage of one linear use on the external input rows, and its receipt.
///
/// Rows the edit reaches run `x (W + L diag(s) Rᵀ)ᵀ` through `apply_anchored_linear`, under the
/// factored-edit band; every other row runs `x Wᵀ` through `native_linear`, under the affine band.
fn linear_receipt(
    execution: ExternalExecution<'_>,
    weight: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
    external: ArrayView2<'_, f64>,
    edit: Option<(ArrayView2<'_, f64>, ArrayView1<'_, f64>, ArrayView2<'_, f64>, &PositionScope)>,
) -> Result<StageAgreement, String> {
    let native_rows = native_linear(weight, input).map_err(|error| LiftError::from(error).to_string())?;
    let mut native = (*native_rows).to_owned();
    let mut band = affine_stage_band(weight, None, input)
        .map_err(|mismatch| ReceiptRefusal::from(mismatch).to_string())?;
    if let Some((left, coefficients, right, scope)) = edit {
        let reached: Vec<usize> = (0..input.nrows()).filter(|&row| scope.reaches(row)).collect();
        if !reached.is_empty() {
            let reached_input = input.select(Axis(0), &reached);
            let factors = FactorView::new(left, right).map_err(|error| LiftError::from(error).to_string())?;
            let edited = apply_anchored_linear(weight, 1.0, factors, coefficients, reached_input.view())
                .map_err(|error| LiftError::from(error).to_string())?;
            let edited_band =
                factored_edit_stage_band(weight, left, coefficients, right, None, reached_input.view())
                    .map_err(|mismatch| ReceiptRefusal::from(mismatch).to_string())?;
            for (index, &row) in reached.iter().enumerate() {
                native.row_mut(row).assign(&edited.row(index));
                band.row_mut(row).assign(&edited_band.row(index));
            }
        }
    }
    compare_stage(execution, external, native.view(), band.view(), band.view())
        .map_err(|refusal| refusal.to_string())
}

/// `x Wᵀ` through `native_linear`, with its `affine_stage_band` as the rows' forward-error radius.
fn projected_rows(
    weight: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let rows = native_linear(weight, input).map_err(|error| LiftError::from(error).to_string())?;
    let radius = affine_stage_band(weight, None, input)
        .map_err(|mismatch| ReceiptRefusal::from(mismatch).to_string())?;
    Ok(((*rows).to_owned(), radius))
}

/// `heads × tokens × tokens` rows as `(heads · tokens) × tokens`, with every entry past its query
/// (`s > t`, which no stage reads) set to zero, so a comparison sees only the causal entries.
fn causal_rows(values: ArrayView3<'_, f64>) -> Array2<f64> {
    let (heads, tokens, keys) = values.dim();
    Array2::from_shape_fn((heads * tokens, keys), |(row, key)| {
        let (head, query) = (row / tokens, row % tokens);
        if key > query { 0.0 } else { values[[head, query, key]] }
    })
}

/// Agreement counts of one certified stage over the export's sequences.
#[derive(Default)]
struct Tally {
    agrees: usize,
    refutes: usize,
    unresolved: usize,
    worst: Option<(usize, StageAgreement)>,
}

impl Tally {
    fn add(&mut self, sequence: usize, agreement: StageAgreement) {
        if agreement.agrees {
            self.agrees += 1;
        } else if agreement.refutes {
            self.refutes += 1;
        } else {
            self.unresolved += 1;
        }
        if self.worst.is_none_or(|(_, worst)| agreement.ratio > worst.ratio) {
            self.worst = Some((sequence, agreement));
        }
    }

    fn report(&self, what: &str) -> Result<StageReport, String> {
        let (worst_sequence, worst) = self
            .worst
            .ok_or_else(|| format!("{what}: the export holds no sequences"))?;
        Ok(StageReport {
            agrees: self.agrees,
            refutes: self.refutes,
            unresolved: self.unresolved,
            worst_sequence,
            worst_witness: worst.witness,
            worst_discrepancy: worst.discrepancy,
            worst_band: worst.band,
            worst_ratio: worst.ratio,
        })
    }
}

/// The largest measured discrepancy of one stage over the export's sequences.
#[derive(Default)]
struct Measured {
    worst: Option<MeasuredReport>,
}

impl Measured {
    /// One sequence's stage, with the native evaluation's own radius where it has one.
    fn add(
        &mut self,
        sequence: usize,
        external: ArrayView2<'_, f64>,
        native: ArrayView2<'_, f64>,
        radius: Option<ArrayView2<'_, f64>>,
    ) -> Result<(), String> {
        if external.dim() != native.dim() {
            return Err(format!(
                "sequence {sequence}: external {:?} and native {:?} rows differ in shape",
                external.dim(),
                native.dim()
            ));
        }
        for ((row, column), &value) in native.indexed_iter() {
            let difference = (external[[row, column]] - value).abs();
            if !difference.is_finite() {
                return Err(format!("sequence {sequence}: a non-finite entry at ({row}, {column})"));
            }
            if self.worst.as_ref().is_none_or(|worst| difference > worst.largest) {
                self.worst = Some(MeasuredReport {
                    largest: difference,
                    sequence,
                    witness: (row, column),
                    native_radius: radius.map(|radius| radius[[row, column]]),
                });
            }
        }
        Ok(())
    }

    fn report(self, what: &str) -> Result<MeasuredReport, String> {
        self.worst
            .ok_or_else(|| format!("{what}: the export holds no sequences"))
    }
}

/// One (setting, layer)'s attention-stage counts.
#[derive(Default)]
struct AttentionTally {
    scores: Tally,
    pattern: Measured,
    mixed: Tally,
    residual_mismatches: usize,
    layer_output: Measured,
    shifted_scores: Tally,
    shifted_mixed: Tally,
    unedited_output: Option<Measured>,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(USAGE.to_string());
    }
    let execute_dir = PathBuf::from(flag(&args, "--run")?);
    let settings_path = PathBuf::from(flag(&args, "--settings")?);
    let report_path = PathBuf::from(flag(&args, "--out")?);
    let text = std::fs::read_to_string(execute_dir.join("execute.json"))
        .map_err(|error| format!("read execute.json in {}: {error}", execute_dir.display()))?;
    let execute: Execute =
        serde_json::from_str(&text).map_err(|error| format!("execute.json: {error}"))?;
    let text = std::fs::read_to_string(&settings_path)
        .map_err(|error| format!("read {}: {error}", settings_path.display()))?;
    let declared: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", settings_path.display()))?;
    if declared != execute.declared {
        return Err(format!(
            "{} is not the settings execute.json declares",
            settings_path.display()
        ));
    }
    if execute.stage != "execute" {
        return Err(format!("stage {:?}; expected execute", execute.stage));
    }
    let registry_dir = PathBuf::from(&execute.harvest);
    let loaded = load_export(&registry_dir)?;
    let (export, registry) = (&loaded.export, &loaded.registry);
    let config = &export.config;
    if export.checkpoint != execute.checkpoint
        || export.sequences != execute.sequences
        || export.tokens != execute.tokens
    {
        return Err("the registry and execute exports hold different checkpoints or token rows".to_string());
    }
    let fingerprint = registry.teacher_fingerprint();
    let native_logits = float64_array(
        &export.files,
        &registry_dir,
        "native_logits",
        (export.sequences * config.seq_len, config.vocab),
    )?;
    println!(
        "[load] checkpoint={} tensors={} use_sites={} teacher_fingerprint={:#018x} native_logits={}x{}",
        export.checkpoint,
        export.tensors.len(),
        export.use_sites.len(),
        fingerprint.0,
        native_logits.nrows(),
        native_logits.ncols()
    );

    let weight = |id: &str| loaded.weight(id);
    let output = weight("W_O.0")?;
    let later_output = weight("W_O.1")?;
    let unembedding = weight("W_U")?;
    let width = config.n_heads * config.d_head;
    if execute.edit.tensor_id != "W_O.0" || output.dim() != (config.d_model, width) || execute.edit.head >= config.n_heads {
        return Err(format!(
            "execute.json edits {} head {}; expected W_O.0 of shape ({}, {width}) and a head below {}",
            execute.edit.tensor_id, execute.edit.head, config.d_model, config.n_heads
        ));
    }
    // The export's factors must be the registered head block, negated, and its unit columns.
    let block = execute.edit.head * config.d_head..(execute.edit.head + 1) * config.d_head;
    let left = float64_array(&execute.files, &execute_dir, &execute.edit.left, (config.d_model, config.d_head))?;
    let right = float64_array(&execute.files, &execute_dir, &execute.edit.right, (width, config.d_head))?;
    let expected_left = output.slice(s![.., block.clone()]).mapv(|value| -value);
    let expected_right = Array2::from_shape_fn((width, config.d_head), |(column, term)| {
        if column == block.start + term { 1.0 } else { 0.0 }
    });
    if left != expected_left || right != expected_right {
        return Err("the exported edit factors are not the registered W_O.0 head block and its unit columns".to_string());
    }
    let coefficients = Array1::<f64>::ones(config.d_head);
    let site = UseSiteId::read(&TensorId(execute.edit.tensor_id.clone()), execute.edit.ordinal);
    let later_site = UseSiteId::read(&TensorId("W_O.1".to_string()), 0);
    let unembedding_site = UseSiteId::read(&TensorId("W_U".to_string()), 0);
    let execution = ExternalExecution {
        dtype: &execute.external_dtype,
        device: &execute.external_device,
        tf32_matmul: execute.tf32_matmul,
    };
    println!(
        "[setting] edit={} head={} executor dtype={} device={} tf32_matmul={} settings={}",
        site.0,
        execute.edit.head,
        execute.external_dtype,
        execute.external_device,
        execute.tf32_matmul,
        execute.settings.len()
    );

    // Torch divides the scores by math.sqrt(d_head) and the native core multiplies them by its
    // reciprocal. Both round nothing, so they are one program, only when √d_head is a power of two.
    let root = (config.d_head as f64).sqrt();
    let whole = root as u64;
    if whole as f64 != root || !whole.is_power_of_two() {
        return Err(format!(
            "d_head {}: √d_head is not a power of two, so torch's division and the native multiplication round differently",
            config.d_head
        ));
    }
    let score_scale = 1.0 / root;
    let geometry = AttentionGeometry {
        model_dim: config.d_model,
        n_heads: config.n_heads,
        n_kv_heads: config.n_heads,
        head_dim: config.d_head,
    };
    // Learned absolute positions: the source rotates no plane.
    let no_rotary = || RotaryEmbedding {
        pairing: RotaryPairing::HalfSplit,
        inverse_frequencies: Vec::new(),
        attention_scaling: 1.0,
    };
    let core = RotaryCausalAttention::new(geometry, no_rotary(), score_scale).map_err(|error| error.to_string())?;
    let mut layers = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        let [query, key, value, layer_output] = ["W_Q", "W_K", "W_V", "W_O"].map(|name| weight(&format!("{name}.{layer}")));
        layers.push(
            NativeAttentionLayer::new(geometry, no_rotary(), score_scale, query?, key?, value?, layer_output?)
                .map_err(|error| error.to_string())?,
        );
    }
    let positions: Vec<i64> = (0..config.seq_len as i64).collect();

    let stage_ids = ["mixed.0", "write.0", "mixed.1", "write.1", "resid_post.1", "logits"];
    let mut stage_columns = vec![("logits".to_string(), 1, config.vocab)];
    for layer in 0..config.n_layers {
        for (stage, per_token, columns) in [
            ("resid_pre", 1, config.d_model),
            ("scores", config.n_heads, config.seq_len),
            ("pattern", config.n_heads, config.seq_len),
            ("mixed", 1, width),
            ("write", 1, config.d_model),
            ("resid_post", 1, config.d_model),
        ] {
            stage_columns.push((format!("{stage}.{layer}"), per_token, columns));
        }
    }
    let mut stages = BTreeMap::new();
    for (id, per_token, columns) in stage_columns {
        let per_setting = *execute
            .rows_per_setting
            .get(&id)
            .ok_or_else(|| format!("execute.json has no rows for {id}"))?;
        if per_setting != execute.sequences * per_token * config.seq_len {
            return Err(format!("{id}: {per_setting} rows per setting, expected sequences x {per_token} x T"));
        }
        let rows = float64_array(&execute.files, &execute_dir, &id, (per_setting * execute.settings.len(), columns))?;
        stages.insert(id, rows);
    }
    // One (setting, sequence) of a stage: its T rows, in the setting and sequence order execute.json declares.
    let rows_of = |id: &str, setting: usize, sequence: usize| -> Array2<f64> {
        let start = (setting * execute.sequences + sequence) * config.seq_len;
        stages[id].slice(s![start..start + config.seq_len, ..]).to_owned()
    };
    // One (setting, sequence) of a per-head stage: heads x query x key, from rows over head and query.
    let heads_of = |id: &str, setting: usize, sequence: usize| -> Array3<f64> {
        let (heads, tokens) = (config.n_heads, config.seq_len);
        let start = (setting * execute.sequences + sequence) * heads * tokens;
        let rows = stages[id].slice(s![start..start + heads * tokens, ..]);
        Array3::from_shape_fn((heads, tokens, tokens), |(head, query, key)| rows[[head * tokens + query, key]])
    };

    let mut reports = Vec::new();
    let mut attention_reports = Vec::new();
    let mut control_failures = 0usize;
    for (index, setting) in execute.settings.iter().enumerate() {
        let scope = match &setting.positions {
            Some(positions) => Some(PositionScope::declared(positions.clone()).map_err(|error| error.to_string())?),
            None if setting.name == "all_on" => None,
            None => Some(PositionScope::every()),
        };
        let expected_substituted = if scope.is_some() { vec![site.0.clone()] } else { Vec::new() };
        if setting.substituted != expected_substituted {
            return Err(format!(
                "setting {} substituted {:?}, expected {:?}",
                setting.name, setting.substituted, expected_substituted
            ));
        }
        let record = match &scope {
            Some(scope) => {
                scope.check_within(config.seq_len).map_err(|error| error.to_string())?;
                let delta = FactoredEdit::new(left.clone(), right.clone())
                    .map_err(|error| LiftError::from(error).to_string())?;
                Some(
                    ParameterEditRecord::new(registry, EditScope::UseSite(site.clone()), scope.clone(), delta)
                        .map_err(|error| error.to_string())?,
                )
            }
            None => None,
        };
        if let Some(record) = &record {
            if record.registry() != fingerprint {
                return Err(format!("setting {}: the record was checked against a different registry", setting.name));
            }
        }
        let mut tallies: [Tally; 3] = Default::default();
        let mut attention: Vec<AttentionTally> = (0..config.n_layers).map(|_| AttentionTally::default()).collect();
        for sequence in 0..execute.sequences {
            let experiment = ParameterExperiment {
                unit: ExperimentUnit {
                    group: 0,
                    sequence: sequence as i64,
                    length: config.seq_len,
                },
                edits: record.iter().cloned().collect(),
                readouts: vec![
                    ParameterReadout::UseSiteInput(site.clone()),
                    ParameterReadout::UseSiteOutput(site.clone()),
                    ParameterReadout::UseSiteInput(later_site.clone()),
                    ParameterReadout::UseSiteOutput(later_site.clone()),
                    ParameterReadout::UseSiteInput(unembedding_site.clone()),
                    ParameterReadout::Output {
                        positions: (0..config.seq_len).collect(),
                    },
                ],
            };
            let shapes = experiment
                .readout_shapes(registry, config.vocab)
                .map_err(|error| error.to_string())?;
            let executed = ExecutedExperiment {
                readouts: stage_ids
                    .iter()
                    .map(|id| rows_of(*id, index, sequence))
                    .collect(),
                roundoff: ForwardRoundoff::Unresolved,
            };
            executed.check(&shapes).map_err(|error| error.to_string())?;
            let blocks = &executed.readouts;
            tallies[0].add(
                sequence,
                linear_receipt(
                    execution,
                    output.view(),
                    blocks[0].view(),
                    blocks[1].view(),
                    scope.as_ref().map(|scope| (left.view(), coefficients.view(), right.view(), scope)),
                )?,
            );
            tallies[1].add(
                sequence,
                linear_receipt(execution, later_output.view(), blocks[2].view(), blocks[3].view(), None)?,
            );
            tallies[2].add(
                sequence,
                linear_receipt(execution, unembedding.view(), blocks[4].view(), blocks[5].view(), None)?,
            );

            for (layer, (native_layer, tally)) in layers.iter().zip(attention.iter_mut()).enumerate() {
                let residual = rows_of(&format!("resid_pre.{layer}"), index, sequence);
                let [queries, keys, values] = [
                    AttentionProjection::Query,
                    AttentionProjection::Key,
                    AttentionProjection::Value,
                ]
                .map(|projection| projected_rows(native_layer.weight(projection), residual.view()));
                let ((queries, query_radius), (keys, key_radius), (values, value_radius)) = (queries?, keys?, values?);
                let value_rows = ProjectedRows {
                    values: values.view(),
                    radius: value_radius.view(),
                };
                let native = core
                    .attend_projected(
                        ProjectedRows {
                            values: queries.view(),
                            radius: query_radius.view(),
                        },
                        ProjectedRows {
                            values: keys.view(),
                            radius: key_radius.view(),
                        },
                        value_rows,
                        &positions,
                    )
                    .map_err(|error| error.to_string())?;
                let external_scores = heads_of(&format!("scores.{layer}"), index, sequence);
                let score_band = causal_rows(native.score_radius.view());
                tally.scores.add(
                    sequence,
                    compare_stage(
                        execution,
                        causal_rows(external_scores.view()).view(),
                        causal_rows(native.scores.view()).view(),
                        score_band.view(),
                        score_band.view(),
                    )
                    .map_err(|refusal| refusal.to_string())?,
                );
                // Positive control: the same native scores against the next sequence's torch scores.
                let other = (sequence + 1) % execute.sequences;
                let shifted_scores = heads_of(&format!("scores.{layer}"), index, other);
                tally.shifted_scores.add(
                    sequence,
                    compare_stage(
                        execution,
                        causal_rows(shifted_scores.view()).view(),
                        causal_rows(native.scores.view()).view(),
                        score_band.view(),
                        score_band.view(),
                    )
                    .map_err(|refusal| refusal.to_string())?,
                );
                // The softmax and the value read at torch's own inputs, taken as exact.
                let exact = Array3::<f64>::zeros(external_scores.dim());
                let pattern = core
                    .weights_at_scores(external_scores.view(), exact.view())
                    .map_err(|error| error.to_string())?;
                let external_pattern = heads_of(&format!("pattern.{layer}"), index, sequence);
                tally.pattern.add(
                    sequence,
                    causal_rows(external_pattern.view()).view(),
                    causal_rows(pattern.weights.view()).view(),
                    Some(causal_rows(pattern.weight_radius.view()).view()),
                )?;
                let (mixed, mixed_radius) = core
                    .mix_at_weights(external_pattern.view(), exact.view(), value_rows)
                    .map_err(|error| error.to_string())?;
                let external_mixed = rows_of(&format!("mixed.{layer}"), index, sequence);
                tally.mixed.add(
                    sequence,
                    compare_stage(
                        execution,
                        external_mixed.view(),
                        mixed.view(),
                        mixed_radius.view(),
                        mixed_radius.view(),
                    )
                    .map_err(|refusal| refusal.to_string())?,
                );
                let shifted_mixed = rows_of(&format!("mixed.{layer}"), index, other);
                tally.shifted_mixed.add(
                    sequence,
                    compare_stage(
                        execution,
                        shifted_mixed.view(),
                        mixed.view(),
                        mixed_radius.view(),
                        mixed_radius.view(),
                    )
                    .map_err(|refusal| refusal.to_string())?,
                );
                let write = rows_of(&format!("write.{layer}"), index, sequence);
                let residual_out = rows_of(&format!("resid_post.{layer}"), index, sequence);
                tally.residual_mismatches += residual
                    .iter()
                    .zip(write.iter())
                    .zip(residual_out.iter())
                    .filter(|((before, written), after)| (**before + **written).to_bits() != after.to_bits())
                    .count();
                // The whole layer through mpd-block's program; layer 0's output read carries the edit.
                let factors = FactorView::new(left.view(), right.view())
                    .map_err(|error| LiftError::from(error).to_string())?;
                let reads = match &scope {
                    Some(scope) if layer == 0 => AttentionLayerReads {
                        output: ProjectionRead::Edited(ScopedEdit {
                            edit: factors,
                            positions: scope,
                        }),
                        ..AttentionLayerReads::native()
                    },
                    _ => AttentionLayerReads::native(),
                };
                let executed = native_layer
                    .execute(reads, ProjectedRows::exact(residual.view()), &positions)
                    .map_err(|error| error.to_string())?;
                tally
                    .layer_output
                    .add(sequence, residual_out.view(), executed.output.view(), None)?;
                if scope.is_some() && layer == 0 {
                    let unedited = native_layer
                        .execute(AttentionLayerReads::native(), ProjectedRows::exact(residual.view()), &positions)
                        .map_err(|error| error.to_string())?;
                    tally.unedited_output.get_or_insert_with(Measured::default).add(
                        sequence,
                        residual_out.view(),
                        unedited.output.view(),
                        None,
                    )?;
                }
            }
        }
        for (use_site, tally) in [&site, &later_site, &unembedding_site].iter().zip(tallies) {
            let stage = tally.report(&format!("setting {}", setting.name))?;
            println!(
                "[receipt] setting={} use={} sequences={} agrees={} refutes={} unresolved={} worst_ratio={:.3e} sequence={} witness={:?} discrepancy={:.3e} band={:.3e}",
                setting.name,
                use_site.0,
                execute.sequences,
                stage.agrees,
                stage.refutes,
                stage.unresolved,
                stage.worst_ratio,
                stage.worst_sequence,
                stage.worst_witness,
                stage.worst_discrepancy,
                stage.worst_band
            );
            reports.push(UseReport {
                setting: setting.name.clone(),
                use_site: use_site.0.clone(),
                sequences: execute.sequences,
                stage,
            });
        }
        for (layer, tally) in attention.into_iter().enumerate() {
            let what = format!("setting {} layer {layer}", setting.name);
            let scores = tally.scores.report(&what)?;
            let mixed = tally.mixed.report(&what)?;
            let pattern = tally.pattern.report(&what)?;
            let layer_output = tally.layer_output.report(&what)?;
            for (stage, report) in [("scores", &scores), ("mixed", &mixed)] {
                println!(
                    "[receipt] setting={} layer={layer} stage={stage} sequences={} agrees={} refutes={} unresolved={} worst_ratio={:.3e} sequence={} witness={:?} discrepancy={:.3e} band={:.3e}",
                    setting.name,
                    execute.sequences,
                    report.agrees,
                    report.refutes,
                    report.unresolved,
                    report.worst_ratio,
                    report.worst_sequence,
                    report.worst_witness,
                    report.worst_discrepancy,
                    report.worst_band
                );
            }
            println!(
                "[receipt] setting={} layer={layer} stage=pattern measured={:.3e} native_radius={:.3e} sequence={} witness={:?}; residual_mismatches={}; layer_output measured={:.3e} sequence={} witness={:?}",
                setting.name,
                pattern.largest,
                pattern.native_radius.unwrap_or(f64::NAN),
                pattern.sequence,
                pattern.witness,
                tally.residual_mismatches,
                layer_output.largest,
                layer_output.sequence,
                layer_output.witness
            );
            let shifted_scores_refuted = tally.shifted_scores.refutes;
            let shifted_mixed_refuted = tally.shifted_mixed.refutes;
            let unedited_output = tally
                .unedited_output
                .map(|measured| measured.report(&what))
                .transpose()?;
            println!(
                "[control] setting={} layer={layer} shifted_sequence scores_refuted={shifted_scores_refuted}/{} mixed_refuted={shifted_mixed_refuted}/{} unedited_output={}",
                setting.name,
                execute.sequences,
                execute.sequences,
                unedited_output.as_ref().map_or("none".to_string(), |measured| format!(
                    "{:.3e} sequence={} witness={:?}",
                    measured.largest, measured.sequence, measured.witness
                ))
            );
            // The edit has to move the layer's output further than the edited layer's own discrepancy.
            let edit_visible = unedited_output
                .as_ref()
                .is_none_or(|measured| measured.largest > layer_output.largest);
            if shifted_scores_refuted != execute.sequences
                || shifted_mixed_refuted != execute.sequences
                || !edit_visible
            {
                control_failures += 1;
            }
            attention_reports.push(AttentionReport {
                setting: setting.name.clone(),
                layer,
                sequences: execute.sequences,
                scores,
                pattern,
                mixed,
                residual_mismatches: tally.residual_mismatches,
                layer_output,
                shifted_scores_refuted,
                shifted_mixed_refuted,
                unedited_output,
            });
        }
    }
    let refuted: usize = reports.iter().map(|report| report.stage.refutes).sum::<usize>()
        + attention_reports
            .iter()
            .map(|report| report.scores.refutes + report.mixed.refutes)
            .sum::<usize>();
    let mismatched: usize = attention_reports.iter().map(|report| report.residual_mismatches).sum();
    let report = Report {
        checkpoint: export.checkpoint,
        teacher_fingerprint: format!("{:#018x}", fingerprint.0),
        external_dtype: execute.external_dtype.clone(),
        external_device: execute.external_device.clone(),
        tf32_matmul: execute.tf32_matmul,
        receipts: reports,
        attention: attention_reports,
    };
    let text = serde_json::to_string_pretty(&report).map_err(|error| format!("report: {error}"))?;
    std::fs::write(&report_path, text).map_err(|error| format!("write {}: {error}", report_path.display()))?;
    if refuted > 0 || mismatched > 0 || control_failures > 0 {
        return Err(format!(
            "{refuted} stage receipts certify a violation, {mismatched} residual entries are not torch's own resid_pre + write, and {control_failures} (setting, layer) controls failed"
        ));
    }
    Ok(())
}

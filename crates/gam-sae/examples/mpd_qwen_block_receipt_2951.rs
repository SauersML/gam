//! Executed-block receipt for Qwen3's SwiGLU MLP under parameter edits (#2951 A12).
//!
//! `bench/mpd_qwen_block_2951.py execute` runs the source `Qwen3MLP` in float64 under
//! declared literal-component edits on the leading rows of an `export_block.py`
//! harvest, and writes one float64 `.npy` per array id plus `manifest.json`. This
//! example reads those arrays, the harvested weights and the same float32 rows, and
//! calls [`swiglu_block_receipt`] once per setting and once per declared control.
//! Every native stage, band and comparison is the owner's; this file assembles
//! views, and writes what the owner returned.
//!
//! The norm stage is the layer's `post_attention_layernorm`, the source's `Qwen3RMSNorm`,
//! run by the driver on the harvested pre-norm rows. Its source normalizes in float32 inside
//! a float64 module, as the manifest records from the installed source, so the receipt
//! declares [`ExternalRmsNormProgram::Binary32Internal`] and [`rms_norm_stage`] takes that
//! program's band. That band's rsqrt step is two correctly rounded operations, so the receipt
//! also requires the driver's check that `torch.rsqrt` returned exactly `1 / torch.sqrt` at the
//! stage's float32 arguments. Its control declares the same stage `Binary64` and must be refuted: a
//! binary64 band cannot hold the float32 evaluation, so the declared program is load-bearing.
//!
//! The exit status is non-zero when a setting's certified stage or the norm stage does not
//! agree, when the norm stage's binary64 control is not refuted, when
//! a refuting control does not report `refutes` on every stage it names, when an
//! identity control's two settings executed different binary64 values at any stage,
//! or when a distinct control's two settings executed the same values at every stage.
//! Identity and distinct controls compare the two external executions with each
//! other, bit for bit, because the native and external sides of one receipt sum in
//! different orders: only two executions of one computation can agree exactly.
//!
//! ```text
//! cargo run --profile test -p gam-sae --example mpd_qwen_block_receipt_2951 -- \
//!     --run RUN_DIR --settings settings.json --out report.json
//! ```

use gam_sae::parameter_decomposition::apply::FactorView;
use gam_sae::parameter_decomposition::occurrence::PositionScope;
use gam_sae::parameter_decomposition::receipts::{
    EditedRead, ExternalExecution, ExternalRmsNormProgram, FactoredEditViews, MeasuredDiscrepancy,
    StageAgreement, SwigluBlockReceipt, SwigluBlockReceiptInputs, rms_norm_stage, swiglu_block_receipt,
};
use memmap2::Mmap;
use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "support/npy_header.rs"]
mod npy_header;
use npy_header::{NpyFloat, parse_npy_float_header, parse_npy_header};

const READS: [&str; 3] = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"];
const STAGES: [&str; 5] = ["gate", "activation", "up", "hidden", "output"];

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            println!("[mpd_qwen_block_receipt_2951] error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn flag(args: &[String], name: &str) -> Result<PathBuf, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| format!("missing {name}"))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))
}

fn mapped(path: &Path) -> Result<Mmap, String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    // SAFETY: the exported array is opened read-only and never written through this
    // mapping for the lifetime of the read.
    unsafe { Mmap::map(&file).map_err(|err| format!("mmap {}: {err}", path.display())) }
}

/// The shape and values of a float64 `.npy` of one or two axes.
fn float64(path: &Path) -> Result<(Vec<usize>, Vec<f64>), String> {
    let mmap = mapped(path)?;
    let header = parse_npy_float_header(&mmap, path)?;
    if !matches!(header.float, NpyFloat::F8) {
        return Err(format!("{} must hold <f8 values", path.display()));
    }
    let end = header
        .shape
        .iter()
        .try_fold(8_usize, |bytes, &extent| bytes.checked_mul(extent))
        .and_then(|bytes| bytes.checked_add(header.data_off))
        .ok_or_else(|| format!("{}: size overflows", path.display()))?;
    if end != mmap.len() {
        return Err(format!("{} holds {} bytes; its header needs {end}", path.display(), mmap.len()));
    }
    let values = mmap[header.data_off..end]
        .chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]])
        })
        .collect();
    Ok((header.shape, values))
}

fn matrix(path: &Path) -> Result<Array2<f64>, String> {
    let (shape, values) = float64(path)?;
    let [rows, cols] = shape[..] else {
        return Err(format!("{} must have two axes; it has {shape:?}", path.display()));
    };
    Array2::from_shape_vec((rows, cols), values).map_err(|err| format!("{}: {err}", path.display()))
}

fn vector(path: &Path) -> Result<Array1<f64>, String> {
    let (shape, values) = float64(path)?;
    if shape.len() != 1 {
        return Err(format!("{} must have one axis; it has {shape:?}", path.display()));
    }
    Ok(Array1::from(values))
}

/// The leading `rows` rows of a harvest's float32 `post_norm.npy`, widened exactly to
/// float64: the same values the torch driver consumed.
fn post_norm_rows(harvest: &Path, rows: usize) -> Result<Array2<f64>, String> {
    let path = harvest.join("post_norm.npy");
    let mmap = mapped(&path)?;
    let (available, width, elem, is_f4, data_off) = parse_npy_header(&mmap, &path)?;
    if !is_f4 {
        return Err(format!("{} must hold <f4 rows", path.display()));
    }
    if available < rows || data_off + rows * width * elem > mmap.len() {
        return Err(format!("{} holds fewer than {rows} rows", path.display()));
    }
    Ok(Array2::from_shape_fn((rows, width), |(row, column)| {
        let off = data_off + (row * width + column) * elem;
        f64::from(f32::from_le_bytes([mmap[off], mmap[off + 1], mmap[off + 2], mmap[off + 3]]))
    }))
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key:?}"))
}

fn indices(value: &Value, what: &str) -> Result<Vec<usize>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{what} must be a list"))?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| format!("{what} holds a non-index {entry}"))
        })
        .collect()
}

/// The rows an edit reaches: `null` is every row, a list is strictly increasing
/// declared positions of the one `(1, rows)` unit.
fn position_scope(value: Option<&Value>, what: &str) -> Result<PositionScope, String> {
    match value {
        None | Some(Value::Null) => Ok(PositionScope::every()),
        Some(positions) => PositionScope::declared(indices(positions, what)?)
            .map_err(|refusal| format!("{what} refused: {refusal}")),
    }
}

/// The native side of one read: its factor ids, coefficients and reached rows.
#[derive(Clone)]
struct NativeEdit {
    left: String,
    right: String,
    coefficients: Array1<f64>,
    rows: PositionScope,
}

/// One receipt: the native edits, and the setting whose executed arrays they meet.
struct Case {
    name: String,
    edits: BTreeMap<String, NativeEdit>,
    externals: String,
}

/// The arrays every receipt of one run reads.
struct Run {
    inputs: Array2<f64>,
    weights: BTreeMap<String, Array2<f64>>,
    matrices: BTreeMap<String, Array2<f64>>,
    vectors: BTreeMap<String, Array1<f64>>,
    dtype: String,
    device: String,
    tf32_matmul: bool,
}

fn setting_edits(setting: &Value, run: &Run) -> Result<BTreeMap<String, NativeEdit>, String> {
    let mut edits = BTreeMap::new();
    let declared_edits = setting
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("setting {setting} names no edits"))?;
    for declared in declared_edits {
        let coefficients_id = text(declared, "coefficients")?;
        let coefficients = run
            .vectors
            .get(coefficients_id)
            .ok_or_else(|| format!("unknown coefficients id {coefficients_id:?}"))?
            .clone();
        let edit = NativeEdit {
            left: text(declared, "left")?.to_string(),
            right: text(declared, "right")?.to_string(),
            coefficients,
            rows: position_scope(declared.get("positions"), "positions")?,
        };
        let tensor = text(declared, "tensor_id")?.to_string();
        if edits.insert(tensor.clone(), edit).is_some() {
            return Err(format!("a setting edits {tensor:?} twice"));
        }
    }
    Ok(edits)
}

fn edited_read<'a>(run: &'a Run, edits: &'a BTreeMap<String, NativeEdit>, tensor: &str) -> Result<EditedRead<'a>, String> {
    let weight = run
        .weights
        .get(tensor)
        .ok_or_else(|| format!("no weight {tensor:?}"))?;
    let edit = match edits.get(tensor) {
        None => None,
        Some(edit) => {
            let left = run
                .matrices
                .get(&edit.left)
                .ok_or_else(|| format!("unknown left id {:?}", edit.left))?;
            let right = run
                .matrices
                .get(&edit.right)
                .ok_or_else(|| format!("unknown right id {:?}", edit.right))?;
            let factors = FactorView::new(left.view(), right.view())
                .map_err(|refusal| format!("edit {:?}/{:?} of {tensor:?} refused: {refusal}", edit.left, edit.right))?;
            Some(FactoredEditViews {
                factors,
                coefficients: edit.coefficients.view(),
                rows: &edit.rows,
            })
        }
    };
    Ok(EditedRead {
        weight: weight.view(),
        edit,
    })
}

fn external<'a>(run: &'a Run, setting: &str, stage_name: &str) -> Result<&'a Array2<f64>, String> {
    run.matrices
        .get(&format!("{setting}.{stage_name}"))
        .ok_or_else(|| format!("no executed {stage_name} for setting {setting:?}"))
}

fn swiglu_receipt(run: &Run, case: &Case) -> Result<SwigluBlockReceipt, String> {
    swiglu_block_receipt(SwigluBlockReceiptInputs {
        inputs: run.inputs.view(),
        gate: edited_read(run, &case.edits, "gate_proj.weight")?,
        up: edited_read(run, &case.edits, "up_proj.weight")?,
        down: edited_read(run, &case.edits, "down_proj.weight")?,
        external_gate: external(run, &case.externals, "gate")?.view(),
        external_up: external(run, &case.externals, "up")?.view(),
        external_activation: external(run, &case.externals, "activation")?.view(),
        external_hidden: external(run, &case.externals, "hidden")?.view(),
        external_output: external(run, &case.externals, "output")?.view(),
        external_execution: ExternalExecution {
            dtype: &run.dtype,
            device: &run.device,
            tf32_matmul: run.tf32_matmul,
        },
    })
    .map_err(|refusal| format!("receipt {:?} refused: {refusal}", case.name))
}

fn stage(agreement: &StageAgreement) -> Value {
    json!({
        "agrees": agreement.agrees,
        "refutes": agreement.refutes,
        "witness": [agreement.witness.0, agreement.witness.1],
        "discrepancy": agreement.discrepancy,
        "band": agreement.band,
        "ratio": agreement.ratio,
    })
}

fn measured(discrepancy: &MeasuredDiscrepancy) -> Value {
    json!({
        "largest": discrepancy.largest,
        "witness": [discrepancy.witness.0, discrepancy.witness.1],
        "native_at_witness": discrepancy.native_at_witness,
        "status": "measured",
    })
}

fn certified(receipt: &SwigluBlockReceipt) -> [(&'static str, &StageAgreement); 4] {
    [
        ("gate", &receipt.gate),
        ("up", &receipt.up),
        ("hidden", &receipt.hidden),
        ("output", &receipt.output),
    ]
}

fn single(manifest: &Value, key: &str) -> Result<String, String> {
    let values = manifest
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("manifest names no {key}"))?;
    match &values[..] {
        [value] => value
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{key} is not a string")),
        several => Err(format!("the executed stages report several {key}: {several:?}")),
    }
}

fn load(run_dir: &Path, manifest: &Value) -> Result<Run, String> {
    let rows = manifest
        .get("rows")
        .and_then(Value::as_u64)
        .and_then(|rows| usize::try_from(rows).ok())
        .ok_or("manifest names no rows")?;
    let inputs = post_norm_rows(Path::new(text(manifest, "harvest")?), rows)?;
    let weight_paths = manifest.get("weights").ok_or("manifest names no weights")?;
    let mut weights = BTreeMap::new();
    for read in READS {
        weights.insert(read.to_string(), matrix(Path::new(text(weight_paths, read)?))?);
    }
    let files = manifest
        .get("files")
        .and_then(Value::as_object)
        .ok_or("manifest names no files")?;
    let mut matrices = BTreeMap::new();
    let mut vectors = BTreeMap::new();
    for (id, entry) in files {
        let path = Path::new(text(entry, "path")?);
        if id.starts_with("ones") {
            vectors.insert(id.clone(), vector(path)?);
        } else {
            matrices.insert(id.clone(), matrix(path)?);
        }
    }
    println!(
        "[load] run={} rows={rows} weights={} arrays={}",
        run_dir.display(),
        weights.len(),
        matrices.len() + vectors.len()
    );
    Ok(Run {
        inputs,
        weights,
        matrices,
        vectors,
        dtype: single(manifest, "external_dtype")?,
        device: single(manifest, "external_device")?,
        tf32_matmul: manifest
            .get("tf32_matmul")
            .and_then(Value::as_bool)
            .ok_or("manifest names no tf32_matmul")?,
    })
}

fn control_case(control: &Value, native: &Value, run: &Run) -> Result<Case, String> {
    let name = text(control, "name")?.to_string();
    let mut edits = setting_edits(native, run)?;
    if let Some(moved) = control.get("move_edit") {
        let edit = edits
            .remove(text(moved, "from")?)
            .ok_or_else(|| format!("control {name:?} moves an absent edit"))?;
        edits.insert(text(moved, "to")?.to_string(), edit);
    }
    if let Some(overrides) = control.get("rows_override").and_then(Value::as_object) {
        for (tensor, rows) in overrides {
            let edit = edits
                .get_mut(tensor)
                .ok_or_else(|| format!("control {name:?} rescopes an absent edit"))?;
            edit.rows = position_scope(Some(rows), "rows_override")?;
        }
    }
    if let Some(overrides) = control.get("coefficients_override").and_then(Value::as_object) {
        for (tensor, coefficients) in overrides {
            let edit = edits
                .get_mut(tensor)
                .ok_or_else(|| format!("control {name:?} overrides the coefficients of an absent edit"))?;
            let values = coefficients
                .as_array()
                .ok_or_else(|| format!("control {name:?} lists no coefficients"))?
                .iter()
                .map(|value| value.as_f64().ok_or_else(|| format!("control {name:?} holds a non-number coefficient")))
                .collect::<Result<Vec<f64>, String>>()?;
            edit.coefficients = Array1::from(values);
        }
    }
    Ok(Case {
        name,
        edits,
        externals: text(control, "externals_from")?.to_string(),
    })
}

/// Per stage, the entries at which the two settings a control names executed
/// different binary64 values.
fn executed_differences(control: &Value, run: &Run) -> Result<BTreeMap<&'static str, usize>, String> {
    let native = text(control, "native_from")?;
    let externals = text(control, "externals_from")?;
    let mut differences = BTreeMap::new();
    for stage_name in STAGES {
        let left = external(run, native, stage_name)?;
        let right = external(run, externals, stage_name)?;
        if left.dim() != right.dim() {
            return Err(format!(
                "settings {native:?} and {externals:?} executed {stage_name} at shapes {:?} and {:?}",
                left.dim(),
                right.dim()
            ));
        }
        let differing = left
            .iter()
            .zip(right.iter())
            .filter(|pair| pair.0.to_bits() != pair.1.to_bits())
            .count();
        differences.insert(stage_name, differing);
    }
    Ok(differences)
}

fn control_held(
    control: &Value,
    receipt: &SwigluBlockReceipt,
    differences: &BTreeMap<&'static str, usize>,
) -> Result<bool, String> {
    let stages = certified(receipt);
    match text(control, "kind")? {
        "refute" => {
            let named = control
                .get("stages")
                .and_then(Value::as_array)
                .ok_or("a refuting control names no stages")?;
            let mut held = !named.is_empty();
            for stage_name in named {
                let stage_name = stage_name.as_str().ok_or("a stage name is not a string")?;
                let agreement = stages
                    .iter()
                    .find(|entry| entry.0 == stage_name)
                    .ok_or_else(|| format!("unknown stage {stage_name:?}"))?;
                held &= agreement.1.refutes;
            }
            Ok(held)
        }
        "identity" => Ok(differences.values().all(|&differing| differing == 0)),
        "distinct" => Ok(differences.values().any(|&differing| differing > 0)),
        other => Err(format!("unknown control kind {other:?}")),
    }
}

/// The norm stage's receipt and its binary64 control, from the manifest's `norm` record.
fn norm_receipt(manifest: &Value, run: &Run) -> Result<(Value, bool), String> {
    let norm = manifest.get("norm").ok_or("manifest names no norm stage")?;
    let (class, internal) = (text(norm, "class")?, text(norm, "internal_dtype")?);
    let rsqrt_is_reciprocal_sqrt = norm
        .get("rsqrt_is_reciprocal_sqrt")
        .and_then(Value::as_bool)
        .ok_or("the norm record names no rsqrt_is_reciprocal_sqrt")?;
    let program = match (class, internal, rsqrt_is_reciprocal_sqrt) {
        ("Qwen3RMSNorm", "float32", true) => ExternalRmsNormProgram::Binary32Internal,
        _ => {
            return Err(format!(
                "no declared band program for a {class} that normalizes in {internal} \
                 (rsqrt_is_reciprocal_sqrt: {rsqrt_is_reciprocal_sqrt})"
            ));
        }
    };
    let epsilon = norm
        .get("epsilon")
        .and_then(Value::as_f64)
        .ok_or("the norm record names no epsilon")?;
    let gain = vector(Path::new(text(norm, "gain")?))?;
    let array = |key: &str| -> Result<&Array2<f64>, String> {
        let id = text(norm, key)?;
        run.matrices
            .get(id)
            .ok_or_else(|| format!("the norm stage's {key} {id:?} is not an exported array"))
    };
    let (inputs, output) = (array("inputs")?, array("output")?);
    let execution = ExternalExecution {
        dtype: text(norm, "dtype")?,
        device: text(norm, "device")?,
        tf32_matmul: run.tf32_matmul,
    };
    let receipt = |program| {
        rms_norm_stage(execution, program, epsilon, gain.view(), inputs.view(), output.view())
            .map_err(|refusal| format!("the norm stage refused: {refusal}"))
    };
    let certified = receipt(program)?;
    let binary64 = receipt(ExternalRmsNormProgram::Binary64)?;
    let passed = certified.agrees && binary64.refutes;
    println!(
        "[norm] {class} internal={internal} agrees={} binary64_control_refutes={}",
        certified.agrees, binary64.refutes
    );
    Ok((
        json!({
            "class": class,
            "internal_dtype": internal,
            "rsqrt_is_reciprocal_sqrt": rsqrt_is_reciprocal_sqrt,
            "program": format!("{program:?}"),
            "epsilon": epsilon,
            "stage": stage(&certified),
            "binary64_control": stage(&binary64),
            "passed": passed,
        }),
        passed,
    ))
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().collect();
    let run_dir = flag(&args, "--run")?;
    let declaration = read_json(&flag(&args, "--settings")?)?;
    let out = flag(&args, "--out")?;
    let manifest = read_json(&run_dir.join("manifest.json"))?;
    let run = load(&run_dir, &manifest)?;
    let settings = manifest
        .get("settings")
        .and_then(Value::as_array)
        .ok_or("manifest names no settings")?;

    let mut passed = true;
    let mut setting_reports = Vec::new();
    for setting in settings {
        let name = text(setting, "name")?.to_string();
        let case = Case {
            name: name.clone(),
            edits: setting_edits(setting, &run)?,
            externals: name.clone(),
        };
        let receipt = swiglu_receipt(&run, &case)?;
        let agrees = certified(&receipt).iter().all(|entry| entry.1.agrees);
        passed &= agrees;
        println!("[setting] {name} certified_stages_agree={agrees}");
        setting_reports.push(json!({
            "name": name,
            "certified_stages_agree": agrees,
            "gate": stage(&receipt.gate),
            "up": stage(&receipt.up),
            "activation": measured(&receipt.activation_measured),
            "hidden": stage(&receipt.hidden),
            "hidden_propagation": measured(&receipt.hidden_measured_propagation),
            "output": stage(&receipt.output),
            "end_to_end": measured(&receipt.end_to_end_measured),
        }));
    }

    let (norm_report, norm_passed) = norm_receipt(&manifest, &run)?;
    passed &= norm_passed;

    let controls = declaration
        .get("controls")
        .and_then(Value::as_array)
        .ok_or("settings.json declares no controls")?;
    let mut control_reports = Vec::new();
    for control in controls {
        let native_name = text(control, "native_from")?;
        let native = settings
            .iter()
            .find(|setting| setting.get("name").and_then(Value::as_str) == Some(native_name))
            .ok_or_else(|| format!("a control names an unknown native setting {native_name:?}"))?;
        let case = control_case(control, native, &run)?;
        let receipt = swiglu_receipt(&run, &case)?;
        let differences = executed_differences(control, &run)?;
        let held = control_held(control, &receipt, &differences)?;
        passed &= held;
        println!("[control] {} held={held}", case.name);
        control_reports.push(json!({
            "name": case.name,
            "kind": text(control, "kind")?,
            "held": held,
            "executed_differing_entries": differences,
            "gate": stage(&receipt.gate),
            "up": stage(&receipt.up),
            "hidden": stage(&receipt.hidden),
            "output": stage(&receipt.output),
        }));
    }

    let report = json!({
        "stage": "receipt",
        "run": run_dir.display().to_string(),
        "model": manifest.get("model"),
        "revision": manifest.get("revision"),
        "layer": manifest.get("layer"),
        "rows": manifest.get("rows"),
        "external_execution": {"dtype": run.dtype, "device": run.device, "tf32_matmul": run.tf32_matmul},
        "stage_columns": STAGES,
        "settings": setting_reports,
        "norm": norm_report,
        "controls": control_reports,
        "passed": passed,
    });
    let encoded = serde_json::to_string_pretty(&report).map_err(|err| format!("encode report: {err}"))?;
    std::fs::write(&out, encoded).map_err(|err| format!("write {}: {err}", out.display()))?;
    println!("[receipt] passed={passed} report={}", out.display());
    Ok(passed)
}

//! Native parameter edits for the edit comparison on VPD's 4-layer Pile target (#2951).
//!
//! VPD's case studies ablate one rank-one subcomponent `U_c V_cᵀ` at one position of one
//! prompt, and its emoticon edit rewrites a subcomponent's write vector. Neither ablation
//! at a position nor a subcomponent is a parameter of the network. Each is compiled here
//! into one global edit of the stored matrix by the native edit compiler
//! (`gam_mpd::compile::linear`): the requirement sets the site's output on
//! the control's inputs (the prompt's positions, or the positions where VPD's causal
//! importance marks the subcomponent), and the metric is the second moment `G` of the
//! site's off-target inputs, so the edit is the one that moves off-target outputs least in
//! mean square. A write the explanation leaves free (the emoticon edit's new write
//! direction) is the Fisher-optimal one, `w ∝ H⁻¹ ĝ`, with `H` the off-target Fisher of the
//! model's output pulled back to the site's output and `ĝ` the target's gradient there:
//! the maximizer of the first-order target gain per unit second-order off-target KL.
//!
//! Every tensor comes from `~/mpd-data/vpd/vpd_native_edits.py export` as `<f8` `.npy`
//! under a manifest:
//!
//! ```text
//! {"problems": [{"name", "storage", "native", "inputs", "class": "span" | "sample",
//!                "targets" | "write": {"read", "fisher", "gradient"},
//!                "moment"?, "off_target"?, "out"}]}
//! ```
//!
//! and each compiled plan is written as `<out>.left.npy`, `<out>.right.npy`
//! (`ΔW = left · rightᵀ`) with its status in `<out>.json`.
//!
//! usage: `cargo run --release -p gam-mpd --example mpd_vpd_native_edits_2951 -- MANIFEST.json`

use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use gam_linalg::roundoff::SymmetricAssembly;
use gam_runtime::resource::MemoryGovernor;
use gam_mpd::compile::ControlRealization;
use gam_mpd::compile::linear::{
    Coverage, EditMetric, LinearSiteProblem, OffTargetInputs, Requirement, ResponseClass, compile_linear_site,
};
use gam_mpd::dense::eigh;
use gam_mpd::lift::{TensorId, TensorRegistry, TieOrientation, UseMap, UseSiteId};
use gam_mpd::supports::EvidenceStatus;
use ndarray::{Array1, Array2, Axis};
use npyz::{NpyFile, Order, WriterBuilder};
use serde_json::{Value, json};

fn read_f8(path: &Path) -> Result<(Vec<usize>, Vec<f64>), String> {
    let file = std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let npy = NpyFile::new(BufReader::new(file)).map_err(|error| format!("{}: {error}", path.display()))?;
    if let Order::Fortran = npy.order() {
        return Err(format!("{}: expected C order", path.display()));
    }
    let shape = npy
        .shape()
        .iter()
        .map(|&extent| usize::try_from(extent).map_err(|_| format!("{}: extent {extent}", path.display())))
        .collect::<Result<Vec<_>, _>>()?;
    let values = npy
        .try_data::<f64>()
        .map_err(|npy| format!("{}: expected <f8, got {}", path.display(), npy.dtype().descr()))?
        .collect::<std::io::Result<Vec<f64>>>()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    Ok((shape, values))
}

fn matrix(path: &Path) -> Result<Array2<f64>, String> {
    let (shape, values) = read_f8(path)?;
    let [rows, cols] = shape[..] else {
        return Err(format!("{}: expected two axes, got {shape:?}", path.display()));
    };
    Array2::from_shape_vec((rows, cols), values).map_err(|error| format!("{}: {error}", path.display()))
}

fn vector(path: &Path) -> Result<Array1<f64>, String> {
    let (shape, values) = read_f8(path)?;
    if shape.len() != 1 {
        return Err(format!("{}: expected one axis, got {shape:?}", path.display()));
    }
    Ok(Array1::from(values))
}

fn write_f8(path: &Path, matrix: &Array2<f64>) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|error| format!("create {}: {error}", path.display()))?;
    let (rows, cols) = matrix.dim();
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[rows as u64, cols as u64])
        .writer(BufWriter::new(file))
        .begin_nd()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    writer
        .extend(matrix.iter().copied())
        .map_err(|error| format!("{}: {error}", path.display()))?;
    writer.finish().map_err(|error| format!("{}: {error}", path.display()))
}

fn field<'a>(entry: &'a Value, key: &str) -> Result<&'a str, String> {
    entry[key].as_str().ok_or_else(|| format!("problem field {key:?} is missing"))
}

/// `w = H⁻¹ ĝ / ‖H⁻¹ ĝ‖`, refused when the Fisher is singular within its band.
fn fisher_write(fisher: &Array2<f64>, gradient: &Array1<f64>) -> Result<Array1<f64>, String> {
    let decomposed = eigh(fisher.view(), SymmetricAssembly::Mirrored, None).map_err(|error| error.to_string())?;
    let smallest = decomposed.values.iter().fold(f64::INFINITY, |m, v| m.min(*v));
    if !(smallest > decomposed.band) {
        return Err(format!("the Fisher is singular: smallest eigenvalue {smallest:e}, band {:e}", decomposed.band));
    }
    let along = decomposed.vectors.t().dot(gradient) / &decomposed.values;
    let write = decomposed.vectors.dot(&along);
    let length = write.dot(&write).sqrt();
    Ok(write / length)
}

fn status_json<W: std::fmt::Debug, D: std::fmt::Debug>(status: &EvidenceStatus<W, D>) -> Value {
    match status {
        EvidenceStatus::Exact {
            value,
            numerical_error,
            basis,
            ..
        } => json!({"kind": "exact", "value": value, "band": numerical_error, "basis": format!("{basis:?}")}),
        other => json!({"kind": format!("{other:?}")}),
    }
}

fn compile(entry: &Value, root: &Path, governor: &MemoryGovernor) -> Result<Value, String> {
    let at = |key: &str| -> Result<PathBuf, String> { Ok(root.join(field(entry, key)?)) };
    let name = field(entry, "name")?;
    let storage = TensorId(field(entry, "storage")?.to_string());
    let site = UseSiteId(format!("{}#0", storage.0));
    let native = matrix(&at("native")?)?;
    let inputs = matrix(&at("inputs")?)?;
    let mut registry = TensorRegistry::default();
    registry
        .register_storage(storage.clone(), native.view().into_dyn())
        .map_err(|error| error.to_string())?;
    registry
        .register_use_site(site.clone(), storage.clone(), UseMap::Linear(TieOrientation::Identity))
        .map_err(|error| error.to_string())?;
    let (targets, write) = if entry["write"].is_object() {
        let spec = &entry["write"];
        let read = vector(&root.join(field(spec, "read")?))?;
        let fisher = matrix(&root.join(field(spec, "fisher")?))?;
        let gradient = vector(&root.join(field(spec, "gradient")?))?;
        let write = fisher_write(&fisher, &gradient)?;
        // The set-type target on each input: the native output plus the read times the write.
        let changes = read.insert_axis(Axis(1)).dot(&write.view().insert_axis(Axis(0)));
        (inputs.dot(&native.t()) + changes, Some(write))
    } else {
        (matrix(&at("targets")?)?, None)
    };
    let class = match field(entry, "class")? {
        "span" => ResponseClass::Span(inputs.view()),
        "sample" => ResponseClass::Sample,
        other => return Err(format!("unknown class {other:?}")),
    };
    let metric = match entry["moment"].as_str() {
        Some(path) => EditMetric::input_moment(matrix(&root.join(path))?.view(), SymmetricAssembly::Mirrored)
            .map_err(|error| error.to_string())?,
        None => EditMetric::frobenius(),
    };
    let off_target = match entry["off_target"].as_str() {
        Some(path) => Some(matrix(&root.join(path))?),
        None => None,
    };
    let problem = LinearSiteProblem {
        registry: &registry,
        storage: storage.clone(),
        native: native.view(),
        requirements: vec![Requirement::Linear {
            site: site.clone(),
            inputs: inputs.view(),
            targets: targets.view(),
            target_radius: 0.0,
            class,
        }],
        metric,
        ties: Vec::new(),
        off_target: off_target
            .iter()
            .map(|inputs| OffTargetInputs {
                site: site.clone(),
                inputs: inputs.view(),
            })
            .collect(),
    };
    let report = compile_linear_site(&problem, name, governor).map_err(|error| error.to_string())?;
    let (realization, residual) = match &report.compiled.realization {
        ControlRealization::ExactlyRealized { residual, .. } => ("ExactlyRealized", status_json(residual)),
        ControlRealization::EmpiricallyValidated { residual, .. } => ("EmpiricallyValidated", status_json(residual)),
        ControlRealization::Descriptive { reason, witness, .. } => (
            "Descriptive",
            json!({"reason": format!("{reason:?}"), "witness": witness.as_ref().map(|w| format!("{w:?}"))}),
        ),
    };
    let out = at("out")?;
    let mut terms = 0;
    if let Some(plan) = &report.compiled.plan {
        for edit in plan.edits() {
            terms = edit.delta.term_count();
            write_f8(&out.with_extension("left.npy"), &edit.delta.left().to_owned())?;
            write_f8(&out.with_extension("right.npy"), &edit.delta.right().to_owned())?;
        }
    }
    let summary = json!({
        "name": name,
        "storage": storage.0,
        "columns": inputs.nrows(),
        "right_rank": report.right_rank,
        "realization": realization,
        "residual": residual,
        "covered": report.coverage.iter().all(|c| matches!(c, Coverage::Covered)),
        "metric_norm": report.metric_norm.map(|(value, band)| json!({"value": value, "band": band})),
        "allowance": report.allowance,
        "terms": terms,
        "off_target_sup": report.off_target_damage.as_ref().map(status_json),
        "fisher_write": write.map(|w| w.to_vec()),
    });
    std::fs::write(out.with_extension("json"), serde_json::to_string_pretty(&summary).expect("json"))
        .map_err(|error| error.to_string())?;
    Ok(summary)
}

fn main() -> Result<(), String> {
    let manifest_path = PathBuf::from(std::env::args().nth(1).ok_or("usage: mpd_vpd_native_edits_2951 MANIFEST.json")?);
    let root = manifest_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(&manifest_path).map_err(|error| format!("read {}: {error}", manifest_path.display()))?,
    )
    .map_err(|error| error.to_string())?;
    let governor = MemoryGovernor::global();
    for entry in manifest["problems"].as_array().ok_or("manifest has no problems")? {
        let mut summary = compile(entry, &root, governor)?;
        summary["fisher_write"] = summary["fisher_write"].as_array().map_or(Value::Null, |w| json!(w.len()));
        println!("{summary}");
    }
    Ok(())
}

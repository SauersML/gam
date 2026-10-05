//! Training-only finite composed-program proposal search; heldout scores follow frozen Pareto IDs.
//! EXTRACT.json SETTINGS.json FRESH_OUT host|cuda. Standalone programs, not native Local/Run.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    acceptance::{CostCache, structural_cost},
    artifact::Artifact,
    coder_capture::sha256,
    composed_rule_search::{self, Grammar, UseSpec},
    resident_rule_fit::{self, GroupMeasurement, Settings as FitSettings},
};
use ndarray::{Array2, s};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    layers: Vec<usize>,
    width: usize,
    grammar: Grammar,
    fit: FitSettings,
    #[serde(default)]
    batch_schedule: Option<resident_rule_fit::BatchSchedule>,
    seed: u64,
}
fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn journal(file: &mut File, value: &Value) -> Result<(), String> {
    serde_json::to_writer(&mut *file, value).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())
}
fn integer(v: &Value, key: &str) -> Result<usize, String> {
    usize::try_from(
        v[key]
            .as_u64()
            .ok_or_else(|| format!("missing integer {key}"))?,
    )
    .map_err(|e| e.to_string())
}
fn panel<'a>(manifest: &'a Value, name: &str) -> Result<&'a Value, String> {
    let found: Vec<_> = manifest["panels"]
        .as_array()
        .ok_or("panels absent")?
        .iter()
        .filter(|v| v["name"].as_str() == Some(name))
        .collect();
    if found.len() != 1 {
        return Err(format!("unique {name} panel required"));
    }
    Ok(found[0])
}
fn load(
    root: &Path,
    panel: &Value,
    layer: usize,
    role: &str,
) -> Result<(Array2<f64>, Value), String> {
    let matches: Vec<_> = panel["values"]
        .as_array()
        .ok_or("panel values")?
        .iter()
        .filter(|v| v["layer"].as_u64() == Some(layer as u64) && v["role"].as_str() == Some(role))
        .collect();
    if matches.len() != 1 {
        return Err(format!("unique layer{layer}/{role} required"));
    }
    let descriptor = matches[0];
    let name = descriptor["file"].as_str().ok_or("array file")?;
    if Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err("sibling array filename required".into());
    }
    let path = root.join(name);
    let hash = sha256(&path)?;
    if descriptor["sha256"].as_str() != Some(&hash) {
        return Err(format!("array SHA mismatch {}", path.display()));
    }
    let rows = integer(panel, "rows")?;
    let width = integer(descriptor, "width")?;
    if rows == 0 || width == 0 {
        return Err("positive archive dimensions required".into());
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    if bytes.len()
        != rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(8))
            .ok_or("array size overflow")?
    {
        return Err("array shape mismatch".into());
    }
    let values: Vec<_> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("eight byte chunk")))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("finite native array required".into());
    }
    Ok((
        Array2::from_shape_vec((rows, width), values).map_err(|e| e.to_string())?,
        json!({"layer":layer,"role":role,"descriptor":descriptor,"path":path,"sha256":hash}),
    ))
}
struct Data {
    inputs: Vec<Array2<f64>>,
    targets: Array2<f64>,
    provenance: Value,
    output_widths: Vec<usize>,
}
fn load_data(root: &Path, panel: &Value, layers: &[usize]) -> Result<Data, String> {
    let rows = integer(panel, "rows")?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut descriptors = Vec::new();
    let mut total = 0usize;
    let mut output_widths = Vec::new();
    for &layer in layers {
        let (x, xd) = load(root, panel, layer, "input")?;
        let (y, yd) = load(root, panel, layer, "write")?;
        if x.nrows() != rows || y.nrows() != rows {
            return Err("aligned rows required".into());
        }
        total = total
            .checked_add(y.ncols())
            .ok_or("target width overflow")?;
        output_widths.push(y.ncols());
        inputs.push(x);
        outputs.push(y);
        descriptors.extend([xd, yd]);
    }
    let mut targets = Array2::zeros((rows, total));
    let mut at = 0;
    for y in outputs {
        let end = at + y.ncols();
        targets.slice_mut(s![.., at..end]).assign(&y);
        at = end;
    }
    Ok(Data {
        inputs,
        targets,
        output_widths,
        provenance: json!({"name":panel["name"],"rows":rows,"record":panel["record"],"arrays":descriptors}),
    })
}
#[derive(Clone)]
struct Scored {
    id: usize,
    expression: usize,
    arm: &'static str,
    c32: u64,
    training: f64,
    path: PathBuf,
    measurement: GroupMeasurement,
    groups: Vec<resident_rule_fit::OutputGroup>,
    declarations: gam_mpd::operator_program::Declarations,
}
fn pareto(scores: &[Scored]) -> Vec<usize> {
    scores
        .iter()
        .filter(|candidate| {
            !scores.iter().any(|other| {
                other.c32 <= candidate.c32
                    && other.training <= candidate.training
                    && (other.c32 < candidate.c32 || other.training < candidate.training)
            })
        })
        .map(|v| v.id)
        .collect()
}
fn evaluation_ids(scores: &[Scored], pareto_ids: &[usize], arm_count: usize) -> Vec<usize> {
    let expressions: BTreeSet<_> = scores
        .iter()
        .filter(|v| pareto_ids.contains(&v.id))
        .map(|v| v.expression)
        .collect();
    expressions
        .into_iter()
        .flat_map(|expression| (0..arm_count).map(move |arm| expression * arm_count + arm))
        .collect()
}
fn validate_heldout_shapes(train: &Data, eval: &Data, uses: &[UseSpec]) -> Result<(), String> {
    if eval.inputs.len() != uses.len()
        || eval.output_widths != train.output_widths
        || eval.targets.ncols() != train.targets.ncols()
    {
        return Err("heldout per-use output interfaces differ".into());
    }
    for (input, use_spec) in eval.inputs.iter().zip(uses) {
        if input.ncols() != use_spec.input_width {
            return Err("heldout input interface differs".into());
        }
    }
    Ok(())
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("EXTRACT.json SETTINGS.json FRESH_OUT host|cuda".into());
    }
    let started = Instant::now();
    let manifest_path = Path::new(&args[0]);
    let settings_path = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    if out.exists() {
        return Err("fresh output required".into());
    }
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(manifest_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let settings_value: Value =
        serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let settings: Settings =
        serde_json::from_value(settings_value.clone()).map_err(|e| e.to_string())?;
    if settings.layers.is_empty()
        || settings.width == 0
        || settings
            .layers
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != settings.layers.len()
    {
        return Err("positive width and nonempty unique layers required".into());
    }
    let device = match args[3].as_str() {
        "host" => Device::host(),
        "cuda" => Device::accelerator(GpuPolicy::Required)
            .map_err(|e| e.to_string())?
            .ok_or("CUDA required")?,
        _ => return Err("host|cuda backend required".into()),
    };
    if args[3] == "cuda" && (device.is_host() || !device.float64()) {
        return Err("actual float64 accelerator required".into());
    }
    let root = manifest_path.parent().ok_or("manifest parent")?;
    let train_panel = panel(&manifest, "train")?;
    let eval_panel = panel(&manifest, "eval")?;
    for native_panel in [train_panel, eval_panel] {
        let hash = native_panel["record"]["source"]["checkpoint_sha256"]
            .as_str()
            .ok_or("explicit panel source SHA required")?;
        if hash.len() != 64
            || !hash.bytes().all(|b| b.is_ascii_hexdigit())
            || !native_panel["record"]["config"].is_object()
        {
            return Err("explicit 64-digit source SHA and config object required".into());
        }
    }
    if train_panel["record"]["source"]["checkpoint_sha256"]
        != eval_panel["record"]["source"]["checkpoint_sha256"]
        || train_panel["record"]["config"] != eval_panel["record"]["config"]
    {
        return Err("native panel checkpoint/config mismatch".into());
    }
    let train = load_data(root, train_panel, &settings.layers)?;
    let uses: Vec<_> = settings
        .layers
        .iter()
        .enumerate()
        .map(|(slot, _)| UseSpec {
            input_width: train.inputs[slot].ncols(),
            output_width: train.output_widths[slot],
        })
        .collect();
    let inventory = composed_rule_search::enumerate(&settings.grammar)?;
    let arms: Vec<&'static str> = if uses.len() > 1 {
        vec!["shared", "untied"]
    } else {
        vec!["shared"]
    };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    save(
        &out.join("PROVENANCE.json"),
        &json!({"settings":settings_value,"settings_sha256":sha256(settings_path)?,"extract_sha256":sha256(manifest_path)?,"train":train.provenance,"eval_manifest_only_until_pareto_freeze":eval_panel,"backend":args[3],"seed":settings.seed,"width":settings.width,"layers":settings.layers,"arms":arms,"grammar_truncated":inventory.truncated,"scope":"Finite proposed composed-program grammar only. Training arrays passed as BOTH fit training and fit validation; all optimizer validation fields during search refer to TRAIN. Training selection uses ordinary decoded-f32 F64 training measurement plus complete standalone C32. Heldout arrays loaded only after Pareto IDs and matched-control evaluation IDs saved. Row-disjointness is asserted by the recorded extraction lineage, not proven by unequal array hashes. No native full-model Local/Run, discovered algorithm, or global grammar optimum claim."}),
    )?;
    let arm_count = arms.len();
    let entries:Vec<_>=inventory.expressions.iter().enumerate().flat_map(|(expression,body)|arms.iter().enumerate().map(move |(a,arm)|json!({"id":expression*arm_count+a,"expression_index":expression,"expression":body,"arm":arm,"initial_status":"unmeasured"}))).collect();
    save(
        &out.join("INVENTORY.json"),
        &json!({"expressions":inventory.expressions,"intermediate_expressions":inventory.intermediate_expressions,"truncated":inventory.truncated,"proposals":entries,"proposal_count":entries.len()}),
    )?;
    let mut log = File::create(out.join("journal.jsonl")).map_err(|e| e.to_string())?;
    let mut scores = Vec::new();
    let mut statuses = Vec::new();
    for (expression, expr) in inventory.expressions.iter().enumerate() {
        for (a, &arm) in arms.iter().enumerate() {
            let id = expression * arms.len() + a;
            let candidate_started = Instant::now();
            let candidate = out.join(format!("candidate{id:05}"));
            std::fs::create_dir(&candidate).map_err(|e| e.to_string())?;
            journal(
                &mut log,
                &json!({"id":id,"expression_index":expression,"arm":arm,"stage":"started","expression":expr}),
            )?;
            let attempt = (|| -> Result<Scored, String> {
                let proposal = if arm == "shared" {
                    composed_rule_search::compile(expr, settings.width, &uses, settings.seed)?
                } else {
                    composed_rule_search::compile_untied(
                        expr,
                        settings.width,
                        &uses,
                        settings.seed,
                    )?
                };
                save(
                    &candidate.join("DECLARATION.json"),
                    &json!({"expression":expr,"arm":arm,"seed":settings.seed,"groups":proposal.groups,"trainable":proposal.trainable,"declarations":{"domains":[],"raw_slot_widths":train.inputs.iter().map(|x|x.ncols()).collect::<Vec<_>>(),"parameters":proposal.program.declarations.parameters}}),
                )?;
                let fit = if let Some(schedule) = &settings.batch_schedule {
                    resident_rule_fit::fit_grouped_batched(
                        &device,
                        &proposal.program,
                        &train.inputs,
                        &train.targets,
                        &train.inputs,
                        &train.targets,
                        &proposal.groups,
                        &proposal.trainable,
                        settings.fit.clone(),
                        schedule.clone(),
                    )?
                } else {
                    resident_rule_fit::fit_grouped(
                        &device,
                        &proposal.program,
                        &train.inputs,
                        &train.targets,
                        &train.inputs,
                        &train.targets,
                        &proposal.groups,
                        &proposal.trainable,
                        settings.fit.clone(),
                    )?
                };
                save(
                    &candidate.join("FIT.json"),
                    &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?,
                )?;
                let artifact = Artifact::native(&fit.program)?.f32_literals()?;
                let bytes = artifact.to_bytes()?;
                let path = candidate.join("program.artifact");
                std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
                let saved = std::fs::read(&path).map_err(|e| e.to_string())?;
                let decoded = Artifact::from_bytes(&saved, &artifact.program.declarations)?;
                if decoded.to_bytes()? != saved {
                    return Err("ordinary saved-byte canonical replay mismatch".into());
                }
                let cost = structural_cost(&artifact, &mut CostCache::default())?;
                let decoded_cost = structural_cost(&decoded, &mut CostCache::default())?;
                if cost != decoded_cost {
                    return Err("ordinary decoded C32 mismatch".into());
                }
                let measurement = resident_rule_fit::measure_grouped(
                    &device,
                    &decoded.program,
                    &train.inputs,
                    &train.targets,
                    &proposal.groups,
                    settings.fit.numeric_bytes,
                    settings.fit.forward_rows,
                )?;
                if !measurement.maximum.is_finite() {
                    return Err("nonfinite decoded training maximum".into());
                }
                save(
                    &candidate.join("TRAIN.json"),
                    &json!({"id":id,"arm":arm,"expression":expr,"standalone_cost":cost,"standalone_c32":cost.total(),"artifact_sha256":sha256(&path)?,"training":measurement,"scope":"complete standalone program C32; remaining native model not included"}),
                )?;
                Ok(Scored {
                    id,
                    expression,
                    arm,
                    c32: cost.total(),
                    training: measurement.maximum,
                    path,
                    measurement,
                    groups: proposal.groups,
                    declarations: artifact.program.declarations.clone(),
                })
            })();
            let status = match attempt {
                Ok(score) => {
                    let record = json!({"id":id,"expression_index":expression,"arm":arm,"status":"training_measured","standalone_c32":score.c32,"training_max":score.training,"artifact":score.path,"seconds":candidate_started.elapsed().as_secs_f64()});
                    scores.push(score);
                    record
                }
                Err(error) => {
                    json!({"id":id,"expression_index":expression,"arm":arm,"status":"unresolved","error":error,"seconds":candidate_started.elapsed().as_secs_f64()})
                }
            };
            save(&candidate.join("STATUS.json"), &status)?;
            journal(&mut log, &status)?;
            statuses.push(status);
        }
    }
    let selected = pareto(&scores);
    let frozen_evaluation_ids = evaluation_ids(&scores, &selected, arm_count);
    save(
        &out.join("TRAINING_PARETO.json"),
        &json!({"ids":selected,"frozen_evaluation_ids":frozen_evaluation_ids,"evaluation_policy":"Pareto plus same-expression matched shared/untied controls, even training-dominated; failed controls remain unresolved","frozen_before_heldout_array_load":true,"criteria":["decoded F64 training maximum","complete standalone C32"],"tie_policy":"retain exact ties; dominance requires one strict improvement","candidates":scores.iter().filter(|v|selected.contains(&v.id)).map(|v|json!({"id":v.id,"expression_index":v.expression,"arm":v.arm,"training":v.measurement,"standalone_c32":v.c32,"artifact":v.path})).collect::<Vec<_>>(),"scope":"Pareto of successfully measured finite fitted proposals; failed/unmeasured proposals unresolved, no completeness or optimality claim"}),
    )?;
    let heldout_started = Instant::now();
    let heldout = if frozen_evaluation_ids.is_empty() {
        Err("no frozen evaluation IDs; heldout arrays not loaded".into())
    } else {
        (|| -> Result<Data, String> {
            let data = load_data(root, eval_panel, &settings.layers)?;
            validate_heldout_shapes(&train, &data, &uses)?;
            let ta = train.provenance["arrays"]
                .as_array()
                .ok_or("train provenance")?;
            let va = data.provenance["arrays"]
                .as_array()
                .ok_or("eval provenance")?;
            if ta.iter().zip(va).any(|(a, b)| a["sha256"] == b["sha256"]) {
                return Err("identical train/eval arrays refused; different hashes alone do not prove row disjointness".into());
            }
            Ok(data)
        })()
    };
    let mut evaluations = Vec::new();
    for &id in &frozen_evaluation_ids {
        let evaluation = (|| -> Result<Value, String> {
            let data = heldout.as_ref().map_err(Clone::clone)?;
            let score = scores.iter().find(|v| v.id == id).ok_or(
                "matched control failed during training; no saved candidate, remains unresolved",
            )?;
            let artifact = Artifact::from_bytes(
                &std::fs::read(&score.path).map_err(|e| e.to_string())?,
                &score.declarations,
            )?;
            let measurement = resident_rule_fit::measure_grouped(
                &device,
                &artifact.program,
                &data.inputs,
                &data.targets,
                &score.groups,
                settings.fit.numeric_bytes,
                settings.fit.forward_rows,
            )?;
            Ok(
                json!({"id":score.id,"status":"heldout_measured","measurement":measurement,"artifact_sha256":sha256(&score.path)?}),
            )
        })();
        let record = match evaluation {
            Ok(v) => v,
            Err(error) => json!({"id":id,"status":"heldout_unresolved","error":error}),
        };
        save(
            &out.join(format!("candidate{id:05}")).join("HELDOUT.json"),
            &record,
        )?;
        journal(&mut log, &record)?;
        evaluations.push(record);
    }
    save(
        &out.join("REPORT.json"),
        &json!({"proposal_count":entries.len(),"grammar_truncated":inventory.truncated,"statuses":statuses,"frozen_training_pareto_ids":selected,"frozen_evaluation_ids":frozen_evaluation_ids,"heldout":evaluations,"heldout_provenance":heldout.as_ref().ok().map(|d|&d.provenance),"heldout_panel_error":heldout.as_ref().err(),"heldout_seconds":heldout_started.elapsed().as_secs_f64(),"seconds":started.elapsed().as_secs_f64(),"scope":"automatic finite structural proposals with learned maps, not full native acceptance or global optimality; all errors unresolved and retained"}),
    )
}
fn main() -> Result<(), String> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn score(id: usize, c32: u64, training: f64) -> Scored {
        Scored {
            id,
            expression: id,
            arm: "shared",
            c32,
            training,
            path: PathBuf::from(format!("{id}.artifact")),
            measurement: GroupMeasurement {
                maximum: training,
                worst_row: 0,
                worst_group: 0,
                group_maxima: vec![training],
                scales: vec![],
            },
            groups: vec![],
            declarations: gam_mpd::operator_program::Declarations {
                domains: vec![],
                slots: vec![],
                parameters: 0,
            },
        }
    }
    #[test]
    fn optional_batch_schedule_does_not_change_legacy_settings() {
        let fixture = json!({"layers":[0,1],"width":8,"grammar":{"arguments":1,"max_operations":3,"max_expressions":10000,"unary":["relu"],"binary":["multiply"],"affine":true},"fit":{"iterations":128,"forward_rows":128,"learning_rate":0.01,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"numeric_bytes":536870912,"arithmetic":"f64","backtracking":null},"seed":2951});
        let legacy: Settings =
            serde_json::from_value(fixture.clone()).expect("legacy calibration settings");
        assert!(legacy.batch_schedule.is_none());
        let mut explicit = fixture;
        explicit["batch_schedule"] =
            json!({"ordinary_rows":64,"hard_rows":16,"scan_every":8,"temperature":0.02});
        let explicit: Settings = serde_json::from_value(explicit).expect("optional schedule");
        assert_eq!(
            explicit
                .batch_schedule
                .expect("declared schedule")
                .scan_every,
            8
        );
        assert_eq!(legacy.fit.iterations, explicit.fit.iterations);
    }
    #[test]
    fn matched_controls_freeze_even_dominated_or_failed() {
        let mut shared = score(0, 10, 1.);
        shared.expression = 0;
        let mut untied = score(1, 20, 2.);
        untied.expression = 0;
        untied.arm = "untied";
        assert_eq!(pareto(&[shared.clone(), untied.clone()]), vec![0]);
        assert_eq!(
            evaluation_ids(&[shared.clone(), untied], &[0], 2),
            vec![0, 1]
        );
        assert_eq!(evaluation_ids(&[shared], &[0], 2), vec![0, 1]);
    }
    #[test]
    fn heldout_cannot_mix_group_widths_with_equal_total_width() {
        let data = |widths| Data {
            inputs: vec![Array2::zeros((1, 3)), Array2::zeros((1, 3))],
            targets: Array2::zeros((1, 4)),
            output_widths: widths,
            provenance: Value::Null,
        };
        let train = data(vec![2, 2]);
        let bad = data(vec![1, 3]);
        let uses = vec![
            UseSpec {
                input_width: 3,
                output_width: 2,
            },
            UseSpec {
                input_width: 3,
                output_width: 2,
            },
        ];
        assert!(validate_heldout_shapes(&train, &bad, &uses).is_err());
        assert!(validate_heldout_shapes(&train, &data(vec![2, 2]), &uses).is_ok());
    }
    #[test]
    fn training_pareto_preserves_tradeoffs_and_exact_ties() {
        let scores = vec![
            score(0, 10, 1.),
            score(1, 20, 2.),
            score(2, 5, 3.),
            score(3, 10, 1.),
        ];
        assert_eq!(pareto(&scores), vec![0, 2, 3]);
        assert!(pareto(&[]).is_empty());
    }
}

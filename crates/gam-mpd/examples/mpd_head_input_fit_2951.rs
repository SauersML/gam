//! Native-input weighted head proposals. Extraction uses a GPU; fitting is a separate CPU job.
//! extract TRAIN_EXPORT EVAL_EXPORT OUT_DIR TRAIN_SEQUENCES EVAL_SEQUENCES CONTEXT TRACE_BYTES
//! fit MODEL_EXPORT EXTRACT_DIR OUT_JSON RANKS
//! All heads are reported. These linear diagnostics are not Local/Run acceptance.
use gam_mpd::{
    acceptance::{CostCache, structural_cost},
    artifact::Artifact,
    coder_capture::sha256,
    device_program::DeviceProgram,
    import::import_language_model,
    operator_program::{Node, Operator, SlotValues},
    proposals::{
        CopyMasks, CopyResidualBank, CopyResidualChoice, DataWeightedSvd, HeadApproximation,
    },
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use serde_json::json;
use std::{io::Write, path::Path, sync::Arc, time::Instant};

fn positive(s: &str) -> Result<usize, String> {
    s.parse::<usize>().map_err(|e| e.to_string()).and_then(|v| {
        if v > 0 {
            Ok(v)
        } else {
            Err("positive dimension required".into())
        }
    })
}
fn save(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn read(path: &Path, rows: usize, width: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if rows.checked_mul(width).and_then(|n| n.checked_mul(8)) != Some(bytes.len()) {
        return Err("activation file shape mismatch".into());
    }
    let numbers = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Array2::from_shape_vec((rows, width), numbers).map_err(|e| e.to_string())
}
fn verified_read(
    data: &Path,
    family: &serde_json::Value,
    layer: usize,
    head: usize,
    rows: usize,
    width: usize,
    node: usize,
    operator: &str,
) -> Result<Array2<f64>, String> {
    let matches: Vec<_> = family["heads"]
        .as_array()
        .ok_or("manifest heads")?
        .iter()
        .filter(|h| h["layer"] == layer && h["head"] == head)
        .collect();
    if matches.len() != 1 {
        return Err("manifest needs exactly one matching head".into());
    }
    let entry = matches[0];
    let filename = format!(
        "{}.{layer}.{head}.f64",
        family["name"].as_str().ok_or("family name")?
    );
    if entry["file"] != filename
        || entry["shape"] != json!([rows, width])
        || entry["node"] != node
        || entry["operator"] != operator
        || entry["sha256"] != sha256(&data.join(&filename))?
    {
        return Err("activation manifest lineage, shape or digest mismatch".into());
    }
    read(&data.join(filename), rows, width)
}
fn extract(a: &[String]) -> Result<(), String> {
    if a.len() != 7 {
        return Err(
            "extract TRAIN_EXPORT EVAL_EXPORT OUT_DIR TRAIN_SEQUENCES EVAL_SEQUENCES CONTEXT TRACE_BYTES"
                .into(),
        );
    }
    let output = Path::new(&a[2]);
    if output.exists() {
        return Err("fresh extraction directory required".into());
    }
    let (train_n, eval_n, context) = (positive(&a[3])?, positive(&a[4])?, positive(&a[5])?);
    let trace_budget = positive(&a[6])?;
    let train = import_language_model(Path::new(&a[0]), train_n, context)?;
    let eval = import_language_model(Path::new(&a[1]), eval_n, context)?;
    let (Some(SlotValues::Tokens(train_tokens)), Some(SlotValues::Tokens(eval_tokens))) = (
        train.contract.family.slots.first(),
        eval.contract.family.slots.first(),
    ) else {
        return Err("native token slot required".into());
    };
    let train_sequences: std::collections::BTreeSet<_> =
        train_tokens.chunks_exact(context).collect();
    if eval_tokens
        .chunks_exact(context)
        .any(|row| train_sequences.contains(row))
    {
        return Err("training and evaluation contain an identical token sequence".into());
    }
    if train.record["source"]["checkpoint_sha256"] != eval.record["source"]["checkpoint_sha256"]
        || train.record["config"] != eval.record["config"]
    {
        return Err(
            "training and evaluation exports must identify the same checkpoint and configuration"
                .into(),
        );
    }
    let base = Artifact::native(&split_sites(&train.program)?)?.f32_literals()?;
    let layers = layer_nodes(
        &base.program,
        train.record["config"]["n_layers"]
            .as_u64()
            .ok_or("n_layers")? as usize,
    )?;
    // No labels or saved teacher logits enter this extraction or proposal fit.
    // Confirm identical actual f32 native operators, not just matching metadata.
    let eval_base = Artifact::native(&split_sites(&eval.program)?)?.f32_literals()?;
    if base.program.operators.len() != eval_base.program.operators.len()
        || base
            .program
            .operators
            .iter()
            .zip(&eval_base.program.operators)
            .any(|(x, y)| x.name != y.name || x.matrix_cow() != y.matrix_cow())
    {
        return Err("native matrices differ between training and evaluation exports".into());
    }
    drop(eval_base);
    let device = gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required)
        .map_err(|e| e.to_string())?
        .ok_or("CUDA required")?;
    let resident = DeviceProgram::compile(&device, &base.program)?;
    let requested_trace_bytes = resident
        .bytes_per_row()
        .checked_mul(train.contract.family.rows.max(eval.contract.family.rows))
        .ok_or("trace size overflow")?;
    if requested_trace_bytes > trace_budget {
        return Err(format!(
            "native resident trace needs {requested_trace_bytes} bytes, declared cap {trace_budget}; choose fewer complete sequences"
        ));
    }
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let mut families = Vec::new();
    for (name, imported) in [("train", &train), ("eval", &eval)] {
        let start = Instant::now();
        let trace = resident.forward_edited_intermediates(
            &imported.contract.family,
            |_, _| Ok(()),
            |_, _| Ok(None),
        )?;
        let mut heads = Vec::new();
        for (layer, nodes) in layers.iter().enumerate() {
            for (head, &node) in nodes.reads.iter().enumerate() {
                let matrix = device
                    .download(trace.value(node)?)
                    .map_err(|e| e.to_string())?;
                if matrix.iter().any(|v| !v.is_finite()) {
                    return Err("nonfinite native read".into());
                }
                let filename = format!("{name}.{layer}.{head}.f64");
                let bytes: Vec<_> = matrix.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(output.join(&filename), bytes).map_err(|e| e.to_string())?;
                heads.push(json!({"layer":layer,"head":head,"node":node,"operator":format!("blocks.{layer}.o{head}"),"file":filename,"shape":matrix.dim(),"sha256":sha256(&output.join(&filename))?}));
            }
        }
        let export = Path::new(if name == "train" { &a[0] } else { &a[1] });
        families.push(json!({"name":name,"rows":imported.contract.family.rows,"heads":heads,"seconds":start.elapsed().as_secs_f64(),"source":imported.record["source"],"export_sha256":sha256(&export.join("export.json"))?,"tokens_file_sha256":sha256(&export.join("tokens.f64"))?}));
    }
    save(
        &output.join("manifest.json"),
        &json!({"checkpoint_sha256":train.record["source"]["checkpoint_sha256"],"config":train.record["config"],"context":context,"families":families,"identical_train_eval_token_sequences":0,"requested_resident_trace_bytes":requested_trace_bytes,"resident_trace_budget":trace_budget,"budget_excludes":"native parameters, attention scratch, allocator and CUDA context","native":"original f32 literals, unmodified model CUDA f64; no output head evaluated","fit":"none; GPU allocation ends after extraction"}),
    )
}
fn energy(x: &Array2<f64>) -> f64 {
    x.iter().map(|v| v * v).sum()
}
fn metrics(x: &Array2<f64>, native: &Array2<f64>, candidate: &Array2<f64>) -> serde_json::Value {
    let target = x.dot(&native.t());
    let error = x.dot(&(candidate - native).t());
    let worst = error
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    json!({"relative_RMS":(energy(&error)/energy(&target)).sqrt(),"absolute_worst_L2":worst,"native_contribution_RMS":(energy(&target)/x.nrows() as f64).sqrt()})
}
fn replace(
    base: &Artifact,
    index: usize,
    op: Operator,
    normed: usize,
    attention: usize,
) -> Result<Artifact, String> {
    let mut candidate = base.clone();
    candidate.program.operators[index] = Arc::new(op);
    candidate.bind("native-input weighted attention", &[normed], attention)
}
fn fit(a: &[String]) -> Result<(), String> {
    if a.len() != 4 {
        return Err("fit MODEL_EXPORT EXTRACT_DIR OUT_JSON RANKS".into());
    }
    let output = Path::new(&a[2]);
    if output.exists() {
        return Err("fresh report required".into());
    }
    let ranks: Vec<_> = a[3].split(',').map(positive).collect::<Result<_, _>>()?;
    let data = Path::new(&a[1]);
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(data.join("manifest.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let imported = import_language_model(Path::new(&a[0]), 1, 1)?;
    if manifest["checkpoint_sha256"] != imported.record["source"]["checkpoint_sha256"]
        || manifest["config"] != imported.record["config"]
    {
        return Err("fit checkpoint mismatch".into());
    }
    if manifest["families"].as_array().map(Vec::len) != Some(2)
        || manifest["families"][0]["name"] != "train"
        || manifest["families"][1]["name"] != "eval"
        || manifest["families"][1]["export_sha256"]
            != sha256(&Path::new(&a[0]).join("export.json"))?
        || manifest["identical_train_eval_token_sequences"] != 0
    {
        return Err(
            "training/evaluation manifest mismatch; fit MODEL_EXPORT must be evaluation export"
                .into(),
        );
    }
    let base = Artifact::native(&split_sites(&imported.program)?)?.f32_literals()?;
    let layers = layer_nodes(
        &base.program,
        manifest["config"]["n_layers"].as_u64().ok_or("layers")? as usize,
    )?;
    let heads = manifest["config"]["n_heads"].as_u64().ok_or("heads")? as usize;
    let kv = manifest["config"]["n_kv_heads"]
        .as_u64()
        .ok_or("kv_heads")? as usize;
    if kv == 0 || heads % kv != 0 {
        return Err("invalid query/KV group".into());
    }
    let bank = CopyResidualBank::new(&base, &layers, heads / kv, &ranks, usize::MAX)?;
    let copies = CopyMasks::new(&base, &layers, heads / kv, usize::MAX, usize::MAX)?;
    let train_rows = manifest["families"][0]["rows"]
        .as_u64()
        .ok_or("train rows")? as usize;
    let eval_rows = manifest["families"][1]["rows"]
        .as_u64()
        .ok_or("eval rows")? as usize;
    let start = Instant::now();
    let mut journal = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.with_extension("jsonl"))
        .map_err(|e| e.to_string())?;
    let mut costs = CostCache::default();
    let native_bits = structural_cost(&base, &mut costs)?.total();
    let mut results = Vec::new();
    for (layer, nodes) in layers.iter().enumerate() {
        for head in 0..nodes.reads.len() {
            let name = format!("blocks.{layer}.o{head}");
            let index = base
                .program
                .operators
                .iter()
                .position(|op| op.name == name)
                .ok_or("head operator missing")?;
            let original = &base.program.operators[index];
            let native = original.matrix();
            let x = verified_read(
                data,
                &manifest["families"][0],
                layer,
                head,
                train_rows,
                native.ncols(),
                nodes.reads[head],
                &name,
            )?;
            let eval = verified_read(
                data,
                &manifest["families"][1],
                layer,
                head,
                eval_rows,
                native.ncols(),
                nodes.reads[head],
                &name,
            )?;
            let mut mask = vec![0; layers.len()];
            mask[layer] = 1 << head;
            let copy = copies.compose(&mask)?;
            let predicted = copy.program.operators[index].matrix();
            let fit_native = DataWeightedSvd::new(&native, &x);
            let fit_residual = DataWeightedSvd::new(&(&native - &predicted), &x);
            let mut points = Vec::new();
            for &rank in &ranks {
                for family in [
                    HeadApproximation::NativeSvd,
                    HeadApproximation::CopyResidual,
                ] {
                    let candidate = bank.candidate(CopyResidualChoice {
                        layer,
                        head,
                        rank,
                        family,
                    })?;
                    let matrix = if family == HeadApproximation::NativeSvd {
                        candidate.program.operators[index].matrix()
                    } else {
                        &predicted
                            + &candidate
                                .program
                                .operators
                                .last()
                                .ok_or("residual")?
                                .matrix()
                    };
                    points.push(json!({"family":format!("{family:?}"),"fit":"weight_Frobenius","rank":rank,"C32_bits":structural_cost(&candidate,&mut costs)?.total(),"train":metrics(&x,&native,&matrix),"eval":metrics(&eval,&native,&matrix)}));
                    let weighted = match family {
                        HeadApproximation::NativeSvd => &fit_native,
                        HeadApproximation::CopyResidual => &fit_residual,
                    };
                    match weighted {
                        Err(error) => points.push(json!({"family":format!("{family:?}"),"fit":"training_native_inputs","rank":rank,"unresolved":error})),
                        Ok(fit) => {
                            let op = fit.operator(original, format!("{name}.input_fit"), rank)?;
                            let matrix = if family == HeadApproximation::NativeSvd { op.matrix() } else { &predicted + &op.matrix() };
                            let candidate = if family == HeadApproximation::NativeSvd { replace(&base,index,op,nodes.normed_stream,nodes.attention)? } else {
                                let mut candidate = candidate;
                                let residual = candidate.program.operators.len()-1;
                                candidate.program.operators[residual] = Arc::new(op);
                                if !matches!(&candidate.program.nodes[nodes.attention], Node::Affine {terms,..} if terms.iter().any(|&(_,i)| i==residual)) { return Err("residual not attached".into()); }
                                candidate
                            };
                            points.push(json!({"family":format!("{family:?}"),"fit":"training_native_inputs","rank":rank,"C32_bits":structural_cost(&candidate,&mut costs)?.total(),"input_singular_range":[fit.smallest_input_singular_value,fit.largest_input_singular_value],"train":metrics(&x,&native,&matrix),"eval":metrics(&eval,&native,&matrix)}));
                        }
                    }
                }
            }
            let head_result = json!({"layer":layer,"head":head,"points":points});
            serde_json::to_writer(&mut journal, &head_result).map_err(|e| e.to_string())?;
            writeln!(&mut journal).map_err(|e| e.to_string())?;
            journal.flush().map_err(|e| e.to_string())?;
            results.push(head_result);
            eprintln!(
                "finished layer {layer} head {head} at {:.1}s",
                start.elapsed().as_secs_f64()
            );
        }
    }
    save(
        output,
        &json!({"manifest":manifest,"ranks":ranks,"native_C32_bits":native_bits,"heads":results,"seconds":start.elapsed().as_secs_f64(),"claim":"proposal diagnostic only; neither full Local nor autonomous Run acceptance, no selected mechanism","fit":"SVD minimizes declared training native linear-output squared error; f32 factors before measurements; Copy scale remains original weight fit in both arms; same rank/C32 accounting; all heads reported","evaluation":"evaluation reads never enter factor fitting; separate data exports recorded in manifest; no threshold or calibration"}),
    )
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("extract") => extract(&args[1..]),
        Some("fit") => fit(&args[1..]),
        _ => Err("extract or fit required".into()),
    }
}

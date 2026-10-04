//! Truncated-SVD parameter baseline for the executable-program frontier.
//! EXPORT OUT layer=N ranks=N,... max_bank=N
//! Both MLP matrices vary independently; all hidden units and intervention places remain.
//! This is a compression baseline, not evidence of a discovered algorithm.
use gam_linalg::decompose::svd;
use gam_mpd::acceptance::{CostCache, structural_cost};
use gam_mpd::artifact::Artifact;
use gam_mpd::coder_capture::sha256;
use gam_mpd::counterfactual::Decoder;
use gam_mpd::import::import_language_model;
use gam_mpd::operator_program::{Operator, Provenance, exact_precision};
use gam_mpd::run_check::{layer_nodes, split_sites};
use ndarray::{Array1, Axis};
use serde_json::json;
use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 { return Err("EXPORT OUT layer=N ranks=N,... max_bank=N".into()); }
    let mut options = BTreeMap::new();
    for arg in &args[2..] {
        let (key, value) = arg.split_once('=').ok_or("expected key=value")?;
        if !["layer", "ranks", "max_bank"].contains(&key) || options.insert(key, value).is_some() {
            return Err(format!("unknown or duplicate option {key}"));
        }
    }
    let parse = |key| options.get(key).ok_or_else(|| format!("declare {key}"))?.parse::<usize>().map_err(|e| e.to_string());
    let (layer, max_bank) = (parse("layer")?, parse("max_bank")?);
    let ranks: Vec<usize> = options.get("ranks").ok_or("declare ranks")?.split(',').map(|s| s.parse().map_err(|e: std::num::ParseIntError| e.to_string())).collect::<Result<_, _>>()?;
    if ranks.is_empty() || ranks.contains(&0) || ranks.iter().collect::<std::collections::BTreeSet<_>>().len() != ranks.len() {
        return Err("nonempty distinct positive ranks required".into());
    }
    let declared = ranks.len().checked_mul(ranks.len()).and_then(|n| n.checked_add(1)).ok_or("bank count overflow")?;
    if declared > max_bank { return Err(format!("complete bank needs {declared} including native, max_bank={max_bank}")); }
    let (export, out) = (Path::new(&args[0]), Path::new(&args[1]));
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    if out.join("BANK.json").exists() { return Err("use a fresh output directory".into()); }
    let started = Instant::now();
    let imported = import_language_model(export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let decoder = Decoder::from_export(export)?;
    let layers = layer_nodes(&native, decoder.layers())?;
    let nodes = layers.get(layer).ok_or("layer outside decoder")?;
    let base = Artifact::native(&native)?;
    let mut sources = BTreeMap::new();
    for entry in std::fs::read_dir(export).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_file() && path.extension().is_some_and(|e| e == "f64" || e == "json") {
            sources.insert(path.file_name().ok_or("missing filename")?.to_string_lossy().to_string(), sha256(&path)?);
        }
    }
    let mut alternatives = Vec::new();
    let mut spectral = Vec::new();
    for suffix in ["c_fc", "down_proj"] {
        let name = format!("blocks.{layer}.{suffix}");
        let index = native.operators.iter().position(|op| op.name == name).ok_or_else(|| format!("missing {name}"))?;
        let original = &native.operators[index];
        let matrix = original.matrix();
        let decomposition = svd(matrix.view(), false).map_err(|e| e.to_string())?;
        let mut variants = BTreeMap::new();
        for &rank in &ranks {
            if rank > decomposition.singular_values.len() { return Err(format!("rank {rank} exceeds {name} dimensions")); }
            let kept: Vec<usize> = (0..rank).collect();
            let roots = Array1::from_iter(decomposition.singular_values.iter().take(rank).map(|s| s.sqrt()));
            let left = decomposition.u.select(Axis(1), &kept) * &roots;
            let right = decomposition.vt.select(Axis(0), &kept) * &roots.insert_axis(Axis(1));
            let precision = exact_precision(left.iter().chain(right.iter()).copied()).map_err(|e| e.to_string())?;
            let provenance = Provenance::derived(&[&original.provenance], format!("native matrix truncated SVD rank {rank}; balanced factors; f32 literal rounding at serialization"));
            let op = Operator::low_rank(name.clone(), original.rows.clone(), original.cols.clone(), left, right, precision, provenance).map_err(|e| e.to_string())?;
            variants.insert(rank, Arc::new(op));
        }
        spectral.push(json!({"operator":name,"shape":matrix.dim(),"singular_values":decomposition.singular_values.to_vec(),"numerical_band":decomposition.band}));
        alternatives.push((index, variants));
    }
    let mut bank = Vec::new();
    let mut records = Vec::new();
    let mut costs = CostCache::default();
    for &input_rank in &ranks {
        for &output_rank in &ranks {
            let label = format!("native-svd-L{layer}-in{input_rank}-out{output_rank}");
            let mut artifact = base.clone();
            artifact.program.operators[alternatives[0].0] = alternatives[0].1[&input_rank].clone();
            artifact.program.operators[alternatives[1].0] = alternatives[1].1[&output_rank].clone();
            let artifact = artifact.bind(&format!("MLP {layer}"), &[nodes.normed], nodes.mlp)?.f32_literals()?;
            let bytes = artifact.to_bytes()?;
            let decoded = Artifact::from_bytes(&bytes, &native.declarations)?;
            decoded.validate_coverage(&native)?;
            let cost = structural_cost(&decoded, &mut costs)?;
            let path = out.join(format!("{label}.artifact"));
            std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
            let digest = sha256(&path)?;
            bank.push(json!({"label":label,"artifact":path.canonicalize().map_err(|e|e.to_string())?}));
            records.push(json!({"label":label,"input_rank":input_rank,"output_rank":output_rank,"cost":cost,"sha256":digest}));
            std::fs::write(out.join("BANK.json"), serde_json::to_vec_pretty(&bank).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            eprintln!("saved {label}; {} of {}", bank.len(), declared - 1);
        }
    }
    let report = json!({"method":"truncated-SVD parameter compression baseline; no discovered-mechanism claim","scope":"independent declared ranks of both native MLP matrices; all original hidden units and places retained; no fidelity-based pruning","source_sha256":sources,"layer":layer,"ranks":ranks,"declared_including_native":declared,"generated_including_frontier_native":bank.len()+1,"spectral":spectral,"candidates":records,"seconds":started.elapsed().as_secs_f64(),"acceptance":"none performed here; exact decoded Local and autonomous counterfactual Run required"});
    std::fs::write(out.join("GENERATION.json"), serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(())
}

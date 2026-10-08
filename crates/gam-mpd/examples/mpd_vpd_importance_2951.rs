//! VPD's causal importances on a behavior's prompts (#2951, the graph oracle's search under deletion):
//! per site and subcomponent, the mean importance over every position of the prompts and their
//! counterfactuals, the mean over their target positions, and the share of positions where it is
//! above zero. Under deletion a program's unnamed parts contribute zero, as VPD's masks zero a
//! subcomponent, so these are VPD's own account of which parts a behavior's tokens use.
//!
//! EXPORT DECOMPOSITION OUT_DIR host|gpu BEHAVIOR.json...
//!
//! BEHAVIOR.json: a graph-oracle behavior (`prompts`, each with `token_ids`, `target_positions` and a
//! `counterfactual` with its own). Writes OUT_DIR/<id>.json: {"behavior", "rows", "target_rows",
//! "sites": {site: {"layer", "mean": [C], "target_mean": [C], "active": [C]}}}.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::log_to_stderr,
    explanation_battery::{Decomposition, Vpd},
    library_mdl::sequence_family,
};
use ndarray::{Array1, Axis};
use serde_json::{Value, json};
use std::path::Path;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A prompt's token ids and target positions from a behavior's JSON entry.
fn prompt(p: &Value) -> Result<(Vec<u32>, Vec<usize>), String> {
    let ids = p["token_ids"].as_array().ok_or("a prompt without token_ids")?.iter().map(|t| t.as_u64().map(|t| t as u32).ok_or("a token id")).collect::<Result<Vec<_>, _>>()?;
    let targets = p["target_positions"].as_array().ok_or("a prompt without target_positions")?.iter().map(|t| t.as_u64().map(|t| t as usize).ok_or("a target position")).collect::<Result<Vec<_>, _>>()?;
    Ok((ids, targets))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        return Err("usage: EXPORT DECOMPOSITION OUT_DIR host|gpu BEHAVIOR.json...".into());
    }
    let device = match args[4].as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("a GPU required")?,
        other => return Err(format!("device {other}: host or gpu")),
    };
    let vpd = Vpd::new(&device, Path::new(&args[1]), Decomposition::load(Path::new(&args[2]))?, 16_000_000_000)?;
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out).map_err(error)?;
    for path in &args[5..] {
        let behavior: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))?;
        let id = behavior["id"].as_str().ok_or_else(|| format!("{path}: no id"))?;
        let mut prompts = Vec::new();
        for p in behavior["prompts"].as_array().ok_or_else(|| format!("{path}: no prompts"))? {
            prompts.push(prompt(p)?);
            if p["counterfactual"].is_object() {
                prompts.push(prompt(&p["counterfactual"])?);
            }
        }
        let sites = vpd.factors.len();
        let width = |s: usize| vpd.factors[s].subcomponents();
        let mut sum: Vec<Array1<f64>> = (0..sites).map(|s| Array1::zeros(width(s))).collect();
        let mut target: Vec<Array1<f64>> = (0..sites).map(|s| Array1::zeros(width(s))).collect();
        let mut active: Vec<Array1<f64>> = (0..sites).map(|s| Array1::zeros(width(s))).collect();
        let (mut rows, mut target_rows) = (0usize, 0usize);
        for (ids, targets) in &prompts {
            let family = sequence_family(&[ids.as_slice()])?;
            for (s, g) in vpd.importances(&family)?.into_iter().enumerate() {
                sum[s] += &g.sum_axis(Axis(0));
                active[s] += &g.mapv(|x| f64::from(u8::from(x > 0.0))).sum_axis(Axis(0));
                for &t in targets.iter().filter(|&&t| t < g.nrows()) {
                    target[s] += &g.row(t);
                }
            }
            rows += ids.len();
            target_rows += targets.iter().filter(|&&t| t < ids.len()).count();
        }
        let per = |a: &Array1<f64>, n: usize| a.iter().map(|x| x / n.max(1) as f64).collect::<Vec<_>>();
        let record = json!({
            "behavior": id, "rows": rows, "target_rows": target_rows,
            "sites": (0..sites).map(|s| (vpd.factors[s].name.clone(), json!({"layer": vpd.factors[s].layer,
                "mean": per(&sum[s], rows), "target_mean": per(&target[s], target_rows), "active": per(&active[s], rows)}))).collect::<serde_json::Map<_, _>>(),
        });
        let file = out.join(format!("{id}.json"));
        std::fs::write(&file, serde_json::to_vec(&record).map_err(error)?).map_err(|e| format!("{}: {e}", file.display()))?;
        log::info!("{id}: {} prompts, {rows} positions, {target_rows} targets -> {}", prompts.len(), file.display());
    }
    Ok(())
}

fn main() {
    log_to_stderr();
    if let Err(e) = run() {
        eprintln!("mpd_vpd_importance_2951: {e}");
        std::process::exit(1);
    }
}

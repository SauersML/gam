//! The method's pipeline on a Hugging Face language model at scale (#2951): every stage timed, with
//! the host's peak resident memory and the device's memory in use after it.
//!
//! `mpd_scale_2951 MODEL_DIR TOKENS_DIR OUT_DIR [KEY=VALUE ...]`
//!
//! `MODEL_DIR` is a Hugging Face checkpoint (`config.json`, `model.safetensors`: Qwen2, Qwen3,
//! Llama), `TOKENS_DIR` holds `tokens_fit.f64` and `tokens_eval.f64` (rows of `context` token ids,
//! `bench/scale_2951/qwen3_data.py`). Stages, each one JSON object in `OUT_DIR/scale.json`
//! (rewritten after every stage):
//!
//! 1. `import`: the program read straight from the checkpoint (`import::hugging_face_language_model`,
//!    every weight widened to float64);
//! 2. `cpu_forward`: the model's logits on one evaluation sequence on the CPU;
//! 3. `device_passages`: the model's logits on `passages` evaluation sequences on the device
//!    (`core_device::passages`);
//! 4. `fit`: the explanation of the `scope` sites (`explanation::fit` on `sequences` fit sequences,
//!    its samples on the device where there is one), each site's seconds logged;
//! 5. `evaluate`: `KL(model ‖ explanation)` per token on the passages with the scope replaced, on the
//!    device (`core_device::Evaluator::replaced`).
//!
//! Keys (defaults): `layers` (every block), `scope` (a site-name prefix, `blocks.0.`),
//! `sequences` (2), `passages` (4), `context` (the token rows' width; shorter keeps a prefix), `n`
//! (1e6), `rounds` (2), `blocks` (1), `draws` (1), `device` (`f64`, `any`, `off`), `stop` (the last
//! stage to run).

use gam_mpd::core_device::{Choice, Evaluator, choose, device};
use gam_mpd::counterfactual::read_f64_matrix;
use gam_mpd::explanation::{Passage, Settings, fit, in_execution_order};
use gam_mpd::import::hugging_face_language_model;
use gam_mpd::masked::sites;
use gam_mpd::operator_program::{FamilyInputs, SequenceLayout, SlotValues};
use ndarray::Array2;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The process's peak and current resident memory (GiB), where the host reports them.
fn resident() -> (Option<f64>, Option<f64>) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else { return (None, None) };
    let field = |name: &str| {
        status.lines().find(|l| l.starts_with(name)).and_then(|l| l.split_whitespace().nth(1)).and_then(|kb| kb.parse::<f64>().ok()).map(|kb| kb / (1024.0 * 1024.0))
    };
    (field("VmHWM:"), field("VmRSS:"))
}

/// One sequence of `ids` as the program's rows.
fn sequence(ids: &[f64]) -> FamilyInputs {
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    FamilyInputs {
        rows: ids.len(),
        slots: vec![SlotValues::Tokens(ids.iter().map(|t| *t as u32).collect())],
        layout: Some(SequenceLayout { sequence: vec![0; ids.len()], position: positions }),
    }
}

fn rows(path: &Path, width: usize, count: usize, context: usize) -> Result<Vec<FamilyInputs>, String> {
    let table: Array2<f64> = read_f64_matrix(path, width)?;
    if table.nrows() < count {
        return Err(format!("{}: {} rows, {count} asked", path.display(), table.nrows()));
    }
    Ok((0..count).map(|r| sequence(&table.row(r).to_vec()[..context])).collect())
}

struct Report {
    path: PathBuf,
    stages: Vec<Value>,
    head: Value,
}

impl Report {
    fn stage(&mut self, name: &str, seconds: f64, extra: Value, device: Option<&gam_gpu::tensor::Device>) -> Result<(), String> {
        let (peak, now) = resident();
        let memory = device.and_then(|d| d.memory().ok().flatten()).map(|(free, total)| json!({"used_gib": (total - free) as f64 / (1u64 << 30) as f64, "total_gib": total as f64 / (1u64 << 30) as f64}));
        let entry = json!({"stage": name, "seconds": seconds, "host_peak_gib": peak, "host_now_gib": now, "device": memory, "detail": extra});
        eprintln!("{entry}");
        self.stages.push(entry);
        let text = serde_json::to_string_pretty(&json!({"run": self.head, "stages": self.stages})).map_err(|e| e.to_string())?;
        std::fs::write(&self.path, text).map_err(|e| format!("{}: {e}", self.path.display()))
    }
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_scale_2951 MODEL_DIR TOKENS_DIR OUT_DIR [KEY=VALUE ...]";
    if args.len() < 4 {
        return Err(usage.to_string());
    }
    let (model_dir, tokens, out) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]), PathBuf::from(&args[3]));
    let mut keys: BTreeMap<&str, &str> = BTreeMap::new();
    for pair in &args[4..] {
        let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair}: not KEY=VALUE"))?;
        keys.insert(key, value);
    }
    let count = |key: &str, default: usize| keys.get(key).map_or(Ok(default), |v| v.parse::<usize>().map_err(|e| format!("{key}: {e}")));
    let config: Value = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let all_layers = config["num_hidden_layers"].as_u64().ok_or("config.json: num_hidden_layers")? as usize;
    let layers = count("layers", all_layers)?;
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(tokens.join("tokens.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let width = meta["context"].as_u64().ok_or("tokens.json: context")? as usize;
    let context = count("context", width)?.min(width);
    let (sequences, passages) = (count("sequences", 2)?, count("passages", 4)?);
    let scope = keys.get("scope").copied().unwrap_or("blocks.0.").to_string();
    let settings = Settings {
        observations: keys.get("n").map_or(Ok(1e6), |v| v.parse::<f64>().map_err(|e| format!("n: {e}")))?,
        rounds: count("rounds", 2)?,
        blocks: count("blocks", 1)? != 0,
        draws: count("draws", 1)?,
        seed: 0x5CA1E,
    };
    let stop = keys.get("stop").copied().unwrap_or("evaluate");
    choose(Choice::parse(keys.get("device").copied().unwrap_or("f64"))?);
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut report = Report {
        path: out.join("scale.json"),
        stages: Vec::new(),
        head: json!({"model": model_dir, "layers": layers, "context": context, "sequences": sequences, "passages": passages, "scope": scope,
                     "n": settings.observations, "rounds": settings.rounds, "blocks": settings.blocks, "draws": settings.draws,
                     "threads": rayon::current_num_threads()}),
    };

    let clock = Instant::now();
    let (model, record) = hugging_face_language_model(&model_dir, 0..layers)?;
    let reals = model.real_count();
    let all_sites = in_execution_order(sites(&model));
    report.stage("import", clock.elapsed().as_secs_f64(), json!({"nodes": model.nodes.len(), "operators": model.operators.len(), "reals": reals,
        "sites": all_sites.len(), "config": record["config"]}), None)?;
    if stop == "import" {
        return Ok(());
    }

    let eval = rows(&tokens.join("tokens_eval.f64"), width, passages, context)?;
    let clock = Instant::now();
    let one = Passage::new(&model, eval[0].clone())?;
    report.stage("cpu_forward", clock.elapsed().as_secs_f64(), json!({"tokens": one.base.rows}), None)?;
    drop(one);
    if stop == "cpu_forward" {
        return Ok(());
    }

    let device = device()?;
    let clock = Instant::now();
    let targets: Vec<Passage> = match &device {
        Some(d) => gam_mpd::core_device::passages(d, &model, eval)?,
        None => eval.into_iter().map(|b| Passage::new(&model, b)).collect::<Result<_, _>>()?,
    };
    report.stage("device_passages", clock.elapsed().as_secs_f64(), json!({"passages": targets.len(), "on": device.as_ref().map(|d| d.name())}), device.as_ref())?;
    if stop == "device_passages" {
        return Ok(());
    }

    let chosen: Vec<_> = all_sites.iter().filter(|s| s.name.starts_with(&scope)).cloned().collect();
    if chosen.is_empty() {
        return Err(format!("no site starts with {scope}"));
    }
    let batches = rows(&tokens.join("tokens_fit.f64"), width, sequences, context)?;
    let clock = Instant::now();
    let mut per_site = Vec::new();
    let mut last = Instant::now();
    let explanation = fit(&model, chosen, &batches, &settings, &BTreeMap::new(), |_| Ok(None), |f| {
        per_site.push(json!({"site": f.site.name, "seconds": last.elapsed().as_secs_f64(), "columns": f.library.v.nrows(), "blocks": f.ranks.len()}));
        last = Instant::now();
        Ok(())
    })?;
    report.stage("fit", clock.elapsed().as_secs_f64(), json!({"sites": per_site}), device.as_ref())?;
    if stop == "fit" {
        return Ok(());
    }

    let members: Vec<usize> = (0..explanation.sites.len()).collect();
    let which: Vec<usize> = (0..targets.len()).collect();
    let clock = Instant::now();
    let kls = match &device {
        Some(d) => Evaluator::new(d, &model, &targets)?.replaced(&explanation, &members, &which)?,
        None => gam_mpd::explanation::replaced(&model, &explanation, &members, &targets)?,
    };
    let every: Vec<f64> = kls.iter().flat_map(|(kl, _)| kl.iter().copied()).collect();
    let mean = every.iter().sum::<f64>() / every.len().max(1) as f64;
    report.stage("evaluate", clock.elapsed().as_secs_f64(), json!({"kl_per_token_mean": mean, "tokens": every.len()}), device.as_ref())?;
    Ok(())
}

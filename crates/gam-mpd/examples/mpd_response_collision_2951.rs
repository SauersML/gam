//! Diagnostic: fixed-logit response floors for omitted native MLP interfaces.
//! EVAL_EXPORT FROZEN_SPEC OUT_JSON PARALLEL TILE_ROWS
//! Uses all rows of the existing two-passage/80-episode panel; keeps only clean
//! and four whole-MLP removals per passage. No fitting or acceptance changes.
use gam_mpd::{
    engine::sha256,
    counterfactual::{Decoder, Spec, passages},
    fixed_logit_interval::{Enclosure, Interval},
    import::import_language_model,
    response_collision::shared_prediction_obstruction,
    run_check::{LanguageRun, split_sites},
};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

const SPEC_SHA: &str = "3c05b66324dfb02e33da7eff98784a7ec432123e24fea78018b892efb4256436";
const TOKENS_SHA: &str = "9938c3b6995c4c1b9f74b7bf26b7a17a991941952de9a2e6598fec95003ca8cf";

fn evaluate() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 { return Err("EVAL_EXPORT FROZEN_SPEC OUT_JSON PARALLEL TILE_ROWS".into()); }
    let export = Path::new(&args[1]);
    let spec_path = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    let parallel: usize = args[4].parse().map_err(|e| format!("parallel: {e}"))?;
    let tile: usize = args[5].parse().map_err(|e| format!("tile rows: {e}"))?;
    if parallel == 0 || tile == 0 || tile > 512 { return Err("positive parallel/tile required; tile <=512".into()); }
    if out.exists() { return Err("refusing to overwrite existing report".into()); }
    if sha256(spec_path)? != SPEC_SHA || sha256(&export.join("tokens.f64"))? != TOKENS_SHA {
        return Err("frozen spec/token identity differs".into());
    }
    let started = Instant::now();
    let manifest: Value = serde_json::from_slice(&std::fs::read(export.join("export.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut weight_hashes = serde_json::Map::new();
    for (name, record) in manifest["files"].as_object().ok_or("missing tensor manifest")? {
        if name == "wte" || name == "final_norm.gain" || name.starts_with("blocks.") {
            let actual = sha256(&export.join(format!("{name}.f64")))?;
            if record["sha256"].as_str() != Some(actual.as_str()) { return Err(format!("native tensor identity differs: {name}")); }
            weight_hashes.insert(name.clone(), json!(actual));
        }
    }
    if weight_hashes.len() != 34 { return Err("requires all34 native tensors with verified hashes".into()); }
    let imported = import_language_model(export, 1, 1)?;
    let cfg = &imported.record["config"];
    if cfg["n_layers"] != 4 || cfg["d_mlp"] != 3072 { return Err("requires native four-layer 3072-MLP export".into()); }
    let native = split_sites(&imported.program)?;
    let decoder = Decoder::from_export(export)?;
    let full_spec = Spec::load(spec_path, &decoder)?;
    let source: Value = serde_json::from_slice(&std::fs::read(spec_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if full_spec.rows != 512 || full_spec.episodes.len() != 80 { return Err("requires full frozen panel".into()); }
    let mut indices = Vec::new();
    for passage in 0..2 {
        for layer in 0..5 {
            let id = if layer == 0 { format!("clean/{passage}") } else { format!("remove-mlp/l{}/{passage}", layer-1) };
            let matches: Vec<_> = full_spec.episodes.iter().enumerate().filter(|(_,e)| e.id == id).collect();
            let [(index, episode)] = matches[..] else { return Err(format!("missing/duplicate episode {id}")); };
            let expected = if layer == 0 { json!([]) } else { json!([{"type":"scale_input","site":(layer-1)*6+5,"row":null,"cols":[0,3072],"scale":0.0}]) };
            if source["episodes"][index]["actions"] != expected || episode.passage != passage || episode.donor.is_some() {
                return Err(format!("episode {id} is not the required clean/full-activation-removal pair"));
            }
            indices.push(index);
        }
    }
    let spec = Spec { rows: 512, episodes: indices.iter().map(|&i| full_spec.episodes[i].clone()).collect() };
    let tokens = passages(export, 512)?;
    if tokens.len() != 2 { return Err("requires exactly two declared passages".into()); }
    let run = LanguageRun::new(&decoder, &native, &spec, &tokens, parallel)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(parallel).build().map_err(|e| e.to_string())?;
    let mut sums = [Interval::point(0.0); 4];
    let mut pair_results = Vec::new();
    let mut readout_seconds = 0.0;
    let mut interval_seconds = 0.0;
    for passage in 0..2 {
        let mut pair_sums = [Interval::point(0.0); 4];
        for first in (0..512).step_by(tile) {
            let end = (first+tile).min(512);
            let timer = Instant::now();
            let clean = run.native_episode_log_probs(passage*5, first..end)?;
            readout_seconds += timer.elapsed().as_secs_f64();
            for layer in 0..4 {
                let timer = Instant::now();
                let changed = run.native_episode_log_probs(passage*5+layer+1, first..end)?;
                readout_seconds += timer.elapsed().as_secs_f64();
                let timer = Instant::now();
                let bounds: Vec<_> = pool.install(|| (0..end-first).into_par_iter().map(|row| {
                    let p = clean.row(row);
                    let q = changed.row(row);
                    let a = p.as_slice().ok_or("noncontiguous clean readout")?;
                    let b = q.as_slice().ok_or("noncontiguous changed readout")?;
                    match shared_prediction_obstruction(a,b) {
                        Enclosure::Bounded(value) => Ok(value),
                        Enclosure::Unresolved(reason) => Err(format!("unresolved interval at passage{passage}/layer{layer}/row{}: {reason:?}",first+row)),
                    }
                }).collect::<Result<Vec<_>,String>>())?;
                interval_seconds += timer.elapsed().as_secs_f64();
                for value in bounds { pair_sums[layer] = pair_sums[layer].add(value); }
            }
            println!("completed passage={passage} rows={first}:{end} elapsed={:.3}s",started.elapsed().as_secs_f64());
        }
        for layer in 0..4 {
            sums[layer] = sums[layer].add(pair_sums[layer]);
            let stat = pair_sums[layer].div_positive(Interval::point(512.));
            pair_results.push(json!({"passage":passage,"layer":layer,"statistic_interval":[stat.lo,stat.hi],"conditional_KL_lower_nats_per_token":stat.lo}));
        }
    }
    let groups: Vec<_> = sums.iter().enumerate().map(|(layer,sum)| {
        let stat = sum.div_positive(Interval::point(1024.));
        json!({"layer":layer,"clean_group":"clean","intervention_group":format!("remove-mlp/l{layer}"),"matched_rows":1024,"statistic_interval":[stat.lo,stat.hi],"conditional_worst_group_KL_lower_nats_per_token":stat.lo})
    }).collect();
    let result = json!({
        "scope":"diagnostic only; fixed binary64 CPU native log-probability arrays treated as logits and renormalized in exact real arithmetic; no acceptance change",
        "premise":"candidate makes exactly the same prediction for clean and this removal on every matched passage/token; e.g. wholly unheld activation interface with no other episode-dependent path",
        "theorem":"max of two matched group mean KLs >= mean TV(native_clean,native_changed)^2/2; only lower endpoint bounds achievable KL",
        "exclusions":"no certificate of neural forward/readout rounding, artifact interface coverage, or existing exp/log acceptance arithmetic",
        "arithmetic":"analytic exp enclosure, directed binary64 interval arithmetic under IEEE round-to-nearest and gradual underflow",
        "spec_sha256":SPEC_SHA,"tokens_sha256":TOKENS_SHA,"source_episode_indices":indices,"rows":512,"passages":2,
        "config":cfg,"export_json_sha256":sha256(&export.join("export.json"))?,"verified_native_weight_sha256":weight_hashes,
        "binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,
        "parallel":parallel,"tile_rows":tile,"maximum_live_pair_logit_bytes":2*tile*decoder.embedding().nrows()*8,
        "groups":groups,"pairs":pair_results,"seconds":{"readout_and_teacher_initialization":readout_seconds,"intervals":interval_seconds,"total":started.elapsed().as_secs_f64()}
    });
    std::fs::write(out,serde_json::to_vec_pretty(&result).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}
fn main() -> Result<(), String> { evaluate() }

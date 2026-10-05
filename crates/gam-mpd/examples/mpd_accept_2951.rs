//! The one acceptance path on a language model (#2951, `gam_mpd::acceptance`): minimise the
//! explanation's structural description `C(P)` subject to its native local disagreement
//! `D_local(P) ≤ δ` and its composed counterfactual disagreement `D_run(P) ≤ ε`, over a grid of
//! `(δ, ε)`.
//!
//! `mpd_accept_2951 EXPORT_DIR SPEC.json OUT_DIR [KEY=VALUE ...]`
//!
//! `EXPORT_DIR` is the model (`gam_mpd::import::import_language_model`'s layout, with the passages
//! `SPEC.json`'s episodes run on); `SPEC.json` the frozen episode list `gam_mpd::counterfactual`
//! reads. The native program is split so every site writes its own node
//! (`gam_mpd::run_check::split_sites`).
//!
//! Keys (defaults): `mode=frontier` (or `check`: the native model as its own explanation, whose
//! `D_run` must be zero up to rounding, a check of the mapping of episodes onto the program; or
//! `replay`: decode `artifact=FILE` from its bytes and the declarations alone and score it),
//! `local_export=EXPORT_DIR` and `local=4` (the sequences `D_local` is measured on, their first
//! `context=512` tokens), `ascent=0` (exact evaluations per counterexample-ascent step; `0`
//! declares no ascent), `accounts=DIR` (MLP accounts in `mpd_mlp_functions_2951`'s layout, proposed
//! for their layers' MLPs), `heads=` (layers whose attention heads get `gam_mpd::rules`' copy and
//! match rules proposed, e.g. `2,3`), `deltas=0.05,0.1,0.2`, `epsilons=0.01,0.03,0.1` (the tolerance grid),
//! `certifications=64`, `rounds=16`, `parallel=8` (episodes at once), `groups=` (comma-separated
//! episode groups `D_run` is declared over; empty for all), `batch=1024` (rows executed at once).
//!
//! `OUT_DIR/report.json` gets, per tolerance pair, the accepted explanation's `C(P)` (literals,
//! structure, ties), `D_local` per block, `D_run` per group, its activity listing (an execution
//! cost, reported apart from `C`) and every tried candidate with its outcome; each accepted
//! artifact's bytes are `OUT_DIR/artifact.{i}.{j}.bin`.

use gam_mpd::acceptance::{
    Budget, Constraint, CostCache, Local, Proposer, RunCheck, RunMeasure, Searched, assess, execution_cost, frontier, structural_cost,
};
use gam_mpd::artifact::Artifact;
use gam_mpd::counterfactual::{Decoder, Spec, passages};
use gam_mpd::import::import_language_model;
use gam_mpd::operator_program::{Declarations, Domain, FamilyInputs, SequenceLayout, Slot, SlotValues};
use gam_mpd::proposals::{AccountProposer, HeadRules};
use gam_mpd::run_check::{LanguageRun, layer_nodes, split_sites};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn floats(text: &str) -> Result<Vec<f64>, String> {
    text.split(',').filter(|s| !s.is_empty()).map(|s| s.parse::<f64>().map_err(|e| format!("{s}: {e}"))).collect()
}

/// The sequences' first `context` tokens as a per-position family.
fn family(sequences: &[Vec<u32>], count: usize, context: usize) -> FamilyInputs {
    let (mut tokens, mut sequence, mut position) = (Vec::new(), Vec::new(), Vec::new());
    for (s, row) in sequences.iter().take(count).enumerate() {
        for (p, t) in row.iter().take(context).enumerate() {
            tokens.push(*t);
            sequence.push(s as u32);
            position.push(p as u32);
        }
    }
    FamilyInputs { rows: tokens.len(), slots: vec![SlotValues::Tokens(tokens)], layout: Some(SequenceLayout { sequence, position }) }
}

fn run_summary(measure: &RunMeasure) -> Value {
    let groups: Vec<Value> = measure
        .groups
        .iter()
        .map(|(g, kl, error, effect, n)| {
            let unheld: usize = measure.episodes.iter().filter(|e| &e.group == g).map(|e| e.unheld).sum();
            let top1 = measure.episodes.iter().filter(|e| &e.group == g).map(|e| e.top1_agree).sum::<f64>() / *n as f64;
            json!({"group": g, "kl_per_token": kl, "rounding": error, "native_effect": effect, "top1_agree": top1, "episodes": n, "unheld_places": unheld})
        })
        .collect();
    json!({"worst": measure.worst().map(|w| json!({"group": w.0, "kl_per_token": w.1})), "groups": groups})
}

fn searched_report(searched: &Searched, local: &Local<'_>, baseline_family: &FamilyInputs, batch: usize) -> Result<Value, String> {
    let assessment = &searched.assessment;
    let cost = assessment.cost;
    let listing = execution_cost(&searched.artifact, baseline_family, batch)?;
    Ok(json!({
        "cost": {"total_bits": cost.total(), "literals": cost.literals, "literal_bits": cost.literal_bits(), "structure_bits": cost.structure_bits, "binding_bits": cost.binding_bits},
        "local": {"verdict": format!("{:?}", assessment.local.verdict()), "blocks": assessment.local_measure.blocks, "tested_rows": assessment.local_measure.rows,
                  "declared_rows": assessment.local_measure.family_rows, "counterexamples": assessment.local_measure.counterexamples},
        "run": {"verdict": format!("{:?}", assessment.run.verdict()), "summary": run_summary(&assessment.run_measure)},
        "execution_cost": listing,
        "blocks": searched.artifact.blocks.iter().map(|b| b.name.clone()).collect::<Vec<_>>(),
        "stop": format!("{:?}", searched.stop),
        "steps": searched.steps,
        "local_family_rows": local.family.rows,
    }))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_accept_2951 EXPORT_DIR SPEC.json OUT_DIR [KEY=VALUE ...]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let spec_path = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let keys: BTreeMap<String, String> = args[4..].iter().filter_map(|a| a.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let key = |k: &str, default: &str| keys.get(k).cloned().unwrap_or_else(|| default.to_string());
    let number = |k: &str, default: usize| -> Result<usize, String> { key(k, &default.to_string()).parse::<usize>().map_err(|e| format!("{k}: {e}")) };
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let started = Instant::now();
    let mode = key("mode", "frontier");
    let context = number("context", 512)?;
    let batch = number("batch", 1024)?;
    let decoder = Decoder::from_export(&export)?;
    let mut spec = Spec::load(&spec_path, &decoder)?;
    let groups: Vec<String> = key("groups", "").split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
    if !groups.is_empty() {
        spec.episodes.retain(|e| groups.contains(&e.group));
    }
    let run_passages = passages(&export, spec.rows)?;
    let imported = import_language_model(&export, 1, 1)?;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, decoder.layers())?;
    let run = LanguageRun::new(&decoder, &native, &spec, &run_passages, number("parallel", 8)?)?;
    let local_export = PathBuf::from(key("local_export", &export.display().to_string()));
    let local_sequences = passages(&local_export, context)?;
    let local_family = family(&local_sequences, number("local", 4)?, context);
    let ascent = match number("ascent", 0)? {
        0 => None,
        evaluations => {
            let vocab = native.declarations.domains[0].size as u32;
            Some(gam_mpd::acceptance::Ascent { domain: vec![gam_mpd::acceptance::SlotDomain::Tokens((0..vocab).collect())], pool: Vec::new(), evaluations })
        }
    };
    let local = Local::new(&native, local_family.clone(), ascent, batch);
    eprintln!("{} episodes, {} local rows, {:.0}s to load", spec.episodes.len(), local_family.rows, started.elapsed().as_secs_f64());
    let constraint_of = |d: f64, e: f64| Constraint { local: d, run: e };
    let report = match mode.as_str() {
        "check" => {
            let artifact = Artifact::native(&native)?;
            let timer = Instant::now();
            let bytes = artifact.to_bytes()?;
            let decoded = Artifact::from_bytes(&bytes, &native.declarations)?;
            let round_trip = timer.elapsed().as_secs_f64();
            let mut cache = CostCache::default();
            let cost = structural_cost(&decoded, &mut cache)?;
            let timer = Instant::now();
            let measure = RunMeasure::of(run.episodes(&decoded)?);
            let worst_episode = measure.episodes.iter().max_by(|a, b| a.kl.total_cmp(&b.kl)).cloned();
            json!({"mode": "check", "bytes": bytes.len(), "round_trip_seconds": round_trip, "cost_bits": cost.total(), "literals": cost.literals,
                   "run": run_summary(&measure), "worst_episode": worst_episode, "run_seconds": timer.elapsed().as_secs_f64()})
        }
        "replay" => {
            let path = PathBuf::from(keys.get("artifact").ok_or("replay needs artifact=FILE")?);
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            // The declarations the decoder knows: the vocabulary and one token slot, nothing else.
            let declarations = Declarations { parameters: 0, domains: vec![Domain { size: decoder_vocab(&export)? }], slots: vec![Slot::Token { domain: 0 }] };
            let decoded = Artifact::from_bytes(&bytes, &declarations)?;
            let constraint = constraint_of(floats(&key("deltas", "0.1"))?[0], floats(&key("epsilons", "0.03"))?[0]);
            let assessment = assess(&local, &run, &decoded, constraint, &mut CostCache::default())?;
            json!({"mode": "replay", "artifact": path.display().to_string(), "bytes": bytes.len(), "blocks": decoded.blocks.iter().map(|b| b.name.clone()).collect::<Vec<_>>(),
                   "cost_bits": assessment.cost.total(), "literals": assessment.cost.literals, "local": assessment.local_measure.blocks,
                   "local_verdict": format!("{:?}", assessment.local.verdict()), "run_verdict": format!("{:?}", assessment.run.verdict()),
                   "run": run_summary(&assessment.run_measure)})
        }
        "frontier" => {
            let mut proposers: Vec<Box<dyn Proposer>> = Vec::new();
            if let Some(dir) = keys.get("accounts") {
                let (d_in, d_out) = (native.node_interface(layers[0].normed).map_err(|e| e.to_string())?.width(), native.node_interface(layers[0].mlp).map_err(|e| e.to_string())?.width());
                let accounts = AccountProposer::load(Path::new(dir), layers.clone(), d_in, d_out)?;
                eprintln!("{} MLP accounts from {dir}", accounts.accounts.len());
                proposers.push(Box::new(accounts));
            }
            let start = Artifact::native(&native)?;
            let targets: Vec<usize> = key("heads", "").split(',').filter(|s| !s.is_empty()).map(|s| s.parse::<usize>().map_err(|e| format!("heads: {e}"))).collect::<Result<_, _>>()?;
            if !targets.is_empty() {
                let group = layers[0].queries.len() / layers[0].keys.len().max(1);
                let timer = Instant::now();
                proposers.push(Box::new(HeadRules::new(&start, layers.clone(), targets.clone(), group)?));
                eprintln!("head rules for layers {targets:?}: content planes found in {:.0}s", timer.elapsed().as_secs_f64());
            }
            let proposers: Vec<&dyn Proposer> = proposers.iter().map(|p| p.as_ref()).collect();
            let budget = Budget { certifications: number("certifications", 64)?, rounds: number("rounds", 16)? };
            let (deltas, epsilons) = (floats(&key("deltas", "0.05,0.1,0.2"))?, floats(&key("epsilons", "0.01,0.03,0.1"))?);
            let (points, searches) = frontier(&local, &run, &proposers, &start, &deltas, &epsilons, budget)?;
            let mut details = Vec::new();
            for (point, searched) in points.iter().zip(&searches) {
                let i = deltas.iter().position(|d| *d == point.local_tolerance).unwrap_or(0);
                let j = epsilons.iter().position(|e| *e == point.run_tolerance).unwrap_or(0);
                let path = out.join(format!("artifact.{i}.{j}.bin"));
                std::fs::write(&path, searched.artifact.to_bytes()?).map_err(|e| format!("{}: {e}", path.display()))?;
                details.push(json!({"point": point, "artifact": path.display().to_string(), "searched": searched_report(searched, &local, &local_family, batch)?}));
            }
            json!({"mode": "frontier", "frontier": points, "details": details})
        }
        other => return Err(format!("unknown mode {other}: {usage}")),
    };
    let report = json!({"export": export.display().to_string(), "spec": spec_path.display().to_string(), "episodes": spec.episodes.len(),
                        "local_rows": local_family.rows, "report": report, "seconds": started.elapsed().as_secs_f64()});
    let path = out.join("report.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("{}", serde_json::to_string_pretty(&report["report"]["run"]).unwrap_or_default());
    Ok(())
}

/// The vocabulary the export declares.
fn decoder_vocab(export: &Path) -> Result<usize, String> {
    let text = std::fs::read_to_string(export.join("export.json")).map_err(|e| format!("{}: {e}", export.display()))?;
    let record: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    record["config"]["vocab"].as_u64().map(|v| v as usize).ok_or_else(|| "export config has no vocab".to_string())
}

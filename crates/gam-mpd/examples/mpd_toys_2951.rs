//! The known-mechanism toy suite (#2951, `gam_mpd::toys`): every case explained by the engine,
//! scored on its counterfactual questions beside the baselines that know no internals, and checked
//! for its known mechanism.
//!
//! `mpd_toys_2951 ROOT [CASE ...] [KEY=VALUE ...]`: `ROOT` holds the cases (`bench/toys_2951`
//! writes `~/mpd-data/toys_2951`); no case names every case. Per case:
//!
//! 1. the native model's response to every question, against the measured outcome (dev and
//!    sealed): the reference forward and this crate's interventions must agree;
//! 2. the explanation of every site (`gam_mpd::explanation::fit`, on the case's samples in batches
//!    of `batch` prompts), written to `<case>/engine/library` and read back when fitted alike;
//! 3. its prediction of every question (its own program under the same edits) to
//!    `<case>/engine/predictions.json`;
//! 4. the scores of the explanation, `null` and `transcript` on the dev and held-out splits, the
//!    dev split per tag, and the mechanism checks, to `<case>/engine/report.json`.
//!
//! `mpd_toys_2951 score ROOT CASE PREDICTIONS` scores any explainer's predictions (a JSON object of
//! question id → logits, readouts × classes) the same way; `mpd_toys_2951 chive ROOT CASE
//! PREDICTIONS...` scores CHIVE claim predictions (claim id → P(true), `bench/toys_2951/chive_vpd.py`)
//! by AUROC on the dev and held-out claims.
//!
//! Keys (defaults): `n` (1e6), `rounds` (50), `blocks` (1), `draws` (4), `batch` (256), `seed`,
//! `out` (`<case>/engine`: outputs go to `<out>/<case>/` instead).

use gam_mpd::explanation::{Explanation, Fitted, Settings, fit, in_execution_order};
use gam_mpd::masked::{Library, Site, matrix, sites};
use gam_mpd::operator_program::OperatorProgram;
use gam_mpd::toys::{Case, Explained, Native, Question, Scores, cases, chive, chive_claims, kl_rows, mechanism, score, sealed};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn write_f64(path: &Path, m: &Array2<f64>) -> Result<(), String> {
    let bytes: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    gam_mpd::counterfactual::read_f64_matrix(path, cols)
}

fn save(dir: &Path, fitted: &Fitted, with: &Value) -> Result<(), String> {
    let name = &fitted.site.name;
    for (side, m) in [("v", &fitted.library.v), ("u", &fitted.library.u), ("fisher", &fitted.fisher), ("moment", &fitted.second_moment)] {
        write_f64(&dir.join(format!("{name}.{side}.f64")), m)?;
    }
    let record = json!({"site": name, "ranks": fitted.ranks, "bits": fitted.bits, "fitted_with": with});
    std::fs::write(dir.join(format!("{name}.json")), record.to_string()).map_err(|e| e.to_string())
}

fn load(dir: &Path, model: &OperatorProgram, site: &Site, with: &Value, observations: f64) -> Result<Option<Fitted>, String> {
    let name = &site.name;
    let Ok(text) = std::fs::read_to_string(dir.join(format!("{name}.json"))) else { return Ok(None) };
    let record: Value = serde_json::from_str(&text).map_err(|e| format!("{name}.json: {e}"))?;
    if &record["fitted_with"] != with {
        return Ok(None);
    }
    let w = matrix(model, site)?;
    let (d_out, d_in) = w.dim();
    let read = |side: &str, cols: usize| read_f64(&dir.join(format!("{name}.{side}.f64")), cols);
    let library = Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) };
    let ranks: Vec<usize> = serde_json::from_value(record["ranks"].clone()).map_err(|e| format!("{name} ranks: {e}"))?;
    let bits: Vec<f64> = serde_json::from_value(record["bits"].clone()).map_err(|e| format!("{name} bits: {e}"))?;
    Ok(Some(Fitted::new(site.clone(), w, (library, ranks, bits), (read("fisher", d_out)?, read("moment", d_in)?), observations)?))
}

struct Run {
    settings: Settings,
    batch: usize,
    /// Where each case's outputs go (`<out>/<case>/`), else `<case>/engine/`.
    out: Option<PathBuf>,
}

impl Run {
    fn parse(pairs: &[String]) -> Result<Self, String> {
        let mut run = Self { settings: Settings { observations: 1e6, rounds: 50, blocks: true, draws: 4, seed: 0x7051 }, batch: 256, out: None };
        for pair in pairs {
            let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair}: not KEY=VALUE"))?;
            let count = || value.parse::<usize>().map_err(|e| format!("{key}: {e}"));
            match key {
                "n" => run.settings.observations = value.parse().map_err(|e| format!("n: {e}"))?,
                "rounds" => run.settings.rounds = count()?,
                "blocks" => run.settings.blocks = count()? != 0,
                "draws" => run.settings.draws = count()?,
                "batch" => run.batch = count()?.max(1),
                "seed" => run.settings.seed = count()? as u64,
                "out" => run.out = Some(PathBuf::from(value)),
                other => return Err(format!("unknown key {other}")),
            }
        }
        Ok(run)
    }

    fn fitted_with(&self) -> Value {
        let s = &self.settings;
        json!({"n": s.observations, "rounds": s.rounds, "blocks": s.blocks, "draws": s.draws, "seed": s.seed, "batch": self.batch})
    }
}

fn logits_json(m: &Array2<f64>) -> Value {
    json!(m.outer_iter().map(|r| r.to_vec()).collect::<Vec<_>>())
}

/// The scores of every predictor on the dev and held-out splits, and per tag on the dev split.
fn scores(case: &Case, outcomes: &BTreeMap<String, Array2<f64>>, predictors: &[(&str, &BTreeMap<String, Array2<f64>>)]) -> Result<Value, String> {
    let dev: Vec<&Question> = case.questions.iter().filter(|q| !q.held_out).collect();
    let held: Vec<&Question> = case.questions.iter().filter(|q| q.held_out).collect();
    let mut report = serde_json::Map::new();
    for (name, predictions) in predictors {
        let mut tags = BTreeMap::<&str, Vec<&Question>>::new();
        for q in &dev {
            tags.entry(q.tag.as_str()).or_default().push(q);
        }
        let per_tag: BTreeMap<&str, Scores> = tags.into_iter().map(|(t, qs)| Ok((t, score(&qs, outcomes, predictions)?))).collect::<Result<_, String>>()?;
        let internal: Vec<&Question> = held.iter().copied().filter(|q| q.internal()).collect();
        let prompt: Vec<&Question> = held.iter().copied().filter(|q| !q.internal()).collect();
        report.insert(
            name.to_string(),
            json!({"dev": score(&dev, outcomes, predictions)?, "held_out": score(&held, outcomes, predictions)?,
                   "chive_dev": chive(&dev, outcomes, predictions)?, "chive_held_out": chive(&held, outcomes, predictions)?,
                   "held_out_internal": score(&internal, outcomes, predictions)?, "held_out_prompt": score(&prompt, outcomes, predictions)?,
                   "dev_per_tag": per_tag}),
        );
    }
    Ok(Value::Object(report))
}

fn read_predictions(path: &Path) -> Result<BTreeMap<String, Array2<f64>>, String> {
    let record: Value = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| e.to_string())?;
    record
        .as_object()
        .ok_or("predictions: not an object")?
        .iter()
        .map(|(id, logits)| {
            let rows: Vec<Vec<f64>> = serde_json::from_value(logits.clone()).map_err(|e| format!("{id}: {e}"))?;
            let cols = rows.first().map_or(0, Vec::len);
            Ok((id.clone(), Array2::from_shape_vec((rows.len(), cols), rows.concat()).map_err(|e| format!("{id}: {e}"))?))
        })
        .collect()
}

/// The baselines' predictions: `null` (the clean outcome) and `transcript`.
fn baselines(case: &Case) -> Result<(BTreeMap<String, Array2<f64>>, BTreeMap<String, Array2<f64>>), String> {
    let outputs = case.native_logits(&case.transcript)?;
    let null = case.questions.iter().map(|q| (q.id.clone(), q.clean.clone())).collect();
    let transcript = case.questions.iter().map(|q| (q.id.clone(), case.transcript_prediction(&outputs, q))).collect();
    Ok((null, transcript))
}

fn run_case(root: &Path, dir: &Path, run: &Run) -> Result<Value, String> {
    let started = Instant::now();
    let case = Case::load(dir)?;
    let model = &case.imported.program;
    let out = run.out.as_ref().map_or_else(|| dir.join("engine"), |o| o.join(&case.name));
    let library_dir = out.join("library");
    std::fs::create_dir_all(&library_dir).map_err(|e| format!("{}: {e}", library_dir.display()))?;
    let outcomes = sealed(root, &case)?;

    // 1. The native model under every question's edits, against the measured outcome.
    let native = Native(model);
    let native_kl: Vec<f64> = case
        .questions
        .par_iter()
        .map(|q| {
            let ours = case.predict(&native, q)?;
            let measured = outcomes.get(&q.id).ok_or_else(|| format!("{}: no outcome", q.id))?;
            Ok(kl_rows(measured, &ours).iter().copied().fold(0.0, f64::max))
        })
        .collect::<Result<_, String>>()?;
    let native_worst = native_kl.iter().copied().fold(0.0, f64::max);
    eprintln!("{}: native interventions against the measured outcomes, worst KL {native_worst:.2e}", case.name);

    // 2. The explanation.
    let clock = Instant::now();
    let with = run.fitted_with();
    let chosen = in_execution_order(sites(model));
    let family = &case.imported.contract.family;
    // Batches of whole prompts: `batch` prompts' rows each.
    let per_prompt = if case.imported.kind == "transformer_rows" { case.transcript.ncols() } else { 1 };
    let prompts = family.rows / per_prompt.max(1);
    let batches: Vec<_> = (0..prompts.div_ceil(run.batch))
        .map(|b| family.select(&(b * run.batch * per_prompt..((b + 1) * run.batch).min(prompts) * per_prompt).collect::<Vec<_>>()))
        .collect();
    let explanation: Explanation = fit(model, chosen, &batches, &run.settings, &BTreeMap::new(), |site| load(&library_dir, model, site, &with, run.settings.observations), |f| {
        save(&library_dir, f, &with)
    })?;
    let fit_seconds = clock.elapsed().as_secs_f64();
    let explained = Explained::new(model, &explanation)?;

    // 3. Its predictions.
    let ours: BTreeMap<String, Array2<f64>> = case.questions.par_iter().map(|q| Ok((q.id.clone(), case.predict(&explained, q)?))).collect::<Result<_, String>>()?;
    let file: serde_json::Map<String, Value> = ours.iter().map(|(id, m)| (id.clone(), logits_json(m))).collect();
    std::fs::write(out.join("predictions.json"), Value::Object(file).to_string()).map_err(|e| e.to_string())?;
    let clean_kl: Vec<f64> = case
        .questions
        .par_iter()
        .map(|q| {
            let theirs = case.predict(&explained, &Question { edits: Vec::new(), ..q.clone() })?;
            Ok(kl_rows(&q.clean, &theirs).mean().unwrap_or(0.0))
        })
        .collect::<Result<_, String>>()?;

    // 4. Scores and the mechanism.
    let (null, transcript) = baselines(&case)?;
    let scored = scores(&case, &outcomes, &[("explanation", &ours), ("null", &null), ("transcript", &transcript)])?;
    let found = mechanism(&case, &explained)?;
    let held = |name: &str, key: &str| scored[name]["held_out"][key].as_f64().unwrap_or(f64::NAN);
    let beats = held("explanation", "mean_kl") < held("null", "mean_kl").min(held("transcript", "mean_kl"))
        && held("explanation", "argmax") >= held("null", "argmax").max(held("transcript", "argmax"));
    let library: BTreeMap<String, Value> = explanation
        .sites
        .iter()
        .map(|f| (f.site.name.clone(), json!({"columns": f.library.v.nrows(), "blocks": f.ranks.len(), "library_bits": f.bits.iter().sum::<f64>()})))
        .collect();
    let report = json!({
        "case": case.name, "mechanism_described": case.truth.mechanism, "fitted_with": with,
        "native_check_worst_kl": native_worst,
        "explanation": {"sites": library, "clean_kl_mean": clean_kl.iter().sum::<f64>() / clean_kl.len().max(1) as f64, "fit_seconds": fit_seconds},
        "scores": scored,
        "mechanism": found,
        "pass": {"counterfactual": beats, "mechanism": found.recovered},
        "seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(out.join("report.json"), serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(report)
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "mpd_toys_2951 ROOT [CASE ...] [KEY=VALUE ...] | mpd_toys_2951 score ROOT CASE PREDICTIONS | mpd_toys_2951 chive ROOT CASE PREDICTIONS...";
    if args.first().map(String::as_str) == Some("chive") {
        let (root, case) = (PathBuf::from(args.get(1).ok_or(usage)?), args.get(2).ok_or(usage)?);
        for path in &args[3..] {
            let predictions: BTreeMap<String, f64> =
                serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))?;
            let (dev, held) = chive_claims(&root, case, &predictions)?;
            eprintln!("{path}: AUROC dev {:.3} ({}+{} claims), held out {:.3} ({}+{} claims)", dev.auroc, dev.true_claims, dev.false_claims, held.auroc, held.true_claims, held.false_claims);
        }
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("score") {
        let [root, name, path] = [1, 2, 3].map(|i| args.get(i).cloned().ok_or(usage));
        let root = PathBuf::from(root?);
        let case = Case::load(&root.join(name?))?;
        let outcomes = sealed(&root, &case)?;
        let scored = scores(&case, &outcomes, &[("predictions", &read_predictions(Path::new(&path?))?)])?;
        eprintln!("{}", serde_json::to_string_pretty(&scored).map_err(|e| e.to_string())?);
        return Ok(());
    }
    let root = PathBuf::from(args.first().ok_or(usage)?);
    let (pairs, names): (Vec<String>, Vec<String>) = args[1..].iter().cloned().partition(|a| a.contains('='));
    let run = Run::parse(&pairs)?;
    let dirs: Vec<PathBuf> = if names.is_empty() { cases(&root)? } else { names.iter().map(|n| root.join(n)).collect() };
    let mut table = Vec::new();
    for dir in &dirs {
        // A case the engine cannot explain is a failed case, not the end of the suite.
        let report = match run_case(&root, dir, &run) {
            Ok(report) => report,
            Err(e) => {
                let line = format!("{:<15} engine error: {e}", dir.file_name().and_then(|n| n.to_str()).unwrap_or(""));
                eprintln!("{line}");
                table.push(line);
                continue;
            }
        };
        let s = &report["scores"];
        let line = format!(
            "{:<15} held-out KL ours {:.3} null {:.3} transcript {:.3} | argmax ours {:.2} null {:.2} transcript {:.2} | changed-argmax ours {:.2} | CHIVE AUROC ours {:.2} transcript {:.2} ({}+{} claims) | clean KL {:.1e} | counterfactual {} mechanism {}",
            report["case"].as_str().unwrap_or(""),
            s["explanation"]["held_out"]["mean_kl"].as_f64().unwrap_or(f64::NAN),
            s["null"]["held_out"]["mean_kl"].as_f64().unwrap_or(f64::NAN),
            s["transcript"]["held_out"]["mean_kl"].as_f64().unwrap_or(f64::NAN),
            s["explanation"]["held_out"]["argmax"].as_f64().unwrap_or(f64::NAN),
            s["null"]["held_out"]["argmax"].as_f64().unwrap_or(f64::NAN),
            s["transcript"]["held_out"]["argmax"].as_f64().unwrap_or(f64::NAN),
            s["explanation"]["held_out"]["changed_argmax"].as_f64().unwrap_or(f64::NAN),
            s["explanation"]["chive_held_out"]["auroc"].as_f64().unwrap_or(f64::NAN),
            s["transcript"]["chive_held_out"]["auroc"].as_f64().unwrap_or(f64::NAN),
            s["explanation"]["chive_held_out"]["true_claims"],
            s["explanation"]["chive_held_out"]["false_claims"],
            report["explanation"]["clean_kl_mean"].as_f64().unwrap_or(f64::NAN),
            if report["pass"]["counterfactual"].as_bool() == Some(true) { "PASS" } else { "fail" },
            if report["pass"]["mechanism"].as_bool() == Some(true) { "PASS" } else { "fail" },
        );
        eprintln!("{line}");
        table.push(line);
    }
    eprintln!("\n{}", table.join("\n"));
    Ok(())
}

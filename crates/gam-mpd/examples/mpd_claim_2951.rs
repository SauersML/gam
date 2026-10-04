//! The site-switch claim's exact stage on a language model (#2951): every subset of the layer
//! switches (a layer's sites replaced or native together), and every subset of each layer's sites
//! with the other layers native, each subset run on every passage. The same corners
//! `bench/vpd_2951/vpd_stepA_honesty.py claim` enumerates in VPD's harness (keys `layer_switches`,
//! `sites_layer_{l}`), here for any `gam_mpd::explanation::Replacement`: an explanation run by its
//! own rule on whatever its program computes, or given per-row sets.
//!
//! `mpd_claim_2951 FRONTIER OUT.json {ours:LIBRARY_DIR|vpd:VPD_LIBRARY:VPD_SETS} [passages] [context] [stages]`
//!
//! `stages` (default all) is a comma list of `layers` and layer numbers `L` (`sites_layer_L`), so
//! the stages can run as separate jobs.
//!
//! `FRONTIER` is the engine export of the passages (`~/mpd-data/engine/vpd4l_frontier32`, 32
//! passages of 512). `ours:` reads a fitted explanation (`mpd_e2e_2951`'s `OUT_DIR/library`: per
//! site `{site}.{v,u,fisher,moment}.f64` and `{site}.json`); `vpd:` VPD's subcomponents
//! (`{site}.{v,u}.f64`, `~/mpd-data/pieces/vpd4l_library`) with its published sets (`sites.txt`,
//! `indptr.i64`, `indices.i64`, `~/mpd-data/pieces/vpd4l_sets`, whose rows are the frontier's
//! passages in order). Per key, `OUT.json` gets the KL per word with every enumerated site replaced,
//! each passage's worst subset's mean KL (the exact worst corner) and that subset, and each word's
//! largest KL over the subsets.

use gam_mpd::counterfactual::read_f64_matrix;
use gam_mpd::explanation::{Explanation, Fitted, Given, Passage, Replacement, in_execution_order};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, Site, kl_score_only, matrix, sites};
use gam_mpd::operator_program::OperatorProgram;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn read_i64(path: &Path) -> Result<Vec<i64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("eight bytes"))).collect())
}

/// The fitted sites in `dir`, in execution order.
fn ours(model: &OperatorProgram, dir: &Path) -> Result<Explanation, String> {
    let mut fitted = Vec::new();
    let mut observations = None;
    for site in in_execution_order(sites(model)) {
        let name = site.name.clone();
        let Ok(text) = std::fs::read_to_string(dir.join(format!("{name}.json"))) else { continue };
        let record: Value = serde_json::from_str(&text).map_err(|e| format!("{name}.json: {e}"))?;
        let n = record["fitted_with"]["n"].as_f64().ok_or_else(|| format!("{name}.json: no fitted_with.n"))?;
        if observations.is_some_and(|o| o != n) {
            return Err(format!("{name}: fitted at n = {n}, the others at {observations:?}"));
        }
        observations = Some(n);
        let w = matrix(model, &site)?;
        let (d_out, d_in) = w.dim();
        let read = |side: &str, cols: usize| read_f64_matrix(&dir.join(format!("{name}.{side}.f64")), cols);
        let library = Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) };
        let ranks: Vec<usize> = serde_json::from_value(record["ranks"].clone()).map_err(|e| format!("{name} ranks: {e}"))?;
        let bits: Vec<f64> = serde_json::from_value(record["bits"].clone()).map_err(|e| format!("{name} bits: {e}"))?;
        fitted.push(Fitted::new(site, w, (library, ranks, bits), (read("fisher", d_out)?, read("moment", d_in)?), n)?);
    }
    Ok(Explanation { observations: observations.ok_or_else(|| format!("{}: no fitted site", dir.display()))?, sites: fitted })
}

/// VPD's subcomponents on every site its sets name, in execution order, with its published sets of
/// the first `passages` sequences.
fn vpd(model: &OperatorProgram, library: &Path, sets: &Path, passages: usize, context: usize) -> Result<Given, String> {
    let listed = std::fs::read_to_string(sets.join("sites.txt")).map_err(|e| format!("{}: {e}", sets.display()))?;
    let mut offsets = BTreeMap::new();
    let mut total = 0usize;
    for line in listed.lines() {
        let (name, count) = line.split_once(' ').ok_or_else(|| format!("sites.txt: {line}"))?;
        let count = count.parse::<usize>().map_err(|e| format!("sites.txt: {e}"))?;
        offsets.insert(name.to_string(), (total, count));
        total += count;
    }
    let indptr = read_i64(&sets.join("indptr.i64"))?;
    let indices = read_i64(&sets.join("indices.i64"))?;
    let scope = in_execution_order(sites(model).into_iter().filter(|s| offsets.contains_key(&s.name)).collect());
    let (mut libraries, mut ranks) = (Vec::new(), Vec::new());
    let mut masks: Vec<Vec<Array2<f64>>> = vec![Vec::new(); passages];
    for site in &scope {
        let (offset, count) = offsets[&site.name];
        let (d_out, d_in) = matrix(model, site)?.dim();
        let read = |side: &str, cols: usize| read_f64_matrix(&library.join(format!("{}.{side}.f64", site.name)), cols);
        let l = Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) };
        if l.v.nrows() != count || l.u.nrows() != count {
            return Err(format!("{}: {} read and {} write vectors for {count} subcomponents", site.name, l.v.nrows(), l.u.nrows()));
        }
        for (p, per_site) in masks.iter_mut().enumerate() {
            let mut m = Array2::<f64>::zeros((context, count));
            for r in 0..context {
                let row = p * 512 + r;
                if row + 1 >= indptr.len() {
                    return Err(format!("VPD's sets end before passage {p} row {r}"));
                }
                for &g in &indices[indptr[row] as usize..indptr[row + 1] as usize] {
                    let g = g as usize;
                    if g >= offset && g < offset + count {
                        m[[r, g - offset]] = 1.0;
                    }
                }
            }
            per_site.push(m);
        }
        ranks.push(vec![1; count]);
        libraries.push(l);
    }
    Ok(Given { sites: scope, libraries, ranks, masks })
}

fn stats(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n == 0 {
        return Value::Null;
    }
    let at = |q: f64| sorted[((q * (n - 1) as f64).round() as usize).min(n - 1)];
    json!({"mean": sorted.iter().sum::<f64>() / n as f64, "median": at(0.5), "p95": at(0.95), "max": sorted[n - 1], "count": n})
}

/// Every subset of `switches` (each a group of site indices switched together), every other site
/// native: per passage the worst subset's mean KL and that subset, per word its worst KL, and the
/// KL per word with every switch on.
fn exhaustive(model: &OperatorProgram, replacement: &dyn Replacement, passages: &[Passage], switches: &[Vec<usize>]) -> Result<Value, String> {
    let names: Vec<String> = replacement.sites().iter().map(|s| s.name.clone()).collect();
    let full = (1usize << switches.len()) - 1;
    let mut worst = vec![0.0_f64; passages.len()];
    let mut worst_subset = vec![Vec::<String>::new(); passages.len()];
    let mut word_worst: Vec<Array1<f64>> = passages.iter().map(|p| Array1::zeros(p.base.rows)).collect();
    let mut all_replaced = Vec::new();
    let mut all_replaced_passage = Vec::new();
    for subset in 1..=full {
        let mut members: Vec<usize> = (0..switches.len()).filter(|b| subset >> b & 1 == 1).flat_map(|b| switches[b].iter().copied()).collect();
        members.sort_unstable();
        let masked = replacement.masked(model, &members)?;
        let kls: Vec<Array1<f64>> = passages
            .par_iter()
            .enumerate()
            .map(|(p, passage)| {
                let (trace, _) = replacement.run(&masked, &members, p, &passage.base)?;
                Ok(kl_score_only(&passage.target, &trace.values[masked.program.output]))
            })
            .collect::<Result<_, String>>()?;
        for (p, kl) in kls.iter().enumerate() {
            let mean = kl.mean().unwrap_or(0.0);
            if mean > worst[p] {
                worst[p] = mean;
                worst_subset[p] = members.iter().map(|k| names[*k].clone()).collect();
            }
            word_worst[p].zip_mut_with(kl, |w, k| *w = w.max(*k));
            if subset == full {
                all_replaced.extend(kl.iter().copied());
                all_replaced_passage.push(mean);
            }
        }
    }
    let words: Vec<f64> = word_worst.iter().flat_map(|w| w.iter().copied()).collect();
    Ok(json!({"switches": switches.iter().map(|g| g.iter().map(|k| names[*k].clone()).collect::<Vec<_>>()).collect::<Vec<_>>(), "subsets": full,
              "all_replaced": stats(&all_replaced), "passage_worst_subset": stats(&worst), "word_worst_subset": stats(&words),
              "per_passage": {"all_replaced": all_replaced_passage, "worst": worst, "worst_subset": worst_subset}}))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_claim_2951 FRONTIER OUT.json {ours:LIBRARY_DIR|vpd:VPD_LIBRARY:VPD_SETS} [passages] [context] [stages]";
    let frontier = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let which = args.get(3).ok_or(usage)?;
    let count = |i: usize, default: usize| args.get(i).map_or(Ok(default), |v| v.parse::<usize>().map_err(|e| format!("{v}: {e}")));
    let (passage_count, context) = (count(4, 32)?, count(5, 512)?);
    let started = std::time::Instant::now();
    let imported = import_language_model(&frontier, passage_count, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let passages: Vec<Passage> = (0..passage_count)
        .into_par_iter()
        .map(|p| Passage::new(model, family.select(&(p * context..(p + 1) * context).collect::<Vec<_>>())))
        .collect::<Result<_, _>>()?;
    let replacement: Box<dyn Replacement> = if let Some(dir) = which.strip_prefix("ours:") {
        Box::new(ours(model, Path::new(dir))?)
    } else if let Some((library, sets)) = which.strip_prefix("vpd:").and_then(|rest| rest.split_once(':')) {
        Box::new(vpd(model, Path::new(library), Path::new(sets), passage_count, context)?)
    } else {
        return Err(usage.to_string());
    };
    let replaced: Vec<Site> = replacement.sites();
    let layer_of = |s: &Site| s.name.strip_prefix("blocks.").and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok());
    let mut layers: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, s) in replaced.iter().enumerate() {
        layers.entry(layer_of(s).ok_or_else(|| format!("{}: no layer in its name", s.name))?).or_default().push(k);
    }
    eprintln!("{} sites in {} layers, {} passages of {context}", replaced.len(), layers.len(), passages.len());
    let mut report = json!({"frontier": frontier, "explanation": which, "passages": passage_count, "context": context});
    let write = |report: &Value| -> Result<(), String> {
        let partial = out.with_extension("partial");
        std::fs::write(&partial, serde_json::to_string_pretty(report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        std::fs::rename(&partial, &out).map_err(|e| e.to_string())
    };
    let wanted: Option<Vec<String>> = args.get(6).map(|s| s.split(',').map(str::to_string).collect());
    let runs = |stage: &str| wanted.as_ref().is_none_or(|w| w.iter().any(|s| s == stage));
    let mut stages = Vec::new();
    if runs("layers") {
        stages.push(("layer_switches".to_string(), layers.values().cloned().collect::<Vec<_>>()));
    }
    for (l, members) in &layers {
        if runs(&l.to_string()) {
            stages.push((format!("sites_layer_{l}"), members.iter().map(|k| vec![*k]).collect()));
        }
    }
    for (key, switches) in stages {
        let clock = std::time::Instant::now();
        report[&key] = exhaustive(model, replacement.as_ref(), &passages, &switches)?;
        eprintln!(
            "{key}: all replaced {}, passage worst subset {}, {:.0}s",
            report[&key]["all_replaced"]["mean"],
            report[&key]["passage_worst_subset"]["mean"],
            clock.elapsed().as_secs_f64()
        );
        report["seconds"] = json!(started.elapsed().as_secs_f64());
        write(&report)?;
    }
    Ok(())
}

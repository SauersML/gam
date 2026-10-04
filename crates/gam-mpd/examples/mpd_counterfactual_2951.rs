//! The counterfactual-response benchmark on a language-model export (#2951,
//! `gam_mpd::counterfactual`): per declared native intervention, how far the explanation's own
//! program's response is from the native model's.
//!
//! `mpd_counterfactual_2951 EXPORT_DIR SPEC.json EXPLANATION OUT.json`
//!
//! `EXPORT_DIR` holds the passages (`tokens.f64`). `SPEC.json` is the frozen episode list
//! (`bench/vpd_2951/counterfactual_spec.py`, read by `gam_mpd::counterfactual::Spec`).
//! `EXPLANATION` is `native` (the native model explaining itself: zero disagreement, a check of
//! the evaluator) or `units:LIBRARY_DIR:SELECTIONS_DIR`: the explanation's library
//! (`gam_mpd::counterfactual::load_libraries`) and the selections its own rule made on its own
//! program's states under each episode (`selections.json` `{"offsets": [...], "episodes": {id:
//! index}}`, and the selections as one CSR over rows, `sets.indptr.u64`, `sets.indices.u32` of
//! global unit numbers: selection `i` is rows `i·rows .. (i+1)·rows`), or `fitted:LIBRARY_DIR`, a
//! fitted explanation (`mpd_e2e_2951`'s `OUT_DIR/library`: per site `{site}.{v,u,fisher,moment}.f64`
//! and `{site}.json`) whose own rule (`gam_mpd::counterfactual::FittedRule`) runs inside its
//! program under every episode, its rule priced beyond its library by
//! `gam_mpd::counterfactual::rule_price` on its reads of the clean passages. `OUT.json` gets every
//! episode's scores, their means per group, and the decoder's gap to the export's reference
//! logits of its first passage.

use gam_mpd::counterfactual::{
    Decoder, Explanation, FittedRule, Maps, Program, Scored, Selection, Selector, Spec, evaluate, fitted_libraries, fitted_sites, load_libraries, passages, read_f64_matrix,
    rule_price, summarize,
};
use gam_mpd::explanation::Fitted;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, matrix, sites};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn read_u64(path: &Path) -> Result<Vec<u64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("eight bytes"))).collect())
}

fn read_u32(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("four bytes"))).collect())
}

/// The explanation's selections, one per episode id.
struct Selections {
    rows: usize,
    offsets: Vec<usize>,
    index: BTreeMap<String, usize>,
    indptr: Vec<u64>,
    indices: Vec<u32>,
}

impl Selections {
    fn load(dir: &Path, rows: usize) -> Result<Self, String> {
        let spec: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("selections.json")).map_err(|e| format!("{}: {e}", dir.display()))?).map_err(|e| e.to_string())?;
        let offsets = spec["offsets"].as_array().ok_or("selections.json has no offsets")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
        let index = spec["episodes"].as_object().ok_or("selections.json has no episodes")?.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as usize)).collect();
        Ok(Self { rows, offsets, index, indptr: read_u64(&dir.join("sets.indptr.u64"))?, indices: read_u32(&dir.join("sets.indices.u32"))? })
    }

    fn of(&self, id: &str) -> Result<Selection, String> {
        let i = *self.index.get(id).ok_or_else(|| format!("no selection for episode {id}"))?;
        if (i + 1) * self.rows + 1 > self.indptr.len() {
            return Err(format!("selection {i} is past the sets file"));
        }
        let lists = (0..self.rows).map(|r| self.indices[self.indptr[i * self.rows + r] as usize..self.indptr[i * self.rows + r + 1] as usize].to_vec()).collect();
        Ok(Selection::from_global(lists, &self.offsets))
    }
}

/// The fitted sites in `dir` (each `{site}.json` with its blocks' ranks and bits and the `n` it was
/// fitted at, and its `{site}.{v,u,fisher,moment}.f64`), on the export's model.
fn load_fitted(export: &Path, dir: &Path) -> Result<(Vec<Fitted>, f64), String> {
    let model = import_language_model(export, 1, 1)?.program;
    let mut fitted = Vec::new();
    let mut observations = None;
    for site in sites(&model) {
        let name = site.name.clone();
        let Ok(text) = std::fs::read_to_string(dir.join(format!("{name}.json"))) else { continue };
        let record: Value = serde_json::from_str(&text).map_err(|e| format!("{name}.json: {e}"))?;
        let n = record["fitted_with"]["n"].as_f64().ok_or_else(|| format!("{name}.json: no fitted_with.n"))?;
        if observations.is_some_and(|o| o != n) {
            return Err(format!("{name}: fitted at n = {n}, the others at {observations:?}"));
        }
        observations = Some(n);
        let w = matrix(&model, &site)?;
        let (d_out, d_in) = w.dim();
        let read = |side: &str, cols: usize| read_f64_matrix(&dir.join(format!("{name}.{side}.f64")), cols);
        let library = Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) };
        let ranks: Vec<usize> = serde_json::from_value(record["ranks"].clone()).map_err(|e| format!("{name} ranks: {e}"))?;
        let bits: Vec<f64> = serde_json::from_value(record["bits"].clone()).map_err(|e| format!("{name} bits: {e}"))?;
        fitted.push(Fitted::new(site, w, (library, ranks, bits), (read("fisher", d_out)?, read("moment", d_in)?), n)?);
    }
    Ok((fitted, observations.ok_or_else(|| format!("{}: no fitted site", dir.display()))?))
}

/// The fitted rule run on a program while keeping every site's reads.
struct Recording<'a> {
    rule: FittedRule<'a>,
    reads: Vec<Vec<Array2<f64>>>,
}

impl Selector for Recording<'_> {
    fn select(&mut self, site: usize, input: &Array2<f64>) -> Vec<Vec<(u32, f64)>> {
        self.reads[site].push(input.clone());
        self.rule.select(site, input)
    }
}

/// The decoder's largest gap to the export's reference log-probabilities of its first passage.
fn reference_gap(decoder: &Decoder, export: &Path, tokens: &[u32]) -> Result<Value, String> {
    let record: Value = serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let Some(vocab) = record["files"]["logits_row0"]["shape"][1].as_u64() else {
        return Ok(Value::Null);
    };
    let reference = read_f64_matrix(&export.join("logits_row0.f64"), vocab as usize)?;
    let n = reference.nrows().min(tokens.len());
    let mut native = Program::new(Maps::Native(decoder), &[], None);
    let forward = decoder.forward(&tokens[..n], &mut native, &[]);
    let ours = decoder.log_probs(&forward.residual);
    let mut gap = 0.0_f64;
    for (a, r) in ours.outer_iter().zip(reference.outer_iter()) {
        let max = r.fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        let log_total = r.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
        gap = a.iter().zip(r.iter()).fold(gap, |g, (x, y)| g.max((x - (y - log_total)).abs()));
    }
    Ok(json!(gap))
}

const TILE: usize = 64;

/// Every episode under the selections of a rule run elsewhere on its own program's states.
fn score_units(decoder: &Decoder, spec: &Spec, passages: &[Vec<u32>], library: &Path, sets: &Path) -> Result<(Vec<Scored>, Value), String> {
    let libraries = load_libraries(library, decoder)?;
    let selections = Selections::load(sets, spec.rows)?;
    let counts: Vec<usize> = libraries.iter().map(|l| l.as_ref().map_or(0, |l| l.v.nrows())).collect();
    if selections.offsets.len() != counts.len() + 1 || counts.iter().zip(selections.offsets.windows(2)).any(|(c, o)| *c != o[1] - o[0]) {
        return Err("the selections' offsets do not number the library's units".to_string());
    }
    let scored = evaluate(decoder, spec, passages, Some(&Explanation { libraries: &libraries, selector: Box::new(given_rule(&selections)) }), TILE)?;
    Ok((scored, json!({"reals": libraries.iter().flatten().map(|l| l.v.len() + l.u.len()).sum::<usize>()})))
}

/// Each episode's given selection as its selector.
fn given_rule<'a>(selections: &'a Selections) -> impl Fn(&str) -> Result<Box<dyn Selector + 'a>, String> + Sync + 'a {
    move |id: &str| Ok(Box::new(selections.of(id)?))
}

/// A fitted explanation's rule as every episode's selector.
fn fitted_rule<'a>(sites: &'a [Option<&'a Fitted>]) -> impl Fn(&str) -> Result<Box<dyn Selector + 'a>, String> + Sync + 'a {
    move |_: &str| Ok(Box::new(FittedRule { sites }))
}

/// Every episode under a fitted explanation's own rule, and the rule's price on its reads of the
/// clean passages.
fn score_fitted(export: &Path, decoder: &Decoder, spec: &Spec, passages: &[Vec<u32>], dir: &Path) -> Result<(Vec<Scored>, Value, Value), String> {
    let (fitted, observations) = load_fitted(export, dir)?;
    let by_site = fitted_sites(decoder, &fitted)?;
    let libraries = fitted_libraries(&by_site);
    eprintln!("{} fitted sites at n = {observations}", fitted.len());
    let scored = evaluate(decoder, spec, passages, Some(&Explanation { libraries: &libraries, selector: Box::new(fitted_rule(&by_site)) }), TILE)?;
    let clean: Vec<usize> = spec.episodes.iter().filter(|e| e.actions.is_empty()).map(|e| e.passage).collect();
    let recorded: Vec<Vec<Vec<Array2<f64>>>> = clean
        .par_iter()
        .map(|&p| {
            let mut recording = Recording { rule: FittedRule { sites: &by_site }, reads: vec![Vec::new(); decoder.sites()] };
            let mut program = Program::new(Maps::Units { decoder, libraries: &libraries, selector: &mut recording }, &[], None);
            decoder.forward(&passages[p], &mut program, &[]);
            recording.reads
        })
        .collect();
    let (mut chosen, mut reads) = (Vec::new(), Vec::new());
    for (site, f) in by_site.iter().enumerate() {
        if let Some(f) = f {
            let views: Vec<_> = recorded.iter().flat_map(|r| r[site].iter().map(|x| x.view())).collect();
            reads.push(ndarray::concatenate(Axis(0), &views).map_err(|e| e.to_string())?);
            chosen.push(*f);
        }
    }
    let rule = rule_price(&chosen, &reads, observations)?;
    eprintln!("rule: {} reals at p = {}, {:.4e} bits", rule.reals, rule.precision, rule.bits);
    let library = json!({"reals": fitted.iter().map(|f| f.library.v.len() + f.library.u.len()).sum::<usize>(),
                         "structured_bits": fitted.iter().map(|f| f.bits.iter().sum::<f64>()).sum::<f64>(), "observations": observations});
    Ok((scored, library, json!(rule)))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_counterfactual_2951 EXPORT_DIR SPEC.json {native|units:LIBRARY_DIR:SELECTIONS_DIR|fitted:LIBRARY_DIR} OUT.json";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let spec_path = PathBuf::from(args.get(2).ok_or(usage)?);
    let explanation = args.get(3).ok_or(usage)?.clone();
    let out = PathBuf::from(args.get(4).ok_or(usage)?);
    let started = std::time::Instant::now();
    let decoder = Decoder::from_export(&export)?;
    let spec = Spec::load(&spec_path, &decoder)?;
    let passages = passages(&export, spec.rows)?;
    let native_check = reference_gap(&decoder, &export, &passages[0])?;
    eprintln!("{} episodes; decoder against the export's reference row 0: {native_check}", spec.episodes.len());
    let (scored, library, rule) = if let Some(dir) = explanation.strip_prefix("fitted:") {
        score_fitted(&export, &decoder, &spec, &passages, Path::new(dir))?
    } else if let Some((library, sets)) = explanation.strip_prefix("units:").and_then(|rest| rest.split_once(':')) {
        let (scored, library) = score_units(&decoder, &spec, &passages, Path::new(library), Path::new(sets))?;
        (scored, library, Value::Null)
    } else if explanation == "native" {
        (evaluate(&decoder, &spec, &passages, None, TILE)?, Value::Null, Value::Null)
    } else {
        return Err(usage.to_string());
    };
    let summary = summarize(&scored);
    eprintln!("{}", serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?);
    let report = json!({"explanation": explanation, "spec": spec_path.display().to_string(), "native_check": native_check, "library": library, "rule": rule,
                        "summary": summary, "episodes": scored, "seconds": started.elapsed().as_secs_f64()});
    std::fs::write(&out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

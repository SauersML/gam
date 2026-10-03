//! The counterfactual-response benchmark on a language-model export (#2951,
//! `gam_mpd::counterfactual`): per declared native intervention, how far the explanation's own
//! program's response is from the native model's.
//!
//! `mpd_counterfactual_2951 EXPORT_DIR SPEC.json EXPLANATION OUT.json`
//!
//! `EXPORT_DIR` holds the passages (`tokens.f64`, the first `rows` columns read). `SPEC.json` is
//! the frozen episode list (`bench/vpd_2951/counterfactual_spec.py`):
//!
//! ```text
//! {"rows": 512, "edits": {name: {"site", "rank", "left": file (d_out × rank), "right": file (d_in × rank)}},
//!  "episodes": [{"id", "group", "passage", "donor"?, "interface_rows": [..],
//!                "actions": [{"type": "scale_input", "site", "row" (null: every row), "cols": [a, b], "scale"}
//!                          | {"type": "mix_input" | "mix_output", "site", "row", "alpha"}
//!                          | {"type": "add_map", "site", "edit"}]}]}
//! ```
//!
//! (files relative to the spec). Every passage has a clean episode `clean/{passage}`.
//! `EXPLANATION` is `native` (the native model explaining itself: zero disagreement, a check of
//! the evaluator) or `units:LIBRARY_DIR:SELECTIONS_DIR`: the explanation's library
//! (`gam_mpd::counterfactual::load_libraries`) and the selections its own rule made on its own
//! program's states under each episode (`selections.json` `{"offsets": [...], "episodes": {id:
//! index}}`, and the selections as one CSR over rows, `sets.indptr.u64`, `sets.indices.u32` of
//! global unit numbers: selection `i` is rows `i·rows .. (i+1)·rows`). A donor runs on the
//! selection of the donor passage's clean episode. `OUT.json` gets every episode's scores and
//! their means per group.

use gam_mpd::counterfactual::{Action, Decoder, Donor, Forward, Maps, Program, Rows, Selection, Selector, load_libraries, read_f64_matrix, score};
use ndarray::Array2;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn read_u64(path: &Path) -> Result<Vec<u64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("eight bytes"))).collect())
}

fn read_u32(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("four bytes"))).collect())
}

fn field(value: &Value, key: &str) -> Result<usize, String> {
    value[key].as_u64().map(|v| v as usize).ok_or_else(|| format!("{value} has no {key}"))
}

/// The explanation's selections, read lazily per episode.
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

fn parse_actions(episode: &Value, edits: &BTreeMap<String, (usize, Arc<Array2<f64>>, Arc<Array2<f64>>)>) -> Result<Vec<Action>, String> {
    let mut actions = Vec::new();
    for a in episode["actions"].as_array().ok_or_else(|| format!("{} has no actions", episode["id"]))? {
        let site = field(a, "site")?;
        actions.push(match a["type"].as_str().unwrap_or_default() {
            "scale_input" => {
                let cols = a["cols"].as_array().ok_or("scale_input without cols")?;
                Action::ScaleInput {
                    site,
                    rows: a["row"].as_u64().map_or(Rows::All, |r| Rows::One(r as usize)),
                    cols: (cols[0].as_u64().unwrap_or(0) as usize, cols[1].as_u64().unwrap_or(0) as usize),
                    scale: a["scale"].as_f64().ok_or("scale_input without scale")?,
                }
            }
            "mix_input" => Action::MixInput { site, row: field(a, "row")?, alpha: a["alpha"].as_f64().ok_or("mix without alpha")? },
            "mix_output" => Action::MixOutput { site, row: field(a, "row")?, alpha: a["alpha"].as_f64().ok_or("mix without alpha")? },
            "add_map" => {
                let (edit_site, left, right) = edits.get(a["edit"].as_str().unwrap_or_default()).ok_or("unknown edit")?;
                if *edit_site != site {
                    return Err(format!("edit {} is of site {edit_site}, not {site}", a["edit"]));
                }
                Action::AddMap { site, left: left.clone(), right: right.clone() }
            }
            other => return Err(format!("unknown action {other}")),
        });
    }
    Ok(actions)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_counterfactual_2951 EXPORT_DIR SPEC.json {native|units:LIBRARY_DIR:SELECTIONS_DIR} OUT.json";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let spec_path = PathBuf::from(args.get(2).ok_or(usage)?);
    let explanation = args.get(3).ok_or(usage)?.clone();
    let out = PathBuf::from(args.get(4).ok_or(usage)?);
    let started = std::time::Instant::now();
    let decoder = Decoder::from_export(&export)?;
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(&spec_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let spec_dir = spec_path.parent().unwrap_or(Path::new("."));
    let rows = field(&spec, "rows")?;
    let mut edits = BTreeMap::new();
    for (name, e) in spec["edits"].as_object().into_iter().flatten() {
        let (site, rank) = (field(e, "site")?, field(e, "rank")?);
        let left = read_f64_matrix(&spec_dir.join(e["left"].as_str().unwrap_or_default()), rank)?;
        let right = read_f64_matrix(&spec_dir.join(e["right"].as_str().unwrap_or_default()), rank)?;
        let (d_out, d_in) = decoder.native(site).dim();
        if left.nrows() != d_out || right.nrows() != d_in {
            return Err(format!("edit {name}: left {:?}, right {:?} for a {d_out}×{d_in} site", left.dim(), right.dim()));
        }
        edits.insert(name.clone(), (site, Arc::new(left), Arc::new(right)));
    }
    let (libraries, selections) = match explanation.strip_prefix("units:") {
        Some(rest) => {
            let (library, sets) = rest.split_once(':').ok_or(usage)?;
            let libraries = load_libraries(Path::new(library), &decoder)?;
            let selections = Selections::load(Path::new(sets), rows)?;
            if selections.offsets.len() != libraries.len() + 1 || libraries.iter().zip(selections.offsets.windows(2)).any(|(l, o)| l.v.nrows() != o[1] - o[0]) {
                return Err("the selections' offsets do not number the library's units".to_string());
            }
            (Some(libraries), Some(selections))
        }
        None if explanation == "native" => (None, None),
        None => return Err(usage.to_string()),
    };
    let record: Value = serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let width = record["files"]["tokens"]["shape"][1].as_u64().ok_or("export has no tokens")? as usize;
    let passages: Vec<Vec<u32>> = read_f64_matrix(&export.join("tokens.f64"), width)?.outer_iter().map(|r| r.iter().take(rows).map(|t| *t as u32).collect()).collect();
    let episodes = spec["episodes"].as_array().ok_or("spec has no episodes")?.clone();
    let all_rows: Vec<usize> = (0..rows).collect();
    // The native clean forward of every passage, its layers kept at every row.
    let named: std::collections::BTreeSet<usize> = episodes.iter().filter_map(|e| e["passage"].as_u64().map(|p| p as usize)).collect();
    let clean: BTreeMap<usize, Forward> = named
        .par_iter()
        .map(|&p| {
            let mut native = Program { maps: Maps::Native(&decoder), actions: &[], donor: None, record: Vec::new() };
            (p, decoder.forward(&passages[p], &mut native, &all_rows))
        })
        .collect();
    eprintln!("{} passages' clean forwards, {:.0}s", clean.len(), started.elapsed().as_secs_f64());
    // The decoder against the export's own reference logits of its first row, when it has them.
    let native_check = match (clean.get(&0), record["files"]["logits_row0"]["shape"].as_array()) {
        (Some(c), Some(shape)) => {
            let vocab = shape[1].as_u64().ok_or("logits_row0 shape")? as usize;
            let reference = read_f64_matrix(&export.join("logits_row0.f64"), vocab)?;
            let n = reference.nrows().min(rows);
            let ours = decoder.log_probs(&c.residual.slice(ndarray::s![..n, ..]).to_owned());
            let mut gap = 0.0_f64;
            for (a, r) in ours.outer_iter().zip(reference.outer_iter()) {
                let max = r.fold(f64::NEG_INFINITY, |m, v| m.max(*v));
                let log_total = r.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
                gap = a.iter().zip(r.iter()).fold(gap, |g, (x, y)| g.max((x - (y - log_total)).abs()));
            }
            eprintln!("native decoder against the export's reference log-probabilities of row 0: max gap {gap:.3e}");
            json!(gap)
        }
        _ => Value::Null,
    };
    const TILE: usize = 64;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Value> = episodes
        .par_iter()
        .map(|e| -> Result<Value, String> {
            let id = e["id"].as_str().ok_or("episode without id")?;
            let p = field(e, "passage")?;
            let actions = parse_actions(e, &edits)?;
            let interface: Vec<usize> = e["interface_rows"].as_array().ok_or("episode without interface_rows")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
            let from = actions.iter().map(Action::first_row).min().unwrap_or(0);
            let keys: Vec<(usize, usize, bool)> = actions.iter().filter_map(Action::donor_state).collect();
            let donor_of = |maps: Maps<'_>| -> Result<Donor, String> {
                let d = field(e, "donor")?;
                let mut program = Program { maps, actions: &[], donor: None, record: keys.iter().map(|k| (*k, None)).collect() };
                decoder.forward(&passages[d], &mut program, &[]);
                Ok(Donor { states: program.record.into_iter().map(|(k, v)| v.map(|v| (k, v)).ok_or("donor state not reached")).collect::<Result<_, _>>()? })
            };
            let native_donor = if keys.is_empty() { Donor::default() } else { donor_of(Maps::Native(&decoder))? };
            let mut native = Program { maps: Maps::Native(&decoder), actions: &actions, donor: Some(&native_donor), record: Vec::new() };
            let native_forward = decoder.forward(&passages[p], &mut native, &interface);
            let (explained_forward, selected) = match (&libraries, &selections) {
                (Some(libraries), Some(selections)) => {
                    let own_donor = if keys.is_empty() {
                        Donor::default()
                    } else {
                        let mut donor_selection = selections.of(&format!("clean/{}", field(e, "donor")?))?;
                        donor_of(Maps::Units { libraries, selector: &mut donor_selection })?
                    };
                    let mut selection = selections.of(id)?;
                    let selected = selection.mean_selected();
                    let selector: &mut dyn Selector = &mut selection;
                    let mut explained = Program { maps: Maps::Units { libraries, selector }, actions: &actions, donor: Some(&own_donor), record: Vec::new() };
                    (decoder.forward(&passages[p], &mut explained, &interface), selected)
                }
                _ => {
                    let mut explained = Program { maps: Maps::Native(&decoder), actions: &actions, donor: Some(&native_donor), record: Vec::new() };
                    (decoder.forward(&passages[p], &mut explained, &interface), f64::NAN)
                }
            };
            let scores = score(&decoder, &native_forward, &explained_forward, &clean[&p], &interface, from, TILE);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 100 == 0 {
                eprintln!("{n}/{} episodes, {:.0}s", episodes.len(), started.elapsed().as_secs_f64());
            }
            Ok(json!({"id": id, "group": e["group"], "passage": p, "from_row": from, "selected_per_row": selected,
                      "kl": scores.kl, "native_effect": scores.native_effect, "top1_agree": scores.top1_agree,
                      "interface_kl": scores.interface_kl, "interface_effect": scores.interface_effect}))
        })
        .collect::<Result<_, _>>()?;
    let mut groups: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for r in &results {
        groups.entry(r["group"].as_str().unwrap_or("ungrouped").to_string()).or_default().push(r);
    }
    let mean = |rs: &[&Value], key: &str| rs.iter().map(|r| r[key].as_f64().unwrap_or(0.0)).sum::<f64>() / rs.len().max(1) as f64;
    let layer_mean = |rs: &[&Value], key: &str| -> Vec<f64> {
        let layers = rs.first().and_then(|r| r[key].as_array()).map_or(0, Vec::len);
        (0..layers).map(|l| rs.iter().map(|r| r[key][l].as_f64().unwrap_or(0.0)).sum::<f64>() / rs.len().max(1) as f64).collect()
    };
    let summary: BTreeMap<String, Value> = groups
        .iter()
        .map(|(g, rs)| {
            (g.clone(), json!({"episodes": rs.len(), "kl": mean(rs, "kl"), "native_effect": mean(rs, "native_effect"), "top1_agree": mean(rs, "top1_agree"),
                               "interface_kl": layer_mean(rs, "interface_kl"), "interface_effect": layer_mean(rs, "interface_effect")}))
        })
        .collect();
    let library_reals = libraries.as_ref().map(|ls| ls.iter().map(|l| l.v.len() + l.u.len()).sum::<usize>());
    eprintln!("{}", serde_json::to_string_pretty(&summary).unwrap_or_default());
    let report = json!({"explanation": explanation, "spec": spec_path.display().to_string(), "native_check": native_check, "library_reals": library_reals,
                        "summary": summary, "episodes": results, "seconds": started.elapsed().as_secs_f64()});
    std::fs::write(&out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

//! The counterfactual-response benchmark on a language-model export (#2951,
//! `gam_mpd::counterfactual`): per declared episode, the disagreement between the native model's
//! response to an intervention and the response the explanation's own program predicts.
//!
//! `mpd_counterfactual_2951 EXPORT_DIR LIBRARY_DIR EPISODES_DIR OUT.json`
//!
//! `EXPORT_DIR` is the export whose token rows are the passages (`tokens.f64`, the first
//! `rows` columns read). `LIBRARY_DIR` is the explanation's library (`{site}.v.f64`,
//! `{site}.u.f64` per site, `gam_mpd::counterfactual::load_libraries`). `EPISODES_DIR` holds
//! `episodes.json`
//!
//! ```text
//! {"rows": 512, "offsets": [global first unit of each site, then the total],
//!  "edit"?: {"site", "rank", "left": file (d_out × rank), "right": file (d_in × rank),
//!            "unit", "write": file (d_out)},
//!  "episodes": [{"kind": "clean" | "unit" | "input" | "edit", "passage", "sets",
//!                "row"?, "site"?, "unit"?, "scale"?, "donor"?}]}
//! ```
//!
//! and the explanation's selections as one CSR over rows, `sets.indptr.u64` and
//! `sets.indices.u32` (global unit numbers): selection `i` is rows `i·rows .. (i+1)·rows`. An
//! episode names the selection its explanation chose under its intervention; an `input`
//! episode's donor runs on the selection of the donor passage's clean episode. `OUT.json` gets
//! every episode's scores and the means per kind.

use gam_mpd::counterfactual::{
    Decoder, Explained, ExplanationChange, Native, NativeChange, Selection, load_libraries, read_f64_matrix, score,
};
use ndarray::{Array1, Array2};
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

fn field(episode: &Value, key: &str) -> Result<usize, String> {
    episode[key].as_u64().map(|v| v as usize).ok_or_else(|| format!("episode {episode} has no {key}"))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_counterfactual_2951 EXPORT_DIR LIBRARY_DIR EPISODES_DIR OUT.json";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let library_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let episodes_dir = PathBuf::from(args.get(3).ok_or(usage)?);
    let out = PathBuf::from(args.get(4).ok_or(usage)?);
    let started = std::time::Instant::now();
    let decoder = Decoder::from_export(&export)?;
    let libraries = load_libraries(&library_dir, &decoder)?;
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(episodes_dir.join("episodes.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let rows = spec["rows"].as_u64().ok_or("episodes.json has no rows")? as usize;
    let offsets: Vec<usize> = spec["offsets"].as_array().ok_or("episodes.json has no offsets")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
    if offsets.len() != libraries.len() + 1 || libraries.iter().zip(offsets.windows(2)).any(|(l, o)| l.v.nrows() != o[1] - o[0]) {
        return Err("the offsets do not number the library's units".to_string());
    }
    let record: Value = serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let shape = record["files"]["tokens"]["shape"].as_array().ok_or("export has no tokens")?;
    let width = shape[1].as_u64().ok_or("tokens shape")? as usize;
    let token_rows = read_f64_matrix(&export.join("tokens.f64"), width)?;
    let passages: Vec<Vec<u32>> = token_rows.outer_iter().map(|r| r.iter().take(rows).map(|t| *t as u32).collect()).collect();
    let indptr = read_u64(&episodes_dir.join("sets.indptr.u64"))?;
    let indices = read_u32(&episodes_dir.join("sets.indices.u32"))?;
    let selection = |i: usize| -> Result<Selection, String> {
        if (i + 1) * rows + 1 > indptr.len() {
            return Err(format!("selection {i} is past the sets file"));
        }
        let lists = (0..rows).map(|r| indices[indptr[i * rows + r] as usize..indptr[i * rows + r + 1] as usize].to_vec()).collect();
        Ok(Selection::from_global(lists, &offsets))
    };
    let episodes = spec["episodes"].as_array().ok_or("episodes.json has no episodes")?.clone();
    let clean_sets: BTreeMap<usize, usize> = episodes
        .iter()
        .filter(|e| e["kind"] == "clean")
        .map(|e| Ok((field(e, "passage")?, field(e, "sets")?)))
        .collect::<Result<_, String>>()?;
    let edit = match spec.get("edit").filter(|e| !e.is_null()) {
        Some(e) => {
            let site = field(e, "site")?;
            let rank = field(e, "rank")?;
            let (d_out, d_in) = decoder.native(site).dim();
            let path = |k: &str| episodes_dir.join(e[k].as_str().unwrap_or_default());
            let left = read_f64_matrix(&path("left"), rank)?;
            let right = read_f64_matrix(&path("right"), rank)?;
            let write = read_f64_matrix(&path("write"), d_out)?.row(0).to_owned();
            if left.nrows() != d_out || right.nrows() != d_in {
                return Err(format!("edit: left {:?}, right {:?} for a {d_out}×{d_in} site", left.dim(), right.dim()));
            }
            Some((site, left, right, field(e, "unit")?, write))
        }
        None => None,
    };
    // The native clean residual of every passage an episode names.
    let named: std::collections::BTreeSet<usize> = episodes.iter().filter_map(|e| e["passage"].as_u64().map(|p| p as usize)).collect();
    let clean: BTreeMap<usize, Array2<f64>> = named
        .par_iter()
        .map(|&p| {
            let mut native = Native { decoder: &decoder, record: Vec::new(), intervention: NativeChange::None };
            (p, decoder.residual(&passages[p], &mut native))
        })
        .collect();
    eprintln!("{} passages' clean forwards, {:.0}s", clean.len(), started.elapsed().as_secs_f64());
    // The decoder against the export's own reference logits of its first row, when it has them.
    let native_check = match (clean.get(&0), record["files"]["logits_row0"]["shape"].as_array()) {
        (Some(rows0), Some(shape)) => {
            let vocab = shape[1].as_u64().ok_or("logits_row0 shape")? as usize;
            let reference = read_f64_matrix(&export.join("logits_row0.f64"), vocab)?;
            let ours = decoder.log_probs(&rows0.slice(ndarray::s![..reference.nrows().min(rows0.nrows()), ..]).to_owned());
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
            let kind = e["kind"].as_str().ok_or("episode kind")?;
            let p = field(e, "passage")?;
            let chosen = selection(field(e, "sets")?)?;
            let (native_change, explanation_change, from) = match kind {
                "clean" => (NativeChange::None, ExplanationChange::None, 0),
                "unit" => {
                    let (site, row, unit) = (field(e, "site")?, field(e, "row")?, field(e, "unit")?);
                    let scale = e["scale"].as_f64().ok_or("unit episode has no scale")?;
                    let library = &libraries[site];
                    (
                        NativeChange::Unit { site, row, u: library.u.row(unit).to_owned(), v: library.v.row(unit).to_owned(), scale },
                        ExplanationChange::Unit { site, row, unit, scale },
                        row,
                    )
                }
                "input" => {
                    let (site, row, donor) = (field(e, "site")?, field(e, "row")?, field(e, "donor")?);
                    let mut native = Native { decoder: &decoder, record: vec![(site, None)], intervention: NativeChange::None };
                    decoder.residual(&passages[donor], &mut native);
                    let native_input: Array1<f64> = native.record[0].1.as_ref().ok_or("donor site not reached")?.row(row).to_owned();
                    let donor_selection = selection(*clean_sets.get(&donor).ok_or("donor has no clean episode")?)?;
                    let mut explained = Explained { libraries: &libraries, selection: &donor_selection, record: vec![(site, None)], intervention: ExplanationChange::None };
                    decoder.residual(&passages[donor], &mut explained);
                    let own_input: Array1<f64> = explained.record[0].1.as_ref().ok_or("donor site not reached")?.row(row).to_owned();
                    (NativeChange::Input { site, row, input: native_input }, ExplanationChange::Input { site, row, input: own_input }, row)
                }
                "edit" => {
                    let (site, left, right, unit, write) = edit.as_ref().ok_or("an edit episode without an edit")?;
                    (
                        NativeChange::Edit { site: *site, left: left.clone(), right: right.clone() },
                        ExplanationChange::Write { site: *site, unit: *unit, write: write.clone() },
                        0,
                    )
                }
                other => return Err(format!("unknown episode kind {other}")),
            };
            let mut native = Native { decoder: &decoder, record: Vec::new(), intervention: native_change };
            let native_rows = decoder.residual(&passages[p], &mut native);
            let mut explained = Explained { libraries: &libraries, selection: &chosen, record: Vec::new(), intervention: explanation_change };
            let explained_rows = decoder.residual(&passages[p], &mut explained);
            let (kl, effect, agree) = score(&decoder, &native_rows, &explained_rows, &clean[&p], from, TILE);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 50 == 0 {
                eprintln!("{n}/{} episodes, {:.0}s", episodes.len(), started.elapsed().as_secs_f64());
            }
            let mut result = e.clone();
            result["from_row"] = json!(from);
            result["kl"] = json!(kl);
            result["native_effect"] = json!(effect);
            result["top1_agree"] = json!(agree);
            Ok(result)
        })
        .collect::<Result<_, _>>()?;
    // Means per kind (unit episodes also per scale and per whether the unit was selected).
    let mut groups: BTreeMap<String, (f64, f64, f64, f64)> = BTreeMap::new();
    for r in &results {
        let mut key = r["kind"].as_str().unwrap_or_default().to_string();
        if key == "unit" {
            key = format!("unit scale {} {}", r["scale"], if r["selected"].as_bool().unwrap_or(false) { "selected" } else { "unselected" });
        }
        let g = groups.entry(key).or_default();
        g.0 += 1.0;
        g.1 += r["kl"].as_f64().unwrap_or(0.0);
        g.2 += r["native_effect"].as_f64().unwrap_or(0.0);
        g.3 += r["top1_agree"].as_f64().unwrap_or(0.0);
    }
    let summary: BTreeMap<String, Value> = groups
        .into_iter()
        .map(|(k, (n, kl, effect, agree))| (k, json!({"episodes": n, "kl": kl / n, "native_effect": effect / n, "top1_agree": agree / n})))
        .collect();
    eprintln!("{}", serde_json::to_string_pretty(&summary).unwrap_or_default());
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"summary": summary, "native_check": native_check, "episodes": results, "seconds": started.elapsed().as_secs_f64()})).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

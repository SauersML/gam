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
//! global unit numbers: selection `i` is rows `i·rows .. (i+1)·rows`). `OUT.json` gets every
//! episode's scores, their means per group, and the decoder's gap to the export's reference
//! logits of its first passage.

use gam_mpd::counterfactual::{Decoder, Explanation, Maps, Program, Selection, Selector, Spec, evaluate, load_libraries, passages, read_f64_matrix, summarize};
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

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_counterfactual_2951 EXPORT_DIR SPEC.json {native|units:LIBRARY_DIR:SELECTIONS_DIR} OUT.json";
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
    const TILE: usize = 64;
    let (scored, library_reals) = match explanation.strip_prefix("units:") {
        Some(rest) => {
            let (library, sets) = rest.split_once(':').ok_or(usage)?;
            let libraries = load_libraries(Path::new(library), &decoder)?;
            let selections = Selections::load(Path::new(sets), spec.rows)?;
            let counts: Vec<usize> = libraries.iter().map(|l| l.as_ref().map_or(0, |l| l.v.nrows())).collect();
            if selections.offsets.len() != counts.len() + 1 || counts.iter().zip(selections.offsets.windows(2)).any(|(c, o)| *c != o[1] - o[0]) {
                return Err("the selections' offsets do not number the library's units".to_string());
            }
            let selector = |id: &str| -> Result<Box<dyn Selector>, String> { Ok(Box::new(selections.of(id)?)) };
            let scored = evaluate(&decoder, &spec, &passages, Some(&Explanation { libraries: &libraries, selector: &selector }), TILE)?;
            (scored, json!(libraries.iter().flatten().map(|l| l.v.len() + l.u.len()).sum::<usize>()))
        }
        None if explanation == "native" => (evaluate(&decoder, &spec, &passages, None, TILE)?, Value::Null),
        None => return Err(usage.to_string()),
    };
    let summary = summarize(&scored);
    eprintln!("{}", serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?);
    let report = json!({"explanation": explanation, "spec": spec_path.display().to_string(), "native_check": native_check, "library_reals": library_reals,
                        "summary": summary, "episodes": scored, "seconds": started.elapsed().as_secs_f64()});
    std::fs::write(&out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

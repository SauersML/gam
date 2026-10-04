//! The method end to end on a language model, beside VPD's decomposition (#2951,
//! `gam_mpd::explanation`).
//!
//! `mpd_e2e_2951 SCOPE OUT_DIR TRAIN FRONTIER VPD_LIBRARY VPD_SETS [KEY=VALUE ...]`
//!
//! `SCOPE` is a site-name prefix (`blocks.0.`), `all`, or `smoke`: `blocks.0.` at sizes that run
//! in minutes (2 training sequences, 2 passages of 128 positions, n = 1e5, 2 rounds, 1 draw).
//! `TRAIN` and `FRONTIER` are engine exports of the model (`~/mpd-data/engine/vpd4l_e2e_train`,
//! `~/mpd-data/engine/vpd4l_frontier32`, whose sequences are `VPD_SETS`' rows in order),
//! `VPD_LIBRARY` VPD's subcomponents (`{site}.{v,u}.f64`, `~/mpd-data/pieces/vpd4l_library`) and
//! `VPD_SETS` its published sets (`sites.txt`, `indptr.i64`, `indices.i64`,
//! `~/mpd-data/pieces/vpd4l_sets`).
//!
//! The scope's sites get libraries fitted on the training export's first `sequences` sequences in
//! execution order on hybrid inputs (`gam_mpd::explanation::fit`, whatever `gam_mpd::site_fit`
//! holds when it runs), written to `OUT_DIR/library/` (`{site}.{v,u,fisher,moment}.f64`,
//! `{site}.json` with its blocks, their bits and the settings; a site whose files match the
//! settings is read back, not refitted). The explanation then runs autonomously on the frontier
//! export's first `passages` passages, every site outside the scope native, and so does VPD's
//! published decomposition on the same sites with its own published sets (chosen on the model's
//! states, residual off). Both are scored alike, and `OUT_DIR/e2e.json` (rewritten after every
//! stage) gets, for `ours` and `vpd`:
//!
//! * `e2e`: `KL(model ‖ explanation)` per token with every scope site replaced;
//! * `site_switch`: the site-switch claim's worst subset of replaced sites per passage
//!   (`gam_mpd::explanation::site_switch`: exhaustive up to its limit, else its search with
//!   `random` random subsets);
//! * `bits`: the description bits per word of the blocks that ran, each block priced by
//!   `gam_mpd::describe::Structured` in the statistics its site was fitted in (VPD's
//!   subcomponents in the same geometry), and the library's bits paid once.
//!
//! A rerun into the same `OUT_DIR` with the same settings resumes: fitted sites, VPD's prices
//! (`library/{site}.vpd_bits.json`) and every finished stage of `e2e.json` are read back.
//!
//! Keys (defaults): `sequences` (4), `passages` (32), `context` (512), `n` (1e6), `start` (`own`:
//! each site starts from its units or Fisher-SVD pieces; `vpd`: from VPD's subcomponents, so `ours`
//! is this code's selection and execution of VPD's library), `rounds` (50; 0 keeps each site's
//! starting pieces), `blocks` (1: gate the library in blocks; 0: every subcomponent its own),
//! `draws` (4), `random` (64), `device` (`f64`: samples, passages and every evaluation on a float64
//! accelerator when there is one; `any`: the Apple GPU's f32 too; `off`: the CPU;
//! `gam_mpd::core_device`).

use gam_mpd::blocks::Describe;
use gam_mpd::counterfactual::read_f64_matrix;
use gam_mpd::explanation::{Explanation, Fitted, Given, Passage, Replacement, Settings, SiteSwitch, bits_per_word, fit, in_execution_order, replaced, site_switch};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, Site, matrix, sites};
use gam_mpd::operator_program::{FamilyInputs, OperatorProgram};
use ndarray::{Array1, Array2, s};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn write_f64(path: &Path, m: &Array2<f64>) -> Result<(), String> {
    let bytes: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
    let partial = path.with_extension("partial");
    std::fs::write(&partial, bytes).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, path).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_i64(path: &Path) -> Result<Vec<i64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("eight bytes"))).collect())
}

/// The run's settings, from `KEY=VALUE` arguments over the scope's defaults.
struct Run {
    prefix: Option<String>,
    train: PathBuf,
    sequences: usize,
    frontier: PathBuf,
    passages: usize,
    context: usize,
    settings: Settings,
    /// Whether every site starts from VPD's subcomponents.
    vpd_start: bool,
    random: usize,
    vpd: PathBuf,
    vpd_sets: PathBuf,
}

impl Run {
    fn parse(scope: &str, [train, frontier, vpd, vpd_sets]: [PathBuf; 4], pairs: &[String]) -> Result<Self, String> {
        let smoke = scope == "smoke";
        let mut run = Self {
            prefix: match scope {
                "all" => None,
                "smoke" => Some("blocks.0.".to_string()),
                p => Some(p.to_string()),
            },
            train,
            sequences: if smoke { 2 } else { 4 },
            frontier,
            passages: if smoke { 2 } else { 32 },
            context: if smoke { 128 } else { 512 },
            settings: Settings {
                observations: if smoke { 1e5 } else { 1e6 },
                rounds: if smoke { 2 } else { 50 },
                blocks: true,
                draws: if smoke { 1 } else { 4 },
                seed: 0x517E,
            },
            vpd_start: false,
            random: if smoke { 4 } else { 64 },
            vpd,
            vpd_sets,
        };
        for pair in pairs {
            let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair}: not KEY=VALUE"))?;
            let count = || value.parse::<usize>().map_err(|e| format!("{key}: {e}"));
            match key {
                "sequences" => run.sequences = count()?,
                "passages" => run.passages = count()?,
                "context" => run.context = count()?,
                "n" => run.settings.observations = value.parse().map_err(|e| format!("n: {e}"))?,
                "rounds" => run.settings.rounds = count()?,
                "blocks" => run.settings.blocks = count()? != 0,
                "start" => {
                    run.vpd_start = match value {
                        "own" => false,
                        "vpd" => true,
                        other => return Err(format!("start: {other} is neither own nor vpd")),
                    }
                }
                "draws" => run.settings.draws = count()?,
                "random" => run.random = count()?,
                "device" => gam_mpd::core_device::choose(gam_mpd::core_device::Choice::parse(value)?),
                other => return Err(format!("unknown key {other}")),
            }
        }
        Ok(run)
    }

    /// What a fitted site's files must have been fitted with to be read back.
    fn fitted_with(&self) -> Value {
        let s = &self.settings;
        json!({"train": self.train, "sequences": self.sequences, "context": self.context, "n": s.observations, "rounds": s.rounds,
               "blocks": s.blocks, "draws": s.draws, "seed": s.seed, "start": if self.vpd_start { "vpd" } else { "own" }})
    }
}

/// A fitted site's files in `dir`.
fn save(dir: &Path, fitted: &Fitted, with: &Value) -> Result<(), String> {
    let name = &fitted.site.name;
    write_f64(&dir.join(format!("{name}.v.f64")), &fitted.library.v)?;
    write_f64(&dir.join(format!("{name}.u.f64")), &fitted.library.u)?;
    write_f64(&dir.join(format!("{name}.fisher.f64")), &fitted.fisher)?;
    write_f64(&dir.join(format!("{name}.moment.f64")), &fitted.second_moment)?;
    let record = json!({"site": name, "ranks": fitted.ranks, "bits": fitted.bits, "fitted_with": with});
    std::fs::write(dir.join(format!("{name}.json")), record.to_string()).map_err(|e| e.to_string())
}

/// A fitted site read back from `dir`, when its files are there and were fitted `with` these
/// settings (selected at `n = observations`).
fn load(dir: &Path, model: &OperatorProgram, site: &Site, with: &Value, observations: f64) -> Result<Option<Fitted>, String> {
    let name = &site.name;
    let Ok(text) = std::fs::read_to_string(dir.join(format!("{name}.json"))) else { return Ok(None) };
    let record: Value = serde_json::from_str(&text).map_err(|e| format!("{name}.json: {e}"))?;
    if &record["fitted_with"] != with {
        eprintln!("{name}: fitted with {}, refitting", record["fitted_with"]);
        return Ok(None);
    }
    let w = matrix(model, site)?;
    let (d_out, d_in) = w.dim();
    let read = |side: &str, cols: usize| read_f64_matrix(&dir.join(format!("{name}.{side}.f64")), cols);
    let library = Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) };
    let ranks: Vec<usize> = serde_json::from_value(record["ranks"].clone()).map_err(|e| format!("{name} ranks: {e}"))?;
    let bits: Vec<f64> = serde_json::from_value(record["bits"].clone()).map_err(|e| format!("{name} bits: {e}"))?;
    Ok(Some(Fitted::new(site.clone(), w, (library, ranks, bits), (read("fisher", d_out)?, read("moment", d_in)?), observations)?))
}

/// VPD's subcomponents at `site`.
fn vpd_library(run: &Run, model: &OperatorProgram, site: &Site) -> Result<Library, String> {
    let (d_out, d_in) = matrix(model, site)?.dim();
    let read = |side: &str, cols: usize| read_f64_matrix(&run.vpd.join(format!("{}.{side}.f64", site.name)), cols);
    Ok(Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) })
}

/// VPD's published decomposition on `scope` (in execution order): its library and, per passage, its
/// published sets (the first `context` positions of each of the first `passages` sequences).
fn vpd(run: &Run, model: &OperatorProgram, scope: &[Site]) -> Result<Given, String> {
    let listed = std::fs::read_to_string(run.vpd_sets.join("sites.txt")).map_err(|e| format!("{}: {e}", run.vpd_sets.display()))?;
    let mut offsets = BTreeMap::new();
    let mut total = 0usize;
    for line in listed.lines() {
        let (name, count) = line.split_once(' ').ok_or_else(|| format!("sites.txt: {line}"))?;
        offsets.insert(name.to_string(), (total, count.parse::<usize>().map_err(|e| format!("sites.txt: {e}"))?));
        total += offsets[name].1;
    }
    let indptr = read_i64(&run.vpd_sets.join("indptr.i64"))?;
    let indices = read_i64(&run.vpd_sets.join("indices.i64"))?;
    let (mut libraries, mut ranks) = (Vec::new(), Vec::new());
    let mut masks: Vec<Vec<Array2<f64>>> = vec![Vec::new(); run.passages];
    for site in scope {
        let &(offset, count) = offsets.get(&site.name).ok_or_else(|| format!("{}: not in VPD's sets", site.name))?;
        let library = vpd_library(run, model, site)?;
        if library.v.nrows() != count || library.u.nrows() != count {
            return Err(format!("{}: {} read and {} write vectors for {count} subcomponents", site.name, library.v.nrows(), library.u.nrows()));
        }
        for (p, per_site) in masks.iter_mut().enumerate() {
            let mut m = Array2::<f64>::zeros((run.context, count));
            for r in 0..run.context {
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
        libraries.push(library);
    }
    Ok(Given { sites: scope.to_vec(), libraries, ranks, masks })
}

fn stats(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n == 0 {
        return Value::Null;
    }
    json!({"mean": sorted.iter().sum::<f64>() / n as f64, "median": sorted[n / 2], "max": sorted[n - 1], "count": n})
}

/// Stage (a) and (d) of one replacement: every site replaced, its per-token KL, and the bits of
/// the blocks that ran (`bits` per site and block, `ranks` their widths).
fn replaced_and_bits(model: &OperatorProgram, replacement: &dyn Replacement, passages: &[Passage], bits: &[Vec<f64>], ranks: &[Vec<usize>]) -> Result<Value, String> {
    let names: Vec<String> = replacement.sites().iter().map(|s| s.name.clone()).collect();
    let members: Vec<usize> = (0..names.len()).collect();
    let runs = replaced(model, replacement, &members, passages)?;
    let every: Vec<f64> = runs.iter().flat_map(|(kl, _)| kl.iter().copied()).collect();
    let per_passage: Vec<f64> = runs.iter().map(|(kl, _)| kl.mean().unwrap_or(0.0)).collect();
    let words = every.len() as f64;
    let mut per_word = Vec::new();
    // Per site: blocks and columns on, and their bits, per word.
    let mut on = vec![(0.0, 0.0, 0.0); names.len()];
    for (_, masks) in &runs {
        per_word.extend(bits_per_word(masks, bits)?.iter().copied());
        for (k, m) in masks.iter().enumerate() {
            let widths = Array1::from_iter(ranks[k].iter().map(|r| *r as f64));
            on[k].0 += m.sum() / words;
            on[k].1 += m.dot(&widths).sum() / words;
            on[k].2 += m.dot(&Array1::from(bits[k].clone())).sum() / words;
        }
    }
    let library_once: f64 = bits.iter().map(|b| b.iter().sum::<f64>()).sum();
    let per_site: BTreeMap<String, Value> = names
        .iter()
        .enumerate()
        .map(|(k, n)| (n.clone(), json!({"blocks": bits[k].len(), "blocks_on_per_word": on[k].0, "columns_on_per_word": on[k].1,
                                         "bits_per_word": on[k].2, "library_bits": bits[k].iter().sum::<f64>()})))
        .collect();
    Ok(json!({
        "e2e": {"kl_per_token": stats(&every), "per_passage": per_passage},
        "bits": {"description_per_word": stats(&per_word), "library_once": library_once,
                 "library_per_word_over_passages": library_once / words, "blocks_on_per_word": on.iter().map(|o| o.0).sum::<f64>(),
                 "columns_on_per_word": on.iter().map(|o| o.1).sum::<f64>(), "per_site": per_site},
    }))
}

fn switch_summary(names: &[String], attack: &SiteSwitch) -> Value {
    let words: Vec<f64> = attack.word_worst.iter().flatten().copied().collect();
    let worst_named: Vec<Vec<&str>> = attack.worst_subset.iter().map(|s| s.iter().map(|k| names[*k].as_str()).collect()).collect();
    let alone: BTreeMap<&str, f64> = names.iter().zip(&attack.alone).map(|(n, a)| (n.as_str(), a.iter().sum::<f64>() / a.len().max(1) as f64)).collect();
    json!({"exhaustive": attack.exhaustive, "forwards": attack.forwards, "all_replaced": stats(&attack.all_replaced),
           "passage_worst_subset": stats(&attack.worst), "word_worst_subset": stats(&words), "site_alone_mean_kl": alone,
           "per_passage": {"all_replaced": attack.all_replaced, "worst": attack.worst, "worst_subset": worst_named}})
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_e2e_2951 SCOPE OUT_DIR TRAIN FRONTIER VPD_LIBRARY VPD_SETS [KEY=VALUE ...]";
    if args.len() < 7 {
        return Err(usage.to_string());
    }
    let scope = &args[1];
    let out = PathBuf::from(&args[2]);
    let run = Run::parse(scope, [3, 4, 5, 6].map(|i| PathBuf::from(&args[i])), &args[7..])?;
    let library_dir = out.join("library");
    std::fs::create_dir_all(&library_dir).map_err(|e| format!("{}: {e}", library_dir.display()))?;
    let started = Instant::now();
    let mut seconds = BTreeMap::<&str, f64>::new();
    let training = import_language_model(&run.train, run.sequences, run.context)?;
    let model = &training.program;
    let frontier = import_language_model(&run.frontier, run.passages, run.context)?.contract.family;
    let chosen = in_execution_order(sites(model).into_iter().filter(|s| run.prefix.as_ref().is_none_or(|p| s.name.starts_with(p.as_str()))).collect());
    if chosen.is_empty() {
        return Err(format!("no site of the model starts with {:?}", run.prefix));
    }
    let names: Vec<String> = chosen.iter().map(|s| s.name.clone()).collect();
    eprintln!("scope {scope}: {} sites {names:?}", names.len());
    let with = run.fitted_with();
    let resumed: Option<Value> = std::fs::read_to_string(out.join("e2e.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .filter(|r| r["fitted_with"] == with && r["passages"] == json!(run.passages) && r["random"] == json!(run.random));
    let mut report = json!({
        "fitted_with": with,
        "scope": scope, "sites": names, "train": run.train, "frontier": run.frontier, "sequences": run.sequences, "passages": run.passages,
        "context": run.context, "observations": run.settings.observations, "rounds": run.settings.rounds, "blocks": run.settings.blocks,
        "draws": run.settings.draws, "random": run.random, "vpd_library": run.vpd, "vpd_sets": run.vpd_sets, "start": if run.vpd_start { "vpd" } else { "own" },
        "pricing": "gam_mpd::describe::Structured in each site's fit statistics (its training inputs under the explanation upstream), both libraries",
        "selection": "ours: gam_mpd::site_fit::Selector on the explanation's own reads in the site's mean training Fisher; vpd: its published sets (model states, residual off)",
    });
    let write = |report: &Value| -> Result<(), String> {
        let text = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
        let partial = out.join("e2e.json.partial");
        std::fs::write(&partial, text).map_err(|e| e.to_string())?;
        std::fs::rename(&partial, out.join("e2e.json")).map_err(|e| e.to_string())
    };

    for key in ["ours", "vpd"] {
        if let Some(earlier) = resumed.as_ref().map(|r| &r[key]).filter(|v| v.is_object()) {
            eprintln!("{key}: stages {:?} read back", earlier.as_object().map(|o| o.keys().collect::<Vec<_>>()));
            report[key] = earlier.clone();
        }
    }

    // Fit, in execution order on hybrid inputs.
    let clock = Instant::now();
    let batches: Vec<_> = (0..run.sequences).map(|s| training.contract.family.select(&(s * run.context..(s + 1) * run.context).collect::<Vec<_>>())).collect();
    let starts: BTreeMap<String, Library> = if run.vpd_start {
        chosen.iter().map(|site| Ok((site.name.clone(), vpd_library(&run, model, site)?))).collect::<Result<_, String>>()?
    } else {
        BTreeMap::new()
    };
    let explanation: Explanation = fit(model, chosen.clone(), &batches, &run.settings, &starts, |site| load(&library_dir, model, site, &with, run.settings.observations), |f| {
        save(&library_dir, f, &with)
    })?;
    drop(batches);
    seconds.insert("fit", clock.elapsed().as_secs_f64());
    report["fit"] = explanation
        .sites
        .iter()
        .map(|f| {
            let mut histogram = BTreeMap::<usize, usize>::new();
            for r in &f.ranks {
                *histogram.entry(*r).or_default() += 1;
            }
            (f.site.name.clone(), json!({"columns": f.library.v.nrows(), "blocks": f.ranks.len(), "ranks": histogram, "library_bits": f.bits.iter().sum::<f64>()}))
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
    write(&report)?;

    // The passages, the model's own logits on them, and VPD's decomposition of the same sites.
    let clock = Instant::now();
    let bases: Vec<FamilyInputs> = (0..run.passages).map(|p| frontier.select(&(p * run.context..(p + 1) * run.context).collect::<Vec<_>>())).collect();
    let passages: Vec<Passage> = match gam_mpd::core_device::device()? {
        Some(device) => gam_mpd::core_device::passages(&device, model, bases)?,
        None => bases.into_par_iter().map(|b| Passage::new(model, b)).collect::<Result<_, _>>()?,
    };
    seconds.insert("passages", clock.elapsed().as_secs_f64());
    let clock = Instant::now();
    let given = vpd(&run, model, &chosen)?;
    // VPD's subcomponents priced as ours, in each site's fit statistics (read back when priced at
    // these settings).
    let vpd_bits: Vec<Vec<f64>> = explanation
        .sites
        .iter()
        .zip(&given.libraries)
        .map(|(f, library)| {
            let path = library_dir.join(format!("{}.vpd_bits.json", f.site.name));
            let earlier: Option<Vec<f64>> = std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str::<Value>(&text).ok())
                .filter(|r| r["fitted_with"] == with)
                .and_then(|r| serde_json::from_value(r["bits"].clone()).ok())
                .filter(|b: &Vec<f64>| b.len() == library.v.nrows());
            if let Some(bits) = earlier {
                return Ok(bits);
            }
            // Started from VPD's subcomponents with no rounds, our library is VPD's, priced alike.
            if run.vpd_start && run.settings.rounds == 0 && f.ranks.iter().all(|r| *r == 1) && f.library.v == library.v && f.library.u == library.u {
                return Ok(f.bits.clone());
            }
            let describe = f.description(model, run.settings.observations)?;
            let columns: Vec<usize> = (0..library.v.nrows()).collect();
            let bits = gam_mpd::combine::map(&columns, |&c| describe.bits(0, library.u.slice(s![c..c + 1, ..]), library.v.slice(s![c..c + 1, ..])))
                .into_iter()
                .collect::<Result<Vec<f64>, String>>()?;
            std::fs::write(&path, json!({"fitted_with": with, "bits": bits}).to_string()).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(bits)
        })
        .collect::<Result<_, String>>()?;
    seconds.insert("vpd_pricing", clock.elapsed().as_secs_f64());
    eprintln!("{} passages and VPD's decomposition priced, {:.0}s", passages.len(), clock.elapsed().as_secs_f64());

    // (a), (d): every site replaced.
    let clock = Instant::now();
    let our_bits: Vec<Vec<f64>> = explanation.sites.iter().map(|f| f.bits.clone()).collect();
    let our_ranks: Vec<Vec<usize>> = explanation.sites.iter().map(|f| f.ranks.clone()).collect();
    for (key, replacement, bits, ranks) in [("ours", &explanation as &dyn Replacement, &our_bits, &our_ranks), ("vpd", &given as &dyn Replacement, &vpd_bits, &given.ranks)] {
        if report[key]["e2e"].is_null() {
            let clock = Instant::now();
            let scored = replaced_and_bits(model, replacement, &passages, bits, ranks)?;
            report[key]["e2e"] = scored["e2e"].clone();
            report[key]["bits"] = scored["bits"].clone();
            seconds.insert(if key == "ours" { "replaced_ours" } else { "replaced_vpd" }, clock.elapsed().as_secs_f64());
        }
    }
    seconds.insert("replaced", clock.elapsed().as_secs_f64());
    eprintln!(
        "every site replaced: KL per token ours {}, VPD {}; description bits per word ours {}, VPD {}",
        report["ours"]["e2e"]["kl_per_token"]["mean"],
        report["vpd"]["e2e"]["kl_per_token"]["mean"],
        report["ours"]["bits"]["description_per_word"]["mean"],
        report["vpd"]["bits"]["description_per_word"]["mean"]
    );
    write(&report)?;

    // (b): the site-switch claim.
    let clock = Instant::now();
    for (key, replacement) in [("ours", &explanation as &dyn Replacement), ("vpd", &given as &dyn Replacement)] {
        if !report[key]["site_switch"].is_null() {
            continue;
        }
        let clock = Instant::now();
        let attack = site_switch(model, replacement, &passages, run.random, run.settings.seed)?;
        seconds.insert(if key == "ours" { "site_switch_ours" } else { "site_switch_vpd" }, clock.elapsed().as_secs_f64());
        report[key]["site_switch"] = switch_summary(&names, &attack);
        eprintln!("site switch {key}: worst subset per passage {}", report[key]["site_switch"]["passage_worst_subset"]);
        write(&report)?;
    }
    seconds.insert("site_switch", clock.elapsed().as_secs_f64());

    seconds.insert("total", started.elapsed().as_secs_f64());
    report["seconds"] = json!(seconds);
    write(&report)?;
    eprintln!("{}", serde_json::to_string_pretty(&json!({"ours": report["ours"]["e2e"], "vpd": report["vpd"]["e2e"]})).unwrap_or_default());
    Ok(())
}

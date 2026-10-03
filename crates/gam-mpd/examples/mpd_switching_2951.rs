//! The simplest causal switching functions that reproduce each site's selected sets (#2951,
//! `gam_mpd::gates` on `gam_mpd::site_fit::measure`).
//!
//! `mpd_switching_2951 EXPORT_DIR LIBRARY_DIR OUT.json OBSERVATIONS TRAIN EVAL [DRAWS] [SITES] [CONTEXT]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`): its first
//! `TRAIN` sequences of `CONTEXT` positions (default 512) fit the switching functions and the next
//! `EVAL` score them. `LIBRARY_DIR` holds each site's library (`{site}.v.f64`, `{site}.u.f64`, raw
//! float64, as `mpd_site_fit_2951` writes them); `SITES` (comma-separated, default every site with
//! a library) picks the sites. Each site's inputs are gathered on both runs of sequences
//! (`gam_mpd::site_fit::samples`, `DRAWS` sampled-label passes per sequence, default 4), its
//! subcomponents priced as `mpd_site_fit_2951` prices them (in the training statistics), and every
//! input's sets selected by the site's code at `n = OBSERVATIONS` (`gam_mpd::site_fit::measure`).
//!
//! Each subcomponent then gets the cheapest switching function of its own amplitude `a = v·x` on
//! the training inputs (`gam_mpd::gates::best`), as the amplitude or as its magnitude `|a|`, the
//! smaller total kept: its own bits plus the escaped listing of the selected sets. A magnitude
//! function with no units is the one-line rule "on when `|v·x| > τ`". On the eval inputs every
//! function decides from the amplitude alone, and `OUT.json` gets per site and in total: the
//! functions' kinds and own bits, their decisions' disagreement with the selected sets per input,
//! both L0s, and the bits per input of listing the selected sets under the functions (the
//! corrections that turn the functions' sets into the selection's) beside the same under each
//! subcomponent's base rate. `OUT.switches.json` holds the functions per site.

use gam_mpd::gates::{self, Feature, Switch};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, matrix, sites};
use gam_mpd::site_fit::{measure, samples};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde_json::json;
use std::path::PathBuf;

fn read_library(dir: &std::path::Path, name: &str, d_in: usize, d_out: usize) -> Result<Library, String> {
    let read = |side: &str, cols: usize| -> Result<Array2<f64>, String> {
        let path = dir.join(format!("{name}.{side}.f64"));
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
        Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
    };
    Ok(Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) })
}

/// Every input's amplitudes `x · Vᵀ` (inputs × subcomponents).
fn amplitudes(reads: &Array2<f32>, library: &Library) -> Array2<f64> {
    reads.mapv(f64::from).dot(&library.v.t())
}

/// Per subcomponent, its on-labels over the inputs.
fn labels(sets: &[Vec<u32>], pieces: usize) -> Vec<Vec<bool>> {
    let mut on = vec![vec![false; sets.len()]; pieces];
    for (t, set) in sets.iter().enumerate() {
        for c in set {
            on[*c as usize][t] = true;
        }
    }
    on
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_switching_2951 EXPORT_DIR LIBRARY_DIR OUT.json OBSERVATIONS TRAIN EVAL [DRAWS] [SITES] [CONTEXT]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let library_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let train: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let eval: usize = args.get(6).ok_or(usage)?.parse().map_err(|e| format!("EVAL: {e}"))?;
    let draws: usize = args.get(7).map_or(Ok(4), |v| v.parse()).map_err(|e| format!("DRAWS: {e}"))?;
    let wanted: Option<Vec<String>> = args.get(8).filter(|s| *s != "all").map(|s| s.split(',').map(str::to_string).collect());
    let context: usize = args.get(9).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let imported = import_language_model(&export, train + eval, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let chosen: Vec<_> = sites(model)
        .into_iter()
        .filter(|s| wanted.as_ref().is_none_or(|w| w.contains(&s.name)) && library_dir.join(format!("{}.v.f64", s.name)).exists())
        .collect();
    if chosen.is_empty() {
        return Err(format!("no site of {wanted:?} has a library in {}", library_dir.display()));
    }
    let started = std::time::Instant::now();
    let sequence = |s: usize| family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>());
    let fitted_on = samples(model, &chosen, (0..train).map(sequence), draws, 0x517E)?;
    let scored_on = samples(model, &chosen, (train..train + eval).map(sequence), draws, 0x5C0E)?;
    eprintln!("samples of {} sites on {train} + {eval} sequences, {:.0}s", chosen.len(), started.elapsed().as_secs_f64());
    let maps: Vec<Array2<f64>> = chosen.iter().map(|site| matrix(model, site)).collect::<Result<_, _>>()?;
    // Subcomponents priced as the site fit prices them, in the training statistics.
    let description = gam_mpd::describe::Structured::new(
        chosen
            .iter()
            .zip(maps.iter().zip(&fitted_on))
            .map(|(site, (w, g))| {
                let measured = gam_mpd::pieces::Site { w: w.clone(), second_moment: g.second_moment.clone(), mean: Array1::zeros(w.ncols()), fisher: g.fisher.clone() };
                let (writers, readers) = gam_mpd::describe::declared_charts(model, site)?;
                gam_mpd::describe::Geometry::new(gam_mpd::describe::Metric::of(&measured, observations), writers, readers)
            })
            .collect::<Result<_, String>>()?,
    );
    let mut per_site = Vec::new();
    let mut functions = Vec::new();
    let eval_inputs = eval * context;
    let (mut listing_total, mut base_total, mut wrong_total, mut function_total) = (0.0, 0.0, 0usize, 0.0);
    for (k, site) in chosen.iter().enumerate() {
        let (d_out, d_in) = maps[k].dim();
        let library = read_library(&library_dir, &site.name, d_in, d_out)?;
        let pieces = library.v.nrows();
        let (train_round, train_sets) = measure(k, &maps[k], &fitted_on[k], &description, observations, &library)?;
        let (eval_round, eval_sets) = measure(k, &maps[k], &scored_on[k], &description, observations, &library)?;
        let (train_a, eval_a) = (amplitudes(&fitted_on[k].reads, &library), amplitudes(&scored_on[k].reads, &library));
        let (train_y, eval_y) = (labels(&train_sets, pieces), labels(&eval_sets, pieces));
        // Each subcomponent's cheapest function of its own amplitude, as itself or its magnitude.
        let switches: Vec<Switch> = (0..pieces)
            .into_par_iter()
            .map(|c| {
                let column = train_a.column(c);
                let signed = Feature { site: k, piece: c, lag: 0, magnitude: false };
                let magnitude = Feature { magnitude: true, ..signed };
                let fit_on = |f: Feature| {
                    let x = Array2::from_shape_fn((column.len(), 1), |(t, _)| f.value(column[t]));
                    gates::best(x.view(), &train_y[c], &[f], 0.0)
                };
                let (by_size, by_sign) = (fit_on(magnitude), fit_on(signed));
                if by_size.total_bits() <= by_sign.total_bits() { by_size } else { by_sign }
            })
            .collect();
        // Their decisions on the eval inputs, and the listing of the selected sets under them.
        let scored: Vec<(f64, f64, Vec<bool>)> = (0..pieces)
            .into_par_iter()
            .map(|c| {
                let switch = &switches[c];
                let column = eval_a.column(c);
                let x = Array2::from_shape_fn((column.len(), switch.features.len()), |(t, i)| switch.features[i].value(column[t]));
                let decided: Vec<bool> = (0..column.len()).map(|t| switch.on(&x.row(t).to_vec())).collect();
                let rate = gates::base(&train_y[c]);
                let none = Array2::<f64>::zeros((column.len(), 0));
                (switch.label_bits(x.view(), &eval_y[c]), rate.label_bits(none.view(), &eval_y[c]), decided)
            })
            .collect();
        let inputs = eval_sets.len();
        let listing: f64 = scored.iter().map(|s| s.0).sum();
        let base: f64 = scored.iter().map(|s| s.1).sum();
        let wrong: usize = scored.iter().zip(&eval_y).map(|(s, y)| s.2.iter().zip(y).filter(|(a, b)| a != b).count()).sum();
        let decided_l0 = scored.iter().map(|s| s.2.iter().filter(|o| **o).count()).sum::<usize>() as f64 / inputs as f64;
        let own: f64 = switches.iter().map(|s| s.function_bits).sum();
        let mut kinds = std::collections::BTreeMap::<String, usize>::new();
        for s in &switches {
            let kind = match (s.features.first(), s.threshold()) {
                (None, _) => (if s.beta > 0.0 { "always on" } else { "always off" }).to_string(),
                (Some(_), Some(_)) => "|a| > τ".to_string(),
                (Some(f), None) => format!("{} {} units", if f.magnitude { "|a|" } else { "a" }, s.units.len()),
            };
            *kinds.entry(kind).or_default() += 1;
        }
        eprintln!(
            "{}: {pieces} subcomponents {kinds:?}; eval: selected L0 {:.2}, switched L0 {decided_l0:.2}, {:.2} disagreements per input, listing {:.1} bits per input (base rates {:.1}); own bits {own:.0}, {:.0}s",
            site.name,
            eval_round.l0,
            wrong as f64 / inputs as f64,
            listing / inputs as f64,
            base / inputs as f64,
            started.elapsed().as_secs_f64()
        );
        per_site.push(json!({
            "site": site.name, "pieces": pieces, "kinds": kinds, "function_bits": own,
            "train": {"code": train_round.code, "l0": train_round.l0},
            "eval": {"code": eval_round.code, "description": eval_round.description, "error": eval_round.error, "l0_selected": eval_round.l0,
                "l0_switched": decided_l0, "disagreements_per_input": wrong as f64 / inputs as f64,
                "listing_bits_per_input": listing / inputs as f64, "base_rate_bits_per_input": base / inputs as f64},
            "thresholds": switches.iter().map(|s| s.threshold()).collect::<Vec<_>>(),
        }));
        functions.push(json!({"name": site.name, "switches": switches}));
        listing_total += listing;
        base_total += base;
        wrong_total += wrong;
        function_total += own;
        let report = json!({
            "observations": observations, "train": train, "eval": eval, "sites": per_site,
            "total": {"function_bits": function_total, "listing_bits_per_input": listing_total / eval_inputs as f64,
                "base_rate_bits_per_input": base_total / eval_inputs as f64, "disagreements_per_input": wrong_total as f64 / eval_inputs as f64},
        });
        std::fs::write(&out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        std::fs::write(out.with_extension("switches.json"), serde_json::to_string(&functions).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    Ok(())
}

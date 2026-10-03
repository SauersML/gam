//! A from-scratch library trained through the model's own masked forward (#2951).
//!
//! `mpd_e2e_train_2951 EXPORT_DIR LIBRARY_DIR OUT_DIR OBSERVATIONS TRAIN [PASSES] [CONTEXT]`
//!
//! `LIBRARY_DIR` holds a library per site (`{site}.v.f64`, `{site}.u.f64`, as
//! `mpd_site_fit_2951` writes them; a site without files stays native). Every sequence of the
//! export's first `TRAIN` (of `CONTEXT` positions, default 512) is coded in turn, `PASSES` times
//! (default 1): its sets are each site's own code's selection on the sequence's clean reads
//! (`gam_mpd::site_fit::measure`, the description `gam_mpd::blocks::Generic` in the model's
//! statistics on the training sequences), and the library takes one exact-gradient step of the
//! masked forward's KL under the box claim on those sets (`gam_mpd::masked::step_pieces`, every
//! piece on still the map). The sets are the switching function; the step makes the library agree
//! with it where the whole model, not one site, measures the error. After every sequence the
//! library goes to `OUT_DIR/{site}.{v,u}.f64` and the step's totals to `OUT_DIR/steps.json`.

use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Claim, Library, Masked, Running, Target, matrix, sites, step_pieces};
use gam_mpd::site_fit::{measure, samples};
use ndarray::{Array1, Array2};
use serde_json::json;
use std::path::{Path, PathBuf};

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_f64(path: &Path, m: &Array2<f64>) -> Result<(), String> {
    let bytes: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
    let partial = path.with_extension("partial");
    std::fs::write(&partial, bytes).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, path).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_e2e_train_2951 EXPORT_DIR LIBRARY_DIR OUT_DIR OBSERVATIONS TRAIN [PASSES] [CONTEXT]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let given = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let train: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let passes: usize = args.get(6).map_or(Ok(1), |v| v.parse()).map_err(|e| format!("PASSES: {e}"))?;
    let context: usize = args.get(7).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let imported = import_language_model(&export, train, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let sequence = |s: usize| family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>());
    let (mut chosen, mut libraries) = (Vec::new(), Vec::new());
    for site in sites(model) {
        let v_path = given.join(format!("{}.v.f64", site.name));
        if !v_path.exists() {
            continue;
        }
        let w = matrix(model, &site)?;
        let (d_out, d_in) = w.dim();
        libraries.push(Library { v: read_f64(&v_path, d_in)?, u: read_f64(&given.join(format!("{}.u.f64", site.name)), d_out)?, mean: Array1::zeros(d_in) });
        chosen.push(site);
    }
    if chosen.is_empty() {
        return Err(format!("{}: no library", given.display()));
    }
    // The description, in the model's statistics on the training sequences.
    let started = std::time::Instant::now();
    let measured = samples(model, &chosen, (0..train).map(sequence), 2, 0xDE5C)?;
    let maps: Vec<Array2<f64>> = chosen.iter().map(|site| matrix(model, site)).collect::<Result<_, _>>()?;
    let statistics: Vec<gam_mpd::pieces::Site> = maps
        .iter()
        .zip(&measured)
        .map(|(w, m)| gam_mpd::pieces::Site { w: w.clone(), second_moment: m.second_moment.clone(), mean: Array1::zeros(w.ncols()), fisher: m.fisher.clone() })
        .collect();
    let description = gam_mpd::blocks::Generic::new(&statistics, observations);
    drop((statistics, measured));
    eprintln!("statistics of {} sites on {train} sequences, {:.0}s", chosen.len(), started.elapsed().as_secs_f64());
    let mut masked = Masked::build(model, chosen.clone(), libraries)?;
    let mut running = Running::default();
    let mut log = Vec::new();
    for pass in 0..passes {
        for s in 0..train {
            let started = std::time::Instant::now();
            let inputs = sequence(s);
            let target = Target::every_row(model.execute(&inputs, false).map_err(|e| e.to_string())?.values[model.output].clone());
            // The switching function: each site's own code's sets on this sequence.
            let local = samples(model, &chosen, std::iter::once(inputs.clone()), 2, 0x5E7 + (pass * train + s) as u64)?;
            let mut masks = Vec::new();
            let mut l0 = 0.0;
            for (k, sample) in local.iter().enumerate() {
                let library = masked.library(k)?;
                let (_, sets) = measure(k, &maps[k], sample, &description, observations, &library)?;
                let mut mask = Array2::<f64>::zeros((inputs.rows, library.v.nrows()));
                for (r, on) in sets.iter().enumerate() {
                    for &c in on {
                        mask[[r, c as usize]] = 1.0;
                    }
                }
                l0 += mask.sum() / inputs.rows as f64;
                masks.push(mask);
            }
            let step = step_pieces(&mut masked, &inputs, &target, &masks, 2, 0xF00D + (pass * train + s) as u64, &mut running, Claim::Box)?;
            let rows = inputs.rows as f64;
            eprintln!(
                "pass {pass} sequence {s}: L0 {l0:.1}, box KL {:?} per token, {:.0}s",
                step.map(|(b, a)| (b / rows, a / rows)),
                started.elapsed().as_secs_f64()
            );
            log.push(json!({"pass": pass, "sequence": s, "l0": l0, "before": step.map(|(b, _)| b / rows), "after": step.map(|(_, a)| a / rows)}));
            for (k, site) in chosen.iter().enumerate() {
                let library = masked.library(k)?;
                write_f64(&out.join(format!("{}.v.f64", site.name)), &library.v)?;
                write_f64(&out.join(format!("{}.u.f64", site.name)), &library.u)?;
            }
            std::fs::write(out.join("steps.json"), json!({"observations": observations, "steps": log}).to_string()).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

//! Libraries fitted on each site's own inputs (#2951, `gam_mpd::site_fit`).
//!
//! `mpd_site_fit_2951 EXPORT_DIR OUT_DIR OBSERVATIONS TRAIN [DRAWS] [ROUNDS] [SITES] [CONTEXT] [library:DIR]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`) whose first
//! `TRAIN` sequences of `CONTEXT` positions (default 512) are the inputs. Every site of the model
//! (`gam_mpd::masked::sites`), or those named in `SITES` (comma-separated, `all` for every one),
//! gets its reads, sensitivities and written Fisher from `DRAWS` sampled-label reverse passes per
//! sequence (default 4, `gam_mpd::site_fit::samples`), and a library of `d_in + d_out`
//! subcomponents fitted to its code at `n = OBSERVATIONS` for at most `ROUNDS` rounds (default 50,
//! `gam_mpd::site_fit::fit`), each subcomponent described by `gam_mpd::blocks::Generic` in those
//! statistics. After every round the library goes to `OUT_DIR/{site}.v.f64` (pieces × d_in) and
//! `OUT_DIR/{site}.u.f64` (pieces × d_out), raw float64, the `library:DIR` start of
//! `mpd_pieces_masked_2951`, and its rounds to `OUT_DIR/{site}.rounds.json`.
//!
//! With `library:DIR` nothing is fitted: each site's given library (`DIR/{site}.{v,u}.f64`, as
//! above) is measured under the same code (`gam_mpd::site_fit::measure`, its sets selected from all
//! on), and the measurement goes to `OUT_DIR/{site}.measure.json`.

use gam_mpd::import::import_language_model;
use gam_mpd::masked::{matrix, sites};
use gam_mpd::masked::Library;
use gam_mpd::site_fit::{Settings, fit, measure, samples};
use ndarray::{Array1, Array2};
use serde_json::json;
use std::path::PathBuf;

fn write_f64(path: &PathBuf, m: &Array2<f64>) -> Result<(), String> {
    let bytes: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
    let partial = path.with_extension("partial");
    std::fs::write(&partial, bytes).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, path).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_site_fit_2951 EXPORT_DIR OUT_DIR OBSERVATIONS TRAIN [DRAWS] [ROUNDS] [SITES] [CONTEXT] [library:DIR]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let observations: f64 = args.get(3).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let train: usize = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let draws: usize = args.get(5).map_or(Ok(4), |v| v.parse()).map_err(|e| format!("DRAWS: {e}"))?;
    let rounds: usize = args.get(6).map_or(Ok(50), |v| v.parse()).map_err(|e| format!("ROUNDS: {e}"))?;
    let wanted: Option<Vec<String>> = args.get(7).filter(|s| *s != "all").map(|s| s.split(',').map(str::to_string).collect());
    let context: usize = args.get(8).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let given = match args.get(9) {
        Some(a) => Some(PathBuf::from(a.strip_prefix("library:").ok_or(usage)?)),
        None => None,
    };
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let imported = import_language_model(&export, train, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let chosen: Vec<_> = sites(model).into_iter().filter(|s| wanted.as_ref().is_none_or(|w| w.contains(&s.name))).collect();
    if chosen.is_empty() {
        return Err(format!("no site matches {wanted:?}"));
    }
    let started = std::time::Instant::now();
    let sequence = |s: usize| family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>());
    let gathered = samples(model, &chosen, (0..train).map(sequence), draws, 0x517E)?;
    eprintln!("samples of {} sites on {train} sequences, {draws} draws each, {:.0}s", chosen.len(), started.elapsed().as_secs_f64());
    let maps: Vec<Array2<f64>> = chosen.iter().map(|site| matrix(model, site)).collect::<Result<_, _>>()?;
    // Each word pays for the weights that ran on it, described in these statistics (the reads
    // uncentred, as the masked program reads them).
    let statistics: Vec<gam_mpd::pieces::Site> = maps
        .iter()
        .zip(&gathered)
        .map(|(w, g)| gam_mpd::pieces::Site { w: w.clone(), second_moment: g.second_moment.clone(), mean: Array1::zeros(w.ncols()), fisher: g.fisher.clone() })
        .collect();
    let description = gam_mpd::blocks::Generic::new(&statistics, observations);
    drop(statistics);
    for (k, ((site, w), sample)) in chosen.iter().zip(&maps).zip(&gathered).enumerate() {
        let (d_out, d_in) = w.dim();
        if let Some(dir) = &given {
            let read = |side: &str, cols: usize| -> Result<Array2<f64>, String> {
                let path = dir.join(format!("{}.{side}.f64", site.name));
                let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
                Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
            };
            let library = Library { v: read("v", d_in)?, u: read("u", d_out)?, mean: Array1::zeros(d_in) };
            let round = measure(k, w, sample, &description, observations, &library)?;
            eprintln!("{} given library of {}: code {:.1} bits per input (description {:.1}, error {:.1}), L0 {:.2}, corner {:.2}",
                site.name, library.v.nrows(), round.code, round.description, round.error, round.l0, round.corner_share);
            let record = json!({"site": site.name, "observations": observations, "pieces": library.v.nrows(), "code": round.code,
                "description": round.description, "error": round.error, "l0": round.l0, "corner_share": round.corner_share});
            std::fs::write(out.join(format!("{}.measure.json", site.name)), record.to_string()).map_err(|e| e.to_string())?;
            continue;
        }
        let settings = Settings { observations, pieces: d_in + d_out, rounds, seed: 0xF17 + k as u64 };
        let started = std::time::Instant::now();
        let mut log = Vec::new();
        let (v_path, u_path) = (out.join(format!("{}.v.f64", site.name)), out.join(format!("{}.u.f64", site.name)));
        let rounds_path = out.join(format!("{}.rounds.json", site.name));
        let library = fit(k, w, sample, &description, settings, |round, library| {
            eprintln!(
                "{} round {}: code {:.1} bits per input (description {:.1}, error {:.1}), L0 {:.2}, corner {:.2}, reseeded {}, read steps {}, {:.0}s",
                site.name,
                round.round,
                round.code,
                round.description,
                round.error,
                round.l0,
                round.corner_share,
                round.reseeded,
                round.read_steps,
                started.elapsed().as_secs_f64()
            );
            log.push(json!({"round": round.round, "code": round.code, "description": round.description, "error": round.error, "l0": round.l0,
                "corner_share": round.corner_share, "reseeded": round.reseeded, "read_steps": round.read_steps, "seconds": started.elapsed().as_secs_f64()}));
            let written = write_f64(&v_path, &library.v)
                .and_then(|_| write_f64(&u_path, &library.u))
                .and_then(|_| std::fs::write(&rounds_path, json!({"site": site.name, "observations": observations, "rounds": log}).to_string()).map_err(|e| e.to_string()));
            if let Err(e) = written {
                eprintln!("{}: {e}", site.name);
            }
        })?;
        write_f64(&v_path, &library.v)?;
        write_f64(&u_path, &library.u)?;
        eprintln!("{}: {} subcomponents written, {:.0}s", site.name, library.v.nrows(), started.elapsed().as_secs_f64());
    }
    Ok(())
}

//! One site's library and the sets its selection turns on at given reads (#2951, `gam_mpd::site_fit`).
//!
//! `mpd_site_sets_2951 EXPORT_DIR TRAIN SITE OBSERVATIONS LIBRARY_DIR [ROUNDS] [READS.f64 ...]`
//!
//! The site `SITE`'s statistics (reads, written Fisher, reads' second moment) come from the first
//! `TRAIN` sequences of the language-model export `EXPORT_DIR` (`gam_mpd::site_fit::samples`, 4
//! sampled-label draws per sequence), and its code at `n = OBSERVATIONS` is the site-fit driver's
//! (`mpd_site_fit_2951`): each subcomponent priced by `gam_mpd::describe::Tiered`. The library is
//! `LIBRARY_DIR/{SITE}.{v,u}.f64` when there; otherwise it is fitted on those statistics from the
//! units the site reads, as the driver fits it (`gam_mpd::site_fit::fit`, `d_in + d_out`
//! subcomponents, at most `ROUNDS` rounds, default 50), and written there with its rounds.
//!
//! Every input's sets are then selected under that one code (`gam_mpd::site_fit::measure`, from
//! all on; the sets depend on the input's read alone): the training inputs' to
//! `LIBRARY_DIR/{SITE}.train.sets.json`, and for each `READS.f64` (raw float64, inputs × d_in, the
//! site's reads at inputs of interest) to `LIBRARY_DIR/{stem}.sets.json`, with each input's code.

use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, matrix, sites};
use gam_mpd::operator_program::Node;
use gam_mpd::site_fit::{Samples, Settings, fit, measure, samples};
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
    let usage = "mpd_site_sets_2951 EXPORT_DIR TRAIN SITE OBSERVATIONS LIBRARY_DIR [ROUNDS] [READS.f64 ...]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let train: usize = args.get(2).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let name = args.get(3).ok_or(usage)?;
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let dir = PathBuf::from(args.get(5).ok_or(usage)?);
    let rounds: usize = args.get(6).map_or(Ok(50), |v| v.parse()).map_err(|e| format!("ROUNDS: {e}"))?;
    let probes: Vec<PathBuf> = args.iter().skip(7).map(PathBuf::from).collect();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let context = 512;
    let imported = import_language_model(&export, train, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let site = sites(model).into_iter().find(|s| &s.name == name).ok_or_else(|| format!("no site {name}"))?;
    let started = std::time::Instant::now();
    let sequence = |s: usize| family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>());
    let sample = samples(model, std::slice::from_ref(&site), (0..train).map(sequence), 4, 0x517E)?.remove(0);
    eprintln!("{name}: samples on {train} sequences, {:.0}s", started.elapsed().as_secs_f64());
    let w = matrix(model, &site)?;
    let (d_out, d_in) = w.dim();
    // The site-fit driver's code: each subcomponent priced exactly in its interfaces' declared
    // charts where that could change a set, by Generic's closed form elsewhere.
    let statistics = vec![gam_mpd::pieces::Site { w: w.clone(), second_moment: sample.second_moment.clone(), mean: Array1::zeros(d_in), fisher: sample.fisher.clone() }];
    let (writers, readers) = gam_mpd::describe::declared_charts(model, &site)?;
    let description = gam_mpd::describe::Tiered {
        cheap: gam_mpd::blocks::Generic::new(&statistics, observations),
        exact: gam_mpd::describe::Structured::new(vec![gam_mpd::describe::Geometry::new(
            gam_mpd::describe::Metric::of(&statistics[0], observations),
            writers,
            readers,
        )?]),
    };
    let (v_path, u_path) = (dir.join(format!("{name}.v.f64")), dir.join(format!("{name}.u.f64")));
    let library = if v_path.exists() && u_path.exists() {
        Library { v: read_f64(&v_path, d_in)?, u: read_f64(&u_path, d_out)?, mean: Array1::zeros(d_in) }
    } else {
        // The driver's start: the units a site reads, else its Fisher-whitened singular pieces.
        let reads_units = site.reads.len() == 1 && matches!(model.nodes[site.reads[0]], Node::Pointwise { .. });
        let exact = if reads_units { gam_mpd::pieces::unit_pieces(&w, gam_mpd::pieces::Units::Read) } else { gam_mpd::pieces::fisher_svd(&statistics[0])? };
        let start = Library { v: exact.v.t().to_owned(), u: exact.u, mean: Array1::zeros(d_in) };
        let settings = Settings { observations, pieces: d_in + d_out, rounds, seed: 0xF17 };
        let mut log = Vec::new();
        let library = fit(0, &w, &sample, &description, settings, Some(&start), |round, _| {
            eprintln!(
                "{name} round {}: code {:.1} bits per input (description {:.1}, error {:.1}), L0 {:.2}, reseeded {}, {:.0}s",
                round.round,
                round.code,
                round.description,
                round.error,
                round.l0,
                round.reseeded,
                started.elapsed().as_secs_f64()
            );
            log.push(json!({"round": round.round, "code": round.code, "description": round.description, "error": round.error, "l0": round.l0,
                "reseeded": round.reseeded, "seconds": started.elapsed().as_secs_f64()}));
        })?;
        write_f64(&v_path, &library.v)?;
        write_f64(&u_path, &library.u)?;
        let record = json!({"site": name, "export": export, "train": train, "observations": observations, "rounds": log});
        std::fs::write(dir.join(format!("{name}.rounds.json")), record.to_string()).map_err(|e| e.to_string())?;
        library
    };
    // Every input's sets under the one code; a probe's inputs carry the training statistics.
    let select = |samples: &Samples, out: &Path| -> Result<(), String> {
        let (round, sets) = measure(0, &w, samples, &description, observations, &library)?;
        eprintln!("{}: {} inputs, code {:.1} bits per input (description {:.1}, error {:.1}), L0 {:.2}", out.display(), sets.len(), round.code, round.description, round.error, round.l0);
        let record = json!({"site": name, "pieces": library.v.nrows(), "observations": observations, "code": round.code,
            "description": round.description, "error": round.error, "l0": round.l0, "sets": sets});
        std::fs::write(out, record.to_string()).map_err(|e| format!("{}: {e}", out.display()))
    };
    select(&sample, &dir.join(format!("{name}.train.sets.json")))?;
    for probe in &probes {
        let reads = read_f64(probe, d_in)?;
        let rows = reads.nrows();
        let probed = Samples { reads: reads.mapv(|v| v as f32), sensitivity: Array1::ones(rows), fisher: sample.fisher.clone(), second_moment: sample.second_moment.clone() };
        let stem = probe.file_stem().and_then(|s| s.to_str()).ok_or("probe name")?;
        select(&probed, &dir.join(format!("{stem}.sets.json")))?;
    }
    Ok(())
}

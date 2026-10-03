//! Every subcomponent's description bits under the team's pricing (#2951): the exact lattice
//! description in the program's declared charts (`gam_mpd::describe::Structured`, `Describe::bits_at`),
//! in the model's statistics on the export's first `TRAIN` sequences.
//!
//! `mpd_program_bits_2951 EXPORT_DIR LIBRARY_DIR OBSERVATIONS TRAIN OUT_DIR [CONTEXT]`
//!
//! `LIBRARY_DIR` holds a library per site (`{site}.v.f64`, `{site}.u.f64`, rows = subcomponents; a
//! site without files is skipped). Writes `OUT_DIR/{site}.bits.f64` (one f64 per subcomponent, in
//! the library's order) and `OUT_DIR/bits.json` (per site its count, mean and total bits). The
//! natural-language autoencoder charges each word these bits for every subcomponent its text
//! decodes to.

use gam_mpd::blocks::Describe;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{matrix, sites};
use gam_mpd::site_fit::samples;
use ndarray::{Array1, Array2, s};
use rayon::prelude::*;
use serde_json::json;
use std::path::{Path, PathBuf};

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_program_bits_2951 EXPORT_DIR LIBRARY_DIR OBSERVATIONS TRAIN OUT_DIR [CONTEXT]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let given = PathBuf::from(args.get(2).ok_or(usage)?);
    let observations: f64 = args.get(3).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let train: usize = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let out = PathBuf::from(args.get(5).ok_or(usage)?);
    let context: usize = args.get(6).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
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
        let (d_out, d_in) = matrix(model, &site)?.dim();
        libraries.push((read_f64(&v_path, d_in)?, read_f64(&given.join(format!("{}.u.f64", site.name)), d_out)?));
        chosen.push(site);
    }
    if chosen.is_empty() {
        return Err(format!("{}: no library", given.display()));
    }
    let started = std::time::Instant::now();
    let measured = samples(model, &chosen, (0..train).map(sequence), 2, 0xDE5C)?;
    let description = gam_mpd::describe::Structured::new(
        chosen
            .iter()
            .zip(&measured)
            .map(|(site, m)| {
                let w = matrix(model, site)?;
                let statistics = gam_mpd::pieces::Site { mean: Array1::zeros(w.ncols()), w, second_moment: m.second_moment.clone(), fisher: m.fisher.clone() };
                let (writers, readers) = gam_mpd::describe::declared_charts(model, site)?;
                gam_mpd::describe::Geometry::new(gam_mpd::describe::Metric::of(&statistics, observations), writers, readers)
            })
            .collect::<Result<_, String>>()?,
    );
    drop(measured);
    eprintln!("statistics of {} sites on {train} sequences, {:.0}s", chosen.len(), started.elapsed().as_secs_f64());
    let mut summary = Vec::new();
    for (k, (site, (v, u))) in chosen.iter().zip(&libraries).enumerate() {
        let clock = std::time::Instant::now();
        let bits: Vec<f64> = (0..v.nrows())
            .into_par_iter()
            .map(|c| description.bits_at(k, c, u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..])))
            .collect::<Result<_, String>>()?;
        std::fs::write(out.join(format!("{}.bits.f64", site.name)), bits.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>())
            .map_err(|e| e.to_string())?;
        let total: f64 = bits.iter().sum();
        eprintln!("{}: {} subcomponents, mean {:.0} bits, {:.0}s", site.name, bits.len(), total / bits.len() as f64, clock.elapsed().as_secs_f64());
        summary.push(json!({"site": site.name, "subcomponents": bits.len(), "mean": total / bits.len() as f64, "total": total}));
    }
    std::fs::write(out.join("bits.json"), json!({"observations": observations, "train": train, "sites": summary}).to_string()).map_err(|e| e.to_string())
}

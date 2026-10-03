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
//! `gam_mpd::site_fit::fit`), each subcomponent described by `gam_mpd::describe::Structured` in
//! those statistics (the charts its site's interfaces declare, `gam_mpd::describe::declared_charts`).
//! Every site starts from a library that is its map, read off the model alone: a site
//! that reads a layer of units (a pointwise map's output, an MLP's hidden layer) from those units
//! (`gam_mpd::pieces::unit_pieces`), any other from its Fisher-whitened singular pieces
//! (`gam_mpd::pieces::fisher_svd`); the rest of its subcomponents start writing nothing and grow by
//! splits. After every round the library goes to `OUT_DIR/{site}.v.f64` (pieces × d_in) and
//! `OUT_DIR/{site}.u.f64` (pieces × d_out), raw float64, the `library:DIR` start of
//! `mpd_pieces_masked_2951`, and its rounds to `OUT_DIR/{site}.rounds.json`.
//!
//! Every library (fitted or given) is then made blocks of any rank by the same code
//! (`gam_mpd::site_fit::blocks`: merges and splits, each kept when the site's code falls), from both
//! the fine start (every subcomponent its own block) and the coarse one (the whole site one block),
//! the lower kept and written as the site's library in `OUT_DIR` (`{site}.{v,u}.f64` in its blocks'
//! column order, `{site}.ranks.json`), both starts' codes in `OUT_DIR/{site}.blocks.json`.
//!
//! With `library:DIR` nothing is fitted: each site's given library (`DIR/{site}.{v,u}.f64`, as
//! above) is measured under the same code (`gam_mpd::site_fit::measure`, its sets selected from all
//! on), the measurement goes to `OUT_DIR/{site}.measure.json`, and every input's selected sets to
//! `OUT_DIR/sets/` as `mpd_pieces_masked_2951` takes its `SETS` (`indptr.i64`, `indices.i64` CSR
//! over the positions, subcomponents numbered site after site, `sites.txt`): a start for its
//! selection from this code's own.

use gam_mpd::import::import_language_model;
use gam_mpd::masked::{matrix, sites};
use gam_mpd::masked::Library;
use gam_mpd::operator_program::Node;
use gam_mpd::site_fit::{Settings, ard, blocks, fit, measure, samples};
use ndarray::{Array1, Array2};
use serde_json::json;
use std::path::PathBuf;

fn write_f64(path: &PathBuf, m: &Array2<f64>) -> Result<(), String> {
    let bytes: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
    let partial = path.with_extension("partial");
    std::fs::write(&partial, bytes).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// How much the subcomponents' own sizes exceed the map they sum to, `Σ_c ‖u_c‖‖v_c‖ / ‖W‖_F`:
/// near one for subcomponents that add up, large for ones that cancel.
fn spread(w: &Array2<f64>, library: &Library) -> f64 {
    let norm = |r: ndarray::ArrayView1<f64>| r.dot(&r).sqrt();
    let total: f64 = library.u.outer_iter().zip(library.v.outer_iter()).map(|(u, v)| norm(u) * norm(v)).sum();
    total / w.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Where the sites' libraries go and the code they are measured in.
#[derive(Clone, Copy)]
struct Coding<'a> {
    out: &'a std::path::Path,
    description: &'a dyn gam_mpd::blocks::Describe,
    observations: f64,
}

/// The site's library as blocks of any rank, chosen by its code (`gam_mpd::site_fit::blocks`) from
/// both starts, every column its own block (merging up) and the whole site one block (splitting
/// down), the lower total kept; written as the site's library, `{site}.{v,u}.f64` in its blocks'
/// column order and `{site}.ranks.json`, with both starts' codes in `{site}.blocks.json`.
fn write_blocks(coding: &Coding<'_>, name: &str, k: usize, w: &Array2<f64>, sample: &gam_mpd::site_fit::Samples, library: &Library) -> Result<(), String> {
    let Coding { out, description, observations } = *coding;
    let columns = library.v.nrows();
    let (fine, fine_round) = blocks(k, w, sample, description, observations, library, &vec![1; columns])?;
    let (coarse, coarse_round) = blocks(k, w, sample, description, observations, library, &[columns])?;
    let capacity = columns.max(w.nrows() + w.ncols());
    let (evidence, evidence_round) = ard(k, w, sample, description, observations, (library, &vec![1; columns], capacity), 50)?;
    let (chosen, round) = [(&fine, &fine_round), (&coarse, &coarse_round), (&evidence, &evidence_round)]
        .into_iter()
        .fold(None, |best: Option<(&gam_mpd::site_fit::Blocked, &gam_mpd::site_fit::Round)>, c| match best {
            Some(b) if b.1.code <= c.1.code => Some(b),
            _ => Some(c),
        })
        .ok_or("no partition")?;
    write_f64(&out.join(format!("{name}.v.f64")), &chosen.library.v)?;
    write_f64(&out.join(format!("{name}.u.f64")), &chosen.library.u)?;
    std::fs::write(out.join(format!("{name}.ranks.json")), json!(chosen.ranks).to_string()).map_err(|e| e.to_string())?;
    let histogram = |ranks: &[usize]| {
        let mut h = std::collections::BTreeMap::<usize, usize>::new();
        for r in ranks {
            *h.entry(*r).or_default() += 1;
        }
        h
    };
    eprintln!(
        "{name} blocks: code {:.1} bits per input (description {:.1}, error {:.1}), ranks {:?}, {:.2} on per input; merges/splits from the fine start {:.1}, the coarse {:.1}; by evidence {:.1}",
        round.code,
        round.description,
        round.error,
        histogram(&chosen.ranks),
        round.l0,
        fine_round.code,
        coarse_round.code,
        evidence_round.code
    );
    let record = |b: &gam_mpd::site_fit::Blocked, r: &gam_mpd::site_fit::Round| {
        json!({"ranks": histogram(&b.ranks), "code": r.code, "description": r.description, "error": r.error, "l0": r.l0})
    };
    let report = json!({"site": name, "observations": observations, "fine": record(&fine, &fine_round), "coarse": record(&coarse, &coarse_round),
        "evidence": record(&evidence, &evidence_round)});
    std::fs::write(out.join(format!("{name}.blocks.json")), report.to_string()).map_err(|e| e.to_string())
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
    // Each subcomponent priced by its exact lattice description in the charts its site's interfaces
    // declare (heads, rotary planes) wherever that price could change a set, by Generic's closed
    // form elsewhere (gam_mpd::describe::Tiered, gam_mpd::site_fit's prices).
    let description = gam_mpd::describe::Tiered {
        cheap: gam_mpd::blocks::Generic::new(&statistics, observations),
        exact: gam_mpd::describe::Structured::new(
            chosen
                .iter()
                .zip(&statistics)
                .map(|(site, measured)| {
                    let (writers, readers) = gam_mpd::describe::declared_charts(model, site)?;
                    gam_mpd::describe::Geometry::new(gam_mpd::describe::Metric::of(measured, observations), writers, readers)
                })
                .collect::<Result<_, String>>()?,
        ),
    };
    drop(statistics);
    // `library:DIR`: every site's selected sets, written as one CSR at the end.
    let mut selected: Vec<(String, usize, Vec<Vec<u32>>)> = Vec::new();
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
            let (round, chosen_sets) = measure(k, w, sample, &description, observations, &library)?;
            write_blocks(&Coding { out: &out, description: &description, observations }, &site.name, k, w, sample, &library)?;
            selected.push((site.name.clone(), library.v.nrows(), chosen_sets));
            eprintln!("{} given library of {}: code {:.1} bits per input (description {:.1}, error {:.1}), L0 {:.2}, Σ|u||v|/|W| {:.1}",
                site.name, library.v.nrows(), round.code, round.description, round.error, round.l0, spread(w, &library));
            let record = json!({"site": site.name, "observations": observations, "pieces": library.v.nrows(), "code": round.code,
                "description": round.description, "error": round.error, "l0": round.l0, "spread": spread(w, &library)});
            std::fs::write(out.join(format!("{}.measure.json", site.name)), record.to_string()).map_err(|e| e.to_string())?;
            continue;
        }
        let settings = Settings { observations, pieces: d_in + d_out, rounds, seed: 0xF17 + k as u64 };
        let started = std::time::Instant::now();
        let mut log = Vec::new();
        let (v_path, u_path) = (out.join(format!("{}.v.f64", site.name)), out.join(format!("{}.u.f64", site.name)));
        let rounds_path = out.join(format!("{}.rounds.json", site.name));
        // The start: the units a site reads, else its Fisher-whitened singular pieces.
        let reads_units = site.reads.len() == 1 && matches!(model.nodes[site.reads[0]], Node::Pointwise { .. });
        let exact = if reads_units {
            gam_mpd::pieces::unit_pieces(w, gam_mpd::pieces::Units::Read)
        } else {
            gam_mpd::pieces::fisher_svd(&gam_mpd::pieces::Site { w: w.clone(), second_moment: sample.second_moment.clone(), mean: Array1::zeros(d_in), fisher: sample.fisher.clone() })?
        };
        let start = Library { v: exact.v.t().to_owned(), u: exact.u, mean: Array1::zeros(d_in) };
        let library = fit(k, w, sample, &description, settings, Some(&start), |round, library| {
            eprintln!(
                "{} round {}: code {:.1} bits per input (description {:.1}, error {:.1}), L0 {:.2}, reseeded {}, read rung {}, {:.0}s",
                site.name,
                round.round,
                round.code,
                round.description,
                round.error,
                round.l0,
                round.reseeded,
                round.read_steps,
                started.elapsed().as_secs_f64()
            );
            log.push(json!({"round": round.round, "code": round.code, "description": round.description, "error": round.error, "l0": round.l0,
                "reseeded": round.reseeded, "read_steps": round.read_steps, "seconds": started.elapsed().as_secs_f64()}));
            let written = write_f64(&v_path, &library.v)
                .and_then(|_| write_f64(&u_path, &library.u))
                .and_then(|_| std::fs::write(&rounds_path, json!({"site": site.name, "observations": observations, "rounds": log}).to_string()).map_err(|e| e.to_string()));
            if let Err(e) = written {
                eprintln!("{}: {e}", site.name);
            }
        })?;
        write_f64(&v_path, &library.v)?;
        write_f64(&u_path, &library.u)?;
        write_blocks(&Coding { out: &out, description: &description, observations }, &site.name, k, w, sample, &library)?;
        // What all on leaves of the map, in the reads' second moment M: ‖(W − Σ u vᵀ) M^½‖ / ‖W M^½‖.
        let weighted = |e: &Array2<f64>| (&e.dot(&sample.second_moment) * e).sum().sqrt();
        let left = weighted(&(w - &library.u.t().dot(&library.v))) / weighted(w);
        eprintln!(
            "{}: {} subcomponents written, all on leaves {left:.2e} of the map, Σ|u||v|/|W| {:.1}, {:.0}s",
            site.name,
            library.v.nrows(),
            spread(w, &library),
            started.elapsed().as_secs_f64()
        );
    }
    if !selected.is_empty() {
        let dir = out.join("sets");
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let rows = selected[0].2.len();
        let (mut indptr, mut indices) = (vec![0i64], Vec::new());
        for r in 0..rows {
            let mut offset = 0i64;
            for (_, pieces, sets) in &selected {
                indices.extend(sets[r].iter().map(|c| offset + i64::from(*c)));
                offset += *pieces as i64;
            }
            indptr.push(indices.len() as i64);
        }
        let bytes = |values: &[i64]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(dir.join("indptr.i64"), bytes(&indptr)).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("indices.i64"), bytes(&indices)).map_err(|e| e.to_string())?;
        let listed: String = selected.iter().map(|(name, pieces, _)| format!("{name} {pieces}\n")).collect();
        std::fs::write(dir.join("sites.txt"), listed).map_err(|e| e.to_string())?;
    }
    Ok(())
}

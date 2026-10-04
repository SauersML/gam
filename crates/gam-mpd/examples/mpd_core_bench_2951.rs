//! Wall time of each stage of the explanation's core path at one site of a language model, on the
//! CPU (#2951, `gam_mpd::explanation`).
//!
//! `mpd_core_bench_2951 TRAIN LIBRARY SITE [KEY=VALUE ...]`
//!
//! `TRAIN` is an engine export of the model, `LIBRARY` a directory of subcomponents
//! (`{SITE}.{v,u}.f64`, VPD's `~/mpd-data/pieces/vpd4l_library`) or `units` (the site's read
//! units), `SITE` one of the model's sites.
//! On the export's first `sequences` sequences (1) of `context` positions (512) it times:
//!
//! * `samples`: the site's reads, mean written Fisher and second moment (`site_fit::samples`,
//!   `draws` sampled-label passes per sequence, 1);
//! * `cheap` and `exact`: every block's cheap price, and `priced` blocks' structured price
//!   (`describe::Tiered`), per block;
//! * `selector` and `code_site`: every input's sets by the run-time selection
//!   (`site_fit::Selector`) and by the certified sparse code (`sparse_code::code_site`, `nodes`
//!   branch-and-bound nodes, 64), at the cheap prices and `n` (1e5); with the codes per input
//!   (description plus error bits) both choose and how many inputs each codes in fewer bits.
//!
//! One JSON line of every stage's seconds and the comparison goes to stdout.

use gam_mpd::blocks::{Describe, Generic};
use gam_mpd::counterfactual::read_f64_matrix;
use gam_mpd::describe::{Geometry, Metric, Structured, Tiered, declared_charts};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, matrix, sites};
use gam_mpd::site_fit::{Selector, samples};
use gam_mpd::sparse_code::{Metric as Coded, Problem, code_site};
use ndarray::{Array1, Array2, s};
use rayon::prelude::*;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

/// Per input, `Σ_on bits + κ ‖y − Σ_on z_c u_c‖²_F` of the sets `on` (inputs × blocks, rank-one).
fn codes(reads: &Array2<f64>, w: &Array2<f64>, library: &Library, fisher: &Array2<f64>, bits: &[f64], on: &Array2<f64>, observations: f64) -> Array1<f64> {
    let kappa = observations / (2.0 * std::f64::consts::LN_2);
    let z = reads.dot(&library.v.t()) * on;
    let residual = reads.dot(&w.t()) - z.dot(&library.u);
    let weighted = residual.dot(fisher);
    let bits = Array1::from(bits.to_vec());
    on.dot(&bits) + (&residual * &weighted).sum_axis(ndarray::Axis(1)) * kappa
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_core_bench_2951 TRAIN LIBRARY SITE [KEY=VALUE ...]";
    if args.len() < 4 {
        return Err(usage.to_string());
    }
    let (train, dir, name) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]), args[3].clone());
    let (mut sequences, mut context, mut draws, mut observations, mut nodes, mut priced) = (1usize, 512usize, 1usize, 1e5f64, 64usize, 64usize);
    for pair in &args[4..] {
        let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair}: not KEY=VALUE"))?;
        let count = || value.parse::<usize>().map_err(|e| format!("{key}: {e}"));
        match key {
            "sequences" => sequences = count()?,
            "context" => context = count()?,
            "draws" => draws = count()?,
            "n" => observations = value.parse().map_err(|e| format!("n: {e}"))?,
            "nodes" => nodes = count()?,
            "priced" => priced = count()?,
            other => return Err(format!("unknown key {other}")),
        }
    }
    let imported = import_language_model(&train, sequences, context)?;
    let model = &imported.program;
    let site = sites(model).into_iter().find(|s| s.name == name).ok_or_else(|| format!("{name}: not a site"))?;
    let w = matrix(model, &site)?;
    let (d_out, d_in) = w.dim();
    // `LIBRARY` a directory of subcomponents, or `units`: the site's read units (an MLP output's
    // starting library).
    let library = if dir.as_os_str() == "units" {
        let exact = gam_mpd::pieces::unit_pieces(&w, gam_mpd::pieces::Units::Read);
        Library { v: exact.v.t().to_owned(), u: exact.u, mean: Array1::zeros(d_in) }
    } else {
        Library { v: read_f64_matrix(&dir.join(format!("{name}.v.f64")), d_in)?, u: read_f64_matrix(&dir.join(format!("{name}.u.f64")), d_out)?, mean: Array1::zeros(d_in) }
    };
    let columns = library.v.nrows();
    let mut seconds = serde_json::Map::new();

    let clock = Instant::now();
    let batches = (0..sequences).map(|s| imported.contract.family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>()));
    let sample = samples(model, std::slice::from_ref(&site), batches, draws, 0x5EED)?.swap_remove(0);
    seconds.insert("samples".into(), json!(clock.elapsed().as_secs_f64()));
    let reads = sample.reads.mapv(f64::from);
    let rows = reads.nrows();

    let statistics = gam_mpd::pieces::Site { w: w.clone(), second_moment: sample.second_moment.clone(), mean: Array1::zeros(d_in), fisher: sample.fisher.clone() };
    let (writers, readers) = declared_charts(model, &site)?;
    let describe = Tiered {
        cheap: Generic::new(std::slice::from_ref(&statistics), observations),
        exact: Structured::new(vec![Geometry::new(Metric::of(&statistics, observations), writers, readers)?]),
    };
    let block = |c: usize| (library.u.slice(s![c..c + 1, ..]), library.v.slice(s![c..c + 1, ..]));
    let clock = Instant::now();
    let bits: Vec<f64> = (0..columns)
        .into_par_iter()
        .map(|c| {
            let (u, v) = block(c);
            describe.cheap(0, u, v).map(|b| b.unwrap_or(0.0))
        })
        .collect::<Result<_, String>>()?;
    seconds.insert("cheap_per_block".into(), json!(clock.elapsed().as_secs_f64() / columns as f64));
    let sampled: Vec<usize> = (0..columns).step_by((columns / priced.max(1)).max(1)).take(priced).collect();
    let clock = Instant::now();
    let exact: Vec<f64> = sampled
        .par_iter()
        .map(|&c| {
            let (u, v) = block(c);
            gam_linalg::faer_ndarray::with_nested_parallel(|| describe.bits(0, u, v))
        })
        .collect::<Result<_, String>>()?;
    let exact_seconds = clock.elapsed().as_secs_f64();
    seconds.insert("exact_per_block".into(), json!(exact_seconds / sampled.len().max(1) as f64));
    seconds.insert("exact_all_blocks_estimate".into(), json!(exact_seconds * columns as f64 / sampled.len().max(1) as f64));
    let ratio: f64 = sampled.iter().zip(&exact).map(|(c, e)| e / bits[*c].max(f64::MIN_POSITIVE)).sum::<f64>() / sampled.len().max(1) as f64;

    let ranks = vec![1usize; columns];
    let clock = Instant::now();
    let selector = Selector::new(&w, &sample.fisher, &library, &ranks, &bits, observations)?;
    seconds.insert("selector_setup".into(), json!(clock.elapsed().as_secs_f64()));
    let clock = Instant::now();
    let held = selector.select(&reads);
    seconds.insert("selector".into(), json!(clock.elapsed().as_secs_f64()));
    let selector_codes = codes(&reads, &w, &library, &sample.fisher, &bits, &held, observations);

    let targets = reads.dot(&w.t());
    let problem = |nodes: usize| Problem {
        reads: reads.view(),
        targets: targets.view(),
        v: library.v.view(),
        u: library.u.view(),
        ranks: &ranks,
        bits: &bits,
        metric: Coded::Mean(sample.fisher.view()),
        observations,
        nodes,
    };
    let mut compared = serde_json::Map::new();
    for (label, branch) in [("code_site_root", 0usize), ("code_site", nodes)] {
        let clock = Instant::now();
        let coding = code_site(&problem(branch), None)?;
        seconds.insert(label.into(), json!(clock.elapsed().as_secs_f64()));
        let on = Array2::from_shape_fn((rows, columns), |(t, c)| f64::from(u8::from(coding.sets[t].binary_search(&(c as u32)).is_ok())));
        let own = codes(&reads, &w, &library, &sample.fisher, &bits, &on, observations);
        let better = own.iter().zip(&selector_codes).filter(|(a, b)| *a < *b).count();
        let worse = own.iter().zip(&selector_codes).filter(|(a, b)| *a > *b).count();
        let certified = coding.upper.iter().zip(&coding.lower).filter(|(u, l)| *u - *l <= 1.0).count();
        compared.insert(
            label.into(),
            json!({"code_per_input": own.mean(), "l0": on.sum() / rows as f64, "better_than_selector": better, "worse_than_selector": worse,
                   "certified_within_a_bit": certified, "largest_gap": (&coding.upper - &coding.lower).fold(0.0_f64, |m, g| m.max(*g))}),
        );
    }
    println!(
        "{}",
        json!({"site": name, "rows": rows, "columns": columns, "d_in": d_in, "d_out": d_out, "n": observations, "seconds": seconds,
               "exact_over_cheap": ratio, "selector": {"code_per_input": selector_codes.mean(), "l0": held.sum() / rows as f64}, "code_site": compared})
    );
    Ok(())
}

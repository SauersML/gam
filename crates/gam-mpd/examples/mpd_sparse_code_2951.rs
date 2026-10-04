//! Time and certificate of every input's sparse code at one site of a language model (#2951,
//! `gam_mpd::sparse_code::code_site`).
//!
//! `mpd_sparse_code_2951 EXPORT_DIR LIBRARY_DIR SITE OBSERVATIONS SEQUENCES [CONTEXT] [BITS] [NODES]`
//!
//! The site's reads on the export's first `SEQUENCES` sequences (`CONTEXT` positions, default 512)
//! and its mean output Fisher come from `gam_mpd::site_fit::samples` (one draw); its library is
//! `LIBRARY_DIR/{SITE}.{v,u}.f64`, rank one, every piece `BITS` bits (default 32 times its reals);
//! every input is coded in the mean metric with `NODES` branch-and-bound nodes (default 64). It
//! reports the seconds, the inputs' mean code and active pieces, and how many inputs are certified
//! within a bit of their optimum.

use gam_mpd::import::import_language_model;
use gam_mpd::masked::{matrix, sites};
use gam_mpd::site_fit::samples;
use gam_mpd::sparse_code::{Metric, Problem, code_site};
use ndarray::Array2;
use std::path::PathBuf;

fn read(path: &std::path::Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_sparse_code_2951 EXPORT_DIR LIBRARY_DIR SITE OBSERVATIONS SEQUENCES [CONTEXT] [BITS] [NODES]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let library = PathBuf::from(args.get(2).ok_or(usage)?);
    let name = args.get(3).ok_or(usage)?;
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let sequences: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("SEQUENCES: {e}"))?;
    let context: usize = args.get(6).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let bits_given: Option<f64> = args.get(7).map(|v| v.parse()).transpose().map_err(|e| format!("BITS: {e}"))?;
    let nodes: usize = args.get(8).map_or(Ok(64), |v| v.parse()).map_err(|e| format!("NODES: {e}"))?;
    let imported = import_language_model(&export, sequences, context)?;
    let program = &imported.program;
    let family = &imported.contract.family;
    let site = sites(program).into_iter().find(|s| &s.name == name).ok_or_else(|| format!("{name}: not a site"))?;
    let w = matrix(program, &site)?;
    let (d_out, d_in) = w.dim();
    let measured = samples(program, std::slice::from_ref(&site), (0..sequences).map(|s| family.select(&(s * context..(s + 1) * context).collect::<Vec<_>>())), 1, 0x5EED)?;
    let measured = &measured[0];
    let reads = measured.reads.mapv(f64::from);
    let targets = reads.dot(&w.t());
    let v = read(&library.join(format!("{name}.v.f64")), d_in)?;
    let u = read(&library.join(format!("{name}.u.f64")), d_out)?;
    let pieces = v.nrows();
    let ranks = vec![1; pieces];
    let bits = vec![bits_given.unwrap_or(32.0 * (d_in + d_out) as f64); pieces];
    let problem = Problem {
        reads: reads.view(),
        targets: targets.view(),
        v: v.view(),
        u: u.view(),
        ranks: &ranks,
        bits: &bits,
        metric: Metric::Mean(measured.fisher.view()),
        observations,
        nodes,
    };
    let started = std::time::Instant::now();
    let coding = code_site(&problem, None)?;
    let seconds = started.elapsed().as_secs_f64();
    let rows = reads.nrows() as f64;
    let certified = coding.upper.iter().zip(coding.lower.iter()).filter(|(u, l)| *u - *l <= 1.0).count();
    let gap = (&coding.upper - &coding.lower).iter().fold(0.0_f64, |m, g| m.max(*g));
    eprintln!(
        "{name}: {} inputs, {pieces} pieces, {seconds:.1}s ({:.2} ms per input); code {:.1} bits per input, L0 {:.2}; {certified} certified within a bit, largest gap {gap:.2}",
        reads.nrows(),
        1000.0 * seconds / rows,
        coding.upper.sum() / rows,
        coding.sets.iter().map(Vec::len).sum::<usize>() as f64 / rows
    );
    Ok(())
}

//! The box claim certified (#2951): per word, a sound upper bound on `KL(model ‖ masked)` over every
//! off gate anywhere in `[0, 1]` (`gam_mpd::certify`), against the worst point an adversary finds.
//!
//! `mpd_certify_2951 toy EXPORT_DIR OUT.json [ROWS] [BUDGET] [LEAVES] [OBSERVATIONS]`
//!
//! A small export (`gam_mpd::import::import`, its first `ROWS` samples): every site starts from its
//! exact Fisher-whitened singular subcomponents (`gam_mpd::pieces::fisher_svd`), and two sets per
//! word are chosen by the masked selection, under the corner claim (`select`) and under the box claim
//! (`select_boxed`), each coded in `OBSERVATIONS` (default 1000). For each set it reports per word:
//! * the masks' own KL;
//! * the engine's box estimate (the masks plus `box_excess_at`, its three-step adversary);
//! * the worst KL of a 40-step, 8-start adversary;
//! * the certificate with at most `BUDGET` symbols per word (default 4096);
//! * the certificate after `LEAVES` leaves of branching on that word alone (default 33).
//!
//! `mpd_certify_2951 lm EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json SEQUENCES CONTEXT [BUDGET] [LEAVES] [FREE] [STEPS]`
//!
//! A language-model export (`import_language_model`), the first `CONTEXT` positions of its first
//! `SEQUENCES` sequences. Positions only read earlier ones, so a prefix is exact. The sites with a
//! library in `LIBRARY_DIR` (`{site}.v.f64`, `{site}.u.f64`) run on its subcomponents, with nothing
//! beyond them. `SETS_DIR` gives each position's set (`bench/vpd_2951/vpd_sets_export.py`'s CSR over
//! 512-position sequences). `FREE` (default `all`) is a comma-separated list of site-name prefixes
//! whose off gates are free; every other off gate stays at 0. The adversary takes `STEPS` steps
//! (default 40) from 6 starts on the sequence's total KL. Each word's certificate uses `BUDGET`
//! symbols (default 512), and the sequence's box is split into `LEAVES` leaves (default 1: no
//! branching).

use gam_mpd::blocks::Describe;
use gam_mpd::certify::{Gates, adversary, certify, certify_branching};
use gam_mpd::import::{import, import_language_model};
use gam_mpd::masked::{Coder, Library, Masked, Target, box_excess_at, forward, select, select_boxed, site_statistics, sites};
use gam_mpd::operator_program::{FamilyInputs, OperatorProgram};
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % (cols * 8) != 0 {
        return Err(format!("{}: {} bytes are not rows of {cols} float64", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((bytes.len() / (cols * 8), cols), values).map_err(|e| e.to_string())
}

fn read_i64(path: &Path) -> Result<Vec<i64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> Result<T, String> {
    args.get(i).map_or(Ok(default), |v| v.parse().map_err(|_| format!("argument {i}: cannot read {v}")))
}

/// Mean, median and largest of finite values, and how many are infinite.
fn summary(values: &Array1<f64>) -> Value {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    finite.sort_by(f64::total_cmp);
    let n = finite.len();
    json!({
        "mean": if n > 0 { finite.iter().sum::<f64>() / n as f64 } else { f64::NAN },
        "median": if n > 0 { finite[n / 2] } else { f64::NAN },
        "max": finite.last().copied().unwrap_or(f64::NAN),
        "infinite": values.len() - n,
    })
}

/// The native logits on `family` and their radius.
fn reference(program: &OperatorProgram, family: &FamilyInputs) -> Result<(Target, Array2<f64>), String> {
    let trace = program.execute(family, true).map_err(|e| e.to_string())?;
    let radius = trace.band(program.output).ok_or("no bands")?;
    Ok((Target::every_row(trace.values[program.output].clone()), radius))
}

fn l0(masks: &[Array2<f64>]) -> f64 {
    let rows = masks.first().map_or(1, |m| m.nrows()).max(1);
    masks.iter().map(|m| m.iter().filter(|x| **x > 0.0).count()).sum::<usize>() as f64 / rows as f64
}

fn toy(args: &[String]) -> Result<(), String> {
    let usage = "mpd_certify_2951 toy EXPORT_DIR OUT.json [ROWS] [BUDGET] [LEAVES] [OBSERVATIONS]";
    let dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let imported = import(&dir)?;
    let rows: usize = arg(args, 4, imported.contract.family.rows)?;
    let budget: usize = arg(args, 5, 4096)?;
    let leaves: usize = arg(args, 6, 33)?;
    let observations: f64 = arg(args, 7, 1000.0)?;
    let program = &imported.program;
    let family = imported.contract.family.select(&(0..rows.min(imported.contract.family.rows)).collect::<Vec<_>>());
    let chosen = sites(program);
    let statistics = site_statistics(program, &chosen, [family.clone()], 2, 0x5EED)?;
    let mut libraries = Vec::new();
    for site in &statistics {
        let library = fisher_svd(site)?;
        libraries.push(Library { v: library.v.t().to_owned(), u: library.u, mean: Array1::zeros(site.w.ncols()) });
    }
    let fishers: Vec<Array2<f64>> = statistics.iter().map(|s| s.fisher.clone()).collect();
    let description = gam_mpd::blocks::Generic::new(
        &statistics
            .iter()
            .map(|m| gam_mpd::pieces::Site { w: m.w.clone(), second_moment: m.second_moment.clone(), mean: Array1::zeros(m.mean.len()), fisher: m.fisher.clone() })
            .collect::<Vec<_>>(),
        observations,
    );
    let masked = Masked::build(program, chosen, libraries)?;
    let costs: Vec<Array1<f64>> = (0..masked.sites.len())
        .map(|k| {
            let library = masked.library(k)?;
            (0..library.v.nrows())
                .map(|c| description.bits(k, library.u.slice(ndarray::s![c..c + 1, ..]), library.v.slice(ndarray::s![c..c + 1, ..])))
                .collect::<Result<Array1<f64>, String>>()
        })
        .collect::<Result<_, _>>()?;
    let (target, radius) = reference(program, &family)?;
    let all_on: Vec<Array2<f64>> = (0..masked.sites.len()).map(|k| Array2::ones((family.rows, masked.blocks(k)))).collect();
    let coder = Coder::ran(costs.clone(), family.rows);
    let started = Instant::now();
    let corner = select(&masked, &family, &target, all_on.clone(), &coder, observations, 2)?.0;
    let boxed = select_boxed(&masked, &family, &target, all_on, &coder, observations, 2, &fishers)?.0;
    eprintln!("selected both sets in {:.1}s", started.elapsed().as_secs_f64());
    let mut report = Vec::new();
    for (name, masks) in [("corner", corner), ("box", boxed)] {
        let started = Instant::now();
        let (kl, _, _) = forward(&masked, &masked.family(&family, &masks), &target)?;
        let estimate = &kl + &box_excess_at(&masked, &family, &target, &masks, &fishers)?;
        let gates = Gates::claim(&masks);
        let found = adversary(&masked, &family, &target, &gates, None, 40, 8, 0xAD5)?;
        let root = certify(&masked, &family, &target, Some(&radius), &gates, budget)?;
        let mut branched = Array1::<f64>::zeros(family.rows);
        for r in 0..family.rows {
            let one = family.select(&[r]);
            let row_target = Target::every_row(target.logits.select(ndarray::Axis(0), &[r]));
            let row_radius = radius.select(ndarray::Axis(0), &[r]);
            let row_gates = Gates {
                lower: gates.lower.iter().map(|m| m.select(ndarray::Axis(0), &[r])).collect(),
                upper: gates.upper.iter().map(|m| m.select(ndarray::Axis(0), &[r])).collect(),
            };
            branched[r] = certify_branching(&masked, &one, &row_target, Some(&row_radius), row_gates, budget, leaves)?.kl[0];
        }
        for r in 0..family.rows {
            if root[r] < found[r] || branched[r] < found[r] {
                return Err(format!("{name}: row {r} certified {} / {} below the adversary's {}", root[r], branched[r], found[r]));
            }
        }
        let ratio = |a: &Array1<f64>| Array1::from_iter(a.iter().zip(found.iter()).map(|(c, f)| c / f.max(1e-12)));
        eprintln!(
            "{name}: L0 {:.2}, KL {:.4}, engine estimate {:.4}, adversary {:.4}, certified {:.4}, branched {:.4} ({:.1}s)",
            l0(&masks),
            kl.mean().unwrap_or(0.0),
            estimate.mean().unwrap_or(0.0),
            found.mean().unwrap_or(0.0),
            root.mean().unwrap_or(0.0),
            branched.mean().unwrap_or(0.0),
            started.elapsed().as_secs_f64()
        );
        report.push(json!({
            "sets": name,
            "l0": l0(&masks),
            "kl": summary(&kl),
            "engine_estimate": summary(&estimate),
            "adversary": summary(&found),
            "certified": summary(&root),
            "branched": summary(&branched),
            "certified_over_adversary": summary(&ratio(&root)),
            "branched_over_adversary": summary(&ratio(&branched)),
            "rows": {
                "kl": kl.to_vec(), "engine_estimate": estimate.to_vec(), "adversary": found.to_vec(),
                "certified": root.to_vec(), "branched": branched.to_vec(),
            },
        }));
    }
    let record = json!({
        "mode": "toy", "export": dir, "rows": family.rows, "budget": budget, "leaves": leaves, "observations": observations,
        "sites": masked.sites.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
        "sets": report,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn lm(args: &[String]) -> Result<(), String> {
    let usage = "mpd_certify_2951 lm EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json SEQUENCES CONTEXT [BUDGET] [LEAVES] [FREE] [STEPS]";
    let export = PathBuf::from(args.get(2).ok_or(usage)?);
    let library_dir = PathBuf::from(args.get(3).ok_or(usage)?);
    let sets_dir = PathBuf::from(args.get(4).ok_or(usage)?);
    let out = PathBuf::from(args.get(5).ok_or(usage)?);
    let sequences: usize = arg(args, 6, 1)?;
    let context: usize = arg(args, 7, 16)?;
    let budget: usize = arg(args, 8, 512)?;
    let leaves: usize = arg(args, 9, 1)?;
    let free: String = arg(args, 10, "all".to_string())?;
    let steps: usize = arg(args, 11, 40)?;
    let started = Instant::now();
    let imported = import_language_model(&export, sequences, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    for site in sites(model) {
        let v_path = library_dir.join(format!("{}.v.f64", site.name));
        if !v_path.exists() {
            continue;
        }
        let w = gam_mpd::masked::matrix(model, &site)?;
        let (d_out, d_in) = w.dim();
        let v = read_f64(&v_path, d_in)?;
        let u = read_f64(&library_dir.join(format!("{}.u.f64", site.name)), d_out)?;
        libraries.push(Library { v, u, mean: Array1::zeros(d_in) });
        chosen.push(site);
    }
    let masked = Masked::build(model, chosen, libraries)?;
    eprintln!("built {} sites in {:.1}s", masked.sites.len(), started.elapsed().as_secs_f64());
    // The given sets, over 512-position sequences, pieces numbered site after site in `sites.txt`.
    let listed = std::fs::read_to_string(sets_dir.join("sites.txt")).map_err(|e| format!("{}: {e}", sets_dir.display()))?;
    let mut offsets = vec![0usize];
    let mut site_of_listing = Vec::new();
    for line in listed.lines().filter(|l| !l.is_empty()) {
        let (name, pieces) = line.split_once(' ').ok_or("sites.txt: name pieces")?;
        let k = masked.sites.iter().position(|s| s.name == name).ok_or_else(|| format!("sites.txt: {name} is not a site"))?;
        let pieces: usize = pieces.parse().map_err(|e| format!("sites.txt: {e}"))?;
        if pieces != masked.pieces(k) {
            return Err(format!("{name}: {pieces} listed, {} in the library", masked.pieces(k)));
        }
        site_of_listing.push(k);
        offsets.push(offsets[offsets.len() - 1] + pieces);
    }
    let indptr = read_i64(&sets_dir.join("indptr.i64"))?;
    let indices = read_i64(&sets_dir.join("indices.i64"))?;
    const SEQUENCE: usize = 512;
    let mut masks: Vec<Array2<f64>> = (0..masked.sites.len()).map(|k| Array2::zeros((family.rows, masked.blocks(k)))).collect();
    for s in 0..sequences {
        for p in 0..context {
            let at = s * SEQUENCE + p;
            let row = s * context + p;
            for &i in &indices[indptr[at] as usize..indptr[at + 1] as usize] {
                let i = i as usize;
                let listing = offsets.partition_point(|o| *o <= i) - 1;
                masks[site_of_listing[listing]][[row, i - offsets[listing]]] = 1.0;
            }
        }
    }
    let (target, radius) = reference(model, family)?;
    let mut gates = Gates::claim(&masks);
    if free != "all" {
        let prefixes: Vec<&str> = free.split(',').collect();
        for (k, site) in masked.sites.iter().enumerate() {
            if !prefixes.iter().any(|p| site.name.starts_with(p)) {
                gates.upper[k] = gates.lower[k].clone();
            }
        }
    }
    eprintln!("sets read: L0 {:.1} per word, {:.1}s", l0(&masks), started.elapsed().as_secs_f64());
    let (kl, _, _) = forward(&masked, &masked.family(family, &masks), &target)?;
    eprintln!("corner KL {:.4} ({:.1}s)", kl.mean().unwrap_or(0.0), started.elapsed().as_secs_f64());
    let found = adversary(&masked, family, &target, &gates, None, steps, 6, 0xAD5)?;
    eprintln!("adversary {:.4} ({:.1}s)", found.mean().unwrap_or(0.0), started.elapsed().as_secs_f64());
    let at = Instant::now();
    let branched = certify_branching(&masked, family, &target, Some(&radius), gates, budget, leaves)?;
    let seconds = at.elapsed().as_secs_f64();
    for r in 0..family.rows {
        if branched.kl[r] < found[r] {
            return Err(format!("row {r}: certified {} below the adversary's {}", branched.kl[r], found[r]));
        }
    }
    eprintln!("certified {:.4}, branched {:.4} ({seconds:.1}s)", branched.root.mean().unwrap_or(0.0), branched.kl.mean().unwrap_or(0.0));
    let record = json!({
        "mode": "lm", "export": export, "library": library_dir, "sets": sets_dir, "sequences": sequences, "context": context,
        "budget": budget, "leaves": branched.leaves, "free": free, "steps": steps, "seconds": seconds,
        "l0": l0(&masks),
        "kl": summary(&kl), "adversary": summary(&found), "certified": summary(&branched.root), "branched": summary(&branched.kl),
        "rows": { "kl": kl.to_vec(), "adversary": found.to_vec(), "certified": branched.root.to_vec(), "branched": branched.kl.to_vec() },
    });
    std::fs::write(&out, serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("toy") => toy(&args),
        Some("lm") => lm(&args),
        _ => Err("mpd_certify_2951 {toy|lm} ...".to_string()),
    }
}

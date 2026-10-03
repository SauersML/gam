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
//! `mpd_certify_2951 lm EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json SEQUENCES CONTEXT [BUDGET] [LEAVES] [FREE] [STEPS] [OBSERVATIONS] [ROUNDING] [CLAIMS]`
//!
//! A language-model export (`import_language_model`), the first `CONTEXT` positions of its first
//! `SEQUENCES` sequences (or of sequences `a..b` for `SEQUENCES` `a:b`, sequence `a` alone for `a:`). Positions only read earlier ones, so a prefix is exact. The sites with a
//! library in `LIBRARY_DIR` (`{site}.v.f64`, `{site}.u.f64`) run on its subcomponents plus the
//! residual `W − Σ u vᵀ` as exact rank-one pieces that are always on (VPD's delta component, held on), so
//! every gate on is the model. `SETS_DIR` gives each position's set (`bench/vpd_2951/vpd_sets_export.py`'s CSR over
//! 512-position sequences; the export's sequences must be the sets' rows in order, as
//! `vpd4l_frontier32` is for `vpd4l_sets`), or `select:DIR`, a masked selection's checkpoint (the
//! `OUT.select/` of `mpd_pieces_masked_2951` on an export whose sequences these are), or `ladder:τ1,τ2,…`: at each level `τ` a word's pieces of
//! the given library whose `|z_c| ‖u_c‖` at every gate on is below `τ` times the word's largest are off.
//! Several sources, comma-separated, are each certified. `FREE` (default `all`) is a comma-separated list of site-name prefixes
//! whose off gates are free; every other off gate stays at 0. The adversary takes `STEPS` steps
//! (default 40) from 6 starts on the sequence's total KL. Each word's certificate uses `BUDGET`
//! symbols (default 512), and the sequence's box is split into `LEAVES` leaves (default 1: no
//! branching). With `OBSERVATIONS` positive, a second set per word is chosen by the masked selection under
//! the box claim (`select_boxed`, coded in `OBSERVATIONS`) from the given sets, and certified the same way.
//! `CLAIMS` lists what each set is certified under: `box` (the default) and `restore:K`, at most `K`
//! off subcomponents restored per word (`gam_mpd::certify::Gates::restoring`), and `sites`, the
//! site-switch claim (`gam_mpd::certify::certify_sites`): per sequence, each site within `FREE` runs
//! its explanation (the given set's subcomponents, the residual and every other one removed) or its
//! native map, at every word alike, in any combination or anything between; sites outside `FREE`
//! run native. Under `sites` the lower side is exact: every word's largest KL over every subset of
//! the free sites replaced when there are at most 8 of them, else over all of them and each alone
//! (no adversary, no branching).
//! `ROUNDING` `real` reports the relaxation's bound without any rounding enclosure (not a proof; the
//! default `sound` is one).

use gam_mpd::blocks::Describe;
use gam_mpd::certify::{Gates, Relaxation, adversary, certify, certify_branching, certify_sites, certify_widening};
use gam_mpd::import::{import, import_language_model};
use gam_mpd::masked::{Coder, Library, Masked, Target, box_excess_at, forward, score_only, select, select_boxed, site_statistics, sites};
use gam_mpd::operator_program::{FamilyInputs, OperatorProgram};
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2, Zip};
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
        let estimate = &kl + &box_excess_at(&masked, &family, &target, &masks)?;
        let gates = Gates::claim(&masks);
        let found = adversary(&masked, &family, &target, &gates, None, 40, 8, 0xAD5)?;
        let root = certify(&masked, &family, &target, Some(&radius), &gates, Relaxation::sound(budget))?;
        let mut branched = Array1::<f64>::zeros(family.rows);
        for r in 0..family.rows {
            let one = family.select(&[r]);
            let row_target = Target::every_row(target.logits.select(ndarray::Axis(0), &[r]));
            let row_radius = radius.select(ndarray::Axis(0), &[r]);
            let row_gates = Gates {
                lower: gates.lower.iter().map(|m| m.select(ndarray::Axis(0), &[r])).collect(),
                upper: gates.upper.iter().map(|m| m.select(ndarray::Axis(0), &[r])).collect(),
                restored: gates.restored,
            };
            branched[r] = certify_branching(&masked, &one, &row_target, Some(&row_radius), row_gates, Relaxation::sound(budget), leaves)?.kl[0];
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
    let usage = "mpd_certify_2951 lm EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json SEQUENCES CONTEXT [BUDGET] [LEAVES] [FREE] [STEPS] [OBSERVATIONS] [ROUNDING] [CLAIMS]";
    let export = PathBuf::from(args.get(2).ok_or(usage)?);
    let library_dir = PathBuf::from(args.get(3).ok_or(usage)?);
    let sets_dir = PathBuf::from(args.get(4).ok_or(usage)?);
    let out = PathBuf::from(args.get(5).ok_or(usage)?);
    // `SEQUENCES` is a count from the first sequence, a range `a:b`, or one sequence `a:`.
    let spec: String = arg(args, 6, "1".to_string())?;
    let (first, sequences) = match spec.split_once(':') {
        Some((a, b)) => {
            let a: usize = a.parse().map_err(|e| format!("SEQUENCES {spec}: {e}"))?;
            let b: usize = if b.is_empty() { a + 1 } else { b.parse().map_err(|e| format!("SEQUENCES {spec}: {e}"))? };
            (a, b.checked_sub(a).filter(|n| *n > 0).ok_or_else(|| format!("SEQUENCES {spec}: an empty range"))?)
        }
        None => (0, spec.parse().map_err(|e| format!("SEQUENCES {spec}: {e}"))?),
    };
    let context: usize = arg(args, 7, 16)?;
    let budget: usize = arg(args, 8, 512)?;
    let leaves: usize = arg(args, 9, 1)?;
    let free: String = arg(args, 10, "all".to_string())?;
    let steps: usize = arg(args, 11, 40)?;
    let observations: f64 = arg(args, 12, 0.0)?;
    // `ROUNDING` (`sound`, the default, or `real`): `real` drops every rounding enclosure
    // (`gam_mpd::certify::Relaxation`), the relaxation's own bound in real arithmetic, not a proof.
    let rounding = match args.get(13).map(String::as_str) {
        None | Some("sound") => true,
        Some("real") => false,
        Some(other) => return Err(format!("ROUNDING {other}: expected sound or real; {usage}")),
    };
    // `CLAIMS`: comma-separated `box` (the default) and `restore:K` (at most `K` off subcomponents
    // restored per word, `gam_mpd::certify::Gates::restoring`).
    // `sites` is the site-switch claim (`gam_mpd::certify::certify_sites`).
    #[derive(Clone, Copy)]
    enum Claim {
        Gates(Option<usize>),
        Sites,
        Singles(usize),
    }
    let claims: Vec<Claim> = arg(args, 14, "box".to_string())?
        .split(',')
        .map(|c| match c {
            "box" => Ok(Claim::Gates(None)),
            "sites" => Ok(Claim::Sites),
            other if other.starts_with("singles:") => {
                other["singles:".len()..].parse().map(Claim::Singles).map_err(|e| format!("CLAIMS {other}: {e}"))
            }
            other => other
                .strip_prefix("restore:")
                .and_then(|k| k.parse().ok())
                .map(|k| Claim::Gates(Some(k)))
                .ok_or_else(|| format!("CLAIMS {other}: expected box, sites, singles:B or restore:K; {usage}")),
        })
        .collect::<Result<_, _>>()?;
    let started = Instant::now();
    let imported = import_language_model(&export, first + sequences, context)?;
    let model = &imported.program;
    let family = &imported.contract.family.select(&(first * context..(first + sequences) * context).collect::<Vec<_>>());
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    // Per site, how many of its pieces are the given library's; the rest hold the residual `W − Σ u vᵀ`.
    let mut given = Vec::new();
    for site in sites(model) {
        let v_path = library_dir.join(format!("{}.v.f64", site.name));
        if !v_path.exists() {
            continue;
        }
        let w = gam_mpd::masked::matrix(model, &site)?;
        let (d_out, d_in) = w.dim();
        let v = read_f64(&v_path, d_in)?;
        let u = read_f64(&library_dir.join(format!("{}.u.f64", site.name)), d_out)?;
        let delta = &w - &u.t().dot(&v);
        // The residual as exact rank-one pieces along the narrower side.
        let (dv, du) = if d_out < d_in { (delta.clone(), Array2::eye(d_out)) } else { (Array2::eye(d_in), delta.t().to_owned()) };
        given.push(v.nrows());
        let v = ndarray::concatenate(ndarray::Axis(0), &[v.view(), dv.view()]).map_err(|e| e.to_string())?;
        let u = ndarray::concatenate(ndarray::Axis(0), &[u.view(), du.view()]).map_err(|e| e.to_string())?;
        libraries.push(Library { v, u, mean: Array1::zeros(d_in) });
        chosen.push(site);
    }
    let masked = Masked::build(model, chosen.clone(), libraries)?;
    eprintln!("built {} sites in {:.1}s", masked.sites.len(), started.elapsed().as_secs_f64());
    let all_on: Vec<Array2<f64>> = (0..masked.sites.len()).map(|k| Array2::ones((family.rows, masked.blocks(k)))).collect();
    let mut named: Vec<(String, Vec<Array2<f64>>)> = Vec::new();
    let sets_arg = sets_dir.to_string_lossy().to_string();
    for spec in sets_arg.split(',') {
    let sets_dir = PathBuf::from(spec.strip_prefix("select:").unwrap_or(spec));
    let label = sets_dir.file_name().map_or_else(|| spec.to_string(), |n| n.to_string_lossy().to_string());
    if let Some(levels) = spec.strip_prefix("ladder:") {
        // Each word's pieces of the given library ranked by `|z_c| ‖u_c‖` at every gate on; at level `τ`
        // the pieces below `τ` times the word's largest are off.
        let trace = masked.program.execute(&masked.family(family, &all_on), false).map_err(|e| e.to_string())?;
        let mut amplitude: Vec<Array2<f64>> = Vec::new();
        for k in 0..masked.sites.len() {
            let norms: Array1<f64> = masked.library(k)?.u.outer_iter().map(|u| u.dot(&u).sqrt()).collect();
            let z = &trace.values[masked.z[k]];
            amplitude.push(Array2::from_shape_fn(z.dim(), |(r, c)| if c < given[k] { z[[r, c]].abs() * norms[c] } else { f64::INFINITY }));
        }
        let largest: Vec<f64> = (0..family.rows)
            .map(|r| amplitude.iter().flat_map(|a| a.row(r).iter().copied().filter(|v| v.is_finite()).collect::<Vec<_>>()).fold(0.0, f64::max))
            .collect();
        for level in levels.split(',') {
            let tau: f64 = level.parse().map_err(|e| format!("ladder level {level}: {e}"))?;
            let masks = amplitude.iter().map(|a| Array2::from_shape_fn(a.dim(), |(r, c)| if a[[r, c]] > tau * largest[r] { 1.0 } else { 0.0 })).collect();
            named.push((format!("ladder {level}"), masks));
        }
    } else if spec.starts_with("select:") {
        // A masked selection's checkpoint (`OUT.select/` of `mpd_pieces_masked_2951`): per finished
        // sequence of its export, per site in the driver's order, CSR over its 512 positions.
        let loaded = gam_mpd::checkpoint::load(&sets_dir)?.ok_or_else(|| format!("{}: no checkpoint", sets_dir.display()))?;
        let mut masks: Vec<Array2<f64>> = (0..masked.sites.len())
            .map(|k| Array2::from_shape_fn((family.rows, masked.blocks(k)), |(_, c)| if c >= given[k] { 1.0 } else { 0.0 }))
            .collect();
        for s in 0..sequences {
            let sites = loaded.sets.get(first + s).and_then(|x| x.as_ref()).ok_or_else(|| format!("{}: no sets for sequence {}", sets_dir.display(), first + s))?;
            if sites.len() != masked.sites.len() {
                return Err(format!("{}: {} sites, {} here", sets_dir.display(), sites.len(), masked.sites.len()));
            }
            for (k, (indptr, indices)) in sites.iter().enumerate() {
                for p in 0..context {
                    for &i in &indices[indptr[p] as usize..indptr[p + 1] as usize] {
                        if i as usize >= given[k] {
                            return Err(format!("{}: piece {i} beyond site {k}'s {}", sets_dir.display(), given[k]));
                        }
                        masks[k][[s * context + p, i as usize]] = 1.0;
                    }
                }
            }
        }
        named.push((label.clone(), masks));
    } else {
        // The given sets, over 512-position sequences, pieces numbered site after site in `sites.txt`; every
        // residual piece is on.
        let listed = std::fs::read_to_string(sets_dir.join("sites.txt")).map_err(|e| format!("{}: {e}", sets_dir.display()))?;
        let mut offsets = vec![0usize];
        let mut site_of_listing = Vec::new();
        for line in listed.lines().filter(|l| !l.is_empty()) {
            let (name, pieces) = line.split_once(' ').ok_or("sites.txt: name pieces")?;
            let k = masked.sites.iter().position(|s| s.name == name).ok_or_else(|| format!("sites.txt: {name} is not a site"))?;
            let pieces: usize = pieces.parse().map_err(|e| format!("sites.txt: {e}"))?;
            if pieces != given[k] {
                return Err(format!("{name}: {pieces} listed, {} in the library", given[k]));
            }
            site_of_listing.push(k);
            offsets.push(offsets[offsets.len() - 1] + pieces);
        }
        let indptr = read_i64(&sets_dir.join("indptr.i64"))?;
        let indices = read_i64(&sets_dir.join("indices.i64"))?;
        const SEQUENCE: usize = 512;
        let mut masks: Vec<Array2<f64>> = (0..masked.sites.len())
            .map(|k| Array2::from_shape_fn((family.rows, masked.blocks(k)), |(_, c)| if c >= given[k] { 1.0 } else { 0.0 }))
            .collect();
        for s in 0..sequences {
            for p in 0..context {
                let at = (first + s) * SEQUENCE + p;
                let row = s * context + p;
                for &i in &indices[indptr[at] as usize..indptr[at + 1] as usize] {
                    let i = i as usize;
                    let listing = offsets.partition_point(|o| *o <= i) - 1;
                    masks[site_of_listing[listing]][[row, i - offsets[listing]]] = 1.0;
                }
            }
        }
        named.push((label.clone(), masks));
    }
    }
    let (target, radius) = reference(model, family)?;
    eprintln!("reference logits' radius: largest {:.3e}", radius.iter().copied().fold(0.0_f64, f64::max));
    // The given library's pieces on per word (the residual pieces are always on).
    let listed_l0 = |masks: &[Array2<f64>]| -> f64 {
        masks.iter().zip(&given).map(|(m, g)| m.slice(ndarray::s![.., ..*g]).iter().filter(|x| **x > 0.0).count()).sum::<usize>() as f64 / family.rows as f64
    };
    let start = named[0].1.clone();
    let mut sets = named;
    if observations > 0.0 {
        // The masked selection under the box claim from the given sets, every word coded in `OBSERVATIONS`.
        let statistics = site_statistics(model, &chosen, [family.clone()], 2, 0x5EED)?;
        let fishers: Vec<Array2<f64>> = statistics.iter().map(|s| s.fisher.clone()).collect();
        let description = gam_mpd::blocks::Generic::new(
            &statistics
                .iter()
                .map(|m| gam_mpd::pieces::Site { w: m.w.clone(), second_moment: m.second_moment.clone(), mean: Array1::zeros(m.mean.len()), fisher: m.fisher.clone() })
                .collect::<Vec<_>>(),
            observations,
        );
        let costs: Vec<Array1<f64>> = (0..masked.sites.len())
            .map(|k| {
                let library = masked.library(k)?;
                (0..library.v.nrows())
                    .map(|c| description.bits(k, library.u.slice(ndarray::s![c..c + 1, ..]), library.v.slice(ndarray::s![c..c + 1, ..])))
                    .collect::<Result<Array1<f64>, String>>()
            })
            .collect::<Result<_, _>>()?;
        let coder = Coder::ran(costs, family.rows);
        let boxed = select_boxed(&masked, family, &target, start, &coder, observations, 2, &fishers)?.0;
        eprintln!("box selection done ({:.1}s)", started.elapsed().as_secs_f64());
        sets.push(("box".to_string(), boxed));
    }
    let mut report = Vec::new();
    let free_site: Vec<bool> = masked.sites.iter().map(|site| free == "all" || free.split(',').any(|p| site.name.starts_with(p))).collect();
    for (name, masks, claim) in sets.into_iter().flat_map(|(n, m)| claims.iter().map(move |c| (n.clone(), m.clone(), *c))) {
        let restored = match claim {
            Claim::Gates(restored) => restored,
            Claim::Singles(batch) => {
                // Every word with each one of its free off subcomponents restored alone, at full
                // strength: the exact worst single restoration at the word, by exhaustion. A word's KL
                // reads only the words up to it, so each candidate runs that prefix, `batch` at a time.
                let layout = family.layout.as_ref().ok_or("singles: a sequence layout")?;
                let at = Instant::now();
                let mut worst = Array1::<f64>::zeros(family.rows);
                let mut argmax: Vec<Option<(usize, usize)>> = vec![None; family.rows];
                let mut candidates = vec![0usize; family.rows];
                for r in 0..family.rows {
                    let prefix: Vec<usize> = (0..family.rows)
                        .filter(|&j| layout.sequence[j] == layout.sequence[r] && layout.position[j] <= layout.position[r])
                        .collect();
                    let at_row = prefix.iter().position(|&j| j == r).ok_or("singles: a row outside its prefix")?;
                    let base = family.select(&prefix);
                    let base_masks: Vec<Array2<f64>> = masks.iter().map(|m| m.select(ndarray::Axis(0), &prefix)).collect();
                    let off: Vec<(usize, usize)> = masks
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| free == "all" || free.split(',').any(|p| masked.sites[*k].name.starts_with(p)))
                        .flat_map(|(k, m)| (0..m.ncols()).filter(move |&b| m[[r, b]] <= 0.0).map(move |b| (k, b)))
                        .collect();
                    candidates[r] = off.len();
                    for chunk in off.chunks(batch.max(1)) {
                        let mut batch_family = base.clone();
                        for _ in 1..chunk.len() {
                            batch_family = batch_family.append(&base).map_err(|e| e.to_string())?;
                        }
                        let n = prefix.len();
                        let batch_masks: Vec<Array2<f64>> = base_masks
                            .iter()
                            .enumerate()
                            .map(|(k, m)| {
                                let mut stacked = Array2::<f64>::zeros((n * chunk.len(), m.ncols()));
                                for (c, &(site, block)) in chunk.iter().enumerate() {
                                    stacked.slice_mut(ndarray::s![c * n..(c + 1) * n, ..]).assign(m);
                                    if site == k {
                                        stacked[[c * n + at_row, block]] = 1.0;
                                    }
                                }
                                stacked
                            })
                            .collect();
                        let logits = target.logits.select(ndarray::Axis(0), &prefix);
                        let views: Vec<_> = (0..chunk.len()).map(|_| logits.view()).collect();
                        let batch_target = Target::every_row(ndarray::concatenate(ndarray::Axis(0), &views).map_err(|e| e.to_string())?);
                        let kl = score_only(&masked, &masked.family(&batch_family, &batch_masks), &batch_target)?;
                        for (c, &candidate) in chunk.iter().enumerate() {
                            let value = kl[c * n + at_row];
                            if value > worst[r] || argmax[r].is_none() {
                                worst[r] = value;
                                argmax[r] = Some(candidate);
                            }
                        }
                    }
                }
                let seconds = at.elapsed().as_secs_f64();
                let (kl, _, _) = forward(&masked, &masked.family(family, &masks), &target)?;
                eprintln!(
                    "{name} under singles: L0 {:.1}, corner KL {:.4}, worst single restoration {:.4} (largest {:.4}) over {} candidates ({seconds:.1}s)",
                    listed_l0(&masks),
                    kl.mean().unwrap_or(0.0),
                    worst.mean().unwrap_or(0.0),
                    worst.iter().copied().fold(0.0, f64::max),
                    candidates.iter().sum::<usize>()
                );
                report.push(json!({
                    "sets": name, "claim": "singles", "l0": listed_l0(&masks), "seconds": seconds,
                    "kl": summary(&kl), "worst_single": summary(&worst),
                    "rows": {
                        "kl": kl.to_vec(), "worst_single": worst.to_vec(), "candidates": candidates,
                        "argmax": argmax.iter().map(|a| a.map(|(k, b)| json!([masked.sites[k].name, b]))).collect::<Vec<_>>(),
                    },
                }));
                continue;
            }
            Claim::Sites => {
                // A replaced site keeps only the set's subcomponents: its residual pieces are off too.
                let on: Vec<Array2<f64>> = masks
                    .iter()
                    .zip(&given)
                    .map(|(m, g)| Array2::from_shape_fn(m.dim(), |(r, c)| if c < *g { m[[r, c]] } else { 0.0 }))
                    .collect();
                let hybrid = |replaced: &dyn Fn(usize) -> bool| -> Vec<Array2<f64>> {
                    on.iter().enumerate().map(|(k, m)| if replaced(k) { m.clone() } else { Array2::ones(m.dim()) }).collect()
                };
                let all = hybrid(&|k| free_site[k]);
                let (kl, _, _) = forward(&masked, &masked.family(family, &all), &target)?;
                let mut found = kl.clone();
                let mut alone = Vec::new();
                let switched: Vec<usize> = (0..masked.sites.len()).filter(|&k| free_site[k]).collect();
                // Every subset of the free sites when there are at most `EXHAUSTIVE` of them (the exact
                // worst corner), else all replaced and each alone.
                const EXHAUSTIVE: usize = 8;
                let subsets: Vec<u64> = if switched.len() <= EXHAUSTIVE {
                    (1..(1u64 << switched.len()) - 1).collect()
                } else {
                    (0..switched.len()).map(|b| 1u64 << b).collect()
                };
                for subset in subsets {
                    let (one, _, _) = forward(&masked, &masked.family(family, &hybrid(&|k| switched.iter().position(|&j| j == k).is_some_and(|b| subset >> b & 1 == 1))), &target)?;
                    Zip::from(&mut found).and(&one).for_each(|f, &o| *f = f.max(o));
                    if subset.is_power_of_two() {
                        alone.push(json!({"site": masked.sites[switched[subset.trailing_zeros() as usize]].name, "kl": summary(&one)}));
                    }
                }
                let at = Instant::now();
                let certified = certify_sites(&masked, family, &target, rounding.then_some(&radius), &on, &free_site, Relaxation { budget, rounding })?;
                let seconds = at.elapsed().as_secs_f64();
                for r in 0..family.rows {
                    if certified[r] < found[r] {
                        return Err(format!("{name} under sites: row {r} certified {} below the exact {}", certified[r], found[r]));
                    }
                }
                eprintln!(
                    "{name} under sites: L0 {:.1}, all replaced KL {:.4}, worst evaluated {:.4}, certified {:.4} ({seconds:.1}s)",
                    listed_l0(&masks),
                    kl.mean().unwrap_or(0.0),
                    found.mean().unwrap_or(0.0),
                    certified.mean().unwrap_or(0.0)
                );
                report.push(json!({
                    "sets": name, "claim": "sites", "l0": listed_l0(&masks), "seconds": seconds,
                    "kl": summary(&kl), "evaluated": summary(&found), "certified": summary(&certified), "alone": alone,
                    "rows": { "kl": kl.to_vec(), "evaluated": found.to_vec(), "certified": certified.to_vec() },
                }));
                continue;
            }
        };
        let claim = restored.map_or_else(|| "box".to_string(), |k| format!("restore:{k}"));
        let mut gates = Gates { restored, ..Gates::claim(&masks) };
        // The off pieces outside `FREE` stay off.
        for (k, site) in masked.sites.iter().enumerate() {
            let pinned = free != "all" && !free.split(',').any(|p| site.name.starts_with(p));
            if pinned {
                gates.upper[k] = gates.lower[k].clone();
            }
        }
        let (kl, _, _) = forward(&masked, &masked.family(family, &masks), &target)?;
        let found = adversary(&masked, family, &target, &gates, None, steps, 6, 0xAD5)?;
        let at = Instant::now();
        // Where the relaxation widens: every node's mean half-width, ball and center at the root.
        let (_, widening) = certify_widening(&masked, family, &target, rounding.then_some(&radius), &gates, Relaxation { budget, rounding })?;
        let lost = widening.iter().find(|w| w.reach + w.ball > w.center).map(|w| json!({"node": w.node, "kind": w.kind, "reach": w.reach, "ball": w.ball, "center": w.center}));
        let branched = certify_branching(&masked, family, &target, rounding.then_some(&radius), gates, Relaxation { budget, rounding }, leaves)?;
        let seconds = at.elapsed().as_secs_f64();
        for r in 0..family.rows {
            if branched.kl[r] < found[r] {
                return Err(format!("{name} under {claim}: row {r} certified {} below the adversary's {}", branched.kl[r], found[r]));
            }
        }
        eprintln!(
            "{name} under {claim}: L0 {:.1}, corner KL {:.4}, adversary {:.4}, certified {:.4}, branched {:.4} ({seconds:.1}s)",
            listed_l0(&masks),
            kl.mean().unwrap_or(0.0),
            found.mean().unwrap_or(0.0),
            branched.root.mean().unwrap_or(0.0),
            branched.kl.mean().unwrap_or(0.0)
        );
        report.push(json!({
            "sets": name, "claim": claim, "l0": listed_l0(&masks), "leaves": branched.leaves, "seconds": seconds,
            "kl": summary(&kl), "adversary": summary(&found), "certified": summary(&branched.root), "branched": summary(&branched.kl),
            "rows": { "kl": kl.to_vec(), "adversary": found.to_vec(), "certified": branched.root.to_vec(), "branched": branched.kl.to_vec() },
            "first_lost_node": lost,
            "widening": widening.iter().map(|w| json!([w.node, w.kind, w.reach, w.ball, w.center])).collect::<Vec<_>>(),
        }));
        eprintln!("{name} under {claim}: first node wider than its center: {lost:?}");
    }
    let record = json!({
        "mode": "lm", "export": export, "library": library_dir, "given": sets_dir, "first": first, "sequences": sequences, "context": context,
        "budget": budget, "free": free, "steps": steps, "observations": observations, "rounding": rounding, "sets": report,
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

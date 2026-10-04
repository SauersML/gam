//! Readers shared across a library's sites, priced per word (#2951).
//!
//! `mpd_shared_2951 EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS TRAIN SITES`
//!
//! `EXPORT_DIR` a language-model export, `LIBRARY_DIR` per site `{site}.{v,u}.f64` (as
//! `mpd_site_fit_2951` writes them), `SETS_DIR` every position's sets over those libraries (CSR,
//! `sites.txt`, as `mpd_site_fit_2951 … library:DIR` writes them), `SITES` the comma-separated sites
//! that read one node. Their readers are given a shared basis, the body of a rule every column may
//! bind to: the leading `r` directions of the readers' error-weighted scatter in the node's metric
//! (`C^{1/2} (Σ_c (u_cᵀ F u_c) v_c v_cᵀ) C^{1/2}`, mapped back by `C^{+1/2}`), for `r` doubling up to
//! the node's width. Every column is described with the basis as a reader chart beside the site's
//! declared ones (`gam_mpd::describe::Chart::frames`, a column naming the directions it uses), and a
//! direction is sent once, its entries as literals. Library once: every column's description plus
//! the directions used; per token, each direction the on columns use once plus their descriptions;
//! both against every column described alone, for each `r`.

use gam_mpd::describe::{Chart, Description, Geometry, LITERAL_BITS, Metric, declared_charts};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{matrix, sites};
use gam_mpd::site_fit::samples;
use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;
use serde_json::json;
use std::path::Path;

fn read_raw<T: Copy>(path: &Path, decode: fn([u8; 8]) -> T) -> Result<Vec<T>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| decode(c.try_into().expect("eight bytes"))).collect())
}

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let values = read_raw(path, f64::from_le_bytes)?;
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
}

/// Per position, the on columns of each named site (site index, column).
fn sets(dir: &Path, names: &[String]) -> Result<Vec<Vec<(usize, usize)>>, String> {
    let listed = std::fs::read_to_string(dir.join("sites.txt")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut offsets = vec![0usize];
    let mut listed_names = Vec::new();
    for line in listed.lines() {
        let (name, n) = line.split_once(' ').ok_or(format!("sites.txt: {line}"))?;
        offsets.push(offsets[offsets.len() - 1] + n.parse::<usize>().map_err(|e| e.to_string())?);
        listed_names.push(name.to_string());
    }
    let indptr = read_raw(&dir.join("indptr.i64"), i64::from_le_bytes)?;
    let indices = read_raw(&dir.join("indices.i64"), i64::from_le_bytes)?;
    let mut out = Vec::new();
    for p in 0..indptr.len() - 1 {
        let mut on = Vec::new();
        for &i in &indices[indptr[p] as usize..indptr[p + 1] as usize] {
            let i = i as usize;
            let k = offsets.partition_point(|o| *o <= i) - 1;
            if let Some(site) = names.iter().position(|n| *n == listed_names[k]) {
                on.push((site, i - offsets[k]));
            }
        }
        out.push(on);
    }
    Ok(out)
}

/// The readers' basis in the node's metric `C`: the eigenvectors of
/// `C^{1/2} (Σ_c w_c v_c v_cᵀ) C^{1/2}` (`v_c` the readers, `w_c = u_cᵀ F u_c` the weight their
/// error is paid at) in decreasing eigenvalue, mapped back by `C^{+1/2}` (`d × d`, columns), so the
/// first `r` carry the most of the columns' error-weighted energy any `r` directions can.
fn basis(moment: &Array2<f64>, readers: &[(&Array2<f64>, Vec<f64>)]) -> Result<Array2<f64>, String> {
    use gam_linalg::roundoff::SymmetricAssembly;
    let d = moment.nrows();
    let e = gam_linalg::decompose::eigh(((moment + &moment.t()) * 0.5).view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let (mut root, mut inverse) = (e.vectors.clone(), e.vectors.clone());
    for (i, l) in e.values.iter().enumerate() {
        let (a, b) = if *l > e.band { (l.sqrt(), 1.0 / l.sqrt()) } else { (0.0, 0.0) };
        root.column_mut(i).mapv_inplace(|x| x * a);
        inverse.column_mut(i).mapv_inplace(|x| x * b);
    }
    let (root, inverse) = (root.dot(&e.vectors.t()), inverse.dot(&e.vectors.t()));
    let mut scatter = Array2::<f64>::zeros((d, d));
    for (v, weights) in readers {
        let whitened = v.dot(&root);
        let weighted = &whitened * &Array1::from(weights.clone()).insert_axis(Axis(1));
        scatter += &whitened.t().dot(&weighted);
    }
    let s = gam_linalg::decompose::eigh(((&scatter + &scatter.t()) * 0.5).view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|a, b| s.values[*b].total_cmp(&s.values[*a]));
    Ok(inverse.dot(&s.vectors.select(Axis(1), &order)))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_shared_2951 EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS TRAIN SITES";
    let arg = |i: usize| args.get(i).ok_or(usage.to_string());
    let (export, library, sets_dir, out) = (Path::new(arg(1)?), Path::new(arg(2)?), Path::new(arg(3)?), Path::new(arg(4)?));
    let observations: f64 = arg(5)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let train: usize = arg(6)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let names: Vec<String> = arg(7)?.split(',').map(str::to_string).collect();
    const CONTEXT: usize = 512;
    let imported = import_language_model(export, train, CONTEXT)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let chosen: Vec<_> = names.iter().map(|n| sites(model).into_iter().find(|s| s.name == *n).ok_or(format!("no site {n}"))).collect::<Result<_, _>>()?;
    if chosen.iter().any(|s| s.reads != chosen[0].reads) {
        return Err("the sites do not read one node".to_string());
    }
    let sequence = |s: usize| family.select(&(s * CONTEXT..(s + 1) * CONTEXT).collect::<Vec<_>>());
    let gathered = samples(model, &chosen, (0..train).map(sequence), 2, 0x5A4E)?;
    let maps: Vec<Array2<f64>> = chosen.iter().map(|site| matrix(model, site)).collect::<Result<_, _>>()?;
    let mut libraries = Vec::new();
    for (site, w) in chosen.iter().zip(&maps) {
        let (d_out, d_in) = w.dim();
        libraries.push((read_f64(&library.join(format!("{}.u.f64", site.name)), d_out)?, read_f64(&library.join(format!("{}.v.f64", site.name)), d_in)?));
    }
    // Per site, its metric, its declared charts, and every column described alone.
    let mut metrics = Vec::new();
    let mut declared = Vec::new();
    let mut alone: Vec<Vec<Description>> = Vec::new();
    for (k, site) in chosen.iter().enumerate() {
        let started = std::time::Instant::now();
        let g = &gathered[k];
        let measured = gam_mpd::pieces::Site { w: maps[k].clone(), second_moment: g.second_moment.clone(), mean: Array1::zeros(maps[k].ncols()), fisher: g.fisher.clone() };
        let metric = Metric::of(&measured, observations);
        let (writers, readers) = declared_charts(model, site)?;
        let plain = Geometry::new(metric.clone(), writers.clone(), readers.clone())?;
        alone.push(describe_all(&plain, &libraries[k])?);
        eprintln!("{}: {} columns described alone, {:.0}s", site.name, libraries[k].0.nrows(), started.elapsed().as_secs_f64());
        metrics.push(metric);
        declared.push((writers, readers));
    }
    // The basis, its directions weighted by the error each column's reader is paid at.
    let weighted: Vec<(&Array2<f64>, Vec<f64>)> = libraries
        .iter()
        .zip(&gathered)
        .map(|((u, v), g)| (v, u.rows().into_iter().map(|r| r.dot(&g.fisher.dot(&r))).collect()))
        .collect();
    let directions = basis(&gathered[0].second_moment, &weighted)?;
    let d = directions.ncols();
    let positions = sets(sets_dir, &names)?;
    let words = positions.len() as f64;
    let before_word: f64 = positions.iter().flatten().map(|&(k, c)| alone[k][c].total()).sum::<f64>() / words;
    let before_once: f64 = alone.iter().flatten().map(Description::total).sum();
    eprintln!("alone: once {before_once:.0} bits, per word {before_word:.0}");
    let mut scans = Vec::new();
    let mut r = 1;
    while r < d {
        let started = std::time::Instant::now();
        let chart = Chart::frames("shared basis", directions.slice(s![.., ..r]).to_owned(), &vec![1; r])?;
        let mut shared: Vec<Vec<Description>> = Vec::new();
        for (k, (writers, readers)) in declared.iter().enumerate() {
            let mut with_basis = readers.clone();
            with_basis.push(chart.clone());
            shared.push(describe_all(&Geometry::new(metrics[k].clone(), writers.clone(), with_basis)?, &libraries[k])?);
        }
        // Each direction a basis user names is sent once, its d entries as literals.
        let mut named = vec![false; r];
        for d in shared.iter().flatten().filter(|d| d.reader.0 == "shared basis") {
            for j in d.reader.1.iter().filter_map(|m| m.first()) {
                named[*j] = true;
            }
        }
        let direction_bits: Vec<f64> = named.iter().map(|n| if *n { LITERAL_BITS * directions.nrows() as f64 } else { 0.0 }).collect();
        let used = |d: &Description| -> Vec<usize> { if d.reader.0 == "shared basis" { d.reader.1.iter().filter_map(|m| m.first().copied()).collect() } else { Vec::new() } };
        let once = shared.iter().flatten().map(Description::total).sum::<f64>() + direction_bits.iter().sum::<f64>();
        let mut per_word = 0.0;
        for on in &positions {
            let mut paid = std::collections::BTreeSet::new();
            for &(k, c) in on {
                per_word += shared[k][c].total();
                paid.extend(used(&shared[k][c]));
            }
            per_word += paid.iter().map(|j| direction_bits[*j]).sum::<f64>();
        }
        per_word /= words;
        let taking = shared.iter().flatten().filter(|d| d.reader.0 == "shared basis").count();
        eprintln!(
            "r = {r}: once {once:.0} bits ({:+.1}%), per word {per_word:.0} ({:+.1}%); {taking} of {} columns bind to the basis, {} directions used, {:.0}s",
            100.0 * (once - before_once) / before_once,
            100.0 * (per_word - before_word) / before_word,
            shared.iter().map(Vec::len).sum::<usize>(),
            named.iter().filter(|n| **n).count(),
            started.elapsed().as_secs_f64()
        );
        scans.push(json!({"directions": r, "once_bits": once, "per_word_bits": per_word, "columns_binding": taking, "directions_used": named.iter().filter(|n| **n).count()}));
        r *= 2;
    }
    let report = json!({
        "sites": names,
        "observations": observations,
        "words": words,
        "alone": {"once_bits": before_once, "per_word_bits": before_word},
        "shared_basis": scans,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// Every column of a library described in `geometry`.
fn describe_all(geometry: &Geometry, (u, v): &(Array2<f64>, Array2<f64>)) -> Result<Vec<Description>, String> {
    (0..u.nrows())
        .into_par_iter()
        .map(|c| gam_linalg::faer_ndarray::with_nested_parallel(|| geometry.describe(u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]))))
        .collect()
}

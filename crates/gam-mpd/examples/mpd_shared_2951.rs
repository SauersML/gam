//! Readers shared across a library's sites, priced per word (#2951).
//!
//! `mpd_shared_2951 EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS TRAIN SITES`
//!
//! `EXPORT_DIR` a language-model export, `LIBRARY_DIR` per site `{site}.{v,u}.f64` (as
//! `mpd_site_fit_2951` writes them), `SETS_DIR` every position's sets over those libraries (CSR,
//! `sites.txt`, as `mpd_site_fit_2951 … library:DIR` writes them), `SITES` the comma-separated sites
//! that read one node. Their readers are proposed as atoms (directions equal up to scale across the
//! columns, greedily), every column is described with the atoms as a reader chart beside the identity
//! (`gam_mpd::describe::Chart::frames`), and an atom is sent on its users' finest reader lattice.
//! Per word, the program that ran pays each atom its on columns use once, plus each on column's
//! description given the atoms; against every on column described alone.

use gam_mpd::codec::signed_delta_len_bits;
use gam_mpd::describe::{Chart, Description, Geometry, Metric, declared_charts};
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

/// Atoms proposed from `rows` (n × d): greedily, each unclaimed row (largest first) with every
/// unclaimed row within `1 − |cos| ≤ 1/d` of it (one coordinate's share of the angle), as their
/// sign-aligned mean. Only a proposal: each column's description takes an atom or not by its total.
fn propose(rows: &Array2<f64>) -> Array2<f64> {
    let norms: Vec<f64> = rows.rows().into_iter().map(|r| r.dot(&r).sqrt()).collect();
    let unit: Vec<Array1<f64>> = rows.rows().into_iter().zip(&norms).map(|(r, n)| if *n > 0.0 { &r / *n } else { r.to_owned() }).collect();
    let mut order: Vec<usize> = (0..rows.nrows()).filter(|i| norms[*i] > 0.0).collect();
    order.sort_by(|a, b| norms[*b].total_cmp(&norms[*a]));
    let mut claimed = vec![false; rows.nrows()];
    let mut atoms = Vec::new();
    for &i in &order {
        if claimed[i] {
            continue;
        }
        let mut sum = unit[i].clone();
        claimed[i] = true;
        let mut members = 1;
        for &j in &order {
            if claimed[j] {
                continue;
            }
            let c = unit[i].dot(&unit[j]);
            if 1.0 - c.abs() <= 1.0 / rows.ncols() as f64 {
                sum.scaled_add(c.signum(), &unit[j]);
                claimed[j] = true;
                members += 1;
            }
        }
        if members > 1 {
            let n = sum.dot(&sum).sqrt();
            atoms.push(sum / n);
        }
    }
    let mut out = Array2::<f64>::zeros((rows.ncols(), atoms.len()));
    for (j, a) in atoms.iter().enumerate() {
        out.column_mut(j).assign(a);
    }
    out
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
    let readers: Vec<_> = libraries.iter().map(|(_, v)| v.view()).collect();
    let atoms = propose(&ndarray::concatenate(Axis(0), &readers).map_err(|e| e.to_string())?);
    eprintln!("{} readers, {} atoms proposed", readers.iter().map(|r| r.nrows()).sum::<usize>(), atoms.ncols());
    let atoms_chart = Chart::frames("shared readers", atoms.clone(), &vec![1; atoms.ncols()])?;
    // Per site, every column described alone and with the atoms.
    let mut alone: Vec<Vec<Description>> = Vec::new();
    let mut shared: Vec<Vec<Description>> = Vec::new();
    for (k, site) in chosen.iter().enumerate() {
        let started = std::time::Instant::now();
        let g = &gathered[k];
        let measured = gam_mpd::pieces::Site { w: maps[k].clone(), second_moment: g.second_moment.clone(), mean: Array1::zeros(maps[k].ncols()), fisher: g.fisher.clone() };
        let (writers, readers) = declared_charts(model, site)?;
        let mut with_atoms = readers.clone();
        with_atoms.push(atoms_chart.clone());
        let plain = Geometry::new(Metric::of(&measured, observations), writers.clone(), readers)?;
        let sharing = Geometry::new(Metric::of(&measured, observations), writers, with_atoms)?;
        let (u, v) = &libraries[k];
        let both: Vec<(Description, Description)> = (0..u.nrows())
            .into_par_iter()
            .map(|c| {
                let (uc, vc) = (u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]));
                Ok((plain.describe(uc, vc)?, sharing.describe(uc, vc)?))
            })
            .collect::<Result<_, String>>()?;
        let (a, b): (Vec<_>, Vec<_>) = both.into_iter().unzip();
        eprintln!("{}: {} columns described, {:.0}s", site.name, a.len(), started.elapsed().as_secs_f64());
        alone.push(a);
        shared.push(b);
    }
    // Each atom on its users' finest reader lattice (the generic core's reader exponent), scaled to
    // a unit largest entry as the pivot chart scales a reader.
    let mut finest = vec![i32::MIN; atoms.ncols()];
    let mut users = vec![0usize; atoms.ncols()];
    for (a, b) in alone.iter().flatten().zip(shared.iter().flatten()) {
        if b.reader.0 == "shared readers"
            && let Some(atom) = b.reader.1.first().and_then(|m| m.first())
        {
            users[*atom] += 1;
            if let Some(p) = a.choice.as_ref().and_then(|c| c.reader_exponent()) {
                finest[*atom] = finest[*atom].max(p);
            }
        }
    }
    let atom_bits: Vec<f64> = (0..atoms.ncols())
        .map(|j| {
            if users[j] == 0 {
                return 0.0;
            }
            let column = atoms.column(j);
            let top = column.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            let scale = 2f64.powi(finest[j]) / top;
            column.iter().map(|x| signed_delta_len_bits((x * scale).round() as i64).map_or(f64::INFINITY, |b| b as f64)).sum::<f64>()
                + signed_delta_len_bits(i64::from(finest[j])).map_or(0.0, |b| b as f64)
        })
        .collect();
    // Per word: the description of the columns on, alone, against the atoms they use once plus
    // their descriptions given the atoms.
    let positions = sets(sets_dir, &names)?;
    let (mut before, mut after, mut words) = (0.0, 0.0, 0.0);
    for on in &positions {
        let mut used = std::collections::BTreeSet::new();
        for &(k, c) in on {
            before += alone[k][c].total();
            let d = &shared[k][c];
            after += d.total();
            if d.reader.0 == "shared readers"
                && let Some(atom) = d.reader.1.first().and_then(|m| m.first())
            {
                used.insert(*atom);
            }
        }
        after += used.iter().map(|j| atom_bits[*j]).sum::<f64>();
        words += 1.0;
    }
    let taking = shared.iter().flatten().filter(|d| d.reader.0 == "shared readers").count();
    eprintln!(
        "per word: {:.0} bits alone, {:.0} with shared readers ({:.1}%); {} of {} columns read an atom, {} atoms used",
        before / words,
        after / words,
        100.0 * (after - before) / before,
        taking,
        shared.iter().map(Vec::len).sum::<usize>(),
        users.iter().filter(|u| **u > 0).count()
    );
    let library_once_before: f64 = alone.iter().flatten().map(Description::total).sum();
    let library_once_after: f64 = shared.iter().flatten().map(Description::total).sum::<f64>() + atom_bits.iter().sum::<f64>();
    let report = json!({
        "sites": names,
        "observations": observations,
        "words": words,
        "per_word_bits": {"alone": before / words, "shared": after / words},
        "library_once_bits": {"alone": library_once_before, "shared": library_once_after},
        "atoms_proposed": atoms.ncols(),
        "atoms_used": users.iter().filter(|u| **u > 0).count(),
        "columns_reading_an_atom": taking,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

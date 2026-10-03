//! Structured descriptions of rank-k blocks (`gam_mpd::describe`) on a trained toy (#2951).
//!
//! `mpd_describe_2951 modadd EXPORT_DIR OUT.json OBSERVATIONS`
//!
//! `EXPORT_DIR` a `transformer` export on a finite family whose token slots carry the operands
//! (e.g. `~/mpd-data/engine/p31_s0_generic`, the mod-31 adder on all 961 inputs). The code is the
//! blocks' one total (`gam_mpd::blocks`): over every word the description bits of the blocks that
//! ran on it plus `n KL / ln 2`. Two descriptions price a block:
//!
//! * generic (`gam_mpd::blocks::Generic`): a rank-k map's `k (d_in + d_out − k)` reals;
//! * structured (`gam_mpd::describe::Structured`): its cheapest description over the identity, the
//!   harmonic chart of the site's reads on the family (their profiles over the operands'
//!   characters) and, on a site writing the residual the readout reads directly, the harmonic chart
//!   of the unembedding over the classes.
//!
//! Under each, the rank-one point (Fisher-SVD subcomponents, selection passes until one no longer
//! lowers the total) and the blocks fitted from it (`fit_blocks`). Every point reports bits per
//! word, KL per word and the active description per word, under both descriptions and under the
//! generic family alone on the structured families' exact lattice code; every final
//! structured block its family, modes and reals against its generic statement and its columns as
//! generic rank-one subcomponents.
//!
//! `mpd_describe_2951 vpd EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS FIRST SEQUENCES STATISTICS`
//!
//! VPD's four-layer model (`gam_mpd::import::import_language_model`, contexts of 512) and its
//! rank-one subcomponents (`LIBRARY_DIR`, per site `{site}.v.f64` and `{site}.u.f64`; `SETS_DIR` the
//! per-token sets as `mpd_blocks_2951 vpd` reads them), on the attention sites. Every subcomponent
//! is described generic and structured: query and key writers in the heads' and rotary planes'
//! coordinates, key writers also in the frames of the layer's query writers, value writers in the
//! heads', output readers in the heads' and in the frames of the layer's value writers, and every
//! residual reader in the frames of the earlier layers' output writers. Per site: the mean bits of a
//! subcomponent under each, the families chosen, and the active description per word on tokens
//! `FIRST..FIRST + SEQUENCES` of the given sets (`Σ_c count_c bits_c / words`). Statistics are
//! measured on the first `STATISTICS` of those sequences.

use gam_mpd::blocks::{Bits, Blocked, Coded, Describe, Generic, fit_blocks, measure, reselect};
use gam_mpd::describe::{Chart, Core, Geometry, Metric, Structured};
use gam_mpd::import::import;
use gam_mpd::masked::{Library, Site, Target, read_values, site_statistics, sites};
use gam_mpd::operator_program::{LabelKind, Node, OperatorBody, OperatorProgram, SlotValues};
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array2, s};
use serde_json::json;
use std::path::Path;

fn say(name: &str, bits: &Bits) {
    let (per_word, kl, active, rank) = bits.per_row();
    eprintln!(
        "{name}: {per_word:.2} bits/word (described {:.0}, KL {:.0}), KL {kl:.5} nats/word, {active:.2} blocks and {rank:.2} rank-one equivalents on per word, {} blocks",
        bits.described, bits.kl, bits.blocks
    );
}

/// The unembedding when `node` reaches the readout through identity terms only (a residual write
/// the readout reads directly), on the side's coordinates.
fn direct_readout(program: &OperatorProgram, node: usize) -> Option<Array2<f64>> {
    let tokens = |i: &gam_mpd::operator_program::Interface| i.groups().iter().any(|g| g.label.kind == LabelKind::Token);
    let mut at = node;
    loop {
        let mut next = None;
        for (index, n) in program.nodes.iter().enumerate() {
            let Node::Affine { terms, .. } = n else { continue };
            for (argument, op) in terms {
                if *argument != at {
                    continue;
                }
                let operator = &program.operators[*op];
                if tokens(&operator.rows) && !tokens(&operator.cols) {
                    return Some(operator.matrix());
                }
                if matches!(operator.body, OperatorBody::Identity) {
                    next = Some(index);
                }
            }
        }
        at = next?;
    }
}

/// The written side's readout chart: per written node the unembedding where the readout reads it
/// directly, zero elsewhere (the readout does not see it), as one `classes × d_out` token map.
fn readout_map(program: &OperatorProgram, site: &Site) -> Result<Option<Array2<f64>>, String> {
    let interfaces = program.interfaces().map_err(|e| e.to_string())?;
    let widths: Vec<usize> = site.writes.iter().map(|n| interfaces[*n].width()).collect();
    let maps: Vec<Option<Array2<f64>>> = site.writes.iter().map(|n| direct_readout(program, *n)).collect();
    let Some(classes) = maps.iter().flatten().map(|m| m.nrows()).next() else { return Ok(None) };
    let mut out = Array2::<f64>::zeros((classes, widths.iter().sum()));
    let mut at = 0;
    for (w, m) in widths.iter().zip(&maps) {
        if let Some(m) = m {
            out.slice_mut(s![.., at..at + w]).assign(m);
        }
        at += w;
    }
    Ok(Some(out))
}

/// Each block's firing fraction over the coded rows.
fn firing(blocked: &Blocked, k: usize, b: usize) -> f64 {
    let (on, rows) = blocked.masks.iter().fold((0.0, 0.0), |(on, rows), masks| (on + masks[k].column(b).sum(), rows + masks[k].nrows() as f64));
    on / rows.max(1.0)
}

/// The description bits of the blocks on, per word, under `describe`.
fn active_description(blocked: &Blocked, describe: &dyn Describe) -> Result<f64, String> {
    let mut prices = Vec::new();
    for (k, ranks) in blocked.ranks.iter().enumerate() {
        let mut site = Vec::new();
        for c in 0..ranks.len() {
            let (u, v) = blocked.factors(k, c);
            site.push(describe.bits(k, u, v)?);
        }
        prices.push(site);
    }
    let (mut bits, mut rows) = (0.0, 0.0);
    for masks in &blocked.masks {
        rows += masks.first().map_or(0, |m| m.nrows()) as f64;
        for (k, m) in masks.iter().enumerate() {
            for (c, price) in prices[k].iter().enumerate() {
                bits += m.column(c).sum() * price;
            }
        }
    }
    Ok(bits / rows.max(1.0))
}

/// The rank-one point under `coded`: selection passes until one no longer lowers the total.
fn rank_one_point(coded: &Coded<'_>, libraries: Vec<Library>, rows: usize) -> Result<(Blocked, Bits), String> {
    let masks = vec![libraries.iter().map(|l| Array2::<f64>::ones((rows, l.v.nrows()))).collect()];
    let mut blocked = Blocked::rank_one(libraries, masks);
    blocked.price(coded)?;
    let (mut bits, _) = measure(coded, &blocked)?;
    loop {
        let next = reselect(coded, &blocked)?;
        let (next_bits, _) = measure(coded, &next)?;
        if next_bits.total() >= bits.total() {
            return Ok((blocked, bits));
        }
        (blocked, bits) = (next, next_bits);
    }
}

fn core_name(core: &Core) -> String {
    match core {
        Core::Generic { rank } => format!("generic rank {rank}"),
        Core::Rotation { reflections } => format!("rotation-scaling ({} reflected of {})", reflections.iter().filter(|r| **r).count(), reflections.len()),
        Core::Diagonal => "diagonal".to_string(),
        Core::SameSubspace { rank, core } => format!("same subspace rank {rank}, {core:?} core"),
    }
}

fn modadd(dir: &Path, out: &Path, observations: f64) -> Result<(), String> {
    let imported = import(dir)?;
    let program = imported.program;
    let family = imported.contract.family;
    let trace = program.execute(&family, false).map_err(|e| e.to_string())?;
    let target = Target::every_row(trace.values[program.output].clone());
    let chosen = sites(&program);
    // The operands: every token slot that varies over the family.
    let operands: Vec<&Vec<u32>> = family
        .slots
        .iter()
        .filter_map(|v| match v {
            SlotValues::Tokens(t) if t.iter().any(|x| *x != t[0]) => Some(t),
            _ => None,
        })
        .collect();
    let period = operands.iter().flat_map(|t| t.iter()).max().map_or(0, |m| *m as usize + 1);
    let labels = Array2::from_shape_fn((family.rows, operands.len()), |(r, i)| operands[i][r] as usize);
    eprintln!("{} rows, {} operands of period {period}", family.rows, operands.len());

    let statistics = site_statistics(&program, &chosen, [family.clone()], 16, 0x5EED)?;
    let mut libraries = Vec::new();
    let mut structured_sites = Vec::new();
    let mut lattice_sites = Vec::new();
    for (site, measured) in chosen.iter().zip(&statistics) {
        let library = fisher_svd(measured)?;
        libraries.push(Library { v: library.v.t().to_owned(), u: library.u, mean: measured.mean.clone() });
        let reads = read_values(&trace, site)?;
        let readers = vec![Chart::harmonic("operand characters of the reads", reads.view(), labels.view(), period)?];
        let mut writers = Vec::new();
        if let Some(map) = readout_map(&program, site)? {
            let classes = Array2::from_shape_fn((map.nrows(), 1), |(c, _)| c);
            writers.push(Chart::harmonic("class characters of the readout", map.view(), classes.view(), map.nrows())?);
        }
        eprintln!("{}: {}×{}, {} subcomponents, {} writer charts", site.name, measured.w.nrows(), measured.w.ncols(), libraries.last().map_or(0, |l| l.u.nrows()), writers.len());
        structured_sites.push(Geometry::new(Metric::of(measured, observations), writers, readers, false)?);
        lattice_sites.push(Geometry::new(Metric::of(measured, observations), Vec::new(), Vec::new(), false)?);
    }
    // The generic family alone on the same exact lattice code, so the structured families are
    // compared with the identity charts under one code.
    let lattice = Structured { sites: lattice_sites };
    let generic = Generic::new(&statistics, observations);
    let structured = Structured { sites: structured_sites };
    let batches = vec![(family.clone(), target)];
    let coded_generic = Coded { model: &program, sites: chosen.clone(), batches: batches.clone(), observations, samples: 16, describe: &generic };
    let coded_structured = Coded { model: &program, sites: chosen.clone(), batches, observations, samples: 16, describe: &structured };

    let mut points = Vec::new();
    let mut fitted = Vec::new();
    for (name, coded) in [("generic", &coded_generic), ("structured", &coded_structured)] {
        let started = std::time::Instant::now();
        let (rank_one, rank_one_bits) = rank_one_point(coded, libraries.clone(), family.rows)?;
        say(&format!("rank one, {name}"), &rank_one_bits);
        let (blocked, block_bits) = fit_blocks(coded, rank_one.clone(), true)?;
        say(&format!("blocks, {name}"), &block_bits);
        eprintln!("  {name}: {:.1} s", started.elapsed().as_secs_f64());
        for (point, decomposition, bits) in [("rank one", &rank_one, &rank_one_bits), ("blocks", &blocked, &block_bits)] {
            let (per_word, kl, active_blocks, active_rank) = bits.per_row();
            points.push(json!({
                "point": format!("{point}, fitted under the {name} description"),
                "bits_per_word": per_word,
                "kl_per_word": kl,
                "active_description_bits_per_word": bits.described / bits.rows.max(1.0),
                "active_generic_description_bits_per_word": active_description(decomposition, &generic)?,
                "active_structured_description_bits_per_word": active_description(decomposition, &structured)?,
                "active_lattice_generic_description_bits_per_word": active_description(decomposition, &lattice)?,
                "active_blocks_per_word": active_blocks,
                "active_rank_one_equivalents_per_word": active_rank,
                "blocks": bits.blocks,
            }));
        }
        fitted.push(blocked);
    }
    // Every block of the structured fit: its family, against its generic statement and its columns
    // as generic rank-one subcomponents.
    let blocked = &fitted[1];
    let mut per_block = Vec::new();
    for (k, site) in chosen.iter().enumerate() {
        for (b, rank) in blocked.ranks[k].iter().enumerate() {
            let (u, v) = blocked.factors(k, b);
            let d = structured.sites[k].describe(u, v)?;
            let mut rank_one_bits = 0.0;
            for c in 0..*rank {
                rank_one_bits += generic.bits(k, u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]))?;
            }
            per_block.push(json!({
                "site": site.name,
                "rank": rank,
                "firing": firing(blocked, k, b),
                "writer": d.writer.0, "writer_modes": d.writer.1,
                "reader": d.reader.0, "reader_modes": d.reader.1,
                "core": core_name(&d.core),
                "reals": d.reals,
                "bits": d.bits(),
                "error_bits": d.kl_bits,
                "generic_bits": generic.bits(k, u, v)?,
                "lattice_generic_bits": lattice.bits(k, u, v)?,
                "rank_one_generic_bits": rank_one_bits,
            }));
        }
    }
    let report = json!({ "observations": observations, "points": points, "blocks": per_block });
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn read_raw<T: Copy>(path: &Path, decode: fn([u8; 8]) -> T) -> Result<Vec<T>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| decode(c.try_into().expect("eight bytes"))).collect())
}

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let values = read_raw(path, f64::from_le_bytes)?;
    if values.len() % cols != 0 {
        return Err(format!("{}: {} values in rows of {cols}", path.display(), values.len()));
    }
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
}

/// Per listed site (`sites.txt` order), each subcomponent's count of words on, over words
/// `first·context..(first + sequences)·context` of the given sets.
fn counts(dir: &Path, context: usize, first: usize, sequences: usize) -> Result<Vec<(String, Vec<f64>)>, String> {
    let listed = std::fs::read_to_string(dir.join("sites.txt")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    let mut offsets = vec![0usize];
    for line in listed.lines() {
        let (name, n) = line.split_once(' ').ok_or(format!("sites.txt: {line}"))?;
        let n: usize = n.parse().map_err(|e| format!("sites.txt: {e}"))?;
        offsets.push(offsets[offsets.len() - 1] + n);
        out.push((name.to_string(), vec![0.0; n]));
    }
    let indptr = read_raw(&dir.join("indptr.i64"), i64::from_le_bytes)?;
    let indices = read_raw(&dir.join("indices.i64"), i64::from_le_bytes)?;
    let (start, end) = (first * context, (first + sequences) * context);
    if indptr.len() <= end {
        return Err(format!("{}: fewer than {end} words of sets", dir.display()));
    }
    for &i in &indices[indptr[start] as usize..indptr[end] as usize] {
        let i = i as usize;
        let k = offsets.partition_point(|o| *o <= i) - 1;
        out[k].1[i - offsets[k]] += 1.0;
    }
    Ok(out)
}

/// The frames chart of a library's writers (`u`, `C × d`), one group a subcomponent.
fn writer_frames(name: &str, u: &[&Array2<f64>]) -> Result<Option<Chart>, String> {
    if u.is_empty() {
        return Ok(None);
    }
    let views: Vec<_> = u.iter().map(|x| x.t()).collect();
    let columns = ndarray::concatenate(ndarray::Axis(1), &views).map_err(|e| e.to_string())?;
    let widths = vec![1; columns.ncols()];
    Ok(Some(Chart::frames(name, columns, &widths)?))
}

fn vpd(dir: &Path, library_dir: &Path, sets_dir: &Path, out: &Path, observations: f64, first: usize, sequences: usize, statistics_sequences: usize) -> Result<(), String> {
    use rayon::prelude::*;
    const CONTEXT: usize = 512;
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let config = |key: &str| record["config"][key].as_u64().map(|v| v as usize).ok_or(format!("config.{key}"));
    let (heads, head_dim) = (config("n_heads")?, config("head_dim")?);
    let imported = gam_mpd::import::import_language_model(dir, first + sequences, CONTEXT)?;
    let program = imported.program;
    let family = imported.contract.family;
    let kinds = ["q", "k", "v", "o"];
    let chosen: Vec<Site> = sites(&program)
        .into_iter()
        .filter(|s| kinds.iter().any(|k| s.name.ends_with(&format!(".{k}"))) && library_dir.join(format!("{}.v.f64", s.name)).exists())
        .collect();
    let rows: Vec<usize> = (first * CONTEXT..(first + statistics_sequences) * CONTEXT).collect();
    let statistics = site_statistics(&program, &chosen, [family.select(&rows)], 4, 0x5EED)?;
    let mut libraries: Vec<(Array2<f64>, Array2<f64>)> = Vec::new();
    for (site, measured) in chosen.iter().zip(&statistics) {
        let (d_out, d_in) = measured.w.dim();
        let v = read_f64(&library_dir.join(format!("{}.v.f64", site.name)), d_in)?;
        let u = read_f64(&library_dir.join(format!("{}.u.f64", site.name)), d_out)?;
        libraries.push((u, v));
    }
    let index = |l: usize, kind: &str| chosen.iter().position(|s| s.name == format!("blocks.{l}.{kind}"));
    let heads_chart = Chart::coordinates("heads", heads * head_dim, &(0..heads).map(|h| (h * head_dim..(h + 1) * head_dim).collect()).collect::<Vec<_>>())?;
    let half = head_dim / 2;
    let planes: Vec<Vec<usize>> = (0..heads).flat_map(|h| (0..half).map(move |i| vec![h * head_dim + i, h * head_dim + i + half])).collect();
    let rotary_chart = Chart::coordinates("rotary planes", heads * head_dim, &planes)?;
    let sets = counts(sets_dir, CONTEXT, first, sequences)?;
    let words = (sequences * CONTEXT) as f64;
    let generic = Generic::new(&statistics, observations);
    let mut report = Vec::new();
    for (k, site) in chosen.iter().enumerate() {
        let layer: usize = site.name.split('.').nth(1).and_then(|x| x.parse().ok()).ok_or(format!("{}: no layer", site.name))?;
        let kind = site.name.rsplit('.').next().unwrap_or("");
        let earlier: Vec<&Array2<f64>> = (0..layer).filter_map(|l| index(l, "o")).map(|i| &libraries[i].0).collect();
        let residual_readers: Vec<Chart> = writer_frames("earlier output writers", &earlier)?.into_iter().collect();
        let (writers, readers): (Vec<Chart>, Vec<Chart>) = match kind {
            "q" => (vec![heads_chart.clone(), rotary_chart.clone()], residual_readers),
            "k" => {
                let mut w = vec![heads_chart.clone(), rotary_chart.clone()];
                if let Some(q) = index(layer, "q") {
                    w.extend(writer_frames("the layer's query writers", &[&libraries[q].0])?);
                }
                (w, residual_readers)
            }
            "v" => (vec![heads_chart.clone()], residual_readers),
            _ => {
                let mut r = vec![heads_chart.clone()];
                if let Some(v) = index(layer, "v") {
                    r.extend(writer_frames("the layer's value writers", &[&libraries[v].0])?);
                }
                (Vec::new(), r)
            }
        };
        let metric = Metric::of(&statistics[k], observations);
        let structured = Geometry::new(metric.clone(), writers, readers, false)?;
        let lattice = Geometry::new(metric, Vec::new(), Vec::new(), false)?;
        let (u, v) = &libraries[k];
        let started = std::time::Instant::now();
        let described: Vec<(f64, f64, f64, String, String, String)> = (0..u.nrows())
            .into_par_iter()
            .map(|c| {
                let (uc, vc) = (u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]));
                let d = structured.describe(uc, vc)?;
                let plain = lattice.describe(uc, vc)?;
                Ok((generic.bits(k, uc, vc)?, plain.total(), d.total(), d.writer.0.clone(), d.reader.0.clone(), core_name(&d.core)))
            })
            .collect::<Result<_, String>>()?;
        let count = &sets.iter().find(|(n, _)| *n == site.name).ok_or(format!("{}: not in the sets", site.name))?.1;
        let mean = |f: &dyn Fn(&(f64, f64, f64, String, String, String)) -> f64| described.iter().map(f).sum::<f64>() / described.len().max(1) as f64;
        let active = |f: &dyn Fn(&(f64, f64, f64, String, String, String)) -> f64| described.iter().zip(count).map(|(d, n)| f(d) * n).sum::<f64>() / words;
        let mut families = std::collections::BTreeMap::<String, usize>::new();
        for d in &described {
            *families.entry(format!("writer: {}, reader: {}, {}", d.3, d.4, d.5)).or_default() += 1;
        }
        eprintln!(
            "{}: {} subcomponents in {:.1} s; mean bits generic {:.0}, lattice {:.0}, structured {:.0}; active per word generic {:.1}, lattice {:.1}, structured {:.1}",
            site.name,
            described.len(),
            started.elapsed().as_secs_f64(),
            mean(&|d| d.0),
            mean(&|d| d.1),
            mean(&|d| d.2),
            active(&|d| d.0),
            active(&|d| d.1),
            active(&|d| d.2)
        );
        report.push(json!({
            "site": site.name,
            "subcomponents": described.len(),
            "mean_bits": {"generic": mean(&|d| d.0), "lattice_generic": mean(&|d| d.1), "structured": mean(&|d| d.2)},
            "active_description_bits_per_word": {"generic": active(&|d| d.0), "lattice_generic": active(&|d| d.1), "structured": active(&|d| d.2)},
            "active_subcomponents_per_word": count.iter().sum::<f64>() / words,
            "families": families,
        }));
    }
    let report = json!({"observations": observations, "words": words, "sites": report});
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let number = |i: usize, what: &str| -> Result<f64, String> { args.get(i).ok_or(format!("missing {what}"))?.parse::<f64>().map_err(|e| format!("{what}: {e}")) };
    match args.get(1).map(String::as_str) {
        Some("modadd") if args.len() == 5 => modadd(Path::new(&args[2]), Path::new(&args[3]), number(4, "OBSERVATIONS")?),
        Some("vpd") if args.len() == 10 => vpd(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
            number(6, "OBSERVATIONS")?,
            number(7, "FIRST")? as usize,
            number(8, "SEQUENCES")? as usize,
            number(9, "STATISTICS")? as usize,
        ),
        _ => Err("usage: mpd_describe_2951 modadd EXPORT_DIR OUT.json OBSERVATIONS | vpd EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS FIRST SEQUENCES STATISTICS".to_string()),
    }
}

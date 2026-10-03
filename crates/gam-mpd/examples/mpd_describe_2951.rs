//! Structured descriptions of rank-k blocks (`gam_mpd::describe`) on a trained toy (#2951).
//!
//! `mpd_describe_2951 EXPORT_DIR OUT.json OBSERVATIONS`
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

use gam_mpd::blocks::{Bits, Blocked, Coded, Describe, Generic, fit_blocks, measure, reselect};
use gam_mpd::describe::{Chart, Core, Metric, Structured, StructuredSite, describe};
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

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: mpd_describe_2951 EXPORT_DIR OUT.json OBSERVATIONS".to_string());
    }
    let (dir, out) = (Path::new(&args[1]), Path::new(&args[2]));
    let observations: f64 = args[3].parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
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
        structured_sites.push(StructuredSite { metric: Metric::of(measured, observations), writers, readers, same_space: false });
        lattice_sites.push(StructuredSite { metric: Metric::of(measured, observations), writers: Vec::new(), readers: Vec::new(), same_space: false });
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
            let s_ = &structured.sites[k];
            let d = describe(&u.t().dot(&v), *rank, &s_.metric, &s_.writers, &s_.readers, false)?;
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

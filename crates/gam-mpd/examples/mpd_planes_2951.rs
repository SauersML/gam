//! The Fourier mechanism of the mod-31 adder written by hand as blocks, scored by the per-word total
//! against the fitted solution (#2951).
//!
//! `mpd_planes_2951 EXPORT_DIR OUT.json OBSERVATIONS`
//!
//! `EXPORT_DIR` a `transformer` export on the whole family of a modular adder (e.g.
//! `~/mpd-data/engine/p31_s0_generic`, all 961 inputs). Each decomposed site's map `W` is cut by
//! the frequency of what it writes over the inputs: with `X` the site's reads and `Y = X Wᵀ` its
//! written values over the family, and `Π_f` the projector onto the character pair `cos, sin` of
//! `2π (f · (a, b))/p` for `f` one of `(k, 0)`, `(0, k)`, `(k, k)`, `(k, −k)`, the plane `f` is the
//! least-squares map `W_f = (X⁺ Π_f Y)ᵀ`, of rank at most two (a rotation-scaling between the reads
//! and the writes along that character); the constant character gives one more block, and the rest
//! of `W` (other characters, and what `W` does off the reads' span) one more, so the blocks sum to
//! `W`. A plane with nothing beyond the site's rounding band is left out.
//!
//! The total is the blocks' one total (`gam_mpd::blocks`) under the box claim (every word's error
//! its masks' KL plus what its off blocks anywhere in `[0, 1]` would add), every block described on
//! the exact lattice code with the harmonic charts (`gam_mpd::describe::Structured`: a reader in
//! the operand characters of a site's reads where no decomposed site upstream moves them, a writer
//! in the class characters of the unembedding where the readout reads the site directly), in the
//! logit-space Gauss–Newton metric, the price recalibrated against the exact rounding KL until
//! within a factor of two. Points, each decoded (every block replaced by its exact-priced
//! description, `Geometry::describe_exact`, in decoding order) and measured by the exact masked
//! forward:
//!
//! * the planes all on, and selected (selection passes until one no longer lowers the total);
//! * the fitted solution: Fisher-SVD rank-one subcomponents selected, then `fit_blocks`;
//! * the fit seeded from the selected planes, to see whether it keeps them.
//!
//! Per point, the terms per word (description bits of the blocks that ran, `n KL / ln 2`), blocks
//! and rank-one equivalents on, and per plane block its frequency, rank, firing and description
//! bits against its columns described as rank-one subcomponents.

use gam_mpd::blocks::{Bits, Blocked, Coded, fit_blocks, measure, reselect, rounding_error};
use gam_mpd::dense::{QrMode, qr, svd};
use gam_mpd::describe::{Chart, Geometry, Metric, Structured, logit_gauss_newton};
use gam_mpd::import::import;
use gam_mpd::masked::{Library, Site, Target, matrix, read_values, site_statistics, sites};
use gam_mpd::operator_program::{LabelKind, Node, OperatorBody, OperatorProgram, SlotValues};
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2, Axis, s};
use serde_json::{Value, json};
use std::path::Path;

fn say(name: &str, bits: &Bits) {
    let (per_word, kl, active, rank) = bits.per_row();
    eprintln!(
        "{name}: {per_word:.1} bits/word (described {:.1}, error {:.1}), KL {kl:.6} nats/word, {active:.2} blocks and {rank:.2} rank-one equivalents on per word, {} blocks",
        bits.described / bits.rows.max(1.0),
        bits.kl / bits.rows.max(1.0),
        bits.blocks
    );
}

/// The unembedding when `node` reaches the readout through identity terms only.
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

/// The readout on a site's written coordinates, when the readout reads the site directly.
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

/// `blocked` with block `c` of site `k` replaced by the factors `u`, `v`.
fn replaced(blocked: &Blocked, k: usize, c: usize, u: &Array2<f64>, v: &Array2<f64>) -> Result<Blocked, String> {
    let mut out = blocked.clone();
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for b in 0..blocked.ranks[k].len() {
        let (bu, bv) = if b == c { (u.view(), v.view()) } else { blocked.factors(k, b) };
        us.push(bu);
        vs.push(bv);
    }
    out.libraries[k] = std::sync::Arc::new(Library {
        u: ndarray::concatenate(Axis(0), &us).map_err(|e| e.to_string())?,
        v: ndarray::concatenate(Axis(0), &vs).map_err(|e| e.to_string())?,
        mean: blocked.libraries[k].mean.clone(),
    });
    out.ranks[k][c] = u.nrows();
    Ok(out)
}

/// The decoded point (module note): every block that runs replaced by its exact-priced description
/// in decoding order, then measured together. `(description bits, error bits, KL nats)` per word,
/// and each block's decoded bits.
fn decoded(coded: &Coded<'_>, blocked: &Blocked, geometry: &Structured) -> Result<(f64, f64, f64, Vec<Vec<f64>>), String> {
    let mut out = blocked.clone();
    let mut prices: Vec<Vec<f64>> = Vec::new();
    for (k, ranks) in blocked.ranks.iter().enumerate() {
        let mut site = Vec::new();
        for c in 0..ranks.len() {
            let on: f64 = blocked.masks.iter().map(|m| m[k].column(c).sum()).sum::<f64>();
            if on == 0.0 {
                site.push(0.0);
                continue;
            }
            let (base, _) = measure(coded, &out)?;
            let (u, v) = blocked.factors(k, c);
            let current = out.clone();
            let d = geometry.sites[k].describe_exact(u, v, &mut |d| {
                let (bits, _) = measure(coded, &replaced(&current, k, c, &d.u, &d.v)?)?;
                Ok((bits.kl - base.kl) / on)
            })?;
            site.push(d.bits());
            out = replaced(&out, k, c, &d.u, &d.v)?;
        }
        prices.push(site);
    }
    let (bits, _) = measure(coded, &out)?;
    let mut described = 0.0;
    for masks in &blocked.masks {
        for (k, m) in masks.iter().enumerate() {
            for (c, price) in prices[k].iter().enumerate() {
                described += m.column(c).sum() * price;
            }
        }
    }
    let rows = bits.rows.max(1.0);
    Ok((described / rows, bits.kl / rows, bits.kl_nats / rows, prices))
}

/// Selection passes from `blocked` until one no longer lowers the total.
fn selected(coded: &Coded<'_>, mut blocked: Blocked) -> Result<(Blocked, Bits), String> {
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

/// A map as balanced rank-r factors `(u: r × d_out, v: r × d_in)` over its singular values beyond
/// `band` (the whole site's rounding band, so a block that is rounding of the site has none).
fn factors(w: &Array2<f64>, band: f64) -> Result<(Array2<f64>, Array2<f64>), String> {
    let d = svd(w.view(), false).map_err(|e| format!("{e:?}"))?;
    let kept: Vec<usize> = (0..d.singular_values.len()).filter(|&j| d.singular_values[j] > d.band.max(band)).collect();
    let roots = Array1::from_iter(kept.iter().map(|&j| d.singular_values[j].sqrt()));
    let u = (d.u.select(Axis(1), &kept) * &roots).t().to_owned();
    let v = (d.vt.select(Axis(0), &kept).t().to_owned() * &roots).t().to_owned();
    Ok((u, v))
}

/// The plane blocks of a site (module note): per character pair `±f` of the operands (`f` one of
/// `(k, 0)`, `(0, k)`, `(k, k)`, `(k, −k)`, `k = 1..=p/2`) a block of rank at most two, the constant,
/// and the rest; `(label, u, v)` with the empty ones (nothing beyond the site's rounding band) left
/// out.
fn planes(w: &Array2<f64>, x: &Array2<f64>, labels: &Array2<usize>, period: usize) -> Result<Vec<(String, Array2<f64>, Array2<f64>)>, String> {
    let rows = x.nrows();
    let y = x.dot(&w.t());
    let band = svd(w.view(), false).map_err(|e| format!("{e:?}"))?.band;
    // X⁺ over the reads' singular values beyond the band.
    let dx = svd(x.view(), false).map_err(|e| format!("{e:?}"))?;
    let kept: Vec<usize> = (0..dx.singular_values.len()).filter(|&j| dx.singular_values[j] > dx.band).collect();
    let inverse = Array1::from_iter(kept.iter().map(|&j| 1.0 / dx.singular_values[j]));
    let pinv = (dx.vt.select(Axis(0), &kept).t().to_owned() * &inverse).dot(&dx.u.select(Axis(1), &kept).t());
    let phase = |r: usize, f: (i64, i64)| {
        let t = f.0 * labels[[r, 0]] as i64 + f.1 * labels[[r, 1]] as i64;
        2.0 * std::f64::consts::PI * (t.rem_euclid(period as i64)) as f64 / period as f64
    };
    let mut pairs: Vec<(String, (i64, i64))> = vec![("constant".to_string(), (0, 0))];
    for k in 1..=(period / 2) as i64 {
        for (name, f) in [("a", (k, 0)), ("b", (0, k)), ("a+b", (k, k)), ("a−b", (k, -k))] {
            pairs.push((format!("{k}({name})"), f));
        }
    }
    let mut out = Vec::new();
    let mut explained = Array2::<f64>::zeros(y.dim());
    for (label, f) in pairs {
        let mut columns = vec![Array1::from_iter((0..rows).map(|r| phase(r, f).cos()))];
        if f != (0, 0) {
            columns.push(Array1::from_iter((0..rows).map(|r| phase(r, f).sin())));
        }
        let basis = Array2::from_shape_fn((rows, columns.len()), |(r, c)| columns[c][r]);
        let q = qr(basis.view(), QrMode::Economic).map_err(|e| format!("{e:?}"))?.q.ok_or("no Q")?;
        let projected = q.dot(&q.t().dot(&y));
        explained += &projected;
        let (u, v) = factors(&pinv.dot(&projected).t().to_owned(), band)?;
        if u.nrows() > 0 {
            out.push((label, u, v));
        }
    }
    // The rest: what the characters above leave of the written values, and W off the reads' span.
    let span = pinv.dot(x);
    let rest = pinv.dot(&(&y - &explained)).t().to_owned() + &(w - &w.dot(&span));
    let (u, v) = factors(&rest, band)?;
    if u.nrows() > 0 {
        out.push(("rest".to_string(), u, v));
    }
    Ok(out)
}

/// A decomposition's report: its bits, its decoded bits, and per block its label, rank, firing and
/// decoded bits.
fn report(name: &str, coded: &Coded<'_>, blocked: &Blocked, bits: &Bits, geometry: &Structured, labels: Option<&[Vec<String>]>) -> Result<Value, String> {
    say(name, bits);
    let (described, error, kl, prices) = decoded(coded, blocked, geometry)?;
    eprintln!("{name}, decoded: {:.1} bits/word (described {described:.1}, error {error:.1}), KL {kl:.6} nats/word", described + error);
    let (per_word, kl_priced, active, rank) = bits.per_row();
    let mut per_block = Vec::new();
    for (k, ranks) in blocked.ranks.iter().enumerate() {
        for (c, r) in ranks.iter().enumerate() {
            let on: f64 = blocked.masks.iter().map(|m| m[k].column(c).sum()).sum::<f64>();
            let rows: f64 = blocked.masks.iter().map(|m| m[k].nrows() as f64).sum();
            if on == 0.0 {
                continue;
            }
            let (u, v) = blocked.factors(k, c);
            let mut columns = 0.0;
            for j in 0..*r {
                columns += coded.describe.bits(k, u.slice(s![j..j + 1, ..]), v.slice(s![j..j + 1, ..]))?;
            }
            per_block.push(json!({
                "site": coded.sites[k].name,
                "label": labels.map(|l| l[k][c].clone()),
                "rank": r,
                "firing": on / rows,
                "decoded_bits": prices[k][c],
                "priced_bits": coded.describe.bits(k, u, v)?,
                "columns_as_rank_one_priced_bits": columns,
            }));
        }
    }
    Ok(json!({
        "point": name,
        "bits_per_word": per_word,
        "described_bits_per_word": bits.described / bits.rows.max(1.0),
        "error_bits_per_word": bits.kl / bits.rows.max(1.0),
        "kl_per_word": kl_priced,
        "decoded_bits_per_word": described + error,
        "decoded_described_bits_per_word": described,
        "decoded_error_bits_per_word": error,
        "decoded_kl_per_word": kl,
        "active_blocks_per_word": active,
        "active_rank_one_equivalents_per_word": rank,
        "blocks_on": per_block,
    }))
}

fn run(dir: &Path, out: &Path, observations: f64) -> Result<(), String> {
    let imported = import(dir)?;
    let program = imported.program;
    let family = imported.contract.family;
    let trace = program.execute(&family, false).map_err(|e| e.to_string())?;
    let target = Target::every_row(trace.values[program.output].clone());
    let chosen = sites(&program);
    let operands: Vec<&Vec<u32>> = family
        .slots
        .iter()
        .filter_map(|v| match v {
            SlotValues::Tokens(t) if t.iter().any(|x| *x != t[0]) => Some(t),
            _ => None,
        })
        .collect();
    let period = operands.iter().flat_map(|t| t.iter()).max().map_or(0, |m| *m as usize + 1);
    if operands.len() != 2 {
        return Err(format!("{} operands; the planes are cut for two", operands.len()));
    }
    let labels = Array2::from_shape_fn((family.rows, 2), |(r, i)| operands[i][r] as usize);
    // Every node a decomposed site's decoded weights can move.
    let mut moved = vec![false; program.nodes.len()];
    for (index, node) in program.nodes.iter().enumerate() {
        moved[index] = chosen.iter().any(|s| s.writes.contains(&index)) || node.arguments().iter().any(|a| moved[*a]);
    }
    let statistics = site_statistics(&program, &chosen, [family.clone()], 16, 0x5EED)?;
    let metrics = logit_gauss_newton(&program, &chosen, &family, &trace, 64)?;
    let (mut geometries, mut lattice_sites, mut plane_libraries, mut plane_ranks, mut plane_labels, mut svd_libraries) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for ((site, measured), metric) in chosen.iter().zip(&statistics).zip(&metrics) {
        let x = read_values(&trace, site)?;
        let readers = if site.reads.iter().any(|n| moved[*n]) {
            Vec::new()
        } else {
            vec![Chart::harmonic("operand characters of the reads", x.view(), labels.view(), period)?]
        };
        let mut writers = Vec::new();
        if let Some(map) = readout_map(&program, site)? {
            let classes = Array2::from_shape_fn((map.nrows(), 1), |(c, _)| c);
            writers.push(Chart::harmonic("class characters of the readout", map.view(), classes.view(), map.nrows())?);
        }
        let metric_of = || Metric { fisher: metric.clone(), ..Metric::of(measured, observations) };
        geometries.push(Geometry::new(metric_of(), writers, readers, false)?);
        lattice_sites.push(Geometry::new(metric_of(), Vec::new(), Vec::new(), false)?);
        let w = matrix(&program, site)?;
        let blocks = planes(&w, &x, &labels, period)?;
        let check = blocks.iter().fold(Array2::<f64>::zeros(w.dim()), |acc, (_, u, v)| acc + u.t().dot(v));
        let error = (&check - &w).iter().fold(0.0_f64, |m, e| m.max(e.abs())) / w.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
        eprintln!(
            "{}: {}×{}, plane blocks {:?}, Σ blocks − W {error:.1e}",
            site.name,
            w.nrows(),
            w.ncols(),
            blocks.iter().map(|(l, u, _)| format!("{l}: rank {}", u.nrows())).collect::<Vec<_>>()
        );
        let stack = |parts: Vec<&Array2<f64>>| -> Result<Array2<f64>, String> { ndarray::concatenate(Axis(0), &parts.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(|e| e.to_string()) };
        plane_libraries.push(Library {
            u: stack(blocks.iter().map(|b| &b.1).collect())?,
            v: stack(blocks.iter().map(|b| &b.2).collect())?,
            mean: Array1::zeros(w.ncols()),
        });
        plane_ranks.push(blocks.iter().map(|b| b.1.nrows()).collect::<Vec<_>>());
        plane_labels.push(blocks.iter().map(|b| b.0.clone()).collect::<Vec<_>>());
        let library = fisher_svd(measured)?;
        svd_libraries.push(Library { v: library.v.t().to_owned(), u: library.u, mean: measured.mean.clone() });
    }
    drop(trace);
    let mut structured = Structured::new(geometries);
    let lattice = Structured::new(lattice_sites);
    let batches = vec![(family.clone(), target)];
    let ones = |ranks: &[Vec<usize>]| vec![ranks.iter().map(|r| Array2::<f64>::ones((family.rows, r.len()))).collect::<Vec<_>>()];
    // Calibrate the description's rounding price on the fitted solution, then keep it for every
    // point, so every point is fitted and selected under one price.
    let mut calibrations = Vec::new();
    let (fitted, fitted_bits) = loop {
        let coded = Coded { model: &program, sites: chosen.clone(), batches: batches.clone(), observations, samples: 16, describe: &structured, boxed: Some(metrics.clone()) };
        let svd_ranks: Vec<Vec<usize>> = svd_libraries.iter().map(|l| vec![1; l.v.nrows()]).collect();
        let (rank_one, _) = selected(&coded, Blocked::rank_one(svd_libraries.clone(), ones(&svd_ranks)))?;
        let (fitted, fitted_bits) = fit_blocks(&coded, rank_one, true)?;
        let (measured, priced, _) = rounding_error(&coded, &fitted)?;
        eprintln!("rounding: measured {measured:.1} bits against {priced:.1} priced");
        calibrations.push(json!({"measured": measured, "priced": priced}));
        let ratio = if priced > 0.0 { measured / priced } else { f64::INFINITY };
        if (0.5..=2.0).contains(&ratio) || measured <= 0.0 || calibrations.len() >= 6 {
            break (fitted, fitted_bits);
        }
        structured = structured.scaled(if ratio.is_finite() { ratio } else { 1e3 });
    };
    let coded = Coded { model: &program, sites: chosen.clone(), batches, observations, samples: 16, describe: &structured, boxed: Some(metrics.clone()) };
    let mut planes_on = Blocked::new(plane_libraries, plane_ranks.clone(), ones(&plane_ranks));
    planes_on.price(&coded)?;
    let (planes_on_bits, _) = measure(&coded, &planes_on)?;
    let (planes_selected, planes_selected_bits) = selected(&coded, planes_on.clone())?;
    let (seeded, seeded_bits) = fit_blocks(&coded, planes_selected.clone(), true)?;
    let points = vec![
        report("planes, all on", &coded, &planes_on, &planes_on_bits, &structured, Some(plane_labels.as_slice()))?,
        report("planes, selected", &coded, &planes_selected, &planes_selected_bits, &structured, Some(plane_labels.as_slice()))?,
        report("fitted from rank-one subcomponents", &coded, &fitted, &fitted_bits, &structured, None)?,
        report("fitted from the selected planes", &coded, &seeded, &seeded_bits, &structured, None)?,
    ];
    // The fitted and the planes under the plain lattice code too (no charts), decoded.
    let mut plain = Vec::new();
    for (name, blocked) in [("planes, selected", &planes_selected), ("fitted from rank-one subcomponents", &fitted)] {
        let (described, error, kl, _) = decoded(&coded, blocked, &lattice)?;
        eprintln!("{name}, lattice generic, decoded: {:.1} bits/word (described {described:.1}, error {error:.1}), KL {kl:.6}", described + error);
        plain.push(json!({"point": name, "decoded_bits_per_word": described + error, "decoded_described_bits_per_word": described, "decoded_error_bits_per_word": error, "decoded_kl_per_word": kl}));
    }
    let report = json!({
        "observations": observations,
        "claim": "box",
        "calibrations": calibrations,
        "points": points,
        "lattice_generic": plain,
        "plane_blocks": chosen.iter().zip(&plane_labels).zip(&plane_ranks).map(|((s, l), r)| json!({"site": s.name, "labels": l, "ranks": r})).collect::<Vec<_>>(),
    });
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("mpd_planes_2951 EXPORT_DIR OUT.json OBSERVATIONS".to_string());
    }
    let observations = args[3].parse::<f64>().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    run(Path::new(&args[1]), Path::new(&args[2]), observations)
}

//! The Fourier mechanism of the mod-31 adder written by hand as blocks, scored by the description
//! length against the fitted solution (#2951), per word and with the library paid once.
//!
//! `mpd_planes_2951 EXPORT_DIR OUT.json OBSERVATIONS`
//!
//! `EXPORT_DIR` a `transformer` export on the whole family of a modular adder (e.g.
//! `~/mpd-data/engine/p31_s0_generic`, all 961 inputs). Each decomposed site's map `W` is cut by
//! the characters its reads carry: with `X` the site's reads and `Y = X Wᵀ` its written values over
//! the family, a character pair `cos`, `sin` of `2π (f · (a, b))/p` (`f` one of `(k, 0)`, `(0, k)`,
//! `(k, k)`, `(k, −k)`) arrives along its carrier `Xᵀ Q_f` (`Q_f` its orthonormal profiles); in
//! decreasing share of the output `‖Q_fᵀ Y‖` (the constant last), each carrier less the directions
//! already taken is the plane's reader `R_f` and the plane is `W R_f R_fᵀ`, of rank at most two;
//! what the readers leave is one more block, the rest, so the blocks sum to `W` and none exceeds it
//! (`planes`). A plane with nothing beyond the site's rounding band is left out.
//!
//! Every point is measured under the corner claim (every token's error the exact KL of the program
//! its masks run: off blocks absent). Every block is described on the exact lattice code with the
//! harmonic charts (`gam_mpd::describe::Structured`: a reader in the operand characters of a site's
//! reads where no decomposed site upstream moves them, a writer in the class characters of the
//! unembedding where the readout reads the site directly), in the logit-space Gauss–Newton metric,
//! the price recalibrated against the exact rounding KL until within a factor of two. Each point is
//! decoded (every block replaced by its exact-priced description, `Geometry::describe_exact`, in
//! decoding order, and measured by the exact masked forward) under two codes:
//!
//! * per word: each word pays the description of the blocks that ran on it, plus `n KL / ln 2`;
//! * library paid once: every block that runs is described once (its precision weighing its bits
//!   once against its error on every word), each block names the words it runs on (a bit for every
//!   word, else the enumerative code of the subset), plus `n KL / ln 2` per word; and the same
//!   library pruned under this code (blocks deleted while the total falls, `pruned`), what it can
//!   afford at its `n`.
//!
//! The points: each site's whole map as one block; the planes all on, and selected (selection
//! passes until one no longer lowers the per-word total); the fitted solution (Fisher-SVD rank-one
//! subcomponents selected, then `fit_blocks`) and the fit seeded from the selected planes, each as
//! fitted and with its blocks all on. Per point the terms per word under both codes, blocks and
//! rank-one equivalents on, and per block its label, rank, firing, decoded bits and whether the
//! pruned library keeps it.
//!
//! Measured on p31 (`n` = 10⁴, 10⁵, 10⁶). With the library paid once: whole sites 56.3, 81.1, 100.5
//! bits a word; the planes all on 114, 141, 213, and pruned under that code 77.9, 120.5, 178.0. The
//! pruned planes are the key frequencies (7, 8, 5 at W_O and W_in; 7, 8, 5 (a+b) first at W_out; 37,
//! 57, 72 of 163 planes kept). Per word: whole sites 53,969, 77,818, 96,484; planes selected 47,716
//! (KL 3.2 nats) and 62,692 at 10⁴ and 10⁵; the fit from rank-one subcomponents 43,975 (KL 3.0) and
//! 29,289; the fit seeded from the planes 10,979 at 10⁴, as input-gated rank-one slices. Under both
//! codes the objective prices the planes above something else. A plane's writer is sent in the
//! site's own coordinates (W_out's 20 kept planes are rank 40 on a 32-wide side; W_K's planes act
//! only through the query). No p31 site has character charts on both sides, so the rotation core
//! never applies.

use gam_mpd::blocks::{Bits, Blocked, Coded, fit_blocks, measure, reselect, rounding_error};
use gam_mpd::codec::subset_code_len_bits;
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

/// A decoded point's terms per word.
struct Decoded {
    /// Description bits: under the per-word code the blocks that ran on each word; with the library
    /// paid once every block that runs, once.
    described: f64,
    /// With the library paid once, the words each block runs on: a bit for "every word", else the
    /// enumerative code of the subset.
    bindings: f64,
    /// `n KL / ln 2`.
    error: f64,
    kl: f64,
    /// Each block's decoded bits and (with the library paid once) bindings.
    prices: Vec<Vec<f64>>,
    binding: Vec<Vec<f64>>,
    /// The decoded blocks.
    blocked: Blocked,
}

impl Decoded {
    fn total(&self) -> f64 {
        self.described + self.bindings + self.error
    }
}

/// The decoded point (module note): every block that runs replaced by its exact-priced description
/// in decoding order, then measured together. Under the per-word code (`once` false) a block's
/// precision weighs its bits on every word it runs on against its error there; with the library
/// paid once, its bits once against its error on every word.
fn decoded(coded: &Coded<'_>, blocked: &Blocked, geometry: &Structured, once: bool) -> Result<Decoded, String> {
    let mut out = blocked.clone();
    let mut prices: Vec<Vec<f64>> = Vec::new();
    let mut binding: Vec<Vec<f64>> = Vec::new();
    let words: usize = blocked.masks.iter().map(|m| m.first().map_or(0, |m| m.nrows())).sum();
    for (k, ranks) in blocked.ranks.iter().enumerate() {
        let (mut site, mut named) = (Vec::new(), Vec::new());
        for c in 0..ranks.len() {
            let on: usize = blocked.masks.iter().map(|m| m[k].column(c).iter().filter(|x| **x > 0.0).count()).sum();
            if on == 0 {
                site.push(0.0);
                named.push(0.0);
                continue;
            }
            named.push(1.0 + if on == words { 0.0 } else { subset_code_len_bits(words, on).map_err(|e| e.to_string())? as f64 });
            let (base, _) = measure(coded, &out)?;
            let (u, v) = blocked.factors(k, c);
            let current = out.clone();
            let d = geometry.sites[k].describe_exact(u, v, &mut |d| {
                let (bits, _) = measure(coded, &replaced(&current, k, c, &d.u, &d.v)?)?;
                Ok(if once { bits.kl - base.kl } else { (bits.kl - base.kl) / on as f64 })
            })?;
            site.push(d.bits());
            out = replaced(&out, k, c, &d.u, &d.v)?;
        }
        prices.push(site);
        binding.push(named);
    }
    let (bits, _) = measure(coded, &out)?;
    let mut described = 0.0;
    for masks in &blocked.masks {
        for (k, m) in masks.iter().enumerate() {
            for (c, price) in prices[k].iter().enumerate() {
                described += if once { 0.0 } else { m.column(c).iter().filter(|x| **x > 0.0).count() as f64 * price };
            }
        }
    }
    let mut bindings = 0.0;
    if once {
        described = prices.iter().flatten().sum();
        bindings = binding.iter().flatten().sum();
    }
    let rows = bits.rows.max(1.0);
    Ok(Decoded { described: described / rows, bindings: bindings / rows, error: bits.kl / rows, kl: bits.kl_nats / rows, prices, binding, blocked: out })
}

/// With the library paid once, the decoded library's blocks deleted while the total falls: deleting
/// a block saves its bits and bindings against the error it adds on every word, measured exactly;
/// the deletions that save alone are tried together, halved while the total does not fall, and the
/// passes repeat until none is kept. What stays is the library this code can afford at its `n`.
/// The pruned point and per block whether it stayed.
fn pruned(coded: &Coded<'_>, d: &Decoded) -> Result<(Decoded, Vec<Vec<bool>>), String> {
    let mut kept: Vec<Vec<bool>> = d.binding.iter().map(|b| b.iter().map(|x| *x > 0.0).collect()).collect();
    let deleted = |blocked: &Blocked, which: &[(usize, usize)]| -> Result<Blocked, String> {
        let mut out = blocked.clone();
        for &(k, c) in which {
            let (u, v) = blocked.factors(k, c);
            out = replaced(&out, k, c, &Array2::zeros((1, u.ncols())), &Array2::zeros((1, v.ncols())))?;
        }
        Ok(out)
    };
    let mut current = d.blocked.clone();
    loop {
        let (base, _) = measure(coded, &current)?;
        let mut savings = Vec::new();
        for (k, site) in kept.iter().enumerate() {
            for (c, on) in site.iter().enumerate() {
                if *on {
                    let (bits, _) = measure(coded, &deleted(&current, &[(k, c)])?)?;
                    let delta = bits.kl - base.kl - d.prices[k][c] - d.binding[k][c];
                    if delta < 0.0 {
                        savings.push((delta, k, c));
                    }
                }
            }
        }
        savings.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut take = savings.len();
        let mut accepted = false;
        while take > 0 {
            let which: Vec<(usize, usize)> = savings[..take].iter().map(|s| (s.1, s.2)).collect();
            let trial = deleted(&current, &which)?;
            let (bits, _) = measure(coded, &trial)?;
            let saved: f64 = which.iter().map(|&(k, c)| d.prices[k][c] + d.binding[k][c]).sum();
            if bits.kl - base.kl < saved {
                current = trial;
                for (k, c) in which {
                    kept[k][c] = false;
                }
                accepted = true;
                break;
            }
            take /= 2;
        }
        if !accepted {
            break;
        }
    }
    let (bits, _) = measure(coded, &current)?;
    let rows = bits.rows.max(1.0);
    let sum = |x: &[Vec<f64>]| -> f64 { x.iter().zip(&kept).flat_map(|(x, k)| x.iter().zip(k).filter(|(_, k)| **k).map(|(x, _)| *x)).sum() };
    let out = Decoded {
        described: sum(&d.prices) / rows,
        bindings: sum(&d.binding) / rows,
        error: bits.kl / rows,
        kl: bits.kl_nats / rows,
        prices: d.prices.clone(),
        binding: d.binding.clone(),
        blocked: current,
    };
    Ok((out, kept))
}

/// The same blocks, every one on for every word.
fn all_on(blocked: &Blocked) -> Blocked {
    let mut out = blocked.clone();
    out.masks.iter_mut().flatten().for_each(|m| m.fill(1.0));
    out
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

/// A map as balanced factors `(u: r × d_out, v: r × d_in)` over its at most `rank` leading singular
/// values beyond `band` (the whole site's rounding band, so a block that is rounding of the site
/// has none).
fn factors(w: &Array2<f64>, band: f64, rank: usize) -> Result<(Array2<f64>, Array2<f64>), String> {
    let d = svd(w.view(), false).map_err(|e| format!("{e:?}"))?;
    let kept: Vec<usize> = (0..d.singular_values.len()).filter(|&j| d.singular_values[j] > d.band.max(band)).take(rank).collect();
    let roots = Array1::from_iter(kept.iter().map(|&j| d.singular_values[j].sqrt()));
    let u = (d.u.select(Axis(1), &kept) * &roots).t().to_owned();
    let v = (d.vt.select(Axis(0), &kept).t().to_owned() * &roots).t().to_owned();
    Ok((u, v))
}

/// The plane blocks of a site (module note): per character pair `±f` of the operands (`f` one of
/// `(k, 0)`, `(0, k)`, `(k, k)`, `(k, −k)`, `k = 1..=p/2`) a block of rank at most two, the constant,
/// and the rest; `(label, u, v)` with the empty ones (nothing beyond the site's rounding band) left
/// out.
///
/// A character's carrier is the reads' content at it, `M_f = Xᵀ Q_f` (`Q_f` the orthonormal
/// profiles `cos`, `sin` of the character over the inputs): the input directions along which the
/// character arrives. Taken in decreasing share of the site's output `‖Q_fᵀ Y‖` (the constant last,
/// its carrier the mean read every other carrier overlaps), each carrier less the directions already
/// taken spans the plane's reader `R_f` (orthonormal, its directions beyond the reads' rounding
/// band), and the plane is `W R_f R_fᵀ`. The readers are orthogonal, so the planes and the rest
/// `W (I − Σ R_f R_fᵀ)` sum to `W` with every block at most `W`'s norm. Least squares
/// (`(X⁺ Π_f Y)ᵀ`, exact on character-invariant reads) is ill-posed where two characters arrive
/// along the same directions: on p31 the residual after attention carries `cos ka` and `cos kb` on
/// one plane, and separating them through the reads' near-null directions gave planes of 10⁸ `‖W‖`
/// that cancelled to `W` and decoded to KL ~10⁸ nats. Here the first of such characters takes the
/// shared plane and the other keeps only what it carries elsewhere.
fn planes(w: &Array2<f64>, x: &Array2<f64>, labels: &Array2<usize>, period: usize) -> Result<Vec<(String, Array2<f64>, Array2<f64>)>, String> {
    let rows = x.nrows();
    let y = x.dot(&w.t());
    let band = svd(w.view(), false).map_err(|e| format!("{e:?}"))?.band;
    let floor = svd(x.view(), false).map_err(|e| format!("{e:?}"))?.band;
    let phase = |r: usize, f: (i64, i64)| {
        let t = f.0 * labels[[r, 0]] as i64 + f.1 * labels[[r, 1]] as i64;
        2.0 * std::f64::consts::PI * (t.rem_euclid(period as i64)) as f64 / period as f64
    };
    let mut pairs: Vec<(String, (i64, i64))> = Vec::new();
    for k in 1..=(period / 2) as i64 {
        for (name, f) in [("a", (k, 0)), ("b", (0, k)), ("a+b", (k, k)), ("a−b", (k, -k))] {
            pairs.push((format!("{k}({name})"), f));
        }
    }
    let mut characters = Vec::new();
    for (label, f) in pairs {
        let basis = Array2::from_shape_fn((rows, 2), |(r, c)| if c == 0 { phase(r, f).cos() } else { phase(r, f).sin() });
        let q = qr(basis.view(), QrMode::Economic).map_err(|e| format!("{e:?}"))?.q.ok_or("no Q")?;
        let share = q.t().dot(&y).iter().map(|e| e * e).sum::<f64>();
        characters.push((label, q, share));
    }
    characters.sort_by(|a, b| b.2.total_cmp(&a.2));
    characters.push(("constant".to_string(), Array2::from_elem((rows, 1), 1.0 / (rows as f64).sqrt()), 0.0));
    let mut taken = Array2::<f64>::zeros((x.ncols(), 0));
    let mut out = Vec::new();
    for (label, q, _) in characters {
        // The carrier less the readers taken, twice (one Gram–Schmidt pass loses orthogonality to
        // rounding as the readers accumulate).
        let mut carrier = x.t().dot(&q);
        for _ in 0..2 {
            carrier = &carrier - &taken.dot(&taken.t().dot(&carrier));
        }
        let d = svd(carrier.view(), false).map_err(|e| format!("{e:?}"))?;
        let kept: Vec<usize> = (0..d.singular_values.len()).filter(|&j| d.singular_values[j] > floor).collect();
        if kept.is_empty() {
            continue;
        }
        let reader = d.u.select(Axis(1), &kept);
        let (u, v) = factors(&w.dot(&reader).dot(&reader.t()), band, kept.len())?;
        taken = ndarray::concatenate(Axis(1), &[taken.view(), reader.view()]).map_err(|e| e.to_string())?;
        if u.nrows() > 0 {
            out.push((label, u, v));
        }
    }
    // The rest: what the readers leave of W (W off the reads' span, and directions no listed
    // character arrives along).
    let rest = out.iter().fold(w.clone(), |acc, (_, u, v)| acc - &u.t().dot(v));
    let (u, v) = factors(&rest, band, usize::MAX)?;
    if u.nrows() > 0 {
        out.push(("rest".to_string(), u, v));
    }
    Ok(out)
}

/// A decomposition's report: its bits, and decoded under the per-word code and with the library
/// paid once (as it stands, and pruned), per block its label, rank, firing, decoded bits and whether
/// the pruned library keeps it.
fn report(name: &str, coded: &Coded<'_>, blocked: &Blocked, bits: &Bits, geometry: &Structured, labels: Option<&[Vec<String>]>) -> Result<Value, String> {
    say(name, bits);
    let word = decoded(coded, blocked, geometry, false)?;
    eprintln!("{name}, decoded per word: {:.1} bits/word (described {:.1}, error {:.1}), KL {:.6} nats/word", word.total(), word.described, word.error, word.kl);
    let once = decoded(coded, blocked, geometry, true)?;
    eprintln!(
        "{name}, library once: {:.1} bits/word (library {:.1}, bindings {:.1}, error {:.1}), KL {:.6} nats/word",
        once.total(),
        once.described,
        once.bindings,
        once.error,
        once.kl
    );
    let (lean, kept) = pruned(coded, &once)?;
    eprintln!(
        "{name}, library once, pruned: {:.1} bits/word (library {:.1}, bindings {:.1}, error {:.1}), KL {:.6} nats/word, {} of {} blocks kept",
        lean.total(),
        lean.described,
        lean.bindings,
        lean.error,
        lean.kl,
        kept.iter().flatten().filter(|k| **k).count(),
        once.binding.iter().flatten().filter(|b| **b > 0.0).count()
    );
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
                "decoded_bits": word.prices[k][c],
                "decoded_bits_library_once": once.prices[k][c],
                "kept_library_once": kept[k][c],
                "priced_bits": coded.describe.bits(k, u, v)?,
                "columns_as_rank_one_priced_bits": columns,
            }));
        }
    }
    let terms = |d: &Decoded| json!({"bits_per_word": d.total(), "described_bits_per_word": d.described, "binding_bits_per_word": d.bindings, "error_bits_per_word": d.error, "kl_per_word": d.kl});
    Ok(json!({
        "point": name,
        "bits_per_word": per_word,
        "described_bits_per_word": bits.described / bits.rows.max(1.0),
        "error_bits_per_word": bits.kl / bits.rows.max(1.0),
        "kl_per_word": kl_priced,
        "decoded_per_word": terms(&word),
        "decoded_library_once": terms(&once),
        "decoded_library_once_pruned": terms(&lean),
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
    let (mut geometries, mut plane_libraries, mut plane_ranks, mut plane_labels, mut svd_libraries) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
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
        geometries.push(Geometry::new(Metric { fisher: metric.clone(), ..Metric::of(measured, observations) }, writers, readers)?);
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
    let batches = vec![(family.clone(), target)];
    let ones = |ranks: &[Vec<usize>]| vec![ranks.iter().map(|r| Array2::<f64>::ones((family.rows, r.len()))).collect::<Vec<_>>()];
    let mut calibrations: Vec<Value> = Vec::new();
    let mut points = Vec::new();
    let write = |points: &[Value], calibrations: &[Value]| -> Result<(), String> {
        let report = json!({
            "observations": observations,
            "claim": "corner",
            "calibrations": calibrations,
            "points": points,
            "plane_blocks": chosen.iter().zip(&plane_labels).zip(&plane_ranks).map(|((s, l), r)| json!({"site": s.name, "labels": l, "ranks": r})).collect::<Vec<_>>(),
        });
        std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    };
    // Every point also with its blocks all on: with the library paid once, a block that runs on
    // every word names no words.
    let on_everywhere = |coded: &Coded<'_>, blocked: &Blocked| -> Result<(Blocked, Bits), String> {
        let mut on = all_on(blocked);
        on.price(coded)?;
        let (bits, _) = measure(coded, &on)?;
        Ok((on, bits))
    };
    // The points that run every block need no selection, and their decoded descriptions price
    // themselves exactly, so they come first: each site's whole map, and the planes all on.
    {
        let coded = Coded { model: &program, sites: chosen.clone(), batches: batches.clone(), observations, samples: 16, describe: &structured };
        let planes = Blocked::new(plane_libraries.clone(), plane_ranks.clone(), ones(&plane_ranks));
        let (whole, whole_bits) = on_everywhere(&coded, &planes.whole())?;
        points.push(report("whole sites", &coded, &whole, &whole_bits, &structured, None)?);
        let (planes_on, planes_on_bits) = on_everywhere(&coded, &planes)?;
        points.push(report("planes, all on", &coded, &planes_on, &planes_on_bits, &structured, Some(plane_labels.as_slice()))?);
        write(&points, &calibrations)?;
    }
    // The price is calibrated on the selected planes (the point in question), then kept for every
    // selected and fitted point. Each point is written as it lands.
    let (planes_selected, planes_selected_bits) = loop {
        let coded = Coded { model: &program, sites: chosen.clone(), batches: batches.clone(), observations, samples: 16, describe: &structured };
        let (planes_selected, planes_selected_bits) = selected(&coded, Blocked::new(plane_libraries.clone(), plane_ranks.clone(), ones(&plane_ranks)))?;
        let (measured, priced, _) = rounding_error(&coded, &planes_selected)?;
        eprintln!("rounding: measured {measured:.1} bits against {priced:.1} priced");
        calibrations.push(json!({"measured": measured, "priced": priced}));
        let ratio = if priced > 0.0 { measured / priced } else { f64::INFINITY };
        if (0.5..=2.0).contains(&ratio) || measured <= 0.0 || calibrations.len() >= 6 {
            break (planes_selected, planes_selected_bits);
        }
        structured = structured.scaled(if ratio.is_finite() { ratio } else { 1e3 });
    };
    let coded = Coded { model: &program, sites: chosen.clone(), batches, observations, samples: 16, describe: &structured };
    points.push(report("planes, selected", &coded, &planes_selected, &planes_selected_bits, &structured, Some(plane_labels.as_slice()))?);
    write(&points, &calibrations)?;
    let svd_ranks: Vec<Vec<usize>> = svd_libraries.iter().map(|l| vec![1; l.v.nrows()]).collect();
    let (rank_one, _) = selected(&coded, Blocked::rank_one(svd_libraries.clone(), ones(&svd_ranks)))?;
    let (fitted, fitted_bits) = fit_blocks(&coded, rank_one, true)?;
    points.push(report("fitted from rank-one subcomponents", &coded, &fitted, &fitted_bits, &structured, None)?);
    let (fitted_on, fitted_on_bits) = on_everywhere(&coded, &fitted)?;
    points.push(report("fitted from rank-one subcomponents, all on", &coded, &fitted_on, &fitted_on_bits, &structured, None)?);
    write(&points, &calibrations)?;
    let (seeded, seeded_bits) = fit_blocks(&coded, planes_selected, true)?;
    points.push(report("fitted from the selected planes", &coded, &seeded, &seeded_bits, &structured, None)?);
    let (seeded_on, seeded_on_bits) = on_everywhere(&coded, &seeded)?;
    points.push(report("fitted from the selected planes, all on", &coded, &seeded_on, &seeded_on_bits, &structured, None)?);
    write(&points, &calibrations)
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

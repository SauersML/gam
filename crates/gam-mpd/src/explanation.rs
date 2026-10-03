//! An explanation of a model, fitted and run end to end (#2951).
//!
//! # The method
//!
//! * **Libraries.** Every replaced site gets a library in blocks from [`super::site_fit`]: started
//!   at the units it reads or its Fisher-whitened singular pieces, fitted to the site's code, then
//!   gated in blocks by merges and splits from both ends and by evidence, the lowest code kept.
//!   Sites are fitted in execution order, each on its inputs under the explanation fitted so far:
//!   every earlier site replaced and running its own selection, every later one native.
//! * **Selection.** A row's blocks on at a site are chosen from that row's read alone by the
//!   site's code ([`super::site_fit::measure_blocks`]) in the site's mean written Fisher on its
//!   training inputs (a forward has no sampled-label gradients of its own), each block priced at
//!   the description bits its fit measured.
//! * **Execution.** The explanation runs autonomously
//!   ([`OperatorProgram::execute_with_gates`]): each site's selection reads what the explanation's
//!   own program computed, never the model's.
//!
//! # Evaluation
//!
//! Any [`Replacement`] (this explanation, or given per-row sets such as VPD's published ones) is
//! scored on passages against the model's own next-token distributions:
//!
//! * [`replaced`]: `KL(model ‖ replacement)` per row with every site replaced;
//! * [`site_switch`]: the site-switch claim, each site replaced or native for a whole passage, its
//!   worst subset per passage: every subset up to [`EXHAUSTIVE`] sites, else the search of
//!   `bench/vpd_2951/vpd_stepA_honesty.py`'s `sites` mode (each site alone, all but one, all,
//!   random subsets, then single-site flips while the passage's mean KL rises), a lower bound;
//! * [`bits_per_word`]: the description bits of the blocks that ran, per row.

use super::blocks::Describe;
use super::describe::{Geometry, Metric, Structured, declared_charts};
use super::masked::{Library, Masked, Site, Target, kl_score_only, matrix, sites};
use super::operator_program::{FamilyInputs, Node, OperatorProgram, Trace};
use super::site_fit::{self, Blocked, Round, Samples};
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// The most sites whose every subset [`site_switch`] tries.
pub const EXHAUSTIVE: usize = 10;

/// What [`fit`] fits with.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// The code's `n`.
    pub observations: f64,
    /// The most rounds of a site's fit ([`site_fit::fit`]).
    pub rounds: usize,
    /// The most rounds of its blocks by evidence ([`site_fit::ard`]).
    pub evidence_rounds: usize,
    /// Sampled-label reverse passes per training batch ([`site_fit::samples`]).
    pub draws: usize,
    pub seed: u64,
}

/// One replaced site: its library in blocks and what its selection reads.
#[derive(Clone, Debug)]
pub struct Fitted {
    /// The site, in the model's nodes.
    pub site: Site,
    /// Its map (`d_out × d_in`).
    pub w: Array2<f64>,
    pub library: Library,
    /// Its blocks' ranks, in column order.
    pub ranks: Vec<usize>,
    /// Each block's description bits: what the selection charges a row that runs it.
    pub bits: Vec<f64>,
    /// The mean written Fisher and the reads' (uncentred) second moment on its training inputs.
    pub fisher: Array2<f64>,
    pub second_moment: Array2<f64>,
    /// Each block's bits by its factors ([`Priced`]).
    prices: HashMap<u64, f64>,
}

/// A block's key in a price table: its shape and the bits of its factors.
fn block_key(u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> u64 {
    let mut hasher = DefaultHasher::new();
    (u.dim(), v.dim()).hash(&mut hasher);
    for x in u.iter().chain(v.iter()) {
        x.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Every block's column runs `(start, end)` from its ranks.
fn runs(ranks: &[usize]) -> Vec<(usize, usize)> {
    ranks
        .iter()
        .scan(0, |at, r| {
            *at += r;
            Some((*at - r, *at))
        })
        .collect()
}

/// A fitted site's price table as a description: each of its blocks costs the bits its fit
/// measured, so the selection run on every forward never searches a description again.
struct Priced<'a>(&'a HashMap<u64, f64>);

impl Describe for Priced<'_> {
    fn bits(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<f64, String> {
        self.0.get(&block_key(u, v)).copied().ok_or_else(|| format!("site {site}: a block the fit did not price"))
    }

    fn decode(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Option<(Array2<f64>, Array2<f64>, f64)>, String> {
        Err(format!("site {site}: a price table holds bits, not the {:?} × {:?} block's description", u.dim(), v.dim()))
    }
}

impl Fitted {
    /// A site's library in blocks of `ranks`, each block priced at `bits`.
    pub fn new(
        site: Site,
        w: Array2<f64>,
        (library, ranks, bits): (Library, Vec<usize>, Vec<f64>),
        fisher: Array2<f64>,
        second_moment: Array2<f64>,
    ) -> Result<Self, String> {
        let (d_out, d_in) = w.dim();
        if library.v.ncols() != d_in || library.u.ncols() != d_out || library.u.nrows() != library.v.nrows() {
            return Err(format!("{}: library {:?}, {:?} for a {d_out}×{d_in} map", site.name, library.v.dim(), library.u.dim()));
        }
        if ranks.contains(&0) || ranks.iter().sum::<usize>() != library.v.nrows() || bits.len() != ranks.len() {
            return Err(format!("{}: {} bits for blocks {ranks:?} of {} columns", site.name, bits.len(), library.v.nrows()));
        }
        if fisher.dim() != (d_out, d_out) || second_moment.dim() != (d_in, d_in) {
            return Err(format!("{}: statistics {:?}, {:?} for a {d_out}×{d_in} map", site.name, fisher.dim(), second_moment.dim()));
        }
        let prices = runs(&ranks)
            .into_iter()
            .zip(&bits)
            .map(|((a, b), bits)| (block_key(library.u.slice(s![a..b, ..]), library.v.slice(s![a..b, ..])), *bits))
            .collect();
        Ok(Self { site, w, library, ranks, bits, fisher, second_moment, prices })
    }

    /// Each row's blocks on (rows × blocks, 0 or 1), chosen from that row's read (`reads`, rows ×
    /// d_in) alone by the site's code at `n = observations` in its mean written Fisher.
    pub fn select(&self, reads: &Array2<f64>, observations: f64) -> Result<Array2<f64>, String> {
        let rows = reads.nrows();
        let samples = Samples {
            reads: reads.mapv(|x| x as f32).as_standard_layout().into_owned(),
            sensitivity: Array1::ones(rows),
            fisher: self.fisher.clone(),
            second_moment: Array2::zeros((0, 0)),
            };
        let (_, sets) = site_fit::measure_blocks(0, &self.w, &samples, &Priced(&self.prices), observations, &self.library, &self.ranks)?;
        let mut on = Array2::<f64>::zeros((rows, self.ranks.len()));
        for (r, set) in sets.iter().enumerate() {
            for c in set {
                on[[r, *c as usize]] = 1.0;
            }
        }
        Ok(on)
    }

    /// Its blocks' description in its training statistics (the charts its interfaces in `program`
    /// declare): what [`fit`] priced it in, for pricing any other library of the site alike.
    pub fn description(&self, program: &OperatorProgram, observations: f64) -> Result<Structured, String> {
        let statistics = super::pieces::Site { w: self.w.clone(), second_moment: self.second_moment.clone(), mean: Array1::zeros(self.w.ncols()), fisher: self.fisher.clone() };
        let (writers, readers) = declared_charts(program, &self.site)?;
        Ok(Structured::new(vec![Geometry::new(Metric::of(&statistics, observations), writers, readers)?]))
    }
}

/// Sites replaced for evaluation: their libraries in blocks and the rule that picks each row's
/// blocks on.
pub trait Replacement: Sync {
    /// The replaced sites, in the model's nodes, in execution order.
    fn sites(&self) -> Vec<Site>;
    /// The model with the sites `members` (indices into [`Replacement::sites`], ascending) replaced.
    fn masked(&self, model: &OperatorProgram, members: &[usize]) -> Result<Masked, String>;
    /// One forward of `masked` (from [`Replacement::masked`] of `members`) on passage `passage`,
    /// `base` its rows: the trace and every member's blocks on (rows × blocks).
    fn run(&self, masked: &Masked, members: &[usize], passage: usize, base: &FamilyInputs) -> Result<(Trace, Vec<Array2<f64>>), String>;
}

/// An explanation: its sites in execution order, run by their own selection at `n = observations`.
#[derive(Clone, Debug)]
pub struct Explanation {
    pub observations: f64,
    pub sites: Vec<Fitted>,
}

impl Explanation {
    /// One autonomous forward of `masked` ([`Replacement::masked`] of `members`) on `base`: each
    /// member's blocks chosen from the read its own program computed.
    pub fn execute(&self, masked: &Masked, members: &[usize], base: &FamilyInputs) -> Result<(Trace, Vec<Array2<f64>>), String> {
        if masked.sites.len() != members.len() {
            return Err(format!("{} masked sites for {} members", masked.sites.len(), members.len()));
        }
        let gates = masked.gates();
        if gates.len() != members.len() {
            return Err("a masked site without its gate".to_string());
        }
        let mut chosen: Vec<Array2<f64>> = (0..members.len()).map(|k| Array2::zeros((base.rows, masked.blocks(k)))).collect();
        let family = masked.family(base, &chosen);
        let trace = masked
            .program
            .execute_with_gates(&family, &gates, |z, values| {
                let k = masked.z.iter().position(|n| *n == z).ok_or("an unknown gated amplitude")?;
                let views: Vec<_> = masked.sites[k].reads.iter().map(|n| values[*n].view()).collect();
                let reads = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
                let on = self.sites[members[k]].select(&reads, self.observations)?;
                let expanded = masked.expand(k, &on);
                chosen[k] = on;
                Ok(expanded)
            })
            .map_err(|e| e.to_string())?;
        Ok((trace, chosen))
    }
}

impl Replacement for Explanation {
    fn sites(&self) -> Vec<Site> {
        self.sites.iter().map(|f| f.site.clone()).collect()
    }

    fn masked(&self, model: &OperatorProgram, members: &[usize]) -> Result<Masked, String> {
        let chosen: Vec<&Fitted> = members.iter().map(|m| &self.sites[*m]).collect();
        Masked::build_blocks(
            model,
            chosen.iter().map(|f| f.site.clone()).collect(),
            chosen.iter().map(|f| f.library.clone()).collect(),
            chosen.iter().map(|f| f.ranks.clone()).collect(),
        )
    }

    fn run(&self, masked: &Masked, members: &[usize], passage: usize, base: &FamilyInputs) -> Result<(Trace, Vec<Array2<f64>>), String> {
        self.execute(masked, members, base).map_err(|e| format!("passage {passage}: {e}"))
    }
}

/// Given per-row sets: per passage, every site's blocks on (rows × blocks), the same whatever the
/// program computes (VPD's published sets, chosen on the model's own states).
pub struct Given {
    pub sites: Vec<Site>,
    pub libraries: Vec<Library>,
    pub ranks: Vec<Vec<usize>>,
    /// Per passage, per site.
    pub masks: Vec<Vec<Array2<f64>>>,
}

impl Replacement for Given {
    fn sites(&self) -> Vec<Site> {
        self.sites.clone()
    }

    fn masked(&self, model: &OperatorProgram, members: &[usize]) -> Result<Masked, String> {
        Masked::build_blocks(
            model,
            members.iter().map(|m| self.sites[*m].clone()).collect(),
            members.iter().map(|m| self.libraries[*m].clone()).collect(),
            members.iter().map(|m| self.ranks[*m].clone()).collect(),
        )
    }

    fn run(&self, masked: &Masked, members: &[usize], passage: usize, base: &FamilyInputs) -> Result<(Trace, Vec<Array2<f64>>), String> {
        let given = self.masks.get(passage).ok_or_else(|| format!("no given sets for passage {passage}"))?;
        let chosen: Vec<Array2<f64>> = members.iter().map(|m| given[*m].clone()).collect();
        if chosen.iter().any(|m| m.nrows() != base.rows) {
            return Err(format!("passage {passage}: given sets do not cover its {} rows", base.rows));
        }
        let trace = masked.program.execute(&masked.family(base, &chosen), false).map_err(|e| e.to_string())?;
        Ok((trace, chosen))
    }
}

/// The sites of `model` in execution order (by their first written node).
pub fn in_execution_order(mut chosen: Vec<Site>) -> Vec<Site> {
    chosen.sort_by_key(|s| s.writes.iter().copied().min().unwrap_or(usize::MAX));
    chosen
}

/// A site's library in blocks on `samples` (`site_fit`): its fit from its units (`reads_units`) or
/// its Fisher-whitened singular pieces, then its blocks by merges and splits from the fine and the
/// coarse start and by evidence, the lowest code kept.
fn site_library(w: &Array2<f64>, samples: &Samples, describe: &dyn Describe, settings: &Settings, reads_units: bool, name: &str) -> Result<(Blocked, Round), String> {
    let (d_out, d_in) = w.dim();
    let observations = settings.observations;
    let exact = if reads_units {
        super::pieces::unit_pieces(w, super::pieces::Units::Read)
    } else {
        let statistics = super::pieces::Site { w: w.clone(), second_moment: samples.second_moment.clone(), mean: Array1::zeros(d_in), fisher: samples.fisher.clone() };
        super::pieces::fisher_svd(&statistics)?
    };
    let start = Library { v: exact.v.t().to_owned(), u: exact.u, mean: Array1::zeros(d_in) };
    let fit_settings = site_fit::Settings { observations, pieces: d_in + d_out, rounds: settings.rounds, seed: settings.seed ^ 0xF17 };
    let library = site_fit::fit(0, w, samples, describe, fit_settings, Some(&start), |round, _| {
        log::info!("{name} round {}: code {:.1} bits per input (description {:.1}, error {:.1}), L0 {:.2}", round.round, round.code, round.description, round.error, round.l0);
    })?;
    let columns = library.v.nrows();
    let fine = site_fit::blocks(0, w, samples, describe, observations, &library, &vec![1; columns])?;
    let coarse = site_fit::blocks(0, w, samples, describe, observations, &library, &[columns])?;
    let evidence = site_fit::ard(0, w, samples, describe, observations, (&library, &vec![1; columns], columns.max(d_out + d_in)), settings.evidence_rounds)?;
    log::info!("{name} blocks: fine start {:.1}, coarse start {:.1}, by evidence {:.1} bits per input", fine.1.code, coarse.1.code, evidence.1.code);
    Ok([fine, coarse, evidence].into_iter().fold(None, |best: Option<(Blocked, Round)>, c| match best {
        Some(b) if b.1.code <= c.1.code => Some(b),
        _ => Some(c),
    }).expect("three candidates"))
}

/// The explanation of `model`'s sites `chosen` fitted on `batches` at `settings` (module note):
/// sites in execution order, a site whose every later read precedes the current group's writes
/// sampled with it, each group on its inputs under the explanation fitted so far. `known` gives a
/// site already fitted (a resumed run), `done` sees every site as it is fitted.
pub fn fit(
    model: &OperatorProgram,
    chosen: Vec<Site>,
    batches: &[FamilyInputs],
    settings: &Settings,
    mut known: impl FnMut(&Site) -> Result<Option<Fitted>, String>,
    mut done: impl FnMut(&Fitted) -> Result<(), String>,
) -> Result<Explanation, String> {
    let ordered = in_execution_order(chosen);
    let mut explanation = Explanation { observations: settings.observations, sites: Vec::new() };
    let mut at = 0;
    while at < ordered.len() {
        // The group: the next sites whose reads all precede the group's first write.
        let mut end = at + 1;
        let mut first_write = ordered[at].writes.iter().copied().min().unwrap_or(usize::MAX);
        while end < ordered.len() && ordered[end].reads.iter().all(|r| *r < first_write) {
            first_write = first_write.min(ordered[end].writes.iter().copied().min().unwrap_or(usize::MAX));
            end += 1;
        }
        let group = &ordered[at..end];
        let mut fitted: Vec<Option<Fitted>> = group.iter().map(&mut known).collect::<Result<_, _>>()?;
        let missing: Vec<usize> = (0..group.len()).filter(|i| fitted[*i].is_none()).collect();
        if !missing.is_empty() {
            let started = std::time::Instant::now();
            // The group's inputs under the explanation so far: its selections fixed as they ran.
            let members: Vec<usize> = (0..explanation.sites.len()).collect();
            let hybrid = if members.is_empty() { None } else { Some(explanation.masked(model, &members)?) };
            let (program, families) = match &hybrid {
                None => (model, batches.to_vec()),
                Some(masked) => {
                    let families = batches
                        .iter()
                        .map(|b| explanation.execute(masked, &members, b).map(|(_, masks)| masked.family(b, &masks)))
                        .collect::<Result<Vec<_>, _>>()?;
                    (&masked.program, families)
                }
            };
            let in_program = sites(program);
            let mapped: Vec<Site> = missing
                .iter()
                .map(|i| in_program.iter().find(|s| s.name == group[*i].name).cloned().ok_or_else(|| format!("{}: not a site of the hybrid program", group[*i].name)))
                .collect::<Result<_, _>>()?;
            let gathered = site_fit::samples(program, &mapped, families, settings.draws, settings.seed ^ (at as u64).wrapping_mul(0x9E37_79B9))?;
            log::info!("samples of {} sites on {} batches, {:.0}s", mapped.len(), batches.len(), started.elapsed().as_secs_f64());
            for ((i, site), sample) in missing.iter().zip(&mapped).zip(gathered) {
                let started = std::time::Instant::now();
                let w = matrix(program, site)?;
                let statistics = super::pieces::Site { w: w.clone(), second_moment: sample.second_moment.clone(), mean: Array1::zeros(w.ncols()), fisher: sample.fisher.clone() };
                let (writers, readers) = declared_charts(program, site)?;
                let describe = Structured::new(vec![Geometry::new(Metric::of(&statistics, settings.observations), writers, readers)?]);
                let reads_units = site.reads.len() == 1 && matches!(program.nodes[site.reads[0]], Node::Pointwise { .. });
                let (blocked, round) = site_library(&w, &sample, &describe, settings, reads_units, &site.name)?;
                let bits = runs(&blocked.ranks)
                    .into_par_iter()
                    .map(|(a, b)| describe.bits(0, blocked.library.u.slice(s![a..b, ..]), blocked.library.v.slice(s![a..b, ..])))
                    .collect::<Result<Vec<f64>, String>>()?;
                let site_fitted = Fitted::new(group[*i].clone(), w, (blocked.library, blocked.ranks, bits), sample.fisher, sample.second_moment)?;
                log::info!(
                    "{}: {} blocks of {} columns, code {:.1} bits per input (description {:.1}, error {:.1}), {:.2} on, {:.0}s",
                    site.name,
                    site_fitted.ranks.len(),
                    site_fitted.library.v.nrows(),
                    round.code,
                    round.description,
                    round.error,
                    round.l0,
                    started.elapsed().as_secs_f64()
                );
                done(&site_fitted)?;
                fitted[*i] = Some(site_fitted);
            }
        }
        explanation.sites.extend(fitted.into_iter().flatten());
        at = end;
    }
    Ok(explanation)
}

/// A passage scored: its rows and the model's own logits on them.
pub struct Passage {
    pub base: FamilyInputs,
    pub target: Target,
}

impl Passage {
    /// The passage `base` with the model's logits on it.
    pub fn new(model: &OperatorProgram, base: FamilyInputs) -> Result<Self, String> {
        let logits = model.execute(&base, false).map_err(|e| e.to_string())?.values.swap_remove(model.output);
        Ok(Self { base, target: Target::every_row(logits) })
    }
}

/// Per passage, `KL(model ‖ replacement)` per row with the sites `members` replaced, and every
/// member's blocks on.
pub fn replaced(model: &OperatorProgram, replacement: &dyn Replacement, members: &[usize], passages: &[Passage]) -> Result<Vec<(Array1<f64>, Vec<Array2<f64>>)>, String> {
    let masked = replacement.masked(model, members)?;
    passages
        .par_iter()
        .enumerate()
        .map(|(p, passage)| {
            let (trace, masks) = replacement.run(&masked, members, p, &passage.base)?;
            Ok((kl_score_only(&passage.target, &trace.values[masked.program.output]), masks))
        })
        .collect()
}

/// The site-switch claim's attack ([`site_switch`]).
#[derive(Clone, Debug, serde::Serialize)]
pub struct SiteSwitch {
    /// Whether every subset was tried (else the search's lower bound).
    pub exhaustive: bool,
    /// Per passage: the mean KL with every site replaced, the worst subset's mean KL and that
    /// subset (site indices).
    pub all_replaced: Vec<f64>,
    pub worst: Vec<f64>,
    pub worst_subset: Vec<Vec<usize>>,
    /// Per site, per passage: the mean KL with that site alone replaced.
    pub alone: Vec<Vec<f64>>,
    /// Per passage, per row: the largest KL any subset tried gave it.
    pub word_worst: Vec<Vec<f64>>,
    /// Subsets evaluated (passage forwards).
    pub forwards: usize,
}

/// The members of subset `bits` of `sites` sites.
fn members_of(bits: u64, sites: usize) -> Vec<usize> {
    (0..sites).filter(|j| bits >> j & 1 == 1).collect()
}

/// The site-switch claim (module note) of `replacement` on `passages`: every subset of its sites
/// when there are at most [`EXHAUSTIVE`], else the search with `random` random subsets per
/// passage (seeded by `seed`).
pub fn site_switch(model: &OperatorProgram, replacement: &dyn Replacement, passages: &[Passage], random: usize, seed: u64) -> Result<SiteSwitch, String> {
    let count = replacement.sites().len();
    if count == 0 || count > 63 {
        return Err(format!("a site-switch claim over {count} sites"));
    }
    let full = (1u64 << count) - 1;
    let rows: Vec<usize> = passages.iter().map(|p| p.base.rows).collect();
    let mut result = SiteSwitch {
        exhaustive: count <= EXHAUSTIVE,
        all_replaced: vec![0.0; passages.len()],
        worst: vec![f64::NEG_INFINITY; passages.len()],
        worst_subset: vec![Vec::new(); passages.len()],
        alone: vec![vec![0.0; passages.len()]; count],
        word_worst: rows.iter().map(|r| vec![0.0; *r]).collect(),
        forwards: 0,
    };
    // One subset on some passages: their per-row KL, folded into the result.
    let visit = |bits: u64, on: &[usize], result: &mut SiteSwitch| -> Result<Vec<f64>, String> {
        let members = members_of(bits, count);
        let masked = replacement.masked(model, &members)?;
        let kls: Vec<Array1<f64>> = on
            .par_iter()
            .map(|&p| {
                let (trace, _) = replacement.run(&masked, &members, p, &passages[p].base)?;
                Ok(kl_score_only(&passages[p].target, &trace.values[masked.program.output]))
            })
            .collect::<Result<_, String>>()?;
        result.forwards += on.len();
        let mut means = Vec::with_capacity(on.len());
        for (&p, kl) in on.iter().zip(kls) {
            let mean = kl.mean().unwrap_or(0.0);
            for (w, k) in result.word_worst[p].iter_mut().zip(kl.iter()) {
                *w = w.max(*k);
            }
            if mean > result.worst[p] {
                result.worst[p] = mean;
                result.worst_subset[p] = members.clone();
            }
            if bits == full {
                result.all_replaced[p] = mean;
            }
            if bits.count_ones() == 1 {
                result.alone[bits.trailing_zeros() as usize][p] = mean;
            }
            means.push(mean);
        }
        Ok(means)
    };
    let every: Vec<usize> = (0..passages.len()).collect();
    if count <= EXHAUSTIVE {
        for bits in 1..=full {
            visit(bits, &every, &mut result)?;
        }
        return Ok(result);
    }
    // Each site alone, all but one, all, then random subsets.
    let mut tried = vec![full];
    for j in 0..count {
        tried.push(1 << j);
        tried.push(full ^ (1 << j));
    }
    let mut state = seed | 1;
    for _ in 0..random {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        tried.push((state & full).max(1));
    }
    for bits in tried {
        visit(bits, &every, &mut result)?;
    }
    // Single-site flips from each passage's worst subset while its mean KL rises.
    for p in 0..passages.len() {
        loop {
            let best = result.worst[p];
            let current: u64 = result.worst_subset[p].iter().map(|j| 1u64 << j).sum();
            for j in 0..count {
                let flipped = current ^ (1 << j);
                if flipped != 0 {
                    visit(flipped, &[p], &mut result)?;
                }
            }
            if result.worst[p] <= best {
                break;
            }
        }
    }
    Ok(result)
}

/// Per row, the description bits of the blocks that ran: `Σ_sites Σ_{blocks on} bits`, with
/// `masks` per site (rows × blocks) and `bits` per site and block.
pub fn bits_per_word(masks: &[Array2<f64>], bits: &[Vec<f64>]) -> Result<Array1<f64>, String> {
    let rows = masks.first().map_or(0, |m| m.nrows());
    let mut total = Array1::<f64>::zeros(rows);
    for (m, b) in masks.iter().zip(bits) {
        if m.ncols() != b.len() || m.nrows() != rows {
            return Err(format!("masks {:?} for {} blocks over {rows} rows", m.dim(), b.len()));
        }
        total += &m.dot(&Array1::from(b.clone()));
    }
    Ok(total)
}

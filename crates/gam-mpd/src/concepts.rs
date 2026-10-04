//! Concepts: named groups of subcomponents, the vocabulary of a text that alone says which weights
//! ran on a word (#2951).
//!
//! # The object
//!
//! A decomposition says, per word `t`, which subcomponents may run: a set `S_t` of a universe of
//! `N`. A *concept* is a group `g` of subcomponents with a name. A word's text is the names of the
//! concepts it invokes, and the text is the only channel: decoding it runs the model with every
//! member of every named concept on and everything else off. Concepts are disjoint, and a
//! subcomponent can be a concept alone. The natural-language autoencoder
//! (`bench/vpd_2951/vpd_nl_autoencoder.py`) writes each concept's name in English from its members'
//! weights and contexts and prices the text under a fixed language model.
//!
//! # The code
//!
//! A word pays the objective's three terms, with nothing sent beside the text:
//!
//! ```text
//! names:     KT(|Z_g|, T) per concept in the vocabulary (invoked on some word)
//! program:   B_g = Σ_{j∈g} b_j on every word that invokes g (the description of the weights that ran)
//! error:     n KL_t / ln 2 of the model running the decoded program
//! library:   L_subset(N, |g|) + label bits, per concept in the vocabulary
//! ```
//!
//! with `T` the coded words, `Z_g` the words invoking `g` (only a word whose set holds a member
//! may invoke it), `b_j` member `j`'s description bits (from the library's own description, so the
//! code is library-agnostic) and `KL_t` the exact KL of the program word `t`'s text decodes to,
//! measured by running the model ([`Oracle::kl`]). `KT(k, n) = −log₂ [Γ(k + ½) Γ(n − k + ½) /
//! (π Γ(n + 1))]`, the Krichevsky–Trofimov code of the invocation stream, pays for its own rate.
//! The program term is what keeps the code honest: without it the one name "everything" (every
//! subcomponent on at every word) would cost almost nothing, since that program is the model
//! itself.
//!
//! # Prices
//!
//! The error is not a sum over concepts, so the fit proposes with prices: `d_tj = n/ln 2 · (KL_t
//! with j off − KL_t with j on)`, subcomponent `j` flipped alone at the current programs, for every
//! member of every set ([`Oracle::prices`]). A concept's error at a word is `m_tg =
//! Σ_{j∈g∩S_t} d_tj`, and under the prices the total is a sum over concepts, so changes to
//! disjoint concepts add. Flips interact (dropping many members each nearly free can cost more
//! than all their prices), so the prices only propose and the exact total decides every change.
//! The prices are measured at the sets themselves to start with, and again at the current programs
//! whenever no proposal is left at stale prices.
//!
//! # The fit
//!
//! Given its members, a concept's invocations are fitted by hard EM on its priced bits: the rate
//! `π` from `Z`, then a word invokes the concept when its name and program cost less than the
//! error it removes, `−log₂ π + B_g < −log₂(1 − π) + m_tg`, until `Z` repeats or the bits stop
//! falling; the least total seen over the starts is kept.
//!
//! Every subcomponent that may run starts as its own concept, invoked wherever a set holds it: the
//! decomposition's own sets are the first programs. A round proposes changes under the prices,
//! each the concepts it replaces, the concepts it becomes and its predicted saving:
//!
//! * a *refit* of one concept's invocations (EM from its current invocations, from every word
//!   that needs it, and from none);
//! * when no refit is proposed, *merges*: a merge can only save on words that need both groups
//!   (elsewhere the merged concept costs at least what the cheaper of the two choices did), at
//!   most one name per such word, `log₂ 2(T + 1)` bits, plus the two libraries it replaces. Each
//!   open group proposes partners in order of shared words while that bound exceeds the merged
//!   concept's library, skipping a partner when the merged concept could not save even paying only
//!   the cheaper of its program and its error on every word that needs a member; it proposes the
//!   first whose fitted merge lowers the priced total and closes when none does. Each merged
//!   concept peels: a member whose own concept would cost less than its share of the group leaves,
//!   the most saving first, while that lowers the priced total.
//!
//! The non-overlapping proposals, most saving first, are tried in the exact code: the most saving
//! `k` (at first all, then twice the last kept count) are kept when the exact total falls, else
//! `k` is halved; a single refused change is set aside (a refit until the prices change, a pair
//! for good). Rounds repeat until nothing is proposed at fresh prices or the deadline passes;
//! every kept round lowers the exact total, so the fit can stop at any round.
//!
//! # Coding new words
//!
//! A [`Model`] freezes each concept's rate at its KT estimate. New words are coded by the same
//! descent with the vocabulary frozen ([`Model::encode`]): every concept a set touches invoked at
//! first, then refits of each concept's invocations at its frozen rate, decided by the exact total
//! of those words. [`Model::decode`] turns the invoked concepts back into the program.
//!
//! # What a concept's name does and does not carry
//!
//! Decoding a text to a program is referential faithfulness: the names pick weights, and the KL of
//! the picked program is measured. It does not make the names' English meaning an explanation
//! (renaming every concept preserves the code); that needs the text's words to predict declared
//! interventions. For candidate groups with costs `c_g`, the best partition `min Σ c_g z_g` s.t.
//! `Σ_{g ∋ j} z_g = 1` is bounded below by any `α` with `Σ_{j ∈ g} α_j ≤ c_g` (`Σ_j α_j`), which
//! gives agglomeration a stopping gap.

use super::codec::subset_code_len_bits;
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;
use std::collections::HashSet;
use std::f64::consts::{LN_2, PI};
use std::time::Instant;

/// Per-word sets over a universe: CSR, each row strictly ascending.
#[derive(Clone, Debug)]
pub struct Sets {
    pub universe: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
}

impl Sets {
    pub fn new(universe: usize, indptr: Vec<usize>, indices: Vec<u32>) -> Result<Self, String> {
        if indptr.first() != Some(&0) || indptr.last() != Some(&indices.len()) || indptr.windows(2).any(|w| w[0] > w[1]) {
            return Err("sets: indptr is not a CSR row pointer over the indices".to_string());
        }
        for w in indptr.windows(2) {
            let row = &indices[w[0]..w[1]];
            if row.windows(2).any(|p| p[0] >= p[1]) || row.last().is_some_and(|j| *j as usize >= universe) {
                return Err(format!("sets: a row is not strictly ascending within a universe of {universe}"));
            }
        }
        Ok(Self { universe, indptr, indices })
    }

    pub fn rows(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn row(&self, t: usize) -> &[u32] {
        &self.indices[self.indptr[t]..self.indptr[t + 1]]
    }

    /// The program bits of each word's own set: what the decomposition's sets cost as programs.
    pub fn program_bits(&self, program: &[f64]) -> Vec<f64> {
        (0..self.rows()).map(|t| self.row(t).iter().map(|j| program[*j as usize]).sum()).collect()
    }

    /// Each element's (word, price) pairs, words ascending, from prices aligned with the indices.
    fn columns(&self, prices: &[f64]) -> Vec<Vec<(u32, f64)>> {
        let mut out = vec![Vec::new(); self.universe];
        for t in 0..self.rows() {
            for k in self.indptr[t]..self.indptr[t + 1] {
                out[self.indices[k] as usize].push((t as u32, prices[k]));
            }
        }
        out
    }
}

/// Runs the model on programs, every listed subcomponent on and every other off.
pub trait Oracle {
    /// Per word, the exact KL in nats of the model running that word's program.
    fn kl(&mut self, programs: &[Vec<u32>]) -> Result<Vec<f64>, String>;
    /// At the programs, per member of every word's set (aligned with `sets.indices`), its price in
    /// nats: the word's KL with that subcomponent off minus with it on, the rest of the programs
    /// kept (module note, "Prices").
    fn prices(&mut self, programs: &[Vec<u32>], sets: &Sets) -> Result<Vec<f64>, String>;
}

/// The Krichevsky–Trofimov code length, in bits, of a binary sequence of `n` symbols with `k` ones.
pub fn kt_bits(k: u64, n: u64) -> f64 {
    assert!(k <= n, "kt_bits: {k} ones in {n} symbols");
    // The empty sequence has probability exactly one. Evaluating the gamma identity here
    // introduces a rounding residual, charging (or crediting) unused branches in the code.
    if n == 0 {
        return 0.0;
    }
    (ln_gamma(n as f64 + 1.0) + PI.ln() - ln_gamma(k as f64 + 0.5) - ln_gamma((n - k) as f64 + 0.5)) / LN_2
}

/// The KT estimate of a rate from `k` ones in `n`.
fn kt_rate(k: u64, n: u64) -> f64 {
    (k as f64 + 0.5) / (n as f64 + 1.0)
}

/// A named group of subcomponents and its fitted invocations.
#[derive(Clone, Debug)]
pub struct Concept {
    /// Members, ascending.
    pub members: Vec<u32>,
    /// The words that need some member, ascending, each with its priced error if not invoked (`m_tg`).
    pub needed: Vec<u32>,
    pub error: Vec<f64>,
    /// Per needed word, whether it invokes the concept.
    pub invoked: Vec<bool>,
    /// `B_g`: the program bits of invoking it.
    pub program: f64,
    /// Its names, program and priced error bits (module note, "Prices").
    pub bits: f64,
}

impl Concept {
    pub fn invocations(&self) -> usize {
        self.invoked.iter().filter(|z| **z).count()
    }

    /// Its library bits: none outside the vocabulary, else its members as a subset of the universe
    /// plus its label.
    fn library(&self, universe: usize, label_bits: f64) -> f64 {
        if self.invocations() == 0 {
            return 0.0;
        }
        subset_code_len_bits(universe, self.members.len()).map_or(f64::INFINITY, |b| b as f64) + label_bits
    }

    fn total(&self, universe: usize, label_bits: f64) -> f64 {
        self.bits + self.library(universe, label_bits)
    }
}

/// The bits of invocations `z` over the needed words (module note, "The code").
fn concept_bits(error: &[f64], z: &[bool], program: f64, words: u64) -> f64 {
    let invoked = z.iter().filter(|x| **x).count() as u64;
    let names = if invoked == 0 { 0.0 } else { kt_bits(invoked, words) };
    names + invoked as f64 * program + error.iter().zip(z).filter(|(_, on)| !**on).map(|(e, _)| e).sum::<f64>()
}

/// The words that need some of `members`, ascending, and their summed prices.
fn needs(members: &[u32], columns: &[Vec<(u32, f64)>]) -> (Vec<u32>, Vec<f64>) {
    let mut pairs: Vec<(u32, f64)> = members.iter().flat_map(|j| columns[*j as usize].iter().copied()).collect();
    pairs.sort_unstable_by_key(|p| p.0);
    let mut needed: Vec<u32> = Vec::new();
    let mut error: Vec<f64> = Vec::new();
    for (t, d) in pairs {
        if needed.last() == Some(&t) {
            *error.last_mut().expect("a needed word") += d;
        } else {
            needed.push(t);
            error.push(d);
        }
    }
    (needed, error)
}

/// Fit the concept of `members` (module note, "The fit"), from each of `inits` (invoking words,
/// ascending); the least-bits result.
fn fit_concept(members: Vec<u32>, columns: &[Vec<(u32, f64)>], program: &[f64], words: u64, inits: &[&[u32]]) -> Concept {
    let (needed, error) = needs(&members, columns);
    let cost: f64 = members.iter().map(|j| program[*j as usize]).sum();
    let mut best: Option<(Vec<bool>, f64)> = None;
    for init in inits {
        let mut z: Vec<bool> = needed.iter().map(|t| init.binary_search(t).is_ok()).collect();
        let mut bits = concept_bits(&error, &z, cost, words);
        loop {
            let pi = kt_rate(z.iter().filter(|x| **x).count() as u64, words);
            let (on, off) = (-pi.log2() + cost, -(1.0 - pi).log2());
            let next: Vec<bool> = error.iter().map(|e| on < off + e).collect();
            if next == z {
                break;
            }
            let b = concept_bits(&error, &next, cost, words);
            if b >= bits {
                break;
            }
            (z, bits) = (next, b);
        }
        if best.as_ref().is_none_or(|(_, b)| bits < *b) {
            best = Some((z, bits));
        }
    }
    let (invoked, bits) = best.expect("at least one start");
    Concept { members, needed, error, invoked, program: cost, bits }
}

/// A concept's errors and bits under new prices, its invocations kept (the sets fix the words
/// that need it).
fn reprice(c: &mut Concept, columns: &[Vec<(u32, f64)>], words: u64) {
    c.error = needs(&c.members, columns).1;
    c.bits = concept_bits(&c.error, &c.invoked, c.program, words);
}

/// A lower bound on what merging `a` and `b` changes the total by: on every word that needs a
/// member the merged concept pays at least the cheaper of its program and its error, and its
/// names and library are not negative.
fn merge_bound(a: &Concept, b: &Concept, universe: usize, label_bits: f64) -> f64 {
    let (mut i, mut j, mut floor) = (0, 0, 0.0);
    let program = a.program + b.program;
    while i < a.needed.len() || j < b.needed.len() {
        let (ta, tb) = (a.needed.get(i).copied().unwrap_or(u32::MAX), b.needed.get(j).copied().unwrap_or(u32::MAX));
        let error = match ta.cmp(&tb) {
            std::cmp::Ordering::Less => {
                i += 1;
                a.error[i - 1]
            }
            std::cmp::Ordering::Greater => {
                j += 1;
                b.error[j - 1]
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
                a.error[i - 1] + b.error[j - 1]
            }
        };
        floor += program.min(error);
    }
    floor - a.total(universe, label_bits) - b.total(universe, label_bits)
}

fn invoking(c: &Concept) -> Vec<u32> {
    c.needed.iter().zip(&c.invoked).filter(|(_, z)| **z).map(|(t, _)| *t).collect()
}

fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = a.iter().chain(b).copied().collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn intersection(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().copied().filter(|x| b.binary_search(x).is_ok()).collect()
}

/// One round's record of the fit.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Round {
    pub concepts: usize,
    pub vocabulary: usize,
    /// Changes proposed (refits, merges), tried in the exact code, and kept.
    pub refits: usize,
    pub merges: usize,
    pub kept: usize,
    pub peels: usize,
    /// Exact evaluations the round spent.
    pub tries: usize,
    /// The exact total after the round, its mean KL per word, and the mean program size.
    pub total_bits: f64,
    pub kl: f64,
    pub size: f64,
}

/// The fitted vocabulary of a set of words.
#[derive(Clone, Debug)]
pub struct Fit {
    pub universe: usize,
    pub words: u64,
    pub label_bits: f64,
    /// Every concept, in the vocabulary or not (a subcomponent alone that no word invokes).
    pub concepts: Vec<Concept>,
    pub rounds: Vec<Round>,
    /// The exact total of the sets as their own programs (where the fit starts), the fitted exact
    /// total, and every fitted word's exact KL.
    pub start_bits: f64,
    pub total_bits: f64,
    pub kl: Vec<f64>,
}

/// Each element's (word, price in bits) pairs from the oracle's prices in nats.
fn priced_columns(sets: &Sets, prices: &[f64], scale: f64) -> Result<Vec<Vec<(u32, f64)>>, String> {
    if prices.len() != sets.indices.len() || prices.iter().any(|x| !x.is_finite()) {
        return Err(format!("{} prices for {} set members", prices.len(), sets.indices.len()));
    }
    Ok(sets.columns(&prices.iter().map(|x| x * scale).collect::<Vec<_>>()))
}

/// Every word's program: the members of every concept it invokes, ascending.
fn programs_of<'a>(concepts: impl Iterator<Item = &'a Concept>, words: u64) -> Vec<Vec<u32>> {
    let mut programs: Vec<Vec<u32>> = vec![Vec::new(); words as usize];
    for c in concepts {
        for (t, on) in c.needed.iter().zip(&c.invoked) {
            if *on {
                programs[*t as usize].extend(&c.members);
            }
        }
    }
    programs.par_iter_mut().for_each(|p| p.sort_unstable());
    programs
}

/// The exact code of a vocabulary (module note, "The code"): names, programs and library from the
/// concepts, the KL from the oracle on every word's decoded program.
fn exact<'a>(concepts: impl Iterator<Item = &'a Concept> + Clone, words: u64, universe: usize, label_bits: f64, observations: f64, oracle: &mut dyn Oracle) -> Result<(f64, Vec<f64>), String> {
    let mut bits = 0.0;
    for c in concepts.clone() {
        let z = c.invocations() as u64;
        if z > 0 {
            bits += kt_bits(z, words) + z as f64 * c.program + c.library(universe, label_bits);
        }
    }
    let kl = oracle.kl(&programs_of(concepts, words))?;
    if kl.len() != words as usize {
        return Err(format!("oracle: {} KLs for {words} words", kl.len()));
    }
    Ok((bits + observations / LN_2 * kl.iter().sum::<f64>(), kl))
}

/// Peel a merged concept (module note, "The fit"): the member whose leaving saves the most, while
/// one does; the concepts it becomes and the bits saved.
fn peel(mut c: Concept, columns: &[Vec<(u32, f64)>], program: &[f64], words: u64, universe: usize, label_bits: f64) -> (Vec<Concept>, f64, usize) {
    let mut out = Vec::new();
    let mut saved = 0.0;
    let mut peeled = 0;
    while c.members.len() > 1 {
        let z = invoking(&c);
        let before = c.total(universe, label_bits);
        let best = c
            .members
            .par_iter()
            .map(|j| {
                let rest: Vec<u32> = c.members.iter().copied().filter(|m| m != j).collect();
                let kept = fit_concept(rest, columns, program, words, &[&z]);
                let own: Vec<u32> = columns[*j as usize].iter().map(|p| p.0).collect();
                let alone = fit_concept(vec![*j], columns, program, words, &[&own, &intersection(&own, &z)]);
                let delta = kept.total(universe, label_bits) + alone.total(universe, label_bits) - before;
                (delta, *j, kept, alone)
            })
            .min_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        match best {
            Some((delta, _, kept, alone)) if delta < 0.0 => {
                saved -= delta;
                peeled += 1;
                out.push(alone);
                c = kept;
            }
            _ => break,
        }
    }
    out.push(c);
    (out, saved, peeled)
}

/// The exact total decides (module note, "The fit"): of `count` proposals sorted most saving first,
/// the most saving `k` from `k = min(count, trust)`, halved while `evaluate(k)`'s exact total does
/// not fall below `total`. The kept count with its exact total and KL, or none when even the most
/// saving single change is refused; and the evaluations spent.
fn halve(count: usize, trust: usize, total: f64, mut evaluate: impl FnMut(usize) -> Result<(f64, Vec<f64>), String>) -> Result<(Option<(usize, f64, Vec<f64>)>, usize), String> {
    let mut k = count.min(trust.max(1));
    let mut tries = 0;
    while k > 0 {
        let (t, kl) = evaluate(k)?;
        tries += 1;
        if t < total {
            return Ok((Some((k, t, kl)), tries));
        }
        k /= 2;
    }
    Ok((None, tries))
}

/// What a proposed change replaces.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Refit(usize),
    Merge(usize, usize),
}

/// What every concept's code is priced in: the subcomponents' program bits, the coded words, the
/// universe and a name's library bits.
struct Pricing<'a> {
    program: &'a [f64],
    words: u64,
    universe: usize,
    label_bits: f64,
}

/// Merge proposals of the open groups (module note, "The fit"), as (group, partner, priced change, merged concept).
fn propose_merges(
    concepts: &[Option<Concept>],
    alive: &[usize],
    open: &mut [bool],
    refused: &HashSet<(usize, usize)>,
    columns: &[Vec<(u32, f64)>],
    pricing: &Pricing<'_>,
) -> Vec<(usize, usize, f64, Concept)> {
    let Pricing { program, words, universe, label_bits } = *pricing;
    let name_bound = (2.0 * (words as f64 + 1.0)).log2();
    let lib = |c: &Concept| c.library(universe, label_bits);
    // Which concepts each word needs.
    let mut start = vec![0usize; words as usize + 1];
    for g in alive {
        for t in &concepts[*g].as_ref().expect("alive").needed {
            start[*t as usize + 1] += 1;
        }
    }
    for t in 0..words as usize {
        start[t + 1] += start[t];
    }
    let mut fill = start.clone();
    let mut needing = vec![0u32; start[words as usize]];
    for g in alive {
        for t in &concepts[*g].as_ref().expect("alive").needed {
            needing[fill[*t as usize]] = *g as u32;
            fill[*t as usize] += 1;
        }
    }
    let opened: Vec<usize> = alive.iter().copied().filter(|g| open[*g]).collect();
    let proposals: Vec<(usize, Option<(usize, f64, Concept)>)> = opened
        .par_iter()
        .map_init(
            || (vec![0u32; concepts.len()], Vec::<usize>::new()),
            |(count, touched), &g| {
                let a = concepts[g].as_ref().expect("alive");
                for t in &a.needed {
                    for h in &needing[start[*t as usize]..start[*t as usize + 1]] {
                        let h = *h as usize;
                        if h != g && !refused.contains(&(g.min(h), g.max(h))) {
                            if count[h] == 0 {
                                touched.push(h);
                            }
                            count[h] += 1;
                        }
                    }
                }
                let mut ranked: Vec<(u32, usize)> = touched.iter().map(|h| (count[*h], *h)).collect();
                for h in touched.drain(..) {
                    count[h] = 0;
                }
                ranked.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
                let za = invoking(a);
                for (shared, h) in ranked {
                    let b = concepts[h].as_ref().expect("alive");
                    let members = union(&a.members, &b.members);
                    let merged_library = subset_code_len_bits(universe, members.len()).map_or(f64::INFINITY, |x| x as f64) + label_bits;
                    if f64::from(shared) * name_bound + lib(a) + lib(b) <= merged_library {
                        break;
                    }
                    if merge_bound(a, b, universe, label_bits) >= 0.0 {
                        continue;
                    }
                    let zb = invoking(b);
                    let merged = fit_concept(members, columns, program, words, &[&union(&za, &zb), &intersection(&za, &zb)]);
                    let delta = merged.total(universe, label_bits) - a.total(universe, label_bits) - b.total(universe, label_bits);
                    if delta < 0.0 {
                        return (g, Some((h, delta, merged)));
                    }
                }
                (g, None)
            },
        )
        .collect();
    let mut out = Vec::new();
    for (g, p) in proposals {
        match p {
            Some((h, delta, merged)) => out.push((g, h, delta, merged)),
            None => open[g] = false,
        }
    }
    out
}

/// Fit the vocabulary of `sets` (module note, "The fit"): `prices` (nats, aligned with the
/// indices) are the members' prices at the sets themselves, `program[j]` is subcomponent `j`'s
/// description bits, every concept's name costs `label_bits` in the library, and `oracle` runs the
/// model on these words for the exact KL and fresh prices at `observations`. No round starts after
/// `deadline`.
pub fn fit(sets: &Sets, prices: &[f64], program: &[f64], label_bits: f64, observations: f64, deadline: Instant, oracle: &mut dyn Oracle) -> Result<Fit, String> {
    let universe = sets.universe;
    let words = sets.rows() as u64;
    let scale = observations / LN_2;
    // Every subcomponent alone, invoked wherever a set holds it.
    let mut needed_of: Vec<Vec<u32>> = vec![Vec::new(); universe];
    for t in 0..sets.rows() {
        for j in sets.row(t) {
            needed_of[*j as usize].push(t as u32);
        }
    }
    let mut concepts: Vec<Option<Concept>> = needed_of
        .into_iter()
        .enumerate()
        .filter(|(_, needed)| !needed.is_empty())
        .map(|(j, needed)| {
            let n = needed.len();
            Some(Concept { members: vec![j as u32], needed, error: vec![0.0; n], invoked: vec![true; n], program: program[j], bits: 0.0 })
        })
        .collect();
    let (mut total, mut kl) = exact(concepts.iter().flatten(), words, universe, label_bits, observations, oracle)?;
    let start_bits = total;
    log::info!("concepts: start {} concepts, the own sets' exact total {total:.0} bits", concepts.len());
    let mut open = vec![true; concepts.len()];
    let mut refused_pairs: HashSet<(usize, usize)> = HashSet::new();
    let mut refused: HashSet<usize> = HashSet::new();
    let mut columns = priced_columns(sets, prices, scale)?;
    concepts.par_iter_mut().flatten().for_each(|c| reprice(c, &columns, words));
    // Whether the prices were measured at the current programs.
    let mut fresh = true;
    let mut trust = usize::MAX;
    let mut rounds = Vec::new();
    while Instant::now() < deadline {
        let alive: Vec<usize> = (0..concepts.len()).filter(|g| concepts[*g].is_some()).collect();
        let vocabulary = alive.iter().filter(|g| concepts[**g].as_ref().expect("alive").invocations() > 0).count();
        let mut changes: Vec<(Kind, Vec<Concept>, f64)> = alive
            .par_iter()
            .filter(|g| !refused.contains(*g))
            .filter_map(|&g| {
                let c = concepts[g].as_ref().expect("alive");
                let r = fit_concept(c.members.clone(), &columns, program, words, &[&invoking(c), &c.needed, &[]]);
                let saving = c.total(universe, label_bits) - r.total(universe, label_bits);
                (saving > 0.0 && r.invoked != c.invoked).then(|| (Kind::Refit(g), vec![r], saving))
            })
            .collect();
        let refits = changes.len();
        let mut peels = 0;
        if changes.is_empty() {
            let mut merges = propose_merges(&concepts, &alive, &mut open, &refused_pairs, &columns, &Pricing { program, words, universe, label_bits });
            merges.sort_by(|x, y| x.2.total_cmp(&y.2).then(x.0.cmp(&y.0)));
            let mut used = vec![false; concepts.len()];
            let matched: Vec<(usize, usize, Concept, f64)> = merges
                .into_iter()
                .filter_map(|(g, h, delta, merged)| {
                    if used[g] || used[h] {
                        return None;
                    }
                    used[g] = true;
                    used[h] = true;
                    Some((g, h, merged, -delta))
                })
                .collect();
            let peeled: Vec<(Vec<Concept>, f64, usize)> =
                matched.par_iter().map(|(_, _, merged, _)| peel(merged.clone(), &columns, program, words, universe, label_bits)).collect();
            for ((g, h, _, saving), (parts, saved, n)) in matched.into_iter().zip(peeled) {
                peels += n;
                changes.push((Kind::Merge(g, h), parts, saving + saved));
            }
        }
        let merges = changes.len() - refits;
        if changes.is_empty() {
            if fresh {
                break;
            }
            columns = priced_columns(sets, &oracle.prices(&programs_of(concepts.iter().flatten(), words), sets)?, scale)?;
            concepts.par_iter_mut().flatten().for_each(|c| reprice(c, &columns, words));
            refused.clear();
            open.iter_mut().for_each(|o| *o = true);
            fresh = true;
            log::info!("concepts: prices measured again at the programs of round {}", rounds.len());
            continue;
        }
        changes.sort_by(|x, y| y.2.total_cmp(&x.2));
        let (decision, tries) = halve(changes.len(), trust, total, |k| {
            let gone: HashSet<usize> = changes[..k]
                .iter()
                .flat_map(|c| match c.0 {
                    Kind::Refit(g) => vec![g],
                    Kind::Merge(g, h) => vec![g, h],
                })
                .collect();
            let candidate = concepts
                .iter()
                .enumerate()
                .filter(|(g, c)| c.is_some() && !gone.contains(g))
                .map(|(_, c)| c.as_ref().expect("alive"))
                .chain(changes[..k].iter().flat_map(|c| c.1.iter()));
            exact(candidate, words, universe, label_bits, observations, oracle)
        })?;
        let kept = match decision {
            Some((k, t, k_l)) => {
                (total, kl) = (t, k_l);
                trust = 2 * k;
                for (kind, parts, _) in changes.drain(..k) {
                    match kind {
                        Kind::Refit(g) => concepts[g] = None,
                        Kind::Merge(g, h) => {
                            concepts[g] = None;
                            concepts[h] = None;
                        }
                    }
                    for part in parts {
                        concepts.push(Some(part));
                        open.push(true);
                    }
                }
                fresh = false;
                k
            }
            None => {
                match changes[0].0 {
                    Kind::Refit(g) => {
                        refused.insert(g);
                    }
                    Kind::Merge(g, h) => {
                        refused_pairs.insert((g.min(h), g.max(h)));
                    }
                }
                0
            }
        };
        let mean_kl = kl.iter().sum::<f64>() / words.max(1) as f64;
        let size = concepts.iter().flatten().map(|c| (c.invocations() * c.members.len()) as f64).sum::<f64>() / words.max(1) as f64;
        rounds.push(Round { concepts: alive.len(), vocabulary, refits, merges, kept, peels, tries, total_bits: total, kl: mean_kl, size });
        log::info!(
            "concepts: round {} concepts {} vocabulary {vocabulary} refits {refits} merges {merges} kept {kept} ({tries} tries) exact total {total:.0} bits, KL {mean_kl:.4}, {size:.1} on per word",
            rounds.len(),
            alive.len()
        );
    }
    let concepts: Vec<Concept> = concepts.into_iter().flatten().collect();
    Ok(Fit { universe, words, label_bits, concepts, rounds, start_bits, total_bits: total, kl })
}

/// A concept as the frozen model holds it.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Frozen {
    pub members: Vec<u32>,
    /// `π`: the rate at which a word invokes it.
    pub invoked: f64,
    /// `B_g`.
    pub program: f64,
}

/// A word's bits under a [`Model`].
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Bits {
    /// The names (the ideal code of the invocations; the text's own bits are measured separately).
    pub names: f64,
    /// The description of the weights the text decodes to.
    pub program: f64,
    /// The exact KL in nats of the decoded program.
    pub kl: f64,
}

/// New words coded by a [`Model`]: per word the concepts it invokes (ascending) and its bits.
#[derive(Clone, Debug)]
pub struct Coded {
    pub invoked: Vec<Vec<u32>>,
    pub bits: Vec<Bits>,
    pub rounds: Vec<Round>,
}

/// The vocabulary of a [`Fit`] with frozen rates, for coding words it was not fitted on.
#[derive(Clone, Debug)]
pub struct Model {
    pub universe: usize,
    /// The vocabulary: concepts some fitted word invokes.
    pub concepts: Vec<Frozen>,
    /// Per subcomponent, its concept in the vocabulary.
    pub concept_of: Vec<Option<u32>>,
    /// Per fitted word, the vocabulary concepts it invokes (ascending).
    pub fitted: Vec<Vec<u32>>,
}

impl Model {
    pub fn new(fit: &Fit) -> Self {
        let mut concept_of = vec![None; fit.universe];
        let mut concepts = Vec::new();
        let mut fitted = vec![Vec::new(); fit.words as usize];
        for c in &fit.concepts {
            let z = c.invocations() as u64;
            if z == 0 {
                continue;
            }
            for j in &c.members {
                concept_of[*j as usize] = Some(concepts.len() as u32);
            }
            for t in invoking(c) {
                fitted[t as usize].push(concepts.len() as u32);
            }
            concepts.push(Frozen { members: c.members.clone(), invoked: kt_rate(z, fit.words), program: c.program });
        }
        Self { universe: fit.universe, concepts, concept_of, fitted }
    }

    /// The bits of invoking (or not) concept `c`: its name, and its program when invoked.
    fn cost(&self, c: usize, on: bool) -> f64 {
        let f = &self.concepts[c];
        if on { -f.invoked.log2() + f.program } else { -(1.0 - f.invoked).log2() }
    }

    /// Code new words (module note, "Coding new words"): `sets` their sets, `prices` (nats,
    /// aligned with the indices) the members' prices at the sets themselves, `oracle` the model on
    /// these words, at `observations`; no round starts after `deadline`.
    pub fn encode(&self, sets: &Sets, prices: &[f64], observations: f64, deadline: Instant, oracle: &mut dyn Oracle) -> Result<Coded, String> {
        let words = sets.rows();
        let scale = observations / LN_2;
        // Per vocabulary concept, the words that need it and whether each invokes it.
        let mut needed: Vec<Vec<u32>> = vec![Vec::new(); self.concepts.len()];
        for t in 0..words {
            let mut touched: Vec<u32> = sets.row(t).iter().filter_map(|j| self.concept_of[*j as usize]).collect();
            touched.sort_unstable();
            touched.dedup();
            for c in touched {
                needed[c as usize].push(t as u32);
            }
        }
        let mut invoked: Vec<Vec<bool>> = needed.iter().map(|n| vec![true; n.len()]).collect();
        let programs = |invoked: &[Vec<bool>]| -> Vec<Vec<u32>> {
            let mut out: Vec<Vec<u32>> = vec![Vec::new(); words];
            for (c, (n, z)) in needed.iter().zip(invoked).enumerate() {
                for (t, on) in n.iter().zip(z) {
                    if *on {
                        out[*t as usize].extend(&self.concepts[c].members);
                    }
                }
            }
            out.par_iter_mut().for_each(|p| p.sort_unstable());
            out
        };
        let names: f64 = (0..self.concepts.len()).map(|c| self.cost(c, false)).sum::<f64>() * words as f64;
        let code = |invoked: &[Vec<bool>]| -> f64 {
            names
                + invoked
                    .iter()
                    .enumerate()
                    .map(|(c, z)| z.iter().filter(|on| **on).count() as f64 * (self.cost(c, true) - self.cost(c, false)))
                    .sum::<f64>()
        };
        let mut kl = oracle.kl(&programs(&invoked))?;
        let mut total = code(&invoked) + scale * kl.iter().sum::<f64>();
        // Each concept's priced error at the words that need it.
        let errors = |p: &[f64]| -> Result<Vec<Vec<f64>>, String> {
            if p.len() != sets.indices.len() || p.iter().any(|x| !x.is_finite()) {
                return Err(format!("{} prices for {} set members", p.len(), sets.indices.len()));
            }
            let mut error: Vec<Vec<f64>> = needed.iter().map(|n| vec![0.0; n.len()]).collect();
            for t in 0..words {
                for k in sets.indptr[t]..sets.indptr[t + 1] {
                    if let Some(c) = self.concept_of[sets.indices[k] as usize] {
                        let i = needed[c as usize].binary_search(&(t as u32)).expect("a needed word");
                        error[c as usize][i] += scale * p[k];
                    }
                }
            }
            Ok(error)
        };
        let mut error = errors(prices)?;
        let mut refused: HashSet<usize> = HashSet::new();
        let mut fresh = true;
        let mut trust = usize::MAX;
        let mut rounds = Vec::new();
        while Instant::now() < deadline {
            // Refits at the frozen rate: per concept the invocations its prices choose, and the saving.
            let mut changes: Vec<(usize, Vec<bool>, f64)> = (0..self.concepts.len())
                .into_par_iter()
                .filter(|c| !refused.contains(c))
                .filter_map(|c| {
                    let (on, off) = (self.cost(c, true), self.cost(c, false));
                    let next: Vec<bool> = error[c].iter().map(|e| on < off + e).collect();
                    let saving: f64 = invoked[c]
                        .iter()
                        .zip(&next)
                        .zip(&error[c])
                        .filter(|((a, b), _)| a != b)
                        .map(|((now, _), e)| if *now { on - off - e } else { off + e - on })
                        .sum();
                    (saving > 0.0).then_some((c, next, saving))
                })
                .collect();
            if changes.is_empty() {
                if fresh {
                    break;
                }
                error = errors(&oracle.prices(&programs(&invoked), sets)?)?;
                refused.clear();
                fresh = true;
                continue;
            }
            changes.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)));
            let (decision, tries) = halve(changes.len(), trust, total, |k| {
                let mut candidate = invoked.clone();
                for (c, next, _) in &changes[..k] {
                    candidate[*c] = next.clone();
                }
                let k_l = oracle.kl(&programs(&candidate))?;
                Ok((code(&candidate) + scale * k_l.iter().sum::<f64>(), k_l))
            })?;
            let kept = match decision {
                Some((k, t, k_l)) => {
                    (total, kl) = (t, k_l);
                    trust = 2 * k;
                    for (c, next, _) in changes.drain(..k) {
                        invoked[c] = next;
                    }
                    fresh = false;
                    k
                }
                None => {
                    refused.insert(changes[0].0);
                    0
                }
            };
            let mean_kl = kl.iter().sum::<f64>() / words.max(1) as f64;
            let size = programs(&invoked).iter().map(Vec::len).sum::<usize>() as f64 / words.max(1) as f64;
            rounds.push(Round { concepts: self.concepts.len(), vocabulary: self.concepts.len(), refits: changes.len() + kept, merges: 0, kept, peels: 0, tries, total_bits: total, kl: mean_kl, size });
            log::info!("concepts: coding round {} refits {} kept {kept} ({tries} tries) exact total {total:.0} bits, KL {mean_kl:.4}, {size:.1} on per word", rounds.len(), changes.len() + kept);
        }
        let mut coded = vec![Vec::new(); words];
        for (c, (n, z)) in needed.iter().zip(&invoked).enumerate() {
            for (t, on) in n.iter().zip(z) {
                if *on {
                    coded[*t as usize].push(c as u32);
                }
            }
        }
        let silent: f64 = (0..self.concepts.len()).map(|c| self.cost(c, false)).sum();
        let bits = coded
            .iter()
            .zip(&kl)
            .map(|(cs, k)| Bits {
                names: silent + cs.iter().map(|c| -self.concepts[*c as usize].invoked.log2() + (1.0 - self.concepts[*c as usize].invoked).log2()).sum::<f64>(),
                program: cs.iter().map(|c| self.concepts[*c as usize].program).sum(),
                kl: *k,
            })
            .collect();
        Ok(Coded { invoked: coded, bits, rounds })
    }

    /// The program a list of invoked concepts decodes to: all their members, ascending.
    pub fn decode(&self, invoked: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = invoked.iter().flat_map(|c| self.concepts[*c as usize].members.iter().copied()).collect();
        out.sort_unstable();
        out
    }
}

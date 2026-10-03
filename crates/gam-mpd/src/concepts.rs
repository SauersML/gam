//! Concepts: named groups of subcomponents, the vocabulary of a text that alone says which weights
//! ran on a word (#2951).
//!
//! # The object
//!
//! A decomposition says, per word `t`, which subcomponents ran: a set `S_t` of a universe of `N`.
//! A *concept* is a group `g` of subcomponents with a name. A word's text is the names of the
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
//! error:     m_tg = Σ_{j∈g∩S_t} d_tj on every word t that needs g's members and does not invoke it
//! library:   L_subset(N, |g|) + label bits, per concept in the vocabulary
//! ```
//!
//! with `T` the coded words, `Z_g` the words invoking `g`, `b_j` member `j`'s description bits
//! (from the library's own description, so the code is library-agnostic) and `d_tj` the price in
//! bits of leaving `j` off at word `t`, `n ΔKL_tj / ln 2` for the exact KL its absence adds there
//! (the caller measures it). `KT(k, n) = −log₂ [Γ(k + ½) Γ(n − k + ½) / (π Γ(n + 1))]`, the
//! Krichevsky–Trofimov code of the invocation stream, pays for its own rate. The error term is the
//! fit's price for KL; every number quoted for a fitted vocabulary is the exact KL of the decoded
//! text. The program term is what keeps the code honest: without it the one name "everything"
//! (every subcomponent on at every word) would cost almost nothing, since that program is the
//! model itself. The total is a sum over concepts, so changes to disjoint concepts add.
//!
//! # The fit
//!
//! Given its members, a concept's invocations are fitted by hard EM on its bits: the rate `π` from
//! `Z`, then a word invokes the concept when its name and program cost less than the error it
//! removes, `−log₂ π + B_g < −log₂(1 − π) + m_tg` (only a word that needs some member may invoke
//! it), until `Z` repeats or the bits stop falling; the least total seen is kept.
//!
//! Every subcomponent that ran starts as its own concept. Groups merge: a merge can only save on
//! words that need both groups (elsewhere the merged concept costs at least what the cheaper of
//! the two choices did), at most one name per such word, `log₂ 2(T + 1)` bits, plus the two
//! libraries it replaces. Each open group proposes partners in order of shared words while that
//! bound exceeds the merged concept's library; it proposes the first whose fitted merge lowers the
//! priced total and closes when none does. A greedy matching of the proposals, most saving first,
//! is formed, and each merged concept peels: a member whose own concept would cost less than its
//! share of the group (it disagrees with the others on when it is needed) leaves, the most saving
//! first, while that lowers the priced total.
//!
//! The prices `d_tj` are single drops; dropping several members at once interacts. So the priced
//! total only proposes, and the exact total decides: every word's text is decoded, the model runs
//! each word's decoded program ([`Oracle`]), and the code is names + program + library +
//! `n Σ_t KL_t / ln 2` with the exact KL. A round's changes are kept only when the exact total
//! falls; a refused set is halved by predicted saving until a single refused change is set aside
//! (its pair is never proposed again). Rounds repeat until no group is open.
//!
//! # Coding new words
//!
//! A [`Model`] freezes each concept's rate at its KT estimate. A new word invokes a concept when
//! that is cheaper ([`Model::encode`]); [`Model::decode`] turns the invoked concepts back into the
//! program. Its bits split into the names, the program and the predicted error ([`Bits`]).
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
use std::f64::consts::{LN_2, PI};

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
}

/// Per-word sets with the price, in bits, of leaving each member off (aligned with the indices).
#[derive(Clone, Debug)]
pub struct Priced {
    pub sets: Sets,
    pub missing: Vec<f64>,
}

impl Priced {
    pub fn new(sets: Sets, missing: Vec<f64>) -> Result<Self, String> {
        if missing.len() != sets.indices.len() || missing.iter().any(|x| !x.is_finite()) {
            return Err("priced sets: one finite price per member of every set".to_string());
        }
        Ok(Self { sets, missing })
    }

    /// Each element's (word, price) pairs, words ascending.
    fn columns(&self) -> Vec<Vec<(u32, f64)>> {
        let mut out = vec![Vec::new(); self.sets.universe];
        for t in 0..self.sets.rows() {
            for k in self.sets.indptr[t]..self.sets.indptr[t + 1] {
                out[self.sets.indices[k] as usize].push((t as u32, self.missing[k]));
            }
        }
        out
    }

    /// The program bits of each word's own set: what the decomposition's sets cost as programs.
    pub fn program_bits(&self, program: &[f64]) -> Vec<f64> {
        (0..self.sets.rows()).map(|t| self.sets.row(t).iter().map(|j| program[*j as usize]).sum()).collect()
    }
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
    /// The words that need some member, ascending, each with its error if not invoked (`m_tg`).
    pub needed: Vec<u32>,
    pub error: Vec<f64>,
    /// Per needed word, whether it invokes the concept.
    pub invoked: Vec<bool>,
    /// `B_g`: the program bits of invoking it.
    pub program: f64,
    /// Its names, program and error bits (module note, "The code").
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

/// Fit the concept of `members` (module note, "The fit"), from each of `inits` (invoking words,
/// ascending); the least-bits result.
fn fit_concept(members: Vec<u32>, columns: &[Vec<(u32, f64)>], program: &[f64], words: u64, inits: &[&[u32]]) -> Concept {
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
    pub open: usize,
    /// Changes proposed, tried in the exact code, and kept.
    pub proposed: usize,
    pub kept: usize,
    pub peels: usize,
    /// The exact total after the round, and its mean KL per word.
    pub total_bits: f64,
    pub kl: f64,
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
    /// The exact total (module note, "The fit") and every fitted word's exact KL.
    pub total_bits: f64,
    pub kl: Vec<f64>,
}

/// Runs the model on programs: per word, the exact KL in nats of the model restricted to that
/// word's program (every listed subcomponent on, every other off).
pub type Oracle<'a> = dyn FnMut(&[Vec<u32>]) -> Result<Vec<f64>, String> + 'a;

/// The exact code of a vocabulary (module note, "The fit"): names, programs and library from the
/// concepts, the KL from the oracle on every word's decoded program.
fn exact<'a>(concepts: impl Iterator<Item = &'a Concept>, words: u64, universe: usize, label_bits: f64, observations: f64, oracle: &mut Oracle<'_>) -> Result<(f64, Vec<f64>), String> {
    let mut programs: Vec<Vec<u32>> = vec![Vec::new(); words as usize];
    let mut bits = 0.0;
    for c in concepts {
        let z = c.invocations() as u64;
        if z > 0 {
            bits += kt_bits(z, words) + z as f64 * c.program + c.library(universe, label_bits);
        }
        for (t, on) in c.needed.iter().zip(&c.invoked) {
            if *on {
                programs[*t as usize].extend(&c.members);
            }
        }
    }
    for p in &mut programs {
        p.sort_unstable();
    }
    let kl = oracle(&programs)?;
    if kl.len() != programs.len() {
        return Err(format!("oracle: {} KLs for {} words", kl.len(), programs.len()));
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

/// Fit the vocabulary of `priced` (module note, "The fit"): `program[j]` is subcomponent `j`'s
/// description bits, every concept's name costs `label_bits` in the library, and `oracle` runs the
/// model for the exact KL at `observations`.
pub fn fit(priced: &Priced, program: &[f64], label_bits: f64, observations: f64, oracle: &mut Oracle<'_>) -> Result<Fit, String> {
    let universe = priced.sets.universe;
    let words = priced.sets.rows() as u64;
    let columns = priced.columns();
    let name_bound = (2.0 * (words as f64 + 1.0)).log2();
    let mut concepts: Vec<Option<Concept>> = (0..universe)
        .filter(|j| !columns[*j].is_empty())
        .map(|j| {
            let own: Vec<u32> = columns[j].iter().map(|p| p.0).collect();
            Some(fit_concept(vec![j as u32], &columns, program, words, &[&own, &[]]))
        })
        .collect();
    let mut open = vec![true; concepts.len()];
    let mut refused: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let lib = |c: &Concept| c.library(universe, label_bits);
    let (mut total, mut kl) = exact(concepts.iter().flatten(), words, universe, label_bits, observations, oracle)?;
    log::info!("concepts: start {} concepts, exact total {total:.0} bits", concepts.len());
    let mut rounds = Vec::new();
    loop {
        let alive: Vec<usize> = (0..concepts.len()).filter(|g| concepts[*g].is_some()).collect();
        let vocabulary = alive.iter().filter(|g| concepts[**g].as_ref().expect("alive").invocations() > 0).count();
        let opened: Vec<usize> = alive.iter().copied().filter(|g| open[*g]).collect();
        if opened.is_empty() {
            break;
        }
        // Which concepts each word needs.
        let mut start = vec![0usize; words as usize + 1];
        for g in &alive {
            for t in &concepts[*g].as_ref().expect("alive").needed {
                start[*t as usize + 1] += 1;
            }
        }
        for t in 0..words as usize {
            start[t + 1] += start[t];
        }
        let mut fill = start.clone();
        let mut needing = vec![0u32; start[words as usize]];
        for g in &alive {
            for t in &concepts[*g].as_ref().expect("alive").needed {
                needing[fill[*t as usize]] = *g as u32;
                fill[*t as usize] += 1;
            }
        }
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
                        let zb = invoking(b);
                        let merged = fit_concept(members, &columns, program, words, &[&union(&za, &zb), &intersection(&za, &zb)]);
                        let delta = merged.total(universe, label_bits) - a.total(universe, label_bits) - b.total(universe, label_bits);
                        if delta < 0.0 {
                            return (g, Some((h, delta, merged)));
                        }
                    }
                    (g, None)
                },
            )
            .collect();
        let mut accepted: Vec<(usize, usize, f64, Concept)> = Vec::new();
        for (g, p) in proposals {
            match p {
                Some((h, delta, merged)) => accepted.push((g, h, delta, merged)),
                None => open[g] = false,
            }
        }
        accepted.sort_by(|x, y| x.2.total_cmp(&y.2).then(x.0.cmp(&y.0)));
        let mut used = vec![false; concepts.len()];
        // Each change: the pair it replaces, the concepts it becomes, its predicted saving.
        let mut changes: Vec<(usize, usize, Vec<Concept>, f64)> = Vec::new();
        for (g, h, delta, merged) in accepted {
            if used[g] || used[h] {
                continue;
            }
            used[g] = true;
            used[h] = true;
            changes.push((g, h, vec![merged], -delta));
        }
        let peeled: Vec<(Vec<Concept>, f64, usize)> =
            changes.par_iter().map(|(_, _, parts, _)| peel(parts[0].clone(), &columns, program, words, universe, label_bits)).collect();
        let mut peels = 0;
        for (change, (parts, saved, n)) in changes.iter_mut().zip(peeled) {
            change.2 = parts;
            change.3 += saved;
            peels += n;
        }
        changes.sort_by(|x, y| y.3.total_cmp(&x.3).then(x.0.cmp(&y.0)));
        let proposed = changes.len();
        // The exact total decides: the most saving `k` changes, halved while refused.
        let mut k = changes.len();
        let mut kept = 0;
        while k > 0 {
            let gone: std::collections::HashSet<usize> = changes[..k].iter().flat_map(|c| [c.0, c.1]).collect();
            let candidate = concepts
                .iter()
                .enumerate()
                .filter(|(g, c)| c.is_some() && !gone.contains(g))
                .map(|(_, c)| c.as_ref().expect("alive"))
                .chain(changes[..k].iter().flat_map(|c| c.2.iter()));
            let (t, k_l) = exact(candidate, words, universe, label_bits, observations, oracle)?;
            if t < total {
                (total, kl) = (t, k_l);
                kept = k;
                break;
            }
            if k == 1 {
                refused.insert((changes[0].0.min(changes[0].1), changes[0].0.max(changes[0].1)));
            }
            k /= 2;
        }
        for (g, h, parts, _) in changes.drain(..kept) {
            concepts[g] = None;
            concepts[h] = None;
            for part in parts {
                concepts.push(Some(part));
                open.push(true);
            }
        }
        let mean_kl = kl.iter().sum::<f64>() / words.max(1) as f64;
        rounds.push(Round { concepts: alive.len(), vocabulary, open: opened.len(), proposed, kept, peels, total_bits: total, kl: mean_kl });
        log::info!(
            "concepts: round {} concepts {} vocabulary {} open {} proposed {proposed} kept {kept} peels {peels} exact total {total:.0} bits, KL {mean_kl:.4}",
            rounds.len(),
            alive.len(),
            vocabulary,
            opened.len()
        );
    }
    let concepts: Vec<Concept> = concepts.into_iter().flatten().collect();
    Ok(Fit { universe, words, label_bits, concepts, rounds, total_bits: total, kl })
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
    /// The predicted error of what the text leaves off (`Σ m_tg` over needed, uninvoked concepts).
    pub error: f64,
}

/// The vocabulary of a [`Fit`] with frozen rates, for coding words it was not fitted on.
#[derive(Clone, Debug)]
pub struct Model {
    pub universe: usize,
    /// The vocabulary: concepts some fitted word invokes.
    pub concepts: Vec<Frozen>,
    /// Per subcomponent, its concept in the vocabulary.
    pub concept_of: Vec<Option<u32>>,
    /// The names' bits of a word that invokes nothing.
    silent: f64,
}

impl Model {
    pub fn new(fit: &Fit) -> Self {
        let mut concept_of = vec![None; fit.universe];
        let mut concepts = Vec::new();
        for c in &fit.concepts {
            let z = c.invocations() as u64;
            if z == 0 {
                continue;
            }
            for j in &c.members {
                concept_of[*j as usize] = Some(concepts.len() as u32);
            }
            concepts.push(Frozen { members: c.members.clone(), invoked: kt_rate(z, fit.words), program: c.program });
        }
        let silent = concepts.iter().map(|c| -(1.0 - c.invoked).log2()).sum();
        Self { universe: fit.universe, concepts, concept_of, silent }
    }

    /// The concepts a word invokes (ascending) and its bits (module note, "Coding new words"):
    /// `row` its set and `missing` each member's price of being left off.
    pub fn encode(&self, row: &[u32], missing: &[f64]) -> (Vec<u32>, Bits) {
        let mut error: Vec<(u32, f64)> = Vec::new();
        let mut lost = 0.0;
        for (j, d) in row.iter().zip(missing) {
            match self.concept_of[*j as usize] {
                Some(c) => error.push((c, *d)),
                None => lost += d,
            }
        }
        error.sort_unstable_by_key(|e| e.0);
        let mut bits = Bits { names: self.silent, program: 0.0, error: lost };
        let mut invoked = Vec::new();
        let mut i = 0;
        while i < error.len() {
            let c = error[i].0;
            let mut m = 0.0;
            while i < error.len() && error[i].0 == c {
                m += error[i].1;
                i += 1;
            }
            let f = &self.concepts[c as usize];
            let (on, off) = (-f.invoked.log2(), -(1.0 - f.invoked).log2());
            if on + f.program < off + m {
                invoked.push(c);
                bits.names += on - off;
                bits.program += f.program;
            } else {
                bits.error += m;
            }
        }
        (invoked, bits)
    }

    /// The program a list of invoked concepts decodes to: all their members, ascending.
    pub fn decode(&self, invoked: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = invoked.iter().flat_map(|c| self.concepts[*c as usize].members.iter().copied()).collect();
        out.sort_unstable();
        out
    }
}

//! Concepts: groups of subcomponents that run together, each named once per word (#2951).
//!
//! # The object
//!
//! A decomposition says, per word `t`, which subcomponents ran: a set `S_t` of a universe of `N`.
//! A *concept* is a group `g` of subcomponents with one binary latent `z_tg` per word: the word
//! invokes the concept or not, and each member's state is coded given that choice. Concepts are
//! disjoint; a subcomponent in no concept is coded on its own. The concepts are the vocabulary of a
//! description of a word's computation: the word's text names the concepts it invokes (the
//! natural-language autoencoder `bench/vpd_2951/vpd_nl_autoencoder.py` renders each one as a label
//! read off its members' weights), and decoding the text turns on each named concept's core, the
//! members more often on than off when it is invoked ([`Model::decode`]).
//!
//! # The code
//!
//! Every stream is a Krichevsky–Trofimov sequential code, so a rate costs no separate parameter
//! bits (the KT mixture already pays about `½ log₂ n` for it):
//!
//! ```text
//! concept g:    KT(|Z_g|, T) + Σ_{j∈g} [KT(k¹_j, |Z_g|) + KT(k⁰_j, T − |Z_g|)]
//! on its own:   KT(n_j, T)
//! library:      L_subset(N, |g|) + label bits, per concept
//! ```
//!
//! with `T` the coded words, `Z_g` the words invoking `g`, `k¹_j` the words of `Z_g` on which
//! member `j` ran, `k⁰_j` the other words on which it ran, `n_j` all of them, and
//! `KT(k, n) = −log₂ [Γ(k + ½) Γ(n − k + ½) / (π Γ(n + 1))]`. The label bits are what the
//! concept's name costs in the language the text is scored in, declared by the caller. The total
//! is a sum over groups, so merging disjoint pairs changes it by the sum of their own changes.
//!
//! # The fit
//!
//! Given its members, a concept's invocations are fitted by hard EM on its exact bits: the rates
//! from `Z`, then each word invokes the concept when its log-odds under those rates is positive
//! (only a word on which some member ran may invoke it), until `Z` repeats or the bits stop
//! falling; the least total seen is kept. Groups merge agglomeratively. Each open group ranks its
//! partners by the information between their invocations, `T Î(z_g; z_h)` bits (what a joint
//! code of the two indicators saves over separate ones), and proposes, in that order, the first
//! whose fitted merge lowers the total, while that information can still pay for the merge's
//! added library bits; a group with no such partner closes (a merged group is new and open). A
//! greedy matching of the proposals, most saving first, is applied at once, and rounds repeat
//! until no group is open. There is no threshold: a merge is kept exactly when the code is
//! shorter.
//!
//! # Coding new words
//!
//! A [`Model`] freezes every rate at its KT estimate from the fitted counts. A new word invokes a
//! concept when that is cheaper ([`Model::encode`]). Its bits split into the concept choices (what
//! the text carries), the members given those choices and the subcomponents on their own
//! ([`Bits`]), against the independent code that sends every subcomponent at its own rate.
//!
//! # What a concept's name does and does not carry
//!
//! A concept's invocation alone does not determine its members: two words with member patterns
//! `(1, 0)` and `(0, 1)` can invoke the same concept, and the conditional member streams carry the
//! difference. A text that names concepts is exact only together with those corrections. Decoding
//! a text to the native on-sets is referential faithfulness; it does not make the words' English
//! meaning an explanation (renaming every concept preserves the code). That needs the text's rules
//! to predict declared interventions. For candidate groups with costs `c_g`, the best partition
//! `min Σ c_g z_g` s.t. `Σ_{g ∋ j} z_g = 1` is bounded below by any `α` with
//! `Σ_{j ∈ g} α_j ≤ c_g` (`Σ_j α_j`), which gives agglomeration a stopping gap.

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

    /// Each element's rows, ascending.
    pub fn columns(&self) -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); self.universe];
        for t in 0..self.rows() {
            for j in self.row(t) {
                out[*j as usize].push(t as u32);
            }
        }
        out
    }
}

/// The Krichevsky–Trofimov code length, in bits, of a binary sequence of `n` symbols with `k` ones.
pub fn kt_bits(k: u64, n: u64) -> f64 {
    assert!(k <= n, "kt_bits: {k} ones in {n} symbols");
    // The empty sequence has probability exactly one. Evaluating the gamma identity here
    // introduces a rounding residual, charging (or crediting) unused branches in the code.
    if n == 0 { return 0.0; }
    (ln_gamma(n as f64 + 1.0) + PI.ln() - ln_gamma(k as f64 + 0.5) - ln_gamma((n - k) as f64 + 0.5)) / LN_2
}

/// The KT estimate of a rate from `k` ones in `n`.
fn kt_rate(k: u64, n: u64) -> f64 {
    (k as f64 + 0.5) / (n as f64 + 1.0)
}

/// A group of subcomponents and its fitted invocations.
#[derive(Clone, Debug)]
pub struct Concept {
    /// Members, ascending.
    pub members: Vec<u32>,
    /// The words that invoke it, ascending (for a lone subcomponent, the words it ran on).
    pub invoked: Vec<u32>,
    /// Per member, the words it ran on (`n_j`) and those of them that invoke the group (`k¹_j`).
    pub ran: Vec<u64>,
    pub ran_invoked: Vec<u64>,
    /// Its data bits (module note, "The code").
    pub bits: f64,
}

impl Concept {
    fn alone(j: u32, column: &[u32], words: u64) -> Self {
        let n = column.len() as u64;
        Self { members: vec![j], invoked: column.to_vec(), ran: vec![n], ran_invoked: vec![n], bits: kt_bits(n, words) }
    }

    /// Its library bits: none alone, else its members as a subset of the universe plus its label.
    fn library(&self, universe: usize, label_bits: f64) -> f64 {
        if self.members.len() < 2 {
            return 0.0;
        }
        subset_code_len_bits(universe, self.members.len()).map_or(f64::INFINITY, |b| b as f64) + label_bits
    }
}

/// The words on which any of a group's members ran, ascending, with the members (by position)
/// that ran on each.
struct Local {
    words: Vec<u32>,
    start: Vec<usize>,
    member: Vec<u32>,
    ran: Vec<u64>,
}

impl Local {
    fn new(members: &[u32], columns: &[Vec<u32>]) -> Self {
        let mut pairs: Vec<(u32, u32)> = members
            .iter()
            .enumerate()
            .flat_map(|(i, j)| columns[*j as usize].iter().map(move |t| (*t, i as u32)))
            .collect();
        pairs.sort_unstable();
        let mut words = Vec::new();
        let mut start = Vec::new();
        let mut member = Vec::with_capacity(pairs.len());
        for (t, i) in pairs {
            if words.last() != Some(&t) {
                words.push(t);
                start.push(member.len());
            }
            member.push(i);
        }
        start.push(member.len());
        let ran = members.iter().map(|j| columns[*j as usize].len() as u64).collect();
        Self { words, start, member, ran }
    }

    /// The bits of invocations `z` (one per local word) and their member counts `k¹`.
    fn bits(&self, z: &[bool], words: u64) -> (f64, u64, Vec<u64>) {
        let mut invoked = 0u64;
        let mut k1 = vec![0u64; self.ran.len()];
        for (w, on) in z.iter().enumerate() {
            if *on {
                invoked += 1;
                for i in &self.member[self.start[w]..self.start[w + 1]] {
                    k1[*i as usize] += 1;
                }
            }
        }
        let mut bits = kt_bits(invoked, words);
        for (n, k) in self.ran.iter().zip(&k1) {
            bits += kt_bits(*k, invoked) + kt_bits(n - k, words - invoked);
        }
        (bits, invoked, k1)
    }

    /// Hard EM on the exact bits from `z` (module note, "The fit"): the least-bits invocations seen.
    fn fit(&self, mut z: Vec<bool>, words: u64) -> (Vec<bool>, f64, Vec<u64>) {
        let (mut bits, mut invoked, mut k1) = self.bits(&z, words);
        loop {
            let pi = kt_rate(invoked, words);
            let mut base = (pi / (1.0 - pi)).ln();
            let weight: Vec<f64> = self
                .ran
                .iter()
                .zip(&k1)
                .map(|(n, k)| {
                    let q = kt_rate(*k, invoked);
                    let a = kt_rate(n - k, words - invoked);
                    base += ((1.0 - q) / (1.0 - a)).ln();
                    (q / a).ln() - ((1.0 - q) / (1.0 - a)).ln()
                })
                .collect();
            let next: Vec<bool> = (0..self.words.len())
                .map(|w| base + self.member[self.start[w]..self.start[w + 1]].iter().map(|i| weight[*i as usize]).sum::<f64>() > 0.0)
                .collect();
            if next == z {
                return (z, bits, k1);
            }
            let (b, n, k) = self.bits(&next, words);
            if b >= bits {
                return (z, bits, k1);
            }
            (z, bits, invoked, k1) = (next, b, n, k);
        }
    }
}

/// Fit the group of `members` from initial invocations (the local words in `init`, ascending).
fn fit_group(members: Vec<u32>, columns: &[Vec<u32>], words: u64, inits: &[&[u32]]) -> Concept {
    let local = Local::new(&members, columns);
    let mut best: Option<(Vec<bool>, f64, Vec<u64>)> = None;
    for init in inits {
        let mut z = vec![false; local.words.len()];
        let mut i = 0;
        for (w, t) in local.words.iter().enumerate() {
            while i < init.len() && init[i] < *t {
                i += 1;
            }
            z[w] = i < init.len() && init[i] == *t;
        }
        let fitted = local.fit(z, words);
        if best.as_ref().is_none_or(|b| fitted.1 < b.1) {
            best = Some(fitted);
        }
    }
    let (z, bits, ran_invoked) = best.expect("at least one start");
    let invoked = local.words.iter().zip(&z).filter(|(_, on)| **on).map(|(t, _)| *t).collect();
    Concept { members, invoked, ran: local.ran, ran_invoked, bits }
}

fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i] < b[j]) {
            out.push(a[i]);
            i += 1;
        } else if i == a.len() || b[j] < a[i] {
            out.push(b[j]);
            j += 1;
        } else {
            out.push(a[i]);
            i += 1;
            j += 1;
        }
    }
    out
}

fn intersection(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// `T Î(z_a; z_b)` in bits from the two invocation counts and their overlap, when the association
/// is positive (zero otherwise).
fn information(a: u64, b: u64, both: u64, words: u64) -> f64 {
    let n11 = both as f64;
    let n10 = (a - both) as f64;
    let n01 = (b - both) as f64;
    let n00 = (words + both - a - b) as f64;
    if n11 * n00 <= n10 * n01 {
        return 0.0;
    }
    let t = words as f64;
    let term = |n: f64, x: f64, y: f64| if n > 0.0 { n * (n * t / (x * y)).log2() } else { 0.0 };
    let (ra, rb) = (a as f64, b as f64);
    term(n11, ra, rb) + term(n10, ra, t - rb) + term(n01, t - ra, rb) + term(n00, t - ra, t - rb)
}

/// The module note's total over `groups`: their bits, their library, and the subcomponents of the
/// universe in none of them (never ran).
fn total_bits<'a>(groups: impl Iterator<Item = &'a Concept>, universe: usize, words: u64, label_bits: f64) -> f64 {
    let mut ran = 0;
    let mut total = 0.0;
    for g in groups {
        ran += g.members.len();
        total += g.bits + g.library(universe, label_bits);
    }
    total + (universe - ran) as f64 * kt_bits(0, words)
}

/// One round's record of the fit.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Round {
    pub groups: usize,
    pub concepts: usize,
    pub open: usize,
    pub merges: usize,
    pub total_bits: f64,
}

/// The fitted concepts of a set of words.
#[derive(Clone, Debug)]
pub struct Fit {
    pub universe: usize,
    pub words: u64,
    pub label_bits: f64,
    /// Every group: concepts (two or more members) and lone subcomponents that ran.
    pub groups: Vec<Concept>,
    pub rounds: Vec<Round>,
}

impl Fit {
    /// The total of the module note's code: every group's bits, the library, and the subcomponents
    /// that never ran.
    pub fn total_bits(&self) -> f64 {
        total_bits(self.groups.iter(), self.universe, self.words, self.label_bits)
    }

    /// The independent code of the same words: every subcomponent at its own rate.
    pub fn independent_bits(sets: &Sets) -> f64 {
        let words = sets.rows() as u64;
        sets.columns().iter().map(|c| kt_bits(c.len() as u64, words)).sum()
    }
}

/// Fit the concepts of `sets` (module note, "The fit"), each concept's name costing `label_bits`.
pub fn fit(sets: &Sets, label_bits: f64) -> Fit {
    let universe = sets.universe;
    let words = sets.rows() as u64;
    let columns = sets.columns();
    let mut groups: Vec<Option<Concept>> =
        (0..universe).filter(|j| !columns[*j].is_empty()).map(|j| Some(Concept::alone(j as u32, &columns[j], words))).collect();
    let mut open: Vec<bool> = vec![true; groups.len()];
    let library = |g: &Concept| g.library(universe, label_bits);
    let mut rounds = Vec::new();
    loop {
        let alive: Vec<usize> = (0..groups.len()).filter(|g| groups[*g].is_some()).collect();
        let total = total_bits(groups.iter().flatten(), universe, words, label_bits);
        let concepts = groups.iter().flatten().filter(|g| g.members.len() > 1).count();
        let opened: Vec<usize> = alive.iter().copied().filter(|g| open[*g]).collect();
        // Which groups each word invokes.
        let mut start = vec![0usize; words as usize + 1];
        for g in &alive {
            for t in &groups[*g].as_ref().expect("alive").invoked {
                start[*t as usize + 1] += 1;
            }
        }
        for t in 0..words as usize {
            start[t + 1] += start[t];
        }
        let mut fill = start.clone();
        let mut invoking = vec![0u32; start[words as usize]];
        for g in &alive {
            for t in &groups[*g].as_ref().expect("alive").invoked {
                invoking[fill[*t as usize]] = *g as u32;
                fill[*t as usize] += 1;
            }
        }
        let proposals: Vec<(usize, Option<(usize, f64, Concept)>)> = opened
            .par_iter()
            .map_init(
                || (vec![0u64; groups.len()], Vec::<usize>::new()),
                |(count, touched), &g| {
                    let a = groups[g].as_ref().expect("alive");
                    for t in &a.invoked {
                        for h in &invoking[start[*t as usize]..start[*t as usize + 1]] {
                            let h = *h as usize;
                            if h != g {
                                if count[h] == 0 {
                                    touched.push(h);
                                }
                                count[h] += 1;
                            }
                        }
                    }
                    let mut ranked: Vec<(f64, usize)> = touched
                        .iter()
                        .map(|h| {
                            let b = groups[*h].as_ref().expect("alive");
                            (information(a.invoked.len() as u64, b.invoked.len() as u64, count[*h], words), *h)
                        })
                        .filter(|(i, _)| *i > 0.0)
                        .collect();
                    for h in touched.drain(..) {
                        count[h] = 0;
                    }
                    ranked.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
                    for (info, h) in ranked {
                        let b = groups[h].as_ref().expect("alive");
                        let members = union(&a.members, &b.members);
                        let added = subset_code_len_bits(universe, members.len()).map_or(f64::INFINITY, |x| x as f64) + label_bits
                            - library(a)
                            - library(b);
                        if info <= added {
                            break;
                        }
                        let both = union(&a.invoked, &b.invoked);
                        let common = intersection(&a.invoked, &b.invoked);
                        let merged = fit_group(members, &columns, words, &[&both, &common]);
                        let delta = merged.bits + library(&merged) - a.bits - library(a) - b.bits - library(b);
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
        let mut used = vec![false; groups.len()];
        let mut merges = 0;
        for (g, h, _, merged) in accepted {
            if used[g] || used[h] {
                continue;
            }
            used[g] = true;
            used[h] = true;
            groups[g] = None;
            groups[h] = None;
            groups.push(Some(merged));
            open.push(true);
            used.push(true);
            merges += 1;
        }
        rounds.push(Round { groups: alive.len(), concepts, open: opened.len(), merges, total_bits: total });
        log::info!("concepts: round {} groups {} concepts {} open {} merges {} total {:.0} bits", rounds.len(), alive.len(), concepts, opened.len(), merges, total);
        if opened.is_empty() {
            break;
        }
    }
    let groups: Vec<Concept> = groups.into_iter().flatten().collect();
    Fit { universe, words, label_bits, groups, rounds }
}

/// A concept's frozen rates.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Rates {
    pub members: Vec<u32>,
    /// `π`: the rate at which a word invokes it.
    pub invoked: f64,
    /// Per member, the rate it runs given the concept invoked (`q`) and not invoked (`a`).
    pub on: Vec<f64>,
    pub off: Vec<f64>,
}

/// A word's bits under a [`Model`].
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Bits {
    /// The concept choices: what the text carries.
    pub choices: f64,
    /// The members of every concept given its choice.
    pub members: f64,
    /// The subcomponents in no concept.
    pub alone: f64,
    /// The independent code of the same set.
    pub independent: f64,
}

/// Frozen rates of a [`Fit`], for coding words it was not fitted on.
#[derive(Clone, Debug)]
pub struct Model {
    pub universe: usize,
    pub concepts: Vec<Rates>,
    /// Per subcomponent, its concept and its position in it.
    pub concept_of: Vec<Option<(u32, u32)>>,
    /// Per subcomponent, its own rate (the independent code, and the code of one in no concept).
    pub rate: Vec<f64>,
    /// Bits of a word that ran nothing, independent code; and of every concept's choice and
    /// members and every lone subcomponent when nothing ran.
    empty_independent: f64,
    empty_choices: f64,
    empty_members: f64,
    empty_alone: f64,
}

fn bernoulli_bits(on: bool, p: f64) -> f64 {
    -(if on { p } else { 1.0 - p }).log2()
}

impl Model {
    pub fn new(fit: &Fit) -> Self {
        let words = fit.words;
        let mut rate = vec![kt_rate(0, words); fit.universe];
        let mut concept_of = vec![None; fit.universe];
        let mut concepts = Vec::new();
        for g in &fit.groups {
            for (j, n) in g.members.iter().zip(&g.ran) {
                rate[*j as usize] = kt_rate(*n, words);
            }
            if g.members.len() < 2 {
                continue;
            }
            let z = g.invoked.len() as u64;
            for (i, j) in g.members.iter().enumerate() {
                concept_of[*j as usize] = Some((concepts.len() as u32, i as u32));
            }
            concepts.push(Rates {
                members: g.members.clone(),
                invoked: kt_rate(z, words),
                on: g.ran_invoked.iter().map(|k| kt_rate(*k, z)).collect(),
                off: g.ran.iter().zip(&g.ran_invoked).map(|(n, k)| kt_rate(n - k, words - z)).collect(),
            });
        }
        let empty_independent = rate.iter().map(|p| bernoulli_bits(false, *p)).sum();
        let empty_choices = concepts.iter().map(|c| bernoulli_bits(false, c.invoked)).sum();
        let empty_members = concepts.iter().flat_map(|c| c.off.iter().map(|a| bernoulli_bits(false, *a))).sum();
        let empty_alone = (0..fit.universe).filter(|j| concept_of[*j].is_none()).map(|j| bernoulli_bits(false, rate[j])).sum();
        Self { universe: fit.universe, concepts, concept_of, rate, empty_independent, empty_choices, empty_members, empty_alone }
    }

    /// The concepts a word with set `row` invokes (ascending) and its bits (module note,
    /// "Coding new words"): each touched concept is invoked when that is the cheaper choice.
    pub fn encode(&self, row: &[u32]) -> (Vec<u32>, Bits) {
        let mut bits = Bits {
            choices: self.empty_choices,
            members: self.empty_members,
            alone: self.empty_alone,
            independent: self.empty_independent,
        };
        let mut touched: Vec<(u32, u32)> = Vec::new();
        for j in row {
            let j = *j as usize;
            let p = self.rate[j];
            bits.independent += bernoulli_bits(true, p) - bernoulli_bits(false, p);
            match self.concept_of[j] {
                Some(c) => touched.push(c),
                None => bits.alone += bernoulli_bits(true, p) - bernoulli_bits(false, p),
            }
        }
        touched.sort_unstable();
        let mut invoked = Vec::new();
        let mut i = 0;
        while i < touched.len() {
            let c = touched[i].0;
            let rates = &self.concepts[c as usize];
            let mut ran = vec![false; rates.members.len()];
            while i < touched.len() && touched[i].0 == c {
                ran[touched[i].1 as usize] = true;
                i += 1;
            }
            let side = |q: &[f64]| -> f64 { ran.iter().zip(q).map(|(x, p)| bernoulli_bits(*x, *p)).sum() };
            let off_members: f64 = rates.off.iter().map(|a| bernoulli_bits(false, *a)).sum();
            let on = bernoulli_bits(true, rates.invoked) + side(&rates.on);
            let off = bernoulli_bits(false, rates.invoked) + side(&rates.off);
            bits.members -= off_members;
            bits.choices -= bernoulli_bits(false, rates.invoked);
            if on < off {
                invoked.push(c);
                bits.choices += bernoulli_bits(true, rates.invoked);
                bits.members += on - bernoulli_bits(true, rates.invoked);
            } else {
                bits.choices += bernoulli_bits(false, rates.invoked);
                bits.members += off - bernoulli_bits(false, rates.invoked);
            }
        }
        (invoked, bits)
    }

    /// The set a list of invoked concepts decodes to: each one's members more often on than off
    /// when it is invoked, ascending.
    pub fn decode(&self, invoked: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = invoked
            .iter()
            .flat_map(|c| {
                let r = &self.concepts[*c as usize];
                r.members.iter().zip(&r.on).filter(|(_, q)| **q > 0.5).map(|(j, _)| *j)
            })
            .collect();
        out.sort_unstable();
        out
    }
}

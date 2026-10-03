//! Rank-k gated subcomponents, their ranks chosen by the code (#2951).
//!
//! A site's subcomponents are blocks of its library's columns, one gate each per input
//! ([`super::masked`], "Blocks"). Which columns share a gate, and each block's rank, are decided by
//! one total, summed over every coded input (word):
//!
//! ```text
//! Σ_words [ Σ_{blocks on} bits(block) + n KL / ln 2 ],
//! ```
//!
//! `bits(block)` the shortest description of the block's weights at the precision the output
//! needs ([`Describe`]): each word pays for the weights that ran on it ([`Coder::ran`]), with no
//! library amortised and no listing, so reuse is not free, everything on is not free, and a block
//! doing two jobs pays its whole description on every word where either runs. One gate scales a
//! block along a line; the block's internal columns are a coordinate choice (`(UR)(VR^{-T})ᵀ` is the
//! same block for any invertible `R`), so only the whole-block gate is offered.
//!
//! [`Generic`] describes a rank-`r` block on a `d_in → d_out` site by its map `U Vᵀ` up to `GL(r)`:
//! `r (d_in + d_out − r)` reals (`r` its numerical rank after balancing, so duplicate columns
//! collapse), sent on one lattice step `δ` chosen against the expected KL of rounding to it,
//! `n/(2 ln 2) · δ²/12 · [tr F tr(VᵀCV) + tr C tr(UᵀFU)]` (`C` the reads' second moment, `F` the
//! written value's Fisher, the factors balanced so `UᵀU = VᵀV = S`). Structured families plug in
//! through the same trait.
//!
//! The moves, each only a proposal decided by the exact total:
//!
//! * **Merge.** Two blocks of a site that co-fire become one, their columns concatenated, so the
//!   map is unchanged. On an input where exactly one was on, the merged gate takes whichever of on
//!   and off its second-order KL bits plus description predict cheaper. Each block proposes its
//!   most co-firing partner (largest Jaccard overlap of their inputs, ties by more shared inputs,
//!   then by the `|cos|` of their maps in `⟨U_i V_iᵀ, U_j V_jᵀ⟩ = tr[(U_iᵀU_j)(V_jᵀV_i)]`).
//! * **Split.** A block of rank ≥ 2 becomes rank-one blocks along its principal coordinates (its
//!   columns rotated within `GL(r)` so its output columns are orthonormal and its members'
//!   coordinates uncorrelated), each on wherever the block was; the selection then keeps on only
//!   what each input uses.
//! * **Shrink.** A block of rank ≥ 2 loses its weakest balanced direction (the core's last
//!   singular pair): fewer reals on every word it runs on, against the KL of the direction lost.
//! * **Grow.** A block gains the top singular pair of `(I − P_U) G (I − P_V)`, `G = Σ_on g xᵀ` the
//!   KL's gradient in its map over the words it runs on (`g` at the written value): the part of the
//!   gradient no change inside the block's spans reaches, stepped by Gauss–Newton in the written
//!   Fisher, more payload for the same gate.
//!
//! A round's merges are kept only when the exact total (every input's masked forward) falls; a
//! refused set is halved by predicted saving until a single refused merge is set aside. Grows,
//! shrinks and splits are tested one at a time. Every kept merge is followed by the selection, kept when it
//! lowers the total again.

use super::dense::{QrMode, eigh, qr, svd};
use super::masked::{Coder, Library, Masked, Site, Target, fisher, forward, mask_gradients, read_values, select};
use super::operator_program::{FamilyInputs, OperatorProgram};
use gam_linalg::faer_ndarray::fast_ab;
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::LN_2;
use std::sync::Arc;

/// A block's description bits per word it runs on (module note): `u` is `r × d_out`, `v` is
/// `r × d_in` (the library's convention, the map `uᵀ v`), on site `site`.
pub trait Describe: Sync {
    fn bits(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<f64, String>;
}

/// The generic description (module note), from each site's reads' second moment (the masked
/// program reads `z = Vᵀx` uncentred) and written Fisher.
pub struct Generic {
    sites: Vec<GenericSite>,
    observations: f64,
}

struct GenericSite {
    moment: Array2<f64>,
    fisher: Array2<f64>,
    trace_c: f64,
    trace_f: f64,
}

impl Generic {
    /// From the sites' measured statistics ([`super::masked::site_statistics`]), `n` the code's.
    pub fn new(statistics: &[super::pieces::Site], observations: f64) -> Self {
        let sites = statistics
            .iter()
            .map(|site| {
                let moment = symmetric(&site.second_moment);
                let fisher = symmetric(&site.fisher);
                let (trace_c, trace_f) = (moment.diag().sum(), fisher.diag().sum());
                GenericSite { moment, fisher, trace_c, trace_f }
            })
            .collect();
        Self { sites, observations }
    }
}

impl Describe for Generic {
    fn bits(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<f64, String> {
        let (u, v) = balanced(u, v)?;
        let r = u.nrows();
        if r == 0 {
            return Ok(0.0);
        }
        let reals = (r * (u.ncols() + v.ncols() - r)) as f64;
        let metric = &self.sites[site];
        let tv = (&fast_ab(&v, &metric.moment) * &v).sum();
        let tu = (&fast_ab(&u, &metric.fisher) * &u).sum();
        // The expected KL bits of rounding every factor entry to a step `δ`, per `δ²`.
        let a = self.observations / (2.0 * LN_2) / 12.0 * (metric.trace_f * tv + metric.trace_c * tu);
        if !(a > 0.0) {
            return Ok(0.0);
        }
        let step = (reals / (2.0 * a * LN_2)).sqrt();
        let largest = u.iter().chain(v.iter()).fold(0.0_f64, |m, x| m.max(x.abs()));
        Ok(reals * ((2.0 * largest / step).log2().max(0.0) + 1.0 / (2.0 * LN_2)))
    }
}

/// The block `uᵀ v` in balanced factors (`u'ᵀ v'` the same map, `u' u'ᵀ = v' v'ᵀ = S` diagonal,
/// descending): thin QRs `U = Q_U R_U`, `V = Q_V R_V` of both sides, the core `R_U R_Vᵀ = A S Bᵀ`,
/// `U' = Q_U A √S`, `V' = Q_V B √S`. Only the core's singular values within its decomposition's
/// band are dropped, never one side's directions alone: a direction small on one side and large
/// on the other is a full direction of the map.
pub fn balanced(u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let (d_out, d_in) = (u.ncols(), v.ncols());
    if u.nrows() == 0 {
        return Ok((Array2::zeros((0, d_out)), Array2::zeros((0, d_in))));
    }
    let thin = |f: ArrayView2<'_, f64>| -> Result<(Array2<f64>, Array2<f64>), String> {
        let qr = qr(f.t(), QrMode::Economic).map_err(|e| format!("{e:?}"))?;
        Ok((qr.q.ok_or("a thin QR without Q")?, qr.r))
    };
    let ((q_u, r_u), (q_v, r_v)) = (thin(u)?, thin(v)?);
    let decomposed = svd(r_u.dot(&r_v.t()).view(), false).map_err(|e| format!("{e:?}"))?;
    let kept: Vec<usize> = (0..decomposed.singular_values.len()).filter(|&j| decomposed.singular_values[j] > decomposed.band).collect();
    let roots = Array1::from_iter(kept.iter().map(|&j| decomposed.singular_values[j].sqrt()));
    let a = decomposed.u.select(Axis(1), &kept) * &roots;
    let b = decomposed.vt.select(Axis(0), &kept).t().to_owned() * &roots;
    Ok((q_u.dot(&a).t().to_owned(), q_v.dot(&b).t().to_owned()))
}

/// What the code is measured on: the model, its decomposed sites (in the model's node indices, as
/// [`super::masked::sites`] gives them), the coded inputs in batches (a sequence each, or any rows)
/// each with its target, and how blocks are described.
pub struct Coded<'a> {
    pub model: &'a OperatorProgram,
    pub sites: Vec<Site>,
    pub batches: Vec<(FamilyInputs, Target)>,
    /// `n` of the code.
    pub observations: f64,
    /// Sampled-label reverse passes per Fisher diagonal.
    pub samples: usize,
    pub describe: &'a dyn Describe,
}

/// A blocked decomposition: per site its library (shared between decompositions that differ
/// elsewhere) and blocks (their ranks, in column order, stable ids and description bits once
/// priced), and per batch every site's masks (`rows × B`).
#[derive(Clone, Debug)]
pub struct Blocked {
    pub libraries: Vec<Arc<Library>>,
    pub ranks: Vec<Vec<usize>>,
    pub ids: Vec<Vec<usize>>,
    pub masks: Vec<Vec<Array2<f64>>>,
    priced: Vec<Vec<Option<f64>>>,
    next_id: usize,
}

impl Blocked {
    /// Every piece its own block.
    pub fn rank_one(libraries: Vec<Library>, masks: Vec<Vec<Array2<f64>>>) -> Self {
        let ranks: Vec<Vec<usize>> = libraries.iter().map(|l| vec![1; l.v.nrows()]).collect();
        Self::new(libraries, ranks, masks)
    }

    /// Site `k`'s columns gated in blocks of `ranks[k]`, with `masks` per batch and site.
    pub fn new(libraries: Vec<Library>, ranks: Vec<Vec<usize>>, masks: Vec<Vec<Array2<f64>>>) -> Self {
        let mut next_id = 0;
        let ids = ranks
            .iter()
            .map(|r| {
                let ids = (next_id..next_id + r.len()).collect();
                next_id += r.len();
                ids
            })
            .collect();
        let priced = ranks.iter().map(|r| vec![None; r.len()]).collect();
        Self { libraries: libraries.into_iter().map(Arc::new).collect(), ranks, ids, masks, priced, next_id }
    }

    /// Every site's columns as one block, on for every input: the dense model, described.
    pub fn whole(&self) -> Self {
        let ranks: Vec<Vec<usize>> = self.libraries.iter().map(|l| vec![l.v.nrows()]).collect();
        let masks = self.masks.iter().map(|m| m.iter().map(|m| Array2::ones((m.nrows(), 1))).collect()).collect();
        let ids = (0..ranks.len()).map(|k| vec![self.next_id + k]).collect();
        let priced = ranks.iter().map(|_| vec![None]).collect();
        Self { libraries: self.libraries.clone(), ranks, ids, masks, priced, next_id: self.next_id + self.ranks.len() }
    }

    /// The same blocks, every one off for every input.
    pub fn off(&self) -> Self {
        let mut out = self.clone();
        out.masks.iter_mut().flatten().for_each(|m| m.fill(0.0));
        out
    }

    /// Block `c` of site `k`'s description bits, once priced.
    pub fn bits(&self, k: usize, c: usize) -> Option<f64> {
        self.priced[k][c]
    }

    /// The masked program of this decomposition.
    pub fn masked(&self, coded: &Coded<'_>) -> Result<Masked, String> {
        Masked::build_blocks(coded.model, coded.sites.clone(), self.libraries.iter().map(|l| (**l).clone()).collect(), self.ranks.clone())
    }

    fn starts(&self, k: usize) -> Vec<usize> {
        let mut out = vec![0];
        for r in &self.ranks[k] {
            out.push(out.last().copied().unwrap_or(0) + r);
        }
        out
    }

    fn index(&self, k: usize, id: usize) -> Option<usize> {
        self.ids[k].iter().position(|i| *i == id)
    }

    /// Block `c` of site `k`'s factors `(u, v)`.
    pub fn factors(&self, k: usize, c: usize) -> (ArrayView2<'_, f64>, ArrayView2<'_, f64>) {
        let start = self.starts(k)[c];
        let end = start + self.ranks[k][c];
        (self.libraries[k].u.slice(s![start..end, ..]), self.libraries[k].v.slice(s![start..end, ..]))
    }

    /// Price every block not yet priced ([`Describe`]).
    pub fn price(&mut self, coded: &Coded<'_>) -> Result<(), String> {
        let costs = costs(coded, self)?;
        self.priced = costs.iter().map(|c| c.iter().map(|x| Some(*x)).collect()).collect();
        Ok(())
    }
}

/// Every block's description bits: its price when known, else described now (blocks in parallel).
pub fn costs(coded: &Coded<'_>, blocked: &Blocked) -> Result<Vec<Array1<f64>>, String> {
    use rayon::prelude::*;
    (0..blocked.ranks.len())
        .map(|k| {
            let values: Vec<f64> = (0..blocked.ranks[k].len())
                .into_par_iter()
                .map(|c| match blocked.priced[k][c] {
                    Some(bits) => Ok(bits),
                    None => {
                        let (u, v) = blocked.factors(k, c);
                        coded.describe.bits(k, u, v)
                    }
                })
                .collect::<Result<_, String>>()?;
            Ok(Array1::from(values))
        })
        .collect()
}

/// `⟨U_i V_iᵀ, U_j V_jᵀ⟩_F = tr[(U_iᵀU_j)(V_jᵀV_i)]` of two blocks of `library`, given by their
/// column ranges, without forming either map.
pub fn block_inner(library: &Library, (si, ki): (usize, usize), (sj, kj): (usize, usize)) -> f64 {
    let (ui, vi) = (library.u.slice(s![si..si + ki, ..]), library.v.slice(s![si..si + ki, ..]));
    let (uj, vj) = (library.u.slice(s![sj..sj + kj, ..]), library.v.slice(s![sj..sj + kj, ..]));
    (&ui.dot(&uj.t()) * &vi.dot(&vj.t())).sum()
}

/// The cosine of two blocks' maps in the Frobenius inner product ([`block_inner`]).
pub fn block_cosine(library: &Library, a: (usize, usize), b: (usize, usize)) -> f64 {
    let norm = block_inner(library, a, a).max(0.0).sqrt() * block_inner(library, b, b).max(0.0).sqrt();
    if norm > 0.0 { block_inner(library, a, b) / norm } else { 0.0 }
}

fn symmetric(m: &Array2<f64>) -> Array2<f64> {
    (m + &m.t()) * 0.5
}

/// A decomposition's code, totalled over the scored rows of every batch (module note).
#[derive(Clone, Debug, Default)]
pub struct Bits {
    /// `Σ_words Σ_{blocks on}` description bits.
    pub described: f64,
    /// `n Σ KL / ln 2`.
    pub kl: f64,
    /// The scored rows, their summed KL (nats), active blocks and active rank (`Σ k_c` over the
    /// blocks on: rank-one equivalents).
    pub rows: f64,
    pub kl_nats: f64,
    pub active_blocks: f64,
    pub active_rank: f64,
    pub blocks: usize,
    pub pieces: usize,
}

impl Bits {
    pub fn total(&self) -> f64 {
        self.described + self.kl
    }

    /// `(bits, KL nats, active blocks, active rank-one equivalents)` per scored row.
    pub fn per_row(&self) -> (f64, f64, f64, f64) {
        let rows = self.rows.max(1.0);
        (self.total() / rows, self.kl_nats / rows, self.active_blocks / rows, self.active_rank / rows)
    }
}

/// The exact code of `blocked` (module note) and each batch's per-row KL.
pub fn measure(coded: &Coded<'_>, blocked: &Blocked) -> Result<(Bits, Vec<Array1<f64>>), String> {
    let masked = blocked.masked(coded)?;
    let costs = costs(coded, blocked)?;
    let mut bits = Bits { blocks: blocked.ranks.iter().map(Vec::len).sum(), pieces: blocked.ranks.iter().flatten().sum(), ..Bits::default() };
    let scale = coded.observations / LN_2;
    let mut kls = Vec::new();
    for ((inputs, target), masks) in coded.batches.iter().zip(&blocked.masks) {
        let described = Coder::ran(costs.clone(), inputs.rows).bits(masks);
        let (kl, _, _) = forward(&masked, &masked.family(inputs, masks), target)?;
        for r in (0..inputs.rows).filter(|r| target.scores(*r)) {
            bits.rows += 1.0;
            bits.described += described[r];
            bits.kl_nats += kl[r];
            bits.kl += scale * kl[r];
            for (k, m) in masks.iter().enumerate() {
                for (b, rank) in blocked.ranks[k].iter().enumerate() {
                    if m[[r, b]] > 0.0 {
                        bits.active_blocks += 1.0;
                        bits.active_rank += *rank as f64;
                    }
                }
            }
        }
        kls.push(kl);
    }
    Ok((bits, kls))
}

/// Every batch's masks selected again ([`select`]) under the blocks' description bits.
pub fn reselect(coded: &Coded<'_>, blocked: &Blocked) -> Result<Blocked, String> {
    let masked = blocked.masked(coded)?;
    let costs = costs(coded, blocked)?;
    let mut out = blocked.clone();
    for (b, (inputs, target)) in coded.batches.iter().enumerate() {
        let coder = Coder::ran(costs.clone(), inputs.rows);
        let (masks, _) = select(&masked, inputs, target, blocked.masks[b].clone(), &coder, coded.observations, coded.samples)?;
        out.masks[b] = masks;
    }
    Ok(out)
}

/// One proposed merge: site, the two blocks' ids, its predicted saving (bits), and per batch the
/// merged gate.
struct Merge {
    site: usize,
    a: usize,
    b: usize,
    saving: f64,
    /// The merged block's description bits.
    bits: f64,
    gates: Vec<Array1<f64>>,
}

/// Each block's most co-firing partner (module note, "Merge"), as `(site, a, b)` block indices
/// with `a < b`, without repeats.
fn partners(blocked: &Blocked, refused: &BTreeSet<(usize, usize, usize)>) -> Vec<(usize, usize, usize)> {
    let mut out = BTreeSet::new();
    for k in 0..blocked.ranks.len() {
        let width = blocked.ranks[k].len();
        let mut fired = vec![0.0f64; width];
        let mut shared: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for masks in &blocked.masks {
            let m = &masks[k];
            for row in m.outer_iter() {
                let on: Vec<usize> = (0..width).filter(|&c| row[c] > 0.0).collect();
                for (i, &a) in on.iter().enumerate() {
                    fired[a] += 1.0;
                    for &b in &on[i + 1..] {
                        *shared.entry((a, b)).or_insert(0.0) += 1.0;
                    }
                }
            }
        }
        let starts = blocked.starts(k);
        let range = |c: usize| (starts[c], blocked.ranks[k][c]);
        // Per block, its best partner by (Jaccard, shared inputs, |cos|).
        let mut best: Vec<Option<((f64, f64, f64), usize)>> = vec![None; width];
        for (&(a, b), &n) in &shared {
            let (ia, ib) = (blocked.ids[k][a], blocked.ids[k][b]);
            if refused.contains(&(k, ia.min(ib), ia.max(ib))) {
                continue;
            }
            let jaccard = n / (fired[a] + fired[b] - n);
            let cos = block_cosine(&blocked.libraries[k], range(a), range(b)).abs();
            let key = (jaccard, n, cos);
            for (me, other) in [(a, b), (b, a)] {
                let better = best[me].as_ref().is_none_or(|(old, _)| key.partial_cmp(old) == Some(std::cmp::Ordering::Greater));
                if better {
                    best[me] = Some((key, other));
                }
            }
        }
        for (a, partner) in best.iter().enumerate() {
            if let Some((_, b)) = partner {
                out.insert((k, a.min(*b), a.max(*b)));
            }
        }
    }
    out.into_iter().collect()
}

/// The predicted saving of each candidate merge and its merged gates, from every batch's mask
/// gradients and Fisher diagonals (module note, "Merge").
fn predict(coded: &Coded<'_>, blocked: &Blocked, candidates: &[(usize, usize, usize)]) -> Result<Vec<Merge>, String> {
    let masked = blocked.masked(coded)?;
    let costs = costs(coded, blocked)?;
    let scale = coded.observations / LN_2;
    let mut merges: Vec<Merge> = Vec::new();
    for &(site, a, b) in candidates {
        let ((ua, va), (ub, vb)) = (blocked.factors(site, a), blocked.factors(site, b));
        let u = ndarray::concatenate(Axis(0), &[ua, ub]).map_err(|e| e.to_string())?;
        let v = ndarray::concatenate(Axis(0), &[va, vb]).map_err(|e| e.to_string())?;
        let bits = coded.describe.bits(site, u.view(), v.view())?;
        merges.push(Merge { site, a: blocked.ids[site][a], b: blocked.ids[site][b], saving: 0.0, bits, gates: Vec::new() });
    }
    for (batch, (inputs, target)) in coded.batches.iter().enumerate() {
        let masks = &blocked.masks[batch];
        let family = masked.family(inputs, masks);
        let (_, trace, cotangent) = forward(&masked, &family, target)?;
        let g = mask_gradients(&masked, &family, &trace, cotangent)?;
        let h = fisher(&masked, &family, &trace, target, coded.samples, 0xB10C + batch as u64, false)?;
        for (merge, &(k, a, b)) in merges.iter_mut().zip(candidates) {
            let (m, g, h) = (&masks[k], &g[k], &h[k].0);
            let (cost_a, cost_b, cost_ab) = (costs[k][a], costs[k][b], merge.bits);
            let mut gate = Array1::<f64>::zeros(inputs.rows);
            for r in 0..inputs.rows {
                let (on_a, on_b) = (m[[r, a]] > 0.0, m[[r, b]] > 0.0);
                if !target.scores(r) {
                    gate[r] = f64::from(on_a || on_b);
                    continue;
                }
                match (on_a, on_b) {
                    (true, true) => {
                        gate[r] = 1.0;
                        merge.saving += cost_a + cost_b - cost_ab;
                    }
                    (false, false) => {}
                    (true, false) | (false, true) => {
                        let (on, off) = if on_a { (a, b) } else { (b, a) };
                        let cost_on = if on_a { cost_a } else { cost_b };
                        // Gate on: the off block's columns join; gate off: the on block's leave.
                        let join = scale * (g[[r, off]] + 0.5 * h[[r, off]]) + cost_ab - cost_on;
                        let leave = scale * (-g[[r, on]] + 0.5 * h[[r, on]]) - cost_on;
                        if join <= leave {
                            gate[r] = 1.0;
                            merge.saving -= join;
                        } else {
                            merge.saving -= leave;
                        }
                    }
                }
            }
            merge.gates.push(gate);
        }
    }
    Ok(merges)
}

/// `blocked` with `merges` applied: each pair's columns made one block at the first one's place,
/// gated by the merge's gates.
fn apply(blocked: &Blocked, merges: &[&Merge]) -> Result<Blocked, String> {
    let mut out = blocked.clone();
    let mut by_site: BTreeMap<usize, Vec<&Merge>> = BTreeMap::new();
    for m in merges {
        by_site.entry(m.site).or_default().push(m);
    }
    for (k, merges) in by_site {
        let starts = blocked.starts(k);
        let width = blocked.ranks[k].len();
        // Each block's new position: the second of a pair follows the first.
        let mut follows: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
        let mut absorbed = BTreeSet::new();
        for (i, m) in merges.iter().enumerate() {
            let (a, b) = (blocked.index(k, m.a).ok_or("merge of a missing block")?, blocked.index(k, m.b).ok_or("merge of a missing block")?);
            follows.insert(a.min(b), (a.max(b), i));
            absorbed.insert(a.max(b));
        }
        let (mut columns, mut ranks, mut ids, mut priced) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut gate_of: Vec<(usize, Option<usize>)> = Vec::new();
        for c in 0..width {
            if absorbed.contains(&c) {
                continue;
            }
            columns.extend(starts[c]..starts[c + 1]);
            match follows.get(&c) {
                Some(&(b, i)) => {
                    columns.extend(starts[b]..starts[b + 1]);
                    ranks.push(blocked.ranks[k][c] + blocked.ranks[k][b]);
                    ids.push(out.next_id);
                    out.next_id += 1;
                    priced.push(Some(merges[i].bits));
                    gate_of.push((c, Some(i)));
                }
                None => {
                    ranks.push(blocked.ranks[k][c]);
                    ids.push(blocked.ids[k][c]);
                    priced.push(blocked.priced[k][c]);
                    gate_of.push((c, None));
                }
            }
        }
        let library = &blocked.libraries[k];
        out.libraries[k] = Arc::new(Library { v: library.v.select(Axis(0), &columns), u: library.u.select(Axis(0), &columns), mean: library.mean.clone() });
        out.ranks[k] = ranks;
        out.ids[k] = ids;
        out.priced[k] = priced;
        for (batch, masks) in out.masks.iter_mut().enumerate() {
            let old = &blocked.masks[batch][k];
            let mut new = Array2::<f64>::zeros((old.nrows(), gate_of.len()));
            for (j, (c, merge)) in gate_of.iter().enumerate() {
                match merge {
                    Some(i) => new.column_mut(j).assign(&merges[*i].gates[batch]),
                    None => new.column_mut(j).assign(&old.column(*c)),
                }
            }
            masks[k] = new;
        }
    }
    Ok(out)
}

/// `blocked` with block `c` of site `k` replaced by the rows `(u, v)`, gated in blocks of `parts`,
/// each on wherever block `c` was.
fn replace_block(blocked: &Blocked, k: usize, c: usize, u: &Array2<f64>, v: &Array2<f64>, parts: &[usize]) -> Result<Blocked, String> {
    let (start, rank) = (blocked.starts(k)[c], blocked.ranks[k][c]);
    let library = &blocked.libraries[k];
    let stack = |old: &Array2<f64>, new: &Array2<f64>| -> Result<Array2<f64>, String> {
        ndarray::concatenate(Axis(0), &[old.slice(s![..start, ..]), new.view(), old.slice(s![start + rank.., ..])]).map_err(|e| e.to_string())
    };
    let mut out = blocked.clone();
    out.libraries[k] = Arc::new(Library { v: stack(&library.v, v)?, u: stack(&library.u, u)?, mean: library.mean.clone() });
    out.ranks[k].splice(c..=c, parts.iter().copied());
    out.ids[k].splice(c..=c, out.next_id..out.next_id + parts.len());
    out.next_id += parts.len();
    out.priced[k].splice(c..=c, std::iter::repeat_n(None, parts.len()));
    for masks in out.masks.iter_mut() {
        let old = masks[k].clone();
        let column = old.column(c).insert_axis(Axis(1));
        let views: Vec<_> = std::iter::once(old.slice(s![.., ..c]))
            .chain(std::iter::repeat_n(column.view(), parts.len()))
            .chain(std::iter::once(old.slice(s![.., c + 1..])))
            .collect();
        masks[k] = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// `blocked` with block `c` of site `k` split into rank-one blocks along its principal coordinates
/// (module note, "Split"), each on where the block was: from balanced factors `U' = O S^{1/2}`
/// (`O` orthonormal) the gauge `A = S^{-1/2} R`, `R` diagonalising the members' coordinates
/// `S^{1/2} M S^{1/2}` (`M` the second moment of `z' = V'ᵀ x` over the inputs where the block is
/// on), gives `U'' = U' A` with orthonormal columns and uncorrelated coordinates, `V'' = V' A^{-T}`.
pub(crate) fn split_block(blocked: &Blocked, k: usize, c: usize, moment: &Array2<f64>) -> Result<Blocked, String> {
    let (u, v) = blocked.factors(k, c);
    let (u, v) = balanced(u, v)?;
    let s = Array1::from_iter(u.outer_iter().map(|row| row.dot(&row)));
    let roots = s.mapv(f64::sqrt);
    let scaled = Array2::from_shape_fn(moment.dim(), |(i, j)| roots[i] * moment[[i, j]] * roots[j]);
    let r = eigh(symmetric(&scaled).view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?.vectors;
    let down = Array2::from_shape_fn(r.dim(), |(i, j)| r[[i, j]] / roots[i]);
    let up = Array2::from_shape_fn(r.dim(), |(i, j)| r[[i, j]] * roots[i]);
    let (u_new, v_new) = (down.t().dot(&u), up.t().dot(&v));
    replace_block(blocked, k, c, &u_new, &v_new, &vec![1; u_new.nrows()])
}

/// `blocked` with block `c` of site `k` in balanced factors less its weakest direction (module
/// note, "Shrink"), or `None` when it has one direction.
fn shrink_block(blocked: &Blocked, k: usize, c: usize) -> Result<Option<Blocked>, String> {
    let (u, v) = blocked.factors(k, c);
    let (u, v) = balanced(u, v)?;
    let r = u.nrows();
    if r < 2 {
        return Ok(None);
    }
    let (u, v) = (u.slice(s![..r - 1, ..]).to_owned(), v.slice(s![..r - 1, ..]).to_owned());
    Ok(Some(replace_block(blocked, k, c, &u, &v, &[r - 1])?))
}

/// Per block, the direction a grow would add (module note, "Grow"): `(u: 1 × d_out, v: 1 × d_in)`,
/// or `None` where the block's reach covers the gradient.
fn grow_directions(coded: &Coded<'_>, blocked: &Blocked) -> Result<Vec<Vec<Option<(Array2<f64>, Array2<f64>)>>>, String> {
    let masked = blocked.masked(coded)?;
    // Per site and block: the gradient `G = Σ_on g xᵀ`, and per member row `(g, x)` for the
    // curvature `Σ_on (pᵀ F p)(qᵀ x)²` with `F` the written Fisher.
    let mut gradients: Vec<Vec<Array2<f64>>> = Vec::new();
    let mut reads: Vec<Vec<Vec<Array2<f64>>>> = Vec::new();
    let mut fishers: Vec<Array2<f64>> = Vec::new();
    for (batch, ((inputs, target), masks)) in coded.batches.iter().zip(&blocked.masks).enumerate() {
        let family = masked.family(inputs, masks);
        let (_, trace, cotangent) = forward(&masked, &family, target)?;
        let back = super::derivatives::vjp(&masked.program, &family, &trace, cotangent).map_err(|e| e.to_string())?;
        let curvature = fisher(&masked, &family, &trace, target, coded.samples, 0x6205 + batch as u64, true)?;
        for (k, site) in masked.sites.iter().enumerate() {
            let x = read_values(&trace, site)?;
            let written: Vec<Array2<f64>> =
                site.writes.iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
            let g = ndarray::concatenate(Axis(1), &written.iter().map(|w| w.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
            if batch == 0 {
                gradients.push(vec![Array2::zeros((g.ncols(), x.ncols())); blocked.ranks[k].len()]);
                reads.push(vec![Vec::new(); blocked.ranks[k].len()]);
                fishers.push(Array2::zeros((g.ncols(), g.ncols())));
            }
            fishers[k] += &curvature[k].1.clone().ok_or("no written Fisher")?;
            for c in 0..blocked.ranks[k].len() {
                let members: Vec<usize> = (0..inputs.rows).filter(|&r| masks[k][[r, c]] > 0.0 && target.scores(r)).collect();
                if members.is_empty() {
                    continue;
                }
                let xm = x.select(Axis(0), &members);
                gradients[k][c] += &g.select(Axis(0), &members).t().dot(&xm);
                reads[k][c].push(xm);
            }
        }
    }
    let batches = coded.batches.len().max(1) as f64;
    let mut out = Vec::new();
    for k in 0..blocked.ranks.len() {
        let fisher = &fishers[k] / batches;
        let mut site = Vec::new();
        for c in 0..blocked.ranks[k].len() {
            let (u, v) = blocked.factors(k, c);
            let (u, v) = balanced(u, v)?;
            // `(I − P_U) G (I − P_V)`: what no change inside the block's spans reaches.
            let project = |m: &Array2<f64>, basis: &Array2<f64>| -> Result<Array2<f64>, String> {
                if basis.nrows() == 0 {
                    return Ok(m.clone());
                }
                let q = qr(basis.t(), QrMode::Economic).map_err(|e| format!("{e:?}"))?.q.ok_or("a thin QR without Q")?;
                Ok(m - &q.dot(&q.t().dot(m)))
            };
            let outside = project(&project(&gradients[k][c], &u)?.t().to_owned(), &v)?.t().to_owned();
            let top = svd(outside.view(), false).map_err(|e| format!("{e:?}"))?;
            if top.singular_values.is_empty() || !(top.singular_values[0] > top.band) {
                site.push(None);
                continue;
            }
            let (p, q) = (top.u.column(0).to_owned(), top.vt.row(0).to_owned());
            // The Gauss–Newton step along `−p qᵀ`: `σ / Σ_on (pᵀ F p)(qᵀ x)²`.
            let pfp = p.dot(&fisher.dot(&p));
            let spread: f64 = reads[k][c].iter().map(|x| x.dot(&q).mapv(|a| a * a).sum()).sum();
            let curvature = pfp * spread;
            if !(curvature > 0.0) {
                site.push(None);
                continue;
            }
            let step = top.singular_values[0] / curvature;
            site.push(Some(((&p * -step).insert_axis(Axis(0)), q.insert_axis(Axis(0)))));
        }
        out.push(site);
    }
    Ok(out)
}

/// The second moment of each rank ≥ 2 block's balanced coordinates `z' = V'ᵀ x` ([`balanced`])
/// over the inputs where it is on.
fn member_moments(coded: &Coded<'_>, blocked: &Blocked) -> Result<Vec<Vec<Array2<f64>>>, String> {
    let masked = blocked.masked(coded)?;
    let mut readers: Vec<Vec<Option<Array2<f64>>>> = Vec::new();
    for k in 0..blocked.ranks.len() {
        let mut site = Vec::new();
        for c in 0..blocked.ranks[k].len() {
            site.push(if blocked.ranks[k][c] >= 2 {
                let (u, v) = blocked.factors(k, c);
                Some(balanced(u, v)?.1)
            } else {
                None
            });
        }
        readers.push(site);
    }
    let mut out: Vec<Vec<Array2<f64>>> =
        readers.iter().map(|site| site.iter().map(|v| v.as_ref().map_or_else(|| Array2::zeros((0, 0)), |v| Array2::zeros((v.nrows(), v.nrows())))).collect()).collect();
    for ((inputs, target), masks) in coded.batches.iter().zip(&blocked.masks) {
        let (_, trace, _) = forward(&masked, &masked.family(inputs, masks), target)?;
        for (k, moments) in out.iter_mut().enumerate() {
            let x = read_values(&trace, &masked.sites[k])?;
            for (c, moment) in moments.iter_mut().enumerate() {
                let Some(v) = &readers[k][c] else { continue };
                let members: Vec<usize> = (0..inputs.rows).filter(|&r| masks[k][[r, c]] > 0.0).collect();
                let z = x.select(Axis(0), &members).dot(&v.t());
                *moment += &z.t().dot(&z);
            }
        }
    }
    Ok(out)
}

/// The code-chosen blocks from `blocked` (module note): merge rounds until none is kept, then (when
/// `refine`) grows, shrinks and splits one block at a time, repeated until nothing changes the
/// decomposition. Returns it and its code.
pub fn fit_blocks(coded: &Coded<'_>, mut blocked: Blocked, refine: bool) -> Result<(Blocked, Bits), String> {
    blocked.price(coded)?;
    let (mut current, _) = measure(coded, &blocked)?;
    let mut refused: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
    let mut refused_refinements: BTreeSet<(usize, usize)> = BTreeSet::new();
    loop {
        let mut changed = false;
        // Merge rounds.
        loop {
            let candidates = partners(&blocked, &refused);
            let mut merges: Vec<Merge> = predict(coded, &blocked, &candidates)?.into_iter().filter(|m| m.saving > 0.0).collect();
            merges.sort_by(|x, y| y.saving.total_cmp(&x.saving));
            // Disjoint: a block in one merge per round, best first.
            let mut used = BTreeSet::new();
            merges.retain(|m| {
                let free = !used.contains(&(m.site, m.a)) && !used.contains(&(m.site, m.b));
                if free {
                    used.extend([(m.site, m.a), (m.site, m.b)]);
                }
                free
            });
            let mut kept = false;
            let mut take = merges.len();
            while take > 0 {
                let trial = apply(&blocked, &merges.iter().take(take).collect::<Vec<_>>())?;
                let (bits, _) = measure(coded, &trial)?;
                log::info!(
                    "blocks: {take} merges predicted to save {:.1} bits: {:.1} -> {:.1}",
                    merges.iter().take(take).map(|m| m.saving).sum::<f64>(),
                    current.total(),
                    bits.total()
                );
                if bits.total() < current.total() {
                    (blocked, current) = (trial, bits);
                    kept = true;
                    break;
                }
                if take == 1 {
                    refused.insert((merges[0].site, merges[0].a.min(merges[0].b), merges[0].a.max(merges[0].b)));
                    merges.remove(0);
                    take = merges.len();
                } else {
                    take /= 2;
                }
            }
            if !kept {
                break;
            }
            changed = true;
            let reselected = reselect(coded, &blocked)?;
            let (bits, _) = measure(coded, &reselected)?;
            if bits.total() < current.total() {
                (blocked, current) = (reselected, bits);
            }
        }
        // Grows, shrinks and splits, one block at a time.
        if refine {
            let moments = member_moments(coded, &blocked)?;
            let mut grows = grow_directions(coded, &blocked)?;
            let state = &blocked;
            let candidates: Vec<(usize, usize)> = (0..state.ranks.len())
                .flat_map(|k| (0..state.ranks[k].len()).map(move |c| (k, state.ids[k][c])))
                .filter(|key| !refused_refinements.contains(key))
                .collect();
            for (k, id) in candidates {
                let Some(c) = blocked.index(k, id) else { continue };
                let rank = blocked.ranks[k][c];
                let mut trials = Vec::new();
                if let Some((du, dv)) = grows[k][c].take() {
                    let (u, v) = blocked.factors(k, c);
                    let u = ndarray::concatenate(Axis(0), &[u, du.view()]).map_err(|e| e.to_string())?;
                    let v = ndarray::concatenate(Axis(0), &[v, dv.view()]).map_err(|e| e.to_string())?;
                    trials.push(("grow", replace_block(&blocked, k, c, &u, &v, &[rank + 1])?));
                }
                if rank >= 2 {
                    if let Some(shrunk) = shrink_block(&blocked, k, c)? {
                        trials.push(("shrink", shrunk));
                    }
                    trials.push(("split", reselect(coded, &split_block(&blocked, k, c, &moments[k][c])?)?));
                }
                let mut kept = false;
                for (what, mut trial) in trials {
                    trial.price(coded)?;
                    let (bits, _) = measure(coded, &trial)?;
                    log::info!("blocks: {what} of a rank-{rank} block: {:.1} -> {:.1}", current.total(), bits.total());
                    if bits.total() < current.total() {
                        (blocked, current) = (trial, bits);
                        kept = true;
                        break;
                    }
                }
                if kept {
                    changed = true;
                    break;
                }
                refused_refinements.insert((k, id));
            }
        }
        if !changed {
            return Ok((blocked, current));
        }
    }
}

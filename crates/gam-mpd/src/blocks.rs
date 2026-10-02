//! Rank-k gated subcomponents, their ranks chosen by the code (#2951).
//!
//! A site's subcomponents are blocks of its library's columns, one gate each per input
//! ([`super::masked`], "Blocks"). Which columns share a gate is decided by the code the selection
//! minimises, totalled over every coded input:
//!
//! ```text
//! library bits + rate bits + Σ_t listing_t + n Σ_t KL_t / ln 2.
//! ```
//!
//! * **Library.** A block of rank `k` on a `d_in → d_out` site is its map `U Vᵀ` up to `GL(k)`:
//!   `k (d_in + d_out − k)` reals at the caller's bits per real, `k` the map's numerical rank, so a
//!   block whose columns duplicate one another is charged once and copying a block cannot lower
//!   its cost. The library's bits are spread over the rows it serves.
//! * **Listing.** The context code of the coded sets ([`Coder`]), every rate estimated from those
//!   sets, plus each estimated rate's parameter cost `½ log₂ n` (`n` the events it is estimated
//!   from) as the rate bits: the plug-in code made a two-part code.
//!
//! The moves, each only a proposal:
//!
//! * **Merge.** Two blocks of a site that co-fire become one, their columns concatenated, so the
//!   map is unchanged: one listing where there were two and `2 k_a k_b` reals fewer. On an input
//!   where exactly one was on, the merged gate takes whichever of on and off its second-order KL
//!   bits plus listing predict cheaper. Each block proposes its most co-firing partner (largest
//!   Jaccard overlap of their inputs, ties by more shared inputs, then by the `|cos|` of their maps
//!   in the Frobenius inner product `⟨U_i V_iᵀ, U_j V_jᵀ⟩ = tr[(U_iᵀU_j)(V_jᵀV_i)]`).
//! * **Split.** A block of rank ≥ 2 becomes rank-one blocks along its principal coordinates (its
//!   columns rotated within `GL(k)` so its output columns are orthonormal and its members'
//!   coordinates uncorrelated), each on wherever the block was, and the selection then lists only
//!   what each input uses.
//!
//! A round's merges are kept only when the exact total (every input's masked forward) falls; a
//! refused set is halved by predicted saving until a single refused merge is set aside. A split is
//! tested alone, after the selection. Every kept move is followed by the selection under the new
//! counts, kept when it lowers the total again.

use super::dense::eigh;
use super::masked::{Context, Library, Masked, Site, Target, fisher, forward, mask_gradients, previous_inputs, select};
use super::operator_program::{FamilyInputs, OperatorProgram};
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, Axis, s};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// What the code is measured on: the model, its decomposed sites (in the model's node indices, as
/// [`super::masked::sites`] gives them), and the coded inputs in batches (a sequence each, or any
/// rows), each with its target.
pub struct Coded<'a> {
    pub model: &'a OperatorProgram,
    pub sites: Vec<Site>,
    pub batches: Vec<(FamilyInputs, Target)>,
    /// `n` of the code.
    pub observations: f64,
    /// Sampled-label reverse passes per Fisher diagonal.
    pub samples: usize,
    pub bits_per_real: f64,
    /// The rows the library is spread over: its bits enter the total as `library · rows / library_rows`.
    pub library_rows: f64,
}

/// A blocked decomposition: per site its library (shared between decompositions that differ
/// elsewhere) and blocks (their ranks, in column order, and stable ids), and per batch every site's
/// masks (`rows × B`).
#[derive(Clone, Debug)]
pub struct Blocked {
    pub libraries: Vec<Arc<Library>>,
    pub ranks: Vec<Vec<usize>>,
    pub ids: Vec<Vec<usize>>,
    pub masks: Vec<Vec<Array2<f64>>>,
    next_id: usize,
}

impl Blocked {
    /// Every piece its own block.
    pub fn rank_one(libraries: Vec<Library>, masks: Vec<Vec<Array2<f64>>>) -> Self {
        let ranks: Vec<Vec<usize>> = libraries.iter().map(|l| vec![1; l.v.nrows()]).collect();
        let mut next_id = 0;
        let ids = ranks
            .iter()
            .map(|r| {
                let ids = (next_id..next_id + r.len()).collect();
                next_id += r.len();
                ids
            })
            .collect();
        Self { libraries: libraries.into_iter().map(Arc::new).collect(), ranks, ids, masks, next_id }
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
}

/// The numerical rank of the block of `library`'s columns `start..start + k`: the singular values
/// of `U Vᵀ` are the roots of the eigenvalues of `G_u^{1/2} G_v G_u^{1/2}` (`G` the columns' Grams),
/// counted beyond that decomposition's band.
pub fn block_rank(library: &Library, start: usize, k: usize) -> Result<usize, String> {
    let (u, v) = (library.u.slice(s![start..start + k, ..]), library.v.slice(s![start..start + k, ..]));
    let (half, _) = roots(&u.dot(&u.t()))?;
    let m = symmetric(&half.dot(&v.dot(&v.t())).dot(&half));
    let d = eigh(m.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    Ok(d.values.iter().filter(|l| **l > d.band).count())
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

/// `(M^{1/2}, M^{-1/2})` of a symmetric positive semidefinite matrix over its eigenvalues beyond
/// the decomposition's band (zero on the rest).
fn roots(m: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let d = eigh(symmetric(m).view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let (mut half, mut inverse) = (d.vectors.clone(), d.vectors.clone());
    for (j, l) in d.values.iter().enumerate() {
        let (h, i) = if *l > d.band { (l.sqrt(), 1.0 / l.sqrt()) } else { (0.0, 0.0) };
        half.column_mut(j).mapv_inplace(|x| x * h);
        inverse.column_mut(j).mapv_inplace(|x| x * i);
    }
    Ok((half.dot(&d.vectors.t()), inverse.dot(&d.vectors.t())))
}

/// A decomposition's code, totalled over the scored rows of every batch (module note).
#[derive(Clone, Debug, Default)]
pub struct Bits {
    pub library: f64,
    pub rates: f64,
    pub listing: f64,
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
        self.library + self.rates + self.listing + self.kl
    }

    /// `(bits, KL nats, active blocks, active rank-one equivalents)` per scored row.
    pub fn per_row(&self) -> (f64, f64, f64, f64) {
        let rows = self.rows.max(1.0);
        (self.total() / rows, self.kl_nats / rows, self.active_blocks / rows, self.active_rank / rows)
    }
}

/// The plug-in context counts of every batch's masks.
fn context(coded: &Coded<'_>, blocked: &Blocked) -> Context {
    let widths: Vec<usize> = blocked.ranks.iter().map(Vec::len).collect();
    let mut context = Context::new(&widths);
    for ((inputs, _), masks) in coded.batches.iter().zip(&blocked.masks) {
        context.absorb(masks, &previous_inputs(inputs));
    }
    context
}

/// The rate bits of `context`: `½ log₂ n` per estimated rate, `n` its events (module note).
fn rate_bits(context: &Context) -> f64 {
    let fresh: f64 = context.new.iter().map(|c| c.sum()).sum();
    let mut bits = 0.0;
    for (new, was_on) in context.new.iter().zip(&context.was_on) {
        bits += new.len() as f64 * 0.5 * fresh.max(1.0).log2();
        bits += was_on.iter().filter(|n| **n > 0.0).map(|n| 0.5 * n.log2()).sum::<f64>();
    }
    bits
}

/// Every site's library bits ([`block_rank`] reals per block at `bits_per_real`), undivided.
pub fn library_bits(blocked: &Blocked, bits_per_real: f64) -> Result<f64, String> {
    let mut bits = 0.0;
    for (k, library) in blocked.libraries.iter().enumerate() {
        let (d_in, d_out) = (library.v.ncols(), library.u.ncols());
        let starts = blocked.starts(k);
        for (b, r) in blocked.ranks[k].iter().enumerate() {
            let rank = block_rank(library, starts[b], *r)?;
            bits += (rank * (d_in + d_out - rank.min(d_in + d_out))) as f64 * bits_per_real;
        }
    }
    Ok(bits)
}

/// The exact code of `blocked` (module note) and each batch's per-row KL.
pub fn measure(coded: &Coded<'_>, blocked: &Blocked) -> Result<(Bits, Vec<Array1<f64>>), String> {
    let masked = blocked.masked(coded)?;
    let context = context(coded, blocked);
    let mut bits = Bits {
        rates: rate_bits(&context),
        blocks: blocked.ranks.iter().map(Vec::len).sum(),
        pieces: blocked.ranks.iter().flatten().sum(),
        ..Bits::default()
    };
    let scale = coded.observations / std::f64::consts::LN_2;
    let mut kls = Vec::new();
    for ((inputs, target), masks) in coded.batches.iter().zip(&blocked.masks) {
        let coder = context.coder(previous_inputs(inputs));
        let listing = coder.bits(masks);
        let (kl, _, _) = forward(&masked, &masked.family(inputs, masks), target)?;
        for r in (0..inputs.rows).filter(|r| target.scores(*r)) {
            bits.rows += 1.0;
            bits.listing += listing[r];
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
    bits.library = library_bits(blocked, coded.bits_per_real)? * bits.rows / coded.library_rows.max(1.0);
    Ok((bits, kls))
}

/// Every batch's masks selected again ([`select`]) under the plug-in counts of the current ones.
pub fn reselect(coded: &Coded<'_>, blocked: &Blocked) -> Result<Blocked, String> {
    let masked = blocked.masked(coded)?;
    let context = context(coded, blocked);
    let mut out = blocked.clone();
    for (b, (inputs, target)) in coded.batches.iter().enumerate() {
        let coder = context.coder(previous_inputs(inputs));
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
    let context = context(coded, blocked);
    let scale = coded.observations / std::f64::consts::LN_2;
    let share = coded.batches.iter().map(|(_, t)| t.scored_rows() as f64).sum::<f64>() / coded.library_rows.max(1.0);
    let fresh: f64 = context.new.iter().map(|c| c.sum()).sum();
    let mut merges: Vec<Merge> = candidates
        .iter()
        .map(|&(site, a, b)| {
            let (ka, kb) = (blocked.ranks[site][a] as f64, blocked.ranks[site][b] as f64);
            let (d_in, d_out) = (blocked.libraries[site].v.ncols() as f64, blocked.libraries[site].u.ncols() as f64);
            // `2 k_a k_b` reals fewer, and one block's rates.
            let fixed = 2.0 * ka * kb * coded.bits_per_real * share * f64::from(ka + kb <= d_in.min(d_out)) + 0.5 * fresh.max(1.0).log2();
            Merge { site, a: blocked.ids[site][a], b: blocked.ids[site][b], saving: fixed, gates: Vec::new() }
        })
        .collect();
    for (batch, (inputs, target)) in coded.batches.iter().enumerate() {
        let masks = &blocked.masks[batch];
        let coder = context.coder(previous_inputs(inputs));
        let family = masked.family(inputs, masks);
        let (_, trace, cotangent) = forward(&masked, &family, target)?;
        let g = mask_gradients(&masked, &family, &trace, cotangent)?;
        let h = fisher(&masked, &family, &trace, target, coded.samples, 0xB10C + batch as u64, false)?;
        for (merge, &(k, a, b)) in merges.iter_mut().zip(candidates) {
            let (m, g, h) = (&masks[k], &g[k], &h[k].0);
            let (cost_a, cost_b) = (coder.costs[k][a], coder.costs[k][b]);
            // The merged block fires at least as often as either, so lists at most the cheaper.
            let cost_ab = cost_a.min(cost_b);
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
                        let listed = (0..m.ncols()).filter(|&c| m[[r, c]] > 0.0).count().max(2) as f64;
                        merge.saving += cost_a + cost_b - cost_ab - listed.log2();
                    }
                    (false, false) => {}
                    (true, false) | (false, true) => {
                        let (on, off) = if on_a { (b, a) } else { (a, b) };
                        let cost_on = coder.costs[k][on];
                        // Gate on: the off block joins; gate off: the on block leaves.
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
        let (mut columns, mut ranks, mut ids) = (Vec::new(), Vec::new(), Vec::new());
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
                    gate_of.push((c, Some(i)));
                }
                None => {
                    ranks.push(blocked.ranks[k][c]);
                    ids.push(blocked.ids[k][c]);
                    gate_of.push((c, None));
                }
            }
        }
        let library = &blocked.libraries[k];
        out.libraries[k] = Arc::new(Library { v: library.v.select(Axis(0), &columns), u: library.u.select(Axis(0), &columns), mean: library.mean.clone() });
        out.ranks[k] = ranks;
        out.ids[k] = ids;
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

/// `blocked` with block `c` of site `k` split into rank-one blocks along its principal coordinates
/// on its members' coordinates `z` (module note, "Split"), each on where the block was.
pub(crate) fn split_block(blocked: &Blocked, k: usize, c: usize, second_moment: &Array2<f64>) -> Result<Blocked, String> {
    let (start, rank) = (blocked.starts(k)[c], blocked.ranks[k][c]);
    let library = &blocked.libraries[k];
    let (u, v) = (library.u.slice(s![start..start + rank, ..]), library.v.slice(s![start..start + rank, ..]));
    // U = P Λ^{1/2} … : `A₁ = P Λ^{-1/2}` makes the output columns orthonormal, then `R`
    // diagonalises the members' coordinates `Λ^{1/2} Pᵀ S P Λ^{1/2}`; `U' = U A₁ R`,
    // `V' = V P Λ^{1/2} R`, so `U' V'ᵀ = U Vᵀ`. Directions beyond the Gram's band carry no map.
    let gram = eigh(symmetric(&u.dot(&u.t())).view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let kept: Vec<usize> = (0..rank).filter(|&j| gram.values[j] > gram.band).collect();
    let p = gram.vectors.select(Axis(1), &kept);
    let (mut down, mut up) = (p.clone(), p.clone());
    for (col, &j) in kept.iter().enumerate() {
        let l = gram.values[j];
        down.column_mut(col).mapv_inplace(|x| x / l.sqrt());
        up.column_mut(col).mapv_inplace(|x| x * l.sqrt());
    }
    let coordinates = symmetric(&up.t().dot(second_moment).dot(&up));
    let r = eigh(coordinates.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?.vectors;
    let (a, b) = (down.dot(&r), up.dot(&r));
    let (u_new, v_new) = (a.t().dot(&u), b.t().dot(&v));
    let parts = kept.len();
    let mut out = blocked.clone();
    let stack = |old: &Array2<f64>, new: &Array2<f64>| -> Result<Array2<f64>, String> {
        ndarray::concatenate(Axis(0), &[old.slice(s![..start, ..]), new.view(), old.slice(s![start + rank.., ..])]).map_err(|e| e.to_string())
    };
    out.libraries[k] = Arc::new(Library { v: stack(&library.v, &v_new)?, u: stack(&library.u, &u_new)?, mean: library.mean.clone() });
    out.ranks[k].splice(c..=c, std::iter::repeat_n(1, parts));
    let fresh: Vec<usize> = (out.next_id..out.next_id + parts).collect();
    out.next_id += parts;
    out.ids[k].splice(c..=c, fresh);
    for masks in out.masks.iter_mut() {
        let old = masks[k].clone();
        let column = old.column(c).insert_axis(Axis(1));
        let views: Vec<_> =
            std::iter::once(old.slice(s![.., ..c])).chain(std::iter::repeat_n(column.view(), parts)).chain(std::iter::once(old.slice(s![.., c + 1..]))).collect();
        masks[k] = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// The second moment of each block's coordinates `z` over the inputs where it is on.
fn member_moments(coded: &Coded<'_>, blocked: &Blocked) -> Result<Vec<Vec<Array2<f64>>>, String> {
    let masked = blocked.masked(coded)?;
    let mut out: Vec<Vec<Array2<f64>>> = blocked.ranks.iter().map(|r| r.iter().map(|k| Array2::zeros((*k, *k))).collect()).collect();
    for ((inputs, target), masks) in coded.batches.iter().zip(&blocked.masks) {
        let (_, trace, _) = forward(&masked, &masked.family(inputs, masks), target)?;
        for (k, moments) in out.iter_mut().enumerate() {
            let z = &trace.values[masked.z[k]];
            let starts = blocked.starts(k);
            for (c, moment) in moments.iter_mut().enumerate() {
                if blocked.ranks[k][c] < 2 {
                    continue;
                }
                let members: Vec<usize> = (0..inputs.rows).filter(|&r| masks[k][[r, c]] > 0.0).collect();
                let zc = z.slice(s![.., starts[c]..starts[c + 1]]).select(Axis(0), &members);
                *moment += &zc.t().dot(&zc);
            }
        }
    }
    Ok(out)
}

/// The code-chosen blocks from `blocked` (module note): merge rounds until none is kept, then
/// splits, repeated until neither changes the decomposition. Returns it and its code.
pub fn fit_blocks(coded: &Coded<'_>, mut blocked: Blocked, splits: bool) -> Result<(Blocked, Bits), String> {
    let (mut current, _) = measure(coded, &blocked)?;
    let mut refused: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
    let mut refused_splits: BTreeSet<(usize, usize)> = BTreeSet::new();
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
        // Splits, one at a time.
        if splits {
            let moments = member_moments(coded, &blocked)?;
            let state = &blocked;
            let candidates: Vec<(usize, usize)> = (0..state.ranks.len())
                .flat_map(|k| (0..state.ranks[k].len()).filter(move |&c| state.ranks[k][c] >= 2).map(move |c| (k, state.ids[k][c])))
                .filter(|key| !refused_splits.contains(key))
                .collect();
            for (k, id) in candidates {
                let Some(c) = blocked.index(k, id) else { continue };
                let trial = reselect(coded, &split_block(&blocked, k, c, &moments[k][c])?)?;
                let (bits, _) = measure(coded, &trial)?;
                log::info!("blocks: split of a rank-{} block: {:.1} -> {:.1}", blocked.ranks[k][c], current.total(), bits.total());
                if bits.total() < current.total() {
                    (blocked, current) = (trial, bits);
                    changed = true;
                    break;
                }
                refused_splits.insert((k, id));
            }
        }
        if !changed {
            return Ok((blocked, current));
        }
    }
}

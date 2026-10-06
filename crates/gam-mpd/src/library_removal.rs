//! The removal search of a library explanation (#2951): which prior groups a converged fit
//! (`library_mdl`) deletes, each deletion accepted only when it does not increase the code length
//! `F` on the fit's fixed collection of experiments.
//!
//! # Groups without effect
//!
//! A group has no effect on any experiment when every one of its parameters multiplies an input
//! that is exactly zero or writes a value that nothing downstream reads: a rotary plane of a head
//! whose value coordinates are all removed, the gate of an MLP function whose output is removed.
//! The analysis runs on the explanation's flat program through its final normed stream. A removed
//! group's entries, and a fixed operator's zero entries, are exactly zero under every weight
//! sample, so the facts below hold for every sample of the posterior, not only its mean. For every
//! coordinate of every node it finds the set `Z` of single groups whose removal makes that
//! coordinate identically zero, and the set `U` of single groups whose removal leaves it read by
//! nothing that reaches the output, or every group when the coordinate already is zero or unread:
//!
//! * forwards, a sum `y_r = Σ_c A_rc x_c + b_r` is zero once every nonzero term is,
//!   `Z(y_r) = Z(b_r) ∩ ⋂_c ({G(A_rc)} ∪ Z(x_c))`; a product is zero once either factor is; a
//!   pointwise law with `φ(0) = 0`, a norm, a head's output coordinate (its value's) keep their
//!   input's zeros;
//! * backwards, an input coordinate is unread once every term reading it is removed or writes an
//!   unread value, `U(x_c) = ⋂_r ({G(A_rc)} ∪ U(y_r))`; a factor of a product once the other factor
//!   is zero or the product unread; a head's query and key once every value coordinate is zero or
//!   unread, and a rotary plane of the query once the key's plane is zero (and conversely); a norm's
//!   input once every output it scales is zero or unread.
//!
//! A block's read is patched, `x' = (I − P) x + P s` with `s` the same read on the source sequence
//! and `P` spanning directions that mix the read's coordinates, so its coordinates are taken as
//! one in both directions: a coordinate of the patched read is zero only once every coordinate of
//! the read is (a coordinate zero in `x` and in `s` can be nonzero in `x'`), and the read is unread
//! only once every coordinate is.
//!
//! A parameter `A_rc` has no effect after removing the groups in `Z(x_c) ∪ U(y_r)`, and a group is
//! killed by the groups in the intersection of that set over its parameters. A group killed by
//! every group already has no effect: it is removed exactly, first. Hybrids replace whole blocks
//! of `P` by `M`'s, and the residual stream every block writes is read through the next norm, so
//! the facts hold in every experiment.
//!
//! # Proposals
//!
//! A unit is a group with the groups its removal kills, transitively (removals only add zeros):
//! the gate, up direction and output of an MLP function form one unit, since removing any of them
//! silences the function. Each group's change of the expected data term when its entries become
//! exactly zero is estimated to second order about the posterior mean,
//! `−g_G · μ_G + ½ μ_Gᵀ H μ_G − ½ Σ_{j∈G} H_jj σ_j²` ([`Posterior::removal_data`]), with `g` the data
//! term's gradient at the mean and `H` its Gauss–Newton matrix, both measured on the fixed collection
//! in the same passes ([`Curvature`], `library_mdl`'s module note); the description falls by the
//! group's cost (`KL_G`, its variance's precision and scale) and the code of which groups are active
//! changes with their count. A unit's prediction is the mean of its roots' data estimates (each
//! root's removal silences the same functions; for an MLP function under compensation, times the
//! share of its activations' energy the surviving functions do not reproduce,
//! `Compensation::unexplained`) minus the cost of all its groups, plus the change of the subset
//! code when it alone is removed. Deletion is a finite step, so the prediction is a
//! proposal order and nothing more: every unit is ranked by it, most negative first, and only the
//! exact evaluation below decides.
//!
//! # The search
//!
//! The ranked units are taken in segments, each on top of the removals accepted so far. A segment
//! starts with the predicted-safe set, every remaining unit predicted to lower `F`, proposed whole;
//! on rejection the proposal halves (the first `n/2`, `n/4`, … units) until one is accepted, then
//! bisection between the longest accepted and the shortest rejected prefix. The longest accepted
//! prefix is removed; the unit after it is the one the segment ended on, its own effect being the
//! difference of the two prefixes (a single-unit test), and it is kept; the next segment starts
//! after it, again with the whole remaining safe set. A segment over `n` safe units costs at most
//! `log2 n + 3` evaluations, one when the whole set is accepted. Each proposal also removes the
//! groups it leaves without effect. Once no unit predicted to lower `F` remains, the rest are
//! proposed whole and then by prefixes of length 1, 3, 7, … until one is rejected (with bisection
//! as above); when neither the whole nor the first alone is accepted the round ends, and the units
//! after the first are counted as tested only jointly (`Removal::untested`).
//!
//! Every evaluation is `F` on the fixed collection at one common weight sample per batch, with the
//! MLPs that lose functions compensated by least squares when compensation is on
//! (`library_compensation`). An accepted proposal lowers that realization of the sampled objective;
//! its expectation over the posterior is estimated, not bounded, by it, and a proposal chosen on the
//! same draws can fit their particulars. A round that accepts nothing found no removal that lowers
//! the realized `F`; it does not show that none exists.
//!
//! Every proposal and its outcome are written as one JSON line to the round's log, with every
//! active group's posterior summaries and every unit's prediction at the start of the round.

use crate::{
    interchange,
    library_compensation::Compensation,
    library_mdl::{Curvature, Explanation, Posterior, Removal},
    operator_program::{Interface, Law, Node, OperatorBody, OperatorProgram},
    resident_causal_fit::fixed_head_target::Head,
    run_check::LayerNodes,
};
use ndarray::Array2;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    f64::consts::LN_2,
    io::Write,
    path::Path,
    time::Instant,
};

fn error(e: impl std::fmt::Display) -> String {
    format!("library removal: {e}")
}

/// The single groups whose removal makes a quantity identically zero (or unread), ascending, or
/// every group when it already is.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kill {
    All,
    Of(Vec<u32>),
}

impl Kill {
    fn none() -> Self {
        Self::Of(Vec::new())
    }

    fn is_none(&self) -> bool {
        matches!(self, Self::Of(v) if v.is_empty())
    }

    fn contains(&self, g: u32) -> bool {
        match self {
            Self::All => true,
            Self::Of(v) => v.binary_search(&g).is_ok(),
        }
    }

    fn union(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::All, _) | (_, Self::All) => Self::All,
            (Self::Of(a), Self::Of(b)) => {
                let mut out: Vec<u32> = a.iter().chain(b).copied().collect();
                out.sort_unstable();
                out.dedup();
                Self::Of(out)
            }
        }
    }

    /// `self ∩ (a ∪ b ∪ {extra})`.
    fn meet(&mut self, a: &Self, b: &Self, extra: Option<u32>) {
        if matches!(a, Self::All) || matches!(b, Self::All) {
            return;
        }
        match self {
            Self::All => {
                let mut joined = a.union(b);
                if let (Self::Of(v), Some(g)) = (&mut joined, extra)
                    && let Err(at) = v.binary_search(&g)
                {
                    v.insert(at, g);
                }
                *self = joined;
            }
            Self::Of(v) => v.retain(|e| Some(*e) == extra || a.contains(*e) || b.contains(*e)),
        }
    }
}

/// `⋂_i (A_i ∪ B_i)` over the coordinates of `sets` (`A`) and, when given, `also` (`B`); without
/// `also` it is `⋂_i A_i` (the union's neutral element, the empty set, stands in for each `B_i`).
fn intersection(sets: &[Kill], also: Option<&[Kill]>) -> Kill {
    let none = Kill::none();
    let mut k = Kill::All;
    for (i, a) in sets.iter().enumerate() {
        k.meet(a, also.map_or(&none, |b| &b[i]), None);
        if k.is_none() {
            break;
        }
    }
    k
}

/// The zero sets of a block's read after its patch: a patch `x' = (I − P) x + P s` replaces the
/// read by a mix of every coordinate of the base's read and the source's (the same node of the same
/// program on another sequence) through directions that span the coordinates, so a coordinate is
/// zero only once every coordinate is (`z` the read's zero sets before the patch).
fn patched(z: &[Kill]) -> Vec<Kill> {
    vec![intersection(z, None); z.len()]
}

/// An operator's entries: each entry's group (a library operator), or which entries are nonzero (a
/// fixed one).
enum Entries {
    Groups(Array2<u32>),
    Dense(Array2<bool>),
    Diagonal(Vec<bool>),
}

/// How a node reads a library operator.
#[derive(Clone, Copy, Debug)]
enum Use {
    /// A term `A x` of an affine node: the node and `x`.
    Term { node: usize, input: usize },
    /// The bias of an affine node, or a constant node.
    Column { node: usize },
    /// `x A` (the operator transposed): the node and `x`.
    Transposed { node: usize, input: usize },
}

/// The explanation's flat program through its final normed stream, with each operator's entries,
/// for the analysis of which groups have an effect (module note).
pub struct Structure {
    flat: OperatorProgram,
    interfaces: Vec<Interface>,
    widths: Vec<usize>,
    hidden: usize,
    reads: BTreeSet<usize>,
    entries: BTreeMap<usize, Entries>,
    /// Per group, its parameters: per cell its operator, rows and columns.
    groups: Vec<Vec<(usize, Vec<usize>, std::ops::Range<usize>)>>,
    /// Per library operator, the nodes reading it.
    uses: BTreeMap<usize, Vec<Use>>,
    /// Per law group of each pointwise node, its law, by coordinate.
    laws: BTreeMap<usize, Vec<Law>>,
}

impl Structure {
    pub fn new(explanation: &Explanation) -> Result<Self, String> {
        let sites: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let (flat, _, reads) = interchange::sites(&explanation.artifact, &sites)?;
        let hidden = Head::of(&flat)?.hidden;
        let interfaces = flat.interfaces().map_err(error)?;
        let widths: Vec<usize> = interfaces.iter().map(|i| i.width()).collect();
        let trainable: BTreeSet<usize> = explanation.trainable.iter().copied().collect();
        let mut entries = BTreeMap::new();
        for &op in &trainable {
            let operator = &flat.operators[op];
            entries.insert(op, Entries::Groups(Array2::from_elem((operator.rows.width(), operator.cols.width()), u32::MAX)));
        }
        let groups: Vec<Vec<(usize, Vec<usize>, std::ops::Range<usize>)>> =
            explanation.groups.iter().map(|g| g.cells.iter().map(|c| (c.operator, c.rows.clone(), c.cols.clone())).collect()).collect();
        for (g, cells) in groups.iter().enumerate() {
            for (op, rows, cols) in cells {
                let Some(Entries::Groups(membership)) = entries.get_mut(op) else {
                    return Err(error(format!("{}: operator {op} is not a library operator", explanation.groups[g].name)));
                };
                for &r in rows {
                    for c in cols.clone() {
                        *membership.get_mut((r, c)).ok_or_else(|| error(format!("{}: an entry outside its operator", explanation.groups[g].name)))? = g as u32;
                    }
                }
            }
        }
        if entries.values().any(|e| matches!(e, Entries::Groups(m) if m.iter().any(|g| *g == u32::MAX))) {
            return Err(error("a library entry in no group"));
        }
        let mut uses: BTreeMap<usize, Vec<Use>> = BTreeMap::new();
        let mut laws = BTreeMap::new();
        for (n, node) in flat.nodes.iter().enumerate().take(hidden + 1) {
            let mut fixed = |op: usize| {
                if entries.contains_key(&op) {
                    return;
                }
                let operator = &flat.operators[op];
                let pattern = match &operator.body {
                    OperatorBody::Identity => Entries::Diagonal(vec![true; operator.rows.width()]),
                    OperatorBody::Diagonal { values, .. } => Entries::Diagonal(values.iter().map(|v| *v != 0.0).collect()),
                    _ => Entries::Dense(operator.matrix_cow().mapv(|v| v != 0.0)),
                };
                entries.insert(op, pattern);
            };
            match node {
                Node::Affine { terms, bias } => {
                    for (input, op) in terms {
                        fixed(*op);
                        if trainable.contains(op) {
                            uses.entry(*op).or_default().push(Use::Term { node: n, input: *input });
                        }
                    }
                    if let Some(b) = bias {
                        fixed(*b);
                        if trainable.contains(b) {
                            uses.entry(*b).or_default().push(Use::Column { node: n });
                        }
                    }
                }
                Node::Constant { operator } => {
                    fixed(*operator);
                    if trainable.contains(operator) {
                        uses.entry(*operator).or_default().push(Use::Column { node: n });
                    }
                }
                Node::Transposed { input, operator } => {
                    fixed(*operator);
                    if trainable.contains(operator) {
                        uses.entry(*operator).or_default().push(Use::Transposed { node: n, input: *input });
                    }
                }
                Node::Pointwise { input, laws: per_group } => {
                    let interface = &interfaces[*input];
                    let mut by_coordinate = vec![Law::Identity; widths[*input]];
                    for (g, law) in per_group.iter().enumerate() {
                        for i in interface.range(g) {
                            by_coordinate[i] = *law;
                        }
                    }
                    laws.insert(n, by_coordinate);
                }
                Node::Param { .. } | Node::Call { .. } => return Err(error("a call in the flat program")),
                Node::Feature { .. }
                | Node::Raw { .. }
                | Node::Bilinear { .. }
                | Node::Softmax { .. }
                | Node::Mix { .. }
                | Node::Hadamard { .. }
                | Node::Readout { .. }
                | Node::Outer { .. }
                | Node::Concat { .. }
                | Node::Gain { .. }
                | Node::Attend { .. }
                | Node::RmsNorm { .. } => {}
            }
        }
        if let Some(op) = trainable.iter().find(|op| !uses.contains_key(op)) {
            return Err(error(format!("library operator {} is read by no node before the final normed stream", flat.operators[*op].name)));
        }
        Ok(Self { flat, interfaces, widths, hidden, reads: reads.into_iter().collect(), entries, groups, uses, laws })
    }

    /// Entry `(r, c)` of operator `op`: `None` when it is exactly zero, else its group (none for a
    /// fixed entry).
    fn entry(&self, op: usize, r: usize, c: usize, active: &[bool]) -> Option<Option<u32>> {
        match &self.entries[&op] {
            Entries::Groups(m) => {
                let g = m[[r, c]];
                active[g as usize].then_some(Some(g))
            }
            Entries::Dense(nonzero) => nonzero[[r, c]].then_some(None),
            Entries::Diagonal(d) => (r == c && d[r]).then_some(None),
        }
    }

    /// The kill of a bias or constant entry `(r, 0)`.
    fn column(&self, op: usize, r: usize, active: &[bool]) -> Kill {
        match self.entry(op, r, 0, active) {
            None => Kill::All,
            Some(None) => Kill::none(),
            Some(Some(g)) => Kill::Of(vec![g]),
        }
    }

    /// `f(c, group)` over the nonzero entries of row `r` of `op`, while it returns true.
    fn row(&self, op: usize, r: usize, active: &[bool], mut f: impl FnMut(usize, Option<u32>) -> bool) {
        match &self.entries[&op] {
            Entries::Groups(m) => {
                for (c, g) in m.row(r).iter().enumerate() {
                    if active[*g as usize] && !f(c, Some(*g)) {
                        return;
                    }
                }
            }
            Entries::Dense(nonzero) => {
                for (c, n) in nonzero.row(r).iter().enumerate() {
                    if *n && !f(c, None) {
                        return;
                    }
                }
            }
            Entries::Diagonal(d) => {
                if d[r] {
                    f(r, None);
                }
            }
        }
    }

    /// `f(r, group)` over the nonzero entries of column `c` of `op`, while it returns true.
    fn col(&self, op: usize, c: usize, active: &[bool], mut f: impl FnMut(usize, Option<u32>) -> bool) {
        match &self.entries[&op] {
            Entries::Groups(m) => {
                for (r, g) in m.column(c).iter().enumerate() {
                    if active[*g as usize] && !f(r, Some(*g)) {
                        return;
                    }
                }
            }
            Entries::Dense(nonzero) => {
                for (r, n) in nonzero.column(c).iter().enumerate() {
                    if *n && !f(r, None) {
                        return;
                    }
                }
            }
            Entries::Diagonal(d) => {
                if d[c] {
                    f(c, None);
                }
            }
        }
    }

    /// Per coordinate of every node through the final normed stream, `Z` (module note).
    fn zeros(&self, active: &[bool]) -> Result<Vec<Vec<Kill>>, String> {
        let mut zero: Vec<Vec<Kill>> = Vec::with_capacity(self.hidden + 1);
        for (n, node) in self.flat.nodes.iter().enumerate().take(self.hidden + 1) {
            let w = self.widths[n];
            let z: Vec<Kill> = match node {
                Node::Feature { .. } | Node::Raw { .. } | Node::Softmax { .. } | Node::Readout { .. } => vec![Kill::none(); w],
                Node::Constant { operator } => (0..w).map(|r| self.column(*operator, r, active)).collect(),
                Node::Affine { terms, bias } => (0..w)
                    .into_par_iter()
                    .map(|r| {
                        let mut k = bias.map_or(Kill::All, |b| self.column(b, r, active));
                        for (x, op) in terms {
                            if k.is_none() {
                                break;
                            }
                            self.row(*op, r, active, |c, g| {
                                k.meet(&zero[*x][c], &Kill::none(), g);
                                !k.is_none()
                            });
                        }
                        k
                    })
                    .collect(),
                Node::Transposed { input, operator } => (0..w)
                    .into_par_iter()
                    .map(|c| {
                        let mut k = Kill::All;
                        self.col(*operator, c, active, |r, g| {
                            k.meet(&zero[*input][r], &Kill::none(), g);
                            !k.is_none()
                        });
                        k
                    })
                    .collect(),
                Node::Bilinear { left, right, .. } => {
                    let mut k = Kill::All;
                    for (l, r) in zero[*left].iter().zip(&zero[*right]) {
                        k.meet(l, r, None);
                    }
                    vec![k]
                }
                Node::Mix { weights, payloads } => (0..w)
                    .map(|i| {
                        let mut k = Kill::All;
                        for (c, p) in payloads {
                            k.meet(&zero[*weights][*c], &zero[*p][i], None);
                        }
                        k
                    })
                    .collect(),
                Node::Pointwise { input, .. } => self.laws[&n]
                    .iter()
                    .zip(&zero[*input])
                    .map(|(law, z)| match law {
                        Law::Zero => Kill::All,
                        law if law.apply(0.0) == 0.0 => z.clone(),
                        _ => Kill::none(),
                    })
                    .collect(),
                Node::Hadamard { left, right } => zero[*left].iter().zip(&zero[*right]).map(|(l, r)| l.union(r)).collect(),
                Node::Outer { left, right } => {
                    let mut out = Vec::with_capacity(w);
                    for (i, j) in self.outer(*left, *right)? {
                        out.push(zero[*left][i].union(&zero[*right][j]));
                    }
                    out
                }
                Node::Concat { parts } => parts.iter().flat_map(|p| zero[*p].iter().cloned()).collect(),
                Node::Gain { input, .. } | Node::RmsNorm { input, .. } => zero[*input].clone(),
                Node::Attend { value, .. } => zero[*value].clone(),
                Node::Param { .. } | Node::Call { .. } => return Err(error("a call in the flat program")),
            };
            if z.len() != w {
                return Err(error(format!("node {n}: {} zero sets for width {w}", z.len())));
            }
            zero.push(if self.reads.contains(&n) { patched(&z) } else { z });
        }
        Ok(zero)
    }

    /// The coordinate pairs `(i, j)` of an outer product's output, in its order.
    fn outer(&self, left: usize, right: usize) -> Result<Vec<(usize, usize)>, String> {
        let (l, r) = (&self.interfaces[left], &self.interfaces[right]);
        let mut out = Vec::with_capacity(l.width() * r.width());
        for g1 in 0..l.group_count() {
            for g2 in 0..r.group_count() {
                for i in l.range(g1) {
                    for j in r.range(g2) {
                        out.push((i, j));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Per coordinate of every node through the final normed stream, `U` given the zeros `zero`
    /// (module note).
    fn unread(&self, active: &[bool], zero: &[Vec<Kill>]) -> Result<Vec<Vec<Kill>>, String> {
        let mut unread: Vec<Vec<Kill>> = self.widths[..=self.hidden].iter().map(|w| vec![Kill::All; *w]).collect();
        unread[self.hidden] = vec![Kill::none(); self.widths[self.hidden]];
        let every = intersection;
        for n in (0..=self.hidden).rev() {
            if self.reads.contains(&n) {
                let k = every(&unread[n], None);
                unread[n] = vec![k; self.widths[n]];
            }
            let u = std::mem::take(&mut unread[n]);
            if u.iter().all(|k| *k == Kill::All) {
                unread[n] = u;
                continue;
            }
            match &self.flat.nodes[n] {
                Node::Feature { .. } | Node::Raw { .. } | Node::Constant { .. } => {}
                Node::Affine { terms, .. } => {
                    for (x, op) in terms {
                        for (r, ur) in u.iter().enumerate() {
                            if *ur == Kill::All {
                                continue;
                            }
                            let target = &mut unread[*x];
                            self.row(*op, r, active, |c, g| {
                                target[c].meet(ur, &Kill::none(), g);
                                true
                            });
                        }
                    }
                }
                Node::Transposed { input, operator } => {
                    for (c, uc) in u.iter().enumerate() {
                        if *uc == Kill::All {
                            continue;
                        }
                        let target = &mut unread[*input];
                        self.col(*operator, c, active, |r, g| {
                            target[r].meet(uc, &Kill::none(), g);
                            true
                        });
                    }
                }
                Node::Bilinear { left, right, .. } => {
                    for i in 0..self.widths[*left] {
                        let (zl, zr) = (zero[*left][i].clone(), zero[*right][i].clone());
                        unread[*left][i].meet(&u[0], &zr, None);
                        unread[*right][i].meet(&u[0], &zl, None);
                    }
                }
                Node::Softmax { scores } => {
                    let k = every(&u, None);
                    for s in scores {
                        unread[*s][0].meet(&k, &Kill::none(), None);
                    }
                }
                Node::Readout { input, .. } => {
                    let k = every(&u, None);
                    for x in unread[*input].iter_mut() {
                        x.meet(&k, &Kill::none(), None);
                    }
                }
                Node::Mix { weights, payloads } => {
                    for (c, p) in payloads {
                        let k = every(&u, Some(&zero[*p]));
                        unread[*weights][*c].meet(&k, &Kill::none(), None);
                        for (i, ui) in u.iter().enumerate() {
                            unread[*p][i].meet(ui, &zero[*weights][*c], None);
                        }
                    }
                }
                Node::Pointwise { input, .. } => {
                    for (i, (ui, law)) in u.iter().zip(&self.laws[&n]).enumerate() {
                        if *law != Law::Zero {
                            unread[*input][i].meet(ui, &Kill::none(), None);
                        }
                    }
                }
                Node::Hadamard { left, right } => {
                    for (i, ui) in u.iter().enumerate() {
                        let (zl, zr) = (zero[*left][i].clone(), zero[*right][i].clone());
                        unread[*left][i].meet(ui, &zr, None);
                        unread[*right][i].meet(ui, &zl, None);
                    }
                }
                Node::Outer { left, right } => {
                    for (o, (i, j)) in self.outer(*left, *right)?.into_iter().enumerate() {
                        let (zl, zr) = (zero[*left][i].clone(), zero[*right][j].clone());
                        unread[*left][i].meet(&u[o], &zr, None);
                        unread[*right][j].meet(&u[o], &zl, None);
                    }
                }
                Node::Concat { parts } => {
                    let mut offset = 0;
                    for p in parts {
                        for i in 0..self.widths[*p] {
                            unread[*p][i].meet(&u[offset + i], &Kill::none(), None);
                        }
                        offset += self.widths[*p];
                    }
                }
                Node::Gain { input, .. } => {
                    for (i, ui) in u.iter().enumerate() {
                        unread[*input][i].meet(ui, &Kill::none(), None);
                    }
                }
                Node::RmsNorm { input, .. } => {
                    // Each input coordinate scales every output through the norm, except outputs
                    // whose own input is zero.
                    let k = every(&u, Some(&zero[*input]));
                    for (i, ui) in u.iter().enumerate() {
                        unread[*input][i].meet(ui, &Kill::none(), None);
                        unread[*input][i].meet(&k, &Kill::none(), None);
                    }
                }
                Node::Attend { query, key, value, rotary, .. } => {
                    for (i, ui) in u.iter().enumerate() {
                        unread[*value][i].meet(ui, &Kill::none(), None);
                    }
                    // The scores reach the output through every value coordinate.
                    let scores = every(&u, Some(&zero[*value]));
                    let width = self.widths[*query];
                    let mut partners: Vec<Vec<usize>> = (0..width).map(|i| vec![i]).collect();
                    if let Some(rotary) = rotary {
                        for (a, b) in rotary.pairs() {
                            if a < width && b < width {
                                partners[a] = vec![a, b];
                                partners[b] = vec![a, b];
                            }
                        }
                    }
                    for (i, plane) in partners.iter().enumerate() {
                        let (mut silent_key, mut silent_query) = (Kill::All, Kill::All);
                        for p in plane {
                            silent_key.meet(&zero[*key][*p], &Kill::none(), None);
                            silent_query.meet(&zero[*query][*p], &Kill::none(), None);
                        }
                        unread[*query][i].meet(&scores, &silent_key, None);
                        unread[*key][i].meet(&scores, &silent_query, None);
                    }
                }
                Node::Param { .. } | Node::Call { .. } => return Err(error("a call in the flat program")),
            }
            unread[n] = u;
        }
        Ok(unread)
    }

    /// Per group, the single groups whose removal leaves it without effect on any experiment, given
    /// the groups `active` (every group for an inactive one, and for one already without effect).
    fn kills(&self, active: &[bool]) -> Result<Vec<Kill>, String> {
        if active.len() != self.groups.len() {
            return Err(error("one activity per group required"));
        }
        let zero = self.zeros(active)?;
        let unread = self.unread(active, &zero)?;
        Ok(self
            .groups
            .par_iter()
            .enumerate()
            .map(|(g, cells)| {
                if !active[g] {
                    return Kill::All;
                }
                let mut k = Kill::All;
                'cells: for (op, rows, cols) in cells {
                    for u in self.uses.get(op).map_or(&[][..], Vec::as_slice) {
                        for &r in rows {
                            for c in cols.clone() {
                                match *u {
                                    Use::Term { node, input } => k.meet(&zero[input][c], &unread[node][r], None),
                                    Use::Column { node } => k.meet(&unread[node][r], &Kill::none(), None),
                                    Use::Transposed { node, input } => k.meet(&zero[input][r], &unread[node][c], None),
                                }
                                if k.is_none() {
                                    break 'cells;
                                }
                            }
                        }
                    }
                }
                k
            })
            .collect())
    }

    /// The active groups without effect on any experiment once the groups `active` are the
    /// explanation's, with those that become so after their removal, ascending.
    pub fn dead(&self, active: &[bool]) -> Result<Vec<usize>, String> {
        let mut active = active.to_vec();
        let mut dead = Vec::new();
        loop {
            let kills = self.kills(&active)?;
            let new: Vec<usize> = (0..active.len()).filter(|g| active[*g] && kills[*g] == Kill::All).collect();
            if new.is_empty() {
                break;
            }
            for g in new {
                active[g] = false;
                dead.push(g);
            }
        }
        dead.sort_unstable();
        Ok(dead)
    }

    /// The units of the active groups (module note): each with the groups whose removal makes all
    /// of it inactive or without effect (its roots), ascending; every active group is in a unit.
    pub fn units(&self, active: &[bool]) -> Result<Vec<Unit>, String> {
        let kills = self.kills(active)?;
        let mut killed_by: Vec<Vec<usize>> = vec![Vec::new(); active.len()];
        for (h, k) in kills.iter().enumerate() {
            if let (true, Kill::Of(by)) = (active[h], k) {
                for g in by {
                    if *g as usize != h {
                        killed_by[*g as usize].push(h);
                    }
                }
            }
        }
        let mut units: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
        for g in (0..active.len()).filter(|g| active[*g]) {
            let mut members = BTreeSet::from([g]);
            let mut frontier = vec![g];
            while let Some(f) = frontier.pop() {
                for &h in &killed_by[f] {
                    if members.insert(h) {
                        frontier.push(h);
                    }
                }
            }
            units.entry(members.into_iter().collect()).or_default().push(g);
        }
        Ok(units.into_iter().map(|(groups, roots)| Unit { groups, roots, data: 0.0, predicted: 0.0 }).collect())
    }
}

/// A unit of removal (module note): its groups, the groups whose removal alone silences it, the
/// predicted change of the data term and of `F` when it alone is removed, in nats.
#[derive(Clone, Debug)]
pub struct Unit {
    pub groups: Vec<usize>,
    pub roots: Vec<usize>,
    pub data: f64,
    pub predicted: f64,
}

/// Per group, the posterior's summaries: `Σ μ̃²/σ̃²` along its operators' rotated axes, where the
/// deviations `σ̃` are (`μᵀ Σ⁻¹ μ` over the group, the data estimate's twice), the mean `|μ|` and
/// the root mean square `σ` over its active parameters.
struct Summary {
    signal: f64,
    mean_abs: f64,
    rms_sd: f64,
}

fn summaries(explanation: &Explanation, posterior: &Posterior) -> Result<Vec<Summary>, String> {
    let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
    let rotated: Vec<_> = (0..posterior.mean.len()).map(|i| posterior.rotated_mean(i)).collect();
    explanation
        .groups
        .par_iter()
        .map(|group| {
            let (mut signal, mut abs, mut square, mut count) = (0.0, 0.0, 0.0, 0.0);
            for cell in &group.cells {
                let i = *position.get(&cell.operator).ok_or_else(|| error(format!("{}: not a library operator", group.name)))?;
                for &r in &cell.rows {
                    for c in cell.cols.clone() {
                        let (mu, s) = (posterior.mean[i][[r, c]], posterior.log_sd[i][[r, c]]);
                        if s == f64::NEG_INFINITY {
                            continue;
                        }
                        let variance = (2.0 * s).exp();
                        signal += rotated[i][[r, c]].powi(2) / variance;
                        abs += mu.abs();
                        square += variance;
                        count += 1.0;
                    }
                }
            }
            let count: f64 = if count > 0.0 { count } else { 1.0 };
            Ok(Summary { signal, mean_abs: abs / count, rms_sd: (square / count).sqrt() })
        })
        .collect()
}

/// The layer a library group belongs to, from its name (`library.l{l}...`).
fn layer(name: &str) -> Option<usize> {
    let digits: String = name.strip_prefix("library.l")?.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The nats of the code of which `k` of `n` groups are active (`library_mdl`'s module note).
fn subset_nats(n: usize, k: usize) -> Result<f64, String> {
    crate::codec::subset_code_len_bits(n, k).map(|bits| bits as f64 * LN_2).map_err(error)
}

/// The round's log: one JSON line per record, appended.
struct Journal {
    file: Option<std::io::BufWriter<std::fs::File>>,
}

impl Journal {
    fn open(path: Option<&Path>) -> Result<Self, String> {
        let file = match path {
            Some(path) => Some(std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(error)?)),
            None => None,
        };
        Ok(Self { file })
    }

    fn write(&mut self, record: Value) -> Result<(), String> {
        let Some(file) = &mut self.file else { return Ok(()) };
        serde_json::to_writer(&mut *file, &record).map_err(error)?;
        file.write_all(b"\n").map_err(error)?;
        file.flush().map_err(error)
    }
}

/// The proposals' search (module note).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Search {
    /// Dead groups first, then ranked units in galloping segments.
    Ranked,
    /// The previous search, kept to compare against: prefixes of the active groups in increasing
    /// `KL_G`, the longest bisection finds that does not increase `F`.
    Prefix,
}

/// One removal round on `posterior` (module note) with `objective` the exact `F` in nats of a
/// trial posterior on the fixed collection, compensated by `compensation` when given, logged to
/// `log` when given. `posterior` becomes the accepted one.
pub fn round(
    search: Search,
    explanation: &Explanation,
    posterior: &mut Posterior,
    compensation: Option<&Compensation>,
    curvature: &Curvature,
    objective: &mut dyn FnMut(&Posterior) -> Result<f64, String>,
    log: Option<&Path>,
) -> Result<Removal, String> {
    let started = Instant::now();
    let structure = Structure::new(explanation)?;
    let mut journal = Journal::open(log)?;
    let candidates = posterior.active.iter().filter(|a| **a).count();
    let before = objective(posterior)?;
    let mut current = before;
    let mut evaluations: Vec<(usize, f64)> = Vec::new();
    // The groups removed as without effect, and each rejected unit's first group with its own
    // effect in bits.
    let (mut dead_removed, mut singles, mut untested) = (0, Vec::new(), 0);
    let trial = |posterior: &Posterior, groups: &[usize]| -> Result<Posterior, String> {
        match compensation {
            Some(c) => c.proposal(posterior, groups),
            None => {
                let mut t = posterior.clone();
                t.remove(groups);
                Ok(t)
            }
        }
    };
    let names = |groups: &[usize]| -> Vec<usize> { groups.iter().filter_map(|g| layer(&explanation.groups[*g].name)).collect::<BTreeSet<_>>().into_iter().collect() };
    match search {
        Search::Ranked => {
            // Groups without effect, removed exactly.
            let dead = structure.dead(&posterior.active)?;
            if !dead.is_empty() {
                let timed = Instant::now();
                let mut removed = posterior.clone();
                removed.remove(&dead);
                // The data term cannot change: the prediction is the description's change.
                let predicted = removed.description() - posterior.description();
                let change = objective(&removed)? - current;
                let accepted = change <= 0.0;
                journal.write(json!({
                    "event": "proposal", "kind": "dead", "groups": dead, "layers": names(&dead),
                    "predicted_bits": predicted / LN_2, "measured_bits": change / LN_2,
                    "accepted": accepted, "seconds": timed.elapsed().as_secs_f64(),
                }))?;
                evaluations.push((dead.len(), change / LN_2));
                if accepted {
                    *posterior = removed;
                    current += change;
                    dead_removed = dead.len();
                }
            }
            let costs = posterior.costs();
            let divergences = posterior.divergences();
            let summary = summaries(explanation, posterior)?;
            let data_rise = posterior.removal_data(curvature);
            let mut units = structure.units(&posterior.active)?;
            let (total, active) = (posterior.active.len(), posterior.active.iter().filter(|a| **a).count());
            let subset = subset_nats(total, active)?;
            // The subset code's change on removing `m` of the groups now active.
            let mut fewer: BTreeMap<usize, f64> = BTreeMap::new();
            let mut subset_change = |m: usize| -> Result<f64, String> {
                if let Some(c) = fewer.get(&m) {
                    return Ok(*c);
                }
                let c = subset_nats(total, active.checked_sub(m).ok_or_else(|| error("more groups removed than active"))?)? - subset;
                fewer.insert(m, c);
                Ok(c)
            };
            // Compensation leaves of a function's data rise only the share of its activations the
            // surviving functions do not reproduce.
            let mut share = vec![1.0; posterior.active.len()];
            if let Some(c) = compensation {
                for (groups, s) in c.unexplained(posterior)? {
                    for g in groups {
                        share[g] = s;
                    }
                }
            }
            for unit in &mut units {
                unit.data = unit.roots.iter().map(|g| data_rise[*g] * share[*g]).sum::<f64>() / unit.roots.len() as f64;
                unit.predicted = unit.data - unit.groups.iter().map(|g| costs[*g]).sum::<f64>() + subset_change(unit.groups.len())?;
            }
            journal.write(json!({
                "event": "round", "active": posterior.active.iter().filter(|a| **a).count(), "objective_bits": current / LN_2,
                "groups": (0..posterior.active.len()).filter(|g| posterior.active[*g]).map(|g| json!({
                    "id": g, "name": explanation.groups[g].name, "layer": layer(&explanation.groups[g].name),
                    "size": explanation.groups[g].cells.iter().map(|c| c.rows.len() * c.cols.len()).sum::<usize>(),
                    "kl_bits": divergences[g] / LN_2, "cost_bits": costs[g] / LN_2, "signal": summary[g].signal, "data_rise_bits": data_rise[g] / LN_2,
                    "mean_abs_mu": summary[g].mean_abs, "rms_sigma": summary[g].rms_sd,
                })).collect::<Vec<Value>>(),
                "units": units.iter().map(|u| json!({"groups": u.groups, "roots": u.roots, "predicted_bits": u.predicted / LN_2})).collect::<Vec<Value>>(),
            }))?;
            units.sort_by(|a, b| a.predicted.total_cmp(&b.predicted));
            let mut rest: &[Unit] = &units;
            while !rest.is_empty() {
                // The predicted-safe set: the leading units predicted to lower `F`. Without one, the
                // units predicted not to lower `F` are proposed whole, then from the first alone.
                let safe = rest.iter().take_while(|u| u.predicted < 0.0).count();
                let tail = safe == 0;
                let base: &Posterior = posterior;
                // The groups of the first `k` units still active, with those their removal leaves
                // without effect.
                let proposal = |k: usize| -> Result<Vec<usize>, String> {
                    let mut groups: Vec<usize> = rest[..k].iter().flat_map(|u| u.groups.iter().copied()).filter(|g| base.active[*g]).collect();
                    groups.sort_unstable();
                    groups.dedup();
                    let mut active = base.active.clone();
                    for g in &groups {
                        active[*g] = false;
                    }
                    groups.extend(structure.dead(&active)?);
                    groups.sort_unstable();
                    Ok(groups)
                };
                let mut changes: BTreeMap<usize, f64> = BTreeMap::new();
                let mut test = |k: usize, kind: &str| -> Result<bool, String> {
                    let timed = Instant::now();
                    let groups = proposal(k)?;
                    let change = objective(&trial(base, &groups)?)? - current;
                    if !change.is_finite() {
                        return Err(error("a nonfinite removal objective"));
                    }
                    let removed_now = active - base.active.iter().filter(|a| **a).count();
                    let predicted = rest[..k].iter().map(|u| u.data).sum::<f64>() - groups.iter().map(|g| costs[*g]).sum::<f64>() + subset_change(removed_now + groups.len())?
                        - subset_change(removed_now)?;
                    journal.write(json!({
                        "event": "proposal", "kind": kind, "units": k, "groups": groups, "layers": names(&groups),
                        "predicted_bits": predicted / LN_2, "measured_bits": change / LN_2,
                        "accepted": change <= 0.0, "seconds": timed.elapsed().as_secs_f64(),
                    }))?;
                    evaluations.push((groups.len(), change / LN_2));
                    changes.insert(k, change);
                    Ok(change <= 0.0)
                };
                let accepted = if tail { gallop(rest.len(), true, &mut test)? } else { gallop_down(safe, &mut test)? };
                if accepted > 0 {
                    let next = trial(base, &proposal(accepted)?)?;
                    *posterior = next;
                    current += changes[&accepted];
                }
                if accepted == rest.len() {
                    break;
                }
                if !tail && accepted == safe {
                    rest = &rest[safe..];
                    continue;
                }
                // The unit the segment ended on: its own effect on top of the accepted prefix.
                let blocked = &rest[accepted];
                let marginal = changes[&(accepted + 1)] - changes.get(&accepted).copied().unwrap_or(0.0);
                singles.push((blocked.groups[0], marginal / LN_2));
                journal.write(json!({
                    "event": "proposal", "kind": "blocked", "units": 1, "groups": blocked.groups, "layers": names(&blocked.groups),
                    "predicted_bits": blocked.predicted / LN_2, "measured_bits": marginal / LN_2, "accepted": false, "seconds": 0.0,
                }))?;
                rest = &rest[accepted + 1..];
                while rest.first().is_some_and(|u| u.groups.iter().all(|g| !posterior.active[*g])) {
                    rest = &rest[1..];
                }
                if tail && accepted == 0 {
                    untested = rest.len();
                    journal.write(json!({"event": "end", "untested_units": untested, "reason": "the units predicted not to lower F were rejected whole and the first of them alone"}))?;
                    break;
                }
            }
        }
        Search::Prefix => {
            let divergences = posterior.divergences();
            let mut order: Vec<usize> = (0..divergences.len()).filter(|g| posterior.active[*g]).collect();
            order.sort_by(|a, b| divergences[*a].total_cmp(&divergences[*b]));
            let base = posterior.clone();
            let mut changes: BTreeMap<usize, f64> = BTreeMap::new();
            let mut test = |k: usize| -> Result<bool, String> {
                let timed = Instant::now();
                let change = objective(&trial(&base, &order[..k])?)? - current;
                if !change.is_finite() {
                    return Err(error("a nonfinite removal objective"));
                }
                journal.write(json!({
                    "event": "proposal", "kind": "prefix", "units": k, "groups": order[..k], "layers": names(&order[..k]),
                    "measured_bits": change / LN_2, "accepted": change <= 0.0, "seconds": timed.elapsed().as_secs_f64(),
                }))?;
                evaluations.push((k, change / LN_2));
                changes.insert(k, change);
                Ok(change <= 0.0)
            };
            let accepted = bisect(order.len(), &mut test)?;
            if accepted > 0 {
                *posterior = trial(&base, &order[..accepted])?;
                current += changes[&accepted];
            }
        }
    }
    let removed = candidates - posterior.active.iter().filter(|a| **a).count();
    journal.write(json!({
        "event": "end", "search": format!("{search:?}"), "candidates": candidates, "removed": removed,
        "before_bits": before / LN_2, "after_bits": current / LN_2, "evaluations": evaluations.len(), "seconds": started.elapsed().as_secs_f64(),
    }))?;
    Ok(Removal { candidates, removed, dead: dead_removed, before_bits: before / LN_2, after_bits: current / LN_2, evaluations, singles, untested })
}

/// The longest accepted prefix of `n` units found by testing the whole (when `whole`), then lengths
/// 1, 3, 7, … until one is rejected, then bisection between the longest accepted and the shortest
/// rejected length; `test(k, kind)` evaluates the prefix of length `k`.
fn gallop(n: usize, whole: bool, test: &mut impl FnMut(usize, &str) -> Result<bool, String>) -> Result<usize, String> {
    if n == 0 {
        return Ok(0);
    }
    if whole && test(n, "whole")? {
        return Ok(n);
    }
    // `low` is accepted (or empty) and `high` rejected, or untested when it is `n` without `whole`.
    let (mut low, mut high, mut step) = (0, n, 1);
    let rejected = whole;
    loop {
        let next = (low + step).min(high);
        if next == high && rejected {
            break;
        }
        if test(next, "gallop")? {
            low = next;
            step *= 2;
            if low == high {
                return Ok(low);
            }
        } else {
            high = next;
            break;
        }
    }
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if test(middle, "bisect")? {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(low)
}

/// The longest accepted prefix of `n` units found by testing the whole, then halving the length
/// until one is accepted, then bisection between the longest accepted and the shortest rejected
/// length; `test(k, kind)` evaluates the prefix of length `k`.
fn gallop_down(n: usize, test: &mut impl FnMut(usize, &str) -> Result<bool, String>) -> Result<usize, String> {
    if n == 0 || test(n, "whole")? {
        return Ok(n);
    }
    let (mut low, mut high, mut next) = (0, n, n / 2);
    while next > 0 {
        if test(next, "halve")? {
            low = next;
            break;
        }
        high = next;
        next /= 2;
    }
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if test(middle, "bisect")? {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(low)
}

/// The previous search's bisection: the whole first, then bisection between the empty prefix and
/// the shortest rejected one.
fn bisect(n: usize, test: &mut impl FnMut(usize) -> Result<bool, String>) -> Result<usize, String> {
    if n == 0 || test(n)? {
        return Ok(n);
    }
    let (mut low, mut high) = (0, n);
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if test(middle)? {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(low)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        interchange::{Batch, Experiment, Interchange, Targets},
        library_mdl,
        operator_program::SlotValues,
        run_check::{layer_nodes, split_sites},
    };
    use gam_gpu::tensor::{Device, posterior_normal};
    use rand::{SeedableRng, rngs::StdRng};

    /// The tiny two-layer decoder (tanh-GELU MLPs), or made like Qwen3 (gated SiLU MLPs, a norm on
    /// every head's query and key, one key-value head shared by both query heads): its split
    /// program, layers and six sequences of twelve tokens.
    fn tiny(tag: &str, qwen3: bool) -> (OperatorProgram, Vec<LayerNodes>, Vec<Vec<u32>>) {
        let dir = if qwen3 { crate::test_support::tiny_qwen3_export(tag, 2) } else { crate::test_support::tiny_export(tag, 2) };
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        (native, layers, tokens.chunks(12).map(<[u32]>::to_vec).collect())
    }

    fn removed(posterior: &Posterior, groups: &[usize]) -> Posterior {
        let mut out = posterior.clone();
        out.remove(groups);
        out
    }

    /// `P`'s final normed stream alone on `sequences` at `posterior`'s mean.
    fn hidden(ic: &mut Interchange, posterior: &Posterior, sequences: &[Vec<u32>]) -> Array2<f64> {
        ic.load(&posterior.mean).expect("the means load");
        let program = ic.models().1.program;
        let family = library_mdl::sequence_family(&sequences.iter().map(Vec::as_slice).collect::<Vec<_>>()).expect("a family");
        let trace = program.forward(&family).expect("the forward pass");
        program.device().download(trace.value(program.hidden()).expect("the hidden value")).expect("the download")
    }

    /// The curvature at which `posterior`'s `σ` is stationary, `H_jj = 1/σ_j² − 1/v_G` on the
    /// diagonal, and the gradient at which its `μ` is, `g_j = −μ_j/v_G`: the removal estimates then
    /// rank a group by `½ Σ μ_j²/σ_j²`.
    fn stationary(explanation: &Explanation, posterior: &Posterior) -> Curvature {
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let mut curvature = Curvature::new(explanation.groups.len());
        for (g, group) in explanation.groups.iter().enumerate() {
            let entries: Vec<(f64, f64)> = group
                .cells
                .iter()
                .flat_map(|cell| cell.rows.iter().flat_map(move |r| cell.cols.clone().map(move |c| (cell.operator, *r, c))))
                .map(|(op, r, c)| (posterior.mean[position[&op]][[r, c]], (2.0 * posterior.log_sd[position[&op]][[r, c]]).exp()))
                .filter(|(_, variance)| *variance > 0.0)
                .collect();
            let prior = entries.iter().map(|(mu, variance)| mu * mu + variance).sum::<f64>() / entries.len().max(1) as f64;
            for (mu, variance) in entries {
                let h = 1.0 / variance - 1.0 / prior;
                curvature.quadratic[g] += h * mu * mu;
                curvature.noise[g] += h * variance;
                curvature.slope[g] -= mu * mu / prior;
            }
        }
        curvature
    }

    /// The measured curvature and gradient of `posterior` on the evidence: one forward pass per
    /// batch at the posterior mean, reversed for the divergence's gradient and for a sampled-label
    /// draw, weighted by `weight`.
    fn measured(ic: &mut Interchange, explanation: &Explanation, posterior: &Posterior, evidence: &[Evidence], weight: f64, rng: &mut StdRng) -> Curvature {
        use rand::RngExt;
        ic.load(&posterior.means()).expect("the means load");
        let mut curvature = Curvature::new(explanation.groups.len());
        for e in evidence {
            let rows: usize = e.experiments.iter().map(|x| e.batch.length() - x.position).sum();
            let uniforms: Vec<f64> = (0..rows).map(|_| rng.random::<f64>()).collect();
            let evaluation = ic.evaluate_labelled(&e.batch, &e.experiments, Some(&e.targets), true, Some(&uniforms)).expect("the labelled evaluation");
            let factor = evaluation.factor.expect("the Gauss–Newton factor");
            let d = ic.models().1.program.device();
            let host = |gradient: &BTreeMap<usize, gam_gpu::tensor::Tensor>| -> Vec<Array2<f64>> {
                explanation
                    .trainable
                    .iter()
                    .zip(&posterior.mean)
                    .map(|(op, mean)| gradient.get(op).map_or_else(|| Array2::zeros(mean.dim()), |g| d.download(g).expect("the download")))
                    .collect()
            };
            posterior.add_curvature(&host(&factor.gradient), weight, &mut curvature).expect("the draw");
            posterior.add_slope(&host(&evaluation.gradient), weight * LN_2, &mut curvature).expect("the gradient");
        }
        curvature
    }

    fn interchange(native: &OperatorProgram, layers: &[LayerNodes], explanation: &Explanation, device: &Device) -> Interchange {
        let reads = interchange::reads(native, layers).expect("the reads");
        Interchange::new(device, native, layers, &explanation.artifact, &explanation.trainable, reads, 1 << 30, 64).expect("the experiments")
    }

    #[test]
    fn a_live_dependency_stays_live_through_the_intersection() {
        // A coordinate no single removal silences keeps the intersection empty.
        assert_eq!(intersection(&[Kill::Of(vec![1, 2]), Kill::none(), Kill::Of(vec![2])], None), Kill::none());
        assert_eq!(intersection(&[Kill::Of(vec![1, 2]), Kill::Of(vec![2, 3])], None), Kill::Of(vec![2]));
        assert_eq!(intersection(&[Kill::All, Kill::Of(vec![4])], None), Kill::Of(vec![4]));
        assert_eq!(intersection(&[], None), Kill::All);
        // With a second set per coordinate, `⋂_i (A_i ∪ B_i)`.
        assert_eq!(intersection(&[Kill::Of(vec![1]), Kill::Of(vec![2])], Some(&[Kill::Of(vec![2]), Kill::none()])), Kill::Of(vec![2]));
    }

    #[test]
    fn a_patch_can_make_a_zero_coordinate_nonzero() {
        // Base and source reads are both zero in their second coordinate, and the patch projects
        // onto the diagonal: `(I − P) x + P s` is not.
        let (x, source) = ([1.0, 0.0], [3.0, 0.0]);
        let p = [[0.5, 0.5], [0.5, 0.5]];
        let patched_read: Vec<f64> = (0..2).map(|i| x[i] - (0..2).map(|j| p[i][j] * x[j]).sum::<f64>() + (0..2).map(|j| p[i][j] * source[j]).sum::<f64>()).collect();
        assert_eq!(patched_read, vec![2.0, 1.0]);
        // So a coordinate that removing group 5 zeroes in the clean read stays live after the patch
        // while another coordinate is live.
        assert_eq!(patched(&[Kill::none(), Kill::Of(vec![5])]), vec![Kill::none(), Kill::none()]);
        // Once every coordinate is zero, so is every coordinate of the patched read.
        assert_eq!(patched(&[Kill::Of(vec![5]), Kill::Of(vec![5, 7])]), vec![Kill::Of(vec![5]), Kill::Of(vec![5])]);
    }

    #[test]
    fn groups_without_effect_are_found_exactly() {
        for qwen3 in [false, true] {
            let (native, layers, sequences) = tiny(&format!("removal_dead_{qwen3}"), qwen3);
            let explanation = library_mdl::explanation(&native, &layers).expect("the library");
            let posterior = Posterior::new(&explanation, 1000).expect("the posterior");
            let structure = Structure::new(&explanation).expect("the structure");
            assert!(structure.dead(&posterior.active).unwrap().is_empty(), "the starting library has no group without effect");
            // A head whose value coordinates are all removed: its planes have no effect.
            let (planes, values) = explanation.layers[1].heads[0].clone();
            let silent = removed(&posterior, &values);
            assert_eq!(structure.dead(&silent.active).unwrap(), planes);
            // An MLP function whose output (or gate) is removed: its other groups have no effect.
            let function = explanation.layers[0].functions[3].clone();
            for (k, &g) in function.iter().enumerate() {
                let rest: Vec<usize> = function.iter().copied().filter(|h| *h != g).collect();
                assert_eq!(structure.dead(&removed(&posterior, &[g]).active).unwrap(), rest, "removing part {k} of a function");
            }
            // One plane fewer leaves every other group in use (under a query norm the plane's query
            // rows still scale the others).
            assert!(structure.dead(&removed(&posterior, &planes[..1]).active).unwrap().is_empty());
            // A function is one unit; a value coordinate and a plane are units alone.
            let units = structure.units(&posterior.active).unwrap();
            let unit_of = |g: usize| units.iter().find(|u| u.groups.contains(&g)).expect("every group in a unit");
            assert_eq!(unit_of(function[0]).groups, function);
            assert_eq!(unit_of(function[0]).roots, function);
            assert_eq!(unit_of(values[0]).groups, vec![values[0]]);
            assert_eq!(unit_of(planes[0]).groups, vec![planes[0]]);
            // Removing the groups found without effect leaves `P` unchanged exactly.
            let device = Device::host();
            let mut ic = interchange(&native, &layers, &explanation, &device);
            let planted: Vec<usize> = values.iter().copied().chain([explanation.layers[0].functions[5][0], *explanation.layers[1].functions[2].last().unwrap()]).collect();
            let start = removed(&posterior, &planted);
            let dead = structure.dead(&start.active).unwrap();
            assert!(dead.len() >= planes.len() + 2, "{} groups without effect", dead.len());
            let (before, after) = (hidden(&mut ic, &start, &sequences), hidden(&mut ic, &removed(&start, &dead), &sequences));
            assert!(before.iter().zip(&after).all(|(a, b)| a == b), "removing groups without effect changed P");
        }
    }

    #[test]
    fn removing_the_groups_found_without_effect_never_changes_the_explanation() {
        // Random removals on a GELU library, a Qwen3-like one and one whose MLP region is a call of
        // a reusable body: every group the analysis finds without effect is removed and `P`'s
        // final normed stream stays the same, bit for bit.
        use rand::RngExt;
        for (case, qwen3) in [("gelu", false), ("qwen3", true), ("body", false)] {
            let (native, layers, sequences) = tiny(&format!("removal_random_{case}"), qwen3);
            let mut explanation = library_mdl::explanation(&native, &layers).expect("the library");
            if case == "body" {
                explanation = crate::library_bodies::rewrite(&explanation, 0, &[1, 4, 6, 9]).expect("the rewrite").0;
            }
            let posterior = Posterior::new(&explanation, 1000).expect("the posterior");
            let structure = Structure::new(&explanation).expect("the structure");
            let device = Device::host();
            let mut ic = interchange(&native, &layers, &explanation, &device);
            let mut rng = StdRng::seed_from_u64(9);
            let mut found = 0;
            for _ in 0..12 {
                let chosen: Vec<usize> = (0..posterior.active.len()).filter(|g| posterior.active[*g] && rng.random::<f64>() < 0.3).collect();
                let start = removed(&posterior, &chosen);
                let dead = structure.dead(&start.active).unwrap();
                found += dead.len();
                let (before, after) = (hidden(&mut ic, &start, &sequences), hidden(&mut ic, &removed(&start, &dead), &sequences));
                assert!(before.iter().zip(&after).all(|(a, b)| a == b), "{case}: removing groups without effect changed P");
            }
            assert!(found > 0, "{case}: random removals leave some groups without effect");
        }
    }

    #[test]
    fn the_search_continues_past_a_rejected_unit() {
        // Groups whose removal F would accept on either side, in the ranking, of one it rejects:
        // the ranked search removes both sides and the groups without effect, the prefix search
        // stops at the rejected one.
        let (native, layers, _) = tiny("removal_continue", false);
        let explanation = library_mdl::explanation(&native, &layers).expect("the library");
        let mut posterior = Posterior::new(&explanation, 1000).expect("the posterior");
        let (planes, values) = explanation.layers[1].heads[0].clone();
        posterior.remove(&values);
        // Three value coordinates of layer 0 with posterior noise `K` times their mean square, so
        // the predictions rank them by K: the middle one is the needed group.
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let chosen = &explanation.layers[0].heads[0].1;
        let (first, needed, last) = (chosen[0], chosen[1], chosen[2]);
        for (g, k) in [(first, 8.0), (needed, 4.0), (last, 2.0)] {
            let cell = &explanation.groups[g].cells[0];
            let i = position[&cell.operator];
            let square = cell.cols.clone().map(|c| posterior.mean[i][[cell.rows[0], c]].powi(2)).sum::<f64>() / cell.cols.len() as f64;
            for c in cell.cols.clone() {
                posterior.log_sd[i][[cell.rows[0], c]] = 0.5 * (k * square).ln();
            }
        }
        let start = posterior.clone();
        let start_active = start.active.clone();
        // F: the description plus a data cost for each removed group other than the free ones.
        let free: BTreeSet<usize> = planes.iter().copied().chain([first, last]).collect();
        let mut objective = |trial: &Posterior| -> Result<f64, String> {
            let data = (0..trial.active.len()).filter(|g| start_active[*g] && !trial.active[*g] && !free.contains(g)).count() as f64 * 1e6;
            Ok(data + trial.description())
        };
        let log = std::env::temp_dir().join(format!("gam_mpd_removal_continue_{}.jsonl", std::process::id()));
        let curvature = stationary(&explanation, &start);
        let ranked = round(Search::Ranked, &explanation, &mut posterior, None, &curvature, &mut objective, Some(&log)).expect("the ranked search");
        let mut prefix_posterior = start.clone();
        let prefix = round(Search::Prefix, &explanation, &mut prefix_posterior, None, &curvature, &mut objective, None).expect("the prefix search");
        for g in free.iter() {
            assert!(!posterior.active[*g], "{} survived the ranked search", explanation.groups[*g].name);
        }
        assert!(posterior.active[needed]);
        assert_eq!(ranked.removed, free.len());
        assert!(ranked.after_bits < prefix.after_bits, "ranked {} bits, prefix {} bits", ranked.after_bits, prefix.after_bits);
        assert!(prefix_posterior.active[last], "the prefix search stops before the needed group");
        let lines: Vec<Value> = std::fs::read_to_string(&log).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        std::fs::remove_file(&log).unwrap();
        let kinds: Vec<&str> = lines.iter().filter_map(|l| l["kind"].as_str()).collect();
        assert_eq!(kinds.first(), Some(&"dead"));
        assert!(kinds.contains(&"blocked"));
        assert!(lines.iter().any(|l| l["event"] == "round") && lines.last().unwrap()["event"] == "end");
    }

    /// One batch of the fixed collection: its sequences, experiments and `M`'s targets.
    struct Evidence {
        batch: Batch,
        experiments: Vec<Experiment>,
        targets: Targets,
    }

    #[test]
    fn on_a_tiny_model_the_search_removes_groups_without_effect_keeps_needed_ones_and_beats_the_prefix_search() {
        let (native, layers, sequences) = tiny("removal_model", false);
        let explanation = library_mdl::explanation(&native, &layers).expect("the library");
        let device = Device::host();
        let mut ic = interchange(&native, &layers, &explanation, &device);
        let variables = ic.variables().to_vec();
        let mut rng = StdRng::seed_from_u64(5);
        let evidence: Vec<Evidence> = (0..sequences.len() / 2)
            .map(|b| {
                let pick = |offset: usize| (0..2).map(|i| sequences[(2 * b + i + offset) % sequences.len()].clone()).collect();
                let batch = Batch::new(pick(0), pick(2)).unwrap();
                let experiments = interchange::sample(&mut rng, 2, &variables, 2 * layers.len(), 12).unwrap();
                let targets = ic.targets(&batch, &experiments).unwrap();
                Evidence { batch, experiments, targets }
            })
            .collect();
        // The collection stands for N tokens of M's behaviour: the data term weighs its scored tokens
        // by N over their count, and the posterior's precision is N's.
        let tokens = 1_000_000.0;
        let scored: usize = evidence.iter().flat_map(|e| &e.experiments).map(|e| 12 - e.position).sum();
        let weight = tokens / scored as f64;
        let objective = |ic: &mut Interchange, posterior: &Posterior| -> Result<f64, String> {
            let mut bits = 0.0;
            for (b, e) in evidence.iter().enumerate() {
                let theta: Vec<Array2<f64>> = posterior
                    .mean
                    .iter()
                    .zip(&posterior.log_sd)
                    .enumerate()
                    .map(|(i, (mean, log_sd))| {
                        let cols = mean.ncols();
                        Array2::from_shape_fn(mean.dim(), |(r, c)| {
                            let s = log_sd[[r, c]];
                            if s == f64::NEG_INFINITY { mean[[r, c]] } else { mean[[r, c]] + s.exp() * f64::from(posterior_normal(b as u64, i as u64, (r * cols + c) as u64)) }
                        })
                    })
                    .collect();
                ic.load(&theta)?;
                bits += ic.evaluate_resident(&e.batch, &e.experiments, &e.targets, false)?.bits.iter().flatten().sum::<f64>();
            }
            Ok(weight * bits * LN_2 + posterior.description())
        };
        let mut posterior = Posterior::new(&explanation, tokens as usize).expect("the posterior");
        // Planted groups without effect: the planes of a head without values, and the gates of five
        // functions without outputs.
        let (planes, values) = explanation.layers[1].heads[0].clone();
        let outputs: Vec<usize> = (0..5).map(|i| *explanation.layers[0].functions[i].last().unwrap()).collect();
        let gates: Vec<usize> = (0..5).map(|i| explanation.layers[0].functions[i][0]).collect();
        posterior.remove(&values);
        posterior.remove(&outputs);
        let planted: Vec<usize> = planes.iter().chain(&gates).copied().collect();
        let start = posterior.clone();
        let compensation = Compensation::new(&mut ic, &explanation, &start, &sequences, 2).expect("the compensation");
        let curvature = measured(&mut ic, &explanation, &start, &evidence, weight, &mut rng);
        let ranked = round(Search::Ranked, &explanation, &mut posterior, Some(&compensation), &curvature, &mut |p: &Posterior| objective(&mut ic, p), None).expect("the ranked search");
        let mut prefix_posterior = start.clone();
        let prefix = round(Search::Prefix, &explanation, &mut prefix_posterior, Some(&compensation), &curvature, &mut |p: &Posterior| objective(&mut ic, p), None).expect("the prefix search");
        for g in &planted {
            assert!(!posterior.active[*g], "{} survived", explanation.groups[*g].name);
        }
        // Removing groups without effect leaves the data term unchanged: F changes by the
        // description's change alone.
        let (dead_size, dead_change) = ranked.evaluations[0];
        assert_eq!(dead_size, planted.len());
        let description = (removed(&start, &planted).description() - start.description()) / LN_2;
        assert!((dead_change - description).abs() <= 1e-9 * ranked.before_bits.abs(), "the data term moved by {} bits", dead_change - description);
        // The group the prefix search proposes first is needed (removing it alone, compensated,
        // raises F), and the search keeps it.
        let divergences = start.divergences();
        let first = (0..divergences.len()).filter(|g| start.active[*g] && !planted.contains(g)).min_by(|a, b| divergences[*a].total_cmp(&divergences[*b])).unwrap();
        let without_dead = removed(&start, &planted);
        let alone = compensation.proposal(&without_dead, &[first]).unwrap();
        assert!(objective(&mut ic, &alone).unwrap() > objective(&mut ic, &without_dead).unwrap(), "the first group is needed");
        assert!(posterior.active[first]);
        assert!(ranked.removed >= planted.len());
        assert!(ranked.after_bits < prefix.after_bits, "ranked {} bits, prefix {} bits", ranked.after_bits, prefix.after_bits);
        // The removals beyond the groups without effect were accepted on one weight sample per
        // batch; they also lower `F` on samples they were not chosen on.
        let dead_only = removed(&start, &planted);
        let mut fresh = |p: &Posterior, key: u64| -> f64 {
            let mut bits = 0.0;
            for e in &evidence {
                let theta: Vec<Array2<f64>> = p
                    .mean
                    .iter()
                    .zip(&p.log_sd)
                    .enumerate()
                    .map(|(i, (mean, log_sd))| {
                        let cols = mean.ncols();
                        Array2::from_shape_fn(mean.dim(), |(r, c)| {
                            let s = log_sd[[r, c]];
                            if s == f64::NEG_INFINITY { mean[[r, c]] } else { mean[[r, c]] + s.exp() * f64::from(posterior_normal(key, i as u64, (r * cols + c) as u64)) }
                        })
                    })
                    .collect();
                ic.load(&theta).unwrap();
                bits += ic.evaluate_resident(&e.batch, &e.experiments, &e.targets, false).unwrap().bits.iter().flatten().sum::<f64>();
            }
            weight * bits * LN_2 + p.description()
        };
        for key in 100..103 {
            let (kept, searched) = (fresh(&dead_only, key), fresh(&posterior, key));
            assert!(searched < kept, "on fresh noise {key} the searched posterior scores {searched} nats against {kept}");
        }
    }

    #[test]
    fn the_searches_find_a_monotone_boundary_in_logarithmically_many_evaluations() {
        for n in [1, 2, 7, 1000] {
            for boundary in [0, 1, n / 2, n - 1, n] {
                for whole in [true, false] {
                    let mut tested = 0;
                    let found = gallop(n, whole, &mut |k, _| {
                        tested += 1;
                        Ok(k <= boundary)
                    })
                    .unwrap();
                    assert_eq!(found, boundary, "n {n}, whole {whole}");
                    let bound = 2 * (usize::BITS - boundary.max(1).leading_zeros()) as usize + 2;
                    assert!(tested <= bound, "{tested} evaluations for the boundary {boundary} of {n}");
                }
                let mut tested = 0;
                let found = gallop_down(n, &mut |k, _| {
                    tested += 1;
                    Ok(k <= boundary)
                })
                .unwrap();
                assert_eq!(found, boundary, "n {n}, halving");
                assert!(tested <= 3 + (usize::BITS - n.leading_zeros()) as usize, "{tested} evaluations halving to the boundary {boundary} of {n}");
                assert!(boundary < n || tested == 1, "the whole set took {tested} evaluations");
                let mut tested = 0;
                assert_eq!(bisect(n, &mut |k| {
                    tested += 1;
                    Ok(k <= boundary)
                }).unwrap(), boundary);
                assert!(tested <= 1 + (usize::BITS - n.leading_zeros()) as usize);
            }
        }
        assert!(gallop(1, true, &mut |_, _| Err("nonfinite".into())).is_err());
        assert!(gallop_down(1, &mut |_, _| Err("nonfinite".into())).is_err());
    }
}

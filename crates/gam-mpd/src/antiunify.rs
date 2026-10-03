//! Rule discovery by anti-unification over a saturated e-graph (#2951), after babble (Cao et al.,
//! POPL 2023), reimplemented over [`super::egraph`].
//!
//! # Candidates
//!
//! Two value classes of the extracted program are a candidate pair when their least general
//! generalization ([`Pattern`]) has at least one operator hole whose two leaves differ, and the
//! pair's basis invariants agree within their derived bands. Which invariants depends on the change
//! of basis a family allows:
//!
//! * every family: the traces of the first three powers of a square product of two holes along a
//!   path (`B · f(A x)` pairs `P = B A`), which an invertible change `M` of the coordinates the path
//!   enters and leaves by moves only to its similar `M P M⁻¹`; and per rotation plane of a bilinear
//!   score `⟨Q x, K y⟩`, the singular values of `Q_gᵀ K_g`;
//! * the orthogonal families also: the singular spectrum of each operator hole's two leaves, and of
//!   each path product.
//!
//! A leaf's reals are exact on its lattice, but a leaf that stands for a basis change of another
//! is only defined to the grid each entry was rounded on, at most the coarsest lattice holding the
//! entry, so each spectrum is compared within the SVD's own band plus Weyl's bound `‖(h_ij)‖_F`
//! (`h_ij` that lattice's half-step) for each leaf (propagated through a product by
//! `‖ΔB‖‖A‖ + ‖B‖‖ΔA‖ + ‖ΔB‖‖ΔA‖`, and into `tr(P^k)` by `k n ‖P‖^{k−1} ‖ΔP‖`). The fingerprints
//! nominate pairs; only the code accepts them.
//!
//! # The generalization
//!
//! `AU(a, b)` over e-classes is the class itself when `a = b`; otherwise, over every pair of e-nodes
//! of the same kind and data, the node with the children's generalizations, keeping the one with
//! the most operator holes, then the most nodes. Two operator leaves of one interface pair and one
//! present-block pattern become an operator hole; two value classes that do not generalize further
//! become an argument (a value hole). Sums generalize their summands by best pairing.
//!
//! # Bindings
//!
//! A rule stores its body once: the operators of one instance. Each other instance (a call site)
//! is `round_p(fl(M_rows · body · M_colsᵀ)) + residual` per hole, on the instance operator's own
//! lattice `p`, which decodes bit-exactly, so a rule is lossless. The coordinate map `M` acts on the
//! pattern's boundary interfaces (its arguments and its root; equal interfaces share one map) and
//! is the identity on interior interfaces. Its family is one of [`BindingFamily`]: identity,
//! permutation (an assignment on the cross-covariance), orthogonal (the polar factor, sent as signs
//! and a Cayley skew), block-diagonal over the interface's groups, linear (one invertible `M` per
//! space acting by `M⁻ᵀ` on its column side, the exact invariance of a residual stream with no
//! norm; the least-norm `M − I` of a Sylvester equation, sent as `M − I`), or general (least
//! squares, one map per side). Maps are solved by alternating over spaces until the code stops shortening; the
//! binding lattice is the one that minimizes binding plus residual bits.
//!
//! # The library
//!
//! Instances of one body with one pattern are one rule. Rules are accepted greedily by saving, each
//! only when the decoded message shrinks: the program's message without the call sites' reals, plus
//! the library section ([`RuleLibrary::section`]), which [`RuleLibrary::decode_instances`] reads back
//! into the call sites' exact operators.

use super::codec::{
    BitReader, BitString, decode_fixed_index, decode_prefix_integer, decode_signed_prefix_integer, encode_fixed_index,
    encode_prefix_integer, encode_signed_prefix_integer,
};
use super::dense::{solve, svd};
use super::egraph::{Choices, ClassData, EgraphError, LeafId, Normalization, Saturation, Term};
use super::operator_program::{Interface, OperatorBody, OperatorProgram, ProgramError};
use super::precision::{DecodableArtifact, DeclaredPrecision, LatticeCode};
use egg::{Id, Language};
use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, ArrayView2, s};
use std::collections::{BTreeMap, BTreeSet, HashMap};

// ------------------------------------------------------------------------------------ patterns

/// A generalization of two e-classes.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Both sides are this class.
    Shared(Id),
    /// A value hole: the rule's argument, `a` in the body and `b` at the call.
    Argument { a: Id, b: Id },
    /// An operator hole: leaf `a` in the body, leaf `b` at the call.
    Operator { a: LeafId, b: LeafId },
    /// One node kind with generalized children; `a` and `b` are the two e-nodes, and `chosen`
    /// counts the node pairs of this subtree that are both the extraction's own choices.
    Node { a: Term, b: Term, children: Vec<Pattern>, chosen: usize },
}

impl Pattern {
    /// `(operator holes, extracted node pairs, nodes)`: the order generalizations are preferred in.
    /// A rule is priced on the extracted program with its pattern's nodes pinned, so between two
    /// generalizations with as many holes, the one through the nodes the extraction chose pins the
    /// shorter program.
    fn score(&self) -> (usize, usize, usize) {
        match self {
            Self::Operator { a, b } => (usize::from(a != b), 0, 1),
            Self::Node { children, chosen, .. } => {
                let (holes, nodes) = children.iter().map(Pattern::score).fold((0, 1), |(h, n), (ch, _, cn)| (h + ch, n + cn));
                (holes, *chosen, nodes)
            }
            _ => (0, 0, 0),
        }
    }

    /// The pattern's shape with the body's classes: equal keys generalize the same body the same way.
    fn skeleton(&self) -> String {
        match self {
            Self::Shared(id) => format!("#{id}"),
            Self::Argument { a, .. } => format!("?{a}"),
            Self::Operator { a, .. } => format!("@{a}"),
            Self::Node { a, children, .. } => {
                let head = format!("{:?}", a.clone().map_children(|_| Id::from(0usize)));
                let inner: Vec<String> = children.iter().map(Pattern::skeleton).collect();
                format!("{head}[{}]", inner.join(","))
            }
        }
    }

    /// Every operator hole, in pattern order, with the roles of its two sides.
    fn holes(&self, egraph: &super::egraph::ProgramGraph, boundary: bool, out: &mut Vec<HoleSite>) {
        let Self::Node { a, children, .. } = self else { return };
        match a {
            Term::Apply(_) | Term::Constant(_) => {
                let (operator, argument) = (&children[0], children.get(1));
                if let Self::Operator { a: la, b: lb } = operator {
                    // The body's input: an argument hole, or a value both sides read (the rule is
                    // still applied to it, and a call may read it through a change of basis).
                    let cols = match argument {
                        Some(Self::Argument { a: x, .. }) => Role::Boundary(value_interface(egraph, *x)),
                        Some(Self::Shared(x)) if matches!(egraph[*x].data, ClassData::Value(_)) => {
                            Role::Boundary(value_interface(egraph, *x))
                        }
                        _ => Role::Interior,
                    };
                    let rows = if boundary {
                        Role::Boundary(egraph.analysis.leaves[*la as usize].operator.rows.clone())
                    } else {
                        Role::Interior
                    };
                    out.push(HoleSite { a: *la, b: *lb, rows, cols });
                }
                if let Some(argument) = argument {
                    argument.holes(egraph, false, out);
                }
            }
            Term::Sum(_) | Term::Scale(_) => {
                for child in children {
                    child.holes(egraph, boundary, out);
                }
            }
            _ => {
                for child in children {
                    child.holes(egraph, false, out);
                }
            }
        }
    }

    /// Every operator hole whose two leaves differ, as `(body, call)` leaves.
    fn operator_pairs(&self, out: &mut Vec<(LeafId, LeafId)>) {
        if let Self::Operator { a, b } = self
            && a != b
        {
            out.push((*a, *b));
        }
        if let Self::Node { children, .. } = self {
            for child in children {
                child.operator_pairs(out);
            }
        }
    }

    /// The pairs of holes along a path: an applied hole whose argument reaches, through pointwise
    /// laws and sums, another applied hole.
    fn paths(&self, out: &mut Vec<(LeafId, LeafId, LeafId, LeafId)>) {
        let Self::Node { a, children, .. } = self else { return };
        if let (Term::Apply(_), Self::Operator { a: outer_a, b: outer_b }) = (a, &children[0]) {
            let mut inner = Vec::new();
            children[1].first_applied(&mut inner);
            for (inner_a, inner_b) in inner {
                out.push((*outer_a, *outer_b, inner_a, inner_b));
            }
        }
        for child in children {
            child.paths(out);
        }
    }

    fn first_applied(&self, out: &mut Vec<(LeafId, LeafId)>) {
        let Self::Node { a, children, .. } = self else { return };
        if let (Term::Apply(_), Some(Self::Operator { a, b })) = (a, children.first()) {
            out.push((*a, *b));
        } else if matches!(a, Term::Pointwise { .. } | Term::Sum(_) | Term::Scale(_)) {
            for child in children {
                child.first_applied(out);
            }
        }
    }

    /// The `(query, key)` hole pairs of bilinear scores.
    fn scores(&self, out: &mut Vec<((LeafId, LeafId), (LeafId, LeafId))>) {
        let Self::Node { a, children, .. } = self else { return };
        if let Term::Bilinear { .. } = a {
            let (mut left, mut right) = (Vec::new(), Vec::new());
            children[0].first_applied(&mut left);
            children[1].first_applied(&mut right);
            if let (Some(q), Some(k)) = (left.first(), right.first()) {
                out.push((*q, *k));
            }
        }
        for child in children {
            child.scores(out);
        }
    }

    /// The e-node each pattern class takes in the body (`side = false`) or the call (`true`).
    fn pins(&self, egraph: &super::egraph::ProgramGraph, class: Id, side: bool, out: &mut HashMap<Id, Term>) {
        let Self::Node { a, b, children, .. } = self else { return };
        let term = if side { b } else { a };
        out.insert(egraph.find(class), term.clone());
        for (child, id) in children.iter().zip(term.children()) {
            if let Self::Node { .. } = child {
                child.pins(egraph, *id, side, out);
            } else if let Self::Operator { a, b } = child {
                out.insert(egraph.find(*id), Term::Leaf(if side { *b } else { *a }));
            }
        }
    }
}

fn value_interface(egraph: &super::egraph::ProgramGraph, class: Id) -> Interface {
    match &egraph[class].data {
        ClassData::Value(interface) => interface.clone(),
        _ => Interface::constant(),
    }
}

/// What an operator hole's side attaches to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    Interior,
    /// A boundary interface of the pattern; equal interfaces share one coordinate map.
    Boundary(Interface),
}

#[derive(Clone, Debug, PartialEq)]
struct HoleSite {
    a: LeafId,
    b: LeafId,
    rows: Role,
    cols: Role,
}

/// The least general generalization over the e-graph, memoized per class pair.
struct Generalizer<'a> {
    saturation: &'a Saturation,
    /// The extraction's node per class.
    choices: &'a Choices,
    memo: HashMap<(Id, Id), Option<Pattern>>,
    active: BTreeSet<(Id, Id)>,
}

impl Generalizer<'_> {
    fn leaf_pair(&self, a: Id, b: Id) -> Option<Pattern> {
        let egraph = &self.saturation.egraph;
        let (ClassData::Operator(left), ClassData::Operator(right)) = (&egraph[a].data, &egraph[b].data) else { return None };
        let (la, lb) = (&egraph.analysis.leaves[left.best as usize], &egraph.analysis.leaves[right.best as usize]);
        let present = |op: &super::operator_program::Operator| match &op.body {
            OperatorBody::Dense { present, .. } => Some(present.clone()),
            _ => None,
        };
        let same = la.operator.rows == lb.operator.rows
            && la.operator.cols == lb.operator.cols
            && present(&la.operator).is_some()
            && present(&la.operator) == present(&lb.operator);
        same.then_some(Pattern::Operator { a: left.best, b: right.best })
    }

    fn generalize(&mut self, a: Id, b: Id) -> Option<Pattern> {
        let egraph = &self.saturation.egraph;
        let (a, b) = (egraph.find(a), egraph.find(b));
        if a == b {
            return Some(Pattern::Shared(a));
        }
        match (&egraph[a].data, &egraph[b].data) {
            (ClassData::Operator(_), ClassData::Operator(_)) => return self.leaf_pair(a, b),
            (ClassData::Value(x), ClassData::Value(y)) if x == y => {}
            _ => return None,
        }
        if let Some(found) = self.memo.get(&(a, b)) {
            return found.clone();
        }
        if !self.active.insert((a, b)) {
            return Some(Pattern::Argument { a, b });
        }
        let mut best: Option<Pattern> = None;
        let (left_nodes, right_nodes) = (egraph[a].nodes.clone(), egraph[b].nodes.clone());
        for na in &left_nodes {
            for nb in &right_nodes {
                // A node reading its own class (`I x` in the class of `x`) has no finite extraction.
                let cyclic = |node: &Term, class: Id| node.children().iter().any(|c| egraph.find(*c) == class);
                if !na.matches(nb) || cyclic(na, a) || cyclic(nb, b) {
                    continue;
                }
                let children = match na {
                    Term::Sum(_) => self.pair_summands(na.children(), nb.children()),
                    _ => na
                        .children()
                        .iter()
                        .zip(nb.children())
                        .map(|(x, y)| self.generalize(*x, *y))
                        .collect::<Option<Vec<Pattern>>>(),
                };
                let Some(children) = children else { continue };
                let extracted = |class: Id, node: &Term| {
                    self.choices.get(&class).is_some_and(|(_, term)| {
                        term.clone().map_children(|c| egraph.find(c)) == node.clone().map_children(|c| egraph.find(c))
                    })
                };
                let below: usize = children.iter().map(|child| child.score().1).sum();
                let chosen = below + usize::from(extracted(a, na) && extracted(b, nb));
                let candidate = Pattern::Node { a: na.clone(), b: nb.clone(), children, chosen };
                if best.as_ref().is_none_or(|current| candidate.score() > current.score()) {
                    best = Some(candidate);
                }
            }
        }
        self.active.remove(&(a, b));
        let found = Some(best.unwrap_or(Pattern::Argument { a, b }));
        self.memo.insert((a, b), found.clone());
        found
    }

    /// Summands paired greedily by the best generalization, in the body's order.
    fn pair_summands(&mut self, left: &[Id], right: &[Id]) -> Option<Vec<Pattern>> {
        if left.len() != right.len() {
            return None;
        }
        let mut unused: Vec<Id> = right.to_vec();
        let mut out = Vec::new();
        for &x in left {
            let mut chosen: Option<(usize, Pattern)> = None;
            for (at, &y) in unused.iter().enumerate() {
                let Some(pattern) = self.generalize(x, y) else { continue };
                if chosen.as_ref().is_none_or(|(_, current)| pattern.score() > current.score()) {
                    chosen = Some((at, pattern));
                }
            }
            let (at, pattern) = chosen?;
            unused.remove(at);
            out.push(pattern);
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------------- fingerprints

fn lattice_half_step(operator: &super::operator_program::Operator) -> f64 {
    match &operator.body {
        OperatorBody::Dense { precision, .. } | OperatorBody::LowRank { precision, .. } | OperatorBody::Diagonal { precision, .. } => {
            precision.worst_case_error()
        }
        OperatorBody::Identity => 0.0,
    }
}

/// Half the step of the coarsest dyadic lattice holding the nonzero `value`: the largest rounding
/// error a grid `value` lies on can have left in it.
fn coarsest_half_step(value: f64) -> f64 {
    let bits = value.abs().to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (mantissa, scale) = if exponent == 0 { (fraction, -1074) } else { (fraction | (1_u64 << 52), exponent - 1075) };
    2.0_f64.powi(scale + mantissa.trailing_zeros() as i32 - 1)
}

/// `‖(h_ij)‖_F`, each `h_ij` the half-step of the coarsest lattice holding entry `ij` (the
/// operator's own `half_step` for a zero): Weyl's bound on the spectral move of a matrix whose
/// entries were each rounded to some grid they lie on. Units moved by different powers of two
/// (the canonical gauge) were rounded on different grids, so no single step bounds them all.
fn resolution(matrix: &Array2<f64>, half_step: f64) -> f64 {
    let squares: f64 = matrix
        .iter()
        .map(|&v| if v == 0.0 { half_step } else { coarsest_half_step(v).max(half_step) })
        .map(|h| h * h)
        .sum();
    super::egraph::outward(squares.sqrt().next_up(), matrix.len())
}

fn singular_values(matrix: &Array2<f64>) -> Option<(Vec<f64>, f64)> {
    let decomposed = svd(matrix.view(), false).ok()?;
    Some((decomposed.singular_values.to_vec(), decomposed.band))
}

fn within(left: &[f64], right: &[f64], band: f64) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(x, y)| (x - y).abs() <= band)
}

/// Two leaves' singular spectra agree within their SVD bands and their resolutions.
fn spectra_agree(a: &Array2<f64>, half_a: f64, b: &Array2<f64>, half_b: f64) -> bool {
    match (singular_values(a), singular_values(b)) {
        (Some((sa, band_a)), Some((sb, band_b))) => within(&sa, &sb, band_a + band_b + resolution(a, half_a) + resolution(b, half_b)),
        _ => false,
    }
}

/// The spectrum and, when square, `tr(P^k)` for `k = 1, 2, 3` of `P = B A`, with the bands a
/// resolution `ΔA`, `ΔB` (spectral norms) moves them by.
fn product_fingerprint(b: &Array2<f64>, delta_b: f64, a: &Array2<f64>, delta_a: f64) -> Option<(Vec<f64>, f64, Vec<f64>, Vec<f64>)> {
    let product = b.dot(a);
    let (norm_a, norm_b) = (singular_values(a)?.0.first().copied().unwrap_or(0.0), singular_values(b)?.0.first().copied().unwrap_or(0.0));
    let rounding = accumulation_growth(a.nrows()) * b.mapv(f64::abs).dot(&a.mapv(f64::abs)).iter().map(|v| v * v).sum::<f64>().sqrt();
    let delta = (delta_b * norm_a + norm_b * delta_a + delta_b * delta_a + rounding).next_up();
    let (spectrum, band) = singular_values(&product)?;
    let norm = spectrum.first().copied().unwrap_or(0.0) + delta;
    let (mut traces, mut trace_bands) = (Vec::new(), Vec::new());
    if product.nrows() == product.ncols() {
        let n = product.nrows() as f64;
        let mut power = product.clone();
        for k in 1..=3 {
            traces.push(power.diag().sum());
            let magnitude = power.mapv(f64::abs).diag().sum();
            let movement = k as f64 * n * norm.powi(k - 1) * delta;
            trace_bands.push((movement + accumulation_growth(product.nrows() * k as usize) * magnitude * norm.powi(k)).next_up());
            power = power.dot(&product);
        }
    }
    Some((spectrum, band + delta, traces, trace_bands))
}

/// The binding families a pattern's invariants admit. An orthogonal change of basis keeps every
/// hole's singular spectrum and every path product's spectrum and traces, so the orthogonal
/// families (identity, permutation, orthogonal, block-diagonal) need all of them to agree; an
/// invertible change of basis `M` moves a path product `P = B A` to its similar `M P M⁻¹`, so the
/// linear families need only the traces `tr(P^k)`. Bilinear joint spectra `Q_gᵀ K_g` are kept by
/// both. Nothing else is asked: the code decides.
fn admissible_families(saturation: &Saturation, pattern: &Pattern) -> Vec<BindingFamily> {
    let leaves = &saturation.egraph.analysis.leaves;
    let data = |leaf: LeafId| {
        let l = &leaves[leaf as usize];
        (&l.center, resolution(&l.center, lattice_half_step(&l.operator)))
    };
    let mut pairs = Vec::new();
    pattern.operator_pairs(&mut pairs);
    let mut orthogonal = pairs.iter().all(|&(a, b)| {
        let (la, lb) = (&leaves[a as usize], &leaves[b as usize]);
        spectra_agree(&la.center, lattice_half_step(&la.operator), &lb.center, lattice_half_step(&lb.operator))
    });
    let mut paths = Vec::new();
    pattern.paths(&mut paths);
    for (outer_a, outer_b, inner_a, inner_b) in paths {
        let ((ba, dba), (aa, daa)) = (data(outer_a), data(inner_a));
        let ((bb, dbb), (ab, dab)) = (data(outer_b), data(inner_b));
        if ba.ncols() != aa.nrows() {
            continue;
        }
        let (Some(left), Some(right)) = (product_fingerprint(ba, dba, aa, daa), product_fingerprint(bb, dbb, ab, dab)) else {
            return Vec::new();
        };
        orthogonal &= within(&left.0, &right.0, left.1 + right.1);
        for ((x, y), (bx, by)) in left.2.iter().zip(&right.2).zip(left.3.iter().zip(&right.3)) {
            if (x - y).abs() > bx + by {
                return Vec::new();
            }
        }
    }
    let mut scores = Vec::new();
    pattern.scores(&mut scores);
    for ((qa, qb), (ka, kb)) in scores {
        let (q_a, k_a, q_b, k_b) = (&leaves[qa as usize], &leaves[ka as usize], &leaves[qb as usize], &leaves[kb as usize]);
        if q_a.operator.rows != k_a.operator.rows {
            continue;
        }
        for group in 0..q_a.operator.rows.group_count() {
            let range = q_a.operator.rows.range(group);
            let joint = |q: &super::egraph::Leaf, k: &super::egraph::Leaf| {
                let (qg, kg) = (q.center.slice(s![range.clone(), ..]).to_owned(), k.center.slice(s![range.clone(), ..]).to_owned());
                product_fingerprint(
                    &qg.t().to_owned(),
                    resolution(&qg, lattice_half_step(&q.operator)),
                    &kg,
                    resolution(&kg, lattice_half_step(&k.operator)),
                )
            };
            let (Some(left), Some(right)) = (joint(q_a, k_a), joint(q_b, k_b)) else { return Vec::new() };
            if !within(&left.0, &right.0, left.1 + right.1) {
                return Vec::new();
            }
        }
    }
    FAMILIES.into_iter().filter(|family| orthogonal || !family.orthogonal()).collect()
}

// -------------------------------------------------------------------------------------- bindings

/// The family a call site's coordinate maps are drawn from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BindingFamily {
    Identity,
    Permutation,
    Orthogonal,
    BlockDiagonal,
    /// One invertible map `M` per space, `M⁻ᵀ` on its column side: the change of basis a space with
    /// no norm admits exactly. Sent as `M − I`.
    Linear,
    General,
}

const FAMILIES: [BindingFamily; 6] = [
    BindingFamily::Identity,
    BindingFamily::Permutation,
    BindingFamily::Orthogonal,
    BindingFamily::BlockDiagonal,
    BindingFamily::Linear,
    BindingFamily::General,
];

impl BindingFamily {
    fn index(self) -> usize {
        FAMILIES.iter().position(|f| *f == self).unwrap_or(0)
    }

    /// Whether one map serves both sides of a space (`M` on rows, `M⁻ᵀ` on columns).
    fn tied(self) -> bool {
        self != Self::General
    }

    /// Whether the family's maps are orthogonal.
    fn orthogonal(self) -> bool {
        !matches!(self, Self::Linear | Self::General)
    }
}

/// One coordinate map as sent: what the decoder rebuilds the map from.
#[derive(Clone, Debug, PartialEq)]
pub enum MapCode {
    Identity,
    /// `permutation[i]` is the body coordinate that call coordinate `i` reads.
    Permutation(Vec<usize>),
    /// `D · C(S)`, `C(S) = (I − S)(I + S)⁻¹`, per group (one group for the orthogonal family):
    /// the signs of `D` and the upper triangle of each group's skew `S` on the binding lattice.
    Cayley { groups: Vec<std::ops::Range<usize>>, signs: Vec<bool>, skew: Vec<f64> },
    /// The matrix's entries on the binding lattice.
    Matrix(Array2<f64>),
    /// `I + Δ`, with `Δ`'s entries on the binding lattice.
    Shift(Array2<f64>),
}

/// The map a code decodes to: exactly the matrix the decoder computes.
fn decode_map(code: &MapCode, width: usize) -> Result<Array2<f64>, EgraphError> {
    Ok(match code {
        MapCode::Identity => Array2::eye(width),
        MapCode::Permutation(permutation) => {
            let mut m = Array2::zeros((width, width));
            for (i, &j) in permutation.iter().enumerate() {
                m[[i, j]] = 1.0;
            }
            m
        }
        MapCode::Cayley { groups, signs, skew } => {
            let mut m = Array2::zeros((width, width));
            let mut next = skew.iter();
            for group in groups {
                let n = group.len();
                let mut s_matrix = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    for j in i + 1..n {
                        let v = *next.next().ok_or_else(|| EgraphError::Inconsistent("a short skew".to_string()))?;
                        s_matrix[[i, j]] = v;
                        s_matrix[[j, i]] = -v;
                    }
                }
                let eye = Array2::<f64>::eye(n);
                let x = solve((&eye - &s_matrix).view(), (&eye + &s_matrix).view())
                    .map_err(|error| EgraphError::Inconsistent(format!("a Cayley map: {error:?}")))?;
                let c = x.t().to_owned();
                for (i, row) in group.clone().enumerate() {
                    let sign = if signs[row] { -1.0 } else { 1.0 };
                    for (j, col) in group.clone().enumerate() {
                        m[[row, col]] = sign * c[[i, j]];
                    }
                }
            }
            m
        }
        MapCode::Matrix(matrix) => matrix.clone(),
        MapCode::Shift(delta) => delta + &Array2::<f64>::eye(width),
    })
}

/// The map a slot's map acts by on one side: `M⁻ᵀ` on the column side of a linear slot, else `M`.
fn oriented(family: BindingFamily, map: &Array2<f64>, rows: bool) -> Result<Array2<f64>, EgraphError> {
    if family != BindingFamily::Linear || rows {
        return Ok(map.clone());
    }
    solve(map.t(), Array2::<f64>::eye(map.nrows()).view())
        .map_err(|error| EgraphError::Inconsistent(format!("a singular linear binding: {error:?}")))
}

/// Row signs `D` with `I + D M` invertible for an orthogonal `M`. `I + D M` is reduced row by
/// row: row `k` is `e_k + d_k m_k`, linear in its sign, so its eliminated pivot is `a + d_k b`, and
/// `d_k = sign(a b)` makes it `|a| + |b|`: the pivots add and never cancel. (Signs read off `M`'s
/// diagonal alone can leave `det(D M) = −1`, an eigenvalue `−1` and a singular `I + D M`.)
fn cayley_signs(m: &Array2<f64>) -> Vec<bool> {
    let n = m.nrows();
    let mut reduced: Vec<Array1<f64>> = Vec::with_capacity(n);
    let mut negative = Vec::with_capacity(n);
    for k in 0..n {
        let mut a = Array1::<f64>::zeros(n);
        a[k] = 1.0;
        let mut b = m.row(k).to_owned();
        for (j, row) in reduced.iter().enumerate() {
            let (fa, fb) = (a[j] / row[j], b[j] / row[j]);
            a.scaled_add(-fa, row);
            b.scaled_add(-fb, row);
        }
        let flip = a[k] * b[k] < 0.0;
        negative.push(flip);
        reduced.push(if flip { &a - &b } else { &a + &b });
    }
    negative
}

/// The Cayley code of an orthogonal target on `groups` at lattice `bits`, or `None` when
/// `I + D M` is singular on a group.
fn cayley_code(target: &Array2<f64>, groups: &[std::ops::Range<usize>], bits: i32) -> Option<MapCode> {
    let lattice = DeclaredPrecision::new(bits).ok()?;
    let mut signs = vec![false; target.nrows()];
    let mut skew = Vec::new();
    for group in groups {
        let block = target.slice(s![group.clone(), group.clone()]).to_owned();
        let n = group.len();
        let mut signed = block.clone();
        for (i, negative) in cayley_signs(&block).into_iter().enumerate() {
            signs[group.start + i] = negative;
            if negative {
                signed.row_mut(i).mapv_inplace(|v| -v);
            }
        }
        let eye = Array2::<f64>::eye(n);
        let st = solve((&eye + &signed).t(), (&eye - &signed).t()).ok()?;
        let s_matrix = st.t().to_owned();
        for i in 0..n {
            for j in i + 1..n {
                let v = 0.5 * (s_matrix[[i, j]] - s_matrix[[j, i]]);
                skew.push(super::operator_program::round_to_lattice(v, lattice).ok()?);
            }
        }
    }
    Some(MapCode::Cayley { groups: groups.to_vec(), signs, skew })
}

/// `U Vᵀ` for `C = U Σ Vᵀ`: the orthogonal matrix nearest `C` in trace.
fn polar(c: &Array2<f64>) -> Option<Array2<f64>> {
    let decomposed = svd(c.view(), false).ok()?;
    Some(decomposed.u.dot(&decomposed.vt))
}

/// The permutation maximizing `Σ_i C[i, π(i)]` (Hungarian method with potentials).
fn assignment(c: &Array2<f64>) -> Vec<usize> {
    let n = c.nrows();
    let cost = |i: usize, j: usize| -c[[i - 1, j - 1]];
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let (mut p, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let (i0, mut delta, mut j1) = (p[j0], f64::INFINITY, 0);
            for j in 1..=n {
                if !used[j] {
                    let current = cost(i0, j) - u[i0] - v[j];
                    if current < minv[j] {
                        minv[j] = current;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut permutation = vec![0; n];
    for j in 1..=n {
        if p[j] > 0 {
            permutation[p[j] - 1] = j - 1;
        }
    }
    permutation
}

/// A map slot: a boundary interface, and for the general family the side.
type Slot = (Interface, bool);

fn slot_of(role: &Role, rows: bool, family: BindingFamily) -> Option<Slot> {
    match role {
        Role::Interior => None,
        Role::Boundary(interface) => Some((interface.clone(), !family.tied() && rows)),
    }
}

/// The map slots of a pattern's holes within a family, in hole order (rows side, then columns).
fn slots_for(roles: &[(Role, Role)], family: BindingFamily) -> Vec<Slot> {
    let mut slots: Vec<Slot> = Vec::new();
    for (rows, cols) in roles {
        for slot in [slot_of(rows, true, family), slot_of(cols, false, family)].into_iter().flatten() {
            if !slots.contains(&slot) {
                slots.push(slot);
            }
        }
    }
    slots
}

/// A priced call site: its maps, the message bits and the message itself.
#[derive(Clone, Debug)]
pub struct Binding {
    pub family: BindingFamily,
    /// Per boundary slot (interface, side), the decoded map.
    pub maps: Vec<(Interface, bool, Array2<f64>)>,
    /// Bits of the call site's section (family, maps, lattices, residuals).
    pub bits: u64,
    /// Bits spent on residuals alone.
    pub residual_bits: u64,
    message: BitString,
}

/// `round_p(fl(M_rows · body · M_colsᵀ))`, the call's prediction of an instance on lattice `p`.
fn predict(body: &Array2<f64>, rows: Option<&Array2<f64>>, cols: Option<&Array2<f64>>, lattice: DeclaredPrecision) -> Result<Array2<f64>, EgraphError> {
    let mut out = body.clone();
    if let Some(m) = rows {
        out = m.dot(&out);
    }
    if let Some(m) = cols {
        out = out.dot(&m.t());
    }
    for v in out.iter_mut() {
        *v = super::operator_program::round_to_lattice(*v, lattice)?;
    }
    Ok(out)
}

fn present_values(values: &Array2<f64>, operator: &super::operator_program::Operator) -> Vec<f64> {
    let OperatorBody::Dense { present, .. } = &operator.body else { return values.iter().copied().collect() };
    let mut out = Vec::new();
    for ((r, c), keep) in present.indexed_iter() {
        if *keep {
            out.extend(values.slice(s![operator.rows.range(r), operator.cols.range(c)]).iter().copied());
        }
    }
    out
}

fn precision_of(operator: &super::operator_program::Operator) -> Option<DeclaredPrecision> {
    match &operator.body {
        OperatorBody::Dense { precision, .. } => Some(*precision),
        _ => None,
    }
}

/// Write one call site's section: family, map codes on the binding lattice, then per hole its
/// lattice, a residual flag and the residual's lattice code.
fn write_call(
    saturation: &Saturation,
    holes: &[HoleSite],
    family: BindingFamily,
    slots: &[Slot],
    codes: &[MapCode],
    bits_lattice: i32,
) -> Result<(BitString, u64, Vec<Array2<f64>>), EgraphError> {
    let leaves = &saturation.egraph.analysis.leaves;
    let mut out = BitString::new();
    encode_fixed_index(&mut out, family.index(), FAMILIES.len()).map_err(ProgramError::from)?;
    let lattice = DeclaredPrecision::new(bits_lattice).map_err(|m| EgraphError::Program(ProgramError::Code(m)))?;
    let mut reals: Vec<f64> = Vec::new();
    let mut maps = Vec::new();
    for ((interface, _), code) in slots.iter().zip(codes) {
        match code {
            MapCode::Identity => {}
            MapCode::Permutation(permutation) => {
                let mut remaining: Vec<usize> = (0..permutation.len()).collect();
                for target in permutation {
                    let at = remaining.iter().position(|r| r == target).unwrap_or(0);
                    encode_fixed_index(&mut out, at, remaining.len()).map_err(ProgramError::from)?;
                    remaining.remove(at);
                }
            }
            MapCode::Cayley { signs, skew, .. } => {
                for sign in signs {
                    out.push_bit(*sign);
                }
                reals.extend(skew);
            }
            MapCode::Matrix(matrix) | MapCode::Shift(matrix) => reals.extend(matrix.iter().copied()),
        }
        maps.push(decode_map(code, interface.width())?);
    }
    if !reals.is_empty() {
        LatticeCode::encode(&reals, lattice).map_err(|m| EgraphError::Program(ProgramError::Code(m)))?.write(&mut out).map_err(|m| EgraphError::Program(ProgramError::Code(m)))?;
    }
    let map_for = |role: &Role, rows: bool| -> Result<Option<Array2<f64>>, EgraphError> {
        let Some(slot) = slot_of(role, rows, family) else { return Ok(None) };
        slots.iter().position(|s| *s == slot).map(|at| oriented(family, &maps[at], rows)).transpose()
    };
    let mut residual_bits = 0;
    for hole in holes {
        let (body, instance) = (&leaves[hole.a as usize], &leaves[hole.b as usize]);
        let instance_lattice = precision_of(&instance.operator).ok_or_else(|| EgraphError::Inconsistent("a non-dense call operator".to_string()))?;
        let before = out.len_bits();
        encode_signed_prefix_integer(&mut out, i64::from(instance_lattice.fraction_bits())).map_err(ProgramError::from)?;
        let (rows_map, cols_map) = (map_for(&hole.rows, true)?, map_for(&hole.cols, false)?);
        let predicted = predict(&body.center, rows_map.as_ref(), cols_map.as_ref(), instance_lattice)?;
        let residual = &instance.center - &predicted;
        let nonzero = residual.iter().any(|v| *v != 0.0);
        out.push_bit(nonzero);
        if nonzero {
            let values = present_values(&residual, &instance.operator);
            LatticeCode::encode(&values, instance_lattice)
                .and_then(|code| code.write(&mut out))
                .map_err(|m| EgraphError::Program(ProgramError::Code(m)))?;
        }
        residual_bits += out.len_bits() - before;
    }
    Ok((out, residual_bits, maps))
}

/// The cross-covariance of a slot given the other side's current maps.
fn cross(saturation: &Saturation, holes: &[HoleSite], slot: &Slot, family: BindingFamily, current: &HashMap<Slot, Array2<f64>>) -> (Array2<f64>, Array2<f64>) {
    let leaves = &saturation.egraph.analysis.leaves;
    let width = slot.0.width();
    let (mut c, mut gram) = (Array2::<f64>::zeros((width, width)), Array2::<f64>::zeros((width, width)));
    for hole in holes {
        let (a, b) = (&leaves[hole.a as usize].center, &leaves[hole.b as usize].center);
        if slot_of(&hole.rows, true, family).as_ref() == Some(slot) {
            let other = slot_of(&hole.cols, false, family).and_then(|s| current.get(&s).cloned());
            let moved = match &other {
                Some(k) => a.dot(&k.t()),
                None => a.clone(),
            };
            c += &b.dot(&moved.t());
            gram += &moved.dot(&moved.t());
        }
        if slot_of(&hole.cols, false, family).as_ref() == Some(slot) {
            let other = slot_of(&hole.rows, true, family).and_then(|s| current.get(&s).cloned());
            let moved = match &other {
                Some(r) => r.dot(a),
                None => a.clone(),
            };
            c += &b.t().dot(&moved);
            gram += &moved.t().dot(&moved);
        }
    }
    (c, gram)
}

/// The groups of an interface as coordinate ranges.
fn groups_of(interface: &Interface) -> Vec<std::ops::Range<usize>> {
    (0..interface.group_count()).map(|g| interface.range(g)).collect()
}

/// A slot's map target within a family, from its cross-covariance.
fn project(family: BindingFamily, interface: &Interface, c: &Array2<f64>, gram: &Array2<f64>) -> Option<Array2<f64>> {
    let width = interface.width();
    match family {
        BindingFamily::Identity => Some(Array2::eye(width)),
        BindingFamily::Permutation => {
            let permutation = assignment(c);
            decode_map(&MapCode::Permutation(permutation), width).ok()
        }
        BindingFamily::Orthogonal => polar(c),
        BindingFamily::BlockDiagonal => {
            let mut m = Array2::zeros((width, width));
            for group in groups_of(interface) {
                let block = polar(&c.slice(s![group.clone(), group.clone()]).to_owned())?;
                m.slice_mut(s![group.clone(), group]).assign(&block);
            }
            Some(m)
        }
        BindingFamily::General => {
            let solved = least_norm_solve(gram.t(), c.t())?;
            Some(solved.t().to_owned())
        }
        BindingFamily::Linear => None,
    }
}

/// `min ‖X‖_F` among the minimizers of `‖A X − B‖_F`: the thin SVD with every singular value
/// within the decomposition's own band of zero dropped.
fn least_norm_solve(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let decomposed = svd(a, false).ok()?;
    let rank = decomposed.singular_values.iter().filter(|&&value| value > decomposed.band).count();
    let projected = decomposed.u.slice(s![.., ..rank]).t().dot(&b);
    let scaled = Array2::from_shape_fn(projected.dim(), |(row, col)| projected[[row, col]] / decomposed.singular_values[row]);
    Some(decomposed.vt.slice(s![..rank, ..]).t().dot(&scaled))
}

/// The map `M = I + Δ` of a linear slot with the other slots at `current`. A hole with the slot on
/// its rows asks `M X = B` (`X` the body with its column map applied); a hole with the slot on its
/// columns, whose map there is `M⁻ᵀ`, asks `B M = Y` (`Y` the body with its row map applied). Both
/// are linear in `Δ`, and the least-squares normal equations are the Sylvester equation
/// `Δ G_x + G_b Δ = C` with `G_x = Σ X Xᵀ`, `G_b = Σ Bᵀ B`, `C = Σ (B − X) Xᵀ + Σ Bᵀ (Y − B)`. In the
/// eigenbases `G_b = U Λ Uᵀ`, `G_x = V M Vᵀ` it is `Z_ij (λ_i + μ_j) = (Uᵀ C V)_ij`; a denominator
/// within the two eigendecompositions' bands leaves `Z_ij = 0`, so `Δ` is the least-norm solution:
/// the smallest change of basis the instances determine.
fn linear_target(saturation: &Saturation, holes: &[HoleSite], slot: &Slot, current: &HashMap<Slot, Array2<f64>>) -> Option<Array2<f64>> {
    let leaves = &saturation.egraph.analysis.leaves;
    let family = BindingFamily::Linear;
    let width = slot.0.width();
    let (mut g_x, mut g_b, mut c) = (Array2::<f64>::zeros((width, width)), Array2::<f64>::zeros((width, width)), Array2::<f64>::zeros((width, width)));
    let other = |role: &Role, rows: bool| -> Result<Option<Array2<f64>>, EgraphError> {
        let Some(s) = slot_of(role, rows, family) else { return Ok(None) };
        current.get(&s).map(|m| oriented(family, m, rows)).transpose()
    };
    for hole in holes {
        let (a, b) = (&leaves[hole.a as usize].center, &leaves[hole.b as usize].center);
        if slot_of(&hole.rows, true, family).as_ref() == Some(slot) {
            let x = match other(&hole.cols, false).ok()? {
                Some(k) => a.dot(&k.t()),
                None => a.clone(),
            };
            g_x += &x.dot(&x.t());
            c += &(b - &x).dot(&x.t());
        }
        if slot_of(&hole.cols, false, family).as_ref() == Some(slot) {
            let y = match other(&hole.rows, true).ok()? {
                Some(r) => r.dot(a),
                None => a.clone(),
            };
            g_b += &b.t().dot(b);
            c += &b.t().dot(&(&y - b));
        }
    }
    let (eb, ex) = (svd(g_b.view(), false).ok()?, svd(g_x.view(), false).ok()?);
    let band = eb.band + ex.band;
    let mut z = eb.u.t().dot(&c).dot(&ex.u);
    for ((i, j), v) in z.indexed_iter_mut() {
        let denominator = eb.singular_values[i] + ex.singular_values[j];
        *v = if denominator > band { *v / denominator } else { 0.0 };
    }
    Some(eb.u.dot(&z).dot(&ex.u.t()) + Array2::<f64>::eye(width))
}

/// The code of a map target at lattice `bits`.
fn encode_target(family: BindingFamily, interface: &Interface, target: &Array2<f64>, bits: i32) -> Option<MapCode> {
    match family {
        BindingFamily::Identity => Some(MapCode::Identity),
        BindingFamily::Permutation => {
            let permutation: Vec<usize> =
                target.outer_iter().map(|row| row.iter().position(|v| *v == 1.0).unwrap_or(0)).collect();
            Some(MapCode::Permutation(permutation))
        }
        BindingFamily::Orthogonal => cayley_code(target, &[0..interface.width()], bits),
        BindingFamily::BlockDiagonal => cayley_code(target, &groups_of(interface), bits),
        BindingFamily::General | BindingFamily::Linear => {
            let lattice = DeclaredPrecision::new(bits).ok()?;
            let shift = family == BindingFamily::Linear;
            let mut rounded = if shift { target - &Array2::<f64>::eye(target.nrows()) } else { target.clone() };
            for v in rounded.iter_mut() {
                *v = super::operator_program::round_to_lattice(*v, lattice).ok()?;
            }
            Some(if shift { MapCode::Shift(rounded) } else { MapCode::Matrix(rounded) })
        }
    }
}

/// The largest magnitude a family's sent reals take for `target`, which bounds the binding lattice.
fn largest_real(target: &Array2<f64>) -> f64 {
    target.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0)
}

/// The cheapest binding of a call site within `family`: maps alternated over slots until the code
/// stops shortening, each priced at every binding lattice from the one that rounds every real to
/// zero to the finest binary64 holds.
fn bind(saturation: &Saturation, holes: &[HoleSite], family: BindingFamily) -> Result<Option<Binding>, EgraphError> {
    let roles: Vec<(Role, Role)> = holes.iter().map(|h| (h.rows.clone(), h.cols.clone())).collect();
    let slots = slots_for(&roles, family);
    let mut current: HashMap<Slot, Array2<f64>> = slots.iter().map(|s| (s.clone(), Array2::eye(s.0.width()))).collect();
    let mut best: Option<Binding> = None;
    loop {
        let mut targets = Vec::new();
        for slot in &slots {
            let target = if family == BindingFamily::Linear {
                linear_target(saturation, holes, slot, &current)
            } else {
                let (c, gram) = cross(saturation, holes, slot, family, &current);
                project(family, &slot.0, &c, &gram)
            };
            let Some(target) = target else { return Ok(best) };
            targets.push(target);
        }
        let largest = targets.iter().map(largest_real).fold(1.0_f64, f64::max);
        let top = 52 - largest.log2().ceil() as i32;
        let bottom = -(largest.log2().ceil() as i32) - 1;
        let mut round_best: Option<Binding> = None;
        let lattices: Vec<i32> = if family == BindingFamily::Identity || family == BindingFamily::Permutation { vec![0] } else { (bottom..=top).collect() };
        for bits in lattices {
            let codes: Option<Vec<MapCode>> = slots.iter().zip(&targets).map(|(slot, t)| encode_target(family, &slot.0, t, bits)).collect();
            let Some(codes) = codes else { continue };
            let (message, residual_bits, maps) = match write_call(saturation, holes, family, &slots, &codes, bits) {
                Ok(written) => written,
                Err(EgraphError::Inconsistent(_)) => continue,
                Err(other) => return Err(other),
            };
            let bits_total = message.len_bits();
            if round_best.as_ref().is_none_or(|b| bits_total < b.bits) {
                round_best = Some(Binding {
                    family,
                    maps: slots.iter().zip(maps).map(|((i, side), m)| (i.clone(), *side, m)).collect(),
                    bits: bits_total,
                    residual_bits,
                    message,
                });
            }
        }
        let Some(found) = round_best else { return Ok(best) };
        if best.as_ref().is_some_and(|b| found.bits >= b.bits) {
            return Ok(best);
        }
        for (slot, (_, _, map)) in slots.iter().zip(&found.maps) {
            current.insert(slot.clone(), map.clone());
        }
        best = Some(found);
        if slots.is_empty() || family == BindingFamily::Identity {
            return Ok(best);
        }
    }
}

// --------------------------------------------------------------------------------------- library

/// A call site: an instance of a rule's body.
#[derive(Clone, Debug)]
pub struct CallSite {
    pub class: Id,
    /// The generalization of the body and this instance.
    pub pattern: Pattern,
    /// The instance's operators, one per hole, as leaves.
    pub leaves: Vec<LeafId>,
    pub binding: Binding,
}

/// A rule: a pattern whose body is stored once, and its call sites.
#[derive(Clone, Debug)]
pub struct Rule {
    pub skeleton: String,
    /// The body's class and its operator holes (leaves), in pattern order.
    pub body: Id,
    pub holes: Vec<LeafId>,
    pub roles: Vec<(Role, Role)>,
    pub calls: Vec<CallSite>,
    /// Bits saved: the call sites' replaced reals minus the rule's section.
    pub saving: i64,
}

/// The accepted rules and the decoded message lengths around them.
#[derive(Clone, Debug)]
pub struct RuleLibrary {
    pub rules: Vec<Rule>,
    /// The normalized program's message.
    pub bits_before: u64,
    /// The program (with the rules' call sites pinned) without the call sites' reals, plus the
    /// library section.
    pub bits_after: u64,
    /// The program the rules index into.
    pub program: OperatorProgram,
    /// Per rule: hole body operator indices; per call: instance operator indices.
    pub operator_indices: Vec<(Vec<usize>, Vec<Vec<usize>>)>,
    /// The library section: every rule's header and call sites.
    pub section: BitString,
}

/// Leaf → operator index of a lowered program after pruning (operators keep their order).
fn leaf_indices(saturation: &Saturation, program: &OperatorProgram, leaves: &BTreeSet<LeafId>) -> BTreeMap<LeafId, usize> {
    let mut out = BTreeMap::new();
    for leaf in leaves {
        let operator = &saturation.egraph.analysis.leaves[*leaf as usize].operator;
        if let Some(at) = program.operators.iter().position(|op| **op == *operator) {
            out.insert(*leaf, at);
        }
    }
    out
}

/// The value classes of the extracted program, in extraction order.
fn extracted_classes(saturation: &Saturation, choices: &Choices) -> Vec<Id> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut stack = vec![saturation.egraph.find(saturation.root)];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some((_, term)) = choices.get(&id) else { continue };
        if matches!(saturation.egraph[id].data, ClassData::Value(_)) {
            out.push(id);
        }
        stack.extend(term.children().iter().map(|c| saturation.egraph.find(*c)));
    }
    out.sort_unstable();
    out
}

/// Discover rules in a normalized program and accept them by decoded bits.
pub fn discover_rules(normalization: &Normalization) -> Result<RuleLibrary, EgraphError> {
    let saturation = &normalization.saturation;
    let choices = &normalization.extraction.choices;
    let bits_before = normalization.extraction.bits;
    let classes = extracted_classes(saturation, choices);
    let mut generalizer = Generalizer { saturation, choices, memo: HashMap::new(), active: BTreeSet::new() };
    // Candidate pairs grouped by body and pattern.
    let mut groups: BTreeMap<(Id, String), Vec<(Id, Pattern, Vec<BindingFamily>)>> = BTreeMap::new();
    for (i, &a) in classes.iter().enumerate() {
        for &b in &classes[i + 1..] {
            let Some(pattern) = generalizer.generalize(a, b) else { continue };
            if !matches!(pattern, Pattern::Node { .. }) || pattern.score().0 == 0 {
                continue;
            }
            let families = admissible_families(saturation, &pattern);
            if families.is_empty() {
                continue;
            }
            groups.entry((a, pattern.skeleton())).or_default().push((b, pattern, families));
        }
    }
    // Price every group as a rule.
    let mut candidates: Vec<Rule> = Vec::new();
    for ((body, skeleton), calls) in groups {
        let mut rule_holes: Option<(Vec<LeafId>, Vec<(Role, Role)>)> = None;
        let mut sites = Vec::new();
        let mut replaced = 0i64;
        for (class, pattern, families) in calls {
            let mut holes = Vec::new();
            pattern.holes(&saturation.egraph, true, &mut holes);
            if holes.iter().any(|h| h.a == h.b) {
                holes.retain(|h| h.a != h.b);
            }
            if holes.is_empty() {
                continue;
            }
            let body_leaves: Vec<LeafId> = holes.iter().map(|h| h.a).collect();
            let roles: Vec<(Role, Role)> = holes.iter().map(|h| (h.rows.clone(), h.cols.clone())).collect();
            if let Some((leaves, _)) = &rule_holes {
                if *leaves != body_leaves {
                    continue;
                }
            } else {
                rule_holes = Some((body_leaves, roles));
            }
            let mut cheapest: Option<Binding> = None;
            for family in families {
                if let Some(binding) = bind(saturation, &holes, family)?
                    && cheapest.as_ref().is_none_or(|c| binding.bits < c.bits)
                {
                    cheapest = Some(binding);
                }
            }
            let Some(binding) = cheapest else { continue };
            let instance_reals: u64 = holes
                .iter()
                .map(|h| saturation.egraph.analysis.leaves[h.b as usize].operator.code_bits().map(|(_, reals)| reals))
                .sum::<Result<u64, ProgramError>>()?;
            replaced += instance_reals as i64 - binding.bits as i64;
            sites.push(CallSite { class, pattern, leaves: holes.iter().map(|h| h.b).collect(), binding });
        }
        let Some((holes, roles)) = rule_holes else { continue };
        if sites.is_empty() {
            continue;
        }
        let header = rule_header_bits(holes.len(), sites.len(), saturation)?;
        candidates.push(Rule { skeleton, body, holes, roles, calls: sites, saving: replaced - header as i64 });
    }
    candidates.sort_by(|x, y| y.saving.cmp(&x.saving).then(x.body.cmp(&y.body)));
    // Accept greedily: a rule whose call operators are all unclaimed and that shortens the message.
    let mut accepted: Vec<Rule> = Vec::new();
    let mut claimed: BTreeSet<LeafId> = BTreeSet::new();
    let mut program = normalization.extraction.program.clone();
    let mut bits_after = bits_before;
    let mut operator_indices = Vec::new();
    let mut section = BitString::new();
    for rule in candidates {
        let touched: Vec<LeafId> = rule.holes.iter().chain(rule.calls.iter().flat_map(|c| c.leaves.iter())).copied().collect();
        if rule.saving <= 0 || touched.iter().any(|leaf| claimed.contains(leaf)) {
            continue;
        }
        let mut trial = accepted.clone();
        trial.push(rule.clone());
        let Some(priced) = price(normalization, &trial)? else { continue };
        if priced.bits < bits_after {
            claimed.extend(touched);
            accepted = trial;
            bits_after = priced.bits;
            program = priced.program;
            operator_indices = priced.operator_indices;
            section = priced.section;
        }
    }
    Ok(RuleLibrary { rules: accepted, bits_before, bits_after, program, operator_indices, section })
}

fn rule_header_bits(holes: usize, calls: usize, saturation: &Saturation) -> Result<u64, EgraphError> {
    let operators = saturation.egraph.analysis.leaves.len().max(2);
    let index = u64::from(super::codec::fixed_index_len_bits(operators).map_err(ProgramError::from)?);
    Ok(super::codec::prefix_integer_len_bits(holes as u64 + 1).map_err(ProgramError::from)?
        + super::codec::prefix_integer_len_bits(calls as u64 + 1).map_err(ProgramError::from)?
        + (holes as u64) * (1 + calls as u64) * index)
}

/// A priced library.
struct Priced {
    program: OperatorProgram,
    bits: u64,
    operator_indices: Vec<(Vec<usize>, Vec<Vec<usize>>)>,
    section: BitString,
}

/// The program with every rule's body and call sites pinned to their pattern's e-nodes and leaves,
/// and the length of that program's message without the call sites' reals plus the library
/// section; `None` when two patterns pin one class differently or a pinned leaf is absent.
fn price(normalization: &Normalization, rules: &[Rule]) -> Result<Option<Priced>, EgraphError> {
    let saturation = &normalization.saturation;
    let mut pins: HashMap<Id, Term> = HashMap::new();
    for rule in rules {
        for call in &rule.calls {
            let mut own = HashMap::new();
            call.pattern.pins(&saturation.egraph, rule.body, false, &mut own);
            call.pattern.pins(&saturation.egraph, call.class, true, &mut own);
            for (class, term) in own {
                if pins.get(&class).is_some_and(|existing| *existing != term) {
                    return Ok(None);
                }
                pins.insert(class, term);
            }
        }
    }
    let extraction = match saturation.extract(&pins) {
        Ok(extraction) => extraction,
        Err(EgraphError::Inconsistent(_)) => return Ok(None),
        Err(other) => return Err(other),
    };
    let program = extraction.program;
    let indices = leaf_indices(saturation, &program, &extraction.leaves);
    let mut section = BitString::new();
    encode_prefix_integer(&mut section, rules.len() as u64 + 1).map_err(ProgramError::from)?;
    let operators = program.operators.len().max(1);
    let mut removed = 0u64;
    let mut operator_indices = Vec::new();
    for rule in rules {
        let body: Option<Vec<usize>> = rule.holes.iter().map(|l| indices.get(l).copied()).collect();
        let Some(body) = body else { return Ok(None) };
        encode_prefix_integer(&mut section, rule.holes.len() as u64 + 1).map_err(ProgramError::from)?;
        for &index in &body {
            encode_fixed_index(&mut section, index, operators).map_err(ProgramError::from)?;
        }
        encode_prefix_integer(&mut section, rule.calls.len() as u64 + 1).map_err(ProgramError::from)?;
        let mut calls = Vec::new();
        for call in &rule.calls {
            let instance: Option<Vec<usize>> = call.leaves.iter().map(|l| indices.get(l).copied()).collect();
            let Some(instance) = instance else { return Ok(None) };
            for &index in &instance {
                encode_fixed_index(&mut section, index, operators).map_err(ProgramError::from)?;
                removed += program.operators[index].code_bits()?.1;
            }
            section.append(&call.binding.message);
            calls.push(instance);
        }
        operator_indices.push((body, calls));
    }
    let bits = program.code_bits()? - removed + section.len_bits();
    Ok(Some(Priced { program, bits, operator_indices, section }))
}

impl RuleLibrary {
    /// Read the section back: every call site's operator matrix, rebuilt from the body operators of
    /// `self.program`, the maps, the lattices and the residuals. The decoder knows each rule's hole
    /// roles and each call's map slots from the pattern the program's nodes carry.
    pub fn decode_instances(&self) -> Result<Vec<(usize, Array2<f64>)>, EgraphError> {
        let mut reader: BitReader<'_> = self.section.reader();
        let program_error = |e: super::codec::CodecError| EgraphError::Program(ProgramError::from(e));
        let rules = decode_prefix_integer(&mut reader).map_err(program_error)? - 1;
        let operators = self.program.operators.len().max(1);
        let mut out = Vec::new();
        for rule_index in 0..rules as usize {
            let rule = self.rules.get(rule_index).ok_or_else(|| EgraphError::Inconsistent("more rules than accepted".to_string()))?;
            let holes = decode_prefix_integer(&mut reader).map_err(program_error)? as usize - 1;
            let mut body = Vec::with_capacity(holes);
            for _ in 0..holes {
                body.push(decode_fixed_index(&mut reader, operators).map_err(program_error)?);
            }
            let calls = decode_prefix_integer(&mut reader).map_err(program_error)? as usize - 1;
            for _ in 0..calls {
                let mut instance = Vec::with_capacity(holes);
                for _ in 0..holes {
                    instance.push(decode_fixed_index(&mut reader, operators).map_err(program_error)?);
                }
                let family = FAMILIES[decode_fixed_index(&mut reader, FAMILIES.len()).map_err(program_error)?];
                let slots = slots_for(&rule.roles, family);
                let mut codes = Vec::with_capacity(slots.len());
                let mut real_count = 0usize;
                for (interface, _) in &slots {
                    let width = interface.width();
                    let groups = match family {
                        BindingFamily::BlockDiagonal => groups_of(interface),
                        _ => vec![0..width],
                    };
                    codes.push(match family {
                        BindingFamily::Identity => MapCode::Identity,
                        BindingFamily::Permutation => {
                            let mut remaining: Vec<usize> = (0..width).collect();
                            let mut permutation = Vec::with_capacity(width);
                            for _ in 0..width {
                                let at = decode_fixed_index(&mut reader, remaining.len()).map_err(program_error)?;
                                permutation.push(remaining.remove(at));
                            }
                            MapCode::Permutation(permutation)
                        }
                        BindingFamily::Orthogonal | BindingFamily::BlockDiagonal => {
                            let signs = (0..width).map(|_| reader.read_bit()).collect::<Result<Vec<bool>, _>>().map_err(program_error)?;
                            let count: usize = groups.iter().map(|g| g.len() * (g.len() - 1) / 2).sum();
                            real_count += count;
                            MapCode::Cayley { groups, signs, skew: vec![0.0; count] }
                        }
                        BindingFamily::General => {
                            real_count += width * width;
                            MapCode::Matrix(Array2::zeros((width, width)))
                        }
                        BindingFamily::Linear => {
                            real_count += width * width;
                            MapCode::Shift(Array2::zeros((width, width)))
                        }
                    });
                }
                let reals = if real_count > 0 {
                    let code = LatticeCode::read(&mut reader).map_err(|m| EgraphError::Program(ProgramError::Code(m)))?;
                    code.decode().map_err(|m| EgraphError::Program(ProgramError::Code(m)))?
                } else {
                    Vec::new()
                };
                let mut next = reals.into_iter();
                let mut maps = Vec::with_capacity(slots.len());
                for ((interface, _), code) in slots.iter().zip(codes) {
                    let filled = match code {
                        MapCode::Cayley { groups, signs, skew } => {
                            MapCode::Cayley { groups, signs, skew: next.by_ref().take(skew.len()).collect() }
                        }
                        MapCode::Matrix(matrix) => {
                            let values: Vec<f64> = next.by_ref().take(matrix.len()).collect();
                            MapCode::Matrix(Array2::from_shape_vec(matrix.dim(), values).map_err(|e| EgraphError::Inconsistent(e.to_string()))?)
                        }
                        MapCode::Shift(delta) => {
                            let values: Vec<f64> = next.by_ref().take(delta.len()).collect();
                            MapCode::Shift(Array2::from_shape_vec(delta.dim(), values).map_err(|e| EgraphError::Inconsistent(e.to_string()))?)
                        }
                        other => other,
                    };
                    maps.push(decode_map(&filled, interface.width())?);
                }
                let slot_map = |role: &Role, rows: bool| -> Result<Option<Array2<f64>>, EgraphError> {
                    let Some(slot) = slot_of(role, rows, family) else { return Ok(None) };
                    slots.iter().position(|s| *s == slot).map(|at| oriented(family, &maps[at], rows)).transpose()
                };
                for ((body_index, instance_index), (rows, cols)) in body.iter().zip(&instance).zip(&rule.roles) {
                    let lattice_bits = decode_signed_prefix_integer(&mut reader).map_err(program_error)?;
                    let lattice = DeclaredPrecision::new(lattice_bits as i32).map_err(|m| EgraphError::Program(ProgramError::Code(m)))?;
                    let body_matrix = self.program.operators[*body_index].matrix();
                    let (rows_map, cols_map) = (slot_map(rows, true)?, slot_map(cols, false)?);
                    let mut value = predict(&body_matrix, rows_map.as_ref(), cols_map.as_ref(), lattice)?;
                    if reader.read_bit().map_err(program_error)? {
                        let code = LatticeCode::read(&mut reader).map_err(|m| EgraphError::Program(ProgramError::Code(m)))?;
                        let residual = code.decode().map_err(|m| EgraphError::Program(ProgramError::Code(m)))?;
                        let operator = &self.program.operators[*instance_index];
                        let OperatorBody::Dense { present, .. } = &operator.body else {
                            return Err(EgraphError::Inconsistent("a non-dense call operator".to_string()));
                        };
                        let mut next_residual = residual.into_iter();
                        for ((r, c), keep) in present.indexed_iter() {
                            if *keep {
                                for v in value.slice_mut(s![operator.rows.range(r), operator.cols.range(c)]).iter_mut() {
                                    *v += next_residual.next().unwrap_or(0.0);
                                }
                            }
                        }
                    }
                    out.push((*instance_index, value));
                }
            }
        }
        reader.finish().map_err(program_error)?;
        Ok(out)
    }
}

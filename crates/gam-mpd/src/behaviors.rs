//! Behaviour discovery by a two-part code over a partition of the family (#2951).
//!
//! The model is asked which behaviours it has: the family's rows are partitioned into groups
//! `S_1 … S_K`, each explained by its own subprogram `P_k`, and the partition, the subprograms and
//! `K` itself are whatever makes the whole description shortest,
//!
//! ```text
//! T = L_int(K) + L(library) + Σ_k L(P_k | library) + Σ_k Σ_{r ∈ S_k} KL(model_r ‖ P_k,r)/ln 2 + L(Π).
//! ```
//!
//! Nothing is declared but the family, the model and the row features the recognizer may read. No
//! behaviour is named in advance, no tolerance or group count is set.
//!
//! # The codes
//!
//! Every term is the length of a message a decoder reads back, in the codec's own codes
//! (`codec`):
//!
//! * **The subprograms.** Each `P_k` is sent in its own message (the engine's decoded program
//!   code). An operator that occurs exactly equal in two or more subprograms is sent once in a
//!   library: the library is its size in the prefix code followed by those operators' codes, and
//!   each subprogram's operator slots then carry one bit (inline or library) and, for a library
//!   operator, a fixed index into the library instead of the operator. So shared rules are paid
//!   once ([`joint_program_bits`]).
//! * **The data.** Row `r` of group `k` costs `KL(model_r ‖ P_k,r)/ln 2`, the excess code length of
//!   the model's output coded with `P_k`'s distribution.
//! * **The partition** `L(Π)` codes each row's group through a recognizer: a decision tree over
//!   declared categorical row features ([`RowFeature`], e.g. a slot's token), whose tests are
//!   `feature == value`. The tree is sent preorder: one bit per node (leaf or test); a test sends its
//!   feature as a fixed index and its value as a fixed index into the feature's alphabet; a leaf
//!   sends its default group as a fixed index into `K`, then, for every other group in ascending
//!   order, the positions of its rows among the leaf's rows not yet placed, in the enumerative
//!   subset code (`L_int(n_k + 1) + ⌈log₂ C(m, n_k)⌉`); the default group takes the rest. The
//!   decoder knows every row's feature values, so it knows each leaf's rows. A one-leaf tree is the
//!   plain enumerative code of the labels; a rule that recognizes a group from the input costs its
//!   tests and leaves nothing to enumerate. A feature that is not a declared input but a computed
//!   predicate carries its definition's bits, paid once when the tree reads it, after the subset
//!   of such features read.
//!
//! # When a partition pays
//!
//! Splitting never pays by restriction alone. If a subprogram is the model with pieces removed and
//! each piece `b` of code `c_b` is kept exactly when the rows need it more than it costs, a group
//! pays `Σ_b min(c_b, e_b(S))` with `e_b(S)` the rows' summed need, additive over rows, and
//! `min(c, x + y) ≤ min(c, x) + min(c, y)`: the union program is never longer than the parts,
//! which also repeat the program's structure and pay `L(Π)`. Two disjoint modules of one model
//! are therefore one behaviour to this code. A partition pays exactly where, on a group's rows,
//! the computation simplifies beyond restriction: a unit whose law is fixed there (always off,
//! always linear), a routing that is constant there, an output that is constant there, and the
//! group is cheaper to recognize from the input than the model's own gate or router is to send.
//! That is what a behaviour is under this code: a recognizable region on which the network is a
//! shorter program. The region-scoped moves that expose the simplification are library
//! primitives; this module only partitions and scores.
//!
//! # The search
//!
//! The fitter ([`GroupFitter`]) returns, for a row set, a subprogram fitted to exactly those rows
//! (the model's shortest program on those rows, started from the program of a group that held
//! them when there is one) with its data bits on them, and, when a
//! reassignment needs them, its screened data bits on every row of the family. Fits are memoized
//! by row set while their groups are current.
//!
//! 1. Start from one group, the whole family.
//! 2. **Seeds**, in the caller's order (rows whose active structure is most concentrated first).
//!    A seed is screened first: the program and data bits of every group it touches, less those
//!    of their refits on the rows they keep and of the seed's own fit. A seed whose screened
//!    saving is positive is carved out as a new group and relaxed; it is kept only when the total
//!    drops. The screen is the engine's rule for proposals: it ranks and admits, never accepts.
//!    The order only orders.
//! 3. **Relax** to a fixpoint: fit the recognizer to the labels; move every row to the group whose
//!    program codes it cheapest plus its screened label cost in its leaf; refit the changed
//!    groups; keep the move only when the total drops. A group left empty is removed.
//! 4. **Merge** every pair of groups (one program fitted on the union), then relax; kept only when
//!    the total drops.
//! 5. Repeat 2–4 until a full pass keeps nothing. Every kept move strictly lowers `T`, and the
//!    states are finite, so the search ends; `K` is what survives.
//!
//! The ranking during the search uses screened data bits and a screened label cost; the reported
//! total recomputes every code exactly, and each group's fidelity is certified by the fitter on
//! its rows ([`GroupFitter::certify`]).

use super::codec::{CodecError, fixed_index_len_bits, prefix_integer_len_bits, subset_code_len_bits};
use statrs::function::gamma::ln_gamma;
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::LN_2;
use std::fmt;
use std::cell::OnceCell;
use std::rc::Rc;

/// A refused discovery.
#[derive(Debug)]
pub enum BehaviorError {
    Code(CodecError),
    Fit(String),
    Declaration(String),
}

impl fmt::Display for BehaviorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Code(error) => write!(f, "behaviours: {error:?}"),
            Self::Fit(message) => write!(f, "behaviours: fit: {message}"),
            Self::Declaration(message) => write!(f, "behaviours: {message}"),
        }
    }
}

impl std::error::Error for BehaviorError {}

impl From<CodecError> for BehaviorError {
    fn from(error: CodecError) -> Self {
        Self::Code(error)
    }
}

/// A categorical feature of every row the recognizer may test.
#[derive(Clone, Debug, PartialEq)]
pub struct RowFeature {
    pub name: String,
    /// The number of values, known to the decoder.
    pub alphabet: usize,
    /// One value per row, each below `alphabet`.
    pub values: Vec<u32>,
    /// `None` for a declared input the decoder computes from the family (a slot's token); the
    /// definition's code length for a computed predicate, paid once when the tree reads it.
    pub definition_bits: Option<u64>,
}

// ------------------------------------------------------------------------------ the program library

/// A subprogram's code as the library sees it: its whole message length and, per operator slot,
/// an exact content key and that operator's bits in the message.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramParts {
    pub bits: u64,
    pub operators: Vec<(String, u64)>,
}

/// The joint code of several subprograms with a shared library (module note): the library's
/// length and each subprogram's length given the library.
pub fn joint_program_bits(programs: &[&ProgramParts]) -> Result<(u64, Vec<u64>), BehaviorError> {
    let mut holders: BTreeMap<&str, (BTreeSet<usize>, u64)> = BTreeMap::new();
    for (index, parts) in programs.iter().enumerate() {
        for (key, bits) in &parts.operators {
            let entry = holders.entry(key.as_str()).or_insert((BTreeSet::new(), *bits));
            if entry.1 != *bits {
                return Err(BehaviorError::Declaration(format!("operator {key} has two code lengths")));
            }
            entry.0.insert(index);
        }
    }
    let library: BTreeMap<&str, u64> =
        holders.into_iter().filter(|(_, (users, _))| users.len() > 1).map(|(key, (_, bits))| (key, bits)).collect();
    let mut library_bits = prefix_integer_len_bits(library.len() as u64 + 1)?;
    library_bits += library.values().sum::<u64>();
    let reference = if library.is_empty() { 0 } else { u64::from(fixed_index_len_bits(library.len())?) };
    let conditional = programs
        .iter()
        .map(|parts| {
            if library.is_empty() {
                return parts.bits;
            }
            let mut bits = parts.bits + parts.operators.len() as u64;
            for (key, own) in &parts.operators {
                if library.contains_key(key.as_str()) {
                    bits = bits - own + reference;
                }
            }
            bits
        })
        .collect();
    Ok((library_bits, conditional))
}

// ------------------------------------------------------------------------------------ the recognizer

/// The recognizer: a decision tree whose tests are `feature == value`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Recognizer {
    Leaf { default: usize },
    Split { feature: usize, value: u32, equal: Box<Recognizer>, other: Box<Recognizer> },
}

impl Recognizer {
    /// The leaf index (preorder among leaves) of `row`.
    fn leaf_of(&self, features: &[RowFeature], row: usize) -> usize {
        let mut node = self;
        let mut offset = 0;
        loop {
            match node {
                Self::Leaf { .. } => return offset,
                Self::Split { feature, value, equal, other } => {
                    if features[*feature].values[row] == *value {
                        node = equal;
                    } else {
                        offset += equal.leaf_count();
                        node = other;
                    }
                }
            }
        }
    }

    fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Split { equal, other, .. } => equal.leaf_count() + other.leaf_count(),
        }
    }

    fn features_read(&self, out: &mut BTreeSet<usize>) {
        if let Self::Split { feature, equal, other, .. } = self {
            out.insert(*feature);
            equal.features_read(out);
            other.features_read(out);
        }
    }

    /// Every leaf's path of tests (`feature`, `value`, `equal`) and default group, preorder.
    pub fn paths(&self) -> Vec<(Vec<(usize, u32, bool)>, usize)> {
        let mut out = Vec::new();
        fn walk(node: &Recognizer, path: &mut Vec<(usize, u32, bool)>, out: &mut Vec<(Vec<(usize, u32, bool)>, usize)>) {
            match node {
                Recognizer::Leaf { default } => out.push((path.clone(), *default)),
                Recognizer::Split { feature, value, equal, other } => {
                    path.push((*feature, *value, true));
                    walk(equal, path, out);
                    path.pop();
                    path.push((*feature, *value, false));
                    walk(other, path, out);
                    path.pop();
                }
            }
        }
        walk(self, &mut Vec::new(), &mut out);
        out
    }
}

/// The screened length of the enumerative subset code, `L_int(k + 1) + log₂ C(n, k)`.
fn screened_subset_bits(universe: usize, cardinality: usize) -> Result<f64, BehaviorError> {
    let log_binomial = ln_gamma(universe as f64 + 1.0)
        - ln_gamma(cardinality as f64 + 1.0)
        - ln_gamma((universe - cardinality) as f64 + 1.0);
    Ok(prefix_integer_len_bits(cardinality as u64 + 1)? as f64 + (log_binomial / LN_2).max(0.0))
}

/// The leaf's default group: its most frequent, the lowest on ties.
fn default_of(counts: &[usize]) -> usize {
    let mut best = 0;
    for (k, &n) in counts.iter().enumerate() {
        if n > counts[best] {
            best = k;
        }
    }
    best
}

/// A leaf's label code (module note), exact or screened.
fn leaf_bits<F>(counts: &[usize], subset: F) -> Result<f64, BehaviorError>
where
    F: Fn(usize, usize) -> Result<f64, BehaviorError>,
{
    let default = default_of(counts);
    let mut bits = f64::from(fixed_index_len_bits(counts.len())?);
    let mut remaining: usize = counts.iter().sum();
    for (k, &n) in counts.iter().enumerate() {
        if k == default {
            continue;
        }
        bits += subset(remaining, n)?;
        remaining -= n;
    }
    Ok(bits)
}

fn exact_leaf_bits(counts: &[usize]) -> Result<u64, BehaviorError> {
    Ok(leaf_bits(counts, |n, k| Ok(subset_code_len_bits(n, k)? as f64))? as u64)
}

fn screened_leaf_bits(counts: &[usize]) -> Result<f64, BehaviorError> {
    leaf_bits(counts, screened_subset_bits)
}

fn split_bits(features: &[RowFeature], feature: usize) -> Result<u64, BehaviorError> {
    Ok(u64::from(fixed_index_len_bits(features.len())?) + u64::from(fixed_index_len_bits(features[feature].alphabet)?))
}

fn label_counts(rows: &[usize], labels: &[usize], groups: usize) -> Vec<usize> {
    let mut counts = vec![0; groups];
    for &row in rows {
        counts[labels[row]] += 1;
    }
    counts
}

/// Grow the tree while a test lowers the leaves' screened label code, the test's own cost aside
/// (so a test that only pays with the tests below it is not missed); [`prune`] then keeps a test
/// only where its exact code pays.
fn grow(
    rows: &[usize],
    labels: &[usize],
    groups: usize,
    features: &[RowFeature],
    allowed: &BTreeSet<usize>,
) -> Result<Recognizer, BehaviorError> {
    let counts = label_counts(rows, labels, groups);
    let leaf = Recognizer::Leaf { default: default_of(&counts) };
    if counts.iter().filter(|n| **n > 0).count() < 2 {
        return Ok(leaf);
    }
    let parent = screened_leaf_bits(&counts)?;
    let mut best: Option<(f64, usize, u32)> = None;
    for &feature in allowed {
        let mut by_value: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &row in rows {
            by_value.entry(features[feature].values[row]).or_insert_with(|| vec![0; groups])[labels[row]] += 1;
        }
        if by_value.len() < 2 {
            continue;
        }
        for (value, equal) in &by_value {
            let other: Vec<usize> = counts.iter().zip(equal).map(|(a, b)| a - b).collect();
            let bits = screened_leaf_bits(equal)? + screened_leaf_bits(&other)?;
            if best.is_none_or(|(b, _, _)| bits < b) {
                best = Some((bits, feature, *value));
            }
        }
    }
    match best {
        Some((bits, feature, value)) if bits < parent => {
            let (equal, other): (Vec<usize>, Vec<usize>) =
                rows.iter().partition(|&&row| features[feature].values[row] == value);
            Ok(Recognizer::Split {
                feature,
                value,
                equal: Box::new(grow(&equal, labels, groups, features, allowed)?),
                other: Box::new(grow(&other, labels, groups, features, allowed)?),
            })
        }
        _ => Ok(leaf),
    }
}

/// Whether a code length is computed exactly (the report) or screened (the search's ranking).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lengths {
    Exact,
    Screened,
}

impl Lengths {
    fn leaf(self, counts: &[usize]) -> Result<f64, BehaviorError> {
        match self {
            Self::Exact => Ok(exact_leaf_bits(counts)? as f64),
            Self::Screened => screened_leaf_bits(counts),
        }
    }
}

/// Collapse every test whose subtree's code is not shorter than one leaf's; returns the pruned
/// tree and its bits.
fn prune(
    node: Recognizer,
    rows: &[usize],
    labels: &[usize],
    groups: usize,
    features: &[RowFeature],
    lengths: Lengths,
) -> Result<(Recognizer, f64), BehaviorError> {
    let counts = label_counts(rows, labels, groups);
    let leaf_total = 1.0 + lengths.leaf(&counts)?;
    match node {
        Recognizer::Leaf { .. } => Ok((Recognizer::Leaf { default: default_of(&counts) }, leaf_total)),
        Recognizer::Split { feature, value, equal, other } => {
            let (equal_rows, other_rows): (Vec<usize>, Vec<usize>) =
                rows.iter().partition(|&&row| features[feature].values[row] == value);
            let (equal, equal_bits) = prune(*equal, &equal_rows, labels, groups, features, lengths)?;
            let (other, other_bits) = prune(*other, &other_rows, labels, groups, features, lengths)?;
            let split_total = 1.0 + split_bits(features, feature)? as f64 + equal_bits + other_bits;
            if split_total < leaf_total {
                Ok((Recognizer::Split { feature, value, equal: Box::new(equal), other: Box::new(other) }, split_total))
            } else {
                Ok((Recognizer::Leaf { default: default_of(&counts) }, leaf_total))
            }
        }
    }
}

/// The partition code of a tree fitted with `allowed` features, the computed predicates it reads
/// paid once (module note).
fn fit_with(
    labels: &[usize],
    groups: usize,
    features: &[RowFeature],
    allowed: &BTreeSet<usize>,
    lengths: Lengths,
) -> Result<(Recognizer, f64), BehaviorError> {
    let rows: Vec<usize> = (0..labels.len()).collect();
    let grown = grow(&rows, labels, groups, features, allowed)?;
    let (tree, tree_bits) = prune(grown, &rows, labels, groups, features, lengths)?;
    let predicates = predicate_bits(&tree, features)? as f64;
    Ok((tree, tree_bits + predicates))
}

fn predicate_bits(tree: &Recognizer, features: &[RowFeature]) -> Result<u64, BehaviorError> {
    let computed: Vec<usize> = (0..features.len()).filter(|f| features[*f].definition_bits.is_some()).collect();
    if computed.is_empty() {
        return Ok(0);
    }
    let mut read = BTreeSet::new();
    tree.features_read(&mut read);
    let used: Vec<usize> = computed.iter().copied().filter(|f| read.contains(f)).collect();
    let definitions: u64 = used.iter().map(|f| features[*f].definition_bits.unwrap_or(0)).sum();
    Ok(subset_code_len_bits(computed.len(), used.len())? + definitions)
}

fn fit_lengths(
    labels: &[usize],
    groups: usize,
    features: &[RowFeature],
    lengths: Lengths,
) -> Result<(Recognizer, f64), BehaviorError> {
    if labels.iter().any(|l| *l >= groups) || features.iter().any(|f| f.values.len() != labels.len()) {
        return Err(BehaviorError::Declaration("labels or feature values do not cover the rows".to_string()));
    }
    let mut allowed: BTreeSet<usize> =
        (0..features.len()).filter(|f| features[*f].definition_bits.is_none()).collect();
    let mut best = fit_with(labels, groups, features, &allowed, lengths)?;
    loop {
        let mut improved: Option<(usize, (Recognizer, f64))> = None;
        for candidate in (0..features.len()).filter(|f| !allowed.contains(f)) {
            let mut trial = allowed.clone();
            trial.insert(candidate);
            let fitted = fit_with(labels, groups, features, &trial, lengths)?;
            let current = improved.as_ref().map_or(best.1, |(_, b)| b.1);
            if fitted.1 < current {
                improved = Some((candidate, fitted));
            }
        }
        match improved {
            Some((feature, fitted)) => {
                allowed.insert(feature);
                best = fitted;
            }
            None => return Ok(best),
        }
    }
}

/// The shortest recognizer of `labels` over `groups` groups the search finds, and its exact bits:
/// the declared features always, and each computed predicate added while it shortens the code.
pub fn fit_recognizer(labels: &[usize], groups: usize, features: &[RowFeature]) -> Result<(Recognizer, u64), BehaviorError> {
    let (tree, bits) = fit_lengths(labels, groups, features, Lengths::Exact)?;
    Ok((tree, bits as u64))
}

// ------------------------------------------------------------------------------------------ fitting

/// What the search needs from a subprogram fitter.
pub trait GroupFitter {
    type Program: Clone;
    type Certificate: Clone + fmt::Debug;

    /// The number of rows of the family.
    fn rows(&self) -> usize;

    /// A subprogram fitted to exactly `rows` (ascending, nonempty) and its screened data bits on
    /// them. `within`, when given, is the group `rows` came from, less its removed rows: a start
    /// for a fitter that searches from a program, sufficient statistics for one that keeps them.
    fn fit(&mut self, rows: &[usize], within: Option<Within<'_, Self::Program>>) -> Result<(Self::Program, f64), BehaviorError>;

    /// The program's screened data bits, `n KL(model_r ‖ program_r)/ln 2`, on every row.
    fn row_bits(&mut self, program: &Self::Program) -> Result<Vec<f64>, BehaviorError>;

    /// The program's code, operator by operator, for the library.
    fn parts(&self, program: &Self::Program) -> Result<ProgramParts, BehaviorError>;

    /// The program's certified fidelity and data bits on `rows`.
    fn certify(&mut self, program: &Self::Program, rows: &[usize]) -> Result<(Self::Certificate, f64), BehaviorError>;
}

/// The group a fit's rows came from: its program, and its rows that the fit's rows exclude.
pub struct Within<'a, P> {
    pub program: &'a P,
    /// Ascending.
    pub removed: &'a [usize],
}

struct Fitted<P> {
    program: P,
    parts: ProgramParts,
    /// Screened data bits on the rows it was fitted to.
    data_bits: f64,
    /// Screened data bits on every row, computed when a reassignment first needs them.
    row_bits: OnceCell<Vec<f64>>,
}

/// The itemised code of a partition.
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionCode {
    /// `L_int(K)`.
    pub count_bits: u64,
    pub library_bits: u64,
    /// Per group, `L(P_k | library)`.
    pub program_bits: Vec<u64>,
    /// Per group, its rows' data bits.
    pub data_bits: Vec<f64>,
    /// `L(Π)` (integral when exact).
    pub partition_bits: f64,
}

impl PartitionCode {
    pub fn total(&self) -> f64 {
        (self.count_bits + self.library_bits + self.program_bits.iter().sum::<u64>()) as f64
            + self.partition_bits
            + self.data_bits.iter().sum::<f64>()
    }
}

struct State<P> {
    labels: Vec<usize>,
    members: Vec<Vec<usize>>,
    groups: Vec<Rc<Fitted<P>>>,
    recognizer: Recognizer,
    code: PartitionCode,
}

impl<P> Clone for State<P> {
    fn clone(&self) -> Self {
        Self {
            labels: self.labels.clone(),
            members: self.members.clone(),
            groups: self.groups.clone(),
            recognizer: self.recognizer.clone(),
            code: self.code.clone(),
        }
    }
}

/// A seed: a row set proposed as a behaviour, with where it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seed {
    pub rows: Vec<usize>,
    pub description: String,
}

/// One test of a recognition rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleTest {
    pub feature: String,
    pub value: u32,
    pub equal: bool,
}

/// One leaf of the recognizer whose default group is this behaviour.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RulePath {
    pub tests: Vec<RuleTest>,
    /// Rows reaching the leaf.
    pub leaf_rows: usize,
    /// Of those, rows of this behaviour.
    pub own_rows: usize,
}

/// One discovered behaviour.
#[derive(Clone, Debug)]
pub struct Behaviour<P, C> {
    /// The recognizer's leaves that default to it (an empty list: it is coded as exceptions).
    pub rule: Vec<RulePath>,
    pub rows: Vec<usize>,
    pub program: P,
    /// Its message on its own.
    pub own_program_bits: u64,
    /// Its message given the library.
    pub program_bits: u64,
    /// Its rows' certified data bits.
    pub data_bits: f64,
    pub certificate: C,
    /// The group with the longest program: the general model on everything else.
    pub general: bool,
}

/// What [`discover`] returns.
#[derive(Clone, Debug)]
pub struct Discovery<P, C> {
    pub behaviours: Vec<Behaviour<P, C>>,
    pub recognizer: Recognizer,
    /// The exact code of the discovered partition, data bits certified.
    pub code: PartitionCode,
    /// The exact code of the one-group description, data bits certified.
    pub one_group: PartitionCode,
    /// Subprograms fitted during the search.
    pub fits: usize,
    /// Seeds whose screened saving was positive and that were therefore evaluated in full.
    pub evaluated_seeds: usize,
    /// The moves kept, in order.
    pub moves: Vec<String>,
}

struct Search<'a, F: GroupFitter> {
    fitter: &'a mut F,
    features: &'a [RowFeature],
    memo: BTreeMap<Vec<usize>, Rc<Fitted<F::Program>>>,
    fits: usize,
}

/// The ascending rows of `rows` not in the ascending `removed`.
fn difference(rows: &[usize], removed: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(rows.len());
    let mut j = 0;
    for &row in rows {
        while j < removed.len() && removed[j] < row {
            j += 1;
        }
        if j >= removed.len() || removed[j] != row {
            out.push(row);
        }
    }
    out
}

impl<F: GroupFitter> Search<'_, F> {
    fn fitted(&mut self, rows: Vec<usize>, within: Option<Within<'_, F::Program>>) -> Result<Rc<Fitted<F::Program>>, BehaviorError> {
        if let Some(found) = self.memo.get(&rows) {
            return Ok(Rc::clone(found));
        }
        let (program, data_bits) = self.fitter.fit(&rows, within)?;
        let parts = self.fitter.parts(&program)?;
        self.fits += 1;
        let fitted = Rc::new(Fitted { program, parts, data_bits, row_bits: OnceCell::new() });
        self.memo.insert(rows, Rc::clone(&fitted));
        Ok(fitted)
    }

    fn row_bits<'f>(&mut self, fitted: &'f Fitted<F::Program>) -> Result<&'f [f64], BehaviorError> {
        if fitted.row_bits.get().is_none() {
            let bits = self.fitter.row_bits(&fitted.program)?;
            if bits.len() != self.fitter.rows() {
                return Err(BehaviorError::Fit(format!("{} row costs for {} rows", bits.len(), self.fitter.rows())));
            }
            fitted.row_bits.get_or_init(|| bits);
        }
        fitted.row_bits.get().map(Vec::as_slice).ok_or_else(|| BehaviorError::Fit("row costs were not kept".to_string()))
    }

    /// Keep only the fits of `state`'s groups.
    fn forget_except(&mut self, state: &State<F::Program>) {
        let keep: BTreeSet<&Vec<usize>> = state.members.iter().collect();
        self.memo.retain(|rows, _| keep.contains(rows));
    }

    /// The state of `labels`: empty groups removed, groups renumbered by first row, each fitted.
    fn state(&mut self, labels: &[usize]) -> Result<State<F::Program>, BehaviorError> {
        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for &label in labels {
            let next = map.len();
            map.entry(label).or_insert(next);
        }
        let labels: Vec<usize> = labels.iter().map(|l| map[l]).collect();
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); map.len()];
        for (row, &label) in labels.iter().enumerate() {
            members[label].push(row);
        }
        let mut groups = Vec::with_capacity(members.len());
        for rows in &members {
            groups.push(self.fitted(rows.clone(), None)?);
        }
        let (recognizer, partition_bits) = fit_lengths(&labels, groups.len(), self.features, Lengths::Screened)?;
        let parts: Vec<&ProgramParts> = groups.iter().map(|g| &g.parts).collect();
        let (library_bits, program_bits) = joint_program_bits(&parts)?;
        let code = PartitionCode {
            count_bits: prefix_integer_len_bits(groups.len() as u64)?,
            library_bits,
            program_bits,
            data_bits: groups.iter().map(|g| g.data_bits).collect(),
            partition_bits,
        };
        Ok(State { labels, members, groups, recognizer, code })
    }

    /// Move rows to their cheapest group, refit, until the total stops dropping.
    fn relax(&mut self, mut state: State<F::Program>) -> Result<State<F::Program>, BehaviorError> {
        loop {
            let groups = state.groups.len();
            let leaves = state.recognizer.leaf_count();
            let leaf_of: Vec<usize> =
                (0..state.labels.len()).map(|row| state.recognizer.leaf_of(self.features, row)).collect();
            let mut counts = vec![vec![0usize; groups]; leaves];
            for (row, &label) in state.labels.iter().enumerate() {
                counts[leaf_of[row]][label] += 1;
            }
            let mut bits: Vec<&[f64]> = Vec::with_capacity(groups);
            let fitted: Vec<Rc<Fitted<F::Program>>> = state.groups.clone();
            for group in &fitted {
                bits.push(self.row_bits(group)?);
            }
            // The screened label cost: the adaptive (Krichevsky–Trofimov) code of the label in its
            // leaf, the per-row rate of the leaf's enumerative code.
            let mut labels = state.labels.clone();
            for (row, label) in labels.iter_mut().enumerate() {
                let leaf = &counts[leaf_of[row]];
                let n: usize = leaf.iter().sum();
                let cost = |k: usize| bits[k][row] - ((leaf[k] as f64 + 0.5) / (n as f64 + groups as f64 / 2.0)).log2();
                let mut best = *label;
                for k in 0..groups {
                    if cost(k) < cost(best) {
                        best = k;
                    }
                }
                *label = best;
            }
            if labels == state.labels {
                return Ok(state);
            }
            let candidate = self.state(&labels)?;
            if candidate.code.total() < state.code.total() {
                state = candidate;
            } else {
                return Ok(state);
            }
        }
    }

    /// The screened saving of carving `seed` out of the groups that hold it: every touched
    /// group's program and data bits before, less its refit on the rows it keeps and the seed's
    /// own program and data bits. The library, the count and the partition code are left to the
    /// full evaluation; the screen ranks and admits, it does not accept.
    fn screen(&mut self, state: &State<F::Program>, seed: &[usize]) -> Result<f64, BehaviorError> {
        let carved = self.fitted(seed.to_vec(), None)?;
        let mut saving = -(carved.parts.bits as f64 + carved.data_bits);
        let touched: BTreeSet<usize> = seed.iter().map(|r| state.labels[*r]).collect();
        for group in touched {
            let before = &state.groups[group];
            saving += before.parts.bits as f64 + before.data_bits;
            let remaining = difference(&state.members[group], seed);
            if !remaining.is_empty() {
                let removed: Vec<usize> = seed.iter().copied().filter(|r| state.labels[*r] == group).collect();
                let after = self.fitted(remaining, Some(Within { program: &before.program, removed: &removed }))?;
                saving -= after.parts.bits as f64 + after.data_bits;
            }
        }
        Ok(saving)
    }
}

/// Discover the behaviours of the model the fitter wraps (module note), seeds in the caller's
/// order.
pub fn discover<F: GroupFitter>(
    fitter: &mut F,
    features: &[RowFeature],
    seeds: &[Seed],
) -> Result<Discovery<F::Program, F::Certificate>, BehaviorError> {
    let rows = fitter.rows();
    if rows == 0 {
        return Err(BehaviorError::Declaration("an empty family".to_string()));
    }
    for feature in features {
        if feature.values.len() != rows || feature.values.iter().any(|v| *v as usize >= feature.alphabet) {
            return Err(BehaviorError::Declaration(format!("feature {} does not cover the rows", feature.name)));
        }
    }
    let mut seed_rows: Vec<Vec<usize>> = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let mut set = seed.rows.clone();
        set.sort_unstable();
        set.dedup();
        if set.is_empty() || set.iter().any(|r| *r >= rows) {
            return Err(BehaviorError::Declaration(format!("seed {} is not a row set", seed.description)));
        }
        seed_rows.push(set);
    }
    let mut search = Search { fitter, features, memo: BTreeMap::new(), fits: 0 };
    let one = search.state(&vec![0; rows])?;
    let mut state = one.clone();
    let mut moves: Vec<String> = Vec::new();
    // A seed is screened once per kept state; `version` counts the kept moves.
    let mut version = 0u64;
    let mut tried: BTreeSet<(u64, usize)> = BTreeSet::new();
    let mut evaluated_seeds = 0;
    loop {
        let mut kept = false;
        for (index, seed) in seeds.iter().enumerate() {
            if !tried.insert((version, index)) || state.members.contains(&seed_rows[index]) {
                continue;
            }
            let saving = search.screen(&state, &seed_rows[index])?;
            if saving <= 0.0 {
                search.forget_except(&state);
                continue;
            }
            evaluated_seeds += 1;
            let fresh = state.groups.len();
            let mut labels = state.labels.clone();
            for &row in &seed_rows[index] {
                labels[row] = fresh;
            }
            let candidate = search.state(&labels)?;
            let candidate = search.relax(candidate)?;
            if candidate.code.total() < state.code.total() {
                moves.push(format!(
                    "seed {}: {} groups, {:.1} -> {:.1} bits",
                    seed.description,
                    candidate.groups.len(),
                    state.code.total(),
                    candidate.code.total()
                ));
                state = candidate;
                version += 1;
                kept = true;
            }
            search.forget_except(&state);
        }
        let mut merged = true;
        while merged {
            merged = false;
            'pairs: for j in 0..state.groups.len() {
                for k in j + 1..state.groups.len() {
                    let labels: Vec<usize> = state.labels.iter().map(|&l| if l == k { j } else { l }).collect();
                    let candidate = search.state(&labels)?;
                    let candidate = search.relax(candidate)?;
                    if candidate.code.total() < state.code.total() {
                        moves.push(format!(
                            "merge {j} and {k}: {} groups, {:.1} -> {:.1} bits",
                            candidate.groups.len(),
                            state.code.total(),
                            candidate.code.total()
                        ));
                        state = candidate;
                        version += 1;
                        merged = true;
                        kept = true;
                        search.forget_except(&state);
                        break 'pairs;
                    }
                    search.forget_except(&state);
                }
            }
        }
        let relaxed = search.relax(state.clone())?;
        if relaxed.code.total() < state.code.total() {
            state = relaxed;
            version += 1;
            kept = true;
        }
        if !kept {
            break;
        }
    }
    let fits = search.fits;
    report(search.fitter, features, state, one, fits, evaluated_seeds, moves)
}

/// Certify every group and render its rule.
fn report<F: GroupFitter>(
    fitter: &mut F,
    features: &[RowFeature],
    mut state: State<F::Program>,
    mut one: State<F::Program>,
    fits: usize,
    evaluated_seeds: usize,
    moves: Vec<String>,
) -> Result<Discovery<F::Program, F::Certificate>, BehaviorError> {
    let certified_code = |fitter: &mut F, state: &mut State<F::Program>| -> Result<(PartitionCode, Vec<F::Certificate>), BehaviorError> {
        let (recognizer, partition_bits) = fit_recognizer(&state.labels, state.groups.len(), features)?;
        state.recognizer = recognizer;
        state.code.partition_bits = partition_bits as f64;
        let mut code = state.code.clone();
        let mut certificates = Vec::new();
        for (k, group) in state.groups.iter().enumerate() {
            let (certificate, data_bits) = fitter.certify(&group.program, &state.members[k])?;
            code.data_bits[k] = data_bits;
            certificates.push(certificate);
        }
        Ok((code, certificates))
    };
    let (one_group, _) = certified_code(fitter, &mut one)?;
    let (code, certificates) = certified_code(fitter, &mut state)?;
    let paths = state.recognizer.paths();
    let groups = state.groups.len();
    let mut leaf_counts = vec![vec![0usize; groups]; paths.len()];
    for (row, &label) in state.labels.iter().enumerate() {
        leaf_counts[state.recognizer.leaf_of(features, row)][label] += 1;
    }
    let longest = (0..groups).max_by_key(|k| (state.groups[*k].parts.bits, std::cmp::Reverse(*k)));
    let mut behaviours = Vec::new();
    for (k, (group, certificate)) in state.groups.iter().zip(certificates).enumerate() {
        let rows = state.members[k].clone();
        let mut rule = Vec::new();
        for ((tests, default), counts) in paths.iter().zip(&leaf_counts) {
            if *default != k {
                continue;
            }
            rule.push(RulePath {
                tests: tests
                    .iter()
                    .map(|(f, value, equal)| RuleTest { feature: features[*f].name.clone(), value: *value, equal: *equal })
                    .collect(),
                leaf_rows: counts.iter().sum(),
                own_rows: counts[k],
            });
        }
        behaviours.push(Behaviour {
            rule,
            rows,
            program: group.program.clone(),
            own_program_bits: group.parts.bits,
            program_bits: code.program_bits[k],
            data_bits: code.data_bits[k],
            certificate,
            general: groups > 1 && longest == Some(k),
        });
    }
    Ok(Discovery { behaviours, recognizer: state.recognizer, code, one_group, fits, evaluated_seeds, moves })
}

/// One seed per value of every feature: the rows where `feature == value`, for the values that
/// occur on at least two rows and not on every row (a group of one row cannot pay for a program).
pub fn atom_seeds(features: &[RowFeature]) -> Vec<Seed> {
    let mut out = Vec::new();
    for feature in features {
        let mut by_value: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (row, value) in feature.values.iter().enumerate() {
            by_value.entry(*value).or_default().push(row);
        }
        for (value, rows) in by_value {
            if rows.len() > 1 && rows.len() < feature.values.len() {
                out.push(Seed { rows, description: format!("{} = {value}", feature.name) });
            }
        }
    }
    out
}

impl RulePath {
    /// `slot0 = 3 ∧ slot1 ≠ 5`, or `always` for the root leaf.
    pub fn render(&self) -> String {
        if self.tests.is_empty() {
            return "always".to_string();
        }
        self.tests
            .iter()
            .map(|t| format!("{} {} {}", t.feature, if t.equal { "=" } else { "≠" }, t.value))
            .collect::<Vec<_>>()
            .join(" ∧ ")
    }
}

//! Exact normalization of an operator program by equality saturation (#2951).
//!
//! An [`OperatorProgram`] is loaded into an e-graph ([`egg`]) whose language is the program's
//! node kinds with the operators as numeric leaves ([`Term`]). The exact algebraic laws below are
//! applied until nothing new is learned (a fixpoint) or until the leaves would outgrow the memory
//! the governor grants (a derived resource bound), and the shortest program is then extracted,
//! priced by the program's own decoded message length.
//!
//! # Values and leaves
//!
//! A value class is a function of the program's inputs with an [`Interface`]; `Apply(A, x)` is
//! `x Aᵀ` row by row, `Sum` adds values of one interface, and `Constant(b)` broadcasts the column
//! `b`. An operator class holds only [`Term::Leaf`] nodes, each an [`Operator`] in the leaf arena
//! ([`Leaves`]) with an entrywise enclosure `|center − exact| ≤ radius` of the exact operator it
//! stands for. A native operator is its own exact value (radius zero).
//!
//! A leaf derived by a law (a product `A B`, a sum `A + B`, a scaled constant row, a diagonal, a
//! routed bias table, a pointwise law of a constant) is computed in binary64 with its forward-error
//! band (Higham's `γ_k` for a `k`-term inner product, plus the propagation of the inputs' radii).
//! When both inputs are exact and lie on dyadic lattices `2^{-p}`, `2^{-q}`, the exact result lies
//! on a known lattice (`p + q` for a product, `max(p, q)` for a sum); if the band is below half
//! that lattice's step, the lattice point nearest the computed value is the exact result, so the
//! leaf is exact with radius zero and is sent on the coarsest lattice that holds it. Otherwise the
//! leaf is rounded to the lattice whose half-step is the band, and its radius is the band plus
//! that rounding.
//!
//! # Certified equality of leaves
//!
//! Two classes merge only on an equality witness: an exact law (a symbolic identity), or two
//! leaves whose exact values are proven equal, both exact (radius zero, on a dyadic lattice) and
//! equal entrywise. Intersecting enclosures are not a witness: they only fail to rule equality
//! out, and `(1 + 2⁻²⁷)(1 − 2⁻²⁷) = 1 − 2⁻⁵⁴` has an enclosure that holds `1` without being the
//! identity. Such pairs stay outside the congruence. A merged class keeps the intersection of the boxes; an exact law that unions two
//! classes whose boxes do not intersect is a band defect and stops saturation with
//! [`EgraphError::Inconsistent`]. No tolerance is involved.
//!
//! # Laws
//!
//! * transpose: `x A`, an operator read in its transposed orientation, is `Aᵀ` applied, so every
//!   law of an applied operator reaches it;
//! * compose: `A (B z) = (A B) z`, and `A b` for a constant `b`;
//! * distribute and fuse: `A (Σ u_i) = Σ A u_i`, `A x + B x = (A + B) x`, and constant biases add;
//! * push-through-mix: `A Σ_j α_j p_j = Σ_j α_j A p_j` in both directions, and
//!   `Σ_j α_j b_j = W α` for constant payloads (the bias table read by the routing weights), with
//!   the split `Σ_j α_j (x_j + b_j) = Σ_j α_j x_j + Σ_j α_j b_j`;
//! * bilinear constant side: `c ⟨a, r⟩ = (c aᵀ) r`; a Hadamard product with a constant is a
//!   diagonal operator;
//! * identities: the identity operator and the identity law vanish; ReLU, the identity and the
//!   zero law of a constant fold; a summand applying an exactly zero operator vanishes.
//!
//! * gains: a [`Node::Gain`] is `Scale(c, x)` with its coefficient a scalar term over the declared
//!   parameters, which stay symbols: a scale moves through an operator and a composition, numeric
//!   scales fold into the operator, and scaled summands of one value factor, so `m1·x − m2·x`
//!   becomes `(m1 − m2)·x` and never `0` although it vanishes at the native setting `m = 1`;
//! * rules: a [`Node::Call`] is its body on the call's arguments, inlined when loaded, so every
//!   application of a rule shares the body's classes.
//!
//! Every law is exact for every input and every parameter setting, not only on a declared family
//! at the native setting.
//!
//! # The unit gauge
//!
//! A gauge orbit has no shortest element under rewriting in both directions, so the unit gauge
//! moves (merging, rescaling, reordering) enter as one exact canonical form applied before saturation
//! ([`canonical_units`]): a ReLU layer's units whose read rows (every term and the bias) are equal
//! merge into one unit with the summed write column; a unit that reads nothing is removed; each
//! remaining unit is moved by the exact power of two that puts its bias's magnitude in `[1, 2)`, or
//! with a zero bias its read row's Euclidean norm (every term and the bias)
//! (`relu(2^e t) = 2^e relu(t)`); and the units are ordered by their bias, then their read row's
//! squared norm. The bias is invariant under every invertible change of the read coordinates and
//! the norm under an orthogonal one, so one layer written in two bases of a space with no norm, or
//! (with zero biases) two orthonormal bases, gets one canonical unit gauge.
//!
//! # Extraction
//!
//! The decoded message length is not additive over a term (operators are sent once and referenced
//! by index), so extraction runs a proxy: each class's cheapest node, with a leaf charged its
//! decoded bits unless it is already paid for. Each candidate is lowered to a program and priced
//! by [`OperatorProgram::code_bits`]; the leaves of the shortest candidate are then marked paid and
//! extraction repeats until the chosen leaf set recurs. The native program is always a candidate,
//! so normalization never lengthens the message.

use super::codec::fixed_index_len_bits;
use super::operator_program::{
    Basis, Coefficient, Declarations, Group, Interface, Label, LabelKind, Law, Node, Operator, OperatorBody,
    OperatorProgram, ProgramError, Provenance, Rotary, Scale, Slot, band_precision, exact_precision, round_to_lattice,
};
use super::precision::DeclaredPrecision;
use egg::{Analysis, DidMerge, EGraph, Id, Language};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array2, s};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// An index into the leaf arena.
pub type LeafId = u32;

/// The pointwise laws in a fixed order, so a law list is an ordered key.
const LAW_ORDER: [Law; 6] = [Law::Relu, Law::Identity, Law::Zero, Law::Silu, Law::Gelu, Law::GeluTanh];

fn law_code(law: Law) -> u8 {
    LAW_ORDER.iter().position(|l| *l == law).map_or(0, |at| at as u8)
}

fn law_of(code: u8) -> Law {
    LAW_ORDER[usize::from(code) % LAW_ORDER.len()]
}

fn scale_code(scale: Scale) -> Option<u32> {
    match scale {
        Scale::One => None,
        Scale::InverseSqrt(n) => Some(n),
    }
}

fn scale_of(code: Option<u32>) -> Scale {
    code.map_or(Scale::One, Scale::InverseSqrt)
}

/// A node of the e-graph language: the program's node kinds, with operators as leaves.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Term {
    /// An operator of the leaf arena.
    Leaf(LeafId),
    Feature { slot: usize, basis: usize },
    Raw { slot: usize },
    /// `[operator]`: the operator's single column on every input.
    Constant([Id; 1]),
    /// `[operator, value]`: `x Aᵀ`.
    Apply([Id; 2]),
    /// Values of one interface, added.
    Sum(Vec<Id>),
    Bilinear { scale: Option<u32>, args: [Id; 2] },
    Softmax(Vec<Id>),
    /// `args[0]` is the weights; payload `i` is `args[i + 1]`, read with weight column `columns[i]`.
    Mix { columns: Vec<usize>, args: Vec<Id> },
    Pointwise { laws: Vec<u8>, args: [Id; 1] },
    Hadamard([Id; 2]),
    Readout { basis: usize, args: [Id; 1] },
    Outer([Id; 2]),
    Concat(Vec<Id>),
    /// `[query, key, value]` attention over each row's sequence; `rotary` is `(base, dims,
    /// half_split)`. No law rewrites it: it is carried through exactly.
    Attend { scale: Option<u32>, rotary: Option<(u32, u32, bool)>, causal: bool, args: [Id; 3] },
    /// `x / √(mean(x²) + ε)` per row, `ε` by its bits; carried through exactly.
    RmsNorm { epsilon: u64, args: [Id; 1] },
    /// `[operator, value]`: `x A`, the operator read in its transposed orientation.
    Transposed([Id; 2]),
    /// A declared parameter (a control gain, an edit's strength) read by a [`Node::Gain`]: a scalar
    /// symbol that is never evaluated at its native setting.
    Parameter(u32),
    /// A scalar held exactly in binary64 (its bits; zero is `+0`).
    Number(u64),
    /// Scalars added.
    ScalarSum(Vec<Id>),
    /// Two scalars multiplied.
    ScalarProduct([Id; 2]),
    /// `[scalar, value]`: the value times the scalar.
    Scale([Id; 2]),
}

impl Language for Term {
    type Discriminant = std::mem::Discriminant<Term>;

    fn discriminant(&self) -> Self::Discriminant {
        std::mem::discriminant(self)
    }

    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(a), Self::Leaf(b)) => a == b,
            (Self::Feature { .. }, Self::Feature { .. }) | (Self::Raw { .. }, Self::Raw { .. }) => self == other,
            (Self::Constant(_), Self::Constant(_))
            | (Self::Apply(_), Self::Apply(_))
            | (Self::Hadamard(_), Self::Hadamard(_))
            | (Self::Outer(_), Self::Outer(_)) => true,
            (Self::Sum(a), Self::Sum(b)) | (Self::Softmax(a), Self::Softmax(b)) | (Self::Concat(a), Self::Concat(b)) => {
                a.len() == b.len()
            }
            (Self::Bilinear { scale: a, .. }, Self::Bilinear { scale: b, .. }) => a == b,
            (Self::Mix { columns: a, args: x }, Self::Mix { columns: b, args: y }) => a == b && x.len() == y.len(),
            (Self::Pointwise { laws: a, .. }, Self::Pointwise { laws: b, .. }) => a == b,
            (Self::Readout { basis: a, .. }, Self::Readout { basis: b, .. }) => a == b,
            (Self::Attend { scale: a, rotary: r, causal: c, .. }, Self::Attend { scale: b, rotary: q, causal: d, .. }) => {
                a == b && r == q && c == d
            }
            (Self::RmsNorm { epsilon: a, .. }, Self::RmsNorm { epsilon: b, .. }) => a == b,
            (Self::Transposed(_), Self::Transposed(_)) => true,
            (Self::Parameter(a), Self::Parameter(b)) => a == b,
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::ScalarSum(a), Self::ScalarSum(b)) => a.len() == b.len(),
            (Self::ScalarProduct(_), Self::ScalarProduct(_)) | (Self::Scale(_), Self::Scale(_)) => true,
            _ => false,
        }
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Leaf(_) | Self::Feature { .. } | Self::Raw { .. } | Self::Parameter(_) | Self::Number(_) => &[],
            Self::Constant(a) | Self::Pointwise { args: a, .. } | Self::Readout { args: a, .. } | Self::RmsNorm { args: a, .. } => a,
            Self::Apply(a) | Self::Hadamard(a) | Self::Outer(a) | Self::Bilinear { args: a, .. } | Self::Transposed(a) => a,
            Self::ScalarProduct(a) | Self::Scale(a) => a,
            Self::Attend { args: a, .. } => a,
            Self::Sum(a) | Self::Softmax(a) | Self::Concat(a) | Self::Mix { args: a, .. } | Self::ScalarSum(a) => a,
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Leaf(_) | Self::Feature { .. } | Self::Raw { .. } | Self::Parameter(_) | Self::Number(_) => &mut [],
            Self::Constant(a) | Self::Pointwise { args: a, .. } | Self::Readout { args: a, .. } | Self::RmsNorm { args: a, .. } => a,
            Self::Apply(a) | Self::Hadamard(a) | Self::Outer(a) | Self::Bilinear { args: a, .. } | Self::Transposed(a) => a,
            Self::ScalarProduct(a) | Self::Scale(a) => a,
            Self::Attend { args: a, .. } => a,
            Self::Sum(a) | Self::Softmax(a) | Self::Concat(a) | Self::Mix { args: a, .. } | Self::ScalarSum(a) => a,
        }
    }
}

/// A refused normalization.
#[derive(Debug)]
pub enum EgraphError {
    Program(ProgramError),
    /// Two classes an exact law equates have disjoint enclosures, or a node is ill-typed: a defect
    /// of a band or of a law, never a property of the input.
    Inconsistent(String),
}

impl fmt::Display for EgraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(f, "equality saturation: {error}"),
            Self::Inconsistent(message) => write!(f, "equality saturation: inconsistent e-graph: {message}"),
        }
    }
}

impl std::error::Error for EgraphError {}

impl From<ProgramError> for EgraphError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

/// `x (1 + 2γ_{k+2})`, rounded up: the outward inflation of a computed nonnegative sum of `k`
/// terms, as in the program's execution bands.
pub(crate) fn outward(value: f64, terms: usize) -> f64 {
    (value * (1.0 + 2.0 * accumulation_growth(terms + 2))).next_up()
}

/// One operator of the arena.
#[derive(Clone, Debug)]
pub struct Leaf {
    pub operator: Operator,
    /// The operator's matrix as executed.
    pub center: Array2<f64>,
    /// `|center − exact| ≤ radius`, entrywise.
    pub radius: Array2<f64>,
    /// The fraction bits of a lattice holding the exact value, when the leaf is exact.
    pub lattice: Option<i32>,
    /// The operator's decoded message length (structure and reals).
    pub bits: u64,
}

impl Leaf {
    pub fn is_exact(&self) -> bool {
        self.lattice.is_some()
    }
}

/// How a derived leaf was formed: the memo key that keeps a law from minting a new leaf for a
/// derivation it has already made.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Derivation {
    Product(LeafId, LeafId),
    Transpose(LeafId),
    Sum(LeafId, LeafId),
    ScaledRow { leaf: LeafId, scale: Option<u32>, cols: Interface },
    ScalarIdentity { value: u64, interface: Interface },
    Diagonal { leaf: LeafId, rows: Interface, cols: Interface },
    Routed { columns: Vec<usize>, leaves: Vec<LeafId>, cols: Interface },
    Pointwise { leaf: LeafId, laws: Vec<u8> },
}

/// A leaf a law formed: its memo key, interfaces, computed value, entrywise band, the lattice
/// holding its exact value when every input is exact, and its provenance.
struct Formed {
    key: Derivation,
    name: String,
    rows: Interface,
    cols: Interface,
    value: Array2<f64>,
    band: Array2<f64>,
    lattice: Option<i32>,
    provenance: Provenance,
}

/// What an e-class is.
#[derive(Clone, Debug)]
pub enum ClassData {
    Operator(OperatorClass),
    Value(Interface),
    /// A scalar, with its exact value when it holds no parameter and every step was exact.
    Scalar(Option<f64>),
    /// An ill-typed node (recorded as a fault).
    Fault,
}

/// An operator class: its interfaces, its cheapest leaf and the intersection of its members'
/// enclosures.
#[derive(Clone, Debug)]
pub struct OperatorClass {
    pub rows: Interface,
    pub cols: Interface,
    pub best: LeafId,
    pub best_bits: u64,
    pub lower: Array2<f64>,
    pub upper: Array2<f64>,
}

impl OperatorClass {
    /// Whether the class is exactly the identity.
    fn is_identity(&self) -> bool {
        self.rows == self.cols
            && self.lower.indexed_iter().all(|((r, c), v)| *v == if r == c { 1.0 } else { 0.0 })
            && self.lower == self.upper
    }

    /// Whether the class's exact value is known: its enclosure is a single point.
    fn is_exact(&self) -> bool {
        self.lower == self.upper
    }

    fn intersects(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && ndarray::Zip::from(&self.lower)
                .and(&self.upper)
                .and(&other.lower)
                .and(&other.upper)
                .all(|a, b, c, d| a.max(*c) <= b.min(*d))
    }
}

/// The e-graph analysis: the leaf arena, the declarations interfaces are read from, and the
/// faults found while building classes.
#[derive(Debug)]
pub struct Leaves {
    pub declarations: Declarations,
    pub bases: Vec<Basis>,
    pub leaves: Vec<Leaf>,
    memo: HashMap<Derivation, LeafId>,
    /// Ill-typed nodes and disjoint enclosures, reported as [`EgraphError::Inconsistent`].
    pub faults: Vec<String>,
    /// Bytes held by leaves and class enclosures.
    pub bytes: usize,
}

/// The e-graph of a program.
pub type ProgramGraph = EGraph<Term, Leaves>;

impl Leaves {
    fn new(declarations: Declarations, bases: Vec<Basis>) -> Self {
        Self { declarations, bases, leaves: Vec::new(), memo: HashMap::new(), faults: Vec::new(), bytes: 0 }
    }

    fn push(&mut self, leaf: Leaf) -> LeafId {
        self.bytes += 4 * leaf.center.len() * std::mem::size_of::<f64>();
        self.leaves.push(leaf);
        (self.leaves.len() - 1) as LeafId
    }

    /// A native operator: its own exact value, except a low-rank body, whose executed product is
    /// banded (exact when the product's lattice resolves it).
    fn native(&mut self, operator: Operator) -> Result<LeafId, EgraphError> {
        let center = operator.matrix();
        let (structure, reals) = operator.code_bits()?;
        let (radius, lattice) = match &operator.body {
            OperatorBody::LowRank { left, right, precision } => {
                let band = left.mapv(f64::abs).dot(&right.mapv(f64::abs)).mapv(|v| outward(accumulation_growth(left.ncols()) * v, left.ncols()));
                match exact_on(&center, &band, 2 * precision.fraction_bits())? {
                    Some(rounded) if rounded == center => {
                        let lattice = exact_precision(center.iter().copied())?.fraction_bits();
                        (Array2::zeros(center.dim()), Some(lattice))
                    }
                    _ => (band, None),
                }
            }
            OperatorBody::Dense { .. } | OperatorBody::Identity | OperatorBody::Diagonal { .. } => {
                let lattice = exact_precision(center.iter().copied())?.fraction_bits();
                (Array2::zeros(center.dim()), Some(lattice))
            }
        };
        Ok(self.push(Leaf { operator, center, radius, lattice, bits: structure + reals }))
    }

    /// Mint (or recall) the leaf a law formed.
    fn derive(&mut self, formed: Formed) -> Result<LeafId, EgraphError> {
        let Formed { key, name, rows, cols, value, band, lattice, provenance } = formed;
        if let Some(&leaf) = self.memo.get(&key) {
            return Ok(leaf);
        }
        let exact = match lattice {
            Some(bits) => exact_on(&value, &band, bits)?,
            None => None,
        };
        let present = |values: &Array2<f64>| {
            Array2::from_shape_fn((rows.group_count(), cols.group_count()), |(r, c)| {
                values.slice(s![rows.range(r), cols.range(c)]).iter().any(|v| *v != 0.0)
            })
        };
        let leaf = match exact {
            Some(rounded) => {
                let lattice = exact_precision(rounded.iter().copied())?;
                let operator =
                    Operator::blocks(name, rows.clone(), cols.clone(), rounded.clone(), present(&rounded), lattice, provenance)?;
                let (structure, reals) = operator.code_bits()?;
                let radius = Array2::zeros(rounded.dim());
                Leaf { operator, center: rounded, radius, lattice: Some(lattice.fraction_bits()), bits: structure + reals }
            }
            None => {
                let largest = value.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let widest = band.iter().fold(0.0_f64, |m, v| m.max(*v));
                let resolution = band_precision(widest, largest)?;
                let operator = Operator::blocks(name, rows.clone(), cols.clone(), value.clone(), present(&value), resolution, provenance)?;
                let center = operator.matrix();
                let mut radius = band.clone();
                ndarray::Zip::from(&mut radius).and(&center).and(&value).for_each(|r, c, v| *r = (*r + (c - v).abs()).next_up());
                let (structure, reals) = operator.code_bits()?;
                Leaf { operator, center, radius, lattice: None, bits: structure + reals }
            }
        };
        let id = self.push(leaf);
        self.memo.insert(key, id);
        Ok(id)
    }

    fn leaf(&self, id: LeafId) -> &Leaf {
        &self.leaves[id as usize]
    }

    /// `A B`.
    fn product(&mut self, a: LeafId, b: LeafId) -> Result<LeafId, EgraphError> {
        let (la, lb) = (self.leaf(a), self.leaf(b));
        let inner = la.center.ncols();
        let value = la.center.dot(&lb.center);
        let (abs_a, abs_b) = (la.center.mapv(f64::abs), lb.center.mapv(f64::abs));
        let mut band = abs_a.dot(&abs_b) * accumulation_growth(inner);
        band += &abs_a.dot(&lb.radius);
        band += &la.radius.dot(&abs_b);
        band += &la.radius.dot(&lb.radius);
        band.mapv_inplace(|v| outward(v, 4 * inner));
        let lattice = la.lattice.zip(lb.lattice).map(|(p, q)| p + q);
        let provenance = Provenance::derived(&[&la.operator.provenance, &lb.operator.provenance], format!("{} {}", la.operator.name, lb.operator.name));
        let name = format!("{}·{}", la.operator.name, lb.operator.name);
        let (rows, cols) = (la.operator.rows.clone(), lb.operator.cols.clone());
        self.derive(Formed { key: Derivation::Product(a, b), name, rows, cols, value, band, lattice, provenance })
    }

    /// `Aᵀ`: the same reals on the same lattice, read the other way.
    fn transpose(&mut self, a: LeafId) -> Result<LeafId, EgraphError> {
        let la = self.leaf(a);
        let value = la.center.t().to_owned();
        let band = la.radius.t().to_owned();
        let lattice = la.lattice;
        let provenance = Provenance::derived(&[&la.operator.provenance], format!("{} transposed", la.operator.name));
        let name = format!("{}ᵀ", la.operator.name);
        let (rows, cols) = (la.operator.cols.clone(), la.operator.rows.clone());
        self.derive(Formed { key: Derivation::Transpose(a), name, rows, cols, value, band, lattice, provenance })
    }

    /// `A + B` over one interface pair.
    fn sum(&mut self, a: LeafId, b: LeafId) -> Result<LeafId, EgraphError> {
        let (a, b) = (a.min(b), a.max(b));
        let (la, lb) = (self.leaf(a), self.leaf(b));
        let value = &la.center + &lb.center;
        let mut band = value.mapv(|v| UNIT_ROUNDOFF * v.abs());
        band += &la.radius;
        band += &lb.radius;
        band.mapv_inplace(|v| outward(v, 3));
        let lattice = la.lattice.zip(lb.lattice).map(|(p, q)| p.max(q));
        let provenance = Provenance::derived(&[&la.operator.provenance, &lb.operator.provenance], format!("{} + {}", la.operator.name, lb.operator.name));
        let name = format!("{}+{}", la.operator.name, lb.operator.name);
        let (rows, cols) = (la.operator.rows.clone(), la.operator.cols.clone());
        self.derive(Formed { key: Derivation::Sum(a, b), name, rows, cols, value, band, lattice, provenance })
    }

    /// `c aᵀ` for a constant column `a`, as a one-row operator onto `cols`.
    fn scaled_row(&mut self, a: LeafId, scale: Option<u32>, cols: Interface) -> Result<LeafId, EgraphError> {
        let la = self.leaf(a);
        let c = scale_of(scale).value();
        let value = la.center.t().mapv(|v| c * v);
        let band = ndarray::Zip::from(&value).and(&la.radius.t()).map_collect(|v, r| outward(UNIT_ROUNDOFF * v.abs() + c.abs() * r, 2));
        let scale_lattice = exact_precision([c])?.fraction_bits();
        let lattice = la.lattice.map(|p| p + scale_lattice);
        let provenance = Provenance::derived(&[&la.operator.provenance], format!("scale {c} times the constant side"));
        let name = format!("row·{}", la.operator.name);
        let rows = Interface::native(1)?;
        self.derive(Formed { key: Derivation::ScaledRow { leaf: a, scale, cols: cols.clone() }, name, rows, cols, value, band, lattice, provenance })
    }

    /// `c I` on `interface`, for an exact number `c`.
    fn scalar_identity(&mut self, c: f64, interface: Interface) -> Result<LeafId, EgraphError> {
        let width = interface.width();
        let value = Array2::from_diag_elem(width, c);
        let band = Array2::zeros((width, width));
        let lattice = Some(exact_precision([c])?.fraction_bits());
        let provenance = Provenance::derived(&[], format!("{c} times the identity"));
        let key = Derivation::ScalarIdentity { value: c.to_bits(), interface: interface.clone() };
        self.derive(Formed { key, name: format!("{c}·I"), rows: interface.clone(), cols: interface, value, band, lattice, provenance })
    }

    /// `diag(a)` from `cols` onto `rows` (equal widths).
    fn diagonal(&mut self, a: LeafId, rows: Interface, cols: Interface) -> Result<LeafId, EgraphError> {
        let la = self.leaf(a);
        let width = la.center.nrows();
        let mut value = Array2::<f64>::zeros((width, width));
        let mut band = Array2::<f64>::zeros((width, width));
        for i in 0..width {
            value[[i, i]] = la.center[[i, 0]];
            band[[i, i]] = la.radius[[i, 0]];
        }
        let lattice = la.lattice;
        let provenance = Provenance::derived(&[&la.operator.provenance], "the constant factor as a diagonal".to_string());
        let name = format!("diag·{}", la.operator.name);
        let key = Derivation::Diagonal { leaf: a, rows: rows.clone(), cols: cols.clone() };
        self.derive(Formed { key, name, rows, cols, value, band, lattice, provenance })
    }

    /// The bias table `W` with column `columns[j]` the constant payload `leaves[j]`, read by
    /// weights of interface `cols`.
    fn routed(&mut self, columns: Vec<usize>, leaves: Vec<LeafId>, cols: Interface) -> Result<LeafId, EgraphError> {
        let rows = self.leaf(leaves[0]).operator.rows.clone();
        let mut value = Array2::<f64>::zeros((rows.width(), cols.width()));
        let mut band = Array2::<f64>::zeros((rows.width(), cols.width()));
        let mut lattice = Some(i32::MIN);
        let mut parts = Vec::new();
        for (&column, &leaf) in columns.iter().zip(&leaves) {
            let l = self.leaf(leaf);
            value.column_mut(column).assign(&l.center.column(0));
            band.column_mut(column).assign(&l.radius.column(0));
            lattice = lattice.zip(l.lattice).map(|(p, q)| p.max(q));
            parts.push(&l.operator.provenance);
        }
        let lattice = lattice.map(|p| p.max(0));
        let provenance = Provenance::derived(&parts, "each payload's constant, read by its routing weight".to_string());
        let key = Derivation::Routed { columns, leaves, cols: cols.clone() };
        self.derive(Formed { key, name: "routed".to_string(), rows, cols, value, band, lattice, provenance })
    }

    /// The exact laws (ReLU, identity, zero) of a constant column, per group of its rows.
    fn pointwise(&mut self, a: LeafId, laws: Vec<u8>) -> Result<LeafId, EgraphError> {
        let la = self.leaf(a);
        let rows = la.operator.rows.clone();
        let mut value = la.center.clone();
        let mut band = la.radius.clone();
        for (group, &law) in laws.iter().enumerate() {
            for i in rows.range(group) {
                value[[i, 0]] = law_of(law).apply(value[[i, 0]]);
                if law_of(law) == Law::Zero {
                    band[[i, 0]] = 0.0;
                }
            }
        }
        let lattice = la.lattice;
        let provenance = Provenance::derived(&[&la.operator.provenance], "a pointwise law of a constant".to_string());
        let name = format!("law·{}", la.operator.name);
        let cols = Interface::constant();
        self.derive(Formed { key: Derivation::Pointwise { leaf: a, laws }, name, rows, cols, value, band, lattice, provenance })
    }

    fn operator_data(&self, leaf: LeafId) -> OperatorClass {
        let l = self.leaf(leaf);
        let lower = ndarray::Zip::from(&l.center).and(&l.radius).map_collect(|c, r| if *r == 0.0 { *c } else { (c - r).next_down() });
        let upper = ndarray::Zip::from(&l.center).and(&l.radius).map_collect(|c, r| if *r == 0.0 { *c } else { (c + r).next_up() });
        OperatorClass {
            rows: l.operator.rows.clone(),
            cols: l.operator.cols.clone(),
            best: leaf,
            best_bits: l.bits,
            lower,
            upper,
        }
    }
}

/// `a + b` when binary64 holds it exactly (the two-sum error is zero), zero as `+0`.
pub fn exact_add(a: f64, b: f64) -> Option<f64> {
    let sum = a + b;
    let virtual_b = sum - a;
    let error = (a - (sum - virtual_b)) + (b - virtual_b);
    (sum.is_finite() && error == 0.0).then_some(if sum == 0.0 { 0.0 } else { sum })
}

/// `a b` when binary64 holds it exactly (the fused product error is zero), zero as `+0`.
pub fn exact_mul(a: f64, b: f64) -> Option<f64> {
    let product = a * b;
    (product.is_finite() && a.mul_add(b, -product) == 0.0 && (product != 0.0 || a == 0.0 || b == 0.0))
        .then_some(if product == 0.0 { 0.0 } else { product })
}

/// The lattice point nearest `value` on `2^{-bits}` in every entry, when `band` is below half the
/// step everywhere, so that point is the exact value; `None` otherwise.
fn exact_on(value: &Array2<f64>, band: &Array2<f64>, bits: i32) -> Result<Option<Array2<f64>>, EgraphError> {
    let Ok(lattice) = DeclaredPrecision::new(bits) else { return Ok(None) };
    let half = lattice.worst_case_error();
    if band.iter().any(|b| *b >= half) {
        return Ok(None);
    }
    let mut rounded = value.clone();
    for v in rounded.iter_mut() {
        match round_to_lattice(*v, lattice) {
            Ok(point) => *v = point,
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(rounded))
}

/// The interface of a value node from its children's classes (the program's own typing rules).
fn value_interface(egraph: &ProgramGraph, term: &Term) -> Result<ClassData, String> {
    let value = |id: Id| match &egraph[id].data {
        ClassData::Value(interface) => Ok(interface.clone()),
        other => Err(format!("class {id} is not a value: {other:?}")),
    };
    let operator = |id: Id| match &egraph[id].data {
        ClassData::Operator(op) => Ok(op.clone()),
        other => Err(format!("class {id} is not an operator: {other:?}")),
    };
    let scalar = |id: Id| match &egraph[id].data {
        ClassData::Scalar(v) => Ok(*v),
        other => Err(format!("class {id} is not a scalar: {other:?}")),
    };
    let declarations = &egraph.analysis.declarations;
    let bases = &egraph.analysis.bases;
    let error = |e: ProgramError| e.to_string();
    let interface = match term {
        Term::Leaf(leaf) => return Ok(ClassData::Operator(egraph.analysis.operator_data(*leaf))),
        Term::Parameter(_) => return Ok(ClassData::Scalar(None)),
        Term::Number(bits) => return Ok(ClassData::Scalar(Some(f64::from_bits(*bits)))),
        Term::ScalarSum(terms) => {
            let mut total = Some(0.0);
            for term in terms {
                let value = scalar(*term)?;
                total = total.zip(value).and_then(|(a, b)| exact_add(a, b));
            }
            return Ok(ClassData::Scalar(total));
        }
        Term::ScalarProduct([a, b]) => {
            let (a, b) = (scalar(*a)?, scalar(*b)?);
            return Ok(ClassData::Scalar(a.zip(b).and_then(|(a, b)| exact_mul(a, b))));
        }
        Term::Scale([c, x]) => {
            scalar(*c)?;
            value(*x)?
        }
        Term::Feature { basis, .. } => {
            bases.get(*basis).ok_or_else(|| format!("basis {basis}"))?.interface(declarations).map_err(error)?
        }
        Term::Raw { slot } => match declarations.slots.get(*slot) {
            Some(Slot::Raw { width }) => Interface::native(*width).map_err(error)?,
            _ => return Err(format!("raw slot {slot}")),
        },
        Term::Constant([op]) => {
            let op = operator(*op)?;
            if op.cols != Interface::constant() {
                return Err("a constant is read from 1".to_string());
            }
            op.rows
        }
        Term::Apply([op, x]) => {
            let (op, x) = (operator(*op)?, value(*x)?);
            if op.cols != x {
                return Err("an applied operator does not read its argument's interface".to_string());
            }
            op.rows
        }
        Term::Sum(children) => {
            let first = value(*children.first().ok_or("an empty sum")?)?;
            for child in children {
                if value(*child)? != first {
                    return Err("a sum of different interfaces".to_string());
                }
            }
            first
        }
        Term::Bilinear { args: [l, r], .. } => {
            if value(*l)?.width() != value(*r)?.width() {
                return Err("bilinear widths differ".to_string());
            }
            Interface::native(1).map_err(error)?
        }
        Term::Softmax(scores) => Interface::uniform(scores.len(), 1, LabelKind::Position, 0).map_err(error)?,
        Term::Mix { args, .. } => value(*args.get(1).ok_or("a mix without payloads")?)?,
        Term::Pointwise { args: [x], .. } => value(*x)?,
        Term::Hadamard([l, r]) => {
            let left = value(*l)?;
            if left.width() != value(*r)?.width() {
                return Err("hadamard widths differ".to_string());
            }
            left
        }
        Term::Readout { basis, .. } => {
            let domain = match bases.get(*basis).ok_or_else(|| format!("basis {basis}"))? {
                Basis::Indicator { domain } | Basis::Characters { domain, .. } => *domain,
            };
            let size = declarations.domains.get(domain).ok_or_else(|| format!("domain {domain}"))?.size;
            Interface::uniform(size, 1, LabelKind::Token, 0).map_err(error)?
        }
        Term::Outer([l, r]) => {
            let (l, r) = (value(*l)?, value(*r)?);
            let mut groups = Vec::with_capacity(l.group_count() * r.group_count());
            for (g1, a) in l.groups().iter().enumerate() {
                for (g2, b) in r.groups().iter().enumerate() {
                    groups.push(Group {
                        width: a.width * b.width,
                        label: Label::new(LabelKind::Pair, (g1 * r.group_count() + g2) as u32),
                    });
                }
            }
            Interface::new(groups).map_err(error)?
        }
        Term::Concat(parts) => {
            let mut groups = Vec::new();
            for part in parts {
                groups.extend(value(*part)?.groups().iter().copied());
            }
            Interface::new(groups).map_err(error)?
        }
        Term::Attend { rotary, args: [q, k, v], .. } => {
            let width = value(*q)?.width();
            if value(*k)?.width() != width || rotary.is_some_and(|(_, dims, _)| dims as usize > width || dims % 2 == 1) {
                return Err("attention query, key and rotary widths disagree".to_string());
            }
            value(*v)?
        }
        Term::RmsNorm { args: [x], .. } => value(*x)?,
        Term::Transposed([op, x]) => {
            let (op, x) = (operator(*op)?, value(*x)?);
            if op.rows != x {
                return Err("a transposed operator does not read its argument's interface".to_string());
            }
            op.cols
        }
    };
    Ok(ClassData::Value(interface))
}

impl Analysis<Term> for Leaves {
    type Data = ClassData;

    fn make(egraph: &mut ProgramGraph, enode: &Term, id: Id) -> ClassData {
        match value_interface(egraph, enode) {
            Ok(data) => data,
            Err(fault) => {
                egraph.analysis.faults.push(format!("class {id}: {fault}"));
                ClassData::Fault
            }
        }
    }

    fn merge(&mut self, a: &mut ClassData, b: ClassData) -> DidMerge {
        match (a, b) {
            (ClassData::Operator(left), ClassData::Operator(right)) => {
                if !left.intersects(&right) {
                    self.faults.push(format!(
                        "an exact law equated operators {} and {} whose enclosures are disjoint",
                        left.best, right.best
                    ));
                    return DidMerge(false, true);
                }
                let lower = ndarray::Zip::from(&left.lower).and(&right.lower).map_collect(|x, y| x.max(*y));
                let upper = ndarray::Zip::from(&left.upper).and(&right.upper).map_collect(|x, y| x.min(*y));
                let changed_left = lower != left.lower || upper != left.upper;
                let changed_right = lower != right.lower || upper != right.upper;
                let right_best = (right.best_bits, right.best) < (left.best_bits, left.best);
                if right_best {
                    left.best = right.best;
                    left.best_bits = right.best_bits;
                }
                left.lower = lower;
                left.upper = upper;
                DidMerge(changed_left || right_best, changed_right || !right_best)
            }
            (ClassData::Value(left), ClassData::Value(right)) => {
                if *left != right {
                    self.faults.push("a law equated values of different interfaces".to_string());
                }
                DidMerge(false, false)
            }
            (ClassData::Scalar(left), ClassData::Scalar(right)) => match (*left, right) {
                (Some(a), Some(b)) if a.to_bits() != b.to_bits() && a != b => {
                    self.faults.push(format!("a law equated the numbers {a} and {b}"));
                    DidMerge(false, false)
                }
                (None, Some(b)) => {
                    *left = Some(b);
                    DidMerge(true, false)
                }
                (Some(_), None) => DidMerge(false, true),
                _ => DidMerge(false, false),
            },
            (ClassData::Fault, _) => DidMerge(false, true),
            (left, _) => {
                self.faults.push("a law equated an operator with a value".to_string());
                *left = ClassData::Fault;
                DidMerge(true, true)
            }
        }
    }
}

// ------------------------------------------------------------------------------------ graph access

fn best_leaf(egraph: &ProgramGraph, class: Id) -> Option<LeafId> {
    match &egraph[class].data {
        ClassData::Operator(op) => Some(op.best),
        _ => None,
    }
}

fn number(egraph: &ProgramGraph, class: Id) -> Option<f64> {
    match &egraph[class].data {
        ClassData::Scalar(value) => *value,
        _ => None,
    }
}

fn interface_of(egraph: &ProgramGraph, class: Id) -> Option<Interface> {
    match &egraph[class].data {
        ClassData::Value(interface) => Some(interface.clone()),
        _ => None,
    }
}

fn nodes(egraph: &ProgramGraph, class: Id) -> Vec<Term> {
    egraph[class].nodes.clone()
}

/// The cheapest constant leaf a value class holds.
fn constant_leaf(egraph: &ProgramGraph, class: Id) -> Option<LeafId> {
    egraph[class]
        .nodes
        .iter()
        .filter_map(|node| match node {
            Term::Constant([op]) => best_leaf(egraph, *op),
            _ => None,
        })
        .min_by_key(|leaf| (egraph.analysis.leaf(*leaf).bits, *leaf))
}

/// `(operator, argument)` of every `Apply` in a class, canonical.
fn applies(egraph: &ProgramGraph, class: Id) -> Vec<(Id, Id)> {
    egraph[class]
        .nodes
        .iter()
        .filter_map(|node| match node {
            Term::Apply([op, x]) => Some((egraph.find(*op), egraph.find(*x))),
            _ => None,
        })
        .collect()
}

/// `(scalar, value)` of every `Scale` in a class, canonical.
fn scales(egraph: &ProgramGraph, class: Id) -> Vec<(Id, Id)> {
    egraph[class]
        .nodes
        .iter()
        .filter_map(|node| match node {
            Term::Scale([c, x]) => Some((egraph.find(*c), egraph.find(*x))),
            _ => None,
        })
        .collect()
}

fn canonical(egraph: &ProgramGraph, ids: impl IntoIterator<Item = Id>) -> Vec<Id> {
    let mut ids: Vec<Id> = ids.into_iter().map(|id| egraph.find(id)).collect();
    ids.sort_unstable();
    ids
}

fn add_union(egraph: &mut ProgramGraph, class: Id, term: Term) -> bool {
    let id = egraph.add(term);
    egraph.union(class, id)
}

fn leaf_class(egraph: &mut ProgramGraph, leaf: LeafId) -> Id {
    egraph.add(Term::Leaf(leaf))
}

fn number_class(egraph: &mut ProgramGraph, value: f64) -> Id {
    egraph.add(Term::Number(if value == 0.0 { 0.0_f64.to_bits() } else { value.to_bits() }))
}

/// A sum of `children`: the child itself when there is one.
fn sum_of(egraph: &mut ProgramGraph, children: Vec<Id>) -> Id {
    if children.len() == 1 {
        return children[0];
    }
    let children = canonical(egraph, children);
    egraph.add(Term::Sum(children))
}

/// The first `Sum` node's children in a class.
fn sum_in(egraph: &ProgramGraph, class: Id) -> Option<Vec<Id>> {
    egraph[class].nodes.iter().find_map(|node| match node {
        Term::Sum(children) => Some(children.clone()),
        _ => None,
    })
}

// ------------------------------------------------------------------------------------------ laws

/// The exact laws of the module note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Equation {
    Transpose,
    Compose,
    ApplyConstant,
    ApplyIdentity,
    ScalarIdentity,
    NumericScale,
    UnitScale,
    ScaleThroughApply,
    ScaleCompose,
    ScalarFold,
    Distribute,
    PushThroughMix,
    FactorMix,
    MixConstants,
    MixSplit,
    Flatten,
    DropZero,
    Fuse,
    BilinearConstant,
    HadamardConstant,
    PointwiseIdentity,
    PointwiseConstant,
}

pub const EQUATIONS: [Equation; 22] = [
    Equation::Transpose,
    Equation::Compose,
    Equation::ApplyConstant,
    Equation::ApplyIdentity,
    Equation::ScalarIdentity,
    Equation::NumericScale,
    Equation::UnitScale,
    Equation::ScaleThroughApply,
    Equation::ScaleCompose,
    Equation::ScalarFold,
    Equation::Distribute,
    Equation::PushThroughMix,
    Equation::FactorMix,
    Equation::MixConstants,
    Equation::MixSplit,
    Equation::Flatten,
    Equation::DropZero,
    Equation::Fuse,
    Equation::BilinearConstant,
    Equation::HadamardConstant,
    Equation::PointwiseIdentity,
    Equation::PointwiseConstant,
];

/// Whether an operator class is exactly zero: its enclosure is the single zero matrix.
fn exactly_zero(egraph: &ProgramGraph, class: Id) -> bool {
    let ClassData::Operator(op) = &egraph[class].data else { return false };
    op.lower == op.upper && op.lower.iter().all(|v| *v == 0.0)
}

/// The exact multiple `c` when an operator class is exactly `c I`.
fn scalar_multiple(egraph: &ProgramGraph, class: Id) -> Option<f64> {
    let ClassData::Operator(op) = &egraph[class].data else { return None };
    if op.rows != op.cols || op.lower != op.upper {
        return None;
    }
    let c = op.lower[[0, 0]];
    op.lower.indexed_iter().all(|((r, k), v)| *v == if r == k { c } else { 0.0 }).then_some(c)
}

impl Equation {
    /// Apply the law everywhere it matches in `class`; whether the e-graph learned something.
    fn apply(self, egraph: &mut ProgramGraph, class: Id) -> Result<bool, EgraphError> {
        let mut changed = false;
        match self {
            Self::Compose => {
                for (op, x) in applies(egraph, class) {
                    let Some(a) = best_leaf(egraph, op) else { continue };
                    for (inner, z) in applies(egraph, x) {
                        let Some(b) = best_leaf(egraph, inner) else { continue };
                        let product = egraph.analysis.product(a, b)?;
                        let leaf = leaf_class(egraph, product);
                        changed |= add_union(egraph, class, Term::Apply([leaf, z]));
                    }
                }
            }
            Self::ApplyConstant => {
                for (op, x) in applies(egraph, class) {
                    let (Some(a), Some(b)) = (best_leaf(egraph, op), constant_leaf(egraph, x)) else { continue };
                    let product = egraph.analysis.product(a, b)?;
                    let leaf = leaf_class(egraph, product);
                    changed |= add_union(egraph, class, Term::Constant([leaf]));
                }
            }
            Self::ApplyIdentity => {
                for (op, x) in applies(egraph, class) {
                    if matches!(&egraph[op].data, ClassData::Operator(data) if data.is_identity()) {
                        changed |= egraph.union(class, x);
                    }
                }
            }
            Self::ScalarIdentity => {
                for (op, x) in applies(egraph, class) {
                    if let Some(c) = scalar_multiple(egraph, op)
                        && c != 1.0
                    {
                        let scalar = number_class(egraph, c);
                        changed |= add_union(egraph, class, Term::Scale([scalar, x]));
                    }
                }
            }
            Self::NumericScale => {
                for (c, x) in scales(egraph, class) {
                    let (Some(value), Some(interface)) = (number(egraph, c), interface_of(egraph, x)) else { continue };
                    let leaf = egraph.analysis.scalar_identity(value, interface)?;
                    let leaf = leaf_class(egraph, leaf);
                    changed |= add_union(egraph, class, Term::Apply([leaf, x]));
                }
            }
            Self::UnitScale => {
                for (c, x) in scales(egraph, class) {
                    if number(egraph, c) == Some(1.0) {
                        changed |= egraph.union(class, x);
                    }
                }
            }
            Self::ScaleThroughApply => {
                for (op, x) in applies(egraph, class) {
                    for (c, z) in scales(egraph, x) {
                        let inner = egraph.add(Term::Apply([op, z]));
                        changed |= add_union(egraph, class, Term::Scale([c, inner]));
                    }
                }
            }
            Self::ScaleCompose => {
                for (c, x) in scales(egraph, class) {
                    for (d, z) in scales(egraph, x) {
                        let product = egraph.add(Term::ScalarProduct([c.min(d), c.max(d)]));
                        changed |= add_union(egraph, class, Term::Scale([product, z]));
                    }
                }
            }
            Self::ScalarFold => changed |= fold_scalar(egraph, class),
            Self::Distribute => {
                for (op, x) in applies(egraph, class) {
                    let Some(children) = sum_in(egraph, x) else { continue };
                    let terms: Vec<Id> = children.iter().map(|child| egraph.add(Term::Apply([op, *child]))).collect();
                    let sum = sum_of(egraph, terms);
                    changed |= egraph.union(class, sum);
                }
            }
            Self::PushThroughMix => {
                for (op, x) in applies(egraph, class) {
                    for node in nodes(egraph, x) {
                        let Term::Mix { columns, args } = node else { continue };
                        let mut moved = vec![args[0]];
                        moved.extend(args[1..].iter().map(|p| egraph.add(Term::Apply([op, *p]))));
                        changed |= add_union(egraph, class, Term::Mix { columns, args: moved });
                    }
                }
            }
            Self::FactorMix => {
                for node in nodes(egraph, class) {
                    let Term::Mix { columns, args } = node else { continue };
                    for (shared, first) in applies(egraph, args[1]) {
                        let mut inputs = vec![args[0], first];
                        for payload in &args[2..] {
                            match applies(egraph, *payload).into_iter().find(|(op, _)| *op == shared) {
                                Some((_, z)) => inputs.push(z),
                                None => break,
                            }
                        }
                        if inputs.len() != args.len() {
                            continue;
                        }
                        let mix = egraph.add(Term::Mix { columns: columns.clone(), args: inputs });
                        changed |= add_union(egraph, class, Term::Apply([shared, mix]));
                    }
                }
            }
            Self::MixConstants => {
                for node in nodes(egraph, class) {
                    let Term::Mix { columns, args } = node else { continue };
                    let leaves: Option<Vec<LeafId>> = args[1..].iter().map(|p| constant_leaf(egraph, *p)).collect();
                    let (Some(leaves), Some(weights)) = (leaves, interface_of(egraph, args[0])) else { continue };
                    let table = egraph.analysis.routed(columns, leaves, weights)?;
                    let table = leaf_class(egraph, table);
                    changed |= add_union(egraph, class, Term::Apply([table, args[0]]));
                }
            }
            Self::MixSplit => {
                for node in nodes(egraph, class) {
                    let Term::Mix { columns, args } = node else { continue };
                    changed |= split_mix(egraph, class, &columns, &args);
                }
            }
            Self::Flatten => {
                for node in nodes(egraph, class) {
                    let Term::Sum(children) = node else { continue };
                    for (at, child) in children.iter().enumerate() {
                        if egraph.find(*child) == egraph.find(class) {
                            continue;
                        }
                        if let Some(inner) = sum_in(egraph, *child) {
                            let flat: Vec<Id> =
                                children.iter().enumerate().filter(|(i, _)| *i != at).map(|(_, c)| *c).chain(inner).collect();
                            let sum = sum_of(egraph, flat);
                            changed |= egraph.union(class, sum);
                        }
                    }
                }
            }
            Self::DropZero => {
                for node in nodes(egraph, class) {
                    let Term::Sum(children) = node else { continue };
                    let vanishes = |child: Id| {
                        egraph[child].nodes.iter().any(|n| matches!(n, Term::Apply([op, _]) if exactly_zero(egraph, *op)))
                    };
                    let kept: Vec<Id> = children.iter().copied().filter(|c| !vanishes(*c)).collect();
                    if !kept.is_empty() && kept.len() < children.len() {
                        let sum = sum_of(egraph, kept);
                        changed |= egraph.union(class, sum);
                    }
                }
            }
            Self::Fuse => {
                for node in nodes(egraph, class) {
                    let Term::Sum(children) = node else { continue };
                    changed |= fuse(egraph, class, &children)?;
                }
            }
            Self::BilinearConstant => {
                for node in nodes(egraph, class) {
                    let Term::Bilinear { scale, args: [l, r] } = node else { continue };
                    for (constant, varying) in [(l, r), (r, l)] {
                        let (Some(a), Some(cols)) = (constant_leaf(egraph, constant), interface_of(egraph, varying)) else { continue };
                        let row = egraph.analysis.scaled_row(a, scale, cols)?;
                        let row = leaf_class(egraph, row);
                        changed |= add_union(egraph, class, Term::Apply([row, varying]));
                    }
                }
            }
            Self::HadamardConstant => {
                for node in nodes(egraph, class) {
                    let Term::Hadamard([l, r]) = node else { continue };
                    let (Some(left), Some(right)) = (interface_of(egraph, l), interface_of(egraph, r)) else { continue };
                    if let Some(a) = constant_leaf(egraph, l) {
                        let diagonal = egraph.analysis.diagonal(a, left.clone(), right)?;
                        let diagonal = leaf_class(egraph, diagonal);
                        changed |= add_union(egraph, class, Term::Apply([diagonal, r]));
                    } else if let Some(b) = constant_leaf(egraph, r) {
                        let diagonal = egraph.analysis.diagonal(b, left.clone(), left)?;
                        let diagonal = leaf_class(egraph, diagonal);
                        changed |= add_union(egraph, class, Term::Apply([diagonal, l]));
                    }
                }
            }
            Self::Transpose => {
                for node in nodes(egraph, class) {
                    let Term::Transposed([op, x]) = node else { continue };
                    let Some(a) = best_leaf(egraph, op) else { continue };
                    let transposed = egraph.analysis.transpose(a)?;
                    let leaf = leaf_class(egraph, transposed);
                    changed |= add_union(egraph, class, Term::Apply([leaf, x]));
                }
            }
            Self::PointwiseIdentity => {
                for node in nodes(egraph, class) {
                    let Term::Pointwise { laws, args: [x] } = node else { continue };
                    if laws.iter().all(|law| law_of(*law) == Law::Identity) {
                        changed |= egraph.union(class, x);
                    }
                }
            }
            Self::PointwiseConstant => {
                for node in nodes(egraph, class) {
                    let Term::Pointwise { laws, args: [x] } = node else { continue };
                    let exact = laws.iter().all(|law| matches!(law_of(*law), Law::Relu | Law::Identity | Law::Zero));
                    if let (true, Some(a)) = (exact, constant_leaf(egraph, x)) {
                        let folded = egraph.analysis.pointwise(a, laws)?;
                        let folded = leaf_class(egraph, folded);
                        changed |= add_union(egraph, class, Term::Constant([folded]));
                    }
                }
            }
        }
        Ok(changed)
    }
}

/// A scalar class that holds an exact number is that number; a one-term sum is its term; nested
/// sums flatten; a product with the number one is its other factor. A parameter is never valued.
fn fold_scalar(egraph: &mut ProgramGraph, class: Id) -> bool {
    let mut changed = false;
    if let Some(value) = number(egraph, class) {
        let folded = number_class(egraph, value);
        changed |= egraph.union(class, folded);
    }
    for node in nodes(egraph, class) {
        if let Term::ScalarSum(terms) = &node {
            if terms.len() == 1 {
                changed |= egraph.union(class, terms[0]);
            } else {
                for (at, term) in terms.iter().enumerate() {
                    if egraph.find(*term) == egraph.find(class) {
                        continue;
                    }
                    let inner = egraph[*term].nodes.iter().find_map(|n| match n {
                        Term::ScalarSum(inner) => Some(inner.clone()),
                        _ => None,
                    });
                    if let Some(inner) = inner {
                        let flat: Vec<Id> = terms.iter().enumerate().filter(|(i, _)| *i != at).map(|(_, t)| *t).chain(inner).collect();
                        let flat = canonical(egraph, flat);
                        changed |= add_union(egraph, class, Term::ScalarSum(flat));
                    }
                }
            }
        }
        if let Term::ScalarProduct([a, b]) = node {
            for (one, other) in [(a, b), (b, a)] {
                if number(egraph, one) == Some(1.0) {
                    changed |= egraph.union(class, other);
                }
            }
        }
    }
    changed
}

/// `Σ_j α_j (x_j + b_j) = Σ_j α_j x_j + Σ_j α_j b_j` when every payload is a constant or a sum
/// with exactly one constant summand, and at least one is a sum.
fn split_mix(egraph: &mut ProgramGraph, class: Id, columns: &[usize], args: &[Id]) -> bool {
    let mut rests: Vec<(usize, Id)> = Vec::new();
    let mut constants: Vec<Id> = Vec::new();
    for (&column, &payload) in columns.iter().zip(&args[1..]) {
        if constant_leaf(egraph, payload).is_some() {
            constants.push(payload);
            continue;
        }
        let split = egraph[payload].nodes.iter().find_map(|n| match n {
            Term::Sum(children) => {
                let constant: Vec<usize> = (0..children.len()).filter(|&i| constant_leaf(egraph, children[i]).is_some()).collect();
                (constant.len() == 1).then(|| (children.clone(), constant[0]))
            }
            _ => None,
        });
        let Some((children, at)) = split else { return false };
        constants.push(children[at]);
        let others: Vec<Id> = children.iter().enumerate().filter(|(i, _)| *i != at).map(|(_, c)| *c).collect();
        let rest = sum_of(egraph, others);
        rests.push((column, rest));
    }
    if rests.is_empty() {
        return false;
    }
    let (rest_columns, rest_args): (Vec<usize>, Vec<Id>) = rests.into_iter().unzip();
    let varying = egraph.add(Term::Mix { columns: rest_columns, args: std::iter::once(args[0]).chain(rest_args).collect() });
    let fixed = egraph.add(Term::Mix { columns: columns.to_vec(), args: std::iter::once(args[0]).chain(constants).collect() });
    let sum = sum_of(egraph, vec![varying, fixed]);
    egraph.union(class, sum)
}

/// One summand read as `coefficient · value`: every `Scale` it holds, and itself with coefficient
/// one.
fn scaled_forms(egraph: &mut ProgramGraph, child: Id) -> Vec<(Id, Id)> {
    let one = number_class(egraph, 1.0);
    let mut forms = scales(egraph, child);
    forms.push((one, egraph.find(child)));
    forms
}

/// Fuse two summands of `children` that read one value: `A x + B x = (A + B) x`,
/// `a·x + b·x = (a + b)·x` with the coefficients kept symbolic, and constants added. A pair whose
/// coefficient is the exact number zero (no parameter in it) cancels.
fn fuse(egraph: &mut ProgramGraph, class: Id, children: &[Id]) -> Result<bool, EgraphError> {
    let mut changed = false;
    for i in 0..children.len() {
        for j in i + 1..children.len() {
            let rest: Vec<Id> = children.iter().enumerate().filter(|(k, _)| *k != i && *k != j).map(|(_, c)| *c).collect();
            let mut fused: Vec<Id> = Vec::new();
            for (op_a, x_a) in applies(egraph, children[i]) {
                for (op_b, x_b) in applies(egraph, children[j]) {
                    let (Some(a), Some(b), true) = (best_leaf(egraph, op_a), best_leaf(egraph, op_b), x_a == x_b) else { continue };
                    let total = egraph.analysis.sum(a, b)?;
                    let total = leaf_class(egraph, total);
                    fused.push(egraph.add(Term::Apply([total, x_a])));
                }
            }
            if let (Some(a), Some(b)) = (constant_leaf(egraph, children[i]), constant_leaf(egraph, children[j])) {
                let total = egraph.analysis.sum(a, b)?;
                let total = leaf_class(egraph, total);
                fused.push(egraph.add(Term::Constant([total])));
            }
            let (left, right) = (scaled_forms(egraph, children[i]), scaled_forms(egraph, children[j]));
            for &(c, x) in &left {
                for &(d, z) in &right {
                    if x != z {
                        continue;
                    }
                    let coefficient = canonical(egraph, [c, d]);
                    let coefficient = egraph.add(Term::ScalarSum(coefficient));
                    if number(egraph, coefficient) == Some(0.0) {
                        if !rest.is_empty() {
                            let sum = sum_of(egraph, rest.clone());
                            changed |= egraph.union(class, sum);
                        }
                        continue;
                    }
                    fused.push(egraph.add(Term::Scale([coefficient, x])));
                }
            }
            for term in fused {
                let mut summands = rest.clone();
                summands.push(term);
                let sum = sum_of(egraph, summands);
                changed |= egraph.union(class, sum);
            }
        }
    }
    Ok(changed)
}

// ------------------------------------------------------------------------------------ saturation

/// Why saturation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaturationStop {
    /// An iteration learned nothing: every law's consequences are in the e-graph.
    Saturated,
    /// The leaves would outgrow the bytes the memory governor granted at the start.
    ResourceBound { bytes: usize, budget: usize },
}

/// What saturation did.
#[derive(Clone, Debug, PartialEq)]
pub struct SaturationReport {
    pub stop: SaturationStop,
    pub iterations: usize,
    pub nodes: usize,
    pub classes: usize,
    pub leaves: usize,
}

impl SaturationReport {
    pub fn saturated(&self) -> bool {
        self.stop == SaturationStop::Saturated
    }
}

/// A saturated (or resource-bounded) e-graph of a program.
pub struct Saturation {
    pub egraph: ProgramGraph,
    pub root: Id,
    pub report: SaturationReport,
}

/// `coefficient` as a scalar class: parameters stay symbols, numbers are exact.
fn coefficient_class(egraph: &mut ProgramGraph, coefficient: &Coefficient) -> Result<Id, EgraphError> {
    Ok(match coefficient {
        Coefficient::Parameter(index) => {
            let index = u32::try_from(*index).map_err(|_| EgraphError::Inconsistent(format!("parameter {index}")))?;
            egraph.add(Term::Parameter(index))
        }
        Coefficient::Number(value) => number_class(egraph, *value),
        Coefficient::Sum(terms) => {
            let children = terms.iter().map(|term| coefficient_class(egraph, term)).collect::<Result<Vec<_>, _>>()?;
            match children.len() {
                0 => number_class(egraph, 0.0),
                1 => children[0],
                _ => egraph.add(Term::ScalarSum(children)),
            }
        }
        Coefficient::Product(terms) => {
            let mut product = number_class(egraph, 1.0);
            for (index, term) in terms.iter().enumerate() {
                let factor = coefficient_class(egraph, term)?;
                product = if index == 0 { factor } else { egraph.add(Term::ScalarProduct([product, factor])) };
            }
            product
        }
    })
}

/// Add `nodes` (the program's, or a rule body's with `params` its arguments) to the e-graph; a
/// call is inlined, so every application of a rule is its body on the call's arguments.
fn build_nodes(
    egraph: &mut ProgramGraph,
    program: &OperatorProgram,
    leaves: &[Id],
    nodes: &[Node],
    params: &[Id],
) -> Result<Vec<Id>, EgraphError> {
    let mut classes: Vec<Id> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let term = match node {
            Node::Feature { slot, basis } => Term::Feature { slot: *slot, basis: *basis },
            Node::Raw { slot } => Term::Raw { slot: *slot },
            Node::Constant { operator } => Term::Constant([leaves[*operator]]),
            Node::Affine { terms, bias } => {
                let mut children = Vec::new();
                for (argument, operator) in terms {
                    children.push(egraph.add(Term::Apply([leaves[*operator], classes[*argument]])));
                }
                if let Some(operator) = bias {
                    children.push(egraph.add(Term::Constant([leaves[*operator]])));
                }
                if children.is_empty() {
                    return Err(EgraphError::Inconsistent("an affine node with no terms".to_string()));
                }
                classes.push(sum_of(egraph, children));
                continue;
            }
            Node::Bilinear { left, right, scale } => {
                Term::Bilinear { scale: scale_code(*scale), args: [classes[*left], classes[*right]] }
            }
            Node::Softmax { scores } => Term::Softmax(scores.iter().map(|s| classes[*s]).collect()),
            Node::Mix { weights, payloads } => Term::Mix {
                columns: payloads.iter().map(|(c, _)| *c).collect(),
                args: std::iter::once(classes[*weights]).chain(payloads.iter().map(|(_, p)| classes[*p])).collect(),
            },
            Node::Pointwise { input, laws } => {
                Term::Pointwise { laws: laws.iter().map(|l| law_code(*l)).collect(), args: [classes[*input]] }
            }
            Node::Hadamard { left, right } => Term::Hadamard([classes[*left], classes[*right]]),
            Node::Readout { input, basis } => Term::Readout { basis: *basis, args: [classes[*input]] },
            Node::Outer { left, right } => Term::Outer([classes[*left], classes[*right]]),
            Node::Concat { parts } => Term::Concat(parts.iter().map(|p| classes[*p]).collect()),
            Node::Attend { query, key, value, scale, rotary, causal } => Term::Attend {
                scale: scale_code(*scale),
                rotary: rotary.map(|r| (r.base, r.dims, r.half_split)),
                causal: *causal,
                args: [classes[*query], classes[*key], classes[*value]],
            },
            Node::RmsNorm { input, epsilon } => Term::RmsNorm { epsilon: epsilon.to_bits(), args: [classes[*input]] },
            Node::Transposed { input, operator } => Term::Transposed([leaves[*operator], classes[*input]]),
            Node::Param { index } => {
                let argument = params.get(*index).copied().ok_or_else(|| EgraphError::Inconsistent(format!("argument {index} outside a call")))?;
                classes.push(argument);
                continue;
            }
            Node::Call { rule, arguments } => {
                let body = program.rules.get(*rule).ok_or_else(|| EgraphError::Inconsistent(format!("rule {rule}")))?;
                let arguments: Vec<Id> = arguments.iter().map(|a| classes[*a]).collect();
                let inner = build_nodes(egraph, program, leaves, &body.nodes, &arguments)?;
                classes.push(inner[body.output]);
                continue;
            }
            Node::Gain { input, coefficient } => {
                let scalar = coefficient_class(egraph, coefficient)?;
                Term::Scale([scalar, classes[*input]])
            }
        };
        classes.push(egraph.add(term));
    }
    Ok(classes)
}

/// Load `program` into an e-graph, its gains' coefficients as symbolic scalars.
fn build(program: &OperatorProgram) -> Result<(ProgramGraph, Id), EgraphError> {
    program.interfaces()?;
    let mut egraph = ProgramGraph::new(Leaves::new(program.declarations.clone(), program.bases.clone()));
    let mut leaves = Vec::with_capacity(program.operators.len());
    for operator in &program.operators {
        let leaf = egraph.analysis.native(Operator::clone(operator))?;
        leaves.push(leaf_class(&mut egraph, leaf));
    }
    let classes = build_nodes(&mut egraph, program, &leaves, &program.nodes, &[])?;
    egraph.rebuild();
    if let Some(fault) = egraph.analysis.faults.first() {
        return Err(EgraphError::Inconsistent(fault.clone()));
    }
    Ok((egraph, classes[program.output]))
}

/// Every pair of distinct operator classes of one interface pair whose enclosures intersect: the
/// candidates for a proven equality.
fn intersecting_operators(egraph: &ProgramGraph) -> Vec<(Id, Id)> {
    let mut groups: HashMap<(Interface, Interface), Vec<(f64, Id)>> = HashMap::new();
    for class in egraph.classes() {
        if let ClassData::Operator(op) = &class.data {
            groups.entry((op.rows.clone(), op.cols.clone())).or_default().push((op.lower[[0, 0]], class.id));
        }
    }
    let mut pairs = Vec::new();
    for (_, mut members) in groups {
        members.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for i in 0..members.len() {
            for j in i + 1..members.len() {
                let (a, b) = (egraph.find(members[i].1), egraph.find(members[j].1));
                let (ClassData::Operator(left), ClassData::Operator(right)) = (&egraph[a].data, &egraph[b].data) else { continue };
                if right.lower[[0, 0]] > left.upper[[0, 0]] {
                    break;
                }
                if a != b && left.intersects(right) {
                    pairs.push((a, b));
                }
            }
        }
    }
    pairs
}

/// Union every pair of operator classes proven equal: both exact and entrywise equal (module note,
/// "Certified equality of leaves").
fn certified_merge(egraph: &mut ProgramGraph) -> bool {
    let mut changed = false;
    for (a, b) in intersecting_operators(egraph) {
        let (a, b) = (egraph.find(a), egraph.find(b));
        let (ClassData::Operator(left), ClassData::Operator(right)) = (&egraph[a].data, &egraph[b].data) else { continue };
        if a != b && left.is_exact() && right.is_exact() && left.lower == right.lower {
            changed |= egraph.union(a, b);
        }
    }
    changed
}

/// Saturate `program` under the laws, its gains kept symbolic, until a fixpoint or until the
/// leaves would exceed the bytes `governor` grants now.
pub fn saturate(program: &OperatorProgram, governor: &MemoryGovernor) -> Result<Saturation, EgraphError> {
    let (mut egraph, root) = build(program)?;
    let budget = governor.remaining_bytes();
    let mut iterations = 0;
    let stop = loop {
        if egraph.analysis.bytes > budget {
            break SaturationStop::ResourceBound { bytes: egraph.analysis.bytes, budget };
        }
        let before = (egraph.total_size(), egraph.number_of_classes());
        let ids: Vec<Id> = egraph.classes().map(|class| class.id).collect();
        let mut changed = false;
        'laws: for equation in EQUATIONS {
            for &id in &ids {
                let class = egraph.find(id);
                changed |= equation.apply(&mut egraph, class)?;
                if egraph.analysis.bytes > budget {
                    break 'laws;
                }
            }
        }
        egraph.rebuild();
        changed |= certified_merge(&mut egraph);
        egraph.rebuild();
        if let Some(fault) = egraph.analysis.faults.first() {
            return Err(EgraphError::Inconsistent(fault.clone()));
        }
        iterations += 1;
        if !changed && before == (egraph.total_size(), egraph.number_of_classes()) {
            break SaturationStop::Saturated;
        }
    };
    let report = SaturationReport {
        stop,
        iterations,
        nodes: egraph.total_size(),
        classes: egraph.number_of_classes(),
        leaves: egraph.analysis.leaves.len(),
    };
    let root = egraph.find(root);
    Ok(Saturation { egraph, root, report })
}

// ------------------------------------------------------------------------------------ extraction

/// Each class's chosen node with its proxy cost.
pub type Choices = HashMap<Id, (u64, Term)>;

/// A lowered candidate.
#[derive(Clone, Debug)]
pub struct Extraction {
    pub program: OperatorProgram,
    /// The decoded message length of `program`.
    pub bits: u64,
    /// The decoded lengths of every candidate extraction, in the order tried.
    pub candidates: Vec<u64>,
    pub choices: Choices,
    pub leaves: BTreeSet<LeafId>,
}

impl Saturation {
    /// The cheapest node per class under the proxy: a leaf costs its decoded bits unless `paid`,
    /// every node its kind and references. A class in `pins` may only take its pinned node.
    pub fn choose(&self, paid: &BTreeSet<LeafId>, pins: &HashMap<Id, Term>) -> Result<Choices, EgraphError> {
        let egraph = &self.egraph;
        let reference = u64::from(fixed_index_len_bits(egraph.number_of_classes().max(2)).map_err(ProgramError::from)?);
        let kind = u64::from(fixed_index_len_bits(16).map_err(ProgramError::from)?);
        let own = |term: &Term| -> u64 {
            match term {
                Term::Leaf(leaf) if paid.contains(leaf) => reference,
                Term::Leaf(leaf) => egraph.analysis.leaf(*leaf).bits + reference,
                Term::Pointwise { laws, .. } => kind + reference + 3 * laws.len() as u64,
                other => kind + reference * (other.children().len() as u64 + 1),
            }
        };
        let mut choices: Choices = HashMap::new();
        loop {
            let mut changed = false;
            for class in egraph.classes() {
                let pinned = pins.get(&class.id);
                for node in &class.nodes {
                    if pinned.is_some_and(|pin| {
                        let canonical_pin = pin.clone().map_children(|c| egraph.find(c));
                        canonical_pin != node.clone().map_children(|c| egraph.find(c))
                    }) {
                        continue;
                    }
                    let mut total = own(node);
                    let mut complete = true;
                    for child in node.children() {
                        match choices.get(&egraph.find(*child)) {
                            Some((cost, _)) => total = total.saturating_add(*cost),
                            None => {
                                complete = false;
                                break;
                            }
                        }
                    }
                    if !complete {
                        continue;
                    }
                    let better = match choices.get(&class.id) {
                        Some((cost, term)) => (total, node) < (*cost, term),
                        None => true,
                    };
                    if better {
                        choices.insert(class.id, (total, node.clone()));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if !choices.contains_key(&self.root) {
            return Err(EgraphError::Inconsistent("the root has no finite extraction".to_string()));
        }
        Ok(choices)
    }

    /// Lower `choices` to a program; a symbolic coefficient becomes a [`Node::Gain`].
    pub fn lower(&self, choices: &Choices) -> Result<(OperatorProgram, BTreeSet<LeafId>), EgraphError> {
        let mut lowering = Lowering {
            saturation: self,
            choices,
            program: OperatorProgram {
                declarations: self.egraph.analysis.declarations.clone(),
                bases: self.egraph.analysis.bases.clone(),
                operators: Vec::new(),
                rules: Vec::new(),
                nodes: Vec::new(),
                output: 0,
            },
            class_node: HashMap::new(),
            leaf_operator: HashMap::new(),
            scaled: HashMap::new(),
            leaves: BTreeSet::new(),
        };
        let output = lowering.value(self.root)?;
        let mut program = lowering.program;
        program.output = output;
        program.prune();
        program.interfaces()?;
        Ok((program, lowering.leaves))
    }

    /// The shortest extraction: the proxy extraction, re-priced by the decoded message length,
    /// repeated with the chosen leaves paid until the chosen leaf set recurs.
    pub fn extract(&self, pins: &HashMap<Id, Term>) -> Result<Extraction, EgraphError> {
        let mut paid: BTreeSet<LeafId> = BTreeSet::new();
        let mut seen: BTreeSet<BTreeSet<LeafId>> = BTreeSet::new();
        let mut candidates = Vec::new();
        let mut best: Option<Extraction> = None;
        loop {
            let choices = self.choose(&paid, pins)?;
            let (program, leaves) = self.lower(&choices)?;
            let bits = program.code_bits()?;
            candidates.push(bits);
            if best.as_ref().is_none_or(|b| bits < b.bits) {
                best = Some(Extraction { program, bits, candidates: Vec::new(), choices, leaves: leaves.clone() });
            }
            if !seen.insert(leaves.clone()) {
                break;
            }
            paid = leaves;
        }
        let mut best = best.ok_or_else(|| EgraphError::Inconsistent("no extraction".to_string()))?;
        best.candidates = candidates;
        Ok(best)
    }

    /// The chosen term below `class`, written out (parameters as `m<i>`).
    pub fn render(&self, choices: &Choices, class: Id) -> String {
        let class = self.egraph.find(class);
        let Some((_, term)) = choices.get(&class) else { return format!("?{class}") };
        let child = |id: &Id| self.render(choices, *id);
        let list = |ids: &[Id], sep: &str| ids.iter().map(child).collect::<Vec<_>>().join(sep);
        match term {
            Term::Leaf(leaf) => self.egraph.analysis.leaf(*leaf).operator.name.clone(),
            Term::Parameter(p) => format!("m{p}"),
            Term::Number(bits) => format!("{}", f64::from_bits(*bits)),
            Term::ScalarSum(terms) => format!("({})", list(terms, " + ")),
            Term::ScalarProduct(pair) => format!("({})", list(pair, "·")),
            Term::Scale([c, x]) => format!("{}·{}", child(c), child(x)),
            Term::Feature { slot, basis } => format!("feature{slot}[{basis}]"),
            Term::Raw { slot } => format!("x{slot}"),
            Term::Constant([op]) => format!("const({})", child(op)),
            Term::Apply([op, x]) => format!("{}({})", child(op), child(x)),
            Term::Sum(children) => format!("({})", list(children, " + ")),
            Term::Bilinear { args, .. } => format!("bilinear({})", list(args, ", ")),
            Term::Softmax(scores) => format!("softmax({})", list(scores, ", ")),
            Term::Mix { args, .. } => format!("mix({})", list(args, ", ")),
            Term::Pointwise { args: [x], laws } => {
                let names: Vec<String> = laws.iter().map(|l| format!("{:?}", law_of(*l))).collect();
                format!("{}({})", names.join("|"), child(x))
            }
            Term::Hadamard(pair) => format!("({})", list(pair, " ⊙ ")),
            Term::Readout { args: [x], .. } => format!("readout({})", child(x)),
            Term::Outer(pair) => format!("outer({})", list(pair, ", ")),
            Term::Concat(parts) => format!("concat({})", list(parts, ", ")),
            Term::Attend { args, .. } => format!("attend({})", list(args, ", ")),
            Term::RmsNorm { args: [x], .. } => format!("rmsnorm({})", child(x)),
            Term::Transposed([op, x]) => format!("{}ᵀ({})", child(op), child(x)),
        }
    }

    /// The parameters the chosen term below `class` reads.
    pub fn parameters_read(&self, choices: &Choices, class: Id) -> BTreeSet<u32> {
        let mut out = BTreeSet::new();
        let mut stack = vec![self.egraph.find(class)];
        let mut visited = BTreeSet::new();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some((_, term)) = choices.get(&id) {
                if let Term::Parameter(p) = term {
                    out.insert(*p);
                }
                stack.extend(term.children().iter().map(|c| self.egraph.find(*c)));
            }
        }
        out
    }
}

struct Lowering<'a> {
    saturation: &'a Saturation,
    choices: &'a Choices,
    program: OperatorProgram,
    class_node: HashMap<Id, usize>,
    leaf_operator: HashMap<LeafId, usize>,
    scaled: HashMap<(u64, Interface), usize>,
    leaves: BTreeSet<LeafId>,
}

impl Lowering<'_> {
    fn term(&self, class: Id) -> Result<Term, EgraphError> {
        let class = self.saturation.egraph.find(class);
        self.choices
            .get(&class)
            .map(|(_, term)| term.clone())
            .ok_or_else(|| EgraphError::Inconsistent(format!("class {class} has no choice")))
    }

    fn operator(&mut self, class: Id) -> Result<usize, EgraphError> {
        let Term::Leaf(leaf) = self.term(class)? else {
            return Err(EgraphError::Inconsistent(format!("operator class {class} chose a non-leaf")));
        };
        if let Some(&index) = self.leaf_operator.get(&leaf) {
            return Ok(index);
        }
        self.leaves.insert(leaf);
        self.program.operators.push(self.saturation.egraph.analysis.leaf(leaf).operator.clone().into());
        let index = self.program.operators.len() - 1;
        self.leaf_operator.insert(leaf, index);
        Ok(index)
    }

    /// `c I` on `interface`, on the coarsest lattice holding `c`.
    fn scaled_identity(&mut self, c: f64, interface: &Interface) -> Result<usize, EgraphError> {
        let key = (c.to_bits(), interface.clone());
        if let Some(&index) = self.scaled.get(&key) {
            return Ok(index);
        }
        let operator = if c == 1.0 {
            Operator::identity("I", interface.clone())
        } else {
            let width = interface.width();
            let present = Array2::from_shape_fn((interface.group_count(), interface.group_count()), |(r, k)| r == k);
            Operator::blocks(
                format!("{c}·I"),
                interface.clone(),
                interface.clone(),
                Array2::from_diag_elem(width, c),
                present,
                exact_precision([c])?,
                Provenance::derived(&[], "a coefficient times the identity".to_string()),
            )?
        };
        self.program.operators.push(operator.into());
        let index = self.program.operators.len() - 1;
        self.scaled.insert(key, index);
        Ok(index)
    }

    /// The chosen scalar below `class` as a coefficient polynomial.
    fn coefficient(&self, class: Id) -> Result<Coefficient, EgraphError> {
        Ok(match self.term(class)? {
            Term::Parameter(p) => Coefficient::Parameter(p as usize),
            Term::Number(bits) => Coefficient::Number(f64::from_bits(bits)),
            Term::ScalarSum(terms) => Coefficient::Sum(terms.iter().map(|t| self.coefficient(*t)).collect::<Result<_, _>>()?),
            Term::ScalarProduct([a, b]) => Coefficient::Product(vec![self.coefficient(a)?, self.coefficient(b)?]),
            other => return Err(EgraphError::Inconsistent(format!("{other:?} is not a scalar"))),
        })
    }

    /// The chosen scalar's value when it is a number, `None` when it reads a parameter.
    fn number(&self, class: Id) -> Result<Option<f64>, EgraphError> {
        Ok(match self.term(class)? {
            Term::Number(bits) => Some(f64::from_bits(bits)),
            _ => None,
        })
    }

    fn interface(&self, class: Id) -> Result<Interface, EgraphError> {
        interface_of(&self.saturation.egraph, self.saturation.egraph.find(class))
            .ok_or_else(|| EgraphError::Inconsistent(format!("class {class} is not a value")))
    }

    /// An affine term for a summand: `(argument node, operator)`, or `None` for a constant.
    fn summand(&mut self, class: Id) -> Result<(Option<(usize, usize)>, Option<usize>), EgraphError> {
        Ok(match self.term(class)? {
            Term::Apply([op, x]) => {
                let operator = self.operator(op)?;
                (Some((self.value(x)?, operator)), None)
            }
            Term::Constant([op]) => (None, Some(self.operator(op)?)),
            Term::Scale([c, x]) if self.number(c)?.is_some() => {
                let c = self.number(c)?.unwrap_or(1.0);
                let interface = self.interface(x)?;
                let operator = self.scaled_identity(c, &interface)?;
                (Some((self.value(x)?, operator)), None)
            }
            _ => {
                let interface = self.interface(class)?;
                let operator = self.scaled_identity(1.0, &interface)?;
                (Some((self.value(class)?, operator)), None)
            }
        })
    }

    fn push(&mut self, class: Id, node: Node) -> usize {
        self.program.nodes.push(node);
        let index = self.program.nodes.len() - 1;
        self.class_node.insert(self.saturation.egraph.find(class), index);
        index
    }

    fn value(&mut self, class: Id) -> Result<usize, EgraphError> {
        let class = self.saturation.egraph.find(class);
        if let Some(&index) = self.class_node.get(&class) {
            return Ok(index);
        }
        let node = match self.term(class)? {
            Term::Feature { slot, basis } => Node::Feature { slot, basis },
            Term::Raw { slot } => Node::Raw { slot },
            Term::Constant([op]) => Node::Constant { operator: self.operator(op)? },
            Term::Scale([c, x]) if self.number(c)?.is_none() => {
                Node::Gain { input: self.value(x)?, coefficient: self.coefficient(c)? }
            }
            Term::Apply(_) | Term::Scale(_) => {
                let (term, bias) = self.summand(class)?;
                Node::Affine { terms: term.into_iter().collect(), bias }
            }
            Term::Sum(children) => {
                let mut terms = Vec::new();
                let mut bias = None;
                for child in children {
                    match self.summand(child)? {
                        (_, Some(constant)) if bias.is_none() => bias = Some(constant),
                        (_, Some(constant)) => {
                            let interface = self.interface(child)?;
                            let identity = self.scaled_identity(1.0, &interface)?;
                            let node = self.program.nodes.len();
                            self.program.nodes.push(Node::Constant { operator: constant });
                            terms.push((node, identity));
                        }
                        (Some(term), None) => terms.push(term),
                        (None, None) => {}
                    }
                }
                Node::Affine { terms, bias }
            }
            Term::Bilinear { scale, args: [l, r] } => {
                Node::Bilinear { left: self.value(l)?, right: self.value(r)?, scale: scale_of(scale) }
            }
            Term::Softmax(scores) => {
                let scores = scores.iter().map(|s| self.value(*s)).collect::<Result<_, _>>()?;
                Node::Softmax { scores }
            }
            Term::Mix { columns, args } => {
                let weights = self.value(args[0])?;
                let mut payloads = Vec::with_capacity(columns.len());
                for (column, payload) in columns.iter().zip(&args[1..]) {
                    payloads.push((*column, self.value(*payload)?));
                }
                Node::Mix { weights, payloads }
            }
            Term::Pointwise { laws, args: [x] } => {
                Node::Pointwise { input: self.value(x)?, laws: laws.iter().map(|l| law_of(*l)).collect() }
            }
            Term::Hadamard([l, r]) => Node::Hadamard { left: self.value(l)?, right: self.value(r)? },
            Term::Readout { basis, args: [x] } => Node::Readout { input: self.value(x)?, basis },
            Term::Outer([l, r]) => Node::Outer { left: self.value(l)?, right: self.value(r)? },
            Term::Concat(parts) => {
                let parts = parts.iter().map(|p| self.value(*p)).collect::<Result<_, _>>()?;
                Node::Concat { parts }
            }
            Term::Attend { scale, rotary, causal, args: [q, k, v] } => Node::Attend {
                query: self.value(q)?,
                key: self.value(k)?,
                value: self.value(v)?,
                scale: scale_of(scale),
                rotary: rotary.map(|(base, dims, half_split)| Rotary { base, dims, half_split }),
                causal,
            },
            Term::RmsNorm { epsilon, args: [x] } => Node::RmsNorm { input: self.value(x)?, epsilon: f64::from_bits(epsilon) },
            Term::Transposed([op, x]) => {
                let operator = self.operator(op)?;
                Node::Transposed { input: self.value(x)?, operator }
            }
            other => return Err(EgraphError::Inconsistent(format!("value class {class} chose {other:?}"))),
        };
        Ok(self.push(class, node))
    }
}

// ------------------------------------------------------------------------------------ unit gauge

/// `2^e` with `2^e ≤ m < 2^{e+1}`, for a positive finite `m`.
fn binade(m: f64) -> i32 {
    let mut e = m.log2().floor() as i32;
    while 2.0_f64.powi(e) > m {
        e -= 1;
    }
    while 2.0_f64.powi(e + 1) <= m {
        e += 1;
    }
    e
}

/// A dense operator's values, present blocks and name, or `None` for another body.
fn dense_parts(operator: &Operator) -> Option<(Array2<f64>, Array2<bool>)> {
    match &operator.body {
        OperatorBody::Dense { values, present, .. } => Some((values.clone(), present.clone())),
        _ => None,
    }
}

/// The unit gauge of the module note, applied to every ReLU layer it applies to exactly: the layer's
/// pre-activation is an affine node read only by the ReLU, the ReLU is read only through affine
/// terms, every operator involved is dense and used once, and the units are width-one groups
/// labelled consecutively. A gain on either side commutes with every move: `relu(α 2^e t) =
/// 2^e relu(α t)` for any `α`.
pub fn canonical_units(program: &OperatorProgram) -> Result<OperatorProgram, EgraphError> {
    let mut program = program.clone();
    let mut uses = vec![0usize; program.operators.len()];
    for node in &program.nodes {
        for op in program.node_operators(node) {
            uses[op] += 1;
        }
    }
    for p in 0..program.nodes.len() {
        let Node::Pointwise { input: a, laws } = program.nodes[p].clone() else { continue };
        let Node::Affine { terms, bias } = program.nodes[a].clone() else { continue };
        if !laws.iter().all(|law| *law == Law::Relu) || p == program.output || a == program.output {
            continue;
        }
        let readers_of = |target: usize| -> Vec<usize> {
            program.nodes.iter().enumerate().filter(|(_, n)| n.arguments().contains(&target)).map(|(i, _)| i).collect()
        };
        if readers_of(a) != vec![p] {
            continue;
        }
        let mut writes: Vec<(usize, usize)> = Vec::new();
        let mut fits = true;
        for reader in readers_of(p) {
            match &program.nodes[reader] {
                Node::Affine { terms: reads, .. } => {
                    writes.extend(reads.iter().filter(|(x, _)| *x == p).map(|(_, op)| (reader, *op)));
                }
                _ => fits = false,
            }
        }
        let reads: Vec<usize> = terms.iter().map(|(_, op)| *op).chain(bias).collect();
        let every: Vec<usize> = reads.iter().chain(writes.iter().map(|(_, op)| op)).copied().collect();
        if !fits
            || writes.is_empty()
            || every.iter().any(|op| uses[*op] != 1 || dense_parts(&program.operators[*op]).is_none())
        {
            continue;
        }
        let units = program.operators[reads[0]].rows.clone();
        let groups = units.groups();
        let first = groups[0].label;
        if groups.iter().enumerate().any(|(i, g)| g.width != 1 || g.label != Label::new(first.kind, first.index + i as u32)) {
            continue;
        }
        let width = units.width();
        let read_parts: Vec<(Array2<f64>, Array2<bool>)> = reads.iter().filter_map(|op| dense_parts(&program.operators[*op])).collect();
        let write_parts: Vec<(Array2<f64>, Array2<bool>)> = writes.iter().filter_map(|(_, op)| dense_parts(&program.operators[*op])).collect();
        let row = |i: usize| -> Vec<f64> { read_parts.iter().flat_map(|(values, _)| values.row(i).to_vec()).collect() };
        let bias_of = |i: usize| -> f64 { bias.map_or(0.0, |op| program.operators[op].matrix()[[i, 0]]) };
        // Merge equal read rows; drop units that read nothing (`relu(0) = 0` on every input).
        let mut kept: Vec<Vec<usize>> = Vec::new();
        for i in 0..width {
            let own = row(i);
            if own.iter().all(|v| *v == 0.0) {
                continue;
            }
            match kept.iter_mut().find(|group| row(group[0]) == own) {
                Some(group) => group.push(i),
                None => kept.push(vec![i]),
            }
        }
        if kept.is_empty() {
            continue;
        }
        // The unit's scale: its bias, invariant under every change of the read coordinates, or,
        // with no bias, its read row's norm, invariant under an orthogonal one.
        let exponents: Vec<i32> = kept
            .iter()
            .map(|group| {
                let b = bias_of(group[0]).abs();
                binade(if b > 0.0 { b } else { row(group[0]).iter().map(|v| v * v).sum::<f64>().sqrt() })
            })
            .collect();
        let mut order: Vec<usize> = (0..kept.len()).collect();
        let key = |k: usize| -> (f64, f64) {
            let scale = 2.0_f64.powi(-exponents[k]);
            let norm: f64 = row(kept[k][0]).iter().map(|v| (v * scale) * (v * scale)).sum();
            (bias_of(kept[k][0]) * scale, norm)
        };
        order.sort_by(|&x, &y| {
            let (kx, ky) = (key(x), key(y));
            kx.0.total_cmp(&ky.0).then(kx.1.total_cmp(&ky.1)).then(x.cmp(&y))
        });
        let unchanged = order.len() == width
            && order.iter().enumerate().all(|(position, &k)| kept[k] == vec![position] && exponents[k] == 0);
        if unchanged {
            continue;
        }
        let new_units = Interface::uniform(order.len(), 1, first.kind, first.index)?;
        let provenance_step = "the canonical unit gauge: merged, power-of-two balanced, ordered".to_string();
        for (&op, (values, present)) in reads.iter().zip(&read_parts) {
            let old = &program.operators[op];
            let mut new_values = Array2::<f64>::zeros((order.len(), values.ncols()));
            let mut new_present = Array2::from_elem((order.len(), present.ncols()), false);
            for (position, &k) in order.iter().enumerate() {
                let scale = 2.0_f64.powi(-exponents[k]);
                new_values.row_mut(position).assign(&values.row(kept[k][0]).mapv(|v| v * scale));
                for &unit in &kept[k] {
                    for c in 0..present.ncols() {
                        new_present[[position, c]] |= present[[unit, c]];
                    }
                }
            }
            let lattice = exact_precision(new_values.iter().copied())?;
            let provenance = Provenance::derived(&[&old.provenance], provenance_step.clone());
            program.operators[op] = Operator::blocks(old.name.clone(), new_units.clone(), old.cols.clone(), new_values, new_present, lattice, provenance)?.into();
        }
        for (&(_, op), (values, present)) in writes.iter().zip(&write_parts) {
            let old = &program.operators[op];
            let mut new_values = Array2::<f64>::zeros((values.nrows(), order.len()));
            let mut new_present = Array2::from_elem((present.nrows(), order.len()), false);
            for (position, &k) in order.iter().enumerate() {
                let scale = 2.0_f64.powi(exponents[k]);
                for &unit in &kept[k] {
                    let mut column = new_values.column_mut(position);
                    column += &values.column(unit).mapv(|v| v * scale);
                    for r in 0..present.nrows() {
                        new_present[[r, position]] |= present[[r, unit]];
                    }
                }
            }
            let lattice = exact_precision(new_values.iter().copied())?;
            let provenance = Provenance::derived(&[&old.provenance], provenance_step.clone());
            program.operators[op] = Operator::blocks(old.name.clone(), old.rows.clone(), new_units.clone(), new_values, new_present, lattice, provenance)?.into();
        }
        program.nodes[p] = Node::Pointwise { input: a, laws: vec![Law::Relu; order.len()] };
    }
    program.interfaces()?;
    Ok(program)
}

// ---------------------------------------------------------------------------------- normalization

/// A program normalized by saturation.
pub struct Normalization {
    /// The shortest program: the extraction, or the native program when no extraction is shorter.
    pub program: OperatorProgram,
    pub bits: u64,
    pub native_bits: u64,
    pub saturation: Saturation,
    pub extraction: Extraction,
}

/// Normalize `program`: the canonical unit gauge, saturation with its gains symbolic, extraction.
pub fn normalize(program: &OperatorProgram, governor: &MemoryGovernor) -> Result<Normalization, EgraphError> {
    let canonical = canonical_units(program)?;
    let saturation = saturate(&canonical, governor)?;
    let extraction = saturation.extract(&HashMap::new())?;
    let native_bits = program.code_bits()?;
    let (chosen, bits) = if native_bits <= extraction.bits {
        (program.clone(), native_bits)
    } else {
        (extraction.program.clone(), extraction.bits)
    };
    Ok(Normalization { program: chosen, bits, native_bits, saturation, extraction })
}

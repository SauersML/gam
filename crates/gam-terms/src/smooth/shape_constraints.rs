//! Exact shape-constraint (monotone / convex / concave) machinery.
//!
//! Shape constraints are admitted only for open, untransformed B-spline
//! control coefficients.  In that chart, non-negative first control-point
//! differences certify monotonicity on every knot span, while non-decreasing
//! Greville-scaled control-polygon slopes certify convexity.  The smooth
//! builder realizes those cones with an invertible coefficient transform and
//! coordinate lower bounds; no sampled evaluation grid is involved.
//!
//! A term's request is a [`ShapeSpec`]: either one conjunction of atomic
//! shapes for a 1-D smooth (`shape=[monotone_increasing, concave]`), or one
//! conjunction per margin of a `te()` tensor product
//! (`shape=[monotone_increasing, none]`). A single 1-D atom keeps the exact
//! box reparameterization; every other request is realized as the exact
//! linear cone `A β ≥ 0` through the general inequality path. This module is
//! the single source of truth for the shape grammar: the formula DSL, the
//! `smooths={...}` override descriptors and the Python `constraints=` rewrite
//! all parse through [`parse_shape_expr`] / [`resolve_shape_spec`].

use super::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TensorBSplineIdentifiability,
    TensorBSplineSpec,
};
use crate::basis::{
    BSplineBasisSpec, BSplineKnotSpec, BasisError, BasisMetadata, OneDimensionalBoundary,
};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView1, s};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A conjunction of atomic shape constraints along one coordinate direction.
///
/// At most one monotone direction and one curvature direction can be held, so
/// the contradictory pairs (increasing with decreasing, convex with concave)
/// are unrepresentable: [`ShapeSet::insert`] rejects them with an explanation
/// of the degenerate function class they would force.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShapeSet {
    monotone: Option<ShapeConstraint>,
    curvature: Option<ShapeConstraint>,
}

impl ShapeSet {
    pub fn single(atom: ShapeConstraint) -> Self {
        match atom {
            ShapeConstraint::None => Self::default(),
            ShapeConstraint::MonotoneIncreasing | ShapeConstraint::MonotoneDecreasing => Self {
                monotone: Some(atom),
                curvature: None,
            },
            ShapeConstraint::Convex | ShapeConstraint::Concave => Self {
                monotone: None,
                curvature: Some(atom),
            },
        }
    }

    /// Add `atom` to the conjunction. Repeating an atom is a no-op; opposing
    /// atoms are an error because their intersection is not a shape class the
    /// user can have meant: it collapses the function to a constant (both
    /// monotone directions) or to an affine function (both curvatures).
    pub fn insert(&mut self, atom: ShapeConstraint) -> Result<(), String> {
        let (slot, opposite_consequence) = match atom {
            ShapeConstraint::None => return Ok(()),
            ShapeConstraint::MonotoneIncreasing | ShapeConstraint::MonotoneDecreasing => (
                &mut self.monotone,
                "is non-increasing and non-decreasing at once, i.e. constant along this \
                 direction; drop the shape (or the term) or keep a single direction",
            ),
            ShapeConstraint::Convex | ShapeConstraint::Concave => (
                &mut self.curvature,
                "is convex and concave at once, i.e. affine along this direction; use a \
                 linear term instead or keep a single curvature",
            ),
        };
        match *slot {
            Some(existing) if existing != atom => Err(format!(
                "contradictory shape constraints {} and {}: the only function satisfying both \
                 {opposite_consequence}",
                existing.dsl_str(),
                atom.dsl_str()
            )),
            _ => {
                *slot = Some(atom);
                Ok(())
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.monotone.is_none() && self.curvature.is_none()
    }

    /// Atoms in canonical order: the monotone direction, then the curvature.
    pub fn atoms(&self) -> impl Iterator<Item = ShapeConstraint> + '_ {
        self.monotone.into_iter().chain(self.curvature)
    }

    /// The atom when the conjunction holds exactly one.
    pub fn single_atom(&self) -> Option<ShapeConstraint> {
        match (self.monotone, self.curvature) {
            (Some(atom), None) | (None, Some(atom)) => Some(atom),
            _ => None,
        }
    }

    pub fn monotone(&self) -> Option<ShapeConstraint> {
        self.monotone
    }

    pub fn curvature(&self) -> Option<ShapeConstraint> {
        self.curvature
    }

    fn from_atoms(atoms: &[ShapeConstraint]) -> Result<Self, String> {
        let mut set = Self::default();
        for &atom in atoms {
            set.insert(atom)?;
        }
        Ok(set)
    }
}

impl fmt::Display for ShapeSet {
    /// Canonical DSL text: `none`, a bare atom, or a bracketed conjunction.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let atoms: Vec<&str> = self.atoms().map(|atom| atom.dsl_str()).collect();
        match atoms.as_slice() {
            [] => f.write_str("none"),
            [atom] => f.write_str(atom),
            many => write!(f, "[{}]", many.join(", ")),
        }
    }
}

/// The shape request of one smooth term.
///
/// * [`ShapeSpec::Joint`] constrains a 1-D smooth (also inside a `by=`
///   wrapper) by a conjunction of atoms.
/// * [`ShapeSpec::PerMargin`] constrains a `te()` tensor product along each
///   margin in margin order; an empty entry leaves that margin free.
///
/// Persisted payloads written before per-margin and multi-atom shapes existed
/// store a bare [`ShapeConstraint`] variant name, and a spec with at most one
/// atom still serializes that way.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(try_from = "ShapeSpecRepr", into = "ShapeSpecRepr")]
pub enum ShapeSpec {
    #[default]
    None,
    Joint(ShapeSet),
    PerMargin(Vec<ShapeSet>),
}

impl ShapeSpec {
    fn from_joint(set: ShapeSet) -> Self {
        if set.is_empty() {
            Self::None
        } else {
            Self::Joint(set)
        }
    }

    fn from_margins(margins: Vec<ShapeSet>) -> Self {
        if margins.iter().all(ShapeSet::is_empty) {
            Self::None
        } else {
            Self::PerMargin(margins)
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            Self::Joint(set) => set.is_empty(),
            Self::PerMargin(margins) => margins.iter().all(ShapeSet::is_empty),
        }
    }

    /// The atom of a 1-D spec holding exactly one; this is the only request
    /// realized by the box reparameterization.
    pub fn single_atom(&self) -> Option<ShapeConstraint> {
        match self {
            Self::Joint(set) => set.single_atom(),
            Self::None | Self::PerMargin(_) => None,
        }
    }
}

impl From<ShapeConstraint> for ShapeSpec {
    fn from(atom: ShapeConstraint) -> Self {
        Self::from_joint(ShapeSet::single(atom))
    }
}

impl PartialEq<ShapeConstraint> for ShapeSpec {
    /// `spec == ShapeConstraint::None` asks "is this term unconstrained?";
    /// `spec == atom` asks "is this exactly the single 1-D shape `atom`?".
    fn eq(&self, other: &ShapeConstraint) -> bool {
        match other {
            ShapeConstraint::None => self.is_none(),
            atom => self.single_atom() == Some(*atom),
        }
    }
}

impl fmt::Display for ShapeSpec {
    /// Canonical DSL text accepted back by [`parse_shape_spec`] in the same
    /// term context.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Joint(set) => write!(f, "{set}"),
            Self::PerMargin(margins) => {
                let parts: Vec<String> = margins.iter().map(ToString::to_string).collect();
                write!(f, "[{}]", parts.join(", "))
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum ShapeSpecRepr {
    Atom(ShapeConstraint),
    Joint {
        joint: Vec<ShapeConstraint>,
    },
    PerMargin {
        per_margin: Vec<Vec<ShapeConstraint>>,
    },
}

impl TryFrom<ShapeSpecRepr> for ShapeSpec {
    type Error = String;

    fn try_from(repr: ShapeSpecRepr) -> Result<Self, String> {
        Ok(match repr {
            ShapeSpecRepr::Atom(atom) => atom.into(),
            ShapeSpecRepr::Joint { joint } => Self::from_joint(ShapeSet::from_atoms(&joint)?),
            ShapeSpecRepr::PerMargin { per_margin } => Self::from_margins(
                per_margin
                    .iter()
                    .map(|atoms| ShapeSet::from_atoms(atoms))
                    .collect::<Result<_, _>>()?,
            ),
        })
    }
}

impl From<ShapeSpec> for ShapeSpecRepr {
    fn from(spec: ShapeSpec) -> Self {
        match spec {
            ShapeSpec::None => Self::Atom(ShapeConstraint::None),
            ShapeSpec::Joint(set) => match set.single_atom() {
                Some(atom) => Self::Atom(atom),
                None if set.is_empty() => Self::Atom(ShapeConstraint::None),
                None => Self::Joint {
                    joint: set.atoms().collect(),
                },
            },
            ShapeSpec::PerMargin(margins) => Self::PerMargin {
                per_margin: margins.iter().map(|set| set.atoms().collect()).collect(),
            },
        }
    }
}

/// Context-free syntax tree of a `shape=` value: an atom or a (possibly
/// nested) list. Its meaning depends on the term it is attached to and is
/// assigned by [`resolve_shape_spec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeExpr {
    Atom(ShapeConstraint),
    List(Vec<ShapeExpr>),
}

impl fmt::Display for ShapeExpr {
    /// Canonical DSL text, round-tripping through [`parse_shape_expr`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Atom(atom) => f.write_str(atom.dsl_str()),
            Self::List(items) => {
                let parts: Vec<String> = items.iter().map(ToString::to_string).collect();
                write!(f, "[{}]", parts.join(", "))
            }
        }
    }
}

impl ShapeExpr {
    /// True when the expression requests nothing anywhere.
    pub fn is_none(&self) -> bool {
        match self {
            Self::Atom(atom) => *atom == ShapeConstraint::None,
            Self::List(items) => items.iter().all(Self::is_none),
        }
    }
}

/// Parse the text of a `shape=` value into its syntax tree.
///
/// Lists may be written `[a, b]`, `c(a, b)` or `(a, b)` and nest (one entry
/// per `te()` margin, each entry itself a conjunction). Atoms are the
/// spellings accepted by [`super::parse_shape_constraint`], optionally quoted.
pub fn parse_shape_expr(raw: &str) -> Result<ShapeExpr, String> {
    let chars: Vec<char> = raw.chars().collect();
    let mut pos = 0usize;
    let expr = parse_shape_expr_at(&chars, &mut pos, raw)?;
    skip_shape_whitespace(&chars, &mut pos);
    if pos != chars.len() {
        return Err(format!(
            "invalid shape constraint {raw:?}: unexpected {:?} after the shape value",
            chars[pos..].iter().collect::<String>()
        ));
    }
    Ok(expr)
}

fn skip_shape_whitespace(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn closing_shape_bracket(open: char) -> char {
    if open == '[' { ']' } else { ')' }
}

fn parse_shape_expr_at(chars: &[char], pos: &mut usize, raw: &str) -> Result<ShapeExpr, String> {
    skip_shape_whitespace(chars, pos);
    let Some(&first) = chars.get(*pos) else {
        return Ok(ShapeExpr::Atom(ShapeConstraint::None));
    };
    // `c(` opens an R-style vector exactly like `(`.
    let list_open = match first {
        '[' | '(' => Some(first),
        'c' if {
            let mut q = *pos + 1;
            skip_shape_whitespace(chars, &mut q);
            chars.get(q) == Some(&'(')
        } =>
        {
            *pos += 1;
            skip_shape_whitespace(chars, pos);
            Some('(')
        }
        _ => None,
    };
    if let Some(open) = list_open {
        *pos += 1;
        let close = closing_shape_bracket(open);
        let mut items = Vec::new();
        skip_shape_whitespace(chars, pos);
        if chars.get(*pos) == Some(&close) {
            *pos += 1;
            return Ok(ShapeExpr::List(items));
        }
        loop {
            items.push(parse_shape_expr_at(chars, pos, raw)?);
            skip_shape_whitespace(chars, pos);
            match chars.get(*pos) {
                Some(',') => *pos += 1,
                Some(&c) if c == close => {
                    *pos += 1;
                    return Ok(ShapeExpr::List(items));
                }
                Some(&c) => {
                    return Err(format!(
                        "invalid shape constraint {raw:?}: expected ',' or '{close}' but found '{c}'"
                    ));
                }
                None => {
                    return Err(format!(
                        "invalid shape constraint {raw:?}: unclosed '{open}'"
                    ));
                }
            }
        }
    }
    let token = if first == '"' || first == '\'' {
        *pos += 1;
        let start = *pos;
        while *pos < chars.len() && chars[*pos] != first {
            *pos += 1;
        }
        if *pos == chars.len() {
            return Err(format!(
                "invalid shape constraint {raw:?}: unterminated quote"
            ));
        }
        let token: String = chars[start..*pos].iter().collect();
        *pos += 1;
        token
    } else {
        let start = *pos;
        while *pos < chars.len()
            && (chars[*pos].is_ascii_alphanumeric() || chars[*pos] == '_' || chars[*pos] == '-')
        {
            *pos += 1;
        }
        if start == *pos {
            return Err(format!(
                "invalid shape constraint {raw:?}: unexpected '{first}'"
            ));
        }
        chars[start..*pos].iter().collect()
    };
    super::parse_shape_constraint(&token).map(ShapeExpr::Atom)
}

/// Syntax tree of a JSON `shape_constraint` descriptor value: a string (in
/// the DSL grammar), `null`, or a (nested) array of those.
pub fn shape_expr_from_json(value: &serde_json::Value) -> Result<ShapeExpr, String> {
    match value {
        serde_json::Value::Null => Ok(ShapeExpr::Atom(ShapeConstraint::None)),
        serde_json::Value::String(raw) => parse_shape_expr(raw),
        serde_json::Value::Array(items) => items
            .iter()
            .map(shape_expr_from_json)
            .collect::<Result<_, _>>()
            .map(ShapeExpr::List),
        other => Err(format!(
            "shape constraint must be a string, null or a list of those, got {other}"
        )),
    }
}

/// Number of tensor margins a `shape=` list addresses for `basis`, looking
/// through `by=` wrappers; `None` for a 1-D (or non-tensor) smooth.
pub fn shape_tensor_margin_count(basis: &SmoothBasisSpec) -> Option<usize> {
    match basis {
        SmoothBasisSpec::TensorBSpline { spec, .. } => Some(spec.marginalspecs.len()),
        SmoothBasisSpec::ByVariable { inner, .. } => shape_tensor_margin_count(inner),
        SmoothBasisSpec::BySmooth { smooth, .. } => shape_tensor_margin_count(smooth),
        _ => None,
    }
}

fn conjunction_from_expr(expr: &ShapeExpr, set: &mut ShapeSet, where_: &str) -> Result<(), String> {
    match expr {
        ShapeExpr::Atom(atom) => set.insert(*atom).map_err(|e| format!("{where_}: {e}")),
        ShapeExpr::List(items) => {
            for item in items {
                match item {
                    ShapeExpr::Atom(atom) => {
                        set.insert(*atom).map_err(|e| format!("{where_}: {e}"))?;
                    }
                    ShapeExpr::List(_) => {
                        return Err(format!(
                            "{where_}: shape list {expr} nests too deeply; a conjunction is a flat \
                             list of shapes such as [monotone_increasing, concave], and nested \
                             lists (one entry per margin) are only meaningful on te() tensor \
                             smooths"
                        ));
                    }
                }
            }
            Ok(())
        }
    }
}

/// Assign meaning to a parsed `shape=` value in the context of its term.
///
/// For a 1-D smooth (`tensor_margins == None`) the value is one conjunction:
/// an atom or a flat list of atoms. For a `te()` product over `d` margins it
/// must be `none` or a list of exactly `d` entries, entry `j` being an atom or
/// flat conjunction constraining the surface along margin `j`.
pub fn resolve_shape_spec(
    expr: &ShapeExpr,
    tensor_margins: Option<usize>,
) -> Result<ShapeSpec, String> {
    if expr.is_none() {
        return Ok(ShapeSpec::None);
    }
    let Some(margins) = tensor_margins else {
        let mut set = ShapeSet::default();
        conjunction_from_expr(expr, &mut set, "shape")?;
        return Ok(ShapeSpec::from_joint(set));
    };
    match expr {
        ShapeExpr::Atom(atom) => {
            let example: Vec<&str> = std::iter::once(atom.dsl_str())
                .chain(std::iter::repeat_n("none", margins.saturating_sub(1)))
                .collect();
            Err(format!(
                "shape={} on a te() tensor smooth over {margins} margins does not say which margin \
                 it constrains; give one entry per margin in margin order, e.g. shape=[{}]",
                atom.dsl_str(),
                example.join(", ")
            ))
        }
        ShapeExpr::List(items) => {
            if items.len() != margins {
                return Err(format!(
                    "shape={expr} on a te() tensor smooth has {} entries but the tensor has \
                     {margins} margins; give exactly one entry (a shape, a [..] conjunction, or \
                     none) per margin in margin order",
                    items.len()
                ));
            }
            let sets = items
                .iter()
                .enumerate()
                .map(|(j, item)| {
                    let mut set = ShapeSet::default();
                    conjunction_from_expr(item, &mut set, &format!("shape margin {j}"))?;
                    Ok(set)
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(ShapeSpec::from_margins(sets))
        }
    }
}

/// [`parse_shape_expr`] followed by [`resolve_shape_spec`].
pub fn parse_shape_spec(raw: &str, tensor_margins: Option<usize>) -> Result<ShapeSpec, String> {
    resolve_shape_spec(&parse_shape_expr(raw)?, tensor_margins)
}

/// Why a raw B-spline marginal chart cannot carry the exact coefficient cone,
/// or `None` when it can.
fn open_bspline_chart_obstruction(spec: &BSplineBasisSpec) -> Option<&'static str> {
    // A cyclic spline cannot be globally monotone unless it is constant, and
    // the periodic coefficient chart requires a wrap-around constraint that
    // the open cumulative transform does not encode.  Natural cubic regression
    // coefficients are knot values rather than raw B-spline control points, so
    // coefficient differences do not certify against cubic overshoot.  Endpoint
    // boundary conditions likewise introduce a raw-basis nullspace transform.
    // Reject all three instead of pretending their transformed coordinates have
    // the control-polygon geometry used by the exact cone below.
    if matches!(&spec.knotspec, BSplineKnotSpec::PeriodicUniform { .. })
        || !matches!(&spec.boundary, OneDimensionalBoundary::Open)
    {
        return Some(
            "a periodic spline cannot be monotone without being constant, and its cyclic \
             coefficient chart has no open control polygon",
        );
    }
    if matches!(
        &spec.knotspec,
        BSplineKnotSpec::NaturalCubicRegression { .. }
    ) {
        return Some(
            "cubic regression (cr) coefficients are knot values, not B-spline control points, \
             so coefficient differences do not certify the shape; use bs=\"ps\"/\"bs\"",
        );
    }
    if !spec.boundary_conditions.is_free() {
        return Some(
            "endpoint boundary conditions re-parameterize the raw control coefficients the \
             exact cone is written in",
        );
    }
    None
}

/// Check that `term` can realize its shape request exactly, explaining the
/// obstruction otherwise. `by=` wrappers are checked on their inner smooth
/// when it is built.
pub(super) fn validate_shape_request(term: &SmoothTermSpec) -> Result<(), BasisError> {
    if term.shape.is_none() {
        return Ok(());
    }
    let name = &term.name;
    let shape = &term.shape;
    let fail = |why: String| -> Result<(), BasisError> {
        Err(BasisError::InvalidInput(format!(
            "shape={shape} is unsupported for term '{name}': {why}"
        )))
    };
    match (&term.basis, shape) {
        (SmoothBasisSpec::ByVariable { .. } | SmoothBasisSpec::BySmooth { .. }, _) => Ok(()),
        (SmoothBasisSpec::BSpline1D { spec, .. }, ShapeSpec::Joint(_)) => {
            match open_bspline_chart_obstruction(spec) {
                Some(why) => fail(why.to_string()),
                None => Ok(()),
            }
        }
        (SmoothBasisSpec::TensorBSpline { spec, .. }, ShapeSpec::PerMargin(margins)) => {
            validate_tensor_shape_request(spec, margins).or_else(fail)
        }
        (SmoothBasisSpec::TensorBSpline { spec, .. }, ShapeSpec::Joint(_)) => fail(format!(
            "a te() tensor smooth needs one shape entry per margin ({} margins), e.g. \
             shape=[monotone_increasing, none]",
            spec.marginalspecs.len()
        )),
        (_, ShapeSpec::PerMargin(_)) => {
            fail("per-margin shape lists apply only to te() tensor-product smooths".to_string())
        }
        (basis, _) => fail(format!(
            "shape constraints need an open B-spline coefficient chart (s(x) with a B-spline \
             basis, or te(...)); the {} basis has no control polygon that certifies the shape",
            basis.structural_kind()
        )),
    }
}

fn validate_tensor_shape_request(
    spec: &TensorBSplineSpec,
    margins: &[ShapeSet],
) -> Result<(), String> {
    if margins.len() != spec.marginalspecs.len() {
        return Err(format!(
            "{} per-margin entries for a tensor with {} margins",
            margins.len(),
            spec.marginalspecs.len()
        ));
    }
    if matches!(
        spec.identifiability,
        TensorBSplineIdentifiability::MarginalSumToZero
    ) {
        return Err(
            "ti() removes every margin's main effect, so each slice along a constrained margin \
             averages to zero over the other margins; a monotone slice with zero average is \
             identically zero and a convex one is only linear, leaving no genuine shape class. \
             Constrain a te() term instead"
                .to_string(),
        );
    }
    for (j, (set, marginal)) in margins.iter().zip(&spec.marginalspecs).enumerate() {
        if set.is_empty() {
            continue;
        }
        if spec.periods.get(j).copied().flatten().is_some() {
            return Err(format!(
                "margin {j} is periodic, and a periodic spline cannot be monotone without being \
                 constant"
            ));
        }
        if let Some(why) = open_bspline_chart_obstruction(marginal) {
            return Err(format!("margin {j}: {why}"));
        }
    }
    Ok(())
}

/// How a validated shape request is realized on a built local term.
pub(super) enum ShapeRealization {
    Unconstrained,
    /// Single 1-D atom: invertible derivative-control chart plus coordinate
    /// lower bounds on every coordinate from `order` on.
    Box {
        order: usize,
        sign: f64,
    },
    /// Exact cone `A β ≥ 0` in the term's local coefficient chart.
    Linear(LinearInequalityConstraints),
}

pub(super) fn plan_shape_realization(
    term: &SmoothTermSpec,
    metadata: &BasisMetadata,
    p_local: usize,
) -> Result<ShapeRealization, BasisError> {
    let fail = |why: &str| -> Result<ShapeRealization, BasisError> {
        Err(BasisError::InvalidInput(format!(
            "shape={} on term '{}' {why}",
            term.shape, term.name
        )))
    };
    if term.shape.is_none() {
        return Ok(ShapeRealization::Unconstrained);
    }
    if let Some((order, sign)) = term.shape.single_atom().and_then(shape_order_and_sign) {
        return Ok(ShapeRealization::Box { order, sign });
    }
    let constraints = match (&term.shape, metadata) {
        (
            ShapeSpec::Joint(set),
            BasisMetadata::BSpline1D {
                knots,
                degree: Some(degree),
                periodic: None,
                ..
            },
        ) => bspline_shape_set_linear_constraints(knots.view(), *degree, *set)?,
        (ShapeSpec::Joint(_), _) => {
            return fail("requires realized open B-spline knot and degree metadata");
        }
        (
            ShapeSpec::PerMargin(margins),
            BasisMetadata::TensorBSpline {
                knots,
                degrees,
                periods,
                is_cr,
                identifiability_transform,
                ..
            },
        ) => {
            if knots.len() != margins.len() || degrees.len() != margins.len() {
                return fail("does not match the realized tensor margin count");
            }
            let knot_valued: Vec<bool> = (0..margins.len())
                .map(|j| {
                    is_cr.get(j).copied().unwrap_or(false)
                        || periods.get(j).copied().flatten().is_some()
                })
                .collect();
            let raw =
                tensor_bspline_shape_linear_constraints(knots, degrees, &knot_valued, margins)?;
            match (raw, identifiability_transform) {
                (Some(raw), Some(z)) => {
                    if z.nrows() != raw.a.ncols() {
                        return fail(
                            "has a tensor identifiability transform that does not match the \
                             raw tensor coefficient count",
                        );
                    }
                    Some(normalize_constraint_rows(raw.a.dot(z), raw.b)?)
                }
                (raw, _) => raw,
            }
        }
        (ShapeSpec::PerMargin(_), _) => {
            return fail("requires realized tensor B-spline metadata");
        }
        (ShapeSpec::None, _) => return Ok(ShapeRealization::Unconstrained),
    };
    match constraints {
        Some(c) if c.a.nrows() > 0 => {
            if c.a.ncols() != p_local {
                return fail("produced a cone whose width does not match the term's coefficients");
            }
            Ok(ShapeRealization::Linear(c))
        }
        // An affine chart satisfies any curvature request vacuously.
        _ => Ok(ShapeRealization::Unconstrained),
    }
}

fn normalize_constraint_rows(
    mut a: Array2<f64>,
    b: Array1<f64>,
) -> Result<LinearInequalityConstraints, BasisError> {
    for (mut row, rhs) in a.rows_mut().into_iter().zip(b.iter()) {
        let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= 0.0 || *rhs != 0.0 {
            return Err(BasisError::InvalidInput(
                "shape-constraint row has no finite direction in the identified tensor chart"
                    .to_string(),
            ));
        }
        row.mapv_inplace(|value| value / norm);
    }
    Ok(LinearInequalityConstraints { a, b })
}

/// Exact cone for a conjunction of shapes on one open B-spline chart.
///
/// A lone atom is [`bspline_shape_linear_constraints`]. Monotonicity together
/// with a curvature is not the plain stack of both row families: under the
/// curvature cone the first-derivative controls are ordered (non-decreasing
/// for convex, non-increasing for concave), so the monotone sign of every
/// control is implied by the sign of the single extreme one. Keeping only that
/// row describes the same cone with `p - 1` instead of `2p - 3` rows, none of
/// them redundant.
pub fn bspline_shape_set_linear_constraints(
    knots: ArrayView1<'_, f64>,
    degree: usize,
    set: ShapeSet,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    let monotone = match set.monotone() {
        Some(atom) => bspline_shape_linear_constraints(knots, degree, atom)?,
        None => None,
    };
    let curvature = match set.curvature() {
        Some(atom) => bspline_shape_linear_constraints(knots, degree, atom)?,
        None => None,
    };
    let (Some(monotone), Some(curvature), Some((_, monotone_sign)), Some((_, curvature_sign))) = (
        monotone.clone(),
        curvature.clone(),
        set.monotone().and_then(shape_order_and_sign),
        set.curvature().and_then(shape_order_and_sign),
    ) else {
        return Ok(monotone.or(curvature));
    };
    if curvature.a.nrows() == 0 || monotone.a.nrows() == 0 {
        return merge_linear_constraints_global(Some(monotone), Some(curvature));
    }
    // Increasing+convex and decreasing+concave are pinned by the first
    // control; the mixed pairs by the last.
    let binding = if monotone_sign == curvature_sign {
        0
    } else {
        monotone.a.nrows() - 1
    };
    let extreme = LinearInequalityConstraints {
        a: monotone.a.slice(s![binding..binding + 1, ..]).to_owned(),
        b: monotone.b.slice(s![binding..binding + 1]).to_owned(),
    };
    merge_linear_constraints_global(Some(extreme), Some(curvature))
}

/// Exact per-margin cone of a tensor-product B-spline in raw (unidentified)
/// coefficients.
///
/// The tensor coefficient of `(i_0, …, i_{d-1})` sits at the row-major index
/// with margin 0 outermost, matching the Khatri–Rao design and the
/// `S_0 ⊗ … ⊗ S_{d-1}` penalty layout. Along margin `j` the surface is, for
/// every fixed index of the other margins, a 1-D B-spline in margin `j` whose
/// control coefficients are that fibre, so the cone is
/// `I_{q_0} ⊗ … ⊗ A_j ⊗ … ⊗ I_{q_{d-1}}` for the 1-D cone `A_j`. Because every
/// other margin's basis is non-negative, a monotone/convex fibre for every
/// index certifies the shape of the whole surface along margin `j`.
///
/// `knot_valued[j]` marks a margin whose recorded knots are one per
/// coefficient (a cubic regression margin's value knots, or a periodic
/// margin's control sites) rather than an open B-spline knot vector. Such a
/// margin can only be unconstrained; it still sets the identity width.
pub fn tensor_bspline_shape_linear_constraints(
    knots: &[Array1<f64>],
    degrees: &[usize],
    knot_valued: &[bool],
    margins: &[ShapeSet],
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    if knots.len() != margins.len()
        || degrees.len() != margins.len()
        || knot_valued.len() != margins.len()
    {
        return Err(BasisError::DimensionMismatch(format!(
            "tensor shape request has {} margins but the basis has {} knot vectors, {} degrees \
             and {} margin kinds",
            margins.len(),
            knots.len(),
            degrees.len(),
            knot_valued.len()
        )));
    }
    let widths = knots
        .iter()
        .zip(degrees)
        .zip(knot_valued)
        .map(|((k, &d), &valued)| {
            if valued {
                return Ok(k.len());
            }
            k.len().checked_sub(d + 1).ok_or_else(|| {
                BasisError::InvalidKnotVector(format!(
                    "tensor margin knot vector of length {} is too short for degree {d}",
                    k.len()
                ))
            })
        })
        .collect::<Result<Vec<usize>, _>>()?;
    let total: usize = widths.iter().product();
    let mut blocks = Vec::<Array2<f64>>::new();
    for (j, set) in margins.iter().enumerate() {
        if set.is_empty() {
            continue;
        }
        if knot_valued[j] {
            return Err(BasisError::InvalidInput(format!(
                "tensor margin {j} is not an open B-spline, so its coefficients are not control \
                 points and cannot certify a shape"
            )));
        }
        let Some(cone) = bspline_shape_set_linear_constraints(knots[j].view(), degrees[j], *set)?
        else {
            continue;
        };
        if cone.a.ncols() != widths[j] {
            return Err(BasisError::DimensionMismatch(format!(
                "tensor margin {j} cone has {} columns but the margin has {} coefficients",
                cone.a.ncols(),
                widths[j]
            )));
        }
        let outer: usize = widths[..j].iter().product();
        let inner: usize = widths[j + 1..].iter().product();
        let m = cone.a.nrows();
        let mut block = Array2::<f64>::zeros((outer * m * inner, total));
        for o in 0..outer {
            for r in 0..m {
                for i in 0..inner {
                    let row = (o * m + r) * inner + i;
                    for c in 0..widths[j] {
                        let value = cone.a[[r, c]];
                        if value != 0.0 {
                            block[[row, (o * widths[j] + c) * inner + i]] = value;
                        }
                    }
                }
            }
        }
        blocks.push(block);
    }
    if blocks.is_empty() {
        return Ok(None);
    }
    let rows: usize = blocks.iter().map(Array2::nrows).sum();
    let mut a = Array2::<f64>::zeros((rows, total));
    let mut offset = 0usize;
    for block in blocks {
        let m = block.nrows();
        a.slice_mut(s![offset..offset + m, ..]).assign(&block);
        offset += m;
    }
    Ok(Some(LinearInequalityConstraints {
        a,
        b: Array1::zeros(rows),
    }))
}

/// Replicate a term-local shape realization over the level blocks of a
/// factor `by=` smooth: coordinate bounds tile, cone rows go block-diagonal,
/// so every level's curve carries the full constraint independently.
pub(super) fn replicate_shape_constraints_over_levels(
    lower_bounds: Option<&Array1<f64>>,
    linear: Option<&LinearInequalityConstraints>,
    block_width: usize,
    levels: usize,
) -> (Option<Array1<f64>>, Option<LinearInequalityConstraints>) {
    let bounds = lower_bounds.map(|lb| {
        let mut tiled = Array1::<f64>::from_elem(block_width * levels, f64::NEG_INFINITY);
        for level in 0..levels {
            tiled
                .slice_mut(s![level * block_width..(level + 1) * block_width])
                .assign(lb);
        }
        tiled
    });
    let rows = linear.map(|c| {
        let m = c.a.nrows();
        let mut a = Array2::<f64>::zeros((m * levels, block_width * levels));
        let mut b = Array1::<f64>::zeros(m * levels);
        for level in 0..levels {
            a.slice_mut(s![
                level * m..(level + 1) * m,
                level * block_width..(level + 1) * block_width
            ])
            .assign(&c.a);
            b.slice_mut(s![level * m..(level + 1) * m]).assign(&c.b);
        }
        LinearInequalityConstraints { a, b }
    });
    (bounds, rows)
}

pub(super) fn shape_order_and_sign(shape: ShapeConstraint) -> Option<(usize, f64)> {
    match shape {
        ShapeConstraint::None => None,
        ShapeConstraint::MonotoneIncreasing => Some((1, 1.0)),
        ShapeConstraint::MonotoneDecreasing => Some((1, -1.0)),
        ShapeConstraint::Convex => Some((2, 1.0)),
        ShapeConstraint::Concave => Some((2, -1.0)),
    }
}

pub fn shape_lower_bounds_local(shape: ShapeConstraint, dim: usize) -> Option<Array1<f64>> {
    let (order, _) = shape_order_and_sign(shape)?;
    box_lower_bounds_local(order, dim)
}

/// Coordinate bounds of the box chart of a derivative order: every coordinate
/// from `order` on is a (scaled) derivative control and must be non-negative.
pub(super) fn box_lower_bounds_local(order: usize, dim: usize) -> Option<Array1<f64>> {
    if dim <= order {
        return None;
    }
    let mut lb = Array1::<f64>::from_elem(dim, f64::NEG_INFINITY);
    for j in order..dim {
        lb[j] = 0.0;
    }
    Some(lb)
}

/// First-derivative control denominators for an open B-spline basis.
///
/// For `f = sum_i beta_i N_{i,d}`, the derivative control multiplying the
/// degree-`d - 1` basis function is
///
/// `d * (beta[i + 1] - beta[i]) / (t[i + d + 1] - t[i + 1])`.
///
/// Dividing every denominator by `d` gives the adjacent Greville-abscissa
/// gaps, but computing the gaps directly from knot differences avoids loss of
/// translation invariance from subtracting two separately averaged abscissae.
pub(crate) fn bspline_first_derivative_control_spans(
    knots: ArrayView1<'_, f64>,
    degree: usize,
) -> Result<Array1<f64>, BasisError> {
    if degree == 0 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let required = degree
        .checked_add(1)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| BasisError::InvalidInput("B-spline degree overflows usize".to_string()))?;
    if knots.len() < required {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required,
            provided: knots.len(),
        });
    }
    if knots.iter().any(|value| !value.is_finite()) {
        return Err(BasisError::InvalidKnotVector(
            "knot vector contains non-finite (NaN or Infinity) values".to_string(),
        ));
    }
    for pair in knots.windows(2) {
        if pair[0] > pair[1] {
            return Err(BasisError::InvalidKnotVector(
                "knot vector is not non-decreasing".to_string(),
            ));
        }
    }

    let coefficient_count = knots.len() - degree - 1;
    let mut spans = Array1::<f64>::zeros(coefficient_count.saturating_sub(1));
    let degree_scale = degree as f64;
    for i in 0..spans.len() {
        let width = knots[i + degree + 1] - knots[i + 1];
        if !width.is_finite() || width <= 0.0 {
            return Err(BasisError::InvalidKnotVector(format!(
                "shape-constrained derivative control span t[{}]-t[{}]={width:.3e} must be finite and positive",
                i + degree + 1,
                i + 1,
            )));
        }
        spans[i] = width / degree_scale;
    }
    Ok(spans)
}

/// Exact continuum shape cone for raw open-B-spline control coefficients.
///
/// The returned rows describe `A * beta >= 0`. Monotonicity rows constrain
/// consecutive control-point differences. Curvature rows constrain consecutive
/// first-derivative controls, using the exact knot-dependent denominators. As a
/// result, the number and values of the rows depend only on the realized spline
/// chart and never on an evaluation grid.
///
/// `ShapeConstraint::None` returns `None`. A nontrivial shape on an affine basis
/// can legitimately return a zero-row constraint: an affine function is both
/// convex and concave without imposing a coefficient restriction.
pub fn bspline_shape_linear_constraints(
    knots: ArrayView1<'_, f64>,
    degree: usize,
    shape: ShapeConstraint,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    let Some((order, sign)) = shape_order_and_sign(shape) else {
        return Ok(None);
    };
    let spans = bspline_first_derivative_control_spans(knots, degree)?;
    let coefficient_count = spans.len() + 1;
    let row_count = coefficient_count.saturating_sub(order);
    let mut a = Array2::<f64>::zeros((row_count, coefficient_count));

    match order {
        1 => {
            for row in 0..row_count {
                a[[row, row]] = -sign;
                a[[row, row + 1]] = sign;
            }
        }
        2 => {
            for row in 0..row_count {
                let left = spans[row];
                let right = spans[row + 1];
                // Positive scaling by left*right turns the reciprocal form
                // [1/left, -(1/left + 1/right), 1/right] into this stable row.
                // Divide both spans by their maximum before adding them so a
                // valid very-large-domain spline cannot overflow in `left + right`.
                let span_scale = left.max(right);
                let left = left / span_scale;
                let right = right / span_scale;
                a[[row, row]] = sign * right;
                a[[row, row + 1]] = -sign * (left + right);
                a[[row, row + 2]] = sign * left;
            }
        }
        _ => {
            return Err(BasisError::InvalidInput(format!(
                "unsupported B-spline shape derivative order {order}"
            )));
        }
    }

    for mut row in a.rows_mut() {
        let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(BasisError::InvalidInput(
                "shape-constraint row has no finite direction".to_string(),
            ));
        }
        row.mapv_inplace(|value| value / norm);
    }

    Ok(Some(LinearInequalityConstraints {
        b: Array1::zeros(row_count),
        a,
    }))
}

pub fn linear_constraints_from_lower_bounds_global(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}

pub fn merge_linear_constraints_global(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    match (first, second) {
        (None, None) => Ok(None),
        (Some(c), None) | (None, Some(c)) => {
            if c.a.nrows() != c.b.len() {
                return Err(BasisError::DimensionMismatch(format!(
                    "linear constraint has {} rows but {} right-hand-side values",
                    c.a.nrows(),
                    c.b.len()
                )));
            }
            Ok(Some(c))
        }
        (Some(a), Some(b)) => {
            if a.a.ncols() != b.a.ncols() {
                return Err(BasisError::DimensionMismatch(format!(
                    "cannot merge linear constraints with {} and {} columns",
                    a.a.ncols(),
                    b.a.ncols()
                )));
            }
            if a.a.nrows() != a.b.len() || b.a.nrows() != b.b.len() {
                return Err(BasisError::DimensionMismatch(format!(
                    "cannot merge linear constraints with row/RHS shapes {}x{}/{} and {}x{}/{}",
                    a.a.nrows(),
                    a.a.ncols(),
                    a.b.len(),
                    b.a.nrows(),
                    b.a.ncols(),
                    b.b.len()
                )));
            }
            let m1 = a.a.nrows();
            let m2 = b.a.nrows();
            let p = a.a.ncols();
            let mut mat = Array2::<f64>::zeros((m1 + m2, p));
            mat.slice_mut(s![0..m1, ..]).assign(&a.a);
            mat.slice_mut(s![m1..(m1 + m2), ..]).assign(&b.a);
            let mut rhs = Array1::<f64>::zeros(m1 + m2);
            rhs.slice_mut(s![0..m1]).assign(&a.b);
            rhs.slice_mut(s![m1..(m1 + m2)]).assign(&b.b);
            Ok(Some(LinearInequalityConstraints { a: mat, b: rhs }))
        }
    }
}

#[cfg(test)]
mod exact_bspline_shape_tests {
    use super::*;
    use ndarray::array;

    fn irregular_cubic_knots() -> Array1<f64> {
        array![0.0, 0.0, 0.0, 0.0, 0.08, 0.37, 0.62, 1.0, 1.0, 1.0, 1.0]
    }

    #[test]
    fn all_shape_rows_are_exact_derivative_control_cones() {
        let knots = irregular_cubic_knots();
        let p = knots.len() - 4;
        let increasing =
            bspline_shape_linear_constraints(knots.view(), 3, ShapeConstraint::MonotoneIncreasing)
                .unwrap()
                .unwrap();
        let decreasing =
            bspline_shape_linear_constraints(knots.view(), 3, ShapeConstraint::MonotoneDecreasing)
                .unwrap()
                .unwrap();
        let convex = bspline_shape_linear_constraints(knots.view(), 3, ShapeConstraint::Convex)
            .unwrap()
            .unwrap();
        let concave = bspline_shape_linear_constraints(knots.view(), 3, ShapeConstraint::Concave)
            .unwrap()
            .unwrap();

        assert_eq!(increasing.a.dim(), (p - 1, p));
        assert_eq!(convex.a.dim(), (p - 2, p));
        assert_eq!(decreasing.a, -&increasing.a);
        assert_eq!(concave.a, -&convex.a);
        assert_eq!(increasing.b, Array1::<f64>::zeros(p - 1));
        assert_eq!(convex.b, Array1::<f64>::zeros(p - 2));

        let spans = bspline_first_derivative_control_spans(knots.view(), 3).unwrap();
        for row in 0..convex.a.nrows() {
            let scale = spans[row].max(spans[row + 1]);
            let left = spans[row] / scale;
            let right = spans[row + 1] / scale;
            let expected = array![right, -(left + right), left];
            let expected = &expected / expected.dot(&expected).sqrt();
            assert_eq!(convex.a.slice(s![row, row..row + 3]), expected.view());
        }
    }

    #[test]
    fn rows_are_invariant_to_positive_affine_knot_units() {
        let knots = irregular_cubic_knots();
        let shifted = knots.mapv(|value| 17.0 + 9.0 * value);
        for shape in [
            ShapeConstraint::MonotoneIncreasing,
            ShapeConstraint::MonotoneDecreasing,
            ShapeConstraint::Convex,
            ShapeConstraint::Concave,
        ] {
            let base = bspline_shape_linear_constraints(knots.view(), 3, shape)
                .unwrap()
                .unwrap();
            let transformed = bspline_shape_linear_constraints(shifted.view(), 3, shape)
                .unwrap()
                .unwrap();
            for (left, right) in base.a.iter().zip(transformed.a.iter()) {
                assert!((left - right).abs() <= 32.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn affine_linear_spline_has_vacuous_curvature_cone() {
        let knots = array![0.0, 0.0, 1.0, 1.0];
        for shape in [ShapeConstraint::Convex, ShapeConstraint::Concave] {
            let constraints = bspline_shape_linear_constraints(knots.view(), 1, shape)
                .unwrap()
                .unwrap();
            assert_eq!(constraints.a.dim(), (0, 2));
            assert!(constraints.b.is_empty());
            assert!(shape_lower_bounds_local(shape, 2).is_none());
        }
    }

    #[test]
    fn derivative_control_geometry_rejects_collapsed_spans() {
        let knots = array![0.0, 0.0, 0.5, 0.5, 1.0, 1.0];
        let err =
            bspline_shape_linear_constraints(knots.view(), 1, ShapeConstraint::MonotoneIncreasing)
                .unwrap_err();
        assert!(err.to_string().contains("must be finite and positive"));
    }

    #[test]
    fn incompatible_constraint_blocks_are_errors_not_dropped_cones() {
        let left = LinearInequalityConstraints {
            a: Array2::eye(2),
            b: Array1::zeros(2),
        };
        let right = LinearInequalityConstraints {
            a: Array2::eye(3),
            b: Array1::zeros(3),
        };
        let err = merge_linear_constraints_global(Some(left), Some(right)).unwrap_err();
        assert!(err.to_string().contains("cannot merge linear constraints"));
    }
}

#[cfg(test)]
mod shape_spec_tests {
    use super::*;
    use ndarray::array;

    fn irregular_cubic_knots() -> Array1<f64> {
        array![0.0, 0.0, 0.0, 0.0, 0.08, 0.37, 0.62, 1.0, 1.0, 1.0, 1.0]
    }

    fn set(atoms: &[ShapeConstraint]) -> ShapeSet {
        ShapeSet::from_atoms(atoms).unwrap()
    }

    /// `kron(left, right)` with `left` outermost, the row-major tensor layout.
    fn kron(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
        let (lr, lc) = left.dim();
        let (rr, rc) = right.dim();
        let mut out = Array2::<f64>::zeros((lr * rr, lc * rc));
        for i in 0..lr {
            for j in 0..lc {
                out.slice_mut(s![i * rr..(i + 1) * rr, j * rc..(j + 1) * rc])
                    .assign(&(right * left[[i, j]]));
            }
        }
        out
    }

    #[test]
    fn dsl_lists_parse_in_every_spelling_and_round_trip() {
        use ShapeConstraint::{Concave, Convex, MonotoneIncreasing};
        let expected = ShapeExpr::List(vec![
            ShapeExpr::Atom(MonotoneIncreasing),
            ShapeExpr::Atom(Concave),
        ]);
        for raw in [
            "[monotone_increasing, concave]",
            "c('increasing', \"concave\")",
            "(mono_inc,ccv)",
            " [ Monotone-Increasing , concave ] ",
        ] {
            assert_eq!(parse_shape_expr(raw).unwrap(), expected, "{raw}");
        }
        let nested = parse_shape_expr("[[increasing, concave], none]").unwrap();
        assert_eq!(
            parse_shape_expr(&nested.to_string()).unwrap(),
            nested,
            "canonical text must re-parse to the same tree"
        );
        assert_eq!(parse_shape_expr("convex").unwrap(), ShapeExpr::Atom(Convex));
        for bad in [
            "[increasing",
            "[increasing concave]",
            "increasing]",
            "'convex",
            "[,]",
        ] {
            assert!(parse_shape_expr(bad).is_err(), "{bad} must be rejected");
        }
        let json = serde_json::json!([["monotone_increasing", "concave"], null]);
        assert_eq!(
            shape_expr_from_json(&json).unwrap().to_string(),
            nested.to_string()
        );
    }

    #[test]
    fn one_dimensional_conjunctions_resolve_and_contradictions_explain_themselves() {
        use ShapeConstraint::{Concave, Convex, MonotoneIncreasing};
        let spec = parse_shape_spec("[monotone_increasing, concave]", None).unwrap();
        assert_eq!(spec, ShapeSpec::Joint(set(&[MonotoneIncreasing, Concave])));
        assert_eq!(spec.single_atom(), None);
        assert_eq!(spec.to_string(), "[monotone_increasing, concave]");
        // Repeats collapse; a lone atom is the legacy single-shape request.
        let single = parse_shape_spec("[convex, convex]", None).unwrap();
        assert_eq!(single, Convex);
        assert_eq!(parse_shape_spec("none", None).unwrap(), ShapeSpec::None);
        assert_eq!(
            parse_shape_spec("[none, none]", None).unwrap(),
            ShapeSpec::None
        );

        let err = parse_shape_spec("[increasing, decreasing]", None).unwrap_err();
        assert!(
            err.contains("contradictory") && err.contains("constant"),
            "{err}"
        );
        let err = parse_shape_spec("[convex, concave]", None).unwrap_err();
        assert!(
            err.contains("contradictory") && err.contains("affine"),
            "{err}"
        );
        let err = parse_shape_spec("[[increasing], none]", None).unwrap_err();
        assert!(err.contains("te() tensor"), "{err}");
    }

    #[test]
    fn tensor_shapes_need_one_entry_per_margin() {
        use ShapeConstraint::{Convex, MonotoneDecreasing, MonotoneIncreasing};
        let spec =
            parse_shape_spec("[monotone_increasing, [decreasing, convex]]", Some(2)).unwrap();
        assert_eq!(
            spec,
            ShapeSpec::PerMargin(vec![
                set(&[MonotoneIncreasing]),
                set(&[MonotoneDecreasing, Convex]),
            ])
        );
        assert_eq!(parse_shape_spec("none", Some(2)).unwrap(), ShapeSpec::None);
        let err = parse_shape_spec("monotone_increasing", Some(2)).unwrap_err();
        assert!(err.contains("shape=[monotone_increasing, none]"), "{err}");
        let err = parse_shape_spec("[increasing, none, none]", Some(2)).unwrap_err();
        assert!(
            err.contains("3 entries") && err.contains("2 margins"),
            "{err}"
        );
        let err = parse_shape_spec("[[convex, concave], none]", Some(2)).unwrap_err();
        assert!(err.contains("margin 0") && err.contains("affine"), "{err}");
    }

    #[test]
    fn persisted_shape_keeps_the_legacy_atom_encoding() {
        use ShapeConstraint::{Concave, Convex, MonotoneDecreasing, MonotoneIncreasing};
        let legacy: ShapeSpec = serde_json::from_str("\"MonotoneIncreasing\"").unwrap();
        assert_eq!(legacy, MonotoneIncreasing);
        assert_eq!(
            serde_json::to_string(&legacy).unwrap(),
            "\"MonotoneIncreasing\""
        );
        assert_eq!(serde_json::to_string(&ShapeSpec::None).unwrap(), "\"None\"");
        for spec in [
            ShapeSpec::Joint(set(&[MonotoneDecreasing, Convex])),
            ShapeSpec::PerMargin(vec![set(&[]), set(&[Concave])]),
        ] {
            let text = serde_json::to_string(&spec).unwrap();
            let back: ShapeSpec = serde_json::from_str(&text).unwrap();
            assert_eq!(back, spec, "{text}");
        }
        let err = serde_json::from_str::<ShapeSpec>("{\"joint\":[\"Convex\",\"Concave\"]}");
        assert!(err.is_err(), "a contradictory payload must not deserialize");
    }

    /// Controls `d_i` of the first derivative, rebuilt into coefficients
    /// `β_{i+1} = β_i + span_i d_i`.
    fn coefficients_from_derivative_controls(spans: &Array1<f64>, controls: &[f64]) -> Array1<f64> {
        let mut beta = Array1::<f64>::zeros(spans.len() + 1);
        for i in 0..spans.len() {
            beta[i + 1] = beta[i] + spans[i] * controls[i];
        }
        beta
    }

    #[test]
    fn monotone_curvature_cone_keeps_only_the_binding_monotone_row() {
        use ShapeConstraint::{Concave, Convex, MonotoneDecreasing, MonotoneIncreasing};
        let knots = irregular_cubic_knots();
        let p = knots.len() - 4;
        let spans = bspline_first_derivative_control_spans(knots.view(), 3).unwrap();
        for (monotone, curvature) in [
            (MonotoneIncreasing, Convex),
            (MonotoneIncreasing, Concave),
            (MonotoneDecreasing, Convex),
            (MonotoneDecreasing, Concave),
        ] {
            let reduced =
                bspline_shape_set_linear_constraints(knots.view(), 3, set(&[monotone, curvature]))
                    .unwrap()
                    .unwrap();
            assert_eq!(reduced.a.dim(), (p - 1, p), "{monotone:?}+{curvature:?}");
            let full = merge_linear_constraints_global(
                bspline_shape_linear_constraints(knots.view(), 3, monotone).unwrap(),
                bspline_shape_linear_constraints(knots.view(), 3, curvature).unwrap(),
            )
            .unwrap()
            .unwrap();
            // Every reduced row is a row of the full stack, so full ⊆ reduced.
            for row in reduced.a.rows() {
                assert!(full.a.rows().into_iter().any(|full_row| full_row == row));
            }
            // Conversely every extreme ray of the reduced cone lies in the
            // full cone: ordered derivative controls anchored at the binding
            // one with the monotone sign.
            let (_, m_sign) = shape_order_and_sign(monotone).unwrap();
            let (_, c_sign) = shape_order_and_sign(curvature).unwrap();
            let n = spans.len();
            for kink in 0..n {
                // Controls are 0 up to (or from) `kink` and then move away
                // from zero in the curvature direction, i.e. a ray of the
                // ordered cone whose binding control sits on its boundary.
                let controls: Vec<f64> = (0..n)
                    .map(|i| {
                        let active = if m_sign == c_sign {
                            i >= kink
                        } else {
                            i < kink
                        };
                        if active { m_sign } else { 0.0 }
                    })
                    .collect();
                let beta = coefficients_from_derivative_controls(&spans, &controls);
                assert!(reduced.a.dot(&beta).iter().all(|&v| v >= -1e-12));
                assert!(
                    full.a.dot(&beta).iter().all(|&v| v >= -1e-12),
                    "{monotone:?}+{curvature:?} ray {kink} escapes the full cone"
                );
            }
            // The kept row is genuinely binding: ordering the controls in the
            // curvature direction but giving the binding one the wrong sign
            // must be cut off.
            let wrong: Vec<f64> = (0..n)
                .map(|i| {
                    if m_sign == c_sign {
                        -m_sign + c_sign * 0.1 * i as f64
                    } else {
                        -m_sign - c_sign * 0.1 * (n - 1 - i) as f64
                    }
                })
                .collect();
            let beta = coefficients_from_derivative_controls(&spans, &wrong);
            let curvature_cone = bspline_shape_linear_constraints(knots.view(), 3, curvature)
                .unwrap()
                .unwrap();
            assert!(curvature_cone.a.dot(&beta).iter().all(|&v| v >= -1e-12));
            assert!(reduced.a.dot(&beta).iter().any(|&v| v < 0.0));
        }
    }

    #[test]
    fn tensor_cone_is_the_margin_cone_kroneckered_with_identities() {
        use ShapeConstraint::{Concave, MonotoneIncreasing};
        let kx = irregular_cubic_knots();
        let kz = array![0.0, 0.0, 0.0, 0.3, 0.55, 1.0, 1.0, 1.0];
        let (qx, qz) = (kx.len() - 4, kz.len() - 3);
        let knots = [kx.clone(), kz.clone()];
        let degrees = [3usize, 2];
        let open = [false, false];

        let ax = bspline_shape_linear_constraints(kx.view(), 3, MonotoneIncreasing)
            .unwrap()
            .unwrap()
            .a;
        let along_x = tensor_bspline_shape_linear_constraints(
            &knots,
            &degrees,
            &open,
            &[set(&[MonotoneIncreasing]), set(&[])],
        )
        .unwrap()
        .unwrap();
        assert_eq!(along_x.a, kron(&ax, &Array2::eye(qz)));

        let az = bspline_shape_linear_constraints(kz.view(), 2, Concave)
            .unwrap()
            .unwrap()
            .a;
        let along_z =
            tensor_bspline_shape_linear_constraints(&knots, &degrees, &open, &[set(&[]), set(&[Concave])])
                .unwrap()
                .unwrap();
        assert_eq!(along_z.a, kron(&Array2::eye(qx), &az));

        let both = tensor_bspline_shape_linear_constraints(
            &knots,
            &degrees,
            &open,
            &[set(&[MonotoneIncreasing]), set(&[Concave])],
        )
        .unwrap()
        .unwrap();
        assert_eq!(both.a.nrows(), along_x.a.nrows() + along_z.a.nrows());
        assert_eq!(both.a.slice(s![..along_x.a.nrows(), ..]), along_x.a);
        assert!(
            tensor_bspline_shape_linear_constraints(&knots, &degrees, &open, &[set(&[]), set(&[])])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn knot_valued_margin_sets_its_identity_width_and_refuses_a_shape() {
        use ShapeConstraint::MonotoneIncreasing;
        // Margin 0 an open cubic B-spline; margin 1 a cubic regression margin
        // whose recorded knots are its five coefficient values.
        let kx = irregular_cubic_knots();
        let kz = array![0.0, 0.2, 0.5, 0.8, 1.0];
        let qx = kx.len() - 4;
        let knots = [kx.clone(), kz.clone()];
        let degrees = [3usize, 3];
        let kinds = [false, true];
        let ax = bspline_shape_linear_constraints(kx.view(), 3, MonotoneIncreasing)
            .unwrap()
            .unwrap()
            .a;
        let along_x = tensor_bspline_shape_linear_constraints(
            &knots,
            &degrees,
            &kinds,
            &[set(&[MonotoneIncreasing]), set(&[])],
        )
        .unwrap()
        .unwrap();
        assert_eq!(along_x.a.ncols(), qx * kz.len());
        assert_eq!(along_x.a, kron(&ax, &Array2::eye(kz.len())));
        assert!(
            tensor_bspline_shape_linear_constraints(
                &knots,
                &degrees,
                &kinds,
                &[set(&[]), set(&[MonotoneIncreasing])],
            )
            .is_err()
        );
    }

    #[test]
    fn factor_level_replication_is_block_diagonal() {
        let cone = LinearInequalityConstraints {
            a: array![[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]],
            b: Array1::zeros(2),
        };
        let lb = array![f64::NEG_INFINITY, 0.0, 0.0];
        let (bounds, rows) = replicate_shape_constraints_over_levels(Some(&lb), Some(&cone), 3, 2);
        let bounds = bounds.unwrap();
        assert_eq!(bounds.slice(s![..3]), lb);
        assert_eq!(bounds.slice(s![3..]), lb);
        let rows = rows.unwrap();
        assert_eq!(rows.a, kron(&Array2::eye(2), &cone.a));
        assert_eq!(rows.b, Array1::<f64>::zeros(4));
    }
}

//! Restricted build-time differentiation for small row-program atoms.
//!
//! [`row_atom`] accepts one scalar expression and emits two backends from that
//! single source: a generic `JetScalar` evaluator for arbitrary derivative
//! order, and a straight-line scalar value/gradient/packed-Hessian schedule.
//! Symbolic zeros are removed before Rust/LLVM see the generated schedule, so
//! it carries neither runtime dependency masks nor the `0*x` work that ordinary
//! forward jets must preserve for IEEE-754 semantics.
//!
//! Local-coordinate programs can request `order2_at_zero`, `third_at_zero`,
//! and `fourth_at_zero`. Those surfaces differentiate the same expression,
//! substitute zero for every primary, canonicalize the remaining parameter
//! polynomial, and rebuild it as a multivariate Horner schedule. Their emitted
//! functions consequently accept only the runtime constants and directions.

use proc_macro::TokenStream;
use proc_macro2::{Ident, Literal, Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    BinOp, Expr, ExprBinary, ExprGroup, ExprLit, ExprParen, ExprPath, ExprUnary, Lit, Result,
    Token, UnOp, Visibility, braced, bracketed, parenthesized, parse_macro_input,
};

mod row_program;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Lowering {
    Generic,
    Order2,
    Third,
    Fourth,
    Order2AtZero,
    ThirdAtZero,
    FourthAtZero,
}

struct RowAtomInput {
    visibility: Visibility,
    name: Ident,
    lowerings: HashSet<Lowering>,
    primaries: Vec<Ident>,
    constants: Vec<Ident>,
    activity_constants: HashSet<usize>,
    scale_constants: HashSet<usize>,
    expression: Expr,
}

impl Parse for RowAtomInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let visibility = input.parse()?;
        input.parse::<Token![fn]>()?;
        let name = input.parse()?;
        let lowering_tokens;
        bracketed!(lowering_tokens in input);
        let mut lowerings = HashSet::new();
        for lowering in Punctuated::<Ident, Token![,]>::parse_terminated(&lowering_tokens)? {
            let lowering = match lowering.to_string().as_str() {
                "generic" => Lowering::Generic,
                "order2" => Lowering::Order2,
                "third" => Lowering::Third,
                "fourth" => Lowering::Fourth,
                "order2_at_zero" => Lowering::Order2AtZero,
                "third_at_zero" => Lowering::ThirdAtZero,
                "fourth_at_zero" => Lowering::FourthAtZero,
                _ => {
                    return Err(syn::Error::new_spanned(
                        lowering,
                        "row_atom lowerings are generic, order2, third, fourth, \
                         order2_at_zero, third_at_zero, and fourth_at_zero",
                    ));
                }
            };
            if !lowerings.insert(lowering) {
                return Err(lowering_tokens.error("row_atom lowering listed more than once"));
            }
        }
        if lowerings.is_empty() {
            return Err(lowering_tokens.error("row_atom requires at least one lowering"));
        }
        let arguments;
        parenthesized!(arguments in input);
        let mut primaries = Vec::new();
        while !arguments.is_empty() && !arguments.peek(Token![;]) {
            primaries.push(arguments.parse::<Ident>()?);
            if arguments.peek(Token![,]) {
                arguments.parse::<Token![,]>()?;
            } else {
                break;
            }
        }
        let mut constants = Vec::new();
        let mut activity_constants = HashSet::new();
        let mut scale_constants = HashSet::new();
        if arguments.peek(Token![;]) {
            arguments.parse::<Token![;]>()?;
            while !arguments.is_empty() {
                let constant = arguments.parse::<Ident>()?;
                arguments.parse::<Token![:]>()?;
                let kind = arguments.parse::<Ident>()?;
                let index = constants.len();
                match kind.to_string().as_str() {
                    "f64" => {}
                    "scale" => {
                        scale_constants.insert(index);
                    }
                    "bool" => {
                        activity_constants.insert(index);
                    }
                    _ => {
                        return Err(syn::Error::new_spanned(
                            kind,
                            "row_atom constants must be explicitly typed `f64`, `scale`, or `bool`",
                        ));
                    }
                }
                constants.push(constant);
                if arguments.peek(Token![,]) {
                    arguments.parse::<Token![,]>()?;
                } else {
                    break;
                }
            }
        }
        if !arguments.is_empty() {
            return Err(arguments.error("invalid row_atom argument list"));
        }
        if primaries.is_empty() {
            return Err(input.error("row_atom requires at least one primary"));
        }
        let mut bindings = HashSet::new();
        for binding in primaries.iter().chain(constants.iter()) {
            if !bindings.insert(binding.to_string()) {
                return Err(syn::Error::new_spanned(
                    binding,
                    "row_atom argument names must be unique",
                ));
            }
        }
        let body;
        braced!(body in input);
        let expression = body.parse()?;
        if !body.is_empty() {
            return Err(body.error("row_atom body must contain exactly one expression"));
        }
        Ok(Self {
            visibility,
            name,
            lowerings,
            primaries,
            constants,
            activity_constants,
            scale_constants,
            expression,
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Node {
    Constant(u64),
    Variable(usize),
    Parameter(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Neg(usize),
    Exp(usize),
    Ln(usize),
    Sqrt(usize),
    Recip(usize),
    Select(usize, usize, usize),
}

#[derive(Clone)]
struct Graph {
    nodes: Vec<Node>,
    interned: HashMap<Node, usize>,
    derivatives: HashMap<(usize, usize), usize>,
}

type Polynomial = BTreeMap<Vec<usize>, f64>;
type RingPolynomial = BTreeMap<Vec<usize>, f64>;

#[derive(Clone, Copy)]
enum ScaleDistribution {
    Value,
    Derivative,
}

/// Advance `values` to the next permutation in lexicographic order; `false`
/// once the last one has been reached.
fn next_permutation(values: &mut [usize]) -> bool {
    let Some(pivot) = (0..values.len().saturating_sub(1))
        .rev()
        .find(|&i| values[i] < values[i + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..values.len())
        .rev()
        .find(|&j| values[j] > values[pivot])
        .expect("a larger element follows the pivot");
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

impl Graph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            interned: HashMap::new(),
            derivatives: HashMap::new(),
        }
    }

    fn intern(&mut self, node: Node) -> usize {
        if let Some(&id) = self.interned.get(&node) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(node.clone());
        self.interned.insert(node, id);
        id
    }

    fn constant(&mut self, value: f64) -> usize {
        // Zero has one spelling. The algebra already discards the sign of a
        // zero (`x * 0` folds to `0` whatever `x` is), and a `-0.0` -- reached
        // by negating an identically-zero derivative -- would intern as a
        // second zero: a channel spelled `-0.0`, or two gates that differ only
        // in the sign of an inactive branch, which lower to two identical
        // `phi`s that LLVM does not unify and therefore two spills of one
        // value (measured on the cause-specific order-2 row).
        let value = if value == 0.0 { 0.0 } else { value };
        self.intern(Node::Constant(value.to_bits()))
    }

    fn constant_value(&self, id: usize) -> Option<f64> {
        match self.nodes[id] {
            Node::Constant(bits) => Some(f64::from_bits(bits)),
            _ => None,
        }
    }

    fn is_zero(&self, id: usize) -> bool {
        self.constant_value(id).is_some_and(|value| value == 0.0)
    }

    fn is_one(&self, id: usize) -> bool {
        self.constant_value(id) == Some(1.0)
    }

    fn add(&mut self, left: usize, right: usize) -> usize {
        if self.is_zero(left) {
            return right;
        }
        if self.is_zero(right) {
            return left;
        }
        if left == right {
            let two = self.constant(2.0);
            return self.mul(two, left);
        }
        if let (Some(left), Some(right)) = (self.constant_value(left), self.constant_value(right)) {
            return self.constant(left + right);
        }
        if let (Node::Neg(left), Node::Neg(right)) =
            (self.nodes[left].clone(), self.nodes[right].clone())
        {
            let sum = self.add(left, right);
            return self.neg(sum);
        }
        if let Node::Neg(value) = self.nodes[left] {
            return self.sub(right, value);
        }
        if let Node::Neg(value) = self.nodes[right] {
            return self.sub(left, value);
        }
        if let Node::Sub(value, removed) = self.nodes[left]
            && removed == right
        {
            return value;
        }
        if let Node::Sub(value, removed) = self.nodes[right]
            && removed == left
        {
            return value;
        }
        self.intern(Node::Add(left, right))
    }

    fn sub(&mut self, left: usize, right: usize) -> usize {
        if self.is_zero(right) {
            return left;
        }
        if self.is_zero(left) {
            return self.neg(right);
        }
        if left == right {
            return self.constant(0.0);
        }
        if let (Some(left), Some(right)) = (self.constant_value(left), self.constant_value(right)) {
            return self.constant(left - right);
        }
        self.intern(Node::Sub(left, right))
    }

    fn mul(&mut self, left: usize, right: usize) -> usize {
        if self.is_zero(left) || self.is_zero(right) {
            return self.constant(0.0);
        }
        if self.is_one(left) {
            return right;
        }
        if self.is_one(right) {
            return left;
        }
        if let (Some(left), Some(right)) = (self.constant_value(left), self.constant_value(right)) {
            return self.constant(left * right);
        }
        // A coefficient of -1 is a sign, not a multiply.
        if self.constant_value(left) == Some(-1.0) {
            return self.neg(right);
        }
        if self.constant_value(right) == Some(-1.0) {
            return self.neg(left);
        }
        // Give negative coefficients the same sign representation as unary
        // negation, so shared signed channels cancel their signs exactly.
        if let Some(coefficient) = self.constant_value(left)
            && coefficient < 0.0
        {
            let magnitude = self.constant(-coefficient);
            let product = self.mul(magnitude, right);
            return self.neg(product);
        }
        if let Some(coefficient) = self.constant_value(right)
            && coefficient < 0.0
        {
            let magnitude = self.constant(-coefficient);
            let product = self.mul(left, magnitude);
            return self.neg(product);
        }
        if let Node::Neg(inner) = self.nodes[left] {
            let product = self.mul(inner, right);
            return self.neg(product);
        }
        if let Node::Neg(inner) = self.nodes[right] {
            let product = self.mul(left, inner);
            return self.neg(product);
        }
        let (left, right) = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        self.intern(Node::Mul(left, right))
    }

    fn div(&mut self, numerator: usize, denominator: usize) -> usize {
        if self.is_zero(numerator) {
            return self.constant(0.0);
        }
        if self.is_one(denominator) {
            return numerator;
        }
        if let (Some(numerator), Some(denominator)) = (
            self.constant_value(numerator),
            self.constant_value(denominator),
        ) {
            return self.constant(numerator / denominator);
        }
        self.intern(Node::Div(numerator, denominator))
    }

    fn neg(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(-value);
        }
        if let Node::Neg(inner) = self.nodes[value] {
            return inner;
        }
        // A negation crosses an activity gate only when it CANCELS one inside
        // (the Hessian of a reciprocal is `-(...)` and the channel that reads
        // it negates again: pushed through, the two annihilate and the gate
        // carries `w·r·r` with no sign flip at all). A fresh negation stays
        // OUTSIDE. Pushed in, it would make the gate's positive value -- which
        // the value channel subtracts -- and its negated twin two distinct
        // gated quantities, two `phi`s and two sign flips; kept outside, one
        // value crosses the gate, `x - Select` stays a subtraction that pairs
        // with the gradient's, and the derivative channels share one negation.
        // Measured on the cause-specific order-2 row: the hand kernel executes
        // two sign flips; pushing every negation through executed six.
        if let Node::Select(activity, when_true, when_false) = self.nodes[value]
            && matches!(self.nodes[when_true], Node::Neg(_))
        {
            let when_true = self.neg(when_true);
            let when_false = self.neg(when_false);
            return self.select(activity, when_true, when_false);
        }
        self.intern(Node::Neg(value))
    }

    fn exp(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(value.exp());
        }
        self.intern(Node::Exp(value))
    }

    fn ln(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(value.ln());
        }
        self.intern(Node::Ln(value))
    }

    fn sqrt(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(value.sqrt());
        }
        self.intern(Node::Sqrt(value))
    }

    fn recip(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(value.recip());
        }
        if let Node::Recip(inner) = self.nodes[value] {
            return inner;
        }
        self.intern(Node::Recip(value))
    }

    fn select(&mut self, activity: usize, when_true: usize, when_false: usize) -> usize {
        if when_true == when_false {
            return when_true;
        }
        self.intern(Node::Select(activity, when_true, when_false))
    }

    fn guard_activities(
        &mut self,
        id: usize,
        activity_constants: &HashSet<usize>,
        memo: &mut HashMap<usize, usize>,
    ) -> usize {
        if let Some(&guarded) = memo.get(&id) {
            return guarded;
        }
        let node = self.nodes[id].clone();
        let guarded = match node {
            Node::Constant(_) | Node::Variable(_) | Node::Parameter(_) => id,
            Node::Add(left, right) => {
                let left = self.guard_activities(left, activity_constants, memo);
                let right = self.guard_activities(right, activity_constants, memo);
                self.add(left, right)
            }
            Node::Sub(left, right) => {
                let left = self.guard_activities(left, activity_constants, memo);
                let right = self.guard_activities(right, activity_constants, memo);
                self.sub(left, right)
            }
            Node::Mul(left, right) => {
                let left = self.guard_activities(left, activity_constants, memo);
                let right = self.guard_activities(right, activity_constants, memo);
                let activity_side = match (self.nodes[left].clone(), self.nodes[right].clone()) {
                    (Node::Parameter(index), _) if activity_constants.contains(&index) => {
                        Some((index, right))
                    }
                    (_, Node::Parameter(index)) if activity_constants.contains(&index) => {
                        Some((index, left))
                    }
                    _ => None,
                };
                if let Some((activity, value)) = activity_side {
                    let zero = self.constant(0.0);
                    self.select(activity, value, zero)
                } else {
                    self.mul(left, right)
                }
            }
            Node::Div(left, right) => {
                let left = self.guard_activities(left, activity_constants, memo);
                let right = self.guard_activities(right, activity_constants, memo);
                self.div(left, right)
            }
            Node::Neg(value) => {
                let value = self.guard_activities(value, activity_constants, memo);
                self.neg(value)
            }
            Node::Exp(value) => {
                let value = self.guard_activities(value, activity_constants, memo);
                self.exp(value)
            }
            Node::Ln(value) => {
                let value = self.guard_activities(value, activity_constants, memo);
                self.ln(value)
            }
            Node::Sqrt(value) => {
                let value = self.guard_activities(value, activity_constants, memo);
                self.sqrt(value)
            }
            Node::Recip(value) => {
                let value = self.guard_activities(value, activity_constants, memo);
                self.recip(value)
            }
            Node::Select(activity, when_true, when_false) => {
                let when_true = self.guard_activities(when_true, activity_constants, memo);
                let when_false = self.guard_activities(when_false, activity_constants, memo);
                self.select(activity, when_true, when_false)
            }
        };
        memo.insert(id, guarded);
        guarded
    }

    fn push_scale(&mut self, scale: usize, value: usize, distribution: ScaleDistribution) -> usize {
        match self.nodes[value].clone() {
            Node::Add(left, right) => {
                let left = self.push_scale(scale, left, distribution);
                let right = self.push_scale(scale, right, distribution);
                self.add(left, right)
            }
            Node::Sub(left, right) => {
                let left = self.push_scale(scale, left, distribution);
                let right = self.push_scale(scale, right, distribution);
                self.sub(left, right)
            }
            Node::Neg(value) => {
                let value = self.push_scale(scale, value, distribution);
                self.neg(value)
            }
            Node::Select(activity, when_true, when_false) => {
                let (when_true, when_false) = match distribution {
                    ScaleDistribution::Value => {
                        (self.mul(scale, when_true), self.mul(scale, when_false))
                    }
                    ScaleDistribution::Derivative => (
                        self.push_scale(scale, when_true, distribution),
                        self.push_scale(scale, when_false, distribution),
                    ),
                };
                self.select(activity, when_true, when_false)
            }
            Node::Mul(left, right) => {
                let (coefficient, remainder) = if matches!(self.nodes[left], Node::Parameter(_)) {
                    (left, right)
                } else if matches!(self.nodes[right], Node::Parameter(_)) {
                    (right, left)
                } else {
                    (left, right)
                };
                let scaled_coefficient = self.mul(scale, coefficient);
                self.mul(scaled_coefficient, remainder)
            }
            _ => self.mul(scale, value),
        }
    }

    fn distribute_scales(
        &mut self,
        id: usize,
        scale_constants: &HashSet<usize>,
        distribution: ScaleDistribution,
        memo: &mut HashMap<usize, usize>,
    ) -> usize {
        if let Some(&normalized) = memo.get(&id) {
            return normalized;
        }
        let node = self.nodes[id].clone();
        let normalized = match node {
            Node::Constant(_) | Node::Variable(_) | Node::Parameter(_) => id,
            Node::Add(left, right) => {
                let left = self.distribute_scales(left, scale_constants, distribution, memo);
                let right = self.distribute_scales(right, scale_constants, distribution, memo);
                self.add(left, right)
            }
            Node::Sub(left, right) => {
                let left = self.distribute_scales(left, scale_constants, distribution, memo);
                let right = self.distribute_scales(right, scale_constants, distribution, memo);
                self.sub(left, right)
            }
            Node::Mul(left, right) => {
                let left = self.distribute_scales(left, scale_constants, distribution, memo);
                let right = self.distribute_scales(right, scale_constants, distribution, memo);
                let scale_side = match (self.nodes[left].clone(), self.nodes[right].clone()) {
                    (Node::Parameter(index), _) if scale_constants.contains(&index) => {
                        Some((left, right))
                    }
                    (_, Node::Parameter(index)) if scale_constants.contains(&index) => {
                        Some((right, left))
                    }
                    _ => None,
                };
                if let Some((scale, value)) = scale_side {
                    self.push_scale(scale, value, distribution)
                } else {
                    self.mul(left, right)
                }
            }
            Node::Div(left, right) => {
                let left = self.distribute_scales(left, scale_constants, distribution, memo);
                let right = self.distribute_scales(right, scale_constants, distribution, memo);
                self.div(left, right)
            }
            Node::Neg(value) => {
                let value = self.distribute_scales(value, scale_constants, distribution, memo);
                self.neg(value)
            }
            Node::Exp(value) => {
                let value = self.distribute_scales(value, scale_constants, distribution, memo);
                self.exp(value)
            }
            Node::Ln(value) => {
                let value = self.distribute_scales(value, scale_constants, distribution, memo);
                self.ln(value)
            }
            Node::Sqrt(value) => {
                let value = self.distribute_scales(value, scale_constants, distribution, memo);
                self.sqrt(value)
            }
            Node::Recip(value) => {
                let value = self.distribute_scales(value, scale_constants, distribution, memo);
                self.recip(value)
            }
            Node::Select(activity, when_true, when_false) => {
                let when_true =
                    self.distribute_scales(when_true, scale_constants, distribution, memo);
                let when_false =
                    self.distribute_scales(when_false, scale_constants, distribution, memo);
                self.select(activity, when_true, when_false)
            }
        };
        memo.insert(id, normalized);
        normalized
    }

    fn substitute_zero_primaries(&mut self, id: usize, memo: &mut HashMap<usize, usize>) -> usize {
        if let Some(&specialized) = memo.get(&id) {
            return specialized;
        }
        let node = self.nodes[id].clone();
        let specialized = match node {
            Node::Constant(_) | Node::Parameter(_) => id,
            Node::Variable(_) => self.constant(0.0),
            Node::Add(left, right) => {
                let left = self.substitute_zero_primaries(left, memo);
                let right = self.substitute_zero_primaries(right, memo);
                self.add(left, right)
            }
            Node::Sub(left, right) => {
                let left = self.substitute_zero_primaries(left, memo);
                let right = self.substitute_zero_primaries(right, memo);
                self.sub(left, right)
            }
            Node::Mul(left, right) => {
                let left = self.substitute_zero_primaries(left, memo);
                let right = self.substitute_zero_primaries(right, memo);
                self.mul(left, right)
            }
            Node::Div(left, right) => {
                let left = self.substitute_zero_primaries(left, memo);
                let right = self.substitute_zero_primaries(right, memo);
                self.div(left, right)
            }
            Node::Neg(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.neg(value)
            }
            Node::Exp(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.exp(value)
            }
            Node::Ln(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.ln(value)
            }
            Node::Sqrt(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.sqrt(value)
            }
            Node::Recip(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.recip(value)
            }
            Node::Select(activity, when_true, when_false) => {
                let when_true = self.substitute_zero_primaries(when_true, memo);
                let when_false = self.substitute_zero_primaries(when_false, memo);
                self.select(activity, when_true, when_false)
            }
        };
        memo.insert(id, specialized);
        specialized
    }

    fn polynomial(
        &self,
        id: usize,
        parameter_count: usize,
        memo: &mut HashMap<usize, Option<Polynomial>>,
    ) -> Option<Polynomial> {
        if let Some(polynomial) = memo.get(&id) {
            return polynomial.clone();
        }
        let zero_exponents = || vec![0; parameter_count];
        let polynomial = match self.nodes[id].clone() {
            Node::Constant(bits) => {
                let value = f64::from_bits(bits);
                let mut polynomial = Polynomial::new();
                if value != 0.0 {
                    polynomial.insert(zero_exponents(), value);
                }
                Some(polynomial)
            }
            Node::Parameter(parameter) => {
                let mut exponents = zero_exponents();
                exponents[parameter] = 1;
                Some([(exponents, 1.0)].into_iter().collect())
            }
            Node::Variable(_)
            | Node::Exp(_)
            | Node::Ln(_)
            | Node::Sqrt(_)
            | Node::Recip(_)
            | Node::Select(_, _, _) => None,
            Node::Neg(value) => self
                .polynomial(value, parameter_count, memo)
                .map(|mut value| {
                    for coefficient in value.values_mut() {
                        *coefficient = -*coefficient;
                    }
                    value
                }),
            Node::Add(left, right) | Node::Sub(left, right) => {
                let mut left = self.polynomial(left, parameter_count, memo)?;
                let right = self.polynomial(right, parameter_count, memo)?;
                let sign = if matches!(self.nodes[id], Node::Add(_, _)) {
                    1.0
                } else {
                    -1.0
                };
                for (exponents, coefficient) in right {
                    let total = left.entry(exponents).or_default();
                    *total += sign * coefficient;
                }
                left.retain(|_, coefficient| *coefficient != 0.0);
                Some(left)
            }
            Node::Mul(left, right) => {
                let left = self.polynomial(left, parameter_count, memo)?;
                let right = self.polynomial(right, parameter_count, memo)?;
                let mut product = Polynomial::new();
                for (left_exponents, left_coefficient) in &left {
                    for (right_exponents, right_coefficient) in &right {
                        let exponents = left_exponents
                            .iter()
                            .zip(right_exponents)
                            .map(|(left, right)| left + right)
                            .collect::<Vec<_>>();
                        *product.entry(exponents).or_default() +=
                            left_coefficient * right_coefficient;
                    }
                }
                product.retain(|_, coefficient| *coefficient != 0.0);
                Some(product)
            }
            Node::Div(numerator, denominator) => {
                let mut numerator = self.polynomial(numerator, parameter_count, memo)?;
                let denominator = self.polynomial(denominator, parameter_count, memo)?;
                let coefficient = denominator.get(&zero_exponents()).copied()?;
                if denominator.len() != 1 || coefficient == 0.0 {
                    None
                } else {
                    for value in numerator.values_mut() {
                        *value /= coefficient;
                    }
                    Some(numerator)
                }
            }
        };
        memo.insert(id, polynomial.clone());
        polynomial
    }

    /// Horner's rule over the parameters in order, with every power of a
    /// parameter taken from the graph's interned power table rather than as a
    /// chain of multiplications.
    ///
    /// A lowering emits several channels (value, gradient, Hessian) of one
    /// polynomial map, and a chain `r * (r * X)` is specific to the channel
    /// whose `X` it wraps, so no channel shares its powers with another; the
    /// interned `r * r` is one node every channel reads. On the Gaussian
    /// joint row that was the whole margin by which the hand kernel, which
    /// names `r²` once, beat the lowering (#932).
    fn polynomial_horner(&mut self, polynomial: &Polynomial, variables: &[usize]) -> usize {
        if polynomial.is_empty() {
            return self.constant(0.0);
        }
        if variables.is_empty() {
            return self.constant(*polynomial.values().next().expect("nonempty polynomial"));
        }
        let variable = variables[0];
        let parameter = self.intern(Node::Parameter(variable));
        let mut coefficients = BTreeMap::<usize, Polynomial>::new();
        for (exponents, coefficient) in polynomial {
            let exponent = exponents[variable];
            let mut coefficient_exponents = exponents.clone();
            coefficient_exponents[variable] = 0;
            coefficients
                .entry(exponent)
                .or_default()
                .insert(coefficient_exponents, *coefficient);
        }
        let mut descending = coefficients.iter().rev();
        let (&highest, leading) = descending.next().expect("nonempty coefficients");
        let mut result = self.polynomial_horner(leading, &variables[1..]);
        let mut previous = highest;
        for (&exponent, coefficient) in descending {
            if previous > exponent {
                let power = self.positive_integer_power(parameter, previous - exponent);
                result = self.mul(result, power);
            }
            let coefficient = self.polynomial_horner(coefficient, &variables[1..]);
            result = self.add(result, coefficient);
            previous = exponent;
        }
        if previous > 0 {
            let power = self.positive_integer_power(parameter, previous);
            result = self.mul(result, power);
        }
        result
    }

    /// Lower the at-zero channels together, counting shared work across the
    /// entire output. Start with the best common Horner order, then improve
    /// individual channels against that shared graph: one parameter order can
    /// prevent otherwise compatible products from being shared by different
    /// channels.
    ///
    /// Horner's rule is per channel, and which products two channels share
    /// depends on the variable order: with the observation weight outermost
    /// every channel ends in its own `w * (...)`, while an order that nests
    /// the weight inside lets `w * r`, `w * q` and `w * k` be one node each
    /// (the hand kernel of the Gaussian joint row names exactly those). The
    /// initial order is exhaustive: every order of the parameters is emitted on
    /// a scratch copy of the graph and the one with the fewest distinct
    /// multiplies (then the fewest additions, then the declaration order)
    /// is kept. At most 720 orders for six parameters; a lowering with more
    /// parameters keeps the declaration order.
    fn normalize_polynomials(&mut self, ids: &[usize], parameter_count: usize) -> Vec<usize> {
        let mut memo = HashMap::new();
        let polynomials = ids
            .iter()
            .map(|&id| self.polynomial(id, parameter_count, &mut memo))
            .collect::<Vec<_>>();
        let order = self.horner_order(&polynomials, parameter_count);
        let mut roots = ids
            .iter()
            .zip(&polynomials)
            .map(|(&id, polynomial)| match polynomial {
                Some(polynomial) => self.polynomial_horner(polynomial, &order),
                None => id,
            })
            .collect::<Vec<_>>();
        if parameter_count <= 6 {
            self.extract_shared_monomials(&polynomials, &mut roots, parameter_count);
            loop {
                let mut improved = false;
                for (index, polynomial) in polynomials.iter().enumerate() {
                    let Some(polynomial) = polynomial else {
                        continue;
                    };
                    // Symmetric duplicate channels must keep one shared root.
                    if polynomials[..index]
                        .iter()
                        .any(|other| other.as_ref() == Some(polynomial))
                    {
                        continue;
                    }
                    let copies = polynomials
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, other)| {
                            (other.as_ref() == Some(polynomial)).then_some(slot)
                        })
                        .collect::<Vec<_>>();
                    let mut best_cost = self.polynomial_schedule_cost(&roots);
                    let maximum_depth = best_cost.2;
                    let mut best_root = roots[index];
                    let mut expressions = vec![(None, polynomial.clone())];
                    // An output can itself be a useful common subexpression.
                    // For example, a scale Hessian contains its scale gradient
                    // plus a lower-degree correction. Expanding both into
                    // separate Horner trees would discard that shared work.
                    for (other_index, other) in polynomials.iter().enumerate() {
                        let Some(other) = other else {
                            continue;
                        };
                        if other == polynomial {
                            continue;
                        }
                        let Some((leading_powers, leading_coefficient)) = other.iter().next()
                        else {
                            continue;
                        };
                        for (powers, coefficient) in polynomial {
                            let Some(shift) = powers
                                .iter()
                                .zip(leading_powers)
                                .map(|(power, leading)| power.checked_sub(*leading))
                                .collect::<Option<Vec<_>>>()
                            else {
                                continue;
                            };
                            let scale = coefficient / leading_coefficient;
                            if !scale.is_finite() {
                                continue;
                            }
                            let mut remainder = polynomial.clone();
                            let mut exact_subset = true;
                            for (other_powers, other_coefficient) in other {
                                let shifted = other_powers
                                    .iter()
                                    .zip(&shift)
                                    .map(|(power, shift)| power + shift)
                                    .collect::<Vec<_>>();
                                if remainder.get(&shifted).copied()
                                    != Some(scale * other_coefficient)
                                {
                                    exact_subset = false;
                                    break;
                                }
                                remainder.remove(&shifted);
                            }
                            if exact_subset {
                                let multiplier =
                                    [(shift, scale)].into_iter().collect::<Polynomial>();
                                expressions
                                    .push((Some((roots[other_index], multiplier)), remainder));
                            }
                        }
                    }
                    let mut candidate_order = (0..parameter_count).collect::<Vec<_>>();
                    loop {
                        for (shared, remainder) in &expressions {
                            let remainder = self.polynomial_horner(remainder, &candidate_order);
                            let candidate = match shared {
                                Some((shared, multiplier)) => {
                                    let multiplier =
                                        self.polynomial_horner(multiplier, &candidate_order);
                                    let product = self.mul(*shared, multiplier);
                                    self.add(product, remainder)
                                }
                                None => remainder,
                            };
                            let mut candidate_roots = roots.clone();
                            for &slot in &copies {
                                candidate_roots[slot] = candidate;
                            }
                            let cost = self.polynomial_schedule_cost(&candidate_roots);
                            if cost < best_cost && cost.2 <= maximum_depth {
                                best_cost = cost;
                                best_root = candidate;
                            }
                        }
                        if !next_permutation(&mut candidate_order) {
                            break;
                        }
                    }
                    if best_root != roots[index] {
                        for &slot in &copies {
                            roots[slot] = best_root;
                        }
                        improved = true;
                    }
                }
                // Every accepted replacement strictly lowers the tuple of
                // nonnegative integer costs, so this reaches a fixed point.
                if !improved {
                    break;
                }
            }
        }
        roots
    }

    /// Introduce common products in all affected outputs together. A product
    /// such as w*q can cost one extra operation in the first rewritten output
    /// and save two in the next; independent channel improvements cannot cross
    /// that barrier. Monomial gcds expose those joint factoring opportunities.
    fn extract_shared_monomials(
        &mut self,
        polynomials: &[Option<Polynomial>],
        roots: &mut Vec<usize>,
        parameter_count: usize,
    ) {
        let terms = polynomials
            .iter()
            .flatten()
            .flat_map(|polynomial| polynomial.keys())
            .collect::<Vec<_>>();
        let mut factors = BTreeSet::new();
        for (index, left) in terms.iter().enumerate() {
            for right in &terms[index + 1..] {
                let factor = left
                    .iter()
                    .zip(*right)
                    .map(|(a, b)| (*a).min(*b))
                    .collect::<Vec<_>>();
                if factor.iter().sum::<usize>() >= 2 {
                    factors.insert(factor);
                }
            }
        }
        loop {
            let initial_cost = self.polynomial_schedule_cost(roots);
            let mut best_cost = initial_cost;
            let mut best_roots = roots.clone();
            for factor in &factors {
                let split = polynomials
                    .iter()
                    .map(|polynomial| {
                        polynomial.as_ref().map(|polynomial| {
                            let mut quotient = Polynomial::new();
                            let mut remainder = Polynomial::new();
                            for (powers, coefficient) in polynomial {
                                if let Some(powers) = powers
                                    .iter()
                                    .zip(factor)
                                    .map(|(power, factor)| power.checked_sub(*factor))
                                    .collect::<Option<Vec<_>>>()
                                {
                                    quotient.insert(powers, *coefficient);
                                } else {
                                    remainder.insert(powers.clone(), *coefficient);
                                }
                            }
                            (quotient, remainder)
                        })
                    })
                    .collect::<Vec<_>>();
                let factor_polynomial = [(factor.clone(), 1.0)].into_iter().collect::<Polynomial>();
                let mut order = (0..parameter_count).collect::<Vec<_>>();
                loop {
                    let shared = self.polynomial_horner(&factor_polynomial, &order);
                    let mut candidate = roots.clone();
                    for (slot, split) in split.iter().enumerate() {
                        let Some((quotient, remainder)) = split else {
                            continue;
                        };
                        if quotient.is_empty() {
                            continue;
                        }
                        let quotient = self.polynomial_horner(quotient, &order);
                        let remainder = self.polynomial_horner(remainder, &order);
                        let product = self.mul(shared, quotient);
                        candidate[slot] = self.add(product, remainder);
                    }
                    let cost = self.polynomial_schedule_cost(&candidate);
                    if cost < best_cost && cost.2 <= initial_cost.2 {
                        best_cost = cost;
                        best_roots = candidate;
                    }
                    if !next_permutation(&mut order) {
                        break;
                    }
                }
            }
            if best_cost == initial_cost {
                break;
            }
            *roots = best_roots;
        }
    }

    /// Sharing is useful only when it does not lengthen the longest serial
    /// arithmetic chain. Negation is a sign bit operation, not an arithmetic
    /// dependency stage. Total output depth breaks otherwise equal choices.
    fn polynomial_schedule_cost(&self, roots: &[usize]) -> (usize, usize, usize, usize) {
        fn depth(graph: &Graph, id: usize, memo: &mut HashMap<usize, usize>) -> usize {
            if let Some(&depth) = memo.get(&id) {
                return depth;
            }
            let result = if is_leaf(graph, id) {
                0
            } else {
                let inputs = children(graph, id)
                    .into_iter()
                    .map(|child| depth(graph, child, memo))
                    .max()
                    .unwrap_or(0);
                inputs + usize::from(!matches!(graph.nodes[id], Node::Neg(_)))
            };
            memo.insert(id, result);
            result
        }
        let mut memo = HashMap::new();
        let depths = roots
            .iter()
            .map(|&id| depth(self, id, &mut memo))
            .collect::<Vec<_>>();
        let (multiplies, additions) = self.operation_count(roots);
        (
            multiplies,
            additions,
            depths.iter().copied().max().unwrap_or(0),
            depths.iter().sum(),
        )
    }

    fn horner_order(
        &self,
        polynomials: &[Option<Polynomial>],
        parameter_count: usize,
    ) -> Vec<usize> {
        let declaration = (0..parameter_count).collect::<Vec<_>>();
        if parameter_count > 6 || polynomials.iter().all(Option::is_none) {
            return declaration;
        }
        let mut order = declaration.clone();
        let mut best: Option<((usize, usize), Vec<usize>)> = None;
        loop {
            let mut scratch = self.clone();
            let roots = polynomials
                .iter()
                .flatten()
                .map(|polynomial| scratch.polynomial_horner(polynomial, &order))
                .collect::<Vec<_>>();
            let cost = scratch.operation_count(&roots);
            if best.as_ref().is_none_or(|(best_cost, _)| cost < *best_cost) {
                best = Some((cost, order.clone()));
            }
            if !next_permutation(&mut order) {
                break;
            }
        }
        best.expect("at least the declaration order was tried").1
    }

    /// Distinct multiplies (and divisions) and distinct additions (and
    /// subtractions) reachable from `roots`; a node shared by several roots
    /// is counted once, which is what interning buys.
    fn operation_count(&self, roots: &[usize]) -> (usize, usize) {
        let mut seen = HashSet::new();
        let mut stack = roots.to_vec();
        let (mut multiplies, mut additions) = (0, 0);
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            match self.nodes[id] {
                Node::Mul(left, right) | Node::Div(left, right) => {
                    multiplies += 1;
                    stack.push(left);
                    stack.push(right);
                }
                Node::Add(left, right) | Node::Sub(left, right) => {
                    additions += 1;
                    stack.push(left);
                    stack.push(right);
                }
                Node::Neg(value)
                | Node::Exp(value)
                | Node::Ln(value)
                | Node::Sqrt(value)
                | Node::Recip(value) => stack.push(value),
                Node::Select(_, when_true, when_false) => {
                    stack.push(when_true);
                    stack.push(when_false);
                }
                Node::Constant(_) | Node::Variable(_) | Node::Parameter(_) => {}
            }
        }
        (multiplies, additions)
    }

    fn ring_polynomial(
        &self,
        id: usize,
        memo: &mut HashMap<usize, RingPolynomial>,
    ) -> RingPolynomial {
        if let Some(polynomial) = memo.get(&id) {
            return polynomial.clone();
        }
        let polynomial = match self.nodes[id].clone() {
            Node::Constant(bits) => {
                let value = f64::from_bits(bits);
                if value == 0.0 {
                    RingPolynomial::new()
                } else {
                    [(Vec::new(), value)].into_iter().collect()
                }
            }
            Node::Neg(value) => self
                .ring_polynomial(value, memo)
                .into_iter()
                .map(|(monomial, coefficient)| (monomial, -coefficient))
                .collect(),
            Node::Add(left, right) | Node::Sub(left, right) => {
                let mut polynomial = self.ring_polynomial(left, memo);
                let sign = if matches!(self.nodes[id], Node::Add(_, _)) {
                    1.0
                } else {
                    -1.0
                };
                for (monomial, coefficient) in self.ring_polynomial(right, memo) {
                    *polynomial.entry(monomial).or_default() += sign * coefficient;
                }
                polynomial.retain(|_, coefficient| *coefficient != 0.0);
                polynomial
            }
            Node::Mul(left, right) => {
                let left = self.ring_polynomial(left, memo);
                let right = self.ring_polynomial(right, memo);
                let mut product = RingPolynomial::new();
                for (left_factors, left_coefficient) in &left {
                    for (right_factors, right_coefficient) in &right {
                        let mut factors =
                            Vec::with_capacity(left_factors.len() + right_factors.len());
                        factors.extend_from_slice(left_factors);
                        factors.extend_from_slice(right_factors);
                        factors.sort_unstable();
                        *product.entry(factors).or_default() +=
                            left_coefficient * right_coefficient;
                    }
                }
                product.retain(|_, coefficient| *coefficient != 0.0);
                product
            }
            Node::Select(_, _, _) => {
                return [(vec![id], 1.0)].into_iter().collect();
            }
            Node::Variable(_)
            | Node::Parameter(_)
            | Node::Div(_, _)
            | Node::Exp(_)
            | Node::Ln(_)
            | Node::Sqrt(_)
            | Node::Recip(_) => [(vec![id], 1.0)].into_iter().collect(),
        };
        memo.insert(id, polynomial.clone());
        polynomial
    }

    fn normalize_ring(&mut self, id: usize) -> usize {
        let node = self.nodes[id].clone();
        if let Node::Select(activity, when_true, when_false) = node {
            let when_true = self.normalize_ring(when_true);
            let when_false = self.normalize_ring(when_false);
            return self.select(activity, when_true, when_false);
        }
        if let Node::Neg(value) = node
            && let Node::Select(activity, when_true, when_false) = self.nodes[value].clone()
        {
            let when_true = self.neg(when_true);
            let when_false = self.neg(when_false);
            let when_true = self.normalize_ring(when_true);
            let when_false = self.normalize_ring(when_false);
            return self.select(activity, when_true, when_false);
        }
        let polynomial = self.ring_polynomial(id, &mut HashMap::new());
        let mut sum = self.constant(0.0);
        for (factors, coefficient) in polynomial {
            let mut term = self.constant(coefficient);
            let mut cursor = 0;
            while cursor < factors.len() {
                let factor = factors[cursor];
                let mut end = cursor + 1;
                while end < factors.len() && factors[end] == factor {
                    end += 1;
                }
                let exponent = end - cursor;
                if exponent % 2 == 0 && !self.is_one(term) {
                    // Keep the coefficient in the product instead of forming
                    // the full unweighted square first. For finite c and h,
                    // (c*h)*h avoids the spurious overflow/underflow of
                    // c*(h*h), at the same multiplication count. The half
                    // power remains interned and shared across channels.
                    let half = self.positive_integer_power(factor, exponent / 2);
                    term = self.mul(term, half);
                    term = self.mul(term, half);
                } else {
                    let power = self.positive_integer_power(factor, exponent);
                    term = self.mul(term, power);
                }
                cursor = end;
            }
            sum = self.add(sum, term);
        }
        sum
    }

    fn positive_integer_power(&mut self, base: usize, exponent: usize) -> usize {
        assert!(exponent > 0, "ring powers require a positive exponent");
        if exponent == 1 {
            return base;
        }

        let half = self.positive_integer_power(base, exponent / 2);
        let square = self.mul(half, half);
        if exponent % 2 == 0 {
            square
        } else {
            self.mul(square, base)
        }
    }

    fn derivative(&mut self, id: usize, variable: usize) -> usize {
        if let Some(&derivative) = self.derivatives.get(&(id, variable)) {
            return derivative;
        }
        let node = self.nodes[id].clone();
        let derivative = match node {
            Node::Constant(_) | Node::Parameter(_) => self.constant(0.0),
            Node::Variable(axis) => self.constant(f64::from(axis == variable)),
            Node::Add(left, right) => {
                let left = self.derivative(left, variable);
                let right = self.derivative(right, variable);
                self.add(left, right)
            }
            Node::Sub(left, right) => {
                let left = self.derivative(left, variable);
                let right = self.derivative(right, variable);
                self.sub(left, right)
            }
            Node::Mul(left, right) => {
                let left_derivative = self.derivative(left, variable);
                let right_derivative = self.derivative(right, variable);
                let first = self.mul(left, right_derivative);
                let second = self.mul(left_derivative, right);
                self.add(first, second)
            }
            Node::Div(numerator, denominator) => {
                let numerator_derivative = self.derivative(numerator, variable);
                let denominator_derivative = self.derivative(denominator, variable);
                let first = self.mul(numerator_derivative, denominator);
                let second = self.mul(numerator, denominator_derivative);
                let top = self.sub(first, second);
                let bottom = self.mul(denominator, denominator);
                self.div(top, bottom)
            }
            Node::Neg(value) => {
                let derivative = self.derivative(value, variable);
                self.neg(derivative)
            }
            Node::Exp(value) => {
                let exp = self.intern(Node::Exp(value));
                let derivative = self.derivative(value, variable);
                self.mul(exp, derivative)
            }
            Node::Ln(value) => {
                let derivative = self.derivative(value, variable);
                let reciprocal = self.recip(value);
                self.mul(derivative, reciprocal)
            }
            Node::Sqrt(value) => {
                let derivative = self.derivative(value, variable);
                let two = self.constant(2.0);
                let sqrt = self.intern(Node::Sqrt(value));
                let denominator = self.mul(two, sqrt);
                self.div(derivative, denominator)
            }
            Node::Recip(value) => {
                let derivative = self.derivative(value, variable);
                let reciprocal = self.intern(Node::Recip(value));
                let reciprocal_squared = self.mul(reciprocal, reciprocal);
                let product = self.mul(derivative, reciprocal_squared);
                self.neg(product)
            }
            Node::Select(activity, when_true, when_false) => {
                let when_true = self.derivative(when_true, variable);
                let when_false = self.derivative(when_false, variable);
                self.select(activity, when_true, when_false)
            }
        };
        self.derivatives.insert((id, variable), derivative);
        derivative
    }
}

enum Binding {
    Primary(usize),
    Constant(usize),
}

fn binding(path: &ExprPath, primaries: &[Ident], constants: &[Ident]) -> Result<Binding> {
    let ident = path
        .path
        .get_ident()
        .ok_or_else(|| syn::Error::new_spanned(path, "row_atom variables must be identifiers"))?;
    if let Some(axis) = primaries.iter().position(|candidate| candidate == ident) {
        return Ok(Binding::Primary(axis));
    }
    constants
        .iter()
        .position(|candidate| candidate == ident)
        .map(Binding::Constant)
        .ok_or_else(|| syn::Error::new_spanned(path, format!("unknown row_atom binding `{ident}`")))
}

fn literal_value(literal: &ExprLit) -> Result<f64> {
    match &literal.lit {
        Lit::Float(value) => value.base10_parse(),
        Lit::Int(value) => value.base10_parse(),
        _ => Err(syn::Error::new_spanned(
            literal,
            "row_atom supports only numeric literals",
        )),
    }
}

fn graph_expression(
    expression: &Expr,
    primaries: &[Ident],
    constants: &[Ident],
    graph: &mut Graph,
) -> Result<usize> {
    match expression {
        Expr::Path(path) => Ok(match binding(path, primaries, constants)? {
            Binding::Primary(axis) => graph.intern(Node::Variable(axis)),
            Binding::Constant(index) => graph.intern(Node::Parameter(index)),
        }),
        Expr::Lit(literal) => Ok(graph.constant(literal_value(literal)?)),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            graph_expression(expr, primaries, constants, graph)
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            let value = graph_expression(expr, primaries, constants, graph)?;
            Ok(graph.neg(value))
        }
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let left = graph_expression(left, primaries, constants, graph)?;
            let right = graph_expression(right, primaries, constants, graph)?;
            let node = match op {
                BinOp::Add(_) => graph.add(left, right),
                BinOp::Sub(_) => graph.sub(left, right),
                BinOp::Mul(_) => graph.mul(left, right),
                BinOp::Div(_) => graph.div(left, right),
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "row_atom supports +, -, *, and /",
                    ));
                }
            };
            Ok(node)
        }
        Expr::Call(call) => {
            if call.args.len() != 1 {
                return Err(syn::Error::new_spanned(
                    call,
                    "row_atom unary functions take one argument",
                ));
            }
            let argument = graph_expression(&call.args[0], primaries, constants, graph)?;
            let node = match row_program::bare_call_name(call, "row_atom unary calls")?
                .to_string()
                .as_str()
            {
                "exp" => graph.exp(argument),
                "ln" => graph.ln(argument),
                "sqrt" => graph.sqrt(argument),
                "recip" => graph.recip(argument),
                name => {
                    return Err(syn::Error::new_spanned(
                        call,
                        format!("unsupported row_atom unary function `{name}`"),
                    ));
                }
            };
            Ok(node)
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_atom expression",
        )),
    }
}

fn jet_expression(
    expression: &Expr,
    primaries: &[Ident],
    constants: &[Ident],
) -> Result<TokenStream2> {
    match expression {
        Expr::Path(path) => match binding(path, primaries, constants)? {
            Binding::Primary(axis) => {
                let variable = &primaries[axis];
                Ok(quote!(*#variable))
            }
            Binding::Constant(index) => {
                let constant = &constants[index];
                Ok(quote!(S::constant(#constant)))
            }
        },
        Expr::Lit(literal) => Ok(quote!(S::constant((#literal) as f64))),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            jet_expression(expr, primaries, constants)
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            let value = jet_expression(expr, primaries, constants)?;
            Ok(quote!({ let value = #value; value.neg() }))
        }
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let left = jet_expression(left, primaries, constants)?;
            let right = jet_expression(right, primaries, constants)?;
            match op {
                BinOp::Add(_) => {
                    Ok(quote!({ let left = #left; let right = #right; left.add(&right) }))
                }
                BinOp::Sub(_) => {
                    Ok(quote!({ let left = #left; let right = #right; left.sub(&right) }))
                }
                BinOp::Mul(_) => {
                    Ok(quote!({ let left = #left; let right = #right; left.mul(&right) }))
                }
                BinOp::Div(_) => Ok(quote!({
                    let left = #left;
                    let right = #right;
                    left.mul(&right.recip())
                })),
                _ => Err(syn::Error::new_spanned(
                    op,
                    "row_atom supports +, -, *, and /",
                )),
            }
        }
        Expr::Call(call) => {
            if call.args.len() != 1 {
                return Err(syn::Error::new_spanned(
                    call,
                    "row_atom unary functions take one argument",
                ));
            }
            let argument = jet_expression(&call.args[0], primaries, constants)?;
            let method = row_program::bare_call_name(call, "row_atom unary calls")?;
            match method.to_string().as_str() {
                "exp" | "ln" | "sqrt" | "recip" => Ok(quote!({
                    let value = #argument;
                    value.#method()
                })),
                name => Err(syn::Error::new_spanned(
                    call,
                    format!("unsupported row_atom unary function `{name}`"),
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_atom expression",
        )),
    }
}

fn node_reference(
    id: usize,
    graph: &Graph,
    primaries: &[Ident],
    constants: &[Ident],
) -> TokenStream2 {
    match graph.nodes[id] {
        Node::Constant(bits) => {
            let literal = Literal::f64_unsuffixed(f64::from_bits(bits));
            quote!(#literal)
        }
        Node::Variable(axis) => {
            let variable = &primaries[axis];
            quote!(#variable)
        }
        Node::Parameter(index) => {
            let constant = &constants[index];
            quote!(#constant)
        }
        _ => {
            let temporary = format_ident!("__row_atom_{id}");
            quote!(#temporary)
        }
    }
}

/// Rust expression defining one plain (non-leaf, non-`Select`) node from the
/// references of its children.
fn node_expression(
    id: usize,
    graph: &Graph,
    primaries: &[Ident],
    constants: &[Ident],
) -> Result<TokenStream2> {
    let reference = |child| node_reference(child, graph, primaries, constants);
    match graph.nodes[id] {
        Node::Add(left, right) => {
            let (left, right) = (reference(left), reference(right));
            Ok(quote!(#left + #right))
        }
        Node::Sub(left, right) => {
            let (left, right) = (reference(left), reference(right));
            Ok(quote!(#left - #right))
        }
        Node::Mul(left, right) => {
            let (left, right) = (reference(left), reference(right));
            Ok(quote!(#left * #right))
        }
        Node::Div(left, right) => {
            let (left, right) = (reference(left), reference(right));
            Ok(quote!(#left / #right))
        }
        Node::Neg(value) => {
            let value = reference(value);
            Ok(quote!(-#value))
        }
        Node::Exp(value) => {
            let value = reference(value);
            Ok(quote!(#value.exp()))
        }
        Node::Ln(value) => {
            let value = reference(value);
            Ok(quote!(#value.ln()))
        }
        Node::Sqrt(value) => {
            let value = reference(value);
            Ok(quote!(#value.sqrt()))
        }
        Node::Recip(value) => {
            let value = reference(value);
            Ok(quote!(#value.recip()))
        }
        Node::Select(..) => Err(syn::Error::new(
            Span::call_site(),
            "row_atom internal schedule error: a Select is defined by its activity group, not as \
             a plain node",
        )),
        Node::Constant(_) | Node::Variable(_) | Node::Parameter(_) => Err(syn::Error::new(
            Span::call_site(),
            "row_atom internal schedule error: a leaf node has no temporary definition",
        )),
    }
}

fn is_leaf(graph: &Graph, id: usize) -> bool {
    matches!(
        graph.nodes[id],
        Node::Constant(_) | Node::Variable(_) | Node::Parameter(_)
    )
}

fn children(graph: &Graph, id: usize) -> Vec<usize> {
    match graph.nodes[id] {
        Node::Constant(_) | Node::Variable(_) | Node::Parameter(_) => Vec::new(),
        Node::Neg(value)
        | Node::Exp(value)
        | Node::Ln(value)
        | Node::Sqrt(value)
        | Node::Recip(value) => vec![value],
        Node::Add(left, right)
        | Node::Sub(left, right)
        | Node::Mul(left, right)
        | Node::Div(left, right) => vec![left, right],
        Node::Select(_, when_true, when_false) => vec![when_true, when_false],
    }
}

/// One lexical level of the generated schedule: the function body, or the
/// inside of one `if activity { ... }` branch.
///
/// Every node `roots` need is defined exactly once at the innermost level that
/// may evaluate it. A `Select` is an activity gate — its branch may only be
/// evaluated when the flag is set, because an inactive row's guarded term may
/// be non-finite and `0 * NaN` is `NaN` — so its branch work is defined inside
/// the branch. Every `Select` on ONE activity that is ready at the same point
/// is lowered into ONE `if` block binding every selected value at once, so
/// work shared by several channels of that activity is done once.
///
/// Before this, each `Select` scheduled its branch from a fresh visited set
/// inside its own `if`, so a subexpression shared by two channels of one
/// activity was defined once per channel. Measured on the cause-specific
/// Royston–Parmar row (release profile, EPYC Milan): the event branch carried
/// two `divsd` — the spline-derivative reciprocal once for the gradient and
/// once, negated, for the Hessian — against the hand kernel's one, and the
/// generated order-2 kernel lost to the hand kernel unanimously over fifteen
/// paired repetitions (`median_ratio=0.9787`, `resolution=0.0031`) while the
/// third and fourth channels, which share nothing across a gate, won.
struct Scheduler<'a> {
    graph: &'a Graph,
    primaries: &'a [Ident],
    constants: &'a [Ident],
    /// Nodes already bound to a name at this level or an enclosing one.
    defined: HashSet<usize>,
    /// Nodes reachable from this level's roots without crossing a `Select`
    /// branch: evaluated unconditionally here, so a branch that also needs one
    /// of them references it instead of recomputing it inside the gate.
    unconditional: HashSet<usize>,
    /// The `Select`s this level lowers, in first-visit order.
    selects: Vec<usize>,
    /// Activities whose group is currently being assembled.
    groups_in_progress: HashSet<usize>,
    /// Plain nodes currently being visited; a re-entry would be a cycle.
    nodes_in_progress: HashSet<usize>,
    out: Vec<TokenStream2>,
}

impl<'a> Scheduler<'a> {
    fn new(
        graph: &'a Graph,
        primaries: &'a [Ident],
        constants: &'a [Ident],
        defined: HashSet<usize>,
    ) -> Self {
        Self {
            graph,
            primaries,
            constants,
            defined,
            unconditional: HashSet::new(),
            selects: Vec::new(),
            groups_in_progress: HashSet::new(),
            nodes_in_progress: HashSet::new(),
            out: Vec::new(),
        }
    }

    fn run(mut self, roots: &[usize]) -> Result<Vec<TokenStream2>> {
        for &root in roots {
            self.survey(root);
        }
        // A divide independent of a transcendental call can overlap that
        // call's instruction stream. In particular, schedule the derivative
        // reciprocal before ln(value), rather than serializing it afterward.
        // Survey stops at activity gates, so this never hoists guarded work
        // out of the lexical branch in which it is valid.
        fn call_free(graph: &Graph, id: usize, memo: &mut HashMap<usize, bool>) -> bool {
            if let Some(&result) = memo.get(&id) {
                return result;
            }
            let result = !matches!(
                graph.nodes[id],
                Node::Exp(_) | Node::Ln(_) | Node::Select(..)
            ) && children(graph, id)
                .into_iter()
                .all(|child| call_free(graph, child, memo));
            memo.insert(id, result);
            result
        }
        let mut divisions = self
            .unconditional
            .iter()
            .copied()
            .filter(|&id| matches!(self.graph.nodes[id], Node::Div(..) | Node::Recip(_)))
            .collect::<Vec<_>>();
        divisions.sort_unstable();
        let mut memo = HashMap::new();
        for id in divisions {
            if call_free(self.graph, id, &mut memo) {
                self.visit(id)?;
            }
        }
        for &root in roots {
            self.visit(root)?;
        }
        Ok(self.out)
    }

    fn gate(&self, select: usize) -> Result<(usize, usize, usize)> {
        match self.graph.nodes[select] {
            Node::Select(activity, when_true, when_false) => Ok((activity, when_true, when_false)),
            _ => Err(syn::Error::new(
                Span::call_site(),
                "row_atom internal schedule error: only a Select has an activity gate",
            )),
        }
    }

    /// Record what this level evaluates unconditionally, stopping at every
    /// `Select` (whose branches belong to the level inside the gate).
    fn survey(&mut self, id: usize) {
        if self.defined.contains(&id) || !self.unconditional.insert(id) {
            return;
        }
        if matches!(self.graph.nodes[id], Node::Select(..)) {
            self.selects.push(id);
            return;
        }
        for child in children(self.graph, id) {
            self.survey(child);
        }
    }

    fn visit(&mut self, id: usize) -> Result<()> {
        if self.defined.contains(&id) || is_leaf(self.graph, id) {
            return Ok(());
        }
        if matches!(self.graph.nodes[id], Node::Select(..)) {
            let (activity, _, _) = self.gate(id)?;
            return self.emit_group(activity, id);
        }
        if !self.nodes_in_progress.insert(id) {
            return Err(syn::Error::new(
                Span::call_site(),
                "row_atom internal schedule error: the expression graph has a cycle",
            ));
        }
        for child in children(self.graph, id) {
            self.visit(child)?;
        }
        self.nodes_in_progress.remove(&id);
        // A child's activity group may already have defined this node while
        // resolving a dependency of a sibling gate.
        if self.defined.contains(&id) {
            return Ok(());
        }
        let temporary = format_ident!("__row_atom_{id}");
        let expression = node_expression(id, self.graph, self.primaries, self.constants)?;
        self.out.push(quote!(let #temporary: f64 = #expression;));
        self.defined.insert(id);
        Ok(())
    }

    /// Define, at this level, every unconditional node a gate's branch reads,
    /// so the branch references it rather than recomputing it inside the gate.
    fn visit_outside_dependencies(&mut self, id: usize, seen: &mut HashSet<usize>) -> Result<()> {
        if !seen.insert(id) || is_leaf(self.graph, id) {
            return Ok(());
        }
        if self.defined.contains(&id) || self.unconditional.contains(&id) {
            return self.visit(id);
        }
        for child in children(self.graph, id) {
            self.visit_outside_dependencies(child, seen)?;
        }
        Ok(())
    }

    /// Lower every pending `Select` on `activity` as one `if` block. `first`
    /// is the gate whose visit requested the group; if the group is already
    /// being assembled further up the stack, that gate is on a dependency path
    /// of its own group and is lowered alone.
    fn emit_group(&mut self, activity: usize, first: usize) -> Result<()> {
        if !self.groups_in_progress.insert(activity) {
            return self.emit_selects(activity, vec![first]);
        }
        let mut members = Vec::new();
        for &select in &self.selects {
            if !self.defined.contains(&select) && self.gate(select)?.0 == activity {
                members.push(select);
            }
        }
        if !members.contains(&first) {
            members.push(first);
        }
        let mut seen = HashSet::new();
        for &select in &members {
            let (_, when_true, when_false) = self.gate(select)?;
            self.visit_outside_dependencies(when_true, &mut seen)?;
            self.visit_outside_dependencies(when_false, &mut seen)?;
        }
        // A member lowered alone while its dependencies were resolved is
        // already bound.
        members.retain(|select| !self.defined.contains(select));
        let result = self.emit_selects(activity, members);
        self.groups_in_progress.remove(&activity);
        result
    }

    fn emit_selects(&mut self, activity: usize, members: Vec<usize>) -> Result<()> {
        if members.is_empty() {
            return Ok(());
        }
        let mut true_roots = Vec::with_capacity(members.len());
        let mut false_roots = Vec::with_capacity(members.len());
        for &select in &members {
            let (_, when_true, when_false) = self.gate(select)?;
            true_roots.push(when_true);
            false_roots.push(when_false);
        }
        let true_definitions = Scheduler::new(
            self.graph,
            self.primaries,
            self.constants,
            self.defined.clone(),
        )
        .run(&true_roots)?;
        let false_definitions = Scheduler::new(
            self.graph,
            self.primaries,
            self.constants,
            self.defined.clone(),
        )
        .run(&false_roots)?;
        let reference = |id| node_reference(id, self.graph, self.primaries, self.constants);
        let true_values: Vec<TokenStream2> = true_roots.iter().map(|&id| reference(id)).collect();
        let false_values: Vec<TokenStream2> = false_roots.iter().map(|&id| reference(id)).collect();
        let names: Vec<Ident> = members
            .iter()
            .map(|select| format_ident!("__row_atom_{select}"))
            .collect();
        let flag = &self.constants[activity];
        let binding = if members.len() == 1 {
            let name = &names[0];
            let true_value = &true_values[0];
            let false_value = &false_values[0];
            quote! {
                let #name: f64 = if #flag {
                    #(#true_definitions)*
                    #true_value
                } else {
                    #(#false_definitions)*
                    #false_value
                };
            }
        } else {
            let types = std::iter::repeat_n(quote!(f64), members.len());
            quote! {
                let (#(#names),*): (#(#types),*) = if #flag {
                    #(#true_definitions)*
                    (#(#true_values),*)
                } else {
                    #(#false_definitions)*
                    (#(#false_values),*)
                };
            }
        };
        self.out.push(binding);
        self.defined.extend(members);
        Ok(())
    }
}

fn schedule_definitions(
    roots: impl IntoIterator<Item = usize>,
    graph: &Graph,
    primaries: &[Ident],
    constants: &[Ident],
) -> Result<Vec<TokenStream2>> {
    let roots = roots.into_iter().collect::<Vec<_>>();
    Scheduler::new(graph, primaries, constants, HashSet::new()).run(&roots)
}

fn constant_parameters(
    constants: &[Ident],
    activity_constants: &HashSet<usize>,
) -> Vec<TokenStream2> {
    constants
        .iter()
        .enumerate()
        .map(|(index, constant)| {
            if activity_constants.contains(&index) {
                quote!(#constant: bool)
            } else {
                quote!(#constant: f64)
            }
        })
        .collect()
}

fn expand(input: RowAtomInput) -> Result<TokenStream2> {
    let RowAtomInput {
        visibility,
        name,
        lowerings,
        primaries,
        constants,
        activity_constants,
        scale_constants,
        expression,
    } = input;
    let mut graph = Graph::new();
    let mut value = graph_expression(&expression, &primaries, &constants, &mut graph)?;
    if !activity_constants.is_empty() {
        value = graph.guard_activities(value, &activity_constants, &mut HashMap::new());
    }
    let differentiated_value = if scale_constants.is_empty() {
        value
    } else {
        graph.distribute_scales(
            value,
            &scale_constants,
            ScaleDistribution::Derivative,
            &mut HashMap::new(),
        )
    };
    value = if scale_constants.is_empty() {
        value
    } else {
        graph.distribute_scales(
            value,
            &scale_constants,
            ScaleDistribution::Value,
            &mut HashMap::new(),
        )
    };
    let dimension = primaries.len();
    let mut gradient = Vec::with_capacity(dimension);
    for axis in 0..dimension {
        gradient.push(graph.derivative(differentiated_value, axis));
    }
    let mut hessian = vec![vec![0usize; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            hessian[row][column] = graph.derivative(gradient[row], column);
        }
    }
    let mut output = Vec::new();
    let constant_parameters = constant_parameters(&constants, &activity_constants);
    let generic_activity_bindings = constants
        .iter()
        .enumerate()
        .filter(|(index, _)| activity_constants.contains(index))
        .map(|(_, activity)| quote!(let #activity: f64 = f64::from(#activity);))
        .collect::<Vec<_>>();

    if lowerings.contains(&Lowering::Generic) {
        let generic_expression = jet_expression(&expression, &primaries, &constants)?;
        output.push(quote! {
            #[inline(always)]
            #visibility fn #name<const K: usize, S: ::gam_math::jet_scalar::JetScalar<K>>(
                #(#primaries: &S,)*
                #(#constant_parameters),*
            ) -> S {
                #(#generic_activity_bindings)*
                #generic_expression
            }
        });
    }

    for (lowering, suffix, at_zero) in [
        (Lowering::Order2, "order2", false),
        (Lowering::Order2AtZero, "order2_at_zero", true),
    ] {
        if !lowerings.contains(&lowering) {
            continue;
        }
        let order2_name = format_ident!("{name}_{suffix}");
        let (value, gradient, hessian) = if at_zero {
            let mut memo = HashMap::new();
            let mut channels = vec![graph.substitute_zero_primaries(value, &mut memo)];
            channels.extend(
                gradient
                    .iter()
                    .map(|&id| graph.substitute_zero_primaries(id, &mut memo)),
            );
            channels.extend(
                hessian
                    .iter()
                    .flatten()
                    .map(|&id| graph.substitute_zero_primaries(id, &mut memo)),
            );
            let normalized = graph.normalize_polynomials(&channels, constants.len());
            let value = normalized[0];
            let gradient = normalized[1..1 + dimension].to_vec();
            let hessian = normalized[1 + dimension..]
                .chunks(dimension)
                .map(<[usize]>::to_vec)
                .collect::<Vec<_>>();
            (value, gradient, hessian)
        } else {
            // The Hessian must expose the same weighted products as the
            // gradient. For example, w*r*r shares w*r with d[-w ln(x)]/dx;
            // retaining w*(r*r) computes an extra multiply for the pair.
            let hessian = hessian
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&id| {
                            // Keep an outer sign outside its gate, so value, gradient
                            // and Hessian can still share one selected contribution.
                            if let Node::Neg(value) = graph.nodes[id] {
                                let value = graph.normalize_ring(value);
                                graph.neg(value)
                            } else {
                                graph.normalize_ring(id)
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (value, gradient.clone(), hessian)
        };
        let mut packed_hessian = Vec::with_capacity(dimension * (dimension + 1) / 2);
        for (row, channels) in hessian.iter().enumerate() {
            packed_hessian.extend_from_slice(&channels[row..]);
        }
        let packed = dimension * (dimension + 1) / 2;
        let gradient_bits = gradient
            .iter()
            .enumerate()
            .fold(0u128, |bits, (axis, &id)| {
                bits | (u128::from(!graph.is_zero(id)) << axis)
            });
        let hessian_bits = packed_hessian
            .iter()
            .enumerate()
            .fold(0u128, |bits, (slot, &id)| {
                bits | (u128::from(!graph.is_zero(id)) << slot)
            });
        let gradient_bits = Literal::u128_unsuffixed(gradient_bits);
        let hessian_bits = Literal::u128_unsuffixed(hessian_bits);
        let primary_parameters = if at_zero {
            quote!()
        } else {
            quote!(#(#primaries: f64,)*)
        };
        let definitions = schedule_definitions(
            std::iter::once(value)
                .chain(gradient.iter().copied())
                .chain(packed_hessian.iter().copied()),
            &graph,
            &primaries,
            &constants,
        )?;
        let value_ref = node_reference(value, &graph, &primaries, &constants);
        let gradient_refs = gradient
            .iter()
            .map(|&id| node_reference(id, &graph, &primaries, &constants));
        let hessian_refs = packed_hessian
            .iter()
            .map(|&id| node_reference(id, &graph, &primaries, &constants));
        let body = quote! {
            #(#definitions)*
            ::gam_math::jet_scalar::StaticOrder2Atom::new(
                #value_ref,
                [#(#gradient_refs),*],
                [#(#hessian_refs),*],
            )
        };
        output.push(quote! {
            #[inline(always)]
            #visibility fn #order2_name(
                #primary_parameters
                #(#constant_parameters),*
            ) -> ::gam_math::jet_scalar::StaticOrder2Atom<
                #dimension,
                #packed,
                #gradient_bits,
                #hessian_bits,
            > {
                #body
            }
        });
    }

    for (lowering, suffix, at_zero) in [
        (Lowering::Third, "third_contracted", false),
        (Lowering::ThirdAtZero, "third_contracted_at_zero", true),
    ] {
        if !lowerings.contains(&lowering) {
            continue;
        }
        let third_name = format_ident!("{name}_{suffix}");
        let mut channels = vec![vec![Vec::new(); dimension]; dimension];
        let mut memo = HashMap::new();
        for row in 0..dimension {
            for column in row..dimension {
                let mut derivatives = (0..dimension)
                    .map(|axis| {
                        let derivative = graph.derivative(hessian[row][column], axis);
                        graph.normalize_ring(derivative)
                    })
                    .collect::<Vec<_>>();
                if at_zero {
                    for derivative in &mut derivatives {
                        *derivative = graph.substitute_zero_primaries(*derivative, &mut memo);
                    }
                }
                channels[row][column] = derivatives;
            }
        }
        if at_zero {
            let flat = channels
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let normalized = graph.normalize_polynomials(&flat, constants.len());
            let mut next = normalized.into_iter();
            for derivative in channels.iter_mut().flatten().flatten() {
                *derivative = next.next().expect("one normalized channel per derivative");
            }
        }
        let mut roots = Vec::new();
        let mut assignments = Vec::new();
        let mut entries = vec![vec![quote!(0.0); dimension]; dimension];
        for (row, columns) in channels.iter().enumerate() {
            for (column, derivatives) in columns.iter().enumerate().skip(row) {
                roots.extend(derivatives.iter().copied());
                let terms = derivatives
                    .iter()
                    .enumerate()
                    .filter(|(_, id)| !graph.is_zero(**id))
                    .map(|(axis, &id)| {
                        let derivative = node_reference(id, &graph, &primaries, &constants);
                        quote!(#derivative * direction[#axis])
                    })
                    .collect::<Vec<_>>();
                let sum = match terms.split_first() {
                    None => continue,
                    Some((first, rest)) => quote!(#first #(+ #rest)*),
                };
                let temporary = format_ident!("__row_atom_third_{row}_{column}");
                assignments.push(quote!(let #temporary = #sum;));
                entries[row][column] = quote!(#temporary);
                entries[column][row] = quote!(#temporary);
            }
        }
        let definitions = schedule_definitions(roots, &graph, &primaries, &constants)?;
        let primary_parameters = if at_zero {
            quote!()
        } else {
            quote!(#(#primaries: f64,)*)
        };
        // The result is a literal, not a zero-filled array overwritten cell
        // by cell: the literal's stores are disjoint, where a zero fill
        // followed by the diagonal's stores overlaps two of them in flight.
        // On the cause-specific fourth channel that overlap was the only
        // difference from the hand kernel's assembly (#932).
        let rows = entries.iter().map(|row| quote!([#(#row),*]));
        let body = quote! {
            #(#definitions)*
            #(#assignments)*
            [#(#rows),*]
        };
        output.push(quote! {
            #[inline(always)]
            #visibility fn #third_name(
                #primary_parameters
                #(#constant_parameters,)*
                direction: &[f64; #dimension],
            ) -> [[f64; #dimension]; #dimension] {
                #body
            }
        });
    }

    for (lowering, suffix, at_zero) in [
        (Lowering::Fourth, "fourth_contracted", false),
        (Lowering::FourthAtZero, "fourth_contracted_at_zero", true),
    ] {
        if !lowerings.contains(&lowering) {
            continue;
        }
        let fourth_name = format_ident!("{name}_{suffix}");
        let mut channels = vec![vec![Vec::<Vec<usize>>::new(); dimension]; dimension];
        let mut memo = HashMap::new();
        for row in 0..dimension {
            for column in row..dimension {
                let third = (0..dimension)
                    .map(|axis| {
                        let derivative = graph.derivative(hessian[row][column], axis);
                        graph.normalize_ring(derivative)
                    })
                    .collect::<Vec<_>>();
                let mut fourth = third
                    .iter()
                    .map(|&id| {
                        (0..dimension)
                            .map(|axis| {
                                let derivative = graph.derivative(id, axis);
                                graph.normalize_ring(derivative)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                if at_zero {
                    for derivative in fourth.iter_mut().flatten() {
                        *derivative = graph.substitute_zero_primaries(*derivative, &mut memo);
                    }
                }
                channels[row][column] = fourth;
            }
        }
        if at_zero {
            let flat = channels
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let normalized = graph.normalize_polynomials(&flat, constants.len());
            let mut next = normalized.into_iter();
            for derivative in channels.iter_mut().flatten().flatten().flatten() {
                *derivative = next.next().expect("one normalized channel per derivative");
            }
        }
        let mut roots = Vec::new();
        let mut assignments = Vec::new();
        let mut entries = vec![vec![quote!(0.0); dimension]; dimension];
        for (row, columns) in channels.iter().enumerate() {
            for (column, derivatives) in columns.iter().enumerate().skip(row) {
                roots.extend(derivatives.iter().flatten().copied());
                let terms = derivatives
                    .iter()
                    .enumerate()
                    .flat_map(|(axis_u, derivatives)| {
                        derivatives
                            .iter()
                            .enumerate()
                            .filter(|(_, id)| !graph.is_zero(**id))
                            .map(|(axis_v, &id)| {
                                let derivative = node_reference(id, &graph, &primaries, &constants);
                                quote!(#derivative * direction_u[#axis_u] * direction_v[#axis_v])
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let sum = match terms.split_first() {
                    None => continue,
                    Some((first, rest)) => quote!(#first #(+ #rest)*),
                };
                let temporary = format_ident!("__row_atom_fourth_{row}_{column}");
                assignments.push(quote!(let #temporary = #sum;));
                entries[row][column] = quote!(#temporary);
                entries[column][row] = quote!(#temporary);
            }
        }
        let definitions = schedule_definitions(roots, &graph, &primaries, &constants)?;
        let primary_parameters = if at_zero {
            quote!()
        } else {
            quote!(#(#primaries: f64,)*)
        };
        // The result is a literal, not a zero-filled array overwritten cell
        // by cell: the literal's stores are disjoint, where a zero fill
        // followed by the diagonal's stores overlaps two of them in flight.
        // On the cause-specific fourth channel that overlap was the only
        // difference from the hand kernel's assembly (#932).
        let rows = entries.iter().map(|row| quote!([#(#row),*]));
        let body = quote! {
            #(#definitions)*
            #(#assignments)*
            [#(#rows),*]
        };
        output.push(quote! {
            #[inline(always)]
            #visibility fn #fourth_name(
                #primary_parameters
                #(#constant_parameters,)*
                direction_u: &[f64; #dimension],
                direction_v: &[f64; #dimension],
            ) -> [[f64; #dimension]; #dimension] {
                #body
            }
        });
    }

    Ok(quote!(#(#output)*))
}

/// Define one row atom and emit exactly its requested build-time lowerings.
///
/// ```text
/// row_atom! {
///     pub(crate) fn row [generic, order2, third, fourth](
///         eta, deriv;
///         weight: scale, event: bool
///     ) {
///         weight * (exp(eta) - event * (eta + ln(deriv)))
///     }
/// }
/// ```
///
/// A normalized local-coordinate atom whose production expansion point is
/// identically zero can instead request the `_at_zero` lowerings. This is
/// exact partial evaluation, not a separate derivative expression.
#[proc_macro]
pub fn row_atom(input: TokenStream) -> TokenStream {
    match expand(parse_macro_input!(input as RowAtomInput)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

/// Define one backend-neutral row program and emit its generic `JetScalar`
/// evaluator plus symbolically sparse order-2 Rust and CUDA functions. The
/// declaration owns the complete algebraic schedule; stable unary primitives
/// are explicit leaves mapped to one Rust derivative-stack builder and one CUDA
/// stack function, or declared `supplied` when a kernel builder has already
/// evaluated the stack at the composition point (the compose's five scalar
/// entries are the stack; nothing is called and the point is not inspected).
/// Both direct backends consume the same symbolic SSA lowering,
/// compute each nonzero gradient and packed Hessian component once, and scatter
/// Hessian symmetry only at the output seam.
///
/// A constant may be declared with the role `name: sign`, a value in
/// `{-1, +1}`: its square is one, so a composition on `scale(x, s)` forms
/// `s·f'` once and reads `f''` as it is, as a hand kernel that knows the sign
/// writes it. The generic and directional surfaces need no such knowledge
/// (`s·s` is exactly one in floating point), so every surface still agrees.
#[proc_macro]
pub fn row_program(input: TokenStream) -> TokenStream {
    match row_program::expand(parse_macro_input!(input as row_program::Input)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

#[cfg(test)]
mod row_atom_tests {
    use super::RowAtomInput;

    #[test]
    fn activity_parameters_are_not_arithmetic_graph_nodes() {
        let mut graph = super::Graph::new();
        let x = graph.intern(super::Node::Variable(0));
        let square = graph.mul(x, x);
        let zero = graph.constant(0.0);
        // Parameter numbering is independent of graph-node numbering.
        let selected = graph.select(100, square, zero);
        assert_eq!(graph.polynomial_schedule_cost(&[selected]), (1, 0, 2, 2));
    }

    /// The activity-gate schedule the cause-specific row was measured into
    /// (#932): every gate on one activity is ONE `if` block per lowering, the
    /// work shared by that activity's channels (the reciprocal of the spline
    /// derivative that feeds both the gradient and the Hessian of
    /// `ln(derivative)`) is defined once, the unconditional exponentials are
    /// defined once, and no inactive zero is negative. Each of these was a
    /// measured deficit against the hand kernel before it was fixed, and each
    /// is a property of the emitted TEXT, so this pins the text.
    #[test]
    fn activity_gates_are_grouped_and_define_shared_work_once() {
        let input = syn::parse_str::<RowAtomInput>(
            "fn atom [order2, third](
                eta_exit, eta_entry, derivative;
                weight: scale, entry_active: bool, event: bool
            ) {
                weight
                    * (exp(eta_exit)
                        - entry_active * exp(eta_entry)
                        - event * (eta_exit + ln(derivative)))
            }",
        )
        .expect("cause-specific row atom");
        let expanded = super::expand(input)
            .expect("expand row atom")
            .to_string()
            .replace(' ', "");
        // Two lowerings (order2, third): one gate block per activity in each.
        assert_eq!(expanded.matches("ifevent{").count(), 2, "{expanded}");
        assert_eq!(expanded.matches("ifentry_active{").count(), 2, "{expanded}");
        // The reciprocal is defined once per lowering, never once per channel.
        assert_eq!(expanded.matches(".recip()").count(), 2, "{expanded}");
        // Both exponentials once per lowering.
        assert_eq!(expanded.matches(".exp()").count(), 4, "{expanded}");
        // An inactive contribution is `0.0`, never `-0.0`.
        assert!(!expanded.contains("-0.0"), "{expanded}");
    }

    /// The at-zero lowering of the Gaussian joint row (#932) shares products
    /// across the entire output, so the
    /// residual's square is one shared node and the weight's products are
    /// shared the way the hand kernel names them. The strongest hand
    /// schedule of this row uses 16 multiplies; the lowering may not use
    /// more.
    #[test]
    fn at_zero_lowering_shares_powers_and_does_not_exceed_the_hand_multiply_count() {
        let input = syn::parse_str::<RowAtomInput>(
            "fn generated_gaussian [order2_at_zero](
                delta_mu, delta_eta;
                obs_weight: f64, standardized_residual: f64, inv_sigma: f64, kappa: f64
            ) {
                obs_weight * ln((1.0 - kappa) + kappa * exp(delta_eta))
                    + 0.5 * obs_weight * (standardized_residual - delta_mu * inv_sigma)
                        * (standardized_residual - delta_mu * inv_sigma)
                        / ((1.0 - kappa) + kappa * exp(delta_eta))
                        / ((1.0 - kappa) + kappa * exp(delta_eta))
            }",
        )
        .expect("gaussian joint row atom");
        let expanded = super::expand(input).expect("expand row atom").to_string();
        let start = expanded
            .find("fn generated_gaussian_order2_at_zero")
            .expect("the order-2 at-zero lowering");
        let body = &expanded[start..];
        let squares = body
            .matches("standardized_residual * standardized_residual")
            .count();
        assert_eq!(squares, 1, "{body}");
        let multiplies = body.matches(" * ").count();
        assert!(multiplies <= 16, "{multiplies} multiplies:\n{body}");
        assert!(
            !body.contains("* - 1.0"),
            "a coefficient of -1 is a sign:\n{body}"
        );
    }

    /// A contracted third or fourth derivative is returned as a literal with
    /// the absent cells written `0.0`, never as a zero-filled array
    /// overwritten cell by cell (whose diagonal stores overlap the fill's in
    /// flight). The cause-specific atom is separable, so both matrices are
    /// diagonal.
    #[test]
    fn contracted_matrices_are_emitted_as_literals() {
        let input = syn::parse_str::<RowAtomInput>(
            "fn cause_specific [third, fourth](
                eta_exit, eta_entry, derivative;
                weight: scale, entry_active: bool, event: bool
            ) {
                weight
                    * (exp(eta_exit)
                        - entry_active * exp(eta_entry)
                        - event * (eta_exit + ln(derivative)))
            }",
        )
        .expect("cause-specific row atom");
        let expanded = super::expand(input).expect("expand row atom").to_string();
        for (surface, literal) in [
            (
                "fn cause_specific_third_contracted",
                "[[__row_atom_third_0_0 , 0.0 , 0.0] , [0.0 , __row_atom_third_1_1 , 0.0] , [0.0 , 0.0 , __row_atom_third_2_2]]",
            ),
            (
                "fn cause_specific_fourth_contracted",
                "[[__row_atom_fourth_0_0 , 0.0 , 0.0] , [0.0 , __row_atom_fourth_1_1 , 0.0] , [0.0 , 0.0 , __row_atom_fourth_2_2]]",
            ),
        ] {
            let start = expanded.find(surface).expect(surface);
            let body = &expanded[start..];
            let body = &body[..body.find("# [inline").unwrap_or(body.len())];
            assert!(!body.contains("let mut out"), "{body}");
            assert!(body.contains(literal), "{body}");
        }
    }

    #[test]
    fn constant_roles_are_explicit_and_structural() {
        let input = syn::parse_str::<RowAtomInput>(
            "fn atom [order2](x; ordinary: f64, weight: scale, active: bool) {
                weight * (x + ordinary) * active
            }",
        )
        .expect("typed row atom");
        assert_eq!(input.constants.len(), 3);
        assert_eq!(input.scale_constants, [1].into_iter().collect());
        assert_eq!(input.activity_constants, [2].into_iter().collect());
    }

    #[test]
    fn untyped_or_unknown_constant_roles_are_rejected() {
        assert!(
            syn::parse_str::<RowAtomInput>("fn atom [order2](x; weight) { weight * x }",).is_err()
        );
        let error = match syn::parse_str::<RowAtomInput>(
            "fn atom [order2](x; weight: coefficient) { weight * x }",
        ) {
            Ok(_) => panic!("unknown role must fail"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("explicitly typed `f64`, `scale`, or `bool`")
        );
    }
}

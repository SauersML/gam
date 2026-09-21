use std::collections::{BTreeMap, HashMap};

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

use crate::removed_spellings;
use crate::smooth::BoundedCoefficientPriorSpec;
use crate::term_builder::{MARGINAL_SLOPE_Z_ALIAS, marginal_slope_z_alias_is_live};
use gam_problem::types::{
    InverseLink, LikelihoodSpec, LinkComponent, LinkFunction, StandardLink, WigglePenaltyConfig,
};

#[derive(Parser)]
#[grammar_inline = r#"
WHITESPACE = _{ " " | "\t" | NEWLINE }

top_function_call = { SOI ~ function_call ~ EOI }
top_expr = { SOI ~ expr ~ EOI }
formula = { SOI ~ expr ~ "~" ~ rhs ~ EOI }
rhs = { term ~ ("+" ~ term)* }
term = { automatic_rest | expr }
automatic_rest = @{ "." ~ !ASCII_DIGIT }

expr = { sum }
sum = { product ~ (add_op ~ product)* }
add_op = { "+" | "-" }
product = { interact ~ (mul_op ~ interact)* }
mul_op = { "*" | "/" }
interact = { power ~ (interact_op ~ power)* }
interact_op = { ":" }
power = { unary ~ (pow_op ~ unary)* }
pow_op = { "^" }
unary = { unary_op* ~ primary }
unary_op = _{ "+" | "-" }

primary = { function_call | list_lit | tuple_lit | ident | backtick_ident | number | string_lit | "(" ~ expr ~ ")" }
list_lit = @{ "[" ~ (list_lit | !"]" ~ ANY)* ~ "]" }
tuple_lit = @{ "(" ~ (!("," | ")") ~ ANY)+ ~ "," ~ (!")" ~ ANY)* ~ ")" }
function_call = { ident ~ "(" ~ arg_list? ~ ")" }
arg_list = { arg ~ ("," ~ arg)* }
arg = { named_arg | expr }
named_arg = { ident ~ "=" ~ expr }

ident = @{ ident_start ~ ident_continue* }
ident_start = _{ ASCII_ALPHA | "_" }
ident_continue = _{ ASCII_ALPHANUMERIC | "_" | "." }
backtick_ident = @{ "`" ~ (!"`" ~ ANY)+ ~ "`" }

number = @{
    "-"?
    ~ (ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT*)? | "." ~ ASCII_DIGIT+)
    ~ (("e" | "E") ~ ("+" | "-")? ~ ASCII_DIGIT+)?
}

string_lit = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" | "'" ~ (!"'" ~ ANY)* ~ "'" }
"#]
struct FormulaParser;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FormulaDslParse {
    pub response_expr: String,
    pub rhs_terms: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CallArgSpec {
    Positional(String),
    Named { key: String, value: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FunctionCallSpec {
    pub name: String,
    pub args: Vec<CallArgSpec>,
}

/// Typed error surface for the formula DSL parser.
///
/// Every variant carries a free-form `reason: String` payload; `Display`
/// emits exactly that payload, so converting a `FormulaDslError` into
/// `String` (via the `From` impl below) is byte-equivalent to the pre-
/// refactor `Err(format!(...))` / `Err("...".to_string())` strings that
/// the same call sites produced. Public entry points keep their existing
/// `Result<_, String>` signatures — CLI input handling stays unchanged —
/// and typed errors flow across the boundary via `From<FormulaDslError>
/// for String`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormulaDslError {
    /// Pest grammar failure, unbalanced delimiters, empty terms, or
    /// missing required parse fragments — i.e. the formula text is
    /// not a well-formed DSL string.
    ParseError { reason: String },
    /// A referenced symbol (link name, blended-link component, term
    /// function name, top-level RHS identifier) is not part of the
    /// supported vocabulary.
    UnknownIdentifier { reason: String },
    /// A named option's value is unparseable, out of range, or not a
    /// finite number / valid integer.
    InvalidArgument { reason: String },
    /// A combination of terms or options is disallowed (duplicate
    /// terms, multiple linkwiggle/link/survmodel, mutually exclusive
    /// option groups in bounded(), wiggle-incompatible links, etc.).
    IncompatibleTerm { reason: String },
    /// A required configuration option is missing or empty (e.g.
    /// `link()` without `type=`, `survmodel()` with no options,
    /// `bounded()` without a required argument).
    MalformedConfig { reason: String },
}

gam_linalg::impl_reason_error_boilerplate! {
    FormulaDslError {
        ParseError,
        UnknownIdentifier,
        InvalidArgument,
        IncompatibleTerm,
        MalformedConfig,
    }
}

/// Inbound conversion from `String` is used by `?` cascades inside `parse_formula`
/// and friends so that internal parser helpers still returning `Result<_, String>`
/// can flow through without each call site needing an explicit `.map_err(...)`.
/// We route into `ParseError` because by construction every internal helper that
/// still produces a raw `String` is itself a parse/term-resolution stage.
impl From<String> for FormulaDslError {
    fn from(reason: String) -> Self {
        FormulaDslError::ParseError { reason }
    }
}

pub(crate) fn parse_formula_dsl(formula: &str) -> Result<FormulaDslParse, String> {
    validate_balanced_delimiters(formula, "invalid formula syntax")?;
    let mut parsed =
        FormulaParser::parse(Rule::formula, formula).map_err(|e| FormulaDslError::ParseError {
            reason: format!("invalid formula syntax: {e}"),
        })?;
    let formula_pair = parsed.next().ok_or_else(|| FormulaDslError::ParseError {
        reason: "invalid formula syntax: empty parse".to_string(),
    })?;

    let mut response_expr: Option<String> = None;
    let mut rhs_terms: Option<Vec<String>> = None;

    for part in formula_pair.into_inner() {
        if part.as_rule() == Rule::rhs {
            rhs_terms = Some(extract_rhs_terms(part)?);
        } else if part.as_rule() == Rule::expr && response_expr.is_none() {
            // A backtick-quoted response (`` `my y` ~ x ``) names its column
            // verbatim; any other response expression is kept as written.
            response_expr = Some(unquote_column(part.as_str()));
        }
    }

    let response_expr = response_expr.ok_or_else(|| FormulaDslError::ParseError {
        reason: "invalid formula: missing response expression".to_string(),
    })?;
    let rhs_terms = rhs_terms.ok_or_else(|| FormulaDslError::ParseError {
        reason: "invalid formula: missing RHS terms".to_string(),
    })?;
    if rhs_terms.is_empty() {
        return Err(FormulaDslError::ParseError {
            reason: "formula has no usable terms".to_string(),
        }
        .into());
    }

    Ok(FormulaDslParse {
        response_expr,
        rhs_terms,
    })
}

/// Quote state for the scanners that walk raw formula text before (or instead
/// of) the grammar: `'...'` and `"..."` string literals and `` `...` ``
/// backtick-quoted column names. Delimiters, operators and whitespace inside
/// any of them are literal text, so `s(`my col`)` never splits on the space
/// and `` `a+b` `` is one column, not a sum.
#[derive(Default)]
struct QuoteTracker {
    open: Option<char>,
}

impl QuoteTracker {
    /// Advance past `ch`. Returns `true` when `ch` is a quote delimiter or
    /// lies inside a quoted span, i.e. when it is not formula syntax.
    fn step(&mut self, ch: char) -> bool {
        match self.open {
            Some(q) => {
                if ch == q {
                    self.open = None;
                }
                true
            }
            None if matches!(ch, '\'' | '"' | '`') => {
                self.open = Some(ch);
                true
            }
            None => false,
        }
    }

    fn is_open(&self) -> bool {
        self.open.is_some()
    }
}

fn delimiter_balance_error(prefix: &str) -> String {
    format!("{prefix}: unbalanced parentheses or quotes")
}

// Pest reports malformed delimiters as a generic parse failure. We validate the
// raw text first so callers get a stable, specific error class for unmatched
// parentheses/quotes instead of whichever grammar branch happened to fail last.
fn validate_balanced_delimiters(input: &str, prefix: &str) -> Result<(), String> {
    let mut stack = Vec::<char>::new();
    let mut quotes = QuoteTracker::default();

    for ch in input.chars() {
        if quotes.step(ch) {
            continue;
        }
        if matches!(ch, '(' | '[' | '{') {
            stack.push(ch);
        } else if matches!(ch, ')' | ']' | '}') {
            let expected = match ch {
                ')' => '(',
                ']' => '[',
                // The enclosing `matches!` guarantees ch is one of ')', ']', '}'.
                _ => '{',
            };
            if stack.pop() != Some(expected) {
                return Err(FormulaDslError::ParseError {
                    reason: delimiter_balance_error(prefix),
                }
                .into());
            }
        }
    }

    if quotes.is_open() || !stack.is_empty() {
        return Err(FormulaDslError::ParseError {
            reason: delimiter_balance_error(prefix),
        }
        .into());
    }
    Ok(())
}

fn extract_rhs_terms(rhs: Pair<'_, Rule>) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut depth = 0_i32;
    let mut quotes = QuoteTracker::default();
    let mut start = 0_usize;
    // Last non-whitespace character seen at depth 0 outside quotes. A top-level
    // `+` only separates terms when it follows a completed operand; when it
    // follows a binary operator (`:`, `*`, `/`, `^`) or another sign — or opens
    // the RHS — it is a UNARY sign, not a separator. Splitting on it there would
    // hand the grammar a truncated fragment like `x:` (from `x:+z`) and surface
    // a confusing "invalid term syntax in `x:`" instead of the dedicated unary
    // diagnostic the grammar raises when the whole `x:+z` term reaches it. This
    // never suppresses a split in a valid formula: no valid Wilkinson-Rogers RHS
    // places `+` immediately after an operator. `-` is already never treated as
    // a separator here (it is rejected downstream as a binary term operator), so
    // the unary `-` path already flows through intact — this restores the same
    // intact flow for unary `+`.
    let mut last_significant: Option<char> = None;
    let text = rhs.as_str();
    for (idx, ch) in text.char_indices() {
        if quotes.step(ch) {
            last_significant = Some(ch);
            continue;
        }
        if matches!(ch, '(' | '[' | '{') {
            depth += 1;
        } else if matches!(ch, ')' | ']' | '}') && depth > 0 {
            depth -= 1;
        } else if ch == '+'
            && depth == 0
            && !matches!(
                last_significant,
                None | Some(':' | '*' | '/' | '^' | '+' | '-')
            )
        {
            let term = text[start..idx].trim();
            if term.is_empty() {
                return Err(FormulaDslError::ParseError {
                    reason: "formula RHS contains an empty term".to_string(),
                }
                .into());
            }
            out.push(term.to_string());
            start = idx + 1;
        }
        if !ch.is_ascii_whitespace() {
            last_significant = Some(ch);
        }
    }
    if quotes.is_open() || depth != 0 {
        return Err(FormulaDslError::ParseError {
            reason: "formula RHS has unbalanced quotes or parentheses".to_string(),
        }
        .into());
    }
    let tail = text[start..].trim();
    if tail.is_empty() {
        return Err(FormulaDslError::ParseError {
            reason: "formula RHS contains an empty term".to_string(),
        }
        .into());
    }
    out.push(tail.to_string());
    Ok(out)
}

/// Wilkinson-Rogers operator-family expansion.
///
/// A raw RHS term (already split on top-level `+`) may use the documented
/// formula operators `:`, `*`, `/`, `^`, and parenthesization. This function
/// expands the term into a normalized list of `WrAtomList`s, where each
/// `WrAtomList` is one resulting model term — either a single atom (linear
/// main effect / function-call term) or multiple atoms (interaction).
///
/// Wilkinson-Rogers semantics implemented here:
/// * `a` produces `{a}` — one main effect.
/// * `a:b` produces `{a:b}` — one interaction.
/// * `a*b` produces `{a, b, a:b}` — crossing (expanded).
/// * `a/b` produces `{a, a:b}` — nesting. Nesting is hierarchical: each `/`
///   crosses its right operand with the product of ALL variables on its left,
///   so `a/b/c` produces `{a, a:b, a:b:c}` (never a spurious `a:c`) and
///   `a*b/c` produces `{a, b, a:b, a:b:c}`, matching R/mgcv `terms()`.
/// * `(a + b + ... )^n` produces every non-empty subset of `{a, b, ...}` of
///   size at most `n`, each subset being one interaction.
/// * `+` unions two term sets; `-` is rejected.
///
/// Atoms inside the AST may be bare identifiers or function calls. Function
/// calls are opaque — they pass through as-is and cannot participate in an
/// `:` interaction with other atoms (smooths/factors require dedicated
/// constructors like `te()`).
type WrAtomList = Vec<String>;

fn expand_wr_term(raw: &str) -> Result<Vec<WrAtomList>, String> {
    let mut parsed = FormulaParser::parse(Rule::top_expr, raw).map_err(|e| {
        FormulaDslError::ParseError {
            reason: format!("invalid term syntax in `{raw}`: {e}"),
        }
        .to_string()
    })?;
    let top = parsed.next().ok_or_else(|| {
        FormulaDslError::ParseError {
            reason: format!("invalid term syntax in `{raw}`: empty parse"),
        }
        .to_string()
    })?;
    let expr = top
        .into_inner()
        .find(|p| p.as_rule() == Rule::expr)
        .ok_or_else(|| {
            FormulaDslError::ParseError {
                reason: format!("invalid term syntax in `{raw}`: missing expr"),
            }
            .to_string()
        })?;
    let interactions = expand_expr(expr, raw)?;
    let normalized: Vec<WrAtomList> = interactions
        .into_iter()
        .map(normalize_interaction)
        .collect();
    // Deduplicate while preserving order: a*b yields {a, b, a:b}; a*b + a
    // would otherwise list `a` twice.
    let mut seen = std::collections::BTreeSet::<Vec<String>>::new();
    let mut out = Vec::<WrAtomList>::new();
    for term in normalized {
        let key = term.clone();
        if seen.insert(key) {
            out.push(term);
        }
    }
    Ok(out)
}

fn normalize_interaction(mut atoms: WrAtomList) -> WrAtomList {
    atoms.sort();
    atoms.dedup();
    atoms
}

fn expand_expr(pair: Pair<'_, Rule>, raw: &str) -> Result<Vec<WrAtomList>, String> {
    match pair.as_rule() {
        Rule::expr => {
            let inner = pair.into_inner().next().ok_or_else(|| {
                FormulaDslError::ParseError {
                    reason: format!("invalid term syntax in `{raw}`: empty expr"),
                }
                .to_string()
            })?;
            expand_expr(inner, raw)
        }
        Rule::sum => {
            let mut iter = pair.into_inner();
            let first = iter.next().ok_or_else(|| {
                FormulaDslError::ParseError {
                    reason: format!("invalid term syntax in `{raw}`: empty sum"),
                }
                .to_string()
            })?;
            let mut acc = expand_expr(first, raw)?;
            while let Some(op) = iter.next() {
                if op.as_rule() != Rule::add_op {
                    return Err(FormulaDslError::ParseError {
                        reason: format!(
                            "invalid term syntax in `{raw}`: expected add operator, got `{:?}`",
                            op.as_rule()
                        ),
                    }
                    .into());
                }
                let op_str = op.as_str().trim();
                let operand = iter.next().ok_or_else(|| {
                    FormulaDslError::ParseError {
                        reason: format!("invalid term syntax in `{raw}`: dangling `{op_str}`"),
                    }
                    .to_string()
                })?;
                if op_str == "-" {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: format!(
                            "binary `-` is not supported inside a formula term in `{raw}` \
                             (use multiple `+` terms or drop the unwanted predictor explicitly)"
                        ),
                    }
                    .into());
                }
                let mut rhs = expand_expr(operand, raw)?;
                acc.append(&mut rhs);
            }
            Ok(acc)
        }
        Rule::product => {
            let mut iter = pair.into_inner();
            let first = iter.next().ok_or_else(|| {
                FormulaDslError::ParseError {
                    reason: format!("invalid term syntax in `{raw}`: empty product"),
                }
                .to_string()
            })?;
            let mut acc = expand_expr(first, raw)?;
            while let Some(op) = iter.next() {
                if op.as_rule() != Rule::mul_op {
                    return Err(FormulaDslError::ParseError {
                        reason: format!(
                            "invalid term syntax in `{raw}`: expected `*` or `/`, got `{:?}`",
                            op.as_rule()
                        ),
                    }
                    .into());
                }
                let op_str = op.as_str().trim();
                let operand = iter.next().ok_or_else(|| {
                    FormulaDslError::ParseError {
                        reason: format!("invalid term syntax in `{raw}`: dangling `{op_str}`"),
                    }
                    .to_string()
                })?;
                let rhs = expand_expr(operand, raw)?;
                acc = match op_str {
                    "*" => wr_cross(acc, rhs),
                    "/" => wr_nest(acc, rhs),
                    other => {
                        return Err(FormulaDslError::ParseError {
                            reason: format!(
                                "invalid term syntax in `{raw}`: unrecognized mul operator `{other}`"
                            ),
                        }
                        .into());
                    }
                };
            }
            Ok(acc)
        }
        Rule::interact => {
            let mut iter = pair.into_inner();
            let first = iter.next().ok_or_else(|| {
                FormulaDslError::ParseError {
                    reason: format!("invalid term syntax in `{raw}`: empty interact"),
                }
                .to_string()
            })?;
            let mut acc = expand_expr(first, raw)?;
            while let Some(op) = iter.next() {
                if op.as_rule() != Rule::interact_op {
                    return Err(FormulaDslError::ParseError {
                        reason: format!(
                            "invalid term syntax in `{raw}`: expected `:`, got `{:?}`",
                            op.as_rule()
                        ),
                    }
                    .into());
                }
                let operand = iter.next().ok_or_else(|| {
                    FormulaDslError::ParseError {
                        reason: format!("invalid term syntax in `{raw}`: dangling `:`"),
                    }
                    .to_string()
                })?;
                let rhs = expand_expr(operand, raw)?;
                acc = wr_interact(acc, rhs, raw)?;
            }
            Ok(acc)
        }
        Rule::power => {
            let mut iter = pair.into_inner();
            let first = iter.next().ok_or_else(|| {
                FormulaDslError::ParseError {
                    reason: format!("invalid term syntax in `{raw}`: empty power"),
                }
                .to_string()
            })?;
            let base = expand_expr(first, raw)?;
            let Some(op) = iter.next() else {
                return Ok(base);
            };
            if op.as_rule() != Rule::pow_op {
                return Err(FormulaDslError::ParseError {
                    reason: format!(
                        "invalid term syntax in `{raw}`: expected `^`, got `{:?}`",
                        op.as_rule()
                    ),
                }
                .into());
            }
            let exponent_pair = iter.next().ok_or_else(|| {
                FormulaDslError::ParseError {
                    reason: format!("invalid term syntax in `{raw}`: dangling `^`"),
                }
                .to_string()
            })?;
            let exp_text = exponent_pair.as_str().trim();
            let n: usize = exp_text.parse().map_err(|_| {
                FormulaDslError::ParseError {
                    reason: format!(
                        "invalid term syntax in `{raw}`: `^` exponent must be a positive integer, got `{exp_text}`"
                    ),
                }
                .to_string()
            })?;
            if n == 0 {
                return Err(FormulaDslError::ParseError {
                    reason: format!(
                        "invalid term syntax in `{raw}`: `^0` is not a meaningful formula expansion"
                    ),
                }
                .into());
            }
            if let Some(extra_op) = iter.next() {
                let extra = extra_op.as_str().trim();
                return Err(FormulaDslError::ParseError {
                    reason: format!(
                        "invalid term syntax in `{raw}`: chained `^` operators are not supported; \
                         use one positive integer exponent, got another `{extra}`"
                    ),
                }
                .into());
            }
            Ok(wr_power(base, n))
        }
        Rule::unary => {
            let unary_start = pair.as_span().start();
            for inner in pair.into_inner() {
                if inner.as_rule() == Rule::primary {
                    let prefix_len = inner.as_span().start().saturating_sub(unary_start);
                    if prefix_len > 0 {
                        let prefix = &raw[unary_start..inner.as_span().start()];
                        if prefix.chars().any(|ch| matches!(ch, '+' | '-')) {
                            return Err(FormulaDslError::IncompatibleTerm {
                                reason: format!(
                                    "unary `+`/`-` is not supported inside a formula term in `{raw}`"
                                ),
                            }
                            .into());
                        }
                    }
                    return expand_expr(inner, raw);
                }
            }
            Err(FormulaDslError::ParseError {
                reason: format!("invalid term syntax in `{raw}`: empty unary"),
            }
            .into())
        }
        Rule::primary => {
            let span = pair.as_str().trim().to_string();
            let inner = pair.into_inner().next();
            match inner {
                Some(child) if child.as_rule() == Rule::expr => expand_expr(child, raw),
                Some(child) => Ok(vec![vec![child.as_str().trim().to_string()]]),
                None => Ok(vec![vec![span]]),
            }
        }
        _ => Err(FormulaDslError::ParseError {
            reason: format!(
                "invalid term syntax in `{raw}`: unexpected node `{:?}`",
                pair.as_rule()
            ),
        }
        .into()),
    }
}

fn wr_cross(left: Vec<WrAtomList>, right: Vec<WrAtomList>) -> Vec<WrAtomList> {
    // a*b = a + b + a:b
    let mut out = Vec::with_capacity(left.len() + right.len() + left.len() * right.len());
    out.extend(left.iter().cloned());
    out.extend(right.iter().cloned());
    for l in &left {
        for r in &right {
            let mut merged: WrAtomList = l.iter().cloned().chain(r.iter().cloned()).collect();
            merged.sort();
            merged.dedup();
            out.push(merged);
        }
    }
    out
}

fn wr_nest(left: Vec<WrAtomList>, right: Vec<WrAtomList>) -> Vec<WrAtomList> {
    // Wilkinson-Rogers nesting `A/B`: keep every term of `A`, then nest `B`
    // within the *whole* left-hand group by crossing each right term with the
    // product of ALL variables appearing anywhere on the left (not with each
    // left term separately). This makes chained nesting hierarchical —
    // `a/b/c` = {a, a:b, a:b:c}, never a spurious `a:c` — and matches R/mgcv
    // `terms()` exactly, including `a*b/c` = {a, b, a:b, a:b:c} and
    // `(a+b)/c` = {a, b, a:b:c}. See `parse_formula_chained_wr_nesting_is_hierarchical`.
    let mut left_atoms: WrAtomList = left.iter().flatten().cloned().collect();
    left_atoms.sort();
    left_atoms.dedup();

    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend(left.iter().cloned());
    for r in &right {
        let mut merged: WrAtomList = left_atoms
            .iter()
            .cloned()
            .chain(r.iter().cloned())
            .collect();
        merged.sort();
        merged.dedup();
        out.push(merged);
    }
    out
}

fn wr_interact(
    left: Vec<WrAtomList>,
    right: Vec<WrAtomList>,
    raw: &str,
) -> Result<Vec<WrAtomList>, String> {
    // a:b — Cartesian merge of every (l, r) pair.
    let mut out = Vec::with_capacity(left.len() * right.len());
    for l in &left {
        for r in &right {
            let combined: WrAtomList = l.iter().cloned().chain(r.iter().cloned()).collect();
            // Reject interactions whose atoms contain function-call syntax.
            // These would build design columns that are not simple products
            // (factors, smooths) and need a dedicated constructor.
            for atom in &combined {
                if !is_backtick_ident(atom) && atom.contains('(') {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: format!(
                            "interaction operator `:` with function-call atom is not supported in `{raw}`. \
                             Use te(...) for smooth interactions or group()/factor() with a separate \
                             interaction strategy for categorical effects."
                        ),
                    }
                    .into());
                }
            }
            let mut merged = combined;
            merged.sort();
            merged.dedup();
            out.push(merged);
        }
    }
    Ok(out)
}

fn wr_power(base: Vec<WrAtomList>, n: usize) -> Vec<WrAtomList> {
    // (e)^n produces every non-empty subset of `base` of size ≤ n,
    // each subset being one interaction. The standard WR semantics
    // treat the base as a *set* of atoms — interactions inside `base`
    // are kept as-is and not re-crossed.
    if base.is_empty() {
        return Vec::new();
    }
    let m = base.len();
    let mut out = Vec::<WrAtomList>::new();
    let max_size = n.min(m);
    for size in 1..=max_size {
        // Enumerate combinations of `size` elements out of m.
        let mut indices: Vec<usize> = (0..size).collect();
        loop {
            let mut merged = WrAtomList::new();
            for &i in &indices {
                merged.extend(base[i].iter().cloned());
            }
            merged.sort();
            merged.dedup();
            out.push(merged);
            // Next combination
            let mut k = size;
            while k > 0 {
                k -= 1;
                if indices[k] != k + m - size {
                    indices[k] += 1;
                    for j in (k + 1)..size {
                        indices[j] = indices[j - 1] + 1;
                    }
                    break;
                }
                if k == 0 {
                    k = usize::MAX;
                    break;
                }
            }
            if k == usize::MAX {
                break;
            }
        }
    }
    out
}

/// A backtick-quoted column reference: `` `my col` ``, `` `x.1` ``, `` `2020` ``,
/// `` `höhe` ``. Everything between the backticks is the column name verbatim,
/// so names with spaces, operators, leading digits or non-ASCII characters can
/// be used anywhere a bare column name can (R's non-syntactic-name quoting).
fn is_backtick_ident(raw: &str) -> bool {
    raw.len() >= 3
        && raw.starts_with('`')
        && raw.ends_with('`')
        && !raw[1..raw.len() - 1].contains('`')
}

/// A column reference: a bare identifier or a backtick-quoted name.
fn is_column_ref(raw: &str) -> bool {
    is_exact_ident(raw) || is_backtick_ident(raw)
}

/// The column name a reference denotes: backticks are removed, a bare
/// identifier is returned unchanged.
fn unquote_column(raw: &str) -> String {
    let trimmed = raw.trim();
    if is_backtick_ident(trimmed) {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

/// Split `a:b:c` on the `:` separators that sit outside backtick quotes.
fn split_outside_backticks(raw: &str, sep: char) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    for (i, ch) in raw.char_indices() {
        if ch == '`' {
            quoted = !quoted;
        } else if ch == sep && !quoted {
            pieces.push(&raw[start..i]);
            start = i + ch.len_utf8();
        }
    }
    pieces.push(&raw[start..]);
    pieces
}

fn is_exact_ident(raw: &str) -> bool {
    let mut chars = raw.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.')
}

pub(crate) fn parse_function_call(input: &str) -> Result<FunctionCallSpec, String> {
    validate_balanced_delimiters(input, "invalid function call syntax")?;
    let mut parsed = FormulaParser::parse(Rule::top_function_call, input).map_err(|e| {
        FormulaDslError::ParseError {
            reason: format!("invalid function call syntax: {e}"),
        }
    })?;
    let top = parsed.next().ok_or_else(|| FormulaDslError::ParseError {
        reason: "invalid function call syntax: empty parse".to_string(),
    })?;
    let call = top
        .into_inner()
        .find(|p| p.as_rule() == Rule::function_call)
        .ok_or_else(|| FormulaDslError::ParseError {
            reason: "invalid function call syntax: missing call".to_string(),
        })?;
    parse_call_pair(call)
}

fn parse_call_pair(call: Pair<'_, Rule>) -> Result<FunctionCallSpec, String> {
    let mut name: Option<String> = None;
    let mut args = Vec::<CallArgSpec>::new();
    for part in call.into_inner() {
        if part.as_rule() == Rule::ident {
            if name.is_none() {
                name = Some(part.as_str().trim().to_string());
            }
        } else if part.as_rule() == Rule::arg_list {
            for a in part.into_inner() {
                if a.as_rule() != Rule::arg {
                    continue;
                }
                let mut a_inner = a.into_inner();
                let Some(first) = a_inner.next() else {
                    continue;
                };
                if first.as_rule() == Rule::named_arg {
                    let mut ni = first.into_inner();
                    let key = ni
                        .next()
                        .ok_or_else(|| FormulaDslError::ParseError {
                            reason: "invalid named argument key".to_string(),
                        })?
                        .as_str()
                        .trim()
                        .to_ascii_lowercase();
                    let value = ni
                        .next()
                        .ok_or_else(|| FormulaDslError::ParseError {
                            reason: "invalid named argument value".to_string(),
                        })?
                        .as_str()
                        .trim()
                        .to_string();
                    args.push(CallArgSpec::Named { key, value });
                } else if first.as_rule() == Rule::expr {
                    args.push(CallArgSpec::Positional(first.as_str().trim().to_string()));
                }
            }
        }
    }
    let name = name.ok_or_else(|| FormulaDslError::ParseError {
        reason: "invalid function call: missing name".to_string(),
    })?;
    Ok(FunctionCallSpec { name, args })
}

#[cfg(test)]
mod tests {
    use super::{
        CallArgSpec, ParsedTerm, formula_response_column, parse_formula, parse_formula_dsl,
        parse_function_call, parse_linkwiggle_formulaspec, parsed_term_column_names,
        parsed_terms_reference_column, validate_marginal_slope_z_alias_exclusion,
        validate_marginal_slope_z_column_exclusion,
    };
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    #[test]
    fn parsed_term_column_names_includes_by_smooth_grouping_variable() {
        // The model's input contract must include a smooth's `by=` column, not
        // just its positional variable. `s(x, by=g)` consumes both `x` and `g`
        // (`term_builder` reads `options["by"]`); dropping `g` from the
        // required/consumable set would either omit a genuine predictor at fit
        // (CLI projected load) or — worse — make predict silently project the
        // `g` column away (#840 regression). A bare main effect and an
        // interaction round out the variant coverage.
        let parsed =
            parse_formula("y ~ s(x, by=g) + z + a:b").expect("formula with a by= smooth parses");
        let mut cols = BTreeSet::<String>::new();
        parsed_term_column_names(&parsed.terms, &mut cols);
        for expected in ["x", "g", "z", "a", "b"] {
            assert!(
                cols.contains(expected),
                "parsed_term_column_names dropped '{expected}'; got {cols:?}"
            );
        }
        // The response is never a *term* column (it is handled separately).
        assert!(
            !cols.contains("y"),
            "response leaked into term columns: {cols:?}"
        );
    }

    #[test]
    fn linkwiggle_parser_does_not_bake_in_cubic_only_restriction() {
        // Regression for #384: the cubic-only constraint belongs to the
        // score-warp / link-deviation `DeviationRuntime`, NOT to this shared
        // parser. `parse_linkwiggle_formulaspec` also feeds `timewiggle` and
        // the location-scale survival path, whose general monotone I-spline
        // value basis honors any `degree >= 2`. So the parser must accept
        // non-cubic degrees and carry them through verbatim; the cubic gate is
        // applied downstream only where the cubic-only runtime is built. A
        // prior fix wrongly forced `degree == 3` here, breaking timewiggle and
        // location-scale callers — this pins that the parser stays general.
        for deg in [2usize, 4, 5, 10] {
            let mut options = BTreeMap::new();
            options.insert("degree".to_string(), deg.to_string());
            options.insert("internal_knots".to_string(), "3".to_string());
            let raw = format!("timewiggle(degree={deg}, internal_knots=3)");
            let spec = parse_linkwiggle_formulaspec(&options, &raw)
                .expect("non-cubic wiggle degree must parse at the shared layer");
            assert_eq!(
                spec.degree, deg,
                "parser must carry the requested degree through verbatim"
            );
        }

        // The only universal lower bound the shared parser enforces is that a
        // polynomial degree is positive.
        let mut zero = BTreeMap::new();
        zero.insert("degree".to_string(), "0".to_string());
        zero.insert("internal_knots".to_string(), "3".to_string());
        let err = parse_linkwiggle_formulaspec(&zero, "linkwiggle(degree=0, internal_knots=3)")
            .expect_err("degree=0 must be rejected");
        assert!(
            err.contains("degree >= 1"),
            "error should state the positive-degree lower bound, got: {err}"
        );
    }

    #[test]
    fn parses_nested_formula_terms() {
        let parsed =
            parse_formula_dsl("log(y) ~ x1 + s(log(x2 + 1), bs=\"tps\", k=10) + te(x3, x4)")
                .expect("parse");
        assert_eq!(parsed.response_expr, "log(y)");
        assert_eq!(parsed.rhs_terms.len(), 3);
        assert_eq!(parsed.rhs_terms[0], "x1");
        assert_eq!(parsed.rhs_terms[1], "s(log(x2 + 1), bs=\"tps\", k=10)");
        assert_eq!(parsed.rhs_terms[2], "te(x3, x4)");
    }

    /// Parse `y ~ {term}` and return the error text.
    fn term_error(term: &str) -> String {
        match parse_formula(&format!("y ~ {term}")) {
            Ok(parsed) => panic!("`{term}` parsed but must be refused: {:?}", parsed.terms),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn removed_formula_spellings_error_and_name_the_canonical_one() {
        // One spelling per behavior (SPEC R25): each removed spelling is a
        // parse error that names the single supported spelling.
        let cases = [
            ("s(x, bs=tp)", "unknown smooth type `tp`; use `tps`"),
            ("s(x, bs='gp')", "unknown smooth type `gp`; use `matern`"),
            ("s(x, bs=cs)", "unknown smooth type `cs`; use `cr`"),
            ("s(x, bs=cc)", "unknown smooth type `cc`; use `cyclic`"),
            ("s(x, bs=cp)", "unknown smooth type `cp`; use `cyclic`"),
            ("s(x, bs=periodic)", "unknown smooth type `periodic`; use `cyclic`"),
            ("s(x, bs='cyclic-ps')", "unknown smooth type `cyclic-ps`; use `cyclic`"),
            ("s(x, bs=mkappa)", "unknown smooth type `mkappa`; use `curv`"),
            ("s(x, bs=web)", "unknown smooth type `web`; use `mjs`"),
            ("te(x, z, bs=c('cc','ps'))", "unknown smooth type `cc`; use `cyclic`"),
            ("te(x, z, bs=['tp','cr'])", "unknown smooth type `tp`; use `tps`"),
            ("s(x, type=ps)", "unknown option `type` in s(); use `bs`"),
            ("s(x, basis_dim=8)", "unknown option `basis_dim` in s(); use `k`"),
            ("s(x, basisdim=8)", "unknown option `basisdim` in s(); use `k`"),
            ("te(x, z, m=2)", "unknown option `m` in te(); use `penalty_order`"),
            ("sphere(lat, lon, kernel=sobolev)", "unknown option `kernel` in sphere(); use `method`"),
            ("sphere(lat, lon, method=wahba)", "unknown sphere method `wahba` in sphere(); use `sobolev`"),
            ("sphere(lat, lon, method='wahba-sobolev')", "unknown sphere method `wahba-sobolev` in sphere(); use `sobolev`"),
            ("s(x, bc=cyclic)", "unknown boundary token `cyclic` in s(bc=...); use `periodic`"),
            ("constrain(x, min=0)", "unknown term function `constrain`; use `linear()`"),
            ("constraint(x, min=0)", "unknown term function `constraint`; use `linear()`"),
            ("box(x, min=0, max=1)", "unknown term function `box`; use `linear()`"),
            ("nonnegative_coef(x)", "unknown term function `nonnegative_coef`; use `nonnegative()`"),
            ("nonpositive_coef(x)", "unknown term function `nonpositive_coef`; use `nonpositive()`"),
            ("re(g)", "unknown term function `re`; use `group()`"),
            ("tensor(x, z)", "unknown term function `tensor`; use `te()`"),
            ("interaction(x, z)", "unknown term function `interaction`; use `te()`"),
            ("periodic(x)", "unknown term function `periodic`; use `cyclic()`"),
            ("cc(x)", "unknown term function `cc`; use `cyclic()`"),
            ("measurejet(x, z)", "unknown term function `measurejet`; use `mjs()`"),
            ("web(x, z)", "unknown term function `web`; use `mjs()`"),
            ("curvature(x, z)", "unknown term function `curvature`; use `curv()`"),
            ("mkappa(x, z)", "unknown term function `mkappa`; use `curv()`"),
            ("linear(x, lower=0)", "unknown option `lower` in linear(); use `min`"),
            ("linear(x, upper=1)", "unknown option `upper` in linear(); use `max`"),
            ("bounded(x, min=0, max=1, pull=uniform)", "unknown option `pull` in bounded(); use `prior`"),
            ("bounded(x, min=0, max=1, prior=log-jacobian)", "unknown bounded() prior `log-jacobian`; use `uniform`"),
            ("bounded(x, min=0, max=1, prior=jacobian)", "unknown bounded() prior `jacobian`; use `uniform`"),
            ("bounded(x, min=0, max=1, prior=none)", "unknown bounded() prior `none`; use `uniform`"),
        ];
        for (term, expected) in cases {
            let err = term_error(term);
            assert!(err.contains(expected), "`{term}` error `{err}` must contain `{expected}`");
        }
    }

    /// `bounded(prior=uniform)` is flat on the coefficient over the box: the
    /// same posterior as the unpenalised boxed `linear()` term, so it must be
    /// that term and publish its truncated posterior mean (#3479), not a
    /// separate latent-logit route whose mode differs from it.
    #[test]
    fn bounded_uniform_prior_is_the_unpenalised_boxed_linear_term() {
        let uniform = parse_formula("y ~ bounded(x, min=1, max=3, prior=uniform)")
            .expect("bounded(prior=uniform) parses");
        match uniform.terms.as_slice() {
            [ParsedTerm::Linear {
                name,
                explicit: true,
                double_penalty: false,
                coefficient_min: Some(min),
                coefficient_max: Some(max),
            }] => {
                assert_eq!(name, "x");
                assert_eq!((*min, *max), (1.0, 3.0));
            }
            other => panic!("bounded(prior=uniform) must lower to the boxed linear term: {other:?}"),
        }
        let err = term_error("bounded(x, min=1, max=3, prior=uniform, double_penalty=true)");
        assert!(
            err.contains("cannot also carry the double_penalty ridge"),
            "a flat box with a ridge must be refused, got `{err}`"
        );
    }

    #[test]
    fn canonical_formula_spellings_parse() {
        for term in [
            "s(x, bs=tps, k=8)",
            "s(x, bs=matern)",
            "s(x, bs=cr, penalty_order=2)",
            "s(x, bs=cyclic)",
            "te(x, z, bs=c('cyclic','ps'))",
            "sphere(lat, lon, method=sobolev)",
            "sphere(lat, lon, method=harmonic)",
            "s(x, bc=periodic)",
            "linear(x, min=0, max=1)",
            "nonnegative(x)",
            "nonpositive(x)",
            "group(g)",
            "te(x, z)",
            "cyclic(x)",
            "mjs(x, z)",
            "curv(x, z)",
            "bounded(x, min=0, max=1, prior=uniform)",
        ] {
            parse_formula(&format!("y ~ {term}"))
                .unwrap_or_else(|err| panic!("canonical `{term}` must parse: {err}"));
        }
    }

    #[test]
    fn function_form_sets_its_basis_and_refuses_a_different_one() {
        for (term, basis) in [
            ("tps(x, z)", "tps"),
            ("cyclic(x)", "cyclic"),
            ("mjs(x, z)", "mjs"),
            ("curv(x, z)", "curv"),
            ("matern(x)", "matern"),
            ("duchon(x)", "duchon"),
        ] {
            let parsed = parse_formula(&format!("y ~ {term}")).expect("parse");
            match &parsed.terms[0] {
                ParsedTerm::Smooth { options, .. } => {
                    assert_eq!(options.get("bs").map(String::as_str), Some(basis), "{term}");
                    assert!(!options.contains_key("type"), "{term}");
                }
                other => panic!("expected a smooth for {term}, got {other:?}"),
            }
        }
        let err = term_error("cyclic(x, bs=ps)");
        assert!(
            err.contains("cyclic() is the `cyclic` smooth and cannot take bs=`ps`"),
            "{err}"
        );
    }

    #[test]
    fn parses_cyclic_formula_function() {
        let parsed = parse_formula("y ~ cyclic(theta, period_start=0, period_end=6.283)")
            .expect("parse cyclic formula");
        match &parsed.terms[0] {
            super::ParsedTerm::Smooth { vars, options, .. } => {
                assert_eq!(vars, &vec!["theta".to_string()]);
                assert_eq!(options.get("bs").map(String::as_str), Some("cyclic"));
                assert_eq!(options.get("period_start").map(String::as_str), Some("0"));
            }
            other => panic!("expected cyclic smooth term, got {other:?}"),
        }
    }

    #[test]
    fn sphere_aliases_all_dispatch_to_intrinsic_s2_basis() {
        // Regression for #383: `s2(lat, lon)` must build the same intrinsic
        // S² (sphere) basis as `sphere()`/`sos()`/`spherical()`. Previously the
        // `s2` arm returned a Smooth without `bs=sphere`, so it silently fell
        // back to a generic Euclidean 2-D smooth and diverged in the
        // spatial-kappa optimizer. All four must be byte-for-byte
        // equivalent in their dispatch (vars + `bs=sphere`).
        for alias in ["sphere", "sos", "spherical", "s2"] {
            let parsed = parse_formula(&format!("y ~ {alias}(lat, lon)"))
                .unwrap_or_else(|e| panic!("parse {alias}: {e}"));
            match &parsed.terms[0] {
                super::ParsedTerm::Smooth { vars, options, .. } => {
                    assert_eq!(
                        vars,
                        &vec!["lat".to_string(), "lon".to_string()],
                        "{alias} should keep (lat, lon) as its variables"
                    );
                    assert_eq!(
                        options.get("bs").map(String::as_str),
                        Some("sphere"),
                        "{alias} must dispatch to the intrinsic sphere basis (bs=sphere)"
                    );
                }
                other => panic!("expected sphere smooth term for {alias}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parses_function_callwithnamed_and_positional_args() {
        let call = parse_function_call("s(log(x + 1), bs=\"duchon\", centers=12)").expect("call");
        assert_eq!(call.name, "s");
        assert_eq!(call.args.len(), 3);
        assert_eq!(
            call.args[0],
            CallArgSpec::Positional("log(x + 1)".to_string())
        );
        assert_eq!(
            call.args[1],
            CallArgSpec::Named {
                key: "bs".to_string(),
                value: "\"duchon\"".to_string()
            }
        );
    }

    #[test]
    fn parses_tensor_boundary_list_options() {
        let call = parse_function_call(
            "te(day_of_week, hour, boundary=['periodic', 'periodic'], period=[7, 24])",
        )
        .expect("call");
        assert_eq!(call.name, "te");
        assert_eq!(call.args.len(), 4);
        assert_eq!(
            call.args[2],
            CallArgSpec::Named {
                key: "boundary".to_string(),
                value: "['periodic', 'periodic']".to_string(),
            }
        );
    }

    #[test]
    fn parse_formula_dsl_reports_unbalanced_parentheses() {
        let err = parse_formula_dsl("y ~ s(x, k=10").expect_err("expected parse failure");
        assert!(err.contains("unbalanced parentheses"));
    }

    #[test]
    fn parse_function_call_reports_unbalanced_parentheses() {
        let err = parse_function_call("s(x, k=10").expect_err("expected parse failure");
        assert!(err.contains("unbalanced parentheses"));
    }

    #[test]
    fn parse_formula_accepts_tuple_smooth_options() {
        let parsed = parse_formula("z ~ te(x, y, k=(20, 20))")
            .expect("tuple-valued smooth option should parse");
        assert_eq!(parsed.terms.len(), 1);

        let dsl = parse_formula_dsl("z ~ te(x, y, k=(20, 20))")
            .expect("tuple-valued smooth option should parse in the DSL layer");
        assert_eq!(dsl.rhs_terms, vec!["te(x, y, k=(20, 20))"]);

        let call = parse_function_call("te(x, y, k=(20, 20))")
            .expect("tuple-valued smooth option should parse as a function call");
        assert_eq!(
            call.args[2],
            CallArgSpec::Named {
                key: "k".to_string(),
                value: "(20, 20)".to_string(),
            }
        );
    }

    #[test]
    fn parse_formula_rejects_unsupported_top_level_rhs_expressions() {
        // Binary `-`, unary `-` and bare parens are not supported as top-level
        // RHS expressions; they must surface through the bare-identifier
        // check in `parse_term` with the same diagnostic. A trailing `- 1` is
        // intercept removal, not term subtraction, and is covered by
        // `intercept_removal_spellings_all_lower_to_the_no_intercept_marker`.
        // WR operators `:`, `*`, `/`, and `^` are intentionally supported by
        // `expand_wr_term` and are exercised by
        // `parse_formula_supports_wr_slash_nesting` /
        // `parse_formula_supports_wr_star_crossing`; they must NOT appear in
        // this list.
        for formula in ["y ~ x - z", "y ~ -x", "y ~ (x)", "y ~ x - 2"] {
            let err = parse_formula(formula).expect_err("expected formula parse failure");
            assert!(err.to_string().contains("unsupported top-level RHS term"));
        }
    }

    /// Wilkinson-Rogers `a/b` (nesting) is documented and implemented by
    /// `expand_wr_term` to yield `{a, a:b}`. This test pins that contract end
    /// to end so a regression in either the grammar's `mul_op = { "*" | "/" }`
    /// branch or in `expand_expr`'s `Rule::product` handling surfaces here
    /// instead of slipping into a silent re-interpretation as a rejection.
    #[test]
    fn parse_formula_supports_wr_slash_nesting() {
        let parsed = parse_formula("y ~ x / z").expect("`/` is supported as WR nesting");
        assert_eq!(parsed.response, "y");
        assert_eq!(parsed.terms.len(), 2);
        let names: Vec<String> = parsed
            .terms
            .iter()
            .map(|t| match t {
                ParsedTerm::Linear { name, .. } => format!("Linear({name})"),
                ParsedTerm::Interaction { vars, .. } => {
                    format!("Interaction({})", vars.join(":"))
                }
                other => format!("Other({other:?})"),
            })
            .collect();
        assert_eq!(
            names,
            vec!["Linear(x)".to_string(), "Interaction(x:z)".to_string()]
        );
    }

    /// Collect the WR term labels of a parsed formula in `:`-joined form,
    /// matching R/mgcv `attr(terms(...), "term.labels")` so nesting/crossing
    /// expansions can be pinned against ground truth.
    fn wr_term_labels(formula: &str) -> Vec<String> {
        let parsed = parse_formula(formula).unwrap_or_else(|e| panic!("parse {formula}: {e}"));
        parsed
            .terms
            .iter()
            .map(|t| match t {
                ParsedTerm::Linear { name, .. } => name.clone(),
                ParsedTerm::Interaction { vars, .. } => vars.join(":"),
                other => format!("Other({other:?})"),
            })
            .collect()
    }

    /// Regression for #2290: chained Wilkinson-Rogers nesting `a/b/c` must be
    /// hierarchical — the deepest interaction crosses `c` with the FULL `a:b`,
    /// giving `{a, a:b, a:b:c}` with NO spurious `a:c`. Before the fix,
    /// `Rule::product`'s left-associative fold distributed `c` over every
    /// accumulated left term (`{a, a:b}`), minting `a:c`. Each expansion is
    /// pinned against R/mgcv `terms()` ground truth.
    #[test]
    fn parse_formula_chained_wr_nesting_is_hierarchical() {
        // R: terms(y ~ a/b/c) -> a  a:b  a:b:c
        assert_eq!(
            wr_term_labels("y ~ a/b/c"),
            vec!["a".to_string(), "a:b".to_string(), "a:b:c".to_string()],
            "a/b/c must nest hierarchically with no spurious a:c"
        );
        // R: terms(y ~ a*b/c) -> a  b  a:b  a:b:c
        assert_eq!(
            wr_term_labels("y ~ a*b/c"),
            vec![
                "a".to_string(),
                "b".to_string(),
                "a:b".to_string(),
                "a:b:c".to_string()
            ],
            "a*b/c nests c within the whole a*b group"
        );
        // R: terms(y ~ x/z) -> x  x:z  (single-level nesting unchanged)
        assert_eq!(
            wr_term_labels("y ~ x/z"),
            vec!["x".to_string(), "x:z".to_string()],
        );
        // R: terms(y ~ a/b/c/d) -> a  a:b  a:b:c  a:b:c:d
        assert_eq!(
            wr_term_labels("y ~ a/b/c/d"),
            vec![
                "a".to_string(),
                "a:b".to_string(),
                "a:b:c".to_string(),
                "a:b:c:d".to_string()
            ],
        );
    }

    /// Wilkinson-Rogers `a*b` (crossing) is documented and implemented by
    /// `expand_wr_term` to yield `{a, b, a:b}`. Pin the contract so a future
    /// refactor of `mul_op` / `Rule::product` handling cannot silently change
    /// the set of model terms.
    #[test]
    fn parse_formula_supports_wr_star_crossing() {
        let parsed = parse_formula("y ~ x * z").expect("`*` is supported as WR crossing");
        assert_eq!(parsed.response, "y");
        assert_eq!(parsed.terms.len(), 3);
        let names: Vec<String> = parsed
            .terms
            .iter()
            .map(|t| match t {
                ParsedTerm::Linear { name, .. } => format!("Linear({name})"),
                ParsedTerm::Interaction { vars, .. } => {
                    format!("Interaction({})", vars.join(":"))
                }
                other => format!("Other({other:?})"),
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "Linear(x)".to_string(),
                "Linear(z)".to_string(),
                "Interaction(x:z)".to_string(),
            ]
        );
    }

    #[test]
    fn parse_formula_rejects_unary_signs_inside_wr_expansion() {
        for formula in ["y ~ x:-z", "y ~ a*-b", "y ~ x/-z", "y ~ x:+z"] {
            let err = parse_formula(formula)
                .expect_err("WR expansion must not silently drop unary signs");
            let msg = err.to_string();
            assert!(
                msg.contains("unary `+`/`-` is not supported"),
                "unexpected error for {formula}: {msg}"
            );
        }
    }

    #[test]
    fn parse_formula_supports_wr_power_crossing() {
        let parsed = parse_formula("y ~ (x + z)^2").expect("`^` is supported as WR power");
        assert_eq!(parsed.response, "y");
        assert_eq!(parsed.terms.len(), 3);
        let names: Vec<String> = parsed
            .terms
            .iter()
            .map(|t| match t {
                ParsedTerm::Linear { name, .. } => format!("Linear({name})"),
                ParsedTerm::Interaction { vars, .. } => {
                    format!("Interaction({})", vars.join(":"))
                }
                other => format!("Other({other:?})"),
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "Linear(x)".to_string(),
                "Linear(z)".to_string(),
                "Interaction(x:z)".to_string(),
            ]
        );
    }

    #[test]
    fn parse_formula_rejects_chained_wr_power() {
        let err = parse_formula("y ~ (x + z)^2^3")
            .expect_err("chained WR powers must not silently drop later exponents");
        let msg = err.to_string();
        assert!(
            msg.contains("chained `^` operators are not supported"),
            "error should explain that chained powers are rejected, got: {msg}"
        );
    }

    #[test]
    fn parsed_terms_reference_column_sees_the_by_smooth_variable() {
        // Regression for #807: the by= grouping variable lives in
        // options["by"], not the smooth's positional vars. The reference
        // predicate must still recognise it, both so the CLI loads the column
        // and so the marginal-slope z-column exclusion check (which reuses this
        // predicate) cannot be fooled into aliasing a reserved z onto a by=.
        let parsed = parse_formula("y ~ s(x, by=g)").expect("parse by-smooth");
        assert!(
            parsed_terms_reference_column(&parsed.terms, "g"),
            "s(x, by=g) references column g via options[\"by\"]"
        );
        assert!(parsed_terms_reference_column(&parsed.terms, "x"));
        assert!(!parsed_terms_reference_column(&parsed.terms, "absent"));
    }

    #[test]
    fn marginal_slope_z_column_validator_detects_linear_and_smooth_reuse() {
        let main = parse_formula("y ~ x + z").expect("parse main");
        let slope = parse_formula("y ~ s(z, bs=duchon, centers=6)").expect("parse slope");

        assert!(parsed_terms_reference_column(&main.terms, "z"));
        assert!(parsed_terms_reference_column(&slope.terms, "z"));

        let err = validate_marginal_slope_z_column_exclusion(
            &main,
            &parse_formula("y ~ 1").expect("parse clean slope"),
            "z",
            "bernoulli marginal-slope",
            "--slope-formula",
        )
        .expect_err("main formula should be rejected");
        assert!(err.contains("cannot also appear in the main formula"));

        let err = validate_marginal_slope_z_column_exclusion(
            &parse_formula("y ~ x").expect("parse clean main"),
            &slope,
            "z",
            "bernoulli marginal-slope",
            "--slope-formula",
        )
        .expect_err("slope formula should be rejected");
        assert!(err.contains("cannot also appear in --slope-formula"));
    }

    /// Column map helper for the alias tests: names -> positional indices.
    fn frame(columns: &[&str]) -> HashMap<String, usize> {
        columns
            .iter()
            .enumerate()
            .map(|(i, c)| ((*c).to_string(), i))
            .collect()
    }

    /// The case the literal-name validator is structurally blind to.
    ///
    /// Every pre-existing test above uses `z_column = "z"`, where the reserved
    /// name and the canonical alias are the same string — so none of them can
    /// distinguish a validator that resolves the alias from one that only
    /// compares text. With `z_column != "z"` they come apart, and `y ~ x + z`
    /// silently put the score into the marginal design (gam#2432).
    #[test]
    fn marginal_slope_alias_validator_rejects_bare_z_when_z_column_is_named_otherwise() {
        let cols = frame(&["y", "x", "pgs_ctn_z"]);
        let main = parse_formula("y ~ x + z").expect("parse main");

        // The literal-name validator accepts it: the term is named `z`, the
        // reserved column is `pgs_ctn_z`, and the two strings differ.
        validate_marginal_slope_z_column_exclusion(
            &main,
            &parse_formula("y ~ 1").expect("parse slope"),
            "pgs_ctn_z",
            "bernoulli marginal-slope",
            "slope_formula",
        )
        .expect("literal-name validator cannot see the alias");

        let err = validate_marginal_slope_z_alias_exclusion(
            &main,
            &cols,
            "pgs_ctn_z",
            "bernoulli marginal-slope",
        )
        .expect_err("bare `z` resolves to the reserved score column");
        assert!(err.contains("reserves z column 'pgs_ctn_z'"), "{err}");
        assert!(err.contains("cannot also appear in the main formula"), "{err}");
    }

    /// Negative controls: the alias guard must not reject anything else.
    #[test]
    fn marginal_slope_alias_validator_accepts_legitimate_formulas() {
        // A baseline that never mentions the score, under either spelling.
        validate_marginal_slope_z_alias_exclusion(
            &parse_formula("y ~ x + w").expect("parse clean main"),
            &frame(&["y", "x", "w", "pgs_ctn_z"]),
            "pgs_ctn_z",
            "bernoulli marginal-slope",
        )
        .expect("a baseline without the score is legitimate");

        // A frame carrying its OWN `z` column keeps it: `column_map_with_alias`
        // inserts with `or_insert`, so the alias is inert and `z` in the main
        // formula means that column, not the score.
        validate_marginal_slope_z_alias_exclusion(
            &parse_formula("y ~ x + z").expect("parse main with real z"),
            &frame(&["y", "x", "z", "pgs_ctn_z"]),
            "pgs_ctn_z",
            "bernoulli marginal-slope",
        )
        .expect("a real `z` column is not the alias");

        // z_column == "z": the literal-name validator already covers this
        // verbatim, so the alias guard must stay out of its way.
        validate_marginal_slope_z_alias_exclusion(
            &parse_formula("y ~ x + z").expect("parse main"),
            &frame(&["y", "x", "z"]),
            "z",
            "bernoulli marginal-slope",
        )
        .expect("alias guard defers to the literal-name validator when they coincide");
    }

    /// The alias guard is an addition, not a replacement: spelling the reserved
    /// column out in full is still rejected, by the original validator.
    #[test]
    fn marginal_slope_literal_z_column_is_still_rejected() {
        let err = validate_marginal_slope_z_column_exclusion(
            &parse_formula("y ~ x + pgs_ctn_z").expect("parse main"),
            &parse_formula("y ~ 1").expect("parse slope"),
            "pgs_ctn_z",
            "bernoulli marginal-slope",
            "slope_formula",
        )
        .expect_err("the reserved column named in full must still be rejected");
        assert!(err.contains("reserves z column 'pgs_ctn_z'"), "{err}");
    }

    #[test]
    fn slope_surface_declarations_are_additive() {
        let parsed = parse_formula("y ~ s(pc1) + slope(z2, s(pc2)) + slope(z3, x3)")
            .expect("parse additive slope surfaces");
        assert_eq!(parsed.terms.len(), 1);
        assert_eq!(parsed.slope_surfaces.len(), 2);
        assert_eq!(parsed.slope_surfaces[0].z_column, "z2");
        assert_eq!(parsed.slope_surfaces[0].terms.len(), 1);
        assert_eq!(parsed.slope_surfaces[1].z_column, "z3");
        assert_eq!(parsed.slope_surfaces[1].terms.len(), 1);
    }

    #[test]
    fn marginal_slope_z_column_validator_reserves_all_surface_z_columns() {
        let main = parse_formula("y ~ x").expect("parse main");
        let slope = parse_formula("y ~ s(pc1) + slope(z2, s(z3)) + slope(z3, x)")
            .expect("parse slope surfaces");
        let err = validate_marginal_slope_z_column_exclusion(
            &main,
            &slope,
            "z1",
            "bernoulli marginal-slope",
            "--slope-formula",
        )
        .expect_err("surface formula should reject another reserved z coordinate");
        assert!(err.contains("reserves z column 'z3'"));
    }

    /// Extract the single `RandomEffect` term's unseen-level policy from a
    /// one-term formula, panicking if the term is not a random-effect block.
    fn random_effect_lenient_unseen(formula: &str) -> bool {
        let parsed = parse_formula(formula).expect("parse random-effect formula");
        let re = parsed.terms.iter().find_map(|t| match t {
            ParsedTerm::RandomEffect { lenient_unseen, .. } => Some(*lenient_unseen),
            _ => None,
        });
        re.unwrap_or_else(|| panic!("{formula} did not lower to a RandomEffect term"))
    }

    #[test]
    fn factor_wrapper_is_strict_on_unseen_levels_while_group_re_are_lenient() {
        // Regression for #2137 (sibling of #2102): `factor(g)` names the
        // categorical level effect of a column seen in training, so an out-of-vocabulary
        // level at predict is a schema mismatch that must raise — NOT be shrunk to
        // the centering point. `group(g)`/`s(g, bs="re")` are genuine
        // random effects that tolerate a held-out group (→ population mean). The
        // parse arm once hardcoded `lenient_unseen: true` for all four wrappers,
        // so `factor(g)` silently averaged an unseen level. Pin the per-wrapper
        // policy at the parse layer, where the whole distinction now lives.
        assert!(
            !random_effect_lenient_unseen("y ~ factor(g)"),
            "factor(g) is strict (lenient_unseen=false) on unseen levels"
        );
        for lenient in ["y ~ group(g)", "y ~ s(g, bs=re)"] {
            assert!(
                random_effect_lenient_unseen(lenient),
                "{lenient} is a genuine random effect: lenient (lenient_unseen=true) on unseen levels"
            );
        }
    }

    #[test]
    fn categorical_wrappers_reject_unknown_options() {
        // pyGAM audit F3: `factor()`/`group()` accept no options, so a
        // stray keyword must be a typed parse error instead of being dropped.
        for formula in [
            "y ~ factor(g, foo=1)",
            "y ~ factor(g, double_penalty=false)",
            "y ~ group(g, bogus=3)",
            "y ~ group(g, k=4)",
        ] {
            let err = match parse_formula(formula) {
                Ok(_) => panic!("{formula} must reject its unknown option"),
                Err(err) => err.to_string(),
            };
            assert!(
                err.contains("does not accept option"),
                "{formula}: unexpected error {err}"
            );
        }
    }

    #[test]
    fn formula_response_column_names_only_a_plain_response() {
        // pyffi and the sklearn metadata read the response column from this
        // owner, so it must equal the CLI's `parse_formula(..).response` for a
        // plain column and refuse survival responses naming several columns.
        let formula = "y_obs ~ s(x) + z";
        let parsed = parse_formula(formula).expect("plain formula parses");
        assert_eq!(formula_response_column(formula), Some(parsed.response));
        for survival in [
            "Surv(entry, exit, event) ~ s(x)",
            "surv(exit, event) ~ s(x)",
            "SurvInterval(left, right, event) ~ s(x)",
        ] {
            assert_eq!(formula_response_column(survival), None, "{survival}");
        }
        assert_eq!(formula_response_column("s(x) + z"), None);
    }

    fn has_no_intercept_marker(parsed: &super::ParsedFormula) -> bool {
        parsed
            .terms
            .iter()
            .any(|t| matches!(t, ParsedTerm::NoIntercept))
    }

    #[test]
    fn intercept_removal_spellings_all_lower_to_the_no_intercept_marker() {
        // R/patsy spellings of "no intercept": a leading `0 +`, a trailing or
        // standalone `- 1`, and `-1 +`. Each must drop the token from the term
        // list and append exactly one `NoIntercept` marker; the remaining terms
        // must be exactly what the formula without the token lowers to.
        for (formula, plain) in [
            ("y ~ 0 + x", "y ~ x"),
            ("y ~ x - 1", "y ~ x"),
            ("y ~ x + z - 1", "y ~ x + z"),
            ("y ~ -1 + x", "y ~ x"),
            ("y ~ x + 0", "y ~ x"),
            ("y ~ 0 + s(x) + z", "y ~ s(x) + z"),
            ("y ~ a*b - 1", "y ~ a*b"),
            ("y ~ s(x, k=5) - 1", "y ~ s(x, k=5)"),
        ] {
            let removed = parse_formula(formula).unwrap_or_else(|e| panic!("{formula}: {e}"));
            let kept = parse_formula(plain).unwrap_or_else(|e| panic!("{plain}: {e}"));
            assert!(has_no_intercept_marker(&removed), "{formula} lost its marker");
            assert!(!has_no_intercept_marker(&kept), "{plain} gained a marker");
            let markers = removed
                .terms
                .iter()
                .filter(|t| matches!(t, ParsedTerm::NoIntercept))
                .count();
            assert_eq!(markers, 1, "{formula} must carry exactly one marker");
            let without_marker: Vec<String> = removed
                .terms
                .iter()
                .filter(|t| !matches!(t, ParsedTerm::NoIntercept))
                .map(|t| format!("{t:?}"))
                .collect();
            let plain_terms: Vec<String> = kept.terms.iter().map(|t| format!("{t:?}")).collect();
            assert_eq!(without_marker, plain_terms, "{formula} vs {plain}");
        }
        // An explicit `1` is the default and adds no marker.
        let explicit = parse_formula("y ~ 1 + x").expect("explicit intercept parses");
        assert!(!has_no_intercept_marker(&explicit));
    }

    #[test]
    fn intercept_removal_rejects_contradictions_and_degenerate_uses() {
        let cases: [(&str, &str); 4] = [
            ("y ~ 1 + x - 1", "both includes the intercept"),
            ("y ~ 0", "no other term"),
            ("y ~ x:0", "cannot take part in an interaction"),
            ("y ~ 0 + 1 + x", "both includes the intercept"),
        ];
        for (formula, needle) in cases {
            let err = parse_formula(formula)
                .expect_err(formula)
                .to_string();
            assert!(err.contains(needle), "{formula}: expected `{needle}` in `{err}`");
        }
    }

    #[test]
    fn intercept_removal_is_rejected_in_auxiliary_formulas() {
        let parsed = parse_formula("y ~ 0 + x").expect("parses");
        let err = super::validate_auxiliary_formula_controls(&parsed, "--predict-noise")
            .expect_err("auxiliary formulas cannot drop the intercept");
        assert!(err.contains("only supported in the main formula"), "{err}");
        assert!(err.contains("--predict-noise"), "{err}");
    }

    #[test]
    fn backtick_quoted_columns_name_non_identifier_columns_verbatim() {
        // Column names with spaces, dots, a leading digit or non-ASCII letters
        // are written between backticks; everything inside is the name.
        let parsed = parse_formula(
            "`log y` ~ s(`my col`) + `x.1` + te(`2nd`, `température`) \
             + s(`a b`, by=`grp id`) + linear(`z z`)",
        )
        .expect("backtick formula parses");
        assert_eq!(parsed.response, "log y");
        let mut cols = BTreeSet::<String>::new();
        parsed_term_column_names(&parsed.terms, &mut cols);
        for expected in ["my col", "x.1", "2nd", "température", "a b", "grp id", "z z"] {
            assert!(cols.contains(expected), "missing `{expected}` in {cols:?}");
        }
        assert!(
            cols.iter().all(|c| !c.contains('`')),
            "backticks leaked into column names: {cols:?}"
        );
        match &parsed.terms[0] {
            ParsedTerm::Smooth { label, vars, .. } => {
                assert_eq!(vars, &vec!["my col".to_string()]);
                assert!(label.contains("`my col`"), "label keeps source text: {label}");
            }
            other => panic!("expected smooth, got {other:?}"),
        }
        match &parsed.terms[1] {
            ParsedTerm::Linear { name, .. } => assert_eq!(name, "x.1"),
            other => panic!("expected linear, got {other:?}"),
        }
    }

    #[test]
    fn backtick_columns_take_part_in_interactions_and_crossing() {
        let labels = wr_term_labels("y ~ `a a`:b + `c.1`*d");
        // Interaction vars are sorted; the backtick column sorts by its bare name.
        assert_eq!(
            labels,
            vec![
                "a a:b".to_string(),
                "c.1".to_string(),
                "d".to_string(),
                "c.1:d".to_string()
            ]
        );
    }

    #[test]
    fn backtick_quoting_protects_operators_inside_the_name() {
        // `a:b` between backticks is ONE column, not an interaction; `x + 1`
        // is one column, not two terms.
        let parsed = parse_formula("y ~ `a:b` + `x + 1`").expect("parses");
        let names: Vec<String> = parsed
            .terms
            .iter()
            .map(|t| match t {
                ParsedTerm::Linear { name, .. } => name.clone(),
                other => format!("Other({other:?})"),
            })
            .collect();
        assert_eq!(names, vec!["a:b".to_string(), "x + 1".to_string()]);
    }

    #[test]
    fn backtick_columns_work_in_survival_responses() {
        let parsed =
            parse_formula("Surv(`entry t`, `exit t`, `event flag`) ~ s(x)").expect("parses");
        let mut cols = BTreeSet::<String>::new();
        parsed_term_column_names(&parsed.terms, &mut cols);
        assert!(cols.contains("x"));
        let (entry, exit, event) = super::parse_surv_response(&parsed.response)
            .expect("Surv response parses")
            .expect("response is a Surv triple");
        assert_eq!(
            (entry.as_deref(), exit.as_str(), event.as_str()),
            (Some("entry t"), "exit t", "event flag")
        );
    }

    #[test]
    fn empty_or_unterminated_backticks_are_rejected() {
        for formula in ["y ~ s(``)", "y ~ `x", "y ~ s(`x)"] {
            assert!(parse_formula(formula).is_err(), "{formula} must not parse");
        }
    }

    #[test]
    fn capital_c_is_refused_with_a_pointer_to_factor() {
        // `factor(g)` is the only categorical level-effect spelling; `C(g)`
        // names nothing else, so it is an error that says what to write.
        let err = parse_formula("y ~ C(g) + x").expect_err("C() is not a term");
        let err = err.to_string();
        assert!(err.contains("`C()` is not a term function"), "{err}");
        assert!(err.contains("factor(g)"), "{err}");
        // Lowercase `c()` only appears inside option values (`k=c(5, 5)`).
        let err = parse_formula("y ~ c(g)").expect_err("c() is not a term");
        assert!(err.to_string().contains("unknown term function"), "{err}");
    }
}

// ---------------------------------------------------------------------------
// Higher-level formula parsing: ParsedFormula, ParsedTerm, and friends
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LinkWiggleFormulaSpec {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
}

pub fn default_linkwiggle_formulaspec() -> LinkWiggleFormulaSpec {
    let cfg = WigglePenaltyConfig::cubic_triple_operator_default();
    LinkWiggleFormulaSpec {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    }
}

#[derive(Clone, Debug)]
pub struct LinkFormulaSpec {
    pub link: String,
    pub mixture_rho: Option<String>,
    pub sas_init: Option<String>,
    pub beta_logistic_init: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SurvivalFormulaSpec {
    pub spec: Option<String>,
    pub survival_distribution: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ParsedFormula {
    pub response: String,
    pub terms: Vec<ParsedTerm>,
    pub slope_surfaces: Vec<SlopeSurfaceSpec>,
    pub linkwiggle: Option<LinkWiggleFormulaSpec>,
    pub timewiggle: Option<LinkWiggleFormulaSpec>,
    pub linkspec: Option<LinkFormulaSpec>,
    pub survivalspec: Option<SurvivalFormulaSpec>,
}

#[derive(Clone, Debug)]
pub struct SlopeSurfaceSpec {
    pub z_column: String,
    pub terms: Vec<ParsedTerm>,
}

pub fn marginal_slope_surfaces(
    slope_formula: &ParsedFormula,
    default_z_column: &str,
) -> Result<Vec<SlopeSurfaceSpec>, String> {
    let mut surfaces = Vec::new();
    if !slope_formula.terms.is_empty() {
        surfaces.push(SlopeSurfaceSpec {
            z_column: default_z_column.to_string(),
            terms: slope_formula.terms.clone(),
        });
    }
    surfaces.extend(slope_formula.slope_surfaces.clone());
    if surfaces.is_empty() {
        surfaces.push(SlopeSurfaceSpec {
            z_column: default_z_column.to_string(),
            terms: Vec::new(),
        });
    }
    let mut seen = std::collections::BTreeSet::<String>::new();
    for surface in &surfaces {
        if !seen.insert(surface.z_column.clone()) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "slope formula declares z column '{}' more than once; each z coordinate needs exactly one slope surface",
                    surface.z_column
                ),
            }
            .into());
        }
    }
    Ok(surfaces)
}

#[derive(Clone, Debug)]
pub enum ParsedTerm {
    Linear {
        name: String,
        explicit: bool,
        double_penalty: bool,
        coefficient_min: Option<f64>,
        coefficient_max: Option<f64>,
    },
    BoundedLinear {
        name: String,
        min: f64,
        max: f64,
        prior: BoundedCoefficientPriorSpec,
        double_penalty: bool,
    },
    RandomEffect {
        name: String,
        /// Unseen-level policy, fixed at parse time by the wrapper the user
        /// wrote. `group(g)`/`s(g, bs="re")` are genuine **random
        /// effects**: a held-out group is shrunk to the population mean, so an
        /// unseen level at predict is tolerated (`true`). `factor(g)` names a
        /// categorical level effect: like a bare `+ g` categorical main
        /// effect, an unseen
        /// level is a schema mismatch that must raise rather than collapse onto
        /// the factor's centering point (`false`, #2137/#2102). Both wrappers
        /// share the penalized-categorical materialization; only this policy
        /// distinguishes them, so seen-level fits are identical.
        lenient_unseen: bool,
    },
    Smooth {
        label: String,
        vars: Vec<String>,
        kind: SmoothKind,
        options: BTreeMap<String, String>,
    },
    LinkWiggle {
        options: BTreeMap<String, String>,
    },
    TimeWiggle {
        options: BTreeMap<String, String>,
    },
    LinkConfig {
        options: BTreeMap<String, String>,
    },
    SurvivalConfig {
        options: BTreeMap<String, String>,
    },
    SlopeSurface {
        z_column: String,
        terms: Vec<ParsedTerm>,
    },
    /// Wilkinson-Rogers interaction term `a:b[:c...]`.
    ///
    /// `vars` is a sorted, deduplicated list of base column names. Each element
    /// must be a bare identifier — interactions with function-call atoms
    /// (smooths, factors, etc.) are rejected upstream because their design
    /// columns are not simple products. The design column is the elementwise
    /// product of the referenced numeric columns.
    Interaction {
        vars: Vec<String>,
        double_penalty: bool,
    },
    /// Model-level marker for `0 + ...` / `... - 1`: the formula removes the
    /// global intercept. It consumes no column and builds no design block;
    /// `term_builder` turns it into [`crate::smooth::ModelLevel::NoIntercept`]
    /// unless a remaining term spans the constant, in which case the intercept
    /// stays (see `docs/formulas.md`, "Removing the intercept").
    NoIntercept,
}

/// Collect the names of every data column the parsed terms consume.
///
/// This is the canonical formula→columns walk shared by the fit-time and
/// predict-time required-column computations (the CLI and PyFFI surfaces both
/// route through it). It includes a smooth's positional `vars` *and* its `by=`
/// grouping/scaling column (`s(x, by=g)`), which `term_builder` reads from
/// `options["by"]` but which is not among the positional variables — omitting
/// it would drop a genuine predictor from the model's input contract.
pub fn parsed_term_column_names(
    terms: &[ParsedTerm],
    out: &mut std::collections::BTreeSet<String>,
) {
    for term in terms {
        match term {
            ParsedTerm::Linear { name, .. }
            | ParsedTerm::BoundedLinear { name, .. }
            | ParsedTerm::RandomEffect { name, .. } => {
                out.insert(name.clone());
            }
            ParsedTerm::Smooth { vars, options, .. } => {
                out.extend(vars.iter().cloned());
                if let Some(by) = options.get("by") {
                    out.insert(by.clone());
                }
            }
            ParsedTerm::Interaction { vars, .. } => {
                out.extend(vars.iter().cloned());
            }
            ParsedTerm::LinkWiggle { .. }
            | ParsedTerm::TimeWiggle { .. }
            | ParsedTerm::LinkConfig { .. }
            | ParsedTerm::SurvivalConfig { .. }
            | ParsedTerm::NoIntercept => {}
            ParsedTerm::SlopeSurface { z_column, terms } => {
                out.insert(z_column.clone());
                parsed_term_column_names(terms, out);
            }
        }
    }
}

pub(crate) fn parsed_terms_reference_column(terms: &[ParsedTerm], column_name: &str) -> bool {
    terms.iter().any(|term| match term {
        ParsedTerm::Linear { name, .. }
        | ParsedTerm::BoundedLinear { name, .. }
        | ParsedTerm::RandomEffect { name, .. } => name == column_name,
        ParsedTerm::Smooth { vars, options, .. } => {
            vars.iter().any(|var| var == column_name)
                || options.get("by").is_some_and(|by| by == column_name)
        }
        ParsedTerm::Interaction { vars, .. } => vars.iter().any(|var| var == column_name),
        ParsedTerm::LinkWiggle { .. }
        | ParsedTerm::TimeWiggle { .. }
        | ParsedTerm::LinkConfig { .. }
        | ParsedTerm::SurvivalConfig { .. }
        | ParsedTerm::NoIntercept => false,
        ParsedTerm::SlopeSurface { z_column, terms } => {
            z_column == column_name || parsed_terms_reference_column(terms, column_name)
        }
    })
}

/// Reject the score entering the main formula through its canonical alias.
///
/// `validate_marginal_slope_z_column_exclusion` compares each parsed term's
/// literal name against the configured `z_column`, so it rejects
/// `disease ~ pgs_ctn_z` — and accepts `disease ~ z`, which
/// `column_map_with_alias(col_map, "z", z_column)` then resolves to *the same
/// column*. The score lands in the marginal design either way; only one
/// spelling is caught. That is not a cosmetic gap: with an intercept-only
/// slope the effective slope direction is `f = a*1 + s*z`, which lies
/// in the span of the effective marginal design `[1, z]` row by row for ANY
/// data, so the fit is degenerate by construction. It used to surface ~12 s
/// later as a `FullyConfounded` refusal from the BMS audit, a message that
/// reads like a solver failure rather than a formula mistake (gam#2432).
///
/// Checked against the frame rather than the formula text because the alias is
/// only live when the frame has no real `z` column of its own — see
/// `marginal_slope_z_alias_is_live`. A frame that genuinely carries `z` keeps
/// it, and writing `z` there is legitimate and still accepted.
pub fn validate_marginal_slope_z_alias_exclusion(
    main_formula: &ParsedFormula,
    col_map: &HashMap<String, usize>,
    z_column: &str,
    context: &str,
) -> Result<(), String> {
    if z_column == MARGINAL_SLOPE_Z_ALIAS {
        // Already covered verbatim by the literal-name exclusion.
        return Ok(());
    }
    if !marginal_slope_z_alias_is_live(col_map, z_column) {
        return Ok(());
    }
    if parsed_terms_reference_column(&main_formula.terms, MARGINAL_SLOPE_Z_ALIAS) {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "{context} reserves z column '{z_column}' as the auxiliary latent score, and the \
                 bare name '{MARGINAL_SLOPE_Z_ALIAS}' in the main formula resolves to it; it \
                 cannot also appear in the main formula. Give the baseline its own covariates \
                 (the score's effect is carried by the slope formula)."
            ),
        }
        .into());
    }
    Ok(())
}

pub fn validate_marginal_slope_z_column_exclusion(
    main_formula: &ParsedFormula,
    slope_formula: &ParsedFormula,
    z_column: &str,
    context: &str,
    slope_label: &str,
) -> Result<(), String> {
    let surfaces = marginal_slope_surfaces(slope_formula, z_column)?;
    // The CLI/configured z column is reserved even when the slope formula
    // is intercept-only (`~ 1`) and therefore contributes no surface terms.
    // Explicit slope(...) declarations may reserve additional z coordinates.
    let mut reserved_z_columns = std::collections::BTreeSet::<&str>::new();
    reserved_z_columns.insert(z_column);
    reserved_z_columns.extend(surfaces.iter().map(|surface| surface.z_column.as_str()));

    for reserved in &reserved_z_columns {
        if parsed_terms_reference_column(&main_formula.terms, reserved) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "{context} reserves z column '{reserved}' as the auxiliary latent score; it cannot also appear in the main formula"
                ),
            }
            .into());
        }
    }
    for reserved in &reserved_z_columns {
        if parsed_terms_reference_column(&slope_formula.terms, reserved) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "{context} reserves z column '{reserved}' as the auxiliary latent score; it cannot also appear in {slope_label}"
                ),
            }
            .into());
        }
        for surface in &surfaces {
            if parsed_terms_reference_column(&surface.terms, reserved) {
                return Err(FormulaDslError::IncompatibleTerm {
                    reason: format!(
                        "{context} reserves z column '{reserved}' as an auxiliary latent score; it cannot also appear in {slope_label}"
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub enum SmoothKind {
    S,
    Te,
    /// Tensor smooth (`t2(...)`) using mgcv's separable penalty decomposition:
    /// the tensor coefficient space is split into marginal penalized/null-space
    /// tensor subspaces, with one smoothing parameter per non-null subspace.
    T2,
    /// Tensor *interaction* smooth (`ti(...)`): a tensor-product smooth whose
    /// marginal main effects are excluded, so the term captures only the pure
    /// interaction between its variables. Materializes through the same tensor
    /// path as [`SmoothKind::Te`] but with per-margin sum-to-zero
    /// identifiability (`TensorBSplineIdentifiability::MarginalSumToZero`).
    Ti,
}

#[derive(Clone, Copy, Debug)]
pub enum LinkMode {
    Strict,
    Flexible,
}

#[derive(Clone, Debug)]
pub struct LinkChoice {
    pub mode: LinkMode,
    pub link: LinkFunction,
    pub mixture_components: Option<Vec<LinkComponent>>,
}

// ---------------------------------------------------------------------------
// Link wiggle / link choice helpers
// ---------------------------------------------------------------------------

pub fn effectivelinkwiggle_formulaspec(
    formula_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    link_choice: Option<&LinkChoice>,
) -> Option<LinkWiggleFormulaSpec> {
    formula_linkwiggle.cloned().or_else(|| {
        link_choice.and_then(|choice| {
            if matches!(choice.mode, LinkMode::Flexible) {
                Some(default_linkwiggle_formulaspec())
            } else {
                None
            }
        })
    })
}

pub(crate) const fn linkname_supports_joint_wiggle(link: LinkFunction) -> bool {
    !matches!(link, LinkFunction::Sas | LinkFunction::BetaLogistic)
}

pub(crate) const fn linkchoice_supports_joint_wiggle(choice: &LinkChoice) -> bool {
    match &choice.mixture_components {
        None => linkname_supports_joint_wiggle(choice.link),
        Some(_) => false,
    }
}

pub fn require_linkchoice_supports_joint_wiggle(
    choice: &LinkChoice,
    context: &str,
) -> Result<(), String> {
    if linkchoice_supports_joint_wiggle(choice) {
        Ok(())
    } else {
        Err(joint_wiggle_unsupported_link_message(context))
    }
}

pub(crate) const fn likelihood_spec_supports_joint_wiggle(likelihood: &LikelihoodSpec) -> bool {
    inverse_link_supports_joint_wiggle(&likelihood.link)
}

pub fn require_likelihood_spec_supports_joint_wiggle(
    likelihood: &LikelihoodSpec,
    context: &str,
) -> Result<(), String> {
    if likelihood_spec_supports_joint_wiggle(likelihood) {
        Ok(())
    } else {
        Err(joint_wiggle_unsupported_link_message(context))
    }
}

/// Family-agnostic capability of the joint link-wiggle machinery: which base
/// inverse links a monotone warp can be fit over AND reconstructed from at
/// predict time. Every state-less standard link qualifies — the warp fit and
/// its saved-model reconstruction evaluate the base inverse link purely through
/// the generic `inverse_link_jet_for_inverse_link` jet dispatch, which carries
/// LogLog and Cauchit exactly as it does Logit/Probit/CLogLog. LogLog/Cauchit
/// were previously omitted, so a binomial `flexible(loglog)`/`flexible(cauchit)`
/// fit that this-gate-agnostically *converged* (see
/// `binomial_inverse_link_supports_joint_wiggle`) then failed at predict when
/// `FittedModel::saved_link_wiggle` re-checked the saved link here (#2155). The
/// state-bearing links (SAS/BetaLogistic/Mixture/LatentCLogLog) carry fitted
/// warp/skew state of their own and are intentionally excluded.
pub const fn inverse_link_supports_joint_wiggle(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(StandardLink::Identity)
            | InverseLink::Standard(StandardLink::Log)
            | InverseLink::Standard(StandardLink::Logit)
            | InverseLink::Standard(StandardLink::Probit)
            | InverseLink::Standard(StandardLink::CLogLog)
            | InverseLink::Standard(StandardLink::LogLog)
            | InverseLink::Standard(StandardLink::Cauchit)
    )
}

pub fn require_inverse_link_supports_joint_wiggle(
    link: &InverseLink,
    context: &str,
) -> Result<(), String> {
    if inverse_link_supports_joint_wiggle(link) {
        Ok(())
    } else {
        Err(joint_wiggle_unsupported_link_message(context))
    }
}

/// Which binomial base links the joint link-wiggle (flexible-link) solver can
/// fit. All five standard binomial probability links qualify: the wiggle kernel
/// (`BinomialMeanWiggleFamily`) evaluates the base inverse link purely through
/// the generic `inverse_link_jet_for_inverse_link` dispatch, which carries full
/// jets for LogLog and Cauchit exactly as it does for Logit/Probit/CLogLog — so
/// `flexible(loglog)` / `flexible(cauchit)` fit through the same machinery
/// (#2155). Previously this gate listed only logit/probit/cloglog while the
/// permissive parse gate `linkname_supports_joint_wiggle` admitted loglog/cauchit,
/// so the config was accepted then aborted deep in the solver. The state-bearing
/// links (SAS/BetaLogistic/Mixture/LatentCLogLog) and identity/log stay out: the
/// warp is defined only over a fixed state-less base probability link.
pub(crate) const fn binomial_inverse_link_supports_joint_wiggle(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(StandardLink::Logit)
            | InverseLink::Standard(StandardLink::Probit)
            | InverseLink::Standard(StandardLink::CLogLog)
            | InverseLink::Standard(StandardLink::LogLog)
            | InverseLink::Standard(StandardLink::Cauchit)
    )
}

pub fn require_binomial_inverse_link_supports_joint_wiggle(
    link: &InverseLink,
    context: &str,
) -> Result<(), String> {
    if binomial_inverse_link_supports_joint_wiggle(link) {
        Ok(())
    } else {
        Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "{context} does not support identity, log, latent-cloglog, SAS, BetaLogistic, or Mixture links; wiggle is only available for jointly fitted standard binomial probability links (logit/probit/cloglog/loglog/cauchit)"
            ),
        }
        .into())
    }
}

pub fn joint_wiggle_unsupported_link_message(context: &str) -> String {
    format!(
        "{context} does not support latent-cloglog, SAS, BetaLogistic, or Mixture links; wiggle is only available for jointly fitted standard links"
    )
}

// ---------------------------------------------------------------------------
// Option-map helpers (shared by formula parsing and term construction)
// ---------------------------------------------------------------------------

/// Local sibling of `term_builder::validate_known_options` used by the
/// parser-side `linear / bounded / nonnegative / nonpositive`
/// branches (which build their `ParsedTerm` here and never enter
/// `term_builder::build_smooth_basis`). Without this, typos like
/// `bounded(x, min=0, max=1, foo=bar)` silently succeed because the
/// `foo` key was just never read.
/// The term functions that build a smooth; their option maps are checked
/// for removed option keys and values before dispatch.
const SMOOTH_TERM_FUNCTIONS: &[&str] = &[
    "smooth", "s", "cyclic", "thinplate", "thin_plate", "tps", "te", "t2", "ti", "fs", "sz",
    "sphere", "sos", "spherical", "s2", "mjs", "curv", "matern", "duchon", "pca",
];

/// A function form (`tps()`, `cyclic()`, `matern()`, ...) fixes its smooth
/// type through `bs=`. A `bs=` the user wrote on it may only repeat that type;
/// any other value names a different smooth and is refused instead of being
/// silently replaced.
fn fix_function_basis(
    options: &mut BTreeMap<String, String>,
    name: &str,
    basis: &str,
    raw: &str,
) -> Result<(), String> {
    if let Some(given) = options.get("bs") {
        let given = given
            .trim()
            .trim_matches(|c| c == '\'' || c == '"')
            .to_ascii_lowercase();
        if given != basis {
            return Err(format!(
                "{name}() is the `{basis}` smooth and cannot take bs=`{given}`; \
                 write s(..., bs=\"{given}\") for that basis: {raw}"
            ));
        }
    }
    options.insert("bs".to_string(), basis.to_string());
    Ok(())
}

fn validate_known_term_options(
    term_name: &str,
    options: &BTreeMap<String, String>,
    known: &[&str],
    raw: &str,
) -> Result<(), String> {
    let known_set: std::collections::BTreeSet<&&str> = known.iter().collect();
    for key in options.keys() {
        if !known_set.contains(&key.as_str()) {
            if let Some(canonical) =
                removed_spellings::canonical_for(removed_spellings::TERM_OPTION_KEYS, key)
                && known_set.contains(&canonical)
            {
                return Err(FormulaDslError::InvalidArgument {
                    reason: format!(
                        "unknown option `{key}` in {term_name}(); use `{canonical}`: {raw}"
                    ),
                }
                .into());
            }
            let known_sorted = {
                let mut v = known.to_vec();
                v.sort_unstable();
                v.join(", ")
            };
            let known_hint = if known.is_empty() {
                "no options".to_string()
            } else {
                format!("[{known_sorted}]")
            };
            return Err(FormulaDslError::InvalidArgument {
                reason: format!(
                    "{term_name}() does not accept option `{key}` (in `{raw}`); known options: {known_hint}"
                ),
            }
            .into());
        }
    }
    Ok(())
}

/// Integer option: returns `Ok(None)` if not present, `Ok(Some(n))` if it
/// parses as a non-negative integer, and `Err(msg)` if the user supplied a
/// value that isn't a valid usize (negative, decimal, garbage). There is
/// deliberately no lenient reader: dropping an invalid value would revert to
/// the default, so `k=-1` or `degree=cubic` would fit as "not specified"
/// instead of being flagged as user mistakes.
pub(crate) fn option_usize(
    map: &BTreeMap<String, String>,
    key: &str,
) -> Result<Option<usize>, String> {
    match map.get(key) {
        None => Ok(None),
        Some(raw) => raw.parse::<usize>().map(Some).map_err(|err| {
            FormulaDslError::InvalidArgument {
                reason: format!(
                    "option `{key}={raw}` is not a non-negative integer; \
                     expected a whole number >= 0: {err}"
                ),
            }
            .into()
        }),
    }
}

/// [`option_usize`] over several alias spellings; errors on the first
/// present-but-unparseable key rather than silently falling through.
pub(crate) fn option_usize_any(
    map: &BTreeMap<String, String>,
    keys: &[&str],
) -> Result<Option<usize>, String> {
    for key in keys {
        if let Some(v) = option_usize(map, key)? {
            return Ok(Some(v));
        }
    }
    Ok(None)
}

/// Float option: `Ok(None)` if absent, `Ok(Some(n))` if parses as a finite
/// f64, `Err` if the user passed an unparseable or non-finite value.
pub(crate) fn option_f64(map: &BTreeMap<String, String>, key: &str) -> Result<Option<f64>, String> {
    match map.get(key) {
        None => Ok(None),
        Some(raw) => match raw.parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(Some(v)),
            Ok(v) => Err(FormulaDslError::InvalidArgument {
                reason: format!("option `{key}={raw}` parses as {v} which is not a finite number"),
            }
            .into()),
            Err(err) => Err(FormulaDslError::InvalidArgument {
                reason: format!(
                    "option `{key}={raw}` is not a valid number; expected a finite decimal: {err}"
                ),
            }
            .into()),
        },
    }
}

/// Boolean option: `Ok(None)` if absent, `Ok(Some(b))` for a recognized
/// truthy/falsy token, and `Err(msg)` for a present-but-unparseable value, so
/// a typo like `double_penalty=ture` is never treated as "not specified".
pub(crate) fn option_bool(
    map: &BTreeMap<String, String>,
    key: &str,
) -> Result<Option<bool>, String> {
    match map.get(key) {
        None => Ok(None),
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => Ok(Some(true)),
            "false" | "0" | "no" | "n" => Ok(Some(false)),
            _ => Err(FormulaDslError::InvalidArgument {
                reason: format!(
                    "option `{key}={raw}` is not a boolean; \
                     expected one of true/false/yes/no/1/0"
                ),
            }
            .into()),
        },
    }
}

pub(crate) fn strip_quotes(v: &str) -> &str {
    let b = v.as_bytes();
    if b.len() >= 2
        && ((b[0] == b'\'' && b[b.len() - 1] == b'\'') || (b[0] == b'"' && b[b.len() - 1] == b'"'))
    {
        &v[1..v.len() - 1]
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// Sub-parsers for formula option blocks
// ---------------------------------------------------------------------------

fn parse_linear_constraint_bounds(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<(Option<f64>, Option<f64>), String> {
    let min = parse_optional_labeled_f64_option(options, "min", raw, "linear")?;
    let max = parse_optional_labeled_f64_option(options, "max", raw, "linear")?;
    if let (Some(min), Some(max)) = (min, max)
        && (!min.is_finite() || !max.is_finite() || min > max)
    {
        return Err(FormulaDslError::InvalidArgument {
            reason: format!(
                "linear coefficient constraints require finite min <= max, got min={min}, max={max}: {raw}"
            ),
        }
        .into());
    }
    Ok((min, max))
}

fn parse_required_f64_option(
    options: &BTreeMap<String, String>,
    key: &str,
    raw: &str,
) -> Result<f64, String> {
    let value = options
        .get(key)
        .ok_or_else(|| FormulaDslError::MalformedConfig {
            reason: format!("bounded() is missing required '{key}' argument: {raw}"),
        })?;
    value.parse::<f64>().map_err(|err| {
        FormulaDslError::InvalidArgument {
            reason: format!(
                "bounded() argument '{key}' must be a finite number, got '{}': {err}: {raw}",
                value
            ),
        }
        .into()
    })
}

fn parse_optional_f64_option(
    options: &BTreeMap<String, String>,
    key: &str,
    raw: &str,
) -> Result<Option<f64>, String> {
    match options.get(key) {
        Some(value) => value.parse::<f64>().map(Some).map_err(|err| {
            FormulaDslError::InvalidArgument {
                reason: format!(
                    "bounded() argument '{key}' must be a finite number, got '{}': {err}: {raw}",
                    value
                ),
            }
            .into()
        }),
        None => Ok(None),
    }
}

fn parse_optional_labeled_f64_option(
    options: &BTreeMap<String, String>,
    key: &str,
    raw: &str,
    fn_label: &str,
) -> Result<Option<f64>, String> {
    match options.get(key) {
        Some(value) => value.parse::<f64>().map(Some).map_err(|err| {
            FormulaDslError::InvalidArgument {
                reason: format!(
                    "{fn_label}() argument '{key}' must be a finite number, got '{value}': {err}: {raw}"
                ),
            }
            .into()
        }),
        None => Ok(None),
    }
}

fn parse_linkwiggle_penalty_orders(raw: Option<&str>) -> Result<Vec<usize>, String> {
    let Some(raw) = raw.map(str::trim) else {
        return Ok(WigglePenaltyConfig::cubic_triple_operator_default().penalty_orders);
    };
    if raw.is_empty() {
        return Ok(WigglePenaltyConfig::cubic_triple_operator_default().penalty_orders);
    }
    let mut out = Vec::<usize>::new();
    for token in raw.split(',') {
        let t = token.trim().to_ascii_lowercase();
        if t.is_empty() {
            continue;
        }
        match t.as_str() {
            "all" => {
                out.extend([1, 2, 3]);
            }
            "slope" | "1" => out.push(1),
            "curvature" | "2" => out.push(2),
            "curvature-change" | "curvature_change" | "3" => out.push(3),
            _ => {
                return Err(FormulaDslError::InvalidArgument {
                    reason: format!(
                        "invalid linkwiggle penalty_order '{t}'; use all|slope|curvature|curvature-change or 1/2/3"
                    ),
                }
                .into());
            }
        }
    }
    if out.is_empty() {
        out.extend(WigglePenaltyConfig::cubic_triple_operator_default().penalty_orders);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

pub fn parse_linkwiggle_formulaspec(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<LinkWiggleFormulaSpec, String> {
    let allowed = [
        "degree",
        "internal_knots",
        "penalty_order",
        "double_penalty",
    ];
    let unknown = options
        .keys()
        .filter(|key| !allowed.contains(&key.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let term_name = raw.split('(').next().unwrap_or("linkwiggle");
    if !unknown.is_empty() {
        return Err(FormulaDslError::InvalidArgument {
            reason: format!(
                "{}() does not support option(s) {}: {raw}",
                term_name,
                unknown.join(", ")
            ),
        }
        .into());
    }
    let defaults = WigglePenaltyConfig::cubic_triple_operator_default();
    // Strict parsing: a present-but-unparseable value (`degree=abc`, `=-3`,
    // `=6.5`) must be rejected, not silently dropped and replaced by the
    // default.
    //
    // This parser is shared by *all* wiggle grammars: `linkwiggle` and
    // `timewiggle` (see `parse_formula`), the standard-model flexible-link
    // wiggle, and the marginal-slope score-warp / link-deviation routing.
    // The general monotone I-spline value basis (`monotone_wiggle_*` in
    // `families::gamlss`, used by `timewiggle` and the location-scale survival
    // path) honors arbitrary `degree >= 2`, while only the cubic-only
    // score-warp / link-deviation `DeviationRuntime` is restricted to 3.
    // A consumer-specific limit therefore must NOT be baked into this shared
    // parser — it is enforced at the routing layer that feeds the cubic-only
    // runtime (`deviation_block_config_from_formula_linkwiggle`). Here we only
    // enforce the universal lower bound that a polynomial degree is positive.
    let degree = option_usize(options, "degree")?.unwrap_or(defaults.degree);
    if degree < 1 {
        return Err(FormulaDslError::InvalidArgument {
            reason: format!("{term_name}() requires degree >= 1: {raw}"),
        }
        .into());
    }
    let num_internal_knots =
        option_usize(options, "internal_knots")?.unwrap_or(defaults.num_internal_knots);
    if num_internal_knots == 0 {
        return Err(FormulaDslError::InvalidArgument {
            reason: format!("{term_name}() requires internal_knots > 0: {raw}"),
        }
        .into());
    }
    // A DEFAULTED ladder adapts to the degree; an EXPLICIT one never does.
    //
    // `parse_linkwiggle_penalty_orders` falls back to
    // `WigglePenaltyConfig::cubic_triple_operator_default()` -- orders [1, 2, 3],
    // chosen for a CUBIC basis. A user writing `linkwiggle(degree=2, ...)`
    // overrides the degree and not the ladder, so the default asks for an
    // order-3 roughness penalty on a degree-2 basis. That penalty is the zero
    // matrix (the 3rd derivative of a degree-2 piecewise polynomial vanishes),
    // `bspline_unit_energy_factor` rightly refuses to build it, and the whole
    // fit dies with "Spline degree 2 is too low for derivative order 3".
    //
    // The user never asked for order 3 here, so dropping it is resolving a
    // default rather than overriding a request. When the ladder IS explicit the
    // orders stay exactly as written and the downstream refusal stands --
    // `wiggle::tests::unsupported_derivative_order_is_rejected_not_clamped`
    // pins that, and it is the reason this clip lives here at the
    // default-resolution seam instead of inside
    // `canonical_wiggle_function_penalties`, where it would have silently
    // clamped explicit requests too.
    let penalty_order_requested = options
        .get("penalty_order")
        .is_some_and(|raw| !raw.trim().is_empty());
    let mut penalty_orders =
        parse_linkwiggle_penalty_orders(options.get("penalty_order").map(String::as_str))?;
    if !penalty_order_requested {
        let supported: Vec<usize> = penalty_orders
            .iter()
            .copied()
            .filter(|&order| order <= degree)
            .collect();
        if supported.is_empty() {
            return Err(FormulaDslError::InvalidArgument {
                reason: format!(
                    "{term_name}(degree={degree}) supports no default penalty order: the default                      ladder {penalty_orders:?} contains no derivative order a degree-{degree}                      basis can penalize"
                ),
            }
            .into());
        }
        penalty_orders = supported;
    }
    let double_penalty =
        option_bool(options, "double_penalty")?.unwrap_or(defaults.double_penalty);
    Ok(LinkWiggleFormulaSpec {
        degree,
        num_internal_knots,
        penalty_orders,
        double_penalty,
    })
}

fn parse_link_formulaspec(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<LinkFormulaSpec, String> {
    let link = options
        .get("type")
        .map(|s| s.trim().to_string())
        .ok_or_else(|| FormulaDslError::MalformedConfig {
            reason: format!("link() requires type=<link-name>: {raw}"),
        })?;
    if link.is_empty() {
        return Err(FormulaDslError::MalformedConfig {
            reason: format!("link() requires a non-empty type: {raw}"),
        }
        .into());
    }
    let mixture_rho = options.get("rho").map(|s| s.trim().to_string());
    let sas_init = options.get("sas_init").map(|s| s.trim().to_string());
    let beta_logistic_init = options
        .get("beta_logistic_init")
        .map(|s| s.trim().to_string());
    Ok(LinkFormulaSpec {
        link,
        mixture_rho,
        sas_init,
        beta_logistic_init,
    })
}

fn parse_survival_formulaspec(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<SurvivalFormulaSpec, String> {
    if options.is_empty() {
        return Err(FormulaDslError::MalformedConfig {
            reason: format!(
                "survmodel() requires at least one named option (e.g., spec=..., distribution=...): {raw}"
            ),
        }
        .into());
    }
    Ok(SurvivalFormulaSpec {
        spec: options.get("spec").map(|s| s.trim().to_string()),
        survival_distribution: options.get("distribution").map(|s| s.trim().to_string()),
    })
}

/// What a `bounded()` prior option asks for.
enum BoundedPriorChoice {
    /// A prior carried on the latent logit coordinate of the interval map.
    Latent(BoundedCoefficientPriorSpec),
    /// `prior=uniform`: flat on the coefficient over `[min, max]`. That is the
    /// box-constrained, unpenalised linear coefficient, so the term lowers to
    /// `linear(x, min, max, double_penalty=false)` and publishes that route's
    /// truncated posterior mean (#2705) instead of a logit-chart mode (#3479).
    FlatOnBox,
}

fn parse_bounded_priorspec(
    options: &BTreeMap<String, String>,
    min: f64,
    max: f64,
    raw: &str,
) -> Result<BoundedPriorChoice, String> {
    let prior_mode = options.get("prior").map(|s| s.to_ascii_lowercase());
    let target = parse_optional_f64_option(options, "target", raw)?;
    let strength = parse_optional_f64_option(options, "strength", raw)?;

    let target_mode = target.is_some() || strength.is_some();
    if prior_mode.is_some() && target_mode {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!("bounded() cannot combine prior=... with target/strength: {raw}"),
        }
        .into());
    }

    if let Some(priorname) = prior_mode {
        return match priorname.as_str() {
            "shrinkage" => Ok(BoundedPriorChoice::Latent(BoundedCoefficientPriorSpec::Shrinkage)),
            "uniform" => Ok(BoundedPriorChoice::FlatOnBox),
            "center" => Ok(BoundedPriorChoice::Latent(BoundedCoefficientPriorSpec::Beta {
                a: 2.0,
                b: 2.0,
            })),
            other => Err(FormulaDslError::InvalidArgument {
                reason: match removed_spellings::canonical_for(
                    removed_spellings::BOUNDED_PRIORS,
                    other,
                ) {
                    Some(canonical) => format!(
                        "unknown bounded() prior `{other}`; use `{canonical}`: {raw}"
                    ),
                    None => format!(
                        "bounded() prior must be one of shrinkage|uniform|center, got '{other}': {raw}"
                    ),
                },
            }
            .into()),
        };
    }

    if target_mode {
        let targetvalue = target.ok_or_else(|| FormulaDslError::MalformedConfig {
            reason: format!("bounded() target is required with strength: {raw}"),
        })?;
        let strengthvalue = strength.ok_or_else(|| FormulaDslError::MalformedConfig {
            reason: format!("bounded() strength is required with target: {raw}"),
        })?;
        if !(min < targetvalue && targetvalue < max) {
            return Err(FormulaDslError::InvalidArgument {
                reason: format!("bounded() target must lie strictly inside ({min}, {max}): {raw}"),
            }
            .into());
        }
        if !strengthvalue.is_finite() || strengthvalue <= 0.0 {
            return Err(FormulaDslError::InvalidArgument {
                reason: format!("bounded() strength must be finite and > 0: {raw}"),
            }
            .into());
        }
        let z = (targetvalue - min) / (max - min);
        let a = 1.0 + strengthvalue * z;
        let b = 1.0 + strengthvalue * (1.0 - z);
        return Ok(BoundedPriorChoice::Latent(BoundedCoefficientPriorSpec::Beta { a, b }));
    }

    // No prior option: shrink toward the null with a REML-estimated strength.
    Ok(BoundedPriorChoice::Latent(BoundedCoefficientPriorSpec::Shrinkage))
}

// ---------------------------------------------------------------------------
// Top-level formula and term parsers
// ---------------------------------------------------------------------------

pub fn formula_rhs_text(formula: &str) -> Result<String, String> {
    let parsed = parse_formula_dsl(formula)?;
    if parsed.rhs_terms.is_empty() {
        return Err(FormulaDslError::ParseError {
            reason: "formula right-hand side cannot be empty".to_string(),
        }
        .into());
    }
    Ok(parsed.rhs_terms.join(" + "))
}

/// Parsed Surv(...) response specification.
///
/// `entry` is `None` for the 2-arg right-censored shorthand
/// `Surv(time, event)`, which matches the R survival/mgcv default: every
/// subject has entry time zero. Callers materialize a zero entry column
/// when this is `None`.
pub fn parse_surv_response(
    lhs: &str,
) -> Result<Option<(Option<String>, String, String)>, FormulaDslError> {
    let trimmed = lhs.trim();
    let call = match parse_function_call(trimmed) {
        Ok(call) => call,
        Err(_) => return Ok(None),
    };
    if !call.name.eq_ignore_ascii_case("surv") {
        return Ok(None);
    }
    let vars = call
        .args
        .iter()
        .filter_map(|arg| match arg {
            CallArgSpec::Positional(v) => Some(unquote_column(v)),
            CallArgSpec::Named { .. } => None,
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    match vars.len() {
        // Right-censored shorthand: Surv(time, event) ≡ Surv(0, time, event)
        // with a synthetic zero entry column. This matches R's
        // `survival::Surv(time, event)` default for left-truncation-free data.
        2 => Ok(Some((None, vars[0].clone(), vars[1].clone()))),
        3 => Ok(Some((
            Some(vars[0].clone()),
            vars[1].clone(),
            vars[2].clone(),
        ))),
        n => Err(FormulaDslError::InvalidArgument {
            reason: format!(
                "Surv(...) expects either Surv(time, event) (right-censored) or \
                 Surv(entry, exit, event) (left-truncated); got {n} columns"
            ),
        }),
    }
}

/// Parsed `SurvInterval(L, R, event)` interval-censored response.
///
/// Returns `Some((left_col, right_col, event_col))` when the left-hand side is a
/// `SurvInterval(...)` call, `None` otherwise (so a plain `Surv(...)` or a bare
/// column response falls through to the other response parsers).
///
/// Interval censoring observes only a bracket `T ∈ (L, R]` — the exact event
/// time is never seen — and its row contribution is the survival-mass difference
/// `log[S(L) − S(R)]`, distinct from both the exact-event point density and the
/// single-sided right-censored survival. A *dedicated call name* (rather than
/// overloading the 3-argument `Surv(entry, exit, event)` delayed-entry form,
/// which is also 3-argument and semantically incompatible) is the unambiguous
/// DSL spelling: it mirrors flexsurv's `Surv(L, R, type="interval2")` intent
/// without colliding with the existing left-truncation grammar.
pub fn parse_surv_interval_response(
    lhs: &str,
) -> Result<Option<(String, String, String)>, FormulaDslError> {
    let trimmed = lhs.trim();
    let call = match parse_function_call(trimmed) {
        Ok(call) => call,
        Err(_) => return Ok(None),
    };
    if !call.name.eq_ignore_ascii_case("survinterval") {
        return Ok(None);
    }
    let vars = call
        .args
        .iter()
        .filter_map(|arg| match arg {
            CallArgSpec::Positional(v) => Some(unquote_column(v)),
            CallArgSpec::Named { .. } => None,
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    match vars.len() {
        3 => Ok(Some((vars[0].clone(), vars[1].clone(), vars[2].clone()))),
        n => Err(FormulaDslError::InvalidArgument {
            reason: format!(
                "SurvInterval(...) expects SurvInterval(L, R, event) (interval-censored, the \
                 observed bracket T ∈ (L, R]); got {n} columns"
            ),
        }),
    }
}

/// The data column a formula's response names: the trimmed left-hand side that
/// [`parse_formula`] stores as the response. `None` when the formula does not
/// parse, or when the response is a `Surv(...)` or `SurvInterval(...)` call
/// naming several columns.
/// Source text of the automatic `.` term: "every data column the rest of the
/// fit does not already use". It is expanded against the data schema by
/// [`crate::inference::automatic_formula::expand_automatic_formula`].
pub const AUTOMATIC_REST_TERM: &str = ".";

pub(crate) fn formula_rhs_terms(formula: &str) -> Result<(String, Vec<String>), String> {
    let parsed = parse_formula_dsl(formula)?;
    Ok((parsed.response_expr, parsed.rhs_terms))
}

pub fn formula_response_column(formula: &str) -> Option<String> {
    let response = parse_formula_dsl(formula).ok()?.response_expr;
    match (
        parse_surv_response(&response),
        parse_surv_interval_response(&response),
    ) {
        (Ok(None), Ok(None)) if !response.is_empty() => Some(response),
        _ => None,
    }
}

fn top_level_formula_separator(input: &str) -> Result<Option<usize>, String> {
    let mut depth = 0_i32;
    let mut quotes = QuoteTracker::default();

    for (idx, ch) in input.char_indices() {
        if quotes.step(ch) {
            continue;
        }
        if matches!(ch, '(' | '[' | '{') {
            depth += 1;
        } else if matches!(ch, ')' | ']' | '}') && depth > 0 {
            depth -= 1;
        } else if ch == '~' && depth == 0 {
            return Ok(Some(idx));
        }
    }

    if quotes.is_open() || depth != 0 {
        return Err(FormulaDslError::ParseError {
            reason: "invalid auxiliary formula syntax: unbalanced parentheses or quotes"
                .to_string(),
        }
        .into());
    }
    Ok(None)
}

pub fn parse_matching_auxiliary_formula(
    formula: &str,
    response: &str,
    flag_name: &str,
) -> Result<(String, ParsedFormula), FormulaDslError> {
    let rhs = formula.trim();
    if top_level_formula_separator(rhs)?.is_some() {
        return Err(FormulaDslError::InvalidArgument {
            reason: format!(
                "{flag_name} expects only the terms after '~', not a full 'response ~ terms' formula; use {flag_name} 's(x)' instead of {flag_name} 'y ~ s(x)' (or pass '1' for an intercept-only noise model)"
            ),
        });
    }
    let parsed_formula = parse_formula(&format!("{response} ~ {rhs}"))?;
    Ok((rhs.to_string(), parsed_formula))
}

pub fn validate_auxiliary_formula_controls(
    parsed_formula: &ParsedFormula,
    flag_name: &str,
) -> Result<(), String> {
    if parsed_formula.linkwiggle.is_some() {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "linkwiggle(...) is only supported in the main formula, not {flag_name}"
            ),
        }
        .into());
    }
    if parsed_formula
        .terms
        .iter()
        .any(|term| matches!(term, ParsedTerm::NoIntercept))
    {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "intercept removal (`0 +` / `- 1`) is only supported in the main formula, \
                 not {flag_name}"
            ),
        }
        .into());
    }
    if parsed_formula.timewiggle.is_some() {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "timewiggle(...) is only supported in the main survival formula, not {flag_name}"
            ),
        }
        .into());
    }
    if parsed_formula.linkspec.is_some() {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!("link(...) is only supported in the main formula, not {flag_name}"),
        }
        .into());
    }
    if parsed_formula.survivalspec.is_some() {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "survmodel(...) is only supported in the main survival formula, not {flag_name}"
            ),
        }
        .into());
    }
    if !parsed_formula.slope_surfaces.is_empty() && flag_name != "--slope-formula" {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "slope(...) is only supported in --slope-formula, not {flag_name}"
            ),
        }
        .into());
    }
    Ok::<(), _>(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterceptToken {
    Keep,
    Remove,
}

/// Classify a whole top-level RHS term as an intercept switch (R/patsy
/// convention): `1`/`+1` keeps the intercept, `0`/`+0`/`-1` removes it.
fn intercept_token(term: &str) -> Option<InterceptToken> {
    let compact: String = term.chars().filter(|c| !c.is_whitespace()).collect();
    match compact.as_str() {
        "1" | "+1" => Some(InterceptToken::Keep),
        "0" | "+0" | "-1" => Some(InterceptToken::Remove),
        _ => None,
    }
}

/// Strip a trailing binary `- 1` from a raw RHS term (`x - 1`, `a*b - 1`),
/// returning the operand in front of it. The top-level splitter never cuts
/// on `-`, so this is where `y ~ x - 1` loses its intercept. The `1` must be
/// a whole token directly after the `-`, and the text before the `-` must be
/// a complete operand (not end in an operator or an opening delimiter).
fn strip_trailing_intercept_removal(term: &str) -> Option<&str> {
    let rest = term
        .trim_end()
        .strip_suffix('1')?
        .trim_end()
        .strip_suffix('-')?
        .trim_end();
    let last = rest.chars().last()?;
    if matches!(
        last,
        ':' | '*' | '/' | '^' | '+' | '-' | '(' | '[' | '{' | ',' | '=' | '~'
    ) {
        return None;
    }
    Some(rest)
}

pub fn parse_formula(formula: &str) -> Result<ParsedFormula, FormulaDslError> {
    let parsed_dsl =
        parse_formula_dsl(formula).map_err(|reason| FormulaDslError::ParseError { reason })?;
    let lhs = parsed_dsl.response_expr.trim();
    if lhs.is_empty() {
        return Err(FormulaDslError::ParseError {
            reason: "formula response (left-hand side) cannot be empty".to_string(),
        });
    }
    let mut terms = Vec::<ParsedTerm>::new();
    let mut linkwiggle: Option<LinkWiggleFormulaSpec> = None;
    let mut timewiggle: Option<LinkWiggleFormulaSpec> = None;
    let mut linkspec: Option<LinkFormulaSpec> = None;
    let mut survivalspec: Option<SurvivalFormulaSpec> = None;
    let mut slope_surfaces = Vec::<SlopeSurfaceSpec>::new();
    // Track seen-term-keys so we can reject exact duplicates like
    // `y ~ smooth(x) + smooth(x)` upfront — without this the duplicate
    // produces a rank-deficient design and the user has no idea why their
    // fit is over-parameterized.
    let mut seen_term_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut expanded_terms = Vec::<String>::new();
    // Intercept tokens (`1`, `0`, `-1`, and a trailing `- 1` on a term) are
    // model-level switches, not terms. They are consumed here, before WR
    // expansion, so `y ~ x - 1` reaches the expander as plain `x`.
    let mut explicit_intercept = false;
    let mut intercept_removed = false;
    for raw in parsed_dsl.rhs_terms {
        let mut trimmed = raw.trim();
        while let Some(rest) = strip_trailing_intercept_removal(trimmed) {
            intercept_removed = true;
            trimmed = rest;
        }
        match intercept_token(trimmed) {
            Some(InterceptToken::Keep) => {
                explicit_intercept = true;
                continue;
            }
            Some(InterceptToken::Remove) => {
                intercept_removed = true;
                continue;
            }
            None => {}
        }
        if trimmed.is_empty() {
            expanded_terms.push(String::new());
            continue;
        }
        // Single function-call terms (smooths, group(), etc.) are opaque to
        // WR expansion; pass them through verbatim so parse_term sees the
        // exact source string. Bare identifiers and any term mentioning
        // `:`, `*`, `/`, `^` are routed through the AST-driven WR expander.
        let is_call = parse_function_call(trimmed).is_ok();
        let needs_expansion = !is_call
            && trimmed
                .chars()
                .scan((0i32, QuoteTracker::default()), |(depth, quotes), ch| {
                    if quotes.step(ch) {
                        return Some(None);
                    }
                    let d_before = *depth;
                    if matches!(ch, '(' | '[' | '{') {
                        *depth += 1;
                    } else if matches!(ch, ')' | ']' | '}') && *depth > 0 {
                        *depth -= 1;
                    }
                    Some(Some((d_before, ch)))
                })
                .flatten()
                .any(|(d, ch)| d == 0 && matches!(ch, ':' | '*' | '/' | '^'));
        if needs_expansion {
            for atoms in
                expand_wr_term(trimmed).map_err(|reason| FormulaDslError::ParseError { reason })?
            {
                if atoms.is_empty() {
                    continue;
                }
                expanded_terms.push(atoms.join(":"));
            }
        } else {
            expanded_terms.push(trimmed.to_string());
        }
    }

    if explicit_intercept && intercept_removed {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "formula `{formula}` both includes the intercept (`1`) and removes it \
                 (`0` / `-1`); keep exactly one of them"
            ),
        });
    }
    for raw in expanded_terms {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        if split_outside_backticks(t, ':')
            .into_iter()
            .any(|atom| intercept_token(atom).is_some())
        {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "intercept token `{t}` in formula `{formula}` cannot take part in an \
                     interaction or nesting operator; write `0 +` or `- 1` as its own \
                     top-level term"
                ),
            });
        }
        if t == AUTOMATIC_REST_TERM {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: "the `.` term (every remaining column) must be expanded against the \
                         data before parsing; fit entry points expand it automatically"
                    .to_string(),
            });
        }
        // Normalize whitespace so `smooth(x)` and `smooth( x )` match,
        // but preserve whitespace inside string literals so that
        // `bs="a b"` and `bs="ab"` do not collide.
        let key: String = {
            let mut acc = String::with_capacity(t.len());
            let mut quotes = QuoteTracker::default();
            for ch in t.chars() {
                if quotes.step(ch) || !ch.is_whitespace() {
                    acc.push(ch);
                }
            }
            acc
        };
        if !seen_term_keys.insert(key.clone()) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "formula `{formula}` lists term `{t}` more than once. \
                     Duplicate terms produce a rank-deficient design; \
                     keep one copy or differentiate them (e.g. distinct k=, bs= options)."
                ),
            });
        }
        match parse_term(t)? {
            ParsedTerm::LinkWiggle { options } => {
                if linkwiggle.is_some() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: "formula can include at most one linkwiggle(...) term".to_string(),
                    });
                }
                linkwiggle = Some(parse_linkwiggle_formulaspec(&options, t)?);
            }
            ParsedTerm::TimeWiggle { options } => {
                if timewiggle.is_some() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: "formula can include at most one timewiggle(...) term".to_string(),
                    });
                }
                timewiggle = Some(parse_linkwiggle_formulaspec(&options, t)?);
            }
            ParsedTerm::LinkConfig { options } => {
                if linkspec.is_some() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: "formula can include at most one link(...) term".to_string(),
                    });
                }
                linkspec = Some(parse_link_formulaspec(&options, t)?);
            }
            ParsedTerm::SurvivalConfig { options } => {
                if survivalspec.is_some() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: "formula can include at most one survmodel(...) term".to_string(),
                    });
                }
                survivalspec = Some(parse_survival_formulaspec(&options, t)?);
            }
            ParsedTerm::SlopeSurface { z_column, terms } => {
                slope_surfaces.push(SlopeSurfaceSpec { z_column, terms });
            }
            other => terms.push(other),
        }
    }
    // Reject self-referential formulas like `y ~ smooth(y)` or `y ~ y`: the
    // response is its own predictor, which is a trivial identity fit and
    // almost certainly a user mistake. Only flag the simple-identifier case
    // (so Surv(entry, exit, event) ~ smooth(time) is left alone — the
    // response is the Surv triple, not the bare "time" column).
    if lhs.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !lhs.is_empty()
        && parsed_terms_reference_column(&terms, lhs)
    {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "formula `{formula}` uses response column `{lhs}` as its own predictor. \
                 This fits y as a function of itself and is almost certainly a typo. \
                 Drop the term that mentions `{lhs}` from the right-hand side."
            ),
        });
    }
    if intercept_removed {
        let has_model_term = terms.iter().any(|term| {
            matches!(
                term,
                ParsedTerm::Linear { .. }
                    | ParsedTerm::BoundedLinear { .. }
                    | ParsedTerm::RandomEffect { .. }
                    | ParsedTerm::Smooth { .. }
                    | ParsedTerm::Interaction { .. }
            )
        });
        if !has_model_term {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "formula `{formula}` removes the intercept and has no other term, so \
                     the model would have no coefficients; drop `0` / `-1` for an \
                     intercept-only model"
                ),
            });
        }
        terms.push(ParsedTerm::NoIntercept);
    }
    Ok(ParsedFormula {
        response: lhs.to_string(),
        terms,
        slope_surfaces,
        linkwiggle,
        timewiggle,
        linkspec,
        survivalspec,
    })
}

pub(crate) fn parse_term(raw: &str) -> Result<ParsedTerm, String> {
    parse_term_quoted(raw).map(unquote_parsed_term)
}

/// Resolve backtick-quoted column references (`` s(`my col`) ``) to the bare
/// column names the term builder looks up. Labels keep the text the user wrote.
fn unquote_parsed_term(term: ParsedTerm) -> ParsedTerm {
    match term {
        ParsedTerm::Linear {
            name,
            explicit,
            double_penalty,
            coefficient_min,
            coefficient_max,
        } => ParsedTerm::Linear {
            name: unquote_column(&name),
            explicit,
            double_penalty,
            coefficient_min,
            coefficient_max,
        },
        ParsedTerm::BoundedLinear {
            name,
            min,
            max,
            prior,
            double_penalty,
        } => ParsedTerm::BoundedLinear {
            name: unquote_column(&name),
            min,
            max,
            prior,
            double_penalty,
        },
        ParsedTerm::RandomEffect {
            name,
            lenient_unseen,
        } => ParsedTerm::RandomEffect {
            name: unquote_column(&name),
            lenient_unseen,
        },
        ParsedTerm::Smooth {
            label,
            vars,
            kind,
            mut options,
        } => {
            if let Some(by) = options.get_mut("by") {
                *by = unquote_column(by);
            }
            ParsedTerm::Smooth {
                label,
                vars: vars.iter().map(|v| unquote_column(v)).collect(),
                kind,
                options,
            }
        }
        ParsedTerm::Interaction {
            vars,
            double_penalty,
        } => {
            let mut vars: Vec<String> = vars.iter().map(|v| unquote_column(v)).collect();
            vars.sort();
            ParsedTerm::Interaction {
                vars,
                double_penalty,
            }
        }
        ParsedTerm::SlopeSurface { z_column, terms } => ParsedTerm::SlopeSurface {
            z_column: unquote_column(&z_column),
            terms,
        },
        other => other,
    }
}

fn parse_term_quoted(raw: &str) -> Result<ParsedTerm, String> {
    fn split_call_args(call: &FunctionCallSpec) -> (Vec<String>, BTreeMap<String, String>) {
        let mut vars = Vec::<String>::new();
        let mut options = BTreeMap::<String, String>::new();
        for arg in &call.args {
            match arg {
                CallArgSpec::Positional(v) => vars.push(v.trim().to_string()),
                CallArgSpec::Named { key, value } => {
                    options.insert(key.to_ascii_lowercase(), strip_quotes(value).to_string());
                }
            }
        }
        (vars, options)
    }

    // Wilkinson-Rogers `:` interaction term. The expander in `parse_formula`
    // produces `a:b[:c...]` for these; parse_term is also reached directly
    // from tests, so handle the syntax here as well.
    let operands = split_outside_backticks(raw, ':');
    if operands.len() > 1 && operands.iter().all(|piece| is_column_ref(piece.trim())) {
        let vars: Vec<String> = operands
            .iter()
            .map(|piece| piece.trim().to_string())
            .collect();
        if vars.len() >= 2 {
            let mut sorted = vars.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() != vars.len() {
                return Err(FormulaDslError::IncompatibleTerm {
                    reason: format!(
                        "interaction term `{raw}` references the same variable more than once"
                    ),
                }
                .into());
            }
            return Ok(ParsedTerm::Interaction {
                vars: sorted,
                double_penalty: true,
            });
        }
    }

    // A term that does not parse as `name(args)` is not a function-call term;
    // the plain-variable handling below is the answer, so there is no error here
    // to report.
    if let Ok(call) = parse_function_call(raw) {
        // `factor(g)` is the one spelling of a categorical level effect. A
        // second name for it would be an option with nothing to choose, so
        // `C(g)` is refused with the spelling to use instead.
        if call.name == "C" {
            let target = split_call_args(&call).0.join(", ");
            return Err(format!(
                "`C()` is not a term function in '{raw}'; write factor({target}) for a \
                 categorical level effect"
            ));
        }
        let name = call.name.to_ascii_lowercase();
        let (vars, mut options) = split_call_args(&call);
        if let Some(canonical) = removed_spellings::canonical_for(removed_spellings::TERM_FUNCTIONS, &name)
        {
            return Err(format!(
                "unknown term function `{name}`; use `{canonical}()`: {raw}"
            ));
        }
        if SMOOTH_TERM_FUNCTIONS.contains(&name.as_str()) {
            removed_spellings::reject_removed_smooth_spellings(&name, &options)
                .map_err(|reason| format!("{reason}: {raw}"))?;
        }
        match name.as_str() {
            "nonnegative" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("nonnegative() expects exactly one variable: {raw}"),
                    }
                    .into());
                }
                validate_known_term_options("nonnegative", &options, &["double_penalty"], raw)?;
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    double_penalty: option_bool(&options, "double_penalty")?
                        .unwrap_or(true),
                    coefficient_min: Some(0.0),
                    coefficient_max: None,
                });
            }
            "nonpositive" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("nonpositive() expects exactly one variable: {raw}"),
                    }
                    .into());
                }
                validate_known_term_options("nonpositive", &options, &["double_penalty"], raw)?;
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    double_penalty: option_bool(&options, "double_penalty")?
                        .unwrap_or(true),
                    coefficient_min: None,
                    coefficient_max: Some(0.0),
                });
            }
            "bounded" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("bounded() expects exactly one variable: {raw}"),
                    }
                    .into());
                }
                validate_known_term_options(
                    "bounded",
                    &options,
                    &[
                        "min",
                        "max",
                        "prior",
                        "target",
                        "strength",
                        "double_penalty",
                    ],
                    raw,
                )?;
                let min = parse_required_f64_option(&options, "min", raw)?;
                let max = parse_required_f64_option(&options, "max", raw)?;
                if !min.is_finite() || !max.is_finite() || min >= max {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "bounded() requires finite min < max, got min={min}, max={max}: {raw}"
                        ),
                    }
                    .into());
                }
                // Unlike a plain `linear()` term, `bounded()` already commits
                // the coefficient to a prior on the box; layering the
                // null-space ridge on top is structurally rejected downstream
                // (`design_construction.rs`: "bounded linear term ... cannot
                // also use double_penalty"), so the default must be `false`,
                // not the `linear()`/`s()` convention of `true`.
                let double_penalty =
                    option_bool(&options, "double_penalty")?.unwrap_or(false);
                return match parse_bounded_priorspec(&options, min, max, raw)? {
                    BoundedPriorChoice::Latent(prior) => Ok(ParsedTerm::BoundedLinear {
                        name: vars[0].clone(),
                        min,
                        max,
                        prior,
                        double_penalty,
                    }),
                    BoundedPriorChoice::FlatOnBox => {
                        if double_penalty {
                            return Err(FormulaDslError::IncompatibleTerm {
                                reason: format!(
                                    "bounded(prior=uniform) is flat on the box and cannot also \
                                     carry the double_penalty ridge; use \
                                     linear(x, min=..., max=...) for a shrunk boxed coefficient: {raw}"
                                ),
                            }
                            .into());
                        }
                        Ok(ParsedTerm::Linear {
                            name: vars[0].clone(),
                            explicit: true,
                            double_penalty: false,
                            coefficient_min: Some(min),
                            coefficient_max: Some(max),
                        })
                    }
                };
            }
            "group" | "factor" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "{name}() expects exactly one variable, got '{}': {raw}",
                            vars.join(",")
                        ),
                    }
                    .into());
                }
                // None of the categorical wrappers take options: every one
                // lowers to a level block whose ridge strength is
                // REML-estimated, so `factor(g, foo=1)` or
                // `group(g, double_penalty=false)` is a typo, not a request.
                validate_known_term_options(&name, &options, &[], raw)?;
                // `factor(g)` forces categorical encoding of the column and,
                // like a bare `+ g` main effect, is strict on unseen levels.
                // `group(g)` is a genuine random effect that shrinks a
                // held-out group to the population mean, so it tolerates
                // unseen levels. Both share the penalized-categorical block;
                // only the unseen policy differs (#2137/#2102).
                let lenient_unseen = name != "factor";
                return Ok(ParsedTerm::RandomEffect {
                    name: vars[0].clone(),
                    lenient_unseen,
                });
            }
            "te" => {
                if vars.len() < 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("te() requires at least two variables: {raw}"),
                    }
                    .into());
                }
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::Te,
                    options,
                });
            }
            "t2" => {
                if vars.len() < 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("t2() requires at least two variables: {raw}"),
                    }
                    .into());
                }
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::T2,
                    options,
                });
            }
            "ti" => {
                // Tensor interaction smooth (mgcv `ti`): structurally a
                // tensor-product smooth, but the marginal main effects are
                // excluded so only the pure interaction is modeled. Shares the
                // tensor materialization path with `te`; the distinct
                // `SmoothKind::Ti` drives per-margin sum-to-zero
                // identifiability in the term builder.
                if vars.len() < 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("ti() requires at least two variables: {raw}"),
                    }
                    .into());
                }
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::Ti,
                    options,
                });
            }
            "fs" | "sz" => {
                if vars.len() != 2 {
                    return Err(format!("{}() expects exactly two variables: {raw}", name));
                }
                options.insert("bs".to_string(), name.clone());
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "thinplate" | "thin_plate" | "tps" => {
                if vars.len() < 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "thinplate()/thin_plate()/tps() requires at least two variables: {raw}"
                        ),
                    }
                    .into());
                }
                fix_function_basis(&mut options, &name, "tps", raw)?;
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "smooth" | "s" | "cyclic" => {
                if vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("smooth()/s() requires at least one variable: {raw}"),
                    }
                    .into());
                }
                // `s(g, bs='re')` with a single variable is a random
                // intercept on the factor `g`. Route it to the
                // dedicated random-effect machinery (which expects a single
                // categorical column) rather than to the factor-smooth path
                // (which requires a numeric companion).
                let bs_is_re = options
                    .get("bs")
                    .map(|v| {
                        v.trim()
                            .trim_matches(|c| c == '\'' || c == '"')
                            .to_ascii_lowercase()
                    })
                    .as_deref()
                    == Some("re");
                if bs_is_re && vars.len() == 1 {
                    // `s(g, bs="re")` is a genuine random effect: lenient on
                    // unseen levels (held-out group → population mean).
                    return Ok(ParsedTerm::RandomEffect {
                        name: vars[0].clone(),
                        lenient_unseen: true,
                    });
                }
                if name == "cyclic" {
                    fix_function_basis(&mut options, &name, "cyclic", raw)?;
                }
                if matches!(name.as_str(), "fs" | "sz") {
                    options.insert("bs".to_string(), name.clone());
                }
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "sphere" | "sos" | "spherical" | "s2" => {
                // `s2()` is an alias for the intrinsic S² (sphere) smooth, just
                // like `sphere()`/`sos()`/`spherical()`. All four share the
                // Wahba/harmonic sphere basis, so they must dispatch through
                // the identical `bs=sphere` route; otherwise `s2()` would
                // silently fall back to a generic Euclidean 2-D smooth over
                // (lat, lon) and diverge in the spatial-kappa optimizer.
                if vars.len() != 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "{name}() expects exactly two variables: latitude and longitude; got {} in {raw}",
                            vars.len()
                        ),
                    }
                    .into());
                }
                fix_function_basis(&mut options, &name, "sphere", raw)?;
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "mjs" => {
                // Measure-jet spline smooth (`basis::measure_jet_smooth` docs)
                // for responses varying along an unknown low-dimensional set
                // inside a higher-dimensional ambient space.
                if vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("{name}() requires at least one variable: {raw}"),
                    }
                    .into());
                }
                fix_function_basis(&mut options, &name, "mjs", raw)?;
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "curv" => {
                // Constant-curvature (M_κ) geodesic-kernel smooth (#944): the
                // κ-generic sibling of sphere()/s2(), interpolating
                // S^d → ℝ^d → H^d through `kappa=` (default 0 = flat).
                if vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("{name}() requires at least one variable: {raw}"),
                    }
                    .into());
                }
                fix_function_basis(&mut options, &name, "curv", raw)?;
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "matern" => {
                if vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("matern() requires at least one variable: {raw}"),
                    }
                    .into());
                }
                fix_function_basis(&mut options, &name, "matern", raw)?;
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "duchon" => {
                if vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("duchon() requires at least one variable: {raw}"),
                    }
                    .into());
                }
                if option_bool(&options, "cyclic")?.unwrap_or(false)
                    || option_bool(&options, "periodic")?.unwrap_or(false)
                {
                    options.insert("cyclic".to_string(), "true".to_string());
                }
                fix_function_basis(&mut options, &name, "duchon", raw)?;
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "pca" => {
                if vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("pca() requires at least one variable: {raw}"),
                    }
                    .into());
                }
                fix_function_basis(&mut options, &name, "pca", raw)?;
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "linkwiggle" => {
                if !vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "linkwiggle() takes named options only; positional args are not supported: {raw}"
                        ),
                    }
                    .into());
                }
                return Ok(ParsedTerm::LinkWiggle { options });
            }
            "timewiggle" => {
                if !vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "timewiggle() takes named options only; positional args are not supported: {raw}"
                        ),
                    }
                    .into());
                }
                return Ok(ParsedTerm::TimeWiggle { options });
            }
            "link" => {
                if !vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "link() takes named options only; positional args are not supported: {raw}"
                        ),
                    }
                    .into());
                }
                return Ok(ParsedTerm::LinkConfig { options });
            }
            "survmodel" => {
                if !vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "survmodel() takes named options only; positional args are not supported: {raw}"
                        ),
                    }
                    .into());
                }
                return Ok(ParsedTerm::SurvivalConfig { options });
            }
            "slope" => {
                validate_known_term_options("slope", &options, &[], raw)?;
                if vars.len() < 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "slope() expects a z column followed by one or more RHS terms; add one slope(z, ...) declaration per vector-z coordinate: {raw}"
                        ),
                    }
                    .into());
                }
                let z_column = vars[0].trim();
                if !is_column_ref(z_column) {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "slope() z column must be a bare column name, got `{z_column}` in {raw}"
                        ),
                    }
                    .into());
                }
                let rhs = vars[1..].join(" + ");
                let parsed = parse_formula(&format!("__slope__ ~ {rhs}"))?;
                if !parsed.slope_surfaces.is_empty() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: format!(
                            "slope() declarations cannot be nested inside another slope(): {raw}"
                        ),
                    }
                    .into());
                }
                validate_auxiliary_formula_controls(&parsed, "slope()")?;
                return Ok(ParsedTerm::SlopeSurface {
                    z_column: z_column.to_string(),
                    terms: parsed.terms,
                });
            }
            "linear" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("linear() expects exactly one variable: {raw}"),
                    }
                    .into());
                }
                validate_known_term_options(
                    "linear",
                    &options,
                    &["min", "max", "double_penalty"],
                    raw,
                )?;
                let (coefficient_min, coefficient_max) =
                    parse_linear_constraint_bounds(&options, raw)?;
                let double_penalty =
                    option_bool(&options, "double_penalty")?.unwrap_or(true);
                if split_outside_backticks(&vars[0], ':').len() > 1 {
                    if coefficient_min.is_some() || coefficient_max.is_some() {
                        return Err(FormulaDslError::IncompatibleTerm {
                            reason: format!(
                                "linear() coefficient bounds are not supported on an interaction: {raw}"
                            ),
                        }
                        .into());
                    }
                    let mut interaction_vars = split_outside_backticks(&vars[0], ':')
                        .into_iter()
                        .map(str::trim)
                        .map(str::to_string)
                        .collect::<Vec<_>>();
                    if interaction_vars.len() < 2
                        || interaction_vars.iter().any(|var| !is_column_ref(var))
                    {
                        return Err(FormulaDslError::InvalidArgument {
                            reason: format!(
                                "linear() interaction must contain at least two bare column names: {raw}"
                            ),
                        }
                        .into());
                    }
                    interaction_vars.sort();
                    let original_len = interaction_vars.len();
                    interaction_vars.dedup();
                    if interaction_vars.len() != original_len {
                        return Err(FormulaDslError::IncompatibleTerm {
                            reason: format!(
                                "linear() interaction references the same variable more than once: {raw}"
                            ),
                        }
                        .into());
                    }
                    return Ok(ParsedTerm::Interaction {
                        vars: interaction_vars,
                        double_penalty,
                    });
                }
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    double_penalty,
                    coefficient_min,
                    coefficient_max,
                });
            }
            _ => {
                return Err(format!(
                    "unknown term function `{name}` in '{raw}'. Supported: bounded(), linear(), nonnegative(), nonpositive(), smooth()/s(), cyclic(), thinplate()/thin_plate()/tps(), te(), t2(), ti(), fs(), sz(), group(), factor(), sphere()/sos()/spherical(), s2(), matern(), duchon(), mjs(), curv(), pca(), slope(), linkwiggle(), timewiggle(), link(), survmodel()"
                ));
            }
        }
    }

    let ident = raw.trim();
    if !is_column_ref(ident) {
        return Err(FormulaDslError::UnknownIdentifier {
            reason: format!("unsupported top-level RHS term: {raw}"),
        }
        .into());
    }

    Ok(ParsedTerm::Linear {
        name: ident.to_string(),
        explicit: false,
        double_penalty: true,
        coefficient_min: None,
        coefficient_max: None,
    })
}

// ---------------------------------------------------------------------------
// Link choice parsing
// ---------------------------------------------------------------------------

pub fn parse_link_choice(
    raw: Option<&str>,
    flexible_flag: bool,
) -> Result<Option<LinkChoice>, FormulaDslError> {
    if raw.is_none() && !flexible_flag {
        return Ok(None);
    }
    let Some(v) = raw else {
        return Ok(Some(LinkChoice {
            mode: LinkMode::Flexible,
            link: LinkFunction::Probit,
            mixture_components: None,
        }));
    };
    let t = v.trim().to_ascii_lowercase();
    if let Some(inner) = t
        .strip_prefix("flexible(")
        .and_then(|s| s.strip_suffix(')'))
    {
        if let Some(components_inner) = inner
            .strip_prefix("blended(")
            .and_then(|s| s.strip_suffix(')'))
            .or_else(|| {
                inner
                    .strip_prefix("mixture(")
                    .and_then(|s| s.strip_suffix(')'))
            })
        {
            parse_link_component_list(components_inner)?;
            return Err(FormulaDslError::IncompatibleTerm {
                reason:
                    "flexible(...) does not support blended(...)/mixture(...) links; wiggle is only supported for jointly fit standard links"
                        .to_string(),
            });
        }
        let link = parse_linkname(inner)?;
        if !linkname_supports_joint_wiggle(link) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason:
                    "flexible(...) does not support sas/beta-logistic links; wiggle is only supported for jointly fit standard links"
                        .to_string(),
            });
        }
        return Ok(Some(LinkChoice {
            mode: LinkMode::Flexible,
            link,
            mixture_components: None,
        }));
    }
    if let Some(inner) = t
        .strip_prefix("blended(")
        .and_then(|s| s.strip_suffix(')'))
        .or_else(|| t.strip_prefix("mixture(").and_then(|s| s.strip_suffix(')')))
    {
        if flexible_flag {
            return Err(FormulaDslError::IncompatibleTerm {
                reason:
                    "--flexible-link cannot be combined with --link blended(...)/mixture(...); blended inverse links are not flexible-link mode"
                        .to_string(),
            });
        }
        let components = parse_link_component_list(inner)?;
        return Ok(Some(LinkChoice {
            mode: LinkMode::Strict,
            link: LinkFunction::Logit,
            mixture_components: Some(components),
        }));
    }

    let link = parse_linkname(&t)?;
    if flexible_flag && !linkname_supports_joint_wiggle(link) {
        return Err(FormulaDslError::IncompatibleTerm {
            reason:
                "--flexible-link does not support sas/beta-logistic links; wiggle is only supported for jointly fit standard links"
                    .to_string(),
        });
    }
    Ok(Some(LinkChoice {
        mode: if flexible_flag {
            LinkMode::Flexible
        } else {
            LinkMode::Strict
        },
        link,
        mixture_components: None,
    }))
}

/// Parse a link name through the canonical vocabulary in
/// [`LinkFunction::from_name`]; the error lists [`LinkFunction::ALL`].
pub fn parse_linkname(v: &str) -> Result<LinkFunction, FormulaDslError> {
    LinkFunction::from_name(v).ok_or_else(|| {
        let unknown = gam_problem::types::UnknownLinkName(v.trim().to_string());
        FormulaDslError::UnknownIdentifier {
            reason: if LinkFunction::canonical_for_unknown(v).is_some() {
                unknown.to_string()
            } else {
                format!(
                    "{unknown}, blended(...)/mixture(...) or flexible(...). \
                     The formula term `link(type=<type>)`, the parenthesized \
                     `family(<type>)` and Python's `link=` accept the same set."
                )
            },
        }
    })
}

pub(crate) fn parse_link_component(v: &str) -> Result<LinkComponent, String> {
    match v.trim() {
        "logit" => Ok(LinkComponent::Logit),
        "probit" => Ok(LinkComponent::Probit),
        "cloglog" => Ok(LinkComponent::CLogLog),
        "loglog" => Ok(LinkComponent::LogLog),
        "cauchit" => Ok(LinkComponent::Cauchit),
        other => Err(FormulaDslError::UnknownIdentifier {
            reason: format!(
                "unsupported blended-link component '{other}'; use probit|logit|cloglog|loglog|cauchit"
            ),
        }
        .into()),
    }
}

pub(crate) fn parse_link_component_list(v: &str) -> Result<Vec<LinkComponent>, String> {
    let mut out = Vec::new();
    for part in v.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let comp = parse_link_component(trimmed)?;
        if out.contains(&comp) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: "blended(...) cannot contain duplicate components".to_string(),
            }
            .into());
        }
        out.push(comp);
    }
    if out.len() < 2 {
        return Err(FormulaDslError::InvalidArgument {
            reason: "blended(...) requires at least two components".to_string(),
        }
        .into());
    }
    Ok(out)
}

#[cfg(test)]
mod formula_dsl_operator_family_issue_219_tests;

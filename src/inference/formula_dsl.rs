use std::collections::BTreeMap;

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

use crate::smooth::BoundedCoefficientPriorSpec;
use crate::types::{
    InverseLink, LikelihoodFamily, LinkComponent, LinkFunction, WigglePenaltyConfig,
};

#[derive(Parser)]
#[grammar_inline = r#"
WHITESPACE = _{ " " | "\t" | NEWLINE }

top_function_call = { SOI ~ function_call ~ EOI }
formula = { SOI ~ expr ~ "~" ~ rhs ~ EOI }
rhs = { term ~ ("+" ~ term)* }
term = { expr }

expr = { sum }
sum = { product ~ (add_op ~ product)* }
add_op = _{ "+" | "-" }
product = { unary ~ (mul_op ~ unary)* }
mul_op = _{ "*" | "/" }
unary = { unary_op* ~ primary }
unary_op = _{ "+" | "-" }

primary = { function_call | list_lit | ident | number | string_lit | "(" ~ expr ~ ")" }
list_lit = @{ "[" ~ (!"]" ~ ANY)* ~ "]" }
function_call = { ident ~ "(" ~ arg_list? ~ ")" }
arg_list = { arg ~ ("," ~ arg)* }
arg = { named_arg | expr }
named_arg = { ident ~ "=" ~ expr }

ident = @{ ident_start ~ ident_continue* }
ident_start = _{ ASCII_ALPHA | "_" }
ident_continue = _{ ASCII_ALPHANUMERIC | "_" | "." }

number = @{
    "-"?
    ~ (ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT*)? | "." ~ ASCII_DIGIT+)
    ~ (("e" | "E") ~ ("+" | "-")? ~ ASCII_DIGIT+)?
}

string_lit = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" | "'" ~ (!"'" ~ ANY)* ~ "'" }
"#]
struct FormulaParser;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormulaDslParse {
    pub response_expr: String,
    pub rhs_terms: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallArgSpec {
    Positional(String),
    Named { key: String, value: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionCallSpec {
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

impl std::fmt::Display for FormulaDslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError { reason }
            | Self::UnknownIdentifier { reason }
            | Self::InvalidArgument { reason }
            | Self::IncompatibleTerm { reason }
            | Self::MalformedConfig { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for FormulaDslError {}

impl From<FormulaDslError> for String {
    fn from(err: FormulaDslError) -> String {
        err.to_string()
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

pub fn parse_formula_dsl(formula: &str) -> Result<FormulaDslParse, String> {
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
        match part.as_rule() {
            Rule::expr if response_expr.is_none() => {
                response_expr = Some(part.as_str().trim().to_string());
            }
            Rule::rhs => {
                rhs_terms = Some(extract_rhs_terms(part)?);
            }
            _ => {}
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

fn delimiter_balance_error(prefix: &str) -> String {
    format!("{prefix}: unbalanced parentheses or quotes")
}

// Pest reports malformed delimiters as a generic parse failure. We validate the
// raw text first so callers get a stable, specific error class for unmatched
// parentheses/quotes instead of whichever grammar branch happened to fail last.
fn validate_balanced_delimiters(input: &str, prefix: &str) -> Result<(), String> {
    let mut stack = Vec::<char>::new();
    let mut in_single = false;
    let mut in_double = false;

    for ch in input.chars() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '(' | '[' | '{' if !in_single && !in_double => stack.push(ch),
            ')' | ']' | '}' if !in_single && !in_double => {
                let expected = match ch {
                    ')' => '(',
                    ']' => '[',
                    // The outer match arm guarantees ch is one of ')', ']', '}'.
                    _ => '{',
                };
                if stack.pop() != Some(expected) {
                    return Err(FormulaDslError::ParseError {
                        reason: delimiter_balance_error(prefix),
                    }
                    .into());
                }
            }
            _ => {}
        }
    }

    if in_single || in_double || !stack.is_empty() {
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
    let mut in_single = false;
    let mut in_double = false;
    let mut start = 0_usize;
    let text = rhs.as_str();
    let bytes = text.as_bytes();
    for (idx, &b) in bytes.iter().enumerate() {
        let ch = b as char;
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '(' | '[' | '{' if !in_single && !in_double => depth += 1,
            ')' | ']' | '}' if !in_single && !in_double && depth > 0 => depth -= 1,
            '+' if !in_single && !in_double && depth == 0 => {
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
            _ => {}
        }
    }
    if in_single || in_double || depth != 0 {
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

pub fn parse_function_call(input: &str) -> Result<FunctionCallSpec, String> {
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
        match part.as_rule() {
            Rule::ident => {
                if name.is_none() {
                    name = Some(part.as_str().trim().to_string());
                }
            }
            Rule::arg_list => {
                for a in part.into_inner() {
                    if a.as_rule() != Rule::arg {
                        continue;
                    }
                    let mut a_inner = a.into_inner();
                    let Some(first) = a_inner.next() else {
                        continue;
                    };
                    match first.as_rule() {
                        Rule::named_arg => {
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
                        }
                        Rule::expr => {
                            args.push(CallArgSpec::Positional(first.as_str().trim().to_string()));
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
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
        CallArgSpec, ParsedTerm, parse_formula, parse_formula_dsl, parse_function_call,
        parsed_terms_reference_column, validate_marginal_slope_z_column_exclusion,
    };

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

    #[test]
    fn parses_cyclic_formula_aliases() {
        let parsed = parse_formula("y ~ cyclic(theta, period_start=0, period_end=6.283)")
            .expect("parse cyclic formula");
        match &parsed.terms[0] {
            super::ParsedTerm::Smooth { vars, options, .. } => {
                assert_eq!(vars, &vec!["theta".to_string()]);
                assert_eq!(options.get("type").map(String::as_str), Some("cyclic"));
                assert_eq!(options.get("period_start").map(String::as_str), Some("0"));
            }
            other => panic!("expected cyclic smooth term, got {other:?}"),
        }
    }

    #[test]
    fn parses_function_callwithnamed_and_positional_args() {
        let call = parse_function_call("s(log(x + 1), type=\"duchon\", centers=12)").expect("call");
        assert_eq!(call.name, "s");
        assert_eq!(call.args.len(), 3);
        assert_eq!(
            call.args[0],
            CallArgSpec::Positional("log(x + 1)".to_string())
        );
        assert_eq!(
            call.args[1],
            CallArgSpec::Named {
                key: "type".to_string(),
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
    fn parse_formula_rejects_unsupported_top_level_rhs_expressions() {
        for formula in ["y ~ x - z", "y ~ -x", "y ~ x / z", "y ~ (x)", "y ~ x - 1"] {
            let err = parse_formula(formula).expect_err("expected formula parse failure");
            assert!(err.to_string().contains("unsupported top-level RHS term"));
        }
    }

    #[test]
    fn marginal_slope_z_column_validator_detects_linear_and_smooth_reuse() {
        let main = parse_formula("y ~ x + z").expect("parse main");
        let logslope = parse_formula("y ~ s(z, type=duchon, centers=6)").expect("parse logslope");

        assert!(parsed_terms_reference_column(&main.terms, "z"));
        assert!(parsed_terms_reference_column(&logslope.terms, "z"));

        let err = validate_marginal_slope_z_column_exclusion(
            &main,
            &parse_formula("y ~ 1").expect("parse clean logslope"),
            "z",
            "bernoulli marginal-slope",
            "--logslope-formula",
        )
        .expect_err("main formula should be rejected");
        assert!(err.contains("cannot also appear in the main formula"));

        let err = validate_marginal_slope_z_column_exclusion(
            &parse_formula("y ~ x").expect("parse clean main"),
            &logslope,
            "z",
            "bernoulli marginal-slope",
            "--logslope-formula",
        )
        .expect_err("logslope formula should be rejected");
        assert!(err.contains("cannot also appear in --logslope-formula"));
    }

    #[test]
    fn logslope_surface_declarations_are_additive() {
        let parsed = parse_formula("y ~ s(pc1) + logslope(z2, s(pc2)) + logslope(z3, x3)")
            .expect("parse additive logslope surfaces");
        assert_eq!(parsed.terms.len(), 1);
        assert_eq!(parsed.logslope_surfaces.len(), 2);
        assert_eq!(parsed.logslope_surfaces[0].z_column, "z2");
        assert_eq!(parsed.logslope_surfaces[0].terms.len(), 1);
        assert_eq!(parsed.logslope_surfaces[1].z_column, "z3");
        assert_eq!(parsed.logslope_surfaces[1].terms.len(), 1);
    }

    #[test]
    fn marginal_slope_z_column_validator_reserves_all_surface_z_columns() {
        let main = parse_formula("y ~ x").expect("parse main");
        let logslope = parse_formula("y ~ s(pc1) + logslope(z2, s(z3)) + logslope(z3, x)")
            .expect("parse logslope surfaces");
        let err = validate_marginal_slope_z_column_exclusion(
            &main,
            &logslope,
            "z1",
            "bernoulli marginal-slope",
            "--logslope-formula",
        )
        .expect_err("surface formula should reject another reserved z coordinate");
        assert!(err.contains("reserves z column 'z3'"));
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
    pub logslope_surfaces: Vec<LogSlopeSurfaceSpec>,
    pub linkwiggle: Option<LinkWiggleFormulaSpec>,
    pub timewiggle: Option<LinkWiggleFormulaSpec>,
    pub linkspec: Option<LinkFormulaSpec>,
    pub survivalspec: Option<SurvivalFormulaSpec>,
}

#[derive(Clone, Debug)]
pub struct LogSlopeSurfaceSpec {
    pub z_column: String,
    pub terms: Vec<ParsedTerm>,
}

pub fn marginal_slope_logslope_surfaces(
    logslope_formula: &ParsedFormula,
    default_z_column: &str,
) -> Result<Vec<LogSlopeSurfaceSpec>, String> {
    let mut surfaces = Vec::new();
    if !logslope_formula.terms.is_empty() {
        surfaces.push(LogSlopeSurfaceSpec {
            z_column: default_z_column.to_string(),
            terms: logslope_formula.terms.clone(),
        });
    }
    surfaces.extend(logslope_formula.logslope_surfaces.clone());
    if surfaces.is_empty() {
        surfaces.push(LogSlopeSurfaceSpec {
            z_column: default_z_column.to_string(),
            terms: Vec::new(),
        });
    }
    let mut seen = std::collections::BTreeSet::<String>::new();
    for surface in &surfaces {
        if !seen.insert(surface.z_column.clone()) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "logslope formula declares z column '{}' more than once; each z coordinate needs exactly one log-slope surface",
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
        coefficient_min: Option<f64>,
        coefficient_max: Option<f64>,
    },
    BoundedLinear {
        name: String,
        min: f64,
        max: f64,
        prior: BoundedCoefficientPriorSpec,
    },
    RandomEffect {
        name: String,
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
    LogSlopeSurface {
        z_column: String,
        terms: Vec<ParsedTerm>,
    },
}

pub fn parsed_terms_reference_column(terms: &[ParsedTerm], column_name: &str) -> bool {
    terms.iter().any(|term| match term {
        ParsedTerm::Linear { name, .. }
        | ParsedTerm::BoundedLinear { name, .. }
        | ParsedTerm::RandomEffect { name } => name == column_name,
        ParsedTerm::Smooth { vars, .. } => vars.iter().any(|var| var == column_name),
        ParsedTerm::LinkWiggle { .. }
        | ParsedTerm::TimeWiggle { .. }
        | ParsedTerm::LinkConfig { .. }
        | ParsedTerm::SurvivalConfig { .. } => false,
        ParsedTerm::LogSlopeSurface { z_column, terms } => {
            z_column == column_name || parsed_terms_reference_column(terms, column_name)
        }
    })
}

pub fn validate_marginal_slope_z_column_exclusion(
    main_formula: &ParsedFormula,
    logslope_formula: &ParsedFormula,
    z_column: &str,
    context: &str,
    logslope_label: &str,
) -> Result<(), String> {
    let surfaces = marginal_slope_logslope_surfaces(logslope_formula, z_column)?;
    for z_column in surfaces.iter().map(|surface| surface.z_column.as_str()) {
        if parsed_terms_reference_column(&main_formula.terms, z_column) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "{context} reserves z column '{z_column}' as the auxiliary latent score; it cannot also appear in the main formula"
                ),
            }
            .into());
        }
    }
    for reserved in surfaces.iter().map(|surface| surface.z_column.as_str()) {
        if parsed_terms_reference_column(&logslope_formula.terms, reserved) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "{context} reserves z column '{reserved}' as the auxiliary latent score; it cannot also appear in {logslope_label}"
                ),
            }
            .into());
        }
        for surface in &surfaces {
            if parsed_terms_reference_column(&surface.terms, reserved) {
                return Err(FormulaDslError::IncompatibleTerm {
                    reason: format!(
                        "{context} reserves z column '{reserved}' as an auxiliary latent score; it cannot also appear in {logslope_label}"
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

pub const fn linkname_supports_joint_wiggle(link: LinkFunction) -> bool {
    !matches!(link, LinkFunction::Sas | LinkFunction::BetaLogistic)
}

pub const fn linkchoice_supports_joint_wiggle(choice: &LinkChoice) -> bool {
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

pub const fn likelihood_family_supports_joint_wiggle(family: LikelihoodFamily) -> bool {
    !matches!(
        family,
        LikelihoodFamily::BinomialLatentCLogLog
            | LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture
    )
}

pub fn require_likelihood_family_supports_joint_wiggle(
    family: LikelihoodFamily,
    context: &str,
) -> Result<(), String> {
    if likelihood_family_supports_joint_wiggle(family) {
        Ok(())
    } else {
        Err(joint_wiggle_unsupported_link_message(context))
    }
}

pub const fn inverse_link_supports_joint_wiggle(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(LinkFunction::Identity)
            | InverseLink::Standard(LinkFunction::Log)
            | InverseLink::Standard(LinkFunction::Logit)
            | InverseLink::Standard(LinkFunction::Probit)
            | InverseLink::Standard(LinkFunction::CLogLog)
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

pub const fn binomial_inverse_link_supports_joint_wiggle(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(LinkFunction::Logit)
            | InverseLink::Standard(LinkFunction::Probit)
            | InverseLink::Standard(LinkFunction::CLogLog)
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
                "{context} does not support identity, log, latent-cloglog, SAS, BetaLogistic, or Mixture links; wiggle is only available for jointly fitted logit/probit/cloglog links"
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

pub fn option_usize(map: &BTreeMap<String, String>, key: &str) -> Option<usize> {
    map.get(key).and_then(|v| v.parse::<usize>().ok())
}

/// Local sibling of `term_builder::validate_known_options` used by the
/// parser-side `linear / bounded / constrain / nonnegative / nonpositive`
/// branches (which build their `ParsedTerm` here and never enter
/// `term_builder::build_smooth_basis`). Without this, typos like
/// `bounded(x, min=0, max=1, foo=bar)` silently succeed because the
/// `foo` key was just never read.
fn validate_known_term_options(
    term_name: &str,
    options: &BTreeMap<String, String>,
    known: &[&str],
    raw: &str,
) -> Result<(), String> {
    let known_set: std::collections::BTreeSet<&&str> = known.iter().collect();
    for key in options.keys() {
        if !known_set.contains(&key.as_str()) {
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

pub fn option_usize_any(map: &BTreeMap<String, String>, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(v) = option_usize(map, key) {
            return Some(v);
        }
    }
    None
}

/// Strict integer option: returns `Ok(None)` if not present, `Ok(Some(n))` if
/// it parses as a non-negative integer, and `Err(msg)` if the user supplied a
/// value that isn't a valid usize (negative, decimal, garbage). Without this
/// the lenient `option_usize` silently drops invalid values and reverts to
/// the default — `k=-1` and `k=1.5` were both accepted as "k not specified"
/// instead of being flagged as user mistakes.
pub fn option_usize_strict(
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

/// Strict variant of `option_usize_any` that errors on the first present-but-
/// unparseable key rather than silently falling through.
pub fn option_usize_any_strict(
    map: &BTreeMap<String, String>,
    keys: &[&str],
) -> Result<Option<usize>, String> {
    for key in keys {
        if let Some(v) = option_usize_strict(map, key)? {
            return Ok(Some(v));
        }
    }
    Ok(None)
}

pub fn option_f64(map: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    map.get(key).and_then(|v| v.parse::<f64>().ok())
}

/// Strict float option: `Ok(None)` if absent, `Ok(Some(n))` if parses as a
/// finite f64, `Err` if the user passed an unparseable value (rather than
/// silently dropping it like the lenient `option_f64`).
pub fn option_f64_strict(map: &BTreeMap<String, String>, key: &str) -> Result<Option<f64>, String> {
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

pub fn option_bool(map: &BTreeMap<String, String>, key: &str) -> Option<bool> {
    map.get(key)
        .and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => Some(true),
            "false" | "0" | "no" | "n" => Some(false),
            _ => None,
        })
}

pub fn strip_quotes(v: &str) -> &str {
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
    let min = parse_optional_f64_option_alias(options, &["min", "lower"], raw, "linear")?;
    let max = parse_optional_f64_option_alias(options, &["max", "upper"], raw, "linear")?;
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

fn parse_optional_f64_option_alias(
    options: &BTreeMap<String, String>,
    keys: &[&str],
    raw: &str,
    fn_label: &str,
) -> Result<Option<f64>, String> {
    let mut found: Option<(&str, f64)> = None;
    for key in keys {
        if let Some(value) = options.get(*key) {
            let parsed = value
                .parse::<f64>()
                .map_err(|err| FormulaDslError::InvalidArgument {
                    reason: format!(
                        "{fn_label}() argument '{key}' must be a finite number, got '{}': {err}: {raw}",
                        value
                    ),
                })?;
            if found.is_some() {
                return Err(FormulaDslError::IncompatibleTerm {
                    reason: format!(
                        "{fn_label}() cannot specify both '{}' and '{}': {raw}",
                        found.expect("present").0,
                        key
                    ),
                }
                .into());
            }
            found = Some((key, parsed));
        }
    }
    Ok(found.map(|(_, v)| v))
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
    if !unknown.is_empty() {
        let term_name = raw.split('(').next().unwrap_or("linkwiggle");
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
    let degree = option_usize(options, "degree").unwrap_or(defaults.degree);
    if degree < 1 {
        return Err(FormulaDslError::InvalidArgument {
            reason: format!("linkwiggle() requires degree >= 1: {raw}"),
        }
        .into());
    }
    let num_internal_knots =
        option_usize(options, "internal_knots").unwrap_or(defaults.num_internal_knots);
    if num_internal_knots == 0 {
        return Err(FormulaDslError::InvalidArgument {
            reason: format!("linkwiggle() requires internal_knots > 0: {raw}"),
        }
        .into());
    }
    let penalty_orders =
        parse_linkwiggle_penalty_orders(options.get("penalty_order").map(String::as_str))?;
    let double_penalty = option_bool(options, "double_penalty").unwrap_or(defaults.double_penalty);
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

fn parse_bounded_priorspec(
    options: &BTreeMap<String, String>,
    min: f64,
    max: f64,
    raw: &str,
) -> Result<BoundedCoefficientPriorSpec, String> {
    let prior_mode = options.get("prior").map(|s| s.to_ascii_lowercase());
    let pull = options.get("pull").map(|s| s.to_ascii_lowercase());
    let target = parse_optional_f64_option(options, "target", raw)?;
    let strength = parse_optional_f64_option(options, "strength", raw)?;

    let target_mode = target.is_some() || strength.is_some();
    if prior_mode.is_some() && pull.is_some() {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!("bounded() cannot combine prior=... with pull=...: {raw}"),
        }
        .into());
    }
    if prior_mode.is_some() && target_mode {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!("bounded() cannot combine prior=... with target/strength: {raw}"),
        }
        .into());
    }
    if pull.is_some() && target_mode {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!("bounded() cannot combine pull=... with target/strength: {raw}"),
        }
        .into());
    }

    if let Some(priorname) = prior_mode {
        return match priorname.as_str() {
            "none" => Ok(BoundedCoefficientPriorSpec::None),
            "uniform" | "log-jacobian" | "log_jacobian" | "jacobian" => {
                Ok(BoundedCoefficientPriorSpec::Uniform)
            }
            "center" => Ok(BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 }),
            _ => Err(FormulaDslError::InvalidArgument {
                reason: format!(
                    "bounded() prior must currently be one of none|uniform|log-jacobian|center, got '{}': {raw}",
                    priorname
                ),
            }
            .into()),
        };
    }

    if let Some(pull_mode) = pull {
        return match pull_mode.as_str() {
            "uniform" | "log-jacobian" | "log_jacobian" | "jacobian" => {
                Ok(BoundedCoefficientPriorSpec::Uniform)
            }
            "center" => Ok(BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 }),
            _ => Err(FormulaDslError::InvalidArgument {
                reason: format!(
                    "bounded() pull must currently be 'uniform'/'log-jacobian' or 'center', got '{}': {raw}",
                    pull_mode
                ),
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
        return Ok(BoundedCoefficientPriorSpec::Beta { a, b });
    }

    Ok(BoundedCoefficientPriorSpec::None)
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

pub fn parse_surv_response(lhs: &str) -> Result<Option<(String, String, String)>, FormulaDslError> {
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
            CallArgSpec::Positional(v) => Some(v.trim().to_string()),
            CallArgSpec::Named { .. } => None,
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if vars.len() != 3 {
        // Help users coming from R/mgcv (`Surv(time, status)` 2-arg form):
        // when they give two columns, name them explicitly in the error so
        // the fix ("prepend an entry column of zeros") is obvious.
        if vars.len() == 2 {
            return Err(FormulaDslError::InvalidArgument {
                reason: format!(
                    "Surv(...) needs three columns: Surv(entry, exit, event). \
                     Got `Surv({}, {})` — if these are (exit, event) from an mgcv-style \
                     left-truncation-free dataset, add a leading entry-time column of \
                     zeros and call `Surv(0_entry, {}, {})`.",
                    vars[0], vars[1], vars[0], vars[1]
                ),
            }
            .into());
        }
        return Err(FormulaDslError::InvalidArgument {
            reason: format!(
                "Surv(...) expects exactly three columns: Surv(entry, exit, event); got {}",
                vars.len()
            ),
        }
        .into());
    }
    Ok(Some((vars[0].clone(), vars[1].clone(), vars[2].clone())))
}

fn top_level_formula_separator(input: &str) -> Result<Option<usize>, String> {
    let mut depth = 0_i32;
    let mut in_single = false;
    let mut in_double = false;

    for (idx, ch) in input.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '(' | '[' | '{' if !in_single && !in_double => depth += 1,
            ')' | ']' | '}' if !in_single && !in_double && depth > 0 => depth -= 1,
            '~' if !in_single && !in_double && depth == 0 => return Ok(Some(idx)),
            _ => {}
        }
    }

    if in_single || in_double || depth != 0 {
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
        }
        .into());
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
    if !parsed_formula.logslope_surfaces.is_empty() && flag_name != "--logslope-formula" {
        return Err(FormulaDslError::IncompatibleTerm {
            reason: format!(
                "logslope(...) is only supported in --logslope-formula, not {flag_name}"
            ),
        }
        .into());
    }
    Ok(())
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
    let mut logslope_surfaces = Vec::<LogSlopeSurfaceSpec>::new();
    // Track seen-term-keys so we can reject exact duplicates like
    // `y ~ smooth(x) + smooth(x)` upfront — without this the duplicate
    // produces a rank-deficient design and the user has no idea why their
    // fit is over-parameterized.
    let mut seen_term_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for raw in parsed_dsl.rhs_terms {
        let t = raw.trim();
        if t.is_empty() || t == "1" {
            continue;
        }
        if t == "0" || t == "-1" {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: "formula terms '0'/'-1' (intercept removal) are not supported yet"
                    .to_string(),
            }
            .into());
        }
        // Normalize whitespace so `smooth(x)` and `smooth( x )` match.
        let key: String = t.split_whitespace().collect();
        if !seen_term_keys.insert(key.clone()) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason: format!(
                    "formula `{formula}` lists term `{t}` more than once. \
                     Duplicate terms produce a rank-deficient design; \
                     keep one copy or differentiate them (e.g. distinct k=, bs= options)."
                ),
            }
            .into());
        }
        match parse_term(t)? {
            ParsedTerm::LinkWiggle { options } => {
                if linkwiggle.is_some() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: "formula can include at most one linkwiggle(...) term".to_string(),
                    }
                    .into());
                }
                linkwiggle = Some(parse_linkwiggle_formulaspec(&options, t)?);
            }
            ParsedTerm::TimeWiggle { options } => {
                if timewiggle.is_some() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: "formula can include at most one timewiggle(...) term".to_string(),
                    }
                    .into());
                }
                timewiggle = Some(parse_linkwiggle_formulaspec(&options, t)?);
            }
            ParsedTerm::LinkConfig { options } => {
                if linkspec.is_some() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: "formula can include at most one link(...) term".to_string(),
                    }
                    .into());
                }
                linkspec = Some(parse_link_formulaspec(&options, t)?);
            }
            ParsedTerm::SurvivalConfig { options } => {
                if survivalspec.is_some() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: "formula can include at most one survmodel(...) term".to_string(),
                    }
                    .into());
                }
                survivalspec = Some(parse_survival_formulaspec(&options, t)?);
            }
            ParsedTerm::LogSlopeSurface { z_column, terms } => {
                logslope_surfaces.push(LogSlopeSurfaceSpec { z_column, terms });
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
        }
        .into());
    }
    Ok(ParsedFormula {
        response: lhs.to_string(),
        terms,
        logslope_surfaces,
        linkwiggle,
        timewiggle,
        linkspec,
        survivalspec,
    })
}

pub fn parse_term(raw: &str) -> Result<ParsedTerm, String> {
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

    let call = parse_function_call(raw).ok();
    if let Some(call) = call {
        let name = call.name.to_ascii_lowercase();
        let (vars, mut options) = split_call_args(&call);
        match name.as_str() {
            "constrain" | "constraint" | "box" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "constrain()/constraint()/box() expects exactly one variable: {raw}"
                        ),
                    }
                    .into());
                }
                validate_known_term_options(
                    "constrain",
                    &options,
                    &["min", "lower", "max", "upper"],
                    raw,
                )?;
                let (coefficient_min, coefficient_max) =
                    parse_linear_constraint_bounds(&options, raw)?;
                if coefficient_min.is_none() && coefficient_max.is_none() {
                    return Err(FormulaDslError::MalformedConfig {
                        reason: format!(
                            "constrain()/constraint()/box() requires at least one of min/lower/max/upper: {raw}"
                        ),
                    }
                    .into());
                }
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min,
                    coefficient_max,
                });
            }
            "nonnegative" | "nonnegative_coef" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("nonnegative() expects exactly one variable: {raw}"),
                    }
                    .into());
                }
                validate_known_term_options("nonnegative", &options, &[], raw)?;
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min: Some(0.0),
                    coefficient_max: None,
                });
            }
            "nonpositive" | "nonpositive_coef" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("nonpositive() expects exactly one variable: {raw}"),
                    }
                    .into());
                }
                validate_known_term_options("nonpositive", &options, &[], raw)?;
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
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
                    &["min", "max", "prior", "pull", "target", "strength"],
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
                let prior = parse_bounded_priorspec(&options, min, max, raw)?;
                return Ok(ParsedTerm::BoundedLinear {
                    name: vars[0].clone(),
                    min,
                    max,
                    prior,
                });
            }
            "group" | "re" => {
                if vars.len() != 1 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "group()/re() expects exactly one variable, got '{}': {raw}",
                            vars.join(",")
                        ),
                    }
                    .into());
                }
                return Ok(ParsedTerm::RandomEffect {
                    name: vars[0].clone(),
                });
            }
            "tensor" | "interaction" | "te" => {
                if vars.len() < 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "tensor()/interaction()/te() requires at least two variables: {raw}"
                        ),
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
                options.insert("type".to_string(), "tps".to_string());
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "smooth" | "s" | "cyclic" | "cc" | "cp" => {
                if vars.is_empty() {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!("smooth()/s() requires at least one variable: {raw}"),
                    }
                    .into());
                }
                if matches!(name.as_str(), "cyclic" | "periodic" | "cc" | "cp") {
                    options.insert("type".to_string(), "cyclic".to_string());
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
            "sphere" | "sos" | "spherical" => {
                if vars.len() != 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "sphere()/sos() expects exactly two variables: latitude and longitude; got {} in {raw}",
                            vars.len()
                        ),
                    }
                    .into());
                }
                options.insert("type".to_string(), "sphere".to_string());
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "sphere" | "s2" | "sos" => {
                if vars.len() != 2 {
                    return Err(format!(
                        "sphere()/s2()/sos() expects exactly two variables (lat, lon): {raw}"
                    ));
                }
                if matches!(name.as_str(), "cyclic" | "cc" | "cp") {
                    options.insert("type".to_string(), "cyclic".to_string());
                }
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
                options.insert("type".to_string(), "matern".to_string());
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
                if option_bool(&options, "cyclic").unwrap_or(false)
                    || option_bool(&options, "periodic").unwrap_or(false)
                {
                    options.insert("cyclic".to_string(), "true".to_string());
                }
                options.insert("type".to_string(), "duchon".to_string());
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
                options.insert("type".to_string(), "pca".to_string());
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
            "logslope" | "log_slope" | "log_slope_surface" => {
                validate_known_term_options("logslope", &options, &[], raw)?;
                if vars.len() < 2 {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "logslope() expects a z column followed by one or more RHS terms; add one logslope(z, ...) declaration per vector-z coordinate: {raw}"
                        ),
                    }
                    .into());
                }
                let z_column = vars[0].trim();
                if !is_exact_ident(z_column) {
                    return Err(FormulaDslError::InvalidArgument {
                        reason: format!(
                            "logslope() z column must be a bare column name, got `{z_column}` in {raw}"
                        ),
                    }
                    .into());
                }
                let rhs = vars[1..].join(" + ");
                let parsed = parse_formula(&format!("__logslope__ ~ {rhs}"))?;
                if !parsed.logslope_surfaces.is_empty() {
                    return Err(FormulaDslError::IncompatibleTerm {
                        reason: format!(
                            "logslope() declarations cannot be nested inside another logslope(): {raw}"
                        ),
                    }
                    .into());
                }
                validate_auxiliary_formula_controls(&parsed, "logslope()")?;
                return Ok(ParsedTerm::LogSlopeSurface {
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
                    &["min", "lower", "max", "upper"],
                    raw,
                )?;
                let (coefficient_min, coefficient_max) =
                    parse_linear_constraint_bounds(&options, raw)?;
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min,
                    coefficient_max,
                });
            }
            _ => {
                return Err(format!(
                    "unknown term function in '{raw}'. Supported: bounded(), linear(), constrain(), nonnegative(), nonpositive(), smooth(), cyclic()/cc()/cp(), thinplate(), tensor(), group(), matern(), duchon(), linkwiggle(), timewiggle(), link(), survmodel()"
                ));
            }
        }
    }

    let ident = raw.trim();
    if !is_exact_ident(ident) {
        return Err(FormulaDslError::UnknownIdentifier {
            reason: format!("unsupported top-level RHS term: {raw}"),
        }
        .into());
    }

    Ok(ParsedTerm::Linear {
        name: ident.to_string(),
        explicit: false,
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
            }
            .into());
        }
        let link = parse_linkname(inner)?;
        if !linkname_supports_joint_wiggle(link) {
            return Err(FormulaDslError::IncompatibleTerm {
                reason:
                    "flexible(...) does not support sas/beta-logistic links; wiggle is only supported for jointly fit standard links"
                        .to_string(),
            }
            .into());
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
            }
            .into());
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
        }
        .into());
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

pub fn parse_linkname(v: &str) -> Result<LinkFunction, FormulaDslError> {
    match v.trim() {
        "identity" => Ok(LinkFunction::Identity),
        "log" => Ok(LinkFunction::Log),
        "logit" | "binomial-logit" => Ok(LinkFunction::Logit),
        "probit" | "binomial-probit" => Ok(LinkFunction::Probit),
        "cloglog" | "binomial-cloglog" => Ok(LinkFunction::CLogLog),
        "sas" => Ok(LinkFunction::Sas),
        "beta-logistic" => Ok(LinkFunction::BetaLogistic),
        other => Err(FormulaDslError::UnknownIdentifier {
            reason: format!(
                "unsupported link type '{other}'; \
                 use one of identity|log|logit|probit|cloglog|binomial-logit|binomial-probit|binomial-cloglog|sas|beta-logistic|blended(...)/mixture(...) or flexible(...). \
                 Both `--link <type>` (CLI flag) and `link(type=<type>)` (formula term) accept the same set."
            ),
        }),
    }
}

pub fn parse_link_component(v: &str) -> Result<LinkComponent, String> {
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

pub fn parse_link_component_list(v: &str) -> Result<Vec<LinkComponent>, String> {
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

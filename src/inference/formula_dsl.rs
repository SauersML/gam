use std::collections::BTreeMap;

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

use crate::smooth::BoundedCoefficientPriorSpec;
use crate::types::{InverseLink, LinkComponent, LinkFunction};

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

primary = { function_call | ident | number | string_lit | "(" ~ expr ~ ")" }
function_call = { ident ~ "(" ~ arg_list? ~ ")" }
arg_list = { arg ~ ("," ~ arg)* }
arg = { named_arg | expr }
named_arg = { ident ~ "=" ~ expr }

ident = @{ ident_start ~ ident_continue* }
ident_start = _{ ASCII_ALPHA | "_" }
ident_continue = _{ ASCII_ALPHANUMERIC | "_" | "." }

number = @{
    "-"? ~ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)? ~ (("e" | "E") ~ ("+" | "-")? ~ ASCII_DIGIT+)?
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

pub fn parse_formula_dsl(formula: &str) -> Result<FormulaDslParse, String> {
    validate_balanced_delimiters(formula, "invalid formula syntax")?;
    let mut parsed = FormulaParser::parse(Rule::formula, formula)
        .map_err(|e| format!("invalid formula syntax: {e}"))?;
    let formula_pair = parsed
        .next()
        .ok_or_else(|| "invalid formula syntax: empty parse".to_string())?;

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

    let response_expr =
        response_expr.ok_or_else(|| "invalid formula: missing response expression".to_string())?;
    let rhs_terms = rhs_terms.ok_or_else(|| "invalid formula: missing RHS terms".to_string())?;
    if rhs_terms.is_empty() {
        return Err("formula has no usable terms".to_string());
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
                    '}' => '{',
                    _ => unreachable!(),
                };
                if stack.pop() != Some(expected) {
                    return Err(delimiter_balance_error(prefix));
                }
            }
            _ => {}
        }
    }

    if in_single || in_double || !stack.is_empty() {
        return Err(delimiter_balance_error(prefix));
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
                    return Err("formula RHS contains an empty term".to_string());
                }
                out.push(term.to_string());
                start = idx + 1;
            }
            _ => {}
        }
    }
    if in_single || in_double || depth != 0 {
        return Err("formula RHS has unbalanced quotes or parentheses".to_string());
    }
    let tail = text[start..].trim();
    if tail.is_empty() {
        return Err("formula RHS contains an empty term".to_string());
    }
    out.push(tail.to_string());
    Ok(out)
}

pub fn parse_function_call(input: &str) -> Result<FunctionCallSpec, String> {
    validate_balanced_delimiters(input, "invalid function call syntax")?;
    let mut parsed = FormulaParser::parse(Rule::top_function_call, input)
        .map_err(|e| format!("invalid function call syntax: {e}"))?;
    let top = parsed
        .next()
        .ok_or_else(|| "invalid function call syntax: empty parse".to_string())?;
    let call = top
        .into_inner()
        .find(|p| p.as_rule() == Rule::function_call)
        .ok_or_else(|| "invalid function call syntax: missing call".to_string())?;
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
                                .ok_or_else(|| "invalid named argument key".to_string())?
                                .as_str()
                                .trim()
                                .to_ascii_lowercase();
                            let value = ni
                                .next()
                                .ok_or_else(|| "invalid named argument value".to_string())?
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
    let name = name.ok_or_else(|| "invalid function call: missing name".to_string())?;
    Ok(FunctionCallSpec { name, args })
}

#[cfg(test)]
mod tests {
    use super::{CallArgSpec, parse_formula_dsl, parse_function_call};

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
    fn parse_formula_dsl_reports_unbalanced_parentheses() {
        let err = parse_formula_dsl("y ~ s(x, k=10").expect_err("expected parse failure");
        assert!(err.contains("unbalanced parentheses"));
    }

    #[test]
    fn parse_function_call_reports_unbalanced_parentheses() {
        let err = parse_function_call("s(x, k=10").expect_err("expected parse failure");
        assert!(err.contains("unbalanced parentheses"));
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
    LinkWiggleFormulaSpec {
        degree: 3,
        num_internal_knots: 10,
        penalty_orders: vec![1, 2, 3],
        double_penalty: true,
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
    pub linkwiggle: Option<LinkWiggleFormulaSpec>,
    pub timewiggle: Option<LinkWiggleFormulaSpec>,
    pub linkspec: Option<LinkFormulaSpec>,
    pub survivalspec: Option<SurvivalFormulaSpec>,
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

pub fn linkname_supports_joint_wiggle(link: LinkFunction) -> bool {
    !matches!(link, LinkFunction::Sas | LinkFunction::BetaLogistic)
}

pub fn linkchoice_supports_joint_wiggle(choice: &LinkChoice) -> bool {
    choice.mixture_components.is_none() && linkname_supports_joint_wiggle(choice.link)
}

pub fn inverse_link_supports_joint_wiggle(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(LinkFunction::Identity)
            | InverseLink::Standard(LinkFunction::Log)
            | InverseLink::Standard(LinkFunction::Logit)
            | InverseLink::Standard(LinkFunction::Probit)
            | InverseLink::Standard(LinkFunction::CLogLog)
    )
}

// ---------------------------------------------------------------------------
// Option-map helpers (shared by formula parsing and term construction)
// ---------------------------------------------------------------------------

pub fn option_usize(map: &BTreeMap<String, String>, key: &str) -> Option<usize> {
    map.get(key).and_then(|v| v.parse::<usize>().ok())
}

pub fn option_usize_any(map: &BTreeMap<String, String>, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(v) = option_usize(map, key) {
            return Some(v);
        }
    }
    None
}

pub fn option_f64(map: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    map.get(key).and_then(|v| v.parse::<f64>().ok())
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
        return Err(format!(
            "linear coefficient constraints require finite min <= max, got min={min}, max={max}: {raw}"
        ));
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
        .ok_or_else(|| format!("bounded() is missing required '{key}' argument: {raw}"))?;
    value.parse::<f64>().map_err(|_| {
        format!(
            "bounded() argument '{key}' must be a finite number, got '{}': {raw}",
            value
        )
    })
}

fn parse_optional_f64_option(
    options: &BTreeMap<String, String>,
    key: &str,
    raw: &str,
) -> Result<Option<f64>, String> {
    match options.get(key) {
        Some(value) => value.parse::<f64>().map(Some).map_err(|_| {
            format!(
                "bounded() argument '{key}' must be a finite number, got '{}': {raw}",
                value
            )
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
            let parsed = value.parse::<f64>().map_err(|_| {
                format!(
                    "{fn_label}() argument '{key}' must be a finite number, got '{}': {raw}",
                    value
                )
            })?;
            if found.is_some() {
                return Err(format!(
                    "{fn_label}() cannot specify both '{}' and '{}': {raw}",
                    found.expect("present").0,
                    key
                ));
            }
            found = Some((key, parsed));
        }
    }
    Ok(found.map(|(_, v)| v))
}

fn parse_linkwiggle_penalty_orders(raw: Option<&str>) -> Result<Vec<usize>, String> {
    let Some(raw) = raw.map(str::trim) else {
        return Ok(vec![1, 2, 3]);
    };
    if raw.is_empty() {
        return Ok(vec![1, 2, 3]);
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
                return Err(format!(
                    "invalid linkwiggle penalty_order '{t}'; use all|slope|curvature|curvature-change or 1/2/3"
                ));
            }
        }
    }
    if out.is_empty() {
        out.extend([1, 2, 3]);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

pub fn parse_linkwiggle_formulaspec(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<LinkWiggleFormulaSpec, String> {
    let degree = option_usize(options, "degree").unwrap_or(3);
    if degree < 1 {
        return Err(format!("linkwiggle() requires degree >= 1: {raw}"));
    }
    let num_internal_knots = option_usize(options, "internal_knots").unwrap_or(7);
    if num_internal_knots == 0 {
        return Err(format!("linkwiggle() requires internal_knots > 0: {raw}"));
    }
    let penalty_orders =
        parse_linkwiggle_penalty_orders(options.get("penalty_order").map(String::as_str))?;
    let double_penalty = option_bool(options, "double_penalty").unwrap_or(true);
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
        .ok_or_else(|| format!("link() requires type=<link-name>: {raw}"))?;
    if link.is_empty() {
        return Err(format!("link() requires a non-empty type: {raw}"));
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
        return Err(format!(
            "survmodel() requires at least one named option (e.g., spec=..., distribution=...): {raw}"
        ));
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
    let beta_a = parse_optional_f64_option(options, "beta_a", raw)?;
    let beta_b = parse_optional_f64_option(options, "beta_b", raw)?;
    let target = parse_optional_f64_option(options, "target", raw)?;
    let strength = parse_optional_f64_option(options, "strength", raw)?;

    let explicit_beta = beta_a.is_some() || beta_b.is_some();
    let target_mode = target.is_some() || strength.is_some();
    if prior_mode.is_some() && pull.is_some() {
        return Err(format!(
            "bounded() cannot combine prior=... with pull=...: {raw}"
        ));
    }
    if prior_mode.is_some() && explicit_beta {
        return Err(format!(
            "bounded() cannot combine prior=... with beta_a/beta_b: {raw}"
        ));
    }
    if prior_mode.is_some() && target_mode {
        return Err(format!(
            "bounded() cannot combine prior=... with target/strength: {raw}"
        ));
    }
    if pull.is_some() && explicit_beta {
        return Err(format!(
            "bounded() cannot combine pull=... with beta_a/beta_b: {raw}"
        ));
    }
    if pull.is_some() && target_mode {
        return Err(format!(
            "bounded() cannot combine pull=... with target/strength: {raw}"
        ));
    }
    if explicit_beta && target_mode {
        return Err(format!(
            "bounded() cannot combine beta_a/beta_b with target/strength: {raw}"
        ));
    }

    if let Some(priorname) = prior_mode {
        return match priorname.as_str() {
            "none" => Ok(BoundedCoefficientPriorSpec::None),
            "uniform" | "log-jacobian" | "log_jacobian" | "jacobian" => {
                Ok(BoundedCoefficientPriorSpec::Uniform)
            }
            "center" => Ok(BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 }),
            _ => Err(format!(
                "bounded() prior must currently be one of none|uniform|log-jacobian|center, got '{}': {raw}",
                priorname
            )),
        };
    }

    if let Some(pull_mode) = pull {
        return match pull_mode.as_str() {
            "uniform" | "log-jacobian" | "log_jacobian" | "jacobian" => {
                Ok(BoundedCoefficientPriorSpec::Uniform)
            }
            "center" => Ok(BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 }),
            _ => Err(format!(
                "bounded() pull must currently be 'uniform'/'log-jacobian' or 'center', got '{}': {raw}",
                pull_mode
            )),
        };
    }

    if explicit_beta {
        let a = beta_a.ok_or_else(|| format!("bounded() beta_a is required with beta_b: {raw}"))?;
        let b = beta_b.ok_or_else(|| format!("bounded() beta_b is required with beta_a: {raw}"))?;
        if !a.is_finite() || !b.is_finite() || a < 1.0 || b < 1.0 {
            return Err(format!(
                "bounded() beta_a and beta_b must be finite and >= 1: {raw}"
            ));
        }
        return Ok(BoundedCoefficientPriorSpec::Beta { a, b });
    }

    if target_mode {
        let targetvalue =
            target.ok_or_else(|| format!("bounded() target is required with strength: {raw}"))?;
        let strengthvalue =
            strength.ok_or_else(|| format!("bounded() strength is required with target: {raw}"))?;
        if !(min < targetvalue && targetvalue < max) {
            return Err(format!(
                "bounded() target must lie strictly inside ({min}, {max}): {raw}"
            ));
        }
        if !strengthvalue.is_finite() || strengthvalue <= 0.0 {
            return Err(format!("bounded() strength must be finite and > 0: {raw}"));
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
        return Err("formula right-hand side cannot be empty".to_string());
    }
    Ok(parsed.rhs_terms.join(" + "))
}

pub fn parse_surv_response(lhs: &str) -> Result<Option<(String, String, String)>, String> {
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
        return Err(format!(
            "Surv(...) expects exactly three columns: Surv(entry, exit, event); got {}",
            vars.len()
        ));
    }
    Ok(Some((vars[0].clone(), vars[1].clone(), vars[2].clone())))
}

pub fn normalizenoise_formula(noise: &str, response: &str) -> String {
    if noise.contains('~') {
        noise.to_string()
    } else {
        format!("{response} ~ {noise}")
    }
}

pub fn parse_matching_auxiliary_formula(
    formula: &str,
    response: &str,
    flag_name: &str,
) -> Result<(String, ParsedFormula), String> {
    let normalized_formula = normalizenoise_formula(formula, response);
    let parsed_formula = parse_formula(&normalized_formula)?;
    let responses_match = if parsed_formula.response == response {
        true
    } else {
        match (
            parse_surv_response(response)?,
            parse_surv_response(&parsed_formula.response)?,
        ) {
            (Some(expected), Some(actual)) => expected == actual,
            _ => false,
        }
    };
    if !responses_match {
        return Err(format!(
            "{flag_name} must use the same response expression as the main formula"
        ));
    }
    Ok((normalized_formula, parsed_formula))
}

pub fn validate_auxiliary_formula_controls(
    parsed_formula: &ParsedFormula,
    flag_name: &str,
) -> Result<(), String> {
    if parsed_formula.linkwiggle.is_some() {
        return Err(format!(
            "linkwiggle(...) is only supported in the main formula, not {flag_name}"
        ));
    }
    if parsed_formula.timewiggle.is_some() {
        return Err(format!(
            "timewiggle(...) is only supported in the main survival formula, not {flag_name}"
        ));
    }
    if parsed_formula.linkspec.is_some() {
        return Err(format!(
            "link(...) is only supported in the main formula, not {flag_name}"
        ));
    }
    if parsed_formula.survivalspec.is_some() {
        return Err(format!(
            "survmodel(...) is only supported in the main survival formula, not {flag_name}"
        ));
    }
    Ok(())
}

pub fn parse_formula(formula: &str) -> Result<ParsedFormula, String> {
    let parsed_dsl = parse_formula_dsl(formula)?;
    let lhs = parsed_dsl.response_expr.trim();
    if lhs.is_empty() {
        return Err("formula response (left-hand side) cannot be empty".to_string());
    }
    let mut terms = Vec::<ParsedTerm>::new();
    let mut linkwiggle: Option<LinkWiggleFormulaSpec> = None;
    let mut timewiggle: Option<LinkWiggleFormulaSpec> = None;
    let mut linkspec: Option<LinkFormulaSpec> = None;
    let mut survivalspec: Option<SurvivalFormulaSpec> = None;
    for raw in parsed_dsl.rhs_terms {
        let t = raw.trim();
        if t.is_empty() || t == "1" {
            continue;
        }
        if t == "0" || t == "-1" {
            return Err(
                "formula terms '0'/'-1' (intercept removal) are not supported yet".to_string(),
            );
        }
        match parse_term(t)? {
            ParsedTerm::LinkWiggle { options } => {
                if linkwiggle.is_some() {
                    return Err("formula can include at most one linkwiggle(...) term".to_string());
                }
                linkwiggle = Some(parse_linkwiggle_formulaspec(&options, t)?);
            }
            ParsedTerm::TimeWiggle { options } => {
                if timewiggle.is_some() {
                    return Err("formula can include at most one timewiggle(...) term".to_string());
                }
                timewiggle = Some(parse_linkwiggle_formulaspec(&options, t)?);
            }
            ParsedTerm::LinkConfig { options } => {
                if linkspec.is_some() {
                    return Err("formula can include at most one link(...) term".to_string());
                }
                linkspec = Some(parse_link_formulaspec(&options, t)?);
            }
            ParsedTerm::SurvivalConfig { options } => {
                if survivalspec.is_some() {
                    return Err("formula can include at most one survmodel(...) term".to_string());
                }
                survivalspec = Some(parse_survival_formulaspec(&options, t)?);
            }
            other => terms.push(other),
        }
    }

    Ok(ParsedFormula {
        response: lhs.to_string(),
        terms,
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
                    return Err(format!(
                        "constrain()/constraint()/box() expects exactly one variable: {raw}"
                    ));
                }
                let (coefficient_min, coefficient_max) =
                    parse_linear_constraint_bounds(&options, raw)?;
                if coefficient_min.is_none() && coefficient_max.is_none() {
                    return Err(format!(
                        "constrain()/constraint()/box() requires at least one of min/lower/max/upper: {raw}"
                    ));
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
                    return Err(format!("nonnegative() expects exactly one variable: {raw}"));
                }
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min: Some(0.0),
                    coefficient_max: None,
                });
            }
            "nonpositive" | "nonpositive_coef" => {
                if vars.len() != 1 {
                    return Err(format!("nonpositive() expects exactly one variable: {raw}"));
                }
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min: None,
                    coefficient_max: Some(0.0),
                });
            }
            "bounded" => {
                if vars.len() != 1 {
                    return Err(format!("bounded() expects exactly one variable: {raw}"));
                }
                let min = parse_required_f64_option(&options, "min", raw)?;
                let max = parse_required_f64_option(&options, "max", raw)?;
                if !min.is_finite() || !max.is_finite() || min >= max {
                    return Err(format!(
                        "bounded() requires finite min < max, got min={min}, max={max}: {raw}"
                    ));
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
                    return Err(format!(
                        "group()/re() expects exactly one variable, got '{}': {raw}",
                        vars.join(",")
                    ));
                }
                return Ok(ParsedTerm::RandomEffect {
                    name: vars[0].clone(),
                });
            }
            "tensor" | "interaction" | "te" => {
                if vars.len() < 2 {
                    return Err(format!(
                        "tensor()/interaction()/te() requires at least two variables: {raw}"
                    ));
                }
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::Te,
                    options,
                });
            }
            "thinplate" | "thin_plate" | "tps" => {
                if vars.len() < 2 {
                    return Err(format!(
                        "thinplate()/thin_plate()/tps() requires at least two variables: {raw}"
                    ));
                }
                options.insert("type".to_string(), "tps".to_string());
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "smooth" | "s" => {
                if vars.is_empty() {
                    return Err(format!(
                        "smooth()/s() requires at least one variable: {raw}"
                    ));
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
                    return Err(format!("matern() requires at least one variable: {raw}"));
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
                    return Err(format!("duchon() requires at least one variable: {raw}"));
                }
                options.insert("type".to_string(), "duchon".to_string());
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "linkwiggle" => {
                if !vars.is_empty() {
                    return Err(format!(
                        "linkwiggle() takes named options only; positional args are not supported: {raw}"
                    ));
                }
                return Ok(ParsedTerm::LinkWiggle { options });
            }
            "timewiggle" => {
                if !vars.is_empty() {
                    return Err(format!(
                        "timewiggle() takes named options only; positional args are not supported: {raw}"
                    ));
                }
                return Ok(ParsedTerm::TimeWiggle { options });
            }
            "link" => {
                if !vars.is_empty() {
                    return Err(format!(
                        "link() takes named options only; positional args are not supported: {raw}"
                    ));
                }
                return Ok(ParsedTerm::LinkConfig { options });
            }
            "survmodel" => {
                if !vars.is_empty() {
                    return Err(format!(
                        "survmodel() takes named options only; positional args are not supported: {raw}"
                    ));
                }
                return Ok(ParsedTerm::SurvivalConfig { options });
            }
            "linear" => {
                if vars.len() != 1 {
                    return Err(format!("linear() expects exactly one variable: {raw}"));
                }
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
                    "unknown term function in '{raw}'. Supported: bounded(), linear(), constrain(), nonnegative(), nonpositive(), smooth(), thinplate(), tensor(), group(), matern(), duchon(), linkwiggle(), timewiggle(), link(), survmodel()"
                ));
            }
        }
    }

    Ok(ParsedTerm::Linear {
        name: raw.trim().to_string(),
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
) -> Result<Option<LinkChoice>, String> {
    if raw.is_none() && !flexible_flag {
        return Ok(None);
    }
    let Some(v) = raw else {
        return Ok(Some(LinkChoice {
            mode: LinkMode::Flexible,
            link: LinkFunction::Logit,
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
            return Err(
                "flexible(...) does not support blended(...)/mixture(...) links; wiggle is only supported for jointly fit standard links"
                    .to_string(),
            );
        }
        let link = parse_linkname(inner)?;
        if !linkname_supports_joint_wiggle(link) {
            return Err(
                "flexible(...) does not support sas/beta-logistic links; wiggle is only supported for jointly fit standard links"
                    .to_string(),
            );
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
            return Err(
                    "--flexible-link cannot be combined with --link blended(...)/mixture(...); blended inverse links are not flexible-link mode"
                        .to_string(),
            );
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
        return Err(
            "--flexible-link does not support sas/beta-logistic links; wiggle is only supported for jointly fit standard links"
                .to_string(),
        );
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

pub fn parse_linkname(v: &str) -> Result<LinkFunction, String> {
    match v.trim() {
        "identity" => Ok(LinkFunction::Identity),
        "log" => Ok(LinkFunction::Log),
        "logit" | "binomial-logit" => Ok(LinkFunction::Logit),
        "probit" | "binomial-probit" => Ok(LinkFunction::Probit),
        "cloglog" | "binomial-cloglog" => Ok(LinkFunction::CLogLog),
        "sas" => Ok(LinkFunction::Sas),
        "beta-logistic" => Ok(LinkFunction::BetaLogistic),
        other => Err(format!(
            "unsupported --link '{other}'; use identity|log|logit|probit|cloglog|binomial-logit|binomial-probit|binomial-cloglog|sas|beta-logistic|blended(...)/mixture(...) or flexible(...)"
        )),
    }
}

pub fn linkname(link: LinkFunction) -> &'static str {
    match link {
        LinkFunction::Identity => "identity",
        LinkFunction::Log => "log",
        LinkFunction::Logit => "logit",
        LinkFunction::Probit => "probit",
        LinkFunction::CLogLog => "cloglog",
        LinkFunction::Sas => "sas",
        LinkFunction::BetaLogistic => "beta-logistic",
    }
}

pub fn parse_link_component(v: &str) -> Result<LinkComponent, String> {
    match v.trim() {
        "logit" => Ok(LinkComponent::Logit),
        "probit" => Ok(LinkComponent::Probit),
        "cloglog" => Ok(LinkComponent::CLogLog),
        "loglog" => Ok(LinkComponent::LogLog),
        "cauchit" => Ok(LinkComponent::Cauchit),
        other => Err(format!(
            "unsupported blended-link component '{other}'; use probit|logit|cloglog|loglog|cauchit"
        )),
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
            return Err("blended(...) cannot contain duplicate components".to_string());
        }
        out.push(comp);
    }
    if out.len() < 2 {
        return Err("blended(...) requires at least two components".to_string());
    }
    Ok(out)
}

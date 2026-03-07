use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

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
    fn parses_function_call_with_named_and_positional_args() {
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

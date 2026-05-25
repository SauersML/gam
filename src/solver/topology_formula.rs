//! Formula-string surgery for `select_topology` AUTO substitution.
//!
//! Given a base formula containing exactly one `s(..., type=AUTO)` smooth
//! term and a typed description of the candidate topology, returns the
//! rewritten formula with the AUTO term replaced by the candidate-specific
//! `s(...)` or `te(...)` term.
//!
//! The Python side stays responsible for translating its `Smooth` subclass
//! instance into the typed `CandidateTopology` description (because the
//! `Smooth` objects live in Python). The string surgery — paren matching,
//! comma splitting, option emission, dimension checks — lives here.

use serde::Deserialize;

/// Typed description of a candidate topology supplied by the Python caller.
///
/// The tag is `kind` with snake_case discriminants matching the variants
/// below.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateTopology {
    PeriodicSplineCurve {
        n_knots: usize,
        degree: usize,
        penalty_order: usize,
    },
    Sphere {
        n_centers: usize,
        penalty_order: usize,
        kernel: String,
        radians: bool,
    },
    Tensor {
        k: Vec<usize>,
        periodic: Vec<bool>,
        /// Optional explicit period strings, one per marginal. `None` for the
        /// default `2*pi`/`None` derivation; `Some` with one entry per
        /// marginal to override (each entry is either `None` or a literal
        /// string emitted verbatim).
        periods: Option<Vec<Option<String>>>,
    },
    Duchon {
        m: usize,
        /// Set when the user passed a bare `int` for centers — we forward it
        /// as a `centers=` option. When the Python side passes an array it
        /// sets this to `None` and the size option is omitted.
        centers_int: Option<usize>,
        /// Marker that this Duchon spec is per-axis periodic. The formula
        /// AUTO path cannot represent per-axis periodic Duchon candidates;
        /// emit the same error the Python printer raised.
        per_axis_periodic: bool,
        length_scale: Option<f64>,
        required_dim: Option<usize>,
    },
}

/// Emit the candidate-specific smooth term (header + required dim).
struct TopologyTerm {
    call: &'static str,
    options: Vec<String>,
    required_dim: Option<usize>,
}

const SIZE_OPTION_KEYS: &[&str] = &[
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "centers",
    "knots",
];
const STRUCTURAL_OPTION_KEYS: &[&str] = &[
    "periodic", "cyclic", "bc", "period", "periods", "origin", "origins",
];

/// Replace the unique `s(..., type=AUTO)` smooth in `base_formula` with the
/// candidate-specific term, returning the rewritten formula.
///
/// When the candidate's required input dimension does not match the AUTO
/// term's covariate count and `strict_dimension` is true, returns an error;
/// when `strict_dimension` is false, returns `Ok(None)` so callers can skip
/// the candidate.
pub fn assemble_candidate_formula(
    base_formula: &str,
    candidate: &CandidateTopology,
    strict_dimension: bool,
) -> Result<Option<String>, String> {
    let auto_span = find_auto_smooth_call(base_formula)?;
    let term_text = &base_formula[auto_span.start..auto_span.end];
    let (vars, user_options, option_keys) = auto_call_parts(term_text)?;
    let topo_term = topology_term(candidate, &option_keys)?;
    if let Some(required) = topo_term.required_dim {
        if vars.len() != required {
            let plural = if vars.len() == 1 { "" } else { "s" };
            let message = format!(
                "{} needs {}-D covariate; AUTO smooth has {} covariate{} ({})",
                candidate_label(candidate),
                required,
                vars.len(),
                plural,
                vars.join(", "),
            );
            if strict_dimension {
                return Err(message);
            }
            return Ok(None);
        }
    }
    let mut candidate_args: Vec<String> =
        Vec::with_capacity(vars.len() + user_options.len() + topo_term.options.len());
    candidate_args.extend(vars.iter().cloned());
    candidate_args.extend(user_options.iter().cloned());
    candidate_args.extend(topo_term.options.iter().cloned());
    let candidate_term = format!("{}({})", topo_term.call, candidate_args.join(", "));
    let mut out = String::with_capacity(base_formula.len() + candidate_term.len());
    out.push_str(&base_formula[..auto_span.start]);
    out.push_str(&candidate_term);
    out.push_str(&base_formula[auto_span.end..]);
    Ok(Some(out))
}

fn candidate_label(candidate: &CandidateTopology) -> &'static str {
    match candidate {
        CandidateTopology::PeriodicSplineCurve { .. } => "circle",
        CandidateTopology::Sphere { .. } => "sphere",
        CandidateTopology::Tensor { .. } => "tensor",
        CandidateTopology::Duchon { .. } => "duchon",
    }
}

#[derive(Debug, Clone, Copy)]
struct AutoSpan {
    start: usize,
    end: usize,
}

fn find_auto_smooth_call(formula: &str) -> Result<AutoSpan, String> {
    let bytes = formula.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        let Some(call_start) = find_smooth_call(formula, idx) else {
            break;
        };
        let open_paren = call_open_paren(formula, call_start);
        let close_paren = matching_paren(formula, open_paren)?;
        let term = &formula[call_start..=close_paren];
        if contains_type_auto(term) {
            return Ok(AutoSpan {
                start: call_start,
                end: close_paren + 1,
            });
        }
        idx = close_paren + 1;
    }
    Err("select_topology requires one s(..., type=AUTO) smooth term".to_string())
}

/// Find the next `s(` or `smooth(` call at or after `from`. Returns the byte
/// index of the leading identifier, or `None` if there is no further match.
fn find_smooth_call(formula: &str, from: usize) -> Option<usize> {
    let bytes = formula.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        // Identifier must be preceded by a non-identifier character (word
        // boundary), matching the Python `\b` semantics.
        let start = i;
        let ch = bytes[i];
        if !is_alpha(ch) {
            i += 1;
            continue;
        }
        if start > 0 && is_ident_byte(bytes[start - 1]) {
            // Mid-identifier; skip the whole identifier.
            while i < bytes.len() && is_ident_byte(bytes[i]) {
                i += 1;
            }
            continue;
        }
        // Read the identifier.
        let mut j = i;
        while j < bytes.len() && is_ident_byte(bytes[j]) {
            j += 1;
        }
        let ident = &formula[i..j];
        // Skip whitespace before `(`.
        let mut k = j;
        while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t') {
            k += 1;
        }
        let lower = ident.to_ascii_lowercase();
        if (lower == "s" || lower == "smooth") && k < bytes.len() && bytes[k] == b'(' {
            return Some(start);
        }
        i = j;
    }
    None
}

fn call_open_paren(formula: &str, call_start: usize) -> usize {
    // Walk forward from `call_start` to the first `(`.
    let bytes = formula.as_bytes();
    let mut i = call_start;
    while i < bytes.len() && bytes[i] != b'(' {
        i += 1;
    }
    i
}

fn matching_paren(text: &str, open_paren: usize) -> Result<usize, String> {
    let bytes = text.as_bytes();
    let mut depth: i32 = 1;
    let mut quote: Option<u8> = None;
    let mut i = open_paren + 1;
    while i < bytes.len() {
        let ch = bytes[i];
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
        } else if ch == b'\'' || ch == b'"' {
            quote = Some(ch);
        } else if ch == b'(' {
            depth += 1;
        } else if ch == b')' {
            depth -= 1;
            if depth == 0 {
                return Ok(i);
            }
        }
        i += 1;
    }
    Err("select_topology: AUTO smooth has unbalanced parentheses".to_string())
}

fn split_top_level_args(arg_text: &str) -> Vec<String> {
    let bytes = arg_text.as_bytes();
    let mut args: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut depth: i32 = 0;
    let mut quote: Option<u8> = None;
    for (i, &ch) in bytes.iter().enumerate() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
        } else if ch == b'\'' || ch == b'"' {
            quote = Some(ch);
        } else if matches!(ch, b'(' | b'[' | b'{') {
            depth += 1;
        } else if matches!(ch, b')' | b']' | b'}') {
            depth -= 1;
        } else if ch == b',' && depth == 0 {
            args.push(arg_text[start..i].trim().to_string());
            start = i + 1;
        }
    }
    let tail = arg_text[start..].trim();
    if !tail.is_empty() {
        args.push(tail.to_string());
    }
    args
}

fn auto_call_parts(term: &str) -> Result<(Vec<String>, Vec<String>, Vec<String>), String> {
    let open = term
        .find('(')
        .ok_or_else(|| "select_topology: AUTO smooth has no argument list".to_string())?;
    // Strip the closing `)`.
    if !term.ends_with(')') {
        return Err("select_topology: AUTO smooth must end with ')'".to_string());
    }
    let arg_text = &term[open + 1..term.len() - 1];
    let args = split_top_level_args(arg_text);
    let mut vars: Vec<String> = Vec::new();
    let mut options: Vec<String> = Vec::new();
    let mut option_keys: Vec<String> = Vec::new();
    for arg in &args {
        let key = named_arg_key(arg);
        match key {
            None => vars.push(arg.clone()),
            Some(k) if k == "type" => continue,
            Some(k) => {
                option_keys.push(k.clone());
                if STRUCTURAL_OPTION_KEYS.contains(&k.as_str()) {
                    continue;
                }
                options.push(arg.clone());
            }
        }
    }
    if vars.is_empty() {
        return Err("select_topology: AUTO smooth must have at least one covariate".to_string());
    }
    Ok((vars, options, option_keys))
}

fn named_arg_key(arg: &str) -> Option<String> {
    let bytes = arg.as_bytes();
    let mut depth: i32 = 0;
    let mut quote: Option<u8> = None;
    for (i, &ch) in bytes.iter().enumerate() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
        } else if ch == b'\'' || ch == b'"' {
            quote = Some(ch);
        } else if matches!(ch, b'(' | b'[' | b'{') {
            depth += 1;
        } else if matches!(ch, b')' | b']' | b'}') {
            depth -= 1;
        } else if ch == b'=' && depth == 0 {
            return Some(arg[..i].trim().to_ascii_lowercase());
        }
    }
    None
}

fn contains_type_auto(term: &str) -> bool {
    // Match the Python regex \btype\s*=\s*(['"]?)AUTO\1(?=\s*(?:,|\)))
    let bytes = term.as_bytes();
    let lower = term.to_ascii_lowercase();
    let needle = "type";
    let mut search_from = 0usize;
    while let Some(pos_rel) = lower[search_from..].find(needle) {
        let pos = search_from + pos_rel;
        // Word boundary before `type`.
        if pos > 0 && is_ident_byte(bytes[pos - 1]) {
            search_from = pos + needle.len();
            continue;
        }
        // After `type`, allow optional whitespace, then `=`, then optional
        // whitespace, then optional quote, then `AUTO`, then matching
        // (optional) quote, then a lookahead at whitespace+`,` or `)`.
        let mut j = pos + needle.len();
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'=' {
            search_from = pos + needle.len();
            continue;
        }
        j += 1;
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        let quote = if j < bytes.len() && (bytes[j] == b'\'' || bytes[j] == b'"') {
            let q = bytes[j];
            j += 1;
            Some(q)
        } else {
            None
        };
        // Compare AUTO case-insensitively.
        if j + 4 > bytes.len() {
            return false;
        }
        let auto = &term[j..j + 4];
        if !auto.eq_ignore_ascii_case("AUTO") {
            search_from = pos + needle.len();
            continue;
        }
        j += 4;
        if let Some(q) = quote {
            if j >= bytes.len() || bytes[j] != q {
                search_from = pos + needle.len();
                continue;
            }
            j += 1;
        }
        let mut k = j;
        while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t') {
            k += 1;
        }
        if k < bytes.len() && (bytes[k] == b',' || bytes[k] == b')') {
            return true;
        }
        search_from = pos + needle.len();
    }
    false
}

fn topology_term(
    candidate: &CandidateTopology,
    option_keys: &[String],
) -> Result<TopologyTerm, String> {
    let has_size = option_keys
        .iter()
        .any(|key| SIZE_OPTION_KEYS.contains(&key.as_str()));
    match candidate {
        CandidateTopology::PeriodicSplineCurve {
            n_knots,
            degree,
            penalty_order,
        } => {
            let mut options: Vec<String> = vec!["type=cyclic".to_string()];
            if !has_size {
                options.push(format!("k={n_knots}"));
            }
            if *degree != 3 {
                options.push(format!("degree={degree}"));
            }
            if *penalty_order != 2 {
                options.push(format!("penalty_order={penalty_order}"));
            }
            Ok(TopologyTerm {
                call: "s",
                options,
                required_dim: Some(1),
            })
        }
        CandidateTopology::Sphere {
            n_centers,
            penalty_order,
            kernel,
            radians,
        } => {
            let mut options: Vec<String> = vec!["type=sphere".to_string()];
            if !has_size {
                options.push(format!("centers={n_centers}"));
            }
            if *penalty_order != 2 {
                options.push(format!("penalty_order={penalty_order}"));
            }
            if kernel != "sobolev" {
                options.push(format!("kernel={}", python_quote(kernel)));
            }
            if *radians {
                options.push("radians=true".to_string());
            }
            Ok(TopologyTerm {
                call: "s",
                options,
                required_dim: Some(2),
            })
        }
        CandidateTopology::Tensor {
            k,
            periodic,
            periods,
        } => {
            let mut options: Vec<String> = Vec::new();
            if !has_size && !k.is_empty() {
                options.push(format!("k={}", format_int_list(k)));
            }
            let any_periodic = periodic.iter().any(|p| *p);
            if any_periodic {
                options.push(format!("periodic={}", format_bool_list(periodic)));
                let owned_periods: Vec<Option<String>> = match periods {
                    Some(values) => values.clone(),
                    None => periodic
                        .iter()
                        .map(|p| if *p { Some("2*pi".to_string()) } else { None })
                        .collect(),
                };
                options.push(format!("period={}", format_period_list(&owned_periods)));
            }
            options.push("identifiability=sum_tozero".to_string());
            Ok(TopologyTerm {
                call: "te",
                options,
                required_dim: Some(periodic.len()),
            })
        }
        CandidateTopology::Duchon {
            m,
            centers_int,
            per_axis_periodic,
            length_scale,
            required_dim,
        } => {
            if *per_axis_periodic {
                return Err(
                    "select_topology cannot fit per-axis periodic Duchon candidates \
                     through the formula AUTO path; use topology.Cylinder or \
                     topology.Torus tensor candidates"
                        .to_string(),
                );
            }
            let order = duchon_formula_order(*m);
            let mut options: Vec<String> =
                vec!["type=duchon".to_string(), format!("order={order}")];
            if !has_size {
                if let Some(c) = centers_int {
                    options.push(format!("centers={c}"));
                }
            }
            if let Some(ls) = length_scale {
                options.push(format!("length_scale={}", python_repr_float(*ls)));
            }
            Ok(TopologyTerm {
                call: "s",
                options,
                required_dim: *required_dim,
            })
        }
    }
}

fn duchon_formula_order(m: usize) -> usize {
    m.saturating_sub(1)
}

fn format_int_list(values: &[usize]) -> String {
    let mut out = String::from("[");
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

fn format_bool_list(values: &[bool]) -> String {
    let mut out = String::from("[");
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(if *value { "true" } else { "false" });
    }
    out.push(']');
    out
}

fn format_period_list(values: &[Option<String>]) -> String {
    let mut out = String::from("[");
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        match value {
            None => out.push_str("None"),
            Some(text) => out.push_str(text),
        }
    }
    out.push(']');
    out
}

fn python_quote(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('\'', "\\'");
    format!("'{}'", escaped)
}

/// Mirror Python's `repr(float)` enough for the formula round-trip: emit at
/// least one decimal point or exponent so the formula parser does not treat
/// the number as an integer literal.
fn python_repr_float(value: f64) -> String {
    let mut text = format!("{}", value);
    if !text.contains('.') && !text.contains('e') && !text.contains('E') && value.is_finite() {
        text.push_str(".0");
    }
    text
}

fn is_alpha(ch: u8) -> bool {
    ch.is_ascii_alphabetic() || ch == b'_'
}

fn is_ident_byte(ch: u8) -> bool {
    ch.is_ascii_alphanumeric() || ch == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_spline_curve_substitution() {
        let candidate = CandidateTopology::PeriodicSplineCurve {
            n_knots: 20,
            degree: 3,
            penalty_order: 2,
        };
        let out = assemble_candidate_formula("y ~ s(t, type=AUTO)", &candidate, true)
            .unwrap()
            .unwrap();
        assert_eq!(out, "y ~ s(t, type=cyclic, k=20)");
    }

    #[test]
    fn sphere_substitution_uses_centers() {
        let candidate = CandidateTopology::Sphere {
            n_centers: 64,
            penalty_order: 2,
            kernel: "sobolev".to_string(),
            radians: false,
        };
        let out = assemble_candidate_formula("y ~ s(lat, lon, type=AUTO)", &candidate, true)
            .unwrap()
            .unwrap();
        assert_eq!(out, "y ~ s(lat, lon, type=sphere, centers=64)");
    }

    #[test]
    fn tensor_substitution_torus() {
        let candidate = CandidateTopology::Tensor {
            k: vec![12, 12],
            periodic: vec![true, true],
            periods: None,
        };
        let out = assemble_candidate_formula("y ~ s(theta, phi, type=AUTO)", &candidate, true)
            .unwrap()
            .unwrap();
        assert!(out.starts_with("y ~ te(theta, phi, k=[12, 12], periodic=[true, true]"));
        assert!(out.contains("identifiability=sum_tozero"));
    }

    #[test]
    fn dimension_mismatch_strict_errors() {
        let candidate = CandidateTopology::Sphere {
            n_centers: 64,
            penalty_order: 2,
            kernel: "sobolev".to_string(),
            radians: false,
        };
        let err = assemble_candidate_formula("y ~ s(x, type=AUTO)", &candidate, true).unwrap_err();
        assert!(err.contains("2-D covariate"));
    }

    #[test]
    fn dimension_mismatch_nonstrict_returns_none() {
        let candidate = CandidateTopology::Sphere {
            n_centers: 64,
            penalty_order: 2,
            kernel: "sobolev".to_string(),
            radians: false,
        };
        let out = assemble_candidate_formula("y ~ s(x, type=AUTO)", &candidate, false).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn duchon_per_axis_periodic_errors() {
        let candidate = CandidateTopology::Duchon {
            m: 2,
            centers_int: Some(64),
            per_axis_periodic: true,
            length_scale: None,
            required_dim: Some(2),
        };
        let err =
            assemble_candidate_formula("y ~ s(x1, x2, type=AUTO)", &candidate, true).unwrap_err();
        assert!(err.contains("per-axis periodic"));
    }

    #[test]
    fn user_size_option_suppresses_default() {
        let candidate = CandidateTopology::PeriodicSplineCurve {
            n_knots: 20,
            degree: 3,
            penalty_order: 2,
        };
        let out = assemble_candidate_formula("y ~ s(t, k=8, type=AUTO)", &candidate, true)
            .unwrap()
            .unwrap();
        // The user's `k=8` is preserved and the candidate does not re-add `k=20`.
        assert!(out.contains("k=8"));
        assert!(!out.contains("k=20"));
    }

    #[test]
    fn missing_auto_term_errors() {
        let candidate = CandidateTopology::PeriodicSplineCurve {
            n_knots: 20,
            degree: 3,
            penalty_order: 2,
        };
        let err = assemble_candidate_formula("y ~ s(t, k=8)", &candidate, true).unwrap_err();
        assert!(err.contains("type=AUTO"));
    }
}

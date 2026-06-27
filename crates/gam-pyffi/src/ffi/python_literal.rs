//! Python-literal formatting helpers.
//!
//! Self-contained seam extracted from the pyffi monolith (issue #780): the two
//! pure formatters that render a Rust string / float into its Python source
//! `repr` -- `python_string_repr` chooses a quote style and escapes control
//! characters the way CPython does, and `python_float_display` mirrors Python's
//! `float.__repr__` integral-valued `N.0` formatting. They depend on nothing
//! outside `std`; callers (the survival default-grid code generators) stay in
//! the parent module via a focused re-import.

pub(crate) fn python_string_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut repr = String::with_capacity(value.len() + 2);
    repr.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => repr.push_str("\\\\"),
            '\t' => repr.push_str("\\t"),
            '\n' => repr.push_str("\\n"),
            '\r' => repr.push_str("\\r"),
            '\'' if quote == '\'' => repr.push_str("\\'"),
            '"' if quote == '"' => repr.push_str("\\\""),
            other => repr.push(other),
        }
    }
    repr.push(quote);
    repr
}

pub(crate) fn python_float_display(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1.0e16 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

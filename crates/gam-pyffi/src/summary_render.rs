//! Summary-payload HTML / repr value rendering helpers.
//!
//! Self-contained seam extracted from the pyffi monolith (issue #780): the pure
//! free functions that render a fitted-model summary `PyDict` payload into HTML
//! and human-readable previews -- recursive value rendering for mappings / lists
//! / tuples, the coefficients-table HTML builder and its column-collector, the
//! float formatter (mantissa-trim + exponent-normalize) and the HTML escaper --
//! together with the two preview/row-limit constants they share. They depend on
//! nothing in the rest of the module; the `#[pyfunction]`s that consume them
//! (`summary_repr`, `summary_html`) stay in the parent module via a focused
//! re-import.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyFloat, PyList, PyTuple};

pub(crate) const SUMMARY_HTML_COEFFICIENT_LIMIT: usize = 50;
pub(crate) const SUMMARY_VALUE_PREVIEW_LIMIT: usize = 6;

pub(crate) fn summary_render_value(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(float_value) = value.cast::<PyFloat>() {
        return Ok(summary_format_float(float_value.extract::<f64>()?));
    }
    if let Ok(mapping) = value.cast::<PyDict>() {
        return summary_render_mapping_value(mapping);
    }
    if let Ok(sequence) = value.cast::<PyList>() {
        return summary_render_list_value(sequence);
    }
    if let Ok(sequence) = value.cast::<PyTuple>() {
        return summary_render_tuple_value(sequence);
    }
    value.str()?.extract::<String>()
}

pub(crate) fn summary_render_mapping_value(value: &Bound<'_, PyDict>) -> PyResult<String> {
    if value.is_empty() {
        return Ok("{}".to_string());
    }
    let mut parts = Vec::new();
    for (index, (key, item_value)) in value.iter().enumerate() {
        if index >= SUMMARY_VALUE_PREVIEW_LIMIT {
            break;
        }
        parts.push(format!(
            "{}: {}",
            key.str()?.extract::<String>()?,
            summary_render_value(&item_value)?
        ));
    }
    let suffix = if value.len() <= SUMMARY_VALUE_PREVIEW_LIMIT {
        String::new()
    } else {
        format!(", ... ({} total)", value.len())
    };
    Ok(format!("{{{}{}}}", parts.join(", "), suffix))
}

pub(crate) fn summary_render_list_value(value: &Bound<'_, PyList>) -> PyResult<String> {
    if value.is_empty() {
        return Ok("[]".to_string());
    }
    let preview_len = value.len().min(SUMMARY_VALUE_PREVIEW_LIMIT);
    let mut parts = Vec::with_capacity(preview_len);
    for index in 0..preview_len {
        parts.push(summary_render_value(&value.get_item(index)?)?);
    }
    let suffix = if value.len() <= SUMMARY_VALUE_PREVIEW_LIMIT {
        String::new()
    } else {
        format!(", ... ({} total)", value.len())
    };
    Ok(format!("[{}{}]", parts.join(", "), suffix))
}

pub(crate) fn summary_render_tuple_value(value: &Bound<'_, PyTuple>) -> PyResult<String> {
    if value.is_empty() {
        return Ok("[]".to_string());
    }
    let preview_len = value.len().min(SUMMARY_VALUE_PREVIEW_LIMIT);
    let mut parts = Vec::with_capacity(preview_len);
    for index in 0..preview_len {
        parts.push(summary_render_value(&value.get_item(index)?)?);
    }
    let suffix = if value.len() <= SUMMARY_VALUE_PREVIEW_LIMIT {
        String::new()
    } else {
        format!(", ... ({} total)", value.len())
    };
    Ok(format!("[{}{}]", parts.join(", "), suffix))
}

pub(crate) fn summary_render_coefficients_html(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    let Some(coefficients_any) = payload.get_item("coefficients")? else {
        return Ok(String::new());
    };
    let Ok(coefficients) = coefficients_any.cast::<PyList>() else {
        return Ok(String::new());
    };
    if coefficients.is_empty() {
        return Ok(String::new());
    }

    let columns = summary_coefficient_columns(coefficients)?;
    let mut header_cells = String::new();
    for column in &columns {
        header_cells.push_str(
            "<th style='text-align:right;padding:0.25rem 0.75rem;\
             border-bottom:1px solid #ddd;'>",
        );
        header_cells.push_str(&summary_html_escape(column));
        header_cells.push_str("</th>");
    }

    let row_limit = coefficients.len().min(SUMMARY_HTML_COEFFICIENT_LIMIT);
    let mut body_rows = String::new();
    for index in 0..row_limit {
        let row_any = coefficients.get_item(index)?;
        let row = row_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("summary coefficient rows must be dictionaries"))?;
        body_rows.push_str("<tr>");
        for column in &columns {
            let rendered = match row.get_item(column.as_str())? {
                Some(value) => summary_render_value(&value)?,
                None => String::new(),
            };
            body_rows.push_str("<td style='text-align:right;padding:0.25rem 0.75rem;'>");
            body_rows.push_str(&summary_html_escape(&rendered));
            body_rows.push_str("</td>");
        }
        body_rows.push_str("</tr>");
    }

    let note = if coefficients.len() > SUMMARY_HTML_COEFFICIENT_LIMIT {
        format!(
            "<p style='margin:0.25rem 0 0 0;color:#666;'>Showing first {} of {} coefficients.</p>",
            SUMMARY_HTML_COEFFICIENT_LIMIT,
            coefficients.len()
        )
    } else {
        String::new()
    };

    Ok(format!(
        "<h4 style='margin:1rem 0 0.35rem 0;'>Coefficients</h4>\
         <table style='border-collapse:collapse;'>\
         <thead><tr>{header_cells}</tr></thead>\
         <tbody>{body_rows}</tbody>\
         </table>\
         {note}"
    ))
}

pub(crate) fn summary_coefficient_columns(
    coefficients: &Bound<'_, PyList>,
) -> PyResult<Vec<String>> {
    let mut columns = Vec::new();
    for index in 0..coefficients.len() {
        let row_any = coefficients.get_item(index)?;
        let row = row_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("summary coefficient rows must be dictionaries"))?;
        for (key, _) in row.iter() {
            let column = key.str()?.extract::<String>()?;
            if !columns.iter().any(|existing| existing == &column) {
                columns.push(column);
            }
        }
    }
    Ok(columns)
}

pub(crate) fn summary_format_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value == f64::INFINITY {
        return "inf".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }

    let exponent = value.abs().log10().floor() as i32;
    let mut out = if !(-4..6).contains(&exponent) {
        let raw = format!("{:.5e}", value);
        summary_normalize_exponent(&raw)
    } else {
        let places = (6 - exponent - 1).max(0) as usize;
        summary_trim_float(format!("{:.*}", places, value))
    };
    if out == "-0" {
        out = "0".to_string();
    }
    out
}

pub(crate) fn summary_normalize_exponent(raw: &str) -> String {
    let Some((mantissa, exponent)) = raw.split_once('e') else {
        return raw.to_string();
    };
    let mantissa = summary_trim_float(mantissa.to_string());
    let (sign, digits) = if let Some(rest) = exponent.strip_prefix('-') {
        ('-', rest)
    } else if let Some(rest) = exponent.strip_prefix('+') {
        ('+', rest)
    } else {
        ('+', exponent)
    };
    let digits = digits.trim_start_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let padded = if digits.len() == 1 {
        format!("0{digits}")
    } else {
        digits.to_string()
    };
    format!("{mantissa}e{sign}{padded}")
}

pub(crate) fn summary_trim_float(mut value: String) -> String {
    if value.contains('.') {
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
    }
    value
}

pub(crate) fn summary_html_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

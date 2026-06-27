//! Regression: the crate does not build from a clean checkout at HEAD.
//!
//! `build.rs` enforces a project ban rule — "underscore-prefixed fn parameter
//! (use the value, restructure the API, or delete the param)" — over every
//! `*.rs` file under `src/` (only `build.rs` itself is exempt). Commit
//! 70b58940b ("refactor: enforce response_geometry API boundaries and
//! centralize test helpers") introduced three parameters that silence the
//! unused-parameter warning with a leading underscore instead of consuming,
//! restructuring, or deleting them:
//!
//!   * src/inference/predict/mod.rs  `fn predict_dispersion_scale(&self, _input: &PredictInput)`
//!   * src/reml_contracts.rs         `fn trace_projected_factor_cached(&self, _, _factor_cache: &ProjectedFactorCache)`
//!   * src/reml_contracts.rs         `fn projected_matrix_cached(&self, _, _factor_cache: &ProjectedFactorCache)`
//!
//! On any state that forces `build.rs` to re-run (a clean build — which is what
//! `cargo build`, `cargo test`, and the `maturin` wheel build all do), the
//! scanner finds these and aborts with:
//!
//!   error: 3 ban violations across 1 rule
//!   error — underscore-prefixed fn parameter ...
//!     error: src/inference/predict/mod.rs:887: _input: &PredictInput,
//!     error: src/reml_contracts.rs:102: _factor_cache: &ProjectedFactorCache,
//!     error: src/reml_contracts.rs:118: _factor_cache: &ProjectedFactorCache,
//!
//! so `cargo build` / `cargo test` exit 101 and the `gamfit` wheel cannot be
//! built. (A *cached* `target/` can hide it: when `build.rs`'s fingerprint is
//! still valid the script is skipped and a stale incremental build appears to
//! succeed — which is how this slipped onto `main`.)
//!
//! While the bug is present the crate cannot build, so this integration-test
//! target cannot even be produced — `cargo test` fails. Once the three
//! parameters are consumed / removed and the crate builds again, the checks
//! below find no underscore-prefixed parameter on those three functions and the
//! test passes, with no further edits.
//!
//! The checks are scoped to the exact functions the offending commit touched
//! (looked up by name inside their specific files), so legitimate pre-existing
//! underscore usages elsewhere — e.g. `predict_noise_scale`, bare `_:`
//! placeholders, or PyO3 `_py: Python<'_>` tokens — are never flagged.

use std::fs;
use std::path::Path;

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// Replace block-comment bodies with spaces (newlines preserved).
fn strip_block_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut in_block = false;
    while i < bytes.len() {
        if in_block {
            if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                out.push_str("  ");
                i += 2;
                in_block = false;
                continue;
            }
            out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            out.push_str("  ");
            i += 2;
            in_block = true;
            continue;
        }
        let ch_len = utf8_len(bytes[i]);
        for b in &bytes[i..(i + ch_len).min(bytes.len())] {
            out.push(*b as char);
        }
        i += ch_len;
    }
    out
}

/// Remove `//` line comments and string-literal bodies from one line. Char
/// literals / lifetimes (`'a`) are left untouched.
fn strip_line_comment_and_strings(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    let mut in_str = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            break;
        }
        if b == b'"' {
            in_str = true;
            i += 1;
            continue;
        }
        let ch_len = utf8_len(b);
        for cb in &bytes[i..(i + ch_len).min(bytes.len())] {
            out.push(*cb as char);
        }
        i += ch_len;
    }
    out
}

fn cleaned_source(src: &str) -> String {
    strip_block_comments(src)
        .lines()
        .map(strip_line_comment_and_strings)
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Return the parameter-list body (text between the matching `(` `)`) of the
/// first `fn <name>(` definition in `cleaned`, or `None` if absent. Skips `(`
/// nested in the generic `<...>` list before the param list.
fn fn_param_list<'a>(cleaned: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("fn {name}");
    let bytes = cleaned.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = cleaned[from..].find(&needle) {
        let kw = from + rel;
        // Whole-word match for `fn` and for the name boundary after it.
        let before_ok = kw == 0 || !is_ident_byte(bytes[kw - 1]);
        let after = kw + needle.len();
        let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
        if before_ok && after_ok {
            // Locate the param-list `(` at angle-depth 0.
            let mut angle: i32 = 0;
            let mut i = after;
            let mut open = None;
            while i < bytes.len() {
                match bytes[i] {
                    b'<' => angle += 1,
                    b'>' => {
                        if angle > 0 {
                            angle -= 1;
                        }
                    }
                    b'(' if angle == 0 => {
                        open = Some(i);
                        break;
                    }
                    b'{' | b';' if angle == 0 => break,
                    _ => {}
                }
                i += 1;
            }
            if let Some(open) = open {
                let mut depth = 0i32;
                let mut j = open;
                while j < bytes.len() {
                    match bytes[j] {
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(&cleaned[open + 1..j]);
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
            }
        }
        from = kw + needle.len();
    }
    None
}

/// Top-level parameter slices (commas inside `<>`, `()`, `[]` do not split).
fn split_top_level_params(inner: &str) -> Vec<&str> {
    let bytes = inner.as_bytes();
    let (mut angle, mut paren, mut brack) = (0i32, 0i32, 0i32);
    let mut start = 0usize;
    let mut parts = Vec::new();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'<' => angle += 1,
            b'>' => {
                if angle > 0 {
                    angle -= 1;
                }
            }
            b'(' => paren += 1,
            b')' => paren -= 1,
            b'[' => brack += 1,
            b']' => brack -= 1,
            b',' if angle == 0 && paren == 0 && brack == 0 => {
                parts.push(&inner[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start <= inner.len() {
        parts.push(&inner[start..]);
    }
    parts
}

/// The underscore-prefixed binding NAME of a parameter, if any, applying the
/// same exemptions `build.rs` does: bare `_` is allowed, and a `Python<...>`
/// GIL token is allowed.
fn underscore_param_name(param: &str) -> Option<String> {
    let mut p = param.trim();
    while let Some(rest) = p.strip_prefix("#[") {
        match rest.find(']') {
            Some(close) => p = rest[close + 1..].trim_start(),
            None => break,
        }
    }
    loop {
        let before = p;
        if let Some(rest) = p.strip_prefix("&mut ") {
            p = rest.trim_start();
        } else if let Some(rest) = p.strip_prefix('&') {
            p = rest.trim_start();
        } else if let Some(rest) = p.strip_prefix("mut ") {
            p = rest.trim_start();
        }
        if p == before {
            break;
        }
    }
    let colon = p.find(':')?;
    let name = p[..colon].trim();
    let ty = p[colon + 1..].trim_start();
    if name == "_" || !name.starts_with('_') {
        return None;
    }
    if ty.starts_with("Python<") {
        return None;
    }
    Some(name.to_string())
}

fn assert_fn_has_no_underscore_param(rel_path: &str, fn_name: &str) {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(manifest).join(rel_path);
    let content =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let cleaned = cleaned_source(&content);
    let Some(params) = fn_param_list(&cleaned, fn_name) else {
        // The function was removed/renamed entirely — the offending parameter
        // cannot survive that, so there is nothing to flag.
        return;
    };
    let offenders: Vec<String> = split_top_level_params(params)
        .into_iter()
        .filter_map(underscore_param_name)
        .collect();
    assert!(
        offenders.is_empty(),
        "{rel_path}: `fn {fn_name}` still declares underscore-prefixed parameter(s) {:?}. \
         build.rs bans these, so the crate does not build from a clean checkout while they \
         exist (introduced by commit 70b58940b). Consume the parameter, restructure the API, \
         or delete it.",
        offenders
    );
}

#[test]
fn predict_dispersion_scale_has_no_underscore_param() {
    assert_fn_has_no_underscore_param("src/inference/predict/mod.rs", "predict_dispersion_scale");
}

#[test]
fn reml_contract_cached_methods_have_no_underscore_params() {
    assert_fn_has_no_underscore_param("src/reml_contracts.rs", "trace_projected_factor_cached");
    assert_fn_has_no_underscore_param("src/reml_contracts.rs", "projected_matrix_cached");
}

#[test]
fn mixed_periodicity_duchon_builder_has_no_underscore_param() {
    assert_fn_has_no_underscore_param(
        "crates/gam-terms/src/basis/periodic_duchon.rs",
        "build_duchon_basis_mixed_periodicity",
    );
}

#[test]
fn effective_seed_budget_has_no_underscore_param() {
    assert_fn_has_no_underscore_param(
        "crates/gam-solve/src/rho_optimizer/seed_screening.rs",
        "effective_seed_budget",
    );
}

//! Bug hunt: the workspace does not build from a clean checkout at HEAD.
//!
//! `build.rs` bans underscore-prefixed function parameters
//! (`scan_for_underscore_fn_args`): a `_name: T` parameter silences the
//! unused-parameter warning by hiding it from the lint instead of using the
//! value, restructuring the API, or deleting the param. Commit 1dff6b53b
//! ("fix(#1569): scale-aware trust-metric floor for coupled survival LS time
//! block") added a method that takes a `ParameterBlockSpec` slice it never
//! reads, and silenced the warning by underscore-prefixing it:
//!
//!   crates/gam-models/src/survival/location_scale/family_solver.rs:2222
//!       fn joint_trust_metric_block_floor(
//!           &self,
//!           block_states: &[ParameterBlockState],
//!           _specs: &[ParameterBlockSpec],   // <-- banned underscore param
//!       ) -> Result<Option<Array1<f64>>, String>
//!
//! On any state that forces `build.rs` to re-run (a clean build — `cargo
//! build`, `cargo test`, or the `maturin` wheel build all do), the scanner
//! finds it and aborts:
//!
//!   error — underscore-prefixed fn parameter (use the value, restructure the API, or delete the param)  (1 hit)
//!     error: crates/gam-models/src/survival/location_scale/family_solver.rs:2222: _specs: &[ParameterBlockSpec],
//!   error: 3 ban violations across 2 rules
//!
//! so `cargo build` / `cargo test` exit 101 and the `gamfit` wheel cannot be
//! built. (A *cached* `target/` can hide it: a valid build-script fingerprint
//! skips the scan and a stale incremental build appears to succeed. A fresh
//! `--release` build, the wheel build, and CI all re-run it and fail.)
//!
//! While the bug is present the crate cannot build, so this integration-test
//! target cannot even be produced — `cargo test` fails. Once the parameter is
//! consumed, the API restructured, or the param deleted, the crate builds and
//! the check below — which extracts the `joint_trust_metric_block_floor`
//! signature and asserts no parameter name is underscore-prefixed — passes,
//! with no further edits.

use std::fs;
use std::path::PathBuf;

/// Extract the parameter-list text of `fn <name>` from `src`: the substring
/// between the first `(` after the function name and its matching `)`.
fn parameter_list(src: &str, name: &str) -> String {
    let needle = format!("fn {name}");
    let fn_at = src
        .find(&needle)
        .unwrap_or_else(|| panic!("function `{name}` not found in family_solver.rs"));
    let open_rel = src[fn_at..]
        .find('(')
        .expect("function signature has no opening paren");
    let open = fn_at + open_rel;
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    for (offset, byte) in bytes[open..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return src[open + 1..open + offset].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced parens in signature of `{name}`");
}

/// Top-level (depth-0 within the param list) parameter names that are
/// underscore-prefixed, i.e. an `_ident` token sitting immediately before a
/// `:` at the outermost comma level. Reference/type underscores after the `:`
/// are not parameter names and are ignored.
fn underscore_param_names(param_list: &str) -> Vec<String> {
    let bytes = param_list.as_bytes();
    let mut depth = 0i32; // nesting from <...>, (...), [...] in types
    let mut token = String::new();
    let mut hits = Vec::new();
    let mut at_name_position = true; // true at the start of each top-level param
    for &byte in bytes {
        let ch = byte as char;
        match ch {
            '<' | '(' | '[' => {
                depth += 1;
                token.clear();
            }
            '>' | ')' | ']' => {
                depth -= 1;
                token.clear();
            }
            ',' if depth == 0 => {
                at_name_position = true;
                token.clear();
            }
            ':' if depth == 0 => {
                if at_name_position && token.starts_with('_') && token.len() > 1 {
                    hits.push(token.clone());
                }
                at_name_position = false;
                token.clear();
            }
            c if c.is_alphanumeric() || c == '_' => token.push(c),
            _ => token.clear(),
        }
    }
    hits
}

#[test]
fn joint_trust_metric_block_floor_has_no_underscore_param() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("crates/gam-models/src/survival/location_scale/family_solver.rs");
    let src =
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));

    let params = parameter_list(&src, "joint_trust_metric_block_floor");
    let banned = underscore_param_names(&params);
    assert!(
        banned.is_empty(),
        "`joint_trust_metric_block_floor` declares underscore-prefixed parameter(s) {banned:?}; \
         build.rs bans these and aborts the whole workspace build (issue #1569 commit 1dff6b53b). \
         Parameter list was: ({params})"
    );
}

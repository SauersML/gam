use std::collections::BTreeSet;
use std::fs;

/// Every `#[pyfunction]` exported by the PyO3 boundary must be registered
/// exactly once in the `#[pymodule]` body, otherwise the symbol compiles but is
/// unreachable from Python.
///
/// The boundary is no longer a single `lib.rs`: the `#[pyfunction]` definitions
/// and the `#[pymodule] fn rust_extension` body live in the source fragments
/// that `lib.rs` pulls in with `include!(...)` (`model_ffi.rs`,
/// `latent_basis_and_sae_ffi.rs`, `reml_latent_fit_ffi.rs`, `geometry_ffi.rs`,
/// `manifold_and_posterior_ffi.rs`). This test scans those fragments — the same
/// text the compiler sees through `lib.rs` — so the registration invariant is
/// checked against the real source layout, not a stale single-file assumption.
#[test]
fn pyffi_every_pyfunction_is_registered_once() {
    // Source fragments `include!`d into crates/gam-pyffi/src/lib.rs. The
    // `#[pyfunction]`s and the `#[pymodule]` body are distributed across these.
    // Derive the list from lib.rs's own `include!("…")` directives rather than
    // hardcoding paths: the fragments have already been relocated once (the
    // flat `src/<x>_ffi.rs` files moved under `src/model/`, `src/latent/`,
    // `src/manifold/`), and a hardcoded list silently rots into a `cannot read`
    // panic on the next move. lib.rs is the single source of truth for which
    // fragments the compiler actually inlines, so read it from there.
    let fragment_files = pyffi_include_fragments();

    // Collect every `#[pyfunction]` name across all fragments.
    let mut pyfns = BTreeSet::new();
    for path in &fragment_files {
        let src = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let lines: Vec<&str> = src.lines().collect();
        let mut i = 0usize;
        while i < lines.len() {
            if lines[i].trim_start().starts_with("#[pyfunction") {
                let mut j = i + 1;
                while j < lines.len() {
                    let t = lines[j].trim_start();
                    if t.starts_with("#[") {
                        j += 1;
                        continue;
                    }
                    if let Some(rest) = t.strip_prefix("fn ") {
                        // Function name terminates at the first `(`, `<`, or
                        // whitespace, so we strip generic / lifetime parameters
                        // like `<'py>` before comparing against
                        // `wrap_pyfunction!` registrations.
                        let name: String = rest
                            .chars()
                            .take_while(|c| !matches!(c, '(' | '<' | ' ' | '\t'))
                            .collect();
                        pyfns.insert(name);
                    }
                    break;
                }
                i = j;
            }
            i += 1;
        }
    }
    assert!(
        !pyfns.is_empty(),
        "no #[pyfunction] exports found; the boundary fragment layout changed"
    );

    // The `#[pymodule] fn rust_extension` body holds every `wrap_pyfunction!`
    // registration. Locate the fragment that defines it and scan from the
    // function start to the end of file (the body runs to EOF in its fragment).
    let mut mod_src = String::new();
    for path in &fragment_files {
        let src = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        if let Some(module_start) = src.find("fn rust_extension") {
            // If a `#[cfg(test)]` module trails the pymodule body in the same
            // fragment, stop before it so test-only helpers do not count.
            let end = src[module_start..]
                .find("#[cfg(test)]")
                .map(|off| module_start + off)
                .unwrap_or(src.len());
            mod_src = src[module_start..end].to_string();
            break;
        }
    }
    assert!(
        !mod_src.is_empty(),
        "could not locate `fn rust_extension` pymodule body in the boundary fragments"
    );

    // Find every `wrap_pyfunction!(...)` invocation. The first identifier inside
    // the parentheses is the registered Rust function. Walk character-by-character
    // so that whitespace / newlines between `wrap_pyfunction!(` and the function
    // name (rustfmt loves to split these) do not hide the registration.
    let needle = "wrap_pyfunction!(";
    let mut regs = Vec::new();
    let bytes = mod_src.as_bytes();
    let mut search_from = 0usize;
    while let Some(off) = mod_src[search_from..].find(needle) {
        let abs = search_from + off + needle.len();
        let mut k = abs;
        while k < bytes.len() && (bytes[k] as char).is_whitespace() {
            k += 1;
        }
        let start = k;
        while k < bytes.len() {
            let c = bytes[k] as char;
            if c.is_ascii_alphanumeric() || c == '_' {
                k += 1;
            } else {
                break;
            }
        }
        if k > start {
            regs.push(mod_src[start..k].to_string());
        }
        search_from = abs;
    }

    let regs_set: BTreeSet<String> = regs.iter().cloned().collect();
    let missing: Vec<String> = pyfns.difference(&regs_set).cloned().collect();
    assert!(
        missing.is_empty(),
        "unregistered #[pyfunction] exports: {missing:?}"
    );
}

/// The boundary fragments `include!`d into `crates/gam-pyffi/src/lib.rs`,
/// resolved to workspace-relative paths. Parsing them out of `lib.rs` keeps this
/// list in lockstep with the real layout: when a fragment is renamed, moved, or
/// added, lib.rs's `include!` directive is the one place that must change, and
/// this test follows it automatically instead of failing on a stale hardcoded
/// path.
fn pyffi_include_fragments() -> Vec<String> {
    const SRC_DIR: &str = "crates/gam-pyffi/src";
    let lib_rs = format!("{SRC_DIR}/lib.rs");
    let lib_src =
        fs::read_to_string(&lib_rs).unwrap_or_else(|e| panic!("read {lib_rs}: {e}"));

    let needle = "include!(\"";
    let mut fragments = Vec::new();
    let mut from = 0usize;
    while let Some(off) = lib_src[from..].find(needle) {
        let start = from + off + needle.len();
        let Some(end_rel) = lib_src[start..].find('"') else {
            break;
        };
        let rel = &lib_src[start..start + end_rel];
        fragments.push(format!("{SRC_DIR}/{rel}"));
        from = start + end_rel;
    }
    assert!(
        !fragments.is_empty(),
        "no include!(\"…\") fragments found in {lib_rs}; the boundary layout changed"
    );
    fragments
}

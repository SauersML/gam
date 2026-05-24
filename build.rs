use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    fs::write(out_dir.join("lint_errors.rs"), "").expect("failed to write lint_errors.rs");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_secs();
    println!("cargo:rustc-env=GAM_BUILD_TIMESTAMP={timestamp}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/terms/penalties");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    emit_python_penalty_manifest(&manifest_dir)
        .expect("failed to emit Python analytic-penalty manifest");

    // Outright TO-DO ban (split below to avoid self-trigger in this file).
    let needle: &str = concat!("TO", "DO");
    let mut todo_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_banned_marker(&manifest_dir, &manifest_dir, needle, &mut todo_offenders);

    // `#[allow(...)]` / `#![allow(...)]` / `#[expect(...)]` /
    // `#![expect(...)]` ban — any lint, anywhere. Every file-level allow
    // or expect is an admission that some lint flagged real code and the
    // author chose to hide it instead of fix it. Build.rs is the single
    // source of "we accept this category"; individual file-level allows
    // are forbidden. `expect` is the promotion form of `allow` — it
    // silences the lint exactly the same way (and additionally errors
    // when the lint does NOT fire), so it has the same "hide the signal"
    // failure mode and is banned identically. Fix the underlying code:
    // rename, restructure, delete, or — if the lint truly is the wrong
    // call site-wide — edit this build.rs to encode the policy here.
    let mut allow_offenders: Vec<(PathBuf, usize, String, String)> = Vec::new();
    scan_for_banned_allow(&manifest_dir, &manifest_dir, &mut allow_offenders);

    // `let _...` ban (any underscore-leading let pattern). This covers
    // `let _ = expr;`, `let _: T = expr;`, `let _name = expr;`,
    // `let mut _name = expr;`, and the tuple-pattern dodge
    // `let (_, _) = expr;` / `let (_a, _b) = expr;` where every binding
    // in the tuple is underscore-prefixed. Every such binding silences
    // the type system's unused-value feedback. Use or delete: bind a real
    // name and read it, or call the expression for its effect without a
    // binding (e.g. `drop(expr)` or just `expr;` for `Result` consumers
    // wrapped behind an explicit check).
    let mut underscore_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_let_underscore(&manifest_dir, &manifest_dir, &mut underscore_offenders);

    // Ignored-test ban: every `#[ignore]` / `#[ignore = "..."]` attribute is
    // a test the suite no longer enforces. Stop the build until they are
    // either deleted or restored to the running suite.
    let mut ignored: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_ignored_tests(&manifest_dir, &manifest_dir, &mut ignored);

    // Banned substrings in code (skipping `//` line comments and `"..."` /
    // `'..'` literals via a lexer-lite). Each entry is matched against the
    // stripped code portion of every line. Build.rs is exempt (it names the
    // needles as part of this scanner's contract).
    let mut substring_offenders: Vec<(PathBuf, usize, &'static str, String)> = Vec::new();
    scan_for_banned_substrings(&manifest_dir, &manifest_dir, &mut substring_offenders);

    // `unsafe` requires a `// SAFETY:` justification within the same line or
    // one of the 3 preceding non-blank lines. Keeps `unsafe` legal but makes
    // every use cite a reason.
    let mut unsafe_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_unsafe_without_safety(&manifest_dir, &manifest_dir, &mut unsafe_offenders);

    // `mem::transmute` requires a `// SAFETY:` justification, same shape
    // as the `unsafe` rule. Transmute is loud enough that the dedicated
    // section keeps it visible even when surrounded by other `unsafe`.
    let mut transmute_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_transmute_without_safety(&manifest_dir, &manifest_dir, &mut transmute_offenders);

    // `panic!(` in non-test code requires a `// SAFETY:` justification.
    // Mirrors the `unsafe` rule: legal but every site cites a reason.
    let mut panic_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_panic_without_safety(&manifest_dir, &manifest_dir, &mut panic_offenders);

    // `#[cfg(any())]` is permanently false, `#[cfg(all())]` permanently
    // true. Both are dead-by-construction guards in the same family as
    // `const X: bool = false;`.
    let mut dead_cfg_gate_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_dead_cfg_gates(&manifest_dir, &manifest_dir, &mut dead_cfg_gate_offenders);

    // Cargo features ban. Any `#[cfg(feature = "...")]` /
    // `#[cfg_attr(feature = "...", ...)]` (at any nesting depth — `all`,
    // `any`, `not`) carves the codebase into conditionally-compiled
    // forks: rustc's `dead_code` lint sees only one fork at a time, the
    // test suite has to enumerate the buildable configurations, and the
    // gate itself is the same "make the lint shut up depending on
    // context" family this scanner exists to ban. Autoderive paths
    // (e.g. GPU vs CPU) from problem characteristics instead.
    let mut feature_cfg_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_feature_cfg_gates(&manifest_dir, &manifest_dir, &mut feature_cfg_offenders);

    // Cargo.toml `[features]` entries are banned for the same reason as
    // the `cfg(feature = ...)` attributes that consume them: without an
    // entry, the gate has nothing to reference. Delete the `[features]`
    // section (and the corresponding gates) rather than papering over.
    let mut cargo_feature_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_cargo_feature_entries(&manifest_dir, &manifest_dir, &mut cargo_feature_offenders);

    // `#[should_panic]` without `expected = "..."` catches any panic and
    // masks unrelated bugs. Require an `expected` string.
    let mut should_panic_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_bare_should_panic(&manifest_dir, &manifest_dir, &mut should_panic_offenders);

    // `eprintln!` / `eprint!` carrying `{:?}` / `{:#?}` debug formatting is
    // a hand-rolled `dbg!`. Test-aware; build.rs exempt.
    let mut debug_eprintln_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_debug_eprintln(&manifest_dir, &manifest_dir, &mut debug_eprintln_offenders);

    // Bare `const <ident>: bool = <literal>;` items are dead-by-construction
    // guards (rustc's `dead_code` cannot prove unreachability through them)
    // or constant truths that should not exist. Real toggles belong in
    // `cfg`/`feature`/runtime config.
    let mut dead_guard_const_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_dead_guard_consts(
        &manifest_dir,
        &manifest_dir,
        &mut dead_guard_const_offenders,
    );

    // Underscore-prefixed function parameter names. `_name: T` silences the
    // unused-parameter warning by hiding it from the lint rather than fixing
    // it. Use the parameter, restructure the API so it isn't passed, or
    // delete the param. Bare `_` placeholders are allowed (rare; arguably
    // legitimate). Build.rs is exempt.
    let mut underscore_fn_arg_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_underscore_fn_args(
        &manifest_dir,
        &manifest_dir,
        &mut underscore_fn_arg_offenders,
    );

    // `#[test]` functions whose bodies contain no assertion-shaped construct
    // (assert macros, panic-shape macros, or `?`-propagation). Such tests
    // silently pass without verifying anything.
    let mut useless_test_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_useless_tests(&manifest_dir, &manifest_dir, &mut useless_test_offenders);

    // `#[cfg(test)]` attribute directly on ANY item inside `src/` (the
    // production tree) — regardless of visibility. Tests must exercise
    // production code, not other test code; `#[cfg(test)]` is not a
    // legitimate way to silence the `dead_code` lint on either pub or
    // private items. The only legitimate use of a top-level `#[cfg(test)]`
    // attribute in `src/` is to gate a private test submodule:
    // `#[cfg(test)] mod tests { ... }`, `mod test_support`, `mod tests_*`,
    // or `mod *_tests`. Every other item — `fn`, `struct`, `enum`,
    // `const`, `static`, `type`, `use`, `impl` — must live unconditionally
    // in production or move inside one of those private test submodules.
    let mut cfg_test_pub_offenders: Vec<(PathBuf, usize, String, String)> = Vec::new();
    scan_for_cfg_test_on_pub_items(&manifest_dir, &manifest_dir, &mut cfg_test_pub_offenders);

    // No-op self-consuming function bodies (e.g. `fn discard(self) {}`). The
    // wrapper-struct launder pattern: a `fn foo(self) {}` exists solely so
    // callers can write `x.foo()` instead of letting an unused-value warning
    // fire. By-reference receivers (`&self`, `&mut self`) are NOT flagged
    // (legitimate empty `Drop::drop(&mut self)` impls live there).
    let mut noop_self_consuming_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_noop_self_consuming_fns(
        &manifest_dir,
        &manifest_dir,
        &mut noop_self_consuming_offenders,
    );

    // Stub-body function ban: multi-parameter (>= 2 args, counting `self`)
    // functions whose entire body is a single trivial sentinel expression —
    // `None`, `Ok(())`, `Err("...".into())`, `Default::default()` /
    // `T::default()`, `Vec::new()` / `vec![]`, `HashMap::new()` / similar
    // no-arg standard constructors, `Array{1,2,3}::zeros(...)`,
    // `Some(Default::default())`, the empty tuple `()`, or a bare literal.
    // The no-arg threshold lifts the rule above legitimate constructors
    // (`fn empty() -> Vec<u32> { Vec::new() }` has no inputs to compute
    // from). For a >=2-arg fn the sentinel body means the parameters are
    // never used — the canonical "delete the real impl to make the compile
    // error go away" shape. Test scope and bodyless trait declarations are
    // exempt; build.rs is exempt.
    let mut stub_body_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_stub_function_bodies(&manifest_dir, &manifest_dir, &mut stub_body_offenders);

    // Dodge-named function identifiers. Names like `discard_*`, `swallow_*`,
    // `silence_*`, `*_for_fixed_lambda`, `*_no_op_*`, `*_intentionally_unused`,
    // `placeholder_for_*`, `dummy_for_*` announce lint-laundering intent in
    // the identifier itself: the only reason such a method exists is to
    // consume a value or compile away a warning. Banning the names by
    // substring makes "rename to look legitimate" cost the same as actually
    // fixing the code. Build.rs is exempt; strict everywhere else.
    let mut dodge_name_offenders: Vec<(PathBuf, usize, String, String)> = Vec::new();
    scan_for_dodge_named_fns(&manifest_dir, &manifest_dir, &mut dodge_name_offenders);

    // Vendoring ban. Reject any `vendor/` directory under the manifest root
    // (any depth). Vendored upstream crates fork the dependency tree from
    // crates.io / git, hide upstream CVE / fix flow, and bypass the same
    // first-party lint gate this scanner enforces. Use `[dependencies] ...
    // = "version"` or `git = ".../"` in `Cargo.toml` instead.
    let mut vendor_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_vendor_directories(&manifest_dir, &manifest_dir, &mut vendor_offenders);

    // Persistent unimplemented/todo/unreachable removal audit. Compares the
    // current set of marker-bearing functions against the on-disk ledger and
    // flags any function whose marker disappeared without a real implementation
    // taking its place (trivial body, or replaced by another panic-shape
    // macro). The ledger is rewritten in place; pruning happens automatically
    // for legitimately-removed sites. Build.rs cannot reach the section
    // builder below without finishing this call, so the audit's ledger
    // write always lands even when other scanners report offenders.
    let mut history_violations: Vec<(PathBuf, usize, String, String)> = Vec::new();
    run_unimplemented_history_audit(&manifest_dir, &mut history_violations);

    // Cross-file scanner: items defined in `src/` whose identifier is
    // referenced by a test/bench file but by NO other production `src/`
    // file (excluding the defining file and `#[cfg(test)]` regions). This
    // catches the deeper version of the `#[cfg(test)]` dodge: a
    // `pub(crate)` / `pub(super)` / private item in `src/` kept alive only
    // because a test names it. Rustc's `dead_code` lint can't see this —
    // the test target counts as a consumer — but lexically it is dead
    // production code. The scan is heuristic and biased toward flagging.
    let mut src_test_only_offenders: Vec<(PathBuf, usize, String, String)> = Vec::new();
    scan_for_src_items_used_only_by_tests(&manifest_dir, &mut src_test_only_offenders);

    let mut sections: Vec<Section> = Vec::new();

    if !todo_offenders.is_empty() {
        sections.push(Section {
            title: format!("{} marker", needle),
            rows: todo_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !allow_offenders.is_empty() {
        sections.push(Section {
            title: "#[allow(...)] / #[expect(...)] (any lint, anywhere — fix the underlying code instead of silencing the lint)".to_string(),
            rows: allow_offenders
                .iter()
                .map(|(r, l, lint, s)| (r.clone(), *l, Some(lint.clone()), s.clone()))
                .collect(),
        });
    }

    if !underscore_offenders.is_empty() {
        sections.push(Section {
            title: "let _ binding (bare `_`, `_name`, or all-underscore tuple pattern)".to_string(),
            rows: underscore_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !ignored.is_empty() {
        sections.push(Section {
            title: "#[ignore] test".to_string(),
            rows: ignored
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !substring_offenders.is_empty() {
        // Group by needle label so each banned pattern gets its own section.
        let mut by_label: std::collections::BTreeMap<&'static str, Vec<(PathBuf, usize, String)>> =
            std::collections::BTreeMap::new();
        for (rel, line_no, label, line) in &substring_offenders {
            by_label
                .entry(*label)
                .or_default()
                .push((rel.clone(), *line_no, line.clone()));
        }
        for (label, rows_in) in by_label {
            sections.push(Section {
                title: label.to_string(),
                rows: rows_in
                    .into_iter()
                    .map(|(r, l, s)| (r, l, None, s))
                    .collect(),
            });
        }
    }

    if !unsafe_offenders.is_empty() {
        sections.push(Section {
            title: "unsafe without `// SAFETY:`".to_string(),
            rows: unsafe_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !transmute_offenders.is_empty() {
        sections.push(Section {
            title: "mem::transmute without `// SAFETY:` justification".to_string(),
            rows: transmute_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !panic_offenders.is_empty() {
        sections.push(Section {
            title: "panic!() in non-test code without `// SAFETY:` justification".to_string(),
            rows: panic_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !dead_cfg_gate_offenders.is_empty() {
        sections.push(Section {
            title: "#[cfg(any())]/#[cfg(all())] (empty-arg cfg — permanently false/true)"
                .to_string(),
            rows: dead_cfg_gate_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !feature_cfg_offenders.is_empty() {
        sections.push(Section {
            title: "#[cfg(feature = ...)] / #[cfg_attr(feature = ...)] (feature gating banned — autoderive paths from problem characteristics; do not branch the codebase on opt-in flags)".to_string(),
            rows: feature_cfg_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !cargo_feature_offenders.is_empty() {
        sections.push(Section {
            title: "Cargo.toml [features] entry (feature definitions banned — delete the [features] section and the corresponding cfg-gates)".to_string(),
            rows: cargo_feature_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !should_panic_offenders.is_empty() {
        sections.push(Section {
            title: "#[should_panic] without `expected = \"...\"`".to_string(),
            rows: should_panic_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !debug_eprintln_offenders.is_empty() {
        sections.push(Section {
            title: "eprintln!/eprint! with {:?} debug formatting (use real logging or delete)"
                .to_string(),
            rows: debug_eprintln_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !dead_guard_const_offenders.is_empty() {
        sections.push(Section {
            title:
                "const <name>: bool = <literal> (dead-by-construction guard — use cfg or delete)"
                    .to_string(),
            rows: dead_guard_const_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !underscore_fn_arg_offenders.is_empty() {
        sections.push(Section {
            title:
                "underscore-prefixed fn parameter (use the value, restructure the API, or delete the param)"
                    .to_string(),
            rows: underscore_fn_arg_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !useless_test_offenders.is_empty() {
        sections.push(Section {
            title:
                "#[test] function without assertions (test must verify something — add assert! / assert_eq! / ? / #[should_panic] or delete the test)"
                    .to_string(),
            rows: useless_test_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !cfg_test_pub_offenders.is_empty() {
        sections.push(Section {
            title:
                "#[cfg(test)] on src/ item (move into a private `#[cfg(test)] mod tests { ... }` / `mod test_support` / `mod tests_*` / `mod *_tests`, or delete the unused item — `#[cfg(test)]` is not a dead_code-lint escape hatch, regardless of visibility)"
                    .to_string(),
            rows: cfg_test_pub_offenders
                .iter()
                .map(|(r, l, item, s)| (r.clone(), *l, Some(item.clone()), s.clone()))
                .collect(),
        });
    }

    if !noop_self_consuming_offenders.is_empty() {
        sections.push(Section {
            title:
                "no-op self-consuming fn (empty body with by-value `self` — likely a `let _` launderer; use or restructure the return value)"
                    .to_string(),
            rows: noop_self_consuming_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !stub_body_offenders.is_empty() {
        sections.push(Section {
            title:
                "stub function body (multi-arg function whose entire body is a sentinel like None/Ok(())/Default::default() — implement the function, return a real Result/Error, or delete the function)"
                    .to_string(),
            rows: stub_body_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !dodge_name_offenders.is_empty() {
        sections.push(Section {
            title:
                "dodge-named function (name announces lint-laundering intent — implement real behavior or restructure so the value isn't unused)"
                    .to_string(),
            rows: dodge_name_offenders
                .iter()
                .map(|(r, l, ident, s)| (r.clone(), *l, Some(ident.clone()), s.clone()))
                .collect(),
        });
    }

    if !vendor_offenders.is_empty() {
        sections.push(Section {
            title:
                "`vendor/` directory present — vendoring forks upstream dependencies past the same lint gate this scanner enforces; use `[dependencies]` in Cargo.toml (crates.io version or `git = ...`) instead"
                    .to_string(),
            rows: vendor_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !history_violations.is_empty() {
        sections.push(Section {
            title: "unimplemented!/todo!/unreachable! removed without real implementation \
                    (history audit — ban_history.txt)"
                .to_string(),
            rows: history_violations
                .iter()
                .map(|(file, line, sig, reason)| {
                    (file.clone(), *line, Some(reason.clone()), sig.clone())
                })
                .collect(),
        });
    }

    if !src_test_only_offenders.is_empty() {
        sections.push(Section {
            title:
                "src/ item referenced only by tests (production code with no production consumers — implement a real caller, delete the item, or move it into a `#[cfg(test)]` private mod if it's genuinely test-support)"
                    .to_string(),
            rows: src_test_only_offenders
                .iter()
                .map(|(r, l, hint, s)| (r.clone(), *l, Some(hint.clone()), s.clone()))
                .collect(),
        });
    }

    if sections.is_empty() {
        return;
    }

    render_report(&sections);
    std::process::exit(1);
}

#[derive(Clone)]
struct PenaltyWrapperManifest {
    kind_tag: String,
    rust_type: String,
    python_wrapper: String,
    row_block_diagonal: bool,
}

fn emit_python_penalty_manifest(manifest_dir: &Path) -> std::io::Result<()> {
    let penalties_dir = manifest_dir.join("src").join("terms").join("penalties");
    let registry = fs::read_to_string(penalties_dir.join("mod.rs"))?;
    let mut wrappers = Vec::new();
    for line in registry.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("register!(") {
            continue;
        }
        let inside = trimmed
            .trim_start_matches("register!(")
            .trim_end_matches(");");
        let mut pieces = inside.split(',').map(str::trim);
        let variant = match pieces.next() {
            Some(value) if !value.is_empty() => value,
            _ => continue,
        };
        let rust_type = match pieces.next() {
            Some(value) if !value.is_empty() => value,
            _ => continue,
        };
        let source = penalty_manifest_source_for_type(&penalties_dir, rust_type)?;
        wrappers.push(PenaltyWrapperManifest {
            kind_tag: manifest_const_string(&source, "KIND_TAG")?,
            rust_type: format!("{variant}:{rust_type}"),
            python_wrapper: manifest_const_string(&source, "PYTHON_WRAPPER")?,
            row_block_diagonal: manifest_const_bool(&source, "ROW_BLOCK_DIAGONAL")?,
        });
    }
    let mut output = String::from(
        "# Generated by build.rs from src/terms/penalties manifests.\n\
         PENALTY_MANIFEST = (\n",
    );
    for wrapper in wrappers {
        output.push_str("    {\n");
        output.push_str(&format!("        \"kind\": {:?},\n", wrapper.kind_tag));
        output.push_str(&format!("        \"rust\": {:?},\n", wrapper.rust_type));
        output.push_str(&format!(
            "        \"python\": {:?},\n",
            wrapper.python_wrapper
        ));
        output.push_str(&format!(
            "        \"row_block_diagonal\": {},\n",
            if wrapper.row_block_diagonal {
                "True"
            } else {
                "False"
            }
        ));
        output.push_str("    },\n");
    }
    output.push_str(")\n");
    fs::write(
        manifest_dir.join("gamfit").join("_penalties_manifest.py"),
        output,
    )
}

fn penalty_manifest_source_for_type(
    penalties_dir: &Path,
    rust_type: &str,
) -> std::io::Result<String> {
    for entry in fs::read_dir(penalties_dir)? {
        let path = entry?.path();
        if path.extension() != Some(OsStr::new("rs"))
            || path.file_name() == Some(OsStr::new("mod.rs"))
        {
            continue;
        }
        let source = fs::read_to_string(&path)?;
        if source.contains(&format!("impl PenaltyManifest for {rust_type}")) {
            return Ok(source);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("missing PenaltyManifest impl for {rust_type}"),
    ))
}

fn manifest_const_string(source: &str, key: &str) -> std::io::Result<String> {
    let needle = format!("const {key}: &'static str = ");
    for line in source.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with(&needle) {
            continue;
        }
        let value = trimmed
            .trim_start_matches(&needle)
            .trim_end_matches(';')
            .trim();
        return Ok(value.trim_matches('"').to_string());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("missing manifest const {key}"),
    ))
}

fn manifest_const_bool(source: &str, key: &str) -> std::io::Result<bool> {
    let needle = format!("const {key}: bool = ");
    for line in source.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with(&needle) {
            continue;
        }
        let value = trimmed
            .trim_start_matches(&needle)
            .trim_end_matches(';')
            .trim();
        return match value {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid bool manifest const {key}: {value}"),
            )),
        };
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("missing manifest const {key}"),
    ))
}

/// One banned-pattern report section: a title plus the file/line rows that
/// matched. `tag` is an optional per-row marker (e.g. `allow(dead_code)`)
/// rendered between line number and snippet.
struct Section {
    title: String,
    rows: Vec<(PathBuf, usize, Option<String>, String)>,
}

fn render_report(sections: &[Section]) {
    let total: usize = sections.iter().map(|s| s.rows.len()).sum();
    eprintln!();
    eprintln!(
        "error: {} ban violation{} across {} rule{}",
        total,
        if total == 1 { "" } else { "s" },
        sections.len(),
        if sections.len() == 1 { "" } else { "s" },
    );
    for section in sections {
        eprintln!();
        eprintln!(
            "── {}  ({} hit{})",
            section.title,
            section.rows.len(),
            if section.rows.len() == 1 { "" } else { "s" },
        );
        for (rel, line_no, tag, line) in &section.rows {
            let trimmed = line.trim();
            let snippet: String = trimmed.chars().take(160).collect();
            match tag {
                Some(t) => eprintln!("  {}:{}: [{}] {}", rel.display(), line_no, t, snippet),
                None => eprintln!("  {}:{}: {}", rel.display(), line_no, snippet),
            }
        }
    }
    eprintln!();
    eprintln!("summary:");
    for section in sections {
        eprintln!("  {:>5}  {}", section.rows.len(), section.title);
    }
    eprintln!();
}

/// Build the table of banned substrings. Each entry is `(needle, label,
/// test_aware)` matched against the "stripped" portion of every line
/// (string/char literal contents replaced by spaces, `//` line comments
/// dropped). When `test_aware` is true the match is suppressed in test
/// scope (test directories or inside `#[cfg(test)]` blocks).
fn banned_substrings() -> &'static [(&'static str, &'static str, bool)] {
    &[
        // Debug residue. Acceptable in test code.
        ("dbg!(", "dbg!", true),
        // Runtime "not done yet" markers — same family as the TO-DO text
        // ban. Acceptable in test scaffolding.
        ("todo!(", "todo!", true),
        ("unimplemented!(", "unimplemented!", true),
        // `unreachable!(` is a drop-in synonym for `unimplemented!(` from
        // the same panic family — propagate via `Result` or restructure so
        // the impossible branch is not expressible.
        ("unreachable!(", "unreachable!", true),
        // Direct panics are handled by `scan_for_panic_without_safety`
        // (a dedicated scanner that requires a `// SAFETY:` justification
        // for non-test panics), not by the lexical substring ban.
        // Vacuous assertions. Neither form belongs anywhere — tests use
        // `assert_eq!` / real predicates, production code uses `Result`.
        ("assert!(true)", "assert!(true)", false),
        ("assert!(false)", "assert!(false)", false),
        // `file!().ends_with(".rs")` is a tautological assertion (the
        // compile-time `file!()` macro always returns the `.rs` source
        // path) commonly used to satisfy `scan_for_useless_tests` without
        // actually asserting anything about the unit under test. Strict
        // everywhere — tests must verify a real property.
        (
            "file!().ends_with(\".rs\")",
            "file!().ends_with(\".rs\")",
            false,
        ),
        (
            "file!().ends_with(\".rs\"",
            "file!().ends_with(\".rs\"",
            false,
        ),
        // Process termination bypasses `Drop`. Build.rs uses
        // `std::process::exit(1)` legitimately at end-of-report and is
        // already exempt from every scanner.
        ("std::process::exit(", "std::process::exit", false),
        ("process::exit(", "process::exit", false),
        ("std::process::abort(", "std::process::abort", false),
        ("process::abort(", "process::abort", false),
        // Library code writing to stdout pollutes downstream consumers.
        // Tests, examples, and benches legitimately print.
        ("println!(", "println!", true),
        ("print!(", "print!", true),
        // Memory-leak primitives. Tests sometimes leak intentionally.
        ("mem::forget(", "mem::forget", true),
        ("Box::leak(", "Box::leak", true),
        // Environment-variable reads (see feedback_no_env_vars memory).
        // Strict everywhere.
        ("env::var(", "env::var", false),
        ("env::var_os(", "env::var_os", false),
        // Env-var iteration is the same read-from-environment loophole as
        // `env::var(...)`; banning the singular form without these would
        // leave `env::vars().any(|(k,v)| k == "FOO" && v == "1")` as a
        // behavior-identical workaround.
        ("env::vars(", "env::vars", false),
        ("env::vars_os(", "env::vars_os", false),
        // `if let Ok/Some/Err(_) = …` — `.is_ok()` / `.is_some()` /
        // `.is_err()` are strictly cleaner. Match-arm forms `Ok(_) =>`
        // etc. are NOT banned: when the inner type is `()` (e.g.
        // `Result<(), E>` from `channel.send`), `Ok(_) =>` is the
        // idiomatic variant-only check and there is nothing to bind.
        // Tripwire — strict everywhere.
        ("if let Ok(_)", "if let Ok(_)", false),
        ("if let Some(_)", "if let Some(_)", false),
        ("if let Err(_)", "if let Err(_)", false),
        // Redundant boolean comparisons — redundant everywhere.
        ("== true", "== true", false),
        ("== false", "== false", false),
        ("!= true", "!= true", false),
        ("!= false", "!= false", false),
        // Sleep hides latency / introduces flakiness. Tests sometimes
        // need a real sleep (e.g. timing scheduler races).
        ("thread::sleep(", "thread::sleep", true),
        ("std::thread::sleep(", "std::thread::sleep", true),
        // Busy-wait primitives. `while Instant::now() < until { spin_loop() }`
        // is a worse `thread::sleep` (pins a core instead of parking).
        // Strict everywhere; legitimate uses are vanishingly rare.
        ("spin_loop()", "spin_loop", false),
        ("thread::yield_now(", "thread::yield_now", false),
        ("std::thread::yield_now(", "std::thread::yield_now", false),
    ]
}

fn scan_for_banned_substrings(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, &'static str, String)>,
) {
    let needles = banned_substrings();
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        let mask = compute_test_mask(content, rel);
        // Use the state-carrying stripper so substrings that appear inside
        // multi-line `"..."` literals (the common false-positive: long
        // multi-line error messages that mention `panic!`, `unsafe`, etc.
        // by name) do not trip the scanner.
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in content.lines().enumerate() {
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            let in_test = mask.get(idx).copied().unwrap_or(false);
            for (needle, label, test_aware) in needles {
                if *test_aware && in_test {
                    continue;
                }
                if stripped.contains(needle) {
                    offenders.push((rel.to_path_buf(), idx + 1, *label, line.to_string()));
                }
            }
        }
    });
}

/// Compute a per-line bitmap of "is this line in test scope?". A line is in
/// test scope if either the file is a test/bench file (under `tests/`,
/// `bench/`, `benches/`, or `crates/*/tests|benches/`), or the line is
/// inside a brace block annotated with `#[cfg(test)]` / `#[cfg(all(test,
/// ...))]` / `#[cfg(any(test, ...))]`.
///
/// Brace tracking uses `strip_strings_and_comments` per line to ignore
/// braces inside string literals and `//` comments. Block comments and
/// raw strings are out of scope (same limitation as the other scanners).
fn compute_test_mask(content: &str, rel: &Path) -> Vec<bool> {
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();
    let mut mask = vec![false; n];

    // File-level test scope.
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    let file_is_test = rel_str.starts_with("tests/")
        || rel_str.starts_with("bench/")
        || rel_str.starts_with("benches/")
        || path_matches_crates_test(&rel_str);
    if file_is_test {
        mask.fill(true);
        return mask;
    }

    // Brace-tracked cfg(test) regions. We maintain a stack of entry brace
    // depths: when a `#[cfg(test)]`-style attribute is seen, we wait for
    // the next `{` to open the gated block, then pop when depth returns to
    // the entry level.
    let mut depth: i32 = 0;
    // Pending attribute: when Some, the very next `{` (which may be on the
    // same line as the attribute or several lines later) opens a gated
    // block.
    let mut pending_attr = false;
    // Stack of entry depths (depth at which the gate opens; we exit when
    // depth drops back to this value).
    let mut gate_stack: Vec<i32> = Vec::new();

    let stripped_all = strip_file_lines(content);
    for (idx, _raw) in lines.iter().enumerate() {
        let stripped = stripped_all.get(idx).cloned().unwrap_or_default();

        // Detect cfg-test attribute on this line. Attribute syntax is
        // `#[cfg(test)]`, `#[cfg(all(test, ...))]`, `#[cfg(any(test,
        // ...))]`. We accept the attribute anywhere on the line.
        if is_cfg_test_attr_line(&stripped) {
            pending_attr = true;
        }

        // Walk braces on this line.
        let bytes = stripped.as_bytes();
        // The line counts as "in test" if the line's starting depth is
        // already inside a gate. Compute before brace walk so the
        // attribute line itself and the brace-open line are not marked
        // (they belong to enclosing scope), and the brace-close line that
        // exits the gate is also not marked. Lines strictly inside the
        // gate ARE marked.
        let inside_at_line_start = !gate_stack.is_empty();
        mask[idx] = inside_at_line_start;

        for &b in bytes {
            if b == b'{' {
                depth += 1;
                if pending_attr {
                    // The brace that just opened belongs to the cfg(test)
                    // attribute target. The gate's "entry depth" is the
                    // outer depth, i.e. depth - 1: we exit when depth
                    // drops back to that value.
                    gate_stack.push(depth - 1);
                    pending_attr = false;
                }
            } else if b == b'}' {
                if let Some(&entry) = gate_stack.last()
                    && depth - 1 == entry
                {
                    gate_stack.pop();
                }
                depth -= 1;
            }
        }
    }

    mask
}

/// Match `crates/<name>/tests/...` or `crates/<name>/benches/...`.
fn path_matches_crates_test(rel: &str) -> bool {
    let Some(rest) = rel.strip_prefix("crates/") else {
        return false;
    };
    // Skip the crate name segment.
    let Some(slash) = rest.find('/') else {
        return false;
    };
    let tail = &rest[slash + 1..];
    tail.starts_with("tests/") || tail.starts_with("benches/")
}

/// Recognize `#[cfg(test)]`, `#[cfg(all(test, ...))]`, `#[cfg(any(test,
/// ...))]` on a stripped line. We match by locating `cfg(` after a `#[`
/// and inspecting the argument list for a bare `test` token (either
/// directly, or inside an `all(...)`/`any(...)` whose token list contains
/// `test`).
fn is_cfg_test_attr_line(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        if bytes[i] == b'#' && bytes[i + 1] == b'[' {
            // Find `cfg(` after this.
            let rest = &stripped[i + 2..];
            // Allow `cfg_attr` to NOT match — we want `cfg(...)`.
            if let Some(pos) = rest.find("cfg(") {
                // Ensure it's `cfg(` not e.g. `cfg_attr(` (`cfg_attr`
                // would include the underscore before `(`, but `find`
                // returns the first match; check the preceding byte to
                // ensure the `c` of `cfg(` is not preceded by an
                // identifier byte).
                let abs = i + 2 + pos;
                let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
                if before_ok {
                    // Extract the parenthesized argument list (balance
                    // parens).
                    let args_start = abs + 4; // past `cfg(`
                    if let Some(end) = find_matching_paren(&bytes[args_start..]) {
                        let args = &stripped[args_start..args_start + end];
                        if cfg_args_contain_test(args) {
                            return true;
                        }
                    }
                }
            }
        }
        i += 1;
    }
    false
}

/// Given bytes starting just *after* an opening `(`, find the index of
/// the matching closing `)`. Returns the position of the `)` relative to
/// the slice start, or None if unbalanced on this line.
fn find_matching_paren(bytes: &[u8]) -> Option<usize> {
    let mut depth: i32 = 1;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// True if the argument list inside `cfg(...)` mentions `test` either as
/// a bare token or inside an `all(...)` / `any(...)` whose own token
/// list contains `test`. The check is recursive in spirit but
/// implemented by scanning for a `test` whole-word token anywhere in the
/// flattened argument string — `cfg(all(test, foo))`, `cfg(any(foo,
/// test))`, and `cfg(test)` all reduce to "contains the bare word
/// `test`", which is the desired behavior since any of these spellings
/// gates the annotated item to the test build.
fn cfg_args_contain_test(args: &str) -> bool {
    let bytes = args.as_bytes();
    let kw = b"test";
    let mut i = 0usize;
    while i + kw.len() <= bytes.len() {
        if &bytes[i..i + kw.len()] == kw {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok = i + kw.len() == bytes.len() || !is_ident_byte(bytes[i + kw.len()]);
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Walks `unsafe` keyword sites and requires a `// SAFETY:` comment within
/// the same line or one of the 3 preceding non-blank lines. The `unsafe`
/// match is by-word so identifiers like `unsafely_typed` do not fire.
fn scan_for_unsafe_without_safety(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("unsafe") {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        // State-aware stripper so the `unsafe` token inside a multi-line
        // `"..."` literal (e.g., a long diagnostic message that names the
        // word "unsafe") is masked out before keyword detection.
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in lines.iter().enumerate() {
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            if !line_has_keyword(stripped, "unsafe") {
                continue;
            }
            if line.contains("SAFETY:") {
                continue;
            }
            // `unsafe impl Send` / `unsafe impl Sync` — the SAFETY
            // rationale belongs on the type's documentation, not the impl
            // line. Skip both spellings, whitespace-tolerant.
            if is_unsafe_marker_impl(stripped) {
                continue;
            }
            // Scan up to 3 preceding non-blank lines for `// SAFETY:`.
            let mut justified = false;
            let mut seen = 0usize;
            let mut k = idx;
            while k > 0 && seen < 3 {
                k -= 1;
                let prev = lines[k];
                if prev.trim().is_empty() {
                    continue;
                }
                seen += 1;
                if prev.contains("SAFETY:") {
                    justified = true;
                    break;
                }
            }
            if !justified {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Returns true when `stripped` declares `unsafe impl Send` or `unsafe
/// impl Sync` (with generics tolerated after the marker trait name).
/// Matches when the next non-whitespace byte after `Send`/`Sync` is `<`,
/// `for`, whitespace, or end-of-line — keeps the check from firing on
/// identifiers like `SendQueue`.
fn is_unsafe_marker_impl(stripped: &str) -> bool {
    for marker in ["Send", "Sync"] {
        let needle_owned = format!("unsafe impl {}", marker);
        let needle = needle_owned.as_str();
        let bytes = stripped.as_bytes();
        let nb = needle.as_bytes();
        let mut i = 0usize;
        while i + nb.len() <= bytes.len() {
            if &bytes[i..i + nb.len()] == nb {
                let after = i + nb.len();
                if after == bytes.len() {
                    return true;
                }
                let c = bytes[after];
                if c == b'<' || c.is_ascii_whitespace() {
                    return true;
                }
                if after + 3 <= bytes.len() && &bytes[after..after + 3] == b"for" {
                    return true;
                }
            }
            i += 1;
        }
    }
    false
}

/// Walks `transmute` keyword sites and requires a `// SAFETY:` comment
/// within the same line or one of the 3 preceding non-blank lines.
/// Whole-word match on `transmute`; additionally the stripped line must
/// contain `mem::` or `transmute::<` so unrelated identifiers are not
/// considered. Build.rs is exempt; strict everywhere else.
fn scan_for_transmute_without_safety(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("transmute") {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in lines.iter().enumerate() {
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            if !line_has_keyword(stripped, "transmute") {
                continue;
            }
            if !stripped.contains("mem::") && !stripped.contains("transmute::<") {
                continue;
            }
            if line.contains("SAFETY:") {
                continue;
            }
            let mut justified = false;
            let mut seen = 0usize;
            let mut k = idx;
            while k > 0 && seen < 3 {
                k -= 1;
                let prev = lines[k];
                if prev.trim().is_empty() {
                    continue;
                }
                seen += 1;
                if prev.contains("SAFETY:") {
                    justified = true;
                    break;
                }
            }
            if !justified {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Walks `panic!(` macro sites in non-test code and requires a `// SAFETY:`
/// comment within the same line or one of the 3 preceding non-blank lines.
/// Mirrors `scan_for_unsafe_without_safety`. The `panic!` family in the
/// history audit still treats any panic-shape in a former-marker body as
/// trivial, regardless of SAFETY comments — this softening applies only
/// to the lexical scanner.
fn scan_for_panic_without_safety(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("panic!(") {
            return;
        }
        let mask = compute_test_mask(content, rel);
        let lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in lines.iter().enumerate() {
            if mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            if !stripped.contains("panic!(") {
                continue;
            }
            if line.contains("SAFETY:") {
                continue;
            }
            let mut justified = false;
            let mut seen = 0usize;
            let mut k = idx;
            while k > 0 && seen < 3 {
                k -= 1;
                let prev = lines[k];
                if prev.trim().is_empty() {
                    continue;
                }
                seen += 1;
                if prev.contains("SAFETY:") {
                    justified = true;
                    break;
                }
            }
            if !justified {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Flags `#[cfg(any())]` (permanently false) and `#[cfg(all())]`
/// (permanently true) attributes. Whitespace inside the parens is
/// tolerated (e.g. `cfg(any( ))`). Operates on the stripped line and
/// requires `#[` on the same line to avoid matching commentary that
/// quotes the attribute syntax. Build.rs is exempt; strict everywhere
/// else.
fn scan_for_dead_cfg_gates(root: &Path, dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("cfg(") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let stripped = strip_strings_and_comments(line);
            if !stripped.contains("#[") {
                continue;
            }
            if line_has_empty_cfg_gate(&stripped) {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Flags any `#[cfg(...)]` / `#![cfg(...)]` / `#[cfg_attr(...)]` /
/// `#![cfg_attr(...)]` attribute whose argument list mentions a
/// `feature = "..."` predicate at any nesting depth. Cargo features
/// carve the codebase into conditionally-compiled forks: they multiply
/// the number of buildable configurations the test suite has to cover,
/// hide dead-code branches from one build's `dead_code` lint while
/// keeping them alive in another, and are the same "make this lint shut
/// up depending on context" family as `#[cfg(test)]` on pub items. The
/// project's autoderived behavior selects paths (e.g. GPU vs CPU) from
/// problem characteristics, not opt-in flags. Build.rs is exempt;
/// strict everywhere else (not test-aware — features are equally bad
/// in tests and benches).
fn scan_for_feature_cfg_gates(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("feature") {
            return;
        }
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in content.lines().enumerate() {
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            // Must be an attribute line (`#[...]` or `#![...]`) and
            // contain a `cfg(`/`cfg_attr(` invocation.
            if !(stripped.contains("#[") || stripped.contains("#![")) {
                continue;
            }
            if !(stripped.contains("cfg(") || stripped.contains("cfg_attr(")) {
                continue;
            }
            if line_has_feature_predicate(stripped) {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// True when `stripped` contains a `feature` identifier immediately
/// followed by optional whitespace and `=` — the shape of the
/// `feature = "..."` predicate Cargo expects inside `cfg(...)`. The
/// check is whole-word on the `feature` token so identifiers like
/// `featureset` or `not_a_feature` don't false-fire.
fn line_has_feature_predicate(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let kw = b"feature";
    let mut i = 0usize;
    while i + kw.len() <= bytes.len() {
        if &bytes[i..i + kw.len()] == kw {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_pos = i + kw.len();
            let after_ok = after_pos == bytes.len() || !is_ident_byte(bytes[after_pos]);
            if before_ok && after_ok {
                // Skip whitespace, then require `=`.
                let mut j = after_pos;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'=' {
                    return true;
                }
            }
        }
        i += 1;
    }
    false
}

/// Flags entries inside any `[features]` (or `[features.<sub>]`) table
/// in `Cargo.toml` files. A feature definition is itself banned —
/// without entries there is nothing for `cfg(feature = "...")` gates to
/// reference. Walks the manifest section-by-section, treating the
/// header line itself as exempt and flagging any `<ident> = <rhs>`
/// assignment underneath. Build.rs is exempt; the scanner only acts on
/// files whose basename is `Cargo.toml`.
fn scan_for_cargo_feature_entries(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let basename = rel
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("");
        if basename != "Cargo.toml" {
            return;
        }
        let mut in_features = false;
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            // Strip TOML `#` comments (TOML has no string-literal `#`
            // confusion at line scope worth handling here).
            let code_part = match trimmed.find('#') {
                Some(p) => trimmed[..p].trim_end(),
                None => trimmed,
            };
            if code_part.starts_with('[') && code_part.ends_with(']') {
                let header = code_part
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim();
                in_features = header == "features"
                    || header.starts_with("features.")
                    || header == "workspace.features"
                    || header.starts_with("workspace.features.");
                continue;
            }
            if !in_features {
                continue;
            }
            if code_part.is_empty() {
                continue;
            }
            // Need a `<ident> = ...` shape. Identifier characters per
            // TOML bare-key rules: letters, digits, `_`, `-`.
            let bytes = code_part.as_bytes();
            let mut j = 0usize;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' || bytes[j] == b'-')
            {
                j += 1;
            }
            if j == 0 {
                continue;
            }
            let mut k = j;
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if k < bytes.len() && bytes[k] == b'=' {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Flags `#[cfg(test)]` (and `#[cfg(all(test, ...))]` / `#[cfg(any(test,
/// ...))]`) attributes that directly annotate ANY item inside the `src/`
/// production tree. Visibility does NOT matter: a private
/// `#[cfg(test)] fn foo()` is the same dodge as a `pub fn foo()` —
/// rustc's `dead_code` lint never sees the item under the normal build
/// while the test target still picks it up. Tests exist to exercise
/// production code, not to keep other test-only code alive.
///
/// Walks forward through blank lines, `//` comments, and additional
/// `#[...]` attribute lines from the `cfg(test)` site until reaching the
/// first non-attribute, non-blank, non-comment line. That line is the
/// gated item.
///
/// Exempt items (the only legitimate uses of a top-level `#[cfg(test)]`
/// in `src/`): private test submodules. The trimmed item line must
/// match `mod <NAME> ...` where `<NAME>` is `tests`, `test_support`,
/// `tests_<...>`, or `<...>_tests`. `pub mod tests` IS flagged —
/// exporting a test-only module's interior is the same dodge.
///
/// `is_cfg_test_attr_line` already excludes `#[cfg(not(test))]`, so
/// negations cannot fire here.
///
/// Only files under `src/` (and `crates/<name>/src/`) are scanned —
/// test directories, benches, examples, and crate roots outside `src/`
/// are not "production surface." Build.rs is exempt.
fn scan_for_cfg_test_on_pub_items(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        // Only `src/` files are in scope. Test/bench/example trees and
        // any other path outside `src/` are skipped. `crates/<name>/src/`
        // is also production surface — accept it too.
        let in_src = rel_str.starts_with("src/")
            || rel_str
                .strip_prefix("crates/")
                .and_then(|rest| rest.find('/').map(|i| &rest[i + 1..]))
                .is_some_and(|tail| tail.starts_with("src/"));
        if !in_src {
            return;
        }
        if !content.contains("cfg(") {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(content);
        let n = lines.len();
        for idx in 0..n {
            let stripped = stripped_lines
                .get(idx)
                .map(String::as_str)
                .unwrap_or(lines[idx]);
            if !is_cfg_test_attr_line(stripped) {
                continue;
            }
            // Walk forward through blank lines, `//` comments, and other
            // `#[...]` attribute lines until reaching the first real item
            // line.
            let mut j = idx + 1;
            let mut item_line: Option<usize> = None;
            while j < n {
                let sj = stripped_lines.get(j).map(String::as_str).unwrap_or(lines[j]);
                let t = sj.trim();
                if t.is_empty() {
                    j += 1;
                    continue;
                }
                // `//` comments are stripped to whitespace by
                // `strip_file_lines`, so a comment-only original line shows
                // as empty here. Defensive: also accept a stripped line that
                // happens to start with `//`.
                if t.starts_with("//") {
                    j += 1;
                    continue;
                }
                if t.starts_with("#[") || t.starts_with("#![") {
                    j += 1;
                    continue;
                }
                item_line = Some(j);
                break;
            }
            let Some(item_idx) = item_line else {
                continue;
            };
            let item_raw = lines.get(item_idx).copied().unwrap_or("");
            let item_trim = item_raw.trim_start();
            // Exempt: private test submodules. Must literally begin with
            // `mod ` (not `pub mod`, not `pub(crate) mod`) and the module
            // name must be `tests`, `test_support`, `tests_*`, or
            // `*_tests`.
            if let Some(rest) = item_trim.strip_prefix("mod ") {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && is_exempt_test_submodule_name(&name) {
                    continue;
                }
            }
            // Capture a short item descriptor (up to the first `{`, `;`,
            // or 80 chars) as the report tag so the row shows both the
            // `cfg(test)` line and what it gates.
            let mut descriptor: String = item_trim
                .chars()
                .take_while(|c| *c != '{' && *c != ';' && *c != '\n')
                .collect();
            descriptor = descriptor.trim().chars().take(80).collect();
            offenders.push((
                rel.to_path_buf(),
                idx + 1,
                descriptor,
                lines[idx].to_string(),
            ));
        }
    });
}

fn is_exempt_test_submodule_name(name: &str) -> bool {
    name == "tests"
        || name == "test_support"
        || name.starts_with("tests_")
        || name.ends_with("_tests")
}

/// True when `stripped` contains `cfg(any(...))` or `cfg(all(...))` with
/// only whitespace between the inner parens.
fn line_has_empty_cfg_gate(stripped: &str) -> bool {
    for marker in ["cfg(any(", "cfg(all("] {
        let mut search_from = 0usize;
        while let Some(rel_pos) = stripped[search_from..].find(marker) {
            let inner_start = search_from + rel_pos + marker.len();
            let bytes = stripped.as_bytes();
            let mut j = inner_start;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b')' {
                return true;
            }
            search_from = inner_start;
        }
    }
    false
}

fn scan_for_bare_should_panic(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("#[should_panic") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#[should_panic]")
                || (trimmed.starts_with("#[should_panic") && !trimmed.contains("expected"))
            {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Flags `const <ident>: bool = false;` and `const <ident>: bool = true;`
/// item declarations. A bare boolean constant set to a literal is either a
/// dead-by-construction guard (`if !RUN_BENCH { return; }` patterns that
/// rustc's `dead_code` lint cannot catch because the const *could* in
/// principle be flipped) or a constant truth that does not need to exist.
/// Real toggles belong in `cfg`/`feature` gates or runtime configuration.
/// Build.rs is exempt.
fn scan_for_dead_guard_consts(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("const ") {
            return;
        }
        let trait_impl_mask = compute_trait_impl_mask(content);
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in content.lines().enumerate() {
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            if !line_has_keyword(stripped, "const") {
                continue;
            }
            if line_matches_bool_literal_const(stripped) {
                if trait_impl_mask.get(idx).copied().unwrap_or(false) {
                    // Trait-impl associated constants are required by the
                    // trait and cannot be replaced by `cfg`. Inherent-impl
                    // and free-item consts are still flagged.
                    continue;
                }
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Build a per-line bitmap of "is this line inside the body of a `impl
/// <Trait> for <Type>` block?". Used to suppress the dead-guard-const
/// check on trait-required associated constants. Brace tracking uses
/// `strip_strings_and_comments` per line so braces inside string literals
/// or `//` comments don't perturb depth. Inherent impls (`impl Foo { ... }`,
/// no ` for `) are NOT marked.
///
/// The detector tolerates multi-line impl headers: when a line opens a
/// brace, we look back up to 3 non-blank lines to find an `impl` token
/// and decide whether the header contains ` for `.
fn compute_trait_impl_mask(content: &str) -> Vec<bool> {
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();
    let mut mask = vec![false; n];
    let stripped_lines: Vec<String> = strip_file_lines(content);

    // Stack entry: depth-just-before-open, is_trait_impl. Only the FIRST `{`
    // following an `impl ...` header is matched with that header — subsequent
    // braces (fn bodies, match arms, struct literals, blocks) inside that
    // impl are classified as non-trait-impl scopes, so dead-guard consts
    // inside, say, a fn inside a trait impl still get correctly flagged.
    let mut stack: Vec<(i32, bool)> = Vec::new();
    let mut depth: i32 = 0;
    // True when we have seen `impl ` on a recent line but not yet found its
    // opening `{`. `pending_is_trait` tracks whether ` for ` was seen during
    // the header span.
    let mut pending_impl = false;
    let mut pending_is_trait = false;

    for (idx, stripped) in stripped_lines.iter().enumerate() {
        let inside_trait_impl = stack.iter().any(|(_, t)| *t);
        mask[idx] = inside_trait_impl;

        // Detect an item-position `impl` header opener. We accept the
        // trimmed line starting with `impl`, `pub impl` (rare), `unsafe impl`,
        // `default impl`, etc. — the common shape is `impl`, optionally
        // preceded by visibility/`unsafe`/`default` modifiers.
        let trimmed = stripped.trim_start();
        let starts_impl = trimmed.starts_with("impl ")
            || trimmed.starts_with("impl<")
            || trimmed.starts_with("unsafe impl ")
            || trimmed.starts_with("unsafe impl<")
            || trimmed.starts_with("default impl ")
            || trimmed.starts_with("default impl<");
        if starts_impl && depth == 0 {
            // Top-level impl: it opens a fresh trait-impl-or-inherent scope.
            pending_impl = true;
            pending_is_trait = stripped.contains(" for ");
        } else if pending_impl && stripped.contains(" for ") {
            // ` for ` showed up on a continuation line of the header.
            pending_is_trait = true;
        }

        let bytes = stripped.as_bytes();
        for &b in bytes {
            if b == b'{' {
                let opened_at_depth = depth;
                depth += 1;
                if pending_impl && opened_at_depth == 0 {
                    stack.push((opened_at_depth, pending_is_trait));
                    pending_impl = false;
                    pending_is_trait = false;
                } else {
                    // A non-impl brace (fn body, match, block, struct literal,
                    // etc.) — push it as a non-trait-impl scope so we balance
                    // correctly on `}` without claiming trait-impl context.
                    stack.push((opened_at_depth, false));
                }
            } else if b == b'}' {
                if let Some(&(entry, _)) = stack.last()
                    && depth - 1 == entry
                {
                    stack.pop();
                }
                depth -= 1;
            }
        }
    }

    mask
}

/// True if `stripped` contains a `const <ident>: bool = <true|false>;`
/// item-style declaration (visibility prefix tolerated; whitespace flexible
/// throughout). Operates on the stripped line.
fn line_matches_bool_literal_const(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let kw = b"const";
    let mut i = 0usize;
    'outer: while i + kw.len() <= bytes.len() {
        if &bytes[i..i + kw.len()] != kw {
            i += 1;
            continue;
        }
        let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
        let after_ok = i + kw.len() < bytes.len() && !is_ident_byte(bytes[i + kw.len()]);
        if !before_ok || !after_ok {
            i += 1;
            continue;
        }
        // Skip whitespace after `const`.
        let mut j = i + kw.len();
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        // Identifier (one or more ident bytes, must start with non-digit).
        if j >= bytes.len() || !(bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
            i += 1;
            continue;
        }
        while j < bytes.len() && is_ident_byte(bytes[j]) {
            j += 1;
        }
        // Whitespace, then `:`.
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b':' {
            i += 1;
            continue;
        }
        j += 1;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        // Type token `bool`.
        let bool_kw = b"bool";
        if j + bool_kw.len() > bytes.len() || &bytes[j..j + bool_kw.len()] != bool_kw {
            i += 1;
            continue;
        }
        let after = j + bool_kw.len();
        if after < bytes.len() && is_ident_byte(bytes[after]) {
            i += 1;
            continue;
        }
        j = after;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'=' {
            i += 1;
            continue;
        }
        j += 1;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        // Match `true` or `false`.
        for lit in [b"true".as_ref(), b"false".as_ref()] {
            if j + lit.len() <= bytes.len() && &bytes[j..j + lit.len()] == lit {
                let after_lit = j + lit.len();
                let bound_ok = after_lit == bytes.len() || !is_ident_byte(bytes[after_lit]);
                if bound_ok {
                    // Optional whitespace then `;`.
                    let mut k = after_lit;
                    while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                        k += 1;
                    }
                    if k < bytes.len() && bytes[k] == b';' {
                        return true;
                    }
                }
            }
        }
        i += 1;
        continue 'outer;
    }
    false
}

/// Flags lines where `eprintln!(` or `eprint!(` appears together with a
/// `{...:?}` / `{...:#?}` debug format spec. This is the hand-rolled `dbg!`
/// pattern: same intent (dump a value with `Debug` formatting to stderr),
/// different spelling. Test-aware via the shared `compute_test_mask`
/// machinery; build.rs is exempt.
///
/// The `eprintln!(` / `eprint!(` token check uses the stripped line (so
/// string-literal occurrences are ignored), but the `:?}` / `:#?}` probe
/// looks at the original line because `strip_strings_and_comments` blanks
/// out string-literal contents and would erase the format spec.
fn scan_for_debug_eprintln(root: &Path, dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("eprint") {
            return;
        }
        let mask = compute_test_mask(content, rel);
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in content.lines().enumerate() {
            if mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            if !stripped.contains("eprintln!(") && !stripped.contains("eprint!(") {
                continue;
            }
            if line.contains(":?}") || line.contains(":#?}") {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Per-line wrapper around `strip_strings_and_comments_stateful` that
/// assumes the line does not start inside an open string. Adequate for
/// most code lines but WRONG for any line that sits inside a multi-line
/// string literal — those need `strip_file_lines` to carry state across
/// line boundaries. Callers that scan for keywords like `unsafe` /
/// `panic` / `let _` (which can legitimately appear as words inside
/// long error-message strings) must use `strip_file_lines` instead, or
/// they'll false-positive on string content.
fn strip_strings_and_comments(line: &str) -> String {
    strip_strings_and_comments_stateful(line, false, 0).0
}

/// Walk every line of `content`, carrying string-open state across line
/// boundaries. Returns one stripped line per source line. This is the
/// correct path for any scanner whose keyword/substring could appear
/// inside a multi-line string literal (Rust permits real newlines inside
/// `"..."`, and `\<newline>` continuations are common in long error
/// messages). Raw strings (`r"..."`, `r#"..."#`) and block comments
/// (`/* ... */`) are still out of scope — the helper handles plain
/// double-quoted strings only, which covers the dominant false-positive
/// case (multi-line `format!`/`panic!`/`write!` message text).
fn strip_file_lines(content: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut in_str = false;
    let mut quote: u8 = 0;
    for line in content.lines() {
        let (stripped, after_in_str, after_quote) =
            strip_strings_and_comments_stateful(line, in_str, quote);
        out.push(stripped);
        in_str = after_in_str;
        quote = after_quote;
    }
    out
}

/// State-aware variant of `strip_strings_and_comments`. Takes the
/// "currently inside a string?" flag and the opening quote byte;
/// returns the stripped line and the post-line state.
fn strip_strings_and_comments_stateful(
    line: &str,
    in_str_in: bool,
    quote_in: u8,
) -> (String, bool, u8) {
    let bytes = line.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    let mut in_str = in_str_in;
    let mut str_quote: u8 = quote_in;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            if c == b'\\' && i + 1 < bytes.len() {
                out.push(b' ');
                out.push(b' ');
                i += 2;
                continue;
            }
            if c == str_quote {
                in_str = false;
                out.push(c);
            } else {
                out.push(b' ');
            }
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            break;
        }
        if c == b'"' {
            in_str = true;
            str_quote = c;
            out.push(c);
            i += 1;
            continue;
        }
        if c == b'\'' {
            // Distinguish char literal from Rust lifetime. A real char
            // literal is `'\...'` (escape) or `'X'` where X is a single
            // non-`'` byte and the closing `'` appears within ~4 bytes.
            // Otherwise (e.g. `'_`, `'static`, `'a,`, `'b>`), leave the
            // `'` alone — it's a lifetime, not a literal.
            let is_char_lit = if i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
                // Escape form: look for closing `'` within next 8 bytes.
                let mut k = i + 2;
                let mut found = false;
                while k < bytes.len() && k < i + 10 {
                    if bytes[k] == b'\'' {
                        found = true;
                        break;
                    }
                    k += 1;
                }
                found
            } else if i + 2 < bytes.len() && bytes[i + 1] != b'\'' && bytes[i + 2] == b'\'' {
                // Single-byte char literal `'X'`.
                true
            } else if i + 3 < bytes.len() && bytes[i + 1] != b'\'' && bytes[i + 3] == b'\'' {
                // Two-byte (rare; e.g. some unicode prefixes). Bounded
                // lookahead within ~4 bytes.
                true
            } else {
                false
            };
            if is_char_lit {
                in_str = true;
                str_quote = c;
                out.push(c);
                i += 1;
                continue;
            }
            // Lifetime: emit the apostrophe verbatim and continue.
            out.push(c);
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    let s = String::from_utf8(out).unwrap_or_else(|_| line.to_string());
    (s, in_str, str_quote)
}

/// True if `line` contains `kw` as a whole word (boundaries on both sides
/// are non-identifier bytes or string ends).
fn line_has_keyword(line: &str, kw: &str) -> bool {
    let bytes = line.as_bytes();
    let kw_bytes = kw.as_bytes();
    if kw_bytes.is_empty() || bytes.len() < kw_bytes.len() {
        return false;
    }
    let mut i = 0usize;
    while i + kw_bytes.len() <= bytes.len() {
        if &bytes[i..i + kw_bytes.len()] == kw_bytes {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok =
                i + kw_bytes.len() == bytes.len() || !is_ident_byte(bytes[i + kw_bytes.len()]);
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn scan_for_banned_marker(
    root: &Path,
    dir: &Path,
    needle: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        if !content.contains(needle) {
            return;
        }
        // For .rs files, strip multi-line string contents so error-message
        // strings that mention the marker by name don't false-fire. Non-.rs
        // files (toml, yaml, shell, etc.) fall through to the raw scan.
        let use_strip = rel.extension().and_then(OsStr::to_str) == Some("rs");
        let stripped_lines = if use_strip {
            Some(strip_file_lines(content))
        } else {
            None
        };
        for (idx, line) in content.lines().enumerate() {
            let probe: &str = match &stripped_lines {
                Some(v) => v.get(idx).map(String::as_str).unwrap_or(line),
                None => line,
            };
            if probe.contains(needle) {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Scan for `#[allow(...)]` / `#![allow(...)]` / `#[expect(...)]` /
/// `#![expect(...)]` — any lint, anywhere. Every such attribute is an
/// admission that some lint flagged real code and the author chose to hide
/// it instead of fix it. The single source of "we accept this category" is
/// `build.rs`; file-level allows/expects are forbidden. The leading
/// `clippy::` / `rustc::` / `rustdoc::` path prefix is tolerated when
/// labelling the first lint in the report row, but every lint token in the
/// parenthesized list triggers the ban regardless of name. Build.rs itself
/// is exempt (it names the attribute syntax as part of this scanner's own
/// contract). String / line-comment contents are masked via
/// `strip_file_lines` so attributes inside `"..."` or `//` text do not
/// false-fire.
fn scan_for_banned_allow(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String, String)>,
) {
    // Attribute prefixes that silence a lint. Both `allow` and `expect`
    // defeat the lint's signal — `expect` is the promotion form ("error if
    // the lint does NOT fire") which is just as load-bearing as `allow` for
    // hiding lint output. Treat them identically.
    const SILENCERS: &[&str] = &["allow(", "expect("];
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !SILENCERS.iter().any(|s| content.contains(s)) {
            return;
        }
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in content.lines().enumerate() {
            let code = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            // Only match when the silencer is part of an attribute on this
            // line: preceded somewhere by `#[` or `#![`. (The inner-attribute
            // form `#![` already contains `#[` as a substring, so a single
            // check covers both.)
            if !code.contains("#[") {
                continue;
            }
            for silencer in SILENCERS {
                let mut search_from = 0usize;
                while let Some(rel_idx) = code[search_from..].find(silencer) {
                    let abs_match = search_from + rel_idx;
                    // Require an attribute context: the silencer must be
                    // preceded (after optional whitespace) by `#[` or `#![`.
                    // Scan backwards over whitespace, then check the prefix
                    // ending — `#[` or `#![` — at that point.
                    let mut k = abs_match;
                    while k > 0 && code.as_bytes()[k - 1].is_ascii_whitespace() {
                        k -= 1;
                    }
                    let prefix = &code[..k];
                    let is_attr = prefix.ends_with("#[") || prefix.ends_with("#![");
                    let start = abs_match + silencer.len();
                    if !is_attr {
                        search_from = start;
                        continue;
                    }
                    let Some(end_rel) = code[start..].find(')') else {
                        break;
                    };
                    let inside = &code[start..start + end_rel];
                    // Collect lint tokens (split on `,`), respecting that
                    // nested parentheses won't appear at this level (a lint
                    // path like `clippy::foo` carries no parens; tool-lint
                    // configs that DO nest, e.g. `reason = "..."`, are not
                    // valid inside allow/expect anyway). Strip the known
                    // tool prefixes (`clippy::`, `rustc::`, `rustdoc::`) when
                    // labelling so the report shows the bare lint name.
                    let mut first_label: Option<String> = None;
                    let mut any_token = false;
                    for tok in inside.split(',') {
                        let trimmed = tok.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        any_token = true;
                        if first_label.is_none() {
                            let bare = trimmed
                                .trim_start_matches("clippy::")
                                .trim_start_matches("rustc::")
                                .trim_start_matches("rustdoc::");
                            first_label = Some(bare.to_string());
                        }
                    }
                    if any_token {
                        let attr = silencer.trim_end_matches('(');
                        let label = first_label.unwrap_or_else(|| "<empty>".to_string());
                        offenders.push((
                            rel.to_path_buf(),
                            idx + 1,
                            format!("{attr}({label})"),
                            line.to_string(),
                        ));
                    }
                    search_from = start + end_rel + 1;
                    if search_from >= code.len() {
                        break;
                    }
                }
            }
        }
    });
}

/// Scan for any `let _...` binding (bare `_`, `_name`, `mut _`, `mut _name`,
/// or tuple pattern `let (_, _) = ...;` where every binding is underscore-named).
/// Skips lines whose `let` is inside a `//`-line-comment or a string literal.
/// Multi-line raw strings and `/* ... */` blocks are out of scope: the scanner
/// is a line-level heuristic, not a full parser. Build.rs is exempt.
fn scan_for_let_underscore(root: &Path, dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("let ") {
            return;
        }
        let mask = compute_test_mask(content, rel);
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in content.lines().enumerate() {
            if mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            if stripped_line_has_let_underscore(stripped) {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// `let _...` detector that runs over an already-stripped line (string and
/// comment contents already masked to spaces). Looks for `let` followed by
/// an optional `mut` and then either a bare `_`-leading binding or a
/// tuple pattern in which every binding is underscore-prefixed. Catches:
///   `let _ = expr;`             `let _: T = expr;`        `let _name = expr;`
///   `let mut _name = expr;`     `let (_, _) = expr;`      `let (_a, _b) = expr;`
///   `let (mut _a, _b) = expr;`  `let (_, (_x, _y)) = expr;`
/// Does NOT flag `let (_, x) = expr;` because `x` is a real binding.
fn stripped_line_has_let_underscore(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if i + 3 < bytes.len()
            && &bytes[i..i + 3] == b"let"
            && (i == 0 || !is_ident_byte(bytes[i - 1]))
            && bytes[i + 3].is_ascii_whitespace()
        {
            let mut j = i + 3;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j + 3 <= bytes.len()
                && &bytes[j..j + 3] == b"mut"
                && j + 3 < bytes.len()
                && bytes[j + 3].is_ascii_whitespace()
            {
                j += 3;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
            }
            if j < bytes.len() && bytes[j] == b'_' {
                return true;
            }
            // Tuple-pattern discard: `let (...) = ...;`. Walk the parenthesized
            // pattern and flag iff every binding-name inside is `_` or `_<ident>`.
            // Conservative: if the heuristic cannot classify (path / struct /
            // refutable patterns, multi-line, type ascription, etc.), do not fire.
            if j < bytes.len()
                && bytes[j] == b'('
                && tuple_pattern_all_underscore(bytes, j) == Some(true)
            {
                return true;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    false
}

/// Walk a parenthesized pattern starting at `start` (which must point at `(`)
/// and decide whether every identifier-shaped binding inside is underscore-
/// prefixed. Returns `Some(true)` iff at least one binding exists and all are
/// underscore-named, `Some(false)` if any binding is a real name, and `None`
/// if the pattern is malformed / multi-line / contains constructs the
/// heuristic cannot classify (paths, struct patterns, refutable shapes like
/// `Some(x)`, type ascriptions, etc.).
fn tuple_pattern_all_underscore(bytes: &[u8], start: usize) -> Option<bool> {
    if start >= bytes.len() || bytes[start] != b'(' {
        return None;
    }
    let mut depth = 0i32;
    let mut k = start;
    let mut any_binding = false;
    let mut all_underscore = true;
    while k < bytes.len() {
        let b = bytes[k];
        match b {
            b'(' => {
                depth += 1;
                k += 1;
            }
            b')' => {
                depth -= 1;
                k += 1;
                if depth == 0 {
                    if !any_binding {
                        return None;
                    }
                    return Some(all_underscore);
                }
            }
            b',' | b' ' | b'\t' => {
                k += 1;
            }
            b'=' => {
                // Reached the `=` of the let without closing the outer paren
                // — malformed or multi-line. Bail conservatively.
                return None;
            }
            _ => {
                if b == b'_' || b.is_ascii_alphabetic() {
                    // Read identifier-shaped run.
                    let s = k;
                    while k < bytes.len() && is_ident_byte(bytes[k]) {
                        k += 1;
                    }
                    let word = &bytes[s..k];
                    // Skip pattern modifiers; they precede the actual binding.
                    if word == b"mut" || word == b"ref" {
                        continue;
                    }
                    // Look ahead: if the next non-space byte is `(`, `{`, or
                    // `::`, this is a path / struct / tuple-struct pattern
                    // (e.g. `Some(x)`, `Foo { .. }`, `path::Variant(..)`) —
                    // bail, the heuristic cannot classify it.
                    let mut m = k;
                    while m < bytes.len() && (bytes[m] == b' ' || bytes[m] == b'\t') {
                        m += 1;
                    }
                    if m < bytes.len() {
                        let nb = bytes[m];
                        if nb == b'(' || nb == b'{' {
                            return None;
                        }
                        if nb == b':' && m + 1 < bytes.len() && bytes[m + 1] == b':' {
                            return None;
                        }
                    }
                    any_binding = true;
                    if !word.starts_with(b"_") {
                        all_underscore = false;
                    }
                } else {
                    // Any other byte (`:` type ascription, `&`, `@`, digits,
                    // `..`, etc.) is outside the heuristic's competence.
                    return None;
                }
            }
        }
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn scan_for_ignored_tests(root: &Path, dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    visit_files(root, dir, &mut |rel, content| {
        // Skip the build script itself: it names the `#[ignore]` attribute
        // as part of this scanner's own contract.
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        // Only Rust files carry test attributes.
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("#[ignore") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            // Match `#[ignore]` and `#[ignore = "..."]`. The `#[ignore` prefix
            // is unique enough that no non-attribute construct collides with it.
            if trimmed.starts_with("#[ignore]") || trimmed.starts_with("#[ignore =") {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Walk `dir` recursively, calling `visitor(rel_path, content)` for every
/// scannable file. Centralizes the directory-skip and extension rules.
/// Flag every `vendor/` directory under `root` as a build-stop violation.
/// Walks the tree manually instead of riding `visit_files` so the
/// directory walker's own blocklist (which intentionally skips dotfiles,
/// build artifacts, etc.) does not silence this rule, and so an empty
/// `vendor/` is still reported.
fn scan_for_vendor_directories(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
        // Skip the same housekeeping noise `visit_files` skips, so we don't
        // descend into `.git`, `target/`, or the lake/python caches just to
        // chase a phantom `vendor/`.
        if name.starts_with('.')
            || name == "target"
            || name.starts_with("target-")
            || name == "node_modules"
            || name == "__pycache__"
            || name == "pydeps"
            || name == "site-packages"
            || name == "venv"
            || name == "dist"
            || name == "build"
            || name == "site"
        {
            continue;
        }
        if !path.is_dir() {
            continue;
        }
        if name == "vendor" {
            let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            offenders.push((
                rel,
                1,
                "vendored crates are forbidden; depend on crates.io or `git = \"…\"` in Cargo.toml"
                    .to_string(),
            ));
            // Don't descend into a forbidden tree — one report per vendor/
            // root is enough.
            continue;
        }
        scan_for_vendor_directories(root, &path, offenders);
    }
}

fn visit_files(root: &Path, dir: &Path, visitor: &mut dyn FnMut(&Path, &str)) {
    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
        if path
            .strip_prefix(root)
            .ok()
            .is_some_and(|rel| rel.starts_with("bench/runtime/pydeps"))
        {
            continue;
        }
        if name.starts_with('.')
            || name == "target"
            || name.starts_with("target-")
            || name == "node_modules"
            || name == "__pycache__"
            || name == "pydeps"
            || name == "site-packages"
            || name == "venv"
            || name == "dist"
            || name == "build"
            || name == "site"
        {
            continue;
        }
        if path.is_dir() {
            visit_files(root, &path, visitor);
            continue;
        }
        let ext = path.extension().and_then(OsStr::to_str).unwrap_or("");
        let basename = path.file_name().and_then(OsStr::to_str).unwrap_or("");
        let scannable = matches!(
            ext,
            "rs" | "py" | "toml" | "yml" | "yaml" | "sh" | "bash" | "json"
        ) || basename == "build.rs"
            || basename == "Makefile";
        if !scannable {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let content = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        visitor(&rel, &content);
    }
}

/// Flags function definitions whose parameter list contains an
/// underscore-prefixed name (e.g. `fn foo(_x: i32)`). Bare `_` placeholders
/// are allowed; only `_<ident>` is banned. Skips closures and `fn(...)`
/// type positions (the `fn` is not followed by an identifier). Handles
/// both `{ ... }` bodies and bodyless trait-method signatures
/// (`fn foo(_x: i32);`). Build.rs is exempt.
fn scan_for_underscore_fn_args(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("fn ") {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(content);
        let n = lines.len();
        let mut idx = 0usize;
        while idx < n {
            let stripped = stripped_lines
                .get(idx)
                .map(String::as_str)
                .unwrap_or(lines[idx]);
            if !line_has_keyword(stripped, "fn") {
                idx += 1;
                continue;
            }
            let Some(fn_pos) = locate_fn_keyword(stripped) else {
                idx += 1;
                continue;
            };
            let after_fn = &stripped[fn_pos + 2..];
            let trimmed_after = after_fn.trim_start();
            let first_byte = trimmed_after.as_bytes().first().copied();
            let is_definition =
                matches!(first_byte, Some(b) if b.is_ascii_alphabetic() || b == b'_');
            if !is_definition {
                idx += 1;
                continue;
            }
            let (sig_start, sig_end_line, sig_end_col_excl) = match find_fn_body_at(&lines, idx) {
                Some((_, (open, _close))) => {
                    let open_line = &stripped_lines[open];
                    let col = open_line.find('{').unwrap_or(open_line.len());
                    (idx, open, col)
                }
                None => {
                    let mut paren: i32 = 0;
                    let mut brack: i32 = 0;
                    let mut found: Option<(usize, usize)> = None;
                    let limit = (idx + 64).min(n);
                    'outer: for (rel, sj) in stripped_lines[idx..limit].iter().enumerate() {
                        let j = idx + rel;
                        for (k, b) in sj.as_bytes().iter().enumerate() {
                            match *b {
                                b'(' => paren += 1,
                                b')' => paren -= 1,
                                b'[' => brack += 1,
                                b']' => brack -= 1,
                                b';' if paren == 0 && brack == 0 => {
                                    found = Some((j, k));
                                    break 'outer;
                                }
                                b'{' if paren == 0 && brack == 0 => {
                                    break 'outer;
                                }
                                _ => {}
                            }
                        }
                    }
                    match found {
                        Some((j, k)) => (idx, j, k),
                        None => {
                            idx += 1;
                            continue;
                        }
                    }
                }
            };
            let mut sig_text = String::new();
            let mut line_offsets: Vec<(usize, usize)> = Vec::new();
            for (rel, stripped_line) in stripped_lines[sig_start..=sig_end_line].iter().enumerate()
            {
                let j = sig_start + rel;
                let part = if j == sig_end_line {
                    &stripped_line[..sig_end_col_excl]
                } else {
                    stripped_line.as_str()
                };
                line_offsets.push((sig_text.len(), j));
                sig_text.push_str(part);
                sig_text.push('\n');
            }
            let sig_bytes = sig_text.as_bytes();
            // Find the first `(` at angle-depth 0 — the actual parameter list
            // opener. A naive `find('(')` mistakes tuple types inside generic
            // bounds (e.g. `fn foo<T: Iter<Item = (i32, i32)>>(_x: i32)`) for
            // the param list and silently misses the real underscore-prefixed
            // params.
            let paren_open = {
                let mut ang: i32 = 0;
                let mut found: Option<usize> = None;
                for (k, &b) in sig_bytes.iter().enumerate() {
                    match b {
                        b'<' => ang += 1,
                        b'>' => {
                            if ang > 0 {
                                ang -= 1;
                            }
                        }
                        b'(' if ang == 0 => {
                            found = Some(k);
                            break;
                        }
                        _ => {}
                    }
                }
                match found {
                    Some(p) => p,
                    None => {
                        idx = sig_end_line + 1;
                        continue;
                    }
                }
            };
            let Some(close_rel) = find_matching_paren(&sig_bytes[paren_open + 1..]) else {
                idx = sig_end_line + 1;
                continue;
            };
            let params_inner = &sig_text[paren_open + 1..paren_open + 1 + close_rel];
            let inner_bytes = params_inner.as_bytes();
            let mut angle: i32 = 0;
            let mut paren_d: i32 = 0;
            let mut brack_d: i32 = 0;
            let mut start_byte: usize = 0;
            let mut params: Vec<(usize, usize)> = Vec::new();
            let mut k = 0usize;
            while k < inner_bytes.len() {
                let b = inner_bytes[k];
                match b {
                    b'(' => paren_d += 1,
                    b')' => paren_d -= 1,
                    b'[' => brack_d += 1,
                    b']' => brack_d -= 1,
                    b'<' => angle += 1,
                    b'>' => {
                        if angle > 0 {
                            angle -= 1;
                        }
                    }
                    b',' if angle == 0 && paren_d == 0 && brack_d == 0 => {
                        params.push((start_byte, k));
                        start_byte = k + 1;
                    }
                    _ => {}
                }
                k += 1;
            }
            if start_byte <= inner_bytes.len() {
                params.push((start_byte, inner_bytes.len()));
            }
            for (ps, pe) in params {
                let raw_param = &params_inner[ps..pe];
                let mut p = raw_param.trim_start();
                while p.starts_with("#[") {
                    if let Some(close) = p.find(']') {
                        p = p[close + 1..].trim_start();
                    } else {
                        break;
                    }
                }
                loop {
                    let before = p;
                    if p.starts_with("&mut ") {
                        p = p[5..].trim_start();
                    } else if p.starts_with('&') {
                        p = p[1..].trim_start();
                    } else if p.starts_with("mut ") {
                        p = p[4..].trim_start();
                    }
                    if p == before {
                        break;
                    }
                }
                let pb = p.as_bytes();
                if pb.is_empty() {
                    continue;
                }
                if pb.iter().all(|c| c.is_ascii_whitespace()) {
                    continue;
                }
                let mut end = 0usize;
                while end < pb.len() && (pb[end] == b'_' || pb[end].is_ascii_alphanumeric()) {
                    end += 1;
                }
                if end == 0 {
                    continue;
                }
                let name = &p[..end];
                let rest = p[end..].trim_start();
                if !rest.starts_with(':') {
                    continue;
                }
                if name.starts_with('_') && name.len() >= 2 {
                    // Compute the byte offset of the parameter NAME (not the
                    // leading whitespace/attribute/mut prefix) so the line
                    // mapping points to the underscore-prefixed identifier
                    // rather than the previous line's trailing comma.
                    let name_off_in_param = raw_param.find(name).unwrap_or(0);
                    let abs = paren_open + 1 + ps + name_off_in_param;
                    let mut hit_line = sig_start;
                    for (off, ln) in &line_offsets {
                        if *off <= abs {
                            hit_line = *ln;
                        } else {
                            break;
                        }
                    }
                    let raw = lines.get(hit_line).copied().unwrap_or("");
                    offenders.push((rel.to_path_buf(), hit_line + 1, raw.to_string()));
                }
            }
            idx = sig_end_line + 1;
        }
    });
}

/// Find the byte position of the `fn` keyword in `stripped` (whole-word).
fn locate_fn_keyword(stripped: &str) -> Option<usize> {
    let bytes = stripped.as_bytes();
    let mut i = 0usize;
    while i + 2 <= bytes.len() {
        if &bytes[i..i + 2] == b"fn" {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok = i + 2 == bytes.len() || !is_ident_byte(bytes[i + 2]);
            if before_ok && after_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Flags `#[test]` functions whose bodies contain no assertion-shaped
/// construct. Recognizes assert/debug_assert macros, panic-shape macros,
/// and `?` propagation. Tests carrying `#[should_panic]` are excluded.
/// Build.rs is exempt.
fn scan_for_useless_tests(root: &Path, dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("#[test]") {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(content);
        let n = lines.len();
        let mut i = 0usize;
        while i < n {
            let s = stripped_lines
                .get(i)
                .map(String::as_str)
                .unwrap_or(lines[i]);
            if !s.contains("#[test]") {
                i += 1;
                continue;
            }
            let mut has_should_panic = s.contains("#[should_panic");
            // `#[should_panic]` is commonly placed BEFORE `#[test]`. Walk back
            // through immediately-preceding attribute lines (and blank/
            // comment lines) so the ordering does not matter.
            let mut back = i;
            while back > 0 {
                back -= 1;
                let prev = stripped_lines
                    .get(back)
                    .map(String::as_str)
                    .unwrap_or(lines[back]);
                let pt = prev.trim();
                if pt.is_empty() || pt.starts_with("//") {
                    continue;
                }
                if pt.starts_with("#[") || pt.starts_with("#![") {
                    if prev.contains("#[should_panic") {
                        has_should_panic = true;
                    }
                    continue;
                }
                break;
            }
            let mut j = i + 1;
            while j < n {
                let sj = stripped_lines
                    .get(j)
                    .map(String::as_str)
                    .unwrap_or(lines[j]);
                let t = sj.trim();
                if t.is_empty() {
                    j += 1;
                    continue;
                }
                if t.starts_with("//") {
                    j += 1;
                    continue;
                }
                if t.starts_with("#[") || t.starts_with("#![") {
                    if sj.contains("#[should_panic") {
                        has_should_panic = true;
                    }
                    j += 1;
                    continue;
                }
                if line_has_keyword(sj, "fn") {
                    break;
                }
                j = n;
                break;
            }
            if j >= n {
                i += 1;
                continue;
            }
            if has_should_panic {
                i = j + 1;
                continue;
            }
            let Some((_sig, (open, close))) = find_fn_body_at(&lines, j) else {
                i = j + 1;
                continue;
            };
            let mut found = false;
            for (offset, raw_line) in lines[open..=close].iter().enumerate() {
                let k = open + offset;
                let line_s = stripped_lines
                    .get(k)
                    .map(String::as_str)
                    .unwrap_or(raw_line);
                if line_s.contains("assert!(")
                    || line_s.contains("assert_eq!(")
                    || line_s.contains("assert_ne!(")
                    || line_s.contains("debug_assert!(")
                    || line_s.contains("debug_assert_eq!(")
                    || line_s.contains("debug_assert_ne!(")
                    || line_s.contains("panic!(")
                    || line_s.contains("unreachable!(")
                    || line_s.contains("unimplemented!(")
                    || line_s.contains("todo!(")
                    || line_contains_propagating_question(line_s)
                {
                    found = true;
                    break;
                }
            }
            if !found {
                let raw = lines.get(i).copied().unwrap_or("");
                offenders.push((rel.to_path_buf(), i + 1, raw.to_string()));
            }
            i = close + 1;
        }
    });
}

/// True when `stripped` contains a `?` followed by `;`, `,`, `.`, `)`,
/// or end-of-line. Heuristic for `?`-propagation in a test body.
fn line_contains_propagating_question(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if bytes[i] == b'?' {
            let mut k = i + 1;
            while k < n && bytes[k] == b' ' {
                k += 1;
            }
            if k == n {
                return true;
            }
            let c = bytes[k];
            if matches!(c, b';' | b',' | b'.' | b')') {
                return true;
            }
        }
        i += 1;
    }
    false
}

// ─────────────────────────────────────────────────────────────────────────────
// Unimplemented-removal history audit.
//
// The lexical bans catch direct uses of `unimplemented!(` / `todo!(` /
// `unreachable!(` in non-test code. They do NOT catch the more subtle cheat
// of *deleting* a marker without writing real code in its place — e.g.
// `fn foo() { unimplemented!(); }` becoming `fn foo() { Err(NotImpl) }`. To
// catch that, we maintain a persistent ledger of every (file, fn-signature)
// pair that currently holds at least one marker. On each build we diff the
// ledger against the current scan: any pair that disappeared must now have a
// substantive function body, otherwise the build fails.
//
// The ledger is a tab-separated, sorted, line-oriented file at the repo root
// (`ban_history.txt`) so git merges resolve naturally as line-level diffs.
// Build.rs rewrites the file on every run; humans should not hand-edit it.
// ─────────────────────────────────────────────────────────────────────────────

const HISTORY_LEDGER_FILENAME: &str = "ban_history.txt";

/// The marker macros tracked by the audit. Same family as the lexical bans
/// for `unimplemented!` / `todo!` / `unreachable!` — every one is a runtime
/// "not done yet" panic dressed differently.
const HISTORY_MARKERS: &[&str] = &["unimplemented!(", "todo!(", "unreachable!("];

/// Macros that, if present anywhere in a function body, mean the body is
/// "still a panic" — even if the original marker was syntactically removed.
/// Includes the tracked markers themselves (catches "swap unimplemented for
/// unreachable") plus bare `panic!(` (catches "swap for a generic panic").
const HISTORY_BODY_REJECT_MACROS: &[&str] = &[
    "unimplemented!(",
    "todo!(",
    "unreachable!(",
    "panic!(",
    // Vacuous-assertion impostor for `unimplemented!`: matches both
    // `assert!(false)` and `assert!(false, "msg")` (no closing paren in
    // the needle).
    "assert!(false",
    // Process-termination impostors. Both qualified (`std::process::exit(`)
    // and unqualified (`process::exit(`) spellings end in `process::exit(`,
    // so a single needle covers both.
    "process::exit(",
    "process::abort(",
];

/// Minimum number of substantive (non-blank, non-comment, non-pure-brace,
/// non-panic) statement lines a function body must contain to count as
/// "actually implemented" after marker removal. Trivial single-line returns
/// (`Err(...)`, `Ok(())`, `default()`) fall below this threshold and are
/// flagged — if a function legitimately has nothing to do, the call site
/// should be deleted instead of leaving a stub.
const HISTORY_MIN_SUBSTANTIVE_BODY_LINES: usize = 2;

fn run_unimplemented_history_audit(
    manifest_dir: &Path,
    violations: &mut Vec<(PathBuf, usize, String, String)>,
) {
    let ledger_path = manifest_dir.join(HISTORY_LEDGER_FILENAME);
    println!("cargo:rerun-if-changed={}", ledger_path.display());

    // Step 1: scan the current tree. For each .rs file (non-test scope only)
    // collect the (file, normalized-fn-signature) pairs that contain at least
    // one history marker, along with the kinds of markers present.
    let mut current: std::collections::BTreeMap<(String, String), (PathBuf, usize, Vec<String>)> =
        std::collections::BTreeMap::new();
    let mut file_contents: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();

    visit_files(manifest_dir, manifest_dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        file_contents.insert(rel_str.clone(), content.to_string());

        let lines: Vec<&str> = content.lines().collect();
        let mask = compute_test_mask(content, rel);
        for (idx, line) in lines.iter().enumerate() {
            if mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let stripped = strip_strings_and_comments(line);
            for &marker in HISTORY_MARKERS {
                if !stripped.contains(marker) {
                    continue;
                }
                if let Some((sig, (open, _close))) = find_enclosing_fn(&lines, idx) {
                    let kind = marker.trim_end_matches('(').to_string();
                    let entry = current
                        .entry((rel_str.clone(), sig.clone()))
                        .or_insert_with(|| (rel.to_path_buf(), open + 1, Vec::new()));
                    if !entry.2.contains(&kind) {
                        entry.2.push(kind);
                    }
                    break;
                }
            }
        }
    });

    // Step 2: load the previous ledger (empty on first run).
    let previous = load_history_ledger(&ledger_path);

    // Step 3: build the next ledger from the current scan. Legitimately-
    // removed entries are simply absent from the next ledger; flagged
    // entries are carried forward so future builds re-check them.
    let mut next_ledger: std::collections::BTreeMap<(String, String), Vec<String>> =
        std::collections::BTreeMap::new();
    for ((file, sig), (_, _, kinds)) in &current {
        next_ledger.insert((file.clone(), sig.clone()), kinds.clone());
    }

    for ((file, sig), prev_kinds) in &previous {
        if current.contains_key(&(file.clone(), sig.clone())) {
            continue;
        }
        let state = match file_contents.get(file) {
            None => HistoryBodyState::FnAbsent,
            Some(c) => body_state_for_signature(c, sig),
        };
        match state {
            HistoryBodyState::FnAbsent | HistoryBodyState::Substantive => {
                // Pruned: legitimate removal (whole fn deleted, or body now
                // substantive).
            }
            HistoryBodyState::Trivial {
                fn_open_line,
                snippet,
            } => {
                violations.push((
                    PathBuf::from(file),
                    fn_open_line + 1,
                    sig.clone(),
                    format!(
                        "marker(s) [{}] removed but body still trivial / panic-shaped — first body line: {}",
                        prev_kinds.join(","),
                        snippet,
                    ),
                ));
                // Keep entry alive so a real fix is required to clear it.
                next_ledger.insert((file.clone(), sig.clone()), prev_kinds.clone());
            }
        }
    }

    // Step 4: write the next ledger if it differs from disk. Avoid spurious
    // mtime churn on identical content.
    save_history_ledger(&ledger_path, &next_ledger);
}

fn load_history_ledger(path: &Path) -> std::collections::BTreeMap<(String, String), Vec<String>> {
    let mut out: std::collections::BTreeMap<(String, String), Vec<String>> =
        std::collections::BTreeMap::new();
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return out,
    };
    for line in content.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(3, '\t').collect();
        if parts.len() != 3 {
            continue;
        }
        let file = parts[0].to_string();
        let sig = parts[1].to_string();
        let kinds: Vec<String> = parts[2]
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        out.insert((file, sig), kinds);
    }
    out
}

fn save_history_ledger(
    path: &Path,
    ledger: &std::collections::BTreeMap<(String, String), Vec<String>>,
) {
    let mut out = String::new();
    out.push_str(
        "# Persistent audit of `unimplemented!(` / `todo!(` / `unreachable!(`\n\
         # sites. Auto-managed by build.rs — do NOT hand-edit. Each non-comment\n\
         # line is tab-separated:\n\
         #   <relative_path>\\t<normalized_fn_signature>\\t<comma_marker_kinds>\n\
         # When an entry disappears from the source tree, build.rs inspects the\n\
         # enclosing function's body. Trivial bodies (under a few code lines)\n\
         # and bodies that still contain any panic-shape macro fail the build,\n\
         # so deleting a `unimplemented!` without writing a real implementation\n\
         # is caught. Legitimately-implemented removals auto-prune.\n\
         #\n\
         # Concurrent branches that both remove markers will produce additive\n\
         # line-level diff conflicts that resolve by union — the next build\n\
         # then re-validates each surviving entry.\n",
    );
    for ((file, sig), kinds) in ledger {
        out.push_str(file);
        out.push('\t');
        out.push_str(sig);
        out.push('\t');
        out.push_str(&kinds.join(","));
        out.push('\n');
    }
    match fs::read_to_string(path) {
        Ok(existing) if existing == out => return,
        _ => {}
    }
    if fs::write(path, out).is_err() {
        // Non-fatal: write failures (e.g. read-only checkout) leave the
        // ledger stale rather than failing the build. Future runs retry.
    }
}

enum HistoryBodyState {
    FnAbsent,
    Trivial {
        fn_open_line: usize,
        snippet: String,
    },
    Substantive,
}

/// Walk the file looking for a function whose normalized signature matches
/// `target_sig`. Returns the body shape: absent, trivial, or substantive.
/// "Substantive" requires at least `HISTORY_MIN_SUBSTANTIVE_BODY_LINES`
/// non-blank, non-comment, non-pure-brace code lines AND no occurrence of
/// any `HISTORY_BODY_REJECT_MACROS` macro in the body.
fn body_state_for_signature(content: &str, target_sig: &str) -> HistoryBodyState {
    let lines: Vec<&str> = content.lines().collect();
    let mut idx = 0;
    while idx < lines.len() {
        let stripped = strip_strings_and_comments(lines[idx]);
        if !line_has_keyword(&stripped, "fn") {
            idx += 1;
            continue;
        }
        if let Some((sig, (open, close))) = find_fn_body_at(&lines, idx) {
            if sig == target_sig {
                let mut code_lines = 0;
                let mut first_snippet: Option<String> = None;
                for raw in lines.iter().take(close + 1).skip(open) {
                    let s = strip_strings_and_comments(raw);
                    let t = s.trim();
                    if t.is_empty() {
                        continue;
                    }
                    if t.chars().all(|c| matches!(c, '{' | '}' | ' ')) {
                        continue;
                    }
                    for &m in HISTORY_BODY_REJECT_MACROS {
                        if s.contains(m) {
                            return HistoryBodyState::Trivial {
                                fn_open_line: open,
                                snippet: raw.trim().chars().take(120).collect::<String>(),
                            };
                        }
                    }
                    if first_snippet.is_none() {
                        first_snippet = Some(raw.trim().chars().take(120).collect::<String>());
                    }
                    code_lines += 1;
                }
                if code_lines < HISTORY_MIN_SUBSTANTIVE_BODY_LINES {
                    return HistoryBodyState::Trivial {
                        fn_open_line: open,
                        snippet: first_snippet.unwrap_or_else(|| "<empty body>".to_string()),
                    };
                }
                return HistoryBodyState::Substantive;
            }
            idx = close + 1;
            continue;
        }
        idx += 1;
    }
    HistoryBodyState::FnAbsent
}

/// Walk backward from `at_line` looking for the most recent line that bears
/// a `fn` keyword AND whose resulting body braces enclose `at_line`. Returns
/// `(normalized_signature, (body_open_line, body_close_line))`.
fn find_enclosing_fn(lines: &[&str], at_line: usize) -> Option<(String, (usize, usize))> {
    let mut start = at_line + 1;
    while start > 0 {
        start -= 1;
        let stripped = strip_strings_and_comments(lines[start]);
        if !line_has_keyword(&stripped, "fn") {
            continue;
        }
        if let Some((sig, (open, close))) = find_fn_body_at(lines, start)
            && open <= at_line
            && at_line <= close
        {
            return Some((sig, (open, close)));
        }
    }
    None
}

/// Starting at `fn_line` (a line containing the `fn` keyword), find the
/// function body's opening `{` and matching `}`. Brace counting uses
/// `strip_strings_and_comments` per line so braces inside string literals
/// or `//` comments don't perturb depth. Block comments and raw strings
/// are out of scope (same limitation as the other scanners). Returns the
/// normalized signature text — everything from `fn_line` up to (and
/// excluding) the body's opening `{`, whitespace-collapsed.
fn find_fn_body_at(lines: &[&str], fn_line: usize) -> Option<(String, (usize, usize))> {
    let mut depth: i32 = 0;
    let mut body_open: Option<usize> = None;
    for j in fn_line..lines.len() {
        let s = strip_strings_and_comments(lines[j]);
        for b in s.bytes() {
            match b {
                b'{' => {
                    depth += 1;
                    if body_open.is_none() {
                        body_open = Some(j);
                    }
                }
                b'}' => {
                    depth -= 1;
                    if depth == 0
                        && let Some(open) = body_open
                    {
                        let mut sig = String::new();
                        for (k, line) in lines.iter().enumerate().take(open + 1).skip(fn_line) {
                            let ss = strip_strings_and_comments(line);
                            let cut = if k == open {
                                ss.find('{').unwrap_or(ss.len())
                            } else {
                                ss.len()
                            };
                            sig.push_str(&ss[..cut]);
                            sig.push(' ');
                        }
                        let normalized: String =
                            sig.split_whitespace().collect::<Vec<_>>().join(" ");
                        return Some((normalized, (open, j)));
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Case-insensitive substring patterns that flag a function identifier as a
/// dodge-named lint launderer. The Codex agent that prompted this scanner
/// invented `fn discard_lambda_adjoint_for_fixed_lambda(self) {}` whose
/// name announces the dodge outright ("I exist to discard a value to
/// satisfy a linter"). Banning the names by string makes "rename it to
/// look legitimate" cost the same as actually fixing the code.
///
/// Survey notes (kept narrow where legitimate uses already exist):
/// - `_for_fixed_` would flag `fit_model_for_fixed_rho` (a real solver
///   entry point in `src/solver/pirls.rs` — "fit with rho held fixed"
///   is genuine mathematical naming). Narrowed to the specific forms
///   the dodge actually used (`_for_fixed_lambda`, `_for_fixed_arg`).
/// - `_placeholder_` would flag a legitimate test about a renderer
///   omitting placeholder metrics. Dropped; `placeholder_for_` retained.
fn dodge_name_substrings() -> &'static [&'static str] {
    &[
        "discard_",
        "_discard_",
        "swallow_",
        "_swallow_",
        "_for_fixed_lambda",
        "_for_fixed_arg",
        "_for_unused",
        "_for_no_op",
        "_no_op_",
        "silence_",
        "ignore_unused",
        "eat_unused",
        "consume_unused",
        "intentionally_unused",
        "placeholder_for_",
        "dummy_for_",
    ]
}

/// Scan `.rs` files for `fn <ident>` declarations whose identifier (case-
/// insensitive) contains any of the banned substrings in
/// `dodge_name_substrings`. Build.rs itself is exempt. Strict everywhere
/// else — the names are self-incriminating regardless of context.
fn scan_for_dodge_named_fns(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String, String)>,
) {
    let needles = dodge_name_substrings();
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("fn ") {
            return;
        }
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in content.lines().enumerate() {
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            if !line_has_keyword(stripped, "fn") {
                continue;
            }
            let Some(fn_pos) = locate_fn_keyword(stripped) else {
                continue;
            };
            // Identifier follows `fn`, optionally after whitespace. Must
            // start with an ident byte (alpha / `_`) and continue with
            // ident bytes — matches the `\bfn\s+(\w+)` shape from the task.
            let bytes = stripped.as_bytes();
            let mut j = fn_pos + 2;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= bytes.len() {
                continue;
            }
            if !(bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
                continue;
            }
            let name_start = j;
            while j < bytes.len() && is_ident_byte(bytes[j]) {
                j += 1;
            }
            if j == name_start {
                continue;
            }
            let name = &stripped[name_start..j];
            let lower = name.to_ascii_lowercase();
            for needle in needles {
                if lower.contains(needle) {
                    offenders.push((
                        rel.to_path_buf(),
                        idx + 1,
                        name.to_string(),
                        line.to_string(),
                    ));
                    break;
                }
            }
        }
    });
}

/// Flags functions whose body is empty (only whitespace between `{` and `}`)
/// AND whose first parameter is a by-value `self` receiver (`self`,
/// `mut self`, `self: T`, `mut self: T`). These are the wrapper-struct
/// laundering pattern: a no-op consume-self method whose only purpose is
/// to satisfy the `let _` ban (`x.discard()` instead of `let _ = x;`).
///
/// By-reference receivers (`&self`, `&mut self`, `self: &Self`,
/// `self: &mut Self`) are NOT flagged — empty `fn drop(&mut self) {}` is a
/// legitimate (if unusual) Drop impl shape.
///
/// Bench files are NOT exempt: laundering in benches is the same failure
/// mode. Build.rs itself IS exempt.
fn scan_for_noop_self_consuming_fns(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("fn ") {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(content);
        let n = lines.len();
        let mut i = 0usize;
        while i < n {
            let s = stripped_lines
                .get(i)
                .map(String::as_str)
                .unwrap_or(lines[i]);
            if !line_has_keyword(s, "fn") {
                i += 1;
                continue;
            }
            let Some((sig, (open, close))) = find_fn_body_at(&lines, i) else {
                i += 1;
                continue;
            };
            if !first_param_is_by_value_self(&sig) {
                i = open + 1;
                continue;
            }
            if fn_body_is_empty(&stripped_lines, open, close) {
                offenders.push((rel.to_path_buf(), i + 1, lines[i].to_string()));
            }
            i = close + 1;
        }
    });
}

/// Parse the first parameter from a normalized fn signature string
/// (whitespace collapsed, string/comment contents stripped). Returns true
/// iff the first parameter is a by-value `self` receiver: `self`,
/// `mut self`, `self: T`, `mut self: T` for any non-reference `T`. `&self`,
/// `&mut self`, `self: &Self`, `self: &mut Self` return false.
fn first_param_is_by_value_self(sig: &str) -> bool {
    let bytes = sig.as_bytes();
    let mut i = 0usize;
    let mut fn_pos: Option<usize> = None;
    while i + 2 <= bytes.len() {
        if &bytes[i..i + 2] == b"fn"
            && (i == 0 || !is_ident_byte(bytes[i - 1]))
            && (i + 2 == bytes.len() || !is_ident_byte(bytes[i + 2]))
        {
            fn_pos = Some(i);
            break;
        }
        i += 1;
    }
    let Some(fp) = fn_pos else {
        return false;
    };
    let mut j = fp + 2;
    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
        j += 1;
    }
    if j >= bytes.len() || !(bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
        return false;
    }
    while j < bytes.len() && is_ident_byte(bytes[j]) {
        j += 1;
    }
    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
        j += 1;
    }
    if j < bytes.len() && bytes[j] == b'<' {
        let mut depth: i32 = 0;
        while j < bytes.len() {
            match bytes[j] {
                b'<' => depth += 1,
                b'>' => {
                    depth -= 1;
                    if depth == 0 {
                        j += 1;
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
    }
    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
        j += 1;
    }
    if j >= bytes.len() || bytes[j] != b'(' {
        return false;
    }
    let params_start = j + 1;
    let Some(end) = find_matching_paren(&bytes[params_start..]) else {
        return false;
    };
    let params = &sig[params_start..params_start + end];
    let first = first_top_level_segment(params);
    let first_trim = first.trim();
    if first_trim.is_empty() {
        return false;
    }
    if first_trim.starts_with('&') {
        return false;
    }
    let after_mut = first_trim
        .strip_prefix("mut ")
        .map(str::trim_start)
        .unwrap_or(first_trim);
    if after_mut == "self" {
        return true;
    }
    if let Some(rest) = after_mut.strip_prefix("self") {
        if let Some(&b) = rest.as_bytes().first()
            && is_ident_byte(b)
        {
            return false;
        }
        let r = rest.trim_start();
        if let Some(tail) = r.strip_prefix(':') {
            let ty = tail.trim_start();
            if ty.starts_with('&') {
                return false;
            }
            return !ty.is_empty();
        }
    }
    false
}

/// Return the leading segment of `params` up to the first top-level `,`
/// (respecting `<>`/`()`/`[]` nesting). If no top-level comma exists,
/// returns the entire input.
fn first_top_level_segment(params: &str) -> &str {
    let bytes = params.as_bytes();
    let mut angle: i32 = 0;
    let mut paren: i32 = 0;
    let mut bracket: i32 = 0;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => angle += 1,
            b'>' => {
                if angle > 0 {
                    angle -= 1;
                }
            }
            b'(' => paren += 1,
            b')' => {
                if paren > 0 {
                    paren -= 1;
                }
            }
            b'[' => bracket += 1,
            b']' => {
                if bracket > 0 {
                    bracket -= 1;
                }
            }
            b',' if angle == 0 && paren == 0 && bracket == 0 => {
                return &params[..i];
            }
            _ => {}
        }
        i += 1;
    }
    params
}

/// True iff the body between the `{` on `open` and the `}` on `close` is
/// only whitespace. Uses the stripped lines so string-literal and comment
/// contents are masked out.
fn fn_body_is_empty(stripped_lines: &[String], open: usize, close: usize) -> bool {
    if open > close || open >= stripped_lines.len() || close >= stripped_lines.len() {
        return false;
    }
    let open_line = stripped_lines[open].as_str();
    let Some(open_brace) = open_line.find('{') else {
        return false;
    };
    if open == close {
        let tail = &open_line[open_brace + 1..];
        let Some(close_brace) = tail.rfind('}') else {
            return false;
        };
        return tail[..close_brace].chars().all(char::is_whitespace);
    }
    let open_tail = &open_line[open_brace + 1..];
    if !open_tail.chars().all(char::is_whitespace) {
        return false;
    }
    for line in &stripped_lines[open + 1..close] {
        if !line.chars().all(char::is_whitespace) {
            return false;
        }
    }
    let close_line = stripped_lines[close].as_str();
    let Some(close_brace) = close_line.rfind('}') else {
        return false;
    };
    close_line[..close_brace].chars().all(char::is_whitespace)
}

/// Flags multi-arg functions whose entire body is a single trivial sentinel
/// expression — the canonical "stub the impl to make the compile error
/// vanish" shape. A no-arg constructor that legitimately returns a
/// default-shaped value (e.g. `fn empty() -> Vec<u32> { Vec::new() }`) has
/// nothing to compute from its inputs because it has no inputs; the
/// threshold is therefore >= 2 parameters (counting `self` toward the
/// total). Trait-method declarations without bodies are skipped via the
/// depth-0 `;`-before-`{` check; test files / `#[cfg(test)]` regions are
/// exempt via `compute_test_mask`. Build.rs is exempt.
fn scan_for_stub_function_bodies(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("fn ") {
            return;
        }
        let mask = compute_test_mask(content, rel);
        let lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(content);
        let n = lines.len();
        let mut i = 0usize;
        while i < n {
            let s = stripped_lines
                .get(i)
                .map(String::as_str)
                .unwrap_or(lines[i]);
            if !line_has_keyword(s, "fn") {
                i += 1;
                continue;
            }
            if mask.get(i).copied().unwrap_or(false) {
                i += 1;
                continue;
            }
            // `find_fn_body_at` walks forward to the first balanced `{...}`
            // regardless of whether a `;` ended the signature earlier (trait
            // method declaration). Guard explicitly: if the signature ends
            // with `;` at depth 0 before any `{`, this is a declaration
            // only — skip.
            if fn_signature_ends_with_semicolon(&stripped_lines, i) {
                i += 1;
                continue;
            }
            let Some((sig, (open, close))) = find_fn_body_at(&lines, i) else {
                i += 1;
                continue;
            };
            let Some(param_count) = signature_param_count(&sig) else {
                i = close + 1;
                continue;
            };
            if param_count < 2 {
                i = close + 1;
                continue;
            }
            let body = extract_body_text(&stripped_lines, open, close);
            if body_is_trivial_sentinel(&body) {
                offenders.push((rel.to_path_buf(), i + 1, lines[i].to_string()));
            }
            i = close + 1;
        }
    });
}

/// True when the line starting at `fn_line` has a `;` at brace/paren/bracket
/// depth zero BEFORE any `{`. That spells a trait-method declaration
/// (`fn foo(...);`) rather than a definition, and the caller must skip it
/// — otherwise `find_fn_body_at` will attach the body of the next function
/// to this declaration and produce a spurious offender.
fn fn_signature_ends_with_semicolon(stripped_lines: &[String], fn_line: usize) -> bool {
    let mut paren: i32 = 0;
    let mut brack: i32 = 0;
    let mut angle: i32 = 0;
    let limit = (fn_line + 64).min(stripped_lines.len());
    for line in &stripped_lines[fn_line..limit] {
        for &b in line.as_bytes() {
            match b {
                b'(' => paren += 1,
                b')' => paren -= 1,
                b'[' => brack += 1,
                b']' => brack -= 1,
                b'<' => angle += 1,
                b'>' => {
                    if angle > 0 {
                        angle -= 1;
                    }
                }
                b';' if paren == 0 && brack == 0 => return true,
                b'{' if paren == 0 && brack == 0 => return false,
                _ => {}
            }
        }
    }
    false
}

/// Count parameters in a normalized fn signature. Returns `None` if the
/// signature is malformed. `self` / `&self` / `&mut self` / `mut self`
/// count as one parameter, matching the natural reading of arity.
fn signature_param_count(sig: &str) -> Option<usize> {
    let bytes = sig.as_bytes();
    let mut i = 0usize;
    let mut fn_pos: Option<usize> = None;
    while i + 2 <= bytes.len() {
        if &bytes[i..i + 2] == b"fn"
            && (i == 0 || !is_ident_byte(bytes[i - 1]))
            && (i + 2 == bytes.len() || !is_ident_byte(bytes[i + 2]))
        {
            fn_pos = Some(i);
            break;
        }
        i += 1;
    }
    let fp = fn_pos?;
    let mut j = fp + 2;
    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
        j += 1;
    }
    if j >= bytes.len() || !(bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
        return None;
    }
    while j < bytes.len() && is_ident_byte(bytes[j]) {
        j += 1;
    }
    let mut k = j;
    while k < bytes.len() && bytes[k].is_ascii_whitespace() {
        k += 1;
    }
    // Skip generic param list `<...>` if present.
    if k < bytes.len() && bytes[k] == b'<' {
        let mut ang: i32 = 0;
        while k < bytes.len() {
            match bytes[k] {
                b'<' => ang += 1,
                b'>' => {
                    ang -= 1;
                    if ang == 0 {
                        k += 1;
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
    }
    // Find the param-list `(` at angle-depth 0.
    let mut ang2: i32 = 0;
    let mut open: Option<usize> = None;
    while k < bytes.len() {
        match bytes[k] {
            b'<' => ang2 += 1,
            b'>' => {
                if ang2 > 0 {
                    ang2 -= 1;
                }
            }
            b'(' if ang2 == 0 => {
                open = Some(k);
                break;
            }
            _ => {}
        }
        k += 1;
    }
    let op = open?;
    let inner_start = op + 1;
    let end_rel = find_matching_paren(&bytes[inner_start..])?;
    let inner = &sig[inner_start..inner_start + end_rel];
    if inner.trim().is_empty() {
        return Some(0);
    }
    let mut count = 0usize;
    let mut a: i32 = 0;
    let mut p: i32 = 0;
    let mut br: i32 = 0;
    let mut seen_nonws = false;
    for &c in inner.as_bytes() {
        match c {
            b'<' => {
                a += 1;
                seen_nonws = true;
            }
            b'>' => {
                if a > 0 {
                    a -= 1;
                }
                seen_nonws = true;
            }
            b'(' => {
                p += 1;
                seen_nonws = true;
            }
            b')' => {
                p -= 1;
                seen_nonws = true;
            }
            b'[' => {
                br += 1;
                seen_nonws = true;
            }
            b']' => {
                br -= 1;
                seen_nonws = true;
            }
            b',' if a == 0 && p == 0 && br == 0 => {
                if seen_nonws {
                    count += 1;
                }
                seen_nonws = false;
            }
            x if !(x as char).is_whitespace() => seen_nonws = true,
            _ => {}
        }
    }
    if seen_nonws {
        count += 1;
    }
    Some(count)
}

/// Read the body between the `{` on `open` and the `}` on `close` from
/// pre-stripped (string/comment-free) source lines and return the trimmed
/// joined text, single-space separated.
fn extract_body_text(stripped_lines: &[String], open: usize, close: usize) -> String {
    if open > close || close >= stripped_lines.len() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    for k in open..=close {
        let line = stripped_lines[k].as_str();
        let segment: &str = if k == open && k == close {
            let after_open = line.find('{').map(|p| p + 1).unwrap_or(line.len());
            let before_close = line[after_open..]
                .rfind('}')
                .map(|p| after_open + p)
                .unwrap_or(line.len());
            &line[after_open..before_close]
        } else if k == open {
            let after_open = line.find('{').map(|p| p + 1).unwrap_or(line.len());
            &line[after_open..]
        } else if k == close {
            let before_close = line.rfind('}').unwrap_or(line.len());
            &line[..before_close]
        } else {
            line
        };
        let t = segment.trim();
        if !t.is_empty() {
            parts.push(t.to_string());
        }
    }
    parts.join(" ").trim().to_string()
}

/// True iff `body` is exactly one of the trivial sentinel expressions a
/// real implementation would never reduce to: `None`, `Ok(())`,
/// `Err("...".into())`, `Default::default()` / `T::default()`, `Vec::new()`,
/// `vec![]`, `HashMap::new()` / `BTreeMap::new()` / similar no-arg standard
/// constructors, `Array1::zeros(0)` / `Array2::zeros((0, 0))` /
/// `Array3::zeros((0, 0, 0))`, `Some(Default::default())` /
/// `Some(T::default())`, the empty tuple `()`, or a bare numeric / bool /
/// empty-string literal. A trailing `;` is tolerated.
fn body_is_trivial_sentinel(body: &str) -> bool {
    let mut s = body.trim();
    if let Some(stripped) = s.strip_suffix(';') {
        s = stripped.trim_end();
    }
    if s.is_empty() {
        // Empty body is handled by `scan_for_noop_self_consuming_fns`; don't
        // double-report here.
        return false;
    }
    if matches!(
        s,
        "None"
            | "Ok(())"
            | "Default::default()"
            | "Vec::new()"
            | "vec![]"
            | "()"
            | "HashMap::new()"
            | "BTreeMap::new()"
            | "HashSet::new()"
            | "BTreeSet::new()"
            | "VecDeque::new()"
            | "String::new()"
            | "Some(Default::default())"
            | "true"
            | "false"
            | "\"\""
    ) {
        return true;
    }
    if let Some(prefix) = s.strip_suffix("::default()")
        && is_path_like(prefix)
    {
        return true;
    }
    if let Some(inner) = s.strip_prefix("Some(").and_then(|r| r.strip_suffix(')'))
        && let Some(prefix) = inner.strip_suffix("::default()")
        && is_path_like(prefix)
    {
        return true;
    }
    if let Some(inner) = s.strip_prefix("Err(").and_then(|r| r.strip_suffix(')'))
        && err_arg_is_string_literal(inner.trim())
    {
        return true;
    }
    if matches!(
        s,
        "Array1::zeros(0)" | "Array2::zeros((0, 0))" | "Array3::zeros((0, 0, 0))"
    ) {
        return true;
    }
    if is_bare_numeric_literal(s) {
        return true;
    }
    false
}

/// True when `s` looks like a path expression: identifier chars, plus `::`,
/// `<`, `>`, `,`, whitespace. Catches `Foo`, `path::Foo`, `Foo<T>`,
/// `crate::path::Bar`.
fn is_path_like(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    s.bytes()
        .all(|b| is_ident_byte(b) || b == b':' || b == b'<' || b == b'>' || b == b',' || b == b' ')
}

/// True when `arg` is a string-literal expression possibly followed by an
/// `.into()` / `.to_string()` / `.to_owned()` conversion. After
/// `strip_strings_and_comments` the literal contents are blanked to
/// spaces but the surrounding quotes survive.
fn err_arg_is_string_literal(arg: &str) -> bool {
    let bytes = arg.as_bytes();
    if bytes.is_empty() || bytes[0] != b'"' {
        return false;
    }
    let mut i = 1usize;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if bytes[i] == b'"' {
            i += 1;
            let tail = arg[i..].trim();
            return tail.is_empty()
                || tail == ".into()"
                || tail == ".to_string()"
                || tail == ".to_owned()";
        }
        i += 1;
    }
    false
}

/// True when `s` is a bare numeric literal: integer or float, optionally
/// suffixed (`0`, `0.0`, `0_usize`, `1u8`, `2.5f64`).
fn is_bare_numeric_literal(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut i = 0usize;
    let mut saw_digit = false;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
        if bytes[i].is_ascii_digit() {
            saw_digit = true;
        }
        i += 1;
    }
    if !saw_digit {
        return false;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
            i += 1;
        }
    }
    while i < bytes.len() && is_ident_byte(bytes[i]) {
        i += 1;
    }
    i == bytes.len()
}

/// Cross-file scanner: items defined in `src/` whose identifier is named by
/// a test/bench file but NOT by any other production `src/` file. Catches
/// items kept alive only because a test references them (rustc's
/// `dead_code` lint can't see this because the test target is a consumer).
///
/// Lexical and heuristic — false positives are accepted. To control noise:
///   * Only definitions with NON-`pub` visibility are scanned (`pub(crate)`,
///     `pub(super)`, `pub(in path)`, or no visibility modifier). Truly
///     `pub` items may be consumed by external callers we can't see.
///   * Definitions inside `impl <Trait> for <Type>` blocks are skipped —
///     trait dispatch is not lexically trackable.
///   * Identifiers re-exported via `pub use` from `src/lib.rs` /
///     `src/main.rs` (and the crates/* equivalents) count as production
///     usage.
///   * Definitions inside `#[cfg(test)]` regions of `src/` files are not
///     collected at all — those are test-only and outside scope (sibling
///     scanners already cover that case).
///   * Common-name exemption list and `< 3 char` length cut the noisiest
///     std/third-party collisions.
fn scan_for_src_items_used_only_by_tests(
    root: &Path,
    offenders: &mut Vec<(PathBuf, usize, String, String)>,
) {
    const EXEMPT_NAMES: &[&str] = &[
        "new",
        "default",
        "iter",
        "len",
        "is_empty",
        "clone",
        "from",
        "into",
        "as_ref",
        "as_mut",
        "next",
        "build",
        "with",
        "to_string",
        "display",
        "fmt",
        "index",
        "borrow",
        "drop",
        "main",
        "deref",
        "deref_mut",
        "hash",
        "eq",
        "ne",
        "cmp",
        "partial_cmp",
        "iter_mut",
        "into_iter",
        "as_slice",
        "as_str",
    ];

    struct SrcFile {
        rel: PathBuf,
        content: String,
        mask: Vec<bool>,
        stripped: Vec<String>,
        trait_impl: Vec<bool>,
    }
    let mut src_files: Vec<SrcFile> = Vec::new();
    let mut test_contents: Vec<(PathBuf, String)> = Vec::new();

    visit_files(root, root, &mut |rel, content| {
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let is_test = rel_str.starts_with("tests/")
            || rel_str.starts_with("bench/")
            || rel_str.starts_with("benches/")
            || path_matches_crates_test(&rel_str);
        if is_test {
            test_contents.push((rel.to_path_buf(), content.to_string()));
            return;
        }
        let is_src = rel_str.starts_with("src/")
            || (rel_str.starts_with("crates/") && rel_str.contains("/src/"));
        if !is_src {
            return;
        }
        let mask = compute_test_mask(content, rel);
        let stripped = strip_file_lines(content);
        let trait_impl = compute_trait_impl_mask(content);
        src_files.push(SrcFile {
            rel: rel.to_path_buf(),
            content: content.to_string(),
            mask,
            stripped,
            trait_impl,
        });
    });

    // Extract definitions from src/ files.
    let mut defs: Vec<(usize, usize, String)> = Vec::new();
    for (fi, sf) in src_files.iter().enumerate() {
        for (idx, stripped) in sf.stripped.iter().enumerate() {
            if sf.mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let trimmed = stripped.trim_start();
            let (vis, rest) = strip_leading_visibility(trimmed);
            if matches!(vis, Visibility::Public) {
                continue;
            }
            let rest = strip_leading_item_modifiers(rest);
            let Some(ident) = extract_item_ident(rest) else {
                continue;
            };
            if ident.starts_with('_') || ident.len() <= 2 {
                continue;
            }
            if EXEMPT_NAMES.contains(&ident.as_str()) {
                continue;
            }
            if sf.trait_impl.get(idx).copied().unwrap_or(false) {
                continue;
            }
            defs.push((fi, idx, ident));
        }
    }

    if defs.is_empty() {
        return;
    }

    // Per-file production token set (whole file, test-masked).
    let mut per_file_tokens: Vec<std::collections::HashSet<String>> =
        Vec::with_capacity(src_files.len());
    for sf in &src_files {
        let mut set: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (idx, stripped) in sf.stripped.iter().enumerate() {
            if sf.mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            extract_ident_tokens_into(stripped, &mut set);
        }
        per_file_tokens.push(set);
    }

    // Per-file per-line token sets (to detect same-file non-definition refs).
    let mut defining_file_token_lines: Vec<Vec<std::collections::HashSet<String>>> =
        Vec::with_capacity(src_files.len());
    for sf in &src_files {
        let mut v: Vec<std::collections::HashSet<String>> = Vec::with_capacity(sf.stripped.len());
        for (idx, stripped) in sf.stripped.iter().enumerate() {
            let mut set: std::collections::HashSet<String> = std::collections::HashSet::new();
            if !sf.mask.get(idx).copied().unwrap_or(false) {
                extract_ident_tokens_into(stripped, &mut set);
            }
            v.push(set);
        }
        defining_file_token_lines.push(v);
    }

    // Test reference set with first-hit hint.
    let mut test_first_hit: std::collections::HashMap<String, PathBuf> =
        std::collections::HashMap::new();
    for (rel, content) in &test_contents {
        let mut local: std::collections::HashSet<String> = std::collections::HashSet::new();
        for line in content.lines() {
            let stripped = strip_strings_and_comments(line);
            extract_ident_tokens_into(&stripped, &mut local);
        }
        for tok in local {
            test_first_hit.entry(tok).or_insert_with(|| rel.clone());
        }
    }
    if test_first_hit.is_empty() {
        return;
    }

    // `pub use` re-exports from src/lib.rs / src/main.rs (and crates/* equivs)
    // count as production usage.
    let mut pub_use_idents: std::collections::HashSet<String> = std::collections::HashSet::new();
    for sf in &src_files {
        let rel_str = sf.rel.to_string_lossy().replace('\\', "/");
        let is_lib_root = rel_str == "src/lib.rs"
            || rel_str.ends_with("/src/lib.rs")
            || rel_str == "src/main.rs"
            || rel_str.ends_with("/src/main.rs");
        if !is_lib_root {
            continue;
        }
        for stripped in &sf.stripped {
            collect_pub_use_idents(stripped, &mut pub_use_idents);
        }
    }

    for (fi, line_idx, ident) in defs {
        if pub_use_idents.contains(&ident) {
            continue;
        }
        let Some(test_hint) = test_first_hit.get(&ident) else {
            continue;
        };
        let mut prod_consumer = false;
        for (other_fi, set) in per_file_tokens.iter().enumerate() {
            if other_fi == fi {
                continue;
            }
            if set.contains(&ident) {
                prod_consumer = true;
                break;
            }
        }
        if !prod_consumer {
            let token_lines = &defining_file_token_lines[fi];
            for (li, toks) in token_lines.iter().enumerate() {
                if li == line_idx {
                    continue;
                }
                if toks.contains(&ident) {
                    prod_consumer = true;
                    break;
                }
            }
        }
        if prod_consumer {
            continue;
        }
        let sf = &src_files[fi];
        let raw = sf.content.lines().nth(line_idx).unwrap_or("").to_string();
        let hint = format!(
            "{} (test ref: {})",
            ident,
            test_hint.to_string_lossy().replace('\\', "/")
        );
        offenders.push((sf.rel.clone(), line_idx + 1, hint, raw));
    }
}

#[derive(Clone, Copy)]
enum Visibility {
    Private,
    Public,
    PubScoped,
}

/// Strip a leading visibility modifier from `trimmed`. Returns the
/// classification and the remainder after the modifier and any trailing
/// whitespace.
fn strip_leading_visibility(trimmed: &str) -> (Visibility, &str) {
    if let Some(rest) = trimmed.strip_prefix("pub(") {
        let bytes = rest.as_bytes();
        let mut depth = 1i32;
        let mut i = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        let after = rest[i + 1..].trim_start();
                        return (Visibility::PubScoped, after);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        return (Visibility::PubScoped, "");
    }
    if let Some(rest) = trimmed.strip_prefix("pub ") {
        return (Visibility::Public, rest.trim_start());
    }
    if trimmed == "pub" {
        return (Visibility::Public, "");
    }
    (Visibility::Private, trimmed)
}

/// Strip leading per-item modifiers (`unsafe`, `async`, `default`,
/// `const fn`, `extern "..."`) before the item-kind keyword.
fn strip_leading_item_modifiers(mut s: &str) -> &str {
    loop {
        let before = s;
        if let Some(rest) = s.strip_prefix("unsafe ") {
            s = rest.trim_start();
            continue;
        }
        if let Some(rest) = s.strip_prefix("async ") {
            s = rest.trim_start();
            continue;
        }
        if let Some(rest) = s.strip_prefix("default ") {
            s = rest.trim_start();
            continue;
        }
        if s.strip_prefix("const fn ").is_some() {
            // `const fn` always denotes a fn item — caller's
            // `extract_item_ident` expects to see the `fn ` keyword
            // first, so return a slice of the ORIGINAL `s` starting at
            // the "fn " position (offset 6 inside "const fn ").
            return &s[6..];
        }
        if let Some(rest) = s.strip_prefix("extern ") {
            let r = rest.trim_start();
            if r.starts_with('"') {
                if let Some(end) = r[1..].find('"') {
                    s = r[end + 2..].trim_start();
                    continue;
                }
            }
            s = r;
            continue;
        }
        if s == before {
            return s;
        }
    }
}

/// Extract the item identifier from a line beginning with an item-kind
/// keyword (`fn`, `struct`, `enum`, `const`, `static`, `type`, `trait`).
/// Returns None if `s` does not begin with one of those.
fn extract_item_ident(s: &str) -> Option<String> {
    for kw in ["fn ", "struct ", "enum ", "const ", "static ", "type ", "trait "] {
        if let Some(rest) = s.strip_prefix(kw) {
            let rest = rest.trim_start();
            let bytes = rest.as_bytes();
            if bytes.is_empty() {
                return None;
            }
            if !(bytes[0].is_ascii_alphabetic() || bytes[0] == b'_') {
                return None;
            }
            let mut end = 0usize;
            while end < bytes.len() && is_ident_byte(bytes[end]) {
                end += 1;
            }
            if end == 0 {
                return None;
            }
            return Some(rest[..end].to_string());
        }
    }
    None
}

/// Extract identifier-shaped tokens from `stripped` into `out`. Runs of
/// `[a-zA-Z_][a-zA-Z0-9_]*` (length >= 1) are collected; numeric-leading
/// runs are skipped as numeric literals.
fn extract_ident_tokens_into(stripped: &str, out: &mut std::collections::HashSet<String>) {
    let bytes = stripped.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        let b = bytes[i];
        if b.is_ascii_alphabetic() || b == b'_' {
            let s = i;
            while i < n && is_ident_byte(bytes[i]) {
                i += 1;
            }
            out.insert(stripped[s..i].to_string());
        } else if b.is_ascii_digit() {
            while i < n && (is_ident_byte(bytes[i]) || bytes[i] == b'.') {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}

/// Collect identifiers re-exported by a `pub use` (or `pub(crate) use`)
/// statement on `stripped`. Captures the trailing-segment ident of each
/// path: `pub use foo::Bar;` → `Bar`; `pub use foo::{A, B as C};` →
/// `A`, `C`. Wildcards yield nothing.
fn collect_pub_use_idents(stripped: &str, out: &mut std::collections::HashSet<String>) {
    let t = stripped.trim_start();
    let rest = if let Some(r) = t.strip_prefix("pub use ") {
        r
    } else if let Some(r) = t.strip_prefix("pub(crate) use ") {
        r
    } else {
        return;
    };
    let rest = rest.trim_end().trim_end_matches(';').trim();
    collect_use_tree_idents(rest, out);
}

fn collect_use_tree_idents(tree: &str, out: &mut std::collections::HashSet<String>) {
    if let Some(brace_start) = tree.find('{') {
        let prefix = &tree[..brace_start];
        let inside = &tree[brace_start + 1..];
        let close = match inside.rfind('}') {
            Some(p) => p,
            None => return,
        };
        let inside = &inside[..close];
        let mut depth = 0i32;
        let mut start = 0usize;
        let bytes = inside.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                b',' if depth == 0 => {
                    let seg = inside[start..i].trim();
                    if !seg.is_empty() {
                        let combined = format!("{}{}", prefix, seg);
                        collect_use_tree_idents(&combined, out);
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }
        let tail = inside[start..].trim();
        if !tail.is_empty() {
            let combined = format!("{}{}", prefix, tail);
            collect_use_tree_idents(&combined, out);
        }
        return;
    }
    let path = tree.trim();
    if path.is_empty() || path == "*" || path.ends_with("::*") {
        return;
    }
    if let Some(as_pos) = path.rfind(" as ") {
        let alias = path[as_pos + 4..].trim().trim_end_matches(',').trim();
        if !alias.is_empty() && alias != "_" {
            out.insert(alias.to_string());
        }
        return;
    }
    let last = path.rsplit("::").next().unwrap_or(path).trim();
    if !last.is_empty() && last != "*" && last != "self" {
        out.insert(last.to_string());
    }
}

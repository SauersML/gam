use std::ffi::OsStr;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::{env, fs};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    fs::write(out_dir.join("lint_errors.rs"), "").expect("failed to write lint_errors.rs");
    // Do not emit wall-clock `cargo:rustc-env=GAM_BUILD_TIMESTAMP=<now>`.
    // Cargo records every build-script `rustc-env` in the crate fingerprint, so a
    // value that changes on every run makes the `gam` lib fingerprint dirty on
    // EVERY build — forcing a full recompile of gam + all downstream test crates
    // (~4min) regardless of whether any source changed, and defeating any shared
    // warm target. The variable was consumed nowhere in the tree, so it is removed
    // and must not be reintroduced.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=crates/gam-terms/src/analytic_penalties/manifest.rs");
    println!("cargo:rerun-if-changed=SPEC.md");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    emit_workspace_scanner_rerun_roots(&manifest_dir);

    // Everything below — the Python penalty-manifest generation and the
    // first-party hygiene scanners — is development/CI tooling that operates on
    // the full source tree and writes generated artifacts back into it (the
    // `gamfit/_penalties_manifest.py` manifest, the `ban_history.txt` ledger).
    // None of it applies when `gam` is built as a published dependency: the
    // `.crate` tarball ships only the Rust sources (the `include` list in
    // Cargo.toml omits `tests/`, `gamfit/`, etc.), there is no first-party tree
    // to lint, and writing into the source directory makes `cargo publish`'s
    // verification build fail with "source directory was modified by build.rs".
    // Detect the published/consumed context by the absence of the `tests/` tree
    // (excluded from the package) and return early so downstream builds — and
    // the publish verification itself — see a plain, side-effect-free build.
    if !manifest_dir.join("tests").is_dir() {
        return;
    }

    // Hard ban (always fatal): Claude may not edit build.rs alone. If the most
    // recent git author of build.rs is Claude, the change must be a co-authored
    // human collaboration. The checks in this file are not to be weakened or
    // removed under any circumstances; a human maintainer must approve and commit
    // any modification to this build script.
    forbid_claude_build_rs_edits(&manifest_dir);
    require_human_spec_edits(&manifest_dir);

    // HARD ban (always fatal): build.rs must not weaken, bypass, or
    // environment-gate its OWN hard-fail gates. Introspects this file's source
    // and aborts if the ban-scanner's terminal exit was removed/made conditional,
    // if any process exit is env-gated, or if temporary-bypass hack language was
    // reintroduced. Not to be weakened or removed.
    forbid_build_rs_self_tampering(&manifest_dir);

    // HARD ban (always fatal): the workspace lint level for `warnings` MUST be
    // `deny`. A demotion to `warn` (or anything else) lets pre-existing warnings
    // ride along and silently rots the tree. This is non-negotiable and cannot
    // be waived by a comment in Cargo.toml — assert it here and exit(1) otherwise.
    assert_warnings_are_denied(&manifest_dir);

    // #932 production derivative specializations are legal only as registered,
    // compiler-paired RowPrograms with mandatory numerical parity pins. This
    // gate is part of the ordinary root-library build, so `cargo check -p gam
    // --lib` rejects an unregistered RowKernel override or generated third/fourth
    // row atom before any test selection can omit the corresponding oracle.
    enforce_production_derivative_specializations(&manifest_dir);

    // HARD ban (always fatal, independent of the demoted aggregate scanner
    // below): no non-experiment tracked file may leak the absolute cluster
    // scratch path segment or the SLURM batch directive keyword. Run first and
    // exit(1) on any hit.
    scan_for_cluster_infra_leaks(&manifest_dir);

    emit_python_penalty_manifest(&manifest_dir)
        .expect("failed to emit Python analytic-penalty manifest");

    // Outright TO-DO ban (split below to avoid self-trigger in this file).
    let needle: &str = concat!("TO", "DO");
    let mut todo_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_banned_marker(&manifest_dir, &manifest_dir, needle, &mut todo_offenders);
    let mut todo_history_violations: Vec<(PathBuf, usize, String, String)> = Vec::new();
    run_todo_marker_history_audit(&manifest_dir, needle, &mut todo_history_violations);

    // Cosmetic wording dodge: a recent commit reworded or deleted an owed-work /
    // deferral note (one the prose ban forbids) from a non-test src file WHILE the
    // comment-stripped code stayed byte-identical. Wording laundered to satisfy
    // build.rs without doing the work — flagged as a hard failure below.
    let mut cosmetic_dodge_violations: Vec<(PathBuf, usize, String, String)> = Vec::new();
    run_cosmetic_wording_dodge_audit(&manifest_dir, &mut cosmetic_dodge_violations);

    // Deferred-work markers wearing a different label (relabel-evasion of the
    // bare TODO ban): a `fixme`/`xxx`/`hack` comment lead-in, or a deferral word
    // carrying an issue ref like `Follow-up (#932):`.
    let mut deferred_marker_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_deferred_work_markers(&manifest_dir, &manifest_dir, &mut deferred_marker_offenders);

    // Owed work disguised as prose ("not yet wired", "deferred to a follow-up",
    // "PLAN (not yet implemented)") — a TODO in a limitation costume. The work
    // must be FINISHED, never reworded or deleted.
    let mut owed_work_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_owed_work_prose(&manifest_dir, &manifest_dir, &mut owed_work_offenders);

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

    // Silent-corruption laundering (Pattern 1): a banned panic!/contract-guard
    // in non-test src deleted and replaced by a benign/NaN value at a site whose
    // own comment says the condition cannot happen. 22 of the last 60 commits.
    let mut silent_corruption_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_silent_corruption_laundering(
        &manifest_dir,
        &manifest_dir,
        &mut silent_corruption_offenders,
    );

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

    // Cargo.toml `[lints.*]` `allow` entries are the moral equivalent of
    // `#[allow(...)]` attributes (already banned crate-wide) — they
    // silence lints at the manifest level instead of fixing the code.
    // Whole-category allows (`clippy.all = "allow"`, `clippy.pedantic = {
    // level = "allow", ... }`) are the worst form: they preemptively
    // turn off every current and future lint in a category. Banned.
    let mut cargo_lint_allow_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_cargo_lint_allows(
        &manifest_dir,
        &manifest_dir,
        &mut cargo_lint_allow_offenders,
    );

    let gamfit_version =
        read_gamfit_project_version(&manifest_dir).expect("failed to read gamfit version");
    let mut gamfit_version_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_non_latest_gamfit_versions(
        &manifest_dir,
        &gamfit_version,
        &mut gamfit_version_offenders,
    );

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
    // delete the param. Bare `_: T` placeholders remain legal because they do
    // not preserve a fake binding name. Build.rs is exempt.
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

    // Sentinel-preserving fake-use ban. This catches the next dodge after
    // `_arg` and `let _ = arg`: a parameter is read in a guard or discard
    // statement, but every path still returns the same trivial sentinel
    // (`None`, `Ok(())`, `false`, empty collection, ...). That is not a real
    // use of the value; it is lint laundering. Use the parameter to compute
    // behavior, validate with a non-sentinel error, restructure the API, or
    // delete the parameter/function.
    let mut noop_sentinel_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_noop_sentinel_control_flow(&manifest_dir, &manifest_dir, &mut noop_sentinel_offenders);

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

    // Tracked-file size ban. Issue #780 keeps source/data artifacts reviewable
    // by requiring every tracked file to stay at or below 10k newline-counted
    // lines. This intentionally scans all Git-tracked files, not only the
    // text extensions handled by `visit_files`, so oversized CSVs or other
    // line-oriented assets cannot bypass the build gate.
    let mut oversized_file_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_oversized_tracked_files(&manifest_dir, &mut oversized_file_offenders);

    // Mechanical part_NNN / *_parts/ / split_parts/ naming is banned (slop). To
    // satisfy MAX_TRACKED_FILE_LINES, split large modules into cohesively-named
    // submodules, never numbered parts. Scans every tracked `.rs` file repo-wide
    // (src/, tests/, crates/, …); non-.rs dataset shards such as
    // bench/datasets/*_parts/part_00N.csv are skipped (the audit is `.rs`-only).
    let mut mechanical_part_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_mechanical_part_files(&manifest_dir, &mut mechanical_part_offenders);

    // Sticky "probation" for files that ever breached the 10k hard limit. Once a
    // tracked file exceeds MAX_TRACKED_FILE_LINES it is recorded in a committed
    // ledger and must then be brought below OVERSIZED_PROBATION_FLOOR_LINES (7k)
    // — not merely back under 10k. This closes the dodge of peeling a thin
    // tag-along satellite so the parent hovers at ~9,900: redemption requires a
    // real ~30% cut, i.e. an actual cohesive seam. Files that drop to <=7k are
    // redeemed and auto-pruned. The ledger is rewritten in place like
    // ban_history.txt / todo_history.txt.
    let mut probation_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_probation_oversized_files(&manifest_dir, &mut probation_offenders);

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
    // Second category emitted by the same scanner: `pub(crate)` /
    // `pub(super)` items in `src/` with ZERO consumers anywhere — not in
    // any other production file and not in any test. Rustc's `dead_code`
    // lint cannot see these in a library-shaped crate (it assumes such
    // items might be used by downstream crates, which is not what the
    // restricted visibility means). Inherent impl methods, free fns,
    // types, and constants all fall under this rule.
    let mut src_test_only_offenders: Vec<(PathBuf, usize, String, String)> = Vec::new();
    let mut src_unreferenced_pub_scoped: Vec<(PathBuf, usize, String, String)> = Vec::new();
    scan_for_src_items_used_only_by_tests(
        &manifest_dir,
        &mut src_test_only_offenders,
        &mut src_unreferenced_pub_scoped,
    );
    // Issue #202: the dead-pub scanner already detects receiver-method
    // call consumers (`extract_ident_tokens_into` extracts identifiers
    // after a `.` because `.` is treated as a non-ident byte that simply
    // advances the cursor before the alphabetic ident scan). The three
    // helpers (`effective_is_all_euclidean`, `effective_metric_weights`,
    // `ProjectedKktResidual::with_metadata`) all have production callers
    // visible to the tokenizer, so no per-item exception list is needed.
    // Leaving this comment as the audit trail; do not reintroduce a
    // bespoke `retain` filter — fix the scanner instead.

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

    if !deferred_marker_offenders.is_empty() {
        sections.push(Section {
            title:
                "deferred-work marker in disguise (a `fixme`/`xxx`/`hack`/`todo` comment lead-in, \
                 or a deferral word carrying an issue ref like `Follow-up (#932):`) — a relabelled \
                 marker is still owed work; implement it or delete the whole note, do not rename \
                 the marker"
                    .to_string(),
            rows: deferred_marker_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !owed_work_offenders.is_empty() {
        sections.push(Section {
            title:
                "owed work disguised as prose to dodge the marker ban (`not yet wired`, `deferred                  to a follow-up`, `PLAN (not yet implemented)`, `not wired into ...`) — this is a                  TODO in a limitation costume. FINISH the work: implement and wire it. Do NOT                  delete the comment, reword it into a 'documented limitation', or defer it again —                  a relabel is the same evasion the bare-marker ban forbids"
                    .to_string(),
            rows: owed_work_offenders
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

    if !silent_corruption_offenders.is_empty() {
        sections.push(Section {
            title: "silent-corruption laundering: a contract-guard was replaced by a benign/NaN value at a site documented as impossible — restore the hard guard (panic with // SAFETY, or a real error), do NOT emit a corrupting sentinel".to_string(),
            rows: silent_corruption_offenders
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

    if !cargo_lint_allow_offenders.is_empty() {
        sections.push(Section {
            title: "Cargo.toml [lints.*] `allow` entry (manifest-level lint silencing banned — fix the code instead of disabling the lint)".to_string(),
            rows: cargo_lint_allow_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !gamfit_version_offenders.is_empty() {
        sections.push(Section {
            title: format!(
                "non-latest gamfit version reference (expected {})",
                gamfit_version
            ),
            rows: gamfit_version_offenders
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

    if !noop_sentinel_offenders.is_empty() {
        sections.push(Section {
            title:
                "sentinel-preserving fake use (parameter is read only by a no-op guard/discard; use it for behavior, return a real error, or delete it)"
                    .to_string(),
            rows: noop_sentinel_offenders
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

    if !oversized_file_offenders.is_empty() {
        sections.push(Section {
            title: "tracked file over 10k lines (split the file; issue #780 line-count gate)"
                .to_string(),
            rows: oversized_file_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !probation_offenders.is_empty() {
        sections.push(Section {
            title:
                "oversized-file probation (this file once exceeded 10k lines; it must now reach \
                    under 7k lines, not merely back under 10k). Shaving to just under the limit — \
                    or peeling a thin tag-along satellite so the parent keeps hovering near 10k — \
                    is exactly the dodge this gate closes. Find a real cohesive seam and cut ~30%."
                    .to_string(),
            rows: probation_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !mechanical_part_offenders.is_empty() {
        sections.push(Section {
            title:
                "mechanical file-splitting banned (`part_<NNN>.rs` / `*_parts/` / `split_parts/` \
                    is line-count splitting, not logical decomposition — split by cohesive concern \
                    into descriptively-named modules; numbered parts are slop)"
                    .to_string(),
            rows: mechanical_part_offenders
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

    if !todo_history_violations.is_empty() {
        sections.push(Section {
            title: format!(
                "{} comment/marker removed without corresponding code change \
                 (history audit — todo_history.txt plus git HEAD)",
                needle
            ),
            rows: todo_history_violations
                .iter()
                .map(|(file, line, reason, site)| {
                    (file.clone(), *line, Some(reason.clone()), site.clone())
                })
                .collect(),
        });
    }

    if !cosmetic_dodge_violations.is_empty() {
        sections.push(Section {
            title:
                "cosmetic wording dodge (history audit): an owed-work/deferral note was reworded \
                 or deleted from a non-test src file while the comment-stripped code stayed \
                 byte-identical — the work was NOT done, only the wording. Do the real work or \
                 restore the honest note; rewording a build.rs-flagged signal to satisfy the \
                 scanner is banned"
                    .to_string(),
            rows: cosmetic_dodge_violations
                .iter()
                .map(|(file, line, reason, site)| {
                    (file.clone(), *line, Some(reason.clone()), site.clone())
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

    if !src_unreferenced_pub_scoped.is_empty() {
        sections.push(Section {
            title:
                "pub(crate)/pub(super) item in src/ with ZERO consumers anywhere (no production caller, no test reference — implement a real caller or delete the item; rustc's `dead_code` lint cannot see this in a library-shaped crate)"
                    .to_string(),
            rows: src_unreferenced_pub_scoped
                .iter()
                .map(|(r, l, hint, s)| (r.clone(), *l, Some(hint.clone()), s.clone()))
                .collect(),
        });
    }

    if sections.is_empty() {
        return;
    }

    render_report(&sections);
    let total_rows: usize = sections.iter().map(|s| s.rows.len()).sum();
    println!(
        "cargo:warning=ban-scanner FAILED: {} violation(s) across {} rule(s); build aborted \u{2014} every offending file:line is on an 'error:' line above",
        total_rows,
        sections.len()
    );
    std::process::exit(1);
}

/// HARD self-integrity gate (always fatal): build.rs must not weaken, bypass, or
/// environment-gate its own hard-fail gates. Reads this file's own source and
/// panics if it detects any of:
///   1. the ban-scanner's terminal hard exit removed or made conditional (e.g.
///      quietly turned into a warning, as happened once and rode `main` for days);
///   2. any `process::exit` gate guarded by an environment variable;
///   3. "temporary unblock"-style hack language anywhere in the file;
///   4. a hardcoded commit-SHA literal — build.rs operates on git history
///      generically and never names a specific commit in code, so a quoted
///      git-SHA-shaped literal is always an allowlist that exempts hand-picked
///      commits from a history audit (the gate-level form of wording laundering).
///
/// Every match needle is assembled from fragments at runtime, so this function
/// never matches its own source. Like the other gates here, it may not be
/// weakened or removed; a human maintainer must approve any change.
fn forbid_build_rs_self_tampering(manifest_dir: &Path) {
    let path = manifest_dir.join("build.rs");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("self-integrity gate: cannot read build.rs at {path:?}: {e}"));
    let lines: Vec<&str> = src.lines().collect();

    // Needles assembled from fragments so they never appear verbatim below.
    let render_call = format!("render_report(&{});", "sections");
    let exit_core_a = format!("std::process::{}(1)", "exit");
    let exit_core_b = format!("process::{}(1)", "exit");
    let exit_any = format!("process::{}(", "exit");
    let env_needles = [
        format!("env::{}(", "var"),
        format!("std::{}", "env"),
        format!("option_{}", "env!"),
    ];

    // ---- (1) the ban-scanner must end in an UNCONDITIONAL hard exit ----
    let render_sites: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.trim() == render_call)
        .map(|(i, _)| i)
        .collect();
    if render_sites.len() != 1 {
        panic!(
            "self-integrity gate: expected exactly one `{render_call}` report site, found {}. \
             The ban-scanner aggregator was moved, duplicated, or removed — restore the single \
             hard-failing report path.",
            render_sites.len()
        );
    }
    let r = render_sites[0];
    let render_indent = lines[r].len() - lines[r].trim_start().len();
    let branch_starts = [
        "if ", "} else", "else ", "else{", "match ", "for ", "while ", "loop ", "loop{",
    ];
    let mut hard_exit_found = false;
    for line in &lines[r + 1..] {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if (t.starts_with(exit_core_a.as_str()) || t.starts_with(exit_core_b.as_str()))
            && indent == render_indent
        {
            hard_exit_found = true;
            break;
        }
        if indent < render_indent {
            // The enclosing function/block closed before any hard exit was reached.
            break;
        }
        if t.starts_with("return") {
            panic!(
                "self-integrity gate: a `return` precedes the ban-scanner's hard exit. The \
                 report path must end in an unconditional process exit, not return early."
            );
        }
        if branch_starts.iter().any(|k| t.starts_with(*k)) {
            panic!(
                "self-integrity gate: the ban-scanner's hard exit was made CONDITIONAL by \
                 `{t}`. The exit must be unconditional — no branch, loop, or env switch may \
                 guard it."
            );
        }
    }
    if !hard_exit_found {
        panic!(
            "self-integrity gate: the ban-scanner no longer ends in an unconditional process \
             exit immediately after `{render_call}`. It appears to have been turned into a \
             warning or otherwise bypassed — restore the hard exit."
        );
    }

    // ---- (2) no process-exit gate may be guarded by an environment variable ----
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.starts_with("//") {
            continue;
        }
        let is_branch = t.starts_with("if ") || t.contains("else if ");
        if !is_branch || !env_needles.iter().any(|n| line.contains(n.as_str())) {
            continue;
        }
        let window_end = (i + 40).min(lines.len());
        for body in &lines[i + 1..window_end] {
            let bt = body.trim();
            if bt.starts_with("//") {
                continue;
            }
            if bt.contains(exit_any.as_str()) {
                panic!(
                    "self-integrity gate: a process exit is reachable under an \
                     environment-variable branch (`{t}`). Env-gating a hard-fail gate in \
                     build.rs is banned — gates must always fire."
                );
            }
        }
    }

    // ---- (3) no "temporary unblock"-style hack language ----
    let hack_phrases = [
        format!("{}{}", "velocity ", "unblock"),
        format!("{}{}", "temp ", "velocity"),
        format!("{}{}", "temporarily ", "non-fatal"),
        format!("{}{}", "temporarily ", "disabled"),
        format!("{}{}", "demoted from a ", "hard"),
        format!("{}{}", "do not ", "merge"),
    ];
    let lower = src.to_lowercase();
    for phrase in &hack_phrases {
        if lower.contains(phrase.as_str()) {
            panic!(
                "self-integrity gate: build.rs contains temporary-bypass hack language \
                 (`{phrase}`). The gates here are not to be temporarily weakened; fix the \
                 underlying violations instead of weakening a gate."
            );
        }
    }

    // ---- (4) no hardcoded commit-SHA literal (allowlist suppression) ----
    // build.rs reads git history dynamically and has NO legitimate reason to name
    // a specific commit in code. A quoted git-SHA-shaped literal is therefore an
    // exemption/allowlist that skips a history audit for hand-picked commits — the
    // gate-level analogue of the comment laundering those audits exist to catch
    // (e.g. a `HUMAN_REVIEWED_DODGE_EXEMPT` list added to the cosmetic-dodge audit
    // to make a laundering commit's violations vanish). This arm is content-based,
    // so it fires on every build regardless of who authored the commit — closing
    // the door a maintainer could otherwise be socially-engineered into opening.
    //
    // A SHA discussed in a COMMENT is unquoted prose, so it is never matched; only
    // quoted string literals are inspected. "SHA-shaped" = a 7..=40 char run of
    // lowercase hex carrying BOTH a digit and an a-f letter, which excludes plain
    // decimal version strings and ordinary English words while catching every
    // realistic abbreviated or full git object name.
    for (line_no, raw) in lines.iter().enumerate() {
        // Odd-position fragments of a `"`-split line are the quoted-string
        // contents. Escapes inside a SHA literal do not occur, so this is exact
        // for the construct being banned.
        for literal in raw.split('"').skip(1).step_by(2) {
            let len = literal.len();
            let all_lower_hex = literal
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
            let has_digit = literal.bytes().any(|b| b.is_ascii_digit());
            let has_hex_letter = literal.bytes().any(|b| (b'a'..=b'f').contains(&b));
            if (7..=40).contains(&len) && all_lower_hex && has_digit && has_hex_letter {
                panic!(
                    "self-integrity gate: build.rs contains a hardcoded commit-SHA literal \
                     (`{literal}` at line {}). build.rs must operate on git history \
                     generically; pinning a commit is an allowlist that exempts hand-picked \
                     commits from a history audit and is banned. Remove the exemption and do \
                     the real work the audit demands instead of suppressing it.",
                    line_no + 1
                );
            }
        }
    }
}

/// Check the git history of `build.rs` and panic if the most recent commit was
/// made by Claude without a human co-author. Claude is allowed to propose
/// changes to this file only in collaboration with a human; the final commit must
/// carry a human author name/email. The checks in this file are not to be weakened
/// or removed under any circumstances, and any edit here requires human review
/// and sign-off.
///
/// The heuristic is simple and practical: the most recent commit author name or
/// email must not be Claude/Anthropic. A human co-author should be the commit
/// author; `Co-Authored-By` alone is intentionally not enough because git log
/// `%an` returns the primary author of the commit.
fn forbid_claude_build_rs_edits(manifest_dir: &Path) {
    // A shallow or partial clone (e.g. a CI / wheel-build fetch made with
    // `--depth`) does not contain the commit that actually last modified
    // build.rs, so the `git log -- build.rs` audit below cannot see it and git
    // instead reports the shallow-boundary tip commit's author — a false
    // positive that blocks the build whenever that tip happens to be a
    // Claude-authored commit to some *other* file. Authoring of build.rs never
    // happens inside a shallow, checkout-only build clone, so the guard is
    // inapplicable there and is skipped. Full working clones — the only place a
    // build.rs commit is actually authored — still enforce it below.
    // `-c safe.directory=*` keeps the audit RUNNING inside containerized
    // builds (maturin's manylinux images bind-mount the source tree with a
    // different owner, and git otherwise refuses with "dubious ownership" —
    // the same failure mode the tracked-file audits hit before they moved to
    // parsing `.git/index` directly). Environmental failures below — no git
    // binary, no repository, an unreadable one — make the audit INAPPLICABLE
    // (like the shallow-clone case): they cannot be an attempt to author
    // build.rs, so they skip rather than fail the build. Genuine non-human
    // authorship remains a hard build failure.
    let shallow = Command::new("git")
        .arg("-c")
        .arg("safe.directory=*")
        .arg("-C")
        .arg(manifest_dir)
        .arg("rev-parse")
        .arg("--is-shallow-repository")
        .output();
    if let Ok(shallow) = shallow {
        if shallow.status.success() && String::from_utf8_lossy(&shallow.stdout).trim() == "true" {
            return;
        }
    }

    let output = match Command::new("git")
        .arg("-c")
        .arg("safe.directory=*")
        .arg("-C")
        .arg(manifest_dir)
        .arg("log")
        .arg("-1")
        .arg("--format=%an|%ae")
        .arg("--")
        .arg("build.rs")
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            println!("cargo:warning=build.rs author audit skipped: git could not be run ({err})");
            return;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("dubious ownership")
            || stderr.contains("not a git repository")
            || stderr.contains("No such file or directory")
        {
            println!(
                "cargo:warning=build.rs author audit skipped: repository not auditable \
                 in this environment ({})",
                stderr.lines().next().unwrap_or("unknown git error")
            );
            return;
        }
        panic!(
            "failed to query git history for build.rs author: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let info = String::from_utf8_lossy(&output.stdout);
    let info = info.trim();
    let is_claude =
        info.to_lowercase().contains("claude") || info.to_lowercase().contains("anthropic");

    if is_claude {
        panic!(
            "Sorry Claude! build.rs was last edited by a non-human author ({info}). \
             Claude may propose changes to build.rs, but only a human may author the \
             commit. The checks in this file are not to be weakened or removed under \
             any circumstances. Please have a human maintainer re-commit any needed \
             change to build.rs.",
        );
    }
}

/// `SPEC.md` holds the project's engineering rules and may only be changed by a
/// human. The build FAILS when **the commit being built edits `SPEC.md`** and
/// its message does not contain the literal phrase `I am a human`. Agents
/// (Claude included) must not edit `SPEC.md` and must not write that phrase;
/// this gate exists precisely so silent agent edits to the rules cannot
/// compile. The check is scoped to the HEAD commit (compared against its first
/// parent) so it flags exactly the offending commit and does not retroactively
/// break unrelated builds. Like the build.rs author gate above, this check must
/// not be weakened or removed under any circumstances. Skipped only when
/// `SPEC.md`, git history, or HEAD's parent is absent (e.g. a published-crate
/// consumer or a shallow checkout with no parent to diff against).
fn require_human_spec_edits(manifest_dir: &Path) {
    if !manifest_dir.join("SPEC.md").is_file() {
        return;
    }
    // Which files did the HEAD commit change (vs its first parent)? If there is
    // no parent (root/shallow clone), git errors and we cannot tell what
    // changed, so we do not block the build.
    let changed = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("diff")
        .arg("--name-only")
        .arg("HEAD~1")
        .arg("HEAD")
        .output();
    let changed = match changed {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let touches_spec = String::from_utf8_lossy(&changed.stdout)
        .lines()
        .any(|line| line.trim() == "SPEC.md");
    if !touches_spec {
        return;
    }

    let msg = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("log")
        .arg("-1")
        .arg("--format=%B")
        .arg("HEAD")
        .output();
    let msg = match msg {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let message = String::from_utf8_lossy(&msg.stdout);

    if !message.contains("I am a human") {
        panic!(
            "SPEC.md gate: the HEAD commit edits SPEC.md but its message does not contain \
             the phrase \"I am a human\".\n\
             SPEC.md holds the project's engineering rules and may only be edited by a human \
             who affirms authorship by writing \"I am a human\" in the commit message. Agents \
             (Claude included) must not modify SPEC.md and must not write that phrase. This \
             check must not be weakened or removed under any circumstances.\n\
             HEAD commit message:\n{}",
            message.trim()
        );
    }
}

/// Assert that the workspace-root `Cargo.toml` pins `[lints.rust] warnings`
/// to `deny`. Any other level (`warn`, `allow`, missing) lets warnings
/// accumulate; that is a build failure, not a soft signal. Comments in
/// Cargo.toml cannot waive this — the policy lives here.
fn assert_warnings_are_denied(manifest_dir: &Path) {
    let cargo_toml = manifest_dir.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", cargo_toml.display()));

    // Find the `[lints.rust]` section and read its `warnings` value.
    let mut in_lints_rust = false;
    let mut found_level: Option<String> = None;
    for raw in content.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_lints_rust = line == "[lints.rust]";
            continue;
        }
        if !in_lints_rust {
            continue;
        }
        let code = line.split('#').next().unwrap_or("").trim();
        if let Some(rest) = code.strip_prefix("warnings") {
            let rest = rest.trim_start();
            if let Some(value) = rest.strip_prefix('=') {
                // Accept both `warnings = "deny"` and the table form
                // `warnings = { level = "deny", ... }`.
                let value = value.trim();
                let level = if let Some(idx) = value.find("level") {
                    value[idx..]
                        .split('=')
                        .nth(1)
                        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                } else {
                    Some(value.trim_matches('"').trim_matches('\'').to_string())
                };
                found_level = level;
                break;
            }
        }
    }

    match found_level.as_deref() {
        Some("deny") => {}
        Some(other) => panic!(
            "[lints.rust] warnings MUST be \"deny\", found \"{other}\" in {}. \
             Restore `warnings = \"deny\"`; warnings are not permitted to accumulate.",
            cargo_toml.display()
        ),
        None => panic!(
            "[lints.rust] warnings = \"deny\" is missing from {}. \
             It MUST be present and set to \"deny\".",
            cargo_toml.display()
        ),
    }
}

#[derive(Clone)]
struct PenaltyWrapperManifest {
    kind_tag: String,
    rust_type: String,
    python_wrapper: String,
    row_block_diagonal: bool,
}

fn emit_python_penalty_manifest(manifest_dir: &Path) -> std::io::Result<()> {
    let penalties_dir = manifest_dir
        .join("crates")
        .join("gam-terms")
        .join("src")
        .join("analytic_penalties");
    let registry = fs::read_to_string(penalties_dir.join("manifest.rs"))?;
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
        let source = penalty_manifest_source_for_type(&registry, rust_type)?;
        wrappers.push(PenaltyWrapperManifest {
            kind_tag: manifest_const_string(&source, "KIND_TAG")?,
            rust_type: format!("{variant}:{rust_type}"),
            python_wrapper: manifest_const_string(&source, "PYTHON_WRAPPER")?,
            row_block_diagonal: manifest_const_bool(&source, "ROW_BLOCK_DIAGONAL")?,
        });
    }
    let mut output = String::from(
        "# Generated by build.rs from crates/gam-terms/src/analytic_penalties/manifest.rs.\n\
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
    // Write ONLY when the content actually changed. This file is declared as a
    // `cargo:rerun-if-changed` input, so unconditionally rewriting it on every
    // build advances its mtime and makes cargo mark `gam` Dirty on the NEXT
    // build ("the file `gamfit/_penalties_manifest.py` has changed") — forcing a
    // full recompile even with no source change. The content guard breaks that
    // self-invalidation loop (mirrors the ban-history ledger writer).
    let manifest_path = manifest_dir.join("gamfit").join("_penalties_manifest.py");
    if let Ok(existing) = fs::read_to_string(&manifest_path) {
        if existing == output {
            return Ok(());
        }
    }
    fs::write(manifest_path, output)
}

fn penalty_manifest_source_for_type(registry: &str, rust_type: &str) -> std::io::Result<String> {
    let marker = format!("impl PenaltyManifest for {rust_type} {{");
    if let Some(start) = registry.find(&marker) {
        let rest = &registry[start..];
        if let Some(end) = rest.find("\n}") {
            return Ok(rest[..end + 2].to_string());
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

fn read_gamfit_project_version(manifest_dir: &Path) -> std::io::Result<String> {
    let path = manifest_dir.join("pyproject.toml");
    let content = fs::read_to_string(&path)?;
    read_toml_version_line(&content).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "pyproject.toml is missing a top-level version",
        )
    })
}

fn scan_for_non_latest_gamfit_versions(
    root: &Path,
    latest: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    require_toml_version(
        root,
        Path::new("crates/gam-pyffi/Cargo.toml"),
        latest,
        offenders,
    );
    require_uv_lock_gamfit_version(root, latest, offenders);

    visit_files(root, root, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        scan_gamfit_version_content(rel, content, latest, offenders);
    });
}

fn require_toml_version(
    root: &Path,
    rel: &Path,
    latest: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    let path = root.join(rel);
    let content = match fs::read_to_string(&path) {
        Ok(content) => content,
        Err(_) => {
            offenders.push((
                rel.to_path_buf(),
                1,
                format!("missing version file; expected gamfit {latest}"),
            ));
            return;
        }
    };
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with("version = ") {
            continue;
        }
        match read_quoted_value_after_prefix(trimmed, "version = ") {
            Some(version) if version == latest => return,
            Some(_) | None => {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
                return;
            }
        }
    }
    offenders.push((
        rel.to_path_buf(),
        1,
        format!("missing version line; expected gamfit {latest}"),
    ));
}

fn require_uv_lock_gamfit_version(
    root: &Path,
    latest: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    let rel = Path::new("uv.lock");
    let content = match fs::read_to_string(root.join(rel)) {
        Ok(content) => content,
        Err(_) => {
            offenders.push((
                rel.to_path_buf(),
                1,
                format!("missing uv.lock gamfit package; expected {latest}"),
            ));
            return;
        }
    };

    let mut inside_package = false;
    let mut inside_gamfit = false;
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed == "[[package]]" {
            inside_package = true;
            inside_gamfit = false;
            continue;
        }
        if !inside_package {
            continue;
        }
        if trimmed == "name = \"gamfit\"" {
            inside_gamfit = true;
            continue;
        }
        if inside_gamfit && trimmed.starts_with("version = ") {
            match read_quoted_value_after_prefix(trimmed, "version = ") {
                Some(version) if version == latest => return,
                Some(_) | None => {
                    offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
                    return;
                }
            }
        }
    }

    offenders.push((
        rel.to_path_buf(),
        1,
        format!("missing gamfit package version; expected {latest}"),
    ));
}

fn read_toml_version_line(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(version) = read_quoted_value_after_prefix(trimmed, "version = ") {
            return Some(version.to_string());
        }
    }
    None
}

fn read_quoted_value_after_prefix<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(prefix)?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn gamfit_versions_in_line(line: &str) -> Vec<String> {
    let mut versions = Vec::new();
    let mut search_start = 0usize;
    while let Some(offset) = line[search_start..].find("0.1.") {
        let start = search_start + offset;
        let mut end = start + "0.1.".len();
        while end < line.len() && line.as_bytes()[end].is_ascii_digit() {
            end += 1;
        }
        if end > start + "0.1.".len() {
            versions.push(line[start..end].to_string());
        }
        search_start = end;
    }
    versions
}

fn scan_gamfit_version_content(
    rel: &Path,
    content: &str,
    latest: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    for (line_idx, line) in content.lines().enumerate() {
        let lower = line.to_ascii_lowercase();
        if !lower.contains("gamfit") || !line.contains("0.1.") {
            continue;
        }
        for version in gamfit_versions_in_line(line) {
            if version != latest {
                offenders.push((rel.to_path_buf(), line_idx + 1, line.to_string()));
            }
        }
    }
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
            "error — {}  ({} hit{})",
            section.title,
            section.rows.len(),
            if section.rows.len() == 1 { "" } else { "s" },
        );
        for (rel, line_no, tag, line) in &section.rows {
            let trimmed = line.trim();
            let snippet: String = trimmed.chars().take(160).collect();
            match tag {
                Some(t) => eprintln!(
                    "  error: {}:{}: [{}] {}",
                    rel.display(),
                    line_no,
                    t,
                    snippet
                ),
                None => eprintln!("  error: {}:{}: {}", rel.display(), line_no, snippet),
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
        // `debug_assert!` family — checks that compile to nothing in
        // release. An invariant worth checking is worth checking in
        // release; one that isn't should be deleted. Banned everywhere
        // (tests included — a test asserting only in debug mode silently
        // passes in release).
        ("debug_assert!(", "debug_assert!", false),
        ("debug_assert_eq!(", "debug_assert_eq!", false),
        ("debug_assert_ne!(", "debug_assert_ne!", false),
        // `hint::black_box(name)` / `std::hint::black_box(name)` — the
        // second-round dodge for silencing an unused-value warning after
        // `let _ = name;` got banned. Benches that need it to prevent
        // optimization across iterations live under `benches/` and are
        // exempt via the test mask.
        ("hint::black_box(", "hint::black_box", true),
        ("std::hint::black_box(", "std::hint::black_box", true),
        ("core::hint::black_box(", "core::hint::black_box", true),
        // `cfg!(debug_assertions)` / `cfg!(test)` — runtime branches whose
        // behavior diverges between debug↔release or test↔non-test. Same
        // pathology as `debug_assert!`: code that only runs in one build
        // configuration silently changes meaning in the other. If the
        // check matters it should be unconditional; if it doesn't it
        // should be deleted. Test-aware (the test build legitimately
        // queries its own configuration).
        ("cfg!(debug_assertions)", "cfg!(debug_assertions)", true),
        ("cfg(debug_assertions)", "cfg(debug_assertions)", true),
        ("cfg!(test)", "cfg!(test)", true),
        // `Arc::strong_count` / `Arc::weak_count` (and the `Rc` siblings)
        // are documented racy primitives — the value can change between
        // observation and use. They're also a common source of
        // tautological assertions (`strong_count > 0` is always true
        // when you hold the Arc you're counting). Tests sometimes need
        // them for refcount-leak detection.
        ("Arc::strong_count(", "Arc::strong_count", true),
        ("Arc::weak_count(", "Arc::weak_count", true),
        ("Rc::strong_count(", "Rc::strong_count", true),
        ("Rc::weak_count(", "Rc::weak_count", true),
        // `ends_with(".rs")` anywhere in the codebase.
        // It's often a tautological assertion (e.g. `file!().ends_with(".rs")`)
        // used to satisfy `scan_for_useless_tests` without
        // actually asserting anything about the unit under test. Strict
        // everywhere — tests must verify a real property.
        ("ends_with(\".rs\")", "ends_with(\".rs\")", false),
        ("ends_with(\".rs\"", "ends_with(\".rs\"", false),
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
/// test scope if either the file is a test/bench/example file (under
/// `tests/`, `bench/`, `benches/`, `examples/`, or
/// `crates/*/tests|benches|examples/`),
/// or the line is inside a brace block annotated with `#[test]`,
/// `#[cfg(test)]`, `#[cfg(all(test, ...))]`, or `#[cfg(any(test, ...))]`.
///
/// `examples/` is included because Cargo example binaries are user-facing
/// demonstration entry points: their `fn main` reports results to stdout
/// and exits with a status code by contract, exactly like a CLI under
/// `tests/` or `bench/`. Several banned-substring rules (`println!`,
/// `process::exit`, `Box::leak`, `hint::black_box`) explicitly document
/// the test/example/bench scope as the legitimate use site for those
/// primitives; the mask is the place that decision is enforced.
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
        || rel_str.starts_with("examples/")
        || path_matches_crates_test_or_example(&rel_str)
        || file_stem_is_exempt_test_module(rel);
    if file_is_test {
        mask.fill(true);
        return mask;
    }

    // File-level `#![cfg(test)]` inner attribute (the Rust-idiomatic way to
    // mark a whole module as test-only when it lives in its own file). When
    // present, the entire file is test scope — the same way an outer
    // `#[cfg(test)] mod foo { ... }` would gate the inlined module body.
    // The brace-tracking loop below cannot model this on its own because
    // an inner attribute has no following `{` to open a gate.
    let stripped_all = strip_file_lines(content);
    for line in &stripped_all {
        if is_cfg_test_inner_attr_line(line) {
            mask.fill(true);
            return mask;
        }
    }

    // Brace-tracked test regions. We maintain a stack of entry brace
    // depths: when a `#[test]` or `#[cfg(test)]`-style attribute is seen,
    // we wait for the next `{` to open the gated block, then pop when depth
    // returns to the entry level.
    let mut depth: i32 = 0;
    // Pending attribute: when Some, the very next `{` (which may be on the
    // same line as the attribute or several lines later) opens a gated
    // block.
    let mut pending_test_scope_attr = false;
    // Stack of entry depths (depth at which the gate opens; we exit when
    // depth drops back to this value).
    let mut gate_stack: Vec<i32> = Vec::new();

    for (idx, _raw) in lines.iter().enumerate() {
        let stripped = stripped_all.get(idx).cloned().unwrap_or_default();

        // Detect a test-scope attribute on this line. Attribute syntax is
        // `#[test]`, `#[cfg(test)]`, `#[cfg(all(test, ...))]`, and
        // `#[cfg(any(test, ...))]`. We accept the attribute anywhere on the
        // line.
        if is_test_attr_line(&stripped) || is_cfg_test_attr_line(&stripped) {
            pending_test_scope_attr = true;
        }

        // Walk braces on this line.
        let bytes = stripped.as_bytes();
        // The line counts as "in test" if the line's starting depth is
        // already inside a gate or if this line is the attribute target
        // that opens one. The latter matters for compact tests like
        // `#[test] fn f() { println!("..."); }`.
        let inside_at_line_start = !gate_stack.is_empty();
        mask[idx] = inside_at_line_start || pending_test_scope_attr;

        for &b in bytes {
            if b == b'{' {
                depth += 1;
                if pending_test_scope_attr {
                    // The brace that just opened belongs to the cfg(test)
                    // attribute target. The gate's "entry depth" is the
                    // outer depth, i.e. depth - 1: we exit when depth
                    // drops back to that value.
                    gate_stack.push(depth - 1);
                    pending_test_scope_attr = false;
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

/// True when the file's stem matches the same naming pattern that
/// `is_exempt_test_submodule_name` accepts for `#[cfg(test)] mod ...`
/// blocks: `tests`, `test_support`, `tests_*`, or `*_tests`. When a
/// mechanically-split `mod foo_tests { ... }` inlined via `include!` is
/// extracted into its own file (the cohesive-module decomposition the
/// part-file ban demands), the module body lands in a file whose stem
/// equals the module name. The whole file is then test scope by the
/// same rule that exempted the inline `mod`.
fn file_stem_is_exempt_test_module(rel: &Path) -> bool {
    let Some(stem) = rel.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    is_exempt_test_submodule_name(stem)
}

/// Recognize `#![cfg(test)]` (inner attribute, applies to the enclosing
/// item — at file top level, the whole module). Mirrors
/// `is_cfg_test_attr_line` but requires the `#![` opener so the outer
/// `#[cfg(test)] item` form (which gates only the next item) is not
/// confused with the inner form.
fn is_cfg_test_inner_attr_line(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let mut i = 0usize;
    while i + 2 < bytes.len() {
        if bytes[i] == b'#' && bytes[i + 1] == b'!' && bytes[i + 2] == b'[' {
            let rest = &stripped[i + 3..];
            if let Some(pos) = rest.find("cfg(") {
                let abs = i + 3 + pos;
                let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
                if before_ok {
                    let args_start = abs + 4;
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

/// Recognize `#[test]` on a stripped line. This is intentionally exact:
/// attribute macros such as `#[test_case]` are not Rust's built-in test
/// harness marker.
fn is_test_attr_line(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let mut i = 0usize;
    while i + 6 < bytes.len() {
        if bytes[i] == b'#' && bytes[i + 1] == b'[' {
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j + 4 <= bytes.len() && &stripped[j..j + 4] == "test" {
                let mut k = j + 4;
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                if k < bytes.len() && bytes[k] == b']' {
                    return true;
                }
            }
        }
        i += 1;
    }
    false
}

/// Match `crates/<name>/tests/...`, `crates/<name>/benches/...`, or
/// `crates/<name>/examples/...`.
fn path_matches_crates_test_or_example(rel: &str) -> bool {
    let Some(rest) = rel.strip_prefix("crates/") else {
        return false;
    };
    // Skip the crate name segment.
    let Some(slash) = rest.find('/') else {
        return false;
    };
    let tail = &rest[slash + 1..];
    tail.starts_with("tests/") || tail.starts_with("benches/") || tail.starts_with("examples/")
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
            // Scan up to 10 preceding non-blank lines for a `SAFETY:` marker.
            // The prior cap was 3, which forced multi-line SAFETY explanations
            // onto a single line of context. Stopping at the first blank line
            // keeps the marker tied to the same logical block as the site.
            let mut justified = false;
            let mut seen = 0usize;
            let mut k = idx;
            while k > 0 && seen < 10 {
                k -= 1;
                let prev = lines[k];
                if prev.trim().is_empty() {
                    break;
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
            // Scan up to 10 preceding non-blank lines for a `SAFETY:` marker.
            // The prior cap was 3, which forced multi-line SAFETY explanations
            // onto a single line of context. Stopping at the first blank line
            // keeps the marker tied to the same logical block as the site.
            let mut justified = false;
            let mut seen = 0usize;
            let mut k = idx;
            while k > 0 && seen < 10 {
                k -= 1;
                let prev = lines[k];
                if prev.trim().is_empty() {
                    break;
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
/// comment within the same line or one of the preceding non-blank lines of
/// the contiguous `//`-comment block immediately above the `panic!(` site.
/// Mirrors `scan_for_unsafe_without_safety`. The `panic!` family in the
/// history audit still treats any panic-shape in a former-marker body as
/// trivial, regardless of SAFETY comments — this softening applies only
/// to the lexical scanner.
/// Bans "silent-corruption laundering": a banned `panic!`/contract-guard in
/// non-test src deleted and replaced by a benign or NaN value at a site whose
/// own comment documents the condition as impossible. This is the
/// most-recurring evasion (Pattern 1): the hard guard that would have caught a
/// real invariant violation is swapped for `f64::NAN` / `0.0` / `Ok(())` /
/// `None` / `Default::default()` / `Array*::zeros` / an all-zero tuple, so the
/// corruption flows downstream silently instead of aborting.
///
/// Two precise, low-false-positive rules, both non-test src/ only:
///   (A) `f64::NAN` (incl. `core::`/`std::` paths) EMITTED AS A VALUE — a
///       match-arm body `=> f64::NAN`, a `return f64::NAN`, or a `f64::NAN`
///       appearing inside a tuple/array literal `(f64::NAN, ...)` / trailing
///       `f64::NAN,`. Comparisons (`.is_nan()`, `!= f64::NAN`) and a NaN used
///       only as a provably-overwritten initial sentinel are NOT matched —
///       only the value-emission forms above.
///   (B) a benign-literal-only body (`0.0` / `Ok(())` / `None` /
///       `Default::default()` / `Array*::zeros` / an all-zero tuple) appearing
///       within ~6 lines AFTER a comment carrying a contract/impossibility
///       token. "Comment says impossible" + "body returns a benign default" is
///       the laundering signature.
///
/// All match tokens are assembled from string fragments so build.rs never
/// self-trips on its own description.
fn scan_for_silent_corruption_laundering(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    // Tokens assembled from fragments so this scanner never flags its own
    // source (build.rs is skipped anyway, but keep the invariant explicit).
    let nan_lit = format!("f64::{}", "NAN");
    let nan_core = format!("core::f64::{}", "NAN");
    let nan_std = format!("std::f64::{}", "NAN");
    let is_nan = format!(".is_{}()", "nan");

    // Contract / impossibility comment tokens, lowercased for matching.
    // NOTE: the bare token "safety" is deliberately EXCLUDED. Every `unsafe`
    // block carries a `// SAFETY:` justification, and a normal `Ok(())` /
    // `f64::NAN` success/return a few lines below that justification is NOT
    // laundering. We require an explicit *impossibility* claim — the strong
    // tokens below — which is the actual laundering signature (e.g. bessel's
    // "silently corrupt", arrow_schur's "unreachable"). Tokens are assembled
    // from fragments so this scanner never self-trips.
    // Only explicit IMPOSSIBILITY-ASSERTION phrases — a comment that documents
    // the path as one that cannot legitimately execute. Broad doc words like
    // "safety", "contract", and "invariant" are deliberately excluded: they
    // appear in routine math/`unsafe` commentary next to perfectly normal
    // returns and would flood the queue with false positives. The phrases below
    // are the ones whose adjacency to a benign/NaN body IS the laundering tell
    // (bessel's "silently corrupt", arrow_schur's "unreachable", evidence's
    // "impossible"). All assembled from fragments so this scanner never
    // self-trips on its own source.
    let contract_tokens: Vec<String> = vec![
        format!("{}", "unreachable"),
        format!("{}", "impossible"),
        format!("{} {}", "silently", "corrupt"),
        format!("{} {}", "programming", "error"),
        format!("{} {}", "must", "override"),
        format!("{} {}", "cannot", "happen"),
        format!("{} {}", "can not", "happen"),
        format!("{} {}", "should", "never"),
        format!("{} {}", "shall", "never"),
        format!("{} {}", "never", "reached"),
        format!("{} {}", "never", "happen"),
        format!("{} {}", "never", "occur"),
    ];

    // Benign-default body fragments (rule B).
    let benign_okunit = format!("{}(())", "Ok");
    let benign_none = "None".to_string();
    let benign_default = format!("{}::default()", "Default");

    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        let has_nan = content.contains(&nan_lit);
        let lower_all = content.to_lowercase();
        let has_contract = contract_tokens.iter().any(|t| lower_all.contains(t));
        if !has_nan && !has_contract {
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
            let strimmed = stripped.trim();

            // The laundering SIGNATURE is "comment says this cannot happen" +
            // "body emits a corrupting sentinel". Both rules below require a
            // contract/impossibility comment within ~6 preceding lines — a bare
            // `f64::NAN`/`0.0` return on its own is a legitimate numeric
            // convention (out-of-domain, diagnostic display); it only becomes
            // laundering when the adjacent comment documents the path as
            // impossible. Scan the RAW preceding lines (comments intact,
            // lowercased), requiring an actual `//` comment line.
            let mut found_contract_comment = false;
            let lo = idx.saturating_sub(6);
            for raw in lines.iter().take(idx).skip(lo) {
                if !raw.contains("//") {
                    continue;
                }
                let raw_lower = raw.to_lowercase();
                if contract_tokens.iter().any(|t| raw_lower.contains(t)) {
                    found_contract_comment = true;
                    break;
                }
            }
            if !found_contract_comment {
                continue;
            }

            // Normalize the body: drop a single leading `return ` / `=> `
            // (value-emission contexts) and a trailing `;` or `,`.
            let mut emit_ctx = false;
            let mut body = strimmed.to_string();
            if let Some(rest) = body.strip_prefix("return ") {
                emit_ctx = true;
                body = rest.to_string();
            }
            if let Some(rest) = body.strip_prefix("=> ") {
                emit_ctx = true;
                body = rest.to_string();
            }
            let body = body.trim().trim_end_matches([';', ',']).trim();

            // ---- Rule (A): NaN emitted as a value at an impossible site ----
            // Only the value-emission forms — NOT a call/constructor argument
            // (`unwrap_or(NAN)`, `from_elem(.., NAN)`, `set_item("k", NAN)`,
            // `fill(NAN)`): a literal tuple/array/return body is the WHOLE
            // expression of the (de-`return`-ed) line, never nested in `foo(`.
            let is_comparison = strimmed.contains(&is_nan)
                || strimmed.contains("!= ")
                || strimmed.contains("== ")
                || strimmed.contains(".partial_cmp")
                || strimmed.contains("matches!");
            let bare_nan = body == nan_lit || body == nan_core || body == nan_std;
            let is_nan_aggregate = {
                let open = body.starts_with('(') || body.starts_with('[');
                let closed = body.ends_with(')') || body.ends_with(']');
                if open && closed && body.len() >= 2 {
                    let inner = &body[1..body.len() - 1];
                    let inner_main = inner.split(';').next().unwrap_or(inner);
                    !inner_main.trim().is_empty()
                        && inner_main.split(',').all(|p| {
                            let p = p.trim();
                            p == nan_lit
                                || p == nan_core
                                || p == nan_std
                                || p == "0.0"
                                || p == "None"
                        })
                        && (inner_main.contains(&nan_lit)
                            || inner_main.contains(&nan_core)
                            || inner_main.contains(&nan_std))
                } else {
                    false
                }
            };
            let _ = emit_ctx;
            if (bare_nan || is_nan_aggregate) && !is_comparison {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
                continue;
            }

            // ---- Rule (B): benign default body at an impossible site ----
            let is_zero_float = body == "0.0" || body == "0.0_f64" || body == "0f64";
            let is_okunit = body == benign_okunit;
            let is_none = body == benign_none;
            let is_default = body == benign_default;
            let is_zeros = (body.starts_with("Array") && body.contains("::zeros"))
                || body.ends_with("::zeros()");
            let is_zero_tuple = body.starts_with('(') && body.ends_with(')') && {
                let inner = &body[1..body.len() - 1];
                !inner.is_empty()
                    && inner.split(',').all(|p| {
                        let p = p.trim();
                        p == "0.0" || p == "0.0_f64" || p == "0f64"
                    })
            };
            let body_is_benign =
                is_zero_float || is_okunit || is_none || is_default || is_zeros || is_zero_tuple;
            if body_is_benign {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

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
            // Scan up to 10 preceding non-blank lines for a `SAFETY:` marker.
            // The prior cap was 3, which forced multi-line SAFETY explanations
            // onto a single line of context. Stopping at the first blank line
            // keeps the marker tied to the same logical block as the site.
            let mut justified = false;
            let mut seen = 0usize;
            let mut k = idx;
            while k > 0 && seen < 10 {
                k -= 1;
                let prev = lines[k];
                if prev.trim().is_empty() {
                    break;
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
            // Both `#[cfg(any())]` and the crate-/module-level `#![cfg(any())]`
            // inner form are dead-by-construction gates; match either.
            if !line_has_attr_marker(&stripped) {
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
            if !line_has_attr_marker(stripped) {
                continue;
            }
            if !(stripped.contains("cfg(") || stripped.contains("cfg_attr(")) {
                continue;
            }
            if line_has_feature_predicate(stripped) && !line_is_cuda_feature_gate(stripped) {
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

fn line_is_cuda_feature_gate(stripped: &str) -> bool {
    let mut rest = stripped;
    let mut saw_cuda = false;
    while let Some(pos) = rest.find("feature") {
        let after = &rest[pos + "feature".len()..];
        let Some(eq_pos) = after.find('=') else {
            return false;
        };
        let after_eq = after[eq_pos + 1..].trim_start();
        if !after_eq.starts_with("\"cuda\"") {
            return false;
        }
        saw_cuda = true;
        rest = &after_eq["\"cuda\"".len()..];
    }
    saw_cuda
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
        let basename = rel.file_name().and_then(OsStr::to_str).unwrap_or("");
        if basename != "Cargo.toml" {
            return;
        }
        let mut in_features = false;
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
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
            // The published `gamfit` wheel requires PyO3's `extension-module`
            // feature: it tells PyO3 NOT to link libpython, because the host
            // interpreter supplies those symbols at import time. A standalone
            // `cargo test` binary for `gam-pyffi`, however, MUST link libpython
            // (there is no host interpreter), so the feature has to be
            // toggleable off for the test harness. PyO3's documented recipe is
            // exactly this default-on passthrough feature plus `cargo test
            // --no-default-features`. There is no `cfg(feature = ...)` fork in
            // our own code — the sole consumer is PyO3's own build script — so
            // the conditionally-compiled-fork hazard this rule guards against
            // does not apply. Sanction the two canonical lines, and only in the
            // FFI crate's manifest.
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            let pyo3_extension_module_idiom = rel_str == "crates/gam-pyffi/Cargo.toml"
                && (code_part == "default = [\"extension-module\"]"
                    || code_part == "extension-module = [\"pyo3/extension-module\"]");
            if k < bytes.len()
                && bytes[k] == b'='
                && code_part != "default = []"
                && !code_part.starts_with("cuda = ")
                && !pyo3_extension_module_idiom
            {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Flags `Cargo.toml` `[lints.*]` entries whose right-hand side sets the
/// lint level to `"allow"`. Mirrors the `#[allow(...)]` ban: a
/// manifest-level allow is the same act of silencing a lint that flagged
/// real code. Both whole-category allows (`clippy.all = "allow"`,
/// `pedantic = { level = "allow", priority = -1 }`) and per-lint allows
/// (`too_many_arguments = "allow"`) are flagged.
///
/// Section headers covered: `[lints]`, `[lints.<group>]`,
/// `[workspace.lints]`, `[workspace.lints.<group>]`.
///
/// Forms detected (all collapse to the same quoted `"allow"` token):
/// - `<lint> = "allow"` (bare string form)
/// - `<lint> = { level = "allow", ... }` (table form)
/// - `<lint>.level = "allow"` (dotted-key form)
///
/// Other levels (`"deny"`, `"warn"`, `"forbid"`) are not flagged — the
/// ban targets the silencing direction only.
fn scan_for_cargo_lint_allows(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let basename = rel.file_name().and_then(OsStr::to_str).unwrap_or("");
        if basename != "Cargo.toml" {
            return;
        }
        let mut in_lints = false;
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            let code_part = match trimmed.find('#') {
                Some(p) => trimmed[..p].trim_end(),
                None => trimmed,
            };
            if code_part.starts_with('[') && code_part.ends_with(']') {
                let header = code_part
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim();
                in_lints = header == "lints"
                    || header.starts_with("lints.")
                    || header == "workspace.lints"
                    || header.starts_with("workspace.lints.");
                continue;
            }
            if !in_lints || code_part.is_empty() {
                continue;
            }
            if code_part.contains("\"allow\"") {
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
                let sj = stripped_lines
                    .get(j)
                    .map(String::as_str)
                    .unwrap_or(lines[j]);
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

/// True when `s` carries an attribute marker in either the outer (`#[...]`)
/// or inner (`#![...]`) form. Both must be tested explicitly: the substring
/// `#[` does NOT occur inside `#![` (the bytes are `#`, `!`, `[` — the `#`
/// is followed by `!`, not `[`), so a lone `contains("#[")` silently misses
/// every crate-/module-level inner attribute. Every scanner that early-outs
/// on "is this even an attribute line?" MUST funnel through this helper so
/// the inner-attribute blind spot cannot be reintroduced one scanner at a
/// time. Operate on a string/comment-stripped line so attribute syntax
/// quoted inside `"..."` or `//` text does not register.
fn line_has_attr_marker(s: &str) -> bool {
    s.contains("#[") || s.contains("#![")
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

/// Build a per-line bitmap of "is this line inside the body of a `trait
/// <Name> { ... }` definition?". Used to suppress the stub-function-body
/// check on trait default-method bodies: a default body of `None` /
/// `Ok(())` / `Default::default()` is the canonical Rust spelling of
/// "implementor may override; the trait itself has nothing useful to do"
/// and is not a stub in the "make the compile error vanish" sense the
/// ban exists to catch. Mirrors the `compute_trait_impl_mask` shape:
/// brace tracking uses the stripped-strings/comments rendering so braces
/// inside literals or `//` comments do not perturb depth. Only the FIRST
/// `{` following a top-level `trait ` header is matched as the trait body
/// — subsequent braces inside (fn bodies, match arms, blocks) are pushed
/// as non-trait-def scopes so we balance correctly on `}`.
fn compute_trait_def_mask(content: &str) -> Vec<bool> {
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();
    let mut mask = vec![false; n];
    let stripped_lines: Vec<String> = strip_file_lines(content);

    let mut stack: Vec<(i32, bool)> = Vec::new();
    let mut depth: i32 = 0;
    let mut pending_trait = false;

    for (idx, stripped) in stripped_lines.iter().enumerate() {
        let inside_trait_def = stack.iter().any(|(_, t)| *t);
        mask[idx] = inside_trait_def;

        let trimmed = stripped.trim_start();
        // Accept item-position `trait` header. Tolerate visibility / safety
        // prefixes (`pub`, `pub(crate)`, `pub(super)`, `unsafe`). Exclude
        // `impl ... for Trait` and `dyn Trait` shapes; those are handled by
        // `compute_trait_impl_mask` (and don't open a trait *definition*).
        let starts_trait = trimmed.starts_with("trait ")
            || trimmed.starts_with("trait<")
            || trimmed.starts_with("pub trait ")
            || trimmed.starts_with("pub trait<")
            || trimmed.starts_with("pub(crate) trait ")
            || trimmed.starts_with("pub(crate) trait<")
            || trimmed.starts_with("pub(super) trait ")
            || trimmed.starts_with("pub(super) trait<")
            || trimmed.starts_with("unsafe trait ")
            || trimmed.starts_with("unsafe trait<")
            || trimmed.starts_with("pub unsafe trait ")
            || trimmed.starts_with("pub unsafe trait<")
            || trimmed.starts_with("pub(crate) unsafe trait ")
            || trimmed.starts_with("pub(crate) unsafe trait<")
            || trimmed.starts_with("pub(super) unsafe trait ")
            || trimmed.starts_with("pub(super) unsafe trait<");
        if starts_trait && depth == 0 {
            pending_trait = true;
        }

        let bytes = stripped.as_bytes();
        for &b in bytes {
            if b == b'{' {
                let opened_at_depth = depth;
                depth += 1;
                if pending_trait && opened_at_depth == 0 {
                    stack.push((opened_at_depth, true));
                    pending_trait = false;
                } else {
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
    let mut raw_hashes: u8 = 0;
    for line in content.lines() {
        let (stripped, after_in_str, after_quote, after_hashes) =
            strip_strings_and_comments_stateful_raw(line, in_str, quote, raw_hashes);
        out.push(stripped);
        in_str = after_in_str;
        quote = after_quote;
        raw_hashes = after_hashes;
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
    let result = strip_strings_and_comments_stateful_raw(line, in_str_in, quote_in, 0);
    (result.0, result.1, result.2)
}

/// Raw-string-aware variant. Tracks Rust raw strings `r#"..."#` so the
/// embedded `"` characters do not toggle the regular-string state machine
/// and leak code outside the literal into the stripped output. `hashes_in`
/// is the active hash count for an open raw string (0 when not in one).
fn strip_strings_and_comments_stateful_raw(
    line: &str,
    in_str_in: bool,
    quote_in: u8,
    hashes_in: u8,
) -> (String, bool, u8, u8) {
    let bytes = line.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    let mut in_str = in_str_in;
    let mut str_quote: u8 = quote_in;
    let mut raw_hashes: u8 = hashes_in;
    // Raw-string mode is signaled by `raw_hashes > 0` *or* by `in_str` with
    // `str_quote == b'"'` and a sentinel; here we use `raw_hashes` separately
    // for the hash count and treat any `raw_hashes != 0` together with
    // `in_str` to mean "inside a raw string with that hash count". Raw
    // strings with zero hashes (`r"..."`) use `raw_hashes = 0` but a
    // distinct in-raw flag tracked via `str_quote == 0` won't work; instead
    // we encode "inside raw with N hashes" as `in_str && str_quote == b'r'`,
    // with `raw_hashes` holding N (including 0).
    while i < bytes.len() {
        let c = bytes[i];
        if in_str && str_quote == b'r' {
            // Inside a raw string. Close only on `"` followed by exactly
            // `raw_hashes` `#` bytes (greedy match is fine — Rust's tokenizer
            // requires exact match, but for a stripper any `"` followed by
            // at least N `#` ends the literal at this position).
            if c == b'"' {
                let need = raw_hashes as usize;
                let mut k = i + 1;
                let mut count = 0usize;
                while k < bytes.len() && bytes[k] == b'#' && count < need {
                    k += 1;
                    count += 1;
                }
                if count == need {
                    in_str = false;
                    str_quote = 0;
                    raw_hashes = 0;
                    // Emit the closing `"` and the `#`s verbatim so the
                    // stripped line keeps positional structure.
                    out.push(b'"');
                    for _ in 0..need {
                        out.push(b'#');
                    }
                    i = k;
                    continue;
                }
            }
            out.push(b' ');
            i += 1;
            continue;
        }
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
        // Raw string opener: `r"..."`, `r#"..."#`, `br"..."`, `br#"..."#`.
        // Detect both `r` and `br` prefixes followed by zero or more `#` and
        // then `"`. Use a word-boundary check so we don't trigger on
        // identifiers like `foo_r"..."` (illegal Rust anyway, but be safe).
        let prev_is_ident = i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if !prev_is_ident
            && (c == b'r' || (c == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'r'))
        {
            let prefix_len = if c == b'b' { 2usize } else { 1usize };
            let mut k = i + prefix_len;
            let mut hashes = 0usize;
            while k < bytes.len() && bytes[k] == b'#' {
                k += 1;
                hashes += 1;
            }
            if k < bytes.len() && bytes[k] == b'"' && hashes <= u8::MAX as usize {
                // Enter raw-string mode.
                in_str = true;
                str_quote = b'r';
                raw_hashes = hashes as u8;
                // Emit `r`/`br`, the `#`s, and the opening `"` verbatim.
                for j in i..=k {
                    out.push(bytes[j]);
                }
                i = k + 1;
                continue;
            }
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
    (s, in_str, str_quote, raw_hashes)
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
        if rel.to_string_lossy().replace('\\', "/") == "build.rs" {
            return;
        }
        if !content.contains(needle) {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            if line.contains(needle) {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Comment-lead deferred-work markers the bare `TODO` token misses. Flags two
/// shapes, both high-precision so legitimate prose is never caught:
///   1. a comment whose lead-in token is itself a pure marker word
///      (`fixme` / `xxx` / `hack` / `todo`) — these never legitimately begin a
///      comment;
///   2. a deferral word carrying an issue reference, e.g. `Follow-up (#932):`,
///      `Deferred(#41)` — the relabel shape that swaps `TODO(#N)` for a synonym
///      while keeping the tracker. A bare "follow-up period" or "deferred POTRF
///      scalar" in prose has no `(#issue)` tail and is NOT matched.
/// Skips build.rs (it names these words as part of its own contract). The marker
/// words are assembled from fragments so this scanner's source never carries the
/// literal tokens it forbids.
fn scan_for_deferred_work_markers(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    // `todo` is intentionally omitted — the bare-token TODO scanner already
    // covers it; listing it here would double-report the same line.
    let pure_markers: Vec<String> = vec![
        format!("{}{}", "fix", "me"),
        format!("{}{}", "x", "xx"),
        "hack".to_string(),
    ];
    let deferral_words: Vec<String> = vec![
        format!("{}-{}", "follow", "up"),
        format!("{}{}", "follow", "up"),
        format!("{}{}", "de", "ferred"),
        format!("{}{}", "de", "fer"),
        "revisit".to_string(),
        format!("{}{}", "stop", "gap"),
        "punt".to_string(),
        format!("{}{}", "t", "bd"),
        format!("{}{}", "w", "ip"),
        "later".to_string(),
        format!("{}{}", "place", "holder"),
    ];
    visit_files(root, dir, &mut |rel, content| {
        if rel.to_string_lossy().replace('\\', "/") == "build.rs" {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let Some(body) = comment_body(line) else {
                continue;
            };
            let body_l = body.to_lowercase();
            let mut hit = pure_markers.iter().any(|w| lead_token_is(&body_l, w));
            if !hit {
                hit = deferral_words.iter().any(|w| {
                    body_l
                        .strip_prefix(w.as_str())
                        .is_some_and(|rest| marker_issue_ref_follows(rest.trim_start()))
                });
            }
            if hit {
                offenders.push((rel.to_path_buf(), idx + 1, line.trim().to_string()));
            }
        }
    });
}

/// The text after a line's comment lead-in (`///`, `//!`, `//`, `/**`, `/*`, or
/// `#`), trimmed; `None` if the line is not a comment. The `#[...]` attribute
/// form is excluded so Rust attributes are not read as comments.
fn comment_body(line: &str) -> Option<&str> {
    let t = line.trim_start();
    for lead in ["///", "//!", "//", "/**", "/*", "#"] {
        if let Some(rest) = t.strip_prefix(lead) {
            if lead == "#" && rest.trim_start().starts_with('[') {
                return None;
            }
            return Some(rest.trim_start());
        }
    }
    None
}

/// True when `body` (already lowercased) begins with `word` as a whole lead-in
/// token — the char right after `word` is a non-alphanumeric boundary (or the
/// body ends). Stops `hack` from matching `hacky`.
fn lead_token_is(body: &str, word: &str) -> bool {
    match body.strip_prefix(word) {
        Some(rest) => rest
            .chars()
            .next()
            .map(|c| !c.is_ascii_alphanumeric())
            .unwrap_or(true),
        None => false,
    }
}

/// True when `rest` opens with an issue reference: a bracket (`(`, `[`, or `{`)
/// then an optional `#`, then at least one digit, then the matching close.
/// Matches the `(#932)` / `[#41]` / `{12}` tail of a relabelled tracker.
fn marker_issue_ref_follows(rest: &str) -> bool {
    let b = rest.as_bytes();
    let close = match b.first() {
        Some(b'(') => b')',
        Some(b'[') => b']',
        Some(b'{') => b'}',
        _ => return false,
    };
    let mut i = 1usize;
    if b.get(i) == Some(&b'#') {
        i += 1;
    }
    let digits_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits_start {
        return false;
    }
    b.get(i) == Some(&close)
}

/// HARD ban: owed work described in prose to dodge the marker bans — "this is
/// not done yet" phrased as a limitation/plan/follow-up so the bare-`TODO` and
/// deferred-marker scanners never see it. A relabel into "documented limitation"
/// is the SAME evasion as renaming the marker: the owed work still has to be
/// FINISHED, not reworded. Scans production `src/` comments only (tests and
/// examples legitimately describe unsupported paths). Phrases are assembled from
/// fragments so build.rs never carries them. Bare control-flow phrasings like
/// "deferred to <fn>" / "deferred until <stage>" are deliberately NOT matched —
/// those describe when a computation runs, not unfinished work.
/// Shared owed-work classifier for BOTH the prose ban (`scan_for_owed_work_prose`,
/// HEAD state) and the cosmetic-dodge audit (the removed-line diff check). `text`
/// MUST already be lowercased — both callers pass lowercased comment text.
///
/// The phrase set is split by linguistic category, which is the crux of avoiding
/// false positives WITHOUT weakening the gate:
///   * STRONG phrases carry an explicit temporal / future-deferral cue ("not yet
///     wired", "will be implemented", "deferred to a follow-up", "...yet"). They
///     ALWAYS mean owed work, so they match unconditionally.
///   * BARE-SCOPE phrases ("not wired into/through", "is/isn't wired") merely
///     describe WHERE something is or is not used. "PG is not wired into the
///     probit families" (it is logistic-only by theorem) and "the flat audit is
///     not wired through W" (structural rank is the intended check) are PERMANENT
///     facts, not deferrals. A bare-scope phrase therefore counts as owed work
///     ONLY when a temporal cue co-occurs on the same line — i.e. it was actually
///     a promise ("isn't wired through Z yet"), not a statement of scope.
///
/// This stays strict against laundering: rewording a temporal promise away still
/// fires the cosmetic-dodge diff check (which inspects the REMOVED line, where the
/// temporal cue still lived), and every strong phrase is untouched. It only stops
/// the gate from manufacturing false positives on honest scope statements — which
/// is exactly what drove rewording in the first place.
fn comment_text_is_owed_work(text: &str) -> bool {
    let strong: Vec<String> = vec![
        format!("not yet {}", "wired"),
        format!("not yet {}", "implemented"),
        format!("not yet {}", "supported"),
        format!("not yet {}", "hooked"),
        format!("not yet {}", "done"),
        format!("not yet {}", "finished"),
        format!("not {} yet", "implemented"),
        format!("not {} yet", "supported"),
        format!("not {} yet", "wired"),
        format!("to be {} yet", "wired"),
        format!("deferred to a {}", "follow"),
        format!("left to a {}", "follow"),
        format!("for a {}-up", "follow"),
        format!("in a {}-up", "follow"),
        format!("to a {}-up", "follow"),
        format!("remains to be {}", "implemented"),
        format!("needs to be {}", "implemented"),
        format!("will be {}", "implemented"),
        format!("will be {}", "wired"),
        format!("yet to be {}", "implemented"),
        format!("yet to be {}", "wired"),
    ];
    if strong.iter().any(|p| text.contains(p.as_str())) {
        return true;
    }
    let bare_scope: Vec<String> = vec![
        format!("not wired {}", "into"),
        format!("not wired {}", "through"),
        format!("is not {}", "wired"),
        format!("{} wired", "isn't"),
    ];
    if bare_scope.iter().any(|p| text.contains(p.as_str())) {
        // A bare scope statement is owed work only if it also promises a future
        // change on the same line. Cues are kept tight so a permanent scope fact
        // that merely happens to contain one of these words elsewhere is safe.
        let temporal = [" yet", " will ", " soon", " later", " eventually", " once "];
        return temporal.iter().any(|c| text.contains(c));
    }
    false
}

fn scan_for_owed_work_prose(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if !rel_str.starts_with("src/") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let Some(text) = line_comment_text(line) else {
                continue;
            };
            if comment_text_is_owed_work(&text) {
                offenders.push((rel.to_path_buf(), idx + 1, line.trim().to_string()));
            }
        }
    });
}

/// The lowercased text of a line's `//` comment (lead or trailing), or `None` if
/// the line carries no line comment. Skips `://` so URLs/paths are not read as
/// comments. Covers `//`, `///`, and `//!`.
fn line_comment_text(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        if bytes[i] == b'/' && bytes[i + 1] == b'/' {
            if i > 0 && bytes[i - 1] == b':' {
                i += 2;
                continue;
            }
            return Some(line[i + 2..].to_lowercase());
        }
        i += 1;
    }
    None
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
            // line: preceded somewhere by `#[` or `#![`. Both forms must be
            // tested — `#[` is NOT a substring of `#![` (bytes `#`, `!`, `[`),
            // so a lone `#[` check silently misses crate-/module-level
            // `#![allow(...)]` / `#![expect(...)]` inner attributes.
            if !line_has_attr_marker(code) {
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
                    let mut any_lint = false;
                    for tok in inside.split(',') {
                        let trimmed = tok.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        any_lint = true;
                        if first_label.is_none() {
                            let bare = trimmed
                                .trim_start_matches("rustc::")
                                .trim_start_matches("rustdoc::");
                            first_label = Some(bare.to_string());
                        }
                    }
                    if any_lint {
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
        // The `let _` ban is STRICT EVERYWHERE — test code included (no test
        // mask). A discarded binding silences the type system's unused-value
        // feedback just as harmfully in a test as in production: a test that
        // `let _ = foo()`s away a `Result` stops asserting `foo` succeeded.
        // Bind a real name and read it, assert on it, or call the expression
        // for its effect without a binding.
        let stripped_lines = strip_file_lines(content);
        for (idx, line) in content.lines().enumerate() {
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
        if (name.starts_with('.') && name != ".github")
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

const MAX_TRACKED_FILE_LINES: usize = 10_000;

/// Cached result of [`collect_repo_files`].
static REPO_FILES: OnceLock<Vec<PathBuf>> = OnceLock::new();

/// Tracked-file list for the line-count and mechanical-part-name audits, read
/// directly from `.git/index`.
///
/// History: the audits originally shelled out to `git ls-files`, but that
/// panicked inside maturin's manylinux / musllinux Docker images (the
/// bind-mounted source tree trips git's `safe.directory` check and the
/// command returns non-zero). A naive replacement that walked the filesystem
/// then broke the wheel build a different way: GitHub Actions steps install
/// CUDA into the workspace at `<root>/cuda_installer-*`, and that 16M-line
/// installer blob is neither a build artifact nor in `.gitignore`, so the
/// line-count audit flagged it and failed the build.
///
/// Parsing `.git/index` directly avoids both failure modes: no dependency on
/// the `git` binary or its `safe.directory` config, and the canonical
/// definition of "what belongs to this repo" — the index, i.e. the staged
/// tree — naturally excludes anything CI dropped into the workspace.
fn collect_repo_files(root: &Path) -> &'static [PathBuf] {
    REPO_FILES
        .get_or_init(|| read_git_index_tracked_files(root))
        .as_slice()
}

/// The two cluster/SLURM infra leak needles, assembled from fragments so this
/// file's own source text never contains either banned string verbatim (the
/// scanner would otherwise flag itself). `needle_a` = the absolute cluster
/// scratch path segment; `needle_b` = the SLURM batch directive keyword.
fn cluster_leak_needles() -> [String; 2] {
    let needle_a = format!("projects{}standard", "/");
    let needle_b = format!("{}BATCH", "S");
    [needle_a, needle_b]
}

/// HARD-FAIL ban: no git-tracked file outside `experiments/` (source OR not)
/// may contain the absolute cluster scratch path segment or the SLURM batch
/// directive keyword. These are cluster-local infra leaks (absolute compute-node
/// paths + SLURM job directives) that must never be committed outside
/// experiment artifacts. Scans only tracked text files (from the git index),
/// skips binaries, and fails the build naming every offender. This is a
/// separate, always-fatal gate — independent of the demoted aggregate
/// ban-scanner below — so the leak can never ship outside `experiments/`.
fn scan_for_cluster_infra_leaks(root: &Path) {
    let needles = cluster_leak_needles();
    let mut offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    for rel in collect_repo_files(root) {
        if rel.starts_with("experiments") {
            continue;
        }
        let path = root.join(rel);
        let content = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => continue,
        };
        // Skip files that are not valid UTF-8 text (binaries can't carry the
        // ASCII needles meaningfully and lossy-decoding them wastes time).
        let text = match std::str::from_utf8(&content) {
            Ok(text) => text,
            Err(_) => continue,
        };
        for (lineno, line) in text.lines().enumerate() {
            for needle in &needles {
                if line.contains(needle.as_str()) {
                    offenders.push((rel.clone(), lineno + 1, needle.clone()));
                }
            }
        }
    }
    if offenders.is_empty() {
        return;
    }
    eprintln!(
        "\n=== BANNED cluster/SLURM INFRA LEAK in tracked file(s) ===\n\
         No tracked file outside experiments/ may contain the absolute cluster scratch path segment or the \
         SLURM batch directive keyword. These are cluster-local infra leaks; cluster/sbatch \
         scripts outside experiments/ belong under /Users/user/, not in the repo. Offenders:"
    );
    for (rel, lineno, needle) in &offenders {
        eprintln!(
            "  {}:{} contains banned string `{}`",
            rel.display(),
            lineno,
            needle
        );
        println!(
            "cargo:warning=banned cluster/SLURM infra leak: {}:{} contains `{}`",
            rel.display(),
            lineno,
            needle
        );
    }
    std::process::exit(1);
}

/// Resolve the path of `.git/index`, following the worktree pointer if `.git`
/// is a file (`gitdir: <path>`) rather than a directory.
fn locate_git_index(root: &Path) -> PathBuf {
    let git = root.join(".git");
    let meta = fs::metadata(&git).unwrap_or_else(|err| {
        panic!(
            "failed to stat {} for tracked-file audit: {err}",
            git.display()
        )
    });
    if meta.is_dir() {
        return git.join("index");
    }
    // `.git` is a file in linked worktrees / submodules; first line is
    // `gitdir: <absolute or relative path to the real gitdir>`.
    let pointer = fs::read_to_string(&git)
        .unwrap_or_else(|err| panic!("failed to read worktree pointer {}: {err}", git.display()));
    let gitdir = pointer
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("gitdir:"))
        .map(|rest| rest.trim())
        .unwrap_or_else(|| panic!("worktree pointer at {} missing gitdir line", git.display()));
    let gitdir_path = if Path::new(gitdir).is_absolute() {
        PathBuf::from(gitdir)
    } else {
        root.join(gitdir)
    };
    gitdir_path.join("index")
}

fn read_git_index_tracked_files(root: &Path) -> Vec<PathBuf> {
    // Source-archive builds (sdists, GitHub tarballs) have no `.git` at all.
    // The tracked-file audits are repo-hygiene gates, not build correctness
    // gates, and "what belongs to this repo" is undefined without an index —
    // so skip them loudly instead of panicking the whole wheel build.
    if !root.join(".git").exists() {
        println!(
            "cargo:warning=no .git at {}; skipping tracked-file audits (source-archive build)",
            root.display()
        );
        return Vec::new();
    }
    let index_path = locate_git_index(root);
    let bytes = fs::read(&index_path).unwrap_or_else(|err| {
        panic!(
            "failed to read git index for tracked-file audit ({}): {err}",
            index_path.display()
        )
    });
    parse_git_index(&bytes).unwrap_or_else(|err| {
        panic!(
            "failed to parse git index at {}: {err}",
            index_path.display()
        )
    })
}

/// Parse the on-disk git index (v2, v3, v4) and return one repo-relative
/// `PathBuf` per stage-0 entry. Index format reference:
/// `Documentation/gitformat-index.txt` in the git source tree.
fn parse_git_index(bytes: &[u8]) -> Result<Vec<PathBuf>, String> {
    if bytes.len() < 12 {
        return Err(format!("index too short for header: {} bytes", bytes.len()));
    }
    if &bytes[0..4] != b"DIRC" {
        return Err(format!("bad signature {:?}", &bytes[0..4]));
    }
    let version = u32::from_be_bytes(bytes[4..8].try_into().expect("4-byte slice"));
    if !matches!(version, 2 | 3 | 4) {
        return Err(format!("unsupported index version {version}"));
    }
    let count = u32::from_be_bytes(bytes[8..12].try_into().expect("4-byte slice")) as usize;
    let mut out: Vec<PathBuf> = Vec::with_capacity(count);
    let mut pos = 12usize;
    let mut prev_path: Vec<u8> = Vec::new();
    for entry_idx in 0..count {
        let entry_start = pos;
        // 62-byte fixed prefix: ctime/mtime (16) + dev/ino/mode/uid/gid/size (24) +
        // sha1 (20) + flags (2).
        if bytes.len() < pos + 62 {
            return Err(format!("truncated entry {entry_idx} fixed header"));
        }
        let flags = u16::from_be_bytes(bytes[pos + 60..pos + 62].try_into().expect("2-byte slice"));
        // Entry mode: 4 bytes at offset 24 within the 62-byte fixed prefix
        // (after ctime/mtime=16 + dev/ino=8). The low 16 bits are the git
        // st_mode; 0o160000 (S_IFGITLINK) marks a submodule gitlink — a commit
        // pointer, NOT a readable file. The line-count audit must skip it, or it
        // panics trying to read the submodule path as a file.
        let entry_mode =
            u32::from_be_bytes(bytes[pos + 24..pos + 28].try_into().expect("4-byte slice"));
        let is_gitlink = (entry_mode & 0o170000) == 0o160000;
        pos += 62;
        let extended = (flags & 0x4000) != 0;
        if extended {
            if version < 3 {
                return Err(format!("entry {entry_idx} has extended flag in v{version}"));
            }
            if bytes.len() < pos + 2 {
                return Err(format!("truncated entry {entry_idx} extended flags"));
            }
            pos += 2;
        }
        let stage = ((flags >> 12) & 0x3) as u8;
        let name_len_hint = (flags & 0x0FFF) as usize;

        let path_bytes: Vec<u8>;
        if version == 4 {
            // v4 path compression: chop N bytes off the previous path, then
            // append a NUL-terminated suffix. No trailing padding.
            let (chop, consumed) = decode_index_varint(&bytes[pos..])
                .map_err(|e| format!("entry {entry_idx} varint: {e}"))?;
            pos += consumed;
            let keep = prev_path.len().checked_sub(chop).ok_or_else(|| {
                format!(
                    "entry {entry_idx} v4 underflow: prev={} chop={chop}",
                    prev_path.len()
                )
            })?;
            let nul_offset = bytes[pos..]
                .iter()
                .position(|b| *b == 0)
                .ok_or_else(|| format!("entry {entry_idx} missing NUL after v4 suffix"))?;
            let mut p = Vec::with_capacity(keep + nul_offset);
            p.extend_from_slice(&prev_path[..keep]);
            p.extend_from_slice(&bytes[pos..pos + nul_offset]);
            pos += nul_offset + 1;
            path_bytes = p;
        } else {
            // v2/v3: name is name_len_hint bytes (or longer if the hint is
            // saturated at 0xFFF), followed by 1-8 NUL bytes padding the
            // entry to a multiple of 8 from `entry_start`.
            let name_end = if name_len_hint < 0x0FFF
                && bytes.len() >= pos + name_len_hint + 1
                && bytes[pos + name_len_hint] == 0
            {
                pos + name_len_hint
            } else {
                pos + bytes[pos..]
                    .iter()
                    .position(|b| *b == 0)
                    .ok_or_else(|| format!("entry {entry_idx} missing NUL terminator"))?
            };
            path_bytes = bytes[pos..name_end].to_vec();
            pos = name_end;
            // Skip 1-8 padding NULs so the entry length is a multiple of 8.
            let raw = pos - entry_start;
            let pad = if raw % 8 == 0 { 8 } else { 8 - (raw % 8) };
            pos += pad;
        }

        prev_path = path_bytes.clone();

        // Stage 0 = ordinary tracked file; stages 1/2/3 are conflict variants
        // of the same path. Take stage 0 only so the audit sees each path once.
        if stage != 0 {
            continue;
        }
        // Submodule gitlinks are commit pointers, not files — skip so the
        // tracked-file line-count audit never tries to read a directory.
        if is_gitlink {
            continue;
        }
        let path_str = std::str::from_utf8(&path_bytes)
            .map_err(|_| format!("entry {entry_idx} path is not UTF-8"))?;
        out.push(PathBuf::from(path_str));
    }
    Ok(out)
}

/// Decode the offset-style variable-length integer used by index v4 to encode
/// the number of bytes to chop off the previous path. Mirrors
/// `decode_varint` in git's `read-cache.c`. Returns (value, bytes_consumed).
fn decode_index_varint(buf: &[u8]) -> Result<(usize, usize), String> {
    let mut consumed = 0usize;
    if consumed >= buf.len() {
        return Err("empty buffer".to_string());
    }
    let mut c = buf[consumed];
    consumed += 1;
    let mut value = (c & 0x7F) as usize;
    while c & 0x80 != 0 {
        if consumed >= buf.len() {
            return Err("truncated varint".to_string());
        }
        value += 1;
        value <<= 7;
        c = buf[consumed];
        consumed += 1;
        value |= (c & 0x7F) as usize;
    }
    Ok((value, consumed))
}

fn scan_for_oversized_tracked_files(root: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    for rel in collect_repo_files(root) {
        let path = root.join(rel);
        let line_count = match count_file_lines(&path) {
            Ok(line_count) => line_count,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                panic!(
                    "failed to read repo file for line-count audit: {}: {err}",
                    rel.display()
                )
            }
        };
        if line_count > MAX_TRACKED_FILE_LINES {
            offenders.push((
                rel.clone(),
                line_count,
                format!("{line_count} lines; limit is {MAX_TRACKED_FILE_LINES}"),
            ));
        }
    }
}

fn count_file_lines(path: &Path) -> std::io::Result<usize> {
    let file = fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buf = [0_u8; 64 * 1024];
    let mut line_count = 0usize;
    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            return Ok(line_count);
        }
        line_count += buf[..read].iter().filter(|byte| **byte == b'\n').count();
    }
}

/// Committed ledger recording every tracked file that has ever breached the
/// 10k hard limit and has not yet earned redemption by dropping to <=7k. Like
/// `ban_history.txt` / `todo_history.txt`, it is auto-managed by build.rs and
/// must not be hand-edited; each non-comment line is a single repo-relative
/// path, sorted for clean git merges.
const OVERSIZED_PROBATION_LEDGER_FILENAME: &str = "oversized_history.txt";

/// A file on probation (once >10k lines) must drop below this to be redeemed.
/// The 7k floor forces a ~30% cut off the 10k limit — enough that a thin
/// tag-along satellite cannot satisfy it, so the author must find a real seam.
const OVERSIZED_PROBATION_FLOOR_LINES: usize = 7_000;

/// Sticky-probation audit. Records any tracked file currently over the 10k hard
/// limit into the probation ledger, then flags any ledger file still over the
/// 7k redemption floor. Files that have dropped to <=7k (or are no longer
/// tracked) are pruned. Emitted probation offenders are only those in the
/// 7k..=10k band — files still over 10k are already reported by the primary
/// `scan_for_oversized_tracked_files` gate, so we do not double-report them.
fn scan_for_probation_oversized_files(root: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    let ledger_path = root.join(OVERSIZED_PROBATION_LEDGER_FILENAME);
    let mut ledger = load_probation_ledger(&ledger_path);

    // Current line count for every tracked file, keyed by repo-relative path.
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for rel in collect_repo_files(root) {
        let path = root.join(rel);
        let line_count = match count_file_lines(&path) {
            Ok(line_count) => line_count,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                panic!(
                    "failed to read repo file for oversized-probation audit: {}: {err}",
                    rel.display()
                )
            }
        };
        counts.insert(rel.to_string_lossy().into_owned(), line_count);
    }

    // Trip: any file currently over the hard limit joins probation.
    for (key, &line_count) in &counts {
        if line_count > MAX_TRACKED_FILE_LINES {
            ledger.insert(key.clone());
        }
    }

    // Enforce + prune. Iterate a snapshot so we can mutate `ledger`.
    for key in ledger.clone() {
        match counts.get(&key) {
            // No longer tracked (deleted/renamed) — nothing to enforce; prune.
            None => {
                ledger.remove(&key);
            }
            // Redeemed: the real ~30% cut landed. Prune.
            Some(&n) if n <= OVERSIZED_PROBATION_FLOOR_LINES => {
                ledger.remove(&key);
            }
            // Still over 10k — the primary gate reports this; keep on probation
            // but do not double-report here.
            Some(&n) if n > MAX_TRACKED_FILE_LINES => {}
            // In the 7k..=10k band while on probation — this is the dodge.
            Some(&n) => {
                offenders.push((
                    PathBuf::from(&key),
                    n,
                    format!(
                        "{n} lines; on probation after exceeding {MAX_TRACKED_FILE_LINES} — must \
                         reach at most {OVERSIZED_PROBATION_FLOOR_LINES}"
                    ),
                ));
            }
        }
    }

    save_probation_ledger(&ledger_path, &ledger);
}

fn load_probation_ledger(path: &Path) -> std::collections::BTreeSet<String> {
    let mut out: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return out,
    };
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        out.insert(trimmed.to_string());
    }
    out
}

fn save_probation_ledger(path: &Path, ledger: &std::collections::BTreeSet<String>) {
    // Never create an empty ledger file just to hold the header: if nothing has
    // ever tripped and no file exists yet, leave the tree clean.
    if ledger.is_empty() && !path.exists() {
        return;
    }
    let mut out = String::new();
    out.push_str(
        "# Persistent audit of files that once exceeded the 10k-line hard limit.\n\
         # Auto-managed by build.rs — do NOT hand-edit. Each non-comment line is a\n\
         # single repo-relative path currently on probation: it breached 10k lines\n\
         # and must be brought to at most 7k lines (a real ~30% cut / cohesive\n\
         # split, not a thin tag-along satellite) before it is auto-pruned here.\n\
         # Concurrent branches produce additive line-level diffs that union-merge;\n\
         # the next build re-validates and prunes redeemed entries.\n",
    );
    for key in ledger {
        out.push_str(key);
        out.push('\n');
    }
    match fs::read_to_string(path) {
        Ok(existing) if existing == out => return,
        _ => {}
    }
    if fs::write(path, out).is_err() {
        // Non-fatal: write failures (e.g. read-only checkout) leave the ledger
        // stale rather than failing the build. Future runs retry.
    }
}

/// Is this repo-relative path a mechanical line-count split rather than a logical
/// module? True when the file stem is `part_<digits>` OR any ancestor directory
/// component ends in `_parts` or is exactly `split_parts`.
fn is_mechanical_part_path(rel: &Path) -> bool {
    if let Some(stem) = rel.file_stem().and_then(|s| s.to_str()) {
        let is_part_n = stem
            .strip_prefix("part_")
            .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()));
        if is_part_n {
            return true;
        }
    }
    for comp in rel.components() {
        if let std::path::Component::Normal(os) = comp {
            if let Some(name) = os.to_str() {
                if name == "split_parts" || name.ends_with("_parts") {
                    return true;
                }
            }
        }
    }
    false
}

/// Ban mechanical file-splitting (`part_<NNN>.rs`, `*_parts/`, `split_parts/`).
/// HARD ban with NO grandfathering: every such path fails the build. Split each
/// module by cohesive concern into descriptively-named modules instead. Scans
/// every `.rs` file anywhere in the repo (src/, tests/, crates/, …), so no
/// future mechanical split can slip in outside src/. Non-code dataset shards
/// such as `bench/datasets/*_parts/part_00N.csv` are skipped because the audit
/// only considers `.rs` files.
fn scan_for_mechanical_part_files(root: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    for rel in collect_repo_files(root) {
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            continue;
        }
        if !is_mechanical_part_path(rel) {
            continue;
        }
        offenders.push((
            rel.clone(),
            0,
            "mechanical line-count split; decompose by cohesive concern into descriptively-named \
             modules — this is a HARD ban, no grandfathering"
                .to_string(),
        ));
    }
}

struct ScannedFile {
    rel: PathBuf,
    content: String,
}

static SCANNABLE_FILES: OnceLock<Vec<ScannedFile>> = OnceLock::new();

fn emit_workspace_scanner_rerun_roots(root: &Path) {
    emit_rerun_if_dir_exists(&root.join("src"));
    emit_rerun_if_dir_exists(&root.join("examples"));
    emit_rerun_if_dir_exists(&root.join("gamfit"));

    let crates_dir = root.join("crates");
    let Ok(entries) = fs::read_dir(&crates_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let crate_src = entry.path().join("src");
        emit_rerun_if_dir_exists(&crate_src);
    }
}

fn emit_rerun_if_dir_exists(path: &Path) {
    if path.is_dir() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DerivativeSpecializationKind {
    RowKernel,
    RowAtom,
    Bespoke,
}

struct DerivativeSpecialization {
    family: &'static str,
    kind: DerivativeSpecializationKind,
    production_sources: &'static [DerivativeAnchorSet],
    discovery_anchor: &'static str,
    parity_pins: &'static [DerivativeAnchorSet],
    retired_identities: &'static [&'static str],
}

struct DerivativeAnchorSet {
    path: &'static str,
    anchors: &'static [&'static str],
}

const PRODUCTION_DERIVATIVE_SPECIALIZATIONS: &[DerivativeSpecialization] = &[
    DerivativeSpecialization {
        family: "BMS rigid Bernoulli",
        kind: DerivativeSpecializationKind::RowKernel,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/bms/row_kernel.rs",
            anchors: &[
                "impl gam_math::jet_tower::RowProgram<2> for BernoulliRigidRowKernel",
                "impl RowKernel<2> for BernoulliRigidRowKernel",
            ],
        }],
        discovery_anchor: "impl RowKernel<2> for BernoulliRigidRowKernel",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/bms/gradient_paths.rs",
            anchors: &[
                "fn rigid_bernoulli_row_kernel_agrees_with_jet_tower_program_all_channels()",
                "verify_kernel_channels(&tower, &claims, 1e-9)",
            ],
        }],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "BMS FLEX Bernoulli",
        kind: DerivativeSpecializationKind::Bespoke,
        production_sources: &[
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/flex_row_program.rs",
                anchors: &[
                    "pub(super) struct BmsFlexRowProgram",
                    "pub(super) fn evaluate<'arena, S: RuntimeJetScalar<'arena>>(",
                    "let intercept = filtered_implicit_solve_runtime_scalar(",
                    "Ok(signed.compose_unary(self.observed_neglog_stack))",
                    "pub(super) fn try_for_each_calibration_order2<E>(",
                    "pub(super) fn try_for_each_calibration_order3_contiguous<E>(",
                    "pub(super) fn try_for_each_calibration_order4_contiguous<E>(",
                    "pub(super) fn try_for_each_order2_finalizer<E>(",
                    "pub(super) fn try_for_each_order3_finalizer<E>(",
                    "pub(super) fn try_for_each_order4_finalizer<E>(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/cell_moment_assembly.rs",
                anchors: &[
                    "pub(super) fn empirical_flex_row_third_contracted_many(",
                    "let jet = plan.evaluate(vars, 3, &workspace)?;",
                    "pub(super) fn empirical_dynamic_fourth_batch_from_plan(",
                    "let jet = plan.evaluate(vars, 4, &workspace)?;",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/row_primary_hessian.rs",
                anchors: &[
                    "pub(super) fn lower_bms_flex_row_order2(",
                    "pub(super) fn lower_bms_flex_row_order2_with_moments(",
                    "pub(super) fn lower_bms_flex_row_order2_from_parts(",
                    "BmsFlexRowProgram::try_for_each_calibration_order2(",
                    "BmsFlexRowProgram::try_for_each_calibration_order3_contiguous(",
                    "BmsFlexRowProgram::try_for_each_calibration_order4_contiguous(",
                    "BmsFlexRowProgram::try_for_each_order2_finalizer(",
                    "BmsFlexRowProgram::try_for_each_order3_finalizer(",
                    "BmsFlexRowProgram::try_for_each_order4_finalizer(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/gpu/row.rs",
                anchors: &[
                    "fn build_generated_row_kernel_source() -> String",
                    "BmsFlexRowProgram::try_for_each_calibration_order2_phase(",
                    "BmsFlexRowProgram::try_for_each_order2_finalizer_phase(",
                    "SOURCE.get_or_init(build_generated_row_kernel_source)",
                ],
            },
        ],
        discovery_anchor: "pub(super) struct BmsFlexRowProgram",
        parity_pins: &[
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/flex_verify_932_tests.rs",
                anchors: &[
                    "fn standard_normal_flex_canonical_derivative_ladder_matches_vgh_t3_t4_932()",
                    ".lower_bms_flex_row_order2_with_moments(",
                    ".row_primary_third_contracted_with_moments(",
                    ".row_primary_fourth_contracted_ordered(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/gpu/row.rs",
                anchors: &[
                    "fn generated_cuda_row_kernel_matches_canonical_cpu_lowering_415()",
                    "fn generated_cuda_row_kernel_r33_matches_canonical_cpu_lowering_932()",
                    "fn bms_flex_row_r33_consumers_match_cpu_oracles_when_cuda_available()",
                    "fn mandatory_required_gpu_workspace_consumes_device_cache_end_to_end_932()",
                    "fn release_measure_generated_bms_full_row_vs_strongest_cpu_932()",
                    "fn generated_source_interprets_compact_canonical_phase_streams()",
                ],
            },
        ],
        retired_identities: &[
            "ROW_KERNEL_BODY",
            "cpu_oracle_outputs",
            "compute_row_analytic_flex_into",
            "compute_row_analytic_flex_into_with_moments",
            "compute_row_analytic_flex_from_parts_into",
        ],
    },
    DerivativeSpecialization {
        family: "survival location-scale",
        kind: DerivativeSpecializationKind::RowKernel,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/location_scale/row_kernel.rs",
            anchors: &[
                "impl gam_math::jet_tower::RowProgram<SLS_ROW_K> for SurvivalLsRowKernel<'_>",
                "impl crate::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_>",
                "struct SlsIndexDerivativeChannels {",
                "fn project_index_diagonal<const CHANNELS: usize, const ORDER: usize>(",
                "fn lower_index_derivative_channels(self) -> SlsIndexDerivativeChannels",
                "let channels = sls_outer_plan(&kernel).lower_index_derivative_channels();",
            ],
        }],
        discovery_anchor: "impl crate::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_>",
        parity_pins: &[
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/location_scale/tests.rs",
                anchors: &[
                    "fn survival_ls_joint_row_kernel_agrees_with_jet_tower_program_all_channels()",
                    "verify_kernel_channels(&tower, &claims, 1e-9)",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/location_scale/row_kernel.rs",
                anchors: &[
                    "fn sls_index_sparse_lowering_matches_generic_jet_all_branches_932()",
                    "fn sls_index_sparse_lowering_matches_independent_fd_all_branches_932()",
                ],
            },
        ],
        retired_identities: &["nll_index_read_channels", "SurvivalIndexNllReadChannels"],
    },
    DerivativeSpecialization {
        family: "survival marginal-slope rigid",
        kind: DerivativeSpecializationKind::RowKernel,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/marginal_slope/row_kernel.rs",
            anchors: &[
                "impl gam_math::jet_tower::RowProgram<4> for SurvivalMarginalSlopeRowKernel",
                "impl RowKernel<4> for SurvivalMarginalSlopeRowKernel",
            ],
        }],
        discovery_anchor: "impl RowKernel<4> for SurvivalMarginalSlopeRowKernel",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/marginal_slope/tests.rs",
            anchors: &[
                "fn rigid_row_kernel_agrees_with_jet_tower_program_all_channels()",
                "verify_kernel_channels(&tower, &claims, 1e-9)",
            ],
        }],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "survival marginal-slope FLEX",
        kind: DerivativeSpecializationKind::Bespoke,
        production_sources: &[
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/marginal_slope/timepoint_exact/flex_jet.rs",
                anchors: &[
                    "trait FlexJet: JetField + Clone {",
                    "struct FlexOuterPlan {",
                    "fn flex_row_nll<J: FlexJet>(",
                    "fn lower_flex_outer_plan_order2(",
                    "fn flex_timepoint_inputs_generic<J: FlexJet + MomentTerm>(",
                    "pub(crate) fn flex_row_nll_value_grad_hess(",
                    "pub(crate) fn flex_row_nll_third_contracted(",
                    "pub(crate) fn flex_row_nll_fourth_contracted(",
                    "pub(crate) fn compute_survival_timepoint_exact_jet(",
                    "pub(crate) fn compute_survival_timepoint_first_order_exact(",
                    "pub(crate) fn compute_survival_timepoint_directional_jet_from_cached(",
                    "pub(crate) fn compute_survival_timepoint_bidirectional_jet_from_cached(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/marginal_slope/flex_sensitivity.rs",
                anchors: &[
                    "let entry = self.compute_survival_timepoint_first_order_exact(",
                    "let (row_nll, grad, _) = self.flex_row_nll_value_grad_hess(",
                    "let entry = self.compute_survival_timepoint_exact_jet(",
                    "let (row_nll, grad, hess) = self.flex_row_nll_value_grad_hess(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/marginal_slope/timepoint_exact/contracted.rs",
                anchors: &[
                    ".compute_survival_timepoint_directional_jet_from_cached(",
                    "self.flex_row_nll_third_contracted(",
                    "let entry_bi = self.compute_survival_timepoint_bidirectional_jet_from_cached(",
                    "self.flex_row_nll_fourth_contracted(",
                ],
            },
        ],
        discovery_anchor: "fn flex_row_nll<J: FlexJet>(",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/marginal_slope/timepoint_exact/flex_jet.rs",
            anchors: &[
                "fn compiled_order2_row_nll_matches_generic_plan()",
                "fn flex_timepoint_first_order_matches_jet2_and_fd_932()",
                "fn flex_timepoint_inputs_jet3_directional_matches_hand_932()",
                "fn flex_timepoint_inputs_jet4_bidirectional_matches_hand_932()",
                "fn flex_timepoint_inputs_nested_dual_matches_jet4_contraction_932()",
                "fn flex_timepoint_inputs_ghw_jet3_jet4_match_hand_932()",
            ],
        }],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "multinomial Fisher",
        kind: DerivativeSpecializationKind::Bespoke,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/multinomial_reml.rs",
            anchors: &[
                "pub struct MultinomialLogitRowProgram<'row>",
                "impl<const M: usize> gam_math::jet_tower::RowProgram<M> for MultinomialLogitRowProgram<'_>",
                "fn eval_expression<S: JetField>",
                "fn negative_log_likelihood_from_normalization",
                "pub(crate) fn value_gradient_hessian_into",
                "fn softmax_fisher_perturbation<S: FisherPerturbation>",
            ],
        }],
        discovery_anchor: "fn softmax_fisher_perturbation<S: FisherPerturbation>",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/multinomial_reml.rs",
            anchors: &[
                "fn multinomial_live_tower_matches_jet_and_fd()",
                "fn multinomial_extreme_tails_share_one_stable_row_program_932()",
            ],
        }],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "SAE reconstruction row jets",
        kind: DerivativeSpecializationKind::Bespoke,
        production_sources: &[
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/row_jet_program.rs",
                anchors: &[
                    "pub struct SaeReconstructionRowProgram {",
                    "pub fn reconstruction_all_columns_dynamic<'arena>(",
                    "pub fn beta_border_order1_dynamic<'arena>(",
                    "pub(crate) trait SaeSoftmaxRowProgramSource {",
                    "struct SoftmaxMoment<'a, S> {",
                    "pub(crate) fn execute_softmax_row_program<S: SaeSoftmaxRowProgramSource>(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/manifold/construction_row_jet_logdet_channels.rs",
                anchors: &[
                    "impl crate::row_jet_program::SaeSoftmaxRowProgramSource for ProductionSoftmaxRowProgram<'_> {",
                    "pub(crate) fn reconstruction_row_program_for_logdet(",
                    "fn fill_reconstruction_channels_from_program_dynamic(",
                    "fn fill_beta_border_channels_from_program_dynamic(",
                    "pub(crate) fn row_jets_for_logdet(",
                    "let scheduled = crate::row_jet_program::execute_softmax_row_program(",
                    "let plan = crate::gpu_kernels::sae_rowjet::plan_softmax_row_jets(",
                    "let input = crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput::from_source(",
                    "let channels = crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/gpu_kernels/sae_rowjet.rs",
                anchors: &[
                    "pub fn plan_softmax_row_jets(",
                    "pub fn execute_softmax_row_jet_tile(",
                    "impl SaeSoftmaxRowProgramSource for InputSource<'_> {",
                    "let scheduled = execute_softmax_row_program(&source, inv_tau, input.sqrt_row_weight);",
                    "pub const COMPLETE_SOFTMAX_KERNEL_SOURCE: &str = r#\"",
                ],
            },
        ],
        discovery_anchor: "SaeSoftmaxRowProgramSource for",
        parity_pins: &[
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/row_jet_program.rs",
                anchors: &[
                    "fn recon_jet_matches_hand_path_value_grad_hess()",
                    "fn compiled_softmax_schedule_matches_generic_tower_all_channels_932()",
                    "fn runtime_row_jets_match_fixed_oracle_above_old_arity_ceiling_932()",
                    "fn softmax_reconstruction_t3_t4_match_independent_fd_witness()",
                    "fn planted_t3_t4_corruption_is_caught_by_fd_witness()",
                    "fn planted_cross_block_sign_flip_is_caught()",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/manifold/tests_row_jet_and_outer_objective_780.rs",
                anchors: &[
                    "pub(crate) fn sae_row_jet_program_matches_production_row_jets_on_converged_cache()",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/gpu_kernels/sae_rowjet.rs",
                anchors: &[
                    "fn complete_cpu_rowjet_contains_coordinate_mixed_and_beta_channels_2304()",
                    "fn memory_ledger_counts_coordinate_and_mixed_tensors_2304()",
                    "fn complete_device_matches_cpu_every_channel_when_admitted_2304()",
                ],
            },
        ],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "Gaussian location-scale",
        kind: DerivativeSpecializationKind::RowAtom,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/gamlss/gaussian/joint_psi.rs",
            anchors: &[
                "fn gaussian_normalized_row [generic, order2, third, fourth]",
                "pub struct GaussianJointRowProgram<'a>",
                "impl gam_math::jet_tower::RowProgram<2> for GaussianJointRowProgram<'_>",
                "fn row_order2(&self, row: usize) -> gam_math::jet_scalar::StaticOrder2Atom<2, 3, 3, 7>",
                "fn row_third_contracted(&self, row: usize, direction: &[f64; 2])",
                "fn row_fourth_contracted(",
                "pub(crate) fn gaussian_joint_psi_firstweights(",
                "pub(crate) fn gaussian_joint_psisecondweights(",
                "pub(crate) fn gaussian_joint_psi_mixed_driftweights(",
                "let atom = program.row_order2(i);",
                "let hessian_direction = program.row_third_contracted(i, &direction);",
                "let hessian_a_b = program.row_fourth_contracted(i, &direction_a, &direction_b);",
                "program.row_fourth_contracted(i, &drift, &psi)",
            ],
        }],
        discovery_anchor: "fn gaussian_normalized_row [generic, order2, third, fourth]",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/gamlss/gaussian/joint_psi.rs",
            anchors: &[
                "fn generated_gaussian_psi_chain_matches_generic_nested_jet_all_channels_932()",
                "fn generated_gaussian_psi_chain_matches_likelihood_finite_differences_932()",
                "fn first_directional_weights_match_jet_third()",
                "fn second_directional_weights_match_jet_fourth()",
                "crate::gamlss::GaussianJointRowProgram::new(&rows)",
            ],
        }],
        retired_identities: &[
            "cross_eta",
            "sea_seb",
            "sdea",
            "e_coef",
            "seab",
            "ma_mb",
            "de_ea",
            "kpi",
            "kdpi",
        ],
    },
    DerivativeSpecialization {
        family: "cause-specific survival",
        kind: DerivativeSpecializationKind::RowAtom,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/base.rs",
            anchors: &[
                "fn cause_specific_row [generic, order2, third, fourth]",
                "pub struct CauseSpecificRowProgram",
                "impl gam_math::jet_tower::RowProgram<3> for CauseSpecificRowProgram",
            ],
        }],
        discovery_anchor: "fn cause_specific_row [generic, order2, third, fourth]",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/base.rs",
            anchors: &[
                "fn cause_specific_live_tower_matches_jet_and_fd()",
                "crate::survival::CauseSpecificRowProgram::new(",
            ],
        }],
        retired_identities: &[],
    },
];

#[derive(Debug)]
struct DerivativeDeclaration {
    line_index: usize,
    source: String,
}

fn normalized_rust_fragment(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn code_anchor_line_indices(source: &str, anchor: &str) -> Vec<usize> {
    strip_file_lines(source)
        .into_iter()
        .enumerate()
        .filter_map(|(line_index, line)| line.contains(anchor).then_some(line_index))
        .collect()
}

fn code_identifier_line_indices(source: &str, identifier: &str) -> Vec<usize> {
    strip_file_lines(source)
        .into_iter()
        .enumerate()
        .filter_map(|(line_index, line)| {
            line.split(|character: char| character != '_' && !character.is_ascii_alphanumeric())
                .any(|token| token == identifier)
                .then_some(line_index)
        })
        .collect()
}

fn derivative_declarations(source: &str, test_mask: &[bool]) -> Vec<DerivativeDeclaration> {
    let lines = strip_file_lines(source);
    let mut declarations = Vec::new();
    let mut line_index = 0usize;
    while line_index < lines.len() {
        if test_mask.get(line_index).copied().unwrap_or(false) {
            line_index += 1;
            continue;
        }
        let trimmed = lines[line_index].trim();
        let starts_relevant_declaration = trimmed.starts_with("impl ")
            || trimmed.starts_with("impl<")
            || trimmed.starts_with("fn ");
        if !starts_relevant_declaration {
            line_index += 1;
            continue;
        }

        let start = line_index;
        let mut source = trimmed.to_string();
        while !source.contains('{') && !source.ends_with(';') && line_index + 1 < lines.len() {
            line_index += 1;
            if test_mask.get(line_index).copied().unwrap_or(false) {
                break;
            }
            source.push(' ');
            source.push_str(lines[line_index].trim());
        }
        declarations.push(DerivativeDeclaration {
            line_index: start,
            source: normalized_rust_fragment(&source),
        });
        line_index += 1;
    }
    declarations
}

fn implemented_trait_from_declaration(declaration: &str) -> Option<&str> {
    let Some(mut implemented) = declaration.strip_prefix("impl") else {
        return None;
    };
    implemented = implemented.trim_start();
    if implemented.starts_with('<') {
        let mut depth = 0usize;
        let mut generic_end = None;
        for (index, byte) in implemented.bytes().enumerate() {
            match byte {
                b'<' => depth += 1,
                b'>' => {
                    depth -= 1;
                    if depth == 0 {
                        generic_end = Some(index + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(generic_end) = generic_end else {
            return None;
        };
        implemented = implemented[generic_end..].trim_start();
    }
    let Some((implemented_trait, _)) = implemented.split_once(" for ") else {
        return None;
    };
    Some(implemented_trait.trim())
}

fn is_row_kernel_declaration(declaration: &str) -> bool {
    let Some(implemented_trait) = implemented_trait_from_declaration(declaration) else {
        return false;
    };
    implemented_trait.starts_with("RowKernel<") || implemented_trait.contains("::RowKernel<")
}

fn is_sae_softmax_row_program_source_declaration(declaration: &str) -> bool {
    let Some(implemented_trait) = implemented_trait_from_declaration(declaration) else {
        return false;
    };
    implemented_trait == "SaeSoftmaxRowProgramSource"
        || implemented_trait.ends_with("::SaeSoftmaxRowProgramSource")
}

fn generated_derivative_modes(declaration: &str) -> Option<(bool, bool)> {
    if !declaration.starts_with("fn ") {
        return None;
    }
    let mode_start = declaration.find('[')?;
    let mode_end = declaration[mode_start + 1..].find(']')? + mode_start + 1;
    let modes = declaration[mode_start + 1..mode_end]
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    let third = modes.contains(&"third");
    let fourth = modes.contains(&"fourth");
    (third || fourth).then_some((third, fourth))
}

fn enforce_derivative_registry_invariants() {
    for (index, specialization) in PRODUCTION_DERIVATIVE_SPECIALIZATIONS.iter().enumerate() {
        assert!(
            !specialization.production_sources.is_empty(),
            "#932 policy self-test: {} has no registered production source",
            specialization.family
        );
        assert!(
            !specialization.parity_pins.is_empty(),
            "#932 policy self-test: {} has no registered parity pin",
            specialization.family
        );
        let discovery_anchor = normalized_rust_fragment(specialization.discovery_anchor);
        assert!(
            specialization
                .production_sources
                .iter()
                .flat_map(|source| source.anchors.iter())
                .any(|anchor| normalized_rust_fragment(anchor).contains(&discovery_anchor)),
            "#932 policy self-test: {} discovery anchor is not owned by a production source",
            specialization.family
        );
        assert!(
            PRODUCTION_DERIVATIVE_SPECIALIZATIONS[index + 1..]
                .iter()
                .all(|other| other.family != specialization.family),
            "#932 policy self-test: duplicate specialization family {}",
            specialization.family
        );
    }
}

fn enforce_derivative_policy_negative_probes() {
    enforce_derivative_registry_invariants();

    let comment_only = "// impl RowKernel<7> for CommentOnlyKernel";
    assert!(
        code_anchor_line_indices(comment_only, "impl RowKernel<7> for CommentOnlyKernel")
            .is_empty(),
        "#932 policy self-test: a comment-only anchor was treated as production code"
    );

    let row_kernel = "impl RowKernel<7> for PlantedKernel {}";
    let row_mask = compute_test_mask(row_kernel, Path::new("crates/gam-models/src/planted.rs"));
    let row_declarations = derivative_declarations(row_kernel, &row_mask);
    assert!(
        row_declarations.iter().any(|declaration| {
            is_row_kernel_declaration(&declaration.source)
                && !specialization_site_is_registered(
                    DerivativeSpecializationKind::RowKernel,
                    "crates/gam-models/src/planted.rs",
                    &declaration.source,
                )
        }),
        "#932 policy self-test: an unregistered RowKernel was not discovered"
    );

    let registered_row_path = "crates/gam-models/src/survival/location_scale/row_kernel.rs";
    let registered_row_source =
        "impl crate::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_> {";
    let registered_row_mask =
        compute_test_mask(registered_row_source, Path::new(registered_row_path));
    assert!(
        derivative_declarations(registered_row_source, &registered_row_mask)
            .iter()
            .any(|declaration| {
                is_row_kernel_declaration(&declaration.source)
                    && specialization_site_is_registered(
                        DerivativeSpecializationKind::RowKernel,
                        registered_row_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: the exact registered survival RowKernel was not admitted"
    );
    let same_file_rogue_row =
        "impl crate::row_kernel::RowKernel<SLS_ROW_K> for PlantedSameFileKernel {";
    let same_file_rogue_row_mask =
        compute_test_mask(same_file_rogue_row, Path::new(registered_row_path));
    assert!(
        derivative_declarations(same_file_rogue_row, &same_file_rogue_row_mask)
            .iter()
            .any(|declaration| {
                is_row_kernel_declaration(&declaration.source)
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::RowKernel,
                        registered_row_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: a rogue RowKernel in the registered survival source was admitted"
    );

    let bounded_helper =
        "impl<const K: usize, T: RowKernel<K>> HyperOperator for PlantedWrapper<K, T> {}";
    let bounded_mask = compute_test_mask(
        bounded_helper,
        Path::new("crates/gam-models/src/planted.rs"),
    );
    assert!(
        derivative_declarations(bounded_helper, &bounded_mask)
            .iter()
            .all(|declaration| !is_row_kernel_declaration(&declaration.source)),
        "#932 policy self-test: a generic RowKernel bound was mistaken for a RowKernel implementation"
    );

    let separate_generated = "row_atom! {\n    fn planted_third [generic, third](x) { x }\n    fn planted_fourth [generic, fourth](x) { x }\n}";
    let generated_mask = compute_test_mask(
        separate_generated,
        Path::new("crates/gam-models/src/planted.rs"),
    );
    let generated_declarations = derivative_declarations(separate_generated, &generated_mask);
    let unregistered_generated = generated_declarations
        .iter()
        .filter(|declaration| {
            generated_derivative_modes(&declaration.source).is_some()
                && !specialization_site_is_registered(
                    DerivativeSpecializationKind::RowAtom,
                    "crates/gam-models/src/planted.rs",
                    &declaration.source,
                )
        })
        .count();
    assert_eq!(
        unregistered_generated, 2,
        "#932 policy self-test: separate generated-third/fourth declarations were not both discovered"
    );

    let registered_gaussian_path = "crates/gam-models/src/gamlss/gaussian/joint_psi.rs";
    let registered_gaussian_atom =
        "fn gaussian_normalized_row [generic, order2, third, fourth](delta_mu) { delta_mu }";
    let registered_gaussian_mask = compute_test_mask(
        registered_gaussian_atom,
        Path::new(registered_gaussian_path),
    );
    assert!(
        derivative_declarations(registered_gaussian_atom, &registered_gaussian_mask)
            .iter()
            .any(|declaration| {
                generated_derivative_modes(&declaration.source).is_some()
                    && specialization_site_is_registered(
                        DerivativeSpecializationKind::RowAtom,
                        registered_gaussian_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: the exact registered Gaussian row atom was not admitted"
    );
    let same_file_rogue_atom = "fn planted_gaussian_row [generic, order2, third, fourth](x) { x }";
    let same_file_rogue_atom_mask =
        compute_test_mask(same_file_rogue_atom, Path::new(registered_gaussian_path));
    assert!(
        derivative_declarations(same_file_rogue_atom, &same_file_rogue_atom_mask)
            .iter()
            .any(|declaration| {
                generated_derivative_modes(&declaration.source).is_some()
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::RowAtom,
                        registered_gaussian_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: a rogue row atom in the registered Gaussian source was admitted"
    );

    let sae_source = "impl SaeSoftmaxRowProgramSource for PlantedSaeSource {}";
    let sae_mask = compute_test_mask(sae_source, Path::new("crates/gam-sae/src/planted.rs"));
    let sae_declarations = derivative_declarations(sae_source, &sae_mask);
    assert!(
        sae_declarations.iter().any(|declaration| {
            is_sae_softmax_row_program_source_declaration(&declaration.source)
                && !specialization_site_is_registered(
                    DerivativeSpecializationKind::Bespoke,
                    "crates/gam-sae/src/planted.rs",
                    &declaration.source,
                )
        }),
        "#932 policy self-test: an unregistered SAE softmax row-program source was not discovered"
    );
    let sae_bound = "fn planted<S: SaeSoftmaxRowProgramSource>(source: &S) {}";
    let sae_bound_mask = compute_test_mask(sae_bound, Path::new("crates/gam-sae/src/planted.rs"));
    assert!(
        derivative_declarations(sae_bound, &sae_bound_mask)
            .iter()
            .all(|declaration| {
                !is_sae_softmax_row_program_source_declaration(&declaration.source)
            }),
        "#932 policy self-test: an SAE source bound was mistaken for a trait implementation"
    );

    let registered_sae_path = "crates/gam-sae/src/gpu_kernels/sae_rowjet.rs";
    let registered_sae_source = "impl SaeSoftmaxRowProgramSource for InputSource<'_> {";
    let registered_sae_mask =
        compute_test_mask(registered_sae_source, Path::new(registered_sae_path));
    assert!(
        derivative_declarations(registered_sae_source, &registered_sae_mask)
            .iter()
            .any(|declaration| {
                is_sae_softmax_row_program_source_declaration(&declaration.source)
                    && specialization_site_is_registered(
                        DerivativeSpecializationKind::Bespoke,
                        registered_sae_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: the exact registered SAE implementation was not admitted"
    );
    let same_file_rogue = "impl SaeSoftmaxRowProgramSource for PlantedSameFileSource {}";
    let same_file_rogue_mask = compute_test_mask(same_file_rogue, Path::new(registered_sae_path));
    assert!(
        derivative_declarations(same_file_rogue, &same_file_rogue_mask)
            .iter()
            .any(|declaration| {
                is_sae_softmax_row_program_source_declaration(&declaration.source)
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::Bespoke,
                        registered_sae_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: a rogue SAE implementation in a registered source file was admitted"
    );

    let retired = "fn planted_retired_identity() {}";
    assert_eq!(
        code_identifier_line_indices(retired, "planted_retired_identity"),
        vec![0],
        "#932 policy self-test: a retired derivative identity was not discovered"
    );
    assert!(
        code_identifier_line_indices(
            "// planted_retired_identity\nfn planted_retired_identity_suffix() {}",
            "planted_retired_identity",
        )
        .is_empty(),
        "#932 policy self-test: retired-identity matching ignored token boundaries or comments"
    );
}

fn enforce_production_derivative_specializations(root: &Path) {
    enforce_derivative_policy_negative_probes();
    let mut violations = Vec::new();
    for specialization in PRODUCTION_DERIVATIVE_SPECIALIZATIONS {
        for production in specialization.production_sources {
            let production_path = root.join(production.path);
            match fs::read_to_string(&production_path) {
                Ok(source) => {
                    let test_mask = compute_test_mask(&source, Path::new(production.path));
                    for anchor in production.anchors {
                        let anchor_lines = code_anchor_line_indices(&source, anchor);
                        if anchor_lines.is_empty() {
                            violations.push(format!(
                                "{} production anchor is missing from {}: {}",
                                specialization.family, production.path, anchor
                            ));
                        } else if anchor_lines
                            .iter()
                            .all(|line| test_mask.get(*line).copied().unwrap_or(false))
                        {
                            violations.push(format!(
                                "{} production anchor is gated by cfg(test) in {}: {}",
                                specialization.family, production.path, anchor
                            ));
                        }
                    }
                }
                Err(error) => violations.push(format!(
                    "{} production source {} cannot be read: {error}",
                    specialization.family, production.path
                )),
            }
        }

        for pin in specialization.parity_pins {
            let pin_path = root.join(pin.path);
            match fs::read_to_string(&pin_path) {
                Ok(source) => {
                    for anchor in pin.anchors {
                        if code_anchor_line_indices(&source, anchor).is_empty() {
                            violations.push(format!(
                                "{} registered parity pin is missing from {}: {}",
                                specialization.family, pin.path, anchor
                            ));
                        }
                    }
                }
                Err(error) => violations.push(format!(
                    "{} pin source {} cannot be read: {error}",
                    specialization.family, pin.path
                )),
            }
        }
    }

    visit_files(
        root,
        &root.join("crates/gam-models/src"),
        &mut |rel, content| {
            let rel_path = rel.to_string_lossy().replace('\\', "/");
            if rel.extension().and_then(OsStr::to_str) != Some("rs") {
                return;
            }
            for specialization in PRODUCTION_DERIVATIVE_SPECIALIZATIONS {
                for identifier in specialization.retired_identities {
                    for line_index in code_identifier_line_indices(content, identifier) {
                        violations.push(format!(
                            "{} retired derivative identity reappeared at {rel_path}:{}: {}",
                            specialization.family,
                            line_index + 1,
                            identifier
                        ));
                    }
                }
            }
            let test_mask = compute_test_mask(content, rel);
            for declaration in derivative_declarations(content, &test_mask) {
                if is_row_kernel_declaration(&declaration.source)
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::RowKernel,
                        &rel_path,
                        &declaration.source,
                    )
                {
                    violations.push(format!(
                        "unregistered production RowKernel specialization at {rel_path}:{}: {}",
                        declaration.line_index + 1,
                        declaration.source
                    ));
                }

                if generated_derivative_modes(&declaration.source).is_some()
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::RowAtom,
                        &rel_path,
                        &declaration.source,
                    )
                {
                    violations.push(format!(
                        "unregistered generated third/fourth row specialization at {rel_path}:{}: {}",
                        declaration.line_index + 1,
                        declaration.source
                    ));
                }
            }
        },
    );

    visit_files(
        root,
        &root.join("crates/gam-sae/src"),
        &mut |rel, content| {
            let rel_path = rel.to_string_lossy().replace('\\', "/");
            if rel.extension().and_then(OsStr::to_str) != Some("rs") {
                return;
            }
            let test_mask = compute_test_mask(content, rel);
            for declaration in derivative_declarations(content, &test_mask) {
                if is_sae_softmax_row_program_source_declaration(&declaration.source)
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::Bespoke,
                        &rel_path,
                        &declaration.source,
                    )
                {
                    violations.push(format!(
                        "unregistered production SAE softmax row-program source at {rel_path}:{}: {}",
                        declaration.line_index + 1,
                        declaration.source
                    ));
                }
            }
        },
    );

    if violations.is_empty() {
        return;
    }
    for violation in &violations {
        println!("cargo:warning=#932 derivative-specialization policy: {violation}");
    }
    panic!(
        "#932 derivative-specialization registry rejected {} violation(s)",
        violations.len()
    );
}

fn specialization_site_is_registered(
    kind: DerivativeSpecializationKind,
    path: &str,
    source_line: &str,
) -> bool {
    let normalized_source = normalized_rust_fragment(source_line);
    PRODUCTION_DERIVATIVE_SPECIALIZATIONS
        .iter()
        .any(|specialization| {
            specialization.kind == kind
                && specialization
                    .production_sources
                    .iter()
                    .filter(|source| source.path == path)
                    .flat_map(|source| source.anchors.iter())
                    .any(|anchor| registered_declaration_matches_anchor(&normalized_source, anchor))
        })
}

fn registered_declaration_matches_anchor(declaration: &str, anchor: &str) -> bool {
    let anchor = normalized_rust_fragment(anchor);
    if !(anchor.starts_with("impl ") || anchor.starts_with("impl<") || anchor.starts_with("fn ")) {
        return false;
    }
    let Some(remainder) = declaration.strip_prefix(&anchor) else {
        return false;
    };
    if remainder.is_empty() {
        return true;
    }
    if anchor.ends_with('(') || remainder.starts_with('(') {
        return true;
    }
    if !remainder.chars().next().is_some_and(char::is_whitespace) {
        return false;
    }
    let declaration_tail = remainder.trim_start();
    declaration_tail.starts_with('{') || declaration_tail.starts_with("where ")
}

fn visit_files(root: &Path, dir: &Path, visitor: &mut dyn FnMut(&Path, &str)) {
    let dir_rel = dir
        .strip_prefix(root)
        .expect("scanner directory must be under the manifest root");
    for file in scannable_files(root) {
        if !dir_rel.as_os_str().is_empty() && !file.rel.starts_with(dir_rel) {
            continue;
        }
        visitor(&file.rel, &file.content);
    }
}

fn scannable_files(root: &Path) -> &'static [ScannedFile] {
    SCANNABLE_FILES
        .get_or_init(|| {
            let mut files = Vec::new();
            collect_scannable_files(root, root, &mut files);
            files
        })
        .as_slice()
}

fn collect_scannable_files(root: &Path, dir: &Path, files: &mut Vec<ScannedFile>) {
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
        if (name.starts_with('.') && name != ".github")
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
            collect_scannable_files(root, &path, files);
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
        let rel = path
            .strip_prefix(root)
            .expect("scanned file must be under the manifest root")
            .to_path_buf();
        files.push(ScannedFile { rel, content });
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
            let (sig_start, sig_end_line, sig_end_col_excl) =
                match find_fn_body_at_stripped(&stripped_lines, idx) {
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
            let Some(paren_open) = find_fn_param_open(&sig_text) else {
                idx = sig_end_line + 1;
                continue;
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
                if name == "_" {
                    continue;
                }
                if name.starts_with('_') {
                    // PyO3 convention: `_py: Python<'_>` (or `Python<'py>`) is
                    // the GIL token that #[pyfunction] / #[pymethods] macros
                    // thread through every entry point. The body of helpers
                    // called from those entry points often doesn't touch it,
                    // but removing the parameter forces the call site to drop
                    // it too, which breaks the macro contract. Exempt it.
                    let type_after_colon = rest[1..].trim_start();
                    if type_after_colon.starts_with("Python<") {
                        continue;
                    }
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

/// Find the actual function-parameter opener in a stripped Rust signature.
/// Starts after `fn <name>` so visibility qualifiers such as `pub(crate)`,
/// `pub(super)`, and `pub(in crate::x)` cannot be mistaken for the parameter
/// list. Generic bounds before the parameter list are skipped at angle-depth.
fn find_fn_param_open(sig_text: &str) -> Option<usize> {
    let sig_bytes = sig_text.as_bytes();
    let fn_pos = locate_fn_keyword(sig_text)?;
    let mut i = fn_pos + 2;
    while i < sig_bytes.len() && sig_bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i + 1 < sig_bytes.len() && sig_bytes[i] == b'r' && sig_bytes[i + 1] == b'#' {
        i += 2;
    }
    while i < sig_bytes.len() && is_ident_byte(sig_bytes[i]) {
        i += 1;
    }

    let mut angle: i32 = 0;
    while i < sig_bytes.len() {
        match sig_bytes[i] {
            b'<' => angle += 1,
            b'>' => {
                if angle > 0 {
                    angle -= 1;
                }
            }
            b'(' if angle == 0 => return Some(i),
            b'{' | b';' if angle == 0 => return None,
            _ => {}
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
        // Index every local `fn` so a `#[test]` that delegates its assertions
        // to a helper can be resolved by following the call into the helper
        // body (see `test_body_reaches_assertion`), instead of relying solely
        // on the helper's *name* matching an `assert_*`/`expect_*`/… prefix.
        let local_fns = index_local_fns(&lines, &stripped_lines);
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
            let Some((_sig, (open, close))) = find_fn_body_at_stripped(&stripped_lines, j) else {
                i = j + 1;
                continue;
            };
            let found =
                test_body_reaches_assertion(&lines, &stripped_lines, open, close, &local_fns);
            if !found {
                let raw = lines.get(i).copied().unwrap_or("");
                if rel
                    .to_string_lossy()
                    .contains("pyffi_fitted_family_roundtrip_bug")
                {
                    eprintln!(
                        "DEBUG useless_test: file={} open={} close={}",
                        rel.display(),
                        open,
                        close
                    );
                    for k in open..=close {
                        let raw_l = lines.get(k).copied().unwrap_or("");
                        let strp = stripped_lines.get(k).map(String::as_str).unwrap_or(raw_l);
                        eprintln!("  L{}: RAW=[{}]", k + 1, raw_l);
                        eprintln!("       STR=[{}]", strp);
                    }
                }
                offenders.push((rel.to_path_buf(), i + 1, raw.to_string()));
            }
            i = close + 1;
        }
    });
}

/// True when `stripped` contains a macro call (`<ident>!(...)`) whose name
/// matches a convention indicating it asserts an invariant: `assert_*`,
/// `expect_*`, `require_*`, `ensure_*`. Lets test bodies delegate to local
/// helper macros (e.g. `expect_invalid_input!(...)`) without losing the
/// "this test has assertions" recognition.
fn line_contains_assertion_helper_macro(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if bytes[i] != b'!' {
            i += 1;
            continue;
        }
        // The byte after `!` should be `(` (or `[` / `{`).
        let next = bytes.get(i + 1).copied();
        if !matches!(next, Some(b'(') | Some(b'[') | Some(b'{')) {
            i += 1;
            continue;
        }
        // Walk back from `!` collecting ASCII ident bytes.
        let mut start = i;
        while start > 0 {
            let b = bytes[start - 1];
            if b == b'_' || b.is_ascii_alphanumeric() {
                start -= 1;
            } else {
                break;
            }
        }
        if start == i {
            i += 1;
            continue;
        }
        let name = &stripped[start..i];
        if name.starts_with("assert_")
            || name.starts_with("expect_")
            || name.starts_with("require_")
            || name.starts_with("ensure_")
        {
            return true;
        }
        i += 1;
    }
    false
}

/// True when `stripped` contains a bare function call (`<ident>(...)` —
/// no trailing `!`) whose name matches the same `assert_*` / `expect_*` /
/// `require_*` / `ensure_*` convention as the helper-macro detector. Lets
/// test bodies delegate to local helper *functions* (e.g.
/// `assert_second_jet_matches_central_difference(&evaluator, coords, tol)`)
/// without losing the "this test has assertions" recognition. The macro
/// shape is already covered by `line_contains_assertion_helper_macro`; this
/// extension matches the function-call shape with the same naming policy.
fn line_contains_assertion_helper_call(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if bytes[i] != b'(' {
            i += 1;
            continue;
        }
        // Skip the macro-call shape `<ident>!(` — handled by the helper-macro
        // detector. We only want the bare-call shape here.
        if i > 0 && bytes[i - 1] == b'!' {
            i += 1;
            continue;
        }
        // Walk back from `(` collecting ASCII ident bytes.
        let mut start = i;
        while start > 0 {
            let b = bytes[start - 1];
            if b == b'_' || b.is_ascii_alphanumeric() {
                start -= 1;
            } else {
                break;
            }
        }
        if start == i {
            i += 1;
            continue;
        }
        let name = &stripped[start..i];
        if name.starts_with("assert_")
            || name.starts_with("expect_")
            || name.starts_with("require_")
            || name.starts_with("ensure_")
        {
            return true;
        }
        i += 1;
    }
    false
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

/// The marker macros tracked by the audit. The first three are the runtime
/// "not done yet" panic family (covered by the lexical bans for non-test
/// code). `panic!(` is included so that any SAFETY-justified panic in
/// production code, once removed, must be replaced by a substantive body —
/// not silently downgraded to `Ok(())` / a generic `Err(...)` stub that
/// returns success on the previously-rejected input.
const HISTORY_MARKERS: &[&str] = &["unimplemented!(", "todo!(", "unreachable!(", "panic!("];

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

fn collect_git_history_marker_functions(
    manifest_dir: &Path,
) -> std::collections::BTreeMap<(String, String), Vec<String>> {
    let mut out: std::collections::BTreeMap<(String, String), Vec<String>> =
        std::collections::BTreeMap::new();
    let git_contents = git_head_files_containing(manifest_dir, HISTORY_MARKERS);
    for (rel_str, content) in git_contents {
        if rel_str == "build.rs" || !rel_str.ends_with(".rs") {
            continue;
        }
        let rel = Path::new(&rel_str);
        let lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(&content);
        let mask = compute_test_mask(&content, rel);
        for (idx, line) in lines.iter().enumerate() {
            if mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let stripped = strip_strings_and_comments(line);
            for &marker in HISTORY_MARKERS {
                if !stripped.contains(marker) {
                    continue;
                }
                if let Some((sig, _)) = find_enclosing_fn(&lines, &stripped_lines, idx) {
                    let kind = marker.trim_end_matches('(').to_string();
                    let entry = out.entry((rel_str.clone(), sig)).or_default();
                    if !entry.contains(&kind) {
                        entry.push(kind);
                    }
                    break;
                }
            }
        }
    }
    out
}

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
        let stripped_lines = strip_file_lines(content);
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
                if let Some((sig, (open, _close))) = find_enclosing_fn(&lines, &stripped_lines, idx)
                {
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

    // Step 2: load the previous ledger (empty on first run) and augment it
    // with marker-bearing functions from git HEAD. The on-disk ledger catches
    // sites seen by prior local builds; git catches sites that existed in the
    // committed tree even if the ledger was stale or a marker was deleted
    // before build.rs had a chance to record it.
    let mut previous = load_history_ledger(&ledger_path);
    for (key, kinds) in collect_git_history_marker_functions(manifest_dir) {
        previous.entry(key).or_insert(kinds);
    }

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
    let stripped_lines = strip_file_lines(content);
    let mut idx = 0;
    while idx < lines.len() {
        let stripped = stripped_lines
            .get(idx)
            .map(String::as_str)
            .unwrap_or(lines[idx]);
        if !line_has_keyword(&stripped, "fn") {
            idx += 1;
            continue;
        }
        if let Some((sig, (open, close))) = find_fn_body_at_stripped(&stripped_lines, idx) {
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
fn find_enclosing_fn(
    lines: &[&str],
    stripped_lines: &[String],
    at_line: usize,
) -> Option<(String, (usize, usize))> {
    let mut start = at_line + 1;
    while start > 0 {
        start -= 1;
        let stripped = stripped_lines
            .get(start)
            .map(String::as_str)
            .unwrap_or(lines[start]);
        if !line_has_keyword(&stripped, "fn") {
            continue;
        }
        if let Some((sig, (open, close))) = find_fn_body_at_stripped(stripped_lines, start)
            && open <= at_line
            && at_line <= close
        {
            return Some((sig, (open, close)));
        }
    }
    None
}

fn find_fn_body_at_stripped(
    stripped_lines: &[String],
    fn_line: usize,
) -> Option<(String, (usize, usize))> {
    let mut depth: i32 = 0;
    let mut body_open: Option<usize> = None;
    for (j, s) in stripped_lines.iter().enumerate().skip(fn_line) {
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
                        for (k, ss) in stripped_lines
                            .iter()
                            .enumerate()
                            .take(open + 1)
                            .skip(fn_line)
                        {
                            let cut = if k == open {
                                ss.find('{').unwrap_or(ss.len())
                            } else {
                                ss.len()
                            };
                            sig.push_str(&ss[..cut]);
                            sig.push(' ');
                        }
                        let normalized = sig.split_whitespace().collect::<Vec<_>>().join(" ");
                        return Some((normalized, (open, j)));
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Maximum number of helper-delegation hops the useless-test gate will follow
/// when resolving whether a `#[test]` reaches an assertion. A test that calls a
/// helper that calls a helper that asserts is still a real test; beyond a few
/// hops the indirection is its own problem and we stop (and flag) rather than
/// chase arbitrarily deep call graphs at build time.
const MAX_DELEGATION_DEPTH: usize = 4;

/// True when a single stripped source line contains an assertion-shaped
/// construct: an `assert`/`debug_assert`/panic-family macro, `?`-propagation,
/// or an `assert_*`/`expect_*`/`require_*`/`ensure_*`-named helper macro or
/// bare call. This is the per-line recognizer for the useless-test gate.
fn line_is_assertion_shaped(line_s: &str) -> bool {
    line_s.contains("assert!(")
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
        || line_contains_assertion_helper_macro(line_s)
        || line_contains_assertion_helper_call(line_s)
}

/// Extract the function name from a normalized signature string produced by
/// `find_fn_body_at` (whitespace-collapsed, body brace stripped), e.g.
/// `"pub fn report_and_check ( ... ) -> bool"` → `Some("report_and_check")`.
/// Returns `None` when the `fn` token is a function-pointer *type*
/// (`fn(f64) -> f64`) with no following identifier.
fn fn_name_from_sig(sig: &str) -> Option<String> {
    let mut toks = sig.split_whitespace();
    while let Some(t) = toks.next() {
        if t == "fn" {
            let next = toks.next()?;
            let name: String = next
                .chars()
                .take_while(|c| *c == '_' || c.is_ascii_alphanumeric())
                .collect();
            return if name.is_empty() { None } else { Some(name) };
        }
    }
    None
}

/// Index every `fn <name>(...) { ... }` defined in the file as
/// `(name, body_open_line, body_close_line)`. Nested and duplicate-named
/// definitions are all retained so any matching definition can satisfy a
/// delegation lookup. Function-pointer *types* (no name) are skipped.
fn index_local_fns(lines: &[&str], stripped_lines: &[String]) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let n = lines.len();
    let mut i = 0usize;
    while i < n {
        let s = stripped_lines
            .get(i)
            .map(String::as_str)
            .unwrap_or(lines[i]);
        if line_has_keyword(s, "fn")
            && let Some((sig, (open, close))) = find_fn_body_at_stripped(stripped_lines, i)
            && let Some(name) = fn_name_from_sig(&sig)
        {
            out.push((name, open, close));
        }
        i += 1;
    }
    out
}

/// Collect identifiers appearing in bare-call position (`<ident>(`, not the
/// macro shape `<ident>!(`) on a stripped source line, appending them to
/// `out`. Used to resolve which local helper functions a test body delegates
/// to.
fn collect_called_idents(stripped: &str, out: &mut Vec<String>) {
    let bytes = stripped.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if bytes[i] != b'(' {
            i += 1;
            continue;
        }
        if i > 0 && bytes[i - 1] == b'!' {
            i += 1;
            continue;
        }
        // The callee identifier ends just before the `(` — unless a turbofish
        // (`name::<…>(`) sits between them. A generic-helper delegation such as
        // `check_bit_identical::<2>(seed, n)` puts the turbofish-closing `>`
        // immediately before the `(`, so step back over the balanced `::<…>`
        // segment to reach the real callee identifier. Without this, every
        // `#[test]` whose only assertions live in a generic helper it calls
        // through a turbofish reads as assertion-less.
        let mut id_end = i;
        if i > 0 && bytes[i - 1] == b'>' {
            let mut depth = 0i32;
            let mut p = i - 1;
            let mut turbofish_open = None;
            loop {
                match bytes[p] {
                    b'>' => depth += 1,
                    b'<' => {
                        depth -= 1;
                        if depth == 0 {
                            turbofish_open = Some(p);
                            break;
                        }
                    }
                    _ => {}
                }
                if p == 0 {
                    break;
                }
                p -= 1;
            }
            if let Some(lt) = turbofish_open
                && lt >= 2
                && bytes[lt - 1] == b':'
                && bytes[lt - 2] == b':'
            {
                id_end = lt - 2;
            }
        }
        let mut start = id_end;
        while start > 0 {
            let b = bytes[start - 1];
            if b == b'_' || b.is_ascii_alphanumeric() {
                start -= 1;
            } else {
                break;
            }
        }
        if start < id_end {
            out.push(stripped[start..id_end].to_string());
        }
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cosmetic-wording-dodge history audit.
//
// Generalizes the "marker removed but code unchanged" idea (see
// `removed_todo_site_violation`) from the bare TODO marker to OWED-WORK /
// DEFERRAL prose. The owner's rule: "ANY wording change made to satisfy
// build.rs is ALWAYS a bad hack." If a commit removes or rewords an owed-work
// note that `scan_for_owed_work_prose` would ban, but the comment-stripped CODE
// of that file is byte-identical before and after, then the owed work was NOT
// done — it was laundered into different wording to dodge the scanner. The
// canonical example is commit 6d6b529e1, which reworded the banned owed-work
// phrasing (a "not-yet-"-style deferral note) across ten files with ZERO
// implementation lines.

/// The exact owed-work / deferral prose fragments that `scan_for_owed_work_prose`
/// bans, assembled from pieces at runtime so this build script never carries the
/// banned phrasing verbatim (which would self-trip the lexical scanner and this
/// very audit). Kept as a standalone helper so the removal-signal here keys on
/// precisely the same phrase set the prose ban enforces — removing one of these
/// notes is what this audit treats as an owed-work signal disappearing.
/// True when `line` is a `//` line-comment carrying an owed-work / deferral
/// signal. Uses the same `line_comment_text` reader and the same
/// `comment_text_is_owed_work` classifier as `scan_for_owed_work_prose`, so the
/// diff check and the HEAD-state ban agree exactly on what counts as owed work —
/// URLs/`://` never match, and a bare scope statement counts only with a temporal
/// cue (see `comment_text_is_owed_work`).
fn cosmetic_dodge_line_is_owed_signal(line: &str) -> bool {
    let Some(text) = line_comment_text(line) else {
        return false;
    };
    comment_text_is_owed_work(&text)
}

/// Collect the non-test `src/*.rs` files a single commit touched. A merge commit
/// (multiple parents) is skipped — its first-parent diff replays the merged
/// branch's already-audited content and would re-flag an already-landed change.
fn cosmetic_dodge_commit_src_files(manifest_dir: &Path, sha: &str) -> Vec<String> {
    let parents = match Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("rev-list")
        .arg("--parents")
        .arg("-n1")
        .arg(sha)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let parents_text = String::from_utf8_lossy(&parents.stdout);
    // Tokens are: <commit> <parent1> [<parent2> ...]. More than one parent ⇒ merge.
    let token_count = parents_text.split_whitespace().count();
    if token_count != 2 {
        return Vec::new();
    }

    let output = match Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("diff")
        .arg("--name-only")
        .arg(format!("{sha}^"))
        .arg(sha)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let mut out = Vec::new();
    for rel in text.lines() {
        let rel = rel.trim();
        if rel.is_empty() {
            continue;
        }
        let rel_norm = rel.replace('\\', "/");
        if !rel_norm.starts_with("src/") || !rel_norm.ends_with(".rs") {
            continue;
        }
        let rel_path = Path::new(rel);
        if rel_path_is_skipped_for_scans(rel_path) || !rel_path_is_scannable(rel_path) {
            continue;
        }
        out.push(rel.to_string());
    }
    out
}

/// Read a single tree blob (`git show <treeish>:<path>`). Returns `None` when the
/// path did not exist at that revision (e.g. added/deleted in the commit).
fn cosmetic_dodge_blob_at(manifest_dir: &Path, treeish: &str, rel: &str) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("show")
        .arg(format!("{treeish}:{rel}"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

/// History audit: flag any recent commit that REMOVED or REWORDED an owed-work /
/// deferral note (per `comment_text_is_owed_work`) from a non-test
/// `src/*.rs` file WHILE that dodge STILL STANDS in the current source — i.e. the
/// signal is still gone AND the comment-stripped code is still byte-identical to
/// the pre-laundering blob. That is the cosmetic wording dodge: the note that
/// recorded owed work was removed, no work has since landed, so the work itself
/// was never done — only the wording was laundered to satisfy the prose ban.
///
/// The verdict keeps the ORIGINAL historical detection (the commit `sha`
/// removed an owed-work signal while changing no code) and ADDS a "still stands
/// at HEAD" requirement evaluated against the working tree. This matters because
/// the historical-only verdict was UN-REMEDIABLE: the frozen `sha^..sha` diff
/// cannot be altered by any forward commit, and restoring the honest note
/// re-trips the live prose ban, so a single past reword would abort every build
/// for the entire history window with no legitimate escape. Requiring the dodge
/// to still stand in the CURRENT tree restores the advertised remediations
/// without relaxing what counts as laundering: doing the real work (which
/// changes the comment-stripped code) or restoring the honest note retires the
/// flag, while an un-remediated laundering remains flagged exactly as before.
///
/// FALSE-POSITIVE control. A commit is flagged ONLY when ALL hold:
///   (a)  a removed `//`-comment line carried an owed-work phrase that is gone
///        from the commit's own after-blob (historical: marks `sha` a laundering
///        commit, not a reorder), AND that signal is still gone from the CURRENT
///        source (a later restore clears it); and
///   (b)  `normalized_file_code_without_comments` is byte-identical across the
///        pre-laundering blob, the commit's after-blob, AND the current tree —
///        i.e. neither the commit nor any later work landed real code.
/// Honest doc edits that don't touch an owed-work phrase never satisfy (a); any
/// commit (or any later commit) that lands real code in the file never satisfies
/// (b). All arms must fire, so an ordinary documentation improvement and a
/// genuine repair are both immune.
fn run_cosmetic_wording_dodge_audit(
    manifest_dir: &Path,
    violations: &mut Vec<(PathBuf, usize, String, String)>,
) {
    let log = match Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("log")
        .arg(format!("-n{TO_DO_HISTORY_GIT_DEPTH}"))
        .arg("--no-merges")
        .arg("--format=%H")
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let log_text = match String::from_utf8(log.stdout) {
        Ok(t) => t,
        Err(_) => return,
    };

    for sha in log_text.lines() {
        let sha = sha.trim();
        if sha.is_empty() {
            continue;
        }
        for rel in cosmetic_dodge_commit_src_files(manifest_dir, sha) {
            // The hunks this commit applied to `rel` (no context, so only
            // genuinely added/removed lines are present).
            let diff = match Command::new("git")
                .arg("-C")
                .arg(manifest_dir)
                .arg("diff")
                .arg("-U0")
                .arg(format!("{sha}^"))
                .arg(sha)
                .arg("--")
                .arg(&rel)
                .output()
            {
                Ok(o) if o.status.success() => o,
                _ => continue,
            };
            let diff_text = String::from_utf8_lossy(&diff.stdout);
            let mut removed_signal_lines: Vec<String> = Vec::new();
            for line in diff_text.lines() {
                if line.starts_with("---") || line.starts_with("+++") {
                    continue;
                }
                if let Some(body) = line.strip_prefix('-')
                    && cosmetic_dodge_line_is_owed_signal(body)
                {
                    removed_signal_lines.push(body.to_string());
                }
            }
            if removed_signal_lines.is_empty() {
                continue;
            }

            let Some(before) = cosmetic_dodge_blob_at(manifest_dir, &format!("{sha}^"), &rel)
            else {
                continue;
            };
            let Some(after) = cosmetic_dodge_blob_at(manifest_dir, sha, &rel) else {
                continue;
            };
            // HEAD-anchor: also read the CURRENT working-tree file (the source
            // actually being compiled). A laundering commit is a LIVE violation
            // only while its dodge STILL STANDS in the current tree. Without this
            // the gate was UN-REMEDIABLE: the frozen `sha^..sha` diff cannot be
            // altered by any forward commit, and restoring the honest note
            // re-trips the live prose ban — so a single past reword aborted
            // EVERY build for the full history window with no legitimate escape.
            // The current-tree arms below restore the advertised remediations
            // (do the real work, or restore the note) WITHOUT relaxing what the
            // historical arms count as laundering. A missing file (renamed or
            // deleted since) is itself real movement, so it clears.
            let Some(current) = std::fs::read_to_string(manifest_dir.join(&rel)).ok() else {
                continue;
            };

            // Trimmed owed-signal lines present in a blob, for "is the signal
            // still there?" comparisons that ignore pure re-indentation.
            let signal_lines = |blob: &str| -> Vec<String> {
                blob.lines()
                    .filter(|l| cosmetic_dodge_line_is_owed_signal(l))
                    .map(|l| l.trim().to_string())
                    .collect::<Vec<_>>()
            };
            let signal_absent_in = |blob: &str| -> bool {
                let present = signal_lines(blob);
                removed_signal_lines
                    .iter()
                    .any(|r| !present.iter().any(|a| a == r.trim()))
            };

            // (a) HISTORICAL: the commit removed the signal from its own
            // after-blob (unchanged from the original audit — this is what makes
            // `sha` a laundering commit rather than a reorder), AND
            // (a') CURRENT: the signal is still gone from the working tree (a
            // later restore of the honest note clears the dodge).
            if !signal_absent_in(&after) || !signal_absent_in(&current) {
                continue;
            }

            // (b) HISTORICAL: the commit itself changed no code (unchanged from
            // the original audit), AND
            // (b') CURRENT: no code has landed in the file since either, so the
            // owed item is genuinely still undone. Any intervening real code
            // work — the advertised "do the real work" remediation — clears this.
            let code_before = normalized_file_code_without_comments(&before);
            if code_before != normalized_file_code_without_comments(&after)
                || code_before != normalized_file_code_without_comments(&current)
            {
                continue;
            }

            let reason = format!(
                "cosmetic wording dodge: an owed-work/deferral note was reworded or deleted but \
                 no code changed (file {rel}, commit {short}) — the work was NOT done, only the \
                 wording. Do the real work or restore the honest note; rewording to satisfy \
                 build.rs is banned.",
                short = &sha[..sha.len().min(9)],
            );
            violations.push((
                PathBuf::from(&rel),
                0,
                reason,
                removed_signal_lines
                    .first()
                    .map(|l| l.trim().to_string())
                    .unwrap_or_default(),
            ));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared git-backed history helpers.

fn git_head_files_containing(
    manifest_dir: &Path,
    needles: &[&str],
) -> std::collections::BTreeMap<String, String> {
    if needles.is_empty() {
        return std::collections::BTreeMap::new();
    }
    let mut cmd = Command::new("git");
    cmd.arg("-C")
        .arg(manifest_dir)
        .arg("grep")
        .arg("-z")
        .arg("-l");
    for needle in needles {
        cmd.arg("-e").arg(needle);
    }
    cmd.arg("HEAD").arg("--");

    let output = match cmd.output() {
        Ok(output) => output,
        Err(_) => return std::collections::BTreeMap::new(),
    };
    if !output.status.success() {
        return std::collections::BTreeMap::new();
    }

    let mut rels = Vec::new();
    for raw in output.stdout.split(|b| *b == 0) {
        if raw.is_empty() {
            continue;
        }
        let spec = match String::from_utf8(raw.to_vec()) {
            Ok(spec) => spec,
            Err(_) => continue,
        };
        let Some(rel) = spec.strip_prefix("HEAD:") else {
            continue;
        };
        let rel_path = Path::new(rel);
        if rel_path_is_skipped_for_scans(rel_path) || !rel_path_is_scannable(rel_path) {
            continue;
        }
        rels.push(rel.to_string());
    }
    git_show_head_files(manifest_dir, &rels)
}

fn git_show_head_files(
    manifest_dir: &Path,
    rels: &[String],
) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    if rels.is_empty() {
        return out;
    }
    let mut child = match Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("cat-file")
        .arg("--batch")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return out,
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("git cat-file stdin must be piped");
        for rel in rels {
            writeln!(stdin, "HEAD:{rel}").expect("failed to write git cat-file query");
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(_) => return out,
    };
    if !output.status.success() {
        return out;
    }
    let bytes = output.stdout;
    let mut pos = 0usize;
    for rel in rels {
        let Some(header_len) = bytes[pos..].iter().position(|b| *b == b'\n') else {
            break;
        };
        let header_end = pos + header_len;
        let header = match std::str::from_utf8(&bytes[pos..header_end]) {
            Ok(header) => header,
            Err(_) => break,
        };
        let Some(size_text) = header.split_whitespace().nth(2) else {
            break;
        };
        let Ok(size) = size_text.parse::<usize>() else {
            break;
        };
        let content_start = header_end + 1;
        let content_end = content_start + size;
        if content_end > bytes.len() {
            break;
        }
        if let Ok(content) = String::from_utf8(bytes[content_start..content_end].to_vec()) {
            out.insert(rel.clone(), content);
        }
        pos = content_end;
        if pos < bytes.len() && bytes[pos] == b'\n' {
            pos += 1;
        }
    }
    out
}

fn rel_path_is_skipped_for_scans(rel: &Path) -> bool {
    if rel.starts_with("bench/runtime/pydeps") {
        return true;
    }
    for component in rel.components() {
        let name = component.as_os_str().to_string_lossy();
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
            return true;
        }
    }
    false
}

fn rel_path_is_scannable(rel: &Path) -> bool {
    let ext = rel.extension().and_then(OsStr::to_str).unwrap_or("");
    let basename = rel.file_name().and_then(OsStr::to_str).unwrap_or("");
    matches!(
        ext,
        "rs" | "py" | "toml" | "yml" | "yaml" | "sh" | "bash" | "json"
    ) || basename == "build.rs"
        || basename == "Makefile"
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent TO-DO marker/comment removal audit.

const TO_DO_HISTORY_LEDGER_FILENAME: &str = "todo_history.txt";
const FN_ANCHOR_PREFIX: &str = "fn ";
const FILE_ANCHOR: &str = "file";
/// Minimum normalized length of a marker's trailing description for the
/// rename-evasion check to key on it. Below this a short, generic tail could
/// collide with unrelated prose and false-positive.
const MIN_REWORD_DESCRIPTION_LEN: usize = 24;
/// How many recent commits the rename-evasion history walk inspects. Bounds the
/// cost of the `git log -G` shell-out while covering realistic relabel windows.
const TO_DO_HISTORY_GIT_DEPTH: usize = 120;

#[derive(Clone)]
struct ToDoHistorySite {
    file: String,
    anchor: String,
    site: String,
    line_no: usize,
}

fn run_todo_marker_history_audit(
    manifest_dir: &Path,
    needle: &str,
    violations: &mut Vec<(PathBuf, usize, String, String)>,
) {
    let ledger_path = manifest_dir.join(TO_DO_HISTORY_LEDGER_FILENAME);
    println!("cargo:rerun-if-changed={}", ledger_path.display());

    let current = collect_current_todo_history_sites(manifest_dir, needle);
    let mut next_ledger: std::collections::BTreeMap<(String, String, String), ToDoHistorySite> =
        std::collections::BTreeMap::new();
    for site in current.values() {
        next_ledger.insert(
            (site.file.clone(), site.anchor.clone(), site.site.clone()),
            site.clone(),
        );
    }

    let git_contents = git_head_files_containing(manifest_dir, &[needle]);
    let mut previous = load_todo_history_ledger(&ledger_path);
    for (rel, content) in &git_contents {
        let rel_path = Path::new(rel);
        for site in collect_todo_history_sites_from_content(rel_path, content, needle).into_values()
        {
            previous
                .entry((site.file.clone(), site.anchor.clone(), site.site.clone()))
                .or_insert(site);
        }
    }
    // Recover markers that were relabelled away in recent commits. The ledger is
    // gitignored (absent in a fresh CI clone) and git HEAD no longer carries the
    // committed-away marker, so without this the rename-evasion would escape CI.
    for (key, site) in collect_reworded_marker_sites_from_git_history(manifest_dir, needle) {
        previous.entry(key).or_insert(site);
    }

    let mut current_contents: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    visit_files(manifest_dir, manifest_dir, &mut |rel, content| {
        current_contents.insert(
            rel.to_string_lossy().replace('\\', "/"),
            content.to_string(),
        );
    });

    for (key, previous_site) in &previous {
        if current.contains_key(key) {
            continue;
        }
        let Some(current_content) = current_contents.get(&previous_site.file) else {
            continue;
        };
        let git_content = git_contents.get(&previous_site.file).map(String::as_str);
        if let Some((line_no, reason)) =
            removed_todo_site_violation(current_content, git_content, previous_site, needle)
        {
            violations.push((
                PathBuf::from(&previous_site.file),
                line_no,
                reason,
                previous_site.site.clone(),
            ));
            next_ledger.insert(key.clone(), previous_site.clone());
        }
    }

    save_todo_history_ledger(&ledger_path, &next_ledger);
}

fn collect_current_todo_history_sites(
    manifest_dir: &Path,
    needle: &str,
) -> std::collections::BTreeMap<(String, String, String), ToDoHistorySite> {
    let mut out = std::collections::BTreeMap::new();
    visit_files(manifest_dir, manifest_dir, &mut |rel, content| {
        for site in collect_todo_history_sites_from_content(rel, content, needle).into_values() {
            out.insert(
                (site.file.clone(), site.anchor.clone(), site.site.clone()),
                site,
            );
        }
    });
    out
}

fn collect_todo_history_sites_from_content(
    rel: &Path,
    content: &str,
    needle: &str,
) -> std::collections::BTreeMap<(String, String, String), ToDoHistorySite> {
    let mut out = std::collections::BTreeMap::new();
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    if rel_str == "build.rs" {
        return out;
    }
    if !content.contains(needle) {
        return out;
    }
    let is_rust = rel.extension().and_then(OsStr::to_str) == Some("rs");
    let lines: Vec<&str> = content.lines().collect();
    let stripped_lines = if is_rust {
        strip_file_lines(content)
    } else {
        Vec::new()
    };
    for (idx, line) in lines.iter().enumerate() {
        if !line.contains(needle) {
            continue;
        }
        let anchor = if is_rust {
            find_enclosing_fn(&lines, &stripped_lines, idx)
                .map(|(sig, _)| format!("{FN_ANCHOR_PREFIX}{sig}"))
                .unwrap_or_else(|| FILE_ANCHOR.to_string())
        } else {
            FILE_ANCHOR.to_string()
        };
        let site = normalize_history_site_line(line);
        let entry = ToDoHistorySite {
            file: rel_str.clone(),
            anchor,
            site,
            line_no: idx + 1,
        };
        out.insert(
            (entry.file.clone(), entry.anchor.clone(), entry.site.clone()),
            entry,
        );
    }
    out
}

fn normalize_history_site_line(line: &str) -> String {
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('\t', " ")
        .chars()
        .take(240)
        .collect()
}

/// Lowercase a comment's prose, dropping comment punctuation and collapsing every
/// run of non-alphanumeric characters to a single space, then trimming. Makes the
/// rename-evasion comparison insensitive to comment style (`//` vs `#`), the
/// leading `:` after a marker, and incidental spacing.
fn normalize_marker_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = true;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            for low in ch.to_lowercase() {
                out.push(low);
            }
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

/// The normalized human description trailing a marker token in `site_line`: the
/// text after `needle`, with an optional leading `(qualifier)` and `:`/spacing
/// stripped. `None` when there is no description to key on.
fn marker_description(site_line: &str, needle: &str) -> Option<String> {
    let pos = site_line.find(needle)?;
    let after = site_line[pos + needle.len()..].trim_start();
    let after = match after.strip_prefix('(') {
        Some(rest) => match rest.find(')') {
            Some(close) => &rest[close + 1..],
            None => rest,
        },
        None => after,
    };
    let desc = normalize_marker_text(after);
    if desc.is_empty() { None } else { Some(desc) }
}

/// True for a line that begins (after indentation) with a `//` or `#` line-comment
/// lead-in, excluding Rust attributes (`#[...]`). Only a comment line can carry a
/// relabelled marker; a description resurfacing inside real code is not one.
fn line_is_comment_lead(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("//") || (t.starts_with('#') && !t.starts_with("#["))
}

/// Minimum count of *significant* (non-stopword, >=3-char) tokens a recorded
/// marker description must carry before the fuzzy paraphrase matcher will key on
/// it. The token-set analogue of `MIN_REWORD_DESCRIPTION_LEN`: too few load-
/// bearing tokens and an innocent comment could coincidentally overlap.
const MIN_REWORD_SIGNIFICANT_TOKENS: usize = 5;

/// Fraction of a recorded description's significant tokens that must reappear in
/// one current comment block for it to count as the same deferred work
/// resurfacing under a paraphrase. High enough that unrelated prose does not
/// trip it; low enough that reordering or light rewording does not escape.
const REWORD_CONTAINMENT_THRESHOLD: f64 = 0.75;

/// A pooled contiguous run of comment-lead lines: its significant-token set, its
/// lowercased joined text, and the 1-based line where the run begins.
struct CommentBlock {
    start_line: usize,
    tokens: std::collections::HashSet<String>,
    lower_text: String,
}

/// True for short, structural words that carry no description signal. Filtered
/// before measuring token overlap so the ratio reflects the load-bearing
/// nouns/verbs of the deferred work, not glue words.
fn reword_is_stopword(t: &str) -> bool {
    const STOP: &[&str] = &[
        "the", "and", "for", "this", "that", "with", "from", "into", "over", "each", "can", "will",
        "its", "are", "was", "were", "has", "have", "had", "not", "but", "use", "uses", "used",
        "via", "per", "out", "off", "all", "any", "one", "two", "new", "old", "should", "would",
        "could", "may", "might", "must", "then", "than", "they", "them", "when", "where", "which",
        "while", "here", "there", "only", "also", "still", "yet", "now", "later", "our", "your",
        "their", "been", "being", "such", "some", "more", "most", "less", "few",
    ];
    STOP.contains(&t)
}

/// The significant tokens of `text`: lowercased alphanumeric words (via the same
/// normalizer the exact path uses) with stopwords and sub-3-char tokens dropped.
fn reword_significant_tokens(text: &str) -> Vec<String> {
    normalize_marker_text(text)
        .split(' ')
        .filter(|t| t.len() >= 3 && !reword_is_stopword(t))
        .map(|t| t.to_string())
        .collect()
}

/// Share of `needle_tokens` present in `haystack`. 1.0 = every described token
/// resurfaced; 0.0 = none did.
fn token_set_containment(
    needle_tokens: &[String],
    haystack: &std::collections::HashSet<String>,
) -> f64 {
    if needle_tokens.is_empty() {
        return 0.0;
    }
    let hit = needle_tokens
        .iter()
        .filter(|t| haystack.contains(*t))
        .count();
    hit as f64 / needle_tokens.len() as f64
}

/// True when `lower_text` (an already-lowercased comment block) still speaks in
/// deferral terms — a relabel synonym or an explicit future marker. Only STRONG
/// cues count (not bare modals like "can"/"should", which pepper ordinary docs):
/// the threat is relabelling a marker into *another* deferral note, and a
/// paraphrase that drops every deferral cue reads as plain documentation of
/// current state, not an owed-work tracker. This gate keeps the fuzzy matcher
/// from flagging a past-tense "we did X" note about the now-finished work. Cue
/// fragments are assembled at runtime so this file never carries the literal
/// marker tokens it forbids.
fn comment_block_has_deferral_cue(lower_text: &str) -> bool {
    use std::sync::OnceLock;
    static CUES: OnceLock<Vec<String>> = OnceLock::new();
    let cues = CUES.get_or_init(|| {
        vec![
            format!("{}{}", "to", "do"),
            format!("{}{}", "fix", "me"),
            format!("{}-{}", "follow", "up"),
            format!("{}{}", "follow", "up"),
            format!("{}{}", "de", "fer"),
            format!("{}{}", "stop", "gap"),
            format!("{}{}", "place", "holder"),
            format!("{}{}", "w", "ip"),
            format!("{}{}", "t", "bd"),
            "revisit".to_string(),
            "punt".to_string(),
            "eventually".to_string(),
            "someday".to_string(),
            "pending".to_string(),
            "unimplemented".to_string(),
            "incomplete".to_string(),
            "stub".to_string(),
            "not yet".to_string(),
            "for now".to_string(),
            "to be done".to_string(),
            "to be implemented".to_string(),
            "come back".to_string(),
            "needs to".to_string(),
            "remains to".to_string(),
            "yet to".to_string(),
            format!("not wired {}", "into"),
            format!("not wired {}", "through"),
            format!("not yet {}", "wired"),
            format!("deferred to a {}", "follow"),
            format!("left to a {}", "follow"),
        ]
    });
    cues.iter().any(|c| lower_text.contains(c.as_str()))
}

/// Pool every maximal run of consecutive comment-lead lines in `content` into a
/// `CommentBlock`. Runs that still literally carry `needle` are skipped — those
/// are live markers handled by the lexical ban, not relabels. Pooling the whole
/// run is what defeats the line-break dodge: a description re-wrapped across
/// several `///` lines lands in one token bag.
fn comment_blocks_without_marker(content: &str, needle: &str) -> Vec<CommentBlock> {
    let lines: Vec<&str> = content.lines().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if !line_is_comment_lead(lines[i]) {
            i += 1;
            continue;
        }
        let start = i;
        let mut joined = String::new();
        loop {
            while i < lines.len() && line_is_comment_lead(lines[i]) {
                joined.push(' ');
                joined.push_str(lines[i]);
                i += 1;
            }
            // Bridge blank lines: a description split across comment fragments by
            // inserting a blank line is still pooled as one block. The block ends
            // only at a real code line, defeating the blank-line-split dodge.
            let mut j = i;
            while j < lines.len() && lines[j].trim().is_empty() {
                j += 1;
            }
            if j > i && j < lines.len() && line_is_comment_lead(lines[j]) {
                i = j;
            } else {
                break;
            }
        }
        if joined.contains(needle) {
            continue;
        }
        out.push(CommentBlock {
            start_line: start + 1,
            tokens: reword_significant_tokens(&joined).into_iter().collect(),
            lower_text: joined.to_lowercase(),
        });
    }
    out
}

/// Detect the rename-evasion: the marker token is gone from its recorded site, but
/// the *deferred description* it carried still survives in a current comment line
/// (relabelled — e.g. the `// TODO:` / `# TODO(x):` lead-in swapped for
/// `# Known limitation`/`# Deferred`). Returns the 1-based line number of the
/// surviving remnant. Independent of git HEAD, so committing the relabel away
/// cannot escape it; gated on a reasonably specific description to avoid
/// coincidental matches.
fn reworded_marker_remnant_survives(
    current_content: &str,
    previous_site: &ToDoHistorySite,
    needle: &str,
) -> Option<usize> {
    let description = marker_description(&previous_site.site, needle)?;

    // (A) Exact relabel: the normalized description survives verbatim in a
    // current comment line. Unchanged from the original audit — the char floor
    // keeps short generic tails from colliding.
    if description.len() >= MIN_REWORD_DESCRIPTION_LEN {
        for (idx, line) in current_content.lines().enumerate() {
            if line.contains(needle) || !line_is_comment_lead(line) {
                continue;
            }
            if normalize_marker_text(line).contains(&description) {
                return Some(idx + 1);
            }
        }
    }

    // (B) Paraphrase / re-wrap relabel: the description's significant tokens
    // resurface (reordered, line-rewrapped, or lightly reworded) across one
    // contiguous comment block that still speaks in deferral terms. Pooling the
    // block matches a description split across several `///` lines as a single
    // bag, defeating the line-break dodge; the strong-cue gate keeps a
    // past-tense note about the finished work from being read as an owed marker.
    let needle_tokens = reword_significant_tokens(&description);
    if needle_tokens.len() >= MIN_REWORD_SIGNIFICANT_TOKENS {
        for block in comment_blocks_without_marker(current_content, needle) {
            if token_set_containment(&needle_tokens, &block.tokens) >= REWORD_CONTAINMENT_THRESHOLD
                && comment_block_has_deferral_cue(&block.lower_text)
            {
                return Some(block.start_line);
            }
        }
    }
    None
}

/// Walk recent git history for marker lines that were *relabelled away* — a hunk
/// that removed a marker-bearing line and, in the same hunk, added a line which
/// preserves that line's description but no longer carries the marker. These are
/// the rename-evasions that survive committing the change, so the working-tree +
/// gitignored-ledger view (absent in a fresh CI clone) would otherwise miss them.
/// Returned as history sites keyed like the ledger so they merge into `previous`.
fn collect_reworded_marker_sites_from_git_history(
    manifest_dir: &Path,
    needle: &str,
) -> std::collections::BTreeMap<(String, String, String), ToDoHistorySite> {
    let mut out = std::collections::BTreeMap::new();
    let output = match Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("log")
        .arg(format!("-n{TO_DO_HISTORY_GIT_DEPTH}"))
        .arg(format!("-G{needle}"))
        .arg("-p")
        .arg("--no-color")
        .arg("-U0")
        .arg("--format=%n")
        .arg("--")
        .arg(".")
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return out,
    };
    let text = match String::from_utf8(output.stdout) {
        Ok(t) => t,
        Err(_) => return out,
    };

    fn flush(
        current_file: &Option<String>,
        removed: &mut Vec<String>,
        added: &mut Vec<String>,
        needle: &str,
        out: &mut std::collections::BTreeMap<(String, String, String), ToDoHistorySite>,
    ) {
        if let Some(rel) = current_file {
            let rel_path = Path::new(rel);
            let scannable = rel != "build.rs"
                && !rel_path_is_skipped_for_scans(rel_path)
                && rel_path_is_scannable(rel_path);
            if scannable {
                for removed_line in removed.iter() {
                    if !removed_line.contains(needle) {
                        continue;
                    }
                    let Some(desc) = marker_description(removed_line, needle) else {
                        continue;
                    };
                    // Exact relabel: a single added line carries the verbatim
                    // description under a new label (original behaviour).
                    let exact_relabel = desc.len() >= MIN_REWORD_DESCRIPTION_LEN
                        && added.iter().any(|added_line| {
                            !added_line.contains(needle)
                                && normalize_marker_text(added_line).contains(&desc)
                        });
                    // Fuzzy relabel: pool the hunk's added lines (minus any still
                    // carrying the marker) so a description reworded or rewrapped
                    // across several added lines is matched as one bag, gated on a
                    // strong deferral cue exactly as the working-tree path is.
                    let pooled: String = added
                        .iter()
                        .filter(|a| !a.contains(needle))
                        .map(|a| a.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let needle_tokens = reword_significant_tokens(&desc);
                    let bag: std::collections::HashSet<String> =
                        reword_significant_tokens(&pooled).into_iter().collect();
                    let fuzzy_relabel = needle_tokens.len() >= MIN_REWORD_SIGNIFICANT_TOKENS
                        && token_set_containment(&needle_tokens, &bag)
                            >= REWORD_CONTAINMENT_THRESHOLD
                        && comment_block_has_deferral_cue(&pooled.to_lowercase());
                    if exact_relabel || fuzzy_relabel {
                        let site = ToDoHistorySite {
                            file: rel.clone(),
                            anchor: FILE_ANCHOR.to_string(),
                            site: normalize_history_site_line(removed_line),
                            line_no: 0,
                        };
                        out.insert(
                            (site.file.clone(), site.anchor.clone(), site.site.clone()),
                            site,
                        );
                    }
                }
            }
        }
        removed.clear();
        added.clear();
    }

    let mut current_file: Option<String> = None;
    let mut removed: Vec<String> = Vec::new();
    let mut added: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.starts_with("diff --git ") {
            flush(&current_file, &mut removed, &mut added, needle, &mut out);
            current_file = None;
            continue;
        }
        if let Some(rest) = line.strip_prefix("+++ b/") {
            flush(&current_file, &mut removed, &mut added, needle, &mut out);
            let rel = rest.trim();
            current_file = if rel == "/dev/null" {
                None
            } else {
                Some(rel.to_string())
            };
            continue;
        }
        if line.starts_with("@@") {
            flush(&current_file, &mut removed, &mut added, needle, &mut out);
            continue;
        }
        if line.starts_with("---") || line.starts_with("+++") {
            continue;
        }
        if let Some(body) = line.strip_prefix('-') {
            removed.push(body.to_string());
        } else if let Some(body) = line.strip_prefix('+') {
            added.push(body.to_string());
        }
    }
    flush(&current_file, &mut removed, &mut added, needle, &mut out);
    out
}

fn removed_todo_site_violation(
    current_content: &str,
    git_content: Option<&str>,
    previous_site: &ToDoHistorySite,
    needle: &str,
) -> Option<(usize, String)> {
    // Rename-evasion: the marker token is gone, but the description it carried
    // still survives in a current comment (relabelled). Relabelling does not
    // resolve the owed work, so it costs exactly what the marker did. Checked
    // first and independent of git HEAD, so committing the relabel away cannot
    // escape it.
    if let Some(line_no) = reworded_marker_remnant_survives(current_content, previous_site, needle)
    {
        return Some((
            line_no,
            "marker removed but its description survives reworded under a different \
             label (rename-evasion) — implement the work or delete the whole note; \
             relabelling the marker does not resolve it"
                .to_string(),
        ));
    }

    if let Some(sig) = previous_site.anchor.strip_prefix(FN_ANCHOR_PREFIX) {
        if let Some(git_content) = git_content {
            if normalized_fn_body_code_without_comments(current_content, sig)
                == normalized_fn_body_code_without_comments(git_content, sig)
            {
                return Some((
                    line_for_history_anchor(current_content, &previous_site.anchor),
                    "marker removed but the enclosing function's code is unchanged from git HEAD"
                        .to_string(),
                ));
            }
        }
        return match body_state_for_signature(current_content, sig) {
            HistoryBodyState::FnAbsent | HistoryBodyState::Substantive => None,
            HistoryBodyState::Trivial {
                fn_open_line,
                snippet,
            } => Some((
                fn_open_line + 1,
                format!("marker removed but enclosing function is still trivial: {snippet}"),
            )),
        };
    }

    if let Some(git_content) = git_content
        && normalized_file_code_without_comments(current_content)
            == normalized_file_code_without_comments(git_content)
    {
        return Some((
            previous_site.line_no,
            "marker removed but file code is unchanged from git HEAD".to_string(),
        ));
    }
    None
}

fn line_for_history_anchor(content: &str, anchor: &str) -> usize {
    let Some(sig) = anchor.strip_prefix(FN_ANCHOR_PREFIX) else {
        return 1;
    };
    let lines: Vec<&str> = content.lines().collect();
    let stripped_lines = strip_file_lines(content);
    let mut idx = 0usize;
    while idx < lines.len() {
        let stripped = stripped_lines
            .get(idx)
            .map(String::as_str)
            .unwrap_or(lines[idx]);
        if line_has_keyword(&stripped, "fn")
            && let Some((found_sig, (open, _close))) =
                find_fn_body_at_stripped(&stripped_lines, idx)
            && found_sig == sig
        {
            return open + 1;
        }
        idx += 1;
    }
    1
}

fn normalized_fn_body_code_without_comments(content: &str, target_sig: &str) -> Option<String> {
    let lines: Vec<&str> = content.lines().collect();
    let stripped_lines = strip_file_lines(content);
    let mut idx = 0usize;
    while idx < lines.len() {
        let stripped = stripped_lines
            .get(idx)
            .map(String::as_str)
            .unwrap_or(lines[idx]);
        if line_has_keyword(stripped, "fn")
            && let Some((sig, (open, close))) = find_fn_body_at_stripped(&stripped_lines, idx)
        {
            if sig == target_sig {
                let mut body = String::new();
                for line in stripped_lines.iter().take(close + 1).skip(open) {
                    body.push_str(line);
                    body.push('\n');
                }
                return Some(normalize_code_text(&body));
            }
            idx = close + 1;
            continue;
        }
        idx += 1;
    }
    None
}

fn normalized_file_code_without_comments(content: &str) -> String {
    normalize_code_text(&strip_file_lines(content).join("\n"))
}

fn normalize_code_text(content: &str) -> String {
    content.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn load_todo_history_ledger(
    path: &Path,
) -> std::collections::BTreeMap<(String, String, String), ToDoHistorySite> {
    let mut out = std::collections::BTreeMap::new();
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return out,
    };
    for line in content.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(4, '\t').collect();
        if parts.len() != 4 {
            continue;
        }
        let line_no = parts[3].parse::<usize>().unwrap_or(1);
        let site = ToDoHistorySite {
            file: parts[0].to_string(),
            anchor: parts[1].to_string(),
            site: parts[2].to_string(),
            line_no,
        };
        out.insert(
            (site.file.clone(), site.anchor.clone(), site.site.clone()),
            site,
        );
    }
    out
}

fn save_todo_history_ledger(
    path: &Path,
    ledger: &std::collections::BTreeMap<(String, String, String), ToDoHistorySite>,
) {
    let mut out = String::new();
    out.push_str(
        "# Persistent audit of TO-DO comment/marker sites.\n\
         # Auto-managed by build.rs — do NOT hand-edit. Each non-comment\n\
         # line is tab-separated:\n\
         #   <relative_path>\\t<anchor>\\t<normalized_marker_line>\\t<line_no>\n\
         # When an entry disappears from the source tree, build.rs inspects\n\
         # the enclosing function or file and also compares against git HEAD.\n\
         # Deleting the marker while leaving code unchanged keeps failing.\n",
    );
    for site in ledger.values() {
        out.push_str(&site.file);
        out.push('\t');
        out.push_str(&site.anchor);
        out.push('\t');
        out.push_str(&site.site.replace('\t', " "));
        out.push('\t');
        out.push_str(&site.line_no.to_string());
        out.push('\n');
    }
    match fs::read_to_string(path) {
        Ok(existing) if existing == out => return,
        _ => {}
    }
    let _ = fs::write(path, out);
}

/// True when the function body spanning `open..=close` (brace lines) reaches an
/// assertion-shaped construct, either directly or by delegating to a local
/// helper function that itself reaches one. This is what lets a test delegate
/// its checks to a helper with an ordinary name (`report_and_check`,
/// `verify_recovers`, …) without the gate false-flagging it as assertion-less:
/// the gate follows the call into the helper body and finds the real `assert!`,
/// rather than trusting (or rejecting) the helper by its name prefix.
fn test_body_reaches_assertion(
    lines: &[&str],
    stripped_lines: &[String],
    open: usize,
    close: usize,
    local_fns: &[(String, usize, usize)],
) -> bool {
    let mut visited: Vec<usize> = vec![open];
    body_reaches_assertion(
        lines,
        stripped_lines,
        open,
        close,
        local_fns,
        &mut visited,
        MAX_DELEGATION_DEPTH,
    )
}

/// Depth- and cycle-bounded recursion behind `test_body_reaches_assertion`.
/// `visited` holds the body-open line of every function already on the stack so
/// mutual recursion between helpers cannot loop. `depth` bounds how many
/// delegation hops are followed.
fn body_reaches_assertion(
    lines: &[&str],
    stripped_lines: &[String],
    open: usize,
    close: usize,
    local_fns: &[(String, usize, usize)],
    visited: &mut Vec<usize>,
    depth: usize,
) -> bool {
    for k in open..=close {
        let line_s = stripped_lines
            .get(k)
            .map(String::as_str)
            .unwrap_or_else(|| lines.get(k).copied().unwrap_or(""));
        if line_is_assertion_shaped(line_s) {
            return true;
        }
    }
    if depth == 0 {
        return false;
    }
    let mut callees: Vec<String> = Vec::new();
    for k in open..=close {
        let line_s = stripped_lines
            .get(k)
            .map(String::as_str)
            .unwrap_or_else(|| lines.get(k).copied().unwrap_or(""));
        collect_called_idents(line_s, &mut callees);
    }
    for callee in &callees {
        for (name, h_open, h_close) in local_fns {
            if name != callee || visited.contains(h_open) {
                continue;
            }
            visited.push(*h_open);
            let reached = body_reaches_assertion(
                lines,
                stripped_lines,
                *h_open,
                *h_close,
                local_fns,
                visited,
                depth - 1,
            );
            if reached {
                return true;
            }
        }
    }
    false
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
            let Some((sig, (open, close))) = find_fn_body_at_stripped(&stripped_lines, i) else {
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
        let trait_def_mask = compute_trait_def_mask(content);
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
            // A `fn` inside the body of a `trait <Name> { ... }` definition
            // is a default-method declaration. A sentinel body (`None`,
            // `Ok(())`, `Default::default()`) there is the idiomatic "opt-in
            // override" spelling, not a stub in the make-the-compile-error-
            // vanish sense the ban targets — implementors that need real
            // behaviour override the default. Skip these.
            if trait_def_mask.get(i).copied().unwrap_or(false) {
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
            let Some((sig, (open, close))) = find_fn_body_at_stripped(&stripped_lines, i) else {
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

/// Flags multi-arg functions that read an argument only through control flow
/// or discard statements that do not change the sentinel returned by the
/// function. This deliberately includes trait default methods: `fn f(&self,
/// arg: T) -> Option<_> { if arg.is_empty() { return None; } None }` is not a
/// meaningful default, it is a renamed `_arg`.
fn scan_for_noop_sentinel_control_flow(
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
            if fn_signature_ends_with_semicolon(&stripped_lines, i) {
                i += 1;
                continue;
            }
            let Some((sig, (open, close))) = find_fn_body_at_stripped(&stripped_lines, i) else {
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
            if body_has_noop_sentinel_control_flow(&body) {
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
    // Peel off a "fake validation prologue" first: a sequence of leading
    // `assert*!(...);` and `if <expr>.is_empty() { return <expr>; }`
    // statements that exist solely to consume otherwise-unused
    // parameters. The real body still has to be a trivial sentinel for
    // the whole function to count as a stub — legitimate validation
    // followed by real logic survives because the tail won't match the
    // sentinel list. Each iteration of `strip_validation_prologue`
    // tries, in order:
    //   * any of `assert!` / `assert_eq!` / `assert_ne!` /
    //     `assert_matches!` (any macro-delimiter style),
    //   * a single-statement `if <cond> { return <expr>; }` guard
    //     (multi-statement bodies survive — they may be real work),
    //   * `drop(<expr>);`,
    //   * `_ = <expr>;` — the non-`let` wildcard-assignment dodge that
    //     `scan_for_let_underscore` cannot see.
    // Terminator-method statements (`.unwrap();`, `.expect("...");`,
    // `.unwrap_or_default();`) are deliberately NOT stripped: many
    // legitimate functions exist solely to perform a side-effectful
    // `.send(...).unwrap();` or similar and return `Ok(())`, and
    // stripping the call would punish that correct shape.
    let after_prologue = strip_validation_prologue(body);
    let mut s = after_prologue.trim();
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

fn body_has_noop_sentinel_control_flow(body: &str) -> bool {
    let s = body.trim();
    if s.is_empty() {
        return false;
    }
    if leading_if_return_preserves_sentinel(s) {
        return true;
    }
    if whole_if_else_preserves_sentinel(s) {
        return true;
    }
    if whole_match_preserves_sentinel(s) {
        return true;
    }
    if leading_discard_or_read_then_sentinel(s) {
        return true;
    }
    if whole_predicate_then_trivial_sentinel(s) {
        return true;
    }
    if contains_predicate_then_trivial_sentinel_return(s) {
        return true;
    }
    false
}

fn leading_if_return_preserves_sentinel(s: &str) -> bool {
    let Some((returned, rest)) = strip_leading_if_return_guard_with_expr(s) else {
        return false;
    };
    let Some(returned_key) = trivial_sentinel_key(returned) else {
        return false;
    };
    let Some(tail_key) = trivial_sentinel_key(rest) else {
        return false;
    };
    returned_key == tail_key
}

fn whole_if_else_preserves_sentinel(s: &str) -> bool {
    let Some((then_body, else_body, rest)) = parse_leading_if_else_blocks(s) else {
        return false;
    };
    if !rest.trim().is_empty() {
        return false;
    }
    let Some(then_key) = trivial_sentinel_key(then_body) else {
        return false;
    };
    let Some(else_key) = trivial_sentinel_key(else_body) else {
        return false;
    };
    then_key == else_key
}

fn whole_match_preserves_sentinel(s: &str) -> bool {
    let Some((inside, rest)) = parse_leading_match_block(s) else {
        return false;
    };
    if !rest.trim().is_empty() {
        return false;
    }
    let Some(arms) = split_match_arms(inside) else {
        return false;
    };
    let mut key: Option<String> = None;
    let mut arm_count = 0usize;
    for arm in arms {
        let Some((_, expr)) = arm.split_once("=>") else {
            return false;
        };
        let Some(expr_key) = trivial_sentinel_key(expr.trim().trim_end_matches(',')) else {
            return false;
        };
        if let Some(existing) = &key {
            if existing != &expr_key {
                return false;
            }
        } else {
            key = Some(expr_key);
        }
        arm_count += 1;
    }
    arm_count >= 2 && key.is_some()
}

fn leading_discard_or_read_then_sentinel(s: &str) -> bool {
    let rest = strip_leading_drop_call(s)
        .or_else(|| strip_leading_wildcard_assignment(s))
        .or_else(|| strip_leading_read_only_statement(s));
    rest.is_some_and(|tail| trivial_sentinel_key(tail).is_some())
}

fn whole_predicate_then_trivial_sentinel(s: &str) -> bool {
    let mut expr = s.trim();
    if let Some(stripped) = expr.strip_suffix(';') {
        expr = stripped.trim_end();
    }
    predicate_then_trivial_sentinel_expr(expr)
}

fn predicate_then_trivial_sentinel_expr(expr: &str) -> bool {
    let expr = expr.trim();
    for method in [".then(", ".then_some("] {
        let Some((predicate, arg)) = split_whole_method_call_arg(expr, method) else {
            continue;
        };
        if predicate_is_read_only_empty_check(predicate) && then_arg_is_trivial_sentinel(arg) {
            return true;
        }
    }
    false
}

fn contains_predicate_then_trivial_sentinel_return(s: &str) -> bool {
    let mut search_from = 0usize;
    while let Some(rel) = s[search_from..].find("return") {
        let pos = search_from + rel;
        let before = if pos == 0 {
            None
        } else {
            s.as_bytes().get(pos - 1).copied()
        };
        let after = s.as_bytes().get(pos + "return".len()).copied();
        if before.map_or(true, |b| !is_ident_byte(b))
            && after.is_some_and(|b| b.is_ascii_whitespace())
        {
            let expr_start = pos + "return".len();
            if let Some((expr, _)) = split_leading_statement(&s[expr_start..])
                && predicate_then_trivial_sentinel_expr(expr)
            {
                return true;
            }
        }
        search_from = pos + "return".len();
    }
    false
}

fn split_whole_method_call_arg<'a>(expr: &'a str, method: &str) -> Option<(&'a str, &'a str)> {
    let call_start = expr.find(method)?;
    let arg_start = call_start + method.len();
    let close_rel = find_matching_paren(&expr.as_bytes()[arg_start..])?;
    if !expr[arg_start + close_rel + 1..].trim().is_empty() {
        return None;
    }
    Some((
        expr[..call_start].trim(),
        expr[arg_start..arg_start + close_rel].trim(),
    ))
}

fn predicate_is_read_only_empty_check(predicate: &str) -> bool {
    let p = trim_balanced_outer_parens(predicate.trim());
    if p.is_empty() || p.contains('!') {
        return false;
    }
    p.ends_with(".is_empty()") || predicate_is_zero_comparison(p)
}

fn trim_balanced_outer_parens(mut s: &str) -> &str {
    loop {
        let trimmed = s.trim();
        if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
            return trimmed;
        }
        let inner = &trimmed[1..trimmed.len() - 1];
        if find_matching_paren(inner.as_bytes()).is_some() {
            return trimmed;
        }
        s = inner;
    }
}

fn predicate_is_zero_comparison(predicate: &str) -> bool {
    let Some((left, right)) = predicate.split_once("==") else {
        return false;
    };
    let left = left.trim();
    let right = right.trim();
    (right == "0" && read_only_zero_compare_operand(left))
        || (left == "0" && read_only_zero_compare_operand(right))
}

fn read_only_zero_compare_operand(operand: &str) -> bool {
    let op = operand.trim();
    !op.is_empty()
        && !op.contains('=')
        && !op.contains('!')
        && op
            .bytes()
            .all(|b| is_ident_byte(b) || b == b'.' || b == b'(' || b == b')')
}

fn then_arg_is_trivial_sentinel(arg: &str) -> bool {
    let a = arg.trim();
    if a.is_empty() {
        return false;
    }
    if trivial_sentinel_key(a).is_some() || expression_is_empty_array_constructor(a) {
        return true;
    }
    if let Some(body) = a.strip_prefix("||") {
        let body = body.trim();
        return trivial_sentinel_key(body).is_some() || expression_is_empty_array_constructor(body);
    }
    matches!(
        a,
        "Vec::new"
            | "String::new"
            | "HashMap::new"
            | "BTreeMap::new"
            | "HashSet::new"
            | "BTreeSet::new"
            | "VecDeque::new"
            | "Default::default"
    )
}

fn expression_is_empty_array_constructor(expr: &str) -> bool {
    let compact: String = expr.chars().filter(|c| !c.is_whitespace()).collect();
    compact.contains("::zeros(0)")
        || compact.contains("::zeros((0,")
        || compact.contains("::from_shape_vec((0,")
        || compact.contains("::from_shape_fn((0,")
}

fn trivial_sentinel_key(expr: &str) -> Option<String> {
    let mut s = expr.trim();
    if let Some(rest) = s.strip_prefix("return ") {
        s = rest.trim();
    }
    if let Some(stripped) = s.strip_suffix(';') {
        s = stripped.trim_end();
    }
    if return_expr_is_trivial_sentinel(s) {
        return Some(collapse_ascii_whitespace(s));
    }
    None
}

fn collapse_ascii_whitespace(s: &str) -> String {
    let mut out = String::new();
    let mut last_was_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}

fn parse_leading_if_else_blocks(s: &str) -> Option<(&str, &str, &str)> {
    let after_if = s.strip_prefix("if ")?;
    let open_in_after_if = find_top_level_open_brace(after_if)?;
    let then_start = open_in_after_if + 1;
    let then_close_rel = find_matching_brace(&after_if.as_bytes()[then_start..])?;
    let then_body = &after_if[then_start..then_start + then_close_rel];
    let after_then = after_if[then_start + then_close_rel + 1..].trim_start();
    let after_else = after_then.strip_prefix("else")?.trim_start();
    if !after_else.starts_with('{') {
        return None;
    }
    let else_body_start = 1usize;
    let else_close_rel = find_matching_brace(&after_else.as_bytes()[else_body_start..])?;
    let else_body = &after_else[else_body_start..else_body_start + else_close_rel];
    let rest = &after_else[else_body_start + else_close_rel + 1..];
    Some((then_body, else_body, rest))
}

fn parse_leading_match_block(s: &str) -> Option<(&str, &str)> {
    let after_match = s.strip_prefix("match ")?;
    let open = find_top_level_open_brace(after_match)?;
    let body_start = open + 1;
    let close_rel = find_matching_brace(&after_match.as_bytes()[body_start..])?;
    let body = &after_match[body_start..body_start + close_rel];
    let rest = &after_match[body_start + close_rel + 1..];
    Some((body, rest))
}

fn find_top_level_open_brace(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut paren: i32 = 0;
    let mut brack: i32 = 0;
    let mut angle: i32 = 0;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
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
            b'{' if paren == 0 && brack == 0 && angle == 0 => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

/// Given bytes starting just after an opening `{`, find the matching `}`.
fn find_matching_brace(bytes: &[u8]) -> Option<usize> {
    let mut depth: i32 = 1;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
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

fn split_match_arms(inside: &str) -> Option<Vec<&str>> {
    let mut arms = Vec::new();
    let bytes = inside.as_bytes();
    let mut start = 0usize;
    let mut paren: i32 = 0;
    let mut brack: i32 = 0;
    let mut brace: i32 = 0;
    let mut angle: i32 = 0;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => paren += 1,
            b')' => paren -= 1,
            b'[' => brack += 1,
            b']' => brack -= 1,
            b'{' => brace += 1,
            b'}' => brace -= 1,
            b'<' => angle += 1,
            b'>' => {
                if angle > 0 {
                    angle -= 1;
                }
            }
            b',' if paren == 0 && brack == 0 && brace == 0 && angle == 0 => {
                let arm = inside[start..i].trim();
                if !arm.is_empty() {
                    arms.push(arm);
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    let tail = inside[start..].trim();
    if !tail.is_empty() {
        arms.push(tail);
    }
    if arms.is_empty() { None } else { Some(arms) }
}

fn strip_leading_read_only_statement(s: &str) -> Option<&str> {
    let (statement, rest) = split_leading_statement(s)?;
    if expression_statement_is_read_only_noop(statement.trim()) {
        Some(rest)
    } else {
        None
    }
}

fn split_leading_statement(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    let mut paren: i32 = 0;
    let mut brack: i32 = 0;
    let mut brace: i32 = 0;
    let mut angle: i32 = 0;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => paren += 1,
            b')' => paren -= 1,
            b'[' => brack += 1,
            b']' => brack -= 1,
            b'{' => brace += 1,
            b'}' => brace -= 1,
            b'<' => angle += 1,
            b'>' => {
                if angle > 0 {
                    angle -= 1;
                }
            }
            b';' if paren == 0 && brack == 0 && brace == 0 && angle == 0 => {
                return Some((&s[..i], &s[i + 1..]));
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn expression_statement_is_read_only_noop(statement: &str) -> bool {
    let s = statement.trim();
    if s.is_empty() || s.contains('=') || s.contains('!') {
        return false;
    }
    let known_read_calls = [
        ".len()",
        ".is_empty()",
        ".capacity()",
        ".nrows()",
        ".ncols()",
        ".shape()",
        ".dim()",
        ".raw_dim()",
        "std::mem::size_of_val(",
        "core::mem::size_of_val(",
    ];
    known_read_calls.iter().any(|needle| s.contains(needle))
}

/// Strip a leading sequence of "fake validation" statements from a body
/// expression and return the tail. A statement is consumed if it is
/// either a complete `assert*!(...);` invocation or an
/// `if <expr>.is_empty() { return <expr>; }` early-return guard.
/// Loops until no further prefix matches. Whitespace between statements
/// is tolerated. If nothing matches, returns the input unchanged so the
/// caller's existing sentinel check still fires.
fn strip_validation_prologue(body: &str) -> &str {
    let mut s = body.trim_start();
    loop {
        if let Some(rest) = strip_leading_assert_call(s) {
            s = rest.trim_start();
            continue;
        }
        if let Some(rest) = strip_leading_if_return_guard(s) {
            s = rest.trim_start();
            continue;
        }
        if let Some(rest) = strip_leading_drop_call(s) {
            s = rest.trim_start();
            continue;
        }
        if let Some(rest) = strip_leading_wildcard_assignment(s) {
            s = rest.trim_start();
            continue;
        }
        break;
    }
    s
}

/// Match a leading assertion macro invocation (any of
/// `assert!`, `assert_eq!`, `assert_ne!`, `assert_matches!`) with
/// balanced parens / square brackets / curly braces and a trailing `;`.
/// Returns the remainder past the `;`, or `None` if no match. Both
/// `assert_matches!(...)` and the rare brace/bracket-delimited forms
/// (`assert!{ ... }`) are accepted — `macro!` delimiters are
/// interchangeable in Rust.
fn strip_leading_assert_call(s: &str) -> Option<&str> {
    for name in ["assert", "assert_eq", "assert_ne", "assert_matches"] {
        let Some(after_name) = s.strip_prefix(name) else {
            continue;
        };
        let after_bang = after_name.strip_prefix('!')?;
        let after_open_ws = after_bang.trim_start();
        let (open, close) = match after_open_ws.as_bytes().first()? {
            b'(' => (b'(', b')'),
            b'[' => (b'[', b']'),
            b'{' => (b'{', b'}'),
            _ => return None,
        };
        let after_open = &after_open_ws[1..];
        let bytes = after_open.as_bytes();
        let mut depth: i32 = 1;
        let mut i = 0usize;
        while i < bytes.len() {
            let b = bytes[i];
            if b == open {
                depth += 1;
            } else if b == close {
                depth -= 1;
                if depth == 0 {
                    let after = after_open[i + 1..].trim_start();
                    return after.strip_prefix(';');
                }
            }
            i += 1;
        }
        return None;
    }
    None
}

/// Match a leading `if <cond> { return <expr>; }` early-return guard,
/// where the gated block is exactly a single `return ...;` statement.
/// The condition itself is not inspected — any predicate counts, since
/// the goal is to peel off cheap "consume the parameter" guards before
/// checking whether the tail of the function reduces to a sentinel.
/// Legitimate functions whose body is `if cond { return X; } <real
/// computation>` are unaffected because the surviving tail won't match
/// the sentinel list. Returns the remainder past the closing brace, or
/// `None` if no match.
fn strip_leading_if_return_guard(s: &str) -> Option<&str> {
    strip_leading_if_return_guard_with_expr(s).map(|(_, rest)| rest)
}

fn strip_leading_if_return_guard_with_expr(s: &str) -> Option<(&str, &str)> {
    let after_if = s.strip_prefix("if ")?;
    let bytes = after_if.as_bytes();
    let mut paren: i32 = 0;
    let mut brack: i32 = 0;
    let mut i = 0usize;
    let open_brace = loop {
        if i >= bytes.len() {
            return None;
        }
        match bytes[i] {
            b'(' => paren += 1,
            b')' => paren -= 1,
            b'[' => brack += 1,
            b']' => brack -= 1,
            b'{' if paren == 0 && brack == 0 => break i,
            _ => {}
        }
        i += 1;
    };
    let rest_after_brace = &after_if[open_brace + 1..];
    let bytes = rest_after_brace.as_bytes();
    let mut depth: i32 = 1;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let inside = rest_after_brace[..i].trim();
                    if inside.starts_with("return ")
                        && inside.ends_with(';')
                        && count_top_level_semicolons(inside) == 1
                    {
                        // Only strip the guard when the returned expression is
                        // itself a trivial sentinel. A return that carries a
                        // computed Err (e.g. `Err(format!(...))`,
                        // `Err(SomeError::Variant(...))`) is real validation —
                        // peeling it would turn a multi-check validator into
                        // a stub-shaped residual `Ok(())` / `None` and produce
                        // a false positive in the stub-body lint.
                        let ret_expr = inside["return ".len()..inside.len() - 1].trim();
                        if return_expr_is_trivial_sentinel(ret_expr) {
                            return Some((ret_expr, &rest_after_brace[i + 1..]));
                        }
                    }
                    return None;
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Count top-level `;` characters in `s` — `;` that lie outside any
/// balanced `()` / `[]` / `{}` / `<...>` group. Used by the early-
/// return guard matcher to confirm that a gated block is a single
/// statement (`return <expr>;`) and not a multi-statement block that
/// happens to start with `return ...; <real work>;`.
fn count_top_level_semicolons(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut paren: i32 = 0;
    let mut brack: i32 = 0;
    let mut brace: i32 = 0;
    let mut angle: i32 = 0;
    let mut count = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => paren += 1,
            b')' => paren -= 1,
            b'[' => brack += 1,
            b']' => brack -= 1,
            b'{' => brace += 1,
            b'}' => brace -= 1,
            b'<' => angle += 1,
            b'>' => {
                if angle > 0 {
                    angle -= 1;
                }
            }
            b';' if paren == 0 && brack == 0 && brace == 0 && angle == 0 => {
                count += 1;
            }
            _ => {}
        }
        i += 1;
    }
    count
}

/// Classify an early-return expression as "trivial sentinel" — the kinds of
/// values a fake-validation guard would return to cheaply consume a param.
/// Real validators return `Err(format!(...))` / `Err(SomeError::Variant(...))`
/// / other computed values, which are NOT trivial and must NOT match.
///
/// Kept in sync with the literal-match set in `body_is_trivial_sentinel`.
fn return_expr_is_trivial_sentinel(expr: &str) -> bool {
    let s = expr.trim();
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
            | "Array1::zeros(0)"
            | "Array2::zeros((0, 0))"
            | "Array3::zeros((0, 0, 0))"
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
    if let Some(inner) = s.strip_prefix("Ok(").and_then(|r| r.strip_suffix(')'))
        && matches!(
            inner.trim(),
            "()" | "Array1::zeros(0)" | "Array2::zeros((0, 0))" | "Array3::zeros((0, 0, 0))"
        )
    {
        return true;
    }
    is_bare_numeric_literal(s)
}

/// Match a leading `drop(<expr>);` statement (any expression) and
/// return the remainder past the `;`. `drop` is a legitimate primitive,
/// but as a leading no-op consumer that produces nothing it has the
/// same fake-use shape as `let _ = expr;` once that ban exists.
fn strip_leading_drop_call(s: &str) -> Option<&str> {
    let after_open = s.strip_prefix("drop(")?;
    let bytes = after_open.as_bytes();
    let mut depth: i32 = 1;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    let after = after_open[i + 1..].trim_start();
                    return after.strip_prefix(';');
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Match a leading `_ = <expr>;` wildcard assignment — the non-`let`
/// form of `let _ = expr;` that escapes the `scan_for_let_underscore`
/// scanner because there is no `let`. The expression body is consumed
/// up to the next top-level `;`.
fn strip_leading_wildcard_assignment(s: &str) -> Option<&str> {
    let after_underscore = s.strip_prefix('_')?;
    // Reject identifier continuations (`_x`, `_1`, `__`): those are
    // ordinary identifiers, not the wildcard pattern.
    if after_underscore
        .as_bytes()
        .first()
        .is_some_and(|b| is_ident_byte(*b))
    {
        return None;
    }
    let rest = after_underscore.trim_start().strip_prefix('=')?;
    // Reject `_ == ...` (comparison) — only the single-`=` assignment
    // form is the wildcard discard.
    if rest.as_bytes().first() == Some(&b'=') {
        return None;
    }
    let rest = rest.trim_start();
    let bytes = rest.as_bytes();
    let mut paren: i32 = 0;
    let mut brack: i32 = 0;
    let mut brace: i32 = 0;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => paren += 1,
            b')' => paren -= 1,
            b'[' => brack += 1,
            b']' => brack -= 1,
            b'{' => brace += 1,
            b'}' => brace -= 1,
            b';' if paren == 0 && brack == 0 && brace == 0 => {
                return Some(&rest[i + 1..]);
            }
            _ => {}
        }
        i += 1;
    }
    None
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
    unreferenced_pub_scoped: &mut Vec<(PathBuf, usize, String, String)>,
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
            || path_matches_crates_test_or_example(&rel_str);
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

    // Extract definitions from src/ files. Visibility is retained so the
    // flag step can apply the right rule per item:
    //   * Private with zero consumers: skipped — rustc's `dead_code`
    //     lint already covers them; flagging would duplicate.
    //   * `pub(crate)` / `pub(super)` with zero consumers: flagged as
    //     unreferenced (the gap rustc cannot see in a library-shaped
    //     crate, which assumes the items might be used downstream).
    //   * Any non-`pub` item consumed only by tests: flagged as test-only
    //     (the original rule).
    // Bare `pub` items are excluded — their consumers may live in
    // downstream crates we cannot see.
    let mut defs: Vec<(usize, usize, String, Visibility)> = Vec::new();
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
            defs.push((fi, idx, ident, vis));
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

    for (fi, line_idx, ident, vis) in defs {
        if pub_use_idents.contains(&ident) {
            continue;
        }
        let test_hint = test_first_hit.get(&ident).cloned();
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
        match (test_hint, vis) {
            (Some(hint_path), _) => {
                let hint = format!(
                    "{} (test ref: {})",
                    ident,
                    hint_path.to_string_lossy().replace('\\', "/")
                );
                offenders.push((sf.rel.clone(), line_idx + 1, hint, raw));
            }
            (None, Visibility::PubScoped) => {
                // Zero consumers anywhere in src/ or tests/, and the
                // item is `pub(crate)` / `pub(super)` — rustc's
                // `dead_code` lint cannot see this in a library-shaped
                // crate. Routed through the dedicated section so the
                // report is distinct from the test-only finding.
                unreferenced_pub_scoped.push((sf.rel.clone(), line_idx + 1, ident.clone(), raw));
            }
            (None, _) => {
                // Private with zero consumers — rustc's `dead_code`
                // already covers this. Skip to avoid duplicate diagnostics.
            }
        }
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
    for kw in [
        "fn ", "struct ", "enum ", "const ", "static ", "type ", "trait ",
    ] {
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

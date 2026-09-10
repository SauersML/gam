use std::ffi::OsStr;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::{env, fs};

fn main() {
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
    // binding (e.g. `drop(expr)` for a value whose Drop is the point, or
    // `expr;` for a `Result` consumer wrapped behind an explicit check).
    // NOT `.ok();` — that spelling is banned for the same reason `let _ =`
    // is: it converts a failure into an `Option` nobody looks at. Handle the
    // error or propagate it.
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

    // Empty brace blocks, on one line or spread over several. An empty
    // function is an unimplemented stub wearing a green check; an empty
    // `else` / match arm is a branch that swallows what reached it. Both
    // are pressure to restructure rather than to fill the block in.
    let mut empty_block_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_empty_blocks(&manifest_dir, &manifest_dir, &mut empty_block_offenders);

    // A `match` whose every arm does nothing. Every match that compiles is
    // exhaustive, so this is the whole-construct form of the empty block above:
    // the scrutinee is computed, dispatched on, and discarded. Catches the
    // `A => ()` and single-line spellings that start from no literal `{}` and
    // so never reach the empty-block scanner at all.
    let mut no_op_match_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_no_op_matches(&manifest_dir, &manifest_dir, &mut no_op_match_offenders);

    // Underscore function parameter names. `_name: T` silences the
    // unused-parameter warning by hiding it from the lint rather than fixing
    // it. Use the parameter, restructure the API so it isn't passed, or
    // delete the param. The bare `_: T` placeholder is banned identically:
    // it is the same admission that the signature carries an argument the
    // body never reads, and dropping the name only makes the dead argument
    // harder to grep for. There is no exemption — not for macro-threaded
    // tokens (`_py: Python<'_>`), not for trait-method signatures, not for
    // callback shims. Change the signature. Build.rs is exempt.
    // Closure parameters with a fake binding name (`|_x| …`). Bare `|_|` is
    // legal: the caller dictates a closure's parameter list, so "I ignore this
    // argument" is a true statement. `_x` is that statement plus a lie.
    let mut underscore_closure_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_underscore_closure_params(
        &manifest_dir,
        &manifest_dir,
        &mut underscore_closure_offenders,
    );

    let mut underscore_fn_arg_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_underscore_fn_args(
        &manifest_dir,
        &manifest_dir,
        &mut underscore_fn_arg_offenders,
    );

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
    // redeemed and auto-pruned. The ledger is rewritten in place.
    let mut probation_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_probation_oversized_files(&manifest_dir, &mut probation_offenders);

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

    if !empty_block_offenders.is_empty() {
        sections.push(Section {
            title: "empty block body (handle the case, or restructure so the block does not exist)"
                .to_string(),
            rows: empty_block_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !no_op_match_offenders.is_empty() {
        sections.push(Section {
            title:
                "match whose every arm does nothing (`A => {}` / `A => ()`) — the scrutinee is \
                    computed, dispatched on and thrown away, so the whole dispatch is dead: make \
                    an arm do the work, or delete the match and whatever feeds it. If the point \
                    is that adding a variant must fail to compile, spell it as an irrefutable \
                    `let (A | B) = value;` — same compile error when the enum grows, no branch, \
                    nothing swallowed."
                    .to_string(),
            rows: no_op_match_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !underscore_closure_offenders.is_empty() {
        sections.push(Section {
            title: "underscore closure parameter, `|_x|` (use the value, or drop the name: `|_|`)"
                .to_string(),
            rows: underscore_closure_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !underscore_fn_arg_offenders.is_empty() {
        sections.push(Section {
            title:
                "underscore fn parameter, `_name: T` or bare `_: T` (use the value, restructure the API, or delete the param)"
                    .to_string(),
            rows: underscore_fn_arg_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !cfg_test_pub_offenders.is_empty() {
        sections.push(Section {
            title:
                "#[cfg(test)] on src/ item (move into a private `#[cfg(test)] mod tests { ... }` / `mod test_support` / `mod tests_*` / `mod *_tests`, or delete the unused item — `#[cfg(test)]` is not a dead_code-lint escape hatch, regardless of visibility). \
                 The accepted names are a PREFIX (`tests_...`) or a SUFFIX (`..._tests`) — `tests` in the MIDDLE does not match, so an issue number tacked on the end (`foo_tests_2333`) is rejected: write `foo_2333_tests`"
                    .to_string(),
            rows: cfg_test_pub_offenders
                .iter()
                .map(|(r, l, item, s)| (r.clone(), *l, Some(item.clone()), s.clone()))
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
    //
    // The report site is matched by CALL, not by the exact argument spelling.
    // Pinning the whole line (`render_report(&sections);`) made any rename of
    // the aggregator's local a build-breaking panic, so the gate resisted
    // ordinary refactors while detecting nothing extra: what it must guarantee
    // is that exactly one report site exists and that it is followed by an
    // unconditional hard exit, neither of which depends on what the argument is
    // called. Argument-agnostic here, same guarantee below.
    let render_call = format!("render_{}(", "report");
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
        .filter(|(_, l)| {
            let t = l.trim();
            t.starts_with(render_call.as_str()) && t.ends_with(");")
        })
        .map(|(i, _)| i)
        .collect();
    if render_sites.len() != 1 {
        panic!(
            "self-integrity gate: expected exactly one `{render_call}...);` report site, found {}. \
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
    // exemption/allowlist that skips a gate for hand-picked commits. This arm is
    // content-based,
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
    // different owner, and git otherwise refuses with "dubious ownership").
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

    let output = Command::new("git")
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
        .expect("failed to run git log for build.rs author audit");

    if !output.status.success() {
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
        // JSON files are immutable data/provenance records: a
        // `gamfit_version` there names the runtime that produced the recorded
        // result. Rewriting it during a release would falsify provenance.
        // Executable version contracts belong in source or manifests, which
        // remain covered by this scan and the two structural checks above.
        if rel_str == "build.rs" || rel.extension().and_then(OsStr::to_str) == Some("json") {
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
        // `ends_with(".rs")` anywhere in the codebase: often a tautological
        // assertion (e.g. `file!().ends_with(".rs")`) that asserts nothing
        // about the unit under test. Strict everywhere — tests must verify a
        // real property.
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
        // ---- no-op family ----
        // A construct that provably does nothing is either dead weight or a
        // silencer for a signal someone did not want to read. Delete it, or
        // make the call site stop requiring it.
        //
        // Empty brace bodies (`fn f() {}`, `else {}`, `_ => {}`) have their
        // own scanner — `scan_for_empty_blocks` — because a lexical needle
        // is dodged by a line break.
        //
        // `_ => ()` and `{ () }` are the brace-free spellings of the same
        // thing: an arm or block that evaluates to nothing. An empty
        // catch-all arm in particular means every variant the author did not
        // enumerate silently does nothing, and a variant added later
        // inherits that silence instead of a compile error.
        ("_ => ()", "_ => ()", false),
        ("{ () }", "empty block body", false),
        // No-op closures. `|_| ()` is the callback spelling of `let _ = x;`:
        // `res.map(|_| ())` exists only to erase a value the caller then
        // ignores — return the value or drop the call. The brace form
        // (`|_| {}`, `|_a, _b| {}`) is `scan_for_empty_blocks`'s job, since a
        // needle would miss every parameter spelling and every line break.
        ("|_| ()", "no-op closure", false),
        // Throwing away the ERROR, which is the signal this table exists to
        // protect. `.map_err(drop)` / `.map_err(|_| ())` turns "it failed
        // because X" into "it failed", and nothing downstream can ever
        // recover X. Propagate it, or map it to a type that still says why.
        //
        // Dropping the SUCCESS payload is a different thing and is NOT banned
        // — see `MAP_PAYLOAD_DISCARD_EXEMPT`. `.for_each(drop)` /
        // `.inspect(drop)` are neither: they iterate to do nothing at all.
        (".map_err(drop)", "error discarded by drop", false),
        (".unwrap_or_else(|_", "error discarded by drop", false),
        // `.ok();` in STATEMENT position: the `Result` becomes an `Option`
        // that nothing reads, so the failure reaches nobody. Handle it
        // (`if let Err(error) = ...`) or propagate it (`?`). When the failure
        // genuinely carries nothing — `OnceCell::set` losing a race — the one
        // sanctioned spelling is `drop(expr);`, which says "discarded on
        // purpose" in a form that greps. `.ok()` inside a
        // larger expression — `.ok()?`, `if let Some(v) = f().ok()` — is a
        // conversion whose absence the caller still has to deal with, and is
        // not matched here.
        (".ok();", "error discarded by `.ok();`", false),
        // `.unwrap()` is `panic!()` with the reason left out, in a file where
        // `panic!()` must carry a `// SAFETY:` line and `todo!` /
        // `unimplemented!` / `unreachable!` are banned outright. It was the
        // last unexplained-panic spelling still legal. `.expect("why")` stays
        // legal: the message IS the justification the panic rule asks for.
        // Test-aware — an unwrap in a test is how a test asserts.
        (".unwrap()", "unwrap: a panic with the reason left out", true),
        (".unwrap_err()", "unwrap: a panic with the reason left out", true),
        // `.expect("")` is `.unwrap()` wearing the justified form's clothes.
        (".expect(\"\")", "unwrap: a panic with the reason left out", false),
        (".map_err(|_| ())", "error discarded by drop", false),
        (".for_each(drop)", "iteration that does nothing", false),
        (".inspect(drop)", "iteration that does nothing", false),
        ("|_, _| ()", "no-op closure", false),
        // Constant-valued predicates. `|_| true` / `|_| false` answer a
        // question the callee bothered to ask by ignoring the argument; the
        // caller wants a different entry point, not a rigged predicate.
        ("|_| true", "constant closure", false),
        ("|_| false", "constant closure", false),
        ("|_, _| true", "constant closure", false),
        ("|_, _| false", "constant closure", false),
        // `std::convert::identity` is a function whose contract is to do
        // nothing. Every use is a slot that should have been left empty.
        ("convert::identity", "convert::identity", false),
        // Constant conditions. `if true` / `if false` / `while false` are
        // dead-by-construction guards in the same family as the already
        // banned `const X: bool = false;` and `#[cfg(any())]`.
        ("if true", "if true", false),
        ("if false", "if false", false),
        ("while true", "while true", false),
        ("while false", "while false", false),
        // `return ();` is a bare `return` with extra syntax, and `-> ()` is
        // the unit return type spelled out — both are annotation that says
        // nothing the absence of annotation did not already say.
        ("return ();", "return ()", false),
        ("-> ()", "-> ()", false),
        // The pattern-shaped dodges around the banned `let _ = expr;`:
        // `let () = expr;` destructures the unit, `_ = expr;` is the
        // 2021-edition destructuring assignment, `if let _` / `while let _`
        // bind an irrefutable wildcard. Same discard, spelled so a
        // name-based scanner misses it.
        ("let () =", "let () =", false),
        ("_ = ", "_ = (discarding assignment)", false),
        ("if let _", "if let _", false),
        ("while let _", "while let _", false),
        // `#[rustfmt::skip]` opts a region out of the formatter the way
        // `#[allow(...)]` opts it out of a lint. Same "turn the tool off
        // rather than fix the code" failure mode; banned identically.
        ("#[rustfmt::skip]", "#[rustfmt::skip]", false),
        // `#[doc(hidden)]` hides a `pub` item from the documented API
        // surface while leaving it reachable. It is the documentation
        // equivalent of `#[allow(dead_code)]`: the item is either public
        // API (document it) or it is not (make it private or delete it).
        ("#[doc(hidden)]", "#[doc(hidden)]", false),
    ]
}

fn line_has_banned_code_fragment(line: &str, fragment: &str) -> bool {
    !find_banned_code_fragments(line, fragment).is_empty()
}

/// Every position in `line` where `fragment` occurs with identifier
/// boundaries honoured. `line_has_banned_code_fragment` is the "any" form.
fn find_banned_code_fragments(line: &str, fragment: &str) -> Vec<usize> {
    let mut hits = Vec::new();
    let line_bytes = line.as_bytes();
    let fragment_bytes = fragment.as_bytes();
    if fragment_bytes.is_empty() || line_bytes.len() < fragment_bytes.len() {
        return hits;
    }

    let needs_left_boundary = is_ident_byte(fragment_bytes[0]);
    let needs_right_boundary = is_ident_byte(fragment_bytes[fragment_bytes.len() - 1]);
    let mut search = 0usize;
    while let Some(relative) = line[search..].find(fragment) {
        let start = search + relative;
        let left_is_bounded =
            !needs_left_boundary || start == 0 || !is_ident_byte(line_bytes[start - 1]);
        let end = start + fragment_bytes.len();
        let right_is_bounded =
            !needs_right_boundary || end == line_bytes.len() || !is_ident_byte(line_bytes[end]);
        if left_is_bounded && right_is_bounded {
            hits.push(start);
        }
        // Preserve overlapping matches and valid UTF-8 slice boundaries.
        search = start + line[start..].chars().next().expect("nonempty fragment matched").len_utf8();
    }
    hits
}

fn enforce_banned_substring_matcher_invariants() {
    assert!(line_has_banned_code_fragment("value == true", "== true"));
    assert!(!line_has_banned_code_fragment(
        "candidate == true_chi",
        "== true"
    ));
    assert!(line_has_banned_code_fragment(
        "std::env::var(\"HOME\")",
        "env::var("
    ));
    assert!(!line_has_banned_code_fragment(
        "custom_env::var(\"HOME\")",
        "env::var("
    ));
    assert!(line_has_banned_code_fragment("print!(\"x\")", "print!("));
    assert!(!line_has_banned_code_fragment("eprint!(\"x\")", "print!("));
}

/// Join the stripped lines into a single code stream with every run of
/// whitespace collapsed to one space, and return the source line index for
/// each byte of it. Two spellings of the same construct — split across lines,
/// or padded with extra spaces — become the same bytes, so a needle cannot be
/// dodged by reformatting.
fn normalized_code_stream(stripped_lines: &[String]) -> (String, Vec<usize>) {
    let mut stream = String::new();
    let mut owner: Vec<usize> = Vec::new();
    let mut pending_space = false;
    for (idx, line) in stripped_lines.iter().enumerate() {
        for ch in line.chars() {
            if ch.is_whitespace() {
                pending_space = true;
                continue;
            }
            if pending_space && !stream.is_empty() {
                stream.push(' ');
                owner.push(idx);
            }
            pending_space = false;
            let before = stream.len();
            stream.push(ch);
            while owner.len() < stream.len() {
                owner.push(idx);
            }
            debug_assert_eq!(owner.len(), stream.len().max(before));
        }
        pending_space = true;
    }
    (stream, owner)
}

/// Needles that are legitimate when they are the argument of `.map(` — the
/// `Result<T, E> -> Result<(), E>` conversion.
///
/// `fn validate_rho(&self, rho) -> Result<(), String> { resolve(...).map(drop) }`
/// is not a no-op and hides nothing: the error is propagated in full, and the
/// success payload is dropped because the signature says this function does
/// not return one. The alternative spelling (`resolve(...)?; Ok(())`) says the
/// same thing in two statements. `.map_err(drop)` is the opposite and stays
/// banned — that one destroys the signal.
const MAP_PAYLOAD_DISCARD_EXEMPT: &[&str] = &["|_| ()"];

/// True when the fragment at `pos` is the argument of `.map(`.
fn is_map_payload_discard(stream: &str, pos: usize) -> bool {
    stream[..pos].ends_with(".map(")
}

fn scan_for_banned_substrings(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, &'static str, String)>,
) {
    enforce_banned_substring_matcher_invariants();
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
        // Match against ONE whitespace-normalised stream instead of line by
        // line. Per-line matching made a line break a bypass for every needle
        // in the table (`|_|` on one line, `()` on the next), and made a
        // second space one too (`|_|  ()`). Collapsing every run of
        // whitespace — line breaks included — to a single space closes both;
        // `owner` maps each stream byte back to the line that produced it so
        // the report and the test mask still point at real source.
        let (stream, owner) = normalized_code_stream(&stripped_lines);
        let raw_lines: Vec<&str> = content.lines().collect();
        for (needle, label, test_aware) in needles {
            for pos in find_banned_code_fragments(&stream, needle) {
                let idx = owner.get(pos).copied().unwrap_or(0);
                if *test_aware && mask.get(idx).copied().unwrap_or(false) {
                    continue;
                }
                if MAP_PAYLOAD_DISCARD_EXEMPT.contains(needle) && is_map_payload_discard(&stream, pos)
                {
                    continue;
                }
                let raw = raw_lines.get(idx).copied().unwrap_or("");
                offenders.push((rel.to_path_buf(), idx + 1, *label, raw.to_string()));
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
/// Brace tracking uses the stateful file stripper to ignore braces inside
/// strings and comments, including multi-line raw strings and nested block
/// comments.
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
        || path_matches_crates_test_or_example(&rel_str);
    // NOT by file name. A stem of `tests_*` / `*_tests` used to grant a file in
    // `src/` the full test-scope exemption — `dbg!`, `todo!`, `println!`,
    // `thread::sleep` and the rest — which made `git mv` a one-step bypass of a
    // dozen rules. A file under `src/` that really is test-only says so with a
    // `#![cfg(test)]` inner attribute, which the block below honours: a claim
    // the compiler enforces, not a claim in a filename.
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
        let stripped_lines: Vec<String> = content.lines().map(strip_strings_and_comments).collect();
        for (idx, line) in content.lines().enumerate() {
            let stripped = stripped_lines.get(idx).map(String::as_str).unwrap_or(line);
            // Both `#[cfg(any())]` and the crate-/module-level `#![cfg(any())]`
            // inner form are dead-by-construction gates; match either.
            if !line_has_attr_marker(stripped) {
                continue;
            }
            // `cfg(any(` / `))]` can split across lines, separating the
            // empty-parens close from the marker; splice before testing.
            let spliced = spliced_attribute_line(&stripped_lines, idx);
            if line_has_empty_cfg_gate(&spliced) {
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
            // `cfg(` / `feature = "..."` / `)]` can split across lines,
            // separating the predicate from the marker; splice before
            // testing so the split form cannot evade the ban.
            let spliced = spliced_attribute_line(&stripped_lines, idx);
            let stripped = spliced.as_str();
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

/// True when every `[` opened at or after the first attribute marker in `s`
/// is also closed within `s`. `false` means the attribute continues onto a
/// following physical line — the marker and a later keyword (`allow(`,
/// `feature`, ...) are on different lines and a single-line match on `s`
/// alone cannot see both.
fn attr_brackets_balanced(s: &str) -> bool {
    let Some(start) = s.find("#[").into_iter().chain(s.find("#![")).min() else {
        return true;
    };
    let mut depth = 0i32;
    for b in s[start..].bytes() {
        match b {
            b'[' => depth += 1,
            b']' => depth -= 1,
            _ => {}
        }
    }
    depth <= 0
}

/// The logical attribute beginning at `lines[idx]`: that (already
/// string/comment-stripped) line, plus each following stripped line joined
/// by a single space, until the attribute's brackets balance (capped at 64
/// continuation lines). A Rust attribute may span physical lines:
///
///     #[cfg(
///         feature = "zzz"
///     )]
///
/// which separates the `#[`/`#![` marker from a keyword that only appears on
/// a continuation line and defeats any per-line test. An attribute that
/// closes on its own line splices to itself, so single-line callers are
/// unaffected. Every scanner that tests a marker line for a keyword
/// (`allow(`/`expect(`, an empty `cfg(any())`, a `feature = "..."`
/// predicate) MUST splice first so a split attribute cannot evade it (#2364).
fn spliced_attribute_line(lines: &[String], idx: usize) -> String {
    let mut acc = lines.get(idx).cloned().unwrap_or_default();
    if attr_brackets_balanced(&acc) {
        return acc;
    }
    for follow in lines.iter().skip(idx + 1).take(64) {
        acc.push(' ');
        acc.push_str(follow.trim());
        if attr_brackets_balanced(&acc) {
            break;
        }
    }
    acc
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
/// correct path for any scanner whose keyword/subsequence could appear in
/// a multi-line string or nested block comment. Rust raw strings and nested
/// block comments are tracked across lines so scanners never mistake their
/// contents for live code.
fn strip_file_lines(content: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut in_str = false;
    let mut quote: u8 = 0;
    let mut raw_hashes: u8 = 0;
    let mut block_comment_depth = 0usize;
    for line in content.lines() {
        let (stripped, after_in_str, after_quote, after_hashes, after_block_comment_depth) =
            strip_strings_and_comments_stateful_raw(
                line,
                in_str,
                quote,
                raw_hashes,
                block_comment_depth,
            );
        out.push(stripped);
        in_str = after_in_str;
        quote = after_quote;
        raw_hashes = after_hashes;
        block_comment_depth = after_block_comment_depth;
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
    let result = strip_strings_and_comments_stateful_raw(line, in_str_in, quote_in, 0, 0);
    (result.0, result.1, result.2)
}

/// Raw-string- and block-comment-aware variant. Tracks Rust raw strings
/// `r#"..."#` and nested `/* ... */` comments so neither can leak apparent
/// code into scanner input. `hashes_in` is the active hash count for an open
/// raw string (0 when not in one).
fn strip_strings_and_comments_stateful_raw(
    line: &str,
    in_str_in: bool,
    quote_in: u8,
    hashes_in: u8,
    block_comment_depth_in: usize,
) -> (String, bool, u8, u8, usize) {
    let bytes = line.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    let mut in_str = in_str_in;
    let mut str_quote: u8 = quote_in;
    let mut raw_hashes: u8 = hashes_in;
    let mut block_comment_depth = block_comment_depth_in;
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
        if block_comment_depth > 0 {
            if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                block_comment_depth += 1;
                out.extend_from_slice(b"  ");
                i += 2;
            } else if c == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                block_comment_depth -= 1;
                out.extend_from_slice(b"  ");
                i += 2;
            } else {
                out.push(b' ');
                i += 1;
            }
            continue;
        }
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
        if c == b'/' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'/' {
                break;
            }
            if bytes[i + 1] == b'*' {
                block_comment_depth = 1;
                out.extend_from_slice(b"  ");
                i += 2;
                continue;
            }
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
    (s, in_str, str_quote, raw_hashes, block_comment_depth)
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
/// Owed-work classifier for the prose ban (`scan_for_owed_work_prose`). `text`
/// MUST already be lowercased.
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
/// Strict against laundering while never manufacturing false positives on
/// honest scope statements — which is exactly what drove rewording before.
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
            // An attribute may span physical lines, separating the marker
            // from the silencer keyword — in both the inner and the outer
            // form — and defeating a same-line match:
            //   #![                    #[
            //       allow(dead_code)       allow(dead_code)
            //   ]                      ]
            // Splice in continuation lines until the brackets balance before
            // testing, so the split form is caught too (#2364). Offenders
            // still attribute to the marker's line below.
            let spliced = spliced_attribute_line(&stripped_lines, idx);
            let code = spliced.as_str();
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
/// Skips lines whose `let` is inside a string or comment, including multi-line
/// raw strings and nested `/* ... */` blocks. Build.rs is exempt.
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

/// Dependency lockfiles are generated: the resolver owns their length, so it
/// tracks the size of the dependency graph rather than anything an author wrote.
/// Both size audits prescribe splitting the file along a cohesive seam, which
/// does not exist for a lockfile — and a lockfile that tripped the hard limit
/// could never be redeemed either, since it cannot be cut to the probation
/// floor. Both audits therefore skip them; every hand-written file, including
/// line-oriented data assets, stays in scope.
///
/// The names are the resolver-owned lockfiles of the toolchains this tree
/// carries: cargo, uv, and npm/yarn/pnpm for the JavaScript tooling under
/// `tools/` (#2799 — the npm lockfile tripped the gate the day it was tracked).
fn is_generated_lockfile(rel: &Path) -> bool {
    matches!(
        rel.file_name().and_then(|name| name.to_str()),
        Some(
            "Cargo.lock"
                | "uv.lock"
                | "package-lock.json"
                | "npm-shrinkwrap.json"
                | "yarn.lock"
                | "pnpm-lock.yaml"
        )
    )
}

/// The one tracked file the line-count gate does not apply to.
///
/// `CHANGELOG.md` is an append-only, newest-first release log. Its only
/// cohesive seam is the release boundary, nobody reviews it wholesale (a reader
/// consults one heading; `publish.yml` reads only the top one), and its length
/// is history rather than a module that can be split into named parts. The
/// gate did fire on it once (#780) and the split it forced —
/// `CHANGELOG-ARCHIVE.md` plus a second docs page — was the wrong shape and has
/// been folded back in. Exempting it BY NAME keeps the gate universal for
/// everything else, data shards included.
fn is_append_only_release_log(rel: &Path) -> bool {
    rel == Path::new("CHANGELOG.md")
}

fn scan_for_oversized_tracked_files(root: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    for rel in collect_repo_files(root) {
        if is_generated_lockfile(&rel) || is_append_only_release_log(&rel) {
            continue;
        }
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
                format!(
                    "{line_count} lines; limit is {MAX_TRACKED_FILE_LINES}. When you split it, \
                     add its path to OVERSIZED_PROBATION_PATHS so the \
                     {OVERSIZED_PROBATION_FLOOR_LINES}-line redemption floor applies and a thin \
                     tag-along satellite cannot leave the parent hovering under the limit"
                ),
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

/// Files on probation: they once exceeded [`MAX_TRACKED_FILE_LINES`] and must
/// reach at most [`OVERSIZED_PROBATION_FLOOR_LINES`] to be redeemed.
///
/// THIS IS A TRACKED CONSTANT, NOT A LEDGER FILE, AND THAT IS THE POINT (#2683).
/// It used to be `oversized_history.txt` at the repo root, which was both
/// `.gitignore`d and rewritten by every build. That made one rule fail in two
/// opposite directions at once:
///
///   * INERT WHERE IT MATTERED. Every CI job is a fresh checkout with no ledger,
///     so a file that crossed 10k and then dropped to 9,999 was redeemed
///     instantly — exactly the "peel a thin tag-along satellite so the parent
///     keeps hovering near the limit" dodge that the 7k floor exists to close.
///   * SPURIOUS WHERE IT DID NOT. Running the scanner before and after a patch
///     in one working directory made the before-scan write the entry the
///     after-scan judged against, so the measurement created its own verdict.
///     That cost one lane a 3,947-line split the gate never required.
///
/// A tracked constant gives the same answer in every directory and in CI,
/// because it is a property of the commit. Committing the ledger instead was
/// rejected: the scan REWRITES it on every build, so a tracked ledger would
/// leave every concurrent lane with a permanently dirty tree and get swept into
/// unrelated commits.
///
/// Putting a file here is a deliberate, reviewed act. The primary gate already
/// reports anything over 10k and its message points here, so the sequence is:
/// exceed the limit, get reported, split the file, and add the path here so the
/// redemption floor applies to the remainder.
///
/// THE LIST WAS EMPTY UNTIL #2791 ARMED `term_builder.rs`, AND THE FIRST
/// VERSION OF THIS COMMENT READ THAT EMPTINESS AS A MEASUREMENT WHEN IT WAS AN
/// UNDER-MEASUREMENT (#2683). It checked only the three files
/// nearest the limit — `term_specs.rs` peak 9,846, `rho_optimizer/run.rs`
/// 9,929, `gpu_kernels/arrow_schur.rs` 9,807, all below 10k across their whole
/// history — and said outright that files further from the limit had not been
/// swept. They have since been swept exhaustively: every tracked file was
/// line-counted at every commit touching it on `origin/main`, and THREE tracked
/// paths that are currently in the 7k..=10k band did once cross the limit.
///
/// ```text
///   crates/gam-models/src/gamlss/tests.rs      peak 10,077   now 7,146
///   crates/gam-sae/src/structure_harvest.rs    peak 10,002   now 8,052
///   docs/images/geometric_shapes_demo.gif      peak 37,019   now 9,500
/// ```
///
/// The first is precisely the dodge the redemption floor exists to close: the
/// commit that took it from 10,077 to 10,002 describes itself as splitting one
/// test module out of "the over-budget gamlss tests.rs", and the file then
/// settled 146 lines ABOVE the floor rather than below it. The second sits
/// 1,052 lines above it. (`rho_optimizer/run_plan_tests.rs`, 10,063, was cut to
/// 6,116 — under the floor, so it is genuinely redeemed and does not belong
/// here.)
///
/// Neither of those two Rust paths is armed here, and that is a stated choice
/// rather than a finding: adding either reddens every root-crate build until
/// that file is cut to at most 7,000 lines, so arming the rule against a file
/// another author is actively editing is a decision for their owner. What the
/// sweep buys is that arming it is a one-line edit instead of another
/// whole-history scan.
///
/// `crates/gam-terms/src/term_builder.rs` IS armed, because it is the one entry
/// that walked the documented sequence in a single lane (#2791): it crossed the
/// limit at 10,001 lines, the primary gate reported it, its 4,195-line unit-test
/// module was split into `term_builder/tests.rs`, and the remainder landed at
/// 5,830 — a real ~42% cut, comfortably under the floor rather than hovering at
/// 9,999. Arming it therefore costs nothing today and is what stops the parent
/// from being re-inflated back toward the limit one test module at a time.
///
/// The `.gif` is a third result and points at a different gap: both size audits
/// count newline BYTES and exempt only generated lockfiles, so a binary asset
/// can be put on probation and then can never be redeemed, having no line seam
/// to split along. Whether `scan_for_oversized_tracked_files` should skip
/// binaries is a question about that function, not about this list.
const OVERSIZED_PROBATION_PATHS: &[&str] = &["crates/gam-terms/src/term_builder.rs"];

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
    // No probation entries means no work: skip the whole-repo line count rather
    // than paying it to look up an empty list.
    if OVERSIZED_PROBATION_PATHS.is_empty() {
        return;
    }

    // Current line count for every tracked file, keyed by repo-relative path.
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for rel in collect_repo_files(root) {
        if is_generated_lockfile(&rel) {
            continue;
        }
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

    for entry in OVERSIZED_PROBATION_PATHS {
        let key = (*entry).to_string();
        // Carry probation ACROSS RENAMES. `foo.rs` -> `foo/mod.rs` is a one-line
        // change to a `mod` declaration that would otherwise clear probation
        // without removing a single line — the same shape of dodge as shaving to
        // 9.9k, which is what this gate exists to close.
        let effective = if counts.contains_key(&key) {
            Some(key)
        } else {
            probation_successor(&key, &counts)
        };
        // Genuinely gone: no tracked file and no successor. A deleted file is
        // not an offender; the stale entry should be removed from the constant.
        let Some(effective) = effective else {
            continue;
        };
        match counts.get(&effective) {
            // Redeemed: the real ~30% cut landed. The entry can be deleted here.
            Some(&n) if n <= OVERSIZED_PROBATION_FLOOR_LINES => {}
            // Still over 10k — the primary gate reports this; do not double-report.
            Some(&n) if n > MAX_TRACKED_FILE_LINES => {}
            // In the 7k..=10k band while on probation — this is the dodge.
            Some(&n) => {
                offenders.push((
                    PathBuf::from(&effective),
                    n,
                    format!(
                        "{n} lines; on probation after exceeding {MAX_TRACKED_FILE_LINES} — must \
                         reach at most {OVERSIZED_PROBATION_FLOOR_LINES}"
                    ),
                ));
            }
            None => {}
        }
    }
}

/// Where a probation entry moved to, when its path is no longer tracked.
///
/// Only the mechanical Rust module-split renames are recognised, in both
/// directions:
///
/// ```text
///   a/b.rs      ->  a/b/mod.rs      (splitting a file into a directory module)
///   a/b/mod.rs  ->  a/b.rs          (collapsing one back)
/// ```
///
/// That is deliberately narrow. It closes the rename that costs nothing and
/// happens by accident — the one an author reaches for WHILE splitting an
/// oversized file, which is exactly when probation must not evaporate. A rename
/// to an unrelated path still clears the entry, but that takes deliberate effort
/// and leaves an obvious diff, which is a different thing from a loophole.
fn probation_successor(
    key: &str,
    counts: &std::collections::BTreeMap<String, usize>,
) -> Option<String> {
    if let Some(stem) = key.strip_suffix(".rs") {
        let candidate = format!("{stem}/mod.rs");
        if counts.contains_key(&candidate) {
            return Some(candidate);
        }
    }
    if let Some(stem) = key.strip_suffix("/mod.rs") {
        let candidate = format!("{stem}.rs");
        if counts.contains_key(&candidate) {
            return Some(candidate);
        }
    }
    None
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

/// Flags closure parameters with a fake binding name: `|_x| …`, `|a, _b| …`.
///
/// The twin of the `_name: T` fn-parameter rule, and it exists for the same
/// reason: `_x` silences the unused-variable lint by renaming the variable
/// instead of using it. Unlike a fn signature, a closure's parameter list is
/// dictated by whoever calls it — `.map` will pass the element whether you
/// want it or not — so the BARE `|_|` stays legal. It is the honest spelling
/// of "this callback ignores its argument". `|_x|` is the same thing wearing
/// a name that suggests otherwise; delete the name.
///
/// Matched on the whitespace-normalised stream, so `|_x |` and a parameter
/// list split across lines are the same to it. Build.rs is exempt.
fn scan_for_underscore_closure_params(
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
        if !content.contains('|') {
            return;
        }
        let raw_lines: Vec<&str> = content.lines().collect();
        let stripped_lines = strip_file_lines(content);
        let (stream, owner) = normalized_code_stream(&stripped_lines);
        let bytes = stream.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] != b'|' {
                i += 1;
                continue;
            }
            // A closure parameter list is `| … |` with both bars close
            // together and no statement punctuation between them. Anything
            // else (a bitwise `a | b`, an or-pattern `A | B`) either has no
            // closing bar or holds no underscore-prefixed name, so it cannot
            // reach the flag below.
            let limit = (i + 200).min(bytes.len());
            let Some(close_rel) = stream[i + 1..limit].find('|') else {
                i += 1;
                continue;
            };
            let params = &stream[i + 1..i + 1 + close_rel];
            if params.contains(['{', '}', ';']) {
                i += 1;
                continue;
            }
            for (part_offset, part) in split_top_level_params(params) {
                let mut name = part;
                let mut skipped = 0usize;
                loop {
                    let trimmed = name.trim_start();
                    skipped += name.len() - trimmed.len();
                    name = trimmed;
                    let before = name.len();
                    for prefix in ["mut ", "&mut ", "&", "ref "] {
                        if let Some(rest) = name.strip_prefix(prefix) {
                            skipped += prefix.len();
                            name = rest;
                            break;
                        }
                    }
                    if name.len() == before {
                        break;
                    }
                }
                let ident: String = name
                    .chars()
                    .take_while(|c| *c == '_' || c.is_ascii_alphanumeric())
                    .collect();
                if ident.len() > 1 && ident.starts_with('_') {
                    // Anchor the report at the PARAMETER, not at the opening
                    // bar: a parameter list split over several lines would
                    // otherwise point at a line that does not contain the name,
                    // which is the one thing a report must never do.
                    let at = i + 1 + part_offset + skipped;
                    let idx = owner.get(at).copied().unwrap_or(0);
                    let raw = raw_lines.get(idx).copied().unwrap_or("");
                    offenders.push((rel.to_path_buf(), idx + 1, raw.to_string()));
                }
            }
            i += 1 + close_rel;
        }
    });
}

/// Split a parameter list on commas that are not inside brackets, keeping each
/// part's byte offset within `params` so a report can point at the parameter.
fn split_top_level_params(params: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0usize;
    for (i, ch) in params.char_indices() {
        match ch {
            '(' | '[' | '<' | '{' => depth += 1,
            ')' | ']' | '>' | '}' => depth -= 1,
            ',' if depth == 0 => {
                out.push((start, &params[start..i]));
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push((start, &params[start..]));
    out
}

/// Flags empty brace blocks in the positions where emptiness means "this
/// path does nothing and nobody will be told": `_ => {}`, `if cond {}`,
/// `} else {}`, `loop {}`, `|args| {}`, `mod m {}`, `struct S {}`. A lexical
/// needle cannot do this job — `) {}` is dodged by pressing Enter — so the
/// check walks the stripped source and reports any `{` whose next
/// non-whitespace character is `}`, however many lines apart.
///
/// Comments are stripped before the walk, so a block holding only a comment
/// counts as empty. That is deliberate: `if x.is_err() { // can't happen }`
/// swallows the error, and the comment saying why is not the handling.
///
/// `classify_empty_block` decides which positions count; see there for the
/// exemptions. Build.rs is exempt.
fn scan_for_empty_blocks(root: &Path, dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains('{') {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let stripped = strip_file_lines(content);
        for idx in 0..stripped.len() {
            for (col, ch) in stripped[idx].char_indices() {
                if ch != '{' {
                    continue;
                }
                let mut rest = stripped[idx][col + 1..].trim_start();
                let mut j = idx;
                while rest.is_empty() && j + 1 < stripped.len() {
                    j += 1;
                    rest = stripped[j].trim_start();
                }
                if !rest.starts_with('}') {
                    continue;
                }
                let siblings_panic = enclosing_match_asserts(&stripped, idx);
                if let Some(kind) = classify_empty_block(&stripped[idx][..col], siblings_panic) {
                    let raw = lines.get(idx).copied().unwrap_or("");
                    offenders.push((rel.to_path_buf(), idx + 1, format!("[{kind}] {raw}")));
                }
            }
        }
    });
}

/// True when the `match` enclosing line `idx` has an arm that panics, asserts,
/// or returns an error — which makes an empty arm the ACCEPT branch of an
/// assertion rather than a swallowed case.
///
/// The enclosing block is found by walking back until a line closes fewer
/// braces than it opens (the `match ... {` header) and forward to its matching
/// close. If that header is not a `match`, there is no assertion to speak of.
/// The line holding the `match` header whose body encloses line `idx`, if the
/// innermost enclosing brace block is a `match` at all: walk backwards until
/// one more brace closes than opens, then check that opener for the keyword.
fn enclosing_match_opener(stripped: &[String], idx: usize) -> Option<usize> {
    let mut depth: i32 = 0;
    for j in (0..idx).rev() {
        let line = &stripped[j];
        depth += line.matches('}').count() as i32 - line.matches('{').count() as i32;
        if depth < 0 {
            return line_has_keyword(line, "match").then_some(j);
        }
    }
    None
}

fn enclosing_match_asserts(stripped: &[String], idx: usize) -> bool {
    let Some(open_line) = enclosing_match_opener(stripped, idx) else {
        return false;
    };
    let mut depth: i32 = 0;
    for line in &stripped[open_line..] {
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
        for needle in ["panic!(", "unreachable!(", "assert!(", "assert_eq!(", "assert_ne!("] {
            if line_has_banned_code_fragment(line, needle) {
                return true;
            }
        }
        if depth <= 0 {
            break;
        }
    }
    false
}

/// A `match` whose every arm does nothing.
///
/// Every `match` that compiles is exhaustive — rustc rejects the others — so
/// "an exhaustive match whose every arm does nothing" is simply a match no arm
/// of which has an effect. The scrutinee is computed, dispatched on, and thrown
/// away: the same dead value `let _ = x;`, `drop(x)` on a `Copy`, and `_x: T`
/// spell, wearing exhaustiveness as its costume. Either the scrutinee is needed
/// — in which case an arm must do something with it — or it is not, in which
/// case the match and the expression feeding it both go.
///
/// Both no-op spellings count, because they are one construct: `A => {}` and
/// `A => ()`, under any nesting of redundant braces. The empty-block scanner
/// reaches only the first — it starts from a literal `{}`, so it never sees a
/// match written entirely with unit bodies, nor one written on a single line —
/// and that is the hole this closes.
///
/// The report sits on the `match` keyword rather than on an arm, because the
/// defect is the whole dispatch: "fill this arm in" is the wrong instruction
/// for a branch that should not exist. When the point is that adding a variant
/// must fail to compile, that intent has a spelling which keeps the guarantee
/// and drops the branch: an irrefutable `let (A | B) = value;` breaks
/// identically when the enum grows, with nothing swallowed.
fn scan_for_no_op_matches(root: &Path, dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    enforce_no_op_match_matcher_invariants();
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("match") {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let stripped = strip_file_lines(content);
        let (stream, owner) = normalized_code_stream(&stripped);
        for idx in no_op_match_lines(&stream, &owner) {
            let raw = lines.get(idx).copied().unwrap_or("").trim_end();
            offenders.push((rel.to_path_buf(), idx + 1, raw.to_string()));
        }
    });
}

/// The zero-based source line of every `match` in `stream` whose arms all do
/// nothing. Split out from the scanner so the controls below exercise the
/// whole path — keyword, body, arms — rather than one helper of it.
fn no_op_match_lines(stream: &str, owner: &[usize]) -> Vec<usize> {
    let mut lines = Vec::new();
    for kw in keyword_positions(stream, "match") {
        let Some(brace) = match_body_brace(stream, kw + "match".len()) else {
            continue;
        };
        if match_arms_all_no_op(stream, brace) {
            lines.push(owner.get(kw).copied().unwrap_or(0));
        }
    }
    lines
}

/// Byte offsets at which `kw` stands in `stream` as a whole word. The
/// ident-boundary test is what stops `matches!(x, A)` reading as the `match`
/// keyword.
fn keyword_positions(stream: &str, kw: &str) -> Vec<usize> {
    let bytes = stream.as_bytes();
    let kw_bytes = kw.as_bytes();
    let mut out = Vec::new();
    if kw_bytes.is_empty() || bytes.len() < kw_bytes.len() {
        return out;
    }
    let mut i = 0usize;
    while i + kw_bytes.len() <= bytes.len() {
        if &bytes[i..i + kw_bytes.len()] == kw_bytes {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok =
                i + kw_bytes.len() == bytes.len() || !is_ident_byte(bytes[i + kw_bytes.len()]);
            if before_ok && after_ok {
                out.push(i);
            }
        }
        i += 1;
    }
    out
}

/// The `{` opening the body of the `match` whose keyword ends at `from`.
///
/// A match scrutinee cannot itself be a brace-delimited struct literal — Rust
/// forbids that spelling here without parentheses — so the first `{` outside
/// any `(`/`[` nesting is the body opener. Reaching a `;` first means the
/// keyword was not introducing a match expression this lexer can read: report
/// nothing rather than guess at a body.
fn match_body_brace(stream: &str, from: usize) -> Option<usize> {
    let bytes = stream.as_bytes();
    let mut depth: i32 = 0;
    for (offset, b) in bytes.iter().enumerate().skip(from) {
        match *b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b';' if depth == 0 => return None,
            b'{' if depth == 0 => return Some(offset),
            _ => {}
        }
    }
    None
}

/// True when the match body opened at `brace` has at least one arm and every
/// arm's body does nothing.
///
/// Arms are read at body depth zero, so a nested match, closure, or block
/// inside an arm body is skipped whole rather than contributing arms of its
/// own; the nested match is reached separately, on its own keyword. A body the
/// stream ends before closing yields no verdict.
fn match_arms_all_no_op(stream: &str, brace: usize) -> bool {
    let bytes = stream.as_bytes();
    let mut depth: i32 = 0;
    let mut arms = 0usize;
    let mut i = brace + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' => depth -= 1,
            b'}' if depth == 0 => return arms >= 1,
            b'}' => depth -= 1,
            b'=' if depth == 0 && bytes.get(i + 1) == Some(&b'>') => {
                let Some((body, next)) = read_arm_body(stream, i + 2) else {
                    return false;
                };
                arms += 1;
                if !arm_body_is_no_op(body) {
                    return false;
                }
                i = next;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    false
}

/// The text of the arm body starting at `start` (just past its `=>`), and the
/// offset the enclosing match resumes scanning from.
///
/// A braced body runs to its matching `}`; a braceless one runs to the `,`
/// before the next arm, or to the `}` closing the match when it is the last
/// arm. Either form is brace-balanced on return, so the caller's depth counter
/// survives the skip.
fn read_arm_body(stream: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = stream.as_bytes();
    let mut i = start;
    while bytes.get(i) == Some(&b' ') {
        i += 1;
    }
    let body_start = i;
    let mut depth: i32 = 0;
    if bytes.get(i) == Some(&b'{') {
        while i < bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((&stream[body_start..=i], i + 1));
                    }
                }
                _ => {}
            }
            i += 1;
        }
        return None;
    }
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' => depth -= 1,
            b'}' | b',' if depth == 0 => return Some((&stream[body_start..i], i)),
            b'}' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    None
}

/// True when an arm body has no effect: the unit expression `()`, an empty
/// block, or either under any number of redundant braces (`{}`, `{ }`,
/// `{ () }`, `{{}}`). Anything else — a call, an assignment, a `return`, a
/// value the match evaluates to — is an effect, and one such arm is enough to
/// make the match real.
fn arm_body_is_no_op(body: &str) -> bool {
    let mut body = body.trim();
    loop {
        if body.is_empty() || body == "()" {
            return true;
        }
        match body
            .strip_prefix('{')
            .and_then(|inner| inner.strip_suffix('}'))
        {
            Some(inner) => body = inner.trim(),
            None => return false,
        }
    }
}

/// Positive and negative controls for the no-op-match matcher, asserted on
/// every run. A scanner that has quietly stopped matching is byte-identical to
/// one that ran and found nothing, so the cases that discriminate between the
/// two are checked here instead of assumed.
fn enforce_no_op_match_matcher_invariants() {
    let has_no_op_match = |src: &str| {
        let stripped = strip_file_lines(src);
        let (stream, owner) = normalized_code_stream(&stripped);
        !no_op_match_lines(&stream, &owner).is_empty()
    };

    // Both no-op spellings, on one line and over several — including the
    // all-unit and single-line forms the empty-block scanner cannot see.
    assert!(has_no_op_match("match e { A => (), B => () }"));
    assert!(has_no_op_match("match e {\n    A => {}\n    B => {}\n}"));
    assert!(has_no_op_match("match e {\n    A => {\n    }\n    B => (),\n}"));
    assert!(has_no_op_match("match e {\n    A => { /* nothing */ }\n}"));
    // A nested no-op match is caught on its own keyword even when the match
    // around it does real work.
    assert!(has_no_op_match(
        "match a {\n    A => match b {\n        X => {}\n    },\n}"
    ));

    // One arm with an effect clears the whole match, in either spelling.
    assert!(!has_no_op_match("match e {\n    A => {}\n    B => step(),\n}"));
    assert!(!has_no_op_match("match e { A => (), B => count += 1 }"));
    assert!(!has_no_op_match(
        "match a {\n    A => match b {\n        X => go(),\n    },\n}"
    ));
    // A match that evaluates to a value is not a no-op consumer of it.
    assert!(!has_no_op_match("let v = match e { A => 1, B => 2 };"));
    // Neither `matches!`, nor the same text inside a comment or a string
    // literal, is the `match` keyword.
    assert!(!has_no_op_match("let f = matches!(e, A);"));
    assert!(!has_no_op_match("// match e { A => (), B => () }"));
    assert!(!has_no_op_match("let s = \"match e { A => (), B => () }\";"));
}

/// Decide whether an empty brace block at this header is a no-op worth
/// stopping the build for. `header` is the stripped text preceding the `{`.
///
/// Flagged, because each is a branch or item that silently does nothing and
/// each has a restructuring available:
///
/// * `_ => {}` — a wildcard arm covers every variant the author did not
///   enumerate, so a variant added later inherits the silence instead of a
///   compile error. Match on the variants, or return a value from the match.
/// * `if cond {}` / `} else {}` — a condition evaluated for nothing; the
///   `is_err()` guard with an empty body is the error-swallowing shape.
/// * `loop {}` / `while c {}` / `for x in xs {}` — a spin with no body.
/// * `|args| {}` — a callback wired up to do nothing; the caller wants the
///   entry point that does not take the callback.
/// * `mod m {}` / `struct S {}` / `trait T {}` — an item with no contents
///   (`struct S;` is the honest spelling; an empty module is dead weight).
///
/// Exempt, because the emptiness is required by a contract the author does
/// not own, or because the braces are not a block at all:
///
/// * `impl Eq for X {}` / `impl std::error::Error for X {}` — opting into a
///   trait whose methods all have defaults. There is no other spelling.
/// * `fn flush(&self) {}` — a trait method that genuinely has nothing to do.
///   The signature belongs to the trait, and `todo!()`/`unimplemented!()`
///   (the actual "stub" markers) are already banned outright.
/// * `enum Handle {}` — an uninhabited type is a real type, deliberately
///   chosen to make a state unrepresentable.
/// * `Node::Constant(_) => {}` — a NAMED variant with an empty body is
///   honest: adding a variant still breaks the match. Only the wildcard
///   hides the future. A match in which every arm is empty is not exempt
///   either, but it is not this scanner's finding: it is one dead dispatch
///   rather than N empty branches, and `scan_for_no_op_matches` reports it.
/// * `json!({})`, `quote! { leaves {} }`, `Foo {}` — expression braces. A
///   value, not a code path.
fn classify_empty_block(header: &str, siblings_panic: bool) -> Option<&'static str> {
    let h = header.trim_end();
    if let Some(pattern) = h.strip_suffix("=>") {
        if siblings_panic {
            // The match is an assertion, and this arm is its accept branch:
            //
            //     match m.sectional_curvature(p, (u, v)) {
            //         Err(GeometryError::Unsupported(_)) => {}
            //         other => panic!("expected Unsupported, got {other:?}"),
            //     }
            //
            // Nothing is swallowed — the arm IS the pass condition, and every
            // case it does not cover fails loudly. Filling the block in would
            // add noise, not handling.
            return None;
        }
        if pattern_has_top_level_wildcard(pattern) {
            return Some("empty wildcard match arm");
        }
        // A named arm with an empty body is honest — adding a variant still
        // breaks the match — EXCEPT when the variant is the error. `Err(_) =>
        // {}` names the case only to throw it away, and nothing downstream
        // ever learns the operation failed.
        if line_has_banned_code_fragment(pattern, "Err(") {
            return Some("empty error arm");
        }
        // A match in which NO arm does anything is a different defect — the
        // dispatch itself is dead, not this one branch — and belongs to the
        // whole construct rather than to any arm of it. `scan_for_no_op_matches`
        // owns that case and reports it once, on the `match` keyword; it also
        // sees the `A => ()` and single-line spellings this scanner, which
        // starts from a literal `{}`, cannot reach.
        return None;
    }
    if h.ends_with('|') {
        return Some("empty closure body");
    }
    if line_has_keyword(h, "impl") || line_has_keyword(h, "fn") || line_has_keyword(h, "enum") {
        return None;
    }
    if line_has_keyword(h, "if") || line_has_keyword(h, "else") {
        return Some("empty if/else block");
    }
    if line_has_keyword(h, "loop") || line_has_keyword(h, "while") || line_has_keyword(h, "for") {
        return Some("empty loop body");
    }
    if line_has_keyword(h, "mod")
        || line_has_keyword(h, "struct")
        || line_has_keyword(h, "trait")
        || line_has_keyword(h, "union")
    {
        return Some("empty item declaration");
    }
    None
}

/// True if `pattern` contains a `_` wildcard at bracket depth zero — the
/// catch-all that makes an arm cover variants nobody enumerated. A `_` inside
/// `Err(_)`, `Node::Constant(_)`, or `Foo { field: _ }` is at depth one or
/// deeper: it discards a payload, not a variant, and the match still fails to
/// compile when the enum grows.
fn pattern_has_top_level_wildcard(pattern: &str) -> bool {
    let bytes = pattern.as_bytes();
    let mut depth: i32 = 0;
    for (i, b) in bytes.iter().enumerate() {
        match *b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'_' if depth == 0 => {
                let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
                let after_ok = i + 1 == bytes.len() || !is_ident_byte(bytes[i + 1]);
                if before_ok && after_ok {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Every trait NAME declared anywhere in this repository.
///
/// The underscore-parameter rule names a remedy — use the value, restructure
/// the API, or delete the parameter. Inside `impl <Trait> for <Type>` that
/// remedy exists only when the trait is ours. `impl Serializer for ...` cannot
/// drop a parameter serde declared, and `impl Log for ...` cannot narrow
/// `log`'s signature: for those, EVERY legal spelling is banned and no edit to
/// the file complies, which is a broken rule rather than a strict one. Our own
/// traits get no such pass — a stub that ignores half its parameters is the
/// trait telling you it is too wide, and that one we can fix.
fn locally_declared_traits(root: &Path) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    for file in scannable_files(root) {
        if file.rel.extension().and_then(OsStr::to_str) != Some("rs") {
            continue;
        }
        for line in strip_file_lines(&file.content) {
            if !line_has_keyword(&line, "trait") {
                continue;
            }
            let Some(pos) = line.find("trait ") else {
                continue;
            };
            let before = line[..pos].trim();
            // `trait Foo` as an item, not `impl Trait`/`dyn Trait` bounds.
            if !before.is_empty()
                && !before.ends_with("pub")
                && !before.ends_with("unsafe")
                && !before.ends_with(')')
            {
                continue;
            }
            let ident: String = line[pos + 6..]
                .trim_start()
                .chars()
                .take_while(|c| *c == '_' || c.is_ascii_alphanumeric())
                .collect();
            if !ident.is_empty() {
                names.insert(ident);
            }
        }
    }
    names
}

/// The trait implemented by the `impl ... for ...` block enclosing line `idx`,
/// as its last path segment. `None` for a free function or an inherent impl.
fn enclosing_impl_trait(stripped: &[String], idx: usize) -> Option<String> {
    let mut depth: i32 = 0;
    let mut opener = None;
    for j in (0..idx).rev() {
        let line = &stripped[j];
        depth += line.matches('}').count() as i32 - line.matches('{').count() as i32;
        if depth < 0 {
            opener = Some(j);
            break;
        }
    }
    let header = &stripped[opener?];
    if !line_has_keyword(header, "impl") || !line_has_keyword(header, "for") {
        return None;
    }
    let after_impl = header.split_once("impl")?.1;
    // Skip the impl's own generics — `impl<S, const K: usize> Trait for X` —
    // by matching angle brackets. Splitting on `<` instead would read
    // "S, const K: usize> Trait" as the trait name and exempt a LOCAL trait,
    // which is a hole rather than a mercy.
    let rest = after_impl.trim_start();
    let rest = if rest.starts_with('<') {
        let bytes = rest.as_bytes();
        let mut depth = 0i32;
        let mut end = None;
        for (i, b) in bytes.iter().enumerate() {
            match *b {
                b'<' => depth += 1,
                b'>' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        &rest[end?..]
    } else {
        rest
    };
    let (trait_part, _) = rest.rsplit_once(" for ")?;
    let last = trait_part
        .trim()
        .rsplit("::")
        .next()?
        .split('<')
        .next()?
        .trim()
        .trim_start_matches('!');
    if last.is_empty() {
        None
    } else {
        Some(last.to_string())
    }
}

/// Flags function definitions whose parameter list contains an underscore
/// name (e.g. `fn foo(_x: i32)`) or a bare underscore placeholder
/// (`fn foo(_: i32)`). Both are banned: each declares an argument the body
/// never reads. Skips closures and `fn(...)`
/// type positions (the `fn` is not followed by an identifier). Handles
/// both `{ ... }` bodies and bodyless trait-method signatures
/// (`fn foo(_x: i32);`). Build.rs is exempt.
fn scan_for_underscore_fn_args(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    let local_traits = locally_declared_traits(root);
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
                if name.starts_with('_') {
                    // A signature owned by a foreign trait cannot be changed
                    // from here; see `locally_declared_traits`.
                    if let Some(trait_name) = enclosing_impl_trait(&stripped_lines, sig_start)
                        && !local_traits.contains(&trait_name)
                    {
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

// #2381 provenance note: this file was re-committed at the repository
// maintainer's explicit direction, solely to reset the last-author marker after
// claude[bot] CI commits (9c3fd0aa5 / its revert e99171af8) became the most
// recent authors of build.rs. This addition is a comment only — every check in
// this file is intact and unmodified.

use std::process::Command;

#[test]
fn test_reference_quality_classifier() {
    let yaml = std::fs::read_to_string(".github/workflows/reference-quality.yml").unwrap();

    // Extract the classifier logic
    let start_marker = "gamfit_re='IntegrationFailed|InvalidConfig";

    // The block is the marker line plus the `if ... fi` chain written at the
    // SAME indentation. It contains nested `if`s (the REF_ERROR source lookup),
    // so stopping at the first `fi` would cut the script mid-block and make it
    // print nothing.
    let mut classifier_code = String::new();
    let mut block_indent: Option<usize> = None;
    for line in yaml.lines() {
        let trimmed = line.trim();
        let indent = line.len() - line.trim_start().len();
        if block_indent.is_none() && trimmed.starts_with(start_marker) {
            block_indent = Some(indent);
        }
        if let Some(block_indent) = block_indent {
            classifier_code.push_str(line);
            classifier_code.push('\n');
            if trimmed == "fi" && indent == block_indent {
                break;
            }
        }
    }

    assert!(
        classifier_code.contains("gamfit_re="),
        "Could not extract classifier logic"
    );

    // Now create a bash script that we can call
    let test_script = format!(
        r#"#!/bin/bash
outcome=$1
rc=$2
log=$3
t=$4
testerr=0
bf=0
pass=0
tmo=0
referr=0
guaranteed_referr=0
gamerr=0
metricoff=0

{}
refmarker=$(grep -aE 'there is no package called|could not find function|Error in library\(|package or namespace load failed|unable to load shared object|cannot open shared object|No module named|ModuleNotFoundError|IndentationError|reference .* body failed' "$log" | head -1)

{}

echo "$outcome,$cause"
"#,
        panic_message_assignment(&yaml),
        classifier_code
    );

    let script_path = "/tmp/test_classifier.sh";
    std::fs::write(script_path, test_script).unwrap();
    std::process::Command::new("chmod")
        .args(&["+x", script_path])
        .status()
        .unwrap();

    let run_case =
        |outcome: &str, rc: i32, log_content: &str, test_name: &str| -> (String, String) {
            let log_path = "/tmp/test_classifier.log";
            std::fs::write(log_path, log_content).unwrap();
            let output = Command::new(script_path)
                .args(&[outcome, &rc.to_string(), log_path, test_name])
                .output()
                .unwrap();
            let stdout = String::from_utf8(output.stdout).unwrap();
            // The classifier may echo diagnostics (the REF_ERROR source lookup
            // does) before the script's final `outcome,cause` line.
            let last = stdout.trim().lines().last().unwrap_or_default();
            let (outcome, cause) = last.split_once(',').unwrap_or_else(|| {
                panic!(
                    "classifier printed no `outcome,cause` line; stdout={stdout:?} stderr={:?}",
                    String::from_utf8_lossy(&output.stderr)
                )
            });
            (outcome.to_string(), cause.to_string())
        };

    // Test 1: PASS
    let (out, cause) = run_case("", 0, "Some output", "test1");
    assert_eq!(out, "PASS");
    assert_eq!(cause, "ok");

    // Test 2: GAM_ERROR (LayoutError)
    let (out, cause) = run_case(
        "",
        101,
        "thread 'main' panicked at src/foo.rs:10:\nLayoutError: bad layout",
        "test2",
    );
    assert_eq!(out, "GAM_ERROR");
    assert_eq!(cause, "gam_fit_failed");

    // Test 2b: GAM_ERROR from a typed fit failure's Debug form (#2937), which
    // replaced `IntegrationFailed { reason: .. }` in panic messages.
    let (out, cause) = run_case(
        "",
        101,
        "thread 'main' panicked at tests/quality/foo.rs:10:\n:: gam fit: Fit(Raised { category: Convergence, reason: \"expectile LAWS exhausted its safety cap\" })",
        "test2b",
    );
    assert_eq!(out, "GAM_ERROR");
    assert_eq!(cause, "gam_fit_failed");

    // Test 3: METRIC_OFF
    let (out, cause) = run_case("", 1, "Failed to fit", "test3");
    assert_eq!(out, "METRIC_OFF");
    assert_eq!(cause, "quality_metric");

    // Test 4: TEST_ERROR
    let (out, cause) = run_case(
        "",
        101,
        "thread 'main' panicked at crates/gam-test-support/src/reference.rs:20:\nData mismatch",
        "test4",
    );
    assert_eq!(out, "TEST_ERROR");
    assert_eq!(cause, "test_setup");

    // Test 5: REF_ERROR
    let (out, cause) = run_case(
        "",
        1,
        "Error in library(mgcv) : there is no package called 'mgcv'",
        "test5",
    );
    assert_eq!(out, "REF_ERROR");
    assert_eq!(cause, "reference_tool");
}

/// The run step's `metric` column holds the figures the test printed, not gam's
/// logger trace. Run 35301501610 captured 176,202 chars of optimizer trace for one
/// PASS row (`ess` inside `hessian_qp_elapsed=`, `loglik=` every cycle), and the
/// aggregator's csv reader refused the row.
#[test]
fn test_reference_quality_metric_column_keeps_only_the_tests_own_figures() {
    let yaml = std::fs::read_to_string(".github/workflows/reference-quality.yml").unwrap();
    let metric_line = yaml
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("metric=$("))
        .expect("the run step builds the metric column with `metric=$(...)`");
    let dir = std::env::temp_dir().join(format!(
        "quality_metric_column_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("metric.sh");
    std::fs::write(
        &script,
        format!("log=$1\n{metric_line}\nprintf '%s' \"$metric\"\n"),
    )
    .unwrap();
    let log = dir.join("case.log");
    let extract = |log_content: &str| -> String {
        std::fs::write(&log, log_content).unwrap();
        let output = Command::new("bash").arg(&script).arg(&log).output().unwrap();
        assert!(
            output.status.success(),
            "metric extraction failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };

    // gam's logger lines, including one appended to libtest's `test <name> ... `,
    // contribute nothing; the test's own println! figures are all kept.
    let traced = extract(
        "test families::module::case ... [INFO] [STAGE] identifiability canonicalise: start rho_dim=9\n\
         [INFO] [PIRLS/JN] cyc=  0/1200 obj=1.914364e2 -loglik=1.643632e2 pen=2.707e1\n\
         [INFO] [joint-newton-tr] phase=line_search cycle=0 r=1.000e0 hessian_qp_elapsed=0.001s\n\
         held-out rmse=0.06613 deviance=3.5443 coverage: 0.95\n",
    );
    assert_eq!(traced, "rmse=0.06613 deviance=3.5443 coverage: 0.95 ");

    // Positive control for the tag filter: the same trace figures printed by the
    // test itself, untagged, are captured.
    assert_eq!(
        extract("obj=1.914364e2 -loglik=1.643632e2\n"),
        "loglik=1.643632e2 "
    );

    // Token boundaries: `ess` inside `hessian_qp_elapsed=` and `acc` inside
    // `accepted_step_inf=` are not metrics. Positive controls: `ess` and `rmse` as
    // whole name parts (`ess_bulk=`, `gam_rmse=`) are.
    assert_eq!(
        extract("hessian_qp_elapsed=0.001s accepted_step_inf=7.085e0 ess_bulk=412 gam_rmse=0.1\n"),
        "ess_bulk=412 rmse=0.1 "
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The results commit names the steps that failed before publishing, each name
/// verbatim. Run 35327329208's body read "streaming;  full": the list was joined on
/// `;` and respaced, which also rewrote the `;` inside step 8's own name.
#[test]
fn test_publish_action_names_failed_steps_verbatim() {
    let action = std::fs::read_to_string(".github/actions/publish-gha-results/action.yml").unwrap();
    let assignment = |name: &str| -> String {
        action
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with(&format!("{name}=$(printf '%s\\n' \"$failed_steps\"")))
            .unwrap_or_else(|| panic!("the action builds `{name}` from `$failed_steps`"))
            .to_string()
    };
    let script = format!(
        "failed_steps=$(printf '8\\tRun quality suite (resilient + streaming; full per-test capture)\\n9\\tAggregate quality pairs (#1561 gate + #2395 paired power)')\n{}\n{}\nprintf '%s\\n%s' \"$failed_numbers\" \"$failed_detail\"\n",
        assignment("failed_numbers"),
        assignment("failed_detail"),
    );
    let output = Command::new("bash").arg("-c").arg(&script).output().unwrap();
    assert!(
        output.status.success(),
        "the failed-step labels did not build: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let (numbers, detail) = stdout
        .split_once('\n')
        .expect("the numbers line, then the detail line");
    assert_eq!(numbers, "8, 9");
    assert_eq!(
        detail,
        "step 8: Run quality suite (resilient + streaming; full per-test capture) | \
         step 9: Aggregate quality pairs (#1561 gate + #2395 paired power)"
    );
}

/// Everything the run step does to build `panicmsg`, lifted out of the workflow
/// so no test can hold a second copy of it.
///
/// That is two statements — the awk over the panic message, then the `if` that
/// falls back to libtest's `Error:` line for a test that returned an error
/// instead of panicking — so the block is read from `panicmsg=$(awk` to the
/// `fi` that closes the fallback. Each line is trimmed, which neither awk nor
/// bash cares about and which keeps the extracted script free of the workflow's
/// YAML indentation.
fn panic_message_assignment(yaml: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for line in yaml.lines() {
        let trimmed = line.trim();
        if lines.is_empty() && !trimmed.starts_with("panicmsg=$(awk") {
            continue;
        }
        lines.push(trimmed);
        if trimmed == "fi" {
            break;
        }
    }
    assert!(
        lines.last().is_some_and(|last| *last == "fi")
            && lines.iter().any(|line| line.ends_with("\"$log\")")),
        "the run step builds `panicmsg` with `panicmsg=$(awk ... \"$log\")` and \
         then falls back to the libtest `Error:` line, closing with `fi`"
    );
    lines.join("\n")
}

/// The `panicmsg` capture keeps every line of the panic message.
///
/// `assert_eq!` writes the two compared values on the lines AFTER the line that
/// says the assertion failed, and those values are the only
/// record of what the code under test produced. A capture that took the
/// location line plus one more dropped them, which is how run 35596444985
/// reported that a multinomial class "must carry one independent λ per (smooth
/// term, penalty)" without reporting how many it carried.
#[test]
fn test_reference_quality_panic_message_keeps_every_line() {
    let yaml = std::fs::read_to_string(".github/workflows/reference-quality.yml").unwrap();
    let assignment = panic_message_assignment(&yaml);
    let dir = std::env::temp_dir().join(format!("quality_panicmsg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("panicmsg.sh");
    std::fs::write(
        &script,
        format!("log=$1\n{assignment}\nprintf '%s' \"$panicmsg\"\n"),
    )
    .unwrap();
    let log = dir.join("case.log");
    let capture = |log_content: &str| -> String {
        std::fs::write(&log, log_content).unwrap();
        let output = Command::new("bash")
            .arg(&script)
            .arg(&log)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "panic capture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };

    // An `assert_eq!` failure: both compared values are carried, and the capture
    // stops at the end of the message rather than swallowing libtest's sections.
    let compared = capture(
        "running 1 test\n\
         thread 'families::case' panicked at tests/quality/families/case.rs:422:9:\n\
         assertion `left == right` failed: class 0 must carry one lambda per penalty\n\
         \x20 left: 2\n\
         \x20right: 4\n\
         note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n\
         test families::case ... FAILED\n",
    );
    assert!(
        compared.contains("left: 2") && compared.contains("right: 4"),
        "the compared values were dropped: {compared}"
    );
    assert!(
        !compared.contains("RUST_BACKTRACE") && !compared.contains("FAILED"),
        "the capture ran past the end of the message: {compared}"
    );

    // A one-line message still reads `<location> :: <message>`, the shape every
    // classifier regex above is written against.
    assert_eq!(
        capture(
            "thread 'misc::case' panicked at tests/quality/misc/case.rs:130:6:\n\
             gam additive fit: Fit(Estimation(did not certify a stationary optimum))\n\
             note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
        ),
        "thread 'misc::case' panicked at tests/quality/misc/case.rs:130:6: :: \
         gam additive fit: Fit(Estimation(did not certify a stationary optimum))"
    );

    // A test declared `-> Result<(), E>` never panics; libtest reports its
    // returned error on an `Error:` line, and that line is the verdict.
    assert_eq!(
        capture(
            "running 1 test\n\
             ---- misc::fit_quality_stress::hifreq_tensor_k4 stdout ----\n\
             Error: \"hifreq_tensor_k4: band does not cover f_B, Q=7.4\"\n\
             test result: FAILED\n"
        ),
        "Error: \"hifreq_tensor_k4: band does not cover f_B, Q=7.4\""
    );

    // A log with neither a panic nor a returned error yields nothing, so the
    // classifier's `${panicmsg:-nonzero exit $rc}` fallback still fires.
    assert_eq!(capture("running 1 test\ntest result: FAILED\n"), "");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Everything the merge job does to fold the shards into ONE suite record,
/// lifted out of the workflow so no test can hold a second copy of it — the
/// same contract as [`panic_message_assignment`] for the run step's
/// `panicmsg`.
///
/// The block is bounded by its own code, not by a marker planted for this
/// test: it opens at `merge_root=`, which names the merged directory, and
/// closes at the `printf` that writes `completeness.tsv`, which is its last
/// statement. What follows in that step — the unrecorded-case listing and the
/// `$GITHUB_ENV` hand-off — is GitHub plumbing rather than the merge, and a
/// script that carried it could not run outside a job. Each line is trimmed, as
/// for the run step, so the extracted script is free of the YAML indentation.
fn merge_script(yaml: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for line in yaml.lines() {
        let trimmed = line.trim();
        if lines.is_empty() && !trimmed.starts_with("merge_root=") {
            continue;
        }
        lines.push(trimmed);
        if trimmed.ends_with("> \"$merge_root/completeness.tsv\"") {
            break;
        }
    }
    assert!(
        lines
            .first()
            .is_some_and(|first| first.starts_with("merge_root="))
            && lines
                .last()
                .is_some_and(|last| last.ends_with("> \"$merge_root/completeness.tsv\"")),
        "the merge job folds the shards with a block from `merge_root=` to the \
         `printf` that writes completeness.tsv"
    );
    lines.join("\n")
}

/// The merge folds every shard's rows into one record, once each, renumbered
/// into a single `idx` sequence, and its completeness marker is denominated in
/// the UNION of what the shards enumerated.
///
/// This is the property the sharding is for. Before it, one job ran all 437
/// cases and a run that outlived its cap published a prefix; a prefix of a
/// measurement looks exactly like a smaller suite, which is how a 60-of-456
/// fragment became the tracked artifact. Merging is now the only place the
/// suite's row count is decided, so it is the only place that can make that
/// mistake again.
#[test]
fn test_reference_quality_merge_folds_every_shard_row_exactly_once() {
    let yaml = std::fs::read_to_string(".github/workflows/reference-quality.yml").unwrap();
    let dir = std::env::temp_dir().join(format!("quality_merge_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Point the block at this fixture instead of the job's absolute paths. The
    // replacement is asserted, so renaming either directory in the workflow
    // fails here rather than silently testing nothing.
    let merged = dir.join("merged");
    let shards = dir.join("shards");
    let script_body = merge_script(&yaml)
        .replace(
            "merge_root=/tmp/quality-report",
            &format!("merge_root={}", merged.display()),
        )
        .replace(
            "shards_root=/tmp/quality-shards",
            &format!("shards_root={}", shards.display()),
        );
    assert!(
        script_body.contains(&format!("merge_root={}", merged.display()))
            && script_body.contains(&format!("shards_root={}", shards.display())),
        "the merge block names /tmp/quality-report and /tmp/quality-shards; it now reads:\n{script_body}"
    );
    let script = dir.join("merge.sh");
    std::fs::write(&script, format!("set +e -u -o pipefail\n{script_body}\n")).unwrap();

    let header = "idx\toutcome\tcause\ttest\trc\tdur_s\tsub_passed\tsub_failed\tmetric\treason\n";
    // Every shard enumerates the whole binary and runs its own slice, so the
    // case lists agree and the row sets are disjoint.
    let cases = "a::t1\na::t2\nb::t3\nb::t4\n";
    let write_shard = |name: &str, rows: &str, measured: &str| {
        let shard = shards.join(name);
        std::fs::create_dir_all(shard.join("logs")).unwrap();
        std::fs::write(shard.join("quality_cases.txt"), cases).unwrap();
        std::fs::write(shard.join("quality_results.tsv"), format!("{header}{rows}")).unwrap();
        std::fs::write(shard.join("quality_results.jsonl"), "").unwrap();
        std::fs::write(
            shard.join("completeness.tsv"),
            format!("measured\t{measured}\n"),
        )
        .unwrap();
        std::fs::write(shard.join("tallies.tsv"), "pass\t1\nassigned\t2\n").unwrap();
    };
    let shard_one = "1\tPASS\tok\ta::t1\t0\t3\t1\t0\trmse=0.1\t\n\
                     2\tPASS\tok\ta::t2\t0\t4\t1\t0\t\t\n";
    let shard_two = "1\tGAM_ERROR\tgam_fit_failed\tb::t3\t101\t9\t0\t1\t\tboom\n\
                     2\tPASS\tok\tb::t4\t0\t2\t1\t0\t\t\n";

    let run = || -> (Vec<String>, String) {
        let _ = std::fs::remove_dir_all(&merged);
        let output = Command::new("bash").arg(&script).output().unwrap();
        assert!(
            output.status.success(),
            "the merge block failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let rows: Vec<String> = std::fs::read_to_string(merged.join("quality_results.tsv"))
            .unwrap()
            .lines()
            .skip(1)
            .map(str::to_string)
            .collect();
        let marker = std::fs::read_to_string(merged.join("completeness.tsv")).unwrap();
        (rows, marker)
    };
    let field = |marker: &str, key: &str| -> String {
        marker
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}\t")))
            .unwrap_or_else(|| panic!("completeness.tsv has no `{key}` row: {marker}"))
            .to_string()
    };

    // Both shards complete: every row survives, once, renumbered 1..4 in shard
    // name order, and the marker is COMPLETE against the enumerated union.
    write_shard("quality-results-shard-1", shard_one, "true");
    write_shard("quality-results-shard-2", shard_two, "true");
    let (rows, marker) = run();
    let names: Vec<&str> = rows
        .iter()
        .map(|row| row.split('\t').nth(3).unwrap())
        .collect();
    assert_eq!(names, ["a::t1", "a::t2", "b::t3", "b::t4"]);
    let indices: Vec<&str> = rows
        .iter()
        .map(|row| row.split('\t').next().unwrap())
        .collect();
    assert_eq!(
        indices,
        ["1", "2", "3", "4"],
        "idx is renumbered across shards"
    );
    assert!(
        rows[2].contains("GAM_ERROR") && rows[2].ends_with("boom"),
        "the outcome columns are carried verbatim: {}",
        rows[2]
    );
    assert_eq!(field(&marker, "completeness"), "COMPLETE");
    assert_eq!(field(&marker, "enumerated"), "4");
    assert_eq!(field(&marker, "rows_written"), "4");
    assert_eq!(field(&marker, "duplicate_rows"), "0");

    // A shard that reached only part of its slice makes the SUITE partial. This
    // is the case the old single job could not express: it published the prefix
    // and called it the suite.
    write_shard(
        "quality-results-shard-2",
        "1\tGAM_ERROR\tgam_fit_failed\tb::t3\t101\t9\t0\t1\t\tboom\n",
        "true",
    );
    let (rows, marker) = run();
    assert_eq!(rows.len(), 3);
    assert_eq!(field(&marker, "completeness"), "PARTIAL");
    assert_eq!(field(&marker, "enumerated"), "4");
    assert_eq!(field(&marker, "rows_written"), "3");

    // A shard whose binary did not enumerate makes the union no measurement of
    // the suite, whatever the other shards wrote (#2744).
    write_shard("quality-results-shard-2", shard_two, "false");
    let (_, marker) = run();
    assert_eq!(field(&marker, "completeness"), "NOT_MEASURED");
    assert_eq!(field(&marker, "measured"), "false");

    // A case two shards both ran is kept once, from the first shard in name
    // order, and counted. Without the dedup `rows_written` would exceed
    // `enumerated` and the suite would read as larger than it is.
    write_shard(
        "quality-results-shard-2",
        &format!("{shard_two}3\tPASS\tok\ta::t1\t0\t5\t1\t0\t\t\n"),
        "true",
    );
    let (rows, marker) = run();
    let names: Vec<&str> = rows
        .iter()
        .map(|row| row.split('\t').nth(3).unwrap())
        .collect();
    assert_eq!(names, ["a::t1", "a::t2", "b::t3", "b::t4"]);
    assert_eq!(
        rows[0].split('\t').nth(5).unwrap(),
        "3",
        "the first shard's row is the one kept"
    );
    assert_eq!(field(&marker, "duplicate_rows"), "1");
    assert_eq!(field(&marker, "completeness"), "COMPLETE");

    std::fs::remove_dir_all(&dir).unwrap();
}

/// The shard plan is derived from the case list and the per-case bounds, and
/// every shard fits the run step's own cap.
///
/// The count is not a number in the workflow: it is whatever packing the bounds
/// force. This pins the two properties that make it a guarantee rather than a
/// hope — each shard's summed bounds fit the budget, and every enumerated case
/// is assigned to exactly one shard — and it pins the budget to the step cap,
/// so raising `timeout-minutes` without re-deriving the plan cannot pass.
#[test]
fn test_reference_quality_shard_plan_fits_every_shard_in_the_step_cap() {
    let yaml = std::fs::read_to_string(".github/workflows/reference-quality.yml").unwrap();
    let dir = std::env::temp_dir().join(format!("quality_plan_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // The plan is a python heredoc in the `plan` job; run the same body.
    let body: String = {
        let mut lines: Vec<&str> = Vec::new();
        let mut inside = false;
        for line in yaml.lines() {
            if !inside {
                if line.trim().starts_with("if ! python3 - <<'PLAN'") {
                    inside = true;
                }
                continue;
            }
            if line.trim() == "PLAN" {
                break;
            }
            lines.push(line.strip_prefix("          ").unwrap_or(line));
        }
        assert!(
            lines.iter().any(|line| line.contains("SHARD BUDGET")),
            "the plan reads the shard budget out of the workflow"
        );
        lines.join("\n")
    };
    let script = dir.join("plan.py");
    std::fs::write(&script, body).unwrap();
    let summary = dir.join("summary.md");
    let outputs = dir.join("outputs.txt");
    std::fs::write(&summary, "").unwrap();
    std::fs::write(&outputs, "").unwrap();
    let output = Command::new("python3")
        .arg(&script)
        .env("GITHUB_STEP_SUMMARY", &summary)
        .env("GITHUB_OUTPUT", &outputs)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "the shard plan did not derive: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let plan = String::from_utf8(output.stdout).unwrap();
    let cases: Vec<String> =
        std::fs::read_to_string("bench/gha_results/reference-quality/quality_cases.txt")
            .unwrap()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_string)
            .collect();
    let mut assigned: Vec<(usize, String)> = plan
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (shard, case) = line.split_once('\t').expect("`<shard>\\t<case>` rows");
            (
                shard.parse::<usize>().expect("a shard number"),
                case.to_string(),
            )
        })
        .collect();
    assert_eq!(
        assigned.len(),
        cases.len(),
        "every enumerated case is assigned once"
    );
    let mut names: Vec<&String> = assigned.iter().map(|(_, case)| case).collect();
    names.sort();
    let mut expected: Vec<&String> = cases.iter().collect();
    expected.sort();
    assert_eq!(names, expected, "the plan covers exactly the case list");

    // The shard count the plan announced, and the budget it packed against.
    let outputs = std::fs::read_to_string(&outputs).unwrap();
    let total: usize = outputs
        .lines()
        .find_map(|line| line.strip_prefix("total="))
        .expect("the plan outputs `total=`")
        .parse()
        .unwrap();
    assert!(total >= 1);
    let summary = std::fs::read_to_string(&summary).unwrap();
    let budget: usize = summary
        .split("budget=")
        .nth(1)
        .and_then(|rest| rest.split('s').next())
        .expect("the plan reports the budget it packed against")
        .parse()
        .unwrap();
    // The budget is the run step's cap, not a second number.
    // The cap is the `timeout-minutes:` under the `# SHARD BUDGET:` COMMENT —
    // matched as a comment line, so the plan script's own mention of the marker
    // in its regex and its error message cannot stand in for it.
    let cap_minutes: usize = yaml
        .lines()
        .skip_while(|line| !line.trim_start().starts_with("# SHARD BUDGET:"))
        .find_map(|line| line.trim().strip_prefix("timeout-minutes:"))
        .expect("the run step carries the `# SHARD BUDGET:` cap")
        .trim()
        .parse()
        .unwrap();
    assert_eq!(
        budget,
        cap_minutes * 60,
        "the plan packs against the step's own cap"
    );

    // Every shard's summed bounds fit the budget. The loads are in the summary
    // table the plan writes, one row per shard.
    let mut shards_seen = 0usize;
    for line in summary.lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // `| shard | cases | bound load | recorded load |`
        if cells.len() != 6 || cells[1].parse::<usize>().is_err() {
            continue;
        }
        shards_seen += 1;
        let load: usize = cells[3]
            .trim_end_matches('s')
            .parse()
            .expect("a bound load in seconds");
        assert!(
            load <= budget,
            "shard {} holds {load}s of bound against a {budget}s cap",
            cells[1]
        );
    }
    assert_eq!(shards_seen, total, "the summary reports every shard's load");
    assert_eq!(
        assigned.iter().map(|(shard, _)| *shard).max().unwrap(),
        total,
        "the assignment uses every shard the plan announced"
    );
    assigned.sort();
    assert!(
        assigned.windows(2).all(|pair| pair[0].1 != pair[1].1),
        "no case is assigned to two shards"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

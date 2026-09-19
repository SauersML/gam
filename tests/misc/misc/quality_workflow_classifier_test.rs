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

panicmsg=$(awk '/panicked at/{{loc=$0; getline; print loc " :: " $0; exit}}' "$log")
refmarker=$(grep -aE 'there is no package called|could not find function|Error in library\(|package or namespace load failed|unable to load shared object|cannot open shared object|No module named|ModuleNotFoundError|IndentationError|reference .* body failed' "$log" | head -1)

{}

echo "$outcome,$cause"
"#,
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

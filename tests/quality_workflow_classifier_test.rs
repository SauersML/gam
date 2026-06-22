use std::process::Command;
use std::io::Write;

#[test]
fn test_reference_quality_classifier() {
    let yaml = std::fs::read_to_string(".github/workflows/reference-quality.yml").unwrap();
    
    // Extract the classifier logic
    let start_marker = "gamfit_re='IntegrationFailed|InvalidConfig";
    let end_marker = "outcome=METRIC_OFF;";
    
    let mut classifier_code = String::new();
    let mut in_block = false;
    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(start_marker) {
            in_block = true;
        }
        if in_block {
            classifier_code.push_str(line);
            classifier_code.push('\n');
            if trimmed == "fi" {
                break;
            }
        }
    }
    
    assert!(classifier_code.contains("gamfit_re="), "Could not extract classifier logic");
    
    // Now create a bash script that we can call
    let test_script = format!(r#"#!/bin/bash
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
"#, classifier_code);

    let script_path = "/tmp/test_classifier.sh";
    std::fs::write(script_path, test_script).unwrap();
    std::process::Command::new("chmod").args(&["+x", script_path]).status().unwrap();

    let run_case = |outcome: &str, rc: i32, log_content: &str, test_name: &str| -> (String, String) {
        let log_path = "/tmp/test_classifier.log";
        std::fs::write(log_path, log_content).unwrap();
        let output = Command::new(script_path)
            .args(&[outcome, &rc.to_string(), log_path, test_name])
            .output()
            .unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        let trimmed = stdout.trim();
        let parts: Vec<&str> = trimmed.split(',').collect();
        (parts[0].to_string(), parts[1].to_string())
    };

    // Test 1: PASS
    let (out, cause) = run_case("", 0, "Some output", "test1");
    assert_eq!(out, "PASS");
    assert_eq!(cause, "ok");

    // Test 2: GAM_ERROR (LayoutError)
    let (out, cause) = run_case("", 101, "thread 'main' panicked at src/foo.rs:10:\nLayoutError: bad layout", "test2");
    assert_eq!(out, "GAM_ERROR");
    assert_eq!(cause, "gam_fit_failed");

    // Test 3: METRIC_OFF
    let (out, cause) = run_case("", 1, "Failed to fit", "test3");
    assert_eq!(out, "METRIC_OFF");
    assert_eq!(cause, "quality_metric");

    // Test 4: TEST_ERROR
    let (out, cause) = run_case("", 101, "thread 'main' panicked at src/test_support/reference.rs:20:\nData mismatch", "test4");
    assert_eq!(out, "TEST_ERROR");
    assert_eq!(cause, "test_setup");

    // Test 5: REF_ERROR
    let (out, cause) = run_case("", 1, "Error in library(mgcv) : there is no package called 'mgcv'", "test5");
    assert_eq!(out, "REF_ERROR");
    assert_eq!(cause, "reference_tool");
}

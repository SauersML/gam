use std::{fs, process::Command};

#[test]
fn cli_fit_reports_degenerate_inputs_at_the_shared_boundary() {
    let cases = [
        ("y,x\n0,0\n1,NaN\n2,1\n", "y ~ x", "column 'x'", "non-finite"),
        ("y,x\n0,0\n1,inf\n2,1\n", "y ~ x", "column 'x'", "non-finite"),
        ("y,x\n0,0\n1,-inf\n2,1\n", "y ~ x", "column 'x'", "non-finite"),
        (
            "y,x\n0,NA\n1,4\n2,NA\n",
            "y ~ x",
            "column 'x'",
            "only one non-missing value",
        ),
        (
            "y,g\n0,only\n1,only\n2,only\n",
            "y ~ g",
            "column 'g'",
            "fewer than two levels",
        ),
        ("y,x,x\n0,0,1\n1,1,0\n2,2,1\n", "y ~ x", "column 'x'", "duplicate"),
    ];
    for (csv, formula, column, problem) in cases {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("data.csv");
        let model = dir.path().join("model.gam");
        fs::write(&input, csv).unwrap();
        let output = Command::new(gam_test_support::gam_binary!())
            .args([
                "fit",
                input.to_str().unwrap(),
                formula,
                "--family",
                "gaussian",
                "--out",
                model.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success(),
            "degenerate input minted a fit: {csv}"
        );
        assert!(
            stderr.contains(column) && stderr.contains(problem),
            "{stderr}"
        );
    }
}

#[test]
fn cli_fit_reports_empty_frame_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("data.csv");
    let model = dir.path().join("model.gam");
    fs::write(&input, "y,x\n").unwrap();
    let output = Command::new(gam_test_support::gam_binary!())
        .args([
            "fit",
            input.to_str().unwrap(),
            "y ~ x",
            "--family",
            "gaussian",
            "--out",
            model.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no rows"));
}

/// The family-support, prior-weight and table-size rules are the same Rust
/// layer the Python API reaches, so the CLI reports the same typed messages.
#[test]
fn cli_fit_reports_family_support_weight_and_row_errors_at_the_shared_boundary() {
    let counts = "y,x,w\n1,0.1,1\n0,0.2,1\n2.5,0.3,1\n3,0.4,1\n1,0.5,1\n";
    let negative_weight = "y,x,w\n1,0.1,1\n0,0.2,1\n2,0.3,-1\n3,0.4,1\n1,0.5,1\n";
    let zero_weights = "y,x,w\n1,0.1,0\n0,0.2,0\n2,0.3,0\n3,0.4,0\n1,0.5,0\n";
    let cases: [(&str, &str, bool, &[&str]); 4] = [
        (
            counts,
            "poisson-log",
            false,
            &["column 'y'", "Poisson family", "first offending row 3 has value 2.5"],
        ),
        (
            negative_weight,
            "gaussian",
            true,
            &["column 'w'", "must be non-negative; found -1 at row 3"],
        ),
        (zero_weights, "gaussian", true, &["column 'w'", "no positive weight"]),
        ("y,x\n0.3,0.5\n", "gaussian", false, &["too few rows"]),
    ];
    for (csv, family, weighted, needles) in cases {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("data.csv");
        let model = dir.path().join("model.gam");
        fs::write(&input, csv).unwrap();
        let mut args = vec![
            "fit".to_string(),
            input.to_str().unwrap().to_string(),
            "y ~ x".to_string(),
            "--family".to_string(),
            family.to_string(),
            "--out".to_string(),
            model.to_str().unwrap().to_string(),
        ];
        if weighted {
            args.extend(["--weights-column".to_string(), "w".to_string()]);
        }
        let output = Command::new(gam_test_support::gam_binary!())
            .args(&args)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "invalid input minted a fit: {csv}");
        for needle in needles {
            assert!(stderr.contains(needle), "missing {needle:?}: {stderr}");
        }
    }
}

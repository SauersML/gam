use std::{fs, process::Command};

#[test]
fn cli_fit_reports_degenerate_inputs_at_the_shared_boundary() {
    let cases = [
        ("y,x\n0,0\n1,NaN\n2,1\n", "column 'x'", "non-finite"),
        ("y,x\n0,0\n1,inf\n2,1\n", "column 'x'", "non-finite"),
        ("y,x\n0,0\n1,-inf\n2,1\n", "column 'x'", "non-finite"),
        ("y,x\n0,4\n1,4\n2,4\n", "column 'x'", "constant"),
        (
            "y,x\n0,NA\n1,4\n2,NA\n",
            "column 'x'",
            "only one non-missing value",
        ),
        (
            "y,g\n0,only\n1,only\n2,only\n",
            "column 'g'",
            "fewer than two levels",
        ),
        ("y,x,x\n0,0,1\n1,1,0\n2,2,1\n", "column 'x'", "duplicate"),
    ];
    for (csv, column, problem) in cases {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("data.csv");
        fs::write(&input, csv).unwrap();
        let output = Command::new(gam_test_support::gam_binary!())
            .args([
                "fit",
                input.to_str().unwrap(),
                "y ~ x",
                "--family",
                "gaussian",
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
    fs::write(&input, "y,x\n").unwrap();
    let output = Command::new(gam_test_support::gam_binary!())
        .args(["fit", input.to_str().unwrap(), "y ~ x"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no rows"));
}

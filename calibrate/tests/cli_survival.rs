#![cfg(feature = "survival-data")]

use std::fs;
use std::process::Command;

use tempfile::tempdir;

#[test]
fn survival_cli_time_varying_trains_without_lambda_flag() {
    let tmp = tempdir().expect("temporary directory");
    let training_path = tmp.path().join("survival_train.tsv");

    let data = "age_entry\tage_exit\tevent_target\tevent_competing\tsample_weight\tpgs\tsex\tpc1\n\
50\t55\t1\t0\t1.0\t-0.2\t0.0\t0.1\n\
52\t56\t0\t1\t1.0\t0.1\t1.0\t-0.2\n\
54\t59\t0\t0\t1.0\t0.3\t0.0\t0.0\n\
58\t63\t1\t0\t1.0\t0.6\t1.0\t0.3\n";
    fs::write(&training_path, data).expect("write training data");

    let exe = env!("CARGO_BIN_EXE_gnomon");
    let output = Command::new(exe)
        .current_dir(tmp.path())
        .args([
            "train",
            "--model-family",
            "survival",
            training_path.to_str().expect("path str"),
            "--num-pcs",
            "1",
            "--survival-enable-time-varying",
            "--max-iterations",
            "1",
            "--reml-max-iterations",
            "1",
        ])
        .output()
        .expect("run gnomon cli");

    assert!(
        !output.status.success(),
        "CLI unexpectedly succeeded: {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("P-IRLS inner loop"),
        "expected P-IRLS failure, stderr: {stderr}"
    );
}

#[test]
fn survival_cli_rejects_retired_barrier_flags() {
    let tmp = tempdir().expect("temporary directory");
    let training_path = tmp.path().join("survival_train.tsv");

    let data = "age_entry\tage_exit\tevent_target\tevent_competing\tsample_weight\tpgs\tsex\tpc1\n\
50\t55\t1\t0\t1.0\t-0.2\t0.0\t0.1\n";
    fs::write(&training_path, data).expect("write training data");

    let exe = env!("CARGO_BIN_EXE_gnomon");
    for (flag, value) in [
        ("--survival-barrier-weight", "1e-4"),
        ("--survival-barrier-scale", "1.0"),
    ] {
        let output = Command::new(exe)
            .current_dir(tmp.path())
            .args([
                "train",
                "--model-family",
                "survival",
                training_path.to_str().expect("path str"),
                "--num-pcs",
                "0",
                flag,
                value,
            ])
            .output()
            .expect("run gnomon cli");

        assert!(!output.status.success(), "CLI unexpectedly accepted {flag}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(flag),
            "expected stderr to mention {flag}, got: {stderr}"
        );
    }
}

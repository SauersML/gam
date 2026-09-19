use std::process::{Command, Output};

fn gam(args: &[&str]) -> Output {
    Command::new(gam_test_support::gam_binary!())
        .args(args)
        .output()
        .expect("spawn gam CLI")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn cli_fit_bad_inputs_exit_nonzero_and_name_the_offending_input() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let nonfinite = scratch.path().join("nonfinite.csv");
    let model = scratch.path().join("model.gam");
    std::fs::write(&nonfinite, "y,x\n1,0\n2,NaN\n").expect("write fixture");

    let output = gam(&[
        "fit",
        nonfinite.to_str().expect("UTF-8 path"),
        "y ~ x",
        "--out",
        model.to_str().expect("UTF-8 path"),
    ]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let error = stderr(&output);
    assert!(error.contains("nonfinite.csv"), "{error}");
    assert!(error.contains("column 'x'"), "{error}");
    assert!(error.contains("non-finite"), "{error}");

    let malformed = gam(&[
        "fit",
        nonfinite.to_str().expect("UTF-8 path"),
        "y ~ s(",
        "--out",
        model.to_str().expect("UTF-8 path"),
    ]);
    assert_eq!(malformed.status.code(), Some(1), "{}", stderr(&malformed));
    assert!(
        stderr(&malformed).contains("formula"),
        "{}",
        stderr(&malformed)
    );

    let missing = gam(&[
        "fit",
        nonfinite.to_str().expect("UTF-8 path"),
        "y ~ absent_column",
        "--out",
        model.to_str().expect("UTF-8 path"),
    ]);
    assert_eq!(missing.status.code(), Some(1), "{}", stderr(&missing));
    assert!(
        stderr(&missing).contains("absent_column"),
        "{}",
        stderr(&missing)
    );

    let wrong_format = scratch.path().join("training.json");
    std::fs::write(&wrong_format, "{}\n").expect("write wrong-format fixture");
    let wrong = gam(&[
        "fit",
        wrong_format.to_str().expect("UTF-8 path"),
        "y ~ x",
        "--out",
        model.to_str().expect("UTF-8 path"),
    ]);
    assert_eq!(wrong.status.code(), Some(1), "{}", stderr(&wrong));
    assert!(
        stderr(&wrong).contains("training.json"),
        "{}",
        stderr(&wrong)
    );
}

#[test]
fn every_post_fit_command_rejects_a_bad_model_with_a_named_error() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let bad_model = scratch.path().join("corrupt-model.gam");
    let data = scratch.path().join("data.csv");
    let out = scratch.path().join("out.csv");
    std::fs::write(&bad_model, "not a saved GAM").expect("write corrupt model");
    std::fs::write(&data, "y,x\n1,2\n").expect("write data");
    let model = bad_model.to_str().expect("UTF-8 path");
    let data = data.to_str().expect("UTF-8 path");
    let out = out.to_str().expect("UTF-8 path");

    for args in [
        vec!["predict", model, data, "--out", out],
        vec!["diagnose", model, data],
        vec!["sample", model, data, "--out", out],
        vec!["generate", model, data, "--out", out],
        vec!["report", model, data, out],
    ] {
        let output = gam(&args);
        assert_eq!(
            output.status.code(),
            Some(1),
            "args={args:?}: {}",
            stderr(&output)
        );
        assert!(
            stderr(&output).contains("corrupt-model.gam"),
            "args={args:?}: {}",
            stderr(&output)
        );
    }
}

#[test]
fn diagnose_rejects_removed_no_op_alo_flag() {
    let output = gam(&["diagnose", "model.gam", "data.csv", "--alo"]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("unexpected argument '--alo'"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn cli_predict_names_the_refused_cell_and_prints_its_remedy() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let training = scratch.path().join("training.csv");
    let model = scratch.path().join("model.gam");
    let out = scratch.path().join("predictions.csv");
    let mut rows = String::from("y,x,g,k\n");
    for i in 0..90 {
        let x = f64::from(i) / 90.0;
        let level = i % 3;
        let code = (i / 3) % 3;
        let y = (6.0 * x).sin()
            + f64::from(level)
            + 0.5 * f64::from(code)
            + 0.05 * f64::from(i % 7);
        rows.push_str(&format!("{y},{x},L{level},{code}\n"));
    }
    std::fs::write(&training, rows).expect("write training fixture");
    let fit = gam(&[
        "fit",
        training.to_str().expect("UTF-8 path"),
        "y ~ s(x) + g + factor(k)",
        "--out",
        model.to_str().expect("UTF-8 path"),
    ]);
    assert!(fit.status.success(), "{}", stderr(&fit));

    let predict = |name: &str, body: &str| {
        let path = scratch.path().join(name);
        std::fs::write(&path, body).expect("write new-data fixture");
        gam(&[
            "predict",
            model.to_str().expect("UTF-8 path"),
            path.to_str().expect("UTF-8 path"),
            "--out",
            out.to_str().expect("UTF-8 path"),
        ])
    };
    let cases = [
        (
            "nan.csv",
            "x,g,k\n0.5,L0,0\nNaN,L1,1\n",
            [
                "non-finite value at row 2, column 'x'",
                "help: Drop or impute",
            ],
        ),
        (
            "unseen_label.csv",
            "x,g,k\n0.5,L0,0\n0.5,LNEW,1\n",
            [
                "unseen level 'LNEW' in categorical column 'g' at row 2",
                "help: Map the label",
            ],
        ),
        (
            "unseen_code.csv",
            "x,g,k\n0.5,L0,0\n0.5,L1,7\n",
            [
                "unseen level '7' in categorical column 'k' at row 2",
                "help: Map the label",
            ],
        ),
    ];
    for (name, body, expected) in cases {
        let output = predict(name, body);
        assert_eq!(output.status.code(), Some(1), "{name}: {}", stderr(&output));
        let error = stderr(&output);
        for needle in expected {
            assert!(
                error.contains(needle),
                "{name}: missing {needle:?} in\n{error}"
            );
        }
    }
}

#[test]
fn expectile_tau_is_held_to_the_expectile_family_and_the_open_unit_interval() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_expectile_frailty_guard.csv"
    );
    let scratch = tempfile::tempdir().expect("scratch directory");
    let model = scratch.path().join("model.gam");
    let model = model.to_str().expect("UTF-8 path");
    let fit = |extra: &[&str]| {
        let mut args = vec!["fit", fixture, "y ~ s(x)", "--out", model];
        args.extend_from_slice(extra);
        gam(&args)
    };

    // Any family but expectile — the inferred one, an explicit one, and the
    // `--predict-noise` location-scale route — refuses the asymmetry.
    for extra in [
        &["--expectile-tau", "0.9"][..],
        &["--family", "gaussian", "--expectile-tau", "0.9"],
        &["--predict-noise", "s(x)", "--expectile-tau", "0.9"],
        &["--family", "gaussian", "--expectile-tau", "0.1,0.9"],
    ] {
        let output = fit(extra);
        assert_eq!(output.status.code(), Some(1), "{extra:?}: {}", stderr(&output));
        assert!(
            stderr(&output).contains("requires family = \"expectile\""),
            "{extra:?}: {}",
            stderr(&output)
        );
    }

    // An out-of-range asymmetry is refused while parsing the flag.
    for tau in ["0", "1", "1.5"] {
        let output = fit(&["--family", "expectile", "--expectile-tau", tau]);
        assert!(!output.status.success(), "tau={tau}: {}", stderr(&output));
        assert!(
            stderr(&output).contains("in (0, 1)"),
            "tau={tau}: {}",
            stderr(&output)
        );
    }

    let expectile = fit(&["--family", "expectile", "--expectile-tau", "0.9"]);
    assert!(expectile.status.success(), "{}", stderr(&expectile));
}

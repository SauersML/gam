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

/// The exit code of each `ErrorCategory`, so a code in an assertion is the
/// category's own number rather than a copy of it.
fn exit_code(category: gam::ErrorCategory) -> Option<i32> {
    Some(category.exit_code())
}

#[test]
fn cli_fit_bad_inputs_exit_with_their_category_and_name_the_offending_input() {
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
    assert_eq!(output.status.code(), exit_code(gam::ErrorCategory::Data), "{}", stderr(&output));
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
    assert_eq!(
        malformed.status.code(),
        exit_code(gam::ErrorCategory::Formula),
        "{}",
        stderr(&malformed)
    );
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
    assert_eq!(
        missing.status.code(),
        exit_code(gam::ErrorCategory::Formula),
        "{}",
        stderr(&missing)
    );
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
    assert_eq!(wrong.status.code(), exit_code(gam::ErrorCategory::Data), "{}", stderr(&wrong));
    assert!(
        stderr(&wrong).contains("training.json"),
        "{}",
        stderr(&wrong)
    );
}

/// A saved model the engine cannot read is a data refusal
/// (`FittedModelError::error_category`), the category `gamfit.load` raises for
/// the same file, on every command that loads one.
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
        vec!["transformation-score", model, data, "--out", out],
        vec!["latent-residual", model, data, "--out", out],
        vec!["diagnose", model, data],
        vec!["residuals", model, data, "--type", "response"],
        vec!["partial-effect", model, "--term", "s(x)"],
        vec!["summary", model],
        vec!["compare", model],
        vec!["sample", model, data, "--out", out],
        vec!["generate", model, data, "--out", out],
        vec!["report", model, data, out],
    ] {
        let output = gam(&args);
        let error = stderr(&output);
        assert_eq!(
            output.status.code(),
            exit_code(gam::ErrorCategory::Data),
            "args={args:?}: {error}"
        );
        assert!(
            error.contains("failed to parse model") && error.contains("corrupt-model.gam"),
            "args={args:?}: {error}"
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

/// `s(g)` on a string column used to fit silently, treating the level codes as
/// a number line. It is refused while the formula is resolved, exits with the
/// formula code, and names the column, the first non-numeric value and the
/// terms that do take a factor. The same refusal reaches Python as
/// `gamfit.errors.FormulaError`.
#[test]
fn a_smooth_of_a_string_column_is_a_formula_error_naming_the_column() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let data = scratch.path().join("categorical.csv");
    let model = scratch.path().join("model.gam");
    let mut csv = String::from("y,x,grp\n");
    for i in 0..60 {
        let level = ["north", "south", "east"][i % 3];
        csv.push_str(&format!("{},{},{level}\n", (i as f64 * 0.37).sin(), i as f64 / 60.0));
    }
    std::fs::write(&data, csv).expect("write fixture");
    let data = data.to_str().expect("UTF-8 path");
    let model = model.to_str().expect("UTF-8 path");

    for formula in ["y ~ s(grp)", "y ~ s(x) + s(grp)", "y ~ te(x, grp)", "y ~ linear(grp)"] {
        let output = gam(&["fit", data, formula, "--out", model]);
        let error = stderr(&output);
        assert_eq!(
            output.status.code(),
            exit_code(gam::ErrorCategory::Formula),
            "{formula}: {error}"
        );
        assert!(error.contains("'grp'"), "{formula}: {error}");
        assert!(error.contains("'north' at row 1"), "{formula}: {error}");
        assert!(error.contains("factor(grp)"), "{formula}: {error}");
        assert!(error.contains("s(x, by=grp)"), "{formula}: {error}");
    }

    let by_factor = gam(&["fit", data, "y ~ s(x, by=grp)", "--out", model]);
    assert_eq!(by_factor.status.code(), Some(0), "{}", stderr(&by_factor));
}

/// Fit `y ~ s(x) + g + factor(k)` on a 90-row table with a string factor `g`
/// (levels `L0..L2`) and a numeric-coded factor `k` (codes `0..2`), returning
/// the saved model's path.
fn fit_factor_fixture(scratch: &std::path::Path) -> std::path::PathBuf {
    let training = scratch.join("training.csv");
    let model = scratch.join("model.gam");
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
    model
}

#[test]
fn cli_predict_names_the_refused_cell_and_prints_its_remedy() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let model = fit_factor_fixture(scratch.path());
    let out = scratch.path().join("predictions.csv");

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
        assert_eq!(
            output.status.code(),
            exit_code(gam::ErrorCategory::Data),
            "{name}: {}",
            stderr(&output)
        );
        let error = stderr(&output);
        for needle in expected {
            assert!(
                error.contains(needle),
                "{name}: missing {needle:?} in\n{error}"
            );
        }
    }
}

/// The post-fit commands load their data through the same model-schema loader
/// as `predict`, so a refused cell exits with the data category and prints the
/// typed refusal's `help:` line on every one of them, not a bare message under
/// the invocation category.
#[test]
fn post_fit_commands_keep_the_data_refusal_category_and_remedy() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let model = fit_factor_fixture(scratch.path());
    let data = scratch.path().join("labeled_nan.csv");
    std::fs::write(&data, "y,x,g,k\n1.0,0.5,L0,0\n2.0,NaN,L1,1\n").expect("write labeled fixture");
    let model = model.to_str().expect("UTF-8 path");
    let data = data.to_str().expect("UTF-8 path");
    let sample_out = scratch.path().join("posterior.csv");
    let generate_out = scratch.path().join("generated.csv");
    let report_out = scratch.path().join("report.html");
    let commands: [Vec<&str>; 5] = [
        vec!["residuals", model, data, "--type", "response"],
        vec!["diagnose", model, data],
        vec![
            "sample",
            model,
            data,
            "--out",
            sample_out.to_str().expect("UTF-8 path"),
        ],
        vec![
            "generate",
            model,
            data,
            "--out",
            generate_out.to_str().expect("UTF-8 path"),
        ],
        vec![
            "report",
            model,
            data,
            report_out.to_str().expect("UTF-8 path"),
        ],
    ];
    for command in commands {
        let output = gam(&command);
        let error = stderr(&output);
        assert_eq!(
            output.status.code(),
            exit_code(gam::ErrorCategory::Data),
            "{}: {error}",
            command[0]
        );
        for needle in [
            "non-finite value at row 2, column 'x'",
            "help: Drop or impute",
        ] {
            assert!(
                error.contains(needle),
                "{}: missing {needle:?} in\n{error}",
                command[0]
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
        assert_eq!(
            output.status.code(),
            exit_code(gam::ErrorCategory::Formula),
            "{extra:?}: {}",
            stderr(&output)
        );
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

/// `--family` takes the names `gamfit.fit(..., family=...)` takes, because
/// both hand the string to the one library resolver (#4574): a bare head, the
/// hyphen spelling, and the parenthesized-link form all name the same Gamma
/// fit, and a spelling the resolver refuses is refused with its message.
#[test]
fn family_flag_accepts_the_library_family_names() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_explicit_family_gamma.csv"
    );
    let scratch = tempfile::tempdir().expect("scratch directory");
    let model = scratch.path().join("model.gam");
    let model = model.to_str().expect("UTF-8 path");
    let fit = |family: &str| gam(&["fit", fixture, "y ~ x", "--family", family, "--out", model]);

    for family in ["gamma", "gamma-log", "Gamma(log)"] {
        let output = fit(family);
        assert!(output.status.success(), "--family {family}: {}", stderr(&output));
    }

    let alias = fit("nb");
    assert!(!alias.status.success(), "{}", stderr(&alias));
    assert!(
        stderr(&alias).contains("unknown family `nb`; use `negative-binomial`"),
        "{}",
        stderr(&alias)
    );
    let misspelled = fit("gama");
    assert!(!misspelled.status.success(), "{}", stderr(&misspelled));
    assert!(stderr(&misspelled).contains("unknown family 'gama'"), "{}", stderr(&misspelled));
}

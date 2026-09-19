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
            exit_code(gam::ErrorCategory::Formula),
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

/// `s(g)` on a string column used to fit silently, treating the level codes as
/// a number line. It is refused while the formula is resolved, exits with the
/// formula code, and names the column, the first non-numeric value and the
/// terms that do take a factor. The same refusal reaches Python as
/// `gamfit.FormulaError`.
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

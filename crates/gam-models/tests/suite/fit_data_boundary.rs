use gam_data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
use gam_models::fit_orchestration::{FitConfig, WorkflowError, materialize};
use ndarray::Array2;

fn dataset(values: Vec<f64>, kind: ColumnKindTag, levels: Vec<&str>) -> EncodedDataset {
    EncodedDataset {
        headers: vec!["offender".into()],
        values: Array2::from_shape_vec((values.len(), 1), values).unwrap(),
        schema: DataSchema {
            columns: vec![SchemaColumn {
                name: "offender".into(),
                kind,
                levels: levels.into_iter().map(str::to_string).collect(),
            }],
        },
        column_kinds: vec![kind],
    }
}

#[test]
fn public_materializer_returns_typed_data_errors_before_design_construction() {
    let cases = [
        (
            dataset(vec![0.0, f64::NAN, 1.0], ColumnKindTag::Continuous, vec![]),
            "has non-finite value NaN",
        ),
        (
            dataset(
                vec![0.0, f64::INFINITY, 1.0],
                ColumnKindTag::Continuous,
                vec![],
            ),
            "has non-finite value inf",
        ),
        (
            dataset(
                vec![0.0, f64::NEG_INFINITY, 1.0],
                ColumnKindTag::Continuous,
                vec![],
            ),
            "has non-finite value -inf",
        ),
        (
            dataset(
                vec![f64::NAN, 4.0, f64::NAN],
                ColumnKindTag::Continuous,
                vec![],
            ),
            "has only one non-missing value",
        ),
        (
            dataset(
                vec![0.0, 0.0, 0.0],
                ColumnKindTag::Categorical,
                vec!["only"],
            ),
            "is a factor with fewer than two levels",
        ),
    ];
    for (data, problem) in cases {
        let error = match materialize("offender ~ 1", &data, &FitConfig::default()) {
            Err(error) => error,
            Ok(_) => panic!("degenerate data must not materialize a fit request"),
        };
        assert!(
            matches!(&error, WorkflowError::InvalidData { column, .. } if column == "offender")
        );
        assert!(error.to_string().contains(problem), "{error}");
    }
}

#[test]
fn public_materializer_types_empty_and_duplicate_column_errors() {
    let empty = EncodedDataset {
        headers: vec!["x".into()],
        values: Array2::zeros((0, 1)),
        schema: DataSchema {
            columns: vec![SchemaColumn {
                name: "x".into(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            }],
        },
        column_kinds: vec![ColumnKindTag::Continuous],
    };
    let duplicate = EncodedDataset {
        headers: vec!["x".into(), "x".into()],
        values: Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap(),
        schema: DataSchema { columns: vec![] },
        column_kinds: vec![ColumnKindTag::Continuous; 2],
    };
    for (data, column, problem) in [
        (empty, "<table>", "no observations"),
        (duplicate, "x", "duplicate name"),
    ] {
        let error = match materialize("x ~ 1", &data, &FitConfig::default()) {
            Err(error) => error,
            Ok(_) => panic!("invalid table must not materialize a fit request"),
        };
        assert!(
            matches!(&error, WorkflowError::InvalidData { column: actual, .. } if actual == column)
        );
        assert!(error.to_string().contains(problem));
    }
}

/// A continuous table from named columns, for the response/weight matrix.
fn table(columns: &[(&str, Vec<f64>)]) -> EncodedDataset {
    let nrows = columns[0].1.len();
    let mut values = Array2::zeros((nrows, columns.len()));
    for (j, (_, column)) in columns.iter().enumerate() {
        for (i, v) in column.iter().enumerate() {
            values[[i, j]] = *v;
        }
    }
    EncodedDataset {
        headers: columns.iter().map(|(name, _)| name.to_string()).collect(),
        values,
        schema: DataSchema {
            columns: columns
                .iter()
                .map(|(name, _)| SchemaColumn {
                    name: name.to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; columns.len()],
    }
}

fn config(family: &str, weights: bool) -> FitConfig {
    FitConfig {
        family: Some(family.to_string()),
        weight_column: weights.then(|| "w".to_string()),
        ..FitConfig::default()
    }
}

fn materialize_error(data: &EncodedDataset, config: &FitConfig) -> WorkflowError {
    match materialize("y ~ x", data, config) {
        Err(error) => error,
        Ok(_) => panic!("invalid data must not materialize a fit request"),
    }
}

/// Each family's support and degeneracy rule is a typed data error at the
/// response column that names the family and the first offending 1-based row.
#[test]
fn response_outside_family_support_is_a_data_error_naming_family_and_row() {
    let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let cases: [(&str, Vec<f64>, &[&str]); 7] = [
        ("poisson", vec![1.0, 0.0, 2.0, -1.0, 3.0, 1.0], &["Poisson", "row 4 has value -1"]),
        ("poisson", vec![1.0, 0.0, 2.5, 1.0, 3.0, 1.0], &["Poisson", "row 3 has value 2.5", "tweedie"]),
        (
            "negative-binomial",
            vec![1.0, 0.5, 2.0, 1.0, 3.0, 1.0],
            &["Negative-Binomial", "row 2 has value 0.5"],
        ),
        ("binomial", vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0], &["Binomial", "row 5 has value 2"]),
        ("binomial", vec![0.0, 1.0, -0.5, 0.0, 1.0, 1.0], &["Binomial", "row 3 has value -0.5"]),
        ("gamma", vec![1.0, 2.0, 0.0, 1.5, 3.0, 1.0], &["Gamma", "row 3 has value 0"]),
        ("poisson", vec![0.0; 6], &["degenerate", "all counts are 0"]),
    ];
    for (family, y, needles) in cases {
        let data = table(&[("y", y), ("x", x.clone())]);
        let error = materialize_error(&data, &config(family, false));
        assert!(
            matches!(&error, WorkflowError::InvalidData { column, .. } if column == "y"),
            "{family}: {error:?}"
        );
        let message = error.to_string();
        for needle in needles {
            assert!(message.contains(needle), "{family}: missing {needle:?} in {message}");
        }
    }
}

/// A zero prior weight excludes its row from the likelihood, so an
/// out-of-support value on that row is accepted, and a response that is
/// degenerate over the positive-weight rows is refused even when an excluded
/// row would have broken the degeneracy.
#[test]
fn zero_weight_rows_are_excluded_from_support_and_degeneracy() {
    let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let w = vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let gamma = table(&[
        ("y", vec![1.0, 2.0, 0.0, 1.5, 3.0, 1.0]),
        ("x", x.clone()),
        ("w", w.clone()),
    ]);
    materialize("y ~ x", &gamma, &config("gamma", true))
        .expect("an excluded zero-weight row is not judged against the Gamma support");

    let binomial = table(&[
        ("y", vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
        ("x", x),
        ("w", w),
    ]);
    let error = materialize_error(&binomial, &config("binomial", true));
    assert!(matches!(&error, WorkflowError::InvalidData { column, .. } if column == "y"));
    assert!(error.to_string().contains("all values are 1"), "{error}");
}

/// Prior weights must be finite, non-negative, and not all zero; each
/// violation is a typed data error at the weight column.
#[test]
fn invalid_prior_weights_are_data_errors_at_the_weight_column() {
    let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let y = vec![0.3, 1.1, 0.7, 1.9, 1.4, 2.2];
    let cases: [(Vec<f64>, &str); 4] = [
        (vec![1.0, 1.0, -1.0, 1.0, 1.0, 1.0], "must be non-negative; found -1 at row 3"),
        (vec![0.0; 6], "no positive weight"),
        (vec![1.0, f64::NAN, 1.0, 1.0, 1.0, 1.0], "non-finite value NaN at row 2"),
        (vec![1.0, 1.0, 1.0, 1.0, f64::INFINITY, 1.0], "non-finite value inf at row 5"),
    ];
    for (w, needle) in cases {
        let data = table(&[("y", y.clone()), ("x", x.clone()), ("w", w)]);
        let error = materialize_error(&data, &config("gaussian", true));
        assert!(
            matches!(&error, WorkflowError::InvalidData { column, .. } if column == "w"),
            "{error:?}"
        );
        assert!(error.to_string().contains(needle), "missing {needle:?} in {error}");
    }
}

/// One observation, a column with no finite value, a non-finite predictor and
/// a NaN response are all refused by name before any design is built.
#[test]
fn degenerate_tables_are_data_errors() {
    let one_row = table(&[("y", vec![0.3]), ("x", vec![0.5])]);
    let error = materialize_error(&one_row, &FitConfig::default());
    assert!(matches!(&error, WorkflowError::InvalidData { column, .. } if column == "<table>"));
    assert!(error.to_string().contains("only one observation"), "{error}");

    let y = vec![0.3, 1.1, 0.7, 1.9];
    for (x, y, column, needle) in [
        (vec![f64::NAN; 4], y.clone(), "x", "has no finite values"),
        (vec![0.1, f64::INFINITY, 0.3, 0.4], y.clone(), "x", "non-finite value inf at row 2"),
        (vec![0.1, 0.2, 0.3, 0.4], vec![0.3, 1.1, f64::NAN, 1.9], "y", "non-finite value NaN at row 3"),
    ] {
        let data = table(&[("y", y), ("x", x)]);
        let error = materialize_error(&data, &FitConfig::default());
        assert!(
            matches!(&error, WorkflowError::InvalidData { column: actual, .. } if actual == column),
            "{error:?}"
        );
        assert!(error.to_string().contains(needle), "missing {needle:?} in {error}");
    }
}

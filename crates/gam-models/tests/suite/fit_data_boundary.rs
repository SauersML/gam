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
            dataset(vec![4.0, 4.0, 4.0], ColumnKindTag::Continuous, vec![]),
            "is constant",
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

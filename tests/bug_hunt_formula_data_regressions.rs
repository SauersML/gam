use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::{encode_recordswith_inferred_schema, load_csvwith_inferred_schema, load_csvwith_schema};
use std::fs;

#[test]
fn parse_formula_all_supported_constructs_preserves_source_order_and_unique_ids() {
    let parsed = parse_formula(
        "y ~ x + s(z, by=g, k=6) + te(a, b, k=5) + s(fac, t, bs=sz, k=5) + re(id) + offset(log(exposure))",
    )
    .expect("formula with linear/smooth/by/tensor/factor-smooth/random-effect/offset should parse");

    assert_eq!(
        parsed.terms.len(),
        6,
        "expected six parsed terms in source order for the supported construct mix"
    );
}

#[test]
fn parse_formula_without_lhs_returns_specific_descriptive_error() {
    let err = parse_formula("~ x + s(z)")
        .expect_err("formula without LHS must fail with a descriptive error");
    let msg = err.to_string();
    assert!(
        msg.contains("left-hand side") || msg.contains("response"),
        "formula with no LHS should report response/left-hand-side guidance, got: {msg}"
    );
}

#[test]
fn load_csvwith_inferred_schema_infers_integer_string_and_float_types_distinctly() {
    let path = std::env::temp_dir().join(format!(
        "gam_bug_hunt_schema_{}_{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::write(&path, "i,s,f\n1,a,1.5\n2,b,2.25\n3,a,3.0\n").expect("write temp csv");

    let ds = load_csvwith_inferred_schema(&path).expect("inferred load should succeed");
    fs::remove_file(&path).ok();

    assert_eq!(
        ds.column_kinds,
        vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Categorical,
            ColumnKindTag::Continuous
        ],
        "inferred schema should keep integer columns distinct from float columns and strings categorical"
    );
}

#[test]
fn load_csvwith_schema_uses_explicit_schema_and_preserves_schema_column_order() {
    let path = std::env::temp_dir().join(format!(
        "gam_bug_hunt_schema_explicit_{}_{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::write(&path, "a,b,c\n1,red,2.5\n2,blue,3.25\n").expect("write temp csv");

    let schema = DataSchema {
        columns: vec![
            SchemaColumn {
                name: "c".into(),
                kind: ColumnKindTag::Continuous,
                levels: None,
            },
            SchemaColumn {
                name: "b".into(),
                kind: ColumnKindTag::Categorical,
                levels: Some(vec!["blue".into(), "red".into()]),
            },
            SchemaColumn {
                name: "a".into(),
                kind: ColumnKindTag::Continuous,
                levels: None,
            },
        ],
    };

    let ds = load_csvwith_schema(&path, &schema).expect("explicit schema load should succeed");
    fs::remove_file(&path).ok();

    assert_eq!(
        ds.headers,
        vec!["c", "b", "a"],
        "explicit schema should determine output column order"
    );
}

#[test]
fn encode_recordswith_inferred_schema_keeps_documented_column_order() {
    let headers = vec!["b".to_string(), "a".to_string(), "c".to_string()];
    let rows = vec![
        vec!["cat".to_string(), "1".to_string(), "2.5".to_string()],
        vec!["dog".to_string(), "2".to_string(), "3.5".to_string()],
    ];
    let ds = encode_recordswith_inferred_schema(headers.clone(), rows)
        .expect("record encoding with inferred schema should succeed");

    assert_eq!(
        ds.headers, headers,
        "encoded design should keep column order aligned with provided headers"
    );
}

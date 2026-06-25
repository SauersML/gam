//! Regression gate: the `gam` CLI's untyped-CSV ingestion must encode a
//! numeric-coded grouping column declared in a factor-by-construction role
//! (`group(g)` / `factor(g)` / `re(g)`, or a categorical response) as a FACTOR,
//! exactly as the typed Python frame does — and must leave a genuinely
//! continuous integer covariate (`s(x)` / bare `+ x`) as `Continuous`.
//!
//! ## The shipped bug this pins
//!
//! `gamfit.fit` (the in-process Python path) generalized strictly better than
//! the `gam` CLI binary on identical data/model across every seed. Root cause:
//! a typed Python frame stamps a categorical-dtype column with
//! `CATEGORICAL_CELL_SENTINEL`, so the column-major inferer
//! (`infer_and_encode_column_major`) forces it to a factor even when its labels
//! parse as numbers ("0","1","2"). An untyped CSV cannot carry that sentinel,
//! so the delimited inferer demoted a numeric-coded grouping column to a single
//! `Continuous` ramp — a strictly lower-capacity design than the factor the
//! Python path built — and so generalized worse.
//!
//! The fix keys the categorical forcing on the *formula role* the user
//! declared (`group(region)`), not on a value heuristic, via
//! `load_dataset_projected_with_categorical_roles`. This test asserts:
//!
//! 1. With `region` in the categorical-role set, the CSV loader encodes it as a
//!    `Categorical` factor with sorted numeric-string levels — byte-identical
//!    kind/levels/codes to the typed-frame (Python) sentinel path.
//! 2. The continuous covariate `age` (an integer column NOT in any categorical
//!    role) stays `Continuous` — the critical safety property: `s(age)` is not
//!    wrongly factorized.
//! 3. The default value-based loader (empty role set) still classifies a
//!    numeric `region` as `Continuous`, so the fix is opt-in by role.

use gam::inference::data::{
    CATEGORICAL_CELL_SENTINEL, infer_and_encode_column_major, load_dataset_projected,
    load_dataset_projected_with_categorical_roles,
};
use gam::inference::model::ColumnKindTag;
use std::collections::HashSet;
use std::io::Write;

/// Write `header_line\n` followed by `rows` to a unique temp CSV path.
fn write_csv(name: &str, header_line: &str, rows: &[&str]) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!("gam_owed_cli_cat_{name}_{nonce}.csv"));
    let mut f = std::fs::File::create(&path).expect("create temp csv");
    writeln!(f, "{header_line}").expect("write header");
    for row in rows {
        writeln!(f, "{row}").expect("write row");
    }
    path
}

fn schema_column<'a>(
    ds: &'a gam::inference::data::EncodedDataset,
    name: &str,
) -> &'a gam::inference::model::SchemaColumn {
    ds.schema
        .columns
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("column '{name}' missing from schema"))
}

fn column_values(ds: &gam::inference::data::EncodedDataset, name: &str) -> Vec<f64> {
    let idx = ds
        .headers
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("column '{name}' missing from headers"));
    ds.values.column(idx).to_vec()
}

#[test]
fn cli_numeric_coded_group_column_forced_to_factor_matches_python_and_keeps_continuous() {
    // A numeric-coded grouping column `region` (0,1,2,3 in non-sorted-by-first
    // appearance order so the level sort is observable), a continuous integer
    // covariate `age`, and a numeric response `y`.
    let header = "y,age,region";
    let rows = [
        "1.0,40,2", "2.0,55,0", "3.0,33,1", "4.0,61,3", "5.0,47,2", "6.0,29,0",
    ];
    let path = write_csv("group", header, &rows);
    let requested: Vec<String> = vec!["y".into(), "age".into(), "region".into()];

    // ---- (1) Role-forced load: `region` is a factor, `age` stays continuous.
    let mut roles = HashSet::<&str>::new();
    roles.insert("region");
    let forced = load_dataset_projected_with_categorical_roles(&path, &requested, &roles)
        .expect("role-forced CSV load");

    let region = schema_column(&forced, "region");
    assert_eq!(
        region.kind,
        ColumnKindTag::Categorical,
        "a numeric-coded group(region) column must be forced to a factor"
    );
    // Levels are the distinct numeric labels, sorted lexicographically (the
    // canonical, row-order-independent factor encoding).
    assert_eq!(
        region.levels,
        vec![
            "0".to_string(),
            "1".to_string(),
            "2".to_string(),
            "3".to_string()
        ],
        "forced-categorical levels must be the sorted distinct numeric labels"
    );

    let age = schema_column(&forced, "age");
    assert_eq!(
        age.kind,
        ColumnKindTag::Continuous,
        "an integer covariate NOT in a categorical role must stay continuous (s(age) must not be factorized)"
    );

    // ---- (2) Python typed-frame parity: the column-major inferer with the
    // categorical sentinel must produce the SAME kind, levels, and codes.
    let region_cells_raw = ["2", "0", "1", "3", "2", "0"];
    let sentinel_cells: Vec<String> = region_cells_raw
        .iter()
        .map(|c| format!("{CATEGORICAL_CELL_SENTINEL}{c}"))
        .collect();
    let sentinel_refs: Vec<&str> = sentinel_cells.iter().map(String::as_str).collect();
    let (py_schema, py_codes) =
        infer_and_encode_column_major("region", &sentinel_refs, 2).expect("typed-frame encode");
    assert_eq!(
        py_schema.kind,
        ColumnKindTag::Categorical,
        "sanity: typed-frame sentinel path encodes region as a factor"
    );
    assert_eq!(
        region.levels, py_schema.levels,
        "CLI forced-categorical levels must match the Python typed-frame levels"
    );
    assert_eq!(
        column_values(&forced, "region"),
        py_codes,
        "CLI forced-categorical level codes must match the Python typed-frame codes exactly"
    );

    // ---- (3) Default value-based load (no roles): region is Continuous, which
    // is exactly the pre-fix CLI behavior — proving the fix is opt-in by role.
    let inferred = load_dataset_projected(&path, &requested).expect("default CSV load");
    assert_eq!(
        schema_column(&inferred, "region").kind,
        ColumnKindTag::Continuous,
        "without a declared categorical role the value-based inferer still sees numeric region as continuous"
    );

    drop(std::fs::remove_file(&path));
}

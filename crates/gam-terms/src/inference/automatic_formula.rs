//! The automatic formula: expansion of the `.` term against a data schema.
//!
//! `y ~ .` means "every column the fit does not otherwise read". This module is
//! the one place that decides which term each such column becomes; the CLI, the
//! Python wrapper (including `GAMRegressor().fit(X, y)` with no formula) and the
//! Rust fit entry points all call [`expand_automatic_formula`], so one table
//! always yields one formula.
//!
//! Every term the expansion emits is penalized with a penalty whose null space
//! is itself penalized, so REML can shrink any of them to exactly zero: a column
//! carrying no signal ends with effective degrees of freedom near 0 instead of
//! the fit spending one unpenalized degree of freedom on it.
//!
//! The per-column rule reads only the column's schema kind and its number of
//! distinct values, and every cut is an identifiability boundary:
//!
//! * one distinct value — the column equals a multiple of the intercept, so no
//!   term on it is identifiable. It is dropped with a note (see
//!   [`AutomaticColumnTerm::Constant`] for why this is a note, not an error);
//! * categorical (string, pandas category) — `factor(col)`, a ridge-penalized
//!   level effect estimated by REML. When every level occurs exactly once the
//!   level effects are aliased one-for-one with the observations, so the column
//!   is an identifier, not a factor, and is dropped with a note;
//! * numeric with two distinct values (0/1 indicators, booleans) — a bare linear
//!   term `col`. Two support points identify an intercept and a slope and
//!   nothing more, so a smooth has no curvature direction to estimate;
//! * numeric with three or more distinct values — `s(col)`. The second-order
//!   difference penalty leaves a two-dimensional null space (constant, linear);
//!   the constant is absorbed by the intercept, so a curvature direction is
//!   identifiable exactly when a third support point exists.

use std::collections::{BTreeSet, HashSet};

use gam_data::{ColumnKindTag, EncodedDataset, canonical_level_bits};

use super::formula_dsl::{AUTOMATIC_REST_TERM, FormulaDslError, formula_rhs_terms};

/// Distinct values at which a numeric column identifies a curvature direction
/// beyond the intercept and slope: the dimension of the second-order
/// difference penalty's null space (constant + linear) plus one.
const SMOOTH_MIN_DISTINCT_VALUES: usize = 3;

/// A formula whose `.` term has been replaced by the terms the data schema
/// implies, with one note per column the expansion dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutomaticFormula {
    pub formula: String,
    pub notes: Vec<String>,
}

/// The term the automatic formula assigns to one column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AutomaticColumnTerm {
    /// `s(col)`: a numeric column with enough support for curvature.
    Smooth,
    /// `col`: a numeric column with exactly two distinct values.
    Linear,
    /// `factor(col)`: a categorical column.
    Factor,
    /// Dropped: a single distinct value. The column is exactly aliased with
    /// the intercept, so dropping it leaves the model unchanged; refusing
    /// would instead fail `fit(X, y)` on any table that holds a column which
    /// happens to be constant in this sample (a filtered subset, a one-site
    /// study), which is the wrong default for an automatic formula.
    Constant,
    /// Dropped: a categorical column whose every level occurs exactly once.
    Identifier,
}

impl AutomaticColumnTerm {
    fn term_text(&self, column: &str) -> Option<String> {
        match self {
            Self::Smooth => Some(format!("s({column})")),
            Self::Linear => Some(column.to_string()),
            Self::Factor => Some(format!("factor({column})")),
            Self::Constant | Self::Identifier => None,
        }
    }

    fn drop_note(&self, column: &str) -> Option<String> {
        match self {
            Self::Constant => Some(format!(
                "automatic formula: dropped column '{column}' because it takes a single value, \
                 which the intercept already represents"
            )),
            Self::Identifier => Some(format!(
                "automatic formula: dropped categorical column '{column}' because every level \
                 occurs exactly once, so its level effects are not identifiable from the \
                 residuals"
            )),
            Self::Smooth | Self::Linear | Self::Factor => None,
        }
    }
}

/// The automatic-formula rule for one column of `dataset`.
pub fn automatic_column_term(dataset: &EncodedDataset, column_index: usize) -> AutomaticColumnTerm {
    let column = dataset.values.column(column_index);
    let distinct: HashSet<u64> = column.iter().map(|&v| canonical_level_bits(v)).collect();
    if distinct.len() <= 1 {
        return AutomaticColumnTerm::Constant;
    }
    match dataset.column_kinds[column_index] {
        ColumnKindTag::Categorical => {
            if distinct.len() == column.len() {
                AutomaticColumnTerm::Identifier
            } else {
                AutomaticColumnTerm::Factor
            }
        }
        ColumnKindTag::Continuous | ColumnKindTag::Binary => {
            if distinct.len() >= SMOOTH_MIN_DISTINCT_VALUES {
                AutomaticColumnTerm::Smooth
            } else {
                AutomaticColumnTerm::Linear
            }
        }
    }
}

/// Whether `formula`'s right-hand side contains the automatic `.` term.
///
/// This is the first parse a fit makes, so a formula that does not parse fails
/// here, and it fails as the parser's error.
pub fn formula_has_automatic_term(formula: &str) -> Result<bool, FormulaDslError> {
    let (_, terms) = formula_rhs_terms(formula)?;
    Ok(terms.iter().any(|term| term.trim() == AUTOMATIC_REST_TERM))
}

/// `formula` with the `.` term removed (an empty right-hand side becomes the
/// intercept `1`). Its columns are the ones the expansion must not re-add.
pub fn formula_without_automatic_term(formula: &str) -> Result<String, FormulaDslError> {
    let (response, terms) = formula_rhs_terms(formula)?;
    let explicit: Vec<&str> = terms
        .iter()
        .map(|term| term.trim())
        .filter(|term| *term != AUTOMATIC_REST_TERM)
        .collect();
    Ok(assemble_formula(&response, &explicit))
}

fn assemble_formula(response: &str, terms: &[&str]) -> String {
    if terms.is_empty() {
        format!("{response} ~ 1")
    } else {
        format!("{response} ~ {}", terms.join(" + "))
    }
}

/// Whether `name` can appear as a bare column reference in a formula.
fn is_formula_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// Replace the `.` term of `formula` with one term per column of `dataset` that
/// is not in `reserved` (the columns the rest of the fit already reads: the
/// response, explicit terms, weights, offsets, ...), in column order.
///
/// A formula without `.` is returned unchanged with no notes.
pub fn expand_automatic_formula(
    formula: &str,
    dataset: &EncodedDataset,
    reserved: &BTreeSet<String>,
) -> Result<AutomaticFormula, String> {
    let (response, terms) = formula_rhs_terms(formula)?;
    let dot_count = terms
        .iter()
        .filter(|term| term.trim() == AUTOMATIC_REST_TERM)
        .count();
    if dot_count == 0 {
        return Ok(AutomaticFormula {
            formula: formula.to_string(),
            notes: Vec::new(),
        });
    }
    if dot_count > 1 {
        return Err("the `.` term (every remaining column) may appear at most once".to_string());
    }

    let mut automatic_terms = Vec::<String>::new();
    let mut notes = Vec::<String>::new();
    for (column_index, name) in dataset.headers.iter().enumerate() {
        if reserved.contains(name) {
            continue;
        }
        if !is_formula_identifier(name) {
            return Err(format!(
                "the `.` term cannot reference column '{name}': a formula column name must \
                 start with a letter or '_' and contain only letters, digits, '_' and '.'; \
                 rename the column or write the formula explicitly"
            ));
        }
        let rule = automatic_column_term(dataset, column_index);
        match rule.term_text(name) {
            Some(term) => automatic_terms.push(term),
            None => notes.extend(rule.drop_note(name)),
        }
    }

    let mut expanded = Vec::<&str>::new();
    for term in &terms {
        let term = term.trim();
        if term == AUTOMATIC_REST_TERM {
            expanded.extend(automatic_terms.iter().map(String::as_str));
        } else {
            expanded.push(term);
        }
    }
    let formula = assemble_formula(&response, &expanded);
    notes.insert(0, format!("automatic formula: {formula}"));
    Ok(AutomaticFormula { formula, notes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_data::{DataSchema, SchemaColumn};
    use ndarray::Array2;

    fn dataset(columns: &[(&str, ColumnKindTag, Vec<f64>)]) -> EncodedDataset {
        let nrows = columns[0].2.len();
        let mut values = Array2::<f64>::zeros((nrows, columns.len()));
        for (j, (_, _, col)) in columns.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                values[[i, j]] = v;
            }
        }
        EncodedDataset {
            headers: columns.iter().map(|(n, _, _)| n.to_string()).collect(),
            values,
            schema: DataSchema {
                columns: columns
                    .iter()
                    .map(|(n, k, _)| SchemaColumn {
                        name: n.to_string(),
                        kind: *k,
                        levels: vec![],
                    })
                    .collect(),
            },
            column_kinds: columns.iter().map(|(_, k, _)| *k).collect(),
        }
    }

    fn mixed() -> EncodedDataset {
        use ColumnKindTag::*;
        dataset(&[
            ("y", Continuous, vec![0.1, 0.4, 0.2, 0.9, 0.5, 0.3]),
            ("x", Continuous, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("g", Categorical, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            ("flag", Binary, vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
            ("two", Continuous, vec![2.5, 7.0, 2.5, 7.0, 2.5, 7.0]),
            ("c", Continuous, vec![3.0; 6]),
            ("id", Categorical, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            ("w", Continuous, vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0]),
        ])
    }

    fn reserved(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn dot_expands_each_column_by_its_identifiability_rule() {
        let out = expand_automatic_formula("y ~ .", &mixed(), &reserved(&["y", "w"])).unwrap();
        assert_eq!(out.formula, "y ~ s(x) + factor(g) + flag + two");
        assert_eq!(out.notes.len(), 3);
        assert!(out.notes[0].contains("y ~ s(x) + factor(g) + flag + two"));
        assert!(out.notes[1].contains("'c'") && out.notes[1].contains("single value"));
        assert!(out.notes[2].contains("'id'") && out.notes[2].contains("exactly once"));
    }

    #[test]
    fn explicit_terms_keep_their_place_and_are_not_re_added() {
        let out = expand_automatic_formula(
            "y ~ s(x, k=5) + . + flag:two",
            &mixed(),
            &reserved(&["y", "x", "flag", "two", "w"]),
        )
        .unwrap();
        assert_eq!(out.formula, "y ~ s(x, k=5) + factor(g) + flag:two");
    }

    #[test]
    fn formula_without_dot_is_untouched() {
        let out = expand_automatic_formula("y ~ s(x)", &mixed(), &reserved(&["y", "x"])).unwrap();
        assert_eq!(out.formula, "y ~ s(x)");
        assert!(out.notes.is_empty());
    }

    #[test]
    fn dot_detection_and_removal() {
        assert!(formula_has_automatic_term("y ~ .").unwrap());
        assert!(formula_has_automatic_term("y ~ x + .").unwrap());
        assert!(!formula_has_automatic_term("y ~ x + .5").unwrap());
        assert!(!formula_has_automatic_term("y ~ s(x.a)").unwrap());
        assert_eq!(formula_without_automatic_term("y ~ .").unwrap(), "y ~ 1");
        assert_eq!(
            formula_without_automatic_term("y ~ . + s(x)").unwrap(),
            "y ~ s(x)"
        );
    }

    #[test]
    fn repeated_dot_and_unaddressable_columns_are_refused() {
        assert!(expand_automatic_formula("y ~ . + .", &mixed(), &reserved(&["y"])).is_err());
        let bad = dataset(&[
            ("y", ColumnKindTag::Continuous, vec![1.0, 2.0, 3.0]),
            ("0", ColumnKindTag::Continuous, vec![1.0, 2.0, 3.0]),
        ]);
        let err = expand_automatic_formula("y ~ .", &bad, &reserved(&["y"])).unwrap_err();
        assert!(err.contains("'0'") && err.contains("rename"), "{err}");
    }

    #[test]
    fn parse_formula_refuses_an_unexpanded_dot() {
        let err = super::super::formula_dsl::parse_formula("y ~ .").unwrap_err();
        assert!(err.to_string().contains("expanded"), "{err}");
    }
}

//! Tail probabilities against the 25-digit references in `tail_references/`, which
//! `tail_references/generate.py` writes with mpmath at the exact `f64` inputs read here.
//!
//! Every reference is an upper tail in `[1e-300, 1 − 1e-12]` (the generator's targets, where
//! they are representable) or a weighted chi-square tail at a statistic chosen the same way.
//! The accuracy asked of all of them is [`RELATIVE_ACCURACY`] against the reference; for the
//! weighted sum the returned bound must also hold, and must itself be that tight.

use gam_math::probability::{
    TailProbability, WeightedChiSquareTerm, chi_square_sf, fisher_snedecor_sf, signed_weighted_chi_square_sf,
    student_t_two_sided_probability,
};

/// The lane's acceptance level for a tail against its reference.
const RELATIVE_ACCURACY: f64 = 1e-10;

/// The references carry 25 significant digits.
const REFERENCE_RELATIVE_ERROR: f64 = 1e-24;

fn rows(table: &'static str) -> impl Iterator<Item = Vec<&'static str>> {
    table.lines().skip(1).filter(|line| !line.is_empty()).map(|line| line.split('\t').collect())
}

fn number(field: &str) -> f64 {
    field.parse().unwrap_or_else(|error| panic!("{field}: {error}"))
}

fn relative_error(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs()
}

fn assert_table(name: &str, table: &'static str, evaluate: impl Fn(&[&str]) -> (f64, f64)) {
    let mut worst = 0.0f64;
    let mut failures = Vec::new();
    let mut count = 0;
    for row in rows(table) {
        let (got, want) = evaluate(&row);
        let error = relative_error(got, want);
        worst = worst.max(error);
        count += 1;
        if !(error <= RELATIVE_ACCURACY) {
            failures.push(format!("{row:?}: got {got:e}, reference {want:e}, relative error {error:.3e}"));
        }
    }
    assert!(count > 0, "{name}: empty table");
    println!("{name}: {count} rows, worst relative error {worst:.3e}");
    assert!(failures.is_empty(), "{name}: {} of {count} rows off by more than {RELATIVE_ACCURACY:e}:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn chi_square_sf_matches_its_references() {
    assert_table("chi_square", include_str!("tail_references/chi_square.tsv"), |row| {
        (chi_square_sf(number(row[1]), number(row[0])), number(row[2]))
    });
}

#[test]
fn student_t_two_sided_probability_matches_its_references() {
    assert_table("student_t", include_str!("tail_references/student_t.tsv"), |row| {
        (student_t_two_sided_probability(number(row[1]), number(row[0])), number(row[2]))
    });
}

#[test]
fn fisher_snedecor_sf_matches_its_references() {
    assert_table("fisher_snedecor", include_str!("tail_references/fisher_snedecor.tsv"), |row| {
        (fisher_snedecor_sf(number(row[2]), number(row[0]), number(row[1])), number(row[3]))
    });
}

fn weighted_cases() -> Vec<(&'static str, Vec<WeightedChiSquareTerm>)> {
    let mut cases: Vec<(&'static str, Vec<WeightedChiSquareTerm>)> = Vec::new();
    for row in rows(include_str!("tail_references/weighted_terms.tsv")) {
        let term = WeightedChiSquareTerm { weight: number(row[1]), degrees_of_freedom: number(row[2]) };
        match cases.last_mut() {
            Some((name, terms)) if *name == row[0] => terms.push(term),
            _ => cases.push((row[0], vec![term])),
        }
    }
    cases
}

fn terms_of<'a>(cases: &'a [(&'static str, Vec<WeightedChiSquareTerm>)], name: &str) -> &'a [WeightedChiSquareTerm] {
    &cases.iter().find(|(case, _)| *case == name).unwrap_or_else(|| panic!("no case {name}")).1
}

/// The bound is honest (`|p̂ − P| ≤ r·P`, less the reference's own rounding) and tight
/// (`r ≤` [`RELATIVE_ACCURACY`]) at every reference, from `1 − 1e-12` down to `1e-300`, on
/// positive, mixed and all-negative weights, fractional degrees of freedom, fifteen decades
/// of weight and a thousand components.
#[test]
fn signed_weighted_chi_square_sf_matches_its_references_within_its_bound() {
    let cases = weighted_cases();
    let mut failures = Vec::new();
    let mut worst = 0.0f64;
    let mut count = 0;
    for row in rows(include_str!("tail_references/weighted_chi_square.tsv")) {
        let terms = terms_of(&cases, row[0]);
        let statistic = number(row[1]);
        let want = number(row[2]);
        let TailProbability { probability, relative_error: bound } = signed_weighted_chi_square_sf(terms, statistic);
        let error = relative_error(probability, want);
        worst = worst.max(error);
        count += 1;
        if !(error <= bound + REFERENCE_RELATIVE_ERROR) {
            failures.push(format!("{row:?}: got {probability:e} ± {bound:.3e} relative, but off by {error:.3e}"));
        }
        if !(bound <= RELATIVE_ACCURACY) {
            failures.push(format!("{row:?}: bound {bound:.3e} (actual error {error:.3e}) is not below {RELATIVE_ACCURACY:e}"));
        }
    }
    println!("weighted: {count} rows, worst relative error {worst:.3e}");
    assert!(count > 0);
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// The exact tail is non-increasing in the statistic, and so is the computed one.
///
/// On a coarse grid, over every case and across the whole range the references span, the
/// values themselves must not increase: neighbouring statistics there are far enough apart
/// that the tail moves by more than its bound. On a fine grid — consecutive doubles, where
/// the exact tail moves by less than one rounding — the value can only be asked not to
/// CONTRADICT monotonicity: the upper end of each interval must reach the lower end of the
/// next.
#[test]
fn signed_weighted_chi_square_sf_is_monotone_in_the_statistic() {
    let cases = weighted_cases();
    let references: Vec<Vec<&str>> = rows(include_str!("tail_references/weighted_chi_square.tsv")).collect();
    for (name, terms) in &cases {
        let statistics: Vec<f64> = references.iter().filter(|row| row[0] == *name).map(|row| number(row[1])).collect();
        let low = statistics.iter().copied().fold(f64::INFINITY, f64::min);
        let high = statistics.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let points = 200;
        let mut previous: Option<(f64, TailProbability)> = None;
        for k in 0..=points {
            let statistic = low + (high - low) * f64::from(k) / f64::from(points);
            let tail = signed_weighted_chi_square_sf(terms, statistic);
            assert!(tail.relative_error < 1.0, "{name} at {statistic}: unresolved {tail:?}");
            if let Some((before, earlier)) = previous {
                assert!(
                    tail.probability <= earlier.probability,
                    "{name}: P(Q > {statistic:e}) = {:e} exceeds P(Q > {before:e}) = {:e}",
                    tail.probability,
                    earlier.probability,
                );
            }
            previous = Some((statistic, tail));
        }
        for &centre in &statistics {
            let mut statistic = centre;
            let mut earlier = signed_weighted_chi_square_sf(terms, statistic);
            for _ in 0..64 {
                let next = statistic.next_up();
                let tail = signed_weighted_chi_square_sf(terms, next);
                assert!(
                    earlier.probability * (1.0 + earlier.relative_error)
                        >= tail.probability * (1.0 - tail.relative_error),
                    "{name}: the intervals at {statistic:e} and {next:e} ({earlier:?}, {tail:?}) contradict monotonicity",
                );
                statistic = next;
                earlier = tail;
            }
        }
    }
}

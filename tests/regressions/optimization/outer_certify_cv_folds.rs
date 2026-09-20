//! pyGAM audit ACC-3: the outer REML/LAML search must certify convergence on
//! ordinary cross-validation folds of small binomial smooths.
//!
//! Each fixture under `tests/data/pygam_outer_certify/` is one training fold
//! (5-fold `StratifiedKFold(shuffle=True, random_state=0)`) that the audit saw
//! refused with `NOT STATIONARY` at the outer tail snap:
//!
//! | fold                       | refusal                                   |
//! |----------------------------|-------------------------------------------|
//! | `binom_add4_n300` fold 0   | `|Pg| 1.456e-3 > 1.343e-3`, line search   |
//! | `nearsep_n200` fold 0, 2   | `|Pg| 8.5e-3 > 2.2e-4`, `StepSizeTooSmall` |
//! | `haberman` fold 2          | `|Pg| 7.5e-6 > 3.65e-6`, `hessian_psd=NO` |
//! | `haberman` fold 4          | `|Pg| 6.99e-3 > 3.65e-6`                  |
//! | `haberman` fold 0, `k = 20` | `|Pg| 5.709e-8 > 1.968e-10`, BFGS cost stall |
//!
//! The `nearsep` folds are the ones the repair in
//! `Gam784BlockTarget::excess` addresses. The #784 block-local Gauss–Hermite
//! correction used to rebuild the penalty's linear term `Σ_k λ_k (S_k β̂)·δ`
//! from a stiff `λ ≈ 8.4e11` times an `S_k β̂` that is a `~3e-15` rounding
//! residue, which put a spurious `±4e-4` linear term along a weak block
//! direction and made `Δ_b` jitter by up to `1.6e-3` between evaluations at
//! the same ρ — so no line search could make progress and the fit refused.
//! The remainder is now written from the mode condition `Sβ̂ = −Xᵀψ'(η̂)`,
//! which reads only the likelihood and carries no λ at all.
//!
//! The `binom_add4_n300` and `haberman` folds already certify on the code this
//! regression was written against; they are kept so the certificate on the
//! whole set the audit reported is pinned.
//!
//! `haberman` fold 0 at `k = 20` is the fold PR #3127's smoothing-correction
//! tests fit (#3239). At a6731fd3 its search stalled on BFGS curvature at
//! `|Pg| = 5.709e-8` against the per-coordinate gradient band's (#3190)
//! `1.968e-10`, and the fit was refused.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

const FOLD_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/pygam_outer_certify"
);

fn fold_dataset(name: &str) -> gam::data::EncodedDataset {
    let path = format!("{FOLD_DIR}/{name}.csv");
    let mut reader = csv::Reader::from_path(&path)
        .unwrap_or_else(|error| panic!("open fold fixture {path}: {error}"));
    let headers: Vec<String> = reader
        .headers()
        .expect("fold fixture header row")
        .iter()
        .map(str::to_string)
        .collect();
    let records: Vec<StringRecord> = reader
        .records()
        .map(|record| record.expect("fold fixture row"))
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode fold fixture")
}

fn assert_fold_certifies(name: &str, formula: &str) {
    init_parallelism();
    let data = fold_dataset(name);
    let config = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let fit = gam::fit_from_formula(formula, &data, &config).unwrap_or_else(|error| {
        panic!("ACC-3: `{formula}` on CV fold `{name}` must fit to a certified optimum: {error}")
    });
    let gam::FitResult::Standard(standard) = &fit else {
        panic!("ACC-3: `{name}` is a standard binomial GAM fit");
    };
    let reml_score = standard.fit.reml_score();
    assert!(
        reml_score.is_some_and(f64::is_finite),
        "ACC-3: CV fold `{name}` minted a fit with no finite REML/LAML criterion: {reml_score:?}"
    );
}

#[test]
fn binom_add4_n300_fold0_outer_certifies() {
    assert_fold_certifies(
        "binom_add4_n300_fold0",
        "y ~ s(x0) + s(x1) + s(x2) + s(x3)",
    );
}

#[test]
fn nearsep_n200_fold0_outer_certifies() {
    assert_fold_certifies("nearsep_n200_fold0", "y ~ s(x) + s(z)");
}

#[test]
fn nearsep_n200_fold2_outer_certifies() {
    assert_fold_certifies("nearsep_n200_fold2", "y ~ s(x) + s(z)");
}

#[test]
fn haberman_fold2_outer_certifies() {
    assert_fold_certifies("haberman_fold2", "y ~ s(age) + s(year) + s(nodes)");
}

#[test]
fn haberman_fold4_outer_certifies() {
    assert_fold_certifies("haberman_fold4", "y ~ s(age) + s(year) + s(nodes)");
}

#[test]
fn haberman_k20_fold0_outer_certifies_3239() {
    assert_fold_certifies(
        "haberman_fold0",
        "y ~ s(age, k=20) + s(year, k=20) + s(nodes, k=20)",
    );
}

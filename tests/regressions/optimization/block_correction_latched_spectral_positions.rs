//! pyGAM audit, convergence fuzzer (`bench/convergence_fuzz`): a default
//! additive fit must certify when the #784 block-local Gauss–Hermite
//! correction is admitted.
//!
//! Once the correction is admitted its block dimension `m` is latched for the
//! fit (#2748), and every later evaluation must integrate the SAME block. The
//! block used to be re-selected at every ρ as the `m` largest-`|γ_r|`
//! positive-curvature directions. That keeps the cardinality but not the
//! block: wherever two directions' `|γ|` cross (a codimension-one surface in
//! ρ) the selection swaps one direction's whole contribution to `Δ_b` for
//! another's, and the latched axis orders are silently reassigned to
//! directions they were never certified on. The criterion then has a jump on
//! that surface, the outer search stalls on it, and the fit is refused. On the
//! fuzzer's `case0/binomial/n1000` `Δ_b` took exactly two values, `4.660e-2`
//! and `5.797e-2`, across the BFGS polish's trial points at `|g| = 1.4e-4`.
//!
//! The block is now latched as its SPECTRAL POSITIONS in the ascending
//! eigen-order of the penalized Hessian. The eigenvector at a fixed position
//! moves continuously with ρ away from an eigenvalue coincidence, which a path
//! through ρ avoids generically, so the criterion is continuous again and the
//! frame-rotation channel differentiates exactly that motion.
//!
//! The fixture is that same `case0/binomial/n1000` training set (eight
//! covariates, `bench/convergence_fuzz/dgp.py`, written with `repr` so every
//! value round-trips exactly). Before the repair it was refused with
//! `NOT STATIONARY (|Pg|=1.065e-2 > bound=1.490e-5)`.
//!
//! The evaluation-level tests (#2623, #2748) build a fresh `RemlState` per
//! call, so the latch there lives for one evaluation and never sees a later ρ;
//! only a whole fit exercises a latch held across the outer search.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/pygam_conv_fuzz_additive/case0_binomial_n1000.csv"
);

fn fixture_dataset() -> gam::data::EncodedDataset {
    let mut reader = csv::Reader::from_path(FIXTURE)
        .unwrap_or_else(|error| panic!("open fuzzer fixture {FIXTURE}: {error}"));
    let headers: Vec<String> = reader
        .headers()
        .expect("fuzzer fixture header row")
        .iter()
        .map(str::to_string)
        .collect();
    let records: Vec<StringRecord> = reader
        .records()
        .map(|record| record.expect("fuzzer fixture row"))
        .collect();
    assert_eq!(records.len(), 1000, "case0/binomial/n1000 carries 1000 rows");
    encode_recordswith_inferred_schema(headers, records).expect("encode fuzzer fixture")
}

#[test]
fn latched_block_correction_keeps_its_block_and_the_fit_certifies() {
    init_parallelism();
    let data = fixture_dataset();
    let config = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let formula = "y ~ s(x0) + s(x1) + s(x2) + s(x3) + s(x4) + s(x5) + s(x6) + s(x7)";
    let fit = gam::fit_from_formula(formula, &data, &config).unwrap_or_else(|error| {
        panic!(
            "`{formula}` on the fuzzer's case0/binomial/n1000 must fit to a certified optimum. \
             A latched #784 block that is re-selected at each rho jumps wherever two \
             directions' |gamma| cross, and no line search can cross that: {error}"
        )
    });
    let gam::FitResult::Standard(standard) = &fit else {
        panic!("case0/binomial/n1000 is a standard binomial GAM fit");
    };
    let reml_score = standard.fit.reml_score();
    assert!(
        reml_score.is_some_and(f64::is_finite),
        "case0/binomial/n1000 minted a fit with no finite REML/LAML criterion: {reml_score:?}"
    );
}

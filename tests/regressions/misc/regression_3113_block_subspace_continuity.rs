//! #3113: under a latched #784 block-local correction the block must be the
//! continuous transport of the admitted one, so the corrected criterion is a
//! smooth function of ρ and the outer search can certify its optimum.
//!
//! The block used to be re-selected at every ρ as the `m` largest-|γ_r|
//! positive-curvature directions. That ranking swaps members wherever two
//! |γ_r| cross, and each swap is a jump of a whole direction's contribution to
//! `Δ_b`. On this 21-row binomial fit (p = 3, admitted block m = 2) the second
//! block axis traded places between adjacent ρ, `Δ_b` alternated between
//! 4.24e-1 and 1.12e-1 across a step of 9e-4 in ρ, and the fit refused with
//! `NOT STATIONARY` at |g| = 4.8e-2.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

/// `sklearn.datasets.make_blobs(n_samples=21, random_state=0)`, the response
/// binarised as `y == min(y) -> 0, else 1`: the fixture #3113 was reported on.
const BLOBS: [(f64, f64, f64); 21] = [
    (2.2102149471297943, 1.2758261795401105, 1.0),
    (1.2893377801973964, 3.449691588146665, 0.0),
    (2.101026038734324, 0.7104798099121032, 1.0),
    (2.9197037202923837, 0.1554986395314949, 1.0),
    (-2.5754569782889982, 1.4978643241541476, 1.0),
    (-0.49772229440120075, 1.5512822553782974, 1.0),
    (1.4201333112919208, 4.637461654822657, 0.0),
    (1.1203136497073731, 5.758060834411365, 0.0),
    (-3.233174203846918, 4.8686576565649125, 1.0),
    (3.5880467357913357, 2.367022429838222, 1.0),
    (-1.874816162548058, 3.074231230437103, 1.0),
    (1.9263584960720843, 4.152430119150692, 0.0),
    (-2.036556194973559, 2.4798079597219362, 1.0),
    (4.325022145420485, -0.5567020146608279, 1.0),
    (2.4703491517041014, 4.098629063682589, 0.0),
    (0.8730512267529372, 4.714385829386762, 0.0),
    (-1.9142308306298577, 2.615579510757787, 1.0),
    (1.1674817738027652, -1.0831328082859901, 1.0),
    (-2.7796993732718316, 3.6953726171650327, 1.0),
    (1.7373078036934886, 4.425462343941218, 0.0),
    (-0.29661333249418464, 4.120262110117534, 1.0),
];

fn blobs_dataset() -> gam::data::EncodedDataset {
    let headers = vec!["x0".to_string(), "x1".to_string(), "y".to_string()];
    let records: Vec<StringRecord> = BLOBS
        .iter()
        .map(|(x0, x1, y)| StringRecord::from(vec![x0.to_string(), x1.to_string(), y.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode the blobs fixture")
}

#[test]
fn latched_block_correction_binomial_blobs_fit_converges_3113() {
    init_parallelism();
    let data = blobs_dataset();
    let config = FitConfig {
        family: Some("binomial-logit".to_string()),
        ..FitConfig::default()
    };
    let formula = "y ~ x0 + x1";
    let fit = gam::fit_from_formula(formula, &data, &config).unwrap_or_else(|error| {
        panic!(
            "#3113: `{formula}` must fit. A block re-selected by |gamma| ranking at every rho \
             swaps a whole direction's contribution to Delta_b in and out of the criterion, and \
             the outer search cannot certify a stationary point of a discontinuous function: \
             {error}"
        )
    });
    let gam::FitResult::Standard(standard) = &fit else {
        panic!("#3113: the blobs fit is a standard GLM fit");
    };
    let reml_score = standard.fit.reml_score().unwrap_or(f64::NAN);
    assert!(
        reml_score.is_finite(),
        "#3113: `{formula}` minted a fit with no finite REML/LAML criterion"
    );
}

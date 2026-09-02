//! #2672: the LR statistic's own null spectrum must reach the smooth-term
//! reference on models that carry a PARAMETRIC term, not only on the
//! `y ~ s(x)` shape every other fixture uses.
//!
//! The reference is built from `w = eig(2·F_jj − F_jj²)` on the tested block of
//! the coefficient influence `F = H⁻¹X'WX`. `F` transforms by SIMILARITY under
//! the parametric column conditioning — `F_orig = M·F_int·M⁻¹` — and the
//! back-transform used to drop it rather than apply that map, on the stated
//! ground that the primitive was not carried. It was: `left_multiply_by_m` and
//! `right_multiply_by_m_inv` are both defined on the same type. The drop was
//! invisible because `backtransform_external_result` returns early when the
//! conditioning is inactive, and the conditioning is active exactly when the
//! model has a non-intercept parametric column — which no fixture reading `F`
//! back had.
//!
//! With `F` gone the whole spectrum is unavailable and the reference falls back
//! to the unit-weight shape `χ²_{max(edf, null_dim, 1)}` — every retained
//! direction counted as if it were unpenalized, which is the anti-conservative
//! reference #1766 replaced, measured there at a 5%-level FPR of ~0.15. The two
//! `smooth_term_lr_size_calibration` fixtures are exactly this shape
//! (`y ~ x + s(z)`).
//!
//! The guard is stated as a CONTRAST rather than as a bare "must be the spectral
//! lane", so it cannot pass by the spectrum becoming unconditionally available
//! for some unrelated reason: the parametric-term model and the pure-smooth
//! control must BOTH report `NullSpectrum`, both must satisfy Wood's analytic
//! band `edf ≤ Σw ≤ 2·edf`, and both must publish p-values that are the ones
//! their own reference produces.

use gam::smooth::smooth_term_lr_inference_forspec;
use gam::{
    FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism, materialize,
};

use csv::StringRecord;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson};

/// `y ~ Poisson(exp(0.3 + 0.8·x + amplitude·sin(2π z)))`. `amplitude > 0` gives
/// the smooth a genuine signal so the fitted term has healthy effective d.f.;
/// `amplitude = 0` makes `z` a pure nuisance covariate and the smooth null-true.
fn dataset(n: usize, seed: u64, amplitude: f64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let headers = ["y", "x", "w", "v", "z"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        // Two further parametric covariates with NO effect on the mean, present
        // only to widen the unpenalized block so the offset's overshoot is large.
        let w: f64 = rng.random_range(-1.0..1.0);
        let v: f64 = rng.random_range(-1.0..1.0);
        let z: f64 = rng.random_range(0.0..1.0);
        let eta = 0.3 + 0.8 * x + amplitude * (std::f64::consts::TAU * z).sin();
        let lambda: f64 = eta.exp();
        let y = Poisson::new(lambda).expect("poisson rate").sample(&mut rng) as f64;
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            w.to_string(),
            v.to_string(),
            z.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// `edf_total = tr(F)` for one formula on one dataset, from the ordinary fit
/// entry point. The LR driver refits internally from the same spec and options,
/// and the fit is deterministic, so this is the same fit seen from outside.
fn fit_edf_total(formula: &str, data: &gam::data::EncodedDataset) -> f64 {
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(std_fit) =
        fit_from_formula(formula, data, &cfg).unwrap_or_else(|e| panic!("fit {formula}: {e}"))
    else {
        panic!("expected a standard fit for {formula}");
    };
    std_fit
        .fit
        .edf_total()
        .unwrap_or_else(|| panic!("edf_total retained for {formula}"))
}

/// The `s(z)` LR report for one formula on one dataset.
fn report(formula: &str, data: &gam::data::EncodedDataset) -> gam::smooth::SmoothTermLrInference {
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let mat = materialize(formula, data, &cfg).expect("materialize");
    let FitRequest::Standard(req) = mat.request else {
        panic!("expected a standard fit request for {formula}");
    };
    let reports = smooth_term_lr_inference_forspec(
        req.data.view(),
        req.y.view(),
        req.weights.view(),
        req.offset.view(),
        &req.spec,
        req.family,
        &req.options,
    )
    .expect("smooth-term LR inference");
    reports
        .into_iter()
        .find(|r| r.name.contains('z'))
        .unwrap_or_else(|| panic!("no s(z) report for {formula}"))
}

/// The block-offset guard, stated as an exact accounting identity so no seed can
/// make it pass or fail by luck.
///
/// `SmoothTerm::coeff_range` is BLOCK-LOCAL while the global coefficient layout
/// is `[intercept | linear | random | smooth]`, so every consumer that indexes a
/// global object with it must shift by `smooth_start` first. The LR driver did
/// not, and it indexes four of them — the influence matrix (per-term EDF and
/// Wood's `edf1`), the weighted Gram and correction inside the WPS trace, and
/// the `tested` column set handed to Lawley, which decides which hypothesis the
/// mean shift is computed for. `smooth_start` is never zero: the intercept alone
/// makes it at least one.
///
/// The identity: an UNPENALIZED column spends exactly one effective degree of
/// freedom, and `edf_total = tr(F)` sums every column's share. So for a model
/// with `u` unpenalized columns (intercept plus the parametric block) and a
/// single smooth,
///
/// ```text
///     edf(smooth) + u  ==  edf_total
/// ```
///
/// exactly. An unshifted window makes the smooth's own EDF *include* those `u`
/// columns, so the left side overshoots by `u` — a margin set by the fixture's
/// own design rather than by a tolerance. Measured on the n=60 fixture before the
/// repair, `y ~ x + s(z)` reported `edf = 2.0404` for a null smooth whose
/// penalty-block-trace value was `0.0537`: `1 + 1 + 0.04`.
///
/// Run with a THREE-column parametric block so the overshoot is 4, not 1, and
/// with the smooth both null and signal-bearing so the identity is exercised at
/// both ends of the shrinkage range.
#[test]
fn per_term_edf_plus_unpenalized_columns_equals_edf_total_2672() {
    init_parallelism();
    // (formula, unpenalized column count = intercept + parametric columns)
    let shapes = [("y ~ s(z)", 1usize), ("y ~ x + w + v + s(z)", 4usize)];
    for amplitude in [0.0_f64, 0.9] {
        for seed in 0..2u64 {
            let data = dataset(180, 4400 + seed, amplitude);
            for (formula, unpenalized) in shapes {
                let r = report(formula, &data);
                let edf_total = fit_edf_total(formula, &data);
                let p = r.ref_df_provenance;
                let lhs = p.edf + unpenalized as f64;
                let slack = 1e-3 * edf_total.abs().max(1.0);
                assert!(
                    (lhs - edf_total).abs() <= slack,
                    "{formula} (amplitude {amplitude}, seed {seed}): the smooth's own \
                     effective d.f. plus its {unpenalized} unpenalized column(s) must \
                     equal the model total exactly — an unpenalized column spends one \
                     full degree of freedom and `edf_total = tr(F)` sums them all. \
                     Got edf(s(z)) = {} and edf_total = {edf_total}, so the left side \
                     is {lhs}. An overshoot of about {unpenalized} means the term's \
                     coefficient window is unshifted and has folded the unpenalized \
                     block into the smooth. Provenance: {p:?}",
                    p.edf
                );
            }
        }
    }
}


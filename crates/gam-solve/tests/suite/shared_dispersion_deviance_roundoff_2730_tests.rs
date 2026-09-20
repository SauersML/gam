//! #2730: the shared-dispersion REML penalty gradient's forward-deviance bar
//! must be denominated in the arithmetic that produced the deviance.
//!
//! `gaussian_reml_multi_shared_dispersion_closed_form` forms its pooled deviance
//! by cancellation — `pooled_ywy − Σ_k c_k²/(1 + λ·δ_k)` — and the gradient used
//! to admit it on a bare `pooled_deviance > 0.0`. That predicate is
//! un-denominated in exactly the near-interpolating regime the gradient's own
//! comment describes: a value sitting at the roundoff floor of that subtraction
//! is positive, is finite, and carries no significant digit, yet it is then used
//! as the DENOMINATOR of `deviance_scale = ½·ν·λ/pooled_deviance`, so the debris
//! is amplified by `1/floor` into every entry a nested metric optimizer follows.
//!
//! The #2694 harvest chart that produced the reported `pooled_deviance =
//! 1.7763568394002505e-15 = 2 ulps of ywy` is not in this repository, as the
//! issue thread records. The regime is therefore reproduced rather than
//! replayed: a genuine, fully validated forward state is given a pooled deviance
//! at that same measured ratio to its own response energy. Nothing else about
//! the state is altered, and `validate_gaussian_reml_forward_fit` accepts it —
//! which is the point, since it is exactly the state the harvest fit handed
//! forward.
//!
//! The arms are separate `#[test]` functions because as one function the first
//! assertion aborts before the later properties ever run, and an arm that never
//! executes is not an arm.

use gam_solve::gaussian_reml::{
    GaussianRemlMultiResult, gaussian_reml_multi_shared_dispersion_closed_form,
    gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit,
};
use ndarray::{Array1, Array2};

const ROWS: usize = 46;
const COEFFICIENTS: usize = 8;
const RESPONSES: usize = 1;

fn design() -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((ROWS, COEFFICIENTS));
    for row in 0..ROWS {
        let t = (row as f64) / ((ROWS - 1) as f64);
        for col in 0..COEFFICIENTS {
            x[[row, col]] = (std::f64::consts::PI * (col as f64) * t).cos();
        }
    }
    x
}

/// Second-difference penalty: rank 6, nullity 2 on eight coefficients.
fn penalty() -> Array2<f64> {
    let bands = COEFFICIENTS - 2;
    let mut difference = Array2::<f64>::zeros((bands, COEFFICIENTS));
    for band in 0..bands {
        difference[[band, band]] = 1.0;
        difference[[band, band + 1]] = -2.0;
        difference[[band, band + 2]] = 1.0;
    }
    let mut s = Array2::<f64>::zeros((COEFFICIENTS, COEFFICIENTS));
    for row in 0..COEFFICIENTS {
        for col in 0..COEFFICIENTS {
            let mut acc = 0.0;
            for band in 0..bands {
                acc += difference[[band, row]] * difference[[band, col]];
            }
            s[[row, col]] = acc;
        }
    }
    s
}

/// A comfortably resolved response: the healthy end of the sweep, so the control
/// arm cannot pass for a near-interpolating reason.
fn response(x: &Array2<f64>) -> Array2<f64> {
    let mut beta = Array1::<f64>::zeros(COEFFICIENTS);
    for index in 0..COEFFICIENTS {
        beta[index] = 0.4 * (-(index as f64) * 0.35).exp() * if index % 2 == 0 { 1.0 } else { -1.0 };
    }
    let mut y = Array2::<f64>::zeros((ROWS, RESPONSES));
    for row in 0..ROWS {
        let mut mean = 0.0;
        for col in 0..COEFFICIENTS {
            mean += x[[row, col]] * beta[col];
        }
        y[[row, 0]] = mean + 1.0e-3 * ((row as f64) * 2.399_963_229_728_653).sin();
    }
    y
}

struct Fixture {
    x: Array2<f64>,
    s: Array2<f64>,
    y: Array2<f64>,
    fit: GaussianRemlMultiResult,
    /// `Σ_{i,j} w_i·y_ij²`, the magnitude that bounds both accumulations whose
    /// difference is the pooled deviance. Weights are absent here, so `w_i = 1`.
    response_energy: f64,
    /// Residual degrees of freedom the gradient uses to turn `σ̂²` into the
    /// pooled deviance.
    shared_nu: f64,
    /// Independent restatement of the forward-error bound the repaired guard
    /// uses: `γ_m·(response energy)` with `m = 3·n·d + 2·p + 1` — two
    /// multiplications and one accumulation per weighted response entry, one
    /// scale multiply and one accumulation per penalty eigendirection, and the
    /// final subtraction. Every input is a machine constant or a dimension the
    /// problem already has; there is nothing here to tune.
    deviance_roundoff: f64,
}

fn fixture() -> Fixture {
    let x = design();
    let s = penalty();
    let y = response(&x);
    let fit =
        gaussian_reml_multi_shared_dispersion_closed_form(x.view(), y.view(), s.view(), None, None)
            .expect("healthy shared-dispersion forward fit");

    let response_energy = y.iter().map(|value| value * value).sum::<f64>();
    let shared_nu = (RESPONSES * (ROWS - fit.cache.nullity)) as f64;
    let operations = 3 * ROWS * RESPONSES + 2 * COEFFICIENTS + 1;
    let accumulated = (operations as f64) * 0.5 * f64::EPSILON;
    let deviance_roundoff = (accumulated / (1.0 - accumulated)) * response_energy;

    assert_eq!(fit.cache.nullity, 2, "fixture must have nullity 2");
    assert!(
        response_energy.is_finite() && response_energy > 0.0,
        "precondition: the response must carry energy for the floor to be denominated in it"
    );
    assert!(
        deviance_roundoff.is_finite() && deviance_roundoff > 0.0,
        "precondition: the derived floor must be a usable positive magnitude, got \
         {deviance_roundoff:e}"
    );

    Fixture {
        x,
        s,
        y,
        fit,
        response_energy,
        shared_nu,
        deviance_roundoff,
    }
}

/// Reproduce the near-interpolating regime on a validated forward state by
/// setting the shared dispersion so the pooled deviance takes the requested
/// value. Only `sigma2` moves; the design, response, penalty, cache,
/// λ and coefficients are the ones the forward solve produced.
fn with_pooled_deviance(case: &Fixture, pooled_deviance: f64) -> GaussianRemlMultiResult {
    let mut fit = case.fit.clone();
    fit.sigma2 = Array1::from_elem(RESPONSES, pooled_deviance / case.shared_nu);
    fit
}

/// THE WITNESS. A pooled deviance at the ratio the #2694 harvest fit measured —
/// two ulps of its own response energy — is positive and finite, so the old
/// `pooled_deviance > 0.0` bar admitted it and then inverted it. It must now be
/// refused, because there is no finite limit to substitute: `deviance_scale`
/// diverges as the chart approaches interpolation.
#[test]
fn a_pooled_deviance_at_the_roundoff_floor_of_its_own_formation_is_refused() {
    let case = fixture();
    let pooled_deviance = 2.0 * f64::EPSILON * case.response_energy;

    // The old bar's own predicate, asserted rather than assumed: this value is
    // exactly what `> 0.0` lets through, so a red here at BASE is a red for the
    // reason this issue names.
    assert!(
        pooled_deviance.is_finite() && pooled_deviance > 0.0,
        "the witness must be admitted by the un-denominated bar, otherwise this test \
         proves nothing about it: pooled_deviance={pooled_deviance:e}"
    );
    assert!(
        pooled_deviance < case.deviance_roundoff,
        "the witness must sit BELOW the derived floor for the repaired bar to be the \
         thing under test: pooled_deviance={pooled_deviance:e}, floor={:e}",
        case.deviance_roundoff
    );

    let tampered = with_pooled_deviance(&case, pooled_deviance);
    let result = gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit(
        case.x.view(),
        case.y.view(),
        case.s.view(),
        None,
        &tampered,
    );
    let error = match result {
        Err(error) => error.to_string(),
        Ok(gradient) => {
            let max_abs = gradient.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            panic!(
                "a pooled deviance of {pooled_deviance:e} — {:.2e} times the response energy \
                 {:e}, i.e. at the roundoff floor {:e} of the cancellation that forms it — was \
                 accepted and inverted into deviance_scale, returning a gradient with \
                 max|entry| = {max_abs:e}",
                pooled_deviance / case.response_energy,
                case.response_energy,
                case.deviance_roundoff
            )
        }
    };
    assert!(
        error.contains("resolved above the roundoff of its own formation"),
        "the refusal must name the regime it covers rather than repeating the old \
         un-denominated wording; got: {error}"
    );
}

/// NON-VACUITY, and the reason the arm above is not a bar that refuses
/// everything. The forward fit's own dispersion — a genuinely resolved deviance
/// on a chart that is nowhere near interpolating — must still be served.
#[test]
fn a_resolved_pooled_deviance_is_still_served() {
    let case = fixture();
    let pooled_deviance = case.fit.sigma2[0] * case.shared_nu;
    assert!(
        pooled_deviance > case.deviance_roundoff,
        "precondition: this fixture is supposed to be the HEALTHY control, but its pooled \
         deviance {pooled_deviance:e} does not clear its own roundoff floor {:e}",
        case.deviance_roundoff
    );

    gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit(
        case.x.view(),
        case.y.view(),
        case.s.view(),
        None,
        &case.fit,
    )
    .expect("the repaired bar must not refuse a resolved forward deviance");
}

/// The verdict must be decided by the derived floor and by nothing else. Straddle
/// it by a factor of two from a single fixture: below it the gradient is refused,
/// above it the SAME state with the SAME everything else is served. A bar sitting
/// anywhere other than this magnitude cannot produce both outcomes.
#[test]
fn the_derived_floor_and_not_some_other_magnitude_decides_the_verdict() {
    let case = fixture();

    let below = with_pooled_deviance(&case, 0.5 * case.deviance_roundoff);
    let below_result = gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit(
        case.x.view(),
        case.y.view(),
        case.s.view(),
        None,
        &below,
    );
    assert!(
        below_result.is_err(),
        "a pooled deviance at half the derived roundoff floor {:e} was served",
        case.deviance_roundoff
    );

    let above = with_pooled_deviance(&case, 2.0 * case.deviance_roundoff);
    let above_result = gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit(
        case.x.view(),
        case.y.view(),
        case.s.view(),
        None,
        &above,
    );
    assert!(
        above_result.is_ok(),
        "a pooled deviance at twice the derived roundoff floor {:e} was refused: {:?}",
        case.deviance_roundoff,
        above_result.err()
    );
}

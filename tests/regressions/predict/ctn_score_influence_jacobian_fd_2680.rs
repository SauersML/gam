//! Regression (#2680): the CTN score-influence Jacobian `J = ∂z/∂θ₁` must be the
//! derivative of the score `score_influence_jacobian` itself emits.
//!
//! ## Why this is a separate test from the value-agreement one
//!
//! `score_influence_jacobian` returns two things computed from the same
//! coefficients: the out-of-fold generated regressor `z` (which the calibrated
//! marginal-slope chain consumes as `z_oof`) and its Jacobian with respect to
//! the Stage-1 coefficients (which becomes the Murphy–Topel absorbed influence
//! block, i.e. the correction that makes Stage-2's β estimating equation
//! orthogonal to Stage-1's estimation error). Those two halves are separately
//! wrong-able, and before #2680 both were: the value path evaluated the
//! pre-`#2306` squared chart `Σ_k I_k(y)·γ_k(x)²`, and the Jacobian carried its
//! matching `2·γ_k` shape factor.
//!
//! Fixing only the value would leave the Jacobian differentiating a function
//! nobody computes — the failure mode is silent, because a Murphy–Topel
//! correction that is wrong by a factor of `2·α_k` still produces finite,
//! plausible standard errors. So the value fix needs its own gate, and the gate
//! has to be **the derivative of the emitted `z`, not of an independently
//! re-derived formula**: differentiating a second transcription of the chart
//! would pass on any chart the two transcriptions happened to share.
//!
//! ## The gate
//!
//! Perturb one Stage-1 coefficient `A[k, j]` by `±δ`, re-run the *production*
//! `score_influence_jacobian` on the perturbed fit, and compare the central
//! difference of its own `z` against the `J` column the unperturbed call
//! reported. This is exact-to-`O(δ²)` for every column, including the shape rows
//! where the chart factor lived, and it is insensitive to which chart is
//! correct — it only asks that the value and the derivative agree.
//!
//! Rows whose score has saturated at a PIT clip boundary are excluded: there `z`
//! is the constant boundary quantile and `J` is identically zero by design, so a
//! finite difference across the boundary measures a kink rather than a
//! derivative. The test asserts that the surviving population is large.
//!
//! Each column is judged against its own magnitude rather than a fixed absolute
//! floor. The disagreement this gate exists to catch is a *factor* on the
//! column (`2·α_k`), so a fraction-of-the-column bar is the natural metric; a
//! fixed floor would instead grade the rows where the sensitivity happens to be
//! near zero, where a central difference has nothing left but truncation and the
//! test would be reporting its own step size.

use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema, fit_model,
    init_parallelism, materialize,
};
use gam_models::marginal_slope_orthogonal::score_influence_jacobian;
use gam_models::transformation_normal::TransformationNormalFitResult;
use ndarray::{Array1, Array2};

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        gam_linalg::utils::splitmix64(&mut self.state)
    }
    fn next_uniform(&mut self) -> f64 {
        let u = ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn ctn_score_influence_jacobian_matches_its_own_finite_difference_2680() {
    init_parallelism();
    // Deliberately the SAME fixture and formula as
    // `ctn_predict_score_reproduces_the_fitted_score_2680`. The CTN inner solve
    // refuses on some perfectly ordinary Gaussian fixtures for reasons that have
    // nothing to do with this issue (a 96-dimensional constraint face whose
    // truncated moments will not converge; a reduced-face KKT residual), so a
    // chart gate should not go shopping for its own fixture and take that risk
    // twice.
    const N: usize = 400;

    let mut rng = SplitMix64::new(4271);
    let headers = vec![
        "x1".to_string(),
        "x2".to_string(),
        "x3".to_string(),
        "y".to_string(),
    ];
    let mut records = Vec::with_capacity(N);
    let mut data = Array2::<f64>::zeros((N, 4));
    for i in 0..N {
        let x1 = rng.next_normal();
        let x2 = 0.3 * x1 + (1.0 - 0.09_f64).sqrt() * rng.next_normal();
        let x3 = rng.next_normal();
        let y = 0.4 * x1 - 0.2 * x2 + 0.15 * x3 + 0.9 * rng.next_normal();
        records.push(csv::StringRecord::from(vec![
            format!("{x1:.17e}"),
            format!("{x2:.17e}"),
            format!("{x3:.17e}"),
            format!("{y:.17e}"),
        ]));
        data[[i, 0]] = x1;
        data[[i, 1]] = x2;
        data[[i, 2]] = x3;
        data[[i, 3]] = y;
    }
    let dataset =
        encode_recordswith_inferred_schema(headers, records).expect("encode fixture dataset");

    let config = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let materialized =
        materialize("y ~ s(x1, k=5) + s(x2, k=5)", &dataset, &config).expect("materialize CTN");
    let FitRequest::TransformationNormal(_) = materialized.request else {
        panic!("expected a TransformationNormal fit request");
    };
    let FitResult::TransformationNormal(tn) =
        fit_model(materialized.request).expect("fit transformation-normal model")
    else {
        panic!("expected a TransformationNormal fit result");
    };

    let response: Array1<f64> = data.column(3).to_owned();
    let offset = Array1::<f64>::zeros(N);

    let base = score_influence_jacobian(&tn, &response, data.view(), &offset)
        .expect("score influence Jacobian at the fitted coefficients");
    let p1 = base.columns.ncols();
    assert_eq!(base.columns.nrows(), N);
    assert_eq!(base.z.len(), N);

    // The design width tells us `p_cov`; `p_resp = p1 / p_cov`. Rebuilding it
    // from the resolved spec is the same route `score_influence_jacobian` takes.
    let design = build_term_collection_design(data.view(), &tn.covariate_spec_resolved)
        .expect("rebuild covariate design");
    let p_cov = design.design.ncols();
    assert_eq!(p1 % p_cov, 0, "Jacobian width {p1} is not a multiple of p_cov {p_cov}");
    let p_resp = p1 / p_cov;
    assert!(
        p_resp >= 2,
        "fixture must have at least one shape coordinate; got p_resp={p_resp}"
    );

    // Re-run the production routine at a perturbed coefficient vector.
    let perturbed_scores = |index: usize, delta: f64| -> Array1<f64> {
        let mut moved: TransformationNormalFitResult = tn.clone();
        {
            let state = moved
                .fit
                .block_states
                .first_mut()
                .expect("one coefficient block");
            state.beta[index] += delta;
        }
        score_influence_jacobian(&moved, &response, data.view(), &offset)
            .expect("score influence Jacobian at the perturbed coefficients")
            .z
    };

    // Probe every response row `k` (the location column and every shape
    // coordinate — the shape rows are where the chart factor lived) at one
    // covariate column each, cycling so the covariate columns are covered too.
    //
    // Each column is judged against ITS OWN magnitude: `max_i |J[i, index]|`
    // over the unsaturated rows. A fixed absolute floor would instead measure
    // finite-difference truncation on the rows where the sensitivity happens to
    // be near zero — a statement about the step size, not about the chart.
    // Scaling by the column's own scale asks the question the test is for: is
    // this column the right column, up to a fraction of what it is.
    let mut checked_entries = 0usize;
    let mut checked_rows = 0usize;
    let mut worst_rel = 0.0_f64;
    let mut worst_abs = 0.0_f64;
    let mut worst_label = String::new();
    // The score saturates at ±Φ⁻¹(1 − 1e-12) ≈ ±7.034; `J` is identically zero
    // there by construction, so those rows are not a derivative test.
    const SATURATION: f64 = 7.0;

    for k in 0..p_resp {
        let j = k % p_cov;
        let index = k * p_cov + j;
        // Step scaled to the coefficient's own magnitude so the central
        // difference has headroom over round-off on both small and large
        // coordinates.
        let beta_k = tn.fit.block_states[0].beta[index];
        let delta = 1.0e-6 * beta_k.abs().max(1.0);
        let plus = perturbed_scores(index, delta);
        let minus = perturbed_scores(index, -delta);
        checked_entries += 1;

        let unsaturated = |i: usize| {
            base.z[i].abs() < SATURATION
                && plus[i].abs() < SATURATION
                && minus[i].abs() < SATURATION
        };
        let column_scale = (0..N)
            .filter(|&i| unsaturated(i))
            .map(|i| base.columns[[i, index]].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            column_scale > 0.0,
            "Jacobian column A[{k},{j}] is identically zero on every unsaturated row; the fixture \
             is not exercising this coefficient"
        );

        for i in 0..N {
            if !unsaturated(i) {
                continue;
            }
            let fd = (plus[i] - minus[i]) / (2.0 * delta);
            let analytic = base.columns[[i, index]];
            let absolute = (fd - analytic).abs();
            let rel = absolute / column_scale;
            if rel > worst_rel {
                worst_rel = rel;
                worst_abs = absolute;
                worst_label = format!(
                    "row {i}, A[{k},{j}] (flat {index}): analytic={analytic:.6e} fd={fd:.6e} \
                     column_scale={column_scale:.6e}"
                );
            }
            checked_rows += 1;
        }
    }

    eprintln!(
        "#2680 CTN Jacobian FD: p_resp={p_resp} p_cov={p_cov} probed {checked_entries} \
         coefficients over {checked_rows} unsaturated rows; worst error {worst_rel:.3e} of the \
         column scale ({worst_abs:.3e} absolute) at {worst_label}"
    );

    assert!(
        checked_rows > 4 * N,
        "too few unsaturated rows survived the FD gate ({checked_rows}); the fixture is not \
         exercising the Jacobian"
    );
    // What the bar has to separate: the pre-#2680 shape factor multiplies the
    // shape columns by `2·α_k`, which is an O(1) FRACTION of the column — tens
    // of percent to several hundred. What the instrument can resolve: a
    // magnitude-scaled central difference on this chain lands near `1e-8`
    // absolute (measured), which against these columns is `~1e-5` of their
    // scale. `1e-3` sits three orders below the defect and two above the
    // instrument, so it is insensitive to the step and cannot be crossed by
    // anything but a real disagreement.
    assert!(
        worst_rel < 1.0e-3,
        "score_influence_jacobian's Jacobian is not the derivative of its own score: \
         worst error {worst_rel:.3e} of the column scale at {worst_label}. A CTN chart is affine \
         in the coefficient matrix, so every sensitivity is the response-basis entry itself — a \
         shape factor here differentiates a function the value path does not compute."
    );
}

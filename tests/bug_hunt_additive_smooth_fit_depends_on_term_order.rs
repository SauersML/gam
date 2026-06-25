//! Bug hunt: a plain additive GAM's fit **depends on the order the smooth terms
//! are written**. Fitting `y ~ s(x) + s(z)` and `y ~ s(z) + s(x)` on the *same*
//! data must give the *same* fitted values — an additive model
//! `μ̂ = β₀ + f̂_x(x) + f̂_z(z)` is symmetric in its terms, REML selects the
//! smoothing parameters by optimizing a term-order-independent marginal
//! likelihood, and addition commutes. The fit is a model invariant and cannot
//! depend on which smooth the user typed first.
//!
//! It does. On a majority of seeds the two orderings land on different REML
//! optima: the predicted mean differs by **~4–10 % of the signal range** and the
//! total EDF by 2–4. The converged smoothing parameters are not a swapped pair —
//! the order-dependence is concentrated in the **double-penalty null-space
//! ridge**: for the data at the first seed below the optimizer returns, in
//! working order `[λ_range, λ_null]` per smooth,
//!
//!   s(x)+s(z): s(x) λ̂=[1.1e4, 3.0e9]   s(z) λ̂=[1.07e13, 0.49]
//!   s(z)+s(x): s(z) λ̂=[1.06e13, 0.19]  s(x) λ̂=[5.8e4, 2.5]
//!
//! i.e. the x-smooth's null-space (linear-component) ridge is `3.0e9` — shrinking
//! x's linear part to near-zero — in one order and `2.5` (essentially no
//! shrinkage) in the other. That single flip reshapes the x main effect and
//! moves the fit by ~6 %. On other seeds the two orders converge to the same
//! optimum (≈1e-8 of range), so an order-invariant additive REML is plainly
//! achievable here; the optimizer simply resolves the flat double-penalty REML
//! valley differently depending on the order the penalty blocks are presented.
//!
//! This is the additive-term-order sibling of the REML-invariance family (#1378
//! row permutation, #1214/#1215 rescale, #1269/#1375 translation, #1456
//! rotation), and is mechanistically adjacent to the double-penalty null-space
//! pathologies (#1266 EDF inflation, #1371 linear-trend annihilation): the
//! null-space ridge λ is the coordinate that lands order-dependently. `s(x)+s(z)`
//! is the single most common GAM form, so a reordering changing the answer is a
//! silent reproducibility hazard.
//!
//! The test fits both term orders, predicts both on the shared training grid via
//! the public `predict_gam` path, and asserts the surfaces agree to a tight
//! fraction of the signal range across a block of seeds. It is RED today (worst
//! ~10 %) and turns GREEN once additive REML is made term-order invariant.

use csv::StringRecord;
use gam_predict::predict_gam;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::{Array1, Array2};

/// Deterministic SplitMix64 — no Python, no external RNG crate.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Additive truth `f(x, z) = sin(3x) + 0.8 z² + cos(2z)` on a 2-D uniform
/// design, light Gaussian noise. Both main effects are genuinely curved (so
/// neither smooth is trivially linear) and they carry different roughness (so
/// an order-dependent smoothing selection has something to disagree about).
fn build(seed: u64) -> (gam::data::EncodedDataset, Vec<(f64, f64)>) {
    let n = 1000usize;
    let mut rng = SplitMix64::new(seed);
    let mut pts = Vec::with_capacity(n);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.unit();
        let z = rng.unit();
        let f = (3.0 * x).sin() + 0.8 * z * z + (2.0 * z).cos();
        let y = f + 0.3 * rng.normal();
        pts.push((x, z));
        rows.push(StringRecord::from(vec![
            x.to_string(),
            z.to_string(),
            y.to_string(),
        ]));
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode dataset"),
        pts,
    )
}

/// Fit `formula` (Gaussian identity) and return `(total_edf, lambdas, fitted
/// mean at every point in `pts`)` through the public predict path.
fn fit_and_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    pts: &[(f64, f64)],
) -> (f64, Vec<f64>, Vec<f64>) {
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) =
        fit_from_formula(formula, data, &cfg).expect("standard additive GAM fit")
    else {
        panic!("expected a standard GAM fit for {formula}");
    };
    let edf = fit.fit.edf_total().expect("fit reports total edf");
    let lambdas = fit.fit.lambdas.to_vec();

    let xi = data
        .headers
        .iter()
        .position(|h| h == "x")
        .expect("x column");
    let zi = data
        .headers
        .iter()
        .position(|h| h == "z")
        .expect("z column");
    let hlen = data.headers.len();
    let m = pts.len();
    let mut grid = Array2::<f64>::zeros((m, hlen));
    for (i, &(x, z)) in pts.iter().enumerate() {
        grid[[i, xi]] = x;
        grid[[i, zi]] = z;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild additive design at the prediction grid");
    let dense = design.design.to_dense();
    let family = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gam(dense, fit.fit.beta.view(), offset.view(), family)
        .expect("predict on the training grid");
    (edf, lambdas, pred.mean.to_vec())
}

#[test]
fn additive_smooth_fit_is_invariant_to_term_order() {
    // A correct order-invariant additive fit reproduces the surface to
    // numerical precision (the symmetric seeds reach ~1e-8 of range). 0.5 % of
    // the signal range is a generous round-off allowance; the observed 4–10 %
    // disagreement is more than an order of magnitude past it.
    const REL_TOL: f64 = 5.0e-3;

    let mut worst_rel = 0.0_f64;
    let mut worst_seed = 0_u64;
    for seed in [1_u64, 2, 6] {
        let (data, pts) = build(seed);
        let (edf_a, lam_a, pred_a) = fit_and_predict("y ~ s(x) + s(z)", &data, &pts);
        let (edf_b, lam_b, pred_b) = fit_and_predict("y ~ s(z) + s(x)", &data, &pts);

        let max_abs: f64 = pred_a
            .iter()
            .zip(&pred_b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let max = pred_a.iter().cloned().fold(f64::MIN, f64::max);
        let min = pred_a.iter().cloned().fold(f64::MAX, f64::min);
        let range = max - min;
        assert!(
            range > 0.5,
            "seed {seed}: degenerate s(x)+s(z) fit (signal range {range:.4}); the invariant \
             would be vacuous"
        );
        let rel = max_abs / range;
        if rel > worst_rel {
            worst_rel = rel;
            worst_seed = seed;
        }

        eprintln!(
            "[term-order] seed={seed} | s(x)+s(z): edf={edf_a:.3} λ̂={lam_a:?} | \
             s(z)+s(x): edf={edf_b:.3} λ̂={lam_b:?} | max|Δμ̂|/range={rel:.3e}"
        );
    }
    eprintln!(
        "[term-order] worst max|Δμ̂|/range across seeds = {worst_rel:.3e} (seed {worst_seed})"
    );

    assert!(
        worst_rel < REL_TOL,
        "y ~ s(x)+s(z) and y ~ s(z)+s(x) fit DIFFERENT surfaces: worst max|Δμ̂| across seeds is \
         {worst_rel:.3e} of the signal range (seed {worst_seed}, tol {REL_TOL:.0e}). An additive \
         model is symmetric in its terms, so the fitted surface must be invariant to the order \
         the smooths are written. It is not — REML resolves the flat double-penalty valley \
         differently per term order, selecting a non-swapped λ̂ (especially the null-space \
         ridge), reshaping the shipped fit."
    );
}

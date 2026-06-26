//! Bug hunt: a `te(...)` tensor-product smooth is **not invariant to the order
//! of its margins**. Fitting `y ~ te(x, z)` and `y ~ te(z, x)` on the *same*
//! data must produce the *same* fitted surface — the two formulas span the
//! identical tensor-product B-spline space (the Kronecker factors are merely
//! relabelled) under the identical per-margin penalty family, and REML selects
//! one smoothing parameter per margin, so the marginal-likelihood objective is
//! symmetric under the swap. The fitted values as a function of `(x, z)` are an
//! invariant of the model and cannot depend on which covariate the user typed
//! first.
//!
//! They DO depend on it. On asymmetric data, `te(x, z)` and `te(z, x)` land on
//! materially different fits: the predicted surface differs by ~2–6 % of the
//! signal range and the total EDF by 0.5–5 across seeds. The cause is the REML
//! smoothing-parameter selection, not the design or the penalty (which are
//! symmetric by construction): the converged `λ̂` are not a swapped pair. For
//! the data built below at the first seed the optimizer returns, in working
//! order,
//!
//!   te(x, z): λ̂ ≈ [8.8e1, 1.07e13]      (margin x wiggly, margin z railed linear)
//!   te(z, x): λ̂ ≈ [1.07e13, 3.2e2]      (margin z railed linear, margin x ≈ 3.2e2)
//!
//! i.e. the x-margin smoothing parameter is 88 in one order and 325 in the
//! other — not a relabelling, a different optimum. Across every seed the
//! second-listed-then-railed structure flips with the typed order, so the model
//! the user ships changes with a cosmetic edit. (The `ti(...)` pure-interaction
//! form, which centres each margin separately, is invariant to the swap to
//! ~1e-5 of range — so an order-independent tensor fit is clearly achievable in
//! this engine; `te` regresses from it.)
//!
//! This is the tensor-margin-order sibling of the REML-invariance family
//! (#1378 row permutation, #1214/#1215 covariate rescale, #1269/#1375 covariate
//! translation, #1456 rotation): an irrelevant relabelling of the inputs moves
//! `λ̂` and reshapes the shipped fit.
//!
//! The test fits both margin orders, predicts both on the shared training grid
//! through the public `predict_gam` path, and asserts the surfaces agree to a
//! tight fraction of the signal range. It is RED today (≈2–6 % disagreement)
//! and turns GREEN once the tensor REML fit is made margin-order invariant.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam_predict::predict_gam;
use ndarray::{Array1, Array2};

/// Deterministic SplitMix64 → no Python, no external RNG crate.
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
    /// Uniform on (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller (one of the pair).
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// An asymmetric true surface `f(x, z) = sin(3x) + 0.8 z² + 1.5 x z + cos(2z)`
/// plus light Gaussian noise on a 2-D uniform design. Asymmetry in `x` vs `z`
/// is the point: it gives each margin a different roughness so that an
/// order-dependent smoothing selection has something to disagree about.
fn build(seed: u64) -> (gam::data::EncodedDataset, Vec<(f64, f64)>) {
    let n = 1200usize;
    let mut rng = SplitMix64::new(seed);
    let mut pts = Vec::with_capacity(n);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.unit();
        let z = rng.unit();
        let f = (3.0 * x).sin() + 0.8 * z * z + 1.5 * x * z + (2.0 * z).cos();
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
/// mean at every point in `pts`)`. The fitted mean is produced through the
/// public design + `predict_gam` path so it is exactly what a user sees.
fn fit_and_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    pts: &[(f64, f64)],
) -> (f64, Vec<f64>, Vec<f64>) {
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) =
        fit_from_formula(formula, data, &cfg).expect("standard tensor GAM fit")
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
        .expect("rebuild tensor design at the prediction grid");
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
fn te_tensor_fit_is_invariant_to_margin_order() {
    // A correct order-invariant tensor fit reproduces the surface to numerical
    // precision (the `ti` sibling reaches ~1e-5 of range on this very data). We
    // accept up to 0.5 % of the signal range as honest REML/linear-algebra
    // round-off; the observed 2–6 % disagreement is an order of magnitude past
    // that, so the bar is clearly RED today and comfortably GREEN once fixed.
    const REL_TOL: f64 = 5.0e-3;

    let mut worst_rel = 0.0_f64;
    let mut worst_seed = 0_u64;
    for seed in [1_u64, 3, 5] {
        let (data, pts) = build(seed);
        let (edf_xz, lam_xz, pred_xz) = fit_and_predict("y ~ te(x, z, k=[8,8])", &data, &pts);
        let (edf_zx, lam_zx, pred_zx) = fit_and_predict("y ~ te(z, x, k=[8,8])", &data, &pts);

        let max_abs: f64 = pred_xz
            .iter()
            .zip(&pred_zx)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let max = pred_xz.iter().cloned().fold(f64::MIN, f64::max);
        let min = pred_xz.iter().cloned().fold(f64::MAX, f64::min);
        let range = max - min;
        assert!(
            range > 0.5,
            "seed {seed}: degenerate te(x,z) fit (signal range {range:.4}); the invariant \
             would be vacuous"
        );
        let rel = max_abs / range;
        if rel > worst_rel {
            worst_rel = rel;
            worst_seed = seed;
        }

        eprintln!(
            "[te-order] seed={seed} | te(x,z): edf={edf_xz:.3} λ̂={lam_xz:?} | \
             te(z,x): edf={edf_zx:.3} λ̂={lam_zx:?} | max|Δμ̂|/range={rel:.3e}"
        );
    }
    eprintln!("[te-order] worst max|Δμ̂|/range across seeds = {worst_rel:.3e} (seed {worst_seed})");

    // A single order-dependent seed is enough to prove the engine ships a
    // margin-order-dependent fit, so the invariant is the worst case over the
    // block: EVERY seed must reproduce the surface across the swap. Today the
    // worst is ~2–6 % of range — many times `REL_TOL` — so this is robustly RED;
    // an order-invariant tensor REML (as `ti` already achieves to ~1e-5) drives
    // every seed well under the bar.
    assert!(
        worst_rel < REL_TOL,
        "te(x,z) and te(z,x) fit DIFFERENT surfaces: worst max|Δμ̂| across seeds is \
         {worst_rel:.3e} of the signal range (seed {worst_seed}, tol {REL_TOL:.0e}). The two \
         formulas span the identical tensor-product space under the identical per-margin \
         penalty family, so the fitted surface must be invariant to the typed margin order. \
         It is not — REML converges to a non-swapped λ̂ pair, reshaping the shipped fit."
    );
}

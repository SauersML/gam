//! #1593 gauge-invariance guard — tensor/ANOVA decomposition `ti(x, z)` is
//! invariant to the arbitrary margin order (which covariate is typed first /
//! which margin is the "main-effect anchor").
//!
//! The issue audits every family/parametrization whose penalized fit could
//! silently depend on an ARBITRARY reference frame that the likelihood is blind
//! to (multinomial reference class #1587, simplex ALR reference #1549, tensor
//! `te` margin order — RED, see
//! `bug_hunt_te_tensor_smooth_not_invariant_to_margin_order`). The ANOVA tensor
//! `ti(x, z)` is the pure-interaction sibling: each margin is centred to its own
//! sum-to-zero constraint BEFORE the Kronecker product, so neither margin is the
//! "main effect anchor" — the construction is symmetric in the two margins by
//! design. REML selects one smoothing parameter per margin, and the
//! per-margin-centred penalty sum is symmetric under the swap, so the fitted
//! interaction surface is a property of the data, not of which covariate the
//! user typed first.
//!
//! Unlike `te` (which lets the constant + both main effects live inside the
//! tensor and selects a non-swapped λ̂ pair, drifting 2–6 % of range under the
//! swap — that is the #1593-class bug guarded RED elsewhere), `ti` SHOULD be
//! margin-order invariant to numerical precision: the `te` bug hunt records the
//! `ti` sibling reproducing the surface to ~1e-5 of range across the swap. This
//! test LOCKS THAT IN as a green guard.
//!
//! It fits the full ANOVA decomposition `y ~ s(x) + s(z) + ti(x, z)` and its
//! margin-swapped relabelling `y ~ s(z) + s(x) + ti(z, x)` on the SAME
//! asymmetric data, predicts both fits on the shared training grid through the
//! public design + `predict_gam` path (exactly what a user sees), and asserts
//! the two surfaces agree to a tight fraction of the signal range — while a
//! refit of one order is confirmed deterministic, so any cross-order drift would
//! be a real frame dependence and not refit noise.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam_predict::predict_gam;
use ndarray::{Array1, Array2};

/// Deterministic SplitMix64 → byte-identical data run-to-run (no external RNG),
/// so any cross-order disagreement is a fit property, not sampling noise.
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

/// An asymmetric true surface with genuine `x·z` interaction plus distinct
/// per-margin main effects, so the ANOVA split is non-trivial and each margin
/// has a different roughness — a margin-order-dependent λ̂ selection would have
/// something to disagree about (this is the same fixture shape the `te` bug-hunt
/// uses, where `te` drifts but `ti` does not).
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

/// Fit `formula` (Gaussian identity) and return the fitted mean at every point
/// in `pts`, produced through the public design + `predict_gam` path so it is
/// exactly the surface a user sees. Resolving the prediction columns by NAME
/// (not position) makes the swapped formula's prediction land on the same
/// physical `(x, z)` points.
fn fit_and_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    pts: &[(f64, f64)],
) -> Vec<f64> {
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) =
        fit_from_formula(formula, data, &cfg).expect("standard ANOVA tensor GAM fit")
    else {
        panic!("expected a standard GAM fit for {formula}");
    };

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
        .expect("rebuild ANOVA tensor design at the prediction grid");
    let dense = design.design.to_dense();
    let family = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gam(dense, fit.fit.beta.view(), offset.view(), family)
        .expect("predict on the training grid");
    pred.mean.to_vec()
}

#[test]
fn ti_anova_tensor_fit_is_invariant_to_margin_order_1593() {
    // The `ti` ANOVA tensor centres each margin separately, so the swap is a
    // pure relabelling of the Kronecker factors under an identical per-margin
    // penalty family; the fitted surface must reproduce to numerical precision.
    // The `te` bug-hunt records the `ti` sibling reaching ~1e-5 of range on this
    // very fixture, so we hold a tight 5e-4 of the signal range — an order of
    // magnitude below the 2–6 % at which the order-DEPENDENT `te` fit drifts, so
    // this is a real guard (not a vacuous bar) and a regression in margin-order
    // symmetry would trip it.
    const REL_TOL: f64 = 5.0e-4;

    let mut worst_rel = 0.0_f64;
    let mut worst_seed = 0_u64;
    for seed in [1_u64, 3, 5] {
        let (data, pts) = build(seed);

        // ANOVA decomposition and its margin-swapped relabelling. Swapping the
        // additive main-effect order AND the `ti` margin order is the full
        // "which margin is the anchor" reparameterization.
        let pred_xz = fit_and_predict("y ~ s(x) + s(z) + ti(x, z, k=[6,6])", &data, &pts);
        let pred_zx = fit_and_predict("y ~ s(z) + s(x) + ti(z, x, k=[6,6])", &data, &pts);

        // A refit of the SAME order must be deterministic, else cross-order
        // drift could not be attributed to the frame choice.
        let pred_xz_again = fit_and_predict("y ~ s(x) + s(z) + ti(x, z, k=[6,6])", &data, &pts);
        let refit_noise: f64 = pred_xz
            .iter()
            .zip(&pred_xz_again)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            refit_noise < 1e-9,
            "seed {seed}: same-order refit is non-deterministic (max|Δμ̂|={refit_noise:.3e}); \
             cannot attribute cross-order drift to the margin frame"
        );

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
            "seed {seed}: degenerate ti(x,z) fit (signal range {range:.4}); the invariant \
             would be vacuous"
        );
        let rel = max_abs / range;
        if rel > worst_rel {
            worst_rel = rel;
            worst_seed = seed;
        }

        eprintln!(
            "[ti-order] seed={seed} | max|Δμ̂|/range={rel:.3e} (refit noise {refit_noise:.3e}, \
             signal range {range:.3})"
        );
    }
    eprintln!("[ti-order] worst max|Δμ̂|/range across seeds = {worst_rel:.3e} (seed {worst_seed})");

    assert!(
        worst_rel < REL_TOL,
        "ti(x,z) and ti(z,x) (with swapped additive main-effect order) fit DIFFERENT surfaces: \
         worst max|Δμ̂| across seeds is {worst_rel:.3e} of the signal range (seed {worst_seed}, \
         tol {REL_TOL:.0e}). The ANOVA tensor centres each margin separately, so the margin \
         order is a pure reparameterization and the fitted interaction surface must be invariant \
         to it (#1593 gauge-invariance class). A drift here is a real margin-anchor dependence \
         of the #1549/#1587 family."
    );
}

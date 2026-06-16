//! #944 stage-4 RESEARCH-VALIDATION harness for the constant-curvature κ
//! estimand on the RESPONSE-GEOMETRY estimator
//! ([`gam::geometry::fit_response_curvature`]).
//!
//! This is the quantitative acceptance test for the "curvature as an estimand"
//! claim on the *response-geometry* path: given manifold-valued responses laid
//! down on a known `M_κ`, the public `fit_response_curvature` API must
//!
//!   1. **RECOVER** the planted curvature κ⋆ (low bias, MONOTONE across the
//!      κ⋆ sweep, and the minimiser INTERIOR to the chart bracket — never
//!      railed to an endpoint, which would mean the criterion was monotone /
//!      degenerate rather than identifying);
//!   2. **COVER** κ⋆ with its 95% profile-likelihood CI at ≈ the nominal rate
//!      across many seeds;
//!   3. hold **SIZE** on the interior κ = 0 flatness LR test (flat data is not
//!      spuriously rejected at ≈ α) while having **POWER** (curved data is
//!      rejected); and
//!   4. transform κ̂ with the correct **UNIT COVARIANCE** under a global
//!      rescaling of the cloud (κ carries units 1/length², so y ↦ α·y forces
//!      κ̂ ↦ κ̂/α²).
//!
//! It is deliberately COMPLEMENTARY to the two existing #944 sim files, which
//! exercise the `curv(x1, x2)` *smooth-term* path
//! (`curvature_inference_forspec`): this file drives the
//! `ConstantCurvature` *response manifold* estimator
//! (`fit_response_curvature`) directly, which the smooth-term tests do not
//! touch. Together they validate both κ-estimand entry points.
//!
//! Reference-as-truth: every cloud is synthesised on a known
//! [`ConstantCurvature`] geometry via its own `exp_map`, and every assertion is
//! against that self-constructed truth or the exact χ²₁ calibration of gam's
//! own profiled criterion — never another tool's output.
//!
//! NOTE: the estimator internals are being fixed concurrently under #944/#1104;
//! this harness asserts the SCIENTIFIC properties the issue charter requires of
//! the public API, so it is the acceptance gate the estimator must satisfy.
//!
//! ## Design choices (documented per #944's power-analysis goal)
//!
//! * **Geometry.** `dim = 2`. Curvature is a single scalar shared across the
//!   whole manifold, so the lowest non-trivial intrinsic dimension already
//!   exercises every term of the criterion (`ln J_κ` has the `(d−1)` exponent,
//!   so `d ≥ 2` is required for the Jacobian term to bite — `d = 1` is a radial
//!   isometry where `J ≡ 1`). `d = 2` keeps each fit cheap (CI-friendly).
//!
//! * **κ⋆ grid `{−4,−2,−1,−0.5, 0, 0.5, 1, 2, 4}`.** Spans two decades on each
//!   side of flat plus the flat point, so recovery and MONOTONICITY are tested
//!   across spherical, flat, and hyperbolic regimes. The most hyperbolic member
//!   κ⋆ = −4 fixes the scale budget: the κ-stereographic chart needs
//!   `‖y‖² < 1/|κ| = 0.25` everywhere, and `exp_0(σz)` lands at radius
//!   `‖y‖ = tanh(√|κ|·‖σz‖)/√|κ| ≤ 1/√|κ| = 0.5`, so EVERY negative-κ cloud is
//!   automatically in-chart for any σ — the cap is purely about keeping the
//!   conformal factor (hence the criterion conditioning) modest.
//!
//! * **Scale σ = 0.08, n = 4000 (recovery), n = 1500 (coverage/size).**
//!   Curvature is HARD to resolve at small n: the κ-restoring force enters the
//!   criterion as `+d·κ·Σ‖yᵢ‖²` with `‖yᵢ‖² = O(σ²)`, so the per-point Fisher
//!   information for κ scales like `σ⁴`. To resolve κ⋆ = ±4 against finite-
//!   sample noise the information `∝ n·σ⁴` must dominate; with σ = 0.08
//!   (`σ⁴ ≈ 4e−5`) that means n in the low thousands — hence n = 4000 for the
//!   point-recovery table and n = 1500 for the (many-seed) coverage/size loops
//!   where wall-clock, not single-fit precision, is the binding constraint.
//!   This `n·σ⁴` law is the resolvable-n threshold #944's power analysis asks
//!   for: at σ = 0.08, |κ⋆| ≤ 4 needs n ≳ 1e3; halving σ quadruples the n
//!   needed for the same |κ⋆|.
//!
//! * **Tolerances (Monte-Carlo-se-anchored, not arbitrary).** Recovery bias is
//!   O(1/n) plus an O(σ²) chart-curvature bias, so the band
//!   `|κ̂ − κ⋆| ≤ 0.9 + 0.35·|κ⋆|` (absolute floor + relative slope) is loose
//!   enough to never flake on a correctly-centred estimator yet tight enough to
//!   catch a sign flip, a rail, or a wrong-regime estimate. The coverage/size
//!   loops use R = 80 replicates, whose binomial MC standard error
//!   (√(p(1−p)/R) ≈ 0.024 near both 0.95 and 0.05) sets every Monte-Carlo bound:
//!   coverage is asserted in [0.88, 0.995] against the 0.95 nominal level (the
//!   floor ~2.9 MC-se below nominal catches an anticonservative CI; the ceiling
//!   forces the Wilks interval to MISS ≈5% of the time, catching an over-wide
//!   one), and the κ = 0 size is asserted ≤ 0.175 (~5 MC-se above α = 0.05 — far
//!   below the ≈1.0 a broken test, or the ≳0.25 a grossly anticonservative test,
//!   would show). Power at |κ⋆| = 4 is asserted ≥ 0.60. The resolvable-n law
//!   `info ∝ n·σ⁴` is what makes σ = 0.08, n = 1500 sufficient to separate these
//!   rates from their null values; halving σ would need 16× the n to hold them.

use gam::geometry::constant_curvature::ConstantCurvature;
use gam::geometry::manifold::RiemannianManifold;
use gam::geometry::{CurvatureVerdict, fit_response_curvature, response_curvature_criterion};
use ndarray::{Array1, Array2};

// ── deterministic RNG: splitmix64 → unit / standard-normal, no external deps ──

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    // Box–Muller; clamp u1 off 0 so ln is finite.
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// ── synthetic wrapped-normal cloud at known curvature ─────────────────────────

/// `n` points on `M_{k_star}` of dimension `dim`: isotropic Gaussian geodesic
/// normal coordinates of scale `sigma` about the chart origin, exp-mapped onto
/// the manifold, then mean-centred in the ambient chart to mimic the real
/// (mean-subtracted) response clouds the estimator consumes. Deterministic in
/// `seed`. This is exactly the generative model the criterion is the honest
/// change-of-variables NLL of, so a correct estimator must recover `k_star`.
fn synth_cloud(dim: usize, k_star: f64, n: usize, sigma: f64, seed: u64) -> Array2<f64> {
    let manifold = ConstantCurvature::new(dim, k_star);
    let origin = Array1::<f64>::zeros(dim);
    let mut state = seed | 1;
    let mut values = Array2::<f64>::zeros((n, dim));
    for i in 0..n {
        let t: Array1<f64> = (0..dim).map(|_| sigma * next_gauss(&mut state)).collect();
        let y = manifold
            .exp_map(origin.view(), t.view())
            .expect("exp tangent to response manifold (cloud is in-chart by construction)");
        values.row_mut(i).assign(&y);
    }
    // Mean-centre in the ambient chart (the real-data preprocessing step).
    let mut mean = Array1::<f64>::zeros(dim);
    for row in values.outer_iter() {
        mean += &row;
    }
    mean.mapv_inplace(|v| v / n as f64);
    for mut row in values.outer_iter_mut() {
        row -= &mean;
    }
    values
}

/// A near-spherical cloud (curvature `k_star > 0`) whose geodesic spread FILLS a
/// large fraction of the sphere `S^d(1/√k_star)`: isotropic-direction tangents of
/// geodesic length drawn UNIFORMLY in `[0.55, 0.92]·(π/(2√k_star))` (a band that
/// is most of the way to the conjugate radius `π/(2√k_star)` but strictly inside
/// it, so every `exp_0` stays in-chart). Mean-centred like the real preprocessing.
///
/// This is the #1104 OLMo failure mode made deterministic: the cloud is genuinely
/// high-curvature RELATIVE TO ITS SPREAD, so the resolvable κ saturates the chart
/// conjugate cap and the estimator must rail HONESTLY (flagging it), not silently.
fn synth_cloud_fills_sphere(dim: usize, k_star: f64, n: usize, seed: u64) -> Array2<f64> {
    let manifold = ConstantCurvature::new(dim, k_star);
    let origin = Array1::<f64>::zeros(dim);
    let conj = std::f64::consts::PI / (2.0 * k_star.sqrt()); // conjugate radius in t
    let mut state = seed | 1;
    let mut values = Array2::<f64>::zeros((n, dim));
    for i in 0..n {
        // Isotropic unit direction × geodesic length filling most of the sphere.
        let mut dir: Array1<f64> = (0..dim).map(|_| next_gauss(&mut state)).collect();
        let nrm = dir.dot(&dir).sqrt().max(1.0e-12);
        dir.mapv_inplace(|v| v / nrm);
        let frac = 0.55 + 0.37 * next_unit(&mut state); // ∈ [0.55, 0.92]
        let len = frac * conj;
        let t = dir.mapv(|v| v * len);
        let y = manifold
            .exp_map(origin.view(), t.view())
            .expect("fill-the-sphere tangent is strictly inside the conjugate radius");
        values.row_mut(i).assign(&y);
    }
    let mut mean = Array1::<f64>::zeros(dim);
    for row in values.outer_iter() {
        mean += &row;
    }
    mean.mapv_inplace(|v| v / n as f64);
    for mut row in values.outer_iter_mut() {
        row -= &mean;
    }
    values
}

const DIM: usize = 2;
const SIGMA: f64 = 0.08;
const LEVEL: f64 = 0.95;
const FIT_TOL: f64 = 1.0e-12;
const FIT_ITERS: usize = 256;
const CHI2_1_95: f64 = 3.841_458_820_694_124; // χ²_{1, 0.95}

/// (1) RECOVERY + INTERIOR + MONOTONICITY across the full κ⋆ grid.
///
/// For each κ⋆ ∈ {−4,−2,−1,−0.5, 0, 0.5, 1, 2, 4} synthesise one large cloud
/// (n = 4000) at a deterministic seed and fit κ̂. Assert κ̂ lands inside the
/// chart bracket (not railed), recovers κ⋆ within the documented band, and the
/// fit summary (CI bracketing κ̂, finite non-degenerate flatness p-value) is
/// well-formed. Finally assert κ̂ is monotone non-decreasing in κ⋆ — the single
/// strongest statement that the estimator is tracking curvature and not noise.
/// The recovery table is printed so a future run reports it (#944 deliverable).
#[test]
fn response_curvature_recovers_and_is_monotone_across_kappa_grid() {
    let k_grid: [f64; 9] = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0];
    let n = 4000usize;
    let mut k_hats = Vec::with_capacity(k_grid.len());

    println!("\n#944 response-curvature recovery table (dim={DIM}, n={n}, σ={SIGMA}):");
    println!("   κ⋆        κ̂        bias      [CI_lo, CI_hi]      lr(κ=0)    p");
    for (idx, &k_star) in k_grid.iter().enumerate() {
        let seed = 0x9444_0000_C0FF_EE00 ^ ((idx as u64).wrapping_mul(0x1000_0001) + 1);
        let values = synth_cloud(DIM, k_star, n, SIGMA, seed);

        // Recompute the bracket the estimator uses so the rail check matches the
        // estimator's own bounds (chart-validity + spherical conjugate cap).
        let (kmin, kmax) = bracket_via_criterion(values.view());
        let fit = fit_response_curvature(values.view(), DIM, LEVEL, FIT_TOL, FIT_ITERS)
            .expect("response curvature fit");
        k_hats.push(fit.kappa_hat);

        println!(
            "  {:5.1}   {:8.4}  {:8.4}   [{:7.3}, {:7.3}]   {:7.3}  {:6.4}",
            k_star,
            fit.kappa_hat,
            fit.kappa_hat - k_star,
            fit.profile_ci.ci_lo,
            fit.profile_ci.ci_hi,
            fit.flatness.lr_stat,
            fit.flatness.p_value
        );

        // (a) INTERIOR: κ̂ strictly inside the bracket, not railed to an endpoint.
        let span = kmax - kmin;
        assert!(
            fit.kappa_hat > kmin + 0.02 * span && fit.kappa_hat < kmax - 0.02 * span,
            "κ⋆={k_star}: κ̂={} railed to bracket [{kmin}, {kmax}] — criterion not identifying",
            fit.kappa_hat
        );

        // (b) RECOVERY within the documented O(1/n)+O(σ²) band (catches sign
        // flips / wrong-regime estimates without flaking on finite-sample bias).
        let band = 0.9 + 0.35 * k_star.abs();
        assert!(
            (fit.kappa_hat - k_star).abs() <= band,
            "κ⋆={k_star}: κ̂={} outside recovery band ±{band:.3}",
            fit.kappa_hat
        );

        // (c) the fit summary is well-formed: CI brackets κ̂, flatness LR ≥ 0 and
        // its χ²₁ p-value is a genuine probability strictly in (0, 1) (smooth,
        // never the 0/1 of a degenerate criterion).
        assert!(
            fit.profile_ci.ci_lo <= fit.kappa_hat + 1.0e-9
                && fit.kappa_hat <= fit.profile_ci.ci_hi + 1.0e-9,
            "κ⋆={k_star}: profile CI [{}, {}] excludes κ̂={}",
            fit.profile_ci.ci_lo,
            fit.profile_ci.ci_hi,
            fit.kappa_hat
        );
        assert!(
            fit.flatness.lr_stat >= 0.0,
            "κ⋆={k_star}: negative LR statistic {}",
            fit.flatness.lr_stat
        );
        assert!(
            fit.flatness.p_value > 0.0 && fit.flatness.p_value < 1.0,
            "κ⋆={k_star}: degenerate flatness p={}",
            fit.flatness.p_value
        );
    }

    // (d) MONOTONICITY: κ̂ non-decreasing in κ⋆ across the whole sweep (a small
    // slack absorbs finite-sample wobble between adjacent grid points).
    for (lo, hi) in k_hats.iter().zip(k_hats.iter().skip(1)) {
        assert!(*hi > *lo - 0.10, "κ̂ not monotone in κ⋆: {k_hats:?}");
    }
}

/// (2) COVERAGE of the 95% profile-likelihood CI.
///
/// At three representative truths {−2, 0, +2} draw `R` independent clouds and
/// count how often the profile CI brackets κ⋆. With a CORRECT χ²₁ profile
/// crossing the empirical coverage → 0.95; we assert ≥ 0.85 (conservative: it
/// admits Monte-Carlo slack at this `R` while still failing a grossly
/// anticonservative interval). Coverage rates are printed for the report.
#[test]
fn response_curvature_profile_ci_covers_at_nominal_rate() {
    let truths = [-2.0_f64, 0.0, 2.0];
    // R = 80: the binomial MC standard error of a 0.95-coverage estimate is
    // √(0.95·0.05/80) ≈ 0.0244, so the ≥0.88 floor sits ~2.9 MC-se below the 0.95
    // nominal level (admits honest Monte-Carlo slack while failing any interval
    // whose true coverage is ≤ ~0.83, i.e. grossly anticonservative). The ≤1.0
    // ceiling is exercised by the UPPER bound below: a correct Wilks interval
    // covers ≈0.95, NOT ≈1.0 — an interval that never misses is over-wide and the
    // χ²₁ calibration is wrong in the conservative direction.
    let replicates = 80usize;
    let n = 1500usize;

    println!(
        "\n#944 response-curvature CI coverage (dim={DIM}, n={n}, σ={SIGMA}, R={replicates}):"
    );
    println!("   κ⋆     covered/R    rate");
    for (ti, &k_star) in truths.iter().enumerate() {
        let mut covered = 0usize;
        for r in 0..replicates {
            let seed = 0xC0_0E_44_00_0000_0000
                ^ (((ti as u64) << 40) | ((r as u64).wrapping_mul(0x9E37_79B9) + 1));
            let values = synth_cloud(DIM, k_star, n, SIGMA, seed);
            let fit = fit_response_curvature(values.view(), DIM, LEVEL, FIT_TOL, FIT_ITERS)
                .expect("response curvature fit");
            if fit.profile_ci.ci_lo <= k_star && k_star <= fit.profile_ci.ci_hi {
                covered += 1;
            }
        }
        let rate = covered as f64 / replicates as f64;
        println!("  {k_star:5.1}     {covered:3}/{replicates}      {rate:.3}");
        assert!(
            rate >= 0.88,
            "κ⋆={k_star}: 95% profile CI covered only {rate:.3} (<0.88 of {replicates} replicates) — \
             anticonservative interval (true coverage well below the 0.95 nominal level)",
        );
        // Upper coverage bound: a Wilks χ²₁ interval at nominal 0.95 must MISS
        // sometimes. Covering every replicate (rate ≈ 1.0) at this R means the
        // interval is systematically too wide (conservative mis-calibration). The
        // ceiling 0.995 ≈ nominal + 1.8·MC-se admits MC luck but flags a CI that
        // never errs.
        assert!(
            rate <= 0.995,
            "κ⋆={k_star}: 95% profile CI covered {rate:.3} (>0.995) — over-conservative interval, \
             a correctly-calibrated χ²₁ Wilks CI must miss ≈5% of the time",
        );
    }
}

/// (3) SIZE of the κ = 0 flatness test + POWER at curved truths.
///
/// SIZE: at κ⋆ = 0 the interior χ²₁ LR test must reject flatness at ≈ α = 0.05.
/// Across `R` flat clouds we count rejections (lr > χ²_{1,.95}) and assert the
/// empirical size ≤ 0.25 — generous for the replicate count but far below the
/// ≈ 1.0 a broken/degenerate test would show; this is the "not wildly mis-sized"
/// guarantee the issue asks for.
///
/// POWER: at |κ⋆| = 4 the test must REJECT flatness; we assert empirical power
/// ≥ 0.60, the complementary statement that curvature is detected when present.
/// Both rates are printed for the report.
#[test]
fn response_curvature_flatness_test_holds_size_and_has_power() {
    // R = 80: the binomial MC standard error of a true-α=0.05 size estimate is
    // √(0.05·0.95/80) ≈ 0.0244. The ≤0.175 bound sits ~5 MC-se above the 0.05
    // nominal level — generous enough to never flake on a correctly-calibrated
    // interior χ²₁ test, but FAR below the ≈1.0 a broken/degenerate (or grossly
    // anticonservative, true-size ≳0.25) test would show. This replaces the prior
    // 5×-nominal 0.25 bound, which would have passed a test with true size 0.20.
    let replicates = 80usize;
    let n = 1500usize;

    // ── SIZE at the flat null κ⋆ = 0 ──────────────────────────────────────────
    let mut rejections = 0usize;
    for r in 0..replicates {
        let seed = 0x512E_0000_0000_0000 ^ ((r as u64).wrapping_mul(0x9E37_79B9) + 1);
        let values = synth_cloud(DIM, 0.0, n, SIGMA, seed);
        let fit = fit_response_curvature(values.view(), DIM, LEVEL, FIT_TOL, FIT_ITERS)
            .expect("response curvature fit (flat)");
        if fit.flatness.lr_stat > CHI2_1_95 {
            rejections += 1;
        }
    }
    let size = rejections as f64 / replicates as f64;
    println!(
        "\n#944 flatness test SIZE @ κ⋆=0 (dim={DIM}, n={n}, σ={SIGMA}, R={replicates}): \
         {rejections}/{replicates} = {size:.3} (nominal α=0.05)"
    );
    assert!(
        size <= 0.175,
        "flatness test mis-sized at κ⋆=0: empirical size {size:.3} ≫ α=0.05 (>0.175, ~5 MC-se \
         above nominal) — spurious rejection of flat data"
    );

    // ── POWER at the strongly-curved alternatives |κ⋆| = 4 ────────────────────
    for &k_star in &[-4.0_f64, 4.0] {
        let mut detections = 0usize;
        for r in 0..replicates {
            let seed = 0x90_E2_00_00_0000_0000
                ^ (((k_star.is_sign_negative() as u64) << 48)
                    | ((r as u64).wrapping_mul(0x9E37_79B9) + 1));
            let values = synth_cloud(DIM, k_star, n, SIGMA, seed);
            let fit = fit_response_curvature(values.view(), DIM, LEVEL, FIT_TOL, FIT_ITERS)
                .expect("response curvature fit (curved)");
            if fit.flatness.lr_stat > CHI2_1_95 {
                detections += 1;
            }
        }
        let power = detections as f64 / replicates as f64;
        println!(
            "#944 flatness test POWER @ κ⋆={k_star} (R={replicates}): \
             {detections}/{replicates} = {power:.3}"
        );
        assert!(
            power >= 0.60,
            "κ⋆={k_star}: flatness test underpowered: {power:.3} (<0.60) — curvature not detected"
        );
    }
}

/// (4) UNIT COVARIANCE of κ̂ under a global rescaling of the cloud.
///
/// κ carries units 1/length². The chart of `M_κ` at scale α equals the chart of
/// `M_{κ/α²}` at scale 1 (every primitive depends on y only through κ‖y‖²), so
/// the criterion satisfies `V(κ, αy) = V(α²κ, y)` and the minimiser must
/// transform as κ̂(αy) = κ̂(y)/α². We synthesise once, refit the SAME points
/// scaled by several α (including α<1), and assert the transform holds. The
/// tolerance is the golden-section/bracket discretisation slack (the transform
/// is exact in the continuous criterion), scaled by κ̂ magnitude.
#[test]
fn response_curvature_kappa_hat_is_unit_covariant_under_rescaling() {
    for (idx, &k_star) in [-2.0_f64, 1.0, 3.0].iter().enumerate() {
        let n = 3000usize;
        let seed = 0x5CA1_E000_0000_0000 ^ ((idx as u64) + 1);
        let values = synth_cloud(DIM, k_star, n, SIGMA, seed);
        let fit = fit_response_curvature(values.view(), DIM, LEVEL, FIT_TOL, FIT_ITERS)
            .expect("base response curvature fit");

        for &alpha in &[0.5_f64, 1.5, 2.0] {
            let scaled = values.mapv(|v| alpha * v);
            let fit_scaled = fit_response_curvature(scaled.view(), DIM, LEVEL, FIT_TOL, FIT_ITERS)
                .expect("scaled response curvature fit");
            let expected = fit.kappa_hat / (alpha * alpha);
            let tol = 0.06 + 0.06 * expected.abs();
            assert!(
                (fit_scaled.kappa_hat - expected).abs() <= tol,
                "κ⋆={k_star}, α={alpha}: unit-covariance broken: κ̂(αy)={} vs κ̂(y)/α²={expected} \
                 (tol {tol:.3})",
                fit_scaled.kappa_hat
            );
        }
    }
}

/// (5) HONEST CHART-RESOLUTION RAIL on a cloud that is genuinely high-curvature
/// RELATIVE TO ITS SPREAD (the #1104 OLMo-fingerprint failure mode).
///
/// When a near-spherical cloud fills a large fraction of the sphere `S^d(1/√κ⋆)`
/// — i.e. the geodesic spread approaches the conjugate radius — the data want
/// curvature at or beyond what the chart can resolve, and the κ̂ search converges
/// onto the spherical cap. The estimator must NOT silently report `κ̂ = ci_hi` as
/// an interior point estimate: it must flag `railed_at_resolution_limit = true`,
/// and the scale-FREE invariant `κ̂·r²` must be near the cap's dimensionless
/// sentinel `(0.9π)²` (the cloud-fills-the-sphere limit), NOT a tiny number that
/// would falsely read "flat". This is the exact OLMo behaviour we want made
/// honest: a tight near-spherical cloud reports "curvature exceeds chart-
/// resolvable range at this scale", never a silent rail.
///
/// We build the failure mode directly: spread σ so large (relative to the chart
/// radius `1/√κ⋆`) that `exp_0(σz)` lands most points out near the equator of the
/// sphere. With κ⋆ = +1 and σ = 0.9 the κ=0 chart radius of the farthest point
/// approaches the conjugate cap, so the optimum is the cap.
#[test]
fn response_curvature_flags_rail_on_high_curvature_relative_to_spread() {
    let dim = 5usize; // matches the OLMo per-layer fingerprint dim
    let n = 2000usize;
    // Strongly-curved sphere with a geodesic spread that fills a large fraction of
    // it: the κ=0 chart radius of the farthest point approaches the conjugate cap,
    // so κ is unresolvable beyond it — the OLMo unit-normalised regime. We cap each
    // tangent at a safe fraction of the conjugate radius `π/(2√κ⋆)` so EVERY draw
    // stays strictly in-chart (no exp_map antipode error), while the spread is still
    // a large fraction of the sphere (the fill-the-sphere / rail condition).
    let k_star = 4.0;
    let seed = 0x1104_0000_DEAD_BEEF_u64; // distinct deterministic seed
    let values = synth_cloud_fills_sphere(dim, k_star, n, seed);

    let fit = fit_response_curvature(values.view(), dim, LEVEL, FIT_TOL, FIT_ITERS)
        .expect("response curvature fit on high-curvature-relative-to-spread cloud");

    println!(
        "\n#1104 chart-resolution rail: κ̂={:.3}  κ̂·r²={:.3}  r={:.3}  railed={}  CI=[{:.3},{:.3}]",
        fit.kappa_hat,
        fit.kappa_r2,
        fit.characteristic_radius,
        fit.railed_at_resolution_limit,
        fit.profile_ci.ci_lo,
        fit.profile_ci.ci_hi,
    );

    // (a) The estimator HONESTLY flags that κ̂ is at the chart-resolution limit:
    // it must NOT pretend this is an interior point estimate.
    assert!(
        fit.railed_at_resolution_limit,
        "high-curvature-relative-to-spread cloud must flag railed_at_resolution_limit; \
         got κ̂={} not flagged (silent rail — the #1104 bug)",
        fit.kappa_hat
    );

    // (b) The scale-FREE invariant κ̂·r² sits near the cloud-fills-the-sphere
    // sentinel (0.9π)² ≈ 8.0 — the cloud genuinely fills the sphere; it does NOT
    // read as flat (which a scale-dependent tiny-r² reading could falsely suggest).
    let sentinel = (0.9 * std::f64::consts::PI).powi(2);
    assert!(
        fit.kappa_r2 > 0.5 * sentinel,
        "scale-free κ̂·r²={:.3} should be near the fill-the-sphere sentinel {:.3}, \
         not a small (falsely-flat) value",
        fit.kappa_r2,
        sentinel
    );
    assert!(
        fit.kappa_r2 <= sentinel + 1.0e-6,
        "κ̂·r²={:.3} cannot exceed the conjugate-cap sentinel {:.3}",
        fit.kappa_r2,
        sentinel
    );

    // (c) The dimensional κ̂ remains finite and positive (a spherical lower bound),
    // and the reported characteristic radius is the scale it is dimensionless to.
    assert!(
        fit.kappa_hat.is_finite() && fit.kappa_hat > 0.0,
        "railed κ̂ must be a finite positive lower bound on |κ|, got {}",
        fit.kappa_hat
    );
    assert!(fit.characteristic_radius > 0.0);

    // (d) Contrast: a WELL-RESOLVED interior cloud (small spread vs chart radius)
    // at the SAME κ⋆ must NOT rail — the flag is specific to the unresolvable
    // regime, not always-on.
    let interior = synth_cloud(dim, k_star, n, 0.04, seed ^ 0xABCD);
    let fit_in = fit_response_curvature(interior.view(), dim, LEVEL, FIT_TOL, FIT_ITERS)
        .expect("interior fit");
    assert!(
        !fit_in.railed_at_resolution_limit,
        "well-resolved interior cloud must NOT flag a rail (κ̂={}, κ̂·r²={})",
        fit_in.kappa_hat, fit_in.kappa_r2
    );
}

/// Recover the chart-validity / conjugate-radius bracket the estimator uses, so
/// the recovery test's "interior, not railed" check is against the SAME bounds
/// the golden-section search ran inside. `response_kappa_bounds` is private, so
/// we reconstruct its two endpoints from the same public quantities: the lower
/// (hyperbolic) bound `−0.999/max‖yᵢ‖²` from the chart origin, and the upper
/// (spherical conjugate) cap `(0.9π / (2·max‖yᵢ−μ‖))²` from the centroid.
fn bracket_via_criterion(values: ndarray::ArrayView2<'_, f64>) -> (f64, f64) {
    let (n_rows, dim) = values.dim();
    let mut r2_max = 0.0_f64;
    for row in values.outer_iter() {
        let r2 = row.dot(&row);
        if r2 > r2_max {
            r2_max = r2;
        }
    }
    let mut centroid = Array1::<f64>::zeros(dim.max(1));
    if n_rows > 0 && dim > 0 {
        for row in values.outer_iter() {
            centroid += &row;
        }
        centroid.mapv_inplace(|v| v / n_rows as f64);
    }
    let mut s2_max = 0.0_f64;
    for row in values.outer_iter() {
        let diff = &row - &centroid;
        let r2 = diff.dot(&diff);
        if r2 > s2_max {
            s2_max = r2;
        }
    }
    let kappa_min = if r2_max > 0.0 {
        -0.999 / r2_max
    } else {
        -1.0e6
    };
    let kappa_max = if s2_max > 0.0 {
        let rho_max = 2.0 * s2_max.sqrt();
        let edge = 0.9 * std::f64::consts::PI / rho_max;
        edge * edge
    } else {
        1.0e6
    };
    // Sanity: the bracket must enclose the criterion's evaluable region — probe
    // both endpoints so a future change to the bound formula that desyncs from
    // the estimator's own bracket fails loudly here rather than silently
    // mis-judging "railed".
    let mid = 0.5 * (kappa_min + kappa_max);
    response_curvature_criterion(values, dim, mid)
        .expect("criterion evaluable at bracket midpoint");
    (kappa_min, kappa_max)
}

/// (6) THE κ·r² RESOLUTION POWER CURVE + the honest flat-floor `sign_resolved`
/// contract (#944 power-analysis deliverable; closes the #1059 / capstone-#977
/// reopening defect "V_p rails to the +κ bound for hyperbolic truth").
///
/// ## The defect this pins, stated so it can lose
///
/// Curvature is resolvable only through the dimensionless product `κ·r²`, and the
/// per-point Fisher information for κ scales like `σ⁴` (module header). At the
/// small-spread negative-κ operating point the earlier recovery table uses
/// (σ = 0.08, κ⋆ < 0 ⇒ `|κ·r²| ≈ 0.025`), a SINGLE cloud's profiled-criterion
/// argmin κ̂ can land on the WRONG side of zero — empirically a coin-flip — and
/// because the chart's spherical cap is the nearer bracket bound, that wrong
/// landing rails toward `+κ`. That is the exact "rails to +1.9 for hyperbolic
/// truth" behaviour the capstone reopening flags.
///
/// The criterion is NOT biased: averaged over clouds it minimises at κ⋆ (the
/// estimand is sound). The failure is a RESOLUTION limit. The honest fix is to
/// (a) establish where κ·r² is large enough that the single-cloud point estimate
/// is reliable, and (b) make the estimator REPORT when it is below that floor so a
/// caller never quotes a sign-confident κ̂ on noise. This test asserts both, and
/// asserts the estimator's CI is honest at the floor (it never confidently claims
/// the WRONG geometry) — converting a silent rail into a reported finding.
///
/// ## What is asserted (all against self-constructed truth)
///
///   * **RESOLVED band.** At a well-resolved hyperbolic operating point
///     (`|κ·r²| ≳ 0.15`), single-cloud sign recovery is reliable across many
///     seeds AND `sign_resolved = true` AND the CI verdict is `Hyperbolic` — the
///     point estimate may be quoted with its sign.
///   * **HONEST FLAT FLOOR.** At the under-resolved operating point
///     (`|κ·r²| ≈ 0.025`, the recovery-table σ = 0.08 hyperbolic case), the
///     estimator must FLAG `sign_resolved = false` on essentially every cloud
///     (the point estimate's sign is noise) AND must NEVER return a CI that
///     confidently claims the WRONG (spherical) sign. A single railed κ̂ > 0 is
///     fine ONLY when `sign_resolved` is false — that is the rail made honest.
///   * **MONOTONE POWER.** The fraction of clouds whose CI confidently resolves
///     the (correct, hyperbolic) sign is monotone non-decreasing as `|κ·r²|` grows
///     across a σ ladder — the power curve the #944 charter asks for.
#[test]
fn response_curvature_sign_resolution_power_curve_and_honest_flat_floor() {
    let dim = DIM;
    let k_star = -2.0; // genuinely hyperbolic truth
    let n = 4000usize;
    // A σ ladder straddling the resolution floor. σ = 0.08 is the recovery-table
    // operating point where |κ·r²| ≈ 0.025 (the coin-flip / rail regime); σ = 0.20
    // lifts |κ·r²| to ≈ 0.16 (reliably resolved). r ≈ 2·(spread) ≈ 2·1.4·σ on this
    // isotropic cloud, so κ·r² ≈ k_star·(2.8σ)² grows quadratically in σ.
    let sigmas = [0.08_f64, 0.12, 0.16, 0.20];
    let reps = 24usize;

    println!(
        "\n#944 κ·r² sign-resolution power curve (dim={dim}, n={n}, κ⋆={k_star}, reps={reps}):"
    );
    println!("   σ      mean|κ·r²|   sign_resolved_rate   correct_sign_rate   wrong_confident");

    let mut resolved_rates: Vec<(f64, f64)> = Vec::with_capacity(sigmas.len());

    for &sigma in &sigmas {
        let mut sum_abs_kr2 = 0.0_f64;
        let mut n_sign_resolved = 0usize; // CI excludes 0 (verdict != Flat)
        let mut n_correct_sign = 0usize; // κ̂ < 0 (the true side)
        let mut n_wrong_confident = 0usize; // CI confidently claims +κ (Spherical) — the BUG
        for r in 0..reps {
            let seed = 0x944C_0FFE_0000_0000_u64
                ^ ((sigma.to_bits()).rotate_left(17))
                ^ ((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) + 1);
            let values = synth_cloud(dim, k_star, n, sigma, seed);
            let fit = fit_response_curvature(values.view(), dim, LEVEL, FIT_TOL, FIT_ITERS)
                .expect("response curvature fit on hyperbolic cloud");

            sum_abs_kr2 += fit.kappa_r2.abs();
            if fit.sign_resolved {
                n_sign_resolved += 1;
            }
            if fit.kappa_hat < 0.0 {
                n_correct_sign += 1;
            }
            // The estimand is hyperbolic; a CI that EXCLUDES 0 on the SPHERICAL side
            // is a confident WRONG-sign claim — the defect the reopening flags. It
            // must essentially never happen, and certainly never while the point
            // estimate also rails positive without a `sign_resolved = false` honesty
            // flag.
            if matches!(fit.profile_ci.verdict, CurvatureVerdict::Spherical) {
                n_wrong_confident += 1;
                // The CONTRACT: a confident-spherical verdict on hyperbolic truth is
                // already a defect, but the unforgivable one is a SILENT rail — κ̂ > 0
                // presented as resolved. `sign_resolved` must at minimum be coupled to
                // the verdict so the two surfaces can never disagree.
                assert!(
                    fit.sign_resolved,
                    "INTERNAL CONTRACT FAIL: verdict=Spherical but sign_resolved=false \
                     (σ={sigma}, κ̂={}, CI=[{:.4},{:.4}]) — the point-estimate honesty flag \
                     desynced from the CI verdict",
                    fit.kappa_hat, fit.profile_ci.ci_lo, fit.profile_ci.ci_hi
                );
            }
            // The CORE honesty contract: whenever the bare point estimate rails to the
            // WRONG (positive) sign, the estimator MUST have flagged the sign as
            // unresolved — never a silent sign-confident κ̂ > 0 on hyperbolic data.
            if fit.kappa_hat > 0.0 {
                assert!(
                    !fit.sign_resolved,
                    "SILENT RAIL: κ̂={} railed positive on hyperbolic truth (κ⋆={k_star}, \
                     σ={sigma}, κ·r²={:.4}) yet sign_resolved=true — the #1059/#977 defect: a \
                     wrong-signed point estimate quoted as resolved",
                    fit.kappa_hat, fit.kappa_r2
                );
            }
        }
        let mean_abs_kr2 = sum_abs_kr2 / reps as f64;
        let resolved_rate = n_sign_resolved as f64 / reps as f64;
        let correct_rate = n_correct_sign as f64 / reps as f64;
        println!(
            "  {sigma:4.2}   {mean_abs_kr2:9.4}   {resolved_rate:18.3}   {correct_rate:17.3}   {n_wrong_confident:>15}"
        );
        resolved_rates.push((mean_abs_kr2, resolved_rate));

        // No cloud, at any σ, may confidently claim the WRONG (spherical) sign on
        // hyperbolic truth more than a calibration-slack fraction of the time. The
        // CI is a 95% region, so a confident wrong-sign verdict is a >2.5σ event per
        // tail; with reps=24 we allow at most 1 (≈4%) before declaring the CI
        // anticonservative on the wrong side.
        assert!(
            n_wrong_confident <= 1,
            "ANTICONSERVATIVE CI: {n_wrong_confident}/{reps} clouds at σ={sigma} confidently \
             claimed SPHERICAL on hyperbolic truth (κ⋆={k_star}) — the CI must not resolve the \
             wrong sign"
        );
    }

    // ── HONEST FLAT FLOOR at the recovery-table operating point σ = 0.08. ──────
    // This is the exact "rails to +κ for hyperbolic truth" regime. Here |κ·r²| is
    // far below the resolution floor, so the estimator must DECLINE to resolve the
    // sign on essentially every cloud: sign_resolved=false dominates. (A handful of
    // genuinely-resolved clouds are fine; what is forbidden is the estimator
    // pretending it can resolve the sign when it cannot.)
    let (floor_kr2, floor_resolved_rate) = resolved_rates[0];
    assert!(
        floor_kr2 < 0.06,
        "the σ=0.08 operating point must sit BELOW the resolution floor (mean|κ·r²|={floor_kr2:.4} \
         should be ≪ 0.1) — otherwise this is not the under-resolved regime the test pins"
    );
    assert!(
        floor_resolved_rate <= 0.40,
        "HONESTY FAIL: at the under-resolved floor (σ=0.08, |κ·r²|≈{floor_kr2:.4}) the estimator \
         flagged sign_resolved=true on {:.0}% of clouds — it must DECLINE to resolve the sign \
         where κ·r² is below the information floor, not quote a sign on noise",
        100.0 * floor_resolved_rate
    );

    // ── RESOLVED band at σ = 0.20: the sign becomes reliably resolvable. ───────
    // Above the floor the estimator must EARN its point estimate: most clouds
    // resolve the (correct) sign, demonstrating the flag is not vacuously always
    // false. This is the right end of the power curve.
    let (top_kr2, top_resolved_rate) = *resolved_rates.last().unwrap();
    assert!(
        top_kr2 >= 0.12,
        "the σ=0.20 operating point must clear the resolution floor (mean|κ·r²|={top_kr2:.4} \
         should be ≳ 0.12) — otherwise the ladder does not reach the resolved band"
    );
    assert!(
        top_resolved_rate >= 0.60,
        "POWER FAIL: at the resolved operating point (σ=0.20, |κ·r²|≈{top_kr2:.4}) only {:.0}% of \
         clouds resolved the sign — above the information floor the estimator must reliably \
         resolve curvature, else `sign_resolved` is vacuously useless",
        100.0 * top_resolved_rate
    );

    // ── MONOTONE POWER: resolution power rises with κ·r² across the ladder. ────
    // The single strongest statement that `sign_resolved` tracks GENUINE resolvable
    // information (κ·r²) and not an artefact: the resolved-rate is monotone
    // non-decreasing along the σ ladder, within a small Monte-Carlo wobble.
    for w in resolved_rates.windows(2) {
        let ((kr2_lo, rate_lo), (kr2_hi, rate_hi)) = (w[0], w[1]);
        assert!(
            rate_hi >= rate_lo - 0.10,
            "MONOTONE POWER FAIL: sign-resolution rate fell from {rate_lo:.3} (|κ·r²|≈{kr2_lo:.4}) \
             to {rate_hi:.3} (|κ·r²|≈{kr2_hi:.4}) — resolution power must grow with κ·r²"
        );
    }
    // And a strict net climb floor-to-top: the resolved band must be meaningfully
    // more resolvable than the flat floor (the power curve has real lift).
    assert!(
        top_resolved_rate >= floor_resolved_rate + 0.30,
        "POWER CURVE FLAT: resolution rate climbed only {:.3}→{:.3} floor→top — the κ·r² power \
         curve must show real lift between the under-resolved and resolved operating points",
        floor_resolved_rate, top_resolved_rate
    );
}

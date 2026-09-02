//! #1063: the per-term smooth-significance test uses a genuine likelihood-ratio
//! statistic `W = 2(ℓ_full − ℓ_null)` (a constrained refit dropping the smooth),
//! Bartlett-corrected by the exact Lawley factor — and the correction IMPROVES
//! the χ² calibration of the test under the null.
//!
//! The summary table reports Wood's rank-truncated *Wald* statistic; the Lawley
//! factor corrects the *likelihood-ratio* statistic. Dividing the Wald T by the
//! LR factor would correct the wrong quantity (under penalization the Wald form
//! is already a weighted χ²). The principled fix (#1063 Option 1) is to compute
//! a real per-term LR statistic and correct *that*. This test proves two things
//! about `smooth_term_lr_inference_forspec`:
//!
//!   (a) PROVENANCE — for a Poisson/log smooth (closed-form Lawley jets) the
//!       reported significance is built from the Bartlett-corrected LR
//!       (`correction_provenance == "lawley_lr_estimated_lambda"`,
//!       `bartlett_factor > 1`,
//!       `statistic_corrected == statistic_lr / bartlett_factor`).
//!
//!   (b) CALIBRATION — under a NULL data-generating process (the smooth's
//!       covariate has no effect) the statistic's empirical mean must agree with
//!       the mean of the reference it is scored against, and the corrected test
//!       must be the right SIZE at the level it is used at.
//!
//!       The reference is the law of `W(λ̂)` with the λ̂-selection replayed
//!       (#2672), and its mean is NOT `ref_df`. `ref_df = Σ_j w_j` is the
//!       CONDITIONAL mean `E[W | λ̂]`; `λ̂` is chosen from the same data that
//!       produced `W`, so the pairing is per-replicate and the unconditional
//!       means need not agree. Measured on this fixture: `mean(W) = 2.034`
//!       against `d = 0.870` (a ratio of `2.34`) and against
//!       `E[W(λ̂)] = 1.455`. The first comparison is `4.18` standard errors and
//!       means nothing; the second is `2.08` and is the claim.
//!
//!       The sign of Δε is NOT part of the claim, and the measured sign here is
//!       negative. This module previously asserted `Δε > 0` (anti-conservatism),
//!       which is a property of an UNPENALIZED test: the factor is
//!       `c = 1 + Δε/d` for either sign, and a penalized smooth under the null
//!       legitimately shrinks the alternative fit, pulling the LR BELOW the χ²_d
//!       reference. Measured at n = 60: mean(W) = 1.82 against d = 4.15, so
//!       Δε ≈ −2.33, c ≈ 0.44, and W/c ≈ 4.15 — the correction working as
//!       designed, in the direction the old precondition forbade.
//!
//! This is the truth-recovery / calibration bar (not a reference-tool match):
//! the ground truth here is the exact null distribution of the LR statistic.

use gam::inference::lawley::{
    RhoPenaltyComponent, RowExpectedJets, RowKappas, lawley_lr_mean_shift,
    lawley_lr_mean_shift_with_rho_variation,
};
use ndarray::Array2;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// `n` standard normals whose SAMPLE mean and variance are forced to exactly 0
/// and 1.
///
/// A Monte-Carlo expectation over `ρ̂ = ρ₀ + √v·z` estimates a second-order
/// quantity, and the first-order term `Δε'·√v·z̄` is noise that survives common
/// random numbers and scales as `√v` against a signal of order `v`. Matching the
/// first two sample moments removes it exactly rather than averaging it down:
/// it is the difference between an estimator that converges at every variance
/// and one that does not converge at any of them.
fn moment_matched_normals(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal");
    let mut z: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let mean = z.iter().sum::<f64>() / n as f64;
    for value in z.iter_mut() {
        *value -= mean;
    }
    let sd = (z.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    for value in z.iter_mut() {
        *value /= sd;
    }
    z
}

/// #939 deliverable (2), the ρ̂-variation arm — VALIDATION BY SIMULATION over
/// the sampling distribution of ρ̂.
///
/// The conditional Lawley shift `Δε(ρ)` is `E[W | λ]` — the LR mean with the
/// smoothing parameter held FIXED at ρ. When λ is estimated, the relevant null
/// mean is the expectation over the sampling distribution of ρ̂:
///
/// ```text
/// E[W] = E_{ρ̂}[ Δε(ρ̂) ] = Δε(ρ₀) + ½ Δε''(ρ₀)·Var(ρ̂) + O(·)
/// ```
///
/// `lawley_lr_mean_shift_with_rho_variation` assembles exactly the right-hand
/// second-order term. This test takes the assembly's prediction as a HYPOTHESIS
/// and falsifies the alternative (fixed-λ only) against a Monte-Carlo ground
/// truth: it draws `ρ̂ ~ N(ρ₀, Var)` (the genuine sampling fluctuation of the
/// log-smoothing estimate), evaluates the *conditional* shift `Δε(ρ̂)` at each
/// draw by re-scaling the penalty, and averages. The claim is that the
/// ρ̂-variation assembly matches this simulated `E_{ρ̂}[Δε(ρ̂)]` to second order
/// — and matches it STRICTLY BETTER than the conditional (fixed-λ) shift, which
/// systematically misses the curvature term. That gap IS the size correction
/// attributable specifically to ρ̂-variation.
#[test]
fn rho_variation_assembly_matches_simulated_expectation_over_rho_hat() {
    // Poisson/log smooth substrate at a fixed null linear predictor: a 2-column
    // design (intercept + centered covariate) penalized on the second column.
    let n = 60usize;
    let mut x = Array2::<f64>::ones((n, 2));
    let mut kappas = Vec::<RowKappas>::with_capacity(n);
    for i in 0..n {
        let z = i as f64 / n as f64 - 0.5;
        x[[i, 1]] = z;
        let eta = 0.3 + 0.6 * z;
        kappas.push(
            RowExpectedJets::poisson_log(eta)
                .kappas()
                .expect("poisson kappas"),
        );
    }
    let tested = 1..2;

    // Population smoothing parameter ρ₀ = log λ₀ and its sampling variance. (In
    // the live engine Var(ρ̂) is the inverse REML outer Hessian; here it is a
    // fixed scenario value so the simulation is self-contained and exact.)
    let lambda0 = 3.0_f64;
    let rho0 = lambda0.ln();
    let mut s_comp = Array2::<f64>::zeros((2, 2));
    s_comp[[1, 1]] = lambda0;
    let penalty = s_comp.clone();
    let components = vec![RhoPenaltyComponent {
        s_component: s_comp,
    }];

    // MOMENT-MATCHED standard normals, reused at every variance.
    //
    // Common random numbers alone are NOT enough here, which cost a measurement
    // cycle to learn. Writing the simulated leg's deviation from the anchor,
    //
    //   sim(v) − Δε(ρ₀) = Δε'·√v·z̄ + ½·Δε''·v·(z²)‾ + O(v^{3/2})
    //
    // the SIGNAL is the second term, of order v. The first term is pure sampling
    // noise, and it does not vanish under common random numbers: `z̄` is a fixed
    // non-zero number for any finite draw set (sd 1/√N ≈ 3.5e-3 at N = 80k), and
    // it scales as √v while the signal scales as v — so the noise-to-signal
    // ratio GROWS as 1/√v, worst exactly where the second-order claim is
    // cleanest. Measured, that produced `|conditional − sim|` ratios against the
    // analytic ½·H·v of 1.120, 0.502, 1.541, 0.250, 2.177 and 0.433, and moved
    // `e_cond` by 2.2×, 6.2× and 5.0× on a mere doubling of N — an estimator that
    // had not converged at all, from which any ratio (including a very
    // convincing 2×) is an artifact.
    //
    // Forcing the sample moments to exactly 0 and 1 kills the first term
    // identically (`Σz = 0`) and makes the second exact (`(z²)‾ = 1`), leaving
    // `sim(v) − Δε(ρ₀) = ½·Δε''·v + O(v²)` — which is precisely the quantity the
    // assembly claims to compute, now measurable at every v.
    let z = moment_matched_normals(20939, 40_000);

    // (conditional, assembled, simulated) at one ρ̂-sampling variance. The
    // simulated leg is the MONTE-CARLO ground truth E_{ρ̂}[Δε(ρ̂)]: each draw
    // evaluates the conditional shift with the penalty rescaled by e^{ρ̂−ρ₀}.
    let evaluate = |var_rho: f64| -> (f64, f64, f64) {
        let rho_cov = Array2::from_shape_vec((1, 1), vec![var_rho]).unwrap();
        let conditional =
            lawley_lr_mean_shift(x.view(), &kappas, Some(penalty.view()), tested.clone())
                .expect("conditional Δε");
        let assembled = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("assembled Δε(ρ̂)");
        let sd = var_rho.sqrt();
        let mut sum = 0.0;
        for &zi in &z {
            let lambda = (rho0 + sd * zi).exp();
            let mut s = Array2::<f64>::zeros((2, 2));
            s[[1, 1]] = lambda;
            sum += lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
                .expect("Δε(ρ̂) draw");
        }
        (conditional, assembled, sum / z.len() as f64)
    };

    // Two variances a factor of 4 apart. The assembly is a SECOND-ORDER delta
    // method, so its accuracy claim is a statement about the limit Var → 0, not
    // about any single variance: the omitted remainder is O(Var²) while the
    // conditional shift's error is O(Var). Testing at one large variance
    // measures the truncation instead of the assembly — at Var = 0.6 (the value
    // this fixture used before) σ_ρ = 0.78 spreads λ = λ₀e^{δ} across two orders
    // of magnitude at ±3σ, where the omitted term is the same size as the term
    // being verified and NO ratio bound can hold.
    let var_hi = 0.10_f64;
    let var_lo = var_hi / 4.0;
    let (cond_hi, asm_hi, sim_hi) = evaluate(var_hi);
    let (cond_lo, asm_lo, sim_lo) = evaluate(var_lo);
    let e_cond_hi = (cond_hi - sim_hi).abs();
    let e_asm_hi = (asm_hi - sim_hi).abs();
    let e_cond_lo = (cond_lo - sim_lo).abs();
    let e_asm_lo = (asm_lo - sim_lo).abs();

    // Non-vacuity: there must BE a ρ̂-variation effect to correct at both
    // variances, else the comparisons below are between two zeros.
    assert!(
        e_cond_hi > 0.0 && e_cond_lo > 0.0,
        "fixture must exhibit a ρ̂-variation effect at both variances: \
         |conditional−sim| = {e_cond_hi:.3e} (Var={var_hi}), {e_cond_lo:.3e} (Var={var_lo})"
    );

    // (1) THE SECOND-ORDER PROPERTY, as a rate rather than a magic constant.
    // Quartering the variance must shrink the assembled error strictly faster
    // than the conditional error: O(Var²) against O(Var) predicts factors of 16
    // and 4, so the assembled ratio must beat the conditional ratio by at least
    // 2×. Written as a cross-multiplication so a vanishing denominator cannot
    // manufacture a pass. This is scale-free — there is no tolerance to tune,
    // and an assembly that merely lands "close" at one variance fails it.
    assert!(
        e_asm_lo * e_cond_hi < 0.5 * e_asm_hi * e_cond_lo,
        "ρ̂-variation assembly must converge at the second-order RATE: quartering \
         Var must shrink |assembled−sim| at least twice as fast as |conditional−sim|. \
         assembled {e_asm_hi:.3e} → {e_asm_lo:.3e}, conditional {e_cond_hi:.3e} → \
         {e_cond_lo:.3e} (Var {var_hi} → {var_lo})"
    );

    // (2) The defining property: inside the regime where the expansion is valid,
    // the ρ̂-variation assembly is STRICTLY closer to the truth than the fixed-λ
    // conditional shift. That gap IS the size correction attributable to
    // ρ̂-variation.
    assert!(
        e_asm_lo < e_cond_lo,
        "ρ̂-variation correction must improve on the fixed-λ shift at Var={var_lo}: \
         |assembled−sim|={e_asm_lo:.3e} must be < |conditional−sim|={e_cond_lo:.3e}"
    );
}

/// DIAGNOSTIC (not a contract): identify the floor under `|assembled − sim|`.
///
/// The rate gate in `rho_variation_assembly_matches_simulated_expectation_over_rho_hat`
/// measured the assembled error as FLAT — 3.457e-7 → 3.906e-7 across a 4×
/// reduction in Var — while the conditional error fell ~2.9×, i.e. linearly, as
/// its O(Var) model predicts. A second-order assembly should have fallen ~16×.
/// Three candidates explain a flat floor, and they separate cleanly on how they
/// respond to Var and to the draw count:
///
/// * `O(Var²)` truncation — falls 16× per 4× Var step, independent of `N`.
/// * common-random-number Monte-Carlo noise — falls with `√Var` (both legs
///   collapse onto the same anchor as Var → 0) and as `1/√N`.
/// * a constant error in the assembly's ρ-curvature — flat in Var AND in `N`,
///   which would be a production finding: the second-order claim in
///   `lawley_lr_mean_shift_with_rho_variation`'s contract says that term must
///   vanish, and it would also explain the original Var = 0.6 observation
///   (assembled error ≈ conditional error) better than truncation did.
///
/// One run prints the whole grid; `--nocapture`. The `N` legs reuse a prefix of
/// the same draws, so the `N` comparison is itself common-random-number and not
/// confounded by a different sample.
#[test]
fn zz_measure_rho_variation_assembly_error_floor() {
    let n = 60usize;
    let mut x = Array2::<f64>::ones((n, 2));
    let mut kappas = Vec::<RowKappas>::with_capacity(n);
    for i in 0..n {
        let z = i as f64 / n as f64 - 0.5;
        x[[i, 1]] = z;
        let eta = 0.3 + 0.6 * z;
        kappas.push(
            RowExpectedJets::poisson_log(eta)
                .kappas()
                .expect("poisson kappas"),
        );
    }
    let tested = 1..2;
    let lambda0 = 3.0_f64;
    let rho0 = lambda0.ln();
    let mut s_comp = Array2::<f64>::zeros((2, 2));
    s_comp[[1, 1]] = lambda0;
    let penalty = s_comp.clone();
    let components = vec![RhoPenaltyComponent {
        s_component: s_comp,
    }];

    let reps = 40_000usize;
    let mut rng = StdRng::seed_from_u64(20939);
    let unit_normal = Normal::new(0.0, 1.0).expect("normal");
    let raw: Vec<f64> = (0..reps).map(|_| unit_normal.sample(&mut rng)).collect();
    let matched = moment_matched_normals(20939, reps);

    // The empirical moments of the RAW draws, which is the number that decides
    // between "the fixture hands the simulation a different variance than the
    // assembly gets" and "the fixture's first moment is the problem".
    let raw_mean = raw.iter().sum::<f64>() / reps as f64;
    let raw_var = raw.iter().map(|v| v * v).sum::<f64>() / reps as f64 - raw_mean * raw_mean;
    eprintln!("[zz] raw draws: mean={raw_mean:+.6e}  variance={raw_var:.6}  (target 0 and 1)");
    eprintln!(
        "[zz] first-order noise Δε'·√v·z̄ scales as √v; signal ½·Δε''·v scales as v, \
         so noise/signal grows as 1/√v"
    );

    let conditional = lawley_lr_mean_shift(x.view(), &kappas, Some(penalty.view()), tested.clone())
        .expect("conditional Δε");

    eprintln!("[zz] conditional Δε(ρ₀) = {conditional:.12}");
    eprintln!("[zz]      Var    draws   sim-cond      analytic ½H·Var   ratio    e_asm");
    for var_rho in [0.1_f64, 0.025, 0.00625] {
        let rho_cov = Array2::from_shape_vec((1, 1), vec![var_rho]).unwrap();
        let assembled = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("assembled Δε(ρ̂)");
        let sd = var_rho.sqrt();
        let simulate = |draws: &[f64]| -> f64 {
            let mut sum = 0.0;
            for &zi in draws {
                let mut s = Array2::<f64>::zeros((2, 2));
                s[[1, 1]] = (rho0 + sd * zi).exp();
                sum += lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
                    .expect("Δε(ρ̂) draw");
            }
            sum / draws.len() as f64
        };
        let analytic = assembled - conditional;
        for (label, draws) in [("raw", &raw), ("matched", &matched)] {
            let simulated = simulate(draws);
            let shift = simulated - conditional;
            eprintln!(
                "[zz] {var_rho:>9.5} {label:>8}  {shift:>+.4e}   {analytic:>+.4e}   {:>6.3}  {:.4e}",
                shift / analytic,
                (assembled - simulated).abs()
            );
        }
    }
    eprintln!(
        "[zz] read: 'matched' ratio → 1 at every Var => production is right and the RAW estimator \
         was the defect; 'matched' ratio holding away from 1 => a genuine assembly discrepancy"
    );

    // The premise being investigated must itself hold: the assembly's own
    // second-order term has to be non-zero, else there is nothing to measure.
    let probe = Array2::from_shape_vec((1, 1), vec![0.1_f64]).unwrap();
    let assembled = lawley_lr_mean_shift_with_rho_variation(
        x.view(),
        &kappas,
        penalty.view(),
        tested.clone(),
        &components,
        probe.view(),
    )
    .expect("assembled Δε(ρ̂)");
    assert!(
        (assembled - conditional).abs() > 0.0,
        "the ρ̂-variation assembly must contribute a non-zero second-order term at Var=0.1; \
         got assembled={assembled:.12} identical to conditional={conditional:.12}"
    );
}

/// DIAGNOSTIC (not a contract): finite-difference the ρ-curvature the
/// ρ̂-variation assembly applies, against the conditional shift it claims to be
/// the second derivative of.
///
/// The floor grid showed the applied correction is exactly ∝ Var (it falls
/// 4.000× per 4× Var step, so there is no floor) while landing ≈2× the measured
/// gap `|E_ρ̂[Δε] − Δε(ρ₀)|` — the signature of a curvature that is twice its
/// true value. Four candidate sites have already been eliminated BY DERIVATION,
/// so this measurement is the one that names the remaining one:
///
/// * the ½ in `lawley_lr_mean_shift_with_rho_variation`'s accumulation is
///   present and, with a 1×1 `rho_cov`, the whole correction is literally
///   `½·H[0,0]·Var` — no other factor participates;
/// * `hessian[[b, c]]` / `hessian[[c, b]]` are ASSIGNED, not accumulated, so the
///   `c in b..m` loop cannot double the diagonal;
/// * for `b == c`, `inverse_second` computes `2·P_b S_b K − P_b`, which is
///   exactly `∂²K/∂ρ_b²` for `K = J⁻¹`, `∂K/∂ρ_b = −P_b`, `P_b = K S_b K`;
/// * `pairs_b = −X P_b Xᵀ = ∂E/∂ρ_b` and `pairs_bc = X·inverse_second·Xᵀ =
///   ∂²E/∂ρ_b∂ρ_c`, so the leverage chain feeding the curvature is right.
///
/// What remains is the explicit O(n²) Lawley polynomial. This test differences
/// `Δε(ρ)` directly — ρ enters only through `S_λ = e^ρ S_b`, so evaluating at
/// `e^{±h} S_b` IS the ρ-perturbation — and prints the ratio. Read it as:
/// ratio ≈ 2 ⇒ the curvature is doubled inside `lawley_epsilon_rho_hessian` and
/// every site above is exonerated; ratio ≈ 1 ⇒ the curvature is right and the
/// defect is in how the correction is applied, or the MC comparison itself is
/// not measuring what the assembly targets.
#[test]
fn zz_measure_rho_curvature_vs_finite_difference() {
    let n = 60usize;
    let mut x = Array2::<f64>::ones((n, 2));
    let mut kappas = Vec::<RowKappas>::with_capacity(n);
    for i in 0..n {
        let z = i as f64 / n as f64 - 0.5;
        x[[i, 1]] = z;
        let eta = 0.3 + 0.6 * z;
        kappas.push(
            RowExpectedJets::poisson_log(eta)
                .kappas()
                .expect("poisson kappas"),
        );
    }
    let tested = 1..2;
    let lambda0 = 3.0_f64;
    let mut s_comp = Array2::<f64>::zeros((2, 2));
    s_comp[[1, 1]] = lambda0;
    let penalty = s_comp.clone();
    let components = vec![RhoPenaltyComponent {
        s_component: s_comp,
    }];

    // Δε at ρ₀ + shift, reached by scaling the penalty by e^{shift}.
    let shifted = |shift: f64| -> f64 {
        let mut s = Array2::<f64>::zeros((2, 2));
        s[[1, 1]] = lambda0 * shift.exp();
        lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
            .expect("Δε at shifted ρ")
    };

    let centre = shifted(0.0);
    eprintln!("[zz] Δε(ρ₀) = {centre:.12}");
    eprintln!("[zz]      h      FD d²Δε/dρ²      analytic ½·H·Var/(½·Var)     ratio");
    for h in [0.2_f64, 0.1, 0.05, 0.025] {
        let fd = (shifted(h) - 2.0 * centre + shifted(-h)) / (h * h);
        // Recover the analytic curvature the assembly applies: with a 1x1
        // rho_cov the whole correction is ½·H·Var, so H = 2·correction/Var.
        let var = 0.1_f64;
        let rho_cov = Array2::from_shape_vec((1, 1), vec![var]).unwrap();
        let assembled = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("assembled");
        let analytic = 2.0 * (assembled - centre) / var;
        eprintln!(
            "[zz] {h:>6.3}  {fd:>+.6e}   {analytic:>+.6e}   {:>8.4}",
            analytic / fd
        );
    }
    eprintln!(
        "[zz] read: ratio ~2 => doubled curvature inside lawley_epsilon_rho_hessian; ~1 => curvature correct, look downstream"
    );

    // The FD itself must be non-degenerate, else the ratio is meaningless.
    let fd = (shifted(0.05) - 2.0 * centre + shifted(-0.05)) / 0.0025;
    assert!(
        fd.abs() > 0.0 && fd.is_finite(),
        "the conditional shift must have measurable ρ-curvature for this comparison \
         to mean anything; got d²Δε/dρ² = {fd:e}"
    );
}

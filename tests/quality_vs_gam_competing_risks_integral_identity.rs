//! Crude (sub-distribution) cumulative-incidence consistency for gam's
//! competing-risks assembly, benchmarked against the mature non-parametric
//! standard — the **Aalen-Johansen** estimator as implemented by Python's
//! `lifelines.AalenJohansenFitter`.
//!
//! Capability under test: gam's crude-risk quadrature
//! (`assemble_competing_risks_cif`) must turn cause-specific *cumulative
//! hazards* into the correct *crude* (with-competing-mortality) cumulative
//! incidence
//!
//!     CIF_crude(t | cause d) = ∫₀ᵗ λ_d(u) · S_total(u) du,
//!     S_total(u) = exp(-[H_d(u) + H_m(u)]).
//!
//! This is the sub-distribution incidence that actually occurs in a population
//! where a competing event (mortality m) removes subjects from the at-risk set;
//! it is strictly below the *net* (cause-specific, competing-event-as-censored)
//! incidence 1 - exp(-H_d(t)). The whole point of the crude assembly is the
//! `S_total` factor, so we pin it down two ways, both on the SAME synthetic
//! competing-risks data (n = 500, seed = 1234):
//!
//!   1. Internal identity, tight: with the *constant* (exponential) hazards
//!      λ_d(t|x) = 0.05·e^{0.3x}, λ_m(t|x) = 0.02·e^{-0.1x} that generated the
//!      data, the per-subject crude incidence has the closed form
//!          CIF_d(t) = λ_d/(λ_d+λ_m) · (1 - e^{-(λ_d+λ_m)t}),
//!      an exact consequence of the integral identity above. gam's discrete
//!      product-limit assembly on a fine time grid must reproduce it.
//!
//!   2. External reference, mature tool: `lifelines.AalenJohansenFitter` fit to
//!      the SAME simulated (time, event∈{0,1,2}) data estimates the marginal
//!      crude CIF for cause d non-parametrically. gam's population-average crude
//!      CIF (assembled from the per-subject true cumulative hazards) must match
//!      the Aalen-Johansen estimate at the evaluation grid.
//!
//! A genuine divergence here is a real bug in the quadrature / `S_total`
//! factor and must fail — the bounds below are NOT to be loosened to pass.

use gam::survival::assemble_competing_risks_cif;
use gam::test_support::reference::{Column, relative_l2, run_python};
use ndarray::Array3;

// Population-level marginal crude incidence carries Monte-Carlo noise at
// n = 500, so the gam-vs-Aalen-Johansen comparison uses a sampling-aware
// bound; the closed-form internal identity (no sampling) uses a numerical one.
const N: usize = 500;
const EVAL_GRID: [f64; 4] = [10.0, 30.0, 50.0, 70.0];

// Cause-specific (constant) hazards that generate the data; the crude CIF is an
// exact integral of these, so they double as the closed-form ground truth.
fn lambda_d(x: f64) -> f64 {
    0.05 * (0.3 * x).exp()
}
fn lambda_m(x: f64) -> f64 {
    0.02 * (-0.1 * x).exp()
}

// Deterministic SplitMix64 → U(0,1), so gam and the Python reference see bit-
// identical simulated data without any RNG dependency.
fn splitmix_u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 53-bit mantissa → (0,1)
    ((z >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
}

#[test]
fn gam_crude_cif_matches_aalen_johansen_and_closed_form() {
    // ---- simulate identical competing-risks data (seed = 1234) ------------
    // Per subject: covariate x, competing exponential latent times T_d ~
    // Exp(λ_d), T_m ~ Exp(λ_m); observed = min, cause = argmin; an independent
    // uniform censoring time on [0, 28] yields ~35% censoring on t∈[0,100].
    let mut rng: u64 = 1234;
    let mut xs = Vec::with_capacity(N);
    let mut times = Vec::with_capacity(N);
    let mut events = Vec::with_capacity(N); // 0 = censored, 1 = disease d, 2 = mortality m
    let mut n_censored = 0usize;
    for _ in 0..N {
        // x ~ U(-2, 2): a continuous covariate driving both cause-specific rates.
        let x = -2.0 + 4.0 * splitmix_u01(&mut rng);
        let t_d = -(splitmix_u01(&mut rng)).ln() / lambda_d(x);
        let t_m = -(splitmix_u01(&mut rng)).ln() / lambda_m(x);
        let c = 28.0 * splitmix_u01(&mut rng);
        let (t_event, cause) = if t_d <= t_m { (t_d, 1u8) } else { (t_m, 2u8) };
        let (obs_t, obs_e) = if c < t_event {
            (c, 0u8)
        } else if t_event > 100.0 {
            (100.0, 0u8)
        } else {
            (t_event, cause)
        };
        if obs_e == 0 {
            n_censored += 1;
        }
        xs.push(x);
        times.push(obs_t);
        events.push(obs_e as f64);
    }
    let censor_frac = n_censored as f64 / N as f64;
    eprintln!("competing-risks sim: n={N} censored={censor_frac:.3}");
    assert!(
        (0.25..0.45).contains(&censor_frac),
        "synthetic censoring fraction {censor_frac:.3} drifted from the ~35% spec target"
    );

    // ---- gam crude CIF on a FINE time grid via assemble_competing_risks_cif
    // Feed the TRUE per-subject cumulative hazards H_k(t|x) = λ_k(x)·t (k∈{d,m})
    // into gam's product-limit assembly. Endpoint 0 = disease d, 1 = mortality m.
    // A dense grid (Δt = 0.25 up to 70) makes the discrete assembly converge to
    // the continuous integral identity; the evaluation grid points are sampled
    // from it for both the closed-form and Aalen-Johansen comparisons.
    let dt = 0.25_f64;
    let n_steps = (70.0 / dt) as usize; // grid endpoints 0.25 .. 70.0
    let fine_times: Vec<f64> = (1..=n_steps).map(|k| k as f64 * dt).collect();
    let n_times = fine_times.len();
    let cumulative = Array3::from_shape_fn((2, N, n_times), |(endpoint, row, t_idx)| {
        let x = xs[row];
        let rate = if endpoint == 0 { lambda_d(x) } else { lambda_m(x) };
        rate * fine_times[t_idx]
    });
    let cif = assemble_competing_risks_cif(ndarray::aview1(&fine_times), cumulative.view())
        .expect("gam assembles competing-risks crude CIF from cumulative hazards");

    // Index of each evaluation time within the fine grid (exact multiples of dt).
    let grid_idx: Vec<usize> = EVAL_GRID
        .iter()
        .map(|&t| {
            fine_times
                .iter()
                .position(|&u| (u - t).abs() < 0.5 * dt)
                .expect("evaluation time lies on the fine grid")
        })
        .collect();

    // gam population-average crude incidence for cause d at the grid.
    let gam_crude_d: Vec<f64> = grid_idx
        .iter()
        .map(|&ti| {
            (0..N).map(|row| cif.cif[0][[row, ti]]).sum::<f64>() / N as f64
        })
        .collect();

    // ---- closed-form crude incidence (the exact integral identity) --------
    // Per subject CIF_d(t) = λ_d/(λ_d+λ_m)·(1 - e^{-(λ_d+λ_m)t}); average it.
    let closed_form_d: Vec<f64> = EVAL_GRID
        .iter()
        .map(|&t| {
            (0..N)
                .map(|row| {
                    let ld = lambda_d(xs[row]);
                    let lm = lambda_m(xs[row]);
                    let tot = ld + lm;
                    ld / tot * (1.0 - (-tot * t).exp())
                })
                .sum::<f64>()
                / N as f64
        })
        .collect();

    let rel_closed = relative_l2(&gam_crude_d, &closed_form_d);
    eprintln!(
        "crude CIF_d grid={EVAL_GRID:?} gam={gam_crude_d:?} closed_form={closed_form_d:?} \
         rel_l2(gam,closed)={rel_closed:.5}"
    );
    // The integral identity is exact; gam's product-limit assembly on Δt=0.25
    // is a deterministic quadrature of it, so agreement is a pure numerical
    // (discretization) question — 0.5% is generous for this step size and any
    // larger gap signals a bug in the S_total factor or the assembly recursion.
    assert!(
        rel_closed < 5e-3,
        "gam crude CIF diverges from the exact integral identity: rel_l2={rel_closed:.5}"
    );

    // ---- mature reference: lifelines Aalen-Johansen on the SAME data ------
    let r = run_python(
        &[
            Column::new("t", &times),
            Column::new("event", &events),
        ],
        r#"
import numpy as np
from lifelines import AalenJohansenFitter

t = np.asarray(df["t"], dtype=float)
e = np.asarray(df["event"], dtype=float).round().astype(int)
grid = np.array([10.0, 30.0, 50.0, 70.0])

# Crude (sub-distribution) cumulative incidence of cause 1 (disease d) in the
# presence of the competing event 2 (mortality m). Aalen-Johansen is the
# standard non-parametric estimator of this quantity.
ajf = AalenJohansenFitter(calculate_variance=False)
ajf.fit(durations=t, event_observed=e, event_of_interest=1)
ci = ajf.cumulative_density_
# Step function: value at each grid time is the last estimate at or before it.
xs_t = ci.index.values.astype(float)
ys = ci.values[:, 0].astype(float)
out = []
for g in grid:
    mask = xs_t <= g
    out.append(float(ys[mask][-1]) if mask.any() else 0.0)
emit("cif_d", out)
"#,
    );
    let aj_crude_d = r.vector("cif_d");
    assert_eq!(aj_crude_d.len(), EVAL_GRID.len(), "AJ grid length mismatch");

    let rel_aj = relative_l2(&gam_crude_d, aj_crude_d);
    eprintln!(
        "crude CIF_d gam={gam_crude_d:?} aalen_johansen={aj_crude_d:?} rel_l2(gam,AJ)={rel_aj:.4}"
    );
    // gam's analytic-hazard crude CIF and the non-parametric Aalen-Johansen
    // estimate target the SAME marginal sub-distribution incidence; they differ
    // only by Aalen-Johansen's Monte-Carlo sampling noise at n=500. A relative
    // L2 of 0.06 over the [10,70] grid comfortably covers that sampling spread
    // while still rejecting any structural error (e.g. dropping the S_total
    // factor, which would inflate the crude CIF toward the net 1-e^{-H_d}).
    assert!(
        rel_aj < 0.06,
        "gam crude CIF disagrees with lifelines Aalen-Johansen beyond sampling noise: rel_l2={rel_aj:.4}"
    );
}

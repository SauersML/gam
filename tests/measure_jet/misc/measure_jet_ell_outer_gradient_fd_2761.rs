//! #2761: the FULL joint-REML gradient w.r.t. the measure-jet representer range
//! `ψ = ln ℓ` must match a central finite difference of the COMPLETE outer
//! criterion — not just of the penalty block.
//!
//! `learn_length_scale` is on by default (#2761), so every `mjs` fit now enrolls
//! `ln ℓ` as a real outer coordinate. Five fixtures in this target refuse with
//!
//! ```text
//!   line_search=StepSizeTooSmall after 50 attempt(s)
//!   [the direction descended but no step improved the objective]
//! ```
//!
//! at a checkpoint that reproduces to nine digits across unrelated changes. That
//! message has exactly two causes — a gradient that disagrees with its
//! objective, or an objective that is not smooth in the coordinate — and the
//! ℓ-profile probe already excluded one variant of the second (the design
//! refused to BUILD past `ℓ ≈ 2.8`; fixed by pulling the energy back through its
//! factor). This test settles the first, on the same fixture, through the
//! generic outer runner's structured analytic-vs-FD audit: the same instrument
//! `matern_2d_iso_kappa_outer_gradient_fd` uses for the Matérn `log κ`.
//!
//! `ln ℓ` is the exact analogue of that `log κ` — the module header calls it
//! "matérn's `log_kappa` analog" — so it is held to the same standard.

use gam::smooth::SpatialLengthScaleOptimizationOptions;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use csv::StringRecord;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

// `measure_jet_perf_parity`'s fixture: a 1-D curve in 3-D, the shape whose
// outer search refuses.
const N_TRAIN: usize = 1_500;
const SIGMA: f64 = 0.10;
const TRAIN_SEED: u64 = 1_039;

fn clamp_unit_open(x: f64) -> f64 {
    x.max(1.0e-6).min(1.0 - 1.0e-6)
}

fn latent_to_coords(t: f64) -> [f64; 3] {
    [
        clamp_unit_open(t),
        clamp_unit_open(0.5 + 0.5 * (2.0 * std::f64::consts::PI * t).sin()),
        clamp_unit_open(t * t),
    ]
}

fn truth(t: f64) -> f64 {
    (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (4.0 * std::f64::consts::PI * t).cos()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["x0", "x1", "x2", "y"]
        .iter()
        .cloned()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let t = latent.sample(&mut rng);
            let coords = latent_to_coords(t);
            let y = truth(t) + noise.sample(&mut rng);
            StringRecord::from(vec![
                coords[0].to_string(),
                coords[1].to_string(),
                coords[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// The same audit with the double-penalty null component switched off. It is a
/// discriminator, not a duplicate: the null component's shipped matrix is the
/// rebuilt metric-consistent ridge, whose `ln ℓ` jets the producer does NOT
/// emit (it differentiates the raw pullback instead), so if that desync were
/// the whole gradient error this arm would agree.
#[test]
fn measure_jet_ell_outer_gradient_matches_fd_without_double_penalty() {
    let record = audit("mjs(x0, x1, x2, centers=16, double_penalty=false)");
    let analytic = record.analytic_psi_gradient[0];
    let fd = record.finite_difference_psi_gradient[0];
    assert!(
        record.psi_fd_uncertainty[0] <= 5e-3 * fd.abs().max(1e-6),
        "the FD oracle did not resolve the single-penalty mjs ln-ell component:          fd={fd:.6e} uncertainty={:.3e}",
        record.psi_fd_uncertainty[0]
    );
    let gap = (analytic - fd).abs();
    let scale = analytic.abs().max(fd.abs()).max(1e-6);
    assert!(
        gap / scale < 5e-3,
        "single-penalty mjs ln-ell outer-gradient analytic != FD: analytic={analytic:.6e}          fd={fd:.6e} rel={:.3e}. With no null component in the model this cannot be the          producer/builder ridge desync (#2761).",
        gap / scale
    );
}

#[test]
fn measure_jet_ell_outer_gradient_matches_fd() {
    let audit = audit("mjs(x0, x1, x2, centers=16)");
    assert!(
        audit.theta.len() >= 2,
        "mjs(x0,x1,x2) must enroll at least one rho and the ln-ell coordinate; theta={:?}",
        audit.theta
    );
    assert_eq!(
        audit.psi_dim, 1,
        "single-scale mjs with learn_length_scale owns exactly one psi axis (ln ell)"
    );
    let analytic = audit.analytic_psi_gradient[0];
    let fd = audit.finite_difference_psi_gradient[0];
    assert!(
        analytic.is_finite() && fd.is_finite(),
        "non-finite mjs ln-ell gradient: analytic={analytic} fd={fd}"
    );
    // The oracle must have RESOLVED the component first: an unresolved finite
    // difference agrees with everything, so a gate that only checks the gap can
    // pass on a measurement that measured nothing.
    assert!(
        audit.psi_fd_uncertainty[0] <= 5e-3 * fd.abs().max(1e-6),
        "the outer-gradient FD oracle did not resolve the mjs ln-ell component: \
         fd={fd:.6e} uncertainty={:.3e} at step {:.3e} (order {})",
        audit.psi_fd_uncertainty[0],
        audit.psi_steps[0],
        audit.psi_fd_orders[0],
    );
    let gap = (analytic - fd).abs();
    let scale = analytic.abs().max(fd.abs()).max(1e-6);
    // Same bar as the Matern log-kappa gate: the audit Ridders-extrapolates and
    // reports its own uncertainty, so the tolerance describes the gradient
    // rather than the oracle's truncation.
    assert!(
        gap / scale < 5e-3,
        "mjs ln-ell outer-gradient analytic != FD: analytic={analytic:.6e} fd={fd:.6e} \
         gap={gap:.3e} rel={:.3e} step={:.3e} oracle_unc={:.3e} order={} theta={:?}. \
         An outer search cannot line-search on this: the direction it is handed is \
         not a descent direction of the objective it is minimizing (#2761).",
        gap / scale,
        audit.psi_steps[0],
        audit.psi_fd_uncertainty[0],
        audit.psi_fd_orders[0],
        audit.theta,
    );
}

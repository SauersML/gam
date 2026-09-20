#![cfg(test)]
//! #2933 F24 — the exact-A θ-adjoint on an embedded-sphere row differentiates the
//! Riemannian conversion the row factors.
//!
//! A sphere row of `A` is `P·M·P − ⟨g, x⟩·P + xxᵀ` with `P = I − xxᵀ`, and its cross
//! block is `P·M_tβ`. A coordinate direction moves `P`, the Weingarten scalar
//! `⟨g, x⟩` and the pin as well as `M`, and a decoder direction moves `⟨g, x⟩`. The
//! towers used to contract only the ambient `∂M/∂θ` against the whole inverse,
//! which prices the pin's `xᵀ ∂M x` and misses every conversion term
//! (`SphereRowConversion`). The oracle is a central difference of the priced
//! `log|A|` at a converged mode:
//! - a coordinate moves along a unit geodesic `cos s·x + sin s·v`, `v ⟂ x`, and
//!   the tower's slot functional contracts `v`;
//! - a decoder coefficient moves along its own axis.
//!
//! The F07 metric channel contracts `dB_raw` through the same tower, so its sphere rows
//! read the row residual as well. Its oracle is the central difference of `⟨X, Φ(θ)⟩`
//! for the conditioned evidence factor `Φ`, along the same directions.

use super::tests_recovery_split_780::FiniteDifferenceStratumCertificate;
use super::tests_sphere_ard_logdet_trace_2933::{
    converged_anchor, fixed_state_cache, sphere_logdet_fixture,
};
use super::*;

/// A copy of `state` holding the anchor's frozen gates, with `edit` applied and the
/// basis refreshed at the edited state.
fn edited_state(
    state: &SaeManifoldTerm,
    edit: impl FnOnce(&mut SaeManifoldTerm),
) -> SaeManifoldTerm {
    let mut edited = state.clone();
    edited.decoder_repulsion_gate = state.decoder_repulsion_gate.clone();
    edited.barrier_coactivation_gate = state.barrier_coactivation_gate.clone();
    edited.amplitude_barrier_gate = state.amplitude_barrier_gate;
    edited.streaming_gates_frozen = state.streaming_gates_frozen;
    edit(&mut edited);
    edited
        .refresh_basis_from_current_coords()
        .expect("basis refresh at the edited state");
    edited
}

/// The priced exact `log|A|` of `state` held fixed, with its stratum.
fn exact_log_det(
    state: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (f64, FiniteDifferenceStratumCertificate) {
    let (endpoint, cache) = fixed_state_cache(state, target, rho);
    let log_det = endpoint
        .exact_observed_information_log_dets(rho, target.view(), &cache)
        .expect("priced exact log|A| at the endpoint");
    (log_det, FiniteDifferenceStratumCertificate::from_arrow_cache(&cache))
}

/// Richardson-extrapolated central difference of `functional` along `endpoint(s)`, with
/// every endpoint in the anchor's stratum.
fn richardson_slope(
    label: &str,
    stratum: &FiniteDifferenceStratumCertificate,
    endpoint: &dyn Fn(f64) -> SaeManifoldTerm,
    functional: &dyn Fn(&SaeManifoldTerm) -> (f64, FiniteDifferenceStratumCertificate),
) -> (f64, f64) {
    let central = |h: f64| {
        let (plus, plus_stratum) = functional(&endpoint(h));
        let (minus, minus_stratum) = functional(&endpoint(-h));
        stratum.assert_same_stratum(&format!("{label} (+{h:e})"), &plus_stratum);
        stratum.assert_same_stratum(&format!("{label} (-{h:e})"), &minus_stratum);
        (plus - minus) / (2.0 * h)
    };
    let coarse = central(2.0e-4);
    let fine = central(1.0e-4);
    ((4.0 * fine - coarse) / 3.0, fine - coarse)
}

/// The worst relative gap between the θ-adjoint `gamma` and the Richardson slope of
/// `functional`, over two unit tangent geodesics at rows 1 and 6 and two decoder
/// coefficients, with the largest analytic tangent value and the per-direction report.
fn sphere_direction_gaps(
    state: &SaeManifoldTerm,
    cache: &ArrowFactorCache,
    gamma: &SaeArrowVector,
    functional: &dyn Fn(&SaeManifoldTerm) -> (f64, FiniteDifferenceStratumCertificate),
) -> (f64, f64, Vec<String>) {
    let stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(cache);
    let mut report = Vec::new();
    let mut worst = 0.0_f64;
    let mut tangent_signal = 0.0_f64;
    let mut record = |label: String, fd: f64, change: f64, analytic: f64| {
        let gap = (fd - analytic).abs() / analytic.abs().max(1.0);
        worst = worst.max(gap);
        report.push(format!(
            "{label}: fd {fd:.9e} (step change {change:.2e}) analytic {analytic:.9e} gap {gap:.3e}"
        ));
    };

    for row in [1_usize, 6] {
        let vars = state
            .row_vars_for_cache_row(row, cache)
            .expect("row layout of the cache");
        let slots: Vec<usize> = (0..3)
            .map(|axis| {
                vars.iter()
                    .position(|var| {
                        matches!(*var, SaeLocalRowVar::Coord { atom: 0, axis: held } if held == axis)
                    })
                    .expect("the row holds every axis of the sphere atom")
            })
            .collect();
        let x: [f64; 3] = std::array::from_fn(|axis| state.assignment.coords[0].row(row)[axis]);
        let base = cache.row_offsets[row];
        for seed in 0..2 {
            let raw: [f64; 3] =
                std::array::from_fn(|axis| (if axis == seed { 1.0 } else { 0.0 }) - x[seed] * x[axis]);
            let norm = raw.iter().map(|value| value * value).sum::<f64>().sqrt();
            let v = raw.map(|value| value / norm);
            let analytic: f64 = (0..3).map(|axis| v[axis] * gamma.t[base + slots[axis]]).sum();
            let endpoint = |s: f64| {
                edited_state(state, |edited| {
                    let mut flat = edited.assignment.coords[0].as_flat().clone();
                    for axis in 0..3 {
                        flat[row * 3 + axis] = s.cos() * x[axis] + s.sin() * v[axis];
                    }
                    edited.assignment.coords[0].set_flat(flat.view());
                })
            };
            let label = format!("row {row} tangent {seed}");
            let (fd, change) = richardson_slope(&label, &stratum, &endpoint, functional);
            tangent_signal = tangent_signal.max(analytic.abs());
            record(label, fd, change, analytic);
        }
    }

    let beta = state.flatten_beta();
    for index in [0_usize, beta.len() / 2] {
        let endpoint = |s: f64| {
            edited_state(state, |edited| {
                let mut moved = beta.clone();
                moved[index] += s;
                edited
                    .set_flat_beta(moved.view())
                    .expect("decoder coefficients at the endpoint");
            })
        };
        let label = format!("decoder {index}");
        let (fd, change) = richardson_slope(&label, &stratum, &endpoint, functional);
        record(label, fd, change, gamma.beta[index]);
    }
    (worst, tangent_signal, report)
}

#[test]
fn sphere_exact_a_theta_adjoint_matches_fd_of_the_log_det_2933_f24() {
    let (term, target, rho) = sphere_logdet_fixture();
    let anchor = converged_anchor(&term, &target, &rho);
    let (state, cache) = fixed_state_cache(&anchor, &target, &rho);
    assert!(
        !state.last_frames_active,
        "the fixture's border must be the full decoder, so a channel index is a flat β index"
    );
    let gamma = state
        .exact_a_theta_adjoint_joint(&rho, target.view(), &cache)
        .expect("exact-A joint θ-adjoint at the converged mode");
    let (worst, _, report) = sphere_direction_gaps(&state, &cache, &gamma, &|endpoint| {
        exact_log_det(endpoint, &target, &rho)
    });
    assert!(
        worst <= 1.0e-5,
        "the sphere-row exact-A θ-adjoint must be the derivative of the priced log|A|: {}",
        report.join("; ")
    );
}

/// The θ half of the F07 metric channel on sphere rows, against `⟨X, Φ(θ)⟩` with a fresh
/// evidence factor at every endpoint and ρ held fixed. Without the target, the tower
/// refused every sphere row here.
#[test]
fn sphere_evidence_metric_theta_channel_matches_fd_of_the_evidence_factor_2933_f24() {
    let (term, target, rho) = sphere_logdet_fixture();
    let anchor = converged_anchor(&term, &target, &rho);
    let (state, cache) = fixed_state_cache(&anchor, &target, &rho);
    assert!(
        !state.last_frames_active,
        "the fixture's border must be the full decoder, so a channel index is a flat β index"
    );
    let layout = (cache.delta_t_len(), cache.k);
    let dim = layout.0 + layout.1;
    let weight = Array2::from_shape_fn((dim, dim), |(i, j)| {
        ((i + 2 * j + 1) as f64 * 0.19).sin() + ((2 * i + j + 1) as f64 * 0.19).sin()
    });
    let (_, gamma) = state
        .evidence_metric_derivative_channels(&rho, target.view(), &cache, &weight, None)
        .expect("the metric θ-channel converts the sphere rows");
    let (worst, tangent_signal, report) =
        sphere_direction_gaps(&state, &cache, &gamma, &|endpoint| {
            let (_, moved) = fixed_state_cache(endpoint, &target, &rho);
            assert_eq!(
                (moved.delta_t_len(), moved.k),
                layout,
                "an endpoint must keep the row layout"
            );
            let metric = crate::manifold::tests::dense_evidence_metric(&moved);
            (
                (&weight * &metric).sum(),
                FiniteDifferenceStratumCertificate::from_arrow_cache(&moved),
            )
        });
    assert!(
        tangent_signal > 1.0e-4,
        "non-vacuity: the metric θ-channel must carry signal along a sphere tangent: {}",
        report.join("; ")
    );
    assert!(
        worst <= 1.0e-5,
        "the sphere-row metric θ-channel must be the derivative of ⟨X, Φ⟩: {}",
        report.join("; ")
    );
}

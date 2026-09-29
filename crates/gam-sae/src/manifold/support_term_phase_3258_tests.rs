//! #3258 — the support certificate on the noisy ring, where a weak periodic ARD prior is
//! the only thing that selects an atom's phase.
//!
//! The `pair_chart_fit_is_certified_reml_on_a_noisy_ring` cloud (256 rows, 8 periodic
//! atoms, top 2, seed `0xC0FFEE00D15EA5E5`) is solved once per `(log λ, ARD α)` and the
//! certificate's verdict at the reached state is pinned with and without the phase orbits.

use super::super::*;
use super::*;
use std::f64::consts::TAU;

/// The noisy-ring support term at its seed, and the centered target.
fn noisy_ring_seed_3258() -> (SaeSupportSparseTerm, Array2<f64>) {
    let n = 256usize;
    let mut cloud = Array2::<f64>::zeros((n, 2));
    let mut z = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = move || {
        z ^= z >> 12;
        z ^= z << 25;
        z ^= z >> 27;
        (z.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for i in 0..n {
        let theta = TAU * (i as f64) / (n as f64);
        cloud[[i, 0]] = theta.cos() + 0.04 * noise();
        cloud[[i, 1]] = theta.sin() + 0.04 * noise();
    }
    let mean = cloud.mean_axis(ndarray::Axis(0)).expect("rows");
    let target = Array2::from_shape_fn((n, 2), |(row, column)| cloud[[row, column]] - mean[column]);
    let basis = vec!["periodic".to_string(); 8];
    let dims = vec![1usize; 8];
    let random_state = 0xC0FF_EE00_D15E_A5E5u64;
    let d_max = sae_support_effective_atom_dims(&basis, &dims)
        .expect("effective dims")
        .into_iter()
        .max()
        .unwrap_or(1);
    let admission = crate::front_door::admit_topk_manifold(n, 2, 8, d_max, 2).expect("admission");
    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: target.view(),
        atom_basis: &basis,
        atom_dim: &dims,
        support_k: 2,
        random_state,
        admission,
    })
    .expect("seed");
    let retained = seed.retained_atom_indices;
    let term = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained.iter().map(|&atom| basis[atom].clone()).collect(),
        atom_dim: retained.iter().map(|&atom| dims[atom]).collect(),
        output_dim: 2,
        random_state,
    })
    .expect("term seed")
    .term;
    (term, target)
}

/// #3258 pins. Where a weak periodic ARD prior is the only thing that selects an atom's
/// phase, the direct certificate on the slice off the exact symmetries cannot hold (the
/// phase direction's curvature `αρ/‖ξ‖²` is below the shift it needs), and profiling the
/// phase out must decide the verdict instead: the phase-profiled certificate is certified
/// or names the unresolved phases, and withholding the phase orbits refuses at the same
/// state. Where the prior is strong the phase is resolved directly and nothing is profiled.
/// Each prior strength is its own test so one state's outcome never masks another's.
///
/// The helper is not itself a `#[test]`, so a state it cannot read is returned as the
/// refusal that names it and the calling test turns that into the failure. What the pin
/// asserts about a state it CAN read stays an `assert!` here, each carrying `diagnostic`,
/// which is built unconditionally at the state it describes so no assertion's message
/// depends on which assertion fires.
fn phase_pin_3258(log_lambda: f64, alpha: f64, weak: bool) -> Result<(), String> {
    let (seed, target) = noisy_ring_seed_3258();
    let k = seed.k_atoms();
    {
        let lambda = vec![log_lambda.exp(); k];
        let ard = vec![vec![alpha]; k];
        let mut term = seed.clone();
        let tolerance = term.fixed_point_tolerance();
        let report = term
            .solve_fixed_point(target.view(), &lambda, &ard, tolerance, 1.0)
            .map_err(|error| {
                format!("log λ = {log_lambda}, α = {alpha:e}: the fixed point certifies: {error}")
            })?;
        assert!(report.recurred, "log λ = {log_lambda}, α = {alpha:e}: {report:?}");
        let parameter_scale = term.parameter_iterate_scale().expect("parameter scale");
        let bound = tolerance * parameter_scale;
        let (newton, _) = term
            .exact_newton_solve(target.view(), &lambda, &ard)
            .expect("exact Newton displacement");
        let (offsets, beta_dim) = term.beta_layout().expect("beta layout");
        let generators = term
            .support_exact_symmetry_generators(&ard, &offsets, beta_dim)
            .expect("generators");
        let phase_orbits = term
            .support_phase_orbits(&ard, &offsets, beta_dim, bound)
            .expect("phase orbits");
        let verdict = term
            .support_kantorovich_certificate_on_slice(
                target.view(),
                &lambda,
                &ard,
                &newton,
                bound,
                &generators,
                &phase_orbits,
            )
            .expect("certificate");
        let withheld = term
            .support_kantorovich_certificate_on_slice(
                target.view(),
                &lambda,
                &ard,
                &newton,
                bound,
                &generators,
                &[],
            )
            .expect("certificate without the phase orbits");
        // The state this pin was read at, formatted once, before any assertion and
        // independently of all of them. Every assertion below carries it, so the failing
        // one reports the whole state rather than the one field it happened to test.
        let diagnostic = format!(
            "[#3258 pin] log λ = {log_lambda}, α = {alpha:e}: {} phase orbits, unresolved \
             {:?}, verdict {verdict:?}; without the orbits {withheld:?}",
            phase_orbits.len(),
            report.phase_unresolved_atoms,
        );
        if weak {
            match &verdict {
                SupportKantorovichVerdict::Certified { profiled_phases, .. } => {
                    assert!(*profiled_phases > 0, "{diagnostic}");
                    assert!(report.phase_unresolved_atoms.is_empty(), "{diagnostic}");
                }
                SupportKantorovichVerdict::PhaseUnresolved { phase_atoms, .. } => {
                    assert!(!phase_atoms.is_empty(), "{diagnostic}");
                    assert_eq!(&report.phase_unresolved_atoms, phase_atoms, "{diagnostic}");
                }
                SupportKantorovichVerdict::NotCertified(reason) => {
                    return Err(format!(
                        "α = {alpha:e}: the phase-profiled certificate refused: {reason}; \
                         {diagnostic}"
                    ));
                }
            }
            assert!(
                matches!(withheld, SupportKantorovichVerdict::NotCertified(_)),
                "α = {alpha:e}: the weak phase must defeat the direct certificate; {diagnostic}"
            );
        } else {
            assert!(
                matches!(
                    verdict,
                    SupportKantorovichVerdict::Certified { profiled_phases: 0, .. }
                ),
                "α = {alpha:e}: a strong prior resolves the phase directly; {diagnostic}"
            );
            assert!(report.phase_unresolved_atoms.is_empty(), "{diagnostic}");
        }
    }
    Ok(())
}

#[test]
fn phase_pin_3258_strong_prior() {
    phase_pin_3258(0.0, 1.0, false).expect("#3258 pin at log λ = 0, α = 1");
}

#[test]
fn phase_pin_3258_alpha_1e3() {
    phase_pin_3258(2.0, 1.0e-3, true).expect("#3258 pin at log λ = 2, α = 1e-3");
}

#[test]
fn phase_pin_3258_alpha_1e4() {
    phase_pin_3258(4.0, 1.0e-4, true).expect("#3258 pin at log λ = 4, α = 1e-4");
}

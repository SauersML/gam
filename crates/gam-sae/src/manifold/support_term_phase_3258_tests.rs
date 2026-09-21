//! #3258 — the direction behind the support certificate's refusals on the noisy ring.
//!
//! The `pair_chart_fit_is_certified_reml_on_a_noisy_ring` cloud (256 rows, 8 periodic
//! atoms, top 2, seed `0xC0FFEE00D15EA5E5`) is solved once per `(log λ, ARD α)` and the
//! dense exact Hessian `A` at the reached state is decomposed onto the per-atom periodic
//! phase generators `ξ_a` (`δt = 1` on the atom's rows, the harmonic pairs counter-rotated
//! on its decoder) and onto the exact symmetry generators the certificate projects off.

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

/// Pencil index of every `(row, slot)` coordinate of `atom` (one-dimensional atoms).
fn atom_pencil_indices_3258(
    term: &SaeSupportSparseTerm,
    atom: usize,
) -> Vec<(usize, usize, usize)> {
    let mut row_start = vec![0usize; term.n_obs() + 1];
    for row in 0..term.n_obs() {
        row_start[row + 1] = row_start[row] + term.assignment.coords_row(row).len();
    }
    term.atom_rows[atom]
        .iter()
        .map(|&(row, slot)| {
            let offset: usize = term.assignment.support_indices(row)[..slot]
                .iter()
                .map(|&other| term.assignment.atom_coord_dim(other as usize))
                .sum();
            (row, slot, row_start[row] + offset)
        })
        .collect()
}

struct PhaseAtom3258 {
    atom: usize,
    alpha: f64,
    kappa: f64,
    rho: f64,
    phi: f64,
    /// `(pencil index, cos κt, cos(κt − φ), sin κt)` per routed coordinate.
    coordinates: Vec<(usize, f64, f64, f64)>,
    generator: Array1<f64>,
}

fn phase_atoms_3258(term: &SaeSupportSparseTerm, ard: &[Vec<f64>]) -> Vec<PhaseAtom3258> {
    let (offsets, beta_dim) = term.beta_layout().expect("beta layout");
    let t_len = term.coordinate_state_len();
    let output_dim = term.output_dim();
    let mut out = Vec::new();
    for atom in 0..term.k_atoms() {
        if term.atoms[atom].basis_kind() != &SaeAtomBasisKind::Periodic
            || term.assignment.atom_coord_dim(atom) != 1
            || term.atom_rows[atom].is_empty()
        {
            continue;
        }
        let period = term.atom_ard_axis_periods(atom)[0].expect("periodic ARD axis");
        let kappa = TAU / period;
        let alpha = ard[atom][0];
        let indices = atom_pencil_indices_3258(term, atom);
        let (mut sine, mut cosine) = (0.0_f64, 0.0_f64);
        for &(row, slot, _) in &indices {
            let (sin, cos) = (kappa * term.assignment.coords_for_slot(row, slot)[0]).sin_cos();
            sine += sin;
            cosine += cos;
        }
        let rho = sine.hypot(cosine);
        let phi = sine.atan2(cosine);
        let mut generator = Array1::<f64>::zeros(t_len + beta_dim);
        let coordinates: Vec<(usize, f64, f64, f64)> = indices
            .iter()
            .map(|&(row, slot, index)| {
                let t = term.assignment.coords_for_slot(row, slot)[0];
                generator[index] = 1.0;
                (
                    index,
                    (kappa * t).cos(),
                    (kappa * t - phi).cos(),
                    (kappa * t).sin(),
                )
            })
            .collect();
        let decoder = term.atoms[atom].decoder_coefficients();
        let basis_size = term.atoms[atom].basis_size();
        for harmonic in 1..=(basis_size - 1) / 2 {
            let frequency = TAU * harmonic as f64;
            let (sine_index, cosine_index) = (2 * harmonic - 1, 2 * harmonic);
            for output in 0..output_dim {
                generator[t_len + offsets[atom] + sine_index * output_dim + output] =
                    frequency * decoder[[cosine_index, output]];
                generator[t_len + offsets[atom] + cosine_index * output_dim + output] =
                    -frequency * decoder[[sine_index, output]];
            }
        }
        out.push(PhaseAtom3258 {
            atom,
            alpha,
            kappa,
            rho,
            phi,
            coordinates,
            generator,
        });
    }
    out
}

/// Lowest eigenvalues of `M` on the slice orthogonal to the orthonormal columns `q`.
fn slice_spectrum_3258(matrix: &Array2<f64>, q: &Array2<f64>, count: usize) -> Vec<f64> {
    let dim = matrix.nrows();
    let projector = Array2::<f64>::eye(dim) - q.dot(&q.t());
    let mut sliced = projector.dot(matrix).dot(&projector);
    let stiffness = 2.0 * matrix.diag().iter().fold(0.0_f64, |a, b| a.max(b.abs())) + 1.0;
    sliced = sliced + q.dot(&q.t()) * stiffness;
    let (values, _) = sliced.eigh(Side::Lower).expect("slice spectrum");
    values.iter().take(count).copied().collect()
}

#[test]
fn noisy_ring_weak_direction_is_the_periodic_phase_3258() {
    let (seed, target) = noisy_ring_seed_3258();
    let k = seed.k_atoms();
    for (log_lambda, alpha) in [(0.0_f64, 1.0_f64), (0.0, 1.0e-2), (2.0, 1.0e-3)] {
        let lambda = vec![log_lambda.exp(); k];
        let ard = vec![vec![alpha]; k];
        let mut term = seed.clone();
        let tolerance = term.fixed_point_tolerance();
        let solved = term.solve_fixed_point(target.view(), &lambda, &ard, tolerance, 1.0);
        println!(
            "[#3258] log λ = {log_lambda}, α = {alpha:e}: solve_fixed_point {}",
            match &solved {
                Ok(_) => "Ok".to_string(),
                Err(error) => format!("Err({error})"),
            }
        );
        let (offsets, beta_dim) = term.beta_layout().expect("beta layout");
        let t_len = term.coordinate_state_len();
        let full = dense_exact_hessian_2576(&term, &target, &lambda, &ard);
        let kept: Vec<usize> = (0..full.nrows())
            .filter(|&index| !full.row(index).iter().all(|value| *value == 0.0))
            .collect();
        let exact = full
            .select(ndarray::Axis(0), &kept)
            .select(ndarray::Axis(1), &kept);
        let system = term
            .assemble_arrow_schur(target.view(), &lambda, &ard)
            .expect("arrow");
        let mut gradient = Array1::<f64>::zeros(t_len + beta_dim);
        for (row, block) in system.rows.iter().enumerate() {
            gradient
                .slice_mut(ndarray::s![
                    system.row_offsets[row]..system.row_offsets[row + 1]
                ])
                .assign(&block.gt);
        }
        gradient.slice_mut(ndarray::s![t_len..]).assign(&system.gb);
        let generators = term
            .support_exact_symmetry_generators(&ard, &offsets, beta_dim)
            .expect("generators");
        let null_kept: Vec<Array1<f64>> = generators
            .iter()
            .map(|g| g.select(ndarray::Axis(0), &kept))
            .collect();
        let phases = phase_atoms_3258(&term, &ard);
        let phase_kept: Vec<Array1<f64>> = phases
            .iter()
            .map(|p| p.generator.select(ndarray::Axis(0), &kept))
            .collect();
        let null_span = support_orthonormal_span(kept.len(), &null_kept).expect("N span");
        let phase_span = support_orthonormal_span(kept.len(), &phase_kept).expect("Ξ span");
        let mut both = null_kept.clone();
        both.extend(phase_kept.iter().cloned());
        let both_span = support_orthonormal_span(kept.len(), &both).expect("N ∪ Ξ span");
        println!(
            "[#3258]   pencil {} kept of {}, {} exact generators, {} phase atoms, basis sizes {:?}, \
             ARD periods {:?}",
            kept.len(),
            full.nrows(),
            generators.len(),
            phases.len(),
            (0..k)
                .map(|atom| term.atoms[atom].basis_size())
                .collect::<Vec<_>>(),
            (0..k)
                .map(|atom| term.atom_ard_axis_periods(atom).to_vec())
                .collect::<Vec<_>>(),
        );
        let (values, vectors) = exact.eigh(Side::Lower).expect("A spectrum");
        for mode in 0..6.min(values.len()) {
            let v = vectors.column(mode);
            let on_null = null_span.t().dot(&v);
            let on_phase = phase_span.t().dot(&v);
            let per_atom: Vec<String> = phases
                .iter()
                .zip(&phase_kept)
                .map(|(p, g)| {
                    let unit = g / g.dot(g).sqrt();
                    format!("{}:{:.3}", p.atom, unit.dot(&v).powi(2))
                })
                .collect();
            println!(
                "[#3258]   λ{} = {:.4e}: share on N {:.4}, on Ξ {:.4}, per atom [{}]",
                mode + 1,
                values[mode],
                on_null.dot(&on_null),
                on_phase.dot(&on_phase),
                per_atom.join(", ")
            );
        }
        // A_Φ: the Hessian of the phase-profiled objective, A + α diag(cos(κt−φ) − cos κt)
        // − (α/ρ) c cᵀ on each phase atom's coordinate block.
        let mut profiled = full.clone();
        for p in &phases {
            let unit = &p.generator / p.generator.dot(&p.generator).sqrt();
            let along_gradient = gradient.dot(&unit);
            let prior_gradient: f64 = p
                .coordinates
                .iter()
                .map(|c| p.alpha / p.kappa * c.3)
                .sum::<f64>()
                / p.generator.dot(&p.generator).sqrt();
            let curvature = unit.dot(&full.dot(&unit));
            let action = full.dot(&unit);
            let action_off = &action - &(&unit * curvature);
            let norm_c = p.coordinates.iter().map(|c| c.2 * c.2).sum::<f64>().sqrt();
            for &(i, cos_raw, cos_shifted, _) in &p.coordinates {
                profiled[[i, i]] += p.alpha * (cos_shifted - cos_raw);
                for &(j, _, cos_other, _) in &p.coordinates {
                    profiled[[i, j]] -= p.alpha / p.rho * cos_shifted * cos_other;
                }
            }
            let profiled_action = profiled.dot(&unit);
            println!(
                "[#3258]   atom {}: rows {}, α {:.1e}, κ {:.4}, ρ {:.4e}, φ {:.3e}, ‖c‖ {:.3}, ‖ξ‖ {:.4e}; \
                 ξ̂ᵀAξ̂ {:.4e} vs α Σcos κt/‖ξ‖² {:.4e} and αρ/‖ξ‖² {:.4e}; ‖(I−ξ̂ξ̂ᵀ)Aξ̂‖ {:.4e}; \
                 g·ξ̂ {:.4e} vs ∇E·ξ̂ {:.4e}; ‖A_Φ ξ̂‖ {:.4e}",
                p.atom,
                p.coordinates.len(),
                p.alpha,
                p.kappa,
                p.rho,
                p.phi,
                norm_c,
                p.generator.dot(&p.generator).sqrt(),
                curvature,
                p.alpha * p.coordinates.iter().map(|c| c.1).sum::<f64>()
                    / p.generator.dot(&p.generator),
                p.alpha * p.rho / p.generator.dot(&p.generator),
                action_off.dot(&action_off).sqrt(),
                along_gradient,
                prior_gradient,
                profiled_action.dot(&profiled_action).sqrt(),
            );
        }
        let profiled = profiled
            .select(ndarray::Axis(0), &kept)
            .select(ndarray::Axis(1), &kept);
        println!(
            "[#3258]   slice spectra: A off N {:?}; A off N∪Ξ {:?}; A_Φ off N∪Ξ {:?}",
            slice_spectrum_3258(&exact, &null_span, 3),
            slice_spectrum_3258(&exact, &both_span, 3),
            slice_spectrum_3258(&profiled, &both_span, 3),
        );
        let parameter_scale = term.parameter_iterate_scale().expect("parameter scale");
        let bound = tolerance * parameter_scale;
        let (newton, _) = term
            .exact_newton_solve(target.view(), &lambda, &ard)
            .expect("exact Newton displacement");
        let verdict = term
            .support_kantorovich_certificate(target.view(), &lambda, &ard, &newton, bound)
            .expect("certificate");
        println!(
            "[#3258]   bound {bound:.4e}, ‖Δ‖∞ {:.3e}, gradient band {:.3e}, verdict {verdict:?}",
            newton.max_abs(),
            term.gradient_rounding_band(target.view(), &lambda, &ard)
                .expect("band"),
        );
    }
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

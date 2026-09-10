// Included by construction_exact_hessian.rs in the construction module.

/// Apply the same Euclidean spectral pseudoinverse as the dense exact-A
/// solve, using Ritz directions of A rather than eigendirections of (A, B).
/// B supplies only each direction's classification scale. In particular,
/// neither an aggregate Rayleigh quotient nor a B-orthogonal projection can
/// define this inverse: both make the answer depend on the right-hand side.
fn solve_exact_stationarity_krylov<A, B>(
    rhs: &SaeArrowVector,
    apply_a: &A,
    apply_b: &B,
) -> Result<SaeArrowVector, String>
where
    A: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    B: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    use gam_linalg::lanczos::{
        SymmetricLanczosOptions, symmetric_lanczos_eigenpairs,
        symmetric_lanczos_eigenpairs_with_original_vectors,
    };

    let total_t = rhs.t.len();
    let dim = total_t + rhs.beta.len();
    let mut flat_rhs = Array1::zeros(dim);
    flat_rhs.slice_mut(s![..total_t]).assign(&rhs.t);
    flat_rhs.slice_mut(s![total_t..]).assign(&rhs.beta);
    let norm = |v: &Array1<f64>| v.iter().fold(0.0_f64, |n, &x| n.hypot(x));
    let rhs_norm = norm(&flat_rhs);
    if !rhs_norm.is_finite() {
        return Err("exact-stationarity Krylov solve: non-finite right-hand side".into());
    }
    let split = |v: &Array1<f64>| SaeArrowVector {
        t: v.slice(s![..total_t]).to_owned(),
        beta: v.slice(s![total_t..]).to_owned(),
    };
    if rhs_norm == 0.0 {
        return Ok(split(&flat_rhs));
    }
    let apply = |input: &[f64], output: &mut [f64]| -> Result<(), String> {
        let vector = SaeArrowVector {
            t: Array1::from_vec(input[..total_t].to_vec()),
            beta: Array1::from_vec(input[total_t..].to_vec()),
        };
        let value = apply_a(&vector)?;
        if value.t.len() != total_t || value.beta.len() != dim - total_t {
            return Err(
                "exact-stationarity Krylov solve: operator changed vector dimensions".into(),
            );
        }
        for (slot, &value) in output
            .iter_mut()
            .zip(value.t.iter().chain(value.beta.iter()))
        {
            *slot = value;
        }
        Ok(())
    };

    // Account for the retained basis, lifted Ritz vectors, small eigensystems,
    // and fixed work vectors. Refuse before allocating above the live budget.
    let budget = crate::manifold::sae_host_in_core_budget_bytes().0 as u128;
    let storage = |steps: usize| {
        let n = dim as u128;
        let m = steps as u128;
        8 * (3 * n * m + 4 * m * m + 12 * n)
    };
    if storage(1) > budget {
        return Err(format!(
            "exact-stationarity Krylov solve: one direction needs {} bytes, above budget {budget}",
            storage(1)
        ));
    }
    let (mut low, mut high) = (1, dim);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if storage(mid) <= budget {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let max_steps = low;
    let mut steps = (dim.ilog2() as usize + 1).min(max_steps);
    let tolerance = f64::EPSILON.sqrt();

    // An independent probe estimates the arithmetic scale even when the RHS
    // excites only a small eigenvalue. The identifiability scale is still
    // measured separately as v'Bv for every actual RHS Ritz direction.
    let mut state = 2828_u64;
    let norm_start: Vec<f64> = (0..dim)
        .map(|_| {
            let bits = gam_linalg::utils::splitmix64(&mut state);
            (bits >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5
        })
        .collect();
    let rhs_start = flat_rhs.to_vec();
    let mut scale_image = vec![0.0; dim];
    apply(&norm_start, &mut scale_image)?;
    let initial_scale =
        norm(&Array1::from_vec(scale_image)) / norm(&Array1::from_vec(norm_start.clone()));
    loop {
        let mut options = SymmetricLanczosOptions {
            max_steps: steps,
            // Exhaust the numerical invariant subspace, not the ambient
            // dimension: normalizing a roundoff-only Lanczos remainder creates
            // duplicate basis vectors when a RHS excites only a few modes.
            residual_tol: dim as f64 * f64::EPSILON * initial_scale,
            local_reorthogonalize: false,
            full_reorthogonalize: true,
        };
        let scale_pairs = symmetric_lanczos_eigenpairs(dim, &norm_start, options, &apply)?;
        let scale_index = (0..scale_pairs.eigenvalues.len())
            .max_by(|&a, &b| {
                scale_pairs.eigenvalues[a]
                    .abs()
                    .total_cmp(&scale_pairs.eigenvalues[b].abs())
            })
            .ok_or("exact-stationarity Krylov solve: empty scale spectrum")?;
        let scale_last = scale_pairs.eigenvalues.len() - 1;
        let scale_error =
            scale_pairs.residual_norm * scale_pairs.eigenvectors[[scale_last, scale_index]].abs();
        let mut spectral_norm = scale_pairs.eigenvalues[scale_index].abs();
        let scale_resolved = scale_error <= tolerance * spectral_norm || scale_error == 0.0;
        drop(scale_pairs);

        options.residual_tol = dim as f64 * f64::EPSILON * spectral_norm.max(initial_scale);
        let pairs =
            symmetric_lanczos_eigenpairs_with_original_vectors(dim, &rhs_start, options, &apply)?;
        spectral_norm = pairs
            .eigenvalues
            .iter()
            .fold(spectral_norm, |s, &v| s.max(v.abs()));
        let vectors = pairs
            .original_eigenvectors
            .as_ref()
            .ok_or("exact-stationarity Krylov solve: missing lifted Ritz vectors")?;
        let coefficients = vectors.t().dot(&flat_rhs);
        let mut inverse_coefficients = Array1::zeros(coefficients.len());
        let mut projected_coefficients = coefficients.clone();
        let mut discarded = Vec::new();
        let mut classification_resolved = true;
        for index in 0..pairs.eigenvalues.len() {
            let direction = split(&vectors.column(index).to_owned());
            let b_direction = apply_b(&direction)?;
            let metric = sae_inner(&direction, &b_direction);
            if !(metric.is_finite() && metric > 0.0) {
                return Err(format!(
                    "exact-stationarity Krylov solve: invalid v'Bv={metric}"
                ));
            }
            let magnitude = pairs.eigenvalues[index].abs();
            let floor = sae_exact_a_direction_floor(dim, spectral_norm, metric);
            let error = pairs.residual_norm
                * pairs.eigenvectors[[pairs.eigenvalues.len() - 1, index]].abs();
            // A Ritz interval crossing the cutoff has not decided which
            // spectral function to apply. Grow the subspace instead of guessing.
            classification_resolved &= error < (magnitude - floor).abs() || error == 0.0;
            if magnitude > floor {
                inverse_coefficients[index] = coefficients[index] / pairs.eigenvalues[index];
            } else {
                projected_coefficients[index] = 0.0;
                discarded.push(index);
            }
        }
        let solution = vectors.dot(&inverse_coefficients);
        let projected_rhs = vectors.dot(&projected_coefficients);
        let mut applied = vec![0.0; dim];
        apply(&solution.to_vec(), &mut applied)?;
        let residual = Array1::from_vec(applied) - &projected_rhs;
        let residual_norm = norm(&residual);
        let solution_norm = norm(&solution);
        let physical_scale = spectral_norm * solution_norm + norm(&projected_rhs);
        let mut normal = vec![0.0; dim];
        apply(&residual.to_vec(), &mut normal)?;
        let normal_norm = norm(&Array1::from_vec(normal));
        let solved_coefficients = vectors.t().dot(&solution);
        let null_mass = discarded
            .iter()
            .fold(0.0_f64, |n, &i| n.hypot(solved_coefficients[i]));
        if scale_resolved
            && classification_resolved
            && solution.iter().all(|x| x.is_finite())
            && residual_norm <= tolerance * physical_scale
            && normal_norm <= tolerance * spectral_norm * physical_scale
            && null_mass <= tolerance * solution_norm
        {
            return Ok(split(&solution));
        }
        if steps == max_steps {
            return Err(format!(
                "exact-stationarity Krylov pseudoinverse did not certify in {steps} directions \
                 (dimension {dim}, budget {budget} bytes): residual {residual_norm:.6e} / \
                 backward scale {physical_scale:.6e}, null mass {null_mass:.6e}, \
                 scale resolved={scale_resolved}, classification resolved={classification_resolved}"
            ));
        }
        // Geometric growth keeps repeated basis construction linear in the
        // final Krylov dimension, instead of nesting inverse solves per null.
        steps = steps.saturating_mul(2).min(max_steps);
    }
}

#[cfg(test)]
mod tests_null_space_policy_2828 {
    use super::*;
    use ndarray::array;

    fn compare(a: &Array2<f64>, b: &Array2<f64>, rhs: &Array1<f64>, expected: &Array1<f64>) {
        let split = |v: &Array1<f64>| SaeArrowVector {
            t: v.slice(s![..v.len() - 1]).to_owned(),
            beta: v.slice(s![v.len() - 1..]).to_owned(),
        };
        let flatten =
            |v: &SaeArrowVector| Array1::from_iter(v.t.iter().chain(v.beta.iter()).copied());
        let (eigenvalues, eigenvectors) = a.eigh(Side::Lower).expect("symmetric test operator");
        let spectral_norm = eigenvalues.iter().fold(0.0_f64, |s, &v| s.max(v.abs()));
        let metric_scale = Array1::from_iter(
            eigenvectors
                .columns()
                .into_iter()
                .map(|v| v.dot(&b.dot(&v))),
        );
        let dense = ExactHessianSpectralBlock {
            operator: a.clone(),
            eigenvalues,
            eigenvectors,
            metric_scale,
            spectral_norm,
        }
        .solve_stationarity(&split(rhs))
        .expect("dense truncated pseudoinverse");
        let apply_a = |v: &SaeArrowVector| Ok(split(&a.dot(&flatten(v))));
        let apply_b = |v: &SaeArrowVector| Ok(split(&b.dot(&flatten(v))));
        let matrix_free = solve_exact_stationarity_krylov(&split(rhs), &apply_a, &apply_b)
            .expect("matrix-free truncated pseudoinverse");
        for (label, actual) in [
            ("dense", flatten(&dense)),
            ("matrix-free", flatten(&matrix_free)),
        ] {
            for index in 0..rhs.len() {
                assert!(
                    (actual[index] - expected[index]).abs()
                        <= 1.0e-6 * (1.0 + expected[index].abs()),
                    "{label} response[{index}]={} vs {}, rhs={rhs:?}",
                    actual[index],
                    expected[index]
                );
            }
        }
    }

    #[test]
    fn both_routes_drop_null_rhs_without_changing_resolved_components_2828() {
        let floor = f64::EPSILON.sqrt();
        let a = Array2::from_diag(&array![0.5 * floor, 1.0, -2.0]);
        // B does not commute with A: a B-orthogonal projection is not the
        // Euclidean projection that defines the dense Moore--Penrose inverse.
        let b = array![[1.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 1.0]];
        for rhs in [
            array![1.0, 0.0, 0.0],
            array![1.0, 1.0, 0.5],
            array![floor, 1.0, 0.5],
        ] {
            let expected = array![0.0, rhs[1], -0.5 * rhs[2]];
            compare(&a, &b, &rhs, &expected);
        }
        // The last RHS used to evade the aggregate-Rayleigh detector entirely.
        let unfiltered = array![2.0, 1.0, -0.25];
        let quotient = unfiltered.dot(&a.dot(&unfiltered)) / unfiltered.dot(&b.dot(&unfiltered));
        assert!(quotient.abs() > floor);
    }

    #[test]
    fn both_routes_keep_resolved_modes_on_both_sides_of_zero_2828() {
        let floor = f64::EPSILON.sqrt();
        let a = Array2::from_diag(&array![0.5 * floor, 2.0 * floor, -2.0 * floor]);
        let b = Array2::eye(3);
        for rhs in [
            array![1.0, 0.0, 0.0],
            array![1.0, floor, floor],
            array![0.0, 2.0 * floor, -floor],
        ] {
            let expected = array![0.0, rhs[1] / (2.0 * floor), -rhs[2] / (2.0 * floor)];
            compare(&a, &b, &rhs, &expected);
        }
    }

    #[test]
    fn weak_band_excitation_survives_a_rotated_resolved_spectrum_2828() {
        let dim = 32;
        let floor = f64::EPSILON.sqrt();
        let v = Array1::from_iter((1..=dim).map(|i| i as f64));
        let rotation = Array2::eye(dim)
            - Array2::from_shape_fn((dim, dim), |(i, j)| 2.0 * v[i] * v[j] / v.dot(&v));
        let mut spectrum = Array1::from_iter((0..dim).map(|i| 1.0 + i as f64 / dim as f64));
        spectrum[0] = 0.5 * floor;
        spectrum[1] = 2.0 * floor;
        spectrum[2] = -2.0 * floor;
        let a = rotation
            .dot(&Array2::from_diag(&spectrum))
            .dot(&rotation.t());
        let b = Array2::from_shape_fn((dim, dim), |(i, j)| {
            if i == j {
                1.0
            } else if i.abs_diff(j) == 1 {
                0.125
            } else {
                0.0
            }
        });
        for excitation in [floor, 1.0e-10] {
            let mut coefficients = Array1::ones(dim);
            coefficients.slice_mut(s![..3]).fill(excitation);
            let rhs = rotation.dot(&coefficients);
            let mut response = &coefficients / &spectrum;
            response[0] = 0.0;
            compare(&a, &b, &rhs, &rotation.dot(&response));
        }
    }

    #[test]
    fn repeated_a_eigenspace_with_split_metric_scales_2828() {
        // A one-vector RHS Krylov space cannot identify the two different
        // rank decisions in this eigenspace. Keep this gate live while the
        // draft's repeated-eigenspace policy is completed.
        let a = Array2::from_diag(&array![1.0e-8, 1.0e-8]);
        let b = Array2::from_diag(&array![0.1, 10.0]);
        compare(&a, &b, &array![1.0, 1.0], &array![1.0e8, 0.0]);
    }

    #[test]
    fn entirely_discarded_band_has_zero_minimum_norm_response_2828() {
        let floor = f64::EPSILON.sqrt();
        let a = Array2::from_diag(&array![0.5 * floor, -0.25 * floor]);
        compare(&a, &Array2::eye(2), &array![1.0, -2.0], &Array1::zeros(2));
    }
}

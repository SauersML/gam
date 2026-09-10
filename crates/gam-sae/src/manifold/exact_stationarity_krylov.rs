// Included by construction_exact_hessian.rs in the construction module.

/// Resolve numerically repeated A eigenspaces in the B metric before applying
/// the direction-dependent rank rule. Without this step, an arbitrary rotation
/// of a repeated eigenspace can change which directions the rule retains.
///
/// The cluster envelope accounts for projection, its transpose product,
/// eigendecomposition, and lifting. Replacing a cluster by its mean changes A
/// only within that arithmetic envelope; distinct resolved eigenvalues keep
/// their eigenvectors. Both representations use this same convention.
fn canonicalize_exact_a_rank_clusters<B>(
    values: &mut Array1<f64>,
    vectors: &mut Array2<f64>,
    spectral_norm: f64,
    apply_b: &B,
) -> Result<(), String>
where
    B: Fn(&Array1<f64>) -> Result<Array1<f64>, String>,
{
    let n = vectors.nrows();
    let gamma = n as f64 * f64::EPSILON / (1.0 - n as f64 * f64::EPSILON);
    let envelope = 4.0 * gamma * spectral_norm;
    let mut start = 0;
    while start < values.len() {
        let mut end = start + 1;
        while end < values.len() && values[end] - values[start] <= envelope {
            end += 1;
        }
        if end - start > 1 {
            let block = vectors.slice(s![.., start..end]).to_owned();
            let width = end - start;
            let mut metric = Array2::zeros((width, width));
            for col in 0..width {
                let image = apply_b(&block.column(col).to_owned())?;
                if image.len() != n || image.iter().any(|x| !x.is_finite()) {
                    return Err("exact-A cluster metric returned an invalid vector".into());
                }
                for row in 0..=col {
                    let value = block.column(row).dot(&image);
                    metric[[row, col]] = value;
                    metric[[col, row]] = value;
                }
            }
            let (_, rotation) = metric
                .eigh(Side::Lower)
                .map_err(|e| format!("exact-A repeated-space metric decomposition: {e:?}"))?;
            vectors
                .slice_mut(s![.., start..end])
                .assign(&block.dot(&rotation));
            let anchor = values[start];
            let mean = anchor
                + values
                    .slice(s![start..end])
                    .iter()
                    .map(|value| value - anchor)
                    .sum::<f64>()
                    / width as f64;
            values.slice_mut(s![start..end]).fill(mean);
        }
        start = end;
    }
    Ok(())
}

/// Euclidean spectral pseudoinverse of A, with the dense route's rank rule.
/// A + sqrt(eps) B generates the trial space so repeated A eigenvalues do not
/// hide different metric directions. Only the projected physical A is inverted.
/// Physical A residuals and unresolved B directions drive further expansion.
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
    let flatten = |v: &SaeArrowVector| Array1::from_iter(v.t.iter().chain(v.beta.iter()).copied());
    let split = |v: &Array1<f64>| SaeArrowVector {
        t: v.slice(s![..total_t]).to_owned(),
        beta: v.slice(s![total_t..]).to_owned(),
    };
    let norm = |v: &Array1<f64>| v.iter().fold(0.0_f64, |n, &x| n.hypot(x));
    let flat_rhs = flatten(rhs);
    let rhs_norm = norm(&flat_rhs);
    if !rhs_norm.is_finite() {
        return Err("exact-stationarity Krylov solve: non-finite RHS".into());
    }
    if rhs_norm == 0.0 {
        return Ok(split(&flat_rhs));
    }
    let checked = |v: SaeArrowVector| -> Result<Array1<f64>, String> {
        if v.t.len() != total_t || v.beta.len() != dim - total_t {
            return Err("exact-stationarity operator changed vector dimensions".into());
        }
        let flat = flatten(&v);
        if flat.iter().any(|x| !x.is_finite()) {
            return Err("exact-stationarity operator returned a non-finite vector".into());
        }
        Ok(flat)
    };
    let a_flat = |v: &Array1<f64>| checked(apply_a(&split(v))?);
    let b_flat = |v: &Array1<f64>| checked(apply_b(&split(v))?);
    let tolerance = f64::EPSILON.sqrt();
    let gamma = dim as f64 * f64::EPSILON / (1.0 - dim as f64 * f64::EPSILON);
    let a_slice = |input: &[f64], output: &mut [f64]| -> Result<(), String> {
        let value = a_flat(&Array1::from_vec(input.to_vec()))?;
        for (slot, &value) in output.iter_mut().zip(value.iter()) {
            *slot = value;
        }
        Ok(())
    };
    let generate = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
        Ok(a_flat(v)? + tolerance * b_flat(v)?)
    };

    // Bound simultaneous bases, physical images, eigensystems and work vectors.
    let budget = crate::manifold::sae_host_in_core_budget_bytes().0 as u128;
    let storage = |steps: usize| {
        let (n, m) = (dim as u128, steps as u128);
        8 * (9 * n * m + 8 * m * m + 20 * n)
    };
    if storage(1) > budget {
        return Err(format!(
            "exact-stationarity Krylov solve: one direction needs {} bytes, budget {budget}",
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
    let mut state = 2828_u64;
    let norm_start = Array1::from_iter((0..dim).map(|_| {
        let bits = gam_linalg::utils::splitmix64(&mut state);
        (bits >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5
    }));
    let initial_a = norm(&a_flat(&norm_start)?) / norm(&norm_start);
    let initial_b = norm(&b_flat(&norm_start)?) / norm(&norm_start);
    let rhs_start = flat_rhs.to_vec();

    loop {
        let options = SymmetricLanczosOptions {
            max_steps: steps,
            residual_tol: gamma * initial_a,
            local_reorthogonalize: false,
            full_reorthogonalize: true,
        };
        let scale_pairs =
            symmetric_lanczos_eigenpairs(dim, &norm_start.to_vec(), options, &a_slice)?;
        let scale_index = (0..scale_pairs.eigenvalues.len())
            .max_by(|&a, &b| {
                scale_pairs.eigenvalues[a]
                    .abs()
                    .total_cmp(&scale_pairs.eigenvalues[b].abs())
            })
            .ok_or("exact-stationarity Krylov solve: empty norm probe")?;
        let mut spectral_norm = scale_pairs.eigenvalues[scale_index].abs();
        let scale_error = scale_pairs.residual_norm
            * scale_pairs.eigenvectors[[scale_pairs.eigenvalues.len() - 1, scale_index]].abs();
        let scale_resolved = scale_error <= tolerance * spectral_norm || scale_error == 0.0;
        drop(scale_pairs);
        let generator_options = SymmetricLanczosOptions {
            residual_tol: gamma * (spectral_norm.max(initial_a) + tolerance * initial_b),
            ..options
        };
        let generate_slice = |input: &[f64], output: &mut [f64]| -> Result<(), String> {
            let image = generate(&Array1::from_vec(input.to_vec()))?;
            for (slot, &value) in output.iter_mut().zip(image.iter()) {
                *slot = value;
            }
            Ok(())
        };
        let pairs = symmetric_lanczos_eigenpairs_with_original_vectors(
            dim,
            &rhs_start,
            generator_options,
            &generate_slice,
        )?;
        let mut basis = pairs
            .original_eigenvectors
            .ok_or("exact-stationarity Krylov solve: missing trial basis")?;
        let (last_residual, last_scale) = loop {
            let mut applied_basis = Array2::zeros(basis.raw_dim());
            for col in 0..basis.ncols() {
                applied_basis
                    .column_mut(col)
                    .assign(&a_flat(&basis.column(col).to_owned())?);
            }
            let mut projected = basis.t().dot(&applied_basis);
            for row in 0..projected.nrows() {
                for col in 0..row {
                    let value = 0.5 * (projected[[row, col]] + projected[[col, row]]);
                    projected[[row, col]] = value;
                    projected[[col, row]] = value;
                }
            }
            let (mut values, rotation) = projected
                .eigh(Side::Lower)
                .map_err(|e| format!("exact-stationarity projected A decomposition: {e:?}"))?;
            spectral_norm = values.iter().fold(spectral_norm, |s, &v| s.max(v.abs()));
            let mut vectors = basis.dot(&rotation);
            canonicalize_exact_a_rank_clusters(&mut values, &mut vectors, spectral_norm, &b_flat)?;
            let arithmetic = gamma * spectral_norm;
            let coefficients = vectors.t().dot(&flat_rhs);
            let mut projected_coefficients = coefficients.clone();
            let mut inverse_coefficients = Array1::zeros(values.len());
            let mut discarded = Vec::new();
            let mut classification_resolved = true;
            let mut seed: Option<Array1<f64>> = None;
            let mut worst_seed_excess = 0.0_f64;
            for index in 0..values.len() {
                let direction = vectors.column(index).to_owned();
                let a_image = a_flat(&direction)?;
                let b_image = b_flat(&direction)?;
                let metric = direction.dot(&b_image);
                if !(metric.is_finite() && metric > 0.0) {
                    return Err(format!(
                        "exact-stationarity Krylov solve: invalid v'Bv={metric}"
                    ));
                }
                let magnitude = values[index].abs();
                let floor = sae_exact_a_direction_floor(dim, spectral_norm, metric);
                let gap = (magnitude - floor).abs();
                let a_residual = a_image - values[index] * &direction;
                let a_error = norm(&a_residual);
                classification_resolved &= a_error <= gap + arithmetic;
                // Project away every represented direction, including other A
                // eigenvalues. Unrepresented B components can split an A
                // eigenspace's rank even when its A residual is exactly zero.
                let leakage = &b_image - &vectors.dot(&vectors.t().dot(&b_image));
                let leak_norm = norm(&leakage);
                let allowance = gap.max(gamma * norm(&b_image));
                let b_unresolved = magnitude > arithmetic && leak_norm > allowance;
                if b_unresolved {
                    classification_resolved = false;
                    let excess = leak_norm - allowance;
                    if excess > worst_seed_excess {
                        worst_seed_excess = excess;
                        seed = Some(leakage);
                    }
                } else if seed.is_none() && a_error > gamma * spectral_norm {
                    seed = Some(a_residual);
                }
                if magnitude > floor {
                    inverse_coefficients[index] = coefficients[index] / values[index];
                } else {
                    projected_coefficients[index] = 0.0;
                    discarded.push(index);
                }
            }
            let solution = vectors.dot(&inverse_coefficients);
            let projected_rhs = vectors.dot(&projected_coefficients);
            let residual = a_flat(&solution)? - &projected_rhs;
            let residual_norm = norm(&residual);
            let solution_norm = norm(&solution);
            let physical_scale = spectral_norm * solution_norm + norm(&projected_rhs);
            let normal_norm = norm(&a_flat(&residual)?);
            let solved_coefficients = vectors.t().dot(&solution);
            let null_mass = discarded
                .iter()
                .fold(0.0_f64, |n, &i| n.hypot(solved_coefficients[i]));
            // sqrt(eps) backward accuracy can miss an O(1) response to a
            // sqrt(eps)-sized excitation of a retained band mode. Require
            // machine-roundoff backward accuracy before accepting this inverse.
            if scale_resolved
                && classification_resolved
                && solution.iter().all(|x| x.is_finite())
                && residual_norm <= gamma * physical_scale
                && normal_norm <= gamma * spectral_norm * physical_scale
                && null_mass <= tolerance * solution_norm
            {
                return Ok(split(&solution));
            }
            if basis.ncols() >= steps {
                break (residual_norm, physical_scale);
            }
            let Some(mut seed) = seed else {
                break (residual_norm, physical_scale);
            };
            for _ in 0..2 {
                let correction = basis.dot(&basis.t().dot(&seed));
                seed -= &correction;
            }
            let seed_norm = norm(&seed);
            if !seed_norm.is_finite() || seed_norm == 0.0 {
                break (residual_norm, physical_scale);
            }
            seed /= seed_norm;
            let remaining = steps - basis.ncols();
            let projected_generate = |input: &[f64], output: &mut [f64]| -> Result<(), String> {
                let mut image = generate(&Array1::from_vec(input.to_vec()))?;
                for _ in 0..2 {
                    let correction = basis.dot(&basis.t().dot(&image));
                    image -= &correction;
                }
                for (slot, &value) in output.iter_mut().zip(image.iter()) {
                    *slot = value;
                }
                Ok(())
            };
            let extra = symmetric_lanczos_eigenpairs_with_original_vectors(
                dim,
                &seed.to_vec(),
                SymmetricLanczosOptions {
                    max_steps: remaining,
                    ..generator_options
                },
                &projected_generate,
            )?
            .original_eigenvectors
            .ok_or("exact-stationarity expansion lost its basis")?;
            // Reorthogonalize the joined basis as well as the projected
            // operator: neither a lucky breakdown nor a roundoff remainder is
            // permission to add a duplicate direction.
            let old_width = basis.ncols();
            let mut joined = Array2::zeros((dim, old_width + extra.ncols()));
            joined.slice_mut(s![.., ..old_width]).assign(&basis);
            let mut width = old_width;
            for col in 0..extra.ncols() {
                let mut vector = extra.column(col).to_owned();
                for _ in 0..2 {
                    let live = joined.slice(s![.., ..width]);
                    let correction = live.dot(&live.t().dot(&vector));
                    vector -= &correction;
                }
                let length = norm(&vector);
                if length > gamma {
                    joined.column_mut(width).assign(&(&vector / length));
                    width += 1;
                }
            }
            if width == old_width {
                break (residual_norm, physical_scale);
            }
            basis = joined.slice(s![.., ..width]).to_owned();
        };
        if steps == max_steps {
            return Err(format!(
                "exact-stationarity Krylov pseudoinverse did not certify in {steps} directions \
                 (dimension {dim}, budget {budget} bytes): residual {last_residual:.6e} / \
                 backward scale {last_scale:.6e}, scale resolved={scale_resolved}"
            ));
        }
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
        let (mut eigenvalues, mut eigenvectors) =
            a.eigh(Side::Lower).expect("symmetric test operator");
        let spectral_norm = eigenvalues.iter().fold(0.0_f64, |s, &v| s.max(v.abs()));
        canonicalize_exact_a_rank_clusters(
            &mut eigenvalues,
            &mut eigenvectors,
            spectral_norm,
            &|v| Ok(b.dot(v)),
        )
        .expect("canonical repeated-space rank policy");
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
        // A one-vector A-only Krylov space cannot identify the two different
        // rank decisions in this eigenspace; metric directions must be present.
        let a = Array2::from_diag(&array![1.0e-8, 1.0e-8]);
        let b = Array2::from_diag(&array![0.1, 10.0]);
        compare(&a, &b, &array![1.0, 1.0], &array![1.0e8, 0.0]);
    }

    #[test]
    fn repeated_a_eigenspace_resolves_rotated_metric_directions_2828() {
        let rotation = array![
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5]
        ];
        let a = 1.0e-8 * Array2::eye(4);
        let b = rotation
            .dot(&Array2::from_diag(&array![0.1, 0.2, 1.0, 10.0]))
            .dot(&rotation.t());
        compare(
            &a,
            &b,
            &rotation.dot(&Array1::ones(4)),
            &rotation.dot(&array![1.0e8, 1.0e8, 0.0, 0.0]),
        );
    }

    #[test]
    fn resolved_distinct_eigenvalues_are_not_rotated_by_the_metric_2828() {
        let mut values = array![1.0, 1.0 + 1.0e-9];
        let original = values.clone();
        let mut vectors = Array2::eye(2);
        let b = array![[2.0, 1.0], [1.0, 2.0]];
        canonicalize_exact_a_rank_clusters(&mut values, &mut vectors, original[1], &|v| {
            Ok(b.dot(v))
        })
        .expect("resolved spectrum");
        assert_eq!(values, original);
        assert_eq!(vectors, Array2::eye(2));
    }

    #[test]
    fn cancelled_generating_operator_still_resolves_physical_a_2828() {
        let a = Array2::from_diag(&array![0.875, 0.25, -1.0]);
        let b = (Array2::eye(3) - &a) / f64::EPSILON.sqrt();
        // A + sqrt(eps) B is I. Its one-dimensional RHS Krylov space must
        // expand from unresolved physical-A/B directions before accepting.
        compare(
            &a,
            &b,
            &array![1.0, 1.0, 1.0],
            &array![1.0 / 0.875, 0.0, 0.0],
        );
    }

    #[test]
    fn entirely_discarded_band_has_zero_minimum_norm_response_2828() {
        let floor = f64::EPSILON.sqrt();
        let a = Array2::from_diag(&array![0.5 * floor, -0.25 * floor]);
        compare(&a, &Array2::eye(2), &array![1.0, -2.0], &Array1::zeros(2));
    }
}

// Included by construction_exact_hessian.rs in the construction module.

/// Resolve numerically repeated pencil eigenspaces against the substitution operator
/// before applying the direction-dependent band edge (#2267, #2933 F07). Its owner is
/// gam-solve's [`gam_solve::arrow_schur::canonicalize_exact_a_rank_clusters`], the one
/// convention the dense, Krylov and reduced exact-A routes apply.
fn canonicalize_exact_a_rank_clusters<B>(
    values: &mut Array1<f64>,
    vectors: &mut Array2<f64>,
    spectral_norm: f64,
    apply_edge: &B,
) -> Result<(), String>
where
    B: Fn(&Array1<f64>) -> Result<Array1<f64>, String>,
{
    gam_solve::arrow_schur::canonicalize_exact_a_rank_clusters(
        values,
        vectors,
        spectral_norm,
        apply_edge,
    )
}

/// Covariant spectral pseudoinverse of the pencil `(A, Φ)`, with the dense route's band
/// (#2933 F07). `A + √ε Φ` generates the trial space so repeated `A` eigenvalues do not
/// hide different metric directions. Rayleigh--Ritz on the projected pencil
/// `(VᵀAV, VᵀΦV)` gives `Φ`-orthonormal Ritz pairs `(μᵢ, wᵢ)`, and only their retained
/// span is inverted. Physical residuals and unresolved Ritz pairs drive further expansion.
/// `apply_b` is the metric `Φ`; `apply_b_raw` is the physical majorizer `Φ` conditions.
/// The two agree off the evidence factor's pins; where they differ the positive band edge
/// rises to the stiffness `Φ` substituted, as on the dense route (#2267).
///
/// Each operator is applied in the coordinates it acts on, so its images are priced at
/// their dimension; see [`solve_exact_stationarity_krylov_with_rounding`].
fn solve_exact_stationarity_krylov<A, B, R>(
    rhs: &SaeArrowVector,
    apply_a: &A,
    apply_b: &B,
    apply_b_raw: &R,
) -> Result<SaeArrowVector, String>
where
    A: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    B: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    R: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    let dim = rhs.t.len() + rhs.beta.len();
    solve_exact_stationarity_krylov_with_rounding(rhs, apply_a, apply_b, apply_b_raw, dim)
}

/// [`solve_exact_stationarity_krylov`] with the operators' images priced at
/// `operator_terms` rounded terms per entry.
///
/// Every operator-denominated bar is the accumulation band `γ` of that many terms: the
/// Lanczos stopping tolerances, the Ritz residual that seeds an expansion, and the
/// physical and dual residual bars that certify the solve. An operator applied in its own
/// coordinates is priced at their dimension. One pulled back through dense lifts, such as
/// `Tᵀ·A·T + E` on a learned frame's tangent coordinates (#2933 F39), accumulates the
/// lifts' inner sums too: the dimension `A` is applied in, plus each lift's inner
/// dimension. Priced at the trial space's dimension instead, a solve whose every Ritz pair
/// is resolved can fail its dual bar by the lifts' own rounding and refuse an exact
/// answer. Orthogonalizing the trial basis stays denominated in the trial space's own
/// dimension.
fn solve_exact_stationarity_krylov_with_rounding<A, B, R>(
    rhs: &SaeArrowVector,
    apply_a: &A,
    apply_b: &B,
    apply_b_raw: &R,
    operator_terms: usize,
) -> Result<SaeArrowVector, String>
where
    A: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    B: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    R: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    use gam_linalg::faer_ndarray::FaerCholesky;
    use gam_linalg::lanczos::{
        SymmetricLanczosOptions, symmetric_lanczos_eigenpairs,
        symmetric_lanczos_eigenpairs_with_original_vectors,
    };
    use gam_linalg::triangular::{
        back_substitution_lower_transpose, forward_substitution_lower_vector,
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
    let b_raw_flat = |v: &Array1<f64>| checked(apply_b_raw(&split(v))?);
    let tolerance = f64::EPSILON.sqrt();
    if operator_terms < dim {
        return Err(format!(
            "exact-stationarity Krylov solve: operator images priced at {operator_terms} terms, \
             under the dimension {dim} they are applied in"
        ));
    }
    let band = |terms: usize| terms as f64 * f64::EPSILON / (1.0 - terms as f64 * f64::EPSILON);
    let gamma = band(dim);
    let operator_gamma = band(operator_terms);
    let a_slice = |input: &[f64], output: &mut [f64]| -> Result<(), String> {
        let value = a_flat(&Array1::from_vec(input.to_vec()))?;
        for (slot, &value) in output.iter_mut().zip(value.iter()) {
            *slot = value;
        }
        Ok(())
    };
    let b_slice = |input: &[f64], output: &mut [f64]| -> Result<(), String> {
        let value = b_flat(&Array1::from_vec(input.to_vec()))?;
        for (slot, &value) in output.iter_mut().zip(value.iter()) {
            *slot = value;
        }
        Ok(())
    };
    let generate = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
        Ok(a_flat(v)? + tolerance * b_flat(v)?)
    };

    // Bound simultaneous bases, physical and metric images, projected pencils,
    // eigensystems and work vectors.
    let budget = crate::manifold::sae_host_in_core_budget_bytes().0 as u128;
    let storage = |steps: usize| {
        let (n, m) = (dim as u128, steps as u128);
        8 * (12 * n * m + 12 * m * m + 24 * n)
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
            residual_tol: operator_gamma * initial_a,
            local_reorthogonalize: false,
            full_reorthogonalize: true,
        };
        // The spectral scales of both operands: every Ritz value's numerical resolution is
        // denominated in them.
        let largest = |pairs: &gam_linalg::lanczos::SymmetricLanczosEigenpairs| {
            let index = (0..pairs.eigenvalues.len())
                .max_by(|&a, &b| pairs.eigenvalues[a].abs().total_cmp(&pairs.eigenvalues[b].abs()))?;
            let value = pairs.eigenvalues[index].abs();
            let error = pairs.residual_norm
                * pairs.eigenvectors[[pairs.eigenvalues.len() - 1, index]].abs();
            Some((value, error <= tolerance * value || error == 0.0))
        };
        let scale_pairs =
            symmetric_lanczos_eigenpairs(dim, &norm_start.to_vec(), options, &a_slice)?;
        let (mut spectral_norm, operator_resolved) =
            largest(&scale_pairs).ok_or("exact-stationarity Krylov solve: empty norm probe")?;
        drop(scale_pairs);
        let metric_options = SymmetricLanczosOptions {
            residual_tol: operator_gamma * initial_b,
            ..options
        };
        let metric_pairs =
            symmetric_lanczos_eigenpairs(dim, &norm_start.to_vec(), metric_options, &b_slice)?;
        let (mut metric_norm, metric_resolved) =
            largest(&metric_pairs).ok_or("exact-stationarity Krylov solve: empty metric probe")?;
        drop(metric_pairs);
        let scale_resolved = operator_resolved && metric_resolved;
        let generator_options = SymmetricLanczosOptions {
            residual_tol: operator_gamma * (spectral_norm.max(initial_a) + tolerance * metric_norm.max(initial_b)),
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
            let width = basis.ncols();
            let mut applied_basis = Array2::zeros(basis.raw_dim());
            let mut metric_basis = Array2::zeros(basis.raw_dim());
            for col in 0..width {
                let column = basis.column(col).to_owned();
                applied_basis.column_mut(col).assign(&a_flat(&column)?);
                metric_basis.column_mut(col).assign(&b_flat(&column)?);
            }
            let symmetrized = |mut projected: Array2<f64>| {
                for row in 0..projected.nrows() {
                    for col in 0..row {
                        let value = 0.5 * (projected[[row, col]] + projected[[col, row]]);
                        projected[[row, col]] = value;
                        projected[[col, row]] = value;
                    }
                }
                projected
            };
            let projected_a = symmetrized(basis.t().dot(&applied_basis));
            let projected_b = symmetrized(basis.t().dot(&metric_basis));
            drop(applied_basis);
            drop(metric_basis);
            let lower = projected_b
                .cholesky(Side::Lower)
                .map_err(|e| format!("exact-stationarity projected metric is not positive definite: {e:?}"))?
                .lower_triangular();
            // `L⁻¹ VᵀAV L⁻ᵀ` for `VᵀΦV = LLᵀ`.
            let mut half = Array2::<f64>::zeros((width, width));
            for col in 0..width {
                half.column_mut(col)
                    .assign(&forward_substitution_lower_vector(&lower, projected_a.column(col)));
            }
            let mut whitened = Array2::<f64>::zeros((width, width));
            for col in 0..width {
                whitened
                    .column_mut(col)
                    .assign(&forward_substitution_lower_vector(&lower, half.row(col)));
            }
            let (mut values, rotation) = symmetrized(whitened)
                .eigh(Side::Lower)
                .map_err(|e| format!("exact-stationarity projected pencil decomposition: {e:?}"))?;
            let mut coordinates = Array2::<f64>::zeros((width, width));
            for col in 0..width {
                coordinates
                    .column_mut(col)
                    .assign(&back_substitution_lower_transpose(&lower, rotation.column(col)));
            }
            let mut vectors = basis.dot(&coordinates);
            let curvature_norm = values.iter().fold(0.0_f64, |s, &v| s.max(v.abs()));
            canonicalize_exact_a_rank_clusters(&mut values, &mut vectors, curvature_norm, &|v| {
                Ok(b_flat(v)? - b_raw_flat(v)?)
            })?;
            let coefficients = vectors.t().dot(&flat_rhs);
            let mut inverse_coefficients = Array1::zeros(values.len());
            let mut band_images: Vec<Array1<f64>> = Vec::new();
            let mut band_coefficients: Vec<f64> = Vec::new();
            let mut classification_resolved = true;
            let mut seed: Option<Array1<f64>> = None;
            let mut worst_seed_excess = 0.0_f64;
            // `ΦVc` and `‖ΦV‖²_F`, for the basis's `Φ`-orthonormality defect below.
            let mut metric_image_of_coefficients = Array1::<f64>::zeros(dim);
            let mut metric_images_frobenius_sq = 0.0_f64;
            for index in 0..values.len() {
                let direction = vectors.column(index).to_owned();
                let a_image = a_flat(&direction)?;
                let b_image = b_flat(&direction)?;
                metric_image_of_coefficients.scaled_add(coefficients[index], &b_image);
                metric_images_frobenius_sq += b_image.dot(&b_image);
                let metric = direction.dot(&b_image);
                if !(metric.is_finite() && metric > 0.0) {
                    return Err(format!(
                        "exact-stationarity Krylov solve: invalid w'Φw={metric}"
                    ));
                }
                spectral_norm = spectral_norm.max(norm(&a_image) / norm(&direction));
                metric_norm = metric_norm.max(norm(&b_image) / norm(&direction));
                let substituted = (metric - direction.dot(&b_raw_flat(&direction)?)).max(0.0);
                let norm_sq = direction.dot(&direction);
                let resolution = sae_exact_a_pencil_resolution(
                    dim,
                    norm_sq,
                    spectral_norm,
                    metric_norm,
                    values[index],
                );
                let floor = sae_exact_a_band_edge(values[index], resolution, substituted);
                let magnitude = values[index].abs();
                let gap = (magnitude - floor).abs();
                let ritz_residual = &a_image - &(values[index] * &b_image);
                // To first order the Ritz value lies within `‖r‖₂‖w‖₂` of a pencil
                // eigenvalue, so its side of the band edge is decided only when that error
                // is under the gap to the edge.
                let ritz_error = norm(&ritz_residual) * norm_sq.sqrt();
                if ritz_error > gap + resolution {
                    classification_resolved = false;
                    let excess = ritz_error - gap - resolution;
                    if excess > worst_seed_excess {
                        worst_seed_excess = excess;
                        seed = Some(ritz_residual);
                    }
                } else if seed.is_none() && norm(&ritz_residual) > operator_gamma * spectral_norm * norm_sq.sqrt() {
                    seed = Some(ritz_residual);
                }
                if magnitude > floor {
                    inverse_coefficients[index] = coefficients[index] / values[index];
                } else {
                    band_images.push(b_image);
                    band_coefficients.push(coefficients[index]);
                }
            }
            let solution = vectors.dot(&inverse_coefficients);
            let mut projected_rhs = flat_rhs.clone();
            for (image, &coefficient) in band_images.iter().zip(band_coefficients.iter()) {
                projected_rhs.scaled_add(-coefficient, image);
            }
            let residual = a_flat(&solution)? - &projected_rhs;
            let residual_norm = norm(&residual);
            let solution_norm = norm(&solution);
            let solution_metric_norm = norm(&inverse_coefficients);
            // Removing the band's dual components rounds at the scale of the right-hand side
            // and of what it removes, as on the dense route.
            let physical_scale = spectral_norm * solution_norm
                + rhs_norm
                + norm(&(&flat_rhs - &projected_rhs))
                + norm(&projected_rhs);
            let dual_norm = norm(&vectors.t().dot(&residual));
            let dual_scale = curvature_norm * solution_metric_norm + norm(&coefficients);
            // The Ritz vectors are `Φ`-orthonormal only to the Rayleigh--Ritz arithmetic,
            // `VᵀΦV = I + E` with `‖E‖` of order `κ(VᵀΦV)·ε`. On a basis that spans the space,
            // `ΦV(VᵀΦV)⁻¹Vᵀ = I`, so even the exact pseudoinverse on this basis leaves the dual
            // residual `Ec` and the physical residual `ΦVEc`, to first order in `E`. That floor
            // belongs to the basis, not to the solve's convergence, and `Ec = Vᵀ(ΦVc) − c` is
            // measured here. At #2828 item 2's in-band state, `κ(VᵀΦV)` reached `1.3e9` and
            // `‖Ec‖ = 1.4e-13` against the `γ` bar of `1.3e-14`, so a converged solve was refused
            // (job 1273989). A basis that misses directions still leaves `(I − ΦVVᵀ)rhs`, which
            // this floor does not cover, and a defect above `√ε` of the coefficients is no
            // basis to certify on.
            let gram_defect = norm(&(vectors.t().dot(&metric_image_of_coefficients) - &coefficients));
            let physical_gram_defect = metric_images_frobenius_sq.sqrt() * gram_defect;
            let band_mass = band_images
                .iter()
                .fold(0.0_f64, |n, image| n.hypot(image.dot(&solution)));
            // sqrt(eps) backward accuracy can miss an O(1) response to a
            // sqrt(eps)-sized excitation of a retained band mode. Require
            // machine-roundoff backward accuracy before accepting this inverse.
            if scale_resolved
                && classification_resolved
                && solution.iter().all(|x| x.is_finite())
                && gram_defect <= tolerance * norm(&coefficients)
                && residual_norm <= operator_gamma * physical_scale + physical_gram_defect
                && dual_norm <= operator_gamma * dual_scale + gram_defect
                && band_mass <= tolerance * solution_metric_norm
            {
                return Ok(split(&solution));
            }
            if basis.ncols() >= steps {
                break (residual_norm, physical_scale);
            }
            let Some(mut seed) = seed.or(Some(residual)) else {
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

    /// `L⁻ᵀx` for a lower-triangular `L`, written out here so the expected responses do
    /// not borrow the production triangular kernels.
    fn lower_transpose_solve(lower: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
        let n = lower.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut sum = x[i];
            for k in (i + 1)..n {
                sum -= lower[[k, i]] * out[k];
            }
            out[i] = sum / lower[[i, i]];
        }
        out
    }

    /// The pencil `A = LQ diag(μ) QᵀLᵀ` in the metric `Φ = LLᵀ`. Its eigenvectors are
    /// `W = L⁻ᵀQ`, so a right-hand side `rhs = LQc` has dual coefficients `Wᵀrhs = c` and
    /// the covariant pseudoinverse response is `L⁻ᵀQ(c ⊘ μ)` on the retained directions.
    fn pencil(lower: &Array2<f64>, rotation: &Array2<f64>, mu: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        let frame = lower.dot(rotation);
        let a = frame.dot(&Array2::from_diag(mu)).dot(&frame.t());
        let a = (&a + &a.t()) * 0.5;
        (a, lower.dot(&lower.t()))
    }

    fn compare(
        a: &Array2<f64>,
        b: &Array2<f64>,
        b_raw: &Array2<f64>,
        rhs: &Array1<f64>,
        expected: &Array1<f64>,
    ) {
        let split = |v: &Array1<f64>| SaeArrowVector {
            t: v.slice(s![..v.len() - 1]).to_owned(),
            beta: v.slice(s![v.len() - 1..]).to_owned(),
        };
        let flatten =
            |v: &SaeArrowVector| Array1::from_iter(v.t.iter().chain(v.beta.iter()).copied());
        let metric = super::tests_pencil_classification_2933::DensePencilMetric::new(b.clone(), b_raw)
            .expect("positive-definite test metric");
        let dense = SaeManifoldTerm::exact_hessian_spectral_block(a.clone(), &metric)
            .expect("dense pencil geometry")
            .solve_stationarity(&split(rhs))
            .expect("dense truncated pseudoinverse")
            .step;
        let apply_a = |v: &SaeArrowVector| Ok(split(&a.dot(&flatten(v))));
        let apply_b = |v: &SaeArrowVector| Ok(split(&b.dot(&flatten(v))));
        let apply_b_raw = |v: &SaeArrowVector| Ok(split(&b_raw.dot(&flatten(v))));
        let matrix_free =
            solve_exact_stationarity_krylov(&split(rhs), &apply_a, &apply_b, &apply_b_raw)
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
        // `Φ` does not commute with `A`: the band's dual components are removed along
        // `ΦW_Z`, not along a Euclidean projection.
        let lower = array![[1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [0.0, 0.25, 1.0]];
        let rotation = Array2::<f64>::eye(3);
        let mu = array![0.5 * floor, 1.0, -2.0];
        let (a, b) = pencil(&lower, &rotation, &mu);
        for c in [
            array![1.0, 0.0, 0.0],
            array![1.0, 1.0, 0.5],
            array![floor, 1.0, 0.5],
        ] {
            let rhs = lower.dot(&rotation.dot(&c));
            let response = array![0.0, c[1], -0.5 * c[2]];
            let expected = lower_transpose_solve(&lower, &rotation.dot(&response));
            compare(&a, &b, &b, &rhs, &expected);
        }
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
            compare(&a, &b, &b, &rhs, &expected);
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
        let lower = Array2::from_shape_fn((dim, dim), |(i, j)| {
            if i == j {
                1.0
            } else if i == j + 1 {
                0.125
            } else {
                0.0
            }
        });
        let (a, b) = pencil(&lower, &rotation, &spectrum);
        for excitation in [floor, 1.0e-10] {
            let mut coefficients = Array1::ones(dim);
            coefficients.slice_mut(s![..3]).fill(excitation);
            let rhs = lower.dot(&rotation.dot(&coefficients));
            let mut response = &coefficients / &spectrum;
            response[0] = 0.0;
            let expected = lower_transpose_solve(&lower, &rotation.dot(&response));
            compare(&a, &b, &b, &rhs, &expected);
        }
    }

    #[test]
    fn repeated_a_eigenspace_with_split_metric_scales_2828() {
        // `A`'s eigenspace is repeated, but the pencil curvatures `λ/b` are not: one
        // direction clears the band and the other does not. A one-vector A-only Krylov space
        // cannot identify the two decisions; metric directions must be present.
        let a = Array2::from_diag(&array![1.0e-8, 1.0e-8]);
        let b = Array2::from_diag(&array![0.1, 10.0]);
        compare(&a, &b, &b, &array![1.0, 1.0], &array![1.0e8, 0.0]);
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
        let edge = array![[2.0, 1.0], [1.0, 2.0]];
        canonicalize_exact_a_rank_clusters(&mut values, &mut vectors, original[1], &|v| {
            Ok(edge.dot(v))
        })
        .expect("resolved spectrum");
        assert_eq!(values, original);
        assert_eq!(vectors, Array2::<f64>::eye(2));
    }

    #[test]
    fn cancelled_generating_operator_still_resolves_physical_a_2828() {
        let a = Array2::from_diag(&array![0.875, 0.25, -1.0]);
        let b = (Array2::eye(3) - &a) / f64::EPSILON.sqrt();
        // A + sqrt(eps) B is I. Its one-dimensional RHS Krylov space must expand from
        // unresolved physical Ritz pairs before accepting. The pencil curvatures
        // `(7√ε, √ε/3, −√ε/2)` retain only the first direction.
        compare(
            &a,
            &b,
            &b,
            &array![1.0, 1.0, 1.0],
            &array![1.0 / 0.875, 0.0, 0.0],
        );
    }

    #[test]
    fn entirely_discarded_band_has_zero_minimum_norm_response_2828() {
        let floor = f64::EPSILON.sqrt();
        let a = Array2::from_diag(&array![0.5 * floor, -0.25 * floor]);
        let b = Array2::eye(2);
        compare(&a, &b, &b, &array![1.0, -2.0], &Array1::zeros(2));
    }

    #[test]
    fn a_direction_only_substituted_stiffness_resolves_is_in_band_on_both_routes_2267() {
        // `Φ` carries unit stiffness on directions 0 and 2 that `B_raw` does not: the
        // evidence factor substituted it. Direction 0 sits at 2·√ε, above the pencil floor
        // but under its substituted stiffness 1, so both routes drop it. Direction 2 has
        // the same curvature with the opposite sign and stays resolved, because the
        // negative side keeps the bare floor.
        let floor = f64::EPSILON.sqrt();
        let a = Array2::from_diag(&array![2.0 * floor, 1.0, -2.0 * floor]);
        let b = Array2::eye(3);
        let b_raw = Array2::from_diag(&array![0.0, 1.0, 0.0]);
        let rhs = array![floor, 1.0, floor];
        compare(&a, &b, &b_raw, &rhs, &array![0.0, 1.0, -0.5]);
        // Nothing substituted: the bare floor keeps direction 0.
        compare(&a, &b, &b, &rhs, &array![0.5, 1.0, -0.5]);
    }
}

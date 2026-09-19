// #2234 — the declared compact chart orbit of one dense exact-`A` evaluation, integrated exactly.
//
// Included into `construction.rs` beside `construction_exact_hessian.rs`, whose spectral block
// it extends.
//
// Laplace prices the orbit tangent `τ` with the chord curvature `τᵀAτ`. At an inner root
// accepted at tolerance the dropped connection term `∇f·γ″` dominates the intrinsic orbit
// curvature, so the chord reads negative and the evidence refuses a state whose exact orbit sits
// at its ARD minimum (#2234 stall pin; outer2080's sd1 and acc13). The evaluation instead
// integrates the orbit coordinate exactly (`compact_orbit.rs`) and prices the complement in the
// pencil, through the stiffened operator
//
//   A_s = P_ΦᵀAP_Φ + ΦTN⁻¹TᵀΦ,   N = TᵀΦT,   P_Φ = I − TN⁻¹TᵀΦ.
//
// In the basis `[T, Q]` with `QᵀΦT = 0` it is `diag(N, QᵀAQ)`: the pencil `(A_s, Φ)` carries
// `μ = 1` on the orbit and the complement's pencil elsewhere, so the orbit is never classified.
//
// Every consumer of the evaluation's inverse still needs the exact `A⁺`, the implicit-function
// response of the inner root. Eliminating the orbit coordinate gives it from the same single
// decomposition (#2267's pin):
//
//   G_⊥ = P_Φ·A_s⁺·P_Φᵀ,   V = T − G_⊥AT,   s = TᵀAT − (AT)ᵀG_⊥(AT),   A⁺ = G_⊥ + V s⁻¹ Vᵀ,
//
// with `s` diagonalized in `N`'s metric and each orbit curvature classified by the same band edge
// every pencil direction is classified by.

use super::compact_orbit::{
    CircleOrbitGenerator, CircleOrbitIntegral, CompactOrbitLaplaceReason, CompactOrbitPricing,
};

#[cfg(test)]
#[path = "tests_orbit_elimination_2234.rs"]
mod tests_orbit_elimination_2234;

#[cfg(test)]
#[path = "tests_orbit_gradient_2234.rs"]
mod tests_orbit_gradient_2234;

/// The orbit images a stiffened operator was built from, before its decomposition.
pub(crate) struct OrbitStiffening {
    /// `T`, one declared tangent per closure-certified circle orbit.
    tangents: Array2<f64>,
    /// `ΦT`.
    metric_images: Array2<f64>,
    /// `AT` on the unstiffened operator.
    operator_images: Array2<f64>,
    /// `N = TᵀΦT`.
    gram: Array2<f64>,
    /// `N⁻¹`.
    gram_inverse: Array2<f64>,
    /// `‖A‖_F` of the unstiffened operator, the scale an orbit curvature resolves against.
    operator_frobenius: f64,
}

/// The exact-`A` pseudo-inverse beside a block whose spectrum prices `A_s`.
pub(crate) struct OrbitElimination {
    stiffening: OrbitStiffening,
    /// `TᵀAT + N`, the stiffening's middle factor `A − A_s` is rebuilt from.
    stiffening_middle: Array2<f64>,
    /// `σ`: `s·u = σ·N·u` for the orbit Schur complement, ascending.
    curvatures: Array1<f64>,
    /// `T·U` with `UᵀNU = I`: the `Φ`-orthonormal orbit directions.
    directions: Array2<f64>,
    /// `V·U`: the response columns `A⁺` carries along each orbit direction.
    response_directions: Array2<f64>,
    /// `ΦTU`: each orbit direction's metric image, the dual component a band removes.
    direction_metric_images: Array2<f64>,
    /// Band edge of every `σ`, from [`sae_exact_a_band_edge`].
    edges: Array1<f64>,
}

/// `f(M)` for a symmetric positive-definite `k×k` matrix, off its eigensystem.
fn symmetric_positive_function(
    matrix: &Array2<f64>,
    label: &'static str,
    function: impl Fn(f64) -> f64,
) -> Result<Array2<f64>, String> {
    let (values, vectors) = matrix
        .eigh(Side::Lower)
        .map_err(|error| format!("{label}: eigendecomposition failed: {error:?}"))?;
    if values.iter().any(|&value| !(value.is_finite() && value > 0.0)) {
        return Err(format!(
            "{label}: the orbit tangents' metric Gram is not positive definite, spectrum {values:?}"
        ));
    }
    Ok(vectors
        .dot(&Array2::from_diag(&values.mapv(function)))
        .dot(&vectors.t()))
}

impl SaeManifoldTerm {
    /// Every atom's orbit pricing at `cache`, with the orbits that share a connected block of `A`
    /// or `Φ` kept on Laplace ([`Self::separate_coupled_compact_orbits`]). Both evaluation routes
    /// read their orbit set here, so the arrow route refuses exactly the states whose dense
    /// evaluation integrates an orbit. Two or more certified orbits are first separated by the
    /// row blocks and mass carriers alone, which already couple every atom a dense row holds;
    /// only when an orbit survives that pass are the border columns probed: `A`'s through the one
    /// exact-Hessian apply the dense materialization's probes run, and `Φ`'s through the joint
    /// metric the dense block is classified in.
    pub(crate) fn separated_compact_orbit_pricing(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<CompactOrbitPricing>, String> {
        let pricings = self.compact_orbit_pricing(rho, cache)?;
        let certified = pricings
            .iter()
            .filter(|pricing| matches!(pricing, CompactOrbitPricing::ExactCircle(_)))
            .count();
        if certified < 2 {
            return Ok(pricings);
        }
        let carriers = self.ordered_mass_hessian_carriers(rho, cache)?;
        let by_rows = Self::separate_coupled_compact_orbits(
            pricings.clone(),
            &cache.row_offsets,
            cache.k,
            &carriers,
            None,
        )?;
        if !by_rows
            .iter()
            .any(|pricing| matches!(pricing, CompactOrbitPricing::ExactCircle(_)))
        {
            return Ok(by_rows);
        }
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
        let residual = self.prepare_residual_curvature_rows(target, cache)?;
        let metric = ArrowMetric::Joint(cache).prepare()?;
        let mut operator_columns = Array2::<f64>::zeros((dim, k));
        let mut metric_columns = Array2::<f64>::zeros((dim, k));
        for column in 0..k {
            let mut unit = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(k),
            };
            unit.beta[column] = 1.0;
            let image = self.apply_exact_hessian_prepared(rho, cache, &unit, &prepared, &residual)?;
            operator_columns
                .slice_mut(s![..total_t, column])
                .assign(&image.t);
            operator_columns
                .slice_mut(s![total_t.., column])
                .assign(&image.beta);
            let mut flat = Array1::<f64>::zeros(dim);
            flat[total_t + column] = 1.0;
            metric_columns
                .column_mut(column)
                .assign(&metric.apply(flat.view())?);
        }
        Self::separate_coupled_compact_orbits(
            pricings,
            &cache.row_offsets,
            cache.k,
            &carriers,
            Some((operator_columns.view(), metric_columns.view())),
        )
    }

    /// `A_s = A − ΦTN⁻¹(AT)ᵀ − ATN⁻¹(ΦT)ᵀ + ΦTN⁻¹(TᵀAT + N)N⁻¹(ΦT)ᵀ`, which in `[T, Q]` coordinates
    /// is `diag(N, QᵀAQ)`, and the images it was built from. With no tangent the operator is
    /// returned unchanged.
    pub(crate) fn stiffen_compact_orbits(
        a: Array2<f64>,
        tangents: Array2<f64>,
        metric: &dyn ExactAPencilMetric,
    ) -> Result<(Array2<f64>, Option<OrbitStiffening>), String> {
        let dim = a.nrows();
        let k = tangents.ncols();
        if k == 0 {
            return Ok((a, None));
        }
        if a.ncols() != dim || tangents.nrows() != dim || metric.dim() != dim {
            return Err(format!(
                "stiffen_compact_orbits: operator {:?}, tangents {:?}, metric dimension {}",
                a.dim(),
                tangents.dim(),
                metric.dim()
            ));
        }
        let operator_frobenius = a.iter().map(|value| value * value).sum::<f64>().sqrt();
        let mut metric_images = Array2::<f64>::zeros((dim, k));
        for column in 0..k {
            metric_images
                .column_mut(column)
                .assign(&metric.apply(tangents.column(column))?);
        }
        let operator_images = a.dot(&tangents);
        let mut gram = tangents.t().dot(&metric_images);
        let chord = tangents.t().dot(&operator_images);
        for row in 0..k {
            for column in (row + 1)..k {
                let average = 0.5 * (gram[[row, column]] + gram[[column, row]]);
                gram[[row, column]] = average;
                gram[[column, row]] = average;
            }
        }
        let gram_inverse =
            symmetric_positive_function(&gram, "stiffen_compact_orbits", f64::recip)?;
        let projector = metric_images.dot(&gram_inverse);
        let middle = &chord + &gram;
        let mut stiffened = a;
        stiffened -= &projector.dot(&operator_images.t());
        stiffened -= &operator_images.dot(&projector.t());
        stiffened += &projector.dot(&middle).dot(&projector.t());
        for row in 0..dim {
            for column in (row + 1)..dim {
                let average = 0.5 * (stiffened[[row, column]] + stiffened[[column, row]]);
                stiffened[[row, column]] = average;
                stiffened[[column, row]] = average;
            }
        }
        Ok((
            stiffened,
            Some(OrbitStiffening {
                tangents,
                metric_images,
                operator_images,
                gram,
                gram_inverse,
                operator_frobenius,
            }),
        ))
    }
}

impl OrbitStiffening {
    /// `P_Φᵀx = x − ΦTN⁻¹Tᵀx`.
    fn project_dual(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        let coefficients = self.gram_inverse.dot(&self.tangents.t().dot(&x));
        &x - &self.metric_images.dot(&coefficients)
    }

    /// `P_Φy = y − TN⁻¹(ΦT)ᵀy`.
    fn project_primal(&self, y: ArrayView1<'_, f64>) -> Array1<f64> {
        let coefficients = self.gram_inverse.dot(&self.metric_images.t().dot(&y));
        &y - &self.tangents.dot(&coefficients)
    }

    /// Eliminate the orbit coordinates against the stiffened block's resolved complement.
    pub(crate) fn eliminate(
        self,
        block: &ExactHessianSpectralBlock,
        metric: &dyn ExactAPencilMetric,
    ) -> Result<OrbitElimination, String> {
        let dim = block.eigenvalues.len();
        let k = self.tangents.ncols();
        let mut complement_response = Array2::<f64>::zeros((dim, k));
        for column in 0..k {
            let dual = self.project_dual(self.operator_images.column(column));
            let solved = block.apply_stiffened_pseudo_inverse(dual.view());
            complement_response
                .column_mut(column)
                .assign(&self.project_primal(solved.view()));
        }
        let response = &self.tangents - &complement_response;
        let mut schur = self.tangents.t().dot(&self.operator_images)
            - self.operator_images.t().dot(&complement_response);
        for row in 0..k {
            for column in (row + 1)..k {
                let average = 0.5 * (schur[[row, column]] + schur[[column, row]]);
                schur[[row, column]] = average;
                schur[[column, row]] = average;
            }
        }
        let gram_inverse_root =
            symmetric_positive_function(&self.gram, "orbit elimination", |value| value.sqrt().recip())?;
        let whitened = gram_inverse_root.dot(&schur).dot(&gram_inverse_root);
        let (curvatures, whitened_vectors) = whitened
            .eigh(Side::Lower)
            .map_err(|error| format!("orbit elimination: Schur eigendecomposition failed: {error:?}"))?;
        let coordinates = gram_inverse_root.dot(&whitened_vectors);
        let directions = self.tangents.dot(&coordinates);
        let response_directions = response.dot(&coordinates);
        let direction_metric_images = self.metric_images.dot(&coordinates);
        let mut edges = Array1::<f64>::zeros(k);
        for index in 0..k {
            let direction = directions.column(index);
            let resolution = sae_exact_a_pencil_resolution(
                dim,
                direction.dot(&direction),
                self.operator_frobenius,
                block.metric_frobenius,
                curvatures[index],
            );
            let substituted = direction
                .dot(&metric.substituted_image(direction)?)
                .max(0.0);
            edges[index] = sae_exact_a_band_edge(curvatures[index], resolution, substituted);
        }
        let stiffening_middle = self.tangents.t().dot(&self.operator_images) + &self.gram;
        Ok(OrbitElimination {
            stiffening: self,
            stiffening_middle,
            curvatures,
            directions,
            response_directions,
            direction_metric_images,
            edges,
        })
    }
}

/// One orbit's evidence terms at one dense evaluation (#2234).
pub(crate) struct CompactOrbitValue {
    /// `u`, `v` and their complement images `G_⊥u`, `G_⊥v`, which the differential reads again.
    pub(crate) trigonometric: (Array1<f64>, Array1<f64>),
    pub(crate) complement_images: (Array1<f64>, Array1<f64>),
    /// `(a, b, d) = (uᵀG_⊥u, uᵀG_⊥v, vᵀG_⊥v)`.
    pub(crate) coupling_forms: [f64; 3],
    pub(crate) integral: CircleOrbitIntegral,
}

/// Every eliminated orbit's evidence terms at one dense evaluation, in tangent-column order.
pub(crate) struct CompactOrbitsValue {
    pub(crate) orbits: Vec<CompactOrbitValue>,
    /// `log det N` over every orbit tangent at once.
    pub(crate) log_gram_det: f64,
    /// `−log det N − 2·Σ log I_k + K·log 2π`. Added to the priced `log|A_s|` it gives the
    /// log-determinant the criterion ranks: `½log|A|` becomes
    /// `½log|A_s| − ½log det N − Σ log I_k + ½K·log 2π`, where each `½log 2π` restores the Laplace
    /// constant an integrated orbit dimension no longer carries (#2933 F26 pairs one per integrated
    /// dimension with the coordinate prior's normalizer). The orbits are integrated one by one
    /// because [`SaeManifoldTerm::separate_coupled_compact_orbits`] admitted only orbits in
    /// distinct connected blocks of `A` and `Φ`, whose complement couples no two of them.
    pub(crate) log_det_correction: f64,
}

impl SaeManifoldTerm {
    /// Price the eliminated orbits of `block`, one per generator in tangent-column order. The block
    /// must already be classified without refusal, so the complement pseudo-inverse is positive
    /// semidefinite and every coupling is a nonnegative quadratic form, which the quadrature's
    /// lower bound needs.
    pub(crate) fn price_compact_orbits(
        generators: &[CircleOrbitGenerator],
        block: &ExactHessianSpectralBlock,
    ) -> Result<CompactOrbitsValue, String> {
        let orbit = block
            .orbit
            .as_ref()
            .ok_or_else(|| "price_compact_orbits: the block carries no eliminated orbit".to_string())?;
        if orbit.gram().nrows() != generators.len() {
            return Err(format!(
                "price_compact_orbits: {} generators for {} eliminated orbit tangents",
                generators.len(),
                orbit.gram().nrows()
            ));
        }
        let dim = block.eigenvalues.len();
        let mut orbits = Vec::with_capacity(generators.len());
        for generator in generators {
            let (u, v) = generator.trigonometric_images(dim);
            let complement_u = orbit.complement_apply(block, u.view());
            let complement_v = orbit.complement_apply(block, v.view());
            let coupling_forms = [
                u.dot(&complement_u),
                0.5 * (u.dot(&complement_v) + v.dot(&complement_u)),
                v.dot(&complement_v),
            ];
            let integral = generator
                .integrand(coupling_forms[0], coupling_forms[1], coupling_forms[2])
                .integrate()?;
            orbits.push(CompactOrbitValue {
                trigonometric: (u, v),
                complement_images: (complement_u, complement_v),
                coupling_forms,
                integral,
            });
        }
        let (gram_values, _) = orbit
            .gram()
            .eigh(Side::Lower)
            .map_err(|error| format!("price_compact_orbits: Gram eigendecomposition failed: {error:?}"))?;
        let log_gram_det = gram_values.iter().map(|value| value.ln()).sum::<f64>();
        let log_integrals = orbits
            .iter()
            .map(|value| value.integral.log_integral)
            .sum::<f64>();
        let log_det_correction = -log_gram_det - 2.0 * log_integrals
            + orbit.gram().nrows() as f64 * std::f64::consts::TAU.ln();
        Ok(CompactOrbitsValue {
            orbits,
            log_gram_det,
            log_det_correction,
        })
    }
}

impl OrbitElimination {
    /// `N = TᵀΦT`, whose log-determinant is the orbit coordinate's Jacobian in the evidence.
    pub(crate) fn gram(&self) -> &Array2<f64> {
        &self.stiffening.gram
    }

    /// `G_⊥x = P_Φ·A_s⁺·P_Φᵀx`: the complement's pseudo-inverse, blind to the orbit.
    pub(crate) fn complement_apply(
        &self,
        block: &ExactHessianSpectralBlock,
        x: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let dual = self.stiffening.project_dual(x);
        let solved = block.apply_stiffened_pseudo_inverse(dual.view());
        self.stiffening.project_primal(solved.view())
    }
}

impl ExactHessianSpectralBlock {
    /// `A_s⁺v` over the directions this block retains.
    fn apply_stiffened_pseudo_inverse(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let coefficients = self.eigenvectors.t().dot(&v);
        let mut scaled = Array1::<f64>::zeros(coefficients.len());
        for index in 0..coefficients.len() {
            if self.eigenvalues[index].abs() > self.rank_floor(index) {
                scaled[index] = coefficients[index] / self.eigenvalues[index];
            }
        }
        self.eigenvectors.dot(&scaled)
    }

    /// `A⁺` over the resolved directions as `(basis, inverse weights)`, `A⁺ = B·diag(w)·Bᵀ`.
    ///
    /// Without an eliminated orbit `B` is the retained pencil eigenvectors and `w = 1/μ`. With one,
    /// the columns are the complement's retained directions projected off the orbit, weighted
    /// `1/μ`, followed by the orbit response columns `VU` of every resolved orbit curvature,
    /// weighted `1/σ` (#2234).
    pub(crate) fn retained_pseudo_inverse_factors(&self) -> (Array2<f64>, Array1<f64>) {
        let dim = self.eigenvalues.len();
        let retained: Vec<usize> = (0..dim)
            .filter(|&index| self.eigenvalues[index].abs() > self.rank_floor(index))
            .collect();
        let Some(orbit) = self.orbit.as_ref() else {
            let basis = self.eigenvectors.select(ndarray::Axis(1), &retained);
            let weights = retained
                .iter()
                .map(|&index| 1.0 / self.eigenvalues[index])
                .collect();
            return (basis, weights);
        };
        let resolved: Vec<usize> = (0..orbit.curvatures.len())
            .filter(|&index| orbit.curvatures[index].abs() > orbit.edges[index])
            .collect();
        let mut basis = Array2::<f64>::zeros((dim, retained.len() + resolved.len()));
        let mut weights = Array1::<f64>::zeros(retained.len() + resolved.len());
        for (position, &index) in retained.iter().enumerate() {
            basis
                .column_mut(position)
                .assign(&orbit.stiffening.project_primal(self.eigenvectors.column(index)));
            weights[position] = 1.0 / self.eigenvalues[index];
        }
        for (offset, &index) in resolved.iter().enumerate() {
            let position = retained.len() + offset;
            basis
                .column_mut(position)
                .assign(&orbit.response_directions.column(index));
            weights[position] = 1.0 / orbit.curvatures[index];
        }
        (basis, weights)
    }

    /// `A x` for the unstiffened operator: `A_s x + ΦTN⁻¹(ATᵀx) + AT·N⁻¹(ΦT)ᵀx − ΦTN⁻¹(TᵀAT + N)N⁻¹(ΦT)ᵀx`.
    fn apply_unstiffened_operator(&self, orbit: &OrbitElimination, x: ArrayView1<'_, f64>) -> Array1<f64> {
        let stiffening = &orbit.stiffening;
        let projector_dual = stiffening.gram_inverse.dot(&stiffening.metric_images.t().dot(&x));
        let operator_dual = stiffening.operator_images.t().dot(&x);
        let mut image = self.operator.dot(&x);
        image += &stiffening
            .metric_images
            .dot(&stiffening.gram_inverse.dot(&operator_dual));
        image += &stiffening.operator_images.dot(&projector_dual);
        image -= &stiffening
            .metric_images
            .dot(&stiffening.gram_inverse.dot(&orbit.stiffening_middle.dot(&projector_dual)));
        image
    }

    /// [`Self::solve_stationarity`] through the eliminated orbit: `x = A⁺·rhs` for the unstiffened
    /// `A`, certified by the physical residual `A x − rhs` after removing its dual components along
    /// the held-out band (the complement's in-band directions and the in-band orbit directions).
    ///
    /// The held-out primal vectors, `P_ΦW_Z` for the complement and `TU` for the orbit, belong to a
    /// `Φ`-orthonormal basis with the retained directions. So `yᵀr` IS the dual component of `r`
    /// along `Φy`, and the band is removed as [`Self::solve_stationarity`] removes its own, one
    /// pairing per direction with no Gram inverted. This rounds at `γ·|Φy||y||r|` per direction. A
    /// normal-equations projection onto the images `Φy` rounds at the square of `Φ`'s
    /// conditioning instead: a collapsed chart's 69 held-out images in 70 dimensions left 0.46 of
    /// the backward scale (#2263 item-4 replay), and a real Qwen3-8B chart's image Gram read
    /// indefinite at −2.9e-7 against 3.3e9.
    fn solve_orbit_eliminated_stationarity(
        &self,
        orbit: &OrbitElimination,
        rhs: &SaeArrowVector,
    ) -> Result<ExactStationaritySolve, String> {
        let total_t = rhs.t.len();
        let dim = total_t + rhs.beta.len();
        if self.eigenvalues.len() != dim {
            return Err(format!(
                "orbit-eliminated stationarity solve: spectrum {}, RHS dimension {dim}",
                self.eigenvalues.len()
            ));
        }
        let mut flat_rhs = Array1::<f64>::zeros(dim);
        flat_rhs.slice_mut(s![..total_t]).assign(&rhs.t);
        flat_rhs.slice_mut(s![total_t..]).assign(&rhs.beta);
        if !flat_rhs.iter().all(|value| value.is_finite()) {
            return Err("orbit-eliminated stationarity solve: RHS contains a non-finite value".to_string());
        }
        let (basis, weights) = self.retained_pseudo_inverse_factors();
        let coefficients = basis.t().dot(&flat_rhs) * &weights;
        let solution = basis.dot(&coefficients);
        let residual = &self.apply_unstiffened_operator(orbit, solution.view()) - &flat_rhs;
        let band_orbit: Vec<usize> = (0..orbit.curvatures.len())
            .filter(|&index| orbit.curvatures[index].abs() <= orbit.edges[index])
            .collect();
        let mut removed = Array1::<f64>::zeros(dim);
        for (position, &index) in self.band.iter().enumerate() {
            let primal = orbit.stiffening.project_primal(self.eigenvectors.column(index));
            let dual = orbit.stiffening.project_dual(self.band_metric_images.column(position));
            removed.scaled_add(primal.dot(&residual), &dual);
        }
        for &index in &band_orbit {
            removed.scaled_add(
                orbit.directions.column(index).dot(&residual),
                &orbit.direction_metric_images.column(index),
            );
        }
        let remainder = &residual - &removed;
        let norm = |vector: &Array1<f64>| vector.dot(vector).max(0.0).sqrt();
        // Removing the band's dual components rounds at the scale of what it removes too.
        let scale = orbit.stiffening.operator_frobenius * norm(&solution) + norm(&flat_rhs) + norm(&removed);
        let tolerance = f64::EPSILON.sqrt();
        let remainder_norm = norm(&remainder);
        if !solution.iter().all(|value| value.is_finite())
            || !(remainder_norm == 0.0 || (scale > 0.0 && remainder_norm <= tolerance * scale))
        {
            return Err(format!(
                "orbit-eliminated stationarity solve failed certification: physical residual \
                 off the held-out band {remainder_norm:.6e} / backward scale {scale:.6e}, \
                 tolerance {tolerance:.6e}, orbit curvatures {:?} against edges {:?}",
                orbit.curvatures, orbit.edges
            ));
        }
        let mut band: Vec<ExactABandDirection> = self
            .band
            .iter()
            .map(|&index| ExactABandDirection {
                magnitude: self.eigenvalues[index].abs(),
                edge: self.rank_floor(index),
            })
            .collect();
        band.extend(band_orbit.iter().map(|&index| ExactABandDirection {
            magnitude: orbit.curvatures[index].abs(),
            edge: orbit.edges[index],
        }));
        // The factors carry one column per retained stiffened direction plus one per resolved
        // orbit curvature. Every orbit tangent is a retained stiffened direction (`A_sτ = Φτ`), and
        // projecting it off the orbit removes it, so `A`'s retained rank is the column count less
        // the tangent count.
        let retained_rank = weights.len().checked_sub(orbit.curvatures.len()).ok_or_else(|| {
            format!(
                "orbit-eliminated stationarity solve: {} pseudo-inverse columns cannot hold {} orbit tangents",
                weights.len(),
                orbit.curvatures.len()
            )
        })?;
        // Every orbit tangent is stiffened to its metric (`μ = 1`), so the stiffened block's
        // negative directions are `A`'s on the complement; the orbit's own resolved negative
        // curvatures join them.
        let negative_curvature = ResolvedNegativeCurvature::of_directions(
            (0..self.eigenvalues.len())
                .map(|index| (self.eigenvalues[index], self.rank_floor(index)))
                .chain(
                    orbit
                        .curvatures
                        .iter()
                        .copied()
                        .zip(orbit.edges.iter().copied()),
                ),
        );
        Ok(ExactStationaritySolve {
            step: SaeArrowVector {
                t: solution.slice(s![..total_t]).to_owned(),
                beta: solution.slice(s![total_t..]).to_owned(),
            },
            band,
            retained_rank,
            negative_curvature,
        })
    }
}

/// The differential of the orbit-integrated log-determinant
/// `log|A_s|_reg − log det N − 2·log I + log 2π` on its rank stratum (#2234): the weights and
/// direct legs the dense channels contract in place of the stiffened block's own pricing weights.
///
/// With `S = ΦTN⁻¹`, `M = TᵀAT + N`, `X` the stiffened block's `dA` weight, `F = MSᵀ − (AT)ᵀ`,
/// `G = XFᵀ` and `H = SᵀXS`:
/// * `dA`: `P_ΦXP_Φᵀ`, `P_Φ = I − TSᵀ`;
/// * `dΦ`: the block's metric weight `+ sym(2·P_ΦGN⁻¹Tᵀ) + THTᵀ − TN⁻¹Tᵀ`;
/// * `dT`: `2ΦGN⁻¹ − 2ΦTN⁻¹GᵀS − 2ΦTSᵀGN⁻¹ − 2A·XS + 2AT·H + 2ΦT·H − 2ΦTN⁻¹`.
///
/// With `X = A_s⁻¹` these reduce to the complement form `tr(G_⊥dA) − 2·tr(sym(G_⊥ATN⁻¹Tᵀ)dΦ)`.
/// Each coupling form `f = xᵀG_⊥y` of the orbit integral moves, with `V = T − G_⊥AT`, as
///
/// ```text
///   df = −(G_⊥x)ᵀ dA (G_⊥y) − ⟨G_⊥x(TN⁻¹Vᵀy)ᵀ + TN⁻¹Vᵀx(G_⊥y)ᵀ, dΦ⟩ − ⟨ΦG_⊥x(N⁻¹Vᵀy)ᵀ + ΦG_⊥y(N⁻¹Vᵀx)ᵀ, dT⟩,
/// ```
///
/// and each integral's coordinate and log-precision legs are expectations under its node weights.
/// Every orbit's tangent moves only with its own atom's decoder coordinates: `∂τ_β/∂C = −K̃`.
struct CompactOrbitDifferential {
    /// Weight on `dA` (full log-determinant units).
    operator_weight: Array2<f64>,
    /// Weight on `dΦ` (full log-determinant units).
    metric_weight: Array2<f64>,
    /// θ legs that reach neither `A` nor `Φ` (full units).
    theta: SaeArrowVector,
    /// `(atom, ∂/∂log α)` of each orbit integral at fixed `A`, on its atom's ARD axis (full units).
    log_precisions: Vec<(usize, f64)>,
}

fn orbit_outer(left: &Array1<f64>, right: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn((left.len(), right.len()), |(row, col)| left[row] * right[col])
}

fn orbit_symmetrized(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t()) * 0.5
}

impl SaeManifoldTerm {
    fn compact_orbit_differential(
        generators: &[CircleOrbitGenerator],
        block: &ExactHessianSpectralBlock,
        pricing: &ExactHessianPricing,
        metric: &dyn ExactAPencilMetric,
        total_t: usize,
    ) -> Result<CompactOrbitDifferential, String> {
        let orbit = block.orbit.as_ref().ok_or_else(|| {
            "compact_orbit_differential: the block carries no eliminated orbit".to_string()
        })?;
        let stiffening = &orbit.stiffening;
        let dim = block.eigenvalues.len();
        let k = stiffening.tangents.ncols();
        let tangents = &stiffening.tangents;
        let images = &stiffening.metric_images;
        let operator_images = &stiffening.operator_images;
        let gram_inverse = &stiffening.gram_inverse;
        let weight = &pricing.a_derivative;
        let projector = images.dot(gram_inverse);

        // dA through the stiffening: P_Φ X P_Φᵀ.
        let left = weight - &tangents.dot(&projector.t().dot(weight));
        let mut operator_weight = &left - &left.dot(&projector).dot(&tangents.t());

        // dΦ and dT through the stiffening and −log det N.
        let factor = orbit.stiffening_middle.dot(&projector.t()) - &operator_images.t();
        let spread = weight.dot(&factor.t());
        let sandwich = projector.t().dot(&weight.dot(&projector));
        let projected_spread = &spread - &tangents.dot(&projector.t().dot(&spread));
        let lifted = projected_spread.dot(gram_inverse).dot(&tangents.t());
        let mut metric_weight = pricing.metric_derivative.clone();
        metric_weight += &(&lifted + &lifted.t());
        metric_weight += &tangents.dot(&sandwich).dot(&tangents.t());
        metric_weight -= &tangents.dot(gram_inverse).dot(&tangents.t());
        let weighted_projector = weight.dot(&projector);
        let mut metric_spread = Array2::<f64>::zeros((dim, k));
        let mut operator_weighted_projector = Array2::<f64>::zeros((dim, k));
        for column in 0..k {
            metric_spread
                .column_mut(column)
                .assign(&metric.apply(spread.column(column))?);
            operator_weighted_projector
                .column_mut(column)
                .assign(&block.apply_unstiffened_operator(orbit, weighted_projector.column(column)));
        }
        let mut tangent_weight = metric_spread.dot(gram_inverse) * 2.0;
        tangent_weight -= &(images.dot(gram_inverse).dot(&spread.t().dot(&projector)) * 2.0);
        tangent_weight -= &(images.dot(&projector.t().dot(&spread)).dot(gram_inverse) * 2.0);
        tangent_weight -= &(&operator_weighted_projector * 2.0);
        tangent_weight += &(operator_images.dot(&sandwich) * 2.0);
        tangent_weight += &(images.dot(&sandwich) * 2.0);
        tangent_weight -= &(images.dot(gram_inverse) * 2.0);

        // −2·Σ log I_k: expectations under each quadrature's node weights. Every coupling form reads
        // the complement of ALL eliminated orbits, so its `dT` leg reaches every tangent column.
        let values = Self::price_compact_orbits(generators, block)?;
        let mut response = tangents.clone();
        for column in 0..k {
            let solved = orbit.complement_apply(block, operator_images.column(column));
            let mut target = response.column_mut(column);
            target -= &solved;
        }
        let one_minus_cos = |angle: f64| {
            let half = (0.5 * angle).sin();
            2.0 * half * half
        };
        let mut theta = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(dim - total_t),
        };
        let mut log_precisions = Vec::with_capacity(generators.len());
        for (generator, value) in generators.iter().zip(values.orbits.iter()) {
            let (u, v) = &value.trigonometric;
            let (complement_u, complement_v) = &value.complement_images;
            let angles = &value.integral.angles;
            let node_weights = &value.integral.weights;
            let expect = |function: &dyn Fn(f64) -> f64| -> f64 {
                angles
                    .iter()
                    .zip(node_weights.iter())
                    .map(|(&angle, &node_weight)| node_weight * function(angle))
                    .sum()
            };
            let eta = generator.eta;
            let kappa = generator.kappa;
            let coupling_scale = 0.5 * eta * eta * kappa * kappa;
            let mean_one_minus_cos = expect(&one_minus_cos);
            let mean_sin = expect(&|angle: f64| angle.sin());
            let weight_a = coupling_scale * expect(&|angle: f64| one_minus_cos(angle).powi(2));
            let weight_b =
                -2.0 * coupling_scale * expect(&|angle: f64| one_minus_cos(angle) * angle.sin());
            let weight_d = coupling_scale * expect(&|angle: f64| angle.sin().powi(2));
            let forms: [(f64, &Array1<f64>, &Array1<f64>, &Array1<f64>, &Array1<f64>); 3] = [
                (weight_a, u, u, complement_u, complement_u),
                (weight_b, u, v, complement_u, complement_v),
                (weight_d, v, v, complement_v, complement_v),
            ];
            for (form_weight, x, y, complement_x, complement_y) in forms {
                if form_weight == 0.0 {
                    continue;
                }
                // d(−2·log I) = −2·(∂log I/∂f)·df, and every leg of df carries a leading minus.
                let scale = 2.0 * form_weight;
                operator_weight += &(orbit_symmetrized(&orbit_outer(complement_x, complement_y)) * scale);
                let coefficient_x = gram_inverse.dot(&response.t().dot(x));
                let coefficient_y = gram_inverse.dot(&response.t().dot(y));
                let lift_x = tangents.dot(&coefficient_x);
                let lift_y = tangents.dot(&coefficient_y);
                let metric_leg = &orbit_outer(complement_x, &lift_y) + &orbit_outer(&lift_x, complement_y);
                metric_weight += &(orbit_symmetrized(&metric_leg) * scale);
                let metric_complement_x = metric.apply(complement_x.view())?;
                let metric_complement_y = metric.apply(complement_y.view())?;
                tangent_weight += &(orbit_outer(&metric_complement_x, &coefficient_y) * scale);
                tangent_weight += &(orbit_outer(&metric_complement_y, &coefficient_x) * scale);
            }
            for &(slot, row_weight, coordinate) in &generator.prior_rows {
                let (sin, cos) = (kappa * coordinate).sin_cos();
                let resultant = -eta * kappa * row_weight * (-sin * mean_one_minus_cos + cos * mean_sin);
                let moved_u = row_weight * kappa * cos;
                let moved_v = -row_weight * kappa * sin;
                let coupling = weight_a * 2.0 * complement_u[slot] * moved_u
                    + weight_b * (complement_v[slot] * moved_u + complement_u[slot] * moved_v)
                    + weight_d * 2.0 * complement_v[slot] * moved_v;
                theta.t[slot] -= 2.0 * (resultant + coupling);
            }
            let integrand =
                generator.integrand(value.coupling_forms[0], value.coupling_forms[1], value.coupling_forms[2]);
            let [qa, qb, qd] = integrand.coupling;
            let mean_ard =
                -(integrand.resultant_cos * mean_one_minus_cos + integrand.resultant_sin * mean_sin);
            let mean_coupling = expect(&|angle: f64| {
                let lowered = one_minus_cos(angle);
                let sin = angle.sin();
                qa * lowered * lowered - 2.0 * qb * lowered * sin + qd * sin * sin
            });
            log_precisions.push((generator.atom, -2.0 * (mean_ard + 2.0 * mean_coupling)));
        }
        // Each tangent's border block is −K̃C on its own atom, so its column's weight reaches that
        // atom's decoder coordinates as −K̃ᵀ.
        for (column_of_tangent, generator) in generators.iter().enumerate() {
            let rank = generator.border_rank;
            for column in 0..generator.basis_size {
                for channel in 0..rank {
                    let moved =
                        tangent_weight[[generator.border_start + column * rank + channel, column_of_tangent]];
                    if moved == 0.0 {
                        continue;
                    }
                    for j in 0..generator.basis_size {
                        theta.beta[generator.border_start - total_t + j * rank + channel] -=
                            generator.closure[[column, j]] * moved;
                    }
                }
            }
        }
        Ok(CompactOrbitDifferential {
            operator_weight,
            metric_weight,
            theta,
            log_precisions,
        })
    }
}

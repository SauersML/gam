// Included in implicit_psi_derivative.rs: these jets share its coefficient and
// row projections. A logarithmic Riesz representative is not homogeneous.
// Write D = K_u - r K_r - delta K. Each logarithmic block contributes
// -a_m c_m r^(2m-d), so D is a polynomial and is itself homogeneous.
// Consequently the missing raw-axis jets are
//   K_a: D/d,
//   K_ab: (D_a + D_b)/d.
// The latter identity also accounts for the mixed global/radial derivatives;
// substituting an isotropic second derivative on every axis would be wrong.
#[derive(Debug, Clone)]
struct DuchonLogarithmicPsiCorrection {
    data: Arc<Array2<f64>>,
    centers: Arc<Array2<f64>>,
    metric: Vec<f64>,
    coefficients: Vec<(usize, f64)>,
}

impl DuchonLogarithmicPsiCorrection {
    fn new(
        data: ArrayView2<'_, f64>,
        centers: ArrayView2<'_, f64>,
        eta: &[f64],
        kind: &RadialScalarKind,
    ) -> Option<Arc<Self>> {
        let coefficients = Self::coefficients(kind)?;
        Some(Arc::new(Self {
            data: shared_owned_data_matrix_from_view(data),
            centers: shared_owned_centers_matrix_from_view(centers),
            metric: centered_aniso_metric_weights(eta),
            coefficients,
        }))
    }

    fn coefficients(kind: &RadialScalarKind) -> Option<Vec<(usize, f64)>> {
        let RadialScalarKind::Duchon { dim, coeffs, .. } = kind else {
            return None;
        };
        if !dim.is_multiple_of(2) {
            return None;
        }
        let coefficients: Vec<_> = coeffs
            .a
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(m, a)| **a != 0.0 && 2 * *m >= *dim)
            .filter_map(|(m, a)| {
                let degree = m - dim / 2;
                let (_, log_coefficient) = duchon_polyharmonic_block_taylor_r2j(m, *dim, degree);
                (log_coefficient != 0.0).then_some((degree, -a * log_coefficient))
            })
            .collect();
        if coefficients.is_empty() {
            return None;
        }
        Some(coefficients)
    }

    fn radial(&self, r2: f64) -> (f64, f64) {
        Self::evaluate(&self.coefficients, r2)
    }

    fn evaluate(coefficients: &[(usize, f64)], r2: f64) -> (f64, f64) {
        let mut value = KahanSum::default();
        let mut radial = KahanSum::default();
        for &(degree, coefficient) in coefficients {
            value.add(coefficient * r2.powi(degree as i32));
            if degree > 0 {
                radial.add(2.0 * degree as f64 * coefficient * r2.powi(degree as i32 - 1));
            }
        }
        (value.sum(), radial.sum())
    }
}

impl ImplicitDesignPsiDerivative {
    fn with_logarithmic_correction(
        mut self,
        correction: Option<Arc<DuchonLogarithmicPsiCorrection>>,
    ) -> Self {
        self.logarithmic_correction = correction;
        self
    }

    fn logarithmic_axis(&self, axis: usize, components: &[f64]) -> (f64, f64) {
        match self.axis_combinations.as_ref() {
            Some(_) => self
                .transformed_axis_combination(axis)
                .iter()
                .fold((0.0, 0.0), |(value, total), &(raw, weight)| {
                    (value + weight * components[raw], total + weight)
                }),
            None => (components[axis], 1.0),
        }
    }

    fn add_logarithmic_correction(
        &self,
        key: ProjectedJetKey,
        rows: std::ops::Range<usize>,
        matrix: &mut Array2<f64>,
    ) {
        let Some(correction) = self.logarithmic_correction.as_ref() else {
            return;
        };
        let mut raw = Array2::<f64>::zeros((rows.len(), self.n_knots));
        let mut components = vec![0.0; self.n_axes];
        for (local, row) in rows.enumerate() {
            for center in 0..self.n_knots {
                for (axis, component) in components.iter_mut().enumerate() {
                    let displacement =
                        correction.data[[row, axis]] - correction.centers[[center, axis]];
                    *component = correction.metric[axis] * displacement * displacement;
                }
                let r2 = components.iter().sum::<f64>();
                // The existing marked collision carrier already emits the exact
                // first and second jets there. Correct every non-collision pair.
                if r2 == 0.0 {
                    continue;
                }
                let (value, radial) = correction.radial(r2);
                let scalar = match key {
                    ProjectedJetKey::FirstRaw(axis) => {
                        let (_, total) = self.logarithmic_axis(axis, &components);
                        value * total
                    }
                    ProjectedJetKey::SecondDiagonal(axis) => {
                        let (component, total) = self.logarithmic_axis(axis, &components);
                        2.0 * total * (radial * component + self.effective_share(axis) * value)
                    }
                    ProjectedJetKey::SecondCross(a, b) => {
                        let (sa, ca) = self.logarithmic_axis(a, &components);
                        let (sb, cb) = self.logarithmic_axis(b, &components);
                        radial * (sa * cb + sb * ca)
                            + value * (self.effective_share(a) * cb + self.effective_share(b) * ca)
                    }
                };
                raw[[local, center]] = self.chart_scale * scalar / self.n_axes as f64;
            }
        }
        match key {
            ProjectedJetKey::FirstRaw(_) => *matrix += &raw,
            ProjectedJetKey::SecondDiagonal(_) | ProjectedJetKey::SecondCross(_, _) => {
                *matrix += &self.project_matrix_rows(raw);
            }
        }
    }

    fn logarithmic_jet_chunk(
        &self,
        key: ProjectedJetKey,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        match key {
            ProjectedJetKey::FirstRaw(axis) => self.row_chunk_first_raw(axis, rows),
            ProjectedJetKey::SecondDiagonal(axis) => self.row_chunk_second_diag(axis, rows),
            ProjectedJetKey::SecondCross(a, b) => self.row_chunk_second_cross(a, b, rows),
        }
    }

    fn logarithmic_transpose(
        &self,
        key: ProjectedJetKey,
        v: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert_eq!(v.len(), self.n);
        let width = match key {
            ProjectedJetKey::FirstRaw(_) => self.n_knots,
            _ => self.p_out(),
        };
        let mut result = Array1::<f64>::zeros(width);
        for start in (0..self.n).step_by(IMPLICIT_MATVEC_CHUNK_SIZE) {
            let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(self.n);
            result += &self
                .logarithmic_jet_chunk(key, start..end)?
                .t()
                .dot(&v.slice(ndarray::s![start..end]));
        }
        Ok(result)
    }

    fn logarithmic_forward(
        &self,
        key: ProjectedJetKey,
        u: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        let mut result = Array1::<f64>::zeros(self.n);
        for start in (0..self.n).step_by(IMPLICIT_MATVEC_CHUNK_SIZE) {
            let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(self.n);
            result
                .slice_mut(ndarray::s![start..end])
                .assign(&self.logarithmic_jet_chunk(key, start..end)?.dot(u));
        }
        Ok(result)
    }
}

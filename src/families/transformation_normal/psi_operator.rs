use super::*;

pub(crate) struct TensorKroneckerPsiOperator {
    pub(crate) response_val_basis: Arc<Array2<f64>>,
    pub(crate) covariate_design: DesignMatrix,
    pub(crate) covariate_derivs: Vec<CustomFamilyBlockPsiDerivative>,
    pub(crate) covariate_first_cache: Arc<Vec<Mutex<Option<Arc<Array2<f64>>>>>>,
}

impl TensorKroneckerPsiOperator {
    pub(crate) fn n_data(&self) -> usize {
        self.response_val_basis.nrows()
    }

    pub(crate) fn p_resp(&self) -> usize {
        self.response_val_basis.ncols()
    }

    pub(crate) fn p_cov(&self) -> usize {
        self.covariate_design.ncols()
    }

    pub(crate) fn p_out(&self) -> usize {
        self.p_resp() * self.p_cov()
    }

    pub(crate) fn cov_deriv(
        &self,
        axis: usize,
    ) -> Result<&CustomFamilyBlockPsiDerivative, crate::terms::basis::BasisError> {
        self.covariate_derivs.get(axis).ok_or_else(|| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker psi axis {axis} out of bounds for {} axes",
                self.covariate_derivs.len()
            ))
        })
    }

    pub(crate) fn cov_forward_first(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(crate::faer_ndarray::fast_av(&deriv.x_psi, u));
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi operator for axis {axis}"
            )));
        };
        op.forward_mul(deriv.implicit_axis, u)
    }

    pub(crate) fn cov_transpose_first(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(crate::faer_ndarray::fast_atv(&deriv.x_psi, v));
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi transpose operator for axis {axis}"
            )));
        };
        op.transpose_mul(deriv.implicit_axis, v)
    }

    pub(crate) fn cov_forward_second(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == self.cov_deriv(axis_e)?.implicit_group_id
        {
            if deriv_d.implicit_axis == self.cov_deriv(axis_e)?.implicit_axis {
                return op.forward_mul_second_diag(deriv_d.implicit_axis, u);
            }
            return op.forward_mul_second_cross(
                deriv_d.implicit_axis,
                self.cov_deriv(axis_e)?.implicit_axis,
                u,
            );
        }
        if let Some(rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = rows.get(axis_e)
            && mat.nrows() == self.n_data()
            && mat.ncols() == self.p_cov()
        {
            return Ok(crate::faer_ndarray::fast_av(mat, u));
        }
        Ok(Array1::<f64>::zeros(self.n_data()))
    }

    pub(crate) fn cov_transpose_second(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == self.cov_deriv(axis_e)?.implicit_group_id
        {
            if deriv_d.implicit_axis == self.cov_deriv(axis_e)?.implicit_axis {
                return op.transpose_mul_second_diag(deriv_d.implicit_axis, v);
            }
            return op.transpose_mul_second_cross(
                deriv_d.implicit_axis,
                self.cov_deriv(axis_e)?.implicit_axis,
                v,
            );
        }
        if let Some(rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = rows.get(axis_e)
            && mat.nrows() == self.n_data()
            && mat.ncols() == self.p_cov()
        {
            return Ok(crate::faer_ndarray::fast_atv(mat, v));
        }
        Ok(Array1::<f64>::zeros(self.p_cov()))
    }

    pub(crate) fn cov_first_axis_row_chunk(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.slice(s![rows, ..]).to_owned());
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi row chunk operator for axis {axis}"
            )));
        };
        if self.cov_first_axis_cache_fits_budget() && op.as_materializable().is_some() {
            let cached = self.materialize_cov_first_axis_arc(axis)?;
            return Ok(cached.slice(s![rows, ..]).to_owned());
        }
        op.row_chunk_first(deriv.implicit_axis, rows)
    }

    pub(crate) fn cov_first_axis_row_chunk_streaming(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.slice(s![rows, ..]).to_owned());
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi streaming row chunk operator for axis {axis}"
            )));
        };
        op.row_chunk_first(deriv.implicit_axis, rows)
    }

    pub(crate) fn cov_second_axis_row_chunk(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        let deriv_e = self.cov_deriv(axis_e)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == deriv_e.implicit_group_id
        {
            if deriv_d.implicit_axis == deriv_e.implicit_axis {
                return op.row_chunk_second_diag(deriv_d.implicit_axis, rows);
            }
            return op.row_chunk_second_cross(deriv_d.implicit_axis, deriv_e.implicit_axis, rows);
        }
        if let Some(second_rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = second_rows.get(axis_e)
            && mat.nrows() == self.n_data()
            && mat.ncols() == self.p_cov()
        {
            return Ok(mat.slice(s![rows, ..]).to_owned());
        }
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p_cov())))
    }

    pub(crate) fn lifted_row_chunk_from_cov(
        &self,
        rows: std::ops::Range<usize>,
        cov: &Array2<f64>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let n_rows = rows.end - rows.start;
        if cov.nrows() != n_rows || cov.ncols() != self.p_cov() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker covariate row chunk shape {}x{} != expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                n_rows,
                self.p_cov()
            )));
        }
        let resp = self.response_val_basis.slice(s![rows, ..]);
        Ok(dense_rowwise_kronecker(resp, cov.view()))
    }

    pub(crate) fn materialize_cov_first_axis_uncached(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.clone());
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi materialization for axis {axis}"
            )));
        };
        let mat_op = op.as_materializable().ok_or_else(|| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "covariate psi operator for axis {axis} does not support dense materialization"
            ))
        })?;
        mat_op.materialize_first(deriv.implicit_axis)
    }

    pub(crate) fn cov_first_axis_cache_fits_budget(&self) -> bool {
        let policy = ResourcePolicy::default_library();
        let per_axis_bytes = self
            .n_data()
            .saturating_mul(self.p_cov())
            .saturating_mul(std::mem::size_of::<f64>());
        let all_axes_bytes = per_axis_bytes.saturating_mul(self.covariate_derivs.len());
        per_axis_bytes <= policy.max_single_materialization_bytes
            && all_axes_bytes <= policy.max_operator_cache_bytes
    }

    pub(crate) fn materialize_cov_first_axis_arc(
        &self,
        axis: usize,
    ) -> Result<Arc<Array2<f64>>, crate::terms::basis::BasisError> {
        if axis >= self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker psi axis {axis} out of bounds for {} axes",
                self.covariate_derivs.len()
            )));
        }
        let axis_cache = &self.covariate_first_cache[axis];
        let mut cache = axis_cache.lock().map_err(|_| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker covariate first-derivative cache mutex poisoned for axis {axis}"
            ))
        })?;
        if let Some(cached) = cache.as_ref() {
            return Ok(cached.clone());
        }

        let materialized = Arc::new(self.materialize_cov_first_axis_uncached(axis)?);
        *cache = Some(materialized.clone());
        Ok(materialized)
    }

    pub(crate) fn materialize_cov_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.materialize_cov_directional(&unit.view())
    }

    /// Per-axis covariate first-derivative materialization for axis `axis`,
    /// equivalent to the unit-vector dispatch through
    /// [`materialize_cov_directional`].
    pub(crate) fn materialize_cov_first_axis(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok((*self.materialize_cov_first_axis_arc(axis)?).clone())
    }

    /// Directional `Σ_j v_psi[j] · ∂C/∂ψ_j` returning an `n × p_cov` matrix.
    /// Calling with `v_psi = e_axis` matches [`materialize_cov_first_axis`] at axis.
    /// Used by the directional outer-HVP path to compute `cov_v` once per HVP
    /// instead of materializing each per-axis cov_j.
    pub(crate) fn materialize_cov_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let mut out = Array2::<f64>::zeros((self.n_data(), self.p_cov()));
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.materialize_cov_first_axis(axis)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
    }

    pub(crate) fn lifted_forward(
        &self,
        resp_basis: &Array2<f64>,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let beta = u
            .to_owned()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|_| {
                crate::terms::basis::BasisError::InvalidInput(
                    "tensor psi coefficient reshape failed".to_string(),
                )
            })?;
        let mut out = Array1::<f64>::zeros(n);
        for j in 0..p_resp {
            let cov_part = self.cov_forward_first(axis, &beta.row(j))?;
            ndarray::Zip::from(&mut out)
                .and(&cov_part)
                .and(resp_basis.column(j))
                .par_for_each(|o, &c, &r| *o += r * c);
        }
        Ok(out)
    }

    pub(crate) fn lifted_transpose(
        &self,
        resp_basis: &Array2<f64>,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..p_resp {
            let mut weighted_v = Array1::<f64>::zeros(n);
            ndarray::Zip::from(&mut weighted_v)
                .and(resp_basis.column(j))
                .and(v)
                .par_for_each(|w, &r, &vi| *w = r * vi);
            let cov_block = self.cov_transpose_first(axis, &weighted_v.view())?;
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov])
                .assign(&cov_block);
        }
        Ok(out)
    }

    pub(crate) fn lifted_forward_second(
        &self,
        resp_basis: &Array2<f64>,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let beta = u
            .to_owned()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|_| {
                crate::terms::basis::BasisError::InvalidInput(
                    "tensor psi second coefficient reshape failed".to_string(),
                )
            })?;
        let mut out = Array1::<f64>::zeros(n);
        for j in 0..p_resp {
            let cov_part = self.cov_forward_second(axis_d, axis_e, &beta.row(j))?;
            ndarray::Zip::from(&mut out)
                .and(&cov_part)
                .and(resp_basis.column(j))
                .par_for_each(|o, &c, &r| *o += r * c);
        }
        Ok(out)
    }

    pub(crate) fn lifted_transpose_second(
        &self,
        resp_basis: &Array2<f64>,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..p_resp {
            let mut weighted_v = Array1::<f64>::zeros(n);
            ndarray::Zip::from(&mut weighted_v)
                .and(resp_basis.column(j))
                .and(v)
                .par_for_each(|w, &r, &vi| *w = r * vi);
            let cov_block = self.cov_transpose_second(axis_d, axis_e, &weighted_v.view())?;
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov])
                .assign(&cov_block);
        }
        Ok(out)
    }

    pub(crate) fn materialize_lifted(
        &self,
        resp_basis: &Array2<f64>,
        cov: &Array2<f64>,
    ) -> Array2<f64> {
        dense_rowwise_kronecker(resp_basis.view(), cov.view())
    }

    /// Internal directional accumulator on a chosen response basis:
    /// returns `Σ_j v_psi[j] · lifted_forward(resp_basis, j, β)`.
    ///
    /// Skips axes with `v_psi[j] == 0`, so calls with `v_psi = e_k` are
    /// numerically equivalent to a single direct `lifted_forward(_, k, β)` call
    /// (no extra n-vector accumulation, no rounding).
    pub(crate) fn lifted_forward_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let mut out = Array1::<f64>::zeros(self.n_data());
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.lifted_forward(resp_basis, axis, beta)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
    }

    /// Internal directional accumulator on a chosen response basis:
    /// returns `Σ_j v_psi[j] · lifted_transpose(resp_basis, j, residual)`.
    ///
    /// Returns a single `(p_resp · p_cov)`-vector, NOT a stack indexed by axis.
    pub(crate) fn lifted_transpose_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.lifted_transpose(resp_basis, axis, residual)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
    }

    /// Internal bilinear directional accumulator on a chosen response basis:
    /// returns `Σ_{j,k} v_psi[j] · w_psi[k] · lifted_transpose_second(resp_basis, j, k, residual)`.
    /// Mirror of [`lifted_forward_second_directional`] for the transpose direction.
    pub(crate) fn lifted_transpose_second_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        if v_psi.len() != q || w_psi.len() != q {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vectors length ({}, {}) do not match {} ψ axes",
                v_psi.len(),
                w_psi.len(),
                q
            )));
        }
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..q {
            for k in j..q {
                let coef = if j == k {
                    v_psi[j] * w_psi[j]
                } else {
                    v_psi[j] * w_psi[k] + v_psi[k] * w_psi[j]
                };
                if coef == 0.0 {
                    continue;
                }
                let contrib = self.lifted_transpose_second(resp_basis, j, k, residual)?;
                out.scaled_add(coef, &contrib);
            }
        }
        Ok(out)
    }

    /// Internal bilinear directional accumulator on a chosen response basis:
    /// returns `Σ_{j,k} v_psi[j] · w_psi[k] · lifted_forward_second(resp_basis, j, k, β)`.
    pub(crate) fn lifted_forward_second_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        if v_psi.len() != q || w_psi.len() != q {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vectors length ({}, {}) do not match {} ψ axes",
                v_psi.len(),
                w_psi.len(),
                q
            )));
        }
        let mut out = Array1::<f64>::zeros(self.n_data());
        for j in 0..q {
            for k in j..q {
                let coef = if j == k {
                    v_psi[j] * w_psi[j]
                } else {
                    v_psi[j] * w_psi[k] + v_psi[k] * w_psi[j]
                };
                if coef == 0.0 {
                    continue;
                }
                let contrib = self.lifted_forward_second(resp_basis, j, k, beta)?;
                out.scaled_add(coef, &contrib);
            }
        }
        Ok(out)
    }

    /// Directional `Σ_j v_psi[j] · ∂(X · β)/∂ψ_j` on the value response basis.
    /// Calling with `v_psi = e_k` returns the same n-vector as
    /// [`forward_mul`](Self::forward_mul) at axis `k` (zero entries skipped).
    pub(crate) fn forward_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_forward_directional(&resp_basis, v_psi, beta)
    }

    /// Directional transpose against the value response basis.
    /// Calling with `v_psi = e_k` matches the per-axis `transpose_mul(k, residual)`
    /// surface on the trait.
    pub(crate) fn transpose_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_transpose_directional(&resp_basis, v_psi, residual)
    }

    /// Bilinear directional `Σ_{j,k} v_psi[j] · w_psi[k] · ∂²(X · β)/∂ψ_j∂ψ_k`
    /// on the value response basis. With `v_psi = e_a, w_psi = e_b` matches
    /// `forward_mul_second_diag(a)` (when a==b) or `forward_mul_second_cross(a,b)`.
    pub(crate) fn forward_second_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_forward_second_directional(&resp_basis, v_psi, w_psi, beta)
    }

    /// Bilinear directional second-order transpose on the value response basis.
    /// With `v_psi = e_a, w_psi = e_b` matches the per-axis-pair
    /// `transpose_mul_second_diag(a)` (when a==b) or `transpose_mul_second_cross(a,b)`.
    pub(crate) fn transpose_second_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_transpose_second_directional(&resp_basis, v_psi, w_psi, residual)
    }
}

impl CustomFamilyPsiDerivativeOperator for TensorKroneckerPsiOperator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn n_data(&self) -> usize {
        TensorKroneckerPsiOperator::n_data(self)
    }

    fn p_out(&self) -> usize {
        TensorKroneckerPsiOperator::p_out(self)
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        // Per-axis trait method routes through the directional accumulator with
        // a unit basis vector e_axis. The accumulator skips zero entries, so
        // this loops once over `axis` and yields the same p-vector that
        // `lifted_transpose(_, axis, v)` would, while keeping the directional
        // kernel as a live production caller (substrate for the future HVP).
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.transpose_directional(&unit.view(), v)
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.forward_directional(&unit.view(), u)
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit = Array1::<f64>::zeros(q);
        unit[axis] = 1.0;
        self.transpose_second_directional(&unit.view(), &unit.view(), v)
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit_d = Array1::<f64>::zeros(q);
        let mut unit_e = Array1::<f64>::zeros(q);
        unit_d[axis_d] = 1.0;
        unit_e[axis_e] = 1.0;
        self.transpose_second_directional(&unit_d.view(), &unit_e.view(), v)
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        // Bilinear directional accumulator with `v_psi = w_psi = e_axis` skips
        // every (j,k) pair except (axis, axis), so this is numerically equivalent
        // to `lifted_forward_second(_, axis, axis, u)`.
        let q = self.covariate_derivs.len();
        let mut unit = Array1::<f64>::zeros(q);
        unit[axis] = 1.0;
        self.forward_second_directional(&unit.view(), &unit.view(), u)
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit_d = Array1::<f64>::zeros(q);
        let mut unit_e = Array1::<f64>::zeros(q);
        unit_d[axis_d] = 1.0;
        unit_e[axis_e] = 1.0;
        self.forward_second_directional(&unit_d.view(), &unit_e.view(), u)
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_first_axis_row_chunk(axis, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_second_axis_row_chunk(axis, axis, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_second_axis_row_chunk(axis_d, axis_e, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for TensorKroneckerPsiOperator {
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self.materialize_lifted(&self.response_val_basis, &self.materialize_cov_first(axis)?))
    }
}

// ---------------------------------------------------------------------------
// Per-evaluation ψ workspace
// ---------------------------------------------------------------------------

/// Per-evaluation ψ workspace for `TransformationNormalFamily`.
///
/// The CTN row-streaming first-order ψ kernel ([`scop_psi_terms`]) walks all
/// `n` rows serially and is invoked once per ψ axis. At large scale that is
/// the dominant outer-evaluation cost. All `n_psi` axes share the same per-row
/// state — `γ`, `h`, `h'`, `endpoint_q`, the response basis rows `rv`/`rd`,
/// the covariate row, and the row weight. The only per-axis input is
/// `cov_psi[axis] = ∂C/∂ψ_axis`, which this workspace streams in bounded row
/// chunks so the exact all-axis gradient never has to cache all
/// `n * p_cov * n_psi` derivative entries at once.
///
/// This workspace
///
/// 1. precomputes all per-axis ψ first-order terms in a single rayon
///    fold/reduce over rows so the per-row state is loaded once and reused
///    across axes, and
/// 2. caches the resulting per-axis terms so subsequent
///    `first_order_terms(idx)` lookups are O(p_total) clones rather than full
///    row walks.
///
/// `second_order_terms` are cached as a full symmetric ψ-pair table after the
/// first request so full-Hessian outer evaluations do not repeatedly traverse
/// the same CTN rows for each `(ψ_i, ψ_j)` callback. The mixed
/// `hessian_directional_derivative` hook still delegates to the direct path.
pub(crate) struct TransformationNormalPsiWorkspaceCacheEntry {
    pub(crate) objective_psi: f64,
    pub(crate) score_psi: Array1<f64>,
    pub(crate) op_arc: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) axis: usize,
    pub(crate) trace_axes: Arc<Vec<usize>>,
    pub(crate) trace_axis_pos: usize,
    pub(crate) row_gamma: Arc<Array2<f64>>,
    pub(crate) row_h: Arc<Array1<f64>>,
    pub(crate) row_h_prime: Arc<Array1<f64>>,
    pub(crate) endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    pub(crate) beta: Arc<Array1<f64>>,
}

pub(crate) struct TransformationNormalPsiWorkspaceAxisSnapshot {
    pub(crate) op_arc: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) axis: usize,
    pub(crate) row_gamma: Arc<Array2<f64>>,
    pub(crate) row_h: Arc<Array1<f64>>,
    pub(crate) row_h_prime: Arc<Array1<f64>>,
    pub(crate) endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    pub(crate) beta: Arc<Array1<f64>>,
}

pub(crate) struct TransformationNormalPsiWorkspacePairCacheEntry {
    pub(crate) objective_psi_psi: f64,
    pub(crate) score_psi_psi: Array1<f64>,
    pub(crate) op_arc: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    pub(crate) axis_i: usize,
    pub(crate) axis_j: usize,
    pub(crate) trace_axes: Arc<Vec<usize>>,
    pub(crate) trace_axis_i_pos: usize,
    pub(crate) trace_axis_j_pos: usize,
    pub(crate) row_gamma: Arc<Array2<f64>>,
    pub(crate) row_h: Arc<Array1<f64>>,
    pub(crate) row_h_prime: Arc<Array1<f64>>,
    pub(crate) endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    pub(crate) beta: Arc<Array1<f64>>,
}

pub(crate) struct TransformationNormalPsiWorkspace {
    pub(crate) family: TransformationNormalFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    pub(crate) cache: Mutex<Option<Vec<TransformationNormalPsiWorkspaceCacheEntry>>>,
    pub(crate) pair_cache:
        Mutex<Option<Vec<Vec<Option<Arc<TransformationNormalPsiWorkspacePairCacheEntry>>>>>>,
}

impl TransformationNormalPsiWorkspace {
    pub(crate) fn new(
        family: TransformationNormalFamily,
        block_states: Vec<ParameterBlockState>,
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Self {
        Self {
            family,
            block_states,
            derivative_blocks,
            cache: Mutex::new(None),
            pair_cache: Mutex::new(None),
        }
    }

    /// Compute all per-axis ψ first-order terms in a single parallel row pass.
    ///
    /// Each row's per-row state (γ, h, h', endpoint_q, rv, rd, cov_row, weight)
    /// is loaded once and reused across every ψ axis, in contrast to the
    /// per-axis [`scop_psi_terms`] path that reloads it once per axis. Op
    /// counts are identical to the per-axis path; only the loop nesting and
    /// reduction shape change.
    pub(crate) fn compute_all_axes(
        &self,
    ) -> Result<Vec<TransformationNormalPsiWorkspaceCacheEntry>, String> {
        crate::families::block_layout::block_count::validate_block_count::<TransformationNormalError>(
            "TransformationNormalFamily",
            1,
            self.block_states.len(),
        )?;
        if self.derivative_blocks.is_empty() {
            return Ok(Vec::new());
        }
        let block_derivs = &self.derivative_blocks[0];
        let n_psi = block_derivs.len();
        if n_psi == 0 {
            return Ok(Vec::new());
        }

        let beta = &self.block_states[0].beta;
        let row = self.family.row_quantities(beta)?;
        let n = self.family.response_val_basis.nrows();
        let p_resp = self.family.response_val_basis.ncols();
        let p_cov = self.family.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "TransformationNormalPsiWorkspace beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                beta.len()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("ψ workspace beta reshape failed: {e}"))?;

        // Resolve and validate the shared tensor-Kronecker ψ operator across
        // every axis. CTN is single-block so all entries point at the same
        // operator instance; we still loop to validate the contract.
        let mut op_arcs: Vec<Arc<dyn CustomFamilyPsiDerivativeOperator>> =
            Vec::with_capacity(n_psi);
        let mut axes: Vec<usize> = Vec::with_capacity(n_psi);
        for deriv in block_derivs.iter() {
            let op_arc = deriv
                .implicit_operator
                .as_ref()
                .ok_or_else(|| {
                    "TransformationNormalFamily ψ workspace requires implicit operator on each axis"
                        .to_string()
                })?
                .clone();
            // Validate tensor-Kronecker backing without holding a reference
            // across the move into `op_arcs`.
            if op_arc
                .as_any()
                .downcast_ref::<TensorKroneckerPsiOperator>()
                .is_none()
            {
                return Err(
                    "TransformationNormalFamily ψ workspace requires tensor-backed operator"
                        .to_string(),
                );
            }
            axes.push(deriv.implicit_axis);
            op_arcs.push(op_arc);
        }
        // The shared instance is whichever axis we resolve first; downcast it
        // again for row-chunk streaming. CTN's `build_tensor_psi_derivatives`
        // guarantees this is the same instance across every axis.
        let shared_op_arc = Arc::clone(&op_arcs[0]);
        let Some(op) = shared_op_arc
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
        else {
            return Err(
                "TransformationNormalFamily ψ workspace lost tensor-backed operator after validation"
                    .to_string(),
            );
        };

        let weights = self.family.effective_weights();
        let h = row.h.as_ref();
        let h_prime = row.h_prime.as_ref();
        let endpoint_q = row.endpoint_q.as_ref();
        let endpoint_basis =
            [
                self.family.response_upper_basis.as_slice().ok_or_else(|| {
                    "ψ workspace endpoint upper basis is not contiguous".to_string()
                })?,
                self.family.response_lower_basis.as_slice().ok_or_else(|| {
                    "ψ workspace endpoint lower basis is not contiguous".to_string()
                })?,
            ];

        // Single-pass row walk: for each row, load the per-row state once and
        // accumulate every axis's `objective_psi`/`score_psi` in lockstep.
        struct PsiAllAxesAccum {
            pub(crate) objective_psi: Vec<f64>,
            pub(crate) score_psi: Vec<Array1<f64>>,
        }

        impl PsiAllAxesAccum {
            pub(crate) fn new(n_psi: usize, p_total: usize) -> Self {
                Self {
                    objective_psi: vec![0.0; n_psi],
                    score_psi: (0..n_psi).map(|_| Array1::<f64>::zeros(p_total)).collect(),
                }
            }

            pub(crate) fn merge(mut self, rhs: Self) -> Self {
                for (a, v) in rhs.objective_psi.into_iter().enumerate() {
                    self.objective_psi[a] += v;
                }
                for (a, score) in rhs.score_psi.into_iter().enumerate() {
                    self.score_psi[a].scaled_add(1.0, &score);
                }
                self
            }
        }

        let policy = ResourcePolicy::default_library();
        let row_bytes = p_cov
            .saturating_mul(n_psi + 1)
            .saturating_mul(std::mem::size_of::<f64>())
            .max(1);
        let target_chunk_bytes =
            (16 * 1024 * 1024).min((policy.max_single_materialization_bytes / 8).max(row_bytes));
        let chunk_rows = (target_chunk_bytes / row_bytes).clamp(1, n.max(1));
        let row_chunks: Vec<(usize, usize)> = (0..n)
            .step_by(chunk_rows)
            .map(|start| (start, (start + chunk_rows).min(n)))
            .collect();

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let partials: Vec<Result<PsiAllAxesAccum, String>> = row_chunks
            .into_par_iter()
            .map(|(start, end)| {
                let cov = self
                    .family
                    .covariate_design
                    .try_row_chunk(start..end)
                    .map_err(|e| format!("ψ workspace covariate row chunk {start}..{end}: {e}"))?;
                let mut cov_psi_chunks: Vec<Array2<f64>> = Vec::with_capacity(n_psi);
                for &axis in &axes {
                    let cov_psi = op
                        .cov_first_axis_row_chunk_streaming(axis, start..end)
                        .map_err(|e| {
                            format!("ψ workspace covariate ψ row chunk axis {axis} {start}..{end}: {e}")
                        })?;
                    if cov_psi.nrows() != end - start || cov_psi.ncols() != p_cov {
                        return Err(TransformationNormalError::InvalidInput { reason: format!(
                            "ψ workspace covariate derivative chunk shape {}x{} for axis {axis} rows {start}..{end} != expected {}x{}",
                            cov_psi.nrows(),
                            cov_psi.ncols(),
                            end - start,
                            p_cov
                        ) }.into());
                    }
                    cov_psi_chunks.push(cov_psi);
                }

                let mut acc = PsiAllAxesAccum::new(n_psi, p_total);
                let mut gamma = vec![0.0; p_resp];
                let mut h_factor = vec![0.0; p_resp];
                let mut hp_factor = vec![0.0; p_resp];
                let mut endpoint_factor = vec![[0.0_f64; 2]; p_resp];
                let mut gamma_psi = vec![0.0; p_resp];
                let mut hpsi_cov_factor = vec![0.0; p_resp];
                let mut hppsi_cov_factor = vec![0.0; p_resp];
                let mut hpsi_psi_factor = vec![0.0; p_resp];
                let mut hppsi_psi_factor = vec![0.0; p_resp];
                let mut endpoint_psi = [0.0_f64; 2];
                let mut endpoint_psi_cov_factor = vec![[0.0_f64; 2]; p_resp];
                let mut endpoint_psi_psi_factor = vec![[0.0_f64; 2]; p_resp];

                for local_i in 0..(end - start) {
                    let i = start + local_i;
                    let cov_row = cov.row(local_i);
                    let rv = self.family.response_val_basis.row(i);
                    let rd = self.family.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hi = h[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let q = &endpoint_q[i];
                    let gamma_row = row.gamma.row(i);

                    for k in 0..p_resp {
                        gamma[k] = gamma_row[k];
                    }

                    h_factor[0] = rv[0];
                    hp_factor[0] = rd[0];
                    for k in 1..p_resp {
                        h_factor[k] = 2.0 * rv[k] * gamma[k];
                        hp_factor[k] = 2.0 * rd[k] * gamma[k];
                    }
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_factor[0][e] = basis[0];
                        for k in 1..p_resp {
                            endpoint_factor[k][e] = 2.0 * basis[k] * gamma[k];
                        }
                    }

                    for axis_idx in 0..n_psi {
                        let psi_row = cov_psi_chunks[axis_idx].row(local_i);

                        for k in 0..p_resp {
                            gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                        }

                        let mut h_psi = rv[0] * gamma_psi[0];
                        let mut hp_psi = rd[0] * gamma_psi[0];
                        for k in 1..p_resp {
                            h_psi += 2.0 * rv[k] * gamma[k] * gamma_psi[k];
                            hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
                        }

                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_psi[e] = basis[0] * gamma_psi[0];
                            endpoint_psi_psi_factor[0][e] = basis[0];
                            endpoint_psi_cov_factor[0][e] = 0.0;
                            for k in 1..p_resp {
                                endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
                                endpoint_psi_cov_factor[k][e] = 2.0 * basis[k] * gamma_psi[k];
                                endpoint_psi_psi_factor[k][e] = 2.0 * basis[k] * gamma[k];
                            }
                        }

                        acc.objective_psi[axis_idx] += wi
                            * (hi * h_psi
                                - hp_psi * inv_hp
                                + endpoint_chain_first(q, endpoint_psi));

                        hpsi_psi_factor[0] = rv[0];
                        hppsi_psi_factor[0] = rd[0];
                        hpsi_cov_factor[0] = 0.0;
                        hppsi_cov_factor[0] = 0.0;
                        for k in 1..p_resp {
                            hpsi_cov_factor[k] = 2.0 * rv[k] * gamma_psi[k];
                            hppsi_cov_factor[k] = 2.0 * rd[k] * gamma_psi[k];
                            hpsi_psi_factor[k] = 2.0 * rv[k] * gamma[k];
                            hppsi_psi_factor[k] = 2.0 * rd[k] * gamma[k];
                        }

                        let score_axis = &mut acc.score_psi[axis_idx];
                        for k in 0..p_resp {
                            for c in 0..p_cov {
                                let idx = k * p_cov + c;
                                let h_a = h_factor[k] * cov_row[c];
                                let hp_a = hp_factor[k] * cov_row[c];
                                let hpsi_a = hpsi_cov_factor[k] * cov_row[c]
                                    + hpsi_psi_factor[k] * psi_row[c];
                                let hppsi_a = hppsi_cov_factor[k] * cov_row[c]
                                    + hppsi_psi_factor[k] * psi_row[c];
                                let endpoint_a = [
                                    endpoint_factor[k][0] * cov_row[c],
                                    endpoint_factor[k][1] * cov_row[c],
                                ];
                                let endpoint_psi_a = [
                                    endpoint_psi_cov_factor[k][0] * cov_row[c]
                                        + endpoint_psi_psi_factor[k][0] * psi_row[c],
                                    endpoint_psi_cov_factor[k][1] * cov_row[c]
                                        + endpoint_psi_psi_factor[k][1] * psi_row[c],
                                ];
                                score_axis[idx] += wi
                                    * (h_a * h_psi + hi * hpsi_a - hppsi_a * inv_hp
                                        + hp_psi * hp_a * inv_hp_sq
                                        + endpoint_chain_second(
                                            q,
                                            endpoint_psi,
                                            endpoint_a,
                                            endpoint_psi_a,
                                        ));
                            }
                        }
                    }
                }
                Ok(acc)
            })
            .collect();
        let mut accum = PsiAllAxesAccum::new(n_psi, p_total);
        for partial in partials {
            accum = accum.merge(partial?);
        }

        // Stash the cached numeric data plus per-axis operator handles. The
        // matrix-free `TransformationNormalPsiHessianOperator` is reconstructed
        // on each `first_order_terms()` call from the cached Arc-shared row
        // state — Arc clones are O(1) and the cached operator instance carries
        // no per-evaluation mutable state.
        let PsiAllAxesAccum {
            objective_psi,
            mut score_psi,
        } = accum;
        let beta_arc = Arc::new(beta.clone());
        let trace_axes = Arc::new(axes.clone());
        let mut out: Vec<TransformationNormalPsiWorkspaceCacheEntry> = Vec::with_capacity(n_psi);
        for (axis_idx, &axis) in axes.iter().enumerate() {
            // Take the per-axis score buffer out of the accumulator without
            // cloning. The order matches the construction order so each axis
            // is consumed exactly once.
            let score_axis = std::mem::replace(&mut score_psi[axis_idx], Array1::<f64>::zeros(0));
            out.push(TransformationNormalPsiWorkspaceCacheEntry {
                objective_psi: objective_psi[axis_idx],
                score_psi: score_axis,
                op_arc: Arc::clone(&op_arcs[axis_idx]),
                axis,
                trace_axes: Arc::clone(&trace_axes),
                trace_axis_pos: axis_idx,
                row_gamma: Arc::clone(&row.gamma),
                row_h: Arc::clone(&row.h),
                row_h_prime: Arc::clone(&row.h_prime),
                endpoint_q: Arc::clone(&row.endpoint_q),
                beta: Arc::clone(&beta_arc),
            });
        }
        Ok(out)
    }

    pub(crate) fn axis_snapshots(
        &self,
    ) -> Result<Vec<TransformationNormalPsiWorkspaceAxisSnapshot>, String> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| "TransformationNormalPsiWorkspace cache poisoned".to_string())?;
        if guard.is_none() {
            let computed = self.compute_all_axes()?;
            *guard = Some(computed);
        }
        let cached = guard.as_ref().expect("populated above");
        Ok(cached
            .iter()
            .map(|entry| TransformationNormalPsiWorkspaceAxisSnapshot {
                op_arc: Arc::clone(&entry.op_arc),
                axis: entry.axis,
                row_gamma: Arc::clone(&entry.row_gamma),
                row_h: Arc::clone(&entry.row_h),
                row_h_prime: Arc::clone(&entry.row_h_prime),
                endpoint_q: Arc::clone(&entry.endpoint_q),
                beta: Arc::clone(&entry.beta),
            })
            .collect())
    }

    pub(crate) fn compute_pair_cache(
        &self,
    ) -> Result<Vec<Vec<Option<Arc<TransformationNormalPsiWorkspacePairCacheEntry>>>>, String> {
        let axes = self.axis_snapshots()?;
        let n_psi = axes.len();
        if n_psi == 0 {
            return Ok(Vec::new());
        }

        let pair_count = n_psi * (n_psi + 1) / 2;
        let pair_from_index = |pair_idx: usize| -> (usize, usize) {
            let span = 2 * n_psi + 1;
            let discriminant = span * span - 8 * pair_idx;
            let row = ((span as f64 - (discriminant as f64).sqrt()) * 0.5) as usize;
            let row_start = row * (2 * n_psi - row + 1) / 2;
            (row, row + pair_idx - row_start)
        };
        let trace_axes = Arc::new(axes.iter().map(|entry| entry.axis).collect::<Vec<_>>());

        let op = axes[0]
            .op_arc
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .ok_or_else(|| {
                "TransformationNormalPsiWorkspace psi-psi pair cache requires tensor-backed operator"
                    .to_string()
            })?;
        for (psi_index, entry) in axes.iter().enumerate() {
            if entry
                .op_arc
                .as_any()
                .downcast_ref::<TensorKroneckerPsiOperator>()
                .is_none()
            {
                return Err(TransformationNormalError::InvalidInput { reason: format!(
                    "TransformationNormalPsiWorkspace psi-psi pair cache requires tensor-backed operator at axis {psi_index}"
                ) }.into());
            }
        }

        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let p_total = self.family.response_val_basis.ncols() * p_cov;
        let policy = ResourcePolicy::default_library();
        let rows_per_chunk = crate::resource::rows_for_target_bytes(
            policy.row_chunk_target_bytes,
            p_cov.saturating_mul(n_psi + 2).max(1),
        )
        .max(1)
        .min(n.max(1));

        struct PsiPairCacheAccum {
            pub(crate) objective: f64,
            pub(crate) score: Array1<f64>,
        }

        impl PsiPairCacheAccum {
            pub(crate) fn new(p_total: usize) -> Self {
                Self {
                    objective: 0.0,
                    score: Array1::<f64>::zeros(p_total),
                }
            }
        }

        let mut accum: Vec<PsiPairCacheAccum> = (0..pair_count)
            .map(|_| PsiPairCacheAccum::new(p_total))
            .collect();
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .map_err(|e| {
                    format!(
                        "TransformationNormalPsiWorkspace psi-psi pair cache covariate chunk {start}..{end}: {e}"
                    )
                })?;
            let mut cov_psi_chunks: Vec<Array2<f64>> = Vec::with_capacity(n_psi);
            for (psi_index, entry) in axes.iter().enumerate() {
                let cov_psi = op
                    .cov_first_axis_row_chunk_streaming(entry.axis, rows.clone())
                    .map_err(|e| {
                        format!(
                            "TransformationNormalPsiWorkspace psi-psi pair cache first-axis chunk \
                             psi_index={psi_index}, axis={} rows {start}..{end}: {e}",
                            entry.axis
                        )
                    })?;
                if cov_psi.nrows() != end - start || cov_psi.ncols() != p_cov {
                    return Err(TransformationNormalError::InvalidInput { reason: format!(
                        "TransformationNormalPsiWorkspace psi-psi pair cache first-axis chunk shape {}x{} \
                         for psi_index={psi_index}, axis={} rows {start}..{end} != expected {}x{}",
                        cov_psi.nrows(),
                        cov_psi.ncols(),
                        entry.axis,
                        end - start,
                        p_cov
                    ) }.into());
                }
                cov_psi_chunks.push(cov_psi);
            }

            for pair_idx in 0..pair_count {
                let (psi_i, psi_j) = pair_from_index(pair_idx);
                let entry_i = &axes[psi_i];
                let entry_j = &axes[psi_j];
                let cov_ij = op
                    .cov_second_axis_row_chunk(entry_i.axis, entry_j.axis, rows.clone())
                    .map_err(|e| {
                        format!(
                            "TransformationNormalPsiWorkspace psi-psi pair cache second-axis chunk \
                             pair=({psi_i},{psi_j}), axes=({}, {}) rows {start}..{end}: {e}",
                            entry_i.axis, entry_j.axis
                        )
                    })?;
                if cov_ij.nrows() != end - start || cov_ij.ncols() != p_cov {
                    return Err(TransformationNormalError::InvalidInput { reason: format!(
                        "TransformationNormalPsiWorkspace psi-psi pair cache second-axis chunk shape {}x{} \
                         for pair=({psi_i},{psi_j}), axes=({}, {}) rows {start}..{end} != expected {}x{}",
                        cov_ij.nrows(),
                        cov_ij.ncols(),
                        entry_i.axis,
                        entry_j.axis,
                        end - start,
                        p_cov
                    ) }.into());
                }
                let (objective_chunk, score_chunk, _) =
                    self.family.scop_psi_psi_value_score_hvp_from_cov(
                        entry_i.beta.as_ref(),
                        entry_i.row_gamma.slice(s![start..end, ..]),
                        entry_i.row_h.slice(s![start..end]),
                        entry_i.row_h_prime.slice(s![start..end]),
                        cov.view(),
                        cov_psi_chunks[psi_i].view(),
                        cov_psi_chunks[psi_j].view(),
                        cov_ij.view(),
                        start,
                        &entry_i.endpoint_q[start..end],
                        None,
                    )?;
                accum[pair_idx].objective += objective_chunk;
                accum[pair_idx].score.scaled_add(1.0, &score_chunk);
            }
        }

        let mut table = vec![vec![None; n_psi]; n_psi];
        for (pair_idx, acc) in accum.into_iter().enumerate() {
            let (i, j) = pair_from_index(pair_idx);
            if !acc.objective.is_finite() || !acc.score.iter().all(|v: &f64| v.is_finite()) {
                return Err(TransformationNormalError::NonFinite { reason: format!(
                    "TransformationNormalPsiWorkspace psi-psi pair cache produced non-finite values at \
                     psi_i={i}, psi_j={j}: obj_finite={}, score_all_finite={}",
                    acc.objective.is_finite(),
                    acc.score.iter().all(|v: &f64| v.is_finite()),
                ) }.into());
            }
            let entry_i = &axes[i];
            let entry_j = &axes[j];
            let entry = Arc::new(TransformationNormalPsiWorkspacePairCacheEntry {
                objective_psi_psi: acc.objective,
                score_psi_psi: acc.score,
                op_arc: Arc::clone(&entry_i.op_arc),
                axis_i: entry_i.axis,
                axis_j: entry_j.axis,
                trace_axes: Arc::clone(&trace_axes),
                trace_axis_i_pos: i,
                trace_axis_j_pos: j,
                row_gamma: Arc::clone(&entry_i.row_gamma),
                row_h: Arc::clone(&entry_i.row_h),
                row_h_prime: Arc::clone(&entry_i.row_h_prime),
                endpoint_q: Arc::clone(&entry_i.endpoint_q),
                beta: Arc::clone(&entry_i.beta),
            });
            table[i][j] = Some(Arc::clone(&entry));
            table[j][i] = Some(entry);
        }
        Ok(table)
    }
}

impl ExactNewtonJointPsiWorkspace for TransformationNormalPsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| "TransformationNormalPsiWorkspace cache poisoned".to_string())?;
        if guard.is_none() {
            let computed = self.compute_all_axes()?;
            *guard = Some(computed);
        }
        let cached = guard.as_ref().expect("populated above");
        if psi_index >= cached.len() {
            return Ok(None);
        }
        let entry = &cached[psi_index];
        // Reconstruct the matrix-free first-order Hessian operator on each
        // call. Arc-cloning shared row state and `op_arc` is O(1); the
        // operator carries no mutable per-evaluation state. The cached
        // numeric `score_psi` buffer is cloned because `ExactNewtonJointPsiTerms`
        // is not `Clone`-derivable through the `dyn HyperOperator` field.
        let hessian_psi_operator: Arc<dyn HyperOperator> =
            Arc::new(TransformationNormalPsiHessianOperator::new_with_trace_axes(
                Arc::new(self.family.clone()),
                (*entry.beta).clone(),
                Arc::clone(&entry.op_arc),
                entry.axis,
                Arc::clone(&entry.trace_axes),
                entry.trace_axis_pos,
                Arc::clone(&entry.row_gamma),
                Arc::clone(&entry.row_h),
                Arc::clone(&entry.row_h_prime),
                Arc::clone(&entry.endpoint_q),
            ));
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi: entry.objective_psi,
            score_psi: entry.score_psi.clone(),
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(hessian_psi_operator),
        }))
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let start = std::time::Instant::now();
        let entry = {
            let mut guard = self
                .pair_cache
                .lock()
                .map_err(|_| "TransformationNormalPsiWorkspace pair cache poisoned".to_string())?;
            if guard.is_none() {
                let computed = self.compute_pair_cache()?;
                *guard = Some(computed);
            }
            let cached = guard.as_ref().expect("populated above");
            if psi_i >= cached.len() || psi_j >= cached.len() {
                return Ok(None);
            }
            cached[psi_i][psi_j].as_ref().map(Arc::clone)
        };
        let Some(entry) = entry else {
            return Ok(None);
        };

        let hessian_psi_psi_operator: Box<dyn HyperOperator> = Box::new(
            TransformationNormalPsiPsiHessianOperator::new_with_trace_axes(
                Arc::new(self.family.clone()),
                entry.beta.as_ref().clone(),
                Arc::clone(&entry.op_arc),
                entry.axis_i,
                entry.axis_j,
                Arc::clone(&entry.trace_axes),
                entry.trace_axis_i_pos,
                entry.trace_axis_j_pos,
                Arc::clone(&entry.row_gamma),
                Arc::clone(&entry.row_h),
                Arc::clone(&entry.row_h_prime),
                Arc::clone(&entry.endpoint_q),
            ),
        );
        log::info!(
            "[STAGE] CTN psi-psi workspace pair (psi_i={}, psi_j={}, axes={},{}) elapsed={:.3}s",
            psi_i,
            psi_j,
            entry.axis_i,
            entry.axis_j,
            start.elapsed().as_secs_f64(),
        );
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi: entry.objective_psi_psi,
            score_psi_psi: entry.score_psi_psi.clone(),
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(hessian_psi_psi_operator),
        }))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| "TransformationNormalPsiWorkspace cache poisoned".to_string())?;
        if guard.is_none() {
            let computed = self.compute_all_axes()?;
            *guard = Some(computed);
        }
        let cached = guard.as_ref().expect("populated above");
        if psi_index >= cached.len() {
            return Ok(None);
        }
        let entry = &cached[psi_index];
        if d_beta_flat.len() != entry.beta.len() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "TransformationNormalPsiWorkspace psi dH direction length {} != expected {}",
                    d_beta_flat.len(),
                    entry.beta.len()
                ),
            }
            .into());
        }
        let row_quantities = TransformationNormalRowQuantityCache {
            beta: Arc::clone(&entry.beta),
            gamma: Arc::clone(&entry.row_gamma),
            h: Arc::clone(&entry.row_h),
            h_prime: Arc::clone(&entry.row_h_prime),
            h_lower: Arc::new(Array1::zeros(entry.row_h.len())),
            h_upper: Arc::new(Array1::zeros(entry.row_h.len())),
            endpoint_q: Arc::clone(&entry.endpoint_q),
            log_likelihood: 0.0,
        };
        let op = TransformationNormalPsiDhMatrixFreeOperator::new(
            Arc::new(self.family.clone()),
            entry.beta.as_ref().clone(),
            d_beta_flat.clone(),
            Arc::clone(&entry.op_arc),
            entry.axis,
            row_quantities,
        );
        Ok(Some(DriftDerivResult::Operator(Arc::new(op))))
    }
}

pub(crate) fn extract_covariate_penalty_factor(
    penalty: &PenaltyMatrix,
) -> Result<Array2<f64>, String> {
    match penalty {
        PenaltyMatrix::Dense(matrix) => Ok(matrix.clone()),
        PenaltyMatrix::Blockwise { .. } => Ok(penalty.to_dense()),
        PenaltyMatrix::Labeled { inner, .. } => extract_covariate_penalty_factor(inner),
        PenaltyMatrix::Fixed { inner, .. } => extract_covariate_penalty_factor(inner),
        PenaltyMatrix::KroneckerFactored { .. } => Err(
            "transformation covariate psi penalties must be single-block, not already Kronecker-factored"
                .to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Psi derivative builder (for κ optimization of the covariate basis)
// ---------------------------------------------------------------------------

/// Build `CustomFamilyBlockPsiDerivative` objects for the tensor product.
///
/// Given covariate-side psi derivatives (∂cov_design/∂ψ_a and ∂S_cov/∂ψ_a),
/// this constructs the corresponding tensor-product-space psi derivatives that
/// the REML evaluator needs.
///
/// Each output entry contains:
/// - implicit `x_psi` / `x_psi_psi` operators that preserve Kronecker structure
/// - factored tensor penalty derivatives matching the SCOP penalty layout:
///   `E_shape ⊗ ∂S_cov/∂ψ` at the same covariate penalty index `m`.
pub fn build_tensor_psi_derivatives(
    family: &TransformationNormalFamily,
    covariate_psi_derivs: &[CustomFamilyBlockPsiDerivative],
) -> Result<Vec<CustomFamilyBlockPsiDerivative>, String> {
    let p_resp = family.response_val_basis.ncols();
    let n_axes = covariate_psi_derivs.len();
    let mut shape_resp = Array2::<f64>::eye(p_resp);
    shape_resp[[0, 0]] = 0.0;
    let shared_operator: Arc<dyn CustomFamilyPsiDerivativeOperator> =
        Arc::new(TensorKroneckerPsiOperator {
            response_val_basis: Arc::new(family.response_val_basis.clone()),
            covariate_design: family.covariate_design.clone(),
            covariate_derivs: covariate_psi_derivs.to_vec(),
            covariate_first_cache: Arc::new(
                (0..n_axes).map(|_| Mutex::new(None)).collect::<Vec<_>>(),
            ),
        });

    let mut derivs = Vec::with_capacity(n_axes);
    for a in 0..n_axes {
        let cov_deriv = &covariate_psi_derivs[a];
        let s_psi_penalty_components = cov_deriv
            .s_psi_penalty_components
            .as_ref()
            .map(|components| lift_covariate_penalty_derivative_components(components, &shape_resp))
            .transpose()?
            .or_else(|| {
                cov_deriv.s_psi_components.as_ref().map(|components| {
                    lift_dense_covariate_penalty_derivative_components(components, &shape_resp)
                })
            });
        let s_psi_psi_penalty_components = cov_deriv
            .s_psi_psi_penalty_components
            .as_ref()
            .map(|rows| {
                rows.iter()
                    .map(|cov_pen_pairs| -> Result<_, String> {
                        lift_covariate_penalty_derivative_components(cov_pen_pairs, &shape_resp)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .or_else(|| {
                cov_deriv.s_psi_psi_components.as_ref().map(|rows| {
                    rows.iter()
                        .map(|cov_pen_pairs| {
                            lift_dense_covariate_penalty_derivative_components(
                                cov_pen_pairs,
                                &shape_resp,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            });

        let mut deriv = CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::<f64>::zeros((0, 0)),
            Array2::<f64>::zeros((0, 0)),
            None,
            None,
            None,
            None,
        );
        deriv.s_psi_penalty_components = s_psi_penalty_components;
        deriv.s_psi_psi_penalty_components = s_psi_psi_penalty_components;
        deriv.implicit_operator = Some(Arc::clone(&shared_operator));
        deriv.implicit_axis = a;
        deriv.implicit_group_id = Some(0);
        derivs.push(deriv);
    }

    Ok(derivs)
}

pub(crate) fn lift_dense_covariate_penalty_derivative_components(
    components: &[(usize, Array2<f64>)],
    shape_resp: &Array2<f64>,
) -> Vec<(usize, PenaltyMatrix)> {
    let mut out = Vec::with_capacity(components.len());
    for &(idx, ref ds_cov) in components {
        push_lifted_covariate_penalty_component(&mut out, idx, ds_cov.clone(), shape_resp);
    }
    out
}

pub(crate) fn lift_covariate_penalty_derivative_components(
    components: &[(usize, PenaltyMatrix)],
    shape_resp: &Array2<f64>,
) -> Result<Vec<(usize, PenaltyMatrix)>, String> {
    let mut out = Vec::with_capacity(components.len());
    for (idx, ds_cov) in components {
        push_lifted_covariate_penalty_component(
            &mut out,
            *idx,
            extract_covariate_penalty_factor(ds_cov)?,
            shape_resp,
        );
    }
    Ok(out)
}

pub(crate) fn push_lifted_covariate_penalty_component(
    out: &mut Vec<(usize, PenaltyMatrix)>,
    cov_penalty_idx: usize,
    ds_cov: Array2<f64>,
    shape_resp: &Array2<f64>,
) {
    out.push((
        cov_penalty_idx,
        PenaltyMatrix::KroneckerFactored {
            left: shape_resp.clone(),
            right: ds_cov,
        },
    ));
}

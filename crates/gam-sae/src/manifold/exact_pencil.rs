// Included by construction_exact_hessian.rs. This owner uses the actual SPD
// Newton metric to resolve curvature and the physical Euclidean quotient
// measure to price it. Every derivative is on a fixed rank stratum.

impl ExactHessianSpectralBlock {
    /// Resolution in the normalized Newton metric is uniform across directions.
    fn rank_floor(&self) -> f64 {
        sae_exact_a_direction_floor(self.eigenvalues.len(), self.spectral_norm, 1.0)
    }

    fn from_operators(operator: Array2<f64>, metric: Array2<f64>) -> Result<Self, String> {
        let dim = operator.nrows();
        if operator.ncols() != dim || metric.dim() != (dim, dim)
            || !operator.iter().chain(metric.iter()).all(|x| x.is_finite())
        {
            return Err("exact-A pencil requires equally sized finite symmetric operators".into());
        }
        if dim == 0 {
            return Ok(Self {
                operator, eigenvalues: Array1::zeros(0), eigenvectors: Array2::zeros((0, 0)),
                lower: Array2::zeros((0, 0)), whitened_operator: Array2::zeros((0, 0)),
                spectral_norm: 0.0,
            });
        }
        let factor = metric.cholesky(Side::Lower)
            .map_err(|e| format!("exact-A pencil metric is not positive definite: {e}"))?;
        let lower = factor.lower_triangular();
        let whitened_operator = Self::whiten(&lower, &operator);
        let (eigenvalues, orthogonal_vectors) = whitened_operator.eigh(Side::Lower)
            .map_err(|e| format!("exact-A generalized eigendecomposition: {e}"))?;
        let eigenvectors = Self::solve_lower_transpose(&lower, &orthogonal_vectors);
        let spectral_norm = eigenvalues.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if !eigenvalues.iter().chain(eigenvectors.iter()).all(|x| x.is_finite()) {
            return Err("exact-A pencil produced non-finite spectral geometry".into());
        }
        Ok(Self { operator, eigenvalues, eigenvectors, lower, whitened_operator, spectral_norm })
    }

    fn whiten(lower: &Array2<f64>, matrix: &Array2<f64>) -> Array2<f64> {
        use gam_linalg::triangular::forward_substitution_lower_matrix;
        let left = forward_substitution_lower_matrix(lower, matrix);
        let mut result = forward_substitution_lower_matrix(lower, left.t()).reversed_axes();
        for i in 0..result.nrows() {
            for j in 0..i {
                let value = 0.5 * (result[[i, j]] + result[[j, i]]);
                result[[i, j]] = value;
                result[[j, i]] = value;
            }
        }
        result
    }

    fn solve_lower_transpose(lower: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::zeros(rhs.raw_dim());
        for column in 0..rhs.ncols() {
            out.column_mut(column).assign(
                &gam_linalg::triangular::back_substitution_lower_transpose(lower, rhs.column(column)),
            );
        }
        out
    }

    fn unwhiten_derivative(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let left = Self::solve_lower_transpose(&self.lower, matrix);
        Self::solve_lower_transpose(&self.lower, &left.t().to_owned()).reversed_axes()
    }

    fn retained_indices(&self) -> Vec<usize> {
        (0..self.eigenvalues.len())
            .filter(|&i| self.eigenvalues[i].abs() > self.rank_floor()).collect()
    }

    fn physical_basis(&self, indices: &[usize]) -> Array2<f64> {
        Array2::from_shape_fn((self.eigenvalues.len(), indices.len()), |(i, j)| {
            self.eigenvectors[[i, indices[j]]]
        })
    }

    /// Orthogonal projection in physical coordinates. Q is B-orthonormal, so
    /// Q Q' would not be a projector. The small Gram solve is essential.
    fn project_physical(&self, indices: &[usize], vector: &Array1<f64>) -> Result<Array1<f64>, String> {
        if indices.is_empty() { return Ok(Array1::zeros(vector.len())); }
        let basis = self.physical_basis(indices);
        let gram = basis.t().dot(&basis);
        let factor = gram.cholesky(Side::Lower)
            .map_err(|e| format!("exact-A physical quotient Gram factor: {e}"))?;
        let rhs = basis.t().dot(vector).insert_axis(ndarray::Axis(1));
        Ok(basis.dot(&factor.solve_mat(&rhs)).column(0).to_owned())
    }
}

impl SaeManifoldTerm {
    fn classify_exact_hessian_basin(
        block: &ExactHessianSpectralBlock,
        e_diag: &Array1<f64>, total_t: usize, label: &'static str,
    ) -> Result<ExactHessianBasin, SaeCriterionError> {
        let dim = block.eigenvalues.len();
        if total_t > dim || e_diag.len() < total_t {
            return Err(SaeCriterionError::Numerical("exact-A clamp dimensions disagree".into()));
        }
        let retained = block.retained_indices();
        let physical_basis = block.physical_basis(&retained);
        let gram = physical_basis.t().dot(&physical_basis);
        let (gram_lower, mut log_det) = if retained.is_empty() {
            (Array2::zeros((0, 0)), 0.0)
        } else {
            let factor = gram.cholesky(Side::Lower)
                .map_err(|e| format!("exact-A quotient volume factor: {e}"))?;
            let lower = factor.lower_triangular();
            let log_volume = -2.0 * lower.diag().iter().map(|x| x.ln()).sum::<f64>();
            (lower, log_volume)
        };
        let orthogonal_vectors = block.lower.t().dot(&block.eigenvectors);
        let negative: Vec<_> = (0..dim).filter(|&i| block.eigenvalues[i] < -block.rank_floor()).collect();
        let complement: Vec<_> = (0..dim).filter(|&i| block.eigenvalues[i] >= -block.rank_floor()).collect();
        for &i in &complement {
            if block.eigenvalues[i] > block.rank_floor() { log_det += block.eigenvalues[i].ln(); }
        }
        let mut physical_clamp = Array2::zeros((dim, dim));
        for i in 0..total_t { physical_clamp[[i, i]] = e_diag[i]; }
        let clamp = ExactHessianSpectralBlock::whiten(&block.lower, &physical_clamp);
        let q = negative.len();
        let basis = Array2::from_shape_fn((dim, q), |(i, j)| orthogonal_vectors[[i, negative[j]]]);
        let mut basin = basis.t().dot(&clamp).dot(&basis);
        for (i, &index) in negative.iter().enumerate() { basin[[i, i]] += block.eigenvalues[index]; }
        let (values, rotation) = if q == 0 {
            (Array1::zeros(0), Array2::zeros((0, 0)))
        } else {
            basin.eigh(Side::Lower).map_err(|e| format!("exact-A basin eigendecomposition: {e}"))?
        };
        let vectors = basis.dot(&rotation);
        // Frobenius norm bounds the clamp operator norm. Cancellation requires
        // the arithmetic certificate to include both assembled operands.
        let clamp_norm = clamp.iter().fold(0.0_f64, |norm, &x| norm.hypot(x));
        let floor = sae_exact_a_direction_floor(dim, block.spectral_norm + clamp_norm, 1.0);
        let mut inverse_values = Array1::zeros(q);
        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(SaeCriterionError::Numerical("exact-A basin curvature is non-finite".into()));
            }
            if value < -floor {
                return Err(SaeCriterionError::IndefiniteObservedInformation { block: label });
            }
            if value > floor { log_det += value.ln(); inverse_values[i] = value.recip(); }
        }
        Ok(ExactHessianBasin {
            log_det, negative, complement, basis, rotation, vectors, inverse_values,
            retained, physical_basis, gram_lower, orthogonal_vectors, clamp,
        })
    }

    fn exact_hessian_basin_differential(
        block: &ExactHessianSpectralBlock, total_t: usize, basin: &ExactHessianBasin,
    ) -> Result<ExactHessianPricing, SaeCriterionError> {
        let dim = block.eigenvalues.len();
        let q = basin.negative.len();
        let mut s = Array2::<f64>::zeros((dim, dim));
        let mut t = Array2::<f64>::zeros((dim, dim));
        for &i in &basin.complement {
            if block.eigenvalues[i] <= block.rank_floor() { continue; }
            let u = basin.orthogonal_vectors.column(i);
            for r in 0..dim { for c in 0..dim { s[[r,c]] += u[r] * u[c] / block.eigenvalues[i]; } }
        }
        let mut inverse = Array2::<f64>::zeros((q,q));
        for (i, &weight) in basin.inverse_values.iter().enumerate() {
            let v = basin.vectors.column(i);
            for r in 0..dim { for c in 0..dim { t[[r,c]] += weight*v[r]*v[c]; } }
            for r in 0..q { for c in 0..q { inverse[[r,c]] += weight*basin.rotation[[r,i]]*basin.rotation[[c,i]]; } }
        }
        s += &t;
        if q > 0 && !basin.complement.is_empty() {
            let other = Array2::from_shape_fn((dim, basin.complement.len()), |(i,j)| {
                basin.orthogonal_vectors[[i,basin.complement[j]]]
            });
            let e_cross = basin.basis.t().dot(&basin.clamp).dot(&other);
            let mut response = inverse.dot(&e_cross);
            for (i, &negative) in basin.negative.iter().enumerate() {
                for (j, &complement) in basin.complement.iter().enumerate() {
                    response[[i,j]] /= block.eigenvalues[negative]-block.eigenvalues[complement];
                }
            }
            let cross = basin.basis.dot(&response).dot(&other.t());
            s += &(&cross + &cross.t());
        }
        // Reverse the physical quotient volume -logdet(Q'Q). Rotations inside
        // the retained subspace cancel, including repeated eigenvalues. Only
        // retained/null gaps enter its projector derivative.
        let gram_inverse = gam_linalg::triangular::cholesky_solve_matrix(
            &basin.gram_lower, &Array2::<f64>::eye(basin.retained.len()),
        );
        let q_ginv = basin.physical_basis.dot(&gram_inverse);
        let h = q_ginv.mapv(|x| -2.0*x);
        let grad_u = gam_linalg::triangular::forward_substitution_lower_matrix(&block.lower, &h);
        let null: Vec<_> = (0..dim).filter(|&i| block.eigenvalues[i].abs() <= block.rank_floor()).collect();
        if !null.is_empty() && !basin.retained.is_empty() {
            let null_basis = Array2::from_shape_fn((dim,null.len()), |(i,j)| basin.orthogonal_vectors[[i,null[j]]]);
            let mut response = null_basis.t().dot(&grad_u);
            for (i,&discarded) in null.iter().enumerate() {
                for (j,&kept) in basin.retained.iter().enumerate() {
                    response[[i,j]] /= 2.0*(block.eigenvalues[kept]-block.eigenvalues[discarded]);
                }
            }
            let kept_basis = Array2::from_shape_fn((dim,basin.retained.len()), |(i,j)| basin.orthogonal_vectors[[i,basin.retained[j]]]);
            let cross = null_basis.dot(&response).dot(&kept_basis.t());
            s += &(&cross + &cross.t());
        }
        let a_derivative = block.unwhiten_derivative(&s);
        let e_derivative = block.unwhiten_derivative(&t);
        let sc_tf = s.dot(&block.whitened_operator) + t.dot(&basin.clamp);
        let mut grad_lower = ExactHessianSpectralBlock::solve_lower_transpose(&block.lower, &sc_tf).mapv(|x| -2.0*x);
        let volume_matrix = q_ginv.dot(&basin.physical_basis.t()).mapv(|x| 2.0*x);
        let volume_lower = gam_linalg::triangular::forward_substitution_lower_matrix(
            &block.lower, volume_matrix.t(),
        ).reversed_axes();
        grad_lower += &volume_lower;
        let w = block.lower.t().dot(&grad_lower);
        let mut j = Array2::zeros((dim,dim));
        for r in 0..dim { for c in 0..=r { j[[r,c]]=0.5*w[[r,c]]; j[[c,r]]=j[[r,c]]; } }
        let b_derivative = block.unwhiten_derivative(&j);
        if !a_derivative.iter().chain(b_derivative.iter()).chain(e_derivative.iter()).all(|x| x.is_finite()) {
            return Err(SaeCriterionError::Numerical("exact-A pencil differential is non-finite".into()));
        }
        Ok(ExactHessianPricing {
            a_derivative, b_derivative,
            clamp_diagonal_derivative: Array1::from_iter((0..total_t).map(|i| e_derivative[[i,i]])),
        })
    }
}

#[cfg(test)]
mod pencil_tests {
    use super::*;

    fn price(a: &Array2<f64>, b: &Array2<f64>, e: &Array1<f64>) -> (f64, ExactHessianPricing) {
        let block = ExactHessianSpectralBlock::from_operators(a.clone(), b.clone())
            .expect("finite symmetric pencil with positive metric");
        let basin = SaeManifoldTerm::classify_exact_hessian_basin(&block, e, e.len(), "pencil control")
            .expect("positive priced basin");
        let derivative = SaeManifoldTerm::exact_hessian_basin_differential(&block, e.len(), &basin)
            .expect("fixed-stratum pencil differential");
        (basin.log_det, derivative)
    }

    fn max_abs(matrix: &Array2<f64>) -> f64 {
        matrix.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn fully_degenerate_cluster_diagonalizes_direct_e_diag_2267() {
        let dim = 24;
        let a = Array2::<f64>::eye(dim) * -3.0;
        let e = Array1::from_shape_fn(dim, |i| 4.0 + 0.5 * i as f64);
        let (value, derivative) = price(&a, &Array2::eye(dim), &e);
        let expected = e.iter().map(|x| (x - 3.0).ln()).sum::<f64>();
        assert!((value - expected).abs() < 1.0e-11);
        let inverse = Array2::from_diag(&e.mapv(|x| (x - 3.0).recip()));
        assert!(max_abs(&(&derivative.a_derivative - &inverse)) < 1.0e-12);
        assert!(max_abs(&derivative.b_derivative) < 1.0e-12);
        assert!((&derivative.clamp_diagonal_derivative - &inverse.diag())
            .iter().all(|x| x.abs() < 1.0e-12));
    }

    #[test]
    fn nearly_degenerate_distinct_spectrum_preserves_eigenpairs_2515() {
        let q = 0.5_f64.sqrt();
        let rotation = ndarray::array![[q, -q], [q, q]];
        let diagonal = Array2::from_diag(&ndarray::array![3.0, 3.0 + 1.0e-9]);
        let a = rotation.dot(&diagonal).dot(&rotation.t());
        let block = ExactHessianSpectralBlock::from_operators(a.clone(), Array2::eye(2))
            .expect("near-degenerate positive pencil");
        let residual = a.dot(&block.eigenvectors)
            - block.eigenvectors.dot(&Array2::from_diag(&block.eigenvalues));
        assert!(max_abs(&residual) < 64.0 * f64::EPSILON * (3.0 + 1.0e-9));
        let (value, derivative) = price(&a, &Array2::eye(2), &ndarray::array![1.0, 2.0]);
        assert!((value - diagonal.diag().iter().map(|x| x.ln()).sum::<f64>()).abs() < 1.0e-12);
        let inverse = rotation.dot(&diagonal.mapv(|x| if x == 0.0 { 0.0 } else { x.recip() })).dot(&rotation.t());
        assert!(max_abs(&(&derivative.a_derivative - &inverse)) < 1.0e-12);
    }

    #[test]
    fn full_rank_pencil_price_and_gradient_do_not_depend_on_metric_2820() {
        let b = ndarray::array![[2.0, 0.4], [0.4, 3.0]];
        for (a, e) in [
            (ndarray::array![[2.0, 0.3], [0.3, 1.0]], Array1::zeros(2)),
            (ndarray::array![[-1.0, 0.2], [0.2, -2.0]], Array1::from_vec(vec![4.0, 5.0])),
        ] {
            let (value, derivative) = price(&a, &b, &e);
            let priced = &a + &Array2::from_diag(&e);
            let determinant = priced[[0,0]]*priced[[1,1]] - priced[[0,1]]*priced[[1,0]];
            let inverse = ndarray::array![[priced[[1,1]], -priced[[0,1]]], [-priced[[1,0]], priced[[0,0]]]] / determinant;
            assert!((value-determinant.ln()).abs() < 2.0e-13);
            assert!(max_abs(&(&derivative.a_derivative-&inverse)) < 2.0e-13);
            assert!(max_abs(&derivative.b_derivative) < 2.0e-13,
                "B must cancel from the full-rank physical determinant: {:?}", derivative.b_derivative);
        }
    }

    #[test]
    fn deflated_pencil_value_and_all_operand_derivatives_match_finite_differences_2820() {
        let a = ndarray::array![[1.5, 1.5], [1.5, 1.5 + 1.0e-12]];
        let b = ndarray::array![[2.0, 0.3], [0.3, 1.0]];
        let e = Array1::zeros(2);
        let (value, derivative) = price(&a, &b, &e);
        assert!(max_abs(&derivative.b_derivative) > 1.0e-2,
            "the physical retained span must have measurable B sensitivity");
        // Perturb A along its rank-one support: an arbitrary ambient A
        // perturbation would cross the deliberately narrow rank band.
        let da = ndarray::array![[0.2, 0.2], [0.2, 0.2]];
        let db = ndarray::array![[0.2, -0.1], [-0.1, 0.3]];
        let h = 1.0e-5;
        for (left, right, analytic) in [
            (price(&(&a+&(&da*h)), &b, &e).0, price(&(&a-&(&da*h)), &b, &e).0, (&derivative.a_derivative*&da).sum()),
            (price(&a, &(&b+&(&db*h)), &e).0, price(&a, &(&b-&(&db*h)), &e).0, (&derivative.b_derivative*&db).sum()),
        ] {
            let fd = (left-right)/(2.0*h);
            assert!((fd-analytic).abs() < 2.0e-8*(1.0+fd.abs()), "FD={fd:e}, analytic={analytic:e}");
        }
        let angle = 0.61_f64;
        let r = ndarray::array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
        let (rotated_value, rotated_derivative) = price(&r.t().dot(&a).dot(&r), &r.t().dot(&b).dot(&r), &e);
        assert!((value-rotated_value).abs() < 2.0e-13);
        assert!(max_abs(&(rotated_derivative.a_derivative-r.t().dot(&derivative.a_derivative).dot(&r))) < 2.0e-12);
        assert!(max_abs(&(rotated_derivative.b_derivative-r.t().dot(&derivative.b_derivative).dot(&r))) < 2.0e-12);
    }

    #[test]
    fn pencil_stationarity_solves_and_certifies_the_physical_retained_span_2820() {
        let a = ndarray::array![[1.5, 1.5], [1.5, 1.5 + 1.0e-12]];
        let b = ndarray::array![[2.0, 0.3], [0.3, 1.0]];
        let block = ExactHessianSpectralBlock::from_operators(a.clone(), b)
            .expect("positive metric");
        let rhs = SaeArrowVector { t: ndarray::array![0.7], beta: ndarray::array![-0.4] };
        let solution = block.solve_stationarity(&rhs).expect("physical Galerkin certificate");
        let x = ndarray::array![solution.t[0], solution.beta[0]];
        let retained = block.retained_indices();
        assert_eq!(retained.len(), 1);
        let q = block.eigenvectors.column(retained[0]);
        let physical_rhs = ndarray::array![0.7, -0.4];
        assert!(q.dot(&(a.dot(&x)-&physical_rhs)).abs() < 2.0e-13);
        let projection = block.project_physical(&retained, &x).expect("physical range projector");
        assert!((&projection-&x).iter().all(|v| v.abs() < 2.0e-13));
        // Q Q' is deliberately not the Euclidean orthogonal projector.
        assert!((q.dot(&q)-1.0).abs() > 0.1);
    }
}

use super::*;
use crate::matrix::{
    CoefficientTransformOperator, DenseDesignMatrix, DenseRightProductView, DesignMatrix,
};

// ─── Binomial auxiliary terms for link-parameter ext_coord construction ───
//
// Mirrors the `BinomialAuxTerms` in estimate.rs but is local to the REML
// module so that hyper.rs can compute per-observation likelihood derivatives
// without depending on the estimate module's private helpers.

pub(crate) const LINK_BINOMIAL_AUX_MU_EPS: f64 = 1e-12;

#[derive(Clone, Copy)]
pub(crate) struct LinkBinomialAux {
    /// dℓ_i/dμ_i = w_i (y_i/μ_i − (1−y_i)/(1−μ_i))
    pub(crate) a1: f64,
    /// d²ℓ_i/dμ_i² = w_i (−y_i/μ_i² − (1−y_i)/(1−μ_i)²)
    pub(crate) a2: f64,
    /// μ(1−μ) — binomial variance function
    pub(crate) variance: f64,
    /// dVar/dμ = 1−2μ
    pub(crate) variancemu_scale: f64,
}

#[inline]
pub(crate) fn link_binomial_aux(yi: f64, wi: f64, mu: f64) -> LinkBinomialAux {
    let mu = if mu.is_finite() {
        mu.clamp(LINK_BINOMIAL_AUX_MU_EPS, 1.0 - LINK_BINOMIAL_AUX_MU_EPS)
    } else {
        0.5
    };
    let one_minusmu = 1.0 - mu;
    let a1 = wi * (yi / mu - (1.0 - yi) / one_minusmu);
    let a2 = wi * (-(yi / (mu * mu)) - (1.0 - yi) / (one_minusmu * one_minusmu));
    LinkBinomialAux {
        a1,
        a2,
        variance: mu * one_minusmu,
        variancemu_scale: 1.0 - 2.0 * mu,
    }
}

#[derive(Clone)]
enum TauTauDesignTerm {
    Dense(Array2<f64>),
    Implicit(HyperDesignDerivative),
}

#[derive(Clone)]
enum TauDesignTerm {
    Dense(Array2<f64>),
    Implicit(HyperDesignDerivative),
}

#[derive(Clone)]
enum TauPairBasis {
    Original,
    Transformed {
        qs: std::sync::Arc<Array2<f64>>,
        free_basis_opt: std::sync::Arc<Option<Array2<f64>>>,
    },
}

struct TauTauPairHyperOperator {
    x_tau_i: TauDesignTerm,
    x_tau_j: TauDesignTerm,
    x_tau_tau: Option<TauTauDesignTerm>,
    x_design: std::sync::Arc<DesignMatrix>,
    basis: TauPairBasis,
    w_diag: std::sync::Arc<Array1<f64>>,
    c_x_tau_i_beta: Option<Array1<f64>>,
    c_x_tau_j_beta: Option<Array1<f64>>,
    d_cross: Option<Array1<f64>>,
    c_xij_beta: Option<Array1<f64>>,
    s_tau_tau: Option<Array2<f64>>,
    p: usize,
}

fn build_active_design_matrix(
    x_transformed: &DesignMatrix,
    free_basis_opt: Option<&Array2<f64>>,
) -> Result<DesignMatrix, String> {
    match (x_transformed, free_basis_opt) {
        (DesignMatrix::Dense(dense), Some(z)) => {
            let op = CoefficientTransformOperator::new(dense.clone(), z.clone())?;
            Ok(DesignMatrix::Dense(DenseDesignMatrix::from(std::sync::Arc::new(
                op,
            ))))
        }
        (_, None) => Ok(x_transformed.clone()),
        (DesignMatrix::Sparse(_), Some(_)) => Err(
            "implicit hyper-operator requires a dense/operator-backed transformed design when an active free-basis projection is present".to_string(),
        ),
    }
}

impl super::unified::HyperOperator for TauTauPairHyperOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p);

        let x_v = self.x_design.dot(v);
        let mut out = Array1::<f64>::zeros(self.p);

        let tau_forward = |term: &TauDesignTerm, u: &Array1<f64>| -> Array1<f64> {
            match term {
                TauDesignTerm::Dense(dense) => dense.dot(u),
                TauDesignTerm::Implicit(deriv) => match &self.basis {
                    TauPairBasis::Original => deriv
                        .forward_mul_original(u)
                        .expect("tau pair operator original forward product should be shape-consistent"),
                    TauPairBasis::Transformed { qs, free_basis_opt } => deriv
                        .transformed_forward_mul(qs.as_ref(), free_basis_opt.as_ref().as_ref(), u)
                        .expect("tau pair operator transformed forward product should be shape-consistent"),
                },
            }
        };
        let tau_transpose = |term: &TauDesignTerm, y: &Array1<f64>| -> Array1<f64> {
            match term {
                TauDesignTerm::Dense(dense) => dense.t().dot(y),
                TauDesignTerm::Implicit(deriv) => match &self.basis {
                    TauPairBasis::Original => deriv
                        .transpose_mul_original(y)
                        .expect("tau pair operator original transpose product should be shape-consistent"),
                    TauPairBasis::Transformed { qs, free_basis_opt } => deriv
                        .transformed_transpose_mul(
                            qs.as_ref(),
                            free_basis_opt.as_ref().as_ref(),
                            y,
                        )
                        .expect("tau pair operator transformed transpose product should be shape-consistent"),
                },
            }
        };
        let tau_tau_forward = |term: &TauTauDesignTerm, u: &Array1<f64>| -> Array1<f64> {
            match term {
                TauTauDesignTerm::Dense(dense) => dense.dot(u),
                TauTauDesignTerm::Implicit(deriv) => match &self.basis {
                    TauPairBasis::Original => deriv
                        .forward_mul_original(u)
                        .expect("tau-tau pair operator original forward product should be shape-consistent"),
                    TauPairBasis::Transformed { qs, free_basis_opt } => deriv
                        .transformed_forward_mul(qs.as_ref(), free_basis_opt.as_ref().as_ref(), u)
                        .expect("tau-tau pair operator transformed forward product should be shape-consistent"),
                },
            }
        };
        let tau_tau_transpose = |term: &TauTauDesignTerm, y: &Array1<f64>| -> Array1<f64> {
            match term {
                TauTauDesignTerm::Dense(dense) => dense.t().dot(y),
                TauTauDesignTerm::Implicit(deriv) => match &self.basis {
                    TauPairBasis::Original => deriv
                        .transpose_mul_original(y)
                        .expect("tau-tau pair operator original transpose product should be shape-consistent"),
                    TauPairBasis::Transformed { qs, free_basis_opt } => deriv
                        .transformed_transpose_mul(
                            qs.as_ref(),
                            free_basis_opt.as_ref().as_ref(),
                            y,
                        )
                        .expect("tau-tau pair operator transformed transpose product should be shape-consistent"),
                },
            }
        };

        let x_tau_i_v = tau_forward(&self.x_tau_i, v);
        let x_tau_j_v = tau_forward(&self.x_tau_j, v);
        let w_x_tau_i_v = &*self.w_diag * &x_tau_i_v;
        let w_x_tau_j_v = &*self.w_diag * &x_tau_j_v;

        if let Some(x_tau_tau) = self.x_tau_tau.as_ref() {
            let w_x_v = &*self.w_diag * &x_v;
            out += &tau_tau_transpose(x_tau_tau, &w_x_v);

            let x_tau_tau_v = tau_tau_forward(x_tau_tau, v);
            out += &self
                .x_design
                .transpose_vector_multiply(&(&*self.w_diag * &x_tau_tau_v));
        }

        out += &tau_transpose(&self.x_tau_i, &w_x_tau_j_v);
        out += &tau_transpose(&self.x_tau_j, &w_x_tau_i_v);

        if let Some(c_x_tau_i_beta) = self.c_x_tau_i_beta.as_ref() {
            out += &tau_transpose(&self.x_tau_j, &(c_x_tau_i_beta * &x_v));
            out += &self
                .x_design
                .transpose_vector_multiply(&(c_x_tau_i_beta * &x_tau_j_v));
        }

        if let Some(c_x_tau_j_beta) = self.c_x_tau_j_beta.as_ref() {
            out += &tau_transpose(&self.x_tau_i, &(c_x_tau_j_beta * &x_v));
            out += &self
                .x_design
                .transpose_vector_multiply(&(c_x_tau_j_beta * &x_tau_i_v));
        }

        if let Some(d_cross) = self.d_cross.as_ref() {
            out += &self.x_design.transpose_vector_multiply(&(d_cross * &x_v));
        }

        if let Some(c_xij_beta) = self.c_xij_beta.as_ref() {
            out += &self
                .x_design
                .transpose_vector_multiply(&(c_xij_beta * &x_v));
        }

        if let Some(s_tau_tau) = self.s_tau_tau.as_ref() {
            out += &s_tau_tau.dot(v);
        }

        out
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.p, self.p));
        let mut basis = Array1::<f64>::zeros(self.p);
        for j in 0..self.p {
            basis[j] = 1.0;
            let col = self.mul_vec(&basis);
            out.column_mut(j).assign(&col);
            basis[j] = 0.0;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        matches!(&self.x_tau_i, TauDesignTerm::Implicit(_))
            || matches!(&self.x_tau_j, TauDesignTerm::Implicit(_))
            || self
                .x_tau_tau
                .as_ref()
                .is_some_and(|term| matches!(term, TauTauDesignTerm::Implicit(_)))
    }
}

impl<'a> RemlState<'a> {
    fn get_pairwisesecond_penalty_components(
        hyper_dirs: &[DirectionalHyperParam],
        i: usize,
        j: usize,
    ) -> Result<Vec<PenaltyDerivativeComponent>, EstimationError> {
        if let Some(components) = hyper_dirs
            .get(i)
            .map(|dir| dir.penaltysecond_components_for(j))
            .transpose()?
            .flatten()
        {
            return Ok(components);
        }
        Ok(hyper_dirs
            .get(j)
            .map(|dir| dir.penaltysecond_components_for(i))
            .transpose()?
            .flatten()
            .unwrap_or_default())
    }

    pub(crate) fn validate_penalty_component_shapes(
        components: &[PenaltyDerivativeComponent],
        p: usize,
        label: &str,
    ) -> Result<(), EstimationError> {
        for component in components {
            if component.matrix.nrows() != p || component.matrix.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "{} shape mismatch for penalty {}: expected {}x{}, got {}x{}",
                    label,
                    component.penalty_index,
                    p,
                    p,
                    component.matrix.nrows(),
                    component.matrix.ncols()
                )));
            }
        }
        Ok(())
    }

    fn transform_penalty_components(
        components: &[PenaltyDerivativeComponent],
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Vec<PenaltyDerivativeComponent> {
        components
            .iter()
            .map(|component| PenaltyDerivativeComponent {
                penalty_index: component.penalty_index,
                matrix: HyperPenaltyDerivative::from(
                    component
                        .matrix
                        .transformed(qs, free_basis_opt)
                        .expect("valid transformed hyper penalty component"),
                ),
            })
            .collect()
    }

    fn tau_design_forward_mul(
        term: &TauDesignTerm,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match term {
            TauDesignTerm::Dense(dense) => {
                if dense.ncols() != u.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense tau design forward_mul width mismatch: matrix={}x{}, vector={}",
                        dense.nrows(),
                        dense.ncols(),
                        u.len()
                    )));
                }
                Ok(dense.dot(u))
            }
            TauDesignTerm::Implicit(deriv) => deriv.transformed_forward_mul(qs, free_basis_opt, u),
        }
    }

    fn tau_design_forward_mul_in_basis(
        term: &TauDesignTerm,
        basis: &TauPairBasis,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match basis {
            TauPairBasis::Original => Self::tau_design_forward_mul_original(term, u),
            TauPairBasis::Transformed { qs, free_basis_opt } => {
                Self::tau_design_forward_mul(term, qs.as_ref(), free_basis_opt.as_ref().as_ref(), u)
            }
        }
    }

    fn tau_design_forward_mul_original(
        term: &TauDesignTerm,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match term {
            TauDesignTerm::Dense(dense) => {
                if dense.ncols() != u.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense tau design original forward_mul width mismatch: matrix={}x{}, vector={}",
                        dense.nrows(),
                        dense.ncols(),
                        u.len()
                    )));
                }
                Ok(dense.dot(u))
            }
            TauDesignTerm::Implicit(deriv) => deriv.forward_mul_original(u),
        }
    }

    fn tau_design_transpose_mul(
        term: &TauDesignTerm,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match term {
            TauDesignTerm::Dense(dense) => {
                if dense.nrows() != v.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense tau design transpose_mul height mismatch: matrix={}x{}, vector={}",
                        dense.nrows(),
                        dense.ncols(),
                        v.len()
                    )));
                }
                Ok(dense.t().dot(v))
            }
            TauDesignTerm::Implicit(deriv) => {
                deriv.transformed_transpose_mul(qs, free_basis_opt, v)
            }
        }
    }

    fn tau_design_transpose_mul_in_basis(
        term: &TauDesignTerm,
        basis: &TauPairBasis,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match basis {
            TauPairBasis::Original => Self::tau_design_transpose_mul_original(term, v),
            TauPairBasis::Transformed { qs, free_basis_opt } => Self::tau_design_transpose_mul(
                term,
                qs.as_ref(),
                free_basis_opt.as_ref().as_ref(),
                v,
            ),
        }
    }

    fn tau_design_transpose_mul_original(
        term: &TauDesignTerm,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match term {
            TauDesignTerm::Dense(dense) => {
                if dense.nrows() != v.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense tau design original transpose_mul height mismatch: matrix={}x{}, vector={}",
                        dense.nrows(),
                        dense.ncols(),
                        v.len()
                    )));
                }
                Ok(dense.t().dot(v))
            }
            TauDesignTerm::Implicit(deriv) => deriv.transpose_mul_original(v),
        }
    }

    fn tau_tau_design_forward_mul_in_basis(
        term: &TauTauDesignTerm,
        basis: &TauPairBasis,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match term {
            TauTauDesignTerm::Dense(dense) => {
                if dense.ncols() != u.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense tau-tau design forward_mul width mismatch: matrix={}x{}, vector={}",
                        dense.nrows(),
                        dense.ncols(),
                        u.len()
                    )));
                }
                Ok(dense.dot(u))
            }
            TauTauDesignTerm::Implicit(deriv) => match basis {
                TauPairBasis::Original => deriv.forward_mul_original(u),
                TauPairBasis::Transformed { qs, free_basis_opt } => {
                    deriv.transformed_forward_mul(qs.as_ref(), free_basis_opt.as_ref().as_ref(), u)
                }
            },
        }
    }

    fn tau_tau_design_transpose_mul_in_basis(
        term: &TauTauDesignTerm,
        basis: &TauPairBasis,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match term {
            TauTauDesignTerm::Dense(dense) => {
                if dense.nrows() != v.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense tau-tau design transpose_mul height mismatch: matrix={}x{}, vector={}",
                        dense.nrows(),
                        dense.ncols(),
                        v.len()
                    )));
                }
                Ok(dense.t().dot(v))
            }
            TauTauDesignTerm::Implicit(deriv) => match basis {
                TauPairBasis::Original => deriv.transpose_mul_original(v),
                TauPairBasis::Transformed { qs, free_basis_opt } => deriv
                    .transformed_transpose_mul(qs.as_ref(), free_basis_opt.as_ref().as_ref(), v),
            },
        }
    }

    fn build_tau_penalty_derivative_data<F>(
        rho: &Array1<f64>,
        p_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
        penalty_components_per_dir: &[Vec<PenaltyDerivativeComponent>],
        pairwise_second_components: F,
    ) -> Result<
        (
            Vec<Array2<f64>>,
            Vec<Vec<Option<Array2<f64>>>>,
            Vec<Vec<Option<Array2<f64>>>>,
            Vec<Vec<Option<Array2<f64>>>>,
        ),
        EstimationError,
    >
    where
        F: Fn(usize, usize) -> Result<Vec<PenaltyDerivativeComponent>, EstimationError>,
    {
        let psi_dim = hyper_dirs.len();
        let k_count = rho.len();

        let s_tau_list: Vec<Array2<f64>> = penalty_components_per_dir
            .iter()
            .map(|components| {
                Self::validate_penalty_component_shapes(components, p_dim, "S_tau")?;
                components.iter().try_fold(
                    Array2::<f64>::zeros((p_dim, p_dim)),
                    |mut acc, component| {
                        if component.penalty_index >= k_count {
                            return Err(EstimationError::InvalidInput(format!(
                                "penalty_index {} out of bounds for rho dimension {}",
                                component.penalty_index, k_count
                            )));
                        }
                        component
                            .matrix
                            .scaled_add_to(&mut acc, rho[component.penalty_index].exp())?;
                        Ok(acc)
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut a_k_tau_j_mats: Vec<Vec<Option<Array2<f64>>>> = vec![vec![None; k_count]; psi_dim];
        let mut ds_k_dtau_j_mats: Vec<Vec<Option<Array2<f64>>>> =
            vec![vec![None; k_count]; psi_dim];
        for (j, components) in penalty_components_per_dir.iter().enumerate() {
            for component in components {
                let k = component.penalty_index;
                if k < k_count {
                    a_k_tau_j_mats[j][k] = Some(component.matrix.scaled_materialize(rho[k].exp()));
                    ds_k_dtau_j_mats[j][k] = Some(component.matrix.scaled_materialize(1.0));
                }
            }
        }

        let mut s_tau_tau: Vec<Vec<Option<Array2<f64>>>> = vec![vec![None; psi_dim]; psi_dim];
        for i in 0..psi_dim {
            for j in i..psi_dim {
                let second_components = pairwise_second_components(i, j)?;
                if second_components.is_empty() {
                    continue;
                }
                Self::validate_penalty_component_shapes(&second_components, p_dim, "S_tau_tau")?;
                let total = second_components.iter().try_fold(
                    Array2::<f64>::zeros((p_dim, p_dim)),
                    |mut acc, component| {
                        if component.penalty_index >= k_count {
                            return Err(EstimationError::InvalidInput(format!(
                                "penalty_index {} out of bounds for rho dimension {}",
                                component.penalty_index, k_count
                            )));
                        }
                        component
                            .matrix
                            .scaled_add_to(&mut acc, rho[component.penalty_index].exp())?;
                        Ok(acc)
                    },
                )?;
                s_tau_tau[i][j] = Some(total.clone());
                if i != j {
                    s_tau_tau[j][i] = Some(total);
                }
            }
        }

        Ok((s_tau_list, s_tau_tau, a_k_tau_j_mats, ds_k_dtau_j_mats))
    }

    fn build_tau_design_data_in_basis(
        hyper_dirs: &[DirectionalHyperParam],
        basis: &TauPairBasis,
        beta_eval: &Array1<f64>,
    ) -> Result<
        (
            Vec<TauDesignTerm>,
            Vec<Array1<f64>>,
            Vec<Vec<Option<TauTauDesignTerm>>>,
        ),
        EstimationError,
    > {
        let psi_dim = hyper_dirs.len();
        let x_tau_terms: Vec<TauDesignTerm> = hyper_dirs
            .iter()
            .map(|dir| {
                if dir.has_implicit_operator() {
                    Ok(TauDesignTerm::Implicit(dir.x_tau_original.clone()))
                } else {
                    match basis {
                        TauPairBasis::Original => Ok(TauDesignTerm::Dense(dir.x_tau_dense())),
                        TauPairBasis::Transformed { qs, free_basis_opt } => {
                            Ok(TauDesignTerm::Dense(dir.transformed_x_tau(
                                qs.as_ref(),
                                free_basis_opt.as_ref().as_ref(),
                            )?))
                        }
                    }
                }
            })
            .collect::<Result<Vec<_>, EstimationError>>()?;
        let x_tau_beta_list = x_tau_terms
            .iter()
            .map(|x_tau| Self::tau_design_forward_mul_in_basis(x_tau, basis, beta_eval))
            .collect::<Result<Vec<_>, _>>()?;

        let mut x_tau_tau: Vec<Vec<Option<TauTauDesignTerm>>> = vec![vec![None; psi_dim]; psi_dim];
        for i in 0..psi_dim {
            for j in i..psi_dim {
                let xij = hyper_dirs[i]
                    .x_tau_tau_entry_at(j)
                    .or_else(|| hyper_dirs[j].x_tau_tau_entry_at(i))
                    .map(|entry| {
                        if entry.uses_implicit_storage() {
                            Ok::<TauTauDesignTerm, EstimationError>(TauTauDesignTerm::Implicit(
                                entry,
                            ))
                        } else {
                            match basis {
                                TauPairBasis::Original => Ok::<TauTauDesignTerm, EstimationError>(
                                    TauTauDesignTerm::Dense(entry.materialize()),
                                ),
                                TauPairBasis::Transformed { qs, free_basis_opt } => {
                                    Ok::<TauTauDesignTerm, EstimationError>(
                                        TauTauDesignTerm::Dense(entry.transformed(
                                            qs.as_ref(),
                                            free_basis_opt.as_ref().as_ref(),
                                        )?),
                                    )
                                }
                            }
                        }
                    })
                    .transpose()?;
                if xij.is_some() {
                    x_tau_tau[j][i] = xij.clone();
                }
                x_tau_tau[i][j] = xij;
            }
        }

        Ok((x_tau_terms, x_tau_beta_list, x_tau_tau))
    }

    fn canonical_penalty_matrices(
        penalties: &[crate::construction::CanonicalPenalty],
    ) -> Vec<Array2<f64>> {
        penalties
            .iter()
            .map(|cp| {
                let r = cp.full_width_root();
                r.t().dot(&r)
            })
            .collect()
    }

    fn build_tau_tau_pair_callback(
        basis: TauPairBasis,
        pld: std::sync::Arc<super::penalty_logdet::PenaltyPseudologdet>,
        s_tau_list: std::sync::Arc<Vec<Array2<f64>>>,
        s_tau_tau: std::sync::Arc<Vec<Vec<Option<Array2<f64>>>>>,
        beta_eval: std::sync::Arc<Array1<f64>>,
        x_design: std::sync::Arc<DesignMatrix>,
        x_tau_terms: std::sync::Arc<Vec<TauDesignTerm>>,
        x_tau_beta_list: std::sync::Arc<Vec<Array1<f64>>>,
        x_tau_tau: std::sync::Arc<Vec<Vec<Option<TauTauDesignTerm>>>>,
        u: std::sync::Arc<Array1<f64>>,
        w_diag: std::sync::Arc<Array1<f64>>,
        c_array: std::sync::Arc<Array1<f64>>,
        d_array: std::sync::Arc<Array1<f64>>,
        p_dim: usize,
        is_gaussian_identity: bool,
    ) -> Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync> {
        Box::new(
            move |i: usize, j: usize| -> super::unified::HyperCoordPair {
                let ld_s_ij = pld.tau_hessian_component(
                    &s_tau_list[i],
                    &s_tau_list[j],
                    s_tau_tau[i][j].as_ref().map(|m| m as &Array2<f64>),
                );

                let x_tau_i_beta = &x_tau_beta_list[i];
                let x_tau_j_beta = &x_tau_beta_list[j];
                let w_x_tau_j_beta = w_diag.as_ref() * x_tau_j_beta;
                let a_ij_likelihood = x_tau_i_beta.dot(&w_x_tau_j_beta);
                let x_tau_tau_beta = match x_tau_tau[i][j].as_ref() {
                    Some(term) => {
                        Self::tau_tau_design_forward_mul_in_basis(term, &basis, beta_eval.as_ref())
                            .expect("valid X_tau_tau beta product")
                    }
                    None => Array1::<f64>::zeros(u.len()),
                };
                let a_ij_design2 = if x_tau_tau[i][j].is_some() {
                    -u.dot(&x_tau_tau_beta)
                } else {
                    0.0
                };
                let a_ij_penalty = 0.5
                    * s_tau_tau[i][j]
                        .as_ref()
                        .map(|s_ij| beta_eval.dot(&s_ij.dot(beta_eval.as_ref())))
                        .unwrap_or(0.0);
                let a_ij = a_ij_likelihood + a_ij_design2 + a_ij_penalty;

                let x_tau_i = &x_tau_terms[i];
                let x_tau_j = &x_tau_terms[j];
                let term1 = match x_tau_tau[i][j].as_ref() {
                    Some(term) => {
                        Self::tau_tau_design_transpose_mul_in_basis(term, &basis, u.as_ref())
                            .expect("valid X_tau_tau^T u product")
                    }
                    None => Array1::<f64>::zeros(p_dim),
                };
                let term2 = Self::tau_design_transpose_mul_in_basis(
                    x_tau_j,
                    &basis,
                    &(w_diag.as_ref() * x_tau_i_beta),
                )
                .expect("valid X_tau_j^T W X_tau_i beta product");
                let term3 =
                    Self::tau_design_transpose_mul_in_basis(x_tau_i, &basis, &w_x_tau_j_beta)
                        .expect("valid X_tau_i^T W X_tau_j beta product");
                let c_x_tau_i_beta = c_array.as_ref() * x_tau_i_beta;
                let term4 = x_design.transpose_vector_multiply(&(&c_x_tau_i_beta * x_tau_j_beta));
                let term5 = if x_tau_tau[i][j].is_some() {
                    x_design.transpose_vector_multiply(&(w_diag.as_ref() * &x_tau_tau_beta))
                } else {
                    Array1::<f64>::zeros(p_dim)
                };
                let term6 = s_tau_tau[i][j]
                    .as_ref()
                    .map(|s_ij| s_ij.dot(beta_eval.as_ref()))
                    .unwrap_or_else(|| Array1::<f64>::zeros(p_dim));
                let g_ij = term1 - &term2 - &term3 - &term4 - &term5 - &term6;

                let c_x_tau_j_beta =
                    (!is_gaussian_identity).then(|| c_array.as_ref() * x_tau_j_beta);
                let d_cross = (!is_gaussian_identity)
                    .then(|| d_array.as_ref() * &(x_tau_i_beta * x_tau_j_beta));
                let c_xij_beta = if !is_gaussian_identity && x_tau_tau[i][j].is_some() {
                    Some(c_array.as_ref() * &x_tau_tau_beta)
                } else {
                    None
                };
                let b_operator: Box<dyn super::unified::HyperOperator> =
                    Box::new(TauTauPairHyperOperator {
                        x_tau_i: x_tau_i.clone(),
                        x_tau_j: x_tau_j.clone(),
                        x_tau_tau: x_tau_tau[i][j].clone(),
                        x_design: std::sync::Arc::clone(&x_design),
                        basis: basis.clone(),
                        w_diag: std::sync::Arc::clone(&w_diag),
                        c_x_tau_i_beta: (!is_gaussian_identity).then_some(c_x_tau_i_beta.clone()),
                        c_x_tau_j_beta,
                        d_cross,
                        c_xij_beta,
                        s_tau_tau: s_tau_tau[i][j].clone(),
                        p: p_dim,
                    });

                super::unified::HyperCoordPair {
                    a: a_ij,
                    g: g_ij,
                    b_mat: Array2::<f64>::zeros((0, 0)),
                    b_operator: Some(b_operator),
                    ld_s: ld_s_ij,
                }
            },
        )
    }

    fn build_rho_tau_pair_callback(
        pld: std::sync::Arc<super::penalty_logdet::PenaltyPseudologdet>,
        s_k_unscaled: std::sync::Arc<Vec<Array2<f64>>>,
        s_tau_list: std::sync::Arc<Vec<Array2<f64>>>,
        ds_k_dtau_j_mats: std::sync::Arc<Vec<Vec<Option<Array2<f64>>>>>,
        lambdas: std::sync::Arc<Array1<f64>>,
        a_k_tau_j_mats: std::sync::Arc<Vec<Vec<Option<Array2<f64>>>>>,
        beta_eval: std::sync::Arc<Array1<f64>>,
        p_dim: usize,
    ) -> Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync> {
        Box::new(
            move |k: usize, j: usize| -> super::unified::HyperCoordPair {
                let s_tau_j = s_tau_list.get(j);
                let a_k_tau_j = a_k_tau_j_mats
                    .get(j)
                    .and_then(|row| row.get(k))
                    .and_then(|entry| entry.as_ref());
                let ds_k_dtau_j = ds_k_dtau_j_mats
                    .get(j)
                    .and_then(|row| row.get(k))
                    .and_then(|entry| entry.as_ref());
                let ld_s_kj = match (s_k_unscaled.get(k), lambdas.get(k), s_tau_j) {
                    (Some(s_k), Some(&lambda_k), Some(s_tau_j)) => {
                        pld.rho_tau_hessian_component(s_k, lambda_k, s_tau_j, ds_k_dtau_j)
                    }
                    _ => 0.0,
                };
                let a_kj = 0.5
                    * a_k_tau_j
                        .map(|a_kt| beta_eval.dot(&a_kt.dot(beta_eval.as_ref())))
                        .unwrap_or(0.0);
                let g_kj = a_k_tau_j
                    .map(|a_kt| -(a_kt.dot(beta_eval.as_ref())))
                    .unwrap_or_else(|| Array1::<f64>::zeros(p_dim));
                let b_kj = a_k_tau_j
                    .cloned()
                    .unwrap_or_else(|| Array2::<f64>::zeros((p_dim, p_dim)));

                super::unified::HyperCoordPair {
                    a: a_kj,
                    g: g_kj,
                    b_mat: b_kj,
                    b_operator: None,
                    ld_s: ld_s_kj,
                }
            },
        )
    }

    pub(crate) fn compute_joint_hyper_eval_with_order(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
        order: crate::solver::outer_strategy::OuterEvalOrder,
    ) -> Result<
        (
            f64,
            Array1<f64>,
            crate::solver::outer_strategy::HessianResult,
        ),
        EstimationError,
    > {
        let t_outer_start = std::time::Instant::now();
        let rho = theta.slice(s![..rho_dim]).to_owned();

        if !hyper_dirs.is_empty() {
            let requested_hessian = matches!(
                order,
                crate::solver::outer_strategy::OuterEvalOrder::ValueGradientHessian
            );
            let firth_pair_terms_unavailable = self.config.firth_bias_reduction
                && matches!(self.config.link_function(), LinkFunction::Logit);
            let tau_tau_policy = super::exact_tau_tau_hessian_policy_with_firth(
                self.x().nrows(),
                self.x().ncols(),
                hyper_dirs,
                firth_pair_terms_unavailable,
            );
            let downgrade_exact_tau_tau =
                requested_hessian && tau_tau_policy.prefer_gradient_only();
            if downgrade_exact_tau_tau {
                log::warn!(
                    "[OUTER] disabling exact tau Hessian; using gradient-only outer eval \
                     (n={}, p={}, psi_dim={}, implicit_tau={}, implicit_multidim_duchon={}, firth_pair_gap={}, dense_tau_cache={:.1} MiB, gradient_plan={:.1} MiB, exact_hessian_plan={:.1} MiB, budget={:.1} MiB)",
                    self.x().nrows(),
                    self.x().ncols(),
                    hyper_dirs.len(),
                    tau_tau_policy.any_has_implicit,
                    tau_tau_policy.implicit_multidim_duchon,
                    tau_tau_policy.firth_pair_terms_unavailable,
                    tau_tau_policy.estimated_dense_tau_cache_bytes as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.gradient_plan.total_bytes() as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.hessian_plan.total_bytes() as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.budget_bytes as f64 / (1024.0 * 1024.0),
                );
            }
            let eval_mode = match if downgrade_exact_tau_tau {
                crate::solver::outer_strategy::OuterEvalOrder::ValueAndGradient
            } else {
                order
            } {
                crate::solver::outer_strategy::OuterEvalOrder::ValueAndGradient => {
                    super::unified::EvalMode::ValueAndGradient
                }
                crate::solver::outer_strategy::OuterEvalOrder::ValueGradientHessian => {
                    super::unified::EvalMode::ValueGradientHessian
                }
            };
            let result = self.evaluate_unified_with_psi_ext(&rho, eval_mode, hyper_dirs)?;
            let cost = result.cost;
            let grad = result
                .gradient
                .unwrap_or_else(|| Array1::zeros(theta.len()));
            log::info!(
                "[outer-timing] compute_joint_hyper_eval (unified, rho_dim={}, psi_dim={}): {:.3}s  cost={:.6e}",
                rho_dim,
                hyper_dirs.len(),
                t_outer_start.elapsed().as_secs_f64(),
                cost,
            );
            Ok((
                cost,
                grad,
                if requested_hessian && !downgrade_exact_tau_tau {
                    result.hessian
                } else {
                    crate::solver::outer_strategy::HessianResult::Unavailable
                },
            ))
        } else {
            let cost = self.compute_cost(&rho)?;
            let grad = self.compute_gradient(&rho)?;
            log::info!(
                "[outer-timing] compute_joint_hyper_eval (rho-only, dim={}): {:.3}s  cost={:.6e}",
                rho_dim,
                t_outer_start.elapsed().as_secs_f64(),
                cost,
            );
            Ok((
                cost,
                grad,
                if matches!(
                    order,
                    crate::solver::outer_strategy::OuterEvalOrder::ValueGradientHessian
                ) {
                    crate::solver::outer_strategy::HessianResult::Analytic(
                        self.compute_lamlhessian_consistent(&rho)?,
                    )
                } else {
                    crate::solver::outer_strategy::HessianResult::Unavailable
                },
            ))
        }
    }

    pub(crate) fn build_tau_unified_objects_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Vec<super::unified::HyperCoord>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        let t_tau = std::time::Instant::now();
        // Guard: non-sparse tau coordinate construction requires dense design.
        // Skip for large models that would blow memory.
        let n_x = self.x().nrows();
        let p_x = self.x().ncols();
        const HYPER_MAX_DENSE_WORK: usize = 50_000_000;
        if n_x.saturating_mul(p_x) > HYPER_MAX_DENSE_WORK
            && bundle.backend_kind() != GeometryBackendKind::SparseExactSpd
        {
            log::warn!(
                "skipping tau hyper-coordinate construction (n={n_x}, p={p_x}): \
                 dense design materialization too large; falling back to rho-only REML"
            );
            let identity_pair: Box<
                dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync,
            > = Box::new(|_, _| super::unified::HyperCoordPair::zero());
            let identity_pair2: Box<
                dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync,
            > = Box::new(|_, _| super::unified::HyperCoordPair::zero());
            return Ok((Vec::new(), identity_pair, identity_pair2));
        }
        let backend_label;
        let result = if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            backend_label = "sparse_exact";
            let ext_coords = self.build_tau_hyper_coords_sparse_exact(rho, bundle, hyper_dirs)?;
            let (ext_pair_fn, rho_ext_pair_fn) =
                self.build_tau_pair_callbacks_sparse_exact(rho, bundle, hyper_dirs)?;
            Ok((ext_coords, ext_pair_fn, rho_ext_pair_fn))
        } else if matches!(
            bundle.pirls_result.coordinate_frame,
            crate::pirls::PirlsCoordinateFrame::TransformedQs
        ) && self
            .active_constraint_free_basis(bundle.pirls_result.as_ref())
            .is_none()
        {
            backend_label = "original_basis";
            let ext_coords = self.build_tau_hyper_coords_original_basis(rho, bundle, hyper_dirs)?;
            let (ext_pair_fn, rho_ext_pair_fn) =
                self.build_tau_pair_callbacks_original_basis(rho, bundle, hyper_dirs)?;
            Ok((ext_coords, ext_pair_fn, rho_ext_pair_fn))
        } else {
            backend_label = "generic";
            let ext_coords = self.build_tau_hyper_coords(rho, bundle, hyper_dirs)?;
            let (ext_pair_fn, rho_ext_pair_fn) =
                self.build_tau_pair_callbacks(rho, bundle, hyper_dirs)?;
            Ok((ext_coords, ext_pair_fn, rho_ext_pair_fn))
        };
        log::info!(
            "[outer-timing] build_tau_unified_objects_from_bundle ({}, n={}, p={}, psi_dim={}): {:.1}ms",
            backend_label,
            n_x,
            p_x,
            hyper_dirs.len(),
            t_tau.elapsed().as_secs_f64() * 1000.0,
        );
        result
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified HyperCoord builders for τ (directional) hyperparameters
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build [`HyperCoord`] objects for τ (directional / design-moving)
    /// hyperparameters.
    ///
    /// This produces generic `(a_j, g_j, B_j, ld_s_j)` tuples that the
    /// unified evaluator in `unified.rs` can consume for first-order
    /// directional hyperparameter derivatives.
    ///
    /// # Field derivation
    ///
    /// For each τ direction j, holding β fixed at β̂:
    ///
    /// | Field   | Formula |
    /// | `a`     | `−u^T X_{τ_j} β̂ + 0.5 β̂^T S_{τ_j} β̂ [+ Φ_{τ_j}\|_β]` |
    /// | `g`     | `X_{τ_j}^T u − X^T diag(w) X_{τ_j} β̂ − S_{τ_j} β̂ [− (g_φ)_{τ_j}]` |
    /// | `B`     | `X_{τ_j}^T W X + X^T W X_{τ_j} + X^T diag(c ⊙ X_{τ_j} β̂) X + S_{τ_j} [− Firth drifts]` |
    /// | `ld_s`  | `tr(S⁺ S_{τ_j})` |
    ///
    /// For Gaussian identity link, `c = 0` and W is constant, so the
    /// third-derivative correction in B vanishes and `b_depends_on_beta = false`.
    pub(crate) fn build_tau_hyper_coords(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        let psi_dim = hyper_dirs.len();

        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        // --- Transformed design, beta, penalty basis ---
        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();

        let e_eval;
        if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            e_eval = reparam_result.e_transformed.dot(z);
        } else {
            e_eval = reparam_result.e_transformed.clone();
        }
        let p_dim = beta_eval.len();
        if p_dim == 0 {
            return Ok((0..psi_dim)
                .map(|j| super::unified::HyperCoord {
                    a: 0.0,
                    g: Array1::zeros(0),
                    drift: super::unified::HyperCoordDrift::none(),
                    ld_s: 0.0,
                    b_depends_on_beta: false,
                    is_penalty_like: hyper_dirs[j].is_penalty_like,
                })
                .collect::<Vec<_>>());
        }

        // Working residual u = w ⊙ (z − η̂).
        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let w_diag = &pirls_result.finalweights;
        let c_array = &pirls_result.solve_c_array;

        // Whether third-derivative corrections are needed (non-Gaussian).
        let is_gaussian_identity = matches!(self.config.link_function(), LinkFunction::Identity);
        let b_depends_on_beta = !is_gaussian_identity;

        // Firth operator (Firth-logit only).
        // Match the active coefficient basis exactly: when constraints project
        // to β = Z β_free, the Jeffreys objects for τ/ψ derivatives must use
        // the projected design X_eff = X Z rather than the cached full-basis
        // operator from the unconstrained transformed space.
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);

        // --- Implicit operator activation ---
        // Check whether any tau direction has an implicit design derivative and
        // the problem is large enough to benefit from implicit B_j operators.
        // When active, we keep the active-basis X design operator-backed.
        let any_has_implicit = hyper_dirs.iter().any(|d| d.has_implicit_operator());
        let n_obs = pirls_result.x_transformed.nrows();
        let implicit_n_axes = if any_has_implicit {
            hyper_dirs
                .iter()
                .find_map(|d| d.implicit_first_axis_info())
                .map(|(op, _)| op.n_axes())
                .unwrap_or(0)
        } else {
            0
        };
        let use_implicit_requested = any_has_implicit
            && crate::terms::basis::should_use_implicit_operators(n_obs, p_dim, implicit_n_axes);
        let x_design_shared: Option<std::sync::Arc<DesignMatrix>> = if use_implicit_requested {
            Some(std::sync::Arc::new(
                build_active_design_matrix(&pirls_result.x_transformed, free_basis_opt.as_ref())
                    .map_err(EstimationError::InvalidInput)?,
            ))
        } else {
            None
        };
        let use_implicit = x_design_shared.is_some();
        let need_x_dense = firth_logit_active || !use_implicit;
        let x_dense_arc = if need_x_dense {
            Some(
                pirls_result
                    .x_transformed
                    .try_to_dense_arc("build_tau_hyper_coords requires dense transformed design")
                    .map_err(EstimationError::InvalidInput)?,
            )
        } else {
            None
        };
        let x_dense_owned = if need_x_dense {
            free_basis_opt.as_ref().map(|z| {
                DenseRightProductView::new(
                    x_dense_arc.as_ref().expect("dense X should exist").as_ref(),
                )
                .with_factor(z)
                .materialize()
            })
        } else {
            None
        };
        let x_dense =
            if need_x_dense {
                Some(x_dense_owned.as_ref().unwrap_or_else(|| {
                    x_dense_arc.as_ref().expect("dense X should exist").as_ref()
                }))
            } else {
                None
            };

        let firth_op = if firth_logit_active {
            let x_dense = x_dense.expect("Firth hyper terms require dense active-basis design");
            if free_basis_opt.is_none() {
                if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                    Some(cached.as_ref().clone())
                } else {
                    Some(Self::build_firth_dense_operator(
                        x_dense,
                        &pirls_result.final_eta,
                        self.weights,
                    )?)
                }
            } else {
                Some(Self::build_firth_dense_operator(
                    x_dense,
                    &pirls_result.final_eta,
                    self.weights,
                )?)
            }
        } else {
            None
        };
        let w_diag_shared: Option<std::sync::Arc<Array1<f64>>> = if use_implicit {
            Some(std::sync::Arc::new(w_diag.clone()))
        } else {
            None
        };

        let mut coords = Vec::with_capacity(psi_dim);

        for j in 0..psi_dim {
            let implicit_first = if use_implicit {
                hyper_dirs[j].implicit_first_axis_info()
            } else {
                None
            };
            let implicit_tau_available = hyper_dirs[j].has_implicit_operator();
            let mut x_tau_j_dense: Option<Array2<f64>> = None;
            let penalty_components_j = Self::transform_penalty_components(
                hyper_dirs[j].penalty_first_components(),
                &reparam_result.qs,
                free_basis_opt.as_ref(),
            );
            let s_tau_j = penalty_components_j.iter().try_fold(
                Array2::<f64>::zeros((p_dim, p_dim)),
                |mut acc, component| {
                    if component.penalty_index >= rho.len() {
                        return Err(EstimationError::InvalidInput(format!(
                            "penalty_index {} out of bounds for rho dimension {}",
                            component.penalty_index,
                            rho.len()
                        )));
                    }
                    component
                        .matrix
                        .scaled_add_to(&mut acc, rho[component.penalty_index].exp())?;
                    Ok(acc)
                },
            )?;

            // --- a_j: fixed-β cost derivative (envelope term) ---
            // a_j = −u^T (X_{τ_j} β̂) + 0.5 β̂^T S_{τ_j} β̂  [+ Φ_{τ_j}|_β for Firth]
            let x_tau_beta_j = if implicit_tau_available {
                hyper_dirs[j].x_tau_original.transformed_forward_mul(
                    &reparam_result.qs,
                    free_basis_opt.as_ref(),
                    &beta_eval,
                )?
            } else {
                x_tau_j_dense
                    .get_or_insert(
                        hyper_dirs[j]
                            .transformed_x_tau(&reparam_result.qs, free_basis_opt.as_ref())?,
                    )
                    .dot(&beta_eval)
            };
            let mut a_j = -u.dot(&x_tau_beta_j) + 0.5 * beta_eval.dot(&s_tau_j.dot(&beta_eval));
            // Firth partial: Φ_{τ_j}|_β = 0.5 tr(I_r^{-1} I_{r,τ_j}).
            let mut firth_tau_kernel_j: Option<FirthTauPartialKernel> = None;
            if let Some(op) = firth_op.as_ref() {
                let x_tau_j = x_tau_j_dense.get_or_insert(
                    hyper_dirs[j].transformed_x_tau(&reparam_result.qs, free_basis_opt.as_ref())?,
                );
                let need_kernel = x_tau_j.iter().any(|v| *v != 0.0);
                let tau_bundle = Self::firth_exact_tau_kernel(op, x_tau_j, &beta_eval, need_kernel);
                a_j += tau_bundle.phi_tau_partial;
                if need_kernel {
                    firth_tau_kernel_j = Some(tau_bundle.tau_kernel.expect(
                        "firth_exact_tau_kernel should return kernel when need_kernel=true",
                    ));
                }
            }

            // --- g_j: fixed-β score (the implicit-function RHS) ---
            // g_j = X_{τ_j}^T u − X^T diag(w)(X_{τ_j} β̂) − S_{τ_j} β̂  [− (g_φ)_{τ_j}]
            let weighted_x_tau_beta_j = w_diag * &x_tau_beta_j;
            let xt_weighted_x_tau_beta_j = if let Some(x_design) = x_design_shared.as_ref() {
                x_design.transpose_vector_multiply(&weighted_x_tau_beta_j)
            } else {
                x_dense
                    .expect("dense X should exist when the hyper drift is not implicit")
                    .t()
                    .dot(&weighted_x_tau_beta_j)
            };
            let x_tau_t_u = if implicit_tau_available {
                hyper_dirs[j].x_tau_original.transformed_transpose_mul(
                    &reparam_result.qs,
                    free_basis_opt.as_ref(),
                    &u,
                )?
            } else {
                x_tau_j_dense
                    .get_or_insert(
                        hyper_dirs[j]
                            .transformed_x_tau(&reparam_result.qs, free_basis_opt.as_ref())?,
                    )
                    .t()
                    .dot(&u)
            };
            let mut g_j = x_tau_t_u - xt_weighted_x_tau_beta_j - s_tau_j.dot(&beta_eval);
            if let Some(op) = firth_op.as_ref() {
                let x_tau_j = x_tau_j_dense.get_or_insert(
                    hyper_dirs[j].transformed_x_tau(&reparam_result.qs, free_basis_opt.as_ref())?,
                );
                let tau_bundle = Self::firth_exact_tau_kernel(op, x_tau_j, &beta_eval, false);
                g_j -= &tau_bundle.gphi_tau;
            }

            // --- B_j: fixed-β Hessian drift ---
            // B_j = X_{τ_j}^T W X + X^T W X_{τ_j} + S_{τ_j}
            //     [+ X^T diag(c ⊙ X_{τ_j} β̂) X]  (non-Gaussian only)
            //     [− Firth Hessian drifts]

            // --- b_operator: implicit operator for B_j · v (when activated) ---
            // When the problem is large enough and this coordinate has an implicit
            // design derivative, we build an ImplicitHyperOperator that computes
            // B_j · v without materializing the full (p × p) matrix.
            //
            // Note: the ImplicitHyperOperator covers the three dominant terms:
            //   (∂X/∂ψ_d)^T (W · (X · v)) + X^T (W · ((∂X/∂ψ_d) · v)) + S_{ψ_d} · v
            // Third-derivative corrections (non-Gaussian) and Firth drifts are NOT
            // included in the implicit operator — they are small relative to the
            // dominant terms and would require additional storage. For problems large
            // enough to trigger implicit operators, these corrections contribute
            // negligibly to the stochastic trace estimate.
            let b_operator: Option<std::sync::Arc<dyn super::unified::HyperOperator>> =
                if use_implicit {
                    if let Some((implicit_deriv, axis)) = implicit_first {
                        Some(std::sync::Arc::new(super::unified::ImplicitHyperOperator {
                            implicit_deriv,
                            axis,
                            x_design: x_design_shared.clone().unwrap(),
                            w_diag: w_diag_shared.clone().unwrap(),
                            s_psi: s_tau_j.clone(),
                            p: p_dim,
                        }))
                    } else {
                        None
                    }
                } else {
                    None
                };

            // Materialize the dense B_j matrix only when we are not using the
            // operator-backed fast path. The HyperCoord drift wrapper can carry
            // either representation cleanly, without a dummy payload contract.
            let dense_b = if b_operator.is_some() {
                None
            } else {
                let x_tau_j = x_tau_j_dense.get_or_insert(
                    hyper_dirs[j].transformed_x_tau(&reparam_result.qs, free_basis_opt.as_ref())?,
                );
                let x_dense = x_dense.expect("dense X should exist when materializing hyper drift");
                let mut b_j = Self::weighted_cross(x_tau_j, x_dense, w_diag);
                b_j += &Self::weighted_cross(x_dense, x_tau_j, w_diag);
                b_j += &s_tau_j;

                if !is_gaussian_identity {
                    // Third-derivative correction: X^T diag(c ⊙ X_{τ_j} β̂) X.
                    let c_x_tau_beta = c_array * &x_tau_beta_j;
                    let mut weighted_scratch = Array2::<f64>::zeros((0, 0));
                    b_j +=
                        &Self::xt_diag_x_dense_into(x_dense, &c_x_tau_beta, &mut weighted_scratch);
                }

                // Firth Hessian drifts: −(H_φ)_{τ_j}|_β.
                // The D(H_φ)[β_{τ_j}] part is NOT included here because it
                // depends on β_{τ_j} (the IFT solve result), which the unified
                // evaluator computes itself. Only the fixed-β partial goes in B_j.
                if let Some(op) = firth_op.as_ref() {
                    if let Some(kernel) = firth_tau_kernel_j.as_ref() {
                        let eye = Array2::<f64>::eye(p_dim);
                        let hphi_tau_partial =
                            Self::firth_hphi_tau_partial_apply(op, x_tau_j, kernel, &eye);
                        b_j -= &hphi_tau_partial;
                    }
                }

                Some(b_j)
            };

            // --- ld_s_j: penalty pseudo-logdet derivative ---
            // ld_s_j = tr(S⁺ S_{τ_j}).
            // Uses the exact pseudoinverse (via fixed_subspace_penalty_trace).
            let ld_s_j =
                self.fixed_subspace_penalty_trace(&e_eval, &s_tau_j, pirls_result.ridge_passport)?;

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                drift: super::unified::HyperCoordDrift::from_parts(dense_b, b_operator),
                ld_s: ld_s_j,
                b_depends_on_beta,
                is_penalty_like: hyper_dirs[j].is_penalty_like,
            });
        }

        Ok(coords)
    }

    /// Sparse-exact τ builder in the original sparse/native coefficient basis.
    ///
    /// Unlike the dense/transformed path, this builder keeps `β`, `X`, `X_τ`,
    /// and `S_τ` in the original sparse-native coordinates used by the sparse
    /// Cholesky factor. The fixed-β Hessian drift is attached as an operator,
    /// not a dense matrix, so the unified evaluator can compute
    /// `tr(H^{-1} B_τ)` exactly without falling back to dense spectral
    /// materialization.
    pub(crate) fn build_tau_hyper_coords_original_basis(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        let psi_dim = hyper_dirs.len();

        let pirls_result = bundle.pirls_result.as_ref();
        let beta_eval = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta_eval.len();
        let n_obs = self.y.len();
        if p_dim == 0 {
            return Ok((0..psi_dim)
                .map(|j| super::unified::HyperCoord {
                    a: 0.0,
                    g: Array1::zeros(0),
                    drift: super::unified::HyperCoordDrift::none(),
                    ld_s: 0.0,
                    b_depends_on_beta: false,
                    is_penalty_like: hyper_dirs[j].is_penalty_like,
                })
                .collect());
        }

        for (j, dir) in hyper_dirs.iter().enumerate() {
            if dir.x_tau_original.nrows() != n_obs || dir.x_tau_original.ncols() != p_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "X_tau shape mismatch for sparse exact tau coord {}: expected {}x{}, got {}x{}",
                    j,
                    n_obs,
                    p_dim,
                    dir.x_tau_original.nrows(),
                    dir.x_tau_original.ncols()
                )));
            }
            Self::validate_penalty_component_shapes(
                dir.penalty_first_components(),
                p_dim,
                "S_tau",
            )?;
        }

        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let w_diag = std::sync::Arc::new(pirls_result.finalweights.clone());
        let x_design = self.x().clone();

        let is_gaussian_identity = matches!(self.config.link_function(), LinkFunction::Identity);
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle
                .firth_dense_operator_original
                .as_ref()
                .or(bundle.firth_dense_operator.as_ref())
            {
                Some(cached.as_ref().clone())
            } else {
                let x_dense_arc = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact tau coords require dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(Self::build_firth_dense_operator(
                    x_dense_arc.as_ref(),
                    &pirls_result.final_eta,
                    self.weights,
                )?)
            }
        } else {
            None
        };
        let pld = super::penalty_logdet::PenaltyPseudologdet::from_penalties(
            &self.canonical_penalties,
            &rho.mapv(f64::exp).to_vec(),
            bundle.ridge_passport.penalty_logdet_ridge(),
            p_dim,
        )
        .map_err(EstimationError::InvalidInput)?;

        let mut coords = Vec::with_capacity(psi_dim);
        for dir in hyper_dirs {
            let s_tau_j = dir.penalty_total_at(rho, p_dim)?;
            let x_tau_beta_j = dir.x_tau_original.forward_mul_original(&beta_eval)?;
            let weighted_x_tau_beta_j = &*w_diag * &x_tau_beta_j;

            let mut a_j = -u.dot(&x_tau_beta_j) + 0.5 * beta_eval.dot(&s_tau_j.dot(&beta_eval));
            let mut g_j = dir.x_tau_original.transpose_mul_original(&u)?
                - self.x().transpose_vector_multiply(&weighted_x_tau_beta_j)
                - s_tau_j.dot(&beta_eval);

            let mut firth_hphi_tau_partial = None;
            if let Some(op) = firth_op.as_ref() {
                let x_tau_dense = dir.x_tau_dense();
                let need_kernel = dir.x_tau_original.any_nonzero();
                let tau_bundle =
                    Self::firth_exact_tau_kernel(op, &x_tau_dense, &beta_eval, need_kernel);
                g_j -= &tau_bundle.gphi_tau;
                a_j += tau_bundle.phi_tau_partial;
                if let Some(kernel) = tau_bundle.tau_kernel.as_ref() {
                    let eye = Array2::<f64>::eye(p_dim);
                    firth_hphi_tau_partial = Some(Self::firth_hphi_tau_partial_apply(
                        op,
                        &x_tau_dense,
                        kernel,
                        &eye,
                    ));
                }
            }

            let c_x_tau_beta = if is_gaussian_identity {
                None
            } else {
                Some(&pirls_result.solve_c_array * &x_tau_beta_j)
            };

            let ld_s_j = pld.tau_gradient_component(&s_tau_j);

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                drift: super::unified::HyperCoordDrift::from_operator(std::sync::Arc::new(
                    super::unified::SparseDirectionalHyperOperator {
                        x_tau: dir.x_tau_original.clone(),
                        x_design: x_design.clone(),
                        w_diag: w_diag.clone(),
                        s_tau: s_tau_j,
                        c_x_tau_beta,
                        firth_hphi_tau_partial,
                        p: p_dim,
                    },
                )),
                ld_s: ld_s_j,
                b_depends_on_beta: !is_gaussian_identity,
                is_penalty_like: dir.is_penalty_like,
            });
        }

        Ok(coords)
    }

    pub(crate) fn build_tau_hyper_coords_sparse_exact(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        if bundle.sparse_exact.is_none() {
            return Err(EstimationError::InvalidInput(
                "missing sparse exact evaluation payload".to_string(),
            ));
        }
        self.build_tau_hyper_coords_original_basis(rho, bundle, hyper_dirs)
    }

    /// Sparse-exact τ×τ and ρ×τ pair callbacks in original coordinates.
    ///
    /// This path keeps first- and second-order design derivatives operator-backed
    /// when possible and returns operator-backed τ×τ Hessian drifts to avoid
    /// caching dense `X_τj` blocks across anisotropy directions.
    pub(crate) fn build_tau_pair_callbacks_original_basis(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        let pirls_result = bundle.pirls_result.as_ref();
        let beta_eval = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta_eval.len();
        let lambdas = rho.mapv(f64::exp);

        if p_dim == 0 {
            let tau_tau_pair_fn = move |_: usize, _: usize| super::unified::HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(p_dim),
                b_mat: Array2::zeros((p_dim, p_dim)),
                b_operator: None,
                ld_s: 0.0,
            };
            let rho_tau_pair_fn = move |_: usize, _: usize| super::unified::HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(p_dim),
                b_mat: Array2::zeros((p_dim, p_dim)),
                b_operator: None,
                ld_s: 0.0,
            };
            return Ok((Box::new(tau_tau_pair_fn), Box::new(rho_tau_pair_fn)));
        }

        let penalty_components_per_dir: Vec<Vec<PenaltyDerivativeComponent>> = hyper_dirs
            .iter()
            .map(|dir| dir.penalty_first_components().to_vec())
            .collect();
        let (s_tau_list, s_tau_tau, a_k_tau_j_mats, ds_k_dtau_j_mats) =
            Self::build_tau_penalty_derivative_data(
                rho,
                p_dim,
                hyper_dirs,
                &penalty_components_per_dir,
                |i, j| Self::get_pairwisesecond_penalty_components(hyper_dirs, i, j),
            )?;

        // Use block-factored penalty logdet when penalties are disjoint.
        let pld = super::penalty_logdet::PenaltyPseudologdet::from_penalties(
            &self.canonical_penalties,
            &lambdas.as_slice().unwrap_or(&[]),
            bundle.ridge_passport.penalty_logdet_ridge(),
            p_dim,
        )
        .map_err(EstimationError::InvalidInput)?;

        let x_design = std::sync::Arc::new(self.x().clone());

        let basis = TauPairBasis::Original;
        let (x_tau_terms, x_tau_beta_list, x_tau_tau) =
            Self::build_tau_design_data_in_basis(hyper_dirs, &basis, &beta_eval)?;

        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let w_diag = pirls_result.finalweights.clone();
        let c_array = pirls_result.solve_c_array.clone();
        let d_array = pirls_result.solve_d_array.clone();
        let is_gaussian_identity = matches!(self.config.link_function(), LinkFunction::Identity);

        let s_tau_tau = std::sync::Arc::new(s_tau_tau);
        let pld = std::sync::Arc::new(pld);
        let s_k_unscaled =
            std::sync::Arc::new(Self::canonical_penalty_matrices(&self.canonical_penalties));
        let beta_eval = std::sync::Arc::new(beta_eval);
        let x_tau_terms = std::sync::Arc::new(x_tau_terms);
        let x_tau_beta_list = std::sync::Arc::new(x_tau_beta_list);
        let x_tau_tau = std::sync::Arc::new(x_tau_tau);
        let u = std::sync::Arc::new(u);
        let w_diag = std::sync::Arc::new(w_diag);
        let c_array = std::sync::Arc::new(c_array);
        let d_array = std::sync::Arc::new(d_array);
        let lambdas = std::sync::Arc::new(lambdas);
        let a_k_tau_j_mats = std::sync::Arc::new(a_k_tau_j_mats);
        let ds_k_dtau_j_mats = std::sync::Arc::new(ds_k_dtau_j_mats);
        let s_tau_list = std::sync::Arc::new(s_tau_list);

        let tau_tau_pair_fn = Self::build_tau_tau_pair_callback(
            basis,
            std::sync::Arc::clone(&pld),
            std::sync::Arc::clone(&s_tau_list),
            std::sync::Arc::clone(&s_tau_tau),
            std::sync::Arc::clone(&beta_eval),
            std::sync::Arc::clone(&x_design),
            std::sync::Arc::clone(&x_tau_terms),
            std::sync::Arc::clone(&x_tau_beta_list),
            std::sync::Arc::clone(&x_tau_tau),
            std::sync::Arc::clone(&u),
            std::sync::Arc::clone(&w_diag),
            std::sync::Arc::clone(&c_array),
            std::sync::Arc::clone(&d_array),
            p_dim,
            is_gaussian_identity,
        );

        let rho_tau_pair_fn = Self::build_rho_tau_pair_callback(
            std::sync::Arc::clone(&pld),
            std::sync::Arc::clone(&s_k_unscaled),
            std::sync::Arc::clone(&s_tau_list),
            std::sync::Arc::clone(&ds_k_dtau_j_mats),
            std::sync::Arc::clone(&lambdas),
            std::sync::Arc::clone(&a_k_tau_j_mats),
            std::sync::Arc::clone(&beta_eval),
            p_dim,
        );

        Ok((tau_tau_pair_fn, rho_tau_pair_fn))
    }

    pub(crate) fn build_tau_pair_callbacks_sparse_exact(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        self.build_tau_pair_callbacks_original_basis(rho, bundle, hyper_dirs)
    }

    /// Build pair callbacks for τ×τ and ρ×τ second-order [`HyperCoordPair`]
    /// entries.
    ///
    /// The returned closures produce `HyperCoordPair` objects on demand for:
    /// - `tau_tau_pair_fn(i, j)` : τ_i × τ_j entries
    /// - `rho_tau_pair_fn(k, j)` : ρ_k × τ_j entries
    ///
    /// These produce the second-order pair data used by the unified
    /// evaluator for τ-τ and ρ-τ Hessian blocks.
    ///
    /// # Field derivation (τ_i × τ_j pair)
    ///
    /// | Field   | Formula |
    /// | `a`     | `β̂^T S_{τ_i} β_{τ_j} + 0.5 β̂^T S_{τ_i τ_j} β̂` |
    /// | `g`     | second score involving X_{τ_i}, X_{τ_j}, X_{τ_i τ_j}, S_{τ_i τ_j}` |
    /// | `B`     | cross-design + cross-curvature + second-design + S_{τ_i τ_j}` |
    /// | `ld_s`  | `tr(S⁺ S_{τ_i τ_j}) − tr(S⁺ S_{τ_i} S⁺ S_{τ_j}) + 2 tr(Σ₊⁻² L_i L_jᵀ)` |
    ///
    /// # Notes
    ///
    /// The closures capture transformed design matrices, penalty objects, and
    /// the PIRLS state. They are `Send + Sync` so that the unified evaluator
    /// can call them from parallel contexts.
    ///
    /// All `a`, `g`, `B` fields are fixed-beta objects: they depend only on
    /// the current beta-hat and hyperparameters, not on IFT-derived mode
    /// responses. The IFT solves (beta_i, beta_j, beta_ij) are handled
    /// entirely by the unified evaluator after receiving these pair objects.
    pub(crate) fn build_tau_pair_callbacks(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
        }
        let p_dim = beta_eval.len();
        let lambdas = rho.mapv(f64::exp);

        let penalty_components_per_dir: Vec<Vec<PenaltyDerivativeComponent>> = hyper_dirs
            .iter()
            .map(|dir| {
                Self::transform_penalty_components(
                    dir.penalty_first_components(),
                    &reparam_result.qs,
                    free_basis_opt.as_ref(),
                )
            })
            .collect();
        let (s_tau_list, s_tau_tau, a_k_tau_j_mats, ds_k_dtau_j_mats) =
            Self::build_tau_penalty_derivative_data(
                rho,
                p_dim,
                hyper_dirs,
                &penalty_components_per_dir,
                |i, j| {
                    Ok(Self::transform_penalty_components(
                        &Self::get_pairwisesecond_penalty_components(hyper_dirs, i, j)?,
                        &reparam_result.qs,
                        free_basis_opt.as_ref(),
                    ))
                },
            )?;

        // ── Build PenaltyPseudologdet for ld_s pair computations ──
        //
        // Canonical penalty pseudo-logdeterminant: eigendecomposes S once and
        // provides tau_hessian_component / rho_tau_hessian_component methods
        // for all derivative queries on log|S|₊.
        let ct_eval: Vec<crate::construction::CanonicalPenalty> =
            if let Some(z) = free_basis_opt.as_ref() {
                reparam_result
                    .canonical_transformed
                    .iter()
                    .map(|cp| {
                        let projected_root = cp.root.dot(z);
                        crate::construction::CanonicalPenalty::from_dense_root(
                            projected_root,
                            z.ncols(),
                        )
                    })
                    .collect()
            } else {
                reparam_result.canonical_transformed.clone()
            };
        let pld = super::penalty_logdet::PenaltyPseudologdet::from_penalties(
            &ct_eval,
            &lambdas.as_slice().unwrap_or(&[]),
            bundle.ridge_passport.penalty_logdet_ridge(),
            p_dim,
        )
        .map_err(EstimationError::InvalidInput)?;
        let s_k_unscaled = Self::canonical_penalty_matrices(&ct_eval);

        let x_design = std::sync::Arc::new(
            build_active_design_matrix(&pirls_result.x_transformed, free_basis_opt.as_ref())
                .map_err(EstimationError::InvalidInput)?,
        );

        let qs_eval = std::sync::Arc::new(reparam_result.qs.clone());
        let free_basis_eval: std::sync::Arc<Option<Array2<f64>>> =
            std::sync::Arc::new(free_basis_opt.clone());
        let basis = TauPairBasis::Transformed {
            qs: Arc::clone(&qs_eval),
            free_basis_opt: Arc::clone(&free_basis_eval),
        };
        let (x_tau_terms, x_tau_beta_list, x_tau_tau) =
            Self::build_tau_design_data_in_basis(hyper_dirs, &basis, &beta_eval)?;

        // Working residual u = w ⊙ (z − η̂).
        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let w_diag = pirls_result.finalweights.clone();
        let c_array = pirls_result.solve_c_array.clone();
        let d_array = pirls_result.solve_d_array.clone();
        let is_gaussian_identity = matches!(self.config.link_function(), LinkFunction::Identity);

        // Capture into Arc for shared ownership in closures.
        let s_tau_tau = Arc::new(s_tau_tau);
        let pld = Arc::new(pld);
        let s_k_unscaled = Arc::new(s_k_unscaled);
        let s_tau_list = Arc::new(s_tau_list);
        let beta_eval = Arc::new(beta_eval);
        let x_tau_terms = Arc::new(x_tau_terms);
        let x_tau_beta_list = Arc::new(x_tau_beta_list);
        let x_tau_tau = Arc::new(x_tau_tau);
        let u = Arc::new(u);
        let w_diag = Arc::new(w_diag);
        let c_array = Arc::new(c_array);
        let d_array = Arc::new(d_array);
        let lambdas = Arc::new(lambdas);
        let a_k_tau_j_mats = Arc::new(a_k_tau_j_mats);
        let ds_k_dtau_j_mats = Arc::new(ds_k_dtau_j_mats);

        // ─── τ×τ pair callback ───────────────────────────────────────────
        let tau_tau_pair_fn = Self::build_tau_tau_pair_callback(
            basis,
            Arc::clone(&pld),
            Arc::clone(&s_tau_list),
            Arc::clone(&s_tau_tau),
            Arc::clone(&beta_eval),
            Arc::clone(&x_design),
            Arc::clone(&x_tau_terms),
            Arc::clone(&x_tau_beta_list),
            Arc::clone(&x_tau_tau),
            Arc::clone(&u),
            Arc::clone(&w_diag),
            Arc::clone(&c_array),
            Arc::clone(&d_array),
            p_dim,
            is_gaussian_identity,
        );

        let rho_tau_pair_fn = Self::build_rho_tau_pair_callback(
            Arc::clone(&pld),
            Arc::clone(&s_k_unscaled),
            Arc::clone(&s_tau_list),
            Arc::clone(&ds_k_dtau_j_mats),
            Arc::clone(&lambdas),
            Arc::clone(&a_k_tau_j_mats),
            Arc::clone(&beta_eval),
            p_dim,
        );

        Ok((tau_tau_pair_fn, rho_tau_pair_fn))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified HyperCoord builders for link parameters (SAS ε/log δ, mixture ρ)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build [`HyperCoord`] objects for SAS link parameters (ε, log δ).
    ///
    /// For each SAS parameter θ_j (j=0: epsilon, j=1: log_delta), the
    /// fixed-β derivative objects are:
    ///
    /// | Field   | Formula |
    /// | `a`     | `−∂ℓ/∂θ_j\|_{η fixed}` (direct likelihood derivative) |
    /// | `g`     | `−X^T (∂u/∂θ_j)\|_{η fixed}` (score sensitivity) |
    /// | `B`     | `X^T diag(∂W_explicit/∂θ_j) X` (working-weight drift) |
    /// | `ld_s`  | `0` (penalties don't depend on link parameters) |
    ///
    /// The working-weight derivative `∂W/∂θ_j` uses **observed information**
    /// weights, not Fisher. For a non-canonical link:
    ///   W_obs = W_F − (y−μ)·B,  B = (h''V − h'²V') / V²
    /// so the θ-derivative includes the residual correction:
    ///   ∂W_obs/∂θ = ∂W_F/∂θ + (dμ/dθ)·B − (y−μ)·∂B/∂θ
    ///
    /// This is exact for REML/LAML evaluation with learnable link
    /// parameters. The IFT-mediated `c ⊙ X dβ/dθ` part is handled
    /// separately by the unified evaluator's third-derivative correction.
    ///
    /// SAS epsilon reparameterization (tanh bounding) is NOT applied here;
    /// the caller should apply the chain rule `grad[ε_raw] *= d_eps/d_raw`
    /// after the unified evaluator returns.
    pub(crate) fn build_sas_link_ext_coords(
        &self,
        bundle: &EvalShared,
        sas_state: &crate::types::SasLinkState,
        is_beta_logistic: bool,
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        use crate::mixture_link::{
            beta_logistic_inverse_link_jetwith_param_partials,
            sas_inverse_link_jetwith_param_partials,
        };

        let pirls_result = bundle.pirls_result.as_ref();
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        // Guard: SAS link ext coords require dense design materialization.
        let n_x = pirls_result.x_transformed.nrows();
        let p_x = pirls_result.x_transformed.ncols();
        const LINK_EXT_MAX_DENSE_WORK: usize = 50_000_000;
        if n_x.saturating_mul(p_x) > LINK_EXT_MAX_DENSE_WORK {
            log::warn!(
                "skipping SAS link ext coordinate construction (n={n_x}, p={p_x}): \
                 dense design materialization too large"
            );
            return Ok(Vec::new());
        }

        // Transformed design matrix (dense required for link-param B construction).
        let x_dense_arc = pirls_result
            .x_transformed
            .try_to_dense_arc("build_sas_link_ext_coords requires dense transformed design")
            .map_err(EstimationError::InvalidInput)?;
        let x_dense_owned = free_basis_opt.as_ref().map(|z| {
            DenseRightProductView::new(x_dense_arc.as_ref())
                .with_factor(z)
                .materialize()
        });
        let x_dense = x_dense_owned
            .as_ref()
            .unwrap_or_else(|| x_dense_arc.as_ref());

        let p_dim = x_dense.ncols();
        let nobs = pirls_result.final_eta.len();
        let aux_dim = 2usize; // epsilon, log_delta

        if p_dim == 0 {
            return Ok((0..aux_dim)
                .map(|_| super::unified::HyperCoord {
                    a: 0.0,
                    g: Array1::zeros(0),
                    drift: super::unified::HyperCoordDrift::none(),
                    ld_s: 0.0,
                    b_depends_on_beta: true,
                    is_penalty_like: false,
                })
                .collect());
        }

        // Per-observation link jet with parameter partials.
        let mut direct_ll = [0.0_f64; 2];
        let mut du_by_j = [Array1::<f64>::zeros(nobs), Array1::<f64>::zeros(nobs)];
        let mut dw_explicit_by_j = [Array1::<f64>::zeros(nobs), Array1::<f64>::zeros(nobs)];

        for i in 0..nobs {
            let eta_i = pirls_result.final_eta[i].clamp(-30.0, 30.0);
            let jets = if is_beta_logistic {
                beta_logistic_inverse_link_jetwith_param_partials(
                    eta_i,
                    sas_state.log_delta,
                    sas_state.epsilon,
                )
            } else {
                sas_inverse_link_jetwith_param_partials(
                    eta_i,
                    sas_state.epsilon,
                    sas_state.log_delta,
                )
            };
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let yi = self.y[i];
            let wi = self.weights[i].max(0.0);
            let aux = link_binomial_aux(yi, wi, mu);

            for j in 0..aux_dim {
                let dj = if j == 0 {
                    jets.djet_depsilon
                } else {
                    jets.djet_dlog_delta
                };
                let dmu = dj.mu;
                let dd1 = dj.d1;
                let dd2 = dj.d2;
                direct_ll[j] += aux.a1 * dmu;
                du_by_j[j][i] = aux.a2 * dmu * d1 + aux.a1 * dd1;
                let variance = aux.variance;
                let variance_param = aux.variancemu_scale * dmu;
                let numerator = d1 * d1;
                let numerator_param = 2.0 * d1 * dd1;
                // Fisher weight derivative: ∂W_F/∂θ = ∂(h'²/V)/∂θ
                let dw_fisher = wi * (numerator_param * variance - numerator * variance_param)
                    / (variance * variance);

                // Observed correction: W_obs = W_F − (y−μ)·B where
                //   B = (h''·V − h'²·V') / V²
                // so ∂W_obs/∂θ = ∂W_F/∂θ + (dμ/dθ)·B − (y−μ)·∂B/∂θ
                //
                // For binomial: V = μ(1−μ), V' = 1−2μ, V'' = −2.
                let h2 = jets.jet.d2;
                let resid = yi - mu;
                let v_prime = aux.variancemu_scale; // 1 − 2μ
                let v_dprime = -2.0_f64; // V''(μ) for binomial

                let b_num = h2 * variance - numerator * v_prime;
                let var_sq = variance * variance;
                let b_val = b_num / var_sq;

                // ∂B/∂θ via quotient rule on b_num / V²:
                //   dV/dθ  = V'·dμ/dθ
                //   dV'/dθ = V''·dμ/dθ
                //   d(b_num)/dθ = dd2·V + h''·dV − 2·h'·dd1·V' − h'²·dV'
                let d_var = v_prime * dmu; // dV/dθ
                let d_vprime = v_dprime * dmu; // dV'/dθ
                let db_num =
                    dd2 * variance + h2 * d_var - 2.0 * d1 * dd1 * v_prime - numerator * d_vprime;
                // Quotient rule: (db_num·V² − b_num·2V·dV) / V⁴
                //              = (db_num − 2·B·dV) / V²
                let db_val = (db_num - 2.0 * b_val * d_var * variance) / var_sq;

                // OBSERVED weight derivative for the outer REML (see response.md Section 3):
                //   dW_obs/dtheta = dW_Fisher/dtheta + (dmu/dtheta)*B - (y-mu)*dB/dtheta
                // This is the exact Laplace derivative, not the PQL surrogate.
                dw_explicit_by_j[j][i] = dw_fisher + wi * (dmu * b_val - resid * db_val);
            }
        }

        // Build HyperCoord for each link parameter.
        let mut coords = Vec::with_capacity(aux_dim);
        let mut weighted_scratch = Array2::<f64>::zeros((0, 0));
        for j in 0..aux_dim {
            // a_j = dF/dθ_j|_{β fixed} = -dℓ/dθ_j|_{η fixed}.
            let a_j = -direct_ll[j];

            // g_j = -X^T (du/dθ_j) — the score sensitivity at fixed β.
            let g_j = {
                let xt_du = x_dense.t().dot(&du_by_j[j]);
                -xt_du
            };

            // B_j = X^T diag(dw_obs_j) X — the fixed-β observed Hessian drift.
            // Uses observed-information weight derivatives (not Fisher) for exact
            // REML/LAML with non-canonical links (SAS, beta-logistic).
            let b_j =
                Self::xt_diag_x_dense_into(x_dense, &dw_explicit_by_j[j], &mut weighted_scratch);

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                drift: super::unified::HyperCoordDrift::from_dense(b_j),
                ld_s: 0.0,
                // Link parameters affect working weights through the link,
                // and the working weights depend on β through η = Xβ,
                // so B_j depends on β.
                b_depends_on_beta: true,
                // Link parameters are design-moving (not penalty-like):
                // they change W through the link function, not through
                // penalty matrix derivatives. Not eligible for EFS.
                is_penalty_like: false,
            });
        }

        Ok(coords)
    }

    /// Build [`HyperCoord`] objects for mixture (blended inverse-link) logit
    /// parameters.
    ///
    /// Similar structure to SAS coords but with K−1 free logits instead of
    /// (ε, log δ). Each logit ρ_j controls the softmax weight of the j-th
    /// component link function.
    ///
    /// Uses **observed information** weight derivatives (not Fisher) for
    /// exact REML/LAML with non-canonical links. See the doc comment on
    /// `build_sas_link_ext_coords` for the mathematical derivation.
    pub(crate) fn build_mixture_link_ext_coords(
        &self,
        bundle: &EvalShared,
        mix_state: &crate::types::MixtureLinkState,
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        use crate::mixture_link::mixture_inverse_link_jetwith_rho_partials_into;

        let pirls_result = bundle.pirls_result.as_ref();
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        // Guard: mixture link ext coords require dense design materialization.
        let n_x = pirls_result.x_transformed.nrows();
        let p_x = pirls_result.x_transformed.ncols();
        const LINK_EXT_MAX_DENSE_WORK: usize = 50_000_000;
        if n_x.saturating_mul(p_x) > LINK_EXT_MAX_DENSE_WORK {
            log::warn!(
                "skipping mixture link ext coordinate construction (n={n_x}, p={p_x}): \
                 dense design materialization too large"
            );
            return Ok(Vec::new());
        }

        let x_dense_arc = pirls_result
            .x_transformed
            .try_to_dense_arc("build_mixture_link_ext_coords requires dense transformed design")
            .map_err(EstimationError::InvalidInput)?;
        let x_dense_owned = free_basis_opt.as_ref().map(|z| {
            DenseRightProductView::new(x_dense_arc.as_ref())
                .with_factor(z)
                .materialize()
        });
        let x_dense = x_dense_owned
            .as_ref()
            .unwrap_or_else(|| x_dense_arc.as_ref());

        let p_dim = x_dense.ncols();
        let nobs = pirls_result.final_eta.len();
        let aux_dim = mix_state.rho.len(); // K-1 free logits

        if aux_dim == 0 {
            return Ok(Vec::new());
        }

        if p_dim == 0 {
            return Ok((0..aux_dim)
                .map(|_| super::unified::HyperCoord {
                    a: 0.0,
                    g: Array1::zeros(0),
                    drift: super::unified::HyperCoordDrift::none(),
                    ld_s: 0.0,
                    b_depends_on_beta: true,
                    is_penalty_like: false,
                })
                .collect());
        }

        let mut direct_ll = vec![0.0_f64; aux_dim];
        let mut du_by_j: Vec<Array1<f64>> =
            (0..aux_dim).map(|_| Array1::<f64>::zeros(nobs)).collect();
        let mut dw_explicit_by_j: Vec<Array1<f64>> =
            (0..aux_dim).map(|_| Array1::<f64>::zeros(nobs)).collect();
        let mut mix_partials = vec![
            crate::mixture_link::InverseLinkJet {
                mu: 0.0,
                d1: 0.0,
                d2: 0.0,
                d3: 0.0,
            };
            aux_dim
        ];

        for i in 0..nobs {
            let jet = mixture_inverse_link_jetwith_rho_partials_into(
                mix_state,
                pirls_result.final_eta[i],
                &mut mix_partials,
            );
            let mu = jet.mu;
            let d1 = jet.d1;
            let yi = self.y[i];
            let wi = self.weights[i].max(0.0);
            let aux = link_binomial_aux(yi, wi, mu);

            for j in 0..aux_dim {
                let dj = mix_partials[j];
                let dmu = dj.mu;
                let dd1 = dj.d1;
                let dd2 = dj.d2;
                direct_ll[j] += aux.a1 * dmu;
                du_by_j[j][i] = aux.a2 * dmu * d1 + aux.a1 * dd1;
                let variance = aux.variance;
                let variance_param = aux.variancemu_scale * dmu;
                let numerator = d1 * d1;
                let numerator_param = 2.0 * d1 * dd1;
                // Fisher weight derivative: ∂W_F/∂θ = ∂(h'²/V)/∂θ
                let dw_fisher = wi * (numerator_param * variance - numerator * variance_param)
                    / (variance * variance);

                // Observed correction: W_obs = W_F − (y−μ)·B where
                //   B = (h''·V − h'²·V') / V²
                // so ∂W_obs/∂θ = ∂W_F/∂θ + (dμ/dθ)·B − (y−μ)·∂B/∂θ
                //
                // For binomial: V = μ(1−μ), V' = 1−2μ, V'' = −2.
                let h2 = jet.d2;
                let resid = yi - mu;
                let v_prime = aux.variancemu_scale; // 1 − 2μ
                let v_dprime = -2.0_f64; // V''(μ) for binomial

                let b_num = h2 * variance - numerator * v_prime;
                let var_sq = variance * variance;
                let b_val = b_num / var_sq;

                // ∂B/∂θ via quotient rule on b_num / V²:
                //   dV/dθ  = V'·dμ/dθ
                //   dV'/dθ = V''·dμ/dθ
                //   d(b_num)/dθ = dd2·V + h''·dV − 2·h'·dd1·V' − h'²·dV'
                let d_var = v_prime * dmu; // dV/dθ
                let d_vprime = v_dprime * dmu; // dV'/dθ
                let db_num =
                    dd2 * variance + h2 * d_var - 2.0 * d1 * dd1 * v_prime - numerator * d_vprime;
                let db_val = (db_num - 2.0 * b_val * d_var * variance) / var_sq;

                // OBSERVED weight derivative for the outer REML (see response.md Section 3):
                //   dW_obs/dtheta = dW_Fisher/dtheta + (dmu/dtheta)*B - (y-mu)*dB/dtheta
                // This is the exact Laplace derivative, not the PQL surrogate.
                dw_explicit_by_j[j][i] = dw_fisher + wi * (dmu * b_val - resid * db_val);
            }
        }

        let mut coords = Vec::with_capacity(aux_dim);
        let mut weighted_scratch = Array2::<f64>::zeros((0, 0));
        for j in 0..aux_dim {
            let a_j = -direct_ll[j];
            let g_j = {
                let xt_du = x_dense.t().dot(&du_by_j[j]);
                -xt_du
            };
            // B_j = X^T diag(dw_obs_j) X — the fixed-β observed Hessian drift.
            // Uses observed-information weight derivatives (not Fisher) for exact
            // REML/LAML with non-canonical links (mixture/blended).
            let b_j =
                Self::xt_diag_x_dense_into(x_dense, &dw_explicit_by_j[j], &mut weighted_scratch);

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                drift: super::unified::HyperCoordDrift::from_dense(b_j),
                ld_s: 0.0,
                b_depends_on_beta: true,
                is_penalty_like: false,
            });
        }

        Ok(coords)
    }
}

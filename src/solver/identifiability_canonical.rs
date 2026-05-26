// Cross-block identifiability canonicalisation.
//
// The pre-fit `audit_identifiability` (see `identifiability_audit.rs`)
// runs a joint RRQR on `[X_block_0 | X_block_1 | ...]` and reports per-
// block (block_idx, local_col) drops attributing each demoted joint
// column back to its origin. This module converts that report into a
// concrete coordinate transform applied to the inner solver: each
// block's design is wrapped via `CoefficientTransformOperator` with a
// (p_raw × r_reduced) selection matrix `T_i`, its penalties are
// pulled back as `T_iᵀ S_k T_i`, and the inner solve operates on
// reduced specs. Coefficients in the reduced space lift back to the
// raw space via `β_raw = T_i θ`.
//
// The canonicalisation accepts joint rank deficiency: if the audit
// returns `fatal=true` because `joint_rank < p_total` and RRQR
// attributed the drops, we proceed with the reduced specs rather than
// refusing the fit. A residual fatal case with empty attribution
// (>2-way structural alias the RRQR couldn't deterministically pin to
// a single block) is the only condition that still aborts.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use crate::families::custom_family::{CustomFamilyError, ParameterBlockSpec, PenaltyMatrix};
use crate::linalg::matrix::{
    CoefficientTransformOperator, DenseDesignMatrix, DesignMatrix,
};
use crate::solver::identifiability_audit::{IdentifiabilityAudit, audit_identifiability};

/// Specs after pre-fit cross-block identifiability canonicalisation.
///
/// `reduced_specs[i]` carries an `r_i`-column design wrapping the raw
/// `p_i`-column design via `CoefficientTransformOperator`. Penalties
/// are pulled back as `T_iᵀ S_k T_i`. `per_block_transform[i]` is the
/// raw-to-reduced selection matrix `T_i` of shape `(p_i_raw, r_i)`.
pub struct CanonicalSpecs {
    pub reduced_specs: Vec<ParameterBlockSpec>,
    pub per_block_transform: Vec<Array2<f64>>,
    pub audit: IdentifiabilityAudit,
}

impl CanonicalSpecs {
    /// Lift reduced-space block coefficients θ_i back to the raw space
    /// via `β_i_raw = T_i · θ_i`. Dropped raw coordinates receive zero.
    pub fn lift_block_betas_to_raw(&self, theta_blocks: &[Array1<f64>]) -> Vec<Array1<f64>> {
        assert_eq!(
            theta_blocks.len(),
            self.per_block_transform.len(),
            "lift_block_betas_to_raw: theta blocks ({}) != transforms ({})",
            theta_blocks.len(),
            self.per_block_transform.len(),
        );
        let mut out = Vec::with_capacity(theta_blocks.len());
        for (theta, transform) in theta_blocks.iter().zip(self.per_block_transform.iter()) {
            assert_eq!(
                theta.len(),
                transform.ncols(),
                "lift_block_betas_to_raw: theta length {} != transform ncols {}",
                theta.len(),
                transform.ncols(),
            );
            out.push(transform.dot(theta));
        }
        out
    }

    /// Raw block dimensions (rows of each `T_i`). Used to bound expansion.
    pub fn raw_block_dims(&self) -> Vec<usize> {
        self.per_block_transform
            .iter()
            .map(|t| t.nrows())
            .collect()
    }

    /// Reduced block dimensions (cols of each `T_i`).
    pub fn reduced_block_dims(&self) -> Vec<usize> {
        self.per_block_transform
            .iter()
            .map(|t| t.ncols())
            .collect()
    }

    /// Lift a reduced-space joint matrix `M_red` (total_r × total_r) to
    /// raw-space (total_p × total_p) via `T_full · M_red · T_fullᵀ`
    /// where `T_full = blockdiag(T_i)`. For selection-T this places the
    /// reduced block at surviving raw indices and leaves the rest zero.
    pub fn lift_joint_matrix_to_raw(&self, m_red: &Array2<f64>) -> Array2<f64> {
        let raw_dims = self.raw_block_dims();
        let red_dims = self.reduced_block_dims();
        let total_p: usize = raw_dims.iter().sum();
        let total_r: usize = red_dims.iter().sum();
        assert_eq!(
            m_red.nrows(),
            total_r,
            "lift_joint_matrix_to_raw: m_red rows {} != total reduced dim {}",
            m_red.nrows(),
            total_r,
        );
        assert_eq!(
            m_red.ncols(),
            total_r,
            "lift_joint_matrix_to_raw: m_red cols {} != total reduced dim {}",
            m_red.ncols(),
            total_r,
        );
        let mut out = Array2::<f64>::zeros((total_p, total_p));
        let mut raw_off_i = 0usize;
        let mut red_off_i = 0usize;
        for (i, t_i) in self.per_block_transform.iter().enumerate() {
            let p_i = raw_dims[i];
            let r_i = red_dims[i];
            let mut raw_off_j = 0usize;
            let mut red_off_j = 0usize;
            for (j, t_j) in self.per_block_transform.iter().enumerate() {
                let p_j = raw_dims[j];
                let r_j = red_dims[j];
                if r_i > 0 && r_j > 0 {
                    let m_ij = m_red.slice(ndarray::s![
                        red_off_i..red_off_i + r_i,
                        red_off_j..red_off_j + r_j
                    ]);
                    let lifted = t_i.dot(&m_ij).dot(&t_j.t());
                    out.slice_mut(ndarray::s![
                        raw_off_i..raw_off_i + p_i,
                        raw_off_j..raw_off_j + p_j
                    ])
                    .assign(&lifted);
                }
                raw_off_j += p_j;
                red_off_j += r_j;
            }
            raw_off_i += p_i;
            red_off_i += r_i;
        }
        out
    }
}

/// Run the pre-fit cross-block identifiability audit and produce
/// canonicalised specs.
///
/// Behaviour:
///   - If the audit reports `dropped_columns` (whether or not it set
///     `fatal=true`), we build per-block selection matrices that drop
///     those local columns and proceed with the reduced specs.
///   - If the audit is `fatal=true` *without* attributed drops, we
///     refuse: this is the >2-way structural alias case where RRQR
///     couldn't pin the redundancy on a single (block, local_col)
///     pair, and silently absorbing it would change model semantics
///     beyond what canonicalisation can repair.
///   - If the audit cleanly passes (no drops, not fatal), each `T_i`
///     is the identity and the reduced specs equal the raw specs
///     modulo cloning.
pub fn canonicalize_for_identifiability(
    specs: &[ParameterBlockSpec],
) -> Result<CanonicalSpecs, CustomFamilyError> {
    let audit = audit_identifiability(specs).map_err(|reason| {
        CustomFamilyError::DimensionMismatch {
            reason: format!("pre-fit identifiability audit failed: {reason}"),
        }
    })?;

    if audit.fatal && audit.dropped_columns.is_empty() {
        return Err(CustomFamilyError::IdentifiabilityFailure { audit });
    }

    let mut per_block_transform: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    let mut reduced_specs: Vec<ParameterBlockSpec> = Vec::with_capacity(specs.len());

    for (block_idx, spec) in specs.iter().enumerate() {
        let p_raw = spec.design.ncols();
        let dropped_locals: Vec<usize> = audit
            .dropped_columns
            .iter()
            .enumerate()
            .filter_map(|(drop_pos, drop)| {
                if drop.block == spec.name {
                    let block_matches_audit = audit
                        .blocks
                        .get(block_idx)
                        .map(|b| b.block_name == spec.name)
                        .unwrap_or(false);
                    if block_matches_audit {
                        Some(drop.column)
                    } else {
                        let _ = drop_pos;
                        Some(drop.column)
                    }
                } else {
                    None
                }
            })
            .collect();
        let mut dropped_sorted = dropped_locals.clone();
        dropped_sorted.sort_unstable();
        dropped_sorted.dedup();
        for &col in &dropped_sorted {
            if col >= p_raw {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "canonicalize_for_identifiability: audit reported dropped column \
                         {col} for block '{}' which has only {} columns",
                        spec.name, p_raw,
                    ),
                });
            }
        }
        let kept: Vec<usize> = (0..p_raw)
            .filter(|c| !dropped_sorted.binary_search(c).is_ok())
            .collect();
        let r_block = kept.len();

        let mut t_i = Array2::<f64>::zeros((p_raw, r_block));
        for (col_out, &raw_col) in kept.iter().enumerate() {
            t_i[[raw_col, col_out]] = 1.0;
        }

        let reduced_design = if dropped_sorted.is_empty() {
            spec.design.clone()
        } else {
            build_reduced_design(&spec.design, &kept, &spec.name, &t_i)?
        };

        let reduced_penalties: Vec<PenaltyMatrix> = spec
            .penalties
            .iter()
            .map(|p| pull_back_penalty(p, &kept))
            .collect();

        let reduced_initial_beta = match &spec.initial_beta {
            Some(beta_raw) => {
                if beta_raw.len() != p_raw {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "canonicalize_for_identifiability: block '{}' initial_beta \
                             length {} != design ncols {}",
                            spec.name,
                            beta_raw.len(),
                            p_raw,
                        ),
                    });
                }
                let mut theta = Array1::<f64>::zeros(r_block);
                for (out_idx, &raw_col) in kept.iter().enumerate() {
                    theta[out_idx] = beta_raw[raw_col];
                }
                Some(theta)
            }
            None => None,
        };

        reduced_specs.push(ParameterBlockSpec {
            name: spec.name.clone(),
            design: reduced_design,
            offset: spec.offset.clone(),
            penalties: reduced_penalties,
            // Pulled-back penalties may carry an enlarged structural
            // nullspace (a column dropped from a smooth's pure-span
            // basis adds that direction to the penalty kernel).
            // Falling back to eigenvalue-based rank detection in the
            // pseudo-logdet path is the safe choice when the
            // selection-T pullback changes the kernel structurally.
            nullspace_dims: Vec::new(),
            initial_log_lambdas: spec.initial_log_lambdas.clone(),
            initial_beta: reduced_initial_beta,
        });
        per_block_transform.push(t_i);
    }

    Ok(CanonicalSpecs {
        reduced_specs,
        per_block_transform,
        audit,
    })
}

fn build_reduced_design(
    raw: &DesignMatrix,
    kept: &[usize],
    block_name: &str,
    t_i: &Array2<f64>,
) -> Result<DesignMatrix, CustomFamilyError> {
    let inner_dense = match raw {
        DesignMatrix::Dense(d) => d.clone(),
        DesignMatrix::Sparse(_) => {
            let dense = raw
                .try_to_dense_by_chunks(&format!(
                    "canonicalize_for_identifiability sparse->dense block '{block_name}'"
                ))
                .map_err(|reason| CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "canonicalize_for_identifiability: densify sparse block '{block_name}' \
                         failed: {reason}"
                    ),
                })?;
            DenseDesignMatrix::from(dense)
        }
    };
    // Hot path: when the inner is already a materialised dense Array2,
    // slice the kept columns directly. This avoids carrying the full
    // raw-width inner through every PIRLS iteration when many columns
    // were dropped.
    if let Some(arr) = inner_dense.as_dense_ref() {
        let reduced = Array2::<f64>::from_shape_fn((arr.nrows(), kept.len()), |(i, j)| {
            arr[[i, kept[j]]]
        });
        return Ok(DesignMatrix::Dense(DenseDesignMatrix::from(reduced)));
    }
    // Operator-backed inner (Lazy): preserve the operator structure
    // by wrapping with CoefficientTransformOperator on the selection T.
    let op = CoefficientTransformOperator::new(inner_dense, t_i.clone()).map_err(|reason| {
        CustomFamilyError::DimensionMismatch {
            reason: format!(
                "canonicalize_for_identifiability: build CoefficientTransformOperator \
                 for block '{block_name}': {reason}"
            ),
        }
    })?;
    Ok(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op))))
}

fn pull_back_penalty(penalty: &PenaltyMatrix, kept: &[usize]) -> PenaltyMatrix {
    let label = penalty.precision_label().map(|s| s.to_string());
    let dense = penalty.as_dense_cow();
    let reduced = Array2::<f64>::from_shape_fn((kept.len(), kept.len()), |(i, j)| {
        dense[[kept[i], kept[j]]]
    });
    let base = PenaltyMatrix::Dense(reduced);
    match label {
        Some(lbl) => base.with_precision_label(lbl),
        None => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::matrix::DenseDesignMatrix;
    use ndarray::Array2;

    fn spec_from_dense(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
        let n = design.nrows();
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
            offset: Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
        }
    }

    fn linspace(n: usize) -> Array1<f64> {
        if n <= 1 {
            return Array1::<f64>::zeros(n.max(1));
        }
        let step = 2.0 / (n as f64 - 1.0);
        Array1::from_iter((0..n).map(|i| -1.0 + step * i as f64))
    }

    #[test]
    fn canonical_clean_specs_identity_transform() {
        let n = 32;
        let x = linspace(n);
        let mut p = Array2::<f64>::zeros((n, 2));
        let mut s = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            p[[i, 0]] = 1.0;
            p[[i, 1]] = x[i];
            s[[i, 0]] = x[i] * x[i];
            s[[i, 1]] = x[i] * x[i] * x[i];
        }
        let specs = [spec_from_dense("p", p), spec_from_dense("s", s)];
        let canon =
            canonicalize_for_identifiability(&specs).expect("clean canonical must succeed");
        assert_eq!(canon.reduced_specs.len(), 2);
        assert_eq!(canon.per_block_transform[0].dim(), (2, 2));
        assert_eq!(canon.per_block_transform[1].dim(), (2, 2));
        let theta = vec![Array1::from(vec![0.5, -0.25]), Array1::from(vec![1.0, 2.0])];
        let raw = canon.lift_block_betas_to_raw(&theta);
        assert_eq!(raw[0].as_slice().unwrap(), &[0.5, -0.25]);
        assert_eq!(raw[1].as_slice().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn canonical_drops_aliased_column() {
        let n = 64;
        let x = linspace(n);
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        let mut smooth = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            smooth[[i, 0]] = 1.0;
            smooth[[i, 1]] = x[i] * x[i];
            smooth[[i, 2]] = x[i] * x[i] * x[i];
        }
        let specs = [
            spec_from_dense("intercept", parametric),
            spec_from_dense("smooth_with_const", smooth),
        ];
        let canon =
            canonicalize_for_identifiability(&specs).expect("aliased canonical must reduce");
        let total_red: usize = canon.reduced_block_dims().iter().sum();
        assert_eq!(
            total_red, 3,
            "expected 3 surviving directions; got {total_red}"
        );
        let raw_dims: Vec<usize> = canon.raw_block_dims();
        assert_eq!(raw_dims, vec![1, 3]);
        let theta: Vec<Array1<f64>> = canon
            .reduced_block_dims()
            .iter()
            .map(|&r| Array1::from_iter((0..r).map(|j| (j + 1) as f64)))
            .collect();
        let raw_betas = canon.lift_block_betas_to_raw(&theta);
        let zero_count: usize = raw_betas
            .iter()
            .map(|b| b.iter().filter(|&&v| v == 0.0).count())
            .sum();
        assert!(
            zero_count >= 1,
            "at least one dropped raw coordinate must lift to zero; raw_betas={:?}",
            raw_betas,
        );
    }

    #[test]
    fn canonical_pulls_back_penalty_to_kept_indices() {
        let n = 32;
        let x = linspace(n);
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        let mut smooth = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            smooth[[i, 0]] = 1.0;
            smooth[[i, 1]] = x[i] * x[i];
            smooth[[i, 2]] = x[i] * x[i] * x[i];
        }
        let mut smooth_spec = spec_from_dense("smooth_with_const", smooth);
        let mut s = Array2::<f64>::zeros((3, 3));
        s[[1, 1]] = 4.0;
        s[[2, 2]] = 9.0;
        s[[1, 2]] = 1.5;
        s[[2, 1]] = 1.5;
        smooth_spec.penalties = vec![PenaltyMatrix::Dense(s.clone())];
        smooth_spec.initial_log_lambdas = Array1::from(vec![0.0]);
        let specs = [
            spec_from_dense("intercept", parametric),
            smooth_spec,
        ];
        let canon = canonicalize_for_identifiability(&specs)
            .expect("aliased canonical with penalty must reduce");
        let smooth_reduced = &canon.reduced_specs[1];
        assert_eq!(smooth_reduced.penalties.len(), 1);
        let dense_red = smooth_reduced.penalties[0].as_dense_cow().into_owned();
        let r = smooth_reduced.design.ncols();
        assert_eq!(dense_red.dim(), (r, r));
    }
}

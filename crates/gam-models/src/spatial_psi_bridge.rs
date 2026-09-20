//! Spatial-ψ derivative bridge.
//!
//! Self-contained translation layer between the smooth-side
//! [`TermCollectionSpec`]/[`TermCollectionDesign`] world and the lower-level
//! generic family engine ([`crate::custom_family`]). It takes the
//! resolved spatial length-scale terms and produces the per-axis
//! [`CustomFamilyBlockPsiDerivative`] blocks the engine consumes.
//!
//! Keeping this here (a *higher* layer than `custom_family`) lets the engine
//! stay ignorant of `gam_terms::smooth`: family modules call into this
//! bridge instead of the engine reaching up into smooth.

use crate::custom_family::{
    CustomFamilyBlockPsiDerivative, CustomFamilyPsiDerivativeOperator,
    EmbeddedImplicitPsiDerivativeOperator, build_embedded_dense_psi_operator,
};
use crate::fit_orchestration::drivers::{
    spatial_length_scale_term_indices, try_build_spatial_log_kappa_derivativeinfo_list,
};
use gam_linalg::matrix::EmbeddedSquareBlock;
use gam_terms::smooth::{TermCollectionDesign, TermCollectionSpec};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

pub(crate) fn wrap_spatial_implicit_psi_operator(
    op: Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
    global_range: Range<usize>,
    total_p: usize,
) -> Arc<dyn CustomFamilyPsiDerivativeOperator> {
    Arc::new(
        EmbeddedImplicitPsiDerivativeOperator::new(op, global_range, total_p)
            .expect("spatial implicit psi operator should embed into full coefficient space"),
    )
}

/// Per-block transform applied by the shared spatial-ψ derivative engine.
///
/// The engine (see [`build_block_spatial_psi_derivatives_with_transform`]) owns
/// the policy of *which* spatial length-scale terms become ψ-derivative blocks,
/// how their embedded design/penalty matrices and implicit operators are
/// assembled, and how anisotropic cross-axis rows are wired. Family modules that
/// need a coordinate change applied uniformly to every assembled block — e.g. a
/// time-varying survival covariate that tensorizes each spatial design row
/// against a time basis — invert the dependency by *providing a transform* here
/// instead of re-implementing the assembly loop.
///
/// Every hook defaults to the identity, so the canonical (untransformed)
/// path is just [`build_block_spatial_psi_derivatives`].
pub(crate) trait SpatialPsiBlockTransform {
    /// Transform an assembled implicit ψ-derivative operator (already embedded
    /// into the full coefficient space). The default returns it unchanged.
    fn transform_operator(
        &self,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    ) -> Result<Arc<dyn CustomFamilyPsiDerivativeOperator>, String> {
        Ok(op)
    }

    /// Transform a materialized (already embedded) penalty block. Default: identity.
    fn transform_penalty(&self, penalty: Array2<f64>) -> Array2<f64> {
        penalty
    }

    /// ψ-derivatives of penalties the transform adds beside the term
    /// collection's own (a time margin's `G_x ⊗ S_t`, whose covariate Gram moves
    /// with every spatial axis of the block), given each axis's untransformed
    /// design derivative. Default: the transform owns no penalties.
    fn owned_penalty_psi_components(
        &self,
        axes: &[SpatialPsiAxisDesign],
    ) -> Result<OwnedPenaltyPsiComponents, String> {
        Ok(OwnedPenaltyPsiComponents::none(axes.len()))
    }
}

/// One spatial ψ axis of a block as the engine assembles it, before any
/// transform: the embedded design-derivative operator (`None` when the axis
/// moves no design column), the operator's axis index, the anisotropy group and
/// the block columns the axis moves.
pub(crate) struct SpatialPsiAxisDesign {
    pub(crate) operator: Option<Arc<dyn CustomFamilyPsiDerivativeOperator>>,
    pub(crate) axis: usize,
    pub(crate) group: Option<usize>,
    pub(crate) range: Range<usize>,
}

/// ψ-derivative components of penalties a transform owns: `first[a]` and
/// `second[a][b]`, each a list of `(block penalty index, matrix)`.
pub(crate) struct OwnedPenaltyPsiComponents {
    pub(crate) first: Vec<Vec<(usize, Array2<f64>)>>,
    pub(crate) second: Vec<Vec<Vec<(usize, Array2<f64>)>>>,
}

impl OwnedPenaltyPsiComponents {
    /// No owned penalties along `psi_dim` axes.
    pub(crate) fn none(psi_dim: usize) -> Self {
        Self {
            first: vec![Vec::new(); psi_dim],
            second: vec![vec![Vec::new(); psi_dim]; psi_dim],
        }
    }
}

/// The canonical no-op transform: blocks are emitted exactly as assembled.
pub(crate) struct IdentitySpatialPsiBlockTransform;

impl SpatialPsiBlockTransform for IdentitySpatialPsiBlockTransform {}

/// Exact coefficient-chart transform for spatial design and penalty jets.
///
/// `raw_from_current` is the fixed section `T` in `beta_raw = T beta_current`.
/// Every derivative artifact is moved through the same chart:
/// `X_psi -> X_psi T` (through the matrix-free first and second
/// design-derivative actions) and `S_psi -> T' S_psi T`.
pub(crate) struct CoefficientSpatialPsiBlockTransform {
    raw_from_current: Arc<Array2<f64>>,
}

impl CoefficientSpatialPsiBlockTransform {
    pub(crate) fn new(raw_from_current: &Array2<f64>) -> Result<Self, String> {
        if raw_from_current.iter().any(|value| !value.is_finite()) {
            return Err(
                "spatial psi coefficient transform contains a non-finite value".to_string(),
            );
        }
        Ok(Self {
            raw_from_current: Arc::new(raw_from_current.clone()),
        })
    }
}

struct CoefficientTransformedPsiDerivativeOperator {
    base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    raw_from_current: Arc<Array2<f64>>,
}

impl CoefficientTransformedPsiDerivativeOperator {
    fn current_to_raw(
        &self,
        current: &ArrayView1<'_, f64>,
        context: &str,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        if current.len() != self.raw_from_current.ncols() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "{context}: current coefficient length {} does not match transform width {}",
                current.len(),
                self.raw_from_current.ncols(),
            )));
        }
        Ok(self.raw_from_current.dot(current))
    }

    fn raw_transpose_to_current(
        &self,
        raw: &Array1<f64>,
        context: &str,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        if raw.len() != self.raw_from_current.nrows() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "{context}: raw coefficient length {} does not match transform height {}",
                raw.len(),
                self.raw_from_current.nrows(),
            )));
        }
        Ok(self.raw_from_current.t().dot(raw))
    }

    fn transform_rows(
        &self,
        raw: Array2<f64>,
        context: &str,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        if raw.ncols() != self.raw_from_current.nrows() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "{context}: raw row width {} does not match transform height {}",
                raw.ncols(),
                self.raw_from_current.nrows(),
            )));
        }
        Ok(raw.dot(self.raw_from_current.as_ref()))
    }
}

impl CustomFamilyPsiDerivativeOperator for CoefficientTransformedPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.base.n_data()
    }

    fn p_out(&self) -> usize {
        self.raw_from_current.ncols()
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let raw = self.base.transpose_mul(axis, v)?;
        self.raw_transpose_to_current(&raw, "coefficient-transformed psi transpose_mul")
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let raw = self.current_to_raw(u, "coefficient-transformed psi forward_mul")?;
        self.base.forward_mul(axis, &raw.view())
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let raw = self.base.transpose_mul_second_diag(axis, v)?;
        self.raw_transpose_to_current(
            &raw,
            "coefficient-transformed psi transpose_mul_second_diag",
        )
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let raw = self
            .base
            .transpose_mul_second_cross(axis_d, axis_e, v)?;
        self.raw_transpose_to_current(
            &raw,
            "coefficient-transformed psi transpose_mul_second_cross",
        )
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let raw = self.current_to_raw(
            u,
            "coefficient-transformed psi forward_mul_second_diag",
        )?;
        self.base.forward_mul_second_diag(axis, &raw.view())
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, gam_terms::basis::BasisError> {
        let raw = self.current_to_raw(
            u,
            "coefficient-transformed psi forward_mul_second_cross",
        )?;
        self.base
            .forward_mul_second_cross(axis_d, axis_e, &raw.view())
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        let raw = self.base.row_chunk_first(axis, rows)?;
        self.transform_rows(raw, "coefficient-transformed psi row_chunk_first")
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), gam_terms::basis::BasisError> {
        if out.len() != self.p_out() {
            return Err(gam_terms::basis::BasisError::Other(format!(
                "coefficient-transformed psi row output length {} does not match {}",
                out.len(),
                self.p_out(),
            )));
        }
        let mut raw = Array1::<f64>::zeros(self.raw_from_current.nrows());
        self.base
            .row_vector_first_into(axis, row, raw.view_mut())?;
        out.assign(&self.raw_from_current.t().dot(&raw));
        Ok(())
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        let raw = self.base.row_chunk_second_diag(axis, rows)?;
        self.transform_rows(
            raw,
            "coefficient-transformed psi row_chunk_second_diag",
        )
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, gam_terms::basis::BasisError> {
        let raw = self
            .base
            .row_chunk_second_cross(axis_d, axis_e, rows)?;
        self.transform_rows(
            raw,
            "coefficient-transformed psi row_chunk_second_cross",
        )
    }
}

impl SpatialPsiBlockTransform for CoefficientSpatialPsiBlockTransform {
    fn transform_operator(
        &self,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    ) -> Result<Arc<dyn CustomFamilyPsiDerivativeOperator>, String> {
        if op.p_out() != self.raw_from_current.nrows() {
            return Err(format!(
                "spatial psi operator width {} does not match coefficient transform height {}",
                op.p_out(),
                self.raw_from_current.nrows(),
            ));
        }
        Ok(Arc::new(CoefficientTransformedPsiDerivativeOperator {
            base: op,
            raw_from_current: Arc::clone(&self.raw_from_current),
        }))
    }

    fn transform_penalty(&self, penalty: Array2<f64>) -> Array2<f64> {
        self.raw_from_current
            .t()
            .dot(&penalty.dot(self.raw_from_current.as_ref()))
    }
}

pub(crate) fn build_block_spatial_psi_derivatives(
    data: ndarray::ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<Option<Vec<CustomFamilyBlockPsiDerivative>>, String> {
    build_block_spatial_psi_derivatives_with_transform(
        data,
        resolvedspec,
        design,
        &IdentitySpatialPsiBlockTransform,
    )
}

/// Shared exact-derivative / spatial-ψ engine.
///
/// Builds the per-axis [`CustomFamilyBlockPsiDerivative`] blocks for every
/// spatial length-scale term, threading every materialized penalty matrix and
/// every assembled design-derivative operator through `transform`. Every axis
/// that moves a design column carries an operator (implicit, or the embedded
/// dense one over `x_psi_local`/`x_psi_psi_local` and the anisotropic cross
/// designs), so the design derivative is never also shipped materialized.
/// Family modules consume this engine and supply a [`SpatialPsiBlockTransform`]
/// rather than duplicating the block-assembly, cross-axis, and
/// operator-embedding logic.
pub(crate) fn build_block_spatial_psi_derivatives_with_transform(
    data: ndarray::ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    transform: &dyn SpatialPsiBlockTransform,
) -> Result<Option<Vec<CustomFamilyBlockPsiDerivative>>, String> {
    let spatial_terms = spatial_length_scale_term_indices(resolvedspec);
    let Some(info_list) =
        try_build_spatial_log_kappa_derivativeinfo_list(data, resolvedspec, design, &spatial_terms)
            .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    let psi_dim = info_list.len();
    let axis_lookup: HashMap<(usize, usize), usize> = info_list
        .iter()
        .enumerate()
        .filter_map(|(idx, info)| {
            info.aniso_group_id
                .map(|gid| ((gid, info.implicit_axis), idx))
        })
        .collect();
    let axes = info_list
        .iter()
        .map(|info| -> Result<SpatialPsiAxisDesign, String> {
            let operator = match info.implicit_operator.as_ref() {
                Some(op) => Some(wrap_spatial_implicit_psi_operator(
                    Arc::clone(op),
                    info.global_range.clone(),
                    info.total_p,
                )),
                None if !info.x_psi_local.is_empty() => Some(
                    build_embedded_dense_psi_operator(
                        &info.x_psi_local,
                        &info.x_psi_psi_local,
                        info.aniso_cross_designs.as_ref(),
                        info.global_range.clone(),
                        info.total_p,
                        info.implicit_axis,
                    )
                    .map_err(|error| error.to_string())?,
                ),
                None => None,
            };
            Ok(SpatialPsiAxisDesign {
                operator,
                axis: info.implicit_axis,
                group: info.aniso_group_id,
                range: info.global_range.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut owned = transform.owned_penalty_psi_components(&axes)?;
    if owned.first.len() != psi_dim
        || owned.second.len() != psi_dim
        || owned.second.iter().any(|row| row.len() != psi_dim)
    {
        return Err(format!(
            "spatial psi transform published owned penalty components for {} axes, the block has \
             {psi_dim}",
            owned.first.len()
        ));
    }
    let collected: Result<Vec<CustomFamilyBlockPsiDerivative>, String> = info_list
        .into_iter()
        .enumerate()
        .map(|(psi_idx, info)| {
            let design_operator = axes[psi_idx]
                .operator
                .clone()
                .map(|op| transform.transform_operator(op))
                .transpose()?;
            let penalty_indices = info.penalty_indices.clone();
            let embed_penalty = |local: &Array2<f64>| -> Array2<f64> {
                let embedded = if local.nrows() == 0 || local.ncols() == 0 {
                    Array2::<f64>::zeros((info.total_p, info.total_p))
                } else {
                    EmbeddedSquareBlock::new(local, info.global_range.clone(), info.total_p)
                        .materialize()
                };
                transform.transform_penalty(embedded)
            };
            let mut s_components: Vec<(usize, Array2<f64>)> = info
                .penalty_indices
                .into_iter()
                .zip(
                    info.s_psi_components_local
                        .into_iter()
                        .map(|local| embed_penalty(&local)),
                )
                .collect();
            s_components.extend(std::mem::take(&mut owned.first[psi_idx]));
            // Build s_psi_psi_components with cross-penalty terms
            let mut s_psi_psi_comp_rows = vec![Vec::<(usize, Array2<f64>)>::new(); psi_dim];
            s_psi_psi_comp_rows[psi_idx] = penalty_indices
                .iter()
                .copied()
                .zip(info.s_psi_psi_components_local.iter().map(&embed_penalty))
                .collect();
            if let (Some(gid), Some(cross_penalty_provider)) = (
                info.aniso_group_id,
                info.aniso_cross_penalty_provider.as_ref(),
            ) {
                for ((group_id, axis_j), global_j) in &axis_lookup {
                    if *group_id != gid || *axis_j == info.implicit_axis {
                        continue;
                    }
                    let local_components =
                        cross_penalty_provider(*axis_j).map_err(|err| err.to_string())?;
                    if local_components.is_empty() {
                        continue;
                    }
                    s_psi_psi_comp_rows[*global_j] = penalty_indices
                        .iter()
                        .copied()
                        .zip(local_components.iter().map(embed_penalty))
                        .collect();
                }
            }
            for (row, owned_row) in s_psi_psi_comp_rows
                .iter_mut()
                .zip(std::mem::take(&mut owned.second[psi_idx]))
            {
                row.extend(owned_row);
            }
            Ok(CustomFamilyBlockPsiDerivative {
                penalty_index: Some(info.penalty_index),
                x_psi: Array2::<f64>::zeros((0, 0)),
                s_psi: Array2::<f64>::zeros((0, 0)),
                s_psi_components: Some(s_components),
                s_psi_penalty_components: None,
                x_psi_psi: None,
                s_psi_psi: None,
                s_psi_psi_components: Some(s_psi_psi_comp_rows),
                s_psi_psi_penalty_components: None,
                implicit_operator: design_operator,
                implicit_axis: info.implicit_axis,
                implicit_group_id: info.aniso_group_id,
            })
        })
        .collect();
    Ok(Some(collected?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit_orchestration::drivers::freeze_term_collection_from_design;
    use gam_terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use gam_terms::smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, build_term_collection_design,
    };

    #[test]
    fn build_block_spatial_psi_derivatives_supports_3d_aniso_matern() {
        let n = 24usize;
        let mut data = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
            data[[i, 2]] = (2.5 * std::f64::consts::PI * t).cos();
        }

        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "spatial".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::EqualMass { num_centers: 6 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(0.45),
                        nu: MaternNu::ThreeHalves,
                        include_intercept: false,
                        double_penalty: false,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: Some(vec![0.0, 0.0, 0.0]),
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None.into(),
                joint_null_rotation: None,
            }],
            level: Default::default(),
        };
        let base_design =
            build_term_collection_design(data.view(), &spec).expect("build base spatial design");
        let resolvedspec = freeze_term_collection_from_design(&spec, &base_design)
            .expect("freeze spatial term spec");
        let resolved_design = build_term_collection_design(data.view(), &resolvedspec)
            .expect("rebuild frozen spatial design");
        let derivs =
            build_block_spatial_psi_derivatives(data.view(), &resolvedspec, &resolved_design)
                .expect("3D anisotropic Matern psi derivatives should build")
                .expect("3D anisotropic Matern psi derivatives should be present");
        assert_eq!(derivs.len(), 3);
        assert!(derivs.iter().all(|deriv| deriv.implicit_operator.is_some()));

        let raw_p = resolved_design.design.ncols();
        let current_p = raw_p - 1;
        let mut raw_from_current = Array2::<f64>::zeros((raw_p, current_p));
        for col in 0..current_p {
            raw_from_current[[col, col]] = 1.0;
            raw_from_current[[raw_p - 1, col]] = 0.05 * (col + 1) as f64;
        }
        let transform = CoefficientSpatialPsiBlockTransform::new(&raw_from_current)
            .expect("finite coefficient transform");
        let transformed = build_block_spatial_psi_derivatives_with_transform(
            data.view(),
            &resolvedspec,
            &resolved_design,
            &transform,
        )
        .expect("transformed spatial psi derivatives should build")
        .expect("transformed spatial psi derivatives should be present");

        for (raw, current) in derivs.iter().zip(transformed.iter()) {
            let raw_op = raw.implicit_operator.as_ref().expect("raw operator");
            let current_op = current
                .implicit_operator
                .as_ref()
                .expect("coefficient-transformed operator");
            assert_eq!(current_op.p_out(), current_p);
            let axis = raw.implicit_axis;
            let u = Array1::from_iter((0..current_p).map(|idx| 0.25 + idx as f64));
            let raw_u = raw_from_current.dot(&u);
            let expected_forward = raw_op
                .forward_mul(axis, &raw_u.view())
                .expect("raw forward action");
            let actual_forward = current_op
                .forward_mul(axis, &u.view())
                .expect("transformed forward action");
            assert!(
                actual_forward
                    .iter()
                    .zip(expected_forward.iter())
                    .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12),
                "transformed forward action must equal X_psi T u",
            );

            let rows = 3..9;
            let expected_rows = raw_op
                .row_chunk_first(axis, rows.clone())
                .expect("raw first rows")
                .dot(&raw_from_current);
            let actual_rows = current_op
                .row_chunk_first(axis, rows)
                .expect("transformed first rows");
            assert!(
                actual_rows
                    .iter()
                    .zip(expected_rows.iter())
                    .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12),
                "transformed row chunks must equal X_psi[rows] T",
            );

            for ((raw_index, raw_penalty), (current_index, current_penalty)) in raw
                .s_psi_components
                .as_ref()
                .expect("raw first penalty derivatives")
                .iter()
                .zip(
                    current
                        .s_psi_components
                        .as_ref()
                        .expect("transformed first penalty derivatives"),
                )
            {
                assert_eq!(raw_index, current_index);
                let expected = raw_from_current
                    .t()
                    .dot(&raw_penalty.dot(&raw_from_current));
                assert!(
                    current_penalty
                        .iter()
                        .zip(expected.iter())
                        .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12),
                    "transformed penalty derivative must equal T' S_psi T",
                );
            }
        }
    }
}

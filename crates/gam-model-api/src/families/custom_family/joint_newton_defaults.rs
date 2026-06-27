use crate::families::custom_family::family_trait::CustomFamily;
use gam_linalg::faer_ndarray::fast_atb;
use gam_linalg::matrix::{LinearOperator, SignedWeightsView};
use gam_problem::{BlockWorkingSet, CustomFamilyError, ParameterBlockSpec, ParameterBlockState};
use ndarray::{Array1, Array2, s};

pub(crate) fn joint_hessian_has_cross_block_coupling(
    hessian: &Array2<f64>,
    block_states: &[ParameterBlockState],
) -> bool {
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    if hessian.nrows() != total || hessian.ncols() != total {
        // Shape disagreement is handled (loudly) by the symmetrizer/consumers;
        // here we only answer the coupling question and must not claim coupling
        // for a malformed matrix.
        return false;
    }
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(block_states.len());
    let mut start = 0usize;
    for state in block_states {
        let end = start + state.beta.len();
        ranges.push((start, end));
        start = end;
    }
    for (a, (ra_start, ra_end)) in ranges.iter().copied().enumerate() {
        for (rb_start, rb_end) in ranges.iter().copied().skip(a + 1) {
            for i in ra_start..ra_end {
                for j in rb_start..rb_end {
                    if hessian[[i, j]] != 0.0 || hessian[[j, i]] != 0.0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub(crate) fn exact_newton_joint_hessian_from_exact_blocks<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
) -> Result<Option<Array2<f64>>, String> {
    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }
    if evaluation
        .blockworking_sets
        .iter()
        .any(|working_set| !matches!(working_set, BlockWorkingSet::ExactNewton { .. }))
    {
        return Ok(None);
    }

    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, (state, working_set)) in block_states
        .iter()
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = state.beta.len();
        let end = start + p_block;
        let BlockWorkingSet::ExactNewton { hessian, .. } = working_set else {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "exact_newton_joint_hessian default: block {block_idx} working set is not ExactNewton after filter"
                ),
            }
            .into());
        };
        let dense = hessian.to_dense();
        if dense.nrows() != p_block || dense.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian default: block {block_idx} Hessian shape {}x{} != expected {p_block}x{p_block}",
                dense.nrows(),
                dense.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&dense);
        start = end;
    }
    Ok(Some(joint))
}

pub(crate) fn exact_newton_joint_hessian_from_working_sets<F: CustomFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Result<Option<Array2<f64>>, String> {
    if block_states.len() != specs.len() {
        return Err(format!(
            "exact_newton_joint_hessian_with_specs default: block state count {} != spec count {}",
            block_states.len(),
            specs.len()
        ));
    }
    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian_with_specs default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }

    let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, ((state, spec), working_set)) in block_states
        .iter()
        .zip(specs.iter())
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = spec.design.ncols();
        if state.beta.len() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian_with_specs default: block {block_idx} beta length {} != design cols {p_block}",
                state.beta.len()
            ) }.into());
        }
        let end = start + p_block;
        let dense = match working_set {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal {
                working_weights, ..
            } => spec
                .design
                .xt_diag_x_signed_op(SignedWeightsView::from_array(working_weights))?,
        };
        if dense.nrows() != p_block || dense.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian_with_specs default: block {block_idx} Hessian shape {}x{} != expected {p_block}x{p_block}",
                dense.nrows(),
                dense.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&dense);
        start = end;
    }
    Ok(Some(joint))
}

pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_blocks<
    F: CustomFamily + ?Sized,
>(
    family: &F,
    block_states: &[ParameterBlockState],
    d_beta_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    validate_flat_direction_length(
        d_beta_flat,
        total,
        "exact_newton_joint_hessian_directional_derivative default",
    )?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, state) in block_states.iter().enumerate() {
        let p_block = state.beta.len();
        let end = start + p_block;
        let d_beta_block = d_beta_flat.slice(s![start..end]).to_owned();
        let Some(local) = family.exact_newton_hessian_directional_derivative(
            block_states,
            block_idx,
            &d_beta_block,
        )?
        else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian_directional_derivative default: block {block_idx} dH shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
}

/// Block-diagonal aggregator for the joint second directional derivative.
///
/// Mirrors `exact_newton_joint_hessian_directional_derivative_from_blocks`:
/// for a beta-independent joint Hessian the answer is identically zero;
/// otherwise we ask each block for `D²H_b[u_b, v_b]` via
/// `exact_newton_hessian_second_directional_derivative` and place those
/// per-block contributions on the joint diagonal.
///
/// The previous default returned `Some(zeros)` for beta-independent and
/// `None` (no aggregation at all) for beta-dependent families, silently
/// dropping the per-block `d²H` overrides that families like
/// `OneBlockQuarticExactFamily` provide for the outer Hessian's drift
/// contribution.  Aggregating here mirrors the first-derivative path so
/// outer REML receives the curvature term whenever the per-block
/// `exact_newton_hessian_second_directional_derivative` is implemented.
pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_from_blocks<
    F: CustomFamily + ?Sized,
>(
    family: &F,
    block_states: &[ParameterBlockState],
    d_beta_u_flat: &Array1<f64>,
    d_betav_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    validate_flat_direction_length(d_beta_u_flat, total, "joint exact-newton d2H u")?;
    validate_flat_direction_length(d_betav_flat, total, "joint exact-newton d2H v")?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, state) in block_states.iter().enumerate() {
        let p_block = state.beta.len();
        let end = start + p_block;
        let u_block = d_beta_u_flat.slice(s![start..end]).to_owned();
        let v_block = d_betav_flat.slice(s![start..end]).to_owned();
        let Some(local) = family.exact_newton_hessian_second_directional_derivative(
            block_states,
            block_idx,
            &u_block,
            &v_block,
        )?
        else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessiansecond_directional_derivative default: block {block_idx} d2H shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
}

pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_working_sets<
    F: CustomFamily + ?Sized,
>(
    family: &F,
    block_states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    d_beta_flat: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    if block_states.len() != specs.len() {
        return Err(format!(
            "exact_newton_joint_hessian_directional_derivative_with_specs default: block state count {} != spec count {}",
            block_states.len(),
            specs.len()
        ));
    }
    let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    validate_flat_direction_length(
        d_beta_flat,
        total,
        "exact_newton_joint_hessian_directional_derivative_with_specs default",
    )?;
    if !family.exact_newton_joint_hessian_beta_dependent() {
        return Ok(Some(Array2::zeros((total, total))));
    }

    let evaluation = family.evaluate(block_states)?;
    if evaluation.blockworking_sets.len() != block_states.len() {
        return Err(format!(
            "exact_newton_joint_hessian_directional_derivative_with_specs default: working-set count {} != block count {}",
            evaluation.blockworking_sets.len(),
            block_states.len()
        ));
    }

    let mut joint = Array2::<f64>::zeros((total, total));
    let mut start = 0usize;
    for (block_idx, ((state, spec), working_set)) in block_states
        .iter()
        .zip(specs.iter())
        .zip(evaluation.blockworking_sets.iter())
        .enumerate()
    {
        let p_block = spec.design.ncols();
        let end = start + p_block;
        let d_beta_block = d_beta_flat.slice(s![start..end]).to_owned();
        let local = match working_set {
            BlockWorkingSet::ExactNewton { .. } => family
                .exact_newton_hessian_directional_derivative(
                    block_states,
                    block_idx,
                    &d_beta_block,
                )?,
            BlockWorkingSet::Diagonal {
                working_weights, ..
            } => {
                let solver_design = spec.solver_design();
                let mut d_eta = solver_design.apply(&d_beta_block);
                let mut geometry_correction = Array2::<f64>::zeros((p_block, p_block));
                if let Some(geometry) = family.block_geometry_directional_derivative(
                    block_states,
                    block_idx,
                    spec,
                    &d_beta_block,
                )? {
                    if geometry.d_offset.len() != d_eta.len() {
                        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                            "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} geometry offset derivative length {} != eta length {}",
                            geometry.d_offset.len(),
                            d_eta.len()
                        ) }.into());
                    }
                    d_eta += &geometry.d_offset;
                    if let Some(d_design) = geometry.d_design {
                        if d_design.nrows() != solver_design.nrows() || d_design.ncols() != p_block
                        {
                            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                                "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} d_design shape {}x{} != expected {}x{}",
                                d_design.nrows(),
                                d_design.ncols(),
                                solver_design.nrows(),
                                p_block
                            ) }.into());
                        }
                        d_eta += &d_design.dot(&state.beta);

                        let x_dense = solver_design.to_dense();
                        let mut weighted_x = x_dense.clone();
                        let mut weighted_dx = d_design.clone();
                        ndarray::Zip::from(weighted_x.rows_mut())
                            .and(weighted_dx.rows_mut())
                            .and(working_weights.view())
                            .for_each(|mut wx_row, mut wdx_row, &wi| {
                                wx_row.mapv_inplace(|value| value * wi);
                                wdx_row.mapv_inplace(|value| value * wi);
                            });
                        geometry_correction += &fast_atb(&d_design, &weighted_x);
                        geometry_correction += &fast_atb(&x_dense, &weighted_dx);
                    }
                }
                family
                    .diagonalworking_weights_directional_derivative(
                        block_states,
                        block_idx,
                        &d_eta,
                    )?
                    .map(|dw| {
                        let mut local = solver_design
                            .xt_diag_x_signed_op(SignedWeightsView::from_array(&dw))?;
                        local += &geometry_correction;
                        Ok::<Array2<f64>, String>(local)
                    })
                    .transpose()?
            }
        };
        let Some(local) = local else {
            return Ok(None);
        };
        if local.nrows() != p_block || local.ncols() != p_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "exact_newton_joint_hessian_directional_derivative_with_specs default: block {block_idx} dH shape {}x{} != expected {p_block}x{p_block}",
                local.nrows(),
                local.ncols()
            ) }.into());
        }
        joint.slice_mut(s![start..end, start..end]).assign(&local);
        start = end;
    }
    Ok(Some(joint))
}
pub(crate) fn validate_flat_direction_length(
    direction: &Array1<f64>,
    expected: usize,
    context: &str,
) -> Result<(), String> {
    if direction.len() != expected {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{context}: direction length mismatch: got {}, expected {expected}",
                direction.len()
            ),
        }
        .into());
    }
    Ok::<(), _>(())
}

#[cfg(test)]
mod tests {
    use super::joint_hessian_has_cross_block_coupling;
    use gam_problem::ParameterBlockState;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn joint_hessian_coupling_probe_detects_off_diagonal_blocks() {
        // Two blocks of width 2 each → a 4×4 joint Hessian. Only `beta.len()`
        // is read, so the `eta` lengths are immaterial.
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: Array1::zeros(3),
            },
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: Array1::zeros(3),
            },
        ];

        // Strictly block-diagonal (per-block curvature, zero off-blocks): the
        // trait default shape, NOT coupling.
        let block_diagonal = array![
            [1.0_f64, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.3],
            [0.0, 0.0, 0.3, 2.0],
        ];
        assert!(
            !joint_hessian_has_cross_block_coupling(&block_diagonal, &states),
            "block-diagonal joint Hessian must not be treated as coupled"
        );

        // A single nonzero off-diagonal-block entry (and its transpose) is
        // genuine cross-block curvature the block-diagonal default can never
        // produce, so it must be trusted as coupled.
        let mut coupled = block_diagonal.clone();
        coupled[[0, 2]] = 1.0e-9;
        coupled[[2, 0]] = 1.0e-9;
        assert!(
            joint_hessian_has_cross_block_coupling(&coupled, &states),
            "a nonzero off-diagonal block must be detected as coupling"
        );

        // A matrix whose dimension disagrees with the total β width is
        // malformed; the probe must answer the coupling question with `false`
        // rather than claim coupling for a mis-shaped Hessian.
        let wrong_shape = Array2::<f64>::zeros((3, 3));
        assert!(
            !joint_hessian_has_cross_block_coupling(&wrong_shape, &states),
            "shape disagreement must not be claimed as coupling"
        );
    }
}

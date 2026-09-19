use super::*;

pub(crate) fn validate_penalty_specs(
    specs: &[PenaltySpec],
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    for (idx, spec) in specs.iter().enumerate() {
        validate_penalty_spec_shape(idx, spec, p, context)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrefitSeparationDiagnostic {
    pub(crate) column_index: usize,
    pub(crate) threshold: f64,
    pub(crate) positive_above_threshold: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct PrefitLinearSeparationDiagnostic {
    min_signed_margin: f64,
    num_unpenalized_columns: usize,
    column_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PrefitRegularityDiagnostic {
    RankDeficient {
        rank: usize,
        num_unpenalized_columns: usize,
        min_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },
    NearDegenerate {
        num_unpenalized_columns: usize,
        condition_number: f64,
        min_eigenvalue: f64,
        max_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },
}

fn prefit_binary_response_classes(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
) -> Option<Vec<Option<bool>>> {
    let mut class = Vec::with_capacity(y.len());
    let mut active_rows = 0usize;
    let mut has_negative = false;
    let mut has_positive = false;
    for (&yi, &wi) in y.iter().zip(w.iter()) {
        if !wi.is_finite() || wi <= 0.0 {
            class.push(None);
            continue;
        }
        if !yi.is_finite() {
            return None;
        }
        active_rows += 1;
        if yi <= f64::EPSILON {
            has_negative = true;
            class.push(Some(false));
        } else if yi >= 1.0 - f64::EPSILON {
            has_positive = true;
            class.push(Some(true));
        } else {
            return None;
        }
    }
    if active_rows == 0 || !has_negative || !has_positive {
        return None;
    }
    Some(class)
}

fn canonical_unpenalized_column_mask<'a>(
    penalties: impl IntoIterator<Item = &'a CanonicalPenalty>,
    p: usize,
) -> Vec<bool> {
    let mut unpenalized = vec![true; p];
    for penalty in penalties {
        let scale = penalty
            .local
            .diag()
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1.0);
        let tol = 1e-12 * scale;
        for local_col in 0..penalty.col_range.len() {
            let global_col = penalty.col_range.start + local_col;
            if global_col < p && penalty.local[[local_col, local_col]].abs() > tol {
                unpenalized[global_col] = false;
            }
        }
    }
    unpenalized
}

fn unpenalized_column_indices(unpenalized_columns: &[bool]) -> Vec<usize> {
    unpenalized_columns
        .iter()
        .enumerate()
        .filter_map(|(idx, &unpenalized)| unpenalized.then_some(idx))
        .collect()
}

pub(crate) fn detect_prefit_unpenalized_rank_deficiency_in_design(
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    unpenalized_columns: &[bool],
) -> Result<Option<PrefitRegularityDiagnostic>, EstimationError> {
    if x.nrows() != w.len() || x.ncols() != unpenalized_columns.len() {
        return Ok(None);
    }

    let column_indices = unpenalized_column_indices(unpenalized_columns);
    let q = column_indices.len();
    if q <= 1 {
        return Ok(None);
    }

    let mut active_rows = 0usize;
    let mut gram = Array2::<f64>::zeros((q, q));
    let p = x.ncols();
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, x.nrows());
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let rows = end - start;
        x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit rank check failed to stream design rows: {err}"
                ))
            })?;
        for local_row in 0..rows {
            let weight = w[start + local_row];
            if !weight.is_finite() {
                return Ok(None);
            }
            if weight <= 0.0 {
                continue;
            }
            active_rows += 1;
            for (local_col_a, &global_col_a) in column_indices.iter().enumerate() {
                let value_a = chunk[[local_row, global_col_a]];
                if !value_a.is_finite() {
                    return Ok(None);
                }
                for (local_col_b, &global_col_b) in
                    column_indices[..=local_col_a].iter().enumerate()
                {
                    let value_b = chunk[[local_row, global_col_b]];
                    if !value_b.is_finite() {
                        return Ok(None);
                    }
                    gram[[local_col_a, local_col_b]] += weight * value_a * value_b;
                }
            }
        }
    }
    if active_rows == 0 {
        return Ok(None);
    }
    for row in 0..q {
        for col in 0..row {
            gram[[col, row]] = gram[[row, col]];
        }
    }

    let (eigenvalues, _) = gram
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    let spectral_scale = eigenvalues
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    // Rank tolerance is the floating-point noise floor for the Gram entries.
    // Each Gram entry is a sum of `active_rows` products with error ~eps per
    // term; the spectral perturbation bound is `O(active_rows · eps ·
    // λ_max(Gram))`. A looser cutoff (the previous `1e-10 · λ_max`) demotes
    // genuine full-rank-but-ill-conditioned designs as rank-deficient — e.g.
    // two columns differing by a 1e-7 input perturbation yield λ_min ≈ 1e-14,
    // well above the noise floor but inside the old 1e-10 cutoff. Such cases
    // must be classified as NearDegenerate via the condition-number branch
    // below, not as exact rank loss.
    let tolerance = (active_rows.max(q) as f64) * f64::EPSILON * spectral_scale;
    let rank = eigenvalues
        .iter()
        .filter(|&&value| value > tolerance)
        .count();
    let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    if rank < q {
        return Ok(Some(PrefitRegularityDiagnostic::RankDeficient {
            rank,
            num_unpenalized_columns: q,
            min_eigenvalue,
            tolerance,
            column_indices,
        }));
    }

    // Full numeric rank, but the unpenalized normal equations may still be
    // near-singular along a direction (quasi-/near-degenerate). The condition
    // number of the unpenalized Gram is a cheap, principled upfront signal:
    // beyond CONDITION_TOL the unpenalized solve loses too many digits and the
    // fit grinds/diverges instead of converging. CONDITION_TOL is a Gram
    // condition number (≈ design condition squared); 1e12 corresponds to a
    // design condition ≈ 1e6, strictly looser than the noise-floor exact-rank
    // tolerance above so the two checks are nested and consistent.
    const CONDITION_TOL: f64 = 1e12;
    let max_eigenvalue = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if min_eigenvalue.is_finite() && min_eigenvalue > 0.0 && max_eigenvalue.is_finite() {
        let condition_number = max_eigenvalue / min_eigenvalue;
        if condition_number.is_finite() && condition_number > CONDITION_TOL {
            return Ok(Some(PrefitRegularityDiagnostic::NearDegenerate {
                num_unpenalized_columns: q,
                condition_number,
                min_eigenvalue,
                max_eigenvalue,
                tolerance: CONDITION_TOL,
                column_indices,
            }));
        }
    }

    Ok(None)
}

pub(crate) fn reject_prefit_unpenalized_rank_deficiency(
    w: ArrayView1<'_, f64>,
    x_fit: &DesignMatrix,
    penalties: &[CanonicalPenalty],
) -> Result<(), EstimationError> {
    let unpenalized_columns = canonical_unpenalized_column_mask(penalties, x_fit.ncols());
    match detect_prefit_unpenalized_rank_deficiency_in_design(w, x_fit, &unpenalized_columns)? {
        Some(PrefitRegularityDiagnostic::RankDeficient {
            rank,
            num_unpenalized_columns,
            min_eigenvalue,
            tolerance,
            column_indices,
        }) => Err(EstimationError::PrefitRankDeficientDesignDetected {
            rank,
            num_unpenalized_columns,
            min_eigenvalue,
            tolerance,
            column_indices,
        }),
        Some(PrefitRegularityDiagnostic::NearDegenerate {
            num_unpenalized_columns,
            condition_number,
            min_eigenvalue,
            max_eigenvalue,
            tolerance,
            column_indices,
        }) => Err(EstimationError::PrefitNearDegenerateDesignDetected {
            num_unpenalized_columns,
            condition_number,
            min_eigenvalue,
            max_eigenvalue,
            tolerance,
            column_indices,
        }),
        None => Ok(()),
    }
}

fn separator_from_column_extrema(
    unpenalized_columns: &[bool],
    min_pos: &[f64],
    max_pos: &[f64],
    min_neg: &[f64],
    max_neg: &[f64],
) -> Option<PrefitSeparationDiagnostic> {
    for col in 0..unpenalized_columns.len() {
        if !unpenalized_columns[col] {
            continue;
        }
        if min_pos[col] > max_neg[col] {
            return Some(PrefitSeparationDiagnostic {
                column_index: col,
                threshold: 0.5 * (min_pos[col] + max_neg[col]),
                positive_above_threshold: true,
            });
        }
        if min_neg[col] > max_pos[col] {
            return Some(PrefitSeparationDiagnostic {
                column_index: col,
                threshold: 0.5 * (min_neg[col] + max_pos[col]),
                positive_above_threshold: false,
            });
        }
    }

    None
}

pub(crate) fn detect_prefit_binomial_single_column_separation_in_design(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    unpenalized_columns: &[bool],
) -> Result<Option<PrefitSeparationDiagnostic>, EstimationError> {
    if x.nrows() != y.len() || x.nrows() != w.len() || x.ncols() != unpenalized_columns.len() {
        return Ok(None);
    }
    let Some(class) = prefit_binary_response_classes(y, w) else {
        return Ok(None);
    };
    let p = x.ncols();
    if p == 0 {
        return Ok(None);
    }

    let mut min_pos = vec![f64::INFINITY; p];
    let mut max_pos = vec![f64::NEG_INFINITY; p];
    let mut min_neg = vec![f64::INFINITY; p];
    let mut max_neg = vec![f64::NEG_INFINITY; p];
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, x.nrows());
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let rows = end - start;
        x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit binomial separation check failed to stream design rows: {err}"
                ))
            })?;
        for local_row in 0..rows {
            let Some(is_positive) = class[start + local_row] else {
                continue;
            };
            for col in 0..p {
                if !unpenalized_columns[col] {
                    continue;
                }
                let value = chunk[[local_row, col]];
                if !value.is_finite() {
                    return Ok(None);
                }
                if is_positive {
                    min_pos[col] = min_pos[col].min(value);
                    max_pos[col] = max_pos[col].max(value);
                } else {
                    min_neg[col] = min_neg[col].min(value);
                    max_neg[col] = max_neg[col].max(value);
                }
            }
        }
    }

    Ok(separator_from_column_extrema(
        unpenalized_columns,
        &min_pos,
        &max_pos,
        &min_neg,
        &max_neg,
    ))
}

fn certify_prefit_binomial_linear_separator(
    class: &[Option<bool>],
    x: &DesignMatrix,
    column_indices: &[usize],
    direction: &[f64],
) -> Result<Option<PrefitLinearSeparationDiagnostic>, EstimationError> {
    if x.nrows() != class.len() || column_indices.len() != direction.len() {
        return Ok(None);
    }
    let direction_norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !direction_norm.is_finite() || direction_norm <= 0.0 {
        return Ok(None);
    }

    let p = x.ncols();
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, x.nrows());
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    let mut min_signed_margin = f64::INFINITY;
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let rows = end - start;
        x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit binomial linear-separation certification failed to stream design rows: {err}"
                ))
            })?;
        for local_row in 0..rows {
            let Some(is_positive) = class[start + local_row] else {
                continue;
            };
            let mut dot = 0.0;
            let mut magnitude = 0.0;
            for (local_col, &global_col) in column_indices.iter().enumerate() {
                let value = chunk[[local_row, global_col]];
                if !value.is_finite() {
                    return Ok(None);
                }
                let term = direction[local_col] * value;
                dot += term;
                magnitude += term.abs();
            }
            if !magnitude.is_finite() {
                return Ok(None);
            }
            let signed_margin = if is_positive { dot } else { -dot };
            // The margin is a rounded sum of `q` products: it certifies a side only
            // beyond that sum's rounding band `gamma_q * sum |d_j x_ij|` (#2469).
            let tolerance =
                gam_linalg::roundoff::accumulation_growth(column_indices.len()) * magnitude;
            if signed_margin <= tolerance {
                return Ok(None);
            }
            min_signed_margin = min_signed_margin.min(signed_margin / direction_norm);
        }
    }
    if !min_signed_margin.is_finite() {
        return Ok(None);
    }

    Ok(Some(PrefitLinearSeparationDiagnostic {
        min_signed_margin,
        num_unpenalized_columns: column_indices.len(),
        column_indices: column_indices.to_vec(),
    }))
}

fn detect_prefit_binomial_linear_combination_separation_in_design(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    unpenalized_columns: &[bool],
) -> Result<Option<PrefitLinearSeparationDiagnostic>, EstimationError> {
    if x.nrows() != y.len() || x.nrows() != w.len() || x.ncols() != unpenalized_columns.len() {
        return Ok(None);
    }
    let Some(class) = prefit_binary_response_classes(y, w) else {
        return Ok(None);
    };
    let column_indices = unpenalized_column_indices(unpenalized_columns);
    let q = column_indices.len();
    if q == 0 {
        return Ok(None);
    }

    let p = x.ncols();
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, x.nrows());
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    let mut direction = vec![0.0_f64; q];
    let max_passes = (8 * q.max(1)).clamp(16, 128);
    for _ in 0..max_passes {
        let mut mistakes = 0usize;
        for start in (0..x.nrows()).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(x.nrows());
            let rows = end - start;
            x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
                .map_err(|err| {
                    EstimationError::LayoutError(format!(
                        "pre-fit binomial linear-separation check failed to stream design rows: {err}"
                    ))
                })?;
            for local_row in 0..rows {
                let Some(is_positive) = class[start + local_row] else {
                    continue;
                };
                let sign = if is_positive { 1.0 } else { -1.0 };
                let mut dot = 0.0;
                let mut magnitude = 0.0;
                let mut row_norm_sq = 0.0;
                for (local_col, &global_col) in column_indices.iter().enumerate() {
                    let value = chunk[[local_row, global_col]];
                    if !value.is_finite() {
                        return Ok(None);
                    }
                    let term = direction[local_col] * value;
                    dot += term;
                    magnitude += term.abs();
                    row_norm_sq += value * value;
                }
                if !row_norm_sq.is_finite() {
                    return Ok(None);
                }
                let signed_margin = sign * dot;
                // Same rounding band as the certificate: a margin inside it is a mistake.
                let margin_tolerance = gam_linalg::roundoff::accumulation_growth(q) * magnitude;
                if signed_margin > margin_tolerance {
                    continue;
                }
                mistakes += 1;
                if row_norm_sq <= 0.0 {
                    continue;
                }
                let update_scale = sign / row_norm_sq;
                for (local_col, &global_col) in column_indices.iter().enumerate() {
                    direction[local_col] += update_scale * chunk[[local_row, global_col]];
                }
            }
        }
        if mistakes == 0 {
            return certify_prefit_binomial_linear_separator(
                &class,
                x,
                &column_indices,
                &direction,
            );
        }
    }

    certify_prefit_binomial_linear_separator(&class, x, &column_indices, &direction)
}

fn prefit_binomial_separation_supported_link(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(StandardLink::Logit | StandardLink::Probit | StandardLink::CLogLog)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    )
}

pub(crate) fn reject_prefit_binomial_separation(
    cfg: &RemlConfig,
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x_fit: &DesignMatrix,
    penalties: &[CanonicalPenalty],
) -> Result<(), EstimationError> {
    if !matches!(cfg.likelihood.spec.response, ResponseFamily::Binomial)
        || !prefit_binomial_separation_supported_link(&cfg.link_kind)
        || cfg.firth_bias_reduction
    {
        return Ok(());
    }
    // The certificate reads every parametric scalar column whatever its ridge. A
    // one-column penalty block is a parametric effect's null-recovery ridge
    // (b7b874a2a): along a separating direction REML sends its λ toward zero, so it
    // bounds nothing there. Multi-column blocks are basis expansions, and enough
    // basis columns separate any response, so their columns stay out (#2898);
    // only the directions inside them that no roughness penalty bounds are read
    // (`penalty_free_block_frames`).
    let certified_columns = canonical_unpenalized_column_mask(
        penalties
            .iter()
            .filter(|penalty| penalty.col_range.len() > 1),
        x_fit.ncols(),
    );
    let frames = penalty_free_block_frames(penalties, x_fit.ncols());
    if frames.is_empty() {
        return reject_separation_along_columns(y, w, x_fit, &certified_columns, |columns| {
            columns.to_vec()
        });
    }

    let direct_columns = unpenalized_column_indices(&certified_columns);
    let projected = project_onto_penalty_free_directions(x_fit, &direct_columns, &frames)?;
    let mut direct_mask = vec![false; projected.ncols()];
    direct_mask[..direct_columns.len()].fill(true);
    if let Some(diagnostic) =
        detect_prefit_binomial_single_column_separation_in_design(y, w, &projected, &direct_mask)?
    {
        return Err(EstimationError::PrefitPerfectSeparationDetected {
            column_index: direct_columns[diagnostic.column_index],
            threshold: diagnostic.threshold,
            positive_above_threshold: diagnostic.positive_above_threshold,
        });
    }
    reject_linear_separation_along_columns(
        y,
        w,
        &projected,
        &vec![true; projected.ncols()],
        |projected_columns| {
            let mut global = Vec::new();
            for &column in projected_columns {
                match direct_columns.get(column) {
                    Some(&direct) => global.push(direct),
                    None => {
                        let frame = frame_of_projected_column(&frames, column - direct_columns.len());
                        global.extend_from_slice(&frames[frame].columns);
                    }
                }
            }
            global.sort_unstable();
            global.dedup();
            global
        },
    )
}

/// Raise the single-column certificate over `columns`, then the linear one;
/// `support` maps the linear certificate's columns to the realized design's.
fn reject_separation_along_columns(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    columns: &[bool],
    support: impl Fn(&[usize]) -> Vec<usize>,
) -> Result<(), EstimationError> {
    if let Some(diagnostic) =
        detect_prefit_binomial_single_column_separation_in_design(y, w, x, columns)?
    {
        return Err(EstimationError::PrefitPerfectSeparationDetected {
            column_index: diagnostic.column_index,
            threshold: diagnostic.threshold,
            positive_above_threshold: diagnostic.positive_above_threshold,
        });
    }
    reject_linear_separation_along_columns(y, w, x, columns, support)
}

fn reject_linear_separation_along_columns(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    columns: &[bool],
    support: impl Fn(&[usize]) -> Vec<usize>,
) -> Result<(), EstimationError> {
    if let Some(diagnostic) =
        detect_prefit_binomial_linear_combination_separation_in_design(y, w, x, columns)?
    {
        return Err(EstimationError::PrefitLinearSeparationDetected {
            min_signed_margin: diagnostic.min_signed_margin,
            num_unpenalized_columns: diagnostic.num_unpenalized_columns,
            column_indices: support(&diagnostic.column_indices),
        });
    }
    Ok(())
}

/// Directions inside one multi-column penalty block that no roughness penalty
/// bounds, as an orthonormal frame over the block's penalized columns.
#[derive(Clone, Debug)]
struct PenaltyFreeFrame {
    /// The realized design columns the frame's rows stand for.
    columns: Vec<usize>,
    /// `columns.len() × m`, orthonormal columns.
    frame: Array2<f64>,
}

/// The directions of each multi-column penalty block that are either free of
/// every penalty on the block or bounded only by a null-space ridge.
///
/// A null-space ridge is the one-column ridge's argument on a basis: it is the
/// penalty on a block whose range is exactly the null space its companions
/// leave, so along those directions it is the only bound, and along a
/// separating one REML sends its λ toward zero. A double penalty ships the
/// ridge beside its roughness penalty, and that pair is symmetric in every
/// algebraic invariant (each one's range is the other's null space); the
/// ridge is the smaller member, the low-order kernel of the roughness penalty
/// it completes, and a pair of equal ranks names neither. A tensor product's
/// margin penalty never qualifies: its companions leave free only part of
/// its range. Blocks whose column ranges overlap another block's are read as
/// before, not at all.
fn penalty_free_block_frames(penalties: &[CanonicalPenalty], p: usize) -> Vec<PenaltyFreeFrame> {
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for penalty in penalties {
        let range = &penalty.col_range;
        if range.len() > 1 && range.end <= p && !ranges.contains(range) {
            ranges.push(range.clone());
        }
    }
    let overlaps = |left: &std::ops::Range<usize>, right: &std::ops::Range<usize>| {
        left.start < right.end && right.start < left.end
    };

    let mut frames = Vec::new();
    for range in &ranges {
        if ranges
            .iter()
            .any(|other| other != range && overlaps(other, range))
        {
            continue;
        }
        let members: Vec<&CanonicalPenalty> = penalties
            .iter()
            .filter(|penalty| {
                penalty.col_range == *range && penalty.local.dim() == (range.len(), range.len())
            })
            .collect();
        // A column no penalty on the block touches is already read on its own.
        let unpenalized = canonical_unpenalized_column_mask(members.iter().copied(), p);
        let local: Vec<usize> = (0..range.len())
            .filter(|&column| !unpenalized[range.start + column])
            .collect();
        let dim = local.len();
        if dim == 0 {
            continue;
        }
        // Each penalty at unit spectral scale, so none hides under another's rounding band.
        let normalized: Vec<(Array2<f64>, usize)> = members
            .iter()
            .filter_map(|penalty| {
                let scale = penalty
                    .positive_eigenvalues
                    .iter()
                    .fold(0.0_f64, |acc, &value| acc.max(value));
                (scale.is_finite() && scale > 0.0).then(|| {
                    let block = penalty
                        .local
                        .select(Axis(0), &local)
                        .select(Axis(1), &local);
                    (block / scale, penalty.positive_eigenvalues.len())
                })
            })
            .collect();
        let sum_except = |skip: Option<usize>| {
            let mut sum = Array2::<f64>::zeros((dim, dim));
            for (index, (matrix, _)) in normalized.iter().enumerate() {
                if Some(index) != skip {
                    sum += matrix;
                }
            }
            sum
        };
        let Some(free) = super::rho_domain::psd_null_frame(&sum_except(None)) else {
            continue;
        };
        let mut ridge_frames = Vec::new();
        if normalized.len() > 1 {
            for (ridge, (_, rank)) in normalized.iter().enumerate() {
                let Some(companion_free) = super::rho_domain::psd_null_frame(&sum_except(Some(ridge)))
                else {
                    continue;
                };
                let ridge_only = companion_free.ncols() - free.ncols();
                let companion_rank = dim - companion_free.ncols();
                if ridge_only > 0 && ridge_only == *rank && *rank < companion_rank {
                    ridge_frames.push(companion_free);
                }
            }
        }
        if ridge_frames.is_empty() && free.ncols() > 0 {
            ridge_frames.push(free);
        }
        let columns: Vec<usize> = local.iter().map(|&column| range.start + column).collect();
        frames.extend(ridge_frames.into_iter().map(|frame| PenaltyFreeFrame {
            columns: columns.clone(),
            frame,
        }));
    }
    frames
}

/// The frame a projected column past the direct columns belongs to.
fn frame_of_projected_column(frames: &[PenaltyFreeFrame], mut offset: usize) -> usize {
    for (index, frame) in frames.iter().enumerate() {
        if offset < frame.frame.ncols() {
            return index;
        }
        offset -= frame.frame.ncols();
    }
    unreachable!("projected column past every frame")
}

/// `[X_direct, X_B1·F1, X_B2·F2, …]`: the design read along the directions the
/// certificate may use, the direct columns first.
fn project_onto_penalty_free_directions(
    x: &DesignMatrix,
    direct_columns: &[usize],
    frames: &[PenaltyFreeFrame],
) -> Result<DesignMatrix, EstimationError> {
    let width =
        direct_columns.len() + frames.iter().map(|frame| frame.frame.ncols()).sum::<usize>();
    let p = x.ncols();
    let mut projected = Array2::<f64>::zeros((x.nrows(), width));
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, x.nrows());
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let rows = end - start;
        x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit binomial separation check failed to stream design rows: {err}"
                ))
            })?;
        for local_row in 0..rows {
            let row = chunk.row(local_row);
            let mut target = projected.row_mut(start + local_row);
            for (column, &global) in direct_columns.iter().enumerate() {
                target[column] = row[global];
            }
            let mut column = direct_columns.len();
            for frame in frames {
                for direction in frame.frame.columns() {
                    target[column] = frame
                        .columns
                        .iter()
                        .zip(direction.iter())
                        .map(|(&global, &weight)| row[global] * weight)
                        .sum();
                    column += 1;
                }
            }
        }
    }
    Ok(DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(projected),
    ))
}

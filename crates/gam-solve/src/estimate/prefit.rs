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

/// One coordinate of the linear-separation search, `z_i = Σ_k weights[k]·x_i[columns[k]]`:
/// a parametric column (weight one), or a direction of a penalty block's null space.
#[derive(Clone, Debug, PartialEq)]
struct PrefitSeparationCoordinate {
    columns: Vec<usize>,
    weights: Vec<f64>,
    null_space: bool,
}

impl PrefitSeparationCoordinate {
    fn column(col: usize) -> Self {
        Self {
            columns: vec![col],
            weights: vec![1.0],
            null_space: false,
        }
    }

    fn value(&self, row: ArrayView1<'_, f64>) -> f64 {
        self.columns
            .iter()
            .zip(&self.weights)
            .map(|(&col, &weight)| weight * row[col])
            .sum()
    }
}

/// The directions of each multi-column penalty block on which the prior is flat,
/// or bounded only by the block's null-space ridge.
///
/// A smooth's roughness penalty leaves its low-order polynomial part unpenalized,
/// and a double-penalty smooth adds a second penalty on exactly that null space
/// (Marra & Wood 2011). Along a separating direction in the null space REML sends
/// the ridge's λ toward zero, as it does for a parametric column's one-column
/// ridge (b7b874a2a), so the ridge bounds nothing there; without the ridge the
/// prior is flat there to begin with. Either way the posterior is improper along
/// a separator in that space. The null space is the smooth's polynomial part, not
/// a basis expansion, so it cannot separate an arbitrary response (#2898).
///
/// A penalty is read as its block's null-space ridge when its rank and the rank
/// of the rest of the block sum to the rank of the whole block (the ranges are
/// complementary) and its rank is strictly the smaller (a tie names no ridge, so
/// the block reads as unridged); the directions returned are then
/// the null space of the rest of the block. A block with no such ridge returns
/// the null space of the whole block. Ranks are read off each penalty's root,
/// scaled to unit Frobenius norm so that no penalty's scale buries another's, by
/// [`gam_linalg::roundoff::factor_rank_partition`].
fn penalty_null_space_directions(
    penalties: &[CanonicalPenalty],
    p: usize,
) -> Result<Vec<PrefitSeparationCoordinate>, EstimationError> {
    let mut coordinates = Vec::new();
    let mut visited: Vec<std::ops::Range<usize>> = Vec::new();
    for penalty in penalties {
        let range = penalty.col_range.clone();
        if range.len() < 2 || range.end > p || visited.contains(&range) {
            continue;
        }
        visited.push(range.clone());
        let overlaps_another_block = penalties.iter().any(|other| {
            other.col_range != range
                && other.col_range.start < range.end
                && range.start < other.col_range.end
        });
        if overlaps_another_block {
            continue;
        }
        let block: Vec<&CanonicalPenalty> = penalties
            .iter()
            .filter(|other| other.col_range == range)
            .collect();
        let dim = range.len();
        let mut roots = Vec::with_capacity(block.len());
        for member in &block {
            let frobenius = member.root.iter().map(|v| v * v).sum::<f64>().sqrt();
            if member.root.ncols() != dim || !frobenius.is_finite() || frobenius <= 0.0 {
                roots.clear();
                break;
            }
            roots.push(member.root.mapv(|v| v / frobenius));
        }
        if roots.len() != block.len() {
            continue;
        }
        let stacked_rank = |members: &mut dyn Iterator<Item = &Array2<f64>>| {
            let members: Vec<_> = members.map(|root| root.view()).collect();
            let stack = ndarray::concatenate(Axis(0), &members).map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit null-space separation check failed to stack penalty roots: {err}"
                ))
            })?;
            gam_linalg::roundoff::factor_rank_partition(&stack)
                .map_err(EstimationError::EigendecompositionFailed)
        };
        let whole = stacked_rank(&mut roots.iter())?;
        // A lone penalty has no rest of the block to be complementary to.
        let ridge_candidates = if roots.len() > 1 { 0..roots.len() } else { 0..0 };
        let mut ridge_released = false;
        for ridge in ridge_candidates {
            let ridge_rank = gam_linalg::roundoff::factor_rank_partition(&roots[ridge])
                .map_err(EstimationError::EigendecompositionFailed)?
                .rank;
            let rest = stacked_rank(
                &mut roots
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != ridge)
                    .map(|(_, root)| root),
            )?;
            if ridge_rank == 0 || ridge_rank >= rest.rank || ridge_rank + rest.rank != whole.rank
            {
                continue;
            }
            ridge_released = true;
            push_null_directions(&mut coordinates, &range, &rest);
        }
        if !ridge_released {
            push_null_directions(&mut coordinates, &range, &whole);
        }
    }
    Ok(coordinates)
}

/// The right singular vectors past `partition.rank` span the null space of the
/// partitioned factor's quadratic.
fn push_null_directions(
    coordinates: &mut Vec<PrefitSeparationCoordinate>,
    range: &std::ops::Range<usize>,
    partition: &gam_linalg::roundoff::FactorRankPartition,
) {
    for direction in partition.right_vectors.rows().into_iter().skip(partition.rank) {
        coordinates.push(PrefitSeparationCoordinate {
            columns: range.clone().collect(),
            weights: direction.to_vec(),
            null_space: true,
        });
    }
}

fn detect_prefit_binomial_linear_combination_separation_in_design(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    coordinates: &[PrefitSeparationCoordinate],
) -> Result<Option<PrefitLinearSeparationDiagnostic>, EstimationError> {
    if x.nrows() != y.len() || x.nrows() != w.len() {
        return Ok(None);
    }
    let p = x.ncols();
    if coordinates
        .iter()
        .any(|coordinate| coordinate.columns.iter().any(|&col| col >= p))
    {
        return Ok(None);
    }
    let Some(class) = prefit_binary_response_classes(y, w) else {
        return Ok(None);
    };
    let q = coordinates.len();
    if q == 0 {
        return Ok(None);
    }

    // Certify in the design's own columns: the search coordinates only propose
    // a direction, and the certificate's rounding band is read off the columns
    // the fit actually multiplies.
    let mut column_indices: Vec<usize> = coordinates
        .iter()
        .flat_map(|coordinate| coordinate.columns.iter().copied())
        .collect();
    column_indices.sort_unstable();
    column_indices.dedup();
    let certify = |direction: &[f64]| {
        let mut beta = vec![0.0_f64; column_indices.len()];
        for (coordinate, &d) in coordinates.iter().zip(direction) {
            for (&col, &weight) in coordinate.columns.iter().zip(&coordinate.weights) {
                let slot = column_indices
                    .binary_search(&col)
                    .expect("every coordinate column is in the certified support");
                beta[slot] += d * weight;
            }
        }
        certify_prefit_binomial_linear_separator(&class, x, &column_indices, &beta)
    };

    let Some(statistics) = prefit_coordinate_statistics(&class, x, coordinates)? else {
        return Ok(None);
    };
    for direction in prefit_threshold_separator_proposals(&statistics)? {
        if let Some(diagnostic) = certify(&direction)? {
            return Ok(Some(diagnostic));
        }
    }

    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, x.nrows());
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    let mut z = vec![0.0_f64; q];
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
                let row = chunk.row(local_row);
                let mut dot = 0.0;
                let mut magnitude = 0.0;
                let mut row_norm_sq = 0.0;
                for (local_col, coordinate) in coordinates.iter().enumerate() {
                    let value = coordinate.value(row);
                    if !value.is_finite() {
                        return Ok(None);
                    }
                    z[local_col] = value;
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
                for (local_col, &value) in z.iter().enumerate() {
                    direction[local_col] += update_scale * value;
                }
            }
        }
        if mistakes == 0 {
            break;
        }
    }

    if let Some(diagnostic) = certify(&direction)? {
        return Ok(Some(diagnostic));
    }
    Ok(prefit_null_space_quasi_separator(&statistics, coordinates))
}

/// Per-class extrema, Gram matrix and column sums of the search coordinates
/// over the rows that carry a class, from one streaming pass over the design.
struct PrefitCoordinateStatistics {
    min_pos: Vec<f64>,
    max_pos: Vec<f64>,
    min_neg: Vec<f64>,
    max_neg: Vec<f64>,
    gram: Array2<f64>,
    column_sums: Array1<f64>,
    active_rows: usize,
    /// The largest `Σ_k |weights[k]·x_i[columns[k]]|` over classed rows: the
    /// scale of the coordinate's rounding band.
    max_magnitude: Vec<f64>,
}

impl PrefitCoordinateStatistics {
    /// The coordinate's value when it takes one value on every classed row.
    fn constant_value(&self, k: usize) -> Option<f64> {
        let value = self.min_pos[k];
        (value.is_finite()
            && value == self.max_pos[k]
            && value == self.min_neg[k]
            && value == self.max_neg[k])
            .then_some(value)
    }
}

/// `None` when a coordinate is not finite on some classed row.
fn prefit_coordinate_statistics(
    class: &[Option<bool>],
    x: &DesignMatrix,
    coordinates: &[PrefitSeparationCoordinate],
) -> Result<Option<PrefitCoordinateStatistics>, EstimationError> {
    let q = coordinates.len();
    let p = x.ncols();
    let mut statistics = PrefitCoordinateStatistics {
        min_pos: vec![f64::INFINITY; q],
        max_pos: vec![f64::NEG_INFINITY; q],
        min_neg: vec![f64::INFINITY; q],
        max_neg: vec![f64::NEG_INFINITY; q],
        gram: Array2::<f64>::zeros((q, q)),
        column_sums: Array1::<f64>::zeros(q),
        active_rows: 0,
        max_magnitude: vec![0.0; q],
    };
    let mut z = vec![0.0_f64; q];
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, x.nrows());
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let rows = end - start;
        x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit binomial threshold-separation check failed to stream design rows: {err}"
                ))
            })?;
        for local_row in 0..rows {
            let Some(is_positive) = class[start + local_row] else {
                continue;
            };
            statistics.active_rows += 1;
            let row = chunk.row(local_row);
            for (k, coordinate) in coordinates.iter().enumerate() {
                let value = coordinate.value(row);
                if !value.is_finite() {
                    return Ok(None);
                }
                z[k] = value;
                let magnitude: f64 = coordinate
                    .columns
                    .iter()
                    .zip(&coordinate.weights)
                    .map(|(&col, &weight)| (weight * row[col]).abs())
                    .sum();
                statistics.max_magnitude[k] = statistics.max_magnitude[k].max(magnitude);
                if is_positive {
                    statistics.min_pos[k] = statistics.min_pos[k].min(value);
                    statistics.max_pos[k] = statistics.max_pos[k].max(value);
                } else {
                    statistics.min_neg[k] = statistics.min_neg[k].min(value);
                    statistics.max_neg[k] = statistics.max_neg[k].max(value);
                }
            }
            for a in 0..q {
                statistics.column_sums[a] += z[a];
                for b in 0..=a {
                    statistics.gram[[a, b]] += z[a] * z[b];
                }
            }
        }
    }
    for a in 0..q {
        for b in 0..a {
            statistics.gram[[b, a]] = statistics.gram[[a, b]];
        }
    }
    Ok(Some(statistics))
}

/// A quasi-complete separator along one penalty null-space direction: the
/// classes' values of `z_k` meet at a threshold `t` without crossing it,
/// `max_neg ≤ t ≤ min_pos` (or mirrored), and some row lies strictly past it.
/// Then `±(z_k − t)` is ≥ 0 on every row and > 0 on some, so the likelihood
/// rises without bound along that direction toward a supremum it never
/// attains, and the flat prior leaves the posterior improper there (Albert &
/// Anderson 1984). The offset `t` must be expressible, so either `t = 0` or a
/// parametric coordinate takes one nonzero value on every classed row.
///
/// Rows tied at the threshold have margin exactly zero, which no rounding band
/// can certify, so the comparisons read the coordinate values as computed:
/// identical design rows give identical values, and a row whose value the
/// arithmetic cannot tell from `t` is tied to it. A strict separator is left to
/// the exact certificate above; only null-space directions are read, because
/// their null-space ridge keeps each inner problem bounded while REML drives
/// its λ to its rail, so a flat-prior fit returns a railed optimum instead of
/// refusing, and nothing downstream would engage the Jeffreys prior.
fn prefit_null_space_quasi_separator(
    statistics: &PrefitCoordinateStatistics,
    coordinates: &[PrefitSeparationCoordinate],
) -> Option<PrefitLinearSeparationDiagnostic> {
    let constant = coordinates
        .iter()
        .enumerate()
        .filter(|(_, coordinate)| !coordinate.null_space)
        .find_map(|(j, _)| {
            statistics
                .constant_value(j)
                .filter(|&value| value != 0.0)
                .map(|_| j)
        });
    for (k, coordinate) in coordinates.iter().enumerate() {
        if !coordinate.null_space {
            continue;
        }
        let (min_pos, max_pos) = (statistics.min_pos[k], statistics.max_pos[k]);
        let (min_neg, max_neg) = (statistics.min_neg[k], statistics.max_neg[k]);
        // The row past the threshold must clear the coordinate's rounding band,
        // or a direction the arithmetic cannot tell from constant would qualify.
        let band = gam_linalg::roundoff::accumulation_growth(coordinate.columns.len())
            * statistics.max_magnitude[k];
        let threshold = if max_neg <= min_pos && max_pos - min_neg > band {
            0.5 * (max_neg + min_pos)
        } else if max_pos <= min_neg && max_neg - min_pos > band {
            0.5 * (max_pos + min_neg)
        } else {
            continue;
        };
        let offset = if threshold == 0.0 {
            None
        } else if let Some(j) = constant {
            Some(j)
        } else {
            continue;
        };
        let mut column_indices: Vec<usize> = coordinate
            .columns
            .iter()
            .chain(offset.iter().flat_map(|&j| coordinates[j].columns.iter()))
            .copied()
            .collect();
        column_indices.sort_unstable();
        column_indices.dedup();
        return Some(PrefitLinearSeparationDiagnostic {
            min_signed_margin: 0.0,
            num_unpenalized_columns: column_indices.len(),
            column_indices,
        });
    }
    None
}

/// Exact separator proposals, one per search coordinate whose values the two
/// classes do not interleave: `±(e_k − t·a)`, with `t` the midpoint of the gap
/// and `a` the least-squares representation of the constant in the coordinates.
///
/// The perceptron needs on the order of `(R/γ)²` updates, and a step response
/// on a grid of `n` points has a margin `γ` near `R/n`, so it cannot find the
/// separator of the very step it exists for. A threshold on one coordinate is
/// that separator whenever the constant lies in the coordinates' span; when it
/// does not, the proposal fails the certificate and nothing is claimed.
fn prefit_threshold_separator_proposals(
    statistics: &PrefitCoordinateStatistics,
) -> Result<Vec<Vec<f64>>, EstimationError> {
    let PrefitCoordinateStatistics {
        min_pos,
        max_pos,
        min_neg,
        max_neg,
        gram,
        column_sums,
        active_rows,
        ..
    } = statistics;
    let q = min_pos.len();

    // `a = G⁺ Zᵀ1`, the pseudo-inverse cut at the Gram's rounding floor, the
    // same floor the pre-fit rank check reads.
    let (eigenvalues, eigenvectors) = gram
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Ok(Vec::new());
    }
    let spectral_scale = eigenvalues
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    let floor = ((*active_rows).max(q) as f64) * f64::EPSILON * spectral_scale;
    let mut constant = Array1::<f64>::zeros(q);
    for (i, &value) in eigenvalues.iter().enumerate() {
        if value > floor {
            let v = eigenvectors.column(i);
            constant.scaled_add(v.dot(column_sums) / value, &v);
        }
    }

    let mut proposals = Vec::new();
    for k in 0..q {
        let (threshold, sign) = if min_pos[k] > max_neg[k] {
            (0.5 * (min_pos[k] + max_neg[k]), 1.0)
        } else if min_neg[k] > max_pos[k] {
            (0.5 * (min_neg[k] + max_pos[k]), -1.0)
        } else {
            continue;
        };
        let mut direction: Vec<f64> = constant.iter().map(|&c| -sign * threshold * c).collect();
        direction[k] += sign;
        proposals.push(direction);
    }
    Ok(proposals)
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
    // basis columns separate any response, so their columns stay out (#2898).
    let certified_columns = canonical_unpenalized_column_mask(
        penalties
            .iter()
            .filter(|penalty| penalty.col_range.len() > 1),
        x_fit.ncols(),
    );
    if let Some(diagnostic) = detect_prefit_binomial_single_column_separation_in_design(
        y,
        w,
        x_fit,
        &certified_columns,
    )? {
        return Err(EstimationError::PrefitPerfectSeparationDetected {
            column_index: diagnostic.column_index,
            threshold: diagnostic.threshold,
            positive_above_threshold: diagnostic.positive_above_threshold,
        });
    }
    // A smooth's null space (its intercept-free polynomial part) is penalized only
    // by its double-penalty ridge, which REML releases along a separator exactly as
    // it releases a one-column ridge, so its directions join the search.
    let coordinates: Vec<PrefitSeparationCoordinate> =
        unpenalized_column_indices(&certified_columns)
            .into_iter()
            .map(PrefitSeparationCoordinate::column)
            .chain(penalty_null_space_directions(penalties, x_fit.ncols())?)
            .collect();
    if let Some(diagnostic) =
        detect_prefit_binomial_linear_combination_separation_in_design(y, w, x_fit, &coordinates)?
    {
        return Err(EstimationError::PrefitLinearSeparationDetected {
            min_signed_margin: diagnostic.min_signed_margin,
            num_unpenalized_columns: diagnostic.num_unpenalized_columns,
            column_indices: diagnostic.column_indices,
        });
    }

    Ok(())
}

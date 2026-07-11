//! Cross-replicate circle-coordinate stability modulo the exact `O(2)` gauge.
//!
//! Reconstruction quality cannot certify a latent angle (#2260). This module
//! compares replicate coordinate vectors after quotienting only the circle's
//! rotation/reflection gauge. It also certifies that every replicate spans both
//! circle-embedding axes, so two collapsed coordinate vectors cannot receive a
//! vacuous perfect score merely because a rotation can align their constants.

use ndarray::ArrayView2;

#[derive(Clone, Debug, PartialEq)]
pub struct CircularReplicateCoverage {
    pub replicate: usize,
    pub minimum_embedding_eigenvalue: f64,
    pub maximum_embedding_eigenvalue: f64,
    pub isotropic_coverage: f64,
    pub rank_resolution: f64,
    pub well_posed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CircularPairConcordance {
    pub left: usize,
    pub right: usize,
    pub rotation_score: Option<f64>,
    pub reflection_score: Option<f64>,
    pub aligned_score: Option<f64>,
    pub reflected: Option<bool>,
    pub phase_shift: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CircularConcordanceReport {
    pub n_replicates: usize,
    pub n_rows: usize,
    pub period: f64,
    pub coverage: Vec<CircularReplicateCoverage>,
    pub pairs: Vec<CircularPairConcordance>,
    pub minimum_aligned_score: Option<f64>,
    pub mean_aligned_score: Option<f64>,
}

fn embedding_coverage(cos: &[f64], sin: &[f64], replicate: usize) -> CircularReplicateCoverage {
    let n = cos.len();
    let mut g00 = 0.0;
    let mut g01 = 0.0;
    let mut g11 = 0.0;
    for row in 0..n {
        g00 += cos[row] * cos[row];
        g01 += cos[row] * sin[row];
        g11 += sin[row] * sin[row];
    }
    let trace = g00 + g11;
    let discriminant = ((g00 - g11).powi(2) + 4.0 * g01 * g01).sqrt();
    let minimum = (0.5 * (trace - discriminant)).max(0.0);
    let maximum = 0.5 * (trace + discriminant);
    // Backward-error resolution for a sum of `n` two-dimensional outer
    // products. This is derived from machine epsilon and problem size, not a
    // user-facing agreement threshold.
    let rank_resolution = f64::EPSILON * 2.0 * n as f64;
    CircularReplicateCoverage {
        replicate,
        minimum_embedding_eigenvalue: minimum,
        maximum_embedding_eigenvalue: maximum,
        isotropic_coverage: (2.0 * minimum / trace).clamp(0.0, 1.0),
        rank_resolution,
        well_posed: minimum > rank_resolution,
    }
}

/// Compare replicate circle coordinates modulo rotation/reflection.
///
/// `coordinates` is `(replicates, rows)` with corresponding rows referring to
/// the same observations in every replicate. `period` is the coordinate period.
/// For pair `(a, b)`, the rotation score is
/// `|mean exp(i(theta_a-theta_b))|`; the reflection score replaces the minus by
/// plus. Their maximum is the exact `O(2)`-aligned circular concordance. Scores
/// are withheld when either embedding is numerically rank deficient.
pub fn circular_concordance(
    coordinates: ArrayView2<'_, f64>,
    period: f64,
) -> Result<CircularConcordanceReport, String> {
    let (n_replicates, n_rows) = coordinates.dim();
    if n_replicates < 2 || n_rows < 2 {
        return Err(format!(
            "circular_concordance requires at least two replicates and two aligned rows; got {n_replicates}x{n_rows}"
        ));
    }
    if !(period.is_finite() && period > 0.0) {
        return Err(format!(
            "circular_concordance period must be finite and positive, got {period}"
        ));
    }
    if coordinates.iter().any(|value| !value.is_finite()) {
        return Err("circular_concordance coordinates must be finite".to_string());
    }

    let mut cos = vec![vec![0.0; n_rows]; n_replicates];
    let mut sin = vec![vec![0.0; n_rows]; n_replicates];
    for replicate in 0..n_replicates {
        for row in 0..n_rows {
            let phase = std::f64::consts::TAU
                * coordinates[[replicate, row]].rem_euclid(period)
                / period;
            let (sin_phase, cos_phase) = phase.sin_cos();
            sin[replicate][row] = sin_phase;
            cos[replicate][row] = cos_phase;
        }
    }

    let coverage = (0..n_replicates)
        .map(|replicate| embedding_coverage(&cos[replicate], &sin[replicate], replicate))
        .collect::<Vec<_>>();
    let mut pairs = Vec::with_capacity(n_replicates * (n_replicates - 1) / 2);
    let mut aggregate = Vec::with_capacity(pairs.capacity());
    for left in 0..n_replicates {
        for right in (left + 1)..n_replicates {
            if !(coverage[left].well_posed && coverage[right].well_posed) {
                pairs.push(CircularPairConcordance {
                    left,
                    right,
                    rotation_score: None,
                    reflection_score: None,
                    aligned_score: None,
                    reflected: None,
                    phase_shift: None,
                });
                continue;
            }
            let mut rotation_cos = 0.0;
            let mut rotation_sin = 0.0;
            let mut reflection_cos = 0.0;
            let mut reflection_sin = 0.0;
            for row in 0..n_rows {
                let ca = cos[left][row];
                let sa = sin[left][row];
                let cb = cos[right][row];
                let sb = sin[right][row];
                rotation_cos += ca * cb + sa * sb;
                rotation_sin += sa * cb - ca * sb;
                reflection_cos += ca * cb - sa * sb;
                reflection_sin += sa * cb + ca * sb;
            }
            let denominator = n_rows as f64;
            let rotation_score = rotation_cos.hypot(rotation_sin) / denominator;
            let reflection_score = reflection_cos.hypot(reflection_sin) / denominator;
            let reflected = reflection_score > rotation_score;
            let (aligned_score, shift_radians) = if reflected {
                (
                    reflection_score,
                    reflection_sin.atan2(reflection_cos),
                )
            } else {
                (rotation_score, rotation_sin.atan2(rotation_cos))
            };
            let aligned_score = aligned_score.clamp(0.0, 1.0);
            aggregate.push(aligned_score);
            pairs.push(CircularPairConcordance {
                left,
                right,
                rotation_score: Some(rotation_score.clamp(0.0, 1.0)),
                reflection_score: Some(reflection_score.clamp(0.0, 1.0)),
                aligned_score: Some(aligned_score),
                reflected: Some(reflected),
                phase_shift: Some(
                    (shift_radians / std::f64::consts::TAU * period).rem_euclid(period),
                ),
            });
        }
    }

    let all_pairs_well_posed = aggregate.len() == pairs.len();
    let minimum_aligned_score = all_pairs_well_posed
        .then(|| aggregate.iter().copied().reduce(f64::min))
        .flatten();
    let mean_aligned_score = all_pairs_well_posed
        .then(|| aggregate.iter().sum::<f64>() / aggregate.len() as f64);
    Ok(CircularConcordanceReport {
        n_replicates,
        n_rows,
        period,
        coverage,
        pairs,
        minimum_aligned_score,
        mean_aligned_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn quotient_recovers_rotation_and_reflection_without_accepting_collapse() {
        let n = 37;
        let mut coordinates = Array2::<f64>::zeros((3, n));
        for row in 0..n {
            let base = ((row * row + 3 * row + 1) as f64 / n as f64).rem_euclid(1.0);
            coordinates[[0, row]] = base;
            coordinates[[1, row]] = (base + 0.23).rem_euclid(1.0);
            coordinates[[2, row]] = (0.41 - base).rem_euclid(1.0);
        }
        let report = circular_concordance(coordinates.view(), 1.0).expect("report");
        assert!(report.coverage.iter().all(|entry| entry.well_posed));
        let resolution = f64::EPSILON.sqrt();
        assert!(report.pairs[0].aligned_score.unwrap() >= 1.0 - resolution);
        assert_eq!(report.pairs[0].reflected, Some(false));
        assert!(report.pairs[1].aligned_score.unwrap() >= 1.0 - resolution);
        assert_eq!(report.pairs[1].reflected, Some(true));

        coordinates.row_mut(2).fill(0.0);
        let collapsed = circular_concordance(coordinates.view(), 1.0).expect("report");
        assert!(!collapsed.coverage[2].well_posed);
        assert_eq!(collapsed.pairs[1].aligned_score, None);
        assert_eq!(collapsed.mean_aligned_score, None);
    }
}

//! The joint latent law of a `K ≥ 2` score vector on the survival
//! marginal-slope declared-law path (gam#2929).
//!
//! With one slope surface per score the row index is
//!
//! ```text
//!     η = α(q, r) + rᵀz,        Σ_m w_m Φ(−(α + rᵀu_m)) = Φ(−q),        r = s_f·g,
//! ```
//!
//! and only the law of the scalar DRIVE `rᵀz` enters the anchor
//! (`Descent.Portability.GaussianAnchor.drive_anchor_closed_form` for a Gaussian
//! drive, `MarginalAnchor.exists_unique_anchor` for a finite one). The drive's
//! direction moves with the slope coefficients, so its law cannot be projected
//! once and stored as one scalar grid: what the fit declares is a finite law of
//! the score VECTOR per context, and each evaluation projects it onto the row's
//! own `r`. That is `M` joint nodes in `ℝ^K`, not a product grid.
//!
//! # The law per context
//!
//! The context is the marginal-index span `a(C)`, the same block the gam#2766
//! conditional covariance is fitted on. The row's law is the transport of ONE
//! pooled residual law through that context's affine map,
//!
//! ```text
//!     u_m(a) = μ + L(a)·ε_m ,
//! ```
//!
//! with `μ` the weighted score mean, `L(a)` the Cholesky factor of the conditional
//! covariance `Σ(a)` when the pair-wise Rao gate escalated to one and of the pooled
//! `Σ̄` otherwise, and `{ε_m, w_m}` the weighted empirical law of the whitened
//! training residuals `ε_i = L(a_i)⁻¹(z_i − μ)`. On Gaussian scores this is
//! `z | a ~ N(μ, Σ(a))` in the limit, so the anchor is `q·√(1 + rᵀΣ(a)r) − rᵀμ` —
//! the gam#2766 closed form — and on any other residual shape it is the anchor
//! of that shape, which no closed form reproduces.
//!
//! # Compression
//!
//! `n` residual vectors are compressed to at most [`DEFAULT_JOINT_LATENT_NODES`]
//! joint nodes by a deterministic weighted k-means, and the compressed law is then
//! moved by one affine map onto the empirical residual law's exact weighted mean
//! and covariance. A sample no larger than the node budget is kept as it is. The
//! compressed law IS the declared law: the fit anchors on it, it is persisted, and
//! prediction replays the same nodes.
//!
//! # Derivatives
//!
//! [`joint_anchor_into`] differentiates `G(α, q, r) = Σ_m w_m F(α + rᵀu_m) − F(q)`,
//! `F(x) = Φ(−x)`, implicitly through every order the anchored order-two row
//! reads: `α_q, α_r` through `α_qqq, α_qqr, α_qrr`. Every partial of `G` in `(α, r)`
//! is a moment of `F^{(n)}(η_m)` against `u_m`, and the formulas are the scalar
//! anchor's (`AnchorDerivatives::at`) with the slope index promoted to a vector.

use super::*;

use crate::bms::ConditionalScoreCovariance;
use serde::{Deserialize, Serialize};

/// Joint nodes a `K ≥ 2` law is compressed to. The scalar grids carry 65 nodes
/// on one axis; a law in several dimensions has to resolve direction as well as
/// radius, so it carries about twice that.
pub(crate) const DEFAULT_JOINT_LATENT_NODES: usize = 128;

/// Lloyd iterations the compression may spend before it keeps the partition it
/// has. Assignments almost always settle far earlier; the cap only bounds a
/// cycling partition, and the moment correction that follows makes the declared
/// law's first two moments exact whatever the partition.
const JOINT_LAW_LLOYD_ITERATIONS: usize = 60;

/// The persisted joint law of a `K ≥ 2` score vector (gam#2929): the pooled
/// whitened residual law and the affine map that transports it to a context.
///
/// This is fit state prediction must replay. The coefficients of a fit on it are
/// defined against its anchor, so a predictor that lowered the same coefficients
/// in closed form — or re-derived the law from its own sample — would evaluate a
/// different model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurvivalJointLatentLaw {
    /// Number of latent-score coordinates `K`. Always `≥ 2`.
    pub score_dim: usize,
    /// The whitened residual nodes `ε_m`, `M × K`.
    pub residual_nodes: Vec<Vec<f64>>,
    /// Their weights, positive and summing to one.
    pub weights: Vec<f64>,
    /// The weighted score mean `μ`.
    pub score_mean: Vec<f64>,
    /// The lower Cholesky factor `L̄` of the pooled score covariance, `K × K`.
    pub pooled_factor: Vec<Vec<f64>>,
    /// The conditional covariance `Σ(a)` the transport reads in place of `L̄`,
    /// when the gam#2766 pair gate escalated to one. Its factor is evaluated on
    /// the marginal-index span, so prediction must rebuild that span.
    #[serde(default)]
    pub conditional: Option<ConditionalScoreCovariance>,
}

impl SurvivalJointLatentLaw {
    pub fn validate(&self, context: &str) -> Result<(), String> {
        let k = self.score_dim;
        if k < 2 {
            return Err(format!(
                "{context}: a joint latent law needs at least two score coordinates, got {k}"
            ));
        }
        let m = self.residual_nodes.len();
        if m <= k || self.weights.len() != m {
            return Err(format!(
                "{context}: a joint latent law over K={k} scores needs more than K nodes with one \
                 weight each; got {m} nodes and {} weights",
                self.weights.len()
            ));
        }
        for (index, node) in self.residual_nodes.iter().enumerate() {
            if node.len() != k || node.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "{context}: joint latent node {index} must hold {k} finite coordinates"
                ));
            }
        }
        let mut total = 0.0;
        for (index, &weight) in self.weights.iter().enumerate() {
            if !(weight.is_finite() && weight > 0.0) {
                return Err(format!(
                    "{context}: joint latent weight {index} must be finite and positive, got {weight}"
                ));
            }
            total += weight;
        }
        if (total - 1.0).abs() > 1e-8 {
            return Err(format!(
                "{context}: joint latent weights must sum to one, got {total}"
            ));
        }
        if self.score_mean.len() != k || self.score_mean.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "{context}: joint latent score mean must hold {k} finite values"
            ));
        }
        if self.pooled_factor.len() != k {
            return Err(format!(
                "{context}: joint latent pooled factor must be {k}x{k}"
            ));
        }
        for (row, values) in self.pooled_factor.iter().enumerate() {
            if values.len() != k || values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "{context}: joint latent pooled factor row {row} must hold {k} finite values"
                ));
            }
            if !(values[row] > 0.0) {
                return Err(format!(
                    "{context}: joint latent pooled factor has a non-positive pivot at {row}"
                ));
            }
            if values[row + 1..].iter().any(|&value| value != 0.0) {
                return Err(format!(
                    "{context}: joint latent pooled factor must be lower triangular (row {row})"
                ));
            }
        }
        if let Some(model) = self.conditional.as_ref()
            && model.score_dim != k
        {
            return Err(format!(
                "{context}: joint latent conditional covariance is K={} but the law is K={k}",
                model.score_dim
            ));
        }
        Ok(())
    }

    /// Materialise the law for `n_rows` rows whose marginal-index span is
    /// `conditioning`. The span is required exactly when the law carries a
    /// conditional covariance.
    pub(crate) fn runtime(
        &self,
        conditioning: Option<ArrayView2<'_, f64>>,
        n_rows: usize,
    ) -> Result<JointLatentLawRuntime, String> {
        self.validate("survival marginal-slope joint latent law")?;
        let k = self.score_dim;
        let pooled: Vec<f64> = self
            .pooled_factor
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let factors = materialize_joint_factors(
            k,
            pooled,
            self.conditional.as_ref(),
            conditioning,
            n_rows,
        )?;
        let weights = self.weights.clone();
        let log_weights = weights.iter().map(|weight| weight.ln()).collect();
        Ok(JointLatentLawRuntime {
            score_dim: k,
            node_count: self.residual_nodes.len(),
            residual_nodes: self
                .residual_nodes
                .iter()
                .flat_map(|node| node.iter().copied())
                .collect(),
            weights,
            log_weights,
            score_mean: self.score_mean.clone(),
            factors,
        })
    }
}

/// The transport factor `L(a)` per row, or the pooled one for every row.
#[derive(Clone, Debug)]
enum JointFactors {
    /// `K × K`, row-major, lower triangular.
    Pooled(Vec<f64>),
    /// `n × K × K`, row-major.
    PerRow(Arc<Vec<f64>>),
}

fn materialize_joint_factors(
    k: usize,
    pooled: Vec<f64>,
    conditional: Option<&ConditionalScoreCovariance>,
    conditioning: Option<ArrayView2<'_, f64>>,
    n_rows: usize,
) -> Result<JointFactors, String> {
    let Some(model) = conditional else {
        return Ok(JointFactors::Pooled(pooled));
    };
    let a_block = conditioning.ok_or_else(|| {
        "survival marginal-slope joint latent law carries a conditional covariance and needs the \
         marginal-index span it was fitted on"
            .to_string()
    })?;
    if a_block.nrows() != n_rows {
        return Err(format!(
            "survival marginal-slope joint latent law span has {} rows, expected {n_rows}",
            a_block.nrows()
        ));
    }
    let mut stack = vec![0.0; n_rows * k * k];
    let mut factor = Array2::<f64>::zeros((k, k));
    for row in 0..n_rows {
        model.factor_into(a_block.row(row), &mut factor)?;
        let target = &mut stack[row * k * k..(row + 1) * k * k];
        for (slot, value) in target.iter_mut().zip(factor.iter()) {
            *slot = *value;
        }
    }
    Ok(JointFactors::PerRow(Arc::new(stack)))
}

/// A joint law ready for the row program: flattened nodes, log-weights, and the
/// per-row transport.
#[derive(Clone, Debug)]
pub(crate) struct JointLatentLawRuntime {
    score_dim: usize,
    node_count: usize,
    residual_nodes: Vec<f64>,
    weights: Vec<f64>,
    log_weights: Vec<f64>,
    score_mean: Vec<f64>,
    factors: JointFactors,
}

impl JointLatentLawRuntime {
    #[inline]
    pub(crate) fn score_dim(&self) -> usize {
        self.score_dim
    }

    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        self.node_count
    }

    /// The node weights, one per residual node, summing to one.
    #[inline]
    pub(crate) fn weights(&self) -> &[f64] {
        &self.weights
    }

    #[inline]
    fn factor(&self, row: usize) -> Result<&[f64], String> {
        let kk = self.score_dim * self.score_dim;
        match &self.factors {
            JointFactors::Pooled(factor) => Ok(factor),
            JointFactors::PerRow(stack) => stack.get(row * kk..(row + 1) * kk).ok_or_else(|| {
                format!(
                    "survival marginal-slope joint latent law was materialised for {} rows; row \
                     {row} is outside it",
                    stack.len() / kk
                )
            }),
        }
    }

    /// `u_m = μ + L(a_row)·ε_m`, row-major into `out` (`M × K`).
    pub(crate) fn row_nodes_into(&self, row: usize, out: &mut [f64]) -> Result<(), String> {
        let k = self.score_dim;
        if out.len() != self.node_count * k {
            return Err(format!(
                "survival marginal-slope joint latent node buffer holds {} values, expected {}",
                out.len(),
                self.node_count * k
            ));
        }
        let factor = self.factor(row)?;
        for m in 0..self.node_count {
            let residual = &self.residual_nodes[m * k..(m + 1) * k];
            for j in 0..k {
                let mut value = self.score_mean[j];
                for i in 0..=j {
                    value += factor[j * k + i] * residual[i];
                }
                out[m * k + j] = value;
            }
        }
        Ok(())
    }
}

/// Lower Cholesky factor of a `K × K` covariance, row-major, with each pivot
/// floored at the band gam#2766 floors a zero innovation inside
/// (`128·K·ε·max second moment`). A collinear score geometry is a real input;
/// the floor keeps the whitening finite along its degenerate direction, where
/// every residual is zero anyway.
fn lower_cholesky_with_floor(matrix: ArrayView2<'_, f64>) -> Result<Vec<f64>, String> {
    let k = matrix.nrows();
    let max_diagonal = (0..k).map(|j| matrix[[j, j]].abs()).fold(0.0_f64, f64::max);
    if !(max_diagonal.is_finite() && max_diagonal > 0.0) {
        return Err(format!(
            "survival marginal-slope joint latent law needs a positive finite score variance; \
             largest diagonal entry is {max_diagonal}"
        ));
    }
    let floor = 128.0 * k as f64 * f64::EPSILON * max_diagonal;
    let mut factor = vec![0.0; k * k];
    for j in 0..k {
        let mut pivot = matrix[[j, j]];
        for i in 0..j {
            pivot -= factor[j * k + i] * factor[j * k + i];
        }
        let pivot = pivot.max(floor).sqrt();
        factor[j * k + j] = pivot;
        for r in j + 1..k {
            let mut value = matrix[[r, j]];
            for i in 0..j {
                value -= factor[r * k + i] * factor[j * k + i];
            }
            factor[r * k + j] = value / pivot;
        }
    }
    if factor.iter().any(|value| !value.is_finite()) {
        return Err(
            "survival marginal-slope joint latent law: the score covariance factor is not finite"
                .to_string(),
        );
    }
    Ok(factor)
}

/// Solve `L·x = b` in place for a row-major lower-triangular `L`.
#[inline]
fn forward_substitute(factor: &[f64], k: usize, values: &mut [f64]) {
    for j in 0..k {
        let mut value = values[j];
        for i in 0..j {
            value -= factor[j * k + i] * values[i];
        }
        values[j] = value / factor[j * k + j];
    }
}

/// Weighted mean and covariance of `points` (`count × K`, row-major).
fn weighted_moments(points: &[f64], weights: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let count = weights.len();
    let total: f64 = weights.iter().sum();
    let mut mean = vec![0.0; k];
    for row in 0..count {
        for j in 0..k {
            mean[j] += weights[row] * points[row * k + j];
        }
    }
    for value in mean.iter_mut() {
        *value /= total;
    }
    let mut covariance = vec![0.0; k * k];
    for row in 0..count {
        for i in 0..k {
            let di = points[row * k + i] - mean[i];
            for j in 0..=i {
                covariance[i * k + j] += weights[row] * di * (points[row * k + j] - mean[j]) / total;
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            covariance[j * k + i] = covariance[i * k + j];
        }
    }
    (mean, covariance)
}

#[inline]
fn splitmix_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

#[inline]
fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

/// Compress a weighted sample in `ℝ^K` to at most `node_count` joint nodes whose
/// weighted mean and covariance are the sample's exactly. Deterministic: the
/// k-means++ seeding draws from a fixed-seed generator over the sample in its
/// given order.
fn compress_joint_sample(
    points: &[f64],
    weights: &[f64],
    k: usize,
    node_count: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let active: Vec<usize> = (0..weights.len()).filter(|&row| weights[row] > 0.0).collect();
    if active.len() <= k {
        return Err(format!(
            "survival marginal-slope joint latent law needs more than K={k} positive-weight rows; \
             got {}",
            active.len()
        ));
    }
    let sample: Vec<f64> = active
        .iter()
        .flat_map(|&row| points[row * k..(row + 1) * k].iter().copied())
        .collect();
    let sample_weights: Vec<f64> = active.iter().map(|&row| weights[row]).collect();
    let total: f64 = sample_weights.iter().sum();
    let count = active.len();
    if count <= node_count {
        return Ok((
            sample,
            sample_weights.iter().map(|weight| weight / total).collect(),
        ));
    }

    // k-means++ seeding: the first center by weight, every later one by
    // weight times squared distance to the nearest center already chosen.
    let mut state = 0x2929_u64;
    let mut centers: Vec<f64> = Vec::with_capacity(node_count * k);
    let mut nearest = vec![f64::INFINITY; count];
    let mut next = {
        let target = splitmix_unit(&mut state) * total;
        let mut cumulative = 0.0;
        let mut chosen = count - 1;
        for (index, &weight) in sample_weights.iter().enumerate() {
            cumulative += weight;
            if cumulative >= target {
                chosen = index;
                break;
            }
        }
        chosen
    };
    while centers.len() / k < node_count {
        centers.extend_from_slice(&sample[next * k..(next + 1) * k]);
        let center = &centers[centers.len() - k..];
        let mut mass = 0.0;
        for index in 0..count {
            let distance = squared_distance(&sample[index * k..(index + 1) * k], center);
            if distance < nearest[index] {
                nearest[index] = distance;
            }
            mass += sample_weights[index] * nearest[index];
        }
        if !(mass > 0.0) {
            // Every remaining point coincides with a center: fewer distinct
            // points than nodes, and the partition below is exact.
            break;
        }
        let target = splitmix_unit(&mut state) * mass;
        let mut cumulative = 0.0;
        next = count - 1;
        for index in 0..count {
            cumulative += sample_weights[index] * nearest[index];
            if cumulative >= target {
                next = index;
                break;
            }
        }
    }
    let clusters = centers.len() / k;

    // Lloyd iterations.
    let mut assignment = vec![usize::MAX; count];
    let mut masses = vec![0.0; clusters];
    let mut sums = vec![0.0; clusters * k];
    for _ in 0..JOINT_LAW_LLOYD_ITERATIONS {
        let mut changed = false;
        for index in 0..count {
            let point = &sample[index * k..(index + 1) * k];
            let mut best = 0usize;
            let mut best_distance = f64::INFINITY;
            for cluster in 0..clusters {
                let distance = squared_distance(point, &centers[cluster * k..(cluster + 1) * k]);
                if distance < best_distance {
                    best_distance = distance;
                    best = cluster;
                }
            }
            nearest[index] = best_distance;
            if assignment[index] != best {
                assignment[index] = best;
                changed = true;
            }
        }
        masses.fill(0.0);
        sums.fill(0.0);
        for index in 0..count {
            let cluster = assignment[index];
            masses[cluster] += sample_weights[index];
            for j in 0..k {
                sums[cluster * k + j] += sample_weights[index] * sample[index * k + j];
            }
        }
        for cluster in 0..clusters {
            if masses[cluster] > 0.0 {
                for j in 0..k {
                    centers[cluster * k + j] = sums[cluster * k + j] / masses[cluster];
                }
            } else {
                // An emptied cluster takes the worst-served point.
                let worst = (0..count)
                    .max_by(|&left, &right| {
                        (sample_weights[left] * nearest[left])
                            .total_cmp(&(sample_weights[right] * nearest[right]))
                    })
                    .unwrap_or(0);
                centers[cluster * k..(cluster + 1) * k]
                    .copy_from_slice(&sample[worst * k..(worst + 1) * k]);
                nearest[worst] = 0.0;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut nodes = Vec::with_capacity(clusters * k);
    let mut node_weights = Vec::with_capacity(clusters);
    for cluster in 0..clusters {
        if masses[cluster] > 0.0 {
            nodes.extend_from_slice(&centers[cluster * k..(cluster + 1) * k]);
            node_weights.push(masses[cluster] / total);
        }
    }
    let node_total: f64 = node_weights.iter().sum();
    for weight in node_weights.iter_mut() {
        *weight /= node_total;
    }
    if node_weights.len() <= k {
        return Err(format!(
            "survival marginal-slope joint latent law compressed to {} nodes, which cannot carry \
             a K={k} covariance",
            node_weights.len()
        ));
    }

    // Moment correction: x ↦ ē + L_e·L_c⁻¹·(x − c̄), which carries the compressed
    // law's weighted mean and covariance onto the sample's.
    let (sample_mean, sample_covariance) = weighted_moments(&sample, &sample_weights, k);
    let (node_mean, node_covariance) = weighted_moments(&nodes, &node_weights, k);
    let sample_factor =
        lower_cholesky_with_floor(ArrayView2::from_shape((k, k), &sample_covariance).map_err(
            |error| format!("joint latent sample covariance shape: {error}"),
        )?)?;
    let node_factor =
        lower_cholesky_with_floor(ArrayView2::from_shape((k, k), &node_covariance).map_err(
            |error| format!("joint latent node covariance shape: {error}"),
        )?)?;
    let mut centered = vec![0.0; k];
    for node in 0..node_weights.len() {
        for j in 0..k {
            centered[j] = nodes[node * k + j] - node_mean[j];
        }
        forward_substitute(&node_factor, k, &mut centered);
        for j in 0..k {
            let mut value = sample_mean[j];
            for i in 0..=j {
                value += sample_factor[j * k + i] * centered[i];
            }
            nodes[node * k + j] = value;
        }
    }
    if nodes.iter().any(|value| !value.is_finite()) {
        return Err(
            "survival marginal-slope joint latent law moment correction produced a non-finite node"
                .to_string(),
        );
    }
    Ok((nodes, node_weights))
}

/// Why a per-score fit over `K ≥ 2` scores cannot anchor on the joint latent
/// law under the latent measures the gate settled on, or `None` when it can.
///
/// The joint law transports one pooled residual law by `μ + L(a)·ε`: a moving
/// covariance reaches each row's law, a moving mean or shape does not. A
/// local-empirical measure is the gate's finding that a score's conditional law
/// moves on the span, so the joint law would anchor on a law other than the one
/// the measure estimated.
pub(crate) fn joint_latent_law_measure_refusal(
    score_dim: usize,
    per_score_measure: &[crate::bms::LatentMeasureKind],
) -> Option<String> {
    per_score_measure
        .iter()
        .any(|measure| matches!(measure, crate::bms::LatentMeasureKind::LocalEmpirical { .. }))
        .then(|| {
            format!(
                "a local-empirical latent law on a score of a per-score slope over K={score_dim} \
                 scores is refused: the joint law transports one pooled residual law by \
                 μ + L(a)·ε, which follows a moving covariance but not a moving mean or shape of \
                 a score's conditional law, so it would anchor on a law other than the one the \
                 latent measure estimated"
            )
        })
}

/// Build the joint law of the score vector the fit consumes and its runtime.
///
/// `scores` are the calibrated scores the row program sees; `covariance` is the
/// field [`resolve_score_covariance_field`] resolved on them, whose conditional
/// model — when present — supplies the transport; `conditioning` is the span it
/// was fitted on.
pub(crate) fn build_joint_latent_law(
    scores: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    covariance: &ScoreCovarianceField,
    conditioning: Option<ArrayView2<'_, f64>>,
    node_count: usize,
) -> Result<(SurvivalJointLatentLaw, JointLatentLawRuntime), String> {
    let (n, k) = scores.dim();
    if k < 2 {
        return Err(format!(
            "survival marginal-slope joint latent law needs K ≥ 2 scores, got {k}"
        ));
    }
    if weights.len() != n {
        return Err(format!(
            "survival marginal-slope joint latent law weight length {} does not match {n} rows",
            weights.len()
        ));
    }
    if covariance.dim() != k {
        return Err(format!(
            "survival marginal-slope joint latent law: score covariance is K={} but scores are K={k}",
            covariance.dim()
        ));
    }
    let model = covariance.model();
    let score_mean: Vec<f64> = match model {
        Some(model) => model.score_mean.clone(),
        None => {
            let points: Vec<f64> = scores.iter().copied().collect();
            let positive: Vec<f64> = weights.iter().map(|&weight| weight.max(0.0)).collect();
            weighted_moments(&points, &positive, k).0
        }
    };
    let pooled = lower_cholesky_with_floor(covariance.pooled_covariance().to_dense().view())?;
    let factors = materialize_joint_factors(k, pooled.clone(), model, conditioning, n)?;
    let runtime_shell = JointLatentLawRuntime {
        score_dim: k,
        node_count: 0,
        residual_nodes: Vec::new(),
        weights: Vec::new(),
        log_weights: Vec::new(),
        score_mean: score_mean.clone(),
        factors,
    };
    let mut residuals = vec![0.0; n * k];
    for row in 0..n {
        let factor = runtime_shell.factor(row)?;
        let target = &mut residuals[row * k..(row + 1) * k];
        for j in 0..k {
            target[j] = scores[[row, j]] - score_mean[j];
        }
        forward_substitute(factor, k, target);
        if target.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "survival marginal-slope joint latent law: whitened residual at row {row} is not finite"
            ));
        }
    }
    let positive_weights: Vec<f64> = weights.iter().map(|&weight| weight.max(0.0)).collect();
    let (nodes, node_weights) = compress_joint_sample(&residuals, &positive_weights, k, node_count)?;
    let persisted = SurvivalJointLatentLaw {
        score_dim: k,
        residual_nodes: nodes.chunks(k).map(<[f64]>::to_vec).collect(),
        weights: node_weights.clone(),
        score_mean,
        pooled_factor: pooled.chunks(k).map(<[f64]>::to_vec).collect(),
        conditional: model.cloned(),
    };
    persisted.validate("survival marginal-slope joint latent law")?;
    let log_weights = node_weights.iter().map(|weight| weight.ln()).collect();
    let runtime = JointLatentLawRuntime {
        node_count: node_weights.len(),
        residual_nodes: nodes,
        weights: node_weights,
        log_weights,
        ..runtime_shell
    };
    log::debug!(
        "[survival-marginal-slope latent-z] the row index is anchored on the joint law of K={k} \
         scores: {} nodes, {} transport (gam#2929)",
        runtime.node_count,
        if model.is_some() {
            "conditional Σ(a)"
        } else {
            "pooled Σ̄"
        },
    );
    Ok((persisted, runtime))
}

/// The arithmetic the joint anchor's implicit derivatives and the anchored
/// pullback are formed in: `f64` for the order-two lowering, and [`Rate`] — a
/// value with its derivative along one fixed direction — for the directional
/// derivative of the row Hessian. Carrying the same formulas in `Rate` is the
/// chain rule taken one order further, at the same `O(M·K²)` per row whatever
/// the number of scores.
pub(crate) trait AnchorScalar:
    Copy
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
{
    fn real(x: f64) -> Self;
    fn value(self) -> f64;
    fn exp(self) -> Self;
    fn recip(self) -> Self;
    fn is_zero(self) -> bool;
}

impl AnchorScalar for f64 {
    #[inline(always)]
    fn real(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn value(self) -> f64 {
        self
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline(always)]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline(always)]
    fn is_zero(self) -> bool {
        self == 0.0
    }
}

/// A value `v` and its derivative `t` along one fixed direction: `ε² = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rate {
    pub(crate) v: f64,
    pub(crate) t: f64,
}

impl std::ops::Add for Rate {
    type Output = Rate;
    #[inline(always)]
    fn add(self, o: Rate) -> Rate {
        Rate {
            v: self.v + o.v,
            t: self.t + o.t,
        }
    }
}

impl std::ops::Sub for Rate {
    type Output = Rate;
    #[inline(always)]
    fn sub(self, o: Rate) -> Rate {
        Rate {
            v: self.v - o.v,
            t: self.t - o.t,
        }
    }
}

impl std::ops::Mul for Rate {
    type Output = Rate;
    #[inline(always)]
    fn mul(self, o: Rate) -> Rate {
        Rate {
            v: self.v * o.v,
            t: self.v * o.t + self.t * o.v,
        }
    }
}

impl std::ops::Neg for Rate {
    type Output = Rate;
    #[inline(always)]
    fn neg(self) -> Rate {
        Rate {
            v: -self.v,
            t: -self.t,
        }
    }
}

impl std::ops::AddAssign for Rate {
    #[inline(always)]
    fn add_assign(&mut self, o: Rate) {
        self.v += o.v;
        self.t += o.t;
    }
}

impl AnchorScalar for Rate {
    #[inline(always)]
    fn real(x: f64) -> Self {
        Rate { v: x, t: 0.0 }
    }
    #[inline(always)]
    fn value(self) -> f64 {
        self.v
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.v.exp();
        Rate { v: e, t: self.t * e }
    }
    #[inline(always)]
    fn recip(self) -> Self {
        let inv = 1.0 / self.v;
        Rate {
            v: inv,
            t: -self.t * inv * inv,
        }
    }
    #[inline(always)]
    fn is_zero(self) -> bool {
        self.v == 0.0 && self.t == 0.0
    }
}

/// Moment scratch for [`joint_anchor_formulas`].
#[derive(Clone, Debug)]
pub(crate) struct JointAnchorMoments<S = f64> {
    v1: Vec<S>,
    v2: Vec<S>,
    v3: Vec<S>,
    m2: Vec<S>,
    m3: Vec<S>,
}

impl<S: AnchorScalar> JointAnchorMoments<S> {
    pub(crate) fn new(k: usize) -> Self {
        let zero = S::real(0.0);
        Self {
            v1: vec![zero; k],
            v2: vec![zero; k],
            v3: vec![zero; k],
            m2: vec![zero; k * k],
            m3: vec![zero; k * k],
        }
    }
}

/// The anchor of a joint law and its implicit derivatives in `(q, r)`.
#[derive(Clone, Debug)]
pub(crate) struct JointAnchorDerivatives<S = f64> {
    pub(crate) alpha: S,
    pub(crate) a_q: S,
    pub(crate) a_qq: S,
    pub(crate) a_qqq: S,
    pub(crate) a_r: Vec<S>,
    pub(crate) a_qr: Vec<S>,
    pub(crate) a_qqr: Vec<S>,
    /// `K × K`, row-major, symmetric.
    pub(crate) a_rr: Vec<S>,
    /// `K × K`, row-major, symmetric.
    pub(crate) a_qrr: Vec<S>,
}

impl<S: AnchorScalar> JointAnchorDerivatives<S> {
    pub(crate) fn new(k: usize) -> Self {
        let zero = S::real(0.0);
        Self {
            alpha: zero,
            a_q: zero,
            a_qq: zero,
            a_qqq: zero,
            a_r: vec![zero; k],
            a_qr: vec![zero; k],
            a_qqr: vec![zero; k],
            a_rr: vec![zero; k * k],
            a_qrr: vec![zero; k * k],
        }
    }
}

/// Solve the anchor of the joint law `{u_m, w_m}` at the marginal index `q`
/// and the drive `d_m = rᵀu_m`, and differentiate it through order three.
///
/// The root is found on the STANDARDISED drive: with `m̄, s` the drive's weighted
/// mean and spread, `α = α̃ − m̄` where `α̃` is the scalar anchor of the nodes
/// `(d_m − m̄)/s` at observed slope `s`, whose Gaussian seed `q·√(1 + s²)` is
/// the drive closed form. The moments are formed with every density divided by
/// the largest one, which cancels in each implicit ratio and keeps the sums
/// finite where the densities themselves would underflow.
pub(crate) fn joint_anchor_into(
    q: f64,
    nodes: &[f64],
    drive: &[f64],
    weights: &[f64],
    log_weights: &[f64],
    standardized: &mut [f64],
    moments: &mut JointAnchorMoments,
    out: &mut JointAnchorDerivatives,
) -> Result<(), String> {
    let m_count = drive.len();
    let k = out.a_r.len();
    if nodes.len() != m_count * k
        || weights.len() != m_count
        || log_weights.len() != m_count
        || standardized.len() != m_count
    {
        return Err(format!(
            "survival marginal-slope joint anchor buffers disagree: nodes={}, drive={m_count}, \
             weights={}, K={k}",
            nodes.len(),
            weights.len()
        ));
    }
    // A degenerate drive puts every node at `m̄`, where `Φ(−(α + m̄)) = Φ(−q)`.
    let alpha = joint_anchor_root(q, drive, weights, log_weights, standardized)?;
    joint_anchor_formulas(alpha, q, nodes, drive, log_weights, moments, out)
}

/// The implicit derivatives of the joint anchor at an already solved root:
/// every partial of `G(α, q, r) = Σ_m w_m F(α + rᵀu_m) − F(q)` in `(α, r)` is a
/// moment of `F^{(n)}(η_m)` against `u_m`, and the formulas are the scalar
/// anchor's (`AnchorDerivatives::at`) with the slope index promoted to a vector.
///
/// In `f64` this is the order-two lowering's anchor. In [`Rate`], with `alpha`,
/// `q` and the drive carrying their derivatives along one direction, every
/// output carries its own directional derivative — the chain rule through the
/// same formulas, which is what the moving-Hessian outer gradient reads.
pub(crate) fn joint_anchor_formulas<S: AnchorScalar>(
    alpha: S,
    q: S,
    nodes: &[f64],
    drive: &[S],
    log_weights: &[f64],
    moments: &mut JointAnchorMoments<S>,
    out: &mut JointAnchorDerivatives<S>,
) -> Result<(), String> {
    let m_count = drive.len();
    let k = out.a_r.len();
    if nodes.len() != m_count * k || log_weights.len() != m_count {
        return Err(format!(
            "survival marginal-slope joint anchor formula buffers disagree: nodes={}, \
             drive={m_count}, log weights={}, K={k}",
            nodes.len(),
            log_weights.len()
        ));
    }
    let zero = S::real(0.0);
    let half = S::real(0.5);
    let one = S::real(1.0);
    let mut shift = f64::NEG_INFINITY;
    for m in 0..m_count {
        let eta = alpha.value() + drive[m].value();
        let log_density = log_weights[m] - 0.5 * eta * eta;
        if log_density > shift {
            shift = log_density;
        }
    }
    if !shift.is_finite() {
        return Err(format!(
            "survival marginal-slope joint anchor has no finite node density at α={}",
            alpha.value()
        ));
    }
    moments.v1.fill(zero);
    moments.v2.fill(zero);
    moments.v3.fill(zero);
    moments.m2.fill(zero);
    moments.m3.fill(zero);
    let mut s1 = zero;
    let mut s2 = zero;
    let mut s3 = zero;
    for m in 0..m_count {
        let eta = alpha + drive[m];
        let density = (S::real(log_weights[m] - shift) - half * eta * eta).exp();
        // F′ = −φ, F″ = xφ, F‴ = (1 − x²)φ for F(x) = Φ(−x).
        let f1 = -density;
        let f2 = eta * density;
        let f3 = (one - eta * eta) * density;
        s1 += f1;
        s2 += f2;
        s3 += f3;
        let u = &nodes[m * k..(m + 1) * k];
        for i in 0..k {
            let ui = S::real(u[i]);
            moments.v1[i] += f1 * ui;
            moments.v2[i] += f2 * ui;
            moments.v3[i] += f3 * ui;
            for j in 0..=i {
                let product = S::real(u[i] * u[j]);
                moments.m2[i * k + j] += f2 * product;
                moments.m3[i * k + j] += f3 * product;
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            moments.m2[j * k + i] = moments.m2[i * k + j];
            moments.m3[j * k + i] = moments.m3[i * k + j];
        }
    }
    let phi_q = (S::real(-shift) - half * q * q).exp();
    if !(s1.value().is_finite() && s1.value() < 0.0 && phi_q.value().is_finite()) {
        return Err(format!(
            "survival marginal-slope joint anchor has no finite gradient: G_α={:e}, φ(q)={:e} \
             at α={}, q={}",
            s1.value(),
            phi_q.value(),
            alpha.value(),
            q.value()
        ));
    }
    let g_q = phi_q;
    let g_qq = -(q * phi_q);
    let g_qqq = -((one - q * q) * phi_q);
    let two = S::real(2.0);
    let three = S::real(3.0);
    // Every implicit derivative is `−(…)/G_α`.
    let inv = -s1.recip();
    let a_q = g_q * inv;
    for i in 0..k {
        out.a_r[i] = moments.v1[i] * inv;
    }
    let a_qq = (g_qq + s2 * a_q * a_q) * inv;
    for i in 0..k {
        out.a_qr[i] = (moments.v2[i] * a_q + s2 * a_q * out.a_r[i]) * inv;
    }
    for i in 0..k {
        for j in 0..k {
            out.a_rr[i * k + j] = (moments.m2[i * k + j]
                + moments.v2[i] * out.a_r[j]
                + moments.v2[j] * out.a_r[i]
                + s2 * out.a_r[i] * out.a_r[j])
                * inv;
        }
    }
    let a_qqq = (g_qqq + s3 * a_q * a_q * a_q + three * s2 * a_q * a_qq) * inv;
    for i in 0..k {
        out.a_qqr[i] = ((moments.v3[i] + s3 * out.a_r[i]) * a_q * a_q
            + two * s2 * a_q * out.a_qr[i]
            + (moments.v2[i] + s2 * out.a_r[i]) * a_qq)
            * inv;
    }
    for i in 0..k {
        for j in 0..k {
            out.a_qrr[i * k + j] = ((moments.m3[i * k + j] + moments.v3[i] * out.a_r[j]) * a_q
                + moments.v2[i] * out.a_qr[j]
                + (moments.v3[j] + s3 * out.a_r[j]) * a_q * out.a_r[i]
                + s2 * (out.a_qr[j] * out.a_r[i] + a_q * out.a_rr[i * k + j])
                + (moments.v2[j] + s2 * out.a_r[j]) * out.a_qr[i])
                * inv;
        }
    }
    out.alpha = alpha;
    out.a_q = a_q;
    out.a_qq = a_qq;
    out.a_qqq = a_qqq;
    Ok(())
}

/// Buffers for one row of the anchored vector program, sized once per law.
pub(crate) struct JointAnchorRowWorkspace {
    score_dim: usize,
    nodes: Box<[f64]>,
    drive: Box<[f64]>,
    standardized: Box<[f64]>,
    moments: JointAnchorMoments,
    entry: JointAnchorDerivatives,
    exit: JointAnchorDerivatives,
    /// `∂feature/∂primary` over [`JOINT_ACTIVE_FEATURES`], row-major.
    jacobian: Box<[f64]>,
    derivative_cells: Box<[f64]>,
}

impl JointAnchorRowWorkspace {
    pub(crate) fn new(law: &JointLatentLawRuntime) -> Self {
        let k = law.score_dim();
        let m = law.node_count();
        let dimension = 3 + k;
        Self {
            score_dim: k,
            nodes: vec![0.0; m * k].into_boxed_slice(),
            drive: vec![0.0; m].into_boxed_slice(),
            standardized: vec![0.0; m].into_boxed_slice(),
            moments: JointAnchorMoments::new(k),
            entry: JointAnchorDerivatives::new(k),
            exit: JointAnchorDerivatives::new(k),
            jacobian: vec![0.0; JOINT_ACTIVE_FEATURES.len() * dimension].into_boxed_slice(),
            derivative_cells: vec![0.0; dimension + dimension * dimension].into_boxed_slice(),
        }
    }

    pub(crate) fn derivatives(&self) -> (ArrayView1<'_, f64>, ArrayView2<'_, f64>) {
        let dimension = 3 + self.score_dim;
        let (gradient, hessian) = self.derivative_cells.split_at(dimension);
        (
            ArrayView1::from(gradient),
            ArrayView2::from_shape((dimension, dimension), hessian)
                .expect("joint anchor row derivative buffer shape is invariant"),
        )
    }
}

/// The five features the anchored vector frame reaches, in pullback order.
const JOINT_ACTIVE_FEATURES: [usize; 5] = [
    FEATURE_Q0,
    FEATURE_Q1,
    FEATURE_QD1,
    FEATURE_LINEAR0,
    FEATURE_LINEAR1,
];

/// Value, primary gradient and primary Hessian of one row of the per-score
/// survival marginal-slope program anchored on a joint latent law, over the
/// primaries `(q₀, q₁, q̇₁, g₁ … g_K)`.
///
/// The features are those of [`AnchoredStaticSlopeGeometry`] with the scalar
/// slope promoted to a vector: `α(q₀, r)`, `α(q₁, r)`, `α_q(q₁, r)·q̇₁`, and the
/// observed linear channel `s_f·gᵀz` twice; the likelihood is the one
/// `row_program!` declaration, and the second-order pullback carries the
/// anchors' curvature in `(q, r)` and the rate feature's, which is a third
/// derivative of `α`.
pub(crate) fn row_primary_anchored_vector_into(
    row: usize,
    q0: f64,
    q1: f64,
    qd1: f64,
    slopes: &[f64],
    z: &[f64],
    w: f64,
    w_entry: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
    law: &JointLatentLawRuntime,
    workspace: &mut JointAnchorRowWorkspace,
) -> Result<f64, String> {
    let k = law.score_dim();
    if slopes.len() != k || z.len() != k || workspace.score_dim != k {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope joint anchored row dimension mismatch: slopes={}, z={}, \
                 law K={k}, workspace K={}",
                slopes.len(),
                z.len(),
                workspace.score_dim
            ),
        }
        .into());
    }
    if z.iter().any(|value| !value.is_finite()) || slopes.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival marginal-slope vector scores and slopes must be finite".to_string(),
        }
        .into());
    }
    if !(probit_scale.is_finite() && probit_scale > 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!(
                "survival marginal-slope joint anchored row needs a finite positive probit scale, \
                 got {probit_scale}"
            ),
        }
        .into());
    }
    let JointAnchorRowWorkspace {
        score_dim: _,
        nodes,
        drive,
        standardized,
        moments,
        entry,
        exit,
        jacobian,
        derivative_cells,
    } = workspace;
    law.row_nodes_into(row, nodes)?;
    let s = probit_scale;
    let mut linear_dot = 0.0;
    for i in 0..k {
        linear_dot += slopes[i] * z[i];
    }
    let linear = s * linear_dot;
    for m in 0..law.node_count() {
        let mut value = 0.0;
        for i in 0..k {
            value += slopes[i] * nodes[m * k + i];
        }
        drive[m] = s * value;
    }
    joint_anchor_into(
        q0,
        nodes,
        drive,
        &law.weights,
        &law.log_weights,
        standardized,
        moments,
        entry,
    )?;
    joint_anchor_into(
        q1,
        nodes,
        drive,
        &law.weights,
        &law.log_weights,
        standardized,
        moments,
        exit,
    )?;
    let features = [
        entry.alpha,
        exit.alpha,
        exit.a_q * qd1,
        linear,
        linear,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let (value, feature_gradient, feature_hessian, [neg_eta0, neg_eta1, adjusted_derivative]) =
        rigid_feature_frame_order2(
            &features,
            w,
            w_entry,
            d,
            s,
            follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(),
        );
    let inputs = RigidRowInputs {
        row,
        wi: w,
        wi_entry: w_entry,
        di: d,
        z_sum: 0.0,
        covariance_ones: 0.0,
        probit_scale: s,
        qd1_lower: derivative_guard,
        anchor: None,
    };
    validate_rigid_row_admission::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
        qd1,
        &inputs,
        neg_eta0,
        neg_eta1,
        adjusted_derivative,
    )?;

    anchored_pullback(
        entry,
        exit,
        qd1,
        z,
        s,
        &feature_gradient,
        &feature_hessian,
        jacobian,
        derivative_cells,
    );
    Ok(value)
}

/// Pull the anchored vector frame's feature gradient and Hessian back to the
/// primaries `(q₀, q₁, q̇₁, g₁ … g_K)`: `g = Jᵀ∇F`, `H = Jᵀ∇²F J + Σ_f F_f ∇²f`.
///
/// In `f64` this is the order-two lowering. In [`Rate`], with the anchors, the
/// rate channel and the feature derivatives carrying their derivatives along
/// one direction, the `t` part of the Hessian is the contracted third
/// `Σ_c ℓ_abc δ_c`.
pub(crate) fn anchored_pullback<S: AnchorScalar>(
    entry: &JointAnchorDerivatives<S>,
    exit: &JointAnchorDerivatives<S>,
    qd1: S,
    z: &[f64],
    probit_scale: f64,
    feature_gradient: &[S; RIGID_FEATURE_DIMENSION],
    feature_hessian: &[[S; RIGID_FEATURE_DIMENSION]; RIGID_FEATURE_DIMENSION],
    jacobian: &mut [S],
    derivative_cells: &mut [S],
) {
    let k = entry.a_r.len();
    let dimension = 3 + k;
    let zero = S::real(0.0);
    let s = S::real(probit_scale);
    // Rows follow `JOINT_ACTIVE_FEATURES`: Q0, Q1, QD1, L0, L1.
    jacobian.fill(zero);
    jacobian[0] = entry.a_q;
    jacobian[dimension + 1] = exit.a_q;
    jacobian[2 * dimension + 1] = exit.a_qq * qd1;
    jacobian[2 * dimension + 2] = exit.a_q;
    for i in 0..k {
        jacobian[3 + i] = s * entry.a_r[i];
        jacobian[dimension + 3 + i] = s * exit.a_r[i];
        jacobian[2 * dimension + 3 + i] = s * exit.a_qr[i] * qd1;
        let linear = S::real(probit_scale * z[i]);
        jacobian[3 * dimension + 3 + i] = linear;
        jacobian[4 * dimension + 3 + i] = linear;
    }
    let (gradient, hessian) = derivative_cells.split_at_mut(dimension);
    for axis in 0..dimension {
        let mut channel = zero;
        for (slot, &feature) in JOINT_ACTIVE_FEATURES.iter().enumerate() {
            channel += feature_gradient[feature] * jacobian[slot * dimension + axis];
        }
        gradient[axis] = channel;
    }
    for left in 0..dimension {
        for right in left..dimension {
            let mut channel = zero;
            for (left_slot, &left_feature) in JOINT_ACTIVE_FEATURES.iter().enumerate() {
                let left_value = jacobian[left_slot * dimension + left];
                if left_value.is_zero() {
                    continue;
                }
                for (right_slot, &right_feature) in JOINT_ACTIVE_FEATURES.iter().enumerate() {
                    channel += left_value
                        * feature_hessian[left_feature][right_feature]
                        * jacobian[right_slot * dimension + right];
                }
            }
            hessian[left * dimension + right] = channel;
            hessian[right * dimension + left] = channel;
        }
    }
    // Σ_f g_f·∂²f/∂p∂p: the entry and exit anchors, and the rate feature.
    let s2 = s * s;
    let mut add = |left: usize, right: usize, value: S| {
        hessian[left * dimension + right] += value;
        if left != right {
            hessian[right * dimension + left] += value;
        }
    };
    let g0 = feature_gradient[FEATURE_Q0];
    let g1 = feature_gradient[FEATURE_Q1];
    let gd = feature_gradient[FEATURE_QD1];
    add(0, 0, g0 * entry.a_qq);
    add(1, 1, g1 * exit.a_qq + gd * exit.a_qqq * qd1);
    add(1, 2, gd * exit.a_qq);
    for i in 0..k {
        add(0, 3 + i, g0 * s * entry.a_qr[i]);
        add(1, 3 + i, g1 * s * exit.a_qr[i] + gd * s * exit.a_qqr[i] * qd1);
        add(2, 3 + i, gd * s * exit.a_qr[i]);
        for j in 0..=i {
            add(
                3 + i,
                3 + j,
                s2 * (g0 * entry.a_rr[i * k + j]
                    + g1 * exit.a_rr[i * k + j]
                    + gd * exit.a_qrr[i * k + j] * qd1),
            );
        }
    }
}

/// The saved model's index and its time rate at one `(row, t)` cell:
/// `η = α(q, r) + rᵀz` and `η′ = α_q(q, r)·q̇`, with `r = s_f·g` on the row's
/// transported joint law. The same anchor the fit's row program solves, and no
/// likelihood around it.
pub(crate) fn joint_anchored_index_and_rate(
    row: usize,
    q: f64,
    qd: f64,
    slopes: &[f64],
    z: &[f64],
    probit_scale: f64,
    law: &JointLatentLawRuntime,
    workspace: &mut JointAnchorRowWorkspace,
) -> Result<(f64, f64), String> {
    let k = law.score_dim();
    if slopes.len() != k || z.len() != k || workspace.score_dim != k {
        return Err(format!(
            "survival marginal-slope joint anchored prediction dimension mismatch: slopes={}, \
             z={}, law K={k}",
            slopes.len(),
            z.len()
        ));
    }
    if !(q.is_finite() && qd.is_finite() && probit_scale.is_finite() && probit_scale > 0.0)
        || slopes.iter().chain(z.iter()).any(|value| !value.is_finite())
    {
        return Err(format!(
            "survival marginal-slope joint anchored prediction needs finite inputs at row {row}: \
             q={q}, q'={qd}, probit scale={probit_scale}"
        ));
    }
    law.row_nodes_into(row, &mut workspace.nodes)?;
    let mut linear_dot = 0.0;
    for i in 0..k {
        linear_dot += slopes[i] * z[i];
    }
    for m in 0..law.node_count() {
        let mut value = 0.0;
        for i in 0..k {
            value += slopes[i] * workspace.nodes[m * k + i];
        }
        workspace.drive[m] = probit_scale * value;
    }
    joint_anchor_into(
        q,
        &workspace.nodes,
        &workspace.drive,
        &law.weights,
        &law.log_weights,
        &mut workspace.standardized,
        &mut workspace.moments,
        &mut workspace.exit,
    )?;
    Ok((
        workspace.exit.alpha + probit_scale * linear_dot,
        workspace.exit.a_q * qd,
    ))
}

/// The anchor of the drive law `{d_m, w_m}` at `q`: the root of
/// `Σ_m w_m Φ(−(α + d_m)) = Φ(−q)`, solved on the standardised drive whose
/// Gaussian seed is the drive closed form.
pub(crate) fn joint_anchor_root(
    q: f64,
    drive: &[f64],
    weights: &[f64],
    log_weights: &[f64],
    standardized: &mut [f64],
) -> Result<f64, String> {
    let m_count = drive.len();
    if weights.len() != m_count || log_weights.len() != m_count || standardized.len() != m_count {
        return Err(format!(
            "survival marginal-slope joint anchor root buffers disagree: drive={m_count}, weights={}",
            weights.len()
        ));
    }
    let mean: f64 = drive.iter().zip(weights).map(|(d, w)| w * d).sum();
    let spread = drive
        .iter()
        .zip(weights)
        .map(|(d, w)| w * (d - mean) * (d - mean))
        .sum::<f64>()
        .sqrt();
    let alpha = if spread > 0.0 && spread.is_finite() {
        for m in 0..m_count {
            standardized[m] = (drive[m] - mean) / spread;
        }
        solve_anchor(
            q,
            spread,
            AnchorGrid {
                nodes: standardized,
                weights,
                log_weights,
            },
        )? - mean
    } else {
        q - mean
    };
    if !alpha.is_finite() {
        return Err(format!(
            "survival marginal-slope joint anchor is not finite at q={q}"
        ));
    }
    Ok(alpha)
}

// ── The directional derivative of the per-score row Hessian ─────────────────

/// The law one per-score row integrates against.
pub(crate) enum PerScoreRowLaw<'a> {
    /// The Gaussian closed form at the row's covariance `Σ(a_row)`.
    ClosedForm { covariance: ArrayView2<'a, f64> },
    /// The row's transported joint law, its nodes (`M × K`) and log-weights,
    /// with both anchors already solved on the real values.
    Anchored {
        nodes: &'a [f64],
        log_weights: &'a [f64],
        alpha_entry: f64,
        alpha_exit: f64,
    },
}

/// Buffers for [`per_score_row_hessian_derivative_into`], sized once per lane.
pub(crate) struct PerScoreHessianDerivativeWorkspace {
    score_dim: usize,
    drive: Box<[f64]>,
    drive_rate: Box<[Rate]>,
    moments: JointAnchorMoments,
    moments_rate: JointAnchorMoments<Rate>,
    entry: JointAnchorDerivatives,
    exit: JointAnchorDerivatives,
    entry_rate: JointAnchorDerivatives<Rate>,
    exit_rate: JointAnchorDerivatives<Rate>,
    linear_direction: Box<[Rate]>,
    variance_direction: Box<[Rate]>,
    jacobian: Box<[Rate]>,
    derivative_cells: Box<[Rate]>,
    third: Box<[f64]>,
}

impl PerScoreHessianDerivativeWorkspace {
    pub(crate) fn new(score_dim: usize, node_count: usize) -> Self {
        let dimension = 3 + score_dim;
        let zero = Rate::real(0.0);
        Self {
            score_dim,
            drive: vec![0.0; node_count].into_boxed_slice(),
            drive_rate: vec![zero; node_count].into_boxed_slice(),
            moments: JointAnchorMoments::new(score_dim),
            moments_rate: JointAnchorMoments::new(score_dim),
            entry: JointAnchorDerivatives::new(score_dim),
            exit: JointAnchorDerivatives::new(score_dim),
            entry_rate: JointAnchorDerivatives::new(score_dim),
            exit_rate: JointAnchorDerivatives::new(score_dim),
            linear_direction: vec![zero; score_dim].into_boxed_slice(),
            variance_direction: vec![zero; score_dim].into_boxed_slice(),
            jacobian: vec![zero; JOINT_ACTIVE_FEATURES.len() * dimension].into_boxed_slice(),
            derivative_cells: vec![zero; dimension + dimension * dimension].into_boxed_slice(),
            third: vec![0.0; dimension * dimension].into_boxed_slice(),
        }
    }

    /// `Σ_c ℓ_abc δ_c` of the last row evaluated, `(3 + K) × (3 + K)`.
    pub(crate) fn third(&self) -> ArrayView2<'_, f64> {
        let dimension = 3 + self.score_dim;
        ArrayView2::from_shape((dimension, dimension), &self.third)
            .expect("per-score Hessian derivative buffer shape is invariant")
    }
}

/// `D_p H[δ]` of one per-score row: the derivative along `δ` of the row NLL's
/// Hessian over the primaries `(q₀, q₁, q̇₁, g₁ … g_K)`, which is the contracted
/// third `Σ_c ℓ_abc δ_c`, written to [`PerScoreHessianDerivativeWorkspace::third`].
///
/// Every quantity the order-two pullback reads is formed in [`Rate`]: the
/// features and the anchor's implicit derivatives carry their derivative along
/// `δ`, the feature gradient carries `∇²_f ℓ·δf`, and the feature Hessian carries
/// the nine-feature program's contracted third along `δf`. The pullback the
/// order-two row lowers then returns the Hessian together with its directional
/// derivative. The width is a runtime quantity: a row costs `O(M·K²)` on a joint
/// law and `O(K²)` in closed form, for any number of scores.
pub(crate) fn per_score_row_hessian_derivative_into(
    row: usize,
    primaries: &[f64],
    direction: &[f64],
    z: &[f64],
    w: f64,
    w_entry: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
    law: PerScoreRowLaw<'_>,
    workspace: &mut PerScoreHessianDerivativeWorkspace,
) -> Result<(), String> {
    let k = workspace.score_dim;
    let dimension = 3 + k;
    if primaries.len() != dimension || direction.len() != dimension || z.len() != k {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope per-score Hessian derivative row {row} has {} primaries, \
                 {} direction entries and {} scores for K={k}",
                primaries.len(),
                direction.len(),
                z.len()
            ),
        }
        .into());
    }
    let PerScoreHessianDerivativeWorkspace {
        score_dim: _,
        drive,
        drive_rate,
        moments,
        moments_rate,
        entry,
        exit,
        entry_rate,
        exit_rate,
        linear_direction,
        variance_direction,
        jacobian,
        derivative_cells,
        third,
    } = workspace;
    let s = probit_scale;
    let slopes = &primaries[3..];
    let slope_direction = &direction[3..];
    let mut linear_value = 0.0;
    let mut linear_rate = 0.0;
    for i in 0..k {
        linear_value += slopes[i] * z[i];
        linear_rate += slope_direction[i] * z[i];
    }
    let linear = Rate {
        v: s * linear_value,
        t: s * linear_rate,
    };
    let qd1 = Rate {
        v: primaries[2],
        t: direction[2],
    };
    let (features, feature_direction, closed_form_covariance) = match law {
        PerScoreRowLaw::ClosedForm { covariance } => {
            if covariance.dim() != (k, k) {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                    reason: format!(
                        "survival marginal-slope per-score closed-form covariance is {:?} for K={k}",
                        covariance.dim()
                    ),
                }
                .into());
            }
            let mut variance = 0.0;
            let mut variance_rate = 0.0;
            for i in 0..k {
                let mut sigma_g = 0.0;
                let mut sigma_delta = 0.0;
                for j in 0..k {
                    sigma_g += covariance[[i, j]] * slopes[j];
                    sigma_delta += covariance[[i, j]] * slope_direction[j];
                }
                variance += slopes[i] * sigma_g;
                variance_rate += 2.0 * slope_direction[i] * sigma_g;
                linear_direction[i] = Rate::real(s * z[i]);
                // ∂V/∂g = 2Σg, the one feature-Jacobian row that moves along δ.
                variance_direction[i] = Rate {
                    v: 2.0 * sigma_g,
                    t: 2.0 * sigma_delta,
                };
            }
            (
                static_slope_feature_frame(
                    primaries[0],
                    primaries[1],
                    primaries[2],
                    linear.v,
                    variance,
                    0.0,
                ),
                static_slope_feature_frame(
                    direction[0],
                    direction[1],
                    direction[2],
                    linear.t,
                    variance_rate,
                    0.0,
                ),
                Some(covariance),
            )
        }
        PerScoreRowLaw::Anchored {
            nodes,
            log_weights,
            alpha_entry,
            alpha_exit,
        } => {
            let m_count = log_weights.len();
            if nodes.len() != m_count * k || drive.len() != m_count {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                    reason: format!(
                        "survival marginal-slope per-score anchored row {row} has {} node values \
                         and {m_count} log-weights for K={k} on a {}-node workspace",
                        nodes.len(),
                        drive.len()
                    ),
                }
                .into());
            }
            for m in 0..m_count {
                let node = &nodes[m * k..(m + 1) * k];
                let mut value = 0.0;
                let mut rate = 0.0;
                for i in 0..k {
                    value += slopes[i] * node[i];
                    rate += slope_direction[i] * node[i];
                }
                drive[m] = s * value;
                drive_rate[m] = Rate {
                    v: s * value,
                    t: s * rate,
                };
            }
            joint_anchor_formulas(
                alpha_entry,
                primaries[0],
                nodes,
                &drive[..],
                log_weights,
                moments,
                entry,
            )?;
            joint_anchor_formulas(
                alpha_exit,
                primaries[1],
                nodes,
                &drive[..],
                log_weights,
                moments,
                exit,
            )?;
            // dα = α_q·δq + α_r·(s·δg): the anchor's own rate along δ, which
            // every implicit derivative below is differentiated through.
            let lift = |anchor: &JointAnchorDerivatives, q_rate: f64| -> Rate {
                let mut rate = anchor.a_q * q_rate;
                for i in 0..k {
                    rate += anchor.a_r[i] * s * slope_direction[i];
                }
                Rate {
                    v: anchor.alpha,
                    t: rate,
                }
            };
            let alpha0 = lift(entry, direction[0]);
            let alpha1 = lift(exit, direction[1]);
            joint_anchor_formulas(
                alpha0,
                Rate {
                    v: primaries[0],
                    t: direction[0],
                },
                nodes,
                &drive_rate[..],
                log_weights,
                moments_rate,
                entry_rate,
            )?;
            joint_anchor_formulas(
                alpha1,
                Rate {
                    v: primaries[1],
                    t: direction[1],
                },
                nodes,
                &drive_rate[..],
                log_weights,
                moments_rate,
                exit_rate,
            )?;
            let rate_feature = exit_rate.a_q * qd1;
            (
                [
                    alpha0.v,
                    alpha1.v,
                    rate_feature.v,
                    linear.v,
                    linear.v,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    alpha0.t,
                    alpha1.t,
                    rate_feature.t,
                    linear.t,
                    linear.t,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                None,
            )
        }
    };
    let anchored = closed_form_covariance.is_none();
    let follow_up_varying = if anchored {
        follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>()
    } else {
        follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>()
    };
    let (_, feature_gradient, feature_hessian, [neg_eta0, neg_eta1, adjusted_derivative]) =
        rigid_feature_frame_order2(&features, w, w_entry, d, s, follow_up_varying);
    let inputs = RigidRowInputs {
        row,
        wi: w,
        wi_entry: w_entry,
        di: d,
        z_sum: 0.0,
        covariance_ones: 0.0,
        probit_scale: s,
        qd1_lower: derivative_guard,
        anchor: None,
    };
    if anchored {
        validate_rigid_row_admission::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
            primaries[2],
            &inputs,
            neg_eta0,
            neg_eta1,
            adjusted_derivative,
        )?;
    } else {
        validate_rigid_row_admission::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
            primaries[2],
            &inputs,
            neg_eta0,
            neg_eta1,
            adjusted_derivative,
        )?;
    }
    let feature_third = rigid_feature_frame_third_contracted(
        &features,
        w,
        w_entry,
        d,
        s,
        follow_up_varying,
        &feature_direction,
    );
    let feature_gradient_rate: [Rate; RIGID_FEATURE_DIMENSION] = std::array::from_fn(|a| {
        let mut rate = 0.0;
        for b in 0..RIGID_FEATURE_DIMENSION {
            rate += feature_hessian[a][b] * feature_direction[b];
        }
        Rate {
            v: feature_gradient[a],
            t: rate,
        }
    });
    let feature_hessian_rate: [[Rate; RIGID_FEATURE_DIMENSION]; RIGID_FEATURE_DIMENSION] =
        std::array::from_fn(|a| {
            std::array::from_fn(|b| Rate {
                v: feature_hessian[a][b],
                t: feature_third[a][b],
            })
        });
    match closed_form_covariance {
        Some(covariance) => closed_form_vector_pullback(
            &feature_gradient_rate,
            &feature_hessian_rate,
            &linear_direction[..],
            &variance_direction[..],
            covariance,
            &mut derivative_cells[..],
        ),
        None => anchored_pullback(
            entry_rate,
            exit_rate,
            qd1,
            z,
            s,
            &feature_gradient_rate,
            &feature_hessian_rate,
            &mut jacobian[..],
            &mut derivative_cells[..],
        ),
    }
    for (slot, cell) in third.iter_mut().zip(derivative_cells[dimension..].iter()) {
        *slot = cell.t;
    }
    Ok(())
}

/// Pull the closed-form per-score frame's feature derivatives back to the
/// primaries `(q₀, q₁, q̇₁, g₁ … g_K)` through `(q, s·gᵀz, gᵀΣg)`:
/// `g = Jᵀ∇F`, `H = Jᵀ∇²F J + 2·(F_V0 + F_V1)·Σ` on the score block.
///
/// `linear_direction` is `∂L/∂g = s·z` and `variance_direction` is `∂V/∂g = 2Σg`.
/// In [`Rate`] the latter carries `2Σδg`, the only feature-Jacobian row that
/// moves along a direction, and the Hessian's `t` part is `D_p H[δ]`.
fn closed_form_vector_pullback<S: AnchorScalar>(
    feature_gradient: &[S; RIGID_FEATURE_DIMENSION],
    feature_hessian: &[[S; RIGID_FEATURE_DIMENSION]; RIGID_FEATURE_DIMENSION],
    linear_direction: &[S],
    variance_direction: &[S],
    covariance: ArrayView2<'_, f64>,
    derivative_cells: &mut [S],
) {
    let k = linear_direction.len();
    let dimension = 3 + k;
    let pair = |left: usize, right: usize, other_left: usize, other_right: usize| -> S {
        feature_hessian[left][right]
            + feature_hessian[left][other_right]
            + feature_hessian[other_left][right]
            + feature_hessian[other_left][other_right]
    };
    let gradient_linear = feature_gradient[FEATURE_LINEAR0] + feature_gradient[FEATURE_LINEAR1];
    let gradient_variance =
        feature_gradient[FEATURE_VARIANCE0] + feature_gradient[FEATURE_VARIANCE1];
    let linear_linear = pair(FEATURE_LINEAR0, FEATURE_LINEAR0, FEATURE_LINEAR1, FEATURE_LINEAR1);
    let linear_variance = pair(
        FEATURE_LINEAR0,
        FEATURE_VARIANCE0,
        FEATURE_LINEAR1,
        FEATURE_VARIANCE1,
    );
    let variance_variance = pair(
        FEATURE_VARIANCE0,
        FEATURE_VARIANCE0,
        FEATURE_VARIANCE1,
        FEATURE_VARIANCE1,
    );
    let (gradient, hessian) = derivative_cells.split_at_mut(dimension);
    for identity in 0..3 {
        gradient[identity] = feature_gradient[identity];
        for other in 0..3 {
            hessian[identity * dimension + other] = feature_hessian[identity][other];
        }
    }
    for score in 0..k {
        let primary = 3 + score;
        let linear = linear_direction[score];
        let variance = variance_direction[score];
        gradient[primary] = gradient_linear * linear + gradient_variance * variance;
        for identity in 0..3 {
            let channel = (feature_hessian[identity][FEATURE_LINEAR0]
                + feature_hessian[identity][FEATURE_LINEAR1])
                * linear
                + (feature_hessian[identity][FEATURE_VARIANCE0]
                    + feature_hessian[identity][FEATURE_VARIANCE1])
                    * variance;
            hessian[identity * dimension + primary] = channel;
            hessian[primary * dimension + identity] = channel;
        }
    }
    let curvature = S::real(2.0) * gradient_variance;
    for left in 0..k {
        let to_linear =
            linear_linear * linear_direction[left] + linear_variance * variance_direction[left];
        let to_variance =
            linear_variance * linear_direction[left] + variance_variance * variance_direction[left];
        for right in left..k {
            let channel = to_linear * linear_direction[right]
                + to_variance * variance_direction[right]
                + curvature * S::real(covariance[[left, right]]);
            hessian[(3 + left) * dimension + 3 + right] = channel;
            hessian[(3 + right) * dimension + 3 + left] = channel;
        }
    }
}

impl SurvivalMarginalSlopeFamily {
    /// `D_β H[u]` of the per-score joint Hessian: `Σ_i J_iᵀ T_i[J_i u] J_i`, with
    /// `T_i[δ] = Σ_c ℓ_abc δ_c` the contracted third of the row program in the
    /// primaries `(q₀, q₁, q̇₁, g₁ … g_K)`, closed form or anchored
    /// ([`per_score_row_hessian_derivative_into`]). The Jacobian of the
    /// primaries in `β` is the constant design rows the dense per-score Hessian
    /// lane scatters through, so no design curvature enters.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_per_z(
        &self,
        block_states: &[ParameterBlockState],
        d_beta: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let total = slices.total;
        if d_beta.len() != total {
            return Err(format!(
                "survival marginal-slope per-score Hessian direction has {} entries for {total} coefficients",
                d_beta.len()
            ));
        }
        let k = self.score_dim();
        let dim = 3 + k;
        let beta_time = &block_states[0].beta;
        let probit_scale = self.probit_frailty_scale();
        let law = self.joint_latent_law();
        let pooled_covariance = (law.is_none() && !self.score_covariance.is_conditional())
            .then(|| self.score_covariance.at_row(0).to_dense());
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((total, total));
                let mut slope_workspace = self.slope_row_workspace()?;
                let node_count = law.map_or(0, JointLatentLawRuntime::node_count);
                let mut workspace = PerScoreHessianDerivativeWorkspace::new(k, node_count);
                let mut j = Array2::<f64>::zeros((dim, total));
                let mut contracted = Array2::<f64>::zeros((dim, total));
                let mut primaries = vec![0.0; dim];
                let mut direction = vec![0.0; dim];
                let mut nodes = vec![0.0; node_count * k];
                let mut drive = vec![0.0; node_count];
                let mut standardized = vec![0.0; node_count];
                for row in range {
                    primaries[0] = self.design_entry.dot_row(row, beta_time)
                        + self.offset_entry[row]
                        + block_states[1].eta[row];
                    primaries[1] = self.design_exit.dot_row(row, beta_time)
                        + self.offset_exit[row]
                        + block_states[1].eta[row];
                    primaries[2] = self.design_derivative_exit.dot_row(row, beta_time)
                        + self.derivative_offset_exit[row];
                    self.fill_slope_values_for_row(row, block_states, &mut slope_workspace)?;
                    self.design_entry
                        .row_chunk_into(row..row + 1, j.slice_mut(s![0..1, slices.time.clone()]))
                        .map_err(|e| format!("per-score dH entry row: {e}"))?;
                    self.design_exit
                        .row_chunk_into(row..row + 1, j.slice_mut(s![1..2, slices.time.clone()]))
                        .map_err(|e| format!("per-score dH exit row: {e}"))?;
                    self.design_derivative_exit
                        .row_chunk_into(row..row + 1, j.slice_mut(s![2..3, slices.time.clone()]))
                        .map_err(|e| format!("per-score dH derivative row: {e}"))?;
                    self.marginal_design
                        .row_chunk_into(
                            row..row + 1,
                            j.slice_mut(s![0..1, slices.marginal.clone()]),
                        )
                        .map_err(|e| format!("per-score dH marginal row: {e}"))?;
                    for col in slices.marginal.clone() {
                        j[[1, col]] = j[[0, col]];
                    }
                    let channel_rows = slope_workspace.channel_rows();
                    for coord in 0..k {
                        j.slice_mut(s![3 + coord, slices.slope.clone()])
                            .assign(&channel_rows.row(coord));
                    }
                    let slopes = slope_workspace.values();
                    primaries[3..].copy_from_slice(slopes);
                    for axis in 0..dim {
                        direction[axis] = j.row(axis).dot(d_beta);
                    }
                    let z_row = self.z.row(row);
                    let z = z_row
                        .as_slice()
                        .ok_or_else(|| "per-score dH score row must be contiguous".to_string())?;
                    let row_covariance: Array2<f64>;
                    let row_law = match law {
                        Some(law) => {
                            law.row_nodes_into(row, &mut nodes)?;
                            for m in 0..node_count {
                                let mut value = 0.0;
                                for i in 0..k {
                                    value += slopes[i] * nodes[m * k + i];
                                }
                                drive[m] = probit_scale * value;
                            }
                            let alpha_entry = joint_anchor_root(
                                primaries[0],
                                &drive,
                                &law.weights,
                                &law.log_weights,
                                &mut standardized,
                            )?;
                            let alpha_exit = joint_anchor_root(
                                primaries[1],
                                &drive,
                                &law.weights,
                                &law.log_weights,
                                &mut standardized,
                            )?;
                            PerScoreRowLaw::Anchored {
                                nodes: &nodes,
                                log_weights: &law.log_weights,
                                alpha_entry,
                                alpha_exit,
                            }
                        }
                        None => PerScoreRowLaw::ClosedForm {
                            covariance: match pooled_covariance.as_ref() {
                                Some(covariance) => covariance.view(),
                                None => {
                                    row_covariance = self.score_covariance.at_row(row).to_dense();
                                    row_covariance.view()
                                }
                            },
                        },
                    };
                    per_score_row_hessian_derivative_into(
                        row,
                        &primaries,
                        &direction,
                        z,
                        self.weights[row],
                        self.entry_weight(row),
                        self.event[row],
                        self.derivative_guard,
                        probit_scale,
                        row_law,
                        &mut workspace,
                    )?;
                    let third = workspace.third();
                    // Jᵀ T J through one contracted row per primary: O(P²·p + P·p²).
                    contracted.fill(0.0);
                    for a in 0..dim {
                        for b in 0..dim {
                            let coefficient = third[[a, b]];
                            if coefficient == 0.0 {
                                continue;
                            }
                            for col in 0..total {
                                contracted[[a, col]] += coefficient * j[[b, col]];
                            }
                        }
                    }
                    for a in 0..dim {
                        for ca in 0..total {
                            let left = j[[a, ca]];
                            if left == 0.0 {
                                continue;
                            }
                            for cb in 0..total {
                                acc[[ca, cb]] += left * contracted[[a, cb]];
                            }
                        }
                    }
                }
                Ok(acc)
            },
            |mut left, right| -> Result<Array2<f64>, String> {
                left += &right;
                Ok(left)
            },
        )?;
        Ok(accumulated.unwrap_or_else(|| Array2::<f64>::zeros((total, total))))
    }
}

/// The per-score row workspace a lane holds: the Gaussian closed form, or the
/// anchor of the family's joint latent law.
pub(crate) enum VectorRowWorkspace<'family> {
    ClosedForm(RigidVectorRowWorkspace<'family>),
    Anchored {
        law: &'family JointLatentLawRuntime,
        workspace: JointAnchorRowWorkspace,
    },
}

impl<'family> VectorRowWorkspace<'family> {
    pub(crate) fn for_family(family: &'family SurvivalMarginalSlopeFamily) -> Result<Self, String> {
        Ok(match family.joint_latent_law() {
            Some(law) => Self::Anchored {
                law,
                workspace: JointAnchorRowWorkspace::new(law),
            },
            None => Self::ClosedForm(RigidVectorRowWorkspace::new(&family.score_covariance)?),
        })
    }

    pub(crate) fn evaluate_row(
        &mut self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        slopes: &[f64],
        z: &[f64],
        w: f64,
        w_entry: f64,
        d: f64,
        derivative_guard: f64,
        probit_scale: f64,
    ) -> Result<f64, String> {
        match self {
            Self::ClosedForm(workspace) => row_primary_closed_form_vector_into(
                row,
                q0,
                q1,
                qd1,
                slopes,
                z,
                w,
                w_entry,
                d,
                derivative_guard,
                probit_scale,
                workspace,
            ),
            Self::Anchored { law, workspace } => row_primary_anchored_vector_into(
                row,
                q0,
                q1,
                qd1,
                slopes,
                z,
                w,
                w_entry,
                d,
                derivative_guard,
                probit_scale,
                law,
                workspace,
            ),
        }
    }

    pub(crate) fn derivatives(&self) -> (ArrayView1<'_, f64>, ArrayView2<'_, f64>) {
        match self {
            Self::ClosedForm(workspace) => workspace.derivatives(),
            Self::Anchored { workspace, .. } => workspace.derivatives(),
        }
    }
}

#[cfg(test)]
mod joint_latent_law_tests {
    use super::super::test_support::{gauss_hermite_probabilists, skewed_grid};
    use super::*;
    use gam_math::jet_scalar::{JetScalar, OneSeed};
    use gam_math::nested_dual::JetField;
    use gam_math::probability::{normal_cdf, normal_pdf};

    /// `[Φ(−x), F′, F″, F‴, F⁗]` with `F(x) = Φ(−x)`.
    #[inline]
    fn joint_survival_cdf_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        [
            normal_cdf(-x),
            -pdf,
            x * pdf,
            (1.0 - x * x) * pdf,
            (x * x * x - 3.0 * x) * pdf,
        ]
    }

    /// `[F′, F″, F‴, F⁗, F⁽⁵⁾]` of the same `F`.
    #[inline]
    fn joint_survival_cdf_prime_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        let x2 = x * x;
        [
            -pdf,
            x * pdf,
            (1.0 - x2) * pdf,
            (x2 * x - 3.0 * x) * pdf,
            -(x2 * x2 - 6.0 * x2 + 3.0) * pdf,
        ]
    }

    /// `[φ, φ′, φ″, φ‴, φ⁗]`.
    #[inline]
    fn joint_pdf_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        let x2 = x * x;
        [
            pdf,
            -x * pdf,
            (x2 - 1.0) * pdf,
            -(x2 * x - 3.0 * x) * pdf,
            (x2 * x2 - 6.0 * x2 + 3.0) * pdf,
        ]
    }

    /// `[1/x, −1/x², 2/x³, −6/x⁴, 24/x⁵]`.
    #[inline]
    fn joint_reciprocal_stack(x: f64) -> [f64; 5] {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        [inv, -inv2, 2.0 * inv3, -6.0 * inv3 * inv, 24.0 * inv3 * inv2]
    }

    /// The joint anchor as a jet in whatever `q` and the drive carry: Newton's
    /// iteration in the jet algebra from the exact real root, three steps (exact
    /// through seventh order), with the value pinned to the root. The scalar
    /// anchor's `anchor_jet` with the projected nodes `b·u_m` promoted to jets
    /// `d_m = rᵀu_m`.
    fn joint_anchor_jet<T: JetField>(
        alpha_star: f64,
        q: &T,
        drive: &[T],
        weights: &[f64],
    ) -> T {
        let mut alpha = q.constant_like(alpha_star);
        for _ in 0..3 {
            let mut g = q.compose_unary(joint_survival_cdf_stack(q.value())).neg();
            let mut g_alpha = q.constant_like(0.0);
            for (d, &w) in drive.iter().zip(weights.iter()) {
                let eta = alpha.add(d);
                let eta_value = eta.value();
                g = g.add(&eta.compose_unary(joint_survival_cdf_stack(eta_value)).scale(w));
                g_alpha = g_alpha.add(&eta.compose_unary(joint_survival_cdf_prime_stack(eta_value)).scale(w));
            }
            let step = g.mul(&g_alpha.compose_unary(joint_reciprocal_stack(g_alpha.value())));
            alpha = alpha.sub(&step);
        }
        alpha.with_value(alpha_star)
    }

    /// `∂α/∂q = φ(q) / Σ_m w_m φ(α + d_m)` as a jet, at an anchor jet lifted by
    /// [`joint_anchor_jet`].
    fn joint_anchor_q_derivative_jet<T: JetField>(
        alpha: &T,
        q: &T,
        drive: &[T],
        weights: &[f64],
    ) -> T {
        let phi_q = q.compose_unary(joint_pdf_stack(q.value()));
        let mut density = q.constant_like(0.0);
        for (d, &w) in drive.iter().zip(weights.iter()) {
            let eta = alpha.add(d);
            density = density.add(&eta.compose_unary(joint_pdf_stack(eta.value())).scale(w));
        }
        phi_q.mul(&density.compose_unary(joint_reciprocal_stack(density.value())))
    }

    /// The law one per-score row integrates against, for the jet program.
    enum JetRowLaw<'a> {
        /// The Gaussian closed form at the row's covariance `Σ(a_row)`.
        ClosedForm { covariance: ArrayView2<'a, f64> },
        /// The row's transported joint law, its nodes (`M × K`) and weights, with
        /// both anchors already solved on the real values.
        Anchored {
            nodes: &'a [f64],
            weights: &'a [f64],
            alpha_entry: f64,
            alpha_exit: f64,
        },
    }

    /// The per-score row NLL over the primaries `(q₀, q₁, q̇₁, g₁ … g_K)` in any
    /// jet: the same features the order-two lowerings pull back — the closed form's
    /// `(q, s·gᵀz, gᵀΣg)` or the anchored frame's `(α₀, α₁, α_q·q̇₁, s·gᵀz)` — fed to
    /// the one `row_program!` declaration. With [`OneSeed`] primaries seeded along a
    /// direction this returns the contracted third `Σ_c ℓ_abc δ_c` the moving-
    /// Hessian outer gradient needs.
    fn per_score_row_nll_jet<const P: usize, S: JetScalar<P>>(
        row: usize,
        primaries: &[S; P],
        z: &[f64],
        w: f64,
        d: f64,
        derivative_guard: f64,
        probit_scale: f64,
        law: JetRowLaw<'_>,
        drive: &mut Vec<S>,
    ) -> Result<S, String> {
        let k = P - 3;
        if z.len() != k {
            return Err(format!(
                "survival marginal-slope per-score jet row has {} scores for {k} slope primaries",
                z.len()
            ));
        }
        let s = probit_scale;
        let zero = primaries[0].constant_like(0.0);
        let mut linear = zero;
        for i in 0..k {
            linear = linear.add(&primaries[3 + i].scale(s * z[i]));
        }
        let features: [S; RIGID_FEATURE_DIMENSION] = match law {
            JetRowLaw::ClosedForm { covariance } => {
                let mut variance = zero;
                for i in 0..k {
                    for j in 0..k {
                        let coefficient = covariance[[i, j]];
                        if coefficient != 0.0 {
                            variance =
                                variance.add(&primaries[3 + i].mul(&primaries[3 + j]).scale(coefficient));
                        }
                    }
                }
                static_slope_feature_frame(primaries[0], primaries[1], primaries[2], linear, variance, zero)
            }
            JetRowLaw::Anchored {
                nodes,
                weights,
                alpha_entry,
                alpha_exit,
            } => {
                drive.clear();
                for index in 0..weights.len() {
                    let mut value = zero;
                    for i in 0..k {
                        value = value.add(&primaries[3 + i].scale(s * nodes[index * k + i]));
                    }
                    drive.push(value);
                }
                let alpha0 = joint_anchor_jet(alpha_entry, &primaries[0], drive, weights);
                let alpha1 = joint_anchor_jet(alpha_exit, &primaries[1], drive, weights);
                let rate = joint_anchor_q_derivative_jet(&alpha1, &primaries[1], drive, weights)
                    .mul(&primaries[2]);
                [alpha0, alpha1, rate, linear, linear, zero, zero, zero, zero]
            }
        };
        let (nll, [neg_eta0, neg_eta1, adjusted_derivative]) =
            rigid_feature_frame_program::<P, S>(&features, w, w, d, s, 0.0);
        let inputs = RigidRowInputs {
            row,
            wi: w,
            wi_entry: w,
            di: d,
            z_sum: 0.0,
            covariance_ones: 0.0,
            probit_scale: s,
            qd1_lower: derivative_guard,
            anchor: None,
        };
        validate_rigid_row_admission::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
            primaries[2].value(),
            &inputs,
            neg_eta0,
            neg_eta1,
            adjusted_derivative,
        )?;
        Ok(nll)
    }

    impl SurvivalMarginalSlopeFamily {
        /// The fixed-width `OneSeed` lowering of `D_β H[u]`, `P = 3 + K` primaries
        /// at compile time: the oracle the width-free assembly is gated against.
        fn one_seed_hessian_directional_derivative_per_z<const P: usize>(
            &self,
            block_states: &[ParameterBlockState],
            d_beta: &Array1<f64>,
        ) -> Result<Array2<f64>, String> {
            let slices = block_slices(self, block_states);
            let total = slices.total;
            if d_beta.len() != total {
                return Err(format!(
                    "survival marginal-slope per-score Hessian direction has {} entries for {total} coefficients",
                    d_beta.len()
                ));
            }
            let k = P - 3;
            let beta_time = &block_states[0].beta;
            let probit_scale = self.probit_frailty_scale();
            let law = self.joint_latent_law();
            let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                self.n,
                |range| -> Result<Array2<f64>, String> {
                    let mut acc = Array2::<f64>::zeros((total, total));
                    let mut slope_workspace = self.slope_row_workspace()?;
                    let mut j = Array2::<f64>::zeros((P, total));
                    let mut drive_jets: Vec<OneSeed<P>> = Vec::new();
                    let node_count = law.map_or(0, JointLatentLawRuntime::node_count);
                    let mut nodes = vec![0.0; node_count * k];
                    let mut drive = vec![0.0; node_count];
                    let mut standardized = vec![0.0; node_count];
                    for row in range {
                        let q0 = self.design_entry.dot_row(row, beta_time)
                            + self.offset_entry[row]
                            + block_states[1].eta[row];
                        let q1 = self.design_exit.dot_row(row, beta_time)
                            + self.offset_exit[row]
                            + block_states[1].eta[row];
                        let qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                            + self.derivative_offset_exit[row];
                        self.fill_slope_values_for_row(row, block_states, &mut slope_workspace)?;
                        self.design_entry
                            .row_chunk_into(row..row + 1, j.slice_mut(s![0..1, slices.time.clone()]))
                            .map_err(|e| format!("per-score dH entry row: {e}"))?;
                        self.design_exit
                            .row_chunk_into(row..row + 1, j.slice_mut(s![1..2, slices.time.clone()]))
                            .map_err(|e| format!("per-score dH exit row: {e}"))?;
                        self.design_derivative_exit
                            .row_chunk_into(row..row + 1, j.slice_mut(s![2..3, slices.time.clone()]))
                            .map_err(|e| format!("per-score dH derivative row: {e}"))?;
                        self.marginal_design
                            .row_chunk_into(
                                row..row + 1,
                                j.slice_mut(s![0..1, slices.marginal.clone()]),
                            )
                            .map_err(|e| format!("per-score dH marginal row: {e}"))?;
                        for col in slices.marginal.clone() {
                            j[[1, col]] = j[[0, col]];
                        }
                        let channel_rows = slope_workspace.channel_rows();
                        for coord in 0..k {
                            j.slice_mut(s![3 + coord, slices.slope.clone()])
                                .assign(&channel_rows.row(coord));
                        }
                        let slopes = slope_workspace.values();
                        let z_row = self.z.row(row);
                        let z = z_row
                            .as_slice()
                            .ok_or_else(|| "per-score dH score row must be contiguous".to_string())?;
                        let real: [f64; P] = std::array::from_fn(|axis| match axis {
                            0 => q0,
                            1 => q1,
                            2 => qd1,
                            other => slopes[other - 3],
                        });
                        let primaries: [OneSeed<P>; P] = std::array::from_fn(|axis| {
                            OneSeed::seed_direction(real[axis], axis, j.row(axis).dot(d_beta))
                        });
                        let covariance = if law.is_none() {
                            Some(self.score_covariance.at_row(row).to_dense())
                        } else {
                            None
                        };
                        let row_law = match (law, covariance.as_ref()) {
                            (Some(law), _) => {
                                law.row_nodes_into(row, &mut nodes)?;
                                for m in 0..node_count {
                                    let mut value = 0.0;
                                    for i in 0..k {
                                        value += slopes[i] * nodes[m * k + i];
                                    }
                                    drive[m] = probit_scale * value;
                                }
                                let alpha_entry = joint_anchor_root(
                                    q0,
                                    &drive,
                                    &law.weights,
                                    &law.log_weights,
                                    &mut standardized,
                                )?;
                                let alpha_exit = joint_anchor_root(
                                    q1,
                                    &drive,
                                    &law.weights,
                                    &law.log_weights,
                                    &mut standardized,
                                )?;
                                JetRowLaw::Anchored {
                                    nodes: &nodes,
                                    weights: &law.weights,
                                    alpha_entry,
                                    alpha_exit,
                                }
                            }
                            (None, Some(covariance)) => JetRowLaw::ClosedForm {
                                covariance: covariance.view(),
                            },
                            (None, None) => {
                                return Err(
                                    "per-score dH has neither a joint law nor a row covariance"
                                        .to_string(),
                                );
                            }
                        };
                        let nll = per_score_row_nll_jet::<P, OneSeed<P>>(
                            row,
                            &primaries,
                            z,
                            self.weights[row],
                            self.event[row],
                            self.derivative_guard,
                            probit_scale,
                            row_law,
                            &mut drive_jets,
                        )?;
                        let third = nll.contracted_third();
                        for a in 0..P {
                            for b in 0..P {
                                let coefficient = third[a][b];
                                if coefficient == 0.0 {
                                    continue;
                                }
                                for ca in 0..total {
                                    let left = j[[a, ca]] * coefficient;
                                    if left == 0.0 {
                                        continue;
                                    }
                                    for cb in 0..total {
                                        acc[[ca, cb]] += left * j[[b, cb]];
                                    }
                                }
                            }
                        }
                    }
                    Ok(acc)
                },
                |mut left, right| -> Result<Array2<f64>, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
            Ok(accumulated.unwrap_or_else(|| Array2::<f64>::zeros((total, total))))
        }
    }

    /// A two-dimensional law with genuine dependence and skew: the skewed
    /// scalar grid on the first axis, five nodes on the second, and weights
    /// tilted by `exp(0.3·x·y)`.
    fn skewed_joint_law() -> (Vec<f64>, Vec<f64>) {
        let first = skewed_grid();
        let second_nodes = [-1.5, -0.5, 0.0, 0.7, 1.8];
        let second_weights = [0.15, 0.25, 0.3, 0.2, 0.1];
        let mut nodes = Vec::new();
        let mut weights = Vec::new();
        for (x, wx) in first.nodes.iter().zip(first.weights.iter()) {
            for (y, wy) in second_nodes.iter().zip(second_weights.iter()) {
                nodes.push(*x);
                nodes.push(*y);
                weights.push(wx * wy * (0.3 * x * y).exp());
            }
        }
        let total: f64 = weights.iter().sum();
        for weight in weights.iter_mut() {
            *weight /= total;
        }
        (nodes, weights)
    }

    fn solve_joint(
        q: f64,
        r: &[f64],
        nodes: &[f64],
        weights: &[f64],
    ) -> JointAnchorDerivatives {
        let k = r.len();
        let m = weights.len();
        let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        let drive: Vec<f64> = (0..m)
            .map(|index| (0..k).map(|i| r[i] * nodes[index * k + i]).sum())
            .collect();
        let mut standardized = vec![0.0; m];
        let mut moments = JointAnchorMoments::new(k);
        let mut out = JointAnchorDerivatives::new(k);
        joint_anchor_into(
            q,
            nodes,
            &drive,
            weights,
            &log_weights,
            &mut standardized,
            &mut moments,
            &mut out,
        )
        .expect("joint anchor");
        out
    }

    /// At one score the joint anchor is the scalar anchor, derivative for
    /// derivative.
    #[test]
    fn joint_anchor_is_the_scalar_anchor_at_one_score() {
        let grid = skewed_grid();
        for &q in &[-1.7, -0.3, 0.4, 2.1] {
            for &b in &[-0.9, 0.2, 1.3] {
                let alpha = solve_anchor(q, b, grid.view()).expect("scalar anchor root");
                let scalar =
                    AnchorDerivatives::at(alpha, q, b, grid.view()).expect("scalar anchor");
                let joint = solve_joint(q, &[b], &grid.nodes, &grid.weights);
                let close = |left: f64, right: f64, tol: f64, name: &str| {
                    assert!(
                        (left - right).abs() <= tol * (1.0 + right.abs()),
                        "q={q} b={b} {name}: joint {left} vs scalar {right}"
                    );
                };
                close(joint.alpha, scalar.alpha, 1e-10, "α");
                close(joint.a_q, scalar.a_q, 1e-9, "α_q");
                close(joint.a_r[0], scalar.a_b, 1e-9, "α_b");
                close(joint.a_qq, scalar.a_qq, 1e-8, "α_qq");
                close(joint.a_qr[0], scalar.a_qb, 1e-8, "α_qb");
                close(joint.a_rr[0], scalar.a_bb, 1e-8, "α_bb");
                close(joint.a_qqq, scalar.a_qqq, 1e-7, "α_qqq");
                close(joint.a_qqr[0], scalar.a_qqb, 1e-7, "α_qqb");
                close(joint.a_qrr[0], scalar.a_qbb, 1e-7, "α_qbb");
            }
        }
    }

    /// Every implicit derivative the anchored vector row reads matches a
    /// central difference of the solved joint anchor on a dependent skewed law.
    #[test]
    fn joint_anchor_derivatives_match_finite_differences() {
        let (nodes, weights) = skewed_joint_law();
        let h = 1e-4;
        for &q in &[-1.2, 0.35, 1.8] {
            for r in [[0.6, -0.4], [-0.9, 0.8], [1.3, 0.25]] {
                let base = solve_joint(q, &r, &nodes, &weights);
                let at = |dq: f64, dr: [f64; 2]| {
                    solve_joint(q + dq, &[r[0] + dr[0], r[1] + dr[1]], &nodes, &weights)
                };
                let unit = |i: usize, step: f64| {
                    let mut dr = [0.0, 0.0];
                    dr[i] = step;
                    dr
                };
                let stencil = |f: &dyn Fn(f64) -> f64| {
                    (-f(2.0 * h) + 8.0 * f(h) - 8.0 * f(-h) + f(-2.0 * h)) / (12.0 * h)
                };
                let check = |value: f64, fd: f64, tol: f64, name: &str| {
                    assert!(
                        (value - fd).abs() <= tol * (1.0 + fd.abs()),
                        "q={q} r={r:?} {name}: analytic {value} vs fd {fd}"
                    );
                };
                check(base.a_q, stencil(&|t| at(t, [0.0, 0.0]).alpha), 1e-8, "α_q");
                for i in 0..2 {
                    check(base.a_r[i], stencil(&|t| at(0.0, unit(i, t)).alpha), 1e-8, "α_r");
                }
                let central = |f: &dyn Fn(f64) -> f64| (f(h) - f(-h)) / (2.0 * h);
                check(base.a_qq, central(&|t| at(t, [0.0, 0.0]).a_q), 1e-6, "α_qq");
                check(base.a_qqq, central(&|t| at(t, [0.0, 0.0]).a_qq), 1e-5, "α_qqq");
                for i in 0..2 {
                    check(base.a_qr[i], central(&|t| at(t, [0.0, 0.0]).a_r[i]), 1e-6, "α_qr (in q)");
                    check(base.a_qr[i], central(&|t| at(0.0, unit(i, t)).a_q), 1e-6, "α_qr (in r)");
                    check(base.a_qqr[i], central(&|t| at(0.0, unit(i, t)).a_qq), 1e-5, "α_qqr");
                    for j in 0..2 {
                        check(
                            base.a_rr[i * 2 + j],
                            central(&|t| at(0.0, unit(j, t)).a_r[i]),
                            1e-6,
                            "α_rr",
                        );
                        check(
                            base.a_qrr[i * 2 + j],
                            central(&|t| at(0.0, unit(j, t)).a_qr[i]),
                            1e-5,
                            "α_qrr",
                        );
                    }
                }
            }
        }
    }

    fn persisted_law(
        nodes: &[f64],
        weights: &[f64],
        mean: [f64; 2],
        covariance: [[f64; 2]; 2],
    ) -> SurvivalJointLatentLaw {
        let l00 = covariance[0][0].sqrt();
        let l10 = covariance[1][0] / l00;
        let l11 = (covariance[1][1] - l10 * l10).sqrt();
        SurvivalJointLatentLaw {
            score_dim: 2,
            residual_nodes: nodes.chunks(2).map(<[f64]>::to_vec).collect(),
            weights: weights.to_vec(),
            score_mean: mean.to_vec(),
            pooled_factor: vec![vec![l00, 0.0], vec![l10, l11]],
            conditional: None,
        }
    }

    /// The anchored vector row's primary gradient and Hessian match central
    /// differences of its own value and gradient, censored and event rows alike.
    #[test]
    fn anchored_vector_row_matches_finite_differences() {
        let (nodes, weights) = skewed_joint_law();
        let law = persisted_law(&nodes, &weights, [0.1, -0.2], [[1.0, 0.4], [0.4, 1.3]])
            .runtime(None, 1)
            .expect("runtime");
        let mut workspace = JointAnchorRowWorkspace::new(&law);
        let z = [0.9, -1.3];
        for event in [0.0, 1.0] {
            let evaluate = |p: [f64; 5], workspace: &mut JointAnchorRowWorkspace| {
                let value = row_primary_anchored_vector_into(
                    0,
                    p[0],
                    p[1],
                    p[2],
                    &p[3..5],
                    &z,
                    1.1,
                    1.1,
                    event,
                    1e-8,
                    0.9,
                    &law,
                    workspace,
                )
                .expect("anchored row");
                let (gradient, hessian) = workspace.derivatives();
                (value, gradient.to_owned(), hessian.to_owned())
            };
            let base = [-0.4, 0.3, 0.8, 0.7, -0.45];
            let (_, gradient, hessian) = evaluate(base, &mut workspace);
            let h = 1e-5;
            for axis in 0..5 {
                let shifted = |step: f64| {
                    let mut p = base;
                    p[axis] += step;
                    p
                };
                let mut scratch = JointAnchorRowWorkspace::new(&law);
                let (plus, g_plus, _) = evaluate(shifted(h), &mut scratch);
                let (minus, g_minus, _) = evaluate(shifted(-h), &mut scratch);
                let (plus2, _, _) = evaluate(shifted(2.0 * h), &mut scratch);
                let (minus2, _, _) = evaluate(shifted(-2.0 * h), &mut scratch);
                let fd = (-plus2 + 8.0 * plus - 8.0 * minus + minus2) / (12.0 * h);
                assert!(
                    (gradient[axis] - fd).abs() <= 1e-6 * (1.0 + fd.abs()),
                    "event={event} axis {axis}: gradient {} vs fd {fd}",
                    gradient[axis]
                );
                for other in 0..5 {
                    let fd = (g_plus[other] - g_minus[other]) / (2.0 * h);
                    assert!(
                        (hessian[[other, axis]] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                        "event={event} H[{other},{axis}] {} vs fd {fd}",
                        hessian[[other, axis]]
                    );
                }
            }
        }
    }

    /// Gaussian is the special case: on a product Gauss–Hermite law transported
    /// by `L = chol(Σ)` the anchored vector row IS the closed-form vector row at
    /// that `Σ` — value, gradient and Hessian — to quadrature tolerance.
    #[test]
    fn anchored_vector_row_on_a_gaussian_law_is_the_closed_form() {
        let (axis_nodes, axis_weights) = gauss_hermite_probabilists(33).expect("GH");
        let mut nodes = Vec::new();
        let mut weights = Vec::new();
        for (x, wx) in axis_nodes.iter().zip(axis_weights.iter()) {
            for (y, wy) in axis_nodes.iter().zip(axis_weights.iter()) {
                nodes.push(*x);
                nodes.push(*y);
                weights.push(wx * wy);
            }
        }
        let total: f64 = weights.iter().sum();
        for weight in weights.iter_mut() {
            *weight /= total;
        }
        let covariance = [[1.2, -0.5], [-0.5, 0.9]];
        let law = persisted_law(&nodes, &weights, [0.0, 0.0], covariance)
            .runtime(None, 1)
            .expect("runtime");
        let field = ScoreCovarianceField::pooled(
            MarginalSlopeCovariance::full(Array2::from_shape_fn((2, 2), |(i, j)| covariance[i][j]))
                .expect("covariance"),
        );
        let mut closed = RigidVectorRowWorkspace::new(&field).expect("closed workspace");
        let mut anchored = JointAnchorRowWorkspace::new(&law);
        for event in [0.0, 1.0] {
            for (q0, q1, qd1, slopes, z) in [
                (-0.6, 0.2, 0.7, [0.5, 0.3], [0.4, -0.8]),
                (0.1, 1.1, 1.3, [-0.8, 0.6], [-1.2, 0.5]),
            ] {
                let closed_value = row_primary_closed_form_vector_into(
                    0, q0, q1, qd1, &slopes, &z, 1.0, 1.0, event, 1e-8, 0.95, &mut closed,
                )
                .expect("closed form");
                let anchored_value = row_primary_anchored_vector_into(
                    0, q0, q1, qd1, &slopes, &z, 1.0, 1.0, event, 1e-8, 0.95, &law, &mut anchored,
                )
                .expect("anchored");
                let (closed_gradient, closed_hessian) = closed.derivatives();
                let (anchored_gradient, anchored_hessian) = anchored.derivatives();
                assert!(
                    (closed_value - anchored_value).abs() <= 1e-7 * (1.0 + closed_value.abs()),
                    "event={event}: value closed {closed_value} vs anchored {anchored_value}"
                );
                for a in 0..5 {
                    assert!(
                        (closed_gradient[a] - anchored_gradient[a]).abs()
                            <= 1e-7 * (1.0 + closed_gradient[a].abs()),
                        "event={event}: gradient[{a}] closed {} vs anchored {}",
                        closed_gradient[a],
                        anchored_gradient[a]
                    );
                    for b in 0..5 {
                        assert!(
                            (closed_hessian[[a, b]] - anchored_hessian[[a, b]]).abs()
                                <= 1e-6 * (1.0 + closed_hessian[[a, b]].abs()),
                            "event={event}: H[{a},{b}] closed {} vs anchored {}",
                            closed_hessian[[a, b]],
                            anchored_hessian[[a, b]]
                        );
                    }
                }
            }
        }
    }

    /// The per-score jet program seeded along a direction is the row program
    /// the order-two lanes lower: its base channels are their value, gradient
    /// and Hessian, and its ε-Hessian — the contracted third `Σ_c ℓ_abc δ_c` the
    /// moving-Hessian outer gradient reads — is a central difference of that
    /// Hessian along the direction, on the closed form and on a joint law alike.
    #[test]
    fn per_score_jet_program_returns_the_contracted_third() {
        let (nodes, weights) = skewed_joint_law();
        let covariance = [[1.0, 0.4], [0.4, 1.3]];
        let law = persisted_law(&nodes, &weights, [0.1, -0.2], covariance)
            .runtime(None, 1)
            .expect("runtime");
        let covariance_dense = Array2::from_shape_fn((2, 2), |(i, j)| covariance[i][j]);
        let field = ScoreCovarianceField::pooled(
            MarginalSlopeCovariance::full(covariance_dense.clone()).expect("covariance"),
        );
        let z = [0.9, -1.3];
        let (w, s, guard) = (1.1, 0.9, 1e-8);
        let direction = [0.3, -0.7, 0.2, 0.5, -0.4];
        let base = [-0.4, 0.3, 0.8, 0.7, -0.45];
        for anchored in [false, true] {
            for event in [0.0, 1.0] {
                let order2 = |p: [f64; 5]| -> (f64, Array1<f64>, Array2<f64>) {
                    if anchored {
                        let mut workspace = JointAnchorRowWorkspace::new(&law);
                        let value = row_primary_anchored_vector_into(
                            0, p[0], p[1], p[2], &p[3..5], &z, w, w, event, guard, s, &law,
                            &mut workspace,
                        )
                        .expect("anchored order-two row");
                        let (gradient, hessian) = workspace.derivatives();
                        (value, gradient.to_owned(), hessian.to_owned())
                    } else {
                        let mut workspace =
                            RigidVectorRowWorkspace::new(&field).expect("closed workspace");
                        let value = row_primary_closed_form_vector_into(
                            0, p[0], p[1], p[2], &p[3..5], &z, w, w, event, guard, s, &mut workspace,
                        )
                        .expect("closed-form order-two row");
                        let (gradient, hessian) = workspace.derivatives();
                        (value, gradient.to_owned(), hessian.to_owned())
                    }
                };
                let mut row_nodes = vec![0.0; 2 * law.node_count()];
                law.row_nodes_into(0, &mut row_nodes).expect("row nodes");
                let drive: Vec<f64> = (0..law.node_count())
                    .map(|m| s * (base[3] * row_nodes[2 * m] + base[4] * row_nodes[2 * m + 1]))
                    .collect();
                let mut standardized = vec![0.0; law.node_count()];
                let alpha_entry =
                    joint_anchor_root(base[0], &drive, &law.weights, &law.log_weights, &mut standardized)
                        .expect("entry root");
                let alpha_exit =
                    joint_anchor_root(base[1], &drive, &law.weights, &law.log_weights, &mut standardized)
                        .expect("exit root");
                let row_law = if anchored {
                    JetRowLaw::Anchored {
                        nodes: &row_nodes,
                        weights: &law.weights,
                        alpha_entry,
                        alpha_exit,
                    }
                } else {
                    JetRowLaw::ClosedForm {
                        covariance: covariance_dense.view(),
                    }
                };
                let primaries: [OneSeed<5>; 5] = std::array::from_fn(|axis| {
                    OneSeed::seed_direction(base[axis], axis, direction[axis])
                });
                let mut drive_jets = Vec::new();
                let jet = per_score_row_nll_jet::<5, OneSeed<5>>(
                    0, &primaries, &z, w, event, guard, s, row_law, &mut drive_jets,
                )
                .expect("jet row");
                let (value, gradient, hessian) = order2(base);
                let label = if anchored { "anchored" } else { "closed form" };
                assert!(
                    (jet.base.0.v - value).abs() <= 1e-10 * (1.0 + value.abs()),
                    "{label} event={event}: jet value {} vs order-two {value}",
                    jet.base.0.v
                );
                for a in 0..5 {
                    assert!(
                        (jet.base.0.g[a] - gradient[a]).abs() <= 1e-9 * (1.0 + gradient[a].abs()),
                        "{label} event={event}: jet gradient[{a}] {} vs order-two {}",
                        jet.base.0.g[a],
                        gradient[a]
                    );
                    for b in 0..5 {
                        assert!(
                            (jet.base.0.h[a][b] - hessian[[a, b]]).abs()
                                <= 1e-8 * (1.0 + hessian[[a, b]].abs()),
                            "{label} event={event}: jet H[{a},{b}] {} vs order-two {}",
                            jet.base.0.h[a][b],
                            hessian[[a, b]]
                        );
                    }
                }
                let third = jet.contracted_third();
                let h = 1e-5;
                let shifted = |t: f64| -> [f64; 5] {
                    std::array::from_fn(|axis| base[axis] + t * direction[axis])
                };
                let (_, _, plus) = order2(shifted(h));
                let (_, _, minus) = order2(shifted(-h));
                for a in 0..5 {
                    for b in 0..5 {
                        let fd = (plus[[a, b]] - minus[[a, b]]) / (2.0 * h);
                        assert!(
                            (third[a][b] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                            "{label} event={event}: contracted third[{a},{b}] {} vs fd {fd}",
                            third[a][b]
                        );
                    }
                }
            }
        }
    }

    fn gaussians(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let u1 = splitmix_unit(&mut state);
            let u2 = splitmix_unit(&mut state);
            let radius = (-2.0 * u1.ln()).sqrt();
            out.push(radius * (std::f64::consts::TAU * u2).cos());
            out.push(radius * (std::f64::consts::TAU * u2).sin());
        }
        out.truncate(n);
        out
    }

    /// The builder keeps the residual law's first two moments exactly, the
    /// persisted law replays its own runtime bitwise through a serde round trip,
    /// and a moving conditional correlation reaches the per-row nodes.
    #[test]
    fn builder_keeps_the_residual_moments_and_replays_bitwise() {
        let n = 6_000;
        let x = gaussians(n, 0x2929_01);
        let e0 = gaussians(n, 0x2929_02);
        let e1 = gaussians(n, 0x2929_03);
        let mut scores = Array2::<f64>::zeros((n, 2));
        // The conditioning span as a marginal design presents it: an intercept
        // column and the covariate.
        let mut design = Array2::<f64>::ones((n, 2));
        for row in 0..n {
            let t = x[row].clamp(-2.0, 2.0) / 2.0;
            design[[row, 1]] = t;
            let phi = 0.8 * t;
            scores[[row, 0]] = e0[row];
            scores[[row, 1]] = phi * e0[row] + (1.0 - phi * phi).sqrt() * e1[row];
        }
        let weights = Array1::<f64>::ones(n);
        let pooled = marginal_slope_covariance_from_scores(scores.view(), &weights).expect("pooled");
        let model = crate::bms::ConditionalScoreCovariance::fit(
            scores.view(),
            weights.view(),
            design.view(),
        )
        .expect("conditional fit")
        .expect("a moving correlation must escalate");
        let field =
            ScoreCovarianceField::conditional(pooled, model, design.view()).expect("field");
        let (persisted, runtime) = build_joint_latent_law(
            scores.view(),
            weights.view(),
            &field,
            Some(design.view()),
            DEFAULT_JOINT_LATENT_NODES,
        )
        .expect("joint law");
        assert_eq!(runtime.node_count(), DEFAULT_JOINT_LATENT_NODES);

        // Exact residual moments.
        let factors = materialize_joint_factors(
            2,
            persisted.pooled_factor.iter().flatten().copied().collect(),
            persisted.conditional.as_ref(),
            Some(design.view()),
            n,
        )
        .expect("factors");
        let shell = JointLatentLawRuntime {
            factors,
            ..runtime.clone()
        };
        let mut residuals = vec![0.0; 2 * n];
        for row in 0..n {
            for j in 0..2 {
                residuals[2 * row + j] = scores[[row, j]] - persisted.score_mean[j];
            }
            forward_substitute(shell.factor(row).expect("factor"), 2, &mut residuals[2 * row..2 * row + 2]);
        }
        let (sample_mean, sample_covariance) =
            weighted_moments(&residuals, &vec![1.0; n], 2);
        let flat: Vec<f64> = persisted.residual_nodes.iter().flatten().copied().collect();
        let (node_mean, node_covariance) = weighted_moments(&flat, &persisted.weights, 2);
        for j in 0..2 {
            assert!((sample_mean[j] - node_mean[j]).abs() < 1e-10, "mean {j}");
        }
        for index in 0..4 {
            assert!(
                (sample_covariance[index] - node_covariance[index]).abs() < 1e-10,
                "covariance cell {index}: sample {} vs nodes {}",
                sample_covariance[index],
                node_covariance[index]
            );
        }

        // Bitwise replay through serde.
        let json = serde_json::to_string(&persisted).expect("serialize");
        let loaded: SurvivalJointLatentLaw = serde_json::from_str(&json).expect("deserialize");
        let replayed = loaded.runtime(Some(design.view()), n).expect("replayed runtime");
        let mut fitted_nodes = vec![0.0; 2 * runtime.node_count()];
        let mut replayed_nodes = vec![0.0; 2 * runtime.node_count()];
        for row in [0, n / 3, n - 1] {
            runtime.row_nodes_into(row, &mut fitted_nodes).expect("fitted nodes");
            replayed.row_nodes_into(row, &mut replayed_nodes).expect("replayed nodes");
            for (left, right) in fitted_nodes.iter().zip(replayed_nodes.iter()) {
                assert_eq!(left.to_bits(), right.to_bits(), "row {row} node replay");
            }
        }

        // The correlation of the row's law follows the covariate.
        let correlation_at = |t: f64| {
            let row = (0..n)
                .min_by(|&a, &b| {
                    (design[[a, 1]] - t).abs().total_cmp(&(design[[b, 1]] - t).abs())
                })
                .expect("rows");
            let mut buffer = vec![0.0; 2 * runtime.node_count()];
            runtime.row_nodes_into(row, &mut buffer).expect("nodes");
            let (_, covariance) = weighted_moments(&buffer, &persisted.weights, 2);
            covariance[2] / (covariance[0] * covariance[3]).sqrt()
        };
        let low = correlation_at(-0.9);
        let high = correlation_at(0.9);
        assert!(
            low < -0.4 && high > 0.4,
            "the row law must carry the planted moving correlation; got {low:.3} at t=-0.9 and \
             {high:.3} at t=0.9"
        );
    }

    /// A dependent, skewed score sample in `ℝ^K`:
    /// `z_i = e_i + 0.5·e_{i−1} + 0.3·(e_i² − 1)`.
    fn dependent_scores(n: usize, k: usize, seed: u64) -> Array2<f64> {
        let e = gaussians(n * k, seed);
        Array2::from_shape_fn((n, k), |(row, i)| {
            let own = e[row * k + i];
            let previous = if i == 0 { 0.0 } else { e[row * k + i - 1] };
            own + 0.5 * previous + 0.3 * (own * own - 1.0)
        })
    }

    /// A per-score family over `K` scores on `n` rows: a two-column time basis,
    /// one marginal covariate and an intercept-plus-covariate slope surface per
    /// score, anchored on the joint law of a dependent skewed sample or lowered
    /// in closed form at that sample's pooled covariance. Returns the family, a
    /// coefficient vector `[time, marginal, slope]` and the marginal design.
    fn per_score_family(
        k: usize,
        n: usize,
        anchored: bool,
    ) -> (SurvivalMarginalSlopeFamily, Array1<f64>, Array2<f64>) {
        let z = dependent_scores(n, k, 0x2929_0200 + k as u64);
        let weights = Array1::from_iter((0..n).map(|i| 0.6 + ((i * 13 + 4) % 5) as f64 * 0.1));
        let event =
            Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
        let x: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64).collect();
        let entry =
            Array2::from_shape_fn((n, 2), |(i, c)| if c == 0 { 1.0 } else { -0.8 + 0.3 * x[i] });
        let exit =
            Array2::from_shape_fn((n, 2), |(i, c)| if c == 0 { 1.0 } else { 0.4 + 0.5 * x[i] });
        let derivative =
            Array2::from_shape_fn((n, 2), |(i, c)| if c == 0 { 0.0 } else { 0.2 + 0.1 * x[i] });
        let marginal = Array2::from_shape_fn((n, 1), |(i, _)| x[i]);
        let raw_slope =
            Array2::from_shape_fn((n, 2 * k), |(i, c)| if c % 2 == 0 { 1.0 } else { x[i] });
        let ranges: Vec<std::ops::Range<usize>> = (0..k).map(|i| 2 * i..2 * i + 2).collect();
        let slope_layout = SlopeTopology::per_score(ranges, 2 * k)
            .expect("per-score topology")
            .materialize_identity(DesignMatrix::from(raw_slope), &Array1::zeros(n))
            .expect("per-score slope layout");
        let pooled =
            marginal_slope_covariance_from_scores(z.view(), &weights).expect("pooled covariance");
        let field = ScoreCovarianceField::pooled(pooled);
        let latent_law = anchored.then(|| {
            let (_, runtime) =
                build_joint_latent_law(z.view(), weights.view(), &field, None, 3 * n / 4)
                    .expect("joint law");
            Arc::new(SurvivalLatentLaw::from_joint(
                crate::bms::LatentMeasureKind::StandardNormal,
                runtime,
            ))
        });
        let family = SurvivalMarginalSlopeFamily {
            jeffreys_armed: false,
            latent_law,
            n,
            entry_at_origin: Arc::new(Array1::from_elem(n, false)),
            event: Arc::new(event),
            weights: Arc::new(weights),
            z: Arc::new(z),
            score_covariance: field,
            gaussian_frailty_sd: None,
            family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(entry),
            design_exit: DesignMatrix::from(exit),
            design_derivative_exit: DesignMatrix::from(derivative),
            offset_entry: Arc::new(Array1::from_elem(n, -0.3)),
            offset_exit: Arc::new(Array1::from_elem(n, 0.2)),
            derivative_offset_exit: Arc::new(Array1::from_elem(n, 0.9)),
            marginal_design: DesignMatrix::from(marginal.clone()),
            slope_layout,
            score_warp: None,
            link_dev: None,
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            flex_jet_arenas: new_flex_jet_arena_pool(),
        };
        let mut beta = Array1::<f64>::zeros(3 + 2 * k);
        beta[0] = 0.15;
        beta[1] = 0.6;
        beta[2] = 0.25;
        for score in 0..k {
            beta[3 + 2 * score] = 0.35 - 0.04 * score as f64;
            beta[3 + 2 * score + 1] = 0.1;
        }
        (family, beta, marginal)
    }

    /// Block states `[time, marginal, slope]` at `beta`, the marginal block's
    /// `eta` read through its design.
    fn per_score_states(marginal: &Array2<f64>, beta: &Array1<f64>) -> Vec<ParameterBlockState> {
        let n = marginal.nrows();
        let marginal_beta = beta.slice(s![2..3]).to_owned();
        vec![
            ParameterBlockState {
                beta: beta.slice(s![0..2]).to_owned(),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                eta: marginal.dot(&marginal_beta),
                beta: marginal_beta,
            },
            ParameterBlockState {
                beta: beta.slice(s![3..]).to_owned(),
                eta: Array1::zeros(n),
            },
        ]
    }

    fn per_score_direction(len: usize) -> Array1<f64> {
        Array1::from_shape_fn(len, |c| 0.075 * ((c * 7 + 3) % 5) as f64 - 0.15)
    }

    /// The width-free `D_β H[u]` of the per-score joint Hessian is the
    /// fixed-width `OneSeed` lowering of the same row program at every width the
    /// jets are compiled for, on the closed form and on a joint law.
    #[test]
    fn per_score_hessian_derivative_matches_one_seed_jets_k2_to_k8() {
        for k in 2..=8 {
            for anchored in [false, true] {
                let (family, beta, marginal) = per_score_family(k, 30, anchored);
                let states = per_score_states(&marginal, &beta);
                let direction = per_score_direction(beta.len());
                let assembled = family
                    .exact_newton_joint_hessian_directional_derivative_per_z(&states, &direction)
                    .expect("width-free per-score dH");
                let oracle = match 3 + k {
                    5 => family.one_seed_hessian_directional_derivative_per_z::<5>(&states, &direction),
                    6 => family.one_seed_hessian_directional_derivative_per_z::<6>(&states, &direction),
                    7 => family.one_seed_hessian_directional_derivative_per_z::<7>(&states, &direction),
                    8 => family.one_seed_hessian_directional_derivative_per_z::<8>(&states, &direction),
                    9 => family.one_seed_hessian_directional_derivative_per_z::<9>(&states, &direction),
                    10 => family.one_seed_hessian_directional_derivative_per_z::<10>(&states, &direction),
                    11 => family.one_seed_hessian_directional_derivative_per_z::<11>(&states, &direction),
                    other => panic!("no OneSeed frame at {other} primaries"),
                }
                .expect("OneSeed per-score dH");
                let scale = oracle.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
                let mut worst = 0.0_f64;
                for ((index, &left), &right) in assembled.indexed_iter().zip(oracle.iter()) {
                    let gap = (left - right).abs();
                    worst = worst.max(gap / (1.0 + right.abs()));
                    assert!(
                        gap <= 1e-8 * (1.0 + right.abs()) + 1e-11 * scale,
                        "K={k} anchored={anchored} dH{index:?}: width-free {left} vs OneSeed {right}"
                    );
                }
                eprintln!(
                    "[2929 dH vs OneSeed] K={k} anchored={anchored} max|dH|={scale:.3e} \
                     worst relative gap={worst:.3e}"
                );
            }
        }
    }

    /// At `K = 12`, past every fixed-width frame, the width-free `D_β H[u]` is a
    /// central difference of the dense per-score joint Hessian along `u`.
    #[test]
    fn per_score_hessian_derivative_matches_finite_differences_at_k12() {
        let k = 12;
        for anchored in [false, true] {
            let (family, beta, marginal) = per_score_family(k, 30, anchored);
            let direction = per_score_direction(beta.len());
            let assembled = family
                .exact_newton_joint_hessian_directional_derivative_per_z(
                    &per_score_states(&marginal, &beta),
                    &direction,
                )
                .expect("width-free per-score dH at K=12");
            let h = 1e-5;
            let hessian_at = |step: f64| {
                let shifted = &beta + &(&direction * step);
                family
                    .evaluate_exact_newton_joint_dense_per_z(&per_score_states(&marginal, &shifted))
                    .expect("dense per-score joint Hessian")
                    .2
            };
            let fd = (hessian_at(h) - hessian_at(-h)) / (2.0 * h);
            let scale = fd.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
            let mut worst = 0.0_f64;
            for ((index, &left), &right) in assembled.indexed_iter().zip(fd.iter()) {
                let gap = (left - right).abs();
                worst = worst.max(gap / (1.0 + right.abs()));
                assert!(
                    gap <= 1e-5 * (1.0 + right.abs()),
                    "K=12 anchored={anchored} dH{index:?}: width-free {left} vs central difference {right}"
                );
            }
            eprintln!(
                "[2929 dH vs FD] K=12 anchored={anchored} max|dH|={scale:.3e} worst relative gap={worst:.3e}"
            );
        }
    }

    /// A local-empirical measure on any score of a `K ≥ 2` per-score fit refuses
    /// the joint law by name; standard-normal and global-empirical measures do
    /// not. The measures are read through their persisted serde form.
    #[test]
    fn joint_law_refuses_a_local_empirical_score_measure_2929() {
        use crate::bms::LatentMeasureKind;
        let grid = r#"{"nodes": [-1.0, 1.0], "weights": [0.5, 0.5]}"#;
        let local: LatentMeasureKind = serde_json::from_str(&format!(
            r#"{{"kind": "local-empirical", "feature_cols": [0], "centers": [[0.0], [1.0]], "grids": [{grid}, {grid}], "top_k": 1, "bandwidth": 1.0}}"#
        ))
        .expect("local-empirical measure");
        let global: LatentMeasureKind =
            serde_json::from_str(&format!(r#"{{"kind": "global-empirical", "grid": {grid}}}"#))
                .expect("global-empirical measure");
        assert!(matches!(local, LatentMeasureKind::LocalEmpirical { .. }));
        assert!(matches!(global, LatentMeasureKind::GlobalEmpirical { .. }));
        for k in [2, 3, 12] {
            let mut measures = vec![global.clone(); k];
            measures[0] = LatentMeasureKind::StandardNormal;
            assert!(
                joint_latent_law_measure_refusal(k, &measures).is_none(),
                "K={k}: standard-normal and global-empirical scores anchor on the joint law"
            );
            measures[k - 1] = local.clone();
            let reason = joint_latent_law_measure_refusal(k, &measures).unwrap_or_else(|| {
                panic!("K={k}: a local-empirical score must refuse the joint law")
            });
            assert!(
                reason.starts_with(&format!(
                    "a local-empirical latent law on a score of a per-score slope over K={k} \
                     scores is refused"
                )) && reason.contains("moving mean or shape"),
                "K={k}: unexpected refusal {reason}"
            );
        }
    }

    /// gam#2926: the several-score closed-form certificate is the closed form's
    /// anchoring residual under the joint law. On a product Gauss–Hermite law
    /// transported by the family's own `Σ` the closed form solves the anchoring
    /// equation, so every row's residual vanishes to quadrature tolerance; on a
    /// skewed, shifted law it is the direct sum `Σ_m w_m Φ(−(q·c + s·rᵀu_m)) − Φ(−q)`
    /// at exit and entry, and it does not vanish.
    #[test]
    fn joint_certificate_residual_is_the_closed_form_residual_under_the_joint_law_2926() {
        let n = 30;
        let (family, beta, marginal) = per_score_family(2, n, false);
        let states = per_score_states(&marginal, &beta);
        let dense = family.score_covariance.pooled_covariance().to_dense();
        let covariance = [[dense[[0, 0]], dense[[0, 1]]], [dense[[1, 0]], dense[[1, 1]]]];

        let (axis_nodes, axis_weights) = gauss_hermite_probabilists(41).expect("GH");
        let mut gh_nodes = Vec::new();
        let mut gh_weights = Vec::new();
        for (x, wx) in axis_nodes.iter().zip(axis_weights.iter()) {
            for (y, wy) in axis_nodes.iter().zip(axis_weights.iter()) {
                gh_nodes.push(*x);
                gh_nodes.push(*y);
                gh_weights.push(wx * wy);
            }
        }
        let total: f64 = gh_weights.iter().sum();
        for weight in gh_weights.iter_mut() {
            *weight /= total;
        }
        let gaussian = persisted_law(&gh_nodes, &gh_weights, [0.0, 0.0], covariance)
            .runtime(None, n)
            .expect("Gaussian runtime");
        let mut workspace =
            super::super::calibration::JointCertificateWorkspace::new(&family, &gaussian)
                .expect("certificate workspace");
        for row in 0..n {
            let anchors = family
                .closed_form_joint_certificate_anchors(row, &states, &gaussian, &mut workspace)
                .expect("Gaussian certificate anchors");
            for (residual, sd, scale) in anchors {
                assert!(
                    residual.abs() <= 1e-9,
                    "row {row}: the closed form solves the anchoring equation on a Gaussian law; \
                     residual {residual:.3e}"
                );
                assert!(sd > 0.0 && scale > 0.0 && scale <= 0.25, "row {row}: sd={sd} scale={scale}");
            }
        }

        let (nodes, weights) = skewed_joint_law();
        let mean = [0.1, -0.2];
        let skewed = persisted_law(&nodes, &weights, mean, covariance)
            .runtime(None, n)
            .expect("skewed runtime");
        let mut workspace =
            super::super::calibration::JointCertificateWorkspace::new(&family, &skewed)
                .expect("certificate workspace");
        let mut slopes = family.slope_row_workspace().expect("slope workspace");
        let mut buffer = vec![0.0; 2 * skewed.node_count()];
        let scale = family.probit_frailty_scale();
        let mut largest = 0.0_f64;
        for row in 0..n {
            let anchors = family
                .closed_form_joint_certificate_anchors(row, &states, &skewed, &mut workspace)
                .expect("skewed certificate anchors");
            let values = family.row_dynamic_q_values(row, &states).expect("row q values");
            family
                .fill_slope_values_for_row(row, &states, &mut slopes)
                .expect("row slopes");
            let r = slopes.values();
            skewed.row_nodes_into(row, &mut buffer).expect("row nodes");
            let variance = scale
                * scale
                * (r[0] * r[0] * covariance[0][0]
                    + 2.0 * r[0] * r[1] * covariance[0][1]
                    + r[1] * r[1] * covariance[1][1]);
            for (anchor, q) in anchors.iter().zip([values.q1, values.q0]) {
                let alpha = q * (1.0 + variance).sqrt();
                let direct = buffer
                    .chunks_exact(2)
                    .zip(skewed.weights().iter())
                    .map(|(u, w)| w * normal_cdf(-(alpha + scale * (r[0] * u[0] + r[1] * u[1]))))
                    .sum::<f64>()
                    - normal_cdf(-q);
                assert!(
                    (anchor.0 - direct).abs() <= 1e-12,
                    "row {row}, q={q}: certificate residual {} vs direct {direct}",
                    anchor.0
                );
                assert!(
                    (anchor.2 - normal_cdf(q) * normal_cdf(-q)).abs() <= 1e-15,
                    "row {row}: π(1−π) {}",
                    anchor.2
                );
                largest = largest.max(direct.abs());
            }
        }
        assert!(
            largest > 1e-3,
            "a skewed, shifted joint law must leave the closed form a visible residual; largest \
             {largest:.3e}"
        );
    }
}

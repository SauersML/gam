//! Sphere-tangent behavioral embedding — the geometry layer of the Rung-2
//! two-block manifold-SAE fit (behavior as a jointly-fitted data block).
//!
//! # The map
//!
//! Each token carries, besides its activation `x_i`, a behavioral summary
//! `p_i` — a next-token distribution over a (possibly restricted) token set of
//! size `V`. The statistical-manifold half-density map
//!
//! ```text
//!   q_i = sqrt(p_i),        ‖q_i‖₂ = 1   (since Σ_j p_ij = 1),
//! ```
//!
//! sends each distribution to a point on the unit sphere `S^{V-1}`. On that
//! sphere the ambient Euclidean geometry is *locally the behavioral geometry*:
//! for a small displacement `Δq` in the tangent space,
//!
//! ```text
//!   KL(p ‖ p+dp) ≈ ½ Σ_j dp_j² / p_j = 2 ‖Δq‖²                           (★)
//! ```
//!
//! (using `dq = dp/(2√p)`), so *ordinary least squares distance in `q`-space is
//! nats*. This is the whole reason to fit behavior on the sphere: the same
//! quadratic reconstruction loss the activation block already minimizes measures
//! KL in the behavior block, with no bespoke likelihood.
//!
//! # The tangent chart
//!
//! We linearize the sphere at a single data-derived basepoint `q̄` (the
//! *extrinsic mean*: the normalized Euclidean mean of the rows, a closed-form,
//! deterministic reference — no Karcher iteration). The behavioral coordinate of
//! a row is the tangential component of its chord to `q̄`, expressed in an
//! orthonormal basis `E` (`V × (V-1)`) of the tangent hyperplane `T_{q̄}S =
//! {v : v·q̄ = 0}`, and scaled by `√2` so that, by (★), squared Euclidean length
//! in the coordinate *is* nats:
//!
//! ```text
//!   c_i = Eᵀ q_i,          y_i = √2 · c_i,        ‖y_i‖² = 2‖c_i‖² ≈ KL.
//! ```
//!
//! (`Eᵀ q̄ = 0`, so `Eᵀ(I − q̄q̄ᵀ) = Eᵀ` and the tangential projection is just
//! `Eᵀ q_i`.) The reduction to the `V-1`-dimensional `E`-basis (rather than
//! keeping an ambient `V`-vector with a null direction along `q̄`) makes the
//! behavior decoder identifiable: there is no unfittable radial column.
//!
//! # Exact round-trip
//!
//! Because `{q̄} ∪ columns(E)` is a complete orthonormal basis of `ℝ^V` and
//! `‖q_i‖ = 1`, the radial component is recoverable from the tangent coordinate:
//! `(q̄·q_i)² = 1 − ‖c_i‖²`. On the near hemisphere (`q̄·q_i > 0`, where every
//! non-degenerate behavioral row lands) the decode
//!
//! ```text
//!   q = √(1 − ‖c‖²) · q̄ + E c,        p = q ⊙ q,        c = y/√2,
//! ```
//!
//! inverts the embedding exactly — [`SphereTangentEmbedding::decode`] recovers
//! the original distribution to machine precision. This is what lets a
//! downstream consumer (Rung 3) turn a *decoded* behavior point on a fitted
//! chart back into an honest distribution and measure realized KL.
//!
//! Everything here is a closed-form linear map plus an elementwise square root;
//! no autodiff, no finite differences, no magic constants (the `√2` and the
//! `q = √p` map are the exact geometry of (★), documented above).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A fitted sphere-tangent chart for a behavioral token set: the basepoint `q̄`
/// and an orthonormal tangent basis `E`, with the exact forward (`embed`) and
/// inverse (`decode`) maps between distributions over the `V`-token set and
/// nats-unit tangent coordinates.
///
/// Construction ([`Self::fit`]) is the only place the basepoint is chosen; the
/// same chart then embeds arbitrary further rows ([`Self::embed`]) and decodes
/// arbitrary tangent coordinates ([`Self::decode`]), so a train-time chart round
/// trips out-of-sample behavior consistently.
#[derive(Clone, Debug)]
pub struct SphereTangentEmbedding {
    /// Extrinsic-mean basepoint `q̄` on the unit sphere `S^{V-1}` (length `V`).
    basepoint: Array1<f64>,
    /// Orthonormal tangent basis `E` (`V × (V-1)`); every column is a unit
    /// vector orthogonal to `q̄` and to the other columns.
    tangent_basis: Array2<f64>,
}

impl SphereTangentEmbedding {
    /// Fit the chart from raw behavioral summaries `prob_rows` (`n × V`), each
    /// row a non-negative distribution-like vector over the `V`-token set, and
    /// return the chart together with the nats-unit tangent target `Y`
    /// (`n × (V-1)`).
    ///
    /// Rows need not be pre-normalized: the half-density map divides by the row
    /// sum, so `q_i = √(p_i / Σ_j p_ij)`. A row must be non-negative with a
    /// strictly positive sum (an all-zero row carries no distribution and is a
    /// caller error, surfaced rather than silently imputed). The basepoint is
    /// the normalized Euclidean mean of the `q_i`; it is undefined only if that
    /// mean is the zero vector (antipodally balanced rows), which is likewise
    /// surfaced.
    pub fn fit(prob_rows: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>), String> {
        let (n, v) = prob_rows.dim();
        if n == 0 || v < 2 {
            return Err(format!(
                "SphereTangentEmbedding::fit: need n ≥ 1 rows and V ≥ 2 tokens; got ({n}, {v})"
            ));
        }
        // Rows → unit-sphere half-densities q_i, accumulating the extrinsic mean.
        let mut q = Array2::<f64>::zeros((n, v));
        let mut mean = Array1::<f64>::zeros(v);
        for i in 0..n {
            let row = prob_rows.row(i);
            let mut sum = 0.0_f64;
            for &value in row.iter() {
                if !(value.is_finite() && value >= 0.0) {
                    return Err(format!(
                        "SphereTangentEmbedding::fit: row {i} has a non-finite or negative \
                         probability entry ({value})"
                    ));
                }
                sum += value;
            }
            if !(sum > 0.0) {
                return Err(format!(
                    "SphereTangentEmbedding::fit: row {i} sums to {sum}; a behavioral summary \
                     must have positive mass"
                ));
            }
            let inv_sqrt_sum = 1.0 / sum.sqrt();
            let mut q_row = q.row_mut(i);
            for j in 0..v {
                let qij = prob_rows[[i, j]].sqrt() * inv_sqrt_sum;
                q_row[j] = qij;
                mean[j] += qij;
            }
        }
        let mean_norm = mean.dot(&mean).sqrt();
        if !(mean_norm > 0.0) {
            return Err(
                "SphereTangentEmbedding::fit: the extrinsic mean of the half-densities is the \
                 zero vector (antipodally balanced behavior); no basepoint is defined"
                    .to_string(),
            );
        }
        let basepoint = &mean / mean_norm;
        let tangent_basis = tangent_basis_orthogonal_to(basepoint.view())?;

        // Y = √2 · Q E   (c_i = Eᵀ q_i, then the nats scaling).
        let root_two = std::f64::consts::SQRT_2;
        let mut target = q.dot(&tangent_basis);
        target.mapv_inplace(|value| root_two * value);

        Ok((
            Self {
                basepoint,
                tangent_basis,
            },
            target,
        ))
    }

    /// Token-set size `V`.
    pub fn vocab(&self) -> usize {
        self.basepoint.len()
    }

    /// Behavioral tangent dimension `p_y = V - 1` (the width of the behavior
    /// decoder block `C_k` and of the nats-unit target `Y`).
    pub fn behavior_dim(&self) -> usize {
        self.tangent_basis.ncols()
    }

    /// The basepoint half-density `q̄` (length `V`).
    pub fn basepoint(&self) -> ArrayView1<'_, f64> {
        self.basepoint.view()
    }

    /// Embed further behavioral summaries onto this (already-fitted) chart,
    /// returning their nats-unit tangent coordinates (`m × (V-1)`). Uses the
    /// chart's fixed basepoint/basis, so out-of-sample rows are placed
    /// consistently with the training rows.
    pub fn embed(&self, prob_rows: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let v = self.vocab();
        let (m, v_in) = prob_rows.dim();
        if v_in != v {
            return Err(format!(
                "SphereTangentEmbedding::embed: rows have {v_in} tokens; chart is over {v}"
            ));
        }
        let mut q = Array2::<f64>::zeros((m, v));
        for i in 0..m {
            let row = prob_rows.row(i);
            let mut sum = 0.0_f64;
            for &value in row.iter() {
                if !(value.is_finite() && value >= 0.0) {
                    return Err(format!(
                        "SphereTangentEmbedding::embed: row {i} has a non-finite or negative entry \
                         ({value})"
                    ));
                }
                sum += value;
            }
            if !(sum > 0.0) {
                return Err(format!(
                    "SphereTangentEmbedding::embed: row {i} sums to {sum}"
                ));
            }
            let inv_sqrt_sum = 1.0 / sum.sqrt();
            let mut q_row = q.row_mut(i);
            for j in 0..v {
                q_row[j] = prob_rows[[i, j]].sqrt() * inv_sqrt_sum;
            }
        }
        let root_two = std::f64::consts::SQRT_2;
        let mut coords = q.dot(&self.tangent_basis);
        coords.mapv_inplace(|value| root_two * value);
        Ok(coords)
    }

    /// Decode a nats-unit tangent coordinate `y` (length `V-1`) back to the
    /// half-density `q` on the sphere:
    /// `q = √(1 − ‖c‖²) q̄ + E c` with `c = y/√2`. Exact inverse of the
    /// embedding on the near hemisphere `q̄·q > 0`; for `‖c‖ ≥ 1` (a coordinate
    /// past the hemisphere boundary, which no embedded row produces) the radial
    /// term is clamped to zero so the result stays a finite point on the
    /// equator rather than becoming imaginary.
    pub fn decode_sphere(&self, y: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let py = self.behavior_dim();
        if y.len() != py {
            return Err(format!(
                "SphereTangentEmbedding::decode_sphere: coordinate has length {}; chart tangent \
                 dim is {py}",
                y.len()
            ));
        }
        let inv_root_two = std::f64::consts::FRAC_1_SQRT_2;
        // c = y / √2, then E c (ambient tangent vector).
        let c = &y.to_owned() * inv_root_two;
        let tangent = self.tangent_basis.dot(&c);
        let radial_sq = 1.0 - c.dot(&c);
        let radial = if radial_sq > 0.0 {
            radial_sq.sqrt()
        } else {
            0.0
        };
        let mut q = &tangent + &(&self.basepoint * radial);
        // Guard against round-off drift off the sphere so `p = q⊙q` normalizes.
        let norm = q.dot(&q).sqrt();
        if norm > 0.0 {
            q.mapv_inplace(|value| value / norm);
        }
        Ok(q)
    }

    /// Decode a nats-unit tangent coordinate back to a distribution `p` over the
    /// `V`-token set (`p = q ⊙ q`, which sums to 1 since `‖q‖ = 1`).
    pub fn decode(&self, y: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let q = self.decode_sphere(y)?;
        Ok(q.mapv(|value| value * value))
    }

    /// Local (flat-metric) predicted dose in nats for a tangent displacement
    /// `Δy`: `‖Δy‖²`. By construction of the `√2` scaling this equals the
    /// second-order KL between the two decoded distributions, and it is the
    /// calibration target the unit-speed behavior decoder is fit to reproduce
    /// (a step `Δt` of the latent producing `Δy = (d(ΨC)/dt)·Δt` costs
    /// `‖Δy‖²` nats).
    pub fn predicted_nats(delta_y: ArrayView1<'_, f64>) -> f64 {
        delta_y.dot(&delta_y)
    }

    /// Exact KL divergence `Σ_j p_a[j] · log(p_a[j] / p_b[j])` in nats between
    /// two distributions over the token set. Used to *measure* the realized dose
    /// against [`Self::predicted_nats`]; terms where `p_a[j] = 0` contribute `0`
    /// (the `0·log 0` convention), and a `p_b[j] = 0` against a positive
    /// `p_a[j]` is `+∞` (genuinely infinite divergence), surfaced as such.
    pub fn exact_kl(p_a: ArrayView1<'_, f64>, p_b: ArrayView1<'_, f64>) -> Result<f64, String> {
        if p_a.len() != p_b.len() {
            return Err(format!(
                "SphereTangentEmbedding::exact_kl: length mismatch {} vs {}",
                p_a.len(),
                p_b.len()
            ));
        }
        let mut kl = 0.0_f64;
        for (&a, &b) in p_a.iter().zip(p_b.iter()) {
            if a > 0.0 {
                kl += a * (a / b).ln();
            }
        }
        Ok(kl)
    }

    /// Fisher–Rao geodesic distance between two half-densities on the sphere:
    /// `2·arccos(q_a·q_b)` (the standard convention under which `KL ≈ ½ d_FR²`
    /// locally). Provided as the exact curved behavioral distance a calibration
    /// consumer can compare a chart's arc length against.
    pub fn fisher_rao_distance(q_a: ArrayView1<'_, f64>, q_b: ArrayView1<'_, f64>) -> f64 {
        let dot = q_a.dot(&q_b).clamp(-1.0, 1.0);
        2.0 * dot.acos()
    }
}

/// Build an orthonormal basis `E` (`V × (V-1)`) of the hyperplane orthogonal to
/// the unit vector `axis`, via a single Householder reflector that maps a pivot
/// standard basis vector onto `axis`.
///
/// The reflector `H = I − 2vvᵀ` with `v = (e_p − axis)/‖e_p − axis‖` maps
/// `e_p ↦ axis` and is orthogonal, so `{H e_j : j ≠ p}` are orthonormal and each
/// orthogonal to `H e_p = axis`. The pivot `p = argmax_j |axis_j|` maximizes
/// `‖e_p − axis‖` (it is `≥ √(1 − 1/V) > 0` for a unit vector), so the reflector
/// is always well-conditioned — no near-zero denominator even when `axis` nearly
/// coincides with a coordinate direction.
fn tangent_basis_orthogonal_to(axis: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
    let v = axis.len();
    if v < 2 {
        return Err(format!("tangent_basis_orthogonal_to: need V ≥ 2; got {v}"));
    }
    // Pivot = argmax |axis_j|.
    let mut pivot = 0usize;
    let mut best = axis[0].abs();
    for j in 1..v {
        let a = axis[j].abs();
        if a > best {
            best = a;
            pivot = j;
        }
    }
    // w = e_pivot − axis; normalize to the Householder unit vector.
    let mut w = axis.to_owned();
    w.mapv_inplace(|value| -value);
    w[pivot] += 1.0;
    let w_norm = w.dot(&w).sqrt();
    if !(w_norm > 0.0) {
        // Only possible if axis == e_pivot exactly; then the tangent basis is
        // just the other coordinate axes, so use a zero reflector (H = I).
        w.fill(0.0);
    } else {
        w.mapv_inplace(|value| value / w_norm);
    }
    // Columns H e_j = e_j − 2 w w_j for j ≠ pivot.
    let mut basis = Array2::<f64>::zeros((v, v - 1));
    let mut col = 0usize;
    for j in 0..v {
        if j == pivot {
            continue;
        }
        let two_wj = 2.0 * w[j];
        for i in 0..v {
            let e_ij = if i == j { 1.0 } else { 0.0 };
            basis[[i, col]] = e_ij - two_wj * w[i];
        }
        col += 1;
    }
    Ok(basis)
}

/// The behavioral data block of a Rung-2 two-block manifold-SAE fit: the fitted
/// sphere-tangent chart, the (unscaled, nats-unit) behavior target `Y`
/// (`n × p_y`), the activation/behavior output split, and the REML-selected
/// relative block weight `λ_y` (stored on the log scale).
///
/// # How it plugs into the existing term
///
/// The two-block fit is realized as an **output-space augmentation** of the
/// ordinary `SaeManifoldTerm`: each atom's decoder is widened to
/// `p̃ = p_x + p_y = [B_k | C_k]`, and the fit target is the stack
/// `Z̃ = [Z | √λ_y · Y]` ([`Self::augmented_target`]). Because both output
/// blocks are decoded from the SAME per-row basis `Φ_k(t_ik)` and the SAME gate
/// `a_ik`, the latent coordinate `t` and the routing `a` are shared by
/// construction — the whole arrow-Schur / REML / smoothness / evidence stack
/// then operates on the wider output with no bespoke behavior likelihood. The
/// `√λ_y` scaling makes the single Gaussian reconstruction dispersion `φ̂` play
/// the role of the activation-block noise while the behavior block carries noise
/// `φ̂ / λ_y`; `λ_y = φ_x / φ_y` is exactly the variance ratio REML selects
/// (fixed here in Increment 2; REML-live in Increment 3).
///
/// The block keeps `Y` **unscaled** so the weight can be changed without
/// re-embedding, and so [`Self::split_decoder`] can recover the true behavior
/// decoder `C_k` (un-doing the `√λ_y`) from a fitted augmented decoder.
#[derive(Clone, Debug)]
pub struct BehaviorBlock {
    /// The fitted sphere-tangent chart the behavior target was embedded through.
    pub embedding: SphereTangentEmbedding,
    /// Nats-unit behavior target `Y` (`n × p_y`), **unscaled** by `λ_y`.
    pub target: Array2<f64>,
    /// Activation output width `p_x` — the split point in the augmented output:
    /// columns `[0, p_x)` are activation, `[p_x, p_x + p_y)` are behavior.
    pub activation_dim: usize,
    /// `log(λ_y)`; the relative weight of the behavior block. Fixed in Inc2,
    /// REML-selected in Inc3. `λ_y = 1` (log 0) weights nats-in-behavior equally
    /// with the activation reconstruction's own units.
    pub log_lambda_y: f64,
}

impl BehaviorBlock {
    /// Build the behavior block from raw behavioral summaries `prob_rows`
    /// (`n × V`) and the activation output width `p_x`, at a fixed initial
    /// `log(λ_y)`. Fits the sphere-tangent chart and stores the nats-unit target.
    pub fn fit(
        prob_rows: ArrayView2<'_, f64>,
        activation_dim: usize,
        log_lambda_y: f64,
    ) -> Result<Self, String> {
        if activation_dim == 0 {
            return Err("BehaviorBlock::fit: activation_dim must be positive".into());
        }
        if !log_lambda_y.is_finite() {
            return Err(format!(
                "BehaviorBlock::fit: log_lambda_y must be finite; got {log_lambda_y}"
            ));
        }
        let (embedding, target) = SphereTangentEmbedding::fit(prob_rows)?;
        Ok(Self {
            embedding,
            target,
            activation_dim,
            log_lambda_y,
        })
    }

    /// Behavior tangent width `p_y = V - 1`.
    pub fn behavior_dim(&self) -> usize {
        self.embedding.behavior_dim()
    }

    /// Augmented output width `p̃ = p_x + p_y`.
    pub fn augmented_dim(&self) -> usize {
        self.activation_dim + self.behavior_dim()
    }

    /// The behavior block weight `λ_y = exp(log_lambda_y)`.
    pub fn lambda_y(&self) -> f64 {
        self.log_lambda_y.exp()
    }

    /// `√λ_y`, the per-column scaling applied to the behavior target so a single
    /// shared dispersion realizes the block variance ratio.
    pub fn sqrt_lambda_y(&self) -> f64 {
        (0.5 * self.log_lambda_y).exp()
    }

    /// Stack the activation target `Z` (`n × p_x`) with the `√λ_y`-scaled
    /// behavior target to form the augmented fit target `Z̃ = [Z | √λ_y · Y]`
    /// (`n × p̃`). This is what the two-block term is fit against; a change to
    /// `λ_y` is realized by re-stacking (Inc3 lifts this to an in-loop
    /// output-column weight so `λ_y` moves under REML without re-stacking).
    pub fn augmented_target(&self, activation: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let (n, px) = activation.dim();
        if px != self.activation_dim {
            return Err(format!(
                "BehaviorBlock::augmented_target: activation has {px} columns; block activation_dim \
                 is {}",
                self.activation_dim
            ));
        }
        if self.target.nrows() != n {
            return Err(format!(
                "BehaviorBlock::augmented_target: activation has {n} rows but behavior target has {}",
                self.target.nrows()
            ));
        }
        let py = self.behavior_dim();
        let sqrt_lambda = self.sqrt_lambda_y();
        let mut augmented = Array2::<f64>::zeros((n, px + py));
        for i in 0..n {
            for j in 0..px {
                augmented[[i, j]] = activation[[i, j]];
            }
            for j in 0..py {
                augmented[[i, px + j]] = sqrt_lambda * self.target[[i, j]];
            }
        }
        Ok(augmented)
    }

    /// Split a fitted augmented decoder `B̃_k` (`M × p̃`) into the activation
    /// decoder `B_k` (`M × p_x`) and the **true** behavior decoder `C_k`
    /// (`M × p_y`), un-doing the `√λ_y` scaling so `C_k` decodes directly into
    /// nats-unit behavior tangent coordinates.
    pub fn split_decoder(
        &self,
        augmented_decoder: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let px = self.activation_dim;
        let py = self.behavior_dim();
        let (m, p_tot) = augmented_decoder.dim();
        if p_tot != px + py {
            return Err(format!(
                "BehaviorBlock::split_decoder: decoder has {p_tot} output columns; expected \
                 p_x + p_y = {px} + {py} = {}",
                px + py
            ));
        }
        let inv_sqrt_lambda = 1.0 / self.sqrt_lambda_y();
        let mut b = Array2::<f64>::zeros((m, px));
        let mut c = Array2::<f64>::zeros((m, py));
        for row in 0..m {
            for j in 0..px {
                b[[row, j]] = augmented_decoder[[row, j]];
            }
            for j in 0..py {
                c[[row, j]] = inv_sqrt_lambda * augmented_decoder[[row, px + j]];
            }
        }
        Ok((b, c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// The tangent basis is orthonormal and orthogonal to the axis.
    #[test]
    fn tangent_basis_is_orthonormal_and_orthogonal_to_axis() {
        let mut axis = Array1::<f64>::from(vec![0.3, -0.5, 0.2, 0.7, -0.34]);
        let norm = axis.dot(&axis).sqrt();
        axis.mapv_inplace(|v| v / norm);
        let e = tangent_basis_orthogonal_to(axis.view()).unwrap();
        assert_eq!(e.dim(), (5, 4));
        // Columns ⟂ axis.
        for col in 0..e.ncols() {
            let dot = e.column(col).dot(&axis);
            assert!(dot.abs() < 1e-12, "column {col} not ⟂ axis: {dot}");
        }
        // Orthonormal columns: Eᵀ E = I.
        let gram = e.t().dot(&e);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - expected).abs() < 1e-12,
                    "EᵀE[{i},{j}] = {} != {expected}",
                    gram[[i, j]]
                );
            }
        }
    }

    /// Round-trip: embedding then decoding recovers the original distribution to
    /// machine precision (every row is in the near hemisphere of its own mean).
    #[test]
    fn embed_decode_round_trips_distributions() {
        // A handful of distinct distributions over V = 6 tokens.
        let rows = vec![
            vec![0.4, 0.2, 0.1, 0.1, 0.1, 0.1],
            vec![0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
            vec![0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            vec![0.05, 0.05, 0.6, 0.1, 0.1, 0.1],
        ];
        let n = rows.len();
        let v = rows[0].len();
        let mut p = Array2::<f64>::zeros((n, v));
        for (i, row) in rows.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                p[[i, j]] = value;
            }
        }
        let (chart, y) = SphereTangentEmbedding::fit(p.view()).unwrap();
        assert_eq!(chart.behavior_dim(), v - 1);
        for i in 0..n {
            let decoded = chart.decode(y.row(i)).unwrap();
            for j in 0..v {
                assert!(
                    (decoded[j] - p[[i, j]]).abs() < 1e-10,
                    "row {i} token {j}: decoded {} != original {}",
                    decoded[j],
                    p[[i, j]]
                );
            }
        }
    }

    /// The nats calibration (★): for a small displacement between two nearby
    /// distributions, the flat predicted dose `‖Δy‖²` matches the exact KL to
    /// second order (relative error shrinks quadratically as the step shrinks).
    #[test]
    fn predicted_nats_matches_exact_kl_to_second_order() {
        let base = Array1::from(vec![0.25, 0.25, 0.2, 0.15, 0.15]);
        let v = base.len();
        // Two rows: the base, and the base nudged by ε along a fixed direction.
        let dir = Array1::from(vec![0.1, -0.05, -0.02, -0.02, -0.01]);
        let make = |eps: f64| -> Array2<f64> {
            let mut p = Array2::<f64>::zeros((2, v));
            for j in 0..v {
                p[[0, j]] = base[j];
                p[[1, j]] = base[j] + eps * dir[j];
            }
            p
        };
        let mut prev_rel: Option<f64> = None;
        for &eps in &[0.2_f64, 0.1, 0.05, 0.025] {
            let p = make(eps);
            let (chart, y) = SphereTangentEmbedding::fit(p.view()).unwrap();
            let delta_y = &y.row(1).to_owned() - &y.row(0).to_owned();
            let predicted = SphereTangentEmbedding::predicted_nats(delta_y.view());
            // Measure exact KL between the two decoded distributions (which equal
            // the originals by the round-trip property).
            let p0 = chart.decode(y.row(0)).unwrap();
            let p1 = chart.decode(y.row(1)).unwrap();
            let kl = SphereTangentEmbedding::exact_kl(p1.view(), p0.view()).unwrap();
            let rel = (predicted - kl).abs() / kl.max(1e-12);
            if let Some(prev) = prev_rel {
                // Halving ε must cut the relative discrepancy (second-order term)
                // by roughly 4×; assert it at least strictly decreases with a
                // comfortable margin.
                assert!(
                    rel < prev * 0.6,
                    "relative KL error did not fall second-order: {prev} → {rel} at ε={eps}"
                );
            }
            prev_rel = Some(rel);
        }
    }

    /// Selection-for-mattering, geometry side: a distribution that does not move
    /// off the basepoint has zero tangent coordinate, hence zero behavioral dose
    /// — an activation pattern with no behavioral correlate earns nothing from
    /// the behavior target.
    #[test]
    fn constant_behavior_has_zero_tangent_target() {
        let base = vec![0.3, 0.3, 0.2, 0.2];
        let n = 5;
        let v = base.len();
        let mut p = Array2::<f64>::zeros((n, v));
        for i in 0..n {
            for j in 0..v {
                p[[i, j]] = base[j];
            }
        }
        let (_chart, y) = SphereTangentEmbedding::fit(p.view()).unwrap();
        for value in y.iter() {
            assert!(
                value.abs() < 1e-12,
                "constant behavior gave nonzero target {value}"
            );
        }
    }

    /// Fisher–Rao distance is zero for identical distributions and grows with
    /// separation, and `KL ≈ ½ d_FR²` holds to leading order.
    #[test]
    fn fisher_rao_matches_half_kl_leading_order() {
        let a = Array1::from(vec![0.25_f64, 0.25, 0.25, 0.25]);
        let mut b = Array1::from(vec![0.26_f64, 0.25, 0.25, 0.24]);
        let bsum: f64 = b.iter().sum();
        b.mapv_inplace(|v| v / bsum);
        let qa = a.mapv(|v| v.sqrt());
        let qb = b.mapv(|v| v.sqrt());
        let self_dist = SphereTangentEmbedding::fisher_rao_distance(qa.view(), qa.view());
        assert!(
            self_dist < 1e-9,
            "self F-R distance should be 0, got {self_dist}"
        );
        let d = SphereTangentEmbedding::fisher_rao_distance(qa.view(), qb.view());
        let kl = SphereTangentEmbedding::exact_kl(a.view(), b.view()).unwrap();
        let half_dsq = 0.5 * d * d;
        let rel = (half_dsq - kl).abs() / kl;
        assert!(rel < 0.05, "½ d_FR² = {half_dsq} vs KL = {kl}, rel {rel}");
    }
}

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

/// Project activation rows onto the LayerNorm sphere: map each row `x_i` to the
/// unit direction `u_i = x_i / ‖x_i‖₂`, optionally rescaled to a fixed `radius`
/// (the RMSNorm convention `√p`, so the projected rows sit on the same sphere the
/// downstream LayerNorm reads). Returns the projected rows (`n × p`) and the
/// per-row Euclidean norms (`n`) — the quotiented-out radial scale, kept for
/// diagnostics.
///
/// # Why fit here instead of in flat ℝᵖ
///
/// Post-LayerNorm the residual-stream geometry is a sphere × scale product: the
/// model reads only the *direction* `u_i`, and the per-token norm `‖x_i‖` is a
/// nuisance the LayerNorm discards. A curved atom fitted to reconstruct the flat
/// activation `x_i = ‖x_i‖·u_i` under the Euclidean metric cannot tell the
/// behaviorally-relevant direction from the nuisance radius: the squared-error
/// objective weights each token by `‖x_i‖²`, so high-norm tokens dominate and the
/// fitted closed curve *bends* toward them, absorbing norm variation into the
/// decoder as spurious higher-harmonic curvature — and leaving a residual whose
/// dominant component is the radial (norm-direction) term, which *rotates* with
/// the latent and so violates the additive-isotropic-Gaussian residual the
/// reconstruction certificate is conditional on. Fitting the projected `u_i`
/// (an atom as a submanifold of the LN sphere) is invariant to `‖x_i‖` by
/// construction: the spurious curvature and the radial residual both vanish.
///
/// A row of exact zero norm carries no direction and is a caller error, surfaced
/// rather than silently imputed.
pub fn ln_sphere_project(
    rows: ArrayView2<'_, f64>,
    radius: Option<f64>,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    let (n, p) = rows.dim();
    if n == 0 || p == 0 {
        return Err(format!(
            "ln_sphere_project: need a non-empty (n × p) matrix; got ({n}, {p})"
        ));
    }
    let scale = match radius {
        Some(r) if r.is_finite() && r > 0.0 => r,
        Some(r) => {
            return Err(format!(
                "ln_sphere_project: radius must be finite and positive; got {r}"
            ));
        }
        None => 1.0,
    };
    let mut projected = Array2::<f64>::zeros((n, p));
    let mut norms = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row = rows.row(i);
        let mut sq = 0.0_f64;
        for &value in row.iter() {
            if !value.is_finite() {
                return Err(format!(
                    "ln_sphere_project: row {i} has a non-finite entry ({value})"
                ));
            }
            sq += value * value;
        }
        let norm = sq.sqrt();
        if !(norm > 0.0) {
            return Err(format!(
                "ln_sphere_project: row {i} has zero norm; a direction is undefined"
            ));
        }
        norms[i] = norm;
        let inv = scale / norm;
        let mut out = projected.row_mut(i);
        for j in 0..p {
            out[j] = rows[[i, j]] * inv;
        }
    }
    Ok((projected, norms))
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
    let w_norm = f64::sqrt(w.dot(&w));
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

    /// A copy of this block re-weighted to a new `log(λ_y)`. Because the target
    /// `Y` is stored **unscaled**, only the scalar weight changes — the chart and
    /// the embedded behavior are untouched — so a two-block REML fit can sweep
    /// `λ_y` without ever re-embedding. The new augmented target is recovered by
    /// [`Self::augmented_target`] at the updated weight.
    pub fn with_log_lambda_y(&self, log_lambda_y: f64) -> Result<Self, String> {
        if !log_lambda_y.is_finite() {
            return Err(format!(
                "BehaviorBlock::with_log_lambda_y: log_lambda_y must be finite; got {log_lambda_y}"
            ));
        }
        let mut next = self.clone();
        next.log_lambda_y = log_lambda_y;
        Ok(next)
    }

    /// The profiled two-block REML criterion's dependence on `log(λ_y)` beyond
    /// what the single-block engine sees: the change-of-variables Jacobian
    /// `−(n·p_y/2)·log λ_y` that the `√λ_y` target scaling introduces into the
    /// Gaussian negative-log-marginal-likelihood.
    ///
    /// The engine's REML criterion is computed on the **scaled** augmented target
    /// `[Z | √λ_y·Y]` and so treats all `p̃` output columns as one homoscedastic
    /// block with a single dispersion `φ̂`. But the honest two-block model has
    /// `φ_y = φ_x/λ_y`; writing the behavior likelihood in the scaled variable
    /// `Ỹ = √λ_y·Y` contributes a Jacobian `∏_i (√λ_y)^{p_y} = λ_y^{n p_y/2}`,
    /// whose negative log is this term. **Add it** to the engine criterion (a
    /// quantity that is *minimised*) to obtain the two-block REML criterion whose
    /// minimiser over `log λ_y` is the variance-ratio REML estimate. Without it
    /// the criterion is monotone in `λ_y` — behavior residuals shrink for free in
    /// scaled units — and `λ_y` is unidentifiable.
    pub fn reml_log_lambda_jacobian(&self, n_obs: usize) -> f64 {
        -0.5 * (n_obs as f64) * (self.behavior_dim() as f64) * self.log_lambda_y
    }

    /// One REML variance-ratio update of `log(λ_y)` from an augmented fit residual
    /// `R̃ = fitted − target` (`n × p̃`), expressed in the **scaled** augmented
    /// units the fit saw (columns `[0, p_x)` activation, `[p_x, p̃)` the
    /// `√λ_y`-scaled behavior).
    ///
    /// The profiled two-block REML criterion (engine criterion `+`
    /// [`Self::reml_log_lambda_jacobian`]) is stationary in `log λ_y` at
    ///
    /// ```text
    ///   λ_y = (R_x / p_x) / (R_y / p_y),   R_x = ‖R̃_x‖²,  R_y = ‖R̃_y‖²/λ_y,
    /// ```
    ///
    /// i.e. the ratio of the two blocks' per-output-channel residual variances —
    /// the classical REML estimate of the variance-component ratio `φ_x/φ_y`
    /// under the shared mean structure. (Derivation: with `φ̂ = (R_x + λ R_y)/(n
    /// p̃)`, `d/d log λ [ (n p̃/2) log φ̂ − (n p_y/2) log λ ] = 0` gives
    /// `λ R_y p_x = R_x p_y` — the profiled-likelihood stationary point, no grid
    /// search, no magic constants.) `R̃_y` already carries a factor `√λ_y`, so
    /// `‖R̃_y‖² = λ_y·R_y`; the update reads the current `λ_y` off `self` to undo
    /// it. Returns the new `log(λ_y)`; the caller re-stacks the target at it (via
    /// [`Self::with_log_lambda_y`]) and refits — a block-coordinate (EM-like)
    /// descent on the joint criterion.
    pub fn reml_updated_log_lambda_y(
        &self,
        augmented_residual: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let px = self.activation_dim;
        let py = self.behavior_dim();
        let (_n, p_tot) = augmented_residual.dim();
        if p_tot != px + py {
            return Err(format!(
                "BehaviorBlock::reml_updated_log_lambda_y: residual has {p_tot} columns; expected \
                 p_x + p_y = {px} + {py} = {}",
                px + py
            ));
        }
        let mut rss_x = 0.0_f64;
        let mut rss_y_scaled = 0.0_f64;
        for row in augmented_residual.rows() {
            for j in 0..px {
                let r = row[j];
                rss_x += r * r;
            }
            for j in px..p_tot {
                let r = row[j];
                rss_y_scaled += r * r;
            }
        }
        // Undo the √λ_y scaling to get the unscaled behavior RSS.
        let lambda = self.lambda_y();
        let rss_y = rss_y_scaled / lambda;
        // Per-channel residual variances; the ratio is the REML variance ratio.
        // A behavior block with no residual variance (perfectly reconstructed, or
        // an all-constant behavior with a zero target) carries no information to
        // set the ratio: surface it so the driver holds λ_y rather than diverging.
        if !(rss_y > 0.0) {
            return Err(format!(
                "BehaviorBlock::reml_updated_log_lambda_y: behavior residual sum of squares is \
                 {rss_y} (no behavioral residual variance); λ_y is not identifiable from this fit"
            ));
        }
        let var_x = rss_x / px as f64;
        let var_y = rss_y / py as f64;
        if !(var_x > 0.0) {
            return Err(format!(
                "BehaviorBlock::reml_updated_log_lambda_y: activation residual variance is {var_x}"
            ));
        }
        Ok((var_x / var_y).ln())
    }
}

/// A generic output data block of a **multi**-block REML fit: a named
/// (unscaled) target `Y_ℓ` (`n × p_ℓ`) decoded from the SAME shared latent
/// coordinate as every other block, plus its own REML-selected relative block
/// weight `λ_ℓ` (log scale).
///
/// # Why this exists — the block-generic core
///
/// [`BehaviorBlock`] is the two-block machinery specialized to a behavior
/// target: the augmented fit `Z̃ = [Z | √λ_y·Y]`, the single shared dispersion
/// `φ̂`, and the closed-form variance-ratio `λ_y = (R_x/p_x)/(R_y/p_y)` are
/// derived in `behavior_fit.rs`. **Nothing in that derivation cares that `Y` is
/// behavior** — it uses only the block's *width* `p_ℓ` and its *residual
/// variance*. Generalising to `Z̃ = [Z | √λ_1·Y_1 | … | √λ_{K-1}·Y_{K-1}]` and
/// profiling the `K` dispersions gives, at the joint stationary point,
///
/// ```text
///   λ_ℓ = (R_x / p_x) / (R_ℓ / p_ℓ)   for every block ℓ,
/// ```
///
/// exactly the per-block variance ratio — **decoupled** across blocks even
/// though the profiled criterion couples them through the shared `φ̂` (the
/// coupling cancels: summing the `K-1` stationarity equations forces
/// `φ̂ = R_x/p_x`, and each `λ_ℓ` then reads only its own residual against the
/// anchor). So one shared latent + per-block decoders + a per-block closed-form
/// `λ_ℓ` update is the whole generalization; see
/// [`SaeManifoldTerm::run_multiblock_reml_fit`](crate::manifold::SaeManifoldTerm::run_multiblock_reml_fit).
///
/// # Clients
///
/// * **Curved crosscoder** — each block is the NEXT layer's activations, so one
///   shared latent coordinate is decoded into several layers at once and `λ_ℓ`
///   REML-selects each layer's relevance (see the `curved_crosscoder` example).
/// * **Development-coder** (follow-up) — each block is a later training
///   checkpoint's activations along the *checkpoint* axis.
///
/// The target is kept **unscaled** (like [`BehaviorBlock::target`]) so `λ_ℓ` can
/// move under REML without re-forming `Y_ℓ`, and so a fitted decoder can be
/// returned to honest units via [`Self::split_honest_decoder`] (un-doing `√λ_ℓ`).
#[derive(Clone, Debug)]
pub struct OutputBlock {
    /// A short label for the block (e.g. the layer name), for diagnostics only.
    pub label: String,
    /// Unscaled target `Y_ℓ` (`n × p_ℓ`).
    pub target: Array2<f64>,
    /// `log(λ_ℓ)`; the relative inferential weight of this block. Moved by the
    /// closed-form REML variance-ratio update, never a knob.
    pub log_lambda: f64,
}

impl OutputBlock {
    /// Build a block from a label, an (unscaled) target, and an initial
    /// `log(λ_ℓ)`. The target must be non-empty and `log_lambda` finite.
    pub fn new(
        label: impl Into<String>,
        target: Array2<f64>,
        log_lambda: f64,
    ) -> Result<Self, String> {
        let (n, p) = target.dim();
        if n == 0 || p == 0 {
            return Err(format!(
                "OutputBlock::new: target must be a non-empty (n × p_ℓ) matrix; got ({n}, {p})"
            ));
        }
        if !log_lambda.is_finite() {
            return Err(format!(
                "OutputBlock::new: log_lambda must be finite; got {log_lambda}"
            ));
        }
        Ok(Self {
            label: label.into(),
            target,
            log_lambda,
        })
    }

    /// Block width `p_ℓ` (the number of output columns this block occupies).
    pub fn block_dim(&self) -> usize {
        self.target.ncols()
    }

    /// The block weight `λ_ℓ = exp(log_lambda)`.
    pub fn lambda(&self) -> f64 {
        self.log_lambda.exp()
    }

    /// `√λ_ℓ`, the per-column scaling applied to the target so a single shared
    /// dispersion realizes the block's variance ratio.
    pub fn sqrt_lambda(&self) -> f64 {
        (0.5 * self.log_lambda).exp()
    }

    /// A copy re-weighted to a new `log(λ_ℓ)` (the target is untouched, so a REML
    /// sweep re-weights without re-forming `Y_ℓ`).
    pub fn with_log_lambda(&self, log_lambda: f64) -> Result<Self, String> {
        if !log_lambda.is_finite() {
            return Err(format!(
                "OutputBlock::with_log_lambda: log_lambda must be finite; got {log_lambda}"
            ));
        }
        let mut next = self.clone();
        next.log_lambda = log_lambda;
        Ok(next)
    }

    /// Un-do the `√λ_ℓ` scaling on a fitted decoder slice `C̃_ℓ` (`M × p_ℓ`)
    /// carved from the augmented decoder, returning the **honest-units** decoder
    /// `C_ℓ = C̃_ℓ / √λ_ℓ` that reconstructs this block's target in its own units.
    pub fn split_honest_decoder(&self, scaled_decoder: ArrayView2<'_, f64>) -> Array2<f64> {
        let inv = 1.0 / self.sqrt_lambda();
        scaled_decoder.mapv(|value| inv * value)
    }

    /// One REML variance-ratio update of `log(λ_ℓ)` from the anchor residual sum
    /// of squares `R_x = ‖R̃_x‖²` (over the `p_x` anchor columns) and this
    /// block's **scaled** residual sum of squares `R̃_ℓ = ‖R̃_ℓ‖²` (over its
    /// `p_ℓ` columns, which carry the `√λ_ℓ` factor). Returns the new
    /// `log(λ_ℓ) = log((R_x/p_x)/(R_ℓ/p_ℓ))` with `R_ℓ = R̃_ℓ/λ_ℓ`.
    ///
    /// This is the per-block instance of [`BehaviorBlock::reml_updated_log_lambda_y`]
    /// (identical arithmetic in identical order, so a single-block multi-block fit
    /// reproduces the two-block update bit-for-bit). A block with no residual
    /// variance (`R_ℓ = 0`: perfectly reconstructed, or a constant target) carries
    /// no information to set the ratio and is surfaced as an error so the driver
    /// holds `λ_ℓ` rather than diverging.
    pub fn reml_updated_log_lambda(
        &self,
        rss_x: f64,
        p_x: usize,
        rss_block_scaled: f64,
    ) -> Result<f64, String> {
        let py = self.block_dim();
        // Undo the √λ_ℓ scaling to get the unscaled block RSS.
        let lambda = self.lambda();
        let rss_y = rss_block_scaled / lambda;
        if !(rss_y > 0.0) {
            return Err(format!(
                "OutputBlock::reml_updated_log_lambda: block '{}' residual sum of squares is \
                 {rss_y} (no residual variance); λ_ℓ is not identifiable from this fit",
                self.label
            ));
        }
        let var_x = rss_x / p_x as f64;
        let var_y = rss_y / py as f64;
        if !(var_x > 0.0) {
            return Err(format!(
                "OutputBlock::reml_updated_log_lambda: anchor residual variance is {var_x}"
            ));
        }
        Ok((var_x / var_y).ln())
    }
}

/// Stacked-column offset bookkeeping for a crosscoder target
/// `Z̃ = [Z | √λ_1·Y_1 | … | √λ_{L-1}·Y_{L-1}]` and the block-columned decoders
/// carved out of it.
///
/// # Why this exists — one owner of the offset arithmetic
///
/// Every consumer of the augmented layout — stacking the target
/// ([`stack_augmented_target`]), reading a block's residual sum of squares
/// ([`SaeManifoldTerm::run_multiblock_reml_fit`]'s `augmented_block_rss`), and
/// carving the honest per-layer decoder
/// (`B_k^(ℓ) = C̃_k[:, off_ℓ..off_ℓ+p_ℓ] / √λ_ℓ`,
/// [`SaeManifoldTerm::layer_decoder`]) — recomputed `off_ℓ = p_x + Σ_{m<ℓ} p_m`
/// by hand. This type owns that arithmetic once ([`Self::block_range`],
/// [`Self::total_dim`]) and carries the fitted per-block weight `λ_ℓ` alongside
/// the widths and labels, so a caller reads a layer's decoder in honest units
/// without re-deriving either the offsets or the `√λ_ℓ` unscaling.
///
/// The anchor block `Z` (`p_x` columns) is implicit at `[0, p_x)`; the `L-1`
/// output blocks follow it in order. `block_dims`, `labels`, and
/// `block_log_lambda` are parallel (one entry per output block); the type's
/// constructors are the only way to build one, so the three stay in lock-step.
#[derive(Clone, Debug, PartialEq)]
pub struct CrosscoderLayout {
    /// Anchor width `p_x` (the leading `[0, p_x)` column block).
    p_x: usize,
    /// Per-output-block width `p_ℓ`, in stacked-column order.
    block_dims: Vec<usize>,
    /// Per-output-block label (diagnostics only), parallel to `block_dims`.
    labels: Vec<String>,
    /// Per-output-block fitted `log(λ_ℓ)`, parallel to `block_dims`. The honest
    /// per-layer decoder divides by `√λ_ℓ = exp(½·log λ_ℓ)`.
    block_log_lambda: Vec<f64>,
}

impl CrosscoderLayout {
    /// Build a layout from the anchor width and parallel per-block
    /// `(dim, label, log λ)` vectors. The three block vectors must have equal
    /// length; `p_x` and every block dim must be non-zero; every `log λ_ℓ`
    /// finite. Zero output blocks is valid (an anchor-only / plain layout,
    /// `total_dim() == p_x`).
    pub fn new(
        p_x: usize,
        block_dims: Vec<usize>,
        labels: Vec<String>,
        block_log_lambda: Vec<f64>,
    ) -> Result<Self, String> {
        if p_x == 0 {
            return Err("CrosscoderLayout::new: anchor width p_x must be non-zero".to_string());
        }
        if block_dims.len() != labels.len() || block_dims.len() != block_log_lambda.len() {
            return Err(format!(
                "CrosscoderLayout::new: block_dims ({}), labels ({}), and block_log_lambda ({}) \
                 must have equal length",
                block_dims.len(),
                labels.len(),
                block_log_lambda.len()
            ));
        }
        for (l, &dim) in block_dims.iter().enumerate() {
            if dim == 0 {
                return Err(format!(
                    "CrosscoderLayout::new: block {l} ('{}') has width 0",
                    labels[l]
                ));
            }
        }
        for (l, &ll) in block_log_lambda.iter().enumerate() {
            if !ll.is_finite() {
                return Err(format!(
                    "CrosscoderLayout::new: block {l} ('{}') log λ is {ll} (not finite)",
                    labels[l]
                ));
            }
        }
        Ok(Self {
            p_x,
            block_dims,
            labels,
            block_log_lambda,
        })
    }

    /// Build a layout from the anchor width and the fitted [`OutputBlock`]s (their
    /// widths, labels, and converged `log λ_ℓ`). Infallible: an `OutputBlock` is
    /// already validated to carry a non-zero width and a finite `log λ_ℓ`.
    pub fn from_blocks(p_x: usize, blocks: &[OutputBlock]) -> Self {
        Self {
            p_x,
            block_dims: blocks.iter().map(|b| b.block_dim()).collect(),
            labels: blocks.iter().map(|b| b.label.clone()).collect(),
            block_log_lambda: blocks.iter().map(|b| b.log_lambda).collect(),
        }
    }

    /// Anchor width `p_x` (the leading `[0, p_x)` column block).
    pub fn anchor_dim(&self) -> usize {
        self.p_x
    }

    /// Number of output blocks `L-1` (excludes the anchor).
    pub fn num_blocks(&self) -> usize {
        self.block_dims.len()
    }

    /// Per-output-block widths `p_ℓ`, in stacked-column order.
    pub fn block_dims(&self) -> &[usize] {
        &self.block_dims
    }

    /// Per-output-block labels, parallel to [`Self::block_dims`].
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Per-output-block fitted `log(λ_ℓ)`, parallel to [`Self::block_dims`].
    pub fn block_log_lambda(&self) -> &[f64] {
        &self.block_log_lambda
    }

    /// Total augmented width `p̃ = p_x + Σ_ℓ p_ℓ`.
    pub fn total_dim(&self) -> usize {
        self.p_x + self.block_dims.iter().sum::<usize>()
    }

    /// The half-open column range `[off_ℓ, off_ℓ + p_ℓ)` of output block `ℓ` in
    /// the stacked target / decoder, `off_ℓ = p_x + Σ_{m<ℓ} p_m`.
    ///
    /// # Panics
    /// If `l >= num_blocks()`. Callers that take an untrusted index bounds-check
    /// against [`Self::num_blocks`] first (e.g. [`SaeManifoldTerm::layer_decoder`]).
    pub fn block_range(&self, l: usize) -> std::ops::Range<usize> {
        assert!(
            l < self.block_dims.len(),
            "CrosscoderLayout::block_range: block {l} out of range (L-1 = {})",
            self.block_dims.len()
        );
        let start = self.p_x + self.block_dims[..l].iter().sum::<usize>();
        start..start + self.block_dims[l]
    }

    /// `log(λ_ℓ)` for output block `ℓ`.
    pub fn log_lambda(&self, l: usize) -> f64 {
        self.block_log_lambda[l]
    }

    /// `√λ_ℓ = exp(½·log λ_ℓ)`, the per-column target scaling. Computed exactly as
    /// [`OutputBlock::sqrt_lambda`], so a layout built from the fitted blocks
    /// unscales a decoder bit-for-bit like the by-hand [`OutputBlock::split_honest_decoder`].
    pub fn sqrt_lambda(&self, l: usize) -> f64 {
        (0.5 * self.block_log_lambda[l]).exp()
    }
}

/// Stack an anchor target `Z` (`n × p_x`) with the `√λ_ℓ`-scaled targets of a
/// list of output blocks to form the augmented multi-block fit target
/// `Z̃ = [Z | √λ_1·Y_1 | … | √λ_{K-1}·Y_{K-1}]` (`n × p̃`,
/// `p̃ = p_x + Σ_ℓ p_ℓ`).
///
/// For a single block this is byte-identical to
/// [`BehaviorBlock::augmented_target`] (same per-entry formula, same order), so
/// the multi-block fit reduces to the two-block fit at `K = 2`. The `√λ_ℓ` per
/// column is what lets the single shared reconstruction dispersion `φ̂` play the
/// anchor's noise while block `ℓ` carries noise `φ̂/λ_ℓ`.
pub fn stack_augmented_target(
    anchor: ArrayView2<'_, f64>,
    blocks: &[OutputBlock],
) -> Result<Array2<f64>, String> {
    let (n, px) = anchor.dim();
    if n == 0 || px == 0 {
        return Err(format!(
            "stack_augmented_target: anchor must be a non-empty (n × p_x) matrix; got ({n}, {px})"
        ));
    }
    for block in blocks {
        if block.target.nrows() != n {
            return Err(format!(
                "stack_augmented_target: block '{}' has {} rows but anchor has {n}",
                block.label,
                block.target.nrows()
            ));
        }
    }
    // The column offsets and total width are owned by the layout (no by-hand
    // `off_ℓ` accumulation here). `√λ_ℓ` per block matches the two-block path
    // ([`OutputBlock::sqrt_lambda`]) bit-for-bit.
    let layout = CrosscoderLayout::from_blocks(px, blocks);
    let mut augmented = Array2::<f64>::zeros((n, layout.total_dim()));
    for i in 0..n {
        for j in 0..px {
            augmented[[i, j]] = anchor[[i, j]];
        }
        for (l, block) in blocks.iter().enumerate() {
            let sqrt_lambda = layout.sqrt_lambda(l);
            for (jj, col) in layout.block_range(l).enumerate() {
                augmented[[i, col]] = sqrt_lambda * block.target[[i, jj]];
            }
        }
    }
    Ok(augmented)
}

/// The multi-block profiled REML criterion (the quantity minimised over the
/// block weights), evaluated at a fitted state's UNSCALED residual sums of
/// squares. Up to `log λ`-independent constants it is
///
/// ```text
///   C = (n·p̃/2)·log((R_x + Σ_ℓ λ_ℓ·R_ℓ)/(n·p̃)) − Σ_ℓ (n·p_ℓ/2)·log λ_ℓ ,
/// ```
///
/// (`p̃ = p_x + Σ p_ℓ`, `n = n_obs`), the profiled Gaussian negative-log-marginal
/// plus the `√λ_ℓ` target-scaling Jacobian ([`OutputBlock::reml_updated_log_lambda`]).
/// The `−(n p_ℓ/2)·log λ_ℓ` term diverges to `+∞` as `λ_ℓ → 0`, so the criterion
/// PENALISES a vanishing weight — which is exactly what a plain fixed-point λ
/// update (that treats the residual as frozen) fails to see, letting the shared
/// coordinate trade a down-weighted block away in a positive-feedback runaway.
/// A driver that only accepts a `λ` step when this criterion decreases (Armijo
/// backtracking) is monotone and cannot diverge; its stationary point is the
/// same per-block variance ratio the closed form targets.
///
/// Returns `+∞` for a non-positive pooled residual (an invalid state a caller's
/// line search should reject).
pub fn profiled_reml_criterion(
    n_obs: usize,
    p_x: usize,
    rss_x: f64,
    block_rss_unscaled: &[f64],
    block_dims: &[usize],
    block_log_lambda: &[f64],
) -> f64 {
    let n = n_obs as f64;
    let mut p_tilde = p_x as f64;
    let mut pooled = rss_x;
    let mut jac = 0.0_f64;
    for ((&rss, &dim), &log_lambda) in block_rss_unscaled
        .iter()
        .zip(block_dims.iter())
        .zip(block_log_lambda.iter())
    {
        pooled += log_lambda.exp() * rss;
        p_tilde += dim as f64;
        jac += (dim as f64) * log_lambda;
    }
    if !(pooled > 0.0) {
        return f64::INFINITY;
    }
    0.5 * n * p_tilde * (pooled / (n * p_tilde)).ln() - 0.5 * n * jac
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

    /// The closed-form REML λ_y update is the per-channel residual-variance
    /// ratio, computed from a residual expressed in the SCALED augmented units
    /// (the behavior columns carry √λ_y), and it is the stationary point of the
    /// profiled criterion `(n·p̃/2)·log((R_x + λ·R_y)/(n·p̃)) + jacobian(λ)`.
    #[test]
    fn reml_lambda_update_is_variance_ratio_and_criterion_stationary_point() {
        let n = 7usize;
        let p_x = 3usize;
        let vocab = 4usize; // p_y = 3
        // Any valid block; the update only reads dims and the current λ_y.
        let mut probs = Array2::<f64>::zeros((n, vocab));
        for i in 0..n {
            for j in 0..vocab {
                probs[[i, j]] = 1.0 + ((i * vocab + j) as f64 * 0.618).sin().abs();
            }
        }
        let current_log_lambda = 0.7_f64;
        let block = BehaviorBlock::fit(probs.view(), p_x, current_log_lambda).unwrap();
        let p_y = block.behavior_dim();

        // A synthetic residual with known per-block sums of squares, written in
        // scaled units: behavior columns get an extra √λ_y factor.
        let sqrt_lambda = block.sqrt_lambda_y();
        let mut residual = Array2::<f64>::zeros((n, p_x + p_y));
        let mut rss_x = 0.0_f64;
        let mut rss_y = 0.0_f64; // unscaled
        for i in 0..n {
            for j in 0..p_x {
                let r = 0.3 + 0.1 * (i as f64) - 0.05 * (j as f64);
                residual[[i, j]] = r;
                rss_x += r * r;
            }
            for j in 0..p_y {
                let r = 0.1 - 0.02 * (i as f64) + 0.03 * (j as f64);
                residual[[i, p_x + j]] = sqrt_lambda * r;
                rss_y += r * r;
            }
        }
        let updated = block.reml_updated_log_lambda_y(residual.view()).unwrap();
        let expected = ((rss_x / p_x as f64) / (rss_y / p_y as f64)).ln();
        assert!(
            (updated - expected).abs() < 1e-12,
            "update {updated} != variance ratio {expected}"
        );

        // Stationarity: the profiled criterion C(logλ) = (n·p̃/2)·log(R_x+λR_y)
        // − (n·p_y/2)·logλ (constants dropped) has zero derivative at the
        // update: λR_y·p_x = R_x·p_y ⟺ dC/dlogλ = (np̃/2)·λR_y/(R_x+λR_y) −
        // n·p_y/2 = 0. Verify the identity at the returned point.
        let lambda_hat = updated.exp();
        let p_tot = (p_x + p_y) as f64;
        let dc = 0.5 * (n as f64) * p_tot * lambda_hat * rss_y / (rss_x + lambda_hat * rss_y)
            - 0.5 * (n as f64) * (p_y as f64);
        assert!(
            dc.abs() < 1e-9,
            "criterion derivative at the update is {dc}, not 0"
        );
        // And the Jacobian term itself matches its formula.
        let jac = block.reml_log_lambda_jacobian(n);
        assert!((jac + 0.5 * (n as f64) * (p_y as f64) * current_log_lambda).abs() < 1e-12);

        // A behavior block with an identically-zero behavior residual carries no
        // information about λ_y: surfaced as an error, not a silent divergence.
        let mut inert = residual.clone();
        for i in 0..n {
            for j in p_x..p_x + p_y {
                inert[[i, j]] = 0.0;
            }
        }
        assert!(block.reml_updated_log_lambda_y(inert.view()).is_err());
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

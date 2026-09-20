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
//! in the coordinate is nats at the basepoint. Away from it the chart's Fisher
//! metric adds a radial term ([`SphereTangentEmbedding::predicted_nats`]):
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
/// same chart then embeds arbitrary further rows (`Self::embed`) and decodes
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
    pub(crate) fn behavior_dim(&self) -> usize {
        self.tangent_basis.ncols()
    }

    /// Decode a nats-unit tangent coordinate `y` (length `V-1`) back to the
    /// half-density `q` on the sphere:
    /// `q = √(1 − ‖c‖²) q̄ + E c` with `c = y/√2`. Exact inverse of the
    /// embedding on the near hemisphere `q̄·q > 0`; for `‖c‖ ≥ 1` (a coordinate
    /// past the hemisphere boundary, which no embedded row produces) the radial
    /// term is clamped to zero so the result stays a finite point on the
    /// equator rather than becoming imaginary.
    pub(crate) fn decode_sphere(&self, y: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
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

    /// Decode a row-aligned matrix of nats-unit tangent coordinates back to
    /// probability distributions.  This is the batched public inverse used by
    /// the behavior-fit report; it delegates every row to [`Self::decode`] so
    /// the scalar and batched hemisphere/normalization contracts cannot drift.
    pub(crate) fn decode_rows(&self, y: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if y.ncols() != self.behavior_dim() {
            return Err(format!(
                "SphereTangentEmbedding::decode_rows: coordinates have {} columns; chart tangent dim is {}",
                y.ncols(),
                self.behavior_dim()
            ));
        }
        let mut probabilities = Array2::<f64>::zeros((y.nrows(), self.vocab()));
        for row in 0..y.nrows() {
            let decoded = self.decode(y.row(row))?;
            probabilities.row_mut(row).assign(&decoded);
        }
        Ok(probabilities)
    }

    /// Predicted dose in nats of a tangent displacement `Δy` taken at chart point
    /// `y`: `Δyᵀ G(y) Δy`, with the chart's pulled-back Fisher metric
    /// `G(y) = I + y yᵀ/(2 − ‖y‖²)`.
    ///
    /// The decode `q = √(1 − ‖c‖²) q̄ + E c`, `c = y/√2`, moves `q` by
    /// `dq = E dc − (cᵀdc/√(1 − ‖c‖²)) q̄`, so `‖dq‖² = ‖dc‖² + (cᵀdc)²/(1 − ‖c‖²)`
    /// and by (★) `KL(p ‖ p + dp) = 2‖dq‖² + O(‖dq‖³) = ‖Δy‖² + (yᵀΔy)²/(2 − ‖y‖²)`.
    /// At the basepoint (`y = 0`) this is exactly `‖Δy‖²`. Away from it the flat
    /// `‖Δy‖²` alone under-prices a radial step by `cos²θ`, where `θ` is the
    /// point's angle from `q̄`. The metric diverges at the hemisphere boundary
    /// `‖y‖² = 2`, past which the chart represents no distribution, so such a `y`
    /// is refused. A latent step `Δt` producing `Δy = (d(ΨC)/dt)·Δt` at `y = ΨC`
    /// costs this dose.
    pub fn predicted_nats(
        y: ArrayView1<'_, f64>,
        delta_y: ArrayView1<'_, f64>,
    ) -> Result<f64, String> {
        if y.len() != delta_y.len() {
            return Err(format!(
                "SphereTangentEmbedding::predicted_nats: the point has length {} but the \
                 displacement has length {}",
                y.len(),
                delta_y.len()
            ));
        }
        let norm_sq = y.dot(&y);
        if !(norm_sq < 2.0) {
            return Err(format!(
                "SphereTangentEmbedding::predicted_nats: ‖y‖² = {norm_sq} is not inside the \
                 chart's hemisphere (‖y‖² < 2), so the chart has no finite Fisher metric there"
            ));
        }
        let cross = y.dot(&delta_y);
        Ok(delta_y.dot(&delta_y) + cross * cross / (2.0 - norm_sq))
    }

    /// Exact KL divergence `Σ_j p_a[j] · log(p_a[j] / p_b[j])` in nats between
    /// two distributions over the token set. Used to *measure* the realized dose
    /// against [`Self::predicted_nats`]; terms where `p_a[j] = 0` contribute `0`
    /// (the `0·log 0` convention), and a `p_b[j] = 0` against a positive
    /// `p_a[j]` is `+∞` (genuinely infinite divergence), surfaced as such.
    pub(crate) fn exact_kl(p_a: ArrayView1<'_, f64>, p_b: ArrayView1<'_, f64>) -> Result<f64, String> {
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
}

/// Build an orthonormal basis `E` (`V × (V-1)`) of the hyperplane orthogonal to
/// the unit vector `axis`, via a single Householder reflector that maps a pivot
/// standard basis vector onto `±axis`.
///
/// The reflector `H = I − 2wwᵀ` with `w = (axis + s·e_p)/‖axis + s·e_p‖`,
/// `s = sign(axis_p)`, maps `e_p ↦ −s·axis` and is orthogonal, so
/// `{H e_j : j ≠ p}` are orthonormal and each orthogonal to `axis`. The sign is
/// the one that ADDS magnitudes in the pivot entry, `axis_p + s = s(|axis_p| + 1)`,
/// so `‖axis + s·e_p‖² = 2(1 + |axis_p|) ≥ 2` and no entry of `w` is formed by
/// cancellation. The opposite sign, `e_p − axis`, subtracts `1 − |axis_p|`, which
/// cancels exactly when `axis` nearly coincides with a coordinate direction — the
/// pivot `p = argmax_j |axis_j|` makes `|axis_p|` as large as possible, so that
/// choice would sit on the cancelling side for every basepoint.
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
    // w = axis + sign(axis_pivot)·e_pivot, normalized: the pivot entry adds
    // magnitudes, so ‖w‖ ≥ √2 for a unit axis and w carries no cancellation.
    let mut w = axis.to_owned();
    w[pivot] += 1.0_f64.copysign(axis[pivot]);
    let w_norm = f64::sqrt(w.dot(&w));
    if !(w_norm.is_finite() && w_norm > 0.0) {
        return Err(format!(
            "tangent_basis_orthogonal_to: axis is not a finite unit vector (‖axis + s·e_p‖ = {w_norm})"
        ));
    }
    w.mapv_inplace(|value| value / w_norm);
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
/// `Z̃ = [Z | √λ_y · Y]`. Because both output
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
    log_lambda_y: f64,
    lambda_y: f64,
    sqrt_lambda_y: f64,
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
        let lambda_y = gam_problem::checked_exp_log_strength(log_lambda_y)
            .map_err(|error| format!("BehaviorBlock::fit: {error}"))?;
        let sqrt_lambda_y = gam_problem::checked_exp_log_strength(0.5 * log_lambda_y)
            .map_err(|error| format!("BehaviorBlock::fit square-root strength: {error}"))?;
        let (embedding, target) = SphereTangentEmbedding::fit(prob_rows)?;
        Ok(Self {
            embedding,
            target,
            activation_dim,
            log_lambda_y,
            lambda_y,
            sqrt_lambda_y,
        })
    }

    /// Behavior tangent width `p_y = V - 1`.
    pub(crate) fn behavior_dim(&self) -> usize {
        self.embedding.behavior_dim()
    }

    /// Augmented output width `p̃ = p_x + p_y`.
    pub(crate) fn augmented_dim(&self) -> usize {
        self.activation_dim + self.behavior_dim()
    }

    /// The behavior block weight `λ_y = exp(log_lambda_y)`.
    pub fn lambda_y(&self) -> f64 {
        self.lambda_y
    }

    /// `√λ_y`, the per-column scaling applied to the behavior target so a single
    /// shared dispersion realizes the block variance ratio.
    pub(crate) fn sqrt_lambda_y(&self) -> f64 {
        self.sqrt_lambda_y
    }

    pub fn log_lambda_y(&self) -> f64 {
        self.log_lambda_y
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
    /// `λ_y` without ever re-embedding.
    pub(crate) fn with_log_lambda_y(&self, log_lambda_y: f64) -> Result<Self, String> {
        let lambda_y = gam_problem::checked_exp_log_strength(log_lambda_y)
            .map_err(|error| format!("BehaviorBlock::with_log_lambda_y: {error}"))?;
        let sqrt_lambda_y =
            gam_problem::checked_exp_log_strength(0.5 * log_lambda_y).map_err(|error| {
                format!("BehaviorBlock::with_log_lambda_y square-root strength: {error}")
            })?;
        let mut next = self.clone();
        next.log_lambda_y = log_lambda_y;
        next.lambda_y = lambda_y;
        next.sqrt_lambda_y = sqrt_lambda_y;
        Ok(next)
    }
}

/// Stacked-column offset bookkeeping for a crosscoder target
/// `Z̃ = [Z | √λ_1·Y_1 | … | √λ_{L-1}·Y_{L-1}]` and the block-columned decoders
/// carved out of it.
///
/// # Why this exists — one owner of the offset arithmetic
///
/// Every consumer of the augmented layout — reading a block's residual sum of
/// squares and carving the honest per-layer decoder
/// (`B_k^(ℓ) = C̃_k[:, off_ℓ..off_ℓ+p_ℓ] / √λ_ℓ`,
/// `SaeManifoldTerm::layer_decoder`) — recomputed `off_ℓ = p_x + Σ_{m<ℓ} p_m`
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
    /// Exact cached `√λ_ℓ`, parallel to `block_log_lambda`.
    block_sqrt_lambda: Vec<f64>,
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
        gam_problem::validate_log_strengths(block_log_lambda.iter().copied()).map_err(|error| {
            format!(
                "CrosscoderLayout::new: block {} ('{}') has invalid log λ: {error}",
                error.coordinate, labels[error.coordinate]
            )
        })?;
        let block_sqrt_lambda = block_log_lambda
            .iter()
            .copied()
            .map(|log_lambda| {
                gam_problem::checked_exp_log_strength(0.5 * log_lambda)
                    .expect("half of a validated log strength remains canonical")
            })
            .collect();
        Ok(Self {
            p_x,
            block_dims,
            labels,
            block_log_lambda,
            block_sqrt_lambda,
        })
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
    /// against [`Self::num_blocks`] first (e.g. `SaeManifoldTerm::layer_decoder`).
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

    /// `√λ_ℓ = exp(½·log λ_ℓ)`, the per-column target scaling, computed once from
    /// the validated `log λ_ℓ` so every consumer unscales by the same value.
    pub fn sqrt_lambda(&self, l: usize) -> f64 {
        self.block_sqrt_lambda[l]
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

    /// A basepoint that nearly coincides with a coordinate direction (every row
    /// concentrated on one token) keeps the tangent basis orthogonal to it at
    /// round-off relative to the axis's small entries, not at `√ε` absolute.
    #[test]
    fn tangent_basis_stays_orthogonal_to_a_near_coordinate_axis() {
        for &small in &[1e-3_f64, 1e-6, 1e-8, 1e-10] {
            let mut axis = Array1::<f64>::from(vec![small, 1.0, 0.5 * small, 0.3 * small]);
            let norm = axis.dot(&axis).sqrt();
            axis.mapv_inplace(|v| v / norm);
            let e = tangent_basis_orthogonal_to(axis.view()).unwrap();
            for col in 0..e.ncols() {
                let dot = e.column(col).dot(&axis);
                assert!(
                    dot.abs() <= 1e-14 * small,
                    "axis minority scale {small}: column {col} has axis component {dot}"
                );
            }
            let gram = e.t().dot(&e);
            for i in 0..e.ncols() {
                for j in 0..e.ncols() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((gram[[i, j]] - expected).abs() < 1e-14);
                }
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
            // The metric is read at row 1, the first argument of the KL below.
            let predicted = SphereTangentEmbedding::predicted_nats(y.row(1), delta_y.view())
                .expect("both rows lie inside the chart's hemisphere");
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

}

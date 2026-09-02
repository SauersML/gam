//! #2 (reviewer condition) — the DISTILLED / AMORTIZED encoder.
//!
//! Our held-out reconstruction quality comes from a per-row test-time
//! optimization (the Kantorovich-certified Newton encode of [`crate::encode`]
//! with an exact multi-start fallback). A sparse-autoencoder's held-out number
//! comes from ONE matmul. A reviewer will (correctly) demand the distilled
//! encoder as the PRIMARY out-of-sample row and the exact solve as an oracle
//! line. This module trains that distilled encoder: a cheap map
//!
//! ```text
//!     x  ↦  (gate logits, per-atom coords, amplitudes)
//! ```
//!
//! that predicts, in one matmul, the same per-row solution the exact solver
//! converges to. It is fit by **evidence maximization** (empirical-Bayes ridge,
//! MacKay's evidence framework) against the exact solver's outputs on the
//! training stream — a closed-form, analytic fit: no autodiff, no grid search,
//! REML-style variance components, in keeping with `SPEC.md`.
//!
//! Capacity is justified by EVIDENCE, not by a knob (`SPEC.md`: no magic
//! constants). We fit a linear encoder and a linear-plus-diagonal-quadratic
//! encoder and keep whichever has the higher pooled log marginal likelihood —
//! the Bayesian model-comparison quantity. The linear model is the null; the
//! quadratic head must EARN its extra capacity through the evidence Occam
//! balance, so the encoder recovers the linear map exactly when the data are
//! linear (`SPEC.md`: default recovers the null, capacity is opt-in).
//!
//! The AMORTIZATION GAP — EV(exact) − EV(amortized), coordinate-error
//! distribution, gate agreement — is the deployed encode error and is reported
//! as a first-class fit artifact ([`AmortizationErrorStats`] here; the
//! explained-variance halves are assembled at the term level where the decoder
//! is in scope).

use gam_linalg::faer_ndarray::FaerSvd;
use ndarray::{Array1, Array2, ArrayView2};

/// Relative convergence tolerance for the evidence (MacKay) fixed point. This
/// is a NUMERIC iteration tolerance on `log λ`, not a model knob: it only
/// decides when the closed-form variance-component iteration has stopped
/// moving to f64 working precision, and a tighter value cannot change the fitted
/// encoder beyond rounding.
const EVIDENCE_REL_TOL: f64 = 1.0e-10;

/// Hard ceiling on evidence fixed-point iterations. The MacKay recursion
/// contracts geometrically, so convergence is reached far below this; the cap
/// only guarantees termination (it is never a wall-clock budget — `SPEC.md`).
const EVIDENCE_MAX_ITERS: usize = 500;

/// How many trailing `(log λ, |Δ log λ|)` iterates the exhaustion refusal
/// reports. Enough to read the residual's trend — a shrinking `|Δ|` is a budget
/// shortfall, a flat one is a limit cycle — and to see a period-2 alternation
/// for what it is.
const EVIDENCE_REFUSAL_TAIL: usize = 8;

/// The distilled per-row solution the amortized encoder predicts in one matmul:
/// gate logits, per-atom latent coordinates, and per-atom amplitudes, laid out
/// exactly like the exact solver's converged state.
#[derive(Debug, Clone)]
pub struct AmortizedCode {
    /// Gate logits, `n × K` (the pre-assignment routing scores).
    pub logits: Array2<f64>,
    /// Per-atom latent coordinates, one `n × d_k` block per atom.
    pub coords: Vec<Array2<f64>>,
    /// Per-atom amplitudes (assignment masses), `n × K`. Masses are
    /// non-negative by construction, so the raw regression prediction is
    /// clamped at zero (out-of-domain negative mass carries no meaning).
    pub amplitudes: Array2<f64>,
}

/// Per-axis PERIODICITY of one atom's latent coordinates, one entry per latent
/// axis (its length equals the atom's `latent_dim`): `Some(period)` marks a
/// PERIODIC (circle-phase) axis whose coordinate wraps modulo `period` — `1.0`
/// for the fraction-of-period convention the periodic/torus/cylinder bases use,
/// `TAU` for a longitude in radians — and `None` marks a flat (Euclidean or
/// bounded-interval) axis with no wrap. This is a STRUCTURAL property, read from
/// the atom's basis kind / `LatentManifold` (`Circle { period }` ⇒
/// `Some(period)`; `Euclidean`/`Interval` ⇒ `None`), never inferred from the
/// coordinate values.
pub type AxisPeriods = Vec<Option<f64>>;

// ── Why periodic coordinates MUST be regressed through a circular embedding ──
//
// A circle coordinate `t ∈ [0, period)` identifies `0` with `period`: the seam
// is a labelling artefact, not a feature of the geometry. When the chart seam
// lands INSIDE the data cloud, the supervision targets for `t` straddle it —
// some rows carry `t ≈ 0⁺`, physically-adjacent rows carry `t ≈ period⁻`. A raw
// least-squares regression of `t` estimates the conditional ARITHMETIC mean
// `E[t | x]`, which for such a straddling conditional collapses to the midpoint
// (`mean(0.02·P, 0.98·P) = 0.5·P`) — the ANTIPODE of the true location, the
// single most-wrong point on the circle. Raw regression is therefore not merely
// noisy on periodic axes; it is antipode-BIASED wherever the seam is in-cloud.
//
// The population-consistent estimator on the circle regresses the EMBEDDING
// `θ ↦ (cos θ, sin θ)`, `θ = 2π t / period`. Each embedding coordinate is an
// ordinary (seam-invariant) scalar, so `E[cos θ | x]` and `E[sin θ | x]` are
// well-posed linear-regression targets; `atan2(Ê[sin], Ê[cos])` recovers the
// conditional MEAN DIRECTION — the Fréchet mean of the circular conditional
// under the chord metric — which is exactly the intrinsic circular mean and is
// invariant to where the seam is drawn. Squared error in the embedding equals
// `2(1 − cos Δθ)`, the squared chordal distance, a monotone wrap-aware loss,
// unlike the raw `(Δt)²` that charges a `0.02`-vs-`0.98` pair its full `0.96`.
// Scoring mirrors this: per-axis coordinate error is wrapped by the period,
// `err = min(|Δ| mod P, P − (|Δ| mod P))`, before it is squared or quantiled.

/// The reconstruction-independent half of the amortization-gap artifact: how far
/// the amortized (one-matmul) prediction sits from the exact per-row solution in
/// coordinate, gate, and amplitude space. The explained-variance halves
/// (`EV(exact)` vs `EV(amortized)`) are assembled at the term level, where the
/// decoder is available to turn a code into a reconstruction.
#[derive(Debug, Clone)]
pub struct AmortizationErrorStats {
    /// Root-mean-square error of the predicted latent coordinates against the
    /// exact coordinates, pooled over every (row, atom, axis).
    pub coord_rmse: f64,
    /// Coordinate absolute-error quantiles `[min, q25, median, q75, max]`,
    /// pooled over every (row, atom, axis). The full distribution, not just a
    /// summary, because the amortization gap is heavy-tailed: a handful of
    /// hard rows dominate the deployed error.
    pub coord_abs_err_quantiles: [f64; 5],
    /// Fraction of (row, atom) pairs whose assignment mass lies on the same side
    /// of the caller's activity floor. Assignment masses, unlike raw logits, are
    /// softmax-shift invariant and include threshold-gate and ordered Beta--Bernoulli-prior effects.
    pub support_agreement: f64,
    /// Root-mean-square error of predicted amplitudes against exact amplitudes,
    /// pooled over every (row, atom).
    pub amplitude_rmse: f64,
}

/// The exact per-row test-time optimizer's solution on the held-out rows, the
/// oracle input to [`crate::manifold`]'s `amortization_gap`: the exact
/// reconstruction plus the exact code (gate logits, per-atom coordinate blocks,
/// amplitudes) against which the one-matmul amortized encode is scored.
pub struct ExactRowSolution<'a> {
    /// Exact reconstruction of the held-out rows (the oracle line's EV numerator).
    pub recon: ArrayView2<'a, f64>,
    /// Exact per-(row, atom) gate logits.
    pub logits: ArrayView2<'a, f64>,
    /// Exact per-atom coordinate blocks (one `(rows × axes)` matrix per atom).
    pub coords: &'a [Array2<f64>],
    /// Exact per-(row, atom) amplitudes.
    pub amplitudes: ArrayView2<'a, f64>,
}

/// The full amortization-gap artifact (reviewer condition #2): the deployed
/// distilled encoder's cost, side by side with the exact-solve oracle. The gap
/// between `ev_exact` and `ev_amortized` IS the held-out reconstruction quality
/// a reviewer must be shown as the primary out-of-sample row (one matmul), with
/// the exact solve as the oracle line.
#[derive(Debug, Clone)]
pub struct AmortizationGap {
    /// Held-out explained variance of the EXACT per-row solve (the oracle line).
    /// `None` when the target is degenerate (constant).
    pub ev_exact: Option<f64>,
    /// Held-out explained variance of the AMORTIZED one-matmul encode (the
    /// PRIMARY deployed number).
    pub ev_amortized: Option<f64>,
    /// `ev_exact − ev_amortized`: the explained-variance the amortization costs.
    /// The honest gap between the test-time optimizer and the deployed encoder.
    pub ev_gap: Option<f64>,
    /// Coordinate / gate / amplitude error of the amortized code against the
    /// exact code on the same held-out rows.
    pub errors: AmortizationErrorStats,
    /// The joint multi-start-fallback fraction on the exact solution — the share
    /// of rows whose co-active atoms couple beyond the per-atom certificate, the
    /// encode-tax cost multiplier ([`crate::encode::joint_encode_fallback_fraction`]).
    pub joint_multistart_fraction: f64,
    /// Whether the encoder admitted the diagonal-quadratic head over the linear
    /// null (capacity justified by evidence).
    pub used_quadratic_head: bool,
    /// Pooled log marginal likelihood of the trained encoder.
    pub encoder_log_evidence: f64,
    /// Feature count of the trained encoder.
    pub encoder_feature_dim: usize,
    /// Effective degrees of freedom per target of the trained encoder.
    pub encoder_effective_dof: f64,
}

/// A per-column affine standardization `(v − mean) / scale` with a
/// zero-variance-safe scale (a constant column standardizes to the zero column
/// and contributes nothing, rather than dividing by zero).
#[derive(Debug, Clone)]
struct Standardizer {
    mean: Array1<f64>,
    scale: Array1<f64>,
}

impl Standardizer {
    /// Fit column means and (population) standard deviations; a column whose
    /// standard deviation underflows to zero gets unit scale so it maps to the
    /// zero column after centering.
    fn fit(data: ArrayView2<'_, f64>) -> Self {
        let (n, d) = data.dim();
        let mut mean = Array1::<f64>::zeros(d);
        let mut scale = Array1::<f64>::ones(d);
        if n == 0 {
            return Self { mean, scale };
        }
        for col in 0..d {
            let mut acc = 0.0;
            for row in 0..n {
                acc += data[[row, col]];
            }
            let m = acc / n as f64;
            mean[col] = m;
            let mut var = 0.0;
            for row in 0..n {
                let c = data[[row, col]] - m;
                var += c * c;
            }
            let sd = (var / n as f64).sqrt();
            scale[col] = if sd > 0.0 && sd.is_finite() { sd } else { 1.0 };
        }
        Self { mean, scale }
    }

    /// Apply the standardization to a fresh matrix (same column count).
    fn apply(&self, data: ArrayView2<'_, f64>) -> Array2<f64> {
        let (n, d) = data.dim();
        let mut out = Array2::<f64>::zeros((n, d));
        for row in 0..n {
            for col in 0..d {
                out[[row, col]] = (data[[row, col]] - self.mean[col]) / self.scale[col];
            }
        }
        out
    }
}

/// The feature map the encoder regresses on: standardized ambient features,
/// optionally augmented with their diagonal squares. The augmentation (the
/// small nonlinear head) is admitted only when the evidence prefers it.
#[derive(Debug, Clone)]
enum FeatureMap {
    /// Standardize `x`; the feature vector is the standardized row.
    Linear { std: Standardizer },
    /// Standardize `x`, append the elementwise squares of the standardized
    /// row, then standardize the whole `[x, x²]` block. A bounded `2p`-wide
    /// nonlinear head (diagonal quadratic only — cross terms are `O(p²)` and are
    /// excluded so the design never blows up memory, `SPEC.md`).
    Quadratic {
        raw_std: Standardizer,
        feat_std: Standardizer,
    },
}

impl FeatureMap {
    /// Build the `n × F` design matrix for these rows under this feature map.
    fn design(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        match self {
            FeatureMap::Linear { std } => std.apply(x),
            FeatureMap::Quadratic { raw_std, feat_std } => {
                let z = raw_std.apply(x);
                let (n, p) = z.dim();
                let mut raw = Array2::<f64>::zeros((n, 2 * p));
                for row in 0..n {
                    for col in 0..p {
                        let v = z[[row, col]];
                        raw[[row, col]] = v;
                        raw[[row, p + col]] = v * v;
                    }
                }
                feat_std.apply(raw.view())
            }
        }
    }
}

/// The evidence-maximizing multi-output ridge solved in the rotated (SVD) basis.
/// Carries the fitted weights, the pooled log marginal likelihood (the capacity
/// arbiter), and the effective degrees of freedom (for the artifact).
#[derive(Debug, Clone)]
struct EvidenceRidge {
    /// Regression weights, `F × T` (standardized features → standardized
    /// targets). Prediction is `Ŷ_std = Φ_std · weights`.
    weights: Array2<f64>,
    /// Pooled log marginal likelihood at the converged variance components.
    log_evidence: f64,
    /// Effective degrees of freedom per target, `Σ_i s_i²/(s_i²+λ)`.
    effective_dof: f64,
}

/// Fit a single-penalty multi-output ridge by MacKay evidence maximization.
///
/// `design` is `n × F` (standardized features), `targets` is `n × T`
/// (standardized targets). One prior precision `α` (shared over all `F·T`
/// weights) and one noise precision `β` (shared over all `n·T` residuals) are
/// estimated by the closed-form empirical-Bayes fixed point
///
/// ```text
///     γ      = Σ_i s_i²/(s_i² + λ),           λ = α/β
///     α_new  = γ·T / Σ_t ‖w_t‖²
///     β_new  = (n − γ)·T / Σ_t ‖y_t − Φ w_t‖²
/// ```
///
/// which converges to the marginal-likelihood maximizer (empirical Bayes /
/// REML variance components). Everything is computed in the rotated basis
/// `Φ = U S Vᵀ` so each iteration is `O((n + F)·r + r·T)`.
fn fit_evidence_ridge(
    design: ArrayView2<'_, f64>,
    targets: ArrayView2<'_, f64>,
) -> Result<EvidenceRidge, String> {
    let (n, f_dim) = design.dim();
    let t_dim = targets.ncols();
    if targets.nrows() != n {
        return Err(format!(
            "fit_evidence_ridge: design has {n} rows but targets have {}",
            targets.nrows()
        ));
    }
    if n == 0 || f_dim == 0 || t_dim == 0 {
        return Ok(EvidenceRidge {
            weights: Array2::zeros((f_dim, t_dim)),
            log_evidence: f64::NEG_INFINITY,
            effective_dof: 0.0,
        });
    }
    // Thin SVD Φ = U S Vᵀ: U (n×r), s (r), Vt (r×F), r = min(n, F).
    let design_owned = design.to_owned();
    let (u_opt, svals, vt_opt) = design_owned
        .svd(true, true)
        .map_err(|e| format!("fit_evidence_ridge: SVD failed: {e:?}"))?;
    let u = u_opt.ok_or_else(|| "fit_evidence_ridge: SVD returned no U".to_string())?;
    let vt = vt_opt.ok_or_else(|| "fit_evidence_ridge: SVD returned no Vt".to_string())?;
    let r = svals.len();

    // Rotate the targets into the left-singular basis: Z = Uᵀ Y, r×T. Also cache
    // the total target energy ‖y_t‖² so the residual sum-of-squares can be read
    // off the rotated coordinates (the in-space part) plus the orthogonal tail.
    let z = u.t().dot(&targets); // r×T
    let mut y_energy = vec![0.0_f64; t_dim];
    for col in 0..t_dim {
        let mut acc = 0.0;
        for row in 0..n {
            let v = targets[[row, col]];
            acc += v * v;
        }
        y_energy[col] = acc;
    }
    let mut z_energy = vec![0.0_f64; t_dim]; // Σ_i z_it²  (energy captured in-space)
    for col in 0..t_dim {
        let mut acc = 0.0;
        for i in 0..r {
            let v = z[[i, col]];
            acc += v * v;
        }
        z_energy[col] = acc;
    }

    let s2: Vec<f64> = svals.iter().map(|s| s * s).collect();
    // Initialize the variance components from the data scale: α from the mean
    // in-space signal energy, β from the residual after a mild ridge. These are
    // data-derived starts, not knobs; the fixed point is globally attracting for
    // this convex-in-log problem, so the start only affects iteration count.
    let mut alpha = 1.0_f64;
    let mut beta = 1.0_f64;
    let n_t = (n * t_dim) as f64;
    // A machine-precision noise floor RELATIVE to the total target energy. When
    // the design can interpolate the targets the residual sum-of-squares → 0 and
    // an unfloored `β = (n−γ)/RSS` overflows to `+∞` (the divergence guard then
    // aborts a perfectly-fittable problem). Flooring RSS at `ε·‖Y‖²` caps the
    // fitted noise precision at f64 resolution — we cannot claim a noise level
    // below the data's own rounding — keeping `β` finite without a magic knob.
    // The same floor on `Σ‖w‖²` caps `α` when the fit shrinks to the null.
    let total_energy: f64 = y_energy.iter().sum();
    let energy_floor = (total_energy * f64::EPSILON).max(f64::MIN_POSITIVE);
    let mut effective_dof = 0.0_f64;
    let mut last_log_lambda = f64::NAN;
    // SPEC 20 — the MacKay fixed point must CERTIFY, not merely terminate. The
    // docstring's geometric-contraction argument makes the cap a safety bound,
    // but an assumption is not a certificate: if the loop exhausts the cap
    // without the relative log-λ residual passing `EVIDENCE_REL_TOL`, the final
    // iterate is a NON-converged evidence state and must not be minted as an
    // `EvidenceRidge` indistinguishable from a converged one.
    let mut evidence_converged = false;
    // Refusal evidence. `|Δ log λ|` against the threshold it was tested at, the
    // fitted state that produced it, and a short tail of iterates — the two
    // exhaustion modes (a residual still contracting toward the tolerance, and
    // one parked on a limit cycle) are indistinguishable without them.
    let mut last_delta = f64::NAN;
    let mut last_gamma = f64::NAN;
    let mut last_w_sq_sum = f64::NAN;
    let mut last_rss_sum = f64::NAN;
    let mut tail: Vec<(f64, f64)> = Vec::new();
    for _ in 0..EVIDENCE_MAX_ITERS {
        let lambda = (alpha / beta).max(f64::MIN_POSITIVE);
        // γ = Σ_i s_i²/(s_i²+λ); pooled ‖w‖² and RSS across targets.
        let mut gamma = 0.0_f64;
        for i in 0..r {
            gamma += s2[i] / (s2[i] + lambda);
        }
        let mut w_sq_sum = 0.0_f64;
        let mut rss_sum = 0.0_f64;
        for col in 0..t_dim {
            let mut w_sq = 0.0_f64;
            let mut rss_in = 0.0_f64;
            for i in 0..r {
                let denom = s2[i] + lambda;
                let coeff = svals[i] / denom; // w-coordinate = coeff · z
                let zi = z[[i, col]];
                w_sq += (coeff * zi) * (coeff * zi);
                let shrink = lambda / denom; // residual-in-space factor
                rss_in += (shrink * zi) * (shrink * zi);
            }
            w_sq_sum += w_sq;
            // Residual = in-space shrunk residual + orthogonal tail (y not
            // reachable by Φ). The tail is ‖y‖² − ‖z‖².
            let tail = (y_energy[col] - z_energy[col]).max(0.0);
            rss_sum += rss_in + tail;
        }
        effective_dof = gamma;
        last_gamma = gamma;
        last_w_sq_sum = w_sq_sum;
        last_rss_sum = rss_sum;
        // NULL TERMINATION. Once the fitted weight energy has fallen to the
        // data's own resolution floor, every λ from here upward yields the same
        // estimator to f64 precision: α is unidentifiable, and the `α = γT/‖w‖²`
        // update below is no longer the MacKay map — with `‖w‖²` clamped to the
        // floor its denominator stops responding to λ, so the iterate crosses
        // the clamp instead of contracting through it and settles into a limit
        // cycle the log-λ tolerance can never close (measured on the pure-noise
        // fixture: |Δ log λ| pinned at 2.55e-1, 1.3e8× the threshold, for all
        // 500 iterations, with ‖w‖² = 1.6e-14 against a 6.7e-14 floor).
        //
        // That state is not a failure to converge — it IS the null solution, and
        // it is certified by a STRONGER condition than the residual test: the
        // residual test asks whether λ has stopped moving, this establishes that
        // λ cannot matter. `effective_dof` (γ ≈ 0 here) carries the null verdict
        // to the caller. The exhaustion refusal below still governs every case
        // where the floor is NOT engaged, i.e. where λ is identifiable and the
        // fixed point is genuinely unreached.
        if w_sq_sum <= energy_floor {
            evidence_converged = true;
            break;
        }
        // MacKay updates with strictly-positive floors (a perfectly-fit or
        // perfectly-shrunk direction must not divide by zero).
        alpha = (gamma * t_dim as f64) / w_sq_sum.max(energy_floor);
        let well_determined = (n_t - gamma * t_dim as f64).max(f64::MIN_POSITIVE);
        beta = well_determined / rss_sum.max(energy_floor);
        if !(alpha.is_finite() && beta.is_finite()) {
            return Err("fit_evidence_ridge: variance components diverged".to_string());
        }
        let log_lambda = (alpha / beta).ln();
        if last_log_lambda.is_finite() {
            last_delta = (log_lambda - last_log_lambda).abs();
        }
        tail.push((log_lambda, last_delta));
        if tail.len() > EVIDENCE_REFUSAL_TAIL {
            tail.remove(0);
        }
        if last_log_lambda.is_finite()
            && (log_lambda - last_log_lambda).abs() <= EVIDENCE_REL_TOL * (1.0 + log_lambda.abs())
        {
            evidence_converged = true;
            break;
        }
        last_log_lambda = log_lambda;
    }
    if !evidence_converged {
        let log_lambda = (alpha / beta).ln();
        let threshold = EVIDENCE_REL_TOL * (1.0 + log_lambda.abs());
        let tail_text = tail
            .iter()
            .map(|(value, delta)| format!("({value:.17e}, {delta:.3e})"))
            .collect::<Vec<_>>()
            .join(" ");
        return Err(format!(
            "fit_evidence_ridge: MacKay evidence iteration exhausted its \
             {EVIDENCE_MAX_ITERS}-iteration safety cap without meeting the relative \
             log-λ fixed-point tolerance {EVIDENCE_REL_TOL:.1e} (last log λ = \
             {log_lambda:.17e}, previous = {last_log_lambda:.17e}, |Δ| = {last_delta:.6e} \
             against threshold {threshold:.6e}, i.e. {:.3e}× too large; fitted state \
             γ = {last_gamma:.6e}, ‖w‖² = {last_w_sq_sum:.6e}, RSS = {last_rss_sum:.6e}, \
             energy floor = {energy_floor:.6e}); last {} iterates (log λ, |Δ|) = \
             [{tail_text}] — a |Δ| still shrinking across them is a budget shortfall, \
             a |Δ| pinned flat is a limit cycle; refusing to mint a non-converged \
             evidence ridge — the cap never selects the estimator",
            last_delta / threshold,
            tail.len()
        ));
    }

    let lambda = (alpha / beta).max(f64::MIN_POSITIVE);
    // Weights: w_t = V diag(s_i/(s_i²+λ)) z_t. Compute the r×T rotated
    // coefficient matrix, then lift by V (= Vtᵀ).
    let mut rotated = Array2::<f64>::zeros((r, t_dim));
    for i in 0..r {
        let coeff = svals[i] / (s2[i] + lambda);
        for col in 0..t_dim {
            rotated[[i, col]] = coeff * z[[i, col]];
        }
    }
    let weights = vt.t().dot(&rotated); // (F×r)·(r×T) = F×T

    // Pooled log marginal likelihood (MacKay evidence). Per target, with
    // A = αI + βΦᵀΦ (F×F), log|A| = Σ_i ln(α + β s_i²) + (F − r) ln α:
    //   ln p(y) = (F/2)ln α + (n/2)ln β − (β/2)RSS − (α/2)‖w‖²
    //             − ½ln|A| − (n/2)ln(2π).
    let mut log_det_a = 0.0_f64;
    for i in 0..r {
        log_det_a += (alpha + beta * s2[i]).ln();
    }
    log_det_a += (f_dim.saturating_sub(r)) as f64 * alpha.ln();
    let two_pi = std::f64::consts::TAU;
    let mut w_sq_sum = 0.0_f64;
    let mut rss_sum = 0.0_f64;
    for col in 0..t_dim {
        for i in 0..r {
            let denom = s2[i] + lambda;
            let coeff = svals[i] / denom;
            let zi = z[[i, col]];
            w_sq_sum += (coeff * zi) * (coeff * zi);
            let shrink = lambda / denom;
            rss_sum += (shrink * zi) * (shrink * zi);
        }
        rss_sum += (y_energy[col] - z_energy[col]).max(0.0);
    }
    let log_evidence = t_dim as f64
        * (0.5 * f_dim as f64 * alpha.ln() + 0.5 * n as f64 * beta.ln()
            - 0.5 * log_det_a
            - 0.5 * n as f64 * two_pi.ln())
        - 0.5 * beta * rss_sum
        - 0.5 * alpha * w_sq_sum;

    Ok(EvidenceRidge {
        weights,
        log_evidence,
        effective_dof,
    })
}

/// A trained distilled encoder: a feature map, the evidence-ridge weights, and
/// the target standardization needed to turn a standardized prediction back into
/// (logits, coords, amplitudes) in the exact solver's layout.
#[derive(Debug, Clone)]
pub struct LearnedAmortizedEncoder {
    feature_map: FeatureMap,
    /// `F × T` standardized-feature → standardized-target weights.
    weights: Array2<f64>,
    /// Target de-standardization (`Ŷ = Ŷ_std ⊙ scale + mean`).
    target_std: Standardizer,
    k_atoms: usize,
    coord_dims: Vec<usize>,
    /// Per-atom, per-axis periodicity of the latent coordinates. A `Some(period)`
    /// axis was regressed through its circular embedding `(cos, sin)` and is
    /// inverted back to `[0, period)` by `atan2` on `predict`; a `None` axis was
    /// regressed and is emitted RAW. Length `K`; inner length = the atom's
    /// latent dim.
    coord_periods: Vec<AxisPeriods>,
    /// Pooled log marginal likelihood of the winning feature map (the capacity
    /// the evidence justified). Reported in the artifact.
    pub log_evidence: f64,
    /// Number of features `F` in the winning design.
    pub feature_dim: usize,
    /// Effective degrees of freedom per target of the winning fit.
    pub effective_dof: f64,
    /// `true` when the evidence admitted the diagonal-quadratic head over the
    /// linear null.
    pub used_quadratic_head: bool,
}

impl LearnedAmortizedEncoder {
    /// Assemble the `n × T` standardized target matrix from the exact solver's
    /// per-row solution: `K` gate logits, the coordinate block, then `K`
    /// amplitudes. A NON-PERIODIC axis contributes one raw-`t` target column; a
    /// PERIODIC axis (`coord_periods[atom][axis] = Some(period)`) contributes TWO
    /// columns — its circular embedding `(cos 2πt/period, sin 2πt/period)` — so
    /// the regression estimates the seam-invariant conditional mean direction
    /// rather than the antipode-biased arithmetic mean of the wrapped angle (see
    /// the module-level derivation above the `AxisPeriods` alias). Returns the
    /// target matrix and the ORIGINAL per-atom coordinate dims (the emitted
    /// coordinate width, not the embedded target width).
    fn stack_targets(
        logits: ArrayView2<'_, f64>,
        coords: &[Array2<f64>],
        amplitudes: ArrayView2<'_, f64>,
        coord_periods: &[AxisPeriods],
    ) -> Result<(Array2<f64>, Vec<usize>), String> {
        let (n, k) = logits.dim();
        if amplitudes.dim() != (n, k) {
            return Err(format!(
                "LearnedAmortizedEncoder: amplitudes {:?} must match logits ({n}, {k})",
                amplitudes.dim()
            ));
        }
        if coords.len() != k {
            return Err(format!(
                "LearnedAmortizedEncoder: {} coord blocks but K={k}",
                coords.len()
            ));
        }
        if coord_periods.len() != k {
            return Err(format!(
                "LearnedAmortizedEncoder: {} axis-period blocks but K={k}",
                coord_periods.len()
            ));
        }
        let coord_dims: Vec<usize> = coords.iter().map(|c| c.ncols()).collect();
        // Target width: one column per flat axis, TWO per periodic axis.
        let mut coord_target_width = 0usize;
        for (atom, &d) in coord_dims.iter().enumerate() {
            if coord_periods[atom].len() != d {
                return Err(format!(
                    "LearnedAmortizedEncoder: atom {atom} has {} axis periods but latent dim {d}",
                    coord_periods[atom].len()
                ));
            }
            for axis in 0..d {
                if let Some(period) = coord_periods[atom][axis] {
                    if !(period.is_finite() && period > 0.0) {
                        return Err(format!(
                            "LearnedAmortizedEncoder: atom {atom} axis {axis} period {period} must be finite and positive"
                        ));
                    }
                    coord_target_width += 2;
                } else {
                    coord_target_width += 1;
                }
            }
        }
        let t_dim = 2 * k + coord_target_width;
        let mut targets = Array2::<f64>::zeros((n, t_dim));
        for col in 0..k {
            for row in 0..n {
                targets[[row, col]] = logits[[row, col]];
            }
        }
        let mut offset = k;
        for (atom, coord) in coords.iter().enumerate() {
            if coord.nrows() != n {
                return Err(format!(
                    "LearnedAmortizedEncoder: coord block {atom} has {} rows, expected {n}",
                    coord.nrows()
                ));
            }
            let d = coord_dims[atom];
            for axis in 0..d {
                match coord_periods[atom][axis] {
                    Some(period) => {
                        let w = std::f64::consts::TAU / period;
                        for row in 0..n {
                            let ang = coord[[row, axis]] * w;
                            targets[[row, offset]] = ang.cos();
                            targets[[row, offset + 1]] = ang.sin();
                        }
                        offset += 2;
                    }
                    None => {
                        for row in 0..n {
                            targets[[row, offset]] = coord[[row, axis]];
                        }
                        offset += 1;
                    }
                }
            }
        }
        for col in 0..k {
            for row in 0..n {
                targets[[row, offset + col]] = amplitudes[[row, col]];
            }
        }
        Ok((targets, coord_dims))
    }

    /// Fit the distilled encoder against the exact solver's training-stream
    /// solution, treating EVERY latent axis as flat (Euclidean). `x` is the
    /// `n × p` ambient corpus; `logits`/`amplitudes` are `n × K`; `coords` is one
    /// `n × d_k` block per atom. The evidence chooses between the linear and the
    /// diagonal-quadratic feature map.
    ///
    /// Atoms with PERIODIC (circle-phase) axes — periodic/torus/cylinder/sphere-
    /// longitude — MUST instead use [`Self::fit_with_axis_periods`], which
    /// regresses those axes through their circular embedding; a raw fit of a
    /// periodic coordinate whose seam lands in-cloud is antipode-biased (see the
    /// module-level derivation). This entry point is the all-flat special case.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        logits: ArrayView2<'_, f64>,
        coords: &[Array2<f64>],
        amplitudes: ArrayView2<'_, f64>,
    ) -> Result<Self, String> {
        let coord_periods: Vec<AxisPeriods> =
            coords.iter().map(|c| vec![None; c.ncols()]).collect();
        Self::fit_with_axis_periods(x, logits, coords, amplitudes, &coord_periods)
    }

    /// Fit the distilled encoder with per-atom, per-axis PERIODICITY threaded
    /// from the atoms' basis kinds (`Circle { period } ⇒ Some(period)`,
    /// `Euclidean`/`Interval ⇒ None`). Periodic axes are regressed through their
    /// circular embedding `(cos, sin)` and inverted to `[0, period)` on
    /// `predict`; flat axes are regressed raw and behave EXACTLY as [`Self::fit`].
    /// `coord_periods` has one [`AxisPeriods`] per atom, each of length equal to
    /// that atom's latent dim.
    pub fn fit_with_axis_periods(
        x: ArrayView2<'_, f64>,
        logits: ArrayView2<'_, f64>,
        coords: &[Array2<f64>],
        amplitudes: ArrayView2<'_, f64>,
        coord_periods: &[AxisPeriods],
    ) -> Result<Self, String> {
        let (n, _p) = x.dim();
        let k_atoms = logits.ncols();
        if n == 0 {
            return Err("LearnedAmortizedEncoder::fit: empty training corpus".to_string());
        }
        let (targets, coord_dims) = Self::stack_targets(logits, coords, amplitudes, coord_periods)?;
        let target_std = Standardizer::fit(targets.view());
        let targets_std = target_std.apply(targets.view());

        // Linear feature map (the null capacity).
        let lin_std = Standardizer::fit(x);
        let linear_map = FeatureMap::Linear { std: lin_std };
        let lin_design = linear_map.design(x);
        let lin_fit = fit_evidence_ridge(lin_design.view(), targets_std.view())?;

        // Diagonal-quadratic feature map (opt-in capacity). Built only if there
        // is more than one feature to square meaningfully; the evidence decides
        // whether to keep it.
        let raw_std = Standardizer::fit(x);
        let z = raw_std.apply(x);
        let (nn, p) = z.dim();
        let mut raw = Array2::<f64>::zeros((nn, 2 * p));
        for row in 0..nn {
            for col in 0..p {
                let v = z[[row, col]];
                raw[[row, col]] = v;
                raw[[row, p + col]] = v * v;
            }
        }
        let feat_std = Standardizer::fit(raw.view());
        let quad_map = FeatureMap::Quadratic { raw_std, feat_std };
        let quad_design = quad_map.design(x);
        let quad_fit = fit_evidence_ridge(quad_design.view(), targets_std.view())?;

        // Model selection by pooled log evidence: the quadratic head is admitted
        // ONLY when it raises the marginal likelihood over the linear null.
        let use_quadratic = quad_fit.log_evidence > lin_fit.log_evidence;
        let (feature_map, fit) = if use_quadratic {
            (quad_map, quad_fit)
        } else {
            (linear_map, lin_fit)
        };
        let feature_dim = fit.weights.nrows();

        Ok(Self {
            feature_map,
            weights: fit.weights,
            target_std,
            k_atoms,
            coord_dims,
            coord_periods: coord_periods.to_vec(),
            log_evidence: fit.log_evidence,
            feature_dim,
            effective_dof: fit.effective_dof,
            used_quadratic_head: use_quadratic,
        })
    }

    /// Predict the distilled per-row solution for fresh rows `x` (`m × p`) in one
    /// matmul: standardize features, apply the weights, de-standardize, then
    /// split into (logits, coords, amplitudes). Amplitudes are clamped at zero
    /// (masses are non-negative).
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<AmortizedCode, String> {
        let design = self.feature_map.design(x);
        if design.ncols() != self.weights.nrows() {
            return Err(format!(
                "LearnedAmortizedEncoder::predict: design width {} != weight rows {}",
                design.ncols(),
                self.weights.nrows()
            ));
        }
        let m = design.nrows();
        let pred_std = design.dot(&self.weights); // m×T
        // De-standardize into the full target vector.
        let t_dim = self.target_std.mean.len();
        let mut pred = Array2::<f64>::zeros((m, t_dim));
        for row in 0..m {
            for col in 0..t_dim {
                pred[[row, col]] =
                    pred_std[[row, col]] * self.target_std.scale[col] + self.target_std.mean[col];
            }
        }
        let k = self.k_atoms;
        let mut logits = Array2::<f64>::zeros((m, k));
        for col in 0..k {
            for row in 0..m {
                logits[[row, col]] = pred[[row, col]];
            }
        }
        let mut coords = Vec::with_capacity(k);
        let mut offset = k;
        for (atom, &d) in self.coord_dims.iter().enumerate() {
            let mut block = Array2::<f64>::zeros((m, d));
            for axis in 0..d {
                match self.coord_periods[atom][axis] {
                    // Periodic axis: two predicted columns (ĉ, ŝ) estimate the
                    // conditional mean direction; invert via atan2 and wrap into
                    // `[0, period)`. `rem_euclid` maps the `(-period/2, period/2]`
                    // principal branch onto the canonical `[0, period)` chart.
                    Some(period) => {
                        let w = std::f64::consts::TAU / period;
                        for row in 0..m {
                            let c = pred[[row, offset]];
                            let s = pred[[row, offset + 1]];
                            block[[row, axis]] = (s.atan2(c) / w).rem_euclid(period);
                        }
                        offset += 2;
                    }
                    None => {
                        for row in 0..m {
                            block[[row, axis]] = pred[[row, offset]];
                        }
                        offset += 1;
                    }
                }
            }
            coords.push(block);
        }
        let mut amplitudes = Array2::<f64>::zeros((m, k));
        for col in 0..k {
            for row in 0..m {
                amplitudes[[row, col]] = pred[[row, offset + col]].max(0.0);
            }
        }
        Ok(AmortizedCode {
            logits,
            coords,
            amplitudes,
        })
    }

    /// The number of atoms this encoder predicts a code for.
    pub fn k_atoms(&self) -> usize {
        self.k_atoms
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    /// A deterministic pseudo-random generator (linear congruential) so the
    /// tests carry no `rand` dependency and are bit-reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            // Box–Muller from two uniforms.
            let u1 = self.next_f64().max(1.0e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// A pure-noise target (no dependence on x) must shrink to the null: the
    /// encoder predicts (near) the target mean, so held-out coordinate RMSE is
    /// no worse than predicting the mean would be. Empirical-Bayes ridge recovers
    /// the null rather than chasing noise.
    #[test]
    fn shrinks_to_null_on_pure_noise() {
        let mut rng = Lcg(7);
        let n = 300usize;
        let p = 5usize;
        let x = Array::from_shape_fn((n, p), |_| rng.normal());
        let mut coord = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            coord[[row, 0]] = rng.normal(); // independent of x
        }
        let logits = Array2::<f64>::from_elem((n, 1), 1.0);
        let amplitudes = Array2::<f64>::from_elem((n, 1), 1.0);
        let enc = LearnedAmortizedEncoder::fit(
            x.view(),
            logits.view(),
            std::slice::from_ref(&coord),
            amplitudes.view(),
        )
        .expect("fit");
        // Held-out noise.
        let x_te = Array::from_shape_fn((150, p), |_| rng.normal());
        let mut coord_te = Array2::<f64>::zeros((150, 1));
        for row in 0..150 {
            coord_te[[row, 0]] = rng.normal();
        }
        let code = enc.predict(x_te.view()).expect("predict");
        // Prediction variance must be a small fraction of the target variance —
        // the encoder is shrinking toward the (constant) mean, not fitting noise.
        let mut pmean = 0.0;
        for row in 0..150 {
            pmean += code.coords[0][[row, 0]];
        }
        pmean /= 150.0;
        let mut pvar = 0.0;
        for row in 0..150 {
            let d = code.coords[0][[row, 0]] - pmean;
            pvar += d * d;
        }
        pvar /= 150.0;
        assert!(
            pvar < 0.25,
            "on pure noise the encoder must shrink to the mean (pred var={pvar} should be «1)"
        );
    }

    /// A NON-PERIODIC (all-`None`) fit is bit-identical to the previous raw
    /// behavior: `fit` and `fit_with_axis_periods(..all None..)` produce the same
    /// prediction, and — crucially — the coordinates pass THROUGH raw (no `atan2`
    /// wrap), so a target with magnitude far outside `[0, 1)` is reproduced at its
    /// true scale (an `atan2`-inverted path could never emit `|t| > period/2`).
    #[test]
    fn non_periodic_axis_is_bit_identical_raw_passthrough() {
        let mut rng = Lcg(4242);
        let n = 300usize;
        let p = 5usize;
        let k = 2usize;
        // Linear map to a wide-range coordinate (well outside [0,1)).
        let w_coord = Array::from_shape_fn((p, k), |_| 3.0 * rng.normal());
        let make = |rng: &mut Lcg, n: usize| {
            let x = Array::from_shape_fn((n, p), |_| rng.normal());
            let coords_flat = x.dot(&w_coord);
            let coords: Vec<Array2<f64>> = (0..k)
                .map(|a| {
                    let mut c = Array2::<f64>::zeros((n, 1));
                    for row in 0..n {
                        c[[row, 0]] = coords_flat[[row, a]];
                    }
                    c
                })
                .collect();
            let logits = Array2::<f64>::from_elem((n, k), 1.0);
            let amplitudes = Array2::<f64>::from_elem((n, k), 1.0);
            (x, logits, coords, amplitudes)
        };
        let (x_tr, lg_tr, co_tr, am_tr) = make(&mut rng, n);
        let enc_default =
            LearnedAmortizedEncoder::fit(x_tr.view(), lg_tr.view(), &co_tr, am_tr.view())
                .expect("default fit");
        let all_none: Vec<AxisPeriods> = co_tr.iter().map(|c| vec![None; c.ncols()]).collect();
        let enc_none = LearnedAmortizedEncoder::fit_with_axis_periods(
            x_tr.view(),
            lg_tr.view(),
            &co_tr,
            am_tr.view(),
            &all_none,
        )
        .expect("all-none fit");

        let (x_te, ..) = make(&mut rng, 120);
        let code_default = enc_default.predict(x_te.view()).expect("predict default");
        let code_none = enc_none.predict(x_te.view()).expect("predict none");
        let mut saw_out_of_unit = false;
        for atom in 0..k {
            for row in 0..code_default.coords[atom].nrows() {
                let a = code_default.coords[atom][[row, 0]];
                let b = code_none.coords[atom][[row, 0]];
                assert!(
                    a == b,
                    "all-None periods must reproduce the raw fit bit-for-bit: {a} vs {b}"
                );
                if a.abs() > 1.0 {
                    saw_out_of_unit = true;
                }
            }
        }
        assert!(
            saw_out_of_unit,
            "a flat axis must pass through raw (out-of-[0,1) coords survive, not atan2-wrapped)"
        );
    }

}

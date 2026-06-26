//! Behaviorally-anchored SAE head (issue #912).
//!
//! # What this is
//!
//! An unsupervised SAE dictionary — manifold or linear — pins a concept's
//! *shape* and its coordinate up to isometry. Both are functionals of `p(x)`.
//! It cannot pin the map from coordinate to behavior, `p(y|x)`, which is
//! formally independent of `p(x)`: two models with identical activation
//! geometry can read out differently. The behavioral coefficient an atom
//! "gets" from a post-hoc probe is therefore not a refinement of the
//! dictionary — it is the *only* labeled source of the one thing the
//! dictionary structurally lacks.
//!
//! The fix is to make the behavior part of the model. Instead of treating the
//! auxiliary `u` as a fixed covariate in a conditional Gaussian *prior* on the
//! latent codes (`LatentIdMode::AuxPrior`, the iVAE gauge), this module
//! promotes the auxiliary signal to a *modeled outcome*: a GLM behavioral head
//!
//! ```text
//!   g(E[y_n | t_n]) = a + t_n · w
//! ```
//!
//! whose design columns are the latent codes `t_n` themselves. The head's
//! coefficients `(a, w)` live in the β tier and the head log-likelihood enters
//! the *same* Laplace/REML objective as the reconstruction channel, so REML
//! balances reconstruction vs. behavioral fit automatically — no hand-tuned
//! trade-off scalar (magic by default). Because the design depends on `ψ` (the
//! latent codes move during the joint fit), the head couples to the latent
//! block exactly the way the arrow-Schur border already hosts a β-border
//! coupled to per-row latent blocks.
//!
//! # The three pieces
//!
//! 1. [`BehavioralHead`] — the head GLM itself: value + gradient of the head
//!    log-likelihood w.r.t. the head coefficients `(a, w)` AND w.r.t. the
//!    latent codes `t` (the cross-channel coupling), under a [`RowSubsampleMask`]
//!    weighting so unlabeled rows carry zero head weight (semi-supervised).
//!
//! 2. [`LeakageAbsorber`] — the #461 Neyman-orthogonal device. Joint fitting
//!    can sculpt the dictionary to *encode the label* (rediscover your own
//!    probe). The absorber widens the reconstruction design with the head's
//!    score-influence directions so the dictionary update is orthogonalized
//!    against the label channel. The boundary it enforces is precisely
//!    "orient what `p(x)` put there" vs. "hallucinate geometry from the label"
//!    — the novel statistical content of the whole construction.
//!
//! 3. [`head_feature_significance`] — per-feature (per-atom) significance of
//!    the behavioral loading via [`wood_smooth_test`], converted to an
//!    FDR-controlled report through the e-BH multiplicity machinery
//!    ([`e_benjamini_hochberg`]). Features whose behavioral signal survives
//!    orthogonalization AND the multiplicity correction are the reportable
//!    ones.

use crate::inference::smooth_test::{SmoothTestInput, SmoothTestScale, wood_smooth_test};
use crate::inference::structure_evidence::{e_benjamini_hochberg, log_e_from_p_calibrator};
use gam_linalg::faer_ndarray::FaerSvd;
use crate::solver::row_measure::RowSubsampleMask;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Outcome family for the behavioral head.
///
/// `Binomial` is the canonical safety-probe family (a binary label —
/// deception, harmfulness — read out from the latent codes). `Multinomial`
/// covers a categorical behavioral label with `n_classes` levels via a
/// shared-design softmax head; class 0 is the reference. Both are the same
/// families already in `src/families/`; this enum only selects the head's
/// link + log-likelihood, it does not re-implement the families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxOutcomeFamily {
    /// Logistic head: `logit P(y=1 | t) = a + t·w`. A single binary label
    /// pins roughly one gauge dimension, which is why `AuxOutcome` composes
    /// with `DimSelection` ARD + the isometry pin rather than replacing them.
    Binomial,
    /// Softmax head over `n_classes` levels (class 0 reference). The behavioral
    /// subspace it can orient has dimension at most `n_classes − 1`, still
    /// low-dimensional — the geometry does the shape work.
    Multinomial { n_classes: usize },
}

impl AuxOutcomeFamily {
    /// Number of linear-predictor channels the head produces per row.
    /// Binomial = 1; Multinomial with `K` classes = `K − 1` (reference-coded).
    pub fn n_eta_channels(&self) -> usize {
        match self {
            AuxOutcomeFamily::Binomial => 1,
            AuxOutcomeFamily::Multinomial { n_classes } => n_classes.saturating_sub(1),
        }
    }

    /// A single binary label pins ~1 gauge dimension; a `K`-level categorical
    /// label pins at most `K − 1`. This is the honest behavioral-subspace
    /// dimension the head can orient — the load-bearing scope assumption from
    /// the Khemakhem precondition discussion (the geometry does the rest).
    pub fn behavioral_subspace_dim(&self) -> usize {
        self.n_eta_channels()
    }
}

/// The behavioral head GLM.
///
/// Stores the labels `y` (length `n`), the per-row head weights `w_row`
/// (length `n`; 0 ⇒ unlabeled, the semi-supervised seam), and the family.
/// The design is *not* stored: it is the live latent-code matrix `t`
/// (`n × d`) passed at evaluation time, because `t` moves during the joint
/// fit. The head coefficients are `(intercept, loadings)` with one
/// `(1 + d)`-vector per η-channel.
#[derive(Debug, Clone)]
pub struct BehavioralHead {
    family: AuxOutcomeFamily,
    /// For `Binomial`: the 0/1 label per row. For `Multinomial`: the class
    /// index in `0..n_classes` per row (stored as `f64`, integral-valued).
    y: Array1<f64>,
    /// Per-row head-channel weight. `0.0` on unlabeled rows (semi-supervised);
    /// `1.0` on labeled rows by default. Derived from a [`RowSubsampleMask`] via
    /// [`BehavioralHead::with_row_measure`].
    w_row: Array1<f64>,
}

impl BehavioralHead {
    /// Build a head from labels and an explicit per-row weight vector.
    ///
    /// `family.n_eta_channels()` channels each get a `(1 + d)` coefficient
    /// block at fit time. Validates label range against the family.
    pub fn new(
        family: AuxOutcomeFamily,
        y: Array1<f64>,
        w_row: Array1<f64>,
    ) -> Result<Self, String> {
        let n = y.len();
        if w_row.len() != n {
            return Err(format!(
                "BehavioralHead: w_row length {} != labels length {n}",
                w_row.len()
            ));
        }
        for &v in w_row.iter() {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!(
                    "BehavioralHead: row weights must be finite and ≥ 0, got {v}"
                ));
            }
        }
        match family {
            AuxOutcomeFamily::Binomial => {
                for (i, &label) in y.iter().enumerate() {
                    if label != 0.0 && label != 1.0 {
                        return Err(format!(
                            "BehavioralHead(Binomial): label[{i}] = {label} is not 0/1"
                        ));
                    }
                }
            }
            AuxOutcomeFamily::Multinomial { n_classes } => {
                if n_classes < 2 {
                    return Err(format!(
                        "BehavioralHead(Multinomial): need ≥ 2 classes, got {n_classes}"
                    ));
                }
                for (i, &label) in y.iter().enumerate() {
                    let k = label as usize;
                    if k as f64 != label || k >= n_classes {
                        return Err(format!(
                            "BehavioralHead(Multinomial): label[{i}] = {label} not an \
                             integer class index in 0..{n_classes}"
                        ));
                    }
                }
            }
        }
        Ok(Self { family, y, w_row })
    }

    /// Build a head where every row carries unit head weight (fully supervised).
    pub fn fully_supervised(family: AuxOutcomeFamily, y: Array1<f64>) -> Result<Self, String> {
        let n = y.len();
        Self::new(family, y, Array1::from_elem(n, 1.0))
    }

    /// Build a head whose per-row weights come from a [`RowSubsampleMask`]: rows
    /// outside the measure (unlabeled) get weight 0, rows inside get the
    /// measure's weight. This is the semi-supervised seam — no new mechanism,
    /// the same row-weighting the inner solver already enforces for ρ-coherence.
    pub fn with_row_measure(
        family: AuxOutcomeFamily,
        y: Array1<f64>,
        measure: &RowSubsampleMask,
    ) -> Result<Self, String> {
        let n = y.len();
        let (indices, weights) = measure.indices_and_weights(n);
        let mut w_row = Array1::<f64>::zeros(n);
        for &idx in &indices {
            if idx < n {
                w_row[idx] = weights[idx];
            }
        }
        Self::new(family, y, w_row)
    }

    pub fn family(&self) -> AuxOutcomeFamily {
        self.family
    }

    pub fn n_obs(&self) -> usize {
        self.y.len()
    }

    /// Number of head coefficients given latent dimension `d`: one
    /// `(1 + d)` block (intercept + per-axis loading) per η-channel.
    pub fn n_coeffs(&self, latent_dim: usize) -> usize {
        self.family.n_eta_channels() * (1 + latent_dim)
    }

    /// Total effective head-channel sample size `Σ_n w_row[n]` — the number of
    /// labeled rows (weighted). Zero ⇒ a vacuous head (every row unlabeled),
    /// which the validator rejects: a head with no labels cannot anchor a gauge.
    pub fn effective_labeled_count(&self) -> f64 {
        self.w_row.iter().sum()
    }

    /// Per-row, per-channel linear predictor `η[n, c] = a_c + t_n · w_c`.
    fn eta(&self, t: ArrayView2<'_, f64>, coeffs: ArrayView1<'_, f64>) -> Array2<f64> {
        let (n, d) = t.dim();
        let n_eta = self.family.n_eta_channels();
        let mut eta = Array2::<f64>::zeros((n, n_eta));
        for c in 0..n_eta {
            let base = c * (1 + d);
            let a = coeffs[base];
            for row in 0..n {
                let mut acc = a;
                for axis in 0..d {
                    acc += t[[row, axis]] * coeffs[base + 1 + axis];
                }
                eta[[row, c]] = acc;
            }
        }
        eta
    }

    /// Negative head log-likelihood and its gradient w.r.t. **both** the head
    /// coefficients and the latent codes `t`.
    ///
    /// Returns `(nll, grad_coeffs, grad_t)` where:
    /// * `nll = −Σ_n w_row[n] · log p(y_n | η_n)` (weighted),
    /// * `grad_coeffs` has length `n_coeffs(d)` — the head-coefficient gradient
    ///   that drives the β-tier update,
    /// * `grad_t` is `(n, d)` — the cross-channel coupling that flows into the
    ///   latent-block gradient (the arrow-Schur border coupling).
    ///
    /// Convention matches the rest of the latent objective: this is the
    /// *negative* log-likelihood, so it adds to the joint cost and its gradient
    /// adds to the joint gradient.
    pub fn neg_loglik_and_grad(
        &self,
        t: ArrayView2<'_, f64>,
        coeffs: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let (n, d) = t.dim();
        if n != self.y.len() {
            return Err(format!(
                "BehavioralHead: latent rows {n} != labels {}",
                self.y.len()
            ));
        }
        let n_eta = self.family.n_eta_channels();
        if coeffs.len() != n_eta * (1 + d) {
            return Err(format!(
                "BehavioralHead: coeffs length {} != n_eta·(1+d) = {}",
                coeffs.len(),
                n_eta * (1 + d)
            ));
        }
        let eta = self.eta(t, coeffs);
        let mut nll = 0.0_f64;
        let mut grad_coeffs = Array1::<f64>::zeros(n_eta * (1 + d));
        let mut grad_t = Array2::<f64>::zeros((n, d));

        match self.family {
            AuxOutcomeFamily::Binomial => {
                for row in 0..n {
                    let w = self.w_row[row];
                    if w == 0.0 {
                        continue;
                    }
                    let e = eta[[row, 0]];
                    // Numerically-stable logistic NLL: log(1+exp(η)) − y·η.
                    let log1p = if e > 0.0 {
                        e + (-e).exp().ln_1p()
                    } else {
                        e.exp().ln_1p()
                    };
                    let y = self.y[row];
                    nll += w * (log1p - y * e);
                    // dNLL/dη = p − y, p = σ(η).
                    let p = 1.0 / (1.0 + (-e).exp());
                    let r = w * (p - y);
                    grad_coeffs[0] += r;
                    for axis in 0..d {
                        grad_coeffs[1 + axis] += r * t[[row, axis]];
                        grad_t[[row, axis]] += r * coeffs[1 + axis];
                    }
                }
            }
            AuxOutcomeFamily::Multinomial { .. } => {
                for row in 0..n {
                    let w = self.w_row[row];
                    if w == 0.0 {
                        continue;
                    }
                    // Softmax over the K−1 free channels plus the implicit
                    // reference channel (η_0 ≡ 0). LSE includes the 0 term.
                    let mut max_eta = 0.0_f64;
                    for c in 0..n_eta {
                        if eta[[row, c]] > max_eta {
                            max_eta = eta[[row, c]];
                        }
                    }
                    let mut denom = (0.0 - max_eta).exp();
                    for c in 0..n_eta {
                        denom += (eta[[row, c]] - max_eta).exp();
                    }
                    let lse = max_eta + denom.ln();
                    let label = self.y[row] as usize;
                    // log p(y) = η_y − lse, with η_0 = 0 for the reference class.
                    let eta_y = if label == 0 {
                        0.0
                    } else {
                        eta[[row, label - 1]]
                    };
                    nll += w * (lse - eta_y);
                    // dNLL/dη_c = p_c − 1{y = c+1}, for free channel c (class c+1).
                    for c in 0..n_eta {
                        let p_c = (eta[[row, c]] - lse).exp();
                        let indicator = if label == c + 1 { 1.0 } else { 0.0 };
                        let r = w * (p_c - indicator);
                        let base = c * (1 + d);
                        grad_coeffs[base] += r;
                        for axis in 0..d {
                            grad_coeffs[base + 1 + axis] += r * t[[row, axis]];
                            grad_t[[row, axis]] += r * coeffs[base + 1 + axis];
                        }
                    }
                }
            }
        }
        Ok((nll, grad_coeffs, grad_t))
    }

    /// Per-row, per-channel head working weights `s[n, c]` — the diagonal of the
    /// Fisher information in η-space (`p(1−p)` logistic, `p_c(1−p_c)` softmax
    /// diagonal). These weight the score-influence directions the leakage
    /// absorber appends to the reconstruction design (the realized label-channel
    /// leverage per row), times the row's head weight so unlabeled rows
    /// contribute nothing to the absorber, exactly as they contribute nothing
    /// to the head likelihood.
    pub fn head_working_weights(
        &self,
        t: ArrayView2<'_, f64>,
        coeffs: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let (n, d) = t.dim();
        let n_eta = self.family.n_eta_channels();
        if coeffs.len() != n_eta * (1 + d) {
            return Err("BehavioralHead::head_working_weights: coeff length mismatch".to_string());
        }
        let eta = self.eta(t, coeffs);
        let mut s = Array2::<f64>::zeros((n, n_eta));
        match self.family {
            AuxOutcomeFamily::Binomial => {
                for row in 0..n {
                    let p = 1.0 / (1.0 + (-eta[[row, 0]]).exp());
                    s[[row, 0]] = self.w_row[row] * p * (1.0 - p);
                }
            }
            AuxOutcomeFamily::Multinomial { .. } => {
                for row in 0..n {
                    let mut max_eta = 0.0_f64;
                    for c in 0..n_eta {
                        if eta[[row, c]] > max_eta {
                            max_eta = eta[[row, c]];
                        }
                    }
                    let mut denom = (0.0 - max_eta).exp();
                    for c in 0..n_eta {
                        denom += (eta[[row, c]] - max_eta).exp();
                    }
                    let lse = max_eta + denom.ln();
                    for c in 0..n_eta {
                        let p_c = (eta[[row, c]] - lse).exp();
                        s[[row, c]] = self.w_row[row] * p_c * (1.0 - p_c);
                    }
                }
            }
        }
        Ok(s)
    }
}

/// The #461 Neyman-orthogonal leakage absorber for the behavioral head.
///
/// # The boundary it enforces
///
/// You want the behavioral channel to *orient* the existing manifold (fix the
/// frame) but *not* to *invent* geometry absent from `p(x)` (sculpt a manifold
/// to fit the label). The orthogonalization is precisely the boundary between
/// those two — between "orient what's there" and "hallucinate structure from
/// the label." Getting it exactly right is the single most important statistical
/// content of the whole construction.
///
/// # Mechanism (mirrors the survival/BMS install)
///
/// The head's score-influence directions in latent-code space are the rows of
/// the *score-influence Jacobian*
///
/// ```text
///   Z[n, :] = √s_n · ∂η_n/∂t_n = √s_n · w   (per η-channel)
/// ```
///
/// i.e. the realized, per-row, Fisher-weighted directions along which a change
/// in the latent codes moves the label-channel linear predictor. We
/// orthonormalize their span (thin QR) to obtain the *label-channel subspace*
/// `Q` in latent-code space. The dictionary (reconstruction) update is then
/// projected onto the orthogonal complement of `Q`:
///
/// ```text
///   Δt_recon  ←  (I − Q Qᵀ) Δt_recon
/// ```
///
/// so the reconstruction channel can only move the codes in directions the
/// label channel does *not* already explain. Equivalently, the reconstruction
/// design is widened with `Q` as a null-penalized absorbed block, making the
/// dictionary's estimating equation orthogonal to `span(Q)` — the label channel
/// orients the frame, but cannot drag the dictionary toward encoding the label.
#[derive(Debug, Clone)]
pub struct LeakageAbsorber {
    /// Orthonormal basis `Q ∈ ℝ^{d × r}` of the label-channel subspace in
    /// latent-code space (`r ≤ min(d, n_eta)`). The reconstruction update is
    /// projected onto `range(Q)^⊥`.
    q: Array2<f64>,
}

impl LeakageAbsorber {
    /// Build the absorber from the head's score-influence Jacobian.
    ///
    /// `score_influence` is `(n × (d · n_eta))`: for each row, the stacked
    /// per-channel Fisher-weighted direction `√s_{n,c} · w_c` flattened over
    /// channels. We take the *column* span (the directions in latent-code space
    /// the label channel is sensitive to), reduced to an orthonormal basis via
    /// thin SVD with a relative tolerance, so a rank-deficient or vacuous label
    /// channel yields a low-rank (possibly empty) `Q` and the absorber is a
    /// no-op — the honest behavior when the label pins little gauge.
    pub fn from_score_influence(
        score_influence: ArrayView2<'_, f64>,
        latent_dim: usize,
    ) -> Result<Self, String> {
        let (n, cols) = score_influence.dim();
        if latent_dim == 0 {
            return Ok(Self {
                q: Array2::<f64>::zeros((0, 0)),
            });
        }
        if cols % latent_dim != 0 {
            return Err(format!(
                "LeakageAbsorber: score_influence has {cols} columns, not a multiple of \
                 latent_dim {latent_dim}"
            ));
        }
        let n_eta = cols / latent_dim;
        // Accumulate the (d × d) latent-space Gram of the label-channel
        // directions: G = Σ_n Σ_c v_{n,c} v_{n,c}ᵀ where v_{n,c} ∈ ℝ^d is the
        // c-th channel's score-influence direction for row n.
        let mut gram = Array2::<f64>::zeros((latent_dim, latent_dim));
        for row in 0..n {
            for c in 0..n_eta {
                let base = c * latent_dim;
                for i in 0..latent_dim {
                    let vi = score_influence[[row, base + i]];
                    for j in 0..latent_dim {
                        gram[[i, j]] += vi * score_influence[[row, base + j]];
                    }
                }
            }
        }
        // Eigenvectors of G via SVD (G symmetric PSD) give the principal
        // label-channel directions; keep those above a relative tolerance.
        let (u_opt, sv, _vt) = gram
            .svd(true, false)
            .map_err(|e| format!("LeakageAbsorber: SVD of label-channel Gram failed: {e}"))?;
        let u = u_opt.ok_or_else(|| "LeakageAbsorber: SVD did not return U".to_string())?;
        let max_sv = sv.iter().cloned().fold(0.0_f64, f64::max);
        let tol = max_sv * (latent_dim as f64) * f64::EPSILON;
        let rank = sv.iter().filter(|&&s| s > tol).count();
        let mut q = Array2::<f64>::zeros((latent_dim, rank));
        for col in 0..rank {
            for r in 0..latent_dim {
                q[[r, col]] = u[[r, col]];
            }
        }
        Ok(Self { q })
    }

    /// Rank of the absorbed label-channel subspace (`r`). Zero ⇒ the absorber
    /// is a no-op (the label channel pins no direction the dictionary must be
    /// orthogonalized against).
    pub fn rank(&self) -> usize {
        self.q.ncols()
    }

    /// Orthonormal basis `Q` of the absorbed subspace (`d × r`).
    pub fn basis(&self) -> ArrayView2<'_, f64> {
        self.q.view()
    }

    /// Project a reconstruction-channel latent update `Δt` (`n × d`) onto the
    /// orthogonal complement of the label-channel subspace, in-place semantics
    /// returned as a fresh array. This is the operative orthogonalization: the
    /// dictionary update keeps only the component the label channel does not
    /// already explain.
    pub fn orthogonalize_recon_update(&self, delta_t: ArrayView2<'_, f64>) -> Array2<f64> {
        let (_, d) = delta_t.dim();
        if self.q.ncols() == 0 || self.q.nrows() != d {
            return delta_t.to_owned();
        }
        // Δt − (Δt Q) Qᵀ.
        let proj_coords = delta_t.dot(&self.q); // (n × r)
        let proj = proj_coords.dot(&self.q.t()); // (n × d)
        let mut out = delta_t.to_owned();
        out -= &proj;
        out
    }

    /// The absorbed block to *append* to the reconstruction design: per-row
    /// rows `Z_infl[n, :] = Δ`-direction in latent-code space, here the
    /// row's latent code projected onto `Q` — the realized label-channel
    /// leverage. Width `r`. This mirrors the survival/BMS install where the
    /// realized Stage-1 leakage directions are appended as a null-penalized
    /// absorbed block, making the β estimating equation orthogonal to its span.
    pub fn absorbed_design_block(&self, t: ArrayView2<'_, f64>) -> Array2<f64> {
        if self.q.ncols() == 0 {
            return Array2::<f64>::zeros((t.nrows(), 0));
        }
        t.dot(&self.q)
    }
}

/// Per-feature (per-atom) behavioral-significance report for the head.
#[derive(Debug, Clone)]
pub struct HeadFeatureSignificance {
    /// Wald statistic per latent axis (feature).
    pub statistic: Vec<f64>,
    /// Raw p-value per latent axis.
    pub p_value: Vec<f64>,
    /// Indices of features rejected by e-BH at the chosen FDR level — the
    /// features whose behavioral loading is statistically real after the
    /// multiplicity correction. These are the reportable behaviorally-anchored
    /// atoms.
    pub fdr_rejected: Vec<usize>,
    /// The FDR level the rejection set was computed at.
    pub alpha: f64,
}

/// Per-feature significance of the head's behavioral loadings, with FDR control.
///
/// For each latent axis (feature) `k`, the behavioral loading is the head
/// coefficient `w_{c,k}` across η-channels. We test the null `w_{·,k} = 0`
/// (the atom carries no behavioral signal) with [`wood_smooth_test`] on the
/// coefficient block for that axis, using the head's posterior covariance.
/// Raw p-values are first Bonferroni-combined across η-channels for each axis,
/// then converted to calibrated e-values with
/// [`log_e_from_p_calibrator`] and fed to [`e_benjamini_hochberg`] for an
/// FDR-controlled rejection set that needs no independence assumption across
/// atoms — exactly the case BH cannot legally handle when atoms share tokens.
///
/// `coeffs` is the fitted head coefficient vector (`n_eta · (1 + d)`),
/// `covariance` its posterior covariance of the same size, `latent_dim` is `d`,
/// `n_eta` the channel count, and `alpha` the target FDR.
pub fn head_feature_significance(
    coeffs: ArrayView1<'_, f64>,
    covariance: &Array2<f64>,
    latent_dim: usize,
    n_eta: usize,
    residual_df: f64,
    alpha: f64,
) -> Result<HeadFeatureSignificance, String> {
    let block = 1 + latent_dim;
    if coeffs.len() != n_eta * block {
        return Err(format!(
            "head_feature_significance: coeffs length {} != n_eta·(1+d) = {}",
            coeffs.len(),
            n_eta * block
        ));
    }
    if covariance.nrows() != coeffs.len() || covariance.ncols() != coeffs.len() {
        return Err(format!(
            "head_feature_significance: covariance must be {0}×{0}, got {1}×{2}",
            coeffs.len(),
            covariance.nrows(),
            covariance.ncols()
        ));
    }
    let beta = coeffs.to_owned();
    let mut statistic = Vec::with_capacity(latent_dim);
    let mut p_value = Vec::with_capacity(latent_dim);
    let mut log_e_values = Vec::with_capacity(latent_dim);
    for axis in 0..latent_dim {
        // Gather the coefficient indices for this axis across all channels:
        // for channel c the loading is at `c·block + 1 + axis`. wood_smooth_test
        // tests a contiguous range, so for the common Binomial (n_eta = 1) case
        // the range is a single coefficient; for multinomial we test each
        // channel's loading and combine the minimum p-value with Bonferroni.
        let mut best_p = 1.0_f64;
        let mut best_stat = 0.0_f64;
        let mut tested_channels = 0_usize;
        for c in 0..n_eta {
            let idx = c * block + 1 + axis;
            let input = SmoothTestInput {
                beta: beta.view(),
                covariance,
                influence_matrix: None,
                coeff_range: idx..idx + 1,
                edf: 1.0,
                nullspace_dim: 1,
                residual_df,
                scale: SmoothTestScale::Estimated,
            };
            if let Some(res) = wood_smooth_test(input) {
                tested_channels += 1;
                if res.p_value < best_p {
                    best_p = res.p_value;
                    best_stat = res.statistic;
                }
            }
        }
        let axis_p = if tested_channels > 0 {
            (best_p * tested_channels as f64).min(1.0)
        } else {
            1.0
        };
        statistic.push(best_stat);
        p_value.push(axis_p);
        // Numerical tail routines can underflow a mathematically positive
        // p-value to exactly zero; calibrate at the smallest positive f64 while
        // still reporting the Bonferroni-adjusted p-value above.
        let calibration_p = axis_p.max(f64::MIN_POSITIVE);
        log_e_values.push(log_e_from_p_calibrator(calibration_p).map_err(|err| {
            format!("head_feature_significance: invalid calibrated axis p-value: {err}")
        })?);
    }
    let fdr_rejected = e_benjamini_hochberg(&log_e_values, alpha);
    Ok(HeadFeatureSignificance {
        statistic,
        p_value,
        fdr_rejected,
        alpha,
    })
}

/// Reduce a per-row score-influence Jacobian `(n × d)` to a rank-truncated
/// orthonormal basis of its latent-code column span. Convenience used by the
/// absorber install site when the caller already has the `(n × d)` Jacobian
/// for a single η-channel (the common Binomial case). Returns `Q ∈ ℝ^{d × r}`,
/// `r` = numerical rank of the column span (rank-deficient inputs yield a
/// low-rank `Q`, so a vacuous label channel orthogonalizes against nothing).
pub fn orthonormal_span(jacobian: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let (n, d) = jacobian.dim();
    if d == 0 || n == 0 {
        return Ok(Array2::<f64>::zeros((d, 0)));
    }
    // The right singular vectors of `J` span its row space = the column span
    // of `Jᵀ` in ℝ^d. Equivalently the eigenvectors of the d × d Gram `JᵀJ`;
    // route through the SVD bridge for a stable, rank-truncated basis.
    let gram = jacobian.t().dot(&jacobian);
    let (u_opt, sv, _vt) = gram
        .svd(true, false)
        .map_err(|e| format!("orthonormal_span: SVD failed: {e}"))?;
    let u = u_opt.ok_or_else(|| "orthonormal_span: SVD did not return U".to_string())?;
    let max_sv = sv.iter().cloned().fold(0.0_f64, f64::max);
    let tol = max_sv * (d as f64) * f64::EPSILON;
    let rank = sv.iter().filter(|&&s| s > tol).count();
    let mut q = Array2::<f64>::zeros((d, rank));
    for col in 0..rank {
        for r in 0..d {
            q[[r, col]] = u[[r, col]];
        }
    }
    Ok(q)
}

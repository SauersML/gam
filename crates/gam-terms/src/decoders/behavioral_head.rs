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


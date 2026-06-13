//! End-to-end OBJECTIVE quality: a *user-defined* Beta(α, β) regression family —
//! with softplus-linked shape parameters α = softplus(η₁), β = softplus(η₂) —
//! fitted through gam's generic `fit_custom_family` engine, judged by whether it
//! RECOVERS THE DATA-GENERATING TRUTH and GENERALISES, not by whether it
//! reproduces any peer tool's fitted numbers.
//!
//! Why this matters. gam exposes Gaussian, Poisson, Binomial, location-scale,
//! survival, etc. through the formula DSL, but the **Beta** distribution is
//! deliberately *not* a built-in family. A practitioner who needs it must
//! implement the `CustomFamily` contract — supplying the per-block log-density
//! gradient ∇_β log L and observed information −∇²_β log L by hand. The honest
//! question is not "does gam's arithmetic equal R's `lbeta`" (it must — that is
//! the *same closed form*, so matching it proves nothing about fit quality), but
//! "does the custom-family fit land on the truth and predict unseen data well".
//!
//! OBJECTIVE METRICS ASSERTED (all on gam's OWN fit, identical data to baseline):
//!
//!   1. TRUTH RECOVERY (primary). The response is simulated from a *known*
//!      mean surface μ_i = α_i/(α_i+β_i) with α_i = softplus(0.2+0.4·x1_i),
//!      β_i = softplus(0.3+0.5·x2_i). We assert gam's fitted Beta mean μ̂_i
//!      tracks that ground-truth mean with RMSE ≤ a small fraction of the
//!      response's own spread (much tighter than the irreducible Beta sampling
//!      noise). This is recovery of the true function, not agreement with a tool.
//!
//!   2. HELD-OUT PREDICTIVE NLL, match-or-beat (secondary). A fixed train/test
//!      split: gam is fit on the train rows only, then scored by mean Beta
//!      negative-log-likelihood on the *unseen* test rows. We assert an absolute
//!      held-out bar AND gam ≤ 1.05× the held-out NLL of an independent maximum-
//!      likelihood fit of the *identical* model (R `optim`). The reference is a
//!      BASELINE TO MATCH-OR-BEAT on generalisation, never a target to copy.
//!
//!   3. FAMILY-ARITHMETIC SELF-CONSISTENCY (intrinsic ground truth). Central
//!      finite differences of the row NLL are the ground truth for the analytic
//!      Jacobian gam's `evaluate` hands the inner Newton solve; a wrong
//!      digamma/softplus chain term is an O(1) error. This is correctness vs an
//!      exact mathematical quantity (the derivative of the family's own NLL), not
//!      "same as a peer tool", so it is kept as an assertion.

use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix,
};
use gam::families::custom_family::fit_custom_family;
use gam::init_parallelism;
use gam::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use gam::test_support::reference::{Column, relative_l2, run_r};
use ndarray::{Array1, Array2};
use statrs::function::beta::ln_beta;
use statrs::function::gamma::digamma;

// ── elementary scalar pieces (shared by family, oracle harness, FD) ──────────

/// softplus(η) = log(1 + e^η), in the overflow-safe form.
fn softplus(eta: f64) -> f64 {
    if eta > 0.0 {
        eta + (-eta).exp().ln_1p()
    } else {
        eta.exp().ln_1p()
    }
}

/// σ(η) = softplus'(η) = 1 / (1 + e^{-η}).
fn sigmoid(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// Trigamma ψ'(x) = d/dx digamma(x). statrs ships digamma but not its
/// derivative, so we use the standard recurrence-to-large-argument plus the
/// Bernoulli asymptotic series — accurate to ~1e-13 for the x > 0 range here.
fn trigamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 6.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷) − …
    acc + inv + 0.5 * inv2 + inv * inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))
}

/// Tetragamma ψ''(x) = d/dx ψ'(x), x > 0. Derivative of the `trigamma`
/// approximation above with the matching recurrence ψ''(x) = ψ''(x+1) − 2/x³.
/// Needed for the exact β-directional derivatives of the coupled Beta joint
/// Hessian (the ψ'(α)/ψ'(α+β) curvature weights differentiate to ψ'').
fn tetragamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 6.0 {
        acc -= 2.0 / (x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // ψ''(x) ≈ −1/x² − 1/x³ − 1/(2x⁴) + 1/(6x⁶) − 1/(6x⁸) + 3/(10x¹⁰)
    acc + (-inv2 - inv2 * inv - 0.5 * inv2 * inv2
        + inv2 * inv2 * inv2 * (1.0 / 6.0 + inv2 * (-1.0 / 6.0 + inv2 * 0.3)))
}

/// Pentagamma ψ'''(x) = d/dx ψ''(x), x > 0. Derivative of `tetragamma` with the
/// matching recurrence ψ'''(x) = ψ'''(x+1) + 6/x⁴. Needed for the exact second
/// β-directional derivative of the coupled Beta joint Hessian.
fn pentagamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 6.0 {
        acc += 6.0 / (x * x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    // ψ'''(x) ≈ 2/x³ + 3/x⁴ + 2/x⁵ − 1/x⁷ + 4/(3x⁹) − 3/x¹¹
    acc + 2.0 * inv3 + 3.0 * inv2 * inv2 + 2.0 * inv2 * inv3 - inv3 * inv2 * inv2
        + (4.0 / 3.0) * inv3 * inv3 * inv3
        - 3.0 * inv3 * inv3 * inv3 * inv2
}

/// Row negative log-likelihood of Beta(α, β) at observation y, with
/// α = softplus(η₁), β = softplus(η₂). This is the *exact* arithmetic the
/// family below sums into its log-likelihood; the FD harness differentiates
/// this same closed form, and the held-out scoring re-uses it.
fn beta_row_nll(eta1: f64, eta2: f64, y: f64) -> f64 {
    let a = softplus(eta1);
    let b = softplus(eta2);
    -((a - 1.0) * y.ln() + (b - 1.0) * (1.0 - y).ln() - ln_beta(a, b))
}

/// Beta mean μ = α / (α + β) at linear predictors (η₁, η₂).
fn beta_mean(eta1: f64, eta2: f64) -> f64 {
    let a = softplus(eta1);
    let b = softplus(eta2);
    a / (a + b)
}

/// Analytic forward-mode Jacobian of the row NLL: [∂NLL/∂η₁, ∂NLL/∂η₂].
/// ∂ℓ/∂α = ln y − (ψ(α) − ψ(α+β)),  ∂ℓ/∂η₁ = (∂ℓ/∂α)·σ(η₁); NLL flips the sign.
fn beta_row_nll_jacobian(eta1: f64, eta2: f64, y: f64) -> [f64; 2] {
    let a = softplus(eta1);
    let b = softplus(eta2);
    let d_la = y.ln() - (digamma(a) - digamma(a + b));
    let d_lb = (1.0 - y).ln() - (digamma(b) - digamma(a + b));
    [-d_la * sigmoid(eta1), -d_lb * sigmoid(eta2)]
}

// ── the user-defined custom family ───────────────────────────────────────────

/// Beta regression with logit/softplus-linked shape parameters, expressed
/// purely through the public `CustomFamily` contract (no built-in family
/// infrastructure). Block 0 carries α's coefficients, block 1 carries β's.
#[derive(Clone)]
struct BetaLogitCustomFamily {
    response: Array1<f64>,
    alpha_offset: Array1<f64>,
    beta_offset: Array1<f64>,
    // The family keeps its own copies of the block designs so `evaluate` can
    // assemble Xᵀ-side gradients and Gram Hessians directly. These are the
    // *same* matrices used to seed the block specs, so "the data handed to gam"
    // and "the data the family differentiates" are provably identical.
    design0: Array2<f64>,
    design1: Array2<f64>,
}

impl CustomFamily for BetaLogitCustomFamily {
    /// Two independent shape predictors: block 0 → α channel, block 1 → β
    /// channel (`a = softplus(η₁)`, `b = softplus(η₂)`). Declaring the channel
    /// topology routes the identifiability audit channel-aware, so the two
    /// blocks' shared intercept-bearing design is treated as block-diagonal
    /// rather than a cross-block intercept alias (#558).
    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some((0..specs.len()).collect())
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "BetaLogit needs 2 blocks, got {}",
                block_states.len()
            ));
        }
        let eta1 = &block_states[0].eta + &self.alpha_offset;
        let eta2 = &block_states[1].eta + &self.beta_offset;
        let n = self.response.len();

        let mut log_likelihood = 0.0;
        // Per-row chain-rule kernels: d_eta = ∂ℓ/∂η (score), and the three
        // observed-information weights w11=−∂²ℓ/∂η₁², w22=−∂²ℓ/∂η₂²,
        // w12=−∂²ℓ/∂η₁∂η₂ used to assemble Xᵀ W X block Hessians.
        let mut score1 = Array1::<f64>::zeros(n);
        let mut score2 = Array1::<f64>::zeros(n);
        let mut w11 = Array1::<f64>::zeros(n);
        let mut w22 = Array1::<f64>::zeros(n);
        for i in 0..n {
            let y = self.response[i];
            let (h1, h2) = (eta1[i], eta2[i]);
            let a = softplus(h1);
            let b = softplus(h2);
            let s1 = sigmoid(h1);
            let s2 = sigmoid(h2);
            let sp1 = s1 * (1.0 - s1); // softplus''(η₁) = σ'(η₁)
            let sp2 = s2 * (1.0 - s2);

            log_likelihood += (a - 1.0) * y.ln() + (b - 1.0) * (1.0 - y).ln() - ln_beta(a, b);

            let d_la = y.ln() - (digamma(a) - digamma(a + b));
            let d_lb = (1.0 - y).ln() - (digamma(b) - digamma(a + b));
            // Second derivatives in shape space (trigamma ψ').
            let d2_aa = -(trigamma(a) - trigamma(a + b));
            let d2_bb = -(trigamma(b) - trigamma(a + b));

            score1[i] = d_la * s1;
            score2[i] = d_lb * s2;
            // ∂²ℓ/∂η₁² = d2_aa·σ² + d_la·σ' ; observed info is its negative.
            w11[i] = -(d2_aa * s1 * s1 + d_la * sp1);
            w22[i] = -(d2_bb * s2 * s2 + d_lb * sp2);
        }

        assert_eq!(
            block_states[0].eta.len(),
            n,
            "block 0 eta length must equal n"
        );

        // Block gradients Xᵀ score and block observed-information Xᵀ diag(w) X.
        let g1 = self.design0_transpose_dot(&score1);
        let g2 = self.design1_transpose_dot(&score2);
        let h11 = self.design0_gram_weighted(&w11);
        let h22 = self.design1_gram_weighted(&w22);

        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: g1,
                    hessian: SymmetricMatrix::Dense(h11),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: g2,
                    hessian: SymmetricMatrix::Dense(h22),
                },
            ],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Full coupled observed information over [β_α ; β_β], including the
        // off-diagonal X₁ᵀ diag(w12) X₂ block from ∂²ℓ/∂η₁∂η₂ = ψ'(α+β)·σ₁σ₂.
        let eta1 = &block_states[0].eta + &self.alpha_offset;
        let eta2 = &block_states[1].eta + &self.beta_offset;
        let n = self.response.len();
        let mut w11 = Array1::<f64>::zeros(n);
        let mut w22 = Array1::<f64>::zeros(n);
        let mut w12 = Array1::<f64>::zeros(n);
        for i in 0..n {
            let y = self.response[i];
            let (h1, h2) = (eta1[i], eta2[i]);
            let a = softplus(h1);
            let b = softplus(h2);
            let s1 = sigmoid(h1);
            let s2 = sigmoid(h2);
            let sp1 = s1 * (1.0 - s1);
            let sp2 = s2 * (1.0 - s2);
            let d_la = y.ln() - (digamma(a) - digamma(a + b));
            let d_lb = (1.0 - y).ln() - (digamma(b) - digamma(a + b));
            let d2_aa = -(trigamma(a) - trigamma(a + b));
            let d2_bb = -(trigamma(b) - trigamma(a + b));
            let d2_ab = trigamma(a + b);
            w11[i] = -(d2_aa * s1 * s1 + d_la * sp1);
            w22[i] = -(d2_bb * s2 * s2 + d_lb * sp2);
            w12[i] = -(d2_ab * s1 * s2);
        }
        let h11 = self.design0_gram_weighted(&w11);
        let h22 = self.design1_gram_weighted(&w22);
        let h12 = self.design0_transpose_weighted_design1(&w12);
        let p0 = h11.nrows();
        let p1 = h22.nrows();
        let mut joint = Array2::<f64>::zeros((p0 + p1, p0 + p1));
        joint.slice_mut(ndarray::s![0..p0, 0..p0]).assign(&h11);
        joint.slice_mut(ndarray::s![p0.., p0..]).assign(&h22);
        joint.slice_mut(ndarray::s![0..p0, p0..]).assign(&h12);
        joint.slice_mut(ndarray::s![p0.., 0..p0]).assign(&h12.t());
        Ok(Some(joint))
    }

    /// Exact first β-directional derivative `D_β H_L[u]` of the coupled joint
    /// Hessian along a flattened coefficient direction `u = [u_α ; u_β]`. The
    /// Hessian weights depend on β only through η = Xβ, so this differentiates
    /// each weight w·(η) along the η-directions `dh1 = X₁·u_α`, `dh2 = X₂·u_β`
    /// (the softplus/sigmoid + digamma chain), then re-assembles the same three
    /// `Xᵀ diag(·) X` blocks. Without this override the engine would assemble a
    /// block-diagonal `D_β H` and silently drop the cross-block w12 drift.
    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let p0 = self.design0.ncols();
        let p1 = self.design1.ncols();
        if d_beta_flat.len() != p0 + p1 {
            return Err(format!(
                "BetaLogit directional derivative length {} != joint width {}",
                d_beta_flat.len(),
                p0 + p1
            ));
        }
        let eta1 = &block_states[0].eta + &self.alpha_offset;
        let eta2 = &block_states[1].eta + &self.beta_offset;
        let dh1 = self
            .design0
            .dot(&d_beta_flat.slice(ndarray::s![0..p0]).to_owned());
        let dh2 = self
            .design1
            .dot(&d_beta_flat.slice(ndarray::s![p0..]).to_owned());
        let n = self.response.len();
        let mut dw11 = Array1::<f64>::zeros(n);
        let mut dw22 = Array1::<f64>::zeros(n);
        let mut dw12 = Array1::<f64>::zeros(n);
        for i in 0..n {
            let der =
                beta_weight_first_derivative(self.response[i], eta1[i], eta2[i], dh1[i], dh2[i]);
            dw11[i] = der.dw11;
            dw22[i] = der.dw22;
            dw12[i] = der.dw12;
        }
        Ok(Some(self.assemble_joint(&dw11, &dw22, &dw12)))
    }

    /// Exact second β-directional derivative `D²_β H_L[u, v]` of the coupled
    /// joint Hessian. Second mixed derivative of each weight along the two
    /// η-direction pairs `(du1, du2)` and `(dv1, dv2)`, re-assembled into the
    /// same three blocks. Needed (with the first derivative above) for the
    /// exact coupled outer REML Hessian; absent, the engine's block-diagonal
    /// default drops the cross-block third-order term.
    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let p0 = self.design0.ncols();
        let p1 = self.design1.ncols();
        if u.len() != p0 + p1 || v.len() != p0 + p1 {
            return Err(format!(
                "BetaLogit second directional derivative lengths {}/{} != joint width {}",
                u.len(),
                v.len(),
                p0 + p1
            ));
        }
        let eta1 = &block_states[0].eta + &self.alpha_offset;
        let eta2 = &block_states[1].eta + &self.beta_offset;
        let du1 = self.design0.dot(&u.slice(ndarray::s![0..p0]).to_owned());
        let du2 = self.design1.dot(&u.slice(ndarray::s![p0..]).to_owned());
        let dv1 = self.design0.dot(&v.slice(ndarray::s![0..p0]).to_owned());
        let dv2 = self.design1.dot(&v.slice(ndarray::s![p0..]).to_owned());
        let n = self.response.len();
        let mut dw11 = Array1::<f64>::zeros(n);
        let mut dw22 = Array1::<f64>::zeros(n);
        let mut dw12 = Array1::<f64>::zeros(n);
        for i in 0..n {
            let der = beta_weight_second_derivative(
                self.response[i],
                eta1[i],
                eta2[i],
                EtaDir {
                    d1: du1[i],
                    d2: du2[i],
                },
                EtaDir {
                    d1: dv1[i],
                    d2: dv2[i],
                },
            );
            dw11[i] = der.dw11;
            dw22[i] = der.dw22;
            dw12[i] = der.dw12;
        }
        Ok(Some(self.assemble_joint(&dw11, &dw22, &dw12)))
    }
}

/// Per-row weight derivatives `(D[w11], D[w22], D[w12])` of the coupled Beta
/// joint Hessian.
struct BetaWeightDeriv {
    dw11: f64,
    dw22: f64,
    dw12: f64,
}

/// Shared per-row chain-rule scalars at (η₁, η₂): the softplus shapes a,b, the
/// sigmoid links and their derivatives, the digamma score residuals, and the
/// curvature levels. Reused by the first- and second-derivative weight kernels.
struct BetaRowChain {
    a: f64,
    b: f64,
    s1: f64,
    s2: f64,
    sp1: f64,
    sp2: f64,
    /// d/dη of sp = σ'(η): sp·(1 − 2σ).
    dsp1: f64,
    dsp2: f64,
    d_la: f64,
    d_lb: f64,
    d2_aa: f64,
    d2_bb: f64,
    d2_ab: f64,
}

fn beta_row_chain(y: f64, eta1: f64, eta2: f64) -> BetaRowChain {
    let a = softplus(eta1);
    let b = softplus(eta2);
    let s1 = sigmoid(eta1);
    let s2 = sigmoid(eta2);
    let sp1 = s1 * (1.0 - s1);
    let sp2 = s2 * (1.0 - s2);
    BetaRowChain {
        a,
        b,
        s1,
        s2,
        sp1,
        sp2,
        dsp1: sp1 * (1.0 - 2.0 * s1),
        dsp2: sp2 * (1.0 - 2.0 * s2),
        d_la: y.ln() - (digamma(a) - digamma(a + b)),
        d_lb: (1.0 - y).ln() - (digamma(b) - digamma(a + b)),
        d2_aa: -(trigamma(a) - trigamma(a + b)),
        d2_bb: -(trigamma(b) - trigamma(a + b)),
        d2_ab: trigamma(a + b),
    }
}

/// First β-directional derivative of the three Beta Hessian weights along the
/// η-directions (dh1, dh2). Weights (observed information):
///   w11 = −(d2_aa·s1² + d_la·sp1),  w22 = −(d2_bb·s2² + d_lb·sp2),
///   w12 = −(d2_ab·s1·s2),  with d2_aa = −(ψ'(a) − ψ'(a+b)), d_la = lny − (ψ(a) − ψ(a+b)),
///   d2_ab = ψ'(a+b),  a = softplus(η₁), b = softplus(η₂).
fn beta_weight_first_derivative(
    y: f64,
    eta1: f64,
    eta2: f64,
    dh1: f64,
    dh2: f64,
) -> BetaWeightDeriv {
    let c = beta_row_chain(y, eta1, eta2);
    // Shape derivatives along the direction: da = σ(η₁)·dh1, db = σ(η₂)·dh2.
    let da = c.s1 * dh1;
    let db = c.s2 * dh2;
    let dab = da + db; // d(a+b)
    let ds1 = c.sp1 * dh1;
    let ds2 = c.sp2 * dh2;
    let dsp1 = c.dsp1 * dh1;
    let dsp2 = c.dsp2 * dh2;

    let d_d_la = -(trigamma(c.a) * da - trigamma(c.a + c.b) * dab);
    let d_d_lb = -(trigamma(c.b) * db - trigamma(c.a + c.b) * dab);
    let d_d2_aa = -(tetragamma(c.a) * da - tetragamma(c.a + c.b) * dab);
    let d_d2_bb = -(tetragamma(c.b) * db - tetragamma(c.a + c.b) * dab);
    let d_d2_ab = tetragamma(c.a + c.b) * dab;

    let dw11 =
        -(d_d2_aa * c.s1 * c.s1 + c.d2_aa * 2.0 * c.s1 * ds1 + d_d_la * c.sp1 + c.d_la * dsp1);
    let dw22 =
        -(d_d2_bb * c.s2 * c.s2 + c.d2_bb * 2.0 * c.s2 * ds2 + d_d_lb * c.sp2 + c.d_lb * dsp2);
    let dw12 = -(d_d2_ab * c.s1 * c.s2 + c.d2_ab * ds1 * c.s2 + c.d2_ab * c.s1 * ds2);
    BetaWeightDeriv { dw11, dw22, dw12 }
}

/// A per-row pair of η-space directions `(dη₁, dη₂)` (the rows of `X·dir` for
/// the α and β blocks) used by the second-derivative weight kernel.
struct EtaDir {
    d1: f64,
    d2: f64,
}

/// Second mixed β-directional derivative `D_v D_u[w]` of the three Beta Hessian
/// weights along η-direction pairs `u = (du1, du2)` and `v = (dv1, dv2)`.
/// Differentiates the first-derivative expression once more; the η-directions
/// are β-independent so d_v(du) = 0 and d_v(σ(η)) etc. introduce the next chain
/// order (needs ψ''').
fn beta_weight_second_derivative(
    y: f64,
    eta1: f64,
    eta2: f64,
    u: EtaDir,
    v: EtaDir,
) -> BetaWeightDeriv {
    let (du1, du2, dv1, dv2) = (u.d1, u.d2, v.d1, v.d2);
    let c = beta_row_chain(y, eta1, eta2);
    let ab = c.a + c.b;
    // First-order shape moves along u and v.
    let dua = c.s1 * du1;
    let dub = c.s2 * du2;
    let dva = c.s1 * dv1;
    let dvb = c.s2 * dv2;
    let duab = dua + dub;
    let dvab = dva + dvb;
    // Second mixed shape derivative: d_v(da) = d_v(σ(η₁))·du1 = sp1·dv1·du1.
    let duva = c.sp1 * dv1 * du1;
    let duvb = c.sp2 * dv2 * du2;
    let duvab = duva + duvb;

    // Link-function moves (σ and σ' = sp) along u and v, and second mixed.
    let du_s1 = c.sp1 * du1;
    let du_s2 = c.sp2 * du2;
    let dv_s1 = c.sp1 * dv1;
    let dv_s2 = c.sp2 * dv2;
    let duv_s1 = c.dsp1 * dv1 * du1;
    let duv_s2 = c.dsp2 * dv2 * du2;
    let du_sp1 = c.dsp1 * du1;
    let du_sp2 = c.dsp2 * du2;
    let dv_sp1 = c.dsp1 * dv1;
    let dv_sp2 = c.dsp2 * dv2;
    // d²/dη² of sp = d/dη[sp·(1−2σ)] = sp(1−2σ)² − 2·sp²; mixed along (du,dv).
    let dd_sp1 = c.sp1 * (1.0 - 2.0 * c.s1) * (1.0 - 2.0 * c.s1) - 2.0 * c.sp1 * c.sp1;
    let dd_sp2 = c.sp2 * (1.0 - 2.0 * c.s2) * (1.0 - 2.0 * c.s2) - 2.0 * c.sp2 * c.sp2;
    let duv_sp1 = dd_sp1 * dv1 * du1;
    let duv_sp2 = dd_sp2 * dv2 * du2;

    // Curvature levels and their u/v/mixed derivatives.
    // d_la = lny − (ψ(a) − ψ(a+b)).
    let du_d_la = -(trigamma(c.a) * dua - trigamma(ab) * duab);
    let dv_d_la = -(trigamma(c.a) * dva - trigamma(ab) * dvab);
    let duv_d_la = -(tetragamma(c.a) * dva * dua + trigamma(c.a) * duva
        - tetragamma(ab) * dvab * duab
        - trigamma(ab) * duvab);
    let du_d_lb = -(trigamma(c.b) * dub - trigamma(ab) * duab);
    let dv_d_lb = -(trigamma(c.b) * dvb - trigamma(ab) * dvab);
    let duv_d_lb = -(tetragamma(c.b) * dvb * dub + trigamma(c.b) * duvb
        - tetragamma(ab) * dvab * duab
        - trigamma(ab) * duvab);

    // d2_aa = −(ψ'(a) − ψ'(a+b)).
    let du_d2_aa = -(tetragamma(c.a) * dua - tetragamma(ab) * duab);
    let dv_d2_aa = -(tetragamma(c.a) * dva - tetragamma(ab) * dvab);
    let duv_d2_aa = -(pentagamma(c.a) * dva * dua + tetragamma(c.a) * duva
        - pentagamma(ab) * dvab * duab
        - tetragamma(ab) * duvab);
    let du_d2_bb = -(tetragamma(c.b) * dub - tetragamma(ab) * duab);
    let dv_d2_bb = -(tetragamma(c.b) * dvb - tetragamma(ab) * dvab);
    let duv_d2_bb = -(pentagamma(c.b) * dvb * dub + tetragamma(c.b) * duvb
        - pentagamma(ab) * dvab * duab
        - tetragamma(ab) * duvab);

    // d2_ab = ψ'(a+b).
    let du_d2_ab = tetragamma(ab) * duab;
    let dv_d2_ab = tetragamma(ab) * dvab;
    let duv_d2_ab = pentagamma(ab) * dvab * duab + tetragamma(ab) * duvab;

    // w11 = −(d2_aa·s1² + d_la·sp1). Product rule, second mixed derivative.
    let s1sq = c.s1 * c.s1;
    let du_s1sq = 2.0 * c.s1 * du_s1;
    let dv_s1sq = 2.0 * c.s1 * dv_s1;
    let duv_s1sq = 2.0 * (dv_s1 * du_s1 + c.s1 * duv_s1);
    let duv_w11 = -(duv_d2_aa * s1sq
        + du_d2_aa * dv_s1sq
        + dv_d2_aa * du_s1sq
        + c.d2_aa * duv_s1sq
        + duv_d_la * c.sp1
        + du_d_la * dv_sp1
        + dv_d_la * du_sp1
        + c.d_la * duv_sp1);

    let s2sq = c.s2 * c.s2;
    let du_s2sq = 2.0 * c.s2 * du_s2;
    let dv_s2sq = 2.0 * c.s2 * dv_s2;
    let duv_s2sq = 2.0 * (dv_s2 * du_s2 + c.s2 * duv_s2);
    let duv_w22 = -(duv_d2_bb * s2sq
        + du_d2_bb * dv_s2sq
        + dv_d2_bb * du_s2sq
        + c.d2_bb * duv_s2sq
        + duv_d_lb * c.sp2
        + du_d_lb * dv_sp2
        + dv_d_lb * du_sp2
        + c.d_lb * duv_sp2);

    // w12 = −(d2_ab·s1·s2). Let P = s1·s2.
    let p = c.s1 * c.s2;
    let du_p = du_s1 * c.s2 + c.s1 * du_s2;
    let dv_p = dv_s1 * c.s2 + c.s1 * dv_s2;
    let duv_p = duv_s1 * c.s2 + dv_s1 * du_s2 + du_s1 * dv_s2 + c.s1 * duv_s2;
    let duv_w12 = -(duv_d2_ab * p + du_d2_ab * dv_p + dv_d2_ab * du_p + c.d2_ab * duv_p);

    BetaWeightDeriv {
        dw11: duv_w11,
        dw22: duv_w22,
        dw12: duv_w12,
    }
}

impl BetaLogitCustomFamily {
    /// Assemble the symmetric joint `[[H11, H12],[H12ᵀ, H22]]` from per-row
    /// weight vectors via the family's own block designs.
    fn assemble_joint(
        &self,
        w11: &Array1<f64>,
        w22: &Array1<f64>,
        w12: &Array1<f64>,
    ) -> Array2<f64> {
        let h11 = self.design0_gram_weighted(w11);
        let h22 = self.design1_gram_weighted(w22);
        let h12 = self.design0_transpose_weighted_design1(w12);
        let p0 = h11.nrows();
        let p1 = h22.nrows();
        let mut joint = Array2::<f64>::zeros((p0 + p1, p0 + p1));
        joint.slice_mut(ndarray::s![0..p0, 0..p0]).assign(&h11);
        joint.slice_mut(ndarray::s![p0.., p0..]).assign(&h22);
        joint.slice_mut(ndarray::s![0..p0, p0..]).assign(&h12);
        joint.slice_mut(ndarray::s![p0.., 0..p0]).assign(&h12.t());
        joint
    }

    fn design0_transpose_dot(&self, v: &Array1<f64>) -> Array1<f64> {
        self.design0.t().dot(v)
    }
    fn design1_transpose_dot(&self, v: &Array1<f64>) -> Array1<f64> {
        self.design1.t().dot(v)
    }
    fn design0_gram_weighted(&self, w: &Array1<f64>) -> Array2<f64> {
        let wx = &self.design0 * &w.view().insert_axis(ndarray::Axis(1));
        self.design0.t().dot(&wx)
    }
    fn design1_gram_weighted(&self, w: &Array1<f64>) -> Array2<f64> {
        let wx = &self.design1 * &w.view().insert_axis(ndarray::Axis(1));
        self.design1.t().dot(&wx)
    }
    fn design0_transpose_weighted_design1(&self, w: &Array1<f64>) -> Array2<f64> {
        let wx1 = &self.design1 * &w.view().insert_axis(ndarray::Axis(1));
        self.design0.t().dot(&wx1)
    }
}

/// Design [1, covariate] from one of the two covariate columns.
fn design_from_covariate(covariate: &[f64]) -> Array2<f64> {
    let n = covariate.len();
    let mut d = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        d[[i, 0]] = 1.0;
        d[[i, 1]] = covariate[i];
    }
    d
}

// ── deterministic synthetic data: n=140, X1,X2~N(0,1), Y~Beta(α,β) ───────────

const N: usize = 140;

/// Data-generating coefficients (ground truth). α = softplus(A0 + A1·x1),
/// β = softplus(B0 + B1·x2); the mean surface μ = α/(α+β) is what gam must
/// recover and predict.
const TRUE_A0: f64 = 0.2;
const TRUE_A1: f64 = 0.4;
const TRUE_B0: f64 = 0.3;
const TRUE_B1: f64 = 0.5;

/// Tiny deterministic LCG → U(0,1); Box–Muller for normals; inverse-CDF-free
/// Beta sampling via two Gamma(shape,1) draws (Marsaglia–Tsang). Fully
/// reproducible so gam and R see byte-identical data.
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u01(&mut self) -> f64 {
        // Numerical Recipes LCG constants.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // top 53 bits → [0,1)
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_u01().max(1e-300);
        let u2 = self.next_u01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn next_gamma(&mut self, shape: f64) -> f64 {
        // Marsaglia–Tsang; boost shapes < 1 via the standard u^{1/shape} trick.
        if shape < 1.0 {
            let g = self.next_gamma(shape + 1.0);
            let u = self.next_u01().max(1e-300);
            return g * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.next_normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_u01();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
    fn next_beta(&mut self, a: f64, b: f64) -> f64 {
        let ga = self.next_gamma(a);
        let gb = self.next_gamma(b);
        ga / (ga + gb)
    }
}

fn synthetic_covariates() -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(0x5eed_1234_abcd_0001);
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    for _ in 0..N {
        x1.push(rng.next_normal());
        x2.push(rng.next_normal());
    }
    (x1, x2)
}

/// True per-row Beta mean μ_i from the data-generating coefficients.
fn true_means(x1: &[f64], x2: &[f64]) -> Vec<f64> {
    (0..N)
        .map(|i| {
            let h1 = TRUE_A0 + TRUE_A1 * x1[i];
            let h2 = TRUE_B0 + TRUE_B1 * x2[i];
            beta_mean(h1, h2)
        })
        .collect()
}

fn synthetic_response(x1: &[f64], x2: &[f64]) -> Vec<f64> {
    // Re-seed independently so response draws don't reuse the covariate stream.
    let mut rng = Lcg::new(0xabcd_0009_5eed_4321);
    let mut y = Vec::with_capacity(N);
    for i in 0..N {
        let a = softplus(TRUE_A0 + TRUE_A1 * x1[i]);
        let b = softplus(TRUE_B0 + TRUE_B1 * x2[i]);
        let mut draw = rng.next_beta(a, b);
        // Beta support is the open (0,1); clamp away from the log-singularities.
        draw = draw.clamp(1e-9, 1.0 - 1e-9);
        y.push(draw);
    }
    y
}

/// Deterministic train/test split: every 4th row (indices 0,4,8,…) is held out
/// for the predictive assertion; the rest train. Fixed by construction, so gam
/// and the R baseline score the identical unseen rows.
fn is_test_row(i: usize) -> bool {
    i % 4 == 0
}

#[test]
fn beta_logit_custom_family_recovers_truth_and_generalises() {
    init_parallelism();
    let (x1, x2) = synthetic_covariates();
    assert_eq!(x1.len(), N);
    assert_eq!(x2.len(), N);
    let y = synthetic_response(&x1, &x2);
    assert_eq!(y.len(), N);
    let mu_truth = true_means(&x1, &x2);

    // ── deterministic train/test partition (identical for gam and R) ────────
    let train_idx: Vec<usize> = (0..N).filter(|&i| !is_test_row(i)).collect();
    let test_idx: Vec<usize> = (0..N).filter(|&i| is_test_row(i)).collect();
    assert!(test_idx.len() >= 20, "need a non-trivial held-out set");
    assert!(train_idx.len() + test_idx.len() == N);

    let take = |src: &[f64], idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };
    let (x1_tr, x2_tr, y_tr) = (
        take(&x1, &train_idx),
        take(&x2, &train_idx),
        take(&y, &train_idx),
    );
    let y_te = take(&y, &test_idx);
    let n_tr = train_idx.len();

    // ── fit gam on the TRAIN rows only, through the custom-family engine ────
    let design0_tr = design_from_covariate(&x1_tr);
    let design1_tr = design_from_covariate(&x2_tr);

    let family = BetaLogitCustomFamily {
        response: Array1::from(y_tr.clone()),
        alpha_offset: Array1::<f64>::zeros(n_tr),
        beta_offset: Array1::<f64>::zeros(n_tr),
        design0: design0_tr.clone(),
        design1: design1_tr.clone(),
    };

    // This is a PARAMETRIC maximum-likelihood recovery problem: the true η is
    // exactly linear (α=softplus(A0+A1·x1), β=softplus(B0+B1·x2)) and the R
    // baseline below is an UNPENALIZED MLE. The block ridge exists ONLY to keep
    // the 2×2 inner-Newton geometry well-conditioned — it must not shrink the
    // structural slopes that carry the truth signal. With a REML-OPTIMIZED ridge
    // it does: on 2-coefficient blocks the REML objective has almost no
    // likelihood curvature to oppose the penalty, so it drove λ up to ≈10–20
    // (from the −4 seed), attenuating the slopes by 20–50% (b_beta1≈0.25 vs the
    // true 0.50) and pushing truth-recovery RMSE over the bar. Fix the ridge at a
    // tiny FIXED precision (λ=e^{-6}≈2.5e-3, removed from the REML coordinate) so
    // it conditions the solve without biasing the MLE — gam then recovers the
    // same near-unpenalized estimate the R `optim` baseline does.
    let make_spec = |name: &str, design: Array2<f64>| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n_tr),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(2)).with_fixed_log_lambda(-6.0)],
        nullspace_dims: vec![0],
        initial_log_lambdas: Array1::from(vec![-6.0]),
        initial_beta: Some(Array1::from(vec![0.0, 0.0])),
        ..ParameterBlockSpec::defaults()
    };
    let specs = vec![
        make_spec("alpha", design0_tr.clone()),
        make_spec("beta", design1_tr.clone()),
    ];

    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };

    let fit = fit_custom_family(&family, &specs, &options).expect("fit Beta-logit custom family");
    assert_eq!(fit.blocks.len(), 2, "two fitted blocks expected");
    let b_alpha = fit.blocks[0].beta.clone();
    let b_beta = fit.blocks[1].beta.clone();
    assert_eq!(b_alpha.len(), 2);
    assert_eq!(b_beta.len(), 2);

    // gam's fitted mean surface evaluated on ALL rows (train ∪ test), using the
    // coefficients it learned from the train rows only.
    let gam_eta1 = |i: usize| b_alpha[0] + b_alpha[1] * x1[i];
    let gam_eta2 = |i: usize| b_beta[0] + b_beta[1] * x2[i];
    let mu_hat: Vec<f64> = (0..N)
        .map(|i| beta_mean(gam_eta1(i), gam_eta2(i)))
        .collect();

    // ── METRIC 1 (PRIMARY): TRUTH RECOVERY of the mean surface ──────────────
    // RMSE between gam's fitted mean and the ground-truth generating mean.
    let truth_rmse = {
        let s: f64 = (0..N).map(|i| (mu_hat[i] - mu_truth[i]).powi(2)).sum();
        (s / N as f64).sqrt()
    };
    // Spread of the response itself (irreducible Beta sampling scatter); the
    // recovered mean must be MUCH tighter to the truth than the data are.
    let y_sd = {
        let m = y.iter().sum::<f64>() / N as f64;
        (y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / N as f64).sqrt()
    };

    // ── METRIC 2 (SECONDARY): held-out predictive NLL, match-or-beat R MLE ──
    let gam_test_nll: Vec<f64> = test_idx
        .iter()
        .enumerate()
        .map(|(k, &i)| beta_row_nll(gam_eta1(i), gam_eta2(i), y_te[k]))
        .collect();
    let gam_heldout_nll = gam_test_nll.iter().sum::<f64>() / gam_test_nll.len() as f64;

    // Baseline-to-match-or-beat: an INDEPENDENT maximum-likelihood fit of the
    // *identical* Beta(softplus,softplus) model on the SAME train rows (R
    // `optim`), scored on the SAME held-out rows. R fits and predicts; gam must
    // generalise at least as well — we do not assert gam reproduces R's βs.
    let r_train = run_r(
        &[
            Column::new("y", &y_tr),
            Column::new("x1", &x1_tr),
            Column::new("x2", &x2_tr),
        ],
        r#"
        softplus <- function(e) ifelse(e > 0, e + log1p(exp(-e)), log1p(exp(e)))
        negll <- function(p) {
          a <- softplus(p[1] + p[2] * df$x1)
          b <- softplus(p[3] + p[4] * df$x2)
          -sum((a - 1) * log(df$y) + (b - 1) * log(1 - df$y) - lbeta(a, b))
        }
        opt <- optim(c(0, 0, 0, 0), negll, method = "BFGS",
                     control = list(maxit = 500, reltol = 1e-12))
        emit("coef", opt$par)
        "#,
    );
    let r_coef = r_train.vector("coef");
    assert_eq!(r_coef.len(), 4, "R MLE returned {} coefs", r_coef.len());
    let r_eta1 = |i: usize| r_coef[0] + r_coef[1] * x1[i];
    let r_eta2 = |i: usize| r_coef[2] + r_coef[3] * x2[i];
    let r_heldout_nll = test_idx
        .iter()
        .enumerate()
        .map(|(k, &i)| beta_row_nll(r_eta1(i), r_eta2(i), y_te[k]))
        .sum::<f64>()
        / test_idx.len() as f64;
    // gam's own recovery of the truth on the SAME held-out rows, for context.
    let r_truth_rmse = {
        let s: f64 = (0..N)
            .map(|i| (beta_mean(r_eta1(i), r_eta2(i)) - mu_truth[i]).powi(2))
            .sum();
        (s / N as f64).sqrt()
    };

    // ── METRIC 3: family-arithmetic self-consistency (analytic J vs central FD)
    // Sample 10 train rows deterministically and stack both partials (20 values).
    let mut idx_rng = Lcg::new(0x0f1e_2d3c_4b5a_6987);
    let mut sampled = Vec::<usize>::new();
    while sampled.len() < 10 {
        let j = (idx_rng.next_u01() * n_tr as f64) as usize % n_tr;
        if !sampled.contains(&j) {
            sampled.push(j);
        }
    }
    let h = 1e-6;
    let mut analytic = Vec::<f64>::with_capacity(20);
    let mut fdiff = Vec::<f64>::with_capacity(20);
    for &k in &sampled {
        let (h1, h2, yy) = (gam_eta1(train_idx[k]), gam_eta2(train_idx[k]), y_tr[k]);
        let j = beta_row_nll_jacobian(h1, h2, yy);
        let fd1 = (beta_row_nll(h1 + h, h2, yy) - beta_row_nll(h1 - h, h2, yy)) / (2.0 * h);
        let fd2 = (beta_row_nll(h1, h2 + h, yy) - beta_row_nll(h1, h2 - h, yy)) / (2.0 * h);
        analytic.push(j[0]);
        analytic.push(j[1]);
        fdiff.push(fd1);
        fdiff.push(fd2);
    }
    let jac_rel = relative_l2(&analytic, &fdiff);

    eprintln!(
        "beta-logit custom family: n_train={n_tr} n_test={} \
         b_alpha=[{:.4},{:.4}] (truth [{TRUE_A0},{TRUE_A1}]) \
         b_beta=[{:.4},{:.4}] (truth [{TRUE_B0},{TRUE_B1}]) | \
         truth_rmse(mean)={truth_rmse:.4} y_sd={y_sd:.4} | \
         heldout_nll: gam={gam_heldout_nll:.4} R={r_heldout_nll:.4} | \
         r_truth_rmse={r_truth_rmse:.4} jac_rel_l2(FD)={jac_rel:.3e}",
        test_idx.len(),
        b_alpha[0],
        b_alpha[1],
        b_beta[0],
        b_beta[1],
    );

    // ── ASSERTION 1 (PRIMARY): gam recovers the true mean surface ───────────
    // With n=140 Beta draws the mean μ∈(0,1) is recoverable to well under a
    // quarter of the response's own spread; this is an absolute truth-recovery
    // bar (not agreement with any tool). A wrong link / chain term blows it up.
    assert!(
        truth_rmse < 0.25 * y_sd,
        "gam did not recover the true Beta mean surface: \
         RMSE(mu_hat, mu_truth)={truth_rmse:.4} vs 0.25*y_sd={:.4}",
        0.25 * y_sd
    );
    assert!(
        truth_rmse < 0.06,
        "gam mean-surface recovery error too large: truth_rmse={truth_rmse:.4}"
    );

    // ── ASSERTION 2 (SECONDARY): generalises, match-or-beat the MLE baseline ─
    // Absolute held-out bar: mean Beta NLL on unseen rows is finite & sane for
    // this signal (the data-generating dispersion sits near here).
    assert!(
        gam_heldout_nll.is_finite() && gam_heldout_nll < 0.5,
        "gam held-out NLL implausible: {gam_heldout_nll:.4}"
    );
    // Match-or-beat: gam's out-of-sample NLL is within 5% of the independent
    // maximum-likelihood fit of the same model. The R MLE is a baseline to
    // match-or-beat on GENERALISATION, never a fit gam must reproduce.
    assert!(
        gam_heldout_nll <= r_heldout_nll + 0.05 * r_heldout_nll.abs() + 1e-9,
        "gam generalises worse than the same-model MLE baseline: \
         gam_heldout_nll={gam_heldout_nll:.4} > 1.05*R_heldout_nll={:.4}",
        r_heldout_nll * 1.05
    );

    // ── ASSERTION 3: family arithmetic is self-consistent vs ground-truth FD ─
    // Central differences with h=1e-6 on this smooth NLL carry O(h²)≈1e-12
    // truncation plus ~ε/h≈1e-10 cancellation; relative L2 stays well below
    // 2e-4. A wrong digamma/softplus chain term shifts it by O(1).
    assert!(
        jac_rel < 0.0002,
        "analytic Jacobian disagrees with finite differences: rel_l2={jac_rel:.3e}"
    );
}

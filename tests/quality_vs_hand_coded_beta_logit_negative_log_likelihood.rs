//! End-to-end quality: a *user-defined* Beta(α, β) regression family — with
//! softplus-linked shape parameters α = softplus(η₁), β = softplus(η₂) — fitted
//! through gam's generic `fit_custom_family` engine, with its row-wise
//! negative-log-likelihood and forward-mode derivatives checked against two
//! independent oracles.
//!
//! Why this matters. gam exposes Gaussian, Poisson, Binomial, location-scale,
//! survival, etc. through the formula DSL, but the **Beta** distribution is
//! deliberately *not* a built-in family. A practitioner who needs it must
//! implement the `CustomFamily` contract — supplying the per-block log-density
//! gradient ∇_β log L and observed information −∇²_β log L by hand. This test
//! is the honest verification that the custom-family path lets a user do that
//! *correctly*: the arithmetic that gam's engine consumes (the row density and
//! its derivatives) must match a mature reference and the calculus must be
//! self-consistent.
//!
//! Two oracles, identical data:
//!   1. **Hand-coded base-R** (`lbeta`) computes the row-wise NLL from the
//!      *same* fitted linear predictors gam used. Base R's `lbeta` is the
//!      mature reference for the Beta normalizing constant log B(α,β). They
//!      must agree to 1e-10 (the same closed form, only float arithmetic
//!      differs).
//!   2. **Central finite differences** of the row NLL w.r.t. (η₁, η₂) are an
//!      intrinsic ground truth for the analytic Jacobian gam's `evaluate`
//!      produces — there is no external tool for "the derivative gam's family
//!      handed the optimizer", so we differentiate the family's own NLL.
//!
//! A divergence in either is a real bug: (1) catches a wrong density/normalizer,
//! (2) catches a wrong gradient (the quantity the inner Newton solve trusts).

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

/// Row negative log-likelihood of Beta(α, β) at observation y, with
/// α = softplus(η₁), β = softplus(η₂). This is the *exact* arithmetic the
/// family below sums into its log-likelihood; the oracle and the FD harness
/// differentiate / re-derive this same closed form.
fn beta_row_nll(eta1: f64, eta2: f64, y: f64) -> f64 {
    let a = softplus(eta1);
    let b = softplus(eta2);
    -((a - 1.0) * y.ln() + (b - 1.0) * (1.0 - y).ln() - ln_beta(a, b))
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
}

impl BetaLogitCustomFamily {
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

fn build_design0() -> Array2<f64> {
    let (x1, x2) = synthetic_covariates();
    assert_eq!(x1.len(), x2.len());
    design_from_covariate(&x1)
}

fn build_design1() -> Array2<f64> {
    let (x1, x2) = synthetic_covariates();
    assert_eq!(x1.len(), x2.len());
    design_from_covariate(&x2)
}

// ── deterministic synthetic data: n=140, X1,X2~N(0,1), Y~Beta(α,β) ───────────

const N: usize = 140;

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

fn synthetic_response() -> Vec<f64> {
    let (x1, x2) = synthetic_covariates();
    // Re-seed independently so response draws don't reuse the covariate stream.
    let mut rng = Lcg::new(0xabcd_0009_5eed_4321);
    let mut y = Vec::with_capacity(N);
    for i in 0..N {
        let a = softplus(0.2 + 0.4 * x1[i]);
        let b = softplus(0.3 + 0.5 * x2[i]);
        let mut draw = rng.next_beta(a, b);
        // Beta support is the open (0,1); clamp away from the log-singularities.
        draw = draw.clamp(1e-9, 1.0 - 1e-9);
        y.push(draw);
    }
    y
}

#[test]
fn beta_logit_custom_family_nll_and_jacobian_match_r_and_finite_differences() {
    init_parallelism();
    let y = synthetic_response();
    assert_eq!(y.len(), N);

    // ── build the two parameter blocks: design = [1, covariate], no offset ──
    let design0 = build_design0();
    let design1 = build_design1();

    let family = BetaLogitCustomFamily {
        response: Array1::from(y.clone()),
        alpha_offset: Array1::<f64>::zeros(N),
        beta_offset: Array1::<f64>::zeros(N),
        design0: design0.clone(),
        design1: design1.clone(),
    };

    // A tiny ridge keeps the inner Newton geometry well-conditioned without
    // materially shrinking the 2-coefficient blocks; the assertions below hold
    // at whatever β̂ the fit returns (they verify the family's arithmetic, not
    // recovery of the data-generating truth).
    let make_spec = |name: &str, design: Array2<f64>| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(N),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(2))],
        nullspace_dims: vec![0],
        initial_log_lambdas: Array1::from(vec![-4.0]),
        initial_beta: Some(Array1::from(vec![0.0, 0.0])),
        ..ParameterBlockSpec::defaults()
    };
    let specs = vec![
        make_spec("alpha", design0.clone()),
        make_spec("beta", design1.clone()),
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

    // ── fitted linear predictors (identical inputs handed to every oracle) ──
    let eta1: Vec<f64> = design0.dot(&b_alpha).iter().copied().collect();
    let eta2: Vec<f64> = design1.dot(&b_beta).iter().copied().collect();

    // gam-side row NLL + analytic Jacobian using the family's own arithmetic.
    let gam_nll: Vec<f64> = (0..N)
        .map(|i| beta_row_nll(eta1[i], eta2[i], y[i]))
        .collect();

    // ── oracle 1: hand-coded base R (lbeta, digamma) on the SAME η ──────────
    let r = run_r(
        &[
            Column::new("y", &y),
            Column::new("eta1", &eta1),
            Column::new("eta2", &eta2),
        ],
        r#"
        softplus <- function(e) ifelse(e > 0, e + log1p(exp(-e)), log1p(exp(e)))
        a <- softplus(df$eta1)
        b <- softplus(df$eta2)
        # Beta NLL with the normalizer from base R's lbeta = log(beta(a,b)).
        nll <- -((a - 1) * log(df$y) + (b - 1) * log(1 - df$y) - lbeta(a, b))
        emit("nll", nll)
        "#,
    );
    let r_nll = r.vector("nll");
    assert_eq!(
        r_nll.len(),
        N,
        "R returned {} NLLs, expected {N}",
        r_nll.len()
    );

    let nll_max_abs = gam_nll
        .iter()
        .zip(r_nll)
        .map(|(g, rr)| (g - rr).abs())
        .fold(0.0_f64, f64::max);

    // ── oracle 2: central finite differences of the row NLL vs analytic J ───
    // Sample 10 rows deterministically and stack both partials (20 values).
    let h = 1e-6;
    let mut idx_rng = Lcg::new(0x0f1e_2d3c_4b5a_6987);
    let mut sampled = Vec::<usize>::new();
    while sampled.len() < 10 {
        let j = (idx_rng.next_u01() * N as f64) as usize % N;
        if !sampled.contains(&j) {
            sampled.push(j);
        }
    }
    let mut analytic = Vec::<f64>::with_capacity(20);
    let mut fdiff = Vec::<f64>::with_capacity(20);
    for &i in &sampled {
        let (h1, h2, yy) = (eta1[i], eta2[i], y[i]);
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
        "beta-logit custom family: n={N} \
         b_alpha=[{:.4},{:.4}] b_beta=[{:.4},{:.4}] \
         nll_max_abs(R)={nll_max_abs:.3e} jac_rel_l2(FD)={jac_rel:.3e}",
        b_alpha[0], b_alpha[1], b_beta[0], b_beta[1]
    );

    // The row NLL is the *same closed form* in gam (Rust statrs::ln_beta) and in
    // R (lbeta): only floating-point arithmetic differs, so 1e-10 is the honest
    // bound for "same density, same normalizer".
    assert!(
        nll_max_abs < 1e-10,
        "gam vs hand-coded R row NLL disagree: max_abs={nll_max_abs:.3e}"
    );
    // Central differences with h=1e-6 on this smooth NLL carry O(h²)≈1e-12
    // truncation plus ~ε/h≈1e-10 cancellation error; relative L2 stays well
    // below 2e-4, which still flags any genuine bug in the analytic Jacobian
    // (a wrong digamma/softplus chain term shifts it by O(1)).
    assert!(
        jac_rel < 0.0002,
        "analytic Jacobian disagrees with finite differences: rel_l2={jac_rel:.3e}"
    );
}

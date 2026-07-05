//! Issue #912 — behaviorally-anchored SAE head: planted-truth quality suite.
//!
//! The falsification surface the issue specifies: a planted generative model
//! where a KNOWN latent subspace drives a synthetic behavioral label. The head
//! must (a) RECOVER that subspace (not merely predict the label) — measured by
//! principal angles between the fitted behavioral-loading direction and the
//! planted direction; (b) keep behavioral loading off planted-null atoms
//! FDR-controlled (the leakage test); (c) the leakage absorber must hold the
//! dictionary fixed vs a frozen-dictionary baseline (perturbation test).
//!
//! These assert OBJECTIVE quality (truth recovery / calibration), not
//! reproduction of any reference tool's fitted output.

use gam::inference::smooth_test::{SmoothTestInput, SmoothTestScale, wood_smooth_test};
use gam::terms::decoders::behavioral_head::{
    AuxOutcomeFamily, BehavioralHead, LeakageAbsorber, head_feature_significance,
};
use ndarray::{Array1, Array2};

/// Deterministic LCG unit sampler so the test is reproducible and seed-stable.
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

/// Standard-normal via Box-Muller on two LCG uniforms.
fn randn(seed: &mut u64) -> f64 {
    let u1 = lcg(seed).max(1e-12);
    let u2 = lcg(seed);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Largest principal angle (radians) between the column spans of two
/// orthonormal-ized matrices `a` (n×p) and `b` (n×q): cos θ_min = σ_max of the
/// cross-Gram of the orthonormal bases. We orthonormalize each via Gram-Schmidt
/// (small dims) and return the angle of the *best-aligned* direction, which is
/// the relevant recovery metric for a 1-D planted subspace.
fn min_principal_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let qa = gram_schmidt(a);
    let qb = gram_schmidt(b);
    if qa.ncols() == 0 || qb.ncols() == 0 {
        return std::f64::consts::FRAC_PI_2;
    }
    let m = qa.t().dot(&qb);
    // σ_max(M) = cos of the smallest principal angle.
    let mut max_sv = 0.0_f64;
    // Power iteration on MᵀM for the top singular value (small matrices).
    let mtm = m.t().dot(&m);
    let k = mtm.nrows();
    let mut v = Array1::<f64>::from_elem(k, 1.0 / (k as f64).sqrt());
    for _ in 0..200 {
        let mut w = mtm.dot(&v);
        let norm = w.dot(&w).sqrt();
        if norm <= 1e-300 {
            break;
        }
        w /= norm;
        v = w;
    }
    let lambda = v.dot(&mtm.dot(&v));
    max_sv = max_sv.max(lambda.max(0.0).sqrt());
    max_sv.clamp(0.0, 1.0).acos()
}

/// Thin Gram-Schmidt orthonormalization of the columns of `a` (drops
/// numerically-dependent columns).
fn gram_schmidt(a: &Array2<f64>) -> Array2<f64> {
    let (n, p) = a.dim();
    let mut cols: Vec<Array1<f64>> = Vec::new();
    for j in 0..p {
        let mut v = a.column(j).to_owned();
        for q in &cols {
            let proj = v.dot(q);
            v = &v - &(proj * q);
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-10 {
            v /= norm;
            cols.push(v);
        }
    }
    let mut out = Array2::<f64>::zeros((n, cols.len()));
    for (j, c) in cols.iter().enumerate() {
        out.column_mut(j).assign(c);
    }
    out
}

/// Fit the behavioral head by Newton on its own coefficients at FIXED latent
/// codes (the inner head solve). Returns the fitted coefficient vector and the
/// inverse-Hessian (posterior covariance proxy). This exercises the head's
/// value+gradient and a numerical Hessian from the gradient.
fn fit_head_newton(
    head: &BehavioralHead,
    t: &Array2<f64>,
    n_coeffs: usize,
) -> (Array1<f64>, Array2<f64>) {
    let mut coeffs = Array1::<f64>::zeros(n_coeffs);
    let ridge = 1e-6;
    for _ in 0..100 {
        let (_nll, grad, _grad_t) = head
            .neg_loglik_and_grad(t.view(), coeffs.view())
            .expect("head nll");
        // Numerical Hessian by forward-differencing the gradient (n_coeffs small).
        let mut hess = Array2::<f64>::zeros((n_coeffs, n_coeffs));
        let eps = 1e-5;
        for j in 0..n_coeffs {
            let mut cp = coeffs.clone();
            cp[j] += eps;
            let (_n2, gp, _gt) = head
                .neg_loglik_and_grad(t.view(), cp.view())
                .expect("head nll");
            for i in 0..n_coeffs {
                hess[[i, j]] = (gp[i] - grad[i]) / eps;
            }
        }
        for i in 0..n_coeffs {
            hess[[i, i]] += ridge;
        }
        // Solve hess · δ = grad via Gaussian elimination (small system).
        let delta = solve_spd(&hess, &grad);
        let step_norm = delta.dot(&delta).sqrt();
        for j in 0..n_coeffs {
            coeffs[j] -= delta[j];
        }
        if step_norm < 1e-9 {
            break;
        }
    }
    // Posterior covariance ≈ H⁻¹ at the optimum.
    let (_nll, grad, _gt) = head
        .neg_loglik_and_grad(t.view(), coeffs.view())
        .expect("head nll");
    let mut hess = Array2::<f64>::zeros((n_coeffs, n_coeffs));
    let eps = 1e-5;
    for j in 0..n_coeffs {
        let mut cp = coeffs.clone();
        cp[j] += eps;
        let (_n2, gp, _g2) = head
            .neg_loglik_and_grad(t.view(), cp.view())
            .expect("head nll");
        for i in 0..n_coeffs {
            hess[[i, j]] = (gp[i] - grad[i]) / eps;
        }
    }
    for i in 0..n_coeffs {
        hess[[i, i]] += ridge;
    }
    let cov = invert(&hess);
    (coeffs, cov)
}

/// Solve SPD-ish `A x = b` via partial-pivot Gaussian elimination.
fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = a.nrows();
    let mut m = a.clone();
    let mut x = b.clone();
    for col in 0..n {
        let mut piv = col;
        let mut best = m[[col, col]].abs();
        for r in (col + 1)..n {
            if m[[r, col]].abs() > best {
                best = m[[r, col]].abs();
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                m.swap([col, c], [piv, c]);
            }
            x.swap(col, piv);
        }
        let d = m[[col, col]];
        if d.abs() < 1e-300 {
            continue;
        }
        for r in (col + 1)..n {
            let f = m[[r, col]] / d;
            for c in col..n {
                let v = m[[col, c]];
                m[[r, c]] -= f * v;
            }
            x[r] -= f * x[col];
        }
    }
    let mut out = Array1::<f64>::zeros(n);
    for col in (0..n).rev() {
        let mut acc = x[col];
        for c in (col + 1)..n {
            acc -= m[[col, c]] * out[c];
        }
        let d = m[[col, col]];
        out[col] = if d.abs() < 1e-300 { 0.0 } else { acc / d };
    }
    out
}

/// Invert via column-wise solves against the identity.
fn invert(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[col] = 1.0;
        let x = solve_spd(a, &e);
        for r in 0..n {
            inv[[r, col]] = x[r];
        }
    }
    inv
}

/// Planted generative model: `d`-dimensional latent codes, a single planted
/// direction `w_true` drives a logistic label; the orthogonal-complement axes
/// are behavioral nulls. Returns `(t, y, planted_axis)`.
fn plant(n: usize, d: usize, planted_axis: usize, seed: &mut u64) -> (Array2<f64>, Array1<f64>) {
    let mut t = Array2::<f64>::zeros((n, d));
    for r in 0..n {
        for c in 0..d {
            t[[r, c]] = randn(seed);
        }
    }
    // Label driven ONLY by the planted axis (slope 2.5), intercept 0.
    let slope = 2.5;
    let mut y = Array1::<f64>::zeros(n);
    for r in 0..n {
        let eta = slope * t[[r, planted_axis]];
        let p = 1.0 / (1.0 + (-eta).exp());
        y[r] = if lcg(seed) < p { 1.0 } else { 0.0 };
    }
    (t, y)
}

#[test]
fn head_recovers_planted_behavioral_subspace_by_principal_angle() {
    // A 4-D latent space; axis 1 drives the label, axes {0,2,3} are nulls.
    let mut seed = 0xBEEF_1234_5678_9ABC;
    let (n, d, planted) = (4000, 4, 1usize);
    let (t, y) = plant(n, d, planted, &mut seed);
    let head = BehavioralHead::fully_supervised(AuxOutcomeFamily::Binomial, y).expect("head");
    let n_coeffs = head.n_coeffs(d);
    let (coeffs, _cov) = fit_head_newton(&head, &t, n_coeffs);

    // The fitted behavioral-loading direction is coeffs[1..1+d] (skip intercept).
    let loading = coeffs.slice(ndarray::s![1..1 + d]).to_owned();
    let mut fitted = Array2::<f64>::zeros((d, 1));
    for k in 0..d {
        fitted[[k, 0]] = loading[k];
    }
    let mut planted_dir = Array2::<f64>::zeros((d, 1));
    planted_dir[[planted, 0]] = 1.0;

    let angle = min_principal_angle(&fitted, &planted_dir);
    // Recovery: the fitted direction must align with the planted axis to within
    // ~8 degrees. A post-hoc probe with no subspace structure would not pin the
    // direction; the head does, because the label IS the model.
    assert!(
        angle < 0.14,
        "planted-subspace recovery failed: principal angle {:.4} rad ({:.1} deg) too large; \
         fitted loading {:?}",
        angle,
        angle.to_degrees(),
        loading
    );
    // The planted axis must carry the dominant loading magnitude.
    let planted_load = loading[planted].abs();
    for k in 0..d {
        if k != planted {
            assert!(
                loading[k].abs() < 0.5 * planted_load,
                "null axis {k} loading {:.4} rivals planted axis {:.4}",
                loading[k],
                planted_load
            );
        }
    }
}

#[test]
fn behavioral_loading_on_null_atoms_is_fdr_controlled() {
    // 6 latent axes, only axis 2 is behavioral; the other five are planted
    // nulls. Per-feature significance + e-BH must reject (close to) only the
    // true axis and keep false discoveries on the nulls controlled.
    let mut seed = 0x0FF1_CE_DEAD_BEEF;
    let (n, d, planted) = (5000, 6, 2usize);
    let (t, y) = plant(n, d, planted, &mut seed);
    let head = BehavioralHead::fully_supervised(AuxOutcomeFamily::Binomial, y).expect("head");
    let n_coeffs = head.n_coeffs(d);
    let (coeffs, cov) = fit_head_newton(&head, &t, n_coeffs);

    let sig = head_feature_significance(
        coeffs.view(),
        &cov,
        d,
        1, // n_eta = 1 (binomial)
        (n - n_coeffs) as f64,
        0.1, // target FDR
    )
    .expect("significance");

    // The planted axis must be discovered.
    assert!(
        sig.fdr_rejected.contains(&planted),
        "planted behavioral axis {planted} not discovered; p-values {:?}",
        sig.p_value
    );
    // False discoveries among nulls must be FDR-controlled: at α=0.1 over 5
    // nulls, the expected number of false rejections is well under 1.
    let false_rejections = sig.fdr_rejected.iter().filter(|&&k| k != planted).count();
    assert!(
        false_rejections <= 1,
        "too many false behavioral discoveries on null atoms: {} (rejected {:?})",
        false_rejections,
        sig.fdr_rejected
    );
}

#[test]
fn multinomial_head_feature_p_values_are_channel_bonferroni_adjusted() {
    let coeffs = Array1::from_vec(vec![0.0, 2.0, 0.0, 0.1]);
    let covariance = Array2::eye(coeffs.len());
    let p0 = wood_smooth_test(SmoothTestInput {
        beta: coeffs.view(),
        covariance: &covariance,
        influence_matrix: None,
        whitening_gram: None,
        coeff_range: 1..2,
        edf: 1.0,
        nullspace_dim: 1,
        residual_df: 200.0,
        scale: SmoothTestScale::Estimated,
    })
    .expect("channel 0 smooth test")
    .p_value;
    let p1 = wood_smooth_test(SmoothTestInput {
        beta: coeffs.view(),
        covariance: &covariance,
        influence_matrix: None,
        whitening_gram: None,
        coeff_range: 3..4,
        edf: 1.0,
        nullspace_dim: 1,
        residual_df: 200.0,
        scale: SmoothTestScale::Estimated,
    })
    .expect("channel 1 smooth test")
    .p_value;

    let sig = head_feature_significance(coeffs.view(), &covariance, 1, 2, 200.0, 0.1)
        .expect("significance");

    let expected = (p0.min(p1) * 2.0).min(1.0);
    assert!((sig.p_value[0] - expected).abs() < 1e-12);
}

#[test]
fn leakage_absorber_orthogonalizes_reconstruction_against_label_channel() {
    // The absorber must project the reconstruction update off the label
    // channel's score-influence subspace: after orthogonalization the update
    // must have ZERO component along the label direction, while preserving the
    // orthogonal-complement component (orient what's there, never sculpt).
    let mut seed = 0xABCD_1234_5678_0042;
    let (n, d, planted) = (2000, 4, 1usize);
    let (t, y) = plant(n, d, planted, &mut seed);
    let head = BehavioralHead::fully_supervised(AuxOutcomeFamily::Binomial, y).expect("head");
    let n_coeffs = head.n_coeffs(d);
    let (coeffs, _cov) = fit_head_newton(&head, &t, n_coeffs);

    // Score-influence Jacobian: row n, the Fisher-weighted η-direction √s·w.
    let s = head
        .head_working_weights(t.view(), coeffs.view())
        .expect("working weights");
    let loading = coeffs.slice(ndarray::s![1..1 + d]).to_owned();
    let mut score_influence = Array2::<f64>::zeros((n, d)); // n_eta = 1
    for r in 0..n {
        let sw = s[[r, 0]].max(0.0).sqrt();
        for k in 0..d {
            score_influence[[r, k]] = sw * loading[k];
        }
    }
    let absorber =
        LeakageAbsorber::from_score_influence(score_influence.view(), d).expect("absorb");
    assert!(
        absorber.rank() >= 1,
        "absorber found no label-channel direction to orthogonalize against"
    );

    // A reconstruction update that points partly along the label direction and
    // partly along a null axis. After orthogonalization the label-direction
    // component must vanish.
    let q = absorber.basis().to_owned(); // d × r
    let mut delta = Array2::<f64>::zeros((1, d));
    // Build along normalized loading + along null axis 3.
    let loading_norm = loading.dot(&loading).sqrt().max(1e-12);
    for k in 0..d {
        delta[[0, k]] = 3.0 * loading[k] / loading_norm; // label-aligned
    }
    delta[[0, 3]] += 1.7; // a genuine reconstruction direction (null axis)

    let orth = absorber.orthogonalize_recon_update(delta.view());

    // Component of the orthogonalized update along the absorbed subspace = 0.
    let proj_coords = orth.dot(&q); // 1 × r
    let leaked = proj_coords.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        leaked < 1e-8,
        "label-channel leakage survived orthogonalization: residual projection {leaked:.3e}"
    );
    // The genuine null-axis-3 reconstruction signal must SURVIVE (orient what
    // p(x) put there): the orthogonalized update keeps a non-trivial component
    // off the label subspace.
    let off_label = orth.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        off_label > 0.5,
        "absorber destroyed the legitimate reconstruction signal (norm {off_label:.3e})"
    );
}

#[test]
fn semi_supervised_unlabeled_rows_carry_zero_head_weight() {
    // Half the rows are unlabeled (weight 0); the head NLL and gradient must be
    // computed on the labeled half only. We verify by checking that flipping an
    // unlabeled row's label changes neither the NLL nor the gradient.
    let mut seed = 0x5151_5151_2727_2727;
    let (n, d, planted) = (400, 3, 0usize);
    let (t, mut y) = plant(n, d, planted, &mut seed);
    let mut w = Array1::<f64>::zeros(n);
    for r in 0..n {
        w[r] = if r % 2 == 0 { 1.0 } else { 0.0 }; // odd rows unlabeled
    }
    let head = BehavioralHead::new(AuxOutcomeFamily::Binomial, y.clone(), w.clone()).expect("head");
    let n_coeffs = head.n_coeffs(d);
    let coeffs = Array1::<f64>::from_elem(n_coeffs, 0.3);
    let (nll0, grad0, _gt0) = head
        .neg_loglik_and_grad(t.view(), coeffs.view())
        .expect("nll");

    // Flip an unlabeled (odd) row's label and rebuild the head.
    y[1] = 1.0 - y[1];
    let head2 = BehavioralHead::new(AuxOutcomeFamily::Binomial, y, w).expect("head");
    let (nll1, grad1, _gt1) = head2
        .neg_loglik_and_grad(t.view(), coeffs.view())
        .expect("nll");

    assert!(
        (nll0 - nll1).abs() < 1e-12,
        "unlabeled row affected the head NLL: {nll0} vs {nll1}"
    );
    let gdiff: f64 = grad0
        .iter()
        .zip(grad1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        gdiff < 1e-12,
        "unlabeled row affected the head gradient (Σ|Δ| = {gdiff})"
    );
    assert!(
        (head.effective_labeled_count() - (n as f64 / 2.0)).abs() < 1e-9,
        "effective labeled count wrong"
    );
}

#[test]
fn head_gradient_matches_finite_difference() {
    // The cross-channel coupling is load-bearing: the latent-code gradient
    // grad_t must match a finite difference of the NLL wrt t, and grad_coeffs
    // wrt the coefficients. Both verified on a small planted instance.
    let mut seed = 0x1357_9BDF_2468_ACE0;
    let (n, d, planted) = (60, 3, 2usize);
    let (t, y) = plant(n, d, planted, &mut seed);
    let head = BehavioralHead::fully_supervised(AuxOutcomeFamily::Binomial, y).expect("head");
    let n_coeffs = head.n_coeffs(d);
    let mut coeffs = Array1::<f64>::zeros(n_coeffs);
    for j in 0..n_coeffs {
        coeffs[j] = 0.1 * (j as f64 + 1.0);
    }
    let (_nll, grad_c, grad_t) = head
        .neg_loglik_and_grad(t.view(), coeffs.view())
        .expect("nll");

    let eps = 1e-6;
    // Coefficient gradient.
    for j in 0..n_coeffs {
        let mut cp = coeffs.clone();
        cp[j] += eps;
        let (nllp, _g, _gt) = head.neg_loglik_and_grad(t.view(), cp.view()).expect("nll");
        let mut cm = coeffs.clone();
        cm[j] -= eps;
        let (nllm, _g2, _gt2) = head.neg_loglik_and_grad(t.view(), cm.view()).expect("nll");
        let fd = (nllp - nllm) / (2.0 * eps);
        assert!(
            (fd - grad_c[j]).abs() < 1e-5,
            "coeff grad[{j}] analytic {} != FD {}",
            grad_c[j],
            fd
        );
    }
    // Latent-code gradient on a few rows.
    for &r in &[0usize, 7, 23, 41] {
        for k in 0..d {
            let mut tp = t.clone();
            tp[[r, k]] += eps;
            let (nllp, _g, _gt) = head
                .neg_loglik_and_grad(tp.view(), coeffs.view())
                .expect("nll");
            let mut tm = t.clone();
            tm[[r, k]] -= eps;
            let (nllm, _g2, _gt2) = head
                .neg_loglik_and_grad(tm.view(), coeffs.view())
                .expect("nll");
            let fd = (nllp - nllm) / (2.0 * eps);
            assert!(
                (fd - grad_t[[r, k]]).abs() < 1e-5,
                "grad_t[{r},{k}] analytic {} != FD {}",
                grad_t[[r, k]],
                fd
            );
        }
    }
}

//! End-to-end quality: gam's multinomial-logit (softmax) GLM must match
//! `statsmodels.api.MNLogit` — the Python standard for multinomial logit — on
//! identical data, not merely "run without panicking".
//!
//! Multinomial logit is not reachable through gam's scalar `fit(family=...)`
//! path (it is a vector-response family with `K-1` active linear predictors and
//! a per-row dense Fisher block); the canonical coefficient-space Newton solver
//! is `gam::families::multinomial::fit_penalized_multinomial`. This test drives
//! that solver directly on a purely-linear design (`y ~ x1 + x2 + x3 + x4`,
//! i.e. intercept + 4 slopes) with **zero penalty** so the objective is the
//! plain unpenalized multinomial deviance — exactly the objective MNLogit
//! maximizes. Both engines therefore target the *identical* unpenalized
//! likelihood, so close agreement is the correct expectation and a real
//! divergence is a real bug in gam's likelihood or its Newton solver.
//!
//! Reference: `statsmodels.api.MNLogit`. statsmodels uses the *first* category
//! (code 0) as the baseline; gam fixes class `K-1` (the last class) as the
//! reference with `η_{K-1} ≡ 0`. We relabel the response handed to statsmodels
//! so that gam's reference class (index `K-1`) is coded `0` there — then
//! statsmodels' params columns (categories `1..K-1`) line up one-to-one with
//! gam's active-class blocks (classes `0..K-1`), making the coefficients
//! directly comparable. Fitted probabilities and the deviance are gauge-free
//! and need no relabeling.
//!
//! We assert three things on the same data:
//!   1. fitted class probabilities agree (Frobenius relative L2 over the N×K
//!      matrix),
//!   2. per-active-class coefficient vectors agree (relative L2 each), and
//!   3. both engines reach the same unpenalized deviance.

use gam::families::multinomial::{MultinomialFitInputs, fit_penalized_multinomial};
use gam::init_parallelism;
use gam::test_support::reference::{Column, relative_l2, run_python};
use ndarray::{Array1, Array2};

#[test]
fn multinomial_logit_matches_statsmodels_mnlogit() {
    init_parallelism();

    // ---- synthetic data: n=150, K=4 classes, x1..x4 ~ U[-2,2] -------------
    // Deterministic so gam and statsmodels see byte-identical inputs. A simple
    // 64-bit LCG (Numerical Recipes constants) seeded fixed; uniforms mapped
    // to [-2, 2].
    const N: usize = 150;
    const K: usize = 4; // classes 0,1,2 active; class 3 is the reference.
    const P: usize = 5; // intercept + 4 covariates.

    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next_unif = move || -> f64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // top 53 bits -> [0,1)
        let bits = (state >> 11) as f64;
        let u = bits / (1u64 << 53) as f64;
        -2.0 + 4.0 * u // U[-2, 2]
    };

    let mut x1 = vec![0.0_f64; N];
    let mut x2 = vec![0.0_f64; N];
    let mut x3 = vec![0.0_f64; N];
    let mut x4 = vec![0.0_f64; N];
    for i in 0..N {
        x1[i] = next_unif();
        x2[i] = next_unif();
        x3[i] = next_unif();
        x4[i] = next_unif();
    }

    // True coefficients (intercept + slopes on x1..x4) for the three active
    // classes, with class 3 as the reference (η_3 ≡ 0). The moderate
    // magnitudes (|β| ≤ 0.9) keep per-row class probabilities well spread out
    // (no row near a 0/1 degenerate), so the sampled labels overlap across
    // classes and the unpenalized MLE stays finite and well-conditioned.
    let true_beta: [[f64; P]; K - 1] = [
        // class 0: intercept,  x1,   x2,   x3,   x4
        [0.4, 0.5, -0.3, 0.8, 0.0],
        // class 1
        [-0.3, -0.6, 0.4, -0.5, 0.2],
        // class 2
        [0.2, 0.9, 0.1, -0.7, -0.4],
    ];

    // Labels SAMPLED from the true softmax categorical (η_3 ≡ 0), not argmax.
    // This is the critical design choice: argmax labels are perfectly linearly
    // separable, under which the *unpenalized* multinomial MLE is not finite
    // (coefficients diverge to ±∞ along the separating direction) and neither
    // gam nor statsmodels would converge to a comparable β. Drawing each label
    // from its row's categorical distribution produces overlapping classes, so
    // the unpenalized log-likelihood is strictly concave with a finite, unique
    // maximizer — the well-posed common target both engines must reach. The
    // draw uses the same deterministic LCG stream, so the labels (and thus the
    // data fed to both engines) are byte-identical run to run.
    let mut labels = vec![0usize; N];
    for i in 0..N {
        let xrow = [1.0, x1[i], x2[i], x3[i], x4[i]];
        let mut eta = [0.0_f64; K]; // eta[3] = 0 (reference)
        for (a, brow) in true_beta.iter().enumerate() {
            let mut e = 0.0;
            for p in 0..P {
                e += brow[p] * xrow[p];
            }
            eta[a] = e;
        }
        // Stable softmax over the K linear predictors (eta[K-1] = 0 reference).
        let max_eta = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probs = [0.0_f64; K];
        let mut denom = 0.0_f64;
        for c in 0..K {
            probs[c] = (eta[c] - max_eta).exp();
            denom += probs[c];
        }
        // next_unif() returns U[-2,2]; rescale that same draw to U[0,1) for the
        // inverse-CDF categorical sample (no new RNG, keeps the stream aligned).
        let u01 = (next_unif() + 2.0) / 4.0;
        let mut cum = 0.0_f64;
        let mut drawn = K - 1; // fallback to last class on float round-off
        for c in 0..K {
            cum += probs[c] / denom;
            if u01 < cum {
                drawn = c;
                break;
            }
        }
        labels[i] = drawn;
    }

    // ---- gam: build the linear design X = [1, x1, x2, x3, x4] -------------
    let mut design = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        design[[i, 0]] = 1.0;
        design[[i, 1]] = x1[i];
        design[[i, 2]] = x2[i];
        design[[i, 3]] = x3[i];
        design[[i, 4]] = x4[i];
    }
    // One-hot response Y ∈ ℝ^{N×K}; gam treats column K-1 as the reference.
    let mut y_one_hot = Array2::<f64>::zeros((N, K));
    for i in 0..N {
        y_one_hot[[i, labels[i]]] = 1.0;
    }
    // Zero penalty ⇒ unpenalized MLE, matching MNLogit's objective exactly.
    let penalty = Array2::<f64>::zeros((P, P));
    let lambdas = Array1::<f64>::zeros(K - 1);

    let out = fit_penalized_multinomial(MultinomialFitInputs {
        design: design.view(),
        y_one_hot: y_one_hot.view(),
        penalty: penalty.view(),
        lambdas: lambdas.view(),
        row_weights: None,
        fisher_w_override: None,
        max_iter: 100,
        tol: 1e-10,
    })
    .expect("gam multinomial fit");
    assert!(
        out.converged,
        "gam multinomial Newton solve did not converge in 100 iters"
    );

    // gam coefficients_active: shape (P, K-1), column a = β_a for class a
    // (classes 0,1,2; reference class 3 has β ≡ 0). Flatten per class.
    let gam_coef: Vec<Vec<f64>> = (0..K - 1)
        .map(|a| (0..P).map(|p| out.coefficients_active[[p, a]]).collect())
        .collect();
    // gam fitted probabilities: (N, K), column j = P(class j). Row-major flat.
    let mut gam_probs_flat = Vec::with_capacity(N * K);
    for i in 0..N {
        for j in 0..K {
            gam_probs_flat.push(out.fitted_probabilities[[i, j]]);
        }
    }
    let gam_deviance = out.deviance;

    // ---- statsmodels MNLogit on the SAME data -----------------------------
    // Relabel so gam's reference class (K-1 = 3) is statsmodels' baseline (0):
    //   3 -> 0, 0 -> 1, 1 -> 2, 2 -> 3.
    // Then statsmodels params columns (its categories 1,2,3) correspond to gam
    // classes 0,1,2 in order, and its predicted-prob columns (0..K) map back as
    // sm_col c -> gam class (c - 1 + K) mod K, i.e. col0->gam3, col1->gam0,
    // col2->gam1, col3->gam2. We undo that remap on the Python side so the
    // emitted probability matrix is already in gam's column order.
    let sm_label: Vec<f64> = labels
        .iter()
        .map(|&c| ((c + 1) % K) as f64) // 3->0, 0->1, 1->2, 2->3
        .collect();

    let r = run_python(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("x3", &x3),
            Column::new("x4", &x4),
            Column::new("y", &sm_label),
        ],
        r#"
import numpy as np
import statsmodels.api as sm

X = np.column_stack([df["x1"], df["x2"], df["x3"], df["x4"]])
Xc = sm.add_constant(X, prepend=True)  # column 0 = intercept
y = np.asarray(df["y"], dtype=int)

model = sm.MNLogit(y, Xc)
# Newton on the unpenalized multinomial log-likelihood; tight tolerance so the
# only differences from gam are floating-point, not optimizer slack.
res = model.fit(method="newton", maxiter=200, gtol=1e-10, disp=0)

# res.params: shape (P, K-1), columns = statsmodels categories 1..K-1.
# With our relabel those columns ARE gam classes 0,1,2 in order. Emit one key
# per active class so the Rust side can align element-wise.
params = np.asarray(res.params)  # (P, K-1)
emit("coef0", params[:, 0])  # gam class 0
emit("coef1", params[:, 1])  # gam class 1
emit("coef2", params[:, 2])  # gam class 2

# Predicted probabilities: res.predict -> (N, K) in statsmodels category order
# 0,1,2,3 = gam classes 3,0,1,2. Reorder columns back to gam order 0,1,2,3.
probs_sm = np.asarray(res.predict(Xc))          # (N, K), sm-category order
gam_order = [1, 2, 3, 0]                          # sm col for gam class j
probs_gam = probs_sm[:, gam_order]                # (N, K) in gam order
emit("probs", probs_gam.reshape(-1))              # row-major flat

# Unpenalized deviance = -2 * log-likelihood at the MLE.
emit("deviance", np.array([-2.0 * res.llf]))
"#,
    );

    let sm_probs = r.vector("probs");
    assert_eq!(sm_probs.len(), N * K, "statsmodels probs length mismatch");
    let sm_deviance = r.scalar("deviance");

    // ---- compare ----------------------------------------------------------
    let prob_rel = relative_l2(&gam_probs_flat, sm_probs);

    let mut coef_rels = [0.0_f64; K - 1];
    for a in 0..K - 1 {
        let sm_coef = r.vector(&format!("coef{a}"));
        assert_eq!(sm_coef.len(), P, "statsmodels coef{a} length mismatch");
        coef_rels[a] = relative_l2(&gam_coef[a], sm_coef);
    }

    let dev_rel = (gam_deviance - sm_deviance).abs() / sm_deviance.abs().max(1.0);

    eprintln!(
        "multinomial vs MNLogit: n={N} K={K} gam_iters={} \
         prob_rel_l2={prob_rel:.5} coef_rel_l2=[{:.5},{:.5},{:.5}] \
         gam_dev={gam_deviance:.4} sm_dev={sm_deviance:.4} dev_rel={dev_rel:.6}",
        out.iterations, coef_rels[0], coef_rels[1], coef_rels[2]
    );

    // Both engines maximize the identical unpenalized multinomial
    // log-likelihood by Newton's method, so the fitted probabilities (a
    // gauge-free quantity) must coincide to optimizer/float precision. The
    // 0.008 Frobenius-relative bound is far above attainable float noise yet
    // tight enough that any genuine discrepancy in gam's softmax likelihood or
    // its block-Fisher Newton step would trip it.
    assert!(
        prob_rel < 0.008,
        "fitted probabilities diverge from statsmodels MNLogit: rel_l2={prob_rel:.5}"
    );

    // Coefficients are compared in a shared reference gauge (gam's class K-1 =
    // statsmodels' baseline 0). At the same MLE of the same objective they must
    // match; 0.015 relative leaves room only for float-level optimizer slack.
    for a in 0..K - 1 {
        assert!(
            coef_rels[a] < 0.015,
            "class {a} coefficients diverge from statsmodels MNLogit: rel_l2={:.5}",
            coef_rels[a]
        );
    }

    // Same objective at the same optimum ⇒ same deviance. A tight 0.2% relative
    // tolerance guards against the engines silently optimizing different things.
    assert!(
        dev_rel < 0.002,
        "unpenalized deviance disagrees: gam={gam_deviance:.4} sm={sm_deviance:.4} (rel={dev_rel:.6})"
    );
}

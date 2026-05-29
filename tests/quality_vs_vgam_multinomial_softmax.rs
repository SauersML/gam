//! End-to-end quality: gam's penalized multinomial-logit (softmax) GAM with
//! smooth terms must agree with VGAM — the mature R reference for multiclass
//! categorical regression with smooth predictors — on the *fitted class
//! probability surface*.
//!
//! Reference tool: `VGAM::vgam(..., family = multinomial())`. VGAM is the
//! canonical R package for vector-generalized (multinomial / proportional-odds)
//! models; `vgam()` is its smooth-term entry point (`vglm()` is the strictly
//! parametric sibling and does not accept `s()`), so a multinomial GAM with
//! `s(x1) + s(x2) + x3` is fit there exactly the way a practitioner would.
//!
//! gam fits the same model through the dedicated vector-response driver
//! `gam::families::multinomial::fit_penalized_multinomial_formula` (the scalar
//! `fit_from_formula` path explicitly rejects `family="multinomial"` because
//! the likelihood is a K-1-block vector family). Both engines use the canonical
//! reference-class softmax gauge (η_{K-1} ≡ 0); gam reports its level order via
//! `MultinomialSavedModel::class_levels`, and we pin VGAM's factor levels to the
//! *same* order so the reference class — and therefore the entire identified
//! probability simplex — coincides.
//!
//! The quantity that matters and is gauge-invariant is the predicted
//! probability matrix P ∈ ℝ^{N×K} (rows on the simplex). Coefficients depend on
//! the spline basis (gam: REML-penalized regression splines; VGAM: smoothing
//! splines), so they are NOT directly comparable; the fitted probabilities are.
//! We assert agreement of P column-by-column (Pearson per class) and overall
//! (relative Frobenius / relative-L2 over the flattened matrix). With a strong
//! true signal (per-class linear slopes ±1.5/−0.8/0 plus a cubic and a sigmoid
//! smooth) both penalized fits recover essentially the same surface, so tight
//! agreement is the correct expectation and a real divergence is a real bug.

use csv::StringRecord;
use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const N: usize = 300;
const K: usize = 3;

/// One stable softmax draw: builds the K class log-odds (reference class 2 is
/// pinned to η = 0), exponentiates, normalizes, and samples a class from a
/// uniform draw. Identical math feeds gam and VGAM (we hand both the same
/// predictors and the same realized labels), so the comparison is honest.
fn true_eta(x1: f64, x2: f64, x3: f64) -> [f64; K] {
    // Smooth + parametric structure shared across the K-1 active classes, with
    // distinct per-class linear slopes on x3 [+1.5, -0.8, 0(ref)].
    let cubic = 2.0 * x1.powi(3) - 1.0 * x1; // s(x1): cubic shape
    let sigmoid = 3.0 / (1.0 + (-6.0 * (x2 - 0.5)).exp()) - 1.5; // s(x2): sigmoid shape
    // Active classes 0 and 1; reference class 2 has eta = 0.
    let eta0 = 0.6 + cubic + 0.5 * sigmoid + 1.5 * x3;
    let eta1 = -0.4 - 0.5 * cubic + sigmoid - 0.8 * x3;
    [eta0, eta1, 0.0]
}

fn softmax(eta: &[f64; K]) -> [f64; K] {
    let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut ex = [0.0; K];
    let mut s = 0.0;
    for k in 0..K {
        ex[k] = (eta[k] - m).exp();
        s += ex[k];
    }
    for k in 0..K {
        ex[k] /= s;
    }
    ex
}

#[test]
fn gam_multinomial_softmax_matches_vgam() {
    init_parallelism();

    // ---- synthesize the shared dataset (fixed seed, fed to both engines) ----
    let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform x1");
    let u01 = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x2");
    let ux3 = Uniform::new(-1.5_f64, 1.5_f64).expect("uniform x3");
    let udraw = Uniform::new(0.0_f64, 1.0_f64).expect("uniform draw");

    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut x3 = Vec::with_capacity(N);
    let mut cls_code = Vec::with_capacity(N); // realized class index 0..K
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = u01.sample(&mut rng);
        let c = ux3.sample(&mut rng);
        let p = softmax(&true_eta(a, b, c));
        let u = udraw.sample(&mut rng);
        // inverse-CDF class sample
        let mut acc = 0.0;
        let mut chosen = K - 1;
        for k in 0..K {
            acc += p[k];
            if u <= acc {
                chosen = k;
                break;
            }
        }
        x1.push(a);
        x2.push(b);
        x3.push(c);
        cls_code.push(chosen);
    }

    // Class labels gam will treat as categorical (non-numeric strings so the
    // inferred schema marks the response column Categorical, not Continuous).
    let label = |code: usize| format!("c{code}");

    // ---- fit with gam: y ~ s(x1) + s(x2) + x3, multinomial driver ----------
    let headers: Vec<String> = ["x1", "x2", "x3", "y"].iter().map(|s| s.to_string()).collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                x3[i].to_string(),
                label(cls_code[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode multinomial dataset");

    let cfg = FitConfig::default();
    // init_lambda=1.0 warm-start; the REML/LAML outer loop selects per-class λ.
    let model = fit_penalized_multinomial_formula(&ds, "y ~ s(x1) + s(x2) + x3", &cfg, 1.0, 50, 1e-8)
        .expect("gam multinomial fit");
    assert_eq!(model.class_levels.len(), K, "gam should recover K=3 classes");

    // gam fitted probabilities at the training rows (predict reuses the frozen
    // training basis/penalty — no refit). Columns follow `model.class_levels`.
    let gam_probs = predict_multinomial_formula(&model, &ds).expect("gam predict probabilities");
    assert_eq!(gam_probs.dim(), (N, K), "gam probability matrix shape");

    // gam's level order is order-of-first-appearance; the reference class is the
    // LAST level (eta = 0). We pin VGAM to this exact order so both engines
    // share the same reference and the column-k probabilities are comparable.
    let gam_levels: Vec<String> = model.class_levels.clone();
    // Map each gam level string "c{code}" back to its integer code for R. The
    // gam level sequence has length K, but every reference column handed to
    // `run_r` must have exactly N rows (the harness rejects ragged columns), so
    // we tile the K-length order cyclically to length N. R recovers the order
    // via `unique(round(df$levorder))`, which preserves first-occurrence order,
    // and the first K tiled entries are exactly gam's level sequence.
    let level_codes: Vec<f64> = (0..N)
        .map(|i| {
            let lvl = &gam_levels[i % K];
            lvl.trim_start_matches('c')
                .parse::<f64>()
                .expect("gam level label is c<code>")
        })
        .collect();

    // ---- fit the SAME model with VGAM::vgam (the mature reference) ---------
    // Pass the integer class code plus the three predictors; rebuild the factor
    // in R with levels ordered exactly as gam reports (so reference = last).
    let cls_f64: Vec<f64> = cls_code.iter().map(|&c| c as f64).collect();
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("x3", &x3),
            Column::new("cls", &cls_f64),
            // The gam level order, emitted as the integer codes in that order,
            // so R can set identical factor levels.
            Column::new("levorder", &level_codes),
        ],
        r#"
        suppressPackageStartupMessages(library(VGAM))
        # Reconstruct the factor with levels in gam's order: the last level is
        # the multinomial reference class for both engines. `levorder` repeats
        # the K codes across rows (only its first K entries matter); take the
        # unique codes in first-seen order to recover gam's level sequence.
        lev_codes <- unique(round(df$levorder))
        lev_codes <- lev_codes[seq_len(length(lev_codes))]
        lev_labels <- paste0("c", lev_codes)
        yfac <- factor(paste0("c", round(df$cls)), levels = lev_labels)
        dat <- data.frame(x1 = df$x1, x2 = df$x2, x3 = df$x3, y = yfac)
        # vgam: smooth s(x1), s(x2) plus parametric x3, multinomial family.
        # multinomial() in VGAM uses the LAST factor level as the reference,
        # matching gam's eta_{K-1} = 0 gauge once levels are pinned above.
        m <- vgam(y ~ s(x1) + s(x2) + x3, family = multinomial(), data = dat)
        # type="response" returns the N x K fitted probability matrix with
        # columns in factor-level order == gam's class_levels order.
        pr <- predict(m, type = "response")
        # Emit column-major flattened (col 0 rows, col 1 rows, ...) to match the
        # Rust unflatten below.
        emit("nrow", nrow(pr))
        emit("ncol", ncol(pr))
        emit("probs", as.numeric(as.vector(pr)))
        "#,
    );

    let vg_nrow = r.scalar("nrow") as usize;
    let vg_ncol = r.scalar("ncol") as usize;
    assert_eq!(vg_nrow, N, "VGAM fitted-prob rows");
    assert_eq!(vg_ncol, K, "VGAM fitted-prob cols");
    let vg_flat = r.vector("probs"); // column-major: [col0_rows..., col1_rows..., ...]
    assert_eq!(vg_flat.len(), N * K, "VGAM flattened prob length");

    // ---- compare the probability matrices, column-by-column and overall ----
    // gam_probs is row-major (N,K); rebuild matching flat vectors per class.
    let mut overall_gam = Vec::with_capacity(N * K);
    let mut overall_vg = Vec::with_capacity(N * K);
    let mut worst_class_pearson = 1.0_f64;
    for k in 0..K {
        let mut gk = Vec::with_capacity(N);
        let mut vk = Vec::with_capacity(N);
        for i in 0..N {
            gk.push(gam_probs[[i, k]]);
            vk.push(vg_flat[k * N + i]); // column-major: column k starts at k*N
        }
        let corr_k = pearson(&gk, &vk);
        worst_class_pearson = worst_class_pearson.min(corr_k);
        eprintln!(
            "class {} ({}): pearson={corr_k:.5}",
            k, gam_levels[k]
        );
        overall_gam.extend_from_slice(&gk);
        overall_vg.extend_from_slice(&vk);
    }
    // Relative Frobenius distance of the whole probability matrix.
    let frob_rel = relative_l2(&overall_gam, &overall_vg);
    let overall_corr = pearson(&overall_gam, &overall_vg);

    eprintln!(
        "multinomial s(x1)+s(x2)+x3: N={N} K={K} converged={} iters={} \
         frob_rel={frob_rel:.4} overall_pearson={overall_corr:.5} \
         worst_class_pearson={worst_class_pearson:.5} lambdas={:?}",
        model.converged, model.iterations, model.lambdas
    );

    // Both engines fit a REML/penalized multinomial GAM on identical data with
    // the same reference-class softmax gauge, so the identified probability
    // surface must essentially coincide. The only legitimate gap is the spline
    // basis difference (gam's penalized regression splines vs VGAM's smoothing
    // splines), which perturbs probabilities by well under a percent on a
    // strong signal. pearson > 0.998 (per class and overall) and a relative
    // Frobenius distance < 0.05 are tight bounds justified by that shared
    // objective; a larger gap signals a real softmax/penalty divergence.
    assert!(
        worst_class_pearson > 0.998,
        "per-class fitted-probability agreement too low: worst pearson={worst_class_pearson:.5}"
    );
    assert!(
        overall_corr > 0.998,
        "overall fitted-probability correlation too low: pearson={overall_corr:.5}"
    );
    assert!(
        frob_rel < 0.05,
        "fitted-probability matrices diverge from VGAM: relative Frobenius={frob_rel:.4}"
    );
}

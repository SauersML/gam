//! End-to-end OBJECTIVE quality: gam's NUTS posterior for a penalized
//! binomial-logit smooth must **recover the known truth and be well
//! calibrated**, not merely reproduce PyMC's draws.
//!
//! The data are generated from a known latent function
//!     η_true(x) = 0.3 + 0.8 · sin(2π x / 10),   y ~ Bernoulli(logit⁻¹(η_true)).
//! Because the generating function is known exactly, the honest quality
//! question is not "does gam's posterior look like PyMC's posterior?" (matching
//! a peer NUTS engine proves nothing — both could be miscalibrated together),
//! but rather:
//!   (A) TRUTH RECOVERY — does the posterior MEAN of the linear predictor
//!       η = Xβ track the true η over the design? We assert
//!       RMSE(gam_post_mean_η, η_true) is below a principled bar set by the
//!       Bernoulli observation noise, and additionally that gam's recovery
//!       error is no worse than PyMC's by more than 10% (match-or-beat on
//!       accuracy — PyMC is the BASELINE, not the target).
//!   (B) CALIBRATION — do gam's pointwise 90% posterior credible intervals for
//!       η actually contain the TRUTH ~90% of the time? We assert empirical
//!       coverage within a tolerance band of the 0.90 nominal level. A correct
//!       Bayesian smoother must be calibrated against the truth; this is an
//!       objective uncertainty claim, independent of any reference tool.
//!
//! PyMC remains in the file as a BASELINE on the same objective metric
//! (truth-recovery RMSE) — gam must match or beat it — and its R-hat is used
//! only to confirm the baseline run itself converged. The pass/fail criteria
//! are gam-vs-truth, never gam-vs-PyMC.
//!
//! A failure here is a real quality shortfall in gam's posterior, never a
//! reason to loosen the bounds or touch gam source.

use gam::inference::model::{FittedFamily, FittedModel, FittedModelPayload, ModelKind};
use gam::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use gam::test_support::reference::{Column, QualityPair, rmse, run_python};
use gam::types::{LikelihoodSpec, StandardLink};
use gam::{
    FitConfig, FitResult, fit_from_formula, hmc::NutsConfig, init_parallelism,
    load_csvwith_inferred_schema, sample::sample_saved_model,
};
use ndarray::{Array1, Array2};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// prostate.csv source: the `prostate` PCA-feature benchmark shipped in
/// `bench/datasets/` (two leading principal-component scores `pc1`, `pc2` and a
/// binary outcome `y`). Real data => no known latent truth, so the new arm
/// asserts OBJECTIVE held-out classification quality (log-loss + AUC) and a
/// match-or-beat against a PyMC-NUTS baseline fit on the identical design.
const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");
// Posterior-sampling budget for the REAL prostate arm (n=490, p=47). This arm
// only asserts robust posterior-MEAN held-out probabilities (log-loss / AUC
// match-or-beat) plus an R-hat<1.1 convergence gate — it is NOT a per-point
// credible-interval or sampler-fidelity test, so the draw count need not scale
// with n·p. For an easy log-concave Bernoulli-logit posterior these counts give
// 2×250 = 500 effective draws (MC error on the posterior-mean probability well
// under the 5% match-or-beat tolerance and the 0.02 log-loss margin) and a
// reliably sub-1.1 R-hat, while keeping the silent post-fit sampling block well
// under the 360s suite cap on top of the ~84s GAM REML fit. Tune is matched to
// draws so NUTS step-size / mass-matrix adaptation is fully converged.
const REAL_DATA_POSTERIOR_SAMPLES: usize = 250;
const REAL_DATA_POSTERIOR_WARMUP: usize = 250;
const REAL_DATA_POSTERIOR_CHAINS: usize = 2;

/// Deterministic splitmix64 stream → uniform(0,1). Keeps the synthetic data
/// fully reproducible with no external RNG crate, so gam and PyMC see byte-for
/// -byte identical inputs.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        // 53-bit mantissa in [0,1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn inv_logit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// True latent linear predictor at x — the function the smooth must recover.
fn eta_true(x: f64) -> f64 {
    0.3 + 0.8 * (2.0 * std::f64::consts::PI * x / 10.0).sin()
}

/// Linear-interpolated quantile of an (already sorted) sample. `q` in [0,1].
fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    let m = sorted.len();
    assert!(m > 0, "quantile of empty sample");
    if m == 1 {
        return sorted[0];
    }
    let pos = q * (m as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Mean binary cross-entropy (log-loss) of predicted probabilities `p` against
/// 0/1 labels `y`. Lower is better; the constant base-rate predictor sits at the
/// label entropy, so a model that beats it is genuinely informative. Clamped
/// away from {0,1} so a single confident miss cannot send the metric to +inf.
fn log_loss(p: &[f64], y: &[f64]) -> f64 {
    assert_eq!(p.len(), y.len(), "log_loss length mismatch");
    let eps = 1e-12;
    let s: f64 = p
        .iter()
        .zip(y)
        .map(|(&pi, &yi)| {
            let pc = pi.clamp(eps, 1.0 - eps);
            -(yi * pc.ln() + (1.0 - yi) * (1.0 - pc).ln())
        })
        .sum();
    s / p.len().max(1) as f64
}

/// Area under the ROC curve via the Mann-Whitney U statistic (average rank of
/// the positive scores). 0.5 is chance, 1.0 is perfect ranking. Handles ties by
/// assigning the mean rank within each tie group.
fn auc(score: &[f64], y: &[f64]) -> f64 {
    assert_eq!(score.len(), y.len(), "auc length mismatch");
    let n = score.len();
    let n_pos = y.iter().filter(|&&v| v > 0.5).count();
    let n_neg = n - n_pos;
    assert!(n_pos > 0 && n_neg > 0, "auc needs both classes");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).expect("finite scores"));
    // Fractional ranks (1-based), averaging within ties.
    let mut ranks = vec![0.0f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && score[idx[j]] == score[idx[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + j) as f64 / 2.0; // mean of ranks (i+1)..=j
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    let sum_pos_ranks: f64 = (0..n).filter(|&k| y[k] > 0.5).map(|k| ranks[k]).sum();
    let u = sum_pos_ranks - (n_pos * (n_pos + 1)) as f64 / 2.0;
    u / (n_pos as f64 * n_neg as f64)
}


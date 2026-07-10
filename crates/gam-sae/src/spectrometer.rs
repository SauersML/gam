//! Dimension spectrometer — inverting SAE scaling laws into an intrinsic-dimension
//! measurement.
//!
//! # Theory
//!
//! Data concentrated on a `d`-dimensional manifold, quantized by a width-`K`
//! linear dictionary that codes every row with a **single** atom (active budget
//! `s = 1`), incurs an excess reconstruction loss that decays as a power law in
//! the dictionary width:
//!
//! ```text
//!     L(K) − σ²  ∝  K^(−2/d)
//! ```
//!
//! where `σ²` is the irreducible noise floor (off-manifold variance the rank-1
//! code cannot represent). This is the classical vector-quantization rate: `K`
//! codewords tile a `d`-manifold with cells of linear size `K^(−1/d)`, and the
//! mean squared quantization error scales as the square of that size. Taking logs,
//!
//! ```text
//!     log(L(K) − σ²)  =  log c  −  (2/d) · log K,
//! ```
//!
//! so regressing `log(excess loss)` on `log K` over a ladder of widths yields a
//! slope `m = −2/d`, and the intrinsic dimension is recovered as `d̂ = −2/m`.
//!
//! # `s` is forced to 1
//!
//! The `−2/d` exponent is a **single-atom** (nearest-codeword) VQ rate. With an
//! interpolative budget `s ≥ 2` the reconstruction is a span of `s` atoms, whose
//! error obeys a *different* exponent and, worse, exhibits planar degeneracies (a
//! 2-sparse code of points near a plane collapses the effective rate). The
//! spectrometer therefore forces `active = 1` on every rung regardless of the
//! template config, so what it measures is the quantity the theory describes.
//!
//! # The doubling ladder is derived, not searched (SPEC rule 17)
//!
//! SPEC forbids grid search. The `K` ladder here is **not** a hyper-parameter grid
//! to be swept for the best model — it is the *estimand's own design*. Estimating
//! the exponent of a power law is a log-log slope estimation; the statistically
//! optimal, information-balanced design for a slope in `log K` is a set of widths
//! *equally spaced in `log K`*, i.e. a geometric (doubling) ladder. The ladder is
//! thus derived from the shape of the law being measured, not tuned. Its span
//! (`k_min`, number of doublings) is the measurement aperture, exactly analogous
//! to choosing the frequency range of a spectrometer — reported, not optimized.
//!
//! # Numerics (SPEC rules 2, 18, 19)
//!
//! Every per-rung fit runs through the collapsed-linear lane
//! ([`crate::sparse_dict::fit_sparse_dictionary`]) with its `N×K`-free routing, so
//! the spectrometer inherits its memory contract. Losses are accumulated in `f64`.
//! The scaling-law fit is a closed-form linear regression (of `log(excess)` on
//! `log K`) whose only nuisance is the noise floor `σ²`; `σ²` is *profiled out*
//! (variable projection): for any candidate `σ²` the slope/intercept are the
//! closed-form ordinary-least-squares solution, leaving a smooth one-dimensional
//! objective in `σ²` alone. That 1-D profile is minimized by golden-section
//! bracketing **to a numerical tolerance** (not a fixed sweep count and not a
//! wall-clock budget — the same converge-to-tolerance discipline the lane's CG
//! block solver uses), over the *derived* bracket `σ² ∈ [0, min_k L_k)`: a noise
//! variance is non-negative and the plateau cannot exceed the smallest achieved
//! loss. No finite differences are used (the regression and its standard errors
//! are closed form); the golden ratio and the reporting confidence quantile are
//! mathematical constants, not tuned knobs.
//!
//! # High-`K` finite-sample bias, and the stable window (derived, not tuned)
//!
//! The `K^(−2/d)` rate is a *population* quantization rate: it assumes each of the
//! `K` cells holds enough points to estimate its atom without overfitting. When the
//! points-per-atom `n = N/K` gets small (large `K`, or few rows), the single-atom
//! fit overfits its handful of assigned points and the **held-in** loss dips below
//! the population rate — biasing the local log-log slope more negative and hence
//! biasing `d̂ = −2/m` *downward*. On real data this is visible as `d̂` drifting
//! down along the ladder (e.g. a 33→6 slide as `K` climbs into the token-starved
//! regime). This is a finite-sample artifact, not a change in intrinsic dimension.
//!
//! We control it with a **derived** reliable window, not a magic cutoff. Fitting one
//! atom (the leading local direction, plus a per-point scale) to `n = N/K` points
//! underestimates the population cell residual by a leading factor `≈ 1 − d/n`
//! (a single atom captures the best of `≈ d` local tangent directions estimated from
//! `n` points — the standard fitted-degrees-of-freedom shrinkage). In log-loss this
//! adds a term `≈ −d/n = −d·K/N`, which over one octave (`K → 2K`) steepens the slope
//! by `≈ (d / ln 2)·(K/N)`. Requiring that finite-sample slope bias to stay below the
//! law's own per-octave signal slope `2/d` gives the condition
//!
//! ```text
//!     n = N/K  ≥  d² / (2 ln 2)              (floored at 2, so a cell's residual
//!                                             has positive degrees of freedom).
//! ```
//!
//! Every constant here is derived — `ln 2` from the octave spacing of the ladder,
//! `2/d` from the VQ law, the `d/n` shrinkage from fitted-DOF counting — so the
//! threshold is **permissive for low `d`** (a `d = 1` circle stays reliable to very
//! large `K`, matching the tests) and **strict for high `d`** (a `d ≈ 33` manifold
//! needs hundreds of tokens per atom, which is exactly where the real-data drift
//! appears). The threshold depends on `d`, which is what we are estimating, so the
//! window is found by a short deterministic fixed-point: seed with the full-ladder
//! `d̂`, keep the rungs meeting the threshold, refit, repeat until the rung set
//! stops changing.
//!
//! The report exposes the drift rather than hiding it: it carries the full-ladder
//! `d̂`, the stable-window `d̂` (the trustworthy default), and a drop-last `d̂`
//! (drop the single largest-`K` rung), plus per-rung points-per-atom and a
//! `small_nk_regime` flag that fires when the ladder reaches below the threshold.

use crate::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
use ndarray::ArrayView2;

/// Configuration for a dimension-spectrometer sweep.
///
/// The dictionary-width ladder is specified structurally as a starting width
/// `k_min` and a number of doublings; the actual rung widths are
/// `k_min · 2^j` for `j = 0..=n_doublings` (so `n_doublings + 1` rungs). This
/// geometric ladder is a *derived* design for log-log slope estimation, not a
/// tuned grid (see the module docs, SPEC rule 17).
///
/// [`Self::dict`] is the template forwarded to every per-rung fit. Its `n_atoms`
/// is overwritten with the rung width and its `active` budget is **forced to 1**
/// on every rung (the `−2/d` exponent is a single-atom VQ rate); all other knobs
/// (minibatch, epochs, score tile, ridges, tolerance, score mode) are forwarded
/// unchanged.
#[derive(Clone, Copy, Debug)]
pub struct SpectrometerConfig {
    /// Smallest dictionary width on the ladder (the first rung, `j = 0`).
    pub k_min: usize,
    /// Number of doublings; the ladder has `n_doublings + 1` rungs, the largest
    /// being `k_min · 2^n_doublings`.
    pub n_doublings: usize,
    /// Template forwarded to each per-rung [`fit_sparse_dictionary`]. `n_atoms` is
    /// set to the rung width and `active` is forced to `1` per rung.
    pub dict: SparseDictConfig,
}

impl SpectrometerConfig {
    /// A spectrometer over the ladder `k_min · 2^j`, `j = 0..=n_doublings`, with
    /// the forwarded fit template left at its shared defaults (`active` is forced
    /// to 1 per rung regardless).
    pub fn new(k_min: usize, n_doublings: usize) -> Self {
        Self {
            k_min,
            n_doublings,
            dict: SparseDictConfig::default(),
        }
    }
}

impl Default for SpectrometerConfig {
    fn default() -> Self {
        // A six-doubling aperture from K=4 (rungs 4..=256): wide enough in log K to
        // pin a slope, cheap enough to run as a diagnostic.
        Self::new(4, 6)
    }
}

/// Result of a dimension-spectrometer sweep.
///
/// The headline `slope` / `d_hat` / `noise_floor` / standard errors are the
/// **stable-window** estimate (the rungs that pass the derived points-per-atom
/// threshold — the trustworthy default). The full-ladder and drop-last estimates
/// are carried alongside so any high-`K` finite-sample drift is visible.
#[derive(Clone, Debug)]
pub struct SpectrometerReport {
    /// Per-rung `(K, mean per-row reconstruction loss)` in ladder order (ascending
    /// `K`). The loss is the mean over rows of the squared reconstruction residual
    /// `‖x_i − c_i d_{a_i}‖²`, accumulated in `f64`.
    pub rungs: Vec<(usize, f64)>,
    /// Number of data rows `N` (used to form points-per-atom `N/K`).
    pub n_rows: usize,
    /// Points-per-atom `N/K` for each rung, aligned with [`Self::rungs`]. Small
    /// values are the token-starved regime where the single-atom fit overfits.
    pub points_per_atom: Vec<f64>,
    /// Estimated noise floor `σ²` (the loss plateau), profiled out of the
    /// stable-window log-log regression. Same units as the per-rung loss.
    pub noise_floor: f64,
    /// Fitted slope `m` of `log(L(K) − σ²)` on `log K` over the stable window.
    pub slope: f64,
    /// Standard error of the slope from the stable-window regression residuals.
    pub slope_se: f64,
    /// Intrinsic-dimension estimate `d̂ = −2/m` over the stable window (primary).
    pub d_hat: f64,
    /// Delta-method standard error of the stable-window `d̂`: `|2/m²| · SE(m)`.
    pub d_hat_se: f64,
    /// `d̂` fitted over the FULL ladder (all rungs). May be biased downward when the
    /// ladder reaches into the small-`N/K` regime; compare against [`Self::d_hat`].
    pub d_hat_full_ladder: f64,
    /// `d̂` fitted over the ladder with the single largest-`K` rung dropped — a cheap
    /// robustness probe: if it disagrees with the full-ladder value the tail is
    /// biasing the estimate.
    pub d_hat_drop_last: f64,
    /// Smallest `K` retained in the stable window.
    pub stable_window_lo_k: usize,
    /// Largest `K` retained in the stable window.
    pub stable_window_hi_k: usize,
    /// Number of rungs in the stable window.
    pub stable_rung_count: usize,
    /// The derived points-per-atom threshold `τ = max(d̂²/(2 ln 2), 2)` used to form
    /// the stable window (evaluated at the stable-window `d̂`).
    pub min_points_per_atom: f64,
    /// `true` when the slope's confidence interval contains zero — the retained rungs
    /// are statistically indistinguishable from the noise floor, the log-log slope
    /// is not resolved, and `d̂ = −2/m` is unreliable (near a division by zero).
    pub floor_saturated: bool,
    /// `true` when the ladder extends below the points-per-atom threshold (the stable
    /// window is a strict subset of the ladder, or some rung is token-starved). When
    /// set, [`Self::d_hat_full_ladder`] is untrustworthy and the drift is real.
    pub small_nk_regime: bool,
}

/// Two-sided 95% quantile of the standard normal — the confidence level at which
/// the slope's interval is formed to decide floor saturation. This is a
/// distributional constant (a reporting convention), not a tuned knob: with a
/// ladder of several rungs the normal approximation to the slope's sampling
/// distribution is adequate; a Student-`t` quantile would widen the interval
/// slightly (making saturation marginally *easier* to flag).
const NORMAL_95_QUANTILE: f64 = 1.959_963_984_540_054;

/// Relative width the golden-section profile of `σ²` is driven below before
/// stopping — a numerical convergence tolerance (cf. the lane's `CG_REL_TOL`),
/// not a wall-clock budget.
const PROFILE_REL_TOL: f64 = 1.0e-12;

/// Generous safety cap on golden-section iterations. Golden section contracts the
/// bracket by the golden ratio each step, so reaching [`PROFILE_REL_TOL`] takes a
/// few dozen steps; this cap only guards against a pathological non-terminating
/// bracket (it is never the reason the loop stops on well-posed input).
const PROFILE_MAX_ITERS: usize = 400;

/// Run the dimension spectrometer: fit a single-atom (`s = 1`) dictionary at each
/// rung of the doubling ladder, measure the per-rung mean reconstruction loss, and
/// invert the fitted scaling law into an intrinsic-dimension estimate.
///
/// The fits run in fixed ladder order (ascending `K`); each forwards the template
/// [`SpectrometerConfig::dict`] with `n_atoms` set to the rung width and `active`
/// forced to 1. The regression and its standard errors are computed in `f64`.
pub fn dimension_spectrometer(
    data: ArrayView2<'_, f32>,
    cfg: &SpectrometerConfig,
) -> Result<SpectrometerReport, String> {
    if data.nrows() == 0 || data.ncols() == 0 {
        return Err("dimension_spectrometer requires a non-empty N×P matrix".to_string());
    }
    if !data.iter().all(|v| v.is_finite()) {
        return Err("dimension_spectrometer input must be finite".to_string());
    }
    if cfg.k_min == 0 {
        return Err("dimension_spectrometer requires k_min >= 1".to_string());
    }
    // Need at least four rungs: the scaling law L(K) = σ² + c·K^m has THREE fitted
    // parameters (the profiled-out noise floor σ², plus the log-log slope and
    // intercept), so a residual degree of freedom for the slope's standard error
    // needs `nrung − 3 ≥ 1`. With only three rungs the three parameters interpolate
    // the data exactly (RSS ≈ 0) and the slope SE is undefined; the fit would report
    // a spurious near-zero uncertainty. `fit_scaling_law` still accepts three rungs
    // (returning an infinite SE / floor-saturated verdict) so the drop-last probe and
    // the stable-window fallback stay well-posed, but the primary entry demands four.
    if cfg.n_doublings < 3 {
        return Err(
            "dimension_spectrometer requires n_doublings >= 3 (>= 4 rungs): the scaling law has \
             three parameters (σ², slope, intercept), so a slope standard error needs at least \
             one residual degree of freedom (nrung − 3 >= 1)"
                .to_string(),
        );
    }

    // Build the ladder K_j = k_min · 2^j, j = 0..=n_doublings, guarding overflow.
    let mut widths: Vec<usize> = Vec::with_capacity(cfg.n_doublings + 1);
    for j in 0..=cfg.n_doublings {
        let k = cfg
            .k_min
            .checked_shl(j as u32)
            .ok_or_else(|| format!("dimension_spectrometer: rung width k_min·2^{j} overflows usize"))?;
        widths.push(k);
    }

    let mut rungs: Vec<(usize, f64)> = Vec::with_capacity(widths.len());
    for &k in &widths {
        let mut rung_cfg = cfg.dict;
        rung_cfg.n_atoms = k;
        // The VQ exponent −2/d is a single-atom rate; force it on every rung.
        rung_cfg.active = 1;
        let fit = fit_sparse_dictionary(data, &rung_cfg)
            .map_err(|e| format!("dimension_spectrometer: fit at K={k} failed: {e}"))?;
        let loss = rung_loss(&fit, data);
        rungs.push((k, loss));
    }

    analyze_ladder(&rungs, data.nrows())
}

/// Derived points-per-atom threshold `τ = max(d̂²/(2 ln 2), 2)`.
///
/// A single-atom cell fit to `n = N/K` points underestimates the population cell
/// residual by a leading factor `≈ 1 − d/n` (the atom captures the best of `≈ d`
/// local tangent directions estimated from `n` points). Over one octave that
/// steepens the log-log slope by `≈ (d/ln 2)·(K/N)`; requiring that finite-sample
/// slope bias to stay below the law's per-octave signal slope `2/d` gives
/// `n ≥ d²/(2 ln 2)`. The floor of 2 keeps a cell's residual non-degenerate
/// (residual degrees of freedom `> 0` needs more than one point per cell). Every
/// constant is derived (see the module docs); nothing here is tuned.
fn threshold_points_per_atom(d_hat: f64) -> f64 {
    let statistical = (d_hat * d_hat) / (2.0 * std::f64::consts::LN_2);
    statistical.max(2.0)
}

/// Select the reliable rung window by a short deterministic fixed-point: keep the
/// rungs whose points-per-atom `N/K` meets [`threshold_points_per_atom`] evaluated
/// at the current `d̂`, refit to update `d̂`, and repeat until the retained set
/// stops changing. Returns rung indices (ascending `K`). If the seed `d̂` is not a
/// usable positive dimension (e.g. the fit is floor-saturated) the whole ladder is
/// returned — windowing a non-resolved slope would be meaningless.
fn select_stable_window(rungs: &[(usize, f64)], n_rows: usize, seed_d: f64) -> Vec<usize> {
    let full: Vec<usize> = (0..rungs.len()).collect();
    if !(seed_d.is_finite() && seed_d > 0.0) {
        return full;
    }
    let mut d = seed_d;
    let mut prev: Vec<usize> = full.clone();
    for _ in 0..8 {
        let tau = threshold_points_per_atom(d);
        let mut window: Vec<usize> = (0..rungs.len())
            .filter(|&i| (n_rows as f64) / (rungs[i].0 as f64) >= tau)
            .collect();
        // A slope needs at least three rungs; if the threshold is stricter than the
        // ladder can honour, fall back to the three least token-starved (smallest-K)
        // rungs and let `small_nk_regime` carry the warning.
        if window.len() < 3 {
            window = (0..rungs.len().min(3)).collect();
        }
        let sub: Vec<(usize, f64)> = window.iter().map(|&i| rungs[i]).collect();
        match fit_scaling_law(&sub) {
            Ok(law) if law.d_hat.is_finite() && law.d_hat > 0.0 => {
                if window == prev {
                    return window;
                }
                d = law.d_hat;
                prev = window;
            }
            _ => return window,
        }
    }
    prev
}

/// Fit the scaling law over the full ladder, the derived stable window, and the
/// drop-last ladder, and assemble the report. `n_rows` is `N` (for points-per-atom).
fn analyze_ladder(rungs: &[(usize, f64)], n_rows: usize) -> Result<SpectrometerReport, String> {
    if rungs.len() < 3 {
        return Err("analyze_ladder requires at least 3 rungs".to_string());
    }
    let full = fit_scaling_law(rungs)?;

    // Stable window: seed the fixed-point with the full-ladder d̂ (unless the slope
    // is not resolved, in which case windowing is meaningless and we keep it full).
    let seed_d = if full.floor_saturated {
        f64::NAN
    } else {
        full.d_hat
    };
    let window = select_stable_window(rungs, n_rows, seed_d);
    let window_sub: Vec<(usize, f64)> = window.iter().map(|&i| rungs[i]).collect();
    let stable = fit_scaling_law(&window_sub)?;

    // Drop-last robustness probe (only meaningful with a rung to spare).
    let d_hat_drop_last = if rungs.len() >= 4 {
        fit_scaling_law(&rungs[..rungs.len() - 1])
            .map(|law| law.d_hat)
            .unwrap_or(full.d_hat)
    } else {
        full.d_hat
    };

    let points_per_atom: Vec<f64> = rungs
        .iter()
        .map(|&(k, _)| n_rows as f64 / k as f64)
        .collect();
    let d_for_threshold = if stable.d_hat.is_finite() && stable.d_hat > 0.0 {
        stable.d_hat
    } else {
        full.d_hat
    };
    let threshold = threshold_points_per_atom(d_for_threshold);
    let small_nk_regime =
        window.len() < rungs.len() || points_per_atom.iter().any(|&r| r < threshold);

    let lo_k = rungs[window[0]].0;
    let hi_k = rungs[window[window.len() - 1]].0;

    Ok(SpectrometerReport {
        rungs: rungs.to_vec(),
        n_rows,
        points_per_atom,
        noise_floor: stable.sigma2,
        slope: stable.slope,
        slope_se: stable.slope_se,
        d_hat: stable.d_hat,
        d_hat_se: stable.d_hat_se,
        d_hat_full_ladder: full.d_hat,
        d_hat_drop_last,
        stable_window_lo_k: lo_k,
        stable_window_hi_k: hi_k,
        stable_rung_count: window.len(),
        min_points_per_atom: threshold,
        floor_saturated: stable.floor_saturated,
        small_nk_regime,
    })
}

/// Mean over rows of the squared single-atom reconstruction residual
/// `‖x_i − Σ_j c_{ij} d_{a_{ij}}‖²`, accumulated in `f64`. Reconstruction is done
/// per row into a length-`P` scratch, so no `N×P` (let alone `N×K`) object is
/// formed.
fn rung_loss(fit: &crate::sparse_dict::SparseDictFit, data: ArrayView2<'_, f32>) -> f64 {
    let n = data.nrows();
    let p = data.ncols();
    let s = fit.indices.ncols();
    let mut recon = vec![0.0f64; p];
    let mut acc = 0.0f64;
    for i in 0..n {
        for c in 0..p {
            recon[c] = 0.0;
        }
        for j in 0..s {
            let cj = fit.codes[[i, j]] as f64;
            if cj == 0.0 {
                continue;
            }
            let atom = fit.indices[[i, j]] as usize;
            let drow = fit.decoder.row(atom);
            for c in 0..p {
                recon[c] += cj * drow[c] as f64;
            }
        }
        let xi = data.row(i);
        let mut ri = 0.0f64;
        for c in 0..p {
            let r = xi[c] as f64 - recon[c];
            ri += r * r;
        }
        acc += ri;
    }
    acc / n as f64
}

/// The fitted scaling law and the quantities derived from it.
struct ScalingLaw {
    sigma2: f64,
    slope: f64,
    slope_se: f64,
    d_hat: f64,
    d_hat_se: f64,
    floor_saturated: bool,
}

/// A single ordinary-least-squares line `y = intercept + slope · t`.
struct LineFit {
    slope: f64,
    rss: f64,
}

/// Fit `y = log(L − σ²)` against `t = log K` by ordinary least squares for a
/// FIXED `σ²`. `t`, its mean `t_bar`, and `Stt = Σ(t − t̄)²` are precomputed by the
/// caller (they do not depend on `σ²`). Returns `None` if any excess loss is
/// non-positive at this `σ²` (the fit is undefined there).
fn ols_log_excess(losses: &[f64], t: &[f64], t_bar: f64, stt: f64, sigma2: f64) -> Option<LineFit> {
    let nrung = losses.len();
    let mut y = vec![0.0f64; nrung];
    let mut y_bar = 0.0f64;
    for i in 0..nrung {
        let excess = losses[i] - sigma2;
        if excess <= 0.0 {
            return None;
        }
        let yi = excess.ln();
        y[i] = yi;
        y_bar += yi;
    }
    y_bar /= nrung as f64;

    // slope = Σ(t − t̄)(y − ȳ) / Σ(t − t̄)²; since Σ(t − t̄) = 0, the numerator is
    // Σ(t − t̄) y.
    let mut sty = 0.0f64;
    for i in 0..nrung {
        sty += (t[i] - t_bar) * y[i];
    }
    let slope = sty / stt;
    let intercept = y_bar - slope * t_bar;

    let mut rss = 0.0f64;
    for i in 0..nrung {
        let resid = y[i] - intercept - slope * t[i];
        rss += resid * resid;
    }
    Some(LineFit { slope, rss })
}

/// Profile out the noise floor `σ²` and fit the log-log slope, then invert it to a
/// dimension estimate with its delta-method standard error.
fn fit_scaling_law(rungs: &[(usize, f64)]) -> Result<ScalingLaw, String> {
    let nrung = rungs.len();
    if nrung < 3 {
        return Err("fit_scaling_law requires at least 3 rungs".to_string());
    }
    let losses: Vec<f64> = rungs.iter().map(|&(_, l)| l).collect();
    if !losses.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return Err("fit_scaling_law: per-rung losses must be finite and non-negative".to_string());
    }
    let t: Vec<f64> = rungs.iter().map(|&(k, _)| (k as f64).ln()).collect();
    let t_bar = t.iter().sum::<f64>() / nrung as f64;
    let stt: f64 = t.iter().map(|&ti| (ti - t_bar) * (ti - t_bar)).sum();
    if stt <= 0.0 {
        return Err("fit_scaling_law: log-K design is degenerate (all rungs equal width)".to_string());
    }

    // The plateau cannot exceed the smallest achieved loss; a variance floor is
    // non-negative. This DERIVED bracket [0, L_min) is the golden-section domain.
    let l_min = losses.iter().cloned().fold(f64::INFINITY, f64::min);
    // Keep the upper end strictly below L_min so the smallest excess stays positive
    // (log-defined). The gap is a floating-point safety margin, not a tuned knob.
    let hi = l_min * (1.0 - 1.0e-9);
    let lo = 0.0f64;

    let objective = |sigma2: f64| -> f64 {
        match ols_log_excess(&losses, &t, t_bar, stt, sigma2) {
            Some(fit) => fit.rss,
            None => f64::INFINITY,
        }
    };
    let sigma2 = golden_section_min(objective, lo, hi);

    let fit = ols_log_excess(&losses, &t, t_bar, stt, sigma2)
        .ok_or_else(|| "fit_scaling_law: profiled σ² left an undefined log-excess".to_string())?;
    let slope = fit.slope;

    // Standard error of the OLS slope: s² = RSS/(n−3), SE(m) = sqrt(s² / Stt).
    // The fitted model is L_k = σ² + c·K^m — THREE parameters, with σ² profiled
    // out by the golden section over the SAME rungs (variable projection), so
    // the residual degrees of freedom are n−3, not n−2. With n−2 an nrung=3
    // exact fit (RSS≈0 by construction) would report a finite ~0 SE and an
    // unsaturated floor from 3 points fitting 3 parameters.
    let dof = (nrung as f64) - 3.0;
    let s2 = if dof > 0.0 { fit.rss / dof } else { f64::INFINITY };
    let slope_se = (s2 / stt).sqrt();

    let d_hat = -2.0 / slope;
    // Delta method: d = −2/m ⇒ dd/dm = 2/m², SE(d) = |2/m²| · SE(m).
    let d_hat_se = (2.0 / (slope * slope)) * slope_se;

    // Saturated when the slope's confidence interval contains zero (|m| within the
    // 95% half-width) or the slope is non-negative (no decay resolved at all).
    let ci_half = NORMAL_95_QUANTILE * slope_se;
    let floor_saturated = !(slope < 0.0) || slope.abs() <= ci_half;

    Ok(ScalingLaw {
        sigma2,
        slope,
        slope_se,
        d_hat,
        d_hat_se,
        floor_saturated,
    })
}

/// Minimize a unimodal 1-D function on `[lo, hi]` by golden-section bracketing,
/// contracting until the bracket is below [`PROFILE_REL_TOL`] (relative) or the
/// safety cap [`PROFILE_MAX_ITERS`] is hit. Returns the bracket midpoint.
fn golden_section_min<F: FnMut(f64) -> f64>(mut f: F, mut lo: f64, mut hi: f64) -> f64 {
    // Inverse golden ratio 1/φ = (√5 − 1)/2.
    let inv_phi = (5.0f64.sqrt() - 1.0) / 2.0;
    if !(hi > lo) {
        return lo;
    }
    let mut c = hi - inv_phi * (hi - lo);
    let mut d = lo + inv_phi * (hi - lo);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..PROFILE_MAX_ITERS {
        if (hi - lo).abs() <= PROFILE_REL_TOL * (1.0 + lo.abs() + hi.abs()) {
            break;
        }
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - inv_phi * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + inv_phi * (hi - lo);
            fd = f(d);
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic splitmix64 state stepper.
    fn split_next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) from the top 53 bits.
    fn split_unit(state: &mut u64) -> f64 {
        (split_next(state) >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box–Muller (one draw; the paired value is discarded for
    /// simplicity — this is test data generation, not a hot path).
    fn split_normal(state: &mut u64) -> f64 {
        let u1 = split_unit(state).max(1.0e-12);
        let u2 = split_unit(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A seeded orthonormal frame in `p` dimensions: the eigenvectors of a seeded
    /// symmetric Gaussian matrix (eigenvectors of a real symmetric matrix are
    /// orthonormal). Columns `0..cols` are returned as the embedding directions.
    /// This matches the house pattern (`sparse_dict::tests::planted`) of sourcing
    /// orthonormal directions from an eigendecomposition rather than vendoring a QR.
    fn orthonormal_frame(p: usize, cols: usize, seed: u64) -> Array2<f32> {
        use gam_linalg::faer_ndarray::FaerEigh;
        let mut state = seed;
        let mut a = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                a[[i, j]] = split_normal(&mut state);
            }
        }
        let sym = &a + &a.t();
        let (_evals, evecs) = sym.eigh(faer::Side::Lower).expect("orthonormal frame eig");
        let mut frame = Array2::<f32>::zeros((p, cols));
        for c in 0..cols {
            for r in 0..p {
                frame[[r, c]] = evecs[[r, c]] as f32;
            }
        }
        frame
    }

    /// Points on a unit circle (d = 1) embedded in `p` dims via a seeded
    /// orthonormal 2-plane, with small isotropic additive noise.
    fn circle(n: usize, p: usize, noise: f32, seed: u64) -> Array2<f32> {
        let frame = orthonormal_frame(p, 2, seed);
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            let theta = std::f64::consts::TAU * split_unit(&mut state);
            let c0 = theta.cos() as f32;
            let c1 = theta.sin() as f32;
            for r in 0..p {
                let signal = c0 * frame[[r, 0]] + c1 * frame[[r, 1]];
                let eps = noise * split_normal(&mut state) as f32;
                x[[i, r]] = signal + eps;
            }
        }
        x
    }

    /// Points on a 2-torus (d = 2): a product of two circles with independent
    /// uniform phases, embedded in `p` dims via a seeded orthonormal 4-plane, with
    /// small isotropic additive noise.
    fn torus(n: usize, p: usize, noise: f32, seed: u64) -> Array2<f32> {
        let frame = orthonormal_frame(p, 4, seed);
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            let phi1 = std::f64::consts::TAU * split_unit(&mut state);
            let phi2 = std::f64::consts::TAU * split_unit(&mut state);
            let a0 = phi1.cos() as f32;
            let a1 = phi1.sin() as f32;
            let a2 = phi2.cos() as f32;
            let a3 = phi2.sin() as f32;
            for r in 0..p {
                let signal = a0 * frame[[r, 0]]
                    + a1 * frame[[r, 1]]
                    + a2 * frame[[r, 2]]
                    + a3 * frame[[r, 3]];
                let eps = noise * split_normal(&mut state) as f32;
                x[[i, r]] = signal + eps;
            }
        }
        x
    }

    /// Shared per-rung fit template for the manifold tests: pinned to CPU, single
    /// atom is forced by the spectrometer regardless.
    fn dict_template() -> SparseDictConfig {
        SparseDictConfig {
            n_atoms: 1,
            active: 1,
            minibatch: 1024,
            max_epochs: 25,
            score_tile: 256,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-6,
            score_mode: gam_gpu::GpuPolicy::Off,
        }
    }

    fn assert_losses_decreasing(rungs: &[(usize, f64)]) {
        for w in rungs.windows(2) {
            assert!(
                w[1].1 <= w[0].1 + 1.0e-9,
                "loss must not increase with K: K={} loss={} then K={} loss={}",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
    }

    #[test]
    fn spectrometer_recovers_circle_dimension_one() {
        // d = 1 falls off steeply (K^−2), so it approaches any floor fast; keep the
        // aperture at 4..=128 and the noise tiny so the excess stays above the floor
        // across the ladder.
        let p = 64usize;
        let x = circle(4000, p, 3.0e-4, 0x00C1_2345);
        let cfg = SpectrometerConfig {
            k_min: 4,
            n_doublings: 5, // rungs 4,8,16,32,64,128
            dict: dict_template(),
        };
        let report = dimension_spectrometer(x.view(), &cfg).expect("circle spectrometer");
        assert_eq!(report.rungs.len(), 6);
        assert_losses_decreasing(&report.rungs);
        assert!(
            report.slope < 0.0,
            "slope must be negative (loss decays in K), got {}",
            report.slope
        );
        assert!(
            !report.floor_saturated,
            "circle ladder should resolve a slope (not floor-saturated); slope={} se={}",
            report.slope, report.slope_se
        );
        assert!(
            (report.d_hat - 1.0).abs() < 0.5,
            "d̂ should recover 1 for the circle, got {} (slope {}, σ²={})",
            report.d_hat,
            report.slope,
            report.noise_floor
        );
        // Well-sampled (min N/K = 4000/128 ≈ 31 ≫ τ): the whole ladder is the stable
        // window, so no drift is flagged and the full-ladder d̂ equals the primary.
        assert!(
            !report.small_nk_regime,
            "well-sampled circle must not flag small-N/K (stable rungs {} of {})",
            report.stable_rung_count,
            report.rungs.len()
        );
        assert_eq!(report.stable_rung_count, report.rungs.len());
        assert!((report.d_hat - report.d_hat_full_ladder).abs() < 1.0e-9);
    }

    #[test]
    fn spectrometer_recovers_torus_dimension_two() {
        // d = 2 falls off slowly (K^−1), staying well above the floor; a full
        // 4..=256 aperture is comfortable.
        let p = 64usize;
        let x = torus(6000, p, 3.0e-4, 0x00D2_9876);
        let cfg = SpectrometerConfig {
            k_min: 4,
            n_doublings: 6, // rungs 4,8,16,32,64,128,256
            dict: dict_template(),
        };
        let report = dimension_spectrometer(x.view(), &cfg).expect("torus spectrometer");
        assert_eq!(report.rungs.len(), 7);
        assert_losses_decreasing(&report.rungs);
        assert!(
            report.slope < 0.0,
            "slope must be negative (loss decays in K), got {}",
            report.slope
        );
        assert!(
            !report.floor_saturated,
            "torus ladder should resolve a slope (not floor-saturated); slope={} se={}",
            report.slope, report.slope_se
        );
        assert!(
            (report.d_hat - 2.0).abs() < 0.5,
            "d̂ should recover 2 for the 2-torus, got {} (slope {}, σ²={})",
            report.d_hat,
            report.slope,
            report.noise_floor
        );
        // Well-sampled (min N/K = 6000/256 ≈ 23 ≫ τ): full ladder is the stable window.
        assert!(
            !report.small_nk_regime,
            "well-sampled torus must not flag small-N/K (stable rungs {} of {})",
            report.stable_rung_count,
            report.rungs.len()
        );
        assert_eq!(report.stable_rung_count, report.rungs.len());
    }

    #[test]
    fn scaling_law_recovers_planted_power_law_with_floor() {
        // A synthetic, fit-free check of the regression core: losses built as
        // σ² + c·K^m with a known slope must be inverted to the right dimension, and
        // the profiled σ² must recover the planted floor.
        let sigma2_true = 0.02f64;
        let c = 1.5f64;
        let m_true = -1.0f64; // d = 2
        let mut rungs: Vec<(usize, f64)> = Vec::new();
        let mut k = 4usize;
        for _ in 0..7 {
            let loss = sigma2_true + c * (k as f64).powf(m_true);
            rungs.push((k, loss));
            k *= 2;
        }
        let law = fit_scaling_law(&rungs).expect("scaling law fit");
        assert!(
            (law.slope - m_true).abs() < 1.0e-3,
            "recovered slope {} should match planted {m_true}",
            law.slope
        );
        assert!(
            (law.d_hat - 2.0).abs() < 1.0e-2,
            "recovered d̂ {} should match planted 2",
            law.d_hat
        );
        assert!(
            (law.sigma2 - sigma2_true).abs() < 1.0e-3,
            "profiled σ² {} should recover planted floor {sigma2_true}",
            law.sigma2
        );
        assert!(!law.floor_saturated, "clean power law must not read as saturated");
    }

    #[test]
    fn flat_losses_flag_floor_saturation() {
        // Losses at the floor (no decay in K) must flag saturation, not report a
        // spurious finite dimension.
        let rungs: Vec<(usize, f64)> =
            [4, 8, 16, 32, 64].iter().map(|&k| (k, 0.05f64)).collect();
        let law = fit_scaling_law(&rungs).expect("flat scaling law fit");
        assert!(
            law.floor_saturated,
            "flat losses must flag floor saturation (slope {} se {})",
            law.slope, law.slope_se
        );
    }

    #[test]
    fn stable_window_resists_high_k_overfit_drift() {
        // Reproduce the real-data pathology fit-free, then check the hardening. Build
        // a d=8 ladder whose losses follow the population VQ rate σ² + c·K^(−2/d) but
        // with the DERIVED finite-sample overfit factor (1 − d/n) applied per rung
        // (n = N/K). At small N/K (large K) that factor collapses the held-in loss,
        // steepening the tail and biasing the FULL-ladder d̂ downward — the 33→6 slide.
        // The stable window must drop those token-starved rungs and recover a d̂ far
        // closer to the truth, and the small-N/K regime must be flagged.
        let d_true = 8.0f64;
        let n_rows = 2000usize;
        let sigma2 = 1.0e-3f64;
        let c = 1.0f64;
        let mut rungs: Vec<(usize, f64)> = Vec::new();
        let mut k = 4usize;
        for _ in 0..9 {
            // rungs 4 .. 1024
            let population = sigma2 + c * (k as f64).powf(-2.0 / d_true);
            let n_cell = n_rows as f64 / k as f64;
            // Derived overfit shrinkage 1 − d/n, clamped away from ≤0 (a cell with
            // n ≤ d overfits to ~zero residual — the degenerate token-starved regime).
            let overfit = (1.0 - d_true / n_cell).max(0.05);
            rungs.push((k, population * overfit));
            k *= 2;
        }

        let report = analyze_ladder(&rungs, n_rows).expect("ladder analysis");

        // The tail is token-starved (N/K = 2000/1024 ≈ 2 ≪ τ(8) ≈ 46), so the drift is
        // real and must be flagged, and the stable window must be a strict subset.
        assert!(
            report.small_nk_regime,
            "token-starved tail must flag small-N/K regime"
        );
        assert!(
            report.stable_rung_count < report.rungs.len(),
            "stable window must drop the token-starved rungs (kept {} of {})",
            report.stable_rung_count,
            report.rungs.len()
        );
        assert!(
            report.stable_window_hi_k < rungs[rungs.len() - 1].0,
            "stable window must exclude the largest-K rung ({} vs top {})",
            report.stable_window_hi_k,
            rungs[rungs.len() - 1].0
        );
        // Full-ladder d̂ is biased DOWN by the overfit tail; the stable window corrects
        // it substantially UPWARD, landing much closer to the truth.
        assert!(
            report.d_hat_full_ladder < d_true - 1.0,
            "full-ladder d̂ {} should be biased below the true d={d_true}",
            report.d_hat_full_ladder
        );
        assert!(
            report.d_hat > report.d_hat_full_ladder + 1.0,
            "stable-window d̂ {} must correct the full-ladder d̂ {} upward",
            report.d_hat,
            report.d_hat_full_ladder
        );
        assert!(
            (report.d_hat - d_true).abs() < (report.d_hat_full_ladder - d_true).abs(),
            "stable-window d̂ {} must be closer to d={d_true} than full-ladder d̂ {}",
            report.d_hat,
            report.d_hat_full_ladder
        );
        // Drop-last is a reported robustness probe (not a monotone de-bias guarantee —
        // beyond the overfit crater the tail returns to the shallow population slope,
        // so removing one rung need not move d̂ in a fixed direction). It must be a
        // finite, usable number.
        assert!(
            report.d_hat_drop_last.is_finite() && report.d_hat_drop_last > 0.0,
            "drop-last d̂ must be a finite positive estimate, got {}",
            report.d_hat_drop_last
        );
        // Points-per-atom is exposed so the drift is inspectable, aligned to the ladder.
        assert_eq!(report.points_per_atom.len(), report.rungs.len());
        assert!(
            (report.points_per_atom[0] - n_rows as f64 / rungs[0].0 as f64).abs() < 1.0e-9
        );
    }

    #[test]
    fn threshold_scales_like_dimension_squared() {
        // The derived threshold τ = max(d²/(2 ln2), 2): permissive for low d (a circle
        // stays reliable to large K), strict for high d (a high-dimensional manifold
        // needs many points per atom — exactly where real data drifts).
        assert!(
            (threshold_points_per_atom(1.0) - 2.0).abs() < 1.0e-12,
            "d=1 threshold floors at 2, got {}",
            threshold_points_per_atom(1.0)
        );
        let t_low = threshold_points_per_atom(4.0);
        let t_high = threshold_points_per_atom(16.0);
        // Quadratic scaling: quadrupling d multiplies the threshold by ~16.
        assert!(
            (t_high / t_low - 16.0).abs() < 1.0e-6,
            "threshold must scale like d² (ratio {} for d 4→16)",
            t_high / t_low
        );
        assert!(
            t_high > 180.0,
            "a d=16 manifold must demand many points per atom, got {t_high}"
        );
    }
}

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
#[derive(Clone, Debug)]
pub struct SpectrometerReport {
    /// Per-rung `(K, mean per-row reconstruction loss)` in ladder order (ascending
    /// `K`). The loss is the mean over rows of the squared reconstruction residual
    /// `‖x_i − c_i d_{a_i}‖²`, accumulated in `f64`.
    pub rungs: Vec<(usize, f64)>,
    /// Estimated noise floor `σ²` (the loss plateau), profiled out of the
    /// log-log regression. Same units as the per-rung loss.
    pub noise_floor: f64,
    /// Fitted slope `m` of `log(L(K) − σ²)` on `log K`.
    pub slope: f64,
    /// Standard error of the slope from the log-log regression residuals.
    pub slope_se: f64,
    /// Intrinsic-dimension estimate `d̂ = −2/m`.
    pub d_hat: f64,
    /// Delta-method standard error of `d̂`: `|2/m²| · SE(m)`.
    pub d_hat_se: f64,
    /// `true` when the slope's confidence interval contains zero — the last rungs
    /// are statistically indistinguishable from the noise floor, the log-log slope
    /// is not resolved, and `d̂ = −2/m` is unreliable (near a division by zero).
    pub floor_saturated: bool,
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
    // Need at least three rungs: two regression parameters (slope, intercept) plus
    // one residual degree of freedom for the slope's standard error.
    if cfg.n_doublings < 2 {
        return Err(
            "dimension_spectrometer requires n_doublings >= 2 (>= 3 rungs) to estimate a slope \
             and its standard error"
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

    let law = fit_scaling_law(&rungs)?;
    Ok(SpectrometerReport {
        rungs,
        noise_floor: law.sigma2,
        slope: law.slope,
        slope_se: law.slope_se,
        d_hat: law.d_hat,
        d_hat_se: law.d_hat_se,
        floor_saturated: law.floor_saturated,
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

    // Standard error of the OLS slope: s² = RSS/(n−2), SE(m) = sqrt(s² / Stt).
    let dof = (nrung as f64) - 2.0;
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
            score_mode: gam_gpu::GpuMode::Off,
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
}

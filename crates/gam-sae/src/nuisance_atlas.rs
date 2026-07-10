//! Nuisance-atlas pre-pass: regress the *known* nuisance manifolds out of an
//! activation matrix before any dictionary / coordinate charting runs, so the
//! atoms a chart later discovers are semantic rather than positional or
//! frequency echoes.
//!
//! # Why this is a prerequisite
//!
//! A residual-stream activation `x_i ∈ ℝᵖ` for token `i` carries, on top of the
//! semantic content we want to chart, two large and *known* nuisance signals:
//!
//! * **the rotary / positional helix** — RoPE rotates coordinate pairs by a
//!   position-dependent angle, so token position paints a smooth low-dimensional
//!   helical manifold across the residual stream. A circle/coordinate chart run
//!   naively will happily "discover" this helix and report it as a feature; it is
//!   an artifact of position, not meaning.
//! * **token-frequency directions** — unigram (log-)frequency correlates with a
//!   direction (and a little curvature) in activation space; rare-vs-common is a
//!   nuisance axis that otherwise leaks into every chart.
//!
//! Both are *linear in known covariates* (position → Fourier features at the
//! rotary frequencies; log-frequency → a low-degree polynomial), so the honest
//! thing is to **project them out in closed form** and chart the residual. This
//! module builds that nuisance design and performs the ordinary-least-squares
//! regress-out, reporting the fraction of activation variance the atlas absorbs.
//!
//! # The math (closed form, no finite differences)
//!
//! Given activations `X` (`N×P`, f32 lifted to f64) and a nuisance design
//! `Z` (`N×M`, with an intercept column so the projection also removes the mean),
//! the OLS fit is the normal-equations solve
//!
//! ```text
//!   B = (ZᵀZ)⁻¹ ZᵀX      (M×P),      X̂ = Z B,      R = X − X̂.
//! ```
//!
//! `ZᵀZ` is factorised by the LLT→LDLT→LBLT symmetric fallback (a rank-deficient
//! design — e.g. more nuisance columns than distinct positions — degrades
//! gracefully and is flagged, never panics). The **variance absorbed** is the
//! coefficient of determination of the nuisance regression, aggregated over
//! output dimensions and centred (the intercept makes `R` column-mean-zero):
//!
//! ```text
//!   absorbed = 1 − Σ_j ‖R_{·j}‖² / Σ_j ‖X_{·j} − x̄_j‖².
//! ```
//!
//! Because the projector `Z(ZᵀZ)⁻¹Zᵀ` is idempotent, re-fitting the same design
//! on the residual absorbs ≈ 0 further variance and `Zᵀ R = 0` exactly (up to
//! round-off) — both pinned by the tests. With a purely-semantic input the atlas
//! absorbs only the `M/N` in-sample overfit floor, so a large absorbed fraction
//! is genuine nuisance structure, not the regression fitting noise.

use gam_linalg::faer_ndarray::{FaerEigh, fast_ab, fast_ata, fast_atb};
use ndarray::{Array2, ArrayView2};

/// The standard RoPE frequency base `θ_base = 10000` used by Qwen3 and most
/// GPT-style rotary embeddings. It is the *model's* documented constant (pass
/// the value the activations were produced with), not a tuning knob.
pub const DEFAULT_ROPE_BASE: f64 = 10000.0;

/// Configuration for building the nuisance design from per-token covariates.
#[derive(Clone, Copy, Debug)]
pub struct NuisanceAtlasConfig {
    /// Number of positional Fourier harmonic pairs `(cos θ_k·p, sin θ_k·p)` built
    /// from token position at RoPE-style geometric frequencies. `0` disables the
    /// positional block.
    pub positional_harmonics: usize,
    /// RoPE frequency base `θ_base` (see [`DEFAULT_ROPE_BASE`]); the `k`-th
    /// harmonic uses `θ_k = θ_base^(−k/H)`, spanning one radian/token down to
    /// `θ_base^(−1)` across `H` harmonics — the rotary geometry, not a fit grid.
    pub rope_base: f64,
    /// Polynomial degree in standardized token log-frequency (`1` = a single
    /// frequency direction; `≥2` adds curvature). `0` disables the frequency
    /// block.
    pub token_frequency_degree: usize,
    /// Ridge added to the `ZᵀZ` diagonal (Tikhonov). `0` is plain OLS; a small
    /// positive value keeps a near-collinear design well posed.
    pub ridge: f64,
}

impl Default for NuisanceAtlasConfig {
    fn default() -> Self {
        Self {
            positional_harmonics: 8,
            rope_base: DEFAULT_ROPE_BASE,
            token_frequency_degree: 2,
            ridge: 0.0,
        }
    }
}

/// Result of a nuisance-atlas regress-out.
#[derive(Clone, Debug)]
pub struct NuisanceAtlasFit {
    /// OLS coefficients `B` (`M×P`), one column of loadings per activation dim.
    pub coefficients: Array2<f64>,
    /// Fraction of total (centred) activation variance the atlas absorbs — the
    /// aggregate `R²` of the nuisance regression.
    pub variance_absorbed: f64,
    /// Per-activation-dimension absorbed fraction (length `P`).
    pub per_dim_absorbed: Vec<f64>,
    /// Number of nuisance design columns `M` (including the intercept).
    pub n_design: usize,
    /// Set when `ZᵀZ` was not positive-definite (the LLT factor failed and a
    /// LDLT/LBLT fallback was used) — the design is rank-deficient / collinear.
    pub design_rank_deficient: bool,
}

impl NuisanceAtlasFit {
    /// The regressed-out residual `R = X − Z B` (`N×P`, back in f32), the
    /// nuisance-free activations to hand to the downstream chart.
    pub fn residual(&self, x: ArrayView2<'_, f32>, design: ArrayView2<'_, f64>) -> Array2<f32> {
        let xhat = fast_ab(&design.to_owned(), &self.coefficients);
        let mut out = Array2::<f32>::zeros(x.raw_dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                out[[i, j]] = (x[[i, j]] as f64 - xhat[[i, j]]) as f32;
            }
        }
        out
    }
}

/// Positional Fourier features at RoPE-style geometric frequencies: for each of
/// `harmonics` frequencies `θ_k = base^(−k/H)` (`k = 0..H`), the two columns
/// `cos(θ_k·p)` and `sin(θ_k·p)`. Returns an `N × 2H` matrix (empty `N×0` when
/// `harmonics == 0`). Positions are the per-token sequence positions.
pub fn positional_fourier_features(positions: &[i64], harmonics: usize, base: f64) -> Array2<f64> {
    let n = positions.len();
    let mut z = Array2::<f64>::zeros((n, 2 * harmonics));
    if harmonics == 0 {
        return z;
    }
    let h = harmonics as f64;
    for k in 0..harmonics {
        let theta = base.powf(-(k as f64) / h);
        for (i, &p) in positions.iter().enumerate() {
            let ang = theta * p as f64;
            z[[i, 2 * k]] = ang.cos();
            z[[i, 2 * k + 1]] = ang.sin();
        }
    }
    z
}

/// Token-frequency polynomial features: the per-token log-frequency standardised
/// to zero mean / unit variance, raised to powers `1..=degree`. Returns an
/// `N × degree` matrix (empty `N×0` when `degree == 0`). A degenerate (constant)
/// log-frequency yields all-zero columns.
pub fn token_frequency_features(log_freq: &[f64], degree: usize) -> Array2<f64> {
    let n = log_freq.len();
    let mut z = Array2::<f64>::zeros((n, degree));
    if degree == 0 || n == 0 {
        return z;
    }
    let mean = log_freq.iter().sum::<f64>() / n as f64;
    let var = log_freq
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f64>()
        / n as f64;
    let sd = var.sqrt();
    if sd <= 0.0 {
        return z; // constant frequency carries no direction
    }
    for (i, &v) in log_freq.iter().enumerate() {
        let s = (v - mean) / sd;
        let mut power = 1.0;
        for d in 0..degree {
            power *= s;
            z[[i, d]] = power;
        }
    }
    z
}

/// Assemble the full nuisance design `Z` (`N×M`) for the given per-token
/// covariates: a leading intercept column, then the positional Fourier block and
/// the token-frequency polynomial block per `config`. `positions` and `log_freq`
/// must each have length `N` (either may be ignored by setting its block size to
/// zero in the config).
pub fn build_nuisance_design(
    n_rows: usize,
    positions: &[i64],
    log_freq: &[f64],
    config: &NuisanceAtlasConfig,
) -> Result<Array2<f64>, String> {
    if positions.len() != n_rows {
        return Err(format!(
            "build_nuisance_design: positions has {} entries but N = {n_rows}",
            positions.len()
        ));
    }
    if log_freq.len() != n_rows {
        return Err(format!(
            "build_nuisance_design: log_freq has {} entries but N = {n_rows}",
            log_freq.len()
        ));
    }
    let pos = positional_fourier_features(positions, config.positional_harmonics, config.rope_base);
    let freq = token_frequency_features(log_freq, config.token_frequency_degree);
    let m = 1 + pos.ncols() + freq.ncols();
    let mut z = Array2::<f64>::zeros((n_rows, m));
    for i in 0..n_rows {
        z[[i, 0]] = 1.0; // intercept
        for c in 0..pos.ncols() {
            z[[i, 1 + c]] = pos[[i, c]];
        }
        for c in 0..freq.ncols() {
            z[[i, 1 + pos.ncols() + c]] = freq[[i, c]];
        }
    }
    Ok(z)
}

/// Regress the nuisance design `Z` (`N×M`) out of activations `X` (`N×P`) in
/// closed form and report the variance absorbed. `Z` should carry an intercept
/// column (as [`build_nuisance_design`] emits) so the projection removes the mean
/// and the absorbed fraction is a centred `R²`.
pub fn fit_nuisance_atlas(
    x: ArrayView2<'_, f32>,
    design: ArrayView2<'_, f64>,
    ridge: f64,
) -> Result<NuisanceAtlasFit, String> {
    let n = x.nrows();
    let p = x.ncols();
    let m = design.ncols();
    if n == 0 || p == 0 {
        return Err("fit_nuisance_atlas: activations must be a non-empty N×P matrix".to_string());
    }
    if design.nrows() != n {
        return Err(format!(
            "fit_nuisance_atlas: design has {} rows but X has {n}",
            design.nrows()
        ));
    }
    if m == 0 {
        return Err("fit_nuisance_atlas: design must have at least one column".to_string());
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err("fit_nuisance_atlas: activations must be finite".to_string());
    }
    if !(ridge.is_finite() && ridge >= 0.0) {
        return Err("fit_nuisance_atlas: ridge must be finite and >= 0".to_string());
    }

    // Lift activations to f64 once (all accumulation is f64, house style).
    let mut xf = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            xf[[i, j]] = x[[i, j]] as f64;
        }
    }
    let design_owned = design.to_owned();

    // Normal equations: G = ZᵀZ (M×M), C = ZᵀX (M×P).
    let mut gram = fast_ata(&design_owned);
    if ridge > 0.0 {
        for d in 0..m {
            gram[[d, d]] += ridge;
        }
    }
    let cross = fast_atb(&design_owned, &xf);

    // Symmetric eigendecomposition of the (tiny, M×M) Gram gives a closed-form
    // pseudo-inverse solve `B = G⁺ C` that degrades gracefully on a rank-deficient
    // design: eigen-directions below the relative floor `ε·M·λ_max` are dropped
    // (their variance is unidentifiable, not fit), and their presence flags the
    // design. The floor is derived from f64 machine epsilon and the problem size,
    // not a tuned constant.
    let (evals, evecs) = gram
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("fit_nuisance_atlas: nuisance Gram eigensolve failed: {e:?}"))?;
    let lam_max = evals.iter().cloned().fold(0.0f64, f64::max);
    let floor = f64::EPSILON * m as f64 * lam_max.max(f64::MIN_POSITIVE);
    let mut design_rank_deficient = false;
    // VᵀC (M×P), scaled by the pseudo-inverse eigenvalues.
    let vtc = fast_atb(&evecs, &cross);
    let mut scaled = Array2::<f64>::zeros((m, p));
    for k in 0..m {
        if evals[k] > floor {
            let inv = 1.0 / evals[k];
            for j in 0..p {
                scaled[[k, j]] = vtc[[k, j]] * inv;
            }
        } else {
            design_rank_deficient = true; // direction dropped from the fit
        }
    }
    // B = V · scaled (M×P).
    let coefficients = fast_ab(&evecs, &scaled);

    // Fitted values and residual → centred R² per dim and aggregate.
    let xhat = fast_ab(&design_owned, &coefficients);
    let mut ss_res = vec![0.0f64; p];
    let mut ss_tot = vec![0.0f64; p];
    let mut means = vec![0.0f64; p];
    for j in 0..p {
        let mut acc = 0.0;
        for i in 0..n {
            acc += xf[[i, j]];
        }
        means[j] = acc / n as f64;
    }
    for j in 0..p {
        let mut sr = 0.0;
        let mut st = 0.0;
        for i in 0..n {
            let r = xf[[i, j]] - xhat[[i, j]];
            sr += r * r;
            let t = xf[[i, j]] - means[j];
            st += t * t;
        }
        ss_res[j] = sr;
        ss_tot[j] = st;
    }
    let per_dim_absorbed: Vec<f64> = (0..p)
        .map(|j| {
            if ss_tot[j] <= 0.0 {
                0.0
            } else {
                1.0 - ss_res[j] / ss_tot[j]
            }
        })
        .collect();
    let tot: f64 = ss_tot.iter().sum();
    let res: f64 = ss_res.iter().sum();
    let variance_absorbed = if tot <= 0.0 { 0.0 } else { 1.0 - res / tot };

    Ok(NuisanceAtlasFit {
        coefficients,
        variance_absorbed,
        per_dim_absorbed,
        n_design: m,
        design_rank_deficient,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG uniform in `[0,1)` (no RNG dep → reproducible).
    fn u01(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(state: &mut u64) -> f64 {
        let u1 = u01(state).max(f64::MIN_POSITIVE);
        let u2 = u01(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A unit vector in ℝᵖ from a pseudo-random draw.
    fn unit_vec(p: usize, state: &mut u64) -> Vec<f64> {
        let mut v: Vec<f64> = (0..p).map(|_| gauss(state)).collect();
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() {
            *x /= nrm;
        }
        v
    }

    /// Column projection fraction of X's variance lying along a unit direction.
    fn variance_along(x: &Array2<f32>, dir: &[f64]) -> f64 {
        let n = x.nrows();
        let p = x.ncols();
        let mut proj = vec![0.0f64; n];
        for i in 0..n {
            let mut d = 0.0;
            for j in 0..p {
                d += x[[i, j]] as f64 * dir[j];
            }
            proj[i] = d;
        }
        let mean = proj.iter().sum::<f64>() / n as f64;
        proj.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
    }

    fn total_centered_energy(x: &Array2<f32>) -> f64 {
        let n = x.nrows();
        let p = x.ncols();
        let mut e = 0.0;
        for j in 0..p {
            let mut mean = 0.0;
            for i in 0..n {
                mean += x[[i, j]] as f64;
            }
            mean /= n as f64;
            for i in 0..n {
                let v = x[[i, j]] as f64 - mean;
                e += v * v;
            }
        }
        e
    }

    /// A planted nuisance+semantic dataset with the ground-truth axes.
    struct Planted {
        x: Array2<f32>,
        positions: Vec<i64>,
        log_freq: Vec<f64>,
        d_pos: Vec<f64>,
        d_freq: Vec<f64>,
        planted_fraction: f64,
    }

    /// Plant X = positional-helix + frequency-direction + semantic + noise, where
    /// the nuisance signals live exactly in the design's column space so the
    /// atlas can absorb them.
    fn plant(n: usize, p: usize, config: &NuisanceAtlasConfig, seed: u64) -> Planted {
        let mut s = seed;
        let d_pos = unit_vec(p, &mut s);
        let mut d_freq = unit_vec(p, &mut s);
        // Orthogonalise d_freq against d_pos so the two nuisance axes are distinct.
        let dot: f64 = d_pos.iter().zip(&d_freq).map(|(a, b)| a * b).sum();
        for (f, pv) in d_freq.iter_mut().zip(&d_pos) {
            *f -= dot * pv;
        }
        let fn2 = d_freq.iter().map(|x| x * x).sum::<f64>().sqrt();
        for f in d_freq.iter_mut() {
            *f /= fn2;
        }
        // Two semantic directions, orthogonalised against BOTH nuisance axes so
        // the residual carries no nuisance-axis leakage from the semantic content
        // (makes the "variance removed" assertions exact).
        let orthonormalize = |v: &mut Vec<f64>, against: &[&Vec<f64>]| {
            for u in against {
                let dot: f64 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                for (vc, uc) in v.iter_mut().zip(u.iter()) {
                    *vc -= dot * uc;
                }
            }
            let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for vc in v.iter_mut() {
                *vc /= nrm;
            }
        };
        let mut d_sem0 = unit_vec(p, &mut s);
        orthonormalize(&mut d_sem0, &[&d_pos, &d_freq]);
        let mut d_sem1 = unit_vec(p, &mut s);
        orthonormalize(&mut d_sem1, &[&d_pos, &d_freq, &d_sem0]);

        // The positional helix scalar must be a function the Fourier block spans:
        // use the lowest RoPE frequency's cosine (in the column space by design).
        let theta0 = config.rope_base.powf(0.0); // k = 0 → θ = 1
        let mut x = Array2::<f32>::zeros((n, p));
        let mut positions = Vec::with_capacity(n);
        let mut log_freq = Vec::with_capacity(n);
        // Track sums + sums-of-squares so the planted fraction is a CENTERED
        // variance ratio, matching the atlas's centred R² (the intercept removes
        // the mean of the frequency covariate, which is not absorbable variance).
        let mut pos_sum = 0.0f64;
        let mut pos_sq = 0.0f64;
        let mut freq_sum = 0.0f64;
        let mut freq_sq = 0.0f64;
        let mut sem_sq = 0.0f64;
        for i in 0..n {
            let pos = (i % 128) as i64; // sequence position within a 128-token window
            positions.push(pos);
            let lf = 3.0 * u01(&mut s); // token log-frequency covariate
            log_freq.push(lf);

            let pos_scalar = 2.0 * (theta0 * pos as f64).cos();
            let freq_scalar = 1.3 * lf;
            let sem0 = 1.5 * gauss(&mut s);
            let sem1 = 1.0 * gauss(&mut s);
            pos_sum += pos_scalar;
            pos_sq += pos_scalar * pos_scalar;
            freq_sum += freq_scalar;
            freq_sq += freq_scalar * freq_scalar;
            sem_sq += sem0 * sem0 + sem1 * sem1;
            for j in 0..p {
                let val = pos_scalar * d_pos[j]
                    + freq_scalar * d_freq[j]
                    + sem0 * d_sem0[j]
                    + sem1 * d_sem1[j]
                    + 0.02 * gauss(&mut s);
                x[[i, j]] = val as f32;
            }
        }
        let nf = n as f64;
        let pos_var = pos_sq - pos_sum * pos_sum / nf;
        let freq_var = freq_sq - freq_sum * freq_sum / nf;
        let sem_var = sem_sq; // semantic scalars are mean-zero by construction
        let planted_fraction = (pos_var + freq_var) / (pos_var + freq_var + sem_var);
        Planted {
            x,
            positions,
            log_freq,
            d_pos,
            d_freq,
            planted_fraction,
        }
    }

    #[test]
    fn atlas_absorbs_planted_nuisance_and_leaves_semantics() {
        let config = NuisanceAtlasConfig::default();
        let (n, p) = (2000usize, 24usize);
        let planted = plant(n, p, &config, 0x51EDC0FFEE_u64);
        let x = &planted.x;
        let design = build_nuisance_design(n, &planted.positions, &planted.log_freq, &config)
            .expect("design");
        let fit = fit_nuisance_atlas(x.view(), design.view(), config.ridge).expect("fit");

        // Absorbed fraction is in the neighbourhood of the planted nuisance
        // fraction (the atlas captures the positional + frequency energy).
        let planted_fraction = planted.planted_fraction;
        assert!(
            (fit.variance_absorbed - planted_fraction).abs() < 0.06,
            "absorbed {} should track planted nuisance fraction {planted_fraction}",
            fit.variance_absorbed
        );

        // The nuisance directions are stripped from the residual: variance along
        // d_pos and d_freq drops to a small fraction of the input's.
        let resid = fit.residual(x.view(), design.view());
        let pos_before = variance_along(x, &planted.d_pos);
        let pos_after = variance_along(&resid, &planted.d_pos);
        let freq_before = variance_along(x, &planted.d_freq);
        let freq_after = variance_along(&resid, &planted.d_freq);
        assert!(
            pos_after < 0.05 * pos_before,
            "positional variance not removed: {pos_after} vs {pos_before}"
        );
        assert!(
            freq_after < 0.05 * freq_before,
            "frequency variance not removed: {freq_after} vs {freq_before}"
        );

        // Semantic energy is preserved: the residual keeps most of the total.
        let kept = total_centered_energy(&resid) / total_centered_energy(x);
        assert!(
            kept > 1.0 - planted_fraction - 0.06,
            "residual dropped too much energy ({kept} kept, planted nuisance {planted_fraction})"
        );

        // OLS orthogonality: ZᵀR ≈ 0.
        let mut max_ztr = 0.0f64;
        for c in 0..design.ncols() {
            for j in 0..p {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += design[[i, c]] * resid[[i, j]] as f64;
                }
                max_ztr = max_ztr.max(acc.abs() / n as f64);
            }
        }
        assert!(max_ztr < 1.0e-4, "ZᵀR not ~0 (max {max_ztr})");
    }

    #[test]
    fn purely_semantic_input_absorbs_only_overfit_floor() {
        // No positional or frequency structure: X is random semantic content.
        // The atlas can only fit the in-sample M/N overfit floor.
        let config = NuisanceAtlasConfig::default();
        let (n, p) = (2000usize, 24usize);
        let mut s = 0xABCD_1234_u64;
        let mut x = Array2::<f32>::zeros((n, p));
        let mut positions = Vec::with_capacity(n);
        let mut log_freq = Vec::with_capacity(n);
        for i in 0..n {
            positions.push((i % 128) as i64);
            log_freq.push(3.0 * u01(&mut s));
            for j in 0..p {
                x[[i, j]] = gauss(&mut s) as f32;
            }
        }
        let design = build_nuisance_design(n, &positions, &log_freq, &config).expect("design");
        let m = design.ncols();
        let fit = fit_nuisance_atlas(x.view(), design.view(), config.ridge).expect("fit");
        // Absorbed ≈ overfit floor M/N (here ~19/2000 < 1%); allow generous slack.
        let floor = m as f64 / n as f64;
        assert!(
            fit.variance_absorbed < 5.0 * floor + 0.01,
            "purely-semantic absorbed {} should sit near the M/N={floor} floor",
            fit.variance_absorbed
        );
    }

    #[test]
    fn projector_is_idempotent_refit_absorbs_nothing() {
        // Re-fitting the same design on the residual absorbs ≈ 0 further variance.
        let config = NuisanceAtlasConfig::default();
        let (n, p) = (1500usize, 20usize);
        let planted = plant(n, p, &config, 0x0DDBA11_u64);
        let x = &planted.x;
        let design = build_nuisance_design(n, &planted.positions, &planted.log_freq, &config)
            .expect("design");
        let fit = fit_nuisance_atlas(x.view(), design.view(), config.ridge).expect("fit");
        let resid = fit.residual(x.view(), design.view());
        let refit = fit_nuisance_atlas(resid.view(), design.view(), config.ridge).expect("refit");
        assert!(
            refit.variance_absorbed.abs() < 1.0e-6,
            "idempotent projector must absorb ~0 on refit, got {}",
            refit.variance_absorbed
        );
    }

    #[test]
    fn rank_deficient_design_is_flagged_not_panicked() {
        // A duplicated nuisance column makes ZᵀZ singular; plain OLS (ridge 0)
        // must fall back off LLT and flag it rather than panic.
        let n = 300usize;
        let p = 8usize;
        let mut s = 7u64;
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = gauss(&mut s) as f32;
            }
        }
        // Design: intercept + one covariate + its exact duplicate.
        let mut design = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let v = u01(&mut s);
            design[[i, 0]] = 1.0;
            design[[i, 1]] = v;
            design[[i, 2]] = v; // collinear
        }
        let fit = fit_nuisance_atlas(x.view(), design.view(), 0.0).expect("fit");
        assert!(
            fit.design_rank_deficient,
            "a collinear design must be flagged rank-deficient"
        );
        assert!(fit.variance_absorbed.is_finite());
    }
}

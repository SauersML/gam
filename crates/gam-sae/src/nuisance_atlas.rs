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

use gam_linalg::faer_ndarray::fast_ab;
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


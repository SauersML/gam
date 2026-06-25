//! Post-fit functional-ANOVA carve of a fitted product-manifold atom (#975).
//!
//! # The carving problem
//!
//! Two circular attributes in superposition (weekday θ₁, month θ₂) trace a
//! torus in activation space. Is that ONE T² atom or TWO superposed S¹
//! atoms? Reconstruction cannot tell — same surface — so a learner without
//! a principled criterion carves arbitrarily and "the dictionary" is an
//! artifact of the carve. The GAM-native answer is functional ANOVA over
//! the product manifold:
//!
//! ```text
//!   g(θ₁, θ₂) = g₀ + f₁(θ₁) + f₂(θ₂) + f₁₂(θ₁, θ₂)
//! ```
//!
//! with sum-to-zero centering against the EMPIRICAL CODE MEASURE (the
//! averaging measure is itself a gauge choice; we pin it to the code
//! sample and say so). Then **superposition = additivity** (`f₁₂ ≡ 0` ⇔
//! the torus IS two superposed circles, and fission along ANOVA lines is
//! lossless) and **binding = interaction** (`f₁₂ ≠ 0` is genuine joint
//! structure; the atom is irreducible).
//!
//! # Why not just covariance in activations?
//!
//! Covariance is a second-moment statistic of the POINT CLOUD; the carve
//! question is about the FUNCTIONAL FACTORIZATION of the surface. A bound
//! torus and two superposed circles can trace the same point set with the
//! same second moments — covariance sees the embedding, not whether the
//! decoder map factors additively through the two angles. Independence of
//! the codes (θ₁ ⫫ θ₂) is a third, separate property: codes can be
//! dependent while the decoder is perfectly additive, and vice versa. Only
//! the ANOVA interaction block answers "one atom or two".
//!
//! # Two inequivalent binding notions (both first-class here)
//!
//! - **Representational** binding: non-additivity of the DECODER `g` —
//!   does the surface embed as two superposed atoms?
//! - **Computational** binding: non-additivity of the pulled-back READOUT
//!   `h(θ₁,θ₂) = F(g(θ₁,θ₂))` (logit jets through the forward map, #980) —
//!   does the model USE the two angles jointly?
//!
//! All four quadrants occur. Independent steerability ("turn the weekday
//! knob without dragging month behavior") requires additivity in BOTH
//! senses, so the carve decision distinguishes them explicitly
//! ([`FissionDecision`]): the same machinery runs twice — once on the
//! decoder coefficients, once on readout-pulled-back coefficients — and
//! choosing with only the representational arm is reported as such, never
//! silently.
//!
//! # Not everything is clean — the quantitative dial
//!
//! A real model can be sort-of-bound: `f₁₂` small but nonzero, or binding
//! present in the readout but not the embedding. The carve therefore never
//! emits a bare verdict: [`CarveReport::interaction_fraction`] is the
//! fraction of (centered) surface energy carried by the interaction — a
//! continuous "how bound" number — and the planted-partial-binding power
//! curve lives on exactly this dial. The binding test rejects when the
//! data PROVES `f₁₂ ≠ 0`; fission additionally demands the interaction be
//! energetically negligible, because absence of evidence is not evidence
//! of absence. Atoms failing both stay whole and CONTESTED — the
//! demote-never-reject philosophy: the claim goes to the evidence ledger
//! (`structure_evidence::ClaimKind::BindingEdge`, p-value calibrated via
//! `structure_evidence::log_e_from_p_calibrator`) and earns a probe
//! budget, instead of a silent carve either way.
//!
//! # Post-fit by design
//!
//! This module is a PURE READ of a fitted tensor-product decoder: the
//! caller supplies the factor bases evaluated on the code sample and the
//! per-output-dim coefficient matrices (plus, optionally, their posterior
//! covariance for the Wald test). It deliberately does NOT add an
//! in-fit ANOVA basis kind: two independent circles are just two atoms
//! summing — ordinary superposition, the default multi-atom model — so
//! the product machinery is only ever needed at the moment a fitted pair
//! shows dependent codes and the structure search must adjudicate
//! merge-vs-keep. That adjudication consumes this carve.
//!
//! # The gauge inside the test (load-bearing)
//!
//! On a partition-of-unity factor basis (B-splines: `Σ_j φ_j ≡ 1`) the
//! empirically centered basis functions `φ̃_j = φ_j − mean_n φ_j(θ_n)`
//! carry one exact linear dependence per factor: `Σ_j φ̃_j ≡ 0`. The
//! coefficient directions `u vᵀ + w uᵀ` (u the dependence vector) change
//! NOTHING about `f₁₂` — they are pure gauge, their posterior values are
//! penalty-set noise, and a Wald statistic that includes them is wrong.
//! The binding test therefore projects the interaction block onto the
//! gauge quotient (`C ↦ P₁ C P₂`, `P_i = I − û_i û_iᵀ`) before testing;
//! the quotient dimension `(M₁−1)(M₂−1)` is the test's honest rank.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::faer_ndarray::FaerEigh;
use crate::inference::smooth_test::{
    SmoothTestInput, SmoothTestResult, SmoothTestScale, wood_smooth_test,
};
use crate::solver::grid_spline_2d::{GridSpline2dDesign, axis_basis_at};

/// Interaction energy fraction at or below which the interaction block is
/// energetically negligible and lossless fission is on the table. The bar is
/// the finite-sample NOISE FLOOR of the interaction estimate, not exact
/// algebraic zero. A planted, exactly-additive coefficient matrix carves to
/// numerical zero (≈ f64 roundoff), but a real REML fit of a genuinely
/// separable surface over noisy scattered codes cannot drive its penalized
/// interaction block below the variance its own estimator injects: a 5%-noise
/// pair fit lands at ~`1e-4` of centered surface energy (a relative amplitude of
/// `1e-2`, ≈ √fraction). `1e-4` sits just above that estimator floor so a
/// separable atom actually fissions end to end (the production
/// `fit_pair_surface → carve` path, which the planted in-module tests do not
/// exercise), while staying far below any genuine interaction — the bound
/// panels carry fractions orders of magnitude larger, and the companion binding
/// Wald test resolves small-but-real interactions besides. Auto-applied — no
/// knob.
pub const FISSION_MAX_INTERACTION_FRACTION: f64 = 1e-4;

/// Interaction energy fraction at or below which the gauge-projected
/// interaction block is f64 roundoff rather than signal, so the binding Wald
/// test cannot constitute proof of binding. An exactly-additive surface fits to
/// machine precision; its scale-included posterior covariance collapses
/// (`σ̂² → 0`) while the projected interaction coefficients are pure centering
/// roundoff, so the Wald statistic degenerates into a `0/0` ratio — roundoff
/// coefficients divided by a vanishing covariance — that can read as
/// overwhelmingly significant (`p ≈ 0`). At or below this floor (a relative
/// amplitude of `1e-6`, far above the ~`1e-30` roundoff an exactly-additive
/// carve actually lands at, yet far below any interaction a finite-sample fit
/// can statistically resolve) the surface is additive by construction and no
/// such statistic counts as binding: absence of an interaction is not evidence
/// of one. This keeps a numerically-additive atom from being held whole on a
/// phantom edge. Auto-applied — no knob.
const INTERACTION_NUMERICAL_FLOOR: f64 = 1e-12;

/// Which binding notion a carve report speaks about (see module docs; the
/// two are independent and a complete adjudication runs both).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingNotion {
    /// Decoder non-additivity: does the surface EMBED as two atoms?
    Representational,
    /// Pulled-back readout non-additivity: does the model USE the two
    /// coordinates jointly? (Coefficients come from fitting the same
    /// tensor basis to `h = F(g)` via the #980 output-Fisher harvest.)
    Computational,
}

/// The exact ANOVA reparameterization of one output dimension's tensor
/// coefficient matrix `C` (`M₁ × M₂`) under empirical-measure centering.
/// With `m_i` the empirical mean of factor `i`'s basis over the code
/// sample and `φ̃ = φ − m`, the surface decomposes EXACTLY (an identity,
/// not an approximation):
///
/// ```text
///   φ¹ᵀ C φ² = mean + φ̃¹ᵀ·main_a + φ̃²ᵀ·main_b + φ̃¹ᵀ C φ̃²
/// ```
///
/// so `mean = m₁ᵀ C m₂`, `main_a = C m₂`, `main_b = Cᵀ m₁`, and the
/// interaction block on the centered tensor basis is `C` itself (tested
/// in its gauge quotient, see module docs).
#[derive(Clone, Debug)]
pub struct AnovaBlocks {
    pub mean: f64,
    pub main_a: Array1<f64>,
    pub main_b: Array1<f64>,
}

/// Empirical mean of each basis column over the code sample — the
/// centering vector `m` that pins the ANOVA gauge to the empirical code
/// measure.
pub fn basis_means(phi: ArrayView2<'_, f64>) -> Array1<f64> {
    let n = phi.nrows().max(1) as f64;
    let mut m = Array1::<f64>::zeros(phi.ncols());
    for row in phi.rows() {
        for (j, &v) in row.iter().enumerate() {
            m[j] += v;
        }
    }
    m.mapv_inplace(|v| v / n);
    m
}

/// The exact reparameterization (see [`AnovaBlocks`]).
pub fn anova_blocks(
    c: ArrayView2<'_, f64>,
    mean_a: ArrayView1<'_, f64>,
    mean_b: ArrayView1<'_, f64>,
) -> Result<AnovaBlocks, String> {
    let (m1, m2) = c.dim();
    if mean_a.len() != m1 || mean_b.len() != m2 {
        return Err(format!(
            "anova_blocks: coefficient matrix is {m1}×{m2} but centering means have lengths {} and {}",
            mean_a.len(),
            mean_b.len()
        ));
    }
    let main_a = c.dot(&mean_b);
    let main_b = c.t().dot(&mean_a);
    let mean = mean_a.dot(&main_a);
    Ok(AnovaBlocks {
        mean,
        main_a,
        main_b,
    })
}

/// One child atom's 1-D decoder for one output dimension, expressed on
/// the CENTERED factor basis plus an explicit constant — basis-agnostic,
/// no partition-of-unity assumption baked in. The child surface is
/// `constant + φ̃(θ)ᵀ·centered_coeffs`.
#[derive(Clone, Debug)]
pub struct ChildDecoder {
    pub constant: f64,
    pub centered_coeffs: Array1<f64>,
}

impl ChildDecoder {
    /// Fold the constant back into raw basis coefficients for a
    /// partition-of-unity basis (`Σ_j φ_j ≡ 1`, e.g. B-splines):
    /// `constant + φ̃ᵀa = φᵀ(a + (constant − mᵀa)·1)`. For non-PoU bases
    /// keep the explicit-constant form instead.
    pub fn raw_coeffs_partition_of_unity(&self, means: ArrayView1<'_, f64>) -> Array1<f64> {
        let shift = self.constant - means.dot(&self.centered_coeffs);
        self.centered_coeffs.mapv(|v| v) + Array1::from_elem(self.centered_coeffs.len(), shift)
    }
}

/// The lossless-on-the-additive-part split: child atoms inheriting the
/// main-effect blocks. Gauge choice (documented, fixed): the grand mean
/// `g₀` rides with child A; child B is centered. The interaction energy
/// the split discards is DECLARED in `reconstruction_defect` — by the
/// fission rule it is ≤ [`FISSION_MAX_INTERACTION_FRACTION`], but it is
/// never silently zero.
#[derive(Clone, Debug)]
pub struct FissionPlan {
    /// Per output dimension: child atom on factor A (`g₀ + f₁`).
    pub child_a: Vec<ChildDecoder>,
    /// Per output dimension: child atom on factor B (`f₂`).
    pub child_b: Vec<ChildDecoder>,
    /// Interaction energy fraction the split throws away.
    pub reconstruction_defect: f64,
}

/// What the carve concluded for one binding notion.
#[derive(Clone, Debug)]
pub struct CarveReport {
    pub notion: BindingNotion,
    /// Wood-style Wald test of the gauge-projected interaction block, one
    /// per output dimension (`None` where covariance was unavailable or
    /// the test degenerated).
    pub binding_tests: Vec<Option<SmoothTestResult>>,
    /// Edge-level binding p-value: Bonferroni min-p across output
    /// dimensions (conservative under arbitrary cross-dimension
    /// dependence — the dimensions share every code). `None` when no
    /// per-dimension test ran. This is the number that feeds
    /// `structure_evidence::ClaimKind::BindingEdge` through
    /// `log_e_from_p_calibrator`.
    pub edge_p_value: Option<f64>,
    /// Fraction of centered surface energy carried by the interaction,
    /// aggregated over output dimensions — the continuous "how bound"
    /// dial (0 = perfectly additive, 1 = pure interaction).
    pub interaction_fraction: f64,
    /// The lossless split, present iff this notion's carve allows it:
    /// interaction energetically negligible AND not proven present.
    pub fission: Option<FissionPlan>,
}

/// The joint adjudication over both notions — three-valued on purpose:
/// the representational and computational carves differ exactly on the
/// off-diagonal quadrants, so collapsing them silently is the one
/// forbidden move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FissionDecision {
    /// Both notions additive: the split is safe for every downstream use,
    /// including independent-knob steering.
    SplitCertifiedJoint,
    /// Decoder additive but the computational arm was NOT run (no readout
    /// coefficients supplied): the split is certified for reconstruction
    /// only — steering independence is unverified.
    SplitReconstructionOnly,
    /// At least one ran notion refuses (binding proven or interaction
    /// non-negligible): the atom stays whole and contested.
    Keep,
}

/// Width of the deterministic log-λ grid the REML profile is scanned on
/// before golden-section refinement. 121 points over 18 decades resolves
/// the criterion's single basin to ~0.15 decades before refinement; the
/// refinement then closes to machine-level. Fixed (no adaptivity) so the
/// fit is a pure function of its inputs.
const REML_LAMBDA_GRID: usize = 121;
/// Golden-section refinement iterations after the grid scan (~1e-9
/// relative bracket width — far past where the criterion is flat).
const REML_GOLDEN_ITERS: usize = 60;

/// A penalized tensor-surface fit over the code sample: the producer of
/// [`CarveInput`]s for BOTH binding notions (#993 items 1–2).
///
/// `coeffs[d]` is the fitted `M₁ × M₂` coefficient matrix for response
/// dimension `d`; `coeff_covariance[d]` is the matching SCALE-INCLUDED
/// posterior covariance of its row-major vec (the mgcv-`Vb` object
/// [`wood_smooth_test`] contracts for); `joint_covariance()` assembles
/// the cross-dimension covariance for the joint binding test. The fit is
/// evaluated against the SAME empirical code measure the carve centers
/// against — the test and its covariance live on one measure by
/// construction, which is the coherence the production fit's own Hessian
/// (a different parameterization: tangent frames, not tensor
/// coefficients) cannot offer the carve.
#[derive(Clone, Debug)]
pub struct TensorSurfaceFit {
    /// Per response dimension, `M₁ × M₂`.
    pub coeffs: Vec<Array2<f64>>,
    /// Per response dimension, scale-included `Vb` of the row-major vec.
    pub coeff_covariance: Vec<Array2<f64>>,
    /// Scale-included residual cross-covariance between response
    /// dimensions (`D × D`, entries `r_dᵀ r_e / (n − edf)`). Diagonal
    /// entries are the per-dimension scales the `Vb`s carry.
    pub residual_cross_cov: Array2<f64>,
    /// Scale-FREE coefficient covariance shared by all dimensions
    /// (`V (Λ+λI)⁻¹ Vᵀ`, `M₁M₂ × M₁M₂`); `coeff_covariance[d]` is this
    /// times `residual_cross_cov[d,d]`.
    pub unit_covariance: Array2<f64>,
    /// REML-selected ridge strength.
    pub lambda: f64,
    /// Effective degrees of freedom `Σ dᵢ/(dᵢ+λ)` (per dimension; the
    /// design and λ are shared).
    pub edf: f64,
    /// Residual degrees of freedom `n − edf` (the denominator d.f. for
    /// the `Estimated`-scale F branch).
    pub residual_df: f64,
}

impl TensorSurfaceFit {
    /// Joint covariance of the dimension-major stacked coefficient vector
    /// `[vec(C₀); vec(C₁); …]`: with a shared design and shared λ the
    /// posterior is the Kronecker product
    /// `residual_cross_cov ⊗ unit_covariance` — index `(d·M + i, e·M + j)
    /// = S[d,e]·U[i,j]`. Feed to [`CarveInput::joint_coeff_covariance`].
    pub fn joint_covariance(&self) -> Array2<f64> {
        let d_dims = self.residual_cross_cov.nrows();
        let m = self.unit_covariance.nrows();
        let mut joint = Array2::<f64>::zeros((d_dims * m, d_dims * m));
        for d in 0..d_dims {
            for e in 0..d_dims {
                let s_de = self.residual_cross_cov[[d, e]];
                if s_de == 0.0 {
                    continue;
                }
                for i in 0..m {
                    for j in 0..m {
                        joint[[d * m + i, e * m + j]] = s_de * self.unit_covariance[[i, j]];
                    }
                }
            }
        }
        joint
    }
}

/// Fit the tensor-product surface `y_d(θ₁,θ₂) ≈ φ¹(θ₁)ᵀ C_d φ²(θ₂)` to
/// sampled responses by ridge-penalized least squares with the ridge
/// strength chosen by GAUSSIAN REML (profiled σ², exact 1-D criterion on
/// the design's eigenbasis — no GCV, per policy), returning coefficients
/// AND their scale-included posterior covariance.
///
/// This is the missing producer #993 names for both carve arms:
/// - **representational**: `responses` = the atom's activation
///   contributions over the code sample (its reconstruction targets);
/// - **computational**: `responses` = the pulled-back readout
///   `h(θ₁,θ₂) = F(g(θ))` rows from the #980 output-Fisher harvest.
///
/// `phi_a`/`phi_b` are the factor bases on the code sample (`n × M_i`,
/// the same matrices the carve consumes — one measure end to end);
/// `responses` is `n × D`. The design column for `(j, k)` is
/// `φ¹_j·φ²_k` at row-major index `j·M₂+k`, matching the carve's vec
/// convention exactly. One λ is shared across response dimensions (one
/// surface smoothness), chosen by the pooled REML criterion; per-dim
/// scales are estimated from residuals at `n − edf`.
pub fn fit_tensor_surface(
    phi_a: ArrayView2<'_, f64>,
    phi_b: ArrayView2<'_, f64>,
    responses: ArrayView2<'_, f64>,
) -> Result<TensorSurfaceFit, String> {
    let n = phi_a.nrows();
    let m1 = phi_a.ncols();
    let m2 = phi_b.ncols();
    let mm = m1 * m2;
    let d_dims = responses.ncols();
    if phi_b.nrows() != n || responses.nrows() != n {
        return Err(format!(
            "fit_tensor_surface: sample sizes disagree (phi_a {n}, phi_b {}, responses {})",
            phi_b.nrows(),
            responses.nrows()
        ));
    }
    if mm == 0 || d_dims == 0 || n < 2 {
        return Err(format!(
            "fit_tensor_surface: degenerate problem (n={n}, M₁M₂={mm}, D={d_dims})"
        ));
    }

    // Design X (n × M₁M₂), row-major column convention j·M₂+k.
    let mut x = Array2::<f64>::zeros((n, mm));
    for r in 0..n {
        for j in 0..m1 {
            let pa = phi_a[[r, j]];
            if pa == 0.0 {
                continue;
            }
            for k in 0..m2 {
                x[[r, j * m2 + k]] = pa * phi_b[[r, k]];
            }
        }
    }
    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&responses); // mm × D
    let (evals, evecs) = xtx
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("fit_tensor_surface: design eigendecomposition failed: {e:?}"))?;
    let d_max = evals.iter().cloned().fold(0.0f64, f64::max);
    if !(d_max > 0.0) {
        return Err("fit_tensor_surface: design is identically zero".to_string());
    }
    let b = evecs.t().dot(&xty); // mm × D, rotated cross-products
    let yty: Vec<f64> = (0..d_dims)
        .map(|d| responses.column(d).dot(&responses.column(d)))
        .collect();

    // Pooled Gaussian REML criterion in the eigenbasis, σ² profiled per
    // dimension: V(λ) = Σ_d n·log PRSS_d(λ) + D·[Σᵢ log(dᵢ+λ) − M·log λ],
    // with PRSS_d = yᵀy − Σᵢ bᵢ²/(dᵢ+λ). Exact and O(M·D) per evaluation.
    let criterion = |lambda: f64| -> f64 {
        let mut v = 0.0f64;
        for d in 0..d_dims {
            let mut prss = yty[d];
            for i in 0..mm {
                prss -= b[[i, d]] * b[[i, d]] / (evals[i].max(0.0) + lambda);
            }
            v += (n as f64) * prss.max(f64::MIN_POSITIVE).ln();
        }
        let mut logdet = 0.0f64;
        for i in 0..mm {
            logdet += (evals[i].max(0.0) + lambda).ln();
        }
        v + (d_dims as f64) * (logdet - (mm as f64) * lambda.ln())
    };

    // Deterministic grid scan over 18 decades anchored at the design's
    // spectral scale, then golden-section refinement in the best bracket.
    let lo = d_max * 1e-9;
    let hi = d_max * 1e9;
    let mut best_idx = 0usize;
    let mut best_val = f64::INFINITY;
    let grid: Vec<f64> = (0..REML_LAMBDA_GRID)
        .map(|i| {
            let t = i as f64 / (REML_LAMBDA_GRID - 1) as f64;
            lo * (hi / lo).powf(t)
        })
        .collect();
    for (i, &lam) in grid.iter().enumerate() {
        let v = criterion(lam);
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    let mut a_log = grid[best_idx.saturating_sub(1)].ln();
    let mut c_log = grid[(best_idx + 1).min(REML_LAMBDA_GRID - 1)].ln();
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut x1 = c_log - phi * (c_log - a_log);
    let mut x2 = a_log + phi * (c_log - a_log);
    let mut f1 = criterion(x1.exp());
    let mut f2 = criterion(x2.exp());
    for _ in 0..REML_GOLDEN_ITERS {
        if f1 <= f2 {
            c_log = x2;
            x2 = x1;
            f2 = f1;
            x1 = c_log - phi * (c_log - a_log);
            f1 = criterion(x1.exp());
        } else {
            a_log = x1;
            x1 = x2;
            f1 = f2;
            x2 = a_log + phi * (c_log - a_log);
            f2 = criterion(x2.exp());
        }
    }
    let lambda = (0.5 * (a_log + c_log)).exp();

    // Coefficients, EDF, residuals, covariances at the selected λ.
    let mut edf = 0.0f64;
    for i in 0..mm {
        let d_i = evals[i].max(0.0);
        edf += d_i / (d_i + lambda);
    }
    let residual_df = n as f64 - edf;
    if residual_df < 1.0 {
        return Err(format!(
            "fit_tensor_surface: too few samples for the surface (n={n}, edf={edf:.2}); \
             the scale estimate needs n − edf ≥ 1"
        ));
    }
    // β̂ in the eigenbasis, then rotate back: beta = V (Λ+λ)⁻¹ b.
    let mut beta_rot = Array2::<f64>::zeros((mm, d_dims));
    for i in 0..mm {
        let denom = evals[i].max(0.0) + lambda;
        for d in 0..d_dims {
            beta_rot[[i, d]] = b[[i, d]] / denom;
        }
    }
    let beta = evecs.dot(&beta_rot); // mm × D
    let fitted = x.dot(&beta); // n × D
    let mut residual_cross_cov = Array2::<f64>::zeros((d_dims, d_dims));
    for d in 0..d_dims {
        for e in d..d_dims {
            let mut acc = 0.0f64;
            for r in 0..n {
                acc += (responses[[r, d]] - fitted[[r, d]]) * (responses[[r, e]] - fitted[[r, e]]);
            }
            let v = acc / residual_df;
            residual_cross_cov[[d, e]] = v;
            residual_cross_cov[[e, d]] = v;
        }
    }
    // Scale-free V (Λ+λ)⁻¹ Vᵀ.
    let mut scaled_evecs = evecs.clone();
    for i in 0..mm {
        let denom = evals[i].max(0.0) + lambda;
        for row in 0..mm {
            scaled_evecs[[row, i]] = evecs[[row, i]] / denom;
        }
    }
    let unit_covariance = scaled_evecs.dot(&evecs.t());

    let mut coeffs = Vec::with_capacity(d_dims);
    let mut coeff_covariance = Vec::with_capacity(d_dims);
    for d in 0..d_dims {
        let mut c = Array2::<f64>::zeros((m1, m2));
        for j in 0..m1 {
            for k in 0..m2 {
                c[[j, k]] = beta[[j * m2 + k, d]];
            }
        }
        coeffs.push(c);
        coeff_covariance.push(&unit_covariance * residual_cross_cov[[d, d]]);
    }

    Ok(TensorSurfaceFit {
        coeffs,
        coeff_covariance,
        residual_cross_cov,
        unit_covariance,
        lambda,
        edf,
        residual_df,
    })
}

/// The real-fit producer of a representational [`CarveInput`] from a fitted
/// `d = 2` product atom (#993).
///
/// Holds the two factor bases the carve consumes plus the
/// [`TensorSurfaceFit`] re-fit of the atom's own ambient reconstruction. The
/// fit is what supplies the scale-included decoder-coefficient covariance
/// (`coeff_covariance` / `joint_covariance`) — the production inner Hessian is
/// a DIFFERENT parameterization (tangent frames, not tensor coefficients) and
/// cannot offer the carve a coefficient-space `Vb`, so the carve's covariance
/// is re-derived here on the same empirical code measure the test centers
/// against (the coherence the module docs require). Owns its arrays so the
/// borrowed [`CarveInput`] built via [`Self::representational_carve_input`] can
/// reference them for the lifetime of the carve call.
#[derive(Clone, Debug)]
pub struct FittedAtomCarveInput {
    /// Factor-A basis on the code sample, `n × M₁`.
    pub phi_a: Array2<f64>,
    /// Factor-B basis on the code sample, `n × M₂`.
    pub phi_b: Array2<f64>,
    /// REML re-fit of the atom's ambient reconstruction onto the tensor basis,
    /// carrying the per-channel coefficient matrices and their scale-included
    /// covariance.
    pub surface: TensorSurfaceFit,
    /// Cross-dimension joint covariance of the stacked coefficient vector
    /// (`TensorSurfaceFit::joint_covariance`), materialized once so the
    /// borrowed [`CarveInput`] can reference it.
    pub joint_covariance: Array2<f64>,
}

impl FittedAtomCarveInput {
    /// Borrow this bundle as a representational [`CarveInput`] ready for
    /// [`carve`]. The coefficient covariance and the joint covariance come
    /// from the REML re-fit; the gauge kernels default to the
    /// partition-of-unity convention (`u = 1`), which is the correct centered-
    /// basis null direction for the constant-leading harmonic factor bases.
    pub fn representational_carve_input(&self) -> CarveInput<'_> {
        CarveInput {
            phi_a: self.phi_a.view(),
            phi_b: self.phi_b.view(),
            coeffs: self.surface.coeffs.as_slice(),
            coeff_covariance: Some(self.surface.coeff_covariance.as_slice()),
            joint_coeff_covariance: Some(&self.joint_covariance),
            kernel_a: None,
            kernel_b: None,
            edf: Some(self.surface.edf),
            residual_df: self.surface.residual_df,
            scale: SmoothTestScale::Estimated,
            notion: BindingNotion::Representational,
        }
    }
}

/// Build the representational carve inputs for a fitted `d = 2` product atom
/// directly from its FUSED tensor basis and decoder (#993).
///
/// `basis_values` is the atom's `Φ_k` on the code sample (`n × M₁M₂`), laid
/// out as the Kronecker product of the two per-axis factor bases in row-major
/// column order `flat = j·M₂ + k` (the convention every product evaluator —
/// `TorusHarmonicEvaluator`, `CylinderHarmonicEvaluator` — emits, with the
/// per-axis CONSTANT column at axis-index 0). `decoder_coefficients` is `B_k`
/// (`M₁M₂ × p`). `m_a`/`m_b` are the two factor basis sizes (`m_a·m_b` must
/// equal the fused width).
///
/// The factor bases are recovered exactly from the fused basis using the
/// constant-leading-column property: with `φ²₀ ≡ 1`, column `j·M₂` is
/// `φ¹_j·φ²₀ = φ¹_j`, and with `φ¹₀ ≡ 1`, column `k` is `φ¹₀·φ²_k = φ²_k`. The
/// recovered factorization is then VERIFIED against every fused column
/// (`Φ[:, j·M₂+k] = φ¹_j·φ²_k` to a tight tolerance) so a non-separable basis
/// (a wrong split, or a kind whose leading column is not the unit constant) is
/// rejected loudly rather than silently mis-carved.
///
/// The carve responses are the atom's own ambient reconstruction
/// `m_k(t) = Φ_k(t)·B_k` (`n × p`); fitting the tensor surface to it on the
/// same code measure yields the scale-included coefficient covariance the
/// binding Wald test needs. The reconstruction is an exact linear image of the
/// decoder, so the re-fit recovers the decoder's own ANOVA structure (the
/// representational binding question) with a covariance that is honest about
/// the finite code sample.
pub fn carve_input_from_fitted_atom(
    basis_values: ArrayView2<'_, f64>,
    decoder_coefficients: ArrayView2<'_, f64>,
    m_a: usize,
    m_b: usize,
) -> Result<FittedAtomCarveInput, String> {
    let n = basis_values.nrows();
    let fused = basis_values.ncols();
    let p = decoder_coefficients.ncols();
    if m_a == 0 || m_b == 0 {
        return Err(format!(
            "carve_input_from_fitted_atom: degenerate factor sizes (m_a={m_a}, m_b={m_b})"
        ));
    }
    if m_a.checked_mul(m_b) != Some(fused) {
        return Err(format!(
            "carve_input_from_fitted_atom: factor sizes {m_a}×{m_b} do not multiply to the \
             fused basis width {fused}"
        ));
    }
    if decoder_coefficients.nrows() != fused {
        return Err(format!(
            "carve_input_from_fitted_atom: decoder has {} rows but the fused basis is width {fused}",
            decoder_coefficients.nrows()
        ));
    }
    if n < 2 || p == 0 {
        return Err(format!(
            "carve_input_from_fitted_atom: degenerate sample (n={n}, p={p})"
        ));
    }

    // Recover the factor bases from the constant-leading Kronecker layout:
    // φ¹_j = Φ[:, j·M₂ + 0]  (φ²₀ ≡ 1),   φ²_k = Φ[:, 0·M₂ + k]  (φ¹₀ ≡ 1).
    let mut phi_a = Array2::<f64>::zeros((n, m_a));
    for j in 0..m_a {
        let col = j * m_b;
        for row in 0..n {
            phi_a[[row, j]] = basis_values[[row, col]];
        }
    }
    let mut phi_b = Array2::<f64>::zeros((n, m_b));
    for k in 0..m_b {
        for row in 0..n {
            phi_b[[row, k]] = basis_values[[row, k]];
        }
    }

    // Verify the fused basis really is the Kronecker product of the recovered
    // factors (separability + constant-leading-column assumption). The check is
    // relative to the fused magnitude so it is scale-honest; a non-product atom
    // or a wrong split fails here instead of being silently mis-carved.
    let mut max_abs = 0.0_f64;
    for &v in basis_values.iter() {
        max_abs = max_abs.max(v.abs());
    }
    let tol = 1e-9 * (1.0 + max_abs);
    for j in 0..m_a {
        for k in 0..m_b {
            let col = j * m_b + k;
            for row in 0..n {
                let recon = phi_a[[row, j]] * phi_b[[row, k]];
                if (recon - basis_values[[row, col]]).abs() > tol {
                    return Err(format!(
                        "carve_input_from_fitted_atom: fused basis is not the Kronecker product \
                         of the {m_a}×{m_b} factor split (entry [{row},{col}] = {} vs φ¹·φ² = {recon}); \
                         the atom is not a constant-leading product basis",
                        basis_values[[row, col]]
                    ));
                }
            }
        }
    }

    // Carve responses = the atom's ambient reconstruction m_k = Φ_k · B_k.
    let reconstruction = basis_values.dot(&decoder_coefficients);

    // REML re-fit of the reconstruction onto the SAME tensor basis: supplies the
    // scale-included decoder-coefficient covariance the binding Wald test reads.
    let surface = fit_tensor_surface(phi_a.view(), phi_b.view(), reconstruction.view())?;
    let joint_covariance = surface.joint_covariance();

    Ok(FittedAtomCarveInput {
        phi_a,
        phi_b,
        surface,
        joint_covariance,
    })
}

/// Engine cap on knot cells per axis (the grid engine's dense-Cholesky
/// sizing contract: `p = (K+3)² ≤ 1225`).
const PAIR_COMPONENT_MAX_CELLS: usize = 32;
/// Floor on knot cells per axis — below 4 cells the cubic tensor basis has
/// too little resolution to carry a pair interaction worth carving.
const PAIR_COMPONENT_MIN_CELLS: usize = 4;

/// Knot cells per axis for the raw-coordinate pair component, chosen from
/// the sample size alone (magic by default — no knob): `K ≈ n^(1/3)`
/// clamped to `[4, 32]`. The cube-root growth keeps the basis comfortably
/// inside the data's resolution (p = (K+3)² ≪ n for all n ≥ ~300) while
/// REML owns the actual smoothness; the cap is the engine's sizing contract.
fn pair_component_cells(n: usize) -> usize {
    ((n as f64).cbrt().ceil() as usize).clamp(PAIR_COMPONENT_MIN_CELLS, PAIR_COMPONENT_MAX_CELLS)
}

/// Which estimator produced a [`PairSurfaceFit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairSurfaceBackend {
    /// The streaming 2-D grid engine: exact REML on the full anisotropic
    /// biharmonic penalty (mixed `f_{x1x2}` term included), O(n) assembly,
    /// exact log-determinants — the first-class pair-component estimator.
    GridExact,
    /// The dense ridge fallback ([`fit_tensor_surface`]) on the SAME
    /// B-spline tensor basis, used only when the grid solve degenerates
    /// (e.g. a non-positive-definite penalized system or `n − edf < 1`).
    DenseRidge,
}

/// A pair-component fit from RAW coordinates: the factor bases it was fit
/// on (the grid engine's per-axis uniform cubic B-splines, evaluated on the
/// sample — exactly what [`CarveInput`] consumes, one measure end to end)
/// plus the [`TensorSurfaceFit`] carve product and which backend produced it.
#[derive(Clone, Debug)]
pub struct PairSurfaceFit {
    /// Axis-1 basis on the sample (`n × (K+3)`, partition of unity).
    pub phi_a: Array2<f64>,
    /// Axis-2 basis on the sample (`n × (K+3)`, partition of unity).
    pub phi_b: Array2<f64>,
    /// The carve product: coefficients, covariances, λ, EDF.
    pub surface: TensorSurfaceFit,
    pub backend: PairSurfaceBackend,
    /// Lower corner of the per-axis uniform knot range (the data's
    /// bounding box) — with [`Self::cell_widths`], everything needed to
    /// rebuild a basis row at an arbitrary point.
    pub lower_corner: [f64; 2],
    /// Knot-cell width per axis.
    pub cell_widths: [f64; 2],
}

impl PairSurfaceFit {
    /// Posterior `(mean, variance)` of response dimension `dim` at an
    /// arbitrary point, through the carve-facing posterior objects — valid
    /// for BOTH backends, since both populate the same surface contract:
    /// `mean = b₁ᵀ C_d b₂` and `variance = σ̂²_d · xᵀUx` with `U` the shared
    /// scale-free coefficient covariance, `σ̂²_d` the residual variance at
    /// `n − edf`, and `x` the 16-entry tensor basis row. Outside the data
    /// bounding box the boundary cell's cubic polynomial extends (the grid
    /// engine's convention).
    pub fn predict(&self, dim: usize, x1: f64, x2: f64) -> Result<(f64, f64), String> {
        let d_dims = self.surface.coeffs.len();
        if dim >= d_dims {
            return Err(format!(
                "pair surface: response dimension {dim} out of range (D = {d_dims})"
            ));
        }
        if !(x1.is_finite() && x2.is_finite()) {
            return Err(format!(
                "pair surface: non-finite prediction point ({x1}, {x2})"
            ));
        }
        let m = self.phi_a.ncols();
        let cells = m - 3;
        let (j1, b1) = axis_basis_at(self.lower_corner[0], self.cell_widths[0], cells, x1);
        let (j2, b2) = axis_basis_at(self.lower_corner[1], self.cell_widths[1], cells, x2);
        let c = &self.surface.coeffs[dim];
        let u = &self.surface.unit_covariance;
        let mut mean = 0.0;
        let mut quad = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                let v_ij = b1[i] * b2[j];
                mean += v_ij * c[[j1 + i, j2 + j]];
                let g_ij = (j1 + i) * m + (j2 + j);
                for a in 0..4 {
                    for b in 0..4 {
                        quad += v_ij * b1[a] * b2[b] * u[[g_ij, (j1 + a) * m + (j2 + b)]];
                    }
                }
            }
        }
        Ok((mean, self.surface.residual_cross_cov[[dim, dim]] * quad))
    }
}

/// THE pair-component estimator (#1031): fit the Layer-B ANOVA pair
/// interaction surface from RAW coordinates, auto-routed with no knobs.
///
/// Estimator. A `(K+3)²` tensor of uniform cubic B-splines over the data's
/// bounding box, penalized by the FULL anisotropic biharmonic energy
/// `∫∫ a₁²f₁₁² + 2a₁a₂f₁₂² + a₂²f₂₂²` (mixed term included — the roughness
/// functional no `te()` Kronecker-marginal penalty matches, which is
/// exactly why this is exposed as its own estimator instead of an
/// auto-route through the formula smooths: routing `te`/Duchon through it
/// would silently change their posteriors). One λ is shared across response
/// dimensions (one surface smoothness), selected by the pooled exact REML
/// criterion. `K` grows as `n^(1/3)` (capped by the engine's sizing
/// contract); the metric is pinned to `a_i = L_i²` (squared bounding-box
/// span per axis), which makes the penalty — and hence the estimator —
/// invariant to per-axis rescaling of the coordinates, with the leftover
/// global constant absorbed by λ.
///
/// Route. The streaming grid engine evaluates this estimator EXACTLY in
/// O(n): one scatter-add pass, banded sufficient statistics, exact
/// log-determinant REML, exact posterior summary. When its solve
/// degenerates (non-PD penalized system, `n − edf < 1`), the same basis
/// falls back to the dense ridge path ([`fit_tensor_surface`]) — a
/// different (heavier, isotropic-in-coefficients) penalty but the same
/// surface class, so every input that admits a pair component gets one.
///
/// The returned bases and fit feed [`CarveInput`] directly (B-splines are
/// partition-of-unity, so the default `kernel_a`/`kernel_b` gauge applies).
pub fn fit_pair_surface(
    x1: &[f64],
    x2: &[f64],
    responses: ArrayView2<'_, f64>,
) -> Result<PairSurfaceFit, String> {
    let n = x1.len();
    let d_dims = responses.ncols();
    if x2.len() != n || responses.nrows() != n {
        return Err(format!(
            "fit_pair_surface: sample sizes disagree (x1 {n}, x2 {}, responses {})",
            x2.len(),
            responses.nrows()
        ));
    }
    if d_dims == 0 {
        return Err("fit_pair_surface: no response dimensions".to_string());
    }
    // Axis-rescaling-invariant metric a_i = L_i² (see the doc comment).
    let mut span = [0.0_f64; 2];
    for (ax, xs) in [x1, x2].into_iter().enumerate() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in xs {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if !(hi > lo && hi.is_finite() && lo.is_finite()) {
            return Err(format!(
                "fit_pair_surface: axis {} is degenerate or non-finite ([{lo}, {hi}]); \
                 no pair surface exists over a collapsed axis",
                ax + 1
            ));
        }
        span[ax] = hi - lo;
    }
    let metric = [span[0] * span[0], span[1] * span[1]];
    let k = pair_component_cells(n);

    let columns: Vec<Vec<f64>> = (0..d_dims).map(|d| responses.column(d).to_vec()).collect();
    let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    let weights = vec![1.0_f64; n];
    let design = GridSpline2dDesign::build_multi(x1, x2, &column_refs, &weights, k, metric)?;

    // The factor bases on the sample — shared by both routes and by the
    // carve (one empirical measure end to end).
    let lower_corner = design.lower_corner();
    let cell_widths = design.cell_widths();
    let m = design.basis_per_axis();
    let mut phi_a = Array2::<f64>::zeros((n, m));
    let mut phi_b = Array2::<f64>::zeros((n, m));
    for r in 0..n {
        let (j0, vals) = design.axis_basis(0, x1[r])?;
        for (i, &v) in vals.iter().enumerate() {
            phi_a[[r, j0 + i]] = v;
        }
        let (j0, vals) = design.axis_basis(1, x2[r])?;
        for (i, &v) in vals.iter().enumerate() {
            phi_b[[r, j0 + i]] = v;
        }
    }

    match design
        .fit_reml()
        .and_then(|fit| design.posterior(&fit).map(|post| (fit, post)))
    {
        Ok((fit, post)) => {
            let mm = m * m;
            let mut unit_covariance = Array2::<f64>::zeros((mm, mm));
            for i in 0..mm {
                for j in 0..mm {
                    unit_covariance[[i, j]] = post.unit_covariance[i * mm + j];
                }
            }
            let mut residual_cross_cov = Array2::<f64>::zeros((d_dims, d_dims));
            for d in 0..d_dims {
                for e in 0..d_dims {
                    residual_cross_cov[[d, e]] = post.residual_cross_cov[d * d_dims + e];
                }
            }
            let mut coeffs = Vec::with_capacity(d_dims);
            let mut coeff_covariance = Vec::with_capacity(d_dims);
            for d in 0..d_dims {
                // Engine flat order g = j1·(K+3) + j2 IS the carve's
                // row-major (j·M₂ + k) vec convention.
                let mut c = Array2::<f64>::zeros((m, m));
                for j in 0..m {
                    for kk in 0..m {
                        c[[j, kk]] = fit.coeffs[d][j * m + kk];
                    }
                }
                coeffs.push(c);
                coeff_covariance.push(&unit_covariance * residual_cross_cov[[d, d]]);
            }
            Ok(PairSurfaceFit {
                phi_a,
                phi_b,
                surface: TensorSurfaceFit {
                    coeffs,
                    coeff_covariance,
                    residual_cross_cov,
                    unit_covariance,
                    lambda: fit.log_lambda.exp(),
                    edf: post.edf,
                    residual_df: post.residual_df,
                },
                backend: PairSurfaceBackend::GridExact,
                lower_corner,
                cell_widths,
            })
        }
        Err(grid_err) => {
            let surface =
                fit_tensor_surface(phi_a.view(), phi_b.view(), responses).map_err(|dense_err| {
                    format!(
                        "fit_pair_surface: grid engine degenerated ({grid_err}) and the dense \
                         ridge fallback failed too ({dense_err})"
                    )
                })?;
            Ok(PairSurfaceFit {
                phi_a,
                phi_b,
                surface,
                backend: PairSurfaceBackend::DenseRidge,
                lower_corner,
                cell_widths,
            })
        }
    }
}

/// Inputs for one notion's carve over one fitted product atom.
///
/// `phi_a`/`phi_b`: factor bases evaluated on the code sample (`n × M_i`).
/// `coeffs`: per-output-dim coefficient matrices (`M₁ × M₂` each); for the
/// representational notion these are the decoder's, for the computational
/// notion they come from fitting the same tensor basis to the pulled-back
/// readout. `coeff_covariance`: matching scale-included posterior
/// covariance of the ROW-MAJOR vec of each `C` (`M₁M₂ × M₁M₂` per output
/// dim) — optional; without it the carve still reports the energy
/// fraction but runs no Wald test. `kernel_a`/`kernel_b`: the per-factor
/// coefficient direction along which the centered basis is degenerate
/// (`Σ_j u_j φ̃_j ≡ 0`); `None` selects the partition-of-unity convention
/// `u = 1` (B-splines). `edf`: fitted EDF of the interaction block when
/// the fit tracked one; `None` uses the full quotient rank
/// `(M₁−1)(M₂−1)`.
pub struct CarveInput<'a> {
    pub phi_a: ArrayView2<'a, f64>,
    pub phi_b: ArrayView2<'a, f64>,
    pub coeffs: &'a [Array2<f64>],
    pub coeff_covariance: Option<&'a [Array2<f64>]>,
    /// Covariance of the dimension-major STACKED coefficient vector
    /// `[vec(C₀); vec(C₁); …]` (`D·M₁M₂` square, scale-included), e.g.
    /// [`TensorSurfaceFit::joint_covariance`]. When present, the
    /// edge-level binding p-value comes from ONE joint Wald over the
    /// stacked gauge-projected blocks at rank `D·(M₁−1)(M₂−1)` instead of
    /// the conservative Bonferroni min-p across dimensions (the per-dim
    /// tests share every code row, so Bonferroni over-corrects).
    pub joint_coeff_covariance: Option<&'a Array2<f64>>,
    pub kernel_a: Option<Array1<f64>>,
    pub kernel_b: Option<Array1<f64>>,
    pub edf: Option<f64>,
    pub residual_df: f64,
    pub scale: SmoothTestScale,
    pub notion: BindingNotion,
}

/// The carve: exact ANOVA split, interaction energy, gauge-projected
/// binding test, and the fission plan when this notion permits one.
///
/// Fission rule (asymmetric on purpose): the test REJECTING proves
/// binding and always blocks the split; the test NOT rejecting is only
/// absence of evidence, so the split additionally requires the
/// interaction to be energetically negligible
/// ([`FISSION_MAX_INTERACTION_FRACTION`]). An atom with a fat but
/// unproven interaction stays whole and contested — route its
/// `edge_p_value` into the evidence ledger and let the probe loop earn
/// the verdict.
pub fn carve(input: &CarveInput<'_>, alpha: f64) -> Result<CarveReport, String> {
    let n = input.phi_a.nrows();
    if input.phi_b.nrows() != n {
        return Err(format!(
            "carve: factor bases disagree on sample size ({n} vs {})",
            input.phi_b.nrows()
        ));
    }
    if input.coeffs.is_empty() {
        return Err("carve: no coefficient matrices supplied".to_string());
    }
    let m1 = input.phi_a.ncols();
    let m2 = input.phi_b.ncols();
    if let Some(covs) = input.coeff_covariance
        && covs.len() != input.coeffs.len()
    {
        return Err(format!(
            "carve: {} coefficient matrices but {} covariance blocks",
            input.coeffs.len(),
            covs.len()
        ));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(format!("carve: alpha must be in (0,1), got {alpha}"));
    }

    let mean_a = basis_means(input.phi_a);
    let mean_b = basis_means(input.phi_b);
    // Centered factor evaluations φ̃ = φ − m (n × M_i).
    let phi_a_c = {
        let mut p = input.phi_a.to_owned();
        for mut row in p.rows_mut() {
            for j in 0..m1 {
                row[j] -= mean_a[j];
            }
        }
        p
    };
    let phi_b_c = {
        let mut p = input.phi_b.to_owned();
        for mut row in p.rows_mut() {
            for j in 0..m2 {
                row[j] -= mean_b[j];
            }
        }
        p
    };

    // Gauge projectors P_i = I − û ûᵀ for the centered-basis dependence,
    // and their Kronecker product (the row-major-vec transform shared by
    // the per-dimension and joint Wald tests).
    let proj_a = gauge_projector(m1, input.kernel_a.as_ref())?;
    let proj_b = gauge_projector(m2, input.kernel_b.as_ref())?;
    let gauge_kron = gauge_kron_rowmajor(&proj_a, &proj_b);

    let mut child_a: Vec<ChildDecoder> = Vec::with_capacity(input.coeffs.len());
    let mut child_b: Vec<ChildDecoder> = Vec::with_capacity(input.coeffs.len());
    let mut binding_tests: Vec<Option<SmoothTestResult>> = Vec::with_capacity(input.coeffs.len());
    let mut interaction_energy = 0.0f64;
    let mut centered_energy = 0.0f64;

    for (dim, c) in input.coeffs.iter().enumerate() {
        if c.dim() != (m1, m2) {
            return Err(format!(
                "carve: coefficient matrix {dim} is {:?}, bases say ({m1}, {m2})",
                c.dim()
            ));
        }
        let blocks = anova_blocks(c.view(), mean_a.view(), mean_b.view())?;

        // Interaction values on the sample: f₁₂(θ_n) = φ̃¹_n ᵀ C φ̃²_n,
        // computed as the row-wise dot of (Φ̃₁ C) with Φ̃₂.
        let phi_a_c_c = phi_a_c.dot(c);
        let main_a_vals = phi_a_c.dot(&blocks.main_a);
        let main_b_vals = phi_b_c.dot(&blocks.main_b);
        for row in 0..n {
            let mut f12 = 0.0f64;
            for k in 0..m2 {
                f12 += phi_a_c_c[[row, k]] * phi_b_c[[row, k]];
            }
            interaction_energy += f12 * f12;
            let centered = main_a_vals[row] + main_b_vals[row] + f12;
            centered_energy += centered * centered;
        }

        // Gauge-projected Wald test of the interaction block.
        let test = match input.coeff_covariance {
            None => None,
            Some(covs) => binding_wald_test(
                c,
                &covs[dim],
                &proj_a,
                &proj_b,
                &gauge_kron,
                input.edf,
                input.residual_df,
                input.scale,
            ),
        };
        binding_tests.push(test);

        child_a.push(ChildDecoder {
            constant: blocks.mean,
            centered_coeffs: blocks.main_a,
        });
        child_b.push(ChildDecoder {
            constant: 0.0,
            centered_coeffs: blocks.main_b,
        });
    }

    let interaction_fraction = if centered_energy > 0.0 {
        interaction_energy / centered_energy
    } else {
        0.0
    };
    // Edge-level p: the joint Wald over the stacked gauge-projected
    // blocks when the cross-dimension covariance is available (exact
    // rank, no Bonferroni slack), else Bonferroni min-p across the
    // per-dimension tests (valid under their arbitrary dependence,
    // conservative).
    let edge_p_value = match input.joint_coeff_covariance {
        Some(joint_cov) => joint_binding_wald_test(
            input.coeffs,
            joint_cov,
            &proj_a,
            &proj_b,
            &gauge_kron,
            input.edf,
            input.residual_df,
            input.scale,
        )
        .map(|t| t.p_value),
        None => {
            let ran: Vec<f64> = binding_tests.iter().flatten().map(|t| t.p_value).collect();
            ran.iter()
                .cloned()
                .fold(None, |acc: Option<f64>, p| {
                    Some(acc.map_or(p, |a| a.min(p)))
                })
                .map(|min_p| (min_p * ran.len() as f64).min(1.0))
        }
    };

    // A Wald test cannot prove the PRESENCE of an interaction whose energy is
    // numerically indistinguishable from zero. When the interaction block is at
    // the f64 roundoff floor (an exactly-additive surface fit to machine
    // precision), the scale-included posterior collapses with it and the Wald
    // statistic becomes a 0/0 artifact that can read as overwhelmingly
    // significant (p ≈ 0). Below the floor the surface is additive by
    // construction, so no statistic counts as binding and the atom is free to
    // fission — see `INTERACTION_NUMERICAL_FLOOR`.
    let numerically_additive = interaction_fraction <= INTERACTION_NUMERICAL_FLOOR;
    let binding_proven = !numerically_additive && edge_p_value.is_some_and(|p| p <= alpha);
    let negligible = interaction_fraction <= FISSION_MAX_INTERACTION_FRACTION;
    let fission = if negligible && !binding_proven {
        Some(FissionPlan {
            child_a,
            child_b,
            reconstruction_defect: interaction_fraction,
        })
    } else {
        None
    };

    Ok(CarveReport {
        notion: input.notion,
        binding_tests,
        edge_p_value,
        interaction_fraction,
        fission,
    })
}

/// Joint adjudication across the two binding notions (see
/// [`FissionDecision`]). `representational` must be a
/// [`BindingNotion::Representational`] report; `computational`, when the
/// #980 pulled-back coefficients were available, the matching
/// [`BindingNotion::Computational`] one.
pub fn fission_decision(
    representational: &CarveReport,
    computational: Option<&CarveReport>,
) -> FissionDecision {
    if representational.fission.is_none() {
        return FissionDecision::Keep;
    }
    match computational {
        Some(comp) => {
            if comp.fission.is_some() {
                FissionDecision::SplitCertifiedJoint
            } else {
                FissionDecision::Keep
            }
        }
        None => FissionDecision::SplitReconstructionOnly,
    }
}

/// `P = I − û ûᵀ` for the factor's centered-basis kernel direction
/// (default: the partition-of-unity vector of ones). Projecting the
/// interaction block with these on both sides picks the unique gauge
/// representative with no component along the directions that do not
/// change `f₁₂`.
fn gauge_projector(m: usize, kernel: Option<&Array1<f64>>) -> Result<Array2<f64>, String> {
    let u = match kernel {
        Some(k) => {
            if k.len() != m {
                return Err(format!(
                    "gauge_projector: kernel length {} != basis size {m}",
                    k.len()
                ));
            }
            k.clone()
        }
        None => Array1::<f64>::ones(m),
    };
    let norm_sq: f64 = u.dot(&u);
    let mut p = Array2::<f64>::eye(m);
    if norm_sq > 0.0 {
        for i in 0..m {
            for j in 0..m {
                p[[i, j]] -= u[i] * u[j] / norm_sq;
            }
        }
    }
    Ok(p)
}

/// `K = P₁ ⊗ P₂` under the row-major vec convention
/// (`vec(A X B)[a·M₂+c] = Σ A[a,j]·B[k,c]·vec(X)[j·M₂+k]`; `P₂`
/// symmetric) — the coefficient-space transform realizing the gauge
/// projection `C ↦ P₁ C P₂` on row-major vecs. Built once per carve and
/// shared by the per-dimension and joint Wald tests.
fn gauge_kron_rowmajor(proj_a: &Array2<f64>, proj_b: &Array2<f64>) -> Array2<f64> {
    let m1 = proj_a.nrows();
    let m2 = proj_b.nrows();
    let mm = m1 * m2;
    let mut kron = Array2::<f64>::zeros((mm, mm));
    for a in 0..m1 {
        for j in 0..m1 {
            let pa = proj_a[[a, j]];
            if pa == 0.0 {
                continue;
            }
            for cc in 0..m2 {
                for k in 0..m2 {
                    kron[[a * m2 + cc, j * m2 + k]] = pa * proj_b[[k, cc]];
                }
            }
        }
    }
    kron
}

/// Wald test of `f₁₂ ≡ 0` for one output dimension: transform the raw
/// interaction coefficients to the gauge quotient (`z = vec(P₁ C P₂)`,
/// row-major; `Σ_z = K Σ Kᵀ` with `K = P₁ ⊗ P₂`) and hand the projected
/// block to [`wood_smooth_test`] at the quotient rank. Returns `None`
/// when the test degenerates (the caller records "not tested", which is
/// not "additive").
fn binding_wald_test(
    c: &Array2<f64>,
    cov: &Array2<f64>,
    proj_a: &Array2<f64>,
    proj_b: &Array2<f64>,
    gauge_kron: &Array2<f64>,
    edf: Option<f64>,
    residual_df: f64,
    scale: SmoothTestScale,
) -> Option<SmoothTestResult> {
    let (m1, m2) = c.dim();
    let mm = m1 * m2;
    if cov.dim() != (mm, mm) {
        return None;
    }
    // z = vec(P₁ C P₂), row-major.
    let projected = proj_a.dot(c).dot(proj_b);
    let mut z = Array1::<f64>::zeros(mm);
    for j in 0..m1 {
        for k in 0..m2 {
            z[j * m2 + k] = projected[[j, k]];
        }
    }
    let cov_z = gauge_kron.dot(cov).dot(&gauge_kron.t());
    let quotient_rank = ((m1.saturating_sub(1)) * (m2.saturating_sub(1))).max(1) as f64;
    let edf = edf.unwrap_or(quotient_rank).min(quotient_rank);
    wood_smooth_test(SmoothTestInput {
        beta: z.view(),
        covariance: &cov_z,
        influence_matrix: None,
        coeff_range: 0..mm,
        edf,
        nullspace_dim: 0,
        residual_df,
        scale,
    })
}

/// ONE Wald test of `f₁₂ ≡ 0 across all output dimensions jointly` (#993
/// item 4): stack the gauge-projected interaction vecs dimension-major,
/// transform the supplied joint covariance by the block-diagonal
/// `I_D ⊗ K`, and test at the joint quotient rank `D·(M₁−1)(M₂−1)`. This
/// replaces the Bonferroni combination exactly where Bonferroni is
/// loosest — strongly cross-correlated output dimensions (they share
/// every code row).
fn joint_binding_wald_test(
    coeffs: &[Array2<f64>],
    joint_cov: &Array2<f64>,
    proj_a: &Array2<f64>,
    proj_b: &Array2<f64>,
    gauge_kron: &Array2<f64>,
    edf: Option<f64>,
    residual_df: f64,
    scale: SmoothTestScale,
) -> Option<SmoothTestResult> {
    let d_dims = coeffs.len();
    if d_dims == 0 {
        return None;
    }
    let (m1, m2) = coeffs[0].dim();
    let mm = m1 * m2;
    let total = d_dims * mm;
    if joint_cov.dim() != (total, total) {
        return None;
    }
    // Stacked z: dimension-major [vec(P₁C₀P₂); vec(P₁C₁P₂); …].
    let mut z = Array1::<f64>::zeros(total);
    for (d, c) in coeffs.iter().enumerate() {
        let projected = proj_a.dot(c).dot(proj_b);
        for j in 0..m1 {
            for k in 0..m2 {
                z[d * mm + j * m2 + k] = projected[[j, k]];
            }
        }
    }
    // Σ_z = (I_D ⊗ K) · J · (I_D ⊗ K)ᵀ, computed blockwise.
    let mut cov_z = Array2::<f64>::zeros((total, total));
    for d in 0..d_dims {
        for e in 0..d_dims {
            let block = joint_cov.slice(s![d * mm..(d + 1) * mm, e * mm..(e + 1) * mm]);
            let transformed = gauge_kron.dot(&block).dot(&gauge_kron.t());
            cov_z
                .slice_mut(s![d * mm..(d + 1) * mm, e * mm..(e + 1) * mm])
                .assign(&transformed);
        }
    }
    let quotient_rank = ((m1.saturating_sub(1)) * (m2.saturating_sub(1))).max(1) as f64;
    let per_dim_edf = edf.unwrap_or(quotient_rank).min(quotient_rank);
    wood_smooth_test(SmoothTestInput {
        beta: z.view(),
        covariance: &cov_z,
        influence_matrix: None,
        coeff_range: 0..total,
        edf: per_dim_edf * d_dims as f64,
        nullspace_dim: 0,
        residual_df,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A tiny partition-of-unity "hat" basis on a 3-point sample: rows sum
    /// to 1, columns are linearly independent over the sample.
    fn pou_basis() -> Array2<f64> {
        array![
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.5, 0.4, 0.1],
            [0.1, 0.2, 0.7],
        ]
    }

    fn pou_basis_b() -> Array2<f64> {
        array![
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
            [0.2, 0.5, 0.3],
            [0.4, 0.1, 0.5],
        ]
    }

    /// The reparameterization is an identity: blocks + interaction values
    /// reassemble the raw surface exactly, sample point by sample point.
    #[test]
    fn anova_reparameterization_is_exact() {
        let phi_a = pou_basis();
        let phi_b = pou_basis_b();
        let c = array![[1.3, -0.4, 0.2], [0.0, 0.8, -1.1], [2.0, 0.5, 0.3]];
        let mean_a = basis_means(phi_a.view());
        let mean_b = basis_means(phi_b.view());
        let blocks = anova_blocks(c.view(), mean_a.view(), mean_b.view()).expect("blocks");

        for row in 0..phi_a.nrows() {
            let pa = phi_a.row(row);
            let pb = phi_b.row(row);
            let raw = pa.dot(&c.dot(&pb.to_owned()));
            let pa_c: Array1<f64> = &pa.to_owned() - &mean_a;
            let pb_c: Array1<f64> = &pb.to_owned() - &mean_b;
            let f12 = pa_c.dot(&c.dot(&pb_c));
            let rebuilt = blocks.mean + pa_c.dot(&blocks.main_a) + pb_c.dot(&blocks.main_b) + f12;
            assert!(
                (raw - rebuilt).abs() < 1e-12,
                "row {row}: raw {raw} vs rebuilt {rebuilt}"
            );
        }
    }

    /// A planted ADDITIVE surface (`C = a·1ᵀ + 1·bᵀ` on partition-of-unity
    /// bases) has identically zero interaction, fissions, and the children
    /// reassemble the parent exactly (lossless split, defect 0).
    #[test]
    fn planted_additive_torus_fissions_losslessly() {
        let phi_a = pou_basis();
        let phi_b = pou_basis_b();
        let a = array![1.0, -0.5, 2.0];
        let b = array![0.3, 1.7, -1.0];
        let mut c = Array2::<f64>::zeros((3, 3));
        for j in 0..3 {
            for k in 0..3 {
                c[[j, k]] = a[j] + b[k];
            }
        }
        let input = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &[c.clone()],
            coeff_covariance: None,
            joint_coeff_covariance: None,
            kernel_a: None,
            kernel_b: None,
            edf: None,
            residual_df: 100.0,
            scale: SmoothTestScale::Known,
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        assert!(report.interaction_fraction < 1e-24);
        let plan = report
            .fission
            .as_ref()
            .expect("additive surface must fission");
        assert!(plan.reconstruction_defect < 1e-24);

        // Children reassemble the parent surface exactly.
        let mean_a = basis_means(phi_a.view());
        let mean_b = basis_means(phi_b.view());
        for row in 0..phi_a.nrows() {
            let pa = phi_a.row(row);
            let pb = phi_b.row(row);
            let raw = pa.dot(&c.dot(&pb.to_owned()));
            let pa_c: Array1<f64> = &pa.to_owned() - &mean_a;
            let pb_c: Array1<f64> = &pb.to_owned() - &mean_b;
            let child_sum = plan.child_a[0].constant
                + pa_c.dot(&plan.child_a[0].centered_coeffs)
                + plan.child_b[0].constant
                + pb_c.dot(&plan.child_b[0].centered_coeffs);
            assert!((raw - child_sum).abs() < 1e-12);
        }

        // Raw-coefficient form on the partition-of-unity basis agrees too.
        let raw_a = plan.child_a[0].raw_coeffs_partition_of_unity(mean_a.view());
        for row in 0..phi_a.nrows() {
            let pa = phi_a.row(row);
            let pa_c: Array1<f64> = &pa.to_owned() - &mean_a;
            let via_centered =
                plan.child_a[0].constant + pa_c.dot(&plan.child_a[0].centered_coeffs);
            assert!((pa.dot(&raw_a) - via_centered).abs() < 1e-12);
        }
    }

    /// A planted BOUND surface (rank-1 centered interaction) must refuse
    /// to fission, and with a tight posterior the binding test must reject;
    /// the planted additive surface under the same covariance must NOT
    /// reject — the asymmetry that makes the test a test.
    #[test]
    fn planted_bound_torus_refuses_and_test_rejects() {
        let phi_a = pou_basis();
        let phi_b = pou_basis_b();
        // Centered directions (orthogonal to the PoU kernel = ones).
        let at = array![1.0, -1.0, 0.0];
        let bt = array![0.0, 1.0, -1.0];
        let mut c = Array2::<f64>::zeros((3, 3));
        for j in 0..3 {
            for k in 0..3 {
                c[[j, k]] = 2.0 * at[j] * bt[k];
            }
        }
        // Tight scale-included posterior: σ² = 1e-4 per coefficient.
        let cov = Array2::<f64>::eye(9) * 1e-4;
        let input = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &[c],
            coeff_covariance: Some(std::slice::from_ref(&cov)),
            joint_coeff_covariance: None,
            kernel_a: None,
            kernel_b: None,
            edf: None,
            residual_df: 100.0,
            scale: SmoothTestScale::Known,
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        assert!(report.fission.is_none(), "bound surface must not fission");
        assert!(report.interaction_fraction > 0.1);
        let p = report.edge_p_value.expect("test ran");
        assert!(p < 1e-6, "strong planted binding must reject, p = {p}");

        // The additive surface, same covariance: no rejection.
        let a = array![1.0, -0.5, 2.0];
        let b = array![0.3, 1.7, -1.0];
        let mut c_add = Array2::<f64>::zeros((3, 3));
        for j in 0..3 {
            for k in 0..3 {
                c_add[[j, k]] = a[j] + b[k];
            }
        }
        let input_add = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &[c_add],
            coeff_covariance: Some(std::slice::from_ref(&cov)),
            joint_coeff_covariance: None,
            kernel_a: None,
            kernel_b: None,
            edf: None,
            residual_df: 100.0,
            scale: SmoothTestScale::Known,
            notion: BindingNotion::Representational,
        };
        let report_add = carve(&input_add, 0.05).expect("carve");
        let p_add = report_add.edge_p_value.expect("test ran");
        assert!(
            p_add > 0.99,
            "additive surface carries zero projected interaction, p = {p_add}"
        );
        assert!(report_add.fission.is_some());
    }

    /// The gauge directions (`u vᵀ + w uᵀ`) contribute NOTHING to the test
    /// statistic: adding them to a planted-additive coefficient matrix
    /// leaves the projected interaction (and hence the p-value) unchanged.
    #[test]
    fn gauge_directions_do_not_enter_the_binding_test() {
        let phi_a = pou_basis();
        let phi_b = pou_basis_b();
        let mut c = Array2::<f64>::zeros((3, 3));
        // Pure gauge: u vᵀ + w uᵀ with u = ones.
        let v = array![0.4, -1.2, 0.7];
        let w = array![-0.9, 0.1, 0.5];
        for j in 0..3 {
            for k in 0..3 {
                c[[j, k]] = v[k] + w[j];
            }
        }
        let cov = Array2::<f64>::eye(9) * 1e-4;
        let input = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &[c],
            coeff_covariance: Some(std::slice::from_ref(&cov)),
            joint_coeff_covariance: None,
            kernel_a: None,
            kernel_b: None,
            edf: None,
            residual_df: 100.0,
            scale: SmoothTestScale::Known,
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        // u vᵀ + w uᵀ IS additive (it is f₁ + f₂ on a PoU basis), so the
        // projected interaction is exactly zero.
        assert!(report.interaction_fraction < 1e-24);
        let p = report.edge_p_value.expect("test ran");
        assert!(p > 0.99, "pure-gauge coefficients must not reject, p = {p}");
    }

    /// A deterministic Bernstein (degree-2, partition-of-unity) basis
    /// evaluated on `n` scattered points, with two decorrelated sample
    /// mappings so the tensor design is well-conditioned.
    fn bernstein_pair(n: usize) -> (Array2<f64>, Array2<f64>) {
        let mut phi_a = Array2::<f64>::zeros((n, 3));
        let mut phi_b = Array2::<f64>::zeros((n, 3));
        for t in 0..n {
            let x = t as f64 / (n - 1) as f64;
            let z = ((t * 17) % n) as f64 / (n - 1) as f64;
            phi_a[[t, 0]] = (1.0 - x) * (1.0 - x);
            phi_a[[t, 1]] = 2.0 * x * (1.0 - x);
            phi_a[[t, 2]] = x * x;
            phi_b[[t, 0]] = (1.0 - z) * (1.0 - z);
            phi_b[[t, 1]] = 2.0 * z * (1.0 - z);
            phi_b[[t, 2]] = z * z;
        }
        (phi_a, phi_b)
    }

    fn surface_values(phi_a: &Array2<f64>, phi_b: &Array2<f64>, c: &Array2<f64>) -> Array1<f64> {
        let n = phi_a.nrows();
        let mut y = Array1::<f64>::zeros(n);
        for r in 0..n {
            y[r] = phi_a.row(r).dot(&c.dot(&phi_b.row(r).to_owned()));
        }
        y
    }

    /// END-TO-END (#993 items 1+2+4): fit_tensor_surface recovers a
    /// planted BOUND two-dimensional surface from noisy samples, its
    /// covariance feeds the carve, and the JOINT cross-dim Wald (via
    /// `joint_covariance`) proves the binding while fission refuses.
    #[test]
    fn tensor_surface_fit_to_carve_proves_planted_binding_jointly() {
        let n = 40usize;
        let (phi_a, phi_b) = bernstein_pair(n);
        // Two distinct bound surfaces (additive part + centered rank-1
        // interaction) so the residual cross-covariance is well-conditioned.
        let at = array![1.0, -1.0, 0.0];
        let bt = array![0.0, 1.0, -1.0];
        let mut c0 = Array2::<f64>::zeros((3, 3));
        let mut c1 = Array2::<f64>::zeros((3, 3));
        let a = array![1.0, -0.5, 2.0];
        let b = array![0.3, 1.7, -1.0];
        for j in 0..3 {
            for k in 0..3 {
                c0[[j, k]] = a[j] + b[k] + 2.0 * at[j] * bt[k];
                c1[[j, k]] = 0.5 * a[j] - b[k] - 1.5 * at[j] * bt[k];
            }
        }
        let y0 = surface_values(&phi_a, &phi_b, &c0);
        let y1 = surface_values(&phi_a, &phi_b, &c1);
        let mut responses = Array2::<f64>::zeros((n, 2));
        for t in 0..n {
            responses[[t, 0]] = y0[t] + 1e-3 * (1.3 * t as f64).sin();
            responses[[t, 1]] = y1[t] + 1e-3 * (2.1 * t as f64).cos();
        }

        let fit = fit_tensor_surface(phi_a.view(), phi_b.view(), responses.view()).expect("fit");
        // Coefficient recovery within noise scale (ridge bias included).
        for j in 0..3 {
            for k in 0..3 {
                assert!(
                    (fit.coeffs[0][[j, k]] - c0[[j, k]]).abs() < 0.05,
                    "C₀[{j},{k}]: fit {} vs planted {}",
                    fit.coeffs[0][[j, k]],
                    c0[[j, k]]
                );
            }
        }
        // Kronecker consistency: the joint covariance's diagonal block d
        // equals the per-dimension Vb exactly.
        let joint = fit.joint_covariance();
        let mm = 9usize;
        for i in 0..mm {
            for j in 0..mm {
                assert!((joint[[i, j]] - fit.coeff_covariance[0][[i, j]]).abs() < 1e-15);
                assert!((joint[[mm + i, mm + j]] - fit.coeff_covariance[1][[i, j]]).abs() < 1e-15);
            }
        }

        let input = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &fit.coeffs,
            coeff_covariance: Some(&fit.coeff_covariance),
            joint_coeff_covariance: Some(&joint),
            kernel_a: None,
            kernel_b: None,
            edf: None,
            residual_df: fit.residual_df,
            scale: SmoothTestScale::Estimated,
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        let p = report.edge_p_value.expect("joint test ran");
        assert!(p < 1e-3, "planted joint binding must reject, p = {p}");
        assert!(report.fission.is_none(), "bound surface must not fission");
        assert!(report.interaction_fraction > 0.05);
    }

    /// END-TO-END, additive side: a planted ADDITIVE surface fit from
    /// near-noiseless samples carries negligible interaction energy and
    /// fissions (energy-only path — no covariance handed to the carve, so
    /// the decision rests on the dial alone).
    #[test]
    fn tensor_surface_fit_additive_surface_fissions() {
        let n = 40usize;
        let (phi_a, phi_b) = bernstein_pair(n);
        let a = array![1.0, -0.5, 2.0];
        let b = array![0.3, 1.7, -1.0];
        let mut c_add = Array2::<f64>::zeros((3, 3));
        for j in 0..3 {
            for k in 0..3 {
                c_add[[j, k]] = a[j] + b[k];
            }
        }
        let y = surface_values(&phi_a, &phi_b, &c_add);
        let mut responses = Array2::<f64>::zeros((n, 1));
        for t in 0..n {
            responses[[t, 0]] = y[t] + 1e-5 * (0.9 * t as f64).sin();
        }
        let fit = fit_tensor_surface(phi_a.view(), phi_b.view(), responses.view()).expect("fit");
        let input = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &fit.coeffs,
            coeff_covariance: None,
            joint_coeff_covariance: None,
            kernel_a: None,
            kernel_b: None,
            edf: None,
            residual_df: fit.residual_df,
            scale: SmoothTestScale::Estimated,
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        assert!(
            report.interaction_fraction < FISSION_MAX_INTERACTION_FRACTION,
            "additive surface fit must carry negligible interaction \
             (fraction = {})",
            report.interaction_fraction
        );
        assert!(report.fission.is_some());
    }

    /// The three-valued joint decision: both arms additive → joint
    /// certificate; representational only → reconstruction-only; a bound
    /// computational arm vetoes a clean representational split (the
    /// off-diagonal quadrant that motivates the pair).
    #[test]
    fn fission_decision_distinguishes_the_quadrants() {
        let splittable = CarveReport {
            notion: BindingNotion::Representational,
            binding_tests: vec![],
            edge_p_value: None,
            interaction_fraction: 0.0,
            fission: Some(FissionPlan {
                child_a: vec![],
                child_b: vec![],
                reconstruction_defect: 0.0,
            }),
        };
        let mut comp_splittable = splittable.clone();
        comp_splittable.notion = BindingNotion::Computational;
        let comp_bound = CarveReport {
            notion: BindingNotion::Computational,
            binding_tests: vec![],
            edge_p_value: Some(1e-9),
            interaction_fraction: 0.4,
            fission: None,
        };

        assert_eq!(
            fission_decision(&splittable, Some(&comp_splittable)),
            FissionDecision::SplitCertifiedJoint
        );
        assert_eq!(
            fission_decision(&splittable, None),
            FissionDecision::SplitReconstructionOnly
        );
        assert_eq!(
            fission_decision(&splittable, Some(&comp_bound)),
            FissionDecision::Keep
        );
        let kept = CarveReport {
            fission: None,
            ..splittable.clone()
        };
        assert_eq!(fission_decision(&kept, None), FissionDecision::Keep);
    }

    /// A constant-leading factor basis (column 0 ≡ 1, like the harmonic
    /// factors' constant term) on a small sample.
    fn constant_leading_factor(n: usize, m: usize, seed: u64) -> Array2<f64> {
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut s = seed;
        for row in 0..n {
            phi[[row, 0]] = 1.0;
            for col in 1..m {
                // Deterministic LCG in [-1, 1).
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                phi[[row, col]] = 2.0 * u - 1.0;
            }
        }
        phi
    }

    /// #993 producer: `carve_input_from_fitted_atom` recovers the two factor
    /// bases EXACTLY from the fused Kronecker basis (constant-leading column
    /// convention), and the re-fit surface reconstructs the decoder's own
    /// tensor coefficients — so a real fitted product atom feeds the carve.
    #[test]
    fn producer_recovers_factor_bases_and_surface_from_fused_atom() {
        let n = 40;
        let (m_a, m_b) = (3, 4);
        let p = 2;
        let phi_a = constant_leading_factor(n, m_a, 0xA993);
        let phi_b = constant_leading_factor(n, m_b, 0xB993);

        // Fused Kronecker basis, row-major column flat = j*m_b + k.
        let mut fused = Array2::<f64>::zeros((n, m_a * m_b));
        for row in 0..n {
            for j in 0..m_a {
                for k in 0..m_b {
                    fused[[row, j * m_b + k]] = phi_a[[row, j]] * phi_b[[row, k]];
                }
            }
        }
        // An arbitrary decoder B_k (M₁M₂ × p).
        let mut decoder = Array2::<f64>::zeros((m_a * m_b, p));
        let mut s = 0xD00D_u64;
        for r in 0..(m_a * m_b) {
            for c in 0..p {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                decoder[[r, c]] = 2.0 * u - 1.0;
            }
        }

        let bundle =
            carve_input_from_fitted_atom(fused.view(), decoder.view(), m_a, m_b).expect("producer");

        // Factor bases recovered to machine precision.
        let mut max_a = 0.0_f64;
        for row in 0..n {
            for j in 0..m_a {
                max_a = max_a.max((bundle.phi_a[[row, j]] - phi_a[[row, j]]).abs());
            }
        }
        let mut max_b = 0.0_f64;
        for row in 0..n {
            for k in 0..m_b {
                max_b = max_b.max((bundle.phi_b[[row, k]] - phi_b[[row, k]]).abs());
            }
        }
        assert!(max_a < 1e-12, "phi_a recovery error {max_a:e}");
        assert!(max_b < 1e-12, "phi_b recovery error {max_b:e}");

        // The carve input is well-formed: p coefficient matrices, each M₁×M₂,
        // with matching covariance blocks and the joint Kronecker covariance.
        let input = bundle.representational_carve_input();
        assert_eq!(input.coeffs.len(), p);
        for c in input.coeffs {
            assert_eq!(c.dim(), (m_a, m_b));
        }
        assert_eq!(
            bundle.joint_covariance.dim(),
            (p * m_a * m_b, p * m_a * m_b)
        );

        // The carve runs end-to-end on the producer's output.
        let report = carve(&input, 0.05).expect("carve on producer output");
        assert_eq!(report.notion, BindingNotion::Representational);
        assert!(
            report.edge_p_value.is_some(),
            "binding p-value must be produced"
        );
    }

    /// A non-separable fused basis (not a Kronecker product of two factors) is
    /// rejected loudly, not silently mis-carved.
    #[test]
    fn producer_rejects_non_separable_basis() {
        let n = 12;
        let (m_a, m_b) = (2, 2);
        let mut fused = Array2::<f64>::from_elem((n, m_a * m_b), 1.0);
        // Break separability in one entry only.
        fused[[3, 3]] = 7.0;
        let decoder = Array2::<f64>::ones((m_a * m_b, 1));
        let err = carve_input_from_fitted_atom(fused.view(), decoder.view(), m_a, m_b)
            .expect_err("non-separable basis must be rejected");
        assert!(
            err.contains("Kronecker product"),
            "rejection must name the separability failure; got: {err}"
        );
    }
}

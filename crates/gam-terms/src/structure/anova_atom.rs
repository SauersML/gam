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
//! data PROVES `f₁₂ ≠ 0`; fission additionally demands that the interaction
//! energy the split discards be unresolved at the carve's own resolution,
//! its rounding floor plus the posterior's α-level bound ([`carve`]),
//! because absence of evidence is not evidence of absence. Atoms failing
//! both stay whole and CONTESTED — the
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
//! covariance for the resolution bound, and the sample-space binding test
//! of [`InteractionTest`]). It deliberately does NOT add an
//! in-fit ANOVA basis kind: two independent circles are just two atoms
//! summing — ordinary superposition, the default multi-atom model — so
//! the product machinery is only ever needed at the moment a fitted pair
//! shows dependent codes and the structure search must adjudicate
//! merge-vs-keep. That adjudication consumes this carve.
//!
//! # The binding test lives in sample space (gauge-free by construction)
//!
//! On a partition-of-unity factor basis (B-splines: `Σ_j φ_j ≡ 1`) the
//! empirically centered basis functions carry one exact linear dependence
//! per factor, so the interaction coefficients `C` carry gauge directions
//! that change nothing about `f₁₂`. A coefficient-space test must quotient
//! them out and then guess the rank of what is left. [`InteractionTest`]
//! never enters coefficient space: it is the exact Gaussian nested-model
//! test of the additive column space `W = [1, Φ¹, Φ²]` against the full
//! space `F = [W, φ¹ ⊗ φ²]` on the code sample. The interaction directions
//! are the numerical range of the tensor columns projected off `span W`,
//! so any reparameterization of either factor basis leaves the test
//! unchanged, and its rank `k = rank F − rank W` is read off the sample —
//! `(M₁−1)(M₂−1)` for constant-leading or partition-of-unity factors, with
//! no EDF substitute and no floor. When the factor bases do not span the
//! constant, `span F ⊋ span(φ¹ ⊗ φ²)` and the alternative is the full
//! space. Per output dimension the statistic is exactly `F(k, ν)`,
//! `ν = n − rank F`; the edge-level statistic is Wilks' `Λ` over all output
//! dimensions jointly, whose exact null law is evaluated in closed form, or
//! as a certified one-dimensional convolution when the output count and the
//! interaction rank are both odd.
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use gam_linalg::roundoff::factor_singular_band;
use gam_math::probability::{fisher_snedecor_sf, ln_regularized_beta_lower_from_log_x};
use gam_math::special::gauss_legendre;
use statrs::function::beta::ln_beta;

/// Which binding notion a carve report speaks about (see module docs).
///
/// The two are independent, and which of them a given adjudication ran is
/// carried in the answer rather than assumed: `fission_decision` returns
/// [`FissionDecision::SplitReconstructionOnly`] exactly when only the
/// representational carve was supplied, and
/// [`FissionDecision::SplitCertifiedJoint`] only when both ran and both
/// allow the split. So a caller that has no pulled-back readout coefficients
/// still gets a correct, self-describing verdict — it just is not the joint
/// one, and the enum says so.
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
/// interaction block on the centered tensor basis is `C` itself (its
/// binding test runs in sample space, see module docs).
#[derive(Clone, Debug)]
pub(crate) struct AnovaBlocks {
    pub mean: f64,
    pub main_a: Array1<f64>,
    pub main_b: Array1<f64>,
}

/// Empirical mean of each basis column over the code sample — the
/// centering vector `m` that pins the ANOVA gauge to the empirical code
/// measure.
pub(crate) fn basis_means(phi: ArrayView2<'_, f64>) -> Array1<f64> {
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
pub(crate) fn anova_blocks(
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

impl ChildDecoder {}

/// The lossless-on-the-additive-part split: child atoms inheriting the
/// main-effect blocks. Gauge choice (documented, fixed): the grand mean
/// `g₀` rides with child A; child B is centered. The interaction energy
/// the split discards is DECLARED in `reconstruction_defect` — by the
/// fission rule it is unresolved at the carve's resolution ([`carve`]), but
/// it is never silently zero.
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
    /// Exact nested-model F test of the interaction, one per output
    /// dimension ([`InteractionTest::per_dimension`]); the error names why a
    /// dimension has no test.
    pub binding_tests: Vec<Result<BindingFTest, BindingTestUnavailable>>,
    /// Edge-level binding test over all output dimensions jointly
    /// ([`InteractionTest::edge`]). Its p-value ([`Self::edge_p_value`]) is
    /// the number that feeds `structure_evidence::ClaimKind::BindingEdge`
    /// through `log_e_from_p_calibrator`.
    pub edge_test: Result<EdgeBindingTest, BindingTestUnavailable>,
    /// Fraction of centered surface energy carried by the interaction,
    /// aggregated over output dimensions — the continuous "how bound"
    /// dial (0 = perfectly additive, 1 = pure interaction).
    pub interaction_fraction: f64,
    /// The lossless split, present iff this notion's carve allows it:
    /// interaction unresolved at the carve's resolution AND not proven present.
    pub fission: Option<FissionPlan>,
}

impl CarveReport {
    /// The edge-level binding p-value, or why the edge has no test.
    pub fn edge_p_value(&self) -> Result<f64, BindingTestUnavailable> {
        self.edge_test.map(|test| test.p_value())
    }
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

/// Why a binding test has no p-value. Each is a property of the sample,
/// reported instead of a number that would claim a calibration it lacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingTestUnavailable {
    /// The carve was handed no sample-space test ([`CarveInput::interaction_test`]).
    NoStatistics,
    /// The tensor columns add no direction to the additive space on this
    /// sample (`rank F = rank W`): there is no interaction to test.
    NoInteractionDirections,
    /// The full space spans every sample (`rank F = n`): no residual is
    /// left to weigh the interaction against.
    NoResidualDegreesOfFreedom,
    /// Output dimension `dim` lies in the full space to rounding: its
    /// residual carries no variation to calibrate a test against (an exact
    /// reconstruction, for instance).
    NoResidualVariation { dim: usize },
    /// The residual cross-product of the output dimensions is singular
    /// (fewer residual degrees of freedom than outputs, or an output that
    /// is a linear combination of the others in the residual), so Wilks'
    /// `Λ` is undefined.
    ResidualCovarianceRankDeficient { residual_df: usize, dims: usize },
}

/// One output dimension's exact nested-model F test of the interaction:
/// `statistic ~ F(numerator_df, denominator_df)` under no interaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BindingFTest {
    pub statistic: f64,
    pub numerator_df: usize,
    pub denominator_df: usize,
    pub p_value: f64,
}

/// Wilks' likelihood-ratio test of no interaction across all output
/// dimensions jointly: `Λ = |E| / |E + H|` with `E` the residual and `H` the
/// interaction cross-product, `Λ ~ Λ(dims, residual_df, interaction_rank)`
/// under no interaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WilksTest {
    /// `−ln Λ`.
    pub neg_log_lambda: f64,
    pub dims: usize,
    pub interaction_rank: usize,
    pub residual_df: usize,
    pub p_value: f64,
}

/// The edge-level binding test: the F test itself for a single output
/// dimension, Wilks' `Λ` for several.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeBindingTest {
    SingleOutput(BindingFTest),
    Wilks(WilksTest),
}

impl EdgeBindingTest {
    pub fn p_value(&self) -> f64 {
        match self {
            Self::SingleOutput(test) => test.p_value,
            Self::Wilks(test) => test.p_value,
        }
    }
}

/// The exact Gaussian test of "no interaction" on the code sample.
///
/// With `W = [1, Φ¹, Φ²]` (the additive surfaces) and `F = [W, φ¹ ⊗ φ²]`,
/// the responses `Y` (`n × D`) are tested for `E[Y] ∈ span W` against
/// `E[Y] ∈ span F`, rows independent Gaussian with a common covariance.
/// With `Q` an orthonormal basis of `span F ⊖ span W` (`k` columns), the
/// interaction scores `S = Qᵀ Y` (`k × D`) and the full-space residual
/// `E` (`ν = n − rank F` degrees of freedom) are independent, so per output
/// dimension `(‖s_d‖²/k) / (‖e_d‖²/ν) ~ F(k, ν)` exactly, and across outputs
/// `Λ = |EᵀE| / |EᵀE + SᵀS|` is Wilks-distributed exactly.
///
/// Everything is read off `span W` and `span F` alone: reparameterizing
/// either factor basis by any invertible map, or carrying gauge directions
/// in it, leaves the test unchanged. Numerical ranks are the singular
/// directions above each matrix's backward-error band
/// (`gam_linalg::roundoff::factor_singular_band`).
#[derive(Clone, Debug)]
pub struct InteractionTest {
    n: usize,
    m1: usize,
    m2: usize,
    /// `k = rank F − rank W`, the numerator degrees of freedom.
    pub interaction_rank: usize,
    /// `ν = n − rank F`, the residual degrees of freedom.
    pub residual_df: usize,
    pub per_dimension: Vec<Result<BindingFTest, BindingTestUnavailable>>,
    pub edge: Result<EdgeBindingTest, BindingTestUnavailable>,
}

impl InteractionTest {
    /// Run the test for responses `responses` (`n × D`) on factor bases
    /// `phi_a` (`n × M₁`) and `phi_b` (`n × M₂`).
    pub fn from_sample(
        phi_a: ArrayView2<'_, f64>,
        phi_b: ArrayView2<'_, f64>,
        responses: ArrayView2<'_, f64>,
    ) -> Result<Self, String> {
        let n = phi_a.nrows();
        let m1 = phi_a.ncols();
        let m2 = phi_b.ncols();
        let dims = responses.ncols();
        if phi_b.nrows() != n || responses.nrows() != n {
            return Err(format!(
                "InteractionTest: sample sizes disagree (phi_a {n}, phi_b {}, responses {})",
                phi_b.nrows(),
                responses.nrows()
            ));
        }
        if n == 0 || m1 == 0 || m2 == 0 || dims == 0 {
            return Err(format!(
                "InteractionTest: degenerate problem (n={n}, M₁={m1}, M₂={m2}, D={dims})"
            ));
        }
        if phi_a
            .iter()
            .chain(phi_b.iter())
            .chain(responses.iter())
            .any(|v| !v.is_finite())
        {
            return Err("InteractionTest: non-finite basis or response entry".to_string());
        }

        let additive_width = 1 + m1 + m2;
        let full_width = additive_width + m1 * m2;
        let mut full = Array2::<f64>::zeros((n, full_width));
        for row in 0..n {
            full[[row, 0]] = 1.0;
            for j in 0..m1 {
                full[[row, 1 + j]] = phi_a[[row, j]];
            }
            for k in 0..m2 {
                full[[row, 1 + m1 + k]] = phi_b[[row, k]];
            }
            for j in 0..m1 {
                for k in 0..m2 {
                    full[[row, additive_width + j * m2 + k]] = phi_a[[row, j]] * phi_b[[row, k]];
                }
            }
        }
        let additive = full.slice(s![.., ..additive_width]);
        let additive_band = factor_singular_band(
            n,
            additive_width,
            largest_singular_value(additive, "additive space")?,
        );
        let additive_range = numerical_range(additive, additive_band, "additive space")?;
        // The interaction directions: the tensor columns off span W, kept above the
        // FULL matrix's band, so rank W + k is rank F at F's own resolution.
        let full_band = factor_singular_band(
            n,
            full_width,
            largest_singular_value(full.view(), "full space")?,
        );
        let interaction_columns = project_out(
            &additive_range,
            full.slice(s![.., additive_width..]).to_owned(),
        );
        let interaction =
            numerical_range(interaction_columns.view(), full_band, "interaction space")?;
        let interaction_rank = interaction.ncols();
        let full_rank = additive_range.ncols() + interaction_rank;
        let residual_df = n.checked_sub(full_rank).ok_or_else(|| {
            format!("InteractionTest: numerical rank {full_rank} exceeds the sample size {n}")
        })?;

        let additive_residual = project_out(&additive_range, responses.to_owned());
        let scores = interaction.t().dot(&additive_residual); // k × D
        let residual = project_out(&interaction, additive_residual); // n × D

        let unavailable = if interaction_rank == 0 {
            Some(BindingTestUnavailable::NoInteractionDirections)
        } else if residual_df == 0 {
            Some(BindingTestUnavailable::NoResidualDegreesOfFreedom)
        } else {
            None
        };
        let per_dimension: Vec<Result<BindingFTest, BindingTestUnavailable>> = (0..dims)
            .map(|dim| {
                if let Some(reason) = unavailable {
                    return Err(reason);
                }
                let residual_energy = column_energy(residual.column(dim));
                // A residual inside the projections' rounding band of the response
                // is a response in span F: nothing is left to calibrate against.
                let response_norm = column_energy(responses.column(dim)).sqrt();
                if residual_energy.sqrt() <= factor_singular_band(n, full_width, response_norm) {
                    return Err(BindingTestUnavailable::NoResidualVariation { dim });
                }
                let score_energy = column_energy(scores.column(dim));
                let statistic = (score_energy / interaction_rank as f64)
                    / (residual_energy / residual_df as f64);
                Ok(BindingFTest {
                    statistic,
                    numerator_df: interaction_rank,
                    denominator_df: residual_df,
                    p_value: fisher_snedecor_sf(
                        statistic,
                        interaction_rank as f64,
                        residual_df as f64,
                    ),
                })
            })
            .collect();

        let edge = if dims == 1 {
            per_dimension[0].map(EdgeBindingTest::SingleOutput)
        } else if let Some(Err(reason)) = per_dimension.iter().find(|test| test.is_err()) {
            Err(*reason)
        } else if residual_df < dims {
            Err(BindingTestUnavailable::ResidualCovarianceRankDeficient { residual_df, dims })
        } else {
            wilks_test(&residual, &scores, residual_df)?
        };

        Ok(Self {
            n,
            m1,
            m2,
            interaction_rank,
            residual_df,
            per_dimension,
            edge,
        })
    }
}

/// A penalized tensor-surface fit over the code sample: the producer of
/// [`CarveInput`]s for BOTH binding notions (#993 items 1–2).
///
/// `coeffs[d]` is the fitted `M₁ × M₂` coefficient matrix for response
/// dimension `d`; `coeff_covariance[d]` is the matching SCALE-INCLUDED
/// posterior covariance of its row-major vec; `joint_covariance()`
/// assembles the cross-dimension covariance. The carve reads these for its
/// posterior resolution bound, and reads `interaction_test` — the exact
/// sample-space test of the same responses on the same bases — for binding.
/// The fit is evaluated against the SAME empirical code measure the carve
/// centers against, which is the coherence the production fit's own Hessian
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
    /// (`V Λ⁺ Vᵀ/(1 + λ)` over the design's numerical range, `M₁M₂ × M₁M₂`);
    /// `coeff_covariance[d]` is this times `residual_cross_cov[d,d]`.
    pub unit_covariance: Array2<f64>,
    /// REML-selected function-mass weight: the surface is the least-squares
    /// surface shrunk by `1/(1 + λ)`. `∞` is the exact zero surface, `0` no
    /// shrinkage.
    pub lambda: f64,
    /// Effective degrees of freedom `rank(X)/(1 + λ)` (per dimension; the
    /// design and λ are shared).
    pub edf: f64,
    /// Residual degrees of freedom `n − edf` of the shrunk fit, the divisor
    /// of `residual_cross_cov`.
    pub residual_df: f64,
    /// Per response dimension, a bound on the absolute rounding error of every
    /// entry of `coeffs[d]`, carried from the solve that produced them (see
    /// [`fit_tensor_surface`]). [`carve`] reads it to decide whether an
    /// interaction block is distinguishable from what an exactly additive
    /// surface leaves after this solve.
    pub coeff_band: Vec<f64>,
    /// The exact sample-space binding test of `responses` on these factor
    /// bases ([`InteractionTest::from_sample`]); feed to
    /// [`CarveInput::interaction_test`].
    pub interaction_test: InteractionTest,
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
/// sampled responses under a function-mass penalty whose weight is chosen by
/// GAUSSIAN REML (profiled σ², closed-form criterion on the design's
/// eigenbasis — no GCV, per policy), returning the posterior mean coefficients
/// AND their scale-included posterior covariance.
///
/// The penalty is on the surface, never on its coefficients (SPEC rule 5): each
/// response carries `λ‖X vec(C_d)‖²/σ_d²`, the surface's mass over the code
/// sample, so the prior on the fitted values is `X vec(C_d) ~ N(0, (σ_d²/λ) P)`
/// with `P` the projector onto the design's range. A coefficient ridge
/// `λ‖vec(C_d)‖²` changes when the factor bases are rescaled or mixed; this
/// penalty, the selected λ and the fitted surface do not. The posterior mean is
/// the least-squares surface shrunk by `1/(1 + λ)`, and λ = ∞ (the exact zero
/// surface) is selected when the evidence supports no surface.
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
    let spectral_radius = evals
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !spectral_radius.is_finite() {
        return Err("fit_tensor_surface: design eigendecomposition is non-finite".to_string());
    }
    // XᵀX is positive semidefinite.  Permit projection to the PSD cone only
    // inside the eigensolver's dimension-scaled backward-error band; a mode
    // below that band is evidence of invalid arithmetic, not a zero mode.
    let spectral_roundoff = f64::EPSILON * mm as f64 * spectral_radius;
    for (index, &value) in evals.iter().enumerate() {
        if value < -spectral_roundoff {
            return Err(format!(
                "fit_tensor_surface: Gram eigenvalue {index} is {value}, below the PSD \
                 roundoff band -{spectral_roundoff}"
            ));
        }
    }
    // The modes inside that band are the design's numerical null space: no surface
    // over the code sample moves along them, so the fit lives on the rest.
    let range: Vec<usize> = (0..mm).filter(|&i| evals[i] > spectral_roundoff).collect();
    let rank = range.len();
    if rank == 0 {
        return Err("fit_tensor_surface: design is identically zero".to_string());
    }
    if rank >= n {
        return Err(format!(
            "fit_tensor_surface: the tensor basis spans every sample (rank {rank}, n={n}); \
             no residual degrees of freedom are left to weigh the surface against its noise"
        ));
    }
    let b = evecs.t().dot(&xty); // mm × D, rotated cross-products
    for d in 0..d_dims {
        let energy = responses.column(d).dot(&responses.column(d));
        if !(energy.is_finite() && energy > 0.0) {
            return Err(format!(
                "fit_tensor_surface: response {d} has non-positive energy {energy}; \
                 its profiled Gaussian scale has no finite REML optimum"
            ));
        }
    }
    // Least-squares coefficients on the design's range, C_ls = V Λ⁺ Vᵀ Xᵀy.
    let mut ls_rot = Array2::<f64>::zeros((mm, d_dims));
    for &i in &range {
        for d in 0..d_dims {
            ls_rot[[i, d]] = b[[i, d]] / evals[i];
        }
    }
    let ls_beta = evecs.dot(&ls_rot); // mm × D
    let ls_fitted = x.dot(&ls_beta); // n × D
    // B_d = ‖P y_d‖² and RSS_d = ‖y_d − P y_d‖², the latter from the residual itself
    // so a near-exact fit keeps its digits.
    let explained: Vec<f64> = (0..d_dims)
        .map(|d| {
            range
                .iter()
                .map(|&i| b[[i, d]] * b[[i, d]] / evals[i])
                .sum::<f64>()
        })
        .collect();
    let residual_energy: Vec<f64> = (0..d_dims)
        .map(|d| {
            (0..n)
                .map(|r| {
                    let e = responses[[r, d]] - ls_fitted[[r, d]];
                    e * e
                })
                .sum::<f64>()
        })
        .collect();
    for d in 0..d_dims {
        if !(explained[d].is_finite() && residual_energy[d].is_finite()) {
            return Err(format!(
                "fit_tensor_surface: response {d} sufficient statistics are not finite \
                 (explained {}, residual {})",
                explained[d], residual_energy[d]
            ));
        }
    }

    // REML weight. With w = λ/(1 + λ) ∈ [0, 1], B_d, RSS_d and r = rank X, the
    // profiled criterion to minimize is
    //   Σ_d n·log(RSS_d + w·B_d) − D·r·log w.
    // Its w-derivative has the sign of
    //   g(w) = Σ_d n·w·B_d/(RSS_d + w·B_d) − D·r,
    // a sum of increasing concave terms. So:
    // - g(1) ≤ 0: the optimum is the exact null w = 1 (λ = ∞, C = 0);
    // - g(0⁺) ≥ 0: the optimum is w = 0 (λ = 0). A response the basis interpolates
    //   exactly contributes n to g(0⁺), and the carve re-fit is one by construction;
    // - otherwise g has one root. With one response it is w = r·RSS/((n − r)·B).
    //   With several, Newton's method starts from
    //   w₀ = (D·r − n·#{RSS_d = 0 < B_d})/Σ_{RSS_d > 0} n·B_d/RSS_d, where g ≤ 0
    //   because each term is at most its tangent at 0. Concavity keeps every
    //   Newton iterate on that side, so w climbs monotonically to the root, and the
    //   iteration stops when a step no longer increases it.
    // The posterior mean is C = (1 − w)·C_ls and the scale-free covariance is
    // (1 − w)·(XᵀX)⁺. There is no search, grid or tolerance.
    let observations = n as f64;
    let rank_f = rank as f64;
    let responses_f = d_dims as f64;
    let derivative_sign = |w: f64| -> f64 {
        (0..d_dims)
            .map(|d| {
                if explained[d] == 0.0 {
                    0.0
                } else {
                    observations * w * explained[d] / (residual_energy[d] + w * explained[d])
                }
            })
            .sum::<f64>()
            - responses_f * rank_f
    };
    let interpolated = (0..d_dims)
        .filter(|&d| residual_energy[d] == 0.0 && explained[d] > 0.0)
        .count() as f64;
    let weight = if derivative_sign(1.0) <= 0.0 {
        1.0
    } else if observations * interpolated >= responses_f * rank_f {
        0.0
    } else if d_dims == 1 {
        rank_f * residual_energy[0] / ((observations - rank_f) * explained[0])
    } else {
        let tangent_sum: f64 = (0..d_dims)
            .filter(|&d| residual_energy[d] > 0.0)
            .map(|d| observations * explained[d] / residual_energy[d])
            .sum();
        let mut w = (responses_f * rank_f - observations * interpolated) / tangent_sum;
        loop {
            let value = derivative_sign(w);
            if !(value < 0.0) {
                break;
            }
            let slope: f64 = (0..d_dims)
                .map(|d| {
                    let denominator = residual_energy[d] + w * explained[d];
                    observations * explained[d] * residual_energy[d] / (denominator * denominator)
                })
                .sum();
            let next = w - value / slope;
            if !(next > w) {
                break;
            }
            w = next;
        }
        w
    };
    let shrinkage = 1.0 - weight;
    let lambda = if shrinkage > 0.0 {
        weight / shrinkage
    } else {
        f64::INFINITY
    };

    // Coefficients, EDF, residuals, covariances at the selected weight.
    let edf = shrinkage * rank_f;
    let residual_df = observations - edf;
    let beta_rot = ls_rot.mapv(|value| shrinkage * value);
    let beta = ls_beta.mapv(|value| shrinkage * value); // mm × D
    // #2822 — the coefficients' rounding band. The computed
    // β̂ = (1 − w)·V̂ Λ̂⁺ V̂ᵀ(Xᵀy) solves a perturbed system: the Gram accumulation, the
    // backward-stable eigensystem and the truncation of the null modes perturb XᵀX by E
    // with ‖E‖₂ ≤ γₙ·‖|X|ᵀ|X|‖_F + `spectral_roundoff`, Xᵀy and its rotation V̂ᵀXᵀy
    // carry their accumulation bands, the division is correctly rounded, and the final
    // rotation V̂·β_rot accumulates M₁M₂ terms. To first order the pseudo-inverse moves
    // by the solve on the range and by the rotation of the range itself, so per output
    //   ‖Δβ‖₂ ≤ (1 − w)·[‖E‖₂·‖β_ls‖₂·(1/d_min + 1/gap)
    //                    + (‖band(Xᵀy)‖₂ + ‖band(V̂ᵀXᵀy)‖₂)/d_min]
    //           + γ₂·‖β̂‖₂ + ‖band(V̂·β_rot)‖₂,
    // with d_min the smallest retained mode and gap its distance to the largest
    // dropped one (no rotation term when no mode is dropped). This bounds every entry.
    // The interaction coefficients of an exactly additive surface are zero only up to
    // this band: the rotations mix O(‖β̂‖) magnitudes into every entry and cancel them
    // there.
    let n_growth = gam_linalg::roundoff::accumulation_growth(n);
    let mm_growth = gam_linalg::roundoff::accumulation_growth(mm);
    let division_growth = gam_linalg::roundoff::accumulation_growth(2);
    let abs_x = x.mapv(f64::abs);
    let abs_evecs = evecs.mapv(f64::abs);
    let abs_gram = abs_x.t().dot(&abs_x);
    let operator_band =
        n_growth * abs_gram.iter().map(|v| v * v).sum::<f64>().sqrt() + spectral_roundoff;
    let d_min = range
        .iter()
        .map(|&i| evals[i])
        .fold(f64::INFINITY, f64::min);
    let rotation_conditioning = if rank < mm {
        let dropped_max = (0..mm)
            .filter(|&i| evals[i] <= spectral_roundoff)
            .map(|i| evals[i].max(0.0))
            .fold(0.0_f64, f64::max);
        1.0 / (d_min - dropped_max)
    } else {
        0.0
    };
    let abs_xty_sums = abs_x.t().dot(&responses.mapv(f64::abs)); // mm × D
    let abs_rotated_xty_sums = abs_evecs.t().dot(&xty.mapv(f64::abs)); // mm × D
    let abs_rotation_sums = abs_evecs.dot(&beta_rot.mapv(f64::abs)); // mm × D
    let column_norm = |matrix: &Array2<f64>, d: usize| -> f64 {
        matrix.column(d).iter().map(|v| v * v).sum::<f64>().sqrt()
    };
    let mut coeff_band = Vec::with_capacity(d_dims);
    for d in 0..d_dims {
        let beta_norm = column_norm(&beta, d);
        let solve_band = shrinkage
            * (operator_band * column_norm(&ls_beta, d) * (1.0 / d_min + rotation_conditioning)
                + (n_growth * column_norm(&abs_xty_sums, d)
                    + mm_growth * column_norm(&abs_rotated_xty_sums, d))
                    / d_min);
        let band = solve_band
            + division_growth * beta_norm
            + mm_growth * column_norm(&abs_rotation_sums, d);
        if !(band.is_finite() && band >= 0.0) {
            return Err(format!(
                "fit_tensor_surface: coefficient rounding band for response {d} is not finite ({band})"
            ));
        }
        coeff_band.push(band);
    }
    let fitted = ls_fitted.mapv(|value| shrinkage * value); // n × D
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
    // Scale-free (1 − w)·V Λ⁺ Vᵀ over the design's range.
    let mut scaled_evecs = Array2::<f64>::zeros((mm, mm));
    for &i in &range {
        let scale = shrinkage / evals[i];
        for row in 0..mm {
            scaled_evecs[[row, i]] = evecs[[row, i]] * scale;
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

    let interaction_test = InteractionTest::from_sample(phi_a, phi_b, responses)?;

    Ok(TensorSurfaceFit {
        coeffs,
        coeff_covariance,
        residual_cross_cov,
        unit_covariance,
        lambda,
        edf,
        residual_df,
        coeff_band,
        interaction_test,
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
    /// [`carve`]. The coefficient covariance, the joint covariance and the
    /// sample-space binding test all come from the REML re-fit.
    pub fn representational_carve_input(&self) -> CarveInput<'_> {
        CarveInput {
            phi_a: self.phi_a.view(),
            phi_b: self.phi_b.view(),
            coeffs: self.surface.coeffs.as_slice(),
            coeff_band: self.surface.coeff_band.as_slice(),
            coeff_covariance: Some(self.surface.coeff_covariance.as_slice()),
            joint_coeff_covariance: Some(&self.joint_covariance),
            interaction_test: Some(&self.surface.interaction_test),
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
/// same code measure yields the coefficients, their covariance and the
/// sample-space binding test. The reconstruction is an exact linear image of
/// the decoder, so the re-fit recovers the decoder's own ANOVA structure (the
/// representational binding question). It also leaves no residual variation
/// for a sampling test to weigh the interaction against, so the binding test
/// reports [`BindingTestUnavailable::NoResidualVariation`] there and the
/// carve's resolution dial decides.
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
    // coefficients, their covariance and the sample-space binding test.
    let surface = fit_tensor_surface(phi_a.view(), phi_b.view(), reconstruction.view())?;
    let joint_covariance = surface.joint_covariance();

    Ok(FittedAtomCarveInput {
        phi_a,
        phi_b,
        surface,
        joint_covariance,
    })
}

/// Inputs for one notion's carve over one fitted product atom.
///
/// `phi_a`/`phi_b`: factor bases evaluated on the code sample (`n × M_i`).
/// `coeffs`: per-output-dim coefficient matrices (`M₁ × M₂` each); for the
/// representational notion these are the decoder's, for the computational
/// notion they come from fitting the same tensor basis to the pulled-back
/// readout. `coeff_covariance`: matching scale-included posterior
/// covariance of the ROW-MAJOR vec of each `C` (`M₁M₂ × M₁M₂` per output
/// dim) — optional; it sets the posterior part of the carve's resolution
/// bound. `interaction_test`: the sample-space binding test of the
/// responses the coefficients were fitted to, on the same factor bases —
/// optional; without it no binding test runs and the carve reports
/// [`BindingTestUnavailable::NoStatistics`].
pub struct CarveInput<'a> {
    pub phi_a: ArrayView2<'a, f64>,
    pub phi_b: ArrayView2<'a, f64>,
    pub coeffs: &'a [Array2<f64>],
    /// Per output dim, the bound on the absolute rounding error of every entry of
    /// `coeffs[d]` (e.g. [`TensorSurfaceFit::coeff_band`]); `0.0` for coefficients
    /// handed over exactly. The numerical-additivity decision reads it.
    pub coeff_band: &'a [f64],
    pub coeff_covariance: Option<&'a [Array2<f64>]>,
    /// Covariance of the dimension-major STACKED coefficient vector
    /// `[vec(C₀); vec(C₁); …]` (`D·M₁M₂` square, scale-included), e.g.
    /// [`TensorSurfaceFit::joint_covariance`]. Its diagonal blocks stand in
    /// for `coeff_covariance` when that is absent.
    pub joint_coeff_covariance: Option<&'a Array2<f64>>,
    /// The sample-space binding test, e.g. [`TensorSurfaceFit::interaction_test`].
    pub interaction_test: Option<&'a InteractionTest>,
    pub notion: BindingNotion,
}

/// The carve: exact ANOVA split, interaction energy, sample-space binding
/// test, and the fission plan when this notion permits one.
///
/// Fission rule (asymmetric on purpose): the test REJECTING proves
/// binding and always blocks the split; the test NOT rejecting is only
/// absence of evidence, so the split additionally requires the discarded
/// interaction to be unresolved at the carve's own resolution (#2946).
///
/// - What the split discards: the parent is `g₀ + f₁ + f₂ + f₁₂` row by row on
///   any code measure, so the children lose exactly `Ê = Σ_d Σ_n f₁₂,d(θ_n)²`,
///   the #2946 R7 cross-block energy of the two-block partition. Dependent codes
///   leave this identity intact; they only stop `Ê` from being a variance share.
/// - Rounding: `R = Σ_n band_n²`, the numerical-additivity floor below.
/// - Estimation: with `x_n = vec(φ̃¹_n φ̃²_nᵀ)` (row-major), `f₁₂,d(θ_n) = x_nᵀ vec(C_d)`,
///   so the scale-included posterior covariance `Σ_d` gives the posterior error
///   energy `P = Σ_d tr(G Σ_d)`, `G = Σ_n x_n x_nᵀ`, carried with its own
///   accumulation band. `P` is covariant: the gauge directions annihilate every
///   `x_n`, `x` and `Σ` transform contragrediently under any reparameterization of
///   either factor basis, and it scales as the response squared.
///
/// On an exactly additive surface the computed interaction values are estimation
/// error plus rounding. Markov on the second moment bounds the estimation part's
/// energy by `P/α` with probability at least `1 − α`, and the triangle inequality on
/// the per-row values then gives the split's certificate `√Ê ≤ √R + √(P/α)`, at the
/// carve's own `α` — no literal. An additive surface splits with probability at
/// least `1 − 2α` at every sample size; a fixed interaction share grows `Ê` with `n`
/// while `P` stays `O(σ²·rank)`, so it is kept. Without a covariance `P` is zero and
/// only the rounding floor certifies, so a bare noisy estimate stays whole and
/// contested — route its [`CarveReport::edge_p_value`] into the evidence ledger and
/// let the probe loop earn the verdict.
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
    if input.coeff_band.len() != input.coeffs.len() {
        return Err(format!(
            "carve: {} coefficient matrices but {} coefficient rounding bands",
            input.coeffs.len(),
            input.coeff_band.len()
        ));
    }
    if let Some(band) = input
        .coeff_band
        .iter()
        .find(|band| !(band.is_finite() && **band >= 0.0))
    {
        return Err(format!(
            "carve: coefficient rounding band {band} is not finite and non-negative"
        ));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(format!("carve: alpha must be in (0,1), got {alpha}"));
    }
    let dims = input.coeffs.len();
    let (binding_tests, edge_test) = match input.interaction_test {
        None => (
            vec![Err(BindingTestUnavailable::NoStatistics); dims],
            Err(BindingTestUnavailable::NoStatistics),
        ),
        Some(test) => {
            if (test.n, test.m1, test.m2) != (n, m1, m2) || test.per_dimension.len() != dims {
                return Err(format!(
                    "carve: the interaction test was run on n={}, M₁={}, M₂={} with {} outputs, \
                     but the carve has n={n}, M₁={m1}, M₂={m2} with {dims}",
                    test.n,
                    test.m1,
                    test.m2,
                    test.per_dimension.len()
                ));
            }
            (test.per_dimension.clone(), test.edge)
        }
    };

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

    let mut child_a: Vec<ChildDecoder> = Vec::with_capacity(input.coeffs.len());
    let mut child_b: Vec<ChildDecoder> = Vec::with_capacity(input.coeffs.len());
    let mut interaction_energy = 0.0f64;
    // `Σ_n band_n²`: the energy an exactly additive surface can leave in the
    // computed interaction block (see the numerical-additivity decision below).
    let mut interaction_band_energy = 0.0f64;
    let mut centered_energy = 0.0f64;
    let products_growth = gam_linalg::roundoff::accumulation_growth(m1 * m2 + m1 + m2);
    let abs_row_sum = |basis: &Array2<f64>| -> Vec<f64> {
        basis
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|value| value.abs()).sum())
            .collect()
    };
    let abs_row_sum_a = abs_row_sum(&phi_a_c);
    let abs_row_sum_b = abs_row_sum(&phi_b_c);

    for (dim, c) in input.coeffs.iter().enumerate() {
        if c.dim() != (m1, m2) {
            return Err(format!(
                "carve: coefficient matrix {dim} is {:?}, bases say ({m1}, {m2})",
                c.dim()
            ));
        }
        let blocks = anova_blocks(c.view(), mean_a.view(), mean_b.view())?;
        let coefficient_band = input.coeff_band[dim];

        // Interaction values on the sample: f₁₂(θ_n) = φ̃¹_n ᵀ C φ̃²_n,
        // computed as the row-wise dot of (Φ̃₁ C) with Φ̃₂.
        let phi_a_c_c = phi_a_c.dot(c);
        // `Σ_jk |φ̃¹_j|·|C_jk|·|φ̃²_k|`, the absolute sum each `f₁₂` is accumulated
        // from: the rounding band of the products and the centering.
        let abs_phi_a_c_c = phi_a_c.mapv(f64::abs).dot(&c.mapv(f64::abs));
        let main_a_vals = phi_a_c.dot(&blocks.main_a);
        let main_b_vals = phi_b_c.dot(&blocks.main_b);
        for row in 0..n {
            let mut f12 = 0.0f64;
            let mut f12_abs = 0.0f64;
            for k in 0..m2 {
                f12 += phi_a_c_c[[row, k]] * phi_b_c[[row, k]];
                f12_abs += abs_phi_a_c_c[[row, k]] * phi_b_c[[row, k]].abs();
            }
            interaction_energy += f12 * f12;
            // |computed f₁₂ − exact f₁₂| ≤ γ_ops·Σ|φ̃¹||C||φ̃²| + band_C·Σ|φ̃¹_n|·Σ|φ̃²_n|:
            // the products' own rounding plus what the coefficients carry from the
            // solve that produced them.
            let row_band = products_growth * f12_abs
                + coefficient_band * abs_row_sum_a[row] * abs_row_sum_b[row];
            interaction_band_energy += row_band * row_band;
            let centered = main_a_vals[row] + main_b_vals[row] + f12;
            centered_energy += centered * centered;
        }

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
    // No test can prove the PRESENCE of an interaction in these coefficients
    // when their interaction energy is numerically indistinguishable from zero:
    // below the floor the surface is additive by construction, so no statistic
    // counts as binding and the atom is free to fission. The floor is
    // `Σ_n band_n²`, where `band_n` bounds how far the computed `f₁₂(θ_n)` of an
    // exactly additive surface can sit from zero: the rounding of the products and the centering, `γ_ops·Σ_jk |φ̃¹_j||C_jk||φ̃²_k|`,
    // plus what each coefficient carries from its own solve (`coeff_band`). A
    // fitted surface's interaction coefficients cancel O(‖C‖) magnitudes, so the
    // products' band alone, denominated in those roundoff-sized coefficients,
    // sat a whole solve backward error below the floor (#2822).
    let numerically_additive = interaction_energy <= interaction_band_energy;
    let binding_proven =
        !numerically_additive && edge_test.is_ok_and(|test| test.p_value() <= alpha);
    // The posterior error energy of the interaction values (fission rule above):
    // `G = Σ_n x_n x_nᵀ` with `x_n = vec(φ̃¹_n φ̃²_nᵀ)` row-major and
    // `P = Σ_d Σ_ab G_ab Σ_d[a,b]`. Its band is γ over the accumulation depth (the
    // centered factors, their products, `n − 1` row additions and `D·(M₁M₂)² − 1`
    // contractions) times the absolute shadow `Σ_d Σ_ab |G|_ab·|Σ_d[a,b]|`, to first
    // order in `u`.
    let interaction_width = m1 * m2;
    let posterior_blocks: Option<Vec<ArrayView2<'_, f64>>> =
        match (input.coeff_covariance, input.joint_coeff_covariance) {
            (Some(covs), _) => {
                if let Some(dim) = covs
                    .iter()
                    .position(|cov| cov.dim() != (interaction_width, interaction_width))
                {
                    return Err(format!(
                        "carve: coefficient covariance {dim} is {:?}; the tensor block needs \
                         {interaction_width}×{interaction_width}",
                        covs[dim].dim()
                    ));
                }
                Some(covs.iter().map(|cov| cov.view()).collect())
            }
            (None, Some(joint)) => {
                let total = dims * interaction_width;
                if joint.dim() != (total, total) {
                    return Err(format!(
                        "carve: joint coefficient covariance is {:?}; {dims} outputs need \
                         {total}×{total}",
                        joint.dim()
                    ));
                }
                Some(
                    (0..dims)
                        .map(|dim| {
                            let start = dim * interaction_width;
                            let end = start + interaction_width;
                            joint.slice(s![start..end, start..end])
                        })
                        .collect(),
                )
            }
            (None, None) => None,
        };
    let (posterior_energy, posterior_band) = match posterior_blocks {
        None => (0.0, 0.0),
        Some(blocks) => {
            let mut gram = Array2::<f64>::zeros((interaction_width, interaction_width));
            let mut gram_shadow = Array2::<f64>::zeros((interaction_width, interaction_width));
            let mut products = vec![0.0f64; interaction_width];
            for row in 0..n {
                for j in 0..m1 {
                    for k in 0..m2 {
                        products[j * m2 + k] = phi_a_c[[row, j]] * phi_b_c[[row, k]];
                    }
                }
                for a in 0..interaction_width {
                    for b in 0..interaction_width {
                        let term = products[a] * products[b];
                        gram[[a, b]] += term;
                        gram_shadow[[a, b]] += term.abs();
                    }
                }
            }
            let mut energy = 0.0f64;
            let mut shadow = 0.0f64;
            for block in &blocks {
                for a in 0..interaction_width {
                    for b in 0..interaction_width {
                        energy += gram[[a, b]] * block[[a, b]];
                        shadow += gram_shadow[[a, b]] * block[[a, b]].abs();
                    }
                }
            }
            let operations = n + blocks.len() * interaction_width * interaction_width + 6;
            (
                energy,
                gam_linalg::roundoff::accumulation_growth(operations) * shadow,
            )
        }
    };
    if !(posterior_energy.is_finite() && posterior_band.is_finite())
        || posterior_energy < -posterior_band
    {
        return Err(format!(
            "carve: the coefficient covariance gives the interaction values a posterior error \
             energy of {posterior_energy} (band {posterior_band}); a covariance is positive \
             semidefinite"
        ));
    }
    let resolution_band =
        interaction_band_energy.sqrt() + ((posterior_energy + posterior_band) / alpha).sqrt();
    let negligible = interaction_energy.sqrt() <= resolution_band;
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
        edge_test,
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

/// `σ_max` of `matrix`.
fn largest_singular_value(matrix: ArrayView2<'_, f64>, context: &str) -> Result<f64, String> {
    let (_, singular, _) = matrix
        .svd(false, false)
        .map_err(|e| format!("InteractionTest: {context} SVD failed: {e:?}"))?;
    Ok(singular.iter().fold(0.0_f64, |acc, &value| acc.max(value)))
}

/// Orthonormal basis of `matrix`'s numerical range: its left singular vectors
/// whose singular values exceed `band`.
fn numerical_range(
    matrix: ArrayView2<'_, f64>,
    band: f64,
    context: &str,
) -> Result<Array2<f64>, String> {
    let (left, singular, _) = matrix
        .svd(true, false)
        .map_err(|e| format!("InteractionTest: {context} SVD failed: {e:?}"))?;
    let left = left.ok_or_else(|| format!("InteractionTest: {context} SVD returned no U"))?;
    let kept: Vec<usize> = (0..singular.len())
        .filter(|&i| singular[i] > band)
        .collect();
    Ok(left.select(Axis(1), &kept))
}

/// `(I − U Uᵀ) target` for orthonormal `U`, applied twice: the second pass
/// removes what the first leaves along `U` from rounding, so the result is
/// orthogonal to `U` to working precision.
fn project_out(basis: &Array2<f64>, target: Array2<f64>) -> Array2<f64> {
    if basis.ncols() == 0 {
        return target;
    }
    let once = &target - &basis.dot(&basis.t().dot(&target));
    &once - &basis.dot(&basis.t().dot(&once))
}

fn column_energy(column: ArrayView1<'_, f64>) -> f64 {
    column.dot(&column)
}

/// Wilks' test from the full-space residual `E` (`n × D`) and interaction
/// scores `S` (`k × D`). `Λ = |EᵀE| / |EᵀE + SᵀS|`; with `E = U Σ Vᵀ`,
/// `−ln Λ = Σ_i ln(1 + σ_i(A)²)` for `A = Σ⁻¹ Vᵀ Sᵀ`, which never forms a
/// determinant or a cross-product.
fn wilks_test(
    residual: &Array2<f64>,
    scores: &Array2<f64>,
    residual_df: usize,
) -> Result<Result<EdgeBindingTest, BindingTestUnavailable>, String> {
    let (n, dims) = residual.dim();
    let interaction_rank = scores.nrows();
    let (_, residual_singular, right) = residual
        .svd(false, true)
        .map_err(|e| format!("InteractionTest: residual SVD failed: {e:?}"))?;
    let right = right.ok_or_else(|| "InteractionTest: residual SVD returned no Vᵀ".to_string())?;
    let sigma_max = residual_singular
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value));
    let band = factor_singular_band(n, dims, sigma_max);
    if residual_singular.len() < dims || residual_singular.iter().any(|&value| value <= band) {
        return Ok(Err(
            BindingTestUnavailable::ResidualCovarianceRankDeficient { residual_df, dims },
        ));
    }
    let mut whitened = right.dot(&scores.t()); // D × k
    for (mut row, &sigma) in whitened
        .rows_mut()
        .into_iter()
        .zip(residual_singular.iter())
    {
        row.mapv_inplace(|value| value / sigma);
    }
    let (_, canonical, _) = whitened
        .svd(false, false)
        .map_err(|e| format!("InteractionTest: canonical SVD failed: {e:?}"))?;
    let neg_log_lambda: f64 = canonical.iter().map(|&value| (value * value).ln_1p()).sum();
    let p_value = wilks_survival(neg_log_lambda, dims, interaction_rank, residual_df)?;
    Ok(Ok(EdgeBindingTest::Wilks(WilksTest {
        neg_log_lambda,
        dims,
        interaction_rank,
        residual_df,
        p_value,
    })))
}

/// `P(−ln Λ > t)` for `Λ ~ Λ(D, ν, k) = Π_{i=1}^{D} Beta((ν−i+1)/2, k/2)`
/// (independent factors), `ν ≥ D`.
///
/// Four parameter lines reduce to an F law exactly (Rao): `k = 1` and `D = 1`
/// through `1/Λ − 1`, `k = 2` and `D = 2` through `1/√Λ − 1` (the `k` lines by
/// the symmetry `Λ(D, ν, k) = Λ(k, ν + k − D, D)`). Otherwise `−ln Λ` is a sum
/// of independent exponentials whenever `k` or `D` is even, evaluated by
/// [`hypoexponential_survival`], and that sum plus one `−ln Beta(c, ½)` when
/// both are odd ([`WilksLaw`]), evaluated by
/// [`hypoexponential_log_beta_half_survival`].
fn wilks_survival(t: f64, dims: usize, k: usize, nu: usize) -> Result<f64, String> {
    let (d, kf, nuf) = (dims as f64, k as f64, nu as f64);
    if k == 1 {
        return Ok(fisher_snedecor_sf(
            t.exp_m1() * (nuf - d + 1.0) / d,
            d,
            nuf - d + 1.0,
        ));
    }
    if dims == 1 {
        return Ok(fisher_snedecor_sf(t.exp_m1() * nuf / kf, kf, nuf));
    }
    if k == 2 {
        return Ok(fisher_snedecor_sf(
            (0.5 * t).exp_m1() * (nuf - d + 1.0) / d,
            2.0 * d,
            2.0 * (nuf - d + 1.0),
        ));
    }
    if dims == 2 {
        return Ok(fisher_snedecor_sf(
            (0.5 * t).exp_m1() * (nuf - 1.0) / kf,
            2.0 * kf,
            2.0 * (nuf - 1.0),
        ));
    }
    let law = WilksLaw::new(dims, k, nu);
    match law.half_beta_shape {
        None => Ok(hypoexponential_survival(&law.rates, t)),
        Some(shape) => hypoexponential_log_beta_half_survival(&law.rates, shape, t),
    }
}

/// The null law of `−ln Λ(D, ν, k)` as an independent sum `H + L`: `H` a sum
/// of exponentials with `rates`, and `L = −ln Beta(c, ½)` when
/// `half_beta_shape = Some(c)`, which is exactly when `D` and `k` are both
/// odd. With `a_i = (ν − i + 1)/2`:
///
/// - `k` even: `−ln Beta(a, m) = Σ_{j<m} Exp(a + j)` for integer `m` (the
///   Mellin transforms agree: `E[Bˢ] = Π_{j<m} (a + j)/(a + j + s)`), so the
///   rates are `a_i + j`, `i = 1..D`, `j < k/2`.
/// - `k` odd: consecutive factors pair by Legendre duplication,
///   `Beta(α + ½, β)·Beta(α, β) = Beta(2α, 2β)²` in law, so with `α = a_{2l}`,
///   `β = k/2` each pair is `2·(−ln Beta(2a_{2l}, k)) = Σ_{j<k} Exp(a_{2l} + j/2)`.
///   An odd `D` leaves `Beta(a_D, m + ½)`, `m = (k − 1)/2`, unpaired; the
///   product rule `Beta(α, β)·Beta(α + β, γ) = Beta(α, β + γ)` in law splits
///   it into `Beta(a_D, m)`, rates `a_D + j` for `j < m`, and `Beta(a_D + m, ½)`.
struct WilksLaw {
    rates: Vec<f64>,
    half_beta_shape: Option<f64>,
}

impl WilksLaw {
    fn new(dims: usize, k: usize, nu: usize) -> Self {
        let a = |i: usize| (nu as f64 - i as f64 + 1.0) / 2.0;
        if k % 2 == 0 {
            return Self {
                rates: (1..=dims)
                    .flat_map(|i| (0..k / 2).map(move |j| a(i) + j as f64))
                    .collect(),
                half_beta_shape: None,
            };
        }
        let mut rates: Vec<f64> = (1..=dims / 2)
            .flat_map(|l| (0..k).map(move |j| a(2 * l) + 0.5 * j as f64))
            .collect();
        if dims % 2 == 0 {
            return Self {
                rates,
                half_beta_shape: None,
            };
        }
        let half_rank = (k - 1) / 2;
        rates.extend((0..half_rank).map(|j| a(dims) + j as f64));
        Self {
            rates,
            half_beta_shape: Some(a(dims) + half_rank as f64),
        }
    }
}

/// Gauss–Legendre order of each panel rule in
/// [`hypoexponential_log_beta_half_survival`].
const CONVOLUTION_PANEL_ORDER: usize = 16;

/// Panel count at which [`hypoexponential_log_beta_half_survival`] refuses
/// instead of returning an unconverged value.
const CONVOLUTION_MAX_PANELS: usize = 1 << 12;

/// One panel `[left, right]` of the convolution integral, priced by the panel
/// rule on the panel and on its two halves. The halves' sum is kept; the
/// difference is its indicator.
struct ConvolutionPanel {
    left: f64,
    right: f64,
    mass: f64,
    gap: f64,
}

/// `P(H + L > t)` for independent `H = Σ_i Exp(rates[i])` (`rates` not empty)
/// and `L = −ln B`, `B ~ Beta(c, ½)`.
///
/// Conditioning on `L`: `P(H + L > t) = P(L > t) + E[S_H(t − L); L ≤ t]`,
/// with `P(L > t) = I_{e^{−t}}(c, ½)` and `S_H` the survival
/// [`hypoexponential_survival`] evaluates. `L` has density
/// `e^{−cs}(1 − e^{−s})^{−½}/B(c, ½)`, whose `s^{−½}` endpoint the substitution
/// `s = r²` removes:
///
/// ```text
/// E[S_H(t − L); L ≤ t] = (2/B(c, ½)) ∫_0^{√t} e^{−c r²} (r²/(1 − e^{−r²}))^{½} S_H(t − r²) dr.
/// ```
///
/// `S_H` is entire (exponentials times polynomials) and `r²/(1 − e^{−r²})` is
/// analytic on the real line, so the integrand is analytic on the closed
/// interval, and it is positive, so the quadrature sums without cancellation.
/// Every factor is a combination of `e^{±λ r²}` with `λ` at most `q`, the
/// largest of `c` and the rates, so the first panels are at most `1/√q` wide and
/// already resolve each factor at `r = 0`. Each panel carries the 16-point
/// rule on its two halves, with the difference from the whole-panel rule as
/// its indicator. For an integrand analytic around a panel, halving the panel
/// scales the rule's error by roughly `2^{−32}`, so indicators summing below
/// `√ε` of the survival leave an error far below `ε` of it. The panel with the
/// largest indicator is bisected until then; exhausting
/// [`CONVOLUTION_MAX_PANELS`] is a refusal, not a value. When the Chernoff
/// bound `e^{−θt}·Π r_i/(r_i − θ)·B(c − θ, ½)/B(c, ½)`, at
/// `θ = max(0, ρ − (R + 1)/t)` with `ρ` the smallest of `c` and the rates and
/// `R` the rate count, underflows, the survival is below every positive double
/// and is 0.
fn hypoexponential_log_beta_half_survival(
    rates: &[f64],
    shape: f64,
    t: f64,
) -> Result<f64, String> {
    if !(t > 0.0) {
        return Ok(1.0);
    }
    if t == f64::INFINITY {
        return Ok(0.0);
    }
    let log_beta = ln_beta(shape, 0.5);
    let floor = rates.iter().fold(shape, |acc, &r| acc.min(r));
    let theta = (floor - (rates.len() as f64 + 1.0) / t).max(0.0);
    let log_bound = -theta * t - rates.iter().map(|&r| (-theta / r).ln_1p()).sum::<f64>()
        + ln_beta(shape - theta, 0.5)
        - log_beta;
    if log_bound.exp() == 0.0 {
        return Ok(0.0);
    }
    let tail = ln_regularized_beta_lower_from_log_x(-t, shape, 0.5).exp();
    let log_scale = std::f64::consts::LN_2 - log_beta;
    let integrand = |r: f64| {
        let s = r * r;
        let jacobian = if s == 0.0 {
            1.0
        } else {
            (-s / (-s).exp_m1()).sqrt()
        };
        (log_scale - shape * s).exp() * jacobian * hypoexponential_survival(rates, t - s)
    };
    let (nodes, weights) = gauss_legendre(CONVOLUTION_PANEL_ORDER);
    let rule = |left: f64, right: f64| {
        let centre = 0.5 * (left + right);
        let half = 0.5 * (right - left);
        half * nodes
            .iter()
            .zip(&weights)
            .map(|(node, weight)| weight * integrand(centre + half * node))
            .sum::<f64>()
    };
    let priced = |left: f64, right: f64| -> Result<ConvolutionPanel, String> {
        let middle = 0.5 * (left + right);
        if !(middle > left && middle < right) {
            return Err(format!(
                "Wilks survival: convolution panel [{left}, {right}] cannot be halved at \
                 t={t}, c={shape}"
            ));
        }
        let mass = rule(left, middle) + rule(middle, right);
        Ok(ConvolutionPanel {
            left,
            right,
            mass,
            gap: (rule(left, right) - mass).abs(),
        })
    };
    let upper = t.sqrt();
    let fastest = rates.iter().fold(shape, |acc, &r| acc.max(r));
    // Equal panels no wider than `1/√q`, the last ending exactly at `√t`.
    let count = (upper * fastest.sqrt()).ceil().max(1.0) as usize;
    let cut = |i: usize| upper * i as f64 / count as f64;
    let mut panels = (0..count)
        .map(|i| priced(cut(i), cut(i + 1)))
        .collect::<Result<Vec<_>, _>>()?;
    loop {
        let survival = tail + panels.iter().map(|panel| panel.mass).sum::<f64>();
        let gap: f64 = panels.iter().map(|panel| panel.gap).sum();
        if !survival.is_finite() {
            return Err(format!(
                "Wilks survival: non-finite convolution {survival} at t={t}, c={shape}"
            ));
        }
        if gap <= f64::EPSILON.sqrt() * survival {
            return Ok(survival);
        }
        if panels.len() >= CONVOLUTION_MAX_PANELS {
            return Err(format!(
                "Wilks survival did not converge within {CONVOLUTION_MAX_PANELS} panels at \
                 t={t}, c={shape}: relative indicator {:.3e}",
                gap / survival
            ));
        }
        let mut worst = 0;
        for (index, panel) in panels.iter().enumerate() {
            if panel.gap > panels[worst].gap {
                worst = index;
            }
        }
        let panel = panels.swap_remove(worst);
        let middle = 0.5 * (panel.left + panel.right);
        panels.push(priced(panel.left, middle)?);
        panels.push(priced(middle, panel.right)?);
    }
}

/// `P(Σ_i X_i > t)` for independent `X_i ~ Exp(rates[i])`, by uniformization:
/// the sum is the absorption time of the chain that leaves phase `i` at rate
/// `rates[i]`, so with `q = max rate` the survival is
/// `Σ_s Poisson(s; qt)·m_s`, `m_s` the chance that `s` jumps of the discrete
/// chain (stay in `i` w.p. `1 − r_i/q`, advance w.p. `r_i/q`) leave it
/// unabsorbed. Every term is positive, so the sum has no cancellation, and
/// it is carried in logarithms so no term underflows before it matters. `m_s`
/// is non-increasing, so once `s + 2 > qt` the unsummed tail is at most
/// `m_s·w_{s+1}/(1 − qt/(s + 2))` (a geometric bound on the Poisson weights
/// `w`); the sum stops when that is below one rounding unit of the total.
/// When the Chernoff bound `e^{−θt}·Π r_i/(r_i − θ)`, at `θ = max(0, r_min −
/// R/t)`, underflows, the survival is below every positive double and is 0.
fn hypoexponential_survival(rates: &[f64], t: f64) -> f64 {
    if !(t > 0.0) {
        return 1.0;
    }
    if t == f64::INFINITY {
        return 0.0;
    }
    let count = rates.len() as f64;
    let r_min = rates.iter().fold(f64::INFINITY, |acc, &r| acc.min(r));
    let q = rates.iter().fold(0.0_f64, |acc, &r| acc.max(r));
    let theta = (r_min - count / t).max(0.0);
    let log_bound = -theta * t - rates.iter().map(|&r| (-theta / r).ln_1p()).sum::<f64>();
    if log_bound.exp() == 0.0 {
        return 0.0;
    }
    let qt = q * t;
    let ln_qt = qt.ln();
    let stay: Vec<f64> = rates.iter().map(|&r| 1.0 - r / q).collect();
    let advance: Vec<f64> = rates.iter().map(|&r| r / q).collect();
    let mut phase = vec![0.0_f64; rates.len()];
    phase[0] = 1.0;
    let mut next = vec![0.0_f64; rates.len()];
    let mut log_mass = 0.0_f64;
    let mut log_weight = -qt;
    let mut log_sum = -qt;
    let mut step = 0usize;
    loop {
        let s = step as f64;
        if s + 2.0 > qt {
            let log_tail =
                log_mass + log_weight + ln_qt - (s + 1.0).ln() - (-qt / (s + 2.0)).ln_1p();
            if log_tail <= f64::EPSILON.ln() + log_sum {
                break;
            }
        }
        next[0] = phase[0] * stay[0];
        for i in 1..rates.len() {
            next[i] = phase[i] * stay[i] + phase[i - 1] * advance[i - 1];
        }
        let total: f64 = next.iter().sum();
        if total == 0.0 {
            break;
        }
        log_mass += total.ln();
        for (slot, &value) in phase.iter_mut().zip(next.iter()) {
            *slot = value / total;
        }
        step += 1;
        log_weight += ln_qt - (step as f64).ln();
        log_sum = log_add(log_sum, log_weight + log_mass);
    }
    log_sum.exp()
}

/// `ln(eᵃ + eᵇ)`.
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

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

    /// Coefficients alone carry no sampling evidence: a planted BOUND matrix
    /// is kept whole with the edge test reported as not run, and the planted
    /// additive matrix under the same covariance still splits, its interaction
    /// being inside the resolution bound.
    #[test]
    fn planted_bound_coefficients_refuse_fission_without_a_test() {
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
        let cov = Array2::<f64>::eye(9) * 1e-4;
        let input = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &[c],
            coeff_band: &[0.0],
            coeff_covariance: Some(std::slice::from_ref(&cov)),
            joint_coeff_covariance: None,
            interaction_test: None,
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        assert!(report.fission.is_none(), "bound surface must not fission");
        assert!(report.interaction_fraction > 0.1);
        assert_eq!(
            report.edge_p_value(),
            Err(BindingTestUnavailable::NoStatistics)
        );
        assert_eq!(
            report.binding_tests,
            vec![Err(BindingTestUnavailable::NoStatistics)]
        );

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
            coeff_band: &[0.0],
            coeff_covariance: Some(std::slice::from_ref(&cov)),
            joint_coeff_covariance: None,
            interaction_test: None,
            notion: BindingNotion::Representational,
        };
        let report_add = carve(&input_add, 0.05).expect("carve");
        assert!(report_add.fission.is_some());
    }

    /// The gauge directions (`u vᵀ + w uᵀ`) carry no interaction: on a PoU
    /// basis they ARE `f₁ + f₂`, so the centered interaction values vanish.
    #[test]
    fn gauge_directions_carry_no_interaction() {
        let phi_a = pou_basis();
        let phi_b = pou_basis_b();
        let mut c = Array2::<f64>::zeros((3, 3));
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
            coeff_band: &[0.0],
            coeff_covariance: Some(std::slice::from_ref(&cov)),
            joint_coeff_covariance: None,
            interaction_test: None,
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        assert!(report.interaction_fraction < 1e-24);
        assert!(report.fission.is_some());
    }

    /// A deterministic Bernstein (degree-2, partition-of-unity) basis
    /// evaluated on `n` scattered points, with two decorrelated sample
    /// mappings so the tensor design is well-conditioned.
    fn bernstein_pair(n: usize) -> (Array2<f64>, Array2<f64>) {
        bernstein_factors(n, 2, 2)
    }

    /// Bernstein bases of degrees `degree_a` and `degree_b` (partition of
    /// unity, `degree + 1` columns) on the two decorrelated sample mappings of
    /// [`bernstein_pair`].
    fn bernstein_factors(n: usize, degree_a: usize, degree_b: usize) -> (Array2<f64>, Array2<f64>) {
        let bernstein = |degree: usize, x: f64| -> Vec<f64> {
            (0..=degree)
                .map(|j| {
                    let binomial =
                        (0..j).fold(1.0, |acc, i| acc * (degree - i) as f64 / (i + 1) as f64);
                    binomial * x.powi(j as i32) * (1.0 - x).powi((degree - j) as i32)
                })
                .collect()
        };
        let mut phi_a = Array2::<f64>::zeros((n, degree_a + 1));
        let mut phi_b = Array2::<f64>::zeros((n, degree_b + 1));
        for t in 0..n {
            let x = t as f64 / (n - 1) as f64;
            let z = ((t * 17) % n) as f64 / (n - 1) as f64;
            for (j, value) in bernstein(degree_a, x).into_iter().enumerate() {
                phi_a[[t, j]] = value;
            }
            for (k, value) in bernstein(degree_b, z).into_iter().enumerate() {
                phi_b[[t, k]] = value;
            }
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

    fn standard_normal(rng: &mut StdRng) -> f64 {
        let u1 = 1.0 - rng.random::<f64>();
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Kolmogorov–Smirnov `p` of `p_values` against `U(0, 1)`: the asymptotic
    /// Kolmogorov tail at Stephens' finite-sample-corrected statistic.
    fn kolmogorov_smirnov_uniform_p_value(p_values: &[f64]) -> f64 {
        let mut sorted = p_values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let count = sorted.len() as f64;
        let distance = sorted
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let above = (index as f64 + 1.0) / count - value;
                let below = value - index as f64 / count;
                above.max(below)
            })
            .fold(0.0_f64, f64::max);
        let root = count.sqrt();
        let scaled = distance * (root + 0.12 + 0.11 / root);
        // `Q(t) = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²t²}`, summed until a term no longer
        // moves the partial sum.
        let mut tail = 0.0_f64;
        let mut k = 1.0_f64;
        loop {
            let term = 2.0 * (-2.0 * k * k * scaled * scaled).exp();
            let signed = if (k as u64) % 2 == 1 { term } else { -term };
            if tail + signed == tail {
                break;
            }
            tail += signed;
            k += 1.0;
        }
        tail.clamp(0.0, 1.0)
    }

    /// Hold null p-values to `U(0, 1)` in BOTH directions: a two-sided
    /// Kolmogorov–Smirnov test over the whole range, and the empirical size at
    /// each conventional level within 3.5 Monte Carlo standard errors of that
    /// level. A conservative p-value — sizes below nominal, mass piled near
    /// `p = 1` — fails here exactly as an anti-conservative one does.
    fn assert_uniform(p_values: &[f64], label: &str) {
        assert!(
            p_values.iter().all(|value| (0.0..=1.0).contains(value)),
            "{label}: p-values outside [0, 1]"
        );
        let count = p_values.len() as f64;
        let ks = kolmogorov_smirnov_uniform_p_value(p_values);
        assert!(
            ks > 1e-3,
            "{label}: KS p = {ks:.3e} against U(0,1) over {count} replicates"
        );
        for level in [0.01, 0.05, 0.10, 0.20] {
            let size = p_values.iter().filter(|&&value| value <= level).count() as f64 / count;
            let standard_error = (level * (1.0 - level) / count).sqrt();
            assert!(
                (size - level).abs() <= 3.5 * standard_error,
                "{label}: size {size:.4} at level {level} is {:+.1} Monte Carlo SE from nominal",
                (size - level) / standard_error
            );
        }
    }

    /// Seeded null calibration of the per-dimension F test and of the edge
    /// test (#3182). Responses are an additive surface plus rows of
    /// CORRELATED Gaussian noise (`N(0, LLᵀ)`), so the null is true and the
    /// edge statistic has to handle the cross-dimension dependence exactly.
    /// Each configuration exercises one branch of the edge's null law, and
    /// each asserts the configuration's rank so the branch is the one named.
    fn assert_null_calibration(
        degree_a: usize,
        degree_b: usize,
        dims: usize,
        expected_rank: usize,
        seed: u64,
        label: &str,
    ) {
        let n = 40usize;
        let replicates = 2000usize;
        let (phi_a, phi_b) = bernstein_factors(n, degree_a, degree_b);
        let (m1, m2) = (degree_a + 1, degree_b + 1);
        let mut rng = StdRng::seed_from_u64(seed);
        // Additive mean: c[j,k] = a_d[j] + b_d[k].
        let mut mean = Array2::<f64>::zeros((n, dims));
        for d in 0..dims {
            let mut c = Array2::<f64>::zeros((m1, m2));
            for j in 0..m1 {
                for k in 0..m2 {
                    c[[j, k]] = (1.0 + d as f64) * (j as f64 - 0.7) + (0.5 - k as f64) * 0.3;
                }
            }
            mean.column_mut(d)
                .assign(&surface_values(&phi_a, &phi_b, &c));
        }
        // Lower-triangular noise factor with unit diagonal and 0.6 couplings.
        let mut factor = Array2::<f64>::eye(dims);
        for d in 0..dims {
            for e in 0..d {
                factor[[d, e]] = 0.6;
            }
        }
        let mut edge_p = Vec::with_capacity(replicates);
        let mut first_dim_p = Vec::with_capacity(replicates);
        for _ in 0..replicates {
            let mut noise = Array2::<f64>::zeros((n, dims));
            noise.mapv_inplace(|_| standard_normal(&mut rng));
            let responses = &mean + &noise.dot(&factor.t());
            let test = InteractionTest::from_sample(phi_a.view(), phi_b.view(), responses.view())
                .expect("interaction test");
            assert_eq!(
                test.interaction_rank, expected_rank,
                "{label}: interaction rank"
            );
            assert_eq!(test.residual_df, n - m1 * m2, "{label}: residual d.f.");
            first_dim_p.push(test.per_dimension[0].expect("per-dimension test").p_value);
            edge_p.push(test.edge.expect("edge test").p_value());
        }
        assert_uniform(&first_dim_p, &format!("{label}, per-dimension F"));
        assert_uniform(&edge_p, &format!("{label}, edge"));
    }

    /// `k = 4`, `D = 3`: the even-`k` exponential decomposition of Wilks' `Λ`.
    #[test]
    fn interaction_test_is_calibrated_even_rank() {
        assert_null_calibration(2, 2, 3, 4, 0x3182_0001, "k=4, D=3");
    }

    /// `k = 3`, `D = 4`: the duplication pairing of an even number of outputs.
    #[test]
    fn interaction_test_is_calibrated_even_outputs_odd_rank() {
        assert_null_calibration(1, 3, 4, 3, 0x3182_0002, "k=3, D=4");
    }

    /// `k = 4`, `D = 2`: Rao's exact F transform.
    #[test]
    fn interaction_test_is_calibrated_two_outputs() {
        assert_null_calibration(2, 2, 2, 4, 0x3182_0003, "k=4, D=2");
    }

    /// `k = 1`, `D = 3`: Hotelling's exact F transform.
    #[test]
    fn interaction_test_is_calibrated_rank_one() {
        assert_null_calibration(1, 1, 3, 1, 0x3182_0004, "k=1, D=3");
    }

    /// `k = 3`, `D = 3` (#3423): one pair plus the unpaired half-integer Beta
    /// factor, through the certified convolution.
    #[test]
    fn interaction_test_is_calibrated_odd_outputs_odd_rank() {
        assert_null_calibration(1, 3, 3, 3, 0x3423_0001, "k=3, D=3");
    }

    /// `k = 3`, `D = 5` (#3423): two pairs plus the unpaired factor.
    #[test]
    fn interaction_test_is_calibrated_five_outputs_odd_rank() {
        assert_null_calibration(1, 3, 5, 3, 0x3423_0002, "k=3, D=5");
    }

    /// The test reads only `span W` and `span F`: mixing either factor basis by
    /// an invertible map changes neither the rank nor any statistic.
    #[test]
    fn interaction_test_is_invariant_to_factor_reparameterization() {
        let n = 40usize;
        let (phi_a, phi_b) = bernstein_pair(n);
        let mut c = Array2::<f64>::zeros((3, 3));
        for j in 0..3 {
            for k in 0..3 {
                c[[j, k]] = (j as f64 - 1.0) * (k as f64 - 0.5) + 0.3 * j as f64;
            }
        }
        let y = surface_values(&phi_a, &phi_b, &c);
        let mut responses = Array2::<f64>::zeros((n, 2));
        for t in 0..n {
            responses[[t, 0]] = y[t] + 0.2 * (1.3 * t as f64).sin();
            responses[[t, 1]] = -y[t] + 0.2 * (2.1 * t as f64).cos();
        }
        let mix_a = array![[1.0, 0.5, -0.2], [0.0, 2.0, 0.3], [0.4, 0.0, 1.5]];
        let mix_b = array![[0.7, 0.0, 0.0], [1.1, 1.0, 0.0], [-0.3, 0.2, 3.0]];
        let original = InteractionTest::from_sample(phi_a.view(), phi_b.view(), responses.view())
            .expect("test");
        let mixed = InteractionTest::from_sample(
            phi_a.dot(&mix_a).view(),
            phi_b.dot(&mix_b).view(),
            responses.view(),
        )
        .expect("test");
        assert_eq!(original.interaction_rank, 4);
        assert_eq!(mixed.interaction_rank, 4);
        assert_eq!(original.residual_df, mixed.residual_df);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs());
        for (lhs, rhs) in original
            .per_dimension
            .iter()
            .zip(mixed.per_dimension.iter())
        {
            let (lhs, rhs) = (lhs.expect("test"), rhs.expect("test"));
            assert!(close(lhs.statistic, rhs.statistic), "{lhs:?} vs {rhs:?}");
            assert!(close(lhs.p_value, rhs.p_value), "{lhs:?} vs {rhs:?}");
        }
        let (lhs, rhs) = (original.edge.expect("edge"), mixed.edge.expect("edge"));
        assert!(close(lhs.p_value(), rhs.p_value()), "{lhs:?} vs {rhs:?}");
    }

    /// The rates of a Wilks law that is a pure sum of exponentials.
    fn exponential_rates(dims: usize, k: usize, nu: usize) -> Vec<f64> {
        let law = WilksLaw::new(dims, k, nu);
        assert!(
            law.half_beta_shape.is_none(),
            "Λ({dims}, {nu}, {k}) is not a pure exponential sum"
        );
        law.rates
    }

    /// `hypoexponential_survival` against closed forms. Each exponential law
    /// below equals a Beta or F law exactly, so the two evaluations may differ
    /// only by rounding: the uniformization sums at most a few hundred positive
    /// terms, each a product of as many correctly rounded factors, and the
    /// regularized incomplete beta is accurate to a few ulps, so `1e-11` holds
    /// with two orders of margin over `steps·ε`.
    #[test]
    fn hypoexponential_survival_matches_closed_forms() {
        let close = |a: f64, b: f64, label: &str| {
            assert!(
                (a - b).abs() <= 1e-11 * a.abs().max(b.abs()),
                "{label}: {a:e} vs {b:e}"
            )
        };
        for &t in &[0.02, 0.3, 1.0, 2.5, 6.0] {
            // One rate, and Erlang(2).
            close(hypoexponential_survival(&[1.7], t), (-1.7 * t).exp(), "Exp");
            close(
                hypoexponential_survival(&[0.9, 0.9], t),
                (-0.9 * t).exp() * (1.0 + 0.9 * t),
                "Erlang",
            );
            let nu = 23usize;
            let nuf = nu as f64;
            let scale = t / 4.0;
            // D = 1, k = 6: F(6, ν) through 1/Λ − 1.
            let rates = exponential_rates(1, 6, nu);
            close(
                hypoexponential_survival(&rates, scale),
                fisher_snedecor_sf(scale.exp_m1() * nuf / 6.0, 6.0, nuf),
                "D=1",
            );
            // D = 2, k = 4: Rao through 1/√Λ − 1.
            let rates = exponential_rates(2, 4, nu);
            close(
                hypoexponential_survival(&rates, scale),
                fisher_snedecor_sf(
                    (0.5 * scale).exp_m1() * (nuf - 1.0) / 4.0,
                    8.0,
                    2.0 * (nuf - 1.0),
                ),
                "D=2",
            );
            // k = 2, D = 5: the symmetric Rao line.
            let rates = exponential_rates(5, 2, nu);
            close(
                hypoexponential_survival(&rates, scale),
                fisher_snedecor_sf(
                    (0.5 * scale).exp_m1() * (nuf - 4.0) / 5.0,
                    10.0,
                    2.0 * (nuf - 4.0),
                ),
                "k=2",
            );
            // k = 1, D = 4: the pairing against Hotelling's F(D, ν − D + 1).
            let rates = exponential_rates(4, 1, nu);
            close(
                hypoexponential_survival(&rates, scale),
                fisher_snedecor_sf(scale.exp_m1() * (nuf - 3.0) / 4.0, 4.0, nuf - 3.0),
                "k=1",
            );
            // D = 4, k = 3 pairing against its symmetric Λ(3, ν − 1, 4), even k.
            let paired = exponential_rates(4, 3, nu);
            let symmetric = exponential_rates(3, 4, nu - 1);
            close(
                hypoexponential_survival(&paired, scale),
                hypoexponential_survival(&symmetric, scale),
                "symmetry",
            );
        }
        assert_eq!(hypoexponential_survival(&[1.0, 2.0], 0.0), 1.0);
        assert_eq!(hypoexponential_survival(&[1.0, 2.0], f64::INFINITY), 0.0);
        assert_eq!(hypoexponential_survival(&[1.0, 2.0], 1e6), 0.0);
    }

    /// #3423: the convolution for odd `D` and odd `k` against laws known in
    /// closed form. On `D = 1` (odd `k`) and on `k = 1` (odd `D`) the same
    /// split — pairs, `Beta(a_D, m)` as exponentials, and `Beta(a_D + m, ½)`
    /// — must reproduce Snedecor's F exactly, and `Λ(3, ν, 5)` must equal its
    /// symmetric `Λ(5, ν + 2, 3)`, which the split decomposes differently
    /// (one pair and `Beta(a_3, 5/2)` against two pairs and `Beta(a'_5, 3/2)`).
    /// The quadrature's error is far below its `√ε` indicator and the
    /// uniformization's is `steps·ε`, so `1e-10` leaves two orders of margin;
    /// `t` reaches survivals near `1e−14`, so the upper tail is checked
    /// relatively, not only near one.
    #[test]
    fn odd_wilks_convolution_matches_closed_forms_3423() {
        let close = |a: f64, b: f64, label: &str| {
            assert!(
                (a - b).abs() <= 1e-10 * a.abs().max(b.abs()),
                "{label}: {a:e} vs {b:e}"
            )
        };
        let convolved = |dims: usize, k: usize, nu: usize, t: f64| {
            let law = WilksLaw::new(dims, k, nu);
            let shape = law.half_beta_shape.expect("odd D and odd k");
            hypoexponential_log_beta_half_survival(&law.rates, shape, t).expect("converges")
        };
        let nu = 23usize;
        let nuf = nu as f64;
        for &t in &[0.01, 0.1, 0.5, 1.0, 2.0, 3.5] {
            for &k in &[3usize, 5, 7] {
                let kf = k as f64;
                close(
                    convolved(1, k, nu, t),
                    fisher_snedecor_sf(t.exp_m1() * nuf / kf, kf, nuf),
                    &format!("D=1, k={k}, t={t}"),
                );
            }
            for &dims in &[3usize, 5] {
                let d = dims as f64;
                close(
                    convolved(dims, 1, nu, t),
                    fisher_snedecor_sf(t.exp_m1() * (nuf - d + 1.0) / d, d, nuf - d + 1.0),
                    &format!("k=1, D={dims}, t={t}"),
                );
            }
            close(
                wilks_survival(t, 3, 5, nu).expect("survival"),
                wilks_survival(t, 5, 3, nu + 2).expect("survival"),
                &format!("symmetry, t={t}"),
            );
        }
        assert!(fisher_snedecor_sf(3.5_f64.exp_m1() * nuf / 7.0, 7.0, nuf) < 1e-12);
        assert_eq!(convolved(3, 3, nu, 0.0), 1.0);
        assert_eq!(convolved(3, 3, nu, f64::INFINITY), 0.0);
        assert_eq!(convolved(3, 3, nu, 1e6), 0.0);
    }

    /// #3423: the whole odd-`D`, odd-`k` law through its Mellin transform,
    /// `E[Λˢ] = Π_i Π_{j<s} (a_i + j)/(a_i + k/2 + j)`, which is
    /// `1 − s·∫₀^∞ e^{−st}·P(−ln Λ > t) dt` for the survival evaluated here.
    /// The integral is taken with `t = x²` (the survival is `1 − O(x^{Dk})`
    /// at the origin) by 16-point panels of width `1/4` up to `x = 6`, past
    /// which `e^{−st}` is below `ε`.
    #[test]
    fn odd_wilks_survival_matches_its_mellin_transform_3423() {
        let (nodes, weights) = gauss_legendre(16);
        for &(dims, k, nu) in &[(3usize, 3usize, 20usize), (3, 5, 9), (5, 3, 12)] {
            for s in 1..=2usize {
                let sf = s as f64;
                let mut integral = 0.0;
                for panel in 0..24 {
                    let (left, right) = (0.25 * panel as f64, 0.25 * (panel + 1) as f64);
                    let (centre, half) = (0.5 * (left + right), 0.5 * (right - left));
                    for (node, weight) in nodes.iter().zip(&weights) {
                        let x = centre + half * node;
                        let t = x * x;
                        let survival = wilks_survival(t, dims, k, nu).expect("survival");
                        integral += half * weight * 2.0 * x * (-sf * t).exp() * survival;
                    }
                }
                let exact: f64 = (1..=dims)
                    .flat_map(|i| {
                        let a = (nu as f64 - i as f64 + 1.0) / 2.0;
                        (0..s).map(move |j| (a + j as f64) / (a + 0.5 * k as f64 + j as f64))
                    })
                    .product();
                let transform = 1.0 - sf * integral;
                assert!(
                    (transform - exact).abs() <= 1e-10 * exact,
                    "Λ({dims}, {nu}, {k}), s={s}: {transform:e} vs {exact:e}"
                );
            }
        }
    }

    /// END-TO-END (#993 items 1+2+4): fit_tensor_surface recovers a
    /// planted BOUND two-dimensional surface from noisy samples, its
    /// covariance feeds the carve, and the joint two-output edge test (Rao's
    /// exact F for Wilks' Λ) proves the binding while fission refuses.
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
            coeff_band: &fit.coeff_band,
            coeff_covariance: Some(&fit.coeff_covariance),
            joint_coeff_covariance: Some(&joint),
            interaction_test: Some(&fit.interaction_test),
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        let p = report.edge_p_value().expect("edge test ran");
        assert!(p < 1e-3, "planted joint binding must reject, p = {p}");
        assert!(report.fission.is_none(), "bound surface must not fission");
        assert!(report.interaction_fraction > 0.05);
    }

    /// END-TO-END, additive side: a planted ADDITIVE surface fit from
    /// near-noiseless samples splits once the carve holds the fit's covariance:
    /// its interaction energy is estimation noise inside `√R + √(P/α)`. Handed
    /// over bare, the same estimate clears the rounding floor alone and stays
    /// whole, the control showing the posterior term is what certifies the split.
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
        let with_covariance = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &fit.coeffs,
            coeff_band: &fit.coeff_band,
            coeff_covariance: Some(&fit.coeff_covariance),
            joint_coeff_covariance: None,
            interaction_test: Some(&fit.interaction_test),
            notion: BindingNotion::Representational,
        };
        let report = carve(&with_covariance, 0.05).expect("carve");
        assert!(
            report.fission.is_some(),
            "an additive surface with its covariance must split (fraction = {}, p = {:?})",
            report.interaction_fraction,
            report.edge_p_value()
        );

        let bare = CarveInput {
            phi_a: phi_a.view(),
            phi_b: phi_b.view(),
            coeffs: &fit.coeffs,
            coeff_band: &fit.coeff_band,
            coeff_covariance: None,
            joint_coeff_covariance: None,
            interaction_test: Some(&fit.interaction_test),
            notion: BindingNotion::Representational,
        };
        let bare_report = carve(&bare, 0.05).expect("carve");
        assert!(
            bare_report.fission.is_none(),
            "without a covariance only the rounding floor certifies, and a noisy estimate \
             clears it (fraction = {})",
            bare_report.interaction_fraction
        );
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
            edge_test: Err(BindingTestUnavailable::NoStatistics),
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
            edge_test: Ok(EdgeBindingTest::SingleOutput(BindingFTest {
                statistic: 50.0,
                numerator_df: 4,
                denominator_df: 31,
                p_value: 1e-9,
            })),
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
        // The re-fit reproduces the decoder's surface exactly, so no residual
        // variation is left to scale a test against: the test reports that,
        // and the carve's resolution dial alone decides.
        assert_eq!(
            report.edge_test,
            Err(BindingTestUnavailable::NoResidualVariation { dim: 0 })
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

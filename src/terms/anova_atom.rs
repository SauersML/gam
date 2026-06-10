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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::inference::smooth_test::{
    SmoothTestInput, SmoothTestResult, SmoothTestScale, wood_smooth_test,
};

/// Interaction energy fraction at or below which the interaction block is
/// energetically negligible and lossless fission is on the table. A
/// REML/ARD-killed interaction lands at numerical zero; `1e-6` of the
/// centered surface variance corresponds to a relative amplitude of
/// `1e-3` (≈ √fraction), far below behavioral relevance for any consumer
/// of the carve, while sitting far above f64 roundoff accumulated through
/// the centering algebra. Auto-applied — no knob.
pub const FISSION_MAX_INTERACTION_FRACTION: f64 = 1e-6;

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

    // Gauge projectors P_i = I − û ûᵀ for the centered-basis dependence.
    let proj_a = gauge_projector(m1, input.kernel_a.as_ref())?;
    let proj_b = gauge_projector(m2, input.kernel_b.as_ref())?;

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
    let ran: Vec<f64> = binding_tests
        .iter()
        .flatten()
        .map(|t| t.p_value)
        .collect();
    let edge_p_value = ran
        .iter()
        .cloned()
        .fold(None, |acc: Option<f64>, p| {
            Some(acc.map_or(p, |a| a.min(p)))
        })
        .map(|min_p| (min_p * ran.len() as f64).min(1.0));

    let binding_proven = edge_p_value.is_some_and(|p| p <= alpha);
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
    // K = P₁ ⊗ P₂ under the row-major vec convention
    // (vec(A X B)[a·M₂+c] = Σ A[a,j]·B[k,c]·vec(X)[j·M₂+k]; P₂ symmetric).
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
    let cov_z = kron.dot(cov).dot(&kron.t());
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
            let rebuilt =
                blocks.mean + pa_c.dot(&blocks.main_a) + pb_c.dot(&blocks.main_b) + f12;
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
            kernel_a: None,
            kernel_b: None,
            edf: None,
            residual_df: 100.0,
            scale: SmoothTestScale::Known,
            notion: BindingNotion::Representational,
        };
        let report = carve(&input, 0.05).expect("carve");
        assert!(report.interaction_fraction < 1e-24);
        let plan = report.fission.as_ref().expect("additive surface must fission");
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
}

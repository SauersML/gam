//! Reuse versus specialization of one retained function across contexts (#2946).
//!
//! Several contexts (sites, token positions, tasks) observe one retained function through their own basis rows `Φ_c`.
//! Two declared hypotheses explain the observations `y_c`:
//!
//! - **shared**: one coefficient `β` drives every context through its adapter `A_c`, so the prior cross-context
//!   covariance is `Cov(y_c, y_d) = Φ_c A_c Q⁻¹ A_dᵀ Φ_dᵀ`;
//! - **specialized**: each context carries its own `β_c`, so the prior cross-context covariance is `0`.
//!
//! Two arms compare them.
//!
//! - [`compare_reuse`], **declared prior**: `β ~ N(0, Q⁻¹)` with `Q` proper and the noise declared. The coefficients are
//!   integrated exactly by [`gam_solve::gaussian_marginal`] (#2933 comment 5715551157 §A), so the Bayes factor is exact.
//!   With identity adapters, `H_c = Q + Φ_cᵀR_c⁻¹Φ_c`, `H = Q + Σ_c Φ_cᵀR_c⁻¹Φ_c`, `b_c = Φ_cᵀR_c⁻¹y_c` and `b = Σ_c b_c`,
//!   it is
//!
//!   ```text
//!   ½[bᵀH⁻¹b − Σ_c b_cᵀH_c⁻¹b_c] + ½[Σ_c log|H_c| − log|H| − (m − 1)·log|Q|]
//!   ```
//!
//!   for `m` contexts. The noise terms cancel, and the second bracket is the Occam factor of `m` independent coefficient
//!   volumes against one.
//! - [`compare_reuse_reml`], **prior scale not declared**: a penalty set `S_1..S_K` with one REML-fitted strength vector
//!   per hypothesis, fitted by exact multi-penalty Gaussian REML ([`gam_solve::gaussian_reml_multi_penalty`]) with the
//!   dispersion profiled (#2822 M1). Both hypotheses carry the same hyperparameters `(λ_1, …, λ_K, σ²)`, so their
//!   profiled evidences are comparable. The penalties' declared null space `N` (an unpenalized intercept, say) enters
//!   BOTH hypotheses as per-context fixed effects: each context keeps its own null-space coefficients under a flat prior,
//!   and only the penalized part is shared. The hypotheses then integrate the same improper space, `m·dim N` flat
//!   coordinates in one parametrization, so its arbitrary constant cancels from the Bayes factor exactly. Sharing the
//!   null space too would integrate `dim N` flat directions against `m·dim N`, whose constants cannot cancel.
//!
//! **Adapters are charged.** An alignment of the contexts' coordinates belongs to the shared hypothesis, and the shared
//! evidence is the prior-mass mixture over the DECLARED alignments. Picking the best of them after seeing the data is
//! worth at most the log of its normalized prior mass. A continuous alignment fitted post hoc (a rotation chosen to
//! make the functions agree) has no exact evidence, and this API cannot express one. Specialization reads each context
//! in its own coordinates with the same prior; an adapter that is not a prior isometry (`A_cᵀQA_c ≠ Q`) also changes
//! that context's marginal function prior under sharing, and the Bayes factor prices that change too.

use faer::Side;
use gam_linalg::faer_ndarray::{FaerArrayView, FaerCholesky, FaerEigh, HouseholderQr, fast_ab};
use gam_linalg::roundoff::{accumulation_growth, symmetric_spectrum_rounding_band};
use gam_linalg::utils::KahanSum;
use gam_math::special::{logaddexp, logistic};
use gam_solve::gaussian_marginal::{
    GaussianEvidenceParts, GaussianMarginalError, GaussianMarginalModel,
};
use gam_solve::gaussian_reml_multi_penalty::{
    GaussianRemlMultiPenaltyFit, GaussianRemlMultiPenaltyProblem,
    GaussianRemlMultiPenaltyRhoPlacement,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate, s};

/// One context's observations of the retained function.
#[derive(Clone, Copy, Debug)]
pub struct ContextObservations<'a> {
    /// Basis rows `Φ_c`, `n_c × p`.
    pub basis: ArrayView2<'a, f64>,
    /// Responses `y_c`, length `n_c`.
    pub response: ArrayView1<'a, f64>,
    /// Diagonal noise covariance `R_c`, length `n_c`, strictly positive. In [`compare_reuse_reml`] it is the noise
    /// shape, with the common scale profiled.
    pub noise_variance: ArrayView1<'a, f64>,
}

/// One declared way the contexts read a shared coefficient: context `c` sees `A_c β`, so its function is `Φ_c A_c β`.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedAlignment {
    /// One `p × p` adapter per context, in context order.
    pub adapters: Vec<Array2<f64>>,
    /// Relative prior mass, finite and strictly positive. The masses are normalized over the declared set.
    pub prior_weight: f64,
}

impl SharedAlignment {
    /// Every context reads the shared coefficient in the same coordinates, with relative prior mass `1`.
    pub fn identity(context_count: usize, coefficients: usize) -> Self {
        Self {
            adapters: vec![Array2::eye(coefficients); context_count],
            prior_weight: 1.0,
        }
    }
}

/// A comparison of reuse against specialization.
#[derive(Clone, Debug, PartialEq)]
pub struct ReuseComparison {
    /// `log p(y | shared)`: the prior-mass mixture over the declared alignments.
    pub log_evidence_shared: f64,
    /// `log p(y | shared, alignment k)`, in declaration order, before weighting by the alignment's prior mass.
    pub alignment_log_evidence: Vec<f64>,
    /// `log p(y | specialized)`.
    pub log_evidence_specialized: f64,
    /// `log p(y | shared) − log p(y | specialized)`, in nats. Positive favours reuse.
    pub log_bayes_factor: f64,
    /// `log_bayes_factor + ln(π/(1 − π))` for the declared structural prior probability `π` of sharing.
    pub log_posterior_odds: f64,
    /// `P(shared | y)`.
    pub posterior_share_probability: f64,
}

/// The REML arm's comparison with its fitted smoothing strengths, where they sit, and its resolution.
#[derive(Clone, Debug, PartialEq)]
pub struct RemlReuseComparison {
    /// The comparison. Each log evidence is `log p(y | λ̂, σ̂²)`: the negative minimized REML cost of the whitened
    /// problem less `½Σ ln r_i` for the declared noise shape.
    pub comparison: ReuseComparison,
    /// `λ̂` of the shared hypothesis under each declared alignment, one strength per penalty.
    pub alignment_lambdas: Vec<Array1<f64>>,
    /// Per shared fit, per penalty: where `ρ̂ = ln λ̂` sits in the REML owner's derived domain.
    pub alignment_rho_placement: Vec<Vec<GaussianRemlMultiPenaltyRhoPlacement>>,
    /// `λ̂` of the specialized hypothesis, one strength per penalty for every context's block.
    pub specialized_lambdas: Array1<f64>,
    /// Per penalty: where the specialized fit's `ρ̂` sits.
    pub specialized_rho_placement: Vec<GaussianRemlMultiPenaltyRhoPlacement>,
    /// Bound on how far `comparison.log_bayes_factor` can sit from the exact profiled value: each fit's forward-error
    /// bound (widened for the noise whitening) plus its optimality gap `½g_Fᵀ H_FF⁻¹ g_F` over the interior coordinates
    /// `F` of `ρ̂`. The mixture over alignments is 1-Lipschitz in each alignment's evidence, so the shared side
    /// contributes its largest bound. `None` when a fit's placement is unaudited or its interior Hessian block is not
    /// positive definite: the verdict then has no established resolution.
    pub log_bayes_factor_resolution: Option<f64>,
}

/// Why a reuse comparison could not be computed.
#[derive(Debug, PartialEq)]
pub enum ReuseError {
    /// A shape, finiteness, positivity or count requirement failed.
    InvalidInput(String),
    /// The declared-prior evidence refused its input (an improper prior among them).
    Evidence(GaussianMarginalError),
    /// A Gaussian REML problem or fit was refused; an undeclared null space of the penalty set among them.
    Reml(String),
    /// Penalty `penalty` does not annihilate the declared null space: `‖S_k N̂‖_F` exceeds the band inside which its own
    /// spectrum cannot resolve a direction from zero.
    NullSpaceNotAnnihilated {
        penalty: usize,
        residual: f64,
        band: f64,
    },
}

impl std::fmt::Display for ReuseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Reml(message) => f.write_str(message),
            Self::Evidence(error) => write!(f, "reuse comparison: {error}"),
            Self::NullSpaceNotAnnihilated {
                penalty,
                residual,
                band,
            } => write!(
                f,
                "reuse comparison: penalty {penalty} does not annihilate the declared null space: ‖S N̂‖_F = \
                 {residual:e} exceeds its rounding band {band:e}"
            ),
        }
    }
}

impl std::error::Error for ReuseError {}

impl From<GaussianMarginalError> for ReuseError {
    fn from(error: GaussianMarginalError) -> Self {
        Self::Evidence(error)
    }
}

/// Validate the declared structure shared by both arms and return the total alignment prior weight.
fn validate_structure(
    contexts: &[ContextObservations<'_>],
    alignments: &[SharedAlignment],
    prior_share_probability: f64,
) -> Result<f64, ReuseError> {
    if contexts.len() < 2 {
        return Err(ReuseError::InvalidInput(format!(
            "reuse versus specialization needs at least two contexts; got {}",
            contexts.len()
        )));
    }
    if !(prior_share_probability > 0.0 && prior_share_probability < 1.0) {
        return Err(ReuseError::InvalidInput(format!(
            "the structural prior probability of sharing must lie strictly inside (0, 1); got {prior_share_probability}"
        )));
    }
    let coefficients = contexts[0].basis.ncols();
    for (index, context) in contexts.iter().enumerate() {
        let rows = context.basis.nrows();
        if context.basis.ncols() != coefficients {
            return Err(ReuseError::InvalidInput(format!(
                "context {index} basis has {} columns; context 0 has {coefficients}",
                context.basis.ncols()
            )));
        }
        if context.response.len() != rows || context.noise_variance.len() != rows {
            return Err(ReuseError::InvalidInput(format!(
                "context {index} basis has {rows} rows but {} responses and {} noise variances",
                context.response.len(),
                context.noise_variance.len()
            )));
        }
    }
    if alignments.is_empty() {
        return Err(ReuseError::InvalidInput(
            "declare at least one shared alignment; SharedAlignment::identity declares shared coordinates"
                .to_string(),
        ));
    }
    let mut total_weight = 0.0;
    for (index, alignment) in alignments.iter().enumerate() {
        if !(alignment.prior_weight.is_finite() && alignment.prior_weight > 0.0) {
            return Err(ReuseError::InvalidInput(format!(
                "alignment {index} prior weight must be finite and positive; got {}",
                alignment.prior_weight
            )));
        }
        if alignment.adapters.len() != contexts.len() {
            return Err(ReuseError::InvalidInput(format!(
                "alignment {index} has {} adapters for {} contexts",
                alignment.adapters.len(),
                contexts.len()
            )));
        }
        if let Some((context, adapter)) =
            alignment.adapters.iter().enumerate().find(|(_, adapter)| {
                adapter.dim() != (coefficients, coefficients)
                    || adapter.iter().any(|value| !value.is_finite())
            })
        {
            return Err(ReuseError::InvalidInput(format!(
                "alignment {index} adapter for context {context} must be a finite {coefficients} x {coefficients} \
                 matrix; got {}x{}",
                adapter.nrows(),
                adapter.ncols()
            )));
        }
        total_weight += alignment.prior_weight;
    }
    if !total_weight.is_finite() {
        return Err(ReuseError::InvalidInput(
            "the alignment prior weights overflow their sum".to_string(),
        ));
    }
    Ok(total_weight)
}

/// The shared hypothesis's stacked design `[Φ_1A_1; …; Φ_mA_m]`, responses and noise variances.
fn shared_design(
    contexts: &[ContextObservations<'_>],
    adapters: &[Array2<f64>],
) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>), ReuseError> {
    let designs: Vec<Array2<f64>> = contexts
        .iter()
        .zip(adapters)
        .map(|(context, adapter)| fast_ab(&context.basis, adapter))
        .collect();
    let design_views: Vec<ArrayView2<'_, f64>> =
        designs.iter().map(|design| design.view()).collect();
    let response_views: Vec<ArrayView1<'_, f64>> =
        contexts.iter().map(|context| context.response).collect();
    let noise_views: Vec<ArrayView1<'_, f64>> = contexts
        .iter()
        .map(|context| context.noise_variance)
        .collect();
    let shape = |error: ndarray::ShapeError| ReuseError::InvalidInput(error.to_string());
    Ok((
        concatenate(Axis(0), &design_views).map_err(shape)?,
        concatenate(Axis(0), &response_views).map_err(shape)?,
        concatenate(Axis(0), &noise_views).map_err(shape)?,
    ))
}

fn assemble(
    log_evidence_shared: f64,
    alignment_log_evidence: Vec<f64>,
    log_evidence_specialized: f64,
    prior_share_probability: f64,
) -> ReuseComparison {
    // Both arms pass normalized log evidences (exact under a declared prior, at each hypothesis' REML λ in
    // `compare_reuse_reml`), so their difference is the log Bayes factor itself, not a raw `criterion_gap`.
    let log_bayes_factor = log_evidence_shared - log_evidence_specialized;
    let log_posterior_odds =
        log_bayes_factor + prior_share_probability.ln() - (-prior_share_probability).ln_1p();
    ReuseComparison {
        log_evidence_shared,
        alignment_log_evidence,
        log_evidence_specialized,
        log_bayes_factor,
        log_posterior_odds,
        posterior_share_probability: logistic(log_posterior_odds),
    }
}

/// Compare reuse of one function across `contexts` against specialization by exact Gaussian evidence under the
/// declared proper prior precision `Q`.
///
/// `alignments` declares every way the shared coefficient may be read by the contexts, with its prior mass.
/// `prior_share_probability` is the declared structural prior probability of sharing, strictly inside `(0, 1)`: it
/// moves the posterior odds and never the Bayes factor.
pub fn compare_reuse(
    contexts: &[ContextObservations<'_>],
    prior_precision: ArrayView2<'_, f64>,
    alignments: &[SharedAlignment],
    prior_share_probability: f64,
) -> Result<ReuseComparison, ReuseError> {
    let total_weight = validate_structure(contexts, alignments, prior_share_probability)?;
    let mut alignment_log_evidence = Vec::with_capacity(alignments.len());
    let mut log_evidence_shared = f64::NEG_INFINITY;
    for alignment in alignments {
        let (design, response, noise_variance) = shared_design(contexts, &alignment.adapters)?;
        let log_evidence = GaussianMarginalModel::new(
            design.view(),
            response.view(),
            noise_variance.view(),
            prior_precision,
        )?
        .evidence()?
        .log_evidence();
        log_evidence_shared = logaddexp(
            log_evidence_shared,
            (alignment.prior_weight / total_weight).ln() + log_evidence,
        );
        alignment_log_evidence.push(log_evidence);
    }
    let mut specialized = GaussianEvidenceParts {
        quadratic: 0.0,
        log_det: 0.0,
        observations: 0,
    };
    for context in contexts {
        specialized = specialized.independent_sum(
            GaussianMarginalModel::new(
                context.basis,
                context.response,
                context.noise_variance,
                prior_precision,
            )?
            .evidence()?,
        );
    }
    Ok(assemble(
        log_evidence_shared,
        alignment_log_evidence,
        specialized.log_evidence(),
        prior_share_probability,
    ))
}

/// The penalty set with its declared null space `N` split off: an orthonormal `N̂` spanning `N`, an orthonormal
/// complement `U`, and every penalty restricted to the complement, `UᵀS_kU`.
struct PenaltySplit {
    null_basis: Array2<f64>,
    complement: Array2<f64>,
    restricted: Vec<Array2<f64>>,
}

fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    let mut sum = KahanSum::default();
    for value in matrix {
        sum.add(value * value);
    }
    sum.sum().sqrt()
}

/// How far a declared null space escapes one penalty: `‖S N̂‖_F` for the orthonormal basis `N̂` of the declaration,
/// against the band inside which the penalty's own spectrum cannot resolve a direction from zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NullSpaceAnnihilation {
    /// `‖S N̂‖_F`.
    pub residual: f64,
    /// `√k·(p·ε·‖S‖₂ + γ_p‖S‖_F)`. A direction the eigensolver cannot resolve from zero has `‖Sv‖ ≤ p·ε·‖S‖₂`
    /// ([`symmetric_spectrum_rounding_band`]), and the product that measures it rounds by at most `γ_p‖S‖_F` per unit
    /// column. Zero for an empty declaration.
    pub band: f64,
}

impl NullSpaceAnnihilation {
    /// The penalty annihilates the declaration to its own rounding.
    pub fn holds(&self) -> bool {
        self.residual <= self.band
    }
}

fn validate_penalties(penalties: &[Array2<f64>], coefficients: usize) -> Result<(), ReuseError> {
    if penalties.is_empty() {
        return Err(ReuseError::InvalidInput(
            "the REML arm needs at least one penalty".to_string(),
        ));
    }
    if let Some((index, penalty)) = penalties.iter().enumerate().find(|(_, penalty)| {
        penalty.dim() != (coefficients, coefficients)
            || penalty.iter().any(|value| !value.is_finite())
    }) {
        return Err(ReuseError::InvalidInput(format!(
            "penalty {index} must be a finite {coefficients} x {coefficients} matrix; got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        )));
    }
    Ok(())
}

/// An orthonormal basis `N̂` of the declared null space and its orthonormal complement `U`.
fn orthonormal_null_split(
    null_space: ArrayView2<'_, f64>,
    coefficients: usize,
) -> Result<(Array2<f64>, Array2<f64>), ReuseError> {
    let nullity = null_space.ncols();
    if null_space.nrows() != coefficients
        || nullity >= coefficients
        || null_space.iter().any(|value| !value.is_finite())
    {
        return Err(ReuseError::InvalidInput(format!(
            "the declared null space must be a finite {coefficients} x k matrix with k < {coefficients}; got {}x{nullity}",
            null_space.nrows()
        )));
    }
    if nullity == 0 {
        return Ok((Array2::zeros((coefficients, 0)), Array2::eye(coefficients)));
    }
    // Householder QR `N = QR`: the leading `k` columns of `Q` span `N` and the rest are its orthonormal complement. An
    // exactly rank-deficient `N` has a zero on `R`'s diagonal, which the QR's backward error raises to at most
    // `γ_{pk}‖N‖_F`.
    let declared = null_space.to_owned();
    let qr = HouseholderQr::new(FaerArrayView::new(&declared).as_ref());
    let rank_band = accumulation_growth(coefficients * nullity) * frobenius_norm(&declared);
    if let Some(column) = (0..nullity).find(|&column| qr.r()[(column, column)].abs() <= rank_band) {
        return Err(ReuseError::InvalidInput(format!(
            "the declared null space is not of full column rank: column {column} is not resolved above its QR \
             rounding band {rank_band:e}"
        )));
    }
    let mut transposed_q = faer::Mat::<f64>::identity(coefficients, coefficients);
    qr.apply_transpose_on_the_left(transposed_q.as_mut());
    let null_basis = Array2::from_shape_fn((coefficients, nullity), |(row, col)| {
        transposed_q[(col, row)]
    });
    let complement = Array2::from_shape_fn((coefficients, coefficients - nullity), |(row, col)| {
        transposed_q[(nullity + col, row)]
    });
    Ok((null_basis, complement))
}

fn annihilation(
    penalty: &Array2<f64>,
    null_basis: &Array2<f64>,
    index: usize,
) -> Result<NullSpaceAnnihilation, ReuseError> {
    let nullity = null_basis.ncols();
    if nullity == 0 {
        return Ok(NullSpaceAnnihilation {
            residual: 0.0,
            band: 0.0,
        });
    }
    let (eigenvalues, _) = penalty.eigh(Side::Lower).map_err(|error| {
        ReuseError::InvalidInput(format!("penalty {index} spectrum: {error}"))
    })?;
    Ok(NullSpaceAnnihilation {
        residual: frobenius_norm(&fast_ab(penalty, null_basis)),
        band: (nullity as f64).sqrt()
            * (symmetric_spectrum_rounding_band(&eigenvalues.to_vec())
                + accumulation_growth(penalty.nrows()) * frobenius_norm(penalty)),
    })
}

/// Measure, per penalty, how far the declared null space (`p × k`, any full-column-rank basis) escapes it. This is the
/// one predicate [`compare_reuse_reml`] refuses a declaration by ([`ReuseError::NullSpaceNotAnnihilated`]), so a
/// declaration whose every entry [`NullSpaceAnnihilation::holds`] is accepted there.
pub fn null_space_annihilation(
    penalties: &[Array2<f64>],
    null_space: ArrayView2<'_, f64>,
) -> Result<Vec<NullSpaceAnnihilation>, ReuseError> {
    let coefficients = null_space.nrows();
    validate_penalties(penalties, coefficients)?;
    let (null_basis, _) = orthonormal_null_split(null_space, coefficients)?;
    penalties
        .iter()
        .enumerate()
        .map(|(index, penalty)| annihilation(penalty, &null_basis, index))
        .collect()
}

fn split_penalties(
    penalties: &[Array2<f64>],
    null_space: ArrayView2<'_, f64>,
    coefficients: usize,
) -> Result<PenaltySplit, ReuseError> {
    validate_penalties(penalties, coefficients)?;
    let (null_basis, complement) = orthonormal_null_split(null_space, coefficients)?;
    if null_basis.ncols() == 0 {
        return Ok(PenaltySplit {
            null_basis,
            complement,
            restricted: penalties.to_vec(),
        });
    }
    let mut restricted = Vec::with_capacity(penalties.len());
    for (index, penalty) in penalties.iter().enumerate() {
        let measured = annihilation(penalty, &null_basis, index)?;
        if !measured.holds() {
            return Err(ReuseError::NullSpaceNotAnnihilated {
                penalty: index,
                residual: measured.residual,
                band: measured.band,
            });
        }
        let projected = fast_ab(&complement.t(), &fast_ab(penalty, &complement));
        restricted.push((&projected + &projected.t()) * 0.5);
    }
    Ok(PenaltySplit {
        null_basis,
        complement,
        restricted,
    })
}

/// `[A | B]` for two blocks with the same rows.
fn side_by_side(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
    let rows = left.nrows();
    let mut joined = Array2::zeros((rows, left.ncols() + right.ncols()));
    joined.slice_mut(s![.., ..left.ncols()]).assign(left);
    joined.slice_mut(s![.., left.ncols()..]).assign(right);
    joined
}

/// Each restricted penalty `S̃_k` placed `copies` times on the diagonal after `offset` unpenalized coordinates.
fn embedded_penalties(
    restricted: &[Array2<f64>],
    offset: usize,
    copies: usize,
) -> Vec<Array2<f64>> {
    restricted
        .iter()
        .map(|penalty| {
            let width = penalty.nrows();
            let size = offset + copies * width;
            let mut embedded = Array2::zeros((size, size));
            for copy in 0..copies {
                let block = offset + copy * width..offset + (copy + 1) * width;
                embedded.slice_mut(s![block.clone(), block]).assign(penalty);
            }
            embedded
        })
        .collect()
}

struct RemlFit {
    log_evidence: f64,
    lambdas: Array1<f64>,
    placement: Vec<GaussianRemlMultiPenaltyRhoPlacement>,
    resolution: Option<f64>,
}

/// `½g_Fᵀ H_FF⁻¹ g_F` over the interior coordinates `F` of `ρ̂`: to second order, how far the criterion at `ρ̂` sits above
/// its minimum over the owner's domain. A railed coordinate sits at the domain's edge, where that minimum is defined, and
/// contributes no gap. `None` when a placement is unaudited or `H_FF` is not positive definite.
fn optimality_gap(fit: &GaussianRemlMultiPenaltyFit) -> Option<f64> {
    if fit
        .rho_placement
        .contains(&GaussianRemlMultiPenaltyRhoPlacement::Unaudited)
    {
        return None;
    }
    let interior: Vec<usize> = fit
        .rho_placement
        .iter()
        .enumerate()
        .filter(|(_, placement)| **placement == GaussianRemlMultiPenaltyRhoPlacement::Interior)
        .map(|(index, _)| index)
        .collect();
    if interior.is_empty() {
        return Some(0.0);
    }
    let evaluation = &fit.evaluation;
    let gradient = Array1::from_iter(
        interior
            .iter()
            .map(|&index| evaluation.reml_gradient[index]),
    );
    let hessian = Array2::from_shape_fn((interior.len(), interior.len()), |(row, col)| {
        evaluation.reml_hessian[[interior[row], interior[col]]]
    });
    let step = hessian.cholesky(Side::Lower).ok()?.solvevec(&gradient);
    Some(0.5 * gradient.dot(&step).max(0.0))
}

fn reml_fit(
    design: &Array2<f64>,
    response: &Array1<f64>,
    noise_variance: &Array1<f64>,
    penalties: &[Array2<f64>],
    nullity: usize,
    hypothesis: &'static str,
) -> Result<RemlFit, ReuseError> {
    if let Some((row, variance)) = noise_variance
        .iter()
        .enumerate()
        .find(|(_, variance)| !(variance.is_finite() && **variance > 0.0))
    {
        return Err(ReuseError::InvalidInput(format!(
            "the noise variance at stacked row {row} must be finite and positive; got {variance}"
        )));
    }
    let scales = noise_variance.mapv(|variance| variance.sqrt().recip());
    let whitened_design = design * &scales.view().insert_axis(Axis(1));
    let whitened_response = (response * &scales).insert_axis(Axis(1));
    let fit = GaussianRemlMultiPenaltyProblem::new(
        whitened_design.view(),
        whitened_response.view(),
        penalties,
        nullity,
    )
    .and_then(|problem| problem.fit(None))
    .map_err(|error| ReuseError::Reml(format!("{hypothesis} hypothesis REML refused: {error}")))?;
    // `log p(y) = log p(R^{-1/2}y) − ½Σ ln r_i`. Both hypotheses stack the same rows in the same order, so this sum is
    // bitwise the same on both sides and cancels from the Bayes factor.
    let mut log_noise = KahanSum::default();
    for variance in noise_variance {
        log_noise.add(variance.ln());
    }
    let evaluation = &fit.evaluation;
    // Whitening multiplies every entry by `fl(1/√r_i)`, a relative perturbation of at most `γ_3`. The owner's bound is
    // first order in the columnwise backward error `γ_{n·p}` of its design QR and its application to `y`, so the
    // perturbation widens that bound by at most the factor `1 + γ_3/γ_{n·p}`.
    let (rows, columns) = design.dim();
    let whitening = 1.0 + accumulation_growth(3) / accumulation_growth(rows * columns);
    Ok(RemlFit {
        log_evidence: -evaluation.reml_score - 0.5 * log_noise.sum(),
        lambdas: evaluation.lambdas.clone(),
        placement: fit.rho_placement.clone(),
        resolution: optimality_gap(&fit)
            .map(|gap| whitening * evaluation.reml_score_roundoff + gap),
    })
}

/// Compare reuse against specialization when the prior scale is not declared: penalties `S_1..S_K` with one
/// REML-fitted strength vector per hypothesis and the dispersion profiled (#2822 M1).
///
/// `null_space` (`p × k`, `k` may be 0) declares `∩_k ker S_k`, the directions no penalty reaches. With `N̂` its
/// orthonormal basis and `U` the orthonormal complement, context `c` reads `Φ_c(N̂γ_c + ·)`: its own null-space
/// coefficients `γ_c` under a flat prior in BOTH hypotheses. The shared fit then uses `[diag(Φ_1N̂, …, Φ_mN̂) |
/// Φ_1A_1U; …; Φ_mA_mU]` with penalties `diag(0, UᵀS_kU)`; the specialized fit uses `[diag(Φ_cN̂) | diag(Φ_cU)]` with
/// `diag(0, UᵀS_kU, …, UᵀS_kU)` and the same `λ_k` for every context. Both integrate the same `m·k` flat coordinates,
/// so the flat prior's arbitrary constant cancels from the Bayes factor. A penalty that does not annihilate the
/// declared null space is refused with [`ReuseError::NullSpaceNotAnnihilated`], and an undeclared null direction by the
/// REML owner's structural rank check.
pub fn compare_reuse_reml(
    contexts: &[ContextObservations<'_>],
    penalties: &[Array2<f64>],
    null_space: ArrayView2<'_, f64>,
    alignments: &[SharedAlignment],
    prior_share_probability: f64,
) -> Result<RemlReuseComparison, ReuseError> {
    let total_weight = validate_structure(contexts, alignments, prior_share_probability)?;
    let coefficients = contexts[0].basis.ncols();
    let split = split_penalties(penalties, null_space, coefficients)?;
    let context_count = contexts.len();
    let nullity = split.null_basis.ncols();
    let fixed_count = context_count * nullity;
    let rows: usize = contexts.iter().map(|context| context.basis.nrows()).sum();

    // `diag(Φ_1N̂, …, Φ_mN̂)` and `diag(Φ_1U, …, Φ_mU)`, row blocks in context order.
    let penalized_width = coefficients - nullity;
    let mut fixed_effects = Array2::<f64>::zeros((rows, fixed_count));
    let mut specialized_penalized = Array2::<f64>::zeros((rows, context_count * penalized_width));
    let mut start = 0;
    for (index, context) in contexts.iter().enumerate() {
        let end = start + context.basis.nrows();
        if nullity > 0 {
            fixed_effects
                .slice_mut(s![start..end, index * nullity..(index + 1) * nullity])
                .assign(&fast_ab(&context.basis, &split.null_basis));
        }
        specialized_penalized
            .slice_mut(s![
                start..end,
                index * penalized_width..(index + 1) * penalized_width
            ])
            .assign(&fast_ab(&context.basis, &split.complement));
        start = end;
    }

    let shared_penalties = embedded_penalties(&split.restricted, fixed_count, 1);
    let mut alignment_log_evidence = Vec::with_capacity(alignments.len());
    let mut alignment_lambdas = Vec::with_capacity(alignments.len());
    let mut alignment_rho_placement = Vec::with_capacity(alignments.len());
    let mut log_evidence_shared = f64::NEG_INFINITY;
    let mut shared_resolution = Some(0.0_f64);
    for alignment in alignments {
        let (stacked, response, noise_variance) = shared_design(contexts, &alignment.adapters)?;
        let design = side_by_side(&fixed_effects, &fast_ab(&stacked, &split.complement));
        let fit = reml_fit(
            &design,
            &response,
            &noise_variance,
            &shared_penalties,
            fixed_count,
            "shared",
        )?;
        log_evidence_shared = logaddexp(
            log_evidence_shared,
            (alignment.prior_weight / total_weight).ln() + fit.log_evidence,
        );
        alignment_log_evidence.push(fit.log_evidence);
        alignment_lambdas.push(fit.lambdas);
        alignment_rho_placement.push(fit.placement);
        shared_resolution = shared_resolution
            .zip(fit.resolution)
            .map(|(bound, resolution)| bound.max(resolution));
    }

    let response_views: Vec<ArrayView1<'_, f64>> =
        contexts.iter().map(|context| context.response).collect();
    let noise_views: Vec<ArrayView1<'_, f64>> = contexts
        .iter()
        .map(|context| context.noise_variance)
        .collect();
    let shape = |error: ndarray::ShapeError| ReuseError::InvalidInput(error.to_string());
    let response = concatenate(Axis(0), &response_views).map_err(shape)?;
    let noise_variance = concatenate(Axis(0), &noise_views).map_err(shape)?;
    let specialized = reml_fit(
        &side_by_side(&fixed_effects, &specialized_penalized),
        &response,
        &noise_variance,
        &embedded_penalties(&split.restricted, fixed_count, context_count),
        fixed_count,
        "specialized",
    )?;

    Ok(RemlReuseComparison {
        comparison: assemble(
            log_evidence_shared,
            alignment_log_evidence,
            specialized.log_evidence,
            prior_share_probability,
        ),
        alignment_lambdas,
        alignment_rho_placement,
        specialized_lambdas: specialized.lambdas,
        specialized_rho_placement: specialized.placement,
        log_bayes_factor_resolution: shared_resolution
            .zip(specialized.resolution)
            .map(|(shared, specialized)| shared + specialized),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::consts::PI;

    fn euclidean(vector: &Array1<f64>) -> f64 {
        vector.dot(vector).sqrt()
    }

    fn frobenius(matrix: &Array2<f64>) -> f64 {
        matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    fn spectrum_extremes(matrix: &Array2<f64>) -> (f64, f64) {
        let values = matrix
            .eigh(Side::Lower)
            .expect("fixture matrices are symmetric")
            .0;
        let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
        let largest = values.iter().copied().fold(0.0_f64, f64::max);
        (smallest, largest)
    }

    /// First-order rounding band of `log p(y)` for one declared-prior problem, which every fixture here evaluates in the
    /// dual form (`p ≤ n`). Accumulation bands are Higham ASNA Lemma 3.1 (`γ_k·Σ|terms|`), and a strict Cholesky of an
    /// SPD `m × m` matrix carries the backward error `‖ΔM‖₂ ≤ m·γ_{3m+1}·‖M‖₂` (Thm 10.3), as in gam-solve's
    /// `gaussian_marginal` tests; `|Δ log|M|| ≤ m·‖ΔM‖₂/λ_min` to first order.
    fn dual_log_evidence_band(
        basis: &Array2<f64>,
        response: &Array1<f64>,
        noise: &Array1<f64>,
        prior: &Array2<f64>,
    ) -> f64 {
        let gamma = accumulation_growth;
        let (n, p) = basis.dim();
        assert!(p <= n, "the fixture band covers the dual form only");
        let scales = noise.mapv(|variance| variance.sqrt().recip());
        let design = basis * &scales.view().insert_axis(Axis(1));
        let whitened = response * &scales;
        let abs_design = design.mapv(f64::abs);
        let abs_prior = prior.mapv(f64::abs);
        let whitening = gamma(3);
        let chol = |dim: usize, largest: f64| dim as f64 * gamma(3 * dim + 1) * largest;
        let log_magnitude =
            |(smallest, largest): (f64, f64)| smallest.ln().abs().max(largest.ln().abs());

        let gram_terms = abs_design.t().dot(&abs_design);
        let precision = prior + &design.t().dot(&design);
        let precision_spectrum = spectrum_extremes(&precision);
        let prior_spectrum = spectrum_extremes(prior);
        let assembly = gamma(n + 1) * frobenius(&(&abs_prior + &gram_terms))
            + 2.0 * whitening * frobenius(&gram_terms);
        let perturbation = assembly + chol(p, precision_spectrum.1);
        let rhs = design.t().dot(&whitened);
        let rhs_assembly =
            (gamma(n) + 2.0 * whitening) * euclidean(&abs_design.t().dot(&whitened.mapv(f64::abs)));
        let mean = precision
            .cholesky(Side::Lower)
            .expect("fixture posterior precision is SPD")
            .solvevec(&rhs);
        let abs_mean = mean.mapv(f64::abs);
        let mean_error = (perturbation * euclidean(&mean) + rhs_assembly) / precision_spectrum.0;
        let residual = &whitened - &design.dot(&mean);
        let residual_norm = euclidean(&residual);
        let residual_error =
            gamma(p + 1) * euclidean(&(&whitened.mapv(f64::abs) + &abs_design.dot(&abs_mean)));
        let prior_energy = mean.dot(&prior.dot(&mean));
        let quadratic = residual_norm * residual_norm + prior_energy;
        let quadratic_band = precision_spectrum.1 * mean_error * mean_error
            + 2.0 * residual_norm * residual_error
            + residual_error * residual_error
            + gamma(n + 1) * residual_norm * residual_norm
            + gamma(p * p + p + 1) * abs_mean.dot(&abs_prior.dot(&abs_mean))
            + gamma(2) * quadratic
            + 2.0
                * whitening
                * residual_norm
                * (euclidean(&whitened) + frobenius(&abs_design) * euclidean(&mean));
        let noise_log_sum: f64 = noise.iter().map(|variance| variance.ln().abs()).sum();
        let log_det_band = p as f64 * perturbation / precision_spectrum.0
            + gamma(2 * p) * p as f64 * log_magnitude(precision_spectrum)
            + p as f64 * chol(p, prior_spectrum.1) / prior_spectrum.0
            + gamma(2 * p) * p as f64 * log_magnitude(prior_spectrum)
            + gamma(n) * noise_log_sum
            + gamma(2)
                * (noise_log_sum
                    + p as f64 * log_magnitude(precision_spectrum)
                    + p as f64 * log_magnitude(prior_spectrum));
        let log_det_magnitude = noise_log_sum
            + p as f64 * (log_magnitude(precision_spectrum) + log_magnitude(prior_spectrum));
        0.5 * (quadratic_band + log_det_band)
            + gamma(4) * (quadratic + log_det_magnitude + n as f64 * (2.0 * PI).ln())
    }

    /// Band of [`compare_reuse`]'s log Bayes factor: the mixture over alignments is 1-Lipschitz in each alignment's log
    /// evidence, the specialized parts add, and the final combination rounds a few more times.
    fn log_bayes_factor_band(
        contexts: &[ContextObservations<'_>],
        alignments: &[SharedAlignment],
        prior: &Array2<f64>,
        comparison: &ReuseComparison,
    ) -> f64 {
        let mut shared_band = 0.0_f64;
        let total_weight: f64 = alignments
            .iter()
            .map(|alignment| alignment.prior_weight)
            .sum();
        let mut mixture_magnitude = 0.0;
        for (alignment, log_evidence) in alignments.iter().zip(&comparison.alignment_log_evidence) {
            let (design, response, noise) =
                shared_design(contexts, &alignment.adapters).expect("fixture alignment stacks");
            shared_band =
                shared_band.max(dual_log_evidence_band(&design, &response, &noise, prior));
            mixture_magnitude +=
                log_evidence.abs() + (alignment.prior_weight / total_weight).ln().abs();
        }
        let specialized_band: f64 = contexts
            .iter()
            .map(|context| {
                dual_log_evidence_band(
                    &context.basis.to_owned(),
                    &context.response.to_owned(),
                    &context.noise_variance.to_owned(),
                    prior,
                )
            })
            .sum();
        shared_band
            + specialized_band
            + accumulation_growth(alignments.len() + contexts.len() + 4)
                * (mixture_magnitude
                    + comparison.log_evidence_shared.abs()
                    + comparison.log_evidence_specialized.abs())
    }

    /// `x ∈ {−1, 0, 1}` repeated `repeats` times, with the discrete-orthogonal quadratic basis `1, x, 3x² − 2`, whose
    /// Gram is `repeats·diag(3, 2, 6)`.
    fn orthogonal_quadratic_basis(repeats: usize) -> Array2<f64> {
        Array2::from_shape_fn((3 * repeats, 3), |(row, col)| {
            let x = (row % 3) as f64 - 1.0;
            match col {
                0 => 1.0,
                1 => x,
                _ => 3.0 * x * x - 2.0,
            }
        })
    }

    /// `sign·(x² + ε)` at the points of [`orthogonal_quadratic_basis`], with the deterministic dyadic perturbation
    /// `ε = amplitude·(((7·row) mod 5) − 2)/2`.
    fn quadratic_response(repeats: usize, sign: f64, amplitude: f64) -> Array1<f64> {
        Array1::from_shape_fn(3 * repeats, |row| {
            let x = (row % 3) as f64 - 1.0;
            let perturbation = amplitude * (((7 * row) % 5) as f64 - 2.0) / 2.0;
            sign * (x * x + perturbation)
        })
    }

    fn contexts_of<'a>(
        bases: &'a [Array2<f64>],
        responses: &'a [Array1<f64>],
        noise: &'a [Array1<f64>],
    ) -> Vec<ContextObservations<'a>> {
        bases
            .iter()
            .zip(responses)
            .zip(noise)
            .map(|((basis, response), noise_variance)| ContextObservations {
                basis: basis.view(),
                response: response.view(),
                noise_variance: noise_variance.view(),
            })
            .collect()
    }

    /// Coordinate-wise closed form of the declared-noise log Bayes factor for two contexts on one basis with orthogonal
    /// columns `φ_j`, prior `Q = diag(q)` and noise `R = σ²I`. Everything is diagonal, so coordinate `j` contributes
    /// `½[b_j²/(q_j + 2g_j) − (b₁ⱼ² + b₂ⱼ²)/(q_j + g_j)] + ½[2 ln(q_j + g_j) − ln(q_j + 2g_j) − ln q_j]` with
    /// `g_j = ‖φ_j‖²/σ²`, `b_cj = φ_jᵀy_c/σ²` and `b_j = b₁ⱼ + b₂ⱼ`. It shares no code with the Cholesky path. Returns
    /// the value and its rounding band.
    fn orthogonal_basis_log_bayes_factor(
        basis: &Array2<f64>,
        first: &Array1<f64>,
        second: &Array1<f64>,
        noise: f64,
        prior_diagonal: &[f64],
    ) -> (f64, f64) {
        let gamma = accumulation_growth;
        let n = basis.nrows();
        let mut value = 0.0;
        let mut band = 0.0;
        let mut magnitude = 0.0;
        for (j, &q) in prior_diagonal.iter().enumerate() {
            let column = basis.column(j);
            let g = column.dot(&column) / noise;
            let b1 = column.dot(first) / noise;
            let b2 = column.dot(second) / noise;
            let b = b1 + b2;
            let shared_term = b * b / (q + 2.0 * g);
            let specialized_term = (b1 * b1 + b2 * b2) / (q + g);
            let occam_terms = [2.0 * (q + g).ln(), (q + 2.0 * g).ln(), q.ln()];
            let contribution = 0.5
                * (shared_term - specialized_term + occam_terms[0]
                    - occam_terms[1]
                    - occam_terms[2]);
            value += contribution;
            magnitude += contribution.abs();
            // `g` and `b` are inner products of length `n` and one division; each quotient squares and divides a few
            // more times. A relative error `δ` in a logarithm's argument moves it by `δ`.
            band += gamma(2 * n + 8) * (shared_term.abs() + specialized_term.abs())
                + 4.0 * gamma(n + 4)
                + gamma(3) * occam_terms.iter().map(|term| term.abs()).sum::<f64>();
        }
        (value, band + gamma(3 * prior_diagonal.len()) * magnitude)
    }

    #[test]
    fn identical_quadratic_responses_favour_sharing_and_opposite_responses_favour_specialization() {
        let repeats = 10;
        let noise = 0.01;
        let bases = [
            orthogonal_quadratic_basis(repeats),
            orthogonal_quadratic_basis(repeats),
        ];
        let noise_variances = [
            Array1::from_elem(3 * repeats, noise),
            Array1::from_elem(3 * repeats, noise),
        ];
        let prior = Array2::<f64>::eye(3);
        let alignments = [SharedAlignment::identity(2, 3)];
        let first = quadratic_response(repeats, 1.0, 0.0);

        for (label, sign, favours_sharing) in [("identical", 1.0, true), ("opposite", -1.0, false)]
        {
            let responses = [first.clone(), quadratic_response(repeats, sign, 0.0)];
            let contexts = contexts_of(&bases, &responses, &noise_variances);
            let comparison =
                compare_reuse(&contexts, prior.view(), &alignments, 0.5).expect("reuse comparison");
            let (closed_form, closed_form_band) = orthogonal_basis_log_bayes_factor(
                &bases[0],
                &responses[0],
                &responses[1],
                noise,
                &[1.0; 3],
            );
            let band = closed_form_band
                + log_bayes_factor_band(&contexts, &alignments, &prior, &comparison);

            assert!(
                (comparison.log_bayes_factor - closed_form).abs() <= band,
                "{label}: log Bayes factor {:e} against the orthogonal closed form {closed_form:e}, band {band:e}",
                comparison.log_bayes_factor
            );
            if favours_sharing {
                assert!(
                    comparison.log_bayes_factor > band,
                    "{label} responses must favour sharing: log Bayes factor {:e}, band {band:e}",
                    comparison.log_bayes_factor
                );
            } else {
                assert!(
                    comparison.log_bayes_factor < -band,
                    "{label} responses must favour specialization: log Bayes factor {:e}, band {band:e}",
                    comparison.log_bayes_factor
                );
            }
            assert_eq!(
                comparison.posterior_share_probability > 0.5,
                favours_sharing,
                "{label}: posterior share probability {} at an even structural prior",
                comparison.posterior_share_probability
            );
        }
    }

    /// A nonorthogonal two-context fixture with dyadic basis and prior entries, so a dyadic change of basis transforms it
    /// exactly.
    fn nonorthogonal_fixture() -> (
        [Array2<f64>; 2],
        [Array1<f64>; 2],
        [Array1<f64>; 2],
        Array2<f64>,
    ) {
        let bases = [
            array![
                [1.0, 0.5, 0.25],
                [1.0, -0.25, 0.0625],
                [1.0, 1.25, 1.5625],
                [1.0, -1.0, 1.0],
                [1.0, 0.125, 0.015625]
            ],
            array![
                [1.0, 0.75, -0.25],
                [0.5, -0.5, 0.75],
                [1.0, 0.0, 1.0],
                [-0.25, 1.5, 0.5]
            ],
        ];
        let responses = [
            array![0.5, -0.25, 1.0, 0.75, 0.125],
            array![0.25, -0.5, 0.75, 0.5],
        ];
        let noise = [
            array![0.125, 0.25, 0.125, 0.5, 0.25],
            array![0.25, 0.125, 0.25, 0.5],
        ];
        let prior = array![[2.0, 0.5, -0.25], [0.5, 1.5, 0.25], [-0.25, 0.25, 1.0]];
        (bases, responses, noise, prior)
    }

    #[test]
    fn bayes_factor_is_invariant_under_a_consistent_basis_change() {
        let (bases, responses, noise, prior) = nonorthogonal_fixture();
        let contexts = contexts_of(&bases, &responses, &noise);
        let flip = array![[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        let alignments = [
            SharedAlignment {
                adapters: vec![Array2::eye(3), Array2::eye(3)],
                prior_weight: 2.0,
            },
            SharedAlignment {
                adapters: vec![Array2::eye(3), flip],
                prior_weight: 1.0,
            },
        ];
        // A nonorthogonal dyadic T with its exact inverse: β = Tβ̃, Φ̃ = ΦT, Q̃ = TᵀQT, Ã = T⁻¹AT.
        let t = array![[2.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.5]];
        let t_inverse = array![[0.5, -0.5, -1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 2.0]];
        assert_eq!(t.dot(&t_inverse), Array2::<f64>::eye(3));
        let transformed_bases = [bases[0].dot(&t), bases[1].dot(&t)];
        let transformed_prior = t.t().dot(&prior).dot(&t);
        let transformed_alignments: Vec<SharedAlignment> = alignments
            .iter()
            .map(|alignment| SharedAlignment {
                adapters: alignment
                    .adapters
                    .iter()
                    .map(|adapter| t_inverse.dot(adapter).dot(&t))
                    .collect(),
                prior_weight: alignment.prior_weight,
            })
            .collect();
        let transformed_contexts = contexts_of(&transformed_bases, &responses, &noise);

        let original =
            compare_reuse(&contexts, prior.view(), &alignments, 0.5).expect("original comparison");
        let transformed = compare_reuse(
            &transformed_contexts,
            transformed_prior.view(),
            &transformed_alignments,
            0.5,
        )
        .expect("transformed comparison");
        let band = log_bayes_factor_band(&contexts, &alignments, &prior, &original)
            + log_bayes_factor_band(
                &transformed_contexts,
                &transformed_alignments,
                &transformed_prior,
                &transformed,
            );
        assert!(
            (original.log_bayes_factor - transformed.log_bayes_factor).abs() <= band,
            "a consistent basis change moved the log Bayes factor from {:e} to {:e}, band {band:e}",
            original.log_bayes_factor,
            transformed.log_bayes_factor
        );

        // Positive control: changing the basis without transforming the prior is a different function prior, and it
        // moves the Bayes factor by more than the band.
        let inconsistent = compare_reuse(
            &transformed_contexts,
            prior.view(),
            &transformed_alignments,
            0.5,
        )
        .expect("inconsistent comparison");
        let inconsistent_band = log_bayes_factor_band(&contexts, &alignments, &prior, &original)
            + log_bayes_factor_band(
                &transformed_contexts,
                &transformed_alignments,
                &prior,
                &inconsistent,
            );
        assert!(
            (original.log_bayes_factor - inconsistent.log_bayes_factor).abs() > inconsistent_band,
            "an inconsistent basis change left the log Bayes factor at {:e} against {:e}, band {inconsistent_band:e}",
            inconsistent.log_bayes_factor,
            original.log_bayes_factor
        );
    }

    #[test]
    fn a_declared_alignment_is_charged_its_prior_mass() {
        let repeats = 10;
        let noise = 0.01;
        let bases = [
            orthogonal_quadratic_basis(repeats),
            orthogonal_quadratic_basis(repeats),
        ];
        let noise_variances = [
            Array1::from_elem(3 * repeats, noise),
            Array1::from_elem(3 * repeats, noise),
        ];
        let prior = Array2::<f64>::eye(3);
        let opposite_responses = [
            quadratic_response(repeats, 1.0, 0.0),
            quadratic_response(repeats, -1.0, 0.0),
        ];
        let identical_responses = [
            quadratic_response(repeats, 1.0, 0.0),
            quadratic_response(repeats, 1.0, 0.0),
        ];
        let opposite = contexts_of(&bases, &opposite_responses, &noise_variances);
        let identical = contexts_of(&bases, &identical_responses, &noise_variances);
        let identity = SharedAlignment::identity(2, 3);
        let sign_flip = SharedAlignment {
            adapters: vec![Array2::eye(3), -Array2::<f64>::eye(3)],
            prior_weight: 1.0,
        };
        let both = [identity.clone(), sign_flip.clone()];
        let flip_only = [sign_flip];
        let identity_only = [identity];

        let charged =
            compare_reuse(&opposite, prior.view(), &both, 0.5).expect("charged comparison");
        let free = compare_reuse(&opposite, prior.view(), &flip_only, 0.5)
            .expect("post-hoc flip comparison");
        let reference = compare_reuse(&identical, prior.view(), &identity_only, 0.5)
            .expect("identical comparison");
        let band = log_bayes_factor_band(&opposite, &both, &prior, &charged)
            + log_bayes_factor_band(&opposite, &flip_only, &prior, &free)
            + log_bayes_factor_band(&identical, &identity_only, &prior, &reference);

        // Reading the second context through `−I` turns opposite responses into identical ones exactly.
        assert!(
            (free.log_bayes_factor - reference.log_bayes_factor).abs() <= band,
            "a free sign flip gives {:e}, identical responses give {:e}, band {band:e}",
            free.log_bayes_factor,
            reference.log_bayes_factor
        );

        // With both alignments declared at equal mass the mixture pays ln 2 for the flip, less only what the identity
        // alignment still explains: log p(y | shared) ∈ [ℓ_flip − ln 2, ℓ_flip − ln 2 + ln(1 + e^{ℓ_id − ℓ_flip})].
        let identity_evidence = charged.alignment_log_evidence[0];
        let flip_evidence = charged.alignment_log_evidence[1];
        let lower = flip_evidence - std::f64::consts::LN_2;
        let upper = lower + (identity_evidence - flip_evidence).exp().ln_1p();
        assert!(
            charged.log_evidence_shared >= lower - band
                && charged.log_evidence_shared <= upper + band,
            "charged shared evidence {:e} outside [{lower:e}, {upper:e}] ± {band:e}",
            charged.log_evidence_shared
        );
        assert!(
            charged.log_bayes_factor < free.log_bayes_factor - band,
            "declaring the flip among two alignments must cost evidence: charged {:e}, free {:e}, band {band:e}",
            charged.log_bayes_factor,
            free.log_bayes_factor
        );
        assert!(
            charged.log_bayes_factor > band,
            "the charged flip still favours sharing on this fixture: log Bayes factor {:e}, band {band:e}",
            charged.log_bayes_factor
        );
    }

    #[test]
    fn reml_arm_favours_sharing_for_identical_and_specialization_for_opposite_responses() {
        let repeats = 10;
        let bases = [
            orthogonal_quadratic_basis(repeats),
            orthogonal_quadratic_basis(repeats),
        ];
        let noise_shapes = [
            Array1::<f64>::ones(3 * repeats),
            Array1::<f64>::ones(3 * repeats),
        ];
        let penalties = [Array2::<f64>::eye(3)];
        let no_null_space = Array2::<f64>::zeros((3, 0));
        let alignments = [SharedAlignment::identity(2, 3)];
        // A deterministic perturbation keeps the profiled deviance away from a perfect fit.
        let first = quadratic_response(repeats, 1.0, 0.125);

        for (label, sign, favours_sharing) in [("identical", 1.0, true), ("opposite", -1.0, false)]
        {
            let responses = [first.clone(), first.mapv(|value| sign * value)];
            let contexts = contexts_of(&bases, &responses, &noise_shapes);
            let reml = compare_reuse_reml(
                &contexts,
                &penalties,
                no_null_space.view(),
                &alignments,
                0.5,
            )
            .expect("REML comparison");
            let resolution = reml
                .log_bayes_factor_resolution
                .expect("every fit is audited with positive interior curvature");
            let log_bayes_factor = reml.comparison.log_bayes_factor;
            if favours_sharing {
                assert!(
                    log_bayes_factor > resolution,
                    "{label}: REML log Bayes factor {log_bayes_factor:e} must favour sharing beyond {resolution:e}"
                );
            } else {
                assert!(
                    log_bayes_factor < -resolution,
                    "{label}: REML log Bayes factor {log_bayes_factor:e} must favour specialization beyond {resolution:e}"
                );
            }
            assert!(reml.specialized_lambdas[0] > 0.0 && reml.alignment_lambdas[0][0] > 0.0);
        }
    }

    /// Two contexts on [`orthogonal_quadratic_basis`] with unit noise shape: `y₁ = x² + x/2 + ε` and
    /// `y₂ = sign·y₁ + offset`. Both penalized coordinates carry signal well above the noise, so no REML strength rails.
    fn offset_contexts_fixture(
        sign: f64,
        offset: f64,
    ) -> ([Array2<f64>; 2], [Array1<f64>; 2], [Array1<f64>; 2]) {
        let repeats = 10;
        let slope = Array1::from_shape_fn(3 * repeats, |row| ((row % 3) as f64 - 1.0) / 2.0);
        let first = quadratic_response(repeats, 1.0, 0.125) + &slope;
        let second = first.mapv(|value| sign * value + offset);
        (
            [
                orthogonal_quadratic_basis(repeats),
                orthogonal_quadratic_basis(repeats),
            ],
            [first, second],
            [
                Array1::<f64>::ones(3 * repeats),
                Array1::<f64>::ones(3 * repeats),
            ],
        )
    }

    /// The penalty set `{diag(0, 1, 0), diag(0, 0, 1)}`, which leaves the intercept `e₁` unpenalized.
    fn intercept_free_penalties() -> ([Array2<f64>; 2], Array2<f64>) {
        (
            [
                array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            array![[1.0], [0.0], [0.0]],
        )
    }

    #[test]
    fn reml_arm_keeps_each_contexts_null_space_coefficients_in_both_hypotheses() {
        let (penalties, intercept) = intercept_free_penalties();
        let alignments = [SharedAlignment::identity(2, 3)];

        // One shape read at two offsets: the per-context intercepts absorb the offset, so sharing wins.
        let (bases, responses, noise) = offset_contexts_fixture(1.0, 5.0);
        let contexts = contexts_of(&bases, &responses, &noise);
        let offset_shape =
            compare_reuse_reml(&contexts, &penalties, intercept.view(), &alignments, 0.5)
                .expect("REML comparison with a declared intercept");
        let resolution = offset_shape
            .log_bayes_factor_resolution
            .expect("every fit is audited with positive interior curvature");
        assert!(
            offset_shape.comparison.log_bayes_factor > resolution,
            "one shape at two offsets: REML log Bayes factor {:e} must favour sharing beyond {resolution:e}",
            offset_shape.comparison.log_bayes_factor
        );
        assert_eq!(offset_shape.specialized_lambdas.len(), 2);
        assert_eq!(offset_shape.alignment_rho_placement[0].len(), 2);

        // Control: sharing the intercept too (a full-rank penalty, no null space) cannot fit both offsets.
        let full_rank = [Array2::<f64>::eye(3)];
        let shared_intercept = compare_reuse_reml(
            &contexts,
            &full_rank,
            Array2::<f64>::zeros((3, 0)).view(),
            &alignments,
            0.5,
        )
        .expect("REML comparison with a full-rank penalty");
        let control_resolution = shared_intercept
            .log_bayes_factor_resolution
            .expect("every fit is audited with positive interior curvature");
        assert!(
            shared_intercept.comparison.log_bayes_factor < -control_resolution,
            "a shared intercept across a 5-unit offset: REML log Bayes factor {:e} must favour specialization \
             beyond {control_resolution:e}",
            shared_intercept.comparison.log_bayes_factor
        );

        // Control: opposite shapes still favour specialization with the intercepts free.
        let (bases, responses, noise) = offset_contexts_fixture(-1.0, 5.0);
        let contexts = contexts_of(&bases, &responses, &noise);
        let opposite =
            compare_reuse_reml(&contexts, &penalties, intercept.view(), &alignments, 0.5)
                .expect("REML comparison with a declared intercept");
        let opposite_resolution = opposite
            .log_bayes_factor_resolution
            .expect("every fit is audited with positive interior curvature");
        assert!(
            opposite.comparison.log_bayes_factor < -opposite_resolution,
            "opposite shapes: REML log Bayes factor {:e} must favour specialization beyond {opposite_resolution:e}",
            opposite.comparison.log_bayes_factor
        );
    }

    #[test]
    fn reml_bayes_factor_is_invariant_when_a_basis_change_rescales_the_null_space() {
        let (penalties, intercept) = intercept_free_penalties();
        let (bases, responses, noise) = offset_contexts_fixture(1.0, 5.0);
        let alignments = [SharedAlignment::identity(2, 3)];
        let contexts = contexts_of(&bases, &responses, &noise);
        let original =
            compare_reuse_reml(&contexts, &penalties, intercept.view(), &alignments, 0.5)
                .expect("REML comparison");

        // `T` is upper triangular with `Te₁ = e₁/4`, so the transformed null space `T⁻¹e₁` is `e₁` again while each
        // context's fixed-effect column `Φ_cTe₁ = Φ_ce₁/4` shrinks by `s = 1/4`. The flat prior on `m·k = 2` fixed-effect
        // coordinates then moves both log evidences by `−m·k·ln s = 2 ln 4` and leaves the Bayes factor alone. `T` also
        // shears the complement off `e₁⊥`, which the flat coordinates absorb.
        let transform = array![[0.25, 0.5, 0.0], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]];
        let transformed_bases = [bases[0].dot(&transform), bases[1].dot(&transform)];
        let transformed_penalties = [
            transform.t().dot(&penalties[0]).dot(&transform),
            transform.t().dot(&penalties[1]).dot(&transform),
        ];
        let transformed_contexts = contexts_of(&transformed_bases, &responses, &noise);
        let transformed = compare_reuse_reml(
            &transformed_contexts,
            &transformed_penalties,
            intercept.view(),
            &alignments,
            0.5,
        )
        .expect("REML comparison in transformed coordinates");

        let band = original
            .log_bayes_factor_resolution
            .expect("every fit is audited with positive interior curvature")
            + transformed
                .log_bayes_factor_resolution
                .expect("every fit is audited with positive interior curvature");
        let predicted_shift = 2.0 * 4.0_f64.ln();
        assert!(
            predicted_shift > band,
            "positive control: the predicted evidence shift {predicted_shift:e} must exceed the band {band:e}"
        );
        for (label, before, after) in [
            (
                "shared",
                original.comparison.log_evidence_shared,
                transformed.comparison.log_evidence_shared,
            ),
            (
                "specialized",
                original.comparison.log_evidence_specialized,
                transformed.comparison.log_evidence_specialized,
            ),
        ] {
            assert!(
                (after - before - predicted_shift).abs() <= band,
                "{label} log evidence moved by {:e}, predicted {predicted_shift:e} within {band:e}",
                after - before
            );
        }
        assert!(
            (transformed.comparison.log_bayes_factor - original.comparison.log_bayes_factor).abs()
                <= band,
            "log Bayes factor {:e} vs {:e} must agree within {band:e}",
            transformed.comparison.log_bayes_factor,
            original.comparison.log_bayes_factor
        );
    }

    #[test]
    fn reml_arm_refuses_an_undeclared_or_unannihilated_null_space() {
        let (penalties, intercept) = intercept_free_penalties();
        let (bases, responses, noise) = offset_contexts_fixture(1.0, 0.0);
        let contexts = contexts_of(&bases, &responses, &noise);
        let alignments = [SharedAlignment::identity(2, 3)];

        // Positive control: the declared intercept computes.
        assert!(
            compare_reuse_reml(&contexts, &penalties, intercept.view(), &alignments, 0.5).is_ok()
        );
        // The same penalties with no declared null space leave the intercept unpenalized in both fits.
        assert!(matches!(
            compare_reuse_reml(
                &contexts,
                &penalties,
                Array2::<f64>::zeros((3, 0)).view(),
                &alignments,
                0.5
            ),
            Err(ReuseError::Reml(_))
        ));
        // The published predicate agrees: the intercept holds for both penalties, the slope fails only the first.
        let intercept_measured =
            null_space_annihilation(&penalties, intercept.view()).expect("intercept measure");
        assert!(intercept_measured.iter().all(NullSpaceAnnihilation::holds));
        let slope = array![[0.0], [1.0], [0.0]];
        let slope_measured = null_space_annihilation(&penalties, slope.view()).expect("slope measure");
        assert!(
            !slope_measured[0].holds() && slope_measured[1].holds(),
            "the slope escapes diag(0, 1, 0) only: {slope_measured:?}"
        );
        // A declared direction the penalties reach.
        assert!(matches!(
            compare_reuse_reml(&contexts, &penalties, slope.view(), &alignments, 0.5),
            Err(ReuseError::NullSpaceNotAnnihilated { penalty: 0, .. })
        ));
    }

    #[test]
    fn an_improper_prior_and_undeclared_structure_are_refused() {
        let (bases, responses, noise, prior) = nonorthogonal_fixture();
        let contexts = contexts_of(&bases, &responses, &noise);
        let identity = [SharedAlignment::identity(2, 3)];

        // Positive control: the same fixture with a proper prior and declared structure is computable.
        assert!(compare_reuse(&contexts, prior.view(), &identity, 0.5).is_ok());

        let singular = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            compare_reuse(&contexts, singular.view(), &identity, 0.5),
            Err(ReuseError::Evidence(GaussianMarginalError::ImproperPrior(
                _
            )))
        ));
        assert!(matches!(
            compare_reuse(&contexts, prior.view(), &identity, 1.0),
            Err(ReuseError::InvalidInput(_))
        ));
        assert!(matches!(
            compare_reuse(&contexts[..1], prior.view(), &identity, 0.5),
            Err(ReuseError::InvalidInput(_))
        ));
        let one_adapter = [SharedAlignment {
            adapters: vec![Array2::eye(3)],
            prior_weight: 1.0,
        }];
        assert!(matches!(
            compare_reuse(&contexts, prior.view(), &one_adapter, 0.5),
            Err(ReuseError::InvalidInput(_))
        ));
        let silent_noise = [array![0.125, 0.0, 0.125, 0.5, 0.25], noise[1].clone()];
        let silent_contexts = contexts_of(&bases, &responses, &silent_noise);
        assert!(matches!(
            compare_reuse(&silent_contexts, prior.view(), &identity, 0.5),
            Err(ReuseError::Evidence(GaussianMarginalError::InvalidInput(_)))
        ));
    }
}

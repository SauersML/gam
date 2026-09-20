//! Exact Gaussian evidence of a conditionally Gaussian block under a declared proper prior (#2933 comment 5715551157
//! §A).
//!
//! Model: `y = Φβ + ε` with declared noise `ε ~ N(0, R)`, `R = diag(r)`, and a declared proper prior `β ~ N(0, Q⁻¹)`,
//! `Q ≻ 0`. Integrating the coefficients exactly,
//!
//! ```text
//! −log p(y) = ½[yᵀC⁻¹y + log|C| + n·log 2π],                              C = R + ΦQ⁻¹Φᵀ               (primal)
//!           = ½[yᵀR⁻¹y − bᵀH⁻¹b + log|R| + log|H| − log|Q| + n·log 2π],   H = Q + ΦᵀR⁻¹Φ, b = ΦᵀR⁻¹y  (dual)
//! ```
//!
//! and the posterior is `β | y ~ N(H⁻¹b, H⁻¹)`. The primal form factors `C` at `n × n` and the dual form factors `H` at
//! `p × p`; [`GaussianMarginalModel::evidence`] factors the smaller. Every factorization is a strict, unjittered
//! Cholesky whose solves are certified against the unperturbed matrix.
//!
//! This module is the one owner of that evaluator. #2946's reuse-versus-specialization comparison consumes it, and a
//! conditionally Gaussian SAE block (#2933) should call it rather than write another.
//!
//! **Invariances**, pinned by the tests below:
//! - any invertible `T`, nonorthogonal included, with `Φ → ΦT` and `Q → TᵀQT` leaves `ΦQ⁻¹Φᵀ` unchanged, hence the
//!   evidence and the posterior function `Φβ̂`;
//! - duplicating the basis, `φβ → φβ₁ + φβ₂` with each copy at half the prior variance (`[Φ, Φ]` with
//!   `Q → diag(2Q, 2Q)`), leaves `C` unchanged, so a parameter-count charge added to this evidence is a double count.
//!
//! **Scope.** The prior is proper and the noise is declared. An improper fixed-effect space integrates to an arbitrary
//! constant and needs REML with a declared null space, and an unknown noise scale is profiled by
//! [`crate::gaussian_reml`]. Neither belongs here, so a prior precision that fails strict Cholesky is refused.

use std::f64::consts::PI;

use gam_linalg::faer_ndarray::{fast_ab, fast_ata};
use gam_linalg::matrix::symmetrize_in_place;
use gam_linalg::roundoff::{SymmetricAssembly, factor_rank_partition};
use gam_linalg::utils::{
    CertifiedSymmetricSolveError, certified_spd_factorize, validate_finite_symmetric_matrix,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Sufficient statistics of a zero-mean Gaussian marginal `y ~ N(0, C)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianEvidenceParts {
    /// `yᵀC⁻¹y`.
    pub quadratic: f64,
    /// `log|C|`.
    pub log_det: f64,
    /// `dim y`.
    pub observations: usize,
}

impl GaussianEvidenceParts {
    /// The parts of two independent marginals: their joint covariance is block diagonal, so every statistic adds.
    pub fn independent_sum(self, other: Self) -> Self {
        Self {
            quadratic: self.quadratic + other.quadratic,
            log_det: self.log_det + other.log_det,
            observations: self.observations + other.observations,
        }
    }

    /// `log p(y) = −½(yᵀC⁻¹y + log|C| + n·log 2π)`.
    pub fn log_evidence(self) -> f64 {
        -0.5 * (self.quadratic + self.log_det + self.observations as f64 * (2.0 * PI).ln())
    }
}

/// Provenance of a declared prior precision `Q` (#4350). `Q` is an input, not an accumulation of ours, so no rounding
/// of this module can have split its triangles: a declared symmetric matrix must be exactly symmetric, and a caller who
/// assembled it by a full GEMM mirrors or symmetrizes it before declaring it.
const DECLARED_PRIOR_ASSEMBLY: SymmetricAssembly = SymmetricAssembly::Mirrored;

/// Provenance of the posterior precision `Q + WᵀW`: `fast_ata` mirrors its triangular Gram, and the entrywise sum with
/// the exactly symmetric declared prior adds the same two operands in both triangles.
const POSTERIOR_PRECISION_ASSEMBLY: SymmetricAssembly =
    SymmetricAssembly::Mirrored.psd_sum(DECLARED_PRIOR_ASSEMBLY);

/// Why a declared-prior Gaussian evidence could not be computed.
#[derive(Debug, PartialEq)]
pub enum GaussianMarginalError {
    /// A shape, finiteness, symmetry or positivity requirement failed.
    InvalidInput(String),
    /// The declared prior precision failed its strict Cholesky factorization.
    ImproperPrior(CertifiedSymmetricSolveError),
    /// A posterior precision or marginal covariance failed to factor, or a solve against it failed its certificate.
    Solve {
        stage: &'static str,
        error: CertifiedSymmetricSolveError,
    },
    /// The exact constraint `Aβ = y` resolves fewer singular values than it has rows, so `AQ⁻¹Aᵀ` is singular: the
    /// constraint repeats or contradicts itself.
    RankDeficientConstraint { rows: usize, resolved_rank: usize },
}

impl std::fmt::Display for GaussianMarginalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::ImproperPrior(error) => write!(
                f,
                "declared-prior Gaussian evidence refuses an improper prior precision ({error}); an improper \
                 fixed-effect space needs REML with a declared null space"
            ),
            Self::Solve { stage, error } => {
                write!(f, "declared-prior Gaussian evidence: {stage}: {error}")
            }
            Self::RankDeficientConstraint {
                rows,
                resolved_rank,
            } => write!(
                f,
                "declared-prior Gaussian evidence refuses a rank-deficient exact constraint: {rows} rows resolve only \
                 {resolved_rank} singular values above their backward-error band"
            ),
        }
    }
}

impl std::error::Error for GaussianMarginalError {}

/// A validated declared-prior Gaussian block, held in whitened coordinates `W = R^{−1/2}Φ`, `ỹ = R^{−1/2}y`.
#[derive(Clone, Debug)]
pub struct GaussianMarginalModel {
    design: Array2<f64>,
    response: Array1<f64>,
    prior_precision: Array2<f64>,
    prior_log_det: f64,
    log_noise_det: f64,
}

impl GaussianMarginalModel {
    /// Validate `y = Φβ + ε` with `ε ~ N(0, diag(noise_variance))` and `β ~ N(0, prior_precision⁻¹)`, refusing an
    /// improper prior. The declared `prior_precision` must be exactly symmetric ([`DECLARED_PRIOR_ASSEMBLY`]).
    pub fn new(
        basis: ArrayView2<'_, f64>,
        response: ArrayView1<'_, f64>,
        noise_variance: ArrayView1<'_, f64>,
        prior_precision: ArrayView2<'_, f64>,
    ) -> Result<Self, GaussianMarginalError> {
        let (rows, cols) = basis.dim();
        if rows == 0 {
            return Err(GaussianMarginalError::InvalidInput(
                "declared-prior Gaussian evidence needs at least one observation".to_string(),
            ));
        }
        if response.len() != rows || noise_variance.len() != rows {
            return Err(GaussianMarginalError::InvalidInput(format!(
                "the basis has {rows} rows but there are {} responses and {} noise variances",
                response.len(),
                noise_variance.len()
            )));
        }
        if basis
            .iter()
            .chain(response.iter())
            .any(|value| !value.is_finite())
        {
            return Err(GaussianMarginalError::InvalidInput(
                "the basis and responses must be finite".to_string(),
            ));
        }
        if let Some((row, variance)) = noise_variance
            .iter()
            .enumerate()
            .find(|(_, variance)| !(variance.is_finite() && **variance > 0.0))
        {
            return Err(GaussianMarginalError::InvalidInput(format!(
                "the noise variance at row {row} must be finite and positive; got {variance}"
            )));
        }
        let prior = prior_precision.to_owned();
        validate_finite_symmetric_matrix(
            &prior,
            DECLARED_PRIOR_ASSEMBLY,
            "declared prior precision",
        )
        .map_err(|error| GaussianMarginalError::InvalidInput(error.to_string()))?;
        if prior.nrows() != cols {
            return Err(GaussianMarginalError::InvalidInput(format!(
                "the basis has {cols} columns but the prior precision is {0} x {0}",
                prior.nrows()
            )));
        }
        let prior_log_det =
            certified_spd_factorize(&prior, DECLARED_PRIOR_ASSEMBLY, "declared prior precision")
                .map_err(GaussianMarginalError::ImproperPrior)?
                .log_det();

        let mut design = basis.to_owned();
        let mut whitened = response.to_owned();
        let mut log_noise_det = 0.0;
        for (row, &variance) in noise_variance.iter().enumerate() {
            let scale = variance.sqrt().recip();
            design.row_mut(row).mapv_inplace(|value| value * scale);
            whitened[row] *= scale;
            log_noise_det += variance.ln();
        }
        Ok(Self {
            design,
            response: whitened,
            prior_precision: prior,
            prior_log_det,
            log_noise_det,
        })
    }

    /// `n`, the number of observations.
    pub fn observations(&self) -> usize {
        self.design.nrows()
    }

    /// `p`, the number of coefficients.
    pub fn coefficients(&self) -> usize {
        self.design.ncols()
    }

    /// The evidence parts in whichever exact form factors the smaller matrix: the dual `p × p` precision when `p ≤ n`,
    /// the primal `n × n` covariance otherwise.
    pub fn evidence(&self) -> Result<GaussianEvidenceParts, GaussianMarginalError> {
        if self.coefficients() <= self.observations() {
            self.evidence_dual()
        } else {
            self.evidence_primal()
        }
    }

    /// Dual (precision) form: factor `H = Q + WᵀW` at `p × p`.
    pub fn evidence_dual(&self) -> Result<GaussianEvidenceParts, GaussianMarginalError> {
        let precision = self.posterior_precision();
        let factor = certified_spd_factorize(
            &precision,
            POSTERIOR_PRECISION_ASSEMBLY,
            "posterior precision Q + ΦᵀR⁻¹Φ",
        )
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "posterior precision",
            error,
        })?;
        let mean = factor
            .solve(&self.design.t().dot(&self.response))
            .map_err(|error| GaussianMarginalError::Solve {
                stage: "posterior mean",
                error,
            })?
            .into_solution();
        // `yᵀC⁻¹y = ‖ỹ − Wβ̂‖² + β̂ᵀQβ̂` at `β̂ = H⁻¹b`. Every term is non-negative, so the form carries no cancellation,
        // and `β̂` minimizes it, so a solve error in `β̂` enters only at second order.
        let residual = &self.response - &self.design.dot(&mean);
        Ok(GaussianEvidenceParts {
            quadratic: residual.dot(&residual) + mean.dot(&self.prior_precision.dot(&mean)),
            log_det: self.log_noise_det + factor.log_det() - self.prior_log_det,
            observations: self.observations(),
        })
    }

    /// Primal (covariance) form: factor the whitened covariance `C̃ = I + WQ⁻¹Wᵀ` at `n × n`, with
    /// `log|C| = Σ log r + log|C̃|` and `yᵀC⁻¹y = ỹᵀC̃⁻¹ỹ`.
    pub fn evidence_primal(&self) -> Result<GaussianEvidenceParts, GaussianMarginalError> {
        let prior_factor = certified_spd_factorize(
            &self.prior_precision,
            DECLARED_PRIOR_ASSEMBLY,
            "declared prior precision",
        )
        .map_err(GaussianMarginalError::ImproperPrior)?;
        let readout = prior_factor
            .solve_matrix(&self.design.t().to_owned())
            .map_err(|error| GaussianMarginalError::Solve {
                stage: "prior readout Q⁻¹Φᵀ",
                error,
            })?
            .0;
        let mut covariance = fast_ab(&self.design, &readout);
        // `WQ⁻¹Wᵀ` is symmetric analytically, but the two triangles of the computed product are separate
        // accumulations; project their rounding back onto the analytic symmetry before the strict factorization.
        symmetrize_in_place(&mut covariance);
        for index in 0..covariance.nrows() {
            covariance[[index, index]] += 1.0;
        }
        let factor = certified_spd_factorize(
            &covariance,
            SymmetricAssembly::Mirrored,
            "whitened marginal covariance R + ΦQ⁻¹Φᵀ",
        )
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "marginal covariance",
            error,
        })?;
        let weights = factor
            .solve(&self.response)
            .map_err(|error| GaussianMarginalError::Solve {
                stage: "marginal covariance solve",
                error,
            })?
            .into_solution();
        Ok(GaussianEvidenceParts {
            quadratic: self.response.dot(&weights),
            log_det: self.log_noise_det + factor.log_det(),
            observations: self.observations(),
        })
    }

    /// The coefficient posterior `β | y ~ N(H⁻¹b, H⁻¹)`.
    pub fn posterior(&self) -> Result<GaussianPosterior, GaussianMarginalError> {
        let precision = self.posterior_precision();
        let mean = certified_spd_factorize(
            &precision,
            POSTERIOR_PRECISION_ASSEMBLY,
            "posterior precision Q + ΦᵀR⁻¹Φ",
        )
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "posterior precision",
            error,
        })?
        .solve(&self.design.t().dot(&self.response))
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "posterior mean",
            error,
        })?
        .into_solution();
        Ok(GaussianPosterior { mean, precision })
    }

    fn posterior_precision(&self) -> Array2<f64> {
        let mut precision = fast_ata(&self.design);
        precision += &self.prior_precision;
        precision
    }
}

/// The coefficient posterior `β | y ~ N(H⁻¹b, H⁻¹)` of a [`GaussianMarginalModel`].
#[derive(Clone, Debug)]
pub struct GaussianPosterior {
    mean: Array1<f64>,
    precision: Array2<f64>,
}

impl GaussianPosterior {
    /// `E[β | y] = H⁻¹b`.
    pub fn mean(&self) -> ArrayView1<'_, f64> {
        self.mean.view()
    }

    /// `H = Q + ΦᵀR⁻¹Φ`.
    pub fn precision(&self) -> ArrayView2<'_, f64> {
        self.precision.view()
    }

    /// `H⁻¹B`, the posterior covariance applied to the columns of `rhs` in one certified solve. The posterior
    /// covariance of a function read by rows `Φ_*` is `Φ_*·covariance_times(Φ_*ᵀ)`.
    pub fn covariance_times(
        &self,
        rhs: &Array2<f64>,
    ) -> Result<Array2<f64>, GaussianMarginalError> {
        Ok(certified_spd_factorize(
            &self.precision,
            POSTERIOR_PRECISION_ASSEMBLY,
            "posterior precision Q + ΦᵀR⁻¹Φ",
        )
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "posterior precision",
            error,
        })?
        .solve_matrix(rhs)
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "posterior covariance solve",
            error,
        })?
        .0)
    }
}

/// The coefficient posterior under an exact linear constraint `Aβ = y` with the declared proper prior
/// `β ~ N(0, Q⁻¹)`.
///
/// `y = Aβ` is Gaussian with covariance `V = AQ⁻¹Aᵀ`, so the evidence of the constraint value is `N(y; 0, V)` and
/// `β | Aβ = y ~ N(Q⁻¹AᵀV⁻¹y, Q⁻¹ − Q⁻¹AᵀV⁻¹AQ⁻¹)`, a covariance of rank `p − n`. This is the `R → 0` limit of the
/// primal form of [`GaussianMarginalModel`]; the dual form has no such limit, because `H = Q + AᵀR⁻¹A` diverges.
#[derive(Clone, Debug)]
pub struct ExactConstraintPosterior {
    prior_precision: Array2<f64>,
    readout: Array2<f64>,
    constraint_covariance: Array2<f64>,
    mean: Array1<f64>,
    evidence: GaussianEvidenceParts,
}

impl ExactConstraintPosterior {
    /// `yᵀV⁻¹y`, `log|V|` and `n` for `y ~ N(0, V)`, `V = AQ⁻¹Aᵀ`.
    pub fn evidence(&self) -> GaussianEvidenceParts {
        self.evidence
    }

    /// `E[β | Aβ = y] = Q⁻¹AᵀV⁻¹y`.
    pub fn mean(&self) -> ArrayView1<'_, f64> {
        self.mean.view()
    }

    /// `Cov[β | Aβ = y]·B = Q⁻¹B − ZV⁻¹ZᵀB` with `Z = Q⁻¹Aᵀ`, as certified solves.
    pub fn covariance_times(
        &self,
        rhs: &Array2<f64>,
    ) -> Result<Array2<f64>, GaussianMarginalError> {
        let prior_part = certified_spd_factorize(
            &self.prior_precision,
            DECLARED_PRIOR_ASSEMBLY,
            "declared prior precision",
        )
        .map_err(GaussianMarginalError::ImproperPrior)?
        .solve_matrix(rhs)
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "prior covariance solve",
            error,
        })?
        .0;
        let projected = self.readout.t().dot(rhs);
        let correction = certified_spd_factorize(
            &self.constraint_covariance,
            SymmetricAssembly::Mirrored,
            "constraint covariance AQ⁻¹Aᵀ",
        )
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "constraint covariance",
            error,
        })?
        .solve_matrix(&projected)
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "constraint covariance solve",
            error,
        })?
        .0;
        Ok(prior_part - &fast_ab(&self.readout, &correction))
    }
}

/// Condition the declared proper prior `β ~ N(0, Q⁻¹)` on the exact linear constraint `Aβ = y` (`A` is `n × p`,
/// `n ≤ p`).
///
/// The constraint's rank is read off the singular values of `A` itself ([`factor_rank_partition`]): with `Q ≻ 0`,
/// `AQ⁻¹Aᵀ` is singular exactly when `A` is row-rank deficient, and a strict Cholesky of the computed `AQ⁻¹Aᵀ` alone
/// cannot refuse that case, because a singular pivot can round to a positive ulp. A rank-deficient `A` is refused with
/// [`GaussianMarginalError::RankDeficientConstraint`]; nothing is jittered.
pub fn condition_on_exact_constraint(
    constraint: ArrayView2<'_, f64>,
    value: ArrayView1<'_, f64>,
    prior_precision: ArrayView2<'_, f64>,
) -> Result<ExactConstraintPosterior, GaussianMarginalError> {
    let (rows, cols) = constraint.dim();
    if rows == 0 || rows > cols {
        return Err(GaussianMarginalError::InvalidInput(format!(
            "an exact constraint needs between 1 and p rows; got {rows} rows over {cols} coefficients"
        )));
    }
    if value.len() != rows {
        return Err(GaussianMarginalError::InvalidInput(format!(
            "the constraint has {rows} rows but {} values",
            value.len()
        )));
    }
    if constraint
        .iter()
        .chain(value.iter())
        .any(|entry| !entry.is_finite())
    {
        return Err(GaussianMarginalError::InvalidInput(
            "the constraint and its values must be finite".to_string(),
        ));
    }
    let prior = prior_precision.to_owned();
    validate_finite_symmetric_matrix(&prior, DECLARED_PRIOR_ASSEMBLY, "declared prior precision")
        .map_err(|error| GaussianMarginalError::InvalidInput(error.to_string()))?;
    if prior.nrows() != cols {
        return Err(GaussianMarginalError::InvalidInput(format!(
            "the constraint has {cols} columns but the prior precision is {0} x {0}",
            prior.nrows()
        )));
    }
    let constraint = constraint.to_owned();
    let resolved_rank = factor_rank_partition(&constraint)
        .map_err(|error| GaussianMarginalError::InvalidInput(error.to_string()))?
        .rank;
    if resolved_rank < rows {
        return Err(GaussianMarginalError::RankDeficientConstraint {
            rows,
            resolved_rank,
        });
    }
    let readout =
        certified_spd_factorize(&prior, DECLARED_PRIOR_ASSEMBLY, "declared prior precision")
            .map_err(GaussianMarginalError::ImproperPrior)?
            .solve_matrix(&constraint.t().to_owned())
            .map_err(|error| GaussianMarginalError::Solve {
                stage: "prior readout Q⁻¹Aᵀ",
                error,
            })?
            .0;
    let mut covariance = fast_ab(&constraint, &readout);
    // `AQ⁻¹Aᵀ` is symmetric analytically; project the rounding of its two triangles back onto that symmetry.
    symmetrize_in_place(&mut covariance);
    let value = value.to_owned();
    let (weights, log_det) = {
        let factor = certified_spd_factorize(
            &covariance,
            SymmetricAssembly::Mirrored,
            "constraint covariance AQ⁻¹Aᵀ",
        )
        .map_err(|error| GaussianMarginalError::Solve {
            stage: "constraint covariance",
            error,
        })?;
        let weights = factor
            .solve(&value)
            .map_err(|error| GaussianMarginalError::Solve {
                stage: "constraint covariance solve",
                error,
            })?
            .into_solution();
        (weights, factor.log_det())
    };
    Ok(ExactConstraintPosterior {
        prior_precision: prior,
        mean: readout.dot(&weights),
        readout,
        constraint_covariance: covariance,
        evidence: GaussianEvidenceParts {
            quadratic: value.dot(&weights),
            log_det,
            observations: rows,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Side;
    use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
    use gam_linalg::roundoff::{accumulation_growth, symmetric_spectrum_rounding_band};
    use ndarray::{Axis, array, concatenate};

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

    fn largest_log_magnitude((smallest, largest): (f64, f64)) -> f64 {
        smallest.ln().abs().max(largest.ln().abs())
    }

    /// `‖ΔM‖₂` bound of a strict Cholesky of an SPD `dim × dim` matrix with largest eigenvalue `largest`: Higham, ASNA
    /// 2nd ed., Thm 10.3 gives `|ΔM| ≤ γ_{3·dim+1}·|L||Lᵀ|`, and `‖|L||Lᵀ|‖₂ ≤ ‖L‖_F² = tr M ≤ dim·‖M‖₂`.
    fn cholesky_backward_band(dim: usize, largest: f64) -> f64 {
        dim as f64 * accumulation_growth(3 * dim + 1) * largest
    }

    /// First-order forward-error band of `log|M|` from a strict Cholesky of `M` whose assembly left an error of 2-norm
    /// at most `assembly`: `|Δ log|M|| = |tr(M⁻¹ΔM)| ≤ dim·‖ΔM‖₂/λ_min`, plus the accumulation of `dim` pivot
    /// logarithms, each pivot lying in `[λ_min, λ_max]`.
    fn log_det_band(dim: usize, spectrum: (f64, f64), assembly: f64) -> f64 {
        dim as f64 * (assembly + cholesky_backward_band(dim, spectrum.1)) / spectrum.0
            + accumulation_growth(2 * dim) * dim as f64 * largest_log_magnitude(spectrum)
    }

    /// First-order rounding bands of one model's evaluation, built on its whitened inputs. Accumulation bands are
    /// Higham ASNA Lemma 3.1 (`γ_k·Σ|terms|`).
    struct Bands {
        dual_quadratic: f64,
        dual_log_det: f64,
        primal_quadratic: f64,
        primal_log_det: f64,
        /// `‖Δβ̂‖₂` of the dual posterior mean.
        mean: f64,
        /// `‖Δ(Q⁻¹WᵀC̃⁻¹ỹ)‖₂`, the primal (Woodbury) route to the same mean.
        woodbury_mean: f64,
    }

    fn bands(model: &GaussianMarginalModel, noise_variance: &Array1<f64>) -> Bands {
        let gamma = accumulation_growth;
        let design = &model.design;
        let response = &model.response;
        let prior = &model.prior_precision;
        let (n, p) = design.dim();
        let abs_design = design.mapv(f64::abs);
        let abs_response = response.mapv(f64::abs);
        let abs_prior = prior.mapv(f64::abs);
        // Whitening rounds every design and response entry through a square root, a reciprocal and a product.
        let whitening = gamma(3);
        let noise_log_sum: f64 = noise_variance
            .iter()
            .map(|variance| variance.ln().abs())
            .sum();
        let noise_log_band = gamma(n) * noise_log_sum;
        let prior_spectrum = spectrum_extremes(prior);
        let gram_terms = abs_design.t().dot(&abs_design);

        // Dual: H = Q + WᵀW, b = Wᵀỹ, β̂ = H⁻¹b, q = ‖ỹ − Wβ̂‖² + β̂ᵀQβ̂.
        let precision = prior + &design.t().dot(design);
        let precision_spectrum = spectrum_extremes(&precision);
        let precision_assembly = gamma(n + 1) * frobenius(&(&abs_prior + &gram_terms))
            + 2.0 * whitening * frobenius(&gram_terms);
        let precision_perturbation =
            precision_assembly + cholesky_backward_band(p, precision_spectrum.1);
        let rhs = design.t().dot(response);
        let rhs_assembly =
            (gamma(n) + 2.0 * whitening) * euclidean(&abs_design.t().dot(&abs_response));
        let mean = precision
            .cholesky(Side::Lower)
            .expect("fixture posterior precision is SPD")
            .solvevec(&rhs);
        let abs_mean = mean.mapv(f64::abs);
        let mean_error =
            (precision_perturbation * euclidean(&mean) + rhs_assembly) / precision_spectrum.0;
        let residual = response - &design.dot(&mean);
        let residual_norm = euclidean(&residual);
        let residual_error =
            gamma(p + 1) * euclidean(&(&abs_response + &abs_design.dot(&abs_mean)));
        let prior_energy = mean.dot(&prior.dot(&mean));
        // The whitened inputs' own rounding moves q at first order through the residual: `Δq ≈ 2rᵀ(Δỹ − ΔWβ̂)`.
        let whitening_quadratic = 2.0
            * whitening
            * residual_norm
            * (euclidean(response) + frobenius(&abs_design) * euclidean(&mean));
        let dual_quadratic = precision_spectrum.1 * mean_error * mean_error
            + 2.0 * residual_norm * residual_error
            + residual_error * residual_error
            + gamma(n + 1) * residual_norm * residual_norm
            + gamma(p * p + p + 1) * abs_mean.dot(&abs_prior.dot(&abs_mean))
            + gamma(2) * (residual_norm * residual_norm + prior_energy)
            + whitening_quadratic;
        let dual_log_det = log_det_band(p, precision_spectrum, precision_assembly)
            + log_det_band(p, prior_spectrum, 0.0)
            + noise_log_band
            + gamma(2)
                * (noise_log_sum
                    + p as f64 * largest_log_magnitude(precision_spectrum)
                    + p as f64 * largest_log_magnitude(prior_spectrum));

        // Primal: Z = Q⁻¹Wᵀ, C̃ = I + WZ, α = C̃⁻¹ỹ, q = ỹᵀα.
        let readout = prior
            .cholesky(Side::Lower)
            .expect("fixture prior is SPD")
            .solve_mat(&design.t().to_owned());
        let readout_error =
            cholesky_backward_band(p, prior_spectrum.1) * frobenius(&readout) / prior_spectrum.0;
        let product_terms = frobenius(&abs_design.dot(&readout.mapv(f64::abs)));
        let covariance = design.dot(&readout) + Array2::<f64>::eye(n);
        let covariance_spectrum = spectrum_extremes(&covariance);
        let covariance_assembly =
            (gamma(p + 1) + 2.0 * whitening) * product_terms + frobenius(design) * readout_error;
        let covariance_perturbation =
            covariance_assembly + cholesky_backward_band(n, covariance_spectrum.1);
        let weights = covariance
            .cholesky(Side::Lower)
            .expect("fixture marginal covariance is SPD")
            .solvevec(response);
        let weights_norm = euclidean(&weights);
        let primal_quadratic = weights_norm * weights_norm * covariance_perturbation
            + gamma(n) * abs_response.dot(&weights.mapv(f64::abs))
            + 2.0 * whitening * weights_norm * euclidean(response);
        let primal_log_det = log_det_band(n, covariance_spectrum, covariance_assembly)
            + noise_log_band
            + gamma(2) * (noise_log_sum + n as f64 * largest_log_magnitude(covariance_spectrum));
        let weights_error = covariance_perturbation * weights_norm / covariance_spectrum.0
            + 2.0 * whitening * euclidean(response) / covariance_spectrum.0;
        let woodbury_mean = frobenius(&readout) * weights_error
            + readout_error * weights_norm
            + gamma(n) * euclidean(&readout.mapv(f64::abs).dot(&weights.mapv(f64::abs)));

        Bands {
            dual_quadratic,
            dual_log_det,
            primal_quadratic,
            primal_log_det,
            mean: mean_error,
            woodbury_mean,
        }
    }

    /// Band of `−½(q + log|C| + n·ln 2π)` computed from the dispatched form.
    fn log_evidence_band(model: &GaussianMarginalModel, noise_variance: &Array1<f64>) -> f64 {
        let parts = model.evidence().expect("fixture evidence");
        let bands = bands(model, noise_variance);
        let (quadratic_band, log_det_band) = if model.coefficients() <= model.observations() {
            (bands.dual_quadratic, bands.dual_log_det)
        } else {
            (bands.primal_quadratic, bands.primal_log_det)
        };
        0.5 * (quadratic_band + log_det_band)
            + accumulation_growth(4)
                * (parts.quadratic.abs()
                    + parts.log_det.abs()
                    + parts.observations as f64 * (2.0 * PI).ln())
    }

    /// A nonorthogonal fixture with dyadic basis and prior entries, so a dyadic change of basis transforms it exactly.
    fn fixture() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        let basis = array![
            [1.0, 0.5, 0.25],
            [1.0, -0.25, 0.0625],
            [1.0, 1.25, 1.5625],
            [1.0, -1.0, 1.0],
            [1.0, 0.125, 0.015625],
            [1.0, 0.75, -0.25],
            [0.5, -0.5, 0.75],
            [1.0, 0.0, 1.0],
            [-0.25, 1.5, 0.5]
        ];
        let response = array![0.3, -0.2, 1.1, 0.9, 0.05, 0.4, -0.1, 0.8, 0.2];
        let noise = array![0.1, 0.2, 0.15, 0.3, 0.12, 0.25, 0.1, 0.2, 0.3];
        let prior = array![[2.0, 0.5, -0.25], [0.5, 1.5, 0.25], [-0.25, 0.25, 1.0]];
        (basis, response, noise, prior)
    }

    #[test]
    fn primal_and_dual_evidence_agree_within_their_rounding_bands() {
        let (basis, response, noise, prior) = fixture();
        let model =
            GaussianMarginalModel::new(basis.view(), response.view(), noise.view(), prior.view())
                .expect("fixture model");
        let dual = model.evidence_dual().expect("dual evidence");
        let primal = model.evidence_primal().expect("primal evidence");
        let bands = bands(&model, &noise);
        let quadratic_band = bands.dual_quadratic + bands.primal_quadratic;
        let log_det_band = bands.dual_log_det + bands.primal_log_det;

        assert_eq!(dual.observations, 9);
        assert_eq!(primal.observations, 9);
        assert!(
            (dual.quadratic - primal.quadratic).abs() <= quadratic_band,
            "yᵀC⁻¹y: dual {:e} primal {:e}, difference {:e} against band {quadratic_band:e}",
            dual.quadratic,
            primal.quadratic,
            (dual.quadratic - primal.quadratic).abs()
        );
        assert!(
            (dual.log_det - primal.log_det).abs() <= log_det_band,
            "log|C|: dual {:e} primal {:e}, difference {:e} against band {log_det_band:e}",
            dual.log_det,
            primal.log_det,
            (dual.log_det - primal.log_det).abs()
        );

        // Positive control: moving one response by 1e-6 moves yᵀC⁻¹y by far more than the band, so agreement inside
        // the band resolves a material difference.
        let mut moved = response.clone();
        moved[0] += 1e-6;
        let moved_primal =
            GaussianMarginalModel::new(basis.view(), moved.view(), noise.view(), prior.view())
                .expect("moved model")
                .evidence_primal()
                .expect("moved primal evidence");
        assert!(
            (moved_primal.quadratic - dual.quadratic).abs() > quadratic_band,
            "a 1e-6 response change moved yᵀC⁻¹y by {:e}, inside the band {quadratic_band:e}",
            (moved_primal.quadratic - dual.quadratic).abs()
        );
    }

    #[test]
    fn evidence_factors_the_smaller_matrix() {
        let (basis, response, noise, prior) = fixture();
        let tall =
            GaussianMarginalModel::new(basis.view(), response.view(), noise.view(), prior.view())
                .expect("tall model");
        assert_eq!(tall.evidence(), tall.evidence_dual());
        let wide_basis = basis.slice(ndarray::s![..2, ..]).to_owned();
        let wide = GaussianMarginalModel::new(
            wide_basis.view(),
            response.slice(ndarray::s![..2]),
            noise.slice(ndarray::s![..2]),
            prior.view(),
        )
        .expect("wide model");
        assert_eq!((wide.observations(), wide.coefficients()), (2, 3));
        assert_eq!(wide.evidence(), wide.evidence_primal());
    }

    #[test]
    fn evidence_and_posterior_function_are_invariant_under_a_nonorthogonal_basis_change() {
        let (basis, response, noise, prior) = fixture();
        // A nonorthogonal dyadic T with its exact inverse: β = Tβ̃, Φ̃ = ΦT, Q̃ = TᵀQT.
        let t = array![[2.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.5]];
        let t_inverse = array![[0.5, -0.5, -1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 2.0]];
        assert_eq!(t.dot(&t_inverse), Array2::<f64>::eye(3));
        let transformed_basis = basis.dot(&t);
        let transformed_prior = t.t().dot(&prior).dot(&t);

        let original =
            GaussianMarginalModel::new(basis.view(), response.view(), noise.view(), prior.view())
                .expect("original model");
        let transformed = GaussianMarginalModel::new(
            transformed_basis.view(),
            response.view(),
            noise.view(),
            transformed_prior.view(),
        )
        .expect("transformed model");
        let evidence_band =
            log_evidence_band(&original, &noise) + log_evidence_band(&transformed, &noise);
        let original_evidence = original
            .evidence()
            .expect("original evidence")
            .log_evidence();
        let transformed_evidence = transformed
            .evidence()
            .expect("transformed evidence")
            .log_evidence();
        assert!(
            (original_evidence - transformed_evidence).abs() <= evidence_band,
            "a consistent basis change moved log p(y) from {original_evidence:e} to {transformed_evidence:e}, band {evidence_band:e}"
        );

        // The posterior function Φβ̂ is the same function in both coordinates.
        let original_fit = basis.dot(&original.posterior().expect("original posterior").mean());
        let transformed_fit = transformed_basis.dot(
            &transformed
                .posterior()
                .expect("transformed posterior")
                .mean(),
        );
        let original_mean = original
            .posterior()
            .expect("original posterior")
            .mean()
            .to_owned();
        let transformed_mean = transformed
            .posterior()
            .expect("transformed posterior")
            .mean()
            .to_owned();
        let fit_band = frobenius(&basis) * bands(&original, &noise).mean
            + frobenius(&transformed_basis) * bands(&transformed, &noise).mean
            + accumulation_growth(4)
                * (euclidean(&basis.mapv(f64::abs).dot(&original_mean.mapv(f64::abs)))
                    + euclidean(
                        &transformed_basis
                            .mapv(f64::abs)
                            .dot(&transformed_mean.mapv(f64::abs)),
                    ));
        let fit_difference = euclidean(&(&original_fit - &transformed_fit));
        assert!(
            fit_difference <= fit_band,
            "a consistent basis change moved the posterior function by {fit_difference:e}, band {fit_band:e}"
        );

        // Positive control: changing the basis without transforming the prior is a different function prior.
        let inconsistent = GaussianMarginalModel::new(
            transformed_basis.view(),
            response.view(),
            noise.view(),
            prior.view(),
        )
        .expect("inconsistent model");
        let inconsistent_evidence = inconsistent
            .evidence()
            .expect("inconsistent evidence")
            .log_evidence();
        let inconsistent_band =
            log_evidence_band(&original, &noise) + log_evidence_band(&inconsistent, &noise);
        assert!(
            (original_evidence - inconsistent_evidence).abs() > inconsistent_band,
            "an inconsistent basis change left log p(y) at {inconsistent_evidence:e} against {original_evidence:e}, band {inconsistent_band:e}"
        );
    }

    #[test]
    fn duplicating_the_basis_at_half_the_prior_variance_leaves_the_evidence_unchanged() {
        let (basis, response, noise, prior) = fixture();
        let duplicated_basis =
            concatenate(Axis(1), &[basis.view(), basis.view()]).expect("duplicated basis");
        let zeros = Array2::<f64>::zeros((3, 3));
        let halved_variance = prior.mapv(|value| 2.0 * value);
        let duplicated_prior = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[halved_variance.view(), zeros.view()])
                    .expect("top")
                    .view(),
                concatenate(Axis(1), &[zeros.view(), halved_variance.view()])
                    .expect("bottom")
                    .view(),
            ],
        )
        .expect("duplicated prior");

        let original =
            GaussianMarginalModel::new(basis.view(), response.view(), noise.view(), prior.view())
                .expect("original model");
        let duplicated = GaussianMarginalModel::new(
            duplicated_basis.view(),
            response.view(),
            noise.view(),
            duplicated_prior.view(),
        )
        .expect("duplicated model");
        let original_evidence = original
            .evidence()
            .expect("original evidence")
            .log_evidence();
        let duplicated_evidence = duplicated
            .evidence()
            .expect("duplicated evidence")
            .log_evidence();
        let band = log_evidence_band(&original, &noise) + log_evidence_band(&duplicated, &noise);
        assert!(
            (original_evidence - duplicated_evidence).abs() <= band,
            "duplicating the basis at half the prior variance moved log p(y) from {original_evidence:e} to {duplicated_evidence:e}, band {band:e}"
        );

        // Positive control: duplicating at the FULL prior variance doubles the function's prior variance.
        let full_variance_prior = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[prior.view(), zeros.view()])
                    .expect("top")
                    .view(),
                concatenate(Axis(1), &[zeros.view(), prior.view()])
                    .expect("bottom")
                    .view(),
            ],
        )
        .expect("full-variance prior");
        let doubled = GaussianMarginalModel::new(
            duplicated_basis.view(),
            response.view(),
            noise.view(),
            full_variance_prior.view(),
        )
        .expect("doubled model");
        let doubled_evidence = doubled.evidence().expect("doubled evidence").log_evidence();
        let doubled_band =
            log_evidence_band(&original, &noise) + log_evidence_band(&doubled, &noise);
        assert!(
            (original_evidence - doubled_evidence).abs() > doubled_band,
            "doubling the prior variance left log p(y) at {doubled_evidence:e} against {original_evidence:e}, band {doubled_band:e}"
        );
    }

    #[test]
    fn posterior_mean_matches_the_woodbury_primal_route() {
        let (basis, response, noise, prior) = fixture();
        let model =
            GaussianMarginalModel::new(basis.view(), response.view(), noise.view(), prior.view())
                .expect("fixture model");
        let posterior = model.posterior().expect("posterior");
        // H⁻¹Wᵀỹ = Q⁻¹Wᵀ(I + WQ⁻¹Wᵀ)⁻¹ỹ, formed without H.
        let readout = model
            .prior_precision
            .cholesky(Side::Lower)
            .expect("fixture prior is SPD")
            .solve_mat(&model.design.t().to_owned());
        let covariance = model.design.dot(&readout) + Array2::<f64>::eye(model.observations());
        let woodbury_mean = readout.dot(
            &covariance
                .cholesky(Side::Lower)
                .expect("fixture marginal covariance is SPD")
                .solvevec(&model.response),
        );
        let bands = bands(&model, &noise);
        let band = bands.mean + bands.woodbury_mean;
        let difference = euclidean(&(&posterior.mean().to_owned() - &woodbury_mean));
        assert!(
            difference <= band,
            "posterior mean differs from the Woodbury route by {difference:e}, band {band:e}"
        );
        // The covariance operation inverts the posterior precision on a known column.
        let unit = Array2::from_shape_fn((3, 1), |(row, col)| if row == col { 1.0 } else { 0.0 });
        let column = posterior
            .covariance_times(&unit)
            .expect("covariance column");
        let recovered = posterior.precision().dot(&column);
        let precision_spectrum = spectrum_extremes(&posterior.precision().to_owned());
        let recovery_band = 3.0 * accumulation_growth(10) * precision_spectrum.1
            / precision_spectrum.0
            + 3.0 * cholesky_backward_band(3, precision_spectrum.1) / precision_spectrum.0;
        let recovery_error =
            euclidean(&(&recovered.column(0).to_owned() - &unit.column(0).to_owned()));
        assert!(
            recovery_error <= recovery_band,
            "H·covariance_times(e₁) is {recovery_error:e} from e₁, band {recovery_band:e}"
        );

        // Positive control: the prior-only readout Q⁻¹Wᵀỹ, which ignores the data's precision, is materially different.
        let prior_only = readout.dot(&model.response);
        assert!(euclidean(&(&posterior.mean().to_owned() - &prior_only)) > band);
    }

    #[test]
    fn an_improper_prior_and_undeclared_noise_are_refused() {
        let (basis, response, noise, prior) = fixture();
        // Positive control: the proper fixture is accepted.
        assert!(
            GaussianMarginalModel::new(basis.view(), response.view(), noise.view(), prior.view())
                .is_ok()
        );

        let singular = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            GaussianMarginalModel::new(
                basis.view(),
                response.view(),
                noise.view(),
                singular.view()
            ),
            Err(GaussianMarginalError::ImproperPrior(_))
        ));
        let asymmetric = array![[2.0, 0.5, 0.0], [0.25, 1.5, 0.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            GaussianMarginalModel::new(
                basis.view(),
                response.view(),
                noise.view(),
                asymmetric.view()
            ),
            Err(GaussianMarginalError::InvalidInput(_))
        ));
        let mut silent = noise.clone();
        silent[3] = 0.0;
        assert!(matches!(
            GaussianMarginalModel::new(basis.view(), response.view(), silent.view(), prior.view()),
            Err(GaussianMarginalError::InvalidInput(_))
        ));
        let narrow_prior = array![[1.0, 0.0], [0.0, 1.0]];
        assert!(matches!(
            GaussianMarginalModel::new(
                basis.view(),
                response.view(),
                noise.view(),
                narrow_prior.view()
            ),
            Err(GaussianMarginalError::InvalidInput(_))
        ));
    }

    /// A dyadic well-conditioned exact constraint on the fixture prior.
    fn constraint_fixture() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let constraint = array![[1.0, 0.5, 0.25], [0.5, -0.5, 0.75]];
        let value = array![0.3, -0.2];
        let prior = fixture().3;
        (constraint, value, prior)
    }

    struct ConstraintBands {
        quadratic: f64,
        log_det: f64,
        /// `‖V_computed − V‖₂` bound: the formation of `AZ` plus the propagated `Q`-solve error.
        covariance_assembly: f64,
        covariance_spectrum: (f64, f64),
    }

    /// First-order rounding bands of the exact-constraint evidence parts (Higham ASNA Lemma 3.1 and Thm 10.3).
    fn constraint_bands(
        constraint: &Array2<f64>,
        value: &Array1<f64>,
        prior: &Array2<f64>,
    ) -> ConstraintBands {
        let gamma = accumulation_growth;
        let (n, p) = constraint.dim();
        let prior_spectrum = spectrum_extremes(prior);
        let readout = prior
            .cholesky(Side::Lower)
            .expect("fixture prior is SPD")
            .solve_mat(&constraint.t().to_owned());
        let readout_error =
            cholesky_backward_band(p, prior_spectrum.1) * frobenius(&readout) / prior_spectrum.0;
        let mut covariance = constraint.dot(&readout);
        symmetrize_in_place(&mut covariance);
        let covariance_spectrum = spectrum_extremes(&covariance);
        let covariance_assembly =
            2.0 * gamma(p + 1) * frobenius(&constraint.mapv(f64::abs).dot(&readout.mapv(f64::abs)))
                + frobenius(constraint) * readout_error;
        let weights = covariance
            .cholesky(Side::Lower)
            .expect("fixture constraint covariance is SPD")
            .solvevec(value);
        let weights_norm = euclidean(&weights);
        ConstraintBands {
            quadratic: weights_norm
                * weights_norm
                * (covariance_assembly + cholesky_backward_band(n, covariance_spectrum.1))
                + gamma(n) * value.mapv(f64::abs).dot(&weights.mapv(f64::abs)),
            log_det: log_det_band(n, covariance_spectrum, covariance_assembly),
            covariance_assembly,
            covariance_spectrum,
        }
    }

    #[test]
    fn exact_constraint_evidence_is_the_zero_noise_limit_of_the_noisy_evidence() {
        let (constraint, value, prior) = constraint_fixture();
        let n = constraint.nrows();
        let exact = condition_on_exact_constraint(constraint.view(), value.view(), prior.view())
            .expect("exact constraint")
            .evidence();
        let exact_bands = constraint_bands(&constraint, &value, &prior);
        // With C = V + rI: 0 ≤ yᵀV⁻¹y − yᵀC⁻¹y = Σ c_i²·r/(λ_i(λ_i + r)) ≤ r‖y‖²/λ_min² and
        // 0 ≤ log|C| − log|V| = Σ ln(1 + r/λ_i) ≤ n·r/λ_min. The computed λ_min is lowered by the eigensolver's
        // n-dimensional band n·ε·λ_max and by the assembly error of V itself (Weyl), so both brackets stay upper bounds.
        let lowered_smallest = exact_bands.covariance_spectrum.0
            - symmetric_spectrum_rounding_band(&vec![exact_bands.covariance_spectrum.1; n])
            - exact_bands.covariance_assembly;
        assert!(
            lowered_smallest > 0.0,
            "the fixture constraint covariance is resolved from singular"
        );

        // r = 2⁻²⁰, so the whitening scale r^{−1/2} = 2¹⁰ is exact.
        let noise = Array1::from_elem(n, 2.0_f64.powi(-20));
        let noisy_model =
            GaussianMarginalModel::new(constraint.view(), value.view(), noise.view(), prior.view())
                .expect("noisy model");
        let noisy = noisy_model
            .evidence_primal()
            .expect("noisy primal evidence");
        let noisy_bands = bands(&noisy_model, &noise);
        let quadratic_band = exact_bands.quadratic + noisy_bands.primal_quadratic;
        let log_det_band = exact_bands.log_det + noisy_bands.primal_log_det;
        let r = noise[0];
        let quadratic_gap = exact.quadratic - noisy.quadratic;
        let quadratic_bound = r * value.dot(&value) / (lowered_smallest * lowered_smallest);
        assert!(
            quadratic_gap >= -quadratic_band && quadratic_gap <= quadratic_bound + quadratic_band,
            "yᵀV⁻¹y − yᵀC⁻¹y = {quadratic_gap:e} outside [0, {quadratic_bound:e}] ± {quadratic_band:e}"
        );
        let log_det_gap = noisy.log_det - exact.log_det;
        let log_det_bound = n as f64 * r / lowered_smallest;
        assert!(
            log_det_gap >= -log_det_band && log_det_gap <= log_det_bound + log_det_band,
            "log|C| − log|V| = {log_det_gap:e} outside [0, {log_det_bound:e}] ± {log_det_band:e}"
        );

        // Positive control: at r = 2⁻² the noisy log-determinant sits materially above the exact one, so the bracket
        // above is not satisfied by construction.
        let coarse_noise = Array1::from_elem(n, 0.25);
        let coarse_model = GaussianMarginalModel::new(
            constraint.view(),
            value.view(),
            coarse_noise.view(),
            prior.view(),
        )
        .expect("coarse model");
        let coarse = coarse_model
            .evidence_primal()
            .expect("coarse primal evidence");
        let coarse_band = exact_bands.log_det + bands(&coarse_model, &coarse_noise).primal_log_det;
        assert!(
            coarse.log_det - exact.log_det > coarse_band,
            "at r = 1/4 log|C| − log|V| = {:e}, inside the band {coarse_band:e}",
            coarse.log_det - exact.log_det
        );
    }

    #[test]
    fn exact_constraint_covariance_annihilates_the_constraint() {
        let (constraint, value, prior) = constraint_fixture();
        let (n, p) = constraint.dim();
        let posterior =
            condition_on_exact_constraint(constraint.view(), value.view(), prior.view())
                .expect("exact constraint");
        let unit = Array2::<f64>::eye(p);
        let covariance = posterior
            .covariance_times(&unit)
            .expect("covariance columns");
        let annihilated = constraint.dot(&covariance);

        // With ĉ = V̂⁻¹ẐᵀB and AQ⁻¹ = Zᵀ exactly:
        //   A·(Q̂⁻¹B − Ẑĉ) = A(Q̂⁻¹B − Q⁻¹B) − (Ẑ − Z)ᵀB − δproj − r_V − (AẐ − V̂)ĉ,
        // where δproj is the rounding of ẐᵀB and r_V = ẐᵀB + δproj − V̂ĉ the V-solve residual, plus the rounding of
        // the production product Ẑĉ, its subtraction, and this test's final A·.
        let gamma = accumulation_growth;
        let exact_bands = constraint_bands(&constraint, &value, &prior);
        let prior_spectrum = spectrum_extremes(&prior);
        let prior_perturbation = cholesky_backward_band(p, prior_spectrum.1);
        let prior_part = prior
            .cholesky(Side::Lower)
            .expect("fixture prior is SPD")
            .solve_mat(&unit);
        let readout = prior
            .cholesky(Side::Lower)
            .expect("fixture prior is SPD")
            .solve_mat(&constraint.t().to_owned());
        let readout_error = prior_perturbation * frobenius(&readout) / prior_spectrum.0;
        let mut constraint_covariance = constraint.dot(&readout);
        symmetrize_in_place(&mut constraint_covariance);
        let correction = constraint_covariance
            .cholesky(Side::Lower)
            .expect("fixture constraint covariance is SPD")
            .solve_mat(&readout.t().dot(&unit));
        let correction_norm = frobenius(&correction);
        let abs_constraint = constraint.mapv(f64::abs);
        let abs_readout = readout.mapv(f64::abs);
        let abs_correction = correction.mapv(f64::abs);
        let product_terms = frobenius(&abs_readout.dot(&abs_correction));
        let band = frobenius(&constraint) * prior_perturbation * frobenius(&prior_part)
            / prior_spectrum.0
            + readout_error * frobenius(&unit)
            + gamma(p + 1) * frobenius(&abs_readout.t().dot(&unit.mapv(f64::abs)))
            + cholesky_backward_band(n, exact_bands.covariance_spectrum.1) * correction_norm
            + exact_bands.covariance_assembly * correction_norm
            + frobenius(&constraint)
                * (gamma(n + 1) * product_terms
                    + gamma(1) * (frobenius(&prior_part) + product_terms))
            + gamma(p + 1)
                * frobenius(
                    &abs_constraint
                        .dot(&(&prior_part.mapv(f64::abs) + &abs_readout.dot(&abs_correction))),
                );
        assert!(
            frobenius(&annihilated) <= band,
            "‖A·Cov‖_F = {:e}, band {band:e}",
            frobenius(&annihilated)
        );

        // Positive control: the unconditioned prior covariance does not annihilate the constraint.
        assert!(frobenius(&constraint.dot(&prior_part)) > band);
    }

    #[test]
    fn a_rank_deficient_exact_constraint_is_refused() {
        let (constraint, value, prior) = constraint_fixture();
        // Positive control: the full-rank constraint conditions.
        assert!(
            condition_on_exact_constraint(constraint.view(), value.view(), prior.view()).is_ok()
        );
        let repeated = array![[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]];
        assert!(matches!(
            condition_on_exact_constraint(repeated.view(), value.view(), prior.view()),
            Err(GaussianMarginalError::RankDeficientConstraint {
                rows: 2,
                resolved_rank: 1
            })
        ));
        let too_many_rows = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        let four_values = Array1::<f64>::zeros(4);
        assert!(matches!(
            condition_on_exact_constraint(too_many_rows.view(), four_values.view(), prior.view()),
            Err(GaussianMarginalError::InvalidInput(_))
        ));
    }
}

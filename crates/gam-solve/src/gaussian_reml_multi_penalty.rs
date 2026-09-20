//! Exact profiled Gaussian REML over several response columns that share one design, several penalties, one
//! smoothing vector and one dispersion (#2946).
//!
//! Model: `Y = XB + E` with `Y` `n × m`, `X` `n × p`, every column `y_c = Xβ_c + ε_c`, `ε_c ~ N(0, σ²I)`, and the
//! improper prior `β_c ~ N(0, σ²S_λ⁺)` with `S_λ = Σ_k λ_k S_k`, `λ_k = e^{ρ_k}`. The columns are coordinates of one
//! vector-valued response (whitened output directions), so `λ` and `σ²` are shared. Integrating the coefficients and
//! profiling `σ̂² = q/ν`,
//!
//! ```text
//!   V(ρ) = ½[ ν(1 + log(2πq/ν)) + m·log|K| − m·log|S_λ|₊ ],     K = XᵀX + S_λ,   ν = m(n − M_p),
//!   q(ρ) = Σ_c min_β ‖y_c − Xβ‖² + βᵀS_λβ,
//! ```
//!
//! with `M_p = dim ∩_k ker S_k`, the declared null space. At `m = 1`, `K = 1` this is the criterion of
//! [`crate::gaussian_reml::gaussian_reml_closed_form`].
//!
//! **Weights.** Observation weights (`ε_ic ~ N(0, σ²/w_i)`) whiten the positive-weight rows by `√w_i`, count only
//! those rows in `n`, and add the density change `−(m/2)·Σ_{w_i>0} log w_i` to `V`
//! ([`GaussianRemlMultiPenaltyProblem::new_weighted`]).
//!
//! **One reduction serves every column.** One Householder QR `X = QR` rotates the responses once: the tail of `QᵀY`
//! is the unpenalized residual `r0` and its head `Z` (`p × m`) carries everything else. `q` depends on `Z` only through
//! `ZZᵀ`, so when `m > p` a QR of `Zᵀ` compresses the `m` columns to `C` (`p × p`) with `CCᵀ = ZZᵀ`. Each evaluation
//! is then independent of `n` and `m`: one Householder QR of the stacked root
//!
//! ```text
//!   A(λ) = [R; √λ_1 L_1; …; √λ_K L_K],   S_k = L_kᵀL_k,   AᵀA = K,
//! ```
//!
//! applied to `[C; 0]`. Its tail is the penalized residual as a sum of squares (nothing cancels), its head gives
//! `B̂ = R_A⁻¹·head`, and `log|K| = 2Σ log|r_ii|` is priced at root scale, `O(ε√κ(K))`. `log|S_λ|₊` and its ρ-derivatives
//! come from the one owner, [`PenaltyPseudologdet`], at root scale too.
//!
//! **Derivatives** (analytic; `P̃_k = √λ_k L_k R_A⁻¹`, `M_k = √λ_k L_k B̂`, `Ẽ_k = P̃_kᵀM_k`):
//!
//! ```text
//!   τ_k = λ_k tr(K⁻¹S_k) = ‖P̃_k‖²,          b_k = λ_k tr(B̂ᵀS_kB̂) = ‖M_k‖²,
//!   ∂_k q = b_k,                             ∂²_kj q = δ_kj b_k − 2⟨Ẽ_k, Ẽ_j⟩,
//!   ∂_k log|K| = τ_k,                        ∂²_kj log|K| = δ_kj τ_k − ‖P̃_k P̃_jᵀ‖²,
//!   ∂_k V = ½[(ν/q)∂_k q + m ∂_k log|K| − m ∂_k log|S_λ|₊],
//!   ∂²_kj V = ½[(ν/q)∂²_kj q − (ν/q²)∂_k q ∂_j q + m ∂²_kj log|K| − m ∂²_kj log|S_λ|₊].
//! ```
//!
//! `∂_k q` is the envelope theorem at the penalized minimizer, and `∂B̂/∂ρ_j = −λ_j K⁻¹S_jB̂`. The effective degrees
//! of freedom are `edf = tr(K⁻¹XᵀX) = p − Σ_k τ_k`; `τ_k` is published per penalty because it is the only additive
//! split of the total when penalties overlap.
//!
//! **Rank.** Each penalty's root is read off its own spectrum above
//! [`gam_linalg::roundoff::symmetric_spectrum_rounding_band`]. The null space is the rank complement of the stacked
//! unit-norm roots under [`gam_linalg::roundoff::factor_rank_partition`]; the caller declares it, and the declaration,
//! that predicate and the rank that prices `log|S_λ|₊` must agree or construction refuses.
//!
//! **Rounding.** Every published quantity carries a first-order forward-error bound built from the backward errors of
//! the arithmetic that produced it: Householder QR and its application to a right-hand side (Higham, *ASNA* 2nd ed.,
//! Thms 19.4-19.5, `‖ΔA_j‖ ≤ γ_{mn}‖A_j‖`, with the theorems' small constant taken as one), the symmetric
//! eigensolver behind each root (`‖ΔS_k‖₂ ≤ p·ε·‖S_k‖₂`), the stacked-root SVD
//! ([`gam_linalg::roundoff::factor_singular_band`]) and compensated summation. The residual bound uses the envelope
//! theorem (a perturbation moves the minimum by the objective's own first-order change at the minimizer). The
//! log-determinant bounds use `|δ log det(AᵀA)| ≤ 2‖A⁺‖_F‖ΔA‖_F` and `|tr(K⁻¹ΔS)| ≤ ‖ΔS‖₂·tr(K⁻¹)`.
//!
//! **Optimization.** `ρ̂` comes only from the outer optimizer's converged, certified run (SPEC 22) on the derived
//! resolvability domain ([`crate::estimate::rho_domain::resolvability_domain_from_gram_blocks`]).

use std::f64::consts::PI;

use faer::Side;
use gam_linalg::faer_ndarray::{
    FaerArrayView, FaerEigh, HouseholderQr, decomposition_parallelism, fast_ab, fast_ata,
};
use gam_linalg::matrix::array2_bits_fingerprint;
use gam_linalg::roundoff::{
    FactorRankPartition, SymmetricAssembly, UNIT_ROUNDOFF, accumulation_growth, compensated_band,
    factor_rank_partition, factor_singular_band, resolved_eigenvalue_count,
    symmetric_spectrum_rounding_band,
};
use gam_linalg::utils::{KahanSum, validate_finite_symmetric_matrix};
use gam_problem::{DeclaredHessianForm, Derivative, HessianValue, OuterEval};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::estimate::EstimationError;
use crate::estimate::reml::penalty_logdet::PenaltyPseudologdet;
use crate::rho_optimizer::{FallbackPolicy, OuterProblem};

/// A validated multi-response, multi-penalty Gaussian REML problem, reduced once to `p`-sized statistics.
#[derive(Clone, Debug)]
pub struct GaussianRemlMultiPenaltyProblem {
    /// Rows of the `X` and `Y` the problem was built from.
    rows: usize,
    /// Rows that carry density: every row, or the positive-weight rows of a weighted problem.
    observations: usize,
    responses: usize,
    coefficients: usize,
    nullity: usize,
    /// `R` from the QR of `X`, `min(n, p) × p`.
    design_upper: Array2<f64>,
    /// `Z`: the head of `QᵀY`, `min(n, p) × m`.
    rotated_head: Array2<f64>,
    /// `C` with `CCᵀ = ZZᵀ`, at most `min(n, p)` columns.
    compressed_head: Array2<f64>,
    /// `Σ_c r0_c`, the tail energy of `QᵀY`, compensated.
    unpenalized_residual: f64,
    penalties: Vec<Array2<f64>>,
    /// `L_k` with `L_kᵀL_k = S_k` on `S_k`'s resolved range.
    roots: Vec<Array2<f64>>,
    /// `‖S_k − L_kᵀL_k‖₂` bound: the eigensolver's backward error `p·ε·‖S_k‖₂` plus the truncated modes, which lie
    /// inside that same band.
    penalty_rounding: Vec<f64>,
    /// The same bound for the roots [`PenaltyPseudologdet`] forms: the backward error plus its own truncation.
    pseudo_penalty_rounding: Vec<f64>,
    response_frobenius: f64,
    /// `γ_{n·p}`: Householder backward-error growth of the design QR and its application to `Y`.
    design_rotation_growth: f64,
    /// `γ_{m·min(n,p)}` when the columns were compressed, else zero.
    compression_growth: f64,
    rho_lower: Array1<f64>,
    rho_upper: Array1<f64>,
    /// Value fingerprints of the `X` and `Y` this problem was reduced from ([`array2_bits_fingerprint`]): the data
    /// gradient is the envelope derivative at this data's converged fit and refuses any other arrays.
    design_fingerprint: u64,
    response_fingerprint: u64,
    /// The smallest kept singular value of the unit-normalized stacked penalty roots, and the band the rank was judged
    /// against ([`factor_singular_band`]).
    structural_singular_margin: (f64, f64),
    /// The positive-weight rows and their `√w_i`, when the problem is weighted.
    support: Option<ObservationSupport>,
    /// `−(m/2)·Σ_{w_i>0} log w_i`, the density change from whitened back to observed coordinates, and its rounding
    /// bound. Both are zero for an unweighted problem.
    observation_measure: f64,
    observation_measure_roundoff: f64,
}

/// A weighted problem's positive-weight rows and their `√w_i`. `ε_i ~ N(0, σ²/w_i)` is whitened to unit variance by
/// `√w_i`; a zero-weight row is omitted entirely, with no density, no residual degree of freedom and no cotangent.
#[derive(Clone, Debug)]
struct ObservationSupport {
    rows: Vec<usize>,
    root_weights: Vec<f64>,
}

impl ObservationSupport {
    /// `W½M` over the positive-weight rows.
    fn whiten(&self, matrix: ArrayView2<'_, f64>) -> Array2<f64> {
        Array2::from_shape_fn((self.rows.len(), matrix.ncols()), |(row, col)| {
            matrix[[self.rows[row], col]] * self.root_weights[row]
        })
    }
}

/// Each penalty's root on its resolved range with its rounding bands, and the rank partition of the stacked roots
/// normalized to unit spectral norm.
struct PenaltyStructure {
    roots: Vec<Array2<f64>>,
    penalty_rounding: Vec<f64>,
    pseudo_penalty_rounding: Vec<f64>,
    stacked_rows: usize,
    partition: FactorRankPartition,
}

/// The criterion, its analytic ρ-derivatives and its parts at one `ρ`, each with its forward-error bound.
#[derive(Clone, Debug)]
pub struct GaussianRemlMultiPenaltyEvaluation {
    pub rho: Array1<f64>,
    pub lambdas: Array1<f64>,
    /// `V(ρ)`, the negative profiled restricted log-likelihood.
    pub reml_score: f64,
    pub reml_score_roundoff: f64,
    pub reml_gradient: Array1<f64>,
    pub reml_hessian: Array2<f64>,
    /// `q`, the pooled penalized residual quadratic over every column.
    pub residual_quadratic: f64,
    pub residual_quadratic_roundoff: f64,
    /// `ν = m(n − M_p)`.
    pub dispersion_dof: f64,
    /// `σ̂² = q/ν`.
    pub sigma2: f64,
    /// `log|XᵀX + S_λ|`, counted once (the criterion carries it `m` times).
    pub log_det_penalized_normal: f64,
    pub log_det_penalized_normal_roundoff: f64,
    /// `log|S_λ|₊`, counted once.
    pub log_pseudo_det_penalty: f64,
    pub log_pseudo_det_penalty_roundoff: f64,
    /// `τ_k = λ_k tr(K⁻¹S_k)`: the shrinkage penalty `k` exerts; `edf = p − Σ_k τ_k`.
    pub penalty_trace: Array1<f64>,
    /// `tr(K⁻¹XᵀX)`, shared by every column.
    pub edf: f64,
    pub edf_roundoff: f64,
}

/// A converged fit: the evaluation at `ρ̂`, the per-column coefficients and the domain the search ran on.
#[derive(Clone, Debug)]
pub struct GaussianRemlMultiPenaltyFit {
    pub evaluation: GaussianRemlMultiPenaltyEvaluation,
    /// `p × m`: column `c` is `β̂_c`.
    pub coefficients: Array2<f64>,
    /// First-order bound on the Frobenius norm of the coefficients' rounding error.
    pub coefficients_roundoff: f64,
    pub rho_lower: Array1<f64>,
    pub rho_upper: Array1<f64>,
    pub iterations: usize,
    /// The outer optimizer's analytic optimality certificate at `ρ̂`.
    pub certificate: Option<crate::rho_optimizer::OuterCriterionCertificate>,
    /// Per penalty: interior, or railed at which edge of the domain, read off the certificate's railed coordinates.
    pub rho_placement: Vec<GaussianRemlMultiPenaltyRhoPlacement>,
}

/// Cotangents of the shared-dispersion criterion `V` with respect to the design and the responses.
#[derive(Clone, Debug)]
pub struct GaussianRemlMultiPenaltyDataGradient {
    /// `∂V/∂X = m·XK⁻¹ − RB̂ᵀ/σ̂²`, `n × p`, with `R = Y − XB̂` (whitened and scaled back by `√w_i` when weighted).
    pub grad_x: Array2<f64>,
    /// `∂V/∂Y = R/σ̂²`, `n × m` (the same).
    pub grad_y: Array2<f64>,
}

/// Where a fitted log-strength sits in its derived domain, as the outer optimizer's certificate reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianRemlMultiPenaltyRhoPlacement {
    /// Not railed: `∂V/∂ρ_k = 0` to the certificate's resolution.
    Interior,
    /// Railed at the lower edge of the resolvability domain (the penalty is off to the gradient's resolution).
    LowerBound,
    /// Railed at the upper edge (the penalty's range is switched off to the gradient's resolution).
    UpperBound,
    /// The optimizer returned no certificate, so the placement is not established.
    Unaudited,
}

/// The data gradient where the envelope forms are the total derivative, or the typed reason they are not.
#[derive(Clone, Debug)]
pub enum GaussianRemlMultiPenaltyDataGradientOutcome {
    /// Every `ρ̂_k` is interior.
    Interior(GaussianRemlMultiPenaltyDataGradient),
    /// Some `ρ̂_k` is railed (or unaudited). The domain is derived from `XᵀX`, so a railed coordinate moves with `X`
    /// and the total derivative gains `(∂V/∂ρ_k)(∂ρ_bound/∂X)`, which the envelope forms do not carry.
    RhoAtDomainBound {
        placement: Vec<GaussianRemlMultiPenaltyRhoPlacement>,
    },
}

/// `∂V/∂S_k` for every penalty where the envelope forms are the total derivative, or the typed reason they are not.
#[derive(Clone, Debug)]
pub enum GaussianRemlMultiPenaltyPenaltyGradientOutcome {
    /// Every `ρ̂_k` is interior.
    Interior {
        /// `G_k = ½λ_k(B̂B̂ᵀ/σ̂² + m·K⁻¹ − m·S_λ⁺)`, symmetric, with `dV = tr(G_k·dS_k)` over symmetric `dS_k`. The
        /// gradient over all `p²` entries is `G_k` too; an upper-triangle parameterization carries `2G_ij` off the
        /// diagonal.
        gradients: Vec<Array2<f64>>,
        /// `rank(S_λ)`. The derivatives of `log|S_λ|₊` and of `ν` exist only for perturbations that preserve it.
        rank: usize,
        /// The smallest kept singular value of the unit-normalized stacked penalty roots.
        smallest_resolved_singular_value: f64,
        /// The backward-error band that rank was judged against ([`factor_singular_band`]).
        rank_band: f64,
    },
    /// Some `ρ̂_k` is railed or unaudited. The domain is derived from the penalty spectrum, so its edge moves with
    /// `S_k` and the envelope forms miss `(∂V/∂ρ_k)(∂ρ_bound/∂S)`.
    RhoAtDomainBound {
        placement: Vec<GaussianRemlMultiPenaltyRhoPlacement>,
    },
}

/// `A(λ)`'s QR and the pieces every consumer of it needs.
struct StackedRootFactor {
    lambdas: Array1<f64>,
    /// `√λ_k L_k`.
    scaled_roots: Vec<Array2<f64>>,
    qr: HouseholderQr,
    /// `R_A`, `p × p`.
    upper: faer::Mat<f64>,
    rows: usize,
    frobenius: f64,
    /// `Σ_k λ_k·p·ε·‖S_k‖₂`: the roots' formation error as a bound on `‖ΔS_λ‖₂`.
    penalty_rounding: f64,
    /// `γ_{rows·p}`.
    rotation_growth: f64,
}

/// `QᵀA [head; 0]`, split at `p`.
struct StackedProjection {
    head: faer::Mat<f64>,
    tail_energy: f64,
}

fn frobenius(matrix: ArrayView2<'_, f64>) -> f64 {
    let mut sum = KahanSum::default();
    for &value in matrix {
        sum.add(value * value);
    }
    sum.sum().sqrt()
}

fn faer_to_array(matrix: faer::MatRef<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| matrix[(row, col)])
}

impl GaussianRemlMultiPenaltyProblem {
    /// Validate and reduce `Y = XB + E` under penalties `S_1..S_K` with declared null-space dimension
    /// `declared_nullity = dim ∩_k ker S_k`.
    pub fn new(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalties: &[Array2<f64>],
        declared_nullity: usize,
    ) -> Result<Self, EstimationError> {
        Self::reduce(x, y, None, penalties, declared_nullity)
    }

    /// [`Self::new`] with observation weights: `ε_ic ~ N(0, σ²/w_i)`, the same `w_i` for every column.
    ///
    /// The problem is the unweighted one on the whitened rows `√w_i·x_i`, `√w_i·y_i` of the positive-weight support,
    /// with `n` the number of those rows, and `V` carries the density change back to the observed responses,
    /// `−(m/2)·Σ_{w_i>0} log w_i`. That term is constant in `ρ` and in the penalties. A zero-weight row is omitted
    /// entirely: it has no density, no residual degree of freedom, and a zero data cotangent.
    pub fn new_weighted(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        penalties: &[Array2<f64>],
        declared_nullity: usize,
    ) -> Result<Self, EstimationError> {
        Self::reduce(x, y, Some(weights), penalties, declared_nullity)
    }

    /// `dim ∩_k ker S_k` as construction's own rank predicate reads it, for a caller that holds no structural
    /// declaration of the penalties' null space. Passing it to [`Self::new`] makes the declaration check vacuous; a
    /// caller that knows its null space declares it instead.
    pub fn structural_nullity(penalties: &[Array2<f64>]) -> Result<usize, EstimationError> {
        let Some(first) = penalties.first() else {
            crate::bail_invalid_estim!("multi-penalty Gaussian REML structural nullity needs at least one penalty");
        };
        let p = first.nrows();
        Ok(p - Self::penalty_structure(penalties, p)?.partition.rank)
    }

    /// Read each penalty's root off its own spectrum above [`symmetric_spectrum_rounding_band`], and partition the
    /// stacked unit-norm roots by [`factor_rank_partition`]: the one rank predicate construction and
    /// [`Self::structural_nullity`] share.
    fn penalty_structure(penalties: &[Array2<f64>], p: usize) -> Result<PenaltyStructure, EstimationError> {
        let mut roots = Vec::with_capacity(penalties.len());
        let mut penalty_rounding = Vec::with_capacity(penalties.len());
        let mut pseudo_penalty_rounding = Vec::with_capacity(penalties.len());
        let mut normalized_rows = Vec::new();
        for (k, penalty) in penalties.iter().enumerate() {
            if penalty.dim() != (p, p) {
                crate::bail_invalid_estim!(
                    "multi-penalty Gaussian REML penalty {k} is {}x{}, expected {p}x{p}",
                    penalty.nrows(),
                    penalty.ncols()
                );
            }
            // A declared penalty carries the exact-symmetry contract.
            validate_finite_symmetric_matrix(
                penalty,
                SymmetricAssembly::Mirrored,
                "multi-penalty Gaussian REML penalty",
            )
            .map_err(|error| EstimationError::InvalidInput(format!("penalty {k}: {error}")))?;
            let (eigenvalues, eigenvectors) = penalty
                .eigh(Side::Lower)
                .map_err(EstimationError::LinearSystemSolveFailed)?;
            let spectrum = eigenvalues.to_vec();
            let band = symmetric_spectrum_rounding_band(&spectrum);
            if let Some(negative) = spectrum.iter().find(|&&value| value < -band) {
                crate::bail_invalid_estim!(
                    "multi-penalty Gaussian REML penalty {k} is not positive semidefinite: eigenvalue \
                     {negative:.3e} is below the spectrum's rounding band -{band:.3e}"
                );
            }
            let rank = resolved_eigenvalue_count(&spectrum, 0.0);
            if rank == 0 {
                crate::bail_invalid_estim!(
                    "multi-penalty Gaussian REML penalty {k} has no eigenvalue above its rounding band"
                );
            }
            let norm = spectrum.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
            let mut root = Array2::<f64>::zeros((rank, p));
            for (row, index) in (0..p).filter(|&index| spectrum[index] > band).enumerate() {
                let scale = spectrum[index].sqrt();
                for col in 0..p {
                    root[[row, col]] = scale * eigenvectors[[col, index]];
                }
            }
            let owner_threshold =
                crate::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(&spectrum);
            let owner_truncation = spectrum
                .iter()
                .filter(|&&value| value <= owner_threshold)
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            normalized_rows.push(root.mapv(|value| value / norm.sqrt()));
            roots.push(root);
            penalty_rounding.push(2.0 * band);
            pseudo_penalty_rounding.push(band + owner_truncation);
        }

        // The joint null space ∩ ker S_k is invariant to each penalty's positive scale, so its rank is read off the
        // stacked roots normalized to unit spectral norm.
        let stacked_rows: usize = normalized_rows.iter().map(Array2::nrows).sum();
        let mut stacked = Array2::<f64>::zeros((stacked_rows, p));
        let mut offset = 0;
        for block in &normalized_rows {
            stacked
                .slice_mut(ndarray::s![offset..offset + block.nrows(), ..])
                .assign(block);
            offset += block.nrows();
        }
        let partition =
            factor_rank_partition(&stacked).map_err(EstimationError::LinearSystemSolveFailed)?;
        Ok(PenaltyStructure {
            roots,
            penalty_rounding,
            pseudo_penalty_rounding,
            stacked_rows,
            partition,
        })
    }

    fn reduce(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        weights: Option<ArrayView1<'_, f64>>,
        penalties: &[Array2<f64>],
        declared_nullity: usize,
    ) -> Result<Self, EstimationError> {
        let (rows, p) = x.dim();
        let m = y.ncols();
        if rows == 0 || p == 0 || m == 0 || penalties.is_empty() {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML needs at least one row, column, response and penalty; got \
                 n={rows}, p={p}, m={m}, penalties={}",
                penalties.len()
            );
        }
        if y.nrows() != rows {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML row mismatch: X has {rows} rows but Y has {}",
                y.nrows()
            );
        }
        if x.iter().chain(y.iter()).any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!("multi-penalty Gaussian REML design and responses must be finite");
        }
        let (support, observation_measure, observation_measure_roundoff) = match weights {
            None => (None, 0.0, 0.0),
            Some(weights) => {
                if weights.len() != rows {
                    crate::bail_invalid_estim!(
                        "multi-penalty Gaussian REML weights length mismatch: X has {rows} rows but {} weights",
                        weights.len()
                    );
                }
                if weights.iter().any(|weight| !(weight.is_finite() && *weight >= 0.0)) {
                    crate::bail_invalid_estim!("multi-penalty Gaussian REML weights must be finite and non-negative");
                }
                let kept: Vec<usize> = (0..rows).filter(|&row| weights[row] > 0.0).collect();
                let mut log_det = KahanSum::default();
                let mut magnitude = 0.0;
                for &row in &kept {
                    let term = weights[row].ln();
                    log_det.add(term);
                    magnitude += term.abs();
                }
                // Each logarithm is faithful to an ulp (two roundings), then one more rounding for the scale.
                let scale = 0.5 * m as f64;
                let support = ObservationSupport {
                    root_weights: kept.iter().map(|&row| weights[row].sqrt()).collect(),
                    rows: kept,
                };
                (Some(support), -scale * log_det.sum(), scale * compensated_band(3, magnitude))
            }
        };
        let design_fingerprint = array2_bits_fingerprint(&x);
        let response_fingerprint = array2_bits_fingerprint(&y);
        // From here on `X` and `Y` are the whitened positive-weight rows; `√w_i` and the product round each entry
        // twice, a relative backward error the design rotation's growth carries.
        let whitened = support
            .as_ref()
            .map(|support| (support.whiten(x), support.whiten(y)));
        let (x, y, whitening_growth) = match &whitened {
            Some((x, y)) => (x.view(), y.view(), accumulation_growth(2)),
            None => (x, y, 0.0),
        };
        let n = x.nrows();
        if n == 0 {
            crate::bail_invalid_estim!("weighted multi-penalty Gaussian REML needs at least one positive-weight row");
        }

        let PenaltyStructure {
            roots,
            penalty_rounding,
            pseudo_penalty_rounding,
            stacked_rows,
            partition,
        } = Self::penalty_structure(penalties, p)?;
        let nullity = p - partition.rank;
        if nullity != declared_nullity {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML declared null space of dimension {declared_nullity}, but the \
                 penalties' stacked roots leave {nullity}"
            );
        }
        let unit = vec![1.0; penalties.len()];
        let pseudo = PenaltyPseudologdet::from_components(penalties, &unit, 0.0)
            .map_err(EstimationError::InvalidInput)?;
        if pseudo.rank() != partition.rank {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML rank disagreement: the stacked-root predicate gives rank {} but \
                 the penalty pseudo-determinant owner prices rank {}",
                partition.rank,
                pseudo.rank()
            );
        }
        if n <= nullity {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML needs more observed (positive-weight) rows than the null-space \
                 dimension; got n={n}, nullity={nullity}"
            );
        }

        let x_owned = x.to_owned();
        let design_qr = HouseholderQr::new(FaerArrayView::new(&x_owned).as_ref());
        let design_upper = faer_to_array(design_qr.r());
        let head_rows = design_upper.nrows();
        if head_rows + stacked_rows < p {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML penalized normal matrix is singular for every λ: the design rank \
                 bound {head_rows} plus the penalty ranks {stacked_rows} is below p={p}"
            );
        }
        let mut rotated = faer::Mat::<f64>::from_fn(n, m, |row, col| y[[row, col]]);
        design_qr.apply_transpose_on_the_left(rotated.as_mut());
        let rotated_head =
            Array2::from_shape_fn((head_rows, m), |(row, col)| rotated[(row, col)]);
        let mut tail = KahanSum::default();
        for col in 0..m {
            for row in head_rows..n {
                tail.add(rotated[(row, col)] * rotated[(row, col)]);
            }
        }
        drop(rotated);

        let (compressed_head, compression_growth) = if m > head_rows {
            let transposed = rotated_head.t().to_owned();
            let compression = HouseholderQr::new(FaerArrayView::new(&transposed).as_ref());
            (
                faer_to_array(compression.r()).t().to_owned(),
                accumulation_growth(m.saturating_mul(head_rows)),
            )
        } else {
            (rotated_head.clone(), 0.0)
        };

        let gram = fast_ata(&design_upper);
        let (rho_lower, rho_upper) =
            crate::estimate::rho_domain::resolvability_domain_from_gram_blocks(
                &gram,
                penalties.iter().map(|penalty| (0..p, penalty)),
                penalties.len(),
            );

        let problem = Self {
            rows,
            observations: n,
            responses: m,
            coefficients: p,
            nullity,
            response_frobenius: frobenius(y),
            design_upper,
            rotated_head,
            compressed_head,
            unpenalized_residual: tail.sum(),
            penalties: penalties.to_vec(),
            roots,
            penalty_rounding,
            pseudo_penalty_rounding,
            design_rotation_growth: accumulation_growth(n.saturating_mul(p)) + whitening_growth,
            compression_growth,
            rho_lower,
            rho_upper,
            design_fingerprint,
            response_fingerprint,
            structural_singular_margin: (
                partition
                    .singular_values
                    .get(partition.rank.saturating_sub(1))
                    .copied()
                    .unwrap_or(0.0),
                factor_singular_band(
                    stacked_rows,
                    p,
                    partition.singular_values.first().copied().unwrap_or(0.0),
                ),
            ),
            support,
            observation_measure,
            observation_measure_roundoff,
        };
        // `∂q/∂ρ_k = λ_k tr(B̂ᵀS_kB̂) ≥ 0`, so a residual resolved at the domain's lower corner is resolved on the
        // whole domain; an interpolated response has no finite profiled dispersion and is refused.
        // This is the evaluator's own predicate (`q > δq`). The refusal names the pooled residual over every column
        // (`output` 0) at the lower corner, whose smallest coordinate the scalar `rho` field carries.
        let (residual, residual_roundoff) = problem.residual_quadratic_at(problem.rho_lower.view())?;
        if !(residual > residual_roundoff) {
            return Err(EstimationError::ProfiledResidualUnresolved {
                output: 0,
                rho: problem.rho_lower.iter().fold(f64::INFINITY, |acc, value| acc.min(*value)),
                residual,
                resolution: residual_roundoff,
                ywy: problem.response_frobenius * problem.response_frobenius,
                design_columns: p,
                observations: n,
            });
        }
        Ok(problem)
    }

    fn stacked_root_factor(&self, rho: ArrayView1<'_, f64>) -> Result<StackedRootFactor, EstimationError> {
        if rho.len() != self.penalties.len() {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML expects {} log-strengths, got {}",
                self.penalties.len(),
                rho.len()
            );
        }
        let lambdas = Array1::from_vec(
            gam_problem::checked_exp_log_strengths(rho.iter().copied())
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        );
        let p = self.coefficients;
        let scaled_roots: Vec<Array2<f64>> = self
            .roots
            .iter()
            .zip(lambdas.iter())
            .map(|(root, &lambda)| root.mapv(|value| lambda.sqrt() * value))
            .collect();
        let rows = self.design_upper.nrows() + scaled_roots.iter().map(Array2::nrows).sum::<usize>();
        let mut stacked = faer::Mat::<f64>::zeros(rows, p);
        let mut offset = 0;
        for block in std::iter::once(&self.design_upper).chain(scaled_roots.iter()) {
            for row in 0..block.nrows() {
                for col in 0..p {
                    stacked[(offset + row, col)] = block[[row, col]];
                }
            }
            offset += block.nrows();
        }
        let mut energy = KahanSum::default();
        for col in 0..p {
            for row in 0..rows {
                energy.add(stacked[(row, col)] * stacked[(row, col)]);
            }
        }
        let qr = HouseholderQr::new(stacked.as_ref());
        let upper = qr.r().to_owned();
        if (0..p).any(|index| !(upper[(index, index)].is_finite() && upper[(index, index)] != 0.0)) {
            return Err(EstimationError::TrialPointRefused {
                reason: "multi-penalty Gaussian REML penalized normal matrix is singular at this ρ".to_string(),
            });
        }
        let penalty_rounding = self
            .penalty_rounding
            .iter()
            .zip(lambdas.iter())
            .map(|(band, lambda)| lambda * band)
            .sum();
        Ok(StackedRootFactor {
            lambdas,
            scaled_roots,
            qr,
            upper,
            rows,
            frobenius: energy.sum().sqrt(),
            penalty_rounding,
            rotation_growth: accumulation_growth(rows.saturating_mul(p)),
        })
    }

    fn project(&self, factor: &StackedRootFactor, head: ArrayView2<'_, f64>) -> StackedProjection {
        let p = self.coefficients;
        let columns = head.ncols();
        let mut target = faer::Mat::<f64>::zeros(factor.rows, columns);
        for row in 0..head.nrows() {
            for col in 0..columns {
                target[(row, col)] = head[[row, col]];
            }
        }
        factor.qr.apply_transpose_on_the_left(target.as_mut());
        let mut tail = KahanSum::default();
        for col in 0..columns {
            for row in p..factor.rows {
                tail.add(target[(row, col)] * target[(row, col)]);
            }
        }
        StackedProjection {
            head: faer::Mat::<f64>::from_fn(p, columns, |row, col| target[(row, col)]),
            tail_energy: tail.sum(),
        }
    }

    fn solve_upper(factor: &StackedRootFactor, mut rhs: faer::Mat<f64>) -> Array2<f64> {
        faer::linalg::triangular_solve::solve_upper_triangular_in_place(
            factor.upper.as_ref(),
            rhs.as_mut(),
            decomposition_parallelism(),
        );
        faer_to_array(rhs.as_ref())
    }

    /// `(q, bound on |δq|)` at `ρ`; the bound is the residual term of [`Self::evaluate`].
    fn residual_quadratic_at(&self, rho: ArrayView1<'_, f64>) -> Result<(f64, f64), EstimationError> {
        let factor = self.stacked_root_factor(rho)?;
        let projection = self.project(&factor, self.compressed_head.view());
        let fitted = Self::solve_upper(&factor, projection.head);
        let residual = self.unpenalized_residual + projection.tail_energy;
        Ok((
            residual,
            self.residual_roundoff(&factor, residual, projection.tail_energy, frobenius(fitted.view())),
        ))
    }

    /// Envelope-theorem bound on `|δq|`: the data and design perturbations `ΔY`, `ΔA` (QR backward errors) move the
    /// minimum by `2⟨r, ΔY − ΔA·B̂⟩`, the roots' formation error by `tr(B̂ᵀΔS_λB̂)`, plus the compensated sums.
    fn residual_roundoff(
        &self,
        factor: &StackedRootFactor,
        residual: f64,
        penalized_residual: f64,
        coefficient_frobenius: f64,
    ) -> f64 {
        let rotation = self.design_rotation_growth + factor.rotation_growth;
        let perturbation = rotation * (self.response_frobenius + factor.frobenius * coefficient_frobenius)
            + self.compression_growth * self.response_frobenius;
        2.0 * residual.sqrt() * perturbation
            + perturbation * perturbation
            + factor.penalty_rounding * coefficient_frobenius * coefficient_frobenius
            + compensated_band(1, self.unpenalized_residual)
            + compensated_band(1, penalized_residual)
            + UNIT_ROUNDOFF * residual
    }

    /// The criterion, its analytic ρ-gradient and ρ-Hessian, and its parts at `ρ`.
    pub fn evaluate(
        &self,
        rho: ArrayView1<'_, f64>,
    ) -> Result<GaussianRemlMultiPenaltyEvaluation, EstimationError> {
        let factor = self.stacked_root_factor(rho)?;
        let p = self.coefficients;
        let penalty_count = self.penalties.len();
        let m = self.responses as f64;

        let projection = self.project(&factor, self.compressed_head.view());
        let penalized_residual = projection.tail_energy;
        let fitted = Self::solve_upper(&factor, projection.head);
        let coefficient_frobenius = frobenius(fitted.view());
        let residual = self.unpenalized_residual + penalized_residual;
        let residual_roundoff =
            self.residual_roundoff(&factor, residual, penalized_residual, coefficient_frobenius);
        if !(residual > residual_roundoff) {
            return Err(EstimationError::TrialPointRefused {
                reason: format!(
                    "multi-penalty Gaussian REML residual quadratic {residual:.3e} does not exceed its rounding \
                     bound {residual_roundoff:.3e} at this ρ"
                ),
            });
        }

        let inverse = Self::solve_upper(&factor, faer::Mat::<f64>::identity(p, p));
        let inverse_trace = frobenius(inverse.view()).powi(2);
        let rotation = self.design_rotation_growth + factor.rotation_growth;

        let mut log_det = KahanSum::default();
        let mut log_det_magnitude = 0.0;
        for index in 0..p {
            let term = 2.0 * factor.upper[(index, index)].abs().ln();
            log_det.add(term);
            log_det_magnitude += term.abs();
        }
        let log_det_normal = log_det.sum();
        let log_det_normal_roundoff = 2.0 * inverse_trace.sqrt() * rotation * factor.frobenius
            + inverse_trace * factor.penalty_rounding
            + accumulation_growth(p) * log_det_magnitude;

        let shrink: Vec<Array2<f64>> =
            factor.scaled_roots.iter().map(|root| fast_ab(root, &inverse)).collect();
        let moved: Vec<Array2<f64>> =
            factor.scaled_roots.iter().map(|root| fast_ab(root, &fitted)).collect();
        let adjoint: Vec<Array2<f64>> = shrink
            .iter()
            .zip(moved.iter())
            .map(|(left, right)| fast_ab(&left.t(), right))
            .collect();
        let penalty_trace =
            Array1::from_iter(shrink.iter().map(|block| frobenius(block.view()).powi(2)));
        let residual_gradient =
            Array1::from_iter(moved.iter().map(|block| frobenius(block.view()).powi(2)));

        let lambdas = factor.lambdas.to_vec();
        let pseudo = PenaltyPseudologdet::from_components(&self.penalties, &lambdas, 0.0)
            .map_err(|reason| EstimationError::TrialPointRefused { reason })?;
        if pseudo.rank() != p - self.nullity {
            return Err(EstimationError::TrialPointRefused {
                reason: format!(
                    "multi-penalty Gaussian REML penalty pseudo-determinant priced rank {} at this ρ, but the \
                     problem's structural rank is {}",
                    pseudo.rank(),
                    p - self.nullity
                ),
            });
        }
        let (pseudo_gradient, pseudo_hessian) = pseudo.rho_derivatives(&self.penalties, &lambdas);
        let log_pseudo_det = pseudo.value();
        // `inv_evals_sq[i] = σ_i⁻²` over `S_λ`'s positive spectrum: `‖A_S⁺‖²_F = tr(S_λ⁺) = Σ 1/σ_i` and
        // `‖A_S‖₂ = √σ_max` for the stacked root `A_S` the owner factors.
        let mut pseudo_inverse_trace = 0.0;
        let mut sigma_max = 0.0_f64;
        let mut log_pseudo_magnitude = 0.0;
        for &inverse_square in pseudo.inv_evals_sq.iter() {
            pseudo_inverse_trace += inverse_square.sqrt();
            sigma_max = sigma_max.max(inverse_square.sqrt().recip());
            log_pseudo_magnitude += 0.5 * inverse_square.ln().abs();
        }
        let stacked_root_rows = self.roots.iter().map(Array2::nrows).sum::<usize>().max(p);
        let pseudo_penalty_rounding: f64 = self
            .pseudo_penalty_rounding
            .iter()
            .zip(factor.lambdas.iter())
            .map(|(band, lambda)| lambda * band)
            .sum();
        let log_pseudo_det_roundoff = 2.0
            * pseudo_inverse_trace.sqrt()
            * (p as f64).sqrt()
            * factor_singular_band(stacked_root_rows, p, sigma_max.sqrt())
            + pseudo_inverse_trace * pseudo_penalty_rounding
            + accumulation_growth(pseudo.rank()) * log_pseudo_magnitude;

        let nu = m * (self.observations - self.nullity) as f64;
        let dispersion = 0.5 * nu * (1.0 + (2.0 * PI * residual / nu).ln());
        let determinant = 0.5 * m * (log_det_normal - log_pseudo_det);
        let reml_score = dispersion + determinant + self.observation_measure;
        // A weighted score adds the observation measure: its own bound plus the one rounding of that addition. The
        // unweighted measure is an exact zero.
        let measure_roundoff = if self.support.is_some() {
            self.observation_measure_roundoff + UNIT_ROUNDOFF * reml_score.abs()
        } else {
            0.0
        };
        let reml_score_roundoff = 0.5 * nu * residual_roundoff / residual
            + 0.5 * m * (log_det_normal_roundoff + log_pseudo_det_roundoff)
            + accumulation_growth(6)
                * (dispersion.abs() + 0.5 * m * (log_det_normal.abs() + log_pseudo_det.abs()))
            + measure_roundoff;

        let mut reml_gradient = Array1::<f64>::zeros(penalty_count);
        let mut reml_hessian = Array2::<f64>::zeros((penalty_count, penalty_count));
        for k in 0..penalty_count {
            reml_gradient[k] = 0.5
                * (nu / residual * residual_gradient[k] + m * penalty_trace[k]
                    - m * pseudo_gradient[k]);
            for j in 0..=k {
                let diagonal = if j == k { 1.0 } else { 0.0 };
                let mut coupling = 0.0;
                for (left, right) in adjoint[k].iter().zip(adjoint[j].iter()) {
                    coupling += left * right;
                }
                let trace_pair = frobenius(fast_ab(&shrink[k], &shrink[j].t()).view()).powi(2);
                let residual_hessian = diagonal * residual_gradient[k] - 2.0 * coupling;
                let value = 0.5
                    * (nu / residual * residual_hessian
                        - nu / (residual * residual) * residual_gradient[k] * residual_gradient[j]
                        + m * (diagonal * penalty_trace[k] - trace_pair)
                        - m * pseudo_hessian[[k, j]]);
                reml_hessian[[k, j]] = value;
                reml_hessian[[j, k]] = value;
            }
        }

        let trace_total: f64 = penalty_trace.sum();
        let edf = p as f64 - trace_total;
        let squared_entries: usize = shrink.iter().map(Array2::len).sum();
        let edf_roundoff = inverse_trace
            * (factor.penalty_rounding + 2.0 * rotation * factor.frobenius * factor.frobenius)
            + accumulation_growth(squared_entries + p + 1) * trace_total;

        if !(reml_score.is_finite()
            && reml_score_roundoff.is_finite()
            && reml_gradient.iter().all(|value| value.is_finite())
            && reml_hessian.iter().all(|value| value.is_finite()))
        {
            return Err(EstimationError::TrialPointRefused {
                reason: "multi-penalty Gaussian REML evaluation produced a non-finite value".to_string(),
            });
        }
        Ok(GaussianRemlMultiPenaltyEvaluation {
            rho: rho.to_owned(),
            lambdas: factor.lambdas,
            reml_score,
            reml_score_roundoff,
            reml_gradient,
            reml_hessian,
            residual_quadratic: residual,
            residual_quadratic_roundoff: residual_roundoff,
            dispersion_dof: nu,
            sigma2: residual / nu,
            log_det_penalized_normal: log_det_normal,
            log_det_penalized_normal_roundoff: log_det_normal_roundoff,
            log_pseudo_det_penalty: log_pseudo_det,
            log_pseudo_det_penalty_roundoff: log_pseudo_det_roundoff,
            penalty_trace,
            edf,
            edf_roundoff,
        })
    }

    /// Per-column coefficients `B̂` (`p × m`) at `ρ`, with a first-order bound on `‖δB̂‖_F` from the least-squares
    /// perturbation `δB = A⁺(ΔY − ΔA·B̂) + K⁻¹ΔAᵀr − K⁻¹ΔS_λB̂`.
    pub fn coefficients(&self, rho: ArrayView1<'_, f64>) -> Result<(Array2<f64>, f64), EstimationError> {
        let factor = self.stacked_root_factor(rho)?;
        let projection = self.project(&factor, self.rotated_head.view());
        let penalized_residual = projection.tail_energy;
        let coefficients = Self::solve_upper(&factor, projection.head);
        let p = self.coefficients;
        let inverse = Self::solve_upper(&factor, faer::Mat::<f64>::identity(p, p));
        let inverse_frobenius = frobenius(inverse.view());
        let coefficient_frobenius = frobenius(coefficients.view());
        let rotation = self.design_rotation_growth + factor.rotation_growth;
        let roundoff = inverse_frobenius
            * rotation
            * (self.response_frobenius + factor.frobenius * coefficient_frobenius)
            + inverse_frobenius
                * inverse_frobenius
                * (rotation * factor.frobenius * penalized_residual.sqrt()
                    + factor.penalty_rounding * coefficient_frobenius);
        Ok((coefficients, roundoff))
    }

    /// `∂V/∂X` and `∂V/∂Y` at the fit's `ρ̂`, for the `x` and `y` this problem was built from.
    ///
    /// `V = ½[ν(1 + log 2πq/ν) + m·log|K| − m·log|S_λ|₊]` with `q` minimized over `B` and `σ²` profiled, so by the
    /// envelope theorem `∂q/∂X = −2RB̂ᵀ` and `∂q/∂Y = 2R`, while `∂log|K|/∂X = 2XK⁻¹` and `log|S_λ|₊` and `ν` do not
    /// depend on the data. Hence `∂V/∂X = m·XK⁻¹ − RB̂ᵀ/σ̂²` and `∂V/∂Y = R/σ̂²` at fixed `ρ`. When every `ρ̂_k` is
    /// interior, `∂V/∂ρ = 0` and these are the total derivatives to first order. A railed coordinate sits on a domain
    /// edge derived from `XᵀX`, so the edge moves with `X` and `∂V/∂ρ_k ≠ 0` there; that case is returned typed as
    /// [`GaussianRemlMultiPenaltyDataGradientOutcome::RhoAtDomainBound`] instead of an incomplete gradient.
    ///
    /// A weighted problem's `V` reads the data through `x̃_i = √w_i·x_i` and `ỹ_i = √w_i·y_i`, so its cotangents are
    /// these forms in whitened coordinates scaled back by `√w_i`, and zero on the rows its weights omit.
    pub fn data_gradient(
        &self,
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        fit: &GaussianRemlMultiPenaltyFit,
    ) -> Result<GaussianRemlMultiPenaltyDataGradientOutcome, EstimationError> {
        if fit
            .rho_placement
            .iter()
            .any(|placement| *placement != GaussianRemlMultiPenaltyRhoPlacement::Interior)
        {
            return Ok(GaussianRemlMultiPenaltyDataGradientOutcome::RhoAtDomainBound {
                placement: fit.rho_placement.clone(),
            });
        }
        let (n, p, m) = (self.rows, self.coefficients, self.responses);
        if x.dim() != (n, p) || y.dim() != (n, m) || fit.coefficients.dim() != (p, m) {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML data gradient shape mismatch: problem n={n}, p={p}, m={m}; got \
                 X {:?}, Y {:?}, coefficients {:?}",
                x.dim(),
                y.dim(),
                fit.coefficients.dim()
            );
        }
        if array2_bits_fingerprint(&x) != self.design_fingerprint {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML data gradient refuses `x`: its values differ from the design this \
                 problem was built from, so the fit is not converged at it"
            );
        }
        if array2_bits_fingerprint(&y) != self.response_fingerprint {
            crate::bail_invalid_estim!(
                "multi-penalty Gaussian REML data gradient refuses `y`: its values differ from the responses this \
                 problem was built from, so the fit is not converged at them"
            );
        }
        let whitened = self
            .support
            .as_ref()
            .map(|support| (support.whiten(x), support.whiten(y)));
        let (x, y) = match &whitened {
            Some((x, y)) => (x.view(), y.view()),
            None => (x, y),
        };
        let n = x.nrows();
        let factor = self.stacked_root_factor(fit.evaluation.rho.view())?;
        let sigma2 = fit.evaluation.sigma2;
        let residual = &y - &fast_ab(&x, &fit.coefficients);
        // `K⁻¹Xᵀ = R_A⁻¹R_A⁻ᵀXᵀ`: a lower then an upper triangular solve on the `p × n` right-hand side.
        let mut transposed = faer::Mat::<f64>::from_fn(p, n, |row, col| x[[col, row]]);
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            factor.upper.as_ref().transpose(),
            transposed.as_mut(),
            decomposition_parallelism(),
        );
        let inverse_xt = Self::solve_upper(&factor, transposed);
        let fitted_outer = fast_ab(&residual, &fit.coefficients.t());
        let grad_x = Array2::from_shape_fn((n, p), |(row, col)| {
            m as f64 * inverse_xt[[col, row]] - fitted_outer[[row, col]] / sigma2
        });
        let grad_y = residual.mapv(|value| value / sigma2);
        let (grad_x, grad_y) = match &self.support {
            None => (grad_x, grad_y),
            Some(support) => {
                let mut observed_x = Array2::<f64>::zeros((self.rows, p));
                let mut observed_y = Array2::<f64>::zeros((self.rows, m));
                for (index, (&row, &root)) in support.rows.iter().zip(support.root_weights.iter()).enumerate() {
                    observed_x.row_mut(row).assign(&grad_x.row(index).mapv(|value| root * value));
                    observed_y.row_mut(row).assign(&grad_y.row(index).mapv(|value| root * value));
                }
                (observed_x, observed_y)
            }
        };
        Ok(GaussianRemlMultiPenaltyDataGradientOutcome::Interior(
            GaussianRemlMultiPenaltyDataGradient { grad_x, grad_y },
        ))
    }

    /// `∂V/∂S_k` at the fit's `ρ̂`.
    ///
    /// With `q` minimized over `B` and `σ²` profiled, `dq = λ_k tr(B̂B̂ᵀ dS_k)`, `d log|K| = λ_k tr(K⁻¹ dS_k)` and, while
    /// `rank(S_λ)` is locally constant, `d log|S_λ|₊ = λ_k tr(S_λ⁺ dS_k)` with `ν` fixed, so
    /// `∂V/∂S_k = ½λ_k(B̂B̂ᵀ/σ̂² + m·K⁻¹ − m·S_λ⁺)`. A perturbation that adds rank is outside this derivative's domain.
    /// `B̂B̂ᵀ` is read off the compressed head: `CCᵀ = ZZᵀ` makes it equal the full coefficients' Gram.
    pub fn penalty_gradient(
        &self,
        fit: &GaussianRemlMultiPenaltyFit,
    ) -> Result<GaussianRemlMultiPenaltyPenaltyGradientOutcome, EstimationError> {
        if fit
            .rho_placement
            .iter()
            .any(|placement| *placement != GaussianRemlMultiPenaltyRhoPlacement::Interior)
        {
            return Ok(GaussianRemlMultiPenaltyPenaltyGradientOutcome::RhoAtDomainBound {
                placement: fit.rho_placement.clone(),
            });
        }
        let p = self.coefficients;
        let m = self.responses as f64;
        let factor = self.stacked_root_factor(fit.evaluation.rho.view())?;
        let projection = self.project(&factor, self.compressed_head.view());
        let fitted = Self::solve_upper(&factor, projection.head);
        let inverse = Self::solve_upper(&factor, faer::Mat::<f64>::identity(p, p));
        let normal_inverse = fast_ab(&inverse, &inverse.t());
        let coefficient_gram = fast_ab(&fitted, &fitted.t());
        let lambdas = factor.lambdas.to_vec();
        let pseudo = PenaltyPseudologdet::from_components(&self.penalties, &lambdas, 0.0)
            .map_err(|reason| EstimationError::TrialPointRefused { reason })?;
        if pseudo.rank() != p - self.nullity {
            return Err(EstimationError::TrialPointRefused {
                reason: format!(
                    "multi-penalty Gaussian REML penalty gradient: the pseudo-determinant priced rank {} at ρ̂, but \
                     the structural rank is {}",
                    pseudo.rank(),
                    p - self.nullity
                ),
            });
        }
        let pseudo_inverse = fast_ab(&pseudo.w_factor, &pseudo.w_factor.t());
        let sigma2 = fit.evaluation.sigma2;
        let gradients = factor
            .lambdas
            .iter()
            .map(|&lambda| {
                Array2::from_shape_fn((p, p), |(row, col)| {
                    0.5 * lambda
                        * (coefficient_gram[[row, col]] / sigma2 + m * normal_inverse[[row, col]]
                            - m * pseudo_inverse[[row, col]])
                })
            })
            .collect();
        let (smallest_resolved_singular_value, rank_band) = self.structural_singular_margin;
        Ok(GaussianRemlMultiPenaltyPenaltyGradientOutcome::Interior {
            gradients,
            rank: p - self.nullity,
            smallest_resolved_singular_value,
            rank_band,
        })
    }

    /// Minimize `V(ρ)` through the outer optimizer's analytic-Hessian route on the derived domain. Only a converged,
    /// certified optimum returns a fit; `initial_rho` is a warm start.
    pub fn fit(
        &self,
        initial_rho: Option<ArrayView1<'_, f64>>,
    ) -> Result<GaussianRemlMultiPenaltyFit, EstimationError> {
        let penalty_count = self.penalties.len();
        if let Some(rho) = initial_rho {
            if rho.len() != penalty_count || rho.iter().any(|value| !value.is_finite()) {
                crate::bail_invalid_estim!(
                    "multi-penalty Gaussian REML initial ρ must hold {penalty_count} finite values; got {rho:?}"
                );
            }
        }
        // The criterion sums `m` response columns that share `λ` and one
        // dispersion (`ν = m·(n − nullity)`), so the information about `ρ` is
        // carried by `m·n` observations, not `n`: that count is the `n_eff` the
        // outer certificate's resolution `τ_stat = 1/(2·n_eff)` is over (#3192).
        // It is also the number of squared residuals `V` accumulates.
        let (observations, coefficients) = (self.responses * self.observations, self.coefficients);
        let (lower, upper) = (self.rho_lower.clone(), self.rho_upper.clone());
        // One start: the caller's ρ, else each penalty's commensurate-curvature
        // start (`XᵀX = RᵀR` over the penalty's support against `tr S_k`),
        // clamped into the ρ domain. The certified outer search refines it.
        let start_rho = match initial_rho {
            Some(rho) => rho.to_owned(),
            None => {
                let gram_diag = self.design_upper.map_axis(ndarray::Axis(0), |column| {
                    column.iter().map(|value| value * value).sum::<f64>()
                });
                Array1::from_iter(self.penalties.iter().map(|penalty| {
                    let diagonal = penalty.diag();
                    crate::seeding::commensurate_curvature_rho(
                        gram_diag.view(),
                        (0..diagonal.len()).filter(|&c| diagonal[c] > 0.0),
                        diagonal.sum(),
                    )
                    .unwrap_or(0.0)
                }))
            }
        };
        let start_rho = Array1::from_iter(
            start_rho
                .iter()
                .zip(lower.iter().zip(upper.iter()))
                .map(|(value, (lo, hi))| value.max(*lo).min(*hi)),
        );
        let problem = OuterProblem::new(penalty_count)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Dense)
            .with_prefer_gradient_only(false)
            .with_disable_fixed_point(true)
            .with_bounds(lower.clone(), upper.clone())
            .with_fallback_policy(FallbackPolicy::Disabled)
            .with_problem_size(observations, coefficients)
            .with_initial_rho(start_rho);
        let mut objective = problem.build_objective(
            self.clone(),
            multi_penalty_cost,
            multi_penalty_outer_eval,
            None::<fn(&mut GaussianRemlMultiPenaltyProblem)>,
            None::<
                fn(
                    &mut GaussianRemlMultiPenaltyProblem,
                    &Array1<f64>,
                ) -> Result<gam_problem::EfsEval, EstimationError>,
            >,
        );
        let optimum = problem.run(&mut objective, "multi-penalty Gaussian REML")?;
        let evaluation = objective.state.evaluate(optimum.rho.view())?;
        let (coefficients, coefficients_roundoff) =
            objective.state.coefficients(optimum.rho.view())?;
        let rho_placement = match optimum.criterion_certificate.as_ref() {
            None => vec![GaussianRemlMultiPenaltyRhoPlacement::Unaudited; penalty_count],
            Some(certificate) => {
                let mut placement = vec![GaussianRemlMultiPenaltyRhoPlacement::Interior; penalty_count];
                for &index in certificate.lambdas_railed.iter().filter(|&&index| index < penalty_count) {
                    let rho = optimum.rho[index];
                    placement[index] = if rho - lower[index] <= upper[index] - rho {
                        GaussianRemlMultiPenaltyRhoPlacement::LowerBound
                    } else {
                        GaussianRemlMultiPenaltyRhoPlacement::UpperBound
                    };
                }
                placement
            }
        };
        Ok(GaussianRemlMultiPenaltyFit {
            evaluation,
            coefficients,
            coefficients_roundoff,
            rho_lower: lower,
            rho_upper: upper,
            iterations: optimum.iterations,
            certificate: optimum.criterion_certificate,
            rho_placement,
        })
    }
}

fn multi_penalty_cost(
    state: &mut GaussianRemlMultiPenaltyProblem,
    rho: &Array1<f64>,
) -> Result<f64, EstimationError> {
    Ok(state.evaluate(rho.view())?.reml_score)
}

fn multi_penalty_outer_eval(
    state: &mut GaussianRemlMultiPenaltyProblem,
    rho: &Array1<f64>,
) -> Result<OuterEval, EstimationError> {
    let evaluation = state.evaluate(rho.view())?;
    Ok(OuterEval {
        cost: evaluation.reml_score,
        gradient: evaluation.reml_gradient,
        hessian: HessianValue::Dense(evaluation.reml_hessian),
        inner_beta_hint: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian_reml::{
        gaussian_reml_multi_closed_form, gaussian_reml_multi_shared_dispersion_closed_form,
    };
    use gam_linalg::faer_ndarray::fast_atb;
    use gam_linalg::utils::certified_spd_inverse;
    use ndarray::{Axis, array};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, StandardNormal};

    /// `cos(π j x_i)` at `x_i = (i + ½)/n`: a smooth, well-conditioned basis.
    fn cosine_design(n: usize, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, p), |(row, col)| {
            (PI * col as f64 * (row as f64 + 0.5) / n as f64).cos()
        })
    }

    /// `diag(j⁴)` for `j ≥ unpenalized`, zero on the first `unpenalized` modes.
    fn curvature_penalty(p: usize, unpenalized: usize) -> Array2<f64> {
        Array2::from_shape_fn((p, p), |(row, col)| {
            if row == col && row >= unpenalized { (row as f64).powi(4) } else { 0.0 }
        })
    }

    fn normal_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || StandardNormal.sample(&mut *rng))
    }

    fn smooth_truth(x_position: f64, phase: f64) -> f64 {
        (2.0 * PI * x_position + phase).sin() + 0.5 * (PI * x_position).cos()
    }

    fn responses(n: usize, noise: &[f64], rng: &mut StdRng) -> Array2<f64> {
        Array2::from_shape_fn((n, noise.len()), |(row, col)| {
            let position = (row as f64 + 0.5) / n as f64;
            let draw: f64 = StandardNormal.sample(&mut *rng);
            smooth_truth(position, col as f64) + noise[col] * draw
        })
    }

    /// A converged point's distance to the criterion's minimizer, first order: the Newton step `|g|/h` plus the
    /// score-resolution radius `√(2·band/h)` below which no comparison of criterion values separates two points.
    fn resolution_radius(gradient_norm: f64, curvature: f64, band: f64) -> f64 {
        gradient_norm / curvature + (2.0 * band / curvature).sqrt()
    }

    /// [`resolution_radius`] of a fit, at its Hessian's smallest curvature.
    fn fitted_radius(fit: &GaussianRemlMultiPenaltyFit) -> f64 {
        let curvatures = fit
            .evaluation
            .reml_hessian
            .eigh(Side::Lower)
            .expect("the fitted Hessian's spectrum")
            .0;
        assert!(curvatures[0] > 0.0, "the fitted criterion must curve upward; λ_min={}", curvatures[0]);
        resolution_radius(
            frobenius(fit.evaluation.reml_gradient.view().insert_axis(Axis(1))),
            curvatures[0],
            fit.evaluation.reml_score_roundoff,
        )
    }

    #[test]
    fn gaussian_reml_multi_penalty_one_column_one_penalty_reproduces_closed_form() {
        let (n, p) = (120, 8);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_001);
        let y = responses(n, &[0.3], &mut rng);
        let penalty = curvature_penalty(p, 2);

        let old = gaussian_reml_multi_closed_form(x.view(), y.view(), penalty.view(), None, None)
            .expect("closed-form Gaussian REML fits the one-column fixture");
        let old_roundoff = old
            .reml_score_roundoff
            .expect("the closed-form evaluator publishes its score roundoff");
        let problem =
            GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &[penalty.clone()], 2)
                .expect("the multi-penalty problem accepts the one-column fixture");
        let at_old = problem
            .evaluate(array![old.rho].view())
            .expect("the multi-penalty evaluator evaluates at the closed-form optimum");

        // Score at the same ρ: two evaluations of one criterion differ by at most both rounding bounds.
        let score_band = at_old.reml_score_roundoff + old_roundoff;
        let score_gap = (at_old.reml_score - old.reml_score).abs();
        assert!(
            score_gap <= score_band,
            "score at the closed-form ρ̂: multi-penalty {} vs closed form {}, gap {score_gap:.3e} exceeds band \
             {score_band:.3e}",
            at_old.reml_score,
            old.reml_score
        );
        // Positive control: a ρ shift whose quadratic score change is nine bands is resolved by that same band.
        let curvature = at_old.reml_hessian[[0, 0]];
        assert!(curvature > 0.0, "the criterion must curve upward at its optimum; h={curvature:.3e}");
        let shift = 3.0 * (2.0 * score_band / curvature).sqrt();
        let shifted = problem
            .evaluate(array![old.rho + shift].view())
            .expect("the evaluator evaluates beside the optimum");
        assert!(
            (shifted.reml_score - old.reml_score).abs() > score_band,
            "positive control: a shift of {shift:.3e} in ρ changed the score by {:.3e}, inside the band {score_band:.3e}",
            (shifted.reml_score - old.reml_score).abs()
        );

        // edf at the same ρ. The closed form reads `Σ 1/(1 + λδ_i)` off the QR-whitened penalty's singular values,
        // so its first-order error is `tr(K⁻¹)·(‖ΔG‖₂ + λ‖ΔS‖₂)` with the design QR's `‖ΔG‖₂ ≤ 2γ_{np}‖X‖²_F`, the
        // penalty eigensolver's `p·ε·‖S‖₂` and the whitened-root SVD's `2p·ε·δ_max` mapped back through `‖XᵀX‖₂`.
        let lambda = old.rho.exp();
        let gram = fast_ata(&x);
        let normal = &gram + &penalty.mapv(|value| lambda * value);
        let inverse = certified_spd_inverse(
            &normal,
            SymmetricAssembly::Mirrored,
            "test penalized normal matrix",
        )
        .expect("the fixture's penalized normal matrix is SPD");
        let inverse_trace = inverse.inverse().diag().sum();
        let design_energy = frobenius(x.view()).powi(2);
        let gram_norm = gram.iter().fold(0.0_f64, |acc, value| acc.max(value.abs())) * p as f64;
        let delta_max = old
            .cache
            .penalty_eigenvalues
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(*value));
        let penalty_norm = (p as f64 - 1.0).powi(4);
        let old_edf_band = inverse_trace
            * (2.0 * accumulation_growth(n * p) * design_energy
                + lambda
                    * (p as f64 * f64::EPSILON * penalty_norm
                        + 2.0 * p as f64 * f64::EPSILON * delta_max * gram_norm))
            + accumulation_growth(2 * p) * p as f64;
        let edf_band = at_old.edf_roundoff + old_edf_band;
        assert!(
            (at_old.edf - old.edf).abs() <= edf_band,
            "edf at the closed-form ρ̂: multi-penalty {} vs closed form {}, band {edf_band:.3e}",
            at_old.edf,
            old.edf
        );

        // Coefficients at the same ρ. The closed form's first-order error is the least-squares perturbation bound
        // `√tr(K⁻¹)·γ_{np}(‖y‖ + ‖X‖_F‖β‖) + tr(K⁻¹)·(γ_{np}‖X‖_F‖r‖ + λ‖ΔS‖₂‖β‖)` with the same `‖ΔS‖₂` as above.
        let (at_old_coefficients, at_old_coefficients_roundoff) = problem
            .coefficients(array![old.rho].view())
            .expect("coefficients evaluate at the closed-form optimum");
        let old_beta = frobenius(old.coefficients.view());
        let old_penalty_rounding = p as f64 * f64::EPSILON * penalty_norm
            + 2.0 * p as f64 * f64::EPSILON * delta_max * gram_norm;
        let whitening = accumulation_growth(n * p);
        let old_coefficient_band = inverse_trace.sqrt()
            * whitening
            * (frobenius(y.view()) + design_energy.sqrt() * old_beta)
            + inverse_trace
                * (whitening * design_energy.sqrt() * at_old.residual_quadratic.sqrt()
                    + lambda * old_penalty_rounding * old_beta);
        let coefficient_gap = frobenius((&at_old_coefficients - &old.coefficients).view());
        let coefficient_band = at_old_coefficients_roundoff + old_coefficient_band;
        assert!(
            coefficient_gap <= coefficient_band,
            "coefficients at the closed-form ρ̂ differ by {coefficient_gap:.3e}, band {coefficient_band:.3e}"
        );

        // λ̂: both optima lie within their resolution radii of the one minimizer.
        let fit = problem.fit(None).expect("the multi-penalty fit converges on the one-column fixture");
        let new_curvature = fit.evaluation.reml_hessian[[0, 0]];
        assert!(new_curvature > 0.0, "the fitted criterion must curve upward; h={new_curvature:.3e}");
        let new_radius = resolution_radius(
            fit.evaluation.reml_gradient[0].abs(),
            new_curvature,
            fit.evaluation.reml_score_roundoff,
        );
        let old_radius = resolution_radius(at_old.reml_gradient[0].abs(), curvature, score_band);
        let rho_gap = (fit.evaluation.rho[0] - old.rho).abs();
        assert!(
            rho_gap <= new_radius + old_radius,
            "ρ̂: multi-penalty {} vs closed form {}, gap {rho_gap:.3e} exceeds the radii {new_radius:.3e} + \
             {old_radius:.3e}",
            fit.evaluation.rho[0],
            old.rho
        );
        let fitted_band = fit.evaluation.reml_score_roundoff
            + score_band
            + new_curvature * (new_radius + old_radius).powi(2);
        assert!(
            (fit.evaluation.reml_score - old.reml_score).abs() <= fitted_band,
            "fitted scores: multi-penalty {} vs closed form {}, band {fitted_band:.3e}",
            fit.evaluation.reml_score,
            old.reml_score
        );
    }

    #[test]
    fn gaussian_reml_multi_penalty_columns_add_at_a_shared_lambda() {
        let (n, p) = (90, 7);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_002);
        let noise = [0.2, 0.5, 1.1];
        let m = noise.len();
        let y = responses(n, &noise, &mut rng);
        let penalties = [curvature_penalty(p, 2), Array2::<f64>::eye(p)];
        let joint = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, 0)
            .expect("the joint problem accepts three columns");
        let rho = joint
            .clone()
            .fit(None)
            .expect("the joint fit converges")
            .evaluation
            .rho;
        let shared = joint.evaluate(rho.view()).expect("the joint evaluation at ρ̂");
        let (shared_coefficients, shared_coefficients_roundoff) =
            joint.coefficients(rho.view()).expect("joint coefficients at ρ̂");
        let columns: Vec<(GaussianRemlMultiPenaltyEvaluation, Array2<f64>, f64)> = (0..m)
            .map(|col| {
                let column = y.column(col).insert_axis(Axis(1));
                let problem = GaussianRemlMultiPenaltyProblem::new(x.view(), column, &penalties, 0)
                    .expect("a one-column problem");
                let evaluation = problem.evaluate(rho.view()).expect("one-column evaluation at ρ̂");
                let (coefficients, roundoff) =
                    problem.coefficients(rho.view()).expect("one-column coefficients at ρ̂");
                (evaluation, coefficients, roundoff)
            })
            .collect();

        let residual_sum: f64 = columns.iter().map(|column| column.0.residual_quadratic).sum();
        let residual_band = shared.residual_quadratic_roundoff
            + columns.iter().map(|column| column.0.residual_quadratic_roundoff).sum::<f64>()
            + accumulation_growth(m) * residual_sum;
        assert!(
            (shared.residual_quadratic - residual_sum).abs() <= residual_band,
            "pooled residual {} vs the columns' sum {residual_sum}, band {residual_band:.3e}",
            shared.residual_quadratic
        );
        let dof_sum: f64 = columns.iter().map(|column| column.0.dispersion_dof).sum();
        assert_eq!(shared.dispersion_dof, dof_sum, "ν adds over columns exactly");
        for (col, (evaluation, coefficients, roundoff)) in columns.iter().enumerate() {
            let normal_band =
                shared.log_det_penalized_normal_roundoff + evaluation.log_det_penalized_normal_roundoff;
            assert!(
                (shared.log_det_penalized_normal - evaluation.log_det_penalized_normal).abs() <= normal_band,
                "column {col}: log|K| {} vs {}, band {normal_band:.3e}",
                evaluation.log_det_penalized_normal,
                shared.log_det_penalized_normal
            );
            let pseudo_band =
                shared.log_pseudo_det_penalty_roundoff + evaluation.log_pseudo_det_penalty_roundoff;
            assert!(
                (shared.log_pseudo_det_penalty - evaluation.log_pseudo_det_penalty).abs() <= pseudo_band,
                "column {col}: log|S|₊ {} vs {}, band {pseudo_band:.3e}",
                evaluation.log_pseudo_det_penalty,
                shared.log_pseudo_det_penalty
            );
            let column_gap = frobenius(
                (&shared_coefficients.column(col).insert_axis(Axis(1)) - coefficients).view(),
            );
            assert!(
                column_gap <= shared_coefficients_roundoff + roundoff,
                "column {col}: coefficients differ by {column_gap:.3e}, band {:.3e}",
                shared_coefficients_roundoff + roundoff
            );
        }

        // The restricted log-likelihood at a FIXED σ² adds over independent columns, so the pooled profiled score is
        // the columns' sum at the shared σ̂². `∂/∂σ² Σ_c W_c = 0` at σ̂², so the shared σ̂²'s own error is second order.
        let sigma2 = shared.sigma2;
        let mut restricted_sum = KahanSum::default();
        let mut magnitude = 0.0;
        let mut column_band = 0.0;
        for (evaluation, ..) in &columns {
            let terms = [
                evaluation.residual_quadratic / sigma2,
                evaluation.dispersion_dof * (2.0 * PI * sigma2).ln(),
                evaluation.log_det_penalized_normal,
                -evaluation.log_pseudo_det_penalty,
            ];
            for term in terms {
                restricted_sum.add(0.5 * term);
                magnitude += 0.5 * term.abs();
            }
            column_band += 0.5
                * (evaluation.residual_quadratic_roundoff / sigma2
                    + evaluation.log_det_penalized_normal_roundoff
                    + evaluation.log_pseudo_det_penalty_roundoff);
        }
        let identity_band =
            shared.reml_score_roundoff + column_band + accumulation_growth(4 * m + 6) * magnitude;
        let identity_gap = (shared.reml_score - restricted_sum.sum()).abs();
        assert!(
            identity_gap <= identity_band,
            "pooled score {} vs the columns' restricted likelihoods at σ̂² {}, gap {identity_gap:.3e}, band \
             {identity_band:.3e}",
            shared.reml_score,
            restricted_sum.sum()
        );
        // Positive control: letting each column profile its own dispersion is a different criterion, and the band
        // resolves the difference.
        let own_dispersion_sum: f64 = columns.iter().map(|column| column.0.reml_score).sum();
        assert!(
            (shared.reml_score - own_dispersion_sum).abs() > identity_band,
            "positive control: per-column dispersions {own_dispersion_sum} sit inside the band {identity_band:.3e} \
             of the pooled score {}",
            shared.reml_score
        );
    }

    #[test]
    fn gaussian_reml_multi_penalty_invariant_to_rotating_the_output_columns() {
        let (n, p, m) = (80, 6, 10);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_003);
        let noise: Vec<f64> = (0..m).map(|col| 0.3 + 0.1 * col as f64).collect();
        let y = responses(n, &noise, &mut rng);
        let gaussian = normal_matrix(m, m, &mut rng);
        let reflector = HouseholderQr::new(FaerArrayView::new(&gaussian).as_ref());
        let mut orthogonal = faer::Mat::<f64>::identity(m, m);
        reflector.apply_transpose_on_the_left(orthogonal.as_mut());
        let rotation = faer_to_array(orthogonal.as_ref());
        let defect = frobenius((&fast_ata(&rotation) - &Array2::<f64>::eye(m)).view());
        let rotated = fast_ab(&y, &rotation);
        let penalties = [curvature_penalty(p, 2), Array2::<f64>::eye(p)];
        let original = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, 0)
            .expect("the original problem");
        let turned = GaussianRemlMultiPenaltyProblem::new(x.view(), rotated.view(), &penalties, 0)
            .expect("the rotated problem");

        let fits = [
            original.clone().fit(None).expect("the original fit converges"),
            turned.clone().fit(None).expect("the rotated fit converges"),
        ];
        let radii: Vec<f64> = fits.iter().map(fitted_radius).collect();
        for k in 0..penalties.len() {
            let gap = (fits[0].evaluation.rho[k] - fits[1].evaluation.rho[k]).abs();
            assert!(
                gap <= radii[0] + radii[1],
                "ρ̂[{k}]: {} vs rotated {}, gap {gap:.3e} exceeds the radii {:.3e}",
                fits[0].evaluation.rho[k],
                fits[1].evaluation.rho[k],
                radii[0] + radii[1]
            );
        }

        // At one ρ: `q(YO) = tr(OᵀMO)` for the residual Gram `M`, so a computed `O` moves it by at most `defect·q`.
        let rho = fits[0].evaluation.rho.clone();
        let a = original.evaluate(rho.view()).expect("original evaluation");
        let b = turned.evaluate(rho.view()).expect("rotated evaluation");
        let score_band = a.reml_score_roundoff + b.reml_score_roundoff + 0.5 * a.dispersion_dof * defect;
        assert!(
            (a.reml_score - b.reml_score).abs() <= score_band,
            "score {} vs rotated {}, band {score_band:.3e}",
            a.reml_score,
            b.reml_score
        );
        let sigma_band = (a.residual_quadratic_roundoff + b.residual_quadratic_roundoff) / a.dispersion_dof
            + defect * a.sigma2;
        assert!(
            (a.sigma2 - b.sigma2).abs() <= sigma_band,
            "σ̂² {} vs rotated {}, band {sigma_band:.3e}",
            a.sigma2,
            b.sigma2
        );
        assert!(
            (a.edf - b.edf).abs() <= a.edf_roundoff + b.edf_roundoff,
            "edf {} vs rotated {}",
            a.edf,
            b.edf
        );
        let (a_coefficients, a_roundoff) = original.coefficients(rho.view()).expect("original coefficients");
        let (b_coefficients, b_roundoff) = turned.coefficients(rho.view()).expect("rotated coefficients");
        let turned_back = fast_ab(&a_coefficients, &rotation);
        let coefficient_band = b_roundoff
            + a_roundoff * (1.0 + defect)
            + accumulation_growth(m) * frobenius(a_coefficients.view()) * (m as f64).sqrt();
        let coefficient_gap = frobenius((&b_coefficients - &turned_back).view());
        assert!(
            coefficient_gap <= coefficient_band,
            "B̂(YO) − B̂(Y)O = {coefficient_gap:.3e}, band {coefficient_band:.3e}"
        );

        // Positive control: stretching one output direction is not a rotation, and the band resolves it.
        let mut stretched = y.clone();
        stretched.column_mut(0).mapv_inplace(|value| 3.0 * value);
        let c = GaussianRemlMultiPenaltyProblem::new(x.view(), stretched.view(), &penalties, 0)
            .expect("the stretched problem")
            .evaluate(rho.view())
            .expect("stretched evaluation");
        assert!(
            (a.reml_score - c.reml_score).abs() > a.reml_score_roundoff + c.reml_score_roundoff,
            "positive control: a stretched output left the score {} inside the band of {}",
            c.reml_score,
            a.reml_score
        );
    }

    #[test]
    fn gaussian_reml_multi_penalty_recovers_planted_smoothing_ratios() {
        let (n, p, m) = (200, 10, 40);
        let x = cosine_design(n, p);
        let penalties = [curvature_penalty(p, 2), Array2::<f64>::eye(p)];
        // Two coordinates and their ratio, Bonferroni at a declared family-wise false-alarm rate.
        let family_wise_false_alarm = 1.0e-3;
        let z = gam_math::probability::standard_normal_quantile(1.0 - family_wise_false_alarm / (2.0 * 3.0))
            .expect("the normal quantile of the Bonferroni level");
        // `β_c ~ N(0, σ²S_λ⁻¹)` with `σ² = 1`; both penalties are diagonal, so `S_λ` is too.
        let planted = |rho: [f64; 2], seed: u64| -> Array2<f64> {
            let mut rng = StdRng::seed_from_u64(seed);
            let precision: Vec<f64> = (0..p)
                .map(|j| rho[0].exp() * penalties[0][[j, j]] + rho[1].exp() * penalties[1][[j, j]])
                .collect();
            let beta = Array2::from_shape_fn((p, m), |index: (usize, usize)| {
                let draw: f64 = StandardNormal.sample(&mut rng);
                draw / precision[index.0].sqrt()
            });
            fast_ab(&x, &beta) + normal_matrix(n, m, &mut rng)
        };
        // `V` is the negative profiled log-likelihood in ρ, so its Hessian at ρ̂ is the observed information.
        let estimate = |y: &Array2<f64>| -> (Array1<f64>, Array2<f64>) {
            let fit = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, 0)
                .expect("the planted problem")
                .fit(None)
                .expect("the planted fit converges");
            let covariance = certified_spd_inverse(
                &fit.evaluation.reml_hessian,
                SymmetricAssembly::Mirrored,
                "planted ρ̂ information",
            )
            .expect("the planted fit's information is SPD")
            .into_inverse();
            (fit.evaluation.rho, covariance)
        };
        let ratio_sd = |covariance: &Array2<f64>| {
            (covariance[[0, 0]] + covariance[[1, 1]] - 2.0 * covariance[[0, 1]]).sqrt()
        };

        let truth = [-2.0, 1.0];
        let (rho_hat, covariance) = estimate(&planted(truth, 29_460_004));
        for k in 0..2 {
            let sd = covariance[[k, k]].sqrt();
            assert!(
                (rho_hat[k] - truth[k]).abs() <= z * sd,
                "ρ̂[{k}] = {} vs planted {}, outside z·sd = {:.3e}",
                rho_hat[k],
                truth[k],
                z * sd
            );
        }
        let ratio_gap = ((rho_hat[0] - rho_hat[1]) - (truth[0] - truth[1])).abs();
        assert!(
            ratio_gap <= z * ratio_sd(&covariance),
            "log smoothing ratio off by {ratio_gap:.3e}, outside z·sd = {:.3e}",
            z * ratio_sd(&covariance)
        );

        // Positive control: data planted at a 50× smaller mass strength covers its own truth and rejects the first.
        let alternative = [-2.0, 1.0 - 50.0_f64.ln()];
        let (alternative_hat, alternative_covariance) = estimate(&planted(alternative, 29_460_005));
        let alternative_sd = alternative_covariance[[1, 1]].sqrt();
        assert!(
            (alternative_hat[1] - alternative[1]).abs() <= z * alternative_sd,
            "alternative ρ̂[1] = {} vs planted {}, outside z·sd = {:.3e}",
            alternative_hat[1],
            alternative[1],
            z * alternative_sd
        );
        assert!(
            (alternative_hat[1] - truth[1]).abs() > z * alternative_sd,
            "positive control: the alternative fit {} cannot reject the first planted ρ[1] = {} at z·sd = {:.3e}",
            alternative_hat[1],
            truth[1],
            z * alternative_sd
        );
    }

    #[test]
    fn gaussian_reml_multi_penalty_data_gradient_matches_central_differences() {
        let (n, p, m) = (60, 6, 4);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_006);
        let noise = [0.3, 0.4, 0.6, 0.9];
        let y = responses(n, &noise, &mut rng);
        let penalties = [curvature_penalty(p, 2), Array2::<f64>::eye(p)];
        let problem = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, 0)
            .expect("the gradient fixture");
        let fit = problem.fit(None).expect("the gradient fixture fit converges");
        assert!(
            fit.rho_placement
                .iter()
                .all(|placement| *placement == GaussianRemlMultiPenaltyRhoPlacement::Interior),
            "the gradient fixture's ρ̂ must be interior for the envelope forms to be total derivatives; {:?}",
            fit.rho_placement
        );
        let gradient = match problem
            .data_gradient(x.view(), y.view(), &fit)
            .expect("data gradient at the fit")
        {
            GaussianRemlMultiPenaltyDataGradientOutcome::Interior(gradient) => gradient,
            refusal => panic!("an interior fit must return the envelope gradient; got {refusal:?}"),
        };
        let rho = fit.evaluation.rho.clone();

        // `V` at fixed ρ̂ along a unit direction; the analytic cotangents are the fixed-ρ partials.
        let score = |design: &Array2<f64>, response: &Array2<f64>| -> (f64, f64) {
            let evaluation = GaussianRemlMultiPenaltyProblem::new(design.view(), response.view(), &penalties, 0)
                .expect("a perturbed problem")
                .evaluate(rho.view())
                .expect("a perturbed evaluation");
            (evaluation.reml_score, evaluation.reml_score_roundoff)
        };
        let unit = |matrix: Array2<f64>| {
            let norm = frobenius(matrix.view());
            matrix.mapv(|value| value / norm)
        };
        let design_direction = unit(normal_matrix(n, p, &mut rng));
        let response_direction = unit(normal_matrix(n, m, &mut rng));
        // Central differences at `h` and `2h`: `D(h) = g + c·h² + O(h⁴)`, so `(D(2h) − D(h))/3` is the remainder at
        // `h`, and each difference adds `(δV₊ + δV₋)/2h` of rounding. `h = ε^{1/3}` balances the two for unit scales.
        let step = f64::EPSILON.cbrt();
        let central = |perturb: &dyn Fn(f64) -> (Array2<f64>, Array2<f64>), h: f64| -> (f64, f64) {
            let (design_plus, response_plus) = perturb(h);
            let (design_minus, response_minus) = perturb(-h);
            let (plus, plus_roundoff) = score(&design_plus, &response_plus);
            let (minus, minus_roundoff) = score(&design_minus, &response_minus);
            ((plus - minus) / (2.0 * h), (plus_roundoff + minus_roundoff) / (2.0 * h))
        };
        let design_perturb =
            |h: f64| (&x + &design_direction.mapv(|value| h * value), y.clone());
        let response_perturb =
            |h: f64| (x.clone(), &y + &response_direction.mapv(|value| h * value));
        let directional = |analytic: &Array2<f64>, direction: &Array2<f64>| {
            analytic.iter().zip(direction.iter()).map(|(a, b)| a * b).sum::<f64>()
        };

        type Perturbation<'a> = &'a dyn Fn(f64) -> (Array2<f64>, Array2<f64>);
        let design_leg: Perturbation<'_> = &design_perturb;
        let response_leg: Perturbation<'_> = &response_perturb;
        let legs: [(&str, Perturbation<'_>, &Array2<f64>, &Array2<f64>); 2] = [
            ("X", design_leg, &gradient.grad_x, &design_direction),
            ("Y", response_leg, &gradient.grad_y, &response_direction),
        ];
        for (label, perturb, analytic, direction) in legs {
            let (at_h, rounding_h) = central(perturb, step);
            let (at_2h, rounding_2h) = central(perturb, 2.0 * step);
            // The remainder estimate `(D(2h) − D(h))/3` carries both differences' rounding over three, and `D(h)`
            // carries its own.
            let band = (at_2h - at_h).abs() / 3.0 + (4.0 * rounding_h + rounding_2h) / 3.0;
            let expected = directional(analytic, direction);
            assert!(
                expected.abs() > band,
                "∂V/∂{label}: the directional derivative {expected:.3e} must exceed the band {band:.3e}, or agreement \
                 is vacuous"
            );
            assert!(
                (at_h - expected).abs() <= band,
                "∂V/∂{label}: central difference {at_h:.9e} vs analytic {expected:.9e}, band {band:.3e}"
            );
            // Positive control: the gradient without its log|K| leg misses the difference by more than the band.
            if label == "X" {
                let inverse_leg = &gradient.grad_x
                    + &(&(&y - &fast_ab(&x, &fit.coefficients)).dot(&fit.coefficients.t())
                        / fit.evaluation.sigma2);
                let without_log_det = &gradient.grad_x - &inverse_leg;
                let wrong = directional(&without_log_det, direction);
                assert!(
                    (at_h - wrong).abs() > band,
                    "positive control: the X gradient without m·XK⁻¹ gives {wrong:.9e}, inside the band {band:.3e} \
                     of {at_h:.9e}"
                );
            }
        }
    }

    /// fr-compile's regime: smooth, noise-free targets the basis represents only approximately, so the residual is
    /// small against ‖Y‖ but resolved. Profiling σ² makes REML invariant to `Y → cY`: `ρ̂` is unchanged and
    /// `V(cY) = V(Y) + ν·ln c` exactly, since `q` scales by `c²` and the log-determinants do not see `Y`.
    #[test]
    fn gaussian_reml_multi_penalty_certifies_smooth_noise_free_targets_scale_invariantly() {
        let (n, p) = (400, 8);
        let x = cosine_design(n, p);
        let y = Array2::from_shape_fn((n, 2), |(row, col)| {
            let position = (row as f64 + 0.5) / n as f64;
            if col == 0 { position.powf(2.5) } else { (1.5 * position).exp() }
        });
        let scale = 1.0e3;
        let scaled = y.mapv(|value| scale * value);
        let penalties = [curvature_penalty(p, 2), Array2::<f64>::eye(p)];
        let original = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, 0)
            .expect("the noise-free problem is resolved");
        let enlarged = GaussianRemlMultiPenaltyProblem::new(x.view(), scaled.view(), &penalties, 0)
            .expect("the scaled noise-free problem is resolved");
        let fits = [
            original.fit(None).expect("the noise-free fit converges"),
            enlarged.fit(None).expect("the scaled noise-free fit converges"),
        ];
        for fit in &fits {
            assert!(fit.certificate.is_some(), "a returned fit carries the optimizer's analytic certificate");
            assert!(
                fit.evaluation.residual_quadratic > fit.evaluation.residual_quadratic_roundoff,
                "the approximate representation leaves a resolved residual {} against {}",
                fit.evaluation.residual_quadratic,
                fit.evaluation.residual_quadratic_roundoff
            );
        }
        let radii: Vec<f64> = fits.iter().map(fitted_radius).collect();
        for k in 0..penalties.len() {
            let gap = (fits[0].evaluation.rho[k] - fits[1].evaluation.rho[k]).abs();
            assert!(
                gap <= radii[0] + radii[1],
                "ρ̂[{k}]: {} vs scaled {}, gap {gap:.3e} exceeds the radii {:.3e}",
                fits[0].evaluation.rho[k],
                fits[1].evaluation.rho[k],
                radii[0] + radii[1]
            );
        }

        let rho = fits[0].evaluation.rho.clone();
        let a = original.evaluate(rho.view()).expect("original evaluation");
        let b = enlarged.evaluate(rho.view()).expect("scaled evaluation");
        let shift = a.dispersion_dof * scale.ln();
        let shift_band = a.reml_score_roundoff + b.reml_score_roundoff + accumulation_growth(4) * shift.abs();
        assert!(
            (b.reml_score - a.reml_score - shift).abs() <= shift_band,
            "V(cY) − V(Y) = {:.9e} against ν·ln c = {shift:.9e}, band {shift_band:.3e}",
            b.reml_score - a.reml_score
        );
        assert!(
            (a.edf - b.edf).abs() <= a.edf_roundoff + b.edf_roundoff,
            "edf {} vs scaled {}",
            a.edf,
            b.edf
        );
        // Positive control: scaling one column only is not a scale change of the response, and the band resolves it.
        let mut one_column = y.clone();
        one_column.column_mut(0).mapv_inplace(|value| scale * value);
        let c = GaussianRemlMultiPenaltyProblem::new(x.view(), one_column.view(), &penalties, 0)
            .expect("the one-column-scaled problem")
            .evaluate(rho.view())
            .expect("one-column-scaled evaluation");
        let control_band = a.reml_score_roundoff + c.reml_score_roundoff + accumulation_growth(4) * shift.abs();
        assert!(
            (c.reml_score - a.reml_score - shift).abs() > control_band,
            "positive control: scaling one column shifted the score by {:.9e}, inside the band {control_band:.3e} of \
             ν·ln c = {shift:.9e}",
            c.reml_score - a.reml_score
        );
    }

    /// A response with exactly zero penalized components: its column-space part lies in the curvature penalty's null
    /// space and the rest is orthogonal to every column. Then `∂V/∂ρ = −½m·Σ γ/(γ + λδ) < 0` at every ρ, so the
    /// criterion is strictly monotone and its minimizer over the domain is the upper edge. Returns `(X, Y, problem)`.
    fn monotone_asymptote_fixture() -> (Array2<f64>, Array2<f64>, GaussianRemlMultiPenaltyProblem) {
        let (n, p, m) = (120, 8, 2);
        let x = cosine_design(n, p);
        let penalty = curvature_penalty(p, 2);
        let mut rng = StdRng::seed_from_u64(29_460_007);
        let draws = normal_matrix(n, m, &mut rng).mapv(|value| 0.1 * value);
        let gram_inverse = certified_spd_inverse(
            &fast_ata(&x),
            SymmetricAssembly::Mirrored,
            "test design Gram",
        )
        .expect("the cosine design has full column rank")
        .into_inverse();
        let orthogonal = &draws - &fast_ab(&x, &fast_ab(&gram_inverse, &fast_atb(&x, &draws)));
        let null_coefficients = Array2::from_shape_fn((p, m), |(row, col)| {
            if row < 2 {
                1.0 + row as f64 + col as f64
            } else {
                0.0
            }
        });
        let y = fast_ab(&x, &null_coefficients) + &orthogonal;
        let problem = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &[penalty], 2)
            .expect("the null-space fixture");
        (x, y, problem)
    }

    /// What a cold start does on the monotone criterion, pinned so that a certificate change which rails cold starts
    /// shows up as a test change: the optimizer certifies an interior ρ̂ on the decaying asymptote once `|g|` is under
    /// its band, below the edge, with the criterion still descending (sw2e 1181026: ρ̂ = 14.01 on [−21.71, 19.34]).
    /// There the envelope forms are returned; their error `|∂V/∂ρ|·|∂ρ̂/∂X|` is `O(|g|)` by derivation, unmeasured.
    #[test]
    fn gaussian_reml_multi_penalty_cold_start_certifies_on_the_monotone_asymptote() {
        let (x, y, problem) = monotone_asymptote_fixture();
        let fit = problem.fit(None).expect("the cold start converges");
        let certificate = fit
            .certificate
            .as_ref()
            .expect("a returned fit carries the optimizer's certificate");
        assert!(certificate.is_stationary(), "the cold start is certified stationary: {certificate:?}");
        assert!(
            fit.evaluation.reml_gradient[0] < 0.0,
            "the criterion still descends at the certified point; g={:.3e}",
            fit.evaluation.reml_gradient[0]
        );
        assert_eq!(
            fit.rho_placement,
            vec![GaussianRemlMultiPenaltyRhoPlacement::Interior],
            "ρ̂={} on [{}, {}]",
            fit.evaluation.rho[0],
            fit.rho_lower[0],
            fit.rho_upper[0]
        );
        assert!(
            fit.evaluation.rho[0] < fit.rho_upper[0],
            "a cold start stops below the edge; ρ̂={} upper={}",
            fit.evaluation.rho[0],
            fit.rho_upper[0]
        );
        match problem
            .data_gradient(x.view(), y.view(), &fit)
            .expect("the data gradient answers at the certified point")
        {
            GaussianRemlMultiPenaltyDataGradientOutcome::Interior(..) => {}
            refusal => panic!("an interior placement must return the envelope forms; got {refusal:?}"),
        }
    }

    /// The same monotone criterion warm-started at its upper edge, where the projected gradient is zero and ρ̂ stays
    /// railed. From a cold start it certifies on the asymptote instead (sw2e 1181026; pinned above). The edge is
    /// derived from `XᵀX` and moves with `X`, so the data gradient refuses there, typed.
    #[test]
    fn gaussian_reml_multi_penalty_data_gradient_refuses_at_a_domain_bound() {
        let (x, y, problem) = monotone_asymptote_fixture();
        let fit = problem
            .fit(Some(problem.rho_upper.view()))
            .expect("the null-space fixture converges at its domain edge");
        assert!(
            fit.evaluation.reml_gradient[0] < 0.0,
            "the criterion still descends into the upper edge; g={:.3e}",
            fit.evaluation.reml_gradient[0]
        );
        assert_eq!(
            fit.rho_placement,
            vec![GaussianRemlMultiPenaltyRhoPlacement::UpperBound],
            "ρ̂={} on [{}, {}]",
            fit.evaluation.rho[0],
            fit.rho_lower[0],
            fit.rho_upper[0]
        );
        match problem
            .data_gradient(x.view(), y.view(), &fit)
            .expect("the data gradient answers at the edge")
        {
            GaussianRemlMultiPenaltyDataGradientOutcome::RhoAtDomainBound { placement } => {
                assert_eq!(placement, fit.rho_placement, "the refusal names the railed coordinate");
            }
            GaussianRemlMultiPenaltyDataGradientOutcome::Interior(..) => {
                panic!("a railed ρ̂ must not return the envelope forms as a total derivative")
            }
        }
    }

    /// The data gradient is the envelope derivative at THIS data's converged fit, so arrays whose values differ from
    /// the ones the problem was reduced from are refused, naming the argument. One flipped bit is enough.
    #[test]
    fn gaussian_reml_multi_penalty_data_gradient_refuses_arrays_it_was_not_built_from() {
        let (n, p) = (60, 6);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_006);
        let y = responses(n, &[0.3, 0.4, 0.6, 0.9], &mut rng);
        let penalties = [curvature_penalty(p, 2), Array2::<f64>::eye(p)];
        let problem = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, 0)
            .expect("the fingerprint fixture");
        let fit = problem.fit(None).expect("the fingerprint fixture converges");
        let accepted = problem.data_gradient(x.view(), y.view(), &fit);
        assert!(
            matches!(accepted, Ok(GaussianRemlMultiPenaltyDataGradientOutcome::Interior(..))),
            "the arrays the problem was built from are accepted; got {accepted:?} with placement {:?}",
            fit.rho_placement
        );
        let mut flipped_x = x.clone();
        flipped_x[[3, 2]] = f64::from_bits(flipped_x[[3, 2]].to_bits() ^ 1);
        match problem.data_gradient(flipped_x.view(), y.view(), &fit) {
            Err(EstimationError::InvalidInput(message)) => {
                assert!(message.contains("`x`"), "the refusal names x: {message}");
            }
            other => panic!("one flipped bit in x must be refused; got {other:?}"),
        }
        let mut flipped_y = y.clone();
        flipped_y[[5, 1]] = f64::from_bits(flipped_y[[5, 1]].to_bits() ^ 1);
        match problem.data_gradient(x.view(), flipped_y.view(), &fit) {
            Err(EstimationError::InvalidInput(message)) => {
                assert!(message.contains("`y`"), "the refusal names y: {message}");
            }
            other => panic!("one flipped bit in y must be refused; got {other:?}"),
        }
    }

    /// `∂V/∂S_k = ½λ_k(B̂B̂ᵀ/σ̂² + m·K⁻¹ − m·S_λ⁺)` against central differences at fixed ρ̂, along a symmetric perturbation
    /// of each penalty inside its own range, which preserves `rank(S_λ)`. Both penalties annihilate the same two modes,
    /// so `S_λ⁺` is a pseudo-inverse. Mutant: dropping the `−m·S_λ⁺` leg must fall outside the band.
    #[test]
    fn gaussian_reml_multi_penalty_penalty_gradient_matches_central_differences() {
        let (n, p, columns) = (200, 7, 20);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_009);
        let m = columns as f64;
        let slope = Array2::from_shape_fn((p, p), |(row, col)| {
            if row == col && row >= 2 { (row as f64).powi(2) } else { 0.0 }
        });
        let penalties = [curvature_penalty(p, 2), slope];
        // Both strengths are planted material: at ρ = (−3, 0) the slope penalty dominates the low range modes and the
        // curvature penalty the high ones, so each S_k carries a resolvable cotangent. On unplanted smooth responses the
        // slope strength went small and ∂V/∂S_1 was −3.7e-6 against a rounding band of 3.7e-5 (sw2e 1196148, 1197522).
        let planted_precision: Vec<f64> = (0..p)
            .map(|j| (-3.0_f64).exp() * penalties[0][[j, j]] + penalties[1][[j, j]])
            .collect();
        let beta = Array2::from_shape_fn((p, columns), |index: (usize, usize)| {
            let draw: f64 = StandardNormal.sample(&mut rng);
            if index.0 < 2 { draw } else { draw / planted_precision[index.0].sqrt() }
        });
        let y = fast_ab(&x, &beta) + normal_matrix(n, columns, &mut rng);
        let problem = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, 2)
            .expect("the penalty-gradient fixture");
        let fit = problem.fit(None).expect("the penalty-gradient fixture converges");
        let (gradients, rank, smallest_resolved, rank_band) = match problem
            .penalty_gradient(&fit)
            .expect("the penalty gradient at the fit")
        {
            GaussianRemlMultiPenaltyPenaltyGradientOutcome::Interior {
                gradients,
                rank,
                smallest_resolved_singular_value,
                rank_band,
            } => (gradients, rank, smallest_resolved_singular_value, rank_band),
            refusal => panic!("an interior fit must return the penalty gradient; got {refusal:?}"),
        };
        assert_eq!(rank, p - 2, "rank(S_λ) is the complement of the two shared null modes");
        assert!(
            smallest_resolved > rank_band,
            "the reported rank margin {smallest_resolved:.3e} clears its band {rank_band:.3e}"
        );
        let rho = fit.evaluation.rho.clone();
        let lambdas = fit.evaluation.lambdas.clone();
        let score = |perturbed: &[Array2<f64>]| -> (f64, f64) {
            let evaluation = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), perturbed, 2)
                .expect("a perturbed penalty problem")
                .evaluate(rho.view())
                .expect("a perturbed penalty evaluation");
            (evaluation.reml_score, evaluation.reml_score_roundoff)
        };
        // Both penalties are diagonal, so `S_λ` is too and its pseudo-inverse inverts the range modes: an independent
        // construction for the mutant.
        let pseudo_inverse = Array2::from_shape_fn((p, p), |(row, col)| {
            if row == col && row >= 2 {
                1.0 / (lambdas[0] * penalties[0][[row, row]] + lambdas[1] * penalties[1][[row, row]])
            } else {
                0.0
            }
        });
        // Central differences at `h` and `2h`, with the Richardson remainder plus both differences' rounding (see the
        // data gradient test).
        let step = f64::EPSILON.cbrt();
        for k in 0..penalties.len() {
            let block = normal_matrix(p - 2, p - 2, &mut rng);
            let direction = Array2::from_shape_fn((p, p), |(row, col)| {
                if row >= 2 && col >= 2 {
                    // A RELATIVE perturbation `Λ^½·D·Λ^½` on the range block: an absolute `O(1)` direction against
                    // entries up to `6⁴` moved V by less than the difference's own rounding (sw2e 1196148: the mutant
                    // gap 1.1e-5 sat inside the band 3.7e-5).
                    0.5 * (block[[row - 2, col - 2]] + block[[col - 2, row - 2]])
                        * (penalties[k][[row, row]] * penalties[k][[col, col]]).sqrt()
                } else {
                    0.0
                }
            });
            let central = |h: f64| -> (f64, f64) {
                let mut plus = penalties.clone();
                plus[k] = &plus[k] + &direction.mapv(|value| h * value);
                let mut minus = penalties.clone();
                minus[k] = &minus[k] - &direction.mapv(|value| h * value);
                let (above, above_roundoff) = score(&plus);
                let (below, below_roundoff) = score(&minus);
                ((above - below) / (2.0 * h), (above_roundoff + below_roundoff) / (2.0 * h))
            };
            let (at_h, rounding_h) = central(step);
            let (at_2h, rounding_2h) = central(2.0 * step);
            let band = (at_2h - at_h).abs() / 3.0 + (4.0 * rounding_h + rounding_2h) / 3.0;
            let expected: f64 = gradients[k].iter().zip(direction.iter()).map(|(g, d)| g * d).sum();
            assert!(
                expected.abs() > band,
                "∂V/∂S_{k}: the directional derivative {expected:.3e} must exceed the band {band:.3e}, or agreement is \
                 vacuous"
            );
            assert!(
                (at_h - expected).abs() <= band,
                "∂V/∂S_{k}: central difference {at_h:.9e} vs analytic {expected:.9e}, band {band:.3e}"
            );
            let mutant = &gradients[k] + &pseudo_inverse.mapv(|value| 0.5 * lambdas[k] * m * value);
            let wrong: f64 = mutant.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();
            assert!(
                (at_h - wrong).abs() > band,
                "mutant: the gradient without −m·S_λ⁺ gives {wrong:.9e}, inside the band {band:.3e} of {at_h:.9e}"
            );
        }
    }

    /// Weights in `[0.5, 1.5]`, with every seventh row omitted.
    fn observation_weights(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |row| {
            if row % 7 == 3 { 0.0 } else { 1.0 + 0.5 * (2.3 * row as f64).cos() }
        })
    }

    /// `√w_i·M` over the positive-weight rows, whitened the way a caller would.
    fn whiten_positive_rows(matrix: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
        let kept: Vec<usize> = (0..weights.len()).filter(|&row| weights[row] > 0.0).collect();
        Array2::from_shape_fn((kept.len(), matrix.ncols()), |(row, col)| {
            matrix[[kept[row], col]] * weights[kept[row]].sqrt()
        })
    }

    #[test]
    fn gaussian_reml_multi_penalty_weighted_one_penalty_reproduces_the_shared_dispersion_closed_form() {
        let (n, p) = (120, 8);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_014);
        let y = responses(n, &[0.3, 0.5, 0.8], &mut rng);
        let penalty = curvature_penalty(p, 2);
        let weights = observation_weights(n);

        let old = gaussian_reml_multi_shared_dispersion_closed_form(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            None,
        )
        .expect("the shared-dispersion closed form fits the weighted fixture");
        let old_roundoff = old
            .reml_score_roundoff
            .expect("the shared-dispersion closed form publishes its score roundoff");
        let problem = GaussianRemlMultiPenaltyProblem::new_weighted(
            x.view(),
            y.view(),
            weights.view(),
            &[penalty.clone()],
            2,
        )
        .expect("the weighted multi-penalty problem accepts the fixture");
        let at_old = problem
            .evaluate(array![old.rho].view())
            .expect("the weighted evaluator evaluates at the closed-form optimum");

        // Score at the same ρ: two evaluations of one criterion differ by at most both rounding bounds.
        let score_band = at_old.reml_score_roundoff + old_roundoff;
        let score_gap = (at_old.reml_score - old.reml_score).abs();
        assert!(
            score_gap <= score_band,
            "weighted score at the closed-form ρ̂: multi-penalty {} vs closed form {}, gap {score_gap:.3e} exceeds \
             band {score_band:.3e}",
            at_old.reml_score,
            old.reml_score
        );
        // Positive control: a ρ shift whose quadratic score change is nine bands is resolved by that same band.
        let curvature = at_old.reml_hessian[[0, 0]];
        assert!(curvature > 0.0, "the criterion must curve upward at its optimum; h={curvature:.3e}");
        let shift = 3.0 * (2.0 * score_band / curvature).sqrt();
        let shifted = problem
            .evaluate(array![old.rho + shift].view())
            .expect("the evaluator evaluates beside the optimum");
        assert!(
            (shifted.reml_score - old.reml_score).abs() > score_band,
            "positive control: a shift of {shift:.3e} in ρ changed the score by {:.3e}, inside the band {score_band:.3e}",
            (shifted.reml_score - old.reml_score).abs()
        );
        // The weights are read: the same data unweighted is resolved apart at the same ρ.
        let unweighted = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &[penalty.clone()], 2)
            .expect("the unweighted problem accepts the fixture")
            .evaluate(array![old.rho].view())
            .expect("the unweighted evaluator evaluates at the closed-form optimum");
        assert!(
            (unweighted.reml_score - old.reml_score).abs() > score_band + unweighted.reml_score_roundoff,
            "control: the unweighted score {} is inside the band of the weighted closed form {}",
            unweighted.reml_score,
            old.reml_score
        );

        // edf and coefficients at the same ρ, against the closed form's first-order bounds on the whitened rows (the
        // one-column test derives them): the design QR's `γ_{np}`, the penalty eigensolver's `p·ε·‖S‖₂` and the
        // whitened-root SVD's `2p·ε·δ_max` mapped back through `‖X̃ᵀX̃‖₂`.
        let white_x = whiten_positive_rows(&x, &weights);
        let white_y = whiten_positive_rows(&y, &weights);
        let lambda = old.rho.exp();
        let gram = fast_ata(&white_x);
        let normal = &gram + &penalty.mapv(|value| lambda * value);
        let inverse = certified_spd_inverse(
            &normal,
            SymmetricAssembly::Mirrored,
            "test weighted penalized normal matrix",
        )
        .expect("the fixture's weighted penalized normal matrix is SPD");
        let inverse_trace = inverse.inverse().diag().sum();
        let design_energy = frobenius(white_x.view()).powi(2);
        let gram_norm = gram.iter().fold(0.0_f64, |acc, value| acc.max(value.abs())) * p as f64;
        let delta_max = old
            .cache
            .penalty_eigenvalues
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(*value));
        let penalty_norm = (p as f64 - 1.0).powi(4);
        let old_penalty_rounding = p as f64 * f64::EPSILON * penalty_norm
            + 2.0 * p as f64 * f64::EPSILON * delta_max * gram_norm;
        let whitening = accumulation_growth(n * p);
        let old_edf_band = inverse_trace * (2.0 * whitening * design_energy + lambda * old_penalty_rounding)
            + accumulation_growth(2 * p) * p as f64;
        let edf_band = at_old.edf_roundoff + old_edf_band;
        assert!(
            (at_old.edf - old.edf).abs() <= edf_band,
            "weighted edf at the closed-form ρ̂: multi-penalty {} vs closed form {}, band {edf_band:.3e}",
            at_old.edf,
            old.edf
        );
        let (at_old_coefficients, at_old_coefficients_roundoff) = problem
            .coefficients(array![old.rho].view())
            .expect("coefficients evaluate at the closed-form optimum");
        let old_beta = frobenius(old.coefficients.view());
        let old_coefficient_band = inverse_trace.sqrt()
            * whitening
            * (frobenius(white_y.view()) + design_energy.sqrt() * old_beta)
            + inverse_trace
                * (whitening * design_energy.sqrt() * at_old.residual_quadratic.sqrt()
                    + lambda * old_penalty_rounding * old_beta);
        let coefficient_gap = frobenius((&at_old_coefficients - &old.coefficients).view());
        let coefficient_band = at_old_coefficients_roundoff + old_coefficient_band;
        assert!(
            coefficient_gap <= coefficient_band,
            "weighted coefficients at the closed-form ρ̂ differ by {coefficient_gap:.3e}, band {coefficient_band:.3e}"
        );

        // λ̂: both optima lie within their resolution radii of the one minimizer.
        let fit = problem.fit(None).expect("the weighted multi-penalty fit converges");
        let new_curvature = fit.evaluation.reml_hessian[[0, 0]];
        assert!(new_curvature > 0.0, "the fitted criterion must curve upward; h={new_curvature:.3e}");
        let new_radius = resolution_radius(
            fit.evaluation.reml_gradient[0].abs(),
            new_curvature,
            fit.evaluation.reml_score_roundoff,
        );
        let old_radius = resolution_radius(at_old.reml_gradient[0].abs(), curvature, score_band);
        let rho_gap = (fit.evaluation.rho[0] - old.rho).abs();
        assert!(
            rho_gap <= new_radius + old_radius,
            "weighted ρ̂: multi-penalty {} vs closed form {}, gap {rho_gap:.3e} exceeds the radii {new_radius:.3e} + \
             {old_radius:.3e}",
            fit.evaluation.rho[0],
            old.rho
        );
    }

    #[test]
    fn gaussian_reml_multi_penalty_weighted_score_is_the_whitened_score_plus_the_jacobian() {
        let (n, p) = (60, 6);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_011);
        let y = responses(n, &[0.2, 0.4], &mut rng);
        let m = y.ncols();
        let penalty = curvature_penalty(p, 2);
        let weights = observation_weights(n);
        let kept = weights.iter().filter(|&&weight| weight > 0.0).count();
        assert!(kept < n, "the fixture must omit some rows");

        let weighted = GaussianRemlMultiPenaltyProblem::new_weighted(
            x.view(),
            y.view(),
            weights.view(),
            &[penalty.clone()],
            2,
        )
        .expect("the weighted problem accepts the fixture");
        let white = GaussianRemlMultiPenaltyProblem::new(
            whiten_positive_rows(&x, &weights).view(),
            whiten_positive_rows(&y, &weights).view(),
            &[penalty.clone()],
            2,
        )
        .expect("the whitened problem accepts the positive-weight rows");
        let unweighted = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &[penalty.clone()], 2)
            .expect("the unweighted problem accepts the fixture");
        // `−(m/2)·Σ log w_i` over the positive-weight rows, with the evaluator's own bound on its formation.
        let mut log_weights = KahanSum::default();
        let mut magnitude = 0.0;
        for &weight in weights.iter().filter(|&&weight| weight > 0.0) {
            log_weights.add(weight.ln());
            magnitude += weight.ln().abs();
        }
        let jacobian = -0.5 * m as f64 * log_weights.sum();
        let jacobian_roundoff = 0.5 * m as f64 * compensated_band(3, magnitude);

        for rho in [-2.0, 1.5, 4.0] {
            let at = array![rho];
            let observed = weighted.evaluate(at.view()).expect("the weighted evaluator evaluates");
            let whitened = white.evaluate(at.view()).expect("the whitened evaluator evaluates");
            assert_eq!(
                observed.dispersion_dof,
                (m * (kept - 2)) as f64,
                "ν counts the positive-weight rows only"
            );
            let band = observed.reml_score_roundoff + whitened.reml_score_roundoff + jacobian_roundoff;
            let gap = (observed.reml_score - (whitened.reml_score + jacobian)).abs();
            assert!(
                gap <= band,
                "ρ={rho}: weighted score {} vs whitened {} + Jacobian {jacobian}, gap {gap:.3e} exceeds band {band:.3e}",
                observed.reml_score,
                whitened.reml_score
            );
            // Control: the same rows read without their weights are resolved apart by that band.
            let plain = unweighted.evaluate(at.view()).expect("the unweighted evaluator evaluates");
            assert!(
                (plain.reml_score - observed.reml_score).abs() > band + plain.reml_score_roundoff,
                "control at ρ={rho}: the unweighted score {} is inside the band of the weighted {}",
                plain.reml_score,
                observed.reml_score
            );
        }

        // The measure is constant in ρ, so the optimum does not move.
        let observed_fit = weighted.fit(None).expect("the weighted fit converges");
        let whitened_fit = white.fit(None).expect("the whitened fit converges");
        let rho_gap = (observed_fit.evaluation.rho[0] - whitened_fit.evaluation.rho[0]).abs();
        let radii = fitted_radius(&observed_fit) + fitted_radius(&whitened_fit);
        assert!(
            rho_gap <= radii,
            "weighted ρ̂ {} vs whitened ρ̂ {}, gap {rho_gap:.3e} exceeds the radii {radii:.3e}",
            observed_fit.evaluation.rho[0],
            whitened_fit.evaluation.rho[0]
        );
    }

    #[test]
    fn gaussian_reml_multi_penalty_weighted_data_gradient_is_the_whitened_one_scaled_back() {
        let (n, p) = (60, 6);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_012);
        let y = responses(n, &[0.2, 0.4], &mut rng);
        let penalty = curvature_penalty(p, 2);
        let weights = observation_weights(n);
        let (white_x, white_y) = (whiten_positive_rows(&x, &weights), whiten_positive_rows(&y, &weights));

        let weighted = GaussianRemlMultiPenaltyProblem::new_weighted(
            x.view(),
            y.view(),
            weights.view(),
            &[penalty.clone()],
            2,
        )
        .expect("the weighted problem accepts the fixture");
        let white = GaussianRemlMultiPenaltyProblem::new(white_x.view(), white_y.view(), &[penalty.clone()], 2)
            .expect("the whitened problem accepts the positive-weight rows");
        let fit = weighted.fit(None).expect("the weighted fit converges");
        assert!(
            fit.rho_placement.iter().all(|placement| *placement == GaussianRemlMultiPenaltyRhoPlacement::Interior),
            "the fixture's λ̂ must be interior; placement {:?}",
            fit.rho_placement
        );
        let observed = match weighted.data_gradient(x.view(), y.view(), &fit).expect("weighted data gradient") {
            GaussianRemlMultiPenaltyDataGradientOutcome::Interior(gradient) => gradient,
            other => panic!("an interior weighted fit must return the envelope forms, got {other:?}"),
        };
        // The whitened criterion is the weighted one less a constant, so the weighted fit is its fit as well.
        let whitened = match white
            .data_gradient(white_x.view(), white_y.view(), &fit)
            .expect("whitened data gradient")
        {
            GaussianRemlMultiPenaltyDataGradientOutcome::Interior(gradient) => gradient,
            other => panic!("an interior whitened fit must return the envelope forms, got {other:?}"),
        };
        assert_eq!(observed.grad_x.dim(), (n, p));
        assert_eq!(observed.grad_y.dim(), (n, y.ncols()));

        // `∂V/∂x_i = √w_i·∂V/∂x̃_i`: one multiplication, so one rounding of the product.
        let kept: Vec<usize> = (0..n).filter(|&row| weights[row] > 0.0).collect();
        let mut mutant_gap = 0.0_f64;
        let mut mutant_band = 0.0_f64;
        for (index, &row) in kept.iter().enumerate() {
            let root = weights[row].sqrt();
            let pairs = observed
                .grad_x
                .row(row)
                .iter()
                .zip(whitened.grad_x.row(index).iter())
                .chain(observed.grad_y.row(row).iter().zip(whitened.grad_y.row(index).iter()))
                .map(|(&left, &right)| (left, right))
                .collect::<Vec<_>>();
            for (scaled, unscaled) in pairs {
                let expected = root * unscaled;
                let band = UNIT_ROUNDOFF * expected.abs();
                assert!(
                    (scaled - expected).abs() <= band,
                    "row {row}: weighted cotangent {scaled:.15e} vs √w·whitened {expected:.15e}, band {band:.3e}"
                );
                if (scaled - unscaled).abs() > mutant_gap {
                    mutant_gap = (scaled - unscaled).abs();
                    mutant_band = band;
                }
            }
        }
        // Mutant: the whitened cotangent without its `√w_i` is resolved apart.
        assert!(
            mutant_gap > mutant_band,
            "mutant: dropping √w_i moves no cotangent past its band ({mutant_gap:.3e} vs {mutant_band:.3e})"
        );
        for row in (0..n).filter(|&row| weights[row] == 0.0) {
            assert!(
                observed.grad_x.row(row).iter().chain(observed.grad_y.row(row).iter()).all(|value| *value == 0.0),
                "row {row} carries zero weight, so it has no cotangent"
            );
        }
    }

    #[test]
    fn gaussian_reml_multi_penalty_structural_nullity_is_the_predicate_construction_checks() {
        let (n, p) = (40, 7);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_013);
        let y = responses(n, &[0.3], &mut rng);
        for (penalties, expected) in [
            (vec![curvature_penalty(p, 3)], 3),
            (vec![curvature_penalty(p, 2), curvature_penalty(p, 1)], 1),
            (vec![curvature_penalty(p, 2), Array2::<f64>::eye(p)], 0),
        ] {
            let nullity = GaussianRemlMultiPenaltyProblem::structural_nullity(&penalties)
                .expect("the structural nullity of well-formed penalties");
            assert_eq!(nullity, expected, "dim ∩ ker S_k of {} penalties", penalties.len());
            GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, nullity)
                .expect("construction accepts the structural nullity");
            for wrong in [nullity + 1, nullity.wrapping_sub(1)].into_iter().filter(|&wrong| wrong <= p) {
                let refusal = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &penalties, wrong)
                    .expect_err("construction refuses a declaration its predicate contradicts")
                    .to_string();
                assert!(
                    refusal.contains(&format!("declared null space of dimension {wrong}")),
                    "the refusal must name the contradicted declaration {wrong}: {refusal}"
                );
            }
        }
        assert!(
            GaussianRemlMultiPenaltyProblem::structural_nullity(&[]).is_err(),
            "no penalties have no structural nullity"
        );
    }

    #[test]
    fn gaussian_reml_multi_penalty_weighted_refuses_weights_it_cannot_read() {
        let (n, p) = (30, 5);
        let x = cosine_design(n, p);
        let mut rng = StdRng::seed_from_u64(29_460_015);
        let y = responses(n, &[0.3], &mut rng);
        let penalties = [curvature_penalty(p, 2)];
        let build = |weights: &Array1<f64>| {
            GaussianRemlMultiPenaltyProblem::new_weighted(x.view(), y.view(), weights.view(), &penalties, 2)
        };
        build(&observation_weights(n)).expect("control: well-formed weights are accepted");
        let cases: [(&str, Array1<f64>, &str); 5] = [
            ("a negative weight", Array1::from_shape_fn(n, |row| if row == 4 { -0.5 } else { 1.0 }), "non-negative"),
            ("a NaN weight", Array1::from_shape_fn(n, |row| if row == 4 { f64::NAN } else { 1.0 }), "finite"),
            ("one weight short", Array1::ones(n - 1), "weights length mismatch"),
            ("every weight zero", Array1::zeros(n), "at least one positive-weight row"),
            (
                "fewer positive-weight rows than the null space needs",
                Array1::from_shape_fn(n, |row| if row < 2 { 1.0 } else { 0.0 }),
                "positive-weight) rows than the null-space",
            ),
        ];
        for (case, weights, expected) in cases {
            let refusal = build(&weights).expect_err(case).to_string();
            assert!(refusal.contains(expected), "{case}: the refusal must name it ({expected}); got: {refusal}");
        }
    }
}

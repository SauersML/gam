use super::*;

/// Default inner P-IRLS tolerance floor.
///
/// The inner Newton iteration certifies the coefficient mode against this
/// (scale-aware) tolerance independently of the outer REML tolerance. Coupling
/// the two collapses two unrelated convergence concepts: when a user dials the
/// outer tolerance up to e.g. 1e-3 to make the smoothing-parameter search
/// coarser, the inner solve becomes coarse too, returning betas whose
/// stationarity residual is ~1e-3·scale rather than the floating-point noise
/// floor. Outer derivatives then read those imprecise betas as if they were
/// the true mode and accumulate error. Keeping the inner floor at 1e-6 lets
/// the outer loop relax without contaminating the coefficient certificate.
pub(crate) const PIRLS_INNER_TOLERANCE_FLOOR: f64 = 1e-6;

#[derive(Clone)]
pub(crate) struct RemlConfig {
    pub(crate) likelihood: GlmLikelihoodSpec,
    pub(crate) link_kind: InverseLink,
    pub(crate) pirls_convergence_tolerance: f64,
    pub(crate) max_iterations: usize,
    pub(crate) reml_convergence_tolerance: f64,
    pub(crate) firth_bias_reduction: bool,
    /// Forwarded to `pirls::PirlsConfig::geodesic_acceleration`. Off by default.
    pub(crate) geodesic_acceleration: bool,
}

impl RemlConfig {
    pub(crate) fn external(
        likelihood: GlmLikelihoodSpec,
        reml_tol: f64,
        firth_bias_reduction: bool,
    ) -> Self {
        // Inner P-IRLS certifies the coefficient mode against
        // `pirls_convergence_tolerance`; the outer REML iteration certifies
        // the smoothing-parameter optimum against `reml_convergence_tolerance`.
        // These are different concepts and must not be coupled. The inner
        // tolerance is at most the outer tolerance (so a user who *tightens*
        // the outer also tightens the inner), but never coarser than the
        // floor — a coarse outer must not silently pollute the inner mode.
        let pirls_tol = reml_tol.min(PIRLS_INNER_TOLERANCE_FLOOR);
        let link_kind = likelihood.spec.link.clone();
        Self {
            likelihood,
            link_kind,
            pirls_convergence_tolerance: pirls_tol,
            max_iterations: 0,
            reml_convergence_tolerance: reml_tol,
            firth_bias_reduction,
            geodesic_acceleration: false,
        }
        .with_max_iterations(300)
    }

    pub(crate) fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub(crate) fn link_function(&self) -> LinkFunction {
        self.link_kind.link_function()
    }

    pub(crate) fn as_pirls_config(&self) -> pirls::PirlsConfig {
        pirls::PirlsConfig {
            likelihood: self.likelihood.clone(),
            link_kind: self.link_kind.clone(),
            max_iterations: self.max_iterations,
            convergence_tolerance: self.pirls_convergence_tolerance,
            firth_bias_reduction: self.firth_bias_reduction,
            // Caller (the REML runtime) populates this hint just before
            // each `execute_pirls_if_needed` call from the cached final
            // λ of the previous successful PIRLS solve.
            initial_lm_lambda: None,
            geodesic_acceleration: self.geodesic_acceleration,
            // Arrow-Schur structured-inner-solve descriptor. Not used by
            // the standard REML→PIRLS path (β-only); set by the latent
            // driver (`crate::solver::latent_inner::LatentInnerSolver`)
            // which assembles the per-row (t, β) bordered system
            // externally. Default `None` preserves back-compat.
            arrow_schur: None,
        }
    }
}
pub(crate) const MAX_FACTORIZATION_ATTEMPTS: usize = 4;

/// Small ridge added to the rho-space LAML Hessian before inversion, for
/// numerical stability when smoothing parameters are weakly identified.
///
/// **Stabilization semantics:** this ridge is a
/// [`crate::types::StabilizationKind::NumericalPerturbation`] (not an
/// `ExplicitPrior`). It enters only the inverse used to build `V_rho` for
/// the smoothing-correction propagation step. It does NOT enter the LAML
/// objective, its gradient, the saved coefficients, or any user-visible
/// summary — the rho-Hessian itself is recomputed from first principles
/// in every place that consults it. Classified as
/// [`crate::types::StabilizationKind::NumericalPerturbation`]; no ledger
/// record is emitted at this site because the perturbation never escapes the
/// local `V_rho` inverse (it touches no saved coefficient, objective, or
/// user-visible summary).
const LAML_RIDGE: f64 = 1e-8;
/// Minimum penalized-deviance floor, expressed as a fraction of the
/// problem's own deviance scale (the weighted null deviance `D₀`, see
/// [`smooth_floor_dp`]). The floor exists only to keep the profiled
/// dispersion `φ̂ = D_p/(n−M_p)` strictly positive when a smooth fits the
/// data essentially perfectly (`D_p ↓ 0`), so it must trigger on the
/// *relative* smallness `D_p/D₀`, never on an absolute magnitude — an
/// absolute floor silently breaks the exact scale-equivariance of the
/// Gaussian REML fit under a response rescale `y → a·y` (#1127).
pub(crate) const DP_FLOOR: f64 = 1e-12;
/// Width of the smooth transition region for the deviance floor, also as a
/// fraction of the deviance scale `D₀`.
const DP_FLOOR_SMOOTH_WIDTH: f64 = 1e-8;

// Unified rho bound corresponding to lambda in [exp(-RHO_BOUND), exp(RHO_BOUND)].
// Additional headroom reduces frequent contact with the hard box constraints.
pub(crate) const RHO_BOUND: f64 = 30.0;
// Soft interior prior on rho near the box boundaries.
pub(crate) const RHO_SOFT_PRIOR_WEIGHT: f64 = 1e-6;
pub(crate) const RHO_SOFT_PRIOR_SHARPNESS: f64 = 4.0;
// Adaptive cubature guardrails for bounded correction latency.
pub(crate) const AUTO_CUBATURE_MAX_RHO_DIM: usize = 12;
pub(crate) const AUTO_CUBATURE_MAX_EIGENVECTORS: usize = 4;
pub(crate) const AUTO_CUBATURE_TARGET_VAR_FRAC: f64 = 0.95;
pub(crate) const AUTO_CUBATURE_MAX_BETA_DIM: usize = 1600;
pub(crate) const AUTO_CUBATURE_BOUNDARY_MARGIN: f64 = 2.0;

/// Smooth, differentiable approximation of `max(dp, floor)` where the floor
/// and the width of the smoothing band are taken **relative to the supplied
/// deviance `scale`** (the weighted null deviance `D₀` of the response).
///
/// Returns the smoothed value, first derivative, and second derivative with
/// respect to `dp`.
///
/// # Why the floor must be relative (issue #1127)
///
/// The penalized deviance `D_p = Σ wᵢ(yᵢ−μ̂ᵢ)² + β̂ᵀSβ̂` is exactly quadratic
/// in the response, so under a multiplicative rescale `y → a·y` it scales as
/// `D_p → a²·D_p`. The profiled Gaussian REML criterion depends on `D_p` only
/// through `log D_p` (the `(ν/2)·log(2πφ̂)` term, `φ̂ = D_p/ν`), so the rescale
/// shifts the cost by the *additive constant* `ν·log a` and leaves the
/// ρ-gradient — hence the selected `λ̂`, the EDF, and `ŝ(x)/a` — exactly
/// invariant. An **absolute** floor destroys this: when `a` is small enough
/// that `D_p` enters the fixed band (e.g. `D_p ≈ 3.6e-11` at `a = 1e-6` with
/// a band of width `1e-8`), `dp_c` is spuriously inflated toward the absolute
/// floor, `log dp_c` stops tracking `2·log a + const`, and the optimizer
/// converges at an over-smoothed `λ̂` — reshaping, not merely rescaling, the
/// smooth. Scaling both the floor and its width by `D₀ ∝ a²` makes the band a
/// fixed *fraction* of the deviance, so `smooth_floor_dp(a²·dp, a²·D₀) =
/// a²·smooth_floor_dp(dp, D₀)` exactly and equivariance is restored.
///
/// `scale = 1.0` recovers the historical absolute floor byte-for-byte, which
/// is the correct default for callers without a Gaussian response scale in
/// hand (the floor is consumed only on the profiled-Gaussian path).
pub(crate) fn smooth_floor_dp(dp: f64, scale: f64) -> (f64, f64, f64) {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let floor = DP_FLOOR * scale;
    let tau = (DP_FLOOR_SMOOTH_WIDTH * scale).max(f64::MIN_POSITIVE);
    let scaled = (dp - floor) / tau;

    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    let sigma = if scaled >= 0.0 {
        let exp_neg = (-scaled).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = scaled.exp();
        exp_pos / (1.0 + exp_pos)
    };

    let dp_c = floor + tau * softplus;
    let dp_cgrad2 = sigma * (1.0 - sigma) / tau;
    (dp_c, sigma, dp_cgrad2)
}

/// Compute the smoothing parameter uncertainty correction matrix `V_corr = J * V_rho * J^T`.
///
/// This implements the Wood et al. (2016) correction for smoothing parameter uncertainty.
/// The corrected covariance for `beta` is: `V*_beta = V_beta + J * V_rho * J^T`.
/// where:
/// - `V_beta = H^{-1}` (conditional covariance treating `lambda` as fixed)
/// - `J = d(beta)/d(rho)` (Jacobian wrt log-smoothing parameters)
/// - `V_rho = (d^2 LAML / d rho^2)^{-1}` (outer covariance)
///
/// Returns the correction matrix in the ORIGINAL coefficient basis.
///
/// Full correction reference.
/// Let `rho ~ N(mu, Sigma)` with `mu = rho_hat`, `Sigma = V_rho`,
/// and define:
/// - `A(rho) = H_rho^{-1}`
/// - `b(rho) = beta_hat_rho`
///
/// The exact Gaussian-mixture identity is:
///   `Var(beta) = E[A(rho)] + Var(b(rho))`.
///
/// Around `mu`, this routine keeps the first-order terms:
///
///   `E[A(rho)]      ~= A(mu) = Hmu^{-1}`
///   `Var(b(rho))    ~= J Sigma J^T`
///   `Var(beta)      ~= Hmu^{-1} + J V_rho J^T`.
///
/// Equivalent first-order propagation around the outer optimum `rho*`:
///
///   `Var(beta_hat) ~= Var(beta_hat | rho_hat) + (d beta_hat / d rho) Var(rho_hat) (d beta_hat / d rho)^T`
///                  `= V_beta + J V_rho J^T`.
///
/// Components:
///   `J[:,k] = d(beta_hat)/d(rho_k) = -H^{-1}(A_k beta_hat),  A_k = exp(rho_k) S_k`
///   `V_rho  = (d^2 V / d rho^2 at rho*)^{-1}`
///
/// Exact non-Gaussian V_ρ^{-1} requires the full Hessian with:
///   - tr(H^{-1}H_{kℓ})
///   - tr(H^{-1}H_k H^{-1}H_ℓ)
///   - pseudo-det second derivatives in S
///   - and H_{kℓ} terms containing fourth-likelihood derivatives.
///
/// This routine obtains V_ρ^{-1} from the analytic rho-space Hessian selected
/// by `compute_lamlhessian_consistent`, then regularizes before inversion.
/// If that analytic Hessian is unavailable, the correction is skipped rather
/// than synthesized numerically.
///
/// Notes on omitted higher-order terms:
/// - The exact `E[A(rho)]` and `Var(b(rho))` can be written with the Gaussian
///   smoothing/heat operator `exp(0.5 * Delta_Sigma)` (equivalently Wick/Isserlis
///   contractions of high-order derivatives).
/// - Those infinite-series corrections are not expanded in this routine.
pub(crate) struct SmoothingCorrectionComputation {
    pub correction: Option<Array2<f64>>,
    pub hessian_rho: Option<Array2<f64>>,
    /// Regularized inverse outer Hessian `Cov(rho_hat)` in the same rho ordering
    /// as the fitted smoothing-parameter vector. This exposes the #740 quantity
    /// to LR Bartlett inference without changing the production algebra that
    /// computes it.
    pub rho_covariance: Option<Array2<f64>>,
    /// Identified-subspace rank of the rho-Hessian inverse used to build
    /// `correction`. `Some(n)` if the matrix was SPD and fully inverted;
    /// `Some(r)` with `r < n` if the pseudo-inverse dropped non-identified
    /// directions; `None` when no inversion was attempted or it failed before
    /// producing a usable V_ρ. Downstream consumers (e.g. auto-cubature)
    /// use this to decide whether higher-order corrections are even
    /// meaningful — they aren't when V_ρ is rank-deficient.
    pub active_rank: Option<usize>,
}

/// Result of pseudo-inverting the rho-space LAML Hessian on the identified subspace.
///
/// When the outer rho-Hessian has negative or near-zero eigenvalues at convergence
/// (genuine non-convexity, Z₂-saddles from redundant penalty blocks, or weakly
/// identified ρ directions on no-signal data), inverting it naively would either
/// fail or yield an arbitrarily large V_ρ along those directions. Instead we
/// split the spectrum into an identified subspace (eigenvalues strictly above an
/// identifiability floor) and a non-identified subspace (negative, numerical zero,
/// or marginally positive but below the floor). The returned `inverse` is the
/// rank-deficient pseudo-inverse: 1/σ on the identified directions, 0 on the rest.
/// `J · V_ρ · J^T` is then a valid rank-deficient inflation along well-identified
/// rho directions, with zero contribution from non-identified directions.
pub(crate) struct InvertedRhoHessian {
    /// Pseudo-inverse projected onto the identified subspace.
    pub inverse: Array2<f64>,
    /// Number of eigenpairs retained (σ_i > tau).
    pub active_rank: usize,
    /// Eigenpairs dropped for σ_i < −neg_tol (genuine negative curvature).
    pub dropped_negative: usize,
    /// Eigenpairs dropped for marginally positive σ_i in (neg_tol, tau].
    pub dropped_small_positive: usize,
    /// Eigenpairs dropped for |σ_i| ≤ neg_tol (numerical zero).
    pub dropped_numerical_zero: usize,
    /// Smallest eigenvalue (signed) of the input Hessian.
    pub min_eigenvalue: f64,
    /// True whenever active_rank < n (i.e. anything was dropped). Cholesky fast
    /// path always returns false.
    pub repaired_hessian: bool,
    /// Per-eigenvalue classification (length n), aligned with the input matrix's
    /// eigendecomposition order from `eigh`. Used by the [INDEF-HESS] diagnostic.
    /// Empty on the Cholesky fast path (matrix was SPD, no classification needed).
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors as columns, aligned with `eigenvalues`. Empty on the Cholesky
    /// fast path. Carrying these here eliminates a second `eigh` call in the
    /// `[INDEF-HESS]` diagnostic — the slow path computes them once and shares.
    pub eigenvectors: Array2<f64>,
    /// Per-eigenvalue classification labels parallel to `eigenvalues`.
    pub classifications: Vec<EigenClassification>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EigenClassification {
    Active,
    DroppedNegative,
    DroppedSmallPositive,
    DroppedNumericalZero,
}

/// Invert the rho-space LAML Hessian onto the identified subspace.
///
/// Fast path: when the matrix is strictly positive-definite, returns the full
/// Cholesky inverse with `active_rank = n` and `repaired_hessian = false`.
///
/// Slow path: eigendecompose, classify each eigenpair, and assemble the
/// rank-deficient pseudo-inverse. Returns `None` only when the eigendecomposition
/// itself fails (non-finite eigenvalues or eigenvectors). An all-bad spectrum
/// (active_rank == 0) still returns `Some`; the caller is responsible for
/// deciding whether to use a zero-rank covariance.
pub(crate) fn invert_regularized_rho_hessian(
    hessian_rho: &Array2<f64>,
) -> Option<InvertedRhoHessian> {
    let n = hessian_rho.nrows();
    if let Ok(chol) = hessian_rho.cholesky(faer::Side::Lower) {
        let mut inverse = Array2::<f64>::eye(n);
        for col in 0..n {
            let colvec = inverse.column(col).to_owned();
            let solved = chol.solvevec(&colvec);
            inverse.column_mut(col).assign(&solved);
        }
        // Spectral scale / min eigenvalue are not needed when Cholesky succeeds,
        // but we surface coherent placeholders so downstream consumers can rely
        // on the struct fields unconditionally.
        return Some(InvertedRhoHessian {
            inverse,
            active_rank: n,
            dropped_negative: 0,
            dropped_small_positive: 0,
            dropped_numerical_zero: 0,
            min_eigenvalue: f64::NAN,
            repaired_hessian: false,
            eigenvalues: Array1::<f64>::zeros(0),
            eigenvectors: Array2::<f64>::zeros((0, 0)),
            classifications: Vec::new(),
        });
    }

    let (eigenvalues, eigenvectors) = hessian_rho.eigh(faer::Side::Lower).ok()?;
    if eigenvalues.iter().any(|v| !v.is_finite()) || !eigenvectors.iter().all(|v| v.is_finite()) {
        return None;
    }

    let spectral_scale = eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let neg_tol = 64.0 * f64::EPSILON * (n.max(1) as f64) * spectral_scale;
    let tau = (spectral_scale * 1e-10).max(LAML_RIDGE);

    let mut inverse = Array2::<f64>::zeros((n, n));
    let mut classifications = Vec::with_capacity(n);
    let mut active_rank = 0usize;
    let mut dropped_negative = 0usize;
    let mut dropped_small_positive = 0usize;
    let mut dropped_numerical_zero = 0usize;

    for i in 0..n {
        let sigma = eigenvalues[i];
        let class = if sigma > tau {
            EigenClassification::Active
        } else if sigma.abs() <= neg_tol {
            EigenClassification::DroppedNumericalZero
        } else if sigma > 0.0 {
            // 0 < sigma <= tau and |sigma| > neg_tol: marginally positive but
            // below the identifiability floor; 1/sigma would explode.
            EigenClassification::DroppedSmallPositive
        } else {
            // sigma < -neg_tol: genuine negative curvature.
            EigenClassification::DroppedNegative
        };
        classifications.push(class);
        match class {
            EigenClassification::Active => {
                active_rank += 1;
                let inv_lambda = 1.0 / sigma;
                let v = eigenvectors.column(i);
                for row in 0..n {
                    for col in 0..n {
                        inverse[[row, col]] += inv_lambda * v[row] * v[col];
                    }
                }
            }
            EigenClassification::DroppedNegative => dropped_negative += 1,
            EigenClassification::DroppedSmallPositive => dropped_small_positive += 1,
            EigenClassification::DroppedNumericalZero => dropped_numerical_zero += 1,
        }
    }

    Some(InvertedRhoHessian {
        inverse,
        active_rank,
        dropped_negative,
        dropped_small_positive,
        dropped_numerical_zero,
        min_eigenvalue,
        repaired_hessian: active_rank < n,
        eigenvalues,
        eigenvectors,
        classifications,
    })
}

/// Cosine threshold above which two penalty matrices are treated as the
/// structural-redundancy signature in [INDEF-HESS] diagnostics. Pairs with
/// cosine above this AND a dominant-negative eigenvector concentrated on
/// the pair's antisymmetric direction trigger the headline
/// `structural_redundancy_detected` line.
const INDEF_HESS_STRUCTURAL_REDUNDANCY_COS: f64 = 0.999;

/// Penalty-count crossover at which the [INDEF-HESS] pair dump switches from
/// the full O(k²) grid to top-3 pairs only. Bounds log volume on large-scale
/// rho_dim while keeping the per-pair detail useful for small models.
const INDEF_HESS_PAIR_DUMP_GRID_MAX_K: usize = 16;

/// Number of top-cosine pairs to dump when `n_pen > INDEF_HESS_PAIR_DUMP_GRID_MAX_K`.
const INDEF_HESS_PAIR_DUMP_TOP_N: usize = 3;

/// Diagnostic emitted whenever the post-fit rho-Hessian has at least one
/// non-identified direction (active_rank < n). Reports the eigendecomposition,
/// the dominant-negative eigenvector, per-eigenpair classification, and pairwise
/// penalty cosines tr(SᵢSⱼ)/√(tr(Sᵢ²)·tr(Sⱼ²)). A pair cosine ≈ 1.0 combined
/// with the negative eigenvector concentrated on that pair's antisymmetric
/// direction is the structural Z₂-saddle signature.
///
/// Output is capped: when the penalty count exceeds 16, only the top-3
/// highest-cosine pairs are dumped instead of the full O(k²) grid. When the
/// structural-redundancy signature is detected, a single headline line is
/// emitted with the offending pair, cosine, and antisymmetric projection.
fn dump_indefinite_rho_hessian_diagnostic(
    hessian_rho: &Array2<f64>,
    final_rho: &Array1<f64>,
    canonical: &[crate::construction::CanonicalPenalty],
    inverted: Option<&InvertedRhoHessian>,
) {
    let k = hessian_rho.nrows();
    if k == 0 {
        return;
    }

    // Reuse the eigendecomposition already computed by the inverter when present
    // (the slow path always populates it). Only recompute on the rare paths
    // where the diagnostic is called without an `InvertedRhoHessian` (e.g. the
    // eigendecomposition-failed bail in `compute_smoothing_correction`).
    let (eigenvalues_owned, eigenvectors_owned);
    let (eigenvalues_ref, eigenvectors_ref) = match inverted {
        Some(inv) if !inv.eigenvalues.is_empty() && !inv.eigenvectors.is_empty() => {
            (&inv.eigenvalues, &inv.eigenvectors)
        }
        _ => match hessian_rho.eigh(faer::Side::Lower) {
            Ok((evals, evecs)) => {
                eigenvalues_owned = evals;
                eigenvectors_owned = evecs;
                (&eigenvalues_owned, &eigenvectors_owned)
            }
            Err(err) => {
                log::warn!("[INDEF-HESS] eigendecomposition failed: {err}");
                return;
            }
        },
    };

    log::warn!("[INDEF-HESS] rho={:?}", final_rho.as_slice().unwrap_or(&[]),);
    log::warn!(
        "[INDEF-HESS] eigenvalues={:?}",
        eigenvalues_ref.as_slice().unwrap_or(&[]),
    );
    if let Some(inv) = inverted {
        log::warn!(
            "[INDEF-HESS] active_rank={}/{} dropped_negative={} dropped_small_positive={} dropped_numerical_zero={}",
            inv.active_rank,
            k,
            inv.dropped_negative,
            inv.dropped_small_positive,
            inv.dropped_numerical_zero,
        );
        if !inv.classifications.is_empty() {
            let labels: Vec<&'static str> = inv
                .classifications
                .iter()
                .map(|c| match c {
                    EigenClassification::Active => "A",
                    EigenClassification::DroppedNegative => "N",
                    EigenClassification::DroppedSmallPositive => "P",
                    EigenClassification::DroppedNumericalZero => "Z",
                })
                .collect();
            log::warn!(
                "[INDEF-HESS] classifications={:?} (A=active N=neg P=small_pos Z=numerical_zero)",
                labels,
            );
        }
    }

    let mut neg_idx = 0usize;
    let mut min_eig = f64::INFINITY;
    for (i, &v) in eigenvalues_ref.iter().enumerate() {
        if v < min_eig {
            min_eig = v;
            neg_idx = i;
        }
    }
    let v_neg = eigenvectors_ref.column(neg_idx);
    log::warn!(
        "[INDEF-HESS] negative_eigenvalue={:.4e} eigenvector={:?}",
        min_eig,
        v_neg.as_slice().unwrap_or(&[]),
    );

    let n_pen = canonical.len();
    let mut tr_aa = vec![0.0_f64; n_pen];
    for i in 0..n_pen {
        let local = &canonical[i].local;
        let mut s = 0.0;
        for r in 0..local.nrows() {
            for c in 0..local.ncols() {
                s += local[[r, c]] * local[[r, c]];
            }
        }
        tr_aa[i] = s;
    }
    log::warn!(
        "[INDEF-HESS] penalty_count={} ranges={:?} ranks={:?}",
        n_pen,
        (0..n_pen)
            .map(|i| (canonical[i].col_range.start, canonical[i].col_range.end))
            .collect::<Vec<_>>(),
        (0..n_pen).map(|i| canonical[i].rank()).collect::<Vec<_>>(),
    );

    // Collect compatible pairs with their cosines.
    struct PairCos {
        i: usize,
        j: usize,
        cos: f64,
        antisym_proj: f64,
    }
    let mut pairs: Vec<PairCos> = Vec::new();
    for i in 0..n_pen {
        for j in (i + 1)..n_pen {
            let ci = &canonical[i];
            let cj = &canonical[j];
            if ci.col_range != cj.col_range {
                continue;
            }
            let local_i = &ci.local;
            let local_j = &cj.local;
            let mut dot = 0.0;
            for r in 0..local_i.nrows() {
                for c in 0..local_i.ncols() {
                    dot += local_i[[r, c]] * local_j[[r, c]];
                }
            }
            let cos = if tr_aa[i] > 0.0 && tr_aa[j] > 0.0 {
                dot / (tr_aa[i].sqrt() * tr_aa[j].sqrt())
            } else {
                f64::NAN
            };
            let antisym_proj = if v_neg.len() == n_pen {
                (v_neg[i] - v_neg[j]) / std::f64::consts::SQRT_2
            } else {
                f64::NAN
            };
            pairs.push(PairCos {
                i,
                j,
                cos,
                antisym_proj,
            });
        }
    }

    // Headline: structural redundancy detection. Pair cosine above the
    // structural-redundancy threshold AND the dominant-negative eigenvector's
    // top-2 absolute components on indices (i, j) with opposite signs.
    if min_eig < 0.0 && v_neg.len() == n_pen {
        for p in &pairs {
            if !(p.cos > INDEF_HESS_STRUCTURAL_REDUNDANCY_COS) {
                continue;
            }
            let mut indexed: Vec<(usize, f64)> = v_neg
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();
            indexed.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if indexed.len() < 2 {
                continue;
            }
            let top0 = indexed[0].0;
            let top1 = indexed[1].0;
            let (a, b) = if top0 == p.i && top1 == p.j {
                (indexed[0].1, indexed[1].1)
            } else if top0 == p.j && top1 == p.i {
                (indexed[1].1, indexed[0].1)
            } else {
                continue;
            };
            if a * b >= 0.0 {
                continue;
            }
            log::warn!(
                "[INDEF-HESS] structural_redundancy_detected pair=({},{}) cos={:.6} antisym_proj={:.4e}",
                p.i,
                p.j,
                p.cos,
                p.antisym_proj,
            );
            break;
        }
    }

    // Cap output: dump the full grid when small, otherwise only the top-N
    // highest-cosine pairs.
    if n_pen <= INDEF_HESS_PAIR_DUMP_GRID_MAX_K {
        for p in &pairs {
            log::warn!(
                "[INDEF-HESS] pair=({},{}) cos={:.6} tr_ii={:.4e} tr_jj={:.4e} v_neg[i]-v_neg[j]/sqrt2={:.4e}",
                p.i,
                p.j,
                p.cos,
                tr_aa[p.i],
                tr_aa[p.j],
                p.antisym_proj,
            );
        }
        // Note: we no longer log a "ranges_differ" line per skipped pair to
        // keep the diagnostic O(k). The headline pair already captures intent.
    } else {
        let mut top: Vec<&PairCos> = pairs.iter().filter(|p| p.cos.is_finite()).collect();
        top.sort_by(|a, b| {
            b.cos
                .abs()
                .partial_cmp(&a.cos.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for p in top.iter().take(INDEF_HESS_PAIR_DUMP_TOP_N) {
            log::warn!(
                "[INDEF-HESS] top_pair=({},{}) cos={:.6} tr_ii={:.4e} tr_jj={:.4e} v_neg[i]-v_neg[j]/sqrt2={:.4e}",
                p.i,
                p.j,
                p.cos,
                tr_aa[p.i],
                tr_aa[p.j],
                p.antisym_proj,
            );
        }
    }
}

pub(crate) fn compute_smoothing_correction(
    reml_state: &RemlState<'_>,
    final_rho: &Array1<f64>,
    final_fit: &pirls::PirlsResult,
) -> SmoothingCorrectionComputation {
    use crate::faer_ndarray::{FaerCholesky, FaerEigh};

    let n_rho = final_rho.len();
    if n_rho == 0 {
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: None,
            rho_covariance: None,
            active_rank: None,
        };
    }

    let n_coeffs_trans = final_fit.beta_transformed.len();
    let n_coeffs_orig = final_fit.reparam_result.qs.nrows();
    let lambdas: Array1<f64> = final_rho.mapv(f64::exp);

    // Step 1: Compute the Jacobian J = d(beta)/d(rho) in transformed space.
    //
    // Exact implicit-function identity at the inner optimum:
    //   dβ̂/dρ_k = -H^{-1}(S_k^ρ (β̂ - μ_k)),   S_k^ρ = λ_k S_k,
    //   λ_k = exp(ρ_k).
    //
    // In transformed coordinates with root penalties S_k = R_kᵀR_k:
    //   S_k (β̂ - μ_k) = R_kᵀ(R_k (β̂ - μ_k)),
    // so each Jacobian column is one linear solve with H.

    // Use the same objective-consistent inner Hessian surface used by REML:
    // - non-Firth: H = X'W_HX + S (+ stabilization if present)
    // - Firth logit: H_total = H - d²Phi/dβ²
    // Fallback to PIRLS stabilized Hessian only if bundle recovery fails.
    //
    // Conclusion:
    //   J[:,k] = dβ̂/dρ_k must use the Jacobian of the actual stationarity
    //   system G*(β,ρ)=0, i.e. H_total for Firth-adjusted fits. Using only
    //   X'W_HX+S here would be inconsistent with the fitted objective and would
    //   misstate smoothing-parameter uncertainty propagation.
    let h_trans = reml_state
        .objective_innerhessian(final_rho)
        .unwrap_or_else(|_| final_fit.stabilizedhessian_transformed.to_dense());

    // The IFT solve below feeds length-`n_coeffs_trans` right-hand sides into
    // the Cholesky factor of `h_trans`, and faer asserts `rhs.len() == factor.n()`.
    // A Hessian that does not match the coefficient dimension (e.g. a degenerate
    // 0×0 placeholder from a geometry backend that failed to materialize a real
    // dense inner Hessian) would otherwise abort the whole fit inside the solve.
    // Bail to the no-correction branch exactly like the Cholesky-`Err` guard
    // below, so the post-fit smoothing correction is simply skipped.
    if h_trans.nrows() != n_coeffs_trans || h_trans.ncols() != n_coeffs_trans {
        log::warn!(
            "smoothing-correction inner Hessian shape {}x{} does not match coefficient dimension {}; skipping.",
            h_trans.nrows(),
            h_trans.ncols(),
            n_coeffs_trans
        );
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: None,
            rho_covariance: None,
            active_rank: None,
        };
    }

    // Factor the Hessian for solving
    let h_chol = match h_trans.cholesky(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => {
            log::warn!("Cholesky decomposition failed for smoothing correction; skipping.");
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                rho_covariance: None,
                active_rank: None,
            };
        }
    };

    let beta_trans = final_fit.beta_transformed.as_ref();
    let ct = &final_fit.reparam_result.canonical_transformed;

    // Build the stationarity-gradient derivative matrix G_ρ where column k is
    // ∂g(β,ρ)/∂ρ_k = λ_k S_k(β - μ_k), then delegate the IFT solve
    // dβ/dρ = -H⁻¹G_ρ to the canonical evidence helper. This keeps the
    // coefficient-space prediction correction and the joint-evidence
    // Arrow-Schur path on the same hand-derived IFT identity.
    let mut dg_drho_trans = Array2::<f64>::zeros((n_coeffs_trans, n_rho));
    // Per-ρ_k support: the coefficient range its stationarity-gradient
    // derivative ∂g/∂ρ_k is nonzero on. Each column is block-local (only the
    // k-th penalty block), so this is exactly cp.col_range; structurally
    // inactive columns keep an empty support and the cone-of-influence solve
    // skips them entirely (their sensitivity is identically zero). See #779.
    let mut col_supports: Vec<std::ops::Range<usize>> = vec![0..0; n_rho];
    for k in 0..n_rho {
        if k >= ct.len() {
            continue;
        }
        let cp = &ct[k];
        if cp.rank() == 0 {
            continue;
        }
        // S_k(β - μ) — block-local: R^T (R (β[block] - μ)), embedded into p-vector.
        let r = &cp.col_range;
        col_supports[k] = r.start..r.end;
        let beta_block = beta_trans.slice(s![r.start..r.end]);
        let centered = &beta_block - &cp.prior_mean;
        let r_beta = cp.root.dot(&centered);
        for a in 0..cp.block_dim() {
            dg_drho_trans[[r.start + a, k]] = lambdas[k]
                * (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum::<f64>();
        }
    }
    // Lazy/local cone-of-influence propagation (#779): confine each column's
    // sensitivity to the coupling component of `h_trans` containing the moved
    // penalty block, and skip structurally inactive columns. Exact on a
    // block-decoupled Hessian (entries outside the cone are identically zero)
    // and identical to the full joint solve on a fully coupled Hessian.
    let jacobian_trans = match crate::solver::sensitivity::FitSensitivity::from_faer_cholesky(
        &h_chol,
        n_coeffs_trans,
    )
    .mode_response_coned(h_trans.view(), dg_drho_trans.view(), &col_supports)
    {
        Some(jacobian) => jacobian,
        None => {
            log::warn!("IFT beta-rho sensitivity solve failed for smoothing correction; skipping.");
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                rho_covariance: None,
                active_rank: None,
            };
        }
    };

    // Step 2: Build V_rho by inverting the LAML Hessian in rho-space.
    // The authoritative inner-strategy path chooses the rho-space Hessian
    // evaluation policy here. Unified may still perform local numerical
    // salvage inside the exact branch, but the branch choice itself no longer
    // lives inline at the call site.
    let mut hessian_rho = match reml_state.compute_lamlhessian_consistent(final_rho) {
        Ok(h) => h,
        Err(err) => {
            log::warn!(
                "LAML Hessian unavailable ({}); skipping smoothing correction.",
                err
            );
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                rho_covariance: None,
                active_rank: None,
            };
        }
    };

    // Symmetrize the Hessian
    crate::matrix::symmetrize_in_place(&mut hessian_rho);

    // Step 3: Invert Hessian to get V_rho.
    // Add a small ridge before factorization to regularize weakly identified ρ directions.
    add_relative_diag_ridge(&mut hessian_rho, LAML_RIDGE, LAML_RIDGE);

    let inverted = match invert_regularized_rho_hessian(&hessian_rho) {
        Some(inv) => inv,
        None => {
            log::warn!(
                "Eigendecomposition of LAML rho Hessian failed for smoothing correction; skipping."
            );
            dump_indefinite_rho_hessian_diagnostic(
                &hessian_rho,
                final_rho,
                &final_fit.reparam_result.canonical_transformed,
                None,
            );
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: Some(hessian_rho),
                rho_covariance: None,
                active_rank: None,
            };
        }
    };

    let n_rho_total = hessian_rho.nrows();
    if inverted.active_rank == 0 {
        // All directions non-identified. Pseudo-inverse is zero, so J·V_ρ·J^T
        // adds nothing; report no correction (consistent with the prior behavior
        // for fully indefinite Hessians, but now logged with full context).
        log::info!(
            "LAML rho Hessian has no identified directions (active_rank=0/{}, dropped_negative={}, dropped_small_positive={}, dropped_numerical_zero={}, min_eig={:.3e}); smoothing correction is zero, skipping.",
            n_rho_total,
            inverted.dropped_negative,
            inverted.dropped_small_positive,
            inverted.dropped_numerical_zero,
            inverted.min_eigenvalue,
        );
        dump_indefinite_rho_hessian_diagnostic(
            &hessian_rho,
            final_rho,
            &final_fit.reparam_result.canonical_transformed,
            Some(&inverted),
        );
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: Some(hessian_rho),
            rho_covariance: Some(inverted.inverse),
            active_rank: Some(0),
        };
    }

    if inverted.active_rank < n_rho_total {
        log::info!(
            "LAML rho Hessian is rank-deficient on the identified subspace (active_rank={}/{}, dropped_negative={}, dropped_small_positive={}, dropped_numerical_zero={}, min_eig={:.3e}); proceeding with pseudo-inverse V_ρ.",
            inverted.active_rank,
            n_rho_total,
            inverted.dropped_negative,
            inverted.dropped_small_positive,
            inverted.dropped_numerical_zero,
            inverted.min_eigenvalue,
        );
        dump_indefinite_rho_hessian_diagnostic(
            &hessian_rho,
            final_rho,
            &final_fit.reparam_result.canonical_transformed,
            Some(&inverted),
        );
    }

    let repaired_hessian = inverted.repaired_hessian;
    let active_rank_used = inverted.active_rank;
    let v_rho = inverted.inverse;
    let rho_covariance = v_rho.clone();
    if repaired_hessian {
        log::debug!(
            "Applied rank-deficient pseudo-inverse on identified rho-Hessian subspace before smoothing correction."
        );
    }

    // Step 4: Compute V_corr = J * V_rho * J^T in transformed space.
    //
    // This is the first-order smoothing-parameter uncertainty inflation:
    //   Var(β̂) ≈ Var(β̂|ρ̂) + (dβ̂/dρ) Var(ρ̂) (dβ̂/dρ)ᵀ.
    //
    // Here:
    //   J = dβ̂/dρ,  J[:,k] = -H^{-1}(A_k β̂),
    //   V_ρ = (∇²_{ρρ}V)^{-1} evaluated at the final ρ.
    let jv_rho = jacobian_trans.dot(&v_rho); // (n_coeffs_trans x n_rho)
    let v_corr_trans = jv_rho.dot(&jacobian_trans.t()); // (n_coeffs_trans x n_coeffs_trans)

    // Step 5: Transform back to original coefficient basis:
    // V_corr_orig = Qs * V_corr_trans * Qs^T
    let qs = &final_fit.reparam_result.qs;
    let qsv = qs.dot(&v_corr_trans);
    let v_corr_orig = qsv.dot(&qs.t());

    // Validate the result
    if !v_corr_orig.iter().all(|v| v.is_finite()) {
        log::warn!("Non-finite values in smoothing correction matrix; skipping.");
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: Some(hessian_rho),
            rho_covariance: Some(rho_covariance),
            active_rank: Some(active_rank_used),
        };
    }

    // Ensure positive semi-definiteness without hiding a failed rho-space
    // optimum. The smoothing correction V_corr = J·V_ρ·Jᵀ is an SE *inflation*
    // (Var(β̂|ρ̂) + uncertainty-in-ρ̂). It is PSD *by construction*: V_ρ is the
    // (active-subspace) inverse of the SPD ρ-Hessian, so the congruence
    // J·V_ρ·Jᵀ cannot have genuine negative curvature. Crucially it is also
    // rank-deficient — its rank is at most `n_rho`, while it lives in the
    // `n_coeffs_orig`-dimensional coefficient space — so for every model with
    // fewer smoothing parameters than coefficients (i.e. essentially all of
    // them) it has exact-zero eigenvalues. A symmetric eigensolver renders
    // those exact zeros as ±O(ε·‖V_corr‖), so the smallest eigenvalue is
    // routinely a *sub-tolerance negative* that is pure floating-point
    // roundoff, not curvature. Rejecting the whole correction on such a
    // roundoff zero silently dropped Vp for the entire #smooth < #coef regime
    // (every predict() interval lost its smoothing-parameter inflation).
    //
    // So we only treat a *material* negative (below the eigensolver roundoff
    // floor `neg_tol`) as a real failure — that signals a corrupted V_ρ
    // (near-saddle / pinv-imputed direction) and we skip loudly, letting the
    // caller fall back to the honest base covariance. Sub-tolerance negatives
    // are accepted as-is: the matrix is PSD to roundoff, and it is added to the
    // (dominant) base covariance Vb, which keeps Vp PSD without any relabelling
    // or clamping.
    match v_corr_orig.eigh(faer::Side::Lower) {
        // Eigenvectors are unused: only the smallest eigenvalue's magnitude
        // distinguishes roundoff from genuine indefiniteness.
        Ok((eigenvalues, _)) => {
            let min_eig = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let spectral_scale = eigenvalues
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let neg_tol = 64.0 * f64::EPSILON * (n_coeffs_orig.max(1) as f64) * spectral_scale;
            if min_eig < -neg_tol {
                log::warn!(
                    "Smoothing correction has material negative eigenvalue {:.3e} \
                     below tolerance {:.3e}; skipping correction.",
                    min_eig,
                    neg_tol
                );
                return SmoothingCorrectionComputation {
                    correction: None,
                    hessian_rho: Some(hessian_rho),
                    rho_covariance: Some(rho_covariance),
                    active_rank: Some(active_rank_used),
                };
            }
        }
        Err(_) => {
            log::warn!("Eigendecomposition failed for smoothing correction validation.");
        }
    }
    SmoothingCorrectionComputation {
        correction: Some(v_corr_orig),
        hessian_rho: Some(hessian_rho),
        rho_covariance: Some(rho_covariance),
        active_rank: Some(active_rank_used),
    }
}

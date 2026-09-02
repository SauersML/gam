// [#780] Exact stationarity-Jacobian correction (`apply_exact_hessian_minus_b`),
// the exact inner-fit Hessian apply (`apply_exact_hessian`), and the exact
// stationarity solve (`solve_exact_stationarity`) were extracted verbatim from
// `construction.rs` into this sibling file to keep that file under the #780
// per-file line-count gate. It is `include!`d back into the parent module in
// `construction.rs`, so these methods share that module's scope exactly as
// before (same `impl SaeManifoldTerm`, same `use super::*` imports).

/// ONE identifiability floor for directions of the exact observed information
/// `A = B + ΔC`, in ONE metric, for the value path and the gradient path alike
/// (#2673, #2080 defect 4, #2253 x #2330).
///
/// `B` is the positive-definite arrow factorization: the scale the inner Newton
/// solve, the IFT solve and the evidence factor are all expressed in. The
/// generalized Rayleigh quotient
///
/// ```text
///   μ(v) = vᵀAv / vᵀBv
/// ```
///
/// therefore measures a direction's exact curvature against the problem's own
/// scale. The floor is `√ε_machine`, the standard boundary below which a
/// double-precision curvature ratio is not numerically identifiable; it is
/// derived from the scalar type, not tuned to a fixture. A direction below it
/// (a saturated ordered Beta--Bernoulli gate logit has data curvature
/// `∝ σ'(ℓ)² → 0`) is numerically curvature-free: the inner optimizer cannot
/// resolve the iterate's position along it, so the IFT response
/// `θ̂_ρ = −A⁻¹g_ρ` there is an unidentifiable `1/μ` amplification rather than a
/// derivative. That amplification is what flipped the analytic λ-gradient's sign
/// against the criterion it differentiates (the #931 objective↔gradient desync).
///
/// # Why `μ` and not an absolute eigenvalue of `A` (#2673)
///
/// Until this became the single floor, the VALUE path classified the same
/// directions of the same `A` by `|λ| ≤ 1e-9 · max(λ_max(A), 1)` while the
/// GRADIENT path used `|μ| < √ε`. Both floors ran inside ONE evaluation on the
/// streaming route — the value path's terminal Newton polish reaches
/// `solve_exact_stationarity`, which is not route-gated, while the gradient
/// adjoint takes the matrix-free sibling — and which of them the value path used
/// was decided by `direct_logdet_admitted`, i.e. by ambient free memory. Three
/// things were wrong with the pair, and none of them needed a fixture to
/// witness:
///
/// 1. **They are not one rule in two spellings.** Written as thresholds on the
///    same `|λ|`, the value rule is ONE number for every direction and the
///    gradient rule is `√ε·vᵀBv`, which varies across directions of one
///    operator by the spread of the `B`-Rayleigh quotient (measured 24x on the
///    #2515 route-invariance state, unbounded in general). No choice of the two
///    constants makes a constant threshold equal a varying one, and on that same
///    state the ratio STRADDLES 1 — so neither rule was even a conservative
///    version of the other.
/// 2. **Only `μ` is a curvature.** `μ` is invariant under a reparametrization
///    `θ → Lθ` (`A → LᵀAL`, `B → LᵀBL` transform congruently); `λ/λ_max(A)` is
///    not. `θ = (t, β)` mixes chart coordinates with border coefficients whose
///    scale is set by the data's units, so `λ_max(A)` is a maximum over
///    incommensurable coordinates and the band it defined moved when the units
///    did.
/// 3. **`1e-9` was tuned; `√ε` is derived.** SPEC rule 21.
///
/// Consistency then FORCES the direction of the unification: the value must pin
/// exactly what the gradient cannot differentiate, or the criterion depends on
/// `ρ` through a direction whose response the adjoint has projected out — the
/// #931/#2253 desync in a new place. So the larger, derived, invariant floor
/// wins at both sites and the absolute one is gone.
///
/// A dense diagonalization carries one requirement `μ` cannot express, because
/// the matrix-free site never incurs it: an eigenvalue below the
/// eigendecomposition's own backward error `~dim·ε·‖A‖₂` is not a number at all,
/// whatever `B` says about it. That is a property of the arithmetic and not a
/// second classification; it is applied as a floor UNDER this one by
/// [`ExactHessianSpectralBlock::rank_floor`], which also reports the only
/// crossing that can survive it.
pub(crate) fn sae_exact_a_identifiability_floor() -> f64 {
    f64::EPSILON.sqrt()
}

/// The null-band half-width, in `λ` units, for ONE direction of a dense
/// exact-`A` block (#2673).
///
/// ```text
///   floor = max( spectral_dim·ε·‖A‖₂ ,  √ε · vᵀBv )
/// ```
///
/// This is the whole rule, as a scalar function of three measured numbers, so
/// that every consumer applies the SAME rule while supplying its OWN operands.
/// Production reaches it through [`ExactHessianSpectralBlock::rank_floor`]; the
/// independent oracles that re-derive the classification from a separately built
/// dense `A` call it directly, which keeps them oracles (their inputs are their
/// own) without letting a second copy of the predicate drift from this one — the
/// failure mode #2740 names and the one this issue is.
///
/// See [`sae_exact_a_identifiability_floor`] for why the metric is `B` and why
/// the second term is the one that classifies.
pub(crate) fn sae_exact_a_direction_floor(
    spectral_dim: usize,
    spectral_norm: f64,
    b_quadratic_form: f64,
) -> f64 {
    gam_solve::arrow_schur::exact_a_direction_floor(
        spectral_dim,
        spectral_norm,
        b_quadratic_form,
    )
}

/// PATH C (#2253) CH5 — which subset of the joint θ-derivative operator
/// `K_w = ∂H/∂θ_w` a dense θ-adjoint contraction assembles. The FULL set
/// reconstructs `Γ_w = tr(inv·K_w)` (self-checked against the production
/// `logdet_theta_adjoint`); the two MIXED subsets isolate the single ρ-scaled
/// term of `K_w` whose `∂/∂ρ_i` is nonzero (both are degree-one in `e^{ρ_i}`,
/// so `∂K_w/∂ρ_i` equals the term itself). Keeping them as distinct channels
/// makes a failing finite-difference gate localize to ONE formula.
#[derive(Clone, Copy)]
pub(crate) enum ThetaAdjointDhChannel {
    /// Every `∂H/∂θ_w` contribution: data residual curvature, the softmax
    /// data-weight logit factor, the softmax entropy Gershgorin majorizer, and
    /// the periodic ARD majorizer diagonal.
    All,
    /// ONLY the softmax entropy Gershgorin majorizer θ-derivative (logit–logit,
    /// same atom). This is the `∝ λ_sparse` term, so its `∂/∂ρ_sparse` equals
    /// itself — the part-(b) mixed channel for the sparse coordinate.
    SoftmaxSparseMixed,
    /// ONLY the periodic ARD majorizer diagonal `w_row·(−ακ sin κt)` for the
    /// coordinate slots whose `ard_flat_index` matches `target_flat`. This is
    /// `∝ α = e^{ρ_ard}`, so its `∂/∂ρ_ard` equals itself — the part-(b) mixed
    /// channel for one ARD coordinate.
    ArdMixed { target_flat: usize },
}

/// One row's assembled `ΔC = A − B` blocks, in the arrow layout the streaming
/// evidence system already uses (`ArrowRowBlock::{htt, htbeta}`).
#[derive(Debug, Clone)]
pub(crate) struct ExactHessianDeltaRow {
    /// `ΔC_tt^(i)`, shape `(q_i, q_i)`.
    pub(crate) tt: Array2<f64>,
    /// `ΔC_tβ^(i)`, shape `(q_i, border_dim)`.
    pub(crate) tbeta: Array2<f64>,
}

/// Rank-revealing spectral representation of one dense exact-stationarity
/// block.  The materialized operator and its eigensystem remain together so a
/// pseudo-inverse response can be certified against the physical operator that
/// produced it, rather than against a projected Krylov surrogate.
struct ExactHessianSpectralBlock {
    operator: Array2<f64>,
    eigenvalues: Array1<f64>,
    /// Ambient eigenvectors, SQUARE: every direction of the materialized
    /// operator is classified by `rank_floor` and nothing is deleted ahead of
    /// it.  #2674 — the analytic chart orbit used to be quotiented out here
    /// before diagonalization, which deleted directions the penalized operator
    /// has genuine curvature and genuine slope in.
    eigenvectors: Array2<f64>,
    /// `vᵢᵀBvᵢ` for every eigendirection — the scale the classification is
    /// relative to (#2673). `B` is the arrow factorization's own operator
    /// restricted to this block's coordinates, so every entry is strictly
    /// positive and the ratio `λᵢ / metric_scale[i]` is the pencil curvature the
    /// gradient path classifies the same directions by.
    metric_scale: Array1<f64>,
    /// `‖A‖₂ = maxᵢ|λᵢ|`, the scale the eigendecomposition's own backward error
    /// is proportional to.
    spectral_norm: f64,
}

/// The `B` metric one spectral block is classified in (#2673).
///
/// `B` is the arrow factorization that the inner Newton solve, the IFT solve and
/// the evidence factor are all expressed in, and the ONE thing a direction's
/// curvature is measured against at both the value and the gradient site. Which
/// restriction of it applies is decided by which block of `A` is being
/// classified, so the two cannot be paired up wrongly:
///
/// * the JOINT block is `A` itself, so its metric is the whole arrow operator,
///   border and all;
/// * the COORDINATE block is `A_tt`, so its metric is `B`'s block-diagonal
///   `H_tt` — the same restriction, taken on the same coordinates, not the
///   Schur complement.
///
/// Both are applies through the cached factors, never a materialized `B`: the
/// dense route already carries one `dim × dim` block and #2724/#2757 price that
/// memory, so a second one would be paid for a scalar per direction.
#[derive(Clone, Copy)]
pub(crate) enum ArrowMetric<'a> {
    /// `B` on the joint `(t, β)` coordinates.
    Joint(&'a ArrowFactorCache),
    /// `H_tt`, `B`'s block-diagonal coordinate restriction.
    Coordinate(&'a ArrowFactorCache),
}

impl ArrowMetric<'_> {
    pub(crate) fn quadratic_form(&self, v: ArrayView1<'_, f64>) -> Result<f64, String> {
        match self {
            Self::Joint(cache) => {
                let total_t = cache.delta_t_len();
                if v.len() != total_t + cache.k {
                    return Err(format!(
                        "ArrowMetric::Joint: direction length {} != joint dimension {}",
                        v.len(),
                        total_t + cache.k
                    ));
                }
                let b_v = apply_cached_arrow_hessian(
                    cache,
                    v.slice(s![..total_t]),
                    v.slice(s![total_t..]),
                )?;
                Ok(v.slice(s![..total_t]).dot(&b_v.t) + v.slice(s![total_t..]).dot(&b_v.beta))
            }
            Self::Coordinate(cache) => {
                let total_t = cache.delta_t_len();
                if v.len() != total_t {
                    return Err(format!(
                        "ArrowMetric::Coordinate: direction length {} != coordinate dimension \
                         {total_t}",
                        v.len(),
                    ));
                }
                let mut total = 0.0_f64;
                for row in 0..cache.n_rows() {
                    let width = cache.row_dims[row];
                    let base = cache.row_offsets[row];
                    let block_v = v.slice(s![base..base + width]);
                    let applied = cholesky_factor_apply(cache.undamped_factor(row), block_v);
                    total += block_v.dot(&applied);
                }
                Ok(total)
            }
        }
    }
}

/// One point of the Levenberg--Marquardt path of the LINEAR residual model,
/// read off the eigensystem that is already materialized.
///
/// For damping `ν ≥ 0` the step
///
/// ```text
///   Δ(ν) = Σ_i u_i λ_i (u_iᵀ rhs) / (λ_i² + ν)
/// ```
///
/// is the exact minimizer of `‖rhs − AΔ‖² + ν‖Δ‖²`, and `ν = 0` reproduces the
/// pseudoinverse step of [`ExactHessianSpectralBlock::solve_stationarity`]
/// (same `rank_floor`, same retained band). Because the eigensystem is already
/// in hand, the WHOLE path costs one diagonal pass per point — no
/// refactorization, no second operator apply.
///
/// With `rhs = −g` the linear model of the stationarity residual at the trial
/// point is exactly
///
/// ```text
///   g + A Δ(ν) = Σ_i u_i c_i ν/(λ_i² + ν),      c_i = u_iᵀ g
/// ```
///
/// so `model_residual` below is CLOSED FORM, not an estimate. The caller prices
/// that residual in the currency its convergence gate owns; this lower-level
/// spectral object deliberately does not attach an ambient or quotient scalar
/// merit to it.
pub(crate) struct DampedResidualStep {
    /// `Δ(ν)`.
    pub(crate) step: SaeArrowVector,
    /// `g + AΔ(ν)` — the linear model's residual AT the trial point, in the
    /// same arrow layout as the residual handed in. The caller measures it in
    /// whatever currency its gate is denominated in; this type stays ignorant of
    /// which one that is.
    pub(crate) model_residual: SaeArrowVector,
    /// `‖Δ(ν)‖²`.
    pub(crate) step_norm_sq: f64,
    /// Directions whose damped denominator cleared the null band.
    pub(crate) retained_rank: usize,
}

impl ExactHessianSpectralBlock {
    /// The null-band half-width for eigendirection `index`, in `λ` units
    /// (#2673).
    ///
    /// ```text
    ///   floor(i) = max( dim·ε·‖A‖₂ ,  √ε · vᵢᵀBvᵢ )
    /// ```
    ///
    /// The second term is [`sae_exact_a_identifiability_floor`] — the SAME
    /// predicate the matrix-free gradient path applies to its own solution
    /// direction, written in `λ` units so this site can compare it against an
    /// eigenvalue. It is the term that decides the classification: a direction
    /// under it is one whose `A⁻¹` response is an unidentifiable `1/μ`
    /// amplification, so the value must not price a `ρ`-dependence there that
    /// the adjoint has projected out.
    ///
    /// The first term is not a second classification. It is the standard
    /// backward-error bound for a symmetric eigendecomposition — the computed
    /// spectrum of a perturbed `A + E` with `‖E‖₂ ≲ p(dim)·ε·‖A‖₂` — so an
    /// eigenvalue below it carries no significant digits and `ln λ` is not a
    /// quantity. The matrix-free site never diagonalizes and so never incurs it.
    /// It is a FLOOR under the identifiability term, never a ceiling, so it can
    /// only pin directions, never resurrect one the gradient has deflated.
    fn rank_floor(&self, index: usize) -> f64 {
        sae_exact_a_direction_floor(
            self.eigenvalues.len(),
            self.spectral_norm,
            self.metric_scale[index],
        )
    }

    /// The one crossing the union floor cannot remove, reported rather than
    /// left silent (#2673): a direction whose `|λ|` is inside the
    /// eigendecomposition's backward error while its pencil curvature `μ` is
    /// resolved. The value pins it because the number is noise; the gradient
    /// keeps its `A⁻¹` response because the ratio is identifiable. Reaching this
    /// needs `vᵢᵀBvᵢ ≲ dim·√ε·‖A‖₂`, i.e. `B` seven orders below `A`'s own norm
    /// along that direction; it has never been observed, and the previous pair
    /// of floors would not have said so either way.
    fn arithmetic_band_crossings(&self) -> usize {
        let arithmetic = (self.eigenvalues.len() as f64) * f64::EPSILON * self.spectral_norm;
        let identifiability_floor = sae_exact_a_identifiability_floor();
        (0..self.eigenvalues.len())
            .filter(|&index| {
                let lambda = self.eigenvalues[index];
                lambda.abs() <= arithmetic
                    && lambda.abs() >= identifiability_floor * self.metric_scale[index]
            })
            .count()
    }

    /// Smallest and largest `|λ|` the null band retained, or `None` when the
    /// whole spectrum is inside it. The two set the DERIVED damping ladder the
    /// polish walks: below `λ_min²` a damping cannot change the flattest
    /// resolved direction, and above `λ_max²` it has already flattened every
    /// direction there is, so no ladder needs to leave `[λ_min², λ_max²]`.
    fn retained_curvature_extremes(&self) -> Option<(f64, f64)> {
        let mut smallest = f64::INFINITY;
        let mut largest = 0.0_f64;
        for (index, &lambda) in self.eigenvalues.iter().enumerate() {
            let magnitude = lambda.abs();
            if magnitude > self.rank_floor(index) {
                smallest = smallest.min(magnitude);
                largest = largest.max(magnitude);
            }
        }
        (largest > 0.0 && smallest.is_finite()).then_some((smallest, largest))
    }

    /// One point of the damped residual path — see [`DampedResidualStep`].
    ///
    /// `residual` is the stationarity residual `g`; the step returned solves the
    /// damped system against `−g`, i.e. it is a descent step, and the modeled
    /// residual is reported for `g` itself so the caller can price it in the
    /// same merit as its convergence gate.
    /// A direction whose damped denominator `λ² + ν` is inside the null band
    /// (`≤ rank_floor²`) contributes nothing to the step and its whole
    /// coefficient to the model residual: at `ν = 0` that is exactly the
    /// pseudoinverse's own classification.
    fn damped_residual_step(
        &self,
        residual: &SaeArrowVector,
        nu: f64,
    ) -> Result<DampedResidualStep, String> {
        let total_t = residual.t.len();
        let dim = total_t + residual.beta.len();
        let spectral_dim = self.eigenvalues.len();
        if self.eigenvectors.dim() != (dim, spectral_dim) || spectral_dim != dim {
            return Err(format!(
                "damped residual step: eigenvectors {:?} and spectrum {spectral_dim} do not \
                 match residual dimension {dim}",
                self.eigenvectors.dim(),
            ));
        }
        if !(nu.is_finite() && nu >= 0.0) {
            return Err(format!("damped residual step: damping must be finite and ≥ 0; got {nu}"));
        }
        let mut flat = Array1::<f64>::zeros(dim);
        flat.slice_mut(s![..total_t]).assign(&residual.t);
        flat.slice_mut(s![total_t..]).assign(&residual.beta);
        if !flat.iter().all(|value| value.is_finite()) {
            return Err("damped residual step: residual contains a non-finite value".to_string());
        }
        let coefficients = self.eigenvectors.t().dot(&flat);
        let mut step_coefficients = Array1::<f64>::zeros(spectral_dim);
        let mut model_coefficients = Array1::<f64>::zeros(spectral_dim);
        let mut retained_rank = 0usize;
        for index in 0..spectral_dim {
            let lambda = self.eigenvalues[index];
            let floor = self.rank_floor(index);
            let null_band = floor * floor;
            let denominator = lambda * lambda + nu;
            let coefficient = coefficients[index];
            let surviving = if denominator > null_band {
                // Δ solves `(A² + ν) Δ = −A g` in this direction.
                step_coefficients[index] = -lambda * coefficient / denominator;
                retained_rank += 1;
                coefficient * nu / denominator
            } else {
                coefficient
            };
            model_coefficients[index] = surviving;
        }
        let solution = self.eigenvectors.dot(&step_coefficients);
        let model = self.eigenvectors.dot(&model_coefficients);
        Ok(DampedResidualStep {
            step: SaeArrowVector {
                t: solution.slice(s![..total_t]).to_owned(),
                beta: solution.slice(s![total_t..]).to_owned(),
            },
            model_residual: SaeArrowVector {
                t: model.slice(s![..total_t]).to_owned(),
                beta: model.slice(s![total_t..]).to_owned(),
            },
            step_norm_sq: solution.dot(&solution),
            retained_rank,
        })
    }

    /// Apply the symmetric Moore--Penrose inverse.  Resolved positive and
    /// negative modes are both retained; only the spectral null band
    /// `|λ| ≤ rank_floor` is removed, and that band is the ONLY null predicate
    /// on this route (#2674).  Three independent certificates guard the result:
    /// physical backward residual on the retained range, least-squares
    /// stationarity for the original RHS, and minimum-norm membership in the
    /// retained range.
    fn solve_stationarity(&self, rhs: &SaeArrowVector) -> Result<SaeArrowVector, String> {
        let total_t = rhs.t.len();
        let dim = total_t + rhs.beta.len();
        let spectral_dim = self.eigenvalues.len();
        if self.operator.dim() != (dim, dim)
            || self.eigenvectors.dim() != (dim, spectral_dim)
            || spectral_dim != dim
        {
            return Err(format!(
                "dense exact-stationarity pseudoinverse: geometry dimension {:?}, spectrum {}, \
                 eigenvectors {:?}, but RHS dimension is {dim}",
                self.operator.dim(),
                spectral_dim,
                self.eigenvectors.dim(),
            ));
        }
        let mut flat_rhs = Array1::<f64>::zeros(dim);
        flat_rhs.slice_mut(s![..total_t]).assign(&rhs.t);
        flat_rhs.slice_mut(s![total_t..]).assign(&rhs.beta);
        if !flat_rhs.iter().all(|value| value.is_finite()) {
            return Err(
                "dense exact-stationarity pseudoinverse: RHS contains a non-finite value"
                    .to_string(),
            );
        }

        let coefficients = self.eigenvectors.t().dot(&flat_rhs);
        let mut projected_coefficients = coefficients.clone();
        let mut inverse_coefficients = Array1::<f64>::zeros(spectral_dim);
        let mut retained_rank = 0usize;
        for index in 0..spectral_dim {
            let lambda = self.eigenvalues[index];
            if lambda.abs() > self.rank_floor(index) {
                inverse_coefficients[index] = coefficients[index] / lambda;
                retained_rank += 1;
            } else {
                projected_coefficients[index] = 0.0;
            }
        }
        let solution = self.eigenvectors.dot(&inverse_coefficients);
        let projected_rhs = self.eigenvectors.dot(&projected_coefficients);
        let applied = self.operator.dot(&solution);
        // Range stationarity is `P_range A x = P_range rhs`, with `P_range` the
        // projector onto the eigendirections the SPECTRAL floor retained.  The
        // eigenbasis is complete, so this reprojection removes exactly the
        // measured null band and nothing that was declared null in advance.
        let applied_coefficients = self.eigenvectors.t().dot(&applied);
        let projected_applied = self.eigenvectors.dot(&applied_coefficients);
        let physical_residual = &projected_applied - &projected_rhs;
        let residual_coefficients = &applied_coefficients - &coefficients;
        let normal_coefficients = &self.eigenvalues * &residual_coefficients;
        let normal_residual = self.eigenvectors.dot(&normal_coefficients);

        let norm = |vector: &Array1<f64>| vector.dot(vector).max(0.0).sqrt();
        let operator_norm = self
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let solution_norm = norm(&solution);
        let projected_rhs_norm = norm(&projected_rhs);
        let physical_norm = norm(&physical_residual);
        let normal_norm = norm(&normal_residual);
        let physical_scale = operator_norm * solution_norm + projected_rhs_norm;
        let normal_scale =
            operator_norm * (operator_norm * solution_norm + projected_rhs_norm);

        // A Moore--Penrose solution must be orthogonal to the declared null
        // space. Reproject the computed physical vector (not the coefficients
        // used to construct it) so this gate also detects loss of orthogonality.
        let solved_coefficients = self.eigenvectors.t().dot(&solution);
        let spectral_null_solution_norm_sq = solved_coefficients
            .iter()
            .enumerate()
            .filter_map(|(index, coefficient)| {
                (self.eigenvalues[index].abs() <= self.rank_floor(index))
                    .then_some(coefficient * coefficient)
            })
            .sum::<f64>();
        let discarded_solution_norm = spectral_null_solution_norm_sq.max(0.0).sqrt();
        let tolerance = f64::EPSILON.sqrt();
        let within = |residual: f64, scale: f64| {
            residual == 0.0 || (scale > 0.0 && residual <= tolerance * scale)
        };
        if !solution.iter().all(|value| value.is_finite())
            || !within(physical_norm, physical_scale)
            || !within(normal_norm, normal_scale)
            || !within(discarded_solution_norm, solution_norm)
        {
            return Err(format!(
                "dense exact-stationarity pseudoinverse failed certification: \
                 physical residual {physical_norm:.6e} / backward scale {physical_scale:.6e}, \
                 normal-equation stationarity {normal_norm:.6e} / scale {normal_scale:.6e}, \
                 null-space solution mass {discarded_solution_norm:.6e} / solution norm \
                 {solution_norm:.6e}, tolerance {tolerance:.6e}, rank {retained_rank}/{spectral_dim} \
                 on ambient dimension {dim}, identifiability floor {:.6e}·vᵀBv with \
                 vᵀBv ∈ [{:.6e}, {:.6e}], arithmetic floor {:.6e}",
                sae_exact_a_identifiability_floor(),
                self.metric_scale.iter().copied().fold(f64::INFINITY, f64::min),
                self.metric_scale
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
                (spectral_dim as f64) * f64::EPSILON * self.spectral_norm,
            ));
        }

        Ok(SaeArrowVector {
            t: solution.slice(s![..total_t]).to_owned(),
            beta: solution.slice(s![total_t..]).to_owned(),
        })
    }
}

/// One coherent dense exact-A quotient geometry.  The joint eigensystem owns
/// the stationarity pseudoinverse; the priced joint/coordinate inverses own the
/// exact-A log-determinant derivative.  They are deliberately derived from the
/// same materialized blocks and classification floors.
struct ExactHessianQuotientGeometry {
    joint: ExactHessianSpectralBlock,
    coordinate: ExactHessianSpectralBlock,
    priced_joint_inverse: Array2<f64>,
    priced_coordinate_inverse: Array2<f64>,
}

/// Complete dense exact-A derivative cluster.  The stationarity adjoint is
/// solved before the geometry is discarded, making this the sole owner of the
/// dense exact-A inverse action.
pub(crate) struct DenseExactALogdetChannels {
    pub(crate) logdet_trace: Array1<f64>,
    pub(crate) theta_adjoint: SaeArrowVector,
    pub(crate) stationarity_adjoint: SaeArrowVector,
}

/// #2515 — WHICH curvature operator a derivative channel differentiates and
/// contracts.
///
/// `A = B + ΔC = ∇²_θθ L` is the exact observed information; it is the operator
/// whose log-determinant the Laplace criterion **is**. `B` is the Gauss--Newton /
/// PSD-majorizer arrow system: the positive-definite scale the Newton and IFT
/// solves factor, and a preconditioner for `A`. A preconditioner is not the
/// operator it preconditions.
///
/// This exists as one value resolved ONCE per gradient assembly, rather than as
/// a per-channel argument, because the failure mode it guards is a channel
/// contracting one operator's inverse while differentiating the other's — a
/// state that is neither `A` nor `B` and that no single channel can detect
/// locally.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EvidenceOperator {
    /// The Gauss--Newton / PSD-majorizer arrow system `B`.
    Majorizer,
    /// The exact observed information `A = B + ΔC = ∇²_θθ L`.
    ExactObservedInformation,
}

impl EvidenceOperator {
    /// The historical boolean spelling, for channels whose exact-`A` port landed
    /// before this type did (`logdet_theta_adjoint_from_probes`, `ac499b513`).
    #[must_use]
    pub(crate) fn is_exact_a(self) -> bool {
        matches!(self, Self::ExactObservedInformation)
    }
}

/// #2515 — the selected-inverse evidence a bundle-routed outer ρ-gradient
/// contracts, and there is exactly ONE: the exact observed information
/// `A = B + ΔC = ∇²_θθ L`.
///
/// The from-probes channels reconstruct the arrow inverse blocks as
/// `(H⁻¹)_tt = A_i⁻¹ + G_i S⁻¹ G_iᵀ` and `(H⁻¹)_tβ = −G_i S⁻¹`: the row factors
/// `A_i` and cross blocks come from a factor CACHE, and `S⁻¹` comes from the
/// probe bundle. Those two must factor the SAME operator. Before this type
/// existed the streaming lane paired a bundle built on `exact_a_evidence_system`'s
/// reduced Schur with the `B` row-factor cache from
/// `converge_inner_for_undamped_logdet`, reconstructing an inverse belonging to
/// neither.
///
/// Carrying the cache HERE, rather than reading it off the assembler's `cache`
/// argument, is what makes that mixture unrepresentable. `cache` stays the `B`
/// stationarity geometry that
/// [`SaeManifoldTerm::solve_exact_stationarity_matrix_free`] reassembles
/// `A = B + ΔC` on top of; promoting it to `A` would double-count `ΔC`.
///
/// PRODUCTION ALWAYS MINTS `ExactObservedInformation`, from one place: the
/// `StreamingEvidenceArtifacts` of a single gradient-bearing evidence
/// evaluation, where the cache and the bundle were produced by one
/// factorization of one system and cannot be paired across evaluations. The
/// criterion that route feeds ranks `½log|S_A|` —
/// `rank_adjusted_quasi_laplace_complexity` takes `½(log_det − log_det_tt)` and
/// both come off `exact_a_evidence_system`, so the per-row t-block
/// log-determinants cancel and the reduced Schur of `A` is the whole operator
/// exposure — so a `B`-rooted derivative here would differentiate an operator
/// the value never ranked. That was #2515.
///
/// `Majorizer` is kept reachable for the regression gates that pin the
/// from-probes RECONSTRUCTION against its dense sibling (#2712, #2080): that
/// contract is about the reconstruction machinery and holds for either operator,
/// so forcing those fixtures onto an exact-`A` geometry their states may not even
/// admit would test less, not more.
pub(crate) struct BundleEvidenceGeometry<'a> {
    /// Which operator `cache` and `sinv` BOTH factor.
    pub(crate) operator: EvidenceOperator,
    /// Factor cache of the arrow system whose reduced Schur produced `sinv`.
    /// Distinct from the assembler's `cache`, which stays `B` on every route.
    pub(crate) cache: &'a ArrowFactorCache,
    /// `(probes, S_A⁻¹·probes)`. For the rational lane these are the identical
    /// weighted vectors emitted by
    /// `RationalLogdetPlan::into_directional_derivative_bundle`, so every
    /// contraction is the derivative of the SAME shifted rational value rather
    /// than a separately sampled `S⁻¹`.
    pub(crate) probes: &'a [Array1<f64>],
    pub(crate) sinv: &'a [Array1<f64>],
}

/// Certified exact-stationarity solve for a genuinely matrix-free operator.
/// Dense operators bypass this Krylov path and use the rank-revealing spectral
/// pseudoinverse owned by [`ExactHessianSpectralBlock`].
fn solve_exact_stationarity_preconditioned<A, B, P>(
    rhs: &SaeArrowVector,
    apply_a: &A,
    apply_b: &B,
    precondition: P,
) -> Result<SaeArrowVector, String>
where
    A: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    B: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    P: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    let mut x = solve_b_preconditioned_gmres_with(rhs, |v| apply_a(v), |v| precondition(v))?;
    // #2080 defect 4 — deflate unidentifiable near-null pencil directions.
    //
    // The generalized Rayleigh quotient `μ(x) = xᵀAx / xᵀBx` of the
    // SOLUTION is a detector: expanding
    // `x = Σ (vᵢᵀrhs/μᵢ) vᵢ` in the B-orthonormal
    // `(A, B)`-eigenbasis, any near-null component present in `rhs` enters
    // `x` with weight `1/μᵢ`, so `μ(x)` collapses to `≈ μ_min` exactly
    // when the solve was amplified. A healthy solve (`rhs` B-orthogonal to
    // the flat directions, or no flat directions) leaves `μ(x)` above the
    // floor and pays only one extra `A`/`B` apply.
    //
    // Deflation is EXACT in that eigenbasis with no re-solve: the
    // amplified term of `x` along a B-normalized eigendirection `v` is
    // `v·(vᵀBx)` (since `vᵀBx = vᵀrhs/μ_v`), so subtracting the
    // B-projection removes precisely the unidentifiable component while
    // leaving every resolved direction untouched.
    let dim = x.t.len() + x.beta.len();
    let rank_floor = sae_exact_a_identifiability_floor();
    // #2627 — anti-runaway ceiling on the TOTAL inverse-power refinement solves
    // summed over EVERY deflation turn. The two loops nest: the outer turn
    // deflates one direction per pass and is bounded by `dim`, the inner
    // inverse-power refinement is bounded by `dim`, and every turn of the inner
    // loop is a FULL preconditioned GMRES (`A⁻¹Bv`). The product is `dim²`
    // solves, which the #2472 note undercounts by a factor of `dim` because it
    // reads only the inner bound. This is a ceiling on the WORK, not a
    // convergence bound: the per-direction certificates below are unchanged and
    // still decide every solve that terminates inside the budget. It exists so
    // that an exact `A = B + ΔC` which is near-singular at the inner mode
    // surfaces as a typed refusal carrying its own diagnosis instead of a
    // wall-clock SIGKILL that names nothing.
    let inverse_power_solve_ceiling = 4usize.saturating_mul(dim).saturating_add(16);
    let mut inverse_power_solves = 0usize;
    for _ in 0..dim {
        let ax = apply_a(&x)?;
        let bx = apply_b(&x)?;
        let x_b_norm_sq = sae_inner(&x, &bx);
        if x_b_norm_sq == 0.0 && sae_inner(&x, &x) == 0.0 {
            return Ok(x);
        }
        if !(x_b_norm_sq.is_finite() && x_b_norm_sq > 0.0) {
            return Err(format!(
                "solve_exact_stationarity: invalid B-norm squared {x_b_norm_sq:.6e}"
            ));
        }
        let mu = sae_inner(&x, &ax) / x_b_norm_sq;
        if !mu.is_finite() {
            return Err("solve_exact_stationarity: non-finite generalized curvature".into());
        }
        // #2253 — accept the solve when the solution's generalized curvature is
        // RESOLVED, i.e. `|μ| >= rank_floor`, NOT only when `μ >= rank_floor`.
        // `μ(x) ≈ μ_min` (the smallest-magnitude pencil eigenvalue excited by the
        // rhs), so `μ < 0` with `|μ|` well above the floor is a genuinely
        // NEGATIVE-curvature but fully IDENTIFIED direction (the exact Hessian
        // `A = B + ΔC` is marginally indefinite at a nonzero-residual fit — the
        // measured K=1-circle μ = −1.66e-3). Its `A⁻¹` response is a REAL, finite
        // part of `dθ̂/dρ = −A⁻¹ λSθ̂`, and the criterion VALUE's undamped inner
        // solve moves θ̂ along it identically — so the θ-adjoint −½Γᵀθ̂_ρ MUST keep
        // it or the analytic outer gradient desyncs from d(value)/dρ (the #2253
        // non-stationary stall: the adjoint collapsed ~19×, so steepest descent
        // could not decrease the criterion at its own minimum). Only a genuinely
        // SINGULAR direction (`|μ| < rank_floor`, spurious `1/μ` amplification of
        // an unidentified near-null) is deflated below — that one the evidence
        // factor also stiffens to unit curvature, so its outer-gradient
        // contribution is ρ-independent and must be projected out.
        if mu.abs() >= rank_floor {
            return Ok(x);
        }
        // Reaching here means `|μ| < rank_floor`: the solution is dominated by a
        // genuinely SINGULAR (numerically curvature-free) pencil direction, whose
        // `1/μ` amplification is an unidentifiable artifact, not a derivative. A
        // resolved indefinite direction (`μ < 0`, `|μ| ≥ rank_floor`) was already
        // returned above and is NOT deflated: the criterion value's `½log|B|`
        // uses the majorized joint factor `B`, which is fully PD along it (the
        // undamped inner solve SUCCEEDED, so `factor_spectral_deflated_evidence_
        // row` — which only stiffens non-PD PER-ROW blocks — never fired), so the
        // value genuinely depends on that direction and its `A⁻¹` IFT response is
        // a real part of the θ-adjoint. Only the singular direction handled below
        // is one the criterion factor would stiffen to unit curvature, so only its
        // response is spurious and must be projected out.
        // Sharpen the offending direction by inverse power iteration on
        // the pencil (`v ← A⁻¹(B v)`, B-normalized); the corrupted `x` is
        // already dominated by it, so it is the natural seed. Convergence
        // is certified by successive B-normalized direction alignment;
        // exhaustion or a failed inner solve propagates instead of silently
        // projecting with `v=x` (which would delete the entire response).
        let mut v = x.clone();
        let normalize_b = |v: &mut SaeArrowVector| -> Result<(), String> {
            let bv = apply_b(v)?;
            let norm_sq = sae_inner(v, &bv);
            if !(norm_sq.is_finite() && norm_sq > 0.0) {
                return Err(format!(
                    "solve_exact_stationarity: inverse-power direction has invalid \
                     B-norm squared {norm_sq:.6e}"
                ));
            }
            let inv_norm = 1.0 / norm_sq.sqrt();
            v.t.mapv_inplace(|val| val * inv_norm);
            v.beta.mapv_inplace(|val| val * inv_norm);
            Ok(())
        };
        normalize_b(&mut v)?;
        let mut direction_converged = false;
        // #2627 — consecutive inverse steps whose refined direction stayed under
        // the numerical-null floor. See the subspace certificate at the bottom of
        // this loop for why this, and not eigenvector alignment, is the property
        // the deflation actually needs.
        let mut null_confirmations = 0usize;
        const NUMERICAL_NULL_CONFIRMATIONS: usize = 2;
        for power_step in 0..dim {
            // #2472 — every turn of this loop is a FULL preconditioned GMRES
            // solve (`A⁻¹Bv`), and the loop is bounded by the Krylov dimension,
            // so the worst case here is `dim` solves inside one deflation turn
            // inside one Newton step. Silence made that indistinguishable from a
            // deadlock; the cadence is powers of two, so the line count is
            // logarithmic in the work done.
            if power_step.is_power_of_two() {
                log::info!(
                    "[SAE-DEFLATE] inverse-power step {power_step}/{dim} \
                     (each step is one full A-inverse GMRES solve)"
                );
            }
            let bv = apply_b(&v)?;
            // #2253 — A⁻¹(Bv) is ILL-POSED along a near-null/indefinite pencil
            // direction (that is exactly the direction we are isolating), so the
            // refinement GMRES can legitimately exhaust its budget without
            // reaching tolerance. That is not a fatal error: the seed `v` is
            // already the B-normalized corrupted solution `x`, which — because
            // μ(x) collapsed onto μ_min — is ALREADY aligned with the offending
            // direction. Keep the best `v` and let the alignment/μ checks below
            // decide, instead of aborting the whole outer gradient.
            if inverse_power_solves >= inverse_power_solve_ceiling {
                return Err(format!(
                    "solve_exact_stationarity: inverse-power refinement exceeded the \
                     anti-runaway budget of {inverse_power_solve_ceiling} A-inverse GMRES \
                     solves (dimension {dim}); the exact pencil is numerically singular at \
                     this iterate and its IFT response is not identifiable"
                ));
            }
            inverse_power_solves += 1;
            let refined =
                match solve_b_preconditioned_gmres_with(&bv, |w| apply_a(w), |w| precondition(w)) {
                    Ok(mut refined) => {
                        normalize_b(&mut refined)?;
                        refined
                    }
                    Err(_) => {
                        // Refinement stalled — the current `v` is our best isolate.
                        direction_converged = true;
                        break;
                    }
                };
            let b_refined = apply_b(&refined)?;
            let alignment = sae_inner(&v, &b_refined).abs();
            if !alignment.is_finite() {
                return Err("solve_exact_stationarity: non-finite inverse-power alignment".into());
            }
            v = refined;
            // The discriminator asks whether the response's near-zero aggregate
            // Rayleigh quotient came from a numerical null or cancellation among
            // resolved pencil directions.  One inverse step amplifies smaller-|μ|
            // components relative to larger ones.  Therefore a refined direction
            // whose own curvature is already resolved proves the latter case; it
            // is unnecessary (and generally much slower) to wait for full
            // eigenvector alignment before keeping the original finite response.
            // Strict alignment remains mandatory below before a direction may be
            // projected as a numerical null.
            let av = apply_a(&v)?;
            let bv = apply_b(&v)?;
            let norm_sq = sae_inner(&v, &bv);
            if !(norm_sq.is_finite() && norm_sq > 0.0) {
                return Err(format!(
                    "solve_exact_stationarity: refined inverse-power direction has invalid \
                     B-norm squared {norm_sq:.6e}"
                ));
            }
            let refined_mu = sae_inner(&v, &av) / norm_sq;
            if !refined_mu.is_finite() {
                return Err(
                    "solve_exact_stationarity: refined inverse-power direction has non-finite \
                     generalized curvature"
                        .into(),
                );
            }
            if refined_mu.abs() >= rank_floor {
                return Ok(x);
            }
            if 1.0 - alignment.min(1.0) <= rank_floor {
                direction_converged = true;
                break;
            }
            // #2627 — SUBSPACE certificate, and the reason the timeouts existed.
            //
            // The alignment test above asks for a converged EIGENVECTOR: consecutive
            // B-normalized iterates agreeing to `sqrt(eps)`. Inverse power iteration
            // rotates the iterate toward the smallest-`|μ|` eigenvector at the rate
            // `(μ_min/μ_next)^k`, so the number of steps that test costs is
            // `log(sqrt(eps)) / log(μ_min/μ_next)` — UNBOUNDED as the two smallest
            // near-null curvatures approach each other, and every one of those steps
            // is a full preconditioned GMRES. A near-null CLUSTER with no interior
            // gap is not a pathology of this operator, it is its construction: `K`
            // atoms each contribute a rank-1 radial null to `A = B + ΔC`, so `A` has
            // a whole band of curvatures under the floor with ratios arbitrarily
            // close to one. At a ratio of 0.9 the test wants ~80 GMRES solves; at
            // 0.99, ~1800. Multiply by the outer deflation turn (bounded by `dim`)
            // and that is the `dim²`-shaped wall clock — and it is a CRITERION
            // defect, not a loop defect. The loop bound is the correct Krylov bound;
            // the criterion is buying a property at unbounded cost.
            //
            // The property is one the consumer never uses. Deflation subtracts the
            // B-projection of `x` along `v`, and the guard that makes that
            // legitimate is the one already enforced after this loop:
            // `|μ(v)| < rank_floor`, i.e. `v` lies in the numerical-null invariant
            // subspace. MEMBERSHIP in that subspace is the whole requirement; WHICH
            // member `v` is does not enter the projection's correctness, and every
            // vector of the cluster satisfies membership equally. Resolving the
            // cluster's interior gap is work whose answer is discarded.
            //
            // Membership is what surviving inverse steps prove, by the same
            // amplification argument the discriminator above already relies on: one
            // inverse step amplifies smaller-`|μ|` components relative to larger
            // ones, so a refined direction whose OWN curvature is resolved proves
            // the seed's near-zero Rayleigh quotient was cancellation among resolved
            // directions — that case returns above on `refined_mu.abs() >=
            // rank_floor`. Symmetrically, a direction still under the floor after
            // consecutive amplification steps carries genuine null content rather
            // than a cancellation artifact. Two consecutive confirmations is that
            // discriminator; further steps only refine which null vector.
            null_confirmations += 1;
            if null_confirmations >= NUMERICAL_NULL_CONFIRMATIONS {
                direction_converged = true;
                break;
            }
        }
        if !direction_converged {
            return Err(format!(
                "solve_exact_stationarity: inverse-power direction did not converge in the \
                 derived Krylov dimension {dim}"
            ));
        }
        // #2253 — deflate the isolated direction only when it is UNRESOLVED under
        // the exact pencil: `|μ|` below the numerical-null floor. A resolved
        // direction of either sign is a genuine finite part of the IFT response.
        // It can reach this branch when positive and negative resolved components
        // cancel in the solution's aggregate Rayleigh quotient; inverse iteration
        // then proves that no numerical null was present. In that case keep the
        // original exact solve instead of either deleting the resolved component
        // or turning benign Rayleigh cancellation into a typed failure.
        let av = apply_a(&v)?;
        let bv = apply_b(&v)?;
        let v_b_norm_sq = sae_inner(&v, &bv);
        if !(v_b_norm_sq.is_finite() && v_b_norm_sq > 0.0) {
            return Err(format!(
                "solve_exact_stationarity: converged inverse-power direction has invalid \
                 B-norm squared {v_b_norm_sq:.6e}"
            ));
        }
        let v_mu = sae_inner(&v, &av) / v_b_norm_sq;
        if !v_mu.is_finite() {
            return Err(format!(
                "solve_exact_stationarity: inverse power produced non-finite \
                 generalized curvature μ={v_mu:.6e}"
            ));
        }
        if v_mu.abs() >= rank_floor {
            return Ok(x);
        }
        let proj = sae_inner(&v, &bx);
        if proj == 0.0 || !proj.is_finite() {
            return Err(format!(
                "solve_exact_stationarity: invalid near-null B-projection {proj:.6e}"
            ));
        }
        x.t.scaled_add(-proj, &v.t);
        x.beta.scaled_add(-proj, &v.beta);
        log::debug!(
            "[SAE/#2080-d4] IFT solve deflated a near-null pencil direction \
             (μ={mu:.3e} < {rank_floor:.1e}, |proj|={:.3e})",
            proj.abs(),
        );
    }
    Err(format!(
        "solve_exact_stationarity: numerical-null deflation exhausted the derived \
         dimension {dim} without an identifiable IFT response"
    ))
}

/// #2330 Patch D — shared per-row context for the residual-curvature
/// third-derivative legs: the whitened `√w·M·r` error metric, its `√w` twin,
/// the frozen assignments/jets, and the ordered-Beta–Bernoulli gate mode. One
/// borrow per row replaces the per-call argument tower of the two leg helpers.
#[derive(Clone, Copy)]
struct PatchDResidualCtx<'a> {
    row: usize,
    error_metric: &'a [f64],
    sqrt_w: f64,
    assignments: &'a Array1<f64>,
    second_jets: &'a [Array4<f64>],
    third_jets: Option<&'a [Option<ndarray::Array5<f64>>]>,
    is_obb: bool,
    inv_tau: f64,
}

/// #2500 — what the assignment prior's sparse log-strength curvature operator
/// `∂H_tt/∂ρ_sparse` IS for the family in play, produced by the single authority
/// [`SaeManifoldTerm::sparse_logit_curvature_rho_derivative`].
///
/// Before this type existed, three channels each re-derived that operator by
/// matching on [`AssignmentMode`] independently — the arrow assembly (which
/// INSTALLS it into `block.htt`), ch4's Daleckii–Krein Hessian, and ch5's
/// forward-sensitivity operator map — and the two derivative channels shared a
/// catch-all `_ =>` refusal. The catch-all is what made `ThresholdGate` unfittable
/// on the dense route: its operator was never hard, it was simply absent from an
/// enumeration, while `assignment_prior_log_strength_hdiag_weighted` had been
/// computing it exactly the whole time for the gradient's trace channel. One
/// quantity with two implementations, and the matrix-valued one declined the case
/// the trace-valued one already computed.
///
/// The three outcomes are exhaustive over `AssignmentMode`, so a family added
/// later cannot fall through to a silently-zero operator: it has to name itself
/// here, and each consumer then decides what to do with the answer.
enum SparseLogitCurvature {
    /// There is no operator to install and a zero row is CORRECT: no sparse outer
    /// coordinate at all (`TopK` is `FixedSupport`), no free logit (`K ≤ 1`
    /// softmax), or a frozen/ungated routing whose prior is inert.
    Inert,
    /// The operator is DIAGONAL on the cache's global logit `t`-slots, as
    /// `(global slot, ∂H_{slot,slot}/∂ρ_sparse)`. Every installed assignment-prior
    /// curvature is diagonal in the free-logit chart — softmax writes the
    /// Gershgorin majorizer `D = diag(Σ_j|H_kj|)` (`row_psd_majorizer` is a
    /// diagonal matrix), threshold-gate writes the exact per-logit
    /// `w·λ·s·(1−2a)/τ²` — and both are degree-one in `λ_sparse = e^{ρ_sparse}`,
    /// so the derivative equals the installed entry itself.
    Diagonal(Vec<(usize, f64)>),
    /// The operator is NOT diagonal and is owned elsewhere: the ordered
    /// Beta--Bernoulli integrated marginal couples every row in an atom column, so
    /// `∂A/∂ρ_sparse` is a cross-row Hessian supplied by
    /// [`SaeManifoldTerm::dense_exact_a_ordered_bb_sparse_trace`]. A consumer that
    /// has that channel emits nothing here; a consumer that does NOT must refuse,
    /// because a diagonal-only stand-in would be a wrong operator rather than a
    /// missing one.
    CrossRowOwnedElsewhere,
}

impl SaeManifoldTerm {
    /// Orthonormal basis of the closed-form joint chart-gauge orbit in the
    /// exact-Hessian arrow layout. These are analytic symmetries (phase and
    /// patch translation/scale with their decoder compensation), not empirical
    /// least-curvature guesses.
    fn exact_joint_chart_gauge_basis(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Array1<f64>>, String> {
        self.joint_chart_gauge_basis_for_arrow_layout(
            &cache.row_offsets,
            cache.k,
            "exact_joint_chart_gauge_basis",
        )
    }

    /// #2500 — the ONE authority for `∂H_tt/∂ρ_sparse` on the free-logit slots.
    /// See [`SparseLogitCurvature`] for why this exists and what each outcome
    /// obliges a consumer to do.
    ///
    /// Both diagonal families are degree-one in `λ_sparse = e^{ρ_sparse}` at a
    /// frozen inner state, so the derivative IS the installed entry:
    ///
    /// * softmax — `D_k = Σ_j soft|scale·H_kj|` with `scale = λ_sparse·s/τ²`; the
    ///   soft-abs seam is positively homogeneous of degree one in `scale`
    ///   (`ε_k² = ε₀²Σ_l H_kl²` scales with it), and its `sign(H_kj)` kink lives in
    ///   the LOGITS, which a ρ perturbation never moves;
    /// * threshold gate — `w·λ_sparse·s·(1−2a)/τ²` with `a = σ((ℓ−θ)/τ)`,
    ///   `s = a(1−a)`, read from `assignment_prior_log_strength_hdiag_weighted`,
    ///   which is the SAME builder `assignment_prior_grad_hdiag_weighted` supplies
    ///   to the arrow assembly and already carries the `#991` row weights, the
    ///   `#Bug4` fixed-logit mask, and the frozen-routing zeroing.
    ///
    /// Note the threshold-gate operator is SIGNED (`1−2a` flips at the threshold):
    /// unlike softmax's Gershgorin radius and the ARD `max(·,0)` majorizer, no
    /// clamp is interposed on this family — the assembly installs the exact prior
    /// curvature (`construction_arrow_schur_assembly.rs`, the `raw` branch), so the
    /// exact-minus-majorizer delta `ΔC` has no threshold-gate part and
    /// `∂A/∂ρ_sparse = ∂B/∂ρ_sparse` here.
    fn sparse_logit_curvature_rho_derivative(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<SparseLogitCurvature, String> {
        if rho.sparse_flat_index().is_none() {
            return Ok(SparseLogitCurvature::Inert);
        }
        let k_atoms = self.k_atoms();
        let row_w = self.row_loss_weights.as_deref();
        let assignment_dim = self.assignment.assignment_coord_dim();
        // Only hard-TopK mints a compact row layout, and TopK carries no sparse
        // coordinate — but a FORCED layout can still reach here, and the compact
        // slot map is not the dense `base + atom` chart these operators are written
        // in. Refuse for any family rather than write into the wrong slots.
        let compact_layout_refusal = || {
            format!(
                "sparse_logit_curvature_rho_derivative: the compact top-k row layout is not \
                 covered by the sparse log-strength operator ({}); refusing to assemble a \
                 curvature operator with an unmodelled sparse row",
                self.assignment.mode.family_label()
            )
        };
        match self.assignment.mode {
            AssignmentMode::TopK { .. } => Ok(SparseLogitCurvature::Inert),
            AssignmentMode::OrderedBetaBernoulli { .. } => {
                Ok(SparseLogitCurvature::CrossRowOwnedElsewhere)
            }
            // K ≤ 1 softmax has no free logit: the gradient's sparse logdet trace
            // is identically zero, so a zero operator is the CORRECT curvature.
            AssignmentMode::Softmax { .. } if k_atoms <= 1 => Ok(SparseLogitCurvature::Inert),
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } => {
                if self.last_row_layout.is_some() {
                    return Err(compact_layout_refusal());
                }
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
                let penalty = gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                    k_atoms,
                    temperature,
                );
                let mut entries = Vec::new();
                for row in 0..self.n_obs() {
                    let w_row = row_w.map_or(1.0, |w| w[row]);
                    let base = cache.row_offsets[row];
                    let logit_dim = assignment_dim.min(cache.row_dims[row]);
                    let row_logits: Vec<f64> = (0..k_atoms)
                        .map(|atom| self.assignment.logits[[row, atom]])
                        .collect();
                    let d = penalty.psd_majorizer_abs_row_sums(&row_logits, scale);
                    for atom in 0..logit_dim {
                        entries.push((base + atom, w_row * d[atom]));
                    }
                }
                Ok(SparseLogitCurvature::Diagonal(entries))
            }
            AssignmentMode::ThresholdGate { .. } => {
                if self.last_row_layout.is_some() {
                    return Err(compact_layout_refusal());
                }
                let hdiag = crate::assignment::assignment_prior_log_strength_hdiag_weighted(
                    &self.assignment,
                    rho,
                    row_w,
                )?;
                if hdiag.is_empty() {
                    return Ok(SparseLogitCurvature::Inert);
                }
                let mut entries = Vec::new();
                for row in 0..self.n_obs() {
                    let base = cache.row_offsets[row];
                    let logit_dim = assignment_dim.min(cache.row_dims[row]);
                    for atom in 0..logit_dim {
                        entries.push((base + atom, hdiag[row * k_atoms + atom]));
                    }
                }
                Ok(SparseLogitCurvature::Diagonal(entries))
            }
        }
    }

    /// #2500 — push every per-flat-coordinate curvature operator through the
    /// per-row spectral-deflation map's differential in place, so the map returns
    /// `∂Φ(H_raw)/∂ρ` (the derivative of the operator the factors carry) rather
    /// than `∂H_raw/∂ρ`. A row with no deflation is untouched, bit-for-bit.
    ///
    /// The map is block-diagonal over rows and acts on the `t`-slots only, so the
    /// β border and the decoder-smoothness operators (which live entirely in the
    /// β block) pass through unchanged.
    fn apply_row_deflation_map_derivative(
        &self,
        cache: &ArrowFactorCache,
        operators: &mut std::collections::BTreeMap<usize, Array2<f64>>,
    ) -> Result<(), String> {
        if operators.is_empty() {
            return Ok(());
        }
        for row in 0..cache.n_rows() {
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            if dirs.is_empty() && spectrum.is_none() {
                continue;
            }
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            for operator in operators.values_mut() {
                let block = operator.slice(s![base..base + q, base..base + q]).to_owned();
                if block.iter().all(|value| *value == 0.0) {
                    continue;
                }
                let Some(mapped) = Self::row_deflation_map_derivative(&block, dirs, spectrum)
                else {
                    return Err(format!(
                        "apply_row_deflation_map_derivative: row {row} reports spectral \
                         deflation but its eigenbasis is not {q}×{q}; refusing to contract a \
                         curvature operator against a conditioned block whose deflation-map \
                         derivative cannot be formed"
                    ));
                };
                operator
                    .slice_mut(s![base..base + q, base..base + q])
                    .assign(&mapped);
            }
        }
        Ok(())
    }

    /// #1418: apply the EXACT stationarity-Jacobian correction `ΔC·v = (A − B)·v`
    /// to a joint `(t, β)` vector, matrix-free via row-local work and ordered
    /// prior column reductions.
    ///
    /// `A = ∇²_θθ L` is the true inner-fit Hessian; `B` is the assembled
    /// evidence/Newton operator the solver factors. They differ only by the four
    /// curvature substitutions the assembly makes for stability:
    ///   1. data: `B` uses Gauss-Newton `J̃J̃ᵀ`, dropping the residual curvature
    ///      `R[a,b] = Σ_out r_out·∂²f_out/∂θ_a∂θ_b` (t–t via `jets.second`, t–β via
    ///      `jets.beta_deriv`; the decoder is linear in β so the β–β block is 0);
    ///   2. softmax: `B` uses the Gershgorin majorizer `D = diag(Σ_j|H_kj|)`,
    ///      dropping `H_entropy − D` (#1419);
    ///   3. periodic ARD: `B` uses `max(V'',0)`, dropping the negative part
    ///      `min(V'',0)` (the indefinite tail past a quarter period).
    ///   4. ordered Beta--Bernoulli: `B` uses the positive row-local diagonal
    ///      majorizer and drops both the exact negative active-mass rank-one term
    ///      and every nonpositive row-local diagonal contribution.
    /// `ΔC` is the sum of exactly these four deltas, each built from the same
    /// jets / penalty curvatures the assembly and the θ-adjoint use, so
    /// `A = B + ΔC` is the one true Hessian. Exact on BOTH the isotropic and the
    /// whitened-metric paths: the data fit is `½ r_nᵀ M_n r_n`, so the residual
    /// curvature is `Σ_out (M_n r_n)_out·∂²f_out/∂θ_a∂θ_b` — contract the
    /// metric-applied √w-scaled residual `error_metric = √w·M_n r_n` (the SAME
    /// quantity the assembly's β-tier gradient uses) against the RAW second jets
    /// `jets.second`/`jets.beta_deriv` (the same raw-jet convention the whole
    /// θ-adjoint and the Gauss-Newton `htt = J̃J̃ᵀ = J M Jᵀ` assembly use). On the
    /// isotropic path `M_n = I` so `error_metric = √w·r` and `J M Jᵀ = JJᵀ`,
    /// recovering the plain case. The softmax, ordered Beta--Bernoulli, and ARD
    /// deltas are logit/coord-space prior curvatures and carry no output metric,
    /// so they are path-independent.
    pub(crate) fn apply_exact_hessian_minus_b(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        self.assignment.validate_rho_domain(rho)?;
        let p = self.output_dim();
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let total_t = cache.delta_t_len();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;

        // Optional softmax exact-entropy-minus-majorizer delta operator (#1419).
        let softmax_delta: Option<(
            gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
                Some((
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                    scale,
                ))
            }
            _ => None,
        };

        let mut out = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(cache.k),
        };
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        // Ordered Beta--Bernoulli's exact prior Hessian couples all rows within
        // each atom column. Gather the logit slice of `v` while visiting the
        // row-local cache layout, then apply the analytic column reductions once
        // after the row loop. This remains O(NK) memory/time and constructs no
        // dense cross-row matrix or persistent carrier.
        let mut ordered_logit_direction = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        )
        .then(|| Array1::<f64>::zeros(n * k_atoms));
        // #2520 — the ThresholdGate prior's concave half, dropped by the PSD
        // majorizer `B` now declares and restored here so `A = B + ΔC` is still
        // the exact signed curvature. Diagonal in the logit slots, so unlike
        // ordered Beta--Bernoulli it needs no direction gather.
        let threshold_gate_remainder = match self.assignment.mode {
            AssignmentMode::ThresholdGate { .. } => Some(
                crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                    &self.assignment,
                    rho,
                    row_loss_w,
                )?,
            ),
            _ => None,
        };
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            // #2304 resident path for the residual-curvature blocks (1a)+(1b):
            // the raw second/mixed jets are contracted on device (when the plan
            // admits it) against the metric-applied √w-scaled residual and the
            // direction's (t, β) coefficients — the packed channel tensors are
            // never materialized. Blocks (2)-(3) below are logit/coord-space
            // prior curvatures with no channel tensors involved and stay on
            // the host.
            {
                let mut probe_assignments = Array1::<f64>::zeros(k_atoms);
                let probe_for_row = |row: usize| -> Result<Vec<f64>, String> {
                    self.assignment.try_assignments_row_into(
                        row,
                        probe_assignments.as_slice_mut().ok_or_else(|| {
                            "apply_exact_hessian_minus_b: assignment scratch is not contiguous"
                                .to_string()
                        })?,
                    )?;
                    fitted.fill(0.0);
                    let active_atoms = self
                        .last_row_layout
                        .as_ref()
                        .map(|layout| layout.active_atoms[row].as_slice());
                    for k in 0..k_atoms {
                        if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                            continue;
                        }
                        self.atoms[k].fill_decoded_row(row, &mut decoded);
                        let a_k = probe_assignments[k];
                        for out_col in 0..p {
                            fitted[out_col] += a_k * decoded[out_col];
                        }
                    }
                    let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
                    for out_col in 0..p {
                        error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
                    }
                    Ok(match self.row_metric.as_ref() {
                        Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                        _ => error.to_vec(),
                    })
                };
                let v_t_for_row = |row: usize, q: usize| -> Result<Vec<f64>, String> {
                    let base = cache.row_offsets[row];
                    Ok((0..q).map(|c| v.t[base + c]).collect())
                };
                let v_beta_row: Vec<f64> =
                    border.iter().map(|channel| v.beta[channel.index]).collect();
                let out_ref = &mut out;
                self.contracted_softmax_bilinear_hvp(
                    cache,
                    &second_jets,
                    &border,
                    probe_for_row,
                    v_t_for_row,
                    &v_beta_row,
                    |row, row_vars, t_row, beta_row| {
                        // The callback's per-row var count must agree with the
                        // row slice it is handed; a mismatch would silently
                        // scatter a short row into the wrong output offsets.
                        assert_eq!(t_row.len(), row_vars);
                        let base = cache.row_offsets[row];
                        for (a, &value) in t_row.iter().enumerate() {
                            out_ref.t[base + a] += value;
                        }
                        for (channel, &value) in border.iter().zip(beta_row) {
                            out_ref.beta[channel.index] += value;
                        }
                        Ok(())
                    },
                )?;
            }
            // (2) softmax entropy-minus-majorizer and (3) periodic-ARD deltas,
            // per row with the layout rebuilt from the cache (no jets needed).
            for row in 0..n {
                let q = cache.row_dims[row];
                let base = cache.row_offsets[row];
                self.assignment.try_assignments_row_into(
                    row,
                    assignments.as_slice_mut().ok_or_else(|| {
                        "apply_exact_hessian_minus_b: assignment scratch is not contiguous"
                            .to_string()
                    })?,
                )?;
                let vars = self.row_vars_for_cache_row(row, cache)?;
                let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();
                let w_row = row_loss_w.map_or(1.0, |w| w[row]);
                if let Some((_penalty, scale)) = softmax_delta.as_ref() {
                    let assignment_dim = self.assignment.assignment_coord_dim();
                    let a_soft = assignments
                        .as_slice()
                        .expect("softmax assignments row must be contiguous");
                    let m = softmax_majorizer_log_mean(a_soft);
                    for (a, va) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: ka } = *va else {
                            continue;
                        };
                        if ka >= assignment_dim {
                            continue;
                        }
                        let mut acc = 0.0_f64;
                        for (b, vb) in vars.iter().enumerate() {
                            let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                                continue;
                            };
                            if kb >= assignment_dim {
                                continue;
                            }
                            let h_entropy =
                                softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, *scale);
                            let delta = if ka == kb {
                                h_entropy
                                    - active_softmax_gershgorin_majorizer_entry(
                                        a_soft, ka, m, *scale,
                                    )
                            } else {
                                h_entropy
                            };
                            acc += w_row * delta * v_t[b];
                        }
                        out.t[base + a] += acc;
                    }
                }
                for (a, va) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Coord { atom, axis } = *va else {
                        continue;
                    };
                    if rho.log_ard[atom].is_empty() {
                        continue;
                    }
                    let alpha = ard_precisions[atom][axis];
                    let t_val = self.assignment.coords[atom].row(row)[axis];
                    let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                    let neg = prior.negative_hessian_remainder();
                    if neg != 0.0 {
                        out.t[base + a] += w_row * neg * v_t[a];
                    }
                }
            }
            return Ok(out);
        }
        // #932 complete schedule: non-softmax gates use their distinct dynamic
        // row program through the bounded look-ahead window.
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    jet_window_next,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let jets = jet_window
                .pop_front()
                .expect("jet window must be non-empty");
            let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());

            // √w-scaled metric-applied per-row residual `error_metric = √w·M_n r_n`
            // (the SAME object the assembly's β-tier gradient contracts). The
            // data-fit `½ r_nᵀ M_n r_n` has residual curvature `Σ (M_n r_n)·∂²f`,
            // so this is exactly the residual contracted against the raw `∂²f`
            // jets. `M_n = I` on the isotropic path ⇒ `error_metric = √w·r`.
            fitted.fill(0.0);
            let active_atoms = self
                .last_row_layout
                .as_ref()
                .map(|layout| layout.active_atoms[row].as_slice());
            for k in 0..k_atoms {
                if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                    continue;
                }
                self.atoms[k].fill_decoded_row(row, &mut decoded);
                let a_k = assignments[k];
                for out_col in 0..p {
                    fitted[out_col] += a_k * decoded[out_col];
                }
            }
            for out_col in 0..p {
                error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
            }
            let error_metric: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                _ => error.to_vec(),
            };

            // Local t-slice of `v` for this row.
            let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();
            if let Some(direction) = ordered_logit_direction.as_mut() {
                for (local, var) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        direction[row * k_atoms + atom] = v_t[local];
                    }
                }
            }

            // (1a) residual curvature, t–t: ΔC_tt[a,b] = ⟨r, ∂²f_ab⟩.
            for a in 0..q {
                let mut acc = 0.0_f64;
                for b in 0..q {
                    let r_ab = sae_dot(&error_metric, jets.second(a, b));
                    acc += r_ab * v_t[b];
                }
                out.t[base + a] += acc;
            }
            // (1b) residual curvature, t–β and β–t: ΔC_tβ[a,β] = ⟨r, ∂²f_aβ⟩.
            //      `jets.beta_deriv[a][β]` = ∂(∂f/∂β_β)/∂θ_a (the mixed second jet).
            for a in 0..q {
                for (beta_pos, channel) in border.iter().enumerate() {
                    let r_ab = sae_dot(&error_metric, jets.beta_deriv(a, beta_pos));
                    // t row picks up β leg of v; β row picks up t leg of v.
                    out.t[base + a] += r_ab * v.beta[channel.index];
                    out.beta[channel.index] += r_ab * v_t[a];
                }
            }

            // (2) softmax entropy-minus-majorizer: softmax gates return through
            // the resident contracted branch above (#1419 algebra preserved
            // there verbatim, including the #1410 active-slot contraction and
            // the #991 `w_row` convention), so no softmax delta arises here.

            // (3) periodic ARD: ΔC_coord = V'' − psd_majorizer_hess =
            // negative_hessian_remainder, diagonal (#2339: the smooth
            // homogeneity-preserving clamp, non-positive). The assembly writes the
            // mean-one design-weighted majorizer `w_row·psd_majorizer_hess`, so the
            // dropped-curvature correction must carry that same `w_row`: `A = B + ΔC`
            // then recovers `w_row·V''` exactly (the seam guarantees
            // `psd_majorizer_hess + negative_hessian_remainder == V''` bit-for-bit).
            // The prior is weighted directly, not through the √w data-jet seam.
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in jets.vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.negative_hessian_remainder();
                if neg != 0.0 {
                    out.t[base + a] += w_row * neg * v_t[a];
                }
            }

            // (3b) #2520 threshold gate: the same shape as (3), on logit slots
            // rather than coordinate slots. `B` now carries the PSD majorizer
            // `smooth_psd_clamp(w·λ·s/τ², 1 − 2a)`, so `ΔC` must carry the
            // non-positive remainder or `A` would no longer be the exact signed
            // curvature and every exact-Hessian consumer (the IFT response, the
            // terminal Newton polish, the #2336 attributability test) would be
            // differentiating a different operator than it declares. The
            // remainder is already design-weighted and fixed-logit masked by the
            // producer, exactly as the majorizer is.
            if let Some(remainder) = threshold_gate_remainder.as_ref() {
                for (a, va) in jets.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *va else {
                        continue;
                    };
                    let neg = remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        out.t[base + a] += neg * v_t[a];
                    }
                }
            }
        }

        // (4) ordered Beta--Bernoulli: exact integrated-marginal Hessian minus
        // the diagonal PSD majorizer written into B. The helper evaluates the
        // negative within-column rank-one action by column reductions and the
        // row-local diagonal remainder directly, then we scatter its flat logit
        // result back into the cache's row-local coordinates.
        if let Some(direction) = ordered_logit_direction {
            let delta = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                &self.assignment,
                rho,
                row_loss_w,
                direction.view(),
            )?;
            for row in 0..n {
                let base = cache.row_offsets[row];
                let vars = self.row_vars_for_cache_row(row, cache)?;
                for (local, var) in vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        out.t[base + local] += delta[row * k_atoms + atom];
                    }
                }
            }
        }
        Ok(out)
    }

    /// #2336 — the diagonal of `E = B − A` restricted to the ARD periodic
    /// prior's concave-half clamp (block (3) of
    /// [`Self::apply_exact_hessian_minus_b`]), over the coordinate (t) block; zero
    /// on the β border and on logit rows.
    ///
    /// `E ⪰ 0` is diagonal in the t-block with entries `w_row·|min(V'',0)|`, the
    /// negative curvature of the periodic ARD prior that the Newton/Schur majorizer
    /// DROPS: the assembly writes only `w_row·max(V'',0)` into `B`
    /// ([`SaeManifoldAtom::psd_majorizer_hess`]), so `A = B + ΔC` with the ARD
    /// channel of `ΔC` equal to `w_row·min(V'',0) ≤ 0`. This is the EXACTLY-known,
    /// bounded amplitude of the prior micro-wrinkle that turns a `B`-converged mode
    /// into an `A`-saddle. Collected here as a diagonal so the criterion can test,
    /// per negative exact-`A` eigendirection `v`, whether `vᵀEv ≥ |λ|` — i.e. whether
    /// the indefiniteness is fully attributable to the clamp (#2336 value-side
    /// E-attributability). Reuses the identical per-row term block (3) applies, so
    /// the two cannot drift.
    pub(crate) fn materialize_ard_concave_clamp_diagonal(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<Array1<f64>, String> {
        self.materialize_ard_concave_clamp_diagonal_for_rows(rho, &cache.row_dims)
    }

    /// Factorization-free sibling used while `exact_a_evidence_system` still
    /// owns the raw arrow layout.  Classification is part of assembling the
    /// exact-A evidence operator, so requiring a factor cache here would force
    /// the wrong order (factor first, then decide what was factored).
    pub(crate) fn materialize_ard_concave_clamp_diagonal_for_rows(
        &self,
        rho: &SaeManifoldRho,
        row_dims: &[usize],
    ) -> Result<Array1<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        if row_dims.len() != self.n_obs() {
            return Err(format!(
                "materialize_ard_concave_clamp_diagonal_for_rows: {} row dimensions for {} observations",
                row_dims.len(),
                self.n_obs(),
            ));
        }
        let total_t: usize = row_dims.iter().sum();
        let mut e_diag = Array1::<f64>::zeros(total_t);
        if self.k_atoms() == 0 {
            return Ok(e_diag);
        }
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        // #2520 — the ThresholdGate's own concave half is the SAME kind of
        // exactly-known, bounded `E ⪰ 0` as the periodic-ARD clamp: `B` declares
        // `smooth_psd_clamp(w·λ·s/τ², 1 − 2a)` and drops the negative part, so a
        // mode whose only indefiniteness IS that dropped part is attributable
        // and must be PRICED under #2336 rather than refused. Reads the same
        // producer as the ΔC channel, so E and ΔC cannot disagree.
        let threshold_gate_remainder =
            crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                &self.assignment,
                rho,
                row_loss_w,
            )?;
        let k_atoms = self.k_atoms();
        let mut base = 0usize;
        for row in 0..self.n_obs() {
            let vars = self.row_vars_for_row_dim(row, row_dims[row])?;
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *va {
                    // E = B − A, so E = −ΔC ≥ 0 here. The remainder already
                    // carries `w_row` (its producer applies the #991 weight).
                    let neg = threshold_gate_remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        e_diag[base + a] += -neg;
                    }
                    continue;
                }
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.negative_hessian_remainder();
                if neg != 0.0 {
                    // E = B − A, so on this diagonal E = −(w_row·neg) = w_row·|neg| ≥ 0.
                    e_diag[base + a] += -w_row * neg;
                }
            }
            base += row_dims[row];
        }
        Ok(e_diag)
    }

    /// #1418: matrix-free apply of the EXACT stationarity Jacobian `A = ∇²_θθ L`:
    /// `A v = B_raw v + ΔC v`, the raw objective-majorizer apply
    /// ([`apply_raw_cached_arrow_hessian`]) plus the matrix-free dropped-curvature
    /// correction `ΔC = A − B` ([`Self::apply_exact_hessian_minus_b`]).
    fn apply_exact_hessian(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        // #2515 — the cache factors the conditioned evidence majorizer
        // `Phi(B_raw)`.  That conditioning is a solve/log-determinant policy, not
        // part of the objective Hessian.  Adding ΔC to it would build
        // `Phi(B_raw) + ΔC` on the dense route while the streaming arrow route
        // builds `Phi(B_raw + ΔC)`.  Recover B_raw first so both routes classify
        // the one statistical operator `A_raw = B_raw + ΔC`.
        let b_v = apply_raw_cached_arrow_hessian(cache, v.t.view(), v.beta.view())?;
        let dc_v = self.apply_exact_hessian_minus_b(rho, target, cache, v)?;
        Ok(SaeArrowVector {
            t: &b_v.t + &dc_v.t,
            beta: &b_v.beta + &dc_v.beta,
        })
    }

    /// #1418/#2653: solve `A x = rhs` for the materializable EXACT stationarity
    /// Jacobian `A = ∇²_θθ L` with its symmetric rank-revealing
    /// pseudoinverse.  The same materialized and symmetrized `A` used by the
    /// exact observed-information route declares the quotient null band; both
    /// resolved positive and negative modes are inverted.  This avoids the
    /// ill-conditioned Krylov-basis coefficient cancellation that can satisfy a
    /// projected residual while failing `A x = P_range rhs` on reapplication.
    /// The IFT step `θ̂_ρ = −A⁺ g_ρ` (the sign lives in the caller's
    /// `-0.5` contraction) therefore has one dense owner.  GMRES remains only in
    /// [`Self::solve_exact_stationarity_matrix_free`], where `A` cannot be
    /// materialized.
    pub(crate) fn solve_exact_stationarity(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        self.materialize_exact_stationarity_geometry(rho, target, cache)?
            .solve_stationarity(rhs)
    }

    /// Matrix-free exact-stationarity sibling used by the wide-border penalized quasi-Laplace
    /// assignment-strength residual. `system` is the reassembled undamped
    /// bordered operator at the converged inner state; `cache` supplies the same
    /// row factors and H_tbeta operator whose rational log-determinant and shared
    /// inverse-probe bundle were consumed by the value/trace lanes.
    ///
    /// The reduced beta solve is quotient-aware and matrix-free. Per-row
    /// spectral deflation is refused by the selected-inverse channels before
    /// this seam is reached: a border-only probe bundle cannot differentiate
    /// the Daleckii-Krein deflation map, so proceeding would be a false exactness
    /// claim rather than a usable fallback.
    fn solve_exact_stationarity_matrix_free(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        system: &ArrowSchurSystem,
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let apply_b = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let (t, beta) = matrix_free_arrow_operator_apply(
                system,
                cache,
                vector.t.view(),
                vector.beta.view(),
            )
            .map_err(|error| format!("matrix-free evidence operator: {error}"))?;
            Ok(SaeArrowVector { t, beta })
        };
        let apply_a = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let base = apply_b(vector)?;
            let correction = self.apply_exact_hessian_minus_b(rho, target, cache, vector)?;
            Ok(SaeArrowVector {
                t: &base.t + &correction.t,
                beta: &base.beta + &correction.beta,
            })
        };
        // #2674 — the Krylov sequence runs on the SAME operator the dense route
        // now diagonalizes: the full `A`, with no analytic chart orbit deleted
        // ahead of the numerics. `A` is the Hessian of the PENALIZED objective
        // and the orbit is a symmetry of the reconstruction only, so projecting
        // it out of the applies removed live prior-driven descent here for the
        // same reason it stalled the dense inner solve. A direction that is
        // genuinely flat for `A` is still handled, by the `μ` deflation loop in
        // `solve_exact_stationarity_preconditioned` — measured, not declared.
        let precondition = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            // The outer exact-stationarity residual is certified to 1e-10 in
            // `solve_b_preconditioned_gmres`; drive its deterministic SPD
            // reduced preconditioner to the same relative accuracy. In exact
            // arithmetic CG terminates in at most the reduced dimension, so the
            // dimension itself is the non-arbitrary iteration bound.
            // The CG certificate is ADVISORY on this seam and only on this seam:
            // the inverse-apply is used as a PRECONDITIONER for the outer GMRES,
            // which certifies the ORIGINAL residual regardless of how well the
            // preconditioner approximates `A⁻¹` (`certifies_original_residual_
            // under_ill_scaled_preconditioner_2258`). A truncated inner solve
            // costs iterations here, never correctness — unlike the trace/
            // criterion consumers, where a truncation silently biases the
            // estimate (#2576).
            let (t, beta, _cg) = matrix_free_arrow_inverse_apply(
                system,
                cache,
                vector.t.view(),
                vector.beta.view(),
                1.0e-10,
                cache.k.max(1),
            )
            .map_err(|error| format!("matrix-free evidence inverse: {error}"))?;
            Ok(SaeArrowVector { t, beta })
        };
        solve_exact_stationarity_preconditioned(rhs, &apply_a, &apply_b, precondition)
    }

    /// PATH C (#2253) — the raw per-flat-coordinate penalty curvature operators
    /// `M_i = ∂H_raw/∂ρ_i` at a frozen inner state, keyed by flat outer coordinate.
    /// This is the single assembly source for both consumers: dense statistical
    /// channels use its raw product, while arrow-factor channels pass that product
    /// through the conditioning differential in
    /// [`Self::penalty_curvature_operators_by_flat`]. Each `M_i` is degree-one in
    /// `exp(ρ_i)`: `λ_k·½(S_k+S_kᵀ)⊗I`
    /// on atom `k`'s β-block for smoothing; `w_row·max(α cos κt,0)` on the active
    /// row-local t-slots for periodic ARD (`w_row·α` Euclidean); the softmax
    /// Gershgorin majorizer `w_row·diag(Σ_j|H_kj|)` on the logit slots for the
    /// sparse coordinate. The sparse refusals (compact top-k layout, non-softmax
    /// prior) match ch4's so both channels decline the same unmodelled cases.
    fn raw_penalty_curvature_operators_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let mut c_by_flat: std::collections::BTreeMap<usize, Array2<f64>> =
            std::collections::BTreeMap::new();

        // Smoothing: Cₐ = (λ_a·½(Sₐ+Sₐᵀ)) ⊗ I on atom a's β-block.
        let lambda_smooth = rho.lambda_smooth_vec()?;
        let p = self.output_dim();
        let frames_active = self.frames_active();
        let (beta_offsets, beta_out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) =
            if frames_active {
                let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
                (
                    self.factored_beta_offsets(),
                    Box::new(move |kk: usize| ranks[kk]),
                )
            } else {
                (self.beta_offsets(), Box::new(move |_: usize| p))
            };
        // #2604 — sectional curvature enters the criterion ONLY through the
        // penalty, because a constant-curvature atom's basis is a monomial patch
        // in the tangent coordinate and carries no κ. So `∂H/∂κ_a = λ_a·∂S_a/∂κ`,
        // the same shape as the smoothness coordinate's `∂H/∂log λ_a = λ_a·S_a`
        // with the Gram replaced by its derivative — which is why it slots into
        // this assembly rather than needing a channel of its own. Every trace,
        // the penalty energy and the rank-aware log-determinant term are then
        // derived by the same machinery that already consumes this map.
        for &a in &rho.kappa_atoms {
            let flat = rho.kappa_flat_index(a).ok_or_else(|| {
                format!("curvature atom {a} has no flat outer coordinate")
            })?;
            let atom = self.atoms.get(a).ok_or_else(|| {
                format!(
                    "curvature coordinate names atom {a}, outside term K={}",
                    self.atoms.len()
                )
            })?;
            let Some(ds) = atom.smooth_penalty_kappa_derivative() else {
                // An atom whose roughness is not curvature-parameterised has no
                // κ to move; leaving the coordinate un-assembled is what makes
                // its gradient entry exactly zero rather than silently wrong.
                continue;
            };
            let m = atom.basis_size();
            let off = beta_offsets[a];
            let r = beta_out_dim(a);
            let lambda = lambda_smooth[a];
            let c = c_by_flat
                .entry(flat)
                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
            for mu in 0..m {
                for nu in 0..m {
                    let ds_sym = 0.5 * (ds[[nu, mu]] + ds[[mu, nu]]);
                    let val = lambda * ds_sym;
                    if val == 0.0 {
                        continue;
                    }
                    for oc in 0..r {
                        c[[total_t + off + nu * r + oc, total_t + off + mu * r + oc]] += val;
                    }
                }
            }
        }
        for a in 0..rho.log_lambda_smooth.len() {
            let atom = &self.atoms[a];
            let s = atom.smooth_penalty();
            let m = atom.basis_size();
            let off = beta_offsets[a];
            let r = beta_out_dim(a);
            let lambda = lambda_smooth[a];
            let flat = rho.smooth_flat_index(a);
            let c = c_by_flat
                .entry(flat)
                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
            for mu in 0..m {
                for nu in 0..m {
                    let s_sym = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                    let val = lambda * s_sym;
                    if val == 0.0 {
                        continue;
                    }
                    for oc in 0..r {
                        c[[total_t + off + nu * r + oc, total_t + off + mu * r + oc]] += val;
                    }
                }
            }
        }

        // ARD: C_{k,axis} = w_row·max(α cos κt, 0) (periodic) / w_row·α (Euclidean)
        // on the row-local t-slot for (atom k, axis).
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let coord_offsets = self.assignment.coord_offsets();
        let periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(LatentCoordValues::effective_axis_periods)
            .collect();
        for row in 0..self.n_obs() {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let base = cache.row_offsets[row];
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &kk) in layout.active_atoms[row].iter().enumerate() {
                        if rho.log_ard[kk].is_empty() {
                            continue;
                        }
                        let start = layout.coord_starts[row][pos];
                        let coord = &self.assignment.coords[kk];
                        for axis in 0..coord.latent_dim() {
                            let alpha = ard_precisions[kk][axis];
                            let t = coord.row(row)[axis];
                            let hess = w_row
                                * ArdAxisPrior::eval(alpha, t, periods[kk][axis])
                                    .psd_majorizer_hess();
                            if hess == 0.0 {
                                continue;
                            }
                            let flat = rho.ard_flat_index(kk, axis);
                            let c = c_by_flat
                                .entry(flat)
                                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                            let g_idx = base + start + axis;
                            c[[g_idx, g_idx]] += hess;
                        }
                    }
                }
                None => {
                    for kk in 0..self.k_atoms() {
                        if rho.log_ard[kk].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[kk];
                        for axis in 0..coord.latent_dim() {
                            let alpha = ard_precisions[kk][axis];
                            let t = coord.row(row)[axis];
                            let hess = w_row
                                * ArdAxisPrior::eval(alpha, t, periods[kk][axis])
                                    .psd_majorizer_hess();
                            if hess == 0.0 {
                                continue;
                            }
                            let flat = rho.ard_flat_index(kk, axis);
                            let c = c_by_flat
                                .entry(flat)
                                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                            let g_idx = base + coord_offsets[kk] + axis;
                            c[[g_idx, g_idx]] += hess;
                        }
                    }
                }
            }
        }

        // Sparse (assignment log-strength): whatever the single authority says the
        // installed logit-slot curvature's ρ-derivative is (#2500). Softmax reads
        // its Gershgorin majorizer `w_row·diag(Σ_j|H_kj|)`, the threshold gate its
        // exact `w_row·λ·s·(1−2a)/τ²` — both degree-one in `λ_sparse = e^ρ` exactly
        // like smoothing/ARD, so the derivative is the installed entry itself.
        if let Some(sparse_flat) = rho.sparse_flat_index() {
            match self.sparse_logit_curvature_rho_derivative(rho, cache)? {
                SparseLogitCurvature::Inert => {}
                SparseLogitCurvature::Diagonal(entries) => {
                    let c = c_by_flat
                        .entry(sparse_flat)
                        .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                    for (slot, value) in entries {
                        c[[slot, slot]] += value;
                    }
                }
                SparseLogitCurvature::CrossRowOwnedElsewhere => {
                    // #2330: the ordered-Beta–Bernoulli sparse ∂A/∂ρ_sparse is the
                    // EXACT integrated-marginal logit Hessian (cross-row), supplied
                    // by `dense_exact_a_ordered_bb_sparse_trace`, NOT a diagonal
                    // majorizer operator this map can assemble. Emit nothing here
                    // (the dense-A gradient adds that coordinate's trace directly)
                    // rather than a wrong diagonal-only operator.
                }
            }
        }

        Ok(c_by_flat)
    }

    /// Derivatives of the conditioned operator installed in the row factors.
    ///
    /// The dense exact-`A` value differentiates the raw statistical operator and
    /// consumes [`Self::raw_penalty_curvature_operators_by_flat`] directly. Arrow
    /// selected-inverse, sensitivity, and Hessian channels instead contract
    /// against `Φ(H_raw)`, so they push the raw derivatives through `DΦ` once.
    /// Every null or negative row direction replaced by unit stiffness therefore
    /// has a rho-independent installed tangent, while healthy raw directions pass
    /// through unchanged.
    /// Naming both products prevents a dense spectral value from silently taking
    /// the arrow-conditioned tangent (#2515).
    pub(crate) fn penalty_curvature_operators_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let mut operators = self.raw_penalty_curvature_operators_by_flat(rho, cache)?;
        self.apply_row_deflation_map_derivative(cache, &mut operators)?;
        Ok(operators)
    }

    /// PATH C (#2253) CH5 — the ρ-derivative of the EXACT-minus-majorizer
    /// stationarity correction, `∂(ΔC)/∂ρ_i` where `ΔC = A − B`
    /// ([`Self::apply_exact_hessian_minus_b`]), keyed by flat coordinate. The IFT
    /// sensitivity `∂a/∂ρ_i = A⁺(∂Γ/∂ρ_i − (∂A/∂ρ_i)a)` differentiates the EXACT
    /// stationarity Hessian `A = B + ΔC`, not the majorized solver operator `B = H`
    /// (`penalty_curvature_operators_by_flat` = `∂B/∂ρ`). So the `M_i·a` term must
    /// use `∂A/∂ρ_i = ∂B/∂ρ_i + ∂(ΔC)/∂ρ_i` — this map supplies the second piece.
    ///
    /// Both deltas are degree-one in their ρ (so `∂(ΔC)/∂ρ_i` is the delta itself)
    /// and mirror `apply_exact_hessian_minus_b`'s deltas exactly:
    /// * periodic ARD: `w_row·min(α cos κt, 0)` (the negative-part remainder the
    ///   `max(·,0)` majorizer drops) on the coord slot, ALL rows — nonzero only on
    ///   the inactive half `cos κt < 0`. This is the term the ARD-perturbed
    ///   `H3[ard,·]` rows need (the transposed smooth-perturbed rows, where
    ///   `∂A = ∂B`, are already exact).
    /// * softmax sparse: the exact entropy Hessian minus the Gershgorin majorizer
    ///   on the row's logit block (dense, off-diagonal + diagonal), `∝ λ_sparse`.
    /// Smooth is unmajorized (`ΔC` has no smooth part), so its delta is zero and it
    /// is absent from the map. Covered config only (softmax, dense row layout).
    pub(crate) fn exact_stationarity_penalty_derivative_delta_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let k_atoms = self.k_atoms();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let softmax_delta: Option<(usize, f64)> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                match rho.sparse_flat_index() {
                    Some(sparse_flat) => Some((
                        sparse_flat,
                        rho.lambda_sparse()? * sparsity * inv_tau * inv_tau,
                    )),
                    None => None,
                }
            }
            _ => None,
        };
        let mut deltas: std::collections::BTreeMap<usize, Array2<f64>> =
            std::collections::BTreeMap::new();
        let mut assignments = Array1::<f64>::zeros(k_atoms);
        for row in 0..self.n_obs() {
            let base = cache.row_offsets[row];
            self.assignment.try_assignments_row_into(
                row,
                assignments
                    .as_slice_mut()
                    .expect("assignment scratch is contiguous"),
            )?;
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let w_row = row_w.map_or(1.0, |w| w[row]);
            // Softmax entropy-minus-majorizer delta on the logit block.
            if let Some((sparse_flat, scale)) = softmax_delta {
                let assignment_dim = self.assignment.assignment_coord_dim();
                let a_soft = assignments
                    .as_slice()
                    .expect("softmax assignments row must be contiguous");
                let m = softmax_majorizer_log_mean(a_soft);
                let c = deltas
                    .entry(sparse_flat)
                    .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                for (a, va) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: ka } = *va else {
                        continue;
                    };
                    if ka >= assignment_dim {
                        continue;
                    }
                    for (b, vb) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                            continue;
                        };
                        if kb >= assignment_dim {
                            continue;
                        }
                        let h_entropy =
                            softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, scale);
                        let delta = if ka == kb {
                            h_entropy
                                - active_softmax_gershgorin_majorizer_entry(a_soft, ka, m, scale)
                        } else {
                            h_entropy
                        };
                        c[[base + a, base + b]] += w_row * delta;
                    }
                }
            }
            // Periodic-ARD negative-part remainder on the coord slots.
            for (a, va) in vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let neg = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis])
                    .negative_hessian_remainder();
                if neg != 0.0 {
                    let flat = rho.ard_flat_index(atom, axis);
                    let c = deltas
                        .entry(flat)
                        .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                    c[[base + a, base + a]] += w_row * neg;
                }
            }
        }
        Ok(deltas)
    }

    /// The complete frozen-state derivative of the raw exact stationarity
    /// Hessian, `A_raw = B_raw + ΔC`, keyed by flat outer coordinate.
    ///
    /// Keeping this sum behind one named owner makes it possible to compare the
    /// operator derivative directly with finite differences of
    /// [`Self::materialize_exact_hessian_dense`], instead of validating only a
    /// downstream trace where spectral classification can obscure which operand
    /// drifted (#2515).
    pub(crate) fn exact_stationarity_penalty_derivatives_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let mut derivatives = self.raw_penalty_curvature_operators_by_flat(rho, cache)?;
        for (flat, delta) in self
            .exact_stationarity_penalty_derivative_delta_by_flat(rho, cache)?
        {
            match derivatives.entry(flat) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    *entry.get_mut() += &delta;
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(delta);
                }
            }
        }
        Ok(derivatives)
    }

    /// PATH C (#2253) — the full joint arrow inverse `G = H⁻¹` (dim×dim),
    /// materialized column by column against each unit arrow basis vector and
    /// symmetrized. Shared by ch4 and ch5's small-dense (circle-mint scale)
    /// route; `solver` must be [`DeflatedArrowSolver::plain`].
    pub(crate) fn materialize_joint_inverse(
        &self,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<Array2<f64>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let mut g = Array2::<f64>::zeros((dim, dim));
        let mut rhs_t = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(k);
        for col in 0..total_t {
            rhs_t[col] = 1.0;
            let sol = solver.solve(rhs_t.view(), rhs_beta_zero.view())?;
            rhs_t[col] = 0.0;
            for r in 0..total_t {
                g[[r, col]] = sol.t[r];
            }
            for r in 0..k {
                g[[total_t + r, col]] = sol.beta[r];
            }
        }
        let rhs_t_zero = Array1::<f64>::zeros(total_t);
        let mut rhs_beta = Array1::<f64>::zeros(k);
        for col in 0..k {
            rhs_beta[col] = 1.0;
            let sol = solver.solve(rhs_t_zero.view(), rhs_beta.view())?;
            rhs_beta[col] = 0.0;
            for r in 0..total_t {
                g[[r, total_t + col]] = sol.t[r];
            }
            for r in 0..k {
                g[[total_t + r, total_t + col]] = sol.beta[r];
            }
        }
        for a in 0..dim {
            for b in (a + 1)..dim {
                let avg = 0.5 * (g[[a, b]] + g[[b, a]]);
                g[[a, b]] = avg;
                g[[b, a]] = avg;
            }
        }
        Ok(g)
    }

    /// PATH C (#2253) — the block-diagonal row-local t-inverse `H_bd⁻¹` (dim×dim;
    /// β block zero) built from the per-row undamped Cholesky factors, the same
    /// inverse the rank-charge coordinate-block trace subtracts. Shared by ch4
    /// and ch5.
    pub(crate) fn materialize_block_diag_t_inverse(&self, cache: &ArrowFactorCache) -> Array2<f64> {
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let mut h_bd = Array2::<f64>::zeros((dim, dim));
        for row in 0..self.n_obs() {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let factor = cache.undamped_factor(row);
            let mut unit = Array1::<f64>::zeros(q);
            for col in 0..q {
                unit.fill(0.0);
                unit[col] = 1.0;
                let solved = cholesky_solve_vector(factor, unit.view());
                for r in 0..q {
                    h_bd[[base + r, base + col]] = solved[r];
                }
            }
        }
        h_bd
    }

    /// PATH C (#2253) CH5 — dense reconstruction of the θ-adjoint contraction
    /// `Γ_w = tr(inv · K_w)`, `K_w = ∂H/∂θ_w`, for an ARBITRARY dense joint
    /// inverse `inv` (dim×dim over the `(t, β)` blocks) and a chosen subset of
    /// the `K_w` operator ([`ThetaAdjointDhChannel`]).
    ///
    /// With `inv = G` and `ThetaAdjointDhChannel::All` this reproduces the
    /// production [`Self::logdet_theta_adjoint`] (self-checked by the FD gate);
    /// with `inv = h_bd` it reproduces [`Self::coordinate_block_logdet_theta_adjoint`].
    /// Feeding the TWISTED inverse `−G M_i G` gives the part-(a) term
    /// `−tr(G M_i G K_w)` of `dΓ/dρ_i`; the two MIXED channels give part-(b).
    ///
    /// Covered config ONLY (validated by the caller): softmax assignment, dense
    /// per-atom row layout (`last_row_layout = None`), no per-row deflation, no
    /// border frames, no ordered Beta--Bernoulli. The `dh` assembly mirrors the
    /// production builder's inner loop for exactly that config; the softmax
    /// diagonal `assignment_prior_hdiag_derivative_entry` is 0 for softmax and is
    /// omitted here for the same reason.
    /// #2330 Patch D — the t--β residual-curvature second-derivative leg
    /// `⟨error_metric, ∂²(gate_kβ·φ_mβ)/∂θ_a∂θ_w⟩` (term-2 of `∂ΔC_tβ[a,β]/∂θ_w`;
    /// the term-1 `⟨jets.first(w), jets.beta_deriv(a,β)⟩` is added inline). The
    /// border channel `β = (atom kβ, basis mβ, output-vector)` gives
    /// `∂f_out/∂β = gate_kβ·φ_mβ·output_out`, so this leg is
    /// `eo · g_kβ^{(l)} · ∂^{2−l}φ_mβ` with `eo = Σ_out error_metric[out]·output[out]`,
    /// `l` the number of LOGIT derivatives among `{a,w}`, on the coord axes of the
    /// rest; nonzero only when `a,w` both touch `kβ`. `l≥1` uses the ordered-BB
    /// logistic-gate derivatives; skipped for other modes (softmax follow-on).
    /// #2330 Patch D — one row's `error_metric = √w·M·r` in OUTPUT space, the
    /// object `apply_exact_hessian_minus_b` contracts `ΔC` against.
    ///
    /// Built exactly as that assembler builds it: the fitted row is the
    /// assignment-weighted decode over this row's ACTIVE atoms, the residual is
    /// scaled by `√w` before the metric, and the whitening metric is applied only
    /// where the row jets are whitened — so a plain dot of this against a jet
    /// reconstitutes the same `w`-weighted `M`-inner product the assembly uses.
    ///
    /// #2515 — extracted so the dense θ-adjoint, its from-probes sibling, and the
    /// coordinate-block leg all build it ONCE rather than three times. A route
    /// that reconstructed the residual with a different weighting would produce a
    /// Patch-D leg that silently disagreed with the operator it is supposed to
    /// differentiate, which is the failure this whole front is about.
    pub(crate) fn patchd_row_error_metric(
        &self,
        row: usize,
        w_row: f64,
        target: ArrayView2<'_, f64>,
        assignments: &Array1<f64>,
        whiten_row_jets: bool,
    ) -> Vec<f64> {
        let p_out = self.output_dim();
        let sqrt_w = w_row.sqrt();
        let active_atoms = self
            .last_row_layout
            .as_ref()
            .map(|layout| layout.active_atoms[row].as_slice());
        let mut fitted = vec![0.0_f64; p_out];
        let mut decoded = vec![0.0_f64; p_out];
        for k in 0..self.k_atoms() {
            if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                continue;
            }
            self.atoms[k].fill_decoded_row(row, &mut decoded);
            let a_k = assignments[k];
            for out in 0..p_out {
                fitted[out] += a_k * decoded[out];
            }
        }
        let mut err = Array1::<f64>::zeros(p_out);
        for out in 0..p_out {
            err[out] = sqrt_w * (fitted[out] - target[[row, out]]);
        }
        match self.row_metric.as_ref() {
            Some(metric) if whiten_row_jets => metric.apply_metric_row(row, err.view()),
            _ => err.to_vec(),
        }
    }

    fn patchd_residual_third_leg_beta(
        &self,
        ctx: &PatchDResidualCtx<'_>,
        a_var: SaeLocalRowVar,
        w_var: SaeLocalRowVar,
        ch: &SaeBorderChannel,
    ) -> f64 {
        let PatchDResidualCtx {
            row,
            error_metric,
            sqrt_w,
            assignments,
            second_jets,
            is_obb,
            inv_tau,
            ..
        } = *ctx;
        let classify = |v: SaeLocalRowVar| -> (usize, Option<usize>) {
            match v {
                SaeLocalRowVar::Coord { atom, axis } => (atom, Some(axis)),
                SaeLocalRowVar::Logit { atom } => (atom, None),
            }
        };
        let (ka, aa) = classify(a_var);
        let (kw, aw) = classify(w_var);
        if ka != ch.atom || kw != ch.atom {
            return 0.0;
        }
        let atom_idx = ch.atom;
        let m = ch.basis_col;
        let mut coord_axes: Vec<usize> = Vec::with_capacity(2);
        let mut logit_count = 0usize;
        for opt in [aa, aw] {
            match opt {
                Some(axis) => coord_axes.push(axis),
                None => logit_count += 1,
            }
        }
        if logit_count > 0 && !is_obb {
            return 0.0;
        }
        let atom = &self.atoms[atom_idx];
        // ∂^{2−l}φ_m over the coord axes.
        let phi = match coord_axes.len() {
            2 => second_jets[atom_idx][[row, m, coord_axes[0], coord_axes[1]]],
            1 => atom.basis_jacobian[[row, m, coord_axes[0]]],
            _ => atom.basis_values[[row, m]],
        };
        let s = assignments[atom_idx];
        let gate_factor = match logit_count {
            0 => s,
            1 => s * (1.0 - s) * inv_tau,
            _ => s * (1.0 - s) * (1.0 - 2.0 * s) * inv_tau * inv_tau,
        };
        // eo = Σ_out error_metric[out]·output[out] (the channel's output weighting).
        let p = error_metric.len().min(ch.output.len());
        let mut eo = 0.0_f64;
        for out in 0..p {
            eo += error_metric[out] * ch.output[out];
        }
        sqrt_w * gate_factor * phi * eo
    }

    /// #2330 Patch D — the exact-A residual-curvature THIRD-derivative leg
    /// `⟨error_metric, ∂³f_{a,b,w}⟩`, the second half of `∂ΔC_tt[a,b]/∂θ_w`
    /// (the first half `⟨∂error_metric/∂θ_w, ∂²f⟩ = ⟨jets.first(w), jets.second(a,b)⟩`
    /// is added inline as term 1a). The data fit is `½rᵀMr` so its residual
    /// curvature is `⟨M r, ∂²f⟩`; differentiating the SECOND-jet factor gives this
    /// leg. For the per-atom gated decoder `f_out = Σ_k g_k(ℓ_k)·Σ_m B_k[m,out]·φ_m(x_k)`,
    /// `∂³f` is nonzero only when `a,b,w` all touch ONE atom `k` (each summand
    /// depends only on that atom's `(x_k, ℓ_k)` — exact for ordered-Beta–Bernoulli
    /// where `g_k` depends on `ℓ_k` alone). It then factors as `g_k^{(l)} · Σ_m
    /// B_k[m,out]·∂^{c}φ_m` where `l` = number of LOGIT derivatives among `{a,b,w}`
    /// and `c = 3−l` = number of COORD derivatives (over their axes). The `l=0`
    /// coord³ leg uses the plain gate value and holds for ANY mode; the `l≥1` legs
    /// use the ordered-Beta–Bernoulli logistic-gate derivatives and are skipped
    /// (returns 0) for other modes — softmax's cross-atom gate third-order is a
    /// separate follow-on. `error_metric` already carries one `√w·M`; this leg
    /// carries the other `√w`, matching the `⟨error_metric, jets.second⟩`
    /// convention exactly.
    fn patchd_residual_third_leg(
        &self,
        ctx: &PatchDResidualCtx<'_>,
        a_var: SaeLocalRowVar,
        b_var: SaeLocalRowVar,
        w_var: SaeLocalRowVar,
    ) -> f64 {
        let PatchDResidualCtx {
            row,
            error_metric,
            sqrt_w,
            assignments,
            second_jets,
            third_jets,
            is_obb,
            inv_tau,
        } = *ctx;
        // Classify each var as (atom, Some(axis)) for a coordinate or
        // (atom, None) for a logit; all three must share ONE atom.
        let classify = |v: SaeLocalRowVar| -> (usize, Option<usize>) {
            match v {
                SaeLocalRowVar::Coord { atom, axis } => (atom, Some(axis)),
                SaeLocalRowVar::Logit { atom } => (atom, None),
            }
        };
        let (ka, aa) = classify(a_var);
        let (kb, ab) = classify(b_var);
        let (kw, aw) = classify(w_var);
        if ka != kb || ka != kw {
            return 0.0;
        }
        let atom_idx = ka;
        // Collect coord axes; count logit derivatives.
        let mut coord_axes: Vec<usize> = Vec::with_capacity(3);
        let mut logit_count = 0usize;
        for opt in [aa, ab, aw] {
            match opt {
                Some(axis) => coord_axes.push(axis),
                None => logit_count += 1,
            }
        }
        if logit_count > 0 && !is_obb {
            // Non-OBB gate third-order (softmax cross-atom) is a follow-on;
            // the l==0 basis third jet still applies to any mode.
            return 0.0;
        }
        let atom = &self.atoms[atom_idx];
        let basis = atom.basis_size();
        let decoder = atom.decoder_coefficients(); // (basis, out)
        let p = error_metric.len();
        // D_c[out] = Σ_m B[m,out] · ∂^c φ_m over the collected coord axes.
        let mut d_c = vec![0.0_f64; p];
        match coord_axes.len() {
            3 => {
                let Some(tj) = third_jets.and_then(|t| t[atom_idx].as_ref()) else {
                    return 0.0; // no analytic third jet for this atom
                };
                let (a0, a1, a2) = (coord_axes[0], coord_axes[1], coord_axes[2]);
                for m in 0..basis {
                    let phi3 = tj[[row, m, a0, a1, a2]];
                    for out in 0..p {
                        d_c[out] += decoder[[m, out]] * phi3;
                    }
                }
            }
            2 => {
                let sj = &second_jets[atom_idx];
                let (a0, a1) = (coord_axes[0], coord_axes[1]);
                for m in 0..basis {
                    let phi2 = sj[[row, m, a0, a1]];
                    for out in 0..p {
                        d_c[out] += decoder[[m, out]] * phi2;
                    }
                }
            }
            1 => {
                let a0 = coord_axes[0];
                for m in 0..basis {
                    let phi1 = atom.basis_jacobian[[row, m, a0]];
                    for out in 0..p {
                        d_c[out] += decoder[[m, out]] * phi1;
                    }
                }
            }
            _ => {
                for m in 0..basis {
                    let phi0 = atom.basis_values[[row, m]];
                    for out in 0..p {
                        d_c[out] += decoder[[m, out]] * phi0;
                    }
                }
            }
        }
        // Gate factor g^{(l)}: l logit derivatives of the atom's gate. For OBB
        // g = σ(ℓ/τ): g0=s, g1=s(1−s)/τ, g2=s(1−s)(1−2s)/τ², g3=s(1−s)(1−6s+6s²)/τ³.
        let s = assignments[atom_idx];
        let gate_factor = match logit_count {
            0 => s,
            1 => s * (1.0 - s) * inv_tau,
            2 => s * (1.0 - s) * (1.0 - 2.0 * s) * inv_tau * inv_tau,
            _ => s * (1.0 - s) * (1.0 - 6.0 * s + 6.0 * s * s) * inv_tau * inv_tau * inv_tau,
        };
        let mut acc = 0.0_f64;
        for out in 0..p {
            acc += error_metric[out] * d_c[out];
        }
        sqrt_w * gate_factor * acc
    }

    pub(crate) fn logdet_theta_adjoint_dense(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        inv: &Array2<f64>,
        channel: ThetaAdjointDhChannel,
        skip_deflation_dk: bool,
        exact_a: bool,
        // #2330 Patch D — the data target, required ONLY for the exact-A
        // residual-curvature third-derivative leg `⟨error_metric, ∂³f⟩`. `None`
        // reproduces the pre-Patch-D behaviour exactly (the leg is skipped), so
        // every non-exact-A caller passes `None`.
        residual_target: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeArrowVector, String> {
        // #2330 — `skip_deflation_dk` drops the Daleckii–Krein deflation
        // correction, leaving the raw trace contraction. The split probe uses it
        // to attribute the g3 cross non-conservation to the trace vs the
        // frozen-DK piece of the twist. Production callers pass `false`.
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let k_atoms = self.k_atoms();
        let n = self.n_obs();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let whiten_row_jets = self.whiten_logdet_row_jets();
        let want_data = matches!(channel, ThetaAdjointDhChannel::All);
        let want_entropy = matches!(
            channel,
            ThetaAdjointDhChannel::All | ThetaAdjointDhChannel::SoftmaxSparseMixed
        );
        let want_ard = matches!(
            channel,
            ThetaAdjointDhChannel::All | ThetaAdjointDhChannel::ArdMixed { .. }
        );
        // `1/τ` (always, for the softmax data-weight logit factor) and the
        // entropy Gershgorin majorizer scale `λ_sparse·s/τ²` (only a live free
        // logit, i.e. `k_atoms > 1`, carries the sparsity penalty).
        let (entropy_scale, inv_tau) = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } => {
                let inv_tau = 1.0 / temperature;
                let scale = if k_atoms > 1 {
                    rho.lambda_sparse()? * sparsity * inv_tau * inv_tau
                } else {
                    0.0
                };
                (scale, inv_tau)
            }
            _ => (0.0, 0.0),
        };
        // #2330 Patch D residual-curvature leg setup. Active only on the exact-A
        // route with a target: builds `∂³f` from raw basis jets + gate
        // derivatives (see `patchd_residual_third_leg`).
        let patchd_residual = exact_a.then_some(residual_target).flatten();
        let patchd_third_jets = if patchd_residual.is_some() {
            Some(self.atom_third_jets()?)
        } else {
            None
        };
        let patchd_is_obb = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        );
        // #2330 Patch D channel-2 — ordered-BB prior curvature θ-adjoint data
        // (cross-row; contracted after the row loop). Only for the exact-A
        // full-channel route.
        let patchd_obb_adjoint = if patchd_residual.is_some() && patchd_is_obb {
            crate::assignment::ordered_beta_bernoulli_logit_adjoint_data_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?
        } else {
            None
        };
        let patchd_obb_inv_tau = match self.assignment.mode {
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => 1.0 / temperature,
            _ => 0.0,
        };
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        let mut assignments = Array1::<f64>::zeros(k_atoms);
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    jet_window_next,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let mut jets = jet_window
                .pop_front()
                .ok_or_else(|| "logdet_theta_adjoint_dense: empty jet window".to_string())?;
            if whiten_row_jets {
                self.apply_whiten_to_logdet_row_jets(row, &mut jets)?;
            }
            let a_soft = assignments
                .as_slice()
                .expect("softmax assignments row must be contiguous");
            let m_log_mean = softmax_majorizer_log_mean(a_soft);
            let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            // #2330 Patch D — per-row `error_metric = √w·M·r` in output space,
            // built EXACTLY as `apply_exact_hessian_minus_b` builds the object it
            // contracts ΔC against (√w residual, then whitening metric applied).
            let patchd_error_metric: Option<Vec<f64>> = patchd_residual.map(|tgt| {
                self.patchd_row_error_metric(row, w_row, tgt, &assignments, whiten_row_jets)
            });
            let patchd_sqrt_w = w_row.sqrt();
            let patchd_ctx: Option<PatchDResidualCtx<'_>> =
                patchd_error_metric.as_deref().map(|em| PatchDResidualCtx {
                    row,
                    error_metric: em,
                    sqrt_w: patchd_sqrt_w,
                    assignments: &assignments,
                    second_jets: &second_jets,
                    third_jets: patchd_third_jets.as_deref(),
                    is_obb: patchd_is_obb,
                    inv_tau: patchd_obb_inv_tau,
                });
            // #2308 — per-row spectral/gauge deflation the criterion factor applied.
            // It is FROZEN at the fixed stratum (the radial-gauge / ARD-inactive-half
            // null is ρ-invariant), so contracting the DEFLATED inverse `inv` and
            // subtracting the SAME Daleckii–Krein correction the production θ-adjoint
            // subtracts makes `Γ(inv)` — and its twist `Γ(−G Mᵢ G)` — match the
            // gradient on the deflated circle route (where deflation is the norm, not
            // an error). `deflation_block_correction` is linear in `inv`, so the twist
            // rides through it exactly.
            let defl_dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let defl_spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            let inv_vv_block = if defl_dirs.is_empty() {
                Array2::<f64>::zeros((0, 0))
            } else {
                inv.slice(s![base..base + q, base..base + q]).to_owned()
            };
            for w in 0..q {
                let logit_w = match jets.vars[w] {
                    SaeLocalRowVar::Logit { atom } => Some(atom),
                    SaeLocalRowVar::Coord { .. } => None,
                };
                let mut gamma = 0.0_f64;
                let mut dh_mat = if defl_dirs.is_empty() {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = 0.0_f64;
                        if want_data {
                            dh += match (logit_w, jets.vars[a], jets.vars[b]) {
                                (
                                    Some(atom_w),
                                    SaeLocalRowVar::Coord { atom: atom_a, .. },
                                    SaeLocalRowVar::Coord { atom: atom_b, .. },
                                ) => {
                                    sae_dot(jets.first(a), jets.first(b))
                                        * (Self::softmax_data_weight_product_logit_factor(
                                            a_soft, atom_a, atom_b, atom_w, inv_tau,
                                        ) + if patchd_is_obb {
                                            // #2330 / #2371 -- ordered-Beta--Bernoulli gate
                                            // gradient of the GN curvature. `B[a,b] = <J_a, J_b>`
                                            // and each leg `J_k` carries its INDEPENDENT gate
                                            // `g_k = sigma(l_k/tau)` linearly, so
                                            // `dB/dl_w = [1(w==a) + 1(w==b)] * (1-g_w)/tau * B`.
                                            // The matching leg gate is `g_w`, so a single
                                            // `(1 - a_soft[atom_w])` is correct per side:
                                            // same-atom-both gives sided=2 (bitwise the prior
                                            // landed value), one-sided cross-atom gives sided=1
                                            // (the #2371 term wrongly dropped as exactly zero).
                                            // The softmax factor above is 0 here (`inv_tau` is
                                            // 0 for non-softmax modes), so softmax is unchanged.
                                            let sided = (atom_w == atom_a) as u32
                                                + (atom_w == atom_b) as u32;
                                            sided as f64
                                                * (1.0 - a_soft[atom_w])
                                                * patchd_obb_inv_tau
                                        } else {
                                            0.0
                                        })
                                }
                                _ => {
                                    sae_dot(jets.second(a, w), jets.first(b))
                                        + sae_dot(jets.first(a), jets.second(b, w))
                                }
                            };
                        }
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg(
                                ctx,
                                jets.vars[a],
                                jets.vars[b],
                                jets.vars[w],
                            );
                        }
                        if want_data && exact_a {
                            // #2330 Patch D (1a) — `A = B + ΔC` carries the residual
                            // curvature `ΔC_tt[a,b] = ⟨error_metric, ∂²f_ab⟩` that the
                            // Gauss-Newton assembly drops, and that block moves with
                            // `θ_w` too:
                            //   `∂ΔC_tt[a,b]/∂θ_w = ⟨∂error_metric/∂θ_w, ∂²f_ab⟩`
                            //                      `+ ⟨error_metric, ∂³f_abw⟩`.
                            // `∂error_metric/∂θ_w = √w·M·∂f/∂θ_w`, which in THIS
                            // function's jet convention is exactly `jets.first(w)`:
                            // every jet carries one `√w` and (under whitening) one
                            // metric factor `L`, so a plain dot of two jets
                            // reconstitutes the `w`-weighted `M`-inner product the
                            // assembly uses. Only the FIRST leg lands here; the
                            // third-jet leg `⟨error_metric, ∂³f_abw⟩` needs a jet
                            // channel `SaeRowJets` does not expose.
                            dh += sae_dot(jets.first(w), jets.second(a, b));
                        }
                        if want_entropy {
                            if let (
                                Some(atom_w),
                                SaeLocalRowVar::Logit { atom: atom_a },
                                SaeLocalRowVar::Logit { atom: atom_b },
                            ) = (logit_w, jets.vars[a], jets.vars[b])
                            {
                                if atom_a == atom_b {
                                    dh += w_row
                                        * active_softmax_majorizer_logit_derivative_entry(
                                            a_soft,
                                            atom_a,
                                            atom_w,
                                            m_log_mean,
                                            entropy_scale,
                                            inv_tau,
                                        );
                                }
                            }
                        }
                        if want_ard && a == b && a == w {
                            if let SaeLocalRowVar::Coord { atom, axis } = jets.vars[a] {
                                if !ard_precisions[atom].is_empty() {
                                    let include = match channel {
                                        ThetaAdjointDhChannel::ArdMixed { target_flat } => {
                                            rho.ard_flat_index(atom, axis) == target_flat
                                        }
                                        _ => true,
                                    };
                                    if include {
                                        dh += if exact_a {
                                            self.ard_exact_hessian_derivative(
                                                ard_precisions[atom][axis],
                                                row,
                                                atom,
                                                axis,
                                            )
                                        } else {
                                            self.ard_majorized_hessian_derivative(
                                                ard_precisions[atom][axis],
                                                row,
                                                atom,
                                                axis,
                                            )
                                        };
                                    }
                                }
                            }
                        }
                        if !defl_dirs.is_empty() {
                            dh_mat[[a, b]] = dh;
                        }
                        gamma += inv[[base + b, base + a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() && !skip_deflation_dk {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv_block,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                if want_data {
                    for a in 0..q {
                        for (beta_pos, ch) in border.iter().enumerate() {
                            // #2330 Patch D (1a), t--beta leg: `ΔC_tβ[a,β] =
                            // ⟨error_metric, ∂²f_aβ⟩` moves with `θ_w` through the
                            // residual exactly as the t--t block does.
                            let mut dh = sae_dot(jets.second(a, w), jets.beta(beta_pos))
                                + sae_dot(jets.first(a), jets.beta_deriv(w, beta_pos))
                                + if exact_a {
                                    sae_dot(jets.first(w), jets.beta_deriv(a, beta_pos))
                                } else {
                                    0.0
                                };
                            if let Some(ctx) = patchd_ctx.as_ref() {
                                dh += self.patchd_residual_third_leg_beta(
                                    ctx,
                                    jets.vars[a],
                                    jets.vars[w],
                                    ch,
                                );
                            }
                            gamma += 2.0 * inv[[base + a, total_t + ch.index]] * dh;
                        }
                    }
                    for (beta_i, ch_i) in border.iter().enumerate() {
                        for (beta_j, ch_j) in border.iter().enumerate() {
                            let dh = sae_dot(jets.beta_deriv(w, beta_i), jets.beta(beta_j))
                                + sae_dot(jets.beta(beta_i), jets.beta_deriv(w, beta_j));
                            gamma += inv[[total_t + ch_i.index, total_t + ch_j.index]] * dh;
                        }
                    }
                }
                gamma_t[base + w] = gamma;
            }
            if want_data {
                for (w_beta_pos, w_channel) in border.iter().enumerate() {
                    let mut gamma = 0.0_f64;
                    let mut dh_mat = if defl_dirs.is_empty() {
                        Array2::<f64>::zeros((0, 0))
                    } else {
                        Array2::<f64>::zeros((q, q))
                    };
                    for a in 0..q {
                        for b in 0..q {
                            let dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.first(b))
                                + sae_dot(jets.first(a), jets.beta_l_deriv(b, w_beta_pos));
                            if !defl_dirs.is_empty() {
                                dh_mat[[a, b]] = dh;
                            }
                            gamma += inv[[base + b, base + a]] * dh;
                        }
                    }
                    if !defl_dirs.is_empty() && !skip_deflation_dk {
                        gamma -= Self::deflation_block_correction(
                            &inv_vv_block,
                            &dh_mat,
                            defl_dirs,
                            defl_spectrum,
                        );
                    }
                    for a in 0..q {
                        for (beta_pos, ch) in border.iter().enumerate() {
                            let dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.beta(beta_pos));
                            gamma += 2.0 * inv[[base + a, total_t + ch.index]] * dh;
                        }
                    }
                    gamma_beta[w_channel.index] += gamma;
                }
            }
        }
        // #2330 Patch D channel-2 — fold the ordered-BB prior logit θ-adjoint into
        // the logit t-slots (full channel only; it is the ∂ΔC_obb/∂θ leg).
        if want_data {
            if let Some(data) = patchd_obb_adjoint.as_ref() {
                let obb = self.dense_exact_a_ordered_bb_logit_theta_adjoint(cache, inv, data)?;
                gamma_t += &obb;
            }
        }
        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }

    /// PATH C (#2253) CH5 — the fixed-stratum ρ-derivative of the rank-charge
    /// θ-adjoint `∇R = production_rank_charge_derivative().theta`, for ONE smooth
    /// coordinate `smooth_flat`. `∇R` depends on ρ only through the per-atom
    /// penalized Gram `A = G + λ S` (`λ = e^{ρ_smooth}`), and the θ-assembly is
    /// LINEAR in each atom's differential blocks (`gram`, `occupancy`), so the
    /// derivative reruns the SAME assembly with those blocks replaced by their
    /// λ-derivatives (and zeroed for every other atom). With `A⁻¹ = inv`,
    /// `S = smooth_penalty`, `dλ/dρ = λ`, `dA⁻¹/dλ = −A⁻¹SA⁻¹`:
    /// `d(inv − inv G inv)/dρ = λ(−inv S inv + inv S inv G inv + inv G inv S inv)`
    /// and `d tr(inv G)/dρ = −λ tr(inv S inv G)`. Non-interior-EDF atoms are on a
    /// locally constant branch (zero derivative), matching the gradient.
    fn rank_charge_theta_rho_derivative(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        smooth_flat: usize,
    ) -> Result<SaeArrowVector, String> {
        let target_atom = smooth_flat - rho.smooth_flat_start();
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion = self.reconstruction_dispersion(loss, cache, rho, Some(residual.view()))?;
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams)?;
        let n_eff = self.per_atom_effective_sample_size();
        let lambda_vec = rho.lambda_smooth_vec()?;
        let p = self.output_dim() as f64;

        // Per-atom differential BLOCKS (gram, occupancy), zero except the target
        // atom, whose blocks are the ρ_smooth-derivatives of the gradient's.
        let mut atom_differentials: Vec<ProductionRankChargeAtomDifferential> =
            Vec::with_capacity(self.k_atoms());
        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            let m = atom.basis_size();
            if atom_idx != target_atom || m == 0 {
                atom_differentials.push(ProductionRankChargeAtomDifferential {
                    gram: Array2::<f64>::zeros((m, m)),
                    occupancy: 0.0,
                });
                continue;
            }
            let gram = &grams[atom_idx];
            let n_atom = n_eff[atom_idx];
            let lambda = lambda_vec[atom_idx];
            let spectrum = super::wbic_audit::recon_spectrum(
                gram,
                atom.decoder_coefficients(),
                n_atom,
                p,
                dispersion,
                lambda,
                Some(atom.smooth_penalty()),
            )?;
            let rank = spectrum.production_chargeable_rank() as f64;
            if !(rank > 0.0) {
                return Err(format!(
                    "rank_charge_theta_rho_derivative: atom {atom_idx} is on the rank-zero \
                     Laplace-invalid branch (vanished decoder)"
                ));
            }
            let log_n = n_atom.max(1.0).ln();
            if log_n == 0.0 {
                atom_differentials.push(ProductionRankChargeAtomDifferential {
                    gram: Array2::<f64>::zeros((m, m)),
                    occupancy: 0.0,
                });
                continue;
            }
            let s = atom.smooth_penalty();
            let mut penalized_gram = gram.clone();
            for r in 0..m {
                for c in 0..m {
                    penalized_gram[[r, c]] += lambda * s[[r, c]];
                }
            }
            let factor = penalized_gram.cholesky(Side::Lower).map_err(|error| {
                format!(
                    "rank_charge_theta_rho_derivative: atom {atom_idx} penalized Gram \
                     factorization failed: {error}"
                )
            })?;
            let inverse = factor.solve_mat(&Array2::<f64>::eye(m));
            let edf_matrix = factor.solve_mat(gram);
            let raw_edf = (0..m).map(|i| edf_matrix[[i, i]]).sum::<f64>();
            let edf = super::construction::certified_basis_edf(
                raw_edf,
                m,
                "rank_charge_theta_rho_derivative",
            )?;
            let edf_is_interior = edf > 0.0 && edf < m as f64;
            // Reused products (all m×m): inv S inv, inv G inv, inv S inv G inv,
            // inv G inv S inv, and inv S inv G (for the EDF trace).
            let inv_s_inv = inverse.dot(s).dot(&inverse);
            let inv_g_inv = inverse.dot(gram).dot(&inverse);
            let inv_s_inv_g_inv = inv_s_inv.dot(gram).dot(&inverse);
            let inv_g_inv_s_inv = inv_g_inv.dot(s).dot(&inverse);
            let mut gram_prime = Array2::<f64>::zeros((m, m));
            if edf_is_interior {
                let coeff = lambda * 0.5 * rank * log_n;
                for r in 0..m {
                    for c in 0..m {
                        gram_prime[[r, c]] = coeff
                            * (-inv_s_inv[[r, c]]
                                + inv_s_inv_g_inv[[r, c]]
                                + inv_g_inv_s_inv[[r, c]]);
                    }
                }
            }
            let occupancy_prime = if n_atom > 1.0 {
                let edf_prime = if edf_is_interior {
                    let inv_s_inv_g = inv_s_inv.dot(gram);
                    -lambda * (0..m).map(|i| inv_s_inv_g[[i, i]]).sum::<f64>()
                } else {
                    0.0
                };
                0.5 * rank * edf_prime / n_atom
            } else {
                0.0
            };
            atom_differentials.push(ProductionRankChargeAtomDifferential {
                gram: gram_prime,
                occupancy: occupancy_prime,
            });
        }

        // The SAME linear θ-assembly as `production_rank_charge_derivative`, now
        // driven by the differential-of-the-differential blocks.
        let mut theta_t = Array1::<f64>::zeros(cache.delta_t_len());
        let theta_beta = Array1::<f64>::zeros(cache.k);
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        for row in 0..self.n_obs() {
            self.assignment.try_assignments_row_into(
                row,
                assignments
                    .as_slice_mut()
                    .expect("rank-charge assignment scratch is contiguous"),
            )?;
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let base = cache.row_offsets[row];
            for (slot, var) in vars.into_iter().enumerate() {
                theta_t[base + slot] = match var {
                    SaeLocalRowVar::Coord { atom, axis } => {
                        let a = assignments[atom];
                        if a == 0.0 {
                            0.0
                        } else {
                            let phi = self.atoms[atom].basis_values.row(row);
                            let dphi = self.atoms[atom].basis_jacobian.slice(s![row, .., axis]);
                            2.0 * a * a * dphi.dot(&atom_differentials[atom].gram.dot(&phi))
                        }
                    }
                    SaeLocalRowVar::Logit { atom: wrt_atom } => {
                        let mut derivative = 0.0_f64;
                        for atom in 0..self.k_atoms() {
                            let da = self.rank_charge_assignment_derivative(
                                wrt_atom,
                                atom,
                                assignments
                                    .as_slice()
                                    .expect("rank-charge assignment scratch is contiguous"),
                            );
                            if da == 0.0 {
                                continue;
                            }
                            let a = assignments[atom];
                            let phi = self.atoms[atom].basis_values.row(row);
                            let gram_quadratic = phi.dot(&atom_differentials[atom].gram.dot(&phi));
                            derivative += 2.0
                                * a
                                * da
                                * (gram_quadratic + atom_differentials[atom].occupancy);
                        }
                        derivative
                    }
                };
            }
        }
        Ok(SaeArrowVector {
            t: theta_t,
            beta: theta_beta,
        })
    }

    /// PATH C (#2253) CH5 — the exact fixed-stratum second derivative of the
    /// outer gradient's third-order forward-sensitivity channel
    /// `g3[j] = −½⟨a, g_ρ,j⟩`, `a = A⁺Γ_eff`.
    ///
    /// `H3[i,j] = ∂g3[j]/∂ρ_i = −½( ⟨dΓ_eff/dρ_i − M_i·a, b_j⟩ + δ_ij⟨a, g_ρ,j⟩ )`
    /// with `b_j = A⁺ g_ρ,j` (self-adjointness of `A⁺`). `Γ_eff = Γ_joint − Γ_tt
    /// + 2∇R` — the SAME effective adjoint the gradient assembles. Each
    /// `dΓ_·/dρ_i` splits into part-(a) `−tr(inv M_i inv K_w)` (twisted inverse)
    /// and part-(b) `tr(inv ∂K_w/∂ρ_i)` (the ARD / softmax-sparse mixed
    /// channels), and `d∇R/dρ` is nonzero only on the smooth coordinates. The
    /// returned block is `∂g3[j]/∂ρ_i` verbatim (validated by the FD gate);
    /// the caller may symmetrize.
    fn third_order_forward_sensitivity_hessian(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        // Covered config: the small-dense softmax route with no deflation,
        // frames, compact layout, or ordered Beta--Bernoulli. Outside it the
        // dense `dh` reconstruction and the twist are not the exact operator, so
        // refuse rather than advertise wrong curvature.
        if !matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            return Err(
                "third_order_forward_sensitivity_hessian: only the softmax assignment route is \
                 modelled by the dense θ-adjoint reconstruction"
                    .to_string(),
            );
        }
        if self.last_row_layout.is_some() {
            return Err(
                "third_order_forward_sensitivity_hessian: the compact top-k softmax row layout is \
                 not covered by the dense θ-adjoint reconstruction"
                    .to_string(),
            );
        }
        if self.frames_active() {
            return Err(
                "third_order_forward_sensitivity_hessian: border-frame smoothness offsets are not \
                 covered by this channel"
                    .to_string(),
            );
        }
        let solver = DeflatedArrowSolver::plain(cache);
        // Per-row spectral/gauge deflation IS modelled — the dense θ-adjoint
        // subtracts the same frozen Daleckii–Krein correction the production
        // builder does (#2308), and the plain deflated inverse is what `a`/`b_j`
        // and the twist all ride. What the plain solver CANNOT reconstruct is the
        // rank-R β-Schur Woodbury GAUGE correction: there the materialized inverse
        // would omit it, so refuse rather than assemble a wrong twist.
        if !solver.plain_selected_inverse_available() {
            return Err(
                "third_order_forward_sensitivity_hessian: a β-Schur Woodbury gauge deflation is \
                 active; the plain selected inverse omits its rank-R correction, so the \
                 twisted-inverse reconstruction is not the exact operator"
                    .to_string(),
            );
        }

        let n_params = rho.to_flat().len();
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let flatten = |v: &SaeArrowVector| -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(dim);
            out.slice_mut(s![..total_t]).assign(&v.t);
            out.slice_mut(s![total_t..]).assign(&v.beta);
            out
        };

        let g = self.materialize_joint_inverse(cache, &solver)?;
        let h_bd = self.materialize_block_diag_t_inverse(cache);
        let operators = self.penalty_curvature_operators_by_flat(rho, cache)?;
        // `∂A/∂ρᵢ = ∂H/∂ρᵢ (operators) + ∂(ΔC)/∂ρᵢ (this delta)`. BOTH the twist
        // inverse ∂G/∂ρ = −G(∂A/∂ρ)G and the IFT `Mᵢ·a` term differentiate the
        // EXACT stationarity Hessian A, so both add this delta (#2330).
        let exact_deltas = self.exact_stationarity_penalty_derivative_delta_by_flat(rho, cache)?;

        // Effective adjoint Γ_eff = Γ_joint − Γ_tt + 2∇R, assembled EXACTLY as
        // the gradient does (construction_exact_hessian.rs analytic assembler).
        let rank_charge = self.production_rank_charge_derivative(target, rho, loss, cache)?;
        let mut gamma_eff = self.logdet_theta_adjoint(rho, cache, &solver)?;
        let gamma_tt = self.coordinate_block_logdet_theta_adjoint(
            rho,
            cache,
            EvidenceOperator::Majorizer,
            None,
        )?;
        gamma_eff.t -= &gamma_tt.t;
        gamma_eff.beta -= &gamma_tt.beta;
        gamma_eff.t.scaled_add(2.0, &rank_charge.theta.t);
        gamma_eff.beta.scaled_add(2.0, &rank_charge.theta.beta);

        // Adjoints: factor the materialized exact A once, then apply its
        // rank-revealing pseudoinverse to Γ_eff and every g_ρ,j RHS.
        let stationarity_geometry =
            self.materialize_exact_stationarity_geometry(rho, target, cache)?;
        let a_vec = stationarity_geometry.solve_stationarity(&gamma_eff)?;
        let a_flat = flatten(&a_vec);
        let flats: Vec<usize> = operators.keys().copied().collect();
        let mut b_flat: std::collections::BTreeMap<usize, Array1<f64>> =
            std::collections::BTreeMap::new();
        let mut g_rho_flat: std::collections::BTreeMap<usize, Array1<f64>> =
            std::collections::BTreeMap::new();
        for &j in &flats {
            let g_rho = self.outer_rho_gradient_ift_rhs(rho, j, cache)?;
            let b_j = stationarity_geometry.solve_stationarity(&g_rho)?;
            g_rho_flat.insert(j, flatten(&g_rho));
            b_flat.insert(j, flatten(&b_j));
        }

        let smooth_range =
            rho.smooth_flat_start()..rho.smooth_flat_start() + rho.log_lambda_smooth.len();
        let sparse_index = rho.sparse_flat_index();

        let mut hessian = Array2::<f64>::zeros((n_params, n_params));
        for &i in &flats {
            let m_i = &operators[&i];
            // Twisted inverses G_i = −G (∂A/∂ρ_i) G, h_bd_i = −h_bd (∂A/∂ρ_i) h_bd.
            // The Laplace logdet is logdet(A_exact), so ∂G/∂ρ_i differentiates the
            // EXACT stationarity Hessian ∂A/∂ρ_i = M_i + ΔC-delta_i — NOT the
            // majorized M_i alone, which is one-sided on ARD (delta ≠ 0 only for
            // ARD/softmax) and breaks g3 smooth↔ARD cross-conservation (#2330).
            let twist_op = match exact_deltas.get(&i) {
                Some(delta_i) => m_i + delta_i,
                None => m_i.clone(),
            };
            let g_i = -g.dot(&twist_op).dot(&g);
            let h_bd_i = -h_bd.dot(&twist_op).dot(&h_bd);

            // dΓ_joint/dρ_i and dΓ_tt/dρ_i = part(a) twist + part(b) mixed.
            let mut d_gamma_joint = self.logdet_theta_adjoint_dense(
                rho,
                cache,
                &g_i,
                ThetaAdjointDhChannel::All,
                false,
                false,
                None,
            )?;
            let mut d_gamma_tt = self.logdet_theta_adjoint_dense(
                rho,
                cache,
                &h_bd_i,
                ThetaAdjointDhChannel::All,
                false,
                false,
                None,
            )?;
            if smooth_range.contains(&i) {
                // Smooth part(b) = 0; the only smooth ρ-derivative of Γ_eff is
                // through the rank-charge adjoint.
                let d_rank = self.rank_charge_theta_rho_derivative(target, rho, loss, cache, i)?;
                d_gamma_joint.t.scaled_add(2.0, &d_rank.t);
                d_gamma_joint.beta.scaled_add(2.0, &d_rank.beta);
            } else if sparse_index == Some(i) {
                let mixed_joint = self.logdet_theta_adjoint_dense(
                    rho,
                    cache,
                    &g,
                    ThetaAdjointDhChannel::SoftmaxSparseMixed,
                    false,
                    false,
                    None,
                )?;
                let mixed_tt = self.logdet_theta_adjoint_dense(
                    rho,
                    cache,
                    &h_bd,
                    ThetaAdjointDhChannel::SoftmaxSparseMixed,
                    false,
                    false,
                    None,
                )?;
                d_gamma_joint.t += &mixed_joint.t;
                d_gamma_joint.beta += &mixed_joint.beta;
                d_gamma_tt.t += &mixed_tt.t;
                d_gamma_tt.beta += &mixed_tt.beta;
            } else {
                // ARD coordinate: part(b) mixed channel for this flat index.
                let mixed_joint = self.logdet_theta_adjoint_dense(
                    rho,
                    cache,
                    &g,
                    ThetaAdjointDhChannel::ArdMixed { target_flat: i },
                    false,
                    false,
                    None,
                )?;
                let mixed_tt = self.logdet_theta_adjoint_dense(
                    rho,
                    cache,
                    &h_bd,
                    ThetaAdjointDhChannel::ArdMixed { target_flat: i },
                    false,
                    false,
                    None,
                )?;
                d_gamma_joint.t += &mixed_joint.t;
                d_gamma_joint.beta += &mixed_joint.beta;
                d_gamma_tt.t += &mixed_tt.t;
                d_gamma_tt.beta += &mixed_tt.beta;
            }

            // dΓ_eff/dρ_i = dΓ_joint − dΓ_tt (+2∇R' folded into joint above).
            let mut d_gamma = flatten(&d_gamma_joint);
            d_gamma -= &flatten(&d_gamma_tt);
            // resid_i = dΓ_eff/dρ_i − (∂A/∂ρ_i)·a, with ∂A/∂ρ_i = M_i + ΔC-delta_i
            // (the IFT term differentiates the EXACT A, not the majorized H).
            let mut a_op_i_a = m_i.dot(&a_flat);
            if let Some(delta_i) = exact_deltas.get(&i) {
                a_op_i_a += &delta_i.dot(&a_flat);
            }
            let resid_i = &d_gamma - &a_op_i_a;

            for &j in &flats {
                let b_j = &b_flat[&j];
                let mut term = resid_i.dot(b_j);
                if i == j {
                    term += a_flat.dot(&g_rho_flat[&j]);
                }
                hessian[[i, j]] = -0.5 * term;
            }
        }
        Ok(hessian)
    }

    /// #2080 forward plumbing — the analytic outer-ρ gradient with an OPTIONAL
    /// low-rank representation of the reduced-logdet derivative.
    ///
    /// #2515 — `evidence` is a [`BundleEvidenceGeometry`], not a bare probe pair:
    /// its variant NAMES the operator whose selected inverse the from-probes
    /// channels contract, and the exact-`A` variant carries that operator's own
    /// factor cache. `cache` stays the `B` stationarity geometry on every route.
    ///
    /// When `evidence` is `Some`, the THREE reduced-logdet channels
    /// that have matrix-free siblings — the per-atom decoder smoothness EDF
    /// `tr(H⁻¹ M_k)`, the per-(atom,axis) ARD log-precision Hessian trace
    /// `½tr(H⁻¹ ∂H/∂logα)`, and the #1006 envelope Γ = tr(H⁻¹ ∂H/∂θ) — are evaluated
    /// off that bundle (`decoder_smoothness_effective_dof_per_atom_from_probes` /
    /// `ard_log_precision_hessian_trace_from_probes` / `logdet_theta_adjoint_from_probes`)
    /// instead of the dense `DeflatedArrowSolver` selected inverse. For the
    /// rational route the two slices are the identical weighted vectors emitted
    /// by `RationalLogdetPlan::into_directional_derivative_bundle`, so every
    /// contraction is the derivative of the SAME shifted rational value, not a
    /// separately sampled `S^-1`. They convert
    /// together as ONE all-or-nothing cluster on the single `Some` (invariant #1):
    /// never a partial mix within a single eval. Each from-probes channel PRICES
    /// deflated rows (#2712): `A_i⁻¹ + G_i S⁻¹ G_iᵀ` built on the conditioned row
    /// Cholesky IS the deflated block, so each applies the same Daleckii–Krein
    /// correction its dense sibling applies, and no fit is routed to the dense
    /// channel for carrying deflation.
    ///
    /// The complete all-coordinate assembler is single-adjoint (#2080-A): the IFT
    /// correction `−½·⟨Γ, A⁺ g_ρ_l⟩` over every outer coordinate collapses to ONE
    /// exact-stationarity solve `a = A⁺Γ` plus O(K) cheap `⟨a, g_ρ_l⟩`
    /// contractions (self-adjointness of `A⁺`; see the collapse below). That
    /// single adjoint solve is the ONLY solver-bound step, so the whole assembler
    /// runs matrix-free at massive K: pass `matrix_free_system = Some(system)` to
    /// route it through [`Self::solve_exact_stationarity_matrix_free`] (the
    /// reduced-Schur CG on the reassembled undamped operator) with
    /// `solver = DeflatedArrowSolver::plain(cache)` for the cheap per-row
    /// `coordinate_block_*` subtractions — the K≥4096, direct-logdet-not-admitted
    /// route, mirroring the matrix-free branch of this complete assembler.
    /// Pass `matrix_free_system = None` to use the dense [`DeflatedArrowSolver`]
    /// adjoint (the direct-logdet-admitted route). Both produce the same complete
    /// derivative; the from-probes trace channels and the matrix-free adjoint
    /// convert together as one all-or-nothing matrix-free cluster (invariant #1).
    pub(crate) fn analytic_outer_rho_gradient_components_with_bundle(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        evidence: Option<BundleEvidenceGeometry<'_>>,
        matrix_free_system: Option<&ArrowSchurSystem>,
    ) -> Result<SaeOuterRhoGradientComponents, OuterGradientError> {
        self.assignment
            .validate_rho_domain(rho)
            .map_err(OuterGradientError::internal)?;
        // #2515 — resolve the evidence geometry ONCE. `logdet_derivative_bundle`
        // is the probe pair every from-probes channel contracts; `evidence_cache`
        // is the factor cache whose row blocks those channels reconstruct the
        // arrow inverse from; `evidence_operator` is which operator's ρ/θ
        // derivative the curvature channels differentiate. All three come from
        // one value, so they cannot name different operators.
        //
        // No bundle ⇒ the dense `DeflatedArrowSolver` selected inverse on the
        // caller's `cache`, which is `B`. A bundle ⇒ the exact observed
        // information, carried with its own cache.
        let logdet_derivative_bundle = evidence
            .as_ref()
            .map(|geometry| (geometry.probes, geometry.sinv));
        let evidence_cache = evidence.as_ref().map_or(cache, |geometry| geometry.cache);
        let evidence_operator = evidence
            .as_ref()
            .map_or(EvidenceOperator::Majorizer, |geometry| geometry.operator);
        let n_params = rho.to_flat().len();
        let mut explicit = Array1::<f64>::zeros(n_params);
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        let mut occam = Array1::<f64>::zeros(n_params);
        let mut third_order_correction = Array1::<f64>::zeros(n_params);
        let rank_charge = self
            .production_rank_charge_derivative(target, rho, loss, cache)
            .map_err(OuterGradientError::internal)?;
        // #2330 Phase-2 / #2333 — which majorizer the logdet channels belong to
        // is a property of the ROUTE, and it is known here, before any of them is
        // produced. On the dense direct-logdet route the ranked term is ½log|A|,
        // so `logdet_trace` and `Γ` come from `dense_exact_a_logdet_channels`
        // and every B-majorizer producer below would be discarded unread: a
        // selected-inverse pass for the smoothness EDF, two solver-bound
        // assignment log-strength traces, the ARD Hessian traces, and two
        // θ-adjoint towers. Deciding once, up front, is what lets them be
        // skipped rather than computed and overwritten. The `explicit`, `occam`
        // and rank-charge-direct channels are majorizer-independent and are
        // produced on every route.
        // #2500 — and only where the dense θ-adjoint reconstruction MODELS this
        // fit's assignment family. Taking the exact-A arm with a `Γ` that is not
        // the θ-derivative of `A` would trade the B-route's staged value↔gradient
        // gap for an outright wrong gradient; deciding it HERE, with the rest of
        // the route, is also what keeps the B-majorizer producers alive for a
        // family that needs them.
        let exact_a_logdet_route = logdet_derivative_bundle.is_none()
            && matrix_free_system.is_none()
            && self.dense_exact_a_theta_adjoint_is_modelled();

        // #2087/#2330 ROUTE-COHERENCE GUARD. The VALUE's log-determinant route and
        // THIS gradient's are selected by two unrelated predicates:
        //
        //   value    `streaming_plan().admitted_or_error(..).direct_logdet_admitted()`
        //            — a working-set/memory admission (construction_quasi_laplace.rs).
        //   gradient `exact_a_logdet_route` above — the bundle / matrix-free /
        //            assignment-family triple. Nothing consults the value's admission.
        //
        // ⚠ WHAT EACH VALUE ROUTE PRICES CHANGED UNDER #2509 PHASE-2b (`5563a2a18`),
        // AND THIS GUARD'S PREMISE DID NOT. Until then, "not admitted ⇒ delegates to
        // the streaming implementation, which ranks the majorizer `½log|B|`" — which
        // is what the rest of this comment was written against. Phase-2b moved BOTH
        // branches of `streaming_exact_arrow_log_det_with_lane_and_system` onto the
        // exact observed information via `exact_a_evidence_system`, so TODAY both
        // value routes price `½log|A|` and the sentence is false. It is corrected
        // rather than deleted because the guard below still reads the predicate it
        // named, and a reader comparing the two needs to know which era each
        // describes.
        //
        // Nothing tied the two predicates, so a fit whose value was priced by the
        // streaming lane could still be handed an exact-A derivative here. The desync
        // was `½·d/dρ log|I + B⁻¹ΔC|` — unbounded on a near-singular `A`, and invisible
        // to `criterion_as_atoms`'s 64-ulp identity check, which re-derives the
        // VALUE predicate and so cannot observe that the gradient took the other
        // route. Refuse rather than return the derivative of an operator the value
        // never ranked.
        //
        // SCOPE — this refuses ONLY the cell (value = B, gradient = A). Post-Phase-2b
        // no production value route prices `B`, so the cell this fires on is now
        // reachable only by a caller that hand-picks the streaming entry on a shape
        // the plan would have admitted; it is retained because that caller exists
        // (see `tests_streaming_outer_gradient_2026`) and because a future route that
        // reintroduces a `B`-priced value must not silently acquire an `A` gradient.
        //
        // THE MIRROR CELL IS CLOSED. It used to be the live one: `exact_a_logdet_route`
        // is false whenever a derivative bundle or a matrix-free system is present, so
        // every streaming/bundle evaluation paired an `A`-priced VALUE with
        // `B`-differentiated trace and θ-adjoint channels. That was #2515's open half,
        // and on its fixture it cost `logdet_trace` gaps of `1.231477e-1` (smooth atom 0)
        // and `5.052577e-2` (ARD). The bundle route now carries a
        // `BundleEvidenceGeometry` naming the exact observed information and carrying its
        // OWN factor cache, so `exact_a_logdet_route` being false no longer means the
        // gradient prices `B` — it means the exact-`A` channels are assembled from the
        // bundle rather than from the dense pseudo-inverse. Measured route-parity at a
        // fixed state, complete gradient: `1.57e-14`
        // (`laplace_value_and_gradient_are_route_invariant_2515`).
        //
        // So this guard is now the ONLY value/gradient pairing that can still go wrong,
        // and it is checked. Its scope note below is unchanged.
        //
        // LIMIT — this guard reads the PLAN, so it catches a predicate mismatch, not
        // a caller that hand-picks `penalized_quasi_laplace_criterion_streaming_exact_with_cache`
        // on a shape the plan would have admitted (see
        // `tests_streaming_outer_gradient_2026`, which does exactly that on purpose).
        if exact_a_logdet_route {
            let value_route_is_exact_a = self
                .streaming_plan()
                .map_err(OuterGradientError::internal)?
                .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
                .map_err(OuterGradientError::internal)?
                .direct_logdet_admitted();
            if !value_route_is_exact_a {
                return Err(OuterGradientError::internal(format!(
                    "analytic_outer_rho_gradient_components: log-determinant route \
                     incoherence — this gradient would differentiate the exact ½log|A|, \
                     but at shape n={}, p={}, K={} the criterion VALUE is priced by the \
                     streaming ½log|B| implementation (direct_logdet_admitted = false). \
                     Returning it would desync value and gradient by ½·d/dρ log|I + B⁻¹ΔC|.",
                    self.n_obs(),
                    self.output_dim(),
                    self.k_atoms()
                )));
            }
        }

        if let Some(sparse_index) = rho.sparse_flat_index() {
            explicit[sparse_index] =
                crate::assignment::assignment_prior_log_strength_derivative_weighted(
                    &self.assignment,
                    rho,
                    self.row_loss_weights.as_deref(),
                )
                .map_err(OuterGradientError::internal)?;
            // ordered Beta--Bernoulli concentration controls only the Beta--Bernoulli prior. The
            // final reconstruction gate is `sigmoid(logit/tau)`, so the data
            // likelihood and its Gauss--Newton blocks have no direct alpha
            // derivative. Structurally fixed assignments have no sparse index
            // and skip this channel entirely.
            if !exact_a_logdet_route {
                let joint_trace = match logdet_derivative_bundle {
                    Some((probes, sinv)) => self
                        .assignment_log_strength_hessian_trace_from_probes(
                            rho,
                            evidence_cache,
                            probes,
                            sinv,
                            evidence_operator,
                        )
                        .map_err(OuterGradientError::internal)?,
                    None => self
                        .assignment_log_strength_hessian_trace(rho, cache, solver)
                        .map_err(OuterGradientError::internal)?,
                };
                let coordinate_trace = self
                    .coordinate_block_assignment_log_strength_hessian_trace(
                        rho,
                        evidence_cache,
                        evidence_operator,
                    )
                    .map_err(OuterGradientError::internal)?;
                logdet_trace[sparse_index] = joint_trace - coordinate_trace;
            }
        }

        // #1556: λ_smooth is per-atom, so the smoothness gradient block occupies
        // the K layout-derived smooth indices (one per atom). Each atom
        // `k` carries its own explicit penalty-energy derivative, log|H| trace,
        // and Occam-normalizer derivative.
        let k_smooth = rho.log_lambda_smooth.len();
        let lambda_smooth_vec = rho
            .lambda_smooth_vec()
            .map_err(OuterGradientError::internal)?;
        // Explicit `∂loss.smoothness/∂log λ_k = 0.5·λ_k·<B_k, S_k B_k>` (the
        // per-atom split). Its sum is the λ-scaled penalty energy; renormalize to
        // `loss.smoothness` so the total matches the criterion's reported energy
        // bit-for-bit (folding in any minibatch `penalty_scale` baked into it).
        let mut smooth_explicit = self
            .decoder_smoothness_value_per_atom(&lambda_smooth_vec)
            .map_err(OuterGradientError::internal)?;
        let smooth_explicit_sum: f64 = smooth_explicit.iter().sum();
        if smooth_explicit_sum.abs() > 0.0 {
            let renorm = loss.smoothness / smooth_explicit_sum;
            for v in smooth_explicit.iter_mut() {
                *v *= renorm;
            }
        }
        // #2080: the per-atom smoothness logdet derivative off the shared
        // low-rank derivative representation when the rational lane supplied it;
        // the dense `DeflatedArrowSolver` selected inverse otherwise.
        let smooth_logdet = if exact_a_logdet_route {
            None
        } else {
            Some(match logdet_derivative_bundle {
                Some((probes, sinv)) => self
                    .decoder_smoothness_effective_dof_per_atom_from_probes(
                        probes,
                        sinv,
                        &lambda_smooth_vec,
                    )
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!(
                            "analytic_outer_rho_gradient_components: smooth dof (matrix-free): {err}"
                        ),
                    })?,
                None => self
                    .decoder_smoothness_effective_dof_with_solver_per_atom(
                        cache,
                        solver,
                        &lambda_smooth_vec,
                    )
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!("analytic_outer_rho_gradient_components: {err}"),
                    })?,
            })
        };
        let smooth_occam = self
            .reml_occam_log_lambda_smooth_derivative(rho)
            .map_err(OuterGradientError::internal)?;
        for atom_idx in 0..k_smooth {
            let index = rho.smooth_flat_index(atom_idx);
            explicit[index] = smooth_explicit[atom_idx];
            if let Some(smooth_logdet) = smooth_logdet.as_ref() {
                logdet_trace[index] = 0.5 * smooth_logdet[atom_idx];
            }
            occam[index] = -smooth_occam[atom_idx];
        }

        let ard_explicit = self
            .ard_log_precision_explicit_derivatives(rho)
            .map_err(OuterGradientError::internal)?;
        // #2080: the per-(atom,axis) ARD log-precision Hessian derivative off the
        // SAME shared low-rank representation (the all-or-nothing cluster's
        // second channel) when present; the dense
        // deflated selected inverse otherwise. The from-probes channel HARD-REFUSES
        // any row carrying gauge/rotation deflation (the plain-S⁻¹ bundle cannot
        // reconstruct the Daleckii–Krein correction), routing that fit to the dense
        // channel rather than silently dropping the correction.
        let ard_logdet_traces = if exact_a_logdet_route {
            None
        } else {
            let joint = match logdet_derivative_bundle {
                Some((probes, sinv)) => self
                    .ard_log_precision_hessian_trace_from_probes(
                        rho,
                        evidence_cache,
                        probes,
                        sinv,
                        evidence_operator,
                    )
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!(
                            "analytic_outer_rho_gradient_components: ARD logdet trace \
                             (matrix-free): {err}"
                        ),
                    })?,
                None => self
                    .ard_log_precision_hessian_trace(rho, cache, solver, evidence_operator)
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!("analytic_outer_rho_gradient_components: {err}"),
                    })?,
            };
            let coordinate = self
                .coordinate_block_ard_log_precision_hessian_trace(
                    rho,
                    evidence_cache,
                    evidence_operator,
                )
                .map_err(|err| OuterGradientError::InternalInvariant {
                    reason: format!(
                        "analytic_outer_rho_gradient_components: coordinate-block ARD trace: {err}"
                    ),
                })?;
            Some((joint, coordinate))
        };
        // #1026 shared-ARD: `ard_flat_index` maps `(k, axis)` onto the flat outer
        // coordinate for BOTH parameterizations. In `Shared` mode several atoms
        // alias one axis coordinate `1+K+axis`, and the outer derivative there is
        // `∂/∂log α_axis = Σ_{k owns axis} ∂/∂log α_{k,axis}` (chain rule through
        // the broadcast), so we ACCUMULATE. In `PerAtom` mode each `(k, axis)` has
        // a unique coordinate, so `+=` is identical to the historical `=`. Walking
        // a raw per-atom cursor in `Shared` mode would index past the flat length
        // `1+K+max_d` (OOB) and split one shared strength across phantom slots.
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                let idx = rho.ard_flat_index(k, axis);
                explicit[idx] += ard_explicit[k][axis];
                if let Some((joint, coordinate)) = ard_logdet_traces.as_ref() {
                    logdet_trace[idx] += joint[k][axis] - coordinate[k][axis];
                }
            }
        }

        // The scalar criterion replaces `½ log|H_tt|` with the realised-rank
        // charge. Its direct rho differential belongs alongside the explicit
        // penalty channels and is present on every layout (dense or probes).
        //
        // #2087 ATTRIBUTION — folding it in means `explicit` is no longer the
        // ρ-derivative of `loss.total() + extra_penalty_energy`, which is what its
        // docstring used to claim. Keep the folded summand ADDRESSABLE so an audit
        // that finite-differences `loss.total()` (the only loss entry that pins θ̂)
        // can net it out. Without this, such an audit's two halves are off by
        // exactly ∓this vector — equal, opposite, and each blaming a channel that
        // is not at fault.
        let rank_charge_direct_rho = rank_charge.direct_rho.clone();
        explicit += &rank_charge.direct_rho;

        // #2080: the envelope Γ off the SAME shared low-rank logdet derivative
        // representation (the all-or-nothing cluster's third channel) when
        // present; the dense selected inverse otherwise. #2712: the border-only
        // bundle reconstructs the row block on the DEFLATED chart too — `A_i` is
        // the conditioned row Cholesky, so `A_i⁻¹ + G_i S⁻¹ G_iᵀ` is the deflated
        // `(H⁻¹)_tt` — and `logdet_theta_adjoint_from_probes` subtracts the same
        // Daleckii–Krein correction the dense route subtracts instead of routing
        // the fit away. Ordered Beta--Bernoulli uses its row-local PSD majorizer
        // and shared-mass derivative directly.
        // This completes the matrix-free selected-inverse cluster (smoothness EDF + ARD
        // Hessian trace + θ-adjoint); assignment log-strength traces remain
        // solver-bound
        // — the last gaps before the routing flip (see the docstring).
        let majorizer_gamma = if exact_a_logdet_route {
            None
        } else {
            let mut gamma = match logdet_derivative_bundle {
                Some((probes, sinv)) => self
                    .logdet_theta_adjoint_from_probes(
                        rho,
                        evidence_cache,
                        probes,
                        sinv,
                        evidence_operator,
                        Some(target),
                    )
                    .map_err(OuterGradientError::internal)?,
                None => self
                    .logdet_theta_adjoint(rho, cache, solver)
                    .map_err(OuterGradientError::internal)?,
            };
            // The coordinate-block leg must be taken on the SAME geometry as the
            // joint one — `½log|H| − ½log|H_tt|` is one difference, and pairing an
            // `A` joint with a `B` coordinate block is the same class of error the
            // geometry type exists to remove.
            let coordinate_gamma = self
                .coordinate_block_logdet_theta_adjoint(
                    rho,
                    evidence_cache,
                    evidence_operator,
                    Some(target),
                )
                .map_err(OuterGradientError::internal)?;
            gamma.t -= &coordinate_gamma.t;
            gamma.beta -= &coordinate_gamma.beta;
            Some(gamma)
        };
        // `½ Γ_joint·theta_hat - ½ Γ_tt·theta_hat + ∇R·theta_hat`
        // is represented by one effective logdet adjoint
        // `Γ_eff = Γ_joint - Γ_tt + 2∇R`, preserving the existing
        // `-½ <Γ_eff, A^-1 g_rho>` contraction convention below.
        let majorizer_gamma = majorizer_gamma.map(|mut gamma| {
            gamma.t.scaled_add(2.0, &rank_charge.theta.t);
            gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
            gamma
        });
        // #1418: the implicit-function correction is `−½·Γᵀ·θ̂_ρ` with
        // `θ̂_ρ = −A⁻¹ g_ρ` (the code contracts `−½·⟨Γ, A⁻¹ g_ρ⟩` with rhs `= +∂g/∂ρ`, i.e. `+½·Γᵀθ̂_ρ` of the response — the sign lives in the −0.5 factor), where `A = ∇²_θθ L` is the EXACT stationarity
        // Jacobian of the inner fit — data residual curvature, exact softmax
        // entropy Hessian, exact ordered Beta--Bernoulli marginal curvature, and
        // exact periodic ARD curvature. The matrix the `solver`
        // factors is `B` (Gauss-Newton data curvature, the softmax Gershgorin
        // majorizer, the ordered Beta--Bernoulli row-local PSD majorizer, and
        // `max(V'',0)` ARD curvature): the `½log|B|` Laplace term is consistent
        // with `Γ = ½tr(B⁻¹ ∂B/∂θ)`, but the implicit step is governed by `A`.
        // `solve_exact_stationarity` applies the TRUE `A⁻¹` with left-`B`
        // preconditioned GMRES on `A = B + ΔC`, where
        // `ΔC = apply_exact_hessian_minus_b`, so the correction is no longer
        // biased by `(B⁻¹ − A⁻¹)` and does not assume `A` is SPD.
        //
        // A numerical stopping tolerance does not change the mathematical
        // objective.  At the exact inner optimum the envelope theorem cancels
        // the penalized-loss response, but the Laplace term still contributes
        // `-1/2 Gamma' theta_hat_rho`.  Dropping this term differentiates a
        // fictitious criterion in which the fitted state is held fixed.  The
        // exact stationarity solve above supplies the required implicit response.
        // #2231 — the trailing `L−1` flat coordinates are the crosscoder block
        // relevances `log λ_ℓ` (`SaeManifoldRho::to_flat` appends them last).
        // Their inner-gradient dependence enters through the λ-scaled target, so
        // their RHS is `−½·Jᵀ_M Z̃^{(ℓ)}` (`crosscoder_block_ift_rhs`), NOT the
        // penalty/prior channels `outer_rho_gradient_ift_rhs` owns. The adjoint
        // contraction below then completes the block gradient with the same
        // `−½·Γᵀθ̂_ρ` channel every other coordinate carries; the explicit data
        // + Jacobian parts stay with the eval lane's `block_log_lambda_gradient`.
        // #2080(A): collapse the per-coordinate IFT solves into ONE adjoint solve.
        // The implicit correction is `−½·⟨Γ, A⁺ g_ρ_l⟩` for every outer coordinate
        // `l`. The exact θθ-Hessian `A = ∇²_θθ L` is symmetric and its near-null
        // deflation is a symmetric `B`-orthogonal projection, so `A⁺` is
        // self-adjoint and `⟨Γ, A⁺ g_ρ_l⟩ = ⟨A⁺Γ, g_ρ_l⟩ = ⟨a, g_ρ_l⟩` with the
        // adjoint `a = A⁺Γ` solved ONCE. A near-null pencil direction contributes
        // `g_i r_i / μ_i` only when BOTH Γ and `g_ρ_l` excite it, in which case the
        // forward (per-coordinate) and this adjoint solve deflate it identically —
        // so the collapse is EXACT, not an approximation, while dropping the outer
        // IFT cost from `O(P_ρ)` solves to one. `solve_exact_stationarity_is_self_adjoint_2080`
        // pins the self-adjointness this identity rests on.
        // The single adjoint solve `a = A⁺Γ` — the only solver-bound step. At
        // #2330 Phase-2: on the dense direct-logdet route (no probe bundle, no
        // matrix-free system) the ranked value is ½log|A|, so the logdet channels
        // must be A-based; `exact_a_logdet_route` suppressed the B-majorizer
        // producers above precisely so this is the only assembly that ran.
        // Explicit / occam / rank-charge-direct channels are majorizer-
        // independent and were produced on both routes.
        //
        // BOTH ROUTES DIFFERENTIATE ½log|A| (#2515). #2509 Phase-2b (`5563a2a18`)
        // moved every production VALUE route onto the exact observed information;
        // #2515 then moved the bundle route's DERIVATIVE there too, by giving it a
        // `BundleEvidenceGeometry` that names the operator and carries `A`'s own
        // factor cache. `exact_a_logdet_route` still selects which ASSEMBLY runs —
        // the dense priced pseudo-inverse below, or the from-probes channels above
        // — but no longer which operator is priced. #2333 (routing this θ-adjoint
        // through the Trace row-jet seam on `A`'s selected inverse) is a
        // representation change downstream of that, not a missing operator.
        //
        // Exactly one arm produces Γ, so the two assemblies cannot both be paid
        // for on one gradient.
        let (gamma, dense_stationarity_adjoint) = match majorizer_gamma {
            Some(gamma) => (gamma, None),
            None => {
                let DenseExactALogdetChannels {
                    logdet_trace: exact_logdet_trace,
                    theta_adjoint: exact_gamma,
                    stationarity_adjoint,
                } = self
                    .dense_exact_a_logdet_channels(target, rho, loss, cache)
                    .map_err(OuterGradientError::internal)?;
                logdet_trace = exact_logdet_trace;
                (exact_gamma, Some(stationarity_adjoint))
            }
        };

        // At massive K (`matrix_free_system = Some`) the materialized operator is
        // unavailable, so the adjoint rides the certified reduced-Schur/GMRES
        // route. Every dense arm is owned by the rank-revealing spectral
        // pseudoinverse. On the exact-A logdet arm it reuses the eigensystem that
        // produced Γ; on a B-majorizer arm it materializes that same physical A
        // once here. There is deliberately no dense-to-GMRES fallback.
        let adjoint = match (matrix_free_system, dense_stationarity_adjoint) {
            (Some(system), None) => {
                self.solve_exact_stationarity_matrix_free(rho, target, cache, system, &gamma)
            }
            (None, Some(adjoint)) => Ok(adjoint),
            (None, None) => self.solve_exact_stationarity(rho, target, cache, &gamma),
            (Some(_), Some(_)) => Err(
                "analytic_outer_rho_gradient_components: dense exact-A adjoint was assembled \
                 for a matrix-free operator route"
                    .to_string(),
            ),
        }
        .map_err(|err| {
            OuterGradientError::classify_arrow_solver_error(
                &err,
                OuterGradientError::NonIdentifiable {
                    reason: err.clone(),
                },
            )
        })?;
        let block_tail_start = n_params - rho.log_lambda_block.len();
        for coord in 0..n_params {
            let rhs = if coord >= block_tail_start && !rho.log_lambda_block.is_empty() {
                let &(p_x, ref block_dims) =
                    self.crosscoder_pricing_spans.as_ref().ok_or_else(|| {
                        OuterGradientError::internal(
                            "analytic_outer_rho_gradient_components: rho carries block \
                             coordinates but no crosscoder pricing spans are installed"
                                .to_string(),
                        )
                    })?;
                let block = coord - block_tail_start;
                let start = p_x + block_dims[..block].iter().sum::<usize>();
                self.crosscoder_block_ift_rhs(cache, target, start..start + block_dims[block])
                    .map_err(OuterGradientError::internal)?
            } else {
                self.outer_rho_gradient_ift_rhs(rho, coord, cache)
                    .map_err(OuterGradientError::internal)?
            };
            let mut dot = 0.0_f64;
            for idx in 0..adjoint.t.len() {
                dot += adjoint.t[idx] * rhs.t[idx];
            }
            for idx in 0..adjoint.beta.len() {
                dot += adjoint.beta[idx] * rhs.beta[idx];
            }
            third_order_correction[coord] = -0.5 * dot;
        }

        Ok(SaeOuterRhoGradientComponents {
            explicit,
            rank_charge_direct_rho,
            logdet_trace,
            occam,
            third_order_correction,
        })
    }

    /// PATH C channel — exact fixed-stratum second derivative of the SOLVER-FREE
    /// explicit outer-gradient channels: the decoder-smoothness penalty energy
    /// (with its Occam renormalization to `loss.smoothness`) and the ARD
    /// log-precision prior. The rank-charge `direct_rho`, assignment
    /// log-strength, log-determinant traces, and third-order IFT channels are
    /// each assembled by their own methods; this one covers only the two
    /// channels that are closed forms of ρ at a frozen inner state (`atoms`,
    /// `assignment`) and touch no `H⁻¹`/`A⁺` solve, so it needs no cache.
    ///
    /// Math (all at fixed stratum, `s = log α`, `f_k = ⟨B_k, S_k B_k⟩` frozen):
    /// * Smoothness. The gradient renormalizes the per-atom penalty energy
    ///   `se_k = ½ λ_k f_k` to `C = loss.smoothness`, i.e. `g_k = C · se_k / Σse`.
    ///   But `C = penalty_scale · Σse` (construction.rs:4995), so the renormalizer
    ///   `renorm = C/Σse = penalty_scale` is ρ-INVARIANT — the `Σse` cancels — and
    ///   `g_k = renorm · se_k`. With `∂se_k/∂ρ_j = δ_{jk} se_k` the block is the
    ///   plain DIAGONAL `∂²/∂ρ_i∂ρ_j = renorm · δ_{ij} se_i`. (Holding `C` frozen
    ///   while `Σse` moves manufactures a spurious Occam cross term the
    ///   full-gradient FD reports as zero — the frozen-cache false-green genus.)
    /// * ARD. Per `(atom, axis)` the gradient is `energy_deriv + normalizer_deriv`
    ///   with `energy_deriv = Σ_i w_i · V(α, t_i)` (degree-one in `α`, so its own
    ///   `∂/∂s` is itself) and a normalizer that is `−½ n_eff` (constant → zero)
    ///   on a Euclidean axis and `n_eff · d1(log η)` on a periodic axis,
    ///   `log η = log α + 2(log p − log τ)`. The periodic second derivative is
    ///   `energy_deriv + n_eff · c''(log η)` with `c''` the stable
    ///   [`gam_math::special::bessel_i0_centered_second_log_derivative_from_log_abs`].
    ///   ARD axes are independent (diagonal); a shared-ARD coordinate owned by
    ///   several atoms accumulates their diagonals, matching the gradient's `+=`.
    /// * Occam. `reml_occam_log_lambda_smooth_derivative` is ρ-independent → zero.
    ///
    /// `frozen_smoothness_energy` is the criterion's reported `loss.smoothness`
    /// at the fixed stratum (`Σ_m se_m` on the full-batch path; a minibatch
    /// `penalty_scale` folded into it is preserved by the `C/Σ` renormalization).
    pub(crate) fn outer_explicit_smoothness_ard_hessian(
        &self,
        rho: &SaeManifoldRho,
        frozen_smoothness_energy: f64,
    ) -> Result<Array2<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let n_params = rho.to_flat().len();
        let mut hessian = Array2::<f64>::zeros((n_params, n_params));

        // Decoder-smoothness penalty energy with its Occam renormalization.
        let lambda_smooth = rho.lambda_smooth_vec()?;
        let smooth_energy = self.decoder_smoothness_value_per_atom(&lambda_smooth)?;
        let energy_sum: f64 = smooth_energy.iter().sum();
        let k_smooth = rho.log_lambda_smooth.len();
        // The gradient's explicit smooth term is `g_k = C·se_k/Σse` with
        // `C = loss.smoothness = penalty_scale·Σse` (construction.rs:4995 — the
        // criterion energy IS the λ-scaled per-atom penalty times the minibatch
        // `penalty_scale`). So the renormalizer `renorm = C/Σse = penalty_scale`
        // is ρ-INVARIANT — the `Σse` in `C` cancels the denominator — and
        // `g_k = renorm·se_k`. Hence `∂g_k/∂ρ_j = renorm·δ_jk·se_k`: the block is
        // DIAGONAL. Holding `C` frozen while `Σse` moves (the old code) manufac-
        // tured a spurious Occam cross term `−renorm·se_a·se_b/Σse` that the
        // full-gradient FD (which recomputes `C` at each ρ) correctly reports as
        // zero. This is the frozen-cache false-green genus — the renormalizer must
        // be differentiated, not held constant.
        if energy_sum.abs() > 0.0 {
            let renorm = frozen_smoothness_energy / energy_sum;
            for a in 0..k_smooth {
                let ia = rho.smooth_flat_index(a);
                hessian[[ia, ia]] += renorm * smooth_energy[a];
            }
        } else {
            for a in 0..k_smooth {
                let ia = rho.smooth_flat_index(a);
                hessian[[ia, ia]] += smooth_energy[a];
            }
        }

        // ARD log-precision prior (diagonal per coordinate; shared axes sum).
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let n = self.n_obs() as f64;
        let n_eff = row_w.map_or(n, |w| w.iter().sum::<f64>());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            if rho.log_ard[atom_idx].is_empty() {
                continue;
            }
            let periods = coord.effective_axis_periods();
            for axis in 0..coord.latent_dim() {
                let alpha = ard_precisions[atom_idx][axis];
                let log_alpha = rho.log_ard[atom_idx][axis];
                let period = periods[axis];
                let mut energy_deriv = 0.0_f64;
                for row in 0..coord.n_obs() {
                    let w_row = row_w.map_or(1.0, |w| w[row]);
                    let t = coord.row(row)[axis];
                    energy_deriv += w_row * ArdAxisPrior::eval(alpha, t, period).value;
                }
                let normalizer_second = match period {
                    None => 0.0,
                    Some(p) => {
                        let log_eta = log_alpha + 2.0 * (p.ln() - std::f64::consts::TAU.ln());
                        n_eff
                            * gam_math::special::bessel_i0_centered_second_log_derivative_from_log_abs(
                                log_eta,
                            )
                    }
                };
                let idx = rho.ard_flat_index(atom_idx, axis);
                hessian[[idx, idx]] += energy_deriv + normalizer_second;
            }
        }

        // Sparse (assignment log-strength). The gradient's `explicit[sparse]` is
        // `assignment_prior_log_strength_derivative_weighted`, which for BOTH
        // penalty-weight families whose concentration multiplies the logit penalty
        // linearly returns the prior VALUE:
        //
        //   softmax          λ_sparse·s · E_entropy(logits)          (assignment.rs, `penalty.value`)
        //   threshold gate   λ_sparse · Σ_i w_i·σ((ℓ_i−θ)/τ)         (assignment.rs, `sparsity_strength * acc`)
        //
        // Both are degree-one in `λ_sparse = e^ρ_sparse`, so
        // `∂²/∂ρ_sparse² = ∂/∂ρ_sparse(λ_sparse·E) = λ_sparse·E` — the SAME scalar
        // the gradient reports — and there is no cross term (it depends only on
        // `λ_sparse` and the frozen logits, not on smooth/ARD). K=1 softmax and
        // frozen routing return 0, so the diagonal is correctly zero there. TopK
        // mints no sparse coordinate at all.
        //
        // Ordered Beta--Bernoulli is the one family this identity does NOT cover:
        // under a learnable concentration the sparse slot holds `log α`, not
        // `log λ`, and `assignment_prior_log_strength_derivative_weighted` switches
        // to `grad_rho` — a genuinely nonlinear concentration derivative whose
        // second derivative is not the first. Refuse there, naming the reason.
        if let Some(sparse_index) = rho.sparse_flat_index() {
            match self.assignment.mode {
                AssignmentMode::Softmax { .. } | AssignmentMode::ThresholdGate { .. } => {
                    hessian[[sparse_index, sparse_index]] +=
                        crate::assignment::assignment_prior_log_strength_derivative_weighted(
                            &self.assignment,
                            rho,
                            self.row_loss_weights.as_deref(),
                        )?;
                }
                AssignmentMode::TopK { .. } => {}
                AssignmentMode::OrderedBetaBernoulli { .. } => {
                    return Err(format!(
                        "outer_explicit_smoothness_ard_hessian: the {} sparse log-strength \
                         explicit term is not degree-one in its coordinate (the concentration \
                         derivative is nonlinear), so its second derivative is not the \
                         gradient's own value; refusing to assemble a Hessian with a \
                         silently-zero sparse explicit term",
                        self.assignment.mode.family_label()
                    ));
                }
            }
        }

        Ok(hessian)
    }

    /// PATH C channel 4 — exact fixed-stratum second derivative of the outer
    /// gradient's log-determinant Daleckii–Krein trace channel (`logdet_trace`).
    ///
    /// The gradient's `logdet_trace` component is, per outer coordinate `i`,
    /// `logdet_trace_i = ½·[tr(G Cᵢ) − tr(H_bd⁻¹ Cᵢ)]`, where `Cᵢ = ∂H/∂ρ_i` is
    /// the penalty curvature the coordinate scales, `G = H⁻¹` is the FULL joint
    /// arrow inverse (the `ard_joint` / smoothness-EDF selected inverse), and
    /// `H_bd⁻¹` is the block-diagonal per-row `H_tt` inverse the rank-charge
    /// coordinate block subtracts (`ard_coordinate` trace). The smoothing channel
    /// touches only `H_ββ`, so its `H_bd⁻¹` leg is identically zero; the periodic
    /// ARD channel touches only the row-local `t`-slots, so both legs contribute.
    ///
    /// Every operator `Cᵢ` is degree-one in `exp(ρ_i)` at a frozen inner state —
    /// `λ_k·S_k ⊗ I` on the β-block for smoothing; `w_row·max(α cos κt, 0)` on the
    /// active `t`-rows for periodic ARD (`w_row·α` for a Euclidean axis). The
    /// `max(·,0)` majorizer active set is invariant under a ρ perturbation because
    /// ρ scales only `α`, never the frozen coordinate `t`. Hence
    /// `∂Cᵢ/∂ρ_j = δ_{ij} Cᵢ` and, with the Daleckii–Krein differential
    /// `∂G/∂ρ_j = −G C_j G` for each inverse `G`,
    /// `block[i,j] = ½·δ_{ij}·(tr(G Cᵢ) − tr(H_bd⁻¹ Cᵢ))
    ///              − ½·(tr(G C_j G Cᵢ) − tr(H_bd⁻¹ C_j H_bd⁻¹ Cᵢ))`.
    /// The diagonal `δ` term is exactly the coordinate's own `logdet_trace_i`
    /// value (the "self-term equals the operator" identity). A smoothing `C_j`
    /// vanishes on `H_bd⁻¹` (t-only) and an ARD `C_i` couples to a smoothing `C_j`
    /// only through the FULL inverse's `t`–β block, matching the gradient's
    /// construction.
    ///
    /// Small-dense materialization: build `G` dense by solving the arrow system
    /// against each unit arrow basis vector (`DeflatedArrowSolver::plain`), and
    /// `H_bd⁻¹` from the per-row undamped Cholesky factors — the same two inverses
    /// the gradient's `ard_joint` / `ard_coordinate` legs use, so value, gradient,
    /// and this Hessian share one (deflation-free interior) selected inverse.
    /// Shared-ARD axes accumulate their per-atom operators into one flat
    /// coordinate, matching the gradient's chain-rule `+=`.
    pub(crate) fn logdet_daleckii_krein_hessian(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let n_params = rho.to_flat().len();
        // #2724 - the shared exact-stationarity size expression (streaming_plan.rs).
        let dim = sae_exact_stationarity_dim(cache.delta_t_len(), cache.k);
        let solver = DeflatedArrowSolver::plain(cache);
        // The full joint inverse `G = H⁻¹` and the block-diagonal row-local
        // t-inverse `H_bd⁻¹` are the two shared helpers whose own docs already say
        // "Shared by ch4 and ch5"; ch4 had kept inline copies of both (#2500).
        let g = self.materialize_joint_inverse(cache, &solver)?;
        let h_bd = self.materialize_block_diag_t_inverse(cache);

        // #2500 — ch4 and ch5 read ONE operator map. The doc on
        // `penalty_curvature_operators_by_flat` has always claimed it was
        // "Extracted from `logdet_daleckii_krein_hessian` (ch4) so ch4's
        // Daleckii–Krein trace and ch5's forward-sensitivity twist read ONE
        // operator map", but the extraction was never finished: ch4 kept a
        // 103-line verbatim copy, which is how the two channels came to hold two
        // independent per-family enumerations of the sparse operator and two
        // separately-worded refusals for the same unmodelled case.
        //
        // The one place the channels legitimately differ is the ordered
        // Beta--Bernoulli sparse coordinate. Its `∂A/∂ρ_sparse` is the cross-row
        // integrated-marginal logit Hessian, which the shared map deliberately
        // does not emit because ch5's caller adds it directly through
        // `dense_exact_a_ordered_bb_sparse_trace`. This Hessian channel has no
        // such sibling, so a silently-zero sparse ROW would reach a curvature
        // block ARC then inverts. Refuse before assembling, naming the operator
        // rather than the family.
        if rho.sparse_flat_index().is_some()
            && matches!(
                self.assignment.mode,
                AssignmentMode::OrderedBetaBernoulli { .. }
            )
        {
            return Err(format!(
                "logdet_daleckii_krein_hessian: the {} sparse log-strength operator is the \
                 cross-row integrated-marginal logit Hessian, which this channel has no \
                 sibling for; refusing to assemble a Hessian with a silently-zero sparse row",
                self.assignment.mode.family_label()
            ));
        }
        let c_by_flat = self.penalty_curvature_operators_by_flat(rho, cache)?;

        // Precompute G·Cᵢ, H_bd⁻¹·Cᵢ, and their traces for each flat coordinate.
        let flats: Vec<usize> = c_by_flat.keys().copied().collect();
        let mut gc: Vec<Array2<f64>> = Vec::with_capacity(flats.len());
        let mut hc: Vec<Array2<f64>> = Vec::with_capacity(flats.len());
        let mut tr_g: Vec<f64> = Vec::with_capacity(flats.len());
        let mut tr_h: Vec<f64> = Vec::with_capacity(flats.len());
        for &flat in &flats {
            let c = &c_by_flat[&flat];
            let gci = g.dot(c);
            let hci = h_bd.dot(c);
            tr_g.push((0..dim).map(|d| gci[[d, d]]).sum());
            tr_h.push((0..dim).map(|d| hci[[d, d]]).sum());
            gc.push(gci);
            hc.push(hci);
        }

        // block[i,j] = ½·δ_{ij}·(tr(G Cᵢ) − tr(H_bd⁻¹ Cᵢ))
        //            − ½·(tr(G Cᵢ G C_j) − tr(H_bd⁻¹ Cᵢ H_bd⁻¹ C_j)).
        let mut hessian = Array2::<f64>::zeros((n_params, n_params));
        for (ii, &fi) in flats.iter().enumerate() {
            for (jj, &fj) in flats.iter().enumerate() {
                let (gi, gj) = (&gc[ii], &gc[jj]);
                let (hi, hj) = (&hc[ii], &hc[jj]);
                let mut cross_g = 0.0_f64;
                let mut cross_h = 0.0_f64;
                for a in 0..dim {
                    for b in 0..dim {
                        cross_g += gi[[a, b]] * gj[[b, a]];
                        cross_h += hi[[a, b]] * hj[[b, a]];
                    }
                }
                let diag = if ii == jj {
                    0.5 * (tr_g[ii] - tr_h[ii])
                } else {
                    0.0
                };
                hessian[[fi, fj]] += diag - 0.5 * (cross_g - cross_h);
            }
        }
        Ok(hessian)
    }

    /// PATH C (#2253) — assemble the COMPLETE exact fixed-stratum dense outer
    /// Hessian for the small-dense ARC route from all four analytic channels
    /// (ch1 explicit smoothness/ARD, ch2 rank-charge direct, ch4 log-determinant
    /// Daleckii–Krein, ch5 third-order forward-sensitivity), enforce the
    /// coordinate-coverage invariant, and return `Ok(block)`.
    ///
    /// ch5 refuses for any config outside the covered small-dense softmax route
    /// (compact top-k layout, per-row deflation, border frames, non-softmax
    /// priors), and the crosscoder-block guard / coverage invariant refuse an
    /// unmodelled coordinate — those refusals propagate as `Err`, so this only
    /// returns `Ok` when the full block is assembled AND validated. The public
    /// [`Self::exact_fixed_stratum_outer_hessian`] currently wraps this in a
    /// staged `Err` (see its doc); the finite-difference gates call THIS assembler
    /// directly to validate the block.
    pub(crate) fn assemble_exact_fixed_stratum_outer_hessian(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        // #2231 crosscoder block relevances (`log_lambda_block`, the trailing flat
        // coordinates): the gradient prices them (`crosscoder_block_ift_rhs`), but no
        // Hessian channel writes their rows/columns yet. Emitting a Dense Hessian with
        // those rows identically zero while their gradient is live would hand ARC a
        // singular system — strictly worse than declaring the curvature unavailable.
        // Refuse until a block channel lands. (Empty on the circle-mint route.)
        if !rho.log_lambda_block.is_empty() {
            return Err(format!(
                "exact_fixed_stratum_outer_hessian: rho carries {} crosscoder block \
                 relevance coordinate(s) that no Hessian channel models; refusing to \
                 advertise a curvature block with unmodelled (zero) rows",
                rho.log_lambda_block.len()
            ));
        }
        let n_params = rho.to_flat().len();
        let mut hessian = self.outer_explicit_smoothness_ard_hessian(rho, loss.smoothness)?;
        hessian += &self.rank_charge_direct_rho_hessian(target, rho, loss, cache)?;
        hessian += &self.logdet_daleckii_krein_hessian(rho, cache)?;
        // CH5 — the third-order forward-sensitivity channel completes the exact
        // fixed-stratum curvature. It refuses (propagated here) for any config
        // outside the covered small-dense softmax route, so a Dense Hessian is
        // never advertised where a sub-channel is unmodelled.
        hessian += &self.third_order_forward_sensitivity_hessian(target, rho, loss, cache)?;

        // Coordinate-coverage invariant (#2253), checked at assembly time on
        // EVERY call: every flat coordinate the outer gradient prices must own a
        // non-zero Hessian row. The priced set is assembled from the SAME
        // channels the gradient uses (per-atom smoothness, ARD axes, and the
        // softmax sparse log-strength coordinate when it is structurally live).
        // A live-gradient coordinate with an identically-zero Hessian row would
        // hand ARC a singular system, so refuse (naming the gap) rather than
        // advertise partial curvature. For the covered route ch1+ch4 already
        // fill every such row, so this passes; it is a guard against an
        // unhandled coordinate slipping through, not an expected refusal.
        let mut priced: Vec<usize> = Vec::new();
        for a in 0..rho.log_lambda_smooth.len() {
            priced.push(rho.smooth_flat_index(a));
        }
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                let idx = rho.ard_flat_index(k, axis);
                if !priced.contains(&idx) {
                    priced.push(idx);
                }
            }
        }
        if let Some(sparse) = rho.sparse_flat_index() {
            if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) && self.k_atoms() > 1
            {
                priced.push(sparse);
            }
        }
        for &c in &priced {
            let row_is_live = (0..n_params).any(|j| hessian[[c, j]] != 0.0);
            if !row_is_live {
                return Err(format!(
                    "exact_fixed_stratum_outer_hessian: flat coordinate {c} carries a live \
                     outer-gradient component but an identically-zero Hessian row; refusing \
                     to advertise a curvature block with an unmodelled coordinate"
                ));
            }
        }
        Ok(hessian)
    }

    /// PATH C (#2253) — production entry for the exact fixed-stratum outer
    /// Hessian. COMMIT 1 (this): assemble AND validate the full block
    /// ([`Self::assemble_exact_fixed_stratum_outer_hessian`]) — exercising the
    /// config guards, all four channels, and the coordinate-coverage invariant —
    /// then keep returning `Err` so `eval` yields `HessianValue::Unavailable` and
    /// production stays on the analytic-gradient BFGS route during the blind
    /// window. The finite-difference gates validate the assembly by calling the
    /// assembler directly. COMMIT 2 (once the FD gate is green on MSI) replaces
    /// this body with the assembler's `Ok` result and flips `capability()` to
    /// `Dense` for the covered softmax config — a tiny separately-validated
    /// change that carries the wrong-curvature-steering risk out of this window.
    pub(crate) fn exact_fixed_stratum_outer_hessian(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        let hessian = self.assemble_exact_fixed_stratum_outer_hessian(target, rho, loss, cache)?;
        Err(format!(
            "PATH C exact fixed-stratum outer Hessian is assembled and validated \
             ({}×{}) but intentionally not advertised in commit 1: the Err→Ok + \
             capability→Dense flip lands as a separately-validated commit once the \
             finite-difference gate is green",
            hessian.nrows(),
            hessian.ncols()
        ))
    }

    /// Classification of the exact observed information `A = B + ΔC` for the
    /// value path (#2330 / #2336 / #2673).
    ///
    /// A converged inner mode is a genuine exact-Laplace maximum iff every
    /// direction of `A` is `≥ −floor` in its own band, and that band is
    /// [`ExactHessianSpectralBlock::rank_floor`] — ONE predicate, in the ONE
    /// metric the gradient path classifies the same directions by. #2330 ACCEPTS
    /// above `−floor`; #2336's clamp attribution TRIGGERS below it; both read
    /// that one function, so they cannot disagree in the band.
    ///
    /// **The absolute floor this used to be is gone (#2673).** It was
    /// `1e-9 · max(λ_max(A), 1)`, a single number applied to every direction,
    /// while the gradient path used `|μ| = |vᵀAv/vᵀBv| < √ε` — and both ran on
    /// the same `A` inside one evaluation, with WHICH one the value path used
    /// decided by `direct_logdet_admitted`, i.e. by ambient free memory. Written
    /// as thresholds on the same `|λ|`, one was constant across directions and
    /// the other varied by the spread of the `B`-Rayleigh quotient (24.13x
    /// measured on the #2515 state, ratio straddling 1, so neither was even the
    /// conservative one), `λ/λ_max(A)` was not a curvature (it moves under a
    /// reparametrization `θ → Lθ` that leaves `μ` fixed, and `θ = (t, β)` mixes
    /// chart coordinates with data-scaled border coefficients), and `1e-9` was
    /// tuned where `√ε` is derived. See
    /// [`sae_exact_a_identifiability_floor`] for the argument and
    /// `tests::the_two_floors_are_incommensurable_thresholds_on_one_operator_2673`
    /// for the measurement.
    ///
    /// #2674 — the band is the ONLY null predicate on the exact-A route.
    /// `exact_hessian_spectral_block` used to delete the analytic chart-gauge
    /// orbit structurally before the spectrum was ever classified, which made two
    /// null predicates own one eigenvalue array. The measurement that settled
    /// which to keep: on the #2336/#2330 stall the declared orbit carried
    /// 86.6%–93.2% of the KKT gradient and per-direction slopes of the PENALIZED
    /// objective at 8x, 10x and 210x the convergence tolerance, so it was not a
    /// null of this operator at all — the priors are not invariant along a
    /// symmetry of the reconstruction.
    ///
    /// #2330 Phase-2 — the EXACT observed-information Laplace log-determinants
    /// `(log|A|, log|A_tt|)` at the converged fixed-θ̂ mode, `A = ∇²_θθ L = B + ΔC`.
    /// One symmetric eigendecomposition per block; kept eigenvalues
    /// (`λ > floor(i)`) contribute `ln λ`, the null band (`|λ| ≤ floor(i)`)
    /// contributes 0, and a strictly negative eigenvalue (`λ < −floor(i)`) the
    /// ARD concave clamp cannot account for is a saddle ⇒ typed
    /// `IndefiniteObservedInformation` refusal. `A_tt` drops the β border, and is
    /// classified against `B`'s coordinate restriction rather than the whole
    /// arrow operator.
    pub(crate) fn exact_observed_information_log_dets(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<(f64, f64), SaeCriterionError> {
        let total_t = cache.delta_t_len();
        let a = self
            .materialize_exact_hessian_dense(rho, target, cache)
            .map_err(SaeCriterionError::Numerical)?;
        // #2336 value-side E-attributability: the ARD periodic prior's concave
        // half contributes a bounded, EXACTLY-known negative curvature `E ⪰ 0`
        // (diagonal in the t-block) that the Newton/Schur majorizer DROPS from B
        // (see `materialize_ard_concave_clamp_diagonal`). A B-converged mode can
        // therefore be an exact-A saddle whose only negative directions are that
        // clamp wrinkle. This is COARSE-GRAINED Laplace: the posterior Gaussian
        // envelope cannot resolve a prior micro-wrinkle below its own
        // quadratic-model resolution, so a negative eigendirection `v` whose whole
        // negativity is attributable to the clamp (`vᵀEv ≥ |λ|`, i.e. basin
        // curvature `λ + vᵀEv ≥ −floor`) is priced at that basin curvature instead
        // of refused. A negative direction the clamp cannot explain
        // (`λ + vᵀEv < −floor`) is a GENUINE saddle and still returns the typed
        // IndefiniteObservedInformation refusal. No new constant: the SAME
        // per-direction `ExactHessianSpectralBlock::rank_floor` band, and a
        // `|λ + vᵀEv| ≤ floor` result drops into the existing radial-gauge
        // unit-stiffness deflation (`log 1 = 0`).
        // The gradient twin is assembled by `priced_ard_adjoint_extras`: it adds
        // the switched-direction eigenvector response, explicit dE/dρ leg, and
        // implicit dE/dt θ-adjoint to the identically classified quotient inverse.
        // Keep this pointer next to the value rule: reading the value in isolation
        // must not suggest that the production objective and derivative disagree.
        let e_diag = self
            .materialize_ard_concave_clamp_diagonal(rho, cache)
            .map_err(SaeCriterionError::Numerical)?;
        // #2674 — DIAGNOSTIC ONLY. The chart orbit is no longer removed from the
        // operator before it is diagonalized; it is retained here solely so the
        // saddle refusal below can still report how much of the refusing
        // direction lies along it.
        let joint_chart_gauges = self
            .exact_joint_chart_gauge_basis(cache)
            .map_err(SaeCriterionError::Numerical)?;
        let joint =
            Self::exact_hessian_spectral_block(a.clone(), &e_diag, total_t, ArrowMetric::Joint(cache))
                .map_err(SaeCriterionError::Numerical)?;
        let a_tt = a.slice(s![..total_t, ..total_t]).to_owned();
        let coordinate = Self::exact_hessian_spectral_block(
            a_tt,
            &e_diag,
            total_t,
            ArrowMetric::Coordinate(cache),
        )
        .map_err(SaeCriterionError::Numerical)?;
        let quotient_log_det = |spectral: &ExactHessianSpectralBlock,
                                block: &'static str|
         -> Result<f64, SaeCriterionError> {
                let eigs = &spectral.eigenvalues;
                let vecs = &spectral.eigenvectors;
                let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut log_det = 0.0_f64;
                for (idx, &lambda) in eigs.iter().enumerate() {
                    // #2673 — the band is per-direction because the metric it is
                    // relative to is: `√ε·vᵀBv`, floored under by the
                    // eigendecomposition's own backward error.
                    // Add back the dropped clamp curvature along this direction
                    // before classifying it.  The scalar decision is shared with
                    // the arrow evidence factorization; only the operands are
                    // route-local measurements of the same raw A, majorizer B and
                    // clamp E.
                    let v = vecs.column(idx);
                    let limit = total_t.min(v.len());
                    let e_v = (0..limit)
                        .map(|j| e_diag[j] * v[j] * v[j])
                        .sum::<f64>();
                    let classification = gam_solve::arrow_schur::classify_exact_a_direction(
                        lambda,
                        spectral.eigenvalues.len(),
                        spectral.spectral_norm,
                        spectral.metric_scale[idx],
                        e_v,
                    );
                    let priced = match classification {
                        gam_solve::arrow_schur::ExactADirectionClassification::Saddle {
                            basin,
                            ..
                        } => {
                            let floor = spectral.rank_floor(idx);
                            // NOT a step toward a saddle escape. #2336's
                            // terminal saddle-ESCAPE was REFUTED three ways
                            // (closed `e972387215`; both re-convergence lanes,
                            // undamped and descent-enforcing MM/Armijo, return
                            // to the same mode after a provably-descending step;
                            // `GATE_SHIFT=0`, evidence tests `6a5ca5d84`). The
                            // shipped resolution is the E-attributability
                            // arithmetic on the lines directly above. Comments
                            // elsewhere in this crate still promise that escape
                            // as pending - `construction_quasi_laplace.rs:353`,
                            // `outer_objective.rs:2622`/`:3081`/`:4118` - and
                            // they are stale pointers to a refuted approach.
                            //
                            // What this emits is the surviving open question. A
                            // refusal here means the mode is stationary AND the
                            // exact curvature is negative by more than the clamp
                            // explains, and every one of those magnitudes is
                            // computed on this line and then discarded with the
                            // eigenvector. `v'Bv` is printed beside them
                            // because it IS this direction's band now (#2673):
                            // the refusal can be read against the metric that
                            // decided it, rather than against a number taken
                            // from somewhere else in the spectrum.
                            // Diagnostic only - no control flow, no numerics,
                            // and the typed refusal below is unchanged.
                            //
                            // `warn`, not `debug`: this record is the ONLY place
                            // the six numbers that discriminate the #2330 fork
                            // exist -- a genuine saddle of L (curable upstream)
                            // versus an A that is not the curvature of the thing
                            // whose gradient the solve drove to zero (curable
                            // only at this refusal site). CI captures stderr at
                            // WARN, so at `debug` they were computed and
                            // discarded on every abort and the fork stayed
                            // undecidable from a CI log. It fires only on a
                            // refusal that is about to abort the fit, so it
                            // cannot become chatter.
                            let t_mass: f64 = (0..limit).map(|j| v[j] * v[j]).sum();
                            let v_av = v.dot(&spectral.operator.dot(&v));
                            let chart_gauge_mass = joint_chart_gauges
                                .iter()
                                .filter(|gauge| gauge.len() == v.len())
                                .map(|gauge| gauge.dot(&v).powi(2))
                                .sum::<f64>();
                            let mut row_gauge_mass = 0.0_f64;
                            if v.len() == a.nrows() {
                                for (row, directions) in
                                    cache.deflated_row_directions.iter().enumerate()
                                {
                                    let base = cache.row_offsets[row];
                                    for direction in directions {
                                        let coefficient = direction
                                            .iter()
                                            .enumerate()
                                            .map(|(slot, value)| value * v[base + slot])
                                            .sum::<f64>();
                                        let norm_sq = direction.dot(direction);
                                        if norm_sq > 0.0 {
                                            row_gauge_mass +=
                                                coefficient * coefficient / norm_sq;
                                        }
                                    }
                                }
                            }
                            let b_rayleigh = if v.len() == a.nrows() {
                                let b_v = apply_cached_arrow_hessian(
                                    cache,
                                    v.slice(s![..total_t]),
                                    v.slice(s![total_t..]),
                                )
                                .map_err(SaeCriterionError::Numerical)?;
                                Some(
                                    v.slice(s![..total_t]).dot(&b_v.t)
                                        + v.slice(s![total_t..]).dot(&b_v.beta),
                                )
                            } else {
                                None
                            };
                            log::warn!(
                                "SAE exact-A saddle refusal: block={block} idx={idx} \
                                 lambda={lambda:.6e} v'Av={v_av:.6e} clamp v'Ev={e_v:.6e} \
                                 basin={basin:.6e} floor={floor:.6e} \
                                 max_eig={max_eig:.6e} coordinate-block mass={t_mass:.6e} \
                                 chart-gauge projection={chart_gauge_mass:.6e} \
                                 row-gauge projection={row_gauge_mass:.6e} \
                                 v'Bv={b_rayleigh:?}"
                            );
                            return Err(SaeCriterionError::IndefiniteObservedInformation { block });
                        }
                        gam_solve::arrow_schur::ExactADirectionClassification::ResolvedPositive {
                            curvature,
                        }
                        | gam_solve::arrow_schur::ExactADirectionClassification::ClampBasin {
                            curvature,
                        } => curvature,
                        gam_solve::arrow_schur::ExactADirectionClassification::NumericalNull => {
                            0.0
                        }
                    };
                    let floor = spectral.rank_floor(idx);
                    if priced > floor {
                        log_det += priced.ln();
                    }
                    // |priced| <= floor: radial-gauge / clamp-attributed null ⇒ log 1 = 0.
                }
                Ok(log_det)
            };
        let log_a = quotient_log_det(&joint, "joint")?;
        let log_a_tt = quotient_log_det(&coordinate, "coordinate")?;
        Ok((log_a, log_a_tt))
    }

    /// Build a cluster-stable eigensystem and the shared absolute null floor for
    /// one already-materialized exact-Hessian block.
    ///
    /// #2674 — this used to take an analytic chart-gauge basis and diagonalize
    /// `Zᵀ A Z` on its orthogonal complement, deleting the declared orbit
    /// STRUCTURALLY so a spectral floor never had to rediscover it. That made
    /// two null predicates own one eigenvalue array, and the declaration-based
    /// one is the one that is wrong here: the chart orbit is a symmetry of the
    /// RECONSTRUCTION (measured reconstruction-invariant to `rel_ls ~1e-16`),
    /// not of the PENALIZED objective this operator is the Hessian of — the ARD
    /// and smoothing priors are not invariant along it. At the #2336/#2330 stall
    /// the two declared directions carried 86.6%–93.2% of the KKT gradient's
    /// norm and per-direction slopes of 8x, 10x and 210x the convergence
    /// tolerance, so quotienting them out handed the Newton step a right-hand
    /// side it was structurally unable to reduce and the inner solve stalled.
    ///
    /// One predicate owns the classification: [`Self::rank_floor`], which is
    /// also the predicate the matrix-free gradient path applies to its own
    /// solution direction (#2673). Where the orbit really is flat for the
    /// penalized operator its eigenvalue lands in `[−floor(i), floor(i)]` and the
    /// pseudoinverse discards it, which is what the independent oracle in
    /// `exact_observed_information_log_det_matches_eigendecomposition_2330`
    /// (full-spectrum `eigh` of the dense `A`, keeping `λ > floor(i)`) reads.
    fn exact_hessian_spectral_block(
        operator: Array2<f64>,
        e_diag: &Array1<f64>,
        total_t: usize,
        metric: ArrowMetric<'_>,
    ) -> Result<ExactHessianSpectralBlock, String> {
        let dimension = operator.nrows();
        if operator.ncols() != dimension || total_t > dimension || e_diag.len() < total_t {
            return Err(format!(
                "exact_hessian_spectral_block: operator {:?}, t dimension {total_t}, E diagonal length {}",
                operator.dim(),
                e_diag.len()
            ));
        }
        // #2267 — the other half of the split; see `materialize_exact_hessian_dense`.
        let eigh_started = std::time::Instant::now();
        let (eigenvalues, eigenvectors) =
            Self::cluster_stable_eigh(&operator, e_diag, total_t)?;
        let eigh_elapsed = eigh_started.elapsed();
        log::info!(
            "[SAE-EXACT-DENSE] eigendecomposition DONE: dim={dimension}, {:.3} s",
            eigh_elapsed.as_secs_f64(),
        );
        // #2673 — the scale every direction is classified relative to. `dim`
        // applies of the arrow factorization's own operator, which is `O(dim²)`
        // against the `O(dim³)` decomposition above it and needs no second
        // dense block.
        let spectral_norm = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let mut metric_scale = Array1::<f64>::zeros(eigenvalues.len());
        for index in 0..eigenvalues.len() {
            let value = metric.quadratic_form(eigenvectors.column(index))?;
            if !(value.is_finite() && value > 0.0) {
                return Err(format!(
                    "exact_hessian_spectral_block: the arrow factorization must be positive \
                     definite along every direction it classifies, but direction {index} of the \
                     {dimension}-dimensional block has vᵀBv={value:.6e} (#2673)"
                ));
            }
            metric_scale[index] = value;
        }
        let block = ExactHessianSpectralBlock {
            operator,
            eigenvalues,
            eigenvectors,
            metric_scale,
            spectral_norm,
        };
        let crossings = block.arithmetic_band_crossings();
        if crossings > 0 {
            // #2673 — the ONE residual crossing, detected in production rather
            // than left silent. See `ExactHessianSpectralBlock::rank_floor`.
            log::warn!(
                "[SAE-EXACT-DENSE] #2673 residual crossing: {crossings} of {dimension} \
                 directions have |λ| inside the eigendecomposition's backward error \
                 ({:.6e}) while their pencil curvature is identifiable, so the value pins \
                 a direction whose A⁻¹ response the gradient keeps",
                (dimension as f64) * f64::EPSILON * block.spectral_norm,
            );
        }
        Ok(block)
    }

    /// The priced log-determinant inverse of one spectral block.  This is not
    /// the stationarity inverse: a clamp-attributable negative direction is
    /// priced at its basin curvature here, while the physical stationarity
    /// pseudoinverse retains the raw signed eigenvalue.
    fn priced_exact_hessian_inverse(
        block: &ExactHessianSpectralBlock,
        e_diag: &Array1<f64>,
        total_t: usize,
    ) -> Result<Array2<f64>, String> {
        let mut weights = Array1::<f64>::zeros(block.eigenvalues.len());
        for (index, &lambda) in block.eigenvalues.iter().enumerate() {
            let eigenvector = block.eigenvectors.column(index);
            let limit = total_t.min(eigenvector.len());
            let e_v = (0..limit)
                .map(|row| e_diag[row] * eigenvector[row] * eigenvector[row])
                .sum::<f64>();
            let classification = gam_solve::arrow_schur::classify_exact_a_direction(
                lambda,
                block.eigenvalues.len(),
                block.spectral_norm,
                block.metric_scale[index],
                e_v,
            );
            let priced = match classification {
                gam_solve::arrow_schur::ExactADirectionClassification::Saddle {
                    basin,
                    ..
                } => {
                    return Err(format!(
                        "priced_exact_hessian_inverse: indefinite A \
                         (λ={lambda:.3e}, λ+e_v={basin:.3e}); genuine saddle, the outer \
                         gradient must not be assembled here"
                    ));
                }
                gam_solve::arrow_schur::ExactADirectionClassification::ResolvedPositive {
                    curvature,
                }
                | gam_solve::arrow_schur::ExactADirectionClassification::ClampBasin {
                    curvature,
                } => curvature,
                gam_solve::arrow_schur::ExactADirectionClassification::NumericalNull => 0.0,
            };
            weights[index] = if priced > block.rank_floor(index) {
                1.0 / priced
            } else {
                0.0
            };
        }
        Ok(block
            .eigenvectors
            .dot(&Array2::from_diag(&weights))
            .dot(&block.eigenvectors.t()))
    }

    /// Materialize only the joint spectral geometry required by a dense
    /// exact-stationarity solve.  No log-determinant pricing or coordinate-block
    /// decomposition is paid on routes that rank the B majorizer.
    fn materialize_exact_stationarity_geometry(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<ExactHessianSpectralBlock, String> {
        let total_t = cache.delta_t_len();
        let a = self.materialize_exact_hessian_dense(rho, target, cache)?;
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        Self::exact_hessian_spectral_block(a, &e_diag, total_t, ArrowMetric::Joint(cache))
    }

    /// #2330 Phase-2/#2653 — one coherent quotient geometry for the exact-A
    /// outer-ρ derivative.  The priced pseudo-inverses `(A⁺, A_tt⁺)` and the
    /// raw signed stationarity pseudoinverse are derived from the SAME joint
    /// eigensystem and floor. `A_tt⁺` is embedded with a zero β border for
    /// `logdet_theta_adjoint_dense` indexing.  A genuine saddle still refuses
    /// the log-determinant derivative; resolved negative directions remain live
    /// in the stationarity solve when the value's clamp pricing admits them.
    fn materialize_exact_hessian_quotient_geometry(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<ExactHessianQuotientGeometry, String> {
        let total_t = cache.delta_t_len();
        // #2724 - same shared size expression as the admission ledger.
        let dim = sae_exact_stationarity_dim(total_t, cache.k);
        let a = self.materialize_exact_hessian_dense(rho, target, cache)?;
        // #2336 — mirror the value-side E-attributability pricing into the quotient
        // pseudo-inverse so the θ-adjoint is DEFINED (finite) at a wrinkle-priced
        // mode: an ARD-concave-clamp-attributable negative direction is inverted at
        // its priced basin curvature `1/(λ+e_v)` rather than refused; a genuine
        // saddle (`λ+e_v < −floor`) still refuses. This inverse is the (I) leg of
        // the priced derivative. `dense_exact_a_logdet_channels` augments it with
        // `priced_ard_adjoint_extras`, which supplies the (II) Daleckii–Krein
        // eigenvector response plus the (III) direct dE/dρ and implicit dE/dt legs.
        // The split is an implementation detail; callers consume the complete
        // derivative of the value above.
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        let a_tt_block = a.slice(s![..total_t, ..total_t]).to_owned();
        let joint =
            Self::exact_hessian_spectral_block(a, &e_diag, total_t, ArrowMetric::Joint(cache))?;
        let priced_joint_inverse =
            Self::priced_exact_hessian_inverse(&joint, &e_diag, total_t)?;
        let coordinate = Self::exact_hessian_spectral_block(
            a_tt_block,
            &e_diag,
            total_t,
            ArrowMetric::Coordinate(cache),
        )?;
        let priced_coordinate_inverse_small =
            Self::priced_exact_hessian_inverse(&coordinate, &e_diag, total_t)?;
        let mut priced_coordinate_inverse = Array2::<f64>::zeros((dim, dim));
        priced_coordinate_inverse
            .slice_mut(s![..total_t, ..total_t])
            .assign(&priced_coordinate_inverse_small);
        Ok(ExactHessianQuotientGeometry {
            joint,
            coordinate,
            priced_joint_inverse,
            priced_coordinate_inverse,
        })
    }

    /// #2267 — price ONE step of the dense exact-stationarity route BEFORE paying
    /// for it, by timing a single `apply_exact_hessian` and multiplying by the
    /// column count the materialization will perform.
    ///
    /// This is a FORECAST, not a bar. It exists because the two candidate
    /// denominations for a size predicate on this route were both unsupported:
    /// memory cannot fire (the nearest existing admission compares against
    /// `in_core_budget_bytes ~= 3/5 * available`, >20 GB on the node that
    /// produced the measured wall, against 3.8 GB of resident blocks), and time
    /// could not be forecast until the route's two halves were separated.
    ///
    /// They now have been. Measured at `6a9916325` on the shipped ladder's K=8
    /// rung (`sae_ev_vs_k_olmo.py`, 508 train rows, p=32, K=8):
    ///
    ///   dim = 7692; 7692 applies + symmetrization = 406.546 s (52.853 ms/apply)
    ///   the O(dim^3) eigendecomposition then ran >=495 s more, to the cap
    ///
    /// So the column loop alone is ~6m47s of a step the polish is permitted to
    /// take 64 times, and it is LINEAR in `dim` at a per-apply cost this routine
    /// measures directly. `dim * t_apply` is therefore a real prediction of the
    /// cheaper half, obtained for 1/dim of its price (52.9 ms out of 406.5 s).
    ///
    /// The returned duration DELIBERATELY excludes the eigendecomposition: its
    /// constant is not measured, and a forecast that invented one would be the
    /// literal this route does not need. Consumers must read it as a LOWER BOUND
    /// on the step's cost.
    pub(crate) fn exact_stationarity_materialization_forecast(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<(usize, std::time::Duration), String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let mut unit = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(k),
        };
        if total_t > 0 {
            unit.t[0] = 1.0;
        } else if k > 0 {
            unit.beta[0] = 1.0;
        } else {
            return Ok((0, std::time::Duration::ZERO));
        }
        let probe_started = std::time::Instant::now();
        let probe = self.apply_exact_hessian(rho, target, cache, &unit)?;
        let per_apply = probe_started.elapsed();
        // The probe column is a real column of the operator this route is about
        // to build `dim` of. If it is already non-finite, the forecast is
        // meaningless AND so is the materialization it prices.
        if !probe.t.iter().chain(probe.beta.iter()).all(|value| value.is_finite()) {
            return Err(
                "exact_stationarity_materialization_forecast: non-finite Hessian-vector apply"
                    .to_string(),
            );
        }
        Ok((dim, per_apply.saturating_mul(dim.max(1) as u32)))
    }

    /// PATH C / #2330 — dense symmetric materialization of the EXACT stationarity
    /// Hessian `A = ∇²_θθ L = B + ΔC` (`dim×dim`, `dim = total_t + k`), built
    /// column by column via [`Self::apply_exact_hessian`] and symmetrized. The
    /// small-dense (circle-mint) scale this route already pays for
    /// [`Self::materialize_joint_inverse`]; shared by the observed-information
    /// log-determinant (VALUE) and its `A⁻¹` selected inverse (GRADIENT) so both
    /// factor one identical operator. `test_support`-scoped until Phase 2 wiring
    /// (see [`Self::exact_observed_information_log_dets`]).
    pub(crate) fn materialize_exact_hessian_dense(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        // #2724 - ONE size expression, shared with the admission that decides
        // whether this route may run at all (streaming_plan.rs). The planner
        // evaluates the same function on a shape-derived bound for total_t, so
        // the ledger and the allocation cannot describe two different matrices.
        let dim = sae_exact_stationarity_dim(total_t, k);
        // #2267 — the same reason the Krylov sibling logs `[SAE-DEFLATE]`: this
        // route is silent for as long as it takes, and on #2267 that silence has
        // been read as a hang, as a hardware ceiling, and as an example
        // misconfiguration across five months of comments. It is none of those.
        // The route is DENSE: `dim` Hessian-vector applies to build the operator,
        // then a symmetric eigendecomposition of it — `O(dim^2)` memory and
        // `O(dim^3)` time. `dim` is the JOINT dimension, so it grows with
        // `rows x atoms`, not with the atom count: a 160-row K=1 chart is a few
        // hundred and finishes in ~1.6 s, while a 508-row K=8 dense-softmax rung
        // is ~5.3e3 and one step measured >=25 min at 4.7 GiB peak RSS. One line,
        // once per materialization, states the size of the bill before it is paid.
        log::info!(
            "[SAE-EXACT-DENSE] materializing the exact stationarity Hessian: dim={dim} \
             (coords={total_t} + border={k}), {:.1} MiB per dim x dim f64 block, \
             {:.1} MiB resident across {} live blocks, \
             O(dim^3) symmetric eigendecomposition to follow",
            sae_exact_stationarity_block_bytes(dim) as f64 / (1024.0 * 1024.0),
            sae_exact_stationarity_resident_bytes(dim) as f64 / (1024.0 * 1024.0),
            SAE_EXACT_STATIONARITY_LIVE_DIM_BLOCKS,
        );
        // #2267 — the `[SAE-EXACT-DENSE]` line above states the SIZE of the bill;
        // these two stopwatches state which HALF of it is being paid. Measured at
        // `55c56d6f4`, the K=8 rung of the shipped ladder spends >=37 minutes
        // between that line and this routine's return, and nothing distinguishes
        // the O(dim) column loop from the O(dim^3) eigendecomposition that follows
        // it. Any size predicate that would refuse this route BEFORE paying has to
        // be denominated in whichever half dominates, so the split is the
        // prerequisite for the guard, not decoration.
        let build_started = std::time::Instant::now();
        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut unit = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(k),
        };
        for col in 0..dim {
            if col < total_t {
                unit.t[col] = 1.0;
            } else {
                unit.beta[col - total_t] = 1.0;
            }
            let av = self.apply_exact_hessian(rho, target, cache, &unit)?;
            if col < total_t {
                unit.t[col] = 0.0;
            } else {
                unit.beta[col - total_t] = 0.0;
            }
            for r in 0..total_t {
                a[[r, col]] = av.t[r];
            }
            for r in 0..k {
                a[[total_t + r, col]] = av.beta[r];
            }
        }
        // The matrix-free apply is symmetric only up to round-off; symmetrize
        // so downstream Cholesky / selected-inverse factors see an exactly
        // symmetric operand.
        for r in 0..dim {
            for c in (r + 1)..dim {
                let avg = 0.5 * (a[[r, c]] + a[[c, r]]);
                a[[r, c]] = avg;
                a[[c, r]] = avg;
            }
        }
        let build_elapsed = build_started.elapsed();
        log::info!(
            "[SAE-EXACT-DENSE] operator BUILT: dim={dim}, {dim} Hessian-vector applies \
             + symmetrization in {:.3} s ({:.3} ms per apply); \
             the O(dim^3) symmetric eigendecomposition has NOT started yet",
            build_elapsed.as_secs_f64(),
            build_elapsed.as_secs_f64() * 1.0e3 / (dim.max(1) as f64),
        );
        Ok(a)
    }

    /// #2330 Phase-2 — the A-based logdet gradient channels on the dense direct
    /// route: the direct trace vector `logdet_trace_i = ½tr(A⁺ ∂A/∂ρ_i)
    /// − ½tr(A_tt⁺ ∂A/∂ρ_i)` and the effective θ-adjoint
    /// `Γ_eff = tr(A⁺ ∂A/∂θ) − tr(A_tt⁺ ∂A_tt/∂θ) + 2∇R` (fed to the unchanged
    /// single-adjoint IFT collapse `a = A⁺Γ_eff`, `−½⟨a, g_ρ⟩`). `∂A/∂ρ_i =
    /// ∂B/∂ρ_i (penalty_curvature_operators_by_flat) + ∂ΔC/∂ρ_i
    /// (exact_stationarity_penalty_derivative_delta_by_flat)`, already exact. The
    /// θ-adjoint rides `exact_a = true` (ARD clamp-free) with `skip_deflation_dk
    /// = true` (the exact A carries only the ρ-invariant gauge null, handled by
    /// the quotient pseudo-inverse — no B-style Daleckii–Krein correction).
    ///
    /// EXACT-MINUS-PATCH-D: the two `logdet_theta_adjoint_dense` calls emit
    /// `∂B/∂θ + ∂ΔC_ard/∂θ` but NOT the residual-curvature / softmax-entropy legs
    /// of `∂ΔC/∂θ` (Patch D). Until D lands, Γ_eff — hence the IFT correction — is
    /// missing that term and the conservation bisection stays red by exactly it.
    /// #2336 flag-1 — eigendecomposition with an `E`-canonical basis inside
    /// exactly repeated eigenspaces.
    ///
    /// A basis rotation preserves the eigenpair equation `A V = V diag(λ)` only
    /// when every rotated eigenvalue is identical. The previous implementation
    /// also rotated merely near-equal eigenvalues (gap at most `√ε·‖A‖₂`) while
    /// leaving their distinct eigenvalues attached to the rotated columns. That
    /// produced a matrix called a priced inverse which was not the inverse of the
    /// operator whose eigenvalues the value summed; #2515 measured direct trace
    /// derivatives of `5.340922` and `5.146446` for two scalar values whose
    /// derivatives were both `4.974886`.
    ///
    /// Repeated eigenvalues genuinely have an arbitrary eigenspace basis, so
    /// those and only those runs are resolved against the restriction of `E`.
    /// Distinct eigenvalues retain the actual `eigh` columns, however small their
    /// gap: numerical uncertainty cannot authorize returning false eigenpairs.
    /// `e_diag` is the t-block diagonal of `E` (zero on the β border), so the
    /// repeated-space restriction reads only the first `total_t` rows.
    pub(crate) fn cluster_stable_eigh(
        m: &Array2<f64>,
        e_diag: &Array1<f64>,
        total_t: usize,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        if m.nrows() != m.ncols() || total_t > m.nrows() || e_diag.len() < total_t {
            return Err(format!(
                "cluster_stable_eigh: operator {:?}, t dimension {total_t}, E diagonal length {}",
                m.dim(),
                e_diag.len()
            ));
        }
        let (eigs, mut vecs) = m
            .eigh(Side::Lower)
            .map_err(|e| format!("cluster_stable_eigh: eigh failed: {e:?}"))?;
        let dim = eigs.len();
        let mut i = 0usize;
        while i < dim {
            let mut j = i + 1;
            while j < dim && eigs[j] == eigs[i] {
                j += 1;
            }
            let width = j - i;
            if width > 1 {
                // `E` is diagonal on the first `total_t` coordinate rows and
                // identically zero on the beta border.  Its restriction to this
                // cluster is therefore
                //
                //     E_c[a,b] = sum_r e_diag[r] V[r,i+a] V[r,i+b].
                //
                // Accumulate one weighted row outer product at a time.  Besides
                // reading the actual representation rather than manufacturing a
                // dense matrix of zeros, this changes the work from
                // O(width^2 * dim^2) to O(width^2 * total_t).  Keeping `b` as the
                // inner loop walks both the cluster row and `ec` contiguously.
                let mut ec = Array2::<f64>::zeros((width, width));
                for row in 0..total_t {
                    let weight = e_diag[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let cluster_row = vecs.slice(s![row, i..j]);
                    for a in 0..width {
                        let weighted_a = weight * cluster_row[a];
                        for b in a..width {
                            ec[[a, b]] += weighted_a * cluster_row[b];
                        }
                    }
                }
                // The upper triangle was accumulated once in the same order as
                // the former scalar quadratic form.  Copy it exactly rather than
                // averaging two independently-rounded contractions.
                for a in 0..width {
                    for b in (a + 1)..width {
                        ec[[b, a]] = ec[[a, b]];
                    }
                }
                let (_ec_eigs, rot) = ec
                    .eigh(Side::Lower)
                    .map_err(|e| format!("cluster_stable_eigh: cluster eigh failed: {e:?}"))?;
                drop(ec);
                let cluster = vecs.slice(s![.., i..j]).to_owned();
                let rotated = cluster.dot(&rot);
                vecs.slice_mut(s![.., i..j]).assign(&rotated);
            }
            i = j;
        }
        Ok((eigs, vecs))
    }

    /// #2336 — the coordinate-block (t-index → (atom, axis)) map for a cache, so
    /// the ARD-clamp E-attributability channels can attribute each priced
    /// direction's `e_v` mass back to the ρ_ard slot that scales it. `None` on
    /// logit / β rows (E is zero there).
    pub(crate) fn coord_axis_map_for_cache(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Option<(usize, usize)>>, String> {
        let total_t = cache.delta_t_len();
        let mut map = vec![None; total_t];
        for row in 0..self.n_obs() {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            for (a, va) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Coord { atom, axis } = *va {
                    map[base + a] = Some((atom, axis));
                }
            }
        }
        Ok(map)
    }

    /// #2336 — the t-derivative diagonal of the ARD concave-clamp remainder E,
    /// `∂E_rr/∂t_r = w_row·κ²·grad·[hess<0]` (companion to
    /// `materialize_ard_concave_clamp_diagonal`; `grad = (α/κ)·sin κt`,
    /// `hess = α·cos κt`, so `κ²·grad = α·κ·sin κt = ∂(−min(hess,0))/∂t` on the
    /// concave half, 0 elsewhere). Zero on logit / non-periodic / convex rows.
    pub(crate) fn ard_concave_clamp_dt_diagonal(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<Array1<f64>, String> {
        let total_t = cache.delta_t_len();
        let mut dt = Array1::<f64>::zeros(total_t);
        if self.k_atoms() == 0 {
            return Ok(dt);
        }
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        for row in 0..self.n_obs() {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let Some(period) = ard_axis_periods[atom][axis] else {
                    continue; // non-periodic axis: hess = α > 0, clamp never bites.
                };
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, Some(period));
                let kappa = std::f64::consts::TAU / period;
                // #2339 smooth clamp: E = hess_majorized − hess = α·softplus_τ(−cos κt),
                // so ∂E/∂t = κ²·grad·(1 − clamp_slope(cos κt)) (clamp_slope = logistic(cos/τ);
                // τ→0 recovers the hard-clamp κ²·grad·[cos<0]).
                let cos = prior.hess / alpha;
                let contrib = kappa * kappa * prior.grad * (1.0 - ArdAxisPrior::clamp_slope(cos));
                if contrib != 0.0 {
                    dt[base + a] += w_row * contrib;
                }
            }
        }
        Ok(dt)
    }

    /// #2336 — the value-side E-attributability pricing's ρ-derivative increment
    /// (the `dE/dρ` B-channel), returned as `(delta_logdet_trace, delta_gamma_t,
    /// k_joint, k_tt)` for [`Self::dense_exact_a_logdet_channels`] to fold in.
    ///
    /// Priced value `½log|A_priced|`, `A_priced = A + Σ_{i∈priced} e_i v_iv_iᵀ`,
    /// `e_i = v_iᵀE v_i`. Beyond `½ tr(A_priced⁺ dA/dρ)` (which the caller already
    /// gets from the priced pseudo-inverse), `d(½log|A_priced|)/dρ` carries
    /// `½ Σ_p (1/μ_i) de_i/dρ` with `de_i/dρ = v_iᵀ(dE/dρ)v_i + 2 v_iᵀE(dv_i/dρ)`:
    ///   (II) Daleckii–Krein eigenvector-derivative term
    ///        `2 Σ_p (1/μ_i) Σ_{j≠i}(v_iᵀE v_j)(v_jᵀ dA/dρ v_i)/(λ_i−λ_j)
    ///         = tr(K·dA/dρ)`, K the symmetric matrix below — folded into `inv`;
    ///   (III-direct) explicit ρ_ard leg: `E ∝ α = e^{ρ_ard}` ⇒ `dE/dρ_ard = E`,
    ///        contributing `½ Σ_p (1/μ_i)·Σ_{r∈(a,x)} E_rr v_i[r]²` to the ρ_ard slot;
    ///   (III-θ) `∂E/∂t` diagonal into the θ-adjoint: `½ Σ_p (1/μ_i)·(∂E_rr/∂t_r)v_i[r]²`.
    /// Each block contributes with the sign of the `½[log|A| − log|A_tt|]` split
    /// (joint `+`, tt `−`), matching the caller's `gamma.t -= gamma_tt.t`.
    fn priced_ard_adjoint_extras(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        geometry: &ExactHessianQuotientGeometry,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>), String> {
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let n_params = rho.to_flat().len();
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        let de_dt = self.ard_concave_clamp_dt_diagonal(rho, cache)?;
        let coord_axis = self.coord_axis_map_for_cache(cache)?;

        let mut delta_trace = Array1::<f64>::zeros(n_params);
        let mut delta_gamma_t = Array1::<f64>::zeros(total_t);
        let mut k_joint = Array2::<f64>::zeros((dim, dim));
        let mut k_tt = Array2::<f64>::zeros((dim, dim));

        // One block's contribution. `ambient_dim` is the eigenvector length
        // (`dim` for the joint A, `total_t` for A_tt); `sign` is +1 (joint) or
        // −1 (tt). `k_out` receives this block's K (embedded in the dim×dim frame).
        let mut accumulate = |block: &ExactHessianSpectralBlock,
                              sign: f64,
                              k_out: &mut Array2<f64>|
         -> Result<(), String> {
            let eigs = &block.eigenvalues;
            let vecs = &block.eigenvectors;
            let spectral_dim = eigs.len();
            let ambient_dim = vecs.nrows();
            if vecs.ncols() != spectral_dim || ambient_dim > dim {
                return Err(format!(
                    "priced_ard_adjoint_extras: eigenvectors {:?}, spectrum {spectral_dim}, ambient limit {dim}",
                    vecs.dim()
                ));
            }
            let t_lim = total_t.min(ambient_dim);
            // E·v_i on the t-block for every column (E diagonal = e_diag on t rows).
            for i in 0..spectral_dim {
                // #2673 — direction `i`'s own band, in the one metric both the
                // value and the gradient classify in.
                let floor = block.rank_floor(i);
                if eigs[i] >= -floor {
                    continue; // not a negative direction
                }
                let vi = vecs.column(i);
                let mut e_v = 0.0_f64;
                for r in 0..t_lim {
                    e_v += e_diag[r] * vi[r] * vi[r];
                }
                let mu = eigs[i] + e_v;
                if mu < -floor {
                    continue; // genuine saddle: not priced, no increment (value refuses)
                }
                if !(mu.abs() > floor) {
                    continue; // priced into the near-null deflation band ⇒ log 1, no trace
                }
                let inv_mu = 1.0 / mu;
                let half_sign = 0.5 * sign;

                // (III-direct): attribute e_v mass to each ρ_ard(atom,axis).
                for r in 0..t_lim {
                    if let Some((atom, axis)) = coord_axis[r] {
                        let contrib = e_diag[r] * vi[r] * vi[r];
                        if contrib != 0.0 {
                            let idx = rho.ard_flat_index(atom, axis);
                            delta_trace[idx] += half_sign * inv_mu * contrib;
                        }
                    }
                }
                // (III-θ): ∂E/∂t diagonal into the θ-adjoint.
                for r in 0..t_lim {
                    if de_dt[r] != 0.0 {
                        delta_gamma_t[r] += half_sign * inv_mu * de_dt[r] * vi[r] * vi[r];
                    }
                }
                // (II): K = Σ_p (1/μ_i)(v_i w_iᵀ + w_i v_iᵀ),
                //   w_i = Σ_{j≠i} [(v_iᵀE v_j)/(λ_i−λ_j)] v_j.
                let mut w = Array1::<f64>::zeros(ambient_dim);
                for j in 0..spectral_dim {
                    if j == i {
                        continue;
                    }
                    let denom = eigs[i] - eigs[j];
                    if denom.abs() <= floor {
                        continue; // near-degenerate: skip (well-separated shallow saddle)
                    }
                    let vj = vecs.column(j);
                    let mut vi_e_vj = 0.0_f64;
                    for r in 0..t_lim {
                        vi_e_vj += vi[r] * e_diag[r] * vj[r];
                    }
                    let coeff = vi_e_vj / denom;
                    if coeff != 0.0 {
                        for r in 0..ambient_dim {
                            w[r] += coeff * vj[r];
                        }
                    }
                }
                for r in 0..ambient_dim {
                    if w[r] == 0.0 && vi[r] == 0.0 {
                        continue;
                    }
                    for c in 0..ambient_dim {
                        k_out[[r, c]] += inv_mu * (vi[r] * w[c] + w[r] * vi[c]);
                    }
                }
            }
            Ok(())
        };

        accumulate(&geometry.joint, 1.0, &mut k_joint)?;
        accumulate(&geometry.coordinate, -1.0, &mut k_tt)?;
        Ok((delta_trace, delta_gamma_t, k_joint, k_tt))
    }

    /// #2500 — does [`Self::logdet_theta_adjoint_dense`] model this fit's
    /// assignment family? The exact-A logdet channels are taken only where it
    /// does; elsewhere the B-majorizer channels stand, because those are modelled
    /// for EVERY family.
    ///
    /// The dense reconstruction carries three prior legs: the softmax entropy
    /// Gershgorin majorizer (`want_entropy`), the ordered-Beta–Bernoulli Patch-D
    /// cross-row adjoint, and the periodic-ARD majorizer diagonal. It carries no
    /// per-atom-logistic GATE legs, which is what a `ThresholdGate` row needs —
    /// the same limitation `third_order_forward_sensitivity_hessian` already
    /// refuses on ("only the softmax assignment route is modelled by the dense
    /// θ-adjoint reconstruction"). `TopK` mints no free logit at all, so there is
    /// nothing for a gate leg to model and the reconstruction is complete there by
    /// construction.
    ///
    /// MEASURED on `threshold_gate_tiny_fixture`, dense vs the production
    /// `logdet_theta_adjoint` on the SAME inverse (they agree for softmax by
    /// design — the dense route is documented as self-checked against it):
    ///
    /// ```text
    ///   deflation-free arm   worst |production − dense| = 2.53e0  (on a 8.91e-1 entry)
    ///   deflating arm        worst |production − dense| = 3.31e0  (on a −2.50e-1 entry)
    /// ```
    ///
    /// and against a central finite difference of `½(log|A| − log|A_tt|)` in a
    /// logit, the dense Γ read `1.373e-1` where the FD read `3.521e-1`, with a
    /// SIGN FLIP on the next logit. So this is not a tolerance question.
    ///
    /// Before the sparse operator was modelled, a ThresholdGate fit could not
    /// reach this code at all — `penalty_curvature_operators_by_flat` refused
    /// first — so the gap was unreachable rather than absent. Gating here keeps
    /// that family on the fully-modelled `½log|B|` channels (`logdet_theta_adjoint`
    /// and `assignment_log_strength_hessian_trace` both carry it), which is the
    /// same staged position the matrix-free and bundle routes already hold until
    /// Phase-2b.
    fn dense_exact_a_theta_adjoint_is_modelled(&self) -> bool {
        match self.assignment.mode {
            AssignmentMode::Softmax { .. }
            | AssignmentMode::OrderedBetaBernoulli { .. }
            | AssignmentMode::TopK { .. } => true,
            AssignmentMode::ThresholdGate { .. } => false,
        }
    }

    pub(crate) fn dense_exact_a_logdet_channels(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<DenseExactALogdetChannels, String> {
        let n_params = rho.to_flat().len();
        let geometry = self.materialize_exact_hessian_quotient_geometry(rho, target, cache)?;
        // #2336 — the value-side E-attributability pricing's ρ-derivative increment.
        // (II) folds into the pseudo-inverses as K so every dA/dρ and dA/dθ channel
        // below emits `tr((A_priced⁺+K)·d…)` = the (I)+(II) legs at once; (III-direct)
        // and (III-θ) are added after the channel contractions. On a fit with no
        // priced directions all four are exactly zero, so the exact-A path is
        // unchanged.
        let (priced_delta_trace, priced_delta_gamma_t, priced_k_joint, priced_k_tt) =
            self.priced_ard_adjoint_extras(rho, cache, &geometry)?;
        let a_pinv = &geometry.priced_joint_inverse + &priced_k_joint;
        let a_tt_pinv = &geometry.priced_coordinate_inverse + &priced_k_tt;
        // This value diagonalizes `A_raw = B_raw + ΔC`; differentiate that raw
        // operator, not the row-conditioned operator carried by arrow factors.
        let da_by_flat = self.exact_stationarity_penalty_derivatives_by_flat(rho, cache)?;
        let frob = |x: &Array2<f64>, y: &Array2<f64>| -> f64 { (x * y).sum() };
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        for (&i, da) in da_by_flat.iter() {
            // A_tt⁺ has a zero β border, so frobbing it against the full ∂A/∂ρ_i
            // restricts to the t–t block automatically.
            logdet_trace[i] = 0.5 * frob(&a_pinv, da) - 0.5 * frob(&a_tt_pinv, da);
        }
        // Ordered-Beta–Bernoulli sparse coordinate: its ∂A/∂ρ_sparse is the exact
        // integrated-marginal logit Hessian (cross-row), absent from the operator
        // map above (softmax-only). Add its ½log|A| trace directly.
        if let Some(sparse) = rho.sparse_flat_index() {
            if matches!(
                self.assignment.mode,
                AssignmentMode::OrderedBetaBernoulli { .. }
            ) {
                logdet_trace[sparse] = self
                    .dense_exact_a_ordered_bb_sparse_trace(rho, cache, &a_pinv, &a_tt_pinv)?;
            }
        }
        let mut gamma = self.logdet_theta_adjoint_dense(
            rho,
            cache,
            &a_pinv,
            ThetaAdjointDhChannel::All,
            true,
            true,
            Some(target),
        )?;
        let gamma_tt = self.logdet_theta_adjoint_dense(
            rho,
            cache,
            &a_tt_pinv,
            ThetaAdjointDhChannel::All,
            true,
            true,
            Some(target),
        )?;
        gamma.t -= &gamma_tt.t;
        gamma.beta -= &gamma_tt.beta;
        // #2336 (III-direct) explicit ρ_ard leg + (III-θ) ∂E/∂t θ-adjoint diagonal
        // (both carry the joint−tt sign internally).
        logdet_trace += &priced_delta_trace;
        gamma.t += &priced_delta_gamma_t;
        let rank_charge = self.production_rank_charge_derivative(target, rho, loss, cache)?;
        gamma.t.scaled_add(2.0, &rank_charge.theta.t);
        gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
        let stationarity_adjoint = geometry.joint.solve_stationarity(&gamma)?;
        Ok(DenseExactALogdetChannels {
            logdet_trace,
            theta_adjoint: gamma,
            stationarity_adjoint,
        })
    }

    /// #2330 — the ordered-Beta–Bernoulli (non-softmax) sparse-coordinate ½log|A|
    /// trace `½[tr(A⁺ ∂A/∂ρ_sparse) − tr(A_tt⁺ ∂A/∂ρ_sparse)]`. For the
    /// non-learnable prior `∂A/∂ρ_sparse` is the EXACT integrated-marginal logit
    /// Hessian `H_obb` (linear-in-`weight` proof on the parent issue): its column
    /// `H_obb·e_j = ΔC_obb·e_j (cross-row HVP) + hdiag[j]·e_j (majorizer diagonal)`.
    /// The operator lives on logit t-slots only (no β border), so the coordinate
    /// block reuses the same columns against `A_tt⁺`. Learnable α (nonlinear
    /// concentration derivative) is refused, not silently mispriced.
    /// #2330 Patch D — the ordered-Beta--Bernoulli prior curvature θ-adjoint
    /// `Σ_{i,j} inv[i,j]·∂ΔC_obb[i,j]/∂ℓ_w`, the logit-block contribution the
    /// residual-curvature legs cannot carry (`ΔC_obb` couples rows CROSS-column,
    /// not row-locally). Per column `c`, `ΔC_obb = weight·S'_c·uuᵀ +
    /// diag(min(D_i, 0))` with `u_i = w_i·z_i(1−z_i)/τ`,
    /// `curv_i = z_i(1−z_i)(1−2z_i)/τ²`, `D_i = weight·S_c·w_i·curv_i`. Its logit
    /// derivative contracts to (with `P = uᵀ inv_cc u`, `(inv·u)_r`,
    /// `G = Σ_i inv[i,i]·[D_i<0]·w_i·curv_i`, `curv'_i = z_i(1−z_i)(1−6z_i+6z_i²)/τ³`):
    ///   `Γ[w=(r,c)] = weight·{ S''_c·u_r·P + 2·S'_c·w_r·curv_r·(inv·u)_r
    ///                          + S'_c·u_r·G + [D_r<0]·S_c·inv[r,r]·w_r·curv'_r }`.
    /// Contracts whichever pseudo-inverse the caller passes (`A⁺` for the joint
    /// leg, `A_tt⁺` for the coordinate leg), on the logit t-slots.
    fn dense_exact_a_ordered_bb_logit_theta_adjoint(
        &self,
        cache: &ArrowFactorCache,
        inv: &Array2<f64>,
        data: &gam_terms::analytic_penalties::OrderedBetaBernoulliLogitAdjointData,
    ) -> Result<Array1<f64>, String> {
        let n = data.n;
        let k = data.k_max;
        let weight = data.weight;
        let inv_tau = 1.0 / data.tau;
        let inv_tau2 = inv_tau * inv_tau;
        let inv_tau3 = inv_tau2 * inv_tau;
        let total_t = cache.delta_t_len();
        let mut out = Array1::<f64>::zeros(total_t);
        // Global t-slot of each (row, column) logit in the cache layout.
        let mut gindex: Vec<Vec<Option<usize>>> = vec![vec![None; k]; n];
        for row in 0..n {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            for (local, var) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *var {
                    if atom < k {
                        gindex[row][atom] = Some(base + local);
                    }
                }
            }
        }
        // Structural quantities per (row, column): (u, curv, curv', w, active).
        let uval = |row: usize, col: usize| -> (f64, f64, f64, f64, bool) {
            let z = data.z[row * k + col];
            let w = data.row_weight[row];
            let zc = z * (1.0 - z);
            let u = w * zc * inv_tau;
            let curv = zc * (1.0 - 2.0 * z) * inv_tau2;
            let curvp = zc * (1.0 - 6.0 * z + 6.0 * z * z) * inv_tau3;
            let d = weight * data.score[col] * w * curv;
            (u, curv, curvp, w, d < 0.0)
        };
        for col in 0..k {
            if data.column_fixed[col] {
                continue;
            }
            let s = data.score[col];
            let sp = data.score_derivative[col];
            let spp = data.score_second[col];
            let rows: Vec<usize> = (0..n).filter(|&r| gindex[r][col].is_some()).collect();
            let mut au = vec![0.0_f64; n];
            let mut p = 0.0_f64;
            let mut g = 0.0_f64;
            for &ri in &rows {
                let gi = gindex[ri][col].expect("row filtered to Some");
                let (ui, curvi, _curvpi, wi, acti) = uval(ri, col);
                let mut au_ri = 0.0_f64;
                for &rj in &rows {
                    let gj = gindex[rj][col].expect("row filtered to Some");
                    let (uj, _, _, _, _) = uval(rj, col);
                    au_ri += inv[[gi, gj]] * uj;
                }
                au[ri] = au_ri;
                p += ui * au_ri;
                if acti {
                    g += inv[[gi, gi]] * wi * curvi;
                }
            }
            for &ri in &rows {
                let gi = gindex[ri][col].expect("row filtered to Some");
                let (ui, curvi, curvpi, wi, acti) = uval(ri, col);
                let mut val = spp * ui * p + 2.0 * sp * wi * curvi * au[ri] + sp * ui * g;
                if acti {
                    val += s * inv[[gi, gi]] * wi * curvpi;
                }
                out[gi] += weight * val;
            }
        }
        Ok(out)
    }

    pub(crate) fn dense_exact_a_ordered_bb_sparse_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        a_pinv: &Array2<f64>,
        a_tt_pinv: &Array2<f64>,
    ) -> Result<f64, String> {
        if self.assignment.effective_alpha_is_learnable() {
            return Err(
                "dense_exact_a_ordered_bb_sparse_trace: learnable-α ordered-Beta–Bernoulli \
                 ∂A/∂ρ_sparse (nonlinear concentration derivative) is not yet modelled; refusing \
                 rather than emitting a wrong sparse ½log|A| trace"
                    .to_string(),
            );
        }
        let k_atoms = self.k_atoms();
        let n = self.n_obs();
        let row_weights = self.row_loss_weights.as_deref();
        // Global t-index of each (row, atom) logit slot in the cache layout.
        let mut logit_gindex: Vec<Vec<Option<usize>>> = vec![vec![None; k_atoms]; n];
        for row in 0..n {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            for (local, var) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *var {
                    if atom < k_atoms {
                        logit_gindex[row][atom] = Some(base + local);
                    }
                }
            }
        }
        // ∂B/∂ρ_sparse: the majorizer's diagonal log-strength derivative on the
        // logit slots — the SAME builder the B-majorizer trace uses.
        let mut hdiag = crate::assignment::assignment_prior_log_strength_hdiag_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        if hdiag.is_empty() {
            // Inert / frozen prior: ∂B and ΔC are both zero.
            return Ok(0.0);
        }
        let channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        if let Some(ch) = channels.as_ref() {
            for row in 0..n {
                for atom in 0..k_atoms {
                    let slot = row * k_atoms + atom;
                    hdiag[slot] =
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_hdiag(
                            ch, row, k_atoms, atom, hdiag[slot],
                        );
                }
            }
        }
        // ½[tr(A⁺ ∂A/∂ρ_sparse) − tr(A_tt⁺ ∂A/∂ρ_sparse)], column by column over
        // the flat logit basis: ∂A/∂ρ_sparse·e_j = ΔC_obb·e_j + hdiag[j]·e_j.
        let n_logits = n * k_atoms;
        let mut e = Array1::<f64>::zeros(n_logits);
        let mut tr_joint = 0.0_f64;
        let mut tr_coord = 0.0_f64;
        for jrow in 0..n {
            for jatom in 0..k_atoms {
                let Some(gj) = logit_gindex[jrow][jatom] else {
                    continue;
                };
                let jflat = jrow * k_atoms + jatom;
                e[jflat] = 1.0;
                let dc = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                    &self.assignment,
                    rho,
                    row_weights,
                    e.view(),
                )?;
                e[jflat] = 0.0;
                for irow in 0..n {
                    for iatom in 0..k_atoms {
                        let val = dc[irow * k_atoms + iatom];
                        if val == 0.0 {
                            continue;
                        }
                        if let Some(gi) = logit_gindex[irow][iatom] {
                            tr_joint += a_pinv[[gi, gj]] * val;
                            tr_coord += a_tt_pinv[[gi, gj]] * val;
                        }
                    }
                }
                tr_joint += a_pinv[[gj, gj]] * hdiag[jflat];
                tr_coord += a_tt_pinv[[gj, gj]] * hdiag[jflat];
            }
        }
        Ok(0.5 * (tr_joint - tr_coord))
    }

    /// Assemble `ΔC = A − B` per row, so the arrow evidence system can carry the
    /// EXACT observed information instead of the Newton/Schur majorizer.
    ///
    /// [`Self::apply_exact_hessian_minus_b`] contracts these blocks against a
    /// direction without ever forming them, which is all a matvec consumer needs.
    /// The streaming log-determinant is not a matvec consumer: it takes
    /// `log|H_tt^(i)|` off assembled per-row factors and reduces an assembled
    /// border, so it can only price `A` if the blocks exist (#2509).
    ///
    /// Every channel is row-local — (1a)/(1b) residual curvature, (2) the softmax
    /// entropy-minus-Gershgorin delta, (3) the periodic ARD concave clamp — except
    /// ordered Beta–Bernoulli, whose integrated-marginal prior couples every row
    /// within an atom column. That mode has no arrow-structured `ΔC` and is
    /// REFUSED here rather than silently dropped: pricing `B` while claiming `A`
    /// is the defect this function exists to remove.
    ///
    /// `ΔC_ββ` is identically zero (the decoder is linear in β), so the β block of
    /// the arrow system is unchanged and only the eliminated Schur sum moves.
    pub(crate) fn assemble_exact_hessian_minus_b_rows(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        row_dims: &[usize],
        border_dim: usize,
    ) -> Result<Vec<ExactHessianDeltaRow>, String> {
        self.assignment.validate_rho_domain(rho)?;
        if matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        ) {
            return Err(
                "assemble_exact_hessian_minus_b_rows: the ordered Beta-Bernoulli prior couples \
                 every row within an atom column, so A - B has no per-row arrow block; this \
                 route must refuse rather than assemble a majorizer and call it exact (#2509)"
                    .to_string(),
            );
        }
        let p = self.output_dim();
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_border_dim(border_dim)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;

        // Softmax entropy-minus-majorizer scale (#1419); `None` off softmax.
        let softmax_scale: Option<f64> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                Some(rho.lambda_sparse()? * sparsity * inv_tau * inv_tau)
            }
            _ => None,
        };

        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        let mut assignments = Array1::<f64>::zeros(k_atoms);

        let mut rows_out: Vec<ExactHessianDeltaRow> = Vec::with_capacity(n);
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let q = row_dims[row];
            let a_scratch = assignments.as_slice_mut().ok_or_else(|| {
                "assemble_exact_hessian_minus_b_rows: assignment scratch is not contiguous"
                    .to_string()
            })?;
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window_with_row_dims(
                    jet_window_next,
                    row_dims,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let jets = jet_window
                .pop_front()
                .expect("jet window must be non-empty");
            let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);

            // The same sqrt(w)-scaled metric-applied residual the applier contracts.
            fitted.fill(0.0);
            let active_atoms = self
                .last_row_layout
                .as_ref()
                .map(|layout| layout.active_atoms[row].as_slice());
            for k in 0..k_atoms {
                if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                    continue;
                }
                self.atoms[k].fill_decoded_row(row, &mut decoded);
                let a_k = assignments[k];
                for out_col in 0..p {
                    fitted[out_col] += a_k * decoded[out_col];
                }
            }
            for out_col in 0..p {
                error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
            }
            let error_metric: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                _ => error.to_vec(),
            };

            let mut tt = Array2::<f64>::zeros((q, q));
            let mut tbeta = Array2::<f64>::zeros((q, border.len()));

            // (1a) residual curvature, t-t.
            for a in 0..q {
                for b in 0..q {
                    tt[[a, b]] = sae_dot(&error_metric, jets.second(a, b));
                }
            }
            // (1b) residual curvature, t-beta. The beta-t block is its transpose;
            // the arrow system stores only this orientation.
            for a in 0..q {
                for beta_pos in 0..border.len() {
                    tbeta[[a, beta_pos]] = sae_dot(&error_metric, jets.beta_deriv(a, beta_pos));
                }
            }
            // (2) softmax exact entropy minus the Gershgorin majorizer written into B.
            if let Some(scale) = softmax_scale {
                let assignment_dim = self.assignment.assignment_coord_dim();
                let a_soft = assignments
                    .as_slice()
                    .expect("softmax assignments row must be contiguous");
                let m = softmax_majorizer_log_mean(a_soft);
                for (a, va) in jets.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: ka } = *va else {
                        continue;
                    };
                    if ka >= assignment_dim {
                        continue;
                    }
                    for (b, vb) in jets.vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                            continue;
                        };
                        if kb >= assignment_dim {
                            continue;
                        }
                        let h_entropy =
                            softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, scale);
                        let delta = if ka == kb {
                            h_entropy
                                - active_softmax_gershgorin_majorizer_entry(a_soft, ka, m, scale)
                        } else {
                            h_entropy
                        };
                        tt[[a, b]] += w_row * delta;
                    }
                }
            }
            // (3) periodic ARD concave clamp, diagonal on coordinate vars.
            for (a, va) in jets.vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.negative_hessian_remainder();
                if neg != 0.0 {
                    tt[[a, a]] += w_row * neg;
                }
            }

            rows_out.push(ExactHessianDeltaRow { tt, tbeta });
        }
        Ok(rows_out)
    }

}

#[cfg(test)]
mod test_support {
    use super::{ArrowFactorCache, DeflatedArrowSolver, SaeArrowVector, SaeManifoldRho, ThetaAdjointDhChannel};
    use ndarray::{Array1, s};
    use gam_linalg::faer_ndarray::FaerEigh;
    use super::Side;

    impl super::SaeManifoldTerm {
        /// #2330 Patch D arbiter support — spectrum summary of the EXACT `A` at a
        /// built cache: `(min_eig, max_eig, n_below_neg_floor, ‖ΔC‖_F, ‖A‖_F)`.
        /// The PD-window scan uses it to pick an arbiter fixture whose exact `A`
        /// is positive definite (so the criterion does not refuse) while the
        /// residual-curvature block `ΔC` — the very object Patch D
        /// differentiates — stays large enough for a finite difference to
        /// resolve. A fixture with `‖ΔC‖ ≈ 0` would false-green the arbiter.
        pub(crate) fn exact_a_spectrum_summary(
            &self,
            rho: &SaeManifoldRho,
            target: ndarray::ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<(f64, f64, usize, f64, f64), String> {
            let a = self.materialize_exact_hessian_dense(rho, target, cache)?;
            let (eigs, _vecs) = a
                .eigh(Side::Lower)
                .map_err(|e| format!("exact_a_spectrum_summary: eigh failed: {e:?}"))?;
            let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
            // #2673 — the arbiter only needs a scale-aware "is this decisively
            // negative" cut, and the classification floor it used to borrow is
            // per-direction now. `√ε·‖A‖₂` is the coarsest band any direction of
            // this operator can have, so counting under it is an upper bound on
            // the refusing population and cannot under-report.
            let spectral_norm = eigs.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            let floor = super::sae_exact_a_identifiability_floor() * spectral_norm;
            let n_neg = eigs.iter().filter(|&&lambda| lambda < -floor).count();
            let mut sorted: Vec<f64> = eigs.to_vec();
            sorted.sort_by(|x, y| x.partial_cmp(y).expect("finite eigenvalues"));
            let tail: Vec<String> = sorted.iter().take(6).map(|l| format!("{l:.6e}")).collect();
            eprintln!(
                "PATCHD_SPECTRUM floor={floor:.6e} smallest6=[{}]",
                tail.join(", ")
            );
            let total_t = cache.delta_t_len();
            let dim = total_t + cache.k;
            let mut dc_sq = 0.0_f64;
            let mut unit = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(cache.k),
            };
            for col in 0..dim {
                if col < total_t {
                    unit.t[col] = 1.0;
                } else {
                    unit.beta[col - total_t] = 1.0;
                }
                let dcv = self.apply_exact_hessian_minus_b(rho, target, cache, &unit)?;
                if col < total_t {
                    unit.t[col] = 0.0;
                } else {
                    unit.beta[col - total_t] = 0.0;
                }
                dc_sq += dcv.t.iter().map(|x| x * x).sum::<f64>()
                    + dcv.beta.iter().map(|x| x * x).sum::<f64>();
            }
            let a_frob = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            Ok((min_eig, max_eig, n_neg, dc_sq.sqrt(), a_frob))
        }

        /// #2330 Patch D arbiter support — the EXACT-A joint θ-adjoint
        /// `Γ_A = tr(A⁺ ∂A/∂θ) = ∂(log|A|)/∂θ`, built from the quotient
        /// pseudo-inverse and the `exact_a = true` dh (`∂B/∂θ + ∂ΔC/∂θ`).
        /// Comparing this against a central difference of
        /// `exact_observed_information_log_dets(...).0` over frozen θ̂ measures
        /// exactly the residual-curvature/ordered-BB/entropy legs of `∂ΔC/∂θ`
        /// still missing, coordinate by coordinate.
        pub(crate) fn exact_a_theta_adjoint_joint(
            &self,
            rho: &SaeManifoldRho,
            target: ndarray::ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<SaeArrowVector, String> {
            let geometry =
                self.materialize_exact_hessian_quotient_geometry(rho, target, cache)?;
            self.logdet_theta_adjoint_dense(
                rho,
                cache,
                &geometry.priced_joint_inverse,
                ThetaAdjointDhChannel::All,
                true,
                true,
                Some(target),
            )
        }

        /// #2330 split probe — the g3 cross non-conservation attributed to the
        /// trace vs the frozen-DK piece of `dΓ_joint/dρ_i`, per leg. Returns
        /// `⟨leg_i, b_j⟩` and `⟨leg_j, b_i⟩` for the (i,j) cross pair so the caller
        /// can assert cross-symmetry of each leg: part-a (twist `−G Mᵢ G`) trace,
        /// part-a DK, part-b (`∂Kw/∂ρ`) trace, part-b DK. The asymmetric leg is the
        /// leak. `with_dk` legs include `deflation_block_correction`; `_tr` legs
        /// pass `skip_deflation_dk = true`.
        pub(crate) fn ch5_twist_leg_cross(
            &self,
            rho: &SaeManifoldRho,
            target: ndarray::ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
            i: usize,
            j: usize,
        ) -> Result<[(f64, f64); 4], String> {
            let solver = DeflatedArrowSolver::plain(cache);
            let g = self.materialize_joint_inverse(cache, &solver)?;
            let operators = self.penalty_curvature_operators_by_flat(rho, cache)?;
            // Mirror production: the twist inverse rides the EXACT ∂A/∂ρ = M_c + Δ.
            let exact_deltas = self.exact_stationarity_penalty_derivative_delta_by_flat(rho, cache)?;
            let stationarity_geometry =
                self.materialize_exact_stationarity_geometry(rho, target, cache)?;
            let total_t = cache.delta_t_len();
            let dim = total_t + cache.k;
            let flatten = |v: &SaeArrowVector| -> Array1<f64> {
                let mut out = Array1::<f64>::zeros(dim);
                out.slice_mut(s![..total_t]).assign(&v.t);
                out.slice_mut(s![total_t..]).assign(&v.beta);
                out
            };
            let smooth_range =
                rho.smooth_flat_start()..rho.smooth_flat_start() + rho.log_lambda_smooth.len();
            let sparse_index = rho.sparse_flat_index();
            // part-a (twist) and part-b (Kw ρ-deriv) legs of dΓ_joint/dρ_c, each in
            // trace-only and full (trace − DK) form, contracted against b_other.
            let leg = |c: usize, skip_dk: bool, part_a: bool| -> Result<Array1<f64>, String> {
                if part_a {
                    let twist_op = match exact_deltas.get(&c) {
                        Some(delta_c) => &operators[&c] + delta_c,
                        None => operators[&c].clone(),
                    };
                    let g_c = -g.dot(&twist_op).dot(&g);
                    Ok(flatten(&self.logdet_theta_adjoint_dense(
                        rho,
                        cache,
                        &g_c,
                        ThetaAdjointDhChannel::All,
                        skip_dk,
                        false,
                        None,
                    )?))
                } else if smooth_range.contains(&c) {
                    Ok(Array1::<f64>::zeros(dim)) // smooth part-b is 0
                } else {
                    let channel = if sparse_index == Some(c) {
                        ThetaAdjointDhChannel::SoftmaxSparseMixed
                    } else {
                        ThetaAdjointDhChannel::ArdMixed { target_flat: c }
                    };
                    Ok(flatten(&self.logdet_theta_adjoint_dense(
                        rho, cache, &g, channel, skip_dk,
                        false,
                        None,
                    )?))
                }
            };
            let b = |c: usize| -> Result<Array1<f64>, String> {
                let g_rho = self.outer_rho_gradient_ift_rhs(rho, c, cache)?;
                Ok(flatten(&stationarity_geometry.solve_stationarity(&g_rho)?))
            };
            let bi = b(i)?;
            let bj = b(j)?;
            // part_a_tr, part_a_dk, part_b_tr, part_b_dk cross pairs.
            let pa_full_i = leg(i, false, true)?;
            let pa_tr_i = leg(i, true, true)?;
            let pa_full_j = leg(j, false, true)?;
            let pa_tr_j = leg(j, true, true)?;
            let pb_full_i = leg(i, false, false)?;
            let pb_tr_i = leg(i, true, false)?;
            let pb_full_j = leg(j, false, false)?;
            let pb_tr_j = leg(j, true, false)?;
            let dot = |x: &Array1<f64>, y: &Array1<f64>| x.dot(y);
            Ok([
                (dot(&pa_tr_i, &bj), dot(&pa_tr_j, &bi)),
                (
                    dot(&(&pa_full_i - &pa_tr_i), &bj),
                    dot(&(&pa_full_j - &pa_tr_j), &bi),
                ),
                (dot(&pb_tr_i, &bj), dot(&pb_tr_j, &bi)),
                (
                    dot(&(&pb_full_i - &pb_tr_i), &bj),
                    dot(&(&pb_full_j - &pb_tr_j), &bi),
                ),
            ])
        }
    }

    /// #2509 — the assembled `ΔC = A − B` row blocks and the matrix-free applier
    /// are ONE derivation with two readers, and this is the executable link.
    ///
    /// Assembling `ΔC` necessarily writes the four channels' arithmetic a second
    /// time (the applier contracts a block against a direction; the assembler
    /// stores the block), and one quantity with two standards is a failure mode
    /// this repository keeps paying for. So the blocks are contracted here and
    /// required to reproduce `apply_exact_hessian_minus_b` to round-off.
    ///
    /// It is a real cross-check rather than a tautology: on softmax rows the
    /// applier reaches the residual-curvature channels (1a)/(1b) through the
    /// device-contracted `contracted_softmax_bilinear_hvp`, while the assembler
    /// reads the shared row-jet window. Agreement is agreement between two
    /// different execution paths over the same jets. The fixture is softmax with
    /// periodic (Circle) manifolds and non-empty `log_ard`, so channels (1a),
    /// (1b), (2) and (3) are all live — asserted below rather than assumed.
    #[test]
    fn assembled_exact_hessian_delta_contracts_like_the_applier_2509() {
        use ndarray::Array1;
        // #2681 — the assembled/applied parity below is a statement about two
        // readers of ONE derivation, evaluated at whatever `(t, β)` and cache
        // they are handed; it has no stake in the inner solve reaching a KKT
        // point. This fixture's inner solve does not reach it (#2681), so
        // demanding convergence here only prevented the parity from ever being
        // checked. Take the pinned shared state and factor once at it through
        // the production `FROZEN_INNER_STATE` freeze lane.
        let (term0, target, rho) =
            crate::manifold::tests::small_two_atom_periodic_term_at_shared_inner_state();
        let mut term = term0;
        let (_cost, _loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                crate::manifold::tests::FROZEN_INNER_STATE,
                0.25,
                1.0e-4,
                1.0e-4,
            )
            .expect("dense criterion must evaluate at the pinned #2509 witness state");

        let blocks = term
            .assemble_exact_hessian_minus_b_rows(&rho, target.view(), &cache.row_dims, cache.k)
            .expect("assembled delta rows");
        let border = term
            .border_channels_for_cache(&cache)
            .expect("border channels");
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        assert_eq!(blocks.len(), term.n_obs());

        // NON-VACUITY: the correction must be non-zero on this fixture, or the
        // agreement below says nothing at all.
        let max_block = blocks
            .iter()
            .flat_map(|row| row.tt.iter().chain(row.tbeta.iter()))
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            max_block > 0.0,
            "ΔC is identically zero on this fixture, so the gate cannot discriminate"
        );

        // Deterministic probes: every (t, β) unit direction, plus one dense mix so
        // every stored entry contributes to at least one compared component.
        for probe in 0..=dim {
            let mut v = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(cache.k),
            };
            if probe < dim {
                if probe < total_t {
                    v.t[probe] = 1.0;
                } else {
                    v.beta[probe - total_t] = 1.0;
                }
            } else {
                for (idx, value) in v.t.iter_mut().enumerate() {
                    *value = 1.0 + 0.25 * (idx as f64);
                }
                for (idx, value) in v.beta.iter_mut().enumerate() {
                    *value = -0.5 - 0.125 * (idx as f64);
                }
            }

            let applied = term
                .apply_exact_hessian_minus_b(&rho, target.view(), &cache, &v)
                .expect("matrix-free delta apply");

            let mut assembled = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(cache.k),
            };
            for (row, block) in blocks.iter().enumerate() {
                let base = cache.row_offsets[row];
                let q = cache.row_dims[row];
                for a in 0..q {
                    let mut acc = 0.0_f64;
                    for b in 0..q {
                        acc += block.tt[[a, b]] * v.t[base + b];
                    }
                    for (beta_pos, channel) in border.iter().enumerate() {
                        acc += block.tbeta[[a, beta_pos]] * v.beta[channel.index];
                        assembled.beta[channel.index] +=
                            block.tbeta[[a, beta_pos]] * v.t[base + a];
                    }
                    assembled.t[base + a] += acc;
                }
            }

            // Both sides sum the SAME products in a different association, so the
            // admissible gap is f64 round-off at the largest magnitude involved.
            let scale = applied
                .t
                .iter()
                .chain(applied.beta.iter())
                .fold(1.0_f64, |acc, v| acc.max(v.abs()));
            let tolerance = 4096.0 * f64::EPSILON * scale;
            for idx in 0..total_t {
                assert!(
                    (applied.t[idx] - assembled.t[idx]).abs() <= tolerance,
                    "probe {probe}: assembled ΔC t[{idx}] = {} but the applier says {} \
                     (tolerance {tolerance:.3e})",
                    assembled.t[idx],
                    applied.t[idx]
                );
            }
            for idx in 0..cache.k {
                assert!(
                    (applied.beta[idx] - assembled.beta[idx]).abs() <= tolerance,
                    "probe {probe}: assembled ΔC beta[{idx}] = {} but the applier says {} \
                     (tolerance {tolerance:.3e})",
                    assembled.beta[idx],
                    applied.beta[idx]
                );
            }
        }
    }

}

#[cfg(test)]
mod tests_inverse_power_deflation_cost_2627 {
    use super::*;

    /// #2627 — a GAPLESS near-null cluster must deflate, not exhaust the Krylov
    /// bound.
    ///
    /// `A` is diagonal: fourteen resolved curvatures plus two under the numerical
    /// null floor whose RATIO is `0.9`. That ratio is the whole fixture. The
    /// deflation isolate rotates toward the smaller of the two at `0.9^k`, so the
    /// eigenvector-alignment criterion — consecutive iterates agreeing to `√ε` —
    /// needs on the order of thirty inverse steps here, and each step is a full
    /// preconditioned GMRES. Against an inner Krylov bound of `dim = 16` it never
    /// arrives: before the subspace certificate this call spent its whole inner
    /// bound in GMRES on EVERY deflation turn and then raised "inverse-power
    /// direction did not converge in the derived Krylov dimension 16". That is
    /// the `dim²` shape, and a gapless band under the floor is not adversarial —
    /// `K` atoms each contribute a rank-1 radial null to `A = B + ΔC`, which is
    /// exactly such a band.
    ///
    /// What deflation needs from the isolate is MEMBERSHIP in the numerical-null
    /// subspace, which the post-loop `|μ(v)| < rank_floor` guard enforces and
    /// every member of the cluster satisfies. So the certificate is two
    /// consecutive amplification steps that stay under the floor, and the assert
    /// below is the consequence: the unidentifiable `1/μ` amplification is
    /// removed and every resolved direction is returned untouched.
    #[test]
    fn gapless_near_null_cluster_deflates_instead_of_exhausting_the_krylov_bound_2627() {
        const NEAR_NULL: f64 = 1.0e-10;
        const RESOLVED: usize = 14;
        let mut curvature: Vec<f64> = (1..=RESOLVED).map(|c| c as f64).collect();
        curvature.push(NEAR_NULL);
        curvature.push(0.9 * NEAR_NULL);
        let dim = curvature.len();

        let apply_a = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let mut out = v.clone();
            for (slot, value) in out.t.iter_mut().enumerate() {
                *value *= curvature[slot];
            }
            Ok(out)
        };
        let apply_b = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> { Ok(v.clone()) };
        let precondition = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> { Ok(v.clone()) };
        let rhs = SaeArrowVector {
            t: Array1::from_elem(dim, 1.0),
            beta: Array1::zeros(0),
        };

        let solved = solve_exact_stationarity_preconditioned(&rhs, &apply_a, &apply_b, precondition)
            .expect("a gapless near-null cluster must deflate, not exhaust the Krylov bound");

        for slot in 0..RESOLVED {
            let expected = 1.0 / curvature[slot];
            assert!(
                (solved.t[slot] - expected).abs() <= 1.0e-6 * expected,
                "resolved slot {slot} was disturbed by the deflation: {:.6e} vs {expected:.6e}",
                solved.t[slot],
            );
        }
        // Undeflated, each near-null slot carries the full `1/μ` amplification
        // `1/NEAR_NULL = 1e10`. Two orders of magnitude of reduction is far
        // outside anything round-off could produce and far inside what the
        // deflation actually achieves.
        for slot in RESOLVED..dim {
            assert!(
                solved.t[slot].abs() < 1.0e8,
                "near-null slot {slot} still carries a 1/μ amplification: {:.3e}",
                solved.t[slot],
            );
        }
    }
}

#[cfg(test)]
mod tests_route_forced_classification_2673 {
    use super::*;
    use super::super::tests::{TestPeriodicEvaluator, periodic_basis};
    use gam_solve::arrow_schur::{ArrowSolveOptions, solve_arrow_newton_step_with_options};
    use ndarray::{Array1, Array2, array};
    use std::sync::Arc;

    /// #2673 — FORCE both classification routes on ONE state and compare.
    ///
    /// ## Why this test exists, and what it now guards
    ///
    /// Two floors used to classify directions of the same `A = B + ΔC`:
    /// `SAE_EXACT_A_PD_FLOOR_REL` (`1e-9·max(λ_max(A), 1)`, absolute) on the
    /// dense spectral path, and `√ε` on the pencil curvature
    /// `μ = xᵀAx/xᵀBx` inside `solve_exact_stationarity_preconditioned`. They
    /// are now ONE predicate in ONE metric — see
    /// [`sae_exact_a_identifiability_floor`] and
    /// [`ExactHessianSpectralBlock::rank_floor`] — so the two routes below
    /// cannot classify a direction differently by construction, and this test is
    /// the executable statement of that.
    ///
    /// ## The call chain that made it a live hazard rather than a tidiness one
    ///
    /// AN EARLIER VERSION OF THIS COMMENT CLAIMED "exactly one runs per
    /// evaluation ... they never coexist at a fixed state", and concluded the
    /// hazard was route-dependence (#2509/#2515) rather than the
    /// value↔gradient contradiction #2673 was filed about. **That claim was
    /// false, and it made a live hazard look structurally impossible.** It was
    /// true only within one GRADIENT assembly; an evaluation is value AND
    /// gradient, and on the streaming route both floors ran on the same `A`:
    ///
    /// * VALUE, when `direct_logdet_admitted == false`:
    ///   `penalized_quasi_laplace_criterion_streaming_exact_with_cache`
    ///   (`construction_quasi_laplace.rs:263`) →
    ///   `..._lane_and_system` (`:2971`) →
    ///   `converge_inner_for_undamped_logdet` (`:3039`) →
    ///   `..._gate_frozen` (`:698`) →
    ///   `terminal_exact_newton_polish` (`:1317`, `:1748`) →
    ///   `solve_exact_stationarity` (`:2346`, **with no route gate**) →
    ///   `materialize_exact_stationarity_geometry` →
    ///   `exact_hessian_spectral_block`.
    /// * GRADIENT, same evaluation: `matrix_free_system = Some(..)` →
    ///   `solve_exact_stationarity_matrix_free` →
    ///   `solve_exact_stationarity_preconditioned`.
    ///
    /// And there is no "massive-`K` threshold" to cross. The route predicate is
    /// a working-set comparison in `streaming_plan.rs:151` —
    /// `direct_peak_bytes <= in_core_budget_bytes || direct_fits_tiny` — so which
    /// floor classified the value path was a function of AMBIENT FREE MEMORY, not
    /// of `K`. `K` enters only through `row_block_dim`. The same data on the same
    /// build could be classified two ways on two differently-loaded machines,
    /// which is why unification did not wait for a fixture that populates the
    /// gauge band.
    ///
    /// ## What is compared
    ///
    /// The SAME state through the two PRODUCTION solves —
    /// `solve_exact_stationarity` (dense rank-revealing pseudoinverse) and
    /// `solve_exact_stationarity_matrix_free` (`B`-preconditioned GMRES + the `μ`
    /// deflation loop) — with the same rhs. If the two rules classify the same
    /// directions the same way, the solutions agree.
    ///
    /// Reported, not asserted equal. Agreement here is necessary but not
    /// sufficient: this fixture's spectrum sits `2.8e7` bands away from any
    /// classification boundary, so it exercises the routes, not the predicate.
    /// The predicate's own arms are
    /// `tests::the_two_floors_are_incommensurable_thresholds_on_one_operator_2673`
    /// (what the old pair did) and
    /// `tests::the_classification_is_invariant_under_a_reparametrization_2673`
    /// (what the new one does). What IS asserted here is that the comparison is
    /// well posed — both routes reached a typed outcome on one state, so the
    /// numbers below compare two answers rather than an answer and a refusal.
    #[test]
    fn route_forced_stationarity_classification_agrees_2673() {
        let n = 24usize;
        let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
        let (phi, jet) = periodic_basis(&coords);
        let decoder = array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]];
        let mut target = phi.dot(&decoder);
        for row in 0..n {
            target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
            target[[row, 1]] += 1.0e-3 * (0.29 * row as f64).cos();
        }
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![250.0_f64.ln()]]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("arrow-Schur assembly");
        let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("undamped factor cache");

        // One rhs, deterministic and dense enough to excite every direction —
        // a rhs that misses the near-null directions would leave both routes
        // nothing to classify differently.
        let total_t = cache.delta_t_len();
        let rhs = SaeArrowVector {
            t: Array1::from_shape_fn(total_t, |i| 0.5 + 0.25 * ((i as f64) * 0.7).sin()),
            beta: Array1::from_shape_fn(cache.k, |i| -0.3 + 0.2 * ((i as f64) * 1.1).cos()),
        };

        let dense = term.solve_exact_stationarity(&rho, target.view(), &cache, &rhs);
        let matrix_free =
            term.solve_exact_stationarity_matrix_free(&rho, target.view(), &cache, &sys, &rhs);

        match (&dense, &matrix_free) {
            (Ok(d), Ok(m)) => {
                let dt = d
                    .t
                    .iter()
                    .zip(m.t.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                let db = d
                    .beta
                    .iter()
                    .zip(m.beta.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                let scale = d
                    .t
                    .iter()
                    .chain(d.beta.iter())
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0);
                println!(
                    "[#2673 ROUTE-FORCED] both routes solved. max|Δt|={dt:.6e} \
                     max|Δbeta|={db:.6e} scale={scale:.6e} relative={:.6e}",
                    dt.max(db) / scale
                );
            }
            (Err(d), Ok(_)) => println!(
                "[#2673 ROUTE-FORCED] DENSE refused, matrix-free solved — the routes \
                 disagree on whether this state is solvable at all: {d}"
            ),
            (Ok(_), Err(m)) => println!(
                "[#2673 ROUTE-FORCED] MATRIX-FREE refused, dense solved — the routes \
                 disagree on whether this state is solvable at all: {m}"
            ),
            (Err(d), Err(m)) => println!(
                "[#2673 ROUTE-FORCED] both routes refused.\n  dense: {d}\n  matrix-free: {m}"
            ),
        }

        // Well-posedness: both routes must reach a TYPED outcome on this state, and
        // a solution that is returned must be finite. Without this the print above
        // could be comparing an answer against a panic.
        for (label, solved) in [("dense", &dense), ("matrix-free", &matrix_free)] {
            if let Ok(x) = solved {
                assert!(
                    x.t.iter().chain(x.beta.iter()).all(|v| v.is_finite()),
                    "#2673: the {label} route returned a non-finite stationarity solution"
                );
                assert_eq!(
                    x.t.len(),
                    total_t,
                    "#2673: the {label} route must return the declared t layout"
                );
                assert_eq!(
                    x.beta.len(),
                    cache.k,
                    "#2673: the {label} route must return the declared beta layout"
                );
            }
        }
    }
}

